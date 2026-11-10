"""Plan one arm of a Unitree G1 around obstacles with cuRobo's MotionPlanner, body held still.

Run it through the harness, from your shell:

    h.ik(script="references/ik/examples/g1_arm_plan.py", timeout_s=1800)

For the free-hand spans of a performance -- reaching toward a grasp, drawing the hand
back, swinging it clear while turning -- where the body is (nearly) still and the hand
carries nothing. `MotionRetargeter` (g1_sequence.py) tracks targets frame by frame and
does not route around what lies between them; a plan does, and comes back smooth.

How: every joint but one arm's is locked at the pose the span starts from (`START`, by
name; in a real performance, the base/legs/waist your whole-body solve gives at that
frame), the robot tracks only that wrist, and `plan_pose` finds a collision-free path from
the arm's current configuration to `GOAL`, sampled at 24 fps (`interpolation_dt`).

Before you rely on it:
- **One frame of reference.** World coordinates, as in g1_sequence.py: the floating base's
  six joints are locked at the span's base pose, so goals and obstacles are world poses.
- **Obstacles** come from a `frame=` export (`world_scene`, minus the robot and `EXCLUDE`),
  else the example puts a box on the hand's straight path to show the detour. Exclude the
  object about to be grasped, or the goal is unreachable.
- **A plan picks its own duration** (time-optimal). Retime it to the span you have.
- **The locked body must be collision-free**, with itself and the scene: a plan of `None`
  usually means the start (or a locked limb) already collides, not that the goal is hard.
  `self_collision_check=False` does not help: it relaxes the optimizer, not the final check.
- **Grasps** with approach and lift: `plan_grasp` (cuRobo's own
  examples/getting_started/motion_planning.py, `grasp_planning_example`) on the same planner.
"""
import copy
import json
import os
import time
from pathlib import Path

import torch
import yaml
import curobo
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
import dycraft_ik

# The scene's G1 when one is preloaded, else cuRobo's own (see g1_sequence.py).
ROBOTS = json.loads(os.environ.get("DYCRAFT_IK_ROBOTS") or "{}")
ROBOT = dycraft_ik.g1_robot(next(iter(ROBOTS.values()))["urdf"]) if ROBOTS else dycraft_ik.G1_CONFIG
ARM = "left"
TOOL = f"{ARM}_wrist_yaw_link"
ROBOT_INSTANCE = "g1"
EXCLUDE = []
# Joint name -> value (radians, meters) where the span starts; missing joints are 0. The made-up
# start holds both arms a little out: at zero pose cuRobo's G1 has its right upper arm 7 mm inside
# its torso, and a locked joint in collision makes every plan infeasible.
START = {"left_shoulder_pitch_joint": -0.3, "left_shoulder_roll_joint": 0.35, "left_elbow_joint": 0.6,
         "right_shoulder_roll_joint": -0.2}
GOAL = None           # (xyz, wxyz) world pose for the wrist; None = where REACH puts it (a pose known reachable)
REACH = {"left_shoulder_pitch_joint": -1.3, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.2}


def world_scene(export_dir, exclude=()):
    """The exported static scene as a cuRobo mesh scene in world coordinates, or None."""
    meta = json.loads((Path(export_dir) / "ik_meta.json").read_text())
    drop = {ROBOT_INSTANCE, *exclude}
    mesh = {f"{o['instance']}__{o['part']}": {
                "file_path": str((Path(export_dir) / o["mesh"]).resolve()),
                "pose": list(o["pose"]["translation"]) + list(o["pose"]["quaternion_wxyz"])}
            for o in meta.get("obstacles", ()) if o["instance"] not in drop}
    return {"mesh": mesh} if mesh else None


t0 = time.time()
# The full robot, for the start pose's forward kinematics.
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT))
names = list(kin.joint_names)
q0 = torch.tensor([[START.get(n, 0.0) for n in names]], device="cuda")
st = kin.compute_kinematics(JointState.from_position(q0, joint_names=names))
p0 = st.tool_poses[TOOL].position.reshape(3).clone()   # clone: the next FK call reuses the buffer
if GOAL:
    gp, gq = torch.tensor(GOAL[0], device="cuda"), torch.tensor(GOAL[1], device="cuda")
else:                                  # the wrist where REACH (over START) puts it
    reach = kin.compute_kinematics(JointState.from_position(
        torch.tensor([[{**START, **REACH}.get(n, 0.0) for n in names]], device="cuda"), joint_names=names))
    gp, gq = reach.tool_poses[TOOL].position.reshape(3).clone(), reach.tool_poses[TOOL].quaternion.reshape(4).clone()

# The planning robot: same model, one wrist tracked, everything but that arm locked.
if isinstance(ROBOT, dict):
    cfg = copy.deepcopy(ROBOT)
else:
    content = os.path.join(os.path.dirname(curobo.__file__), "content")
    cfg = yaml.safe_load(open(os.path.join(content, "configs", "robot", ROBOT)))
    k = cfg["robot_cfg"]["kinematics"] if "robot_cfg" in cfg else cfg["kinematics"]
    for key in ("urdf_path", "asset_root_path"):
        if k.get(key) and not os.path.isabs(k[key]):
            k[key] = os.path.join(content, "assets", k[key])
k = cfg["robot_cfg"]["kinematics"] if "robot_cfg" in cfg else cfg["kinematics"]
k["tool_frames"] = [TOOL]
k["lock_joints"] = {n: START.get(n, 0.0) for n in names if not n.startswith(ARM + "_shoulder")
                    and not n.startswith(ARM + "_elbow") and not n.startswith(ARM + "_wrist")}
exports = json.loads(os.environ.get("DYCRAFT_IK_EXPORTS") or "{}")
scene = world_scene(next(iter(exports.values())), EXCLUDE) if exports else None
if scene is None:                     # demo: a 10 cm box across the hand's straight path
    mid = ((p0 + gp) / 2).tolist()
    scene = {"cuboid": {"demo_box": {"dims": [0.10, 0.10, 0.10], "pose": mid + [1.0, 0.0, 0.0, 0.0]}}}
planner = MotionPlanner(MotionPlannerCfg.create(robot={"robot_cfg": cfg.get("robot_cfg", cfg)},
                                                 scene_model=scene, interpolation_dt=1 / 24))
planner.warmup()
arm = list(planner.joint_names)
start = JointState.from_position(torch.tensor([[START.get(n, 0.0) for n in arm]], device="cuda"), joint_names=arm)
goal = GoalToolPose(tool_frames=planner.tool_frames, position=gp.reshape(1, 1, 1, 1, 3),
                    quaternion=gq.reshape(1, 1, 1, 1, 4))
# Retried: on the scene's G1 the first plan after building failed every time (4/4) and
# the next one succeeded. A plan of None is cheap to repeat.
for _ in range(3):
    res = planner.plan_pose(goal, start, max_attempts=5)
    ok = bool(res is not None and res.success.any())
    if ok:
        break
result["ok"] = ok
result["joint_names"] = arm
result["seconds"] = round(time.time() - t0, 1)
result["obstacles"] = sorted({**(scene or {}).get("mesh", {}), **(scene or {}).get("cuboid", {})})
if ok:
    plan = res.get_interpolated_plan()                  # may carry the locked joints too
    cols = list(plan.joint_names)
    traj = plan.position.reshape(-1, len(cols))
    result["joint_names"] = cols
    result["q"] = traj.tolist()                         # one row per output frame (24 fps)
    # How far the hand bowed off the straight line: the detour the obstacle forced.
    full = torch.tensor([[START.get(n, 0.0) for n in names]], device="cuda").repeat(len(traj), 1)
    for j, n in enumerate(cols):
        full[:, names.index(n)] = traj[:, j]
    hand = kin.compute_kinematics(JointState.from_position(full, joint_names=names)).tool_poses[TOOL]
    path = hand.position.reshape(-1, 3)
    line = gp - p0
    off = torch.linalg.cross(path - p0, line.expand_as(path)).norm(dim=-1) / line.norm()
    result["frames"] = len(traj)
    result["hand_max_off_line_m"] = round(float(off.max()), 3)
    result["goal_error_mm"] = round(float((path[-1] - gp).norm()) * 1000, 1)
print(json.dumps({k_: v for k_, v in result.items() if k_ != "q"}))
