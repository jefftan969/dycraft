"""Whole-body IK for a Unitree G1 over a whole clip, with cuRobo's MotionRetargeter.

Run it through the harness, from your shell:

    h.ik(script="references/ik/examples/g1_sequence.py", timeout_s=1800)

Per-frame world targets for the pelvis, both feet, and both wrists go in; joint
angles for every frame come out (`result["q"]`, one list per frame, in
`result["joint_names"]` order, radians and meters). The base floats: cuRobo's
G1 config chains six virtual joints (x, y, z, then three rotations) from
`base_link` to `pelvis`, so the pelvis goes where its target says. The first
frame is a 64-seed global search, then each frame is warm-started from the
previous one, so the result is continuous.

Measured on an L40S, 240 frames with self-collision: 17 s through
`h.ik` once the GPU kernels are compiled on that machine (58 s the first time),
target position error median 1.3 mm / p95 4.4 mm, joint steps p95 0.8 deg/frame.

Before you rely on it:
- **It solves your scene's G1, not a look-alike.** When `DYCRAFT_IK_ROBOTS`
  names a preloaded robot, `dycraft_ik.g1_robot` loads cuRobo's G1 set-up
  (spheres, floating base, tool frames) onto that robot's own URDF; joint names
  and units are the scene's, so `q` maps onto its joints by name. Without one it
  falls back to cuRobo's bundled G1 (`g1_29dof_rev_1_0`), which differs from the
  Dex1 G1 only at the wrist (yaw joint 5 mm closer). The gripper fingers past the
  wrist have no spheres of their own: check finger clearance in the scene.
- **Frames and units.** Targets and obstacles share one frame: use your scene's
  world (meters, z up, wxyz quaternions); the floating base goes wherever the
  targets put it. At zero pose cuRobo's G1 has its pelvis at the origin, facing +x,
  which is only where the made-up targets below start.
- **It does not balance the body.** High weights on the feet and pelvis keep them
  where you put them; whether the center of mass is over the support is still
  yours to plan in the targets and to check.
- **Keep the pelvis as heavy as the feet and hands, and smooth the result.** The
  solve is frame by frame with no smoothing. With the pelvis at a fifth of the
  feet's and wrists' weight, the pelvis wandered 5 cm from its target
  and the whole body shook (16 mm/frame^2 against a 2.5 target); at equal weight
  and a 1-frame Gaussian over `q` it came to 3, hands and feet within ~2 cm. The
  script smooths `q` (`SMOOTH_FRAMES`) and reports each link's acceleration next
  to its target's (`accel_p95_mm`): a solve well above its targets is shaking.
  `use_mpc=True` does not help here: it ignores weight 0, so the placeholder
  targets of untracked frames pull the body ~10 cm off.
- **All 14 tracked frames need a target.** The config tracks pelvis, torso,
  shoulders, elbows, wrists, hips, knees, and ankles. Give the ones you have no
  target for weight 0 (their target is then ignored), as below.
- **Obstacles are optional.** Call with `frame=` to export the project: then the
  static scene at that frame is loaded (`world_scene`, in world coordinates, the
  same frame as the targets), minus `EXCLUDE`. Exclude what the robot holds or is
  meant to touch, and solve a clip whose objects move in stages, one export each.

Replace `make_targets` with your own task-space tracks.
"""
import json
import math
import os
import time
from pathlib import Path

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, SequenceGoalToolPose
from curobo.types import JointState, ToolPoseCriteria
import dycraft_ik

# The scene's G1 when one is preloaded, else cuRobo's own.
ROBOTS = json.loads(os.environ.get("DYCRAFT_IK_ROBOTS") or "{}")
ROBOT = dycraft_ik.g1_robot(next(iter(ROBOTS.values()))["urdf"]) if ROBOTS else dycraft_ik.G1_CONFIG
TRACKED = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
           "left_wrist_yaw_link", "right_wrist_yaw_link"]
ROBOT_INSTANCE = "g1"          # the scene's own robot, never an obstacle
# MPC (smoothing over a short horizon) stays off: it does not honour weight-0 tool frames, so the
# placeholder targets of untracked frames pull the body ~10 cm off (measured: 111 mm median error
# against 1.3 mm here). Give every frame a real target before turning it on.
USE_MPC = False
SMOOTH_FRAMES = 1.0             # Gaussian sigma over q, in frames; 0 turns it off
EXCLUDE = []                   # instances the robot holds or touches on purpose


def world_scene(export_dir, exclude=()):
    """The exported static scene as a cuRobo mesh scene in world coordinates, or None."""
    meta = json.loads((Path(export_dir) / "ik_meta.json").read_text())
    drop = {ROBOT_INSTANCE, *exclude}
    mesh = {f"{o['instance']}__{o['part']}": {
                "file_path": str((Path(export_dir) / o["mesh"]).resolve()),
                "pose": list(o["pose"]["translation"]) + list(o["pose"]["quaternion_wxyz"])}
            for o in meta.get("obstacles", ()) if o["instance"] not in drop}
    return {"mesh": mesh} if mesh else None


def make_targets(kin, names, n):
    """Example targets: FK of a made-up motion (walk the base 1 m, bob, swing the
    arms), so the solve's error can be measured. Returns positions (n, len(TRACKED), 3)
    and wxyz quaternions (n, len(TRACKED), 4) on the GPU."""
    t = torch.linspace(0, n / 24.0, n, device="cuda")
    q = torch.zeros(n, len(names), device="cuda")
    q[:, names.index("base_j_x")] = 0.1 * t
    for joint, amp, phase in [("left_shoulder_pitch_joint", 0.6, 0), ("right_shoulder_pitch_joint", 0.6, math.pi),
                              ("left_elbow_joint", 0.4, 0), ("right_elbow_joint", 0.4, math.pi),
                              ("left_knee_joint", 0.3, 0), ("right_knee_joint", 0.3, 0),
                              ("left_hip_pitch_joint", -0.15, 0), ("right_hip_pitch_joint", -0.15, 0)]:
        q[:, names.index(joint)] = amp * (0.5 - 0.5 * torch.cos(t * 1.3 + phase))
    lo, hi = kin.get_joint_limits().position
    q = torch.max(torch.min(q, hi), lo)
    st = kin.compute_kinematics(JointState.from_position(q, joint_names=names))
    pos = torch.stack([st.tool_poses[l].position.reshape(n, 3) for l in TRACKED], 1)
    quat = torch.stack([st.tool_poses[l].quaternion.reshape(n, 4) for l in TRACKED], 1)
    return pos, quat


kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT))
names = list(kin.joint_names)
N = 240
pos, quat = make_targets(kin, names, N)

# Feet and pelvis heavy (they carry the body), wrists lighter on orientation.
weights = {"pelvis": (1.0, 0.1), "left_ankle_roll_link": (1.0, 0.1), "right_ankle_roll_link": (1.0, 0.1),
           "left_wrist_yaw_link": (1.0, 0.05), "right_wrist_yaw_link": (1.0, 0.05)}
criteria = {link: ToolPoseCriteria.track_position_and_orientation(
                xyz=[weights.get(link, (0.0, 0.0))[0]] * 3, rpy=[weights.get(link, (0.0, 0.0))[1]] * 3)
            for link in kin.tool_frames}
t0 = time.time()
exports = json.loads(os.environ.get("DYCRAFT_IK_EXPORTS") or "{}")
scene = world_scene(next(iter(exports.values())), EXCLUDE) if exports else None
cfg = MotionRetargeterCfg.create(robot=ROBOT, tool_pose_criteria=criteria, num_envs=1,
                                 self_collision_check=True, scene_model=scene, use_mpc=USE_MPC)
retargeter = MotionRetargeter(cfg)


def column(link, data):
    """A tracked link's targets, or any placeholder for an untracked (weight 0) one."""
    return data[:, TRACKED.index(link)] if link in TRACKED else data[:, 0]


# (frames, envs, links, goalset, 3|4), links in the config's own order
P = torch.stack([column(l, pos) for l in cfg.tool_frames], 1)
Q = torch.stack([column(l, quat) for l in cfg.tool_frames], 1)
seq = SequenceGoalToolPose(tool_frames=cfg.tool_frames, position=P[:, None, :, None, :].contiguous(),
                           quaternion=Q[:, None, :, None, :].contiguous())
res = retargeter.solve_sequence(seq)
q = res.joint_state.position.reshape(N, -1)


def smooth(q, sigma):
    """Gaussian over time per joint, ends held; base angles are continuous in a sequence, so this is safe."""
    if sigma <= 0:
        return q
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=q.device) / sigma) ** 2)
    k = (k / k.sum())[None, None]
    padded = torch.cat([q[:1].repeat(r, 1), q, q[-1:].repeat(r, 1)])
    return torch.nn.functional.conv1d(padded.T[:, None], k)[:, 0].T.contiguous()


q = smooth(q, SMOOTH_FRAMES)

# How well each tracked target was met.
st = kin.compute_kinematics(JointState.from_position(q, joint_names=names))
err = torch.stack([(st.tool_poses[l].position.reshape(N, 3) - pos[:, i]).norm(dim=-1)
                   for i, l in enumerate(TRACKED)], 1)
result["joint_names"] = names
result["q"] = q.tolist()
result["seconds"] = round(time.time() - t0, 1)
result["obstacles"] = len(scene["mesh"]) if scene else 0
result["robot"] = next(iter(ROBOTS.values()))["urdf"] if ROBOTS else ROBOT
result["position_error_mm"] = {"median": round(float(err.median()) * 1000, 2),
                               "p95": round(float(err.quantile(0.95)) * 1000, 2)}
# Smoothness, which tracking error does not show: per-frame acceleration of each tracked link (mm/frame^2)
# against its target's. A solve well above its targets is shaking.
def accel(x):
    return (x[2:] - 2 * x[1:-1] + x[:-2]).norm(dim=-1) * 1000
result["accel_p95_mm"] = {l: {"solved": round(float(accel(st.tool_poses[l].position.reshape(N, 3)).quantile(0.95)), 2),
                              "target": round(float(accel(pos[:, i]).quantile(0.95)), 2)} for i, l in enumerate(TRACKED)}
print(json.dumps({k: v for k, v in result.items() if k != "q"}))
