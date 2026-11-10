# cuRobo — a collision-free motion planner, and how to use it from `h.ik`

cuRobo is a GPU motion planner: it finds joint trajectories that keep every link
clear of the robot's own body and of the scene, smooth, and sampled at your frame
rate, and it solves hundreds of targets in one batched call. For a robot that has
to move through a scene, that is the job; per-frame IK written by hand checks
nothing between two poses and has no obstacles to avoid.

**Start from the task:**

| the motion | use | runs as it stands |
|---|---|---|
| a humanoid walking, reaching, carrying -- whole body, whole clip | `MotionRetargeter.solve_sequence` (floating base, self-collision, scene obstacles) | `examples/g1_sequence.py` |
| one limb moving past obstacles while the body holds | `MotionPlanner.plan_pose` on that limb, the rest locked | `examples/g1_arm_plan.py` |
| reach, close on, and lift an object | `MotionPlanner.plan_grasp` | cuRobo's `examples/getting_started/motion_planning.py` |
| is this pose reachable; which of these goals | `InverseKinematics.solve_pose`, batched | §IK |

Both G1 examples solve the scene's own G1 (`dycraft_ik.g1_robot`). Measured on an
L40S: the whole-body clip 240 frames in 17 s at 1.3 mm median error; the arm
plan 41 frames round a box in 14 s with 0 mm goal error. What cuRobo leaves to
you: balance, grasp timing, and the targets.

Full V2 docs are mirrored offline in `curobo_docs/` beside this file (a run has no
network); `curobo_docs/README.md` maps question to file. The rest of
this file is what those docs do not say: what differs on this checkout, what is
measured rather than documented, and how a DyCraft scene attaches.

## Which of the three, and why the order below is not it

**A span of motion is the planner's question.** Three solvers live here and they
answer three different things; the sections are ordered by how much machinery
each needs, which is the opposite of how often you want it.

| you are asking | reach for | where |
|---|---|---|
| how does this linkage *get* from A to B, past what is in the way | `MotionPlanner` — `plan_pose`, `plan_cspace`, `plan_grasp` | §Planning |
| is this tip pose reachable at all | `InverseKinematics.solve_pose` | §IK |
| where does this configuration put the links | `Kinematics` | §FK |

Reading §IK first is how this gets decided wrong. IK answers one pose at a time,
so a loop over frames *looks* like it authors a span while checking nothing
between two poses — and the loop is easy to keep extending, because each
individual answer is correct. Measured on a 60 s clip built that way: 46 solver
scripts, 34 `h.ik` calls, previous-frame seeding out of a 384-seed pool, final
tool pose accurate to **3e-8 m** — and `NoPenetration` still reporting 62 mm of
link-on-link self-overlap and 185 mm of forearm through a wall panel, none of it
retimeable (§Planning). Endpoint accuracy is not evidence the path is clear.

Use IK to *choose* a target and a plan to *get there*.

## A humanoid's whole body: `MotionRetargeter`

**For a walking, reaching, carrying humanoid, this is the tool, not a per-limb
`InverseKinematics`.** cuRobo ships a Unitree G1 set up for it
(`curobo/examples/getting_started/humanoid_retargeting.py`, config
`unitree_g1_29dof_retarget.yml`): six virtual joints chained from `base_link` to
`pelvis` make the base float (`extra_links` in the config), fourteen links are
tool frames (pelvis, torso, shoulders, elbows, wrists, hips, knees, ankles), and
`MotionRetargeter.solve_sequence` solves every frame of a clip in one call -- a
64-seed global IK on frame 0, then each frame warm-started from the last, with
joint limits and optional self-collision (`use_mpc=True` adds jerk smoothing at
2-4x the cost). `examples/g1_sequence.py` beside this file is a working start: it
runs as it stands through `h.ik(script=...)` and returns joint angles per frame.

Measured on an L40S: 240 frames, pelvis + feet + wrists tracked, self-
collision on -- 17 s once the kernels are compiled on the machine (58 s the first
time), position error median 1.3 mm / p95 4.4 mm, joint steps p95 0.8 deg/frame.
Recorded runs that wrote their own numpy whole-body IK instead spent 45 minutes of
one CPU core per pass over a similar clip.

What it does not do, so plan it in the targets:

- **Balance.** Weights on the feet and pelvis keep them where you put them; the
  center of mass over the support is yours to plan and check.
- **Obstacles you move or hold.** `scene_model` makes the whole-body solve avoid
  static obstacles (the example loads them from a `frame=` export, in world
  coordinates like the targets; tested: a box on the pelvis path pushes the body
  round it). The scene is fixed for one `solve_sequence`, so exclude what the robot
  holds or is meant to touch, and solve a clip whose objects move in stages, one
  export each.
- **The robot is the scene's, exactly.** `dycraft_ik.g1_robot(urdf)` (staged beside
  your script in `ik/`) returns this config on the URDF `DYCRAFT_IK_ROBOTS` names,
  as the `robot=` dict for `MotionRetargeterCfg` / `MotionPlannerCfg` /
  `KinematicsCfg.from_robot_yaml_file`. It works because every link the config
  names (spheres, self-collision table, tool frames) has the same name on a
  `g1_29dof_*` URDF; it only adds the `base_link` the floating base hangs `pelvis`
  from and three frames the scene's URDF lacks. Compared on the Dex1 G1
  (`g1_29dof_mode_15_with_dex1_1`) against the bundled `g1_29dof_rev_1_0`: same
  29 joints, axes, and limits; the one difference is the wrist yaw joint, 51 mm
  out on the Dex1's against 46 mm, so a solve on the bundled model leaves each
  wrist 5 mm short; the Dex1 adds two prismatic fingers per hand past the wrist.
  On the scene's model a retarget meets its wrist targets to 0.7 mm. Joint names
  and radians are the scene's, so `q` maps onto its joints by name. The fingers
  have no spheres of their own (the hand's sit in the gripper base): check finger
  clearance in the scene.
- **Smoothness.** The default solve is frame by frame with no smoothing, and a
  loose pelvis shakes the whole body: at a fifth of the feet's and wrists' weight
  it wandered 5 cm and accelerated 6x its target. Keep the pelvis
  as heavy as the feet and hands, Gaussian-smooth `q` over ~1 frame, and compare
  each link's per-frame acceleration with its target's, as `g1_sequence.py`
  does; tracking error alone does not show shaking. `use_mpc=True` is not the
  fix: it ignores weight-0 frames, so placeholder targets pull the body ~10 cm
  off, and with real targets on all fourteen it was no smoother.
- **Missing targets.** The config tracks all fourteen frames and wants a target
  for each; give the ones you have none for weight 0 (`ToolPoseCriteria` with
  zero `xyz` and `rpy`), and their targets are ignored. Naming only a subset
  raises `Ordered link names [...] not a subset of [...]`.

### A free hand around obstacles: `MotionPlanner` on one arm

`MotionRetargeter` tracks targets frame by frame and does not route around what
lies between them. For a span where the body is (nearly) still and a hand moves
free -- reaching toward a grasp, drawing back, swinging clear -- lock every joint
but one arm's at the span's start (`lock_joints` in a copy of the config, with
that wrist as the only tool frame) and `plan_pose` the wrist there.
`examples/g1_arm_plan.py` does this and runs as it stands: measured on an L40S,
41 frames at 24 fps, 0 mm goal error, the hand bowing 12 cm round a
10 cm box versus 6 cm without it, 14 s per call. Traps it hit:

- **A locked limb in collision makes every plan `None`.** At zero pose the
  bundled G1's right upper arm is 7 mm inside its torso, so lock the body at a
  collision-free pose. `self_collision_check=False` does not help: it relaxes the
  optimizer, not the final feasibility check.
- **A goal off the reachable set also returns `None`**, not a near miss; derive
  goals from a pose you know (FK) or check with `InverseKinematics` first.
- **`compute_kinematics` reuses its output buffers**: `.clone()` a pose you keep
  across calls.
- **The plan carries the locked joints too**: read it by `plan.joint_names`.
- For grasps with approach and lift, `plan_grasp` on the same planner (cuRobo's
  `examples/getting_started/motion_planning.py`, `grasp_planning_example`).

## Which version this is

Post-0.8.0, unreleased, commit-pinned in `setup_ik.sh`. **0.8.0 is the break**:
its changelog says *"Major refactor to a flat, inheritance-free architecture
(curobov2)"*, *"Reorganise implementation under `curobo._src/`"*, *"New
`MotionPlanner` / `MotionPlannerCfg`"*. Anything you remember from 0.7.x is the
previous generation. `curobo.__version__` reports `0.0.0` — a shallow clone has
no tags for setuptools-scm — so the commit is the only identifier.

## Imports

```python
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner     import MotionPlanner, MotionPlannerCfg, GraspPlanResult
from curobo.kinematics         import Kinematics, KinematicsCfg, KinematicsState
from curobo.robot_builder      import RobotBuilder
from curobo.sphere_fit         import fit_spheres_to_mesh, estimate_sphere_count, SphereFitType
from curobo.types              import GoalToolPose, Pose, JointState, ToolPoseCriteria
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
```

Also present: `batch_motion_planner.BatchMotionPlanner`,
`model_predictive_control.ModelPredictiveControl`, `motion_retargeter`,
`trajectory_optimizer`, `optim`, `rollout`.

`InverseKinematics` is an alias for `IKSolver`; there is **no**
`load_from_config`.

## The `dycraft_ik` seam — frames and units

Two conversions sit between a DyCraft scene and this solver, and both are silent
when skipped. Neither is a cuRobo concern, which is why they are here.

**Targets are in the arm's base frame, and `world_to_base` is a full SE3
inverse.** Subtracting the base translation is the plausible shortcut and it is
wrong the moment the arm is placed with any yaw — measured at **622 mm** of error
on one cell. `base_to_world` is the return trip.

```python
from dycraft_ik import world_to_base, base_to_world, to_authoring_joint_track
p_base, q_base = world_to_base("panda_arm", p_world, q_world)
```

**The solver and `key_joint` speak radians; the source speaks degrees.**
Key a solved `q` with `key_joint(name, value, frame)` as it is, in the order
`joint_names` gives: radians for a revolute joint, meters for a prismatic one.
`to_authoring_joint_track` and `q_to_surface` convert to the source's units,
degrees for a revolute joint, as `limits`, `default`, and `initial_joints` in
`design/` take them. A degree value keyed with `key_joint` is out by 57x, and
inside the limits it is silently a different motion.

## IK

```python
ik = InverseKinematics(InverseKinematicsCfg.create(
    robot=robot_config("panda_arm", tool_links=["hand"]),   # dycraft_ik
    scene_model=scene_config("panda_arm"),                  # dycraft_ik; None = self-collision only
    num_seeds=48, self_collision_check=True,
    position_tolerance=2e-3, orientation_tolerance=2e-2))

goal = GoalToolPose.from_poses({ik.tool_frames[0]: Pose(
    position=torch.tensor([xyz],  dtype=torch.float32, device="cuda"),
    quaternion=torch.tensor([wxyz], dtype=torch.float32, device="cuda"))},
    num_goalset=1)

res = ik.solve_pose(goal,
                    current_state=JointState.from_position(seed, joint_names=NAMES),
                    return_seeds=48)          # all seeds back, not just the best
q  = res.js_solution.position.reshape(-1, NDOF)
pe = res.position_error.reshape(-1)
```

`Pose` takes `(B, 3)` / `(B, 4)` wxyz, `cuda`, `float32`.
`InverseKinematicsResult`: `success`, `position_error`, `rotation_error`,
`js_solution`, `feasible`, `solve_time`, `position_tolerance`,
`orientation_tolerance`, `metrics`, `num_seeds`, `debug_info`.

`InverseKinematicsCfg.create` in full, with defaults:

```
robot=<required>                          num_seeds=32
scene_model=None                          seed_solver_num_seeds=32
self_collision_check=True                 position_tolerance=0.005
collision_cache=None                      orientation_tolerance=0.05
load_collision_spheres=True               success_requires_convergence=True
use_cuda_graph=True                       optimization_dt=None
random_seed=123                           override_optimizer_num_iters={'particle': None, 'lbfgs': None}
optimizer_collision_activation_distance=0.01
                                          override_iters_for_multi_link_ik=None
max_batch_size=1                          velocity_regularization_weight=None
max_goalset=1                             acceleration_regularization_weight=None
multi_env=False                           seed_{position,orientation}_weight=1.0
store_debug=False                         seed_{velocity,acceleration}_weight=0.0
optimizer_configs=['ik/particle_ik.yml', 'ik/lbfgs_ik.yml']
metrics_rollout='metrics_base.yml'        transition_model='ik/transition_ik.yml'
device_cfg=DeviceCfg(...)                 {transition_model,cost_manager}_config_instance_type=<class …>
```

## FK

`Kinematics` answers where the links are for a configuration you already have —
the cheap half of the pair, and what you score a candidate with.

```python
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

# `from_robot_yaml_file` takes a DICT despite the name, which is what
# `robot_config` returns. There is no `KinematicsCfg.create`.
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(
    robot_config("panda_arm", tool_links=["panda_hand"])))

js = JointState.from_position(
    torch.tensor([q], dtype=torch.float32, device="cuda"),
    joint_names=list(kin.joint_names))
st = kin.compute_kinematics(js)
p = st.tool_poses[kin.tool_frames[0]]
pos, quat = p.position, p.quaternion            # base frame; base_to_world for world
```

**`KinematicsCfg` is configured differently from the solvers, and the difference
costs a call.** `InverseKinematicsCfg` and `MotionPlannerCfg` are built with
`.create(...)`; generalizing that to `KinematicsCfg` raises
`AttributeError: type object 'KinematicsCfg' has no attribute 'create'` — measured,
and after 43 s of sphere fitting had already been paid. Its constructors are
`from_robot_yaml_file`, `from_config_file`, `from_config`, `from_basic_urdf`,
`from_content_path`, and `from_data_dict`.

`from_robot_yaml_file(file_path, tool_frames=None, device_cfg=..., urdf_path=None,
**kwargs)` — `file_path` is `Union[str, Dict]`, so the `robot_config` dict goes in
directly and `tool_links` set there already names the tool frames.

`MotionPlanner` also carries `compute_kinematics(js)`, so a planner you have
already built answers FK without a second object.

## Planning

**Plan a trajectory rather than solve a pose per frame.** `InverseKinematics`
answers one question — which configuration puts the tool *here* — and it answers
it independently every time you ask. A planner answers a different question:
give me a configuration *sequence* from A to B. Two things follow, and both are
the reason to reach for `MotionPlanner` before writing a per-frame IK loop.

**It avoids obstacles along the whole path, not just at the endpoints.** IK's
`self_collision_check` and `scene_model` clear the configuration it returns; they
say nothing about the configurations between two returned poses. A plan is
collision-checked over its own interpolated trajectory, so the elbow that swings
through a carcass on the way to a reachable grasp is caught by the planner and
invisible to per-pose IK. That is the failure `NoPenetration` then reports as an
interval you cannot retime your way out of.

**Its output is smooth; a stack of IK solutions is not.** A redundant arm has a
null space, so two neighboring tool poses have many valid configurations and IK
picks from its seed pool each time. Re-solved per frame, the joint that could sit
anywhere wanders between branches: measured on a real 28 s clip authored this way,
the six arm joints moved at most **0.33 deg per frame** — small enough that no
verifier objected — while **22–34 % of consecutive steps reversed direction**. The
prismatic joint in the same clip, authored in closed form, reversed on 0.9 %. That
is a shimmer a viewer sees and no geometric check measures. A planned trajectory
is optimized as one object and does not have it.

So: `plan_pose` / `plan_cspace` for a span, `plan_grasp` for approach → grasp →
lift, and `solve_pose` for the question "is this pose reachable at all". Use IK to
*choose* a target, and a plan to *get there*.

### Three things measured on this checkout

**A mounted robot needs its base link disabled, or every plan returns `None`.**
An arm bolted to a worktop has base-link spheres inside the worktop mesh at every
joint value, so the planner's internal IK clears no seed and `_plan_pose_single`
falls through all `max_attempts` (`motion_planner.py:255`, `if success_count == 0:
continue`). `plan_pose` then returns **`None`** in 0.4 s with no message, which
reads like "no planner in this build". Measured on a bench-mounted Panda: 0 of 4
seeds by default, 0 with `optimizer_collision_activation_distance=0.002`, 0 with
tolerances tightened, 0 with `self_collision_check=False` — and 4 of 4 after one
line:

```python
mp.disable_link_collision(["panda_link0"])     # takes List[str]; enable_link_collision undoes it
```

**`success` already means every interpolated frame is collision-free.** It is
`converged AND feasible AND interpolated_feasible`, and the last term is computed
on the *interpolated* states, not the optimization knots
(`solver/solver_trajopt_result.py:206`). So with `interpolation_dt=1/fps` the
samples the planner checked are your animation frames: a plan whose `success` is
true needs no per-frame collision filter of your own. `success=False` from a plan
that took seconds is the check working, not the call failing.

**Planning the span is ~9x cheaper than solving its frames, and picks its own
duration.** Same 156-frame / 6.5 s span, same scene, one RTX 3090: a per-frame
`solve_pose` loop at 192 seeds cost **83.0 s** of solving, one `plan_pose` cost
**9.3 s**. Both paid the same ~28 s solver build. But the plan came back with 41
interpolated frames, not 156 — a plan is time-optimal, so it answers "get there",
not "get there in 6.5 s". A span whose timing the brief cares about needs the
duration imposed afterwards, or a `plan_cspace` per beat.

### The planner surface

Copy from here rather than from memory — the whole reacher API was renamed in
this generation, so a signature recalled from anywhere else fails as an
`AttributeError` that reads like "no planner in this build".

```python
mp = MotionPlanner(MotionPlannerCfg.create(robot=..., scene_model=...))   # same two dicts

mp.plan_pose(goal_tool_poses, current_state, max_attempts=5, enable_graph_attempt=1)
mp.plan_cspace(goal_state, current_state, max_attempts=5, enable_graph_attempt=1)
mp.warmup(); mp.update_world(scene_cfg); mp.compute_kinematics(js)
```

Both return `TrajOptSolverResult`: `success`, `js_solution`,
`interpolated_trajectory`, `interpolated_last_tstep`, `maximum_trajectory_dt`,
`minimum_trajectory_dt`, `position_error`, `rotation_error`, `goalset_index`.
`plan_grasp` returns `GraspPlanResult` instead — see its own section below.

Four things that are not where a reader tends to look for them:

- **A goalset is not a separate call.** Many candidate poses go into
  `GoalToolPose` and `plan_pose` picks among them; the chosen one comes back as
  `goalset_index`.
- **Per-plan arguments are on the method**, not in a config object:
  `max_attempts`, `enable_graph_attempt`.
- **Many problems at once** is `curobo.batch_motion_planner.BatchMotionPlanner`.
- **Attaching an obstacle to a robot link** is the planner's
  `attachment_manager` property.

`MotionPlannerCfg.create` in full, with defaults:

```
robot=<required>                          interpolation_dt=0.025
scene_model=None                          interpolation_buffer_size=1000
self_collision_check=True                 num_ik_seeds=None
collision_cache=None                      num_trajopt_seeds=None
use_cuda_graph=True                       position_tolerance=0.005
random_seed=123                           orientation_tolerance=0.05
optimizer_collision_activation_distance=0.01
max_batch_size=1     max_goalset=1        multi_env=False     store_debug=False
ik_optimizer_configs=['ik/lbfgs_ik.yml']
ik_transition_model='ik/transition_ik.yml'
trajopt_optimizer_configs=['trajopt/lbfgs_bspline_trajopt.yml']
trajopt_transition_model='trajopt/transition_bspline_trajopt.yml'
graph_planner_config='graph_planner/exact_graph_planner.yml'
graph_planner_rollout='metrics_base.yml'
graph_planner_transition_model='graph_planner/transition_graph_planner.yml'
metrics_rollout='metrics_base.yml'        device_cfg=DeviceCfg(...)
{transition_model,cost_manager}_config_instance_type=<class …>
```

**`interpolation_dt` is how a plan lands on your frame grid.** Set it to your
animation's frame period and `interpolation_trajectory` comes back uniform at
exactly that, one sample per frame to key with `key_joint`. Resampling it yourself
is where a smooth plan would otherwise get its smoothness sampled away.
It is a plain argument: `MotionPlannerCfg.create(..., interpolation_dt=1/24)`
reads back as `0.041667` on `mp.trajopt_solver.config` — measured. The default
0.025 s is finer than a 24 fps frame either way, so leaving it alone means you
resample; setting it means you do not.

**Nothing downstream blocks on joint velocity or smoothness**: the verifiers'
`MotionContinuity` check only advises. That is not permission to ignore it; it is
why a jittering track can still commit, noticed only through that advisory, a
`review()`, or a human.

`TrajOptSolverCfg` additionally carries `minimum_trajectory_dt`,
`maximum_trajectory_dt`, and `interpolation_type`.

### Straight-line constraints

```python
from curobo.types import ToolPoseCriteria

ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True)
ToolPoseCriteria.track_position(xyz=[1., 1., 1.])
ToolPoseCriteria.track_orientation(rpy=[1e-3, 1e-3, 1e-3])
ToolPoseCriteria.track_position_and_orientation(xyz=..., rpy=..., non_terminal_scale=0.1)
ToolPoseCriteria.disabled()
ToolPoseCriteria()                                   # unconstrained default

mp.update_tool_pose_criteria({link: criteria})       # planner STATE, sticky
res = mp.plan_pose(...)
mp.update_tool_pose_criteria({link: ToolPoseCriteria()})   # restore
```


### `plan_grasp` — approach → grasp → lift in one call

```python
positions   = torch.zeros(1, 1, 1, n_grasps, 3, device="cuda", dtype=torch.float32)
quaternions = torch.zeros(1, 1, 1, n_grasps, 4, device="cuda", dtype=torch.float32)
grasp_poses = GoalToolPose(tool_frames=mp.tool_frames,
                           position=positions, quaternion=quaternions)

res = mp.plan_grasp(grasp_poses=grasp_poses, current_state=q_start,
                    grasp_approach_axis="z", grasp_approach_offset=0.1,
                    grasp_lift_axis="z",     grasp_lift_offset=0.1,
                    plan_approach_to_grasp=True, plan_grasp_to_lift=True,
                    grasp_lift_in_tool_frame=True,
                    disable_collision_links=["finger_left", "finger_right"])
```

Internally: `plan_pose` over the goalset → `linear_motion` + `plan_pose` to the
grasp → `linear_motion` + `plan_pose` to the lift, with
`disable_collision_links` off across the middle two. Goalset tensors are
**5-D**, not `(n, 3)`.

`GraspPlanResult`: `{approach,grasp,lift}_{success, trajectory,
trajectory_dt, interpolated_trajectory, interpolated_last_tstep}`, plus
`success`, `status`, `planning_time`, `goalset_index` (which candidate it took).

## Collision — one setup, both solvers

**`scene_model` is not the planner's.** `InverseKinematicsCfg.create` and
`MotionPlannerCfg.create` take the same collision arguments — `scene_model`,
`self_collision_check` (default `True`), `collision_cache`,
`optimizer_collision_activation_distance` (default `0.01`) — and the same two
dicts from `dycraft_ik` feed both. What differs is only *when* the check happens:
IK clears the configuration it hands back, a plan is checked over its whole
interpolated trajectory (§Planning).

Two models sit side by side and it is worth knowing which one refused you:

| against | model | configured by |
|---|---|---|
| the robot's own links | collision spheres | § Robot model, § Why the default is 12 |
| everything else in the room | a mesh world | `scene_config(...)` below |

### The scene model — `scene_config(instance, *, exclude=())`

```python
S = scene_config("panda_arm", exclude=("tool_chest", "spanner"))
ik = InverseKinematics(InverseKinematicsCfg.create(robot=R, scene_model=S))
mp = MotionPlanner(MotionPlannerCfg.create(robot=R, scene_model=S))
```

- **Meshes, not bounding boxes.** A table's AABB is a solid block that fills the
  legroom, so an AABB world would make reaching *under* a tabletop unsolvable for
  a reason that has nothing to do with the arm. Poses are in the solving
  instance's base frame; `scene_config` converts them for you.
- **The solving instance is always excluded for you.** Its links are the
  self-collision model's job, and feeding them in twice makes every pose a
  collision.
- **`exclude` is for what the arm is deliberately about to touch.** This is the
  one that costs a run if you miss it: a body left in the world is an obstacle,
  so the solver will not bring the tool to it, and the failure looks like "that
  grasp is unreachable". Exclude the thing you are grasping — and a surface you
  intend to *press on*, for the same reason. Measured on two real runs, both
  agents arrived here by trial: one went `()` → `("tool_chest", "spanner")` →
  `+ "workbench"` before its poses solved.
- **`scene_model=None` leaves self-collision only.** Useful to tell "the arm
  cannot fold that way" apart from "something is in the way".
- **Obstacles dominate the cost** — one mesh takes a solve from 0.080 s to
  0.972 s at density 12 (§Costs). Pass only what this motion can reach.

`optimizer_collision_activation_distance` (0.01 m) is where the optimizer starts
pushing *away* from a surface, not a guaranteed clearance. After you author a
returned configuration, check it with DyCraft's own tools: `check_motion` over
the frames it affects reports penetration on the evaluated meshes, and `probe`
can measure a specific clearance; the sphere model is a solver constraint,
never a verdict — see
`README.md` beside this file, § "The collision model is a constraint, not a verdict".

**The world changed mid-motion?** Both `InverseKinematics` and `MotionPlanner`
carry `update_world(scene_cfg)`. Rebuild the dict — a body now in the gripper
belongs in `exclude`, a drawer that opened moved — and push it, rather than
constructing a second solver and paying the build again (§Costs).

**Upstream, in the offline mirror:** `curobo_docs/api_scene.md`
(obstacles, `SceneCfg`, mesh worlds — 89 KB, the biggest page there),
`ref_self_collision.md`, `ref_sphere_fitting.md`. Those are local copies of
<https://nvlabs.github.io/curobo/latest/api/curobo.scene.html> and its
neighbors, mirrored because a run has no network.

## Robot model

```python
b = RobotBuilder(urdf_abs_path, asset_root, tool_frames=["hand"])
b.fit_collision_spheres(sphere_density=12.0, fit_type=SphereFitType.MORPHIT,
                        use_collision_mesh=True, compute_metrics=True)
b.compute_collision_matrix(prune_collisions=False, custom_ignore={...})
b.collision_spheres   # {link: [{center, radius}]}
b.collision_matrix    # {link: [link]}  ignore sets
b.link_metrics        # needs compute_metrics
```

`dycraft_ik.robot_config()` does all of this and caches it; call the builder
directly only to refit. (There is no `ee_position` on this build; FK reads off
`state.tool_poses[frame]` — see § FK.)

## Traps

| | |
|---|---|
| `res.success` | **Unusable, both directions.** Tolerances never reach the result (`position_tolerance` reads back `0.0`): `success=False` at 9.09e-08, `success=True` at 2.18e-08 elsewhere. Compare `position_error` / `rotation_error` yourself. |
| joint order | URDF emission order, not sorted: `['j1'…'j7','finger_left','finger_right','cam_tilt']`. A bare vector in the wrong order does not raise — it drives the wrong joints. |
| `build()` | Leaves `cspace.joint_names` `[]` and `default_joint_position` `None`; the loader raises if any active joint is missing. `robot_config` fills it. |
| `save()` | Does not round-trip `build()` — `KeyError: 'robot_cfg'`. |
| sphere fit | Non-deterministic (178 vs 143 spheres on one mesh), no seed parameter. The on-disk cache is what makes two calls agree. |
| `compute_collision_matrix` | Default `batch_size=10000` asked for **66.89 GiB** at 2579 spheres. Use `batch_size=256`, or `prune_collisions=False`. |
| `asset_root_path` | `<export>/bodies/`, not `<export>/` — mesh paths are relative to the URDF. |
| `urdf_path` | Must be absolute, or it resolves against cuRobo's bundled assets. |
| a failed solve | **Not a proof of infeasibility.** 96 seeds reported no solution where 512 succeeded; an approach-tilt grid of 0/10/20/30/40° missed a part that solves at 5°. Print the best error, never a boolean. |
| `sphere_density` | The default is **12**, above upstream's stated 0.1–10.0, because at 4 the model called a DyCraft-clean pose a 7.8 mm penetration. It is not a free margin — see § Density is not free. |

## Costs and sphere density — measured, RTX 3090, warm cache

`num_seeds=48`, one obstacle mesh. Process warmed first, and each solver's own
first three solves discarded, so the three costs below are separated rather
than averaged together:

| density | spheres | build | first solve | steady solve |
|---|---|---|---|---|
| 1 (upstream fitter default) | 182 | 0.5 s | 1.85 s | 0.125 s |
| 4 | 457 | 2.1 s | 2.12 s | 0.274 s |
| 12 (`dycraft_ik` default) | 2631 | 56.5 s | 5.56 s | 1.732 s |
| 32 | 5635 | 261.6 s | 11.6 s | 4.818 s |

**Everything scales with sphere count, monotonically.** Steady solve time is
close to linear — 0.60–0.85 µs per sphere across a 31x range. Build is
superlinear: 2.7 µs per sphere at 182, 46 µs at 5635.

Three separate one-off costs, worth not conflating:

| | |
|---|---|
| CUDA context | ~25 s, once per process, on the first solver constructed |
| solver build | the table above, **once per solver** |
| first `solve_pose` on a new solver | 1.9–11.6 s, on top of the build |

All three are paid **per `h.ik` call** — each call is a fresh process, so a
solver cannot outlive one. The export directory is **not wiped** between calls;
a script may cache solved tracks there.

**Build the solver once, outside the loop.** One per waypoint pays the table
above every time: a run that did that burned 30 minutes on three timeouts. And
prefer a few large `h.ik` calls to many small ones, since the build is re-paid
per call. A `timeout` returns your `stdout`, so print progress markers rather
than guessing where a slow call got to.

Other measured costs: `robot_config()` on a cache hit is 0.009 s; one obstacle
mesh takes a solve from 0.080 s to 0.972 s at density 12.

### Why the default is 12

Accuracy, and nothing else — it is straightforwardly slower than 4. Wrist band
(78 × 78 × 12 mm) vs forearm, true clearance 16.30 mm at rest:

| density | spheres (band / link5) | rest | pushed pose | true |
|---|---|---|---|---|
| 4 | 12 / 37 | 14.52 mm | **−7.83 mm** | +7.96 mm |
| 12 | 12 / 468 | 18.05 mm | 5.88 mm | +7.96 mm |
| 32 | 12 / 686 | 14.05 mm | 6.02 mm | +7.96 mm |
| 64 | 14 / 1132 | 17.98 mm | 6.77 mm | +7.96 mm |

At 4 the model calls a DyCraft-clean pose a 7.8 mm penetration. Not monotonic
at rest, either — the sphere model is a solver constraint, never a verdict.

One thin link sets the density for all 17. `RobotBuilder.refit_link_spheres(
link_name, sphere_density=...)` raises it on one link and `sphere_report()`
gives each link's `coverage` / `protrusion`; `robot_config` exposes only a
global density.

## Environment

`curobo.robot_builder` needs **`pyliblzfse`** (its error says `liblzfse`, which
is not the PyPI name). Kernels need `pip install 'cuda-core[cu12]'` or the first
FK dies with `RuntimeError: No curobo kernel backend available!`. No CPU path.
