# `h.ik` — cuRobo, a fast collision-free motion planner on the GPU

cuRobo plans motion that does not collide: every configuration along the path is
checked against the robot's own body and the scene's obstacles, and the result is
a smooth, time-parameterized joint trajectory at your frame rate. It is the right
tool whenever a robot has to move, not just reach one pose:

- **Collision-free spans.** `MotionPlanner.plan_pose` takes a start and a goal and
  returns the path between them, routed round what is in the way. Measured: one
  G1 arm planned round a 10 cm box in 14 s per call, 41 frames at 24 fps, 0 mm
  goal error, the hand detouring 12 cm (`examples/g1_arm_plan.py`). `plan_grasp`
  does approach, grasp, and lift in one call.
- **A humanoid's whole body over a whole clip.** `MotionRetargeter` solves every
  frame of a floating-base G1 from pelvis, foot, and hand targets in one call, with
  joint limits, self-collision, and scene obstacles: 240 frames in 17 s, 1.3 mm
  median error (`examples/g1_sequence.py`, which runs as it stands). A box on the
  pelvis's path moves the body round it.
- **On the robot in the film.** For a preloaded G1, `dycraft_ik.g1_robot(urdf)`
  loads cuRobo's G1 set-up onto the scene's own URDF; joint names and radians are
  the scene's, so the result keys straight onto its joints.
- **Batched.** Hundreds of targets are one GPU call (264 IK targets for four G1
  tool frames in 40 s), where a hand-written solver re-solves frame by frame:
  a numpy whole-body IK takes tens of minutes of one CPU core per pass, and avoids
  nothing.

What it leaves to you: balance (the center of mass over the support), grasp
timing and the targets themselves. Its output is joint values; you key them into
the project and check the result there.

```python
h.ik(code=None, name=None, frame=None, frames=None, script=None, timeout_s=None)
```

**Your code, your robot, your world, on the GPU.** `h.ik` runs `code` (Python
source) or `script` (a file under the run dir, e.g. `ik/plan.py` written from
your shell) with cuRobo on a GPU, which your own shell cannot reach: its
sandbox sees no GPU device. Build what the motion needs:

- **The robot.** A preloaded robot's original URDF and package root are named in
  `DYCRAFT_IK_ROBOTS` (JSON) and in the reply's `robots`. cuRobo also ships robot
  configs (`curobo/content/configs/robot/`, a Unitree G1 among them, with a
  retargeting variant). For a preloaded G1 you need not choose:
  `dycraft_ik.g1_robot(urdf)` puts cuRobo's G1 set-up on the scene's own URDF,
  so the solve is on the robot in the film (both G1 examples do this). Another
  robot: check a bundled config matches before using it.
  Lock joints, float the base, choose tool frames and collision spheres as the
  motion needs.
- **The world.** Whatever you build, or the project itself: `frame` / `frames`
  (up to 16 zero-based output indices) export this Blender snapshot into
  `ik/exports/<version>_f<frame>/`, named in `DYCRAFT_IK_EXPORTS` (JSON
  `{frame: dir}`) and in the reply's `exports`. **Nothing is exported unless you
  ask.** An export is reused while the design version is unchanged.

The code runs once, with `<run>/ik/` as its working directory. That directory
outlives the call: keep your modules, caches, and intermediate results there
and re-solve only what changed. `timeout_s` raises the default limit (600 s)
up to an hour; the call is a job, so a long solve blocks nothing.

`checkpoint()` is injected beside `result` and snapshots it to the call's
`provenance/result.partial.json`; the parent returns that as `partial_result`
when a call times out or is OOM-killed, which on this op is the failure that
costs a full budget. Call it once per solved waypoint or per pass, not per
frame.

Costs a step. Returns whatever your script assigns to the injected `result`
dict — no output schema — beside `workdir`, `exports`, `robots`, and `script`.
`error_kind: "unavailable"` means no solver is configured; author the
configuration yourself.

**Three solvers sit behind this one op, and the name undersells the planner.**
`MotionPlanner` plans a *span* — collision-checked along the whole path, and
smooth, which a stack of per-frame IK solutions is not; `InverseKinematics`
answers whether one tip pose is reachable, and batches: measured on an L40S,
264 targets for all four G1 tool frames solved in one 40 s call (declare
`max_batch_size` when building the solver; it defaults to 1). `Kinematics` is
FK. Reach for the planner by default and for IK to pick the target it plans
to. Nothing in `dycraft_ik` below is any of them — it only translates.

cuRobo does not balance a body. For a humanoid, keep the center of mass,
support, and grasp locks yourself, starting from its solutions.

**Docs, in the order to reach for them:**

1. `CUROBO.md`, beside this file: how to use cuRobo from here, and where this
   checkout differs from the published docs. It opens with which of the three to use.
2. `curobo_docs/`, beside this file: the official **v0.8.0 / cuRoboV2** docs,
   mirrored offline because a run has no network. `curobo_docs/README.md` maps
   question to file.
3. Runnable examples shipped inside the package, located from inside `h.ik`:

```python
import curobo, os; result["examples"] = os.path.dirname(curobo.__file__) + "/examples"
```

`getting_started/{inverse_kinematics,motion_planning,forward_kinematics,build_robot_model}.py`,
`reference/sphere_fit_comparison.py`.

## An exported scene

| path | what |
|---|---|
| `bodies/<instance>.urdf`, `meshes/` | the scene, one URDF per instance |
| `scene_meta.json` | rest poses, world AABBs, joint limits **and defaults**, surface units |
| `ik_meta.json` | structural self-collisions, obstacle poses |

`dycraft_ik.py` and `_rotation.py` sit in `ik/` itself, so `import dycraft_ik`
works from any call; pass an export's directory as `ik_dir=` to read it.

`ik_meta.json` carries the two things the URDF cannot say:

- **Which self-collisions are structural** — DyCraft's rest-pose overlap set
  (the one `no_self_penetration` allows) ∪ tree adjacency. A solver told to
  avoid a band overlapping the link it wraps has no solutions at all.
  `exact_adjudication: false` means it came from the SAT proxy, i.e. more
  ignore pairs and a less conservative solver.
- **Where every static part is**, as meshes with world poses.

## `dycraft_ik` — translation only

**You build the solver.** These are the arguments it takes; nothing here
solves.

```python
robot_config(instance, *, tool_links, sphere_density=12.0, lock_joints=None,
             extra_ignore=None, refit=False) -> dict     # the robot= dict
scene_config(instance, *, exclude=()) -> dict            # the scene_model= dict
sphere_report(instance) -> {link: {n, coverage, protrusion}}

world_to_base(instance, position, quaternion_wxyz=None, *, meta)
base_to_world(instance, position, quaternion_wxyz=None, *, meta)
base_pose(instance, *, meta)

joint_names(instance, *, meta, ik_meta=None)   # URDF order
default_q(instance, *, meta)                   # committed rest pose, internal units
joint_limits_internal(instance, *, meta)
q_to_surface(instance, q, *, meta)             # -> {joint: deg/m}
q_from_surface(instance, poses, *, meta)
to_authoring_joint_track(instance, qs, *, meta)   # {joint: [deg/m]}, the source's units

load_meta(ik_dir=".") ; load_ik_meta(ik_dir=".")

g1_robot(urdf, *, out_dir=".") -> dict     # cuRobo's G1 set-up on this G1 URDF; the robot= dict
```

There is deliberately no `solve_pose` wrapper: one that builds an
`InverseKinematics` per target re-pays the solver build every time — tens of
seconds at the default sphere density. Build the solver once, outside your loop.

## Five things that fail silently

1. **With an exported robot, targets are in its base frame** — 622 mm on a
   fixed-base arm cell. `world_to_base` is a full SE3 inverse, not a translation
   subtraction. (The G1 examples float the base and work in world coordinates.)
2. **URDF `q = 0` is not the rest pose.** `robot_config` puts the committed
   configuration in `cspace.default_joint_position`.
3. **`result.success` is unreliable in both directions.** Compare
   `position_error` / `rotation_error` against your own tolerances.
4. **Joint order is the URDF's, not alphabetical** — a sorted `q` put the hand
   1.02 m away while every self-consistency check passed.
5. **A solved `q` is in radians and meters, and so is `key_joint`.** Key it as
   it is. `q_to_surface` and `to_authoring_joint_track` give the source's
   degrees, for `limits`, `default`, and `initial_joints` in `design/`.

## Failure kinds

`unavailable`, `bad_args`, `bad_scene_request`, `syntax_error`,
`raised` (with `partial_result`), `unserializable_result`, `timeout`,
`launch_failed`, `runner_died`, `bad_result`. Every call lands in its job's
`provenance/` as `code.py`, `result.json`, `stdout.txt`, on success and
failure alike.

A `timeout` returns the `stdout` your script produced before the kill, so print
progress in anything long. `DYCRAFT_IK_TIMEOUT_S` defaults to 600 and is set
outside the run.

## The collision model is a constraint, not a verdict

cuRobo reasons about spheres; **`NoPenetration` decides whether an animation is
clean**, with exact CAD. At density 12 they agree to 1–2 mm and the spheres err
conservative. `h.ik` decides what to try.

## Setup (operator)

```bash
./setup_ik.sh                      # env at ~/conda_envs/dycraft-ik
IK_PREFIX=/path ./setup_ik.sh
CUROBO_SRC=/path ./setup_ik.sh     # reuse an existing checkout
```

Records `.ik_python` at the repo root. `DYCRAFT_IK_PYTHON` overrides it; empty
string forces `unavailable`. cuRobo is pinned to a commit (no published wheel,
reports `0.0.0`) and has **no CPU fallback**, so a GPU-less box is legitimately
`unavailable` and the script refuses to record a pointer there.
