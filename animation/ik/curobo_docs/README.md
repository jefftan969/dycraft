# cuRobo V2 docs, mirrored offline

A dycraft run has no network (`network.allowedDomains = ["localhost"]`), so
these are local copies of <https://nvlabs.github.io/curobo/latest/> — the
**v0.8.0 / cuRoboV2** documentation, which is the generation installed here.
NVIDIA, Apache-2.0. Refresh with `python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs`.

**Do not read these front to back — 512 KB.** Pick by question:

| question | file | size |
|---|---|---|
| How do I solve IK? | `guide_inverse_kinematics.md` | 5 KB |
| How do I plan a path, or a grasp? | `guide_motion_planning.md` | 4 KB |
| How do I do FK / score a candidate? | `guide_forward_kinematics.md` | 2 KB |
| How is a robot's collision model built? | `guide_build_robot_model.md` | 5 KB |
| What is the self-collision model? | `ref_self_collision.md` | 3 KB |
| What does sphere density do? | `ref_sphere_fitting.md` | 9 KB |
| Runtime / device knobs | `ref_runtime_configuration.md` | 3 KB |
| How does trajopt actually work? | `guide_optimization_motion.md` | 16 KB |
| **Exact signature of** `plan_pose` / `plan_cspace` / `plan_grasp` / `update_tool_pose_criteria`, and every `GraspPlanResult` field | `api_motion_planner.md` | 47 KB |
| `Pose`, `GoalToolPose`, `JointState`, **`ToolPoseCriteria`** | `api_types.md` | 110 KB |
| `Kinematics`, `KinematicsState` | `api_kinematics.md` | 34 KB |
| `RobotBuilder`, sphere fitting, collision matrix | `api_robot_builder.md` | 48 KB |
| Obstacles, `SceneCfg`, mesh worlds | `api_scene.md` | 89 KB |
| Many problems at once | `api_batch_motion_planner.md` | 36 KB |
| `fit_spheres_to_mesh`, `SphereFitType` | `api_sphere_fit.md` | 22 KB |
| — | `api_inverse_kinematics.md` (near-empty: `InverseKinematics` is only documented as an alias of `IKSolver`; use the guide) | 0 KB |

**Two things these docs will not tell you**, both in `../CUROBO.md`:

1. **What is true of *this* checkout.** It is a post-0.8.0 commit, and where it
   disagrees with the site, the checkout wins. `../CUROBO.md` records the
   measured differences — including that `result.success` is unusable in both
   directions, which the site does not mention.
2. **How any of it connects to a dycraft scene** — frames, units, joint order,
   the committed rest pose, obstacle export. That is `dycraft_ik`, and getting
   it wrong fails silently.
