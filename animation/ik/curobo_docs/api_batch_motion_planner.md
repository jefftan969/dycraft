<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.batch_motion_planner.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.batch_motion_planner module

Batch motion planning module.

This module provides batch motion planning for solving multiple independent planning problems in parallel with a single IK + trajectory optimization pass.

Example

[\`\`](#id1)[\`](#id3)python from curobo import BatchMotionPlanner, MotionPlannerCfg

config = MotionPlannerCfg.create(
robot=”franka.yml”, max_batch_size=16,

) planner = BatchMotionPlanner(config) result = planner.plan_pose(goal_tool_poses, current_states) [\`\`](#id5)[\`](#id7)

class BatchMotionPlanner(*config*)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Batch motion planner: solves `batch_size` problems in parallel.

Single IK → (optional graph seed) → TrajOpt pass with no retries. When `multi_env=False`, the PRM graph planner provides trajectory seeds. When `multi_env=True`, graph seeding is unavailable.

Parameters:
**config** (<a href="curobo.motion_planner.html#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo._src.motion.motion_planner_cfg.MotionPlannerCfg"><code class="sourceCode python">MotionPlannerCfg</code></a>) – Planner configuration. `max_batch_size` and `multi_env` are read from `config.ik_solver_config`.

\_\_init\_\_(*config*)
Parameters:
**config** (<a href="curobo.motion_planner.html#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo._src.motion.motion_planner_cfg.MotionPlannerCfg"><em>curobo._src.motion.motion_planner_cfg.MotionPlannerCfg</em></a>)

destroy()
Release all CUDA graph resources.

Call before dropping the last reference to avoid `cudaMallocAsync` warnings about captured allocations freed outside graph replay.

property attachment_manager: curobo.\_src.collision.attachment_manager.AttachmentManager
Attachment manager for attaching/detaching obstacles to robot links.

property batch_size: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

warmup(

*enable_graph=True*,

*num_warmup_iterations=5*,

)

JIT-warmup the solvers and (optionally) the graph planner.

Parameters:
- **enable_graph** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Warmup graph planner if available (multi_env=False).

- **num_warmup_iterations** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of dummy c-space and pose solves.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

plan_pose(

*goal_tool_poses*,

*current_state*,

*use_implicit_goal=True*,

*max_attempts=1*,

*success_ratio=1.0*,

*enable_graph_attempt=0*,

*finetune_attempts=1*,

*initial_iters=None*,

*time_optimal_iters=None*,

*finetune_iters=None*,

*finetune_dt_scale=0.55*,

)

Plan trajectories for a batch of pose targets.

Runs up to `max_attempts` IK -\> TrajOpt passes. On each attempt the full batch is re-solved; per-problem results are locked in on first success (first-success-wins). The loop exits early when the fraction of solved problems reaches `success_ratio`.

Parameters:
- **goal_tool_poses** (<a href="curobo.types.html#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>) – Target poses as `GoalToolPose` `[B,`` ``H,`` ``L,`` ``G,`` ``3/4]`.

- **current_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Initial joint states `(batch_size,`` ``dof)`.

- **use_implicit_goal** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Use IK solution as implicit trajectory goal.

- **max_attempts** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum number of IK + TrajOpt passes.

- **success_ratio** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Exit early when this fraction of the batch has succeeded (0.0-1.0). Default 1.0 means wait for all.

- **enable_graph_attempt** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Attempt index at which to start graph seeding (when graph planner is available).

- **finetune_attempts** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of additional time-optimal refinement passes after the initial solve.

- **initial_iters** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Optional optimizer iteration override for the initial solve.

- **time_optimal_iters** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Optional optimizer iteration override for the first refinement pass.

- **finetune_iters** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Optional optimizer iteration override for later refinement passes.

- **finetune_dt_scale** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Multiplicative factor applied to the trajectory timestep between refinement passes.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[`TrajOptSolverResult`\]

Returns:
TrajOptSolverResult with per-problem success, or None if IK never found any solution across all attempts.

plan_cspace(

*goal_states*,

*current_state*,

*max_attempts=1*,

*success_ratio=1.0*,

*enable_graph_attempt=0*,

)

Plan trajectories for a batch of joint-space targets.

Same retry / first-success-wins logic as <a href="#curobo.batch_motion_planner.BatchMotionPlanner.plan_pose" class="reference internal" title="curobo.batch_motion_planner.BatchMotionPlanner.plan_pose"><code class="sourceCode python">plan_pose</code></a>.

Parameters:
- **goal_states** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Target joint configurations `(batch_size,`` ``dof)`.

- **current_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Initial joint states `(batch_size,`` ``dof)`.

- **max_attempts** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum number of TrajOpt passes.

- **success_ratio** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Exit early when this fraction of the batch has succeeded.

- **enable_graph_attempt** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Attempt index to start graph seeding.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[`TrajOptSolverResult`\]

Returns:
TrajOptSolverResult with per-problem success.

plan_grasp(

*grasp_poses*,

*current_state*,

*grasp_approach_axis='z'*,

*grasp_approach_offset=-0.15*,

*grasp_approach_in_tool_frame=True*,

*grasp_lift_axis='z'*,

*grasp_lift_offset=-0.15*,

*grasp_lift_in_tool_frame=True*,

*plan_approach_to_grasp=True*,

*plan_grasp_to_lift=True*,

*disable_collision_links=None*,

)

Plan grasp motions for a batch: goalset -\> approach -\> grasp -\> lift.

All B problems are planned at every stage (CUDA graph stability). Problems that fail a stage get their current state substituted as the goal for subsequent stages so the optimizer doesn’t diverge.

Return type:
<a href="curobo.motion_planner.html#curobo.motion_planner.GraspPlanResult" class="reference internal" title="curobo._src.motion.motion_planner_result.GraspPlanResult"><code class="sourceCode python">GraspPlanResult</code></a>

Parameters:
- **grasp_poses** (<a href="curobo.types.html#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><em>curobo._src.types.tool_pose.GoalToolPose</em></a>)

- **current_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **grasp_approach_axis** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **grasp_approach_offset** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **grasp_approach_in_tool_frame** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **grasp_lift_axis** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **grasp_lift_offset** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **grasp_lift_in_tool_frame** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **plan_approach_to_grasp** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **plan_grasp_to_lift** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **disable_collision_links** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

property joint_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

property action_dim: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

property default_joint_state: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a>

property kinematics: <a href="curobo.kinematics.html#curobo.kinematics.Kinematics" class="reference internal" title="curobo._src.robot.kinematics.kinematics.Kinematics">curobo._src.robot.kinematics.kinematics.Kinematics</a>

compute_kinematics(

*state*,

)

Return type:
<a href="curobo.kinematics.html#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState"><code class="sourceCode python">KinematicsState</code></a>

Parameters:
**state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

update_world(

*scene_cfg*,

)

Parameters:
**scene_cfg** (*curobo.\_src.geom.types.SceneCfg*)

clear_scene_cache()

reset_seed()

enable_link_collision(

*enable_collision_links*,

)

Parameters:
**enable_collision_links** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

disable_link_collision(

*disable_collision_links*,

)

Parameters:
**disable_collision_links** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

update_tool_pose_criteria(

*tool_pose_criteria*,

)

Parameters:
**tool_pose_criteria** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><em>Dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="curobo.types.html#curobo.types.ToolPoseCriteria" class="reference internal" title="curobo._src.cost.tool_pose_criteria.ToolPoseCriteria"><em>curobo._src.cost.tool_pose_criteria.ToolPoseCriteria</em></a>*\]*)

update_link_inertial(

*link_name*,

*mass=None*,

*com=None*,

*inertia=None*,

)

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **mass** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a> *\|* *None*)

- **com** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **inertia** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

update_links_inertial(

*link_properties*,

)

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Parameters:
**link_properties** (<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><em>dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><em>dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>*\]\]*)

class MotionPlannerCfg(

*ik_solver_config*,

*trajopt_solver_config*,

*graph_planner_config=None*,

*scene_collision_cfg=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Configuration for the motion planner.

Parameters:
- **ik_solver_config** (*curobo.\_src.solver.solver_ik_cfg.IKSolverCfg*)

- **trajopt_solver_config** (*curobo.\_src.solver.solver_trajopt_cfg.TrajOptSolverCfg*)

- **graph_planner_config** (*curobo.\_src.graph_planner.graph_planner_prm_cfg.PRMGraphPlannerCfg*)

- **scene_collision_cfg** (*curobo.\_src.geom.collision.collision_scene.SceneCollisionCfg* *\|* *None*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

ik_solver_config: curobo.\_src.solver.solver_ik_cfg.IKSolverCfg

trajopt_solver_config: curobo.\_src.solver.solver_trajopt_cfg.TrajOptSolverCfg

graph_planner_config: curobo.\_src.graph_planner.graph_planner_prm_cfg.PRMGraphPlannerCfg = None

scene_collision_cfg: curobo.\_src.geom.collision.collision_scene.SceneCollisionCfg \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

device_cfg: <a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a> = DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)

static create(*robot, ik_optimizer_configs=\['ik/lbfgs_ik.yml'\], ik_transition_model='ik/transition_ik.yml', metrics_rollout='metrics_base.yml', trajopt_optimizer_configs=\['trajopt/lbfgs_bspline_trajopt.yml'\], trajopt_transition_model='trajopt/transition_bspline_trajopt.yml', graph_planner_config='graph_planner/exact_graph_planner.yml', graph_planner_rollout='metrics_base.yml', graph_planner_transition_model='graph_planner/transition_graph_planner.yml', scene_model=None, collision_cache=None, self_collision_check=True, device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32), num_ik_seeds=None, num_trajopt_seeds=None, position_tolerance=0.005, orientation_tolerance=0.05, use_cuda_graph=True, random_seed=123, optimizer_collision_activation_distance=0.01, store_debug=False, transition_model_config_instance_type=\<class 'curobo.\_src.transition.robot_state_transition_cfg.RobotStateTransitionCfg'\>, cost_manager_config_instance_type=\<class 'curobo.\_src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg'\>, max_batch_size=1, multi_env=False, max_goalset=1, interpolation_dt=0.025, interpolation_buffer_size=1000*)
Create a MotionPlannerCfg from robot, task, and scene configs.

This factory builds the three components used by MotionPlanner: IK, trajectory optimization, and PRM graph planning. Most integration code should pass a robot, optional scene/collision cache, sizing bounds, and leave the task YAMLs at their defaults.

Config path arguments may be relative YAML names under cuRobo’s content folders, absolute YAML paths, parsed dictionaries, or typed config objects where supported.

Parameters:
- **robot** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\], `RobotCfg`\]) – Robot config. String paths are resolved relative to cuRobo robot configs unless absolute. Dicts and RobotCfg objects are accepted.

- **ik_optimizer_configs** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]\]) – IK optimizer task configs. Defaults to `["ik/lbfgs_ik.yml"]`. Add `"ik/particle_ik.yml"` when broader IK search is needed.

- **ik_transition_model** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – IK transition model task config.

- **metrics_rollout** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – Metrics rollout task config shared by IK, TrajOpt, and graph planner validation.

- **trajopt_optimizer_configs** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]\]) – Trajectory optimization task configs.

- **trajopt_transition_model** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – Trajectory optimization transition model config.

- **graph_planner_config** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – PRM graph planner config. The default enables graph seeding for single-environment planning. This factory currently expects a graph planner config.

- **graph_planner_rollout** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – Rollout config used by the graph planner for collision/feasibility checks.

- **graph_planner_transition_model** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]\]) – Transition model used by graph planner rollout validation.

- **scene_model** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\], <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>\]) – Optional scene config. String paths are resolved relative to cuRobo scene configs. When None, no world obstacles are loaded initially.

- **collision_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]\]) – Optional obstacle cache sizes, for example `{"mesh":`` ``2,`` ``"cuboid":`` ``8}`. When `scene_model` is None, this still preallocates a scene collision checker for later `update_world` calls.

- **self_collision_check** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Enable robot self-collision costs/checks in IK, TrajOpt, and graph planner validation.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Tensor device and dtype configuration.

- **num_ik_seeds** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Number of IK seeds evaluated per problem. `None` uses 32 when `max_batch_size` is 1 and 16 for larger batch capacities.

- **num_trajopt_seeds** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Number of trajectory optimization seeds. `None` uses 4 when `max_batch_size` is 1 and 2 for larger batch capacities.

- **position_tolerance** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Cartesian position tolerance in meters used for IK and TrajOpt success.

- **orientation_tolerance** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Cartesian orientation tolerance in radians used for IK and TrajOpt success.

- **use_cuda_graph** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Enable CUDA graph capture for solver rollouts and optimizers. Defaults to True and should be left enabled for normal integration/runtime code. The value is forwarded to IK, TrajOpt, and graph-planner rollout components. cuRobo manages graph caches internally and pads smaller batches/ goalsets up to `max_batch_size` and `max_goalset`. Set to False only for debugging graph-capture issues; steady-state solve calls can be about 5x slower without CUDA graph replay. Also disabled automatically when `store_debug=True`.

- **random_seed** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Random seed forwarded to IK and TrajOpt seed generation. Graph planner path finding uses its own seed in PRMGraphPlannerCfg.

- **optimizer_collision_activation_distance** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Collision-cost activation distance forwarded to IK and TrajOpt optimizer rollouts. The default is 0.01 m (10 mm). Increasing this distance makes the optimizer react to obstacles earlier and keep more clearance, but can make narrow passages harder or infeasible. Decreasing it allows closer motion near obstacles, but can reduce clearance margin and make collision avoidance less robust.

- **store_debug** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Whether to store debug information. When True, CUDA graphs are disabled automatically.

- **transition_model_config_instance_type** (<a href="https://docs.python.org/3/library/typing.html#typing.Type" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Type</code></a>\[`RobotStateTransitionCfg`\]) – Advanced extension hook for custom transition model config classes.

- **cost_manager_config_instance_type** (<a href="https://docs.python.org/3/library/typing.html#typing.Type" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Type</code></a>\[`RobotCostManagerCfg`\]) – Advanced extension hook for custom cost manager config classes.

- **max_batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum batch size captured/allocated by IK and TrajOpt. Single MotionPlanner calls use one problem; BatchMotionPlanner uses up to this value. Smaller batches are padded internally.

- **multi_env** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Use one collision environment per batch item. This is for BatchMotionPlanner with per-problem worlds; graph seeding is skipped in this mode.

- **max_goalset** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum number of alternative goal poses per problem. Smaller goalsets are padded internally.

- **interpolation_dt** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Time step in seconds for the dense trajectory returned by TrajOpt. Larger values produce fewer waypoints.

- **interpolation_buffer_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum number of dense trajectory waypoints preallocated by TrajOpt. Increase this when using a smaller `interpolation_dt` or longer trajectories. Larger buffers can significantly increase GPU memory usage because state buffers are allocated for every batched seed trajectory.

Return type:
<a href="curobo.motion_planner.html#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo._src.motion.motion_planner_cfg.MotionPlannerCfg"><code class="sourceCode python">MotionPlannerCfg</code></a>

Returns:
MotionPlannerCfg containing IK, TrajOpt, graph planner, and optional scene collision configuration.

\_\_init\_\_(

*ik_solver_config*,

*trajopt_solver_config*,

*graph_planner_config=None*,

*scene_collision_cfg=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Parameters:
- **ik_solver_config** (*curobo.\_src.solver.solver_ik_cfg.IKSolverCfg*)

- **trajopt_solver_config** (*curobo.\_src.solver.solver_trajopt_cfg.TrajOptSolverCfg*)

- **graph_planner_config** (*curobo.\_src.graph_planner.graph_planner_prm_cfg.PRMGraphPlannerCfg*)

- **scene_collision_cfg** (*curobo.\_src.geom.collision.collision_scene.SceneCollisionCfg* *\|* *None*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

Return type:
None
