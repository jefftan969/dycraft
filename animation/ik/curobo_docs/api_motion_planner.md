<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.motion_planner.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.motion_planner module

Motion planning module.

This module provides high-level motion planning combining trajectory optimization and graph planning.

Example

[\`\`](#id1)[\`](#id3)python from curobo import MotionPlanner, MotionPlannerCfg

config = MotionPlannerCfg.create(robot=”franka.yml”) planner = MotionPlanner(config) result = planner.plan_pose(goal_tool_poses, current_state) [\`\`](#id5)[\`](#id7)

class MotionPlanner(*config*)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Single-problem motion planner with retry logic and graph-planner seeding.

Solves one planning problem at a time (`batch_size=1`). For batched planning see `BatchMotionPlanner`.

Parameters:
**config** (<a href="#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo.motion_planner.MotionPlannerCfg"><em>MotionPlannerCfg</em></a>)

\_\_init\_\_(*config*)
Parameters:
**config** (<a href="#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo._src.motion.motion_planner_cfg.MotionPlannerCfg"><em>curobo._src.motion.motion_planner_cfg.MotionPlannerCfg</em></a>)

destroy()
Release all CUDA graph resources.

Call before dropping the last reference to avoid `cudaMallocAsync` warnings about captured allocations freed outside graph replay.

Does not synchronize, the `gc.collect()` + `synchronize()` fence in `GraphExecutor._initialize_cuda_graph` ensures stale `cuGraphExecDestroy` calls complete before any new capture begins.

property attachment_manager: curobo.\_src.collision.attachment_manager.AttachmentManager
Attachment manager for attaching/detaching obstacles to robot links.

property joint_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

property action_dim: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

property default_joint_state: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a>

property kinematics: <a href="curobo.kinematics.html#curobo.kinematics.Kinematics" class="reference internal" title="curobo._src.robot.kinematics.kinematics.Kinematics">curobo._src.robot.kinematics.kinematics.Kinematics</a>

compute_kinematics(*state*)
Return type:
<a href="curobo.kinematics.html#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState"><code class="sourceCode python">KinematicsState</code></a>

Parameters:
**state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

warmup(

*enable_graph=True*,

*warmup_joint_index=0*,

*warmup_joint_delta=0.2*,

*num_warmup_iterations=10*,

)

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

Parameters:
- **enable_graph** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **warmup_joint_index** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **warmup_joint_delta** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **num_warmup_iterations** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

plan_pose(

*goal_tool_poses*,

*current_state*,

*use_implicit_goal=True*,

*max_attempts=5*,

*enable_graph_attempt=1*,

)

Plan a trajectory to reach target tool poses.

Goalset is auto-detected from `goal_tool_poses.num_goalset`. When `num_goalset`` ``>`` ``1` the planner uses a simpler IK+TrajOpt loop without graph seeding.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[`TrajOptSolverResult`\]

Parameters:
- **goal_tool_poses** (<a href="curobo.types.html#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><em>curobo._src.types.tool_pose.GoalToolPose</em></a>)

- **current_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **use_implicit_goal** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **max_attempts** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **enable_graph_attempt** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

plan_cspace(

*goal_state*,

*current_state*,

*max_attempts=5*,

*enable_graph_attempt=1*,

)

Plan a collision-free trajectory to a joint configuration.

Parameters:
- **goal_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Target joint configuration.

- **current_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Initial joint state.

- **max_attempts** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Maximum planning attempts.

- **enable_graph_attempt** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Attempt at which to start graph seeding.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[`TrajOptSolverResult`\]

Returns:
TrajOptSolverResult, or None if planning failed.

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

Plan a grasp motion: goalset -\> approach -\> grasp -\> lift.

Parameters:
- **grasp_poses** (<a href="curobo.types.html#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>) – Candidate grasp poses with `num_goalset` entries per link. Construct via `GoalToolPose` with the desired goalset size.

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

Return type:
<a href="#curobo.motion_planner.GraspPlanResult" class="reference internal" title="curobo._src.motion.motion_planner_result.GraspPlanResult"><code class="sourceCode python">GraspPlanResult</code></a>

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

update_world(*scene_cfg*)
Parameters:
**scene_cfg** (*curobo.\_src.geom.types.SceneCfg*)

clear_scene_cache()

reset_seed()

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

update_tool_pose_criteria(

*tool_pose_criteria*,

)

Parameters:
**tool_pose_criteria** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><em>Dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="curobo.types.html#curobo.types.ToolPoseCriteria" class="reference internal" title="curobo._src.cost.tool_pose_criteria.ToolPoseCriteria"><em>curobo._src.cost.tool_pose_criteria.ToolPoseCriteria</em></a>*\]*)

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
<a href="#curobo.motion_planner.MotionPlannerCfg" class="reference internal" title="curobo._src.motion.motion_planner_cfg.MotionPlannerCfg"><code class="sourceCode python">MotionPlannerCfg</code></a>

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

class GraspPlanResult(

*success=None*,

*approach_success=None*,

*grasp_success=None*,

*lift_success=None*,

*approach_trajectory=None*,

*approach_trajectory_dt=None*,

*approach_interpolated_trajectory=None*,

*grasp_trajectory=None*,

*grasp_trajectory_dt=None*,

*grasp_interpolated_trajectory=None*,

*lift_trajectory=None*,

*lift_trajectory_dt=None*,

*lift_interpolated_trajectory=None*,

*approach_interpolated_last_tstep=None*,

*grasp_interpolated_last_tstep=None*,

*lift_interpolated_last_tstep=None*,

*status=None*,

*planning_time=0.0*,

*goalset_index=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Result of a grasp planning operation.

Parameters:
- **success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **approach_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **grasp_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **grasp_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **lift_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **lift_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **approach_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **status** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **planning_time** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **goalset_index** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

success: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

approach_success: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

grasp_success: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

lift_success: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

approach_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

approach_trajectory_dt: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

approach_interpolated_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

grasp_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

grasp_trajectory_dt: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

grasp_interpolated_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

lift_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

lift_trajectory_dt: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

lift_interpolated_trajectory: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

approach_interpolated_last_tstep: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

grasp_interpolated_last_tstep: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

lift_interpolated_last_tstep: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

status: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

planning_time: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0

goalset_index: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

\_\_init\_\_(

*success=None*,

*approach_success=None*,

*grasp_success=None*,

*lift_success=None*,

*approach_trajectory=None*,

*approach_trajectory_dt=None*,

*approach_interpolated_trajectory=None*,

*grasp_trajectory=None*,

*grasp_trajectory_dt=None*,

*grasp_interpolated_trajectory=None*,

*lift_trajectory=None*,

*lift_trajectory_dt=None*,

*lift_interpolated_trajectory=None*,

*approach_interpolated_last_tstep=None*,

*grasp_interpolated_last_tstep=None*,

*lift_interpolated_last_tstep=None*,

*status=None*,

*planning_time=0.0*,

*goalset_index=None*,

)

Parameters:
- **success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_success** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **approach_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **approach_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **grasp_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **grasp_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **lift_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **lift_trajectory_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_interpolated_trajectory** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

- **approach_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **grasp_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **lift_interpolated_last_tstep** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **status** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **planning_time** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **goalset_index** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

Return type:
None
