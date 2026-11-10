<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.kinematics.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.kinematics module

Forward kinematics module.

Differentiable forward kinematics with Jacobian computation, center of mass calculation, and collision sphere generation.

Example

[\`\`](#id1)[\`](#id3)python from curobo.kinematics import Kinematics, KinematicsCfg from curobo.types import JointState

config = KinematicsCfg.from_robot_yaml_file(“franka.yml”) kin = Kinematics(config) js = JointState.from_position(q, joint_names=kin.joint_names) state = kin.compute_kinematics(js) [\`\`](#id5)[\`](#id7)

class Kinematics(

*config*,

*compute_jacobian=False*,

*compute_spheres=True*,

*compute_com=False*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

CUDA Accelerated Robot Model

Load basic kinematics from an URDF with <a href="#curobo.kinematics.KinematicsCfg.from_basic_urdf" class="reference internal" title="curobo.kinematics.KinematicsCfg.from_basic_urdf"><code class="sourceCode python">from_basic_urdf</code></a>. Check <a href="../getting-started/build_robot_model.html#tutorial-build-robot-model" class="reference internal">Build Robot Model</a> for details on how to also create a geometric representation of the robot. Currently dof is created only for links that we need to compute kinematics. E.g., for robots with many serial chains, add all links of the robot to get the correct dof. This is not an issue if you are loading collision spheres as that will cover the full geometry of the robot.

Initialize kinematics instance with a robot model configuration.

Parameters:
- **config** (<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg"><code class="sourceCode python">KinematicsCfg</code></a>) – Input robot model configuration.

- **compute_jacobian** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **compute_spheres** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **compute_com** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

\_\_init\_\_(

*config*,

*compute_jacobian=False*,

*compute_spheres=True*,

*compute_com=False*,

)

Initialize kinematics instance with a robot model configuration.

Parameters:
- **config** (<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg"><code class="sourceCode python">KinematicsCfg</code></a>) – Input robot model configuration.

- **compute_jacobian** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **compute_spheres** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **compute_com** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

update_batch_size(

*batch*,

*horizon*,

*force_update=False*,

*reset_buffers=False*,

)

Update batch size of the robot model.

Parameters:
- **batch** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Batch dimension of \[batch, horizon, dof\].

- **horizon** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Trajectory horizon dimension.

- **force_update** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Detach gradients of tensors. This is not supported.

- **reset_buffers** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Recreate the tensors even if the batch size is same.

compute_kinematics(

*joint_state*,

*idxs_env=None*,

)

Compute forward kinematics of the robot.

This is the **single public API** for forward kinematics. All external callers must use this method.

Parameters:
- **joint_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Joint state of robot. `joint_names` must be set for validation.

- **idxs_env** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Environment query index. Shape \[num_envs\], compact.

Returns:
Kinematic state of the robot.

Return type:
<a href="#curobo.kinematics.KinematicsState" class="reference internal" title="curobo.kinematics.KinematicsState">KinematicsState</a>

get_robot_link_meshes()
Get meshes of all links of the robot.

Returns:
List of all link meshes.

Return type:
List\[<a href="curobo.scene.html#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh">Mesh</a>\]

get_robot_as_mesh(

*joint_position*,

)

Transform robot links to Cartesian poses using forward kinematics and return as meshes.

Parameters:
**joint_position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>) – Joint configuration of the robot, shape should be \[1, dof\].

Returns:
List of all link meshes.

Return type:
List\[<a href="curobo.scene.html#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh">Mesh</a>\]

get_robot_as_spheres(

*q*,

*filter_valid=True*,

)

Get robot spheres using forward kinematics on given joint configuration q.

Parameters:
- **q** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>) – Joint configuration of the robot, shape should be \[1, dof\].

- **filter_valid** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Filter out spheres with radius \<= 0.

Returns:
List of all robot spheres.

Return type:
List\[<a href="curobo.scene.html#curobo.scene.Sphere" class="reference internal" title="curobo.scene.Sphere">Sphere</a>\]

get_link_poses(

*joint_position*,

*query_link_names*,

)

Get Pose of links at given joint configuration using forward kinematics.

Note that only the links specified in <a href="#curobo.kinematics.KinematicsCfg.tool_frames" class="reference internal" title="curobo.kinematics.KinematicsCfg.tool_frames"><code class="sourceCode python">tool_frames</code></a> are returned.

Parameters:
- **joint_position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>) – Joint configuration of the robot, shape should be \[batch_size, dof\].

- **query_link_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – Names of links to get pose of. This should be a subset of <a href="#curobo.kinematics.KinematicsCfg.tool_frames" class="reference internal" title="curobo.kinematics.KinematicsCfg.tool_frames"><code class="sourceCode python">tool_frames</code></a>.

Returns:
Poses of links at given joint configuration.

Return type:
<a href="curobo.types.html#curobo.types.Pose" class="reference internal" title="curobo.types.Pose">Pose</a>

property all_articulated_joint_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Names of all articulated joints of the robot.

get_self_collision_config()
Get self collision configuration parameters of the robot.

Return type:
`SelfCollisionKinematicsCfg`

get_link_mesh(*link_name*)
Get mesh of a link of the robot.

Return type:
<a href="curobo.scene.html#curobo.scene.Mesh" class="reference internal" title="curobo._src.geom.types.Mesh"><code class="sourceCode python">Mesh</code></a>

Parameters:
**link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

get_link_transform(*link_name*)
Get pose offset of a link from it’s parent joint.

Parameters:
**link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of link to get pose of.

Returns:
Pose of the link.

Return type:
<a href="curobo.types.html#curobo.types.Pose" class="reference internal" title="curobo.types.Pose">Pose</a>

get_all_link_transforms()
Get offset pose of all links with respect to their parent joint.

Return type:
<a href="curobo.types.html#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

get_dof()
Get degrees of freedom of the robot.

Return type:
<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>

property dof: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>
Degrees of freedom of the robot.

property joint_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Names of actuated joints.

property total_spheres: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>
Number of spheres used to approximate robot geometry.

property lock_jointstate: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a>
State of joints that are locked in the kinematic representation.

get_full_js(*joint_state*)
Get state of all joints, including locked joints.

This function will not provide state of mimic joints. If you need mimic joints, use <a href="#curobo.kinematics.Kinematics.get_mimic_js" class="reference internal" title="curobo.kinematics.Kinematics.get_mimic_js"><code class="sourceCode python">get_mimic_js</code></a>.

Parameters:
**joint_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – State containing articulated joints.

Returns:
State of all joints.

Return type:
<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo.types.JointState">JointState</a>

get_mimic_js(*joint_state*)
Get state of mimic joints from active joints.

Current implementation uses a for loop over joints to calculate the state. This can be optimized by using a custom CUDA kernel or a matrix multiplication.

Parameters:
**joint_state** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – State containing articulated joints.

Returns:
State of active, locked, and mimic joints.

Return type:
<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo.types.JointState">JointState</a>

update_kinematics_config(

*new_kin_config*,

)

Update kinematics representation of the robot.

A kinematics representation can be updated with new parameters. Some parameters that could require updating are state of locked joints, when a robot grasps an object. Another instance is when using different planners for different parts of the robot, example updating the state of robot base or another arm. Updations should result in the same tensor dimensions, if not then the instance of this class requires reinitialization.

Parameters:
**new_kin_config** (`KinematicsParams`) – New kinematics representation of the robot.

get_active_js(*full_js*)
Get joint state of active joints of the robot.

Parameters:
**full_js** (<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>) – Joint state of all joints.

Returns:
Joint state of active joints.

Return type:
<a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo.types.JointState">JointState</a>

property base_link: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>
Base link of the robot. Changing requires reinitializing this class.

property robot_spheres
Spheres representing robot geometry (config 0). Shape \[num_spheres, 4\].

property default_joint_position: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Default joint position of the robot. Use <a href="#curobo.kinematics.Kinematics.joint_names" class="reference internal" title="curobo.kinematics.Kinematics.joint_names"><code class="sourceCode python">joint_names</code></a> to get joint names.

property default_joint_state: <a href="curobo.types.html#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a>
Default joint state of the robot. Use <a href="#curobo.kinematics.Kinematics.joint_names" class="reference internal" title="curobo.kinematics.Kinematics.joint_names"><code class="sourceCode python">joint_names</code></a> to get joint names.

get_joint_limits()
Get joint limits of the robot.

Return type:
`JointLimits`

property kinematics_config: curobo.\_src.robot.types.kinematics_params.KinematicsParams
Kinematics configuration of the robot.

class KinematicsCfg(

*device_cfg*,

*tool_frames*,

*kinematics_config*,

*self_collision_config=None*,

*kinematics_parser=None*,

*generator_config=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Configuration for robot kinematics on GPU.

Helper functions are provided to load this configuration from an URDF file or from a cuRobo robot configuration file (<a href="../getting-started/build_robot_model.html#tutorial-build-robot-model" class="reference internal">Build Robot Model</a>). YAML and XRDF files can be loaded with <a href="#curobo.kinematics.KinematicsCfg.from_robot_yaml_file" class="reference internal" title="curobo.kinematics.KinematicsCfg.from_robot_yaml_file"><code class="sourceCode python">from_robot_yaml_file</code></a>.

Parameters:
- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **kinematics_config** (*curobo.\_src.robot.types.kinematics_params.KinematicsParams*)

- **self_collision_config** (*curobo.\_src.robot.types.self_collision_params.SelfCollisionKinematicsCfg* *\|* *None*)

- **kinematics_parser** (*curobo.\_src.robot.parser.parser_base.RobotParser* *\|* *None*)

- **generator_config** (*curobo.\_src.robot.loader.kinematics_loader_cfg.KinematicsLoaderCfg* *\|* *None*)

device_cfg: <a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a>
Device and floating point precision to use for kinematics.

tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Names of links to compute poses with forward kinematics.

kinematics_config: curobo.\_src.robot.types.kinematics_params.KinematicsParams
Tensors representing kinematics of the robot. This can be created using `KinematicsLoader`.

self_collision_config: curobo.\_src.robot.types.self_collision_params.SelfCollisionKinematicsCfg \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Collision pairs to ignore when computing self collision between spheres across all robot links. This also contains distance threshold between spheres pairs and which thread indices for calculating the distances. More details on computing these parameters is in `_build_collision_model`.

kinematics_parser: curobo.\_src.robot.parser.parser_base.RobotParser \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Parser to load kinematics from URDF or USD files. This is used to load kinematics representation of the robot. This is created using `RobotParser`. USD is an experimental feature and might not work for all robots.

generator_config: curobo.\_src.robot.loader.kinematics_loader_cfg.KinematicsLoaderCfg \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Generator config used to create this robot kinematics model.

get_joint_limits()
Get limits of actuated joints of the robot.

Returns:
Joint limits of the robot’s actuated joints.

Return type:
JointLimits

static from_basic_urdf(

*urdf_path*,

*base_link*,

*tool_frames*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Load a cuda robot model from only urdf. This does not support collision queries.

Parameters:
- **urdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path of urdf file.

- **base_link** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of base link.

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – List of tool frames to compute poses for. If None, all links are computed.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device to load robot model. Defaults to DeviceCfg().

Returns:
cuda robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

static from_content_path(

*content_path*,

*tool_frames=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*\*\*kwargs*,

)

Load robot from Contentpath containing paths to robot description files.

Parameters:
- **content_path** (<a href="curobo.types.html#curobo.types.ContentPath" class="reference internal" title="curobo._src.types.content_path.ContentPath"><code class="sourceCode python">ContentPath</code></a>) – Path to robot configuration files.

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]) – List of tool frames to compute poses for. If None, all links are computed.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device to load robot model, defaults to cuda:0.

- **kwargs** (<a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><em>Any</em></a>)

Returns:
cuda robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

static from_robot_yaml_file(

*file_path*,

*tool_frames=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*urdf_path=None*,

*\*\*kwargs*,

)

Load a robot from a cuRobo YAML or XRDF file.

See <a href="../getting-started/build_robot_model.html#tutorial-build-robot-model" class="reference internal">Build Robot Model</a> for the configuration workflow.

Parameters:
- **file_path** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\]) – Path to robot configuration file (yml or xrdf).

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]) – List of tool frames to compute poses for. If None, all links are computed.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device to load robot model, defaults to cuda:0.

- **urdf_path** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – Path to urdf file. This is required when loading a xrdf file.

- **kwargs** (<a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><em>Any</em></a>)

Returns:
cuda robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

static from_config_file(

*file_path*,

*tool_frames=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*urdf_path=None*,

)

Load a robot from a cuRobo YAML or XRDF file.

See <a href="../getting-started/build_robot_model.html#tutorial-build-robot-model" class="reference internal">Build Robot Model</a> for the configuration workflow.

Parameters:
- **file_path** (`Union`\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\]) – Path to robot configuration file (yml or xrdf).

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]) – List of tool frames to compute poses for. If None, all links are computed.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device to load robot model, defaults to cuda:0.

- **urdf_path** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – Path to urdf file. This is required when loading a xrdf file.

Returns:
cuda robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

static from_data_dict(

*data_dict*,

*tool_frames=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Load robot from a dictionary containing data for `KinematicsLoaderCfg`.

See <a href="../getting-started/build_robot_model.html#tutorial-build-robot-model" class="reference internal">Build Robot Model</a> for the data required to load a robot.

Parameters:
- **data_dict** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>\]) – Input dictionary containing robot configuration.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device to load robot model, defaults to cuda:0.

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

Returns:
cuda robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

static from_config(*config*)
Create a robot model configuration from a generator configuration.

Parameters:
**config** (`KinematicsLoaderCfg`) – Input robot generator configuration.

Returns:
robot model configuration.

Return type:
<a href="#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo.kinematics.KinematicsCfg">KinematicsCfg</a>

property cspace: curobo.\_src.robot.types.cspace_params.CSpaceParams
Get cspace parameters of the robot.

property dof: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>
Get the number of actuated joints (degrees of freedom) of the robot

\_\_init\_\_(

*device_cfg*,

*tool_frames*,

*kinematics_config*,

*self_collision_config=None*,

*kinematics_parser=None*,

*generator_config=None*,

)

Parameters:
- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **kinematics_config** (*curobo.\_src.robot.types.kinematics_params.KinematicsParams*)

- **self_collision_config** (*curobo.\_src.robot.types.self_collision_params.SelfCollisionKinematicsCfg* *\|* *None*)

- **kinematics_parser** (*curobo.\_src.robot.parser.parser_base.RobotParser* *\|* *None*)

- **generator_config** (*curobo.\_src.robot.loader.kinematics_loader_cfg.KinematicsLoaderCfg* *\|* *None*)

Return type:
None

class KinematicsState(

*tool_poses=None*,

*tool_jacobians=None*,

*robot_spheres=None*,

*robot_com=None*,

*robot_collision_geometry=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Kinematic state of robot.

All tensor fields preserve the `[batch,`` ``horizon,`` ``...]` shape from FK. No flattening is done; shapes flow end-to-end from input to output.

Parameters:
- **tool_poses** (<a href="curobo.types.html#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><em>curobo._src.types.tool_pose.ToolPose</em></a> *\|* *None*)

- **tool_jacobians** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_spheres** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_com** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_collision_geometry** (*curobo.\_src.robot.types.collision_geometry.RobotCollisionGeometry* *\|* *None*)

tool_poses: <a href="curobo.types.html#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose">curobo._src.types.tool_pose.ToolPose</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Link poses `[batch,`` ``horizon,`` ``num_links,`` ``3/4]`.

tool_jacobians: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Jacobians `[batch,`` ``horizon,`` ``num_links,`` ``6,`` ``dof]`.

robot_spheres: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Collision spheres `[batch,`` ``horizon,`` ``num_spheres,`` ``4]` (x, y, z, r).

robot_com: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Center of mass `[batch,`` ``horizon,`` ``4]` (xyz=global CoM, w=total mass).

robot_collision_geometry: curobo.\_src.robot.types.collision_geometry.RobotCollisionGeometry \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Static collision geometry descriptor.

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

get_link_spheres()
Collision spheres `[batch,`` ``horizon,`` ``num_spheres,`` ``4]`.

Return type:
<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>

clone()

detach()

copy\_(*other*)
Parameters:
**other** (<a href="#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState"><em>curobo._src.robot.kinematics.kinematics_state.KinematicsState</em></a>)

\_\_init\_\_(

*tool_poses=None*,

*tool_jacobians=None*,

*robot_spheres=None*,

*robot_com=None*,

*robot_collision_geometry=None*,

)

Parameters:
- **tool_poses** (<a href="curobo.types.html#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><em>curobo._src.types.tool_pose.ToolPose</em></a> *\|* *None*)

- **tool_jacobians** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_spheres** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_com** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **robot_collision_geometry** (*curobo.\_src.robot.types.collision_geometry.RobotCollisionGeometry* *\|* *None*)

Return type:
None
