<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.types.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.types module

Common data types module.

This module provides common data types used throughout CuRobo for representing robot states, poses, camera observations, and tensor device configurations.

Example

[\`\`](#id1)[\`](#id3)python from curobo.types import JointState, Pose, CameraObservation, LidarObservation, DeviceCfg

\# Create joint state joint_state = JointState.from_position(\[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0\])

\# Create pose pose = Pose(

>
>
> position=\[0.5, 0.0, 0.5\], quaternion=\[1, 0, 0, 0\] \# w, x, y, z
>
>

)

\# Create camera observation camera_obs = CameraObservation(

>
>
> depth_image=depth_tensor, intrinsics=camera_intrinsics, pose=camera_pose,
>
>

)

\# Specify tensor device and dtype device_cfg = DeviceCfg(device=”cuda:0”, dtype=torch.float32) [\`\`](#id5)[\`](#id7)

class JointState(

*position*,

*velocity=None*,

*acceleration=None*,

*joint_names=None*,

*jerk=None*,

*device_cfg=DeviceCfg(device=device(type='cuda'*,

*index=0)*,

*dtype=torch.float32*,

*collision_geometry_dtype=torch.float32*,

*collision_gradient_dtype=torch.float32*,

*collision_distance_dtype=torch.float32)*,

*dt=None*,

*aux_data=\<factory\>*,

*knot=None*,

*knot_dt=None*,

*control_space=None*,

)

Bases: `curobo._src.state.state_base.State`

Joint-space robot state (position and optional derivatives).

Convention: use `joint_state` or `js` for JointState objects; use `q` for raw position tensors.

Parameters:
- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **velocity** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **acceleration** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

- **jerk** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **aux_data** (<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><em>dict</em></a>)

- **knot** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **knot_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **control_space** (*curobo.\_src.types.control_space.ControlSpace* *\|* *None*)

position: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>

velocity: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

acceleration: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

joint_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

jerk: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

device_cfg: <a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a> = DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)

dt: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

aux_data: <a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)">dict</a>

knot: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

knot_dt: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

control_space: curobo.\_src.types.control_space.ControlSpace \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

data_ptr()

property device: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)">torch.device</a>

property dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a>

property shape: <a href="https://docs.pytorch.org/docs/stable/size.html#torch.Size" class="reference external" title="(in PyTorch v2.13)">torch.Size</a>

property ndim: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

static from_numpy(

*joint_names*,

*position*,

*velocity=None*,

*acceleration=None*,

*jerk=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Parameters:
- **joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **position** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>)

- **velocity** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* *None*)

- **acceleration** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* *None*)

- **jerk** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* *None*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

static from_position(

*position*,

*joint_names=None*,

)

Parameters:
- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

static from_state_tensor(

*state_tensor*,

*joint_names=None*,

*dof=7*,

)

static from_list(

*position*,

*velocity*,

*acceleration*,

*device_cfg*,

)

Parameters:
**device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo.types.DeviceCfg"><em>DeviceCfg</em></a>*(device=device(type='cuda',* *index=0),* *dtype=torch.float32,* *collision_geometry_dtype=torch.float32,* *collision_gradient_dtype=torch.float32,* *collision_distance_dtype=torch.float32)*)

static zeros(

*size*,

*device_cfg*,

*joint_names=None*,

)

Parameters:
- **size** (<a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><em>Tuple</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

to(*device_cfg*)
Parameters:
**device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

clone()

detach()

copy_reference(

*in_joint_state*,

)

Copy reference to in_joint_state

Parameters:
**in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

copy\_(

*in_joint_state*,

*allow_clone=True*,

)

Parameters:
- **in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **allow_clone** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

copy_data(*in_joint_state*)
Copy data from in_joint_state to self

Parameters:
**in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

unsqueeze(*idx*)
Parameters:
**idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

squeeze(*dim=0*)
Parameters:
**dim** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* *None*)

view(*\*shape*)

blend(*coeff*, *new_state*)
Parameters:
- **coeff** (*curobo.\_src.state.filter_coeff.FilterCoeff*)

- **new_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

get_state_tensor()

stack(*new_state*)
Parameters:
**new_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

cat(*other_js*, *dim*)
Parameters:
- **other_js** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **dim** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

repeat(*repeat_input*)
Parameters:
**repeat_input** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]*)

repeat_seeds(*num_seeds*)
Parameters:
**num_seeds** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

apply_kernel(*kernel_mat*)

scale(*dt*)
Parameters:
**dt** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

scale_by_dt(*dt*, *new_dt*)
Parameters:
- **dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **new_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

scale_time(*new_dt*)
Parameters:
**new_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

calculate_fd_from_position(

*dt=None*,

)

Parameters:
**dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

reorder(*joint_names*)
Reorder joint state to match the given joint names order.

Parameters:
**joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – List of joint names in the desired order.

Return type:
<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>

Returns:
New JointState with joints reordered to match joint_names.

reindex(*joint_names*)
Reindex joint state in-place to match the given joint names order.

Parameters:
**joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – List of joint names in the desired order.

get_augmented_joint_state(

*joint_names*,

*lock_joints=None*,

)

Return type:
<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><code class="sourceCode python">JointState</code></a>

Parameters:
**lock_joints** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a> *\|* *None*)

append_joints(*joint_state*)
Parameters:
**joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

gather_by_seed_index(*idx*)
Parameters:
**idx** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

copy_only_index(

*in_joint_state*,

*idx*,

)

Parameters:
- **in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

copy_at_index(

*in_joint_state*,

*idx*,

)

Parameters:
- **in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

copy_at_batch_seed_indices(

*in_joint_state*,

*batch_idx*,

*seed_idx*,

)

Parameters:
- **in_joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **batch_idx** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **seed_idx** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

get_trajectory_at_horizon_index(

*horizon_index*,

)

Parameters:
**horizon_index** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

trim_trajectory(

*start_idx*,

*end_idx=None*,

)

Parameters:
- **start_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **end_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* *None*)

index_dof(*idx*)
Parameters:
**idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

\_\_init\_\_(

*position*,

*velocity=None*,

*acceleration=None*,

*joint_names=None*,

*jerk=None*,

*device_cfg=DeviceCfg(device=device(type='cuda'*,

*index=0)*,

*dtype=torch.float32*,

*collision_geometry_dtype=torch.float32*,

*collision_gradient_dtype=torch.float32*,

*collision_distance_dtype=torch.float32)*,

*dt=None*,

*aux_data=\<factory\>*,

*knot=None*,

*knot_dt=None*,

*control_space=None*,

)

Parameters:
- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **velocity** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **acceleration** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **joint_names** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

- **jerk** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **aux_data** (<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><em>dict</em></a>)

- **knot** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **knot_dt** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **control_space** (*curobo.\_src.types.control_space.ControlSpace* *\|* *None*)

Return type:
None

class RobotState(

*joint_state*,

*joint_torque=None*,

*cuda_robot_model_state=None*,

)

Bases: `curobo._src.state.state_base.State`

Parameters:
- **joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **joint_torque** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **cuda_robot_model_state** (<a href="curobo.kinematics.html#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState"><em>curobo._src.robot.kinematics.kinematics_state.KinematicsState</em></a> *\|* *None*)

joint_state: <a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState">curobo._src.state.state_joint.JointState</a>

joint_torque: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

cuda_robot_model_state: <a href="curobo.kinematics.html#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState">curobo._src.robot.kinematics.kinematics_state.KinematicsState</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

data_ptr()

detach()

property robot_spheres: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>

property link_poses: <a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose">curobo._src.types.tool_pose.ToolPose</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>

property tool_poses: <a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose">curobo._src.types.tool_pose.ToolPose</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]

copy_at_batch_seed_indices(

*other*,

*batch_idx*,

*seed_idx*,

)

Copy robot state at specific batch and seed indices

Parameters:
- **other** (<a href="#curobo.types.RobotState" class="reference internal" title="curobo._src.state.state_robot.RobotState"><em>curobo._src.state.state_robot.RobotState</em></a>)

- **batch_idx** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **seed_idx** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

copy_only_index(*other*, *index*)
Copy robot state at specific indices

Parameters:
- **other** (<a href="#curobo.types.RobotState" class="reference internal" title="curobo._src.state.state_robot.RobotState"><em>curobo._src.state.state_robot.RobotState</em></a>)

- **index** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

get_link_pose(*link_name*)
Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
**link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

clone()
Return type:
<a href="#curobo.types.RobotState" class="reference internal" title="curobo._src.state.state_robot.RobotState"><code class="sourceCode python">RobotState</code></a>

copy\_(*other*)
Parameters:
**other** (<a href="#curobo.types.RobotState" class="reference internal" title="curobo._src.state.state_robot.RobotState"><em>curobo._src.state.state_robot.RobotState</em></a>)

\_\_init\_\_(

*joint_state*,

*joint_torque=None*,

*cuda_robot_model_state=None*,

)

Parameters:
- **joint_state** (<a href="#curobo.types.JointState" class="reference internal" title="curobo._src.state.state_joint.JointState"><em>curobo._src.state.state_joint.JointState</em></a>)

- **joint_torque** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **cuda_robot_model_state** (<a href="curobo.kinematics.html#curobo.kinematics.KinematicsState" class="reference internal" title="curobo._src.robot.kinematics.kinematics_state.KinematicsState"><em>curobo._src.robot.kinematics.kinematics_state.KinematicsState</em></a> *\|* *None*)

Return type:
None

class Pose(

*position=None*,

*quaternion=None*,

*rotation=None*,

*batch_size=1*,

*name='ee_link'*,

*normalize_rotation=False*,

)

Bases: <a href="https://docs.python.org/3/library/typing.html#typing.Sequence" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Sequence</code></a>

Pose representation used in CuRobo. You can initialize a pose by calling pose = Pose(position, quaternion).

Parameters:
- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **rotation** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **normalize_rotation** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

position: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Position is represented as x, y, z, in meters

quaternion: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Quaternion is represented as w, x, y, z.

rotation: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Rotation is represents orientation as a 3x3 rotation matrix

batch_size: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a> = 1
Batch size will be initialized from input.

name: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = 'ee_link'
name of link that this pose represents.

normalize_rotation: <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
quaternion input will be normalized when this flag is enabled. This is recommended when a pose comes from an external source as some programs do not send normalized quaternions.

detach()

requires_grad\_(*requires_grad*)
Parameters:
**requires_grad** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

property batch

property device

property ndim

static from_matrix(*matrix*)
Parameters:
**matrix** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

classmethod from_euler_xyz(

*euler_xyz*,

*position=None*,

)

Create a Pose from XYZ Euler angles and optional position.

This uses the extrinsic XYZ convention: rotations are applied around fixed world axes in order X, Y, Z.

Parameters:
- **euler_xyz** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>) – \[…, 3\] Euler angles \[rx, ry, rz\] in radians.

- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Optional \[…, 3\] position. If None, defaults to zero.

Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Returns:
Pose with the specified position (or zero) and the corresponding quaternion.

classmethod from_euler_xyz_intrinsic(

*euler_xyz*,

*position=None*,

)

Create a Pose from XYZ Euler angles using intrinsic convention.

Intrinsic XYZ: each rotation is applied around the current body-frame axis (X → Y → Z). This matches URDF virtual base joint chains where successive revolute joints rotate in the parent frame.

See <a href="#curobo.types.Pose.from_euler_xyz" class="reference internal" title="curobo.types.Pose.from_euler_xyz"><code class="sourceCode python">from_euler_xyz</code></a> for the extrinsic convention.

Parameters:
- **euler_xyz** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>) – \[…, 3\] Euler angles \[rx, ry, rz\] in radians.

- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Optional \[…, 3\] position. If None, defaults to zero.

Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Returns:
Pose with the specified position (or zero) and the corresponding quaternion.

get_rotation_matrix()

get_rotation()

stack(*other_pose*)
Parameters:
**other_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

repeat(*n*)
Repeat pose

Parameters:
**n** (*\[*<a href="https://docs.python.org/3/library/functions.html#type" class="reference external" title="(in Python v3.14)"><em>type</em></a>*\]*) – \[description\]

unsqueeze(*dim=-1*)

squeeze(*dim=-1*)

repeat_seeds(*num_seeds*)
Parameters:
**num_seeds** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

get_index(*b*, *n=None*)
Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
- **b** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **n** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a> *\|* *None*)

apply_kernel(*kernel_mat*)

classmethod from_numpy(

*position*,

*quaternion*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Parameters:
- **position** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>)

- **quaternion** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

classmethod from_list(

*pose*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*q_xyzw=False*,

)

Parameters:
- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

classmethod from_batch_list(

*pose*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*q_xyzw=False*,

)

Parameters:
- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]*)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

to_list(*q_xyzw=False*)

tolist(*q_xyzw=False*)

clone()

to(*device_cfg=None*, *device=None*)
Parameters:
- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a> *\|* *None*)

- **device** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><em>torch.device</em></a> *\|* *None*)

get_matrix(*out_matrix=None*)
Parameters:
**out_matrix** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

get_affine_matrix(*out_matrix=None*)
Parameters:
**out_matrix** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

get_numpy_affine_matrix()

get_numpy_matrix()

inverse()
Inverse of pose

Returns:
inverse pose

Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo.types.Pose">Pose</a>

get_pose_vector()

copy\_(*pose*)
Copies pose data from another memory buffer. This will create a new instance if buffers are not same shape

Parameters:
**pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo.types.Pose"><em>Pose</em></a>) – \_description\_

static cat(*pose_list*)
Parameters:
**pose_list** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>*\]*)

distance(

*other_pose*,

*use_phi3=False*,

)

Parameters:
- **other_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

- **use_phi3** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

angular_distance(

*other_pose*,

*use_phi3=False*,

)

This function computes the angular distance using either phi_3 or axis-angle.

See Huynh, Du Q. “Metrics for 3D rotations: Comparison and analysis.” Journal of Mathematical Imaging and Vision 35 (2009): 155-164 for phi_3 metric.

Parameters:
- **current_quat** – other pose quaternion. Shape: \[…, 4\]

- **other_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

- **use_phi3** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
\[…\]

Return type:
Angular distance in range \[0,1\]. Shape

linear_distance(*other_pose*)
Parameters:
**other_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

multiply(

*other_pose*,

*out_position=None*,

*out_quaternion=None*,

)

Parameters:
- **other_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

- **out_position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **out_quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

transform_point(

*points*,

*out_buffer=None*,

*gp_out=None*,

*gq_out=None*,

*gpt_out=None*,

)

Parameters:
- **points** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **out_buffer** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gp_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gq_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gpt_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

transform_points(

*points*,

*out_buffer=None*,

*gp_out=None*,

*gq_out=None*,

*gpt_out=None*,

)

Parameters:
- **points** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **out_buffer** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gp_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gq_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gpt_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

batch_transform_points(

*points*,

*out_buffer=None*,

*gp_out=None*,

*gq_out=None*,

*gpt_out=None*,

)

Parameters:
- **points** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **out_buffer** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gp_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gq_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gpt_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

batch_transform_points_inverse(

*points*,

*out_buffer=None*,

*gp_out=None*,

*gq_out=None*,

*gpt_out=None*,

)

Parameters:
- **points** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **out_buffer** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gp_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gq_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **gpt_out** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

property shape

compute_offset_pose(*offset*)
Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
**offset** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

compute_local_pose(*world_pose*)
Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
**world_pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>)

contiguous()
Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

\_\_init\_\_(

*position=None*,

*quaternion=None*,

*rotation=None*,

*batch_size=1*,

*name='ee_link'*,

*normalize_rotation=False*,

)

Parameters:
- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **rotation** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **normalize_rotation** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Return type:
None

class ToolPose(

*tool_frames*,

*position*,

*quaternion*,

)

Bases: <a href="https://docs.python.org/3/library/typing.html#typing.Sequence" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Sequence</code></a>

4D FK output: `[batch,`` ``horizon,`` ``num_links,`` ``3/4]`.

No goalset dimension; this represents the current state of the robot, not a goal target.

Parameters:
- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Link/frame names, length must equal `position.shape[2]`.

position: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Position tensor `[batch,`` ``horizon,`` ``num_links,`` ``3]`.

quaternion: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Quaternion tensor `[batch,`` ``horizon,`` ``num_links,`` ``4]` (wxyz).

property batch_size: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property horizon: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property num_links: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property shape

property ndim

property device

get_link_pose(

*link_name*,

*make_contiguous=False*,

)

Extract a single link as a 2D Pose `[B*H,`` ``3/4]`.

Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **make_contiguous** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

to_dict(*make_contiguous=True*)
Convert to dictionary mapping link names to 2D Poses.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>\]

Parameters:
**make_contiguous** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

copy\_(*other*)
Parameters:
**other** (<a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><em>curobo._src.types.tool_pose.ToolPose</em></a>)

requires_grad\_(*requires_grad*)
Parameters:
**requires_grad** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

clone()
Return type:
<a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><code class="sourceCode python">ToolPose</code></a>

detach()
Return type:
<a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><code class="sourceCode python">ToolPose</code></a>

contiguous()
Return type:
<a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><code class="sourceCode python">ToolPose</code></a>

reorder_links(

*ordered_tool_frames*,

)

Reorder links. Returns a new ToolPose.

Return type:
<a href="#curobo.types.ToolPose" class="reference internal" title="curobo._src.types.tool_pose.ToolPose"><code class="sourceCode python">ToolPose</code></a>

Parameters:
**ordered_tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

as_goal(

*ordered_tool_frames=None*,

)

Convert to GoalToolPose by adding a goalset dimension (num_goalset=1).

Parameters:
**ordered_tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]) – Optionally reorder/select a subset of links.

Return type:
<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>

Returns:
GoalToolPose `[B,`` ``H,`` ``L,`` ``1,`` ``3/4]`.

\_\_init\_\_(

*tool_frames*,

*position*,

*quaternion*,

)

Parameters:
- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

Return type:
None

class GoalToolPose(

*tool_frames*,

*position*,

*quaternion*,

)

Bases: <a href="https://docs.python.org/3/library/typing.html#typing.Sequence" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Sequence</code></a>

5D goal specification: `[batch,`` ``horizon,`` ``num_links,`` ``num_goalset,`` ``3/4]`.

Used only on the goal/target side. horizon=1 for static goals, horizon\>1 for per-timestep targets.

Parameters:
- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Link/frame names, length must equal `position.shape[2]`.

position: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Position tensor `[batch,`` ``horizon,`` ``num_links,`` ``num_goalset,`` ``3]`.

quaternion: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Quaternion tensor `[batch,`` ``horizon,`` ``num_links,`` ``num_goalset,`` ``4]` (wxyz).

property batch_size: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property horizon: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property num_links: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property num_goalset: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>

property shape

property ndim

property device

classmethod from_poses(

*pose_dict*,

*ordered_tool_frames=None*,

*num_goalset=1*,

)

Build from per-link Pose objects.

Each Pose has position `[batch,`` ``3]` or `[batch`` ``*`` ``num_goalset,`` ``3]`.

Return type:
<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>

Returns:
GoalToolPose `[batch,`` ``1,`` ``num_links,`` ``num_goalset,`` ``3/4]` (horizon=1).

Parameters:
- **pose_dict** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><em>Dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a>*\]*)

- **ordered_tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

- **num_goalset** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

get_link_pose(

*link_name*,

*make_contiguous=False*,

)

Extract a single link as a 2D Pose `[B*H*G,`` ``3/4]`.

Return type:
<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **make_contiguous** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

to_dict(

*make_contiguous=True*,

)

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>\]

Parameters:
**make_contiguous** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

copy\_(*other*)
Parameters:
**other** (<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><em>curobo._src.types.tool_pose.GoalToolPose</em></a>)

requires_grad\_(

*requires_grad*,

)

Parameters:
**requires_grad** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

clone()
Return type:
<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>

detach()
Return type:
<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>

reorder_links(

*ordered_tool_frames*,

)

Return type:
<a href="#curobo.types.GoalToolPose" class="reference internal" title="curobo._src.types.tool_pose.GoalToolPose"><code class="sourceCode python">GoalToolPose</code></a>

Parameters:
**ordered_tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

\_\_init\_\_(

*tool_frames*,

*position*,

*quaternion*,

)

Parameters:
- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]*)

- **position** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **quaternion** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

Return type:
None

class ToolPoseCriteria(

*terminal_pose_axes_weight_factor=None*,

*non_terminal_pose_axes_weight_factor=None*,

*terminal_pose_convergence_tolerance=None*,

*non_terminal_pose_convergence_tolerance=None*,

*project_distance_to_goal=False*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Criteria for a link pose.

This class is used to define the nature of the cost between the current pose and the goal pose. This used as part of the goalset cost term.

Parameters:
- **terminal_pose_axes_weight_factor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **non_terminal_pose_axes_weight_factor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **terminal_pose_convergence_tolerance** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **non_terminal_pose_convergence_tolerance** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **project_distance_to_goal** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

terminal_pose_axes_weight_factor: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the terminal position and orientation. This is multiplied with the weight.

non_terminal_pose_axes_weight_factor: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the non-terminal position and orientation. This is multiplied with the weight.

terminal_pose_convergence_tolerance: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Convergence tolerance for the terminal position and orientation. This should be of shape (2,). Position unit is meter and orientation unit is radian.

non_terminal_pose_convergence_tolerance: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Convergence tolerance for the non-terminal position and orientation. This should be of shape (2,). Position unit is meter and orientation unit is radian.

project_distance_to_goal: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)">bool</a> = False
If true, the distance is computed after projecting the current pose to the goal frame.

device_cfg: <a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a> = DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)

clone()

copy\_(*other*)
Parameters:
**other** (<a href="#curobo.types.ToolPoseCriteria" class="reference internal" title="curobo._src.cost.tool_pose_criteria.ToolPoseCriteria"><em>curobo._src.cost.tool_pose_criteria.ToolPoseCriteria</em></a>)

static track_position(

*xyz=\[1.0, 1.0, 1.0\]*,

)

Parameters:
**xyz** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

static track_orientation(

*rpy=\[0.001, 0.001, 0.001\]*,

*non_terminal_scale=1.0*,

)

Parameters:
- **rpy** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **non_terminal_scale** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

static track_position_and_orientation(

*xyz=\[1.0, 1.0, 1.0\]*,

*rpy=\[1.0, 1.0, 1.0\]*,

*non_terminal_scale=0.1*,

)

Parameters:
- **xyz** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **rpy** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **non_terminal_scale** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

static linear_motion(

*axis='z'*,

*non_terminal_scale=1.0*,

*project_distance_to_goal=True*,

)

Parameters:
- **axis** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **non_terminal_scale** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **project_distance_to_goal** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

static disabled()
Create criteria that disables pose tracking for this tool frame.

Use this when you want to include a tool frame in the solver but not apply any pose cost to it.

Returns:
ToolPoseCriteria with all weight factors set to zero.

\_\_init\_\_(

*terminal_pose_axes_weight_factor=None*,

*non_terminal_pose_axes_weight_factor=None*,

*terminal_pose_convergence_tolerance=None*,

*non_terminal_pose_convergence_tolerance=None*,

*project_distance_to_goal=False*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Parameters:
- **terminal_pose_axes_weight_factor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **non_terminal_pose_axes_weight_factor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **terminal_pose_convergence_tolerance** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **non_terminal_pose_convergence_tolerance** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **project_distance_to_goal** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* <a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **device_cfg** (<a href="#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

Return type:
None

class CameraObservation(

*name='camera_image'*,

*rgb_image=None*,

*depth_image=None*,

*image_segmentation=None*,

*projection_matrix=None*,

*projection_rays=None*,

*resolution=None*,

*pose=None*,

*intrinsics=None*,

*timestamp=None*,

*depth_to_meter=0.001*,

*feature_grid=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **rgb_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **depth_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **image_segmentation** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **projection_matrix** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **projection_rays** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **resolution** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]* *\|* *None*)

- **pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a> *\|* *None*)

- **intrinsics** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **timestamp** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **depth_to_meter** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **feature_grid** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

name: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = 'camera_image'

rgb_image: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
rgb image format is BxHxWxchannels

depth_image: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

image_segmentation: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

projection_matrix: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

projection_rays: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

resolution: List\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

pose: <a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose">curobo._src.types.pose.Pose</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

intrinsics: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Batch of intrinsics matrices of shape (b, 3, 3). contains the following fields: \[\[fx, 0, cx\], \[0, fy, cy\], \[0, 0, 1\]\]

Type:
intrinsics_matrix

timestamp: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

depth_to_meter: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.001

feature_grid: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Optional neural feature grid. Shape `(num_cameras,`` ``feature_H,`` ``feature_W,`` ``feature_dim)` float16, channels-last with stride 1 on the channel dim. Dense per-pixel features are represented by `feature_H`` ``==`` ``image_H` and `feature_W`` ``==`` ``image_W`. Consumed by the mapper’s per-block feature integration kernel when `feature_dim`` ``>`` ``0`.

Magnitude convention: per-element values must be `O(1)` (e.g. L2-normalized embeddings). The mapper stores per-block feature accumulators as fp16 with a post-frame weight cap; inputs well outside this range can overflow the per-thread footprint sum within a single frame before the cap engages.

filter_depth(

*distance=0.01*,

)

Parameters:
**distance** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

property shape

copy\_(*new_data*)
Parameters:
**new_data** (<a href="#curobo.types.CameraObservation" class="reference internal" title="curobo._src.types.camera.CameraObservation"><em>curobo._src.types.camera.CameraObservation</em></a>)

clone()

to(*device*)
Parameters:
**device** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><em>torch.device</em></a>)

get_pointcloud(

*project_to_pose=False*,

)

Parameters:
**project_to_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

extract_depth_from_structured_pointcloud(

*pointcloud*,

*output_image=None*,

)

Extract depth image from structured pointcloud.

This function assumes the pointcloud maintains the spatial grid structure \[batch, height, width, 3\] where Z-axis represents depth values.

**Important:** This only works for structured pointclouds in camera frame where Z-axis is aligned with depth. If the pointcloud has been transformed to world frame (e.g., via self.pose), this will NOT work correctly.

Parameters:
- **pointcloud** – Structured pointcloud in camera frame. Shape: \[h, w, 3\] or \[b, h, w, 3\].

- **output_image** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Optional pre-allocated output tensor. Shape: \[h, w\] or \[b, h, w\].

Returns:
\[b, h, w\].

Return type:
Depth image extracted from Z-coordinates. Shape

update_projection_rays()

stack(

*new_observation*,

*dim=0*,

)

Parameters:
- **new_observation** (<a href="#curobo.types.CameraObservation" class="reference internal" title="curobo._src.types.camera.CameraObservation"><em>curobo._src.types.camera.CameraObservation</em></a>)

- **dim** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

save_to_file(

*file_path*,

)

Parameters:
**file_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

\_\_init\_\_(

*name='camera_image'*,

*rgb_image=None*,

*depth_image=None*,

*image_segmentation=None*,

*projection_matrix=None*,

*projection_rays=None*,

*resolution=None*,

*pose=None*,

*intrinsics=None*,

*timestamp=None*,

*depth_to_meter=0.001*,

*feature_grid=None*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **rgb_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **depth_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **image_segmentation** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **projection_matrix** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **projection_rays** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **resolution** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]* *\|* *None*)

- **pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a> *\|* *None*)

- **intrinsics** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **timestamp** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **depth_to_meter** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **feature_grid** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

Return type:
None

class LidarObservation(

*name='lidar_range_image'*,

*range_image=None*,

*rgb_image=None*,

*feature_grid=None*,

*pose=None*,

*valid_range_m=None*,

*elevation_range_rad=None*,

*timestamp=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Structured LiDAR range-image observation.

The mapper expects range, RGB, and optional feature tensors to already be registered in the LiDAR pixel frame. Range values are Euclidean distance from the LiDAR origin, not camera z-depth.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **range_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **rgb_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **feature_grid** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a> *\|* *None*)

- **valid_range_m** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **elevation_range_rad** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **timestamp** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

name: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = 'lidar_range_image'

range_image: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Range image in meters. Shape `(num_lidars,`` ``H,`` ``W)` float32.

rgb_image: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
RGB image aligned to `range_image`. Shape `(num_lidars,`` ``H,`` ``W,`` ``3)` uint8.

feature_grid: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Optional neural feature grid registered to the LiDAR image frame. Shape `(num_lidars,`` ``feature_H,`` ``feature_W,`` ``feature_dim)` float16, channels-last with stride 1 on the channel dim.

pose: <a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose">curobo._src.types.pose.Pose</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
LiDAR poses in the map/world frame, leading dimension `num_lidars`.

valid_range_m: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Per-LiDAR valid range bounds in meters. Shape `(num_lidars,`` ``2)` with columns `[min_m,`` ``max_m]`.

elevation_range_rad: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Per-LiDAR elevation bounds in radians. Shape `(num_lidars,`` ``2)` with columns `[min_elevation_rad,`` ``max_elevation_rad]`. For planar scans with `H`` ``==`` ``1`, both entries must be equal.

timestamp: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None

property shape

copy\_(*new_data*)
Parameters:
**new_data** (<a href="#curobo.types.LidarObservation" class="reference internal" title="curobo._src.types.lidar.LidarObservation"><em>curobo._src.types.lidar.LidarObservation</em></a>)

clone()

to(*device*)
Parameters:
**device** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><em>torch.device</em></a>)

\_\_init\_\_(

*name='lidar_range_image'*,

*range_image=None*,

*rgb_image=None*,

*feature_grid=None*,

*pose=None*,

*valid_range_m=None*,

*elevation_range_rad=None*,

*timestamp=None*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **range_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **rgb_image** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **feature_grid** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **pose** (<a href="#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><em>curobo._src.types.pose.Pose</em></a> *\|* *None*)

- **valid_range_m** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **elevation_range_rad** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **timestamp** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

Return type:
None

class ContentPath(

*robot_config_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')*,

*robot_xrdf_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')*,

*robot_urdf_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*robot_asset_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*scene_config_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/scene')*,

*world_asset_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*robot_config_absolute_path=None*,

*robot_xrdf_absolute_path=None*,

*robot_urdf_absolute_path=None*,

*robot_asset_absolute_path=None*,

*scene_config_absolute_path=None*,

*robot_config_file=None*,

*robot_xrdf_file=None*,

*robot_urdf_file=None*,

*robot_asset_subroot_path=None*,

*scene_config_file=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Dataclass to store root path of configuration and assets.

Parameters:
- **robot_config_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_xrdf_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_urdf_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_asset_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **scene_config_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **world_asset_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_config_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_xrdf_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_urdf_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_asset_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **scene_config_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_config_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_xrdf_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_urdf_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_asset_subroot_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **scene_config_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

robot_config_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')
Root path for robot configuration file, either xrdf or yml.

robot_xrdf_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')
Root path for robot XRDF.

robot_urdf_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')
Root path for robot URDF.

robot_asset_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')
Root path for robot meshes and textures.

scene_config_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/scene')
Root path for world description files (yml).

world_asset_root_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> = PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')
Root path for world assets (meshes, nvblox maps).

robot_config_absolute_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Absolute path to the robot configuration file. If this is provided, the [:var:\`robot_config_root_path\`\`](#id9) will be ignored.

robot_xrdf_absolute_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Absolute path to the robot XRDF file. If this is provided, the [:var:\`robot_xrdf_root_path\`](#id11) will be ignored.

robot_urdf_absolute_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Absolute path to the robot URDF file. If this is provided, the [:var:\`robot_urdf_root_path\`](#id13) will be ignored.

robot_asset_absolute_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Absolute path to the robot assets. If this is provided, the [:var:\`robot_asset_root_path\`](#id15) will be ignored.

scene_config_absolute_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Absolute path to the world description file. If this is provided, the [:var:\`scene_config_root_path\`](#id17) will be ignored.

robot_config_file: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Relative path to the robot configuration file. If this is provided, the robot_config_absolute_path is initialized with the concatenation of robot_config_root_path and robot_config_file.

robot_xrdf_file: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Relative path to the robot XRDF file. If this is provided, the robot_xrdf_absolute_path is initialized with the concatenation of robot_xrdf_root_path and robot_xrdf_file.

robot_urdf_file: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Relative path to the robot URDF file. If this is provided, the robot_urdf_absolute_path is initialized with the concatenation of robot_urdf_root_path and robot_urdf_file.

robot_asset_subroot_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Relative path to the robot assets.

scene_config_file: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Relative path to the world description file. If this is provided, the scene_config_absolute_path is initialized with the concatenation of scene_config_root_path and scene_config_file.

get_robot_configuration_path()
Get the robot configuration path.

\_\_init\_\_(

*robot_config_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')*,

*robot_xrdf_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/robot')*,

*robot_urdf_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*robot_asset_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*scene_config_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/configs/scene')*,

*world_asset_root_path=PosixPath('/home/runner/work/curobo/curobo/curobo/content/assets')*,

*robot_config_absolute_path=None*,

*robot_xrdf_absolute_path=None*,

*robot_urdf_absolute_path=None*,

*robot_asset_absolute_path=None*,

*scene_config_absolute_path=None*,

*robot_config_file=None*,

*robot_xrdf_file=None*,

*robot_urdf_file=None*,

*robot_asset_subroot_path=None*,

*scene_config_file=None*,

)

Parameters:
- **robot_config_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_xrdf_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_urdf_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_asset_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **scene_config_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **world_asset_root_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **robot_config_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_xrdf_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_urdf_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_asset_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **scene_config_absolute_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_config_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_xrdf_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_urdf_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **robot_asset_subroot_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **scene_config_file** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

Return type:
None

class DeviceCfg(

*device=device(type='cuda', index=0)*,

*dtype=torch.float32*,

*collision_geometry_dtype=torch.float32*,

*collision_gradient_dtype=torch.float32*,

*collision_distance_dtype=torch.float32*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Configuration for device and data types used in tensor operations.

Parameters:
- **device** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><em>torch.device</em></a>)

- **dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_geometry_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_gradient_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_distance_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

device: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)">torch.device</a> = device(type='cuda', index=0)

dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a> = torch.float32

collision_geometry_dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a> = torch.float32

collision_gradient_dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a> = torch.float32

collision_distance_dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a> = torch.float32

static from_basic(*device*, *dev_id*)
Parameters:
- **device** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **dev_id** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

to_device(*data_tensor*)

to_int8_device(*data_tensor*)

cpu()

as_torch_dict()

is_same_torch_device(*other*)
Check if a torch.device refers to the same physical device as this config.

Handles the case where “cuda” and “cuda:0” are equivalent devices.

Parameters:
**other** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">device</code></a>) – torch.device to compare against.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

Returns:
True if devices refer to the same physical device.

\_\_init\_\_(

*device=device(type='cuda', index=0)*,

*dtype=torch.float32*,

*collision_geometry_dtype=torch.float32*,

*collision_gradient_dtype=torch.float32*,

*collision_distance_dtype=torch.float32*,

)

Parameters:
- **device** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device" class="reference external" title="(in PyTorch v2.13)"><em>torch.device</em></a>)

- **dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_geometry_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_gradient_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

- **collision_distance_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

Return type:
None
