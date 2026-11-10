<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.scene.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.scene module

Scene representation module.

This module provides classes for building and managing scene representations with various obstacle types.

Example

[\`\`](#id1)[\`](#id3)python from curobo.scene import Scene, Cuboid, Mesh, Sphere

\# Build scene programmatically scene = Scene(

>
>
> cuboid=\[
> Cuboid(name=”table”, dims=\[1.0, 1.0, 0.1\], pose=\[0, 0, 0.5, 1, 0, 0, 0\]), Cuboid(name=”wall”, dims=\[0.1, 2.0, 2.0\], pose=\[1.0, 0, 1.0, 1, 0, 0, 0\]),
>
> \], sphere=\[
>
> >
> >
> > Sphere(name=”ball”, radius=0.1, pose=\[0.5, 0.5, 1.0, 1, 0, 0, 0\])
> >
> >
>
> \], mesh=\[
>
> >
> >
> > Mesh(name=”object”, file_path=”model.obj”, pose=\[0, 0, 0.6, 1, 0, 0, 0\])
> >
> >
>
> \]
>
>

)

\# Or load from file/dict scene = Scene.create(scene_dict)

\# Modify scene scene.add_obstacle(Sphere(name=”new_ball”, radius=0.2, pose=\[…\])) scene.remove_obstacle(“old_ball”) [\`\`](#id5)[\`](#id7)

Scene
alias of `curobo._src.geom.types.SceneCfg`

class SceneData(

*cuboids=None*,

*meshes=None*,

*voxels=None*,

*num_envs=1*,

*device_cfg=\<factory\>*,

*scene_model=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Aggregate GPU tensor storage for all scene obstacle types.

This dataclass holds references to individual data storage for each obstacle type (cuboids, meshes, voxels). It provides a unified interface for managing scene obstacles while delegating to the appropriate type-specific storage.

Use this as the primary container for scene data. It can be passed to collision checkers, TSDF integrators, or visualizers.

Example

    >>> scene = SceneData.from_scene_cfg(scene_cfg, device_cfg)
    >>> scene.add_obstacle(Cuboid(name="box", pose=[0,0,0,1,0,0,0], dims=[0.1,0.1,0.1]))
    >>> scene.update_obstacle_pose("box", new_pose)

Parameters:
- **cuboids** (*curobo.\_src.geom.data.data_cuboid.CuboidData* *\|* *None*)

- **meshes** (*curobo.\_src.geom.data.data_mesh.MeshData* *\|* *None*)

- **voxels** (*curobo.\_src.geom.data.data_voxel.VoxelData* *\|* *None*)

- **num_envs** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **scene_model** (*curobo.\_src.geom.types.SceneCfg* *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[curobo.\_src.geom.types.SceneCfg\]* *\|* *None*)

cuboids: curobo.\_src.geom.data.data_cuboid.CuboidData \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Cuboid (OBB) tensor storage. None if no cuboids configured.

meshes: curobo.\_src.geom.data.data_mesh.MeshData \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Mesh tensor storage with Warp BVH. None if no meshes configured.

voxels: curobo.\_src.geom.data.data_voxel.VoxelData \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Voxel grid (ESDF) tensor storage. None if no voxels configured.

num_envs: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a> = 1
Number of parallel environments.

device_cfg: <a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a>
Device configuration for tensor creation.

scene_model: curobo.\_src.geom.types.SceneCfg \| List\[curobo.\_src.geom.types.SceneCfg\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
CPU reference to scene configuration (for reference updates).

get_valid_data()
Return initialized type-specific obstacle storage.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[`Union`\[`CuboidData`, `MeshData`, `VoxelData`\]\]

classmethod create_cache(

*num_envs*,

*device_cfg*,

*cuboid_cache=None*,

*mesh_cache=None*,

*mesh_max_dist=0.1*,

*voxel_cache=None*,

)

Create an empty cache for scene obstacles.

Parameters:
- **num_envs** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of parallel environments.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device and dtype configuration.

- **cuboid_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Maximum number of cuboids per environment. None to skip.

- **mesh_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Maximum number of meshes per environment. None to skip.

- **mesh_max_dist** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Maximum query distance for mesh SDF.

- **voxel_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">dict</code></a>\]) – Voxel cache configuration dict with keys: - layers: Maximum number of voxel grids - dims: Grid dimensions \[x, y, z\] in meters - voxel_size: Size of each voxel in meters

Return type:
<a href="#curobo.scene.SceneData" class="reference internal" title="curobo._src.geom.data.data_scene.SceneData"><code class="sourceCode python">SceneData</code></a>

Returns:
Empty SceneData with allocated GPU buffers.

classmethod from_scene_cfg(

*scene_cfg*,

*device_cfg*,

*num_envs=1*,

*env_idx=0*,

*cuboid_cache=None*,

*mesh_cache=None*,

*mesh_max_dist=0.1*,

*voxel_cache=None*,

)

Create SceneData from a scene configuration.

Parameters:
- **scene_cfg** (`SceneCfg`) – Scene configuration containing obstacles.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device and dtype configuration.

- **num_envs** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of parallel environments.

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index to load the obstacles into.

- **cuboid_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Max cuboids. If None, uses len(scene_cfg.cuboid).

- **mesh_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Max meshes. If None, uses len(scene_cfg.mesh).

- **mesh_max_dist** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Maximum query distance for mesh SDF.

- **voxel_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">dict</code></a>\]) – Voxel cache config. If None, derives from scene_cfg.voxel.

Return type:
<a href="#curobo.scene.SceneData" class="reference internal" title="curobo._src.geom.data.data_scene.SceneData"><code class="sourceCode python">SceneData</code></a>

Returns:
SceneData populated with obstacles from the scene config.

classmethod from_batch_scene_cfg(

*scene_cfg_list*,

*device_cfg*,

*cuboid_cache=None*,

*mesh_cache=None*,

*mesh_max_dist=0.1*,

*voxel_cache=None*,

)

Create SceneData from a list of scene configurations.

Each scene config is loaded into a separate environment.

Parameters:
- **scene_cfg_list** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[`SceneCfg`\]) – List of scene configurations, one per environment.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device and dtype configuration.

- **cuboid_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Max cuboids per env. If None, uses max across scenes.

- **mesh_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Max meshes per env. If None, uses max across scenes.

- **mesh_max_dist** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Maximum query distance for mesh SDF.

- **voxel_cache** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">dict</code></a>\]) – Voxel cache config.

Return type:
<a href="#curobo.scene.SceneData" class="reference internal" title="curobo._src.geom.data.data_scene.SceneData"><code class="sourceCode python">SceneData</code></a>

Returns:
SceneData populated with obstacles from all scene configs.

add_obstacle(

*obstacle*,

*env_idx=0*,

)

Add an obstacle to the scene.

Automatically routes to the appropriate storage based on obstacle type.

Parameters:
- **obstacle** (<a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">Obstacle</code></a>) – Obstacle to add (Cuboid, Mesh, or VoxelGrid).

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index to add to.

Return type:
<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>

Returns:
Index of the added obstacle within its type.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If obstacle type is not supported or cache not initialized.

update_obstacle_pose(

*name*,

*pose*,

*env_idx=0*,

)

Update the pose of an existing obstacle by name.

Searches across all obstacle types to find the obstacle.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of the obstacle to update.

- **pose** (<a href="curobo.types.html#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>) – New pose in world frame.

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If obstacle not found in any type.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

enable_obstacle(

*name*,

*enabled=True*,

*env_idx=0*,

)

Enable or disable an obstacle for collision checking.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of the obstacle.

- **enabled** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – True to enable, False to disable.

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If obstacle not found in any type.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

get_obstacle_names(*env_idx=0*)
Get names of all obstacles in an environment.

Parameters:
**env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]

Returns:
List of obstacle names across all types.

check_obstacle_exists(

*name*,

*env_idx=0*,

)

Check if an obstacle exists in the scene.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

clear(*env_idx=None*)
Clear all obstacles from one or all environments.

Parameters:
**env_idx** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Environment to clear. If None, clears all environments.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

load_from_scene_cfg(

*scene_cfg*,

*env_idx=0*,

*store_reference=True*,

)

Load obstacles from a scene configuration into existing storage.

Parameters:
- **scene_cfg** (`SceneCfg`) – Scene configuration containing obstacles to load.

- **env_idx** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Environment index to load obstacles into.

- **store_reference** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – If True, stores scene_cfg in scene_model.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

has_cuboids()
Check if cuboid data is available.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

has_meshes()
Check if mesh data is available.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

has_voxels()
Check if voxel data is available.

Return type:
<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>

get_active_types()
Get a dictionary of active obstacle types.

Return type:
<a href="https://docs.python.org/3/library/stdtypes.html#dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">dict</code></a>

Returns:
Dict with keys ‘cuboid’, ‘mesh’, ‘voxel’ and boolean values.

to_warp(*mesh_max_dist=None*)
Convert to Warp struct for kernel launches.

Parameters:
**mesh_max_dist** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – Override max query distance for meshes.

Return type:
`SceneDataWarp`

Returns:
SceneDataWarp struct for use in Warp kernels.

\_\_init\_\_(

*cuboids=None*,

*meshes=None*,

*voxels=None*,

*num_envs=1*,

*device_cfg=\<factory\>*,

*scene_model=None*,

)

Parameters:
- **cuboids** (*curobo.\_src.geom.data.data_cuboid.CuboidData* *\|* *None*)

- **meshes** (*curobo.\_src.geom.data.data_mesh.MeshData* *\|* *None*)

- **voxels** (*curobo.\_src.geom.data.data_voxel.VoxelData* *\|* *None*)

- **num_envs** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **scene_model** (*curobo.\_src.geom.types.SceneCfg* *\|* <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[curobo.\_src.geom.types.SceneCfg\]* *\|* *None*)

Return type:
None

class Obstacle(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Base class for all obstacles.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

name: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>
Unique name of obstacle.

pose: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Pose of obstacle as a list with format \[x y z qw qx qy qz\]

scale: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Scale obsctacle. This is only implemented for <a href="#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh"><code class="sourceCode python">Mesh</code></a> and `PointCloud` obstacles.

color: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Color of obstacle to use in visualization.

texture_id: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Texture name for the obstacle.

texture: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Texture to apply to obstacle in visualization.

material: curobo.\_src.geom.types.Material
Material properties to apply in visualization.

device_cfg: <a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg">curobo._src.types.device_cfg.DeviceCfg</a>
Device and floating point precision to use for tensors.

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>*,* *optional*) – process when loading from file. Defaults to True.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#NotImplementedError" class="reference external" title="(in Python v3.14)"><strong>NotImplementedError</strong></a> – requires implementation in derived class.

Returns:
instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

save_as_mesh(

*file_path*,

*transform_with_pose=False*,

)

Save obstacle as a mesh file.

Parameters:
- **file_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to save mesh file.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Transform obstacle with pose before saving.

get_cuboid()
Get oriented bounding box of obstacle (OBB).

Returns:
returns obstacle as a cuboid.

Return type:
<a href="#curobo.scene.Cuboid" class="reference internal" title="curobo.scene.Cuboid">Cuboid</a>

get_mesh(*process=True*)
Get obstacle as a mesh.

Parameters:
**process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>*,* *optional*) – process mesh from file. Defaults to True.

Returns:
obstacle as a mesh.

Return type:
<a href="#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh">Mesh</a>

get_transform_matrix()
Get homogenous transformation matrix from pose.

Returns:
transformation matrix.

Return type:
np.ndarray

get_sphere(*n=1*)
Compute a sphere that fits in the volume of the object.

Parameters:
**n** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – number of spheres

Return type:
<a href="#curobo.scene.Sphere" class="reference internal" title="curobo._src.geom.types.Sphere"><code class="sourceCode python">Sphere</code></a>

Returns:
spheres

get_bounding_spheres(

*num_spheres=None*,

*surface_radius=0.002*,

*fit_type=SphereFitType.MORPHIT*,

*pre_transform_pose=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Compute spheres that approximate the volume of the object.

Parameters:
- **num_spheres** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Number of spheres. `None` for automatic.

- **surface_radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Radius added to surface-sampled spheres.

- **fit_type** (<a href="curobo.sphere_fit.html#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitType"><code class="sourceCode python">SphereFitType</code></a>) – Sphere fit algorithm.

- **pre_transform_pose** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.Pose" class="reference internal" title="curobo._src.types.pose.Pose"><code class="sourceCode python">Pose</code></a>\]) – Optional pose applied before the object pose.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device configuration.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="#curobo.scene.Sphere" class="reference internal" title="curobo._src.geom.types.Sphere"><code class="sourceCode python">Sphere</code></a>\]

Returns:
List of <a href="#curobo.scene.Sphere" class="reference internal" title="curobo.scene.Sphere"><code class="sourceCode python">Sphere</code></a> instances.

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

Return type:
None

class Cuboid(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*dims=\<factory\>*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

Represent obstacle as a cuboid.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dims** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

dims: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]
Dimensions of cuboid in meters \[x_length, y_length, z_length\].

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
Instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*dims=\<factory\>*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dims** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

Return type:
None

class Sphere(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*position=None*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

Obstacle represented as a sphere.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

radius: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Radius of sphere in meters.

position: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Position is deprecated, use pose instead

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
Instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

get_cuboid()
Get oriented bounding box of obstacle (OBB).

Returns:
returns obstacle as a cuboid.

Return type:
<a href="#curobo.scene.Cuboid" class="reference internal" title="curobo.scene.Cuboid">Cuboid</a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*position=None*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **position** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

Return type:
None

class Capsule(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*base=\<factory\>*,

*tip=\<factory\>*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

Represent obstacle as a capsule.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **base** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **tip** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

radius: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Radius of capsule in meters.

base: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]
Base of capsule in meters \[x, y, z\].

tip: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]
Tip of capsule in meters \[x, y, z\].

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
Instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*base=\<factory\>*,

*tip=\<factory\>*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **base** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **tip** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

Return type:
None

class Cylinder(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*height=0.0*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

Obstacle represented as a cylinder.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **height** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

radius: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Radius of cylinder in meters.

height: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Height of cylinder in meters.

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Flag is not used.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
Instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*radius=0.0*,

*height=0.0*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **height** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

Return type:
None

class Mesh(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*file_path=None*,

*file_string=None*,

*urdf_path=None*,

*vertices=None*,

*faces=None*,

*vertex_colors=None*,

*vertex_normals=None*,

*texture_uvs=None*,

*texture_image=None*,

*face_colors=None*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

Obstacle represented as mesh.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **file_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **file_string** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **urdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **vertices** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **faces** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]* *\|* *None*)

- **vertex_colors** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **vertex_normals** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **texture_uvs** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **texture_image** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **face_colors** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

file_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Path to mesh file.

file_string: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Full mesh as a string, loading from this is not implemented yet.

urdf_path: <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Path to urdf file, does not support loading from this yet.

vertices: List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Vertices of mesh.

faces: List\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Faces of mesh.

vertex_colors: List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Vertex colors of mesh. Should be float in range of \[0, 1\].

vertex_normals: List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Vertex normals of mesh.

texture_uvs: List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\] \| <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)">numpy.ndarray</a> \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Per-vertex UV coordinates for textured visualization/export.

texture_image: <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)">numpy.ndarray</a> \| <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Texture image for `texture_uvs`. Expected shape is `(H,`` ``W,`` ``3)`.

face_colors: List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Face colors of mesh. Should be float in range of \[0, 1\].

classmethod from_polygon_faces(

*name*,

*vertices*,

*faces*,

*face_counts*,

*pose=None*,

*scale=None*,

*color=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

*\*\*kwargs*,

)

Create a mesh from flat polygon face data.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of mesh.

- **vertices** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]) – Mesh vertices.

- **faces** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Flat face index buffer.

- **face_counts** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Number of vertices per face.

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]) – Pose of mesh as \[x y z qw qx qy qz\].

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]) – Scale to apply to mesh vertices.

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]) – Color of mesh for visualization.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device configuration used for triangulation.

- **kwargs** (<a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Any</code></a>) – Additional <a href="#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh"><code class="sourceCode python">Mesh</code></a> fields.

Return type:
<a href="#curobo.scene.Mesh" class="reference internal" title="curobo._src.geom.types.Mesh"><code class="sourceCode python">Mesh</code></a>

Returns:
Mesh with triangle faces.

get_trimesh_mesh(

*process=True*,

*process_color=True*,

*transform_with_pose=False*,

)

Create a trimesh instance from the obstacle representation.

Parameters:
- **process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – process flag passed to `trimesh.load`.

- **process_color** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – if True, load mesh visual from texture.

- **transform_with_pose** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><em>bool</em></a>)

Returns:
Instance of obstacle as a trimesh.

Return type:
<a href="https://trimesh.org/trimesh.html#trimesh.Trimesh" class="reference external" title="(in trimesh)">trimesh.Trimesh</a>

to_cpu()
Convert the mesh to CPU memory.

to_gpu()
Convert the mesh to GPU memory.

update_material()
Load material into vertex_colors and face_colors.

get_mesh_data(*process=True*)
Get vertices and faces of mesh.

Parameters:
**process** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – process flag passed to `trimesh.load`.

Returns:
vertices and faces of mesh.

Return type:
Tuple\[List\[List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]\], List\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>\]\]

static from_pointcloud(

*pointcloud*,

*pitch=0.02*,

*name='world_pc'*,

*pose=\[0, 0, 0, 1, 0, 0, 0\]*,

*filter_close_points=0.0*,

)

Create a mesh from a pointcloud via voxelized surface extraction.

Voxelizes the point cloud into a 3D grid and emits quad faces on voxel boundaries (occupied vs. empty), producing a watertight mesh without requiring scikit-image or marching cubes.

Parameters:
- **pointcloud** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><code class="sourceCode python">ndarray</code></a>) – Input pointcloud of shape \[n_pts, 3\].

- **pitch** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Voxel size in meters.

- **name** – Name to assign to created mesh.

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – Pose to assign to created mesh.

- **filter_close_points** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Filter points closer than this distance to the origin. Threshold is in meters and should be positive.

Returns:
Mesh created from pointcloud.

Return type:
<a href="#curobo.scene.Mesh" class="reference internal" title="curobo.scene.Mesh">Mesh</a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*file_path=None*,

*file_string=None*,

*urdf_path=None*,

*vertices=None*,

*faces=None*,

*vertex_colors=None*,

*vertex_normals=None*,

*texture_uvs=None*,

*texture_image=None*,

*face_colors=None*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **file_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **file_string** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **urdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **vertices** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **faces** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>*\]* *\|* *None*)

- **vertex_colors** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **vertex_normals** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

- **texture_uvs** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **texture_image** (<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a> *\|* <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **face_colors** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]\]* *\|* *None*)

Return type:
None

class VoxelGrid(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*dims=\<factory\>*,

*voxel_size=0.02*,

*feature_tensor=None*,

*xyzr_tensor=None*,

*feature_dtype=torch.float32*,

)

Bases: <a href="#curobo.scene.Obstacle" class="reference internal" title="curobo._src.geom.types.Obstacle"><code class="sourceCode python">curobo._src.geom.types.Obstacle</code></a>

VoxelGrid representation of an obstacle. Requires voxel to contain ESDF.

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dims** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **voxel_size** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **feature_tensor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **xyzr_tensor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **feature_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

dims: List\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a>\]
Dimensions of voxel grid in meters \[x_length, y_length, z_length\].

voxel_size: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.02
Voxel size in meters.

feature_tensor: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Feature tensor of voxel grid, typically ESDF.

xyzr_tensor: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
XYZR tensor of voxel grid.

feature_dtype: <a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)">torch.dtype</a> = torch.float32
Data type of feature tensor.

get_grid_shape()
Get shape of voxel grid.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Tuple</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\], <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\], <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]

create_xyzr_tensor(

*transform_to_origin=False*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Create XYZR tensor of voxel grid.

Parameters:
- **transform_to_origin** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Transform points to origin.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>) – Device and floating point precision to use for tensors.

Returns:
XYZR tensor of voxel grid with r referring to voxel size.

Return type:
<a href="#curobo.scene.VoxelGrid.xyzr_tensor" class="reference internal" title="curobo.scene.VoxelGrid.xyzr_tensor">xyzr_tensor</a>

get_occupied_voxels(

*feature_threshold=None*,

)

Get occupied voxels from voxel grid.

Parameters:
**feature_threshold** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – esdf value threshold to consider as occupied.

Return type:
<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>

Returns:
occupied voxels as a tensor of shape \[occupied_voxels\].

clone()
Clone data of voxel grid.

Return type:
<a href="#curobo.scene.VoxelGrid" class="reference internal" title="curobo._src.geom.types.VoxelGrid"><code class="sourceCode python">VoxelGrid</code></a>

\_\_init\_\_(

*name*,

*pose=None*,

*scale=None*,

*color=None*,

*texture_id=None*,

*texture=None*,

*material=\<factory\>*,

*device_cfg=\<factory\>*,

*dims=\<factory\>*,

*voxel_size=0.02*,

*feature_tensor=None*,

*xyzr_tensor=None*,

*feature_dtype=torch.float32*,

)

Parameters:
- **name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>)

- **pose** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **scale** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **color** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]* *\|* *None*)

- **texture_id** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **texture** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a> *\|* *None*)

- **material** (*curobo.\_src.geom.types.Material*)

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

- **dims** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>*\]*)

- **voxel_size** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **feature_tensor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **xyzr_tensor** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a> *\|* *None*)

- **feature_dtype** (<a href="https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype" class="reference external" title="(in PyTorch v2.13)"><em>torch.dtype</em></a>)

Return type:
None
