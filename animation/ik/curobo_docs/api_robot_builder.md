<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.robot_builder.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.robot_builder module

Public API for building robot model configurations.

This module provides a simple interface for creating cuRobo robot configurations from URDF files. It re-exports the main builder and debugger classes from the internal implementation.

Example

Building a new robot model:

    >>> from curobo.robot_builder import RobotBuilder
    >>>
    >>> builder = RobotBuilder("robot.urdf", "assets/")
    >>> builder.fit_collision_spheres()
    >>> builder.compute_collision_matrix()
    >>> config = builder.build()
    >>> builder.save(config, "my_robot.yml")

Editing an existing robot model:

    >>> builder = RobotBuilder.from_config("franka.yml")
    >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
    >>> config = builder.build()
    >>> builder.save(config, "franka_updated.yml")

Debugging collision issues:

    >>> from curobo.robot_builder import RobotDebugger
    >>>
    >>> # Load from YAML
    >>> debugger = RobotDebugger("robot.yml")
    >>>
    >>> # Or load from XRDF
    >>> debugger = RobotDebugger.from_xrdf("robot.xrdf", "robot.urdf", "assets/")
    >>>
    >>> result = debugger.check_retract_collision()
    >>> if result["has_collision"]:
    ...     print(f"Found {result['num_colliding_pairs']} colliding pairs")

Exporting to XRDF format:

    >>> config = builder.build()
    >>> builder.save(config, "robot.yml")  # cuRobo YAML format
    >>> builder.save_xrdf(config, "robot.xrdf")  # XRDF format for Isaac Sim

class RobotBuilder(

*urdf_path*,

*asset_path=''*,

*tool_frames=None*,

*device_cfg=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Build and edit robot model configurations from URDF files.

This class provides a high-level interface for creating cuRobo robot configurations. It handles:

- Loading URDF files and parsing kinematic structure

- Fitting collision spheres to robot link meshes

- Computing optimized self-collision ignore matrices

- Generating and saving cuRobo-compatible configuration files

- Interactive visualization with Viser

The builder supports both creating new configurations from URDF and editing existing configurations (e.g., refitting spheres for specific links).

Typical workflow for creating new robot:
1.  Initialize with URDF path

2.  Fit collision spheres

3.  Compute collision matrix

4.  Build configuration

5.  Save to file

Typical workflow for editing existing config:
1.  Load with <a href="#curobo.robot_builder.RobotBuilder.from_config" class="reference internal" title="curobo.robot_builder.RobotBuilder.from_config"><code class="sourceCode python">from_config</code></a>

2.  Modify (e.g., <a href="#curobo.robot_builder.RobotBuilder.refit_link_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.refit_link_spheres"><code class="sourceCode python">refit_link_spheres</code></a>)

3.  Build configuration

4.  Save to file

Example

Creating new robot model:

    >>> from curobo._src.robot.builder.builder_robot import RobotBuilder
    >>>
    >>> # Create builder
    >>> builder = RobotBuilder("robot.urdf", "assets/")
    >>>
    >>> # Fit collision spheres to all links
    >>> builder.fit_collision_spheres(sphere_density=2.0)
    >>>
    >>> # Compute and optimize collision matrix
    >>> builder.compute_collision_matrix(num_samples=2000)
    >>>
    >>> # Build final configuration
    >>> config = builder.build()
    >>>
    >>> # Save to file
    >>> builder.save(config, "my_robot.yml")

Editing existing robot model:

    >>> # Load existing config
    >>> builder = RobotBuilder.from_config("franka.yml")
    >>>
    >>> # Refit spheres for gripper with higher density
    >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
    >>>
    >>> # Add manual collision ignore
    >>> builder.add_collision_ignore("panda_link1", ["panda_link5"])
    >>>
    >>> # Rebuild and save
    >>> config = builder.build()
    >>> builder.save(config, "franka_updated.yml")

Initialize robot model builder from URDF file.

Parameters:
- **urdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to URDF file.

- **asset_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to mesh assets directory (for resolving relative mesh paths in URDF).

- **base_link** – Robot base link name. If None, auto-detected from URDF root.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration for CUDA operations. Defaults to CUDA:0.

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

\_\_init\_\_(

*urdf_path*,

*asset_path=''*,

*tool_frames=None*,

*device_cfg=None*,

)

Initialize robot model builder from URDF file.

Parameters:
- **urdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to URDF file.

- **asset_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to mesh assets directory (for resolving relative mesh paths in URDF).

- **base_link** – Robot base link name. If None, auto-detected from URDF root.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration for CUDA operations. Defaults to CUDA:0.

- **tool_frames** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*\]* *\|* *None*)

classmethod from_config(

*config_path*,

*device_cfg=None*,

)

Load existing robot configuration for editing.

This classmethod allows loading an existing .yml configuration file and creating a builder instance that can be used to modify the configuration (e.g., refit spheres, update collision ignore matrix).

Parameters:
- **config_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to existing robot configuration .yml file.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration. Defaults to CUDA:0.

Return type:
<a href="#curobo.robot_builder.RobotBuilder" class="reference internal" title="curobo._src.robot.builder.builder_robot.RobotBuilder"><code class="sourceCode python">RobotBuilder</code></a>

Returns:
RobotBuilder instance initialized with existing configuration.

Raises:
- <a href="https://docs.python.org/3/library/exceptions.html#FileNotFoundError" class="reference external" title="(in Python v3.14)"><strong>FileNotFoundError</strong></a> – If config_path doesn’t exist.

- <a href="https://docs.python.org/3/library/exceptions.html#KeyError" class="reference external" title="(in Python v3.14)"><strong>KeyError</strong></a> – If config file is missing required fields.

Example

    >>> builder = RobotBuilder.from_config("franka.yml")
    >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
    >>> config = builder.build()
    >>> builder.save(config, "franka_updated.yml")

fit_collision_spheres(

*sphere_density=1.0*,

*surface_radius=0.002*,

*fit_type=SphereFitType.MORPHIT*,

*use_collision_mesh=False*,

*iterations=200*,

*coverage_weight=None*,

*protrusion_weight=None*,

*compute_metrics=False*,

*clip_links=None*,

)

Fit collision spheres to all robot links with collision meshes.

Loads each link’s mesh, preprocesses it, and fits spheres to approximate its volume. The number of spheres per link scales automatically with the link’s bounding-box volume and *sphere_density*.

Parameters:
- **sphere_density** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Dimensionless density multiplier. `1.0` (default) gives a balanced count; `2.0` doubles it; `0.5` halves it. Practical range: `0.1` – `10.0`.

- **surface_radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Radius added to surface-sampled spheres. Only affects the `SURFACE` fit type and the surface-sampling fallback.

- **fit_type** (<a href="curobo.sphere_fit.html#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitType"><code class="sourceCode python">SphereFitType</code></a>) – Sphere fitting algorithm. `MORPHIT` (default) usually gives the best results.

- **use_collision_mesh** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – When True, use collision geometry instead of visual geometry.

- **iterations** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Optimization iterations for the `MORPHIT` fit type.

- **coverage_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt coverage loss weight. Higher values force spheres to fill the mesh volume more completely (default 1000.0). Only used when *fit_type* is `MORPHIT`.

- **protrusion_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt protrusion loss weight. Higher values penalise sphere surface area outside the mesh (default 10.0). Only used when *fit_type* is `MORPHIT`.

- **compute_metrics** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – When True, compute and log per-link quality metrics (coverage, protrusion, surface gap) after fitting.

- **clip_links** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Tuple</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]\]\]) – Per-link half-plane clipping constraints. Keys are link names; values are `(axis,`` ``offset)` tuples where *axis* is `"x"`, `"y"`, or `"z"` (prefix with `"-"` for negative direction) and *offset* is the plane position along that axis. Spheres on the specified link will not extend past the plane.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\]\]

Returns:
Dictionary mapping link names to lists of sphere dicts, each with keys `"center"` (3-element list) and `"radius"` (float).

Example

    >>> spheres = builder.fit_collision_spheres()
    >>> spheres = builder.fit_collision_spheres(sphere_density=2.0)
    >>> spheres = builder.fit_collision_spheres(
    ...     coverage_weight=2000.0, protrusion_weight=5.0, compute_metrics=True
    ... )
    >>> spheres = builder.fit_collision_spheres(
    ...     clip_links={"base_link": ("z", 0.0)}
    ... )

refit_link_spheres(

*link_name*,

*num_spheres=None*,

*sphere_density=1.0*,

*surface_radius=0.002*,

*fit_type=SphereFitType.MORPHIT*,

*use_collision_mesh=False*,

*iterations=200*,

*coverage_weight=None*,

*protrusion_weight=None*,

*compute_metrics=False*,

*clip_plane=None*,

)

Refit collision spheres for a single link.

Useful for tuning individual links without refitting the whole robot.

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name of link to refit.

- **num_spheres** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Explicit sphere count for this link. When `None`, estimated automatically using *sphere_density*.

- **sphere_density** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Density multiplier (see <a href="#curobo.robot_builder.RobotBuilder.fit_collision_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.fit_collision_spheres"><code class="sourceCode python">fit_collision_spheres</code></a>).

- **surface_radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Surface-sampling sphere radius.

- **fit_type** (<a href="curobo.sphere_fit.html#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitType"><code class="sourceCode python">SphereFitType</code></a>) – Sphere fitting algorithm.

- **use_collision_mesh** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – When True, use collision geometry.

- **iterations** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Optimization iterations for the `MORPHIT` fit type.

- **coverage_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt coverage loss weight (default 1000.0).

- **protrusion_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt protrusion loss weight (default 10.0).

- **compute_metrics** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – When True, log quality metrics for this link.

- **clip_plane** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#tuple" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">tuple</code></a>\]) – Half-plane constraint `((nx,`` ``ny,`` ``nz),`` ``offset)` in link-local coordinates. When provided, spheres that extend past the plane are penalised during optimization and hard-clamped.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\]

Returns:
List of sphere dicts for this link.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If link has no mesh geometry.

Example

    >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
    >>> builder.refit_link_spheres("panda_link0", num_spheres=5)

compute_collision_matrix(

*prune_collisions=True*,

*num_samples=1000*,

*batch_size=10000*,

*seed=345*,

*custom_ignore=None*,

)

Compute optimized self-collision ignore matrix.

This method creates a dictionary of link pairs that should be ignored for collision checking. The matrix includes:

1.  Neighboring links (connected by joints) - always ignored

2.  Links that collide at default joint configuration - ignored (false positives)

3.  Links that never collide (optional, via sampling) - ignored for efficiency

Parameters:
- **prune_collisions** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Whether to sample random configurations to find never- colliding link pairs. This significantly reduces collision checking overhead but takes longer to compute.

- **num_samples** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of random configurations to sample for pruning. More samples = better pruning but slower computation.

- **batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Batch size for parallel collision checking during sampling.

- **seed** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Random seed for reproducible sampling.

- **custom_ignore** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]\]) –

  Additional collision ignore pairs to add. Dict mapping link names to lists of links to ignore. Example:

      {"link_1": ["link_5", "link_6"]}

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]

Returns:
Dictionary mapping link names to lists of links to ignore for collision. This is the self_collision_ignore matrix used in the robot configuration.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If <a href="#curobo.robot_builder.RobotBuilder.fit_collision_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.fit_collision_spheres"><code class="sourceCode python">fit_collision_spheres</code></a> hasn’t been called yet.

Example

    >>> # Compute with default pruning
    >>> matrix = builder.compute_collision_matrix()
    >>>
    >>> # More aggressive pruning
    >>> matrix = builder.compute_collision_matrix(num_samples=5000)
    >>>
    >>> # Skip pruning for faster computation
    >>> matrix = builder.compute_collision_matrix(prune_collisions=False)
    >>>
    >>> # With custom ignores
    >>> matrix = builder.compute_collision_matrix(
    ...     custom_ignore={"link_1": ["link_5"]}
    ... )

add_collision_ignore(

*link_name*,

*ignore_links*,

)

Add links to collision ignore list.

This manually adds link pairs to the collision ignore matrix. Useful when editing existing configurations or when you know certain links should never be checked for collision.

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Link to add ignores for.

- **ignore_links** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – List of link names to ignore collision with.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Example

    >>> builder.add_collision_ignore("link_1", ["link_5", "link_6"])
    >>> # Now link_1 will not be checked against link_5 or link_6

remove_collision_ignore(

*link_name*,

*ignore_links*,

)

Remove links from collision ignore list.

Parameters:
- **link_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Link to remove ignores from.

- **ignore_links** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – List of link names to stop ignoring.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Example

    >>> builder.remove_collision_ignore("link_1", ["link_5"])
    >>> # Now link_1 WILL be checked against link_5

build()
Build final robot configuration.

This creates a `KinematicsLoaderCfg` from all the computed data (spheres, collision matrix, etc.). This config can then be saved to file or used directly to create a `Kinematics`.

Return type:
`KinematicsLoaderCfg`

Returns:
KinematicsLoaderCfg ready for use with cuRobo.

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If <a href="#curobo.robot_builder.RobotBuilder.fit_collision_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.fit_collision_spheres"><code class="sourceCode python">fit_collision_spheres</code></a> hasn’t been called yet.

Example

    >>> builder.fit_collision_spheres()
    >>> builder.compute_collision_matrix()
    >>> config = builder.build()
    >>> # Use config with Kinematics
    >>> from curobo._src.robot.kinematics.kinematics import Kinematics
    >>> robot_model = Kinematics(config)

save(

*config*,

*output_path*,

*include_cspace=True*,

)

Save robot configuration to YAML file.

Parameters:
- **config** (`KinematicsLoaderCfg`) – Robot configuration to save (from <a href="#curobo.robot_builder.RobotBuilder.build" class="reference internal" title="curobo.robot_builder.RobotBuilder.build"><code class="sourceCode python">build</code></a>).

- **output_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to output .yml file.

- **include_cspace** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Whether to include cspace (joint limits) configuration.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Example

    >>> config = builder.build()
    >>> builder.save(config, "my_robot.yml")

save_xrdf(

*config*,

*output_path*,

*geometry_name='collision_model'*,

)

Save robot configuration as XRDF file.

XRDF (eXtended Robot Description Format) is NVIDIA’s format for robot descriptions with collision spheres and self-collision matrices. It’s useful for sharing robot configurations with Isaac Sim and other NVIDIA tools.

The XRDF format contains the same information as the YAML format but structured according to the XRDF specification.

Parameters:
- **config** (`KinematicsLoaderCfg`) – Robot configuration to save (from <a href="#curobo.robot_builder.RobotBuilder.build" class="reference internal" title="curobo.robot_builder.RobotBuilder.build"><code class="sourceCode python">build</code></a>).

- **output_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to output .xrdf file (YAML format).

- **geometry_name** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Name for the geometry section in XRDF (default: “collision_model”).

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Example

    >>> config = builder.build()
    >>> builder.save_xrdf(config, "my_robot.xrdf")
    >>> # Can also save as YAML
    >>> builder.save(config, "my_robot.yml")

Note

XRDF files use YAML format but with a specific structure defined by the XRDF specification. See NVIDIA’s XRDF documentation for details.

visualize(

*config=None*,

*port=8080*,

*show_meshes=False*,

*show_spheres=True*,

*timeout_sec=-1*,

)

Start interactive visualization of robot configuration with Viser.

This creates a web-based visualization server that shows the robot with its collision spheres. Useful for inspecting sphere fitting quality before saving.

Parameters:
- **config** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[`KinematicsLoaderCfg`\]) – Config to visualize. If None, calls <a href="#curobo.robot_builder.RobotBuilder.build" class="reference internal" title="curobo.robot_builder.RobotBuilder.build"><code class="sourceCode python">build</code></a> first.

- **port** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Viser server port (opens in browser at <a href="http://localhost:port" class="reference external">http://localhost:port</a>).

- **show_meshes** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Whether to show original collision meshes.

- **show_spheres** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – Whether to show fitted collision spheres.

- **timeout_sec** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

Return type:
`ViserVisualizer`

Returns:
ViserVisualizer instance. Keep this reference alive to maintain visualization.

Example

    >>> builder.fit_collision_spheres()
    >>> viser = builder.visualize()
    >>> # Opens browser at http://localhost:8080
    >>> # Adjust sphere parameters if needed
    >>> builder.fit_collision_spheres(sphere_density=2.0)
    >>> viser = builder.visualize()  # Visualize again
    >>> # When satisfied, save
    >>> config = builder.build()
    >>> builder.save(config, "robot.yml")

property tool_frames: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Get list of all robot link names.

property collision_link_names: List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]
Get list of links with fitted collision spheres.

property collision_spheres: Dict\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>, List\[Dict\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>
Get fitted collision spheres (None if not yet fitted).

property collision_matrix: Dict\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>, List\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>\]\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>
Get collision ignore matrix (None if not yet computed).

property num_spheres: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a>
Get total number of collision spheres across all links.

property link_metrics: Dict\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>, <a href="curobo.sphere_fit.html#curobo.sphere_fit.SphereFitMetrics" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitMetrics">curobo._src.geom.sphere_fit.types.SphereFitMetrics</a>\]
Per-link sphere fit quality metrics.

Populated when <a href="#curobo.robot_builder.RobotBuilder.fit_collision_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.fit_collision_spheres"><code class="sourceCode python">fit_collision_spheres</code></a> or <a href="#curobo.robot_builder.RobotBuilder.refit_link_spheres" class="reference internal" title="curobo.robot_builder.RobotBuilder.refit_link_spheres"><code class="sourceCode python">refit_link_spheres</code></a> is called with `compute_metrics=True`.

Returns:
Dictionary mapping link names to `SphereFitMetrics`. Empty if metrics have not been computed.

class RobotDebugger(

*config_path*,

*device_cfg=None*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Debug robot model collision configurations.

This class provides tools for diagnosing self-collision issues in robot configurations. Use it to:

- Check for collisions at specific joint configurations

- Identify problematic link pairs

- Analyze collision statistics across random samples

- Visualize collision states

- Inspect collision matrix properties

Example

Load from cuRobo YAML:

    >>> from curobo._src.robot.builder.debugger_robot import RobotDebugger
    >>>
    >>> # Load robot config
    >>> debugger = RobotDebugger("franka.yml")
    >>>
    >>> # Check default joint configuration
    >>> result = debugger.check_default_joint_configuration_collision()
    >>> if result["has_collision"]:
    ...     print(f"Found {result['num_colliding_pairs']} colliding pairs:")
    ...     for pair in result["colliding_pairs"]:
    ...         print(f"  {pair[0]} <-> {pair[1]}")

Load from XRDF:

    >>> # Load from XRDF format
    >>> debugger = RobotDebugger.from_xrdf(
    ...     "robot.xrdf",
    ...     "robot.urdf",
    ...     "assets/"
    ... )
    >>> result = debugger.check_default_joint_configuration_collision()

Check specific configuration:

    >>> q = [0, 0, 0, -1.5, 0, 1.8, 0]  # Joint angles
    >>> result = debugger.check_collision_at_config(q)
    >>>
    >>> # Print collision matrix statistics
    >>> debugger.print_collision_matrix_stats()

Initialize debugger with cuRobo robot configuration (YAML format).

Parameters:
- **config_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to robot configuration .yml file.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration. Defaults to CUDA:0.

Example

    >>> debugger = RobotDebugger("robot.yml")
    >>> result = debugger.check_default_joint_configuration_collision()

Note

For XRDF files, use <a href="#curobo.robot_builder.RobotDebugger.from_xrdf" class="reference internal" title="curobo.robot_builder.RobotDebugger.from_xrdf"><code class="sourceCode python">from_xrdf</code></a> instead.

\_\_init\_\_(

*config_path*,

*device_cfg=None*,

)

Initialize debugger with cuRobo robot configuration (YAML format).

Parameters:
- **config_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to robot configuration .yml file.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration. Defaults to CUDA:0.

Example

    >>> debugger = RobotDebugger("robot.yml")
    >>> result = debugger.check_default_joint_configuration_collision()

Note

For XRDF files, use <a href="#curobo.robot_builder.RobotDebugger.from_xrdf" class="reference internal" title="curobo.robot_builder.RobotDebugger.from_xrdf"><code class="sourceCode python">from_xrdf</code></a> instead.

classmethod from_xrdf(

*xrdf_path*,

*urdf_path=None*,

*asset_path=''*,

*device_cfg=None*,

)

Initialize debugger from XRDF robot configuration.

XRDF (eXtended Robot Description Format) is NVIDIA’s format for robot descriptions. This method converts XRDF to cuRobo format and creates a debugger instance.

Parameters:
- **xrdf_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to XRDF file.

- **urdf_path** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]) – Path to URDF file (required by XRDF). If None, attempts to find it automatically from XRDF content.

- **asset_path** (<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>) – Path to mesh assets directory.

- **device_cfg** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><code class="sourceCode python">DeviceCfg</code></a>\]) – Device configuration. Defaults to CUDA:0.

Return type:
<a href="#curobo.robot_builder.RobotDebugger" class="reference internal" title="curobo._src.robot.builder.debugger_robot.RobotDebugger"><code class="sourceCode python">RobotDebugger</code></a>

Returns:
RobotDebugger instance initialized from XRDF.

Example

    >>> debugger = RobotDebugger.from_xrdf(
    ...     "robot.xrdf",
    ...     "robot.urdf",
    ...     "assets/"
    ... )
    >>> result = debugger.check_default_joint_configuration_collision()

Note

XRDF support in the debugger provides full collision checking functionality. All debugging methods are available when loading from XRDF.

check_default_joint_configuration_collision()
Check for self-collisions at default joint configuration.

The default joint configuration is typically a “home” or “neutral” pose where the robot should NOT have any self-collisions. If collisions are found, they likely indicate incorrectly fitted spheres or missing collision ignore pairs.

Returns:
- has_collision (bool): Whether any collisions were detected

- num_colliding_pairs (int): Number of colliding link pairs

- colliding_pairs (List\[Tuple\[str, str\]\]): List of colliding link name pairs

- max_penetration (float): Maximum penetration depth

- distances (Dict\[Tuple\[str, str\], float\]): Penetration depth per link pair

Return type:
Dictionary with collision information

Example

    >>> result = debugger.check_default_joint_configuration_collision()
    >>> if result["has_collision"]:
    ...     print(f"WARNING: {result['num_colliding_pairs']} pairs colliding!")
    ...     print(f"Max penetration: {result['max_penetration']:.4f}m")
    ...     for (link1, link2), dist in result["distances"].items():
    ...         print(f"  {link1} <-> {link2}: {dist:.4f}m")

check_collision_at_config(

*joint_position*,

)

Check self-collisions at a specific joint configuration.

Parameters:
**joint_position** (`Union`\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\], <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><code class="sourceCode python">ndarray</code></a>, <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Joint angles. Can be list, numpy array, or torch tensor. Must match robot’s DOF.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Dict</code></a>

Returns:
Dictionary with detailed collision information (same format as <a href="#curobo.robot_builder.RobotDebugger.check_default_joint_configuration_collision" class="reference internal" title="curobo.robot_builder.RobotDebugger.check_default_joint_configuration_collision"><code class="sourceCode python">check_default_joint_configuration_collision</code></a>).

Raises:
<a href="https://docs.python.org/3/library/exceptions.html#ValueError" class="reference external" title="(in Python v3.14)"><strong>ValueError</strong></a> – If joint_position length doesn’t match robot DOF.

Example

    >>> # Check specific configuration
    >>> q = [0, -0.5, 0, -1.5, 0, 1.0, 0.785]
    >>> result = debugger.check_collision_at_config(q)
    >>> print(f"Collisions: {result['has_collision']}")

sample_collision_checks(

*num_samples=1000*,

*batch_size=100*,

*seed=42*,

)

Sample random configurations and check for collisions.

This helps identify which link pairs collide most frequently, which can indicate problematic sphere fitting or missing collision ignore pairs.

Parameters:
- **num_samples** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Total number of configurations to sample.

- **batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Batch size for parallel collision checking.

- **seed** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Random seed for reproducibility.

Returns:
- total_samples (int): Number of configurations sampled

- collision_count (int): Number of configs with collisions

- collision_rate (float): Percentage of configs with collisions

- frequent_collisions (List\[Tuple\[Tuple\[str, str\], int\]\]):
  List of (link_pair, count) sorted by frequency

Return type:
Dictionary with collision statistics

Example

    >>> stats = debugger.sample_collision_checks(num_samples=5000)
    >>> print(f"Collision rate: {stats['collision_rate']:.1f}%")
    >>> print("Most frequent collisions:")
    >>> for (link1, link2), count in stats["frequent_collisions"][:10]:
    ...     print(f"  {link1} <-> {link2}: {count} times")

find_never_colliding_pairs(

*num_samples=10000*,

*batch_size=10000*,

*seed=345*,

)

Find link pairs that never collide in sampled configurations.

These pairs can potentially be added to the collision ignore list to improve collision checking performance.

Parameters:
- **num_samples** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Number of configurations to sample.

- **batch_size** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Batch size for collision checking.

- **seed** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Random seed.

Return type:
<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Tuple</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>, <a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">str</code></a>\]\]

Returns:
List of link name pairs that never collided across all samples.

Example

    >>> never_colliding = debugger.find_never_colliding_pairs(num_samples=20000)
    >>> print(f"Found {len(never_colliding)} pairs that never collide:")
    >>> for link1, link2 in never_colliding[:10]:
    ...     print(f"  {link1} <-> {link2}")
    >>> # These can be added to collision_ignore in the config

visualize_collision_at_config(

*joint_position*,

*port=8080*,

)

Visualize robot at specific configuration.

Parameters:
- **joint_position** (`Union`\[<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">List</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\], <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><code class="sourceCode python">ndarray</code></a>, <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><code class="sourceCode python">Tensor</code></a>\]) – Joint angles to visualize.

- **port** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Viser server port.

Return type:
`ViserVisualizer`

Returns:
ViserVisualizer instance.

Example

    >>> q = [0, -0.5, 0, -1.5, 0, 1.0, 0.785]
    >>> viser = debugger.visualize_collision_at_config(q)
    >>> # Opens browser visualization at http://localhost:8080

print_collision_matrix_stats()
Print statistics about the collision matrix.

Shows information about total spheres, collision pairs being checked, and percentage of sphere pairs that are ignored.

Return type:
<a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">None</code></a>

Example

    >>> debugger.print_collision_matrix_stats()
    Collision Matrix Statistics:
      Total spheres: 245
      Total possible pairs: 29,890
      Checked pairs: 8,432
      Ignored pairs: 21,458
      Checking: 28.2% of all possible pairs

property robot_config: <a href="curobo.kinematics.html#curobo.kinematics.KinematicsCfg" class="reference internal" title="curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg">curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg</a>
Get the robot configuration being debugged.

property robot_model: <a href="curobo.kinematics.html#curobo.kinematics.Kinematics" class="reference internal" title="curobo._src.robot.kinematics.kinematics.Kinematics">curobo._src.robot.kinematics.kinematics.Kinematics</a>
Get the robot model instance.
