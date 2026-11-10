<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.sphere_fit.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.sphere_fit module

Sphere fitting utilities for approximating mesh geometry.

cuRobo represents robot links and scene obstacles as collections of spheres for fast GPU collision checking. This module exposes the public API for fitting sphere approximations to triangle meshes and inspecting fit quality.

Typical workflow:

1.  Load a triangle mesh (e.g. with <a href="https://trimesh.org/trimesh.html#module-trimesh" class="reference external" title="(in trimesh)"><code class="sourceCode python">trimesh</code></a>).

2.  Optionally call <a href="#curobo.sphere_fit.estimate_sphere_count" class="reference internal" title="curobo.sphere_fit.estimate_sphere_count"><code class="sourceCode python">estimate_sphere_count</code></a> to pick a reasonable sphere count from the mesh bounding volume.

3.  Call <a href="#curobo.sphere_fit.fit_spheres_to_mesh" class="reference internal" title="curobo.sphere_fit.fit_spheres_to_mesh"><code class="sourceCode python">fit_spheres_to_mesh</code></a> with the desired <a href="#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo.sphere_fit.SphereFitType"><code class="sourceCode python">SphereFitType</code></a> strategy.

4.  Inspect the returned <a href="#curobo.sphere_fit.SphereFitResult" class="reference internal" title="curobo.sphere_fit.SphereFitResult"><code class="sourceCode python">SphereFitResult</code></a> for sphere centers, radii, and <a href="#curobo.sphere_fit.SphereFitMetrics" class="reference internal" title="curobo.sphere_fit.SphereFitMetrics"><code class="sourceCode python">SphereFitMetrics</code></a> fit quality.

Example

    import trimesh
    from curobo.sphere_fit import (
        SphereFitType,
        estimate_sphere_count,
        fit_spheres_to_mesh,
    )

    mesh = trimesh.load("link.obj")
    n = estimate_sphere_count(mesh)
    result = fit_spheres_to_mesh(mesh, n_spheres=n, fit_type=SphereFitType.VOXEL)
    centers, radii, metrics = result.centers, result.radii, result.metrics

class SphereFitMetrics(

*num_spheres=0*,

*coverage=0.0*,

*protrusion=0.0*,

*protrusion_dist_mean=0.0*,

*protrusion_dist_p95=0.0*,

*surface_gap_mean=0.0*,

*surface_gap_p95=0.0*,

*max_uncovered_gap=0.0*,

*volume_ratio=0.0*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Quality metrics for a sphere fit to a single mesh/link.

All distance fields are in metres. Fractions are in `[0,`` ``1]`.

Parameters:
- **num_spheres** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **coverage** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion_dist_mean** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion_dist_p95** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **surface_gap_mean** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **surface_gap_p95** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **max_uncovered_gap** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **volume_ratio** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

num_spheres: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a> = 0
Number of spheres fitted to this link.

coverage: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Fraction of interior sample points covered by at least one sphere.

protrusion: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Fraction of sphere-surface sample points outside the mesh.

protrusion_dist_mean: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Mean distance (m) of protruding sphere-surface points to the mesh.

protrusion_dist_p95: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
95th-percentile distance (m) of protruding points to the mesh.

surface_gap_mean: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Mean gap (m) from mesh surface samples to nearest sphere surface.

surface_gap_p95: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
95th-percentile gap (m) from mesh surface samples to nearest sphere surface.

max_uncovered_gap: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Maximum gap (m) from mesh surface samples to nearest sphere surface.

volume_ratio: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> = 0.0
Total sphere volume divided by mesh volume.

\_\_init\_\_(

*num_spheres=0*,

*coverage=0.0*,

*protrusion=0.0*,

*protrusion_dist_mean=0.0*,

*protrusion_dist_p95=0.0*,

*surface_gap_mean=0.0*,

*surface_gap_p95=0.0*,

*max_uncovered_gap=0.0*,

*volume_ratio=0.0*,

)

Parameters:
- **num_spheres** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **coverage** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion_dist_mean** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **protrusion_dist_p95** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **surface_gap_mean** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **surface_gap_p95** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **max_uncovered_gap** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

- **volume_ratio** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a>)

Return type:
None

class SphereFitResult(

*centers*,

*radii*,

*num_spheres=0*,

*metrics=None*,

*fit_time_s=None*,

*used_mesh=None*,

*history=\<factory\>*,

*debug_info=\<factory\>*,

)

Bases: <a href="https://docs.python.org/3/library/functions.html#object" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">object</code></a>

Result of a sphere fitting operation.

Always contains the fitted sphere geometry (`centers`, `radii`). Quality metrics are populated in <a href="#curobo.sphere_fit.SphereFitResult.metrics" class="reference internal" title="curobo.sphere_fit.SphereFitResult.metrics"><code class="sourceCode python">metrics</code></a> only when `compute_metrics=True` is passed to <a href="#curobo.sphere_fit.fit_spheres_to_mesh" class="reference internal" title="curobo.sphere_fit.fit_spheres_to_mesh"><code class="sourceCode python">fit_spheres_to_mesh</code></a>.

Parameters:
- **centers** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **radii** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **num_spheres** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **metrics** (<a href="#curobo.sphere_fit.SphereFitMetrics" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitMetrics"><em>curobo._src.geom.sphere_fit.types.SphereFitMetrics</em></a> *\|* *None*)

- **fit_time_s** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a> *\|* *None*)

- **used_mesh** (<a href="https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh" class="reference external" title="(in trimesh)"><em>trimesh.base.Trimesh</em></a> *\|* *None*)

- **history** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><em>Tuple</em></a>*\[*<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>*,* <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>*\]\]*)

- **debug_info** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><em>Dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><em>Any</em></a>*\]* *\|* *None*)

\_\_init\_\_(

*centers*,

*radii*,

*num_spheres=0*,

*metrics=None*,

*fit_time_s=None*,

*used_mesh=None*,

*history=\<factory\>*,

*debug_info=\<factory\>*,

)

Parameters:
- **centers** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **radii** (<a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)"><em>torch.Tensor</em></a>)

- **num_spheres** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><em>int</em></a>)

- **metrics** (<a href="#curobo.sphere_fit.SphereFitMetrics" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitMetrics"><em>curobo._src.geom.sphere_fit.types.SphereFitMetrics</em></a> *\|* *None*)

- **fit_time_s** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><em>float</em></a> *\|* *None*)

- **used_mesh** (<a href="https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh" class="reference external" title="(in trimesh)"><em>trimesh.base.Trimesh</em></a> *\|* *None*)

- **history** (<a href="https://docs.python.org/3/library/typing.html#typing.List" class="reference external" title="(in Python v3.14)"><em>List</em></a>*\[*<a href="https://docs.python.org/3/library/typing.html#typing.Tuple" class="reference external" title="(in Python v3.14)"><em>Tuple</em></a>*\[*<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>*,* <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)"><em>numpy.ndarray</em></a>*\]\]*)

- **debug_info** (<a href="https://docs.python.org/3/library/typing.html#typing.Dict" class="reference external" title="(in Python v3.14)"><em>Dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)"><em>str</em></a>*,* <a href="https://docs.python.org/3/library/typing.html#typing.Any" class="reference external" title="(in Python v3.14)"><em>Any</em></a>*\]* *\|* *None*)

Return type:
None

centers: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Sphere centre positions, shape `(N,`` ``3)`, float32, CUDA.

radii: <a href="https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor" class="reference external" title="(in PyTorch v2.13)">torch.Tensor</a>
Sphere radii, shape `(N,)`, float32, CUDA.

num_spheres: <a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)">int</a> = 0
Number of spheres.

metrics: <a href="#curobo.sphere_fit.SphereFitMetrics" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitMetrics">curobo._src.geom.sphere_fit.types.SphereFitMetrics</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Quality metrics. `None` until <a href="#curobo.sphere_fit.fit_spheres_to_mesh" class="reference internal" title="curobo.sphere_fit.fit_spheres_to_mesh"><code class="sourceCode python">fit_spheres_to_mesh</code></a> is called with `compute_metrics=True`.

fit_time_s: <a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)">float</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
Wall-clock time (seconds) for the fitting call.

used_mesh: <a href="https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh" class="reference external" title="(in trimesh)">trimesh.base.Trimesh</a> \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a> = None
The mesh that was actually used for fitting. May differ from the input when hollow-mesh detection replaces it with its convex hull.

history: List\[Tuple\[<a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)">numpy.ndarray</a>, <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray" class="reference external" title="(in NumPy v2.5)">numpy.ndarray</a>\]\]
Per-iteration snapshots of `(centers,`` ``radii)` recorded during optimization. Empty for non-iterative methods (SURFACE, VOXEL).

debug_info: Dict\[<a href="https://docs.python.org/3/library/stdtypes.html#str" class="reference external" title="(in Python v3.14)">str</a>, Any\] \| <a href="https://docs.python.org/3/library/constants.html#None" class="reference external" title="(in Python v3.14)">None</a>
Diagnostic metadata from the fitting pipeline. Typical keys:

- `used_convex_hull` (*bool*): convex hull was substituted for the input.

- `auto_n_spheres` (*bool*): sphere count was determined automatically.

- `requested_n_spheres` (*int \| None*): caller’s original `num_spheres`.

- `resolved_n_spheres` (*int*): sphere count actually used.

- `fallback_used` (*bool*): primary method failed, distance-field fallback ran.

- `fit_type` (*str*): the `SphereFitType` value used.

class SphereFitType(*value*)
Bases: <a href="https://docs.python.org/3/library/enum.html#enum.Enum" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">enum.Enum</code></a>

Supported sphere fit types.

See <a href="../reference/sphere_fitting.html#attach-object-note" class="reference internal">Use Cases</a> for more details.

SURFACE = 'surface'
Samples the surface of the mesh evenly, fixed radius. Fast fallback.

VOXEL = 'voxel'
Bbox grid + SDF filtering. Interior voxels with inscribed radii.

MORPHIT = 'morphit'
Voxel-grid seeding + Adam optimization.

estimate_sphere_count(

*mesh*,

*sphere_density=1.0*,

)

Estimate the number of spheres needed to approximate a mesh.

Uses bounding-box volume scaled by a density multiplier, clamped to `[3,`` ``100`` ``*`` ``sphere_density]`. Both the raw count and the per-link cap scale with *sphere_density*, so `density=0.5` halves the maximum and `density=2.0` doubles it.

Parameters:
- **mesh** (<a href="https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh" class="reference external" title="(in trimesh)"><code class="sourceCode python">Trimesh</code></a>) – Input triangle mesh.

- **sphere_density** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Dimensionless density multiplier. `1.0` (default) gives a balanced sphere count; `2.0` doubles it; `0.5` halves it. Practical range is `0.1` to `10.0`.

Return type:
<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>

Returns:
Estimated sphere count.

fit_spheres_to_mesh(

*mesh*,

*num_spheres=None*,

*sphere_density=1.0*,

*surface_radius=0.005*,

*fit_type=SphereFitType.MORPHIT*,

*iterations=200*,

*compute_metrics=False*,

*coverage_weight=None*,

*protrusion_weight=None*,

*clip_plane=None*,

*device_cfg=DeviceCfg(device=device(type='cuda', index=0), dtype=torch.float32, collision_geometry_dtype=torch.float32, collision_gradient_dtype=torch.float32, collision_distance_dtype=torch.float32)*,

)

Approximate a mesh with spheres.

Parameters:
- **mesh** (<a href="https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh" class="reference external" title="(in trimesh)"><code class="sourceCode python">Trimesh</code></a>) – Input mesh.

- **num_spheres** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>\]) – Explicit number of spheres to fit. When `None` (default), estimated automatically using *sphere_density*.

- **sphere_density** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Dimensionless density multiplier used when *num_spheres* is `None`. Scales both the sphere count estimate and the per-link cap. `1.0` (default) gives a balanced count; `2.0` doubles it; `0.5` halves it. Practical range: `0.1` – `10.0`.

- **surface_radius** (<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>) – Radius added to surface-sampled spheres. Only affects the `SURFACE` fit type and the surface-sampling fallback.

- **fit_type** (<a href="#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitType"><code class="sourceCode python">SphereFitType</code></a>) – Fitting algorithm; see <a href="#curobo.sphere_fit.SphereFitType" class="reference internal" title="curobo.sphere_fit.SphereFitType"><code class="sourceCode python">SphereFitType</code></a>.

- **iterations** (<a href="https://docs.python.org/3/library/functions.html#int" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">int</code></a>) – Optimization iterations (only used by `MORPHIT`).

- **compute_metrics** (<a href="https://docs.python.org/3/library/functions.html#bool" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">bool</code></a>) – When True, compute quality metrics (coverage, protrusion, surface gap, volume ratio) on the result.

- **coverage_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt coverage loss weight. Higher values force spheres to fill the mesh volume more completely. Only used by `MORPHIT`. When `None`, uses the default (1000.0).

- **protrusion_weight** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/functions.html#float" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">float</code></a>\]) – MorphIt protrusion loss weight. Higher values penalise sphere surface area outside the mesh more aggressively. Only used by `MORPHIT`. When `None`, uses the default (10.0).

- **clip_plane** (<a href="https://docs.python.org/3/library/typing.html#typing.Optional" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">Optional</code></a>\[<a href="https://docs.python.org/3/library/stdtypes.html#tuple" class="reference external" title="(in Python v3.14)"><code class="sourceCode python">tuple</code></a>\]) – Half-plane constraint `((nx,`` ``ny,`` ``nz),`` ``offset)` in mesh-local coordinates. Spheres that cross the plane are penalised during MorphIt optimization and hard-clamped afterwards. For non-MorphIt fit types, only the hard clamp is applied. `None` (default) disables clipping.

- **device_cfg** (<a href="curobo.types.html#curobo.types.DeviceCfg" class="reference internal" title="curobo._src.types.device_cfg.DeviceCfg"><em>curobo._src.types.device_cfg.DeviceCfg</em></a>)

Return type:
<a href="#curobo.sphere_fit.SphereFitResult" class="reference internal" title="curobo._src.geom.sphere_fit.types.SphereFitResult"><code class="sourceCode python">SphereFitResult</code></a>

Returns:
A <a href="#curobo.sphere_fit.SphereFitResult" class="reference internal" title="curobo.sphere_fit.SphereFitResult"><code class="sourceCode python">SphereFitResult</code></a> with sphere positions, radii, and optionally quality metrics.
