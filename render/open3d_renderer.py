"""Open3D Filament-backed offscreen renderer.

The default backend for `dycraft.render`, and the one every stored artifact was
drawn with. It takes the world `render/_world.py` composed, lights it with a
world-fixed studio rig (shadows OFF), draws black edge linesets over the box
parts, renders the canonical cameras at 1280x720, and writes
`runs/<id>/renders/<step>_<view>.png`.

**This module owns only the Filament half.** Geometry composition, camera
framing, material packing, UV projection, the room cutaway rule, and the render
cache all live in `_world.py`, shared with `blender_renderer.py` — a second
backend that re-derived any of them would disagree with this one about the
scene rather than about its lighting. The shared names the renderer tests
reach for here are re-exported below.

Every surface draws from a `sdk._material.Material`: RGB base color plus
metallic / roughness / alpha, and the PBR maps it carries. A bound map drives
its own scalar to identity because Filament MULTIPLIES the two — see
`_pbr_material` — which is how the schema's "a map replaces its scalar, it
does not tint it" holds here. UVs come from the mesh wherever the mesh has
them; `_world._triplanar_uvs` parameterizes only the geometry that has none.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..sdk import _primitives as _prim
from . import (
    RENDER_HEIGHT as _RENDER_HEIGHT,
    RENDER_WIDTH as _RENDER_WIDTH,
    CameraSpec,
    RenderRequest,
    RenderResult,
    RenderUnavailable,
    render_image_size,          # noqa: F401 — re-exported, see module docstring
)
from ._world import (           # noqa: F401 — all re-exported, see docstring
    _Plan,
    _WorldPart,
    _aabb_from_world_parts,
    _bbox_metrics,
    _build_object_geometry,
    _build_scene_geometry,
    _build_world_geometry,
    _build_world_geometry_at_defaults,
    _catalog_fingerprint,
    _color_255_to_float,
    _floor_only_for,
    _instance_deformed,
    _material_to_render,
    _materialize,
    _prepare,
    _resolve_asset_path,
    _resolve_mesh_params,
    _resolve_target_bbox,
    _PNG_COMPRESS_LEVEL,      # noqa: F401 — re-exported, see _world
    _save_png,
    _triplanar_uvs,
    _world_box_aabb,
    plan_outputs,
    room_wall_boxes,
    ROOM_SURFACE_TAGS as _ROOM_SURFACE_TAGS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# This backend keys the render cache on `request.config_version`, whose default
# is the package-level `RENDER_CONFIG_VERSION` — historically *this* renderer's
# recipe version, and left that way so no stored artifact's key moves. A second
# backend versions itself instead (see `_world.plan_outputs`'s override).

#: W/m² that maps to a fully-saturated `emissive_color`. Filament's slot is an
#: unscaled color, so the authored radiant exitance has to be squashed into
#: [0,1] somewhere; this is that constant. Chosen at the bright end of the
#: range `Emission.strength` documents (a bare bulb), so a screen at single
#: digits stays visibly dimmer than a lamp.
_EMISSIVE_REFERENCE = 50.0

_EDGE_COLOR = (0.0, 0.0, 0.0)
_BG_COLOR = (0.95, 0.95, 0.95, 1.0)
# Indirect-light intensity multiplier. Higher values make metallic
# surfaces (Material.metallic > 0) reflect a brighter environment, so
# their explicit base color reads instead of going near-black. With
# the previous value (0.35) on the `default` IBL, stainless
# (metallic≈0.9, color (195,195,200)) rendered as RGB ~(88,92,102) —
# near-black. With 1.2 on the `brightday` IBL it reads as ~(130,135,150),
# clearly light gray, while diffuse colors stay within ~10% of their
# explicit RGB.
# Lighting rig.
#
# A single camera-locked headlamp (with shadows) leaves every face oblique
# to the current camera falling back to the IBL term, and Filament's IBL
# environment is bluish, so warm-white surfaces read as navy/charcoal
# on whichever faces the headlamp misses. We instead use a WORLD-FIXED
# studio rig with shadows OFF: one strong key for 3D shading, plus a
# low-intensity white "fill dome" — one light along each principal
# horizontal direction and one from above — that approximates a uniform
# white ambient so every vertical face receives near-frontal white light
# and authored colors read truthfully from all sides. The rig is fixed
# across all six canonical cameras. To keep the ambient term neutral (not
# blue) the IBL is also switched to the neutral-gray environment
# (`_IBL_NAME`); `_AMBIENT` scales that now-secondary IBL term and is kept
# modest so the white rig dominates.
#
# `_LIGHT_RIG` entries are (name, world_position, intensity); each light
# travels toward the scene origin (direction = normalize(-position)).
_AMBIENT = 1.0
_IBL_NAME = "default"   # neutral studio-gray; avoids the bluish-sky tint
_LIGHT_RIG = (
    ("key_light",   (5.0, 10.0, 7.0),  40_000.0),  # primary shaper
    ("fill_front",  (0.0, 10.0, 2.0),  10_000.0),  # lights +Y faces
    ("fill_back",   (0.0, -10.0, 2.0), 10_000.0),  # lights -Y faces
    ("fill_left",   (10.0, 0.0, 2.0),  10_000.0),  # lights +X faces
    ("fill_right",  (-10.0, 0.0, 2.0), 10_000.0),  # lights -X faces
    ("fill_top",    (0.0, 0.0, 12.0),  8_000.0),   # lights top faces
)

# Tag prefix used to identify cutaway wall geometry attached to a specific
# camera so it can be cleared between renders.
_CAMERA_WALLS_TAG = "_walls_for_cam"



# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def _draw_plan(renderer, request: RenderRequest, plan: _Plan,
               result: RenderResult) -> None:
    """Step 4 on an ALREADY-OPEN renderer: per camera, cache check, render, or
    hardlink, write PNG.

    Must be called on the Filament thread.

    The cache-key and filename rules are NOT here — `_world.plan_outputs` owns
    them, shared with every other backend, because a second renderer that
    numbered its PNGs or keyed its cache even slightly differently would break
    the byte-identical-artifact-layout invariant in a way no test of either
    backend alone would catch.
    """
    result.warnings.extend(_populate_static(renderer, plan.world_parts))
    for cam, key, out_path, cached_src in plan_outputs(request, plan):
        if cached_src is not None:
            _materialize(cached_src, out_path)
            result.cache_hits[cam.id] = True
        else:
            _set_camera(renderer, cam)
            _set_walls_for_camera(renderer, cam, plan.room_info)
            img = _render_one(renderer)
            _save_png(img, out_path, request.png_compress_level)
            if request.cache is not None:
                request.cache[key] = out_path
            result.cache_hits[cam.id] = False
        result.render_paths[cam.id] = out_path


# Teardown is deliberately NOT factored into a helper. It has to run in the
# frame that holds the renderer:
#
#     _close_renderer(renderer)
#     del renderer          # drop THIS frame's reference…
#     gc.collect()          # …so the cycle is collectable, here, on this thread
#
# A `_release(renderer)` helper looks equivalent and is not: rebinding the
# helper's parameter leaves the caller's name pointing at the renderer, so the
# collect inside the helper cannot free the cycle. It then survives until some
# later automatic collection on whatever thread trips the gc threshold — and
# Filament aborts the process when that is not the owning thread
# ("Engine::shutdown() called from the wrong thread!"). The duplication below
# is the price of keeping the drop and the collect in the same frame.


def render(request: RenderRequest) -> RenderResult:
    """Render the request and return a RenderResult.

    Raises RenderUnavailable if the Filament context cannot be created.
    """
    plan = _prepare(request)
    result = RenderResult(target_bbox=plan.target_bbox, cameras=plan.cameras)

    # Falls back to the module default so a RenderRequest built before
    # `resolution` existed still renders full-size rather than at (0, 0).
    render_w, render_h = (getattr(request, "resolution", None)
                          or (_RENDER_WIDTH, _RENDER_HEIGHT))

    # Every Filament call in the process runs on one dedicated thread — see
    # "Filament thread ownership". The renderer is opened lazily inside
    # `_draw`, so import-time still has no GL requirement, and it is created,
    # used and freed entirely within this one call on that thread.
    def _draw():
        renderer = _open_renderer(render_w, render_h)
        try:
            _draw_plan(renderer, request, plan, result)
        finally:
            _close_renderer(renderer)
            del renderer          # see the note above: must be THIS frame
            gc.collect()

    _on_render_thread(_draw)
    return result


# ---------------------------------------------------------------------------
# World geometry composition
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Target framing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Open3D wrappers
# ---------------------------------------------------------------------------

def _select_egl_platform() -> None:
    """Ensure `EGL_PLATFORM` is set to something Filament can initialize.

    Open3D's `OffscreenRenderer` calls into Filament, which opens an EGL
    display. On Linux Filament's default path tries the X11 / GLVND EGL
    display first; on a headless host with no X server that returns a
    display handle but `eglInitialize` fails, and Filament then segfaults
    inside Open3D's C extension before a Python exception can be raised
    (so `except BaseException` around the constructor cannot catch it).

    Forcing `EGL_PLATFORM=surfaceless` makes Filament use Mesa's surfaceless
    EGL backend, which works without a display server. We only set it when
    the caller hasn't already picked a platform and there's no X/Wayland
    display in the environment — so workstation runs with a real display
    keep their normal GLX/EGL path.
    """
    if os.environ.get("EGL_PLATFORM"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    os.environ["EGL_PLATFORM"] = "surfaceless"


# ---------------------------------------------------------------------------
# Filament thread ownership
# ---------------------------------------------------------------------------
#
# Filament's JobSystem registers the thread that creates an Engine. Calling
# into that Engine from any *other* thread trips
#
#     utils::PreconditionPanic in JobSystem::getState():263
#     reason: This thread has not been adopted.
#
# which is a `terminate()`, not a Python exception — no caller can catch it,
# the whole process dies. The harness daemon services each op on a
# `ThreadPoolExecutor` worker (`agent/server.py`), and the pool hands out
# different threads over the life of a run, so successive renders would reach
# Filament from different threads and abort the process within a few dozen
# `render()` calls, well before an articulated object's `finish` (six views
# for each limit of each joint).
#
# Filament is equally strict about *destruction*:
#
#     in void filament::FEngine::shutdown():271
#     reason: Engine::shutdown() called from the wrong thread!
#
# and an OffscreenRenderer sits in a reference cycle, so dropping the last
# name for it does not free it — the cyclic collector does, later, on
# whichever thread happens to trip the GC threshold. Pinning renders to one
# thread therefore trades the `getState()` abort for a `shutdown()` one,
# which can kill a run a few compiles in. `atexit` cannot
# police this, because `threading._shutdown` runs first and joins the pool
# worker, so no teardown task can be submitted by then.
#
# The fix is to collect it *here*, on the owning thread, before `_draw`
# returns: the renderer then never survives long enough for another thread to
# free it. Keeping construction and destruction inside one `_draw()` call also
# preserves the invariant that every render sees a pristine renderer.
#
# Two alternatives were tried and rejected, both measured:
#   * caching one renderer and reusing it — `scene.clear_geometry()` does not
#     restore a pristine scene, and all five views came out byte-different
#     from a fresh renderer's, which breaks the byte-identical-artifacts
#     invariant;
#   * caching it and pinning its refcount so it is never destroyed — same
#     byte-identity problem, and the process still aborted at exit.

_RENDER_THREAD = None
_RENDER_THREAD_LOCK = threading.Lock()


def _render_thread() -> ThreadPoolExecutor:
    global _RENDER_THREAD
    if _RENDER_THREAD is None:
        with _RENDER_THREAD_LOCK:
            if _RENDER_THREAD is None:
                _RENDER_THREAD = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="dycraft-filament")
    return _RENDER_THREAD


def _on_render_thread(fn, *args, **kwargs):
    """Run `fn` on the process's single Filament-owning thread and block.

    Exceptions (including `RenderUnavailable`) propagate to the caller
    unchanged via `Future.result()`.
    """
    return _render_thread().submit(fn, *args, **kwargs).result()


def _open_renderer(width: int, height: int):
    try:
        import open3d as o3d
    except ImportError as e:
        raise RenderUnavailable(f"open3d not installed: {e}")
    _select_egl_platform()
    # Open3D logs `[Open3D INFO] EGL headless mode enabled.` on every
    # OffscreenRenderer construction. It says nothing that varies between
    # renders, and it reaches the agent through the daemon's stdout capture,
    # so drop INFO. Warnings and errors still print — a real renderer
    # complaint must stay visible.
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except BaseException:  # noqa: BLE001 — older/minimal Open3D builds
        pass
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    except BaseException as e:  # Filament errors propagate as RuntimeError-likes
        raise RenderUnavailable(f"OffscreenRenderer construction failed: {e}")
    renderer.scene.set_background(np.asarray(_BG_COLOR, dtype=np.float32))
    # Image-based lighting: a neutral-gray IBL cubemap (`_IBL_NAME`) gives
    # PBR metallics (Material.metallic > 0) something to reflect so they
    # don't render near-black (a pure metal's visible color is the
    # Fresnel-weighted BRDF specular lookup into the environment map). We
    # use the neutral `default` environment rather than the bluish
    # `brightday` so the ambient term does not tint warm-white surfaces
    # navy; the world-fixed white directional rig (below) supplies most of
    # the illumination, so the IBL is secondary. The IBL is loaded without
    # a skybox so the flat `_BG_COLOR` backdrop is preserved for
    # documentation-style renders. IBL is purely a renderer-side config;
    # the Material/IR contract is backend-agnostic, so a future Blender
    # backend will handle environment lighting through its own World
    # shader independently.
    resources_dir = os.path.join(os.path.dirname(o3d.__file__), "resources")
    for ibl_name in (_IBL_NAME, "default", "brightday"):
        try:
            renderer.scene.scene.set_indirect_light(
                os.path.join(resources_dir, ibl_name))
            renderer.scene.scene.set_indirect_light_intensity(
                _AMBIENT * 30_000.0)
            break
        except BaseException:
            continue

    # World-fixed studio rig (key + white fill dome, shadows off). Added
    # once here so it persists across all camera binds.
    _install_fixed_lights(renderer)
    return renderer


def _normalize(v):
    n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5 or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _install_fixed_lights(renderer) -> None:
    """Install the world-fixed studio rig (key + white fill dome).

    Each light travels from its world position toward the scene origin
    (direction = normalize(-position)). Shadows are disabled so oblique
    faces are not dropped into darkness. Falls back silently on older
    Open3D builds that lack add_directional_light.

    The rig is installed on a *fresh* `OffscreenRenderer` (single call site,
    in `_open_renderer`), so there is nothing to remove first. A defensive
    `remove_light` would log `[Open3D WARNING] Light <name> is not in the
    scene graph` for each light, which no `except` suppresses because Open3D
    logs rather than raises, and those lines go to Python's `sys.stdout`,
    which the harness daemon ships back in the tool result: they would land in
    the agent's context.
    """
    failed: list[str] = []
    for name, pos, intensity in _LIGHT_RIG:
        direction = _normalize((-pos[0], -pos[1], -pos[2]))
        try:
            renderer.scene.scene.add_directional_light(
                name,
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.array(direction, dtype=np.float32),
                intensity,
                False,   # cast_shadows
            )
        except BaseException as e:  # noqa: BLE001 — see below; never fatal
            # Lighting is not cosmetic here. A rig that silently fails to attach
            # produces a dark frame that reads as a modeling defect rather than
            # as a renderer problem. Losing one light of the rig is survivable, so
            # this still does not raise — but it must not be invisible either.
            failed.append(f"{name} ({type(e).__name__}: {e})")
    if failed:
        print(f"[dycraft] render lighting incomplete: {len(failed)} of "
              f"{len(_LIGHT_RIG)} rig lights failed to attach "
              f"[{'; '.join(failed)}]; the frame will be darker than intended",
              file=sys.stderr)


def _close_renderer(renderer) -> None:
    """Reset the reused renderer between `render()` calls.

    Called on the Filament thread (see "Filament thread ownership"); the
    renderer itself is then released when `_draw`'s local goes out of scope,
    still on that thread.
    """
    try:
        renderer.scene.clear_geometry()
    except BaseException:
        pass


def _part_mesh(p: _WorldPart, color, alpha=1.0):
    """Build the Open3D mesh for a world part; return ``(mesh, problem)``.

    Three tags reach here — `box`, `solid`, and `imported` (`_primitives.py`
    registers no others). Box reuses `_box_mesh` (byte-identical to the prior
    renderer); the two asset tags load their staged OBJ.

    UV provenance follows one rule, applied at the bottom of this function:
    whatever UVs the mesh already carries WIN. A staged OBJ brings its own —
    baked from the B-rep for a solid, authored into the asset for an imported
    one — and the deforming branch below re-reads them from the rest file. Only
    geometry that has none is projected, which here means the AABB fallback
    (and, via `_box_mesh`, an analytic box part). Either way the projection is
    computed in the part's LOCAL frame, before the world transform.

    `problem` is None on the normal path and a sentence otherwise. It exists
    because both fallbacks below draw a **brick where a shaped part belongs**.
    A render is not just a picture here: it is the evidence a reviewer grades
    and the artifact a human looks at. Silently, an unreadable asset would give
    a plausible-looking image of the wrong geometry, a reviewer would rightly
    say the part looks like a box, and the agent would edit geometry that was
    never the problem. Reporting it is the difference between "the asset did
    not load" and "your model is wrong".

    Still a fallback rather than a raise: one missing asset must not cost every
    other part in the frame. See `_populate_static`, which collects these onto
    `RenderResult.warnings`.
    """
    material = p.material or {}
    texture = _usable_texture(material.get("texture"))
    # The tile size, for the one consumer that still needs it. A part loaded
    # from a staged OBJ has its UVs already and never reaches the projection;
    # what does is the analytic box below and the AABB fallback further down.
    scale_m = (material.get("mapping") or {}).get("scale_m")
    prim = getattr(p, "primitive", "box")
    if prim == "box":
        return _box_mesh(p.size, p.se3, color, alpha, texture=texture,
                         scale_m=scale_m), None

    import open3d as o3d
    if prim in _prim.ASSET_PRIMITIVES and getattr(p, "deformed", None) is not None:
        # Deformed geometry supplied by the animation layer, in place of the
        # asset's rest vertices. Neither array comes from the staged file:
        # topology is unchanged by LBS, so the caller already has both and a
        # second parse would only risk the two disagreeing. (The file IS read
        # further down, for its UVs alone — see below.)
        #
        # Normals are deliberately NOT carried over from the asset. A solid's
        # OBJ ships crease-split normals for its CAD edges, and those describe
        # the *rest* surface; reusing them on a bent panel lights it as though
        # it were still flat. Leaving them empty makes the recompute below
        # derive them from the geometry actually being drawn.
        verts, tris = p.deformed
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)),
        )
        # **UVs are the one thing that does have to be re-read.** LBS moves
        # vertices and leaves topology alone, so the rest mesh's per-corner UVs
        # stay valid — but they live in the OBJ this branch otherwise skips, and
        # a mesh built from bare arrays has none at all. That is true of both
        # provenances now: an imported asset's authored atlas coordinates and a
        # CAD solid's B-rep-baked ones are equally absent here, and equally
        # recoverable from the staged file. Filament then samples a
        # single texel for every corner, so a textured skin renders in one flat
        # color: on `CesiumMan` the atlas corner is white, which came out as a
        # uniformly gray figure while the rigid import of the same asset showed
        # its markings correctly. Nothing errors, and the part is still lit and
        # shaded, so it reads as "this asset has no texture".
        # Routed through `texture`, the already-validated result of
        # `_usable_texture` computed at the top of this function, rather than
        # re-reading the raw dict: every decision that depends on a texture has
        # to reach the same answer, and that predicate is what makes the
        # missing-image fallback complete rather than half-applied.
        if texture is not None:
            try:
                rest = o3d.io.read_triangle_mesh(
                    str(p.params[_prim.render_asset_key(prim)]))
                uv = np.asarray(rest.triangle_uvs)
                # Per-corner, so exactly 3 per triangle. A mismatch means the
                # staged file and the deformed arrays disagree about topology —
                # draw untextured rather than pair UVs with the wrong corners.
                if len(uv) == 3 * len(tris):
                    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
            except BaseException:   # noqa: BLE001 — geometry outranks its texture
                pass
    elif prim in _prim.ASSET_PRIMITIVES:
        # An asset-backed part: a build123d-authored solid's tessellated OBJ.
        # Load from file, scale in the mesh-local frame, then apply the world
        # SE3. A solid carries no `scale` by construction, so that step is a
        # no-op for it.
        asset = p.params.get(_prim.render_asset_key(prim))
        try:
            mesh = o3d.io.read_triangle_mesh(str(asset))
            if mesh.is_empty():
                raise ValueError("empty mesh")
            scale = p.params.get("scale", (1.0, 1.0, 1.0))
            if tuple(scale) != (1.0, 1.0, 1.0):
                v = np.asarray(mesh.vertices) * np.asarray(scale, dtype=np.float64)
                mesh.vertices = o3d.utility.Vector3dVector(v)
                # Scaling does not carry normals with it — they transform by the
                # inverse-transpose, so a non-uniform scale invalidates them
                # outright. Drop whatever the file supplied and let the
                # recompute below derive them from the geometry actually being
                # drawn. (A uniform scale would have preserved them, but only
                # `solid` parts ship normals today and they never carry a scale,
                # so distinguishing the two cases would buy nothing.)
                mesh.vertex_normals = o3d.utility.Vector3dVector()
        except Exception as e:  # noqa: BLE001 — one asset must not sink the frame
            # Missing/broken asset: fall back to the mesh's AABB box, placed at
            # its true (possibly off-center) local AABB center — and say so.
            return (
                _box_mesh(p.size, p.se3, color, alpha, center=p.center,
                          texture=texture, scale_m=scale_m),
                f"part {p.name!r}: its {prim} asset could not be loaded "
                f"({type(e).__name__}: {e}); asset={str(asset)!r}. The part was "
                f"drawn as its bounding BOX, so this image shows a brick where "
                f"the real geometry belongs — judge the asset path, not the "
                f"authored shape.",
            )
    else:
        # Unknown primitive: fall back to its bounding box, and report it. A tag
        # added to `_primitives` but not taught to this renderer would otherwise
        # render every such part as a brick with no indication that a whole
        # geometry kind is unsupported here.
        return (
            _box_mesh(p.size, p.se3, color, alpha, texture=texture,
                      scale_m=scale_m),
            f"part {p.name!r}: primitive {prim!r} is not one this renderer "
            f"draws, so it was rendered as its bounding box.",
        )

    # Still in the part's local frame here — see the docstring.
    #
    # A part loaded from a staged OBJ keeps the UVs that came with it: the
    # guard inside `_apply_texture_uvs` returns before projecting anything. The
    # deforming branch above recovers the rest mesh's own coordinates before
    # reaching here, so it too arrives with UVs and the guard returns early.
    # What still reaches the projection from this point is the AABB fallback,
    # whose box has none of its own.
    _apply_texture_uvs(mesh, texture, scale_m)
    t, q = p.se3
    R = _quat_to_matrix(q)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = t
    mesh.transform(transform)
    if not mesh.has_vertex_normals():
        # Only when the asset brought none. `compute_vertex_normals()` averages
        # over every face at a vertex, and our meshes are welded, so calling it
        # unconditionally would smear the exact surface normals a solid's OBJ
        # ships with and round off every CAD edge. `transform` already rotated
        # them, which is exact here — the transform is a rigid motion.
        mesh.compute_vertex_normals()
    mesh.paint_uniform_color(_paint_color(color, texture))
    return mesh, None


def _box_mesh(size, se3, color, alpha=1.0, center=(0.0, 0.0, 0.0),
              texture=None, scale_m=None):
    import open3d as o3d
    w, l, h = size
    cx, cy, cz = center
    mesh = o3d.geometry.TriangleMesh.create_box(w, l, h)
    # create_box places the box with one corner at origin; translate so it is
    # centered at `center` in the part-local frame. `center` is (0,0,0) for the
    # origin-centered primitives (recovering the prior behavior byte-for-byte)
    # and the AABB midpoint for a mesh's fallback box.
    mesh.translate((cx - w / 2, cy - l / 2, cz - h / 2))
    # Before the world transform below: UVs are derived from the part's own
    # frame so the pattern is fixed to the part, not to where it stands.
    #
    # This box has no staged mesh — an analytic part, an asset's AABB fallback,
    # or a room surface synthesized per camera — so there are no authored UVs
    # to keep and the projection is the only thing that can parameterize it.
    _apply_texture_uvs(mesh, texture, scale_m)
    t, q = se3
    R = _quat_to_matrix(q)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = t
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(_paint_color(color, texture))
    return mesh


def _lineset(pts_local, lines, se3):
    """Build a black-edge LineSet from local points + line indices, placed
    by `se3`."""
    import open3d as o3d
    t, q = se3
    R = _quat_to_matrix(q)
    pts = np.asarray(pts_local, dtype=np.float64) @ R.T + np.array(t)
    lines = np.asarray(lines, dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(_EDGE_COLOR, dtype=np.float64), (len(lines), 1))
    )
    return ls


def _box_edges(size, se3):
    w, l, h = size
    pts_local = np.array([
        [-w / 2, -l / 2, -h / 2], [w / 2, -l / 2, -h / 2],
        [-w / 2,  l / 2, -h / 2], [w / 2,  l / 2, -h / 2],
        [-w / 2, -l / 2,  h / 2], [w / 2, -l / 2,  h / 2],
        [-w / 2,  l / 2,  h / 2], [w / 2,  l / 2,  h / 2],
    ], dtype=np.float64)
    lines = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)
    return _lineset(pts_local, lines, se3)


def _part_edges(p):
    """Edge overlay matched to the part's primitive shape (box -> AABB edges)."""
    prim = getattr(p, "primitive", "box")
    if prim in _prim.ASSET_PRIMITIVES:
        # A CAD solid has no canonical primitive silhouette, and a box cage
        # around its AABB actively lies about it — the whole point of a solid is
        # that its shape is NOT its bounding box (a shelled carcass fills 16% of
        # it). Draw no overlay. Membership test rather than `== "solid"` so a
        # new asset tag cannot silently fall through to the box branch.
        return None
    return _box_edges(p.size, p.se3)


def _quat_to_matrix(q):
    """wxyz quaternion → 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    # Assume already normalized; SDK guarantees this.
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


# Successfully decoded textures, keyed by `(content digest, decode mode)` — see
# `_load_texture` for why the mode is part of the key. A miss costs a file read
# plus a PIL decode; the committed 1024x1024 library image measures ~25 ms and a
# 4K source ~280 ms, against ~15 ms for the actual GL draw. The same texture is
# bound once per camera (six canonical views per render) and once per
# frame of a clip, so this is load-bearing rather than an optimization. Keyed on
# the digest, not the path, because a scene render resolves the same staged
# image under a different catalog root.
_TEXTURE_CACHE: dict = {}

# Decode attempts that FAILED, keyed by `(path, digest, mode)`. Separate from
# the hit cache, and keyed by path, because a failure is not a property of the
# content the digest names — it is a property of where we looked. See
# `_load_texture`.
_FAILED_TEXTURES: set = set()

#: The map kinds this backend knows how to bind, and the decode mode each
#: needs. Filament's vocabulary, not dycraft's — a kind absent from here rides
#: through every other layer untouched and is simply not sampled, which is the
#: right degradation: the material still renders, on the scalar the map would
#: have replaced. `channel` is carried in the IR for a packed source but is not
#: honored yet; a packed map binds whole and is wrong in a visible way rather
#: than a silent one, so it is refused below instead.
BASE_COLOR = "base_color"
EMISSION = "emission"
_MAP_MODES = {BASE_COLOR: "RGB", "roughness": "L", "metallic": "L",
              "emission": "RGB"}


def _load_texture(path, digest, mode="RGB"):
    """Decoded `o3d.geometry.Image` for a staged texture, or None.

    `mode` is the PIL mode to convert to: "RGB" for base color, "L" for a
    single-channel map like roughness. Roughness is genuinely one scalar per
    texel and Filament reads it as such; widening it to three channels would
    triple the upload for no information.

    Fail-soft, matching `_part_mesh`'s posture for a missing OBJ: a texture
    that cannot be read leaves the part on its flat `color` rather than taking
    the render down. The negative result is cached too — a broken path should
    cost one decode attempt per process, not one per camera per frame.
    """
    # Keyed on `(digest, mode)`, not the digest alone. A grayscale image is a
    # perfectly good roughness map AND a perfectly good metallic one, so two
    # kinds sharing content — which content addressing makes one file — would
    # otherwise collide, and the first decode would hand its channel count to
    # the second. The uniform map dict makes that more likely, not less.
    key = (digest, mode)
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]
    # A FAILURE is remembered against the PATH as well, and that asymmetry is
    # the point. Success is a property of the content, so it keys on the digest
    # and is shared across roots — which is the whole reason for digest keying.
    # Failure is a property of where we looked: the same staged image resolves
    # under a different root in a scene render than in an object one (see
    # `_material_to_render`'s `base_dir`, whose omission once resolved every
    # path against the process CWD). Keyed by digest alone, one bad root would
    # poison every later, correct lookup of that image in the process and draw
    # it untextured with nothing saying so. A genuinely broken path still costs
    # one decode attempt per process, which is what the negative cache is for.
    fail_key = (str(path), digest, mode)
    if fail_key in _FAILED_TEXTURES:
        return None
    img = None
    try:
        import open3d as o3d
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im.convert(mode))
        img = o3d.geometry.Image(np.ascontiguousarray(arr))
    except Exception:
        img = None
    if img is None:
        _FAILED_TEXTURES.add(fail_key)
    else:
        _TEXTURE_CACHE[key] = img
    return img


def _usable_map(texture, kind):
    """The decoded image for one map kind, or None if absent or undecodable.

    THE per-kind question, so every consumer reaches the same answer about the
    same map. Per KIND and not per texture, because the schema's contract is
    that a map replaces its OWN scalar fallback (`sdk/_material.py::Texture`):
    a roughness map is a valid description of a surface whose base-color image
    happens to be missing, and refusing it because a different file failed
    would discard a map that decoded perfectly well. `_load_texture` caches
    misses as well as hits, so asking repeatedly costs one attempt per process.
    """
    m = (texture or {}).get(kind)
    if not m:
        return None
    return _load_texture(m["path"], m["digest"],
                         mode=_MAP_MODES.get(kind, "L"))


def _usable_texture(texture):
    """`texture` if ANY of its maps decodes, else None.

    Whether this surface is textured AT ALL, which is exactly the question
    `_apply_texture_uvs` needs: a bound roughness map samples through the same
    UVs as an albedo, so geometry with no UVs of its own needs them generated
    whichever kind survived.

    **The vertex color must NOT follow from this**, and that is the one thing
    to keep straight here. Painting white is right only where an albedo is
    actually bound — otherwise the part renders WHITER than the fallback it was
    meant to degrade to — so `_paint_color` asks `_usable_map(..., BASE_COLOR)`
    specifically rather than reusing this answer.
    """
    if not texture:
        return None
    for kind in texture:
        if _usable_map(texture, kind) is not None:
            return texture
    return None


def _paint_color(color, texture):
    """The vertex color to paint, given the material's authored color.

    White under a usable BASE COLOR map, for the same reason `_pbr_material`
    binds white: Filament multiplies its albedo sample by the vertex color AS
    WELL AS by `base_color`, so painting the authored brown would darken the
    wood by its own color a second time. Both multiplicands go to white
    together, or the taught contract — `color` is a fallback, not a tint — is
    false.

    Keyed on that map alone, not on `_usable_texture`: a surface carrying only
    a usable roughness map still draws its authored color, and whitening it
    there would throw the color away for a map that cannot supply one.
    """
    return ((1.0, 1.0, 1.0)
            if _usable_map(texture, BASE_COLOR) is not None else color)


def _apply_texture_uvs(mesh, texture, scale_m):
    """Derive projected UVs for `mesh`, in whatever frame it is currently in.

    Three no-ops, and the second is the load-bearing one.

    No texture, or no tile size: nothing to parameterize, so an untextured
    render is unaffected (`tests/test_texture_render.py`).

    **The mesh already has UVs: they win.** A staged OBJ carries its own
    per-corner coordinates — baked from the B-rep's faces for a CAD solid, or
    authored into the asset for an imported one — and those describe the
    surface far better than a projection can: they wrap a barrel as a cylinder
    instead of cutting it into three planar patches with two seams, and they
    place an atlas where its author meant it rather than where a world axis
    falls. Overwriting them was silent, because a re-projected surface is still
    plausibly textured; you only notice by knowing what it should have looked
    like.

    So renderer-side projection is what happens when there is nothing else, and
    that is now exactly the geometry with no staged asset to have baked into:
    analytic box parts, the AABB fallback for an asset that would not load, and
    the room's floor and walls. `TriangleMesh.create_box` emits no UVs, so
    those reach the projection through this same guard rather than around it.

    `scale_m` is passed rather than read off the texture, because it is not a
    property of any image — it is `material.mapping`, how the surface was laid
    out, and for a staged asset it was already consumed when the UVs were
    baked. This is the one call that still needs it.
    """
    if texture is None or scale_m is None:
        return
    if mesh.has_triangle_uvs():
        return
    import open3d as o3d
    uvs = _triplanar_uvs(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         scale_m)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)


def _pbr_material(material, *, alpha_override=None):
    """Build a Filament MaterialRecord from a renderer-side material dict.

    `material` is the dict produced by `_material_to_render` (color floats
    plus metallic/roughness/alpha). `alpha_override`, when set, replaces
    the material's alpha — used for the per-camera side cutaway walls.
    """
    import open3d as o3d
    mat = o3d.visualization.rendering.MaterialRecord()
    alpha = material["alpha"] if alpha_override is None else alpha_override
    mat.shader = "defaultLitTransparency" if alpha < 1.0 else "defaultLit"
    r, g, b = material["color"]
    mat.base_color = [r, g, b, alpha]
    mat.base_roughness = material["roughness"]
    mat.base_metallic = material["metallic"]
    emission = material.get("emission")
    if emission is not None:
        # **An emissive surface is drawn UNLIT, not with `emissive_color`.**
        #
        # `MaterialRecord.emissive_color` exists in Open3D's Python API and this
        # is the trap: `defaultLit.filamat` declares `baseColor`,
        # `baseMetallic`, `baseRoughness`, `reflectance`, `clearCoat`, and
        # `anisotropy` — and no emissive uniform at all. Setting the field
        # succeeds, changes nothing, and reports nothing. Measured: zero pixels
        # differed between an emissive and a non-emissive render.
        #
        # `defaultUnlit` is the honest equivalent: it draws `base_color` at full
        # brightness irrespective of scene lighting, which is exactly what a
        # rasterizer can say about a glowing surface. The cost is that the part
        # loses its shading and reads flat — correct for a bulb, a simplification
        # for a shade that both reflects and glows.
        #
        # **It still does not illuminate anything.** A rasterizer gathers no
        # light from geometry, where Cycles treats the same material as a real
        # light source. `sdk/_material.py::Emission` documents the asymmetry; do
        # not compensate by brightening the neighborhood.
        er, eg, eb = emission["color"]
        k = min(emission["strength"] / _EMISSIVE_REFERENCE, 1.0)
        # Toward white as strength rises, the way an overdriven emitter reads on
        # any display: hue at low strength, blown out at high.
        mat.shader = ("defaultUnlitTransparency" if alpha < 1.0
                      else "defaultUnlit")
        mat.base_color = [er + (1.0 - er) * k,
                          eg + (1.0 - eg) * k,
                          eb + (1.0 - eb) * k, alpha]
        # An unlit shader samples no PBR maps, so the texture work below would
        # be dead weight; the albedo map is still bound because unlit does
        # sample that.
        img = _usable_map(_usable_texture(material.get("texture")) or {},
                          EMISSION)
        if img is not None:
            mat.albedo_img = img
        return mat

    tex = _usable_texture(material.get("texture"))
    if tex is not None:
        img = _usable_map(tex, BASE_COLOR)
        if img is not None:
            # **A map REPLACES its scalar; it does not tint it** — and honoring
            # that means binding the IDENTITY factor, not ignoring the scalar.
            # Filament MULTIPLIES the sampled albedo by `base_color`: measured,
            # the same wood over a blue base renders gray, and over the
            # library's own brown (150,105,70) at 0.59/0.41/0.27 of its authored
            # value. A material's `color` is the FALLBACK for consumers with no
            # map support, set to an approximation of the image, so leaving it
            # in the product darkens the wood by its own color twice over.
            #
            # White here makes the rendered surface the image as authored, and
            # makes the contract the agent is taught literally true. Alpha is
            # kept, so a translucent textured surface still selects the
            # transparency shader above.
            mat.base_color = [1.0, 1.0, 1.0, alpha]
            mat.albedo_img = img

        # The same rule, once per remaining kind. Filament multiplies
        # `roughness_img` by `base_roughness` exactly as it multiplies the
        # albedo — measured: a uniform 0.5 map under base_roughness 0.05 and
        # 0.95 rendered at mean 92.6 and 164.0, which it could not do if the map
        # replaced the scalar. So each bound map drives its scalar to identity,
        # while the authored value stays what every consumer WITHOUT the map
        # uses.
        #
        # Outside the albedo branch, deliberately: each map replaces its own
        # scalar, so a roughness map that decodes is correct whether or not the
        # base-color image did. Nesting it made one file's decode failure
        # silently discard a different, working map.
        for kind, slot, factor in (
                ("roughness", "roughness_img", "base_roughness"),
                ("metallic", "metallic_img", "base_metallic")):
            m = tex.get(kind)
            if not m or m.get("channel"):
                # A packed source names the channel it lives in, and this
                # backend cannot select one — binding it whole would sample a
                # roughness map's red as roughness. Left unbound, so the surface
                # falls back to its scalar rather than to a confidently wrong
                # number.
                continue
            mimg = _usable_map(tex, kind)
            if mimg is not None:
                setattr(mat, factor, 1.0)
                setattr(mat, slot, mimg)
    return mat


def _line_material():
    import open3d as o3d
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = 1.5
    return mat


def _populate_static(renderer, world_parts):
    """Add all object/scene part geometry; return the problems worth reporting.

    Walls are added per-camera in `_set_walls_for_camera`. The returned list is
    the per-part substitutions `_part_mesh` had to make — an unreadable asset or
    an unknown primitive drawn as a bounding box. It rides out on
    `RenderResult.warnings` so a caller can tell "this render shows the model"
    from "this render shows a brick standing in for the model".
    """
    problems: list[str] = []
    for i, p in enumerate(world_parts):
        mesh, problem = _part_mesh(p, p.material["color"], p.material["alpha"])
        if problem is not None:
            problems.append(problem)
        renderer.scene.add_geometry(
            f"part_{i}", mesh, _pbr_material(p.material)
        )
        # Wireframe overlay matched to the primitive shape: a box gets AABB
        # edges, an asset part gets the edges of its own local AABB.
        edges = _part_edges(p)
        if edges is not None:
            renderer.scene.add_geometry(
                f"part_{i}_edges", edges, _line_material()
            )
    return problems


def _set_walls_for_camera(renderer, camera: CameraSpec, room_info):
    """Clear any prior walls and add the cutaway walls for `camera`.

    Which surfaces those are is `_world.room_wall_boxes`' answer, not this
    function's — the cutaway describes the scene, and both backends have to
    stage the same room. All this
    adds is the Filament half: box meshes, tags, and the removal of last
    camera's walls.
    """
    # Remove prior walls. The tag list comes from `_world.ROOM_SURFACE_TAGS`,
    # the same table `room_wall_boxes` emits from, because a hand-maintained
    # copy here went stale against the six-surface rewrite and left five of the
    # six surfaces permanently in the scene — see that constant for what it
    # cost. Removing a tag that was never added is free; the try/except below
    # already covers it, and covering the whole vocabulary is the point.
    for tag in _ROOM_SURFACE_TAGS:
        try:
            renderer.scene.remove_geometry(f"{_CAMERA_WALLS_TAG}_{tag}")
        except BaseException:
            pass

    # The room is synthesized per camera from the room bounds — there is no
    # staged mesh and so no authored UVs, which is why the tile size travels
    # this far. It is one of the three cases renderer-side projection exists
    # for, alongside an analytic box part and the AABB fallback for an asset
    # that would not load; `_apply_texture_uvs` holds the full list and the
    # rule that decides between them.
    for wall in room_wall_boxes(camera, room_info):
        # **Skipped rather than hidden, and this backend loses something by
        # it.** `camera_visible` means "exists, but the camera may not see it":
        # under Cycles the surface still bounces, so an interior is lit inside
        # a closed room. Filament has no per-object camera-ray flag, so the
        # only way to keep it out of frame here is not to build it — which is
        # exactly what this backend did to every hidden surface before, so
        # nothing regresses. It just does not gain the near-wall and ceiling
        # bounce that the Blender backend now gets.
        if not wall.camera_visible:
            continue
        mat = wall.material
        mesh = _box_mesh(
            wall.size, wall.se3, mat["color"],
            alpha=(mat["alpha"] if wall.alpha is None else wall.alpha),
            texture=_usable_texture(mat.get("texture")),
            scale_m=(mat.get("mapping") or {}).get("scale_m"))
        renderer.scene.add_geometry(
            f"{_CAMERA_WALLS_TAG}_{wall.tag}", mesh,
            _pbr_material(mat, alpha_override=wall.alpha))


def _set_camera(renderer, cam: CameraSpec):
    """Bind the camera. Lighting is world-fixed (see _install_fixed_lights)
    and does not change per camera."""
    eye = np.asarray(cam.eye, dtype=np.float32)
    target = np.asarray(cam.target, dtype=np.float32)
    up = np.asarray(cam.up, dtype=np.float32)
    # `near_clip`/`far_clip` default to -1, which Open3D documents as "derive
    # them from the bounding box of the current geometry". That is the right
    # answer for a camera framed on the whole model and the wrong one for a
    # tight close-up, where the derived near plane can sit in front of the
    # subject and return an empty frame — so a CameraSpec that computed its own
    # gets to use them. See `render.CameraSpec.near`.
    renderer.setup_camera(cam.fov_deg, target, eye, up,
                          cam.near, cam.far)


def _render_one(renderer) -> np.ndarray:
    img = renderer.render_to_image()
    # Open3D returns an Image; convert via np.asarray.
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr.copy()


# ---------------------------------------------------------------------------
# Catalog fingerprinting (for scene-layer cache keys)
# ---------------------------------------------------------------------------
