"""Blender-side half of the `blender` render backend. Runs under the bpy env.

Reads a job spec written by `blender_renderer.py`, builds the scene in Blender,
renders each camera to the PNG path the parent chose, and writes a result file.

**Imports `bpy` and nothing from `dycraft`.** The Blender env is a `bpy` wheel
plus its dependencies; it has no dycraft install and must never need one. Every
decision that requires knowing about the IR — where a part sits, which walls
this camera may see, how a box is parameterized — was already made by the
parent and arrives as plain JSON and `.npy` arrays. What is left here is
Blender vocabulary: meshes, node trees, a camera, and Cycles settings.

Two contracts worth stating, because both are silent when broken:

  * **A map replaces its scalar, it does not tint it.** Binding an Image
    Texture to a Principled socket in Blender genuinely replaces it, so unlike
    Filament there is no identity-factor dance to perform — but the *fallback*
    still matters. A map that will not load must leave the authored scalar in
    place, not leave the socket at Blender's default.
  * **A part that fails is drawn, not skipped.** A missing OBJ, an unreadable
    texture: the picture degrades to the flat color and keeps its geometry,
    matching what the Open3D backend does. A render that silently omits a part
    is worse than one that draws it plainly.
"""

import json
import math
import os
import sys
import traceback
from pathlib import Path

import bpy       # noqa: E402  — the whole point of this interpreter

# This file is launched as a plain script by `blender_renderer._run_child`, so
# `render/` is sys.path[0] and the sibling imports as a top-level module; the
# viewer's serve mode imports it as part of the package instead.
import numpy as np


BASE_COLOR = "base_color"
EMISSION = "emission"
HEIGHT = "height"

#: Bump depth in meters for a `height` map that carries no `strength`. Blender's
#: own `Distance` default is 1.0, i.e. a meter of relief, so this is a floor
#: against an importer-supplied map rather than a tuned value.
_BUMP_DISTANCE_DEFAULT = 0.002

#: Channel index within an RGB(A) source, for a packed map that names one.
_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2, "red": 0, "green": 1, "blue": 2}


def _srgb_to_linear(c):
    """One sRGB component in [0,1] to linear.

    **Not cosmetic, and not optional.** A `Material.color` is authored as an
    sRGB byte triple — it is what the agent picked in a color picker and what
    the schema's fallback contract is stated in. Blender's shader sockets are
    LINEAR, so assigning the sRGB value straight in renders every surface too
    bright and too pale: a warm tan base came out cream, a dark blue-gray frame
    came out light gray. Open3D needs no such step because Filament's
    `base_color` is sRGB, which is why this asymmetry is easy to miss — both
    backends read the identical `color` field and only one of them is wrong.
    """
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

#: When true, a failing `@shader` aborts the render instead of falling back to
#: its scalars. Set per job from `strict_materials`; see `MaterialError`.
_STRICT_MATERIALS = False


def _job_is_v2(job) -> bool:
    """Is this job a Blender-native (v2-or-later) scene?

    `schema_v2` is written per frame by `blender_renderer._frame_spec`, because
    that is where the IR is in scope; the scene reset needs the same answer one
    level up. Read from both rather than duplicating the flag at job level,
    which would be a second thing to keep in step.

    The flag's name predates v3 and means **native generation**, not the exact
    tag `dycraft.scene.v2` — `is_native_scene_schema` decides it. Left spelled
    `schema_v2` because it is a cross-process frame-protocol key; do not
    re-derive it here from a tag string.
    """
    if job.get("schema_v2"):
        return True
    return any(bool(f.get("schema_v2")) for f in (job.get("frames") or []))


def _reset_scene(job):
    global _STRICT_MATERIALS
    # **A v2 scene never falls back.** `strict_materials` is the per-caller
    # request (review and delivery set it); v2 makes it unconditional, preview
    # included. A v2 material *is* its node graph, so a scalar stand-in is not a
    # degraded version of it — it is a different surface, and no image of it is
    # evidence about the one that was authored. v1 keeps the historical
    # fallback, because a v1 material legitimately has scalars to fall back to.
    _STRICT_MATERIALS = bool(job.get("strict_materials")) or _job_is_v2(job)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = job.get("engine", "CYCLES")
    scene.render.resolution_x = job["width"]
    scene.render.resolution_y = job["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    level = job.get("png_compress_level")
    if level is not None:
        # Blender's `compression` is a percentage, zlib's is 0-9. The parent
        # speaks zlib (it is Pillow's knob on the other backend), so map it
        # here rather than making the caller know which renderer it is talking
        # to. Clip frames pass a low level for the same reason they do there:
        # encoding otherwise costs more than the picture.
        scene.render.image_settings.compression = int(
            max(0, min(9, level)) / 9.0 * 100)

    # **AgX — Blender's own default — chosen over Standard.**
    #
    # Standard is a hard clip at linear 1.0, and a scene lit by an emissive body
    # blows straight through it: around a small emissive sphere the peak
    # radiance is tens of times the display ceiling, so everything for a few
    # radii collapses to one flat 255. The ball stops looking like a ball and
    # the pool around it gets a hard rim that is nothing but the clipping
    # contour. AgX rolls the highlights off instead: the white disc shrinks to
    # the ball itself, few pixels saturate, and the frame keeps most of its
    # mean.
    #
    # **What this costs.** AgX desaturates, unevenly and moderately: warm
    # colors and blues keep nearly all their saturation, greens and teals lose
    # about a third, and grays come down about 12%. So `Material.color` survives
    # well in the warm half of the wheel and noticeably less well in the
    # cyan-green half, worth knowing when a scene's identity rests on a teal or
    # a green.
    #
    # The other cost is real: the flat backdrop does not land on exactly the
    # gray Open3D produces, so the two backends are not comparable on
    # background pixels.
    #
    # **v2 scenes set none of this here.** A `@lighting` block owns the view
    # transform, the look, exposure, and white balance, and it runs after this
    # function; pre-setting them would be harmless but pre-setting the World
    # would not, because `_setup_world`'s studio dome is light the author did
    # not ask for. So for v2 the World is left for `_LightingFacade` to build
    # blank and the color management is left at Blender's own defaults, which
    # the block then sets. `_read_view_settings` reports whatever is actually
    # in force, so the review record never claims a setting the author did not
    # choose.
    if not _job_is_v2(job):
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "None"

    if scene.render.engine == "CYCLES":
        _setup_cycles(scene, job)

    if not _job_is_v2(job):
        _setup_world(scene, job)
    return scene


def _setup_cycles(scene, job):
    cycles = scene.cycles
    cycles.samples = job.get("samples", 64)
    cycles.use_adaptive_sampling = True
    # `samples` is only a ceiling: adaptive sampling stops a pixel once its
    # noise estimate falls under this threshold, so raising the ceiling without
    # lowering the threshold buys much less than it looks like it should.
    # Absent (the default for every caller that predates the delivery renderer)
    # leaves Blender's factory 0.01 untouched.
    _adaptive = job.get("adaptive_threshold")
    if _adaptive is not None:
        cycles.adaptive_threshold = float(_adaptive)
    # `use_denoising` and the denoiser choice are set together in
    # `_select_denoiser`, once the compute device is known — the right
    # denoiser depends on it, and getting that pairing wrong is the single
    # most expensive mistake available in this file.

    # Transparent film would make the flat backdrop disappear; the documentation
    # look this pipeline has always had is an opaque light-gray field.
    scene.render.film_transparent = False

    device = job.get("device", "OPTIX")
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None or device == "CPU":
        cycles.device = "CPU"
        _select_denoiser(cycles, "CPU")
        return
    # OPTIX first, then CUDA, then CPU. Not a preference — a 3090 renders these
    # scenes in seconds on OptiX and in minutes on CPU, and a silent fallback to
    # CPU is the difference between a usable backend and one nobody will wait
    # for. So the fallback is announced on stderr rather than taken quietly.
    for backend in (device, "CUDA", "NONE"):
        if backend == "NONE":
            cycles.device = "CPU"
            _select_denoiser(cycles, "CPU")
            print("[blender] no GPU compute device; rendering on CPU",
                  file=sys.stderr)
            return
        try:
            prefs.preferences.compute_device_type = backend
            prefs.preferences.get_devices()
        except (TypeError, AttributeError):
            continue
        usable = [d for d in prefs.preferences.devices
                  if d.type == backend]
        if not usable:
            continue
        for d in prefs.preferences.devices:
            d.use = (d.type == backend)
        cycles.device = "GPU"
        _select_denoiser(cycles, backend)
        return


def _select_denoiser(cycles, compute_backend):
    """Pick the denoiser explicitly. **Never leave this to the default.**

    Blender 4.5 defaults `cycles.denoiser` to OPENIMAGEDENOISE, which runs on
    the CPU and costs a flat few seconds per `render()` call, setup rather than
    denoising work: most of a frame, and invisible, since the render is
    correct, the GPU is genuinely path tracing, and nothing reports that the
    slowest step ran somewhere else. OPTIX on an OptiX device costs about a
    tenth of a second over not denoising at all.

    OptiX denoising needs an OptiX device; on CUDA or CPU there is no such
    device and OpenImageDenoise is the only option, so we pay it there and
    accept the cost rather than shipping noise.
    """
    cycles.use_denoising = True
    want = "OPTIX" if compute_backend == "OPTIX" else "OPENIMAGEDENOISE"
    try:
        cycles.denoiser = want
    except (TypeError, AttributeError) as e:
        # An enum value this build does not carry. Denoising stays on with
        # whatever the default is — slow, but a noisy render is worse.
        print(f"[blender] could not select {want} denoiser ({e}); "
              "using the build default", file=sys.stderr)


def _setup_world(scene, job):
    """A neutral studio dome plus the flat backdrop.

    Two jobs that Open3D splits between an IBL and a background color: the
    world supplies soft omnidirectional fill so a metal has something to
    reflect and no face goes black, and the film shows the same light gray the
    other backend uses so the two look like the same product.
    """
    world = bpy.data.worlds.new("dycraft_world")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputWorld")
    bg_visible = nodes.new("ShaderNodeBackground")
    bg_light = nodes.new("ShaderNodeBackground")
    mix = nodes.new("ShaderNodeMixShader")
    is_camera = nodes.new("ShaderNodeLightPath")

    r, g, b = (_srgb_to_linear(c)
               for c in job.get("background", (0.95, 0.95, 0.95)))
    bg_visible.inputs["Color"].default_value = (r, g, b, 1.0)
    bg_visible.inputs["Strength"].default_value = 1.0
    # The lighting dome is dimmer than the backdrop is bright: a 0.95 gray
    # environment at full strength washes every shadow out, which would throw
    # away the main thing this backend is here for.
    bg_light.inputs["Color"].default_value = (0.55, 0.56, 0.58, 1.0)
    bg_light.inputs["Strength"].default_value = float(
        os.environ.get("DYCRAFT_BLENDER_DOME", "0.6"))

    links.new(is_camera.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(bg_light.outputs["Background"], mix.inputs[1])
    links.new(bg_visible.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])


#: Studio rig for a subject with no room, as (name, direction, W/m², radius
#: fraction). Direction is a UNIT vector from the subject center; the light is
#: placed `_STUDIO_DISTANCE` subject-radii along it, and its power is scaled by
#: distance² so illuminance at the subject is scale-invariant. Radius is a
#: fraction of the subject radius, so shadow softness scales too.
#:
#: **Everything here is relative, and that is the point.** Absolute positions
#: tuned for a 1 m object would light a 5 cm part and a 6 m hall from the same
#: places, and put most of a room scene's lights outside its opaque walls.
_STUDIO_RIG = (
    ("key",  (0.46, 0.69, 0.69), 7.9, 0.25),   # primary shaper, front-left-high
    ("fill", (-0.73, 0.58, 0.36), 4.2, 0.40),  # opposite, keeps the shadow side read
    ("top",  (0.0, 0.0, 1.0), 4.7, 0.50),      # lifts upward faces
)
_STUDIO_DISTANCE = 6.0        # in subject radii

#: Interior rig, for a scene that has a room. Offsets are fractions of the room
#: half-extent from its center, `z` a fraction of room height; power is W per m²
#: of floor. Lights sit INSIDE — a room's walls are opaque and a studio rig
#: outside them contributes almost nothing.
#: W/m² CALIBRATED, not chosen: on a 5.4x5.0x3.0 m room they give a full-frame
#: mean of about 224 with nothing clipping.
_INTERIOR_RIG = (
    ("key",   (0.35, 0.35, 0.88), 2.6, 0.22),
    ("fill",  (-0.55, -0.20, 0.62), 0.9, 0.30),
    ("wash",  (0.0, 0.0, 0.97), 1.4, 0.45),
)


def _setup_rig(framing=None, room=None):
    """The automatic rig, for an IR that carries no lighting of its own.

    An object is lit from outside like a product shot; a scene with a room is
    lit from within like an interior, because a studio rig outside opaque walls
    contributes almost nothing.
    """
    return (_setup_interior_lights(room) if room is not None
            else _setup_studio_lights(framing))


def _bbox_metrics(bbox):
    lo, hi = bbox
    centre = tuple((a + b) / 2.0 for a, b in zip(lo, hi))
    radius = max(max((b - a) / 2.0 for a, b in zip(lo, hi)), 1e-3)
    return centre, radius


#: Global multiplier on every light's power. Exists so the rig can be
#: calibrated against a reference exposure by measurement rather than by
#: guessing watts — the first hand-picked values overexposed a room by ~4x.
_LIGHT_SCALE = float(os.environ.get("DYCRAFT_BLENDER_LIGHT_SCALE", "1.0"))


def _add_area_light(name, pos, energy, radius, target):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy * _LIGHT_SCALE
    data.size = max(radius, 1e-3)
    data.shape = "DISK"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = pos
    _aim(obj, target)
    return obj


def _setup_studio_lights(framing):
    centre, r = _bbox_metrics(framing or ((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)))
    d = _STUDIO_DISTANCE * r
    out = []
    for name, direction, w_per_m2, rad_frac in _STUDIO_RIG:
        pos = tuple(c + u * d for c, u in zip(centre, direction))
        # Inverse-square: the same W at twice the distance is a quarter the
        # illuminance, so scaling power by d² is what makes the rig look the
        # same on a teacup and on a tractor.
        out.append(_add_area_light(name, pos, w_per_m2 * d * d,
                                   rad_frac * r, centre))
    return out


def _setup_interior_lights(room):
    (xmn, ymn, zmn), (xmx, ymx, zmx) = room
    w, dpth, h = xmx - xmn, ymx - ymn, zmx - zmn
    cx, cy = (xmn + xmx) / 2.0, (ymn + ymx) / 2.0
    floor_area = max(w * dpth, 1e-3)
    aim = (cx, cy, zmn + 0.35 * h)      # a little above the floor, not at it
    out = []
    for name, (fx, fy, fz), w_per_m2, rad_frac in _INTERIOR_RIG:
        pos = (cx + fx * w / 2.0, cy + fy * dpth / 2.0, zmn + fz * h)
        out.append(_add_area_light(name, pos, w_per_m2 * floor_area,
                                   rad_frac * min(w, dpth) / 2.0, aim))
    return out


def _aim(obj, target):
    """Point `obj`'s -Z at `target`, +Y up — Blender's camera/light convention."""
    d = np.array(target, dtype=np.float64) - np.array(obj.location)
    n = np.linalg.norm(d)
    if n == 0.0:
        return
    d /= n
    # yaw about Z then pitch about X; the standard track-to for a -Z forward
    # axis with +Y up.
    obj.rotation_euler = (math.acos(max(-1.0, min(1.0, -d[2]))),
                          0.0,
                          math.atan2(-d[0], d[1]))


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

_IMAGE_CACHE = {}
_FAILED_IMAGES = set()


def _load_image(path, digest, non_color):
    """A `bpy.types.Image` for a staged map, or None if it will not decode.

    Cached on `(digest, non_color)` for the reason the Open3D loader caches on
    `(digest, mode)`: the same grayscale file is a perfectly good roughness map
    AND metallic map, and color-space is the one thing that must not be shared
    between two uses of one file. Failures are remembered against the PATH as
    well, so a broken root costs one attempt rather than one per camera.
    """
    key = (digest, non_color)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        try:
            cached.name                 # cheap liveness probe
            return cached
        except ReferenceError:
            # The datablock was collected out from under us. Should not happen,
            # since loads carry a fake user, but reloading costs one decode
            # where propagating costs the whole sequence — see below.
            del _IMAGE_CACHE[key]
    fail_key = (str(path), digest, non_color)
    if fail_key in _FAILED_IMAGES:
        return None
    try:
        img = bpy.data.images.load(str(path), check_existing=False)
        # `.load` is lazy — a nonexistent or corrupt file only reveals itself
        # when something touches the pixels, and by then it is a black texture
        # rather than an error.
        if not img.has_data:
            img.reload()
        if img.size[0] == 0 or img.size[1] == 0:
            raise ValueError("image decoded to zero size")
    except BaseException:       # noqa: BLE001 — a surface outranks its texture
        _FAILED_IMAGES.add(fail_key)
        return None
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    # **The fake user is what keeps this cache from becoming dangling
    # pointers.** `_purge_orphans` collects any datablock with zero users, and
    # `_bake_run` calls it after every part of a viewer-asset bake. Once a
    # part's material is replaced by its baked one, every image the old
    # material referenced is orphaned and goes with it, while `_IMAGE_CACHE`
    # still holds the Python wrappers: the next part that uses the same texture
    # would get back a removed image (`ReferenceError: StructRNA of type Image
    # has been removed`), caught above only at the cost of decoding it again.
    # `test_viewer_bake.py` checks that a bake survives a texture its parts
    # share; it fails only with both this and the liveness probe gone.
    #
    # A fake user says what is true — these are deliberately retained for the
    # next part — so the purge leaves them and still collects the meshes and
    # materials it exists for. Memory stays bounded because the cache is keyed
    # by digest: a fixed set of textures, not per-part growth.
    img.use_fake_user = True
    _IMAGE_CACHE[key] = img
    return img


def _tex_node(nodes, image):
    node = nodes.new("ShaderNodeTexImage")
    node.image = image
    node.extension = "REPEAT"      # library materials TILE; see _triplanar_uvs
    return node


#: Non-fatal problems this child hit, in first-seen order. Drained by `_run`
#: into the result JSON so the parent can put them on `RenderResult.warnings`
#: and the object agent can read them in a tool result.
#:
#: **Deduplicated by message, which is not cosmetic.** `_build_material` runs
#: once per part per frame (`_render_frame` rebuilds the world every frame, and
#: `_purge_orphans` collects it again), so one broken shader on a 58-frame
#: object would otherwise report itself 58 times and bury everything else.
_WARNINGS: list = []


def _warn(message: str) -> None:
    if message not in _WARNINGS:
        _WARNINGS.append(message)
    # stderr as well as the result: a child that dies before writing its result
    # still gets its complaint into `_run_child`'s exception text.
    print(f"[blender] {message}", file=sys.stderr)


def _principled(nt):
    """The tree's Principled BSDF, by name.

    Named rather than searched by type because `_reset_to_scalars` rebuilds it
    and the probe tests look it up the same way; a fresh
    `ShaderNodeBsdfPrincipled` takes this name by default.
    """
    return nt.nodes.get("Principled BSDF")


def _apply_scalars(mat, bsdf, packed, alpha):
    """Bind the four PBR scalars, the fallback every surface starts from."""
    color = packed.get("color", (0.7, 0.7, 0.7))
    lin = [_srgb_to_linear(c) for c in color[:3]]
    bsdf.inputs["Base Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
    # Metallic and roughness are NOT color — they are already linear scalars
    # in [0,1] and converting them would be a second, subtler version of the
    # same bug in the opposite direction.
    bsdf.inputs["Metallic"].default_value = packed.get("metallic", 0.0)
    bsdf.inputs["Roughness"].default_value = packed.get("roughness", 0.9)
    bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        # EEVEE-only in 4.x and gone in some builds; Cycles honors the Alpha
        # socket directly. Set it where it exists so an EEVEE run behaves, and
        # never let its absence take down a Cycles render.
        try:
            mat.blend_method = "BLEND"
        except (AttributeError, TypeError):
            pass


def _reset_to_scalars(mat, packed, alpha):
    """Throw the node tree away and rebuild the plain scalar material.

    The recovery path when an authored shader raises. **All-or-nothing rather
    than keeping what the shader managed to build**: a half-applied graph
    renders as a material nobody authored and reads like a modeling bug, where
    a flat surface plus a warning reads like what it is. Same posture as
    `_uvs_from_rest_obj` returning None rather than pairing UVs with the wrong
    corners.
    """
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _apply_scalars(mat, bsdf, packed, alpha)
    return bsdf


def _run_shader(name, mat, bsdf, spec, packed, alpha):
    """Execute an authored shader body against this material's node tree.

    `spec` is `{"name", "body", "params"}` straight out of the IR — the text the agent
    wrote, carried inline (see `sdk/_material.Shader`). It is `exec`'d here
    because here is the only interpreter that has `bpy`.

    The body receives `nodes`, `links`, `bsdf`, and optionally `mat`, as
    declared in its captured signature. It does not receive `bpy`;
    withholding the module puts `bpy.ops`, `bpy.data`, `bpy.app`, and the whole
    scene out of reach — which matters because this runs once per part per
    frame, so a body with side effects outside its own tree would compound.
    It is a capability reduction rather than a sandbox; the real check ran at
    author time, where `agent/cad/sandbox.check_ast` walked this same
    text under the ordinary allowlist (no imports, no dunder attributes, no
    `exec`/`eval`/`getattr`).

    Returns True when the shader owns the tree, False when it raised and the
    scalars were restored.
    """
    body = spec.get("body")
    fn_name = spec.get("name")
    if not body or not fn_name:
        _warn(f"material {name!r}: shader block has no body or name; "
              f"drew the authored scalars instead")
        return False
    namespace: dict = {}
    try:
        exec(compile(body, f"<shader:{fn_name}>", "exec"),   # noqa: S102
             namespace)
        fn = namespace.get(fn_name)
        if not callable(fn):
            raise NameError(
                f"the shader body defines no function named {fn_name!r}")
        args = (mat.node_tree.nodes, mat.node_tree.links, bsdf)
        if tuple(spec.get("params") or ()) == ("nodes", "links", "bsdf", "mat"):
            args += (mat,)
        fn(*args)
    except BaseException as e:      # noqa: BLE001 — a surface outranks a render
        if _STRICT_MATERIALS:
            raise MaterialError(
                f"material {name!r}: shader {fn_name!r} raised "
                f"{type(e).__name__}: {e}") from None
        _reset_to_scalars(mat, packed, alpha)
        _warn(f"material {name!r}: shader {fn_name!r} raised "
              f"{type(e).__name__}: {e}; drew the authored scalars instead")
        return False
    return True


def _apply_emission(bsdf, nodes, links, packed, texture):
    """Bind `Material.emission` onto Principled's own emission sockets.

    **In Cycles this is a real light source**, not just a bright pixel: it
    illuminates its surroundings, appears in reflections, and is sampled like any
    other emitter. That is the whole difference from Filament, where
    `emissive_color` only makes the surface look bright — see
    `sdk/_material.py::Emission`.

    Straight onto Principled rather than a separate Emission shader mixed in:
    the surface still reflects light normally AND emits, which is what a lit
    lampshade or a backlit screen actually does. A mix would trade one for the
    other.

    **Harness-applied even under a shader**, and that is the boundary worth
    stating: a shader owns the appearance sockets, but emission is a validated
    physical quantity in W/m² with a sizing rule the material skill teaches and
    a scene-lighting interaction of its own. A shader that wants it anyway can
    still link `Emission Color` and win, because Blender replaces a socket's
    link.
    """
    emission = packed.get("emission")
    if emission is None:
        return
    er, eg, eb = (_srgb_to_linear(c) for c in emission["color"])
    bsdf.inputs["Emission Color"].default_value = (er, eg, eb, 1.0)
    bsdf.inputs["Emission Strength"].default_value = float(
        emission["strength"])

    m = texture.get(EMISSION)
    if m:
        # A map drives the color and `strength` stays the multiplier —
        # exactly Principled's own split, and the reason the schema refuses
        # an emission map with no scalar: there would be nothing to scale it
        # by and no fallback for a consumer that cannot sample.
        img = _load_image(m["path"], m["digest"], False)
        if img is not None:
            node = _tex_node(nodes, img)
            links.new(node.outputs["Color"],
                      bsdf.inputs["Emission Color"])


def _build_material(name, packed, alpha_override=None, has_uv=True):
    """A Principled BSDF from the parent's packed material dict.

    `packed` is `_world._material_to_render`'s output — the identical dict the
    Open3D backend consumes — so the two backends can never disagree about what
    a surface *is*, only about how light hits it.

    Blender's sockets genuinely replace rather than multiply, so honoring "a
    map replaces its scalar" is just "bind the map". The scalar stays the
    fallback for every kind that has no map, which is what makes an untextured
    part and a part whose texture went missing look the same.

    `has_uv` says whether the mesh this material is about to be bound to
    actually carries a UV layer. Only the NORMAL map consults it, and the
    asymmetry is deliberate — see the block below. It defaults True because the
    caller that knows is `_build_part`, and a direct caller that does not know
    should get the full graph rather than a silently reduced one.

    **A `shader` block takes a different path and returns early.** It is
    agent-authored Blender node code (`sdk/_material.Shader`), so instead of
    binding maps this builds the scalars as a floor, `exec`s the body, and lets
    it wire whatever it likes into the Principled it is handed. This is the one
    material form the other backends cannot draw, which is why the scalars are
    still bound first: they are what Filament, the viewer, and a failed shader
    all fall back to.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    # `.get` rather than `packed["alpha"]`. `_world._material_to_render` emits
    # all four scalars unconditionally, so this cannot fire from the pipeline;
    # it matches the three `.get`s in `_apply_scalars` so a hand-built spec in
    # a probe or a test needs no boilerplate.
    alpha = (packed.get("alpha", 1.0) if alpha_override is None
             else alpha_override)
    _apply_scalars(mat, bsdf, packed, alpha)

    texture = packed.get("texture") or {}

    shader = packed.get("shader")
    if shader:
        # **The shader owns the tree from here.** It is handed the same
        # Principled the scalars were just written onto, so anything it links
        # replaces that value (Blender replaces a socket's link rather than
        # adding to it) and anything it leaves alone keeps a usable fallback.
        # If it raises, `_run_shader` restores the scalar material and reports.
        #
        # The map loop below is skipped rather than merged: `Material` refuses
        # a shader beside a texture, and `material_from_ir` drops the texture
        # if malformed IR ever carries both, so there is no case to handle.
        _run_shader(name, mat, bsdf, shader, packed, alpha)
        # Re-fetched because a failed shader rebuilt the tree and the handle
        # above is stale. Emission applies on both paths: a fallback surface
        # that is supposed to glow still is.
        _apply_emission(_principled(nt), nt.nodes, nt.links, packed, texture)
        return mat

    for kind, socket, non_color in (
            (BASE_COLOR, "Base Color", False),
            ("roughness", "Roughness", True),
            ("metallic", "Metallic", True)):
        m = texture.get(kind)
        if not m:
            continue
        img = _load_image(m["path"], m["digest"], non_color)
        if img is None:
            continue            # scalar stays; see the docstring
        node = _tex_node(nodes, img)
        channel = (m.get("channel") or "").lower()
        if channel:
            # A packed source (ORM and friends) names the channel it lives in.
            # Open3D leaves these unbound entirely — `_pbr_material` says it
            # "cannot select one" — so this is a capability the schema always
            # described and no backend could use until now.
            sep = nodes.new("ShaderNodeSeparateColor")
            links.new(node.outputs["Color"], sep.inputs["Color"])
            idx = _CHANNEL_INDEX.get(channel)
            if idx is None:
                continue        # unknown channel name: scalar stays
            links.new(sep.outputs[idx], bsdf.inputs[socket])
        else:
            links.new(node.outputs["Color"], bsdf.inputs[socket])

    _apply_emission(bsdf, nodes, links, packed, texture)

    # `normal` and `height` both perturb the shading normal, and Principled has
    # exactly one `Normal` input. They are therefore CHAINED rather than each
    # wired to the BSDF: the normal map supplies the surface frame and Bump
    # perturbs whatever frame it is given, which is what Bump's own `Normal`
    # input is for. Wiring both directly would not blend them — Blender replaces
    # an input's existing link instead of adding to it, so the second one built
    # would silently win and the first would dangle.
    normal_out = None

    normal = texture.get("normal")
    if normal and not has_uv:
        # **A normal map with no UV layer is not a degraded picture, it is a
        # wrong one**, and that is why this is the only kind that is gated.
        # `ShaderNodeNormalMap` interprets the image in TANGENT space, and the
        # tangent basis Cycles derives is a function of the active UV map — with
        # no UV layer there is no basis, and the perturbation it applies is
        # undefined rather than merely flat.
        #
        # Every other kind degrades honestly in the same situation: an Image
        # Texture with nothing to sample against yields a smooth or constant
        # value, so base color reads as one texel and Bump differentiates a
        # near-constant field into no relief. Those are bad pictures of the
        # right surface. This one would be a confident picture of the wrong one,
        # which is the distinction `_pbr_material`'s `channel` skip draws too:
        # fall back to the scalar rather than to a confidently wrong number.
        normal = None
    if normal:
        # `normal` is a declared kind in `sdk/_material.py` that Open3D
        # carries and never samples. **No producer emits one today** — the
        # importers stage `base_color` alone, and the generator that did was
        # removed. Kept wired because the next producer (a procedural
        # material) is exactly the kind that would emit one, and because the
        # convention below is a measured result rather than a guess: tangent
        # space, OpenGL green, which is what `ShaderNodeNormalMap` expects
        # with no inversion.
        img = _load_image(normal["path"], normal["digest"], True)
        if img is not None:
            node = _tex_node(nodes, img)
            nmap = nodes.new("ShaderNodeNormalMap")
            strength = normal.get("strength")
            if strength is not None:
                nmap.inputs["Strength"].default_value = float(strength)
            links.new(node.outputs["Color"], nmap.inputs["Color"])
            normal_out = nmap.outputs["Normal"]

    height = texture.get(HEIGHT)
    if height:
        # A scalar relief field, not a direction field, and that is why it is
        # the safe way to get surface detail out of this pipeline. Bump
        # differentiates the sampled value along the surface, so unlike a normal
        # map it needs no tangent handedness — the mirrored UVs that
        # `_world._triplanar_uvs` produces on negative-facing faces, and the
        # per-triangle projection seams at ~45 degrees, degrade to a gradient
        # discontinuity rather than to lighting that points the wrong way.
        #
        # Shading only: the mesh is untouched, so every verifier, collision
        # check and content hash still measures the surface the author authored.
        img = _load_image(height["path"], height["digest"], True)
        if img is not None:
            node = _tex_node(nodes, img)
            bump = nodes.new("ShaderNodeBump")
            # `Distance` is the relief depth in meters and Blender's default
            # is 1.0 — a meter of bump, which is not a subtle failure. A map
            # that carries no strength therefore gets an explicit floor rather
            # than the node default, and a producer that knows its own relief
            # depth should set `TextureMap.strength` instead of relying on it.
            #
            # Note the kind-dependence: Bump reads `strength` as `Distance` in
            # METERS, NormalMap above reads it as a [0, 1] blend. A producer
            # that computes one number for both kinds silently switches the
            # normal map off.
            dist = height.get("strength")
            bump.inputs["Distance"].default_value = (
                float(dist) if dist is not None else _BUMP_DISTANCE_DEFAULT)
            links.new(node.outputs["Color"], bump.inputs["Height"])
            if normal_out is not None:
                links.new(normal_out, bump.inputs["Normal"])
            normal_out = bump.outputs["Normal"]

    if normal_out is not None:
        links.new(normal_out, bsdf.inputs["Normal"])

    # Six kinds are sampled, and they are NAMED rather than iterated: base
    # color, roughness and metallic in the loop above, then emission, height
    # and normal handled individually because each lands somewhere other than a
    # plain Principled socket. Every other kind travels here and is simply not
    # sampled, which is the schema's required degradation and the reason naming
    # them is safe — an unknown kind cannot be wired wrong if nothing looks for
    # it.
    return mat


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _quat_matrix(q):
    from mathutils import Quaternion
    return Quaternion((q[0], q[1], q[2], q[3])).to_matrix().to_4x4()


def _place(obj, se3):
    from mathutils import Matrix
    t, q = se3["t"], se3["q"]
    obj.matrix_world = Matrix.Translation(t) @ _quat_matrix(q)


def _mesh_from_arrays(name, verts, tris, uvs=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(t) for t in tris])
    mesh.update()
    if uvs is not None and len(uvs) == 3 * len(tris):
        # Per-CORNER, matching `_triplanar_uvs`' `(3 * n_triangles, 2)` layout —
        # which is also Blender's loop layout, so a shared vertex carries a
        # different UV on each face and the projection needs no seam handling.
        layer = mesh.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    mesh.update()
    return mesh


def _shade_from_geometry(obj):
    """Recompute normals from the geometry actually being drawn.

    Deliberately flat-by-angle rather than smooth-everywhere: our meshes are
    welded, and smoothing across a CAD crease rounds off every edge a solid was
    modeled to have — the same trap `_part_mesh` documents on the Open3D side
    when it refuses to call `compute_vertex_normals()` unconditionally.
    """
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    mod = obj.modifiers.new(name="sharp", type="EDGE_SPLIT")
    mod.split_angle = math.radians(30.0)


def _import_asset(spec, name):
    """Load a staged OBJ, keeping the UVs it was baked with. None on failure."""
    before = set(bpy.context.scene.objects)
    try:
        bpy.ops.wm.obj_import(filepath=spec["obj"], forward_axis="Y",
                              up_axis="Z")
    except BaseException:       # noqa: BLE001 — geometry outranks its asset
        return None
    new = [o for o in bpy.context.scene.objects if o not in before]
    if not new:
        return None
    if len(new) > 1:
        # One part is one object. Joining rather than keeping a loose group
        # keeps the material assignment and the world transform below single
        # statements instead of loops that could miss a piece.
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        new = [bpy.context.view_layer.objects.active]
    obj = new[0]
    obj.name = name
    scale = spec.get("scale", (1.0, 1.0, 1.0))
    if tuple(scale) != (1.0, 1.0, 1.0):
        # In the mesh-local frame, before the world SE3 — same order as
        # `_part_mesh`. Applied to the data so the later `matrix_world`
        # assignment does not overwrite it.
        obj.data.transform(_scale_matrix(scale))
    return obj


def _scale_matrix(scale):
    from mathutils import Matrix
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][i] = float(scale[i])
    return m


def _uvs_from_rest_obj(obj_path, n_tris):
    """Per-corner UVs read out of a rest OBJ, or None.

    For a deformed part: LBS moved the vertices and left the topology alone, so
    the rest mesh's UVs are still correct — but they live in a file this path
    otherwise never opens. A length mismatch means the staged file and the
    deformed arrays disagree about topology, and pairing UVs with the wrong
    corners is worse than drawing untextured, so it returns None instead.
    """
    if not obj_path:
        return None
    before = set(bpy.context.scene.objects)
    try:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    except BaseException:       # noqa: BLE001
        return None
    new = [o for o in bpy.context.scene.objects if o not in before]
    if not new:
        return None
    uvs = None
    mesh = new[0].data
    if mesh.uv_layers.active and len(mesh.loops) == 3 * n_tris:
        flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        mesh.uv_layers.active.data.foreach_get("uv", flat)
        uvs = flat.reshape(-1, 2)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    return uvs


def _apply_camera_visibility(obj, spec):
    """Hide `obj` from camera rays while leaving every other ray path intact.

    This is how a room surface can stand behind the camera — or overhead — and
    still light the room. Clearing `visible_camera` alone leaves diffuse,
    glossy, transmission, and shadow rays on, so the surface bounces, colors
    the bounce and casts, but never appears in frame. Deleting it instead
    removes all of that with it.

    Absent or true means "draw it", so every existing caller is unaffected.
    """
    if spec.get("camera_visible", True):
        return
    # Guarded: `visible_camera` is a Cycles-era property. A build without it
    # should render a visible wall rather than crash the frame.
    try:
        obj.visible_camera = False
    except AttributeError:
        pass


#: Prefix for every node this module inserts into an authored material, so a
#: shader's own nodes can never be mistaken for the cutaway's.
_CUTAWAY_NODE_PREFIX = "dycraft_cutaway"

#: `{surface_tag: [enable_node, ...]}` over every replacement part built for
#: the frame currently in hand. Module-level because the parts are built once
#: and then rendered from several cameras, and each camera needs different
#: slots live. Cleared by `_reset_camera_cutaway` at the start of every frame
#: build, so a previous frame's removed nodes can never be written to.
#:
#: **A list per tag, not a node per tag.** An authored shell is many parts
#: (a typical one has eleven) and every one of them carries its own copy of
#: every slot, because the mask lives in each part's own material. A dict of
#: single nodes silently kept only the last part built and left the other ten
#: switched off — which renders as a closed room with one cut part.
_CUTAWAY_CONTROLS: dict = {}


def _reset_camera_cutaway():
    _CUTAWAY_CONTROLS.clear()


def _record_camera_cutaway(controls):
    for tag, node in controls.items():
        _CUTAWAY_CONTROLS.setdefault(tag, []).append(node)

#: Axis index -> the `ShaderNodeSeparateXYZ` output that carries it.
_AXIS_SOCKET = ("X", "Y", "Z")


def _install_camera_cutaway(obj, slots):
    """Make `obj` see-through to CAMERA RAYS ONLY, inside chosen half-spaces.

    The authored-architecture half of the room cutaway. The automatic shell is hidden with `visible_camera`, one
    object per surface, but an authored shell often carries all four walls as one solid, and hiding it would take the
    far wall the shot needs too. So the cut is made per shading point, appended to whatever the material builds::

        Light Path.Is Camera Ray ─┐
                                  ├─ MULTIPLY ── Mix Shader.Fac
        New Geometry.Position ─ Separate XYZ ─ per slot:
            GREATER_THAN/LESS_THAN(p[axis], bound) ─ MULTIPLY(enable) ┐
                                                       MAXIMUM-chain ─┘

        Mix Shader: Fac=0 -> the authored surface, Fac=1 -> Transparent BSDF

    The mask is non-zero only for a primary camera ray inside the slab the suppressed shell surface would occupy.
    Every other ray (diffuse, glossy, transmission, shadow, volume scatter) takes the Fac=0 branch and hits the
    authored surface at full opacity, so the wall still shadows, bounces, reflects, occludes, and shows in a mirror
    inside the room. Alpha, `visible_transmission`, and deleting geometry all fail that test.

    Each slot gets a `ShaderNodeValue` enable, built once and switched per camera by `_set_camera_cutaway`, rather
    than rebuilding the graph per camera and re-linking whatever an authored `@shader` wired into the output.

    Returns `{surface_tag: enable_node}`, empty when there is nothing to build.
    """
    controls: dict = {}
    if not slots:
        return controls
    for slot_mat in obj.data.materials:
        if slot_mat is None or not slot_mat.use_nodes:
            continue
        nt = slot_mat.node_tree
        nodes, links = nt.nodes, nt.links
        out = next((q for q in nodes if q.type == "OUTPUT_MATERIAL"), None)
        if out is None or not out.inputs["Surface"].is_linked:
            # Nothing authored reaches the output; there is no surface to
            # conditionally replace and a Mix would make one up.
            continue
        source = out.inputs["Surface"].links[0].from_socket

        lp = nodes.new("ShaderNodeLightPath")
        lp.name = f"{_CUTAWAY_NODE_PREFIX}_lightpath"
        geo = nodes.new("ShaderNodeNewGeometry")
        geo.name = f"{_CUTAWAY_NODE_PREFIX}_geometry"
        sep = nodes.new("ShaderNodeSeparateXYZ")
        sep.name = f"{_CUTAWAY_NODE_PREFIX}_separate"
        links.new(geo.outputs["Position"], sep.inputs["Vector"])

        acc = None
        for slot in slots:
            axis, sign = int(slot["axis"]), int(slot["sign"])
            cmp_node = nodes.new("ShaderNodeMath")
            cmp_node.name = f"{_CUTAWAY_NODE_PREFIX}_cmp_{slot['tag']}"
            cmp_node.operation = "GREATER_THAN" if sign > 0 else "LESS_THAN"
            links.new(sep.outputs[_AXIS_SOCKET[axis]], cmp_node.inputs[0])
            cmp_node.inputs[1].default_value = float(slot["bound"])

            enable = nodes.new("ShaderNodeValue")
            enable.name = f"{_CUTAWAY_NODE_PREFIX}_enable_{slot['tag']}"
            enable.outputs[0].default_value = 0.0
            gate = nodes.new("ShaderNodeMath")
            gate.name = f"{_CUTAWAY_NODE_PREFIX}_gate_{slot['tag']}"
            gate.operation = "MULTIPLY"
            links.new(cmp_node.outputs["Value"], gate.inputs[0])
            links.new(enable.outputs["Value"], gate.inputs[1])
            controls[slot["tag"]] = enable

            if acc is None:
                acc = gate
            else:
                union = nodes.new("ShaderNodeMath")
                union.name = f"{_CUTAWAY_NODE_PREFIX}_or_{slot['tag']}"
                union.operation = "MAXIMUM"
                links.new(acc.outputs["Value"], union.inputs[0])
                links.new(gate.outputs["Value"], union.inputs[1])
                acc = union

        mask = nodes.new("ShaderNodeMath")
        mask.name = f"{_CUTAWAY_NODE_PREFIX}_mask"
        mask.operation = "MULTIPLY"
        links.new(lp.outputs["Is Camera Ray"], mask.inputs[0])
        links.new(acc.outputs["Value"], mask.inputs[1])

        transparent = nodes.new("ShaderNodeBsdfTransparent")
        transparent.name = f"{_CUTAWAY_NODE_PREFIX}_transparent"
        mix = nodes.new("ShaderNodeMixShader")
        mix.name = f"{_CUTAWAY_NODE_PREFIX}_mix"
        links.new(mask.outputs["Value"], mix.inputs["Fac"])
        # Fac=0 keeps the authored surface; Fac=1 selects Transparent.
        links.new(source, mix.inputs[1])
        links.new(transparent.outputs["BSDF"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return controls


def _set_camera_cutaway(controls, hidden_tags):
    """Switch the cutaway slots this camera needs on, and every other off.

    `controls` is the accumulated `{tag: [enable_node, ...]}` from
    `_record_camera_cutaway`. Switching a float is what makes the walls correct
    per camera without touching the node graph again.
    """
    live = set(hidden_tags or ())
    for tag, nodes in controls.items():
        value = 1.0 if tag in live else 0.0
        for node in nodes:
            node.outputs[0].default_value = value


def _build_part(spec, index, prefix="part"):
    """One part as a Blender object, placed and shaded. None if unbuildable."""
    name = f"{prefix}_{index}"
    kind = spec["kind"]

    if kind == "asset":
        obj = _import_asset(spec, name)
        if obj is None:
            # Degrade to the AABB the parent shipped alongside, exactly as
            # `open3d_renderer._part_mesh` does — a crude box where the object
            # should be, not a hole where the object was. Dropping the part
            # instead produced an empty room on a clip whose asset paths did not
            # resolve, while Open3D drew a scene of boxes: same broken input,
            # and only one of the two pictures said so.
            print(f"[blender] {name}: asset would not import, drawing its "
                  f"bounding box: {spec.get('obj')}", file=sys.stderr)
            fb = spec.get("fallback")
            if not fb:
                return None
            spec = {**spec, **fb, "kind": "mesh"}
            kind = "mesh"
    if kind != "asset":
        verts = np.load(spec["verts_npy"])
        tris = np.load(spec["tris_npy"])
        uvs = None
        if spec.get("uvs_npy"):
            uvs = np.load(spec["uvs_npy"])
        elif spec.get("uv_from_obj"):
            uvs = _uvs_from_rest_obj(spec["uv_from_obj"], len(tris))
        mesh = _mesh_from_arrays(name, verts, tris, uvs)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

    _place(obj, spec["se3"])
    _shade_from_geometry(obj)
    obj.data.materials.clear()
    # **The MESH is asked, not the local `uvs`.** This function has two
    # branches: an `asset` is imported through `_import_asset` and brings its
    # own UVs without ever touching `uvs`, while a `mesh` builds from arrays and
    # may leave it None. Guarding on `uvs is not None` would therefore disable
    # normal maps on exactly the case they matter most for — imported
    # characters. `obj.data.uv_layers` is correct for both.
    obj.data.materials.append(
        _build_material(f"{name}_mat", spec["material"], spec.get("alpha"),
                        has_uv=bool(obj.data.uv_layers)))
    # After the material, including an authored `@shader`: the mask wraps
    # whatever ended up wired to the output rather than competing with it.
    _record_camera_cutaway(
        _install_camera_cutaway(obj, spec.get("cutaway_slots")))
    return obj


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

def _make_camera(scene):
    data = bpy.data.cameras.new("cam")
    # CAD objects use meters; the Blender default (0.1 m) clips small parts.
    data.clip_start = 0.0001
    data.sensor_fit = "VERTICAL"      # CameraSpec.fov_deg is the VERTICAL fov
    obj = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(obj)
    scene.camera = obj
    return obj


def _aim_camera(cam_obj, spec):
    from mathutils import Matrix, Vector

    eye = Vector(spec["eye"])
    target = Vector(spec["target"])
    up = Vector(spec["up"])

    forward = (target - eye)
    forward.normalize()
    right = forward.cross(up)
    if right.length == 0.0:
        # A degenerate `up` reaches us only from a caller-built CameraSpec;
        # `look_at_camera` refuses one. Pick any perpendicular rather than
        # emitting a NaN matrix that renders black with no explanation.
        right = forward.cross(Vector((0.0, 1.0, 0.0)))
        if right.length == 0.0:
            right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    true_up = right.cross(forward)

    # Blender cameras look down -Z with +Y up, so the basis is
    # (right, up, -forward) — the same convention Filament uses, which is why
    # the canonical camera table in `render/__init__.py` needs no adjustment.
    rot = Matrix((
        (right.x, true_up.x, -forward.x, 0.0),
        (right.y, true_up.y, -forward.y, 0.0),
        (right.z, true_up.z, -forward.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    cam_obj.matrix_world = Matrix.Translation(eye) @ rot
    cam_obj.data.angle_y = math.radians(spec["fov_deg"])


# ---------------------------------------------------------------------------
# Blender-native authored lighting (`@lighting`)
# ---------------------------------------------------------------------------

#: Pure numeric names injected into a lighting body's namespace. Kept
#: character-for-character in step with `sdk/_lighting_block.EXTRA_ALLOWED_FREE`
#: — that list is what the author-time check allows, this dict is what actually
#: exists at exec time, and a drift between them is a body that passes review
#: and then fails inside Cycles. `tests/test_lighting_block.py` pins the pair.
_LIGHTING_MATH = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "hypot": math.hypot, "sqrt": math.sqrt,
    "exp": math.exp, "log": math.log, "radians": math.radians,
    "degrees": math.degrees, "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "tau": math.tau, "inf": math.inf,
}

#: Light types the façade will create. Blender's own set for this build; named
#: explicitly so a typo is a clear error rather than a Blender enum failure
#: deep inside the body.
_LIGHT_TYPES = ("POINT", "SUN", "SPOT", "AREA")


class MaterialError(RuntimeError):
    """An authored `@shader` failed while strict materials were in force.

    The scalar fallback in `_apply_shader` is right for a preview — a flat
    color still describes a surface and a picture beats a stall. It is wrong
    for any image someone judges appearance from, which is why
    `strict_materials` turns it into this: the project agent's visual review
    and the delivery render must never show a material the agent did not
    author, because the whole point of those images is to judge the ones it did.
    """

    #: Raised by the author's own block: what the harness reply calls it.
    failure_class = "author_code"


class LightingError(RuntimeError):
    """An authored lighting block failed.

    Raised rather than swallowed, and that is the whole difference from
    `_apply_shader`'s scalar fallback. A material that fails can degrade to a
    flat color and still describe a surface; lighting that fails has no
    degraded form — substituting a rig would produce a plausible image of
    lighting the author never wrote, which is exactly how a scene comes to be
    judged on atmosphere nobody chose.
    """

    #: Raised by the author's own block: what the harness reply calls it.
    failure_class = "author_code"


class _Recorded:
    """A settings struct whose attribute writes are noted as `owner.name`.

    So a lighting refresh can set again exactly what the block set, including
    a value equal to the default it replaced, which a before/after diff of the
    scene cannot see (`update.apply_scene_settings`).
    """

    def __init__(self, holder, owner, written):
        object.__setattr__(self, "_holder", holder)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_written", written)

    def __getattr__(self, name):
        return getattr(self._holder, name)

    def __setattr__(self, name, value):
        setattr(self._holder, name, value)
        self._written.add(f"{self._owner}.{name}")


#: The `owner.name` settings the last `@lighting` block wrote; see `_Recorded`.
LIGHTING_WRITES = set()


class _LightingFacade:
    """The single object a `@lighting` body is handed, as `L`.

    It exposes the lighting and photographic domain and nothing else. `bpy` is
    not passed, so `bpy.ops`, `bpy.data.objects`, `bpy.app` and the module's
    whole surface are unreachable; `agent/cad/sandbox.check_ast`
    independently refuses imports, dunder attributes and
    `exec`/`eval`/`getattr`. What is left is: real light objects and
    datablocks, the complete World node tree, the scene's view settings, and
    managed images resolved by run-relative path.
    """

    def __init__(self, scene, run_dir, asset_base_dir=None):
        self._scene = scene
        self._run_dir = run_dir
        self._asset_base = asset_base_dir or run_dir
        self._created = []
        self._world = self._fresh_world(scene)

    # --- world ---------------------------------------------------------
    @staticmethod
    def _fresh_world(scene):
        """A blank World with an empty node tree and an Output node.

        Blank rather than the studio dome + gray backdrop `_setup_world`
        builds: for a v2 scene the authored World is authoritative, so the
        renderer must not pre-supply light the author did not ask for. A scene
        whose block builds no World renders against black, which is a visible
        fact the review will show rather than a default that hides the omission.
        """
        world = bpy.data.worlds.new("dycraft_authored_world")
        scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.clear()
        out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        out.location = (300, 0)
        return world

    @property
    def world(self):
        return self._world

    @property
    def world_nodes(self):
        return self._world.node_tree.nodes

    @property
    def world_links(self):
        return self._world.node_tree.links

    @property
    def world_output(self):
        for n in self._world.node_tree.nodes:
            if n.type == "OUTPUT_WORLD":
                return n
        return self._world.node_tree.nodes.new("ShaderNodeOutputWorld")

    # --- lights --------------------------------------------------------
    def new_light(self, name, type="POINT"):
        """Create and link a Blender light object; returns the object.

        The object and its `.data` are the real datablocks, so every light
        property this Blender has — energy, size/shape, spot cone and blend,
        sun angle, color, shadow softness and toggles, Cycles per-light
        settings, visibility flags, light linking — is reachable without the
        façade having to enumerate them.
        """
        kind = str(type).upper()
        if kind not in _LIGHT_TYPES:
            raise LightingError(
                f"unknown light type {type!r}; expected one of "
                f"{list(_LIGHT_TYPES)}")
        data = bpy.data.lights.new(name=str(name), type=kind)
        obj = bpy.data.objects.new(str(name), data)
        bpy.context.collection.objects.link(obj)
        self._created.append(obj)
        return obj

    @property
    def lights(self):
        """The light objects created so far, in creation order."""
        return list(self._created)

    def aim(self, obj, target):
        """Point `obj`'s -Z at `target` — Blender's light/camera convention.

        An area light left unrotated emits along -Z, straight down. A light
        with no `aim` and no `rotation_euler` still points along -Z.
        """
        _aim(obj, tuple(target))
        return obj

    # --- photographic settings ----------------------------------------
    @property
    def view(self):
        """`scene.view_settings` — exposure, look, view transform, white
        balance, curve mapping. On this Blender: exposure, gamma, look,
        view_transform, use_white_balance, white_balance_temperature/_tint/
        _whitepoint, use_curve_mapping, curve_mapping, use_hdr_view.

        Note the ordering Blender imposes: `look`'s enum is populated from the
        active `view_transform`, so set the transform first.
        """
        return _Recorded(self._scene.view_settings, "view_settings", LIGHTING_WRITES)

    @property
    def display(self):
        """`scene.display_settings` — the display device."""
        return _Recorded(self._scene.display_settings, "display_settings", LIGHTING_WRITES)

    @property
    def render(self):
        """`scene.render` — film settings relevant to atmosphere
        (`film_transparent`, exposure-adjacent film knobs)."""
        return _Recorded(self._scene.render, "render", LIGHTING_WRITES)

    @property
    def cycles(self):
        """`scene.cycles`, or None on a non-Cycles engine. Bounce counts,
        caustics, clamping, and blur_glossy all change how authored light
        propagates, so they belong to whoever authored the light."""
        cycles = getattr(self._scene, "cycles", None)
        return None if cycles is None else _Recorded(cycles, "cycles", LIGHTING_WRITES)

    # --- managed assets ------------------------------------------------
    def image(self, relpath):
        """Load a managed image (HDRI, EXR, texture) by run-relative path.

        Confined to the run directory, which is the whole filesystem surface
        this façade offers: no open(), no arbitrary path, no network.
        """
        rel = str(relpath)
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise LightingError(
                f"image path must be run-relative, got {rel!r}")
        for base in (self._asset_base, self._run_dir):
            if base is None:
                continue
            cand = Path(base) / rel
            if cand.exists():
                return bpy.data.images.load(str(cand), check_existing=True)
        raise LightingError(f"managed image not found: {rel}")


def _apply_lighting(scene, spec, run_dir, asset_base_dir=None):
    """Execute the scene's `@lighting` block. Returns its created lights.

    Runs LAST — after geometry, materials, and cameras — and nothing in this
    module writes a light, World setting, exposure, view transform, look, or
    white balance after it returns. `_render_frame` asserts that ordering.
    """
    body = (spec or {}).get("body")
    name = (spec or {}).get("name")
    if not body or not name:
        raise LightingError(
            "the scene carries a lighting block with no body or name")
    namespace = dict(_LIGHTING_MATH)
    LIGHTING_WRITES.clear()
    try:
        exec(compile(body, f"<lighting:{name}>", "exec"), namespace)  # noqa: S102
        fn = namespace.get(name)
        if not callable(fn):
            raise LightingError(
                f"the lighting body defines no function named {name!r}")
        facade = _LightingFacade(scene, run_dir, asset_base_dir)
        fn(facade)
    except LightingError as e:
        # Raised by the facade itself (a bad light type, an unresolvable
        # managed image). Re-raised with the block name attached, because the
        # agent's repair loop needs to know WHICH block to fix, and the facade
        # has no reason to know its own caller's name.
        raise LightingError(f"lighting {name!r}: {e}") from None
    except BaseException as e:                              # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        raise LightingError(
            f"lighting {name!r} raised {type(e).__name__}: {e}\n{tb}") from None
    return facade.lights


def _read_view_settings(scene) -> dict:
    """The color management actually in force, for the appearance hash.

    Read back after the block has run rather than taken from what was asked
    for: the review record has to describe the image that exists.
    """
    vs = scene.view_settings
    out = {}
    for attr in ("view_transform", "look", "exposure", "gamma",
                 "use_white_balance", "white_balance_temperature",
                 "white_balance_tint", "use_curve_mapping"):
        try:
            val = getattr(vs, attr)
        except AttributeError:
            continue
        out[attr] = val if isinstance(val, (str, bool, int)) else float(val)
    try:
        out["display_device"] = scene.display_settings.display_device
    except AttributeError:
        pass
    return out


def _install_frame_lighting(scene, frame):
    """Install this frame's lighting. The single entry point for both callers.

    `_render_frame` (canonical stills, review renders, delivery) and
    `_serve_build` (the resident viewer) both come through here, so a v2 scene
    cannot be lit one way in a still and another way in the viewport.

    Returns the light objects created, for teardown.
    """
    if frame.get("lighting"):
        return _apply_lighting(scene, frame["lighting"], frame.get("run_dir"),
                               frame.get("asset_base_dir"))
    if frame.get("schema_v2"):
        # A v2 scene that authored no lighting renders unlit, deliberately:
        # the automatic rig is gone, and an unlit frame is a reportable fact
        # rather than light nobody chose. `_reset_scene` has already left the
        # World blank for this case.
        _warn("this scene authored no @lighting block, so the frame is lit "
              "only by whatever the materials emit")
        return []
    return _setup_rig(frame.get("framing"), frame.get("room"))


def _render_frame(scene, cam_obj, frame):
    """One frame: geometry and materials, then lighting, then render.

    **The order is the contract**, and it is why lighting moved below the part
    loop. A v2 scene's `@lighting` block is the final authority on lights, the
    World and color management, so it has to run after everything else that
    could touch them and nothing may touch them afterwards. An IR with no
    lighting of its own (an object) gets the automatic rig, `_setup_rig`.
    """
    _reset_camera_cutaway()
    parts = []
    for i, spec in enumerate(frame["parts"]):
        obj = _build_part(spec, i)
        if obj is not None:
            parts.append(obj)

    # Lighting is per FRAME, not per scene: a v1 rig is sized from this frame's
    # framing bbox, and a v2 block may read the frame's own room. Either way it
    # is installed here, after geometry and materials exist.
    lights = _install_frame_lighting(scene, frame)

    wall_objs = []
    for cam_spec in frame["cameras"]:
        # Walls are rebuilt per camera rather than hidden, because which
        # surfaces exist at all is a per-camera answer (`room_wall_boxes`) and
        # a stale wall from the previous camera would stand in this one's shot.
        for obj in wall_objs:
            bpy.data.objects.remove(obj, do_unlink=True)
        wall_objs = []
        for i, wall in enumerate(cam_spec.get("walls") or []):
            obj = _build_part(wall, i, prefix=f"wall_{wall['tag']}")
            if obj is not None:
                _apply_camera_visibility(obj, wall)
                wall_objs.append(obj)

        # Authored architecture: switch on the half-spaces THIS eye is behind
        # (see `_install_camera_cutaway`). Must precede the render and follow
        # the wall rebuild, for the same reason walls are per camera.
        _set_camera_cutaway(_CUTAWAY_CONTROLS, cam_spec.get("cutaway_hidden"))

        _aim_camera(cam_obj, cam_spec)
        scene.render.filepath = cam_spec["out"]
        bpy.ops.render.render(write_still=True)

    for obj in wall_objs + parts + lights:
        bpy.data.objects.remove(obj, do_unlink=True)
    _purge_orphans()


def _purge_orphans():
    """Free the mesh and material datablocks the removed objects left behind.

    **`bpy.data.objects.remove()` does not do this.** It unlinks the object;
    the mesh and the material it pointed at stay in `bpy.data` with a zero user
    count, and Blender only collects them on file save/reload — which never
    happens in a background render. In a *still* that is invisible, because the
    process exits. In one process that builds many things (a bake's parts, a
    multi-frame job) it accumulates, megabytes per frame on a large scene.

    Cheap relative to a frame (sub-millisecond against ~1 s of Cycles), so it
    runs unconditionally rather than on a heuristic. Failure is non-fatal: a
    build whose operator context refuses the call leaks memory, which is worse
    than not leaking and better than losing the render.
    """
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except BaseException as e:      # noqa: BLE001 — a leak outranks a crash
        print(f"[blender] orphan purge failed ({e}); memory will grow across "
              "frames", file=sys.stderr)


def _run(job_path):
    job = json.loads(open(job_path, encoding="utf-8").read())
    _WARNINGS.clear()
    scene = _reset_scene(job)
    cam_obj = _make_camera(scene)
    written = []
    for frame in job["frames"]:
        try:
            _render_frame(scene, cam_obj, frame)
        except MaterialError as e:
            return {"ok": False, "error": "shader_failed",
                    "detail": str(e), "written": [],
                    "engine": scene.render.engine,
                    "warnings": list(_WARNINGS)}
        except LightingError as e:
            # An explicit structured diagnostic and NO images, rather than a
            # fallback rig. A misleading render of lighting the author never
            # wrote is worse than a failed one, because the whole point of the
            # visual review is to judge the atmosphere that was authored.
            return {"ok": False, "error": "lighting_failed",
                    "detail": str(e), "written": [],
                    "engine": scene.render.engine,
                    "warnings": list(_WARNINGS)}
        written.extend(c["out"] for c in frame["cameras"])
    return {"ok": True, "written": written,
            "engine": scene.render.engine,
            "device": getattr(getattr(scene, "cycles", None), "device", None),
            # The color management actually in force after the last frame's
            # lighting block ran. The review record's appearance hash needs the
            # real values, not the requested ones.
            "view_settings": _read_view_settings(scene),
            # Non-fatal, and the reason they travel rather than only reaching
            # stderr: a shader that raised produced a picture, so the render
            # succeeds and the only way the agent learns is a tool result.
            "warnings": list(_WARNINGS)}


# ---------------------------------------------------------------------------
# Interactive serve mode
# ---------------------------------------------------------------------------
#
# The viewer's Blender viewport. Same scene build and the same
# `_build_material` / `_run_shader` as a canonical still — the ONLY differences
# are that the scene is built once and then kept, and that frames come back as
# JPEG bytes instead of PNGs on disk.
#
# **`use_persistent_data` is what makes this viable at all**, and it is worth
# recording why rather than leaving it as a flag someone later "cleans up".
# Without it Cycles re-exports the scene, rebuilds the BVH, and re-uploads
# device buffers on *every* `bpy.ops.render.render()` call: hundreds of
# milliseconds of fixed cost, the same for a 1x1 single-sample render as for a
# real one, because none of it is sampling. With the flag set it is under a
# millisecond, and a 640x360 frame renders at tens of frames a second.
#
# Two further measurements shaped the loop below:
#
#   * **Sample count is nearly free here.** With `use_adaptive_sampling` and
#     the production `adaptive_threshold=0.01`, 1 spp and 128 spp cost the same
#     within noise (8.1-13.3 ms at 640x360). So the interactive frames are NOT
#     a lower-quality render — there is no quality being traded away, only
#     pixels. Resolution is the only real knob, which is why that is the only
#     thing the parent varies.
#   * **JPEG to a RAM disk costs 7.5 ms; PNG costs 21.6 ms.** Hence JPEG, and
#     hence `/dev/shm` rather than the run directory.
#
# Rigid re-posing is genuinely incremental: assigning `matrix_world` only
# updates instance transforms, so the per-object BVHs stay resident. Measured,
# re-posing all 16 parts every frame costs no more than re-posing one, and both
# cost what a camera move costs. That is what makes Preview Motion work.

#: Framing marker for the length-prefixed reply channel.
_SERVE_READY = "ready"


def _shm_dir(job):
    """Where to spool the JPEG. RAM if there is any, else the parent's scratch."""
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        return shm
    return job.get("stream_dir") or os.getcwd()


def _serve_build(job):
    """Build the resident scene from a one-frame job. Returns the live state."""
    frame = job["frames"][0]
    scene = _reset_scene(job)
    cam_obj = _make_camera(scene)

    # Index-aligned with `frame["parts"]`, **including the failures**: the
    # parent addresses parts by position when it sends new poses, so a part
    # that would not build has to keep its slot rather than shift every part
    # after it onto the wrong transform.
    _reset_camera_cutaway()
    parts = [_build_part(spec, i) for i, spec in enumerate(frame["parts"])]

    walls = _serve_walls(frame["cameras"][0].get("walls") or [], [])
    _set_camera_cutaway(_CUTAWAY_CONTROLS,
                        frame["cameras"][0].get("cutaway_hidden"))

    # Lighting last, in the same order `_render_frame` uses, so a v2 block sees
    # the finished scene and nothing overwrites what it set afterwards.
    lights = _install_frame_lighting(scene, frame)
    _aim_camera(cam_obj, frame["cameras"][0])

    # The flag this whole mode exists for. Set after the build, since it only
    # governs what survives *between* renders.
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = int(job.get("jpeg_quality", 85))
    return {"scene": scene, "cam": cam_obj, "parts": parts, "lights": lights,
            "walls": walls, "shm": _shm_dir(job), "n": 0, "spool": set()}


def _serve_walls(specs, previous):
    """Replace the wall objects. Rebuilt rather than hidden, as per camera.

    Which room surfaces exist at all is a per-camera answer (`room_wall_boxes`
    parent-side), so an orbit can change the set. Their meshes are removed
    explicitly rather than left to `orphans_purge`, which in this mode would be
    walking the whole of `bpy.data` on every camera move.
    """
    for obj in previous:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    built = []
    for i, wall in enumerate(specs):
        obj = _build_part(wall, i, prefix=f"wall_{wall['tag']}")
        if obj is not None:
            _apply_camera_visibility(obj, wall)
            built.append(obj)
    return built


def _serve_frame(state, req):
    """Apply the request's deltas and return the encoded frame's bytes."""
    scene, cam = state["scene"], state["cam"]

    poses = req.get("se3")
    if poses is not None:
        parts = state["parts"]
        if len(poses) != len(parts):
            raise ValueError(
                f"pose list has {len(poses)} entries for {len(parts)} parts — "
                "the parent's part list changed shape, which this mode cannot "
                "apply incrementally")
        for obj, se3 in zip(parts, poses):
            if obj is not None and se3 is not None:
                _place(obj, se3)

    walls = req.get("walls")
    if walls is not None:
        state["walls"] = _serve_walls(walls, state["walls"])
    if "cutaway_hidden" in req:
        # Same per-camera switch the still path makes, for an orbit that walks
        # the eye outside a different authored wall on every move.
        _set_camera_cutaway(_CUTAWAY_CONTROLS, req.get("cutaway_hidden"))

    cam_spec = req.get("camera")
    if cam_spec is not None:
        _aim_camera(cam, cam_spec)

    width, height = int(req["width"]), int(req["height"])
    if (scene.render.resolution_x, scene.render.resolution_y) != (width, height):
        scene.render.resolution_x = width
        scene.render.resolution_y = height
    samples = req.get("samples")
    if samples is not None and scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)

    state["n"] += 1
    # **PID in the name.** `/dev/shm` is shared, and two viewers on one machine
    # are two of these processes: a fixed filename would have each serving the
    # other's frames, which looks like a rendering bug and is a filename bug.
    # The alternating index is belt-and-braces on top, so a reader still
    # holding the previous path cannot be handed half-written bytes.
    path = os.path.join(state["shm"],
                        f"dycraft_live_{os.getpid()}_{state['n'] % 2}.jpg")
    state["spool"].add(path)
    scene.render.filepath = path[: -len(".jpg")]
    bpy.ops.render.render(write_still=True)
    with open(path, "rb") as fh:
        return fh.read()


def _serve(job_path, req_fd, rep_fd):
    """Build once, then answer frame requests until the parent closes the pipe.

    Request and reply travel on dedicated file descriptors rather than
    stdout/stdin because **Cycles writes render progress to stdout** ("Fra:1
    Mem:..." per tile), which would interleave with the protocol. stdout is the
    parent's to discard; stderr stays the diagnostic channel it is everywhere
    else in this file.
    """
    job = json.loads(open(job_path, encoding="utf-8").read())
    reply = os.fdopen(rep_fd, "wb", buffering=0)

    def send(header, payload=b""):
        header = {**header, "bytes": len(payload)}
        reply.write((json.dumps(header) + "\n").encode("utf-8"))
        if payload:
            reply.write(payload)

    try:
        _WARNINGS.clear()
        state = _serve_build(job)
    except BaseException as e:       # noqa: BLE001
        traceback.print_exc()
        send({"op": _SERVE_READY, "ok": False,
              "error": f"{type(e).__name__}: {e}"})
        reply.close()
        return 1

    send({"op": _SERVE_READY, "ok": True,
          "engine": state["scene"].render.engine,
          "device": getattr(getattr(state["scene"], "cycles", None),
                            "device", None),
          "persistent_data": state["scene"].render.use_persistent_data,
          "parts": len(state["parts"]),
          "built": sum(1 for p in state["parts"] if p is not None),
          "warnings": list(_WARNINGS)})

    with os.fdopen(req_fd, "r", encoding="utf-8") as requests:
        for line in requests:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError as e:
                send({"op": "frame", "seq": -1, "ok": False,
                      "error": f"unparseable request: {e}"})
                continue
            if req.get("op") == "quit":
                break
            seq = req.get("seq", -1)
            _WARNINGS.clear()
            try:
                payload = _serve_frame(state, req)
            except BaseException as e:      # noqa: BLE001 — one bad frame is
                # not a dead session: a request naming an impossible resolution
                # or a pose list of the wrong length should cost that frame and
                # leave the scene, which took a second and a half to build,
                # exactly where it was.
                traceback.print_exc()
                send({"op": "frame", "seq": seq, "ok": False,
                      "error": f"{type(e).__name__}: {e}"})
                continue
            send({"op": "frame", "seq": seq, "ok": True,
                  "width": state["scene"].render.resolution_x,
                  "height": state["scene"].render.resolution_y,
                  "warnings": list(_WARNINGS)}, payload)
    # `/dev/shm` is RAM and nothing else reaps it, so a long-lived viewer that
    # never cleaned up would leave a frame's worth of it behind per session.
    for path in state["spool"]:
        try:
            os.unlink(path)
        except OSError:
            pass
    reply.close()
    return 0


# ---------------------------------------------------------------------------
# Viewer-asset bake mode
# ---------------------------------------------------------------------------
#
# Derives a **viewer-only** asset from a part's real mesh and its real shader:
# unwrap a non-overlapping atlas, evaluate the original graph at every surface
# point, write the result into that point's own texel, and export a GLB the
# browser can render locally.
#
# This is the third thing this file does with a shader, and the three must not
# be confused. `_render_frame` renders it (canonical, the judged artifact).
# `_serve` renders it repeatedly for the live viewport. This *samples* it —
# and the sampling is the reason the derived asset is faithful where a
# tileable bake is not.
#
# **Why an atlas rather than the mesh's own UVs.** `_face_uv_mapper` gives each
# face a surface parameterization anchored at the part origin: a tiling
# lattice, deliberately, so a tiled oak floor tiles. That layout overlaps:
# different faces share texels, so baking into it writes them over each other.
# `smart_project` gives a packed atlas.
#
# **Why this preserves object-space shaders, which a tileable bake cannot.**
# A tiled patch only works if appearance is a stationary function of surface
# position, and shader materials often drive from `TexCoord.Object`, a 3D
# function of the part's own space uncorrelated with any UV. An atlas bake never
# asks the question: it evaluates the graph *at the real surface point*, so
# object space is simply an input it reads correctly. A glaze whose bands are
# cut from object-space Z bakes to within a tenth of a level (of 255) of the
# canonical render, where correct flat scalars are off by several levels.
#
# Three things here are load-bearing and silent when wrong; each has its own
# note below: `_bake_pin_uv_reads`, `_bake_carry_scalars`, and the fact that
# `use_persistent_data` must stay OFF in this mode.

#: Bumped when anything below changes the derived asset. Enters the cache key,
#: so a bump re-bakes rather than serving a stale atlas.
BAKE_VERSION = "vb3"

#: Target texel density on the surface, texels per millimeter. Measured: the
#: whole 16-part rag doll lands at 0.372/255 against its canonical render at
#: 3.0, and the error halves for each doubling (vase 0.236 -> 0.118 -> 0.067 at
#: 512 -> 1024 -> 2048). 3.0 is the point where the cost of more texels stops
#: buying anything visible.
BAKE_TEXELS_PER_MM = 3.0
BAKE_MIN_SIZE = 64
BAKE_MAX_SIZE = 2048
#: Container format for the baked maps. WEBP at quality 100 measured
#: **indistinguishable** from PNG on the GLB round trip (0.088/255 either way)
#: for a tenth of the bytes — the ceramic vase goes 17.3 MB -> 1.9 MB. Set
#: `DYCRAFT_BAKE_IMAGE_FORMAT=AUTO` for lossless PNG if a codec is ever
#: suspected; the fidelity test in `tests/test_viewer_bake.py` covers whichever
#: is configured.
BAKE_IMAGE_FORMAT = os.environ.get("DYCRAFT_BAKE_IMAGE_FORMAT", "WEBP")
BAKE_IMAGE_QUALITY = int(os.environ.get("DYCRAFT_BAKE_IMAGE_QUALITY", "100"))

#: Texels of bleed outside each island, so bilinear sampling and the first
#: mip level do not pull the background in across a seam.
BAKE_MARGIN = 16

#: Atlas texels baked together in one Cycles call. Every bake call syncs the
#: scene and compiles its shaders before it samples anything, and that fixed
#: cost, not the sampling, is what a part's maps cost: one call per map per part
#: takes minutes on a CPU for an object of a hundred parts. Parts are therefore baked in
#: batches, one call per map for the whole batch, and a batch is bounded by
#: texels so its images and Cycles's per-texel buffers stay well under a
#: worker's memory: about 100 bytes a texel, under a gigabyte here.
BAKE_BATCH_TEXELS = 8 * 1024 * 1024

#: How far a bake lowers its own CPU priority (see `main`).
BAKE_NICE = 10

#: Sockets that are *baked into maps*: the ones authored shaders drive with
#: nodes. All four are standard glTF PBR channels.
#:
#: **Only a socket a node drives is baked.** One that holds a plain value would
#: bake to an image of that value, so it is carried as the value instead (see
#: `BAKE_CARRY_SOCKETS`, and `_bake_viewer_material` for the base color). Many
#: shader-authored parts drive no socket at all, so this more than halves the
#: texels baked.
BAKE_MAP_SOCKETS = (("Base Color", False), ("Roughness", True), ("Metallic", True))

#: Sockets carried across as **scalars**, because a bake cannot capture a
#: view-dependent lobe and authored shaders do not drive them from nodes. Every entry maps onto a `KHR_materials_*` extension that
#: Blender's glTF exporter writes and three.js's `GLTFLoader` reads.
#:
#: **Dropping these is the one way to make the derived asset actively
#: misleading.** A 95%-transmissive glass baked without `Transmission Weight`
#: renders as an opaque near-white panel, *further from the canonical render
#: than flat scalars are*.
BAKE_CARRY_SOCKETS = (
    "Metallic", "Roughness", "IOR", "Alpha",
    "Coat Weight", "Coat Roughness", "Coat IOR",
    "Sheen Weight", "Sheen Roughness",
    "Transmission Weight", "Specular IOR Level",
    "Anisotropic", "Anisotropic Rotation",
)


#: Suffix of the per-part sidecar carrying the atlas UVs in the STAGED MESH's
#: own corner order. Written for every baked part. The viewer's native
#: playback reads it (`viewer/native_animation.py::atlas`, served as
#: `/native-atlas`): it draws the evaluated mesh rather than the GLB, so it
#: never sees the UVs baked into the GLB.
#:
#: **Why this is exact.** `_bake_part` reads the layer off `obj.data`, which is
#: pre-modifier, so it is 3 UVs per triangle in triangle order — and the OBJ
#: importer preserves the file's face and corner order, verified against
#: `sdk/_deform_eval.load_obj` on 8 parts up to 18 198 triangles (positions,
#: triangle order, and corner order all identical). The triangle vertex-index
#: triples travel too, so the reader can *check* the pairing instead of
#: trusting it; `_uvs_from_rest_obj` makes the same assumption already and
#: guards only on a count.
BAKE_UV_SIDECAR_SUFFIX = "_atlasuv.npz"


def _bake_write_atlas_uvs(obj, layer_name, slug, out_dir):
    """Save `(uv, tri)` for one part's atlas layer. Returns the filename.

    `uv` is `(3T, 2)` float32 in corner order; `tri` is `(T, 3)` int32 of
    vertex indices, which is what makes the pairing verifiable downstream.
    """
    mesh = obj.data
    n_loops = len(mesh.loops)
    uv = np.empty(n_loops * 2, dtype=np.float32)
    mesh.uv_layers[layer_name].data.foreach_get("uv", uv)
    tri = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", tri)
    if n_loops != 3 * len(mesh.polygons):
        # Only reachable if the staged mesh were not triangles. Every staged
        # OBJ on disk is (5 195 646 faces scanned, zero non-triangular), and a
        # wrong-shaped sidecar is worse than none: the reader would pair UVs
        # with the wrong corners and the surface would sample the wrong
        # islands, silently.
        _warn(f"{slug}: mesh is not triangles ({len(mesh.polygons)} faces, "
              f"{n_loops} corners); atlas UVs not written")
        return None
    name = f"{slug}{BAKE_UV_SIDECAR_SUFFIX}"
    np.savez(os.path.join(out_dir, name),
             uv=uv.reshape(-1, 2), tri=tri.reshape(-1, 3))
    return name


def _bake_atlas_uvs(obj, name="dycraft_atlas"):
    """Add a packed, non-overlapping UV layer and make it the bake target."""
    layer = obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active = layer
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    # The bake writes through the ACTIVE RENDER layer, which is also what
    # `_bake_pin_uv_reads` exists to keep the *shader* away from.
    for uvl in obj.data.uv_layers:
        uvl.active_render = (uvl.name == name)
    return name


def _bake_pin_uv_reads(node_tree, original):
    """Repoint every `TexCoord.UV` consumer at the ORIGINAL uv layer by name.

    **The single most dangerous line in this mode.** `ShaderNodeTexCoord.UV`
    returns whichever layer is `active_render`, and baking has just made that
    the atlas — so a UV-driven shader would be handed atlas coordinates and its
    pattern would follow the island layout instead of the surface. A
    `ShaderNodeUVMap` names its layer explicitly and is immune.

    Without it a UV-driven wood pattern bakes several levels (of 255) off the
    canonical render. It affects UV-driven shaders and none of the object-space
    ones, so a spot check on an object-space shader would pass without it,
    which is exactly why it is worth this much comment.
    """
    if not original:
        return 0
    rewired = 0
    for node in list(node_tree.nodes):
        if node.type != "TEX_COORD" or not node.outputs["UV"].links:
            continue
        uvmap = node_tree.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = original
        for link in list(node.outputs["UV"].links):
            to_socket = link.to_socket
            node_tree.links.remove(link)
            node_tree.links.new(uvmap.outputs["UV"], to_socket)
            rewired += 1
    return rewired


def _bake_carry_scalars(bsdf):
    """The unlinked scalar value of every socket a bake cannot capture."""
    out = {}
    for socket in BAKE_CARRY_SOCKETS:
        inp = bsdf.inputs.get(socket)
        if inp is None or inp.links:
            continue
        try:
            out[socket] = float(inp.default_value)
        except TypeError:
            continue        # a color socket; not one of ours
    return out


def _bake_image(name, size, non_color):
    """A bake target. Float + Non-Color for data, so the sRGB trap is avoided.

    A byte image sRGB-ENCODES what is written to it. Baking a roughness or a
    normal into one and then reading it back as data gives nonsense — measured
    at 119/255 of error on a normal map, and it looks like a broken bake rather
    than a color-management mistake.
    """
    img = bpy.data.images.new(name, size, size, alpha=False,
                              float_buffer=non_color)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def _bake_surface_area(obj):
    verts = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    tris = np.array([list(p.vertices) for p in obj.data.polygons
                     if len(p.vertices) == 3])
    if not len(tris):
        return 0.0
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _bake_uv_fill(obj, layer_name):
    """Fraction of the atlas the islands actually cover, for sizing."""
    layer = obj.data.uv_layers[layer_name]
    uvs = np.empty(len(obj.data.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    if len(uvs) % 3:
        return 0.5
    tri = uvs.reshape(-1, 3, 2)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() * 0.5)
    lo, hi = uvs.min(axis=0), uvs.max(axis=0)
    bbox = max(hi[0] - lo[0], 1e-9) * max(hi[1] - lo[1], 1e-9)
    return max(min(area / bbox, 1.0), 0.05)


def _bake_atlas_size(obj, layer_name):
    """Atlas side in texels, from surface area at `BAKE_TEXELS_PER_MM`.

    Sizing by area rather than uniformly is a ~20x saving at no measurable
    fidelity cost: the rag doll's eyes get 128 and its torso 2048, and the whole
    16-part object comes to 6.4 MB of maps instead of 134 MB.
    """
    area = _bake_surface_area(obj)
    if area <= 0.0:
        return BAKE_MIN_SIZE
    side_mm = math.sqrt(area) * 1000.0
    want = side_mm * BAKE_TEXELS_PER_MM / math.sqrt(_bake_uv_fill(obj, layer_name))
    want = max(want, BAKE_MIN_SIZE)
    size = 2 ** int(round(math.log2(want)))
    return int(min(BAKE_MAX_SIZE, max(BAKE_MIN_SIZE, size)))


def _bake_prepare(scene):
    scene.render.bake.use_clear = True
    scene.render.bake.margin = BAKE_MARGIN
    scene.render.bake.use_selected_to_active = False
    # **Off, and it must stay off.** Persistent data serves a stale frame when a
    # material datablock is *replaced* rather than edited — which is exactly
    # what this mode does to every part. Caught by measurement: whole-object
    # error read 29/255 with it on and 0.099/255 with it off.
    scene.render.use_persistent_data = False
    scene.cycles.samples = 1        # EMIT and NORMAL bakes need no sampling


def _bake_select(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def _bake_targets(staged, image_for):
    """Make each staged part's image the active node of its material, for one
    bake call over all of them. Returns the nodes to remove afterwards."""
    added = []
    for st in staged:
        node_tree = st["node_tree"]
        tex = node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image_for(st)
        node_tree.nodes.active = tex
        added.append((node_tree, tex))
    return added


def _bake_socket(staged, socket, non_color):
    """Bake whatever drives `socket` on every staged part, by tapping it into an
    Emission shader, in one call. Each part's map lands in `st["maps"]`.

    A `DIFFUSE`/`COLOR` bake would be the obvious route and is wrong: it
    returns the diffuse albedo, which on a metal is black. Tapping the socket
    into an Emission and baking `EMIT` returns the socket's own value whatever
    the surface is, and works identically for color and scalar sockets. An
    emission bake reads nothing but the surface it is on, so baking the parts
    together gives each the map it would get alone.
    """
    staged = [st for st in staged if socket in st["bake"]]
    if not staged:
        return
    restores = []
    added = []
    try:
        for st in staged:
            node_tree, bsdf = st["node_tree"], st["bsdf"]
            out_node = next(n for n in node_tree.nodes
                            if n.type == "OUTPUT_MATERIAL")
            if not out_node.inputs["Surface"].links:
                raise ValueError(f"{st['name']}: material output has nothing "
                                 "connected to Surface")
            restore = out_node.inputs["Surface"].links[0].from_socket
            emit = node_tree.nodes.new("ShaderNodeEmission")
            restores.append((node_tree, out_node, restore, emit))
            inp = bsdf.inputs[socket]
            if inp.links:
                node_tree.links.new(inp.links[0].from_socket,
                                    emit.inputs["Color"])
            else:
                value = inp.default_value
                try:
                    emit.inputs["Color"].default_value = (
                        value[0], value[1], value[2], 1.0)
                except TypeError:
                    emit.inputs["Color"].default_value = (value, value, value,
                                                          1.0)
            node_tree.links.new(emit.outputs["Emission"],
                                out_node.inputs["Surface"])
            st["maps"][socket] = _bake_image(f"{st['slug']}_{socket}",
                                             st["size"], non_color)
        added = _bake_targets(staged, lambda st: st["maps"][socket])
        bpy.context.scene.cycles.bake_type = "EMIT"
        _bake_select([st["obj"] for st in staged])
        bpy.ops.object.bake(type="EMIT")
    finally:
        for node_tree, tex in added:
            node_tree.nodes.remove(tex)
        for node_tree, out_node, restore, emit in restores:
            node_tree.nodes.remove(emit)
            node_tree.links.new(restore, out_node.inputs["Surface"])


def _bake_normal(staged):
    """Tangent-space normals, including whatever Bump nodes the graph applied,
    for every staged part in one call."""
    staged = [st for st in staged if "Normal" in st["bake"]]
    if not staged:
        return
    for st in staged:
        st["maps"]["Normal"] = _bake_image(f"{st['slug']}_normal", st["size"],
                                           True)
    added = _bake_targets(staged, lambda st: st["maps"]["Normal"])
    bpy.context.scene.render.bake.normal_space = "TANGENT"
    _bake_select([st["obj"] for st in staged])
    try:
        bpy.ops.object.bake(type="NORMAL")
    finally:
        for node_tree, tex in added:
            node_tree.nodes.remove(tex)


def _bake_viewer_material(name, maps, scalars, uv_layer, base_color=None):
    """A plain Principled material sampling the atlas — a glTF-shaped graph.

    Deliberately the *narrow* subset Blender's glTF exporter recognizes: an
    Image Texture straight into a socket, or through a single Normal Map node.
    Anything cleverer exports as a flat approximation with no warning.
    `base_color` is the shader's own color when no node drives it, so nothing
    was baked for it.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    uvm = nt.nodes.new("ShaderNodeUVMap")
    uvm.uv_map = uv_layer
    for socket, image in maps.items():
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        tex.extension = "EXTEND"
        nt.links.new(uvm.outputs["UV"], tex.inputs["Vector"])
        if socket == "Normal":
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.uv_map = uv_layer
            nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            nt.links.new(tex.outputs["Color"], bsdf.inputs[socket])
    for socket, value in scalars.items():
        inp = bsdf.inputs.get(socket)
        # A socket now driven by a baked map must not be overwritten by the
        # scalar the shader happened to leave on it.
        if inp is not None and not inp.links:
            inp.default_value = value
    if base_color is not None and "Base Color" not in maps:
        bsdf.inputs["Base Color"].default_value = base_color
    return mat


def _bake_export_glb(obj, path):
    """One part as a GLB, in the part's own frame and Z-up.

    **`export_yup=False` is not optional.** glTF's convention is Y-up and
    Blender's exporter converts by default, but the viewer's URDF path resolves
    every `<mesh filename>` in dycraft's Z-up frame and applies the z-up
    rotation once, at the robot group. A Y-up GLB would arrive rotated 90
    degrees relative to every staged OBJ beside it.
    """
    obj.matrix_world = __import__("mathutils").Matrix.Identity(4)
    _bake_select([obj])
    kwargs = dict(filepath=path, export_format="GLB", use_selection=True,
                  export_apply=True, export_yup=False,
                  export_image_format=BAKE_IMAGE_FORMAT,
                  export_image_quality=BAKE_IMAGE_QUALITY,
                  export_normals=True,
                  export_tangents=False, export_texcoords=True,
                  export_materials="EXPORT", export_cameras=False,
                  export_lights=False, export_animations=False)
    try:
        bpy.ops.export_scene.gltf(**kwargs)
    except TypeError:
        # The exporter's keyword surface moves between Blender versions. Retry
        # with only the arguments that have been stable, rather than exporting
        # nothing: a rotated or unlit asset is a visible failure, a missing one
        # is a 404 the viewer already handles.
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB",
                                  use_selection=True, export_yup=False)
    return os.path.getsize(path)


def _bake_stage(spec, index, out_dir):
    """Build one part and ready it to bake: its atlas unwrapped and sized, its
    shader's UV reads pinned. Returns the part's record, or None to skip it."""
    obj = _build_part(spec, index)
    if obj is None:
        return None
    # The IR's own part name, which `pack_bake_job` carries explicitly so the
    # manifest can be matched back onto `ir["parts"]`. `spec["name"]` is the
    # world part's `object/part` form and is only a fallback for a caller that
    # did not set one.
    name = spec.get("ir_part") or spec.get("name") or f"part_{index}"
    if not obj.data.materials or not obj.data.materials[0].use_nodes:
        _warn(f"{name}: no node material to bake; viewer asset skipped")
        return None
    node_tree = obj.data.materials[0].node_tree
    bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        _warn(f"{name}: material has no Principled BSDF; viewer asset skipped")
        return None

    original_uv = (obj.data.uv_layers.active.name
                   if obj.data.uv_layers else None)
    scalars = _bake_carry_scalars(bsdf)
    # The maps this part needs: a socket some node drives, and the normal when
    # the graph bends it, through the BSDF's Normal or a bump on the output's
    # Displacement. Anything else is a plain value, carried as one.
    out_node = next((n for n in node_tree.nodes
                     if n.type == "OUTPUT_MATERIAL"), None)
    bake = [socket for socket, _ in BAKE_MAP_SOCKETS
            if bsdf.inputs[socket].links]
    if bsdf.inputs["Normal"].links or (
            out_node is not None and out_node.inputs["Displacement"].links):
        bake.append("Normal")
    base_color = (None if "Base Color" in bake
                  else tuple(bsdf.inputs["Base Color"].default_value))
    atlas = _bake_atlas_uvs(obj)
    size = _bake_atlas_size(obj, atlas)
    pinned = _bake_pin_uv_reads(node_tree, original_uv)
    slug = _bake_slug(name, index)
    # Before the bakes, because it only needs the layer and this keeps the
    # read next to the unwrap that produced it.
    atlas_uv_file = _bake_write_atlas_uvs(obj, atlas, slug, out_dir)
    return {"obj": obj, "name": name, "slug": slug, "node_tree": node_tree,
            "bsdf": bsdf, "atlas": atlas, "size": size, "scalars": scalars,
            "pinned": pinned, "original_uv": original_uv,
            "atlas_uv_file": atlas_uv_file, "bake": bake,
            "base_color": base_color, "maps": {}}


def _bake_maps(staged):
    """Every map of every staged part: one bake call per map for the lot."""
    if not staged:
        return
    for socket, non_color in BAKE_MAP_SOCKETS:
        _bake_socket(staged, socket, non_color)
    _bake_normal(staged)


def _bake_finish(st, out_dir):
    """Give a baked part its viewer material and export it. Returns its manifest
    entry."""
    obj, atlas = st["obj"], st["atlas"]
    # **Drop the original lattice before export, leaving the atlas alone.**
    # It has done its job — `_bake_pin_uv_reads` needed it by name while the
    # bakes ran — and shipping it costs a second `TEXCOORD` set that the
    # exporter then has every texture reference as `texCoord=1`. That works
    # (three.js does support a second set) but it makes the asset depend on it
    # for the base color and normal maps, which is a dependency worth not
    # having for a set nothing samples.
    for uvl in [u for u in obj.data.uv_layers if u.name != atlas]:
        obj.data.uv_layers.remove(uvl)
    obj.data.uv_layers[atlas].active_render = True

    viewer_mat = _bake_viewer_material(f"{st['slug']}_viewer", st["maps"],
                                       st["scalars"], atlas, st["base_color"])
    obj.data.materials.clear()
    obj.data.materials.append(viewer_mat)
    glb = f"{st['slug']}.glb"
    size_bytes = _bake_export_glb(obj, os.path.join(out_dir, glb))
    return {"part": st["name"], "file": glb, "atlas": st["size"],
            "maps": list(st["bake"]),
            "bytes": size_bytes, "uv_pinned": st["pinned"],
            "scalars": st["scalars"], "original_uv": st["original_uv"],
            # Absent when the mesh was not triangles; the deformation path then
            # keeps the staged mesh's own UVs and its preview appearance.
            **({"atlas_uv": st["atlas_uv_file"]} if st["atlas_uv_file"] else {})}


def _bake_release(st):
    """Take an exported part out of the scene, with its maps.

    A part left in the scene is synced again by every later bake call, so a
    long object's bakes would grow with the parts before them, and its maps,
    already written into its GLB, would hold their memory to the end.
    """
    for image in st["maps"].values():
        bpy.data.images.remove(image)
    st["maps"].clear()
    bpy.data.objects.remove(st["obj"], do_unlink=True)


def _bake_part(spec, index, out_dir):
    """Derive one part's viewer asset. Returns its manifest entry, or None.

    The part stays in the scene, so a caller can render the result."""
    st = _bake_stage(spec, index, out_dir)
    if st is None:
        return None
    _bake_maps([st])
    return _bake_finish(st, out_dir)


def _bake_slug(name, index):
    """A filesystem-safe stem. Part names carry `/` on the scene layer."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    return f"{index:03d}_{safe}" if safe else f"{index:03d}_part"


def _bake_run(job_path, out_dir):
    job = json.loads(open(job_path, encoding="utf-8").read())
    _WARNINGS.clear()
    os.makedirs(out_dir, exist_ok=True)
    scene = _reset_scene(job)
    _bake_prepare(scene)
    frame = job["frames"][0]
    entries = []
    batch, texels = [], 0

    def flush():
        _bake_maps(batch)
        for st in batch:
            entries.append(_bake_finish(st, out_dir))
            _bake_release(st)
        batch.clear()
        _purge_orphans()

    for i, spec in enumerate(frame["parts"]):
        st = _bake_stage(spec, i, out_dir)
        if st is None:
            _purge_orphans()
            continue
        # A part with nothing to bake holds no map, so it costs the batch none.
        cost = st["size"] ** 2 if st["bake"] else 0
        if batch and texels + cost > BAKE_BATCH_TEXELS:
            flush()
            texels = 0
        batch.append(st)
        texels += cost
    if batch:
        flush()
    return {"ok": True, "version": BAKE_VERSION, "parts": entries,
            "texels_per_mm": BAKE_TEXELS_PER_MM,
            "device": getattr(getattr(scene, "cycles", None), "device", None),
            "warnings": list(_WARNINGS)}


def _build_blend_run(job_path, out_blend):
    """Build the first frame's parts into a scene and save it as `out_blend`, for `dycraft.export`.

    The parts and materials are the ones a still of the same IR draws; no camera, light, or room shell is added, since
    an export carries the authored geometry only. Each object takes its part's name.
    """
    job = json.loads(open(job_path, encoding="utf-8").read())
    _WARNINGS.clear()
    scene = _reset_scene(job)
    built = 0
    for i, spec in enumerate(job["frames"][0]["parts"]):
        obj = _build_part(spec, i)
        if obj is None:
            continue
        if spec.get("name"):
            obj.name = spec["name"]
            obj.data.name = spec["name"]
        built += 1
    scene.frame_start = scene.frame_end = 0
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=False)
    return {"ok": True, "parts": built, "warnings": list(_WARNINGS)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--bake"]:
        # `--bake <job.json> <out_dir> <result.json>`: derive the viewer-only
        # atlas assets for every part. See `_bake_run`.
        if len(argv) != 4:
            print("usage: _blender_child.py --bake <job.json> <out_dir> "
                  "<result.json>", file=sys.stderr)
            return 2
        _, job_path, out_dir, result_path = argv
        # **A bake yields the CPU.** It runs while someone is looking at the
        # viewer, and on a machine without a GPU Cycles takes every core: at
        # normal priority it made the browser and the viewer's own server
        # stutter for as long as it ran. Niced, it takes what they leave.
        try:
            os.nice(BAKE_NICE)
        except (AttributeError, OSError):
            pass
        try:
            result = _bake_run(job_path, out_dir)
        except BaseException as e:      # noqa: BLE001
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        return 0 if result.get("ok") else 1
    if argv[:1] == ["--build-blend"]:
        # `--build-blend <job.json> <out.blend> <result.json>`: the scene a still would draw, saved as a .blend.
        if len(argv) != 4:
            print("usage: _blender_child.py --build-blend <job.json> <out.blend> <result.json>", file=sys.stderr)
            return 2
        _, job_path, out_blend, result_path = argv
        try:
            result = _build_blend_run(job_path, out_blend)
        except BaseException as e:      # noqa: BLE001
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        return 0 if result.get("ok") else 1
    if argv[:1] == ["--serve"]:
        # `--serve <job.json> <request_fd> <reply_fd>`: build the scene once and
        # answer frame requests. See `_serve`.
        if len(argv) != 4:
            print("usage: _blender_child.py --serve <job.json> <req_fd> "
                  "<rep_fd>", file=sys.stderr)
            return 2
        return _serve(argv[1], int(argv[2]), int(argv[3]))
    if len(argv) != 2:
        print("usage: _blender_child.py <job.json> <result.json>",
              file=sys.stderr)
        return 2
    job_path, result_path = argv
    try:
        result = _run(job_path)
    except BaseException as e:      # noqa: BLE001
        # Always write a result. The parent treats a missing file as "the child
        # died before it could speak" and reports the raw stderr tail, which is
        # a much worse error message than the one we can write here.
        traceback.print_exc()
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
