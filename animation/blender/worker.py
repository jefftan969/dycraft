"""Standalone bpy worker. Invoked only inside the animation namespace."""
import collections
import contextlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import os

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

# The modules beside this file: already on the path as `/tools` in the sandbox, and added here when a test or a
# scorer's child loads this file by path.
if "__file__" in globals() and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
from animation_diff import animation_changed, animation_snapshot
from frame_export import FrameGeometryCache, armature_table, evaluated_frame
from surface_colors import reset_surface_color_cache, surface_color

#: Bound into the `edit`/`probe` namespace when they resolve. Imported
#: defensively because this module is loaded **outside** Blender on purpose --
#: `verify._offline_verify` runs native verification that way, and the snapshot
#: tests exec this file with `bpy` and `mathutils` replaced by stubs. A
#: Blender-only convenience import must not be what makes the worker
#: unimportable offline, so the namespace is assembled from whatever actually
#: arrived rather than assuming all of it did.
_OPTIONAL_MODULES = {}
for _name in ("mathutils", "bpy_extras"):
    try:
        _OPTIONAL_MODULES[_name] = __import__(_name)
    except ImportError:
        pass


def empty(name, parent=None, matrix=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    if matrix is not None:
        obj.matrix_basis = matrix
    return obj


def origin(element):
    if element is None:
        return Matrix.Identity(4)
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    return Matrix.Translation(xyz) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


#: What the layer draws with, and how hard. Diagnostic renders default to
#: `BLENDER_WORKBENCH` at 1280x720, because a diagnostic answers questions about
#: geometry, motion, contact, and framing, none of which shading changes.
#:
#: On a large project, CPU-only, Workbench draws a 720p frame in under a second
#: where EEVEE at 64 samples takes minutes, for evidence it does not improve.
#: Timings are scene-dependent; re-time rather than extrapolate.
#:
#: Workbench has no ray tracing and no soft shadows, so its display shading
#: differs from Eevee/Cycles materials: pass explicit `render_settings` when the
#: judgement is about appearance. Final delivery may also select it outright
#: when that look suits the authored film.
#:
#: Diagnostic settings are read once, at import. Final-film jobs instead carry
#: an explicit render_settings profile, independent of these defaults.
#: The two directories the worker reads and writes. Inside the bwrap sandbox
#: they are the bind targets `/workspace` and `/output` and these defaults are
#: what `runtime.command` mounts, so the sandboxed path is unchanged.
#:
#: They are indirect so the same code can run **outside** the sandbox, which is
#: what an offline caller needs: `verify._offline_verify` already runs native
#: verification that way, and staging a bare agent's own `.blend` needs the
#: evaluation step the same way. Duplicating the frame loop into a second
#: implementation was the alternative, and a second copy of "what evaluated
#: geometry means" is the thing most worth not having twice.
#: Resolved per call, through the module-level `Path`, rather than once at
#: import. Both matter: `test_snapshot_paths_match_requested_frames` loads this
#: module and then monkeypatches `worker.Path`, so a root built at import time
#: is out of reach of that seam and `render_frames` would try to mkdir the real
#: `/output/frames` under test.
def _workspace():
    return Path(os.environ.get("DYCRAFT_BLENDER_WORKSPACE_ROOT") or "/workspace")


def _output():
    return Path(os.environ.get("DYCRAFT_BLENDER_OUTPUT_ROOT") or "/output")

_ENGINE = os.environ.get("DYCRAFT_BLENDER_ENGINE") or "BLENDER_WORKBENCH"
# Only read when DYCRAFT_BLENDER_ENGINE selects Eevee: `render_settings()` applies
# it under BLENDER_EEVEE_NEXT alone, and Workbench's AA is pinned by
# `render_config.WORKBENCH_AA`. Inert under the current default; the env name is
# kept for compatibility.
_SAMPLES = int(os.environ.get("DYCRAFT_BLENDER_EEVEE_SAMPLES") or 64)
_WIDTH = int(os.environ.get("DYCRAFT_BLENDER_WIDTH") or 1280)
_HEIGHT = int(os.environ.get("DYCRAFT_BLENDER_HEIGHT") or 720)

#: Workbench's own antialiasing, deliberately NOT `_SAMPLES`.
#:
#: They are different quantities on different engines: EEVEE's
#: `taa_render_samples` is a temporal accumulation count, Workbench's
#: `render_aa` is an enum ("OFF"/"FXAA"/"5"/"8"/"11"/"16"/"32") whose default
#: is already 8. Mapping one onto the other is not a translation, it is a
#: quality change nobody asked for, and a high setting makes Workbench slower
#: without improving the evidence. Unset therefore means "leave Blender's
#: default alone".
_WORKBENCH_AA = os.environ.get("DYCRAFT_BLENDER_WORKBENCH_AA") or None


#: The recipe the authored blend is stored with, and the one a delivery render
#: uses. `render_config` may differ from it for a review render; this is what
#: gets written into `animation.blend`, so that choosing Workbench for one
#: preview cannot alter the committed artifact or its manifest file hash.
#:
#: **These are `render_config.final_render_settings()`'s defaults, and they
#: have to stay that way** -- `test_the_stored_recipe_is_the_one_delivery_uses`
#: asserts the two agree. The values are restated rather than imported because
#: this module is loaded outside Blender (offline verification, and the
#: snapshot tests exec it with `bpy` stubbed), where the flat `render_config`
#: name does not resolve.
#:
#: Denoised Cycles at 8 samples, because it is both faster and closer to a
#: 256-sample Cycles reference than EEVEE at 64 (SSIM 0.972 at 2.9 s a frame,
#: against 0.925 at 83 s at 720p): EEVEE's error is rasterization bias rather
#: than sampling noise, so its samples buy almost nothing.
_DELIVERY = {"engine": "CYCLES", "width": 1920, "height": 1080, "samples": 8}


def render_config():
    """The render recipe this worker process will actually use.

    One resolved dict, so that what draws and what gets *recorded* cannot
    disagree. The daemon does not share the worker's environment, so a manifest
    built there could claim an engine the worker never used. The worker is the
    only process that knows, so it is the only process that answers, and the
    value travels out through `evaluate`/`render_frames`.
    """
    config = {"engine": _ENGINE, "width": _WIDTH, "height": _HEIGHT,
              "samples": _SAMPLES}
    if _ENGINE == "BLENDER_WORKBENCH" and _WORKBENCH_AA:
        config["workbench_aa"] = _WORKBENCH_AA
    return config


def render_settings(config=None, *, diagnostic=False):
    """Apply a render recipe to the scene. Defaults to this process's own.

    Two callers with two different jobs, which is why both the recipe and the
    profile are parameters rather than defaults. `persist_render_settings`
    passes `_DELIVERY` on the save path, so whatever it sets is **persisted
    into the author's project**. `render_frames` calls this with
    `diagnostic=True`, in a disposable render scene that is never saved.

    The Workbench draw profile is *not* persisted, and this is the only place
    that stages it outside `apply_final_render_settings`: every render applies
    it for itself, so saving it here would only rewrite the author's project.
    `test_native_animation_movie.py::test_native_batch_matches_frame_renders_and_rejects_bad_replay`
    stages `render_aa='8'` with shadows and cavity on and asserts a delivery
    render comes back with the profile's shadows off and the saved cavity
    untouched.
    """
    config = render_config() if config is None else config
    engine = config["engine"]
    s = bpy.context.scene
    s.render.engine = engine
    s.render.resolution_x = config["width"]
    s.render.resolution_y = config["height"]
    s.render.resolution_percentage = 100
    if engine == "BLENDER_EEVEE_NEXT":
        s.eevee.taa_render_samples = config["samples"]
    elif engine == "BLENDER_WORKBENCH":
        # Per-material base color, not a uniform studio gray: a review frame
        # whose parts are all the same color cannot show which one moved.
        # The colors themselves are bound by `apply_display_colors`, which
        # runs per render and never on the path that saves the blend.
        s.display.shading.color_type = "MATERIAL"
        s.display.shading.show_shadows = True
        if diagnostic:
            # One definition of the Workbench cost profile, in
            # `render_config.apply_workbench_profile`, shared with the explicit
            # `apply_final_render_settings` path -- which is where it is
            # measured and documented. Two copies of an FXAA/no-shadow pair is
            # how a draft and an explicit draft came to disagree about what
            # Workbench draws. Diagnostic-only here because this function is
            # also the save path, and the profile must not reach the blend.
            from render_config import apply_workbench_profile
            apply_workbench_profile(s)
        # After the profile, so an explicitly requested AA outranks it: the
        # profile is a cost decision this layer makes, `workbench_aa` is part
        # of the recipe the operator asked for and that `render_config` reports
        # into the manifest. Validated here rather than at read time because
        # the enum is Blender's and only exists inside it.
        aa = config.get("workbench_aa")
        if aa:
            valid = [i.identifier for i in
                     s.display.bl_rna.properties["render_aa"].enum_items]
            if aa not in valid:
                raise ValueError(
                    f"DYCRAFT_BLENDER_WORKBENCH_AA={aa!r} is not one of {valid}")
            s.display.render_aa = aa
    s.render.image_settings.file_format = "PNG"
    return s


def diagnostic_render_settings(scene):
    """What the diagnostic profile actually applied, for the op payload.

    `effective_render_settings` computes this and the delivery path returns
    it; a diagnostic render reports it too, so the cost of a draft can be
    attributed without a `probe`.
    """
    from render_config import effective_render_settings
    return effective_render_settings(scene, None, [])


#: Blender's factory `Material.diffuse_color`. A material still sitting on it
#: has never been given a viewport swatch by anybody.
_DEFAULT_DIFFUSE = (0.8, 0.8, 0.8, 1.0)


def persist_render_settings():
    """Normalize the blend to the delivery recipe before it is saved.

    `initialize`, `edit`, and `update` write `animation.blend`, and until the
    review
    engine became reachable it did not matter which recipe was in force when
    they did. It matters now: an author who previewed one frame in Workbench
    would otherwise commit a blend whose stored engine is Workbench, changing
    the file hash that `make_manifest` records and that `validate_manifest`
    checks. The engine is a per-render choice, so the stored artifact keeps the
    canonical one and every render applies its own on top.

    `preserve_render_settings` on the job opts out, and the animation layer
    always sets it: there, `final_render` deliberately *does* persist the
    film recipe into the blend (`animation/delivery.py` applies it in an
    `edit` op so the saved project carries the delivered resolution), and its
    manifest records `render_settings` read back from the render itself.
    """
    return render_settings(dict(_DELIVERY))


def display_color(material):
    """The color Workbench draws `material` with: what `apply_display_colors` leaves.

    The render-state digest reads this rather than `diffuse_color` itself,
    because a Workbench render syncs the default gray to the base color
    first, and a check and a delivery must digest the same scene.
    """
    if all(abs(a - b) < 1e-4 for a, b in zip(material.diffuse_color, _DEFAULT_DIFFUSE)):
        red, green, blue, _alpha = surface_color(material)
        return (red, green, blue, 1.0)
    return tuple(material.diffuse_color)


def apply_display_colors():
    """Bind each material's surface color to its Workbench viewport swatch.

    **Workbench `MATERIAL` shading reads `Material.diffuse_color`, not the Principled Base Color**, and nothing writes
    `diffuse_color`: `render/_blender_child.py::_build_material` and agent scripts author through the node tree. So
    nearly every material sits at Blender's default 0.8 gray, and Workbench would draw almost the whole scene in one
    flat tone. `surface_color` already resolves the right value for the viewer; this gives the renderer the same.

    **In memory, and only on a render.** `render_settings` also runs on the `initialize`/`edit` path just before
    `save_as_mainfile`, where this would write a review-mode override into the authored blend and every file hash in
    the bundle manifest; `render_frames`, where nothing saves, is the one caller. It gates the call on the engine of
    the scene actually staged, not on this process's `_ENGINE` or the requested recipe, because a final-film request
    chooses its own engine and a Workbench film drawn by a Cycles-default worker still needs its swatches. Cycles and
    EEVEE never sample `diffuse_color`, so for them it is a no-op.

    **Only a swatch still at the factory default is written**: an author who set `diffuse_color` meant it. Most
    materials set the Base Color as a literal, which `surface_color` returns; its walk through a linked Base Color
    covers the rest.
    """
    count = 0
    for material in bpy.data.materials:
        if not all(abs(a - b) < 1e-4
                   for a, b in zip(material.diffuse_color, _DEFAULT_DIFFUSE)):
            continue
        red, green, blue, _alpha = surface_color(material)
        material.diffuse_color = (red, green, blue, 1.0)
        count += 1
    return count


#: Prefix of an imported character's armature object. Generated, like
#: `link/` and `part/`: `update.generated_record` tracks it across source
#: updates.
RIG_PREFIX = "rig/"


def _swap_in_native_rig(proxy, rig, name, link, loaded_actions):
    """Replace an imported character's proxy part with its frozen rig.

    `proxy` is the part `_build_part` made from the IR's bind-shape OBJ, already
    placed. The blend's skinned mesh takes its place -- same world matrix, so
    the part's recorded frame is unchanged -- and its armature is parented to
    the same link at the same matrix, which is the relation the Armature
    modifier was bound in (both carried identity transforms in the blend).
    Moving the instance therefore moves both; posing bones deforms the mesh
    without moving its object frame, which is what the mechanism checks ask.

    Actions come once per blend, whatever the instance count: they are named
    `<object>/<clip>`, and a second append would only add `.001` copies.
    """
    blend = rig["blend"]
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = [rig["armature"], rig["mesh"]]
        dst.actions = ([a for a in src.actions if a not in bpy.data.actions]
                       if blend not in loaded_actions else [])
    loaded_actions.add(blend)
    armature, mesh = dst.objects
    if armature is None or mesh is None or armature.type != "ARMATURE" \
            or mesh.type != "MESH":
        raise ValueError(f"{name}: {blend} does not hold armature "
                         f"{rig['armature']!r} and mesh {rig['mesh']!r}")
    for action in dst.actions:
        if action is not None:
            action.use_fake_user = True
    world = proxy.matrix_world.copy()
    proxy_mesh = proxy.data
    bpy.data.objects.remove(proxy, do_unlink=True)
    if proxy_mesh.users == 0:
        bpy.data.meshes.remove(proxy_mesh)
    for obj in (armature, mesh):
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = link
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = link.matrix_world.inverted() @ world
    for mod in mesh.modifiers:
        if mod.type == "ARMATURE":
            mod.object = armature
    armature.name = RIG_PREFIX + name
    armature["dycraft_rig"] = name
    mesh["dycraft_rig_digest"] = rig["digest"]
    # Appending leaves the source library behind with no users; it would read
    # as a linked dependency outside the workspace (`audit_dependencies`).
    for library in list(bpy.data.libraries):
        if library.users == 0:
            bpy.data.libraries.remove(library)
    # The caller re-derives the part's frame from `matrix_world`, which Blender
    # does not refresh when `matrix_basis` is set. Without this the mesh read
    # back the blend's identity and was re-seated at the world origin while its
    # armature sat on the link -- right only for an instance placed at the
    # origin, which is where one author found it had to keep a bird.
    bpy.context.view_layer.update()
    return mesh


def initialize(spec):
    import _blender_child as render
    # The DELIVERY recipe, not this process's: a project is born with the
    # canonical recipe stored in it, and `persist_render_settings` keeps it
    # there through every later save. Ours' `_ENGINE` default is Workbench,
    # so without this a new project would carry the diagnostic engine.
    s = render._reset_scene(dict(_DELIVERY))
    # An authored `@shader` that raises fails the import, naming itself
    # (`MaterialError`), rather than leaving the part a neutral gray that a
    # build reports as success.
    render._STRICT_MATERIALS = True
    # Zero, so output index and native frame are the same number: every
    # harness index is a zero-based output frame, keyframes and markers are
    # native, and Blender's default start of 1 would put a gratuitous offset
    # between them. An author may still set another start -- every
    # read goes through `info["frame_start"] + i` -- but nothing has to.
    s.frame_start = 0
    s.frame_end = 0
    s["dycraft_scene_ref"] = spec["scene_ref"]
    # `units` says how event and collision-allowance windows are written.
    # Frames, matching every other index on the surface. Absent means seconds,
    # so seeding it here keeps a new project on one unit throughout.
    s["dycraft_declarations"] = json.dumps(
        {"events": [], "collision_allowances": [], "units": "frames"})
    tree = ET.fromstring(spec["urdf"])
    links = {e.get("name"): empty("link/" + e.get("name"))
             for e in tree.findall("link")}
    joints = {}
    fixed_edges = []
    for j in tree.findall("joint"):
        name = j.get("name")
        parent = links[j.find("parent").get("link")]
        child = links[j.find("child").get("link")]
        base = empty("origin/" + name, parent, origin(j.find("origin")))
        motion = empty("joint/" + name, base)
        child.parent = motion
        child.matrix_basis = Matrix.Identity(4)
        # Instance placement joints are animation roots; object mechanisms
        # retain their original frames even when driven by native constraints.
        if name in spec["joints"]:
            fixed_edges.extend([(base.name, parent.name), (child.name, motion.name)])
            if j.get("type") == "fixed":
                fixed_edges.append((motion.name, base.name))
        if j.get("type") in ("revolute", "prismatic"):
            # The joint's coordinate is one native channel of the joint Empty:
            # `rotation_euler[2]` for a revolute joint, `location[2]` for a
            # prismatic one. Its frame is turned so the joint's axis is that
            # Empty's local Z, and the child is turned back, so every world
            # pose is what the URDF says.
            #
            # One native channel rather than a custom property driving the
            # rotation through drivers: drivers are mutable state that
            # `animation_data_clear()` deletes, a rotation-mode switch leaves
            # writing an ignored channel, and a whole-matrix write can turn the
            # wrong way.
            axis = Vector([float(x) for x in j.find("axis").get("xyz").split()]).normalized()
            onto_axis = Vector((0.0, 0.0, 1.0)).rotation_difference(axis).to_matrix().to_4x4()
            base.matrix_basis = base.matrix_basis @ onto_axis
            child.matrix_basis = onto_axis.inverted()
            default = float(spec["joints"][name]["default"])
            motion.rotation_mode = "XYZ"
            if j.get("type") == "revolute":
                motion.rotation_euler = (0.0, 0.0, default)
            else:
                motion.location = (0.0, 0.0, default)
            motion["joint_type"] = j.get("type")
            motion["axis"] = list(axis)
            motion["limits"] = list(spec["joints"][name]["limits"])
            # `axis` is the motion axis in the joint's own frame, which is
            # local Z by construction; `source_axis` is the URDF's, in its
            # origin frame, which is what a source edit can change.
            joints[name] = {"object": motion.name, "origin": base.name,
                            "parent": parent.name, "child": child.name,
                            "type": j.get("type"), "axis": [0.0, 0.0, 1.0],
                            "source_axis": list(axis), "default": default,
                            "limits": spec["joints"][name]["limits"]}
            # Narrower than `limits` where a preloaded asset declares it, and
            # absent otherwise. Carried on the joint rather than in a parallel
            # table so `verify()` reads both bounds off the one record it
            # already walks per frame.
            narrowed = (spec.get("anatomical_limits") or {}).get(name)
            if narrowed:
                joints[name]["anatomical_limits"] = list(narrowed)
    bpy.context.view_layer.update()
    identities = {}
    rig_actions = set()
    for i, part in enumerate(spec["parts"]):
        obj = render._build_part(part, i)
        if obj is None:
            raise ValueError(f"could not import {part['name']}")
        name = part["name"]
        if part.get("native_rig"):
            obj = _swap_in_native_rig(obj, part["native_rig"], name,
                                      links[name.replace("/", "__", 1)],
                                      rig_actions)
        obj.name = "part/" + name
        obj["dycraft_identity"] = name
        world = obj.matrix_world.copy()
        obj.parent = links[name.replace("/", "__", 1)]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = obj.parent.matrix_world.inverted() @ world
        identities[name] = {"object": obj.name, "link": obj.parent.name,
                            "world_matrix": [list(row) for row in world],
                            "local_matrix": [list(row) for row in obj.matrix_basis]}
    cam_spec = spec["cameras"][0]
    for i, wall in enumerate(cam_spec.get("walls", [])):
        obj = render._build_part(wall, i, prefix="room")
        if obj is None:
            raise ValueError("could not import room surface")
        obj["dycraft_room"] = True
    camera = render._make_camera(s)
    render._aim_camera(camera, cam_spec)
    s.camera = camera
    # Lighting LAST, through the renderer's single entry point, so the native
    # project is lit exactly as a still or the viewer of the same scene is:
    # a v2 scene's `@lighting` block (spec["lighting"], run against
    # `run_dir`/`asset_base_dir`), or an unlit frame for a v2 scene that
    # authored none. `refresh={"lighting": True}` rebuilds through here too.
    # After the camera because the authored block is the final word on
    # lights, World, and color management.
    from update import scene_settings
    unlit = scene_settings(s)
    render.LIGHTING_WRITES.clear()
    render._install_frame_lighting(s, spec)
    # What the block set on the scene itself, so `refresh={'lighting': True}`
    # can apply it again: a refresh otherwise brings over lights and World only.
    # What it changed, and what it wrote even where that was the value already
    # there (`_blender_child._Recorded`).
    lighting_settings = {path: value for path, value in scene_settings(s).items()
                         if unlit.get(path) != value or path in render.LIGHTING_WRITES}
    bpy.context.view_layer.update()
    baseline = {"identities": identities, "joints": joints,
                "room": spec.get("room"), "scene_ref": spec["scene_ref"],
                "links": {name: [list(row) for row in obj.matrix_world]
                          for name, obj in links.items()},
                "room_objects": [obj.name for obj in s.objects if obj.get("dycraft_room")],
                "surface_replacements": spec.get("surface_replacements") or {},
                "fixed_edges": [{"child": child, "parent": parent,
                                 "matrix": [list(row) for row in
                                     (bpy.data.objects[parent].matrix_world.inverted() @
                                      bpy.data.objects[child].matrix_world)]}
                                for child, parent in fixed_edges]}
    from update import generated_record
    baseline["generated"] = generated_record(s)
    baseline["lighting_settings"] = lighting_settings
    baseline["world"] = s.world.name if s.world else None
    (_output() / "baseline.json").write_text(json.dumps(baseline, indent=2))
    return {"objects": identities, "joints": joints}


#: Markers this harness owns. An author's own markers keep their own names and
#: are never touched: the harness materializes a projection of what the author
#: declared, and nothing else.
CUT_PREFIX = "dycraft:"


def _plain_property(value):
    """An ID property as plain dicts, lists, and scalars."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "to_list"):
        value = value.to_list()
    if isinstance(value, dict):
        return {key: _plain_property(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_property(item) for item in value]
    return value


def scene_json(name, default):
    """`scene[name]` as data: a JSON string, or the dict or list assigned to it
    directly, which Blender stores as an ID property."""
    raw = bpy.context.scene.get(name)
    if raw is None or raw == "":
        return default
    if not isinstance(raw, str):
        return _plain_property(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"scene[{name!r}] is a string that is not JSON ({error}); "
                         "assign the value itself, or json.dumps of it") from None


def materialise_cuts():
    """Rewrite the camera markers from `scene['dycraft_cuts']`.

    A camera cut is the one thing that cannot stay clip-relative: Blender's
    own depsgraph reads `timeline_markers`, so an absolute frame has to exist
    in the file. Something must write it, and an author writing it by hand
    would have to retime every marker with every curve, which makes retiming
    expensive enough to push authors toward static cameras.

    So the author declares `[{"camera": name, "from": {"clip": ..., "frame": ...}}]`
    and this projects it onto markers at publish. Retiming a clip then moves
    its cut for free.
    """
    scene = bpy.context.scene
    declared = scene_json("dycraft_cuts", [])
    owned = [m for m in scene.timeline_markers if m.name.startswith(CUT_PREFIX)]
    if not declared and not owned:
        return None
    if not isinstance(declared, list):
        raise ValueError("dycraft_cuts must be a list of {camera, from} records")
    spans = clip_spans()
    resolved = []
    for entry in declared:
        if not isinstance(entry, dict) or not isinstance(entry.get("camera"), str):
            raise ValueError("each cut needs a camera name")
        camera = bpy.data.objects.get(entry["camera"])
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"cut names {entry['camera']!r}, which is not a camera")
        at = entry.get("from") or {}
        if not isinstance(at, dict):
            raise ValueError(f"cut to {entry['camera']!r}: 'from' is {{'clip': name, 'frame': "
                             f"offset}} or {{'frame': index}}, not {at!r}")
        if at.get("clip") is not None:
            span = spans.get(at["clip"]) if isinstance(at["clip"], str) else None
            if span is None:
                known = ", ".join(sorted(spans)) or "no NLA strips"
                raise ValueError(f"cut names unknown clip {at['clip']!r}; this project has {known}")
            offset = frame_number(at.get("frame", 0), f"cut to {entry['camera']!r}")
            # Scaled like a state's offset (`verify.clip_offset`); unscaled,
            # a retimed clip put every cut after its first at the wrong frame.
            from verify import clip_offset
            index = span["cut"] + clip_offset(span, max(offset, 0), clip=at["clip"],
                                              subject=f"cut to {entry['camera']!r}")
            if offset < 0 or index > span["frames"][1]:
                raise ValueError(f"cut offset {offset} falls outside clip {at['clip']!r} "
                                 f"{span['frames']}")
        elif at.get("frame") is not None:
            index = frame_number(at["frame"], f"cut to {entry['camera']!r}")
        else:
            raise ValueError("a cut is placed by from={'clip': ...} or from={'frame': ...}")
        resolved.append((index, entry["camera"]))
    for marker in owned:
        scene.timeline_markers.remove(marker)
    for index, name in sorted(resolved):
        marker = scene.timeline_markers.new(f"{CUT_PREFIX}{name}@{index}",
                                            frame=scene.frame_start + index)
        marker.camera = bpy.data.objects[name]
    if resolved and scene.camera is None:
        scene.camera = bpy.data.objects[min(resolved)[1]]
    return [{"frame": index, "camera": name} for index, name in sorted(resolved)]


def frame_number(value, subject):
    """An authored frame or offset as an int, or a refusal naming `subject`."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{subject}: frame must be a number, not {value!r}") from None


def resolve_states(info, states):
    """Where each storyboard state sits, as an absolute output frame index.

    A state either pins an absolute `frame` or names a place inside a clip.
    The clip-relative form is the one that survives retiming: moving or
    scaling the strip moves the state with it, so a review of the pose stays
    a review of the same pose. Bindings live in `scene['dycraft_states']`,
    beside the clips they refer to, because they are authored in the same
    call that makes the clips.
    """
    bindings = scene_json("dycraft_states", {})
    if not isinstance(bindings, dict):
        raise ValueError("scene['dycraft_states'] must map state ids to "
                         "{'clip': name, 'frame': offset}")
    resolved = {}
    for state in states:
        name = state["id"]
        if state.get("frame") is not None:
            resolved[name] = int(state["frame"])
            continue
        bound = bindings.get(name)
        if not isinstance(bound, dict) or "clip" not in bound:
            raise ValueError(
                f"state {name!r} has no frame and no binding; author "
                f"scene['dycraft_states'] = {{{name!r}: {{'clip': ..., 'frame': ...}}}} "
                "in the call that creates the clips, or pin an absolute frame in plan()")
        span = ((info.get("clips") or {}).get(bound["clip"])
                if isinstance(bound["clip"], str) else None)
        if span is None:
            raise ValueError(f"state {name!r} binds to unknown clip {bound['clip']!r}")
        offset = frame_number(bound.get("frame", 0), f"state {name!r}")
        if offset < 0:
            raise ValueError(f"state {name!r} has a negative offset into its clip")
        # Scaled, not just added: a *scale* is a retime on this surface
        # (`test_blender_nla.py::test_a_retime_moves_a_bound_state_and_keeps_
        # the_review_it_already_had` retimes by `strip.scale = 2.0`). One
        # helper for states, cuts, and windows, so the three cannot disagree
        # again; see `verify.clip_offset`.
        from verify import clip_offset
        frame = span["frames"][0] + clip_offset(span, offset, clip=bound["clip"],
                                                 subject=f"state {name!r}")
        if not 0 <= frame < info["n_frames"] or frame > span["frames"][1]:
            raise ValueError(
                f"state {name!r} resolves to output frame {frame}, outside clip "
                f"{bound['clip']!r} {span['frames']} or the timeline")
        resolved[name] = frame
    return resolved


def state_clips(states):
    """Which clip each state is bound into, or None where it pins a frame."""
    bindings = scene_json("dycraft_states", {})
    bindings = bindings if isinstance(bindings, dict) else {}   # `resolve_states` refuses it
    clips = {}
    for state in states:
        bound = bindings.get(state["id"]) if state.get("frame") is None else None
        clips[state["id"]] = bound.get("clip") if isinstance(bound, dict) else None
    return clips


def clip_spans():
    """Every named NLA strip's inclusive output-frame extent.

    Clip identity belongs to the author, not the harness, so the handle is
    whatever the author called the strip and the extent is the union over
    every object carrying a strip by that name -- one clip is normally one
    strip per animated object, all placed together. A union rather than a
    consistency rule: nothing here needs to forbid an arrangement, and the
    contributing objects are reported so an unintended collision of names is
    visible rather than silently spanning the gap between them.

    Extents are inclusive on both ends. Contiguous clips therefore share their
    join frame, which is the frame the boundary pose sits on and belongs to
    both sides of it.
    """
    start = bpy.context.scene.frame_start
    spans = {}
    for obj in bpy.data.objects:
        data = obj.animation_data
        for track in (data.nla_tracks if data else ()):
            for strip in track.strips:
                lo = int(math.floor(strip.frame_start)) - start
                hi = int(math.ceil(strip.frame_end)) - start
                # A cut goes on the first frame the clip actually occupies,
                # so it rounds the other way from the inclusive span. After a
                # scale a strip can start on a fraction, and a cut placed at
                # the floor would show the next shot one frame early, so it
                # rounds with `ceil(... - 1e-9)`.
                cut = int(math.ceil(strip.frame_start - 1e-9)) - start
                span = spans.setdefault(strip.name,
                                        {"frames": [lo, hi], "cut": cut,
                                         "objects": [], "scales": [], "starts": []})
                span["frames"] = [min(span["frames"][0], lo), max(span["frames"][1], hi)]
                span["cut"] = min(span["cut"], cut)
                span["objects"].append(obj.name)
                # Every contributing strip's scale, not one of them. A state
                # bound into this clip is refused when any of them is non-unit
                # (`resolve_states`), and reporting the set is what makes that
                # refusal diagnosable when two objects disagree.
                span["scales"].append(float(strip.scale))
                # Where the action starts: the strip plays it from its first
                # key, so clip time is counted from there (`clip_offset`).
                span["starts"].append(float(strip.action_frame_start))
    for span in spans.values():
        span["objects"] = sorted(set(span["objects"]))
        span["scales"] = sorted({round(v, 6) for v in span["scales"]})
        span["starts"] = sorted({round(v, 6) for v in span["starts"]})
    return spans


def clip_frames(info, name):
    """The output indices a named clip covers, clamped to the timeline."""
    span = (info.get("clips") or {}).get(name)
    if span is None:
        known = sorted(info.get("clips") or {})
        raise ValueError(f"unknown clip {name!r}; this project has "
                         + (", ".join(known) if known else "no NLA strips"))
    lo = max(0, span["frames"][0])
    hi = min(info["n_frames"] - 1, span["frames"][1])
    if hi < lo:
        raise ValueError(f"clip {name!r} lies outside the output timeline")
    return list(range(lo, hi + 1))


def timeline():
    s = bpy.context.scene
    fps = s.render.fps / s.render.fps_base
    if not math.isfinite(fps) or fps <= 0 or s.frame_end < s.frame_start:
        raise ValueError("invalid scene frame range or effective FPS")
    return {"fps": fps, "n_frames": s.frame_end - s.frame_start + 1,
            "frame_start": s.frame_start, "blender_version": bpy.app.version_string,
            # The film's own frame size, which a bare delivery keeps the aspect of.
            "resolution": [s.render.resolution_x, s.render.resolution_y],
            "declarations": scene_json("dycraft_declarations", {}),
            "scene_ref": s.get("dycraft_scene_ref"),
            "clips": clip_spans(),
            "collision_objects": sorted({str(o.get("dycraft_identity") or o.name)
                                         for o in s.objects
                                         if o.type in ("MESH", "CURVE", "SURFACE", "FONT", "META")})}


def audit_dependencies():
    """Every non-packed dependency must travel with the authoritative blend.

    Split from `audit_bakes` because the two answer different questions and one
    day will want different moments. This one is about what the saved blend
    will still need on disk; the bake check is about whether a *check* can
    trust the caches it finds. Keeping them separate also means an attempt to
    move either earlier does not silently drag the other along -- see the note
    at the save path for why moving this one there does not work yet.
    """
    external = {}
    for library in getattr(bpy.data, "libraries", ()):
        external.setdefault(library.filepath, "linked library")
    # Looked up by name so a Blender build (or a test's fake `bpy`) that does
    # not carry one of these collections is not an AttributeError.
    for attribute, kind in (("images", "image"), ("sounds", "sound"),
                            ("fonts", "font"), ("cache_files", "cache file"),
                            ("movieclips", "movie clip"), ("volumes", "volume")):
        for item in getattr(bpy.data, attribute, ()):
            filepath = getattr(item, "filepath", "")
            # `<builtin>` is Blender's own bundled font; a packed file travels
            # inside the blend and is not a dependency by definition.
            if (not filepath or filepath.startswith("<")
                    or getattr(item, "packed_file", None) is not None):
                continue
            external.setdefault(filepath, kind)
    # Resolved, not raw: `make_paths_relative()` on the save path rewrites a
    # weak reference to `//../donor.blend`, and `blend_paths(absolute=True)`
    # hands back the absolute form of the same path.
    provenance = set()
    for name in ("objects", "meshes", "materials", "node_groups", "actions",
                 "images", "armatures", "worlds", "cameras", "lights", "texts"):
        for item in getattr(bpy.data, name, ()):
            reference = getattr(item, "library_weak_reference", None)
            if reference is not None:
                provenance.add(Path(bpy.path.abspath(reference.filepath)).resolve())
    for raw in bpy.utils.blend_paths(absolute=True, packed=False):
        resolved = Path(bpy.path.abspath(raw)).resolve()
        if resolved.is_relative_to(_workspace()) and resolved.exists():
            continue
        # Which datablock is actually asking for this path decides whether it
        # is a dependency at all. `blend_paths()` answers a wider question than
        # this audit asks: it also reports every `library_weak_reference`, which
        # is Blender's *provenance note* recording where an appended datablock
        # was copied from. The data is fully local; the donor can be deleted and
        # the blend still opens identically.
        #
        # Measured on 4.5.13, appending one object out of a donor blend: four
        # reported paths, of which one is the (transient) library entry and
        # three are provenance notes on the mesh and its two materials -- all at
        # `users == 1`, and the notes **survive save, `pack_all()`,
        # `make_paths_relative()`, and reopen**, rewritten to `//../donor.blend`.
        # Packing cannot absorb them and Python cannot clear them
        # (`LibraryWeakReference` is read-only), so advice to copy or pack them
        # would be unachievable; they are skipped as provenance instead.
        matched = next((kind for path, kind in external.items()
                        if Path(bpy.path.abspath(path)).resolve() == resolved), None)
        if matched is None:
            if resolved in provenance:
                continue
            # Neither a holder this function enumerates nor a provenance note:
            # an unrecognized category, reported rather than assumed benign, so
            # a datablock type added to Blender later cannot slip past.
            matched = "unrecognized reference"
        # `/workspace` rather than `_workspace()` in a sandbox with its own
        # filesystem: the same directory answers to both names there -- a real
        # bind in a job, a symlink to the project in a resident session --
        # while `_workspace()` prints `/native/work` resident and `/workspace`
        # as a job, so one failure told authors two different directories to
        # copy into. With host paths (`runtime.sandbox_mode`) there is no
        # `/workspace`, and the project's own path is the one to name.
        where = "/workspace" if not os.environ.get("DYCRAFT_BLENDER_SANDBOX_MODE") else str(_workspace())
        raise ValueError(
            f"Unpacked dependency outside workspace or missing: {raw} "
            f"({matched}); copy it into {where} or pack it")


#: Modifiers whose transient state lives in a Blender *point cache* -- held in
#: memory unless baked to disk, so a reopened blend starts from nothing.
#: `FLUID` is deliberately absent: Mantaflow has no point cache at all (checked
#: on 4.5.13: the modifier has no `point_cache` attribute and its
#: `domain_settings` expose no `is_baked`), so it is audited off its own
#: on-disk cache below. `DYNAMIC_PAINT` is absent because it bakes per-canvas
#: rather than per-modifier and nothing in this repo authors one yet.
_POINT_CACHE_MODIFIERS = ("CLOTH", "SOFT_BODY")


def _needs_bake(system):
    """True for a particle system that actually simulates something.

    Hair is the exception worth making: a static hair system is generated, not
    solved, and its point cache stays unbaked forever. Only hair with dynamics
    enabled has history to lose. An emitter always does.

    Every other type (`FLIP`, `SPRAY`, `FOAM`, `BUBBLE`, `TRACER`, and their
    combinations) is a system Mantaflow attaches to a liquid domain. Its data
    lives in the domain's own cache, so its point cache reads unbaked even
    after a successful bake; the fluid check below audits that cache instead.
    """
    settings = getattr(system, "settings", None)
    kind = getattr(settings, "type", "EMITTER")
    if kind == "EMITTER":
        return True
    if kind == "HAIR":
        return bool(getattr(settings, "use_hair_dynamics", False))
    return False


def _fluid_domains():
    """`(object, domain_settings)` for every Mantaflow domain in the file."""
    for obj in getattr(getattr(bpy, "data", None), "objects", ()):
        for modifier in getattr(obj, "modifiers", ()):
            if (getattr(modifier, "type", None) == "FLUID"
                    and getattr(modifier, "fluid_type", None) == "DOMAIN"):
                settings = getattr(modifier, "domain_settings", None)
                if settings is not None:
                    yield obj, settings


def simulated_objects():
    """Names of the objects whose evaluated state comes from a simulation cache.

    What a replay refusal may tell the author to bake: rigid bodies, cloth, and
    soft body, simulating particle systems, fluid domains, and geometry-node
    simulation zones. Anything else that differs on replay is not a bake
    problem, so the hint is given only for these.
    """
    names = set()
    for obj in getattr(getattr(bpy, "data", None), "objects", ()):
        if getattr(obj, "rigid_body", None) is not None:
            names.add(obj.name)
        for modifier in getattr(obj, "modifiers", ()):
            kind = getattr(modifier, "type", None)
            if (kind in _POINT_CACHE_MODIFIERS or kind == "FLUID"
                    or (kind == "NODES" and _simulation_tree(getattr(modifier, "node_group", None)))):
                names.add(obj.name)
        if any(_needs_bake(system) for system in getattr(obj, "particle_systems", ()) or ()):
            names.add(obj.name)
    return names


def _simulation_tree(tree, seen=None):
    """Does this node tree, or a group inside it, hold a simulation zone?"""
    seen = set() if seen is None else seen
    if tree is None or tree.as_pointer() in seen:
        return False
    seen.add(tree.as_pointer())
    return any(node.bl_idname in ("GeometryNodeSimulationInput", "GeometryNodeSimulationOutput")
               or _simulation_tree(getattr(node, "node_tree", None), seen)
               for node in tree.nodes)


def audit_bakes():
    """Transient physics caches do not survive reopening, so a check needs them baked.

    All five solvers an author can reach. Rigid body keeps one cache for the whole scene on
    `scene.rigidbody_world`; particles keep theirs on the *system* rather than
    on the modifier that carries it; cloth and soft body keep theirs on the
    modifier. Fluid keeps none of the above and is handled separately.

    **Why this refuses rather than re-solving.** Measured on 4.5.13: an
    *unbaked* rigid body and an unbaked cloth both re-solve correctly after a
    reopen when the frames are stepped in order -- the cube falls 5.000 to
    3.967 over twelve frames -- and `requires_preroll` already forces that
    sequential path for every system named here. So the simulation is not lost,
    and this gate is not protecting against losing it. It protects
    `evaluation.check_replay_frame`, which SHAs evaluated geometry and demands
    the render reproduce the checked frame bit-exactly. Two independent
    re-solves are only bit-equal if nothing about the evaluation order differs
    between them, and a baked cache is the one way to stop asking.

    That makes it a deliberately stricter gate than correctness alone needs: a
    run that relied on re-solving now has to bake. The alternative is a class
    of failure that surfaces as an opaque replay mismatch at delivery.
    """
    unbaked = []
    for obj in getattr(getattr(bpy, "data", None), "objects", ()):
        for modifier in getattr(obj, "modifiers", ()):
            if getattr(modifier, "type", None) not in _POINT_CACHE_MODIFIERS:
                continue
            cache = getattr(modifier, "point_cache", None)
            if cache is not None and not getattr(cache, "is_baked", False):
                unbaked.append(f"{obj.name}/{modifier.name}")
        for system in getattr(obj, "particle_systems", ()) or ():
            if not _needs_bake(system):
                continue
            cache = getattr(system, "point_cache", None)
            if cache is not None and not getattr(cache, "is_baked", False):
                unbaked.append(f"{obj.name}/{getattr(system, 'name', 'particles')}")
    # One cache for every rigid body in the scene, which is why it is named for
    # the scene and not for an object: baking any one of them bakes all.
    # `bpy` itself is looked up by name at every hop, for the reason
    # `audit_dependencies` gives: a test's fake `bpy` need not carry `context`,
    # and a missing collection must not read as a missing bake.
    scene = getattr(getattr(bpy, "context", None), "scene", None)
    world = getattr(scene, "rigidbody_world", None)
    if world is not None:
        cache = getattr(world, "point_cache", None)
        if cache is not None and not getattr(cache, "is_baked", False):
            unbaked.append("scene/rigidbody_world")
    if unbaked:
        raise ValueError(
            f"Bake {', '.join(unbaked)} before checking; transient physics "
            f"caches do not survive reopening. Bake a point cache with "
            f"`bpy.ops.ptcache.bake(bake=True)` under "
            f"`bpy.context.temp_override(scene=..., point_cache=...)`")

    for obj, settings in _fluid_domains():
        raw = getattr(settings, "cache_directory", "") or ""
        if not raw:
            raise ValueError(
                f"Fluid domain {obj.name} has no cache directory; set "
                f"`domain_settings.cache_directory` to a path under the "
                f"workspace and bake with `bpy.ops.fluid.bake_all()`")
        resolved = Path(bpy.path.abspath(raw)).resolve()
        # Where the cache lands is decided when the domain is created, not when
        # the blend is saved. Measured: adding a domain to an *already saved*
        # blend gives `//cache_fluid_<hash>`, inside the workspace, which is
        # the harness's own order and is safe. Adding one before the first save
        # pins it to Blender's temp dir and then rewrites it as
        # `//../blender_XXXX/cache_fluid_<hash>` -- outside the workspace, on
        # the sandbox's tmpfs, gone with the job. It bakes and reads back
        # perfectly inside one call, so nothing fails until the next one.
        if not resolved.is_relative_to(_workspace()):
            raise ValueError(
                f"Fluid domain {obj.name} caches to {resolved}, outside the "
                f"workspace; it will not survive this job. Set "
                f"`domain_settings.cache_directory` to a workspace-relative "
                f"path (save the blend before adding the domain and Blender "
                f"picks one) and re-bake")
        # No `is_baked` flag exists to ask, so the cache itself is the answer:
        # Mantaflow writes `config/`, `data/`, and `mesh/` when it bakes, and
        # does not create the directory at all until then.
        if not (resolved / "data").is_dir() or not any((resolved / "data").iterdir()):
            raise ValueError(
                f"Bake fluid domain {obj.name} before checking; "
                f"{resolved} holds no baked data. Make the domain the active "
                f"object and call `bpy.ops.fluid.bake_all()`")



def prepare_evaluation():
    # Evaluate the geometry the final render sees, not cheaper viewport LODs.
    for obj in bpy.context.scene.objects:
        for modifier in obj.modifiers:
            modifier.show_viewport = modifier.show_render
            if modifier.type in ("SUBSURF", "MULTIRES"):
                modifier.levels = modifier.render_levels
            if modifier.type == "NODES" and modifier.node_group:
                if any(node.bl_idname == "GeometryNodeIsViewport" for node in modifier.node_group.nodes):
                    raise ValueError("Bake viewport-dependent geometry nodes before checking so evaluated geometry matches the final render")


def requires_preroll(scene):
    """Keep sequential evaluation for scenes containing history-dependent systems.

    Even baked physics keeps the conservative path: cache completeness and
    simulation-zone bake coverage are not inferred from a single baked flag.
    """
    if scene.rigidbody_world is not None:
        return True
    seen = set()
    for obj in bpy.data.objects:
        if obj.particle_systems:
            return True
        for modifier in obj.modifiers:
            if modifier.type in ("CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "PARTICLE_SYSTEM"):
                return True
            if modifier.type == "NODES" and _simulation_tree(modifier.node_group, seen):
                return True
    return False


def seek_output_frame(frame):
    """Put the scene on an output frame, pre-rolling only when history demands it.

    Same rule as `evaluate`: a stateless scene jumps straight there, a scene
    with rigid bodies, particles, cloth, or simulation nodes is stepped from the
    start so its caches are the ones that frame actually has.
    """
    info = timeline()
    if type(frame) is not int or not 0 <= frame < info['n_frames']:
        raise ValueError('frame must be a zero-based output frame index inside the timeline')
    for _ in visit_frames([frame], info):
        pass
    return info


def compile_probe(code):
    """A probe as either an expression or a statement block.

    `eval` mode first, because the answer to "where is this object" is an
    expression and returning its value needs no convention. A probe that
    needs a loop, an assignment, or a `try` is a `SyntaxError` there, so
    statement mode covers it. Restricting probes to expressions would buy
    nothing -- `eval` calls whatever it likes, and probe workspaces are
    discarded rather than published -- so it would filter convenience, not
    mutation.

    Statement mode uses `result`, the same name `edit` uses, so one
    convention covers both ops. The eval-mode error is what surfaces when
    neither mode compiles: a malformed expression should be reported as one
    and not as a confusing statement-block failure.
    """
    try:
        return compile(code, "<probe>", "eval"), True
    except SyntaxError as expression_error:
        try:
            return compile(code, "<probe>", "exec"), False
        except SyntaxError:
            raise expression_error from None


class AuthorCodeError(Exception):
    """Raised by code the agent wrote; the message is its traceback, from its own frames."""

    failure_class = "author_code"


@contextlib.contextmanager
def authored():
    """Run code the agent wrote, so that what it raises is reported as the agent's to fix.

    The traceback starts at the agent's first frame and keeps everything below
    it, the harness helpers it called included (`key_joint` given a bad name is
    the call's mistake); the frames above it are this worker's and say nothing
    the agent can act on. Anything raised outside this block is the harness's.
    """
    try:
        yield
    except Exception as error:
        here, tb = sys._getframe().f_code.co_filename, error.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename == here:
            tb = tb.tb_next
        raise AuthorCodeError("".join(
            traceback.format_exception(type(error), error, tb)).rstrip() + _host_path_hint(error)) from error


def _host_path_hint(error) -> str:
    """A line for agent code that named a path only a sandbox with its own filesystem has, where Blender runs with
    host paths (`runtime.sandbox_mode`): what that path is here."""
    if not os.environ.get("DYCRAFT_BLENDER_SANDBOX_MODE") or not isinstance(error, OSError):
        return ""
    name = str(getattr(error, "filename", "") or "")
    mocap = os.environ.get("DYCRAFT_MOCAP_ROOT")
    for inside, real in (("/workspace", str(_workspace())), ("/output", str(_output())), ("/references/mocap", mocap)):
        if real and (name == inside or name.startswith(inside + "/")):
            use = (f"use paths relative to the working directory, {_workspace()}" if inside != "/references/mocap" else
                   "mocap_retarget and dog_retarget accept /references/mocap/... as it is")
            return (f"\n(Blender runs without a private filesystem on this machine, so {inside} does not exist here; "
                    f"it is {real}. {use[0].upper()}{use[1:]}.)")
    return ""


#: How much of what the agent's code printed comes back with its reply.
PRINTED_LIMIT = 4000


@contextlib.contextmanager
def printed_output():
    """Capture what the agent's code prints, kept beside the result as
    `printed.txt` (its last `PRINTED_LIMIT` characters), whether or not it raised."""
    import io
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            yield
    finally:
        if buffer.getvalue():
            (_output() / "printed.txt").write_text(buffer.getvalue()[-PRINTED_LIMIT:])


def failure(error):
    """`result.json` for a request that raised: the class the reply carries, and the
    whole traceback for the log. A `ValueError` outside the agent's code is a
    refusal, whose message is written for the agent; anything else is a fault."""
    kind = getattr(error, "failure_class", None) or (
        "rejection" if isinstance(error, ValueError) else "harness")
    message = str(error) if kind != "harness" else f"{type(error).__name__}: {error}"
    return {"ok": False, "error": message, "failure_class": kind,
            "traceback": traceback.format_exc()}


def run_probe(code, namespace):
    """Evaluate a compiled probe, returning its value under either mode."""
    body, is_expression = code
    if is_expression:
        return eval(body, namespace)
    exec(body, namespace)
    return namespace.get("result")


def probe_frames(job, namespace):
    """Evaluate one expression at each of several output frames, in one pass.

    Frames are visited as every loop visits them (`visit_frames`): jumped to,
    or stepped through from the start where the scene's history demands it.
    The launch is the cost, not the eval, so reading a curve over many
    frames takes one launch rather than one per frame.

    A frame whose expression raises returns its error in place of a value
    rather than losing the frames that worked. An expression that answers at
    frame 0 and fails at 900 is itself the finding, and which frame it turned
    on is the part worth knowing.
    """
    info = timeline()
    frames = job["frames"]
    if (not frames or any(type(i) is not int or not 0 <= i < info["n_frames"] for i in frames)
            or any(b <= a for a, b in zip(frames, frames[1:]))):
        raise ValueError("probe frames must be increasing zero-based output indices "
                         "inside the timeline")
    with authored():
        code = compile_probe(job["code"])
    rows = []
    for i in visit_frames(frames, info):
        try:
            rows.append({"frame": i, "value": run_probe(code, namespace)})
        except Exception as error:                   # noqa: BLE001 - reported, not raised
            rows.append({"frame": i, "error": f"{type(error).__name__}: {error}"})
    return rows


def visit_frames(frames, info=None):
    """Put the scene on each of `frames` in time order, once each, yielding it.

    The one rule every frame loop follows: a scene with no history
    jumps straight to each requested frame; one with rigid bodies, particles,
    cloth, or simulation nodes (`requires_preroll`) is stepped through every
    frame from the start, so its caches are the ones those frames have.

    Walking every frame between the first and the last wanted would make one
    late frame cost the whole stretch before it. A jump and a walk evaluate the
    same scene: they differ only in the sign of stored zeros, which
    `evaluation` canonicalizes.
    """
    info = info or timeline()
    scene = bpy.context.scene
    wanted = sorted(set(frames))
    if not wanted:
        return
    if requires_preroll(scene):
        targets, keep = range(0, wanted[-1] + 1), set(wanted)
    else:
        targets, keep = wanted, None
    if targets[0] > 0:
        # Start the dependency graph where a walk would have, then jump.
        scene.frame_set(info['frame_start'])
    for i in targets:
        scene.frame_set(info['frame_start'] + i)
        if keep is None or i in keep:
            yield i


def keep_declared_units():
    """Write `units: 'frames'` back into declarations rewritten without it.

    An animation project's windows are output frames, and its declarations are
    seeded saying so (`initialize`). A record written back without `units`
    read as seconds, the native layer's unit, and moved every window by a
    factor of the frame rate without failing. Only where the design baseline
    marks the animation layer: the native layer documents seconds and writes no
    `units`.
    """
    scene = bpy.context.scene
    if (not scene.get("dycraft_declarations")
            or not (_workspace() / "design_state" / "baseline.json").is_file()):
        return
    try:
        declarations = scene_json("dycraft_declarations", {})
    except ValueError:
        return                  # `declaration_warnings` names it
    if isinstance(declarations, dict) and "units" not in declarations:
        scene["dycraft_declarations"] = json.dumps({**declarations, "units": "frames"})


def declaration_warnings():
    """Problems in `scene['dycraft_declarations']`, reported by the call that wrote them.

    `verify()` collects them too, but only at the next check, so without this a
    bad record would be discovered however long after it was written. Warned
    rather than refused, like the
    other notes here: `verify()` already skips a bad allowance and fails a
    bad event, so the save itself is safe, and the author learns it now.

    Needs the design's baseline, which the animation layer keeps in the
    workspace. Without it -- the native layer -- nothing is said here and
    `verify()` remains the first reader.
    """
    path = _workspace() / "design_state" / "baseline.json"
    if not bpy.context.scene.get("dycraft_declarations") or not path.is_file():
        return []
    from verify import validate_declarations
    try:
        declarations = scene_json("dycraft_declarations", {})
    except ValueError as error:
        return [{"kind": "declarations",
                 "reason": f"scene['dycraft_declarations'] is not valid JSON: {error}"}]
    try:
        info = timeline()
    except ValueError:
        # A timeline this call left invalid is `verify()`'s to report; a
        # warning must never be what fails a save.
        return []
    return validate_declarations(declarations, json.loads(path.read_text()), info)


#: How a phase spells "how far along": a count of steps done against a count
#: of steps. The first pair present wins. `frame`, where a phase carries one,
#: is the absolute output frame it is at and never a counter: read as one, a
#: check of frames 500-600 was 501 steps into a 101-step job and projected
#: zero seconds left from its first frame.
_PROGRESS_COUNTERS = (("intervals_done", "interval_count"), ("done", "total"))


def write_progress(output, payload, started, stamps):
    """A progress counter plus the two rates and a finish time, atomically.

    `{"frame", "total"}` alone would leave the caller to do this arithmetic,
    and a projection from the frame counter alone comes in optimistic. Polling
    is much of an author's job control, so the estimate is the product.

    Both rates, deliberately. The trailing window (ten steps, or the whole run
    if shorter) tracks a box whose load changes under the render; the run mean
    is what a total-cost decision wants. A large divergence between them is
    itself the signal -- it says the ETA is about to move -- and reporting only
    one hides it.

    Write-temp-then-replace for all three callers, so a poll landing mid-write
    never reads a truncated file.

    `stamps` is the caller's own dict, mutated here and keyed by phase, so a
    run that moves from evaluation into penetration diagnostics times the new
    phase rather than averaging across the boundary: both rates are the
    phase's own, timed from when the phase before it last reported (the job's
    start, for the first), rather than dividing the whole job's time by the
    steps of the current phase. A payload with no recognized counter still
    gets written, with `elapsed_s` and no rates, which is the honest answer
    rather than a fabricated one.
    """
    now = time.time()
    payload = {**payload, "started_at": started, "elapsed_s": round(now - started, 2)}
    pair = next(((d, t) for d, t in _PROGRESS_COUNTERS if d in payload and t in payload), None)
    if pair is not None:
        done, total = payload[pair[0]], payload[pair[1]]
        # `(time, steps done)` points, the first being the phase's start; a
        # report that did not advance the count adds none.
        window = stamps.setdefault(payload.get("phase", ""), [(stamps.get(None, started), 0)])
        if done != window[-1][1]:
            window.append((now, done))
            del window[1:-11]
        # When the phase first reported a completed step. A render's first frame
        # carries the scene's synchronization, so a price taken over few frames
        # is honest only from the second frame on; the first is reported apart.
        first_key = ("first_step", payload.get("phase", ""))
        if done and first_key not in stamps:
            stamps[first_key] = (now, done)
        origin = window[0][0]
        mean = (now - origin) / max(1, done)
        first, last = window[1 if len(window) > 2 else 0], window[-1]
        recent = ((last[0] - first[0]) / (last[1] - first[1])) if last[1] > first[1] else mean
        remaining = recent * max(0, total - done)
        payload.update(seconds_per_step_recent=round(recent, 4),
                       seconds_per_step_mean=round(mean, 4),
                       projected_remaining_s=round(remaining, 1),
                       projected_finish=now + remaining)
        step_at, step_count = stamps.get(first_key, (None, 0))
        if step_at is not None and done > step_count:
            payload["seconds_per_step_after_first"] = round(
                (now - step_at) / (done - step_count), 4)
        if step_at is not None and step_count == 1:
            payload["first_step_s"] = round(step_at - origin, 2)
    stamps[None] = now
    temp = Path(output) / "progress.tmp"
    temp.write_text(json.dumps(payload))
    temp.replace(Path(output) / "progress.json")


def frame_selection_error(frames, info, *, op):
    """Name the offending index and the timeline, which is what reads it.

    The index requested and `n_frames` are the two numbers needed to act on
    it. It is usually an author's only evidence that an *earlier* edit did not
    land, such as a `blender()` call that failed to set `frame_end` and left
    the scene on Blender's default range.
    """
    total = info["n_frames"]
    if not frames:
        return ValueError(
            f"{op} selected no frames; the timeline is [0, {total - 1}] "
            f"({total} frames)")
    bad = [i for i in frames if type(i) is not int or not 0 <= i < total]
    # Capped: a whole-timeline call against a stale `n_frames` would otherwise
    # put every index it asked for into the error string.
    shown = ", ".join(repr(i) for i in bad[:5])
    if len(bad) > 5:
        shown += f", +{len(bad) - 5} more"
    return ValueError(
        f"frame index outside output timeline: requested {shown}; the timeline "
        f"is [0, {total - 1}] ({total} frames)")


def place_states(info, states):
    """`(state_frames, unresolved)`: the output frame each state sits at now.

    Resolved here, the one place that reads the NLA. A state that does
    not resolve is named rather than guessed: its board's frame is where it was
    drawn, not where it is now.
    """
    state_frames, unresolved = {}, {}
    for state in states:
        try:
            state_frames.update(resolve_states(info, [state]))
        except ValueError as error:
            unresolved[state['id']] = str(error)
            continue
        frame = state_frames[state['id']]
        if not 0 <= frame < info['n_frames']:
            del state_frames[state['id']]
            unresolved[state['id']] = (
                f"pinned to output frame {frame}, past the timeline "
                f"[0, {info['n_frames'] - 1}]; the film is shorter than when it was planned")
    return state_frames, unresolved


def evaluate(job):
    from evaluation import FORMAT, MeshWriter, write_mesh, world_geometry, write_signed_record
    from verify import aggregate, frame_indices, measure
    if job.get('place_only'):
        # Where the states sit and nothing else: what a span edit needs.
        # Evaluating the state frames too would, on a scene with a simulation,
        # step it from frame 0 to the last state before every span edit, for
        # frames it then discarded.
        info = {**timeline(), 'render': render_config()}
        info['state_frames'], unresolved = place_states(info, job['states'])
        info.update(frames=[], requires_preroll=bool(requires_preroll(bpy.context.scene)),
                    **({'unresolved_states': unresolved} if unresolved else {}))
        return info
    prepare_evaluation()
    info = timeline()
    info['format'] = FORMAT
    # The worker is the only process that knows its own resolved recipe; see
    # `render_config`. `make_manifest` reads it from here.
    info['render'] = render_config()
    out = _output() / 'evaluation'
    out.mkdir(exist_ok=True)
    frames = job.get('frames')
    if frames is None and job.get('clip') is not None:
        frames = clip_frames(info, job['clip'])
    if frames is None:
        frames = frame_indices(info, job.get('start'), job.get('end'),
                               seconds=job.get('seconds'))
    if not frames or any(type(i) is not int or i < 0 or i >= info['n_frames'] for i in frames):
        raise frame_selection_error(frames, info, op='evaluate')
    # `visit_frames` walks them in order, once each, and what it reports is
    # compared with this list at the end.
    frames = sorted(set(frames))
    known, tables, digests = set(), set(), {}
    geometry_cache = FrameGeometryCache()
    writer = MeshWriter()
    first = 0 if requires_preroll(bpy.context.scene) else min(frames)
    info['evaluation_start_frame'] = first
    # Reported so the caller can decide whether frames are independent enough to
    # be checked in parallel: a scene with rigid bodies, particles, cloth, or
    # simulation nodes has to be stepped from the start and cannot be split.
    info['requires_preroll'] = bool(requires_preroll(bpy.context.scene))
    # Cameras and lights: what `planning.pose_digest` leaves out of a staged
    # pose, so a camera move is not reported as the film leaving it. Here in
    # the index rather than in each frame record, because a new record key
    # would change every frame signature and fail the replay check of
    # evaluations accepted before it.
    info['viewpoints'] = sorted(o.name for o in bpy.context.scene.objects
                                if o.type in ('CAMERA', 'LIGHT'))
    # Bone order for the records' `bone_rotations`; see `bone_rotations`.
    rigs = armature_table(bpy.context.scene)
    if rigs:
        info['rigs'] = rigs
    info['frames'] = []
    # Each frame's identity, in the index rather than the records: a new record
    # key would change every signature (see `viewpoints`). `signatures` lets a
    # later check diff two evaluations without reading every record;
    # `render_states` is what the pixels depend on beyond the signature,
    # skipped by the callers that only place states or check scattered frames.
    signatures, render_states = {}, {}
    digester = None
    if job.get('render_states', True):
        from render_state import RENDER_STATE_VERSION, RenderStates
        bpy.context.scene.frame_set(info['frame_start'])  # one fixed state to walk
        digester = RenderStates(bpy, info, display_color=display_color)
        info['render_state_version'] = RENDER_STATE_VERSION
    started, stamps = time.time(), {}
    def progress(value):
        write_progress(_output(), value, started, stamps)
    def records():
        for i in visit_frames(frames, info):
            record, local, placed = evaluated_frame(i, geometry_cache, with_transforms=True)
            for obj in record['objects']:
                obj['mesh'] = write_mesh(out, *local[obj['name']], known, digests, writer=writer)
            write_signed_record(out, record, tables, transforms=placed)
            signatures[i] = record['signature']
            if digester is not None:
                render_states[i] = digester.frame(i, bpy.context.evaluated_depsgraph_get())
            info['frames'].append(i)
            if len(info['frames']) == len(frames):
                # Whole files before anything reads them back: `measure` stops
                # drawing records at the last frame and goes on to read the
                # evaluation from disk, so the end of this loop is never reached first.
                writer.flush()
            progress({'phase': 'evaluate', 'frame': i, 'done': len(info['frames']),
                      'total': len(frames), 'mesh_assets': len(known)})
            yield record, world_geometry(record, local)
    try:
        stream = records()
        if job.get('verification'):
            spec = job['verification']
            checks = dict(progress=progress, pair_store=spec.get('collision_pair_store'))
            if job.get('reverify'):
                # Every frame evaluated and written first: which frames
                # changed is only known once they all have signatures.
                for _ in stream:
                    pass
                info['complete'] = True
                rows = reverify(job['reverify'], spec, out, info, frames, signatures, checks)
            else:
                rows = measure(out, spec['baseline'], spec['scene'], spec['catalog'],
                               info=info, records=stream, owned=job.get('owned'), frames=frames,
                               **checks)
            reports = aggregate(rows, defer_partial_events=job.get('defer_partial_events', True))
            if rows.get('reverified') is not None:
                scope = {"reverified": rows['reverified'], "carried": info['carried']}
                reports = [{**report, "scope": scope} for report in reports]
            # Beside the reports: what a parallel check merges, and what a
            # later check re-verifies only some frames of (see `verify`).
            (_output() / 'verification_rows.json').write_text(json.dumps(rows))
            (_output() / 'verifier_reports.json').write_text(json.dumps(reports))
        else:
            for _ in stream:
                pass
        if info['frames'] != frames:
            raise ValueError('incomplete or unordered evaluation')
        info['signatures'] = {str(i): value for i, value in signatures.items()}
        if digester is not None:
            info['render_states'] = {str(i): value for i, value in
                                     digester.finish(render_states, signatures).items()}
        info['complete'] = True
        writer.flush()
        (out / 'index.json').write_text(json.dumps(info, indent=2))
        return info
    finally:
        writer.close()
        geometry_cache.close()


def reverify(request, spec, out, info, frames, signatures, checks):
    """Re-verify only the frames a repair can have changed, and merge.

    The last whole-film check's rows are in `/output/base_rows.json` and its
    signatures in `request`. Frames whose signature changed, and those their
    change can reach (`chunking.reverify_frames`), are measured again from the
    evaluation just written; every other frame's rows are carried. The merged
    document reports as a whole-film check of this content would. A changed
    index field, or a base that cannot be read, falls back to verifying every
    frame.
    """
    from chunking import ContextMismatch, merge_scoped, reverify_frames
    from continuity import frame_runs
    from evaluation import RECHECK_FIELDS
    from verify import fill_advisories, fill_diagnostics, measure
    base_path = _output() / 'base_rows.json'
    base = None
    if all(info.get(key) == request['base_index'].get(key) for key in RECHECK_FIELDS) and base_path.is_file():
        base = json.loads(base_path.read_text())

    def whole():
        return measure(out, spec['baseline'], spec['scene'], spec['catalog'], info=info,
                       frames=frames, **checks)
    if base is None or base.get('frames') != frames:
        return whole()
    old = request['base_signatures']
    changed = [frame for frame in frames if old.get(str(frame)) != signatures.get(frame)]
    owned, evaluated = reverify_frames(info, changed)
    if owned:
        # The room as the whole-film check read it, on the film's first frame,
        # whatever frame the re-measured set starts on; a parallel re-check's
        # chunks read it there too (`session.recheck_in_parallel`).
        scoped = measure(out, spec['baseline'], spec['scene'], spec['catalog'], info=info,
                         frames=evaluated, owned_frames=owned, room_frame=frames[0], **checks)
        try:
            rows = merge_scoped(base, scoped)
        except ContextMismatch:
            # Context the index cannot show moved -- the room shell read on
            # another first frame. Sound only measured whole.
            return whole()
    else:
        rows = {**base, 'reverified': [],
                'stats': {key: 0 for key in base['stats']}}
    fill_diagnostics(rows, out, progress=checks.get('progress'), pair_store=checks.get('pair_store'))
    # An instance the repair set moving has no rows on the frames carried.
    fill_advisories(rows, out, spec['baseline'], spec['scene'], spec['catalog'], info=info)
    owned_set = set(owned)
    info.update(changed=frame_runs(changed), reverified=frame_runs(owned),
                carried=frame_runs([frame for frame in frames if frame not in owned_set]))
    rows['reverified'] = frame_runs(owned)
    return rows


#: Ops that work on an evaluation already on disk and never read the blend.
_FROM_DISK_OPS = frozenset({"fill_rows", "measure_rows"})


def fill_rows(job):
    """Complete a merged parallel check's advisory rows (`verify.fill_advisories`).

    `/output` is the check's `evaluated/` directory: its merged evaluation and
    rows. Measured only when a chunk lacks rows for an instance another chunk
    saw move, which `animation/session.py` decides before starting this. With
    `diagnostics`, the overlap spans' diagnostics first, as `reverify` fills
    them after merging a re-check into its base.
    """
    from verify import fill_advisories, fill_diagnostics
    spec = job['verification']
    out = _output() / 'evaluation'
    path = _output() / 'verification_rows.json'
    rows = json.loads(path.read_text())
    if job.get('diagnostics'):
        fill_diagnostics(rows, out, pair_store=spec.get('collision_pair_store'))
    filled = fill_advisories(rows, out, spec['baseline'], spec['scene'], spec['catalog'])
    path.write_text(json.dumps(rows))
    return {'filled_frames': filled}


def measure_rows(job):
    """Measure frames of an evaluation already on disk: one chunk of a parallel re-check.

    `/output/evaluation` is the check's merged evaluation; `frames` are read
    and `owned_frames` reported, as `reverify` measures them in one process.
    The room is read on `room_frame`, the film's first frame, where a
    whole-film check reads it. Writes `/output/verification_rows.json`.
    """
    from verify import measure
    spec = job['verification']
    out = _output() / 'evaluation'
    info = json.loads((out / 'index.json').read_text())
    started, stamps = time.time(), {}
    rows = measure(out, spec['baseline'], spec['scene'], spec['catalog'],
                   info={**info, 'complete': True}, frames=job['frames'],
                   owned_frames=job['owned_frames'], room_frame=job['room_frame'],
                   progress=lambda value: write_progress(_output(), value, started, stamps),
                   pair_store=spec.get('collision_pair_store'))
    (_output() / 'verification_rows.json').write_text(json.dumps(rows))
    return {'measured_frames': len(job['owned_frames'])}


def inspect_frames(job):
    """Structured evaluated scene facts for the visual reviewer; no authored code."""
    info = timeline()
    frames = job["frames"]
    if not 1 <= len(frames) <= 4 or any(type(i) is not int or not 0 <= i < info['n_frames'] for i in frames):
        raise ValueError('inspect requires 1 to 4 valid output frame indices')
    names = job.get('names', [])
    offset = job.get('offset', 0)
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    scene = bpy.context.scene
    objects = sorted((obj for obj in scene.objects if not names or obj.name in names), key=lambda obj: obj.name)
    selected = objects[offset:offset+200]
    records = {}
    for i in visit_frames(frames, info):
        deps = bpy.context.evaluated_depsgraph_get()
        rows = []
        overview = {'room_surfaces': [], 'visible_groups': {}}
        candidates = sorted(scene.objects, key=lambda o: o.name) if job.get('overview') else selected
        for obj in candidates:
            evaluated = obj.evaluated_get(deps)
            bounds = None
            if obj.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META'):
                corners = [evaluated.matrix_world @ Vector(v) for v in evaluated.bound_box]
                bounds = [[min(v[k] for v in corners) for k in range(3)],
                          [max(v[k] for v in corners) for k in range(3)]]
            row = {'name': obj.name, 'type': obj.type, 'hidden': bool(obj.hide_render),
                   'position': list(evaluated.matrix_world.translation), 'bounds': bounds}
            if obj in selected:
                rows.append(row)
            if job.get('overview') and bounds is not None and not obj.hide_render:
                if obj.get('dycraft_room'):
                    overview['room_surfaces'].append(row)
                else:
                    identity = str(obj.get('dycraft_identity', ''))
                    group = identity.split('/')[0] if '/' in identity else obj.name
                    groups = overview['visible_groups']
                    if group not in groups:
                        groups[group] = {'bounds': bounds, 'object_count': 1}
                    else:
                        old = groups[group]['bounds']
                        groups[group]['bounds'] = [[min(old[0][k], bounds[0][k]) for k in range(3)],
                                                   [max(old[1][k], bounds[1][k]) for k in range(3)]]
                        groups[group]['object_count'] += 1
        records[i] = {'frame': i, 'time': i/info['fps'], 'objects': rows,
                      'camera': scene.camera.name if scene.camera else None}
        if job.get('overview'):
            records[i]['overview'] = overview
    return {'timeline': info, 'total_objects': len(objects),
            'next_offset': offset+200 if offset+200 < len(objects) else None,
            'frames': [records[i] for i in frames]}


#: Runs this long or longer are drawn in one call; shorter ones frame by frame.
MIN_RUN = 8


def frame_runs_for_render(frames):
    """`frames` as `[("run", (first, last, step)) | ("frame", i)]`, in time order.

    Evenly spaced stretches of at least `MIN_RUN` frames become runs, which
    covers contiguous ranges and strided ones (`frame_step`) alike; scattered
    frames -- a storyboard's states -- stay single.
    """
    frames = sorted(set(frames))
    out, k = [], 0
    while k < len(frames):
        end = k + 1
        if end < len(frames):
            step = frames[end] - frames[k]
            while end + 1 < len(frames) and frames[end + 1] - frames[end] == step:
                end += 1
            if end - k + 1 >= MIN_RUN:
                out.append(("run", (frames[k], frames[end], step)))
                k = end + 1
                continue
        out.append(("frame", frames[k]))
        k += 1
    return out


@contextlib.contextmanager
def _marker_cameras_suspended(scene, active):
    """Stop timeline cuts from overriding an explicitly requested render view.

    `scene.camera` assigned before `bpy.ops.render.render()` does not survive
    the call: the render pipeline re-applies the camera bound to the timeline
    marker covering the current frame. A diagnostic render of a frame lying
    outside the requested camera's own shot therefore came back drawn by the
    film's camera, while `views` — read off `scene.camera` before the call —
    reported the camera that had been asked for, so nothing downstream could
    see the disagreement. Clearing the bindings for the duration is the only
    thing that beats the switch.

    The markers themselves stay put, and `active` is false whenever the caller
    named no camera, so an ordinary render still cuts exactly like the film.
    Restoration runs on the failure path too: leaving a scene with its cuts
    stripped would silently change every later render of that project.
    """
    if not active:
        yield
        return
    # EVERY bound marker, not only the `CUT_PREFIX` ones this harness wrote.
    # Narrowing it to ours was tried and is wrong: an author's own marker would
    # then still override a camera the caller explicitly named, which is the
    # bug this exists to remove, and it would do it on exactly the frames the
    # caller was trying to look at. Suspending an author's marker costs nothing
    # -- the bindings are restored in the `finally` below, so the project is
    # unchanged either way -- and `record_view` still reports the camera that
    # actually drew, which is what covers anything this does not.
    bound = [(marker, marker.camera) for marker in scene.timeline_markers
             if marker.camera is not None]
    for marker, _ in bound:
        marker.camera = None
    try:
        yield
    finally:
        for marker, camera in bound:
            marker.camera = camera


def render_frames(job):
    import _blender_child as render
    actual_settings = config = aspect = None
    if job.get("fit_aspect"):
        # A size the request left to the defaults follows the project's own
        # aspect ratio, read before any recipe overwrites the scene's resolution.
        from render_config import fit_aspect
        project = bpy.context.scene.render
        aspect = project.resolution_x / project.resolution_y if project.resolution_y else None
    if job.get("render_settings") is not None:
        from render_config import apply_final_render_settings, effective_render_settings
        requested = dict(job["render_settings"])
        if aspect:
            requested["width"], requested["height"] = fit_aspect(
                requested["width"], requested["height"], aspect)
        actual_settings = apply_final_render_settings(bpy, requested)
        s = bpy.context.scene
    elif aspect:
        config = render_config()
        config["width"], config["height"] = fit_aspect(config["width"], config["height"], aspect)
        s = render_settings(config, diagnostic=True)
    else:
        s = render_settings(diagnostic=True)
    if s.render.engine == "BLENDER_WORKBENCH":
        # Both branches, not just the default one, and read off the staged
        # scene rather than off this process's engine or the requested recipe:
        # a final-film request names its own engine, which need not be either.
        # A render that passes `render_settings` (to pick a resolution, say)
        # would otherwise draw most materials on the factory gray swatch.
        apply_display_colors()
    info = timeline()
    # What actually drew. A final-film request carries its own validated
    # recipe; otherwise it is this worker's own. See `render_config`.
    info["render"] = (actual_settings if actual_settings is not None
                      else config if config is not None else render_config())
    state_frames = None
    if job.get("states") is not None:
        state_frames = resolve_states(info, job["states"])
        frames = [state_frames[s["id"]] for s in job["states"]]
    elif job.get("frames") is not None:
        frames = job["frames"]
    elif job.get("clip") is not None:
        frames = clip_frames(info, job["clip"])
    elif any(job.get(k) is not None for k in ("start", "end", "seconds")):
        # An interval, in the same spelling every other frame-addressed op
        # takes. Resolved here rather than by the caller so a range render
        # costs no extra Blender launch to learn the timeline.
        from verify import frame_indices
        frames = frame_indices(info, job.get("start"), job.get("end"),
                               seconds=job.get("seconds"))
    else:
        frames = list(range(info["n_frames"]))
    if not frames or any(type(i) is not int or i < 0 or i >= info["n_frames"] for i in frames):
        raise frame_selection_error(frames, info, op="render")
    # Keep the Cycles scene between frames of ONE render call. Without it
    # Cycles re-exports every object, rebuilds the BVH, and re-uploads device
    # buffers on every `render()` -- a fixed cost that is none of it sampling.
    # `render/_blender_child.py` measured that cost at **226 ms against
    # 0.73 ms** on a 16-part scene, and this layer's scenes are 181-1386
    # objects: render time here correlates with object count at r=0.71 and
    # ~1.26 ms per object per frame, which is that same re-sync.
    #
    # Scoped to multi-frame calls because a single-frame render has nothing to
    # reuse it across, and left off for Workbench, which does not sync a
    # Cycles scene at all.
    #
    # The one documented hazard does not arise here. `_bake_prepare` must keep
    # this OFF because baking *replaces* material datablocks between renders
    # and persistent data then serves a stale frame (measured 29/255 against
    # 0.099/255). A frame loop only advances time: `frame_set` re-evaluates
    # the depsgraph and Cycles re-syncs what actually moved, and no datablock
    # is swapped out underneath it.
    #
    # Memory: the Cycles scene stays resident between frames instead of being
    # freed and rebuilt, so peak is about one frame's peak held for the whole
    # call rather than a higher ceiling. Renders are not run in parallel here,
    # unlike checks. `DYCRAFT_ANIMATION_PERSISTENT_RENDER=0` turns it off if a
    # project ever does hit the OOM guard through this.
    if (len(frames) > 1 and s.render.engine == "CYCLES"
            and os.environ.get("DYCRAFT_ANIMATION_PERSISTENT_RENDER") != "0"):
        s.render.use_persistent_data = True
    explicit_camera = None
    if job.get("eye") is not None:
        cam = render._make_camera(s)
        render._aim_camera(cam, {"eye": job["eye"], "target": job["look_at"],
                                 "up": job.get("up") or [0, 0, 1],
                                 "fov_deg": job.get("fov_deg", 60)})
        s.camera = cam
        explicit_camera = cam
    named_camera = None
    if job.get("camera_name"):
        named_camera = bpy.data.objects.get(job["camera_name"])
        if named_camera is None or named_camera.type != "CAMERA":
            raise ValueError("unknown native camera: " + job["camera_name"])
        s.camera = named_camera
    if s.camera is None:
        raise ValueError("animation has no camera")
    # A render that also checks what it drew (see the end of this function)
    # takes the same gate as a standalone `evaluate`, before drawing anything
    # (a dependency audit): that check is each state's `checks_pass`, so an unbaked
    # simulation or a dependency outside the workspace refuses it exactly as
    # compile would, and refusing first does not throw a finished render away.
    fused_check = job.get("evaluate") is not None
    if fused_check:
        audit_dependencies()
        audit_bakes()
    reference = (_workspace() / 'evaluation') if job.get('verify_replay') else None
    if reference:
        from evaluation import replay_frame
        audit_dependencies()
        audit_bakes()          # a replay must reproduce the checked evaluation
        prepare_evaluation()
        simulated = simulated_objects()
        checked = json.loads((reference / 'index.json').read_text())
        from evaluation import INDEX_FIELDS
        for key in INDEX_FIELDS:
            if checked.get(key) != info.get(key):
                raise ValueError(f'accepted animation replay changed {key}')
        # What each drawn frame is, signed as it is drawn. The delivery
        # record keeps these rather than the certifying evaluation's, so a
        # frame is never later admitted by an identity it did not have.
        from render_state import RenderStates
        s.frame_set(info['frame_start'])
        digester = RenderStates(bpy, info, display_color=display_color)
    paths, views = {}, {}
    drawn, mismatches = {}, []
    geometry_cache = FrameGeometryCache()
    def prepare_frame(i, animated=False):
        if animated and reference:
            # render_pre advances the render graph, but evaluated_depsgraph_get
            # can still expose the previous viewport graph in background bpy.
            # Evaluate this same output frame explicitly before replay checks.
            # This does not skip, retime, or substitute any rendered frame.
            s.frame_set(info["frame_start"] + i)
        if explicit_camera is not None:
            # Re-asserted per frame because `frame_set` can move the active
            # camera. `_marker_cameras_suspended` is what makes it stick
            # through the render itself.
            s.camera = explicit_camera
        if named_camera is not None:
            s.camera = named_camera
            bpy.context.view_layer.update()
        if actual_settings is not None:
            current = effective_render_settings(
                s, actual_settings["compute_backend"], actual_settings["devices"])
            # Workbench's saved shading can itself be animated. Only settings
            # pinned by this render request must stay constant in that mode.
            keys = ("engine", "width", "height", "resolution_percentage") if (
                actual_settings["engine"] == "BLENDER_WORKBENCH") else actual_settings.keys()
            if any(current[k] != actual_settings[k] for k in keys):
                raise ValueError(f"Animation changed final render settings at output frame {i}")
        if reference:
            record, local = evaluated_frame(i, geometry_cache)
            # A frame that does not reproduce the checked one is recorded and
            # drawn: drawing never waits on a verdict, and refusing here would
            # throw away delivery renders that had already run. What stays
            # fatal is above: another project state, a missing input, an
            # unbaked simulation.
            signature, changed = replay_frame(reference, i, record, local)
            drawn[i] = {"signature": signature,
                        "render_state": digester.frame(i, bpy.context.evaluated_depsgraph_get())}
            if changed:
                row = {"frame": i, "changed": changed}
                baked = sorted({n for names in changed.values() for n in names} & simulated)
                if baked:
                    row["bake"] = baked
                mismatches.append(row)
                # On disk as well, so a render killed part way through still
                # says which of its frames did not reproduce (`delivery.admit`).
                with open(_output() / REPLAY_MISMATCH_LOG, "a") as log:
                    log.write(json.dumps(row) + "\n")
    def record_view(i):
        """The optics that drew frame `i`, read after it was drawn.

        The camera's *name*, not only its matrix: a storyboard panel has to say
        which camera framed it, and under a derived camera track that is
        whichever one the timeline markers made active at this frame rather
        than anything the caller asked for. Lens *and* sensor: a field of view
        is not recoverable from focal length alone, and `storyboard()` resolves
        which objects a panel frames from exactly these numbers.

        Called after `render()` rather than inside `prepare_frame`.
        `bpy.ops.render.render()` runs its own marker
        camera switch, so a record taken before it names the camera the caller
        *asked for* while the pixels show the one the markers chose -- measured
        byte-identical to a render of the marker's camera. Since this record is
        also persisted per frame and read back by `storyboard()`, taking it
        early meant persisting wrong optics for every frame at or after the
        first cut.
        """
        views[i] = {'frame': i, 'camera': s.camera.name,
                    'camera_matrix': [list(row) for row in s.camera.matrix_world],
                    'camera_type': s.camera.data.type, 'lens_mm': s.camera.data.lens,
                    'sensor_mm': s.camera.data.sensor_width,
                    'sensor_height_mm': s.camera.data.sensor_height,
                    'sensor_fit': s.camera.data.sensor_fit,
                    'ortho_scale': s.camera.data.ortho_scale,
                    'shift': [s.camera.data.shift_x, s.camera.data.shift_y]}
        return views[i]

    # `bpy.ops.render.render()` performs a marker camera switch of its own, as
    # part of the render and after `prepare_frame` has run -- so an explicitly
    # requested diagnostic camera would be discarded for every frame at or
    # after the first cut, silently: the render comes back identical to one of
    # the marker's camera.
    #
    # An author unable to place a diagnostic camera would move a *delivery*
    # camera instead, an animation-stage edit that trips the storyboard gate,
    # all to look closely at one frame.
    #
    # Unbinding the marker restores the requested camera, byte-identical to a
    # no-markers control.
    #
    # The unbind/restore itself is `_marker_cameras_suspended`, whose `finally`
    # is the part that matters: leaving a project with its cuts stripped
    # because a render raised would silently change every later render of it.
    overridden = any(job.get(k) is not None
                     for k in ("camera_name", "eye"))
    render_started, render_stamps = time.time(), {}
    # Read after `render_settings()` has staged the scene, so it reports what
    # will actually draw rather than what the defaults say.
    diagnostic = None if actual_settings is not None else diagnostic_render_settings(s)

    try:
        with _marker_cameras_suspended(s, overridden):
            if job.get("movie"):
                from movie import render_movie
                if frames != list(range(info["n_frames"])):
                    raise ValueError("Batch movie rendering requires the complete frame range")
                movie = render_movie(bpy, info, _output(), lambda i: prepare_frame(i, animated=True),
                                     record_view=record_view,
                                     progress=lambda value: write_progress(
                                         _output(), value, render_started, render_stamps))
                return {**info, **movie, "views": [views[i] for i in frames],
                        "render_settings": actual_settings or diagnostic,
                        **delivery_identity(reference, drawn, mismatches)}
            def view_written(i):
                # Written immediately after the pixels, so every PNG that
                # exists has its optics on disk beside it -- and so those
                # optics are the ones that drew it, which they were not while
                # this ran before the render (see `record_view`). The result
                # payload carries `views` too, but only a job that returns has
                # a result: a delivery render the box kills is admissible for
                # the frames it did draw, and `camera.json` still has to say
                # what framed them. One file per frame for the same reason
                # `frames/` is -- an interrupted render leaves whole records
                # rather than a truncated one.
                v = _output() / "views" / f"{i:06d}.json"
                v.parent.mkdir(exist_ok=True)
                v.write_text(json.dumps(record_view(i)))

            def draw(i):
                """One frame, the scene already on it."""
                prepare_frame(i)
                p = _output() / "frames" / f"{i:06d}.png"
                p.parent.mkdir(exist_ok=True)
                s.render.filepath = str(p)
                bpy.ops.render.render(write_still=True)
                paths[i] = str(p.relative_to(_output()))
                view_written(i)
                write_progress(_output(), {"phase": "render", "frame": len(paths) - 1,
                                           "done": len(paths), "total": len(set(frames)),
                                           "output_frame": i},
                               render_started, render_stamps)

            if requires_preroll(s):
                for i in visit_frames(frames, info):
                    draw(i)
            else:
                # Evenly spaced runs of `MIN_RUN` frames or more in one render
                # call each; the rest one at a time, jumped to.
                for kind, item in frame_runs_for_render(frames):
                    if kind == "frame":
                        s.frame_set(s.frame_start + item)
                        draw(item)
                        continue
                    from movie import render_movie
                    base = len(paths)
                    batch = render_movie(
                        bpy, info, _output(), lambda i: prepare_frame(i, animated=True),
                        record_view=view_written, run=item, stills=True,
                        progress=lambda value, base=base: write_progress(
                            _output(), {**value, "frame": base + value["done"] - 1,
                                        "done": base + value["done"], "total": len(set(frames)),
                                        "output_frame": value["frame"]},
                            render_started, render_stamps))
                    paths.update(batch["paths"])
            # Evaluate chronologically once, but preserve the caller's order and repeats.
            result = {**info, "paths": [paths[i] for i in frames],
                      "views": [views[i] for i in frames],
                      **delivery_identity(reference, drawn, mismatches)}
            if state_frames is not None:
                # The caller cannot work these out: only this process read the
                # NLA. The frames are what the check that follows must cover, and
                # the clips are how it groups them.
                result["state_frames"] = state_frames
                result["state_clips"] = state_clips(job["states"])
            # The effective profile, on the diagnostic path too. `actual_settings`
            # is only populated when the caller passed explicit `render_settings`,
            # so without this a draft would come back with no record of what it
            # had been drawn with.
            if (actual_settings or diagnostic) is not None:
                result["render_settings"] = actual_settings or diagnostic
            # Evaluate and verify the same frames in this process, rather than
            # letting the caller start a second Blender to do it.
            #
            # The duplicated work is almost nothing -- a `frame_set` plus a
            # depsgraph evaluation is milliseconds -- so this is not about
            # avoiding a second pass over the frames. It is about the start: a
            # worker costs more to start than a handful of frames, and a one- or
            # two-state storyboard refresh is the common call now that
            # invalidation is scoped to the panels a change actually reaches.
            #
            # Declined when a service camera was added for this render: it is an
            # object in the scene that the standalone check would not have seen,
            # and an evaluated record carrying it is not the one the verifiers and
            # the replay check are written against.
            #
            # The frames in time order, once each: `frames` follows the
            # plan's state order and may repeat a frame, and the evaluation
            # needs neither, and plan order would fail it after every panel had
            # been drawn. The caller maps verdicts back to states by frame.
            if fused_check:
                # No render-state digests: a storyboard check certifies no
                # delivery, and the walk would cost more than this refresh.
                result["evaluation"] = evaluate({**job["evaluate"], "op": "evaluate",
                                                 "frames": sorted(set(frames)),
                                                 "render_states": False})
            return result
    finally:
        # The marker restore is `_marker_cameras_suspended`'s `finally`: a
        # canceled or failed render must not leave the
        # film's own cuts unbound, because the next delivery render would then
        # draw every shot from one camera and say nothing about it.
        geometry_cache.close()


#: Each replay mismatch as it is found, one JSON row a line, in the render's
#: output directory.
REPLAY_MISMATCH_LOG = "replay_mismatch.jsonl"


def delivery_identity(reference, drawn, mismatches):
    """What a delivery render reports about what it drew, or nothing for a draft.

    `drawn` is each frame's `(signature, render_state)` as drawn, which Lane
    A's delivery record keeps; `replay_mismatch` names each frame that did not
    reproduce the checked evaluation and what differed.
    """
    if not reference:
        return {}
    from render_state import RENDER_STATE_VERSION
    return {"drawn": {str(i): value for i, value in sorted(drawn.items())},
            "render_state_version": RENDER_STATE_VERSION,
            "replay_mismatch": mismatches}


def _plain(value, path, coerced, nonfinite):
    """One job result as something `json.dumps` can write.

    **Why this exists.** The worker is a separate process in the bwrap sandbox,
    so `result.json` on the bound `/output` is the only channel back to the
    parent: stdout and stderr both go to `worker.log`, which Blender writes to
    as well, and `--unshare-net` closes everything else. The two ends also run
    different interpreters — Blender's 3.11 here, the harness's 3.10 there —
    and this side cannot `import dycraft` at all, since only `/tools` is
    mounted. So what crosses has to be self-describing plain JSON, and there is
    no shared codec to lean on.

    **What it handles.** Agent-authored code naturally returns Blender values
    -- reading a joint's `axis` or `limits` custom property gives back an
    `IDPropertyArray`, and a location gives a `Vector` -- and `json.dumps`
    refuses them, which would fail the job *after* the edit had already
    succeeded, with an error naming neither the field nor the operation.

    Every Blender container answers to `list()` — `Vector`, `Euler`,
    `Quaternion`, `Color`, `IDPropertyArray`, and `bpy_prop_array` as flat
    sequences, `Matrix` as a sequence of rows that recurse — so conversion is
    uniform. What changed is recorded in `coerced` and printed to the log
    rather than done silently: a caller that meant to return plain data should
    be able to see that it did not.

    Non-finite floats are collected rather than converted. `allow_nan=False`
    stays on the write, because a `NaN` reaching the manifest would travel into
    hashes and verification where readers differ — but the error now names the
    path instead of just the type.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            nonfinite.append(path)
        return value
    if isinstance(value, bytes):
        coerced.append(f"{path} (bytes -> length)")
        return {"bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v, f"{path}.{k}", coerced, nonfinite)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]", coerced, nonfinite)
                for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        coerced.append(f"{path} ({type(value).__name__} -> sorted list)")
        return sorted(_plain(v, f"{path}[]", coerced, nonfinite) for v in value)
    kind = type(value).__name__
    # numpy scalars (`np.float32`, `np.int64`, `np.bool_`, as `np.isclose` or a
    # float32 norm returns) are not Python numbers, and `list()` refuses them;
    # arrays would list into such scalars. `.item()` and `.tolist()` give the
    # Python values.
    if isinstance(value, np.generic):
        plain = value.item()
        coerced.append(f"{path} ({kind} -> {type(plain).__name__})")
        return _plain(plain, path, coerced, nonfinite)
    if isinstance(value, np.ndarray):
        coerced.append(f"{path} (ndarray -> list)")
        return _plain(value.tolist(), path, coerced, nonfinite)
    # `IDPropertyGroup` — a nested custom property — is dict-like, not a
    # sequence, and iterating it would yield only its keys.
    if hasattr(value, "to_dict"):
        coerced.append(f"{path} ({kind} -> object)")
        return _plain(value.to_dict(), path, coerced, nonfinite)
    try:
        items = list(value)
    except TypeError:
        raise TypeError(
            f"{path} is a {kind}, which cannot be written to the job result. "
            "Return plain numbers, strings, lists, and dicts — convert a "
            "Blender value with `list(...)` (or `.to_dict()` for a nested "
            "custom property) before returning it.") from None
    coerced.append(f"{path} ({kind} -> list)")
    return [_plain(v, f"{path}[{i}]", coerced, nonfinite)
            for i, v in enumerate(items)]


def write_digest(job):
    """Fingerprint what was just saved, for `native/journal/`.

    Placed here and nowhere else because this is the one point where the
    in-memory scene and the bytes on disk agree: the digest names the version
    the operation published, not some state it passed through.

    Never fails the operation. The journal is diagnostic; an edit that
    succeeded must not be reported as failed because its fingerprint could not
    be written, and a digest bug must not take the harness down with it.
    """
    if not job.get("journal", True):
        return
    try:
        from scene_digest import scene_digest
        (_output() / "digest.json").write_text(json.dumps(scene_digest(bpy)))
    except Exception as error:                       # noqa: BLE001 - see above
        print("DIGEST_FAILED " + repr(error), flush=True)


def run_job(job, *, reopen=True):
    """Everything one worker request does, once the project is reachable.

    Split out of `main` so a resident session (`serve`) can drive the same
    dispatch repeatedly in one process. `reopen=False` is the residency case:
    the project is already in memory from a previous request and re-reading it
    is the cost residency exists to remove. The caller owns that decision --
    this function does not track dirtiness, because what makes a scene dirty
    is which op ran last, which only the loop knows.
    """
    result = _run_operation(job, reopen)
    _write_result(job, result, *_save_project(job))


def _run_authored(job):
    """`edit` or `probe`: the author's code, run in the namespace it is promised."""
    result = {}
    # `json` above all: every dycraft scene stores its declarations
    # as a JSON *string* in `scene['dycraft_declarations']` and
    # `scene['dycraft_states']`, so decoding them is the most natural thing
    # either op does. `mathutils` and `bpy_extras` are the next two most
    # reached-for modules.
    #
    # A convenience binding, not a capability grant: `exec` permits `import`,
    # and so does `compile_probe`, so every name here is reachable with more
    # typing.
    from dycraft_joints import key_joint
    namespace = {"bpy": bpy, "np": np, "math": math, "json": json, "os": os,
                 "collections": collections, "result": result,
                 "key_joint": key_joint,
                 "__name__": "__animation__", **_OPTIONAL_MODULES}
    # Idempotent because this process may outlive the call. A plain
    # insert grew `sys.path` by one entry per request in a resident
    # session -- measured 15 entries to 20 over six calls, unbounded
    # over a run.
    workspace = str(_workspace())
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    # The authored tree by its own name, after the root rather than
    # before it: the root is where the harness's own staged files land
    # and has always won. Both views hold the same bytes (hardlinked
    # by `stage_sources`), so this only matters for an author whose
    # code says `animation_src`.
    from render_config import SOURCE_SUBDIR
    sources = str(Path(workspace) / SOURCE_SUBDIR)
    if sources not in sys.path and Path(sources).is_dir():
        sys.path.insert(sys.path.index(workspace) + 1, sources)
    if job["op"] == "probe":
        if job.get("frames") is not None:
            with printed_output():
                result = probe_frames(job, namespace)
        else:
            if job.get("frame") is not None:
                seek_output_frame(job["frame"])
            with printed_output(), authored():
                result = run_probe(compile_probe(job["code"]), namespace)
    else:
        before = animation_snapshot()
        with printed_output(), authored():
            exec(compile(job["code"], job.get("name") or "blender_edit.py", "exec"),
                 namespace)
        result = namespace["result"]
        # Beside the result, never in it: the result is the author's.
        (_output() / "animation_changed.json").write_text(
            json.dumps(animation_changed(before, animation_snapshot(), timeline()["n_frames"])))
    return result


def _run_operation(job, reopen):
    """Run the job's operation on the project and return its result."""
    if job["op"] == "initialize":
        result = initialize(json.loads(Path(job["spec"]).read_text()))
    else:
        # Measuring an evaluation on disk reads no scene: not loading the
        # blend spares each such worker the scene's load, which can be seconds
        # and hundreds of megabytes of caches.
        if reopen and job["op"] not in _FROM_DISK_OPS:
            bpy.ops.wm.open_mainfile(filepath=str(_workspace() / "animation.blend"), use_scripts=False)
        if job["op"] in ("edit", "probe"):
            result = _run_authored(job)
        elif job["op"] == "update":
            from update import synchronize
            result = synchronize(job)
        elif job["op"] == "evaluate":
            if job.get("states") is None or job.get("verification"):
                # Not for placing states: that certifies nothing, and runs
                # before an edit that may be the one that bakes.
                audit_dependencies()
                audit_bakes()
            result = evaluate(job)
        elif job["op"] == "render":
            result = render_frames(job)
        elif job["op"] == "fill_rows":
            result = fill_rows(job)
        elif job["op"] == "measure_rows":
            result = measure_rows(job)
        elif job["op"] == "inspect":
            result = inspect_frames(job)
        else:
            raise ValueError(f"unknown worker op: {job['op']}")
    return result


def _save_project(job):
    """Save what a publishing op changed: `(unsaved_actions, cuts, declaration_errors)`."""
    unsaved_actions, cuts, declaration_errors = [], None, []
    if job["op"] in ("initialize", "edit", "update"):
        if not job.get("preserve_render_settings"):
            persist_render_settings()
        bpy.ops.file.pack_all()
        # Workspace paths remain valid inside every subsequent isolated job.
        # Saving relative paths makes the accepted bundle relocatable outside it.
        bpy.ops.file.make_paths_relative()
        # NOT the place for `audit_dependencies`: during `build`, the `update`
        # merge legitimately holds a reference to the donor blend under
        # `jobs/<id>/donor/`, which is outside the workspace, and
        # `blend_paths()` still reports it here because the in-memory state has
        # not been through a save/reload cycle yet. The reference is gone by the
        # time `evaluate` reopens the file, which is why the audit lives there.
        # Blender does not write a zero-user datablock. An action the author
        # stashed rather than placed -- an alternative take, a clip lifted out
        # of the sequence to try another one -- therefore does not survive this
        # save, and is simply absent when the next job reopens the file. No
        # error is raised anywhere, by Blender or by the harness, so the loss
        # surfaces later as a clip that cannot be found. Naming them here is
        # the one moment it is cheap: the author still holds them in the call
        # that made them, and `use_fake_user = True` retains them -- a fake
        # user counts as a user, so setting it lifts an action above zero and
        # this stops naming it.
        unsaved_actions = sorted(a.name for a in bpy.data.actions if a.users == 0)
        cuts = materialise_cuts()
        keep_declared_units()
        declaration_errors = declaration_warnings()
        bpy.ops.wm.save_as_mainfile(filepath=str(_workspace() / "animation.blend"), check_existing=False)
        write_digest(job)
    return unsaved_actions, cuts, declaration_errors


def _write_result(job, result, unsaved_actions, cuts, declaration_errors):
    """Write the job's result, its coercions, and its warnings for the parent to read."""
    coerced, nonfinite = [], []
    # An edit's or a probe's result is the agent's value, so a value that cannot
    # be returned is the agent's to fix.
    with authored() if job["op"] in ("edit", "probe") else contextlib.nullcontext():
        payload = _plain(result, "result", coerced, nonfinite)
        if nonfinite:
            raise ValueError(
                "non-finite number(s) in the job result at "
                + ", ".join(nonfinite[:8])
                + (f" (+{len(nonfinite) - 8} more)" if len(nonfinite) > 8 else "")
                + ". NaN and infinity cannot cross this boundary: the result is "
                  "written as JSON and read back by the parent, and JSON readers "
                  "disagree about both.")
    body = {"ok": True, "result": payload}
    if coerced:
        body["coerced"] = coerced
        print("COERCED " + json.dumps(coerced), flush=True)
    # A sidecar rather than a field on `result`: `run_worker` returns
    # `result["result"]` alone, and that payload is the author's own value --
    # a probe's number, a render's paths -- which the harness must not reshape
    # to carry a warning. The session reads this file the way it already reads
    # `baseline.json`.
    notes = {}
    if unsaved_actions:
        notes["unsaved_actions"] = unsaved_actions
        print("UNSAVED_ACTIONS " + json.dumps(unsaved_actions), flush=True)
    if cuts is not None:
        notes["cuts"] = cuts
    if declaration_errors:
        notes["declaration_errors"] = declaration_errors
        print("DECLARATION_ERRORS " + json.dumps(declaration_errors), flush=True)
    if notes:
        (_output() / "warnings.json").write_text(json.dumps(notes))
    (_output() / "result.json").write_text(json.dumps(body, allow_nan=False))


def main():
    job = json.loads(Path(os.environ.get("DYCRAFT_BLENDER_JOB") or "/job.json").read_text())
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    # No `.blend1` save backup. Blender writes one beside every
    # `save_as_mainfile` by default, byte-for-byte the size of the project, and
    # the project lives *in the workspace* -- so the backup would be copied by
    # every job, hashed on every publish, copied into `bundle/` at `finish`, and
    # listed in `ir.json`'s manifest as a file that is by definition the
    # project's previous state.
    # Nothing in this tree ever reads one: recovery is `publish()` versioning
    # and the retained per-job workspaces, which is a strictly better story
    # than a single last-good copy.
    bpy.context.preferences.filepaths.save_version = 0
    run_job(job)


#: Ops that leave the in-memory scene equal to the saved project, so the next
#: request in a resident session can skip `open_mainfile`.
#:
#: Only the saving ops qualify, and they qualify *because* they save: `edit`,
#: `update`, and `initialize` end with `save_as_mainfile`, so memory, and file
#: agree by construction. Every other op is excluded on evidence rather than
#: caution -- `render_frames` applies render settings to the scene and can add
#: a camera through `_make_camera`, `evaluate` and `inspect` walk the timeline
#: with `frame_set`, and `probe` evaluates author code that may do anything.
#: None of that is written back, so the scene and the file diverge and the
#: next request must re-read the file to mean what it means today.
_LEAVES_PROJECT_CLEAN = frozenset({"initialize", "edit", "update"})


def purge_authored_modules(roots):
    """Drop `sys.modules` entries loaded from the agent's own source roots.

    A fresh process re-reads a helper from disk every call. A resident one
    serves `sys.modules` and silently executes the version from the previous
    call -- against files the author rewrites between calls. That failure has
    no error and no symptom except an edit appearing
    not to take, so the purge is what *preserves* today's semantics rather
    than a tidiness measure.
    """
    roots = tuple(str(Path(root).resolve()) for root in roots)

    def authored(path):
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            return False
        return any(resolved.startswith(root + "/") for root in roots)

    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        # A namespace package -- `animation_src` itself, which has no
        # `__init__.py` -- has directories and no file. Kept, it would go on
        # handing `from animation_src import helper` the old helper.
        paths = [] if origin else list(getattr(module, "__path__", None) or ())
        if authored(origin) if origin else paths and all(authored(p) for p in paths):
            del sys.modules[name]


def _resident_mb():
    """This process's resident set, reported from inside.

    The parent cannot read it: `--unshare-pid` puts Blender in its own PID
    namespace, so what the harness sees at `/proc/<pid>` is bubblewrap, which
    is a couple of megabytes however large the project is. Measuring the wrong
    process is worse than not measuring, because drift is exactly what this
    number exists to make visible.
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def serve(requests, replies):
    """Resident loop: one process, many requests, project kept in memory.

    The wire is the one `coal_worker` already uses -- an 8-byte big-endian
    length and a pickle -- on a dedicated pair of streams, because Blender
    writes freely to stdout and a protocol cannot share it.

    Results still land in `<output>/result.json` exactly as a one-shot worker
    writes them, and the reply carries only whether the request survived plus
    session diagnostics. That is deliberate: every existing file-based channel
    between worker and harness -- `verifier_reports.json`, `warnings.json`,
    `progress.json`, the evaluation directory -- keeps working untouched, so
    residency changes where the process lives and nothing about what a request
    means.
    """
    import pickle
    import struct

    def receive():
        header = requests.read(8)
        if not header:
            raise EOFError
        if len(header) != 8:
            raise RuntimeError("truncated worker request header")
        length, = struct.unpack("!Q", header)
        body = requests.read(length)
        if len(body) != length:
            raise RuntimeError("truncated worker request")
        return pickle.loads(body)

    def send(value):
        data = pickle.dumps(value, protocol=4)
        replies.write(struct.pack("!Q", len(data)))
        replies.write(data)
        replies.flush()

    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    # No `.blend1` save backup. Blender writes one beside every
    # `save_as_mainfile` by default, byte-for-byte the size of the project, and
    # the project lives *in the workspace* -- so the backup would be copied by
    # every job, hashed on every publish, copied into `bundle/` at `finish`, and
    # listed in `ir.json`'s manifest as a file that is by definition the
    # project's previous state.
    # Nothing in this tree ever reads one: recovery is `publish()` versioning
    # and the retained per-job workspaces, which is a strictly better story
    # than a single last-good copy.
    bpy.context.preferences.filepaths.save_version = 0
    resident, calls = {"clean": False}, 0
    send({"ready": True})
    while True:
        try:
            request = receive()
        except EOFError:
            return
        job = dict(request["job"])
        os.environ["DYCRAFT_BLENDER_OUTPUT_ROOT"] = request["output"]
        Path(request["output"]).mkdir(parents=True, exist_ok=True)
        calls += 1
        try:
            # `_workspace()` first, and not because the caller might forget it:
            # it is the only root that can match. Authored helpers are imported
            # from the workspace -- `blender` copies `animation_src` into it and
            # `run_job` puts it on `sys.path` -- so a module's `__file__` is a
            # path inside this sandbox. The roots the session knows are host
            # paths on the other side of the bind, and prefix-matching those
            # against child paths silently matches nothing. That left the purge
            # inert and the session serving a helper the author had already
            # rewritten, which is the one failure with no error and no symptom.
            purge_authored_modules([_workspace(), *(request.get("source_roots") or ())])
            # Re-enter the project directory on every request. `publish()`
            # swaps it by renaming the old one aside and deleting it, so a
            # working directory entered once at launch goes on naming that
            # deleted copy, and every relative path an author's script opens
            # fails, though the same code works as a job, which starts in its
            # own copy.
            os.chdir(_workspace())
            # Colors are cached per material *name*, and an author edits a
            # material in place between requests. One job's cache must not
            # answer for the next job's materials.
            reset_surface_color_cache()
            # Re-read when the caller says the project moved under us (a
            # publish), or when the last request left the scene unequal to it.
            reopen = request.get("reopen") or not resident["clean"]
            run_job(job, reopen=reopen)
            # A placement reads the NLA and moves nothing.
            resident["clean"] = job["op"] in _LEAVES_PROJECT_CLEAN or bool(job.get("place_only"))
            send({"ok": True, "calls": calls, "reopened": bool(reopen),
                  "resident_mb": _resident_mb(),
                  "orphans": len([d for d in bpy.data.objects if d.users == 0])})
        except BaseException as error:
            (Path(request["output"]) / "result.json").write_text(json.dumps(failure(error)))
            # The next request must not inherit a half-applied edit. Today the
            # process dies and its successor opens the file; here the file is
            # re-read instead, which reaches the same state -- the last
            # successful save -- without paying for a new process.
            resident["clean"] = False
            send({"ok": False, "calls": calls, "error": str(error)})


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve(sys.stdin.buffer, os.fdopen(int(os.environ.get("DYCRAFT_BLENDER_REPLY_FD", "3")), "wb"))
    else:
        try:
            main()
        except BaseException as error:
            (_output() / "result.json").write_text(json.dumps(failure(error)))
            raise
