"""The Blender half of `import_gltf`: read a skinned glTF or FBX, normalize it, freeze it.

Runs under the Blender interpreter (`render/_blender_env.blender_python()`),
never in the main env, and imports nothing from dycraft -- the launcher strips
`PYTHONPATH` so the child sees only its own `bpy` and `numpy`. The main-env
half (`import_gltf.py`) turns what this writes into an ObjectIR bundle.

    <blender-python> _import_gltf_blender.py JOB.json

`JOB.json` names the source, the output directory and the normalization. This
writes into that directory:

* `character.blend` -- one Armature object `rig` and one skinned mesh object
  `body` (Armature modifier -> `rig`), both with identity transforms, the
  figure standing on z=0 centered on the origin and facing +Y; every glTF
  animation as a fake-user Action named `<object>/<clip>`; images packed.
* `rest.npz` -- the bind-shape mesh in that frame (`verts`, `tris`,
  per-corner `uvs`, per-triangle `material`), for the IR's proxy part.
* `report.json` -- bones, actions, materials, what was dropped, the measured
  facing and the self-check residual.

**Why Blender's importer rather than our own reader.** The animation layer
deforms the character with an Armature modifier, so the rig must be a real
Blender armature, and Blender's glTF importer already resolves everything a
hand-written reader would have to be told: node-level axis conversions an
exporter baked into an ancestor (CesiumMan's `Z_UP`), centimeter armatures
(Mixamo's 0.01 object scale), bind-pose recovery and bone orientation. What is
left for this script is to make the result *uniform* -- identity object
transforms, one mesh, one armature, known names -- so the harness can bind it
without guessing.

**The one trap.** Applying an armature object's scale rescales its rest bones
but not the location channels of its actions, which are stored in bone space
in the *old* units. A Mixamo walk applied naively moves its hips 100x too far.
`_scale_location_channels` rescales them, and `run` measures the
whole normalization against the untouched import with every clip posed, so an
error of that kind cannot be written silently.
"""

import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector


ARMATURE_NAME = "rig"
MESH_NAME = "body"
#: Above this, the normalized posed mesh disagrees with the transformed import
#: and the bundle is refused rather than written.
SELF_CHECK_TOLERANCE_M = 1e-4
#: ...or this fraction of the figure's height, if larger: a 1.75 m figure's
#: float noise posed mid-clip reached 0.114 mm (an Objaverse character).
SELF_CHECK_TOLERANCE_FRACTION = 2e-4
#: ...or, with rest="clip-pose:<name>", this fraction of the figure's largest extent: a whole pose
#: absorbed into the skin is only approximate under linear blend skinning.
SELF_CHECK_POSE_FRACTION = 1e-2
#: The turn that stands a figure up, by which of the file's axes is its up.
#: Most files are +z; a scan lying on its back is -z, a fish modeled nose-up
#: is a horizontal axis. Facing (`--yaw`) is measured after this.
UPRIGHT = {"+z": Matrix.Identity(4), "-z": Matrix.Rotation(math.pi, 4, "X"),
           "+x": Matrix.Rotation(-math.pi / 2, 4, "Y"), "-x": Matrix.Rotation(math.pi / 2, 4, "Y"),
           "+y": Matrix.Rotation(math.pi / 2, 4, "X"), "-y": Matrix.Rotation(-math.pi / 2, 4, "X")}
#: What to do with clips the rig cannot carry faithfully. `all` refuses the
#: import (the default: a clip that no longer matches its source is a bug to
#: see, not to hide); `matching` drops just those clips and records why;
#: `none` drops every clip, which also admits a non-uniformly scaled armature
#: -- its bone-space location keys have no single factor to rescale by, but
#: its rest pose applies like any other.
CLIP_POLICIES = ("all", "matching", "none")


class ImportRefused(Exception):
    pass


def _objects_of(kind):
    return [o for o in bpy.data.objects if o.type == kind]


def _skinned_meshes(armature):
    out = []
    for obj in _objects_of("MESH"):
        if any(m.type == "ARMATURE" and m.object is armature for m in obj.modifiers):
            out.append(obj)
    return out


def _bone_parented_meshes(armature, skinned):
    """Rigid props parented to one bone (a helmet on `Head`), not skinned."""
    return [o for o in _objects_of("MESH")
            if o not in skinned and o.parent is armature
            and o.parent_type == "BONE" and o.parent_bone]


def _object_parented_meshes(armature, skinned):
    """Rigid parts parented to the armature object itself, riding no bone (a robot's head screen)."""
    return [o for o in _objects_of("MESH")
            if o not in skinned and o.parent is armature and o.parent_type == "OBJECT"
            and not any(m.type == "ARMATURE" for m in o.modifiers)]


def _merge_skeleton_copies(armatures, rendered=None):
    """Fold copies of one skeleton into the armature that moves most of the figure.

    Exports often give each mesh its own copy of the skeleton -- a head and a
    body each skinned to an identical rig -- or keep posed copies of a metarig
    that carry one rigid part each (a robot's eyes on the head bone of one copy,
    its base plate on the root of another). A copy has the main armature's bone
    names and parents. Its skinned meshes move to the main armature only where
    its rest is the main one's, since skin is bound to rest; its other children
    hang from the main armature (a bone-parented part from the bone of the same
    name), where they are. A copy whose rest differs under skin, or that carries
    animation, is left, and the import refuses as before. `rendered`, for a
    .blend, limits what counts toward the main armature.
    Returns `(armatures left, {copy: {"skinned": [...], "parts": [...]}})`.
    """
    def moved(a):
        return sum(len(o.data.vertices) for o in _skinned_meshes(a) + _bone_parented_meshes(a, [])
                   if rendered is None or o in rendered)
    main = max(armatures, key=moved)
    def shape(a):
        return {b.name: (b.parent.name if b.parent else None) for b in a.data.bones}
    ref, mw = shape(main), main.matrix_world
    size = max(((mw @ b.tail_local) - (mw @ b.head_local)).length for b in main.data.bones) or 1.0
    size = max(size, max(((mw @ b.head_local) - (mw @ main.data.bones[0].head_local)).length for b in main.data.bones))
    left, merged = [main], {}
    for a in armatures:
        if a is main:
            continue
        ad = a.animation_data
        if shape(a) != ref or (ad and (ad.action or len(ad.nla_tracks))):
            left.append(a)
            continue
        skin = _skinned_meshes(a)
        if skin:
            aw, q = a.matrix_world, a.matrix_world.to_quaternion()
            same = all(((aw @ b.head_local) - (mw @ main.data.bones[b.name].head_local)).length <= 1e-4 * size
                       and ((aw @ b.tail_local) - (mw @ main.data.bones[b.name].tail_local)).length <= 1e-4 * size
                       and (q @ b.matrix_local.to_quaternion()).rotation_difference(
                           mw.to_quaternion() @ main.data.bones[b.name].matrix_local.to_quaternion()).angle <= 1e-3
                       for b in a.data.bones)
            if not same:
                left.append(a)
                continue
        for o in skin:
            for m in o.modifiers:
                if m.type == "ARMATURE" and m.object is a:
                    m.object = main
        children = [o for o in bpy.context.scene.objects if o.parent is a]
        for o in children:
            world, kind, bone = o.matrix_world.copy(), o.parent_type, o.parent_bone
            o.parent = main
            o.parent_type = kind
            if kind == "BONE":
                o.parent_bone = bone
            o.matrix_world = world
        merged[a.name] = {"skinned": sorted(o.name for o in skin),
                          "parts": sorted(o.name for o in children if o not in skin)}
        bpy.data.objects.remove(a)
    bpy.context.view_layer.update()
    return left, merged


def _render_visible():
    """Objects the active scene renders: in a collection the view layer includes
    and renders, and not hidden from render or from the camera."""
    names = set()

    def walk(layer_collection, ok):
        ok = ok and not layer_collection.exclude and not layer_collection.collection.hide_render
        if ok:
            names.update(o.name for o in layer_collection.collection.objects)
        for child in layer_collection.children:
            walk(child, ok)
    walk(bpy.context.view_layer.layer_collection, True)
    return {o for o in bpy.context.scene.objects
            if o.name in names and not o.hide_render and getattr(o, "visible_camera", True)}


def _claim_slots(armature):
    """Name the slot of every take that animates this rig after the rig.

    `_slot_for` finds a slot by its name or by who uses it. glTF names slots
    after objects, but Blender's FBX reader names them after the action
    (`OBArmature|mixamo.com|Layer0`) and a .blend keeps whatever its takes were
    last called. Once `_rest` unassigns a clip nothing finds such a slot, its
    location channels are never rescaled, and a Mixamo FBX walk's hips land
    100x too high (the Grandfather walk's posed mesh was off by 90.8 m). A slot
    the rig uses, or one keying its bones, is renamed after the rig up front.
    """
    bones = {b.name for b in armature.data.bones}
    want = "OB" + armature.name
    for action in bpy.data.actions:
        slots = getattr(action, "slots", None)
        if not slots or any(s.identifier == want for s in slots):
            continue
        hit = next((s for s in slots if armature in s.users()), None)
        if hit is None:
            for slot in slots:
                if slot.target_id_type != "OBJECT":
                    continue
                paths = [fc.data_path for layer in action.layers for strip in layer.strips
                         for bag in getattr(strip, "channelbags", ()) if bag.slot == slot
                         for fc in bag.fcurves]
                if any(p.startswith('pose.bones["') and p.split('"')[1] in bones for p in paths):
                    hit = slot
                    break
        if hit is not None:
            hit.name_display = armature.name


#: Modifiers that only move vertices. The rest change the vertex count, and
#: can change it differently once the scale is applied (a Weld or Mirror merge
#: distance), so the clips are measured with them switched off.
DEFORM_ONLY_MODIFIERS = {
    "ARMATURE", "LATTICE", "CORRECTIVE_SMOOTH", "SMOOTH", "LAPLACIANSMOOTH", "DISPLACE", "HOOK",
    "MESH_DEFORM", "SURFACE_DEFORM", "SHRINKWRAP", "CAST", "WAVE", "SIMPLE_DEFORM", "WARP",
    "LAPLACIANDEFORM", "VERTEX_WEIGHT_EDIT", "VERTEX_WEIGHT_MIX", "VERTEX_WEIGHT_PROXIMITY",
    "DATA_TRANSFER", "NORMAL_EDIT", "WEIGHTED_NORMAL", "UV_PROJECT", "UV_WARP"}
#: Modifiers whose added geometry is part of the figure and is applied before
#: parts are joined (subdivision and geometry nodes stay muted: lower resolution,
#: not missing limbs).
MADE_REAL_MODIFIERS = {"MIRROR", "ARRAY", "SOLIDIFY"}


def _prepare_blend(armature, keep, skinned):
    """Make an opened scene behave like a fresh import for what follows.

    The kept objects become visible and selectable in the scene, the file
    leaves edit or pose mode, animation on every other object is cleared --
    a parent or prop is not part of the rig, and its motion would otherwise
    count as the clip's -- and modifiers that change the vertex count are
    switched off in the viewport while the clips are measured.
    """
    scene_collection = bpy.context.scene.collection
    for obj in keep:
        if obj.name not in scene_collection.objects:
            scene_collection.objects.link(obj)
        obj.hide_viewport = False
        obj.hide_select = False
        obj.hide_set(False)
    view_layer = bpy.context.view_layer
    active = view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        if active not in keep:
            if active.name not in scene_collection.objects:
                scene_collection.objects.link(active)
            active.hide_viewport = False
            active.hide_set(False)
        bpy.ops.object.mode_set(mode="OBJECT")
    # An object keeps the mode it was saved in even when another is active (an
    # Objaverse zebra's rig was left in pose mode behind its ground plane), and
    # becomes the context's mode the moment it is made active.
    for obj in keep:
        if obj.mode != "OBJECT":
            view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = armature
    cleared = sorted(o.name for o in bpy.data.objects
                     if o not in keep and o.animation_data is not None
                     and (o.animation_data.action or len(o.animation_data.nla_tracks)))
    for obj in bpy.data.objects:
        if obj not in keep and obj.animation_data is not None:
            obj.animation_data_clear()
    # Geometry a modifier makes is part of the figure -- a Mirror is often the
    # right arm -- but joining parts keeps only the active part's modifiers, so
    # it is made real first (weights included: a Mirror mirrors .L/.R groups).
    # Shape keys forbid applying; such parts keep the modifier muted, as before.
    applied = []
    for obj in skinned:
        if obj.data.shape_keys is not None:
            continue
        for mod in [m for m in obj.modifiers if m.type in MADE_REAL_MODIFIERS and m.show_viewport]:
            if obj.data.users > 1:
                obj.data = obj.data.copy()
            kind = mod.type
            with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
                bpy.ops.object.modifier_apply(modifier=mod.name)
            applied.append(f"{obj.name}:{kind}")
    muted = [(o, m) for o in skinned for m in o.modifiers
             if m.show_viewport and m.type not in DEFORM_ONLY_MODIFIERS]
    for _obj, mod in muted:
        mod.show_viewport = False
    view_layer.update()
    return {"cleared_animation": cleared, "muted_modifiers": muted, "applied_modifiers": applied}


def _authored_limits(armature):
    """Limit constraints and IK limits the source authored on the rig's bones.

    glTF and Blender's FBX reader carry none; a .blend keeps them, and they
    stay on the rig in the blend, acting on the evaluated pose.
    """
    out = []
    for pb in armature.pose.bones:
        for c in pb.constraints:
            if c.type not in ("LIMIT_ROTATION", "LIMIT_LOCATION", "LIMIT_SCALE") or c.mute:
                continue
            e = {"bone": pb.name, "kind": c.type.lower(), "owner_space": c.owner_space,
                 "influence": round(float(c.influence), 4)}
            for ax in "xyz":
                on = getattr(c, f"use_limit_{ax}", None)
                if on is None:
                    on = getattr(c, f"use_min_{ax}", False) or getattr(c, f"use_max_{ax}", False)
                if on:
                    lo, hi = getattr(c, f"min_{ax}"), getattr(c, f"max_{ax}")
                    if c.type == "LIMIT_ROTATION":
                        lo, hi = math.degrees(lo), math.degrees(hi)
                    e[ax] = [round(float(lo), 4), round(float(hi), 4)]
            out.append(e)
        ik = {ax: [round(math.degrees(getattr(pb, f"ik_min_{ax}")), 3),
                   round(math.degrees(getattr(pb, f"ik_max_{ax}")), 3)]
              for ax in "xyz" if getattr(pb, f"use_ik_limit_{ax}")}
        if ik:
            out.append({"bone": pb.name, "kind": "ik_limit_deg", **ik})
        locks = [ax for ax in "xyz" if getattr(pb, f"lock_ik_{ax}")]
        if locks:
            out.append({"bone": pb.name, "kind": "ik_lock", "axes": locks})
    return out


def _skin_to_bone(obj, armature):
    """Turn a bone-parented mesh into one skinned 100% to that bone.

    Joining it into `body` then keeps it riding the same bone, and the rig has
    one binding mechanism rather than two. Skin is bound at rest and moved by
    the bone's pose, so the mesh is first carried back to where the bone at
    rest holds it: left where a posed file displays it, the pose would move it
    twice (a robot saved with its arms raised shed its arm shells beside it).
    """
    world = obj.matrix_world.copy()
    bone = obj.parent_bone
    pb = armature.pose.bones[bone]
    to_rest = armature.matrix_world @ pb.bone.matrix_local @ pb.matrix.inverted() @ armature.matrix_world.inverted()
    obj.parent = None
    obj.matrix_world = to_rest @ world
    group = obj.vertex_groups.new(name=bone)
    group.add(range(len(obj.data.vertices)), 1.0, "REPLACE")
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = armature


def _evaluated_world(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        m = np.array(obj.matrix_world)
        return co @ m[:3, :3].T + m[:3, 3]
    finally:
        ev.to_mesh_clear()


def _slot_for(action, obj):
    """The action slot that animates `obj`, or None for a legacy action.

    A glTF clip animates every node it touches, and Blender 4.4+ stores that as
    one layered action with one slot per animated object: Soldier's clips carry
    the armature *and* the leaf-bone Empties. Anything asked of "the clip" --
    what moves, what to rescale -- has to be asked of the armature's slot.
    """
    slots = getattr(action, "slots", None)
    if not slots:
        return None
    for slot in slots:
        if slot.identifier == "OB" + obj.name:
            return slot
    for slot in slots:
        if obj in slot.users():
            return slot
    return None


def _set_action(armature, action, frame, defaults=None):
    """Pose `action` at `frame` from rest -- or from `defaults` (`_pose_defaults`)
    -- so a channel the clip does not key cannot keep the last clip's value."""
    ad = armature.animation_data or armature.animation_data_create()
    for track in list(ad.nla_tracks):
        track.mute = True
    ad.action = None
    _zero_pose(armature)
    for (bone, prop, i), value in (defaults or {}).items():
        getattr(armature.pose.bones[bone], prop)[i] = value
    ad.action = action
    if action is not None:
        slot = _slot_for(action, armature)
        if slot is not None:
            ad.action_slot = slot
    bpy.context.scene.frame_set(int(frame), subframe=float(frame) - int(frame))
    bpy.context.view_layer.update()


def _zero_pose(armature):
    for pb in armature.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def _rest(armature):
    ad = armature.animation_data
    if ad is not None:
        ad.action = None
        for track in list(ad.nla_tracks):
            ad.nla_tracks.remove(track)
    _zero_pose(armature)
    bpy.context.view_layer.update()


#: A pose channel's rest value, against which a bone's default is measured.
_POSE_REST = {"location": (0.0, 0.0, 0.0), "rotation_quaternion": (1.0, 0.0, 0.0, 0.0),
              "rotation_euler": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)}


def _pose_props(pb):
    rot = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": None}.get(pb.rotation_mode, "rotation_euler")
    return [p for p in ("location", rot, "scale") if p]


def _keyed_channels(action, armature):
    """{(bone, property, index)} that `action` keys on `armature`'s bones."""
    keyed = set()
    for _bag, fc in _fcurves(action, armature):
        if fc.data_path.startswith('pose.bones["'):
            keyed.add((fc.data_path.split('"')[1], fc.data_path.rsplit(".", 1)[1], fc.array_index))
    return keyed


def _pose_defaults(armature):
    """What each bone holds where a clip does not key it, where that is not rest.

    A clip keys some channels and leaves the rest at the file's stored pose --
    glTF node transforms off the bind pose, an FBX's resting pose -- so those
    values are part of every clip as authored. Objaverse's tuxedo cat keeps its
    whole body 2.3 m (file units) off the bind pose on a bone none of its 46
    clips keys; reset to rest, every clip played displaced and was dropped. The
    O4D skeleton sidecar records these before any clip was evaluated
    (`scene["o4d_defaults"]`); otherwise the pose as loaded is the evidence,
    except where the action active at load keys a channel and so overwrote it.
    Returns {(bone, property, index): value} for the channels off rest.
    """
    ad = armature.animation_data
    active = (_keyed_channels(ad.action, armature)
              if ad is not None and ad.action is not None else set())
    props = {pb.name: _pose_props(pb) for pb in armature.pose.bones}
    values = {(pb.name, p, i): float(v) for pb in armature.pose.bones for p in props[pb.name]
              for i, v in enumerate(getattr(pb, p)) if (pb.name, p, i) not in active}
    recorded = bpy.context.scene.get("o4d_defaults")
    for coll, name, path, i, v in (json.loads(recorded) if recorded else []):
        if coll == "objects" and name == armature.name and path.startswith('pose.bones["'):
            bone, prop = path.split('"')[1], path.rsplit(".", 1)[1]
            if prop in props.get(bone, ()) and isinstance(v, (int, float)):
                values[(bone, prop, int(i))] = float(v)
    return {k: v for k, v in values.items() if abs(v - _POSE_REST[k[1]][k[2]]) > 1e-6}


def _evaluated_local(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


def _default_transform(pb, defaults):
    """A bone's default as (location, rotation quaternion, uniform scale, or None)."""
    def get(prop, i, rest):
        return defaults.get((pb.name, prop, i), rest)
    loc = Vector([get("location", i, 0.0) for i in range(3)])
    if pb.rotation_mode == "QUATERNION":
        rot = Quaternion([get("rotation_quaternion", i, (1.0, 0.0, 0.0, 0.0)[i]) for i in range(4)]).normalized()
    elif pb.rotation_mode == "AXIS_ANGLE":
        rot = Quaternion()
    else:
        rot = Euler([get("rotation_euler", i, 0.0) for i in range(3)], pb.rotation_mode).to_quaternion()
    s = [get("scale", i, 1.0) for i in range(3)]
    return loc, rot, (s[0] if max(s) - min(s) <= 1e-6 * max(map(abs, s)) else None)


def _rebase_channel(action, armature, path, A, b, default):
    """Rewrite one vector channel of `action` (location, quaternion, or scale) as A @ v + b.

    Exact key by key -- the map is affine, so it carries keys and Bezier handles
    alike -- where the components share their key frames; otherwise they are
    first resampled at the union of those frames.
    """
    n = len(default)
    fcs = {fc.array_index: fc for _bag, fc in _fcurves(action, armature) if fc.data_path == path}
    if not fcs:
        return
    frames = [tuple(round(kp.co.x, 4) for kp in fc.keyframe_points) for fc in fcs.values()]
    if len(fcs) < n or any(f != frames[0] for f in frames):
        union = sorted({x for f in frames for x in f})
        vals = {i: [fcs[i].evaluate(x) if i in fcs else default[i] for x in union] for i in range(n)}
        interp = next(iter(fcs.values())).keyframe_points[0].interpolation
        for i in range(n):
            fc = fcs.get(i) or action.fcurve_ensure_for_datablock(armature, path, index=i)
            fc.keyframe_points.clear()
            fc.keyframe_points.add(len(union))
            for kp, x, v in zip(fc.keyframe_points, union, vals[i]):
                kp.co = (x, v)
                kp.interpolation = interp
            fc.update()
            fcs[i] = fc
    for k in range(len(fcs[0].keyframe_points)):
        for attr in ("co", "handle_left", "handle_right"):
            v = np.array([getattr(fcs[i].keyframe_points[k], attr)[1] for i in range(n)])
            w = A @ v + b
            for i in range(n):
                getattr(fcs[i].keyframe_points[k], attr)[1] = float(w[i])
    for fc in fcs.values():
        fc.update()


def _clip_start_defaults(armature, skinned, actions, clip, whole=False):
    """{(bone, property, index): value} for the trunk -- the bones above every bone that carries skin -- as the
    named clip (matched by the end of its name) poses them at its first frame; with `whole`, every bone the clip
    keys. Refused when no clip, or more than one, matches."""
    hits = [a for a in actions if a.name == clip or a.name.endswith("|" + clip) or a.name.endswith(clip)]
    if len(hits) != 1:
        raise ImportRefused(f"rest='{'clip-pose' if whole else 'clip'}:{clip}' matches {len(hits)} clips "
                            f"({sorted(a.name for a in actions)})")
    weighted = set()
    for obj in skinned:
        live = {v.group for vert in obj.data.vertices for v in vert.groups if v.weight > 0}
        weighted |= {g.name for g in obj.vertex_groups if g.index in live}
    trunk = set(b.name for b in armature.data.bones)
    for b in armature.data.bones:
        if b.name in weighted:
            trunk &= {b.name} | {a.name for a in b.parent_recursive}
    keyed = _keyed_channels(hits[0], armature)
    _set_action(armature, hits[0], hits[0].frame_range[0])
    out = {}
    for bone, prop, i in keyed:
        if whole or bone in trunk:
            out[(bone, prop, i)] = float(getattr(armature.pose.bones[bone], prop)[i])
    if armature.animation_data is not None:
        armature.animation_data.action = None
    return out


def _absorb_defaults(armature, skinned, actions, defaults, rest="bind", dropped_keyed=frozenset()):
    """Make the rest pose the figure as displayed, and every clip play from it.

    A bone off its bind pose by default is displayed that way, and every clip
    that leaves it alone holds it there -- an FBX converted from Y-up often
    lies on its back at bind and is stood up by its root bone's default, which
    its clips key as well. Its default is applied as rest, the skin takes the
    shape it deforms to, and the clips that key the bone are re-expressed
    relative to the new rest (pose = default^-1 @ pose: affine in the location
    keys, linear in the quaternion keys, so exact) -- only with `rest="displayed"`,
    since a keyed bone's default is more often just a frame of a clip. Rest bones carry no scale,
    so a default scale is absorbed only when uniform, and then also multiplies
    the location keys of the bones below it (the tuxedo cat's body bone rests
    at 1.3x). A default that cannot be absorbed -- a non-uniform scale, or a
    rotation on a bone whose clips key Euler angles -- becomes a constant key
    in the clips that leave it.
    """
    keyed = {a.name: _keyed_channels(a, armature) for a in actions}
    # Channels the dropped clips keyed count as keyed too: dropping a figure's clips must not change its rest (the
    # clown's file pose tips its root 90 degrees, as its clips do; absorbed once they were dropped, it lay down).
    touched = {(b, p) for ks in keyed.values() for b, p, _i in ks} | set(dropped_keyed)
    # Bones above every bone that carries skin: their default moves the whole body rigidly.
    # (Not `use_deform`: a glTF marks every bone so, IK controls beside the body included.)
    weighted = set()
    for obj in skinned:
        live = {v.group for vert in obj.data.vertices for v in vert.groups if v.weight > 0}
        weighted |= {g.name for g in obj.vertex_groups if g.index in live}
    trunk = set(b.name for b in armature.data.bones)
    for b in armature.data.bones:
        if b.name in weighted:
            trunk &= {b.name} | {a.name for a in b.parent_recursive}
    absorbed, rebased = [], []
    for bone in sorted({b for b, _p, _i in defaults}):
        pb = armature.pose.bones[bone]
        loc, rot, s = _default_transform(pb, defaults)
        if s is None or (abs(s - 1.0) > 1e-6
                         and not all(c.bone.inherit_scale == "FULL" for c in pb.children_recursive)):
            continue
        keyed_here = any((bone, p) in touched for p in ("location", "rotation_quaternion", "rotation_euler", "scale"))
        if keyed_here:
            # A keyed bone's default is usually just some frame of some clip, not a rest. The
            # exception is a bone above the whole body that tips it over: a Y-up bind stood up
            # by its root, which the clips key too. Nothing tells the two apart -- a flying
            # hornet's root and a Mixamo hips flipped mid-clip tip over 90 degrees as well --
            # so that is rebased only on request (`rest="displayed"`).
            # rest="pose" (from rest="clip-pose:<name>") takes every keyed bone's default: the whole pose at a clip's
            # first frame becomes the rest.
            if rest not in ("displayed", "pose"):
                continue
            tilt = (pb.bone.matrix_local.to_3x3() @ rot.to_matrix() @ pb.bone.matrix_local.to_3x3().inverted()
                    ) @ Vector((0.0, 0.0, 1.0))
            if rest == "displayed" and (bone not in trunk or tilt.angle(Vector((0.0, 0.0, 1.0))) < math.radians(45)):
                continue
        euler_keys = any((bone, p) in touched for p in ("rotation_euler", "rotation_axis_angle"))
        if euler_keys and rot.angle > 1e-6:
            continue
        absorbed.append(bone)
        if keyed_here:
            rebased.append(bone)
    grow = {}
    for pb in armature.pose.bones:
        f, up = 1.0, pb.parent
        while up is not None:
            if up.name in absorbed:
                f *= _default_transform(up, defaults)[2]
            up = up.parent
        if abs(f - 1.0) > 1e-9:
            grow[pb.name] = f
    if absorbed:
        _rest(armature)
        held = {k: v for k, v in defaults.items() if k[0] in absorbed}
        muted = []
        for obj in skinned:
            for m in obj.modifiers:
                if m.type != "ARMATURE" and m.show_viewport:
                    m.show_viewport = False
                    muted.append(m)
        try:
            bpy.context.view_layer.update()
            at_rest = {o.name: _evaluated_local(o) for o in skinned}
            for (bone, prop, i), value in held.items():
                getattr(armature.pose.bones[bone], prop)[i] = value
            bpy.context.view_layer.update()
            for obj in skinned:
                delta = _evaluated_local(obj) - at_rest[obj.name]
                mesh = obj.data
                if len(delta) != len(mesh.vertices):
                    raise ImportRefused(f"{obj.name!r} changes topology under its modifiers; "
                                        "cannot carry its bones' default pose into its skin")
                for co_owner in [mesh.vertices] + ([k.data for k in mesh.shape_keys.key_blocks]
                                                    if mesh.shape_keys else []):
                    co = np.empty(len(mesh.vertices) * 3)
                    co_owner.foreach_get("co", co)
                    co_owner.foreach_set("co", (co.reshape(-1, 3) + delta).ravel())
                mesh.update()
        finally:
            for m in muted:
                m.show_viewport = True
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        for pb in armature.pose.bones:
            pb.bone.select = pb.name in absorbed
        bpy.ops.pose.armature_apply(selected=True)
        bpy.ops.object.mode_set(mode="OBJECT")
        _rest(armature)
        for bone in rebased:
            loc, rot, s = _default_transform(armature.pose.bones[bone], defaults)
            w, x, y, z = rot.inverted()
            quat = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
            r_inv = np.array(rot.to_matrix()).T / s
            for action in actions:
                base = f'pose.bones["{bone}"].'
                _rebase_channel(action, armature, base + "location", r_inv, -r_inv @ np.array(loc), tuple(loc))
                _rebase_channel(action, armature, base + "rotation_quaternion", quat, np.zeros(4), tuple(rot))
                _rebase_channel(action, armature, base + "scale", np.eye(3) / s, np.zeros(3), (s, s, s))
        for action in actions:
            for _bag, fc in _fcurves(action, armature):
                if fc.data_path.startswith('pose.bones["') and fc.data_path.endswith(".location"):
                    f = grow.get(fc.data_path.split('"')[1])
                    for kp in fc.keyframe_points if f else ():
                        kp.co.y *= f
                        kp.handle_left.y *= f
                        kp.handle_right.y *= f
    baked = 0
    for action in actions:
        missing = [(k, v) for k, v in defaults.items()
                   if k[0] not in absorbed and k not in keyed[action.name]]
        if not missing:
            continue
        start = action.frame_range[0]
        _set_action(armature, action, start)
        for (bone, prop, i), value in missing:
            fc = action.fcurve_ensure_for_datablock(armature, f'pose.bones["{bone}"].{prop}', index=i)
            if prop == "location":
                value *= grow.get(bone, 1.0)
            fc.keyframe_points.insert(start, value, options={"FAST"})
            baked += 1
    return {"absorbed_into_rest": absorbed, "rebased_keyed_bones": rebased, "constant_keys_added": baked}


def _attach_loose_trees(armature, skinned, actions, defaults=None, reparent=None):
    """Hang the skin-carrying bone trees beside the skeleton from the body.

    A bone tree beside the skeleton -- eyes added after auto-rigging, a crab's
    legs, a dog's collar -- moves only if a clip or a constraint moves it; left
    alone it stays where it is while the body walks off (a Mixamo worker's eyes
    hang in the air when he bows). Each such tree is parented to the body bone
    whose skin lies nearest its own (a vote of the closest vertex pairs: the
    head for eyes, not the head-top marker bone). Rest is unchanged, so the
    figure at rest is, and so are the keys of a tree its clips hold still.
    A tree a clip moves -- a hoof hung from an IK controller whose walk was
    baked into keys -- has its root re-keyed at every frame of that clip, so
    the clip plays as authored while a pose set from rest carries the hoof
    with the leg. Trees with constraints are left to them.

    `reparent` ({bone: new parent}, per asset) hangs bones the same way that
    no rule can find safely: a hoof skinned to a bone under an IK controller
    inside the skeleton stays planted when the leg is posed, but rules that
    catch it also catch rigs that must not change. Each is re-keyed in every
    clip, since its clips move it through its old parents.
    """
    bones = armature.data.bones
    constrained = {pb.name for pb in armature.pose.bones if pb.constraints}
    forced = []
    for name, to in (reparent or {}).items():
        if name not in bones or to not in bones:
            raise ImportRefused(f"reparent {name!r} -> {to!r}: no such bone "
                                f"({name if name not in bones else to!r})")
        moved = {name} | {b.name for b in bones[name].children_recursive}
        if to in moved:
            raise ImportRefused(f"reparent {name!r} -> {to!r}: {to!r} hangs from {name!r}")
        if moved & constrained:
            raise ImportRefused(f"reparent {name!r} -> {to!r}: constrained bones move with it "
                                f"({', '.join(sorted(moved & constrained))})")
        p = {"root": name, "to": to, "override": True}
        if actions:
            p["rekeyed_clips"] = sorted(a.name for a in actions)
        forced.append(p)
    plan = _loose_tree_plan(armature, skinned, actions, constrained,
                            exclude={p["root"] for p in forced}) + forced
    if not plan:
        return plan
    return _hang(armature, plan, defaults)


def _loose_tree_plan(armature, skinned, actions, constrained, exclude=()):
    """The bone trees beside the skeleton to hang from the body: `[{"root", "to", "vertices"[, "rekeyed_clips"]}]`."""
    from collections import Counter
    from mathutils.kdtree import KDTree

    root_of = {}
    for b in armature.data.bones:
        r = b
        while r.parent is not None:
            r = r.parent
        root_of[b.name] = r.name
    to_arm = armature.matrix_world.inverted()
    verts = []                                  # (tree, heaviest bone, armature-space position)
    for obj in skinned:
        names = {g.index: g.name for g in obj.vertex_groups}
        m = to_arm @ obj.matrix_world
        for v in obj.data.vertices:
            per_tree, best = {}, (0.0, None)
            for g in v.groups:
                n = names.get(g.group)
                if n in root_of and g.weight > 0:
                    per_tree[root_of[n]] = per_tree.get(root_of[n], 0.0) + g.weight
                    if g.weight > best[0]:
                        best = (g.weight, n)
            if per_tree:
                tree, w = max(per_tree.items(), key=lambda kv: kv[1])
                if w >= 0.5:
                    verts.append((tree, best[1], m @ v.co))
    count = Counter(t for t, _b, _p in verts)
    if len(count) < 2:
        return []
    main = count.most_common(1)[0][0]
    moving = {}                                 # clip -> bones it keys with changing values
    for action in actions:
        moving[action.name] = set()
        for _bag, fc in _fcurves(action, armature):
            if fc.data_path.startswith('pose.bones["'):
                ys = [kp.co.y for kp in fc.keyframe_points]
                if ys and max(ys) - min(ys) > 1e-6:
                    moving[action.name].add(fc.data_path.split('"')[1])
    body = [(b, p) for t, b, p in verts if t == main and root_of[b] == main]
    kd = KDTree(len(body))
    for i, (_b, p) in enumerate(body):
        kd.insert(p, i)
    kd.balance()
    plan = []
    for tree in sorted(count):
        members = {b for b, r in root_of.items() if r == tree}
        if tree == main or tree in exclude or members & constrained:
            continue
        pts = [p for t, _b, p in verts if t == tree]
        near = sorted((d, body[i][0]) for p in pts[::max(1, len(pts) // 400)]
                      for _co, i, d in [kd.find(p)])[:20]
        p = {"root": tree, "to": Counter(b for _d, b in near).most_common(1)[0][0], "vertices": count[tree]}
        rekey = [a.name for a in actions if members & moving[a.name]]
        if rekey:
            p["rekeyed_clips"] = rekey
        plan.append(p)
    return plan


def _hang(armature, plan, defaults):
    """Parent each plan entry's root to its `to` bone, re-keying it in the clips it lists."""
    # Where each moved tree's root is, in armature space, at every frame of
    # the clips that move it -- before it changes parent. Posing them here
    # must not disturb the pose and clip the file was loaded with.
    ad = armature.animation_data
    loaded = (ad.action if ad else None, getattr(ad, "action_slot", None) if ad else None,
              [t.mute for t in ad.nla_tracks] if ad else [],
              {(pb.name, prop): tuple(getattr(pb, prop)) for pb in armature.pose.bones
               for prop in ("location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale")})
    baked = {}
    for p in plan:
        for name in p.get("rekeyed_clips", ()):
            action = bpy.data.actions[name]
            frames = range(math.floor(action.frame_range[0]), math.ceil(action.frame_range[1]) + 1)
            mats = []
            for f in frames:
                _set_action(armature, action, f, defaults)
                mats.append(armature.pose.bones[p["root"]].matrix.copy())
            baked[(p["root"], name)] = (frames, mats)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edit = armature.data.edit_bones
    for p in plan:
        edit[p["root"]].use_connect = False
        edit[p["root"]].parent = edit[p["to"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    for (root, name), (frames, mats) in baked.items():
        action = bpy.data.actions[name]
        pb = armature.pose.bones[root]
        if pb.rotation_mode == "AXIS_ANGLE":
            pb.rotation_mode = "QUATERNION"
        rot = "rotation_quaternion" if pb.rotation_mode == "QUATERNION" else "rotation_euler"
        base = f'pose.bones["{root}"].'
        for bag, fc in _fcurves(action, armature):
            if fc.data_path in (base + "location", base + "scale", base + "rotation_quaternion",
                                base + "rotation_euler", base + "rotation_axis_angle"):
                bag.fcurves.remove(fc)
        _set_action(armature, action, frames[0], defaults)  # an action takes curves for what it animates
        curves = {(prop, i): action.fcurve_ensure_for_datablock(armature, base + prop, index=i)
                  for prop, n in (("location", 3), (rot, 4 if rot == "rotation_quaternion" else 3),
                                  ("scale", 3)) for i in range(n)}
        prev = None
        for f, m in zip(frames, mats):
            _set_action(armature, action, f, defaults)       # the new parent as the clip poses it
            loc, q, s = armature.convert_space(pose_bone=pb, matrix=m, from_space="POSE",
                                               to_space="LOCAL").decompose()
            if rot == "rotation_quaternion":
                if prev is not None and q.dot(prev) < 0:
                    q.negate()
                r = prev = q
            else:
                r = prev = q.to_euler(pb.rotation_mode, prev) if prev is not None else q.to_euler(pb.rotation_mode)
            for prop, vals in (("location", loc), (rot, r), ("scale", s)):
                for i, v in enumerate(vals):
                    curves[(prop, i)].keyframe_points.insert(f, v, options={"FAST"}).interpolation = "LINEAR"
        for fc in curves.values():
            fc.update()
    if baked:
        action, slot, mutes, pose = loaded
        ad.action = action
        if slot is not None:
            ad.action_slot = slot
        for track, mute in zip(ad.nla_tracks, mutes):
            track.mute = mute
        for (bone, prop), value in pose.items():
            setattr(armature.pose.bones[bone], prop, value)
        bpy.context.view_layer.update()
    return plan


def _weight_unweighted(body, armature):
    """Give the skin's unweighted vertices to the bones beside them.

    An armature leaves a vertex that no deforming bone weights where it is
    while the body moves on: a cow's eyes, added after skinning, stayed in the
    air when she lowered her head. A part that is unweighted throughout (an
    eye, a button) goes rigidly to the deforming bone that carries all the
    bones of the skin around it -- the head for an eye ringed by eyelid and
    brow bones, so a blink does not move it; a stray unweighted vertex in a
    weighted part copies the weights of the nearest weighted vertex of that part.
    """
    from collections import Counter
    from mathutils.kdtree import KDTree

    if any(m.type == "ARMATURE" and m.use_bone_envelopes for m in body.modifiers):
        return None                            # envelopes move unweighted vertices already
    mesh = body.data
    deform = {b.name for b in armature.data.bones if b.use_deform}
    names = {g.index: g.name for g in body.vertex_groups if g.name in deform}
    n = len(mesh.vertices)
    weights = [[(names[g.group], g.weight) for g in v.groups if g.group in names and g.weight > 0]
               for v in mesh.vertices]
    loose = [i for i in range(n) if sum(w for _b, w in weights[i]) < 1e-4]
    if not loose or len(loose) == n:
        return None
    parent = list(range(n))                    # mesh parts: connected components over edges

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in mesh.edges:
        a, b = find(e.vertices[0]), find(e.vertices[1])
        if a != b:
            parent[a] = b
    part = [find(i) for i in range(n)]
    co = [v.co.copy() for v in mesh.vertices]
    weighted = [i for i in range(n) if weights[i] and sum(w for _b, w in weights[i]) >= 1e-4]
    kd = KDTree(len(weighted))
    for k, i in enumerate(weighted):
        kd.insert(co[i], k)
    kd.balance()
    by_part = {}
    for i in weighted:
        by_part.setdefault(part[i], []).append(i)
    whole, copied = Counter(), 0
    groups = {g.name: g for g in body.vertex_groups}

    def group(bone):
        return groups.get(bone) or groups.setdefault(bone, body.vertex_groups.new(name=bone))
    for p, members in _groupby(loose, part).items():
        own = by_part.get(p)
        if own is None:                        # the whole part: one bone, from the skin around it
            vote = Counter(max(weights[weighted[k]], key=lambda bw: bw[1])[0]
                           for i in members[::max(1, len(members) // 200)] for _c, k, _d in kd.find_n(co[i], 8))
            bone = _carrier([b for b, c in vote.items() if c >= 0.1 * sum(vote.values())], armature) \
                or vote.most_common(1)[0][0]
            group(bone).add(members, 1.0, "REPLACE")
            whole[bone] += len(members)
            continue
        near = KDTree(len(own))
        for k, i in enumerate(own):
            near.insert(co[i], k)
        near.balance()
        for i in members:
            _c, k, _d = near.find(co[i])
            for bone, w in weights[own[k]]:
                group(bone).add([i], w, "REPLACE")
            copied += 1
    return {"vertices": len(loose), "whole_parts": dict(whole), "from_neighbours": copied}


def _carrier(names, armature):
    """The deepest deforming bone that is, or is an ancestor of, every bone in `names`."""
    bones = armature.data.bones
    chains = []
    for n in names:
        chain, b = [], bones[n]
        while b is not None:
            chain.append(b.name)
            b = b.parent
        chains.append(chain)
    if not chains:
        return None
    common = set(chains[0]).intersection(*chains[1:])
    return next((n for n in chains[0] if n in common and bones[n].use_deform), None)


def _groupby(items, key):
    out = {}
    for i in items:
        out.setdefault(key[i], []).append(i)
    return out


#: Particle settings measured in world units (hair length, strand radius,
#: child spread, kink, and roughness amplitudes), which scaling an object leaves alone.
_PARTICLE_LENGTHS = ("hair_length", "child_radius", "kink_amplitude", "roughness_1", "roughness_1_size",
                     "roughness_2", "roughness_2_size", "roughness_endpoint", "radius_scale",
                     "particle_size", "display_size")


def _scale_particle_hair(objs, factor):
    """Scale the hair of `objs` with the figure: a hair system's length and
    strand settings are in world units, so a zebra resized to life kept the
    mane and tail it had at file size (a tail longer than the zebra)."""
    done, seen = [], set()
    for obj in objs:
        for ps in getattr(obj, "particle_systems", ()):
            s = ps.settings
            if s.type != "HAIR" or s.name in seen:
                continue
            seen.add(s.name)
            if ps.is_edited:
                done.append({"object": obj.name, "system": ps.name, "rescaled": False,
                             "why": "combed hair keys are not rescaled"})
                continue
            for attr in _PARTICLE_LENGTHS:
                if hasattr(s, attr):
                    setattr(s, attr, getattr(s, attr) * factor)
            done.append({"object": obj.name, "system": ps.name, "rescaled": True})
    return done


def _channelbags(action, obj=None):
    """F-curve containers of an action, layered (4.4+) or legacy.

    With `obj`, only the container of the slot animating it.
    """
    layers = getattr(action, "layers", None)
    if layers:
        slot = _slot_for(action, obj) if obj is not None else None
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", ()):
                    if obj is None or (slot is not None and bag.slot == slot):
                        yield bag
    else:
        yield action


def _fcurves(action, obj=None):
    for bag in list(_channelbags(action, obj)):
        for fc in list(bag.fcurves):
            yield bag, fc


def _scale_location_channels(action, armature, factor):
    n = 0
    for _bag, fc in _fcurves(action, armature):
        if fc.data_path.startswith("pose.bones[") and fc.data_path.endswith(".location"):
            for kp in fc.keyframe_points:
                kp.co.y *= factor
                kp.handle_left.y *= factor
                kp.handle_right.y *= factor
            n += 1
    return n


_OBJECT_TRANSFORM_PATHS = ("location", "rotation_quaternion", "rotation_euler",
                           "rotation_axis_angle", "scale", "delta_location")


def _object_transform_channels(action, obj=None):
    return [(bag, fc) for bag, fc in _fcurves(action, obj)
            if fc.data_path in _OBJECT_TRANSFORM_PATHS]


def _animated_object_channels(action, armature):
    """Object-level channels whose value actually changes over the clip.

    Exporters that key every node every frame (three.js's Mixamo exports do)
    write the armature object's own constant transform into each clip. Those
    are safe to drop once that transform is applied; a channel that *moves*
    is root motion stored on the object, and dropping it would lose it.
    """
    moving = []
    for _bag, fc in _object_transform_channels(action, armature):
        values = [kp.co.y for kp in fc.keyframe_points]
        if values and max(values) - min(values) > 1e-6:
            moving.append(f"{fc.data_path}[{fc.array_index}]")
    return moving


def _drop_object_transform_channels(action):
    """Object-level transform keys would move the rig off its link; drop them."""
    dropped = []
    for bag, fc in list(_object_transform_channels(action)):
        dropped.append(fc.data_path)
        bag.fcurves.remove(fc)
    return sorted(set(dropped))


#: How Blender's glTF importer orients bones. `BLENDER` (its default) keeps
#: each bone along its glTF node's +Y and sizes it in the file's units: on a
#: Mixamo rig that points 46 of Xbot's 52 parent bones away from their child
#: and makes `Hips` 10 m long once the 0.01 armature scale is applied -- tails
#: far outside the figure, which IK chains and anyone reading a tail trip on.
#: `TEMPERANCE` points bones at their children where it can (9/52 left, the
#: branching ones) at body-sized lengths. Deformation does not depend on the
#: choice, and the posed self-check in `run` confirms the clips agree.
BONE_HEURISTIC = "TEMPERANCE"


def _uniform_scale(m, strict=True):
    s = [m.col[i].to_3d().length for i in range(3)]
    if max(s) - min(s) > 1e-6 * max(s):
        if strict:
            raise ImportRefused(f"non-uniform armature scale {s}; the importer "
                                "cannot rescale bone-space location keys for it "
                                "(--clips matching keeps the clips that still "
                                "match, --clips none imports the rig without them)")
        return sum(s) / 3.0
    return s[0]


#: Turn about +Z that brings a facing to DyCraft's front, +Y.
_TURN_TO_FRONT = {("x", 1): 90.0, ("x", -1): -90.0, ("y", 1): 0.0, ("y", -1): 180.0}
#: A head this far (as a fraction of the figure's longest horizontal extent)
#: from the body's center line says which way a long-bodied creature faces.
HEAD_REACH_FRACTION = 0.2


def _head_point(armature):
    """World position of the rig's head (or neck, or skull) bone, if it names one."""
    def depth(b):
        return 0 if b.parent is None else 1 + depth(b.parent)
    for word in ("head", "neck", "skull"):
        hits = [b for b in armature.data.bones if word in b.name.lower()
                and not any(w in b.name.lower() for w in ("top", "end", "nub", "light"))]
        if hits:
            bone = min(hits, key=depth)
            return np.array(armature.matrix_world @ bone.head_local)
    return None


_SIDE_WORD = re.compile(r"left|right")
_SIDE_LETTER = re.compile(r"(?<![a-z])([lr])(?![a-z])")
#: An exporter's running number on a bone name (Sketchfab's `LeftArm_09`,
#: Blender's `.001`), which twins do not share.
_EXPORT_NUMBER = re.compile(r"[._]\d+$")


def _right_vector(armature):
    """Mean world offset from each left bone to its right twin, or None.

    Rigs name their sides -- `LeftArm`/`RightArm`, `thigh.L`/`thigh.R`,
    `Bip01 L Thigh` -- and a figure's right is the one thing its pose cannot
    hide: facing +Y, it lies along +X. Twins are paired without an exporter's
    running number: `LeftArm_09` and `RightArm_017` are one pair, and missing
    that sent nine Sketchfab people to the foot rule, which faced them backwards.
    """
    sides = {}
    for b in armature.data.bones:
        n = _EXPORT_NUMBER.sub("", b.name.lower())
        m = _SIDE_WORD.search(n) or _SIDE_LETTER.search(n)
        if m is None:
            continue
        word = m.group(0)
        key = n[:m.start()] + "#" + n[m.end():]
        sides.setdefault(key, {})["l" if word[0] == "l" else "r"] = b
    pairs = [(s["l"], s["r"]) for s in sides.values() if len(s) == 2]
    if len(pairs) < 2:
        return None
    m = armature.matrix_world
    return np.mean([np.array(m @ r.head_local) - np.array(m @ l.head_local) for l, r in pairs], axis=0)


def _facing(verts, head=None, right=None):
    """Which way the figure faces: from its sides, else its head, else its feet.

    A rig that names its sides says it outright: forward is up x right.
    Without that, an upright figure's head sits over its body and its feet say
    which way it faces (a foot reaches further forward than a heel reaches
    back), while a dog's, bird's or fish's head reaches well out along its body
    and its lowest slice may be a tail on the ground -- which turned
    Objaverse's lion and tiger round -- so there the head decides. Checked by
    eye on 167 Objaverse cast picks, feet and head alone faced 53 the wrong
    way; with the sides first, 17 (rigs naming no sides, or naming them
    mirrored).
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    height = float(hi[2] - lo[2])
    reach = float(max(hi[:2] - lo[:2]))
    if right is not None:
        d = np.array([-right[1], right[0]])          # up x right
        k = int(np.argmax(np.abs(d)))
        if abs(d[k]) > 0.02 * max(reach, height) and abs(d[k]) > 2 * abs(d[1 - k]):
            sign = 1 if d[k] > 0 else -1
            return {"forward": f"{'+' if sign > 0 else '-'}{'xy'[k]}",
                    "signal_m": round(float(abs(d[k])), 5),
                    "yaw_to_front_deg": _TURN_TO_FRONT[("xy"[k], sign)],
                    "why": "the rig's right-side bones lie this way's right of their "
                           "left twins (forward = up x right)"}
    if head is not None:
        d = head[:2] - (lo[:2] + hi[:2]) / 2.0
        k = int(np.argmax(np.abs(d)))
        if abs(d[k]) > HEAD_REACH_FRACTION * reach:
            sign = 1 if d[k] > 0 else -1
            return {"forward": f"{'+' if sign > 0 else '-'}{'xy'[k]}",
                    "signal_m": round(float(abs(d[k])), 5),
                    "yaw_to_front_deg": _TURN_TO_FRONT[("xy"[k], sign)],
                    "why": "the head bone reaches this way from the body's center "
                           "-- a long-bodied figure, which its feet do not orient"}
    if height <= 0.0:
        return {"forward": None, "signal_m": 0.0, "why": "no vertical extent"}
    feet = verts[verts[:, 2] < lo[2] + 0.12 * height]
    body = verts[verts[:, 2] > lo[2] + 0.40 * height]
    if not len(feet) or not len(body):
        return {"forward": None, "signal_m": 0.0,
                "why": "no vertices in the foot or body slice"}
    d = feet[:, :2].mean(axis=0) - body[:, :2].mean(axis=0)
    k = int(np.argmax(np.abs(d)))
    signal = float(abs(d[k]))
    if signal < 1e-3:
        return {"forward": None, "signal_m": round(signal, 5),
                "why": "the foot slice is not displaced from the body's center "
                       "line -- no feet, or a symmetric figure"}
    sign = 1 if d[k] > 0 else -1
    axis = "xy"[k]
    return {"forward": f"{'+' if sign > 0 else '-'}{axis}",
            "signal_m": round(signal, 5),
            "yaw_to_front_deg": _TURN_TO_FRONT[(axis, sign)],
            "why": "the horizontal centroid of the lowest 12% of the mesh is "
                   "displaced this way from the body's -- a foot reaches "
                   "further forward than a heel reaches back"}


def _same_image(a, b):
    """Two image datablocks holding the same picture (an FBX converter duplicates them)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    pa = bpy.path.abspath(a.filepath) if a.filepath else None
    if pa and pa == (bpy.path.abspath(b.filepath) if b.filepath else None):
        return True
    fa, fb = a.packed_file, b.packed_file
    return bool(fa and fb and fa.size == fb.size and bytes(fa.data[:4096]) == bytes(fb.data[:4096]))


def _repair_color_as_alpha(meshes):
    """Cut a shader's alpha that is its base-colour picture's *color* (a converter's slip).

    FBX converters often wire a duplicate of the diffuse texture's color output
    into Alpha (Mixamo's packed skins do), so dark clothing goes see-through in
    any renderer that honors alpha. A cutout driven by the image's real alpha
    channel, or a separate opacity map (hair cards), is left alone.
    """
    fixed = []
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        for bsdf in [n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]:
            alpha, base = bsdf.inputs["Alpha"], bsdf.inputs["Base Color"]
            if not (alpha.is_linked and base.is_linked):
                continue
            al, bl = alpha.links[0], base.links[0]
            if (al.from_node.type == "TEX_IMAGE" and al.from_socket.name == "Color"
                    and bl.from_node.type == "TEX_IMAGE"
                    and _same_image(al.from_node.image, bl.from_node.image)):
                mat.node_tree.links.remove(al)
                alpha.default_value = 1.0
                if hasattr(mat, "surface_render_method"):
                    mat.surface_render_method = "DITHERED"
                if hasattr(mat, "blend_method"):
                    mat.blend_method = "OPAQUE"
                fixed.append(mat.name)
    return sorted(set(fixed))


#: Material names that say the surface is metal (armor, a sword, a buckle, a mech's hull): their Metallic stays.
METAL_NAME = re.compile(r"metal|steel|chrome|silver|armou?r|sword|blade|brass|copper|bronze|(?<![a-z])(iron|gold|mech)(?![a-z])",
                        re.IGNORECASE)


def _reset_untextured_metallic(meshes):
    """Set a fully metallic shader back to non-metal when no texture drives its Metallic (a converter's default).

    Some converters write every material at Metallic 1.0 with no metal map, so skin, cloth and eyes render as metal
    in any physically based renderer. A Metallic driven by a texture (a metallic-roughness map), any value short of
    fully metallic, or a material named as metal (`METAL_NAME`) is left alone. The material's viewport value, which
    Workbench draws with, follows.
    """
    fixed = []
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        if not mat.use_nodes or mat.node_tree is None or METAL_NAME.search(mat.name):
            continue
        for bsdf in [n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]:
            metal = bsdf.inputs["Metallic"]
            if not metal.is_linked and metal.default_value >= 0.99:
                metal.default_value = 0.0
                mat.metallic = 0.0
                fixed.append(mat.name)
    return sorted(set(fixed))


#: The one name every part's render UV map takes before parts are joined.
UV_NAME = "UVMap"


def _unify_uv_maps(meshes):
    """Give every part's render UV map one name, so joining keeps one layer.

    A join merges UV layers by name; parts that name theirs apart (Mixamo's
    `BodyDiffuseUVLayer`, `HairDiffuseUVLayer`, ...) end up as separate layers
    with only the target's active, and every other part samples its textures
    at (0, 0) -- a nurse's scrubs, gloves and hair turned to noise and
    see-through. UV Map nodes naming the old map follow it.
    """
    renamed = []
    for obj in meshes:
        layers = obj.data.uv_layers
        render = next((l for l in layers if l.active_render), layers.active)
        if render is None or render.name == UV_NAME:
            continue
        clash = layers.get(UV_NAME)
        if clash is not None:
            clash.name = UV_NAME + ".prev"
        old, render.name = render.name, UV_NAME
        for slot in obj.material_slots:
            nt = slot.material.node_tree if slot.material and slot.material.use_nodes else None
            for node in (nt.nodes if nt else ()):
                if getattr(node, "uv_map", None) == old:
                    node.uv_map = UV_NAME
        renamed.append(obj.name)
    return renamed


def _material_rows(mesh):
    rows = []
    for mat in mesh.materials:
        row = {"name": mat.name if mat else None, "color": [0.8, 0.8, 0.8],
               "metallic": 0.0, "roughness": 0.5, "alpha": 1.0, "texture": None}
        bsdf = None
        if mat and mat.use_nodes:
            bsdf = next((n for n in mat.node_tree.nodes
                         if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            base = bsdf.inputs["Base Color"]
            row["color"] = [float(v) for v in base.default_value[:3]]
            row["metallic"] = float(bsdf.inputs["Metallic"].default_value)
            row["roughness"] = float(bsdf.inputs["Roughness"].default_value)
            row["alpha"] = float(bsdf.inputs["Alpha"].default_value)
            image = _linked_image(base)
            if image is not None and image.size[0] and image.size[1]:
                px = np.empty(image.size[0] * image.size[1] * image.channels,
                              dtype=np.float32)
                image.pixels.foreach_get(px)
                px = px.reshape(-1, image.channels)[:, :3].astype(np.float64)
                # `pixels` returns an 8-bit image as stored -- byte/255, so
                # sRGB-encoded for an sRGB map (a PNG of byte 128 reads 0.502)
                # -- while `color` is linear. Linearise per pixel, then
                # average; averaging the encoded values made every textured
                # proxy too light (the black crow came out mid-gray).
                if image.colorspace_settings.name == "sRGB" and not image.is_float:
                    px = np.clip(px, 0.0, 1.0)
                    px = np.where(px <= 0.04045, px / 12.92,
                                  ((px + 0.055) / 1.055) ** 2.4)
                # The factor multiplies the map in glTF, but Blender's importer
                # writes the map straight into Base Color and leaves the
                # socket at its 0.8 default, which is not a factor at all.
                row["color"] = [float(v) for v in px.mean(axis=0)]
                row["texture"] = image.name
        rows.append(row)
    return rows


def _linked_image(socket, depth=0):
    if depth > 6 or not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE":
        return node.image
    for inp in node.inputs:
        found = _linked_image(inp, depth + 1)
        if found is not None:
            return found
    return None


def _rest_arrays(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        n = len(mesh.loop_triangles)
        tris = np.empty(n * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        loops = np.empty(n * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("loops", loops)
        mat = np.empty(n, dtype=np.int64)
        mesh.loop_triangles.foreach_get("material_index", mat)
        uvs = None
        if mesh.uv_layers.active is not None:
            uv = np.empty(len(mesh.loops) * 2)
            mesh.uv_layers.active.data.foreach_get("uv", uv)
            uvs = uv.reshape(-1, 2)[loops].reshape(n, 3, 2)
        return co.reshape(-1, 3), tris.reshape(n, 3), uvs, mat
    finally:
        ev.to_mesh_clear()


def _relink_images(src):
    """Point images an FBX recorded by an absolute path elsewhere at a copy near it.

    An FBX keeps the texture paths of the machine that exported it; Blender's own
    image search looks only below the file's directory, which misses the common
    `<Name>/Export/<Name>.fbx` + `<Name>/Textures/` layout. Match by basename,
    case-insensitively, under the file's directory and its parent. An image still
    missing is reported rather than refused: it renders magenta, the rig still works.
    """
    pool = {}
    for root in (src.parent, src.parent.parent):
        for f in sorted(root.rglob("*")):
            if f.is_file():
                pool.setdefault(f.name.lower(), f)
    relinked, missing = {}, []
    for img in bpy.data.images:
        if img.packed_file is not None or img.source not in {"FILE", "SEQUENCE", "TILED"}:
            continue
        path = Path(bpy.path.abspath(img.filepath))
        if path.is_file():
            continue
        hit = pool.get(Path(img.filepath.replace("\\", "/")).name.lower())
        if hit is None:
            missing.append(img.filepath)
            continue
        img.filepath = str(hit)
        img.reload()
        relinked[img.name] = str(hit)
    return relinked, sorted(missing)


def run(job):
    src = Path(job["src"])
    out = Path(job["out"]) if job.get("out") else None
    name = job["name"]
    yaw = job.get("yaw_deg", "auto")
    user_scale = float(job.get("scale", 1.0))
    clip_policy = job.get("clips", "all")
    if clip_policy not in CLIP_POLICIES:
        raise ImportRefused(f"clips must be one of {CLIP_POLICIES}, got {clip_policy!r}")
    rest = job.get("rest", "bind")
    from_clip = str(rest).startswith(("clip:", "clip-pose:"))
    if rest not in ("bind", "displayed") and not from_clip:
        raise ImportRefused(f"rest must be 'bind', 'displayed', 'clip:<clip name>' or 'clip-pose:<clip name>', got {rest!r}")
    up = job.get("up", "+z")
    if up not in UPRIGHT:
        raise ImportRefused(f"up must be one of {sorted(UPRIGHT)}, got {up!r}")
    upright = UPRIGHT[up]
    dropped_clips = {}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    unit_scale = None
    if src.suffix.lower() == ".fbx":
        # Same normalization downstream; only the reader differs. Automatic
        # bone orientation is FBX's counterpart of the glTF bone heuristic
        # (bones point at their children), and leaf bones are FBX's end
        # markers, which carry no weights.
        bpy.ops.import_scene.fbx(filepath=str(src), automatic_bone_orientation=True,
                                 ignore_leaf_bones=True, use_anim=True)
        importer = "fbx"
        relinked_images, missing_images = _relink_images(src)
    elif src.suffix.lower() == ".blend":
        # A native scene -- an author's .blend, or a skeleton-sidecar
        # asset.blend -- is opened, not imported, with its scripts off. Its
        # length unit folds into the scale, so a centimeter scene lands in
        # meters like everything else.
        bpy.ops.wm.open_mainfile(filepath=str(src), load_ui=False, use_scripts=False)
        importer = "blend"
        scene = bpy.context.scene
        import_fps = scene.render.fps / scene.render.fps_base
        unit_scale = float(scene.unit_settings.scale_length or 1.0)
        user_scale *= unit_scale
        for other in [s for s in bpy.data.scenes if s != scene]:
            bpy.data.scenes.remove(other)
        relinked_images, missing_images = _relink_images(src)
    else:
        bpy.ops.import_scene.gltf(filepath=str(src), bone_heuristic=BONE_HEURISTIC)
        importer = "gltf"
        relinked_images, missing_images = {}, []
    bpy.context.view_layer.update()

    armatures = _objects_of("ARMATURE")
    if importer == "blend":
        # A .blend also holds what it does not render: hidden copies, excluded
        # collections, a Rigify metarig that deforms nothing. The character is
        # the armature that moves what renders.
        rendered = _render_visible()
        armatures = [a for a in armatures
                     if any(o in rendered for o in _skinned_meshes(a) + _bone_parented_meshes(a, []))]
    merged_copies = {}
    if len(armatures) > 1:
        armatures, merged_copies = _merge_skeleton_copies(
            armatures, rendered if importer == "blend" else None)
    if len(armatures) != 1:
        raise ImportRefused(
            f"expected exactly one armature, found {len(armatures)} "
            f"({[a.name for a in armatures]}); a multi-rig file is not a "
            "single character")
    arm = armatures[0]
    _claim_slots(arm)
    skinned = _skinned_meshes(arm)
    if importer == "blend":
        skinned = [o for o in skinned if o in rendered]
        arm.data.pose_position = "POSE"
    for obj in _bone_parented_meshes(arm, skinned):
        if importer == "blend" and obj not in rendered:
            continue
        _skin_to_bone(obj, arm)
        skinned.append(obj)
    # A part that rides the armature object but no bone (a robot's head screen) is part of what renders; kept
    # as unweighted skin, `_weight_unweighted` gives it rigidly to the bone that carries the skin around it.
    riding = []
    for obj in _object_parented_meshes(arm, skinned):
        if importer == "blend" and obj not in rendered:
            continue
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        obj.modifiers.new("Armature", "ARMATURE").object = arm
        skinned.append(obj)
        riding.append(obj.name)
    if not skinned:
        raise ImportRefused("the armature deforms no mesh")
    keep = {arm, *skinned}
    dropped = sorted(f"{o.type}:{o.name}" for o in bpy.data.objects if o not in keep)
    blend_prep = _prepare_blend(arm, keep, skinned) if importer == "blend" else None
    alpha_repaired = _repair_color_as_alpha(skinned)
    metal_reset = _reset_untextured_metallic(skinned)

    actions = list(bpy.data.actions)
    if importer == "blend":
        # A scene's actions also animate cameras, props and shape keys; the
        # rig's clips are the ones that key its bones.
        bones = {b.name for b in arm.data.bones}
        actions = [a for a in actions
                   if any(fc.data_path.startswith('pose.bones["')
                          and fc.data_path.split('"')[1] in bones
                          for _bag, fc in _fcurves(a, arm))]
        for action in [a for a in bpy.data.actions if a not in actions]:
            bpy.data.actions.remove(action)
    inspect = out is None
    dropped_keyed = set()
    if clip_policy == "none" and not inspect:
        dropped_keyed = {(b, p) for action in actions for b, p, _i in _keyed_channels(action, arm)}
        for action in actions:
            dropped_clips[action.name] = "--clips none"
            bpy.data.actions.remove(action)
        if arm.animation_data is not None:
            arm.animation_data_clear()
        actions = []
    defaults = _pose_defaults(arm)
    if from_clip:
        # The rest from a clip's first frame: a bind modeled on end (a crow standing on its tail) whose own
        # clip stands it up by the root, while the file's stored pose is the bind and shows nothing to absorb.
        # The trunk's pose at that frame becomes the default, so it is absorbed into the rest and every clip is
        # rebased onto it, as with rest="displayed".
        # rest="clip-pose:<name>" takes the whole pose at that frame: a bind with the wings spread and the beak open
        # (the crow) rests as its clip starts, perched with the wings folded and the beak shut.
        whole = str(rest).startswith("clip-pose:")
        defaults = {**defaults, **_clip_start_defaults(arm, skinned, actions, rest.split(":", 1)[1], whole=whole)}
    # Before the clips are posed for the self-check: an attached tree moves with the body in them.
    attached = _attach_loose_trees(arm, skinned, actions, defaults, reparent=job.get("reparent"))
    # Every clip is posed at its midpoint before and after normalization. One
    # clip is not enough: root motion stored on the armature object, or a
    # location channel the rescale missed, can live in any one of them. Before,
    # each plays from the file's defaults, as authored; after, from rest.
    before = {}
    for action in actions:
        frame = 0.5 * (action.frame_range[0] + action.frame_range[1])
        _set_action(arm, action, frame, defaults)
        before[action.name] = (frame, {o.name: _evaluated_world(o) for o in skinned})
    moving_object_keys = {a.name: m for a in actions
                          if (m := _animated_object_channels(a, arm))}
    pose_defaults = (_absorb_defaults(arm, skinned, actions, defaults,
                                      ("pose" if str(rest).startswith("clip-pose:") else "displayed") if from_clip else rest,
                                      dropped_keyed) | {
        "source": "o4d_defaults" if bpy.context.scene.get("o4d_defaults") else "pose as loaded",
        "channels_off_rest": len(defaults)} if defaults else None)
    _rest(arm)
    raw_rest = np.concatenate([_evaluated_world(o) for o in skinned])
    # Everything about facing and placement is measured on the figure stood upright.
    u3 = np.array(upright)[:3, :3]
    upright_rest = raw_rest @ u3.T
    head, right = _head_point(arm), _right_vector(arm)

    report = {
        "blender": bpy.app.version_string,
        "source": str(src),
        "importer": importer,
        "relinked_images": relinked_images,
        "missing_images": missing_images,
        "import_fps": import_fps,
        "raw": {"armature": arm.name,
                "armature_matrix": [list(r) for r in arm.matrix_world],
                "meshes": sorted(o.name for o in skinned),
                "extent_m": [round(float(v) * (unit_scale or 1.0), 4) for v in
                             raw_rest.max(axis=0) - raw_rest.min(axis=0)],
                "up": up,
                "extent_upright_m": [round(float(v) * (unit_scale or 1.0), 4) for v in
                                     upright_rest.max(axis=0) - upright_rest.min(axis=0)],
                "facing": _facing(upright_rest, None if head is None else u3 @ head,
                                  None if right is None else u3 @ right)},
        "dropped_objects": dropped,
        "merged_skeleton_copies": merged_copies,
        "object_parented_parts": sorted(riding),
        "bones": len(arm.data.bones),
        "actions": [{"name": a.name, "frame_range": list(a.frame_range)}
                    for a in actions],
    }
    if unit_scale is not None:
        report["unit_scale_length"] = unit_scale
    if blend_prep is not None:
        report["blend"] = {"cleared_animation_on": blend_prep["cleared_animation"],
                           "applied_modifiers": blend_prep["applied_modifiers"],
                           "modifiers_off_while_measuring": sorted(
                               f"{o.name}:{m.type}" for o, m in blend_prep["muted_modifiers"])}
    if pose_defaults is not None:
        report["pose_defaults"] = pose_defaults
    if attached:
        report["attached_loose_bones"] = attached
    repairs = {k: v for k, v in (("color_wired_to_alpha", alpha_repaired),
                                 ("untextured_metallic_reset", metal_reset)) if v}
    if repairs:
        report["material_repairs"] = repairs
    if inspect:
        return report
    if moving_object_keys and clip_policy == "all":
        raise ImportRefused(
            "these clips animate the armature object itself rather than its "
            f"bones: {moving_object_keys}. That is root motion the rig cannot "
            "carry (the armature object is bound to its link); re-export with "
            "the motion on the root bone, or pass --clips matching to drop them")
    for action in [a for a in actions if a.name in moving_object_keys]:
        dropped_clips[action.name] = "animates the armature object itself (root motion off the rig)"
        actions.remove(action)
        del before[action.name]
        bpy.data.actions.remove(action)
    if yaw == "auto":
        yaw = report["raw"]["facing"].get("yaw_to_front_deg")
        if yaw is None:
            raise ImportRefused(
                "--yaw auto could not measure which way the figure faces ("
                f"{report['raw']['facing']['why']}); pass --yaw explicitly")
    yaw = float(yaw)

    # Detach everything from its hierarchy, keeping where it is.
    for obj in (arm, *skinned):
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    for obj in list(bpy.data.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)

    rot = Matrix.Rotation(math.radians(yaw), 4, "Z") @ Matrix.Scale(user_scale, 4) @ upright
    rs = np.array(rot)[:3, :3]
    placed = raw_rest @ rs.T
    lo, hi = placed.min(axis=0), placed.max(axis=0)
    shift = Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    N = shift @ rot
    bone_space_factor = _uniform_scale(N @ arm.matrix_world, strict=clip_policy == "all")
    tolerance = max(SELF_CHECK_TOLERANCE_M, SELF_CHECK_TOLERANCE_FRACTION * float(hi[2] - lo[2]))
    if str(rest).startswith("clip-pose:"):
        # A whole pose taken into the skin is not exact under linear blend skinning (a vertex weighted to bones
        # that turned differently): the crow's TakeOff, rebased on its first frame, moved 2.5 mm on a 0.48 m bird.
        tolerance = max(tolerance, SELF_CHECK_POSE_FRACTION * float(max(hi - lo)))

    for obj in (arm, *skinned):
        obj.matrix_world = N @ obj.matrix_world
    hair = _scale_particle_hair(skinned, _uniform_scale(N, strict=False))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (arm, *skinned):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    rescaled, object_keys = 0, {}
    for action in actions:
        rescaled += _scale_location_channels(action, arm, bone_space_factor)
        gone = _drop_object_transform_channels(action)
        if gone:
            object_keys[action.name] = gone

    n4 = np.array(N)
    residuals = {}
    for action in actions:
        frame, posed = before[action.name]
        _set_action(arm, action, frame)
        worst = 0.0
        for obj in skinned:
            expect = posed[obj.name] @ n4[:3, :3].T + n4[:3, 3]
            worst = max(worst, float(np.abs(_evaluated_world(obj) - expect).max()))
        residuals[action.name] = worst
        if worst > tolerance:
            if clip_policy == "all":
                raise ImportRefused(
                    f"normalization changed the posed mesh by {worst:.3g} m (tolerance {tolerance:.3g} m) at "
                    f"{action.name!r} frame {frame}; refusing to write a rig "
                    "whose animation no longer matches its source (--clips "
                    "matching drops such clips and keeps the rest)")
            dropped_clips[action.name] = f"posed mesh off by {worst:.3g} m after normalization"
    for action in [a for a in actions if a.name in dropped_clips]:
        actions.remove(action)
        residuals.pop(action.name, None)
        bpy.data.actions.remove(action)
    _rest(arm)
    if blend_prep is not None:
        for _obj, mod in blend_prep["muted_modifiers"]:
            mod.show_viewport = True

    # One mesh: the largest is the join target, so its modifier stack wins.
    skinned.sort(key=lambda o: -len(o.data.vertices))
    body = skinned[0]
    uv_unified = _unify_uv_maps(skinned) if len(skinned) > 1 else []
    if len(skinned) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in skinned:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
    for mod in list(body.modifiers):
        if mod.type == "ARMATURE" and mod.object is not arm:
            body.modifiers.remove(mod)
    arm_mods = [m for m in body.modifiers if m.type == "ARMATURE"]
    for extra in arm_mods[1:]:
        body.modifiers.remove(extra)
    body.name = body.data.name = MESH_NAME
    arm.name = arm.data.name = ARMATURE_NAME
    weighted_loose = _weight_unweighted(body, arm)

    renamed = []
    for action in actions:
        clip = action.name
        action.name = f"{name}/{clip}"
        action.use_fake_user = True
        start, end = action.frame_range
        renamed.append({"name": action.name, "clip": clip,
                        "frame_start": float(start), "frame_end": float(end),
                        "fps": import_fps,
                        "seconds": round(float(end - start) / import_fps, 4)})
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    verts, tris, uvs, mat = _rest_arrays(body)
    bones = []
    for b in arm.data.bones:
        bones.append({"name": b.name,
                      "parent": b.parent.name if b.parent else None,
                      "head": [round(float(v), 6) for v in b.head_local],
                      "tail": [round(float(v), 6) for v in b.tail_local],
                      "deform": bool(b.use_deform),
                      # Rest frame (columns = the bone's axes, armature space),
                      # for `import_gltf.skeleton_anatomy`; not carried into the IR.
                      "rotation": [[round(float(v), 6) for v in row]
                                   for row in b.matrix_local.to_3x3()]})
    used = {g.name for g in body.vertex_groups}
    groups_without_bone = sorted(used - {b["name"] for b in bones})

    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in (arm, body):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        collection.objects.link(obj)
    bpy.context.scene.render.fps = int(round(import_fps))
    out.mkdir(parents=True, exist_ok=True)
    for img in list(bpy.data.images):
        # A texture that could not be found cannot be packed; drop it from the
        # file rather than let pack_all refuse the whole import over it.
        if img.filepath in missing_images and img.packed_file is None:
            bpy.data.images.remove(img)
    if bpy.data.images:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "character.blend"),
                                compress=True, check_existing=False)
    save = {"verts": verts, "tris": tris, "material": mat}
    if uvs is not None:
        save["uvs"] = uvs
    np.savez(out / "rest.npz", **save)

    report.update({
        "armature": ARMATURE_NAME, "mesh": MESH_NAME, "collection": name,
        "yaw_deg": yaw, "scale": user_scale,
        "normalisation": [list(r) for r in N],
        "bone_space_factor": bone_space_factor,
        "bone_heuristic": BONE_HEURISTIC if importer == "gltf" else None,
        "rescaled_location_channels": rescaled,
        "dropped_object_transform_keys": object_keys,
        "self_check": {"what": "each clip posed at its midpoint, before and "
                               "after normalization",
                       "max_residual_m": max(residuals.values(), default=None),
                       "per_clip_m": residuals,
                       "tolerance_m": tolerance},
        "clip_policy": clip_policy,
        "dropped_clips": dropped_clips,
        "bones": bones,
        "root_bones": [b["name"] for b in bones if b["parent"] is None],
        "vertex_groups_without_bone": groups_without_bone,
        "actions": renamed,
        "materials": _material_rows(body.data),
        "uv_maps_unified": uv_unified,
        "shape_keys": ([k.name for k in body.data.shape_keys.key_blocks]
                       if body.data.shape_keys else []),
        "facing": _facing(verts, _head_point(arm), _right_vector(arm)),
        "extent_m": [round(float(v), 4) for v in verts.max(axis=0) - verts.min(axis=0)],
        "authored_limits": _authored_limits(arm),
    })
    if weighted_loose:
        report["weighted_unweighted_vertices"] = weighted_loose
    if hair:
        report["particle_hair"] = hair
    return report


def main():
    job = json.loads(Path(sys.argv[-1]).read_text())
    try:
        report = {"ok": True, **run(job)}
    except ImportRefused as e:
        report = {"ok": False, "error": str(e)}
    Path(job["report"]).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
