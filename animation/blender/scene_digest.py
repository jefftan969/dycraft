"""Semantic fingerprint of the saved project, for the animation journal.

Computed inside the Blender worker right after `save_as_mainfile`, so it describes the bytes written. Diffing two
answers "what did this `h.blender` call change" without opening a `.blend`: the objects whose transform, parenting,
modifier stack, constraints, or materials moved, and the F-curves that gained, lost, or rewrote keyframes. A byte
diff of two `.blend` files answers nothing, since the format is pointer-laden and reordered on every save.

It is not the animation itself: each curve is reduced to a hash plus five summary numbers, because storing the
keyframes would cost a long run tens to hundreds of megabytes against a few. Replaying the journal's captured
source, or keeping the `.blend`, recovers the values. Keyframe hashing reads `co` alone, not the Bezier handles,
which cost 2.7x more to extract and do not change whether a curve changed.

On a film of a thousand objects and half a million keyframes a digest takes tens of milliseconds, under twenty
seconds over a run. Only operations that save pay it; `render`, `inspect`, `evaluate`, and `probe` never call it,
so the query loop that motivates a resident Blender session is untouched.

Blind spots: mesh edits that keep vertex and polygon counts, and material node graphs (materials are compared by
name). Rigid mechanisms move by transform, so the authored channels cover them; vertex-level deformation authoring
does not appear here.
"""
from __future__ import annotations

import hashlib
import json
import struct

#: Bumped when a field changes meaning. An inspection agent that diffs two
#: digests must refuse to compare across versions rather than report the schema
#: change as scene churn.
#:
#: 2: objects gained `nla`, and channels driven by an NLA strip or a non-first
#:    action slot now hash as animated rather than as their frame-dependent
#:    current value. Both change what an unchanged scene digests to.
SCHEMA = 2


#: Stands in for a channel an F-curve or driver owns. Such a channel's current
#: value is a *cache* of the last frame the depsgraph evaluated, not authored
#: state: an edit that sets `location.x = .05` and keys it saves .05, and the
#: next call re-opens the file, evaluates frame 1, and saves 0. Hashing the
#: evaluated value made every operation after an animated one report the object
#: as changed. The authored value lives in the curve, which is recorded
#: separately and exactly.
ANIMATED = float("-inf")

#: Authored transform channels, in the order they are hashed. `matrix_local`
#: is not used: it folds all of these together, so one animated component would
#: poison the whole transform.
TRANSFORM = (("location", 3), ("rotation_euler", 3), ("rotation_quaternion", 4),
             ("rotation_axis_angle", 4), ("scale", 3), ("delta_location", 3),
             ("delta_rotation_euler", 3), ("delta_rotation_quaternion", 4),
             ("delta_scale", 3))


def _hash(payload):
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _matrix(matrix):
    return _hash(struct.pack("<16d", *[v for row in matrix for v in row]))


def _slot_curves(action, slot):
    """An action's F-curves for one slot, or all of them for a legacy action.

    `action.fcurves` is the first slot's view (see `_iter_curves`). Where a
    binding names its slot -- an object's `action_slot`, an NLA strip's -- that
    is the slot whose curves actually drive it, and on a shared action the
    first slot belongs to somebody else.
    """
    layers = getattr(action, "layers", None)
    if not layers:
        return list(action.fcurves)
    return [curve for layer in layers for keyed in getattr(layer, "strips", ())
            for bag in getattr(keyed, "channelbags", ())
            if slot is None or getattr(bag, "slot", None) == slot
            for curve in bag.fcurves]


def _animated_channels(obj):
    """`(data_path, array_index)` pairs some curve or driver owns."""
    animation = obj.animation_data
    if animation is None:
        return frozenset()
    owned = set()
    groups = [animation.drivers]
    if animation.action is not None:
        groups.append(_slot_curves(animation.action,
                                   getattr(animation, "action_slot", None)))
    # An object animated only through NLA has no active action, so without this
    # its channels read as unanimated and `_transform_hash` records whatever
    # static value they happen to hold at `frame_current` -- a number nobody
    # authored, which changes with the frame the last job left the file on.
    for track in animation.nla_tracks:
        for strip in track.strips:
            if strip.action is not None:
                groups.append(_slot_curves(strip.action,
                                           getattr(strip, "action_slot", None)))
    for group in groups:
        for curve in group:
            owned.add((curve.data_path, curve.array_index))
    return owned


def _transform_hash(owner, animated, prefix=""):
    values = []
    for path, size in TRANSFORM:
        channel = getattr(owner, path, None)
        if channel is None:
            values.extend([ANIMATED] * size)
            continue
        for index in range(size):
            values.append(ANIMATED if (prefix + path, index) in animated
                          else float(channel[index]))
    return _hash(struct.pack(f"<{len(values)}d", *values))


def _plain(value):
    """A custom property as comparable JSON.

    Blender hands back `IDPropertyArray` and `IDPropertyGroup` wrappers rather
    than lists and dicts. `repr()` on those can carry a memory address, which
    would make two digests of an unchanged scene differ — so they are unwrapped
    first and `str` is the last resort, never `repr`.
    """
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "to_list"):
        value = value.to_list()
    try:
        return json.loads(json.dumps(value, default=str, sort_keys=True))
    except (TypeError, ValueError):
        return str(value)


def _custom_properties(owner, animated=frozenset()):
    """Authored state that is not a field.

    The scene carries `dycraft_declarations` here, which is the run's
    declared events and collision allowances: authored animation that no
    transform or F-curve records.
    """
    out = {}
    for key in owner.keys():
        if key.startswith("_"):
            continue
        if (f'["{key}"]', -1) in animated or (f'["{key}"]', 0) in animated:
            out[key] = None                       # authored by a curve instead
            continue
        out[key] = _plain(owner[key])
    return out


def _object_record(obj):
    animated = _animated_channels(obj)
    record = {
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "parent_type": obj.parent_type if obj.parent else None,
        "parent_bone": obj.parent_bone or None if obj.parent else None,
        "transform": _transform_hash(obj, animated),
        "rotation_mode": obj.rotation_mode,
        "parent_inverse": _matrix(obj.matrix_parent_inverse),
        "custom": _custom_properties(obj, animated),
        "hide_render": obj.hide_render,
        "hide_viewport": obj.hide_viewport,
        "data": obj.data.name if obj.data else None,
        "modifiers": [[m.name, m.type, m.show_render] for m in obj.modifiers],
        "constraints": [[c.name, c.type, c.mute, round(c.influence, 6)]
                        for c in obj.constraints],
    }
    data = obj.data
    if data is not None:
        materials = getattr(data, "materials", None)
        if materials is not None:
            record["materials"] = [m.name if m else None for m in materials]
        if obj.type == "MESH":
            record["verts"] = len(data.vertices)
            record["polys"] = len(data.polygons)
        keys = getattr(data, "shape_keys", None)
        if keys is not None:
            record["shape_keys"] = [[k.name, round(k.value, 6)] for k in keys.key_blocks]
    if obj.pose is not None:
        # One hash for the whole pose: a rig has hundreds of bones and naming
        # each would dominate the record, while a changed hash plus the object's
        # action is enough to say "the pose moved, look at these curves". Bone
        # channels are excluded the same way object channels are, and for the
        # same reason.
        record["pose"] = _hash(b"".join(
            _transform_hash(bone, animated,
                            prefix=f'pose.bones["{bone.name}"].').encode()
            for bone in obj.pose.bones))
        record["pose_bones"] = len(obj.pose.bones)
    animation = obj.animation_data
    if animation is not None:
        record["action"] = animation.action.name if animation.action else None
        if animation.nla_tracks:
            # Placement, not content. A strip says when a clip plays, how fast,
            # and which action it plays; moving, scaling, or swapping one
            # retimes the film without touching a single keyframe. The curve
            # table above would report that as no change at all, so `diff()`
            # would return `None` for the operation that re-cut the sequence.
            record["nla"] = [[track.name, track.mute, [[
                strip.name, strip.action.name if strip.action else None,
                getattr(getattr(strip, "action_slot", None), "identifier", None),
                round(strip.frame_start, 6), round(strip.frame_end, 6),
                round(strip.action_frame_start, 6), round(strip.action_frame_end, 6),
                round(strip.scale, 6), round(strip.repeat, 6),
                round(strip.influence, 6), strip.blend_type,
                strip.extrapolation, strip.mute] for strip in track.strips]]
                for track in animation.nla_tracks]
        if animation.drivers:
            # Drivers author motion the F-curve tables above do not carry.
            record["drivers"] = _hash("\x00".join(
                f"{d.data_path}[{d.array_index}]:"
                f"{getattr(d.driver, 'type', '')}:{getattr(d.driver, 'expression', '')}"
                for d in animation.drivers).encode())
    return record


def _curve_record(curve):
    count = len(curve.keyframe_points)
    coords = [0.0] * (count * 2)
    curve.keyframe_points.foreach_get("co", coords)
    record = {"n": count, "h": _hash(struct.pack(f"<{count * 2}f", *coords)),
              "mute": curve.mute}
    if count:
        frames, values = coords[0::2], coords[1::2]
        # The five numbers that make a changed hash legible without the values:
        # when the curve starts and ends, the range it spans, and where it sits.
        record["stats"] = [round(min(frames), 4), round(max(frames), 4),
                           round(min(values), 6), round(max(values), 6),
                           round(sum(values) / count, 6)]
    return record


def _iter_curves(action):
    """Every F-curve in an action, yielded as `(slot identifier, curve)`.

    `action.fcurves` is a backwards-compatibility view of the **first slot
    only**. Blender 4.5 makes every action layered, and one action shared
    between two objects gets a slot per object, whose curves would then vanish
    from this log without a word. Silent loss
    is the one failure a diagnostic record cannot have, so the layered API is
    walked when it exists and `fcurves` is the fallback for legacy actions.
    """
    layers = getattr(action, "layers", None)
    if not layers:
        for curve in action.fcurves:
            yield "", curve
        return
    for layer in layers:
        for strip in getattr(layer, "strips", ()):
            for bag in getattr(strip, "channelbags", ()):
                slot = getattr(bag, "slot", None)
                label = getattr(slot, "identifier", "") or ""
                for curve in bag.fcurves:
                    yield label, curve


def scene_digest(bpy):
    """Fingerprint every authored value that decides what the film looks like."""
    objects = {obj.name: _object_record(obj) for obj in bpy.data.objects}
    curves = {}
    for action in bpy.data.actions:
        for label, curve in _iter_curves(action):
            key = "|".join(filter(None, (action.name, label,
                                         f"{curve.data_path}[{curve.array_index}]")))
            curves[key] = _curve_record(curve)
    scene = bpy.context.scene
    return {
        "schema": SCHEMA,
        "scene": {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "fps": scene.render.fps,
            "fps_base": round(scene.render.fps_base, 6),
            "camera": scene.camera.name if scene.camera else None,
            "engine": scene.render.engine,
            "markers": [[m.name, m.frame, m.camera.name if m.camera else None]
                        for m in scene.timeline_markers],
            # `dycraft_declarations` lives here: the run's declared events and
            # collision allowances, which decide what `check_motion` enforces.
            "custom": _custom_properties(scene),
        },
        "objects": objects,
        "curves": curves,
        "counts": {name: len(getattr(bpy.data, name)) for name in
                   ("objects", "meshes", "materials", "actions", "images",
                    "collections", "armatures", "cameras", "lights")},
        "totals": {"keyframes": sum(c["n"] for c in curves.values()),
                   "curves": len(curves), "objects": len(objects)},
    }


def diff(before, after):
    """What changed between two digests, in the shape an inspection agent reads.

    Returns `None` for an identical pair, so a caller can tell "this operation
    did not touch the scene" from "this operation changed something small" —
    the distinction that makes bisecting a change cheap.
    """
    if before is None or after is None:
        return None
    if before.get("schema") != after.get("schema"):
        raise ValueError("digest schema mismatch; these two are not comparable")
    if before == after:
        return None
    changed = {}
    for field in ("objects", "curves"):
        old, new = before[field], after[field]
        added = sorted(k for k in new if k not in old)
        removed = sorted(k for k in old if k not in new)
        modified = sorted(k for k in new if k in old and new[k] != old[k])
        if added or removed or modified:
            changed[field] = {"added": added, "removed": removed,
                              "modified": modified}
    if before["scene"] != after["scene"]:
        changed["scene"] = {k: [before["scene"][k], after["scene"][k]]
                            for k in after["scene"]
                            if before["scene"].get(k) != after["scene"][k]}
    if before["counts"] != after["counts"]:
        changed["counts"] = {k: [before["counts"][k], after["counts"][k]]
                             for k in after["counts"]
                             if before["counts"].get(k) != after["counts"][k]}
    return changed or None
