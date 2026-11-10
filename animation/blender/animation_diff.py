"""What an edit's animation changed: curves, drivers, and strips before and after, as output frames."""
import math

import bpy


#: Datablock collections whose animation can reach a frame.
_ANIMATED_COLLECTIONS = ("objects", "materials", "node_groups", "lights", "cameras", "worlds",
                         "scenes", "shape_keys", "meshes", "curves", "armatures", "textures",
                         "particles")


def animation_snapshot():
    """Every curve, driver, and NLA strip in the file, as rows to compare.

    `curves` maps `(owner, action, path, index)` to its keys; `strips` maps
    `(owner, track, strip)` to its placement and output frame span; `drivers`
    maps `(owner, path, index)` to its expression and variables. A curve of
    an action is listed under every strip, or active slot, that plays it,
    with the output frames that play reaches.
    """
    scene = bpy.context.scene
    start = scene.frame_start
    curves, strips, drivers = {}, {}, {}
    for collection in _ANIMATED_COLLECTIONS:
        for idb in getattr(bpy.data, collection, ()):
            for owner in (idb, getattr(idb, "node_tree", None)):
                data = getattr(owner, "animation_data", None) if owner is not None else None
                if data is None:
                    continue
                label = f"{collection}:{owner.name}" + (":tree" if owner is not idb else "")
                # `None` reach: the active action, whose keys are output frames.
                plays = [(data.action, None, _played_slot(data))] if data.action else []
                for track in data.nla_tracks:
                    for strip in track.strips:
                        span = (strip.frame_start - start, strip.frame_end - start)
                        extrapolation = getattr(strip, "extrapolation", "NOTHING")
                        reach = (-math.inf if extrapolation == "HOLD" else span[0],
                                 math.inf if extrapolation in ("HOLD", "HOLD_FORWARD") else span[1])
                        strips[(label, track.name, strip.name)] = (
                            strip.action.name if strip.action else None, strip.frame_start,
                            strip.frame_end, strip.scale, strip.repeat, strip.action_frame_start,
                            strip.action_frame_end, strip.blend_in, strip.blend_out, strip.mute,
                            strip.influence, strip.blend_type, extrapolation, track.mute,
                            getattr(getattr(strip, "action_slot", None), "identifier", None), reach)
                        if strip.action is not None:
                            plays.append((strip.action, reach, _played_slot(strip)))
                for action, reach, slot in plays:
                    for curve in _action_fcurves(action, slot):
                        curves[(label, action.name, curve.data_path, curve.array_index, reach)] = tuple(
                            (point.co[0], point.co[1], point.interpolation,
                             tuple(point.handle_left), tuple(point.handle_right))
                            for point in curve.keyframe_points) + (_curve_settings(curve),)
                for curve in data.drivers:
                    driver = curve.driver
                    drivers[(label, curve.data_path, curve.array_index)] = (
                        driver.type, driver.expression,
                        tuple((v.name, v.type, tuple((t.id.name if t.id else None, t.data_path)
                                                     for t in v.targets)) for v in driver.variables))
    return {"curves": curves, "strips": strips, "drivers": drivers}


def _curve_settings(curve):
    """What shapes a curve besides its keys: extrapolation, mute, and every modifier's settings."""
    def values(struct):
        out = []
        for prop in struct.bl_rna.properties:
            if prop.identifier == "rna_type" or prop.type not in (
                    "BOOLEAN", "INT", "FLOAT", "ENUM", "STRING"):
                continue
            value = getattr(struct, prop.identifier)
            if isinstance(value, set):
                value = tuple(sorted(value))
            elif not isinstance(value, (str, bool, int, float)):
                value = tuple(value)
            out.append((prop.identifier, value))
        return tuple(out)
    return (curve.extrapolation, curve.mute, tuple(values(m) for m in curve.modifiers))


#: `_played_slot` before Blender had action slots: every curve plays.
_ANY_SLOT = object()


def _played_slot(holder):
    """The slot an `AnimData` or NLA strip plays (4.4+), else `_ANY_SLOT`."""
    return getattr(holder, "action_slot", _ANY_SLOT)


def _action_fcurves(action, slot=_ANY_SLOT):
    """The F-Curves `slot` plays of an action: its own channelbags on 4.4+, else the flat list.

    Only the owner's own slot. One action can hold every object's curves in
    hundreds of slots, and listing all of them for every user would make each
    edit's two snapshots millions of curves long.
    """
    layers = list(getattr(action, "layers", ()) or ())
    if layers and slot is not _ANY_SLOT:
        if slot is None:
            return []
        return [curve for layer in layers for strip in layer.strips
                for bag in ([strip.channelbag(slot)] if hasattr(strip, "channelbag") else [])
                if bag is not None for curve in bag.fcurves]
    found = [curve for layer in layers
             for strip in layer.strips for bag in getattr(strip, "channelbags", ()) or ()
             for curve in bag.fcurves]
    return found or list(getattr(action, "fcurves", ()))


def animation_changed(before, after, n_frames):
    """`{"changed_frames": [[lo, hi], ...], "n_frames": n}`: the output frames an
    edit's animation reaches, and the film's length after it.

    On an active action a changed key reaches from the key before it to the
    key after it on its curve (all the way out where it is the first or the
    last). On an action a strip plays, whose keys are in the action's own
    time, any change reaches the strip's whole span. A changed strip reaches
    its span before and after. A changed driver, or a curve added or removed,
    reaches its whole play. `span_report` compares these with the
    span an edit declared. `n_frames` is the film's length after the edit.
    """
    n = n_frames
    reached = []

    def reach(lo, hi, clip):
        lo, hi = max(lo, clip[0], 0), min(hi, clip[1], n - 1)
        if lo <= hi:
            reached.append((math.floor(lo), math.ceil(hi)))

    start = bpy.context.scene.frame_start
    for key in set(before["curves"]) | set(after["curves"]):
        old, new = before["curves"].get(key), after["curves"].get(key)
        if old == new:
            continue
        clip = key[4] or (-math.inf, math.inf)
        if old is None or new is None or old[-1] != new[-1] or key[4] is not None:
            reach(-math.inf, math.inf, clip)
            continue
        # A key on one side only -- added, moved, re-shaped, or deleted --
        # reaches from its neighbor before to its neighbor after.
        for keys, other in ((old[:-1], set(new[:-1])), (new[:-1], set(old[:-1]))):
            frames = [point[0] - start for point in keys]
            for k, point in enumerate(keys):
                if point in other:
                    continue
                reach(frames[k - 1] if k else -math.inf,
                      frames[k + 1] if k + 1 < len(frames) else math.inf, clip)
    for key in set(before["strips"]) | set(after["strips"]):
        old, new = before["strips"].get(key), after["strips"].get(key)
        if old != new:
            for row in (old, new):
                if row is not None:
                    reach(*row[-1], (-math.inf, math.inf))
    if any(before["drivers"].get(key) != after["drivers"].get(key)
           for key in set(before["drivers"]) | set(after["drivers"])):
        reach(-math.inf, math.inf, (-math.inf, math.inf))
    runs = []
    for lo, hi in sorted(reached):
        if runs and lo <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], hi)
        else:
            runs.append([lo, hi])
    return {"changed_frames": runs, "n_frames": n}
