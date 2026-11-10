"""The room shell: which box is which surface, and what leaves the room."""
from __future__ import annotations

import numpy as np

try:
    from .continuity import frame_runs
except ImportError:  # Read-only modules staged into isolated Blender.
    from continuity import frame_runs


def out_of_room(escapes, room, faces=None):
    """Room escapes grouped by object, with the frames each was outside.

    Same argument as `penetration_pairs`: one entry per offending object
    rather than one per object per frame.

    Unlike a collision, this test *is* a bounds test — there is no mesh
    question being approximated, the finding is that an object's extent left
    the room volume. Even so the box itself does not go out, because six
    coordinates invite being read as geometry. `outside_by_m` is the answer
    instead: how far past the nearest room face the worst frame reached.

    `faces` is `room_containment`'s `{(axis, sign): tag}`, and only those
    faces are measured. An `open_top` room has no ceiling, and a tall object
    standing in it is not "outside" by its height above the walls. None means
    all six, which is a capped room.
    """
    lo_room, hi_room = np.asarray(room[0], dtype=float), np.asarray(room[1], dtype=float)
    low = [axis for axis in range(3) if faces is None or (axis, -1) in faces]
    high = [axis for axis in range(3) if faces is None or (axis, +1) in faces]
    grouped = {}
    for escape in escapes:
        grouped.setdefault(escape["object"], []).append(escape)
    rows = []
    for name, members in grouped.items():
        overshoot = max(max([float(lo_room[axis] - member["aabb"][0][axis]) for axis in low]
                            + [float(member["aabb"][1][axis] - hi_room[axis]) for axis in high])
                        for member in members)
        rows.append({"object": name, "reason": "outside room",
                     "frames": frame_runs(sorted(member["frame"] for member in members)),
                     "frame_count": len(members),
                     "outside_by_m": round(overshoot, 6)})
    rows.sort(key=lambda row: (-row["frame_count"], -row["outside_by_m"], row["object"]))
    return rows


#: The room shell is six boxes. Which one a given box is, read off its own
#: extent against the room's, so this works on a scene that was rigged before
#: the surfaces carried a tag — including every scene already committed.
_ROOM_AXIS_TAGS = (("wall_xmin", "wall_xmax"), ("wall_ymin", "wall_ymax"),
                   ("floor", "ceiling"))


def room_surface_tag(lo, hi, room):
    """`floor`, `ceiling`, `wall_xmin`, ... for one shell box, or None."""
    if not room:
        return None
    extent = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    axis = int(np.argmin(extent))
    centre = (float(lo[axis]) + float(hi[axis])) / 2.0
    middle = (float(room[0][axis]) + float(room[1][axis])) / 2.0
    return _ROOM_AXIS_TAGS[axis][0 if centre < middle else 1]


def resolve_room_surfaces(boxes, room=None):
    """`{object_name: tag}` for the room shell, however little of it exists.

    **`baseline["room"]` is absent for every uncapped room**, which is why this function exists.
    `render/blender_renderer.py` emits the room box only `if plan.room_info.get("capped")`, and `capped` is
    `"ceiling" not in suppressed` (`render/_world.py`), so `open_top` and `open_air` both arrive with `room=None`.
    `room_surface_tag` alone would then drop the shell body from `bodies`, and `room.floor`, which the layer prompt
    documents as an event operand, would resolve to nothing. `open_air` is the common case.

    Two sources, in order of authority: the declared room box (`room_surface_tag`), exact whenever the room is capped,
    and inference from the surfaces. For a full shell the union of the surface boxes is the room and a midpoint test
    works; for a single surface the union is that surface, so a side is read off the union only when the axis carries
    more than one surface (a lone floor slab would otherwise come out `ceiling`). With exactly one:

    * a horizontal surface is the **floor**: the enclosure vocabulary
      (`sdk/scene_builder.py::_ENCLOSURE_SUPPRESSES`) can suppress the ceiling and never the floor;
    * a lone vertical surface is **unresolved**: no enclosure produces one (`open_air` removes all four walls,
      `open_top` none), so it can only come from an explicit `suppressed_surfaces` list, with no honest default.

    `boxes` is `{object_name: (lo, hi)}` in world space.
    """
    if not boxes:
        return {}
    if room:
        return {name: tag for name, (lo, hi) in boxes.items()
                if (tag := room_surface_tag(lo, hi, room)) is not None}
    axes = {}
    for name, (lo, hi) in boxes.items():
        extent = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
        axis = int(np.argmin(extent))
        axes.setdefault(axis, []).append(
            (name, (float(lo[axis]) + float(hi[axis])) / 2.0))
    out = {}
    for axis, members in axes.items():
        if len(members) == 1:
            # See the docstring: horizontal is the floor by construction,
            # vertical has no defensible default.
            if axis == 2:
                out[members[0][0]] = "floor"
            continue
        middle = (min(c for _, c in members) + max(c for _, c in members)) / 2.0
        for name, centre in members:
            out[name] = _ROOM_AXIS_TAGS[axis][0 if centre < middle else 1]
    return out


#: Every tag `resolve_room_surfaces` can return. Mirrors
#: `render/_world.ROOM_SURFACE_TAGS`; restated rather than imported because
#: this module runs inside the Blender interpreter, where `render` is not on
#: the path.
ROOM_SURFACE_TAGS = tuple(tag for pair in _ROOM_AXIS_TAGS for tag in pair)


def replaced_surfaces(bodies, replaced, lookup):
    """`{"room.<surface>": body index}` for shell surfaces an instance replaced.

    `replace_room_surface` removes the shell surface, so no body is tagged
    `room.floor`, and an event naming it would resolve to nothing on every
    frame: never checked, neither passed nor failed. The prompt tells authors
    to model an interior's floor and declare it this way, and documents
    `room.floor` as the ground of `supported` and `planted`.

    The surface's name stands for the replacing instance's part that spans the
    most of it: the largest bounding-box face across the surface's normal, so a
    floor's slab rather than its skirting. A surface still in the shell keeps
    its shell body.
    """
    out = {}
    for tag, instance in replaced.items():
        alias = "room." + tag
        if alias in lookup:
            continue
        axis = next(i for i, pair in enumerate(_ROOM_AXIS_TAGS) if tag in pair)
        across = [a for a in range(3) if a != axis]
        best = None
        for index, entry in enumerate(bodies):
            if entry[0].split("/")[0] != instance:
                continue
            extent = np.asarray(entry[4], dtype=float) - np.asarray(entry[3], dtype=float)
            area = float(extent[across[0]] * extent[across[1]])
            if best is None or area > best[0]:
                best = (area, index)
        if best is not None:
            out[alias] = best[1]
    return out


def room_containment(boxes, tags):
    """`(lo, hi, faces)` — the volume a body must stay inside, and which
    faces of it are real geometry.

    `faces` is `{(axis, sign): tag}` for the surfaces that exist, and it is
    the point of this function. A single containment box would ask a six-sided
    question of a room that may have one side. An `open_air` scene has a floor and nothing else: there is no
    containment obligation to check, and reporting one would be inventing four
    walls. An `open_top` scene has four walls and a floor, and escaping
    sideways through them is still a defect — so the test runs per face, over
    the faces that were actually built.

    The returned box is the interior: the inward face of each shell surface,
    and the union's extent on any axis with no surface.
    """
    if not boxes or not tags:
        return None, None, {}
    named = {tag: boxes[name] for name, tag in tags.items() if name in boxes}
    lo = np.min([np.asarray(b[0], dtype=float) for b in boxes.values()], axis=0)
    hi = np.max([np.asarray(b[1], dtype=float) for b in boxes.values()], axis=0)
    faces = {}
    for axis, (low_tag, high_tag) in enumerate(_ROOM_AXIS_TAGS):
        if low_tag in named:
            lo[axis] = float(named[low_tag][1][axis])
            faces[(axis, -1)] = low_tag
        if high_tag in named:
            hi[axis] = float(named[high_tag][0][axis])
            faces[(axis, +1)] = high_tag
    return lo, hi, faces


def room_shell(record, geometry, baseline):
    """`(tags, (lo, hi, faces))` for the room shell on one frame.

    Called once per `verify()`, on the first checked frame: the shell is
    static geometry, and re-tagging it every frame would pay for a constant.

    A capped room keeps exactly the test it always had — the declared box,
    all six faces. Otherwise the faces are the surfaces that were built, and
    **a floor alone encloses nothing**: `open_air` is a ground plane, so it
    has no containment obligation, and inventing four walls to test against
    would report a figure walking off-set as "outside room". Sinking through
    that floor goes unreported here, exactly as it did before surfaces were
    resolved at all; it is a support question, not a containment one.
    """
    room_objects = set(baseline.get("room_objects", []))
    boxes = {}
    for obj in record["objects"]:
        if obj.get("hidden") or not (obj.get("room") or obj["name"] in room_objects):
            continue
        verts, _ = geometry[obj["name"]]
        if not len(verts):
            continue
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        if np.isfinite(lo).all() and np.isfinite(hi).all():
            boxes[obj["name"]] = (lo, hi)
    tags = resolve_room_surfaces(boxes, baseline.get("room"))
    room = baseline.get("room")
    if room:
        faces = {(axis, sign): tag for axis, pair in enumerate(_ROOM_AXIS_TAGS)
                 for sign, tag in zip((-1, +1), pair)}
        return tags, (np.asarray(room[0], dtype=float), np.asarray(room[1], dtype=float), faces)
    lo, hi, faces = room_containment(boxes, tags)
    if set(faces.values()) <= {"floor"}:
        faces = {}
    return tags, (lo, hi, faces)


def escaped_faces(lo, hi, containment, tolerance=1e-4):
    """The room faces an object's box has crossed, by tag; empty if none."""
    room_lo, room_hi, faces = containment
    return [tag for (axis, sign), tag in faces.items()
            if (lo[axis] < room_lo[axis] - tolerance if sign < 0
                else hi[axis] > room_hi[axis] + tolerance)]
