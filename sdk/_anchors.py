"""Auto-AABB anchor vocabulary, shared by the object and scene SDKs.

Anchors are named *points* on an entity's AABB. They have no per-anchor
rotation — the entity's local frame is the single reference frame for any
offset interpretation.

Axis convention (part-local, viewer-relative):

    +X = left,   -X = right
    +Y = front,  -Y = back
    +Z = top,    -Z = bottom

This matches the object-agent prompt's viewer convention (viewer at
+Y_world looking at origin → the +Y face of an object presents itself to
the viewer as the object's "front"). The same convention applies to scene
instances, groups, and the room.

27 anchors per AABB:

    * 1 center
    * 6 face centers
    * 12 edge midpoints
    * 8 corners

Naming rule for compound anchors: components appear in axis order Z → Y → X
(`top_front`, never `front_top`; `top_back_right`, never `right_back_top`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping


__all__ = [
    "AUTO_AABB_ANCHOR_NAMES",
    "FACE_ANCHOR_NAMES",
    "EDGE_ANCHOR_NAMES",
    "CORNER_ANCHOR_NAMES",
    "AUTO_AABB_ANCHOR_GROUPS",
    "builtin_anchor_groups",
    "builtin_anchor_names_text",
    "auto_aabb_anchor_position",
    "outward_direction",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Faces — outward direction is a single +/- part-local axis.
FACE_ANCHOR_NAMES = frozenset({
    "top", "bottom",
    "front", "back",
    "left", "right",
})

# Edges — midpoint of two face centers; outward is the normalized bisector
# of the two adjacent face normals.
EDGE_ANCHOR_NAMES = frozenset({
    "top_front", "top_back", "top_left", "top_right",
    "bottom_front", "bottom_back", "bottom_left", "bottom_right",
    "front_left", "front_right", "back_left", "back_right",
})

# Corners — outward is the normalized diagonal from the AABB centroid to
# the corner.
CORNER_ANCHOR_NAMES = frozenset({
    "top_front_left", "top_front_right",
    "top_back_left", "top_back_right",
    "bottom_front_left", "bottom_front_right",
    "bottom_back_left", "bottom_back_right",
})

AUTO_AABB_ANCHOR_NAMES = (
    frozenset({"center"})
    | FACE_ANCHOR_NAMES
    | EDGE_ANCHOR_NAMES
    | CORNER_ANCHOR_NAMES
)
# Sanity: 1 + 6 + 12 + 8 = 27.
assert len(AUTO_AABB_ANCHOR_NAMES) == 27

#: The 27 built-in names grouped the way they are composed, for errors that
#: name a bad anchor, so an author can read the vocabulary off the error that
#: rejected a guess at it.
AUTO_AABB_ANCHOR_GROUPS = {"center": ["center"],
                           "faces": sorted(FACE_ANCHOR_NAMES),
                           "edges": sorted(EDGE_ANCHOR_NAMES),
                           "corners": sorted(CORNER_ANCHOR_NAMES)}


def builtin_anchor_groups():
    """`AUTO_AABB_ANCHOR_GROUPS`, as a fresh copy for a report's evidence."""
    return {group: list(names) for group, names in AUTO_AABB_ANCHOR_GROUPS.items()}


def builtin_anchor_names_text():
    """The built-in names as one line of an error message, by group."""
    return "; ".join(f"{group}: {', '.join(names)}"
                     for group, names in AUTO_AABB_ANCHOR_GROUPS.items())


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------

# Per-token coordinate selectors. The first element of the tuple picks
# "min" / "max" / "center" along the named axis. Used to compose names
# like "top_front_left" → ((+Z=max), (+Y=max), (+X=max)).
#
# Viewer-relative axis convention:
#     top    = +Z = zmax
#     bottom = -Z = zmin
#     front  = +Y = ymax
#     back   = -Y = ymin
#     left   = +X = xmax
#     right  = -X = xmin
_TOKEN_TO_AXIS_AND_SIDE = {
    "top":    ("z", "max"),
    "bottom": ("z", "min"),
    "front":  ("y", "max"),
    "back":   ("y", "min"),
    "left":   ("x", "max"),
    "right":  ("x", "min"),
}


def _axis_value(side: str, lo: float, hi: float, center: float) -> float:
    if side == "max":
        return hi
    if side == "min":
        return lo
    return center


def _unknown_anchor_message(anchor_name: str) -> str:
    """Why an anchor name failed, and what to write instead.

    A bare `KeyError('tcp')` sent one agent through three probe round-trips
    working out that the scene and animation layers see only the 27 auto-AABB
    names — the `tcp` / `pad` contact frames an object author declares are
    object-layer records and are not resolvable from a scene. Say so.
    """
    return (f"{anchor_name!r} is not an auto-AABB anchor name. Use 'center' "
            f"or up to one token per axis: x=left/right, y=front/back, "
            f"z=top/bottom (e.g. 'top_front_left'). Named ObjectIR contact "
            f"frames such as 'tcp' are unavailable here; use a part landmark "
            f"such as '<part>.bottom'.")


def auto_aabb_anchor_position(aabb, anchor_name: str) -> tuple:
    """Position of `anchor_name` on AABB, in the AABB's local frame.

    AABB shape: ((xmin, ymin, zmin), (xmax, ymax, zmax)).

    Raises KeyError(anchor_name) if the name is not in the auto-AABB
    vocabulary (callers should pre-validate via membership in
    AUTO_AABB_ANCHOR_NAMES if they want a structured error).
    """
    (xmn, ymn, zmn), (xmx, ymx, zmx) = aabb
    cx = 0.5 * (xmn + xmx)
    cy = 0.5 * (ymn + ymx)
    cz = 0.5 * (zmn + zmx)

    if anchor_name == "center":
        return (cx, cy, cz)

    tokens = anchor_name.split("_")
    # Validate composition: every token must be a known axis-token, and
    # no axis may be repeated.
    axes_used: dict[str, str] = {}   # axis ("x"/"y"/"z") -> side
    for tok in tokens:
        if tok not in _TOKEN_TO_AXIS_AND_SIDE:
            raise KeyError(_unknown_anchor_message(anchor_name))
        axis, side = _TOKEN_TO_AXIS_AND_SIDE[tok]
        if axis in axes_used:
            raise KeyError(
                f"{anchor_name!r}: {axis.upper()} is named twice; an "
                f"auto-AABB anchor uses each axis at most once")
        axes_used[axis] = side

    # Compose coordinates. Any axis not appearing among the tokens uses
    # the centroid value along that axis.
    x = _axis_value(axes_used.get("x", "center"), xmn, xmx, cx)
    y = _axis_value(axes_used.get("y", "center"), ymn, ymx, cy)
    z = _axis_value(axes_used.get("z", "center"), zmn, zmx, cz)
    return (x, y, z)


# ---------------------------------------------------------------------------
# Outward direction
# ---------------------------------------------------------------------------

# +Z=top, +Y=front, +X=left. Each token contributes a unit-magnitude
# component along its axis with the sign of its side.
_TOKEN_TO_UNIT_AXIS = {
    "top":    (0.0, 0.0, +1.0),
    "bottom": (0.0, 0.0, -1.0),
    "front":  (0.0, +1.0, 0.0),
    "back":   (0.0, -1.0, 0.0),
    "left":   (+1.0, 0.0, 0.0),
    "right":  (-1.0, 0.0, 0.0),
}


class UnknownAnchor(KeyError):
    """No such anchor on this part — neither built-in nor registered.

    Subclasses ``KeyError`` because `auto_aabb_anchor_position` has always
    raised that for an unknown name, and callers already catch it.
    """


def find_anchor_record(anchors, anchor_name: str):
    """Look up a persistent semantic anchor record, or None.

    Accepts either the builder's ``{name: record}`` mapping or the IR's
    ``[record, ...]`` list, so build-time and load-time callers share one
    lookup instead of each writing their own.

    **`Mapping`, not `dict`, and that distinction is the whole function.** The
    probe overlays hand sandboxed code a `_readonly.freeze()` view of the IR
    and the catalog, and a `_ReadOnlyMapping` fails `isinstance(x, dict)`. With
    a `dict` check every declared anchor would read as absent from every probe
    while a verifier (which sees raw dicts) resolves it, and an instance placed
    on a declared anchor could not resolve a world pose, so most scene probes
    would raise `UnknownAnchor`. `_scene_kernel._anchor_dict_offset_surface`
    applies the same rule.
    """
    if not anchors:
        return None
    if isinstance(anchors, Mapping):
        return anchors.get(anchor_name)
    for record in anchors:
        if isinstance(record, Mapping) and record.get("name") == anchor_name:
            return record
    return None


def resolve_base_frame(aabb, anchors, anchor_name: str, *, part_name=None):
    """The base ``Anchor -> Part`` SE3 for `anchor_name`, before any offset.

    **The single resolution point for both anchor kinds.** Two callers use it —
    `object_builder._anchor_to_se3` at build time and
    `_object_kernel.anchor_position` at verify time — so neither derives the
    AABB landmark itself or needs separate teaching about semantic anchors.

    Resolution order, and it is unambiguous because the 27 built-in names are
    reserved and `Object.add_anchor` refuses them:

      1. a built-in AABB name -> the landmark on `aabb`, identity rotation;
      2. otherwise a persistent semantic anchor on that part;
      3. otherwise `UnknownAnchor`, naming the part and what *is* available.

    Returns ``((tx, ty, tz), (qw, qx, qy, qz))`` — internal SE3 form. A
    built-in yields identity rotation, so composing an offset onto it gives
    exactly the AABB landmark plus that offset.
    """
    if anchor_name in AUTO_AABB_ANCHOR_NAMES:
        return (auto_aabb_anchor_position(aabb, anchor_name),
                (1.0, 0.0, 0.0, 0.0))

    record = find_anchor_record(anchors, anchor_name)
    if record is None:
        known = sorted(
            (anchors or {}).keys() if isinstance(anchors, Mapping)
            else [r.get("name") for r in (anchors or [])
                  if isinstance(r, Mapping)]
        )
        where = f" on part {part_name!r}" if part_name else ""
        raise UnknownAnchor(
            f"unknown anchor {anchor_name!r}{where}: it is not one of the "
            f"{len(AUTO_AABB_ANCHOR_NAMES)} built-in AABB anchors and no "
            f"semantic anchor of that name is registered"
            + (f"; this part has {known}" if known
               else "; this part has no semantic anchors")
        )

    rot = record.get("rotation") or (1.0, 0.0, 0.0, 0.0)
    return (tuple(float(v) for v in record["translation"]),
            tuple(float(v) for v in rot))


def outward_direction(anchor_name: str) -> tuple:
    """Unit outward direction for `anchor_name` in the entity's local frame.

    For face anchors this is the outward face normal. For edges, the
    normalized bisector of the two adjacent face normals. For corners,
    the normalized diagonal from AABB centroid to the corner.

    Raises ValueError("center anchor has no outward direction") for
    "center". Raises KeyError(anchor_name) for unknown names.
    """
    if anchor_name == "center":
        raise ValueError("center anchor has no outward direction")
    if anchor_name not in AUTO_AABB_ANCHOR_NAMES:
        raise KeyError(anchor_name)
    tokens = anchor_name.split("_")
    vx = vy = vz = 0.0
    for tok in tokens:
        ux, uy, uz = _TOKEN_TO_UNIT_AXIS[tok]
        vx += ux
        vy += uy
        vz += uz
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    # By construction the sum of orthonormal unit vectors with distinct
    # axes has norm in {1, sqrt(2), sqrt(3)} for faces / edges / corners.
    return (vx / norm, vy / norm, vz / norm)
