"""Sugar assertion expansion helpers (layer-agnostic).

Sugar assertions are user-facing convenience wrappers that emit one or
more *base* assertion records at SDK build time. The IR kernel never
sees a "sugar kind" — it always sees `distance` / `collinear` / `overlap`
/ `angle` / `axis_aligned` records. Failure reports preserve intent
through auto-suffixed names of the form `<user_name>__<sugar_kind>_<aspect>`.

Each helper here is called by both `Object` and `Scene` builders. They
share `assert_distance / assert_collinear / assert_overlap / assert_angle`
on the calling builder; `assert_axis_world_aligned` additionally calls
the private `_assert_axis_aligned` helper on `Object` (the unsigned
axis-vs-world record kind). Validation of refs / numerics is handled by
the underlying base-assertion methods.
"""

from __future__ import annotations

from . import _assertions
from . import _readonly as _ro



# ---------------------------------------------------------------------------
# Suffix scheme
# ---------------------------------------------------------------------------

def _suffix(base_name: str, sugar_kind: str, aspect: str | None = None) -> str:
    """Auto-suffix a sugar's base name for a single expanded record.

    Format: `<base_name>__<sugar_kind>[_<aspect>]`. Examples:
        ('drawer_in_cabinet', 'within', 'hi_x')   -> 'drawer_in_cabinet__within_hi_x'
        ('door_seated',        'contact', None)    -> 'door_seated__contact'
        ('flush_lid_panel',    'flush',   'angle') -> 'flush_lid_panel__flush_angle'
    """
    return (f"{base_name}__{sugar_kind}_{aspect}"
            if aspect is not None
            else f"{base_name}__{sugar_kind}")


# ---------------------------------------------------------------------------
# Sugar bodies (return list of AssertionRefs)
# ---------------------------------------------------------------------------

def assert_contact(builder, *, a, b, tol: float, name: str):
    """`assert_distance(a, b, axis=None, min=0, max=tol)`."""
    return [builder.assert_distance(
        a=a, b=b, min=0.0, max=float(tol),
        name=_suffix(name, "contact"),
    )]


def _axis_unit(axis_char: str):
    """Unit vector for an axis character.

    Reads the shared `_assertions.AXIS_LITERALS` table rather than a private
    copy: this module's `axes=("x", "y", "z")` parameter is where authors learn
    that an axis can be named, so the two vocabularies have to be the same one
    or the surface contradicts itself again.
    """
    return _assertions.AXIS_LITERALS[axis_char]


def _bad_sugar_input(summary: str, *, reason: str, value, repairs):
    """A structured error for malformed author input to a sugar assertion.

    A bare `TypeError` / `NotImplementedError` would reach the agent as a
    `SourceRuntimeError` carrying the interpreter's words and no repair
    ("expected a DirectionRef, got tuple", with nothing about passing
    `Direction.world((0, 0, 1))`). `InvalidAssertion` because every caller is
    building an assertion, and no new class: a class earns its keep by selecting
    different advice, and the advice is on the error.
    """
    from ._errors import InvalidAssertion
    return InvalidAssertion(
        summary,
        refs=[],
        evidence={"reason": reason, "value": repr(value)},
        suggested_repairs=list(repairs),
    )


def _validate_axes_arg(axes, *, label: str = "axes"):
    if isinstance(axes, str):
        axes = tuple(axes)
    elif _ro.is_sequence(axes):
        axes = tuple(axes)
    else:
        raise _bad_sugar_input(
            f"{label} must be a str like 'xy' or a sequence of axis chars, "
            f"got {type(axes).__name__}",
            reason="axes_shape", value=axes,
            repairs=[f"pass {label}='xy' or {label}=('x', 'y') — which axes "
                     f"the containment applies along"])
    if not axes:
        raise _bad_sugar_input(
            f"{label} must be non-empty", reason="axes_empty", value=axes,
            repairs=[f"name at least one axis in {label}"])
    for c in axes:
        if c not in ("x", "y", "z"):
            raise _bad_sugar_input(
                f"{label}: invalid axis {c!r}", reason="axes_name", value=c,
                repairs=[f"{label} names unsigned axes only: 'x', 'y', 'z' — "
                         f"containment applies to both faces of an axis, so a "
                         f"sign has nothing to select here"])
    if len(set(axes)) != len(axes):
        raise _bad_sugar_input(
            f"{label}: duplicate axes in {axes!r}",
            reason="axes_duplicate", value=axes,
            repairs=[f"name each axis at most once in {label}"])
    return axes


# AABB anchor lookup: for axis 'x', the +x face is "left", -x is "right".
# For axis 'y', +y is "front", -y is "back". For 'z', +z "top", -z "bottom".
# To express containment on axis 'x', we compare the +x faces ("left") and
# -x faces ("right") of inner vs outer.
_FACE_ANCHOR_POS_NEG = {
    "x": ("left", "right"),   # +x, -x
    "y": ("front", "back"),    # +y, -y
    "z": ("top", "bottom"),    # +z, -z
}


def assert_within(builder, *, inner, outer, axes=("x", "y", "z"),
                  margin: float = 0.0, name: str):
    """Inner entity's AABB stays inside outer's, on each requested axis.

    Expands to 2*N `assert_distance` records on face anchors:
        for each axis c:
            inner.<+c_face> - outer.<+c_face> on +c <= margin   (suffix hi_c)
            inner.<-c_face> - outer.<-c_face> on +c >= -margin  (suffix lo_c)

    `inner` and `outer` must each be a PartRef (object layer) or
    InstanceRef/GroupRef/RoomRef (scene layer). They share the
    27-anchor vocabulary so face anchors compose uniformly.
    """
    axes = _validate_axes_arg(axes)
    margin_f = float(margin)
    out = []
    for c in axes:
        pos_face, neg_face = _FACE_ANCHOR_POS_NEG[c]
        axis_t = _axis_unit(c)
        # hi side: inner's +c face must be <= outer's +c face on +c.
        # Signed value `(outer.+c - inner.+c) . +c` >= -margin.
        out.append(builder.assert_distance(
            a=builder.anchor(inner, pos_face),
            b=builder.anchor(outer, pos_face),
            axis=axis_t,
            min=-margin_f, max=float("inf"),
            name=_suffix(name, "within", f"hi_{c}"),
        ))
        # lo side: inner's -c face must be >= outer's -c face on +c.
        # Signed value `(inner.-c - outer.-c) . +c` >= -margin.
        out.append(builder.assert_distance(
            a=builder.anchor(outer, neg_face),
            b=builder.anchor(inner, neg_face),
            axis=axis_t,
            min=-margin_f, max=float("inf"),
            name=_suffix(name, "within", f"lo_{c}"),
        ))
    return out


def assert_aabb_overlap(builder, *, a, b, axes, min_overlap: float, name: str):
    """Per-axis interval overlap of two entity AABBs >= min_overlap."""
    axes = _validate_axes_arg(axes)
    out = []
    for c in axes:
        out.append(builder.assert_overlap(
            a=a, b=b, axis=_axis_unit(c),
            min=float(min_overlap), max=float("inf"),
            name=_suffix(name, "aabb_overlap", c),
        ))
    return out


def assert_parallel(builder, *, a, b, tol_deg: float = 1.0, name: str):
    """Strict parallel: angle in [0, tol_deg]."""
    return [builder.assert_angle(
        a=a, b=b, min_deg=0.0, max_deg=float(tol_deg),
        name=_suffix(name, "parallel"),
    )]


def assert_antiparallel(builder, *, a, b, tol_deg: float = 1.0, name: str):
    """Strict antiparallel: angle in [180-tol_deg, 180]."""
    return [builder.assert_angle(
        a=a, b=b, min_deg=180.0 - float(tol_deg), max_deg=180.0,
        name=_suffix(name, "antiparallel"),
    )]


def assert_perpendicular(builder, *, a, b, tol_deg: float = 1.0, name: str):
    """Angle in [90-tol_deg, 90+tol_deg]."""
    return [builder.assert_angle(
        a=a, b=b,
        min_deg=90.0 - float(tol_deg), max_deg=90.0 + float(tol_deg),
        name=_suffix(name, "perpendicular"),
    )]


def assert_axis_world_aligned(builder, *, joint, world_axis,
                              tol_deg: float = 1.0, name: str):
    """`joint.axis()` is colinear with a world axis (unsigned).

    Accepts both signs — `axis=(0,0,+1)` and `axis=(0,0,-1)` both pass
    when `world_axis="z"`. A joint axis is a line, not a signed vector;
    callers who care about the sign should use `assert_parallel` /
    `assert_antiparallel` against a `Direction.world(...)` literal.

    `world_axis` takes an axis name, a 3-vector, or a `Direction.world(...)`
    literal. The ref form is accepted because it is the shape the rest of this
    SDK trains a caller to pass — `joint.axis()` and `anchor.outward()` both
    return a `DirectionRef`, and every other direction argument here takes one.
    Without that, a ref would fall through to `tuple(world_axis)` and raise a
    bare `TypeError` naming neither the argument nor the accepted forms.

    A joint-axis or anchor-outward ref is still refused, for the reason
    `_direction_world_tuple` gives: resolving one needs a pose-aware kinematic
    walk over the in-flight IR.
    """
    from ._refs import Direction, DirectionRef
    if isinstance(world_axis, DirectionRef):
        if world_axis._source != "world":
            raise _bad_sugar_input(
                f"`world_axis` needs a `Direction.world(...)` literal; got "
                f"source={world_axis._source!r}",
                reason="direction_source", value=world_axis._source,
                repairs=["name the axis — world_axis=\"z\" — or pass "
                         "Direction.world((x, y, z))",
                         "to compare against another joint's axis, call "
                         "assert_parallel / assert_antiparallel directly"])
        wd = world_axis
    elif isinstance(world_axis, str):
        try:
            wd = {"x": Direction.world_x(), "y": Direction.world_y(),
                  "z": Direction.world_z()}[world_axis]
        except KeyError:
            raise _bad_sugar_input(
                f"unknown world axis name {world_axis!r}",
                reason="world_axis_name", value=world_axis,
                repairs=['use "x", "y", or "z"',
                         "or pass Direction.world((x, y, z))"])
    else:
        wd = Direction.world(tuple(world_axis))
    return [builder._assert_axis_aligned(
        a=joint.axis(), b=wd, tol_deg=float(tol_deg),
        name=_suffix(name, "axis_aligned"),
    )]


def _direction_world_tuple(direction) -> tuple:
    """Extract a 3-tuple unit axis from a DirectionRef.

    v1 restriction: only `Direction.world(...)` literals are accepted by
    the sugars `assert_flush` and `assert_opens_toward`. Joint-axis and
    anchor-outward directions require pose-aware kinematic walks that
    depend on the in-flight IR; for those cases, the agent should
    expand the sugar manually (two base records).
    """
    from ._refs import DirectionRef
    if not isinstance(direction, DirectionRef):
        raise _bad_sugar_input(
            f"expected a DirectionRef, got {type(direction).__name__}",
            reason="not_a_direction", value=direction,
            repairs=["pass Direction.world((x, y, z)) — or name the axis, "
                     "Direction.world('z') — rather than a bare tuple"])
    if direction._source != "world":
        raise _bad_sugar_input(
            f"this sugar needs a `Direction.world(...)` literal for the axis; "
            f"got source={direction._source!r}",
            reason="direction_source", value=direction._source,
            repairs=["for a joint-axis or anchor-outward direction, expand the "
                     "sugar by hand: call assert_distance(...) and "
                     "assert_antiparallel(...) directly"])
    return tuple(direction._data["vec"])


def assert_flush(builder, *, face_a, face_b, axis_world,
                 tol_pos: float = 1.0e-3, tol_deg: float = 1.0,
                 opposing: bool = True, name: str):
    """Two face anchors lie on a common plane with anti-/parallel normals.

    Expands to:
      - `distance` (signed along `axis_world`), min=-tol_pos, max=tol_pos
        (suffix: `flush_pos`)
      - `angle` antiparallel (or parallel if opposing=False) between
        face_a.outward() and face_b.outward()
        (suffix: `flush_angle_parallel` or `flush_angle_antiparallel`)

    `axis_world` is the common world-frame normal direction of the two
    flush faces (e.g. `Direction.world_z()` for a top-bottom flush mount).
    Provided explicitly so the sugar can be evaluated without resolving
    `face_a.outward()` to a world tuple at SDK build time.
    """
    out = []
    out.append(builder.assert_distance(
        a=face_a, b=face_b,
        axis=_direction_world_tuple(axis_world),
        min=-float(tol_pos), max=float(tol_pos),
        name=_suffix(name, "flush", "pos"),
    ))
    if opposing:
        out.extend(assert_antiparallel(
            builder, a=face_a.outward(), b=face_b.outward(),
            tol_deg=tol_deg, name=_suffix(name, "flush", "angle"),
        ))
    else:
        out.extend(assert_parallel(
            builder, a=face_a.outward(), b=face_b.outward(),
            tol_deg=tol_deg, name=_suffix(name, "flush", "angle"),
        ))
    return out


def assert_opens_toward(builder, *, moving_part, direction,
                        open_pose, closed_pose=None,
                        min_displacement: float = 0.001,
                        probe_anchor: str = "center",
                        name: str):
    """Single-anchor displacement along a target direction between two poses.

    Expands to one `assert_distance` with pose-tagged anchor refs:
        a = moving_part.anchor(probe_anchor).at(closed_pose)
        b = moving_part.anchor(probe_anchor).at(open_pose)
        axis = direction  (must be a `Direction.world(...)` literal in v1)
        min = min_displacement
    """
    closed_pose = closed_pose or {}
    a = builder.anchor(moving_part, probe_anchor).at(closed_pose)
    b = builder.anchor(moving_part, probe_anchor).at(open_pose)
    return [builder.assert_distance(
        a=a, b=b, axis=_direction_world_tuple(direction),
        min=float(min_displacement), max=float("inf"),
        name=_suffix(name, "opens_toward"),
    )]


def assert_motion_sign(builder, *, joint, moving_part, toward,
                       min_displacement: float = 0.005,
                       name: str):
    """Single-joint terse form of `assert_opens_toward`.

    Probes `moving_part.center` between `joint.default` and the joint's
    upper limit. `toward` must be a `Direction.world(...)` literal.
    """
    default_q, _lower, upper_q = builder._joint_default_lower_upper(joint)
    return assert_opens_toward(
        builder,
        moving_part=moving_part,
        direction=toward,
        open_pose={joint: upper_q},
        closed_pose={joint: default_q},
        min_displacement=min_displacement,
        probe_anchor="center",
        name=name,
    )
