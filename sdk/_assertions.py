"""Layer-agnostic helpers for the relational-assertion API.

The object and scene SDKs share four agent-facing predicates —
`assert_distance`, `assert_collinear`, `assert_overlap`, `assert_angle` —
over a shared IR record shape. This module hosts the validators and
record builders so both layers consume one source of truth. There is
also a fifth IR record kind, `axis_aligned`, emitted only by the
`assert_axis_world_aligned` sugar via the object builder's private
`_assert_axis_aligned` method (the kind is unsigned-colinearity, used
for "joint axis lies along world Z" — agents never construct it directly).

`assert_distance` accepts an optional `axis`: when present, the record
holds the signed projection bound; when absent, the Euclidean bound.

Validators take the error class as a parameter (`error_cls`) so each
layer can keep its own structured exception type. The error class must
follow the `_Structured` mixin contract from `_errors.py`
(constructor: `summary, *, refs, evidence, suggested_repairs`).

The shared float-for-IR serializer handles the "inf" / "-inf" round-trip
that the canonical-JSON (`allow_nan=False`) constraint forces.
"""

from __future__ import annotations

import math
import re

from ._refs import (
    AnchorRef,
    DirectionRef,
    WORLD_DIRECTION_OWNER_ID,
    poses_to_dict,
    IDENTITY_ANCHOR_OFFSET,
)
from . import _readonly as _ro


__all__ = [
    "NAME_RE",
    "NAME_MAX_LEN",
    "AXIS_MIN_NORM",
    "check_assertion_name",
    "check_anchor_ref_owned",
    "check_entity_ref_owned",
    "check_direction_ref_owned",
    "check_axis",
    "check_finite_scalar",
    "check_finite_or_inf",
    "float_for_ir",
    "build_distance_record",
    "build_collinear_record",
    "build_overlap_record",
    "build_angle_record",
    "build_axis_aligned_record",
]


NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NAME_MAX_LEN = 64
AXIS_MIN_NORM = 1.0e-12


# ---------------------------------------------------------------------------
# Validators (raise error_cls)
# ---------------------------------------------------------------------------


def check_assertion_name(name, existing_names, error_cls, *, refs) -> None:
    """Validate an assertion name: regex + length + uniqueness in `existing_names`.

    `existing_names` may be any membership-testable container (set, dict,
    list) of already-registered assertion names.
    """
    if not isinstance(name, str):
        raise error_cls(
            f"assertion name must be a string, got {type(name).__name__}",
            refs=list(refs),
            evidence={"param": "assertion.name", "value": repr(name),
                      "constraint": "type:str"},
            suggested_repairs=["pass a string for the assertion name"],
        )
    if len(name) == 0 or len(name) > NAME_MAX_LEN:
        raise error_cls(
            f"assertion name length must be 1..{NAME_MAX_LEN}, got {len(name)}",
            refs=list(refs),
            evidence={"param": "assertion.name", "value": name,
                      "constraint": f"length in 1..{NAME_MAX_LEN}"},
            suggested_repairs=["shorten the name to 64 chars or fewer"],
        )
    if not NAME_RE.match(name):
        raise error_cls(
            f"assertion name {name!r} does not match {NAME_RE.pattern}",
            refs=list(refs),
            evidence={"param": "assertion.name", "value": name,
                      "constraint": f"regex {NAME_RE.pattern}"},
            suggested_repairs=["start name with a letter or underscore; "
                               "use only letters, digits, underscores"],
        )
    if name in existing_names:
        raise error_cls(
            f"duplicate assertion name {name!r}",
            refs=list(refs),
            evidence={"param": "assertion.name", "value": name,
                      "constraint": "unique in assertion namespace"},
            suggested_repairs=["choose a different assertion name"],
        )


def check_anchor_ref_owned(ref, owner_id, label, error_cls, *, refs) -> None:
    """Ensure `ref` is an AnchorRef produced by the builder with id=owner_id."""
    if not isinstance(ref, AnchorRef):
        raise error_cls(
            f"{label} must be an AnchorRef, got {type(ref).__name__}",
            refs=list(refs),
            evidence={"reason": "bad_anchor_ref_type", "param": label,
                      "value": type(ref).__name__},
            suggested_repairs=[
                f"pass the AnchorRef returned by this builder's "
                f".anchor(entity, name) call as {label}",
            ],
        )
    if ref._owner_id != owner_id:
        raise error_cls(
            f"{label} AnchorRef was not produced by this builder",
            refs=list(refs),
            evidence={"reason": "foreign_anchor_ref", "param": label,
                      "entity": ref._entity_name,
                      "anchor": ref._anchor_name},
            suggested_repairs=[
                f"use this builder's .anchor(...) for {label}",
            ],
        )


def check_entity_ref_owned(ref, owner_id, label, error_cls, *, refs,
                           allowed_kinds: tuple = ()) -> None:
    """Ensure `ref` is one of the EntityRef union types owned by this builder.

    `allowed_kinds` is an iterable of class names; e.g. for object-layer
    overlap, `("PartRef",)`. For scene-layer, `("InstanceRef", "GroupRef",
    "RoomRef")`. Empty tuple = accept any of the four.
    """
    accepted = allowed_kinds or ("PartRef", "InstanceRef", "GroupRef", "RoomRef")
    cls_name = type(ref).__name__
    if cls_name not in accepted:
        raise error_cls(
            f"{label} must be one of {accepted}, got {cls_name}",
            refs=list(refs),
            evidence={"reason": "bad_entity_ref_type", "param": label,
                      "value": cls_name, "accepted": list(accepted)},
            suggested_repairs=[
                f"pass a {' / '.join(accepted)} for {label}",
            ],
        )
    if getattr(ref, "_owner_id", None) != owner_id:
        raise error_cls(
            f"{label}: {cls_name} was not produced by this builder",
            refs=list(refs),
            evidence={"reason": "foreign_entity_ref", "param": label,
                      "entity": ref._name},
            suggested_repairs=[
                f"use this builder's API to produce the ref for {label}",
            ],
        )


def check_direction_ref_owned(ref, owner_id, label, error_cls, *, refs) -> None:
    """Ensure `ref` is a DirectionRef. Owner-id must match the builder
    OR the ref must be a literal world direction (owner_id sentinel).
    """
    if not isinstance(ref, DirectionRef):
        raise error_cls(
            f"{label} must be a DirectionRef "
            f"(from joint.axis() / anchor.outward() / Direction.world(...)), "
            f"got {type(ref).__name__}",
            refs=list(refs),
            evidence={"reason": "bad_direction_ref_type", "param": label,
                      "value": type(ref).__name__},
            suggested_repairs=[
                "produce the direction via joint.axis(), "
                "anchor.outward(), or Direction.world((x, y, z))",
            ],
        )
    if ref._owner_id == WORLD_DIRECTION_OWNER_ID:
        return
    if ref._owner_id != owner_id:
        raise error_cls(
            f"{label}: DirectionRef was not produced by this builder",
            refs=list(refs),
            evidence={"reason": "foreign_direction_ref", "param": label,
                      "source": ref._source},
            suggested_repairs=[
                "use this builder's joint.axis() / anchor.outward(), "
                "or Direction.world(...) for literal world directions",
            ],
        )


def check_finite_scalar(x, label: str, error_cls, *, refs) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "type:number"},
        )
    if not math.isfinite(float(x)):
        raise error_cls(
            f"{label} must be finite, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x, "constraint": "finite"},
        )


def check_finite_or_inf(x, label: str, error_cls, *, refs,
                        allow_neg_inf: bool = True,
                        allow_pos_inf: bool = True) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "type:number"},
        )
    f = float(x)
    if math.isnan(f):
        raise error_cls(
            f"{label} must not be NaN", refs=list(refs),
            evidence={"param": label, "value": x, "constraint": "non-nan"},
        )
    if math.isinf(f):
        if f > 0 and not allow_pos_inf:
            raise error_cls(
                f"{label} must not be +inf", refs=list(refs),
                evidence={"param": label, "value": "inf"},
            )
        if f < 0 and not allow_neg_inf:
            raise error_cls(
                f"{label} must not be -inf", refs=list(refs),
                evidence={"param": label, "value": "-inf"},
            )


#: Every spelling of a world axis the SDK accepts, unsigned and signed.
#:
#: The vocabulary exists so the SDK agrees with itself: the sugar assertions
#: take `axes=("x", "y", "z")` — axis *characters* — so `axis=` accepts the same
#: spellings, and an author who generalizes from the surface they were shown
#: and writes `axis="y"` is not refused.
AXIS_LITERALS = {
    "x": (1.0, 0.0, 0.0), "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def coerce_axis(axis, error_cls, *, refs) -> tuple:
    """Extract components from a tuple/list, signed name, or world literal.

    No frame conversion occurs: joint callers interpret the tuple in the
    parent anchor's frame; assertion callers use world coordinates. Only
    `DirectionRef(source="world")` has a literal vector. Joint/anchor refs
    need a finalized IR and pose to resolve.
    """
    if isinstance(axis, DirectionRef):
        try:
            return axis.world_vec()
        except Exception as exc:  # noqa: BLE001 — re-raised as this caller's class
            raise error_cls(
                f"axis: this DirectionRef has source={axis.source!r}, which "
                f"resolves only against a finalized IR",
                refs=list(refs),
                evidence={"reason": "axis_ref_source", "value": repr(axis),
                          "source": axis.source},
                suggested_repairs=[
                    "pass a literal direction — Direction.world((x, y, z)), "
                    "'z', or (0, 0, 1) — for an axis fixed at author time",
                    "resolve joint/anchor directions with `joint_axis_world` "
                    "/ `outward` probes on buildable source; convert the world "
                    "vector to the parent anchor frame before using it as a "
                    "joint axis",
                    str(exc),
                ],
            ) from exc
    if isinstance(axis, str):
        unit = AXIS_LITERALS.get(axis.strip().lower())
        if unit is None:
            raise error_cls(
                f"unknown axis name {axis!r}",
                refs=list(refs),
                evidence={"reason": "axis_name", "value": axis,
                          "accepted": sorted(AXIS_LITERALS)},
                suggested_repairs=[
                    "use 'x', 'y', 'z', a signed form like '-z', or a vector "
                    "(x, y, z), in the frame required by this API",
                ],
            )
        return unit
    if _ro.is_sequence(axis) and len(axis) == 3:
        return tuple(axis)
    raise error_cls(
        f"axis must be (x, y, z), an axis name, or a Direction.world(...) "
        f"literal, got {axis!r}",
        refs=list(refs),
        evidence={"reason": "axis_shape", "value": repr(axis)},
        suggested_repairs=[
            "use (x, y, z), an axis name such as '-z', or a literal "
            "Direction.world((x, y, z)); components use this API's axis "
            "frame (parent anchor for joints, world for assertions)",
        ],
    )


def check_axis(axis, error_cls, *, refs) -> tuple:
    """Validate and return a unit-magnitude axis tuple.

    Accepts the spellings in `coerce_axis`. Rejects non-finite components
    and near-zero norms. Returns (ux, uy, uz) normalized.
    """
    axis = coerce_axis(axis, error_cls, refs=refs)
    for i, c in enumerate(axis):
        check_finite_scalar(c, f"axis[{i}]", error_cls, refs=refs)
    norm = math.sqrt(sum(float(c) * float(c) for c in axis))
    if norm < AXIS_MIN_NORM:
        raise error_cls(
            f"axis norm too small ({norm}); must be > {AXIS_MIN_NORM}",
            refs=list(refs),
            evidence={"reason": "axis_zero", "value": list(axis),
                      "norm": norm},
            suggested_repairs=["pass a non-zero axis vector"],
        )
    return (float(axis[0]) / norm,
            float(axis[1]) / norm,
            float(axis[2]) / norm)


# ---------------------------------------------------------------------------
# IR serialization helper
# ---------------------------------------------------------------------------


def float_for_ir(x: float):
    """Serialize a possibly-infinite float for the IR.

    canonical JSON (`allow_nan=False`) rejects bare inf/-inf tokens, so
    these are mapped to the JSON strings `"inf"` / `"-inf"`. The verifier
    kernel's `_inf_or_float` helper reverses the mapping for arithmetic.
    Finite values pass through.
    """
    f = float(x)
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _ir_poses_dict(poses_frozen, joint_types):
    """Convert a frozen poses tuple from surface units to IR-internal units.

    Returns a plain `{name: value}` dict suitable for IR serialization.
    The SDK surface accepts pose values in the same units as joint
    `limits` / `default` — degrees for revolute, meters for prismatic —
    but the IR (matching `limits` and `default`) stores radians for
    revolute. Conversion is per-key using the joint-type map. Keys
    absent from `joint_types` are passed through unchanged (consistent
    with the universal "unrelated keys silently ignored" rule).
    """
    from . import _rotation as _rot
    surface = poses_to_dict(poses_frozen)
    if joint_types is None:
        return surface
    return _rot.convert_poses_to_internal(surface, joint_types)


# AnchorRef -> compact dict used in the IR record.
def _anchor_dict(ref: AnchorRef, *, assertion_poses=(), joint_types=None) -> dict:
    """Build an anchor IR dict, with assertion-level pose fallback.

    Per-ref poses always win. Assertion-level `assertion_poses` fills
    in only when the ref has no per-ref poses of its own.

    A non-identity offset is serialized as a nested
    `{"translation": [...], "rotation": [...]}` field with rotation in
    intrinsic Euler XYZ degrees. Identity offsets are omitted for
    minimal IR.

    `joint_types` is the owning builder's `{joint_name: joint_type}` map.
    Pose values are surface-units on the ref and are converted to
    IR-internal units (radians for revolute) before serialization.
    """
    d = {"entity": ref._entity_name, "anchor": ref._anchor_name}
    if ref._offset != IDENTITY_ANCHOR_OFFSET:
        (tx, ty, tz), (rx, ry, rz) = ref._offset
        d["offset"] = {
            "translation": [float(tx), float(ty), float(tz)],
            "rotation_euler_deg": [float(rx), float(ry), float(rz)],
        }
    poses = ref._poses or assertion_poses
    if poses:
        d["poses"] = _ir_poses_dict(poses, joint_types)
    return d


def _entity_dict(ref, *, assertion_poses=(), joint_types=None) -> dict:
    """Build an entity dict from PartRef/InstanceRef/GroupRef/RoomRef."""
    cls_name = type(ref).__name__
    kind_map = {
        "PartRef": "part",
        "InstanceRef": "instance",
        "GroupRef": "group",
        "RoomRef": "room",
    }
    kind = kind_map.get(cls_name)
    if kind is None:
        raise TypeError(
            f"expected an EntityRef (PartRef/InstanceRef/GroupRef/RoomRef), "
            f"got {cls_name}"
        )
    d = {"kind": kind, "name": ref._name}
    poses = getattr(ref, "_poses", ()) or assertion_poses
    if poses:
        d["poses"] = _ir_poses_dict(poses, joint_types)
    return d


def _direction_dict(ref: DirectionRef, *, assertion_poses=(), joint_types=None) -> dict:
    """Build a direction dict for the IR.

    Output shape depends on `source`:
        joint_axis     -> {"source", "joint", "poses"?}
        anchor_outward -> {"source", "entity_kind", "entity", "anchor",
                           "anchor_offset"?, "poses"?}
        world          -> {"source", "vec"}
    """
    src = ref._source
    if src == "joint_axis":
        d = {"source": "joint_axis", "joint": ref._data["joint"]}
    elif src == "anchor_outward":
        d = {
            "source": "anchor_outward",
            "entity_kind": ref._data["entity_kind"],
            "entity": ref._data["entity"],
            "anchor": ref._data["anchor"],
        }
        offset = ref._data.get("anchor_offset")
        if offset is not None and offset != IDENTITY_ANCHOR_OFFSET:
            (tx, ty, tz), (rx, ry, rz) = offset
            d["anchor_offset"] = {
                "translation": [float(tx), float(ty), float(tz)],
                "rotation_euler_deg": [float(rx), float(ry), float(rz)],
            }
    elif src == "world":
        vec = ref._data["vec"]
        d = {"source": "world", "vec": [vec[0], vec[1], vec[2]]}
    else:
        raise ValueError(f"unknown DirectionRef source {src!r}")
    if src != "world":
        poses = ref._poses or assertion_poses
        if poses:
            d["poses"] = _ir_poses_dict(poses, joint_types)
    return d


def build_distance_record(
    *, assertion_id: int, name: str,
    a: AnchorRef, b: AnchorRef,
    axis_unit: tuple | None, lo, hi,
    assertion_poses=(),
    joint_types=None,
) -> dict:
    """Build a distance assertion record.

    If `axis_unit` is None the record encodes a Euclidean distance
    bound; otherwise it encodes the signed projection along the
    pre-normalized unit axis.

    `assertion_poses` is an optional frozen poses tuple from the
    assertion-level `poses=` shorthand. It's copied into each ref's
    IR dict only when that ref has no per-ref poses of its own.
    `joint_types` is the owning builder's joint-type map; pose values
    are converted from surface units (degrees for revolute) to
    IR-internal units (radians for revolute) using this map at
    serialization time.
    """
    args = {
        "a": _anchor_dict(a, assertion_poses=assertion_poses, joint_types=joint_types),
        "b": _anchor_dict(b, assertion_poses=assertion_poses, joint_types=joint_types),
        "min": float_for_ir(lo),
        "max": float_for_ir(hi),
    }
    if axis_unit is not None:
        args["axis"] = [axis_unit[0], axis_unit[1], axis_unit[2]]
    return {
        "id": assertion_id,
        "name": name,
        "kind": "distance",
        "args": args,
    }


def build_collinear_record(
    *, assertion_id: int, name: str,
    anchors: list, axis_unit: tuple, tol: float,
    assertion_poses=(),
    joint_types=None,
) -> dict:
    return {
        "id": assertion_id,
        "name": name,
        "kind": "collinear",
        "args": {
            "anchors": [_anchor_dict(ar, assertion_poses=assertion_poses,
                                     joint_types=joint_types)
                        for ar in anchors],
            "axis": [axis_unit[0], axis_unit[1], axis_unit[2]],
            "tol": float(tol),
        },
    }


def build_overlap_record(
    *, assertion_id: int, name: str,
    a, b, axis_unit: tuple, lo, hi,
    assertion_poses=(),
    joint_types=None,
) -> dict:
    return {
        "id": assertion_id,
        "name": name,
        "kind": "overlap",
        "args": {
            "a": _entity_dict(a, assertion_poses=assertion_poses, joint_types=joint_types),
            "b": _entity_dict(b, assertion_poses=assertion_poses, joint_types=joint_types),
            "axis": [axis_unit[0], axis_unit[1], axis_unit[2]],
            "min": float_for_ir(lo),
            "max": float_for_ir(hi),
        },
    }


def build_angle_record(
    *, assertion_id: int, name: str,
    dir_a: DirectionRef, dir_b: DirectionRef,
    min_deg: float, max_deg: float,
    assertion_poses=(),
    joint_types=None,
) -> dict:
    return {
        "id": assertion_id,
        "name": name,
        "kind": "angle",
        "args": {
            "a": _direction_dict(dir_a, assertion_poses=assertion_poses,
                                 joint_types=joint_types),
            "b": _direction_dict(dir_b, assertion_poses=assertion_poses,
                                 joint_types=joint_types),
            "min_deg": float(min_deg),
            "max_deg": float(max_deg),
        },
    }


def build_axis_aligned_record(
    *, assertion_id: int, name: str,
    dir_a: DirectionRef, dir_b: DirectionRef,
    tol_deg: float,
    assertion_poses=(),
    joint_types=None,
) -> dict:
    """Direction `a` is colinear with direction `b` (unsigned).

    Evaluates `min(angle, 180-angle) <= tol_deg`, so an axis pointing
    along +Z is considered aligned with both +Z and -Z. This is the
    right check for "joint axis lies along world Z" — the axis is a
    line, not a signed vector.
    """
    return {
        "id": assertion_id,
        "name": name,
        "kind": "axis_aligned",
        "args": {
            "a": _direction_dict(dir_a, assertion_poses=assertion_poses,
                                 joint_types=joint_types),
            "b": _direction_dict(dir_b, assertion_poses=assertion_poses,
                                 joint_types=joint_types),
            "tol_deg": float(tol_deg),
        },
    }
