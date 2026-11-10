"""Layer-agnostic kernel for the RelationalAssertions verifier.

The object layer (`ObjectRelationalAssertions`) and the scene layer (`SceneRelationalAssertions`) evaluate the same
predicates from `ir["assertions"]`:

    distance      — between two anchors: the signed projection (b - a) . axis_unit if args["axis"] is given, else
                    the Euclidean norm |b - a|; must lie in [lo, hi].
    collinear     — every anchor's perpendicular offset from the axis has norm <= tol.
    overlap       — signed interval overlap of two entity AABBs along an axis,
                    min(a.max, b.max) - max(a.min, b.min) (positive = depth, negative = gap); must lie in [lo, hi].
    angle         — signed angle in degrees between two unit directions ("joint_axis", "anchor_outward",
                    "world"); must lie in [min_deg, max_deg].
    axis_aligned  — *unsigned* colinearity, min(angle, 180 - angle) <= tol_deg, for
                    `assert_axis_world_aligned`: a joint axis is a line, so one along ±world-Z passes.

Anchor, entity, and direction dicts may carry `"poses": {joint_name: q}`; resolvers apply those articulations along
the ancestor chain and ignore unrelated keys. Each layer plugs in its own resolvers (anchor, entity, direction) for
its kinematic-tree walk.
"""

from __future__ import annotations

import math
import types


__all__ = ["evaluate_assertions", "_inf_or_float", "_bound_to_json"]


# Floating-point noise floor for bound comparisons. FK composition over
# rotations + nested anchor offsets accumulates ULP-scale error, so a
# distance / overlap / angle authored to land flush on a bound can come
# back ~1e-15 outside it. Mirrors `_CONTAINMENT_EPS_M` in scene_verify.
# Applied to meters (distance, overlap) and degrees (angle) alike; real
# authoring mistakes are orders of magnitude larger than 1 µm or 1 µ°.
_BOUND_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Inf round-trip helpers
# ---------------------------------------------------------------------------


def _inf_or_float(x):
    """Inverse of `_assertions.float_for_ir`.

    Maps the JSON strings `"inf"` / `"-inf"` back to `math.inf` /
    `-math.inf`. Finite values pass through as floats.
    """
    if isinstance(x, str):
        if x == "inf":
            return math.inf
        if x == "-inf":
            return -math.inf
    return float(x)


def _bound_to_json(x):
    """Serialize a float bound for the report. Inf round-trip-safe."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


# ---------------------------------------------------------------------------
# Vector primitives
# ---------------------------------------------------------------------------


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vscale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vnorm(v):
    return math.sqrt(_vdot(v, v))


# ---------------------------------------------------------------------------
# Per-kind evaluation
# ---------------------------------------------------------------------------


def _surface_offset_from_ir(offset_dict):
    """Convert an IR anchor-offset dict (translation + Euler degrees)
    to the surface 6-tuple form `((tx,ty,tz),(rx,ry,rz))`, or None."""
    if offset_dict is None:
        return None
    t = offset_dict.get("translation") or [0.0, 0.0, 0.0]
    r = offset_dict.get("rotation_euler_deg") or [0.0, 0.0, 0.0]
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(r[0]), float(r[1]), float(r[2])))


def _anchor_world(resolve_anchor_fn, anchor_dict):
    """Resolve an anchor dict to a world-frame position.

    The anchor dict may carry optional 'poses' and 'offset' fields.
    Passes 'poses' and the surface-form 'offset' through to the layer
    resolver so kinematic walks apply pose overrides on the ancestor
    chain and the anchor's translation offset is applied in
    entity-local. Rotation in the offset is irrelevant for point
    queries; only the translation contributes.
    """
    return resolve_anchor_fn(
        anchor_dict["entity"],
        anchor_dict["anchor"],
        anchor_dict.get("poses") or None,
        _surface_offset_from_ir(anchor_dict.get("offset")),
    )


def _eval_distance(resolve_anchor_fn, args):
    """Distance between two anchors.

    If args["axis"] is present, evaluates the signed projection
    (b - a) . axis_unit on that axis. Otherwise, the Euclidean
    distance |b - a|. The result must lie in [lo, hi].
    """
    a_pos = _anchor_world(resolve_anchor_fn, args["a"])
    b_pos = _anchor_world(resolve_anchor_fn, args["b"])
    delta = _vsub(b_pos, a_pos)
    axis = args.get("axis")
    if axis is not None:
        axis_t = tuple(float(c) for c in axis)  # already normalized by SDK
        value = _vdot(delta, axis_t)
        axis_field = list(axis_t)
    else:
        value = _vnorm(delta)
        axis_field = None
    lo = _inf_or_float(args["min"])
    hi = _inf_or_float(args["max"])
    passed = lo - _BOUND_EPS <= value <= hi + _BOUND_EPS
    detail = {
        "kind": "distance",
        "value": value,
        "bounds": {"min": _bound_to_json(lo), "max": _bound_to_json(hi)},
        "anchors": [
            {"entity": args["a"]["entity"], "anchor": args["a"]["anchor"],
             "world_pos": list(a_pos)},
            {"entity": args["b"]["entity"], "anchor": args["b"]["anchor"],
             "world_pos": list(b_pos)},
        ],
    }
    if axis_field is not None:
        detail["axis"] = axis_field
    return passed, detail


def _eval_collinear(resolve_anchor_fn, args):
    axis = tuple(float(c) for c in args["axis"])
    tol = float(args["tol"])
    anchor_dicts = args["anchors"]
    positions = [_anchor_world(resolve_anchor_fn, a) for a in anchor_dicts]
    base = positions[0]
    perp_norms = []
    for p in positions[1:]:
        delta = _vsub(p, base)
        along = _vscale(axis, _vdot(delta, axis))
        perp = _vsub(delta, along)
        perp_norms.append(_vnorm(perp))
    max_perp = max(perp_norms) if perp_norms else 0.0
    passed = max_perp <= tol
    return passed, {
        "kind": "collinear",
        "value": max_perp,
        "bounds": {"tol": tol},
        "axis": list(axis),
        "anchors": [
            {"entity": a["entity"], "anchor": a["anchor"],
             "world_pos": list(p)}
            for a, p in zip(anchor_dicts, positions)
        ],
        "perp_norms": perp_norms,
    }


def _eval_overlap(resolvers, args):
    """Signed interval overlap of two entity AABBs along a unit axis.

        value = min(a.max, b.max) - max(a.min, b.min)

    Positive = overlap depth; negative = gap distance. Result must lie
    in [lo, hi].

    Implementation: project each AABB's 8 corners onto the unit axis and
    take per-AABB min/max. Works for arbitrary axes (not only axis-
    aligned ones in the entity's local frame).
    """
    axis = tuple(float(c) for c in args["axis"])
    aabb_a = resolvers.entity_aabb(args["a"])
    aabb_b = resolvers.entity_aabb(args["b"])
    a_min, a_max = _aabb_projection_extents(aabb_a, axis)
    b_min, b_max = _aabb_projection_extents(aabb_b, axis)
    value = min(a_max, b_max) - max(a_min, b_min)
    lo = _inf_or_float(args["min"])
    hi = _inf_or_float(args["max"])
    passed = lo - _BOUND_EPS <= value <= hi + _BOUND_EPS
    return passed, {
        "kind": "overlap",
        "value": value,
        "bounds": {"min": _bound_to_json(lo), "max": _bound_to_json(hi)},
        "axis": list(axis),
        "entities": [
            {**args["a"], "projection": [a_min, a_max]},
            {**args["b"], "projection": [b_min, b_max]},
        ],
    }


def _eval_angle(resolvers, args):
    """Angle (degrees) between two unit directions; must lie in [min_deg, max_deg]."""
    ua = resolvers.direction(args["a"])
    ub = resolvers.direction(args["b"])
    dot = max(-1.0, min(1.0, _vdot(ua, ub)))
    deg = math.degrees(math.acos(dot))
    lo = float(args["min_deg"])
    hi = float(args["max_deg"])
    passed = lo - _BOUND_EPS <= deg <= hi + _BOUND_EPS
    return passed, {
        "kind": "angle",
        "value": deg,
        "bounds": {"min_deg": lo, "max_deg": hi},
        "directions": [
            {**args["a"], "unit": list(ua)},
            {**args["b"], "unit": list(ub)},
        ],
    }


def _eval_axis_aligned(resolvers, args):
    """Direction `a` is colinear with direction `b` (unsigned: +/- both pass).

    Evaluates `min(angle, 180 - angle) <= tol_deg`. Use for "joint axis
    lies along world Z" checks — an axis vector of -Z is colinear with
    +Z and should pass.
    """
    ua = resolvers.direction(args["a"])
    ub = resolvers.direction(args["b"])
    dot = max(-1.0, min(1.0, _vdot(ua, ub)))
    deg = math.degrees(math.acos(dot))
    value = min(deg, 180.0 - deg)
    tol = float(args["tol_deg"])
    passed = value <= tol + _BOUND_EPS
    return passed, {
        "kind": "axis_aligned",
        "value": value,
        "raw_deg": deg,
        "bounds": {"tol_deg": tol},
        "directions": [
            {**args["a"], "unit": list(ua)},
            {**args["b"], "unit": list(ub)},
        ],
    }


# ---------------------------------------------------------------------------
# Public scalar evaluators (Bucket 2+3) — reusable by both verifier kernel and
# the probe library (sdk/_probes.py). Each takes resolved inputs (positions,
# AABBs, unit directions) and returns the scalar the verifier compares to a
# bound. No pass/fail logic; pure math.
# ---------------------------------------------------------------------------


def eval_distance_scalar(p_a, p_b, axis=None) -> float:
    """Distance between two points.

      * axis=None  → Euclidean `||p_b - p_a||` (always ≥ 0).
      * axis=(...) → signed projection `(p_b - p_a) · unit(axis)` —
                     antisymmetric in `p_a` / `p_b`.

    The axis tuple is normalized internally; callers may pass any non-zero
    vector. Raises `ValueError` on a zero-norm axis input.
    """
    delta = _vsub(p_b, p_a)
    if axis is None:
        return _vnorm(delta)
    axis_t = tuple(float(c) for c in axis)
    n = _vnorm(axis_t)
    if n < 1e-12:
        raise ValueError("eval_distance_scalar: axis must be a non-zero vector")
    unit = (axis_t[0] / n, axis_t[1] / n, axis_t[2] / n)
    return _vdot(delta, unit)


def eval_collinear_scalar(points, axis) -> float:
    """Max perpendicular deviation of `points` from the line through
    `points[0]` along `unit(axis)`. Returns 0 for fewer than 2 points.
    Raises `ValueError` on a zero-norm axis input.
    """
    if len(points) < 2:
        return 0.0
    axis_t = tuple(float(c) for c in axis)
    n = _vnorm(axis_t)
    if n < 1e-12:
        raise ValueError("eval_collinear_scalar: axis must be a non-zero vector")
    unit = (axis_t[0] / n, axis_t[1] / n, axis_t[2] / n)
    base = points[0]
    perps = []
    for p in points[1:]:
        delta = _vsub(p, base)
        along = _vscale(unit, _vdot(delta, unit))
        perp = _vsub(delta, along)
        perps.append(_vnorm(perp))
    return max(perps) if perps else 0.0


def eval_overlap_scalar(aabb_a, aabb_b, axis) -> float:
    """Signed projection overlap on a unit axis.

      `value = min(a_hi, b_hi) - max(a_lo, b_lo)` along `unit(axis)`,
      where (a_lo, a_hi) etc. are projection extents of each AABB onto
      that axis. Positive = overlap depth, negative = gap distance.
    Raises `ValueError` on a zero-norm axis input.
    """
    axis_t = tuple(float(c) for c in axis)
    n = _vnorm(axis_t)
    if n < 1e-12:
        raise ValueError("eval_overlap_scalar: axis must be a non-zero vector")
    unit = (axis_t[0] / n, axis_t[1] / n, axis_t[2] / n)
    a_min, a_max = _aabb_projection_extents(aabb_a, unit)
    b_min, b_max = _aabb_projection_extents(aabb_b, unit)
    return min(a_max, b_max) - max(a_min, b_min)


def eval_angle_scalar(dir_a, dir_b) -> float:
    """Unsigned angle (degrees, in [0, 180]) between two direction vectors.

    Inputs may have any non-zero magnitude; normalized internally.
    Raises `ValueError` on a zero-norm input.
    """
    na = _vnorm(dir_a)
    nb = _vnorm(dir_b)
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("eval_angle_scalar: directions must be non-zero")
    ua = (dir_a[0] / na, dir_a[1] / na, dir_a[2] / na)
    ub = (dir_b[0] / nb, dir_b[1] / nb, dir_b[2] / nb)
    dot = max(-1.0, min(1.0, _vdot(ua, ub)))
    return math.degrees(math.acos(dot))


def _aabb_projection_extents(aabb, axis):
    """Project an AABB's 8 corners onto a unit axis; return (min, max)."""
    (xmn, ymn, zmn), (xmx, ymx, zmx) = aabb
    extents = []
    for x in (xmn, xmx):
        for y in (ymn, ymx):
            for z in (zmn, zmx):
                extents.append(x * axis[0] + y * axis[1] + z * axis[2])
    return min(extents), max(extents)


# kind -> (evaluator, takes the resolver NAMESPACE rather than the bare
# `resolve_anchor_fn`). The convention lives here rather than in a parallel
# tuple at the call site: a new kind then cannot be registered without stating
# how it is called. Split across two places, the second edit was easy to miss,
# and missing it handed the evaluator the wrong first argument — a TypeError at
# verify time rather than an error at registration.
_EVAL_BY_KIND = {
    "distance":     (_eval_distance,     False),
    "collinear":    (_eval_collinear,    False),
    "overlap":      (_eval_overlap,      True),
    "angle":        (_eval_angle,        True),
    "axis_aligned": (_eval_axis_aligned, True),
}


# ---------------------------------------------------------------------------
# Hint generation
# ---------------------------------------------------------------------------


def _hint_for(detail: dict, name: str) -> str:
    kind = detail["kind"]
    if kind == "distance":
        if "axis" in detail:
            return (
                f"Assertion {name!r} failed: signed-axis distance "
                f"{detail['value']:.4g} outside bounds {detail['bounds']}. "
                f"Adjust placement or anchor translations in their local "
                f"frames so the signed projection lies in range."
            )
        return (
            f"Assertion {name!r} failed: Euclidean distance "
            f"{detail['value']:.4g} outside bounds {detail['bounds']}. "
            f"Adjust placement or anchor translations in their local "
            f"frames to reach the intended separation."
        )
    if kind == "collinear":
        return (
            f"Assertion {name!r} failed: maximum perpendicular deviation "
            f"{detail['value']:.4g} > tol {detail['bounds']['tol']:.4g}. "
            f"The anchors are not collinear along the supplied axis."
        )
    if kind == "overlap":
        return (
            f"Assertion {name!r} failed: signed interval overlap "
            f"{detail['value']:.4g} outside bounds {detail['bounds']}. "
            f"Positive = overlap depth; negative = gap. Adjust placement "
            f"or pose to bring the overlap into range."
        )
    if kind == "angle":
        return (
            f"Assertion {name!r} failed: angle {detail['value']:.4g}° "
            f"outside bounds {detail['bounds']}. Check joint axis sign, "
            f"anchor face orientation, or the literal world axis."
        )
    if kind == "axis_aligned":
        return (
            f"Assertion {name!r} failed: colinearity residual "
            f"{detail['value']:.4g}° (raw angle {detail.get('raw_deg', 0):.4g}°) "
            f"exceeds tol_deg {detail['bounds']['tol_deg']:.4g}. "
            f"The joint axis is not along the requested world axis — check "
            f"the anchor frame orientation."
        )
    return (
        f"Assertion {name!r} failed with kind {kind!r}: details {detail}."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_assertions(assertions, resolve_anchor_fn,
                        resolve_entity_aabb_fn=None,
                        resolve_direction_fn=None):
    """Evaluate a list of assertion records against per-call resolvers.

    Args:
        assertions: list of records produced by the assertion-builder
            helpers (see sdk/_assertions.py:build_*_record). Each record
            is `{"id", "name", "kind", "args"}`.
        resolve_anchor_fn: `(entity_name, anchor_name, poses_or_none) -> (x,y,z)`.
            Used by `distance` and `collinear` kinds. Required.
        resolve_entity_aabb_fn: `(entity_dict) -> ((lo3), (hi3))`.
            Used by `overlap`. Required when the assertion list contains
            any overlap records.
        resolve_direction_fn: `(direction_dict) -> (ux, uy, uz)`.
            Used by `angle` and `axis_aligned`. Required when the
            assertion list contains any direction-based records.

    Returns:
        (results, repairs, any_fail).
    """
    # Bundle resolvers into a namespace for kind-specific evaluators.
    resolvers = types.SimpleNamespace(
        anchor=resolve_anchor_fn,
        entity_aabb=resolve_entity_aabb_fn,
        direction=resolve_direction_fn,
    )
    results: list[dict] = []
    repairs: list[str] = []
    any_fail = False
    for a in assertions:
        name = a["name"]
        kind = a["kind"]
        entry = _EVAL_BY_KIND.get(kind)
        if entry is None:
            results.append({
                "name": name, "kind": kind, "verdict": "fail",
                "error": "unknown_kind",
            })
            repairs.append(
                f"Assertion {name!r} has unknown kind {kind!r}; "
                "this should not happen — re-check the IR."
            )
            any_fail = True
            continue
        ev_fn, wants_resolvers = entry
        passed, detail = ev_fn(
            resolvers if wants_resolvers else resolve_anchor_fn, a["args"])
        results.append({
            "name": name,
            "verdict": "pass" if passed else "fail",
            **detail,
        })
        if not passed:
            any_fail = True
            repairs.append(_hint_for(detail, name))
    return results, repairs, any_fail
