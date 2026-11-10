"""Freeform probe surface for the agent's `probe` op.

Mirrors the assertion API (every base assertion has a scalar-returning
probe) and the verifier suite (selected verifiers have their diagnostic
data reachable as a probe). Symbols are organized by layer
(`object_*` / `scene_*`); the agent-facing probe overlay in
`agent/layers/layer.py` aliases them to bare names per layer.

Probes are stateless: each call recomputes from the IR. Probes do NOT
emit pass/fail verdicts — they return scalars, dicts, or lists that the
agent inspects directly. For pass/fail semantics, author an assertion
(`Object.assert_*` / `Scene.assert_*`) which the verifier evaluates.

Pose tagging follows the verifier's universal-pose-tagging convention:
`poses={joint_name: q, ...}`. Keys not on the queried entity's ancestor
chain are silently ignored. Scene-layer pose keys are scoped to the
owning instance; cross-instance form is `"<instance>.<joint>"`.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import _object_kernel as _ok
from . import _scene_kernel as _sk
from . import _rotation as _rot
from ..verify import _ccd_sampler as _ccd
from ..verify._relational import (
    eval_distance_scalar,
    eval_overlap_scalar,
    eval_angle_scalar,
)
from . import _readonly as _ro


# ──────────────────────────────────────────────────────────────────
# Pose-unit conversion at the probe-entry boundary.
#
# Every LLM-facing `poses=` argument carries surface units (degrees for
# revolute, meters for prismatic), matching joint `limits` / `default`.
# Kernel functions and IR storage operate in IR-internal units (radians
# for revolute). The helpers below normalize the incoming surface poses
# to internal poses at the boundary; internal callers (CCD sampler,
# verifier-internal pose flows) stay in internal units throughout.
# ──────────────────────────────────────────────────────────────────


def _object_poses_to_internal(ir, poses):
    """Convert an object-layer surface-poses dict to IR-internal units."""
    if poses is None:
        return None
    joint_types = _rot.object_joint_types_from_ir(ir)
    return _rot.convert_poses_to_internal(poses, joint_types)


def _scene_poses_to_internal(ir, catalog, poses):
    """Convert a scene-layer surface-poses dict to IR-internal units.

    Handles both bare `"<joint>"` and qualified `"<instance>.<joint>"`
    pose keys; per-instance lookup uses the catalog's object IRs.
    """
    if poses is None:
        return None
    return _rot.convert_scene_poses_to_internal(poses, ir, catalog)


# ──────────────────────────────────────────────────────────────────
# Internal helpers — signed_overlap projection and SAT-derived data
# ──────────────────────────────────────────────────────────────────


def _signed_overlap_axis(aabb_a, aabb_b, axis_idx: int) -> float:
    """Signed projection overlap on world axis `axis_idx` (0=x, 1=y, 2=z).

    Returns `min(a_hi, b_hi) - max(a_lo, b_lo)` along that axis:
      positive = overlap depth, negative = gap distance, zero = touching.
    """
    a_lo = aabb_a[0][axis_idx]
    a_hi = aabb_a[1][axis_idx]
    b_lo = aabb_b[0][axis_idx]
    b_hi = aabb_b[1][axis_idx]
    return min(a_hi, b_hi) - max(a_lo, b_lo)


def _signed_overlap_dict(aabb_a, aabb_b) -> dict:
    return {
        "x": _signed_overlap_axis(aabb_a, aabb_b, 0),
        "y": _signed_overlap_axis(aabb_a, aabb_b, 1),
        "z": _signed_overlap_axis(aabb_a, aabb_b, 2),
    }


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _aabb_center(aabb):
    lo, hi = aabb
    return (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))


# ──────────────────────────────────────────────────────────────────
# Object-layer probes — migrated from object_query.py
# ──────────────────────────────────────────────────────────────────

# Pre-existing query symbols (migrated, no changes)
object_list_parts = _ok.list_parts
object_list_joints = _ok.list_joints
object_joint_table = _ok.joint_table
object_local_bbox = _ok.local_bbox
object_frame_of = _ok.frame_of
object_centroid = _ok.centroid
object_part_color = _ok.part_color
object_ir_snapshot = _ok.ir_snapshot
object_part_shape = _ok.part_shape
object_source_path = _ok.object_source_path


def object_global_bbox(ir, part_name=None, poses=None):
    """Object root-frame AABB of the whole object or one part.

    `poses=` follows the SDK surface convention (degrees for revolute).
    """
    return _ok.global_bbox(
        ir, part_name=part_name, poses=_object_poses_to_internal(ir, poses))


def object_outward(ir, part_name, anchor_name, poses=None, anchor_offset=None):
    """World-frame outward direction of an anchor on an object part."""
    return _ok.outward(
        ir, part_name, anchor_name,
        poses=_object_poses_to_internal(ir, poses),
        anchor_offset=anchor_offset)


def object_anchor_position(ir, *args, poses=None):
    """World-frame position of an anchor on an object part.

    Two call forms (dual-form per the anchor redesign):
        object_anchor_position(ir, entity_name, anchor_name)
        object_anchor_position(ir, anchor_ref)

    In the second form, the AnchorRef's offset (translation in
    part-local; rotation is irrelevant for points) and any pose tag
    are honored. In the first form, only the named AABB landmark is
    returned (no offset). `poses=` may be passed in either form to
    override the rest pose; values follow the SDK surface convention
    (degrees for revolute, meters for prismatic).
    """
    from ._refs import AnchorRef, IDENTITY_ANCHOR_OFFSET
    if len(args) == 1 and isinstance(args[0], AnchorRef):
        ref = args[0]
        ref_poses = ref._poses or None
        # Caller-supplied poses override the ref's pose tag.
        if poses is not None:
            from ._refs import poses_to_dict as _ptd
            ref_poses = (_ptd(ref_poses) if ref_poses else {})
            ref_poses = {**ref_poses, **(poses if isinstance(poses, dict) else dict(poses))}
        anchor_offset = (None if ref._offset == IDENTITY_ANCHOR_OFFSET
                         else ref._offset)
        return _ok.anchor_position(ir, ref._entity_name, ref._anchor_name,
                                   poses=_object_poses_to_internal(ir, ref_poses),
                                   anchor_offset=anchor_offset)
    if len(args) != 2:
        raise TypeError(
            "object_anchor_position: expected either an AnchorRef or "
            "an (entity_name, anchor_name) pair"
        )
    entity_name, anchor_name = args
    return _ok.anchor_position(ir, entity_name, anchor_name,
                               poses=_object_poses_to_internal(ir, poses))


def object_penetration_pairs(ir, *, sampled: bool = False, poses=None,
                             exact: bool = True, run_dir=None, cache=None):
    """Object-layer self-penetration pairs.

    Defaults to a rest-pose-or-targeted check (with optional `poses=`
    for a single targeted configuration). With `sampled=True` the full
    CCD sweep is run (15° revolute / 10 cm prismatic + 24 random
    multi-joint samples per the verifier defaults); each penetrating
    pair is reported once with `witnessing_pose` set to its worst-case
    sample.

    `sampled=True` and `poses=` are mutually exclusive.

    **Rows are `PenetrationPair` named tuples, not dicts**
    (`sdk/_object_kernel.py:32`). Read them as `row.a`, `row.b`,
    `row.signed_distance`, `row.exact_volume`, `row.adjudicated`, and
    `row.witnessing_pose` under `sampled=True`; `row["a"]` raises `TypeError`.

    **Exact geometry decides, and the order is load-bearing.** SAT is the broad
    phase; every candidate pose it flags is adjudicated against the parts'
    B-reps, and only then is the list reduced to one row per part pair.
    Reducing first would let a pose whose bounding-box overlap is a false
    positive around a bore evict the shallower pose that genuinely
    penetrated, and adjudication afterwards cannot recover it — the candidate
    is gone. `object_verify.no_self_penetration:150` orders it the same way.

    Rows that survive carry `adjudicated=True` with `exact_volume` and
    `exact_distance`. `adjudicated=False` means SAT's conservative verdict
    stands because nothing could answer — not that the pair is unconfirmed by
    choice.

    `exact=False` returns the raw SAT rows for a fast over-reporting answer.
    It also reduces before adjudicating, so a `sampled=True` result is then a
    bounding-box opinion about which pose is worst.

    Cost, measured: one CAD launch, ~1.4 s, and a whole CCD sweep costs barely
    more than a single pose — 37 samples of a hinged object came to 1.58 s
    against 1.36 s for one — because moving an unrelated joint does not change
    a pair's relative transform, so `cache_key` dedupes almost all of it. On the
    round-geometry fixture the sweep returns **no rows** where the boxes
    reported one, which is the same answer the compile gives.
    """
    if sampled and poses is not None:
        raise ValueError(
            "object_penetration_pairs: sampled=True and poses= are "
            "mutually exclusive — pick CCD sweep OR a single targeted pose"
        )
    internal = _object_poses_to_internal(ir, poses)
    if not sampled:
        candidates = [(p, internal)
                      for p in _ok.penetration_pairs(ir, poses=internal)]
    else:
        candidates = [
            (p, pose or None)
            for pose in _ccd.all_samples(ir.get("joints", []))
            for p in _ok.penetration_pairs(ir, poses=pose or None)]
    if not candidates:
        return []
    if exact:
        from ..verify import _exact_adjudicator as _adj
        kept, _stats = _adj.adjudicate_object_pairs(
            ir, candidates, run_dir=run_dir, cache=cache,
            # **B-rep, explicitly, because that is what judges this layer.**
            # `BACKEND_AUTO` resolves to the mesh narrow phase, which is right
            # above the object layer and wrong here: `no_self_penetration`
            # pins `BACKEND_BREP` so its rows can carry `exact_volume_cm3`,
            # the number the object prompt tells the agent to size a fix from,
            # and a probe that answered on a different backend than the
            # verdict would reintroduce the disagreement it exists to remove.
            backend=_adj.BACKEND_BREP)
    else:
        kept = [p for p, _pose in candidates]
    # Reduce to one row per part pair after adjudication. Keyed on
    # `signed_distance`, the SAT depth, which is what `no_self_penetration`
    # ranks its evidence by — so a probe and the verifier pick the same
    # witness.
    worst_per_pair: dict = {}
    for p in kept:
        key = (p.a, p.b)
        existing = worst_per_pair.get(key)
        if existing is None or p.signed_distance < existing.signed_distance:
            worst_per_pair[key] = p
    return [worst_per_pair[k] for k in sorted(worst_per_pair.keys())]

# Resolver helpers, shared with the verifiers and exposed in the probe overlay.
# resolve_entity_aabb / resolve_direction are passed IR-format dicts whose
# embedded `poses` fields are already in IR-internal units (they were
# converted at builder-serialization time), so no additional conversion
# happens here.
object_resolve_entity_aabb = _ok.resolve_entity_aabb
object_resolve_direction = _ok.resolve_direction


def object_joint_axis_world(ir, joint_name: str, poses=None):
    """Unit world-frame axis of a joint at the requested pose."""
    return _ok.joint_axis_world(
        ir, joint_name, poses=_object_poses_to_internal(ir, poses))


def object_part_world_aabb(ir, part_name: str, poses=None):
    """World-frame AABB of a part at the requested pose."""
    return _ok.part_world_aabb(
        ir, part_name, poses=_object_poses_to_internal(ir, poses))


# ──────────────────────────────────────────────────────────────────
# Object-layer collision probes
# ──────────────────────────────────────────────────────────────────


def _object_pair_obb_sat(ir, a: str, b: str, *, poses=None):
    """Build OBBs for two named parts and return the SATResult + AABBs.

    Returns `(sat: SATResult, aabb_a, aabb_b)`.
    """
    frames = _ok._compute_frames(ir, poses=poses)
    obb_a = _ok.obb_of_part(ir, a, frames)
    obb_b = _ok.obb_of_part(ir, b, frames)
    aabb_a = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, a)), frames[a])
    aabb_b = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, b)), frames[b])
    return _ok.sat_obb_obb(obb_a, obb_b), aabb_a, aabb_b


def _object_pair_exact(ir, a: str, b: str, *, poses, contact_tol, run_dir,
                       cache):
    """Exact verdict for one object part pair: `(collided, gap, exact)`.

    `exact=False` means nothing could answer — no interpreter, or a part
    `geometry_spec` cannot describe — and the caller keeps its box verdict.

    **Only worth calling when the boxes claim contact.** A box contains its
    part, so a pair the boxes separate really is separated: that verdict is
    already sound and a query would only refine a number nobody asked for. The
    boxes are wrong in one direction only, and this answers that direction.
    """
    from ..verify import _exact_adjudicator as _adj
    gaps, _stats = _adj.exact_object_part_gaps(
        ir, [(a, b)], poses=poses, run_dir=run_dir, cache=cache)
    gap = gaps[0] if gaps else None
    if gap is None:
        return None, None, False
    return gap <= float(contact_tol), gap, True


def object_pair_report(ir, a: str, b: str, *,
                       contact_tol: float = 1e-6,
                       poses=None, exact: bool = True,
                       run_dir=None, cache=None) -> dict:
    """Oriented-box collision diagnostic for two parts at the given pose.

    Returns a dict with SAT primaries (collided/signed_distance/
    contact_normal/contact_point), per-axis signed_overlap, and
    center_delta. `contact_tol` defaults to the NoSelfPenetration
    threshold (`_PENETRATION_EPS = 1e-6`), so `collided` matches the
    verifier's **broad phase** exactly at default tol. Pass a larger tol to
    ask "essentially touching."

    **`collided` is a surface verdict, and `exact_gap` is the measurement behind
    it.** The oriented boxes still describe `signed_distance`,
    `contact_normal`, `contact_point`, `signed_overlap`, and `center_delta` —
    they are the repair vector and the box algebra, and they keep their
    meaning. But an OBB bounds a round part as a square prism and a shelled
    body as its full outer box, so a box-derived `collided` reported contact
    that `NoSelfPenetration`'s B-rep pass then cleared: the probe this prompt
    sends you to after a failure disagreed with the verdict that produced it.

    So when the boxes claim contact the pair is measured against the same
    B-reps the verifier uses. `exact_gap` is meters between the surfaces, `0.0`
    when they share material, and `exact` says a query answered.

    **A pair the boxes separate is not queried, and needs no query.** A box
    contains its part, so box-separated is truly separated: `collided=False` is
    sound, `exact` is False, and `exact_gap` is None because no measurement was
    taken rather than because one failed. `exact_gap is None` alongside
    `collided=True` is the other case — nothing could answer, so the
    conservative box verdict stands.

    Cost: one CAD launch (~1.4 s measured), and only for a pair the boxes call
    touching. `exact=False` skips it for the old box-only answer.

    Self-pair (`a == b`) is allowed; returns trivially collided with
    signed_overlap equal to each axis's full extent.
    """
    poses = _object_poses_to_internal(ir, poses)
    sat, aabb_a, aabb_b = _object_pair_obb_sat(ir, a, b, poses=poses)
    overlap = _signed_overlap_dict(aabb_a, aabb_b)
    collided = sat.signed_distance < -float(contact_tol)
    exact_gap, answered = None, False
    # `a != b`: a part against itself shares all of its material by definition,
    # and the self-pair answer is documented as trivial — sending it would buy
    # a CAD launch to confirm an identity.
    if collided and exact and a != b:
        verdict, exact_gap, answered = _object_pair_exact(
            ir, a, b, poses=poses, contact_tol=contact_tol, run_dir=run_dir,
            cache=cache)
        if answered:
            collided = verdict
    return {
        "a": a,
        "b": b,
        "collided": collided,
        "exact": answered,
        "exact_gap": exact_gap,
        "measure": "obb",
        "contact_tol": float(contact_tol),
        "signed_distance": sat.signed_distance,
        "contact_normal": tuple(sat.axis),
        "contact_point": tuple(sat.contact_point) if sat.contact_point is not None else None,
        "signed_overlap": overlap,
        "center_delta": _vsub(_aabb_center(aabb_b), _aabb_center(aabb_a)),
    }


def _aabb_gap(aabb_a, aabb_b) -> float:
    """Lower bound on the surface gap between two world AABBs, 0 if they meet.

    The largest per-axis separation, which is the same formula
    `_gap_candidates` uses below — the two need to
    agree, because both are the prefilter in front of the same exact query and a
    pair one of them drops is a pair the other never measures.

    A *lower* bound, and the direction matters: a box contains its part, so the
    boxes are never further apart than the surfaces. That is what makes "the box
    gap already exceeds `tol`, so the true gap does too" sound, and it is also
    why the bound can never prove contact.
    """
    sep = 0.0
    for k in range(3):
        d = max(aabb_b[0][k] - aabb_a[1][k], aabb_a[0][k] - aabb_b[1][k])
        if d > 0.0:
            sep = max(sep, d)
    return sep


def object_gap(ir, a: str, b: str, *, poses=None, tol: float = 0.005,
               run_dir=None, cache=None):
    """Exact surface gap between two parts of this object, in meters.

    Returns `(gap_m, exact)`. This is the object layer's counterpart to the
    scene layer's `gap`: a surface measurement, where a bounding volume can
    report a contact that the B-rep check in `NoSelfPenetration` clears.

    A box is not a near-miss of this number, it is a different one. `pair_report`
    bounds a round part as a square prism: two 20 mm tubes offset diagonally by
    32 mm read as an 8 mm overlap where the surfaces clear each other by 5.3 mm.

    `0.0` means the two parts share material — measure the overlap with
    `penetration_pairs`, whose rows carry `exact_volume_cm3` once adjudicated.

    `poses` is `{joint: value}` in **surface** units (degrees for a revolute
    joint, meters for a prismatic one), like every other pose-taking probe here.

    `tol` is an ambiguity band, not a threshold: a pair whose boxes already
    separate by more than it is dropped without a CAD query, since the surfaces
    are then at least that far apart too. So the answer is exact at or below
    `tol` and a lower bound above it — widen `tol` to measure further out, at
    the cost of a CAD query. `exact=False` means nothing was measured and the
    number is that lower bound; check it before acting on a small value.
    """
    from ..verify import _exact_adjudicator as _adj

    internal = _object_poses_to_internal(ir, poses)
    aabb_a = _ok.part_world_aabb(ir, a, poses=internal)
    aabb_b = _ok.part_world_aabb(ir, b, poses=internal)
    bound = _aabb_gap(aabb_a, aabb_b)
    if bound > float(tol):
        return bound, False
    gaps, _stats = _adj.exact_object_part_gaps(
        ir, [(a, b)], poses=internal, run_dir=run_dir, cache=cache)
    gap = gaps[0] if gaps else None
    if gap is None:
        return bound, False
    return gap, True


def _normalize_axes(axes) -> "tuple[str, ...]":
    """Normalize `axes` into an ordered subset of ('x', 'y', 'z')."""
    if isinstance(axes, str):
        out = tuple(c for c in axes if c in ("x", "y", "z"))
    else:
        out = tuple(a for a in axes if a in ("x", "y", "z"))
    if not out:
        raise ValueError(f"axes must be a subset of x/y/z, got {axes!r}")
    return out


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def object_within_report(ir, inner: str, outer: str, *,
                         axes="xyz",
                         poses=None) -> dict:
    """Per-axis containment check: is `inner` AABB inside `outer` AABB?

    Returns a dict with per-axis lower/upper margins. `within=True` iff
    all listed-axis margins are >= 0.
    """
    poses = _object_poses_to_internal(ir, poses)
    axes_t = _normalize_axes(axes)
    frames = _ok._compute_frames(ir, poses=poses)
    inner_aabb = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, inner)), frames[inner])
    outer_aabb = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, outer)), frames[outer])
    margins: dict = {}
    overall = True
    for ax in axes_t:
        i = _AXIS_INDEX[ax]
        lower = inner_aabb[0][i] - outer_aabb[0][i]
        upper = outer_aabb[1][i] - inner_aabb[1][i]
        ax_within = (lower >= 0.0) and (upper >= 0.0)
        margins[ax] = {
            "lower_margin": lower,
            "upper_margin": upper,
            "within": ax_within,
        }
        if not ax_within:
            overall = False
    return {
        "inner": inner,
        "outer": outer,
        "axes": list(axes_t),
        "within": overall,
        "margins": margins,
    }


def _rank_key(entry):
    """Sort key for nearest_neighbors / find_clearance_risks.

    Collided pairs come first (0), then by signed_distance ascending
    (most-negative collisions deepest; smallest separations first).
    """
    return (0 if entry["collided"] else 1, entry["signed_distance"])


def _object_pair_entry(ir, a: str, b: str, *, contact_tol: float, poses):
    """Compact per-pair entry for ranked search probes.

    `measure` is carried for the same reason the scene twin carries it: a
    reader has to be able to tell an oriented-box answer from a surface one
    without knowing which probe produced it.
    """
    sat, aabb_a, aabb_b = _object_pair_obb_sat(ir, a, b, poses=poses)
    return {
        "a": a,
        "b": b,
        "measure": "obb",
        "collided": sat.signed_distance < -float(contact_tol),
        # Present on every row, so a caller never has to test for the key.
        # `exact=False` with `exact_gap=None` on a row that is *not* collided
        # means no measurement was taken and none was needed: a box contains
        # its part, so box-separated is truly separated. On a collided row it
        # means nothing could answer and the box verdict stands.
        "exact": False,
        "exact_gap": None,
        "signed_distance": sat.signed_distance,
        "signed_overlap": _signed_overlap_dict(aabb_a, aabb_b),
    }


def _adjudicate_object_rows(ir, entries, *, contact_tol, poses, run_dir,
                            cache):
    """Correct `collided` on the rows a ranked probe is about to return.

    **Only the rows being returned, and only the ones claiming contact.** The
    ranking stays a bounding-box triage — it is over an O(n^2) pool and a box
    bounds its part, so a pair the boxes separate really is separated and ranks
    no worse than it deserves. What a box cannot do is claim contact, so that
    claim is the only thing measured.

    One batch, and therefore one CAD launch whatever `limit` is: `_invoke`
    sends the whole query list to a single child, so the cost (about 1.4 s) is
    per launch, not per pair.
    """
    rows = [e for e in entries if e["collided"] and e["a"] != e["b"]]
    if not rows:
        return entries
    from ..verify import _exact_adjudicator as _adj
    gaps, _stats = _adj.exact_object_part_gaps(
        ir, [(e["a"], e["b"]) for e in rows], poses=poses, run_dir=run_dir,
        cache=cache)
    for entry, gap in zip(rows, gaps):
        entry["exact_gap"] = gap
        entry["exact"] = gap is not None
        if gap is not None:
            entry["collided"] = gap <= float(contact_tol)
    return entries


def object_nearest_neighbors(ir, name: str, *,
                             candidates=None,
                             limit: int = 5,
                             contact_tol: float = 1e-6,
                             poses=None, exact: bool = True,
                             run_dir=None, cache=None) -> list:
    """Return up to `limit` parts nearest to `name`.

    Ordered by `(0 if collided else 1, signed_distance asc)`. Each
    entry has `a`, `b`, `collided`, `signed_distance`, `signed_overlap`.
    `candidates` defaults to all other parts; self filtered.

    **The ranking is a bounding-box triage; the contact claim is measured.** A
    box bounds its part, so a pair the boxes separate really is separated and
    ranking on boxes is sound — that is the right tool for choosing which pair
    to look at. But `collided` is a claim about contact, and an oriented box
    holds a round part as a square prism, so the returned rows that claim
    contact are re-checked against the parts' B-reps: `exact_gap` is the
    measured separation and `exact` says a query answered.

    Only the `limit` rows being returned are measured, in one batch — so one
    CAD launch regardless of `limit`, and none at all when nothing claims
    contact. `object_gap` remains the way to get a number for a pair the
    ranking has separated.
    """
    poses = _object_poses_to_internal(ir, poses)
    if candidates is None:
        candidates = [p["name"] for p in ir["parts"]]
    pool = [c for c in candidates if c != name]
    entries = [_object_pair_entry(ir, name, c, contact_tol=contact_tol, poses=poses)
               for c in pool]
    entries.sort(key=_rank_key)
    out = entries[: max(0, int(limit))]
    if not exact:
        return out
    return _adjudicate_object_rows(ir, out, contact_tol=contact_tol,
                                   poses=poses, run_dir=run_dir, cache=cache)


def object_find_clearance_risks(ir, *,
                                parts=None,
                                limit: int = 10,
                                contact_tol: float = 1e-6,
                                poses=None, exact: bool = True,
                                run_dir=None, cache=None) -> list:
    """Return up to `limit` riskiest part-pairs globally.

    Same sort key as `object_nearest_neighbors` and the same treatment: the
    ranking is a bounding-box triage over every pair, and the returned rows
    that claim contact are measured against the B-reps.
    """
    poses = _object_poses_to_internal(ir, poses)
    pool = [p["name"] for p in ir["parts"]] if parts is None else list(parts)
    entries = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            entries.append(_object_pair_entry(
                ir, pool[i], pool[j], contact_tol=contact_tol, poses=poses))
    entries.sort(key=_rank_key)
    out = entries[: max(0, int(limit))]
    if not exact:
        return out
    return _adjudicate_object_rows(ir, out, contact_tol=contact_tol,
                                   poses=poses, run_dir=run_dir, cache=cache)


# ──────────────────────────────────────────────────────────────────
# Object-layer assertion-mirror probes
# ──────────────────────────────────────────────────────────────────


def _object_anchor_world(ir, anchor_tuple, poses):
    """Resolve `(entity, anchor)` tuple at given pose for object layer."""
    entity, anchor = anchor_tuple
    return _ok.anchor_position(ir, entity, anchor, poses=poses)


def _axis_arg(axis):
    """An axis argument as a vector, accepting the name form authors write.

    `axis='y'` is what the assertion surface already takes —
    `_assertions.AXIS_LITERALS` has held `'x'`, `'-z'` and the rest since
    axes were nameable — but the probes that MIRROR those assertions passed
    the value straight to the evaluator, so the same spelling that works in
    `assert_overlap` died in `overlap_along` with
    `ValueError: could not convert string to float: 'y'`.

    `None` passes through: for `distance_between` and `motion_displacement`
    it means "unprojected distance", which is not an axis.
    """
    if axis is None or not isinstance(axis, str):
        return axis
    from ._assertions import AXIS_LITERALS
    unit = AXIS_LITERALS.get(axis.strip().lower())
    if unit is None:
        raise ValueError(
            f"unknown axis name {axis!r}; use 'x', 'y', 'z', a signed form "
            f"like '-z', or a vector (x, y, z)")
    return unit


def object_distance_between(ir, a, b, *, axis=None, poses=None) -> float:
    """Distance between two anchors. Mirrors `Object.assert_distance`.

    `a` and `b` are `(entity, anchor)` tuples. `axis=None` →
    Euclidean magnitude; `axis=(x, y, z)` → signed projection
    `(p_b - p_a) · unit(axis)` (antisymmetric in `a` / `b`).
    Raises `ValueError` on zero-norm `axis`.
    """
    poses = _object_poses_to_internal(ir, poses)
    p_a = _object_anchor_world(ir, a, poses)
    p_b = _object_anchor_world(ir, b, poses)
    return eval_distance_scalar(p_a, p_b, axis=_axis_arg(axis))


def object_overlap_along(ir, a: str, b: str, *, axis, poses=None) -> float:
    """Signed projection overlap on a unit axis. Mirrors `assert_overlap`.

    `a` and `b` are part names. Returns
    `min(a_hi, b_hi) - max(a_lo, b_lo)` along `unit(axis)` (positive =
    overlap depth, negative = gap distance). Symmetric in `a` / `b`.
    Raises `ValueError` on zero-norm `axis`.
    """
    poses = _object_poses_to_internal(ir, poses)
    frames = _ok._compute_frames(ir, poses=poses)
    aabb_a = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, a)), frames[a])
    aabb_b = _ok._aabb_of_transformed_corners(
        _ok._box_corners(_ok.local_bbox(ir, b)), frames[b])
    return eval_overlap_scalar(aabb_a, aabb_b, _axis_arg(axis))


def object_angle_between(ir, dir_a, dir_b) -> float:
    """Unsigned angle (degrees in [0, 180]) between two world-frame
    direction vectors. Mirrors `assert_angle`.

    Inputs are bare 3-tuples (any non-zero magnitude; normalized
    internally). Raises `ValueError` on zero-norm input.

    **No `poses=`.** The directions are already resolved, so there is nothing
    here for a pose to act on; it belongs on the call that produced them —
    `joint_axis_world(ir, joint, poses=...)` for a joint's axis,
    `outward(ir, entity, anchor, poses=...)` for an anchor's outward.
    """
    return eval_angle_scalar(tuple(dir_a), tuple(dir_b))


def object_motion_displacement(ir, entity: str, *,
                                anchor: str = "center",
                                from_pose=None,
                                to_pose,
                                axis=None) -> float:
    """Signed/euclidean displacement of an anchor between two poses.

    Mirrors the scalar underlying `assert_opens_toward`. `from_pose`
    defaults to rest (None or `{}`); `to_pose` is required. `axis=None`
    is Euclidean displacement (symmetric); axis-projected is
    antisymmetric in `from_pose` / `to_pose`.
    """
    from_pose_i = _object_poses_to_internal(ir, from_pose or None)
    to_pose_i = _object_poses_to_internal(ir, to_pose)
    p_from = _ok.anchor_position(ir, entity, anchor, poses=from_pose_i)
    p_to = _ok.anchor_position(ir, entity, anchor, poses=to_pose_i)
    return eval_distance_scalar(p_from, p_to, axis=_axis_arg(axis))


# ──────────────────────────────────────────────────────────────────
# Object-layer verifier-mirror probe: support_report
# ──────────────────────────────────────────────────────────────────


def object_support_report(ir, *, run_dir=None, cache=None) -> dict:
    """Mirror of `SupportConnected.evidence` at the object layer.

    Returns `{is_supported, floor_z, contacts, grounded, supported,
    unsupported, exact_contact}`. Uses the same contact graph + kinematic-edge
    crediting as the verifier; rest-pose only (matches the verifier).

    `run_dir` and `cache` reach the exact contact pass, and the overlay binds
    them — without them a `solid` part's staged `.brep` cannot be resolved and
    every pair falls back to its bounding box, i.e. the probe would answer from
    boxes about a verdict that came from surfaces. `exact_contact` says which
    happened.

    **One graph build.** `is_supported` and `floor_z` come from the same
    adjacency walk as the rest of the report, so with exact geometry there is
    one adjudication round, not two.
    """
    from ..verify._support import (
        build_object_adjacency, contact_tol_m, joint_tol_m, _bfs_from_floor,
        _global_zmin, _object_part_world_aabbs,
    )
    tol_f = contact_tol_m()
    joint_tol_f = joint_tol_m()
    # `complete_contacts` because this probe answers "which parts touch", not
    # "is it supported". The verifier measures pairs only until connectivity is
    # settled and reports `exact_contact["complete"] = False` for it; a caller
    # asking for the contact list must get all of them, not a spanning subset.
    adj = build_object_adjacency(ir, _ok, tol_f, joint_tol_f,
                                 run_dir=run_dir, cache=cache,
                                 complete_contacts=True)
    reached = _bfs_from_floor(adj)
    part_names = [p["name"] for p in ir.get("parts", [])]
    contacts = []
    seen = set()
    for a, neighbors in adj.edges.items():
        if a == "__floor__":
            continue
        for b in neighbors:
            if b == "__floor__":
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            contacts.append(key)
    grounded = sorted(adj.edges.get("__floor__", set()))
    unsupported = [n for n in part_names if n not in reached]
    return {
        "is_supported": not unsupported,
        "floor_z": _global_zmin(_object_part_world_aabbs(ir, _ok)),
        "tol_m": tol_f,
        "joint_tol_m": joint_tol_f,
        "contacts": sorted(contacts),
        "grounded": grounded,
        "supported": sorted(n for n in part_names if n in reached),
        "unsupported": unsupported,
        "exact_contact": adj.exact,
    }


# ──────────────────────────────────────────────────────────────────
# Scene-layer probes — migrated from scene_query.py
# ──────────────────────────────────────────────────────────────────

# Pre-existing query symbols
scene_list_instances = _sk.list_instances
scene_list_groups = _sk.list_groups
scene_list_assertions = _sk.list_assertions


def scene_instance_pose(ir, catalog, name: str, poses=None, *, memo=None,
                        world_poses=None, base_deltas=None, mount_deltas=None):
    """World pose `((x,y,z), (rx,ry,rz) degrees)` of an instance at any pose.

    Takes the same three pose kwargs as every other probe here, over
    `_scene_kernel.instance_pose`.

    `world_poses` is most useful for an instance *other* than the one being
    asked about: "where does the cup end up if I move the tray to X" is a
    question about the attach forest, and the answer is not the tray's pose.
    """
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.instance_pose(ir, catalog, name, pi, memo=memo,
                             base_deltas=bd, mount_deltas=mount_deltas)


scene_group_pose = _sk.group_pose
scene_instance_bbox = _sk.instance_bbox
scene_group_bbox = _sk.group_bbox
scene_scene_bbox = _sk.scene_bbox


def scene_joint_limits(ir, catalog, instance_name: str) -> dict:
    """Every joint on a placed instance, in **surface units**.

    An `ObjectHandle` reports `joints` as a bare tuple of names — no type, no
    range, no rest value — and the raw catalog IR answers in *internal* units
    (radians for a revolute joint), which nothing else on the agent's surface
    speaks. This answers in surface units, with each joint's type alongside.
    The animation layer needs it most: driving joints is half its job.

    `unit` is returned as data rather than left to the reader. The pose
    predicates spell their arguments `value_deg` / `tol_deg` and those names
    are wrong for a prismatic joint — the value is meters. A caller that reads
    `unit` cannot make that mistake.

    Keyed by joint name because the question is almost always about one named
    joint; order carries no meaning here.

    `default` is **this instance's committed configuration** — its
    `initial_joints` value where the author selected one, and the definition's
    `default=` otherwise. That is what every consumer of this record actually
    wants: `animation/sim/scene_export.py` copies it into `scene_meta.json`,
    from which `dycraft_ik.default_q` seeds each robot's joints, and reporting
    the definition value there would load two cabinets placed from one
    definition into the same configuration however the scene was authored.
    `object_default`
    carries the definition's own value alongside it, so the distinction between
    "what this body is set to" and "what the definition rests at" stays
    inspectable rather than being collapsed.

    Committed-state facts only. For a joint's value under some *other* pose,
    use `instance_pose` (scene) or `pose_at` (animation).
    """
    from . import _rotation as _rot

    inst = _sk._index_instances(ir).get(instance_name)
    if inst is None:
        known = sorted(i["name"] for i in ir.get("instances") or ())
        raise KeyError(
            f"no instance named {instance_name!r}; known: {', '.join(known)}")
    obj = catalog.get(inst["object_ref"])
    if obj is None:
        raise KeyError(
            f"{instance_name!r} references object {inst['object_ref']!r}, "
            f"which is not in the catalog")

    saved = _sk.instance_initial_joints(ir, instance_name) or {}
    out: dict = {}
    for j in obj.get("joints") or ():
        jtype = j.get("type")
        if jtype == "fixed":
            # No value to drive, so no limits to report — but say it exists,
            # or "the joint I wanted is missing" and "it is fixed" look the
            # same from here.
            out[j["name"]] = {"type": "fixed", "limits": None,
                              "default": None, "object_default": None,
                              "axis": None, "unit": None}
            continue
        to_surface = _rot.convert_pose_value_to_surface
        lo, hi = j["limits"]
        obj_default = j["default"]
        committed = saved.get(j["name"], obj_default)
        out[j["name"]] = {
            "type": jtype,
            "limits": (to_surface(jtype, lo), to_surface(jtype, hi)),
            "default": to_surface(jtype, committed),
            "object_default": to_surface(jtype, obj_default),
            "axis": tuple(j["axis"]) if j.get("axis") else None,
            "unit": "m" if jtype == "prismatic" else "deg",
        }
    return out


def scene_list_anchors(ir, catalog, instance_name: str) -> dict:
    """Declared anchors on a placed instance, keyed by part.

    `{part: [{"name", "kind", "translation"}, ...]}` — part-local meters, the
    same frame `part_mount_local` reports. Parts with none are omitted, so an
    empty dict means this object registered no named frames and the 27 built-in
    AABB landmarks are the whole vocabulary.

    The built-ins are deliberately not listed: they exist on every part, and
    `ObjectHandle.part_anchor_names` already names them.

    Read declared anchors here rather than guessing a `<part>.<name>` from an
    error message, or measuring the mesh. An imported robot is where it
    matters: the Panda declares its TCP as `panda_hand.panda_grasptarget`,
    0.105 m along the hand's +Z, which a value measured from the mesh misses
    by about 1.6 mm.

    `kind` is the author's own label (`contact_frame`, `mount_frame`,
    `hinge_axis`, `imported_frame`, …) and is worth reading: it says what the
    frame was *for*, which the name alone often does not.
    """
    inst = _sk._index_instances(ir).get(instance_name)
    if inst is None:
        known = sorted(i["name"] for i in ir.get("instances") or ())
        raise KeyError(
            f"no instance named {instance_name!r}; known: {', '.join(known)}")
    obj = catalog.get(inst["object_ref"])
    if obj is None:
        raise KeyError(
            f"{instance_name!r} references object {inst['object_ref']!r}, "
            f"which is not in the catalog")
    out: dict = {}
    for part in obj.get("parts") or ():
        rows = []
        for a in part.get("anchors") or ():
            # `Mapping`, not `dict`: a probe reads the catalog through a
            # `_readonly.freeze()` view, whose records are `_ReadOnlyMapping`.
            # `Mapping`, not `dict`: the IR may be a frozen view, and a `dict`
            # check would skip every row and return `{}`, which the docstring
            # above reads as "this object registered no named frames".
            if not isinstance(a, Mapping) or not a.get("name"):
                continue
            rows.append({
                "name": a["name"],
                "kind": a.get("kind"),
                "translation": [round(float(v), 6)
                                for v in a.get("translation") or (0.0, 0.0, 0.0)],
            })
        if rows:
            out[part["name"]] = sorted(rows, key=lambda r: r["name"])
    return out


def scene_outward(ir, catalog, entity_name: str, anchor_name: str,
                  poses=None, anchor_offset=None, *, world_poses=None,
                  base_deltas=None):
    """World-frame outward direction of a scene anchor."""
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.outward(ir, catalog, entity_name, anchor_name, poses=pi,
                       anchor_offset=anchor_offset, base_deltas=bd)


def scene_anchor_position(ir, catalog, *args, poses=None, world_poses=None,
                          base_deltas=None):
    """World-frame position of a scene anchor.

    Two call forms:
        scene_anchor_position(ir, catalog, entity_name, anchor_name)
        scene_anchor_position(ir, catalog, anchor_ref)

    `poses=` values follow the SDK surface convention (degrees for revolute
    joints, meters for prismatic); `world_poses=` says where the bodies are.
    Together they are the whole of forward kinematics for an actor that both
    articulates and travels.
    """
    from ._refs import AnchorRef, IDENTITY_ANCHOR_OFFSET
    if len(args) == 1 and isinstance(args[0], AnchorRef):
        ref = args[0]
        ref_poses = ref._poses or None
        if poses is not None:
            from ._refs import poses_to_dict as _ptd
            ref_poses = (_ptd(ref_poses) if ref_poses else {})
            ref_poses = {**ref_poses, **(poses if isinstance(poses, dict) else dict(poses))}
        anchor_offset = (None if ref._offset == IDENTITY_ANCHOR_OFFSET
                         else ref._offset)
        pi, bd = _pose_kwargs(ir, catalog, ref_poses, world_poses, base_deltas)
        return _sk.anchor_position(ir, catalog, ref._entity_name,
                                   ref._anchor_name, poses=pi,
                                   anchor_offset=anchor_offset,
                                   base_deltas=bd)
    if len(args) != 2:
        raise TypeError(
            "scene_anchor_position: expected either an AnchorRef or "
            "an (entity_name, anchor_name) pair"
        )
    entity_name, anchor_name = args
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.anchor_position(ir, catalog, entity_name, anchor_name,
                               poses=pi, base_deltas=bd)


def scene_anchor_pose(ir, catalog, *args, poses=None, world_poses=None,
                      base_deltas=None):
    """World-frame **frame** of a scene anchor: `((x,y,z),(rx,ry,rz))` degrees.

    Same two call forms and the same pose channels as `anchor_position`; what
    it adds is the rotation, and the rotation is the whole reason a declared
    semantic anchor exists. `add_anchor(part, "wall_mount", translation=...,
    rotation=(90, 0, 0))` says which way the mounting face looks, `attach`
    composes that, and this is how you check it landed — `anchor_position`
    reports the same point whether the frame came out right or 90 degrees out,
    and `outward` refuses a declared name because a face normal is not what an
    authored frame is.

    Useful on a built-in landmark too: the rotation there is the body's own
    orientation at that point, which is what a mating check between two
    articulated instances actually compares.
    """
    from ._refs import AnchorRef, IDENTITY_ANCHOR_OFFSET
    if len(args) == 1 and isinstance(args[0], AnchorRef):
        ref = args[0]
        ref_poses = ref._poses or None
        if poses is not None:
            from ._refs import poses_to_dict as _ptd
            ref_poses = (_ptd(ref_poses) if ref_poses else {})
            ref_poses = {**ref_poses,
                         **(poses if isinstance(poses, dict) else dict(poses))}
        anchor_offset = (None if ref._offset == IDENTITY_ANCHOR_OFFSET
                         else ref._offset)
        pi, bd = _pose_kwargs(ir, catalog, ref_poses, world_poses, base_deltas)
        return _sk.anchor_pose(ir, catalog, ref._entity_name,
                               ref._anchor_name, poses=pi,
                               anchor_offset=anchor_offset, base_deltas=bd)
    if len(args) != 2:
        raise TypeError(
            "scene_anchor_pose: expected either an AnchorRef or "
            "an (entity_name, anchor_name) pair"
        )
    entity_name, anchor_name = args
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.anchor_pose(ir, catalog, entity_name, anchor_name,
                           poses=pi, base_deltas=bd)


scene_ir_snapshot = _sk.ir_snapshot


def _attach_depth(ir, name: str) -> int:
    """How many attach edges lie between `name` and the room.

    Used to order world-pose
    resolution parents-first, because a child's placement depends on where its
    parent ended up.
    """
    instances = _sk._index_instances(ir)
    depth, seen, cur = 0, {name}, name
    while True:
        inst = instances.get(cur)
        parent = (inst or {}).get("placement", {}).get("parent")
        if parent is None or parent == _sk._ROOM_RESERVED_NAME:
            return depth
        if parent in seen:      # cannot happen in a committed scene; don't hang
            return depth
        seen.add(parent)
        cur = parent
        depth += 1


def _world_pose_target(value, placed):
    """One `world_poses` value as an internal `(translation, quat)`.

    Three accepted spellings, in the units the rest of this surface speaks
    (meters, then euler-XYZ degrees):

      * `(x, y, z)` — move it there, leave its attitude alone.
      * `((x, y, z), (rx, ry, rz))` — the full pose, exactly what
        `keys("world_pose")` takes.
      * `(x, y, z, rx, ry, rz)` — the same, flat, because a loop computing six
        numbers naturally produces one.

    `placed` supplies whichever half was omitted.
    """
    def _scalars(v, n):
        return (_ro.is_sequence(v) and len(v) == n
                and all(isinstance(x, (int, float)) for x in v))

    if _scalars(value, 3):
        return (tuple(float(x) for x in value), placed[1])
    if _scalars(value, 6):
        t, r = value[:3], value[3:]
    elif (_ro.is_sequence(value) and len(value) == 2
            and all(_scalars(h, 3) for h in value)):
        t, r = value
    else:
        raise ValueError(
            "world_poses values are (x, y, z), ((x,y,z), (rx,ry,rz)) or a flat "
            f"6-sequence in meters + euler-XYZ degrees; got {value!r}")
    return (tuple(float(x) for x in t),
            _rot.euler_xyz_deg_to_quat_wxyz(float(r[0]), float(r[1]),
                                            float(r[2])))


def _pose_kwargs(ir, catalog, poses=None, world_poses=None, base_deltas=None):
    """Normalize this surface's three pose kwargs into the kernel's two.

    Returns `(poses_internal, base_deltas)`.

    `poses` is converted from surface units. `base_deltas` passes through — an
    SE3 is an SE3 on both sides. `world_poses` is the form an author actually
    has, and is divided by the placement here rather than by the caller:

        world = delta o placed   =>   delta = world o placed^-1

    `placed` must exclude the entity's own delta while including its ancestors',
    so resolution runs **parents first** (`_attach_depth`) with each entity
    placed against the deltas resolved so far. That is what makes "put the tray
    at X and the cup on it at Y" mean what it says.

    This is the composition a committed `keys("world_pose")` uses, so a pose tested here is the
    pose it produces.
    """
    poses_internal = _scene_poses_to_internal(ir, catalog, poses)
    if not world_poses:
        return poses_internal, (base_deltas or None)
    out = dict(base_deltas or {})
    clash = sorted(set(out) & set(world_poses))
    if clash:
        raise ValueError(
            f"{clash} given in both world_poses= and base_deltas= — one says "
            "where to put the body, the other how far to displace it, and "
            "there is no sensible way to apply both. Pick one per instance.")
    for name in sorted(world_poses, key=lambda n: (_attach_depth(ir, n), n)):
        placed = _sk._resolve_placed_pose(ir, catalog, name, {},
                                          poses=poses_internal,
                                          base_deltas=out or None)
        want = _world_pose_target(world_poses[name], placed)
        out[name] = _rot.se3_compose(want, _rot.se3_inverse(placed))
    return poses_internal, (out or None)


def _scene_joints_for_ccd(ir, catalog) -> list:
    """Aggregate joint records across instances with qualified
    `<instance>.<joint>` names. Mirrors the verifier's `_scene_joints`
    helper but here for probe-side reuse.
    """
    out = []
    for inst in ir.get("instances", []):
        obj_ref = inst.get("object_ref")
        if obj_ref not in catalog:
            continue
        obj_ir = catalog[obj_ref]
        for j in obj_ir.get("joints", ()):
            if j["type"] == "fixed":
                continue
            out.append({
                "name": f"{inst['name']}.{j['name']}",
                "type": j["type"],
                "default": j["default"],
                "limits": j["limits"],
            })
    return out


def scene_penetration_pairs(ir, catalog, *, sampled: bool = False, poses=None,
                            world_poses=None, base_deltas=None,
                            exact: bool = True, run_dir=None,
                            asset_base_dir=None, cache=None):
    """Scene-layer cross-instance penetration pairs.

    See `object_penetration_pairs` for kwarg semantics. The CCD sweep
    uses qualified `<instance>.<joint>` keys; each scene-level pose dict
    is projected to per-instance poses via `_object_poses_for_instance`
    inside `scene_query.penetration_pairs`.

    `world_poses={"mug": (x, y, z)}` says where the bodies are, so this answers
    *does this body hit anything once I move it there* — see `_pose_kwargs`.

    `sampled=True` is mutually exclusive with a targeted pose in any channel: a
    CCD sweep chooses its own joint samples, and a pose held fixed across them
    would be a different question at every sample.

    **Exact geometry decides, and it is reached over the raw part pairs.** The
    probe builds on `penetration_part_pairs`, adjudicates every SAT part pair
    and pose, and reduces to one row per instance pair afterwards, so the part
    identity (`a_sub`) and the pose of a real overlap reach the adjudicator.

    Rows name the part pair in `a_sub` / `b_sub` and carry `adjudicated=True`
    when exact geometry confirmed them; `adjudicated=False` means nothing could
    answer and SAT's conservative verdict stands.

    `exact=False` returns the raw SAT rows — fast, over-reporting, and with the
    old aggregation order, so a `sampled=True` result is then a bounding-box
    opinion about which pose is worst. `gap` measures a named pair exactly.
    """
    if sampled and (poses is not None or world_poses is not None
                    or base_deltas is not None):
        raise ValueError(
            "scene_penetration_pairs: sampled=True and poses=/world_poses=/"
            "base_deltas= are mutually exclusive — pick CCD sweep OR a single "
            "targeted pose"
        )
    if not sampled:
        pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
        candidates = [(p, pi, bd, None) for p in
                      _sk.penetration_part_pairs(ir, catalog, poses=pi,
                                                 base_deltas=bd)]
    else:
        joints = _scene_joints_for_ccd(ir, catalog)
        candidates = [
            (p, pose or None, None, None)
            for pose in _ccd.all_samples(joints)
            for p in _sk.penetration_part_pairs(ir, catalog,
                                                poses=pose or None)]
    if not candidates:
        return []
    if exact:
        from ..verify import _exact_adjudicator as _adj
        kept, _stats = _adj.adjudicate_scene_pairs(
            ir, catalog, candidates, run_dir=run_dir,
            asset_base_dir=asset_base_dir, cache=cache)
    else:
        kept = [p for p, *_rest in candidates]
    # One row per INSTANCE pair, chosen after adjudication: the same reduction
    # `no_cross_instance_penetration` applies to its own survivors. The part
    # names ride along in `a_sub` / `b_sub`.
    worst_per_pair: dict = {}
    for p in kept:
        key = (p.a, p.b)
        existing = worst_per_pair.get(key)
        if existing is None or p.signed_distance < existing.signed_distance:
            worst_per_pair[key] = p
    return [worst_per_pair[k] for k in sorted(worst_per_pair.keys())]

# Resolver helpers
# resolve_entity_aabb / resolve_direction take IR-format dicts whose
# embedded `poses` are already in internal units (see _probes.py header).
scene_resolve_entity_aabb = _sk.resolve_entity_aabb
scene_resolve_direction = _sk.resolve_direction
scene_room_world_aabb = _sk.room_world_aabb


def scene_instance_world_aabb(ir, catalog, instance_name: str, poses=None, *,
                              world_poses=None, base_deltas=None):
    """World-frame AABB of an instance at the requested pose."""
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.instance_world_aabb(ir, catalog, instance_name,
                                   poses=pi, base_deltas=bd)


def scene_group_world_aabb(ir, catalog, group_name: str, poses=None, *,
                           world_poses=None, base_deltas=None):
    """World-frame AABB of a group at the requested pose."""
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _sk.group_world_aabb(ir, catalog, group_name,
                                poses=pi, base_deltas=bd)


# ──────────────────────────────────────────────────────────────────
# Scene-layer collision probes
# ──────────────────────────────────────────────────────────────────


def _scene_entity_world_aabb(ir, catalog, name: str, poses=None, *,
                             base_deltas=None):
    """Resolve any scene entity name to its world AABB at given pose.

    Accepted forms:
      * `"<instance>"` — an instance name
      * `"<group>"` — a group name
      * `"room"` (the reserved name) — the room's world AABB
      * `"<instance>.<part>"` — part-scoped within an instance

    Internal: takes the kernel's two kwargs, so `poses` is INTERNAL-units
    (radians for revolute). The public probes (`scene_pair_report`,
    `scene_within_report`, etc.) call `_pose_kwargs` at their own entry and pass
    the result down.

    `room` is the one branch that ignores `base_deltas`: it is the world frame,
    and nothing an actor does moves it.
    """
    # Room: the reserved scene-layer entity name.
    if name == _sk._ROOM_RESERVED_NAME and ir.get("room"):
        return _sk.room_world_aabb(ir)

    # Part-scoped: "<instance>.<part>" → world AABB of that part.
    if "." in name:
        inst_name, _, part_name = name.partition(".")
        instances = _sk._index_instances(ir)
        if inst_name not in instances:
            raise KeyError(inst_name)
        inst = instances[inst_name]
        obj_ir = catalog[inst["object_ref"]]
        obj_poses = _sk.effective_object_poses(ir, inst_name, poses)
        # Part's world AABB inside the instance's object root frame.
        local_aabb = _ok.part_world_aabb(obj_ir, part_name, poses=obj_poses)
        # Transform through instance's world pose. `poses` must be forwarded:
        # the bare-instance branch below does, and without it a part-scoped
        # query silently ignores scene-level articulation of the attach chain.
        # `base_deltas` likewise: at the animation layer a part-scoped question
        # is usually about a body that has moved.
        world_pose = _sk._resolve_world_pose(ir, catalog, inst_name, {},
                                             poses=poses,
                                             base_deltas=base_deltas)
        se3 = _sk._se3_from_ir_pose(
            {"translation": list(world_pose[0]), "rotation": list(world_pose[1])})
        corners = _ok._box_corners(local_aabb)
        return _ok._aabb_of_transformed_corners(corners, se3)

    # Bare name: instance or group.
    instances = _sk._index_instances(ir)
    groups = _sk._index_groups(ir)
    if name in instances:
        return _sk.instance_world_aabb(ir, catalog, name, poses=poses,
                                       base_deltas=base_deltas)
    if name in groups:
        return _sk.group_world_aabb(ir, catalog, name, poses=poses,
                                    base_deltas=base_deltas)
    raise KeyError(name)


def _scene_entity_obbs(ir, catalog, name: str, poses=None, *,
                       base_deltas=None):
    """`[(part_name, obb), ...]` for a scene entity, or None if it has none.

    An instance is a set of placed bodies, so it has real oriented boxes — the
    same ones `penetration_part_pairs` tests, built by the same kernel helper so
    the two cannot disagree. A part-scoped `"<instance>.<part>"` name is one of
    them.

    A **group** and the **room** return None. A group is a union of instances
    rather than a placed body, and the room is the world frame; neither has a
    single orientation, so the caller falls back to the world AABB and says so.
    """
    if name == _sk._ROOM_RESERVED_NAME and ir.get("room"):
        return None
    inst_name, _sep, part_name = name.partition(".")
    instances = _sk._index_instances(ir)
    if inst_name not in instances:
        return None                     # a group, or a name only the AABB
                                        # resolver knows how to reach
    obbs = _sk._instance_part_obbs(ir, catalog, inst_name, {}, poses=poses,
                                   base_deltas=base_deltas)
    if not part_name:
        return obbs
    scoped = [(n, o) for n, o in obbs if n == part_name]
    return scoped or None


def _worst_obb_pair(obbs_a, obbs_b):
    """`(SATResult, part_a, part_b)` for the deepest part pair across two sets.

    Deepest by `signed_distance`, which is the aggregation
    `_scene_kernel.penetration_pairs` uses to reduce its part pairs to one row
    per instance pair — so a probe and the sweep answer the same question about
    the same geometry.
    """
    best = None
    for na, oa in obbs_a:
        for nb, ob in obbs_b:
            res = _ok.sat_obb_obb(oa, ob)
            if best is None or res.signed_distance < best[0].signed_distance:
                best = (res, na, nb)
    return best


def _aabb_to_obb(aabb):
    """Convert an axis-aligned AABB to an OBB tuple `(center, R_cols, half)`.

    The R_cols are world axes (identity rotation); SAT on this OBB is
    equivalent to per-axis projection on the AABB.
    """
    lo, hi = aabb
    center = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
    half = (0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2]))
    R_cols = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return (center, R_cols, half)


def _scene_entity_part_refs(ir, catalog, name: str):
    """`[(instance, part), ...]` for a scene entity, or None if it has none.

    The `(instance, part)` spelling `_exact_adjudicator.exact_part_gap` takes.
    A group or the room returns None — a group is a union of instances rather
    than a placed body, and the room owns no geometry at all.
    """
    obbs = _scene_entity_obbs(ir, catalog, name)
    if not obbs:
        return None
    inst = name.partition(".")[0]
    return [(inst, part) for part, _obb in obbs]


def _scene_pair_exact(ir, catalog, a: str, b: str, *, poses, base_deltas,
                      contact_tol, run_dir, asset_base_dir, cache):
    """Exact verdict for a scene entity pair: `(collided, gap, exact)`.

    **Every part pair across the two entities is sent, not just the deepest
    box one.** An instance is many bodies, and the deepest box overlap can be a
    false positive around a bore while a shallower pair is real — the ordering
    error `penetration_pairs` shipped with, and reintroducing it here by
    adjudicating only `_worst_obb_pair`'s winner would be the same bug in a new
    place. `exact_part_gap` returns the minimum over what it is given, which is
    the instance-level answer.

    `(None, None, False)` when nothing could answer, and the caller keeps its
    box verdict.
    """
    refs_a = _scene_entity_part_refs(ir, catalog, a)
    refs_b = _scene_entity_part_refs(ir, catalog, b)
    if not refs_a or not refs_b:
        return None, None, False        # a group or the room: no exact form
    from ..verify import _exact_adjudicator as _adj
    pairs = [(ia, pa, ib, pb, poses, base_deltas)
             for ia, pa in refs_a for ib, pb in refs_b
             if (ia, pa) != (ib, pb)]
    if not pairs:
        return None, None, False
    gap, stats = _adj.exact_part_gap(
        ir, catalog, pairs, run_dir=run_dir, asset_base_dir=asset_base_dir,
        cache=cache)
    # `not_adjudicable` matters as much as a None gap: the minimum over the
    # pairs that *did* answer is not the minimum over all of them, so a
    # partial answer would overstate the gap and clear a real contact.
    if gap is None or stats.get("not_adjudicable"):
        return None, None, False
    return gap <= float(contact_tol), gap, True


def scene_pair_report(ir, catalog, a: str, b: str, *,
                      contact_tol: float = 1e-6,
                      poses=None, world_poses=None, base_deltas=None,
                      exact: bool = True, run_dir=None, asset_base_dir=None,
                      cache=None) -> dict:
    """Scene-layer pair_report (instances / groups / parts / room).

    **`measure` says which geometry answered, and it is the field to read
    first.** `"obb"` means the oriented boxes of the two entities' parts, the
    same ones `penetration_pairs` tests, reduced to the deepest part pair —
    `a_sub` / `b_sub` name it. `"aabb"` means axis-aligned world boxes, which is
    all a **group** or the **room** has: a group is a union of instances and the
    room is the world frame, so neither carries one orientation.

    Instance pairs use each part's own orientation, so this agrees with
    `penetration_pairs` about the same pair: two thin rods turned 45 degrees and
    offset across their long axes overlap as axis-aligned boxes but not as
    oriented ones.

    **`collided` is a surface verdict, and `exact_gap` is the measurement behind
    it.** An OBB is still a box — it bounds a round leg as a square prism and a
    shelled carcass as its full outer box — so a box-derived `collided` would
    claim contact that `NoCrossInstancePenetration` clears. When the boxes claim
    contact the pair is measured with the same narrow phase the verifier uses. `signed_overlap` and `center_delta` remain exact box algebra, which is
    what they are for, and `signed_distance` stays the box quantity.

    A pair the boxes separate is not queried and needs no query: a box contains
    its part, so box-separated is truly separated. `exact_gap is None` with
    `collided=True` means nothing could answer — a group, the room, or a part
    whose collision mesh is not staged — and the conservative box verdict
    stands.

    Nearly free at this layer: the narrow phase is the in-process mesh backend,
    measured at 1-19 ms. `exact=False` skips it.
    """
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    aabb_a = _scene_entity_world_aabb(ir, catalog, a, poses=poses,
                                      base_deltas=base_deltas)
    aabb_b = _scene_entity_world_aabb(ir, catalog, b, poses=poses,
                                      base_deltas=base_deltas)
    obbs_a = _scene_entity_obbs(ir, catalog, a, poses, base_deltas=base_deltas)
    obbs_b = _scene_entity_obbs(ir, catalog, b, poses, base_deltas=base_deltas)
    a_sub = b_sub = None
    if obbs_a and obbs_b:
        # One pair, so descend unconditionally: the agent named these two
        # entities, and P_a x P_b SAT calls on a single pair is nothing next to
        # answering the wrong question.
        sat, a_sub, b_sub = _worst_obb_pair(obbs_a, obbs_b)
        measure = "obb"
    else:
        sat = _ok.sat_obb_obb(_aabb_to_obb(aabb_a), _aabb_to_obb(aabb_b))
        measure = "aabb"
    overlap = _signed_overlap_dict(aabb_a, aabb_b)
    collided = sat.signed_distance < -float(contact_tol)
    exact_gap, answered = None, False
    if collided and exact and a != b:
        verdict, exact_gap, answered = _scene_pair_exact(
            ir, catalog, a, b, poses=poses, base_deltas=base_deltas,
            contact_tol=contact_tol, run_dir=run_dir,
            asset_base_dir=asset_base_dir, cache=cache)
        if answered:
            collided = verdict
    return {
        "a": a,
        "b": b,
        "a_sub": a_sub,
        "b_sub": b_sub,
        "measure": measure,
        "collided": collided,
        "exact": answered,
        "exact_gap": exact_gap,
        "contact_tol": float(contact_tol),
        "signed_distance": sat.signed_distance,
        "contact_normal": tuple(sat.axis),
        "contact_point": tuple(sat.contact_point) if sat.contact_point is not None else None,
        "signed_overlap": overlap,
        "center_delta": _vsub(_aabb_center(aabb_b), _aabb_center(aabb_a)),
    }


def scene_within_report(ir, catalog, inner: str, outer: str, *,
                        axes="xyz",
                        poses=None, world_poses=None, base_deltas=None) -> dict:
    """Scene-layer within_report."""
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    axes_t = _normalize_axes(axes)
    inner_aabb = _scene_entity_world_aabb(ir, catalog, inner, poses=poses,
                                          base_deltas=base_deltas)
    outer_aabb = _scene_entity_world_aabb(ir, catalog, outer, poses=poses,
                                          base_deltas=base_deltas)
    margins: dict = {}
    overall = True
    for ax in axes_t:
        i = _AXIS_INDEX[ax]
        lower = inner_aabb[0][i] - outer_aabb[0][i]
        upper = outer_aabb[1][i] - inner_aabb[1][i]
        ax_within = (lower >= 0.0) and (upper >= 0.0)
        margins[ax] = {
            "lower_margin": lower,
            "upper_margin": upper,
            "within": ax_within,
        }
        if not ax_within:
            overall = False
    return {
        "inner": inner,
        "outer": outer,
        "axes": list(axes_t),
        "within": overall,
        "margins": margins,
    }


def _scene_pair_entry(ir, catalog, a: str, b: str, *, contact_tol: float, poses,
                      base_deltas=None):
    """Compact per-pair entry for the ranked search probes.

    **The world AABB decides whether it is worth descending to the OBBs.** An
    entity's world box is the union of its parts' boxes and each of those
    contains that part's oriented box, so a pair the AABBs separate is genuinely
    separated and `collided=False` needs no further work — only the reported
    margin is a lower bound on the true gap, which understates the separation
    and so ranks the pair as nearer than it is. That is the safe direction for a
    "what is close to this" list.

    A pair the AABBs call *colliding* is the one that might be a box artifact,
    and that is where the OBBs are worth the `P_a x P_b` SAT calls. Ranking
    hundreds of distant pairs at part granularity would cost tens of thousands
    of them for an answer the boxes already settle.
    """
    aabb_a = _scene_entity_world_aabb(ir, catalog, a, poses=poses,
                                      base_deltas=base_deltas)
    aabb_b = _scene_entity_world_aabb(ir, catalog, b, poses=poses,
                                      base_deltas=base_deltas)
    sat = _ok.sat_obb_obb(_aabb_to_obb(aabb_a), _aabb_to_obb(aabb_b))
    measure = "aabb"
    if sat.signed_distance < -float(contact_tol):
        obbs_a = _scene_entity_obbs(ir, catalog, a, poses,
                                    base_deltas=base_deltas)
        obbs_b = _scene_entity_obbs(ir, catalog, b, poses,
                                    base_deltas=base_deltas)
        if obbs_a and obbs_b:
            sat = _worst_obb_pair(obbs_a, obbs_b)[0]
            measure = "obb"
    return {
        "a": a,
        "b": b,
        "measure": measure,
        "collided": sat.signed_distance < -float(contact_tol),
        # Uniform shape — see `_object_pair_entry` for what these mean on a row
        # that was never measured.
        "exact": False,
        "exact_gap": None,
        "signed_distance": sat.signed_distance,
        "signed_overlap": _signed_overlap_dict(aabb_a, aabb_b),
    }


def _adjudicate_scene_rows(ir, catalog, entries, *, contact_tol, poses,
                           base_deltas, run_dir, asset_base_dir, cache):
    """Correct `collided` on the rows a ranked scene probe is about to return.

    Same shape as `_adjudicate_object_rows`, and near-free here: this layer's
    narrow phase is the in-process mesh backend.

    Two passes over the rows: one batched call to warm the shared cache, then a
    per-row read that hits it. `exact_part_gap` reduces to a minimum, so it
    cannot report per row — the same reason `_support._exact_scene_contacts`
    uses this idiom.
    """
    rows = [e for e in entries if e["collided"] and e["a"] != e["b"]]
    if not rows:
        return entries
    from ..verify import _exact_adjudicator as _adj
    warm = [(ia, pa, ib, pb, poses, base_deltas)
            for e in rows
            for ia, pa in (_scene_entity_part_refs(ir, catalog, e["a"]) or ())
            for ib, pb in (_scene_entity_part_refs(ir, catalog, e["b"]) or ())
            if (ia, pa) != (ib, pb)]
    if warm:
        _adj.exact_part_gap(ir, catalog, warm, run_dir=run_dir,
                            asset_base_dir=asset_base_dir, cache=cache)
    for entry in rows:
        verdict, gap, answered = _scene_pair_exact(
            ir, catalog, entry["a"], entry["b"], poses=poses,
            base_deltas=base_deltas, contact_tol=contact_tol, run_dir=run_dir,
            asset_base_dir=asset_base_dir, cache=cache)
        entry["exact_gap"] = gap
        entry["exact"] = answered
        if answered:
            entry["collided"] = verdict
    return entries


def scene_nearest_neighbors(ir, catalog, name: str, *,
                            candidates=None,
                            limit: int = 5,
                            contact_tol: float = 1e-6,
                            poses=None, world_poses=None,
                            base_deltas=None, exact: bool = True,
                            run_dir=None, asset_base_dir=None,
                            cache=None) -> list:
    """Scene-layer nearest_neighbors over instances (default pool).

    A bounding-volume triage whose contact claim is measured, like its
    object-layer twin: the ranking is sound because a box bounds its part, and
    the returned rows that claim contact are re-checked against real geometry
    (`exact_gap`, `exact`). `measure` says which volume produced the ranking —
    see `_scene_pair_entry`. `gap` remains the way to get a number for a pair
    the ranking has separated.
    """
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    if candidates is None:
        candidates = [i["name"] for i in ir.get("instances", [])]
    pool = [c for c in candidates if c != name]
    entries = [_scene_pair_entry(ir, catalog, name, c,
                                  contact_tol=contact_tol, poses=poses,
                                  base_deltas=base_deltas)
               for c in pool]
    entries.sort(key=_rank_key)
    out = entries[: max(0, int(limit))]
    if not exact:
        return out
    return _adjudicate_scene_rows(
        ir, catalog, out, contact_tol=contact_tol, poses=poses,
        base_deltas=base_deltas, run_dir=run_dir,
        asset_base_dir=asset_base_dir, cache=cache)


def scene_find_clearance_risks(ir, catalog, *,
                               instances=None,
                               limit: int = 10,
                               contact_tol: float = 1e-6,
                               poses=None, world_poses=None,
                               base_deltas=None, exact: bool = True,
                               run_dir=None, asset_base_dir=None,
                               cache=None) -> list:
    """Scene-layer find_clearance_risks over instance pairs.

    Same reading as `scene_nearest_neighbors`: a sound bounding-box ranking
    over every pair, with the returned rows' contact claim measured.
    """
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    pool = ([i["name"] for i in ir.get("instances", [])]
            if instances is None else list(instances))
    entries = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            entries.append(_scene_pair_entry(
                ir, catalog, pool[i], pool[j],
                contact_tol=contact_tol, poses=poses,
                base_deltas=base_deltas))
    entries.sort(key=_rank_key)
    out = entries[: max(0, int(limit))]
    if not exact:
        return out
    return _adjudicate_scene_rows(
        ir, catalog, out, contact_tol=contact_tol, poses=poses,
        base_deltas=base_deltas, run_dir=run_dir,
        asset_base_dir=asset_base_dir, cache=cache)


# ──────────────────────────────────────────────────────────────────
# Scene-layer assertion-mirror probes
# ──────────────────────────────────────────────────────────────────


def _scene_anchor_world(ir, catalog, anchor_tuple, poses, base_deltas=None):
    """Resolve `(entity, anchor)` tuple at given pose for scene layer."""
    entity, anchor = anchor_tuple
    return _sk.anchor_position(ir, catalog, entity, anchor, poses=poses,
                               base_deltas=base_deltas)


def scene_distance_between(ir, catalog, a, b, *, axis=None, poses=None,
                           world_poses=None, base_deltas=None) -> float:
    """Distance between two scene-layer anchors.

    `a` and `b` are `(entity, anchor)` tuples. Entity may be an instance,
    group, or the room. For part-scoped anchors use the anchor-name
    form `"<part>.<anchor>"` (see `scene_anchor_position`).
    """
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    p_a = _scene_anchor_world(ir, catalog, a, poses, base_deltas)
    p_b = _scene_anchor_world(ir, catalog, b, poses, base_deltas)
    return eval_distance_scalar(p_a, p_b, axis=_axis_arg(axis))


def scene_overlap_along(ir, catalog, a: str, b: str, *,
                        axis, poses=None, world_poses=None,
                        base_deltas=None) -> float:
    """Scene-layer signed projection overlap on a unit axis.

    `a` and `b` are entity names (instance / group / room / part-scoped
    via `"<instance>.<part>"`).
    """
    poses, base_deltas = _pose_kwargs(ir, catalog, poses, world_poses,
                                      base_deltas)
    aabb_a = _scene_entity_world_aabb(ir, catalog, a, poses=poses,
                                      base_deltas=base_deltas)
    aabb_b = _scene_entity_world_aabb(ir, catalog, b, poses=poses,
                                      base_deltas=base_deltas)
    return eval_overlap_scalar(aabb_a, aabb_b, _axis_arg(axis))


def scene_angle_between(ir, catalog, dir_a, dir_b) -> float:
    """Scene-layer angle (degrees) between two world-frame directions.

    No `poses=` — see `object_angle_between`.
    """
    return eval_angle_scalar(tuple(dir_a), tuple(dir_b))


def scene_motion_displacement(ir, catalog, entity: str, *,
                               anchor: str = "center",
                               from_pose=None,
                               to_pose,
                               axis=None) -> float:
    """Scene-layer displacement of an anchor between two poses.

    Pose keys follow scene convention: bare `"<joint>"` applies to any
    instance with that joint; qualified `"<instance>.<joint>"` targets
    a specific instance.
    """
    from_pose_i = _scene_poses_to_internal(ir, catalog, from_pose or None)
    to_pose_i = _scene_poses_to_internal(ir, catalog, to_pose)
    p_from = _sk.anchor_position(ir, catalog, entity, anchor, poses=from_pose_i)
    p_to = _sk.anchor_position(ir, catalog, entity, anchor, poses=to_pose_i)
    return eval_distance_scalar(p_from, p_to, axis=_axis_arg(axis))


# ──────────────────────────────────────────────────────────────────
# Scene-layer verifier mirrors
# ──────────────────────────────────────────────────────────────────


def scene_support_report(ir, catalog, *, asset_base_dir=None,
                         cache=None) -> dict:
    """Mirror of scene-layer `SupportConnected.evidence`.

    Returns the contact graph + grounding (room face contacts) +
    reachability, plus `exact_contact` saying what decided the edges.
    Rest-pose only (matches the verifier).

    Contact is measured on surfaces, so this answers the same question the
    verifier does — including the contacts a coplanar-box test cannot see, like
    a curved body resting on a tilted one. `asset_base_dir` and `cache` are what
    let that pass run; without them every pair falls back to its bounding box.

    One graph build, for the reason `object_support_report` gives.
    """
    from ..verify._support import (
        build_scene_adjacency, contact_tol_m, _bfs_from_floor,
    )
    tol_f = contact_tol_m()
    adj = build_scene_adjacency(ir, catalog, _sk, tol_f,
                                asset_base_dir=asset_base_dir, cache=cache)
    reached = _bfs_from_floor(adj)
    inst_names = [i["name"] for i in ir.get("instances", [])]
    contacts = []
    seen = set()
    for a, neighbors in adj.edges.items():
        if a == "__floor__":
            continue
        for b in neighbors:
            if b == "__floor__":
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            contacts.append(key)
    grounded = sorted(adj.edges.get("__floor__", set()))
    unsupported = [n for n in inst_names if n not in reached]
    return {
        "is_supported": not unsupported,
        "tol_m": tol_f,
        "contacts": sorted(contacts),
        "grounded": grounded,
        "supported": sorted(n for n in inst_names if n in reached),
        "unsupported": unsupported,
        "exact_contact": adj.exact,
    }


def scene_room_containment_margins(ir, catalog, instance: str, *,
                                    axes="xyz") -> dict:
    """Per-axis containment margins of an instance inside the room.

    Equivalent to `scene_within_report(ir, catalog, instance, '<room>',
    axes='xyz')` but conventionally named to match the `RoomContainment`
    verifier's intent.
    """
    return scene_within_report(ir, catalog, instance,
                               _sk._ROOM_RESERVED_NAME, axes=axes)


# ---------------------------------------------------------------------------
# The scene-layer probe surface
#
# Written out by name rather than assembled with `getattr(_probes, f"scene_{n}")`
# at the call site. The dynamic form would make every one of these functions
# read as unreferenced to static analysis, and a dead-code sweep could delete
# the scene agent's entire probe surface on that evidence. Naming them here is
# the same dispatch with the reference made visible.
#
# The key is the name the agent sees inside `probe`; the `scene_` prefix
# exists only to keep the object- and scene-layer implementations distinct in
# this module's namespace.
# ---------------------------------------------------------------------------

SCENE_PROBES = {
    "list_instances": scene_list_instances,
    "list_groups": scene_list_groups,
    "list_assertions": scene_list_assertions,
    "instance_pose": scene_instance_pose,
    "group_pose": scene_group_pose,
    "instance_bbox": scene_instance_bbox,
    "group_bbox": scene_group_bbox,
    "scene_bbox": scene_scene_bbox,
    "anchor_position": scene_anchor_position,
    # The orientation half of the same question. A declared semantic anchor is
    # a frame, and `anchor_position` answers about its origin only — so an
    # attachment aimed the wrong way had no query that could see it.
    "anchor_pose": scene_anchor_pose,
    "outward": scene_outward,
    "ir_snapshot": scene_ir_snapshot,
    "resolve_entity_aabb": scene_resolve_entity_aabb,
    "resolve_direction": scene_resolve_direction,
    "instance_world_aabb": scene_instance_world_aabb,
    "group_world_aabb": scene_group_world_aabb,
    "room_world_aabb": scene_room_world_aabb,
    "within_report": scene_within_report,
    "distance_between": scene_distance_between,
    "overlap_along": scene_overlap_along,
    "angle_between": scene_angle_between,
    "motion_displacement": scene_motion_displacement,
    "room_containment_margins": scene_room_containment_margins,
    # The one query that reports a joint's range, in the units `poses=` is
    # written in. The catalog IR holds the same range in radians.
    "joint_limits": scene_joint_limits,
    # Declared anchors were resolvable by name long before anything could tell
    # you the names. Both registries, for the same reason `joint_limits` is in
    # both: any layer that can write `<part>.<anchor>` needs to see what exists.
    "list_anchors": scene_list_anchors,
}
#: `penetration_pairs`, `pair_report`, `nearest_neighbors`,
#: `find_clearance_risks`, `support_report`, `gap`, and `room_gap` are
#: deliberately absent: each reaches the exact narrow phase and so needs the
#: adjudicator's
#: asset root, run dir and shared cache, which an agent cannot pass and a plain
#: registry entry cannot carry. `agent/layers/layer.py` binds them by
#: hand and lists them in `_SCENE_EXACT_PROBES`. A registry exists to be
#: bulk-bound; a name that cannot be does not belong in one.


# ──────────────────────────────────────────────────────────────────
# Scene probes that reach the exact narrow phase (bound by hand in
# `agent/layers/layer.py`, see `_SCENE_EXACT_PROBES`)
# ──────────────────────────────────────────────────────────────────


def scene_room_gap(ir, catalog, instance: str, poses=None, *,
                   faces="all", tol: float = 0.005, world_poses=None,
                   base_deltas=None,
                   mount_deltas=None, deform_bounds=None,
                   asset_base_dir=None, cache=None):
    """Exact surface gap between an instance and each room face, in meters.

    `{face: {"gap", "exact", "part", "axis", "side"}}` over `faces` — a face
    name, an iterable of them, or `"all"`. Names are `floor`, `ceiling`, and the
    four walls under the viewer-relative convention (`+X` left, `+Y` front):
    `left`, `right`, `front`, `back`. `gap` is 0.0 when material is outside the
    face, and `part` names the nearest part to it.

    **Scope it to a part when a part is the question.** `instance` takes
    `"oven"` or `"oven.door"`, the same spelling anchors use. The unscoped form
    reduces over every part, so an oven standing on the floor answers 0.0 at
    every door angle — true, and useless for "how much room has the door left".

    The counterpart to `gap` for the one "instance" that is not an instance.
    `room_containment_margins` answers the same question on **bounding boxes**,
    and for a part that rotates those are a different number: a door panel whose
    box dips 22 mm below the floor at 90 degrees is not touching the floor, and a
    run that trusted the box gave away 35 degrees of the door's travel. Use this
    when the answer decides an authored motion; the box is fine for a first look.

    `exact=False` means no boolean answered — nothing was close enough to be
    worth a CAD query, a part's collision mesh is not staged, or a query failed —
    and the number is then the world-AABB lower bound, i.e. a bound and not a
    measurement. Check it before acting on a small value.

    `poses` is in **surface units** — degrees for a revolute joint, meters for a
    prismatic one — like every other pose-taking probe on this surface, and
    `world_poses` says where the bodies are, so "does the moved body clear the
    floor" is askable here too.

    Cost: a face whose bound already exceeds `tol` sends no query, so a
    normally-placed instance pays for the one or two faces it is near.
    """
    from ..verify import _exact_adjudicator as _adj
    # The conversion belongs here because this is the agent boundary.
    # Everything below is the kernel's radians domain: `poses` reaches
    # `_sk.instance_part_world_aabbs` for the bounds and `_exact_face_gap` for
    # the queries, and a probe range-checks nothing, so a revolute value
    # forwarded raw is 57x out and answered without complaint.
    #
    # `animation_room_gap_at` therefore does NOT come through here — it holds
    # frame poses that are already internal.
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _adj.exact_room_gap(
        ir, catalog, instance, pi, bd,
        faces=faces, tol=tol,
        mount_deltas=mount_deltas, deform_bounds=deform_bounds,
        asset_base_dir=asset_base_dir, cache=cache)


def _gap_candidates(scene_ir, catalog, a, b, poses, deltas, tol) -> tuple:
    """`(aabb_lower_bound, ambiguous_part_pairs)` between two placed instances.

    A pair whose boxes separate by more than `tol` is dropped with no CAD
    query: its true gap exceeds `tol` too, because each box contains its part.
    """
    pa = _sk.instance_part_world_aabbs(scene_ir, catalog, a, poses=poses,
                                        base_deltas=deltas)
    pb = _sk.instance_part_world_aabbs(scene_ir, catalog, b, poses=poses,
                                        base_deltas=deltas)
    lower = float("inf")
    cands = []
    for na, (alo, ahi) in pa:
        for nb, (blo, bhi) in pb:
            sep = 0.0
            for k in range(3):
                d = max(blo[k] - ahi[k], alo[k] - bhi[k])
                if d > 0:
                    sep = max(sep, d)
            lower = min(lower, sep)
            if sep <= tol:
                cands.append((a, na, b, nb, poses, deltas, None, None))
    return (0.0 if lower == float("inf") else lower), cands


def _instance_gap(scene_ir, catalog, a, b, poses, deltas, tol, ctx) -> tuple:
    """`(gap, exact)`: the surface gap between two rigidly placed instances.

    A world-AABB gap is only a *lower* bound on the true surface gap, and a
    loose one once anything rotates: a gripper leaning 20 degrees reported a
    30 mm overlap on a grasp the exact narrow phase clears. So the AABB is the
    broad phase only, and the ambiguous part pairs go to exact geometry.

    `exact=False` means `gap` is that AABB lower bound with no known upper
    bound: no pair was ambiguous, or the CAD phase could not answer every one
    (an asset whose collision mesh is not staged), in which case the minimum
    over the rest might not be the minimum at all.
    """
    lower, cands = _gap_candidates(scene_ir, catalog, a, b, poses, deltas, tol)
    if not cands:
        return lower, False
    from ..verify import _exact_adjudicator as _adj
    gap, stats = _adj.exact_part_gap(
        scene_ir, catalog, cands, run_dir=None,
        asset_base_dir=getattr(ctx, "asset_base_dir", None),
        cache=getattr(ctx, "adjudication_cache", None))
    if gap is None or stats.get("not_adjudicable"):
        return lower, False
    return gap, True


def scene_gap(ir, catalog, a: str, b: str, *, poses=None, world_poses=None,
              base_deltas=None, tol: float = 0.005, asset_base_dir=None,
              cache=None):
    """Exact surface gap between two instances at any pose, in meters.

    The same measurement the verifiers use, at a pose you have not
    authored — `poses=` for the joints, `world_poses=` for the bodies. Returns
    `(gap_m, exact)`. No timeline required, so this is the query to sweep:

        probe("[(x, gap('mug', 'table', tol=0.05,"
              "        world_poses={'mug': (x, 0.14, 0.78)})) "
              "for x in [0.30, 0.35, 0.40]]")

    `penetration_pairs` answers over *every* pair at once and is what to reach
    for when you do not yet know what a body will hit. It reaches the same
    narrow phase, so its rows are real overlaps — but it says nothing about how
    far apart the pairs it does *not* report are. Come here to settle a
    number.

    `tol` is the ambiguity band, not a threshold: a pair the boxes separate by
    more than it is dropped without a CAD query, so the answer is exact at or
    below `tol` and a lower bound above it. Widen it to get a real number
    further out; it costs CAD queries.

    Both bodies are their rest solids placed by SE3, which is what a pose
    hypothesis is.
    """
    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.asset_base_dir = asset_base_dir
    ctx.adjudication_cache = cache if cache is not None else {}
    pi, bd = _pose_kwargs(ir, catalog, poses, world_poses, base_deltas)
    return _instance_gap(ir, catalog, a, b, pi or None, bd, float(tol), ctx)


# ---------------------------------------------------------------------------
# Part-vs-part collision inside one instance
# ---------------------------------------------------------------------------


def _self_penetrations_internal(obj_ir, obj_poses, *, exact, asset_base_dir,
                                cache, mount_deltas=None, deform_bounds=None):
    """Self-penetrating part pairs at a pose already in **internal** units.

    The units boundary sits at the two public entry points, not here. Before
    this split, the frame-indexed probe held internal values, converted them to
    surface purely so it could reuse the scene entry point, and that entry point
    converted them straight back — a round trip whose only purpose was to share
    a function.

    `mount_deltas` / `deform_bounds` are the animation layer's deformation
    channels and default to absent, which is the scene-layer entry point's case.
    A pair naming a part that is deforming is dropped from the exact phase and
    marked `deformed: True` rather than adjudicated: the adjudicator would
    intersect its rest solid and, only ever downgrading, would clear it. A
    probe and the verifier it predicts must not answer differently.
    """
    from . import _object_kernel as _ok

    pairs = list(_ok.penetration_pairs(obj_ir, obj_poses,
                                       mount_deltas=mount_deltas or None,
                                       deform_bounds=deform_bounds or None))
    deformed = set(deform_bounds or ())
    rigid, soft = [], []
    for p in pairs:
        (soft if (p.a in deformed or p.b in deformed) else rigid).append(p)
    from ..verify import _exact_adjudicator as _adj

    if rigid and exact:
        rigid, _stats = _adj.adjudicate_object_pairs(
            obj_ir, [(p, obj_poses, mount_deltas or None) for p in rigid],
            run_dir=asset_base_dir, cache=cache)
    # `reported_depth`, not `p.depth`: the latter is SAT's oriented-box number,
    # and this probe is what an agent plans against, so it has to report what
    # `NoPenetration` will. The adjudication above is what populates the signed
    # `exact_distance` that function prefers, so with `exact=False` there is
    # nothing to prefer and the box number stands — same rule as the pairs
    # themselves, which are then SAT's conservative set.
    out = [{"a": p.a, "b": p.b,
            "depth_m": round(_adj.reported_depth(p), 6)} for p in rigid]
    out += [{"a": p.a, "b": p.b, "depth_m": round(p.depth, 6),
             "deformed": True,
             "measure": "inflated bounding boxes; DeformedNoPenetration "
                        "measures the real mesh"}
            for p in soft]
    return out


_REFERENCE_TRACKS: dict = {}


