"""Object-layer verifiers.

The suite (7 verifiers — see `OBJECT_VERIFIERS` at the bottom of this
module): NoSelfPenetration, MinPartDimension, JointRangeNonTrivial,
JointOriginNearGeometry, JointOriginNearBothBodies, SupportConnected,
ObjectRelationalAssertions.

Each verifier:
  * exposes a module-level NAME constant equal to its dotted run-function name
    (used as the verifier identity in cache keys, trace events, and reports);
  * exposes a run(ir, ctx) function. The ctx argument is accepted for
    uniformity with scene-layer verifiers; object verifiers do not read it.
"""

from __future__ import annotations

from ..sdk import object_query
from ..sdk import _primitives as _prim
from .report import VerifierReport, not_applicable
from ._relational import evaluate_assertions
from . import _ccd_sampler as _ccd
from ._ccd_sampler import all_samples, per_joint_only_default
from ._support import support_connected_object


# ---------------------------------------------------------------------------
# NoSelfPenetration
# ---------------------------------------------------------------------------

NO_SELF_PENETRATION = "NoSelfPenetration"


def _frame_to_json(frame):
    """Convert a (translation, euler_deg) frame into JSON-friendly lists."""
    t, e = frame
    return {"translation": [t[0], t[1], t[2]],
            "rotation_euler_deg": [e[0], e[1], e[2]]}


def _pair_evidence(pair) -> dict:
    """Reshape a PenetrationPair into the JSON evidence dict."""
    # No SAT quantity goes out. `depth` / `contact_normal` / `contact_point` /
    # `signed_distance` are all measured on the enclosing boxes, so depth
    # overstates a 40-degree bar through a block as 61 mm against a 33 cm^3 real
    # overlap. What goes out instead is measured on the parts: `exact_volume_cm3`
    # (real shared material) and `closest_points` (the true contact location)
    # for every adjudicated pair, and `separating_shift_m`, a translation tested
    # to clear the pair, where one exists.
    out = {
        "a": pair.a,
        "b": pair.b,
        "a_frame": _frame_to_json(pair.a_frame),
        "b_frame": _frame_to_json(pair.b_frame),
        "a_bbox": [list(pair.a_bbox[0]), list(pair.a_bbox[1])],
        "b_bbox": [list(pair.b_bbox[0]), list(pair.b_bbox[1])],
    }
    # Exact-geometry confirmation, when the adjudicator ran on this pair. Only
    # emitted when present so a run without a CAD interpreter produces the same
    # evidence shape it always did.
    if getattr(pair, "adjudicated", False):
        out["adjudicated"] = True
        if pair.exact_volume is not None:
            # cm^3, NOT m^3. Evidence goes through `round_floats`,
            # the evidence display rounding meant for lengths — every real
            # interference volume (1e-5..1e-4 m^3) rounds to a flat 0.0 there,
            # which reads as "no overlap": the exact opposite of what this row
            # means. In cm^3 the same rounding gives 1 mm^3 resolution, which is
            # the right scale for telling a hairline sliver from a real clash.
            out["exact_volume_cm3"] = pair.exact_volume * 1.0e6
        if getattr(pair, "separating_shift", None) is not None:
            # A translation that was tested, not inferred. Absent when no
            # single-axis move cleared the pair, which says the fix is a
            # reshape rather than a nudge.
            out["separating_shift_m"] = list(pair.separating_shift)
        if pair.closest_points is not None:
            out["closest_points"] = [list(pair.closest_points[0]),
                                     list(pair.closest_points[1])]
    return out


def _allowance_keys(ir: dict) -> set:
    """Build a set of frozensets of (a, b) part names allowed to overlap.

    Allowances are symmetric, so we key by `frozenset({a, b})`; the
    SAT-emitted PenetrationPair has `(a, b)` ordered lexicographically
    but both orientations resolve to the same key here.
    """
    keys = set()
    for rec in ir.get("collision_allowances", ()):
        keys.add(frozenset((rec["a"], rec["b"])))
    return keys


from ._allowance_rules import _allowance_index, _allowance_matches


#: Adjudications per part pair. A pair SAT-flags across a contiguous band of
#: poses, and adjudicating every pose in that band (milliseconds each) would
#: make refining the sweep multiply the *expensive* phase, though the sweep
#: itself costs a small fraction of that a pose. Keeping a spread of the band plus its
#: deepest pose bounds that cost independently of sampling density, which is
#: what lets the step follow the geometry.
#:
#: A spread rather than only the deepest, for the reason `no_self_penetration`
#: already documents: the deepest SAT overlap can be a bounding-box false
#: positive around a bore while a shallower pose in the same band is a real
#: B-rep penetration. Eight preserves that argument at a fixed price.
_MAX_ADJUDICATIONS_PER_PAIR = 8


def _thinnest_part_extent(ir: dict) -> float:
    """The smallest AABB extent of any part, in meters — the feature size a
    sweep must not step over.

    An obstacle is missed when the swept point travels further between
    samples than the obstacle is thick, so the thinnest part in the object
    sets the resolution the whole sweep needs. Taken over parts rather than
    per pair because any part can be the obstacle for any sweep.

    Returns 0.0 when nothing has a usable box, which the sampler reads as
    "no geometry to hand" and answers with the coarse default.
    """
    best = 0.0
    for part in ir.get("parts") or []:
        box = _part_local_aabb(part)
        if box is None:
            continue
        lo, hi = box
        ext = min(abs(float(hi[i]) - float(lo[i])) for i in range(3))
        if ext > 0.0 and (best == 0.0 or ext < best):
            best = ext
    return best


def _joint_sweep_radius(ir: dict, joint: dict) -> float:
    """How far the child's furthest corner sits from this joint's pivot.

    The lever arm that turns an angular step into swept distance. Measured
    to the child's local AABB corners from the pivot's position in the same
    frame, which is what `parent_anchor`/`child_anchor` translations are
    expressed in.
    """
    child = joint.get("child")
    for part in ir.get("parts") or []:
        if part.get("name") != child:
            continue
        box = _part_local_aabb(part)
        if box is None:
            return 0.0
        lo, hi = box
        anchor = (joint.get("child_anchor") or {}).get("translation") \
            or [0.0, 0.0, 0.0]
        worst = 0.0
        for i in range(8):
            corner = (float(hi[0]) if i & 1 else float(lo[0]),
                      float(hi[1]) if i & 2 else float(lo[1]),
                      float(hi[2]) if i & 4 else float(lo[2]))
            d = sum((corner[k] - float(anchor[k])) ** 2 for k in range(3))
            worst = max(worst, d)
        return worst ** 0.5
    return 0.0


def _sweep_steps(ir: dict) -> dict:
    """Per-joint sampling steps derived from the object's own geometry.

    Empty when no part carries a box, so the sampler keeps its defaults.
    """
    feature = _thinnest_part_extent(ir)
    if feature <= 0.0:
        return {}
    steps: dict = {}
    for joint in ir.get("joints") or []:
        jtype = joint.get("type")
        if jtype == "revolute":
            steps[joint["name"]] = _ccd.adaptive_revolute_step_rad(
                feature, _joint_sweep_radius(ir, joint))
        elif jtype == "prismatic":
            steps[joint["name"]] = _ccd.adaptive_prismatic_step_m(feature)
    return steps


def _bounded_candidates(sat_candidates: list) -> tuple:
    """At most `_MAX_ADJUDICATIONS_PER_PAIR` candidates per part pair.

    Keeps the deepest (most negative `signed_distance`) plus an even spread
    over the rest in the order they were swept, so the band's shallow ends
    survive alongside its middle. Returns `(kept, n_dropped)`.
    """
    by_pair: dict = {}
    for item in sat_candidates:
        by_pair.setdefault(frozenset((item[0].a, item[0].b)), []).append(item)
    kept: list = []
    dropped = 0
    for _, items in sorted(by_pair.items(), key=lambda kv: sorted(kv[0])):
        if len(items) <= _MAX_ADJUDICATIONS_PER_PAIR:
            kept.extend(items)
            continue
        deepest = min(range(len(items)),
                      key=lambda i: items[i][0].signed_distance)
        picks = {deepest}
        n_spread = _MAX_ADJUDICATIONS_PER_PAIR - 1
        for k in range(n_spread):
            picks.add(round(k * (len(items) - 1) / max(n_spread - 1, 1)))
        kept.extend(items[i] for i in sorted(picks))
        dropped += len(items) - len(picks)
    return kept, dropped


def _adjudication_repairs(adjudication, rows=()) -> list:
    """What to say about these rows, given whether exact geometry judged them.

    "Every row here was re-checked against the parts' TRUE surfaces and really
    does overlap" is true only of rows exact geometry judged. Made of rows it
    did not judge, it tells an author that box extents (hundreds of
    millimeters, on a large part) are overlaps to size a fix from.

    Branching on `available` alone is not enough: `available` says only that
    the adjudicator could be *reached*, and every pair can still come back
    `not_adjudicable` under it (`{"considered": 7, "sent": 0,
    "not_adjudicable": 7, "available": true}`, which a run with no `run_dir`
    produces, since `geometry_spec` cannot resolve a relative
    `assets/brep/...` path). The author would then be sent to an
    `exact_volume_cm3` those rows do not carry.

    So the test is the per-pair `adjudicated` flag on the rows actually being
    reported: the claim is made only if every reported row carries it.
    """
    reported = [r for r in rows if isinstance(r, dict)]
    ran = bool(isinstance(adjudication, dict)
               and adjudication.get("available")
               and reported
               and all(r.get("adjudicated") for r in reported))
    if ran:
        return [
            "Every row was checked against the true surfaces and really does overlap. "
            "`exact_volume_cm3` measures shared material and `closest_points` locates "
            "it. `separating_shift_m`, when present, is a translation of `b` tested to "
            "clear the pair at `witnessing_pose`; absent, no tested translation cleared "
            "it inside the search bound, which usually means the geometry has to change "
            "rather than move.",
        ]
    return [
        "UNCONFIRMED rows lack exact adjudication; only rows marked `adjudicated: "
        "true` are confirmed. Check `evidence.exact_adjudication` for availability "
        "and unjudged pairs. Bounding boxes can overlap while the surfaces are clear. "
        "Probe `gap(ir, '<a>', '<b>', "
        "poses=<witnessing_pose>)` and check its exactness/fallback flags before "
        "changing geometry.",
    ]


def _resolution_evidence(ir: dict, steps: dict, truncated: list,
                         n_dropped: int) -> dict:
    """At what resolution this verdict was established.

    A discrete sweep can only say "no overlap at the poses I looked at", so
    the poses it looked at are part of the answer. `finest_revolute_step_deg`
    is the tightest step any joint was swept at; `joints_truncated` names the
    joints whose range was too long for the geometry's step at
    `max_samples_per_joint`, so those were swept coarser than asked.
    """
    rev = [v for j in ir.get("joints") or []
           if j.get("type") == "revolute" and (v := steps.get(j["name"]))]
    pris = [v for j in ir.get("joints") or []
            if j.get("type") == "prismatic" and (v := steps.get(j["name"]))]
    out: dict = {
        "adaptive": bool(steps),
        "max_samples_per_joint": _ccd.max_samples_per_joint(),
    }
    if rev:
        import math as _math
        out["finest_revolute_step_deg"] = round(_math.degrees(min(rev)), 4)
        out["coarsest_revolute_step_deg"] = round(_math.degrees(max(rev)), 4)
    if pris:
        out["finest_prismatic_step_m"] = round(min(pris), 6)
    if truncated:
        out["joints_truncated"] = sorted(set(truncated))
    if n_dropped:
        out["adjudications_dropped"] = n_dropped
        out["max_adjudications_per_pair"] = _MAX_ADJUDICATIONS_PER_PAIR
    return out


def no_self_penetration(ir: dict, ctx=None,
                        *, per_joint_only: bool | None = None) -> VerifierReport:
    """Self-penetration check over sampled joint poses (continuous form).

    Sweeps each joint independently across its [lo, hi] (15° steps for
    revolute, 10 cm for prismatic, end-inclusive). For objects with no
    articulated joints, the sweep collapses to a single rest-pose check.

    `per_joint_only` (default True, overridable via the shared
    `DYCRAFT_CCD_PER_JOINT_ONLY` env var) controls the sampling regime:

      * **True** — per-joint sweep only. Each joint walks its range
        while every other joint stays at its authored default. Catches
        single-joint clearance bugs (a door that clips its own
        carcass at 90°, a drawer that hits the floor at full
        extension); simultaneous-multi-joint clashes (two doors on one
        cabinet both opened to intermediate angles into a shared
        mullion) are accepted as physically operable one-at-a-time.
      * **False** — additionally sample a multi-joint random hypercube
        (`DYCRAFT_CCD_RANDOM_SAMPLES`, default 24) to catch
        simultaneous-multi-joint clashes when those matter (folding
        mechanisms, scissor lifts).

    Pairs listed in `ir["collision_allowances"]` are skipped — the
    author has declared the overlap intentional.

    Reports **one row per penetrating pair**, each tagged with the sample pose
    that produced the deepest penetration for that pair *after* exact-geometry
    adjudication (`witnessing_pose`).  Keeping every SAT-positive pose until
    that pass is important: the deepest bounding-box overlap may be a false
    positive around a bore or cavity while a shallower pose really penetrates.
    """
    if per_joint_only is None:
        per_joint_only = per_joint_only_default()
    ir_hash = ir["id"]
    steps = _sweep_steps(ir)
    truncated: list = []
    sample_poses = all_samples(ir.get("joints", []),
                               random_count=(0 if per_joint_only else None),
                               steps=steps, truncated=truncated)
    allowed = _allowance_keys(ir)  # retained for evidence reporting
    allow_index = _allowance_index(ir)
    # Pre-resolve joint-type table for surface-unit conversion (used when
    # checking scoped allowances against witnessing poses).
    if allow_index:
        from ..sdk import _rotation as _rot
        _joint_types = _rot.object_joint_types_from_ir(ir)
    # Keep every SAT-positive pose through exact adjudication.  Do NOT collapse
    # to the deepest SAT witness here: for hollow/concave geometry that witness
    # can be a bounding-box false positive while another, shallower witness for
    # the same part pair is a real B-rep penetration.
    # Blanket allowances are pose-independent, so the pairs carrying one can
    # never survive the filter below. Handing them to the broad phase as an
    # exclusion removes the work rather than the result: on a 138-part umbrella
    # that is 96% of the SAT pairs computed at every one of ~20k poses.
    blanket = frozenset(k for k, recs in (allow_index or {}).items()
                        if any("joint" not in r for r in recs))
    sat_candidates: list = []
    for pose in sample_poses:
        pairs = object_query.penetration_pairs(ir, poses=pose or None,
                                               skip_pairs=blanket)
        for p in pairs:
            pair_key = frozenset((p.a, p.b))
            recs = allow_index.get(pair_key) if allow_index else None
            if recs is not None:
                # Convert sample pose to surface units for scope comparison.
                witnessing_surface = _rot.convert_poses_to_surface(
                    p.witnessing_pose or {}, _joint_types
                )
                if _allowance_matches(recs, witnessing_surface):
                    continue
            # The adjudicator needs the joint configuration this pair was
            # witnessed at, since that is the configuration whose exact
            # geometry it has to place.
            sat_candidates.append((p, pose or None))
    sat_candidates, n_dropped = _bounded_candidates(sat_candidates)
    adjudication = None
    if sat_candidates:
        # Exact-geometry adjudication. SAT bounds a solid by its AABB, so it
        # over-reports
        # — a rod through a tube's bore or a door through a shelled carcass's
        # open front both read as deep penetrations. Re-check every flagged
        # pair/pose against real geometry and drop the ones that are not
        # touching. This can only ever REMOVE candidates: pairs SAT did not flag
        # are never examined, and anything unanswerable keeps its SAT verdict.
        from . import _exact_adjudicator as _adj
        kept, adjudication = _adj.adjudicate_object_pairs(
            ir, sat_candidates,
            run_dir=getattr(ctx, "run_dir", None),
            cache=getattr(ctx, "adjudication_cache", None),
            # The object layer stays on the CAD child. Its repair advice is
            # written against `exact_volume_cm3` — a boolean intersection, and
            # the number this prompt tells the agent to size a fix from — which
            # no mesh query produces. Above this layer the question is only
            # whether two placed bodies overlap, and `_narrow_phase` answers that
            # pair-for-pair identically and far faster.
            backend=_adj.BACKEND_BREP,
            # Ask the meshes first, and spend the boolean only where a volume
            # is actually needed: a boolean between two thin curved parts can
            # take a minute against tens of milliseconds on the mesh, and a
            # batch that runs out its timeout is discarded whole. The mesh is
            # only ever used to CLEAR: a mesh `PENETRATING` escalates to the
            # boolean, so every `exact_volume_cm3` the agent reads still comes
            # from one.
            mesh_prefilter=True,
        )
        # Only now is it safe to choose one evidence row per part pair.  A
        # deeper SAT candidate may have been cleared while a shallower, exact-
        # confirmed candidate survived.
        worst_per_pair: dict = {}
        for p in kept:
            key = (p.a, p.b)
            existing = worst_per_pair.get(key)
            if existing is None or p.signed_distance < existing.signed_distance:
                worst_per_pair[key] = p
        # Worst first, not alphabetical. `signed_distance` is negative by the
        # overlap, so the most-negative pair is the deepest; the exact volume
        # would be the better key but it is only known after adjudication,
        # which runs downstream of this ordering. Names break ties so the
        # order is stable run to run.
        rows = sorted(worst_per_pair.values(),
                      key=lambda pair: (pair.signed_distance, pair.a, pair.b))
    else:
        rows = []
    if rows:
        # The CCD sampler operates in IR-internal units (radians for
        # revolute). The agent / LLM expects surface units (degrees) on
        # every evidence field, matching the convention used by
        # `joint_table`, `assert_*`, `.at(...)`, and every probe `poses=`.
        from ..sdk import _rotation as _rot
        joint_types = _rot.object_joint_types_from_ir(ir)

        defaults = {j["name"]: float(j.get("default", 0.0))
                    for j in ir.get("joints") or []
                    if j.get("type") != "fixed"}

        def _to_surface(p):
            return _rot.convert_poses_to_surface(p, joint_types) if p else {}

        def _swept_joint(p):
            """Which joint left its default in this witness, and for what.

            `per_joint_samples` moves ONE joint and holds every other at its
            authored default, so a witnessing pose is a single-joint sweep.
            Printing eight numbers without saying which one moved makes the
            author reverse-engineer the sampler — and several of these objects
            have non-zero rest poses (a recorded cat's `tail_curl_1..5`
            default to 18/12/10/8/6 degrees), so the moved joint is not the
            one that looks unusual.

            It matters because 36 of 59 confirmed penetrations in the recorded
            corpus exist ONLY while a joint is off its default: 61% of them do
            not exist in the assembled object at all, and the witness usually
            sits at a range extreme. One recorded helmet failed `shell~visor`
            at 42 degrees, then failed the same pair again at 56 after the
            author had WIDENED the range while fixing the clearance.
            """
            if not p:
                return None
            surface = _rot.convert_poses_to_surface(p, joint_types)
            moved = [(name, value) for name, value in surface.items()
                     if abs(float(value) - _rot.convert_poses_to_surface(
                         {name: defaults.get(name, 0.0)}, joint_types
                     ).get(name, 0.0)) > 1e-9]
            if len(moved) != 1:
                return None
            name, value = moved[0]
            return {"joint": name, "value": value,
                    "default": _rot.convert_poses_to_surface(
                        {name: defaults.get(name, 0.0)}, joint_types
                    ).get(name, 0.0)}

        return VerifierReport(
            name=NO_SELF_PENETRATION,
            verdict="fail",
            ir_hash=ir_hash,
            evidence={
                "samples_checked": len(sample_poses),
                "sweep_resolution": _resolution_evidence(
                    ir, steps, truncated, n_dropped),
                "per_joint_only": per_joint_only,
                "pairs": [
                    {**_pair_evidence(p),
                     "witnessing_pose": _to_surface(p.witnessing_pose),
                     **({"swept_joint": sj}
                        if (sj := _swept_joint(p.witnessing_pose)) else {})}
                    for p in rows
                ],
                "allowed_pairs": [sorted(list(k)) for k in allowed],
                # What the exact pass did. `available: false` means the CAD
                # kernel could not answer, so these rows were checked only
                # against bounding boxes and some may be false positives.
                "exact_adjudication": adjudication,
            },
            suggested_repairs=[
                "`separating_shift_m`, when present, is a world-frame translation of "
                "`b` tested to clear this pair at the reported pose. It is not "
                "necessarily minimal or valid at other poses. An absent shift means "
                "no tested correction is available, not that translation is "
                "impossible.",
                # Without this the agent cannot tell how far to move things: it
                # reads `depth`, moves that far, and overshoots into geometry
                # that was correct.
                *_adjudication_repairs(
                    adjudication,
                    [_pair_evidence(p) for p in rows]),
                "If `witnessing_pose` is empty / `{}`, the parts overlap "
                "statically at rest — adjust anchor translations or "
                "part dimensions.",
                "A nonempty `witnessing_pose` identifies the reported joint "
                "configuration; it does not exclude overlap at rest. Inspect the "
                "interface there and adjust geometry, anchors, or limits while "
                "preserving required travel.",
                "If the overlap is intentional, declare it with "
                "`obj.allow_collision(a, b, reason=...)`.",
            ],
        )
    return VerifierReport(
        name=NO_SELF_PENETRATION,
        verdict="pass",
        ir_hash=ir_hash,
        evidence={"pairs": [],
                  "samples_checked": len(sample_poses),
                  "sweep_resolution": _resolution_evidence(
                      ir, steps, truncated, n_dropped),
                  "per_joint_only": per_joint_only,
                  "allowed_pairs": [sorted(list(k)) for k in allowed],
                  "exact_adjudication": adjudication},
        suggested_repairs=[],
    )


# ---------------------------------------------------------------------------
# ObjectRelationalAssertions
# ---------------------------------------------------------------------------

OBJECT_RELATIONAL_ASSERTIONS = "ObjectRelationalAssertions"


def object_relational_assertions(ir: dict, ctx=None) -> VerifierReport:
    """Evaluate ir['assertions'] at the object's rest pose.

    Reuses the layer-agnostic kernel `verify._relational.evaluate_assertions`
    (also driving the scene-layer `SceneRelationalAssertions`). The
    object-layer anchor resolver consults `object_query.anchor_position`
    against the object IR.
    """
    ir_hash = ir["id"]

    def resolve(entity_name: str, anchor_name: str, poses=None,
                anchor_offset=None):
        return object_query.anchor_position(
            ir, entity_name, anchor_name, poses=poses,
            anchor_offset=anchor_offset)

    def resolve_entity_aabb(entity_dict):
        return object_query.resolve_entity_aabb(ir, entity_dict)

    def resolve_direction(direction_dict):
        return object_query.resolve_direction(ir, direction_dict)

    if not ir.get("assertions"):
        return not_applicable(OBJECT_RELATIONAL_ASSERTIONS, ir_hash, "no assertions declared")
    results, repairs, any_fail = evaluate_assertions(
        ir.get("assertions", []),
        resolve,
        resolve_entity_aabb_fn=resolve_entity_aabb,
        resolve_direction_fn=resolve_direction,
    )
    if not any_fail:
        return VerifierReport(
            name=OBJECT_RELATIONAL_ASSERTIONS, verdict="pass",
            ir_hash=ir_hash,
            evidence={"results": results},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=OBJECT_RELATIONAL_ASSERTIONS, verdict="fail",
        ir_hash=ir_hash,
        evidence={"results": results},
        suggested_repairs=repairs,
    )


# ---------------------------------------------------------------------------
# MinPartDimension
# ---------------------------------------------------------------------------

MIN_PART_DIMENSION = "MinPartDimension"

#: The dimension floor, as a fraction of the object's own overall diagonal.
#: This check exists to catch a unit slip and a collapsed dimension, both of
#: which are *ratio* errors — a mm-for-m confusion is a factor of 1000
#: whatever the object's size — so the threshold has to scale with the object
#: or it means something different at every scale. The recorded corpus spans
#: 7.6 mm to 16.35 m in object diagonal, a factor of 2000, and its thinnest
#: part is 8.6e-4 of its own object's diagonal, so 1e-4 clears real work by
#: about 8x while still catching a thousandfold slip on a large object — which
#: a fixed 1 mm floor did not.
_MIN_PART_DIM_REL = 1.0e-4

#: And a floor under the floor, for an object small enough that the relative
#: figure stops meaning anything. 0.1 mm is a hundred times the 1 micron
#: vertex-weld quantum (`_geometry_qc._OBJ_COORD_DECIMALS`), so a part this
#: thin is still a hundred mesh quanta thick; nothing in the corpus is within
#: a factor of two of it.
_MIN_PART_DIM_ABS_M = 1.0e-4


def _part_local_aabb(part: dict):
    """A part's local AABB, or None if its primitive cannot supply one.

    Goes through `_primitives.local_aabb` rather than reading `aabb_min` /
    `aabb_max` directly, because those keys exist only on solids and imports:
    an analytic `box` carries width/length/height and would otherwise read as
    having no box at all, silently collapsing every scale-relative threshold
    below to its absolute floor.
    """
    try:
        return _prim.local_aabb(part.get("primitive"), part.get("params") or {})
    except BaseException:  # noqa: BLE001 — an unmeasurable part is "no box"
        return None


def _object_diagonal(ir: dict) -> float:
    """Diagonal of the box enclosing every part, in meters.

    Local boxes, unposed: this is the object's own scale for comparing its
    parts against each other, not its footprint in the world. Composing rest
    poses would make the figure depend on joint defaults, so an author who
    opened a door would move the threshold that judges their gasket.

    0.0 when no part can supply a box, which callers read as "no scale to
    measure against" and answer with the absolute floor.
    """
    lo = [None, None, None]
    hi = [None, None, None]
    for part in ir.get("parts") or []:
        box = _part_local_aabb(part)
        if box is None:
            continue
        a, b = box
        for i in range(3):
            va, vb = float(a[i]), float(b[i])
            lo[i] = va if lo[i] is None else min(lo[i], va)
            hi[i] = vb if hi[i] is None else max(hi[i], vb)
    if lo[0] is None:
        return 0.0
    return sum((hi[i] - lo[i]) ** 2 for i in range(3)) ** 0.5


def _min_part_dim_floor(ir: dict) -> float:
    """The dimension floor for this object."""
    return max(_MIN_PART_DIM_ABS_M, _MIN_PART_DIM_REL * _object_diagonal(ir))


def min_part_dimension(ir: dict, ctx=None) -> VerifierReport:
    """Fail if any part has a dimension below the object's own scale floor.

    Catches a unit confusion (mm-for-m) and a dimension that collapsed to
    nothing — both of which leave sliver geometry that collides oddly and is
    invisible at any render scale.

    The floor is `max(0.1 mm, 1e-4 x object diagonal)`, not a fixed length,
    because the errors it looks for are ratios: a 1000x slip is a 1000x slip
    on a wristwatch and on a crane. A fixed 1 mm was simultaneously too
    strict at the small end — a 0.4 mm gasket on a 170 mm object is a real
    part and would fail — and vacuous at the large end, where a slipped
    dimension on a 16 m object clears 1 mm easily. The floor is reported in
    the evidence so the number is never a mystery.
    """
    ir_hash = ir["id"]
    floor = _min_part_dim_floor(ir)
    violations = []
    for p in ir["parts"]:
        # Check each authored dimension of the primitive (box:
        # width/length/height). Iterating the
        # actual params keeps box behavior identical and avoids false zeros
        # for non-box primitives that lack width/length/height keys.
        params = p.get("params") or {}
        primitive = p.get("primitive")
        if primitive in _prim.ASSET_PRIMITIVES:
            # Solid params are heterogeneous (path + digest strings, AABB
            # vectors, a nested shape table), so the raw-value loop below
            # doesn't apply. Check the actual bounding-box extents instead.
            (xmn, ymn, zmn), (xmx, ymx, zmx) = _prim.local_aabb(primitive, params)
            dims = (("extent_x", xmx - xmn), ("extent_y", ymx - ymn),
                    ("extent_z", zmx - zmn))
        else:
            dims = tuple((k, float(v)) for k, v in params.items())
        for dim_name, val in dims:
            if val < floor:
                violations.append({
                    "part": p["name"], "dim": dim_name, "value": val,
                    "min_allowed": floor,
                })
    scale = {"min_allowed_m": floor,
             "object_diagonal_m": _object_diagonal(ir),
             "rule": "max(1e-4 m, 1e-4 x object diagonal)"}
    if not violations:
        return VerifierReport(
            name=MIN_PART_DIMENSION, verdict="pass",
            ir_hash=ir_hash, evidence={"violations": [], "scale": scale},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=MIN_PART_DIMENSION, verdict="fail",
        ir_hash=ir_hash, evidence={"violations": violations, "scale": scale},
        suggested_repairs=[
            "Each reported dimension is below `scale.min_allowed_m` = max(1e-4 m, "
            "1e-4 × object diagonal). Check unit conversion: 8 mm is 0.008 m; do not "
            "divide by 1000 twice.",
            "If the intended dimension is below this threshold, the verifier still "
            "rejects it. Adjust the feature, or report the unresolved scale limitation.",
        ],
    )


# ---------------------------------------------------------------------------
# JointRangeNonTrivial
# ---------------------------------------------------------------------------

JOINT_RANGE_NON_TRIVIAL = "JointRangeNonTrivial"
#: The floor on how far a joint actually MOVES its child, in meters.
#:
#: One number for both joint types, because the question is the same one: does
#: this connection carry usable motion, or is it a `fixed` joint wearing a
#: range? An angle cannot answer that on its own — 1° sweeps 0.36 mm at the
#: corpus's shortest lever arm and 161 mm at its longest, a factor of 440 — so
#: a revolute joint is measured by the arc its child's furthest corner travels
#: (`span x radius`) and a prismatic one by its span, which already is that
#: distance.
#:
#: 1 mm sits about 20x below the least-moving joint in the recorded corpus
#: (swept arc: min 20.3 mm over 434 revolute joints; prismatic span: min
#: 11 mm over 69), so it fails nothing real while still catching a joint
#: authored articulated and left functionally rigid.
_MIN_JOINT_TRAVEL_M = 1.0e-3

#: Fallback for a revolute joint whose child has no measurable box, where
#: there is no lever arm to convert an angle into a distance. Deliberately the
#: old fixed threshold: with no geometry the angle is all there is.
_MIN_REVOLUTE_RANGE_RAD = 1.0 * (3.141592653589793 / 180.0)  # 1 degree


def joint_range_non_trivial(ir: dict, ctx=None) -> VerifierReport:
    """Fail if a revolute joint range is < 1° or a prismatic range is < 1 mm.

    Catches joints authored as articulated but functionally fixed.
    Below these thresholds the joint contributes no usable motion and
    should be modeled as a `fixed` joint instead.

    Evidence is in **surface units** — `span_deg` for a revolute joint,
    `span_m` for a prismatic one — so the number an author reads is in the
    same unit as the `limits` they wrote.
    """
    from ..sdk import _rotation as _rot

    ir_hash = ir["id"]
    if all(j["type"] == "fixed" for j in ir["joints"]):
        return not_applicable(JOINT_RANGE_NON_TRIVIAL, ir_hash, "no movable joints")
    violations = []
    for j in ir["joints"]:
        jt = j["type"]
        if jt == "fixed":
            continue
        lo, hi = j["limits"]
        span = float(hi) - float(lo)
        if jt == "revolute":
            radius = _joint_sweep_radius(ir, j)
            if radius <= 0.0:
                # No geometry to convert the angle against.
                if span < _MIN_REVOLUTE_RANGE_RAD:
                    violations.append({
                        "joint": j["name"], "type": jt,
                        "span_deg": _rot.rad_to_deg(span),
                        "min_required_deg": _rot.rad_to_deg(
                            _MIN_REVOLUTE_RANGE_RAD),
                    })
                continue
            travel = abs(span) * radius
            if travel < _MIN_JOINT_TRAVEL_M:
                # Degrees, matching the `limits` the author wrote; nothing
                # agent-facing speaks radians. `travel_m` is what the check
                # actually measured, and `radius_m` says why the same angle
                # passes on a longer part.
                violations.append({
                    "joint": j["name"], "type": jt,
                    "span_deg": _rot.rad_to_deg(span),
                    "radius_m": radius,
                    "travel_m": travel,
                    "min_required_travel_m": _MIN_JOINT_TRAVEL_M,
                })
        elif jt == "prismatic" and abs(span) < _MIN_JOINT_TRAVEL_M:
            violations.append({
                "joint": j["name"], "type": jt, "span_m": span,
                "travel_m": abs(span),
                "min_required_travel_m": _MIN_JOINT_TRAVEL_M,
            })
    if not violations:
        return VerifierReport(
            name=JOINT_RANGE_NON_TRIVIAL, verdict="pass",
            ir_hash=ir_hash, evidence={"violations": []},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=JOINT_RANGE_NON_TRIVIAL, verdict="fail",
        ir_hash=ir_hash, evidence={"violations": violations},
        suggested_repairs=[
            "Each reported joint has travel below `min_required_travel_m`. Revolute "
            "travel is the arc swept by the furthest child corner, derived from "
            "`span_deg` and `radius_m`. Widen the limits, or use `fixed` for a truly "
            "rigid connection.",
            "Revolute joint authoring takes limits in DEGREES; the "
            "kernel stores them as radians internally. If you intended "
            "to author a 90° hinge but passed `pi/2`, that's about 1.5° "
            "— pass `90.0` instead.",
        ],
    )


# ---------------------------------------------------------------------------
# JointOriginNearGeometry
# ---------------------------------------------------------------------------

JOINT_ORIGIN_NEAR_GEOMETRY = "JointOriginNearGeometry"
_JOINT_ORIGIN_TOL_M = 5.0e-3  # 5 mm


def _point_outside_aabb_distance(p, aabb) -> float:
    """Signed distance from `p` to the closed AABB.

    Returns 0 if p is inside or on the surface; otherwise the Euclidean
    distance to the nearest point on the AABB boundary.
    """
    (xmn, ymn, zmn), (xmx, ymx, zmx) = aabb
    dx = max(xmn - p[0], 0.0, p[0] - xmx)
    dy = max(ymn - p[1], 0.0, p[1] - ymx)
    dz = max(zmn - p[2], 0.0, p[2] - zmx)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def joint_origin_near_geometry(ir: dict, ctx=None) -> VerifierReport:
    """Fail if a joint's pivot (`parent_anchor`'s world position at rest) is farther than 5 mm from BOTH parts' world
    AABBs.

    An anchor can legitimately carry a translation offset that puts its world position off the parent's box (a thin
    floor parenting tall walls through long offsets), so a joint counts as geometrically grounded if its pivot sits near
    at least one of the two parts it connects. Only when both are far is the joint floating, the authoring blunder this
    targets. That rule is the whole check: the 5 mm is slack, absolute on purpose, since only a few percent of joints
    fall in the band where it decides anything and the failures it catches are gross, not marginal.

    **Fixed joints are skipped, because a fixed joint has no pivot.** Its anchor pair fixes only the child's transform
    relative to the parent; where the shared frame sits is gauge freedom, so the distance would measure the author's
    choice of construction frame, and the most natural choice (two coincident frames at the parts' common origin) is the
    one that fails. `JointOriginNearBothBodies` skips fixed joints for the same reason.
    """
    from ..sdk import object_query as _oq
    ir_hash = ir["id"]
    if all(j["type"] == "fixed" for j in ir["joints"]):
        return not_applicable(JOINT_ORIGIN_NEAR_GEOMETRY, ir_hash, "no movable joints",
                              fixed_joints_skipped=len(ir["joints"]))
    frames = _oq._compute_frames(ir)
    violations = []
    skipped_fixed = 0
    for j in ir["joints"]:
        if j.get("type") == "fixed":
            # No pivot to be near — see the docstring.
            skipped_fixed += 1
            continue
        pa = j.get("parent_anchor")
        if not pa:
            continue
        pa_t_local = pa.get("translation", [0.0, 0.0, 0.0])
        p_t_local = (float(pa_t_local[0]), float(pa_t_local[1]),
                     float(pa_t_local[2]))
        p_frame = frames.get(j["parent"])
        if p_frame is None:
            continue
        rotated = _oq._rot.quat_rotate_vec(p_frame[1], p_t_local)
        pivot_world = (p_frame[0][0] + rotated[0],
                       p_frame[0][1] + rotated[1],
                       p_frame[0][2] + rotated[2])
        parent_aabb = _oq.part_world_aabb(ir, j["parent"])
        child_aabb = _oq.part_world_aabb(ir, j["child"])
        d_parent = _point_outside_aabb_distance(pivot_world, parent_aabb)
        d_child = _point_outside_aabb_distance(pivot_world, child_aabb)
        d_min = min(d_parent, d_child)
        if d_min > _JOINT_ORIGIN_TOL_M:
            violations.append({
                "joint": j["name"],
                "pivot_world": list(pivot_world),
                "distance_outside_parent_aabb_m": d_parent,
                "distance_outside_child_aabb_m": d_child,
                "tol_m": _JOINT_ORIGIN_TOL_M,
            })
    # Reported either way, so a clean pass says what it did NOT look at
    # rather than implying it checked every joint.
    scope = {"violations": violations, "fixed_joints_skipped": skipped_fixed}
    if not violations:
        return VerifierReport(
            name=JOINT_ORIGIN_NEAR_GEOMETRY, verdict="pass",
            ir_hash=ir_hash, evidence=scope,
            suggested_repairs=[],
        )
    return VerifierReport(
        name=JOINT_ORIGIN_NEAR_GEOMETRY, verdict="fail",
        ir_hash=ir_hash, evidence=scope,
        suggested_repairs=[
            "Each reported pivot is farther than `tol_m` from both parent and child "
            "world AABBs. This is a bounds check, not a measurement of the actual "
            "joint interface.",
            "Inspect the intended pivot and actual surfaces. Adjust parent/child "
            "anchor translations in each part's local frame, or choose appropriate "
            "built-in or registered anchors. The check requires the pivot within "
            "`tol_m` of at least one part's AABB.",
        ],
    )


# ---------------------------------------------------------------------------
# JointOriginNearBothBodies (mechanism-realism guardrail)
# ---------------------------------------------------------------------------

JOINT_ORIGIN_NEAR_BOTH_BODIES = "JointOriginNearBothBodies"
# Reuse the 5 mm tolerance of the sibling JointOriginNearGeometry check.
_JOINT_BOTH_BODIES_TOL_M = _JOINT_ORIGIN_TOL_M


def joint_origin_near_both_bodies(ir: dict, ctx=None) -> VerifierReport:
    """Advisory mechanism-realism guardrail (revolute/prismatic only).

    Where `JointOriginNearGeometry` is a *lenient blunder* check — it fails
    only when the pivot floats away from BOTH parts (``min(d_parent,
    d_child)``) — this verifier targets *mechanism realism*: a movable joint
    whose pivot the CHILD body does not reach, and that no visible connector
    part bridges, renders as an invisible/floating articulation (the elbow
    that pivots from a bare rod gap; the wrist with no hardware).

    A joint is **grounded** iff ``(d_parent <= tol AND d_child <= tol)`` OR
    some part ``c`` not in ``{parent, child}`` *bridges* the pivot: ``c``'s
    world AABB lies within ``tol`` of the pivot AND overlaps the child's
    world AABB (a cylinder pin through box ears, an axle under a wheel, a
    sphere knuckle that genuinely seats into the leaf). Fixed joints are
    skipped.

    Verdict is **`uncertain`** on violation — advisory, surfacing the suspect
    joint to the agent without blocking commit (`uncertain` never sets the
    cheap-suite short-circuit nor the `ok` gate; see
    ``agent/layers/compile.py``). Promote to `fail` only after a clean
    batch run. Lateral offset of the child from the joint axis is reported as
    evidence ONLY, never a trigger — a healthy revolute child always extends
    perpendicular to its axis, so triggering on it would false-positive every
    door.
    """
    from ..sdk import object_query as _oq
    ir_hash = ir["id"]
    if all(j["type"] == "fixed" for j in ir["joints"]):
        return not_applicable(JOINT_ORIGIN_NEAR_BOTH_BODIES, ir_hash, "no movable joints")
    frames = _oq._compute_frames(ir)
    tol = _JOINT_BOTH_BODIES_TOL_M
    # World AABB per part, computed once (reused for the bridging search).
    part_aabbs = {}
    for p in ir["parts"]:
        try:
            part_aabbs[p["name"]] = _oq.part_world_aabb(ir, p["name"])
        except KeyError:
            continue
    violations = []
    for j in ir["joints"]:
        jt = j["type"]
        if jt not in ("revolute", "prismatic"):
            continue
        pa = j.get("parent_anchor")
        if not pa:
            continue
        p_frame = frames.get(j["parent"])
        if p_frame is None:
            continue
        parent_aabb = part_aabbs.get(j["parent"])
        child_aabb = part_aabbs.get(j["child"])
        if parent_aabb is None or child_aabb is None:
            continue
        # Pivot = parent_anchor translation rotated through the parent frame
        # (same computation as JointOriginNearGeometry, kept byte-consistent).
        pa_t_local = pa.get("translation", [0.0, 0.0, 0.0])
        p_t_local = (float(pa_t_local[0]), float(pa_t_local[1]),
                     float(pa_t_local[2]))
        rotated = _oq._rot.quat_rotate_vec(p_frame[1], p_t_local)
        pivot_world = (p_frame[0][0] + rotated[0],
                       p_frame[0][1] + rotated[1],
                       p_frame[0][2] + rotated[2])
        d_parent = _point_outside_aabb_distance(pivot_world, parent_aabb)
        d_child = _point_outside_aabb_distance(pivot_world, child_aabb)
        # Bridging-connector escape: a third part seats at the pivot and
        # overlaps the child (a pin through ears, an axle under a wheel).
        bridging_part = None
        if not (d_parent <= tol and d_child <= tol):
            for name, aabb in part_aabbs.items():
                if name == j["parent"] or name == j["child"]:
                    continue
                if (_point_outside_aabb_distance(pivot_world, aabb) <= tol
                        and _oq._aabbs_overlap(aabb, child_aabb)):
                    bridging_part = name
                    break
        grounded = (d_parent <= tol and d_child <= tol) \
            or bridging_part is not None
        if grounded:
            continue
        # Supplementary evidence only (NOT a trigger): lateral distance of
        # the child AABB center from the joint axis line through the pivot.
        lateral = None
        axis = None
        try:
            axis = _oq.joint_axis_world(ir, j["name"])
        except (KeyError, ValueError):
            # The only two documented outcomes: KeyError for a joint or parent
            # this IR does not name, ValueError for a `fixed` joint, which has
            # no axis to measure against. Both mean "no axis here", and since
            # this is evidence rather than a trigger, the check continues
            # without it. Deliberately NOT `except Exception`: anything else
            # coming out of the kernel is a defect in the kernel, and swallowing
            # it here would cost the evidence on every joint of every object
            # while looking exactly like the two benign cases.
            axis = None
        if axis is not None:
            an = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
            if an > 0.0:
                u = (axis[0] / an, axis[1] / an, axis[2] / an)
                cc = ((child_aabb[0][0] + child_aabb[1][0]) / 2.0,
                      (child_aabb[0][1] + child_aabb[1][1]) / 2.0,
                      (child_aabb[0][2] + child_aabb[1][2]) / 2.0)
                v = (cc[0] - pivot_world[0], cc[1] - pivot_world[1],
                     cc[2] - pivot_world[2])
                vdotu = v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
                perp = (v[0] - vdotu * u[0], v[1] - vdotu * u[1],
                        v[2] - vdotu * u[2])
                lateral = (perp[0] ** 2 + perp[1] ** 2
                           + perp[2] ** 2) ** 0.5
        violations.append({
            "joint": j["name"],
            "type": jt,
            "pivot_world": list(pivot_world),
            "distance_outside_parent_aabb_m": d_parent,
            "distance_outside_child_aabb_m": d_child,
            "lateral_offset_from_axis_m": lateral,
            "bridging_part": bridging_part,
            "tol_m": tol,
        })
    if not violations:
        return VerifierReport(
            name=JOINT_ORIGIN_NEAR_BOTH_BODIES, verdict="pass",
            ir_hash=ir_hash, evidence={"violations": []},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=JOINT_ORIGIN_NEAR_BOTH_BODIES, verdict="uncertain",
        ir_hash=ir_hash, evidence={"violations": violations},
        suggested_repairs=[
            "Advisory: the pivot is farther than `tol_m` from at least one body's "
            "AABB, and no third part's AABB was found near the pivot and overlapping "
            "the child. Inspect the real interface; this bounds heuristic does not "
            "establish a missing mechanism.",
            "If the interface is unsupported, align the relevant anchor to its "
            "hinge/slide edge or add a mechanically appropriate connector between the "
            "pivot and child. Preserve required motion and check the surfaces after "
            "editing.",
            "A connector should physically reach the moving body; proximity to the "
            "parent alone does not establish that connection.",
        ],
    )


# ---------------------------------------------------------------------------
# SupportConnected (object layer)
# ---------------------------------------------------------------------------

SUPPORT_CONNECTED = "SupportConnected"


def support_connected(ir: dict, ctx=None) -> VerifierReport:
    """Fail if any part lacks a contact chain to the object's floor.

    The "floor" is the lowest-z plane of the object's parts at rest;
    any part whose world `zmin` is within `contact_tol` of that plane
    is directly grounded. Other parts must reach the floor through
    contact, **measured on surfaces** — see `_support.build_object_adjacency`.
    Kinematic attachments alone don't create support: a fixed-attached child
    offset 1 m from its parent with nothing touching is unsupported.
    """
    ir_hash = ir["id"]
    ok, unsupported, evidence = support_connected_object(
        ir, object_query,
        run_dir=getattr(ctx, "run_dir", None),
        cache=getattr(ctx, "adjudication_cache", None))
    if ok:
        return VerifierReport(
            name=SUPPORT_CONNECTED, verdict="pass",
            ir_hash=ir_hash, evidence=evidence,
            suggested_repairs=[],
        )
    return VerifierReport(
        name=SUPPORT_CONNECTED, verdict="fail",
        ir_hash=ir_hash, evidence=evidence,
        suggested_repairs=[
            f"Unsupported parts: {unsupported}. No support chain was found "
            "under the contact and nearby-joint checks to the object's floor (the "
            f"lowest z-plane at z={evidence['floor_z']:.4g}).",
            "Inspect unsupported parts and the reported adjacency/contact evidence. "
            "Check actual interfaces and each anchor's local translation before "
            "repositioning parts; a joint in the kinematic tree does not itself "
            "establish physical contact.",
            "Where the design needs physical support, add an appropriate connector or "
            "adjust the contacting surfaces. Preserve intended clearances elsewhere.",
        ],
    )


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

OBJECT_VERIFIERS = (
    (NO_SELF_PENETRATION, no_self_penetration),
    (MIN_PART_DIMENSION, min_part_dimension),
    (JOINT_RANGE_NON_TRIVIAL, joint_range_non_trivial),
    (JOINT_ORIGIN_NEAR_GEOMETRY, joint_origin_near_geometry),
    (JOINT_ORIGIN_NEAR_BOTH_BODIES, joint_origin_near_both_bodies),
    (SUPPORT_CONNECTED, support_connected),
    (OBJECT_RELATIONAL_ASSERTIONS, object_relational_assertions),
)

#: No automatic image critique runs in this suite. A rubric critique on every
#: compile would make a compile cost minutes and, run automatically, would keep
#: funding compiles that move nothing. The agent's visual check is `review()`: a fresh
#: reviewer that renders for itself (`verify/review_images.py`), bought only
#: when the author asks.



__all__ = [
    "NO_SELF_PENETRATION",
    "no_self_penetration",
    "MIN_PART_DIMENSION",
    "min_part_dimension",
    "JOINT_RANGE_NON_TRIVIAL",
    "joint_range_non_trivial",
    "JOINT_ORIGIN_NEAR_GEOMETRY",
    "joint_origin_near_geometry",
    "JOINT_ORIGIN_NEAR_BOTH_BODIES",
    "joint_origin_near_both_bodies",
    "SUPPORT_CONNECTED",
    "support_connected",
    "OBJECT_RELATIONAL_ASSERTIONS",
    "object_relational_assertions",
    "OBJECT_VERIFIERS",
]
