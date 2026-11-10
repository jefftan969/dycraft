"""Scene-layer verifiers.

`SCENE_VERIFIERS`, at the bottom of this module, is the scene layer's suite: the object suite over every placed
definition (`_per_object.py`), the four checks over placed instances below (`INSTANCE_VERIFIERS`), and
AnatomicalJointLimits. All of them are geometric; a visual critique is requested with `review()`, never run
automatically (see the note on `SCENE_VERIFIERS`). Each check uses boxes as the broad phase and lets exact geometry
decide:

  * NoCrossInstancePenetration: penetration between instances at the rest pose, every SAT-positive part pair
    adjudicated exactly.
  * RoomContainment: every instance's geometry inside the room, candidates confirmed by
    `_exact_adjudicator.adjudicate_room_escapes`.
  * SupportConnected: every non-room instance reaches the room (floor, walls, or ceiling) through measured contacts
    within 0.01 m, directly or through grounded instances. Support is purely geometric: an `attach`/`attach_group`
    relationship does not satisfy it, so a lamp shade offset above its post still fails unless its geometry
    touches a supporter.
  * SceneRelationalAssertions: `ir["assertions"]` against rest-pose anchor positions (distance, signed or
    Euclidean; collinear; overlap; angle; axis_aligned).

Every verifier takes `(ir, ctx)` and returns a VerifierReport; `ctx` exposes
`catalog: Mapping[object_ref_hash, ObjectIR_dict]`. The relational kernel is shared with the object layer
(`verify/_relational.py::evaluate_assertions`); this module supplies the scene's anchor resolver, which consults the
catalog.
"""

from __future__ import annotations

import math

from ..sdk import scene_query
from .report import VerifierReport, not_applicable
from ._per_object import PER_OBJECT_VERIFIERS
from ._relational import evaluate_assertions
from ._support import support_connected_scene
from ._exact_adjudicator import reported_depth
from ._ccd_sampler import (all_samples, per_joint_only_default, rest_pose,
                            scene_static_only_default)


# ---------------------------------------------------------------------------
# NoCrossInstancePenetration
# ---------------------------------------------------------------------------

NO_CROSS_INSTANCE_PENETRATION = "NoCrossInstancePenetration"


def _frame_to_json(frame):
    t, e = frame
    return {"translation": [t[0], t[1], t[2]],
            "rotation_euler_deg": [e[0], e[1], e[2]]}


def _pair_evidence(pair) -> dict:
    # `penetration_m` is `reported_depth` — the smaller of SAT's box number and
    # the depth the mesh narrow phase measured on the triangles — and the SAT
    # quantities it supersedes are not shipped. `pair.depth` alone is measured
    # on oriented bounding boxes and is not bounded by the parts inside them:
    # a 100 mm ball can read as 230 mm inside a stand. The probe
    # this layer's agent plans against already reports `reported_depth`
    # (`sdk/_probes.py`), so the verifier agreeing with it is the point.
    out = {
        "a": pair.a,
        "b": pair.b,
        "penetration_m": reported_depth(pair),
        "a_frame": _frame_to_json(pair.a_frame),
        "b_frame": _frame_to_json(pair.b_frame),
        "a_bbox": [list(pair.a_bbox[0]), list(pair.a_bbox[1])],
        "b_bbox": [list(pair.b_bbox[0]), list(pair.b_bbox[1])],
    }
    # Which part-pair drove this instance-level record. Naming them turns an
    # otherwise opaque "these two objects collide" into something actionable.
    if getattr(pair, "a_sub", None) and getattr(pair, "b_sub", None):
        out["deepest_parts"] = [f"{pair.a}.{pair.a_sub}",
                                f"{pair.b}.{pair.b_sub}"]
    if getattr(pair, "adjudicated", False):
        out["adjudicated"] = True
        if pair.exact_volume is not None:
            # cm^3 — see the note in object_verify._pair_evidence: m^3 does not
            # survive the evidence layer's 1 mm float rounding.
            out["exact_volume_cm3"] = pair.exact_volume * 1.0e6
        if pair.closest_points is not None:
            out["closest_points"] = [list(pair.closest_points[0]),
                                     list(pair.closest_points[1])]
    return out


def _scene_joints(ir: dict, catalog) -> list[dict]:
    """Aggregate joint records across instances, with names qualified as
    `<instance>.<joint>` so the resulting pose dicts can be projected
    back per-instance via `scene_query.effective_object_poses`.

    `default` is the **instance's** committed value — its `initial_joints`
    where it has one, the definition's `default` otherwise. The distinction is
    load-bearing here rather than cosmetic: `rest_pose` turns these records
    into a pose dict that names *every* movable joint explicitly, and an
    explicit override outranks saved state, so reporting the definition's
    default would hand the collision sweep a configuration the scene never
    committed and quietly clear a door the author placed open. `limits` stay
    the definition's — an instance selects a value, it does not widen a range.
    """
    out: list[dict] = []
    for inst in ir["instances"]:
        obj_ref = inst["object_ref"]
        if obj_ref not in catalog:
            continue
        obj_ir = catalog[obj_ref]
        saved = scene_query.instance_initial_joints(ir, inst["name"]) or {}
        for j in obj_ir.get("joints", []):
            if j["type"] == "fixed":
                continue
            out.append({
                "name": f"{inst['name']}.{j['name']}",
                "type": j["type"],
                "default": saved.get(j["name"], j["default"]),
                "limits": j["limits"],
            })
    return out


def no_cross_instance_penetration(ir: dict, ctx, *,
                                  per_joint_only: bool | None = None,
                                  static_only: bool | None = None) -> VerifierReport:
    """Cross-instance penetration check over sampled instance-joint poses.

    Reports **one row per penetrating instance pair**, each tagged with the deepest surviving sample pose
    (`witnessing_pose`) after exact geometry has adjudicated every SAT-positive part pair and pose.

    `static_only` (**default True**, `DYCRAFT_SCENE_STATIC_ONLY`) checks the **rest pose only**, every joint at its
    authored `default`, and `evidence.sampling` says which regime ran. The sweep is off by default because a door's
    range is a set of poses it is meant to hold, while a manipulator's is a mechanical envelope almost none of which
    anyone will command (`_ccd_sampler.scene_static_only_default` has the reasoning). This is the scene layer only:
    the object layer's `NoSelfPenetration` still sweeps, because a door passing through its own carcass is a defect
    in the object that no later layer would notice.

    When `static_only` is False, `per_joint_only` (default True, the `DYCRAFT_CCD_PER_JOINT_ONLY` flag the object
    layer shares) selects the regime:

      * **True**: sweep each joint across its range while every other joint, on every instance, stays at its
        default; no multi-joint hypercube. Two neighboring cabinets whose doors would clip only if opened together
        pass, as a real kitchen does, while a door whose swing hits a neighbor's body is still caught.
      * **False**: also sample a multi-joint random hypercube (`DYCRAFT_CCD_RANDOM_SAMPLES`, default 24), for pairs
        that collide only with several joints open (folding mechanisms, scissor lifts).
    """
    if per_joint_only is None:
        per_joint_only = per_joint_only_default()
    if static_only is None:
        static_only = scene_static_only_default()
    ir_hash = ir["id"]
    joints = _scene_joints(ir, ctx.catalog)
    if static_only:
        sample_poses = rest_pose(joints)
        sampling = "rest_only"
    else:
        sample_poses = all_samples(joints,
                                   random_count=(0 if per_joint_only else None))
        sampling = "per_joint" if per_joint_only else "per_joint+hypercube"
    sat_candidates: list = []
    for pose in sample_poses:
        pairs = scene_query.penetration_part_pairs(
            ir, ctx.catalog, poses=pose or None)
        for p in pairs:
            # A deeper SAT proxy overlap can be empty space inside a hollow
            # BRep, while a shallower candidate for the same instances is real.
            sat_candidates.append((p, pose or None))

    # Exact-geometry adjudication, mirroring the object layer: SAT boxes a
    # solid as its AABB, so re-check the flagged
    # pairs against the true surface and drop the ones that are not touching.
    # Downgrade-only — an unanswerable pair keeps its SAT verdict.
    adjudication = None
    if sat_candidates:
        from . import _exact_adjudicator as _adj
        kept, adjudication = _adj.adjudicate_scene_pairs(
            ir, ctx.catalog, sat_candidates,
            run_dir=getattr(ctx, "run_dir", None),
            asset_base_dir=getattr(
                getattr(ctx, "layer_payload", None), "asset_base_dir", None),
            cache=getattr(ctx, "adjudication_cache", None),
        )
        worst_per_pair: dict = {}
        for p in kept:
            key = (p.a, p.b)
            existing = worst_per_pair.get(key)
            if existing is None or p.signed_distance < existing.signed_distance:
                worst_per_pair[key] = p
    else:
        worst_per_pair = {}

    hard_pairs = worst_per_pair

    # Convert CCD witnessing_pose values from IR-internal (radians for
    # revolute) to surface units (degrees) before exposing to the
    # agent. Pose keys are qualified `<instance>.<joint>`; type lookup
    # walks the catalog per instance.
    from ..sdk import _rotation as _rot

    def _to_surface(p):
        if not p:
            return {}
        return _rot.convert_scene_poses_to_surface(p, ir, ctx.catalog)

    if hard_pairs:
        # Worst first, not alphabetical. There is no contact duration at a
        # single witnessing pose, so depth is the severity, and a reader
        # scanning a long list should meet the deepest interference first
        # rather than whichever instance happens to sort earliest. The names
        # break ties, so the order is stable across runs and a diff of two
        # reports shows changes rather than a reshuffle.
        rows = sorted(hard_pairs.values(),
                      key=lambda pair: (-reported_depth(pair), pair.a, pair.b))
        return VerifierReport(
            name=NO_CROSS_INSTANCE_PENETRATION,
            verdict="fail",
            ir_hash=ir_hash,
            evidence={
                "samples_checked": len(sample_poses),
                "per_joint_only": per_joint_only, "sampling": sampling,
                "pairs": [
                    {**_pair_evidence(p),
                     "witnessing_pose": _to_surface(p.witnessing_pose)}
                    for p in rows
                ],
                # What the exact pass did. `available: false` means these rows
                # are unverified bounding-box results and some may be false
                # positives; `not_adjudicable` counts pairs whose geometry lives
                # in a catalog asset that staging does not copy.
                "exact_adjudication": adjudication,
            },
            suggested_repairs=[
                "Inspect each pair at `witnessing_pose`. `penetration_m` is how deep "
                "the pair overlaps, preferring the depth measured on the triangles to "
                "the bounding-box bound; it is an upper bound on the correction, not a "
                "direction to apply it in.",
                "`deepest_parts` names the specific part-pair that drove the "
                "record — an instance is many bodies, so start there rather "
                "than moving the whole object.",
                "`adjudicated: true` confirms overlap of the actual surfaces. Other "
                "rows may be false positives; check `evidence.exact_adjudication` for "
                "availability and unjudged pairs. Inspect the interface and measure "
                "after each edit.",
                "If `witnessing_pose` is empty / `{}`, the instances "
                "overlap statically at rest — re-position one of them.",
                "A nonempty `witnessing_pose` identifies the reported joint "
                "configuration; overlap may also exist at rest. Reposition instances "
                "or revise articulation clearance while preserving required motion.",
                "For a placed instance, edit its pose translation. For an attached "
                "instance, edit parent_anchor translation in the resolved anchor's "
                "frame, including a declared frame's rotation; convert any "
                "world-frame correction first. Scene.attach has no offset argument.",
            ],
        )
    return VerifierReport(
        name=NO_CROSS_INSTANCE_PENETRATION,
        verdict="pass",
        ir_hash=ir_hash,
        evidence={"pairs": [], "samples_checked": len(sample_poses),
                  "per_joint_only": per_joint_only, "sampling": sampling,
                  "exact_adjudication": adjudication},
        suggested_repairs=[],
    )


# ---------------------------------------------------------------------------
# RoomContainment
# ---------------------------------------------------------------------------

ROOM_CONTAINMENT = "RoomContainment"

# Floating-point noise floor for containment comparisons (1 µm). FK
# composition over rotations + nested anchor offsets accumulates ULP-scale
# error, so an instance whose authored AABB sits flush against a room face
# can compute as ~1e-17 m past the wall. Anything below this is taken as
# "on the boundary"; real authoring mistakes are orders of magnitude larger.
_CONTAINMENT_EPS_M = 1.0e-6


def _room_aabb(room: dict) -> tuple:
    w = float(room["width"])
    d = float(room["depth"])
    h = float(room["height"])
    return ((-w / 2.0, -d / 2.0, 0.0), (w / 2.0, d / 2.0, h))


def _aabb_outside(room_aabb, bb) -> dict | None:
    """Return a dict describing the violated axis(es), or None if `bb`
    lies entirely inside `room_aabb` (modulo `_CONTAINMENT_EPS_M`)."""
    rmin, rmax = room_aabb
    bmin, bmax = bb
    violations = {}
    axes = ("x", "y", "z")
    for i, ax in enumerate(axes):
        if bmin[i] < rmin[i] - _CONTAINMENT_EPS_M:
            violations[f"{ax}_below_min"] = bmin[i] - rmin[i]
        if bmax[i] > rmax[i] + _CONTAINMENT_EPS_M:
            violations[f"{ax}_above_max"] = bmax[i] - rmax[i]
    return violations or None


#: `"x_below_min"` / `"x_above_max"` from an `(axis_index, side)` pair, which
#: is the vocabulary `_exact_adjudicator.ROOM_FACES` speaks. Kept so the
#: evidence key names are the ones `scene_sdk.md` documents.
_AXIS = ("x", "y", "z")

_VIOLATION_KEY = {(0, "below"): "x_below_min", (0, "above"): "x_above_max",
                  (1, "below"): "y_below_min", (1, "above"): "y_above_max",
                  (2, "below"): "z_below_min", (2, "above"): "z_above_max"}


def _room_escape_candidates(room_aabb, bb):
    """`[(axis_index, side, amount_m), ...]` — the box's excursions past a face.

    The signed `violations` dict `_aabb_outside` returns is the same
    information; this is the shape `_exact_adjudicator.adjudicate_room_escapes`
    takes, with `amount_m` positive outward.
    """
    (rmin, rmax), (bmin, bmax) = room_aabb, bb
    cands = []
    for k in range(3):
        under = rmin[k] - bmin[k]
        over = bmax[k] - rmax[k]
        if under > _CONTAINMENT_EPS_M:
            cands.append((k, "below", under))
        if over > _CONTAINMENT_EPS_M:
            cands.append((k, "above", over))
    return cands


def room_containment(ir: dict, ctx) -> VerifierReport:
    """Is every instance's geometry inside the room?

    **The AABB is the broad phase, and exact geometry decides.** A bounding box
    contains air: even a tight box inflates around a surface of revolution
    turned in plan (a cylinder of radius R gains `R*(sqrt(2)-1)` at 45 degrees,
    37.3 mm for a 90 mm tyre), which has nothing to do with whether the part
    left the room.

    So candidates go to `_exact_adjudicator.adjudicate_room_escapes`, which
    intersects the part against a slab outside the offending face: material
    actually past the plane shares volume, a part resting flush does not: one
    question, one answer, for every layer that asks it.

    **Downgrade-only, exactly as the pair adjudicators are.** No interpreter, a
    part whose collision mesh is not staged, or a failed boolean all leave the box
    verdict standing, and `evidence.adjudication.available` says which happened.
    A pass can therefore never become a fail because of this pass.
    """
    ir_hash = ir["id"]
    room = ir.get("room")
    if not room:
        return VerifierReport(
            name=ROOM_CONTAINMENT, verdict="fail", ir_hash=ir_hash,
            evidence={"reason": "no_room_in_ir"},
            suggested_repairs=["call scene.set_room(...) before placing instances"],
        )
    # An uncapped room contains nothing, so there is nothing to check. For
    # `open_air` the room is an extent with a floor: a tree taller than the
    # declared height, or a path running past the declared width, is not
    # "outside" anything the scene has. `open_top` has walls, but no lid to
    # hold a tall instance in, and the box test cannot tell a lid from a wall.
    # Decided by the enclosure alone: a ceiling *replaced* with authored
    # geometry is still a lid, and replacements have their own exemption below.
    from ..sdk.scene_builder import room_surfaces_suppressed_by
    enclosure = room.get("enclosure")
    if "ceiling" in room_surfaces_suppressed_by(enclosure):
        return not_applicable(ROOM_CONTAINMENT, ir_hash,
                              f"uncapped room (enclosure {enclosure!r}); nothing "
                              "encloses the scene, so containment does not apply")
    room_aabb = _room_aabb(room)

    # Instances the scene has declared as shell replacements, via
    # `Scene.replace_room_surface`. A replacement wall is *supposed* to occupy
    # the plane of the shell surface it replaces, and a wall with real thickness
    # necessarily straddles it — so containment does not apply to it.
    #
    # **Only to it.** This is keyed off an explicit declaration, not off
    # "looks like a wall" or "overlaps the boundary", so furniture that escapes
    # the room is still the error it always was. The declaration also carries
    # its own cost: naming a surface deletes the shell's version of it, so a
    # scene cannot claim the exemption without giving up the surface.
    exempt = set((room.get("surface_replacements") or {}).values())

    # Pass 1 — the box broad phase. Nothing reaches CAD yet.
    requests = []          # (frame, instance, cands, poses, base_deltas)
    boxes = {}
    for inst in ir.get("instances", []):
        name = inst["name"]
        if name in exempt:
            continue
        bb = scene_query.instance_bbox(ir, ctx.catalog, name)
        cands = _room_escape_candidates(room_aabb, bb)
        if cands:
            boxes[name] = bb
            # `frame=None`: the scene layer has no frame grid, and the field is
            # only ever an opaque key in the returned dict.
            requests.append((None, name, cands, None, None))

    adjudication = None
    confirmed: dict = {}
    if requests:
        from . import _exact_adjudicator as _adj
        confirmed, adjudication = _adj.adjudicate_room_escapes(
            ir, ctx.catalog, requests,
            run_dir=getattr(ctx, "run_dir", None),
            asset_base_dir=getattr(
                getattr(ctx, "layer_payload", None), "asset_base_dir", None),
            cache=getattr(ctx, "adjudication_cache", None),
            # The same tolerance the broad phase flagged these with, so the
            # exact pass answers "is it outside by more than we already called
            # acceptable" rather than "is any vertex past the plane at all".
            slack=_CONTAINMENT_EPS_M)

    # Pass 2 — keep only what exact geometry stands behind, and prefer its
    # measurement: the box excursion is a bound on the real one, and an inflated
    # bound would tell the repair to move by more than the defect.
    out = []
    for _frame, name, cands, _poses, _deltas in requests:
        violations, escapes = {}, []
        for k, side, bound in cands:
            key = (None, name, k, side)
            if key not in confirmed:
                continue
            measured = confirmed[key]
            amount = bound if measured is None else measured
            violations[_VIOLATION_KEY[(k, side)]] = (
                -amount if side == "below" else amount)
            escapes.append({"axis": _AXIS[k], "side": side,
                            "amount_m": amount,
                            "exact": measured is not None})
        if violations:
            bb = boxes[name]
            out.append({
                "instance": name,
                "bbox": [list(bb[0]), list(bb[1])],
                "violations": violations,
                "escapes": escapes,
            })
    if not out:
        return VerifierReport(
            name=ROOM_CONTAINMENT, verdict="pass", ir_hash=ir_hash,
            evidence={"out_of_room": [], "adjudication": adjudication},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=ROOM_CONTAINMENT, verdict="fail", ir_hash=ir_hash,
        evidence={
            "out_of_room": out,
            "room_aabb": [list(room_aabb[0]), list(room_aabb[1])],
            "adjudication": adjudication,
        },
        suggested_repairs=[
            "`out_of_room[].escapes[]` identifies the violated axis and side. With "
            "`exact: true`, `amount_m` measures surface excursion past that room "
            "face; shifting inward by that amount clears that face at the checked "
            "pose. Recheck other room faces, support, and collisions after moving.",
            "Move the offending instance toward the room center along the violating axis.",
            "Increase the room dimensions if the scene fundamentally requires it.",
            "Re-anchor the instance to an interior anchor on its parent.",
            "With `exact: false`, `amount_m` is a bounding-box bound. Check "
            "`evidence.adjudication` and measure the real excursion before choosing a "
            "correction.",
        ],
    )


# ---------------------------------------------------------------------------
# SceneRelationalAssertions
# ---------------------------------------------------------------------------

SCENE_RELATIONAL_ASSERTIONS = "SceneRelationalAssertions"


def scene_relational_assertions(ir: dict, ctx) -> VerifierReport:
    ir_hash = ir["id"]

    def resolve(entity_name: str, anchor_name: str, poses=None,
                anchor_offset=None):
        return scene_query.anchor_position(
            ir, ctx.catalog, entity_name, anchor_name, poses=poses,
            anchor_offset=anchor_offset)

    def resolve_entity_aabb(entity_dict):
        return scene_query.resolve_entity_aabb(ir, ctx.catalog, entity_dict)

    def resolve_direction(direction_dict):
        return scene_query.resolve_direction(ir, ctx.catalog, direction_dict)

    if not ir.get("assertions"):
        return not_applicable(SCENE_RELATIONAL_ASSERTIONS, ir_hash, "no assertions declared")
    results, repairs, any_fail = evaluate_assertions(
        ir.get("assertions", []),
        resolve,
        resolve_entity_aabb_fn=resolve_entity_aabb,
        resolve_direction_fn=resolve_direction,
    )
    if not any_fail:
        return VerifierReport(
            name=SCENE_RELATIONAL_ASSERTIONS, verdict="pass", ir_hash=ir_hash,
            evidence={"results": results},
            suggested_repairs=[],
        )
    return VerifierReport(
        name=SCENE_RELATIONAL_ASSERTIONS, verdict="fail", ir_hash=ir_hash,
        evidence={"results": results},
        suggested_repairs=repairs,
    )


# ---------------------------------------------------------------------------
# SupportConnected (scene layer)
# ---------------------------------------------------------------------------

SUPPORT_CONNECTED = "SupportConnected"


def support_connected(ir: dict, ctx) -> VerifierReport:
    """Fail if any scene instance lacks a contact chain to the room.

    The room's floor, four walls, and ceiling all count as grounded
    surfaces. Wall-mounted shelves pass via lateral contact with a wall
    face. Standalone tables pass via floor contact. Stacked objects
    pass transitively (book → table → floor).

    **Contact is measured on surfaces.** A coplanar-AABB-face test misses every
    contact that is not axis-aligned: a ball tangent on an incline has no
    coplanar face with anything, so the scene would be told it is unsupported
    and could satisfy the check only by floating the ball above the rail. See `_support.build_scene_adjacency` for
    the measurement and for the one way this differs from the penetration
    adjudicators: it is not downgrade-only.
    """
    ir_hash = ir["id"]
    ok, unsupported, evidence = support_connected_scene(
        ir, ctx.catalog, scene_query,
        run_dir=getattr(ctx, "run_dir", None),
        asset_base_dir=getattr(
            getattr(ctx, "layer_payload", None), "asset_base_dir", None),
        cache=getattr(ctx, "adjudication_cache", None),
    )
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
            f"Unsupported instances: {unsupported}. No contact chain to the "
            "room (floor, walls, or ceiling) was found by the support checks.",
            "Inspect the missing links in the support graph and "
            "`evidence.exact_contact`. Measure actual gaps before changing "
            "placement; boxes alone can misrepresent contact on rotated or curved "
            "parts.",
            "Provide a contact chain to a room face or supported instance within the "
            "reported tolerance. Use the intended mounting surface; the object's "
            "named back/top face need not be the one facing a wall/ceiling.",
        ],
    )


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

#: The checks over placed instances, the part of `SCENE_VERIFIERS` that is not
#: an object's own: geometry only.
INSTANCE_VERIFIERS = (
    (NO_CROSS_INSTANCE_PENETRATION, no_cross_instance_penetration),
    (ROOM_CONTAINMENT, room_containment),
    (SUPPORT_CONNECTED, support_connected),
    (SCENE_RELATIONAL_ASSERTIONS, scene_relational_assertions),
)


#: Name of the check below, so a caller can select or skip it by name.
ANATOMICAL_JOINT_LIMITS = "AnatomicalJointLimits"


def anatomical_joint_limits(ir: dict, ctx=None) -> VerifierReport:
    """Committed poses of a preloaded asset, against its anatomical overlay.

    A vendor URDF states what the *machine* can do. For a humanoid used as a
    character that is not the same question as what it should be seen doing:
    the G1's own limits permit a 5 deg backward knee and a 60 deg backward
    elbow, and nothing downstream objects, because the mechanism really does
    move that way. An overlay narrows the ranges where exceeding them reads as
    a broken body, and this is what enforces it.

    **It reads the committed pose, not the reachable range.** Every joint of a
    placed instance has a value the scene committed — its `initial_joints`
    entry, else the definition's `default` — and that is the pose the scene
    ships, renders, and hands to an animation. Judging the asset's *range* here
    instead would fail every scene that places a G1 at all, since the hardware
    range is exactly what the overlay disagrees with.

    Absent an overlay this passes trivially, which is every run that preloads
    nothing. See `agent/assets/preload_asset.py` for the document, its
    only-ever-tighten rule, and where it is resolved from.
    """
    payload = getattr(ctx, "layer_payload", None)
    overlays = getattr(payload, "anatomical_limits", None) or {}
    if not overlays:
        return not_applicable(ANATOMICAL_JOINT_LIMITS, ir.get("id", ""),
                              "no placed asset carries an anatomical overlay")
    catalog = getattr(payload, "catalog", None) or {}
    offenders: list = []
    checked = 0
    for inst in ir.get("instances") or ():
        overlay = overlays.get(inst.get("object_ref"))
        if not overlay:
            continue
        obj = catalog.get(inst["object_ref"]) or {}
        defaults = {j["name"]: j.get("default", 0.0)
                    for j in obj.get("joints") or ()}
        committed = dict(defaults)
        committed.update(inst.get("initial_joints") or {})
        for jname, rule in (overlay.get("joints") or {}).items():
            if jname not in committed:
                continue
            checked += 1
            q = float(committed[jname])
            lo, hi = rule["limits"]
            if lo - 1e-9 <= q <= hi + 1e-9:
                continue
            # Reported in the surface unit the author wrote and reads.
            to_surface = ((lambda v: round(math.degrees(v), 3))
                          if rule["unit"] == "deg" else (lambda v: round(v, 6)))
            offenders.append({
                "instance": inst["name"], "joint": jname,
                "committed": to_surface(q), "allowed": rule["surface"],
                "unit": rule["unit"], "guards": rule.get("guards"),
                "object": obj.get("name"),
            })
    repairs = []
    if offenders:
        repairs = [
            "pose the joint inside its anatomical range with `initial_joints`",
            "if the pose is deliberate and the machine supports it, widen the "
            "overlay in catalogs/anatomical/ rather than working around it here",
        ]
    if not checked:
        return not_applicable(ANATOMICAL_JOINT_LIMITS, ir.get("id", ""),
                              "no placed instance has a joint its overlay limits")
    return VerifierReport(
        name=ANATOMICAL_JOINT_LIMITS,
        verdict="fail" if offenders else "pass",
        ir_hash=ir.get("id", ""),
        evidence={"assets_with_limits": len(overlays),
                  "joints_checked": checked,
                  **({"offenders": offenders} if offenders else {})},
        suggested_repairs=repairs)


#: The scene layer's suite: every placed object's own checks, then the checks
#: over instances, then the anatomical overlay check.
#:
#: No automatic image critique runs on compile. An expensive critique on every
#: compile re-derives from images what the author already holds as
#: measurements, and gates on a rubric it cannot see. The author buys a look deliberately, with
#: `review()`, which also reads the author's own `scene.add_check(...)` claims.
#:
#: Object checks run first so a defect is reported against the definition that
#: has it, before the scene-level checks report the same geometry as a
#: collision between instances.
SCENE_VERIFIERS = (PER_OBJECT_VERIFIERS + INSTANCE_VERIFIERS
                   + ((ANATOMICAL_JOINT_LIMITS, anatomical_joint_limits),))


__all__ = [
    "NO_CROSS_INSTANCE_PENETRATION", "no_cross_instance_penetration",
    "ROOM_CONTAINMENT", "room_containment",
    "SUPPORT_CONNECTED", "support_connected",
    "SCENE_RELATIONAL_ASSERTIONS", "scene_relational_assertions",
    "ANATOMICAL_JOINT_LIMITS", "anatomical_joint_limits",
    "INSTANCE_VERIFIERS", "SCENE_VERIFIERS",
]
