"""SupportConnected verifier kernel, shared by the object and scene layers.

Every entity (an object's part, a scene's instance) must have an undirected chain to a grounded node through:

  1. **Contacts**: parts whose surfaces are within `DYCRAFT_SUPPORT_CONTACT_TOL_M` (default 1 cm), measured
     by `_exact_adjudicator` with the AABBs as the broad phase. A pair nothing could measure falls back to the
     box test, two AABB faces coplanar within the tolerance (`aabbs_in_face_contact`).
  2. **Kinematic edges** (object layer only): a `fixed`, `revolute`, or `prismatic` joint whose pivot lies
     within `DYCRAFT_SUPPORT_JOINT_TOL_M` (default 1 cm) of both parts' boxes, as the point-to-AABB distance in
     each part's local frame, summed. This credits a hinge or slider with a small standoff, which does not
     touch its parent at a non-zero joint pose, and rejects a part held up on a long virtual offset.

Grounding depends on the layer:

  * Scene layer: the room (floor, walls, ceiling) is the grounded node. Nodes are instances, but contact is
    tested per part, so a recessed fixture grounds through its rim on a host's counter.
  * Object layer: the lowest `zmin` across the parts' world AABBs is the floor, and a part within the contact
    tolerance of it is grounded. The root part is not the floor: a human torso grounds through its feet.

Evaluated at the rest pose; NoSelfPenetration covers the other poses.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from ..sdk import _primitives as _prim


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def contact_tol_m() -> float:
    return _env_float("DYCRAFT_SUPPORT_CONTACT_TOL_M", 1.0e-2)


def joint_tol_m() -> float:
    """Tolerance for crediting a kinematic edge as a support edge.

    Default 1 cm: rejects parts suspended via large virtual offsets,
    admits hinges/sliders with small physical standoffs.
    """
    return _env_float("DYCRAFT_SUPPORT_JOINT_TOL_M", 1.0e-2)


def _joint_anchor_offset_distance(ir, joint) -> float:
    """Sum of per-anchor distances from the joint pivot to each part's box.

    The joint pivot sits at `parent_anchor` / `child_anchor`, whose
    stored SE3 translation IS the anchor's position in that part's local
    frame. We measure how far that point lies from the part's actual
    geometry — the point-to-AABB distance against the part's local box, 
    using its actual minimum and maximum bounds (which may be off-center for a mesh). 
    A point on or inside the box scores 0; a point off the box scores the
    Euclidean gap to the nearest surface.

    The total `pa_dist + ca_dist` is the rigid "stick length" gating
    kinematic-edge admission in support_connected: a joint whose pivot
    lands on both parts' surfaces is real face-contact-equivalent rigid
    support, while a large summed distance is a long virtual stick
    (e.g. a part suspended 1 m away via a fixed joint).

    Note this is distance to the *geometry*, not to the 27 sparse AABB
    landmarks — an anchor on the middle of a large face is on the box
    (distance 0) even though it may be far from any named landmark.
    """
    parts_by_name = {p["name"]: p for p in ir.get("parts", ())}

    def _per_anchor_offset(anchor_record, part_name) -> float:
        part = parts_by_name.get(part_name)
        if part is None or not _prim.is_known(part.get("primitive")):
            return 0.0
        lo, hi = _prim.local_aabb(part["primitive"], part["params"])
        # stored_t is the anchor's position in the part's local frame. Measure
        # the point-to-AABB distance against the part's actual local box
        # [lo, hi] per axis. For an origin-centered analytic primitive this is
        # exact; for a mesh whose AABB may
        # be off-center it correctly uses the true min/max rather than assuming
        # the box straddles the origin. Round shapes remain conservative (to
        # their enclosing box).
        stored_t = tuple(float(v) for v in anchor_record["translation"])
        sq = 0.0
        for coord, lo_i, hi_i in zip(stored_t, lo, hi):
            excess = max(lo_i - coord, coord - hi_i, 0.0)  # >0 only outside
            if excess > 0.0:
                sq += excess * excess
        return sq ** 0.5

    pa = joint["parent_anchor"]; ca = joint["child_anchor"]
    pa_off = _per_anchor_offset(pa, joint["parent"])
    ca_off = _per_anchor_offset(ca, joint["child"])
    return pa_off + ca_off


# ---------------------------------------------------------------------------
# AABB face-contact predicate
# ---------------------------------------------------------------------------

def aabbs_in_face_contact(aabb_a, aabb_b, tol: float) -> bool:
    """True iff aabb_a and aabb_b share at least one coplanar-face contact.

    A face contact exists on axis `c` if the +c face of one AABB and the
    -c face of the other are coplanar within `tol`, AND the overlap on
    the other two axes is positive (strict inequality on at least one
    axis avoids "kissing at a single edge" false positives — though we
    allow zero on one axis to admit edge-on-face contacts).
    """
    (axmn, aymn, azmn), (axmx, aymx, azmx) = aabb_a
    (bxmn, bymn, bzmn), (bxmx, bymx, bzmx) = aabb_b

    # X axis face pairs
    for ap, bp in ((axmx, bxmn), (axmn, bxmx)):
        if abs(ap - bp) <= tol:
            ulen = min(aymx, bymx) - max(aymn, bymn)
            vlen = min(azmx, bzmx) - max(azmn, bzmn)
            if ulen > -tol and vlen > -tol and (ulen > 0 or vlen > 0):
                return True
    # Y axis
    for ap, bp in ((aymx, bymn), (aymn, bymx)):
        if abs(ap - bp) <= tol:
            ulen = min(axmx, bxmx) - max(axmn, bxmn)
            vlen = min(azmx, bzmx) - max(azmn, bzmn)
            if ulen > -tol and vlen > -tol and (ulen > 0 or vlen > 0):
                return True
    # Z axis
    for ap, bp in ((azmx, bzmn), (azmn, bzmx)):
        if abs(ap - bp) <= tol:
            ulen = min(axmx, bxmx) - max(axmn, bxmn)
            vlen = min(aymx, bymx) - max(aymn, bymn)
            if ulen > -tol and vlen > -tol and (ulen > 0 or vlen > 0):
                return True
    return False


# ---------------------------------------------------------------------------
# BFS over the connectivity graph
# ---------------------------------------------------------------------------

class _Adjacency(NamedTuple):
    """Undirected adjacency representation used by the BFS.

    `nodes`: list of node names (instances or parts) plus a sentinel
             grounded node "__floor__".
    `edges`: dict[name -> set[name]]. Edges are bidirectional.
    `exact`: what the exact contact pass did, or None when it did not run.
             Carried so the verifier's evidence can say whether an edge came
             from a surface or from a box — they are different claims, and a
             reader who cannot tell has no way to weigh an `unsupported` row.
    """
    nodes: list
    edges: dict
    exact: dict | None = None


def _exact_scene_contacts(ir, catalog, tol, candidates, *, run_dir,
                          asset_base_dir, cache, stats):
    """Which candidate part pairs are within `tol` of each other, on surfaces.

    `candidates` is `[(inst_a, part_a, inst_b, part_b), ...]` already filtered
    by the box broad phase. Returns the subset whose **surfaces** are within
    `tol`, plus the pairs nothing could answer about, which the caller decides
    by box as before.

    **Two passes, and the first one's return value is discarded.**
    `exact_part_gap` reduces its pairs to a minimum, so it cannot report per
    pair — but it batches every query into one answer round and writes each
    record into the shared `cache`, so the per-pair calls that follow are pure
    lookups: the price of exact geometry is the round trip, not the query.
    """
    from . import _exact_adjudicator as _adj

    kw = dict(run_dir=run_dir, asset_base_dir=asset_base_dir, cache=cache)
    requests = [(ia, pa, ib, pb, None, None) for ia, pa, ib, pb in candidates]
    _gap, batch = _adj.exact_part_gap(ir, catalog, requests, **kw)
    stats["backend"] = batch.get("backend")
    stats["available"] = batch.get("available", True)
    if not stats["available"]:
        # Nothing was measured, so every pair falls back to its box verdict —
        # which is what this check has always used. Recorded rather than
        # silent: an edge from a box is a weaker claim than one from a surface.
        return set(), set(candidates)

    touching, unanswered = set(), set()
    for pair in candidates:
        ia, pa, ib, pb = pair
        gap, st = _adj.exact_part_gap(
            ir, catalog, [(ia, pa, ib, pb, None, None)], **kw)
        if gap is None or st.get("not_adjudicable"):
            unanswered.add(pair)
            continue
        stats["measured"] = stats.get("measured", 0) + 1
        if gap <= tol:
            touching.add(pair)
    stats["unanswered"] = len(unanswered)
    return touching, unanswered


def _grounded_on_a_room_face(ir, catalog, instance, tol, *, exact, run_dir,
                             asset_base_dir, cache, stats):
    """Does any part of `instance` reach a room face, measured? None if unknown.

    `exact_room_gap` answers per face against the real surface — a room face is
    an axis-aligned plane, so this is the one exact query in the whole collision
    surface with no tessellation term at all.

    **A gap under `tol` only grounds the instance when it was measured.** The
    fallback number is the world-AABB distance to the face, and a box contains
    its part, so that number is a *lower* bound: under `tol` it proves nothing
    about contact, exactly as `_instance_gap`'s `slack=None` case does one layer
    up. Over `tol` it is decisive in the other direction, and that is reported
    as "not grounded here" rather than as unknown.
    """
    if not exact:
        return None
    from . import _exact_adjudicator as _adj
    try:
        per_face = _adj.exact_room_gap(
            ir, catalog, instance, None, None, faces="all", tol=tol,
            run_dir=run_dir, asset_base_dir=asset_base_dir, cache=cache)
    except Exception:               # noqa: BLE001 — a verifier must not raise
        stats["unanswered"] += 1
        return None
    seen_answer = False
    for rec in (per_face or {}).values():
        gap = rec.get("gap")
        if gap is None:
            continue
        if rec.get("exact"):
            seen_answer = True
            if gap <= tol:
                stats["measured"] += 1
                return True
        elif gap <= tol:
            # A lower bound inside the band: cannot prove contact, and cannot
            # disprove it either. Hand the whole instance back as unknown.
            stats["unanswered"] += 1
            return None
    if seen_answer:
        stats["measured"] += 1
        return False
    stats["unanswered"] += 1
    return None


class _DSU:
    """Union-find over the graph's nodes.

    Here to answer one question cheaply: *could measuring this pair change
    anything?* It cannot when both endpoints already sit in one component,
    because components only ever grow — so an edge inside one is inside it
    forever. That makes skipping such a pair lossless for the verdict rather
    than an approximation, which is what lets the exact contact pass stop
    early (see `build_object_adjacency`).

    Components, not "already reached from the floor": two ungrounded parts can
    still be worth linking, because one of them may reach the floor later. The
    component test does not depend on the order pairs arrive in; a
    reached-from-floor test does.
    """

    __slots__ = ("_parent",)

    def __init__(self, nodes) -> None:
        self._parent = {n: n for n in nodes}

    def find(self, x):
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path halving
            x = parent[x]
        return x

    def union(self, a, b) -> bool:
        """Merge, returning whether this edge actually joined two components."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True

    def connected(self, a, b) -> bool:
        return self.find(a) == self.find(b)


def _bfs_from_floor(adj: _Adjacency) -> set:
    """Return the set of nodes reachable from the synthetic '__floor__'
    node via the contact graph.
    """
    if "__floor__" not in adj.edges:
        return set()
    seen = {"__floor__"}
    frontier = ["__floor__"]
    while frontier:
        nxt = []
        for n in frontier:
            for m in adj.edges.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    return seen


# ---------------------------------------------------------------------------
# Object-layer support
# ---------------------------------------------------------------------------

def _object_part_world_aabbs(ir, object_query) -> dict:
    """Map part-name -> world AABB at rest pose."""
    out = {}
    for p in ir["parts"]:
        out[p["name"]] = object_query.part_world_aabb(ir, p["name"])
    return out


def _global_zmin(aabbs: dict) -> float:
    return min(aabb[0][2] for aabb in aabbs.values())


#: How many pairs the lazy contact pass measures per wave.
#:
#: One pair at a time would stop earliest, but `exact_object_part_gaps`
#: amortizes B-rep loading and interpreter setup across the list it is given,
#: so a per-pair loop hands back much of what the early exit wins. A wave is
#: the compromise: measure a few, fold in what they linked, re-filter the rest.
#: On an object whose joint tree already grounds every part — the common case —
#: the first wave is empty and none of this runs.
CONTACT_WAVE = 12


def _aabb_separation(aabb_a, aabb_b) -> float:
    """Summed per-axis gap between two boxes; 0 when they overlap.

    Orders the lazy contact pass. A pair of flush neighbors is both the
    likeliest to link and the cheapest way to merge two components, so
    measuring those first is what usually makes the second wave empty.
    """
    (alo, ahi), (blo, bhi) = aabb_a, aabb_b
    return sum(max(0.0, alo[k] - bhi[k], blo[k] - ahi[k]) for k in range(3))


#: Stats keys the running total must not accumulate.
#:
#: `considered` is the broad phase's output, fixed before any wave runs;
#: summing each wave's slice of it would report a number smaller than the real
#: candidate count and shrink as the early exit got better. `available` and
#: `backend` describe the adjudicator, not the work — and `available` is a
#: `bool`, which is an `int` in Python, so a plain numeric merge turns two
#: waves of `True` into `2`.
_EXACT_STATS_NOT_SUMMED = frozenset({"considered", "available", "backend",
                                     "complete", "settled_without_measuring"})


def _merge_exact_stats(into: dict, more: dict | None) -> dict:
    """Accumulate one wave's adjudicator stats into the running total."""
    if not more:
        return into
    for key, value in more.items():
        if key in _EXACT_STATS_NOT_SUMMED:
            into[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and isinstance(into.get(key), (int, float)):
            into[key] += value
        else:
            into[key] = value
    return into


def build_object_adjacency(ir, object_query, tol: float,
                           joint_tol: float | None = None, *,
                           run_dir=None, cache=None,
                           exact: bool = True,
                           complete_contacts: bool = False) -> _Adjacency:
    """Build the contact graph for an object's parts.

    Nodes are every part plus the synthetic "__floor__", the lowest-z plane across the parts at rest. Edges:
      * Floor edges: a part whose world AABB `zmin` is within `tol` of the floor.
      * Kinematic edges: a joint whose pivot is within `joint_tol` (default `joint_tol_m()`) of both parts' boxes,
        summed (`_joint_anchor_offset_distance`).
      * Part-part contacts within `tol`, **measured on the B-reps** by `_exact_adjudicator.exact_object_part_gaps`,
        with the AABBs as the broad phase. A coplanar-AABB-face test would miss every contact that is not
        axis-aligned (a shaft in a saddle, a rounded foot on a plate) and invent contact between boxes whose solids
        are apart; it decides only a pair nothing could measure.

    **The two free edge sources are built first, and the measured one stops once it cannot change the answer.**
    Support is reachability, so a measured pair whose endpoints already share a component cannot move the verdict.
    Floor and kinematic edges cost box arithmetic; an exact contact costs a surface-distance query, up to 1.5 s a
    pair on spline geometry. An articulated object has a joint tree, so it is usually connected before any geometry
    is touched, and this order is lossless (`_DSU` says why).

    `complete_contacts=True` measures every candidate pair regardless, for `object_support_report`: that probe
    answers "which parts touch", for which a spanning subset is the wrong answer. The verifier leaves it False and
    records `evidence.exact_contact["complete"]`.

    **The floor datum is a box.** Grounding is a z-extremum question, not a pair question, and
    `exact_object_part_gaps` answers pairs, so a part whose box dips below its surface can ground on a datum its
    geometry does not reach. Measuring it needs a per-part exact z-extremum, a query nothing here makes.
    """
    aabbs = _object_part_world_aabbs(ir, object_query)
    if not aabbs:
        return _Adjacency(nodes=["__floor__"], edges={"__floor__": set()})

    joint_tol_f = float(joint_tol if joint_tol is not None else joint_tol_m())
    floor_z = _global_zmin(aabbs)
    names = list(aabbs.keys())
    nodes = ["__floor__"] + names
    edges: dict = {n: set() for n in nodes}
    dsu = _DSU(nodes)

    def link(a, b) -> None:
        edges[a].add(b)
        edges[b].add(a)
        dsu.union(a, b)

    # 1. Floor edges: anything with zmin within tol of the global zmin.
    for name, aabb in aabbs.items():
        if abs(aabb[0][2] - floor_z) <= tol:
            link("__floor__", name)

    # 2. Kinematic edges: joints whose pivot lies on (or within joint_tol
    # of) both parts' box geometry represent real rigid contact (hinges,
    # sliders, features mounted anywhere on a face). The anchor SE3
    # stored in the joint record IS the pivot's position in each part's
    # local frame; `_joint_anchor_offset_distance` measures the
    # point-to-AABB distance to each part's box and sums them. A large
    # total models a long rigid stick — geometrically present but not a
    # face-contact support, so the kinematic edge is not admitted.
    #
    # Before the measured pass, not after: a joint edge is free, and having it
    # in place is what lets the expensive query stop early.
    for j_rec in ir.get("joints", ()):
        a_name = j_rec.get("parent")
        b_name = j_rec.get("child")
        if a_name not in aabbs or b_name not in aabbs:
            continue
        if _joint_anchor_offset_distance(ir, j_rec) <= joint_tol_f:
            link(a_name, b_name)

    # 3. Part-part contact edges. The boxes are the broad phase — sound because
    # a box contains its part — and **not** `aabbs_in_face_contact`, which is
    # not a superset of what the exact test accepts: it demands coplanarity,
    # and a tangent or corner contact has none, so using it here would filter
    # out the pairs this measurement exists to find.
    candidates = [(names[i], names[j])
                  for i in range(len(names))
                  for j in range(i + 1, len(names))
                  if _aabbs_within_tol(aabbs[names[i]], aabbs[names[j]], tol)]

    def _settle(pair, gap) -> None:
        if gap is not None:
            linked = gap <= tol
        else:
            # Unanswered: keep the box verdict this check has always used.
            linked = aabbs_in_face_contact(aabbs[pair[0]], aabbs[pair[1]], tol)
        if linked:
            link(*pair)

    stats: dict | None = None
    if candidates and exact:
        from . import _exact_adjudicator as _adj
        # Seeded rather than taken from the adjudicator, because the lazy pass
        # may never call it: an object whose joint tree already grounds every
        # part measures nothing, and reporting no `exact_contact` block at all
        # would read as "the exact pass was unavailable" instead of "it had
        # nothing left to decide".
        stats = {"considered": len(candidates), "sent": 0, "cache_hits": 0,
                 "not_adjudicable": 0, "available": None, "backend": None,
                 "complete": bool(complete_contacts),
                 "settled_without_measuring": False}
        if complete_contacts:
            gaps, wave_stats = _adj.exact_object_part_gaps(
                ir, candidates, run_dir=run_dir, cache=cache)
            _merge_exact_stats(stats, wave_stats)
            for pair, gap in zip(candidates, gaps or [None] * len(candidates)):
                _settle(pair, gap)
        else:
            # Closest boxes first: the likeliest to link, so the likeliest to
            # empty the queue in one wave.
            pending = sorted(
                (p for p in candidates if not dsu.connected(*p)),
                key=lambda p: _aabb_separation(aabbs[p[0]], aabbs[p[1]]))
            floor_root = dsu.find("__floor__")
            while pending and any(dsu.find(n) != floor_root for n in names):
                wave, pending = pending[:CONTACT_WAVE], pending[CONTACT_WAVE:]
                gaps, wave_stats = _adj.exact_object_part_gaps(
                    ir, wave, run_dir=run_dir, cache=cache)
                _merge_exact_stats(stats, wave_stats)
                for pair, gap in zip(wave, gaps or [None] * len(wave)):
                    _settle(pair, gap)
                floor_root = dsu.find("__floor__")
                pending = [p for p in pending if not dsu.connected(*p)]
        stats["considered"] = len(candidates)
        stats["complete"] = bool(complete_contacts)
        stats["settled_without_measuring"] = stats["backend"] is None
    elif candidates:
        for pair in candidates:
            _settle(pair, None)

    return _Adjacency(nodes=nodes, edges=edges, exact=stats)


def support_connected_object(ir, object_query, tol: float | None = None,
                              joint_tol: float | None = None, *,
                              run_dir=None, cache=None,
                              exact: bool = True):
    """Returns (verdict, unsupported_names, evidence_dict).

    `verdict` is True iff every part reaches the floor via the contact
    graph (contacts + kinematic edges within tolerances).
    `unsupported_names` lists the parts that don't.

    `run_dir` / `cache` reach the exact contact pass. Without `run_dir` a
    `solid` part's staged `.brep` cannot be resolved, so every pair falls back
    to its bounding box; `evidence.exact_contact` reports what happened.
    """
    tol_f = float(tol if tol is not None else contact_tol_m())
    joint_tol_f = float(joint_tol if joint_tol is not None else joint_tol_m())
    if not ir.get("parts"):
        # `exact_contact` present-and-None here too: a consumer reading the
        # evidence must not have to special-case the empty object.
        return True, [], {"reason": "empty", "tol_m": tol_f,
                          "joint_tol_m": joint_tol_f,
                          "exact_contact": None}
    adj = build_object_adjacency(ir, object_query, tol_f, joint_tol_f,
                                 run_dir=run_dir, cache=cache, exact=exact)
    reached = _bfs_from_floor(adj)
    part_names = [p["name"] for p in ir["parts"]]
    unsupported = [n for n in part_names if n not in reached]
    evidence = {
        "tol_m": tol_f,
        "joint_tol_m": joint_tol_f,
        "floor_z": _global_zmin(_object_part_world_aabbs(ir, object_query)),
        "grounded_directly": sorted(adj.edges["__floor__"]),
        "supported_count": len(reached) - 1,  # excl. "__floor__"
        "unsupported": unsupported,
        # An edge from a surface and an edge from a box are different claims.
        "exact_contact": adj.exact,
    }
    return not unsupported, unsupported, evidence


# ---------------------------------------------------------------------------
# Scene-layer support
# ---------------------------------------------------------------------------

def _scene_instance_part_aabbs(ir, catalog, scene_query) -> dict:
    """Map instance-name -> list of (part_name, world AABB) at rest pose."""
    out = {}
    for inst in ir.get("instances", []):
        out[inst["name"]] = scene_query.instance_part_world_aabbs(
            ir, catalog, inst["name"])
    return out


def _union_aabb(aabbs):
    """Union AABB over a non-empty iterable of AABBs."""
    aabbs = list(aabbs)
    return (
        (min(a[0][0] for a in aabbs),
         min(a[0][1] for a in aabbs),
         min(a[0][2] for a in aabbs)),
        (max(a[1][0] for a in aabbs),
         max(a[1][1] for a in aabbs),
         max(a[1][2] for a in aabbs)),
    )


def _aabbs_within_tol(a, b, tol: float) -> bool:
    """Broad-phase: True iff the two AABBs overlap or are within `tol` on
    every axis.

    A necessary condition for any face contact between parts contained in
    these boxes, so instances whose union boxes are nowhere near each
    other skip the O(parts x parts) part-pair loop.
    """
    (a0, a1) = a
    (b0, b1) = b
    for k in range(3):
        if a1[k] < b0[k] - tol or b1[k] < a0[k] - tol:
            return False
    return True


def _room_aabb_faces(ir):
    """Return the 6 face descriptors of the room interior.

    Each face is `(axis_idx, plane_value, sign)`:
      axis_idx in {0, 1, 2}, sign in {+1, -1}, plane_value is the
      world coordinate where the wall sits.
    """
    room = ir.get("room") or {}
    w = float(room.get("width", 0.0))
    d = float(room.get("depth", 0.0))
    h = float(room.get("height", 0.0))
    return [
        (0, +w / 2.0, +1),  # +x wall ("left" in dycraft convention)
        (0, -w / 2.0, -1),  # -x wall ("right")
        (1, +d / 2.0, +1),  # +y wall ("front")
        (1, -d / 2.0, -1),  # -y wall ("back")
        (2, 0.0,      -1),  # floor
        (2, h,        +1),  # ceiling
    ]


def _instance_touches_room_face(inst_aabb, face, tol: float) -> bool:
    """True iff the instance's outermost face on `axis_idx` in the `sign`
    direction is coplanar with the room face within `tol`.

    Concretely: for the +x wall (axis=0, sign=+1) the instance's `xmax`
    must equal the wall's plane within tol. For the floor
    (axis=2, sign=-1) the instance's `zmin` must equal 0 within tol.
    """
    axis_idx, plane, sign = face
    if sign > 0:
        coord = inst_aabb[1][axis_idx]  # outermost in +axis direction
    else:
        coord = inst_aabb[0][axis_idx]
    return abs(coord - plane) <= tol


def build_scene_adjacency(ir, catalog, scene_query, tol: float, *,
                          run_dir=None, asset_base_dir=None,
                          cache=None, exact: bool = True) -> _Adjacency:
    """Build the contact graph for a scene's instances + room.

    Nodes are **instances**, each a rigid placement of an object that already passed object-layer
    SupportConnected, so its parts need no edges among themselves. The synthetic `__floor__` node is the room:
    contact with its floor, walls, or ceiling connects an instance to it. Edges are found at **part granularity**,
    so a recessed fixture grounds through its rim part resting on a host's counter part even though the
    whole-object union boxes share no face.

    **Contact is decided on surfaces, with the boxes as the broad phase**: `surface_gap <= tol` between parts, and
    the measured gap to each room face for grounding. A box test alone (`aabbs_in_face_contact`, two AABB faces
    coplanar within `tol`) is wrong in both directions. It invents contact between boxes whose solids are apart,
    and it misses contact that is not axis-aligned: a ball tangent partway up an inclined deck shares no coplanar
    box face with it, so the only pose that would pass leaves the ball floating. Unlike the penetration
    adjudicators, which can only clear, this can turn a pass into a fail, because a spurious box edge is what held
    a scene together; that is the correct verdict.

    A pair the exact pass cannot answer keeps its box verdict, and `exact["unanswered"]` counts them.
    """
    parts_by_inst = _scene_instance_part_aabbs(ir, catalog, scene_query)
    nodes = ["__floor__"] + list(parts_by_inst.keys())
    edges: dict = {n: set() for n in nodes}

    union_by_inst = {
        name: _union_aabb([a for _, a in parts])
        for name, parts in parts_by_inst.items() if parts
    }

    faces = _room_aabb_faces(ir)
    room_stats: dict = {"measured": 0, "unanswered": 0}
    for name, parts in parts_by_inst.items():
        grounded = _grounded_on_a_room_face(
            ir, catalog, name, tol, exact=exact, run_dir=run_dir,
            asset_base_dir=asset_base_dir, cache=cache, stats=room_stats)
        if grounded is None:
            # Nothing measurable: keep the box verdict this check has always
            # used. `_instance_touches_room_face` compares the instance's
            # outermost box coordinate against the wall plane, which for a
            # rotated round part sits outside the surface — so it can call a
            # floating body grounded. Conservative for *this* check, which
            # passes on more edges, and counted so the evidence can say so.
            grounded = any(
                _instance_touches_room_face(aabb, face, tol)
                for (_, aabb) in parts
                for face in faces
            )
        if grounded:
            edges["__floor__"].add(name)
            edges[name].add("__floor__")

    # Pass 1 — the box broad phase, per part pair. Sound because a box contains
    # its part: a pair whose boxes are more than `tol` apart on some axis has
    # surfaces at least that far apart, so the exact pass could not link it.
    # **Not `aabbs_in_face_contact`**, which is not a superset of what the exact
    # test accepts — it demands coplanarity, and a corner or tangent contact has
    # none. Using it here would filter out exactly the pairs this change exists
    # to find.
    names = list(parts_by_inst.keys())
    candidates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a not in union_by_inst or b not in union_by_inst:
                continue
            if not _aabbs_within_tol(union_by_inst[a], union_by_inst[b], tol):
                continue
            for pa_name, pa in parts_by_inst[a]:
                for pb_name, pb in parts_by_inst[b]:
                    if _aabbs_within_tol(pa, pb, tol):
                        candidates.append((a, pa_name, b, pb_name))

    # Pass 2 — surfaces decide, for the pairs a box could not settle.
    # `{instance: {part: box}}` — N*P entries, indexed per candidate below.
    # The first draft of this built the full `(instance, part) x (instance,
    # part)` cross product eagerly: 90 000 entries for a 30-instance scene of
    # 10-part objects, on every call, to serve at most one lookup per candidate.
    box_of = {inst: dict(parts) for inst, parts in parts_by_inst.items()}
    stats: dict | None = {"room": room_stats} if exact else None
    touching, unanswered = set(), set(candidates)
    if candidates and exact:
        stats["candidates"] = len(candidates)
        touching, unanswered = _exact_scene_contacts(
            ir, catalog, tol, candidates, run_dir=run_dir,
            asset_base_dir=asset_base_dir, cache=cache, stats=stats)

    for pair in candidates:
        a, pa_name, b, pb_name = pair
        if pair in touching:
            linked = True
        elif pair in unanswered:
            linked = aabbs_in_face_contact(
                box_of[a][pa_name], box_of[b][pb_name], tol)
        else:
            continue                    # measured, and further apart than tol
        if linked:
            edges[a].add(b)
            edges[b].add(a)

    return _Adjacency(nodes=nodes, edges=edges, exact=stats)


def support_connected_scene(ir, catalog, scene_query,
                            tol: float | None = None, *,
                            run_dir=None, asset_base_dir=None, cache=None,
                            exact: bool = True):
    """Returns (verdict, unsupported_names, evidence_dict).

    `run_dir` / `asset_base_dir` / `cache` reach the exact contact pass; without
    them it cannot resolve a part's staged geometry and every pair falls back to
    its bounding box, which `evidence.exact_contact.available` reports.
    """
    tol_f = float(tol if tol is not None else contact_tol_m())
    if not ir.get("instances"):
        return True, [], {"reason": "empty", "tol_m": tol_f,
                          "exact_contact": None}
    adj = build_scene_adjacency(ir, catalog, scene_query, tol_f,
                                run_dir=run_dir,
                                asset_base_dir=asset_base_dir,
                                cache=cache, exact=exact)
    reached = _bfs_from_floor(adj)
    inst_names = [i["name"] for i in ir["instances"]]
    unsupported = [n for n in inst_names if n not in reached]
    evidence = {
        "tol_m": tol_f,
        "grounded_directly": sorted(adj.edges["__floor__"]),
        "supported_count": len(reached) - 1,
        "unsupported": unsupported,
        # What decided the edges. An edge from a surface and an edge from a box
        # are different claims, and an `unsupported` row is only as strong as
        # the pass that produced it.
        "exact_contact": adj.exact,
    }
    return not unsupported, unsupported, evidence
