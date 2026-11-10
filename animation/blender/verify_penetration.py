"""NoPenetration's evidence: overlaps grouped and ranked, their separating shift and depth, allowance overruns."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .continuity import frame_runs
    from .evaluation import load_frame
    from .verify_room import ROOM_SURFACE_TAGS, out_of_room
except ImportError:  # Read-only modules staged into isolated Blender.
    from continuity import frame_runs
    from evaluation import load_frame
    from verify_room import ROOM_SURFACE_TAGS, out_of_room


def penetration_pairs(intervals):
    """Collision intervals grouped by the pair that collided, worst first.

    One interval per contiguous run already, but a pair that dips in and out
    produces many, and the flat list repeats both identities every time. The
    grouped form is what every consumer rebuilt by hand before reading it.

    No bounding-box quantity leaves this function. `separating_shift_m` is
    `directional_separation_m`: the least translation that separates the pair
    over the axis bound and the most promising contact normals, each bisected
    with Coal queries on the evaluated meshes. It is an upper bound on the
    true minimum translation, tight whenever the contact normal is the
    direction the overlap has depth along, which is the ordinary case.

    **It is not a depth.** A shift is what it takes to pull the whole part
    clear, which for an interlocked part is the length of the interlock rather
    than how far the surfaces cross. `max_depth_m` is the depth, measured
    separately (see `deepest_vertex_m`).

    `separating_direction` is the direction that won that search, and it is
    reported because the pair is now a verified fact rather than a bookkeeping
    detail: moving `b` by `separating_shift_m` along it separates the pair at
    `representative_frame`, because that is precisely what the bisection
    confirmed. It moves **b away from a**, in that order.
    """
    grouped = {}
    for entry in intervals:
        grouped.setdefault((entry["a"], entry["b"], entry["kind"]), []).append(entry)
    pairs = []
    for (a, b, kind), entries in grouped.items():
        entries.sort(key=lambda e: e["start_frame"])
        spans = []
        for entry in entries:
            measured = entry.get("directional_separation_m")
            span = {"frames": [entry["start_frame"], entry["end_frame"]],
                    "representative_frame": entry["representative_frame"]}
            if measured is not None and entry.get("separation_direction") is not None:
                span["separating_direction"] = [round(float(v), 6)
                                                for v in entry["separation_direction"]]
            if measured is None:
                # No candidate direction came apart inside its bracket, which
                # non-convex geometry can cause by re-engaging after it
                # separates. The overlap is still real; it just has no
                # measured size, and no bounding-box number is substituted
                # for one, since it would read as a depth. The frames and a
                # render carry the severity instead.
                span["penetration_unresolved"] = True
            else:
                shift = round(float(measured), 6)
                span["separating_shift_m"] = shift
            if entry.get("max_depth_m") is not None:
                span["max_depth_m"] = round(float(entry["max_depth_m"]), 6)
                if entry.get("max_depth_frame") is not None:
                    span["max_depth_frame"] = entry["max_depth_frame"]
            spans.append(span)
        resolved = [s["separating_shift_m"] for s in spans if "separating_shift_m" in s]
        depths = [s["max_depth_m"] for s in spans if "max_depth_m" in s]
        pair = {"a": a, "b": b, "kind": kind,
                "frame_count": sum(e["end_frame"]-e["start_frame"]+1 for e in entries),
                "spans": spans}
        if resolved:
            pair["worst_separating_shift_m"] = max(resolved)
        if depths:
            pair["max_depth_m"] = max(depths)
        pairs.append(pair)
    # Longest first, then by identity. Both recorded agents ranked these by
    # time in contact, and one added the pair name as a tie-break; without it
    # two pairs that tie — common, since an unresolved pair has no shift to
    # separate them — come back in dict order, and an author diffing one
    # report against the next reads the reshuffle as a change. Depth is the
    # other question an author asks, and it gets its own list
    # (`pairs_by_depth`) rather than displacing this order.
    pairs.sort(key=lambda pair: (-pair["frame_count"],
                                 -pair.get("worst_separating_shift_m", 0.),
                                 pair["a"], pair["b"]))
    return pairs


#: How many pairs `pairs_by_depth` names. It is an index into `pairs`, which
#: carries every pair in full, so the head is the part worth ranking.
DEPTH_RANKING_LIMIT = 12

#: How many frames of each overlap span `max_depth_m` is measured on: those
#: with the largest bounding-box bound, which is every frame of a span this
#: short or shorter. The bound that picks them is box slack rather than depth,
#: so one frame would bet the depth on one sample; three keeps the cost bounded
#: per span.
DEPTH_FRAMES = 3


def pairs_by_depth(pairs):
    """The deepest overlaps first, as `{a, b, max_depth_m}` pointers into `pairs`.

    `pairs` stays in contact-time order. "Which overlap is deepest" and "which
    lasts longest" are different questions: a 60 mm crossing held for three
    frames belongs at the top of one list and the bottom of the other.

    A pair with no measured depth is listed after the ranked ones as
    `max_depth_m: None, unresolved: True`, never ranked at zero and never
    dropped, because an unmeasured overlap can be a deep one.
    """
    ranked = sorted((pair for pair in pairs if "max_depth_m" in pair),
                    key=lambda pair: (-pair["max_depth_m"], pair["a"], pair["b"]))
    unresolved = [pair for pair in pairs if "max_depth_m" not in pair]
    return ([{"a": pair["a"], "b": pair["b"], "max_depth_m": pair["max_depth_m"]}
             for pair in ranked[:DEPTH_RANKING_LIMIT]]
            + [{"a": pair["a"], "b": pair["b"], "max_depth_m": None, "unresolved": True}
               for pair in unresolved[:DEPTH_RANKING_LIMIT]])


#: Most inside-tests `deepest_vertex_m` spends per direction. Each is a few
#: ray casts, and candidates are tried deepest-ceiling first, so the budget
#: only runs out on a part whose box encloses a lot of empty space (a hand in
#: a mug's cavity).
_DEPTH_INSIDE_BUDGET = 256


def deepest_vertex_m(a, b):
    """How far the deepest vertex of either mesh lies inside the other.

    Distance is to the other mesh's triangle *surface* (a BVH nearest-point
    query), not to its nearest vertex: one analysis of this layer measured a
    nearest-vertex depth at 6.62 mm where the triangle answer was 0.37 mm, an
    18x error in exactly the direction that re-creates the problem this
    field exists to fix. Inside-ness is `Body.inside`, the same winding test
    containment uses, so only closed shells can report a depth.

    This is a **vertex** depth. Two bars crossing like a plus sign have no
    vertex inside each other and report no depth, although they overlap;
    that pair keeps its `separating_shift_m` and simply does not appear in
    `pairs_by_depth`. Returns `None` when neither direction found a vertex
    inside, which includes that case and open sheets. If the inside-test
    budget runs out the deepest vertex found so far is returned, which is a
    lower bound — the direction a ranking can afford to err in.

    Works on `Body` and `PlacedBody` alike: both carry a rigid `matrix` over a
    `local` body, and the query runs in the target's local frame, where its
    BVH already is.
    """
    best = None
    for source, target in ((a, b), (b, a)):
        local = target.local
        if not any(closed for _, closed in local.components):
            continue
        rotation, origin = target.matrix[:3, :3], target.matrix[:3, 3]
        points = (np.asarray(source.vertices, dtype=float) - origin) @ rotation
        points = points[np.all((points >= local.lo) & (points <= local.hi), axis=1)]
        if not len(points):
            continue
        # A point inside a solid is no further from its surface than from its
        # bounding box, which is arithmetic. Walking points in that order is a
        # branch and bound: once the deepest vertex found so far is at least
        # the next point's box ceiling, nothing left can beat it, so the usual
        # case — the first deep point is inside — stops almost at once.
        ceiling = np.minimum(points - local.lo, local.hi - points).min(axis=1)
        floor = 0. if best is None else best
        tree, tests = local.tree, 0
        for index in np.argsort(-ceiling):
            if ceiling[index] <= floor or tests >= _DEPTH_INSIDE_BUDGET:
                break
            nearest = tree.find_nearest(tuple(points[index]))
            if nearest[0] is None or nearest[3] <= floor:
                continue
            tests += 1
            try:
                inside = local.inside(points[index])
            except ValueError:  # ambiguous winding: no claim either way
                continue
            if inside:
                floor = best = float(nearest[3])
    return best


#: What the two measurements are, carried only on reports that have a pair to
#: explain. How each was arrived at is the part that cannot be read off the
#: value, and a reader who took a separating shift or a box extent for a
#: penetration depth would chase overlaps that are not there.
_SEPARATION_NOTE = ("separating_shift_m along separating_direction is how far b must move "
                    "to come clear of a at representative_frame, bisected on the evaluated "
                    "meshes; an upper bound on the minimum translation, and not a depth. "
                    "max_depth_m is the "
                    "depth: the deepest vertex of either mesh inside the other, measured "
                    "to that mesh's surface; pairs_by_depth ranks on it. "
                    "`penetration_unresolved` means no candidate direction came apart, so "
                    "the overlap is real but unmeasured.")


def _diagnostic_key(kind, a, b, *extra):
    """An exact key for one overlap diagnostic: the code, both meshes, both poses, the inputs."""
    global _DIAGNOSTIC_CODE
    if _DIAGNOSTIC_CODE is None:
        code = hashlib.sha256()
        for module in (__file__, Path(__file__).with_name("collision.py")):
            code.update(Path(module).read_bytes())
        _DIAGNOSTIC_CODE = code.digest()
    digest = hashlib.blake2b(_DIAGNOSTIC_CODE, digest_size=24)
    for part in (kind.encode(), a.local.key, b.local.key,
                 np.asarray(a.matrix, dtype="<f8").tobytes(), np.asarray(b.matrix, dtype="<f8").tobytes(),
                 json.dumps(extra, default=float).encode()):
        digest.update(len(part).to_bytes(4, "little"))
        digest.update(part)
    return "diag:" + digest.hexdigest()


#: The digest of the modules diagnostics are computed by, once per process.
_DIAGNOSTIC_CODE = None


def _measure_diagnostics(hits, shifts, depths, directory, geometry_cache, *, collision_tools,
                         progress=None, cancel=None, store=None):
    """Append the overlap diagnostics `hits`' spans need and `shifts`/`depths` lack.

    The shift at each span's representative frame, and the depth at its
    `DEPTH_FRAMES` highest-bound frames, measured on the evaluated meshes in
    `directory`. `measure` calls this on the spans it saw; `fill_diagnostics`
    on a merged document's, where the rows already there are kept.
    """
    directional_separation, = collision_tools
    have_shift = {(row["a"], row["b"], row["frame"]) for row in shifts}
    have_depth = {(row["a"], row["b"], row["frame"]) for row in depths}
    #: The last frames read. Spans of different pairs share their frames, and
    #: each read is a full record decode, so most visits are to frames already
    #: decoded.
    recent = OrderedDict()

    def frame_bodies(frame):
        if frame in recent:
            recent.move_to_end(frame)
            return recent[frame]
        recent[frame] = diagnostic_body = read_bodies(frame)
        while len(recent) > 16:
            recent.popitem(last=False)
        return diagnostic_body

    def read_bodies(frame):
        record, geometry = load_frame(directory, frame)
        local = getattr(geometry, 'local', None)
        by_name = {obj['name']: obj for obj in record['objects']}
        lookup = {(obj.get('identity') or obj['name']): obj['name'] for obj in record['objects']}
        def diagnostic_body(ref):
            """A placed body, so the bisection below reuses one local BVH.

            `directional_separation` walks its bisection steps, each calling
            `translated`. On a world-space `Body` that copies the vertex array
            and recomputes `mesh_key`, so every step hands Coal a mesh it has
            never seen and pays a full BVH build; on a `PlacedBody` it only
            edits the 4x4 and the cached local BVH is reused. The main frame
            loop already takes the placed path.
            """
            name = lookup[ref]
            obj = by_name[name]
            if local is not None:
                return geometry_cache.placed(*local[name], obj['matrix'], key=obj.get('mesh'))
            return geometry_cache.get(*geometry[name])
        return diagnostic_body
    spans = _overlap_spans(hits)
    for completed, span in enumerate(spans):
        if cancel is not None and cancel.is_set():
            raise RuntimeError('job cancelled')
        # Bounded as `measure`'s frame loop is: the placements of this span
        # and the last, not of every span visited.
        geometry_cache.next_frame()
        if progress:
            progress({'phase': 'penetration_diagnostics', 'frame': span['representative_frame'],
                      'interval_count': len(spans), 'intervals_done': completed})
        pair = {"a": span["a"], "b": span["b"]}
        chosen = span['representative_frame']
        for frame in [chosen] + [f for _, f in span['ranked'] if f != chosen]:
            key = (span["a"], span["b"], frame)
            wants_shift = frame == chosen and key not in have_shift
            if not wants_shift and key in have_depth:
                continue
            body = frame_bodies(frame)
            a, b = body(span['a']), body(span['b'])
            if wants_shift:
                # Stored with the pair verdicts, for the very inputs measured:
                # a failing film is re-checked most, and recomputing these
                # dominates a re-check.
                name = _diagnostic_key("shift", a, b, list(span['direction']), span['bound'])
                shift, found = store.get_value(name) if store is not None else (None, False)
                if not found:
                    shift = directional_separation(a, b, span['direction'], span['bound'])
                    if store is not None:
                        store.put_value(name, shift)
                shifts.append({**pair, "frame": frame, **shift})
                have_shift.add(key)
            # The deepest over the span's highest-bound frames, not the depth
            # at the one frame the shift was measured on: the bound that picks
            # that frame is a box slack, not a depth, and can miss the deepest
            # frame by a wide margin.
            if key not in have_depth:
                name = _diagnostic_key("depth", a, b)
                depth, found = store.get_value(name) if store is not None else (None, False)
                if not found:
                    depth = deepest_vertex_m(a, b)
                    if store is not None:
                        store.put_value(name, depth)
                depths.append({**pair, "frame": frame, "depth": depth})
                have_depth.add(key)
        if progress:
            progress({'phase': 'penetration_diagnostics', 'frame': chosen,
                      'interval_count': len(spans), 'intervals_done': completed + 1})


def fill_diagnostics(bundle, directory, progress=None, cancel=None, pair_store=None):
    """Measure the overlap diagnostics a merged rows document's spans lack.

    A re-check's spans can join an earlier check's, and the joined span's
    representative or depth frames may be ones neither measured; they are
    measured here, on the evaluation in `directory`, which holds every frame.
    Returns `bundle`, extended in place.
    """
    try:
        from .collision import VERSION, GeometryCache, directional_separation
        from .collision_cache import PairCollisionCache
    except ImportError:
        from collision import VERSION, GeometryCache, directional_separation
        from collision_cache import PairCollisionCache
    rows = bundle["rows"]["NoPenetration"]
    store = PairCollisionCache(None, method=VERSION, store=pair_store) if pair_store else None
    _measure_diagnostics(rows["hits"], rows["shifts"], rows["depths"], Path(directory),
                         GeometryCache(), collision_tools=(directional_separation,),
                         progress=progress, cancel=cancel, store=store)
    if store is not None:
        store.flush()
    return bundle


def _overlap_spans(hits):
    """Each pair's overlap runs, as `[{a, b, kind, start_frame, end_frame, ...}]`.

    A run continues while the pair overlaps on consecutive output frames. Its
    `representative_frame` is the first frame with the largest box bound, the
    one the shift is bisected on, and `ranked` its `DEPTH_FRAMES` largest-bound
    frames, the ones depth is measured on. Picking by the box bound is a
    within-pair comparison -- same two meshes, different poses -- so the slack
    that makes the bound unusable *across* pairs largely cancels here.

    Both picks decompose over a split. A run cut into pieces has its
    representative among the pieces' representatives (the first largest bound
    of the whole is the first largest of the piece it lies in), and its
    `ranked` frames among the pieces' `ranked` frames, so the diagnostics each
    piece's process measured are enough to report the whole run exactly.
    `hits` must be in frame order, as `measure` appends them.
    """
    spans, open_spans = [], {}
    for hit in hits:
        frame, pair = hit["frame"], (hit["a"], hit["b"])
        span = open_spans.get(pair)
        if span is None or span["end_frame"] != frame - 1:
            span = {"a": hit["a"], "b": hit["b"], "kind": hit["kind"], "start_frame": frame,
                    "end_frame": frame, "representative_frame": frame,
                    "bound": hit["bound"], "direction": hit["direction"], "ranked": []}
            open_spans[pair] = span
            spans.append(span)
        span["end_frame"] = frame
        span["ranked"].append((hit["bound"], frame))
        span["ranked"].sort(key=lambda item: (-item[0], item[1]))
        del span["ranked"][DEPTH_FRAMES:]
        if hit["bound"] > span["bound"]:
            span.update(representative_frame=frame, kind=hit["kind"],
                        bound=hit["bound"], direction=hit["direction"])
    return spans


#: How far past an allowance's `max_depth_m` a measured depth may sit before it
#: counts: the vertex depth is a BVH distance, exact to well under a millimeter.
ALLOWANCE_DEPTH_SLACK_M = 0.001

#: How many allowed pairs `allowed_contacts` lists, deepest first.
ALLOWED_CONTACTS_SHOWN = 20


def allowed_contacts(rows):
    """Every allowed overlap's deepest measured depth, per pair, deepest first.

    Reported whether or not the allowance states a bound, so an allowance whose
    reason says "a few millimeters" shows what it actually excused. A depth of
    `None` means no vertex of either mesh lay inside the other (bars crossing,
    open sheets): the pair overlapped, by an amount this measure cannot give.
    """
    worst = {}
    for row in rows:
        key = tuple(sorted((row["a"], row["b"])))
        entry = worst.setdefault(key, {"a": key[0], "b": key[1], "max_depth_m": None,
                                       "max_depth_frame": None, "frames": 0, "limit_m": row.get("limit")})
        entry["frames"] += 1
        depth = row.get("depth")
        if depth is not None and (entry["max_depth_m"] is None or depth > entry["max_depth_m"]):
            entry["max_depth_m"], entry["max_depth_frame"] = round(float(depth), 6), row["frame"]
            entry["limit_m"] = row.get("limit")
    ranked = sorted(worst.values(), key=lambda e: -(e["max_depth_m"] or 0.0))
    for entry in ranked:
        if entry["limit_m"] is None:
            entry.pop("limit_m")
    return ranked[:ALLOWED_CONTACTS_SHOWN]


def allowance_overruns(rows):
    """The allowed-overlap rows that went deeper than their allowance's `max_depth_m`."""
    return [{"frame": row["frame"], **dict(zip(("a", "b"), sorted((row["a"], row["b"])))),
             "kind": "allowance_exceeded",
             "depth_m": round(float(row["depth"]), 6), "max_depth_m": row["limit"]}
            for row in rows
            if row.get("limit") is not None and row.get("depth") is not None
            and row["depth"] > row["limit"] + ALLOWANCE_DEPTH_SLACK_M]


def _overrun_spans(overruns):
    spans = {}
    for row in overruns:
        key = (row["a"], row["b"])
        span = spans.setdefault(key, {"a": row["a"], "b": row["b"], "max_depth_m": row["max_depth_m"],
                                      "deepest_m": row["depth_m"], "deepest_frame": row["frame"],
                                      "first_frame": row["frame"], "last_frame": row["frame"], "frames": 0})
        span["frames"] += 1
        span["first_frame"] = min(span["first_frame"], row["frame"])
        span["last_frame"] = max(span["last_frame"], row["frame"])
        if row["depth_m"] > span["deepest_m"]:
            # The bound quoted is the one the deepest overrun broke: a pair can
            # carry allowances with different depths over different windows.
            span["deepest_m"], span["deepest_frame"] = row["depth_m"], row["frame"]
            span["max_depth_m"] = row["max_depth_m"]
    return sorted(spans.values(), key=lambda s: -s["deepest_m"])


def penetration_repair(evidence):
    """The repair for pairs that penetrate, so every penetration failure carries one."""
    if not evidence.get("pairs"):
        return []
    return ["Separate each pair in `pairs` (deepest first in `pairs_by_depth`) by changing the "
            "pose, the path or the parts. Where the overlap is intended, such as an item held, "
            "worn or nested, declare a narrow collision allowance for those parts with a window "
            "and the depth as `max_depth_m`."]


def allowance_overrun_repair(overruns):
    """One repair per pair whose allowed overlap went past its stated bound."""
    out = []
    for span in _overrun_spans(overruns):
        out.append(f"{span['a']} / {span['b']} went {span['deepest_m'] * 1000:.0f} mm deep at frame "
                   f"{span['deepest_frame']}, past its allowance's max_depth_m of "
                   f"{span['max_depth_m'] * 1000:.0f} mm: change the pose or the parts so they meet "
                   "as the allowance says, or raise max_depth_m if that depth is what is meant")
    return out


def _penetration_evidence(context, rows, stats):
    """`NoPenetration`'s evidence: spans per pair, depths, room escapes, coverage."""
    shifts = {(row["a"], row["b"], row["frame"]): row for row in rows["shifts"]}
    depths = {(row["a"], row["b"], row["frame"]): row["depth"] for row in rows["depths"]}

    def measured(table, a, b, frame, what):
        key = (a, b, frame)
        if key not in table:
            raise ValueError(f"no {what} was measured for {a} / {b} at frame {frame}; the "
                             "rows are not a whole check's")
        return table[key]

    entries = []
    for span in _overlap_spans(rows["hits"]):
        a, b, chosen = span["a"], span["b"], span["representative_frame"]
        shift = measured(shifts, a, b, chosen, "separation")
        deepest, deepest_frame = measured(depths, a, b, chosen, "depth"), chosen
        for _, frame in span["ranked"]:
            if frame == chosen:
                continue
            depth = measured(depths, a, b, frame, "depth")
            if depth is not None and (deepest is None or depth > deepest):
                deepest, deepest_frame = depth, frame
        entry = {"a": a, "b": b, "kind": span["kind"], "start_frame": span["start_frame"],
                 "end_frame": span["end_frame"], "representative_frame": chosen,
                 "separation_direction": span["direction"], "max_depth_m": deepest}
        entry.update({key: shift[key] for key in ("directional_separation_m", "separation_direction")
                      if key in shift})
        if deepest is not None:
            entry["max_depth_frame"] = deepest_frame
        entries.append(entry)
    pairs = penetration_pairs(entries)
    # The cache and BVH counters stay: four integers, and the collision-cache
    # test reads them to prove a reuse path was actually taken rather than
    # silently recomputed. They nest under one key: they are query profiling,
    # not findings, and at top level they were four of the six things an agent
    # saw on a passing report.
    penetration = {"allowances": context["allowances"], "stats": stats}
    contacts = allowed_contacts(rows.get("allowed", ()))
    if contacts:
        penetration["allowed_contacts"] = contacts
    overruns = allowance_overruns(rows.get("allowed", ()))
    if overruns:
        penetration["allowances_exceeded"] = _overrun_spans(overruns)
    rejected = [row for row in context["declaration_errors"] if row["kind"] == "collision_allowance"]
    if rejected:
        penetration["invalid_allowances"] = rejected
    # Declared liquids are not checked at all, so a pass says nothing about them.
    if context.get("liquids"):
        penetration["liquids_exempt"] = context["liquids"]
    rejected = [row for row in context["declaration_errors"] if row["kind"] == "liquid"]
    if rejected:
        penetration["invalid_liquids"] = rejected
    if pairs:
        penetration["pairs"] = pairs
        penetration["measure"] = _SEPARATION_NOTE
        by_depth = pairs_by_depth(pairs)
        if by_depth:
            penetration["pairs_by_depth"] = by_depth
    # What the room test could and could not check. An `open_air` scene has
    # no walls, and the honest report is "no containment obligation", not a
    # silent pass that reads the same as a room that held everything in.
    room = context["room"]
    faces = {(axis, sign): tag for axis, sign, tag in (room or {}).get("faces", ())}
    checked_surfaces = sorted(set(faces.values()))
    unchecked = [tag for tag in ROOM_SURFACE_TAGS if tag not in checked_surfaces]
    if unchecked:
        penetration["containment"] = {"checked_surfaces": checked_surfaces,
                                      "unchecked": unchecked}
    if rows["escapes"]:
        penetration["out_of_room"] = out_of_room(rows["escapes"], (room["lo"], room["hi"]), faces)
    # The frames a repair has to reach, in the one shape every failing
    # required check uses: runs of absolute output frames.
    failed = sorted({row["frame"] for row in rows["hits"]} | {row["frame"] for row in rows["escapes"]}
                    | {row["frame"] for row in overruns})
    if failed:
        penetration["frames_failed"] = frame_runs(failed)
    return penetration
