"""Penetration policy: which measured pair-frames count, and how they aggregate.

Standard library only, so it runs under any interpreter and re-applies to a cached geometry pass without touching a
mesh. The input is one *timeline* per candidate pair, one code per evaluated frame:

    kind   INACTIVE (-1)  not co-visible: either body hidden, excluded or absent
           CLEAR     (0)  co-visible, no penetration (including no box overlap)
           CROSSING  (1)  surfaces cross
           CONTAINED (2)  the pair's first body lies inside its second
           CONTAINS  (5)  the second body lies inside the first
           ERROR     (4)  the measurement raised (reported, never counted)
    level  how many ladder rungs the penetration depth reaches (0 = below 1 mm)
    tau    whether the depth reaches the pair's adaptive threshold
    depth  penetration depth in mm (the smallest separating translation found)

A pair-frame counts when it passes the threshold, is not inside the pair's initial overlap (penetrating at its first
co-visible frame, not yet separated and not 2 rungs deeper), and one of its bodies is MOVING (`motion.py`). Two
further classes never count and are reported on their own: *consumed* pair-frames (an object wholly inside an agent
after ingestion, `penetration.consumption`) and *allowed* self pairs (`self_allowance`).
"""
from __future__ import annotations

from collections import Counter

INACTIVE, CLEAR, CROSSING, CONTAINED, ERROR, CONTAINS = -1, 0, 1, 2, 4, 5
PENETRATING = (CROSSING, CONTAINED, CONTAINS)
CONTAINMENT = (CONTAINED, CONTAINS)

#: Ladder rungs, meters. The sweep is a subset; the top two exist so a pair that starts at 20-50 mm can still
#: "deepen" by two rungs.
DEFAULT_LADDER_M = (0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200)
DEFAULT_SWEEP_M = (0.001, 0.002, 0.005, 0.010, 0.020, 0.050)
DEFAULT_PARAMS = {
    #: Rungs above the first co-visible frame's level before an initially overlapping pair counts again.
    "deepen_levels": 2,
    #: Consecutive co-visible clear frames that end the initial overlap; a single clear frame is noise.
    "separation_frames": 2,
    #: Counted frames of one pair separated by at most this many frames are one event.
    "event_gap_frames": 2,
    #: An event shorter than this does not set the episode-level flag.
    "episode_min_event_frames": 3,
    #: Self pairs: a construction overlap is present in at least this share of the pair's first
    #: `construction_window_s` of shared frames ...
    "construction_window_s": 0.5,
    "construction_share": 0.8,
    #: ... and stops protecting a frame deeper than max(factor * starting depth, starting depth + margin).
    "construction_deepen_factor": 2.0,
    "construction_deepen_margin_mm": 20.0,
}


def expand(timeline, n_frames):
    """RLE `[[start, end, kind, level, tau(, depth_mm)], ...]` -> one tuple per frame."""
    codes = [(INACTIVE, 0, False, 0.0)] * n_frames
    for start, end, kind, level, tau, *rest in timeline:
        code = (kind, level, bool(tau), float(rest[0]) if rest else 0.0)
        for f in range(start, end + 1):
            codes[f] = code
    return codes


def compress(codes):
    """Inverse of `expand`: deterministic, inclusive frame ranges."""
    out = []
    for f, code in enumerate(codes):
        row = [f, f, code[0], code[1], bool(code[2]), float(code[3]) if len(code) > 3 else 0.0]
        if out and out[-1][1] == f - 1 and out[-1][2:] == row[2:]:
            out[-1][1] = f
        else:
            out.append(row)
    return out


def depth_mm(code):
    return float(code[3]) if len(code) > 3 else None


def passes_adaptive(code):
    kind, tau = code[0], code[2]
    return kind in PENETRATING and (kind in CONTAINMENT or tau)


def passes_rung(rung_index):
    """Absolute threshold ladder[rung_index]."""
    def check(code):
        kind, level = code[0], code[1]
        return kind in PENETRATING and (kind in CONTAINMENT or level >= rung_index + 1)
    return check


def penetrates(code):
    return code[0] in PENETRATING


def classify_pair(codes, passes, params=DEFAULT_PARAMS):
    """Apply the initial-overlap, re-entry and deepening rule to one pair.

    Whether a pair starts overlapping, and when it separates, are decided by `penetrates` (any measured crossing or
    containment), never by the threshold being scored; only the decision to *count* a frame uses `passes`. Otherwise
    a pair 8 mm deep from its first frame that later sits at 15 mm would be an exempt initial overlap at 5 mm but a
    new penetration at 10 mm, and the threshold sweep would stop being monotone. Frames where either body is hidden
    neither separate the pair nor penetrate.
    """
    deepen = params["deepen_levels"]
    need_clear = params["separation_frames"]
    seq = [f for f, c in enumerate(codes) if c[0] != INACTIVE]
    result = {"t0": seq[0] if seq else None, "initial": False, "base_level": None,
              "counted": [], "exempt_initial": [], "separated_at": None}
    if not seq:
        return result
    first = codes[seq[0]]
    in_initial = penetrates(first)
    result["initial"] = in_initial
    base = first[1] if in_initial else None
    result["base_level"] = base
    clear_run = 0
    for f in seq:
        code = codes[f]
        if in_initial:
            if penetrates(code):
                clear_run = 0
                if passes(code):
                    if code[1] >= base + deepen:
                        result["counted"].append((f, "deepened"))
                    else:
                        result["exempt_initial"].append(f)
            else:
                clear_run += 1
                if clear_run >= need_clear:
                    in_initial = False
                    result["separated_at"] = f
        elif passes(code):
            result["counted"].append((f, "re-entry" if result["initial"] else "new"))
    return result


def events_of(counted, gap):
    """Group counted (frame, reason) into events, bridging gaps of at most `gap` frames."""
    events = []
    for f, reason in counted:
        if events and f - events[-1]["end_frame"] - 1 <= gap:
            ev = events[-1]
            ev["end_frame"] = f
            ev["counted_frames"] += 1
            ev["reasons"][reason] += 1
        else:
            events.append({"start_frame": f, "end_frame": f, "counted_frames": 1,
                           "reasons": Counter({reason: 1})})
    return events


def ranges(frames):
    out = []
    for f in sorted(frames):
        if out and out[-1][1] == f - 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def depth_bucket(code, ladder):
    kind, level = code[0], code[1]
    if kind in CONTAINMENT:
        return "containment"
    if level <= 0:
        return f"<{ladder[0] * 1000:g}mm"
    if level >= len(ladder):
        return f">={ladder[-1] * 1000:g}mm"
    return f"{ladder[level - 1] * 1000:g}-{ladder[level] * 1000:g}mm"


def tag_frames(codes, passes, params=DEFAULT_PARAMS):
    """Every frame where the pair passes the threshold, tagged by its history.

    `initial` inside the overlap the pair had at its first co-visible frame (exempt); `deepened` the same but at
    least `deepen_levels` rungs deeper; `re-entry` after that initial overlap separated; `new` never overlapped at
    the start.
    """
    cls = classify_pair(codes, passes, params)
    tags = {f: "initial" for f in cls["exempt_initial"]}
    tags.update(dict(cls["counted"]))
    return dict(sorted(tags.items())), cls


def self_allowance(pair, codes, window, params):
    """(predicate, reason) for an allowed self pair, else (None, None); the predicate says which frames still count.

    `structural`    the two parts sit on rigid groups joined by a moving joint, or on two connected bones of a rig:
                    never counted.
    `construction`  they overlap stably from their first shared frames (teeth in a head, a tongue in a mouth), or in
                    their rig's rest pose as modeled (layered feathers): counted only where the penetration grows
                    far beyond that overlap.
    """
    if pair.get("jointed"):
        return (lambda code: False), "structural"
    bases = []
    built = pair.get("rest_overlap_mm")
    if built:
        bases.append(float(built))
    seq = [f for f, c in enumerate(codes) if c[0] != INACTIVE][:window]
    over = [f for f in seq if codes[f][0] in PENETRATING]
    if seq and len(over) >= params["construction_share"] * len(seq):
        bases.append(max(depth_mm(codes[f]) or 0.0 for f in over))
    if not bases:
        return None, None
    base = max(bases)
    limit = max(params["construction_deepen_factor"] * base, base + params["construction_deepen_margin_mm"])
    return (lambda code: (depth_mm(code) or 0.0) > limit), "construction"


def evaluate(pairs, n_frames, fps, passes, ladder, params=DEFAULT_PARAMS, *, detail=True):
    """Aggregate one threshold criterion over every measured pair.

    `pairs` are dicts with `a`, `b`, `relation` ('self' | 'cross'), `codes` (expanded, see `expand`) and `dynamic`
    (per frame, whether either body is MOVING). Every pair-frame that passes the threshold is *geometric* and falls
    in exactly one class: **exempt** (inside the pair's initial overlap), **dynamic** (not exempt and a body is
    MOVING: counted), or **static** (not exempt, both bodies STATIC: reported only).
    """
    dynamic_frames, geometric_frames, exempt_frames, static_frames_set = set(), set(), set(), set()
    motion_only_frames = set()
    by_relation = {"self": set(), "cross": set()}
    dyn_pairs_frames = Counter()
    static_pair_frames = exempt_pair_frames = exempt_moving_pair_frames = 0
    buckets, reasons = Counter(), Counter()
    dyn_depths, all_depths = [], []
    events_out, initial_pairs, flagged_pairs, static_pairs = [], [], [], []
    consumed_frames, allowed_frames = set(), set()
    consumed_pair_frames = 0
    allowed_pair_frames = Counter()
    allowed_pairs, consumed_pairs = [], []
    window = max(1, int(round(params["construction_window_s"] * fps))) if fps else 1
    for pair in pairs:
        codes, dynamic = pair["codes"], pair["dynamic"]
        tags, cls = tag_frames(codes, passes, params)
        consumed = set(pair.get("consumed", ())) & set(tags)
        if consumed:
            consumed_frames.update(consumed)
            consumed_pair_frames += len(consumed)
            consumed_pairs.append({"a": pair["a"], "b": pair["b"], "frames": len(consumed),
                                   "frame_ranges": ranges(consumed)})
            tags = {f: t for f, t in tags.items() if f not in consumed}
        spared = []
        if pair["relation"] == "self":
            allowed, reason = self_allowance(pair, codes, window, params)
            if allowed:
                keep = {f for f in tags if allowed(codes[f])}
                spared = [f for f in tags if f not in keep]
                if spared:
                    allowed_frames.update(spared)
                    allowed_pair_frames[reason] += len(spared)
                allowed_pairs.append({"a": pair["a"], "b": pair["b"], "reason": reason,
                                      "allowed_frames": len(spared), "counted_beyond_construction": len(keep)})
                tags = {f: "beyond-construction" for f in sorted(keep)}
        geometric = list(tags)
        exempt = [f for f in geometric if tags[f] == "initial"]
        eligible = [f for f in geometric if tags[f] != "initial"]
        counted = [(f, tags[f]) for f in eligible if dynamic[f]]
        static = [f for f in eligible if not dynamic[f]]
        geometric_frames.update(geometric)
        geometric_frames.update(consumed)
        geometric_frames.update(spared)
        exempt_frames.update(exempt)
        static_frames_set.update(static)
        motion_only_frames.update(f for f in (*geometric, *consumed, *spared) if dynamic[f])
        exempt_pair_frames += len(exempt)
        exempt_moving_pair_frames += sum(1 for f in exempt if dynamic[f])
        static_pair_frames += len(static)
        for f in geometric:
            d = depth_mm(codes[f])
            if d is not None:
                all_depths.append(d)
        if cls["initial"]:
            initial_pairs.append({
                "a": pair["a"], "b": pair["b"], "relation": pair["relation"], "t0": cls["t0"],
                "late_appearing": cls["t0"] > 0,
                "base_bucket": depth_bucket(codes[cls["t0"]], ladder),
                "exempt_frames": len(exempt),
                "exempt_frames_while_moving": sum(1 for f in exempt if dynamic[f]),
                "separated_at": cls["separated_at"],
                "counted_re_entry_frames": sum(1 for _, why in counted if why == "re-entry"),
                "counted_deepened_frames": sum(1 for _, why in counted if why == "deepened")})
        if static:
            static_pairs.append({"a": pair["a"], "b": pair["b"], "relation": pair["relation"],
                                 "static_frames": len(static), "frame_ranges": ranges(static)})
        if not counted:
            continue
        frames = [f for f, _ in counted]
        dynamic_frames.update(frames)
        by_relation[pair["relation"]].update(frames)
        dyn_pairs_frames[pair["relation"]] += len(frames)
        for f, reason in counted:
            buckets[depth_bucket(codes[f], ladder)] += 1
            reasons[reason] += 1
            d = depth_mm(codes[f])
            if d is not None:
                dyn_depths.append(d)
        counted_set = set(frames)
        pair_events = events_of(counted, params["event_gap_frames"])
        for ev in pair_events:
            span = [codes[f] for f in range(ev["start_frame"], ev["end_frame"] + 1) if f in counted_set]
            worst = max(span, key=lambda c: (c[0] in CONTAINMENT, c[1], depth_mm(c) or 0.0))
            events_out.append({
                "a": pair["a"], "b": pair["b"], "relation": pair["relation"],
                "start_frame": ev["start_frame"], "end_frame": ev["end_frame"],
                "counted_frames": ev["counted_frames"],
                "reasons": dict(sorted(ev["reasons"].items())),
                "max_bucket": depth_bucket(worst, ladder),
                "kind": "containment" if any(c[0] in CONTAINMENT for c in span) else "surface_crossing",
                "max_depth_mm": depth_mm(worst)})
        flagged_pairs.append({"a": pair["a"], "b": pair["b"], "relation": pair["relation"],
                              "a_name": pair.get("a_name"), "b_name": pair.get("b_name"),
                              "dynamic_frames": len(frames), "static_frames": len(static),
                              "exempt_frames": len(exempt),
                              "events": len(pair_events), "frame_ranges": ranges(frames)})
    events_out.sort(key=lambda e: (e["start_frame"], e["a"], e["b"]))
    flagged_pairs.sort(key=lambda p: (-p["dynamic_frames"], p["a"], p["b"]))
    static_pairs.sort(key=lambda p: (-p["static_frames"], p["a"], p["b"]))
    minutes = n_frames / fps / 60 if fps else None
    ev_depths = [e["max_depth_mm"] for e in events_out if e["max_depth_mm"] is not None]
    long_events = [e for e in events_out if e["end_frame"] - e["start_frame"] + 1 >= params["episode_min_event_frames"]]

    def stats(values):
        values = sorted(values)
        return {"mean": sum(values) / len(values) if values else None,
                "median": values[len(values) // 2] if values else None,
                "max": values[-1] if values else None}
    rate = (lambda k: k / n_frames if n_frames else None)
    summary = {
        "dynamic_penetration_frame_rate": rate(len(dynamic_frames)),
        "dynamic_invalid_frames": len(dynamic_frames),
        "evaluated_frames": n_frames,
        "all_geometric_penetration_frames": len(geometric_frames),
        "geometric_penetration_frame_rate": rate(len(geometric_frames)),
        "motion_only_frame_rate": rate(len(motion_only_frames)),
        "motion_only_invalid_frames": len(motion_only_frames),
        "initial_overlap_exempt_frames": len(exempt_frames),
        "initial_overlap_exempt_pair_frames": exempt_pair_frames,
        "initial_overlap_exempt_pair_frames_while_moving": exempt_moving_pair_frames,
        "static_only_penetration_frames": len(static_frames_set - dynamic_frames),
        "exempt_only_penetration_frames": len(geometric_frames - dynamic_frames - static_frames_set),
        "dynamic_pair_frames": sum(dyn_pairs_frames.values()),
        "static_pair_frames": static_pair_frames,
        "dynamic_pair_frames_by_relation": {k: dyn_pairs_frames.get(k, 0) for k in ("cross", "self")},
        "cross_object_dynamic_frame_rate": rate(len(by_relation["cross"])),
        "self_penetration_dynamic_frame_rate": rate(len(by_relation["self"])),
        "cross_object_dynamic_invalid_frames": len(by_relation["cross"]),
        "self_penetration_dynamic_invalid_frames": len(by_relation["self"]),
        "events": len(events_out),
        "events_per_minute": len(events_out) / minutes if minutes else None,
        "episode_has_penetration": bool(long_events),
        "flagged_pairs": len(flagged_pairs),
        "static_overlap_pairs": len(static_pairs),
        "counted_reasons": dict(sorted(reasons.items())),
        "depth_buckets_dynamic_pair_frames": dict(sorted(buckets.items())),
        "depth_mm": {"dynamic_pair_frames": stats(dyn_depths), "all_geometric_pair_frames": stats(all_depths),
                     "event_max_mean": sum(ev_depths) / len(ev_depths) if ev_depths else None},
        "initial_overlap_pairs": len(initial_pairs),
        "consumed_frames": len(consumed_frames),
        "consumed_pair_frames": consumed_pair_frames,
        "self_allowed_frames": len(allowed_frames),
        "self_allowed_pair_frames": dict(sorted(allowed_pair_frames.items())),
        "self_allowed_pairs": len(allowed_pairs),
    }
    if detail:
        summary["dynamic_invalid_frame_ranges"] = ranges(dynamic_frames)
        summary["static_only_frame_ranges"] = ranges(static_frames_set - dynamic_frames)
        summary["event_list"] = events_out
        summary["flagged_pair_list"] = flagged_pairs
        summary["static_overlap_list"] = static_pairs[:200]
        summary["initial_overlap_list"] = sorted(initial_pairs, key=lambda p: (p["t0"], p["a"], p["b"]))
        summary["consumed_frame_ranges"] = ranges(consumed_frames)
        summary["consumed_pair_list"] = sorted(consumed_pairs, key=lambda p: (-p["frames"], p["a"], p["b"]))
        summary["self_allowed_list"] = sorted(allowed_pairs, key=lambda p: (-p["allowed_frames"], p["a"], p["b"]))[:300]
    return summary
