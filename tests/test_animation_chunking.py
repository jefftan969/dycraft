"""A chunked check must report exactly what one process reports.

A chunk returns `verify.measure`'s per-frame rows for the frames it owns, the
rows are concatenated (`chunking.merge_rows`), and `verify.aggregate` -- the
function a single pass calls on its own rows -- decides everything that spans
frames. These tests build one pass's rows by hand, split them the way chunks
would have measured them, and require the two aggregations to agree.

What a chunk measures differs from a slice of the whole in one place, and the
split here reproduces it: overlap diagnostics are measured on the spans the
chunk itself sees, at each piece's representative and depth frames. The
whole-film values must be among them (`verify._overlap_spans`).

None of this needs Blender. Each case below is a way a per-verifier merge would
go wrong: labeling the advisory checks `fail`, guessing an event's mode, or
anchoring a `planted` slip where a chunk began.
"""
import copy

import pytest

from dycraft.animation.blender.chunking import CONTEXT, merge_rows, plan_chunks
from dycraft.animation.blender.verify import ROWS_FORMAT, aggregate
from dycraft.animation.blender.verify_penetration import _overlap_spans


def merge_reports(bundles):
    """A parallel check's reports, composed as the session composes them."""
    return aggregate(merge_rows(bundles))

FPS = 24.0


def shift_at(a, b, frame):
    """A deterministic stand-in for `directional_separation` at one frame."""
    return {"directional_separation_m": .001 * (frame % 7 + 1), "resolution_m": 1e-6,
            "separation_direction": [0., 0., 1.], "depth_method": "test"}


def depth_at(a, b, frame):
    return None if frame % 5 == 0 else .0005 * (frame % 11)


def with_diagnostics(bundle):
    """Measure the overlap diagnostics on the spans `bundle`'s own hits form.

    What `measure` does after its frame loop, so a chunk's bundle carries the
    diagnostics of the pieces it saw and a whole pass's those of whole spans.
    """
    rows = bundle["rows"]["NoPenetration"]
    rows["shifts"], rows["depths"] = [], []
    for span in _overlap_spans(rows["hits"]):
        a, b, chosen = span["a"], span["b"], span["representative_frame"]
        rows["shifts"].append({"a": a, "b": b, "frame": chosen, **shift_at(a, b, chosen)})
        for frame in [chosen] + [f for _, f in span["ranked"] if f != chosen]:
            rows["depths"].append({"a": a, "b": b, "frame": frame, "depth": depth_at(a, b, frame)})
    return bundle


def whole(frames, *, events=(), errors=(), hits=(), escapes=(), failures=(), joints=(),
          anatomical=(), slip=(), support=(), moved=(), spikes=(), room=None, stats=1):
    """One process's rows over `frames`, every list in frame order."""
    frames = list(frames)
    return with_diagnostics({
        "format": ROWS_FORMAT, "frames": frames,
        "context": {"fps": FPS, "identity_changed": False,
                    "events": [{"event": event, "window_frames": window} for event, window in events],
                    "declaration_errors": list(errors),
                    "allowances": {"declared": 0, "inherited": 0}, "room": room,
                    "joints": True, "anatomical": False, "weighed": ["dog", "robot"]},
        "stats": {"bvh_builds": stats, "collision_pair_cache_hits": stats},
        "rows": {"AnimationWellFormed": list(failures), "JointLimits": list(joints),
                 "AnatomicalJointLimits": list(anatomical),
                 "NoPenetration": {"hits": list(hits), "escapes": list(escapes)},
                 "EventsSatisfied": [], "moved": list(moved),
                 "MotionContinuity": {"warnings": list(spikes), "rigid_duplicates": [],
                                      "transitions_checked": [{"frame": f, "count": 1} for f in frames],
                                      "topology_changes": [], "boundaries": []},
                 "SupportMargin": list(support), "ContactSlip": list(slip)}})


def with_events(bundle, measured):
    """Attach event rows: `measured` is `{event index: {frame: row fields}}`."""
    rows = [{"event": k, "frame": frame, **fields}
            for frame in bundle["frames"] for k, by_frame in sorted(measured.items())
            if frame in by_frame for fields in [by_frame[frame]]]
    bundle["rows"]["EventsSatisfied"] = rows
    return bundle


def split(bundle, bounds):
    """The bundles chunks owning `bounds` would have measured."""
    chunks = []
    for lo, hi in bounds:
        def own(row, key="frame"):
            return lo <= row[key] <= hi
        part = copy.deepcopy(bundle)
        part["frames"] = [f for f in bundle["frames"] if lo <= f <= hi]
        rows = part["rows"]
        for name in ("AnimationWellFormed", "JointLimits", "AnatomicalJointLimits",
                     "EventsSatisfied", "moved", "SupportMargin", "ContactSlip"):
            rows[name] = [row for row in rows[name] if own(row)]
        rows["NoPenetration"] = {"hits": [h for h in rows["NoPenetration"]["hits"] if own(h)],
                                 "escapes": [e for e in rows["NoPenetration"]["escapes"] if own(e)]}
        rows["MotionContinuity"] = {key: [row for row in value if own(row)]
                                    for key, value in rows["MotionContinuity"].items()}
        part["stats"] = dict(bundle["stats"])
        chunks.append(with_diagnostics(part))
    return chunks


def comparable(reports):
    """Per-process counters legitimately differ; nothing else may."""
    return [{**r, "evidence": {k: v for k, v in r["evidence"].items() if k != "stats"}}
            for r in reports]


def agrees(bundle, bounds):
    """The chunked reports, after checking they equal the single pass's."""
    merged = merge_reports(split(bundle, bounds))
    assert comparable(merged) == comparable(aggregate(bundle))
    return {r["name"]: r for r in merged}


HALVES = [(0, 29), (30, 59)]


def test_a_clean_split_timeline_merges_to_the_shape_of_one_clean_pass():
    reports = agrees(whole(range(60)), HALVES)
    for name, r in reports.items():
        assert "failures" not in r["evidence"], name
        assert r["verdict"] in ("pass", "not_applicable")
    assert reports["AnimationWellFormed"]["evidence"]["frames_checked"] == {
        "start": 0, "end": 59, "count": 60}


def test_findings_concatenate_in_frame_order_and_counts_sum():
    bundle = whole(range(60),
                   failures=[{"frame": 4, "object": "x", "reason": "empty"},
                             {"frame": 40, "object": "x", "reason": "empty"}],
                   joints=[{"frame": f, "joint": "arm__elbow", "q": q, "limits": [0.0, 2.0],
                            "reason": "joint limit or motion axis changed"}
                           for f, q in ((28, 2.1), (29, 2.1), (30, 2.5), (31, 2.5), (32, 2.5))])
    reports = agrees(bundle, HALVES)
    formed = reports["AnimationWellFormed"]
    assert formed["verdict"] == "fail" and [f["frame"] for f in formed["evidence"]["failures"]] == [4, 40]
    joints = reports["JointLimits"]
    finding, = joints["evidence"]["findings"]
    subject, = finding["subjects"]
    assert subject["frames"] == [[28, 32]] and subject["q_range"] == [2.1, 2.5]
    assert joints["evidence"]["failure_count"] == 5
    assert "[2.1, 2.5]" in joints["suggested_repairs"][0]


def test_collision_counters_stay_nested_and_sum_across_processes():
    reports = {r["name"]: r for r in merge_reports(split(whole(range(60), stats=3), HALVES))}
    evidence = reports["NoPenetration"]["evidence"]
    assert evidence["stats"] == {"bvh_builds": 6, "collision_pair_cache_hits": 6}
    assert "pairs" not in evidence and "measure" not in evidence


def hit(frame, a="robot/hand", b="box/body", bound=.01):
    return {"frame": frame, "a": a, "b": b, "kind": "surface_crossing", "bound": bound,
            "direction": [1., 0., 0.]}


def test_an_overlap_crossing_a_boundary_is_one_span_with_the_whole_span_s_measurements():
    """The representative and the depth frames are the whole span's, not a piece's.

    The largest bound is in the second piece and the deepest sampled frame in
    the first, so a merge that kept either piece's measurement would differ.
    """
    bounds = {25: .010, 26: .030, 27: .020, 28: .025, 29: .005, 30: .040, 31: .012, 32: .015}
    reports = agrees(whole(range(60), hits=[hit(f, bound=b) for f, b in bounds.items()]), HALVES)
    evidence = reports["NoPenetration"]["evidence"]
    pair, = evidence["pairs"]
    span, = pair["spans"]
    assert span["frames"] == [25, 32] and span["representative_frame"] == 30
    assert span["separating_shift_m"] == pytest.approx(shift_at(None, None, 30)["directional_separation_m"])
    # Depth over the three largest-bound frames, 30, 26, and 28: 30 has none.
    assert span["max_depth_frame"] == 28 and span["max_depth_m"] == pytest.approx(depth_at(0, 0, 28))
    assert evidence["failure_count"] == 8 and evidence["frames_failed"] == [[25, 32]]


def test_an_overlap_in_four_chunks_agrees_too():
    hits = [hit(f, bound=.001 * ((f * 7) % 13 + 1)) for f in range(5, 55)]
    hits += [hit(f, a="robot/foot", b="room_prop/leg") for f in (14, 15, 44)]
    agrees(whole(range(60), hits=hits), [(0, 14), (15, 29), (30, 44), (45, 59)])


def event(name, predicate="touch", mode="all", window=(0, 59), **params):
    return ({"name": name, "predicate": predicate, "mode": mode, "window": list(window),
             "a": "robot/hand", "b": "box/body", **params}, list(window))


def test_an_any_event_satisfied_only_in_a_later_chunk_passes():
    """No false `fail` on a required check. The later chunk holds the event on
    every frame it owns, so nothing in its row says `any`; a merge that guessed
    the mode would read it as `all` and fail it."""
    bundle = with_events(whole(range(100), events=[event("grip", mode="any", window=(0, 99))]),
                         {0: {f: {"holds": f >= 60} for f in range(100)}})
    reports = agrees(bundle, [(0, 49), (50, 99)])
    row, = reports["EventsSatisfied"]["evidence"]["events"]
    assert reports["EventsSatisfied"]["verdict"] == "pass"
    assert row["ok"] is True and row["satisfied_at"] == pytest.approx(60 / FPS, abs=1e-4)


def test_an_event_is_answered_over_the_union_of_the_chunks_that_saw_it():
    bundle = with_events(whole(range(60), events=[event("seated")]),
                         {0: {f: {"holds": f not in (30, 31)} for f in range(60)}})
    reports = agrees(bundle, HALVES)
    held, = reports["EventsSatisfied"]["evidence"]["events"]
    assert held["checked"] == 60 and held["ok"] is False
    assert held["unsatisfied"] == [[30, 31]] and held["unsatisfied_count"] == 2
    assert reports["EventsSatisfied"]["evidence"]["frames_failed"] == [[30, 31]]


def test_a_planted_event_across_a_boundary_is_anchored_where_the_foot_landed():
    """No false pass. The foot lands at frame 10 and creeps 0.2 mm a frame; a
    chunk starting at 30 must not anchor at its own first frame and measure less
    slip than happened."""
    creep = {f: {"touching": True, "patch": [.0002 * (f - 10), 0.]} for f in range(10, 60)}
    airborne = {f: {"touching": False, "lift": .02} for f in range(10)}
    bundle = with_events(whole(range(60), events=[event("stand", predicate="planted",
                                                         tolerance=.005)]),
                         {0: {**airborne, **creep}})
    reports = agrees(bundle, HALVES)
    row, = reports["EventsSatisfied"]["evidence"]["events"]
    assert row["ok"] is False and row["max_slip_m"] == pytest.approx(.0098)
    assert row["max_slip_frame"] == 59 and row["bearing_intervals"] == 1
    assert row["max_clearance_m"] == pytest.approx(.02)
    assert row["unsatisfied"] == [[36, 59]]
    assert "max_slip_is_lower_bound" not in row


def test_a_supported_event_and_missing_operands_are_answered_per_frame():
    bundle = with_events(
        whole(range(60), events=[event("balanced", predicate="supported", margin=0.0),
                                 event("reach", mode="any", window=(20, 40))]),
        {0: {f: {"margin": None if f < 5 else (-.01 if f < 50 else .02)} for f in range(60)},
         1: {f: ({"missing": ["box/body"]} if f < 35 else {"holds": f == 38}) for f in range(20, 41)}})
    reports = agrees(bundle, HALVES)
    rows = {row["name"]: row for row in reports["EventsSatisfied"]["evidence"]["events"]}
    assert rows["balanced"]["ok"] is False and rows["balanced"]["unsatisfied"] == [[50, 59]]
    assert rows["reach"]["ok"] is True and rows["reach"]["missing_on"] == [[20, 34]]


def test_a_frame_unit_window_is_not_re_read_as_seconds_by_the_aggregator():
    """Coverage comes from the resolved `window_frames`, never from `window / fps`.

    Re-derived from the declared window, a frame-unit `[100, 200]` would read
    as seconds, cover no frames, and fail every `all` event on the default
    path.
    """
    bundle = with_events(whole(range(300), events=[event("gripped", window=(100, 200))]),
                         {0: {f: {"holds": True} for f in range(100, 201)}})
    reports = agrees(bundle, [(0, 149), (150, 299)])
    held, = reports["EventsSatisfied"]["evidence"]["events"]
    assert held["checked"] == 101 and held["ok"] is True and "partial" not in held


def test_an_event_reaching_past_the_checked_frames_is_deferred_or_partial():
    """By default an event is not judged on frames nobody evaluated; asked to
    report what was seen, it answers for those and says so."""
    bundle = with_events(whole(range(100), events=[event("carried", window=(50, 400))]),
                         {0: {f: {"holds": True} for f in range(50, 100)}})
    assert aggregate(bundle)[4]["evidence"]["events"] == {"deferred": 1, "checked": 0}
    row, = aggregate(bundle, defer_partial_events=False)[4]["evidence"]["events"]
    assert row["ok"] is True and row["partial"] is True and row["checked"] == 50


def test_a_window_inside_one_chunk_and_a_zero_width_window():
    bundle = with_events(whole(range(280), events=[event("on_dolly", window=(150, 269)),
                                                   event("tap", window=(42, 42))]),
                         {0: {f: {"holds": True} for f in range(150, 270)},
                          1: {42: {"holds": True}}})
    reports = agrees(bundle, [(0, 139), (140, 279)])
    rows = {row["name"]: row for row in reports["EventsSatisfied"]["evidence"]["events"]}
    assert rows["on_dolly"]["ok"] is True and rows["on_dolly"]["satisfied_at"] == 6.25
    assert rows["tap"]["ok"] is True and rows["tap"]["window_frames"] == [42, 42]


def test_an_invalid_event_survives_the_merge_as_a_failure_placed_nowhere():
    invalid = {"kind": "event", "index": 1, "name": "bogus",
               "reason": "unsupported event predicate 'hovers'"}
    reports = agrees(whole(range(60), errors=[invalid]), HALVES)
    evidence = reports["EventsSatisfied"]["evidence"]
    assert reports["EventsSatisfied"]["verdict"] == "fail"
    assert evidence["events"] == [{"name": "bogus", "ok": False, "status": "invalid",
                                   "index": 1, "reason": invalid["reason"]}]
    assert "frames_failed" not in evidence


def test_a_check_with_every_event_deferred_collapses_to_a_count():
    bundle = whole(range(10, 20), events=[event("early", window=(0, 5)), event("late", window=(30, 40))])
    events = aggregate(bundle)[4]["evidence"]["events"]
    assert events == {"deferred": 2, "checked": 0}



def test_a_partly_covering_check_counts_what_it_judged_and_what_it_deferred():
    """A pass over the events whose windows the check covered is not a pass over
    all of them: the evidence counts the checked and the deferred ones."""
    bundle = with_events(whole(range(100), events=[event("held", window=(10, 20)),
                                                   event("later", window=(150, 160))]),
                         {0: {f: {"holds": True} for f in range(10, 21)}})
    evidence = aggregate(bundle)[4]["evidence"]
    assert evidence["events_checked"] == 1 and evidence["events_deferred"] == 1
    # A check that covered every window adds no counts.
    full = with_events(whole(range(30), events=[event("held", window=(10, 20))]),
                       {0: {f: {"holds": True} for f in range(10, 21)}})
    assert "events_deferred" not in aggregate(full)[4]["evidence"]


def walker(moving=True):
    """A robot whose foot creeps and whose body leaves its support, over 60 frames."""
    slip = [{"frame": f, "part": "robot/foot", "on": "room.floor", "patch": [.001 * f, 0.]}
            for f in range(60)]
    slip += [{"frame": f, "part": "robot/hand", "on": None, "clearance": .4} for f in range(60)]
    slip += [{"frame": f, "part": "dog/paw", "on": "room.floor", "patch": [0., .002 * f]}
             for f in range(30, 60)]
    support = [{"frame": f, "instance": "robot", "margin": .05 if f > 10 else -.02, "bearing": 1}
               for f in range(60)]
    support += [{"frame": f, "instance": "dog", "margin": .03, "bearing": 2} for f in range(30, 60)]
    support += [{"frame": f, "instance": "dog",
                 "why": {"parts_in_frame": 0, "weighed": True, "parts_bearing": 0, "support_points": 0}}
                for f in range(30)]
    moved = ([{"frame": 3, "moved": ["robot/"]}, {"frame": 45, "moved": ["dog/", "robot/"]}]
             if moving else [])
    return whole(range(60), slip=slip, support=support, moved=moved)


def test_the_advisory_checks_stay_advisory_and_keep_every_chunk_s_bodies():
    """Two `uncertain` chunks merge to `uncertain`, with every chunk's `bodies`
    and `parts`, and the flags counted once."""
    reports = agrees(walker(), HALVES)
    support, slip = reports["SupportMargin"], reports["ContactSlip"]
    assert support["verdict"] == slip["verdict"] == "uncertain"
    assert support["evidence"]["diagnostic_only"] is True
    assert support["evidence"]["measured"] == ["dog", "robot"]
    assert support["evidence"]["failure_count"] == 2
    parts = {row["part"]: row for row in slip["evidence"]["parts"]}
    assert set(parts) == {"robot/foot", "dog/paw"}
    # Anchored where the foot first bore, frame 0, though half the film is in
    # the other chunk.
    assert parts["robot/foot"]["max_slip_m"] == pytest.approx(.059)
    assert parts["robot/foot"]["bearing_intervals"] == 1
    assert "max_slip_is_lower_bound" not in parts["robot/foot"]


def test_an_instance_that_never_moved_is_not_reported_however_it_was_measured():
    reports = agrees(walker(moving=False), HALVES)
    assert reports["SupportMargin"]["evidence"]["considered"] == []
    assert reports["ContactSlip"]["evidence"]["parts"] == []


def test_motion_spikes_from_every_chunk_regroup_as_one_pass_groups_them():
    spikes = [{"frame": f, "object": "arm", "channel": "rotation", "reason": "isolated motion spike",
               "frames": [f - 1, f], "step_magnitude": .1 * f, "rate": 2.4 * f, "units": "rad/s"}
              for f in (12, 29, 30, 50)]
    reports = agrees(whole(range(60), spikes=spikes), HALVES)
    spike, = reports["MotionContinuity"]["evidence"]["spikes"]
    assert spike["frames"] == [[11, 12], [28, 30], [49, 50]] and spike["occurrences"] == 4
    assert spike["representative_frame"] == 50 and spike["units"] == "rad/s"
    assert reports["MotionContinuity"]["verdict"] == "uncertain"


def bone_row(frame, value, joint="elbow_R"):
    """One frame of `verify.bone_anatomy`: an imported character's elbow bent backwards."""
    return {"frame": frame, "rig": "rig/t/body", "joint": joint, "measure": "flexion",
            "reason": "bent backwards", "value_deg": value, "limits_deg": [-10.0, 160.0]}


def test_bone_anatomy_rows_across_a_boundary_are_one_grouped_advisory_row():
    """An imported character's per-frame bone rows regroup as one pass groups them:
    one row per joint and measure, its frames stitched across the chunk boundary,
    and its worst excess taken from whichever chunk saw it."""
    bundle = whole(range(60), anatomical=[bone_row(f, -120.0 if f == 35 else -20.0)
                                          for f in range(25, 41)])
    bundle["context"]["anatomical"] = True
    report = agrees(bundle, HALVES)["AnatomicalJointLimits"]
    assert report["verdict"] == "uncertain"
    row, = report["evidence"]["failures"]
    assert row["frames"] == [[25, 40]] and row["frame_count"] == 16
    assert row["worst_deg"] == -120.0 and row["worst_frame"] == 35


def test_a_rig_with_anatomy_that_stays_in_range_passes_rather_than_not_applicable():
    """`measure` sets `context["anatomical"]` for a rig that declares anatomy, so a
    character checked clean reads `pass`, not "no joint carries anatomical limits"."""
    bundle = whole(range(60))
    assert aggregate(bundle)[2]["verdict"] == "not_applicable"
    bundle["context"]["anatomical"] = True
    assert aggregate(bundle)[2]["verdict"] == "pass"


def test_room_escapes_merge_without_a_bounding_box():
    room = {"lo": [-2., -2., 0.], "hi": [2., 2., 3.],
            "faces": [[0, -1, "wall_xmin"], [0, 1, "wall_xmax"], [2, -1, "floor"]]}
    escapes = [{"frame": f, "object": "cart", "aabb": [[1.5, 0, 0], [2.0 + .01 * f, .5, .5]]}
               for f in (28, 29, 58, 59)]
    reports = agrees(whole(range(60), escapes=escapes, room=room), HALVES)
    escape, = reports["NoPenetration"]["evidence"]["out_of_room"]
    assert escape["outside_by_m"] == pytest.approx(.59) and escape["frames"] == [[28, 29], [58, 59]]
    assert reports["NoPenetration"]["evidence"]["frames_failed"] == [[28, 29], [58, 59]]


def test_chunks_that_disagree_or_overlap_are_refused():
    first, second = split(whole(range(60)), HALVES)
    second["context"]["events"] = [{"event": {"name": "x"}, "window_frames": [0, 1]}]
    with pytest.raises(ValueError, match="disagree on events"):
        merge_rows([first, second])
    first, second = split(whole(range(60)), [(0, 30), (30, 59)])
    with pytest.raises(RuntimeError, match="overlap"):
        merge_rows([first, second])
    with pytest.raises(RuntimeError, match="returned no verification rows"):
        merge_rows([first, None])


def test_rows_missing_for_a_checked_event_frame_are_refused():
    bundle = with_events(whole(range(10), events=[event("grip", window=(0, 9))]),
                         {0: {f: {"holds": True} for f in range(9)}})
    with pytest.raises(ValueError, match="rows for 9 of the 10"):
        aggregate(bundle)


def test_chunks_partition_what_they_own_and_overlap_only_for_context():
    plan = plan_chunks(range(60), 4)
    assert [owned for owned, _, _ in plan] == [0, 15, 30, 45]
    assert [high for _, high, _ in plan] == [14, 29, 44, 59]
    owned = [frame for low, high, _ in plan for frame in range(low, high + 1)]
    assert owned == list(range(60)), "owned ranges must partition the timeline exactly"
    for index, (low, high, evaluated) in enumerate(plan):
        assert evaluated[0] == max(0, low - CONTEXT)
        assert evaluated[-1] == min(59, high + CONTEXT)
        if index:
            assert evaluated[0] < low, "a chunk must look back before what it owns"

    # Too few frames to be worth splitting, and a scattered set has no interior.
    assert plan_chunks(range(8), 4) is None
    assert plan_chunks([0, 2, 4, 6, 8, 10] * 20, 4) is None
    assert plan_chunks(range(60), 1) is None


def test_chunk_indexes_merge_every_chunk_s_identities():
    """The merged `index.json` carries every chunk's signatures and digests, not
    only the first chunk's."""
    from dycraft.animation.blender.evaluation import merge_chunk_indexes
    first = {"fps": 24.0, "n_frames": 8, "frames": [0, 1, 2, 3, 4, 5], "render_state_version": "v1",
             "signatures": {str(f): f"s{f}" for f in range(6)},
             "render_states": {str(f): f"r{f}" for f in range(6)}, "evaluation_start_frame": 0}
    second = {**first, "frames": [2, 3, 4, 5, 6, 7], "evaluation_start_frame": 2,
              "signatures": {str(f): f"s{f}" for f in range(2, 8)},
              "render_states": {str(f): f"r{f}" for f in range(2, 8)}}
    merged = merge_chunk_indexes([first, second])
    assert merged["frames"] == list(range(8)) and merged["complete"] is True
    assert merged["signatures"] == {str(f): f"s{f}" for f in range(8)}
    assert merged["render_states"] == {str(f): f"r{f}" for f in range(8)}
    assert merged["evaluation_start_frame"] == 0
    # Chunks that disagree on what a digest is withdraw every digest.
    disagreeing = merge_chunk_indexes([first, {**second, "render_state_version": "v2"}])
    assert set(disagreeing["render_states"].values()) == {None}
    assert disagreeing["signatures"] == merged["signatures"]
    # Each frame's values come from its owner, not from a chunk that only
    # warmed up on it (a motion-blurred digest needs the frame before).
    warmed = {**second, "render_states": {**second["render_states"], "2": None}}
    owned = merge_chunk_indexes([first, warmed], owned=[(0, 3), (4, 7)])
    assert owned["render_states"] == {str(f): f"r{f}" for f in range(8)}


def test_clip_joins_match_one_scan_per_boundary_and_are_linear():
    """`joins` names the worst step across each clip boundary in one pass over
    the steps, rather than one per boundary, which dominates a check on a film
    of many clips; it must name the same steps, ties included."""
    import random
    import time
    from dycraft.animation.blender.continuity import MotionContinuity

    def per_boundary(self, clips):
        order = sorted((int(span["frames"][0]), name) for name, span in clips.items())
        out = {}
        for index, (frame, name) in enumerate(order):
            if index == 0 or frame <= 0:
                continue
            worst = {}
            for (obj, kind), steps in self.steps.items():
                for at, magnitude, _floor, _reversal in steps:
                    if at == frame and magnitude > worst.get(kind, (0, None))[0]:
                        worst[kind] = (magnitude, obj)
            if worst:
                out[(frame, name)] = worst
        return out

    rng = random.Random(7)
    motion = MotionContinuity(24)
    for obj in range(30):
        for kind in ("translation", "rotation", "deformation"):
            for frame in range(1, 400):
                # Coarse magnitudes, so ties are common and the first must win.
                motion.steps[(f"part/{obj}", kind)].append(
                    (frame, rng.choice((0., .1, .2, .2, .3)), .01, False))
    clips = {f"c{n}": {"frames": [n * 5, n * 5 + 4]} for n in range(80)}
    clips["twin"] = {"frames": [100, 120]}           # two clips entering on one frame
    expected = per_boundary(motion, clips)
    started = time.process_time()
    rows = motion.joins(clips)
    elapsed = time.process_time() - started
    got = {(row["frame"], row["enters"]): {kind: (step["magnitude"], step["object"])
                                           for kind, step in row["steps"].items()}
           for row in rows}
    assert got == {key: {kind: (round(m, 6), o) for kind, (m, o) in worst.items()}
                   for key, worst in expected.items()}
    assert elapsed < 1.0, elapsed
