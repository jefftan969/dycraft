"""The penetration rate (`benchmark/score`): its policy and its real geometry.

The rule tests drive `penetration_rules.py` with hand-written timelines. The end-to-end tests write a synthetic
`dycraft.evaluation.v2` export of boxes, one small scene per edge case spaced far apart, and score it, so Coal, erosion
and the pair cache are exercised as on a real film.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


from dycraft.benchmark.score import motion as motion_states
from dycraft.benchmark.score import penetration as pen
from dycraft.benchmark.score import penetration_rules as rules
from dycraft.benchmark.score import records
from dycraft.benchmark.score.common import Settings, run_children
from dycraft.benchmark.score.inputs import Input
C, X, K, I = rules.CROSSING, rules.CLEAR, rules.CONTAINED, rules.INACTIVE


def codes(*spec):
    """spec: (kind, level, tau) repeated by count, e.g. ((C,3,True), 4)."""
    out = []
    for code, count in spec:
        out.extend([code] * count)
    return out


def counted(seq, passes=rules.passes_adaptive, **params):
    return rules.classify_pair(seq, passes, {**rules.DEFAULT_PARAMS, **params})


# ----------------------------------------------------------------- pure rules

def test_initial_overlap_is_exempt_while_it_persists():
    r = counted(codes(((C, 3, True), 10)))
    assert r["initial"] and r["counted"] == [] and len(r["exempt_initial"]) == 10


def test_one_clear_frame_is_not_a_separation():
    r = counted(codes(((C, 3, True), 3), ((X, 0, False), 1), ((C, 3, True), 3)))
    assert r["counted"] == [] and r["separated_at"] is None


def test_re_penetration_after_separation_counts():
    r = counted(codes(((C, 3, True), 3), ((X, 0, False), 2), ((C, 3, True), 3)))
    assert r["separated_at"] == 4
    assert r["counted"] == [(5, "re-entry"), (6, "re-entry"), (7, "re-entry")]


def test_deepening_by_two_rungs_counts_one_rung_does_not():
    base = (C, 3, True)
    r = counted(codes((base, 2), ((C, 4, True), 2), ((C, 5, True), 2), (base, 2)))
    assert [f for f, why in r["counted"]] == [4, 5]
    assert {why for _, why in r["counted"]} == {"deepened"}


def test_initial_state_does_not_depend_on_the_threshold():
    # 8 mm from frame 0 (3 rungs), later 15 mm (4 rungs): one rung deeper is not
    # "deepened", at 5 mm *or* at 10 mm -- the sweep must stay monotone.
    seq = codes(((C, 3, True, 7.9), 5), ((C, 4, True, 15.0), 5))
    for rung in range(len(rules.DEFAULT_LADDER_M)):
        assert counted(seq, rules.passes_rung(rung))["counted"] == []
    # A shallow initial contact (below 1 mm) that later drives 20 mm in counts.
    seq = codes(((C, 0, False, 0.4), 3), ((C, 5, True, 20.0), 3))
    assert [why for _, why in counted(seq)["counted"]] == ["deepened"] * 3


def test_hidden_frames_neither_separate_nor_count():
    r = counted(codes(((C, 3, True), 2), ((I, 0, False), 5), ((C, 3, True), 2)))
    assert r["counted"] == [] and r["separated_at"] is None


def test_new_penetration_counts_and_threshold_is_per_criterion():
    seq = codes(((X, 0, False), 3), ((C, 2, False), 3))   # 2 rungs, below tau
    assert counted(seq)["counted"] == []                      # adaptive tau: no
    one_mm = rules.passes_rung(0)
    five_mm = rules.passes_rung(2)
    assert len(counted(seq, one_mm)["counted"]) == 3
    assert counted(seq, five_mm)["counted"] == []


def test_containment_counts_at_any_threshold():
    seq = codes(((X, 0, False), 2), ((K, 8, True), 2))
    assert len(counted(seq)["counted"]) == 2
    assert len(counted(seq, rules.passes_rung(5))["counted"]) == 2


def test_dynamic_rate_is_union_over_pairs_and_splits_self_cross():
    n = 10
    moving = [True] * n
    a = {"a": "x", "b": "y", "relation": "cross", "dynamic": moving,
         "codes": codes(((X, 0, False), 5), ((C, 4, True), 5))}
    b = {"a": "r/hand", "b": "r/torso", "relation": "self", "dynamic": moving,
         "codes": codes(((X, 0, False), 7), ((C, 4, True), 3))}
    s = rules.evaluate([a, b], n, 10.0, rules.passes_adaptive, list(rules.DEFAULT_LADDER_M))
    assert s["dynamic_invalid_frames"] == 5 and s["dynamic_penetration_frame_rate"] == 0.5
    assert s["cross_object_dynamic_invalid_frames"] == 5 and s["self_penetration_dynamic_invalid_frames"] == 3
    assert s["events"] == 2 and s["dynamic_invalid_frame_ranges"] == [[5, 9]]


def _pair(dynamic, spec, a="x", b="y", relation="cross"):
    return {"a": a, "b": b, "relation": relation, "dynamic": dynamic, "codes": codes(*spec)}


def _score(*pairs, n=12):
    return rules.evaluate(list(pairs), n, 12.0, rules.passes_adaptive, list(rules.DEFAULT_LADDER_M))


def test_A_initial_overlap_static_forever_is_exempt():
    s = _score(_pair([False] * 12, [((C, 4, True, 10.0), 12)]))
    assert s["dynamic_invalid_frames"] == 0 and s["dynamic_penetration_frame_rate"] == 0
    assert s["all_geometric_penetration_frames"] == 12
    assert s["initial_overlap_exempt_frames"] == 12 and s["initial_overlap_exempt_pair_frames"] == 12
    assert s["static_only_penetration_frames"] == 0 and s["exempt_only_penetration_frames"] == 12
    assert s["motion_only_invalid_frames"] == 0


def test_B_initial_overlap_moving_together_stays_exempt():
    s = _score(_pair([True] * 12, [((C, 4, True, 10.0), 12)]))
    assert s["dynamic_invalid_frames"] == 0
    assert s["initial_overlap_exempt_pair_frames_while_moving"] == 12
    assert s["motion_only_invalid_frames"] == 12          # what the motion rule alone would have counted


def test_C_initial_overlap_sliding_at_constant_depth_stays_exempt():
    dyn = [False] * 4 + [True] * 8
    s = _score(_pair(dyn, [((C, 4, True, 10.0), 4), ((C, 4, True, 11.0), 4), ((C, 3, True, 9.0), 4)]))
    assert s["dynamic_invalid_frames"] == 0 and s["initial_overlap_exempt_frames"] == 12


def test_D_initial_overlap_then_separates():
    s = _score(_pair([True] * 12, [((C, 4, True, 10.0), 4), ((X, 0, False, 0.0), 8)]))
    assert s["dynamic_invalid_frames"] == 0 and s["initial_overlap_exempt_frames"] == 4
    assert s["initial_overlap_list"][0]["separated_at"] == 5


def test_E_separation_then_re_penetration_during_motion_counts():
    # also the foot case: sunk at the start, steps out, plants through the ground again
    s = _score(_pair([True] * 12, [((C, 4, True, 10.0), 3), ((X, 0, False, 0.0), 3), ((C, 4, True, 10.0), 6)]))
    assert s["dynamic_invalid_frame_ranges"] == [[6, 11]] and s["counted_reasons"] == {"re-entry": 6}
    assert s["initial_overlap_exempt_frames"] == 3


def test_E_re_penetration_while_both_static_does_not_count():
    dyn = [True] * 6 + [False] * 6
    s = _score(_pair(dyn, [((C, 4, True, 10.0), 3), ((X, 0, False, 0.0), 3), ((C, 4, True, 10.0), 6)]))
    assert s["dynamic_invalid_frames"] == 0 and s["static_only_frame_ranges"] == [[6, 11]]


def test_F_meaningful_deepening_during_motion_counts():
    s = _score(_pair([True] * 12, [((C, 3, True, 8.0), 4), ((C, 4, True, 15.0), 2), ((C, 5, True, 40.0), 4),
                                    ((C, 3, True, 8.0), 2)]))
    assert s["dynamic_invalid_frame_ranges"] == [[6, 9]] and s["counted_reasons"] == {"deepened": 4}
    assert s["initial_overlap_exempt_frames"] == 8                 # 8 mm and 15 mm: one rung is not "meaningful"


def test_G_clear_then_moves_in_then_settles():
    dyn = [True] * 6 + [False] * 6
    s = _score(_pair(dyn, [((X, 0, False, 0.0), 2), ((C, 3, True, 8.0), 10)]))
    assert s["dynamic_invalid_frame_ranges"] == [[2, 5]]
    assert s["static_only_frame_ranges"] == [[6, 11]] and s["initial_overlap_exempt_frames"] == 0


def test_H_moving_body_into_static_body_counts():
    s = _score(_pair([True] * 12, [((X, 0, False, 0.0), 4), ((C, 3, True, 8.0), 8)]))
    assert s["dynamic_invalid_frame_ranges"] == [[4, 11]] and s["counted_reasons"] == {"new": 8}


def test_H_sweep_is_monotone_with_exemptions_and_motion():
    dyn = [False] * 3 + [True] * 9
    pairs = [
        _pair(dyn, [((C, 3, True, 8.0), 4), ((C, 4, True, 15.0), 8)], a="p", b="q"),       # initial, +1 rung
        _pair(dyn, [((C, 2, True, 3.0), 3), ((C, 5, True, 30.0), 9)], a="r", b="s"),       # initial, deepens
        _pair(dyn, [((X, 0, False, 0.0), 5), ((C, 4, True, 12.0), 7)], a="t", b="u"),      # new
    ]
    counts = [rules.evaluate(pairs, 12, 12.0, rules.passes_rung(k), list(rules.DEFAULT_LADDER_M),
                             detail=False)["dynamic_invalid_frames"] for k in range(len(rules.DEFAULT_LADDER_M))]
    assert counts == sorted(counts, reverse=True) and counts[0] > 0


def test_rle_round_trip():
    seq = codes(((I, 0, False, 0.0), 2), ((X, 0, False, 0.0), 3), ((C, 2, True, 3.4), 1), ((K, 8, True, 41.0), 2))
    assert rules.expand(rules.compress(seq), len(seq)) == seq


# ------------------------------------------------------------ motion states

def _states(positions, rotations_deg=None, K=6):
    """MOVING mask for one 0.1 m box following per-frame positions/rotations."""
    m = motion_states
    n = len(positions)
    M = np.tile(np.eye(4), (1, n, 1, 1))
    for t, pos in enumerate(positions):
        M[0, t, :3, 3] = pos
        if rotations_deg is not None:
            a = np.radians(rotations_deg[t])
            M[0, t, :2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    hid = np.zeros((1, n), dtype=np.int64)
    boxes = np.array([[[-0.05] * 3, [0.05] * 3]])
    change, tolerance = m.transition_changes(M, hid, boxes)
    return list(np.flatnonzero(m.classify(change[0], K, tolerance[0])))


def test_motion_A_entirely_static():
    assert _states([(0, 0, 0)] * 30) == []


def test_motion_B_static_move_static_has_no_tails():
    pos = [(0, 0, 0)] * 20 + [(0.01 * k, 0, 0) for k in range(1, 7)] + [(0.06, 0, 0)] * 14
    assert _states(pos) == list(range(20, 26))       # exactly the frames reached by a change


def test_motion_C_float_noise_is_static():
    # float32 round-off of a pose 1.5 m from the origin: +-1 ulp (1.2e-7 m)
    rng = np.random.default_rng(0)
    base = np.float32(1.5)
    pos = [tuple(float(np.float32(base + np.spacing(base) * k)) for k in rng.integers(-1, 2, 3)) for _ in range(30)]
    assert _states(pos) == []
    # the same noise 40 m out is still noise: the tolerance scales with float32 ulp
    far = np.float32(40.0)
    pos = [tuple(float(np.float32(far + np.spacing(far) * k)) for k in rng.integers(-2, 3, 3)) for _ in range(30)]
    assert _states(pos) == []


def test_motion_micron_scale_motion_is_moving():
    pos = [(0, 0, 0)] * 10 + [(5e-6 * k, 0, 0) for k in range(1, 5)] + [(2e-5, 0, 0)] * 10
    assert _states(pos) == [10, 11, 12, 13]


def test_motion_D_small_genuine_motion_is_moving():
    pos = [(0, 0, 0)] * 10 + [(0.0005 * k, 0, 0) for k in range(1, 5)] + [(0.002, 0, 0)] * 10
    assert _states(pos) == [10, 11, 12, 13]


def test_motion_E_rotation_about_a_fixed_origin_is_moving():
    rot = [0.0] * 10 + [2.0 * k for k in range(1, 6)] + [10.0] * 10
    assert _states([(0, 0, 0)] * 25, rot) == list(range(10, 15))


def test_motion_short_pause_inside_motion_stays_moving():
    # moves at transitions 5..7, pauses 3 frames (< 0.5 s = 6 frames), moves 11..12
    x = [0] * 5 + [1, 2, 3, 3, 3, 3, 4, 5] + [5] * 12
    assert _states([(0.01 * v, 0, 0) for v in x]) == list(range(5, 13))


def test_motion_K_motion_from_the_first_frame():
    pos = [(0.01 * k, 0, 0) for k in range(6)] + [(0.05, 0, 0)] * 20
    assert _states(pos) == list(range(0, 6))           # frame 0 uses its forward transition


def test_motion_L_motion_at_the_end_is_kept():
    pos = [(0, 0, 0)] * 20 + [(0.01 * k, 0, 0) for k in range(1, 5)]
    assert _states(pos) == [20, 21, 22, 23]


def test_motion_end_needs_half_a_second_of_evidence_to_settle():
    # last change at frame 20 of 23: frames 21-22 have only 2 future frames, so
    # the 0.5 s window is completed with past transitions, which include the
    # change -> still MOVING. With 0.5 s of stillness before the end: STATIC.
    pos = [(0, 0, 0)] * 17 + [(0.01 * k, 0, 0) for k in range(1, 5)] + [(0.04, 0, 0)] * 2
    assert _states(pos) == [17, 18, 19, 20, 21, 22]
    pos = [(0, 0, 0)] * 10 + [(0.01, 0, 0)] + [(0.01, 0, 0)] * 12
    assert _states(pos) == [10]


def test_motion_frames_for_half_a_second():
    m = motion_states
    assert (m.frames_for(0.5, 24), m.frames_for(0.5, 30), m.frames_for(0.5, 12), m.frames_for(0.5, 1)) == (12, 15, 6, 1)


# ------------------------------------------------------------ real geometry

def _box(sx, sy, sz):
    v = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (-sz / 2, sz / 2)],
                 dtype=np.float32)
    t = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return v, t


def _two_boxes(a, b):
    """Union mesh of two axis boxes given as (centre, size)."""
    va, ta = _box(*a[1]); vb, tb = _box(*b[1])
    v = np.concatenate([va + np.float32(a[0]), vb + np.float32(b[0])]).astype(np.float32)
    return v, np.concatenate([ta, tb + len(va)]).astype(np.int32)


def _write_scene(root: Path, n: int = 12):
    """One lane per case, 10 m apart; fps 12, so 0.5 s = 6 frames."""
    spec = importlib.util.spec_from_file_location("dycraft_evaluation", REPO / "animation" / "blender" / "evaluation.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    ev = root / "evaluation"
    ev.mkdir(parents=True)
    known = set()
    meshes = {k: evaluation.write_mesh(ev, *_box(*dims), known) for k, dims in {
        "table": (1.0, 1.0, 0.2), "cube": (0.1, 0.1, 0.1), "plate": (0.1, 0.1, 0.004),
        "torso": (0.3, 0.2, 0.5), "slab": (1.0, 1.0, 0.003), "treat": (0.02, 0.02, 0.02),
        "head": (0.2, 0.2, 0.2), "tooth": (0.02, 0.02, 0.04), "fang": (0.02, 0.02, 0.08)}.items()}
    # L-shaped "finger": a bar under the slab's edge, overlapping its underside by
    # 0.05 mm for 7 mm, and an upright outside the edge crossing the slab plane.
    under_top = -0.0015 + 0.00005
    finger = _two_boxes(((0.018, 0.0, under_top - 0.005), (0.05, 0.02, 0.01)),
                        ((0.050, 0.0, 0.012), (0.01, 0.02, 0.056)))
    meshes["finger"] = evaluation.write_mesh(ev, *finger, known)
    top, half = 0.1, 0.05
    sunk = lambda depth: top + half - depth           # cube center z for a penetration depth
    lane = {}

    def add(name, mesh, pos, identity=None, hidden=False):
        lane[name] = (identity, meshes[mesh], pos, hidden)

    # F / I: two bodies overlap 10 mm and never move -> diagnostic only
    add("T_static", "table", lambda f: (0, 0, 0))
    add("C_static", "cube", lambda f: (0, 0, sunk(0.010)))
    # G: overlap 10 mm, still for frames 0-5, slides 1 mm/frame from frame 6
    add("T_slide", "table", lambda f: (10, 0, 0))
    add("C_slide", "cube", lambda f: (10 + 0.001 * max(0, f - 5), 0, sunk(0.010)))
    # H: a moving "foot" descends into a static floor, 5 mm per frame from frame 4
    add("T_floor", "table", lambda f: (20, 0, 0))
    add("C_foot", "cube", lambda f: (20, 0, sunk(-0.020 if f < 4 else 0.005 * (f - 3))))
    # J: sinks 4 mm/frame over frames 0-5 to 20 mm, then settles there
    add("T_settle", "table", lambda f: (30, 0, 0))
    add("C_settle", "cube", lambda f: (30, 0, sunk(0.004 * min(f, 5))))
    # self: a moving torso; its hand is driven 20 mm into it from frame 6
    torso_x = lambda f: 40 + 0.01 * f
    add("robot_torso", "torso", lambda f: (torso_x(f), 0, 1.0), identity="robot/torso")
    add("robot_hand", "cube", lambda f: (torso_x(f) + 0.2 + (0.02 if f < 6 else -0.02), 0, 1.0),
        identity="robot/hand")
    # rigidly coupled overlap moving together -> exempt, never measured
    add("R_a", "cube", lambda f: (50 + 0.01 * f, 0, 1.0))
    add("R_b", "cube", lambda f: (50.08 + 0.01 * f, 0, 1.0))
    # threshold: 3 mm into a table while creeping: cube (tau 5 mm) no, plate (tau 2 mm) yes
    add("T_thr", "table", lambda f: (60, 0, 0))
    add("C_thr", "cube", lambda f: (60 + 0.001 * f, 0, sunk(0.003) if f >= 6 else sunk(-0.02)))
    add("T_plate", "table", lambda f: (70, 0, 0))
    add("P_plate", "plate", lambda f: (70 + 0.001 * f, 0, top + 0.002 - 0.003) if f >= 6 else (70, 0, 0.2))
    # hidden body inside a table -> ignored
    add("T_hidden", "table", lambda f: (80, 0, 0))
    add("C_hidden", "cube", lambda f: (80 + 0.001 * f, 0, 0.0), hidden=True)
    # re-entry: creeping; 10 mm at 0-3, clear 4-6, 10 mm at 7-9, clear 10-11
    add("T_reentry", "table", lambda f: (90, 0, 0))
    add("C_reentry", "cube", lambda f: (90 + 0.001 * f, 0, sunk(0.010) if f < 4 or 7 <= f <= 9 else sunk(-0.020)))
    # edge contact: a true separation of 0.05 mm, where a search pruned by projection bounds finds ~7 mm
    add("T_edge", "slab", lambda f: (110, 0, 0))
    add("F_edge", "finger", lambda f: (110.5 - 0.007 + 0.018 - 0.018, 0.001 * f, 0))
    # consumption: a treat enters the (moving) robot torso: 8 mm crossing at 3-4, wholly inside from 5
    add("Treat", "treat", lambda f: (torso_x(f) + (0.4 if f < 3 else 0.152 if f < 5 else 0.0), 0, 1.0))
    # construction self pairs: a tooth inside a moving head from the start (allowed), and a
    # fang 10 mm into it that is driven to 60 mm at frames 8-11 (counted, beyond construction)
    head_x = lambda f: 120 + 0.004 * f
    add("dog_head", "head", lambda f: (head_x(f), 0, 1.0), identity="dog/head")
    add("dog_tooth", "tooth", lambda f: (head_x(f) + 0.001 * (f % 2), 0, 1.0), identity="dog/tooth")
    add("dog_fang", "fang", lambda f: (head_x(f), 0, 1.0 - 0.1 - 0.04 + (0.06 if f >= 8 else 0.01)),
        identity="dog/fang")
    # deepening while moving: 8 mm from the start (creeping), 40 mm at frames 5-8, back to 8 mm
    add("T_deepen", "table", lambda f: (100, 0, 0))
    add("C_deepen", "cube", lambda f: (100 + 0.001 * f, 0, sunk(0.040 if 5 <= f <= 8 else 0.008)))

    for f in range(n):
        objects = []
        for k, (name, (identity, mesh, pos, hidden)) in enumerate(sorted(lane.items())):
            x, y, z = pos(f)
            m = [[1.0, 0, 0, float(np.float32(x))], [0, 1.0, 0, float(np.float32(y))],
                 [0, 0, 1.0, float(np.float32(z))], [0, 0, 0, 1.0]]
            objects.append({"key": f"o{k}", "name": name, "identity": identity, "room": False,
                            "matrix": m, "hidden": hidden, "color": [1, 1, 1, 1], "mesh": mesh})
        (ev / f"{f:06d}.json").write_text(json.dumps({"frame": f, "objects": objects, "transforms": {},
                                                      "joint_positions": {}}))
    (ev / "index.json").write_text(json.dumps({"fps": 12.0, "n_frames": n, "frame_start": 1,
                                               "format": "dycraft.evaluation.v2", "frames": list(range(n)),
                                               "complete": True, "declarations": {}}))
    return ev


def _settings(recompute=False):
    from dycraft.render import _blender_env as blender_env
    return Settings(blender_env.blender_python(), sys.executable, jobs=1, recompute=recompute)


def _run_cli(ev, results, recompute=False):
    inp = Input(path=ev, label="synthetic", evaluation_dir=ev)
    return pen.run(inp, results / "synthetic", _settings(recompute))


def _strip(result):
    return {k: v for k, v in result.items() if k != "timing"}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from dycraft.render import _blender_env as blender_env
    if not blender_env.blender_python():
        pytest.skip("bpy interpreter unavailable")
    if importlib.util.find_spec("coal") is None:
        pytest.skip("coal unavailable in the test interpreter")
    root = tmp_path_factory.mktemp("physical")
    ev = _write_scene(root)
    results = root / "results"
    return ev, results, _run_cli(ev, results)


def _flagged(result, a, b):
    for p in result["primary"]["flagged_pair_list"]:
        if {p["a"], p["b"]} == {a, b}:
            return p
    return None


def _static(result, a, b):
    for p in result["primary"]["static_overlap_list"]:
        if {p["a"], p["b"]} == {a, b}:
            return p
    return None


def _initial(result, a, b):
    for p in result["primary"]["initial_overlap_list"]:
        if {p["a"], p["b"]} == {a, b}:
            return p
    return None


def test_geometry_A_static_initial_overlap_is_exempt(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_static", "C_static") is None and _static(r, "T_static", "C_static") is None
    assert _initial(r, "T_static", "C_static")["exempt_frames"] == 12
    assert r["diagnostics"]["never_moving_pairs"]["overlapping"] >= 1


def test_geometry_C_initial_overlap_that_slides_stays_exempt(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_slide", "C_slide") is None
    init = _initial(r, "T_slide", "C_slide")
    assert init["exempt_frames"] == 12 and init["exempt_frames_while_moving"] == 6


def test_geometry_H_moving_foot_into_static_floor(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_floor", "C_foot")["frame_ranges"] == [[4, 11]]


def test_geometry_G_moves_in_then_settles(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_settle", "C_settle")["frame_ranges"] == [[2, 5]]       # 8, 12, 16, 20 mm
    assert _static(r, "T_settle", "C_settle")["frame_ranges"] == [[6, 11]]


def test_geometry_E_re_entry_counts(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_reentry", "C_reentry")["frame_ranges"] == [[7, 9]]
    ev = [e for e in r["primary"]["event_list"] if {e["a"], e["b"]} == {"T_reentry", "C_reentry"}]
    assert [e["reasons"] for e in ev] == [{"re-entry": 3}]


def test_geometry_F_deepening_while_moving_counts(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_deepen", "C_deepen")["frame_ranges"] == [[5, 8]]
    assert _initial(r, "T_deepen", "C_deepen")["counted_deepened_frames"] == 4


def test_geometry_self_penetration_counts(synthetic):
    _, _, r = synthetic
    p = _flagged(r, "robot/torso", "robot/hand")
    assert p is not None and p["relation"] == "self" and p["frame_ranges"] == [[6, 11]]


def test_geometry_rigid_pair_exempt(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "R_a", "R_b") is None and _static(r, "R_a", "R_b") is None
    assert ["R_a", "R_b"] in r["diagnostics"]["exemptions"]["rigid"]["examples"]


def test_geometry_threshold_behaviour(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_thr", "C_thr") is None                        # 3 mm < tau 5 mm
    assert _flagged(r, "T_plate", "P_plate")["frame_ranges"] == [[6, 11]]  # 3 mm > tau 2 mm
    sweep = {s["threshold_mm"]: s for s in r["sweep"]}
    assert sweep[1]["flagged_pairs"] > sweep[5]["flagged_pairs"]
    rates = [s["dynamic_penetration_frame_rate"] for s in r["sweep"]]
    assert rates == sorted(rates, reverse=True)


def test_geometry_hidden_body_ignored(synthetic):
    _, _, r = synthetic
    assert _flagged(r, "T_hidden", "C_hidden") is None and _static(r, "T_hidden", "C_hidden") is None


def test_geometry_headline_and_diagnostics(synthetic):
    _, _, r = synthetic
    p = r["primary"]
    assert p["evaluated_frames"] == 12
    assert p["dynamic_invalid_frame_ranges"] == [[2, 11]]
    assert p["all_geometric_penetration_frames"] == 12
    assert p["initial_overlap_exempt_pair_frames"] >= 12 + 12 + 4 + 8   # static, slide, re-entry, deepen
    assert p["motion_only_invalid_frames"] >= p["dynamic_invalid_frames"]
    m = r["diagnostics"]["motion"]
    assert m["window_frames"] == 6 and m["fps"] == 12.0


def test_geometry_depth_is_the_true_penetration(synthetic):
    """Depth = smallest separating translation; boxes sunk by known amounts."""
    _, results, _ = synthetic
    raw = json.loads((results / "synthetic" / "penetration_raw.json").read_text())
    timelines = {(p["a"], p["b"]): p["timeline"] for p in raw["pairs"]}

    def at(a, b, frame):
        for start, end, kind, _level, _tau, mm in timelines[(a, b)]:
            if start <= frame <= end and kind == 1:
                return mm
        raise AssertionError((a, b, frame))
    near = lambda got, want: abs(got - want) <= 0.15          # mm; bisection stops at 0.1 mm
    assert near(at("C_foot", "T_floor", 4), 5) and near(at("C_foot", "T_floor", 11), 40)
    assert near(at("C_settle", "T_settle", 8), 20)
    assert near(at("C_thr", "T_thr", 8), 3)
    assert near(at("P_plate", "T_plate", 8), 3)
    assert near(at("robot/hand", "robot/torso", 8), 20)
    assert near(at("C_static", "T_static", 0), 10)


def test_geometry_edge_contact_depth_is_not_overestimated(synthetic):
    _, results, r = synthetic
    raw = json.loads((results / "synthetic" / "penetration_raw.json").read_text())
    p = next(p for p in raw["pairs"] if {p["a"], p["b"]} == {"T_edge", "F_edge"})
    depths = [seg[5] for seg in p["timeline"] if seg[2] == 1]
    assert depths and max(depths) <= 0.2, depths          # 0.05 mm, not ~7 mm
    assert _flagged(r, "T_edge", "F_edge") is None


def test_geometry_consumed_object_is_excluded_and_reported(synthetic):
    _, _, r = synthetic
    cons = r["diagnostics"]["consumption"]
    assert any(c["object"] == "Treat" and c["agent"] == "robot" and c["consumed"] and c["enclosed_frames"] == [5, 11]
               for c in cons), cons
    flagged = _flagged(r, "Treat", "robot/torso")
    assert flagged is not None and flagged["frame_ranges"] == [[3, 4]]     # the ingestion itself still counts
    assert r["primary"]["consumed_frames"] == 7 and "robot" in r["diagnostics"]["agents"]


def test_geometry_construction_self_pairs(synthetic):
    _, _, r = synthetic
    allowed = {(p["a"], p["b"]): p for p in r["primary"]["self_allowed_list"]}
    assert _flagged(r, "dog/head", "dog/tooth") is None
    assert allowed[("dog/head", "dog/tooth")]["reason"] == "construction"
    fang = _flagged(r, "dog/fang", "dog/head")
    assert fang is not None and fang["frame_ranges"] == [[8, 11]]        # 60 mm >> 10 mm at the start


def test_output_is_reproducible(synthetic):
    """A re-run reuses the cached geometry, and a recomputation gives the same result."""
    ev, results, first = synthetic
    raw = results / "synthetic" / "penetration_raw.json"
    stamp = raw.stat().st_mtime_ns
    cached = _run_cli(ev, results)
    assert raw.stat().st_mtime_ns == stamp
    recomputed = _run_cli(ev, results, recompute=True)
    assert _strip(cached) == _strip(first) == _strip(recomputed)


def test_split_geometry_pass_equals_the_whole(synthetic):
    """On this scene, frames split over parallel children merge into the single-process result. (In a rare
    near-degenerate contact a chunk's cached depth can depend on where it starts: README.)"""
    ev, results, first = synthetic
    inp = Input(path=ev, label="synthetic", evaluation_dir=ev)
    split = pen.run(inp, results / "split", Settings(_settings().blender_python, sys.executable, jobs=3))
    assert split["primary"] == first["primary"] and split["sweep"] == first["sweep"]
    queued = pen.run(inp, results / "queued", Settings(_settings().blender_python, sys.executable, jobs=2, chunks=5))
    assert queued["primary"] == first["primary"] and queued["sweep"] == first["sweep"]   # 5 pieces, 2 at a time


# ------------------------------------------------------------ export wrapper

_BUILD_BLEND = r'''
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, 6
mesh = bpy.data.meshes.new("cube")
mesh.from_pydata([(x, y, z) for x in (-.05, .05) for y in (-.05, .05) for z in (-.05, .05)], [],
                 [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
cube = bpy.data.objects.new("Cube", mesh); scene.collection.objects.link(cube)
for f, x in ((1, 0.0), (6, 1.0)):
    cube.location = (x, 0, 0); cube.keyframe_insert("location", frame=f)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); scene.collection.objects.link(cam)
overlay = bpy.data.objects.new("Overlay", mesh.copy()); scene.collection.objects.link(overlay); overlay.parent = cam
text = bpy.data.objects.new("Caption", bpy.data.curves.new("Caption", "FONT")); scene.collection.objects.link(text)
curve = bpy.data.curves.new("Rail", "CURVE"); curve.dimensions = "3D"; curve.bevel_depth = 0.01
spline = curve.splines.new("POLY"); spline.points.add(1)
spline.points[0].co = (0, 1, 0, 1); spline.points[1].co = (1, 1, 0, 1)
rail = bpy.data.objects.new("Rail", curve); scene.collection.objects.link(rail)

def material(name, ior=1.5, alpha_texture=False, transmission=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = transmission
    if alpha_texture:
        noise = m.node_tree.nodes.new("ShaderNodeTexNoise")
        m.node_tree.links.new(noise.outputs["Fac"], bsdf.inputs["Alpha"])
    return m

for name, mat in (("Pond", material("water", ior=1.333, transmission=1.0)),
                  ("GlassBlock", material("glass", ior=1.47, transmission=1.0)),
                  ("Foam", material("foam", alpha_texture=True))):
    obj = bpy.data.objects.new(name, mesh.copy()); scene.collection.objects.link(obj)
    obj.location = (0, 3 + len(name), 0)
    obj.data.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
'''


def test_a_foreign_blend_is_exported_and_scored(tmp_path, pkgroot):
    """Any `.blend` scores through the CLI: exported read-only, captions and camera overlays removed, static curves
    converted, liquid and effect meshes classed by material, and nothing written beside the input."""
    from dycraft.render import _blender_env as blender_env
    interpreter = blender_env.blender_python()
    if not interpreter or importlib.util.find_spec("coal") is None:
        pytest.skip("bpy interpreter or coal unavailable")
    run = tmp_path / "bare_run"
    (run / "bundle").mkdir(parents=True)
    blend = run / "bundle" / "animation.blend"
    script = tmp_path / "build.py"
    script.write_text(_BUILD_BLEND)
    subprocess.run([interpreter, str(script), str(blend)], check=True, capture_output=True, timeout=300)
    before = blend.read_bytes()
    results = tmp_path / "results"
    cmd = [sys.executable, "-m", "dycraft.benchmark.score", str(run), "--out", str(results), "--jobs", "1",
           "--json", str(tmp_path / "rows.json")]
    done = subprocess.run(cmd, cwd=pkgroot, capture_output=True, text=True, timeout=900,
                          env={**__import__("os").environ, "PYTHONPATH": str(pkgroot)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert blend.read_bytes() == before                       # never modified
    (row,) = json.loads((tmp_path / "rows.json").read_text())
    assert row["kind"] == "blend" and row["frames"] == 6 and row["floating_rate"] is not None
    out = Path(row["results"])
    manifest = json.loads((out / "export" / "export_manifest.json").read_text())
    assert manifest["removed_by_reason"] == {"camera_overlay": 1, "text": 1}
    assert manifest["converted"] == [{"name": "Rail", "from": "CURVE"}]
    index = json.loads((out / "export" / "evaluation" / "index.json").read_text())
    assert index["complete"] and index["n_frames"] == 6
    assert sorted(index["collision_objects"]) == ["Cube", "Foam", "GlassBlock", "Pond", "Rail"]
    bodies = json.loads((out / "export" / "bodies.json").read_text())["objects"]
    assert bodies["Pond"].get("class") == "fluid" and bodies["Foam"].get("class") == "effect"
    assert "class" not in bodies["GlassBlock"] and "class" not in bodies["Cube"]
    result = json.loads((out / "penetration.json").read_text())
    assert result["input"]["exported"] and result["primary"]["evaluated_frames"] == 6
    assert result["diagnostics"]["motion"]["bodies_ever_moving"] == 1   # the cube; rail and camera overlay do not
    assert sorted(p.name for p in run.rglob("*")) == ["animation.blend", "bundle"]


# ------------------------------------------------------------ v3 frame records

def test_v3_binary_record_decodes(tmp_path):
    import hashlib
    import struct
    table = {"objects": ["link/a", "part/a/body"], "bodies": [["part/a/body", "a/body", False]], "joints": []}
    tid = hashlib.blake2b(json.dumps(table, sort_keys=True, separators=(",", ":")).encode(), digest_size=16).hexdigest()
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / f"{tid}.json").write_text(json.dumps(table))
    placement = np.eye(4, dtype="<f4"); placement[:3, 3] = (1.5, -2.0, 0.25)
    transforms = np.stack([np.eye(4, dtype="<f4"), placement])
    digest = bytes(range(32))
    payload = b"".join([struct.pack("<4sHHIII16s", b"DYCF", 3, 0, 2, 1, 0, bytes.fromhex(tid)),
                        transforms.tobytes(), digest, np.ones(4, "<f4").tobytes(),
                        b"", np.array([1], np.uint8).tobytes(), b"\x00" * 3, bytes(32)])
    (tmp_path / "000007.bin").write_bytes(payload)
    record, _ = records.read_record(tmp_path, 7)
    (body,) = record["objects"]
    assert body["name"] == "part/a/body" and body["identity"] == "a/body" and body["hidden"] is True
    assert body["mesh"] == digest.hex() and np.allclose(np.asarray(body["matrix"])[:3, 3], (1.5, -2.0, 0.25))
    records.check_format({"format": "dycraft.evaluation.v3"})
    with pytest.raises(ValueError):
        records.check_format({"format": "dycraft.evaluation.v9"})




def test_liquid_effect_pairs_are_dropped_from_scoring():
    raw = {"pairs": [{"a": "Canal_Water", "b": "Propeller_Wash", "a_name": "Canal_Water", "b_name": "Propeller_Wash"},
                     {"a": "east/bolt", "b": "east/actuator", "a_name": "part/east/bolt", "b_name": "part/east/actuator"},
                     {"a": "Canal_Water", "b": "buoy/seam", "a_name": "Canal_Water", "b_name": "part/buoy/seam"}]}
    kept, diag = pen.drop_material_bodies(raw, {"Canal_Water": "fluid", "Propeller_Wash": "effect"})
    assert [p["a"] for p in kept["pairs"]] == ["east/bolt"]
    assert diag["pairs_dropped"] == 2 and diag["by_class"] == {"effect": ["Propeller_Wash"], "fluid": ["Canal_Water"]}


def test_self_allowance_structural_construction_and_safeguard():
    base = rules.DEFAULT_PARAMS
    stable = {"relation": "self", "codes": codes(((C, 4, True, 10.0), 6), ((C, 4, True, 12.0), 3), ((C, 6, True, 45.0), 3))}
    pred, why = rules.self_allowance(stable, stable["codes"], 6, base)
    assert why == "construction" and not pred((C, 4, True, 12.0)) and pred((C, 6, True, 45.0))
    late = {"relation": "self", "codes": codes(((X, 0, False, 0.0), 4), ((C, 4, True, 10.0), 8))}
    assert rules.self_allowance(late, late["codes"], 6, base) == (None, None)
    joint = {"relation": "self", "jointed": True, "codes": codes(((C, 6, True, 175.0), 12))}
    pred, why = rules.self_allowance(joint, joint["codes"], 6, base)
    assert why == "structural" and not pred((C, 8, True, 200.0))


def test_consumption_needs_an_ingestion_contact():
    def pair(spec):
        return {"a": "x", "b": "agent/body", "assembly_a": "x", "assembly_b": "agent", "relation": "cross",
                "codes": codes(*spec)}
    ingested = pair([((X, 0, False, 0.0), 3), ((C, 3, True, 8.0), 2), ((K, 8, True, 90.0), 7)])
    raw = {"enclosures": [{"object": "x", "agent": "agent", "enclosed_frames": [[5, 11]]}]}
    raw["enclosures"][0].update(object_bodies=["x"], agent_bodies=["agent/body"])
    states = {"moving_intervals": {"x": [[0, 11]]}}
    _, report = pen.consumption(raw, [ingested], 12.0, states)
    assert report[0]["consumed"] and ingested["consumed"] == list(range(5, 12))
    born_inside = pair([((K, 8, True, 90.0), 12)])
    raw = {"enclosures": [{"object": "x", "agent": "agent", "enclosed_frames": [[0, 11]],
                           "object_bodies": ["x"], "agent_bodies": ["agent/body"]}]}
    _, report = pen.consumption(raw, [born_inside], 12.0, states)
    assert not report[0]["consumed"] and "consumed" not in born_inside


def test_consumption_rejects_a_static_object_swept_over():
    # a moving hull passing over a static water decal: enclosed after a contact, but the decal never moves, so it
    # was not brought in and is not consumed
    decal = {"a": "decal", "b": "boat/hull", "assembly_a": "decal", "assembly_b": "boat", "relation": "cross",
             "codes": codes(((X, 0, False, 0.0), 3), ((C, 3, True, 8.0), 2), ((K, 8, True, 90.0), 4), ((X, 0, False, 0.0), 3))}
    raw = {"enclosures": [{"object": "decal", "agent": "boat", "enclosed_frames": [[5, 8]],
                           "object_bodies": ["decal"], "agent_bodies": ["boat/hull"]}]}
    _, report = pen.consumption(raw, [decal], 12.0, {"moving_intervals": {"boat/hull": [[0, 11]]}}, persist_s=0.1)
    assert not report[0]["consumed"] and "never moved" in report[0]["rejected"] and "consumed" not in decal


def test_consumption_rejects_a_part_built_into_the_agent_and_brief_blips():
    states = {"moving_intervals": {"dog/muzzle": [[0, 11]], "dog/root": [[0, 11]]}}
    enc = {"object": "muzzle", "agent": "dog", "enclosed_frames": [[5, 11]],
           "object_bodies": ["dog/muzzle"], "agent_bodies": ["dog/root"]}
    # attached to the dog from the first frame: never a separate object
    built_in = {"a": "dog/muzzle", "b": "dog/root", "assembly_a": "muzzle", "assembly_b": "dog", "relation": "cross",
                "codes": codes(((C, 3, True, 8.0), 5), ((K, 8, True, 90.0), 7))}
    _, report = pen.consumption({"enclosures": [enc]}, [built_in], 12.0, states)
    assert not report[0]["consumed"] and "never separate" in report[0]["rejected"]
    # separate, contact, then enclosed for 3 frames and out again: not persistent
    blip = {"a": "dog/muzzle", "b": "dog/root", "assembly_a": "muzzle", "assembly_b": "dog", "relation": "cross",
            "codes": codes(((X, 0, False, 0.0), 3), ((C, 3, True, 8.0), 2), ((K, 8, True, 90.0), 3), ((X, 0, False, 0.0), 4))}
    _, report = pen.consumption({"enclosures": [{**enc, "enclosed_frames": [[5, 7]]}]}, [blip], 12.0, states)
    assert not report[0]["consumed"] and "not persistent" in report[0]["rejected"] and "consumed" not in blip


def test_construction_pair_deepening_late_is_scored_in_frame_order():
    # a stable 10 mm overlap at the start, then two separate deep episodes: frames are grouped into events in time
    # order, whatever the iteration order of the frame sets
    seq = codes(((C, 4, True, 10.0), 8), ((C, 6, True, 60.0), 3), ((C, 4, True, 10.0), 40),
                ((C, 6, True, 60.0), 5), ((C, 4, True, 10.0), 40))
    pair = {"a": "dog/body", "b": "dog/head", "relation": "self", "dynamic": [True] * len(seq), "codes": seq}
    s = rules.evaluate([pair], len(seq), 12.0, rules.passes_adaptive, list(rules.DEFAULT_LADDER_M))
    assert s["dynamic_invalid_frame_ranges"] == [[8, 10], [51, 55]] and s["events"] == 2


_BUILD_RIG = r'''
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig")); scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
# a flat rig: no bone has a parent, as generated characters often are
for name, head, tail in (("trunk", (0, 0, 1.0), (0, 0, 1.5)), ("upper", (0.2, 0, 1.4), (0.2, 0, 1.1)),
                         ("lower", (0.2, 0, 1.1), (0.2, 0, 0.8)), ("apart", (2, 0, 0), (2, 0, 0.3))):
    b = arm.data.edit_bones.new(name); b.head, b.tail = head, tail
bpy.ops.object.mode_set(mode="OBJECT")
def box(name, lo, hi, bone):
    m = bpy.data.meshes.new(name)
    m.from_pydata([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], [],
                  [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    o = bpy.data.objects.new(name, m); scene.collection.objects.link(o)
    mw = o.matrix_world.copy(); o.parent = arm; o.parent_type = "BONE"; o.parent_bone = bone
    bpy.context.view_layer.update(); o.matrix_world = mw
box("TrunkPart", (-0.25, -0.1, 1.0), (0.25, 0.1, 1.5), "trunk")
box("UpperPart", (0.15, -0.05, 1.1), (0.25, 0.05, 1.4), "upper")
box("LowerPart", (0.15, -0.05, 0.8), (0.25, 0.05, 1.1), "lower")
box("ApartPart", (1.9, -0.1, 0.0), (2.1, 0.1, 0.3), "apart")
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
'''


def test_rig_joints_of_a_flat_rig(tmp_path):
    from dycraft.render import _blender_env as blender_env
    interpreter = blender_env.blender_python()
    if not interpreter:
        pytest.skip("bpy interpreter unavailable")
    blend, script = tmp_path / "rig.blend", tmp_path / "build.py"
    script.write_text(_BUILD_RIG)
    subprocess.run([interpreter, str(script), str(blend)], check=True, capture_output=True, timeout=300)
    before = blend.read_bytes()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"blend": str(blend), "output": str(tmp_path / "rig.json")}))
    subprocess.run([interpreter, str(REPO / "benchmark" / "score" / "_rig_child.py"), str(request)],
                   check=True, capture_output=True, timeout=300)
    assert blend.read_bytes() == before
    rig = json.loads((tmp_path / "rig.json").read_text())
    assert rig["object_bone"]["LowerPart"] == "Rig:lower"
    links = {frozenset((a, b)): r for a, b, r in rig["connected"]}
    assert "head on tail" in links[frozenset(("Rig:upper", "Rig:lower"))]    # elbow
    assert "pivot in part" in links[frozenset(("Rig:trunk", "Rig:upper"))]   # shoulder in the trunk
    assert not any("Rig:apart" in k for k in links)                          # no joint to a far bone
    assert frozenset(("Rig:trunk", "Rig:lower")) not in links                 # only direct neighbors
    built = {frozenset((a, b)): d for a, b, d in rig["rest_overlaps"]}
    assert built[frozenset(("TrunkPart", "UpperPart"))] > 0                 # modeled overlapping
    assert frozenset(("TrunkPart", "ApartPart")) not in built


def test_rig_connected_self_pairs_are_structural_others_still_count():
    n = 4
    tl = rules.compress([(C, 3, True, 8.0)] * n)
    raw = {"n_frames": n, "pairs": [
        {"a": "arm/upper", "b": "arm/lower", "a_name": "U", "b_name": "L", "relation": "self", "timeline": tl},
        {"a": "arm/trunk", "b": "arm/lower", "a_name": "T", "b_name": "L", "relation": "self", "timeline": tl}]}
    states = {"moving_intervals": {"U": [[0, n - 1]], "L": [[0, n - 1]], "T": [[0, n - 1]]}}
    rig = ({"U": "R:upper", "L": "R:lower", "T": "R:trunk"}, {frozenset(("R:upper", "R:lower"))}, {})
    pairs = pen.attach_motion(raw, states, ({}, set()), rig)
    assert pairs[0].get("jointed") and not pairs[1].get("jointed")


def test_single_body_enclosure_candidates_come_from_containment_codes():
    # a morsel, one body of the trainer, contained in the dog's jaw then its body
    jaw = {"a": "dog/jaw", "b": "trainer/morsel", "a_name": "J", "b_name": "Mo", "assembly_a": "dog",
           "assembly_b": "trainer", "relation": "cross",
           "codes": codes(((X, 0, False, 0.0), 3), ((C, 3, True, 8.0), 1), ((5, 8, True, 90.0), 2), ((X, 0, False, 0.0), 6))}
    body = {"a": "trainer/morsel", "b": "dog/body", "a_name": "Mo", "b_name": "B", "assembly_a": "trainer",
            "assembly_b": "dog", "relation": "cross",
            "codes": codes(((X, 0, False, 0.0), 6), ((K, 8, True, 90.0), 6))}
    hand = {"a": "trainer/hand", "b": "trainer/morsel", "a_name": "H", "b_name": "Mo", "assembly_a": "trainer",
            "assembly_b": "trainer", "relation": "self", "codes": codes(((X, 0, False, 0.0), 12))}
    out = pen.body_enclosures([jaw, body, hand], {"dog"}, set())
    assert len(out) == 1
    enc, members = out[0]
    assert enc["object"] == "Mo" and enc["agent"] == "dog" and enc["enclosed_frames"] == [[4, 11]]
    assert enc["host"][4] == "J" and enc["host"][6] == "B" and enc["own_bodies"] == ["H"]
    assert pen.body_enclosures([jaw, body, hand], set(), set()) == []   # no agent: no candidate


def test_rest_pose_overlap_is_construction_and_only_far_deeper_counts():
    params = dict(rules.DEFAULT_PARAMS)
    seq = [(C, 3, True, 12.0)] * 6 + [(C, 5, True, 40.0)] * 2
    allow, why = rules.self_allowance({"rest_overlap_mm": 12.8}, seq, 6, params)
    assert why == "construction"
    assert not allow(seq[0]) and allow(seq[-1])          # 12 mm as built; 40 mm > 12.8 + 20
    assert rules.self_allowance({}, [(X, 0, False, 0.0)] * 8, 6, params) == (None, None)


def test_geometry_chunks_merge_by_frame():
    n = 6
    whole = [(X, 0, False, 0.0), (C, 3, True, 8.0), (C, 3, True, 8.0), (X, 0, False, 0.0), (K, 8, True, 90.0), (I, 0, False, 0.0)]
    def part(lo, hi, extra_static=False):
        seq = [whole[f] if lo <= f < hi else ((X, 0, False, 0.0) if whole[f][0] != I else whole[f]) for f in range(n)]
        pairs = [{"a": "a", "b": "b", "timeline": rules.compress(seq)},
                 {"a": "c", "b": "d", "timeline": rules.compress([(X, 0, False, 0.0)] * n)}]
        result = {"n_frames": n, "pairs": pairs, "errors": [], "stats": {"contact_queries": hi - lo},
                  "timing": {"total_s": 1.0}}
        if extra_static:
            pairs.append({"a": "s", "b": "t", "never_moving": True, "timeline": rules.compress([(C, 2, True, 3.0)] * n)})
            result["never_moving_pairs"] = {"measured": 1, "overlapping": 1, "aabb_candidates": 1}
        encs = [{"object": "o", "agent": "g", "enclosed_frames": [[4, 4]] if lo <= 4 < hi else [], "frames_tested": hi - lo}]
        return {**result, "enclosures": [e for e in encs if e["enclosed_frames"]]}
    # As `geometry` runs it: the frame chunks, then one more part, with no frames, for the never-moving pairs.
    merged = pen.merge_chunks([part(0, 2), part(2, 4), part(4, 6), part(0, 0, True)], [(0, 2), (2, 4), (4, 6), (0, 0)])
    ab = next(p for p in merged["pairs"] if p["a"] == "a")
    assert rules.expand(ab["timeline"], n) == whole                       # each frame from its own chunk
    assert not any(p["a"] == "c" for p in merged["pairs"])                # never penetrating: dropped
    assert [p["a"] for p in merged["pairs"] if p.get("never_moving")] == ["s"]   # from the last part, once
    assert merged["never_moving_pairs"]["measured"] == 1
    assert merged["enclosures"][0]["enclosed_frames"] == [[4, 4]]
    assert merged["stats"]["contact_queries"] == 6 and merged["timing"]["chunks"] == 4



def test_a_failed_chunk_stops_the_films_queue(tmp_path):
    """Once one part fails the film cannot be scored: no later part starts, and the running ones are stopped."""
    script = tmp_path / "part.py"
    script.write_text("import sys, time\nopen(sys.argv[2], 'w').write(sys.argv[1])\n"
                      "sys.exit(1) if sys.argv[1] == 'bad' else time.sleep(120)\n")
    jobs = [(label, [label, tmp_path / f"{label}.started"]) for label in ("bad", "slow", "late")]
    started = time.monotonic()
    with pytest.raises(SystemExit, match="bad"):
        run_children(Settings(sys.executable, sys.executable, jobs=2), str(script), jobs, tmp_path)
    assert time.monotonic() - started < 60
    assert not (tmp_path / "late.started").exists()

def test_an_exported_dycraft_bundle_is_scored_with_its_joint_tree_only(tmp_path, monkeypatch):
    """A DyCraft bundle without its evaluation is exported from its blend but scored like a delivered one: the export's
    Blender-parenting records, which would exempt welded parts the joint tree leaves measured, are not used. Any other
    blend is scored with them."""
    export_dir = tmp_path / "export"
    monkeypatch.setattr(pen, "export", lambda inp, out, settings: export_dir)
    dycraft = Input(path=tmp_path, label="d", blend=tmp_path / "animation.blend", baseline=tmp_path / "baseline.json")
    other = Input(path=tmp_path, label="o", blend=tmp_path / "animation.blend")
    assert pen.evaluation_of(dycraft, tmp_path, None) == (export_dir / "evaluation", None)
    assert pen.evaluation_of(other, tmp_path, None) == (export_dir / "evaluation", export_dir / "bodies.json")
