"""The floating rate (`benchmark/score`): its policy and its real geometry.

Pure-rule tests drive `support_rules` and `structure` directly. The end-to-end tests write one synthetic
`dycraft.evaluation.v2` export of boxes (one lane per edge case, 10 m apart, all on one large floor) and score it, so
the Coal distances, BVH contact samples and the cache are exercised as on a real film.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dycraft.benchmark.score import classes as CL
from dycraft.benchmark.score import floating as FL
from dycraft.benchmark.score import structure as S
from dycraft.benchmark.score import support_rules as R
from dycraft.benchmark.score.common import Settings
from dycraft.benchmark.score.inputs import Input, task_id_of

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- pure rules

def test_latch_enters_below_tau_and_leaves_above_exit():
    v = np.array([5, 3, 1.9, 2.5, 2.9, 3.1, 2.5, 1.0, np.inf, 1.0]) / 1000
    active = np.ones(10, dtype=bool)
    got = R.latch(v, active, 0.002, 0.003)
    assert got.tolist() == [False, False, True, True, True, False, False, True, False, True]


def test_latch_is_reset_by_an_inactive_frame():
    v = np.full(6, 0.0025)
    v[0] = 0.001
    active = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert R.latch(v, active, 0.002, 0.003).tolist() == [True, True, False, False, False, False]


def test_task_ids():
    """The brief a film answers comes from a path component; a short cut maps to its brief."""
    assert task_id_of("b17_marble_course_runs__dycraft_codex") == "B17"
    assert task_id_of("B15_dog_training_bare_claude") == "B15"
    assert task_id_of("B01S_g1_scrambles_eggs") == "B01"
    assert task_id_of("my_branching_domino_film") is None


def test_class_seeds_use_name_components():
    names = ["Water cutter / drain_apron", "Water / reservoir", "part/line/drip_tray", "glass_1/drink",
             "part/crow/claw_L0", "part/rig/float_guide"]
    idents = [None, None, "line/drip_tray", None, "crow/claw_L0", "rig/float_guide"]
    classes, _ = CL.seed_classes(names, idents, sidecar_classes={}, deforming=[False] * 6,
                                scale_changing=[False] * 6,
                                task_entry={"self_propelled_airborne": ["crow"], "buoyant": ["float"]})
    assert classes == [CL.SOLID, CL.FLUID, CL.SOLID, CL.FLUID, CL.AGENT, CL.EXEMPT]


def test_wash_is_fluid_only_as_a_water_effect():
    names = ["Propeller_Wash", "Bow wash", "Boat wake", "Wash basin", "Car wash sign board", "spring washer"]
    classes, _ = CL.seed_classes(names, [None] * 6, sidecar_classes={}, deforming=[False] * 6,
                                scale_changing=[False] * 6)
    assert classes == [CL.FLUID, CL.FLUID, CL.FLUID, CL.SOLID, CL.SOLID, CL.SOLID]


def test_cached_geometry_covers_classes_that_only_shrink_contact():
    measured = [CL.SOLID, CL.FLUID, CL.SOLID]
    assert FL.classes_compatible(measured, [CL.SOLID, CL.FLUID, CL.FLUID])      # solid -> fluid: reuse cache
    assert FL.classes_compatible(measured, [CL.CARRIER, CL.TEXT, CL.SOLID])    # contact-capable either way
    assert not FL.classes_compatible(measured, [CL.SOLID, CL.SOLID, CL.SOLID])  # fluid -> solid: re-measure


def test_class_seeds_remove_but_never_add():
    names = ["glass_1/drink", "water_tank/wall", "Water_FinalCached", "Hoist cord", "lounge_title", "boat/hull",
             "table/top"]
    deforming = [True, False, True, True, False, False, False]
    classes, reasons = CL.seed_classes(names, [None] * len(names), sidecar_classes={}, deforming=deforming,
                                      scale_changing=[False] * len(names),
                                      task_entry={"buoyant": ["boat"]})
    assert classes == [CL.FLUID, CL.SOLID, CL.FLUID, CL.CARRIER, CL.TEXT, CL.EXEMPT, CL.SOLID]
    assert "table/top" not in reasons


def _scene(n, fps, positions, box=(0.1, 0.1, 0.1)):
    """Minimal Scene for ballistic tests: one body, centroid = position."""
    M = np.zeros((1, n, 3, 4), dtype=np.float32)
    M[0, :, :3, :3] = np.eye(3)
    M[0, :, :, 3] = positions
    lo = -np.array(box) / 2
    raw = {"names": ["x"], "identities": [None], "n_frames": n, "fps": fps, "M": M,
           "hid": np.zeros((1, n), dtype=np.int32), "active": np.ones((1, n), dtype=bool),
           "mesh_lo": lo[None], "mesh_hi": -lo[None], "mesh_centroid": np.zeros((1, 3)),
           "mesh_volume": np.array([np.prod(box)]), "classes": [CL.SOLID],
           "pairs": np.zeros((0, 2), dtype=np.int32), "runs": np.zeros((0, 3), dtype=np.int32),
           "vals": np.zeros((0, 3), dtype=np.float32), "flags": np.zeros(0, dtype=np.uint8)}
    return R.Scene(raw, np.ones((1, n), dtype=bool), [], {})


@pytest.mark.parametrize("fps", [8.0, 24.0])
def test_free_fall_is_ballistic_at_any_frame_rate(fps):
    n = int(fps)
    t = np.arange(n) / fps
    pos = np.stack([0.5 * t, np.zeros(n), 3.0 - 0.5 * 9.81 * t ** 2], axis=1)
    fit = R.ballistic(_scene(n, fps, pos), (0,), 1, n - 1)
    assert fit["ballistic"], fit


def test_hover_drift_and_eased_drops_are_not_ballistic():
    n, fps = 12, 12.0
    t = np.arange(n) / fps
    hover = np.stack([np.zeros(n), np.zeros(n), np.full(n, 1.0)], axis=1)
    drift = np.stack([np.zeros(n), np.zeros(n), 1.0 - 0.3 * t], axis=1)
    s = np.clip(t / t[-1], 0, 1)
    eased = np.stack([np.zeros(n), np.zeros(n), 1.0 - (3 * s ** 2 - 2 * s ** 3)], axis=1)
    for pos in (hover, drift, eased):
        assert not R.ballistic(_scene(n, fps, pos), (0,), 1, n - 1)["ballistic"]


def test_too_few_samples_are_undetermined():
    pos = np.zeros((3, 3)); pos[:, 2] = 1.0
    fit = R.ballistic(_scene(3, 12.0, pos), (0,), 0, 0)
    assert not fit["determined"] and not fit["ballistic"]


def test_baseline_structure_pairs_skip_world_and_walk_fixed_chains():
    baseline = {
        "identities": {"chair/seat": {"link": "link/chair__seat"}, "chair/back": {"link": "link/chair__back"},
                       "chair/trim": {"link": "link/chair__seat"}, "door/leaf": {"link": "link/door__leaf"},
                       "door/frame": {"link": "link/door__frame"}},
        "joints": {"hinge": {"parent": "link/door__frame", "child": "link/door__leaf", "type": "revolute"}},
        "fixed_edges": [
            {"child": "origin/chair__fix", "parent": "link/chair__seat"},
            {"child": "joint/chair__fix", "parent": "origin/chair__fix"},
            {"child": "link/chair__back", "parent": "joint/chair__fix"},
            {"child": "origin/world__chair", "parent": "link/world"},
            {"child": "joint/world__chair", "parent": "origin/world__chair"},
            {"child": "link/chair__seat", "parent": "joint/world__chair"}]}
    idents = ["chair/seat", "chair/back", "chair/trim", "door/leaf", "door/frame"]
    got = {(min(a, b), max(a, b)): src for a, b, src in S.dycraft_pairs(idents, baseline)}
    # seat, trim (same link) and back (fixed joint) are one rigid group
    assert got[(0, 1)] == got[(0, 2)] == got[(1, 2)] == "rigid_group"
    assert got[(3, 4)] == "joint:revolute"
    assert len(got) == 4                                  # nothing through link/world


def test_joint_pairs_the_whole_rigid_groups():
    # lever pivots on the frame, but physically rides on an axle fixed to the frame
    baseline = {
        "identities": {"m/frame": {"link": "link/m__frame"}, "m/axle": {"link": "link/m__axle"},
                       "m/lever": {"link": "link/m__lever"}, "m/weight": {"link": "link/m__weight"}},
        "joints": {"pivot": {"parent": "link/m__frame", "child": "link/m__lever", "type": "revolute"}},
        "fixed_edges": [
            {"child": "origin/m__ax", "parent": "link/m__frame"}, {"child": "joint/m__ax", "parent": "origin/m__ax"},
            {"child": "link/m__axle", "parent": "joint/m__ax"},
            {"child": "origin/m__w", "parent": "link/m__lever"}, {"child": "joint/m__w", "parent": "origin/m__w"},
            {"child": "link/m__weight", "parent": "joint/m__w"}]}
    idents = ["m/frame", "m/axle", "m/lever", "m/weight"]
    got = {(min(a, b), max(a, b)): src for a, b, src in S.dycraft_pairs(idents, baseline)}
    assert got[(0, 1)] == "rigid_group" and got[(2, 3)] == "rigid_group"
    assert got[(1, 2)] == got[(0, 2)] == got[(1, 3)] == got[(0, 3)] == "joint:revolute"


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _place(R3, t, scale=(1, 1, 1)):
    return np.hstack([R3 @ np.diag(scale), np.reshape(t, (3, 1))])


def _pair_scene(post, part, gap, params=None):
    """A never-moving post and a moving part, in contact (`gap`, m) on every frame, no structure.
    post / part: (local box lo, hi, per-frame (n, 3, 4) placements)."""
    n = len(part[2])
    M = np.stack([post[2], part[2]]).astype(np.float32)
    lo, hi = np.array([post[0], part[0]], dtype=float), np.array([post[1], part[1]], dtype=float)
    raw = {"names": ["post", "part"], "identities": [None, None], "n_frames": n, "fps": 24.0, "M": M,
           "hid": np.array([[0] * n, [1] * n], dtype=np.int32), "active": np.ones((2, n), dtype=bool),
           "mesh_lo": lo, "mesh_hi": hi, "mesh_centroid": (lo + hi) / 2, "mesh_volume": np.prod(hi - lo, axis=1),
           "classes": [CL.SOLID, CL.SOLID], "pairs": np.array([[0, 1]], dtype=np.int32),
           "runs": np.array([[0, 0, n - 1]], dtype=np.int32),
           "vals": np.array([[gap, np.inf, np.inf]], dtype=np.float32), "flags": np.zeros(1, dtype=np.uint8)}
    moving = np.zeros((2, n), dtype=bool)
    moving[1] = True
    sc = R.Scene(raw, moving, [], params or {})
    R.find_anchors(sc)
    return sc


_STILL = [_place(np.eye(3), (0, 0, 0))] * 24
# a bracket wide enough that its corners drift > bond_drift_m per frame in the lever's frame, so
# only a pivot edge can hold the lever
_POST = ((-0.1, -0.1, -0.25), (0.1, 0.1, 0.25), _STILL)
# a lever on the bracket's axis (z through the origin), swinging 60 degrees
_LEVER = ((0.0, -0.01, -0.01), (0.4, 0.01, 0.01), [_place(_rz(60 * k / 23), (0, 0, 0)) for k in range(24)])


def test_undeclared_hinge_supports_the_part_that_turns_on_it():
    sc = _pair_scene(_POST, _LEVER, 0.0)
    assert R.evaluate(sc, detail=False)["floating_rate"] == 0.0
    assert set(R.pair_edges(sc)[0][3].tolist()) == {R.PIVOT}
    off = _pair_scene(_POST, _LEVER, 0.0, {"pivot_support": False})
    assert R.evaluate(off, detail=False)["floating_rate"] == 1.0


def test_pivot_holds_under_nonuniform_object_scale():
    # the same lever and post as unit boxes scaled by their placements
    post = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), [_place(np.eye(3), (0, 0, 0), (0.2, 0.2, 0.5))] * 24)
    lever = ((0.0, -0.5, -0.5), (1.0, 0.5, 0.5), [_place(_rz(60 * k / 23), (0, 0, 0), (0.4, 0.02, 0.02))
                                                  for k in range(24)])
    sc = _pair_scene(post, lever, 0.0)
    assert R.evaluate(sc, detail=False)["floating_rate"] == 0.0
    assert set(R.pair_edges(sc)[0][3].tolist()) == {R.PIVOT}


def test_spinning_while_touching_is_not_a_pivot():
    # a disc turning about its own axis (x = 0.3), rim touching the post's face: the axis misses the post
    post = ((-0.25, -0.25, -0.25), (0.2, 0.25, 0.25), _STILL)
    disc = ((-0.1, -0.1, -0.01), (0.1, 0.1, 0.01), [_place(_rz(90 * k / 23), (0.3, 0, 0)) for k in range(24)])
    sc = _pair_scene(post, disc, 0.0)
    assert R.evaluate(sc, detail=False)["floating_rate"] == 1.0
    _, held, _, kind = R.pair_edges(sc)[0]
    assert not ((kind == R.PIVOT) & (held == 1)).any()          # the disc is its own axle: the post holds nothing


def test_part_riding_on_an_axle_it_touches_turns_on_it():
    # a paddle whose inner edge touches a fixed axle and turns around it: the axis runs through the axle only
    axle = ((-0.02, -0.02, -0.25), (0.02, 0.02, 0.25), _STILL)
    paddle = ((0.02, -0.01, -0.05), (0.3, 0.01, 0.05), [_place(_rz(180 * k / 23), (0, 0, 0)) for k in range(24)])
    sc = _pair_scene(axle, paddle, 0.0)
    assert R.evaluate(sc, detail=False)["floating_rate"] == 0.0
    _, held, holder, kind = R.pair_edges(sc)[0]
    assert ((kind == R.PIVOT) & (held == 1) & (holder == 0)).all()
    assert R.evaluate(_pair_scene(axle, paddle, 0.0, {"pivot_support": False}), detail=False)["floating_rate"] == 1.0


def test_a_slowly_turning_hand_still_turns_on_its_boss():
    # a minute hand turns 0.3 deg in 17 s: too little to travel together (bonded) or to be static (mounted)
    hand = ((0.0, -0.01, -0.01), (0.4, 0.01, 0.01), [_place(_rz(0.5 * k / 23), (0, 0, 0)) for k in range(24)])
    assert R.evaluate(_pair_scene(_POST, hand, 0.0), detail=False)["floating_rate"] == 0.0
    assert R.evaluate(_pair_scene(_POST, hand, 0.0, {"pivot_support": False}), detail=False)["floating_rate"] == 1.0


def test_sliding_or_out_of_contact_is_not_a_pivot():
    slide = ((0.0, -0.01, -0.01), (0.4, 0.01, 0.01), [_place(np.eye(3), (0, 0, 0.01 * k)) for k in range(24)])
    assert R.evaluate(_pair_scene(_POST, slide, 0.0), detail=False)["floating_rate"] == 1.0
    # the hinge motion itself, 20 mm from the post: no contact, no pivot
    assert R.evaluate(_pair_scene(_POST, _LEVER, 0.02), detail=False)["floating_rate"] == 1.0


# ------------------------------------------------------------ real geometry

def _box(sx, sy, sz, z0=None):
    """Axis-aligned box centred at the origin (or spanning z in [z0 - sz, z0] if z0 given)."""
    zlo, zhi = (-sz / 2, sz / 2) if z0 is None else (z0 - sz, z0)
    v = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (zlo, zhi)],
                 dtype=np.float32)
    t = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return v, t


def _sphere(r, n_lat=50, n_lon=60):
    """UV sphere, vertices ordered top-down (so a capped, index-ordered sample
    would only ever see the top of it)."""
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    v = np.array([[r * np.sin(a) * np.cos(b), r * np.sin(a) * np.sin(b), r * np.cos(a)] for a in lat for b in lon],
                 dtype=np.float32)
    t = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, d = a + n_lon, b + n_lon
            t += [[a, c, b], [b, c, d]]
    return v, np.array(t, dtype=np.int32)


def _merge(*parts):
    vs, ts, off = [], [], 0
    for (v, t), shift in parts:
        vs.append(v + np.asarray(shift, dtype=np.float32)); ts.append(t + off); off += len(v)
    return np.concatenate(vs), np.concatenate(ts)


N_FRAMES, FPS, G = 24, 12.0, 9.81


def _write_scene(root: Path):
    spec = importlib.util.spec_from_file_location("dycraft_evaluation", REPO / "animation" / "blender" / "evaluation.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    ev = root / "evaluation"
    ev.mkdir(parents=True)
    known = set()
    mesh = lambda *dims, **kw: evaluation.write_mesh(ev, *_box(*dims, **kw), known)
    m = {"floor": mesh(400, 20, 0.1), "table": mesh(1, 1, 0.2), "cube": mesh(0.1, 0.1, 0.1),
         "wall": mesh(2, 0.1, 2), "base": mesh(0.3, 0.3, 0.3), "plat": mesh(0.5, 0.5, 0.1),
         "ped": mesh(0.2, 0.2, 1.0), "axle": mesh(0.04, 0.2, 1.0), "disc": mesh(0.3, 0.02, 0.3), "deck": mesh(3, 3, 0.1)}
    cords = [mesh(0.02, 0.02, 0.5 + 0.002 * f, z0=0.0) for f in range(N_FRAMES)]
    # a deforming "animal" body (skinned mesh: new digest every frame), bottom at local z = 0
    blobs = [mesh(0.4 + 0.004 * f, 0.2, 0.3, z0=0.3) for f in range(N_FRAMES)]
    # one-mesh trough (bottom + two walls): its box encloses the marble
    m["trough"] = evaluation.write_mesh(ev, *_merge((_box(1.0, 0.1, 0.01), (0, 0, 0.005)),
                                                     (_box(1.0, 0.01, 0.05), (0, 0.045, 0.035)),
                                                     (_box(1.0, 0.01, 0.05), (0, -0.045, 0.035))), known)
    m["marble"] = evaluation.write_mesh(ev, *_sphere(0.0125), known)
    lanes = {}

    def add(name, key, pos, hidden=lambda f: False, rot=lambda f: 0.0, flat=False):
        lanes[name] = (key, pos, hidden, rot, flat)

    add("floor", "floor", lambda f: (190, 0, -0.05))
    # 1: resting cup -> supported
    add("T1_table", "table", lambda f: (0, 0, 0.1))
    add("C1_cup", "cube", lambda f: (0, 0, 0.25))
    # 2: 2 cm above the table from frame 0 -> floating
    add("T2_table", "table", lambda f: (10, 0, 0.1))
    add("C2_hover", "cube", lambda f: (10, 0, 0.27))
    # 3: static cube flush against a wall -> mounted
    add("W3_wall", "wall", lambda f: (20, 0, 1.0))
    add("C3_picture", "cube", lambda f: (20, 0.1, 1.0))
    # 4: cube sliding along a wall -> floating (lateral only)
    add("W4_wall", "wall", lambda f: (30, 0, 1.0))
    add("C4_slide", "cube", lambda f: (29.5 + 0.02 * f, 0.1, 1.0))
    # 5: base slides on the floor, hand touches its side, box held from the side -> supported (bonded)
    bx5 = lambda f: 40 + 0.02 * f
    add("B5_base", "base", lambda f: (bx5(f), 0, 0.15))
    add("H5_hand", "cube", lambda f: (bx5(f) + 0.2, 0, 0.25))
    add("X5_box", "cube", lambda f: (bx5(f) + 0.3, 0, 0.25))
    # 6: ghost carry, 5 cm from the hand -> floating
    bx6 = lambda f: 50 + 0.02 * f
    add("B6_base", "base", lambda f: (bx6(f), 0, 0.15))
    add("H6_hand", "cube", lambda f: (bx6(f) + 0.2, 0, 0.25))
    add("X6_ghost", "cube", lambda f: (bx6(f) + 0.35, 0, 0.25))
    # 7: two touching cubes floating together -> floating (no anchor path)
    add("A7_seat", "cube", lambda f: (60, 0, 1.0))
    add("A7_base", "cube", lambda f: (60.1, 0, 1.0))
    # 8: cube riding a moving platform -> supported
    add("P8_platform", "plat", lambda f: (70 + 0.02 * f, 0, 0.05))
    add("C8_rider", "cube", lambda f: (70 + 0.02 * f, 0, 0.15))
    # 9: pedestal disappears at frame 4, cube falls ballistically, lands -> free flight
    fall = lambda f: max(0.05, 1.05 - 0.5 * G * (max(0, f - 3) / FPS) ** 2)
    add("D9_pedestal", "ped", lambda f: (80, 0, 0.5), hidden=lambda f: f >= 4)
    add("C9_drop", "cube", lambda f: (80, 0, fall(f)))
    # 10: same, but the cube descends at constant speed -> floating
    add("D10_pedestal", "ped", lambda f: (90, 0, 0.5), hidden=lambda f: f >= 4)
    add("C10_fake", "cube", lambda f: (90, 0, max(0.05, 1.05 - 0.2 * max(0, f - 3))))
    # 11: floating robot: base 5 cm above the floor, holding a box -> all floating
    bx11 = lambda f: 100 + 0.02 * f
    add("B11_base", "base", lambda f: (bx11(f), 0, 0.2))
    add("H11_hand", "cube", lambda f: (bx11(f) + 0.2, 0, 0.3))
    add("X11_box", "cube", lambda f: (bx11(f) + 0.3, 0, 0.3))
    # 12: pops in floating at frame 6 -> counted from frame 6
    add("C12_popin", "cube", lambda f: (110, 0, 1.0), hidden=lambda f: f < 6)
    # 13: weight on a deforming cord from a hook mounted on a wall -> supported (carrier)
    add("W13_wall", "wall", lambda f: (120, 0, 1.0))
    add("K13_hook", "cube", lambda f: (120, 0.1, 1.5))
    add("R13_cord", None, lambda f: (120, 0.1, 1.45))
    add("M13_weight", "cube", lambda f: (120, 0.1, 1.45 - (0.5 + 0.002 * f) - 0.05))

    # 14: carried in by a hand that stays grounded, released in mid-air at frame 8, stays there -> dynamic
    bx14 = lambda f: 130 + 0.02 * min(f, 8)
    add("B14_base", "base", lambda f: (bx14(f), 0, 0.15))
    add("H14_hand", "cube", lambda f: (bx14(f) + 0.2, 0, 0.25))
    add("X14_released", "cube", lambda f: (bx14(f) + 0.3 + (0.0 if f < 8 else 0.05), 0, 0.25))

    # 15: disc spinning on an axle through its center, no joint (bare style) -> supported (embedded)
    add("P15_axle", "axle", lambda f: (140, 0, 0.5))
    add("D15_disc", "disc", lambda f: (140, 0, 0.8), rot=lambda f: 15.0 * f)

    # 17: a large (9 m^2) static deck hovering 1 m up is not ground -> floating fixture (static)
    add("L17_mezzanine", "deck", lambda f: (160, 0, 1.0))
    # 18: deforming animal hovering 5 cm above the floor -> scored, floating (carriers are scored)
    add("A18_dog_hover", "blob", lambda f: (170, 0, 0.05))
    # 19: deforming animal standing on the floor -> supported by carrier contact
    add("A19_dog_stand", "blob", lambda f: (175, 0, 0.0))
    # 20: shelf held only by a bracket whose (group) name reads as fluid, joined by parenting
    #     (bodies.json) -> the fluid-classed bracket keeps its structural links: shelf supported
    add("W20_wall", "wall", lambda f: (180, 0, 1.0))
    add("Water / bracket 20", "cube", lambda f: (180, 0.1, 1.2))
    add("P20_shelf", "plat", lambda f: (180, 0.4, 1.2))
    # 21: a self-propelled agent (task manifest B14: crow) flying, a kernel held against it
    #     -> the crow is not scored and holds up what it carries
    add("crow body 21", "cube", lambda f: (190 + 0.03 * f, 0, 2.0))
    add("K21_kernel", "cube", lambda f: (190.1 + 0.03 * f, 0, 2.0))
    # 22: a boat (buoyant, exempt as the brief's floating body) drifting on no support, a crate resting on it
    #     -> the boat is not scored and carries the crate
    add("B22_boat", "plat", lambda f: (200 + 0.02 * f, 0, 0.3))
    add("K22_crate", "cube", lambda f: (200 + 0.02 * f, 0, 0.4))
    # 23: a sticker scaled to zero thickness riding a moving base (singular matrix in the grip check)
    add("Y23_base", "base", lambda f: (210 + 0.02 * f, 0, 0.15))
    add("Z23_sticker", "plat", lambda f: (210 + 0.02 * f, 0, 0.3), flat=True)
    # 24: cubes sliding over a table: 3 mm up -> supported at tau_p 5 mm; 8 mm up -> floating
    add("T24_table", "table", lambda f: (220, 0, 0.1))
    add("C24_gap3mm", "cube", lambda f: (219.8 + 0.01 * f, -0.2, 0.253))
    add("C24_gap8mm", "cube", lambda f: (219.8 + 0.01 * f, 0.2, 0.258))
    # 25: a gear sliding on a static deck that hovers 0.5 m up (base not grounded)
    #     -> strict: floating; local static support: supported; the deck is a grounding failure
    add("P25_deck", "plat", lambda f: (230, 0, 0.5))
    add("G25_gear", "cube", lambda f: (229.9 + 0.01 * f, 0, 0.6))
    # 26: a moving cube 3 cm above another hovering static deck -> floating under both rules
    add("P26_deck", "plat", lambda f: (235, 0, 0.5))
    add("L26_ghost", "cube", lambda f: (234.9 + 0.01 * f, 0, 0.63))
    # 16: dense marble (3000 vertices) rolling along a trough on a pedestal -> supported
    add("P16_pedestal", "ped", lambda f: (150, 0, 0.5))
    add("T16_trough", "trough", lambda f: (150, 0, 1.0))
    add("M16_marble", "marble", lambda f: (149.7 + 0.02 * f, 0, 1.01 + 0.0125))

    for f in range(N_FRAMES):
        objects = []
        for k, (name, (key, pos, hidden, rot, flat)) in enumerate(sorted(lanes.items())):
            x, y, z = pos(f)
            c, s_ = np.cos(np.radians(rot(f))), np.sin(np.radians(rot(f)))   # about +y
            mat = [[float(np.float32(c)), 0, float(np.float32(s_)), float(np.float32(x))],
                   [0, 1.0, 0, float(np.float32(y))],
                   [float(np.float32(-s_)), 0, float(np.float32(c)), float(np.float32(z))], [0, 0, 0, 1.0]]
            if flat:                               # scaled to zero thickness: a singular matrix
                for row in mat[:3]:
                    row[2] = 0.0
            objects.append({"key": f"o{k}", "name": name, "identity": None, "room": False, "matrix": mat,
                            "hidden": bool(hidden(f)), "color": [1, 1, 1, 1],
                            "mesh": cords[f] if key is None else blobs[f] if key == "blob" else m[key]})
        (ev / f"{f:06d}.json").write_text(json.dumps({"frame": f, "objects": objects, "transforms": {},
                                                      "joint_positions": {}}))
    # bare-style sidecar: Blender parenting (structure) for lane 20
    (root / "bodies.json").write_text(json.dumps({
        "objects": {}, "hidden_frames": {}, "types": {n: "MESH" for n in lanes},
        "parents": {"P20_shelf": "Water / bracket 20", "Water / bracket 20": "W20_wall"}}))
    (ev / "index.json").write_text(json.dumps({"fps": FPS, "n_frames": N_FRAMES, "frame_start": 1,
                                               "format": "dycraft.evaluation.v2", "frames": list(range(N_FRAMES)),
                                               "complete": True, "declarations": {}}))
    return ev


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from dycraft.render import _blender_env as blender_env
    if not blender_env.blender_python():
        pytest.skip("bpy interpreter unavailable")
    if importlib.util.find_spec("coal") is None:
        pytest.skip("coal unavailable in the test interpreter")
    root = tmp_path_factory.mktemp("floating")
    ev = _write_scene(root)
    return _score(ev, root / "results" / "synthetic"), (ev, root / "results")


#: The synthetic scene's brief: a self-propelled crow (as B14) and a buoyant boat (as B22).
_TASK = {"self_propelled_airborne": ["crow", "bird"], "buoyant": ["boat"]}


def _score(ev, out, jobs=1, recompute=False, chunks=None):
    from dycraft.render import _blender_env as blender_env
    settings = Settings(blender_env.blender_python(), sys.executable, jobs=jobs, chunks=chunks, recompute=recompute)
    original = CL.task_entry
    CL.task_entry = lambda task: _TASK
    try:
        return FL.run(Input(path=ev, label="synthetic", evaluation_dir=ev, task="B14"), out, settings)
    finally:
        CL.task_entry = original


def _offenders(result):
    return {b["name"]: b for b in result["diagnostics"]["offending_bodies"]}


def _static(result):
    return {b["name"]: b for b in result["diagnostics"]["static_floating"]["top"]}


def test_geometry_expected_offenders(synthetic):
    result, _ = synthetic
    got = _offenders(result)
    # dynamic (animated) bodies: counted by the headline
    assert set(got) == {"C4_slide", "X6_ghost", "C10_fake", "B11_base", "H11_hand", "X11_box",
                        "X14_released", "A18_dog_hover", "C24_gap8mm", "L26_ghost"}, sorted(got)
    # released at frame 8, then hovering motionless: still counted, while at rest too
    assert got["X14_released"]["first_frame"] == 8 and got["X14_released"]["frames"] == N_FRAMES - 8
    assert got["X14_released"]["moving_frames"] < got["X14_released"]["frames"]
    # never-moving fixtures: reported, not counted
    assert set(_static(result)) == {"C2_hover", "A7_seat", "A7_base", "C12_popin", "L17_mezzanine",
                                    "P25_deck", "P26_deck"}, sorted(_static(result))


def test_geometry_first_frame_and_pop_in(synthetic):
    got = _static(synthetic[0])
    assert got["C2_hover"]["first_frame"] == 0 and got["C2_hover"]["frames"] == N_FRAMES
    assert got["C12_popin"]["first_frame"] == 6 and got["C12_popin"]["frames"] == N_FRAMES - 6
    # an animated body floating from its first frame counts from frame 0
    assert _offenders(synthetic[0])["X6_ghost"]["first_frame"] == 0


def test_geometry_causes(synthetic):
    got = {**_offenders(synthetic[0]), **_static(synthetic[0])}
    assert got["C2_hover"]["dominant_cause"] == "no_contact"
    assert got["C4_slide"]["dominant_cause"] == "lateral_only_contact"
    assert got["A7_seat"]["dominant_cause"] == "no_anchor_path"
    assert got["X6_ghost"]["dominant_cause"] == "no_contact"
    assert got["H11_hand"]["dominant_cause"] == "no_anchor_path"


def test_geometry_free_fall_is_legal_fake_fall_is_not(synthetic):
    result, _ = synthetic
    ff = result["diagnostics"]["free_flight"]
    assert any(s["bodies"] == ["C9_drop"] for s in ff["segments"]), ff
    assert _offenders(result)["C10_fake"]["frames"] >= 4


def test_geometry_excluded_bodies_keep_structure_and_agents_hold(synthetic):
    result, _ = synthetic
    d = result["diagnostics"]
    assert d["classes"]["fluid"] == ["Water / bracket 20"] and d["classes"]["agent"] == ["crow body 21"]
    assert "P20_shelf" not in _static(result) and "P20_shelf" not in _offenders(result)
    assert "K21_kernel" not in _offenders(result) and "crow body 21" not in _offenders(result)
    assert d["anchors"]["agents"] == ["B22_boat", "crow body 21"]
    assert "K22_crate" not in _offenders(result) and "B22_boat" not in _offenders(result)


def test_geometry_local_static_support(synthetic):
    result, _ = synthetic
    p, d = result["primary"], result["diagnostics"]
    headline = {b["name"] for b in d["offending_bodies"]}
    strict = {b["name"] for b in d["strict_ground_path"]["offending_bodies"]}
    # the gear on a static deck that hovers: supported by the headline, charged only by the strict rule
    assert "G25_gear" not in headline and "G25_gear" in strict
    assert "L26_ghost" in headline                     # 3 cm above a static deck: still floating
    assert p["strict_floating_rate"] > p["floating_rate"]
    assert d["strict_ground_path"]["inherited_body_frames"] == \
        p["strict_floating_body_frames"] - p["floating_body_frames"] > 0
    # the deck's own missing ground contact is counted once, as a static grounding failure
    assert p["static_grounding_failure_rate"] > 0
    assert any("P25_deck" in g for g in d["static_floating"]["ungrounded_static_examples"])


def test_geometry_mounted_and_anchor(synthetic):
    d = synthetic[0]["diagnostics"]
    assert "C3_picture" in [x["name"] for x in d["mounted_only_supported"]["top"]]
    assert [g["name"] for g in d["anchors"]["ground"]] == ["floor"]
    assert set(d["anchors"]["panels"]) == {"W3_wall", "W4_wall", "W13_wall", "W20_wall"}
    assert d["classes"]["carrier"] == ["A18_dog_hover", "A19_dog_stand", "R13_cord"]


def test_geometry_headline(synthetic):
    p = synthetic[0]["primary"]
    # headline: per dynamic body-frame; the frame-level rate saturates (some lane floats throughout)
    assert p["floating_rate"] == p["floating_body_frames"] / p["animated_body_frames"]
    assert 0 < p["floating_rate"] < 0.5
    assert p["total_frames"] == N_FRAMES and p["floating_frames"] == N_FRAMES and p["floating_frame_rate"] == 1.0
    assert p["all_floating_frame_rate"] == 1.0 and p["static_floating_bodies"] == 7
    offenders = synthetic[0]["diagnostics"]["offending_bodies"]
    assert p["floating_body_frames"] == sum(b["frames"] for b in offenders)


def test_geometry_contact_tolerance(synthetic):
    """The cube 3 mm above a table rests on it at the 5 mm contact tolerance and floats at 2 mm."""
    result, (ev, results) = synthetic
    folder = results / "synthetic"
    raw = FL.load_raw(folder / "floating_raw.npz")
    states = json.loads((folder / "motion.json").read_text())
    tight = FL.apply_rules(raw, states, ev.parent / "bodies.json", None, {"tau_p": 0.002})
    names = {b["name"] for b in tight["diagnostics"]["offending_bodies"]}
    assert "C24_gap3mm" in names and "C24_gap3mm" not in _offenders(result)
    assert tight["primary"]["floating_rate"] > result["primary"]["floating_rate"]


def test_geometry_cached_rerun_is_identical(synthetic):
    """A re-run reuses the cached geometry pass and gives the same result."""
    result, (ev, results) = synthetic
    raw = results / "synthetic" / "floating_raw.npz"
    stamp = raw.stat().st_mtime_ns
    again = _score(ev, results / "synthetic")
    assert raw.stat().st_mtime_ns == stamp
    strip = lambda r: {k: v for k, v in r.items() if k != "timing"}
    assert strip(again) == strip(result)


def test_reach_follows_support_direction_and_fallback_agrees(monkeypatch):
    # 0 anchor; 1 rests on 0 (passive, traversed 0 -> 1); 2 rests on 1; 3 has 2 resting on it
    # (traversal 3 -> 2 only, so 3 is NOT reached); 4 -- 1 symmetric.
    src = np.array([0, 1, 3, 4, 1])
    dst = np.array([1, 2, 2, 1, 4])
    got = R._reach(5, np.array([0]), src, dst)
    assert got.tolist() == [True, True, True, False, True]
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    assert R._reach(5, np.array([0]), src, dst).tolist() == got.tolist()
    assert not R._reach(5, np.array([], dtype=np.int64), src, dst).any()


def test_geometry_split_pass_equals_single_pass(synthetic):
    """On this scene, frames measured by parallel children merge into the single-process pass."""
    result, (ev, results) = synthetic
    split = _score(ev, results / "split", jobs=3)
    assert split["stats"]["chunks"] == 3
    assert split["primary"] == result["primary"]
    queued = _score(ev, results / "queued", jobs=2, chunks=5)       # 5 pieces, 2 at a time
    assert queued["stats"]["chunks"] == 5 and queued["primary"] == result["primary"]
    strip = lambda d: {k: v for k, v in d.items() if k != "classes"}
    assert strip(split["diagnostics"]) == strip(result["diagnostics"])
