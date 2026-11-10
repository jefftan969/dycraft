"""Runtime-behavior tests for the freeform probe library (`sdk/_probes.py`).

Each test exercises a runtime computation
against an independently-derived expected value. No tests round-trip
field names or restate constructor arguments.
"""

from __future__ import annotations

import math
import pytest

from dycraft.sdk import Object, Scene, _probes
from dycraft.sdk._object_kernel import (
    SATResult, sat_obb_obb, obb_of_part, _compute_frames, _project_obb_half,
)
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton


@pytest.fixture(autouse=True)
def _reset_singletons():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


# ──────────────────────────────────────────────────────────────────
# Shared parameterized fixtures
# ──────────────────────────────────────────────────────────────────


def _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(2.0, 0.0, 0.0),
                        extent=1.0):
    """Two cubes of side `extent`. Part `a` is the root at world `a_xyz`;
    part `b` is attached to `a` so its world center sits at `b_xyz`.
    """
    obj = Object("aligned")
    a = obj.make_box("a", extent, extent, extent)
    b = obj.make_box("b", extent, extent, extent)
    # b's center at world b_xyz; a is at world a_xyz (root).
    offset = (b_xyz[0] - a_xyz[0], b_xyz[1] - a_xyz[1], b_xyz[2] - a_xyz[2])
    obj.attach_fixed(name="b_to_a", parent=a, child=b,
                     parent_anchor=obj.anchor(a, "center", translation=offset),
                     child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    return obj.finalize()


def _two_rotated_boxes(rot_deg=30.0, gap=0.1):
    """Two unit cubes: A unrotated at origin (root), B rotated `rot_deg`
    about Z and offset (1.0 + gap, 0, 0) so two faces meet at gap=0.
    `gap > 0` → disjoint along x; `gap < 0` → overlap.
    """
    obj = Object("rotated")
    a = obj.make_box("a", 1.0, 1.0, 1.0)
    b = obj.make_box("b", 1.0, 1.0, 1.0)
    obj.attach_fixed(name="b_to_a", parent=a, child=b,
                     parent_anchor=obj.anchor(a, "center", translation=(1.0 + gap, 0.0, 0.0), rotation=(0.0, 0.0, rot_deg)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    return obj.finalize()


# ──────────────────────────────────────────────────────────────────
# 1-6: pair_report core fields (object layer)
# ──────────────────────────────────────────────────────────────────


def test_pair_report_signed_distance_axis_aligned_overlap():
    """Two unit cubes overlapping by 0.5 in x → signed_distance = -0.5."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(0.5, 0.0, 0.0))
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["collided"] is True
    assert r["signed_distance"] == pytest.approx(-0.5, abs=1e-9)
    # MTV direction must be along ±x.
    nx, ny, nz = r["contact_normal"]
    assert abs(ny) < 1e-9 and abs(nz) < 1e-9
    assert abs(nx) == pytest.approx(1.0, abs=1e-9)


def test_pair_report_rotated_intersecting_uses_obb_not_aabb():
    """Two cubes overlapping with one rotated 30° around Z. The OBB-OBB SAT
    result must differ from the loose-AABB overlap, exercising the rotated
    SAT path."""
    ir = _two_rotated_boxes(rot_deg=30.0, gap=-0.1)  # overlap by 0.1 in x
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["collided"] is True
    # Independently compute the SAT depth via the kernel.
    frames = _compute_frames(ir)
    obb_a = obb_of_part(ir, "a", frames)
    obb_b = obb_of_part(ir, "b", frames)
    sat = sat_obb_obb(obb_a, obb_b)
    assert sat.collided is True
    assert r["signed_distance"] == pytest.approx(sat.signed_distance, abs=1e-12)


def test_pair_report_rotated_separated_reports_positive_signed_distance():
    """A rotated pair with definite separation must report
    `collided=False`, `signed_distance > 0`, and a `None` contact_point.
    Exercises the new total-SAT separated path.
    """
    # B rotated 30° at offset 1.5 in x — clearly separated (gap ~ 0.4 between
    # the rotated cube's extents and A's right face).
    ir = _two_rotated_boxes(rot_deg=30.0, gap=0.5)  # b center at x=1.5
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["collided"] is False
    assert r["signed_distance"] > 0.0
    assert r["contact_point"] is None
    # Independently confirm via SAT kernel.
    frames = _compute_frames(ir)
    sat = sat_obb_obb(obb_of_part(ir, "a", frames), obb_of_part(ir, "b", frames))
    assert sat.collided is False
    assert sat.contact_point is None
    assert r["signed_distance"] == pytest.approx(sat.signed_distance, abs=1e-12)


def test_pair_report_signed_overlap_disjoint():
    """signed_overlap negative when intervals are disjoint."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(2.0, 0.0, 0.0))
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["signed_overlap"]["x"] == pytest.approx(0.5 - 1.5, abs=1e-12)


def test_pair_report_signed_overlap_partial():
    """signed_overlap positive when intervals overlap (partial)."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(0.5, 0.0, 0.0))
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["signed_overlap"]["x"] == pytest.approx(0.5, abs=1e-12)


def test_pair_report_signed_overlap_full_containment():
    """signed_overlap equals the inner box's extent on that axis when fully
    contained."""
    obj = Object("containment")
    a = obj.make_box("a", 2.0, 2.0, 2.0)  # extent 2 → x ∈ [-1, 1]
    b = obj.make_box("b", 0.6, 0.6, 0.6)  # extent 0.6 → x ∈ [-0.3, 0.3]
    obj.attach_fixed(name="b_to_a", parent=a, child=b,
                     parent_anchor=obj.anchor(a, "center"),
                     child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    r = _probes.object_pair_report(ir, "a", "b")
    # min(1, 0.3) − max(-1, -0.3) = 0.3 − (-0.3) = 0.6 (extent of inner box).
    assert r["signed_overlap"]["x"] == pytest.approx(0.6, abs=1e-12)


def test_pair_report_contact_tol_boundary():
    """`collided = signed_distance < -contact_tol`. With signed_distance =
    -1e-3, three tol values produce three outcomes.
    """
    # Construct a pair with signed_distance = -1e-3 (= 1 mm penetration).
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(0.999, 0.0, 0.0))
    # Two unit cubes centered 0.999 apart → overlap = 0.001 → SD = -0.001.

    # Default contact_tol = 1e-6: -0.001 < -1e-6 → collided.
    r = _probes.object_pair_report(ir, "a", "b")
    assert r["collided"] is True

    # contact_tol = 5e-4 (= 0.5 mm): -0.001 < -5e-4 → collided.
    r2 = _probes.object_pair_report(ir, "a", "b", contact_tol=5e-4)
    assert r2["collided"] is True

    # contact_tol = 2e-3 (= 2 mm): -0.001 < -2e-3 is False → not collided.
    r3 = _probes.object_pair_report(ir, "a", "b", contact_tol=2e-3)
    assert r3["collided"] is False


def test_pair_report_center_delta_world():
    """center_delta = c_b - c_a where c_x is each part's world AABB center."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(1.0, 2.0, 3.0))
    r = _probes.object_pair_report(ir, "a", "b")
    dx, dy, dz = r["center_delta"]
    assert dx == pytest.approx(1.0, abs=1e-9)
    assert dy == pytest.approx(2.0, abs=1e-9)
    assert dz == pytest.approx(3.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────
# 7-8: within_report
# ──────────────────────────────────────────────────────────────────


def test_within_report_boundary():
    """Inner unit cube at (0.5, 0, 0) inside outer cube of side 2 at origin:
    upper_margin.x = 1.0 - 1.0 = 0 (boundary touch), lower_margin.x = 0 - (-1) = 1.
    Shift inner to (0.6, 0, 0) → upper_margin.x = -0.1, within=False."""
    obj = Object("within")
    root = obj.make_box("root", 0.001, 0.001, 0.001)
    inner = obj.make_box("inner", 1.0, 1.0, 1.0)
    outer = obj.make_box("outer", 2.0, 2.0, 2.0)
    obj.attach_fixed(name="i_r", parent=root, child=inner,
                     parent_anchor=obj.anchor(root, "center", translation=(0.5, 0.0, 0.0)), child_anchor=obj.anchor(inner, "center"))
    obj.attach_fixed(name="o_r", parent=root, child=outer,
                     parent_anchor=obj.anchor(root, "center"),
                     child_anchor=obj.anchor(outer, "center"))
    obj.declare_root(root)
    ir = obj.finalize()

    r = _probes.object_within_report(ir, "inner", "outer")
    assert r["within"] is True
    assert r["margins"]["x"]["lower_margin"] == pytest.approx(1.0, abs=1e-9)
    assert r["margins"]["x"]["upper_margin"] == pytest.approx(0.0, abs=1e-9)


def test_within_report_violation_outside():
    """Inner box shifted past outer face → upper_margin negative."""
    obj = Object("within_out")
    root = obj.make_box("root", 0.001, 0.001, 0.001)
    inner = obj.make_box("inner", 1.0, 1.0, 1.0)
    outer = obj.make_box("outer", 2.0, 2.0, 2.0)
    obj.attach_fixed(name="i_r", parent=root, child=inner,
                     parent_anchor=obj.anchor(root, "center", translation=(0.6, 0.0, 0.0)), child_anchor=obj.anchor(inner, "center"))
    obj.attach_fixed(name="o_r", parent=root, child=outer,
                     parent_anchor=obj.anchor(root, "center"),
                     child_anchor=obj.anchor(outer, "center"))
    obj.declare_root(root)
    ir = obj.finalize()

    r = _probes.object_within_report(ir, "inner", "outer")
    assert r["within"] is False
    assert r["margins"]["x"]["upper_margin"] == pytest.approx(-0.1, abs=1e-9)


# ──────────────────────────────────────────────────────────────────
# 9-10: nearest_neighbors / find_clearance_risks
# ──────────────────────────────────────────────────────────────────


def _ranked_neighborhood():
    """Query part `q` at origin (root); 5 candidates at known x-axis
    offsets so their signed distances from `q` are known:
      c_overlap     at x = +0.5   (overlap on x: -0.5)
      c_close       at x = +1.01  (gap: 0.01)
      c_mid_a       at x = +1.5   (gap: 0.5)
      c_mid_b       at x = +1.5 z=+2  (gap: 0.5 on z to keep parts distinct)
      c_far         at x = +2.0   (gap: 1.0)
    Each part has unit edges.
    """
    obj = Object("ranked")
    q = obj.make_box("q", 1.0, 1.0, 1.0)
    for name, xpos in [("c_overlap", 0.5),
                       ("c_close", 1.01),
                       ("c_mid_a", 1.5),
                       ("c_mid_b", 1.5),
                       ("c_far", 2.0)]:
        c = obj.make_box(name, 1.0, 1.0, 1.0)
        obj.attach_fixed(name=f"{name}_to_q", parent=q, child=c,
                         parent_anchor=obj.anchor(q, "center", translation=(xpos, 0.0, 0.0)), child_anchor=obj.anchor(c, "center"))
    obj.declare_root(q)
    return obj.finalize()


def test_nearest_neighbors_ranking_and_values():
    """Verify rank key (collided, signed_distance asc) and per-entry values.
    Expected signed_distances from q (unit cube at origin, x ∈ [-0.5, 0.5]):
      c_overlap at x=0.5 (x ∈ [0.0, 1.0]) → overlap of 0.5 → SD = -0.5
      c_close at x=1.01 (x ∈ [0.51, 1.51]) → gap 0.01 → SD = +0.01
      c_mid_a/_b at x=1.5 (x ∈ [1.0, 2.0]) → gap 0.5 → SD = +0.5
      c_far at x=2.0 (x ∈ [1.5, 2.5]) → gap 1.0 → SD = +1.0
    """
    ir = _ranked_neighborhood()
    entries = _probes.object_nearest_neighbors(ir, "q", limit=5)
    # Verify order via signed_distance (collided first, then ascending).
    names = [e["b"] for e in entries]
    assert names[0] == "c_overlap"
    assert entries[0]["collided"] is True
    assert entries[0]["signed_distance"] == pytest.approx(-0.5, abs=1e-9)
    assert names[1] == "c_close"
    assert entries[1]["signed_distance"] == pytest.approx(0.01, abs=1e-9)
    # c_mid_a, c_mid_b tied at 0.5 — order between them is implementation
    # defined but both must precede c_far.
    mid_names = sorted(names[2:4])
    assert mid_names == ["c_mid_a", "c_mid_b"]
    assert entries[2]["signed_distance"] == pytest.approx(0.5, abs=1e-9)
    assert entries[3]["signed_distance"] == pytest.approx(0.5, abs=1e-9)
    assert names[4] == "c_far"
    assert entries[4]["signed_distance"] == pytest.approx(1.0, abs=1e-9)

    # With limit=3, exactly the first three (collided + two smallest gaps).
    top3 = _probes.object_nearest_neighbors(ir, "q", limit=3)
    assert [e["b"] for e in top3] == names[:3]


def test_find_clearance_risks_pool_restriction():
    """Restrict the pool; assert no pair contains an excluded part, and the
    returned top-k matches an independently-computed ranking of the pool.
    """
    ir = _ranked_neighborhood()
    pool = ["q", "c_overlap", "c_close", "c_far"]
    risks = _probes.object_find_clearance_risks(ir, parts=pool, limit=3)
    # All pairs must be subsets of the pool.
    for r in risks:
        assert r["a"] in pool and r["b"] in pool
    # Top entry must be the only colliding pair in the pool: q ↔ c_overlap.
    top = risks[0]
    assert {top["a"], top["b"]} == {"q", "c_overlap"}
    assert top["collided"] is True


# ──────────────────────────────────────────────────────────────────
# SAT promotion parity on existing colliding cases
# ──────────────────────────────────────────────────────────────────


def test_sat_promotion_parity_matches_independent_kernel_call():
    """The new total `sat_obb_obb` returns a SATResult whose
    `signed_distance == -old_depth` and `axis == old_normal` on existing
    colliding fixtures. Independent kernel call confirms the result."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(0.5, 0.0, 0.0))
    frames = _compute_frames(ir)
    obb_a = obb_of_part(ir, "a", frames)
    obb_b = obb_of_part(ir, "b", frames)
    res = sat_obb_obb(obb_a, obb_b)
    assert isinstance(res, SATResult)
    assert res.collided is True
    # Two unit cubes overlapping by 0.5 in x: depth = 0.5, signed_distance = -0.5.
    assert res.signed_distance == pytest.approx(-0.5, abs=1e-12)
    # MTV direction along +x or -x.
    assert abs(res.axis[1]) < 1e-12 and abs(res.axis[2]) < 1e-12
    assert abs(res.axis[0]) == pytest.approx(1.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────
# Verifier behavior preserved at default contact_tol
# ──────────────────────────────────────────────────────────────────


def test_no_self_penetration_pass_at_eps_boundary():
    """A pair with overlap in (-_PENETRATION_EPS, 0] must NOT be reported.
    Verifier threshold = `_PENETRATION_EPS = 1e-6`."""
    from dycraft.verify.object_verify import no_self_penetration
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0),
                            b_xyz=(1.0 + 1e-7, 0.0, 0.0))
    rpt = no_self_penetration(ir)
    assert rpt.verdict == "pass", f"epsilon-near-touching pair should pass: {rpt.evidence}"


def test_no_self_penetration_fail_above_eps():
    """A pair with -1e-3 overlap → must be reported."""
    from dycraft.verify.object_verify import no_self_penetration
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0),
                            b_xyz=(0.999, 0.0, 0.0))
    rpt = no_self_penetration(ir)
    assert rpt.verdict == "fail"
    pairs = rpt.evidence["pairs"]
    assert len(pairs) == 1
    assert {pairs[0]["a"], pairs[0]["b"]} == {"a", "b"}


# ──────────────────────────────────────────────────────────────────
# Self-pair documented trivial result
# ──────────────────────────────────────────────────────────────────


def test_pair_report_self_pair_documented_trivial():
    """`pair_report(ir, 'a', 'a')` returns trivially: signed_overlap.x =
    full x-extent of part `a`."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(2.0, 0.0, 0.0))
    r = _probes.object_pair_report(ir, "a", "a")
    assert r["collided"] is True
    # Full extents: signed_overlap is the part's extent on each axis.
    assert r["signed_overlap"]["x"] == pytest.approx(1.0, abs=1e-12)
    assert r["signed_overlap"]["y"] == pytest.approx(1.0, abs=1e-12)
    assert r["signed_overlap"]["z"] == pytest.approx(1.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────
# KeyError on missing names
# ──────────────────────────────────────────────────────────────────


def test_collision_probes_raise_keyerror_on_missing_names():
    ir = _two_aligned_boxes()
    with pytest.raises(KeyError):
        _probes.object_pair_report(ir, "nonexistent_part", "a")
    with pytest.raises(KeyError):
        _probes.object_within_report(ir, "nonexistent_part", "a")


# ──────────────────────────────────────────────────────────────────
# Probe overlay wiring smoke
# ──────────────────────────────────────────────────────────────────


def test_object_probe_overlay_exposes_new_symbols():
    """Smoke: the probe overlay aliases the four new collision probes and
    the four resolver helpers to bare names accessible from a `probe`
    expression context.
    """
    from dycraft.agent.layers.layer import object_layer_config
    cfg = object_layer_config()
    # Build a dummy ir to drive the overlay factory.
    ir = _two_aligned_boxes()
    overlay = cfg.probe_overlay_factory(None, ir)
    assert overlay is not None
    for name in ("pair_report", "within_report", "nearest_neighbors",
                 "find_clearance_risks",
                 "joint_axis_world", "part_world_aabb",
                 "resolve_entity_aabb", "resolve_direction"):
        assert name in overlay, f"{name} missing from object probe overlay"
        assert callable(overlay[name])


# ──────────────────────────────────────────────────────────────────
# Scene-layer twins of a subset of the object-layer tests
# ──────────────────────────────────────────────────────────────────


def _scene_two_instances(b_offset=(2.0, 0.0, 0.0)):
    """Scene with two instances of a unit-cube object, separated by `b_offset`."""
    obj = Object("unit")
    body = obj.make_box("body", 1.0, 1.0, 1.0)
    obj.declare_root(body)
    obj_ir = obj.finalize()

    scene = Scene("twin")
    scene.set_room(width=10.0, depth=10.0, height=5.0)
    from dycraft.sdk.scene_builder import ObjectHandle
    handle = ObjectHandle(obj_ir)
    # Anchor-based placement lands the cube's bbox-bottom at pose z, so z=0
    # rests the (1 m tall) cube on the floor with its center at 0.5 — the
    # geometry (center at 0.5, bottom flush with the floor) these probe tests
    # expect.
    scene.place(handle, name="a", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    scene.place(handle, name="b",
                pose=((b_offset[0], b_offset[1], 0.0 + b_offset[2]),
                      (0.0, 0.0, 0.0)))
    ir = scene.finalize()
    return ir, {obj_ir["id"]: obj_ir}


def test_scene_pair_report_signed_overlap_disjoint():
    """Scene-twin of test 1 / 4 — two unit cubes 2 m apart in x."""
    ir, catalog = _scene_two_instances(b_offset=(2.0, 0.0, 0.0))
    r = _probes.scene_pair_report(ir, catalog, "a", "b")
    assert r["collided"] is False
    # Cubes at (0, 0, 0.5) and (2, 0, 0.5). World AABBs x ∈ [-0.5, 0.5] and
    # [1.5, 2.5]. signed_overlap.x = min(0.5, 2.5) - max(-0.5, 1.5) = -1.0.
    assert r["signed_overlap"]["x"] == pytest.approx(-1.0, abs=1e-9)
    # signed_distance via SAT on axis-aligned boxes is the separation along
    # the max-margin axis = 1.0.
    assert r["signed_distance"] == pytest.approx(1.0, abs=1e-9)


def test_scene_pair_report_contact_tol_boundary():
    """Scene twin of test 5 — verify contact_tol gating."""
    ir, catalog = _scene_two_instances(b_offset=(0.999, 0.0, 0.0))
    r = _probes.scene_pair_report(ir, catalog, "a", "b")
    assert r["collided"] is True
    r2 = _probes.scene_pair_report(ir, catalog, "a", "b", contact_tol=2e-3)
    assert r2["collided"] is False


def test_scene_within_report_room_containment():
    """Instance fully inside the room → within=True on xyz."""
    ir, catalog = _scene_two_instances(b_offset=(2.0, 0.0, 0.0))
    r = _probes.scene_within_report(ir, catalog, "a", "room", axes="xyz")
    assert r["within"] is True
    # Room is 10 × 10 × 5, instance a is 1 × 1 × 1 at (0, 0, 0.5).
    # Room AABB: x ∈ [-5, 5], y ∈ [-5, 5], z ∈ [0, 5].
    # Instance AABB: x ∈ [-0.5, 0.5], y ∈ [-0.5, 0.5], z ∈ [0, 1].
    assert r["margins"]["x"]["lower_margin"] == pytest.approx(4.5, abs=1e-9)
    assert r["margins"]["z"]["lower_margin"] == pytest.approx(0.0, abs=1e-9)


def test_scene_nearest_neighbors_ranking():
    """3 instances on x-axis at known offsets; verify ranking."""
    obj = Object("unit")
    body = obj.make_box("body", 1.0, 1.0, 1.0)
    obj.declare_root(body)
    obj_ir = obj.finalize()

    scene = Scene("rank3")
    scene.set_room(width=20.0, depth=10.0, height=5.0)
    from dycraft.sdk.scene_builder import ObjectHandle
    h = ObjectHandle(obj_ir)
    scene.place(h, name="q", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    scene.place(h, name="close", pose=((1.5, 0.0, 0.5), (0.0, 0.0, 0.0)))  # gap 0.5
    scene.place(h, name="far", pose=((3.0, 0.0, 0.5), (0.0, 0.0, 0.0)))    # gap 2.0
    ir = scene.finalize()
    catalog = {obj_ir["id"]: obj_ir}

    entries = _probes.scene_nearest_neighbors(ir, catalog, "q", limit=5)
    names = [e["b"] for e in entries]
    # `close` (gap 0.5) before `far` (gap 2.0).
    assert names[0] == "close"
    assert entries[0]["signed_distance"] == pytest.approx(0.5, abs=1e-9)
    assert names[1] == "far"
    assert entries[1]["signed_distance"] == pytest.approx(2.0, abs=1e-9)


def test_scene_pair_report_self_pair_trivial():
    ir, catalog = _scene_two_instances()
    r = _probes.scene_pair_report(ir, catalog, "a", "a")
    assert r["collided"] is True
    assert r["signed_overlap"]["x"] == pytest.approx(1.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────
# Bucket 2+3 tests: pose-tagging, mirror probes, sampled CCD, scene mirrors
# ──────────────────────────────────────────────────────────────────


def _hinge_ir():
    """body + arm hinged on body.right; q=0 → arm flush; q=180° → arm
    sweeps through body. Returns the IR.
    """
    obj = Object("hinge")
    body = obj.make_box("body", 1.0, 1.0, 1.0)
    arm = obj.make_box("arm", 0.5, 0.5, 0.5)
    obj.attach_revolute(
        name="h", parent=body, child=arm,
        axis=(0, 0, 1), limits=(0.0, 180.0), default=0.0,
        parent_anchor=obj.anchor(body, "right"),
        child_anchor=obj.anchor(arm, "left"),
    )
    obj.declare_root(body)
    return obj.finalize()


def test_pair_report_with_poses_shifts_result():
    """pair_report at q=180° must show the arm interpenetrating body.

    Probe `poses=` uses SDK surface units: degrees for revolute. 180.0
    here is 180 degrees (= math.pi rad in IR-internal units).
    """
    ir = _hinge_ir()
    # At q=0 (rest): arm flush against body, no overlap.
    rest = _probes.object_pair_report(ir, "body", "arm")
    assert rest["collided"] is False
    # At q=180° (degrees on the surface): arm rotates through body.
    r = _probes.object_pair_report(ir, "body", "arm",
                                    poses={"h": 180.0})
    assert r["collided"] is True
    assert r["signed_distance"] < 0.0


def test_within_report_with_poses():
    """When the arm rotates, its world AABB changes; within_report
    reflects the pose. Surface units = degrees.
    """
    ir = _hinge_ir()
    # At q=90° (degrees), arm's world AABB shifts.
    r = _probes.object_within_report(ir, "arm", "body", axes="xyz",
                                      poses={"h": 90.0})
    # The result is some dict; just verify it ran without error and
    # has the expected shape.
    assert "within" in r
    assert "margins" in r
    assert set(r["margins"].keys()) == {"x", "y", "z"}


def test_nearest_neighbors_with_poses_changes_distance():
    """At q=180° the arm is on the opposite side of the body; the
    nearest-neighbor distance must differ from rest pose.
    """
    ir = _hinge_ir()
    rest = _probes.object_nearest_neighbors(ir, "body")
    assert len(rest) == 1
    sd_rest = rest[0]["signed_distance"]
    rotated = _probes.object_nearest_neighbors(ir, "body",
                                                poses={"h": 180.0})
    sd_rotated = rotated[0]["signed_distance"]
    assert sd_rest != pytest.approx(sd_rotated, abs=1e-6)


def test_find_clearance_risks_with_poses():
    """Sampled pose changes the ranking output. Surface units = degrees."""
    ir = _hinge_ir()
    rest = _probes.object_find_clearance_risks(ir)
    rotated = _probes.object_find_clearance_risks(ir, poses={"h": 180.0})
    # Different poses give different signed_distance values
    assert rest[0]["signed_distance"] != pytest.approx(
        rotated[0]["signed_distance"], abs=1e-6)


def test_penetration_pairs_sampled_returns_worst_witnessing_pose():
    """A hinge that sweeps through the body should yield exactly one
    PenetrationPair from sampled=True, with witnessing_pose populated."""
    ir = _hinge_ir()
    # Rest pose: nothing overlaps.
    rest = _probes.object_penetration_pairs(ir)
    assert rest == []
    # Sampled: the swept arm hits the body.
    sampled = _probes.object_penetration_pairs(ir, sampled=True)
    assert len(sampled) == 1
    p = sampled[0]
    assert {p.a, p.b} == {"arm", "body"}
    # witnessing_pose must be a non-empty dict with the hinge joint key.
    assert p.witnessing_pose is not None
    assert "h" in p.witnessing_pose


def test_penetration_pairs_sampled_and_poses_mutually_exclusive():
    ir = _hinge_ir()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _probes.object_penetration_pairs(ir, sampled=True,
                                          poses={"h": 30.0})


def test_distance_between_euclidean_and_signed():
    """distance_between(axis=None) is Euclidean; axis=tuple is signed
    projection (antisymmetric in a, b)."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(3.0, 4.0, 0.0))
    # Centers are 5 m apart.
    d = _probes.object_distance_between(ir, ("a", "center"), ("b", "center"))
    assert d == pytest.approx(5.0, abs=1e-9)
    # Projected onto +x: dx = 3.
    d_x = _probes.object_distance_between(
        ir, ("a", "center"), ("b", "center"), axis=(1, 0, 0))
    assert d_x == pytest.approx(3.0, abs=1e-9)
    # Swap order → antisymmetric.
    d_x_swap = _probes.object_distance_between(
        ir, ("b", "center"), ("a", "center"), axis=(1, 0, 0))
    assert d_x_swap == pytest.approx(-3.0, abs=1e-9)


def test_overlap_along_signed_boundary():
    """overlap_along must match the design's signed-projection convention."""
    ir = _two_aligned_boxes(a_xyz=(0.0, 0.0, 0.0), b_xyz=(0.5, 0.0, 0.0))
    # x: a ∈ [-0.5, 0.5], b ∈ [0, 1] → overlap = 0.5
    assert _probes.object_overlap_along(ir, "a", "b", axis=(1, 0, 0)) == \
        pytest.approx(0.5, abs=1e-9)
    # y: full overlap = 1.0
    assert _probes.object_overlap_along(ir, "a", "b", axis=(0, 1, 0)) == \
        pytest.approx(1.0, abs=1e-9)


def test_angle_between_specific_angles_and_zero_norm():
    """angle_between produces specific angles for canonical directions;
    raises ValueError on zero-norm input."""
    ir = _two_aligned_boxes()
    assert _probes.object_angle_between(ir, (1, 0, 0), (1, 0, 0)) == \
        pytest.approx(0.0, abs=1e-9)
    assert _probes.object_angle_between(ir, (1, 0, 0), (0, 1, 0)) == \
        pytest.approx(90.0, abs=1e-9)
    assert _probes.object_angle_between(ir, (1, 0, 0), (-1, 0, 0)) == \
        pytest.approx(180.0, abs=1e-9)
    # Non-unit input is normalized internally.
    assert _probes.object_angle_between(ir, (5, 0, 0), (0, 7, 0)) == \
        pytest.approx(90.0, abs=1e-9)
    with pytest.raises(ValueError):
        _probes.object_angle_between(ir, (0, 0, 0), (1, 0, 0))


def test_motion_displacement_at_rotated_pose():
    """The arm's center, attached via revolute at body.right, sweeps in an
    arc of radius equal to (arm center - pivot)."""
    ir = _hinge_ir()
    # arm center at rest (q=0), computed independently:
    # pivot = body.right_world = (-0.5, 0, 0).
    # arm.left_local = (xmax_arm, 0, 0) = (+0.25, 0, 0); arm origin at
    # world such that arm.left lands at pivot. So arm origin = pivot -
    # (0.25, 0, 0) = (-0.75, 0, 0). arm center is at arm origin = (-0.75, 0, 0).
    # vector from pivot to arm center: (-0.25, 0, 0). |v| = 0.25.
    # At q=90° around +Z through pivot: arm center rotates around pivot;
    # vector goes from (-0.25, 0, 0) to (0, -0.25, 0). Displacement
    # magnitude (Euclidean from q=0 to q=90°): √((0.25)² + (0.25)²) = 0.354.
    # Probe `to_pose=` is in SDK surface units: degrees for revolute.
    disp = _probes.object_motion_displacement(
        ir, "arm", anchor="center", from_pose={}, to_pose={"h": 90.0})
    assert disp == pytest.approx(math.sqrt(2) / 4, abs=1e-6)
    # Along axis (1, 0, 0): change in x from -0.75 to (-0.5 + 0) = -0.5.
    # Δx = -0.5 - (-0.75) = +0.25.
    disp_x = _probes.object_motion_displacement(
        ir, "arm", anchor="center", from_pose={}, to_pose={"h": 90.0},
        axis=(1, 0, 0))
    assert disp_x == pytest.approx(0.25, abs=1e-6)


def test_support_report_kinematic_edge_credit():
    """A hinge attaches arm to body; support_report credits the kinematic
    edge so arm is supported even without face contact at rest pose."""
    ir = _hinge_ir()
    r = _probes.object_support_report(ir)
    assert r["is_supported"] is True
    assert "arm" not in r["unsupported"]
    # Kinematic edge between body and arm appears in contacts.
    assert ("arm", "body") in r["contacts"] or ("body", "arm") in r["contacts"]


def test_scene_distance_between():
    """Distance between two instance centers in scene layer."""
    ir, catalog = _scene_two_instances(b_offset=(3.0, 4.0, 0.0))
    d = _probes.scene_distance_between(
        ir, catalog, ("a", "center"), ("b", "center"))
    # a at (0, 0, 0.5), b at (3, 4, 0.5). Distance = 5.
    assert d == pytest.approx(5.0, abs=1e-9)


def test_scene_angle_between():
    """Scene-layer angle_between uses the same scalar as object layer."""
    ir, catalog = _scene_two_instances()
    deg = _probes.scene_angle_between(ir, catalog, (1, 0, 0), (0, 1, 0))
    assert deg == pytest.approx(90.0, abs=1e-9)


def test_scene_room_containment_margins():
    """Instance inside the room → within=True with positive margins."""
    ir, catalog = _scene_two_instances()
    r = _probes.scene_room_containment_margins(ir, catalog, "a")
    assert r["within"] is True
    # Room 10×10×5; instance 1×1×1 at (0, 0, 0.5).
    # z lower_margin = inner_zmin - outer_zmin = 0 - 0 = 0
    assert r["margins"]["z"]["lower_margin"] == pytest.approx(0.0, abs=1e-9)


def test_scene_support_report_floor_contact():
    """Instance flush with floor (z=0.5 with extent 1.0 → zmin = 0) is
    grounded via floor face contact."""
    ir, catalog = _scene_two_instances()
    r = _probes.scene_support_report(ir, catalog)
    assert r["is_supported"] is True
    # Both instances should be grounded directly via floor face contact.
    assert set(r["grounded"]) >= {"a", "b"}


# list_parts — where a part name comes from


# ──────────────────────────────────────────────────────────────────
# The scene layer's exact-collision probes
# ──────────────────────────────────────────────────────────────────


class _FakeScenePayload:
    """Enough of `SceneAgentCatalog` for the probe overlay factory."""

    def __init__(self, catalog, asset_base_dir):
        self.object_handles = {}
        self.catalog = catalog
        self.asset_base_dir = asset_base_dir


class _FakeSceneCtx:
    def __init__(self, catalog, asset_base_dir="/staged/catalog"):
        self.layer_payload = _FakeScenePayload(catalog, asset_base_dir)
        # `ToolContext.asset_base_dir` is a property over the payload; the
        # binder only does a `getattr`, so mirroring the resolved value is
        # enough and keeps this fake from having to be a dataclass.
        self.asset_base_dir = asset_base_dir
        self.adjudication_cache = {}
        self.run_dir = None


def test_the_scene_layer_can_ask_an_exact_collision_question():
    """`gap` and `room_gap` must be reachable from a scene probe expression
    (the scene layer's probe overlay, `_scene_layer_probe_overlay`, builds on
    `_scene_probe_overlay`).

    Every other collision probe this layer offers is a bounding volume —
    `penetration_pairs` and `pair_report` are the SAT and AABB broad phases,
    `room_containment_margins` and `within_report` are box algebra — while its
    own verifiers are exact: `NoCrossInstancePenetration` adjudicates every SAT
    candidate against real geometry. Without the two functions that answer
    exactly (`scene_gap`, `scene_room_gap`), the agent would plan against one
    number and be judged on another.
    """
    from dycraft.agent.layers.layer import _scene_probe_overlay
    ir, catalog = _scene_two_instances()
    overlay = _scene_probe_overlay(
        _FakeSceneCtx(catalog), ir)
    for name in ("gap", "room_gap"):
        assert name in overlay, f"{name} missing from the scene probe overlay"
        assert callable(overlay[name])


def test_the_scene_exact_probes_carry_the_adjudicators_asset_root():
    """The binding is the point, not the name.

    A bare registry entry would be called with `asset_base_dir=None`, and the
    adjudicator is downgrade-only — so it would answer from the bounding box
    with no error and no signal beyond `exact=False`. That is the failure this
    probe exists to remove, arriving through the wiring instead.
    """
    from dycraft.agent.layers.layer import _scene_probe_overlay
    from dycraft.sdk import _probes as P

    ir, catalog = _scene_two_instances()
    ctx = _FakeSceneCtx(catalog)
    overlay = _scene_probe_overlay(ctx, ir)

    seen = {}

    def spy(_ir, _catalog, *args, **kwargs):
        seen.update(kwargs)
        return (0.0, False)

    real = P.scene_gap
    try:
        P.scene_gap = spy
        overlay2 = _scene_probe_overlay(ctx, ir)
        overlay2["gap"](ir, catalog, "a", "b")
    finally:
        P.scene_gap = real
    assert seen["asset_base_dir"] == ctx.asset_base_dir
    assert seen["cache"] is ctx.adjudication_cache
    # An explicit argument still wins, so a caller can scope its own cache.
    assert overlay is not None


def test_the_two_exact_probes_are_not_in_the_scene_registry():
    """One mechanism per question.

    `SCENE_PROBES`'s difference from `ANIMATION_SCENE_PROBES` is pinned
    name-for-name by `test_agent_animation_prompt_drift`, and the animation
    layer offers these two through its own hand binding rather than through a
    registry. Adding them here as well would put the same name in two
    mechanisms and break that pin — so the tuple below is the documented list
    of what `_scene_probe_overlay` binds by hand.
    """
    from dycraft.agent.layers.layer import (
        _SCENE_EXACT_PROBES, _scene_probe_overlay)
    from dycraft.sdk import _probes as P

    assert not (set(_SCENE_EXACT_PROBES) & set(P.SCENE_PROBES))
    ir, catalog = _scene_two_instances()
    overlay = _scene_probe_overlay(
        _FakeSceneCtx(catalog), ir)
    for name in _SCENE_EXACT_PROBES:
        assert name in overlay, (
            f"{name} is listed in _SCENE_EXACT_PROBES but not bound")


def test_scene_pair_report_uses_obbs_and_agrees_with_the_sweep():
    """Two rods turned 45 degrees, offset perpendicular to their own long axes.

    `pair_report` measures oriented boxes for instance pairs, as
    `penetration_pairs` does; an axis-aligned projection would report an
    overlap for a pair the sweep clears. `scene_sdk.md` points here to
    re-inspect a `NoCrossInstancePenetration` witness, so the two must agree.

    The expected separation is independent of the SAT code: the offset
    (0.20, -0.20) is perpendicular to both rods' long axis, so the gap is
    |(0.2, -0.2)| - 0.05 - 0.05 = 0.1828 m.
    """
    from dycraft.sdk import scene_query
    from dycraft.sdk.scene_builder import ObjectHandle

    obj = Object("rod")
    obj.declare_root(obj.make_box("body", 1.00, 0.10, 0.10))
    rod_ir = obj.finalize()
    catalog = {rod_ir["id"]: rod_ir}
    h = ObjectHandle(rod_ir)

    scene = Scene("skew")
    scene.set_room(width=8.0, depth=8.0, height=8.0)
    scene.place(h, name="a", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 45.0)))
    scene.place(h, name="b", pose=((0.20, -0.20, 0.0), (0.0, 0.0, 45.0)))
    ir = scene.finalize()

    assert scene_query.penetration_pairs(ir, catalog) == [], (
        "fixture sanity: the OBB sweep must clear this pair")
    r = _probes.scene_pair_report(ir, catalog, "a", "b")
    assert r["measure"] == "obb"
    assert r["collided"] is False
    assert r["signed_distance"] == pytest.approx(
        (0.2 ** 2 + 0.2 ** 2) ** 0.5 - 0.1, abs=1e-6)
    # The loose boxes do still overlap, which is why the measure must be OBB;
    # `signed_overlap` stays as box algebra.
    assert r["signed_overlap"]["x"] > 0.0
    assert r["signed_overlap"]["y"] > 0.0
    # And the ranked probes agree, since they share the descent.
    entry = _probes.scene_nearest_neighbors(ir, catalog, "a", limit=1)[0]
    assert entry["collided"] is False


def test_scene_pair_report_says_when_only_a_box_could_answer():
    """A group and the room have no single orientation, so they get the AABB —
    and `measure` has to say so, or the caller cannot tell an oriented answer
    from an axis-aligned one."""
    ir, catalog = _scene_two_instances(b_offset=(2.0, 0.0, 0.0))
    assert _probes.scene_pair_report(ir, catalog, "a", "b")["measure"] == "obb"
    assert _probes.scene_pair_report(
        ir, catalog, "a", "room")["measure"] == "aabb"


def test_scene_pair_report_names_the_part_pair_that_carried_the_verdict():
    """An instance is not one body, so a verdict about it comes from some part
    pair; without the names the caller cannot go and measure the right thing."""
    ir, catalog = _scene_two_instances(b_offset=(0.5, 0.0, 0.0))
    r = _probes.scene_pair_report(ir, catalog, "a", "b")
    assert r["collided"] is True
    assert (r["a_sub"], r["b_sub"]) == ("body", "body")


# ──────────────────────────────────────────────────────────────────
# The object layer's exact-collision probe
# ──────────────────────────────────────────────────────────────────


class _FakeObjectCtx:
    def __init__(self, run_dir="/staged/run"):
        self.run_dir = run_dir
        self.adjudication_cache = {}
        self.layer_payload = None


def test_the_object_layer_can_ask_an_exact_collision_question():
    """`gap` must be reachable from an object probe expression.

    Every other collision probe on this layer is a bounding volume —
    `penetration_pairs` and `pair_report` are OBB, `within_report` and
    `overlap_along` are AABB — while `NoSelfPenetration` judges the same
    geometry with B-rep booleans.
    """
    from dycraft.agent.layers.layer import (
        _OBJECT_EXACT_PROBES, object_layer_config)
    overlay = object_layer_config().probe_overlay_factory(
        _FakeObjectCtx(), _two_aligned_boxes())
    for name in _OBJECT_EXACT_PROBES:
        assert name in overlay, f"{name} missing from the object probe overlay"
        assert callable(overlay[name])


def test_the_object_exact_probe_is_bound_to_the_run_dir_not_an_asset_root():
    """Where a `.brep` lives differs by layer, and getting it wrong is silent.

    On the scene and animation layers the catalog is staged elsewhere and named
    by `asset_base_dir`; on the object layer the agent authors the geometry and
    it is staged under the run dir — which is what
    `object_verify.no_self_penetration` passes as `run_dir`. Bound to the wrong
    one, `geometry_spec` resolves every brep path to nothing and the probe
    answers from the bounding box with only `exact=False` to show it.
    """
    from dycraft.agent.layers.layer import object_layer_config
    from dycraft.sdk import _probes as P

    ctx = _FakeObjectCtx()
    seen = {}

    real = P.object_gap
    try:
        P.object_gap = lambda *a, **k: seen.update(k) or (0.0, True)
        overlay = object_layer_config().probe_overlay_factory(
            ctx, _two_aligned_boxes())
        overlay["gap"]("ir", "a", "b")
    finally:
        P.object_gap = real
    assert seen["run_dir"] == ctx.run_dir
    assert seen["cache"] is ctx.adjudication_cache


def test_object_gap_returns_the_box_bound_when_no_query_is_worth_making():
    """`tol` is an ambiguity band, not a threshold.

    A pair the boxes already separate by more than `tol` has surfaces at least
    that far apart, so it is dropped without a CAD query — and the returned
    number is then the box's lower bound, which is why `exact` travels with it.
    Two 1 m cubes 1 m apart, asked with a 5 mm band: no query, and the bound.
    """
    ir = _two_aligned_boxes()          # unit cubes, centers 2 m apart in x
    gap, exact = _probes.object_gap(ir, "a", "b", tol=0.005)
    assert exact is False
    assert gap == pytest.approx(1.0, abs=1e-9)


def _swinging_arm_ir():
    """A hinged arm that swings *away* from a fixed post.

    body: 0.2 cube at the origin (root).
    arm:  1.0 x 0.1 x 0.1, revolute about Z at the body center, extending +X.
    post: 0.2 cube welded at x = 1.3, so it spans 1.2 .. 1.4.

    At 0 degrees the arm reaches x = 1.0 and the box gap to the post is 0.2 m.
    At 90 degrees it points +Y and the gap is 1.15 m. `_hinge_ir` cannot serve
    here: its arm is flush with its body at every joint value, so the gap is
    0.0 throughout and the pose has nothing to move.
    """
    obj = Object("swing")
    body = obj.make_box("body", 0.2, 0.2, 0.2)
    arm = obj.make_box("arm", 1.0, 0.1, 0.1)
    post = obj.make_box("post", 0.2, 0.2, 0.2)
    obj.attach_revolute(
        name="h", parent=body, child=arm,
        axis=(0, 0, 1), limits=(0.0, 180.0), default=0.0,
        parent_anchor=obj.anchor(body, "center"),
        child_anchor=obj.anchor(arm, "right"))
    obj.attach_fixed(
        name="w", parent=body, child=post,
        parent_anchor=obj.anchor(body, "center", translation=(1.3, 0.0, 0.0)),
        child_anchor=obj.anchor(post, "center"))
    obj.declare_root(body)
    return obj.finalize()


def test_object_gap_reads_poses_in_surface_units():
    """Every pose-taking probe on this surface takes degrees for a revolute
    joint. A raw radian value forwarded here would be 57x out and answered
    without complaint, so the conversion has to happen at this boundary.

    `tol` is left at its default, so both calls stay on the bounding-box path
    and this isolates the unit conversion from anything the CAD kernel does.
    The two expected bounds are geometry, not restatements of the code: the arm
    reaches x = 1.0 at 0 degrees against a post starting at x = 1.2, and points
    +Y at 90 degrees leaving its 0.05 half-width in x.
    """
    ir = _swinging_arm_ir()
    rest, exact_rest = _probes.object_gap(ir, "arm", "post", poses={"h": 0.0})
    swung, exact_swung = _probes.object_gap(ir, "arm", "post",
                                            poses={"h": 90.0})
    assert (exact_rest, exact_swung) == (False, False), (
        "both bounds exceed the default tol, so neither should reach CAD")
    assert rest == pytest.approx(0.2, abs=1e-9)
    assert swung == pytest.approx(1.15, abs=1e-9)
    # And 90 read as *radians* is a different configuration entirely, so the
    # degrees answer must not coincide with it.
    as_radians, _e = _probes.object_gap(
        ir, "arm", "post", poses={"h": 90.0 * 180.0 / math.pi})
    assert swung != pytest.approx(as_radians, abs=1e-6)
