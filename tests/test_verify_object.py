"""Tests for the object-layer verifier suite.

Each test targets non-obvious runtime behavior: the AABB-overlap math under
articulated pose composition, the closed-interval boundary at the limit of
non-overlap, the cache-by-(name, ir_hash) memoization, and the unit a
`JointRangeNonTrivial` violation is reported in.

Coverage here is `NoSelfPenetration`, `JointRangeNonTrivial`, and the scope of
`JointOriginNearGeometry`; the rest of `OBJECT_VERIFIERS` is exercised
elsewhere or not at all.

Run: PYTHONPATH=... pytest dycraft/tests/test_verify_object.py -v
"""

import pytest

from dycraft.sdk import Object
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.verify import OBJECT_VERIFIERS, run_suite
from dycraft.verify.object_verify import (
    NO_SELF_PENETRATION,
    no_self_penetration,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


def _two_boxes_with_offset(offset_y: float):
    """Build a 2-part object where the door rests at `offset_y` along Y from
    the body's center. Body and door are 0.4 m long along Y, so the
    parts touch when offset_y == 0.40 and overlap below that."""
    obj = Object("obj")
    body = obj.make_box("body", 0.50, 0.40, 0.30)
    door = obj.make_box("door", 0.50, 0.40, 0.30)
    obj.attach_fixed(
        "joint", body, door,
        parent_anchor=obj.anchor(body, "center", translation=(0.0, offset_y, 0.0)), child_anchor=obj.anchor(door, "center"),
    )
    obj.declare_root(body)
    return obj.finalize()


def test_passes_when_parts_are_just_disjoint():
    """At offset 0.40 + ε the parts barely clear. The boundary case catches
    off-by-one in the AABB overlap test (closed vs open interval)."""
    ir = _two_boxes_with_offset(offset_y=0.40 + 1e-3)
    report = no_self_penetration(ir)
    assert report.verdict == "pass"
    assert report.evidence["pairs"] == []
    # NoSelfPenetration is now continuous over joint poses; with one
    # fixed joint and no movable joints, only the rest pose is sampled.
    assert report.evidence["samples_checked"] >= 1
    assert report.ir_hash == ir["id"]


def test_fails_with_min_axis_depth_under_articulated_pose():
    """50% overlap in Y. X and Z are identical so the per-axis overlap
    minimum (= depth) is the Y dimension. Catches: depth formula picks
    max overlap instead of min, OR forgets to compose joint anchor into
    the child's AABB."""
    ir = _two_boxes_with_offset(offset_y=0.20)
    report = no_self_penetration(ir)
    assert report.verdict == "fail"
    pair = report.evidence["pairs"][0]
    assert {pair["a"], pair["b"]} == {"body", "door"}
    # The depth formula decides the pair is reported, but the number is a box
    # extent and does not ship as evidence. It is asserted on the probe, which
    # is where a caller asks for it by name.
    from dycraft.sdk import _probes
    probed = _probes.object_pair_report(ir, "body", "door")
    assert probed["signed_distance"] == pytest.approx(-0.20, abs=1e-9)
    assert "depth" not in pair and "contact_normal" not in pair
    assert any("apart" in r or "Reduce" in r or "anchor" in r
               for r in report.suggested_repairs)


def test_run_suite_cache_short_circuits_repeated_runs():
    """run_suite caches by (verifier_name, ir_hash). Mutate the stored
    report; the next run_suite on the same IR must return that mutated
    object, proving the cache hit (not a fresh recomputation)."""
    ir = _two_boxes_with_offset(offset_y=0.20)
    cache: dict = {}
    first = run_suite(OBJECT_VERIFIERS, ir, cache=cache)
    cache[(NO_SELF_PENETRATION, ir["id"])].evidence["sentinel"] = True
    second = run_suite(OBJECT_VERIFIERS, ir, cache=cache)
    assert second[0] is first[0]
    assert second[0].evidence.get("sentinel") is True


def _hinge_with_span(span_deg: float):
    """A body plus a door on a revolute joint of exactly `span_deg`."""
    _reset_object_singleton()   # one Object per exec; this helper is called twice
    obj = Object("obj")
    body = obj.make_box("body", 0.50, 0.40, 0.30)
    door = obj.make_box("door", 0.02, 0.40, 0.30)
    obj.declare_root(body)
    obj.attach_revolute(
        "hinge", body, door, axis=(0.0, 0.0, 1.0),
        limits=(0.0, span_deg), default=0.0,
        parent_anchor=obj.anchor(body, "top_front_left"),
        child_anchor=obj.anchor(door, "top_front_left"),
    )
    return obj.finalize()


def test_joint_range_evidence_is_in_the_unit_the_author_wrote():
    """A dead revolute joint reports `span_deg`, not `span_rad`.

    The author declared `limits` in degrees, so a radian span would be the one
    number in the report an agent has to convert before it can act. The
    verdict is decided on travel, but the *angle* still has to appear in
    degrees beside it — that is the field the author recognizes as the thing
    they wrote, and the repair advice tells them to widen it.

    0.1° on this 0.50 m door sweeps 0.87 mm, just under the floor. Note how
    close that is: 0.5° on the same door is 4.4 mm of real motion and passes,
    which a fixed 1° threshold would call dead.
    """
    from dycraft.verify.object_verify import (
        JOINT_RANGE_NON_TRIVIAL, joint_range_non_trivial,
    )

    report = joint_range_non_trivial(_hinge_with_span(0.1))
    assert report.verdict == "fail"
    row = report.evidence["violations"][0]
    assert row["joint"] == "hinge" and row["type"] == "revolute"
    assert row["span_deg"] == pytest.approx(0.1, abs=1e-9)
    assert row["travel_m"] == pytest.approx(0.000873, abs=2e-6)
    assert "span_rad" not in row and "min_required_rad" not in row
    # The repair text names the same fields the evidence carries.
    assert any("deg" in r.lower() for r in report.suggested_repairs)

    assert joint_range_non_trivial(_hinge_with_span(0.5)).verdict == "pass"
    assert joint_range_non_trivial(_hinge_with_span(90.0)).verdict == "pass"


# ---------------------------------------------------------------------------
# Scale-relative thresholds
#
# Both verifiers scale their threshold to the object, and each test below is a
# case a fixed length would get wrong in one direction or the other.
# ---------------------------------------------------------------------------

def _sheet_object(thickness_m: float, plate_m: float = 0.17):
    """A plate plus a thin sheet lying on it, both authored as boxes."""
    _reset_object_singleton()
    obj = Object("sheet")
    base = obj.make_box("base", plate_m, plate_m, 0.01)
    skin = obj.make_box("skin", plate_m, plate_m, thickness_m)
    obj.attach_fixed(
        "bond", base, skin,
        parent_anchor=obj.anchor(base, "center",
                                 translation=(0.0, 0.0, 0.005 + thickness_m)),
        child_anchor=obj.anchor(skin, "center"))
    obj.declare_root(base)
    return obj.finalize()


def test_a_genuinely_thin_part_passes_when_it_is_thin_for_its_object():
    """0.4 mm of gasket on a 170 mm object is a part, not a mistake.

    The fixed 1 mm floor failed this: a 100x100x0.4 mm sheet builds cleanly,
    has valid mesh QC, and was rejected anyway.
    """
    from dycraft.verify.object_verify import min_part_dimension

    rep = min_part_dimension(_sheet_object(0.0004))
    assert rep.verdict == "pass", rep.evidence
    assert rep.evidence["scale"]["min_allowed_m"] == pytest.approx(1.0e-4)


def test_a_slipped_dimension_on_a_large_object_still_fails():
    """The other direction, which a fixed floor cannot see at all.

    A 10 m object with a 1.2 mm member clears a fixed 1 mm floor easily. Against a ten-thousandth of its own 14.1 m diagonal — 1.41 mm
    — it is caught.
    """
    from dycraft.verify.object_verify import min_part_dimension

    rep = min_part_dimension(_sheet_object(0.0012, plate_m=10.0))
    assert rep.verdict == "fail"
    assert rep.evidence["violations"][0]["part"] == "skin"
    # 1.2 mm clears a fixed 1 mm floor and fails this object's own.
    assert rep.evidence["scale"]["min_allowed_m"] == pytest.approx(1.414e-3,
                                                                   rel=1e-3)


def _hinge_object(arm_len_m: float, span_deg: float):
    _reset_object_singleton()   # one Object per exec; this helper is called twice
    obj = Object("hinge")
    base = obj.make_box("base", 0.10, 0.10, 0.02)
    arm = obj.make_box("arm", arm_len_m, 0.01, 0.01)
    obj.attach_revolute(
        "swing", base, arm, axis=(0.0, 0.0, 1.0),
        limits=(0.0, span_deg), default=0.0,
        parent_anchor=obj.anchor(base, "center"),
        child_anchor=obj.anchor(arm, "center"))
    obj.declare_root(base)
    return obj.finalize()


def test_the_same_angle_passes_on_a_long_arm_and_fails_on_a_short_one():
    """`JointRangeNonTrivial` measures travel, so one angle has two answers.

    This is why the threshold is not in degrees: 0.5° sweeps 1.75 mm at the end
    of a 400 mm arm and 0.063 mm at the end of a 3 mm one, a factor of 28, so a
    fixed degree floor would call both the same thing.
    """
    from dycraft.verify.object_verify import joint_range_non_trivial

    long_arm = joint_range_non_trivial(_hinge_object(0.40, 0.5))
    short_arm = joint_range_non_trivial(_hinge_object(0.003, 0.5))
    assert long_arm.verdict == "pass", long_arm.evidence
    assert short_arm.verdict == "fail", short_arm.evidence
    row = short_arm.evidence["violations"][0]
    assert row["travel_m"] < row["min_required_travel_m"]
    # The evidence has to explain why an angle the author may consider ample
    # was rejected, or the repair advice is unactionable.
    assert "radius_m" in row and "span_deg" in row


def test_a_real_corpus_scale_hinge_is_not_flagged():
    """Guard against over-tightening: a modest 14° joint on a 21 mm arm must
    stay comfortably clear of the floor."""
    from dycraft.verify.object_verify import joint_range_non_trivial

    rep = joint_range_non_trivial(_hinge_object(0.021, 14.0))
    assert rep.verdict == "pass", rep.evidence


# ---------------------------------------------------------------------------
# Sweep resolution
# ---------------------------------------------------------------------------

def test_the_sweep_step_follows_the_thinnest_part():
    """A thin obstacle tightens the step; a chunky object keeps the coarse one.

    The rule is `feature / radius`: an obstacle is stepped over when the swept
    point moves further between samples than the obstacle is thick. A fixed
    coarse step is coarser than the thinnest part subtends for most revolute
    joints, so this is not a hypothetical margin.
    """
    import math

    from dycraft.verify.object_verify import _sweep_steps

    def _obj(post_thickness_m):
        _reset_object_singleton()
        o = Object("swept")
        base = o.make_box("base", 0.30, 0.20, 0.02)
        post = o.make_box("post", post_thickness_m, post_thickness_m, 0.12)
        arm = o.make_box("arm", 0.18, 0.03, 0.02)
        o.declare_root(base)
        o.attach_fixed("mount", base, post,
                       parent_anchor=o.anchor(base, "center"),
                       child_anchor=o.anchor(post, "center"))
        o.attach_revolute("swing", base, arm, axis=(0.0, 0.0, 1.0),
                          limits=(-90.0, 90.0), default=0.0,
                          parent_anchor=o.anchor(base, "center"),
                          child_anchor=o.anchor(arm, "center"))
        return o.finalize()

    thin = _sweep_steps(_obj(0.002))["swing"]
    thick = _sweep_steps(_obj(0.05))["swing"]
    assert math.degrees(thin) < math.degrees(thick), (
        "a thinner obstacle must not be swept at the same step")
    assert math.degrees(thick) <= 15.0 + 1e-9, (
        "the geometry-derived step may only ever refine past the coarse "
        "default, never coarsen past it")


def test_a_pass_reports_the_resolution_it_was_established_at():
    """A discrete sweep can only say "nothing at the poses I looked at", so
    the poses it looked at belong in the evidence.

    Without this an author reads `pass` as proof of clearance.
    """
    ir = _two_boxes_with_offset(offset_y=0.40 + 1e-3)
    rep = no_self_penetration(ir)
    assert rep.verdict == "pass"
    res = rep.evidence["sweep_resolution"]
    assert "max_samples_per_joint" in res
    assert res["adaptive"] in (True, False)


def test_adjudications_are_bounded_per_pair_and_keep_the_deepest():
    """Refining the sweep must not multiply the expensive phase.

    Sampling a pose costs ~0.03 ms; adjudicating one flagged (pair, pose)
    against real B-reps costs ~5 ms, 170x more. Unbounded, a finer sweep would
    pay that 170x tax on every extra sample in a flagged band. The cap keeps a
    spread of the band — the deepest pose plus its ends — so the ordering
    argument `no_self_penetration` documents still holds: a shallower pose can
    be the real penetration when the deepest is a bounding-box artifact.
    """
    from dycraft.sdk._object_kernel import PenetrationPair
    from dycraft.verify.object_verify import (_MAX_ADJUDICATIONS_PER_PAIR,
                                              _bounded_candidates)

    frame = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    bbox = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))

    def _pair(depth, a="body", b="door"):
        return PenetrationPair(
            a=a, b=b, depth=depth, contact_normal=(1.0, 0.0, 0.0),
            contact_point=(0.0, 0.0, 0.0), signed_distance=-depth,
            a_frame=frame, b_frame=frame, a_bbox=bbox, b_bbox=bbox,
            witnessing_pose={"j": depth})

    # 40 candidates for one pair, the deepest buried in the middle.
    depths = [0.01 * i for i in range(40)]
    depths[17] = 5.0
    cands = [(_pair(d), {"j": d}) for d in depths]
    kept, dropped = _bounded_candidates(cands)
    assert len(kept) <= _MAX_ADJUDICATIONS_PER_PAIR
    assert dropped == len(cands) - len(kept)
    assert 5.0 in [p.depth for p, _ in kept], "the deepest pose was dropped"
    assert depths[0] in [p.depth for p, _ in kept], "the band's start was dropped"
    assert depths[-1] in [p.depth for p, _ in kept], "the band's end was dropped"

    # Two pairs each get their own allowance.
    two = ([(_pair(0.01 * i, "a", "b"), {}) for i in range(40)]
           + [(_pair(0.01 * i, "c", "d"), {}) for i in range(40)])
    kept2, _ = _bounded_candidates(two)
    assert len(kept2) <= 2 * _MAX_ADJUDICATIONS_PER_PAIR
    assert {frozenset((p.a, p.b)) for p, _ in kept2} == {
        frozenset(("a", "b")), frozenset(("c", "d"))}


def test_a_thin_obstacle_in_the_arc_is_found_at_the_refined_step():
    """The failure a fixed 15° step would let through.

    A 4 mm post 120 mm out from the pivot subtends 1.9°, so a 15° sweep can
    pass the arm straight through it and report `pass`. This is a typical case,
    not a contrived one.
    """
    _reset_object_singleton()
    o = Object("thin")
    base = o.make_box("base", 0.30, 0.20, 0.02)
    post = o.make_box("post", 0.004, 0.004, 0.10)
    arm = o.make_box("arm", 0.24, 0.02, 0.02)
    o.declare_root(base)
    o.attach_fixed(
        "mount", base, post,
        parent_anchor=o.anchor(base, "center", translation=(0.0, 0.12, 0.06)),
        child_anchor=o.anchor(post, "center"))
    o.attach_revolute(
        "swing", base, arm, axis=(0.0, 0.0, 1.0),
        limits=(-90.0, 90.0), default=-90.0,
        parent_anchor=o.anchor(base, "center", translation=(0.0, 0.0, 0.06)),
        child_anchor=o.anchor(arm, "center", translation=(-0.12, 0.0, 0.0)))
    ir = o.finalize()

    rep = no_self_penetration(ir)
    step_deg = rep.evidence["sweep_resolution"]["finest_revolute_step_deg"]
    assert step_deg < 15.0, (
        f"the sweep ran at {step_deg}deg, which cannot resolve a 4 mm post")


# ---------------------------------------------------------------------------
# A separating translation that was measured, not inferred
# ---------------------------------------------------------------------------

def test_a_reported_shift_really_clears_the_overlap():
    """`depth * contact_normal` was the repair advice and it is a box number.

    This lid overlaps its body by exactly 6 mm, so there is a right answer to
    check against. The shift is bisected on the exact boolean — at the offset
    it reports, the intersection is empty — which is a different claim from
    SAT's depth even when the two happen to agree on an axis-aligned pair.
    """
    _reset_object_singleton()
    obj = Object("clash")
    body = obj.make_box("body", 0.20, 0.14, 0.06)
    lid = obj.make_box("lid", 0.20, 0.14, 0.02)
    obj.declare_root(body)
    obj.attach_fixed(
        "seat", body, lid,
        parent_anchor=obj.anchor(body, "center",
                                 translation=(0.0, 0.0, 0.030 - 0.006)),
        child_anchor=obj.anchor(lid, "center",
                                translation=(0.0, 0.0, -0.010)))
    rep = no_self_penetration(obj.finalize())
    assert rep.verdict == "fail"
    row = rep.evidence["pairs"][0]
    shift = row.get("separating_shift_m")
    if shift is None:
        pytest.skip("no CAD interpreter, so nothing adjudicated this pair")
    assert max(abs(v) for v in shift) == pytest.approx(0.006, abs=2e-4), (
        f"the tested shift {shift} does not match the authored 6 mm overlap")
    # Along Z, which is the thinnest axis of the shared solid.
    assert abs(shift[2]) > abs(shift[0]) and abs(shift[2]) > abs(shift[1])


def test_the_report_carries_no_sat_quantity_and_so_warns_about_none():
    """The advice carries no box quantity to steer anyone wrong.

    "Translate `b` by `depth * contact_normal`" is false for anything rotated or
    concave: for a bar stabbed through a block at 40 degrees, SAT reports tens
    of millimeters of box depth that no single-axis translation extracts.
    `exact_volume_cm3` and `closest_points` are present on every adjudicated row
    and `separating_shift_m` on the rows that have one, so there is no box
    quantity to warn against.
    """
    _reset_object_singleton()
    obj = Object("clash2")
    body = obj.make_box("body", 0.20, 0.14, 0.06)
    lid = obj.make_box("lid", 0.20, 0.14, 0.02)
    obj.declare_root(body)
    obj.attach_fixed(
        "seat", body, lid,
        parent_anchor=obj.anchor(body, "center",
                                 translation=(0.0, 0.0, 0.030 - 0.006)),
        child_anchor=obj.anchor(lid, "center",
                                translation=(0.0, 0.0, -0.010)))
    rep = no_self_penetration(obj.finalize())
    advice = " ".join(rep.suggested_repairs)
    assert "separating_shift_m" in advice
    assert "depth * contact_normal" not in advice, (
        "nothing to warn about: the report no longer carries those fields")
    for row in rep.evidence["pairs"]:
        assert not {"depth", "contact_normal", "contact_point",
                    "signed_distance"} & set(row)


# ---------------------------------------------------------------------------
# Reports that stopped contradicting themselves
# ---------------------------------------------------------------------------

def test_unadjudicated_rows_are_not_claimed_as_confirmed_overlaps():
    """Without the adjudicator, the reply says so first.

    Attempts that run with `available == false` report box-only pairs,
    including "depths" that are box extents, so an agent must not be told
    first and unconditionally that every row "really does overlap".
    """
    from dycraft.verify.object_verify import _adjudication_repairs

    judged = [{"a": "x", "b": "y", "adjudicated": True}]
    ran = " ".join(_adjudication_repairs({"available": True}, judged))
    assert "really does overlap" in ran
    assert "separating_shift_m" in ran, "point at the tested correction"

    for missing in ({"available": False}, None, {}):
        text = " ".join(_adjudication_repairs(missing, judged))
        assert "UNCONFIRMED" in text, missing
        assert "really does overlap" not in text, (
            f"the affirmative claim survived with adjudication={missing!r}")
        assert "Bounding boxes can overlap while the surfaces are clear" in text

    # `available` says only that the adjudicator could be REACHED. Every pair
    # can still come back `not_adjudicable` under it — recorded, when no
    # `run_dir` is set and `geometry_spec` cannot resolve a relative brep
    # path. The rows then carry no `adjudicated` flag and no
    # `exact_volume_cm3`, so the affirmative claim must not be made.
    reachable_but_idle = {"available": True, "considered": 7, "sent": 0,
                          "not_adjudicable": 7, "confirmed": 0}
    unjudged = [{"a": "x", "b": "y"}]
    text = " ".join(_adjudication_repairs(reachable_but_idle, unjudged))
    assert "really does overlap" not in text, (
        "available=true with every pair not_adjudicable still claimed the "
        "rows were confirmed")
    assert "UNCONFIRMED" in text
    # A mix must not be described as if it were all confirmed either.
    mixed = [{"a": "x", "b": "y", "adjudicated": True}, {"a": "p", "b": "q"}]
    assert "really does overlap" not in " ".join(
        _adjudication_repairs({"available": True}, mixed))
    # And no rows at all is not a confirmation.
    assert "really does overlap" not in " ".join(
        _adjudication_repairs({"available": True}, []))


def test_the_witness_names_the_joint_that_left_its_default():
    """`per_joint_samples` moves one joint and holds the rest at their authored
    defaults, so a witness is a single-joint sweep — but nothing said which.

    Many confirmed penetrations exist only off-default, and objects can have
    non-zero rest poses, so the moved joint is not the one that looks
    unusual. Here the witness reads `0.0`, which looks like a rest
    value, while the joint's default is −95°.
    """
    _reset_object_singleton()
    obj = Object("swept")
    base = obj.make_box("base", 0.30, 0.20, 0.02)
    lid = obj.make_box("lid", 0.28, 0.18, 0.01)
    obj.declare_root(base)
    obj.attach_revolute(
        "lid_hinge", base, lid, axis=(0.0, 1.0, 0.0),
        limits=(-95.0, 0.0), default=-95.0,
        parent_anchor=obj.anchor(base, "center",
                                 translation=(-0.15, 0.0, 0.01)),
        child_anchor=obj.anchor(lid, "center", translation=(-0.14, 0.0, 0.0)))
    rep = no_self_penetration(obj.finalize())
    assert rep.verdict == "fail"
    row = rep.evidence["pairs"][0]
    swept = row.get("swept_joint")
    assert swept, "a single-joint witness must name its joint"
    assert swept["joint"] == "lid_hinge"
    assert swept["default"] == pytest.approx(-95.0), (
        "the default has to travel with the value, or the reader cannot tell "
        "which number is the departure")


# ---------------------------------------------------------------------------
# JointOriginNearGeometry — what it is allowed to have an opinion about
# ---------------------------------------------------------------------------

def _two_boxes_sharing_a_distant_frame(kind: str):
    """Two bodies joined through anchors that resolve 1 m from either of them.

    This is the shape ten of sixteen agents produced on first attempt: pick one
    construction frame, give parent and child coincident anchors on it, let the
    composition place the geometry. For a `fixed` joint that is exactly right —
    the relative transform is what places the child, and the frame's own
    location is gauge freedom.
    """
    _reset_object_singleton()
    obj = Object("t")
    a = obj.make_box("a", 0.1, 0.1, 0.1)
    b = obj.make_box("b", 0.1, 0.1, 0.1)
    obj.declare_root(a)
    parent_anchor = obj.anchor(a, "center", translation=(1.0, 1.0, 1.0))
    child_anchor = obj.anchor(b, "center", translation=(1.0, 1.0, 1.0))
    if kind == "fixed":
        obj.attach_fixed("j", a, b, parent_anchor, child_anchor)
    else:
        obj.attach_revolute("j", a, b, axis=(0.0, 0.0, 1.0),
                            limits=(0.0, 90.0), default=0.0,
                            parent_anchor=parent_anchor,
                            child_anchor=child_anchor)
    return obj.finalize()


def test_a_fixed_joints_frame_may_sit_anywhere():
    """A fixed joint has no pivot, so there is nothing for it to be near.

    Flagging a fixed joint's frame would make most of this check's output an
    opinion about a coordinate with no effect on the geometry.
    """
    from dycraft.verify.object_verify import joint_origin_near_geometry

    report = joint_origin_near_geometry(_two_boxes_sharing_a_distant_frame("fixed"))
    assert report.verdict == "not_applicable", report.evidence
    assert report.evidence["fixed_joints_skipped"] == 1, (
        "the report must say what it declined to check, or a clean pass "
        "reads as 'every joint was examined'")


def test_a_movable_joints_pivot_must_still_reach_a_body():
    """The check is not weakened where it means something: a revolute pivot
    the author placed a meter from both bodies is a real authoring blunder."""
    from dycraft.verify.object_verify import joint_origin_near_geometry

    report = joint_origin_near_geometry(
        _two_boxes_sharing_a_distant_frame("revolute"))
    assert report.verdict == "fail"
    assert [v["joint"] for v in report.evidence["violations"]] == ["j"]
    assert report.evidence["fixed_joints_skipped"] == 0


def test_the_two_joint_origin_checks_agree_on_their_scope():
    """`JointOriginNearBothBodies` already skipped fixed joints. Two checks
    measuring the same pivot on different joint sets is how one of them ends
    up reporting on a quantity that does not exist."""
    from dycraft.verify.object_verify import (
        joint_origin_near_both_bodies, joint_origin_near_geometry)

    ir = _two_boxes_sharing_a_distant_frame("fixed")
    assert joint_origin_near_geometry(ir).verdict == "not_applicable"
    assert joint_origin_near_both_bodies(ir).verdict == "not_applicable"
