"""Tests for three scene-layer verifiers: NoCrossInstancePenetration,
RoomContainment, SceneRelationalAssertions. (SupportConnected and
SceneVlmJudge are covered by test_verify_support_connected.py and the
VLM-judge tests respectively; the four are `INSTANCE_VERIFIERS` in
verify/scene_verify.py.)

Each test targets non-obvious runtime behavior:
  * cross-instance AABB depth math under scene-rest-pose composition;
  * the partial-axis violation surface from RoomContainment (a single
    axis pokes out of the room; others stay inside);
  * the inf round-trip from canonical-JSON-encoded assertion bounds
    back to math.inf for arithmetic comparison;
  * the collinear predicate's max-perp formula across N > 2 anchors.

Run: PYTHONPATH=... pytest dycraft/tests/test_verify_scene.py -v
"""

import math
from dataclasses import dataclass
from typing import Mapping

import pytest

from dycraft.sdk import Object, ObjectHandle, Scene
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton
from dycraft.verify import INSTANCE_VERIFIERS, run_suite
from dycraft.verify.scene_verify import (
    no_cross_instance_penetration,
    room_containment,
    scene_relational_assertions,
    support_connected,
)
from dycraft.verify._relational import _inf_or_float


@pytest.fixture(autouse=True)
def _reset_singletons():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


@dataclass
class _Ctx:
    """Minimal stand-in for the SceneAgentContext.layer_payload surface
    the scene verifiers read."""
    catalog: Mapping[str, dict]
    layer_payload: object = None


def _box_ir(obj_name="box", w=1.0, l=1.0, h=1.0, part_name="body"):
    _reset_object_singleton()
    o = Object(obj_name)
    p = o.make_box(part_name, w, l, h)
    o.declare_root(p)
    return o.finalize()


def _ctx_of(*irs):
    return _Ctx(catalog={ir["id"]: ir for ir in irs})


# ---------------------------------------------------------------------------
# NoCrossInstancePenetration: AABB depth under scene pose composition.
# ---------------------------------------------------------------------------

def test_cross_instance_penetration_depth_equals_min_axis_overlap():
    """Two 1x1x1 boxes; b is placed 0.7 m along +X from a's center. X-overlap
    = 0.3, Y/Z overlap = 1.0. min-axis depth must be 0.3 — not 1.0 or the
    mean of the three. Catches: depth formula picks the wrong axis.

    Reported as `penetration_m`, which is `reported_depth`: the smaller of
    this box number and the depth the mesh narrow phase measures. On
    axis-aligned boxes with no exact pass the two agree, which is what makes
    this a depth-formula test and not a backend test."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((0.7, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = no_cross_instance_penetration(ir, _ctx_of(box_ir))
    assert report.verdict == "fail"
    [pair] = report.evidence["pairs"]
    assert {pair["a"], pair["b"]} == {"a", "b"}
    assert pair["penetration_m"] == pytest.approx(0.3, abs=1e-9)
    assert not {"depth", "contact_normal", "contact_point",
                "signed_distance"} & set(pair)


def test_cross_instance_penetration_excludes_grazing_contact():
    """Two 1x1x1 boxes placed exactly edge-to-edge (offset 1.0) must pass.
    Catches: open vs closed interval at the touch boundary."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((1.0 + 1e-3, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = no_cross_instance_penetration(ir, _ctx_of(box_ir))
    assert report.verdict == "pass"
    assert report.evidence["pairs"] == []
    assert report.evidence.get("samples_checked", 0) >= 1


# ---------------------------------------------------------------------------
# NoCrossInstancePenetration: sweep over instance articulation (scene CCD).
# ---------------------------------------------------------------------------


def _slider_ir():
    """Object with a body (1x1x1) and a prismatic arm (1x1x1) along +X.

    At default q=0 the arm's -X face is flush with the body's +X face
    (object-local). At q=2.0 the arm center moves to x=3.0 (object-local).
    """
    _reset_object_singleton()
    o = Object("slider")
    body = o.make_box("body", 1.0, 1.0, 1.0)
    arm = o.make_box("arm", 1.0, 1.0, 1.0)
    o.attach_prismatic(
        "slide", body, arm,
        axis=(1.0, 0.0, 0.0),
        limits=(0.0, 2.0),
        default=0.0,
        parent_anchor=o.anchor(body, "left"),
        child_anchor=o.anchor(arm, "right"),
    )
    o.declare_root(body)
    return o.finalize()


def test_scene_ccd_penetration_pairs_pose_aware():
    """`scene_query.penetration_pairs(ir, catalog, poses=...)` must
    expand each instance's OBB by the articulation. Two slider instances
    placed so they are disjoint at rest pose collide when the
    qualified pose `"a.slide" = 2.0` extends the arm.
    """
    from dycraft.sdk import scene_query

    slider_ir = _slider_ir()
    handle = ObjectHandle(slider_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(8.0, 4.0, 2.0)
    # Instance a at origin; b placed so its world x in [3.0, 5.0].
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((3.5, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    catalog = {slider_ir["id"]: slider_ir}

    # Rest pose: a's world x in [-0.5, 1.5], gap to b is 1.5 — no pair.
    assert scene_query.penetration_pairs(ir, catalog) == []

    # Extend a's slider: a's world x in [-0.5, 3.5], b in [3.0, 5.0] — overlap 0.5.
    pairs = scene_query.penetration_pairs(ir, catalog, poses={"a.slide": 2.0})
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"a", "b"}
    assert pairs[0].depth == pytest.approx(0.5, abs=1e-9)

    # Qualified pose key for a different instance must not move a.
    pairs2 = scene_query.penetration_pairs(
        ir, catalog, poses={"b.slide": 2.0})
    # b at q=2 has local x in [-0.5, 3.5] -> world x in [3.0, 7.0]; a is
    # at [-0.5, 1.5]; gap = 1.5. No penetration.
    assert pairs2 == []


def test_scene_raw_part_pairs_preserve_candidates_before_aggregation():
    """The raw collision API retains multiple colliding part pairs while the
    instance-pair API keeps its one-row-per-instance-pair contract."""
    from dycraft.sdk import scene_query

    slider_ir = _slider_ir()
    handle = ObjectHandle(slider_ir)
    _reset_scene_singleton()
    s = Scene("coincident")
    s.set_room(4.0, 4.0, 3.0)
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    catalog = {slider_ir["id"]: slider_ir}

    raw = scene_query.penetration_part_pairs(ir, catalog)
    aggregated = scene_query.penetration_pairs(ir, catalog)
    assert len(raw) >= 2
    assert len(aggregated) == 1
    assert {(p.a_sub, p.b_sub) for p in raw} >= {
        ("body", "body"), ("arm", "arm"),
    }
    assert aggregated[0].signed_distance == min(
        p.signed_distance for p in raw)


def test_no_cross_instance_penetration_detects_swept_collision():
    """NoCrossInstancePenetration must enumerate joint samples and report
    the failure under the articulated pose."""
    slider_ir = _slider_ir()
    handle = ObjectHandle(slider_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(8.0, 4.0, 2.0)
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((3.5, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    # `static_only=False` explicitly: the sweep is not the default (see
    # `_ccd_sampler.scene_static_only_default`), and this test is about the
    # sweep, so it must ask for it rather than inherit it.
    report = no_cross_instance_penetration(ir, _ctx_of(slider_ir),
                                           static_only=False)
    assert report.verdict == "fail"
    # Under worst-case-per-pair reporting, the failing-pose info lives on
    # each pair's witnessing_pose field, not at the top level.
    [pair] = report.evidence["pairs"]
    assert {pair["a"], pair["b"]} == {"a", "b"}
    failing_pose = pair["witnessing_pose"]
    # The failure must reference the articulated joint via qualified key.
    assert any(k.endswith(".slide") for k in failing_pose)
    # And the slide value that triggered the failure must be > 0 (i.e.
    # not the rest pose, which would be q=0).
    triggering_q = next(
        v for k, v in failing_pose.items() if k.endswith(".slide"))
    assert triggering_q > 0.0


def test_static_only_is_the_default_and_checks_the_rest_pose_only():
    """The scene layer does not sweep joints by default.

    The premise of a sweep is that every value in a joint's range is a pose the
    object is *meant* to hold. True of a door; false of a manipulator, whose
    limits are a mechanical envelope. Sweeping would make the check
    unsatisfiable for an arm bolted to a bench (its elbow reaches below its own
    mounting plane). What a sweep reaches for is answered one layer down, by the
    animation layer's per-frame `NoPenetration` over the trajectory that was
    actually authored.
    """
    slider_ir = _slider_ir()
    handle = ObjectHandle(slider_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(8.0, 4.0, 2.0)
    # Clear at rest (slide default = 0), colliding once the slide extends.
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((3.5, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    rep = no_cross_instance_penetration(ir, _ctx_of(slider_ir))
    assert rep.verdict == "pass", rep.evidence
    assert rep.evidence["sampling"] == "rest_only"
    assert rep.evidence["samples_checked"] == 1

    # The capability is not gone, only off by default.
    swept = no_cross_instance_penetration(ir, _ctx_of(slider_ir),
                                          static_only=False)
    assert swept.verdict == "fail"
    assert swept.evidence["sampling"] == "per_joint"


def test_static_only_honours_its_env_var(monkeypatch):
    """`DYCRAFT_SCENE_STATIC_ONLY=0` restores the sweep without a code change —
    and it is a *separate* switch from the object layer's
    `DYCRAFT_CCD_PER_JOINT_ONLY`, which must keep sweeping self-collision."""
    from dycraft.verify import _ccd_sampler as cs

    monkeypatch.delenv("DYCRAFT_SCENE_STATIC_ONLY", raising=False)
    assert cs.scene_static_only_default() is True
    monkeypatch.setenv("DYCRAFT_SCENE_STATIC_ONLY", "0")
    assert cs.scene_static_only_default() is False
    # Untouched by the scene switch.
    monkeypatch.delenv("DYCRAFT_CCD_PER_JOINT_ONLY", raising=False)
    assert cs.per_joint_only_default() is True


def test_rest_pose_holds_every_joint_at_its_default():
    from dycraft.verify._ccd_sampler import rest_pose

    joints = [{"name": "a.hinge", "type": "revolute", "default": 0.5,
               "limits": [0.0, 1.5]},
              {"name": "a.fixed", "type": "fixed", "default": 0.0,
               "limits": [0.0, 0.0]},
              {"name": "b.slide", "type": "prismatic", "default": 0.02,
               "limits": [0.0, 0.2]}]
    assert rest_pose(joints) == [{"a.hinge": 0.5, "b.slide": 0.02}]
    assert rest_pose([]) == [{}]


def test_no_cross_instance_penetration_pass_reports_samples_checked():
    """Even when there is no articulation conflict, the pass report should
    surface `samples_checked >= 1` so callers can confirm the verifier
    actually swept the joint range."""
    slider_ir = _slider_ir()
    handle = ObjectHandle(slider_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(12.0, 4.0, 2.0)
    # Instances placed far enough apart that the slider's full range
    # never crosses (a's max x = 3.5, b's min x = 5.5).
    s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.place(handle, name="b", pose=((6.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = no_cross_instance_penetration(ir, _ctx_of(slider_ir),
                                           static_only=False)
    assert report.verdict == "pass"
    assert report.evidence["pairs"] == []
    assert report.evidence["samples_checked"] >= 2  # sweep + random samples
    assert report.evidence["sampling"] == "per_joint"


# ---------------------------------------------------------------------------
# Per-part SAT in the scene layer: concave / L-shaped instance accuracy.
# ---------------------------------------------------------------------------


def _l_shape_ir():
    """L-shape: a horizontal arm and a vertical arm fixed-attached at one
    end. The L's union AABB has a concave corner (no geometry on the
    z>0.1, x<0.4 quadrant) that a single OBB per instance would wrongly
    flag as occupied."""
    _reset_object_singleton()
    o = Object("L")
    arm_x = o.make_box("arm_x", 1.0, 0.2, 0.2)
    arm_z = o.make_box("arm_z", 0.2, 0.2, 1.0)
    o.attach_fixed(
        "corner", arm_x, arm_z,
        parent_anchor=o.anchor(arm_x, "left"),
        child_anchor=o.anchor(arm_z, "bottom"),
    )
    o.declare_root(arm_x)
    return o.finalize()


def _small_cube_ir():
    _reset_object_singleton()
    o = Object("cube")
    p = o.make_box("body", 0.1, 0.1, 0.1)
    o.declare_root(p)
    return o.finalize()


def test_per_part_sat_no_false_positive_in_l_shape_concave_corner():
    """A small cube sitting in the empty quadrant of an L-shaped
    instance: the cube's AABB is inside the L's loose union AABB but
    overlaps neither arm. Single-OBB code would have flagged the pair;
    per-part SAT must return []."""
    from dycraft.sdk import scene_query

    l_ir = _l_shape_ir()
    cube_ir = _small_cube_ir()
    l_handle = ObjectHandle(l_ir)
    cube_handle = ObjectHandle(cube_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    s.place(l_handle, name="l", pose=((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)))
    # Cube centered at world (0, 0, 0.6): inside L's loose AABB
    # (x in [-0.5, 0.6], z in [0, 1.1]) but x in [-0.05, 0.05] misses
    # arm_z's x in [0.4, 0.6], and z in [0.55, 0.65] misses arm_x's
    # z in [0, 0.2].
    s.place(cube_handle, name="c", pose=((0.0, 0.0, 0.6), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    catalog = {l_ir["id"]: l_ir, cube_ir["id"]: cube_ir}

    assert scene_query.penetration_pairs(ir, catalog) == []


def test_per_part_sat_detects_overlap_with_l_shape_arm():
    """Same L instance + a cube placed inside arm_z's volume: per-part
    SAT must still report the overlap (no false negative)."""
    from dycraft.sdk import scene_query

    l_ir = _l_shape_ir()
    cube_ir = _small_cube_ir()
    l_handle = ObjectHandle(l_ir)
    cube_handle = ObjectHandle(cube_ir)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    s.place(l_handle, name="l", pose=((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)))
    # Cube centered at world (0.5, 0, 0.6): inside arm_z's
    # x in [0.4, 0.6], z in [0.1, 1.1] volume.
    s.place(cube_handle, name="c", pose=((0.5, 0.0, 0.6), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    catalog = {l_ir["id"]: l_ir, cube_ir["id"]: cube_ir}

    pairs = scene_query.penetration_pairs(ir, catalog)
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"l", "c"}
    assert pairs[0].depth > 0.0


# ---------------------------------------------------------------------------
# RoomContainment: per-axis violation reporting.
# ---------------------------------------------------------------------------

def test_room_containment_reports_which_axis_is_out_of_range():
    """A 1x1x1 box placed so its world AABB exceeds the room only in +X.
    The report must surface that exactly one axis (x_above_max) is
    violated and quantify the overshoot. A regression that flags the
    instance without breaking down which axis broke would still pass a
    weaker test — this one pins it down."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    # Room is 2 x 2 x 2 (so world bounds are x,y ∈ [-1, 1], z ∈ [0, 2]).
    s = Scene("s")
    s.set_room(2.0, 2.0, 2.0)
    # Box centered at x=0.8, z=0.5: extends to x=1.3 (0.3 past x=1.0).
    s.place(handle, name="a", pose=((0.8, 0.0, 0.5), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = room_containment(ir, _ctx_of(box_ir))
    assert report.verdict == "fail"
    out = report.evidence["out_of_room"]
    [entry] = out
    assert entry["instance"] == "a"
    violations = entry["violations"]
    # Exactly one axis is violated, and the overshoot magnitude is exact.
    assert set(violations.keys()) == {"x_above_max"}
    assert violations["x_above_max"] == pytest.approx(0.3, abs=1e-9)


def test_room_containment_passes_when_box_floats_above_floor_inside_room():
    """A flat 1x1x0.1 box centered at z=1.0 inside a 2x2x2 room: all six
    AABB faces are strictly inside the room AABB. Pure passing path —
    catches a regression that conflates "rests on floor" with
    "contained" (NoFloating is deliberately out of scope here)."""
    box_ir = _box_ir("box", 1.0, 1.0, 0.1)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(2.0, 2.0, 2.0)
    s.place(handle, name="a", pose=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = room_containment(ir, _ctx_of(box_ir))
    assert report.verdict == "pass"
    assert report.evidence["out_of_room"] == []
    # No candidate reached the exact pass, so there is nothing to report about
    # it. `adjudication` is present-and-None rather than absent, so a reader
    # can always tell "the exact pass said nothing" from "the exact pass was
    # never wired up".
    assert report.evidence["adjudication"] is None


@pytest.mark.parametrize("enclosure, verdict", [("open_air", "not_applicable"),
                                                ("open_top", "not_applicable"),
                                                ("interior", "fail")])
def test_an_uncapped_room_contains_nothing_so_containment_does_not_apply(enclosure, verdict):
    """The same 0.3 m excursion as above. An `open_air` room is an extent with
    a floor, and `open_top` has no lid: neither encloses the scene, so there
    is nothing to be outside of, and the report says so rather than passing.
    Only a capped room still fails it."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(2.0, 2.0, 2.0, enclosure=enclosure)
    s.place(ObjectHandle(box_ir), name="a", pose=((0.8, 0.0, 0.5), (0.0, 0.0, 0.0)))
    report = room_containment(s.finalize(), _ctx_of(box_ir))
    assert report.verdict == verdict
    if verdict == "not_applicable":
        assert enclosure in report.evidence["not_applicable"]


# ---------------------------------------------------------------------------
# SceneRelationalAssertions: predicate evaluation + inf round-trip.
# ---------------------------------------------------------------------------

def test_distance_along_passes_with_positive_inf_max_after_round_trip():
    """An assertion with max=+inf serializes as the JSON string "inf" in the
    SceneIR (canonical JSON has allow_nan=False). The verifier must map
    that string back to math.inf for comparison; otherwise lo <= signed <= hi
    raises TypeError on str vs float. This test pins down the round-trip."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    a = s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    b = s.place(handle, name="b", pose=((2.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.assert_distance(
        a=s.anchor(a, "center"), b=s.anchor(b, "center"),
        axis=(1.0, 0.0, 0.0), min=0.0, max=float("inf"),
        name="b_east_of_a",
    )
    ir = s.finalize()

    # Sanity: the bound is stored as the string "inf" in the IR.
    [au] = ir["assertions"]
    assert au["args"]["max"] == "inf"

    report = scene_relational_assertions(ir, _ctx_of(box_ir))
    assert report.verdict == "pass"
    [result] = report.evidence["results"]
    assert result["verdict"] == "pass"
    # The signed-axis value is the center-to-center projection: 2.0 m.
    assert result["value"] == pytest.approx(2.0, abs=1e-9)


def test_distance_along_fails_when_signed_projection_undershoots_min():
    """Same setup but with min=3.0: the actual signed projection is 2.0,
    which is below the required minimum. Catches a regression that
    confuses signed with absolute distance (a |2.0| >= 3.0? would also
    fail, but for the wrong reason: the wrong-sign case below shows the
    difference)."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    a = s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    b = s.place(handle, name="b", pose=((2.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.assert_distance(
        a=s.anchor(a, "center"), b=s.anchor(b, "center"),
        axis=(1.0, 0.0, 0.0), min=3.0, max=float("inf"),
        name="b_far_east_of_a",
    )
    ir = s.finalize()

    report = scene_relational_assertions(ir, _ctx_of(box_ir))
    assert report.verdict == "fail"
    # The failing-assertion hint references the actual name.
    assert any("'b_far_east_of_a'" in r for r in report.suggested_repairs)


def test_distance_along_sign_aware_fails_when_axis_is_flipped():
    """b is at +X from a; an assertion that requires signed-along-(-X) to
    be in [0, inf] must FAIL because the projection along -X is -2.0.
    Catches a regression that takes |dot(b-a, axis)|."""
    box_ir = _box_ir("box", 1.0, 1.0, 1.0)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    a = s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    b = s.place(handle, name="b", pose=((2.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.assert_distance(
        a=s.anchor(a, "center"), b=s.anchor(b, "center"),
        axis=(-1.0, 0.0, 0.0), min=0.0, max=float("inf"),
        name="b_west_of_a",
    )
    ir = s.finalize()

    report = scene_relational_assertions(ir, _ctx_of(box_ir))
    assert report.verdict == "fail"
    [result] = report.evidence["results"]
    assert result["value"] == pytest.approx(-2.0, abs=1e-9)


def test_collinear_max_perp_across_n_anchors_picks_the_worst_offender():
    """Three boxes placed mostly along +X but with one off-axis. The
    collinear predicate's reported `value` must be the MAX perpendicular
    deviation, not the mean — and only the worst offender controls the
    pass/fail boundary. A `tol` between the smallest and largest perp
    deviations exposes the bug: mean-based code would pass; max-based
    code must fail."""
    box_ir = _box_ir("box", 0.1, 0.1, 0.1)
    handle = ObjectHandle(box_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 2.0)
    a = s.place(handle, name="a", pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    # b is 0.10 m off-axis (small), c is 0.30 m off-axis (large).
    b = s.place(handle, name="b", pose=((1.0, 0.10, 0.5), (0.0, 0.0, 0.0)))
    c = s.place(handle, name="c", pose=((2.0, 0.30, 0.5), (0.0, 0.0, 0.0)))
    s.assert_collinear(
        s.anchor(a, "center"), s.anchor(b, "center"), s.anchor(c, "center"),
        axis=(1.0, 0.0, 0.0), tol=0.20, name="abc_x_collinear",
    )
    ir = s.finalize()

    report = scene_relational_assertions(ir, _ctx_of(box_ir))
    assert report.verdict == "fail"
    [result] = report.evidence["results"]
    # Reported value is the max perp deviation (= 0.30), not the mean (0.20).
    assert result["value"] == pytest.approx(0.30, abs=1e-9)
    # Both per-anchor perps surface in the evidence in order.
    assert result["perp_norms"][0] == pytest.approx(0.10, abs=1e-9)
    assert result["perp_norms"][1] == pytest.approx(0.30, abs=1e-9)


# ---------------------------------------------------------------------------
# Cache-key fingerprint: INSTANCE_VERIFIERS cached by (name, ir_hash, extra).
# A regression that drops the extra fingerprint would incorrectly serve a
# cached result for one scene back to another scene that uses a different
# catalog subset. We don't test the cache hit/miss directly here (that's
# covered by the object verify cache test); we DO pin the public
# _inf_or_float helper since it's the visible "string → float" contract.
# ---------------------------------------------------------------------------

def test_inf_or_float_helper_round_trips_canonical_json_inf_strings():
    """The verifier-side inverse of the SDK's _float_for_ir. Catches a
    regression that uses float() alone — float('inf') would work but
    float('-inf') would too, so the test below cross-checks both AND a
    finite number AND a passthrough float."""
    assert _inf_or_float("inf") == math.inf
    assert _inf_or_float("-inf") == -math.inf
    assert _inf_or_float(3.14) == pytest.approx(3.14, abs=1e-12)
    assert _inf_or_float("2.5") == pytest.approx(2.5, abs=1e-12)


# ---------------------------------------------------------------------------
# SupportConnected: part-granular scene graph (recessed/inset fixtures).
# ---------------------------------------------------------------------------


def _cabinet_ir():
    """A solid floor cabinet: one box part, footprint 1.0 x 0.6, top at the
    counter height. Placed with bottom on the floor."""
    _reset_object_singleton()
    o = Object("cabinet")
    body = o.make_box("body", 1.0, 0.6, 0.92)
    o.declare_root(body)
    return o.finalize()


def _drop_in_sink_ir():
    """A drop-in sink: a thin rim flange (root) plus a basin hanging below
    it. The sink's union AABB dips below the rim and is nested inside the
    cabinet's box, so the two whole-object union boxes share no coplanar
    face — only the rim *part* rests on the cabinet's top."""
    _reset_object_singleton()
    o = Object("sink")
    rim = o.make_box("rim", 0.6, 0.4, 0.04)
    basin = o.make_box("basin", 0.5, 0.3, 0.30)
    o.attach_fixed("j_basin", rim, basin,
                   parent_anchor=o.anchor(rim, "bottom"),
                   child_anchor=o.anchor(basin, "top"))
    o.declare_root(rim)
    return o.finalize()


def test_recessed_fixture_grounds_via_part_contact():
    """A drop-in sink whose union AABB is buried inside the cabinet's box
    must ground via the real rim-part-on-counter-part face contact.

    Under the old whole-instance union model the sink had no coplanar face
    with the cabinet (basin dips below the counter top, rim top rises above
    it) and read as floating, which once forced a 'ground via the window'
    hack. With part-granular edges the rim part
    (bottom at z=0.92) is coplanar with the cabinet top part (z=0.92)."""
    cabinet_ir = _cabinet_ir()
    sink_ir = _drop_in_sink_ir()
    cabinet = ObjectHandle(cabinet_ir)
    sink = ObjectHandle(sink_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 3.0)
    # Cabinet box rests on the floor (default bottom anchor at z=0):
    # body z[0,0.92], top at the 0.92 counter height.
    s.place(cabinet, name="cab", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    # The sink's union bbox bottom (basin) is 0.32 below the rim-center origin
    # (0.30 basin + 0.02 half-rim), so a bottom anchor at z=0.62 lands the rim
    # center at 0.94 -> rim z[0.92,0.96], basin z[0.62,0.92] (rim flush on the
    # counter top, basin recessed into the cabinet).
    s.place(sink, name="sink", pose=((0.0, 0.0, 0.62), (0.0, 0.0, 0.0)))
    ir = s.finalize()

    report = support_connected(ir, _ctx_of(cabinet_ir, sink_ir))
    assert report.verdict == "pass", report.evidence
    # The cabinet grounds directly on the floor; the sink grounds
    # transitively through the cabinet, NOT directly.
    assert "cab" in report.evidence["grounded_directly"]
    assert "sink" not in report.evidence["grounded_directly"]
    assert report.evidence["unsupported"] == []


def test_recessed_fixture_with_gap_still_floats():
    """Negative control: lift the sink clear of the cabinet entirely, so it
    touches nothing. Part-granular edges must not blanket-pass.

    **The lift is measured from the basin, not the rim**, and that correction
    is the whole point of this docstring. The original fixture put the bottom
    anchor at z=0.67 on the reasoning that it left the *rim* 0.05 above the
    counter — but the basin hangs 0.30 below the rim, and `_cabinet_ir`'s body
    is a solid box with no cutout, so the basin was 0.25 m **inside** the
    cabinet. The exact gap between them is 0.0: they share material.

    Under the coplanar-AABB-face test that was invisible, and the pair produced
    no edge, so the scene failed for a reason that had nothing to do with the
    geometry. Measuring surfaces reports the interpenetration as contact —
    which it is — so the sink came back supported. The scene was still
    rejected, by `NoCrossInstancePenetration`, whose job that is; the
    assertions below pin both halves.
    """
    cabinet_ir = _cabinet_ir()
    sink_ir = _drop_in_sink_ir()
    cabinet = ObjectHandle(cabinet_ir)
    sink = ObjectHandle(sink_ir)

    def _scene(z):
        _reset_scene_singleton()
        s = Scene("s")
        s.set_room(4.0, 4.0, 3.0)
        s.place(cabinet, name="cab", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        s.place(sink, name="sink", pose=((0.0, 0.0, z), (0.0, 0.0, 0.0)))
        return s.finalize()

    # The bottom anchor lands the basin's underside, so z=0.97 leaves the
    # lowest part of the sink 0.05 above the 0.92 counter top — the gap this
    # test always meant to describe.
    ir = _scene(0.97)
    report = support_connected(ir, _ctx_of(cabinet_ir, sink_ir))
    assert report.verdict == "fail", report.evidence
    assert "sink" in report.evidence["unsupported"]

    # And the old placement, kept as the record of what it actually was: the
    # basin shares material with the cabinet, so contact is real and this
    # verifier says so — while the penetration verifier rejects the scene.
    buried = _scene(0.67)
    assert support_connected(
        buried, _ctx_of(cabinet_ir, sink_ir)).verdict == "pass"
    assert no_cross_instance_penetration(
        buried, _ctx_of(cabinet_ir, sink_ir)).verdict == "fail"


# ---------------------------------------------------------------------------
# Kinematic-follow: an instance attached to a moving parent part rides the
# joint (M-G4 / scene.attach kinematics).
# ---------------------------------------------------------------------------


def _trunnion_ir():
    """A trunnion: a base (root) + a thin plate that tilts up about the
    plate's back (-Y) bottom edge via a revolute joint `tilt`.

    Object-local: base is 1.0x1.0x0.2 centered at origin (z[-0.1,0.1]).
    The plate (0.8x0.8x0.06) rests flush on the base top at rest (q=0,
    z[0.10,0.16]); the pivot line is at y=-0.4, z=0.1, axis +X, so a
    positive q lifts the plate's +Y front toward +Z.
    """
    _reset_object_singleton()
    o = Object("trunnion")
    base = o.make_box("base", 1.0, 1.0, 0.2)
    plate = o.make_box("plate", 0.8, 0.8, 0.06)
    o.attach_revolute(
        "tilt", base, plate,
        axis=(1.0, 0.0, 0.0),
        limits=(0.0, 90.0),
        default=0.0,
        parent_anchor=o.anchor(base, "top", translation=(0.0, -0.4, 0.0)),
        child_anchor=o.anchor(plate, "bottom", translation=(0.0, -0.4, 0.0)),
    )
    o.declare_root(base)
    return o.finalize()


def _billet_ir():
    """A small workpiece block (0.3 cube)."""
    _reset_object_singleton()
    o = Object("billet")
    b = o.make_box("billet", 0.3, 0.3, 0.3)
    o.declare_root(b)
    return o.finalize()


def _trunnion_scene(follow):
    """Trunnion placed on the floor with a billet bolted to the plate top
    near its +Y front (y=+0.3 on the plate). `follow` toggles whether the
    billet rides the tilt joint."""
    trunnion_ir = _trunnion_ir()
    billet_ir = _billet_ir()
    trunnion = ObjectHandle(trunnion_ir)
    billet = ObjectHandle(billet_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 3.0)
    # Base bottom on the floor: base center at z=0.1 -> base z[0,0.2].
    tr = s.place(trunnion, name="trunnion",
                 pose=((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)))
    # Billet bottom on the plate top, near the +Y front of the plate.
    s.attach(
        billet, name="billet", parent=tr,
        parent_anchor=s.anchor(tr, "plate.top", translation=(0.0, 0.3, 0.0)),
        child_anchor=s.anchor(billet, "bottom"),
        follow=follow,
    )
    ir = s.finalize()
    return ir, trunnion_ir, billet_ir


def test_attached_child_follows_parent_joint():
    """Default `follow=True`: a billet bolted to the trunnion plate rides
    the `tilt` joint, so plate and billet move together and never
    interpenetrate across the swept range. This is the scene-layer
    analog of authoring the workpiece inside the machine object, without
    collapsing the machine into one object."""
    from dycraft.sdk import scene_query

    ir, trunnion_ir, billet_ir = _trunnion_scene(follow=True)
    catalog = {trunnion_ir["id"]: trunnion_ir, billet_ir["id"]: billet_ir}

    # Rest: billet rests flush on the plate top (grazing, excluded).
    assert scene_query.penetration_pairs(ir, catalog) == []

    # Sweep the trunnion tilt to its limit: the billet must ride along, so
    # still no cross-instance penetration at any sampled angle.
    import math
    for q in (math.radians(d) for d in (15, 30, 45, 60, 90)):
        pairs = scene_query.penetration_pairs(
            ir, catalog, poses={"trunnion.tilt": q})
        assert pairs == [], f"unexpected penetration at q={q}: {pairs}"

    report = no_cross_instance_penetration(ir, _ctx_of(trunnion_ir, billet_ir))
    assert report.verdict == "pass", report.evidence


def test_attached_child_follow_false_collides():
    """Negative control: `follow=False` pins the billet to the plate's
    *rest* frame. When the plate tilts up it sweeps through the static
    billet, so the verifier reports a penetration — proving the follow
    path is what avoids the collision (not a blanket pass)."""
    from dycraft.sdk import scene_query

    ir, trunnion_ir, billet_ir = _trunnion_scene(follow=False)
    catalog = {trunnion_ir["id"]: trunnion_ir, billet_ir["id"]: billet_ir}

    # Rest pose is still clean (billet flush on the plate top).
    assert scene_query.penetration_pairs(ir, catalog) == []

    # As the plate tilts up it sweeps through the static billet (the plate
    # material under the billet rises into it at ~15-30 deg, then swings
    # past at larger angles).
    import math
    pairs = scene_query.penetration_pairs(
        ir, catalog, poses={"trunnion.tilt": math.radians(15)})
    assert len(pairs) == 1
    assert {pairs[0].a, pairs[0].b} == {"trunnion", "billet"}

    # Explicit `static_only=False`: the whole point of this control is the
    # swept tilt, and the sweep is not the scene default.
    report = no_cross_instance_penetration(
        ir, _ctx_of(trunnion_ir, billet_ir), static_only=False)
    assert report.verdict == "fail", report.evidence
