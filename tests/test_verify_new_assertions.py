"""Tests for the new base assertions (overlap, angle) and sugar layer.

Covers:
  * `assert_overlap`: positive overlap depth, negative gap distance,
    pose-tag interaction.
  * `assert_angle`: parallel/antiparallel/perpendicular, world directions.
  * Sugar: `assert_within` expands to face-anchor distance records with
    auto-suffixed names; failures cite the specific subname.
  * Pose tagging: per-ref `.at(...)`, assertion-level `poses=` shorthand,
    silent-ignore of unrelated joint names.
  * `joint.axis()` build-time error for fixed joints.
"""

import math

import pytest

from dycraft.sdk import Object, Direction
from dycraft.sdk._errors import InvalidAnchor
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.verify.object_verify import object_relational_assertions


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


# ---------------------------------------------------------------------------
# Distance merger: axis=None vs axis=tuple
# ---------------------------------------------------------------------------


def _two_box_object():
    o = Object("o")
    body = o.make_box("body", 0.4, 0.4, 0.4)
    door = o.make_box("door", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", body, door,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)), child_anchor=o.anchor(door, "bottom"),
    )
    o.declare_root(body)
    return o, body, door


def test_distance_euclidean_when_axis_omitted():
    o, body, door = _two_box_object()
    o.assert_distance(
        a=o.anchor(body, "center"), b=o.anchor(door, "center"),
        min=0.0, max=1.0,
        name="centers_within_1m",
    )
    ir = o.finalize()
    record = ir["assertions"][0]
    assert record["kind"] == "distance"
    assert "axis" not in record["args"]
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"


def test_distance_signed_projection_when_axis_present():
    o, body, door = _two_box_object()
    o.assert_distance(
        a=o.anchor(body, "top"), b=o.anchor(door, "bottom"),
        axis=(0, 0, 1), min=0.0, max=1.0,
        name="door_above_body",
    )
    ir = o.finalize()
    record = ir["assertions"][0]
    assert record["kind"] == "distance"
    assert record["args"]["axis"] == [0.0, 0.0, 1.0]


def test_distance_euclidean_rejects_negative_min():
    o, body, door = _two_box_object()
    with pytest.raises(Exception):
        o.assert_distance(
            a=o.anchor(body, "center"), b=o.anchor(door, "center"),
            min=-0.1, max=1.0,  # axis=None requires min >= 0
            name="bad",
        )


def test_distance_signed_allows_negative_min():
    o, body, door = _two_box_object()
    # min=-inf signed bound on +Z projection
    o.assert_distance(
        a=o.anchor(body, "top"), b=o.anchor(door, "bottom"),
        axis=(0, 0, 1), min=float("-inf"), max=1.0,
        name="door_anywhere_on_z",
    )
    ir = o.finalize()
    assert ir["assertions"][0]["args"]["min"] == "-inf"


# ---------------------------------------------------------------------------
# assert_overlap (base)
# ---------------------------------------------------------------------------


def test_overlap_positive_for_touching_parts():
    o = Object("o")
    a = o.make_box("a", 0.4, 0.4, 0.4)
    b = o.make_box("b", 0.4, 0.4, 0.4)
    # Stack b on top of a; they share the boundary z plane.
    o.attach_fixed(
        "j", a, b,
        parent_anchor=o.anchor(a, "top"),
        child_anchor=o.anchor(b, "bottom"),
    )
    o.declare_root(a)
    # On z axis, a covers [-0.2, 0.2], b covers [0.2, 0.6]; overlap is 0.
    o.assert_overlap(a=a, b=b, axis=(0, 0, 1), min=0.0, max=0.1,
                     name="z_contact")
    # On x axis, both cover [-0.2, 0.2]; overlap is 0.4.
    o.assert_overlap(a=a, b=b, axis=(1, 0, 0), min=0.39, max=0.41,
                     name="x_full_overlap")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"
    z_overlap = next(r for r in report.evidence["results"]
                     if r["name"] == "z_contact")
    assert z_overlap["value"] == pytest.approx(0.0, abs=1e-9)


def test_overlap_negative_for_gap():
    o = Object("o")
    a = o.make_box("a", 0.4, 0.4, 0.4)
    b = o.make_box("b", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", a, b,
        parent_anchor=o.anchor(a, "top", translation=(0, 0, 0.5)), child_anchor=o.anchor(b, "bottom"),  # b is 0.5m above a's top
    )
    o.declare_root(a)
    o.assert_overlap(a=a, b=b, axis=(0, 0, 1),
                     min=-1.0, max=0.0,  # signed gap allowed
                     name="z_gap")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    z = report.evidence["results"][0]
    assert z["verdict"] == "pass"
    assert z["value"] == pytest.approx(-0.5, abs=1e-6)  # negative = gap of 0.5m


# ---------------------------------------------------------------------------
# assert_angle (base)
# ---------------------------------------------------------------------------


def test_angle_with_joint_axis_and_world_axis():
    o = Object("o")
    a = o.make_box("a", 0.5, 0.5, 0.5)
    b = o.make_box("b", 0.5, 0.5, 0.5)
    hinge = o.attach_revolute(
        "h", a, b,
        parent_anchor=o.anchor(a, "top"),
        child_anchor=o.anchor(b, "bottom"),
        axis=(0, 0, 1),
        limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(a)
    o.assert_angle(a=hinge.axis(), b=Direction.world_z(),
                   min_deg=0.0, max_deg=1.0,
                   name="hinge_is_vertical")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    angle = report.evidence["results"][0]
    assert angle["verdict"] == "pass"
    assert angle["value"] == pytest.approx(0.0, abs=1e-6)


def test_fixed_joint_axis_raises_at_build():
    o = Object("o")
    a = o.make_box("a", 0.5, 0.5, 0.5)
    b = o.make_box("b", 0.5, 0.5, 0.5)
    fixed = o.attach_fixed(
        "f", a, b,
        parent_anchor=o.anchor(a, "top"),
        child_anchor=o.anchor(b, "bottom"),
    )
    o.declare_root(a)
    with pytest.raises(Exception) as exc:
        fixed.axis()
    assert "fixed" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Sugar: assert_within
# ---------------------------------------------------------------------------


def test_within_emits_2N_distance_records_with_suffixes():
    o = Object("o")
    outer = o.make_box("outer", 1.0, 1.0, 1.0)
    inner = o.make_box("inner", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", outer, inner,
        parent_anchor=o.anchor(outer, "center"),
        child_anchor=o.anchor(inner, "center"),
    )
    o.declare_root(outer)
    o.assert_within(inner=inner, outer=outer, axes=("x", "y"),
                    margin=0.0, name="inner_in_outer")
    ir = o.finalize()
    names = [a["name"] for a in ir["assertions"]]
    assert "inner_in_outer__within_lo_x" in names
    assert "inner_in_outer__within_hi_x" in names
    assert "inner_in_outer__within_lo_y" in names
    assert "inner_in_outer__within_hi_y" in names
    # All four should be `distance` kind records.
    for a in ir["assertions"]:
        if a["name"].startswith("inner_in_outer__"):
            assert a["kind"] == "distance"
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"


def test_within_fails_when_inner_pokes_out():
    o = Object("o")
    outer = o.make_box("outer", 0.5, 0.5, 0.5)
    inner = o.make_box("inner", 1.0, 0.4, 0.4)  # wider than outer
    o.attach_fixed(
        "j", outer, inner,
        parent_anchor=o.anchor(outer, "center"),
        child_anchor=o.anchor(inner, "center"),
    )
    o.declare_root(outer)
    o.assert_within(inner=inner, outer=outer, axes=("x",),
                    margin=0.0, name="inner_in_outer")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "fail"
    # Failure cites the specific axis subname.
    fail_names = [r["name"] for r in report.evidence["results"]
                  if r["verdict"] == "fail"]
    assert any("inner_in_outer__within_" in n for n in fail_names)


# ---------------------------------------------------------------------------
# Pose tagging
# ---------------------------------------------------------------------------


def test_anchor_at_pose_returns_different_world_position():
    from dycraft.sdk import object_query as oq
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    hinge = o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(body)
    ir = o.finalize()
    rest_pos = oq.anchor_position(ir, "door", "top")
    open_pos = oq.anchor_position(ir, "door", "top",
                                   poses={"hinge": math.pi / 2})
    assert rest_pos != open_pos


def test_unrelated_pose_key_silently_ignored():
    from dycraft.sdk import object_query as oq
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    hinge = o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(body)
    ir = o.finalize()
    only_hinge = oq.anchor_position(ir, "door", "top",
                                     poses={"hinge": 1.0})
    with_extra = oq.anchor_position(ir, "door", "top",
                                     poses={"hinge": 1.0,
                                            "phantom_joint": 99.0})
    assert only_hinge == with_extra


def test_assertion_level_poses_fills_untagged_refs():
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    hinge = o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(body)
    o.assert_distance(
        a=o.anchor(door, "top"),
        b=o.anchor(door, "bottom"),
        # Surface-unit pose: 90 degrees for a revolute joint. The IR
        # stores the converted radian value (matching `limits`/`default`).
        poses={"hinge": 90.0},
        min=0.0, max=10.0,
        name="door_at_open",
    )
    ir = o.finalize()
    a_args = ir["assertions"][0]["args"]
    # Assertion-level poses copied into both untagged refs and converted
    # to IR-internal radians at serialization time.
    assert a_args["a"]["poses"] == {"hinge": pytest.approx(math.pi / 2)}
    assert a_args["b"]["poses"] == {"hinge": pytest.approx(math.pi / 2)}


def test_per_ref_poses_override_assertion_level():
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    hinge = o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(body)
    o.assert_distance(
        a=o.anchor(door, "center").at({hinge: 0.0}),  # explicit rest, deg
        b=o.anchor(door, "center").at({hinge: 90.0}),  # explicit open, deg
        poses={"hinge": 45.0},  # assertion-level: ignored, per-ref wins
        axis=(1, 0, 0),
        min=-10.0, max=10.0,
        name="door_swing_displacement",
    )
    ir = o.finalize()
    a_args = ir["assertions"][0]["args"]
    # Per-ref poses convert deg → rad before IR serialization.
    assert a_args["a"]["poses"] == {"hinge": 0.0}
    assert a_args["b"]["poses"] == {"hinge": pytest.approx(math.pi / 2)}


# ---------------------------------------------------------------------------
# Pose-unit boundary: SDK surface is degrees; IR + kernel are radians.
# ---------------------------------------------------------------------------


def _hinge_ir():
    """Object with a single revolute hinge on the +X edge of `body`."""
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 180.0), default=0.0,
    )
    o.declare_root(body)
    return o.finalize()


def test_probe_anchor_position_takes_degrees_for_revolute():
    """`object_anchor_position(ir, ref, poses={hinge: 90})` (surface units)
    matches a direct kernel call with `{hinge: pi/2}` (internal units).
    """
    from dycraft.sdk import _probes
    from dycraft.sdk import _object_kernel as ok
    ir = _hinge_ir()
    surface = _probes.object_anchor_position(
        ir, "door", "center", poses={"hinge": 90.0})
    internal = ok.anchor_position(
        ir, "door", "center", poses={"hinge": math.pi / 2})
    assert surface == pytest.approx(internal)


def test_ref_at_takes_degrees_for_revolute():
    """A pose-tagged AnchorRef built via `.at({hinge: 90})` resolves at the
    same world position as a kernel call with `{hinge: pi/2}`.
    """
    from dycraft.sdk import _probes
    from dycraft.sdk import _object_kernel as ok
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    door = o.make_box("door", 0.05, 0.4, 0.4)
    hinge = o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 180.0), default=0.0,
    )
    o.declare_root(body)
    ref = o.anchor(door, "center").at({hinge: 90.0})
    ir = o.finalize()
    surface = _probes.object_anchor_position(ir, ref)
    internal = ok.anchor_position(
        ir, "door", "center", poses={"hinge": math.pi / 2})
    assert surface == pytest.approx(internal)


def test_prismatic_pose_values_pass_through_unchanged():
    """Prismatic joints use meters on both surface and IR — no conversion."""
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    tray = o.make_box("tray", 0.4, 0.4, 0.05)
    o.attach_prismatic(
        "slide", body, tray,
        parent_anchor=o.anchor(body, "front"),
        child_anchor=o.anchor(tray, "back"),
        axis=(0, 1, 0), limits=(0.0, 0.30), default=0.0,
    )
    o.declare_root(body)
    o.assert_distance(
        a=o.anchor(tray, "front").at({"slide": 0.15}),
        b=o.anchor(tray, "back").at({"slide": 0.15}),
        min=0.0, max=10.0,
        name="tray_size_at_mid",
    )
    ir = o.finalize()
    # Prismatic surface == IR (both meters).
    assert ir["assertions"][0]["args"]["a"]["poses"] == {"slide": 0.15}


def test_no_self_penetration_witnessing_pose_in_degrees():
    """The CCD sampler runs in radians internally; the verifier surfaces
    `witnessing_pose` in degrees (matching probes / `.at(...)` / joint_table).

    Setup: body at origin, door hinged off the +X-side outer-front edge
    swinging on world +Z. At rest (q=0), the door extends out in +Y with
    no overlap. As the hinge sweeps past 90° the door wraps around and
    clips the body — CCD sweep finds the penetrating pose somewhere in
    the upper half of the (0, 180) range. The witnessing_pose dict must
    expose that pose in DEGREES (would be ~1.5–3.14 in radians).
    """
    from dycraft.verify.object_verify import no_self_penetration
    o = Object("o")
    body = o.make_box("body", 0.6, 0.6, 0.6)
    door = o.make_box("door", 0.05, 0.6, 0.6)
    o.attach_revolute(
        "hinge", body, door,
        parent_anchor=o.anchor(body, "front_right"),
        child_anchor=o.anchor(door, "back_left"),
        axis=(0, 0, 1), limits=(0.0, 180.0), default=0.0,
    )
    o.declare_root(body)
    ir = o.finalize()
    report = no_self_penetration(ir)
    assert report.verdict == "fail"
    saw_nonzero_hinge_in_degrees = False
    for pair in report.evidence["pairs"]:
        wp = pair["witnessing_pose"]
        if "hinge" in wp and wp["hinge"] > 4.0:
            # Value > pi (~3.14) is impossible if the units were radians
            # over a (0, pi) range. Surface-unit (degrees) sweep can hit
            # up to 180.
            saw_nonzero_hinge_in_degrees = True
            assert wp["hinge"] <= 180.0
    assert saw_nonzero_hinge_in_degrees, (
        "expected at least one penetrating sample's witnessing_pose hinge "
        "to be in degrees (> 4.0), got " + repr(
            [p["witnessing_pose"] for p in report.evidence["pairs"]])
    )


# ---------------------------------------------------------------------------
# assert_axis_world_aligned: unsigned (accepts both +/- axis directions)
# ---------------------------------------------------------------------------


def _hinge_along(axis):
    """Build a minimal object with a single revolute joint along `axis`."""
    o = Object("o")
    a = o.make_box("a", 0.5, 0.5, 0.5)
    b = o.make_box("b", 0.5, 0.5, 0.5)
    hinge = o.attach_revolute(
        "h", a, b,
        parent_anchor=o.anchor(a, "top"),
        child_anchor=o.anchor(b, "bottom"),
        axis=axis,
        limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(a)
    return o, hinge


def test_axis_world_aligned_passes_for_positive_axis():
    o, hinge = _hinge_along((0, 0, 1))
    o.assert_axis_world_aligned(joint=hinge, world_axis="z",
                                 tol_deg=1.0, name="z_aligned")
    ir = o.finalize()
    record = ir["assertions"][0]
    assert record["kind"] == "axis_aligned"
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"


def test_axis_world_aligned_passes_for_negative_axis():
    """An `axis=(0,0,-1)` joint passes world_axis='z'.

    The check is symmetric: requiring strict parallel would force authors
    to flip their kinematically sensible axis sign just to satisfy it.
    """
    o, hinge = _hinge_along((0, 0, -1))
    o.assert_axis_world_aligned(joint=hinge, world_axis="z",
                                 tol_deg=1.0, name="z_aligned")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    result = report.evidence["results"][0]
    assert result["verdict"] == "pass"
    assert result["value"] == pytest.approx(0.0, abs=1e-6)
    assert result["raw_deg"] == pytest.approx(180.0, abs=1e-6)


def test_axis_world_aligned_fails_for_perpendicular_axis():
    o, hinge = _hinge_along((1, 0, 0))
    o.assert_axis_world_aligned(joint=hinge, world_axis="z",
                                 tol_deg=1.0, name="z_aligned")
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "fail"


# ---------------------------------------------------------------------------
# DirectionRef.world_vec() — readable for world literals; raises elsewhere
# ---------------------------------------------------------------------------


def test_direction_world_vec_returns_tuple_for_world_literal():
    v = Direction.world((1.0, 0.0, 0.0)).world_vec()
    assert v == (1.0, 0.0, 0.0)
    assert Direction.world_x().world_vec() == (1.0, 0.0, 0.0)
    assert Direction.world_y().world_vec() == (0.0, 1.0, 0.0)
    assert Direction.world_z().world_vec() == (0.0, 0.0, 1.0)


def test_direction_world_vec_raises_for_joint_axis():
    o = Object("o")
    a = o.make_box("a", 0.1, 0.1, 0.1)
    b = o.make_box("b", 0.1, 0.1, 0.1)
    j = o.attach_revolute(
        "h", a, b, parent_anchor=o.anchor(a, "top"),
        child_anchor=o.anchor(b, "bottom"),
        axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,
    )
    o.declare_root(a)
    # `InvalidAnchor`, not a bare ValueError: authoring mistakes on the ref
    # surface report through the structured error system, so the fix travels
    # with the complaint instead of the agent getting the interpreter's words.
    with pytest.raises(InvalidAnchor) as exc:
        j.axis().world_vec()
    msg = str(exc.value)
    assert exc.value.suggested_repairs
    assert "joint_axis" in msg
    # The author must be pointed at the probe as they can actually call it. The
    # `object_`-prefixed forms are internal `sdk._probes` aliases; the overlay
    # binds the bare names, so pointing at the alias sends them to a NameError.
    #
    # Checked against the whole agent-visible payload — summary plus repairs —
    # because that is what the author reads, and which of the two fields carries
    # the sentence is not the property this test is about.
    agent_sees = msg + " " + " ".join(exc.value.suggested_repairs)
    assert "joint_axis_world" in agent_sees
    assert "object_joint_axis_world" not in agent_sees


def test_direction_world_vec_raises_for_anchor_outward():
    o = Object("o")
    a = o.make_box("a", 0.1, 0.1, 0.1)
    o.declare_root(a)
    with pytest.raises(InvalidAnchor) as exc:
        o.anchor(a, "top").outward().world_vec()
    assert "anchor_outward" in str(exc.value)
    assert exc.value.suggested_repairs


# ---------------------------------------------------------------------------
# allow_collision: blanket + scoped (joint, q_range)
# ---------------------------------------------------------------------------


def _slider_object_overlapping_at_extension():
    """A box body with a slab that prismatic-slides INTO the body.

    At q=0 the slab is outside; at q=0.1 it is fully inside. With no
    allow_collision, NoSelfPenetration fires on the q=0.1 sample.
    """
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    slab = o.make_box("slab", 0.4, 0.4, 0.05)
    o.attach_prismatic(
        "slide", body, slab,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)),
        child_anchor=o.anchor(slab, "bottom"),
        axis=(0.0, 0.0, -1.0),
        limits=(0.0, 0.20), default=0.0,
    )
    o.declare_root(body)
    return o, body, slab


def test_blanket_allow_collision_suppresses_all_samples():
    from dycraft.verify.object_verify import no_self_penetration
    o, body, slab = _slider_object_overlapping_at_extension()
    o.allow_collision(body, slab, reason="slab is inside the cavity at full slide")
    ir = o.finalize()
    assert no_self_penetration(ir).verdict == "pass"


def test_scoped_allow_collision_only_suppresses_in_range():
    from dycraft.verify.object_verify import no_self_penetration
    o, body, slab = _slider_object_overlapping_at_extension()
    # Need a JointRef — rebuild with explicit handle.
    _reset_object_singleton()
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    slab = o.make_box("slab", 0.4, 0.4, 0.05)
    slide = o.attach_prismatic(
        "slide", body, slab,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)),
        child_anchor=o.anchor(slab, "bottom"),
        axis=(0.0, 0.0, -1.0),
        limits=(0.0, 0.20), default=0.0,
    )
    o.allow_collision(body, slab,
                       reason="slab is inside the body only when extended",
                       joint=slide, q_range=(0.05, 0.20))
    o.declare_root(body)
    ir = o.finalize()
    # Penetration only occurs while q is in [0.05, 0.20] (slab moving
    # into body); the scoped allowance covers exactly that range, so
    # the verifier should pass.
    assert no_self_penetration(ir).verdict == "pass"


def test_scoped_allow_collision_does_not_mask_out_of_range_overlap():
    """A scoped allowance must not blanket-exempt all samples.

    Construct an object whose only overlap pose is at q=0.05, then
    scope the allowance to q=(0.10, 0.20) — the overlap at q=0.05
    must still be reported as a failure.
    """
    from dycraft.verify.object_verify import no_self_penetration
    _reset_object_singleton()
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    slab = o.make_box("slab", 0.4, 0.4, 0.05)
    slide = o.attach_prismatic(
        "slide", body, slab,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)),
        child_anchor=o.anchor(slab, "bottom"),
        axis=(0.0, 0.0, -1.0),
        limits=(0.0, 0.20), default=0.0,
    )
    # Scope allowance to a range that does NOT cover where penetration
    # actually happens. The per-joint sweep visits q ∈ {0, 0.10, 0.20};
    # at q=0.20 the slab penetrates body but the allowance covers it.
    # The hypercube samples between the steps and finds overlap in
    # (0.10, 0.18) where the allowance does not apply — so the test
    # explicitly enables the hypercube via per_joint_only=False.
    o.allow_collision(body, slab,
                       reason="(intentionally too-narrow scope)",
                       joint=slide, q_range=(0.18, 0.20))
    o.declare_root(body)
    ir = o.finalize()
    rep = no_self_penetration(ir, per_joint_only=False)
    assert rep.verdict == "fail"


def test_allow_collision_blanket_and_scoped_conflict():
    o = Object("o")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    slab = o.make_box("slab", 0.4, 0.4, 0.05)
    slide = o.attach_prismatic(
        "slide", body, slab,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)),
        child_anchor=o.anchor(slab, "bottom"),
        axis=(0.0, 0.0, -1.0),
        limits=(0.0, 0.20), default=0.0,
    )
    o.allow_collision(body, slab, reason="blanket")
    with pytest.raises(Exception) as exc:
        o.allow_collision(body, slab, reason="scoped",
                           joint=slide, q_range=(0.0, 0.1))
    assert "blanket" in str(exc.value).lower()


def test_allow_collision_rejects_q_range_without_joint():
    o = Object("o")
    a = o.make_box("a", 0.5, 0.5, 0.5)
    b = o.make_box("b", 0.5, 0.5, 0.5)
    o.attach_fixed("j", a, b,
                   parent_anchor=o.anchor(a, "top"),
                   child_anchor=o.anchor(b, "bottom"))
    o.declare_root(a)
    with pytest.raises(Exception) as exc:
        o.allow_collision(a, b, reason="bad", q_range=(0.0, 0.1))
    assert "q_range" in str(exc.value)


def test_allow_collision_rejects_fixed_joint_scope():
    o = Object("o")
    a = o.make_box("a", 0.5, 0.5, 0.5)
    b = o.make_box("b", 0.5, 0.5, 0.5)
    fj = o.attach_fixed("j", a, b,
                        parent_anchor=o.anchor(a, "top"),
                        child_anchor=o.anchor(b, "bottom"))
    o.declare_root(a)
    with pytest.raises(Exception) as exc:
        o.allow_collision(a, b, reason="bad", joint=fj, q_range=(0.0, 1.0))
    assert "fixed" in str(exc.value).lower()


def test_every_registered_assertion_kind_declares_how_it_is_called():
    """The registry's calling convention must match each evaluator's signature.

    `_EVAL_BY_KIND` maps a kind to `(evaluator, takes_resolver_namespace)`, so
    registering a kind with the wrong flag hands the evaluator the wrong first
    argument — a TypeError at verify time rather than an error at registration.
    The
    evaluators name their first parameter for the convention they expect
    (`resolvers` vs `resolve_anchor_fn`), so the two can be checked against each
    other directly.
    """
    import inspect

    from dycraft.verify import _relational as R

    assert R._EVAL_BY_KIND, "registry is empty"
    for kind, entry in R._EVAL_BY_KIND.items():
        fn, wants_resolvers = entry
        first = list(inspect.signature(fn).parameters)[0]
        expected = "resolvers" if wants_resolvers else "resolve_anchor_fn"
        assert first == expected, (
            f"{kind}: registry says takes_resolver_namespace={wants_resolvers}, "
            f"but {fn.__name__}'s first parameter is {first!r}")
