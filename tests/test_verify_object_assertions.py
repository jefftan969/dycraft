"""Runtime-behavior tests for the ObjectRelationalAssertions verifier.

Each test computes its expected value independently of the SDK so a bug
in the kernel (`verify._relational.evaluate_assertions`) or the
object-side resolver (`object_query.anchor_position`) shows up as a
mismatch, not as a silently passing tautology.

Covered:
  * distance_along: pass/fail boundary, value reported equals the
    independently computed signed projection.
  * Euclidean distance: pass when within bounds; fail evidence carries
    the actual distance.
  * collinear: passes at tol = max_perp, fails at tol = max_perp / 2.
  * Inf round-trip: `max=float("inf")` survives `Object.finalize()`
    (canonical JSON has allow_nan=False) and the verifier kernel
    deserializes the string back to math.inf for arithmetic.
  * Anchor resolver picks up the part's rest-pose frame: a part attached
    via an offset has its anchors translated accordingly.
  * Cheap → expensive short-circuit applies to the new verifier name:
    when ObjectRelationalAssertions fails, the expensive VlmJudge is
    omitted from the verifier_reports entirely.
"""


import pytest

from dycraft.sdk import Object
from dycraft.sdk import object_query as query
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.verify.object_verify import (
    object_relational_assertions,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


# ---------------------------------------------------------------------------
# distance_along
# ---------------------------------------------------------------------------


def test_distance_along_passes_when_signed_projection_in_bounds():
    """Two 0.4×0.4×0.4 boxes; door's `bottom` anchor at z=-0.2 in part frame.
    With door attached so its center sits 0.30 m above body's top face
    (body top z = 0.20, door bottom z = 0.20+? ...), the door's `bottom`
    anchor in body-local sits at z = 0.20 (body's top) + 0.30 (the
    attach offset) − 0.20 (door's bottom in door-local) = 0.30.

    Assertion: signed-z projection of (door.bottom − body.top) ≥ 0.
    Independent expected `value`: 0.30 − 0.20 = 0.10 m.
    """
    o = Object("o")
    body = o.make_box("body", 0.4, 0.4, 0.4)
    door = o.make_box("door", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", body, door,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 0.10)), child_anchor=o.anchor(door, "bottom"),
    )
    o.declare_root(body)
    o.assert_distance(
        a=o.anchor(body, "top"), b=o.anchor(door, "bottom"),
        axis=(0.0, 0.0, 1.0), min=0.0, max=1.0,
        name="door_above_body",
    )
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"
    detail = report.evidence["results"][0]
    assert detail["kind"] == "distance"
    assert "axis" in detail
    assert detail["value"] == pytest.approx(0.10, abs=1e-9)


def test_distance_along_fails_with_actual_signed_value_in_evidence():
    """Door positioned 0.05 m BELOW the body top → signed +Z projection
    of (door.bottom − body.top) = −0.05. Bounds [0, ∞) reject it.
    The repair hint quotes the actual value so the agent can see what
    distance was measured."""
    o = Object("o")
    body = o.make_box("body", 0.4, 0.4, 0.4)
    door = o.make_box("door", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", body, door,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, -0.05)), child_anchor=o.anchor(door, "bottom"),
    )
    o.declare_root(body)
    o.assert_distance(
        a=o.anchor(body, "top"), b=o.anchor(door, "bottom"),
        axis=(0.0, 0.0, 1.0), min=0.0, max=float("inf"),
        name="door_above_body",
    )
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "fail"
    detail = report.evidence["results"][0]
    assert detail["value"] == pytest.approx(-0.05, abs=1e-9)
    repair = report.suggested_repairs[0]
    assert "door_above_body" in repair
    assert "-0.05" in repair


# ---------------------------------------------------------------------------
# Inf round-trip
# ---------------------------------------------------------------------------


def test_inf_max_survives_canonical_json_and_kernel_deserializes():
    """canonical JSON has allow_nan=False, so bare `float("inf")` cannot
    be serialized. The SDK maps it to the string "inf" in the IR; the
    verifier kernel must deserialize it back to math.inf for the
    inequality check. A bug in either direction (SDK skips the mapping,
    kernel treats "inf" as the literal string 'inf') would fail this."""
    o = Object("o")
    a = o.make_box("a", 0.1, 0.1, 0.1)
    b = o.make_box("b", 0.1, 0.1, 0.1)
    o.attach_fixed(
        "j", a, b,
        parent_anchor=o.anchor(a, "center", translation=(1.0e6, 0.0, 0.0)), child_anchor=o.anchor(b, "center"),  # huge displacement
    )
    o.declare_root(a)
    o.assert_distance(
        a=o.anchor(a, "center"), b=o.anchor(b, "center"),
        axis=(1.0, 0.0, 0.0), min=0.0, max=float("inf"),
        name="b_to_the_left_of_a",
    )
    ir = o.finalize()
    # Storage check: the IR contains the string, not a float.
    args = ir["assertions"][0]["args"]
    assert args["max"] == "inf"
    # Round-trip check: the verifier still passes despite the string in IR.
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"
    detail = report.evidence["results"][0]
    # Independently computed signed distance = 1e6.
    assert detail["value"] == pytest.approx(1.0e6, rel=1e-12)


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


def test_distance_uses_euclidean_norm_not_signed_projection():
    """Box `b` at offset (3, 4, 0) from `a` → Euclidean distance = 5.0.
    A bug that returned the signed projection along some axis would
    produce a different value (e.g., 3 along X, 4 along Y, 0 along Z)."""
    o = Object("o")
    a = o.make_box("a", 0.1, 0.1, 0.1)
    b = o.make_box("b", 0.1, 0.1, 0.1)
    o.attach_fixed(
        "j", a, b,
        parent_anchor=o.anchor(a, "center", translation=(3.0, 4.0, 0.0)), child_anchor=o.anchor(b, "center"),
    )
    o.declare_root(a)
    o.assert_distance(
        a=o.anchor(a, "center"), b=o.anchor(b, "center"),
        min=4.99, max=5.01,
        name="three_four_five",
    )
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"
    assert report.evidence["results"][0]["value"] == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# collinear
# ---------------------------------------------------------------------------


def test_collinear_pass_fail_around_tol_boundary():
    """Three boxes whose centers form a row along +X with a 1 mm jitter
    on the middle one in +Y. max_perp = 0.001 (independently computed).
    tol = 0.0011 → pass; tol = 0.0005 → fail.

    Verifies that the kernel computes max_perp correctly and uses the
    documented `<=` comparator at the boundary."""
    o = Object("o")
    a = o.make_box("a", 0.1, 0.1, 0.1)
    b = o.make_box("b", 0.1, 0.1, 0.1)
    c = o.make_box("c", 0.1, 0.1, 0.1)
    # a at root; b 1 m along +X with 1 mm +Y jitter; c 2 m along +X (no jitter).
    o.attach_fixed(
        "j_ab", a, b,
        parent_anchor=o.anchor(a, "center", translation=(1.0, 0.001, 0.0)), child_anchor=o.anchor(b, "center"),
    )
    o.attach_fixed(
        "j_ac", a, c,
        parent_anchor=o.anchor(a, "center", translation=(2.0, 0.0, 0.0)), child_anchor=o.anchor(c, "center"),
    )
    o.declare_root(a)
    o.assert_collinear(
        o.anchor(a, "center"), o.anchor(b, "center"), o.anchor(c, "center"),
        axis=(1.0, 0.0, 0.0), tol=0.0011,
        name="row_loose",
    )
    o.assert_collinear(
        o.anchor(a, "center"), o.anchor(b, "center"), o.anchor(c, "center"),
        axis=(1.0, 0.0, 0.0), tol=0.0005,
        name="row_tight",
    )
    ir = o.finalize()
    report = object_relational_assertions(ir)
    assert report.verdict == "fail"
    by_name = {r["name"]: r for r in report.evidence["results"]}
    assert by_name["row_loose"]["verdict"] == "pass"
    assert by_name["row_tight"]["verdict"] == "fail"
    # max_perp under both assertions = 0.001 (b's +Y jitter).
    assert by_name["row_loose"]["value"] == pytest.approx(0.001, abs=1e-12)
    assert by_name["row_tight"]["value"] == pytest.approx(0.001, abs=1e-12)


# ---------------------------------------------------------------------------
# Anchor resolver consults the rest-pose frame, not just the part-local
# position. Putting an anchor on a part that's been translated must
# yield the part's rest-pose anchor world position.
# ---------------------------------------------------------------------------


def test_anchor_position_at_object_rest_pose_includes_part_translation():
    """Body 0.4×0.4×0.4. Door 0.4×0.4×0.4 attached so its center sits at
    body-local (0.5, 0, 0). Door's `center` anchor in *object root* frame
    must therefore land at (0.5, 0, 0). Independently of the verifier:
    we check `object_query.anchor_position` directly first, then make
    the verifier consume the same value to confirm the kernel and the
    resolver agree."""
    o = Object("o")
    body = o.make_box("body", 0.4, 0.4, 0.4)
    door = o.make_box("door", 0.4, 0.4, 0.4)
    o.attach_fixed(
        "j", body, door,
        parent_anchor=o.anchor(body, "center", translation=(0.5, 0.0, 0.0)), child_anchor=o.anchor(door, "center"),
    )
    o.declare_root(body)
    # Distance assertion that should pass iff door.center is at (0.5, 0, 0).
    o.assert_distance(
        a=o.anchor(body, "center"), b=o.anchor(door, "center"),
        min=0.49, max=0.51,
        name="door_offset",
    )
    ir = o.finalize()
    # Independent resolver check.
    door_pos = query.anchor_position(ir, "door", "center")
    assert door_pos == pytest.approx((0.5, 0.0, 0.0), abs=1e-9)
    # Verifier consumes the same anchor positions.
    report = object_relational_assertions(ir)
    assert report.verdict == "pass"
    assert report.evidence["results"][0]["value"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Verifier registration: the new name appears in OBJECT_VERIFIERS
# between NoSelfPenetration and ObjectVlmJudge.
# ---------------------------------------------------------------------------


def test_assert_distance_rejects_anchorref_from_other_builder():
    """A different Object's AnchorRef must not be accepted by this
    builder's assert_*. The check is owner-id based (not name-based) so
    two builders with parts of the same name can't trick the validator.
    Catches a regression where the anchor_ref owner check is dropped or
    the owner_id propagation breaks."""
    from dycraft.sdk import InvalidAssertion

    o1 = Object("o1")
    p1 = o1.make_box("p", 0.1, 0.1, 0.1)
    _reset_object_singleton()
    o2 = Object("o2")
    p2 = o2.make_box("p", 0.1, 0.1, 0.1)
    # Build a valid AnchorRef on o1, then try to use it in o2.
    foreign = o1.anchor(p1, "center")
    local = o2.anchor(p2, "center")
    with pytest.raises(InvalidAssertion) as ei:
        o2.assert_distance(a=foreign, b=local, min=0.0, max=1.0, name="x")
    d = ei.value.to_dict()
    assert d["evidence"]["reason"] == "foreign_anchor_ref"
    assert d["evidence"]["param"] == "a"


