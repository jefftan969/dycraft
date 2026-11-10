"""Tests for dycraft.sdk (object-level v1).

Focused on behavior that isn't obvious from reading the code: FK math
(rotation about a non-origin point, multi-joint composition), Euler
intrinsic-XYZ convention, quaternion sign canonicalization, axis
normalization, AABB-under-rotation, penetration depth, and the cases
where the structural-check BFS actually has to do work.

Run: `conda run -n chord pytest dycraft/tests/test_sdk_object.py -v`
"""

import math

import pytest

from dycraft.sdk import (
    Object,
    BuildError,
    InvalidJoint,
    DanglingReference,
    NotConnectedToRoot,
    object_query as query,
)
from dycraft.sdk.object_builder import _reset_object_singleton


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(a, b, *, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=abs_tol)


def _approx_vec(a, b, *, abs_tol=1e-9):
    return all(_approx(x, y, abs_tol=abs_tol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Determinism: hash stability and what perturbs it.
# ---------------------------------------------------------------------------

def _make_microwave(door_height=0.30, hinge_default=0.0):
    obj = Object("microwave")
    body = obj.make_box("body", 0.50, 0.40, 0.30)
    door = obj.make_box("door", 0.50, 0.02, door_height)
    obj.attach_revolute('hinge', body, door, axis=(0.0, 0.0, 1.0), limits=(0.0, 110.0), default=hinge_default, parent_anchor=obj.anchor(body, "center", translation=(0.25, 0.21000000000000002, 0.0)), child_anchor=obj.anchor(door, "center"))
    obj.declare_root(body)
    return obj.finalize()


def test_hash_is_stable_and_responds_to_geometry_and_state():
    a = _make_microwave()
    _reset_object_singleton()
    b = _make_microwave()
    _reset_object_singleton()
    c = _make_microwave(door_height=0.31)
    _reset_object_singleton()
    d = _make_microwave(hinge_default=45.0)

    assert a["id"] == b["id"], "byte-identical builds must produce the same hash"
    assert a["id"] != c["id"], "geometry change must perturb the hash"
    assert a["id"] != d["id"], "joint default change must perturb the hash"


# ---------------------------------------------------------------------------
# Axis normalization invariance (catches: forgot-to-normalize bug).
# ---------------------------------------------------------------------------

def test_axis_scaling_does_not_change_ir():
    def build(scale):
        obj = Object("o")
        a = obj.make_box("a", 1, 1, 1)
        b = obj.make_box("b", 1, 1, 1)
        obj.attach_revolute('j', a, b, axis=(0.0, 0.0, 1.0 * scale), limits=(-90.0, 90.0), default=30.0, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
        obj.declare_root(a)
        return obj.finalize()

    ir1 = build(1.0)
    _reset_object_singleton()
    ir7 = build(7.3)
    assert ir1["id"] == ir7["id"]


# ---------------------------------------------------------------------------
# Every spelling of an axis the surface teaches must be accepted by every
# consumer that takes one: the sugars teach `axes=("x","y","z")` and
# `object.md` teaches `Direction.world((x,y,z))`, so `axis=` accepts the axis
# letters (`AXIS_LITERALS`) and a `DirectionRef` as well as a tuple.
# `coerce_axis` is the single choke point for joint axes and assertion axes
# alike, so one test covers both.
# ---------------------------------------------------------------------------

def _axis_spellings():
    from dycraft.sdk import Direction
    return [
        ("tuple", (0.0, 0.0, 1.0)),
        ("list", [0.0, 0.0, 1.0]),
        ("literal", "z"),
        ("signed literal", "+z"),
        ("Direction.world", Direction.world((0.0, 0.0, 1.0))),
        ("Direction.world_z", Direction.world_z()),
    ]


@pytest.mark.parametrize("label,axis", _axis_spellings(),
                         ids=[lbl for lbl, _ in _axis_spellings()])
def test_every_taught_axis_spelling_builds_the_same_joint(label, axis):
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_revolute("j", a, b, axis=axis, limits=(-90.0, 90.0),
                        default=0.0, parent_anchor=obj.anchor(a, "center"),
                        child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    assert _approx_vec(ir["joints"][0]["axis"], (0.0, 0.0, 1.0)), label


def test_negative_axis_literal_and_direction_agree():
    """`'-z'` and `Direction.world((0,0,-1))` are the same axis."""
    from dycraft.sdk import Direction

    def build(axis):
        obj = Object("o")
        a = obj.make_box("a", 1, 1, 1)
        b = obj.make_box("b", 1, 1, 1)
        obj.attach_revolute("j", a, b, axis=axis, limits=(-90.0, 90.0),
                            default=0.0,
                            parent_anchor=obj.anchor(a, "center"),
                            child_anchor=obj.anchor(b, "center"))
        obj.declare_root(a)
        return obj.finalize()

    literal = build("-z")
    _reset_object_singleton()
    ref = build(Direction.world((0.0, 0.0, -1.0)))
    assert literal["id"] == ref["id"]


def test_unresolvable_direction_ref_is_refused_and_routed_to_a_probe():
    """A joint/anchor direction is not knowable at build time.

    It must not be silently coerced to something — the refusal is the correct
    answer, and it has to name the probe that *can* resolve it.
    """
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    hinge = obj.attach_revolute(
        "j", a, b, axis=(0.0, 0.0, 1.0), limits=(-90.0, 90.0), default=0.0,
        parent_anchor=obj.anchor(a, "center"),
        child_anchor=obj.anchor(b, "center"))
    with pytest.raises(BuildError) as excinfo:
        obj.assert_distance(a=obj.anchor(a, "center"),
                            b=obj.anchor(b, "center"),
                            axis=hinge.axis(), min=0.0, max=1.0, name="d")
    err = excinfo.value
    assert err.evidence.get("reason") == "axis_ref_source"
    assert err.evidence.get("source") == "joint_axis"
    assert any("joint_axis_world" in r for r in err.suggested_repairs)


# ---------------------------------------------------------------------------
# Quaternion sign canonicalization. Euler (180, 180, 180) composes to the
# identity rotation, but the intermediate quaternion product is
# (-1, 0, 0, 0) — qw is negative. Without sign canonicalization the IR
# would store -1 instead of +1, and equivalent rotations represented by
# quaternions of opposite sign would hash differently. We verify the
# stored qw is non-negative on a case where the unsanitized product is
# negative.
# ---------------------------------------------------------------------------

def test_quaternion_sign_canonicalized_in_ir():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    qw, qx, qy, qz = ir["joints"][0]["parent_anchor"]["rotation"]
    assert qw >= 0.0
    # Euler (180,180,180) intrinsic XYZ composes to the identity rotation.
    assert _approx(qw, 1.0) and _approx(qx, 0.0) and _approx(qy, 0.0) and _approx(qz, 0.0)


# ---------------------------------------------------------------------------
# Euler XYZ convention is intrinsic (catches: extrinsic-vs-intrinsic
# swap, or wrong axis order in the quaternion product).
# ---------------------------------------------------------------------------

def test_intrinsic_xyz_convention_via_known_vector_image():
    """Intrinsic XYZ Euler (90, 90, 0) applied to (0, 1, 0) yields (0, 0, 1).

    Verify by placing a child whose origin is offset (0, 1, 0) in its own
    local frame from the joint, and reading the resulting frame translation.
    With child_anchor t=(0,1,0): inv_child_anchor t=(0,-1,0), so the child
    origin in parent = parent_anchor.t + parent_anchor.R · (0,-1,0) =
    -parent_anchor.R · (0,1,0). For Euler (90,90,0) that's -(0,0,1) =
    (0,0,-1)."""
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center", translation=(0.0, 0.0, -1.0), rotation=(89.99999999999999, 90.0, 0.0)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    t, _q = query.frame_of(ir, "b")
    assert _approx_vec(t, (0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# Revolute FK: rotation about a non-origin axis (catches: dropped the
# "shift to axis → rotate → shift back" composition; treated rotation as
# if it pivoted at the parent origin).
# ---------------------------------------------------------------------------

def test_revolute_rotates_about_joint_origin_not_parent_origin():
    """Joint pivot in parent at (0.5, 0, 0) — parent's `left` anchor on a
    1x1x1 box (viewer-relative +X = left = +X_part). Child b is attached
    so its `right` anchor (-0.5, 0, 0) coincides with the pivot. Rest pose:
    b center at parent's pivot - child's anchor = (0.5,0,0) - (-0.5,0,0) =
    (1, 0, 0). Rotate 180° about +Z through the pivot (0.5, 0, 0):
    (1, 0, 0) maps to (0, 0, 0).

    A naive implementation that rotates the rest-pose translation about
    parent's origin (rather than the joint pivot) would instead produce
    (-1, 0, 0).
    """
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_revolute(
        "j", a, b, axis=(0.0, 0.0, 1.0),
        limits=(0.0, 360.0), default=180.0,
        parent_anchor=obj.anchor(a, "left"),
        child_anchor=obj.anchor(b, "right"),
    )
    obj.declare_root(a)
    ir = obj.finalize()
    t, _q = query.frame_of(ir, "b")
    assert _approx_vec(t, (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Multi-joint chain composition (catches: parent-frame propagation bug,
# wrong order of parent_anchor / motion / inv_child_anchor).
# ---------------------------------------------------------------------------

def test_chain_of_two_revolute_joints_composes_correctly():
    """Root A. B attached to A by revolute about Z, default 90°, pivot at
    A's `left` anchor = (0.5, 0, 0). B's `right` anchor (= (-0.5, 0, 0))
    aligns to the pivot, so B's center at rest = (0.5, 0, 0) - (-0.5, 0, 0)
    = (1, 0, 0). After 90° rotation about Z through (0.5, 0, 0): B's
    center rotates to (0.5, 0.5, 0), oriented R(90, Z).

    C attached to B by fixed joint at B's `left` (= (0.5, 0, 0) in B-local).
    C's `right` (= (-0.5, 0, 0)) aligns to that, so C's center in B's
    frame at rest = (1, 0, 0). Composing through B's pose: C's world
    position = B's center + R(90, Z) @ (1, 0, 0) = (0.5, 0.5, 0) +
    (0, 1, 0) = (0.5, 1.5, 0).

    Catches: parent-frame propagation bugs, wrong order of
    parent_anchor / motion / inv_child_anchor, and offset-folded-into-
    parent_anchor (which would put both pivots at parent origin and
    yield (0, 2, 0) instead).
    """
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    c = obj.make_box("c", 1, 1, 1)
    obj.attach_revolute(
        "ab", a, b,
        axis=(0.0, 0.0, 1.0), limits=(0.0, 180.0), default=90.0,
        parent_anchor=obj.anchor(a, "left"),
        child_anchor=obj.anchor(b, "right"),
    )
    obj.attach_fixed(
        "bc", b, c,
        parent_anchor=obj.anchor(b, "left"),
        child_anchor=obj.anchor(c, "right"),
    )
    obj.declare_root(a)
    ir = obj.finalize()
    tc, _qc = query.frame_of(ir, "c")
    assert _approx_vec(tc, (0.5, 1.5, 0.0))


# ---------------------------------------------------------------------------
# Prismatic FK along a non-axis-aligned direction (catches: applying
# motion in joint frame after parent_anchor rotation but forgetting to
# express the displacement correctly).
# ---------------------------------------------------------------------------

def test_prismatic_translates_along_parent_local_axis():
    """Axis (1, 1, 0) (not unit-norm; SDK normalizes). default = sqrt(2).
    Child should translate by (1, 1, 0) in parent frame."""
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_prismatic('j', a, b, axis=(1.0, 1.0, 0.0), limits=(0.0, 5.0), default=math.sqrt(2.0), parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    t, _q = query.frame_of(ir, "b")
    assert _approx_vec(t, (1.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# global_bbox transforms all 8 corners, not just the translation
# (catches: forgot to apply rotation when projecting corners to world).
# ---------------------------------------------------------------------------

def test_global_bbox_dilates_under_45deg_rotation():
    """Unit cube rotated 45° about Z (via parent_anchor rotation in a
    fixed joint to the root) should have an AABB extent of sqrt(2)/2 in
    x and y, and still 0.5 in z. A naive implementation that only used
    the part's local AABB plus its translation would still give
    extent 0.5 in xy."""
    obj = Object("o")
    root = obj.make_box("root", 1e-6, 1e-6, 1e-6)  # negligible
    body = obj.make_box("body", 1.0, 1.0, 1.0)
    obj.attach_fixed('j', root, body, parent_anchor=obj.anchor(root, "center", rotation=(0.0, 0.0, 45.00000000000001)), child_anchor=obj.anchor(body, "center"))
    obj.declare_root(root)
    ir = obj.finalize()
    (xmn, ymn, zmn), (xmx, ymx, zmx) = query.global_bbox(ir, "body")
    half = math.sqrt(2.0) / 2.0
    assert _approx(xmn, -half) and _approx(xmx, half)
    assert _approx(ymn, -half) and _approx(ymx, half)
    assert _approx(zmn, -0.5) and _approx(zmx, 0.5)


# ---------------------------------------------------------------------------
# Penetration depth = min of per-axis overlaps (catches: returned a
# different axis's overlap, e.g., max instead of min, or always x).
# ---------------------------------------------------------------------------

def test_penetration_depth_is_min_axis_overlap():
    """Two unit cubes overlapping by 0.1 in x, 0.3 in y, 0.7 in z. Depth
    should be 0.1. SAT degenerates to AABB for axis-aligned OBBs."""
    obj = Object("o")
    a = obj.make_box("a", 1.0, 1.0, 1.0)
    b = obj.make_box("b", 1.0, 1.0, 1.0)
    # a is [-0.5, 0.5]^3. Place b at (0.9, 0.7, 0.3) so overlaps are
    # x: 1.0 - 0.9 = 0.1; y: 1.0 - 0.7 = 0.3; z: 1.0 - 0.3 = 0.7.
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center", translation=(0.9, 0.7, 0.3)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    pairs = query.penetration_pairs(ir)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.a, p.b) == ("a", "b")
    assert _approx(p.depth, 0.1)
    # B's center is at (+0.9, +0.7, +0.3) relative to A. MTV picks +x
    # (smallest overlap). Sign points from a toward b (positive d).
    assert _approx_vec(p.contact_normal, (1.0, 0.0, 0.0), abs_tol=1e-9)
    assert _approx(p.signed_distance, -0.1)
    # Contact point lies inside both AABBs.
    cp = p.contact_point
    (axmn, aymn, azmn), (axmx, aymx, azmx) = p.a_bbox
    (bxmn, bymn, bzmn), (bxmx, bymx, bzmx) = p.b_bbox
    assert axmn - 1e-9 <= cp[0] <= axmx + 1e-9
    assert bxmn - 1e-9 <= cp[0] <= bxmx + 1e-9


def test_penetration_clear_at_face_flush():
    """Faces flush at x=0.5 (overlap 0 along x) should not register as
    penetration (eps strictly positive)."""
    obj = Object("o")
    a = obj.make_box("a", 1.0, 1.0, 1.0)
    b = obj.make_box("b", 1.0, 1.0, 1.0)
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center", translation=(1.0, 0.0, 0.0)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    assert query.penetration_pairs(ir) == []


# ---------------------------------------------------------------------------
# SAT-specific tests (rotated OBB cases).
# ---------------------------------------------------------------------------

def test_sat_rotated_intersecting_object_layer():
    """Object layer, rotated + intersecting: a 45°-rotated thin slab with
    a thick cube embedded near its long axis. Loose world AABB definitely
    overlaps; SAT must agree.

    Topology: root -> a (rotated 45° about z) -> b (offset along the
    rotated frame).
    """
    obj = Object("o")
    root = obj.make_box("root", 0.01, 0.01, 0.01)
    a = obj.make_box("a", 0.50, 0.50, 0.10)
    b = obj.make_box("b", 0.30, 0.30, 0.30)
    obj.attach_fixed('ja', root, a, parent_anchor=obj.anchor(root, "center", rotation=(0.0, 0.0, 45.00000000000001)), child_anchor=obj.anchor(a, "center"))
    obj.attach_fixed('jb', a, b, parent_anchor=obj.anchor(a, "center", translation=(0.2, 0.2, 0.0), rotation=(0.0, 0.0, -45.00000000000001)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(root)
    ir = obj.finalize()
    pairs = query.penetration_pairs(ir)
    pair_names = [(p.a, p.b) for p in pairs]
    # Expect the a-b pair to collide. Root is tiny and well outside.
    assert ("a", "b") in pair_names
    ab = next(p for p in pairs if (p.a, p.b) == ("a", "b"))
    assert ab.depth > 1e-3
    # Contact normal is unit-length.
    cn = ab.contact_normal
    assert _approx(cn[0] * cn[0] + cn[1] * cn[1] + cn[2] * cn[2], 1.0,
                   abs_tol=1e-9)
    # Contact point lies inside both world AABBs.
    cp = ab.contact_point
    (axmn, aymn, azmn), (axmx, aymx, azmx) = ab.a_bbox
    (bxmn, bymn, bzmn), (bxmx, bymx, bzmx) = ab.b_bbox
    assert axmn - 1e-6 <= cp[0] <= axmx + 1e-6
    assert aymn - 1e-6 <= cp[1] <= aymx + 1e-6
    assert bxmn - 1e-6 <= cp[0] <= bxmx + 1e-6
    assert bymn - 1e-6 <= cp[1] <= bymx + 1e-6


def test_sat_rotated_non_intersecting_object_layer_regression():
    """Object layer, rotated + non-intersecting: a door and its door shelf.
    Loose world AABB overlaps; SAT must report no collision."""
    obj = Object("fridge")
    root = obj.make_box("root", 0.01, 0.01, 0.01)
    door = obj.make_box("door", 0.43, 0.04, 1.22)
    shelf = obj.make_box("shelf", 0.32, 0.12, 0.03)
    # Door hinged to root: revolute axis -z, default 120 deg.
    obj.attach_revolute('door_hinge', root, door, axis=(0.0, 0.0, -1.0), limits=(0.0, 150.0), default=120.0, parent_anchor=obj.anchor(root, "center", translation=(0.010000000000000009, 0.355, 1.16)), child_anchor=obj.anchor(door, "center"))
    # Shelf fixed-attached to door, inner-face placement.
    obj.attach_fixed('shelf_to_door', door, shelf, parent_anchor=obj.anchor(door, "center", translation=(0.0, -0.14, -0.25)), child_anchor=obj.anchor(shelf, "center"))
    obj.declare_root(root)
    ir = obj.finalize()
    pairs = query.penetration_pairs(ir)
    pair_names = {(p.a, p.b) for p in pairs}
    # Sanity-check that loose AABB *would* have flagged this — door's
    # rotated world bbox envelops the shelf.
    door_aabb = query.global_bbox(ir, "door")
    shelf_aabb = query.global_bbox(ir, "shelf")
    overlaps = all(
        min(door_aabb[1][i], shelf_aabb[1][i])
        - max(door_aabb[0][i], shelf_aabb[0][i]) > 1e-6
        for i in range(3)
    )
    assert overlaps, "broad-phase AABB should overlap (regression precondition)"
    # The actual SAT verdict: no collision.
    assert ("door", "shelf") not in pair_names


def test_sat_rotated_aabb_only_negative():
    """Two boxes whose loose world AABBs overlap but whose rotated true
    OBBs are separate. Confirms SAT rejects the false positive."""
    obj = Object("o")
    root = obj.make_box("root", 0.01, 0.01, 0.01)
    rod = obj.make_box("rod", 1.0, 0.10, 0.10)
    blk = obj.make_box("blk", 0.30, 0.30, 0.30)
    # Rotate rod 45° about z; centered at origin. Its AABB inflates to
    # ~0.78 × 0.78 × 0.10 in world frame.
    obj.attach_fixed('j_rod', root, rod, parent_anchor=obj.anchor(root, "center", rotation=(0.0, 0.0, 45.00000000000001)), child_anchor=obj.anchor(rod, "center"))
    # Rod world AABB inflates to x ∈ [-0.389, 0.389], y ∈ [-0.389, 0.389]
    # (loose enclosure of the rotated long rod). Place blk at (0.50, 0, 0)
    # so its AABB x ∈ [0.35, 0.65] just overlaps the rod's loose AABB
    # (overlap ≈ 0.039 m). The rod's true OBB along its local y axis
    # separates the pair: projecting onto rod-local y = (-0.707, 0.707, 0),
    # the centers project ~0.35 m apart while the half-projections sum to
    # ~0.26 m — about 0.09 m of clearance.
    obj.attach_fixed('j_blk', root, blk, parent_anchor=obj.anchor(root, "center", translation=(0.5, 0.0, 0.0)), child_anchor=obj.anchor(blk, "center"))
    obj.declare_root(root)
    ir = obj.finalize()
    rod_aabb = query.global_bbox(ir, "rod")
    blk_aabb = query.global_bbox(ir, "blk")
    overlaps = all(
        min(rod_aabb[1][i], blk_aabb[1][i])
        - max(rod_aabb[0][i], blk_aabb[0][i]) > 1e-6
        for i in range(3)
    )
    assert overlaps, "loose AABBs must overlap (test precondition)"
    pairs = query.penetration_pairs(ir)
    pair_names = {(p.a, p.b) for p in pairs}
    assert ("blk", "rod") not in pair_names


def test_sat_sign_convention_lexicographic_order():
    """Two cubes overlapping in +x: contact_normal points from a to b
    (i.e., from the lex-first part toward the lex-second part)."""
    obj = Object("o")
    a = obj.make_box("a", 1.0, 1.0, 1.0)
    b = obj.make_box("b", 1.0, 1.0, 1.0)
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center", translation=(0.9, 0.0, 0.0)), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    ir = obj.finalize()
    pairs = query.penetration_pairs(ir)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.a, p.b) == ("a", "b")
    # b at +x from a → contact_normal ≈ (+1, 0, 0).
    assert _approx_vec(p.contact_normal, (1.0, 0.0, 0.0), abs_tol=1e-9)


def test_sat_nearly_parallel_face_axes_no_chatter():
    """Near-zero rotation should produce the same result as no rotation:
    cross-product axes are degenerate and must be skipped without
    affecting the MTV pick."""
    obj1 = Object("o1")
    a1 = obj1.make_box("a", 1.0, 1.0, 1.0)
    b1 = obj1.make_box("b", 1.0, 1.0, 1.0)
    obj1.attach_fixed('j', a1, b1, parent_anchor=obj1.anchor(a1, "center", translation=(0.8, 0.0, 0.0)), child_anchor=obj1.anchor(b1, "center"))
    obj1.declare_root(a1)
    ir1 = obj1.finalize()
    p1 = query.penetration_pairs(ir1)[0]

    _reset_object_singleton()

    obj2 = Object("o2")
    root2 = obj2.make_box("root", 0.01, 0.01, 0.01)
    a2 = obj2.make_box("a", 1.0, 1.0, 1.0)
    b2 = obj2.make_box("b", 1.0, 1.0, 1.0)
    # 1e-5 deg ≈ 1.7e-7 rad — well below EPS_AXIS (1e-6) when squared.
    obj2.attach_fixed('ja', root2, a2, parent_anchor=obj2.anchor(root2, "center", rotation=(0.0, 0.0, 1e-05)), child_anchor=obj2.anchor(a2, "center"))
    obj2.attach_fixed('jb', a2, b2, parent_anchor=obj2.anchor(a2, "center", translation=(0.8, 0.0, 0.0)), child_anchor=obj2.anchor(b2, "center"))
    obj2.declare_root(root2)
    ir2 = obj2.finalize()
    p2 = next(p for p in query.penetration_pairs(ir2) if (p.a, p.b) == ("a", "b"))
    assert _approx(p1.depth, p2.depth, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Structural check: a disconnected subtree (not just an orphan part)
# (catches: BFS that incorrectly walks both directions, or a check that
# only counts parts vs. joints rather than reachability).
# ---------------------------------------------------------------------------

def test_disconnected_subtree_is_detected():
    """A — B is one tree; C — D is a disjoint tree. Root is A. C and D
    must both be flagged."""
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    c = obj.make_box("c", 1, 1, 1)
    d = obj.make_box("d", 1, 1, 1)
    obj.attach_fixed('ab', a, b, parent_anchor=obj.anchor(a, "center", translation=(1.0, 0.0, 0.0)), child_anchor=obj.anchor(b, "center"))
    obj.attach_fixed('cd', c, d, parent_anchor=obj.anchor(c, "center", translation=(1.0, 0.0, 0.0)), child_anchor=obj.anchor(d, "center"))
    obj.declare_root(a)
    with pytest.raises(NotConnectedToRoot) as ei:
        obj.finalize()
    msg = str(ei.value)
    assert "c" in msg and "d" in msg


# ---------------------------------------------------------------------------
# Real structural error cases that are easy to write wrong in the code.
# ---------------------------------------------------------------------------

def test_multi_parent_is_rejected():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    c = obj.make_box("c", 1, 1, 1)
    obj.attach_fixed('j1', a, b, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    with pytest.raises(InvalidJoint):
        obj.attach_fixed('j2', c, b, parent_anchor=obj.anchor(c, "center"), child_anchor=obj.anchor(b, "center"))


def test_root_cannot_be_made_child_after_declaration():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.declare_root(a)
    with pytest.raises(InvalidJoint):
        obj.attach_fixed('j', b, a, parent_anchor=obj.anchor(b, "center"), child_anchor=obj.anchor(a, "center"))


def test_declare_root_after_part_has_incoming_joint_fails():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    obj.attach_fixed('j', a, b, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    with pytest.raises(InvalidJoint):
        obj.declare_root(b)


def test_part_and_joint_names_live_in_separate_namespaces():
    obj = Object("o")
    a = obj.make_box("hinge", 1, 1, 1)
    b = obj.make_box("body", 1, 1, 1)
    # Joint named "hinge" must not collide with the part named "hinge".
    obj.attach_fixed('hinge', a, b, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    obj.declare_root(a)
    obj.finalize()  # no error


# ---------------------------------------------------------------------------
# Cross-object PartRef detection (catches: forgot to validate ref owner,
# accidentally compared by id alone).
# ---------------------------------------------------------------------------

def test_cross_object_partref_is_rejected():
    obj1 = Object("o1")
    a1 = obj1.make_box("a", 1, 1, 1)
    _reset_object_singleton()
    obj2 = Object("o2")
    b2 = obj2.make_box("b", 1, 1, 1)
    with pytest.raises(DanglingReference):
        obj2.attach_fixed('j', a1, b2, parent_anchor=obj2.anchor(a1, "center"), child_anchor=obj2.anchor(b2, "center"))


# ---------------------------------------------------------------------------
# Singleton + finalization lifecycle (the runtime contract the agent
# layer will depend on; cheap to test, easy to regress).
# ---------------------------------------------------------------------------

def test_only_one_object_per_sandbox():
    Object("a")
    with pytest.raises(BuildError):
        Object("b")


def test_finalize_is_idempotent_and_seals_the_builder():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    obj.declare_root(a)
    ir1 = obj.finalize()
    ir2 = obj.finalize()
    assert ir1 is ir2
    with pytest.raises(BuildError):
        obj.make_box("b", 1, 1, 1)


# ---------------------------------------------------------------------------
# Structured exception payloads (design_object_agent.md "Structured exceptions").
# Each subclass must identify the affected object(s) via refs and carry
# subclass-appropriate evidence.
# ---------------------------------------------------------------------------

def test_InvalidJoint_for_axis_zero_carries_full_object_ref_chain():
    obj = Object("o")
    a = obj.make_box("a", 1, 1, 1)
    b = obj.make_box("b", 1, 1, 1)
    with pytest.raises(InvalidJoint) as ei:
        obj.attach_revolute('j', a, b, axis=(0.0, 0.0, 0.0), limits=(-10.0, 10.0), default=0.0, parent_anchor=obj.anchor(a, "center"), child_anchor=obj.anchor(b, "center"))
    d = ei.value.to_dict()
    assert d["code"] == "InvalidJoint"
    kinds = {r["kind"]: r["name"] for r in d["refs"]}
    # The chain must locate object, joint, parent part, and child part.
    assert kinds == {"object": "o", "joint": "j", "part": "b"} or kinds.get("joint") == "j"
    # Evidence carries the actual axis the user passed.
    assert d["evidence"]["reason"] == "axis_zero"
    assert d["evidence"]["value"] == [0.0, 0.0, 0.0]


def test_NotConnectedToRoot_lists_every_unreachable_part_by_name():
    obj = Object("o")
    root = obj.make_box("root", 1, 1, 1)
    lonely1 = obj.make_box("lonely1", 1, 1, 1)
    lonely2 = obj.make_box("lonely2", 1, 1, 1)
    # Attach lonely2 to lonely1, but neither is connected to root.
    obj.attach_fixed('j', lonely1, lonely2, parent_anchor=obj.anchor(lonely1, "center"), child_anchor=obj.anchor(lonely2, "center"))
    obj.declare_root(root)
    with pytest.raises(NotConnectedToRoot) as ei:
        obj.finalize()
    d = ei.value.to_dict()
    assert d["code"] == "NotConnectedToRoot"
    assert d["evidence"]["root"] == "root"
    assert set(d["evidence"]["unreachable"]) == {"lonely1", "lonely2"}
    # refs must list both unreachable parts.
    part_names = {r["name"] for r in d["refs"] if r["kind"] == "part"}
    assert {"lonely1", "lonely2"} <= part_names


def test_DanglingReference_when_partref_owned_by_other_object():
    obj_a = Object("a")
    p_a = obj_a.make_box("p", 1, 1, 1)
    _reset_object_singleton()
    obj_b = Object("b")
    p_b = obj_b.make_box("p", 1, 1, 1)
    with pytest.raises(DanglingReference) as ei:
        # Use obj_a's part as parent inside obj_b's builder.
        obj_b.attach_fixed('j', p_a, p_b, parent_anchor=obj_b.anchor(p_a, "center"), child_anchor=obj_b.anchor(p_b, "center"))
    d = ei.value.to_dict()
    assert d["code"] == "DanglingReference"
    assert d["evidence"]["owner_object_name"] == "b"
    assert any(r["kind"] == "object" and r["name"] == "b"
               for r in d["refs"])


# ---------------------------------------------------------------------------
# Kernel tolerates IRs without a `root_pose` field. finalize() does not write
# one, and hand-authored IRs that omit the key must still resolve FK: the seed
# falls back to identity rather than KeyError-ing.
# ---------------------------------------------------------------------------


def test_kernel_loads_ir_without_root_pose():
    _reset_object_singleton()
    o = Object("cab")
    carcass = o.make_box("carcass", 0.6, 0.6, 0.40)
    toekick = o.make_box("toekick", 0.5, 0.5, 0.10)
    o.attach_fixed("j", carcass, toekick,
                   parent_anchor=o.anchor(carcass, "bottom"),
                   child_anchor=o.anchor(toekick, "top"))
    o.declare_root(carcass)
    ir = o.finalize()
    # Strip the key entirely -- kernel must still resolve FK
    # with the root seeded at identity, so the toekick hangs below the root.
    ir.pop("root_pose", None)
    (_, _, zmn), (_, _, zmx) = query.global_bbox(ir)
    # carcass centered at origin (±0.20); toekick hangs 0.10 below it.
    assert _approx(zmn, -0.30)
    assert _approx(zmx, 0.20)
