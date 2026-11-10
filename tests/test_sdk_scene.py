"""Tests for dycraft.sdk scene layer (v1).

Each test targets behavior with hidden runtime complexity that a careful
static read of the code would not predict: forward-kinematics math
through anchor chains, group root-shadow semantics, attach rotation
through the parent anchor frame, part-scope vs. union AABB resolution,
finite-vs-inf serialization in canonical JSON, hash invariants under
non-normalized inputs, recursion termination when a group's anchor is
queried during its own world-pose resolution, and the auto-root pick.

Tautological tests (name regex, "if X then raise X" structural rules
visible in one line of source, dict-shape inspections) are intentionally
omitted.

Run: `conda run -n chord pytest dycraft/tests/test_sdk_scene.py -v`
"""

import math

import pytest

from dycraft.sdk import (
    Object,
    Scene,
    ObjectHandle,
    InvalidGroup,
    scene_query,
)
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


def _make_box_object(obj_name: str, part_name: str = "body",
                     w: float = 1.0, l: float = 1.0, h: float = 1.0) -> dict:
    _reset_object_singleton()
    o = Object(obj_name)
    p = o.make_box(part_name, w, l, h)
    o.declare_root(p)
    return o.finalize()


def _make_two_part_object(obj_name: str = "chair") -> dict:
    """Chair-like object with seat (root) + back attached via a fixed
    joint at the seat's +Y edge so back's center sits at (0, -0.25,
    0.275) in object-root frame."""
    _reset_object_singleton()
    o = Object(obj_name)
    seat = o.make_box("seat", 0.5, 0.5, 0.05)
    back = o.make_box("back", 0.5, 0.05, 0.5)
    o.attach_fixed('sb', seat, back, parent_anchor=o.anchor(seat, "center", translation=(0.0, -0.25, 0.275)), child_anchor=o.anchor(back, "center"))
    o.declare_root(seat)
    return o.finalize()


def _approx(a, b, *, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=abs_tol)


def _approx_vec(a, b, *, abs_tol=1e-9):
    return all(_approx(x, y, abs_tol=abs_tol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Determinism: hash stability and what perturbs it.
# Catches: float repr drift, key-order leakage into the canonical form.
# ---------------------------------------------------------------------------


def test_scene_hash_is_stable_and_responds_to_pose_change():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}

    def build(z):
        _reset_scene_singleton()
        s = Scene("s")
        s.set_room(2.0, 2.0, 2.0)
        s.place(objects["box"], name="b1",
                pose=((0.0, 0.0, z), (0.0, 0.0, 0.0)))
        return s.finalize()

    ir_a = build(0.5)
    ir_b = build(0.5)
    ir_c = build(0.7)
    assert ir_a["id"] == ir_b["id"]
    assert ir_a["id"] != ir_c["id"]


# ---------------------------------------------------------------------------
# FK with offset rotation. Tests that the offset's translation is
# interpreted in the parent_anchor frame (NOT in the post-rotation child
# frame), and that the child's body extends from the anchor in its
# post-rotation local +Y/-Y direction — the joint anchor stays where the
# offset translation says it does, but the child's centerline is rotated
# 180 deg around it.
#
# Failure modes this catches:
#   - applying offset.translation in the rotated frame (would put the
#     child at (0, 0.95, 0) instead of (0, 1.05, 0));
#   - composing parent_anchor / offset / inverse(child_anchor) in the
#     wrong order;
#   - forgetting to rotate (-ca_pos) when computing the child origin
#     from the post-offset anchor.
# ---------------------------------------------------------------------------


def test_attach_with_180_yaw_offset_flips_orientation_and_extends_correctly():
    box_ir = _make_box_object("box", w=1.0, l=1.0, h=1.0)
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    # Default (bottom-anchor) placement lands the box's bbox bottom at pose z.
    # The box is 1 m tall and centered on its root origin, so z=-0.5 puts the
    # origin back at (0,0,0) — the origin-at-pose geometry this FK check was
    # written against.
    parent = s.place(objects["box"], name="p",
                     pose=((0.0, 0.0, -0.5), (0.0, 0.0, 0.0)))
    s.attach(objects["box"], name="c",
             parent=parent,
             parent_anchor=s.anchor(parent, "front", translation=(0.0, 0.05, 0.0)),
             child_anchor=s.anchor(objects["box"], "front", rotation=(0.0, 0.0, 180.0)))
    ir = s.finalize()
    t, (_rx, _ry, rz) = scene_query.instance_pose(ir, catalog, "c")
    assert _approx_vec(t, (0.0, 1.05, 0.0), abs_tol=1e-6)
    # Yaw is +/-180 (both valid Euler recoveries of the same rotation).
    assert _approx(abs(rz), 180.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Part-scoped anchor. The two parts have very different z-extents even
# though they share an object root frame: seat top z = 0.025, back top
# z = 0.525 (back's local center is offset +0.275 in z via the joint).
# A bug that used the union AABB for part-scoped names would conflate
# the two.
# ---------------------------------------------------------------------------


def test_part_scoped_anchor_uses_part_aabb_not_object_union_aabb():
    chair_ir = _make_two_part_object("chair")
    objects = {"chair": ObjectHandle(chair_ir)}
    catalog = {chair_ir["id"]: chair_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    # Default (bottom-anchor) placement lands the object's union-bbox bottom
    # (the seat bottom, 0.025 below the seat-center origin) at pose z, so
    # z=-0.025 puts the origin back at (0,0,0) — the origin-at-pose geometry
    # the expected part-anchor z's were written against.
    s.place(objects["chair"], name="ch",
            pose=((0.0, 0.0, -0.025), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    seat_top = scene_query.anchor_position(ir, catalog, "ch", "seat.top")
    back_top = scene_query.anchor_position(ir, catalog, "ch", "back.top")
    assert _approx(seat_top[2], 0.025, abs_tol=1e-6)
    assert _approx(back_top[2], 0.525, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# attach_group overrides the root member's room placement. The FK
# computation must consult the group's authoritative attach edge, not
# the root's stale `place()` pose.
#
# Algebraic check: chair (0.4 x 0.4 x 0.5) placed initially at (0,0,0),
# then grouped and attached so the group's `front` (= chair's
# front = (0, 0.2, 0)) coincides with the table's `back`. The 0.05-tall
# tabletop is placed by the default bottom anchor at z=0.975, so its center
# (= origin) lands at z=1.0 and its `back` world = (0, -0.5, 1).
# After attach, chair center = (0, -0.5 - 0.2, 1) = (0, -0.7, 1).
# If the bug is "use the chair's place() pose," the chair stays at
# (0, 0, 0) and this test fails.
# ---------------------------------------------------------------------------


def test_attach_group_overrides_root_member_room_pose():
    chair_ir = _make_box_object("chair", w=0.4, l=0.4, h=0.5)
    table_ir = _make_box_object("table", w=2.0, l=1.0, h=0.05)
    objects = {"chair": ObjectHandle(chair_ir),
               "table": ObjectHandle(table_ir)}
    catalog = {chair_ir["id"]: chair_ir, table_ir["id"]: table_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    table = s.place(objects["table"], name="t",
                    pose=((0.0, 0.0, 0.975), (0.0, 0.0, 0.0)))
    c1 = s.place(objects["chair"], name="c1",
                 pose=((9.9, 9.9, 9.9), (0.0, 0.0, 0.0)))  # nonsense pose
    grp = s.create_group(name="g", members=[c1], root=c1)
    s.attach_group(grp, parent=table,
                   parent_anchor=s.anchor(table, "back"),
                   child_anchor=s.anchor(grp, "front"))
    ir = s.finalize()
    t, _euler = scene_query.instance_pose(ir, catalog, "c1")
    assert _approx_vec(t, (0.0, -0.7, 1.0), abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Recursion termination: resolving a group's world pose requires its
# union AABB (for the group-side anchor in local frame), and computing
# that AABB requires member poses. A naive implementation routes member
# poses through world coords → calls back into the group's own world
# pose → infinite loop.
#
# The loop only shows at *runtime* when the group has been attached (so
# the root member's placement is null/shadowed). A scene where the group
# is unattached or the root is placed directly would not exercise it.
# ---------------------------------------------------------------------------


def test_group_world_pose_resolves_without_recursion_when_attached():
    box_ir = _make_box_object("box", w=1.0, l=1.0, h=1.0)
    parent_ir = _make_box_object("parent", w=4.0, l=4.0, h=0.1)
    objects = {"box": ObjectHandle(box_ir),
               "parent": ObjectHandle(parent_ir)}
    catalog = {box_ir["id"]: box_ir, parent_ir["id"]: parent_ir}

    s = Scene("s")
    s.set_room(8.0, 8.0, 4.0)
    par = s.place(objects["parent"], name="par",
                  pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))
    b1 = s.place(objects["box"], name="b1",
                 pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    b2 = s.attach(objects["box"], name="b2", parent=b1,
                  parent_anchor=s.anchor(b1, "left", translation=(0.5, 0.0, 0.0)),
                  child_anchor=s.anchor(objects["box"], "right"))
    g = s.create_group(name="g", members=[b1, b2], root=b1)
    s.attach_group(g, parent=par,
                   parent_anchor=s.anchor(par, "top"),
                   child_anchor=s.anchor(g, "bottom"))
    ir = s.finalize()

    # A recursing implementation overflows the stack. The numerical asserts
    # also check the group-frame FK:
    #   group root pose = group pose (group's frame = root's frame).
    #   b2 displaced from b1 by 1.5 m in x (0.5 left + 0.5 offset
    #   - (-0.5) right = 1.5).
    p1 = scene_query.instance_pose(ir, catalog, "b1")
    p2 = scene_query.instance_pose(ir, catalog, "b2")
    pg = scene_query.group_pose(ir, catalog, "g")
    assert _approx_vec(p1[0], pg[0], abs_tol=1e-6)
    assert _approx(p2[0][0] - p1[0][0], 1.5, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Infinity in assertion args. canonical JSON has allow_nan=False so bare
# inf tokens would crash finalize. The SDK must round-trip via the
# strings "inf" / "-inf". The test exercises the path by attempting to
# finalize an IR that uses float('inf') and reading back the args.
# ---------------------------------------------------------------------------


def test_inf_in_assertion_args_round_trips_through_canonical_json():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}

    s = Scene("s")
    s.set_room(2.0, 2.0, 2.0)
    a = s.place(objects["box"], name="a",
                pose=((-0.6, 0.0, 0.5), (0.0, 0.0, 0.0)))
    b = s.place(objects["box"], name="b",
                pose=(( 0.6, 0.0, 0.5), (0.0, 0.0, 0.0)))
    s.assert_distance(
        a=s.anchor(a, "center"), b=s.anchor(b, "center"),
        axis=(1.0, 0.0, 0.0), min=0.0, max=float("inf"),
        name="x_separated",
    )
    # Finalize must succeed (canonical JSON would otherwise reject inf).
    ir = s.finalize()
    args = ir["assertions"][0]["args"]
    assert args["max"] == "inf"


# ---------------------------------------------------------------------------
# Cycle prevention on attach_group. The descendant set is computed by
# walking the parent_of map; the cycle check then walks from `parent`
# upward and checks for membership. A bug that compares only against
# `group._name` would miss the case where the proposed parent is a
# descendant of the group (here: `outer` contains `inner` which contains
# `b`, so attaching `outer` to `b` is a cycle).
# ---------------------------------------------------------------------------


def test_attach_group_rejects_cycle_to_descendant():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    a = s.place(objects["box"], name="a",
                pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    b = s.attach(objects["box"], name="b", parent=a,
                 parent_anchor=s.anchor(a, "left"),
                 child_anchor=s.anchor(objects["box"], "right"))
    inner = s.create_group(name="inner", members=[a, b], root=a)
    c = s.attach(objects["box"], name="c", parent=inner,
                 parent_anchor=s.anchor(inner, "top"),
                 child_anchor=s.anchor(objects["box"], "bottom"))
    outer = s.create_group(name="outer", members=[inner, c], root=inner)
    with pytest.raises(InvalidGroup):
        s.attach_group(outer, parent=b,
                       parent_anchor=s.anchor(b, "left"),
                       child_anchor=s.anchor(outer, "right"))


# ---------------------------------------------------------------------------
# create_group: connectivity walks parent pointers WITHIN the proposed
# member set. Two room-placed instances (a, b) with no attach edge form
# two separate trees, so [a, b] cannot form a single group rooted at a.
# A bug that just counts members vs. root presence would let this slip
# through.
# ---------------------------------------------------------------------------


def test_create_group_rejects_disconnected_members():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}
    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    a = s.place(objects["box"], name="a",
                pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    b = s.place(objects["box"], name="b",
                pose=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(InvalidGroup):
        s.create_group(name="g", members=[a, b], root=a)


# ---------------------------------------------------------------------------
# create_group auto-root: omit root and let the SDK pick the unique
# externally-attached member. The runtime behavior is "scan parent_of
# for each member; the unique one whose parent is outside members
# becomes root." A bug that always picks members[0] would miss this.
# ---------------------------------------------------------------------------


def test_create_group_auto_picks_unique_external_member_as_root():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    # Put the externally-anchored member last in the members list to
    # ensure members[0] is NOT the right answer.
    root_member = s.place(objects["box"], name="root_m",
                          pose=((1.0, 2.0, 0.0), (0.0, 0.0, 0.0)))
    child_member = s.attach(objects["box"], name="child_m",
                            parent=root_member,
                            parent_anchor=s.anchor(root_member, "left"),
                            child_anchor=s.anchor(objects["box"], "right"))
    grp = s.create_group(name="g",
                         members=[child_member, root_member])  # no root=
    ir = s.finalize()
    group_record = next(g for g in ir["groups"] if g["name"] == "g")
    assert group_record["root"] == "root_m"
    # And the group pose must coincide with the root member's pose.
    pg = scene_query.group_pose(ir, catalog, "g")
    pr = scene_query.instance_pose(ir, catalog, "root_m")
    assert _approx_vec(pg[0], pr[0], abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Cross-instance penetration: AABB overlap pairs at world rest pose.
# depth = min(per-axis overlap). a + b overlap by (0.1, 0.3, 0.7) → depth
# 0.1. c is disjoint and must not appear. A bug that returned max overlap,
# or one that included disjoint pairs, fails here.
# ---------------------------------------------------------------------------


def test_penetration_depth_is_min_axis_overlap_and_disjoint_pairs_excluded():
    box_ir = _make_box_object("box", w=1.0, l=1.0, h=1.0)
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}

    s = Scene("s")
    s.set_room(8.0, 8.0, 8.0)
    s.place(objects["box"], name="a",
            pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    s.place(objects["box"], name="b",
            pose=((0.9, 0.7, 0.3), (0.0, 0.0, 0.0)))
    s.place(objects["box"], name="c",
            pose=((3.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    pairs = scene_query.penetration_pairs(ir, catalog)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.a, p.b) == ("a", "b")
    assert _approx(p.depth, 0.1, abs_tol=1e-6)
    # SAT degenerates to AABB for axis-aligned OBBs. b's center is at
    # (+0.9, +0.7, +0.3) relative to a; smallest overlap is along +x.
    assert _approx(p.contact_normal[0], 1.0, abs_tol=1e-9)
    assert _approx(p.signed_distance, -0.1, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# SAT-specific scene tests (rotated-pose cases).
# ---------------------------------------------------------------------------


def test_sat_rotated_intersecting_scene_layer():
    """Two instances of a box: one axis-aligned, one rotated 45° about z
    and offset diagonally so their true OBBs intersect. Their world AABBs
    also overlap; SAT must agree with a positive collision."""
    box_ir = _make_box_object("box", w=0.40, l=0.40, h=0.20)
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    s.place(objects["box"], name="a",
            pose=((0.0, 0.0, 0.10), (0.0, 0.0, 0.0)))
    s.place(objects["box"], name="b",
            pose=((0.20, 0.20, 0.10), (0.0, 0.0, 45.0)))
    ir = s.finalize()
    pairs = scene_query.penetration_pairs(ir, catalog)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.a, p.b) == ("a", "b")
    assert p.depth > 1e-3
    cn = p.contact_normal
    # Contact normal is unit-length and lies in the world XY plane (z=0
    # to within numerical tolerance).
    assert _approx(cn[0] * cn[0] + cn[1] * cn[1] + cn[2] * cn[2], 1.0,
                   abs_tol=1e-9)
    assert abs(cn[2]) < 1e-6


def test_sat_rotated_non_intersecting_scene_layer():
    """Two parallel thin rods along world +x. rod_a is axis-aligned;
    rod_b is offset diagonally in (y, z) by (0.10, 0.10) and rotated
    45° about world +x so its square cross-section becomes a diamond.

    rod_b's loose world AABB inflates to ±0.0707 in y and z (compared
    to rod_a's ±0.05), so the world AABBs overlap by ~0.02 m on each
    axis. But the rotated diamond cross-section of rod_b (in the y-z
    plane) does not actually intersect rod_a's square cross-section —
    rod_b's local y axis is a separating axis with ~0.02 m clearance.
    SAT must return [].
    """
    rod_ir = _make_box_object("rod", w=1.0, l=0.10, h=0.10)
    objects = {"rod": ObjectHandle(rod_ir)}
    catalog = {rod_ir["id"]: rod_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    # Place by the rods' center anchor so the pose points are the box
    # centers (rod_b's 45° rotation pivots about its center): rod_a center at
    # the origin, rod_b center offset diagonally by (0.10, 0.10) in (y, z).
    s.place(objects["rod"], name="rod_a",
            pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            anchor=s.anchor(objects["rod"], "center"))
    s.place(objects["rod"], name="rod_b",
            pose=((0.0, 0.10, 0.10), (45.0, 0.0, 0.0)),
            anchor=s.anchor(objects["rod"], "center"))
    ir = s.finalize()
    # Preconditions: loose world AABBs overlap.
    bb_a = scene_query.instance_bbox(ir, catalog, "rod_a")
    bb_b = scene_query.instance_bbox(ir, catalog, "rod_b")
    overlaps = all(
        min(bb_a[1][i], bb_b[1][i]) - max(bb_a[0][i], bb_b[0][i]) > 1e-6
        for i in range(3)
    )
    assert overlaps, "loose AABBs must overlap (test precondition)"
    # SAT verdict: no collision.
    assert scene_query.penetration_pairs(ir, catalog) == []


# ---------------------------------------------------------------------------
# ObjectHandle.part_frame returns the part's translation in object root
# frame. The non-obvious bit is the FK math: the back part's center is
# offset by (0, -0.25, 0.275) from the seat (root) via the fixed joint.
# (The Euler-vs-quat shape is a static property of the function, but the
# translation value here exercises the FK pipeline.)
# ---------------------------------------------------------------------------


def test_object_handle_part_frame_uses_joint_chain_translation():
    chair_ir = _make_two_part_object("chair")
    handle = ObjectHandle(chair_ir)
    t, _euler = handle.part_frame("back")
    assert _approx_vec(t, (0.0, -0.25, 0.275), abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Assertion axis normalization. Two scenes that differ only in the
# *scale* of an assertion's axis vector must produce identical IR hashes
# (because the axis is normalized before storage). A bug that wrote the
# raw axis would split the hash.
# ---------------------------------------------------------------------------


def test_axis_scale_does_not_change_scene_hash():
    box_ir = _make_box_object("box")
    objects = {"box": ObjectHandle(box_ir)}

    def build(axis):
        _reset_scene_singleton()
        s = Scene("s")
        s.set_room(2.0, 2.0, 2.0)
        a = s.place(objects["box"], name="a",
                    pose=((-0.6, 0.0, 0.5), (0.0, 0.0, 0.0)))
        b = s.place(objects["box"], name="b",
                    pose=(( 0.6, 0.0, 0.5), (0.0, 0.0, 0.0)))
        s.assert_distance(
            a=s.anchor(a, "center"), b=s.anchor(b, "center"),
            axis=axis, min=0.0, max=2.0,
            name="x_sep",
        )
        return s.finalize()

    ir1 = build((1.0, 0.0, 0.0))
    ir7 = build((7.3, 0.0, 0.0))
    assert ir1["id"] == ir7["id"]


def test_room_ref_public_dimensions():
    """RoomRef must expose width/depth/height as public properties so
    sandboxed code can introspect the room size."""
    _reset_scene_singleton()
    s = Scene("s")
    room = s.set_room(width=4.0, depth=5.0, height=2.6)
    assert room.name == "room"
    assert room.width == 4.0
    assert room.depth == 5.0
    assert room.height == 2.6


def test_scene_ref_public_attributes_pass_sandbox():
    """The scene-layer refs (RoomRef, InstanceRef, GroupRef, AnchorRef)
    expose only public properties; underscore-prefixed slots remain
    blocked by the AST allowlist.
    """
    from dycraft.agent.cad.sandbox import check_ast, SandboxViolation
    # All-public source should validate cleanly.
    src = (
        "scene = Scene('s')\n"
        "room = scene.set_room(width=4.0, depth=5.0, height=2.6)\n"
        "_ = room.width + room.depth + room.height\n"
        "_ = room.name\n"
        "_ = room.poses\n"
        "i = scene.place(objects['box'], name='a',\n"
        "    pose=((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)))\n"
        "_ = i.name + '_ground'\n"
        "anc = scene.anchor(i, 'top', translation=(0.1, 0.0, 0.0),\n"
        "    rotation=(0.0, 45.0, 0.0))\n"
        "_ = anc.name\n"
        "_ = anc.entity_name\n"
        "_ = anc.entity_kind\n"
        "_ = anc.translation\n"
        "_ = anc.rotation\n"
        "_ = anc.outward().source\n"
        "ir = scene.finalize()\n"
    )
    # No exception means each `.attr` passed the allowlist.
    check_ast(src, mode="exec")
    # And a private-attribute reach must still raise.
    bad = "scene = Scene('s')\nroom = scene.set_room(1.0, 1.0, 1.0)\n_ = room._width\n"
    with pytest.raises(SandboxViolation):
        check_ast(bad, mode="exec")


# ---------------------------------------------------------------------------
# Anchor-based placement. `scene.place` aligns a chosen object anchor to the
# world pose; the anchor defaults to the object's bbox-bottom-center (so the
# object rests on the floor at pose z), and `place` is exactly sugar for an
# attach against the room's floor-center. The object's local origin never
# enters the result.
# ---------------------------------------------------------------------------


def test_place_default_anchor_rests_bbox_bottom_center_at_pose():
    # Root part is offset from the object's bbox bottom (toekick hangs below
    # the root carcass), so "origin at pose" and "bbox-bottom at pose" differ.
    _reset_object_singleton()
    o = Object("cab")
    carcass = o.make_box("carcass", 0.6, 0.6, 0.40)
    toekick = o.make_box("toekick", 0.5, 0.5, 0.10)
    o.attach_fixed("j", carcass, toekick,
                   parent_anchor=o.anchor(carcass, "bottom"),
                   child_anchor=o.anchor(toekick, "top"))
    o.declare_root(carcass)
    obj_ir = o.finalize()
    handle = ObjectHandle(obj_ir)
    catalog = {obj_ir["id"]: obj_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    s.place(handle, name="a", pose=((1.0, 2.0, 0.0), (0.0, 0.0, 0.0)))
    ir = s.finalize()
    (xmn, ymn, zmn), (xmx, ymx, zmx) = scene_query.instance_world_aabb(
        ir, catalog, "a")
    # Default anchor = bbox-bottom-center → it lands exactly at the pose.
    assert _approx(zmn, 0.0, abs_tol=1e-9)
    assert _approx((xmn + xmx) / 2.0, 1.0, abs_tol=1e-9)
    assert _approx((ymn + ymx) / 2.0, 2.0, abs_tol=1e-9)


def test_place_anchor_override_lands_named_anchor_at_pose():
    box_ir = _make_box_object("box", w=1.0, l=1.0, h=1.0)
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}

    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    # Override with the "top" anchor → the box's top face lands at the pose,
    # so the box hangs below z=1.0 (zmax == 1.0).
    s.place(objects["box"], name="a", pose=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            anchor=s.anchor(objects["box"], "top"))
    ir = s.finalize()
    (_, _, zmn), (_, _, zmx) = scene_query.instance_world_aabb(ir, catalog, "a")
    assert _approx(zmx, 1.0, abs_tol=1e-9)
    assert _approx(zmn, 0.0, abs_tol=1e-9)


def test_place_is_sugar_for_attach_against_room_floor_center():
    """place(handle, pose, anchor) ≡ attach against the room "bottom" anchor
    offset by the pose, with the same child anchor. Both must yield identical
    instance world poses and AABBs."""
    box_ir = _make_box_object("box", w=0.8, l=0.6, h=0.5)
    objects = {"box": ObjectHandle(box_ir)}
    catalog = {box_ir["id"]: box_ir}
    t, r = (0.7, -0.4, 0.0), (0.0, 0.0, 30.0)

    _reset_scene_singleton()
    s1 = Scene("s1")
    s1.set_room(4.0, 4.0, 4.0)
    s1.place(objects["box"], name="a", pose=(t, r))
    ir1 = s1.finalize()

    _reset_scene_singleton()
    s2 = Scene("s2")
    s2.set_room(4.0, 4.0, 4.0)
    s2.attach(objects["box"], name="a", parent=s2.room,
              parent_anchor=s2.anchor(s2.room, "bottom",
                                      translation=t, rotation=r),
              child_anchor=s2.anchor(objects["box"], "bottom"))
    ir2 = s2.finalize()

    p1, p2 = (scene_query.instance_pose(ir1, catalog, "a"),
              scene_query.instance_pose(ir2, catalog, "a"))
    assert _approx_vec(p1[0], p2[0], abs_tol=1e-9)
    assert _approx_vec(p1[1], p2[1], abs_tol=1e-9)
    bb1 = scene_query.instance_world_aabb(ir1, catalog, "a")
    bb2 = scene_query.instance_world_aabb(ir2, catalog, "a")
    assert _approx_vec(bb1[0], bb2[0], abs_tol=1e-9)
    assert _approx_vec(bb1[1], bb2[1], abs_tol=1e-9)


# ---------------------------------------------------------------------------
# World boxes are tight: one axis-alignment, not two
# ---------------------------------------------------------------------------


def _rotated_part_scene(instance_deg=-45.0):
    """A rod turned 45 deg *inside* its object, in an instance turned in the
    scene. Both rotations are needed: either alone axis-aligns once."""
    _reset_object_singleton()
    o = Object("diag_rod")
    hub = o.make_box("hub", 0.10, 0.10, 0.10)
    rod = o.make_box("rod", 1.00, 0.10, 0.10)
    o.attach_fixed("j", parent=hub, child=rod,
                   parent_anchor=o.anchor(hub, "center"),
                   child_anchor=o.anchor(rod, "center",
                                         rotation=(0.0, 0.0, 45.0)))
    o.declare_root(hub)
    obj_ir = o.finalize()

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(8.0, 8.0, 8.0)
    s.place(ObjectHandle(obj_ir), name="r",
            pose=((0.0, 0.0, 0.05), (0.0, 0.0, instance_deg)))
    return s.finalize(), {obj_ir["id"]: obj_ir}


def test_a_parts_world_box_is_tight_to_its_own_extent():
    """A part's world box is its own extent, not a box of a box.

    Taking the part's AABB in the *object root* frame (already axis-aligned
    there), then rotating that box's corners by the instance pose and
    re-aligning, pays the rotation inflation twice when a part is turned inside
    an object that is turned in the scene.

    Here the object's -45 deg placement exactly undoes the rod's +45 deg mount,
    so the rod ends up axis-aligned in the world and its true box is its own
    0.10 x 1.00 x 0.10 extent; the box of a box would be 1.10 x 1.10 x 0.10.
    """
    ir, catalog = _rotated_part_scene()
    boxes = dict(scene_query.instance_part_world_aabbs(ir, catalog, "r"))
    (lo, hi) = boxes["rod"]
    assert _approx_vec(lo, (-0.05, -0.5, 0.05), abs_tol=1e-9)
    assert _approx_vec(hi, (0.05, 0.5, 0.15), abs_tol=1e-9)


def test_a_parts_world_box_bounds_the_same_body_as_its_obb():
    """A part's world AABB is exactly the axis-aligned bound of its OBB.

    `_instance_part_obbs` composes the two transforms before placing the part's
    local box, and `instance_part_world_aabbs` must bound the same body, or the
    box and the oriented box disagree while both claim to bound the part.
    """
    for deg in (0.0, -45.0, 17.0, 90.0):
        ir, catalog = _rotated_part_scene(instance_deg=deg)
        boxes = dict(scene_query.instance_part_world_aabbs(ir, catalog, "r"))
        from dycraft.sdk import _scene_kernel as _sk
        for name, obb in _sk._instance_part_obbs(ir, catalog, "r", {}):
            (cx, cy, cz), r_cols, half = obb
            # Support extent of the OBB along each world axis.
            span = [sum(abs(r_cols[k][ax]) * half[k] for k in range(3))
                    for ax in range(3)]
            centre = (cx, cy, cz)
            lo, hi = boxes[name]
            for ax in range(3):
                assert _approx(lo[ax], centre[ax] - span[ax], abs_tol=1e-9), (
                    f"{name} lo[{ax}] at {deg} deg")
                assert _approx(hi[ax], centre[ax] + span[ax], abs_tol=1e-9), (
                    f"{name} hi[{ax}] at {deg} deg")


def test_the_instance_box_is_the_union_of_its_parts():
    """The instance box is the union of its part boxes.

    `instance_bbox` and `instance_world_aabb` delegate to
    `instance_part_world_aabbs`, so there is one derivation of "where is this
    instance" instead of three that can drift; this pins that the aggregate
    spellings stay equal to the parts they aggregate. Whether each part box is
    tight is pinned by the two tests above, which compare against an
    independently derived extent.
    """
    ir, catalog = _rotated_part_scene()
    boxes = [b for _n, b in scene_query.instance_part_world_aabbs(
        ir, catalog, "r")]
    union = (tuple(min(b[0][k] for b in boxes) for k in range(3)),
             tuple(max(b[1][k] for b in boxes) for k in range(3)))
    world = scene_query.instance_world_aabb(ir, catalog, "r")
    assert _approx_vec(world[0], union[0], abs_tol=1e-12)
    assert _approx_vec(world[1], union[1], abs_tol=1e-12)
    # And the rest-pose spelling is the same box, not a fourth answer.
    rest = scene_query.instance_bbox(ir, catalog, "r")
    assert _approx_vec(rest[0], union[0], abs_tol=1e-12)
    assert _approx_vec(rest[1], union[1], abs_tol=1e-12)
