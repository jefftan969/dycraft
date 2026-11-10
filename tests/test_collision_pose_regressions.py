"""Pose and memo correctness in the collision path, cheap enough for the fast suite.

1. ``_object_kernel.frame_of`` takes ``poses``, so the ``a_frame`` /
   ``b_frame`` evidence on a CCD-witnessed ``PenetrationPair`` describes the
   posed configuration. Rest-pose frames would silently break the repair recipe
   the object prompt teaches ("map the world-frame shift back into the parent's
   local frame via ``frame_of``"), handing the agent the closed-door coordinate
   system for an open-door collision.

2. ``_probes._scene_entity_world_aabb`` forwards ``poses`` in its
   ``"<instance>.<part>"`` branch as the bare-instance branch does, so a
   part-scoped query follows scene-level articulation of the attach chain.

3. ``_scene_kernel.penetration_pairs`` passes the memo it built to
   ``instance_world_aabb`` / ``instance_pose`` rather than a fresh ``{}``, so
   each attach chain is walked once.

An off-center part's scene-layer collider (the ``R·c`` term in
``_instance_part_obbs``) is covered by
``test_solid_primitive.test_scene_kernel_agrees_with_the_object_kernel_on_an_offcenter_part``.

Run: `pytest tests/test_collision_pose_regressions.py -v`
"""

from __future__ import annotations

import math

import pytest

from dycraft.sdk import Object, Scene, ObjectHandle
from dycraft.sdk import _object_kernel as _ok
from dycraft.sdk import _scene_kernel as _sk
from dycraft.sdk import _probes as _pr
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton


@pytest.fixture(autouse=True)
def _reset_singletons():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


def _approx_vec(a, b, *, abs_tol=1e-9):
    return all(math.isclose(x, y, rel_tol=1e-9, abs_tol=abs_tol)
               for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. frame_of honors `poses`; CCD evidence carries the witnessing frame.
# ---------------------------------------------------------------------------

def _make_swinging_overlap():
    """Body + a door hinged at the body's edge, overlapping it at every q.

    The door's child anchor is its ``right`` face, so the door's center sits
    0.25 m from the pivot: rotating the hinge changes the door frame's
    *translation as well as* its rotation, and the two boxes overlap at both
    the rest pose and 90°, so `penetration_pairs` reports the pair either way.
    """
    o = Object("cabinet")
    body = o.make_box("body", 0.50, 0.40, 0.30)
    door = o.make_box("door", 0.50, 0.02, 0.30)
    o.attach_revolute("hinge", body, door,
                      axis=(0.0, 0.0, 1.0), limits=(0.0, 90.0), default=0.0,
                      parent_anchor=o.anchor(body, "center",
                                             translation=(0.20, 0.0, 0.0)),
                      child_anchor=o.anchor(door, "right"))
    o.declare_root(body)
    return o.finalize()


def test_frame_of_honors_poses():
    ir = _make_swinging_overlap()
    open_q = {"hinge": math.pi / 2}

    rest_t, rest_rot = _ok.frame_of(ir, "door")
    open_t, open_rot = _ok.frame_of(ir, "door", poses=open_q)

    assert not _approx_vec(rest_t, open_t, abs_tol=1e-6), (
        "door translation must move when the hinge opens; got the same "
        f"translation {rest_t} at rest and at 90deg")
    assert not _approx_vec(rest_rot, open_rot, abs_tol=1e-6), (
        f"door rotation must change with the hinge; got {rest_rot} both times")
    # The root part is pose-independent — a sanity check that `poses` is
    # applied along the kinematic chain rather than to every frame.
    assert _ok.frame_of(ir, "body") == _ok.frame_of(ir, "body", poses=open_q)


def test_penetration_evidence_frames_are_the_witnessing_pose_not_rest():
    ir = _make_swinging_overlap()
    open_q = {"hinge": math.pi / 2}

    pairs = _ok.penetration_pairs(ir, poses=open_q)
    assert pairs, "fixture must overlap at 90deg for this test to mean anything"

    for p in pairs:
        for name, reported in ((p.a, p.a_frame), (p.b, p.b_frame)):
            assert reported == _ok.frame_of(ir, name, poses=open_q), (
                f"{name}'s reported frame must be the witnessing-pose frame")

    # And it must actually differ from rest for the part that moved, else the
    # assertion above would pass vacuously.
    moving = [p for p in pairs if "door" in (p.a, p.b)]
    assert moving, "expected the door to participate in a reported pair"
    p = moving[0]
    reported = p.a_frame if p.a == "door" else p.b_frame
    assert reported != _ok.frame_of(ir, "door"), (
        "regression: the door's evidence frame is the rest-pose frame")


# ---------------------------------------------------------------------------
# 2. Part-scoped scene AABB forwards `poses` through the attach chain.
# ---------------------------------------------------------------------------

def _make_hinged_object() -> dict:
    _reset_object_singleton()
    o = Object("cabinet2")
    body = o.make_box("body", 0.50, 0.40, 0.30)
    door = o.make_box("door", 0.50, 0.02, 0.30)
    o.attach_revolute("hinge", body, door,
                      axis=(0.0, 0.0, 1.0), limits=(0.0, 90.0), default=0.0,
                      parent_anchor=o.anchor(body, "center",
                                             translation=(0.25, 0.21, 0.0)),
                      child_anchor=o.anchor(door, "center"))
    o.declare_root(body)
    return o.finalize()


def _make_unit_box_object() -> dict:
    _reset_object_singleton()
    o = Object("blk")
    p = o.make_box("body", 0.10, 0.10, 0.10)
    o.declare_root(p)
    return o.finalize()


def test_part_scoped_scene_aabb_forwards_poses():
    cab_ir = _make_hinged_object()
    blk_ir = _make_unit_box_object()
    catalog = {cab_ir["id"]: cab_ir, blk_ir["id"]: blk_ir}

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    cab = s.place(ObjectHandle(cab_ir), name="cab",
                  pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    # `follow=True` (the default) makes the block ride the door's joint.
    s.attach(ObjectHandle(blk_ir), name="blk", parent=cab,
             parent_anchor=s.anchor(cab, "door.front"),
             child_anchor=s.anchor(ObjectHandle(blk_ir), "back"))
    scene_ir = s.finalize()

    open_q = {"cab.hinge": math.pi / 2}

    # Invariant: a single-part object's part-scoped AABB is its instance AABB.
    # It held at rest and broke under `poses` because only the bare-instance
    # branch forwarded them.
    rest_part = _pr._scene_entity_world_aabb(scene_ir, catalog, "blk.body")
    rest_inst = _pr._scene_entity_world_aabb(scene_ir, catalog, "blk")
    assert _approx_vec(rest_part[0], rest_inst[0], abs_tol=1e-9)
    assert _approx_vec(rest_part[1], rest_inst[1], abs_tol=1e-9)

    open_part = _pr._scene_entity_world_aabb(scene_ir, catalog, "blk.body",
                                             poses=open_q)
    open_inst = _pr._scene_entity_world_aabb(scene_ir, catalog, "blk",
                                             poses=open_q)
    assert _approx_vec(open_part[0], open_inst[0], abs_tol=1e-9), (
        "regression: the '<instance>.<part>' branch dropped `poses`, so the "
        f"part-scoped AABB stayed at rest. part={open_part[0]}, "
        f"instance={open_inst[0]}")
    assert _approx_vec(open_part[1], open_inst[1], abs_tol=1e-9)

    # Guard against a vacuous pass: the block must actually move.
    assert not _approx_vec(rest_inst[0], open_inst[0], abs_tol=1e-6), (
        "fixture sanity: the attached block should ride the hinge")


# ---------------------------------------------------------------------------
# 3. penetration_pairs shares one world-pose memo across the sweep.
# ---------------------------------------------------------------------------

def test_scene_penetration_pairs_shares_the_world_pose_memo():
    """The memo built for the OBB pass must be reused, not bypassed.

    Counting `_resolve_placement` calls is the cheapest observable proxy: with
    a shared memo each instance's attach chain is walked exactly once per
    `penetration_pairs` call, whereas a fresh `{}` would re-walk it for the
    broad-phase AABB and again for every emitted pair's frame.
    """
    blk_ir = _make_unit_box_object()
    catalog = {blk_ir["id"]: blk_ir}
    handle = ObjectHandle(blk_ir)

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(4.0, 4.0, 4.0)
    a = s.place(handle, name="a", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    # Overlapping on purpose: `b` sits half inside `a`, so a pair is emitted
    # and the pair-frame lookups (the worst offenders) actually run.
    s.attach(handle, name="b", parent=a,
             parent_anchor=s.anchor(a, "center", translation=(0.05, 0.0, 0.0)),
             child_anchor=s.anchor(handle, "center"))
    scene_ir = s.finalize()

    calls = []
    real = _sk._resolve_placement

    def counting(*args, **kwargs):
        calls.append(args[3] if len(args) > 3 else kwargs.get("entity_name"))
        return real(*args, **kwargs)

    _sk._resolve_placement = counting
    try:
        pairs = _sk.penetration_pairs(scene_ir, catalog)
    finally:
        _sk._resolve_placement = real

    assert pairs, "fixture sanity: a and b must overlap"
    for name in ("a", "b"):
        assert calls.count(name) == 1, (
            f"regression: {name!r}'s attach chain was resolved "
            f"{calls.count(name)}x in one penetration_pairs call; the shared "
            "memo is being bypassed")
