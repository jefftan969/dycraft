"""A frozen view must answer exactly what the raw object answers.

The probe overlays hand sandboxed code `_readonly.freeze()` views of `ir`,
`catalog` and `objects` so an expression cannot mutate harness state it does
not own. Those views are `Mapping` / `Sequence` and nothing narrower, so every
`isinstance(x, dict)` or `isinstance(x, (tuple, list))` between the sandbox and
the kernel reads False on a value the agent read out of them.

A `dict` test in `_anchors.find_anchor_record`, for example, would make every
declared semantic anchor read as absent from every probe — and because an
instance PLACED on a declared anchor resolves its world pose through the same
lookup, `instance_pose`, `instance_bbox`, `scene_bbox`, `within_report`,
`overlap_along`, and `room_containment_margins` would raise `UnknownAnchor` too,
while `_probes.scene_list_anchors` would answer that the object registered no
named frames.

A check at one site cannot catch a concrete-type test at another, so this file
asserts the PROPERTY instead: every probe, called both ways, agrees. A new probe joins by being listed in `SCENE_PROBES`;
a new concrete-type check anywhere under one fails here.

See `_readonly.is_mapping` / `is_sequence`, which are what these validators are
supposed to ask.
"""

from __future__ import annotations

import pytest

from dycraft.sdk import _probes as P
from dycraft.sdk import _readonly as RO
from dycraft.sdk._errors import InvalidAnchor


# --------------------------------------------------------------------------
# A scene whose instances are PLACED on a declared anchor, which is the case
# that broke: the world-pose resolution goes through the same lookup, so a
# failure there takes every geometric query with it.
#
# Built with the real builders rather than hand-written IR — `make_box` is
# analytic and needs no CAD, so this stays schema-correct by construction and
# runs on a machine without the pinned build123d.
# --------------------------------------------------------------------------

@pytest.fixture
def scene():
    from dycraft.sdk import Object, Scene
    from dycraft.sdk.scene_builder import ObjectHandle, _reset_scene_singleton
    from dycraft.sdk.object_builder import _reset_object_singleton

    _reset_object_singleton(limit=None)
    _reset_scene_singleton()

    obj = Object("bench")
    top = obj.make_box("top", 1.8, 0.6, 0.05)
    leg = obj.make_box("leg", 0.06, 0.06, 0.85)
    obj.declare_root(top)
    # The declared anchor at the part's own origin — the exact shape nine
    # independent agents reported as "serialized but invisible to the probe".
    obj.add_anchor(top, "datum", translation=(0.0, 0.0, 0.0))
    obj.attach_fixed("leg_fixed", top, leg,
                     obj.anchor(top, "bottom"), obj.anchor(leg, "top"))
    # A movable joint, so `joint_limits` and `motion_displacement` have
    # something to answer about.
    flap = obj.make_box("flap", 0.4, 0.3, 0.02)
    obj.attach_revolute("flap_hinge", top, flap, axis=(1.0, 0.0, 0.0),
                        limits=(0.0, 90.0), default=0.0,
                        parent_anchor=obj.anchor(top, "front"),
                        child_anchor=obj.anchor(flap, "back"))
    obj_ir = obj.finalize()

    scene = Scene("bench_room")
    scene.set_room(width=6.0, depth=5.0, height=3.0)
    handle = ObjectHandle(obj_ir)
    # Both placements resolve through the DECLARED anchor rather than a bbox
    # landmark: `bench_a` by its child anchor, `bench_b` by BOTH — its parent
    # anchor is `bench_a`'s declared frame. That is what puts the lookup under
    # test on the world-pose path, where a miss takes every geometric query
    # with it rather than just the anchor queries.
    bench_a = scene.attach(
        handle, name="bench_a", parent=scene.room,
        parent_anchor=scene.anchor(scene.room, "bottom",
                                   translation=(1.0, 1.0, 0.9)),
        child_anchor=scene.anchor(handle, "top.datum"))
    bench_b = scene.attach(
        handle, name="bench_b", parent=bench_a,
        parent_anchor=scene.anchor(bench_a, "top.datum",
                                   translation=(-2.0, 0.0, 0.0)),
        child_anchor=scene.anchor(handle, "top.datum"))
    # A group, so the three `group_*` probes have something to answer about.
    scene.create_group(name="benches", members=[bench_a, bench_b])
    yield scene.finalize(), {obj_ir["id"]: obj_ir}
    _reset_object_singleton(limit=None)
    _reset_scene_singleton()


#: Every scene probe, with arguments that exercise the declared-anchor path
#: wherever one is accepted. Keyed by the name the agent sees in `probe`.
def _scene_calls():
    return {
        "list_instances": lambda i, c: P.scene_list_instances(i),
        "list_groups": lambda i, c: P.scene_list_groups(i),
        "list_assertions": lambda i, c: P.scene_list_assertions(i),
        "list_anchors": lambda i, c: P.scene_list_anchors(i, c, "bench_a"),
        "instance_pose": lambda i, c: P.scene_instance_pose(i, c, "bench_a"),
        "instance_bbox": lambda i, c: P.scene_instance_bbox(i, c, "bench_a"),
        "scene_bbox": lambda i, c: P.scene_scene_bbox(i, c),
        "anchor_position": lambda i, c: P.scene_anchor_position(
            i, c, "bench_a", "top.datum"),
        "anchor_position_builtin": lambda i, c: P.scene_anchor_position(
            i, c, "bench_a", "bottom"),
        "anchor_pose": lambda i, c: P.scene_anchor_pose(
            i, c, "bench_a", "top.datum"),
        "outward": lambda i, c: P.scene_outward(i, c, "bench_a", "top"),
        "ir_snapshot": lambda i, c: P.scene_ir_snapshot(i),
        "resolve_entity_aabb": lambda i, c: P.scene_resolve_entity_aabb(
            i, c, {"kind": "instance", "name": "bench_a"}),
        "resolve_direction": lambda i, c: P.scene_resolve_direction(
            i, c, {"source": "world", "vec": [0.0, 0.0, 1.0]}),
        "instance_world_aabb": lambda i, c: P.scene_instance_world_aabb(
            i, c, "bench_a"),
        "room_world_aabb": lambda i, c: P.scene_room_world_aabb(i),
        "within_report": lambda i, c: P.scene_within_report(
            i, c, "bench_a", "room"),
        "distance_between": lambda i, c: P.scene_distance_between(
            i, c, ("bench_a", "top.datum"), ("bench_b", "top.datum")),
        "overlap_along": lambda i, c: P.scene_overlap_along(
            i, c, "bench_a", "bench_b", axis=(0, 1, 0)),
        "angle_between": lambda i, c: P.scene_angle_between(
            i, c, (0, 0, 1), (0, 1, 0)),
        "room_containment_margins": lambda i, c:
            P.scene_room_containment_margins(i, c, "bench_a"),
        "joint_limits": lambda i, c: P.scene_joint_limits(i, c, "bench_a"),
        "group_pose": lambda i, c: P.scene_group_pose(i, c, "benches"),
        "group_bbox": lambda i, c: P.scene_group_bbox(i, c, "benches"),
        "group_world_aabb": lambda i, c: P.scene_group_world_aabb(
            i, c, "benches"),
        "motion_displacement": lambda i, c: P.scene_motion_displacement(
            i, c, "bench_a", anchor="top.datum",
            to_pose={"flap_hinge": 45.0}),
    }


def test_every_scene_probe_is_registered_in_this_files_call_table():
    """The parity test below is only as complete as its call table.

    `SCENE_PROBES` is the registry the scene overlay bulk-binds, so a probe
    added there and not here would be exempt from the property this file
    exists to hold.
    """
    missing = set(P.SCENE_PROBES) - set(_scene_calls())
    assert not missing, (
        f"scene probes with no frozen-view parity case: {sorted(missing)}. "
        f"Add them to `_scene_calls`; a probe that reads `ir` or `catalog` "
        f"can hit a concrete-type check under the freeze boundary.")


@pytest.mark.parametrize("probe", sorted(_scene_calls()))
def test_a_scene_probe_answers_the_same_frozen_or_raw(scene, probe):
    ir, catalog = scene
    call = _scene_calls()[probe]
    raw = call(ir, catalog)
    frozen = call(RO.freeze(ir, "ir"), RO.freeze(catalog, "catalog"))
    assert frozen == raw, (
        f"{probe} disagrees across the freeze boundary: a concrete-type check "
        f"somewhere under it read the frozen view as not-a-dict/list")


def test_the_declared_anchor_case_is_actually_exercised(scene):
    """Guard the guard: a fixture that lost its declared anchor would make
    every case above pass while testing nothing."""
    ir, catalog = scene
    top = next(p for p in next(iter(catalog.values()))["parts"]
               if p["name"] == "top")
    assert [a["name"] for a in top["anchors"]] == ["datum"]
    assert P.scene_list_anchors(ir, catalog, "bench_a")["top"], (
        "the raw call must find the anchor, or the parity assertion is vacuous")


def test_a_frozen_catalog_still_lists_declared_anchors(scene):
    """`scene_list_anchors` returned `{}` under freeze, and its own docstring
    reads an empty dict as 'this object registered no named frames' — so the
    one query that makes declared anchors discoverable actively denied them."""
    ir, catalog = scene
    rows = P.scene_list_anchors(
        RO.freeze(ir, "ir"), RO.freeze(catalog, "catalog"), "bench_a")
    assert [r["name"] for r in rows["top"]] == ["datum"]


def test_an_unknown_anchor_is_still_unknown_under_freeze(scene):
    """The fix must not turn a real miss into a silent success."""
    ir, catalog = scene
    with pytest.raises((InvalidAnchor, KeyError)):
        P.scene_anchor_position(RO.freeze(ir, "ir"),
                                RO.freeze(catalog, "catalog"),
                                "bench_a", "top.no_such_anchor")


# --------------------------------------------------------------------------
# The other half of the boundary: a value the agent READ from a frozen view and
# passes back into an SDK call. `objects[...].part_shape(...)` is frozen for the
# same reason `ir` is, so "place it at the counter's centroid" hands a
# `_ReadOnlySequence` to a validator, which must not demand a tuple or a list.
# --------------------------------------------------------------------------

def _frozen_vec():
    return RO.freeze([0.1, 0.2, 0.3], "objects['x'].part_shape('p')['center']")


def test_a_frozen_vector_is_accepted_as_an_anchor_offset():
    from dycraft.sdk._refs import freeze_anchor_offset
    assert freeze_anchor_offset(_frozen_vec(), None) == (
        (0.1, 0.2, 0.3), (0.0, 0.0, 0.0))


def test_a_frozen_vector_is_accepted_as_a_direction():
    from dycraft.sdk._refs import Direction
    assert Direction.world(_frozen_vec()).world_vec() == pytest.approx(
        (0.26726124, 0.53452248, 0.80178373))


def test_a_frozen_vector_is_accepted_as_an_assertion_axis():
    from dycraft.sdk._assertions import coerce_axis
    assert coerce_axis(_frozen_vec(), InvalidAnchor, refs=[]) == (0.1, 0.2, 0.3)


def test_a_frozen_mapping_is_accepted_as_poses():
    from dycraft.sdk._refs import freeze_poses
    assert freeze_poses(RO.freeze({"hinge": 30}, "poses")) == (("hinge", 30.0),)


def test_a_string_is_still_not_a_vector():
    """`Sequence` includes `str`, and `len('xyz') == 3` would pass a shape
    check that a string must fail — which is why `is_sequence` excludes it."""
    from dycraft.sdk._refs import freeze_anchor_offset
    with pytest.raises(Exception):
        freeze_anchor_offset("xyz", None)
