"""Per-instance initial joint state, and semantic frames for scene placement.

Two capabilities that a modeling agent cannot supply by writing better scene
source, because neither is a matter of arithmetic it could do itself:

  * **`initial_joints`** — a saved configuration per *instance*, sharing one
    definition and one set of meshes. Without it, one cabinet open and one
    shut would need the object built twice with different `default=`, which
    doubles the geometry cost and gives the two bodies different identities;
    and the evaluation overrides (`.at({...})`, `poses=`) change what a query
    *asks about*, not what the scene commits, so nothing downstream — a
    render, a collision sweep, an export, an animation's undriven joints —
    would see them.
  * **declared semantic frames on the scene surface** — `<part>.<declared>` as
    a placement or attachment anchor, carrying the authored *rotation*, so
    attachment can use an authored mounting face.

What these tests hold onto, beyond "the argument is accepted":

  * one definition, one `object_ref`, one mesh set, whatever the states;
  * the four-level precedence, order-independently, in surface units;
  * the initial-alignment rule — a placement resolved once, at the state it
    was placed in, and never re-derived when that instance later moves;
  * numerical frame alignment with nontrivial rotations on *both* sides, so a
    commuting axis-aligned case cannot hide a composition error;
  * that the consumers agree: probes, assertions, collision, support, render,
    URDF, animation, and a save/reload round trip.
"""

from __future__ import annotations

import json
import math

import pytest

from dycraft.sdk import _probes as P
from dycraft.sdk import _rotation as R
from dycraft.sdk import _scene_kernel as SK
from dycraft.sdk import ir as IR
from dycraft.sdk._errors import InvalidAnchor, InvalidJoint
from dycraft.sdk.object_builder import (Object, _reset_object_singleton,
                                        take_finalized_objects)
from dycraft.sdk.scene_builder import ObjectHandle, Scene, _reset_scene_singleton


# ---------------------------------------------------------------------------
# Fixtures — plain builders, no CAD interpreter. Boxes are enough: what is
# under test is frames and state, and a box has an exact AABB.
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk():
    """Reset both builders the way `reset_scene_layer_state` does."""
    _reset_object_singleton(limit=None)
    _reset_scene_singleton()
    take_finalized_objects()
    yield
    _reset_object_singleton(limit=1)
    _reset_scene_singleton()
    take_finalized_objects()


def _catalog() -> dict:
    """`{object_ref: ObjectIR}` for everything finalized so far."""
    out = {}
    for handle in take_finalized_objects().values():
        d = getattr(handle, "_ir", handle)
        out[d["id"]] = d
    return out


def make_cabinet(name: str = "cabinet", *, default_deg: float = 15.0) -> Object:
    """A carcass with a hinged door, a declared mount on each of two parts.

    `door_hinge`'s default is deliberately **non-zero**: an instance that omits
    the joint must inherit 15 degrees, not 0, which is the difference between
    "inherits" and "is reset".

    Both parts declare an anchor called `sensor_mount`, at different places and
    with different rotations, so a test can show that `body.sensor_mount` and
    `door.sensor_mount` are two frames rather than one name.
    """
    obj = Object(name)
    body = obj.make_box("body", 0.6, 0.4, 0.8)
    door = obj.make_box("door", 0.6, 0.02, 0.8)
    obj.declare_root(body)
    # A mounting face on the back of the carcass, turned to face -Y.
    obj.add_anchor(body, "wall_mount", translation=(0.0, -0.2, 0.4),
                   rotation=(0, 0, 180))
    obj.add_anchor(body, "sensor_mount", translation=(0.0, -0.2, 0.1),
                   rotation=(0, 0, 0))
    # …and one on the door's outer face, rotated about two axes so the
    # composition below cannot come out right by commuting.
    obj.add_anchor(door, "sensor_mount", translation=(0.1, 0.01, 0.3),
                   rotation=(90, 0, 30))
    obj.attach_revolute("door_hinge", parent=body, child=door,
                        axis=(0, 0, 1), limits=(0, 120), default=default_deg,
                        parent_anchor=obj.anchor(body, "front_right"),
                        child_anchor=obj.anchor(door, "back_right"))
    return obj


def make_sensor(name: str = "sensor") -> Object:
    obj = Object(name)
    b = obj.make_box("body", 0.05, 0.05, 0.02)
    obj.declare_root(b)
    obj.add_anchor(b, "mount", translation=(0.0, 0.0, -0.01),
                   rotation=(0, 0, 0))
    return obj


def make_slider(name: str = "slider") -> Object:
    """A prismatic drawer — the metres half of the unit contract."""
    obj = Object(name)
    case = obj.make_box("case", 0.5, 0.5, 0.3)
    tray = obj.make_box("tray", 0.45, 0.45, 0.2)
    obj.declare_root(case)
    obj.add_anchor(tray, "handle", translation=(0.0, 0.225, 0.0),
                   rotation=(0, 0, 0))
    obj.attach_prismatic("pull", parent=case, child=tray,
                         axis=(0, 1, 0), limits=(0.0, 0.4), default=0.05,
                         parent_anchor=obj.anchor(case, "center"),
                         child_anchor=obj.anchor(tray, "center"))
    return obj


def _se3_of_declared(obj_ir, part, anchor):
    """The declared anchor's `Anchor -> Part` SE3, read straight off the IR.

    An independent second opinion: every assertion about frame alignment below
    composes from this rather than from the kernel helper under test, so a bug
    in that helper cannot make its own check pass.
    """
    for p in obj_ir["parts"]:
        if p["name"] != part:
            continue
        for a in p.get("anchors") or ():
            if a["name"] == anchor:
                return (tuple(a["translation"]), tuple(a["rotation"]))
    raise AssertionError(f"no declared anchor {part}.{anchor}")


def _anchor_world_se3(ir, catalog, instance, part, anchor, poses=None):
    """`part world SE3 ∘ declared anchor in part` — the long way round."""
    part_world = SK.instance_part_world_se3(ir, catalog, instance, part,
                                            poses=poses)
    obj_ir = catalog[
        next(i for i in ir["instances"] if i["name"] == instance)["object_ref"]]
    return R.se3_compose(part_world, _se3_of_declared(obj_ir, part, anchor))


def _assert_se3_close(a, b, *, tol=1e-9, what=""):
    for i, (x, y) in enumerate(zip(a[0], b[0])):
        assert abs(x - y) < tol, f"{what} translation[{i}]: {a[0]} vs {b[0]}"
    # A quaternion and its negation are the same rotation; both sides come from
    # `se3_compose`, which does not canonicalize, so compare either sign.
    same = all(abs(x - y) < tol for x, y in zip(a[1], b[1]))
    flipped = all(abs(x + y) < tol for x, y in zip(a[1], b[1]))
    assert same or flipped, f"{what} rotation: {a[1]} vs {b[1]}"


# ---------------------------------------------------------------------------
# 1. Repeated placements of one definition at different saved states
# ---------------------------------------------------------------------------


def test_one_definition_three_states_one_object_ref(sdk):
    """The catalog holds one definition; the bodies differ.

    The whole value of the feature is in the conjunction. Two cabinets at
    different door angles is easy if you build two objects — and then the
    catalog carries two of everything, the two bodies are not recognisably the
    same product, and an animation cannot target them as one kind. So this
    checks the sharing and the difference in the same breath: identical
    `object_ref`, identical joint defaults still on the definition, and three
    measurably different door transforms.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="shut", pose=((-2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    scene.place(cab, name="open", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 90})
    scene.place(cab, name="untouched", pose=((2, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    catalog = _catalog()

    refs = {i["object_ref"] for i in ir["instances"]}
    assert len(refs) == 1, "three placements produced more than one definition"
    obj_ir = catalog[refs.pop()]
    # The definition is untouched: its default is still the authored 15 deg.
    hinge = next(j for j in obj_ir["joints"] if j["name"] == "door_hinge")
    assert hinge["default"] == pytest.approx(math.radians(15))
    # …and there is exactly one set of parts, i.e. one set of meshes.
    assert [p["name"] for p in obj_ir["parts"]] == ["body", "door"]

    angles = {}
    for name in ("shut", "open", "untouched"):
        _t, q = SK.instance_part_world_se3(ir, catalog, name, "door")
        # Signed rotation about +Z recovered from the quaternion.
        angles[name] = round(math.degrees(2.0 * math.atan2(q[3], q[0])), 6)
    assert angles["shut"] == pytest.approx(0.0, abs=1e-6)
    assert angles["open"] == pytest.approx(90.0, abs=1e-6)
    # The omitted joint inherited the definition's non-zero default, and was
    # not reset to zero by the presence of the feature.
    assert angles["untouched"] == pytest.approx(15.0, abs=1e-6)

    # Bounds differ too, not just frames: the open door sweeps the box out.
    boxes = {n: SK.instance_world_aabb(ir, catalog, n)
             for n in ("shut", "open", "untouched")}
    depth = {n: b[1][1] - b[0][1] for n, b in boxes.items()}
    assert depth["open"] > depth["untouched"] > depth["shut"]


def test_state_changes_the_scene_id_but_not_the_object_ref(sdk):
    """A state-only edit must invalidate scene-level caches and nothing else.

    Both halves matter. If the scene id did not move, a re-verify would answer
    from the previous configuration's cached results. If the `object_ref` did,
    every geometry asset would be re-staged and re-compiled for a change that
    touched no geometry — which is exactly what building the object twice with
    different defaults costs, and what this feature exists to avoid.
    """
    def build(angle):
        _reset_object_singleton(limit=None)
        _reset_scene_singleton()
        take_finalized_objects()
        cab = make_cabinet()
        s = Scene("s")
        s.set_room(4.0, 4.0, 3.0)
        s.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": angle})
        return s.finalize()

    a, b = build(10), build(40)
    assert a["id"] != b["id"], "a state-only edit left the scene id unchanged"
    assert a["instances"][0]["object_ref"] == b["instances"][0]["object_ref"]


def test_a_scene_without_the_feature_serializes_exactly_as_before(sdk):
    """No `initial_joints` anywhere -> the base tag, no new keys.

    The conditional schema tag is the compatibility promise: every scene ever
    committed keeps its `id`, its cache keys and its `scene_ref` checks. A test
    that only asserted "the newer version exists" would not notice an
    unconditional bump re-id-ing the entire corpus.

    **The base is v2, not v1.** `@lighting` replaced the symbolic lighting API
    and tags every new scene v2 unconditionally, so no scene this SDK can build
    is v1 any more — v1 is archived scenes only. `initial_joints` therefore
    bumps to v3, which is what the conditional tag means now. Asserted through
    `SCENE_SCHEMA_VERSION` rather than a literal so the two stay in step.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    assert ir["schema"] == IR.SCENE_SCHEMA_VERSION
    assert "initial_joints" not in ir["instances"][0]
    # The id is the hash of the canonical bytes, so an extra key would show up
    # here even if it were empty.
    without_id = {k: v for k, v in ir.items() if k != "id"}
    assert IR.content_hash(without_id) == ir["id"]


def test_the_schema_bumps_to_v3_only_when_state_is_present(sdk):
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="plain", pose=((-1, 0, 0), (0, 0, 0)))
    scene.place(cab, name="stateful", pose=((1, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 30})
    ir = scene.finalize()
    assert ir["schema"] == IR.SCENE_SCHEMA_VERSION_V3
    assert IR.SCENE_SCHEMA_VERSION_V3 in IR.SCENE_SCHEMA_VERSIONS
    # Per instance, not per scene: the untouched one carries no key.
    by_name = {i["name"]: i for i in ir["instances"]}
    assert "initial_joints" not in by_name["plain"]
    assert by_name["stateful"]["initial_joints"] == {
        "door_hinge": pytest.approx(math.radians(30))}


def test_an_empty_mapping_is_the_same_as_omitting_it(sdk):
    """`initial_joints={}` must not tip a scene into v3 or change its hash.

    An author generating placements in a loop naturally ends up passing an
    empty dict for the ones with nothing to say, and that should cost nothing.
    """
    def build(state):
        _reset_object_singleton(limit=None)
        _reset_scene_singleton()
        take_finalized_objects()
        cab = make_cabinet()
        s = Scene("s")
        s.set_room(4.0, 4.0, 3.0)
        s.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints=state)
        return s.finalize()

    omitted, empty = build(None), build({})
    assert omitted["id"] == empty["id"]
    assert empty["schema"] == IR.SCENE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. Units, validation, and input ownership
# ---------------------------------------------------------------------------


def test_revolute_takes_degrees_and_prismatic_takes_metres(sdk):
    """One boundary conversion, per joint type.

    A prismatic value scaled by 180/pi is a factor of 57 in the wrong
    direction with no error anywhere — the failure mode `scene_joint_type`
    exists to prevent on the query side, and the same trap on the way in.
    """
    cab, sl = make_cabinet(), make_slider()
    scene = Scene("s")
    scene.set_room(6.0, 6.0, 3.0)
    scene.place(cab, name="cab", pose=((-1, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 60})
    scene.place(sl, name="drawer", pose=((1, 0, 0), (0, 0, 0)),
                initial_joints={"pull": 0.25})
    ir = scene.finalize()
    by_name = {i["name"]: i for i in ir["instances"]}
    # Stored internal: radians for revolute, meters unchanged for prismatic.
    assert by_name["cab"]["initial_joints"]["door_hinge"] == \
        pytest.approx(math.radians(60))
    assert by_name["drawer"]["initial_joints"]["pull"] == pytest.approx(0.25)

    catalog = _catalog()
    # …and the round trip back to the surface reports what was authored.
    limits = P.scene_joint_limits(ir, catalog, "cab")["door_hinge"]
    assert limits["default"] == pytest.approx(60.0)
    assert limits["unit"] == "deg"
    pull = P.scene_joint_limits(ir, catalog, "drawer")["pull"]
    assert pull["default"] == pytest.approx(0.25)
    assert pull["unit"] == "m"

    # 0.25 *meters* of travel, not 0.25 radians of anything. Measured as a
    # difference against an explicit zero on the same instance, because the
    # placement anchor is itself resolved at the committed state (see the
    # initial-alignment tests), so the tray's absolute world Y is not the
    # joint value.
    committed = SK.instance_part_world_se3(ir, catalog, "drawer", "tray")
    at_zero = SK.instance_part_world_se3(ir, catalog, "drawer", "tray",
                                          poses={"drawer.pull": 0.0})
    assert committed[0][1] - at_zero[0][1] == pytest.approx(0.25, abs=1e-9)


def test_explicit_zero_is_a_selection_not_an_omission(sdk):
    """`{"door_hinge": 0}` on a joint whose default is 15 must mean 0.

    The one case a truthiness bug eats silently, and the natural spelling for
    "this one is shut".
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="shut", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    ir = scene.finalize()
    assert ir["schema"] == IR.SCENE_SCHEMA_VERSION_V3
    assert ir["instances"][0]["initial_joints"] == {"door_hinge": 0.0}
    _t, q = SK.instance_part_world_se3(ir, _catalog(), "shut", "door")
    assert math.degrees(2.0 * math.atan2(q[3], q[0])) == pytest.approx(0.0)


def test_the_authors_mapping_is_copied(sdk):
    """Mutating the dict afterwards cannot reach the scene.

    An author who builds one `poses`-shaped dict and edits it per placement is
    doing something entirely reasonable; aliasing it into the IR would make the
    committed state depend on what happened after the call.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    state = {"door_hinge": 30}
    scene.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints=state)
    state["door_hinge"] = 999
    state["bogus"] = 1
    ir = scene.finalize()
    assert ir["instances"][0]["initial_joints"] == {
        "door_hinge": pytest.approx(math.radians(30))}


@pytest.mark.parametrize("state, fragment", [
    ({"no_such_joint": 10}, "does not have"),
    ({"door_hinge": "40"}, "must be a number"),
    ({"door_hinge": True}, "must be a number"),
    ({"door_hinge": float("nan")}, "must be finite"),
    ({"door_hinge": float("inf")}, "must be finite"),
    ({"door_hinge": 200}, "outside the joint's authored limits"),
    ({"door_hinge": -5}, "outside the joint's authored limits"),
    ({"cab.door_hinge": 40}, "instance-qualified"),
])
def test_malformed_state_is_refused_with_an_actionable_error(sdk, state,
                                                             fragment):
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    with pytest.raises(InvalidJoint) as e:
        scene.place(cab, name="cab", pose=((0, 0, 0), (0, 0, 0)),
                    initial_joints=state)
    assert fragment in str(e.value)
    assert e.value.suggested_repairs, "no repair offered"


def test_a_range_error_is_reported_in_the_units_it_was_written_in(sdk):
    """Degrees in, degrees in the complaint.

    The limits live in the IR as radians. Reporting `[0.0, 2.094]` for a joint
    the author wrote `limits=(0, 120)` on sends them hunting for a bug in their
    own arithmetic.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    with pytest.raises(InvalidJoint) as e:
        scene.place(cab, name="cab", pose=((0, 0, 0), (0, 0, 0)),
                    initial_joints={"door_hinge": 200})
    msg = str(e.value)
    assert "120" in msg and "deg" in msg
    assert "2.09" not in msg
    assert e.value.evidence["limits"] == pytest.approx([0.0, 120.0])


def test_a_fixed_joint_has_no_configuration_to_select(sdk):
    obj = Object("bench")
    top = obj.make_box("top", 1.0, 0.6, 0.05)
    leg = obj.make_box("leg", 0.05, 0.05, 0.7)
    obj.declare_root(top)
    obj.attach_fixed("leg_mount", top, leg,
                     parent_anchor=obj.anchor(top, "bottom_front_left"),
                     child_anchor=obj.anchor(leg, "top_front_left"))
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    with pytest.raises(InvalidJoint, match="fixed joint"):
        scene.place(obj, name="b", pose=((0, 0, 0), (0, 0, 0)),
                    initial_joints={"leg_mount": 0.1})


def test_a_joint_ref_key_is_accepted(sdk):
    """The same key vocabulary `.at({...})` takes."""
    obj = Object("cab")
    body = obj.make_box("body", 0.4, 0.4, 0.4)
    door = obj.make_box("door", 0.4, 0.02, 0.4)
    obj.declare_root(body)
    hinge = obj.attach_revolute("h", parent=body, child=door,
                                axis=(0, 0, 1), limits=(0, 90), default=0,
                                parent_anchor=obj.anchor(body, "front_right"),
                                child_anchor=obj.anchor(door, "back_right"))
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(obj, name="a", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={hinge: 45})
    ir = scene.finalize()
    assert ir["instances"][0]["initial_joints"] == {
        "h": pytest.approx(math.radians(45))}


# ---------------------------------------------------------------------------
# 3. Effective-configuration precedence
# ---------------------------------------------------------------------------


def _hinge_deg(ir, catalog, instance, poses=None):
    _t, q = SK.instance_part_world_se3(ir, catalog, instance, "door",
                                       poses=poses)
    return math.degrees(2.0 * math.atan2(q[3], q[0]))


@pytest.fixture
def two_cabinets(sdk):
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="left", pose=((-1.5, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    scene.place(cab, name="right", pose=((1.5, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 90})
    ir = scene.finalize()
    return ir, _catalog()


def test_precedence_from_definition_default_up_to_a_qualified_override(
        two_cabinets):
    """All four levels, on one scene."""
    ir, catalog = two_cabinets
    rad = math.radians
    # 4. definition default — reached by an instance that saved nothing.
    #    (`left`/`right` both saved, so check the plain case separately.)
    # 3. saved initial value, with no override in play.
    assert _hinge_deg(ir, catalog, "left") == pytest.approx(0.0, abs=1e-6)
    assert _hinge_deg(ir, catalog, "right") == pytest.approx(90.0, abs=1e-6)
    # 2. a bare override broadcasts and outranks saved state.
    assert _hinge_deg(ir, catalog, "left",
                      poses={"door_hinge": rad(45)}) == pytest.approx(45.0)
    assert _hinge_deg(ir, catalog, "right",
                      poses={"door_hinge": rad(45)}) == pytest.approx(45.0)
    # 1. a qualified override outranks the bare one, for that instance only.
    poses = {"door_hinge": rad(45), "right.door_hinge": rad(120)}
    assert _hinge_deg(ir, catalog, "left", poses=poses) == pytest.approx(45.0)
    assert _hinge_deg(ir, catalog, "right", poses=poses) == pytest.approx(120.0)


def test_qualified_precedence_is_independent_of_insertion_order(two_cabinets):
    """`dict` ordering must not decide the answer.

    `{"hinge": 0, "cab.hinge": 40}` is the ordinary way to write "all shut
    except this one", and a single-pass projection would write whichever key
    it reached last. Same scene, same question, two literals.
    """
    ir, catalog = two_cabinets
    rad = math.radians
    a = _hinge_deg(ir, catalog, "right",
                   poses={"door_hinge": rad(10), "right.door_hinge": rad(100)})
    b = _hinge_deg(ir, catalog, "right",
                   poses={"right.door_hinge": rad(100), "door_hinge": rad(10)})
    assert a == pytest.approx(b)
    assert a == pytest.approx(100.0, abs=1e-6)


def test_a_partial_override_leaves_the_other_joints_alone(sdk):
    """Two joints, one named. The other keeps its *saved* value.

    Not its definition default and not zero — which are the two things a
    "replace the whole dict" merge would produce, and both look plausible in
    isolation.
    """
    obj = Object("rig")
    base = obj.make_box("base", 0.4, 0.4, 0.1)
    arm = obj.make_box("arm", 0.1, 0.1, 0.5)
    tip = obj.make_box("tip", 0.08, 0.08, 0.1)
    obj.declare_root(base)
    obj.attach_revolute("yaw", parent=base, child=arm, axis=(0, 0, 1),
                        limits=(-90, 90), default=10,
                        parent_anchor=obj.anchor(base, "top"),
                        child_anchor=obj.anchor(arm, "bottom"))
    obj.attach_prismatic("extend", parent=arm, child=tip, axis=(0, 0, 1),
                         limits=(0.0, 0.3), default=0.02,
                         parent_anchor=obj.anchor(arm, "top"),
                         child_anchor=obj.anchor(tip, "bottom"))
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(obj, name="r", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"yaw": 45, "extend": 0.2})
    ir = scene.finalize()
    catalog = _catalog()

    def state(poses=None):
        _t, q = SK.instance_part_world_se3(ir, catalog, "r", "arm",
                                            poses=poses)
        yaw = math.degrees(2.0 * math.atan2(q[3], q[0]))
        tz = SK.instance_part_world_se3(ir, catalog, "r", "tip",
                                        poses=poses)[0][2]
        return round(yaw, 6), round(tz, 9)

    yaw0, tip0 = state()
    assert yaw0 == pytest.approx(45.0)
    # Override yaw only; `extend` must hold its saved 0.2, not 0.02 and not 0.
    yaw1, tip1 = state({"r.yaw": 0.0})
    assert yaw1 == pytest.approx(0.0)
    assert tip1 == pytest.approx(tip0), "a partial override moved `extend`"
    # And overriding `extend` only leaves yaw at 45.
    yaw2, _tip2 = state({"r.extend": 0.0})
    assert yaw2 == pytest.approx(45.0)


def test_a_qualified_key_for_another_instance_is_inert(two_cabinets):
    ir, catalog = two_cabinets
    before = SK.anchor_position(ir, catalog, "left", "door.center")
    after = SK.anchor_position(ir, catalog, "left", "door.center",
                               poses={"right.door_hinge": 0.0})
    assert after == pytest.approx(before)


# ---------------------------------------------------------------------------
# 4. Semantic frames for attachment, with rotations on both sides
# ---------------------------------------------------------------------------


def test_attaching_by_declared_frames_aligns_translation_and_rotation(sdk):
    """The two frames coincide, checked independently.

    A translation-only resolution passes every position check and leaves the
    sensor facing whatever direction the object happened to be modeled in.
    So this composes both sides from the raw IR records and compares the full
    SE3, on a parent that is rotated in the scene, whose part is rotated by a
    joint, and whose anchor is rotated about two axes — no pair of which
    commutes with the others.
    """
    cab, sensor = make_cabinet(), make_sensor()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    # The cabinet itself is turned 35 degrees in the room…
    host = scene.place(cab, name="host", pose=((0.4, -0.3, 0), (0, 0, 35)),
                       initial_joints={"door_hinge": 70})
    scene.attach(sensor, name="probe", parent=host,
                 parent_anchor=scene.anchor(host, "door.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"))
    ir = scene.finalize()
    catalog = _catalog()

    parent_frame = _anchor_world_se3(ir, catalog, "host", "door",
                                     "sensor_mount")
    child_frame = _anchor_world_se3(ir, catalog, "probe", "body", "mount")
    _assert_se3_close(parent_frame, child_frame, what="declared-frame attach")

    # Non-degenerate: the frame really is turned, so an identity-rotation bug
    # could not have produced this.
    assert any(abs(v) > 1e-3 for v in parent_frame[1][1:]), \
        "the fixture stopped exercising a rotated frame"

    # And the probe surface reports the same frame both sides.
    pa = P.scene_anchor_pose(ir, catalog, "host", "door.sensor_mount")
    ca = P.scene_anchor_pose(ir, catalog, "probe", "body.mount")
    assert pa[0] == pytest.approx(ca[0], abs=1e-9)
    assert pa[1] == pytest.approx(ca[1], abs=1e-6)


def test_part_qualification_keeps_two_same_named_frames_apart(sdk):
    """`body.sensor_mount` and `door.sensor_mount` are different frames.

    Both parts of the fixture declare `sensor_mount`, deliberately. A resolver
    that looked the name up object-wide would find one of them and be right
    half the time.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    ir = scene.finalize()
    catalog = _catalog()
    on_body = P.scene_anchor_pose(ir, catalog, "c", "body.sensor_mount")
    on_door = P.scene_anchor_pose(ir, catalog, "c", "door.sensor_mount")
    assert on_body[0] != pytest.approx(on_door[0])
    assert on_body[1] != pytest.approx(on_door[1])
    # Each matches its own declared record.
    _assert_se3_close(
        (on_body[0], R.euler_xyz_deg_to_quat_wxyz(*on_body[1])),
        _anchor_world_se3(ir, catalog, "c", "body", "sensor_mount"),
        tol=1e-6, what="body.sensor_mount")
    _assert_se3_close(
        (on_door[0], R.euler_xyz_deg_to_quat_wxyz(*on_door[1])),
        _anchor_world_se3(ir, catalog, "c", "door", "sensor_mount"),
        tol=1e-6, what="door.sensor_mount")


def test_an_anchor_offset_composes_in_the_declared_frames_axes(sdk):
    """The object layer's convention, not a scene-layer variant of it.

    `object_builder._anchor_to_se3` composes `base ∘ T(t) ∘ R(r)`, so "5 mm out
    along the mounting face" follows the face. Checked against a hand-composed
    expectation, and checked for agreement between the position query and the
    frame query, which must apply the offset in the same frame.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    inst = scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 0})
    ir = scene.finalize()
    catalog = _catalog()

    offset = ((0.0, 0.0, 0.03), (0.0, 0.0, 0.0))
    base = _anchor_world_se3(ir, catalog, "c", "door", "sensor_mount")
    expected = R.se3_compose(base, ((0.0, 0.0, 0.03), (1.0, 0.0, 0.0, 0.0)))

    posed = SK.anchor_pose(ir, catalog, "c", "door.sensor_mount",
                           anchor_offset=offset)
    _assert_se3_close((posed[0], R.euler_xyz_deg_to_quat_wxyz(*posed[1])),
                      expected, tol=1e-6, what="offset frame")
    # The declared frame is rotated, so the offset is NOT a plain +Z shift in
    # object axes.
    assert abs(posed[0][2] - base[0][2]) < 0.03 - 1e-4, \
        "the offset was applied in object axes, not the anchor's"

    point = SK.anchor_position(ir, catalog, "c", "door.sensor_mount",
                               anchor_offset=offset)
    assert point == pytest.approx(posed[0], abs=1e-9), \
        "the position and frame queries disagree about the same offset"
    del inst


def test_outward_refuses_a_declared_frame_and_says_what_to_ask_instead(sdk):
    """Deliberate: a declared frame has an orientation, not a face normal.

    Inventing one would be a number with no authored meaning. What changed is
    the message — a bare `KeyError('door.sensor_mount')` reads as "no such
    anchor" for an anchor that resolves fine everywhere else.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    catalog = _catalog()
    with pytest.raises(KeyError) as e:
        P.scene_outward(ir, catalog, "c", "door.sensor_mount")
    msg = str(e.value)
    assert "anchor_pose" in msg
    assert "built-in" in msg
    # Landmarks still work.
    assert P.scene_outward(ir, catalog, "c", "door.front") == \
        pytest.approx((0.0, 1.0, 0.0))


def test_an_unknown_part_anchor_names_both_vocabularies(sdk):
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    inst = scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(InvalidAnchor) as e:
        scene.anchor(inst, "door.nope")
    msg = str(e.value)
    assert "sensor_mount" in msg, "the part's declared anchors were not listed"
    assert "built-in" in msg
    assert e.value.evidence["declared_anchors"] == ["sensor_mount"]


def test_an_unqualified_declared_name_says_to_part_qualify_it(sdk):
    """The mistake this capability creates, diagnosed as itself.

    Making declared anchors usable from a scene turns the *unqualified*
    spelling into the common slip: `wall_mount` is now a frame the author has
    in mind, and reporting `unknown instance anchor 'wall_mount'` is the wrong
    diagnosis — it sends them looking for a name that exists.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    inst = scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(InvalidAnchor) as e:
        scene.anchor(inst, "wall_mount")
    assert "must be part-qualified" in str(e.value)
    assert "'body.wall_mount'" in str(e.value)
    assert e.value.evidence["declared_on"] == ["body"]


def test_a_name_declared_on_two_parts_names_both_qualified_forms(sdk):
    """`sensor_mount` is on the body and on the door; the part decides which."""
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    with pytest.raises(InvalidAnchor) as e:
        scene.anchor(cab, "sensor_mount")
    msg = str(e.value)
    assert "body.sensor_mount" in msg and "door.sensor_mount" in msg
    assert sorted(e.value.evidence["declared_on"]) == ["body", "door"]


def test_a_part_with_no_declared_anchors_says_so(sdk):
    obj = Object("plain")
    obj.declare_root(obj.make_box("body", 0.3, 0.3, 0.3))
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    inst = scene.place(obj, name="p", pose=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(InvalidAnchor) as e:
        scene.anchor(inst, "body.grip")
    assert "declares no semantic anchors" in str(e.value)
    assert "add_anchor" in " ".join(e.value.suggested_repairs)


# ---------------------------------------------------------------------------
# 5. Definition entry paths, and identity
# ---------------------------------------------------------------------------


def test_anchor_accepts_a_builder_a_dict_and_a_handle(sdk):
    """The three spellings `place`/`attach` take, on `anchor` too.

    Without this, a `scene.py` had no callable child-anchor path at all —
    the instance does not exist yet and no `ObjectHandle` constructor is
    injected — so the prompt shipped a place-then-wrap-in-a-single-member-group
    workaround.
    """
    scene = Scene("s")
    scene.set_room(6.0, 6.0, 3.0)

    a = make_sensor("s_builder")
    host = scene.place(make_cabinet("host_a"), name="host",
                       pose=((0, 0, 0), (0, 0, 0)))
    scene.attach(a, name="from_builder", parent=host,
                 parent_anchor=scene.anchor(host, "body.sensor_mount"),
                 child_anchor=scene.anchor(a, "body.mount"))

    b_ir = make_sensor("s_dict").finalize()
    scene.attach(b_ir, name="from_dict", parent=host,
                 parent_anchor=scene.anchor(host, "body.top"),
                 child_anchor=scene.anchor(b_ir, "body.mount"))

    c = ObjectHandle(make_sensor("s_handle").finalize())
    scene.attach(c, name="from_handle", parent=host,
                 parent_anchor=scene.anchor(host, "body.bottom"),
                 child_anchor=scene.anchor(c, "body.mount"))

    ir = scene.finalize()
    assert {i["name"] for i in ir["instances"]} == {
        "host", "from_builder", "from_dict", "from_handle"}


def test_taking_an_anchor_on_a_builder_finalizes_it(sdk):
    """A documented lifecycle, not an accident.

    The anchor path is checked against the definition's parts and declared
    frames, which are only settled at `finalize`. `place` already has this
    lifecycle; making `anchor` share it keeps one rule. The later `attach` is
    free, since `finalize` is idempotent.
    """
    from dycraft.sdk._errors import BuildError

    sensor = make_sensor()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.anchor(sensor, "body.mount")
    with pytest.raises(BuildError, match="finalized"):
        sensor.make_box("late", 0.1, 0.1, 0.1)


def test_the_same_definition_placed_twice_shares_its_anchors(sdk):
    """Repeated identical definitions stay interchangeable.

    `make_cabinet()` called twice produces two builders and one content hash,
    which is the case an author actually hits when a helper is called in a
    loop. Identity is the hash, so this must keep working.
    """
    scene = Scene("s")
    scene.set_room(6.0, 6.0, 3.0)
    first, second = make_cabinet("twin"), make_cabinet("twin")
    a = scene.place(first, name="a", pose=((-1, 0, 0), (0, 0, 0)))
    sensor = make_sensor()
    # The child anchor is taken on `second`, the attach passes `first`: same
    # content hash, so they are the same definition.
    scene.attach(sensor, name="s", parent=a,
                 parent_anchor=scene.anchor(a, "body.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"))
    scene.place(second, name="b", pose=((1, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    refs = {i["object_ref"] for i in ir["instances"]
            if i["object_name"] == "twin"}
    assert len(refs) == 1


def test_two_different_definitions_sharing_a_name_are_refused(sdk):
    """Identity by content hash, not by display name.

    A `scene.py` that builds a variant per station names them all
    `bracket`. The anchor path was validated against one variant's parts and
    declared frames; accepting the other resolved it against geometry it had
    never been checked against — a `KeyError` deep in the kernel at best, and a
    silently wrong frame at worst.
    """
    tall = make_sensor("bracket")
    _reset_object_singleton(limit=None)
    short = Object("bracket")
    body = short.make_box("body", 0.05, 0.05, 0.9)   # a different shape
    short.declare_root(body)
    short.add_anchor(body, "mount", translation=(0.0, 0.0, -0.45))

    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    host = scene.place(make_cabinet(), name="host", pose=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(InvalidAnchor) as e:
        scene.attach(short, name="b", parent=host,
                     parent_anchor=scene.anchor(host, "body.top"),
                     child_anchor=scene.anchor(tall, "body.mount"))
    msg = str(e.value)
    assert "different definition" in msg
    assert "not the same object" in msg, "the same-name case is not called out"
    assert e.value.evidence["anchor_object_ref"] != \
        e.value.evidence["handle_object_ref"]


def test_a_foreign_anchor_ref_is_still_refused(sdk):
    """The pre-existing ownership check must survive the new entry paths."""
    scene_a = Scene("a")
    scene_a.set_room(4.0, 4.0, 3.0)
    sensor = make_sensor()
    foreign = scene_a.anchor(sensor, "body.mount")

    _reset_scene_singleton()
    scene_b = Scene("b")
    scene_b.set_room(4.0, 4.0, 3.0)
    host = scene_b.place(make_cabinet(), name="host",
                         pose=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(InvalidAnchor, match="not produced by this scene"):
        scene_b.attach(sensor, name="s", parent=host,
                       parent_anchor=scene_b.anchor(host, "body.top"),
                       child_anchor=foreign)


# ---------------------------------------------------------------------------
# 6. Initial alignment versus later motion
# ---------------------------------------------------------------------------


def test_placement_is_resolved_at_the_state_it_was_placed_in(sdk):
    """The open cabinet's own box lands on the floor.

    `place` targets the bbox-bottom-center by default, and the box of an open
    cabinet is not the box of a shut one. Resolving the child anchor at the
    definition's default would put the body somewhere the author did not ask
    for, in proportion to how far the door was opened.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="shut", pose=((-2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    scene.place(cab, name="open", pose=((2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 110})
    ir = scene.finalize()
    catalog = _catalog()
    for name, x in (("shut", -2.0), ("open", 2.0)):
        (xmn, ymn, zmn), (xmx, ymx, _zmx) = SK.instance_world_aabb(
            ir, catalog, name)
        assert zmn == pytest.approx(0.0, abs=1e-9), f"{name} is not on the floor"
        assert 0.5 * (xmn + xmx) == pytest.approx(x, abs=1e-9)
        assert 0.5 * (ymn + ymx) == pytest.approx(0.0, abs=1e-9)


def test_moving_a_joint_after_placement_does_not_move_the_root(sdk):
    """No root compensation.

    The failure this rules out is subtle and would look like a physics bug: if
    an anchor-based placement were re-derived at the *evaluation* pose, opening
    a door for a snapshot would slide the whole cabinet sideways to keep its
    growing bbox-bottom on the requested point. Object defaults, saved instance
    state, and time-varying state are three different things.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="c", pose=((0.5, -0.25, 0), (0, 0, 20)),
                initial_joints={"door_hinge": 30})
    ir = scene.finalize()
    catalog = _catalog()
    base = SK.instance_pose(ir, catalog, "c")
    at_committed = SK.instance_part_world_se3(ir, catalog, "c", "door")[0]
    for q in (0.0, math.radians(60), math.radians(120)):
        posed = SK.instance_pose(ir, catalog, "c", poses={"c.door_hinge": q})
        assert posed[0] == pytest.approx(base[0], abs=1e-12)
        assert posed[1] == pytest.approx(base[1], abs=1e-12)
        # …while the door itself does move, or the test proves nothing.
        moved = SK.instance_part_world_se3(
            ir, catalog, "c", "door", poses={"c.door_hinge": q})[0]
        assert moved != pytest.approx(at_committed, abs=1e-9)


def test_a_child_anchor_on_an_articulated_part_is_pinned_at_placement(sdk):
    """The child's *own* placement anchor rides its own joint.

    Placing a body by a landmark on a part that moves is the case where the
    two rules diverge most: the anchor is resolved once, in the configuration
    selected at `place`, and afterwards the body stays put while that part
    swings inside it.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="c", pose=((0, 0, 1.0), (0, 0, 0)),
                anchor=None, initial_joints={"door_hinge": 45})
    ir = scene.finalize()
    catalog = _catalog()

    # The placement used the whole-body bbox bottom at 45 degrees.
    (_xmn, _ymn, zmn), _hi = SK.instance_world_aabb(ir, catalog, "c")
    assert zmn == pytest.approx(1.0, abs=1e-9)

    base = SK.instance_pose(ir, catalog, "c")
    # Swing the door: the root does not compensate, so the *box* moves.
    moved = SK.instance_pose(ir, catalog, "c",
                             poses={"c.door_hinge": math.radians(0)})
    assert moved[0] == pytest.approx(base[0], abs=1e-12)
    # The box changes shape — the door swings about +Z, so it is the Y extent
    # that moves, not the Z floor contact — while the root stays put. Both
    # halves are the rule: alignment was decided once, and the geometry is
    # free to move inside it afterwards.
    box_committed = SK.instance_world_aabb(ir, catalog, "c")
    box_at_zero = SK.instance_world_aabb(
        ir, catalog, "c", poses={"c.door_hinge": 0.0})
    assert (box_at_zero[1][1] - box_at_zero[0][1]) != \
        pytest.approx(box_committed[1][1] - box_committed[0][1], abs=1e-6), \
        "the box did not change, so nothing was pinned"


def test_placing_by_a_part_landmark_on_a_moving_part(sdk):
    """The explicit `anchor=` form of the same rule.

    A body hung by a landmark on its own door is a strange thing to author,
    which is exactly why the rule has to be written down rather than left to
    whichever configuration a helper happened to evaluate in.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="a", pose=((0, 0, 0.5), (0, 0, 0)),
                anchor=scene.anchor(cab, "door.bottom"),
                initial_joints={"door_hinge": 0})
    scene.place(cab, name="b", pose=((2, 0, 0.5), (0, 0, 0)),
                anchor=scene.anchor(cab, "door.bottom"),
                initial_joints={"door_hinge": 90})
    ir = scene.finalize()
    catalog = _catalog()
    # Each one's own door bottom is at the requested height.
    for name in ("a", "b"):
        pos = SK.anchor_position(ir, catalog, name, "door.bottom")
        assert pos[2] == pytest.approx(0.5, abs=1e-9)
    # The two roots differ, because the door's frame differs at the two states.
    assert SK.instance_pose(ir, catalog, "a")[0] != \
        pytest.approx(SK.instance_pose(ir, catalog, "b")[0])


# ---------------------------------------------------------------------------
# 7. Following a moving parent whose committed state differs from its default
# ---------------------------------------------------------------------------


def test_a_child_mounts_at_the_parents_committed_state_and_then_rides_it(sdk):
    """Initial placement, subsequent motion, whole-body motion.

    The parent's committed 70 degrees is neither its definition default (15)
    nor zero, so a consumer that resolved the parent anchor at either would
    put the sensor visibly off the door.
    """
    cab, sensor = make_cabinet(), make_sensor()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    host = scene.place(cab, name="host", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 70})
    scene.attach(sensor, name="probe", parent=host,
                 parent_anchor=scene.anchor(host, "door.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"),
                 follow=True)
    ir = scene.finalize()
    catalog = _catalog()

    # Initial placement: on the door as committed, not as the definition rests.
    _assert_se3_close(_anchor_world_se3(ir, catalog, "host", "door",
                                        "sensor_mount"),
                      _anchor_world_se3(ir, catalog, "probe", "body", "mount"),
                      what="mount at committed state")

    # Subsequent motion: it rides the joint.
    poses = {"host.door_hinge": math.radians(10)}
    _assert_se3_close(
        _anchor_world_se3(ir, catalog, "host", "door", "sensor_mount",
                          poses=poses),
        _anchor_world_se3(ir, catalog, "probe", "body", "mount", poses=poses),
        what="mount under articulation")
    assert SK.instance_pose(ir, catalog, "probe", poses=poses)[0] != \
        pytest.approx(SK.instance_pose(ir, catalog, "probe")[0])

    # Whole-body motion: it rides that too, independently of `follow`.
    delta = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    shifted = SK.instance_pose(ir, catalog, "probe",
                               base_deltas={"host": delta})
    plain = SK.instance_pose(ir, catalog, "probe")
    assert shifted[0][0] - plain[0][0] == pytest.approx(1.0, abs=1e-9)


def test_follow_false_pins_to_the_committed_state_not_the_definition(sdk):
    """`follow=False` is a state, and it is the authored one.

    "Aim at where this articulation sits in the delivered scene and stay
    there" — so the fixture lands where the door is at the committed 70
    degrees, and stays when the door is driven elsewhere. It is emphatically
    not a whole-world freeze: whole-body motion still propagates, checked
    below.
    """
    cab, sensor = make_cabinet(), make_sensor()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    host = scene.place(cab, name="host", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 70})
    scene.attach(sensor, name="fixture", parent=host,
                 parent_anchor=scene.anchor(host, "door.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"),
                 follow=False)
    ir = scene.finalize()
    catalog = _catalog()

    at_committed = _anchor_world_se3(ir, catalog, "host", "door",
                                      "sensor_mount")
    fixture = _anchor_world_se3(ir, catalog, "fixture", "body", "mount")
    _assert_se3_close(at_committed, fixture, what="follow=False placement")

    # It does NOT sit where the definition's default would put the door.
    host_ref = next(i for i in ir["instances"]
                    if i["name"] == "host")["object_ref"]
    at_default = R.se3_compose(
        SK.instance_part_world_se3(ir, catalog, "host", "door",
                                    poses={"host.door_hinge":
                                           math.radians(15)}),
        _se3_of_declared(catalog[host_ref], "door", "sensor_mount"))
    assert fixture[0] != pytest.approx(at_default[0])

    # Driving the joint leaves it alone.
    driven = SK.instance_pose(ir, catalog, "fixture",
                              poses={"host.door_hinge": math.radians(120)})
    assert driven[0] == pytest.approx(
        SK.instance_pose(ir, catalog, "fixture")[0], abs=1e-12)

    # But whole-body motion still reaches it.
    delta = ((0.0, 0.7, 0.0), (1.0, 0.0, 0.0, 0.0))
    shifted = SK.instance_pose(ir, catalog, "fixture",
                               base_deltas={"host": delta})
    assert shifted[0][1] - SK.instance_pose(ir, catalog, "fixture")[0][1] == \
        pytest.approx(0.7, abs=1e-9)


def test_a_nested_attachment_inherits_the_whole_chain(sdk):
    """Two edges deep, through a parent's committed state."""
    cab = make_cabinet()
    a, b = make_sensor("bracket"), make_sensor("tip")
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    host = scene.place(cab, name="host", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 85})
    mid = scene.attach(a, name="bracket", parent=host,
                       parent_anchor=scene.anchor(host, "door.sensor_mount"),
                       child_anchor=scene.anchor(a, "body.mount"))
    scene.attach(b, name="tip", parent=mid,
                 parent_anchor=scene.anchor(mid, "body.top"),
                 child_anchor=scene.anchor(b, "body.mount"))
    ir = scene.finalize()
    catalog = _catalog()

    # The leaf sits above the bracket, which sits on the door as committed.
    _assert_se3_close(_anchor_world_se3(ir, catalog, "host", "door",
                                        "sensor_mount"),
                      _anchor_world_se3(ir, catalog, "bracket", "body",
                                        "mount"),
                      what="bracket")
    tip_at_committed = SK.instance_pose(ir, catalog, "tip")
    # Driving the host's door moves the leaf as well as the bracket.
    poses = {"host.door_hinge": math.radians(5)}
    assert SK.instance_pose(ir, catalog, "tip", poses=poses)[0] != \
        pytest.approx(tip_at_committed[0])
    # And a delta on the host reaches two edges down.
    delta = ((0.0, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))
    lifted = SK.instance_pose(ir, catalog, "tip",
                              base_deltas={"host": delta})
    assert lifted[0][2] - tip_at_committed[0][2] == pytest.approx(0.4,
                                                                   abs=1e-9)


def test_a_group_union_box_contains_its_members_committed_states(sdk):
    """A group of an open and a shut cabinet is wider than two shut ones."""
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    left = scene.place(cab, name="left", pose=((-0.4, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 0})
    right = scene.attach(cab, name="right", parent=left,
                         parent_anchor=scene.anchor(left, "body.left"),
                         child_anchor=scene.anchor(cab, "body.right"),
                         initial_joints={"door_hinge": 100})
    bank = scene.create_group(name="bank", members=[left, right], root=left)
    ir = scene.finalize()
    catalog = _catalog()
    box = SK.group_world_aabb(ir, catalog, "bank")
    single = SK.instance_world_aabb(ir, catalog, "left")
    assert (box[1][1] - box[0][1]) > (single[1][1] - single[0][1]) + 0.2, \
        "the group box did not grow to contain the open door"
    # The group's own bbox anchor also sees the committed states.
    assert SK.anchor_position(ir, catalog, "bank", "front")[1] == \
        pytest.approx(box[1][1], abs=1e-9)
    del bank


# ---------------------------------------------------------------------------
# 8. The consumers agree
# ---------------------------------------------------------------------------


@pytest.fixture
def colliding_scene(sdk):
    """A cabinet placed open, with a post standing where its door swung to.

    The verdict has to *depend on* the committed state or the test proves
    nothing, so the post's position is measured rather than guessed: build the
    cabinet alone, ask where its door sits at 110 degrees, then stand the post
    there. At the definition's 15 degrees the door is nowhere near it.
    """
    cab = make_cabinet()
    probe_scene = Scene("probe")
    probe_scene.set_room(8.0, 8.0, 3.0)
    probe_scene.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)),
                      initial_joints={"door_hinge": 110})
    probe_ir = probe_scene.finalize()
    door_at_110 = SK.instance_part_world_se3(probe_ir, _catalog(), "a",
                                              "door")[0]

    _reset_object_singleton(limit=None)
    _reset_scene_singleton()
    take_finalized_objects()
    cab = make_cabinet()
    post = Object("post")
    post.declare_root(post.make_box("shaft", 0.06, 0.06, 1.2))
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 110})
    scene.place(post, name="post",
                pose=((door_at_110[0], door_at_110[1], 0.0), (0, 0, 0)))
    ir = scene.finalize()
    return ir, _catalog()


def test_scene_collision_sampling_evaluates_the_delivered_state(
        colliding_scene):
    """The sweep's own "rest" is the committed configuration.

    This is the call site that a kernel change alone does not reach.
    `_scene_joints` builds the sampler's records and `rest_pose` turns them
    into a pose dict naming *every* movable joint explicitly — and an explicit
    override outranks saved state, so reporting the definition's default there
    would hand the sweep a configuration the scene never committed and quietly
    clear a door the author placed open.
    """
    from dycraft.verify import scene_verify

    ir, catalog = colliding_scene
    joints = scene_verify._scene_joints(ir, catalog)
    by_name = {j["name"]: j for j in joints}
    assert by_name["a.door_hinge"]["default"] == \
        pytest.approx(math.radians(110))
    # Limits stay the definition's: an instance selects a value, not a range.
    obj_ir = catalog[ir["instances"][0]["object_ref"]]
    hinge = next(j for j in obj_ir["joints"] if j["name"] == "door_hinge")
    assert by_name["a.door_hinge"]["limits"] == hinge["limits"]

    # And the geometry actually overlaps at the committed state while the
    # definition-default configuration is clear — so the distinction is not
    # academic on this fixture. `rest_pose` over those records is exactly what
    # `scene_no_penetration` samples when `static_only` is on.
    from dycraft.verify._ccd_sampler import rest_pose

    sample = rest_pose(joints)[0]
    committed = SK.penetration_part_pairs(ir, catalog, poses=sample or None)
    at_defaults = SK.penetration_part_pairs(
        ir, catalog, poses={"a.door_hinge": math.radians(15)})
    assert committed, "fixture no longer collides at the committed state"
    assert not at_defaults, "fixture also collides at the definition default"
    # …and the no-poses call, which is what every other consumer makes, agrees
    # with the sampled one.
    assert SK.penetration_part_pairs(ir, catalog), \
        "the committed state is not the default answer"


def test_a_scene_assertion_is_evaluated_at_the_committed_state(sdk):
    """An assertion's verdict follows the saved configuration.

    The relational verifier resolves anchors with `poses=None`, which is the
    committed state rather than the definition's defaults — so an
    assertion about where an open door's edge is passes for the instance that
    was placed open and fails for the one that was not.
    """
    from dycraft.verify._relational import evaluate_assertions

    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    opened = scene.place(cab, name="opened", pose=((-2, 0, 0), (0, 0, 0)),
                         initial_joints={"door_hinge": 90})
    shut = scene.place(cab, name="shut", pose=((2, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 0})
    # With the door at 90 degrees the door's center is a long way forward of
    # the carcass center; shut, the two nearly coincide.
    scene.assert_distance(a=scene.anchor(opened, "body.center"),
                          b=scene.anchor(opened, "door.center"),
                          min=0.25, max=1.0, name="opened_swings_out")
    scene.assert_distance(a=scene.anchor(shut, "body.center"),
                          b=scene.anchor(shut, "door.center"),
                          min=0.25, max=1.0, name="shut_swings_out")
    ir = scene.finalize()
    catalog = _catalog()

    def resolve(entity, anchor, poses=None, anchor_offset=None):
        return SK.anchor_position(ir, catalog, entity, anchor, poses=poses,
                                  anchor_offset=anchor_offset)

    results, _repairs, any_fail = evaluate_assertions(
        ir["assertions"], resolve)
    by_name = {r["name"]: r for r in results}
    assert by_name["opened_swings_out"]["verdict"] == "pass"
    assert by_name["shut_swings_out"]["verdict"] == "fail"
    assert any_fail


def test_scene_support_sees_the_committed_state(sdk):
    """`_scene_instance_part_aabbs` asks with no poses; that must mean committed.

    A drawer pulled out either reaches the wall or does not, and the support
    graph is built from part boxes at "rest".
    """
    from dycraft.verify import _support

    sl = make_slider()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(sl, name="pulled", pose=((-1, 0, 0), (0, 0, 0)),
                initial_joints={"pull": 0.4})
    scene.place(sl, name="shut", pose=((1, 0, 0), (0, 0, 0)),
                initial_joints={"pull": 0.0})
    ir = scene.finalize()
    catalog = _catalog()
    boxes = _support._scene_instance_part_aabbs(ir, catalog, SK)

    def tray_offset(inst):
        """The tray's Y offset from its own case, i.e. the joint's travel.

        Measured relative to the case rather than in world Y because the
        placement anchor is resolved at the committed state too, so the
        absolute position mixes the two effects.
        """
        parts = dict(boxes[inst])
        mid = lambda b: 0.5 * (b[0][1] + b[1][1])  # noqa: E731
        return mid(parts["tray"]) - mid(parts["case"])

    assert tray_offset("pulled") - tray_offset("shut") == \
        pytest.approx(0.4, abs=1e-9)
    assert tray_offset("shut") == pytest.approx(0.0, abs=1e-9)


def test_the_renderer_draws_the_committed_state(sdk):
    """What is authored and what is drawn are the same scene."""
    from dycraft.render import _world

    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="open", pose=((-2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 110})
    scene.place(cab, name="shut", pose=((2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 0})
    ir = scene.finalize()
    catalog = _catalog()

    parts, _room = _world._build_scene_geometry(ir, catalog)
    by_name = {p.name: p for p in parts}
    for inst in ("open", "shut"):
        drawn = by_name[f"{inst}/door"].se3
        expected = SK.instance_part_world_se3(ir, catalog, inst, "door")
        _assert_se3_close(drawn, expected, tol=1e-9, what=f"{inst} door")
    # The two are genuinely different, so a shared-definition bug would show.
    assert by_name["open/door"].se3[0] != \
        pytest.approx(by_name["shut/door"].se3[0])


def test_the_urdf_pins_a_following_child_at_the_parents_committed_state(sdk):
    """The export's fixed-joint origin needs no post-correction.

    URDF cannot express a per-instance configuration — that travels in
    `scene_meta.json`, seeded from `scene_joint_limits` — but the frame it pins
    a kinematic-follow child at is exact, and it is a *composition* of two
    transforms that must be taken at the same configuration.
    """
    from dycraft.render import urdf_export

    cab, sensor = make_cabinet(), make_sensor()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    host = scene.place(cab, name="host", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 95})
    scene.attach(sensor, name="probe", parent=host,
                 parent_anchor=scene.anchor(host, "door.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"))
    ir = scene.finalize()
    catalog = _catalog()

    instances = {i["name"]: i for i in ir["instances"]}
    kin = urdf_export._kinematic_parent(ir, catalog, instances["probe"],
                                        instances)
    assert kin is not None, "the follow attach did not become a URDF edge"
    parent_link, origin = kin
    assert parent_link == "host__door"

    # `origin` is the child root in the parent part's LINK frame. Recompose it
    # forward and it must land on the door's declared mount, at 95 degrees.
    p_poses = SK.instance_initial_joints(ir, "host")
    p_obj = catalog[instances["host"]["object_ref"]]
    frames = SK._object_query._compute_frames(p_obj, poses=p_poses)
    incoming = next(j for j in p_obj["joints"] if j["child"] == "door")
    part_in_link = R.se3_inverse(
        (tuple(incoming["child_anchor"]["translation"]),
         tuple(incoming["child_anchor"]["rotation"])))
    link_in_root = R.se3_compose(frames["door"],
                                 R.se3_inverse(part_in_link))
    child_root_in_root = R.se3_compose(link_in_root, origin)
    host_world = SK.instance_pose(ir, catalog, "host")
    host_se3 = (host_world[0], R.euler_xyz_deg_to_quat_wxyz(*host_world[1]))
    child_world = R.se3_compose(host_se3, child_root_in_root)
    probe_world = SK.instance_pose(ir, catalog, "probe")
    assert child_world[0] == pytest.approx(probe_world[0], abs=1e-9)


def test_the_sim_meta_handoff_carries_the_per_instance_state(sdk):
    """The URDF's missing field, supplied by the record the loader reads.

    `scene_export` fills `meta["joints"]` from `scene_joint_limits` and the IK
    helper (`dycraft_ik.default_q`) seeds each robot from it, so the open
    cabinet loads open. Checked at the `scene_joint_limits` boundary, which is the part
    of that chain this change owns.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="open", pose=((-2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 100})
    scene.place(cab, name="rest", pose=((2, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    catalog = _catalog()
    a = P.scene_joint_limits(ir, catalog, "open")["door_hinge"]
    b = P.scene_joint_limits(ir, catalog, "rest")["door_hinge"]
    assert a["default"] == pytest.approx(100.0)
    assert b["default"] == pytest.approx(15.0)
    # The definition's own value stays visible under its own name.
    assert a["object_default"] == pytest.approx(15.0)
    assert b["object_default"] == pytest.approx(15.0)


def test_the_viewer_seeds_each_instance_from_its_own_state(sdk):
    """Two instances of one definition must not share one slider seed."""
    from dycraft.viewer import scene as viewer_scene

    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    scene.place(cab, name="open", pose=((-2, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 100})
    scene.place(cab, name="rest", pose=((2, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
    catalog = _catalog()
    _parts, joints = viewer_scene.scene_parts_and_joints(
        ir, catalog, lambda p: {"name": p["name"]})
    by_name = {j["name"]: j for j in joints}
    assert by_name["open__door_hinge"]["default"] == \
        pytest.approx(math.radians(100))
    assert by_name["rest__door_hinge"]["default"] == \
        pytest.approx(math.radians(15))


# ---------------------------------------------------------------------------
# 9. Save / reload
# ---------------------------------------------------------------------------


def test_the_state_survives_a_json_round_trip(sdk):
    """Reload and the answers are identical.

    A SceneIR is committed as JSON and read back by the animation layer, the
    viewer and the exporters. A state that lived only in the builder would
    evaporate at exactly the boundary it matters most.
    """
    cab, sensor = make_cabinet(), make_sensor()
    scene = Scene("s")
    scene.set_room(8.0, 8.0, 3.0)
    host = scene.place(cab, name="host", pose=((0, 0, 0), (0, 0, 0)),
                       initial_joints={"door_hinge": 65})
    scene.attach(sensor, name="probe", parent=host,
                 parent_anchor=scene.anchor(host, "door.sensor_mount"),
                 child_anchor=scene.anchor(sensor, "body.mount"))
    ir = scene.finalize()
    catalog = _catalog()

    reloaded = json.loads(json.dumps(ir))
    # The content hash still checks out, so no consumer will call it tampered.
    without_id = {k: v for k, v in reloaded.items() if k != "id"}
    assert IR.content_hash(without_id) == reloaded["id"]

    for name in ("host", "probe"):
        before = SK.instance_pose(ir, catalog, name)
        after = SK.instance_pose(reloaded, catalog, name)
        assert after[0] == pytest.approx(before[0], abs=1e-12)
        assert after[1] == pytest.approx(before[1], abs=1e-12)
    assert SK.instance_part_world_se3(reloaded, catalog, "host", "door") == \
        SK.instance_part_world_se3(ir, catalog, "host", "door")
    assert SK.instance_initial_joints(reloaded, "host") == \
        SK.instance_initial_joints(ir, "host")


def test_the_committed_state_cannot_be_edited_through_a_query(sdk):
    """`instance_initial_joints` hands back a copy.

    Every kernel query calls it, and one of them passing the stored dict into
    a mutating merge would edit the frozen scene for every later query.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(cab, name="c", pose=((0, 0, 0), (0, 0, 0)),
                initial_joints={"door_hinge": 30})
    ir = scene.finalize()
    got = SK.instance_initial_joints(ir, "c")
    got["door_hinge"] = 99.0
    got["injected"] = 1.0
    assert SK.instance_initial_joints(ir, "c") == {
        "door_hinge": pytest.approx(math.radians(30))}


def test_a_non_instance_name_has_no_committed_state(sdk):
    """Groups, the room and unknown names answer None, not KeyError.

    Callers hand the result straight to `_compute_frames`, and several of them
    sweep every entity kind.
    """
    cab = make_cabinet()
    scene = Scene("s")
    scene.set_room(4.0, 4.0, 3.0)
    a = scene.place(cab, name="a", pose=((0, 0, 0), (0, 0, 0)))
    scene.create_group(name="grp", members=[a])
    ir = scene.finalize()
    assert SK.instance_initial_joints(ir, "grp") is None
    assert SK.instance_initial_joints(ir, "room") is None
    assert SK.instance_initial_joints(ir, "nope") is None
    assert SK.instance_initial_joints(ir, "a") is None


# ---------------------------------------------------------------------------
# 10. Animation inheritance
# ---------------------------------------------------------------------------


# An instance's `initial_joints` seeding its joint channels in an animation is
# tested against Blender in `test_animation_build_blender.py`.
