"""The natural string form, wherever exactly one thing can be meant.

Authors write the name of a thing where an API takes the handle for it. Four
signatures accept the name; without that, each of these raises:

    scene.place(..., anchor="top.datum")       InvalidAnchor/bad_anchor_ref_type
    obj.allow_collision("lid", "body", ...)    DanglingReference/got_type: str
    obj.allow_collision(..., joint="hinge")    DanglingReference/got_type: str
    overlap_along(..., axis="y")               ValueError: could not convert
                                               string to float: 'y'

None of the four is ambiguous — each parameter names exactly one entity, one
builder namespace, or one axis vocabulary the assertion surface already
accepts — so accepting the name costs nothing, where refusing it sends an
author to write a local coercion wrapper first.

What must NOT change is the diagnosis for a genuinely wrong TYPE, and for a
name that does not exist. Both are still refused, and the refusal for an
unknown name now lists what there is.
"""

from __future__ import annotations

import pytest

from dycraft.sdk import Object, Scene
from dycraft.sdk import _probes as P
from dycraft.sdk._errors import BuildError, DanglingReference, InvalidAnchor
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import ObjectHandle, _reset_scene_singleton


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton(limit=None)
    _reset_scene_singleton()
    yield
    _reset_object_singleton(limit=1)
    _reset_scene_singleton()


def _bench():
    obj = Object("bench")
    top = obj.make_box("top", 1.0, 0.5, 0.05)
    leg = obj.make_box("leg", 0.05, 0.05, 0.8)
    obj.declare_root(top)
    obj.add_anchor(top, "datum", translation=(0.0, 0.0, 0.0))
    obj.attach_fixed("leg_fixed", top, leg,
                     obj.anchor(top, "bottom"), obj.anchor(leg, "top"))
    return obj


# ---------------------------------------------------------------------------
# Scene: an anchor name where an AnchorRef was required
# ---------------------------------------------------------------------------

def _room_with(handle):
    scene = Scene("r")
    scene.set_room(width=5.0, depth=5.0, height=3.0)
    return scene


def test_place_accepts_an_anchor_name():
    handle = ObjectHandle(_bench().finalize())
    scene = _room_with(handle)
    by_name = scene.place(handle, name="a", pose=((0, 0, 0.9), (0, 0, 0)),
                          anchor="top.datum")
    by_ref = scene.place(handle, name="b", pose=((1, 0, 0.9), (0, 0, 0)),
                         anchor=scene.anchor(handle, "top.datum"))
    ir = scene.finalize()
    placements = {i["name"]: i["placement"] for i in ir["instances"]}
    assert placements[by_name._name]["child_anchor"] == \
        placements[by_ref._name]["child_anchor"], (
        "the name form must resolve to the same anchor as the ref form")


def test_place_accepts_a_builtin_landmark_name():
    handle = ObjectHandle(_bench().finalize())
    scene = _room_with(handle)
    scene.place(handle, name="a", pose=((0, 0, 0), (0, 0, 0)), anchor="bottom")
    assert scene.finalize()["instances"]


def test_attach_accepts_names_on_both_sides():
    handle = ObjectHandle(_bench().finalize())
    scene = _room_with(handle)
    parent = scene.place(handle, name="a", pose=((0, 0, 0.9), (0, 0, 0)),
                         anchor="top.datum")
    scene.attach(handle, name="b", parent=parent,
                 parent_anchor="top.datum", child_anchor="top.datum")
    assert len(scene.finalize()["instances"]) == 2


def test_an_anchor_name_that_does_not_exist_is_still_refused():
    handle = ObjectHandle(_bench().finalize())
    scene = _room_with(handle)
    with pytest.raises(InvalidAnchor):
        scene.place(handle, name="a", pose=((0, 0, 0), (0, 0, 0)),
                    anchor="top.no_such_anchor")


def test_a_wrong_type_still_reports_a_wrong_type():
    """Coercing strings must not blur the diagnosis for a real type error."""
    handle = ObjectHandle(_bench().finalize())
    scene = _room_with(handle)
    with pytest.raises(InvalidAnchor) as exc:
        scene.place(handle, name="a", pose=((0, 0, 0), (0, 0, 0)), anchor=42)
    assert "AnchorRef" in str(exc.value)


# ---------------------------------------------------------------------------
# Object: part and joint names where refs were required
# ---------------------------------------------------------------------------

def test_the_whole_object_surface_can_be_driven_by_name():
    obj = Object("t")
    obj.make_box("a", 0.2, 0.2, 0.2)
    obj.make_box("b", 0.2, 0.2, 0.2)
    obj.declare_root("a")
    obj.attach_revolute("hinge", "a", "b", axis=(0, 0, 1),
                        limits=(0.0, 90.0), default=0.0,
                        parent_anchor=obj.anchor("a", "top"),
                        child_anchor=obj.anchor("b", "bottom"))
    obj.allow_collision("a", "b", reason="the lid seats into the rim",
                        joint="hinge", q_range=(0.0, 90.0))
    ir = obj.finalize()
    assert [p["name"] for p in ir["parts"]] == ["a", "b"]
    assert [j["name"] for j in ir["joints"]] == ["hinge"]
    assert len(ir["collision_allowances"]) == 1


def test_a_part_name_this_object_does_not_have_is_refused_with_the_list():
    obj = Object("t")
    obj.make_box("a", 0.2, 0.2, 0.2)
    obj.declare_root("a")
    with pytest.raises(DanglingReference) as exc:
        obj.allow_collision("a", "typo", reason="x")
    assert "typo" in str(exc.value)
    assert exc.value.evidence["known"] == ["a"], (
        "an unknown name should say what names there are — guessing and "
        "reading the error is the cycle this whole change removes")


def test_a_ref_from_another_object_is_still_refused():
    """Names resolve within one builder; that is what makes them unambiguous.
    A ref carries owner identity and must still be checked against it."""
    first = Object("one")
    first.declare_root(first.make_box("body", 0.1, 0.1, 0.1))
    stolen = first._check_owned("body", "part")

    second = Object("two")
    second.make_box("body", 0.1, 0.1, 0.1)
    second.declare_root("body")
    with pytest.raises(DanglingReference):
        second.allow_collision(stolen, "body", reason="x")


def test_a_fixed_joint_is_still_refused_for_a_scoped_allowance():
    """The joint name resolves; the rule about what it may scope stands."""
    obj = Object("t")
    obj.make_box("a", 0.2, 0.2, 0.2)
    obj.make_box("b", 0.2, 0.2, 0.2)
    obj.declare_root("a")
    obj.attach_fixed("welded", "a", "b",
                     obj.anchor("a", "top"), obj.anchor("b", "bottom"))
    with pytest.raises(BuildError) as exc:
        obj.allow_collision("a", "b", reason="x", joint="welded",
                            q_range=(0.0, 1.0))
    assert "fixed" in str(exc.value)


# ---------------------------------------------------------------------------
# Probes: the axis vocabulary the assertions already accept
# ---------------------------------------------------------------------------

def _two_boxes():
    obj = Object("t")
    obj.make_box("a", 0.2, 0.2, 0.2)
    obj.make_box("b", 0.2, 0.2, 0.2)
    obj.declare_root("a")
    obj.attach_fixed("j", "a", "b",
                     obj.anchor("a", "top"), obj.anchor("b", "bottom"))
    return obj.finalize()


@pytest.mark.parametrize("name,vector", [
    ("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)),
    ("-z", (0, 0, -1)), ("+x", (1, 0, 0)),
])
def test_overlap_along_accepts_the_same_axis_names_as_the_assertions(name, vector):
    ir = _two_boxes()
    assert P.object_overlap_along(ir, "a", "b", axis=name) == \
        P.object_overlap_along(ir, "a", "b", axis=vector)


def test_an_unknown_axis_name_says_what_is_accepted():
    with pytest.raises(ValueError) as exc:
        P.object_overlap_along(_two_boxes(), "a", "b", axis="q")
    assert "unknown axis name" in str(exc.value)


def test_distance_between_keeps_none_meaning_unprojected():
    """`axis=None` is not an axis — it means the full distance — so the
    coercion must pass it through rather than reject it."""
    ir = _two_boxes()
    full = P.object_distance_between(ir, ("a", "center"), ("b", "center"))
    along_z = P.object_distance_between(ir, ("a", "center"), ("b", "center"),
                                        axis="z")
    assert full == pytest.approx(abs(along_z))
