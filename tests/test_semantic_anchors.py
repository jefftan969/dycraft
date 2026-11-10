"""Persistent Part-owned semantic Anchors.

Semantic anchors are *authored knowledge*: a hinge line or a mounting face that
no bounding box can tell you. They live in `PartIR.anchors`, are named and
Part-scoped, and are resolved through the same code path as the 27 built-in
AABB landmarks.

The invariants worth breaking a build over:

  * AABB-only objects keep their exact object ids -- semantic anchors must be
    invisible to anything that does not use them;
  * one resolver serves joints and assertions, so a name means the same frame
    in both;
  * `PartIR.anchors` is authoritative and a joint's embedded transform is a
    snapshot derived from it, with drift detectable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.sdk import _anchors as A
from dycraft.sdk._errors import InvalidAnchor
from dycraft.sdk import _object_kernel as K
from dycraft.sdk import _probes as P
from dycraft.sdk import _rotation as R

REPO = Path(__file__).resolve().parents[1]

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="needs the pinned build123d interpreter")


def _build(body: str) -> dict:
    out = W.run_object_source(body, run_dir=Path(tempfile.mkdtemp()))
    assert out["ok"], out.get("build_error") or out.get("message")
    return out["ir"]


_TWO_PART = """
obj = Object("t")
a = obj.add_part("a", Box(0.4, 0.2, 0.1))
b = obj.add_part("b", Box(0.3, 0.06, 0.06))
mount = obj.add_anchor(a, "boom_mount", translation=(0.2, 0, 0.05))
hinge = obj.add_anchor(b, "base_hinge", translation=(-0.15, 0, 0),
                       rotation=(90, 0, 0))
obj.add_anchor(b, "tip_mount", translation=(0.15, 0, 0))
obj.declare_root(a)
obj.attach_revolute("pitch", parent=a, child=b, axis=(0, 1, 0),
                    limits=(-45, 45), default=0,
                    parent_anchor=mount, child_anchor=hinge)
obj.assert_distance(a=mount, b=obj.anchor(b, "top"), min=0.0, max=5.0, name="d1")
ir = obj.finalize()
"""


# ---------------------------------------------------------------------------
# Registry and serialization
# ---------------------------------------------------------------------------

@_needs_cad
def test_anchors_are_stored_under_their_owning_part():
    ir = _build(_TWO_PART)
    by_name = {p["name"]: p for p in ir["parts"]}
    assert [x["name"] for x in by_name["a"]["anchors"]] == ["boom_mount"]
    assert [x["name"] for x in by_name["b"]["anchors"]] == ["base_hinge",
                                                            "tip_mount"]
    rec = by_name["b"]["anchors"][0]
    # `kind` is gone from the API and from new records: nothing ever read it,
    # so authoring it was work with no effect. `source` stays as a constant —
    # every already-staged anchor carries it, and dropping it would move those
    # objects' content hashes for no gain.
    assert set(rec) == {"name", "translation", "rotation", "source"}
    assert rec["source"] == "authored"
    assert len(rec["rotation"]) == 4 and rec["rotation"][0] >= 0.0


@_needs_cad
def test_anchor_list_is_sorted_by_name():
    """Deterministic serialization: insertion order must not leak into the IR,
    or the object id would depend on authoring order."""
    ir = _build("""
obj = Object("t")
a = obj.add_part("a", Box(0.2, 0.2, 0.2))
obj.add_anchor(a, "zeta", translation=(0.1, 0, 0))
obj.add_anchor(a, "alpha", translation=(0, 0.1, 0))
obj.add_anchor(a, "mid", translation=(0, 0, 0.1))
obj.declare_root(a)
ir = obj.finalize()
""")
    assert [x["name"] for x in ir["parts"][0]["anchors"]] == \
        ["alpha", "mid", "zeta"]


@_needs_cad
def test_the_same_anchor_name_may_exist_on_two_parts():
    ir = _build("""
obj = Object("t")
a = obj.add_part("a", Box(0.2, 0.2, 0.2))
b = obj.add_part("b", Box(0.2, 0.2, 0.2))
obj.add_anchor(a, "hinge", translation=(0.1, 0, 0))
obj.add_anchor(b, "hinge", translation=(-0.1, 0, 0))
obj.declare_root(a)
obj.attach_fixed("j", parent=a, child=b,
                 parent_anchor=obj.anchor(a, "top"),
                 child_anchor=obj.anchor(b, "bottom"))
ir = obj.finalize()
""")
    for part in ir["parts"]:
        assert [x["name"] for x in part["anchors"]] == ["hinge"]


@_needs_cad
@pytest.mark.parametrize("bad,reason", [
    ('obj.add_anchor(a, "left", translation=(0, 0, 0))', "reserved_builtin_name"),
    ('obj.add_anchor(a, "top_front_left", translation=(0, 0, 0))',
     "reserved_builtin_name"),
    ('obj.add_anchor(a, "dup", translation=(0, 0, 0))\n'
     'obj.add_anchor(a, "dup", translation=(1, 0, 0))', "duplicate_anchor_name"),
])
def test_invalid_registrations_are_rejected(bad, reason):
    out = W.run_object_source(
        f'obj = Object("t")\na = obj.add_part("a", Box(0.2,0.2,0.2))\n{bad}\n',
        run_dir=Path(tempfile.mkdtemp()))
    assert out["ok"] is False
    assert reason in json.dumps(out["build_error"])


def test_reserved_names_have_a_single_source_of_truth():
    """The reserved set must be `AUTO_AABB_ANCHOR_NAMES` itself, never a copy —
    a second literal list would drift the moment the vocabulary changed."""
    import inspect

    from dycraft.sdk import object_builder
    src = inspect.getsource(object_builder.Object.add_anchor)
    assert "AUTO_AABB_ANCHOR_NAMES" in src
    for literal in ('"left"', '"top"', '"bottom"', '"center"'):
        assert literal not in src, f"reserved set duplicated via {literal}"


# ---------------------------------------------------------------------------
# Legacy compatibility — the sharpest signal in the suite
# ---------------------------------------------------------------------------

@_needs_cad
@pytest.mark.expensive
@pytest.mark.parametrize("example,expected_id", [
    ("m1_live_shape_single.py", "5ad3b6d5224cbf34"),
    ("m1_live_shape_articulated.py", "1a30ec5a5cfbe67d"),
])
def test_legacy_objects_keep_their_exact_object_ids(example, expected_id):
    """Semantic anchors must be invisible to objects that do not use them.

    If either id moves, semantic-anchor work has leaked into the AABB-only path --
    most likely by emitting provenance for built-in anchors, or by changing how
    a built-in resolves. The two assertions below name that invariant directly;
    the hash is the wider net, catching drift they do not describe.

    The pin is the content hash of the IR without `id` and `source`. `source`
    is the content-hash path of the fixture's own text, so it moves with a
    comment edit in the fixture, which says nothing about anchors.

    **Updating these values requires evidence, not a re-run.** Any deliberate
    change to the shape table moves them for reasons that have nothing to do
    with anchors, and a pin refreshed on autopilot silently retires the check.
    Rebuild the fixture at both revisions and diff the two IRs field by field:
    move the pin only when every difference is accounted for and none is
    anchor-related.
    """
    ir = _build((Path(__file__).resolve().parent / "fixtures"
                 / example).read_text())
    # The named invariants come first, deliberately. They are the thing this
    # test is *about*; the hash is a wider net around them. Asserting the hash
    # first would let a shape-table change mask them.
    for part in ir["parts"]:
        assert "anchors" not in part, "a legacy part gained an anchors field"
    for joint in ir.get("joints") or ():
        assert "parent_anchor_ref" not in joint
        assert "child_anchor_ref" not in joint
    from dycraft.sdk.ir import content_hash
    pinned = {key: value for key, value in ir.items() if key not in ("id", "source")}
    assert content_hash(pinned)[:16] == expected_id


@_needs_cad
def test_builtin_anchors_are_never_materialised():
    """All 27 stay virtual. Persisting them would add ~27 records per part and
    churn every object id for data that is a formula over the AABB."""
    ir = _build("""
obj = Object("t")
a = obj.add_part("a", Box(0.2, 0.2, 0.2))
b = obj.add_part("b", Box(0.1, 0.1, 0.1))
obj.declare_root(a)
obj.attach_fixed("j", parent=a, child=b,
                 parent_anchor=obj.anchor(a, "top"),
                 child_anchor=obj.anchor(b, "bottom"))
ir = obj.finalize()
""")
    assert all("anchors" not in p for p in ir["parts"])


# ---------------------------------------------------------------------------
# Shared resolver
# ---------------------------------------------------------------------------

def test_resolver_returns_the_aabb_landmark_for_every_builtin():
    """The no-regression test: for all 27 names the shared resolver must agree
    with the historical formula, with identity rotation."""
    aabb = ((-0.1, -0.2, -0.3), (0.4, 0.5, 0.6))
    for name in sorted(A.AUTO_AABB_ANCHOR_NAMES):
        t, q = A.resolve_base_frame(aabb, None, name)
        assert t == A.auto_aabb_anchor_position(aabb, name), name
        assert q == (1.0, 0.0, 0.0, 0.0), name


def test_resolver_returns_the_registered_frame_for_a_semantic_anchor():
    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    q = R.sign_canonicalize_quat(
        R.normalize_quat(R.euler_xyz_deg_to_quat_wxyz(11, -47, 83)))
    anchors = [{"name": "hinge", "translation": [0.1, 0.2, 0.3],
                "rotation": list(q)}]
    t, got = A.resolve_base_frame(aabb, anchors, "hinge")
    assert np.allclose(t, (0.1, 0.2, 0.3))
    assert np.allclose(got, q)


def test_resolver_accepts_both_mapping_and_list_forms():
    """Build time holds a mapping, the IR holds a list; one lookup serves both."""
    aabb = ((0, 0, 0), (1, 1, 1))
    rec = {"name": "h", "translation": [0.5, 0, 0], "rotation": [1, 0, 0, 0]}
    assert A.resolve_base_frame(aabb, {"h": rec}, "h") == \
        A.resolve_base_frame(aabb, [rec], "h")


def test_unknown_anchor_names_the_part_and_the_candidates():
    aabb = ((0, 0, 0), (1, 1, 1))
    anchors = [{"name": "hinge", "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0]}]
    with pytest.raises(A.UnknownAnchor) as exc:
        A.resolve_base_frame(aabb, anchors, "hinnge", part_name="boom")
    msg = str(exc.value)
    assert "boom" in msg and "hinnge" in msg and "hinge" in msg


def test_unknown_anchor_never_falls_back():
    """A silent fallback to `center` would place a joint somewhere plausible
    and wrong, which is worse than failing."""
    with pytest.raises(A.UnknownAnchor):
        A.resolve_base_frame(((0, 0, 0), (1, 1, 1)), None, "no_such_anchor")


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

@_needs_cad
def test_semantic_anchor_composes_with_offset_by():
    ir = _build("""
obj = Object("t")
a = obj.add_part("a", Box(0.4, 0.2, 0.1))
b = obj.add_part("b", Box(0.2, 0.05, 0.05))
h = obj.add_anchor(a, "hinge", translation=(0.1, 0, 0), rotation=(0, 0, 90))
obj.declare_root(a)
obj.attach_fixed("j", parent=a, child=b,
                 parent_anchor=h.offset_by(translation=(0.05, 0, 0)),
                 child_anchor=obj.anchor(b, "left"))
ir = obj.finalize()
""")
    j = ir["joints"][0]
    # The base frame is rotated 90 deg about Z, so an offset of +0.05 along the
    # anchor's local X lands along the part's +Y. This is the property that
    # makes an authored frame worth having.
    assert np.allclose(j["parent_anchor"]["translation"], (0.1, 0.05, 0.0),
                       atol=1e-9), j["parent_anchor"]
    assert j["parent_anchor_ref"]["anchor"] == "hinge"
    assert "offset" in j["parent_anchor_ref"]


def test_provenance_offset_uses_a_canonical_quaternion():
    """Euler is not canonical: (180,180,180) round-trips to a different triple
    for the same frame, so an Euler provenance record could not be compared by
    value. The quaternion round-trips to ~1e-16."""
    e = (180.0, 180.0, 180.0)
    q = R.euler_xyz_deg_to_quat_wxyz(*e)
    e2 = R.quat_to_euler_xyz_deg(q)
    assert max(abs(x - y) for x, y in zip(e, e2)) > 1.0, \
        "euler round-trip is exact here; the rationale for quaternions changed"
    q2 = R.euler_xyz_deg_to_quat_wxyz(*e2)
    assert np.allclose(R.sign_canonicalize_quat(q),
                       R.sign_canonicalize_quat(q2), atol=1e-12)


# ---------------------------------------------------------------------------
# Joint provenance and drift
# ---------------------------------------------------------------------------

@_needs_cad
def test_joint_keeps_embedded_transforms_and_adds_provenance():
    ir = _build(_TWO_PART)
    j = ir["joints"][0]
    assert set(j["parent_anchor"]) == {"translation", "rotation"}
    assert j["parent_anchor_ref"] == {"anchor": "boom_mount"}
    assert j["child_anchor_ref"]["anchor"] == "base_hinge"


@_needs_cad
def test_provenance_matches_the_embedded_transform():
    K.validate_anchor_provenance(_build(_TWO_PART))


@_needs_cad
def test_tampered_embedded_transform_is_detected():
    ir = _build(_TWO_PART)
    tampered = json.loads(json.dumps(ir))
    tampered["joints"][0]["parent_anchor"]["translation"][2] += 1e-3
    with pytest.raises(K.AnchorProvenanceMismatch) as exc:
        K.validate_anchor_provenance(tampered)
    assert "boom_mount" in str(exc.value)


def test_provenance_validation_is_a_noop_without_semantic_anchors():
    K.validate_anchor_provenance(
        {"parts": [{"name": "a", "primitive": "box",
                    "params": {"width": 1, "length": 1, "height": 1}}],
         "joints": [{"name": "j", "parent": "a", "child": "a",
                     "parent_anchor": {"translation": [0, 0, 0],
                                       "rotation": [1, 0, 0, 0]}}]})


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

@_needs_cad
def test_assertion_stores_a_symbolic_semantic_reference():
    ir = _build(_TWO_PART)
    assert ir["assertions"][0]["args"]["a"] == {"entity": "a",
                                                "anchor": "boom_mount"}


@_needs_cad
def test_assertions_and_joints_resolve_a_semantic_anchor_identically():
    """One resolver, so a name means one frame. If these ever disagreed, a
    verifier would be checking a different geometry than the joint built."""
    ir = _build(_TWO_PART)
    world = P.object_anchor_position(ir, "a", "boom_mount")
    embedded = ir["joints"][0]["parent_anchor"]["translation"]
    # Part 'a' is the root, so its part-local frame is the world frame.
    assert np.allclose(world, embedded, atol=1e-9)


@_needs_cad
def test_an_anchor_can_be_used_by_a_joint_and_an_assertion_at_once():
    ir = _build(_TWO_PART)
    assert ir["joints"][0]["parent_anchor_ref"]["anchor"] == "boom_mount"
    assert ir["assertions"][0]["args"]["a"]["anchor"] == "boom_mount"


@_needs_cad
def test_an_anchor_need_not_be_consumed_by_anything():
    """`tip_mount` is referenced by no joint and no assertion. The registry is
    independently meaningful — that is the point of persisting it."""
    ir = _build(_TWO_PART)
    b = next(p for p in ir["parts"] if p["name"] == "b")
    assert "tip_mount" in [x["name"] for x in b["anchors"]]
    assert "tip_mount" not in json.dumps(ir["joints"])


# ---------------------------------------------------------------------------
# Deferred capability, pinned so it is not mistaken for supported
# ---------------------------------------------------------------------------

@_needs_cad
def test_outward_on_a_semantic_anchor_is_not_supported_yet():
    """Orientation-consuming assertions on semantic anchors are DEFERRED.

    `outward()` is defined only for the 26 face/edge/corner landmarks. It must
    fail loudly on a semantic anchor rather than return a direction derived
    from a name that has no AABB meaning.
    """
    out = W.run_object_source("""
obj = Object("t")
a = obj.add_part("a", Box(0.2, 0.2, 0.2))
h = obj.add_anchor(a, "hinge", translation=(0.1, 0, 0))
d = h.outward()
obj.declare_root(a)
ir = obj.finalize()
""", run_dir=Path(tempfile.mkdtemp()))
    assert out["ok"] is False


def test_outward_still_accepts_scene_part_scoped_anchor_paths():
    """The semantic-anchor guard must not swallow `<part>.<anchor>` paths.

    Scene anchors are legitimately part-scoped — `tabletop.front` is the
    built-in `front` landmark on an instance's `tabletop` part. A guard written
    as `name not in AUTO_AABB_ANCHOR_NAMES` rejects every one of them, which
    broke the scene prompt's own worked example: the object-layer feature
    quietly removed a scene-layer capability.

    Object anchor names are identifiers and cannot contain a dot, so the leaf
    after the last dot is the only vocabulary name in play.
    """
    from dycraft.sdk._refs import AnchorRef

    def _ref(name):
        return AnchorRef(entity_kind="instance", entity_name="i0",
                         anchor_name=name, owner_id=1)

    assert _ref("tabletop.front").outward() is not None
    assert _ref("front").outward() is not None
    for bad, why in (("tabletop.center", "center"),
                     ("center", "center"),
                     ("hinge", "semantic"),
                     ("tabletop.hinge", "semantic")):
        with pytest.raises(InvalidAnchor) as e:
            _ref(bad).outward()
        assert why in str(e.value), (bad, str(e.value))
        assert e.value.suggested_repairs, (
            "an InvalidAnchor must tell the author what to do about it")


# ---------------------------------------------------------------------------
# Reachable from the scene and animation layers, not only the object layer
# ---------------------------------------------------------------------------

def _scene_with_hinged_object():
    """A cabinet whose door carries an authored anchor, placed in a room.

    Built through the SDK rather than hand-written so the anchor travels the
    real path: `add_anchor` -> `PartIR.anchors` -> catalog -> scene query.
    """
    from dycraft.sdk import Object, ObjectHandle, Scene
    from dycraft.sdk.object_builder import _reset_object_singleton
    from dycraft.sdk.scene_builder import _reset_scene_singleton

    _reset_object_singleton()
    o = Object("cab")
    body = o.make_box("body", 0.6, 0.4, 0.8)
    door = o.make_box("door", 0.02, 0.4, 0.8)
    o.declare_root(body)
    o.add_anchor(door, "pull_point", translation=(0.0, 0.15, 0.30))
    o.attach_revolute("hinge", body, door, axis=(0.0, 0.0, 1.0),
                      limits=(0.0, 90.0), default=0.0,
                      parent_anchor=o.anchor(body, "top_front_left"),
                      child_anchor=o.anchor(door, "top_front_left"))
    cab = o.finalize()

    _reset_scene_singleton()
    s = Scene("s")
    s.set_room(width=4.0, depth=4.0, height=3.0)
    s.place(ObjectHandle(cab), name="cab", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    return s.finalize(), {cab["id"]: cab}


def test_a_declared_anchor_resolves_from_the_scene_layer():
    """`<part>.<declared>` must answer, and answer in world coordinates.

    Without it the anchors the object layer records are invisible to every
    scene and animation query, which can then only ask for the 27 AABB
    landmarks, and an agent that wants an authored grasp or contact frame has
    to re-derive the object's forward kinematics by hand.
    """
    from dycraft.sdk import scene_query

    ir, cat = _scene_with_hinged_object()
    pos = scene_query.anchor_position(ir, cat, "cab", "door.pull_point")
    assert pos is not None and len(pos) == 3
    # Inside the instance's own world box: the anchor is on the door.
    lo, hi = scene_query.instance_world_aabb(ir, cat, "cab")
    assert all(lo[k] - 5e-3 <= pos[k] <= hi[k] + 5e-3 for k in range(3)), (pos, lo, hi)


def test_a_declared_anchor_follows_its_joint():
    """The reason this cannot be an AABB landmark: a declared anchor is a point
    in the part's own frame, so it has to ride the part's pose-aware frame.
    Swinging the door must move it; a box landmark on the object-root AABB is a
    rest-only artifact that would not carry the rotation."""
    from dycraft.sdk import scene_query

    ir, cat = _scene_with_hinged_object()
    shut = scene_query.anchor_position(ir, cat, "cab", "door.pull_point")
    open_ = scene_query.anchor_position(ir, cat, "cab", "door.pull_point",
                                       poses={"hinge": 90.0})
    moved = max(abs(a - b) for a, b in zip(shut, open_))
    assert moved > 0.05, (shut, open_)


def test_the_twenty_seven_builtin_names_are_untouched():
    """The declared branch is gated on the name, so every built-in keeps its
    historical object-root-AABB answer byte for byte."""
    from dycraft.sdk import scene_query

    ir, cat = _scene_with_hinged_object()
    for anchor in ("front", "top", "door.front", "door.top_front_left",
                   "center", "door.center"):
        pos = scene_query.anchor_position(ir, cat, "cab", anchor)
        assert pos is not None and len(pos) == 3, anchor


def test_an_unknown_part_anchor_names_what_the_part_does_have():
    """The failure an agent actually hits. A bare "not an auto-AABB name" sent
    one through three probe round-trips; the resolver's message lists the
    part's registry instead."""
    from dycraft.sdk import scene_query

    ir, cat = _scene_with_hinged_object()
    with pytest.raises(Exception) as e:
        scene_query.anchor_position(ir, cat, "cab", "door.nosuch")
    assert "pull_point" in str(e.value), str(e.value)


def test_declared_anchors_can_be_enumerated_not_only_resolved():
    """A name you cannot list is a name you have to guess.

    `<part>.<declared>` resolves from the scene and animation layers, so
    `part_anchor_names` must list the declared anchors as well as the static
    27, or a shipped anchor such as the Panda's TCP
    (`panda_hand.panda_grasptarget`, 0.105 m along the hand's +Z) is reachable
    but undiscoverable, and an author measures the mesh instead.
    """
    from dycraft.sdk import _probes
    from dycraft.sdk.scene_builder import ObjectHandle

    scene_ir, catalog = _scene_with_hinged_object()
    inst = scene_ir["instances"][0]["name"]
    obj = catalog[scene_ir["instances"][0]["object_ref"]]
    part, anchor = _declared_anchor_of(obj)

    listed = _probes.scene_list_anchors(scene_ir, catalog, inst)
    assert part in listed, listed
    row = next(r for r in listed[part] if r["name"] == anchor)
    assert len(row["translation"]) == 3
    # `add_anchor` takes no `kind=`, so the NAME is the meaning, and what this
    # pins is that the declared anchor is discoverable by name at all.
    assert row["name"] == anchor

    handle = ObjectHandle(obj)
    assert anchor in handle.part_anchor_names(part)
    # The bare call keeps its old meaning: the vocabulary common to every part.
    assert anchor not in handle.part_anchor_names()
    assert len(handle.part_anchor_names()) == 27

    with pytest.raises(KeyError):
        handle.part_anchor_names("no_such_part")


def _declared_anchor_of(obj_ir):
    for p in obj_ir.get("parts") or ():
        for a in p.get("anchors") or ():
            if isinstance(a, dict) and a.get("name"):
                return p["name"], a["name"]
    raise AssertionError("fixture carries no declared anchor")
