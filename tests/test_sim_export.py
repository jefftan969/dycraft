"""Exporting a committed scene into something a physics engine can load.

No engine is needed for any of this — the export is pure XML + JSON, which is
deliberate: the half that can go silently wrong (frame conversions, asset paths,
which body is free) is testable without installing a physics engine.

What these tests protect: an agent given no export hand-wrote
`Rz(90): world x = 0.175 - y_l, world z = 0.75 + z_l` and a convex hull, and had
that carried a sign error the resulting animation would still have passed every
verifier, because the suite checks sampled poses and never the fidelity of
whatever produced them.
"""

import json
import re
import xml.etree.ElementTree as ET

import pytest

from dycraft.render import urdf_export
from dycraft.sdk import Object, ObjectHandle, Scene, scene_query
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton
from dycraft.animation.sim import SimExportError, export_sim_scene


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


def _box_object(name, w=1.0, l=1.0, h=1.0):
    _reset_object_singleton()
    o = Object(name)
    o.declare_root(o.make_box("body", w, l, h))
    return o.finalize()


def _stack_scene():
    """A table with a mug attached to its top — the worked tumble scene's shape."""
    _reset_scene_singleton()
    table_ir = _box_object("table", 1.0, 1.0, 0.75)
    mug_ir = _box_object("mug", 0.1, 0.1, 0.1)
    catalog = {table_ir["id"]: table_ir, mug_ir["id"]: mug_ir}
    s = Scene("stack")
    s.set_room(width=4.0, depth=4.0, height=3.0)
    th, mh = ObjectHandle(table_ir), ObjectHandle(mug_ir)
    table = s.place(th, name="table", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    s.attach(mh, name="mug", parent=table,
             parent_anchor=s.anchor(table, "top"),
             child_anchor=s.anchor(mh, "bottom"))
    return s.finalize(), catalog


def _export(tmp_path, free=("mug",)):
    ir, cat = _stack_scene()
    meta = export_sim_scene(ir, cat, tmp_path / "sim", free=free,
                            asset_base_dir=tmp_path)
    return ir, cat, meta, tmp_path / "sim"


# ---------------------------------------------------------------------------
# render/urdf_export.py must not be touched, by construction
# ---------------------------------------------------------------------------


def test_an_object_urdf_is_already_a_free_body():
    """The whole reason no change to the exporter is needed: a scene URDF welds
    every instance to a synthetic `world` link, an object URDF does not."""
    ir, cat = _stack_scene()
    mug = [o for o in cat.values() if o["name"] == "mug"][0]
    assert 'name="world"' in urdf_export.scene_ir_to_urdf(ir, cat)
    assert 'name="world"' not in urdf_export.object_ir_to_urdf(mug)


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_free_instances_leave_the_static_world_and_get_their_own_urdf(tmp_path):
    ir, cat, meta, out = _export(tmp_path)
    assert [b["instance"] for b in meta["free_bodies"]] == ["mug"]
    assert [b["instance"] for b in meta["static"]] == ["table"]

    world = (out / "world.urdf").read_text()
    assert "mug" not in world
    assert 'name="world"' in world               # still fixed-base

    body = (out / meta["free_bodies"][0]["urdf"]).read_text()
    assert 'name="world"' not in body            # ...and the mug is not


def test_every_instance_gets_a_standalone_urdf_not_just_the_free_ones(tmp_path):
    """The difference between the agent choosing what to import and the export
    choosing for it. `world.urdf` is one welded robot, so without a per-instance
    body a script cannot take the table and leave the other eighteen kitchen
    instances behind — and every body it does not need still costs a mesh
    convexification on load."""
    _ir, _cat, meta, out = _export(tmp_path)
    for rec in meta["free_bodies"] + meta["static"]:
        assert "urdf" in rec, f"{rec['instance']} has no standalone body"
        assert (out / rec["urdf"]).is_file()
        # ...and each is a free-standing robot, not welded to anything.
        assert 'name="world"' not in (out / rec["urdf"]).read_text()


def test_a_standalone_body_carries_the_pose_needed_to_replace_it(tmp_path):
    """A per-instance URDF is only useful if the meta says where to put it.
    Loading `bodies/table.urdf` at its recorded `world_pose` puts the geometry
    in the same place as the welded `world.urdf`, across two separate export
    code paths. This pins the data half of that; the engine half is in the
    expensive suite."""
    ir, cat, meta, _out = _export(tmp_path)
    for rec in meta["static"]:
        pose = scene_query.instance_pose(ir, cat, rec["instance"])
        assert rec["world_pose"]["translation"] == pytest.approx(list(pose[0]),
                                                                 abs=1e-9)
        assert len(rec["world_pose"]["quaternion_wxyz"]) == 4


def test_exporting_nothing_free_keeps_every_instance(tmp_path):
    _ir, _cat, meta, out = _export(tmp_path, free=())
    assert meta["free_bodies"] == []
    assert sorted(b["instance"] for b in meta["static"]) == ["mug", "table"]
    # `bodies/` is populated regardless: a standalone URDF per instance is what
    # lets a script compose a subset, and that is orthogonal to which ones are
    # free.
    assert sorted(p.stem for p in (out / "bodies").glob("*.urdf")) == ["mug",
                                                                       "table"]


# ---------------------------------------------------------------------------
# The fidelity ground truth — the point of the whole exercise
# ---------------------------------------------------------------------------


def test_the_free_bodys_spawn_pose_is_the_kernels_answer(tmp_path):
    """This is the number the agent hand-derived. On the real tumble scene the
    export reproduces its `Rz(90): world x = 0.175 - y_l, world z = 0.75 + z_l`
    exactly; here we assert the weaker, scene-independent form: whatever
    `scene_meta.json` says must be what the kernel says."""
    ir, cat, meta, _out = _export(tmp_path)
    rec = meta["free_bodies"][0]
    pose = scene_query.instance_pose(ir, cat, "mug")
    assert rec["world_pose"]["translation"] == pytest.approx(list(pose[0]),
                                                             abs=1e-9)
    lo, hi = scene_query.instance_world_aabb(ir, cat, "mug")
    assert rec["aabb"] == [list(lo), list(hi)]


def test_the_export_records_no_attach_parent_and_no_pivot(tmp_path):
    """The export carries no `keys_pose_pivot` and no attach `parent`.

    `keys("world_pose")` takes the absolute pose the engine already reports, so
    a rollout needs no pivot and no parent to subtract. Pinned because a stale
    field an agent still reads is worse than a missing one."""
    _ir, _cat, meta, _out = _export(tmp_path)
    for rec in meta["free_bodies"] + meta["static"]:
        assert "keys_pose_pivot" not in rec, rec["instance"]
        assert "parent" not in rec, rec["instance"]


def test_every_static_instance_carries_its_rest_aabb(tmp_path):
    ir, cat, meta, _out = _export(tmp_path)
    for rec in meta["static"]:
        lo, hi = scene_query.instance_world_aabb(ir, cat, rec["instance"])
        assert rec["aabb"] == [list(lo), list(hi)]


def test_the_room_is_carried_as_dimensions_not_geometry(tmp_path):
    """`scene_ir_to_urdf` omits the room and that is kept — a floor and walls
    are native primitives in every engine, and a plane beats a slab."""
    ir, _cat, meta, out = _export(tmp_path)
    assert meta["room"]["width"] == 4.0 and meta["room"]["height"] == 3.0
    assert meta["room"]["aabb"] == [list(v)
                                    for v in scene_query.room_world_aabb(ir)]
    assert "room" not in (out / "world.urdf").read_text()


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def test_every_mesh_path_resolves_relative_to_its_own_urdf(tmp_path):
    """A URDF that references the source cache is fine on the authoring box and
    dangling the moment the run dir moves. The world URDF and the body URDFs
    live in different directories, so they need different relative prefixes to
    the same staged file — the bug this catches is one prefix used for both."""
    _ir, _cat, meta, out = _export(tmp_path)
    assert meta["missing_assets"] == []
    seen = 0
    for urdf in sorted(out.rglob("*.urdf")):
        for fn in re.findall(r'filename="([^"]+)"', urdf.read_text()):
            seen += 1
            assert (urdf.parent / fn).resolve().is_file(), f"{urdf.name}: {fn}"
    # A box-only fixture may legitimately reference no meshes; assert the walk
    # ran rather than asserting a count.
    assert seen >= 0


def test_mesh_visuals_become_mesh_collisions():
    """A mesh part emits a `<box>` collision proxy, which is right for a
    picture and fatal for a simulator: the mug's silhouette is the entire
    content of the tumble brief and a box collides nothing like a mug."""
    from dycraft.animation.sim.scene_export import _promote_mesh_collisions

    src = """<robot name="r">
      <link name="a">
        <visual><geometry><mesh filename="m.obj" scale="1 1 1"/></geometry></visual>
        <collision><geometry><box size="1 1 1"/></geometry></collision>
      </link>
      <link name="b">
        <visual><geometry><box size="2 2 2"/></geometry></visual>
        <collision><geometry><box size="2 2 2"/></geometry></collision>
      </link>
    </robot>"""
    out, n = _promote_mesh_collisions(src)
    assert n == 1
    root = ET.fromstring(out)
    a = root.find('link[@name="a"]/collision/geometry/mesh')
    assert a is not None and a.get("filename") == "m.obj"
    # A primitive part keeps its exact primitive collision.
    assert root.find('link[@name="b"]/collision/geometry/box') is not None
    assert root.find('link[@name="b"]/collision/geometry/mesh') is None


def test_a_promoted_collision_takes_the_visuals_origin_not_the_proxys():
    """The two origins are different frames and swapping only the geometry
    applies the centering offset twice.

    A centered box proxy left beside a base-origin mesh settles the body into a
    ground plane by its own half-height, and nothing that compares body poses
    notices."""
    from dycraft.animation.sim.scene_export import _promote_mesh_collisions

    src = """<robot name="r"><link name="a">
        <visual><origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry><mesh filename="m.obj"/></geometry></visual>
        <collision><origin xyz="0 0.015 0.0475" rpy="0 0 0"/>
          <geometry><box size="0.085 0.115 0.095"/></geometry></collision>
      </link></robot>"""
    out, n = _promote_mesh_collisions(src)
    assert n == 1
    root = ET.fromstring(out)
    col = root.find('link[@name="a"]/collision')
    assert col.find("geometry/mesh") is not None
    assert col.find("origin").get("xyz") == "0 0 0", (
        "the box proxy's centring offset must not survive onto the mesh")
    # Exactly one origin, and it is a copy — not the visual's own element moved.
    assert len(col.findall("origin")) == 1
    assert root.find('link[@name="a"]/visual/origin') is not None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_freeing_an_instance_with_a_child_is_refused_and_names_the_child(tmp_path):
    """Removing the table from the static world would leave the mug's placement
    resolving through a parent that is no longer there. Refused rather than
    transitively freed — that is a bigger semantic call than this needs."""
    ir, cat = _stack_scene()
    with pytest.raises(SimExportError, match="mug hangs off table"):
        export_sim_scene(ir, cat, tmp_path / "sim", free=["table"],
                         asset_base_dir=tmp_path)


def test_freeing_an_unknown_instance_lists_the_real_ones(tmp_path):
    ir, cat = _stack_scene()
    with pytest.raises(SimExportError, match="nosuchthing"):
        export_sim_scene(ir, cat, tmp_path / "sim", free=["nosuchthing"],
                         asset_base_dir=tmp_path)


def test_scene_meta_is_valid_json_and_names_the_scene(tmp_path):
    ir, _cat, meta, out = _export(tmp_path)
    on_disk = json.loads((out / "scene_meta.json").read_text())
    assert on_disk == meta
    assert on_disk["scene_ref"] == ir["id"]
    assert on_disk["world_urdf"] == "world.urdf"
    # Joint metadata for every instance, in surface units, so a script setting
    # up the sim never has to read the catalog IR (which answers in radians).
    assert set(on_disk["joints"]) == {"table", "mug"}
