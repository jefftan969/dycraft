"""The URDF -> ObjectIR converter (`agent/assets/import_urdf.py`).

Hermetic: every fixture is a URDF plus meshes written into `tmp_path`. The real
robot descriptions under `assets/robots/` are gitignored and 266 MB, so nothing
here may depend on them.

The load-bearing property is that a converted IR's forward kinematics reproduces
the source URDF's, so `test_fk_matches_an_independent_urdf_fk` computes the
reference from the XML with no dycraft code in the path.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from dycraft.agent.assets import import_urdf as iu


# ---------------------------------------------------------------------------
# Rotation convention
# ---------------------------------------------------------------------------

def _Rx(t):
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                     [0, math.sin(t), math.cos(t)]])


def _Ry(t):
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                     [-math.sin(t), 0, math.cos(t)]])


def _Rz(t):
    return np.array([[math.cos(t), -math.sin(t), 0],
                     [math.sin(t), math.cos(t), 0], [0, 0, 1]])


def test_rpy_is_fixed_axis_xyz_not_the_sdk_intrinsic_helper():
    """URDF `rpy` is `Rz(y) Ry(p) Rx(r)`; the SDK's Euler helper is intrinsic
    `Rx Ry Rz`. They agree whenever at most one angle is non-zero — which is
    every single-axis test case — and disagree badly otherwise, so the contract
    is pinned on compound angles."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        r, p, y = rng.uniform(-math.pi, math.pi, 3)
        ref = _Rz(y) @ _Ry(p) @ _Rx(r)
        got = iu.quat_to_matrix(iu.rpy_to_quat(r, p, y))
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst < 1e-12

    from dycraft.sdk import _rotation as _rot
    r, p, y = -math.pi / 2, 0.3, 0.7
    intrinsic = iu.quat_to_matrix(_rot.euler_xyz_deg_to_quat_wxyz(
        math.degrees(r), math.degrees(p), math.degrees(y)))
    assert np.abs(iu.quat_to_matrix(iu.rpy_to_quat(r, p, y))
                  - intrinsic).max() > 0.5


# ---------------------------------------------------------------------------
# Mesh readers
# ---------------------------------------------------------------------------

_CUBE_OBJ = """\
v -0.5 -0.5 -0.5
v -0.5 -0.5  0.5
v -0.5  0.5 -0.5
v -0.5  0.5  0.5
v  0.5 -0.5 -0.5
v  0.5 -0.5  0.5
v  0.5  0.5 -0.5
v  0.5  0.5  0.5
f 1 2 4
f 1 4 3
f 5 7 8
f 5 8 6
f 1 5 6
f 1 6 2
f 3 4 8
f 3 8 7
f 1 3 7
f 1 7 5
f 2 6 8
f 2 8 4
"""


def _write_binary_stl(path, verts, tris):
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for i in t:
                fh.write(struct.pack("<3f", *verts[i]))
            fh.write(struct.pack("<H", 0))


def _write_ascii_stl(path, verts, tris):
    out = ["solid x"]
    for t in tris:
        out.append(" facet normal 0 0 0\n  outer loop")
        for i in t:
            out.append("   vertex %.9g %.9g %.9g" % tuple(verts[i]))
        out.append("  endloop\n endfacet")
    out.append("endsolid x")
    path.write_text("\n".join(out))


def test_obj_and_both_stl_flavours_agree_after_welding(tmp_path):
    """STL shares no vertices — every triangle carries its own three — so the
    reader's output is only comparable to the OBJ's after `weld`. A reader that
    skipped welding would make `watertight` call a closed cube an open shell."""
    obj = tmp_path / "c.obj"
    obj.write_text(_CUBE_OBJ)
    v_obj, t_obj = iu.load_obj(obj)

    _write_binary_stl(tmp_path / "b.stl", v_obj, t_obj)
    _write_ascii_stl(tmp_path / "a.stl", v_obj, t_obj)

    for name in ("b.stl", "a.stl"):
        v, t = iu.weld(*iu.load_stl(tmp_path / name))
        assert len(v) == 8 and len(t) == 12, name
        wt = iu.watertight(v, t)
        assert wt["closed"] and wt["oriented"], (name, wt)
        assert iu.mass_properties(v, t)["volume"] == pytest.approx(1.0)


def test_unwelded_stl_looks_open_which_is_why_weld_is_mandatory(tmp_path):
    obj = tmp_path / "c.obj"
    obj.write_text(_CUBE_OBJ)
    _write_binary_stl(tmp_path / "b.stl", *iu.load_obj(obj))
    raw = iu.load_stl(tmp_path / "b.stl")
    assert not iu.watertight(*raw)["closed"]


def test_collada_is_refused_by_name_not_silently_empty(tmp_path):
    """Open3D's `.dae` reader returns an empty mesh for every file in
    `assets/robots`, which would surface as a zero-volume part rather than as a
    missing feature."""
    p = tmp_path / "x.dae"
    p.write_text("<COLLADA/>")
    with pytest.raises(iu.ImportError_, match="COLLADA"):
        iu.load_mesh(p)


# ---------------------------------------------------------------------------
# URDF collision primitives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,attrib,volume", [
    ("box", {"size": "0.2 0.3 0.4"}, 0.2 * 0.3 * 0.4),
    ("cylinder", {"radius": "0.1", "length": "0.5"}, math.pi * 0.01 * 0.5),
    ("sphere", {"radius": "0.15"}, 4.0 / 3.0 * math.pi * 0.15 ** 3),
])
def test_primitive_collision_shapes_are_closed_and_measure_up(tag, attrib, volume):
    """The H1 and Go2 declare *all* of their collision volume as primitives, so
    these have to yield real closed geometry — otherwise the converter falls
    back to the visual mesh for collision."""
    v, t = iu.primitive_mesh(tag, attrib)
    wt = iu.watertight(v, t)
    assert wt["closed"] and wt["oriented"], (tag, wt)
    mp = iu.mass_properties(v, t)
    # A tessellated round shape under-reports; a box is exact.
    assert mp["volume"] == pytest.approx(volume, rel=0.05)
    assert mp["volume"] <= volume * 1.001
    assert mp["center"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_mass_properties_of_an_offset_box_finds_the_true_centroid():
    v, t = iu.primitive_mesh("box", {"size": "0.2 0.2 0.2"})
    v = v + np.array([1.0, 2.0, 3.0])
    mp = iu.mass_properties(v, t)
    assert mp["center"] == pytest.approx([1.0, 2.0, 3.0])
    m = 0.008                      # unit density * volume
    assert mp["inertia_diag"][0] == pytest.approx(m * (0.04 + 0.04) / 12.0)


def test_convex_hull_is_outward_oriented():
    """Qhull does not promise consistent winding; an unflipped hull is closed
    but not oriented, and its signed volume partially cancels."""
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(200, 3))
    v, t = iu.convex_hull(pts)
    wt = iu.watertight(v, t)
    assert wt["closed"] and wt["oriented"] and wt["euler_characteristic"] == 2
    assert iu.mass_properties(v, t)["volume_signed"] > 0.0


# ---------------------------------------------------------------------------
# End-to-end conversion
# ---------------------------------------------------------------------------

def _urdf_fixture(tmp_path, *, extra_joint="", extra_link="", frame_links=True):
    """Two real links on a revolute joint, plus (optionally) a fixed frame link
    in the middle and a fixed leaf frame — the two shapes `prune_frame_links`
    handles."""
    (tmp_path / "meshes").mkdir(exist_ok=True)
    (tmp_path / "meshes" / "cube.obj").write_text(_CUBE_OBJ)
    mid = """
  <link name="flange"/>
  <joint name="j_flange" type="fixed">
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <parent link="link1"/><child link="flange"/>
  </joint>
  <link name="tcp"/>
  <joint name="j_tcp" type="fixed">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="flange"/><child link="tcp"/>
  </joint>""" if frame_links else ""
    xml = f"""<?xml version="1.0"?>
<robot name="fix">
  <link name="link0">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry></visual>
    <collision><geometry><cylinder radius="0.5" length="1.0"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="0.1 0.2 1.0" rpy="-1.5707963 0 0.3"/>
    <parent link="link0"/><child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="2.0" effort="1" velocity="1"/>
  </joint>{mid}{extra_link}{extra_joint}
</robot>
"""
    p = tmp_path / "robot.urdf"
    p.write_text(xml)
    return p


def test_end_to_end_bundle_shape(tmp_path):
    urdf = _urdf_fixture(tmp_path)
    out = tmp_path / "out"
    res = iu.convert(urdf, out, object_name="fix", package_root=tmp_path)
    ir, rec = res["ir"], res["recipe"]

    assert (out / "ir.json").is_file() and (out / "import.json").is_file()
    assert ir["root"] == "link0"
    assert [p["primitive"] for p in ir["parts"]] == ["imported", "imported"]
    assert len(ir["joints"]) == 1                    # the two frame links went

    # every part validates through the shared structural guard, and the assets
    # it cites are on disk
    from dycraft.agent.layers.scene_context import referenced_object_assets
    from dycraft.sdk import _primitives as prim
    for p in ir["parts"]:
        prim.validate_params(p["primitive"], p["params"])
    for rel in referenced_object_assets(ir):
        assert (out / rel).is_file()

    # content hash reproduces
    from dycraft.sdk.ir import content_hash
    assert content_hash({k: v for k, v in ir.items() if k != "id"}) == ir["id"]

    # URDF material color carried onto the root part
    assert ir["parts"][0]["material"]["color"] == [255, 0, 0]
    assert rec["axis_remap"].startswith("identity")


def test_frame_links_are_spliced_and_leaves_become_anchors(tmp_path):
    """A DyCraft part is a rigid *body*; a massless URDF frame has no honest
    representation as one. An interior frame's constant transform is composed
    away and a leaf frame is kept as a semantic anchor."""
    urdf = _urdf_fixture(tmp_path)
    ir = iu.convert(urdf, tmp_path / "o", object_name="fix",
                    package_root=tmp_path)["ir"]
    assert {p["name"] for p in ir["parts"]} == {"link0", "link1"}
    anchors = {a["name"]: a for p in ir["parts"] for a in p.get("anchors", ())}
    # `tcp` hung off `flange`, which hung off `link1` at +0.5; splicing the
    # interior frame must carry the leaf to link1 at the composed 0.5 + 0.1.
    assert anchors["tcp"]["translation"] == pytest.approx([0.0, 0.0, 0.6])
    assert anchors["tcp"]["kind"] == "imported_frame"


def test_a_movable_joint_into_a_geomless_link_is_refused(tmp_path):
    """Splicing it would delete a degree of freedom."""
    extra = """
  <link name="ghost"/>
  <joint name="j_ghost" type="revolute">
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="link1"/><child link="ghost"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>"""
    urdf = _urdf_fixture(tmp_path, extra_joint=extra, frame_links=False)
    with pytest.raises(iu.ImportError_, match="degree of freedom"):
        iu.convert(urdf, tmp_path / "o", object_name="fix", package_root=tmp_path)


def test_continuous_joints_need_an_explicit_range(tmp_path):
    """URDF `continuous` is unbounded and DyCraft has no such joint. Choosing a
    range silently would attribute a limit to the asset that it never stated."""
    extra = """
  <link name="wheel">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry></visual>
    <collision><geometry><sphere radius="0.4"/></geometry></collision>
  </link>
  <joint name="spin" type="continuous">
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <parent link="link1"/><child link="wheel"/>
    <axis xyz="0 1 0"/>
  </joint>"""
    urdf = _urdf_fixture(tmp_path, extra_joint=extra, frame_links=False)
    with pytest.raises(iu.ImportError_, match="continuous-range"):
        iu.convert(urdf, tmp_path / "a", object_name="fix",
                   package_root=tmp_path)

    ir = iu.convert(urdf, tmp_path / "b", object_name="fix",
                    package_root=tmp_path, continuous_range_deg=180.0)["ir"]
    spin = next(j for j in ir["joints"] if j["name"] == "spin")
    assert spin["type"] == "revolute"
    assert spin["limits"] == pytest.approx([-math.pi / 2, math.pi / 2])


def test_default_pose_is_surface_units_and_bounds_checked(tmp_path):
    urdf = _urdf_fixture(tmp_path)
    ir = iu.convert(urdf, tmp_path / "a", object_name="fix",
                    package_root=tmp_path,
                    default_pose={"j1": 45.0})["ir"]
    assert ir["joints"][0]["default"] == pytest.approx(math.radians(45.0))

    with pytest.raises(iu.ImportError_, match="outside the URDF limit"):
        iu.convert(urdf, tmp_path / "b", object_name="fix",
                   package_root=tmp_path, default_pose={"j1": 200.0})


def test_default_pose_changes_the_content_hash(tmp_path):
    """The rest pose is part of the object's identity, which is why it cannot be
    patched into a frozen bundle afterwards."""
    urdf = _urdf_fixture(tmp_path)
    a = iu.convert(urdf, tmp_path / "a", object_name="fix",
                   package_root=tmp_path)["ir"]
    b = iu.convert(urdf, tmp_path / "b", object_name="fix",
                   package_root=tmp_path, default_pose={"j1": 30.0})["ir"]
    assert a["id"] != b["id"]


def test_conversion_is_deterministic(tmp_path):
    """Same URDF and same flags must produce the same content hash, or the
    gitignored asset library cannot be rebuilt from `import.json`."""
    urdf = _urdf_fixture(tmp_path)
    ids = {iu.convert(urdf, tmp_path / f"o{i}", object_name="fix",
                      package_root=tmp_path)["ir"]["id"] for i in range(3)}
    assert len(ids) == 1


def test_collision_borrowed_from_visual_is_hulled_and_recorded(tmp_path):
    """A link with no `<collision>` must not put its full-detail visual mesh in
    the narrow phase — the G1's `left_rubber_hand` visual is 45,748 triangles."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "cube.obj").write_text(_CUBE_OBJ)
    (tmp_path / "robot.urdf").write_text("""<?xml version="1.0"?>
<robot name="nc">
  <link name="only">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry></visual>
  </link>
</robot>
""")
    res = iu.convert(tmp_path / "robot.urdf", tmp_path / "o",
                     object_name="nc", package_root=tmp_path)
    part = res["recipe"]["parts"][0]
    assert part["collision"]["borrowed_from_visual"] is True
    assert part["collision"]["convex_hulled"] is True


def test_package_uri_resolves_by_walking_up_to_the_package_dir(tmp_path):
    """A fixed number of `.parent`s is right for one layout and doubles the
    package name for the other: the Panda's URDF is at `<pkg>/panda.urdf`, the
    H1's at `<pkg>/urdf/h1.urdf`."""
    pkg = tmp_path / "robots" / "my_robot"
    (pkg / "meshes").mkdir(parents=True)
    (pkg / "urdf").mkdir()
    (pkg / "meshes" / "cube.obj").write_text(_CUBE_OBJ)
    urdf = pkg / "urdf" / "r.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="r"><link name="a">
  <visual><geometry><mesh filename="package://my_robot/meshes/cube.obj"/></geometry></visual>
</link></robot>
""")
    got = iu._resolve_mesh("package://my_robot/meshes/cube.obj", urdf,
                           urdf.parent.parent)
    assert got == (pkg / "meshes" / "cube.obj").resolve()
    # and the whole conversion works from that layout
    ir = iu.convert(urdf, tmp_path / "o", object_name="r",
                    package_root=urdf.parent.parent)["ir"]
    assert len(ir["parts"]) == 1


# ---------------------------------------------------------------------------
# The load-bearing property
# ---------------------------------------------------------------------------

def test_fk_matches_an_independent_urdf_fk(tmp_path):
    """The converted IR's kinematics must reproduce the source URDF's.

    The reference is computed from the XML here, with no dycraft code in the
    path, so an error shared between converter and kernel cannot hide. This is
    what catches an rpy-convention slip or a wrong anchor assignment — both of
    which leave every part individually plausible and the assembly wrong.
    """
    import xml.etree.ElementTree as ET

    from dycraft.sdk import _object_kernel as ok

    urdf = _urdf_fixture(tmp_path)
    ir = iu.convert(urdf, tmp_path / "o", object_name="fix",
                    package_root=tmp_path)["ir"]

    root = ET.parse(urdf).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        M = np.eye(4)
        M[:3, :3] = iu.quat_to_matrix(iu.rpy_to_quat(*rpy))
        M[:3, 3] = xyz
        joints[j.get("name")] = (j.get("type"), j.find("parent").get("link"),
                                 j.find("child").get("link"), M,
                                 np.array([float(v) for v in a.get("xyz").split()])
                                 if a is not None and a.get("xyz") else None)

    def axis_angle(ax, q):
        ax = ax / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)

    def urdf_fk(q):
        frames = {"link0": np.eye(4)}
        for _ in range(8):
            for jt, parent, child, M, ax in joints.values():
                if parent in frames and child not in frames:
                    T = M.copy()
                    if jt == "revolute":
                        R = np.eye(4)
                        R[:3, :3] = axis_angle(ax, q)
                        T = T @ R
                    frames[child] = frames[parent] @ T
        return frames

    for q in (-1.0, -0.4, 0.0, 0.7, 2.0):
        ref = urdf_fk(q)
        got = ok._compute_frames(ir, poses={"j1": q})
        for name in ("link0", "link1"):
            M = np.eye(4)
            M[:3, :3] = iu.quat_to_matrix(got[name][1])
            M[:3, 3] = got[name][0]
            assert np.abs(ref[name] - M).max() < 1e-12, (q, name)


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

#: 1x1 opaque PNG. Real bytes rather than a stub, because `stage_texture` names
#: the file by content digest and `_IMAGE_MIME` keys off the suffix.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63f8ffff3f0005fe02fea735a2b5"
    "0000000049454e44ae426082")

#: The cube again, but with texture coordinates — and with the last two faces
#: deliberately written `v//vn`, the shape Franka's `hand.obj` and `link5.obj`
#: use for part of their triangles. A reader that assumed every face carries a
#: `vt` would drop the whole mesh's UVs on those two lines.
_CUBE_OBJ_UV = _CUBE_OBJ.replace(
    "f 1 2 4", "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\nf 1/1 2/2 4/4"
).replace("f 1 4 3", "f 1/1 4/4 3/3") \
 .replace("f 5 7 8", "f 5/1 7/3 8/4").replace("f 5 8 6", "f 5/1 8/4 6/2") \
 .replace("f 1 5 6", "f 1/2 5/2 6/4").replace("f 1 6 2", "f 1/2 6/4 2/3") \
 .replace("f 3 4 8", "f 3/1 4/2 8/4").replace("f 3 8 7", "f 3/1 8/4 7/3") \
 .replace("f 1 3 7", "f 1/3 3/2 7/4").replace("f 1 7 5", "f 1/3 7/4 5/3") \
 .replace("f 2 6 8", "f 2//1 6//1 8//1").replace("f 2 8 4", "f 2//1 8//1 4//1")


def _textured_obj(dir_: Path, *, stem="cube", image="colors.png",
                  mtl_on_disk=True, extra_map=None):
    """Write `<stem>.obj` + its MTL + the image(s), and return the OBJ path."""
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{stem}.obj").write_text(
        f"mtllib {stem}.mtl\n" + _CUBE_OBJ_UV)
    if mtl_on_disk:
        lines = [f"newmtl m\nmap_Kd {image}\n"]
        if extra_map:
            lines.append(f"newmtl m2\nmap_Kd {extra_map}\n")
            (dir_ / extra_map).write_bytes(_PNG + b"\0")   # a *different* image
        (dir_ / f"{stem}.mtl").write_text("".join(lines))
    (dir_ / image).write_bytes(_PNG)
    return dir_ / f"{stem}.obj"


def test_uvs_are_per_corner_and_faces_without_vt_do_not_poison_the_mesh(tmp_path):
    """Per-corner is the whole reason texture survives `weld`: a welded cube
    corner has one position and three different UVs, so a per-vertex array
    could not represent it. A `v//vn` face is tolerated with a (0,0) sentinel
    rather than discarding every other face's coordinates."""
    obj = _textured_obj(tmp_path / "m")
    v, t, uv = iu.load_obj(obj, with_uvs=True)
    assert uv.shape == (len(t), 3, 2)
    assert len(v) == 8 and len(t) == 12

    # the two `v//vn` faces got the sentinel; the ten `v/vt` ones did not
    zero = np.all(uv == 0.0, axis=2).all(axis=1)
    assert zero.sum() == 2, zero

    # vertex 1 (index 0) appears with three distinct UVs across its corners —
    # unrepresentable per-vertex
    seen = {tuple(uv[i, k]) for i in range(len(t)) for k in range(3)
            if t[i][k] == 0}
    assert len(seen) == 3, seen

    # the 2-tuple form is unchanged for callers that unpack two values
    assert len(iu.load_obj(obj)) == 2


def test_uvs_survive_the_weld_because_the_weld_only_renumbers_v(tmp_path):
    obj = _textured_obj(tmp_path / "m")
    v, t, uv = iu.load_obj(obj, with_uvs=True)
    v2, t2, keep = iu.weld(v, t, return_keep=True)
    assert len(keep) == len(t) and len(t2) == keep.sum()
    assert uv[keep].shape == (len(t2), 3, 2)


def test_read_obj_texture_follows_mtllib_then_map_kd(tmp_path):
    obj = _textured_obj(tmp_path / "m")
    tex, note = iu.read_obj_texture(obj)
    assert note is None
    assert tex["mime"] == "image/png" and tex["base_color"] == _PNG


def test_a_missing_mtl_is_reported_not_swallowed(tmp_path):
    """Franka's `link0`/`link7` name their collision OBJ as their visual, and
    those two cite MTLs the pybullet distribution does not ship. A flat color
    is the right output; a *silent* flat color is not, because it is
    indistinguishable from a design choice."""
    obj = _textured_obj(tmp_path / "m", mtl_on_disk=False)
    tex, note = iu.read_obj_texture(obj)
    assert tex is None
    assert note and "not on disk" in note


def test_two_different_map_kd_images_are_refused_because_one_mesh_has_one_uv_set(
        tmp_path):
    obj = _textured_obj(tmp_path / "m", extra_map="other.png")
    tex, note = iu.read_obj_texture(obj)
    assert tex is None
    assert note and "UV set" in note


def test_no_mtllib_at_all_is_not_a_loss(tmp_path):
    plain = tmp_path / "c.obj"
    plain.write_text(_CUBE_OBJ)
    assert iu.read_obj_texture(plain) == (None, None)


def test_stage_texture_is_content_addressed_and_idempotent(tmp_path):
    a = iu.stage_texture(tmp_path, {"base_color": _PNG, "mime": "image/png"})
    b = iu.stage_texture(tmp_path, {"base_color": _PNG, "mime": "image/png"})
    assert a == b and a.startswith("assets/textures/") and a.endswith(".png")
    assert (tmp_path / a).read_bytes() == _PNG
    assert iu.stage_texture(tmp_path, None) is None


def _textured_urdf(tmp_path):
    _textured_obj(tmp_path / "meshes")
    (tmp_path / "robot.urdf").write_text("""<?xml version="1.0"?>
<robot name="tex">
  <link name="link0">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry>
      <material name="grey"><color rgba="0.8 0.8 0.8 1"/></material></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="meshes/bare.obj"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <parent link="link0"/><child link="link1"/>
  </joint>
</robot>
""")
    (tmp_path / "meshes" / "bare.obj").write_text(_CUBE_OBJ)   # no mtllib
    return tmp_path / "robot.urdf"


def test_a_textured_visual_mesh_reaches_the_material_and_the_staged_bundle(
        tmp_path):
    """End to end: the image is staged content-addressed, the part names it
    through `material.texture.base_color`, the staged OBJ carries `vt`, and
    `referenced_object_assets` lists it so a catalog cannot be built without
    it. This is the same representation `import_gltf` emits, which is what lets
    one renderer serve a robot and a character."""
    from dycraft.agent.layers.scene_context import referenced_object_assets

    out = tmp_path / "out"
    res = iu.convert(_textured_urdf(tmp_path), out, object_name="tex",
                     package_root=tmp_path)
    ir, rec = res["ir"], res["recipe"]
    p0, p1 = ir["parts"]

    # The shared shape: base color is a map kind carrying image + digest,
    # and the digest is the staged file's own stem by construction.
    base = p0["material"]["texture"]["base_color"]
    rel = base["image"]
    assert rel.startswith("assets/textures/")
    assert base["digest"] == Path(rel).stem
    # No `mapping`: an imported asset's UVs are authored into its OBJ, so there
    # is no tile size, and its absence is what stops a renderer re-projecting.
    assert "mapping" not in p0["material"]
    assert (out / rel).read_bytes() == _PNG
    assert rel in {str(x) for x in referenced_object_assets(ir)}

    # the visual OBJ gained `vt` + `f v/vt`; the collision OBJ did not, because
    # containment has nothing to sample
    vis = (out / p0["params"]["mesh"]).read_text()
    assert "\nvt " in vis and "/" in [l for l in vis.splitlines()
                                      if l.startswith("f ")][0]
    assert "\nvt " not in (out / p0["params"]["collision"]).read_text()

    # color stays as the URDF declared it: a fallback for a consumer that
    # cannot sample, not a tint
    assert p0["material"]["color"] == [204, 204, 204]

    # a link whose mesh has no material gets no `texture` key at all — absent,
    # not null, because a key that is always present would change every
    # existing material's canonical bytes
    assert "texture" not in p1["material"]

    by_link = {p["link"]: p["texture"] for p in rec["parts"]}
    assert by_link["link0"]["file"] == rel
    assert by_link["link0"]["uv_corners"] == \
        p0["params"]["shape"]["n_triangles"]
    assert by_link["link1"] == {"staged": False, "why": None}


def _two_body_urdf(tmp_path, *, visual_obj, collision_box):
    """One link whose visual and collision are deliberately different bodies."""
    (tmp_path / "meshes").mkdir(exist_ok=True)
    (tmp_path / "meshes" / "v.obj").write_text(visual_obj)
    p = tmp_path / "robot.urdf"
    p.write_text(f"""<?xml version="1.0"?>
<robot name="two">
  <link name="base">
    <visual><geometry><mesh filename="meshes/v.obj"/></geometry></visual>
    <collision><geometry><box size="{collision_box}"/></geometry></collision>
  </link>
</robot>
""")
    return p


def test_the_collision_escape_is_measured_and_only_grows_bounds(tmp_path):
    """The stored box is the visual body's; the narrow phase measures the other.

    A landmark wants to sit ON the visible surface and a bound wants to lie
    outside everything measurable, so one box cannot be both — and the box is
    the visual one, because moving anchors off the skin is the worse failure.
    That leaves the broad phase bounding a body it does not measure: on the
    committed Panda the collision mesh escapes on 9 of 11 parts, worst
    `panda_link5` by 5.554 mm, which breaks the prune's own stated contract in
    `_probes._gap_candidates`.

    So the escape is recorded and added to the *bound* producers only. This
    asserts both halves: the bound grows, and `local_bbox` — which feeds the
    27-name anchor vocabulary — does not.
    """
    urdf = _two_body_urdf(tmp_path, visual_obj=_CUBE_OBJ, collision_box="1.02 1 1")
    ir = iu.convert(urdf, tmp_path / "out", object_name="two",
                    package_root=tmp_path)["ir"]
    params = ir["parts"][0]["params"]
    assert params["collide_escape_m"] == pytest.approx(0.01)

    from dycraft.sdk import _object_kernel as ok
    from dycraft.sdk import _primitives as prim

    prim.validate_params("imported", params)          # optional key accepted
    lo, hi = ok.local_bbox(ir, "base")
    assert lo == pytest.approx([-0.5, -0.5, -0.5])    # anchor path untouched
    assert hi == pytest.approx([0.5, 0.5, 0.5])

    glo, ghi = ok._inflate_local_box((lo, hi), None,
                                     margin=ok.collide_margin(ir["parts"][0]))
    assert glo == pytest.approx([-0.51, -0.51, -0.51])
    assert ghi == pytest.approx([0.51, 0.51, 0.51])
    # A part whose proxy sits inside its visual box needs no margin at all.
    assert ok.collide_margin({"params": {}}) == 0.0


def test_mass_from_visual_hulls_an_open_shell_rather_than_borrowing_the_proxy(
        tmp_path):
    """`--mass-from visual` must not silently answer about the other body.

    Under the rule that the visual mesh is the part, falling back to the
    collision proxy would make one `shape` table describe two bodies per part
    and record which only in `import.json`. The hull of the visual is an
    over-estimate of the right body; the proxy is a measurement of the wrong
    one — and on the G1's foot the proxy is a 10 mm sole plate holding 0.7% of
    the visible volume, which is a 143x gap that no diagnostic fires on because
    both meshes are closed.
    """
    open_cube = "\n".join(l for l in _CUBE_OBJ.splitlines()
                          if l not in ("f 2 6 8", "f 2 8 4")) + "\n"
    urdf = _two_body_urdf(tmp_path, visual_obj=open_cube, collision_box="1 1 1")

    rec = iu.convert(urdf, tmp_path / "vis", object_name="two",
                     package_root=tmp_path, mass_from="visual")["recipe"]
    assert rec["parts"][0]["visual"]["closed"] is False
    assert rec["parts"][0]["mass_from"] == "visual_hull"

    # The `collision` preference still falls back the other way, unchanged.
    rec2 = iu.convert(urdf, tmp_path / "col", object_name="two",
                      package_root=tmp_path, mass_from="collision")["recipe"]
    assert rec2["parts"][0]["mass_from"] == "collision"


# ---------------------------------------------------------------------------
# Rest-pose structural collision allowances
# ---------------------------------------------------------------------------

def _overlapping_urdf(tmp_path):
    """Two links whose shells interpenetrate at the rest pose.

    The joint origin sits well inside the parent's 1 m cube, so the child's
    collision box lies partly within it — which is what a real bearing looks
    like, and what `--self-collision-allowances rest-pose` exists to declare.
    """
    (tmp_path / "meshes").mkdir(exist_ok=True)
    (tmp_path / "meshes" / "cube.obj").write_text(_CUBE_OBJ)
    xml = """<?xml version="1.0"?>
<robot name="nested">
  <link name="base">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="arm">
    <visual><geometry><mesh filename="meshes/cube.obj"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
</robot>
"""
    p = tmp_path / "nested.urdf"
    p.write_text(xml)
    return p


def test_rest_pose_overlap_is_declared_as_structure(tmp_path):
    """A child nested in its parent at rest is how the asset is built, not a
    defect the importer should leave for every downstream verifier to
    rediscover."""
    urdf = _overlapping_urdf(tmp_path)
    res = iu.convert(urdf, tmp_path / "out", object_name="nested",
                     package_root=tmp_path)
    allowances = res["ir"]["collision_allowances"]
    assert allowances, "the nested pair must be declared"
    pair = {frozenset((r["a"], r["b"])) for r in allowances}
    assert frozenset({"base", "arm"}) in pair
    reason = next(r["reason"] for r in allowances
                  if frozenset((r["a"], r["b"])) == frozenset({"base", "arm"}))
    # The measurement is in the record, so a reader can tell a bearing from a
    # modeling error without re-running anything.
    assert "rest pose" in reason and "mm" in reason
    assert res["recipe"]["self_collision_allowance_policy"] == "rest-pose"
    assert res["recipe"]["converter_version"] == 2


def test_policy_none_reproduces_the_older_bundle(tmp_path):
    """`none` is the escape hatch, and it must add no allowances at all:
    allowances are inside the content hash, so this is also the only way to
    reconvert an asset and keep its id."""
    urdf = _overlapping_urdf(tmp_path)
    res = iu.convert(urdf, tmp_path / "out", object_name="nested",
                     package_root=tmp_path, self_collision_allowances="none")
    assert res["ir"]["collision_allowances"] == []


def test_declaring_an_allowance_changes_the_object_id(tmp_path):
    """Not a wart — a bundle that allows an overlap is a different input from
    one that does not, and `id` is a content hash. Pinned so nobody 'fixes' it
    by excluding allowances from the hash."""
    urdf = _overlapping_urdf(tmp_path)
    with_it = iu.convert(urdf, tmp_path / "a", object_name="nested",
                         package_root=tmp_path)["ir"]
    without = iu.convert(urdf, tmp_path / "b", object_name="nested",
                         package_root=tmp_path,
                         self_collision_allowances="none")["ir"]
    assert with_it["id"] != without["id"]


def test_a_clear_rest_pose_declares_nothing(tmp_path):
    """The policy must not blanket-allow adjacency. Links that do not touch at
    rest stay checked, so a real clash at a swept pose is still reported."""
    urdf = _urdf_fixture(tmp_path)
    res = iu.convert(urdf, tmp_path / "out", object_name="fix",
                     package_root=tmp_path)
    assert res["ir"]["collision_allowances"] == []


def test_the_declared_pairs_stop_failing_self_penetration(tmp_path):
    """The point of the exercise, end to end."""
    from dycraft.verify.object_verify import OBJECT_VERIFIERS
    check = dict(OBJECT_VERIFIERS)["NoSelfPenetration"]
    urdf = _overlapping_urdf(tmp_path)
    plain = iu.convert(urdf, tmp_path / "a", object_name="nested",
                       package_root=tmp_path,
                       self_collision_allowances="none")["ir"]
    declared = iu.convert(urdf, tmp_path / "b", object_name="nested",
                          package_root=tmp_path)["ir"]
    before = {frozenset((p["a"], p["b"]))
              for p in check(plain, None).evidence["pairs"]}
    after = {frozenset((p["a"], p["b"]))
             for p in check(declared, None).evidence["pairs"]}
    assert frozenset({"base", "arm"}) in before
    assert frozenset({"base", "arm"}) not in after


def test_upgrade_bundle_rederives_allowances_without_the_source_urdf(tmp_path):
    """A bundle outlives the checkout that made it. `upgrade_bundle` reads the
    geometry the bundle already carries, so an asset whose `source_urdf` points
    at an unreachable machine can still be brought forward."""
    urdf = _overlapping_urdf(tmp_path)
    old = iu.convert(urdf, tmp_path / "v1", object_name="nested",
                     package_root=tmp_path,
                     self_collision_allowances="none")["ir"]
    res = iu.upgrade_bundle(tmp_path / "v1", tmp_path / "v2")
    new = res["ir"]

    assert len(new["collision_allowances"]) >= 1
    assert new["id"] != old["id"]
    assert res["recipe"]["upgraded_from"]["ir_id"] == old["id"]
    # A new bundle, never an edit in place: the original keeps its identity.
    assert json.loads((tmp_path / "v1" / "ir.json").read_text())["id"] == old["id"]
    # And the new one is internally consistent, so `--preload-asset` accepts it.
    from dycraft.agent.assets.preload_asset import _load_and_verify
    assert _load_and_verify(tmp_path / "v2", "nested")["id"] == new["id"]


def test_upgrade_refuses_a_bundle_that_already_declares_allowances(tmp_path):
    """Recomputing over an existing set would silently discard an author's
    reasons, or double a converter's own work."""
    urdf = _overlapping_urdf(tmp_path)
    iu.convert(urdf, tmp_path / "v2", object_name="nested",
               package_root=tmp_path)
    with pytest.raises(iu.ImportError_, match="already declares"):
        iu.upgrade_bundle(tmp_path / "v2", tmp_path / "v3")
