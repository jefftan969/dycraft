"""The glTF character importer (`agent/assets/import_gltf.py`).

A character becomes one proxy part plus a frozen native rig: a Blender armature
and one skinned mesh in `assets/rigs/<digest>.blend`, which the native
animation import swaps in for the part. These tests protect that contract from
the file to the evaluated frame.

The fixture character is made by Blender itself, so nothing is downloaded: two
bones, a body, and a foot that points the way it faces, bound to an armature
**at 0.01 object scale** with a clip that keys the root bone's location in the
armature's centimeter units -- the Mixamo shape that makes a naive
normalization move the hips 100x too far.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.agent.assets import import_gltf as ig
from dycraft.agent.assets.import_urdf import digest_bytes, watertight
from dycraft.render._blender_env import blender_child_env, blender_python
from dycraft.sdk.ir import content_hash

needs_blender = pytest.mark.skipif(
    blender_python() is None, reason="needs a configured Blender interpreter")


# ---------------------------------------------------------------------------
# Pure geometry: no Blender
# ---------------------------------------------------------------------------

def _open_box():
    """A unit cube missing its top face: one square boundary loop."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    t = np.array([[0, 2, 1], [0, 3, 2],                 # bottom
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, t


def test_capping_closes_an_open_skin_with_consistent_winding():
    v, t = _open_box()
    assert not watertight(v, t)["closed"]
    cv, ct, n = ig.cap_open_loops(v, t)
    wt = watertight(cv, ct)
    assert n == 1
    assert wt["closed"] and wt["oriented"], wt
    assert len(cv) == len(v) + 1, "one centroid per loop, nothing else added"


def test_a_closed_skin_is_its_own_collision_mesh_and_an_open_one_is_capped():
    v, t = _open_box()
    cv, ct, n = ig.cap_open_loops(v, t)
    same_v, same_t, how = ig.collision_mesh(cv, ct)
    assert how == "skin" and same_t is ct
    _v, _t, how = ig.collision_mesh(v, t)
    assert how.startswith("skin with 1 open loop")
    _v, _t, how = ig.collision_mesh(cv, ct, "hull")
    assert how == "convex hull"


def test_the_proxy_colour_is_an_area_weighted_srgb_mean():
    """Blender reports linear colour; a catalog's `color` is 8-bit sRGB."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [3, 0, 0], [0, 3, 0]], dtype=float)
    t = np.array([[0, 1, 2], [0, 4, 5]])        # areas 0.5 and 4.5
    rows = [{"color": [1.0, 0.0, 0.0], "metallic": 0.0, "roughness": 1.0,
             "alpha": 1.0},
            {"color": [0.0, 0.0, 1.0], "metallic": 1.0, "roughness": 0.0,
             "alpha": 1.0}]
    m = ig.proxy_material(rows, [0, 1], v, t)
    assert m["color"] == [ig._srgb_byte(0.1), 0, ig._srgb_byte(0.9)]
    assert m["metallic"] == pytest.approx(0.9)
    assert m["roughness"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# The fixture character, exported by Blender
# ---------------------------------------------------------------------------

_MAKE_GLB = r'''
import sys, bpy, bmesh
out = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
arm_data = bpy.data.armatures.new("Arm")
arm = bpy.data.objects.new("Arm", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
hips = arm_data.edit_bones.new("hips"); hips.head = (0, 0, 0); hips.tail = (0, 0, 50)
spine = arm_data.edit_bones.new("spine"); spine.head = (0, 0, 50); spine.tail = (0, 0, 100)
spine.parent = hips
bpy.ops.object.mode_set(mode="OBJECT")
arm.scale = (0.01, 0.01, 0.01)               # centimetre armature, as Mixamo

bm = bmesh.new()
def box(lo, hi):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r["verts"]:
        v.co.x = lo[0] if v.co.x < 0 else hi[0]
        v.co.y = lo[1] if v.co.y < 0 else hi[1]
        v.co.z = lo[2] if v.co.z < 0 else hi[2]
box((-20, -10, 10), (20, 10, 100))           # body, in armature units (cm)
box((-10, -35, 0), (10, 10, 10))             # foot, reaching toward -Y
mesh = bpy.data.meshes.new("Figure")
bm.to_mesh(mesh)
body = bpy.data.objects.new("Figure", mesh)
bpy.context.scene.collection.objects.link(body)
body.parent = arm
lower = body.vertex_groups.new(name="hips")
upper = body.vertex_groups.new(name="spine")
for v in mesh.vertices:
    (upper if v.co.z > 50 else lower).add([v.index], 1.0, "REPLACE")
body.modifiers.new("Armature", "ARMATURE").object = arm

arm.animation_data_create()
action = bpy.data.actions.new("stride")
arm.animation_data.action = action
pb = arm.pose.bones["hips"]
pb.location = (0, 0, 0); pb.keyframe_insert("location", frame=1)
# Bone space: Y runs along the bone, and `hips` points up -- so 30 cm up.
pb.location = (0, 30, 0); pb.keyframe_insert("location", frame=11)
sp = arm.pose.bones["spine"]
sp.rotation_mode = "XYZ"
sp.rotation_euler = (0, 0, 0); sp.keyframe_insert("rotation_euler", frame=1)
sp.rotation_euler = (0.6, 0, 0); sp.keyframe_insert("rotation_euler", frame=11)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB",
                          export_animations=True)
'''


@pytest.fixture(scope="module")
def character(tmp_path_factory):
    if blender_python() is None:
        pytest.skip("needs a configured Blender interpreter")
    root = tmp_path_factory.mktemp("gltf")
    script = root / "make.py"
    script.write_text(_MAKE_GLB)
    glb = root / "figure.glb"
    subprocess.run([blender_python(), str(script), str(glb)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    bundle = root / "bundle"
    ir = ig.emit_bundle(glb, bundle, name="figure")
    note = json.loads((bundle / "import.json").read_text())
    return SimpleNamespace(glb=glb, bundle=bundle, ir=ir, note=note, root=root)


@needs_blender
def test_a_character_is_one_part_with_no_joints_and_a_frozen_rig(character):
    ir = character.ir
    assert [p["name"] for p in ir["parts"]] == ["body"]
    assert ir["joints"] == [] and ir["root"] == "body"
    assert ir["parts"][0]["primitive"] == "imported"
    assert ir["id"] == content_hash({k: v for k, v in ir.items() if k != "id"})
    rig = ir["native_rig"]
    assert rig["schema"] == ig.NATIVE_RIG_SCHEMA
    assert rig["part"] == "body"
    assert (rig["armature"], rig["mesh"]) == ("rig", "body")
    blend = character.bundle / rig["blend"]
    assert rig["blend"].startswith(ig.RIGS_SUBDIR + "/")
    assert digest_bytes(blend.read_bytes()) == rig["digest"] == blend.stem
    assert [b["name"] for b in rig["bones"]] == ["hips", "spine"]
    # Body-sized bones, in meters: with Blender's default bone heuristic a
    # centimeter rig's tails land tens of meters out after the 0.01 scale.
    lengths = [float(np.linalg.norm(np.subtract(b["tail"], b["head"])))
               for b in rig["bones"]]
    assert lengths == pytest.approx([0.5, 0.5], abs=0.02)
    assert rig["root_bones"] == ["hips"]
    assert [a["name"] for a in rig["actions"]] == ["figure/stride"]


@needs_blender
def test_the_rig_blend_travels_with_the_catalog(character):
    from dycraft.agent.layers.scene_context import referenced_object_assets
    staged = {p.as_posix() for p in referenced_object_assets(character.ir)}
    assert character.ir["native_rig"]["blend"] in staged


@needs_blender
def test_the_figure_is_normalised_to_metres_floor_origin_and_plus_y(character):
    p = character.ir["parts"][0]["params"]
    lo, hi = np.array(p["aabb_min"]), np.array(p["aabb_max"])
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "feet on z=0"
    assert (lo[:2] + hi[:2]) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert hi - lo == pytest.approx([0.4, 0.45, 1.0], abs=1e-4), \
        "the 0.01 armature scale resolves to metres"
    note = character.note
    assert note["raw"]["facing"]["forward"] == "-y"
    assert note["yaw_deg"] == 180.0 and "auto" in note["yaw_from"]
    assert note["facing_after"]["forward"] == "+y"


@needs_blender
def test_location_keys_are_rescaled_with_the_armature(character):
    """The trap: applying 0.01 scale rescales rest bones but not the clip's
    bone-space location keys. The child poses every clip before and after and
    refuses a mismatch; this reads that the rescale happened and held."""
    note = character.note
    assert note["bone_space_factor"] == pytest.approx(0.01, rel=1e-5)
    assert note["rescaled_location_channels"] >= 1
    assert note["self_check"]["per_clip_m"]
    assert note["self_check"]["max_residual_m"] < 1e-4


@needs_blender
def test_the_proxy_skin_is_closed_and_collides_as_itself(character):
    assert character.note["proxy"]["skin_closed"]
    assert character.note["proxy"]["collision"] == "skin"


@needs_blender
def test_an_explicit_yaw_overrides_the_measurement(character, tmp_path):
    ig.emit_bundle(character.glb, tmp_path / "b", name="figure", yaw=0.0)
    note = json.loads((tmp_path / "b" / "import.json").read_text())
    assert note["yaw_deg"] == 0.0 and note["yaw_from"] == "--yaw"
    assert note["facing_after"]["forward"] == "-y"


@needs_blender
def test_inspect_measures_and_writes_nothing(character, tmp_path):
    before = sorted(p.name for p in character.root.iterdir())
    rep = ig.inspect(character.glb)
    assert rep["raw"]["facing"]["forward"] == "-y"
    assert rep["bones"] == 2
    assert sorted(p.name for p in character.root.iterdir()) == before


@needs_blender
def test_preload_describes_the_rig_and_refuses_a_swapped_blend(character, tmp_path):
    import shutil
    from dycraft.agent.assets.preload_asset import (
        PreloadAssetError, stage_preloaded_assets)
    run = tmp_path / "run"
    stage_preloaded_assets([f"fig={character.bundle}"], run)
    entry = json.loads((run / "preloaded" / "manifest.json").read_text())[
        "assets"]["fig"]
    rig = entry["native_rig"]
    assert rig["blender_objects"] == {"mesh": "part/<instance>/body",
                                      "armature": "rig/<instance>/body"}
    assert [a["name"] for a in rig["actions"]] == ["figure/stride"]
    assert (run / character.ir["native_rig"]["blend"]).is_file()

    tampered = tmp_path / "tampered"
    shutil.copytree(character.bundle, tampered)
    blend = tampered / character.ir["native_rig"]["blend"]
    blend.write_bytes(blend.read_bytes() + b"\0")
    with pytest.raises(PreloadAssetError, match="hashes to"):
        stage_preloaded_assets([f"fig={tampered}"], tmp_path / "run2")


@pytest.mark.expensive
@needs_blender
def test_the_native_project_deforms_the_body_with_the_armature(character, tmp_path):
    """From bundle to evaluated frame: the proxy is replaced by the rig, the
    armature sits on the part's link, the clip deforms the mesh without moving
    its object frame, and the native suite accepts it."""
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.animation.blender.verify import load_frame, verify
    from dycraft.sdk import ObjectHandle, Scene
    from dycraft.sdk.scene_builder import _reset_scene_singleton

    from dycraft.agent.assets.preload_asset import stage_preloaded_assets
    run = tmp_path / "run"
    stage_preloaded_assets([f"fig={character.bundle}"], run)
    ir = character.ir
    _reset_scene_singleton()
    s = Scene("yard")
    s.set_room(width=4.0, depth=4.0, height=3.0)
    s.place(ObjectHandle(ir), name="hero", pose=((0.5, 0.0, 0.0), (0.0, 0.0, 30.0)))
    scene, catalog = s.finalize(), {ir["id"]: ir}

    ctx = SimpleNamespace(run_dir=run, scene_ir=scene, catalog=catalog,
                          asset_base_dir=run)
    inputs, ws = tmp_path / "inputs", tmp_path / "work"
    ws.mkdir()
    spec = prepare_scene(ctx, inputs)
    assert [p["name"] for p in spec["parts"] if p.get("native_rig")] == ["hero/body"]
    ro = [inputs, run]
    run_worker(ws, tmp_path / "init", {"op": "initialize",
                                       "spec": str(inputs / "scene_spec.json")},
               readonly=ro)
    edit = run_worker(ws, tmp_path / "edit", {"op": "edit", "code": '''
s = bpy.context.scene
action = bpy.data.actions['figure/stride']
s.frame_start, s.frame_end = (int(round(f)) for f in action.frame_range)
rig = bpy.data.objects['rig/hero/body']
body = bpy.data.objects['part/hero/body']
ad = rig.animation_data_create()
ad.action = action
if ad.action_slot is None and ad.action_suitable_slots:
    ad.action_slot = ad.action_suitable_slots[0]
result['rig_parent'] = rig.parent.name
result['body_parent'] = body.parent.name
result['modifier'] = [m.object.name for m in body.modifiers if m.type == 'ARMATURE']
result['libraries'] = len(bpy.data.libraries)
'''}, readonly=ro)
    got = edit.get("result", edit)
    assert got["rig_parent"] == got["body_parent"] == "link/hero__body"
    assert got["modifier"] == ["rig/hero/body"]
    assert got["libraries"] == 0

    info = run_worker(ws, tmp_path / "eval", {"op": "evaluate"}, readonly=ro)
    _, first = load_frame(tmp_path / "eval/evaluation", 0)
    _, last = load_frame(tmp_path / "eval/evaluation", len(info["frames"]) - 1)
    a, b = first["part/hero/body"][0], last["part/hero/body"][0]
    # Placed off the origin and turned, the deformed body must sit where the
    # instance is, not where the blend left it: a mesh re-seated at the world
    # origin would part from its armature, which stays on the link.
    base = json.loads((tmp_path / "init/baseline.json").read_text())
    W = np.array(base["identities"]["hero/body"]["world_matrix"])
    assert W[:3, 3] == pytest.approx([0.5, 0.0, 0.0], abs=1e-6)
    # Compared as the box of the proxy's own vertices carried by W: the box of
    # a turned shape is not the turned box, so a center-of-AABB check would
    # report a placement error that is only geometry.
    proxy = [[float(x) for x in l.split()[1:4]] for l in
             (character.bundle / ir["parts"][0]["params"]["mesh"]).read_text().splitlines()
             if l.startswith("v ")]
    placed = np.asarray(proxy) @ W[:3, :3].T + W[:3, 3]
    assert a.min(axis=0) == pytest.approx(placed.min(axis=0), abs=0.01)
    assert a.max(axis=0) == pytest.approx(placed.max(axis=0), abs=0.01)
    rise = b[:, 2].min() - a[:, 2].min()
    assert rise == pytest.approx(0.30, abs=0.02), \
        "the 30 cm hips key must lift the figure 0.30 m, not 30 m or 3 mm"
    baseline = json.loads((tmp_path / "init/baseline.json").read_text())
    reports = verify(tmp_path / "eval/evaluation", baseline, scene, catalog)
    assert all(r["verdict"] != "fail" for r in reports), reports
    # The bone channel's inputs: the index names the rig's bones, and every
    # frame carries their evaluated orientations. `spine` turns 0.6 rad over
    # the clip, relative to `hips`. `rest` holds each bone's rest frame, from
    # which `verify.current_anatomy` derives the anatomy at check time.
    index = json.loads((tmp_path / "eval/evaluation/index.json").read_text())
    rig = index["rigs"]["rig/hero/body"]
    assert {k: rig[k] for k in ("bones", "parents")} == {"bones": ["hips", "spine"], "parents": [-1, 0]}
    assert [sorted(r) for r in rig["rest"]] == [["deform", "head", "rotation", "tail"]] * 2
    from dycraft.animation.blender.continuity import joint_rotations
    from dycraft.animation.blender.evaluation import read_record
    turns = []
    for frame in (0, len(info["frames"]) - 1):
        rows = read_record(tmp_path / "eval/evaluation", frame)["bone_rotations"]["rig/hero/body"]
        turns.append(joint_rotations(rows, [-1, 0])[1])
    angle = math.degrees(math.acos(np.clip((np.trace(turns[1] @ turns[0].T) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(math.degrees(0.6), abs=0.5)
    continuity = next(r for r in reports if r["name"] == "MotionContinuity")
    assert continuity["evidence"]["transitions_checked"] > 0


@needs_blender
def test_clips_none_imports_the_rig_without_its_clips(character, tmp_path):
    """`--clips none` keeps the mesh and rig and drops every clip, naming them:
    the way in for a rig whose clips cannot survive normalization (a
    non-uniformly scaled armature, root motion on a dropped parent)."""
    ir = ig.emit_bundle(character.glb, tmp_path / "b", name="figure", clips="none")
    note = json.loads((tmp_path / "b" / "import.json").read_text())
    assert ir["native_rig"]["actions"] == []
    assert note["clip_policy"] == "none"
    assert list(note["dropped_clips"]) == ["stride"]
    assert [b["name"] for b in ir["native_rig"]["bones"]] == ["hips", "spine"]


@needs_blender
def test_an_unrecognised_skeleton_declares_no_anatomy(character):
    """The fixture's `hips`/`spine` is no known family: the block says so, and
    AnatomicalJointLimits has nothing to check on it."""
    anatomy = character.ir["native_rig"]["anatomy"]
    assert anatomy["skeleton"] == "unknown" and anatomy["joints"] == []
    assert character.note["anatomy"]["joints"] == []


@needs_blender
def test_clips_all_is_the_default_and_keeps_matching_clips(character):
    note = character.note
    assert note["clip_policy"] == "all" and note["dropped_clips"] == {}
    # The tolerance scales with the figure (1 m here) but never below 0.1 mm.
    assert note["self_check"]["tolerance_m"] == pytest.approx(2e-4)


# An FBX that recorded its texture by an absolute path on the exporting machine,
# as Rocketbox's do: the same figure, plus an image texture, exported and then
# moved so the recorded path dangles and only the `<Name>/Export/` +
# `<Name>/Textures/` layout is left to find it by.
_MAKE_FBX = _MAKE_GLB.split("bpy.ops.export_scene.gltf(")[0] + r'''
import os
tex = os.path.join(os.path.dirname(os.path.dirname(out)), "Textures", "skin_color.png")
img = bpy.data.images.new("skin_color", 4, 4)
img.pixels = [0.8, 0.3, 0.2, 1.0] * 16
img.filepath_raw = tex; img.file_format = "PNG"; img.save()
mat = bpy.data.materials.new("skin"); mat.use_nodes = True
node = mat.node_tree.nodes.new("ShaderNodeTexImage"); node.image = img
mat.node_tree.links.new(node.outputs["Color"],
                        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
body.data.materials.append(mat)
bpy.ops.export_scene.fbx(filepath=out, path_mode="ABSOLUTE", add_leaf_bones=True)
'''


@needs_blender
def test_an_fbx_imports_and_finds_its_moved_textures(tmp_path):
    exported = tmp_path / "exported" / "Figure"
    (exported / "Export").mkdir(parents=True)
    (exported / "Textures").mkdir()
    script = tmp_path / "make.py"
    script.write_text(_MAKE_FBX)
    fbx = exported / "Export" / "Figure.fbx"
    subprocess.run([blender_python(), str(script), str(fbx)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    moved = tmp_path / "moved"
    (tmp_path / "exported").rename(moved)
    ir = ig.emit_bundle(moved / "Figure" / "Export" / "Figure.fbx", tmp_path / "b",
                        name="figure", clips="none")
    note = json.loads((tmp_path / "b" / "import.json").read_text())
    assert note["importer"] == "fbx" and note["bone_heuristic"] is None
    assert note["missing_images"] == []
    assert list(note["relinked_images"].values()) == [
        str(moved / "Figure" / "Textures" / "skin_color.png")]
    # FBX end markers are not imported as bones.
    assert [b["name"] for b in ir["native_rig"]["bones"]] == ["hips", "spine"]
    assert ir["parts"][0]["primitive"] == "imported"


@needs_blender
def test_an_fbx_clip_keeps_its_rescaled_location_keys(tmp_path):
    """Blender's FBX reader names an action's slot after the action, not the
    armature, so once the clip is unassigned nothing finds that slot: its
    centimeter hip keys would go unrescaled and the clip be refused. The
    importer names the rig's slot after the rig up front."""
    exported = tmp_path / "Figure" / "Export"
    exported.mkdir(parents=True)
    (tmp_path / "Figure" / "Textures").mkdir()
    script = tmp_path / "make.py"
    script.write_text(_MAKE_FBX)
    fbx = exported / "Figure.fbx"
    subprocess.run([blender_python(), str(script), str(fbx)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(fbx, tmp_path / "b", name="figure", clips="all")
    note = json.loads((tmp_path / "b" / "import.json").read_text())
    assert note["dropped_clips"] == {}
    assert note["rescaled_location_channels"] >= 1
    assert note["self_check"]["max_residual_m"] < 1e-4


# ---------------------------------------------------------------------------
# A native .blend: what a scene carries that a glTF does not
# ---------------------------------------------------------------------------

_MAKE_BLEND = _MAKE_GLB.split("bpy.ops.export_scene.gltf(")[0] + r'''
# The FBX reader's slot naming: after the action, not the object.
action.slots[0].name_display = "Arm|stride"
lim = arm.pose.bones["spine"].constraints.new("LIMIT_ROTATION")
lim.use_limit_x = True; lim.min_x = -0.5; lim.max_x = 0.7; lim.owner_space = "LOCAL"
# A metarig that deforms nothing, a hidden copy of the body, a camera dolly.
meta = bpy.data.objects.new("metarig", bpy.data.armatures.new("metarig"))
bpy.context.scene.collection.objects.link(meta)
copy = body.copy(); copy.data = body.data.copy(); copy.name = "Figure_hidden"
copy.hide_render = True
bpy.context.scene.collection.objects.link(copy)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.scene.collection.objects.link(cam)
cam.animation_data_create(); cam.animation_data.action = bpy.data.actions.new("dolly")
cam.location = (0, -5, 1); cam.keyframe_insert("location", frame=1)
cam.location = (0, -3, 1); cam.keyframe_insert("location", frame=11)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")         # saved mid-edit, as authors do,
bpy.context.view_layer.objects.active = cam   # and with something else selected
bpy.ops.wm.save_as_mainfile(filepath=out)
'''


@pytest.fixture(scope="module")
def blend_character(tmp_path_factory):
    if blender_python() is None:
        pytest.skip("needs a configured Blender interpreter")
    root = tmp_path_factory.mktemp("blend")
    script = root / "make.py"
    script.write_text(_MAKE_BLEND)
    src = root / "figure.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ir = ig.emit_bundle(src, root / "bundle", name="figure", clips="all")
    note = json.loads((root / "bundle" / "import.json").read_text())
    return SimpleNamespace(ir=ir, note=note)


@needs_blender
def test_a_blend_imports_the_rig_that_moves_what_renders(blend_character):
    note, rig = blend_character.note, blend_character.ir["native_rig"]
    assert note["importer"] == "blend"
    assert [b["name"] for b in rig["bones"]] == ["hips", "spine"]
    assert {"ARMATURE:metarig", "MESH:Figure_hidden", "CAMERA:cam"} <= set(note["dropped_objects"])
    assert [a["name"] for a in rig["actions"]] == ["figure/stride"], "the dolly is not a clip"
    assert note["bone_space_factor"] == pytest.approx(0.01, rel=1e-5)
    assert note["self_check"]["max_residual_m"] < 1e-4
    p = blend_character.ir["parts"][0]["params"]
    assert np.subtract(p["aabb_max"], p["aabb_min"]) == pytest.approx([0.4, 0.45, 1.0], abs=1e-4)


@needs_blender
def test_a_blends_authored_limits_stay_on_the_rig_and_are_reported(blend_character):
    limits = blend_character.note["joint_limits"]
    assert isinstance(limits, dict)
    (spine,) = [e for e in limits["bones"] if e["bone"] == "spine"]
    assert spine["kind"] == "limit_rotation" and spine["owner_space"] == "LOCAL"
    assert spine["x"] == pytest.approx([math.degrees(-0.5), math.degrees(0.7)], abs=1e-3)


# ---------------------------------------------------------------------------
# A file whose resting pose is not its bind pose
# ---------------------------------------------------------------------------

_MAKE_DEFAULTS_BLEND = r'''
import sys, bpy, bmesh
out = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
arm_data = bpy.data.armatures.new("Arm")
arm = bpy.data.objects.new("Arm", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
base = arm_data.edit_bones.new("base"); base.head = (0, 0, 0); base.tail = (0, 0, 0.2)
hips = arm_data.edit_bones.new("hips"); hips.head = (0, 0, 0.2); hips.tail = (0, 0, 0.5)
hips.parent = base
spine = arm_data.edit_bones.new("spine"); spine.head = (0, 0, 0.5); spine.tail = (0, 0, 1.0)
spine.parent = hips
bpy.ops.object.mode_set(mode="OBJECT")

bm = bmesh.new()
def box(lo, hi):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r["verts"]:
        v.co.x = lo[0] if v.co.x < 0 else hi[0]
        v.co.y = lo[1] if v.co.y < 0 else hi[1]
        v.co.z = lo[2] if v.co.z < 0 else hi[2]
box((-0.2, -0.1, 0.1), (0.2, 0.1, 1.0))
box((-0.1, -0.35, 0), (0.1, 0.1, 0.1))
mesh = bpy.data.meshes.new("Figure")
bm.to_mesh(mesh)
body = bpy.data.objects.new("Figure", mesh)
bpy.context.scene.collection.objects.link(body)
body.parent = arm
lower = body.vertex_groups.new(name="hips")
upper = body.vertex_groups.new(name="spine")
for v in mesh.vertices:
    (upper if v.co.z > 0.5 else lower).add([v.index], 1.0, "REPLACE")
body.modifiers.new("Armature", "ARMATURE").object = arm

# `nod` keys the spine; `rise` keys the hips, which `nod` leaves alone.
arm.animation_data_create()
sp, hp = arm.pose.bones["spine"], arm.pose.bones["hips"]
sp.rotation_mode = "XYZ"
nod = bpy.data.actions.new("nod"); nod.use_fake_user = True
arm.animation_data.action = nod
sp.rotation_euler = (0, 0, 0); sp.keyframe_insert("rotation_euler", frame=1)
sp.rotation_euler = (0.6, 0, 0); sp.keyframe_insert("rotation_euler", frame=11)
rise = bpy.data.actions.new("rise"); rise.use_fake_user = True
arm.animation_data.action = rise
hp.location = (0, 0, 0); hp.keyframe_insert("location", frame=1)
hp.location = (0, 0.2, 0); hp.keyframe_insert("location", frame=11)
arm.animation_data.action = None
# The resting pose as saved: `base`, which no clip keys, moved and grown 1.3x
# off the bind pose (an Objaverse cat's body bone); the hips nudged up.
sp.rotation_euler = (0, 0, 0)
hp.location = (0, 0.02, 0)
bp = arm.pose.bones["base"]
bp.location = (0, 0.05, 0.1); bp.scale = (1.3, 1.3, 1.3)
# ...and the hips, above the whole skin and keyed by `rise`, turned 90 degrees
# by default: the displayed figure is tipped over from its bind pose (the way a
# Y-up FBX's root bone stands its bind pose up, keyed in every clip).
hp.rotation_quaternion = (0.70711, 0.70711, 0.0, 0.0)
bpy.context.view_layer.update()
ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
zs = [(body.matrix_world @ v.co).z for v in ev.to_mesh().vertices]
import json
json.dump({"height": max(zs) - min(zs)}, open(out + ".json", "w"))
bpy.ops.wm.save_as_mainfile(filepath=out)
'''


@pytest.fixture(scope="module")
def defaults_character(tmp_path_factory):
    if blender_python() is None:
        pytest.skip("needs a configured Blender interpreter")
    root = tmp_path_factory.mktemp("defaults")
    script = root / "make.py"
    script.write_text(_MAKE_DEFAULTS_BLEND)
    src = root / "figure.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ir = ig.emit_bundle(src, root / "bundle", name="figure", clips="all", rest="displayed")
    note = json.loads((root / "bundle" / "import.json").read_text())
    shown = json.loads((root / "figure.blend.json").read_text())
    ig.emit_bundle(src, root / "bind", name="figure", clips="all")
    bind = json.loads((root / "bind" / "import.json").read_text())
    return SimpleNamespace(ir=ir, note=note, shown=shown, bind=bind)


@needs_blender
def test_a_rig_resting_off_its_bind_pose_plays_every_clip_as_authored(defaults_character):
    note = defaults_character.note
    assert note["dropped_clips"] == {}
    assert sorted(a["clip"] for a in note["actions"]) == ["nod", "rise"]
    assert note["self_check"]["max_residual_m"] < 1e-4
    assert note["pose_defaults"]["absorbed_into_rest"] == ["base", "hips"]
    assert note["pose_defaults"]["rebased_keyed_bones"] == ["hips"], "rise's keys re-expressed"
    assert note["pose_defaults"]["constant_keys_added"] == 0
    bind = defaults_character.bind["pose_defaults"]
    assert bind["rebased_keyed_bones"] == [] and bind["absorbed_into_rest"] == ["base"], \
        "by default a keyed bone's default is a clip's business, not the rest's"
    assert defaults_character.bind["self_check"]["max_residual_m"] < 1e-4


@needs_blender
def test_the_rest_of_such_a_rig_is_the_figure_as_displayed(defaults_character):
    p = defaults_character.ir["parts"][0]["params"]
    shown = defaults_character.shown["height"]
    assert shown < 0.9, "grown 1.3x, then tipped over"
    assert p["aabb_max"][2] - p["aabb_min"][2] == pytest.approx(shown, abs=1e-4)


# ---------------------------------------------------------------------------
# Facing: a long-bodied figure whose lowest slice points the wrong way
# ---------------------------------------------------------------------------

_MAKE_QUADRUPED_BLEND = r'''
import sys, bpy, bmesh
out = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
arm_data = bpy.data.armatures.new("Arm")
arm = bpy.data.objects.new("Arm", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
body = arm_data.edit_bones.new("spine"); body.head = (-0.5, 0, 0.55); body.tail = (0.5, 0, 0.55)
# Facing +X, the animal's right side is -Y.
for name, x, y in (("leg_front.L", 0.35, 0.1), ("leg_front.R", 0.35, -0.1),
                   ("leg_back.L", -0.35, 0.1), ("leg_back.R", -0.35, -0.1)):
    b = arm_data.edit_bones.new(name); b.head = (x, y, 0.4); b.tail = (x, y, 0.0); b.parent = body
bpy.ops.object.mode_set(mode="OBJECT")
bm = bmesh.new()
def box(lo, hi):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r["verts"]:
        v.co.x = lo[0] if v.co.x < 0 else hi[0]
        v.co.y = lo[1] if v.co.y < 0 else hi[1]
        v.co.z = lo[2] if v.co.z < 0 else hi[2]
box((-0.5, -0.15, 0.4), (0.5, 0.15, 0.7))
for x in (0.35, -0.35):
    for y in (0.1, -0.1):
        box((x - 0.04, y - 0.04, 0.0), (x + 0.04, y + 0.04, 0.4))
box((-0.95, -0.03, 0.0), (-0.5, 0.03, 0.45))   # a tail down to the ground, behind
mesh = bpy.data.meshes.new("Beast")
bm.to_mesh(mesh)
beast = bpy.data.objects.new("Beast", mesh)
bpy.context.scene.collection.objects.link(beast)
beast.parent = arm
beast.vertex_groups.new(name="spine").add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
beast.modifiers.new("Armature", "ARMATURE").object = arm
bpy.ops.wm.save_as_mainfile(filepath=out)
'''


@needs_blender
def test_a_rig_that_names_its_sides_is_turned_by_them_not_by_its_tail(tmp_path):
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND)
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["raw"]["facing"]["forward"] == "+x", note["raw"]["facing"]
    assert note["yaw_deg"] == pytest.approx(90.0)
    assert note["facing_after"]["forward"] == "+y"


@needs_blender
def test_twins_numbered_apart_by_the_exporter_still_name_the_sides(tmp_path):
    # Sketchfab numbers every bone (`LeftArm_09`, `RightArm_017`); paired only by
    # the whole name, the twins went unmatched and the foot rule turned nine
    # people round. Here the tail would turn the beast round the same way.
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND
                      .replace('"leg_front.L"', '"leg_front.L_09"').replace('"leg_front.R"', '"leg_front.R_017"')
                      .replace('"leg_back.L"', '"leg_back.L_12"').replace('"leg_back.R"', '"leg_back.R_034"'))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    note = ig.inspect(src, name="beast")
    assert note["raw"]["facing"]["forward"] == "+x", note["raw"]["facing"]
    assert "right-side bones" in note["raw"]["facing"]["why"]


@needs_blender
def test_a_figure_lying_on_its_back_is_stood_up_by_up(tmp_path):
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        "arm.rotation_euler = (3.14159265, 0, 0)       # on its back\n"
        "bpy.ops.wm.save_as_mainfile(filepath=out)"))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ir = ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none", up="-z")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["raw"]["up"] == "-z"
    assert note["facing_after"]["forward"] == "+y"
    (obj,) = (tmp_path / "bundle" / "assets" / "meshes").glob("*.obj")
    z = np.array([float(l.split()[3]) for l in obj.read_text().splitlines() if l.startswith("v ")])
    assert z.max() == pytest.approx(0.7, abs=1e-3)
    # Thin legs and tail below, the body box on top: standing, the mean vertex
    # height is 0.26 m; still on its back it would be 0.44.
    assert z.mean() < 0.35


@needs_blender
def test_a_diffuse_wired_into_alpha_is_cut_so_the_skin_stays_opaque(tmp_path):
    """An FBX converter's slip: the diffuse picture's color also drives Alpha, so
    dark clothing renders see-through. The importer cuts that link and says so."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        '''import os
img = bpy.data.images.new("skin", 8, 8)
img.pixels = [0.05, 0.1, 0.4, 1.0] * 64
img.filepath_raw = os.path.join(os.path.dirname(out), "skin.png"); img.file_format = "PNG"; img.save()
dup = img.copy()                                  # the converter's duplicate, same file
mat = bpy.data.materials.new("Skin"); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
a = nt.nodes.new("ShaderNodeTexImage"); a.image = img
b = nt.nodes.new("ShaderNodeTexImage"); b.image = dup
nt.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(b.outputs["Color"], bsdf.inputs["Alpha"])
beast.data.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=out)'''))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["material_repairs"]["color_wired_to_alpha"] == ["Skin"]
    (row,) = [m for m in note["materials"] if m["name"] == "Skin"]
    assert row["alpha"] == pytest.approx(1.0)


@needs_blender
def test_a_mirrored_part_keeps_its_mirrored_half(tmp_path):
    """A Mirror modifier is often a whole limb; joining parts would drop it."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        '''bm2 = bmesh.new()
r = bmesh.ops.create_cube(bm2, size=0.1)
bmesh.ops.translate(bm2, verts=r["verts"], vec=(0.35, 0.25, 0.55))
arm_mesh = bpy.data.meshes.new("Wing"); bm2.to_mesh(arm_mesh)
wing = bpy.data.objects.new("Wing", arm_mesh)
bpy.context.scene.collection.objects.link(wing)
wing.parent = arm
wing.vertex_groups.new(name="leg_front.L").add(list(range(8)), 1.0, "REPLACE")
mir = wing.modifiers.new("Mirror", "MIRROR"); mir.use_axis = (False, True, False)
wing.modifiers.new("Armature", "ARMATURE").object = arm
bpy.ops.wm.save_as_mainfile(filepath=out)'''))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["blend"]["applied_modifiers"] == ["Wing:MIRROR"]
    assert note["proxy"]["vertices_before_weld"] == 6 * 8 + 16, "both wings"


@needs_blender
def test_parts_whose_uv_maps_are_named_apart_keep_their_uvs_when_joined(tmp_path):
    """Mixamo names each part's UV map after the part; a join would keep the
    body's map active and leave every other part sampling its texture at 0,0."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        '''beast.data.uv_layers.new(name="BodyDiffuseUVLayer")
bm2 = bmesh.new()
r = bmesh.ops.create_cube(bm2, size=0.1)
bmesh.ops.translate(bm2, verts=r["verts"], vec=(0.55, 0.0, 0.6))
hat = bpy.data.meshes.new("Hat"); bm2.to_mesh(hat)
uv = hat.uv_layers.new(name="HatDiffuseUVLayer")
for d in uv.data:
    d.uv = (0.75, 0.25)                       # the hat's own corner of the atlas
part = bpy.data.objects.new("Hat", hat)
bpy.context.scene.collection.objects.link(part)
part.parent = arm
part.vertex_groups.new(name="spine").add(list(range(8)), 1.0, "REPLACE")
part.modifiers.new("Armature", "ARMATURE").object = arm
bpy.ops.wm.save_as_mainfile(filepath=out)'''))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert sorted(note["uv_maps_unified"]) == ["Beast", "Hat"]
    (obj,) = (tmp_path / "bundle" / "assets" / "meshes").glob("*.obj")
    vts = [tuple(map(float, l.split()[1:3])) for l in obj.read_text().splitlines() if l.startswith("vt ")]
    assert any(abs(u - 0.75) < 1e-6 and abs(v - 0.25) < 1e-6 for u, v in vts), "the hat's UVs survive"


_TRACE_BONE = r"""
import sys, json, bpy
src, bone, clip, out = sys.argv[-4:]
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False, use_scripts=False)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
act = next(a for a in bpy.data.actions if a.name.split("/")[-1] == clip)
ad = arm.animation_data or arm.animation_data_create()
ad.action = act
if getattr(act, "slots", None):
    ad.action_slot = next((s for s in act.slots if arm in s.users()), act.slots[0])
pts = []
for f in range(1, 11):
    bpy.context.scene.frame_set(f)
    pts.append(list(arm.matrix_world @ arm.pose.bones[bone].head))
spine = arm.data.bones["spine"]
json.dump({"pts": pts, "spine": (arm.matrix_world.to_3x3() @ (spine.tail_local - spine.head_local)).length},
          open(out, "w"))
"""


def _trace_bone(tmp_path, blend, bone, clip):
    script, out = tmp_path / "trace.py", tmp_path / f"trace_{blend.stem}.json"
    script.write_text(_TRACE_BONE)
    subprocess.run([blender_python(), str(script), str(blend), bone, clip, str(out)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    return json.loads(out.read_text())


@needs_blender
def test_a_skinned_bone_tree_beside_the_skeleton_is_hung_from_the_body(tmp_path):
    """Eyes rigged as root bones of their own stay in the air when the head
    bows. A tree its clip moves (an IK foot whose walk was baked into keys)
    is hung from the body too, and re-keyed so the clip plays as authored."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        """bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for name, x in (("eye", 0.52), ("ik", 0.8)):
    b = arm_data.edit_bones.new(name); b.head = (x, 0, 0.6); b.tail = (x + 0.05, 0, 0.6)
bpy.ops.object.mode_set(mode="OBJECT")
for name, x in (("eye", 0.52), ("ik", 0.8)):
    bm2 = bmesh.new()
    r = bmesh.ops.create_cube(bm2, size=0.04)
    bmesh.ops.translate(bm2, verts=r["verts"], vec=(x, 0.0, 0.6))
    me = bpy.data.meshes.new(name); bm2.to_mesh(me)
    part = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(part)
    part.parent = arm
    part.vertex_groups.new(name=name).add(list(range(8)), 1.0, "REPLACE")
    part.modifiers.new("Armature", "ARMATURE").object = arm
ik, sp = arm.pose.bones["ik"], arm.pose.bones["spine"]
sp.rotation_mode = "XYZ"
for frame, z, yaw in ((1, 0.0, 0.0), (10, 0.1, 0.35)):
    ik.location = (0.0, z, 0.0)
    ik.keyframe_insert("location", frame=frame)
    sp.rotation_euler = (0.0, yaw, 0.0)         # the body turns under the foot
    sp.keyframe_insert("rotation_euler", frame=frame)
arm.pose.bones["eye"].keyframe_insert("location", frame=1)     # keyed, but held still
arm.animation_data.action.name = "paw"
arm.animation_data.action.use_fake_user = True
arm.animation_data.action = bpy.data.actions.new("idle")          # a second clip, assigned last
for frame, z in ((1, 0.0), (10, 0.05)):
    ik.location = (0.0, z, 0.0)
    ik.keyframe_insert("location", frame=frame)
bpy.ops.wm.save_as_mainfile(filepath=out)"""))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="matching")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["attached_loose_bones"] == [
        {"root": "eye", "to": "spine", "vertices": 8},
        {"root": "ik", "to": "spine", "vertices": 8, "rekeyed_clips": ["idle", "paw"]}]
    assert note["dropped_clips"] == {}
    ir = json.loads((tmp_path / "bundle" / "ir.json").read_text())
    parent = {b["name"]: b["parent"] for b in ir["native_rig"]["bones"]}
    assert parent["eye"] == "spine" and parent["ik"] == "spine"
    # The foot's path through the clip is the source's, at the bundle's scale:
    # hung from the turning body without re-keying it would swing with it.
    (rig,) = (tmp_path / "bundle" / "assets" / "rigs").glob("*.blend")
    before, after = _trace_bone(tmp_path, src, "ik", "paw"), _trace_bone(tmp_path, rig, "ik", "paw")
    k = after["spine"] / before["spine"]
    d0 = [np.linalg.norm(np.subtract(p, before["pts"][0])) * k for p in before["pts"]]
    d1 = [np.linalg.norm(np.subtract(p, after["pts"][0])) for p in after["pts"]]
    assert max(d0) > 0.05 and np.allclose(d0, d1, atol=1e-4), (d0, d1)



_MAKE_CONTROLLER_HOOF = _MAKE_QUADRUPED_BLEND.replace(
    "bpy.ops.wm.save_as_mainfile(filepath=out)",
    """bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones
ctrl = eb.new("ctrl"); ctrl.head = (0.35, 0.1, 0.0); ctrl.tail = (0.45, 0.1, 0.0); ctrl.parent = eb["spine"]
ctrl.use_deform = False                          # an IK controller hung from the body's root, inside the skeleton
hoof = eb.new("hoof"); hoof.head = (0.35, 0.1, 0.0); hoof.tail = (0.35, 0.1, -0.04); hoof.parent = ctrl
bpy.ops.object.mode_set(mode="OBJECT")
bm2 = bmesh.new()
r = bmesh.ops.create_cube(bm2, size=0.06)
bmesh.ops.translate(bm2, verts=r["verts"], vec=(0.35, 0.1, -0.02))
me = bpy.data.meshes.new("hoof"); bm2.to_mesh(me)
part = bpy.data.objects.new("hoof", me)
bpy.context.scene.collection.objects.link(part)
part.parent = arm
part.vertex_groups.new(name="hoof").add(list(range(8)), 1.0, "REPLACE")
part.modifiers.new("Armature", "ARMATURE").object = arm
c, sp = arm.pose.bones["ctrl"], arm.pose.bones["spine"]
sp.rotation_mode = "XYZ"
for frame, x, yaw in ((1, 0.0, 0.0), (10, 0.15, 0.3)):
    c.location = (0.0, x, 0.0)                   # the walk baked into the controller's keys
    c.keyframe_insert("location", frame=frame)
    sp.rotation_euler = (0.0, yaw, 0.0)
    sp.keyframe_insert("rotation_euler", frame=frame)
arm.animation_data.action.name = "walk"
bpy.ops.wm.save_as_mainfile(filepath=out)""")


@needs_blender
def test_a_hoof_under_an_ik_controller_is_hung_from_its_leg_by_override(tmp_path):
    """A hoof skinned to a bone under an IK controller inside the skeleton stays planted when the leg is posed; no
    rule finds it safely, so a per-asset override names the leg it belongs to. The hoof then hangs from the leg,
    and the clip, which moved it through the controller, plays as authored."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_CONTROLLER_HOOF)
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    with pytest.raises(ig.CharacterImportError, match="no such bone"):
        ig.emit_bundle(src, tmp_path / "bad", name="beast", clips="matching", reparent={"hoof": "leg_front.X"})
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="matching", reparent={"hoof": "leg_front.L"})
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert {"root": "hoof", "to": "leg_front.L", "override": True, "rekeyed_clips": ["walk"]} in note["attached_loose_bones"]
    ir = json.loads((tmp_path / "bundle" / "ir.json").read_text())
    assert {b["name"]: b["parent"] for b in ir["native_rig"]["bones"]}["hoof"] == "leg_front.L"
    (rig,) = (tmp_path / "bundle" / "assets" / "rigs").glob("*.blend")
    before, after = _trace_bone(tmp_path, src, "hoof", "walk"), _trace_bone(tmp_path, rig, "hoof", "walk")
    k = after["spine"] / before["spine"]
    d0 = [np.linalg.norm(np.subtract(p, before["pts"][0])) * k for p in before["pts"]]
    d1 = [np.linalg.norm(np.subtract(p, after["pts"][0])) for p in after["pts"]]
    assert max(d0) > 0.05 and np.allclose(d0, d1, atol=1e-4), (d0, d1)


_MAKE_SKELETON_COPIES = _MAKE_QUADRUPED_BLEND.replace(
    "bpy.ops.wm.save_as_mainfile(filepath=out)",
    """import mathutils
def cube(name, centre, size):
    b = bmesh.new()
    r = bmesh.ops.create_cube(b, size=size)
    bmesh.ops.translate(b, verts=r["verts"], vec=centre)
    me = bpy.data.meshes.new(name); b.to_mesh(me)
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    return o
# The file is saved posed: the front-left leg swung 90 degrees forward, a cap parented to its bone at its foot.
leg = arm.pose.bones["leg_front.L"]
leg.rotation_mode = "XYZ"; leg.rotation_euler = (1.5708, 0.0, 0.0)
bpy.context.view_layer.update()
cap = cube("cap", arm.matrix_world @ leg.tail, 0.04)
cap.parent = arm; cap.parent_type = "BONE"; cap.parent_bone = "leg_front.L"
cap.matrix_world = mathutils.Matrix.Identity(4)
# A plate riding the armature object, no bone.
plate = cube("plate", (0.0, 0.0, 0.75), 0.1)
plate.parent = arm; plate.matrix_world = mathutils.Matrix.Identity(4)
# An identical copy of the skeleton, at rest, skinning a separate head.
copy = bpy.data.objects.new("Arm.copy", arm_data.copy()); bpy.context.scene.collection.objects.link(copy)
head = cube("head", (0.6, 0.0, 0.62), 0.1)
head.parent = copy
head.vertex_groups.new(name="spine").add(list(range(8)), 1.0, "REPLACE")
head.modifiers.new("Armature", "ARMATURE").object = copy
bpy.ops.wm.save_as_mainfile(filepath=out)""")

_CAP_AT_REST = r"""
import sys, json, bpy
import numpy as np
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2])
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
arm.data.pose_position = "REST"
body = next(o for o in bpy.data.objects if o.type == "MESH")
bpy.context.view_layer.update()
g = body.vertex_groups["leg_front.L"].index
idx = [v.index for v in body.data.vertices if any(x.group == g and x.weight > 0.99 for x in v.groups)]
ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
pts = np.array([body.matrix_world @ ev.vertices[i].co for i in idx])
tail = arm.matrix_world @ arm.data.bones["leg_front.L"].tail_local
json.dump({"n": len(idx), "gap": float(np.linalg.norm(pts.mean(axis=0) - np.array(tail))),
           "leg": float(arm.data.bones["leg_front.L"].length)}, open(sys.argv[-1], "w"))
"""


@needs_blender
def test_copies_of_one_skeleton_are_folded_and_posed_parts_kept_where_they_belong(tmp_path):
    """A file that gives a head its own copy of the body's skeleton imports as one character; a part parented to a
    bone of a rig saved posed rides that bone at rest as it did in the pose (its pose is not applied twice); a part
    riding the armature object itself is kept."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_SKELETON_COPIES)
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["merged_skeleton_copies"] == {"Arm.copy": {"skinned": ["head"], "parts": []}}
    assert note["object_parented_parts"] == ["plate"]
    (rig,) = (tmp_path / "bundle" / "assets" / "rigs").glob("*.blend")
    probe, out = tmp_path / "cap.py", tmp_path / "cap.json"
    probe.write_text(_CAP_AT_REST)
    subprocess.run([blender_python(), str(probe), str(rig), str(out)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    cap = json.loads(out.read_text())
    assert cap["n"] == 8 and cap["gap"] < 0.05 * cap["leg"], cap

_SKIN_FACTS = r"""
import sys, json, bpy
src, out = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False, use_scripts=False)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH")
deform = {g.index for g in body.vertex_groups if g.name in arm.data.bones}
unweighted = sum(1 for v in body.data.vertices if sum(g.weight for g in v.groups if g.group in deform) < 1e-4)
hair = {ps.name: ps.settings.hair_length for ps in body.particle_systems}
json.dump({"unweighted": unweighted, "hair": hair}, open(out, "w"))
"""


def _skin_facts(tmp_path, blend):
    script, out = tmp_path / "facts.py", tmp_path / "facts.json"
    script.write_text(_SKIN_FACTS)
    subprocess.run([blender_python(), str(script), str(blend), str(out)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    return json.loads(out.read_text())


@needs_blender
def test_an_unweighted_part_of_the_skin_follows_the_bone_beside_it(tmp_path):
    """A cow's eyes, added after skinning and weighted to nothing, stayed in
    the air when she lowered her head. An eye goes to the head that carries
    the lid and brow around it (a blink must not move it); a stray unweighted
    vertex in a weighted part copies its neighbor."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        """bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
head = arm_data.edit_bones.new("head"); head.head = (0.5, 0, 0.6); head.tail = (0.6, 0, 0.6)
head.parent = arm_data.edit_bones["spine"]
for name, z in (("lid", 0.65), ("brow", 0.59)):
    b = arm_data.edit_bones.new(name); b.head = (0.55, 0, z); b.tail = (0.58, 0, z); b.parent = head
bpy.ops.object.mode_set(mode="OBJECT")
bm3 = bmesh.new(); bm3.from_mesh(beast.data)
n0 = len(bm3.verts)
for z in (0.66, 0.58):                                               # lid above the eye, brow below
    r = bmesh.ops.create_cube(bm3, size=0.03)
    bmesh.ops.translate(bm3, verts=r["verts"], vec=(0.55, 0.0, z))
r = bmesh.ops.create_cube(bm3, size=0.03)
bmesh.ops.translate(bm3, verts=r["verts"], vec=(0.55, 0.0, 0.62))    # the eye, its own part
bm3.to_mesh(beast.data)
beast.vertex_groups.new(name="lid").add(list(range(n0, n0 + 8)), 1.0, "REPLACE")
beast.vertex_groups.new(name="brow").add(list(range(n0 + 8, n0 + 16)), 1.0, "REPLACE")
beast.vertex_groups["spine"].remove([3])                               # a stray vertex of the body
bpy.ops.wm.save_as_mainfile(filepath=out)"""))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none")
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert note["weighted_unweighted_vertices"] == {"vertices": 9, "whole_parts": {"head": 8},
                                                    "from_neighbours": 1}
    (rig,) = (tmp_path / "bundle" / "assets" / "rigs").glob("*.blend")
    assert _skin_facts(tmp_path, rig)["unweighted"] == 0


@needs_blender
def test_hair_is_resized_with_the_figure(tmp_path):
    """A hair system's length is in world units: a figure resized to life
    would keep the hair it had at file size (a zebra's tail longer than the
    zebra) unless the hair is resized with it."""
    script = tmp_path / "make.py"
    script.write_text(_MAKE_QUADRUPED_BLEND.replace(
        "bpy.ops.wm.save_as_mainfile(filepath=out)",
        """beast.modifiers.new("Mane", "PARTICLE_SYSTEM")
s = beast.particle_systems[0].settings
s.type = "HAIR"; s.hair_length = 0.3; s.count = 20
bpy.ops.wm.save_as_mainfile(filepath=out)"""))
    src = tmp_path / "beast.blend"
    subprocess.run([blender_python(), str(script), str(src)], check=True,
                   env=blender_child_env(), capture_output=True, timeout=300)
    ig.emit_bundle(src, tmp_path / "bundle", name="beast", clips="none", scale=2.0)
    note = json.loads((tmp_path / "bundle" / "import.json").read_text())
    assert [h["rescaled"] for h in note["particle_hair"]] == [True]
    (rig,) = (tmp_path / "bundle" / "assets" / "rigs").glob("*.blend")
    (length,) = _skin_facts(tmp_path, rig)["hair"].values()
    assert abs(length - 0.6) < 1e-6
