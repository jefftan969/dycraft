"""`python -m dycraft.export glb`: what it exports from each kind of input, and what the GLB holds.

The input tests need no Blender: they pin which file a run, a reader-layout run, a published film folder, or a
`.blend` exports, and the one-line refusals. The tests marked `expensive` run Blender for real: a small film with
rigid motion, a deforming mesh, a camera cut, and a procedural material, and a one-part object built from its plan.
Each checks the GLB itself (its JSON chunk) and that Blender imports it back.
"""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

from dycraft.export import __main__ as cli
from dycraft.export import glb as G


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _glb_json(path: Path) -> dict:
    data = path.read_bytes()
    assert data[:4] == b"glTF", "not a GLB"
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


# ---------------------------------------------------------------------------------------------------------------- inputs


def test_a_blend_file_exports_beside_itself(tmp_path):
    blend = _touch(tmp_path / "shot.blend")
    source = G.resolve(blend)
    assert (source.kind, source.blend, source.out_dir, source.name) == ("animation", blend, tmp_path, "shot")


@pytest.mark.parametrize("rel", ["animation.blend", "src/bundle/animation.blend", "bundle/animation.blend"])
def test_a_film_is_found_in_every_run_layout(tmp_path, rel):
    """The reader layout's top-level blend, its sealed bundle, and an older run's or a published film's bundle."""
    run = tmp_path / "run"
    blend = _touch(run / rel)
    if rel.startswith("src/"):
        _touch(run / "src" / "meta.json", "{}")
    source = G.resolve(run)
    assert source.kind == "animation" and source.blend == blend.resolve()
    assert source.out_dir == run.resolve(), "the GLB goes in the reader's folder, never inside src/"


def test_the_top_level_blend_wins_over_the_sealed_one(tmp_path):
    run = tmp_path / "run"
    top = _touch(run / "animation.blend")
    _touch(run / "src" / "meta.json", "{}")
    _touch(run / "src" / "bundle" / "animation.blend")
    assert G.resolve(run).blend == top.resolve()


@pytest.mark.parametrize("kind,ir", [("object", {"id": "o", "parts": []}), ("scene", {"id": "s", "instances": []})])
def test_object_and_scene_runs_export_their_committed_ir(tmp_path, kind, ir):
    for inner in (tmp_path / "flat", tmp_path / "reader" / "src"):
        _touch(inner / "ir.json", json.dumps(ir))
        _touch(inner / "meta.json", json.dumps({"layer": kind}))
    flat, reader = G.resolve(tmp_path / "flat"), G.resolve(tmp_path / "reader")
    assert (flat.kind, flat.run_dir) == (kind, (tmp_path / "flat").resolve())
    assert (reader.kind, reader.run_dir, reader.out_dir) == (
        kind, (tmp_path / "reader" / "src").resolve(), (tmp_path / "reader").resolve())


@pytest.mark.parametrize("setup,needle", [
    (lambda d: _touch(d / "meta.json", json.dumps({"layer": "plain"})), "plain run"),
    (lambda d: (_touch(d / "meta.json", json.dumps({"layer": "animation"})), (d / "native").mkdir()),
     "no delivered film yet"),
    (lambda d: d.mkdir(parents=True), "nothing to export"),
])
def test_runs_with_nothing_to_export_are_refused_in_one_line(tmp_path, setup, needle):
    run = tmp_path / "run"
    setup(run)
    with pytest.raises(G.ExportError, match=needle):
        G.resolve(run)


def test_the_command_reports_a_refusal_and_exits_1(tmp_path, capsys):
    assert cli.main(["glb", str(tmp_path / "missing")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("[export] ") and "does not exist" in err and len(err.strip().splitlines()) == 1


def test_the_browser_step_names_what_it_needs(tmp_path, monkeypatch):
    """Without Node it says so; without its packages it gives the one command that installs them."""
    monkeypatch.setattr(G.shutil, "which", lambda name: None)
    with pytest.raises(G.ExportError, match="Node.js 20"):
        G._web(tmp_path / "a.glb", tmp_path / "a.web.glb", tmp_path)
    monkeypatch.setattr(G.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(G, "WEB_DIR", tmp_path / "web")
    with pytest.raises(G.ExportError, match=r"npm --prefix .* ci"):
        G._web(tmp_path / "a.glb", tmp_path / "a.web.glb", tmp_path)


def test_the_web_scripts_and_their_lockfile_ship_together():
    """`--web` runs these two scripts from the package; their pinned dependencies come from the lockfile."""
    for name in ("web_glb.mjs", "resample_safe.mjs", "package.json", "package-lock.json"):
        assert (G.WEB_DIR / name).is_file(), name
    deps = json.loads((G.WEB_DIR / "package.json").read_text())["dependencies"]
    assert {"@gltf-transform/core", "@gltf-transform/functions", "meshoptimizer", "sharp"} <= set(deps)


# ---------------------------------------------------------------------------------------------------------------- Blender

_FILM = r'''
import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, 24
bpy.ops.mesh.primitive_cube_add(size=0.5)
mover = bpy.context.object
mover.name = "mover"
for f, x in ((1, 0.0), (24, 1.0)):
    mover.location.x = x
    mover.keyframe_insert("location", frame=f)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1.0)
cloth = bpy.context.object
cloth.name = "sheet"
basis = cloth.shape_key_add(name="Basis")
bent = cloth.shape_key_add(name="bent")
for v in bent.data:
    v.co.z += 0.4 * (v.co.x + 0.5) ** 2           # a bend, not a lift: a rigid move would export as a translation
for f, w in ((1, 0.0), (24, 1.0)):
    bent.value = w
    bent.keyframe_insert("value", frame=f)
cloth.modifiers.new("wave", "WAVE")                   # deformation under the keys: becomes morph targets
mat = bpy.data.materials.new("checker_paint")
mat.use_nodes = True
nt = mat.node_tree
checker = nt.nodes.new("ShaderNodeTexChecker")
nt.links.new(checker.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
mover.data.materials.append(mat)
for name, f in (("cam_a", 1), ("cam_b", 12)):
    bpy.ops.object.camera_add(location=(0, -4, 1))
    cam = bpy.context.object
    cam.name = name
    sc.timeline_markers.new(name, frame=f).camera = cam
sc.camera = bpy.data.objects["cam_a"]
bpy.ops.wm.save_as_mainfile(filepath=out)
'''

_IMPORT = r'''
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
print("IMPORTED " + json.dumps({"meshes": sum(o.type == "MESH" for o in bpy.data.objects),
                                "cameras": sum(o.type == "CAMERA" for o in bpy.data.objects)}))
'''


def _blender_python():
    from dycraft.render._blender_env import blender_python
    py = blender_python()
    if not py:
        pytest.skip("no Blender interpreter configured (python -m dycraft.install blender)")
    return py


def _reimport(py, glb: Path) -> dict:
    ran = subprocess.run([py, "-c", _IMPORT, "--", str(glb)], capture_output=True, text=True, timeout=300)
    line = next((l for l in ran.stdout.splitlines() if l.startswith("IMPORTED ")), None)
    assert line, ran.stdout[-800:] + ran.stderr[-800:]
    return json.loads(line[len("IMPORTED "):])


@pytest.mark.expensive
def test_a_film_exports_its_motion_cameras_cuts_and_flat_materials(tmp_path):
    """Rigid motion as node animation, shape-key deformation under a modifier as morph targets, both cameras with
    the cut between them in the sidecar, and the procedural material reported as flattened."""
    py = _blender_python()
    blend = tmp_path / "film" / "animation.blend"
    blend.parent.mkdir()
    subprocess.run([py, "-c", _FILM, "--", str(blend)], check=True, capture_output=True, timeout=300)
    lines: list[str] = []
    report = G.export(blend.parent, log=lines.append)
    out = blend.parent / "animation.glb"
    doc = _glb_json(out)
    names = {n.get("name") for n in doc["nodes"]}
    assert {"mover", "sheet", "cam_a", "cam_b"} <= names
    assert len(doc.get("cameras", [])) == 2
    channels = [c["target"]["path"] for a in doc.get("animations", []) for c in a["channels"]]
    assert "translation" in channels and "weights" in channels, channels
    assert any("targets" in p for m in doc["meshes"] for p in m["primitives"]), "the sheet carries morph targets"
    assert not doc.get("extensionsRequired"), "the default GLB requires no extension, so any glTF 2.0 viewer opens it"
    cuts = json.loads((blend.parent / "animation.cuts.json").read_text())
    assert [c["camera"] for c in cuts["cuts"]] == ["cam_a", "cam_b"] and cuts["fps"] == 24
    assert report["flattened_objects"] == ["mover"]
    assert any("procedural materials" in l and "--bake-materials" in l for l in lines)
    assert _reimport(py, out) == {"meshes": 2, "cameras": 2}


@pytest.mark.expensive
def test_an_object_run_exports_the_parts_a_still_draws(tmp_path, monkeypatch):
    """The object is built in Blender from the same world parts a still uses, each object named after its part."""
    py = _blender_python()
    from dycraft.render import _world
    part = _world._WorldPart(name="lamp/base", size=(0.2, 0.2, 0.05), se3=((0.0, 0.0, 0.025), (1.0, 0.0, 0.0, 0.0)),
                             material={"color": (0.7, 0.25, 0.15), "metallic": 0.0, "roughness": 0.6, "alpha": 1.0})
    plan = _world._Plan(world_parts=[part], room_info=None, cameras=[], target_bbox=((0, 0, 0), (1, 1, 1)),
                        ir_hash="h", catalog_fp=None)
    monkeypatch.setattr(_world, "_prepare", lambda request: plan)
    run = tmp_path / "run"
    _touch(run / "meta.json", json.dumps({"layer": "object"}))
    _touch(run / "ir.json", json.dumps({"id": "h", "parts": [{"name": "lamp/base"}]}))
    G.export(run, log=lambda line: None)
    doc = _glb_json(run / "object.glb")
    assert "lamp/base" in {n.get("name") for n in doc["nodes"]}
    assert not (run / "object.cuts.json").exists(), "no camera, no cuts sidecar"
    assert _reimport(py, run / "object.glb") == {"meshes": 1, "cameras": 0}
