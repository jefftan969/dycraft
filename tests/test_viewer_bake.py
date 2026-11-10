"""Derived viewer assets: cache boundary, URDF substitution, and the bake.

The parent half is checkable without bpy and that is where the dangerous bugs
live — a rewritten mesh path leaking into the canonical IR, a cache key that
does not move when the recipe does, a path-traversal in the asset route. Those
are all cheap tests.

The two that spawn Blender are marked `expensive`. One of them is the only
thing in the suite that would catch a lost UV pin or a dropped view-dependent
scalar: both are silent, both produce a plausible-looking asset, and both
measure 5-6/255 against the canonical render where a correct bake measures
under 0.4.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

from dycraft import render as R
from dycraft.render import viewer_bake as VB
from dycraft.viewer import derived as derived_mod
from dycraft.viewer import runs as runs_mod

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_MAT = {"color": [180, 170, 160], "alpha": 1.0, "metallic": 0.0,
        "roughness": 0.6}

#: A shader-authored material, which is the only thing a derived asset exists
#: for. Every object built here carries one, because `viewer/derived.resolve`
#: refuses a run with no `Material(shader=...)`: a scalar or textured material
#: is already drawn correctly by `materials.ts`, so baking it would spend
#: 15-95 s of GPU per object reproducing an appearance three.js can compute.
_SHADER_MAT = {"alpha": 1.0, "shader": {
    "name": "plain_oak",
    "body": "def plain_oak(nodes, links, bsdf):\n"
            "    bsdf.inputs['Roughness'].default_value = 0.4\n"}}

#: A unit-ish box as a staged OBJ, WITH per-corner UVs.
#:
#: `solid`, not `box`: an analytic primitive is emitted as native URDF geometry
#: with no `<mesh filename>` to substitute, so it is neither baked nor
#: substituted — and every shader-bearing part on disk (69 of them) is a
#: `solid`. The UVs matter because they are what `_bake_pin_uv_reads` has to
#: keep a UV-driven shader pointed at.
_CUBE_OBJ = """\
v -0.06 -0.06 0.0
v  0.06 -0.06 0.0
v  0.06  0.06 0.0
v -0.06  0.06 0.0
v -0.06 -0.06 0.06
v  0.06 -0.06 0.06
v  0.06  0.06 0.06
v -0.06  0.06 0.06
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1 2/2 3/3
f 1/1 3/3 4/4
f 5/1 7/3 6/2
f 5/1 8/4 7/3
f 1/1 5/2 6/3
f 1/1 6/3 2/4
f 2/1 6/2 7/3
f 2/1 7/3 3/4
f 3/1 7/2 8/3
f 3/1 8/3 4/4
f 4/1 8/2 5/3
f 4/1 5/3 1/4
"""


def _solid_params(mesh_rel: str) -> dict:
    return {
        "mesh": mesh_rel, "digest": "deadbeef",
        "aabb_min": [-0.06, -0.06, 0.0], "aabb_max": [0.06, 0.06, 0.06],
        "shape": {"volume": 0.000864, "fill_ratio": 1.0,
                  "center": [0.0, 0.0, 0.03],
                  "inertia_diag": [1e-6, 1e-6, 1e-6],
                  "face_census": {"PLANE": 6}},
    }


def _stage_cube(run: Path, name: str = "cube") -> str:
    rel = f"assets/meshes/{name}.obj"
    target = run / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_CUBE_OBJ)
    return rel


def _object_ir(obj_id: str = "objbake1",
               mesh_rel: str = "assets/meshes/cube.obj") -> dict:
    return {
        "id": obj_id, "name": "widget", "root": "body", "schema": "object/v1",
        "parts": [
            {"name": "body", "primitive": "solid",
             "params": _solid_params(mesh_rel), "material": dict(_SHADER_MAT)},
            {"name": "lid", "primitive": "solid",
             "params": _solid_params(mesh_rel), "material": dict(_SHADER_MAT)},
        ],
        "joints": [
            {"name": "hinge", "type": "revolute", "parent": "body",
             "child": "lid", "axis": [1.0, 0.0, 0.0], "limits": [0.0, 1.57],
             "default": 0.0, "id": 1,
             "parent_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                               "translation": [0.0, 0.0, 0.06]},
             "child_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                              "translation": [0.0, 0.0, 0.0]}},
        ],
        "anchors": [],
    }


def _run(tmp_path: Path, ir: dict | None = None, name: str = "run") -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    _stage_cube(run)
    (run / "ir.json").write_text(json.dumps(ir or _object_ir()))
    (run / "meta.json").write_text(json.dumps({"layer": "object"}))
    return run


def _manifest(*parts: str) -> dict:
    return {"ok": True, "version": VB.BAKE_VERSION,
            "parts": [{"part": p, "file": f"{i:03d}_{p}.glb", "atlas": 512,
                       "bytes": 1234, "uv_pinned": 0, "scalars": {}}
                      for i, p in enumerate(parts)]}


def _catalog_object(ref: str) -> dict:
    """An ObjectIR as it is stored in `catalog/`: paths relative to `catalog/`.

    `load_catalog` rebases those to run-relative
    (`catalog/assets/meshes/...`) on the way out, which is the form the bake and
    the substitution both see.
    """
    ir = _object_ir(ref, mesh_rel=f"assets/meshes/{ref}.obj")
    ir["name"] = f"widget-{ref}"
    return ir


def _scene_run(tmp_path: Path, name: str = "scene",
               instances: tuple[tuple[str, str], ...] = (("base", "cat-a"),),
               ) -> Path:
    """A scene run whose catalog holds every object its instances name.

    `instances` is `(instance_name, object_ref)` pairs, so a repeated ref is
    expressible — which is the case worth testing, because two instances of one
    object must share a single cache entry and a single URL.
    """
    run = tmp_path / name
    catalog_dir = run / "catalog"
    catalog_dir.mkdir(parents=True)
    identity = {"rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0]}
    for ref in dict.fromkeys(ref for _, ref in instances):
        _stage_cube(catalog_dir, ref)
        (catalog_dir / f"{ref}.json").write_text(
            json.dumps(_catalog_object(ref)))
    (run / "ir.json").write_text(json.dumps({
        "id": "scene1", "name": "corner", "schema": "dycraft.scene.v1",
        "room": {"width": 2.0, "depth": 2.0, "height": 2.2,
                 "floor_material": dict(_MAT), "wall_material": dict(_MAT)},
        "groups": [], "assertions": [],
        "instances": [
            {"id": i, "name": inst, "object_name": f"widget-{ref}",
             "object_ref": ref,
             "placement": {"kind": "attach", "parent": "room",
                           "parent_anchor": {**identity, "name": "bottom"},
                           "child_anchor": {**identity, "name": "bottom"}}}
            for i, (inst, ref) in enumerate(instances)],
    }))
    (run / "meta.json").write_text(json.dumps({"layer": "scene"}))
    return run


def _animation_run(tmp_path: Path, name: str = "anim",
                   instances: tuple[tuple[str, str], ...] = (("base", "cat-a"),),
                   ) -> Path:
    """An animation run: an AnimationIR plus the `scene.json` it replays.

    `run_urdf` and `derived.resolve` both convert one of these to its staged
    scene, so the fixture is a scene run with an AnimationIR laid over it.
    """
    run = _scene_run(tmp_path, name=name, instances=instances)
    (run / "scene.json").write_text((run / "ir.json").read_text())
    (run / "ir.json").write_text(json.dumps({
        "id": "anim1", "schema": "dycraft.animation.v1", "fps": 24,
        "n_frames": 2, "scene_ref": "scene1", "frames": [], "actors": []}))
    (run / "meta.json").write_text(json.dumps({"layer": "animation"}))
    return run


def _publish(ir_id: str, *parts: str) -> dict:
    """Write a complete manifest for `ir_id` without running Blender."""
    entry = VB.asset_dir(ir_id)
    entry.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(*parts)
    (entry / VB.MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(VB.CACHE_ENV_VAR, str(root))
    return root


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

def test_the_bake_version_is_the_same_on_both_sides_of_the_interpreter():
    """`viewer_bake` cannot import the child, so the version is duplicated.

    It is the cache key's recipe component: if the two drift, the parent serves
    an entry the child would no longer produce, and nothing else notices.
    """
    src = (Path(R.__file__).resolve().parent / "_blender_child.py").read_text()
    found = re.search(r'^BAKE_VERSION = "([^"]+)"', src, re.MULTILINE)
    assert found, "the child no longer declares BAKE_VERSION"
    assert found.group(1) == VB.BAKE_VERSION, (
        "render/viewer_bake.py and render/_blender_child.py disagree about "
        "BAKE_VERSION; a stale derived asset would be served forever")


def test_the_cache_key_covers_both_the_object_and_the_recipe(cache):
    """The IR hash covers geometry and materials; the version covers the bake."""
    a = VB.asset_dir("hash-one")
    b = VB.asset_dir("hash-two")
    assert a != b
    assert VB.BAKE_VERSION in a.name and a.name.startswith("hash-one")
    assert a.parent == cache


def test_an_entry_with_no_manifest_reads_as_absent(cache):
    """A bake that died partway leaves files but no manifest.

    Treating the directory's existence as "done" is what would serve half an
    object; the manifest arriving last is the whole completeness protocol.
    """
    entry = VB.asset_dir("half")
    entry.mkdir(parents=True)
    (entry / "000_body.glb").write_bytes(b"not really a glb")
    assert VB.read_manifest("half") is None


@pytest.mark.parametrize("payload", ['{"ok": false}', "not json at all", "[]"])
def test_a_malformed_or_failed_manifest_reads_as_absent(cache, payload):
    entry = VB.asset_dir("bad")
    entry.mkdir(parents=True)
    (entry / VB.MANIFEST_NAME).write_text(payload)
    assert VB.read_manifest("bad") is None


def test_asset_path_refuses_anything_outside_its_own_entry(cache):
    """The filename arrives from a URL, so this is the traversal guard."""
    entry = VB.asset_dir("ok")
    entry.mkdir(parents=True)
    (entry / "000_body.glb").write_bytes(b"glb")
    (cache / "secret.txt").write_text("nope")
    assert VB.asset_path("ok", "000_body.glb") is not None
    assert VB.asset_path("ok", "../secret.txt") is None
    assert VB.asset_path("ok", "/etc/passwd") is None
    assert VB.asset_path("ok", "missing.glb") is None


# ---------------------------------------------------------------------------
# The canonical state must not move
# ---------------------------------------------------------------------------

def test_substitution_copies_the_ir_rather_than_mutating_it():
    """The IR dict is also what the viewer reports, hashes, and shows as source.

    A rewritten mesh path leaking into it is how a display change becomes a
    data change, so the original is compared field-by-field afterwards.
    """
    ir = _object_ir()
    before = json.dumps(ir, sort_keys=True)
    out = VB.substituted_ir(ir, _manifest("body", "lid"), "__derived__/")
    assert json.dumps(ir, sort_keys=True) == before, "the input IR was mutated"
    assert out is not ir and out["parts"] is not ir["parts"]
    assert [p["params"]["mesh"] for p in out["parts"]] == [
        "__derived__/000_body.glb", "__derived__/001_lid.glb"]
    # Everything else is carried through untouched.
    assert out["id"] == ir["id"] and out["joints"] == ir["joints"]


def test_a_part_with_no_derived_asset_keeps_its_staged_mesh():
    """A partly-bakeable object degrades to a mixture, not to missing geometry."""
    ir = _object_ir()
    for part in ir["parts"]:
        part["params"]["mesh"] = f"assets/meshes/{part['name']}.obj"
    out = VB.substituted_ir(ir, _manifest("body"), "__derived__/")
    meshes = [p["params"]["mesh"] for p in out["parts"]]
    assert meshes == ["__derived__/000_body.glb", "assets/meshes/lid.obj"]


def test_an_analytic_primitive_is_neither_baked_nor_substituted(tmp_path):
    """A `box` has no `<mesh filename>` in the URDF, so a GLB for it is
    unreachable — and it is already drawn correctly by the scalar path."""
    ir = _object_ir()
    ir["parts"].append({
        "name": "peg", "primitive": "box",
        "params": {"width": 0.02, "length": 0.02, "height": 0.02},
        "material": dict(_MAT)})
    ir["joints"].append({
        "name": "peg_mount", "type": "fixed", "parent": "body",
        "child": "peg", "axis": [0.0, 0.0, 1.0], "default": 0.0, "id": 2,
        "parent_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                          "translation": [0.0, 0.0, 0.06]},
        "child_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                         "translation": [0.0, 0.0, 0.0]}})
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=tmp_path,
        backend="blender")
    job = VB.pack_bake_job(request, tmp_path)
    assert [p["ir_part"] for p in job["frames"][0]["parts"]] == ["body", "lid"]


def test_the_canonical_urdf_is_byte_identical_with_a_manifest_present(tmp_path, cache):
    """Whether a derived asset exists must be invisible to everything else.

    `run_urdf(run_dir)` with no flag is what the canonical exporter produces;
    it has to stay that way once a bake lands beside it, or the derived path has
    changed the meaning of the run rather than added a view of it.
    """
    ir = _object_ir()
    run = _run(tmp_path, ir)
    before = runs_mod.run_urdf(run)
    entry = VB.asset_dir(ir["id"])
    entry.mkdir(parents=True)
    (entry / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body", "lid")))
    assert runs_mod.run_urdf(run) == before
    # ...and the stored IR on disk is untouched.
    assert json.loads((run / "ir.json").read_text()) == ir


def test_the_derived_urdf_changes_only_the_mesh_filenames(tmp_path, cache):
    """Same joint tree, same link frames, same collision proxies.

    That is what lets articulation and Preview Motion keep working on exactly
    the graph they work on today.
    """
    ir = _object_ir()
    run = _run(tmp_path, ir)
    entry = VB.asset_dir(ir["id"])
    entry.mkdir(parents=True)
    (entry / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body", "lid")))

    canonical = runs_mod.run_urdf(run)
    derived = runs_mod.run_urdf(run, derived=True)
    strip = lambda text: re.sub(r'filename="[^"]*"', 'filename="X"', text)  # noqa: E731
    assert strip(canonical) == strip(derived), (
        "the derived URDF differs from the canonical one somewhere other than "
        "a mesh filename")
    assert f'filename="__derived__/{ir["id"]}/000_body.glb"' in derived, (
        "the derived mesh URL must carry the object's ir_id: a scene draws "
        "several objects whose manifests both number a file 000_*")


def test_asking_for_derived_without_a_bake_falls_back_to_the_canonical_urdf(tmp_path, cache):
    """The flag is a request, not an assertion.

    The browser sets it from a status poll that can race a cache eviction, and
    the honest degradation is the preview appearance rather than a 404 for every
    mesh in the object.
    """
    run = _run(tmp_path)
    assert runs_mod.run_urdf(run, derived=True) == runs_mod.run_urdf(run)


# ---------------------------------------------------------------------------
# What gets packed, and what gets refused
# ---------------------------------------------------------------------------

def test_the_bake_job_carries_the_ir_part_name_for_every_part(tmp_path):
    """The manifest has to be matchable back onto `ir["parts"]`.

    `_WorldPart.name` is `f"{object}/{part}"`, and recovering the second half by
    splitting is a guess that happens to work — object names may contain a
    slash. So the IR name travels explicitly.
    """
    ir = _object_ir()
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=tmp_path,
        backend="blender")
    job = VB.pack_bake_job(request, tmp_path)
    parts = job["frames"][0]["parts"]
    assert [p["ir_part"] for p in parts] == ["body", "lid"]


def test_the_bake_job_packs_no_cameras_or_lights(tmp_path):
    """An EMIT bake returns the socket's value and a NORMAL bake the geometry.

    Neither consults the lighting rig, so packing one would invent a dependency
    the bake does not have — and would make the derived asset vary with a
    scene's fixtures.
    """
    ir = _object_ir()
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=tmp_path,
        backend="blender")
    frame = VB.pack_bake_job(request, tmp_path)["frames"][0]
    assert frame["cameras"] == []
    assert "lights" not in frame and "environment" not in frame
    assert "framing" not in frame and "room" not in frame


def test_a_scene_shaped_request_is_refused_with_the_reason(tmp_path):
    """A bake takes one object. A scene is baked as its catalog objects, which
    are themselves ObjectIRs — so a scene-shaped request is a caller mistake,
    and `viewer/derived.resolve` is where the fan-out happens instead."""
    request = R.RenderRequest(
        layer="scene", ir={"id": "s1", "instances": []}, catalog={},
        target="all", out_dir=tmp_path / "out", step=0,
        asset_base_dir=tmp_path, backend="blender")
    with pytest.raises(R.RenderUnavailable, match="one object at a time"):
        VB.pack_bake_job(request, tmp_path)


def test_an_ir_with_no_id_cannot_be_cached_and_says_so(tmp_path, cache):
    ir = _object_ir()
    ir.pop("id")
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=tmp_path,
        backend="blender")
    with pytest.raises(R.RenderUnavailable, match="no id"):
        VB.build(request)


def test_an_animation_run_resolves_as_its_staged_scene(tmp_path, cache):
    """Animation runs are supported, deformation and all.

    The bake happens on the rest shape and the atlas UVs travel to the browser
    with the rig, so a baked pattern stays attached to the surface and deforms
    with it. `run_urdf` already converts an animation run to its scene before
    generating geometry, so resolving the same way is what keeps both halves
    agreeing about which catalog objects exist.
    """
    run = _animation_run(tmp_path, instances=(("base", "cat-a"),))
    assert [e.ir_id for e in derived_mod.resolve(run)] == ["cat-a"]

    _publish("cat-a", "body", "lid")
    derived_mod._STATE.clear()
    got = derived_mod.status(run, start=False)
    assert got["status"] == "ready" and got["objects"] == 1

    # And the derived URDF substitutes, through the scene conversion.
    meshes = set(re.findall(r'<mesh filename="([^"]+)"',
                            runs_mod.run_urdf(run, derived=True)))
    assert meshes == {"__derived__/cat-a/000_body.glb",
                      "__derived__/cat-a/001_lid.glb"}


def test_an_animation_run_with_no_staged_scene_is_refused_with_that_reason(tmp_path):
    """The only animation-specific refusal left: nothing to resolve against."""
    run = tmp_path / "anim"
    run.mkdir()
    (run / "ir.json").write_text(json.dumps({
        "id": "a1", "schema": "dycraft.animation.v1", "fps": 30,
        "frames": [], "scene_ref": "s1"}))
    with pytest.raises(derived_mod.DerivedUnavailable) as exc:
        derived_mod.resolve(run)
    assert "scene.json" in str(exc.value)


def test_an_animation_run_of_plain_material_objects_is_left_alone(tmp_path, cache):
    """The promise to every existing deformation animation.

    None of the 13 rigged runs on disk is shader-authored, so none of them gets
    a bake, no derived URDF, and therefore the staged meshes and the URDF-spec
    materials they draw today — with the rig's UVs in the staged mesh's own
    space, which is what that material is addressed in.
    """
    run = _animation_run(tmp_path, instances=(("base", "cat-a"),))
    plain = json.loads((run / "catalog" / "cat-a.json").read_text())
    for part in plain["parts"]:
        part["material"] = dict(_MAT)
    (run / "catalog" / "cat-a.json").write_text(json.dumps(plain))

    with pytest.raises(derived_mod.DerivedUnavailable) as exc:
        derived_mod.resolve(run)
    assert "nothing to derive" in str(exc.value)
    # And asking for the derived URDF returns the canonical one unchanged.
    assert runs_mod.run_urdf(run, derived=True) == runs_mod.run_urdf(run)


def test_status_with_start_false_never_starts_a_bake(tmp_path, cache, monkeypatch):
    """The URDF route and the tests both ask; neither may trigger 90 seconds."""
    called = []
    monkeypatch.setattr(derived_mod.threading, "Thread",
                        lambda *a, **k: called.append(k) or pytest.fail(
                            "status(start=False) started a bake"))
    run = _run(tmp_path)
    assert derived_mod.status(run, start=False)["status"] == "absent"
    assert called == []


def test_status_reports_a_ready_bake_without_touching_blender(tmp_path, cache):
    ir = _object_ir()
    run = _run(tmp_path, ir)
    entry = VB.asset_dir(ir["id"])
    entry.mkdir(parents=True)
    (entry / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body", "lid")))
    got = derived_mod.status(run, start=False)
    assert got["status"] == "ready" and got["parts"] == 2
    assert got["bytes"] == 2468


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

def test_the_derived_asset_route_serves_and_refuses_correctly(tmp_path, cache):
    from starlette.testclient import TestClient

    from dycraft.viewer.app import create_app

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    ir = _object_ir()
    _run(runs_dir, ir, "obj")
    entry = VB.asset_dir(ir["id"])
    entry.mkdir(parents=True)
    (entry / "000_body.glb").write_bytes(b"glTF-ish")
    (entry / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body", "lid")))

    app = create_app(runs_dir=runs_dir)
    base = f"/api/runs/obj/files/__derived__/{ir['id']}"
    with TestClient(app) as client:
        ok = client.get(f"{base}/000_body.glb")
        assert ok.status_code == 200 and ok.content == b"glTF-ish"
        assert "immutable" in ok.headers.get("cache-control", "")
        # Named in the manifest but never written: still a 404, not a 500.
        assert client.get(f"{base}/001_lid.glb").status_code == 404
        assert client.get(f"{base}/../manifest.json").status_code == 404
        # No file segment at all.
        assert client.get(
            "/api/runs/obj/files/__derived__/").status_code == 404
        # The canonical mesh route is unaffected.
        assert client.get(
            "/api/runs/obj/files/assets/meshes/none.obj").status_code == 404


def test_the_asset_route_refuses_an_object_this_run_does_not_draw(tmp_path, cache):
    """The cache is shared by every run on disk, so the id is authorization.

    Without checking it against what this run resolves to, any run's URL could
    read out any other run's baked object — including one from a run the
    browser never opened.
    """
    from starlette.testclient import TestClient

    from dycraft.viewer.app import create_app

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    ir = _object_ir()
    _run(runs_dir, ir, "obj")
    mine = VB.asset_dir(ir["id"])
    mine.mkdir(parents=True)
    (mine / "000_body.glb").write_bytes(b"glTF-ish")
    (mine / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body")))
    # A complete bake belonging to some other run entirely.
    theirs = VB.asset_dir("someone-elses-object")
    theirs.mkdir(parents=True)
    (theirs / "000_body.glb").write_bytes(b"not yours")
    (theirs / VB.MANIFEST_NAME).write_text(json.dumps(_manifest("body")))

    app = create_app(runs_dir=runs_dir)
    with TestClient(app) as client:
        assert client.get(
            f"/api/runs/obj/files/__derived__/{ir['id']}/000_body.glb"
        ).status_code == 200
        got = client.get(
            "/api/runs/obj/files/__derived__/someone-elses-object/000_body.glb")
    assert got.status_code == 404, (
        "the route served a cache entry this run does not draw")


def test_the_status_route_never_raises_for_an_unsupported_run(tmp_path, cache):
    """The browser's contract is to keep the preview and say why, so it needs a
    reason in a 200 rather than an error code it has to interpret."""
    from starlette.testclient import TestClient

    from dycraft.viewer.app import create_app

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    anim = runs_dir / "anim"
    anim.mkdir()
    (anim / "ir.json").write_text(json.dumps({
        "id": "a1", "schema": "dycraft.animation.v1", "fps": 30,
        "frames": [], "scene_ref": "s1"}))
    (anim / "meta.json").write_text(json.dumps({"layer": "animation"}))

    app = create_app(runs_dir=runs_dir)
    with TestClient(app) as client:
        got = client.get("/api/runs/anim/derived", params={"build": 0})
    assert got.status_code == 200
    assert got.json()["status"] == "unavailable"
    assert "scene.json" in got.json()["reason"]


# ---------------------------------------------------------------------------
# Against real Blender
# ---------------------------------------------------------------------------

def _blender_or_skip():
    from dycraft.render import _blender_env as E

    interp = E.blender_python()
    if not interp:
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")
    return interp


def _glb_json(path: Path) -> dict:
    blob = path.read_bytes()
    length, _ = struct.unpack("<II", blob[12:20])
    return json.loads(blob[20:20 + length])


@pytest.mark.expensive
def test_a_baked_asset_is_shaped_the_way_the_browser_needs(tmp_path, cache):
    """Structure, not pixels — and every assertion here is a silent failure.

    * **One `TEXCOORD_0`.** The original tiling lattice is dropped after the
      bake; leaving it makes the exporter reference every texture as
      `texCoord=1`, which works only because three.js happens to support a
      second set.
    * **Z-up `POSITION`.** glTF's convention is Y-up and Blender's exporter
      converts by default. The viewer applies the z-up rotation once at the
      robot group, so a converted asset arrives rotated 90 degrees relative to
      every staged OBJ beside it.
    * **The view-dependent scalars as extensions.** A bake cannot capture them;
      dropping them is what makes a 95%-transmissive pane render as an opaque
      white panel.
    """
    _blender_or_skip()
    run = _run(tmp_path, name="src")
    ir = _object_ir("bakeshape1")
    ir["parts"][0]["material"] = {
        **_MAT, "shader": {"name": "probe", "body": (
            "def probe(nodes, links, bsdf):\n"
            "    tc = nodes.new('ShaderNodeTexCoord')\n"
            "    noise = nodes.new('ShaderNodeTexNoise')\n"
            "    noise.inputs['Scale'].default_value = 40.0\n"
            "    links.new(tc.outputs['Object'], noise.inputs['Vector'])\n"
            "    links.new(noise.outputs['Color'], bsdf.inputs['Base Color'])\n"
            "    bump = nodes.new('ShaderNodeBump')\n"
            "    bump.inputs['Strength'].default_value = 0.3\n"
            "    links.new(noise.outputs['Fac'], bump.inputs['Height'])\n"
            "    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])\n"
            "    bsdf.inputs['Coat Weight'].default_value = 0.4\n"
            "    bsdf.inputs['Coat Roughness'].default_value = 0.15\n"
        )}}
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=run,
        backend="blender")
    manifest = VB.build(request, force=True)

    assert [e["part"] for e in manifest["parts"]] == ["body", "lid"]
    entry = next(e for e in manifest["parts"] if e["part"] == "body")
    gltf = _glb_json(VB.asset_dir(ir["id"]) / entry["file"])

    prim = gltf["meshes"][0]["primitives"][0]
    texcoords = {k for k in prim["attributes"] if k.startswith("TEXCOORD")}
    assert texcoords == {"TEXCOORD_0"}, (
        f"expected only the atlas UV set, got {sorted(texcoords)}")
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    tex = pbr.get("baseColorTexture")
    assert tex is not None, "the GLB carries no baseColorTexture"
    assert tex.get("texCoord", 0) == 0
    assert gltf["materials"][0].get("normalTexture") is not None
    # The probe drives the color and the normal; its roughness is a plain
    # value, carried as one rather than baked into an image of that value.
    assert entry["maps"] == ["Base Color", "Normal"]
    assert pbr.get("metallicRoughnessTexture") is None
    assert pbr.get("roughnessFactor") == pytest.approx(0.6, abs=1e-6)

    # The lid's shader drives nothing, so nothing is baked for it at all.
    lid = next(e for e in manifest["parts"] if e["part"] == "lid")
    assert lid["maps"] == []
    lid_gltf = _glb_json(VB.asset_dir(ir["id"]) / lid["file"])
    assert not lid_gltf.get("images"), "a map was baked for a plain value"
    assert lid_gltf["materials"][0]["pbrMetallicRoughness"].get(
        "roughnessFactor") == pytest.approx(0.4, abs=1e-6)

    # Z-up: the box is 0.12 x 0.12 x 0.06, so the SHORT axis must be Z.
    accessor = gltf["accessors"][prim["attributes"]["POSITION"]]
    extent = [hi - lo for lo, hi in zip(accessor["min"], accessor["max"])]
    assert extent[2] == pytest.approx(0.06, abs=1e-4), (
        f"POSITION extents {extent} are not Z-up; export_yup leaked and the "
        "asset would arrive rotated 90 degrees")

    coat = (gltf["materials"][0].get("extensions") or {}).get(
        "KHR_materials_clearcoat")
    assert coat is not None, "the authored Coat Weight was dropped"
    assert coat["clearcoatFactor"] == pytest.approx(0.4, abs=1e-3)
    assert coat["clearcoatRoughnessFactor"] == pytest.approx(0.15, abs=1e-3)
    assert entry["scalars"]["Coat Weight"] == pytest.approx(0.4)


@pytest.mark.expensive
def test_the_bake_is_idempotent_and_cached(tmp_path, cache):
    """The second call must not re-bake: it is 15-95 s of GPU per object."""
    _blender_or_skip()
    import time

    run = _run(tmp_path, name="src2")
    ir = _object_ir("bakecache1")
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=run,
        backend="blender")
    first = VB.build(request)
    started = time.time()
    second = VB.build(request)
    elapsed = time.time() - started
    assert second == first
    assert elapsed < 1.0, (
        f"the cached path took {elapsed:.1f}s; it re-baked instead of reading "
        "the manifest")


@pytest.mark.expensive
def test_two_parts_sharing_a_texture_both_bake(tmp_path, cache):
    """A bake survives a texture its parts share.

    `_bake_run` purges orphans after every part, and the first part's material,
    replaced by its baked one, leaves the shared image with no user. Two guards
    in `_load_image` cover it: the fake user every load gets, which keeps the
    purge from taking the image, and the liveness probe, which reloads one
    that was taken anyway. This fails only with both gone. A texture is
    required: with no image bound there is nothing to dangle.
    """
    _blender_or_skip()
    from PIL import Image as PILImage

    run = _run(tmp_path, name="src3")
    # `TextureMap` requires a relative path and a 32-hex digest, and
    # `material_from_ir` silently drops a map that has neither, which would
    # leave nothing to share.
    digest = "d1eece480c876b995f4e7e2d3c6455ba"
    rel = f"assets/textures/{digest}.png"
    (run / rel).parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (64, 64), (200, 120, 60)).save(run / rel)
    ir = _object_ir("baketex1")
    for part in ir["parts"]:
        part["material"] = {"color": [255, 255, 255], "texture": {
            "base_color": {"image": rel, "digest": digest}}}
    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=run,
        backend="blender")
    manifest = VB.build(request, force=True)
    assert [e["part"] for e in manifest["parts"]] == ["body", "lid"], (
        manifest.get("warnings"))
    for entry in manifest["parts"]:
        assert (VB.asset_dir(ir["id"]) / entry["file"]).stat().st_size > 1000


#: Renders a UV-driven shader three ways under one Blender process: the
#: canonical graph, the atlas bake, and the atlas bake with the UV pin
#: sabotaged. Same pattern as `test_blender_backend._NODE_PROBE` — the parent
#: suite needs no bpy, and the one thing only a render can answer gets its own
#: interpreter.
_BAKE_FIDELITY_PROBE = r'''
import importlib.util, json, math, sys
import numpy as np
spec = importlib.util.spec_from_file_location("_bc", %(child)r)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
import bpy

job = json.loads(open(%(job)r).read())
frame = job["frames"][0]
spec_part = frame["parts"][0]

def build_scene():
    scene = bc._reset_scene(job)
    cam = bc._make_camera(scene)
    bc._setup_rig(frame["framing"], None)
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_persistent_data = False
    scene.cycles.samples = 48
    return scene, cam

def shoot(scene, cam, tag):
    bc._aim_camera(cam, frame["cameras"][0])
    scene.render.filepath = %(out)r + "/" + tag
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(%(out)r + "/" + tag + ".png", check_existing=False)
    w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[:, :, :3] * 255.0
    bpy.data.images.remove(img)
    return a

def render_variant(bake, sabotage):
    scene, cam = build_scene()
    if bake:
        bc._bake_prepare(scene)
        if sabotage:
            bc._bake_pin_uv_reads = lambda *a, **k: 0
        obj = None
        entry = bc._bake_part(spec_part, 0, %(out)r)
        obj = bpy.context.scene.objects[0]
        scene.cycles.samples = 48
        scene.render.use_persistent_data = False
        return shoot(scene, cam, "bake_sab" if sabotage else "bake"), entry
    obj = bc._build_part(spec_part, 0)
    return shoot(scene, cam, "canonical"), None

ref, _ = render_variant(False, False)
import importlib
pinned, entry = render_variant(True, False)
# Reload so the sabotage cannot leak into a later variant.
spec2 = importlib.util.spec_from_file_location("_bc2", %(child)r)
bc = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(bc)
unpinned, _ = render_variant(True, True)

def err(a):
    return float(np.abs(a - ref).mean())

print("RESULT " + json.dumps({
    "pinned": err(pinned), "unpinned": err(unpinned),
    "uv_pinned": entry["uv_pinned"], "atlas": entry["atlas"],
}))
'''


@pytest.mark.expensive
def test_the_baked_asset_matches_the_canonical_render_and_the_uv_pin_matters(
        tmp_path, cache):
    """The pixel guard, and the only test that can catch a lost UV pin.

    A shader that reads `TexCoord.UV` is baked twice: once properly, and once
    with `_bake_pin_uv_reads` stubbed out. The proper bake has to land within
    half a level of the canonical Cycles render; the sabotaged one has to be
    *visibly* worse, because a test that passed either way would not be
    testing the pin at all.

    Structure alone cannot see this: both variants export one `TEXCOORD_0`, the
    same textures and the same extensions. Only the pixels differ.
    """
    interp = _blender_or_skip()
    import subprocess

    from dycraft.render import _blender_env as E
    from dycraft.render import blender_renderer as B

    run = _run(tmp_path, name="fid")
    ir = _object_ir("bakefid1")
    # Reads UV, and at a scale fine enough that sampling the wrong
    # parameterization cannot come out looking similar by accident.
    ir["parts"][0]["material"] = {
        **_MAT, "shader": {"name": "stripes", "body": (
            "def stripes(nodes, links, bsdf):\n"
            "    tc = nodes.new('ShaderNodeTexCoord')\n"
            "    wave = nodes.new('ShaderNodeTexWave')\n"
            "    wave.inputs['Scale'].default_value = 18.0\n"
            "    links.new(tc.outputs['UV'], wave.inputs['Vector'])\n"
            "    ramp = nodes.new('ShaderNodeValToRGB')\n"
            "    ramp.color_ramp.elements[0].color = (0.85, 0.15, 0.10, 1.0)\n"
            "    ramp.color_ramp.elements[1].color = (0.10, 0.20, 0.80, 1.0)\n"
            "    links.new(wave.outputs['Fac'], ramp.inputs['Fac'])\n"
            "    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])\n"
        )}}
    ir["parts"] = ir["parts"][:1]
    ir["joints"] = []
    ir["root"] = "body"

    request = R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path / "out", step=0, asset_base_dir=run,
        backend="blender")
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    job = VB.pack_bake_job(request, scratch)
    job["frames"][0]["framing"] = [[-0.06, -0.06, 0.0], [0.06, 0.06, 0.06]]
    job["frames"][0]["cameras"] = [{
        "id": "v", "eye": [0.20, 0.22, 0.16], "target": [0.0, 0.0, 0.03],
        "up": [0.0, 0.0, 1.0], "fov_deg": 40.0, "walls": []}]
    job["width"], job["height"] = 480, 360
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))

    probe = tmp_path / "probe.py"
    out = tmp_path / "renders"
    out.mkdir()
    probe.write_text(_BAKE_FIDELITY_PROBE % {
        "child": str(Path(B._CHILD)), "job": str(job_path), "out": str(out)})
    proc = subprocess.run([interp, str(probe)], capture_output=True, text=True,
                          timeout=900, env=E.blender_child_env())
    assert "RESULT" in proc.stdout, (
        f"probe failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    got = json.loads(proc.stdout.split("RESULT", 1)[1].splitlines()[0])

    assert got["uv_pinned"] >= 1, (
        "the shader reads TexCoord.UV but no consumer was repointed")
    assert got["pinned"] < 0.5, (
        f"the baked asset is {got['pinned']:.3f}/255 from the canonical "
        f"render; a correct bake measures well under 0.5")
    assert got["unpinned"] > got["pinned"] * 4, (
        f"sabotaging the UV pin changed the picture by only "
        f"{got['unpinned']:.3f} against {got['pinned']:.3f} — this test is "
        f"not actually exercising the pin")


# ---------------------------------------------------------------------------
# The scene layer
# ---------------------------------------------------------------------------

def test_a_scene_urdf_draws_each_catalog_object_s_own_derived_meshes(tmp_path, cache):
    """The substitution happens in the CATALOG, not in the SceneIR.

    `scene_ir_to_urdf` replicates each catalog object's link tree and reads
    every mesh filename out of that object's own `params`, so rewriting the
    catalog reaches every instance without this layer knowing anything about
    `<instance>__<part>` link naming.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    _publish("cat-a", "body", "lid")
    _publish("cat-b", "body", "lid")

    canonical = runs_mod.run_urdf(run)
    derived = runs_mod.run_urdf(run, derived=True)

    strip = lambda text: re.sub(r'filename="[^"]*"', 'filename="X"', text)  # noqa: E731
    assert strip(canonical) == strip(derived), (
        "the derived scene URDF differs from the canonical one somewhere "
        "other than a mesh filename")
    meshes = set(re.findall(r'<mesh filename="([^"]+)"', derived))
    assert meshes == {
        "__derived__/cat-a/000_body.glb", "__derived__/cat-a/001_lid.glb",
        "__derived__/cat-b/000_body.glb", "__derived__/cat-b/001_lid.glb"}
    # Every link is still there, and still named the way the browser keys on.
    assert canonical.count("<link name=") == derived.count("<link name=")
    assert 'name="base__body"' in derived and 'name="top__body"' in derived


def test_two_objects_in_one_scene_do_not_collide_in_the_derived_namespace(tmp_path, cache):
    """Both manifests number a file `000_*`; only the ir_id separates them.

    This is the reason the URL prefix is per-object rather than flat: a flat
    prefix would collide on the first scene with two baked objects.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    _publish("cat-a", "body")
    _publish("cat-b", "body")
    derived = runs_mod.run_urdf(run, derived=True)
    names = re.findall(r'<mesh filename="(__derived__/[^"]+)"', derived)
    assert len(set(names)) == 2, f"the two objects share a URL: {names}"
    assert all(f.endswith("000_body.glb") for f in names)


def test_a_partially_baked_scene_draws_a_mixture(tmp_path, cache):
    """An object with no complete bake keeps its staged OBJ.

    The alternative — refusing the whole scene until every object is baked —
    would keep a six-object scene gray because of one failure, and the
    per-visual material decision in `useUrdfLoader` is what makes the mixture
    draw correctly on the browser side.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    _publish("cat-a", "body", "lid")

    derived = runs_mod.run_urdf(run, derived=True)
    meshes = set(re.findall(r'<mesh filename="([^"]+)"', derived))
    assert meshes == {
        "__derived__/cat-a/000_body.glb", "__derived__/cat-a/001_lid.glb",
        "catalog/assets/meshes/cat-b.obj"}

    # **The two front-end decisions must agree on every mesh in the mixture.**
    # `geometry-loader` dispatches its LOADER on the file extension, while
    # `useUrdfLoader` decides whether to keep the loaded MATERIAL from the
    # `__derived__/` prefix (`isDerivedMeshFilename`). If those ever disagreed,
    # a mesh would either lose its baked material or be handed none at all.
    for name in meshes:
        assert name.startswith(runs_mod.DERIVED_URL_PREFIX) \
            == name.lower().endswith(".glb"), name


def test_repeated_instances_of_one_object_share_a_single_derived_entry(tmp_path, cache):
    """Two chairs of the same object are one bake, one URL, one download.

    `geometry-loader`'s template cache is keyed by resolved URL, so the browser
    fetches the GLB once and clones it per instance.
    """
    run = _scene_run(tmp_path, instances=(("left", "cat-a"), ("right", "cat-a")))
    entries = derived_mod.resolve(run)
    assert [e.ir_id for e in entries] == ["cat-a"], (
        "one object instanced twice must resolve to one bake")

    _publish("cat-a", "body")
    derived = runs_mod.run_urdf(run, derived=True)
    names = re.findall(r'<mesh filename="(__derived__/[^"]+)"', derived)
    assert names == ["__derived__/cat-a/000_body.glb"] * 2
    assert 'name="left__body"' in derived and 'name="right__body"' in derived


def test_a_scene_bakes_each_object_as_an_object_layer_request(tmp_path, cache, monkeypatch):
    """A catalog object IS an ObjectIR, so `pack_bake_job` stays object-only.

    The asset base is the RUN dir, not `run/catalog`: `load_catalog` has already
    rebased the mesh paths to `catalog/assets/meshes/...`, so basing at
    `catalog/` would double the prefix and resolve nothing.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    seen = []

    def _fake_build(request, **kwargs):
        seen.append(request)
        return _publish(request.ir["id"], "body")

    monkeypatch.setattr(VB, "build", _fake_build)
    # Run the bake bodies inline so the assertions do not race a thread.
    monkeypatch.setattr(
        derived_mod.threading, "Thread",
        lambda target, args=(), **k: type(
            "Inline", (), {"start": lambda _self: target(*args)})())

    derived_mod._STATE.clear()
    # The first poll reports the snapshot it took BEFORE launching, which is
    # `building` even here — the real path is asynchronous and the payload must
    # not claim an answer the bake has not produced yet.
    assert derived_mod.status(run)["status"] == "building"
    assert derived_mod.status(run, start=False)["status"] == "ready"
    assert {r.ir["id"] for r in seen} == {"cat-a", "cat-b"}
    assert len(seen) == 2, "each object must bake exactly once"
    for request in seen:
        assert request.layer == "object", "a scene object must bake as an object"
        assert request.catalog is None
        assert request.asset_base_dir == run, (
            "the bake must resolve `catalog/assets/...` against the run dir")
        mesh = request.ir["parts"][0]["params"]["mesh"]
        assert (run / mesh).is_file(), f"{mesh} does not resolve under {run}"


def test_scene_status_is_ready_only_when_every_object_is(tmp_path, cache, monkeypatch):
    """The URDF is fetched once, so a half-baked scene would draw the rest
    gray with nothing saying so."""
    monkeypatch.setattr(derived_mod.threading, "Thread",
                        lambda *a, **k: pytest.fail("start=False baked"))
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    derived_mod._STATE.clear()

    assert derived_mod.status(run, start=False)["status"] == "absent"
    _publish("cat-a", "body")
    partial = derived_mod.status(run, start=False)
    assert partial["status"] == "absent"
    assert partial["objects"] == 2 and partial["objects_ready"] == 1

    _publish("cat-b", "body", "lid")
    whole = derived_mod.status(run, start=False)
    assert whole["status"] == "ready"
    assert whole["objects"] == 2 and whole["objects_ready"] == 2
    # Summed across the objects, not reported per object.
    assert whole["parts"] == 3
    # No single id identifies a scene, so none is claimed.
    assert "ir_id" not in whole


def test_a_scene_with_no_resolvable_catalog_object_is_refused_with_a_reason(tmp_path, cache):
    run = _scene_run(tmp_path)
    (run / "catalog" / "cat-a.json").unlink()
    with pytest.raises(derived_mod.DerivedUnavailable) as exc:
        derived_mod.resolve(run)
    assert "catalog" in str(exc.value)


def test_a_failed_bake_reports_error_rather_than_absent(tmp_path, cache, monkeypatch):
    """`absent` is a state the browser WAITS on, polling every two seconds.

    A failure never changes on its own, so reporting it as absent is an endless
    poll of a dead bake.
    """
    run = _run(tmp_path)
    ir_id = json.loads((run / "ir.json").read_text())["id"]
    derived_mod._STATE.clear()
    derived_mod._STATE[ir_id] = {"status": "error", "reason": "GPU on fire"}

    got = derived_mod.status(run, start=False)
    assert got["status"] == "error"
    assert "GPU on fire" in got["reason"]
    derived_mod._STATE.clear()


def test_the_canonical_scene_urdf_is_unchanged_by_a_bake_landing_beside_it(tmp_path, cache):
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    before = runs_mod.run_urdf(run)
    _publish("cat-a", "body", "lid")
    _publish("cat-b", "body", "lid")
    assert runs_mod.run_urdf(run) == before
    # And the catalog on disk still names the staged meshes.
    stored = json.loads((run / "catalog" / "cat-a.json").read_text())
    assert stored["parts"][0]["params"]["mesh"] == "assets/meshes/cat-a.obj"


def test_a_scene_where_one_object_fails_still_draws_the_others(tmp_path, cache):
    """`partial`, not `error`: five real materials beat six gray placeholders.

    The failed object keeps its staged mesh in the substituted URDF, and
    `useUrdfLoader` decides per visual which material to keep, so the mixture
    draws correctly on both halves.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    _publish("cat-a", "body", "lid")
    derived_mod._STATE.clear()
    derived_mod._STATE["cat-b"] = {"status": "error", "reason": "smart_project blew up"}

    got = derived_mod.status(run, start=False)
    assert got["status"] == "partial", got
    assert got["objects"] == 2 and got["objects_ready"] == 1
    assert "smart_project blew up" in got["reason"]
    assert got["parts"] == 2, "the ready object's parts are still counted"
    # No id is claimed: "the one that worked" is not what that field means.
    assert "ir_id" not in got

    meshes = set(re.findall(r'<mesh filename="([^"]+)"',
                            runs_mod.run_urdf(run, derived=True)))
    assert meshes == {
        "__derived__/cat-a/000_body.glb", "__derived__/cat-a/001_lid.glb",
        "catalog/assets/meshes/cat-b.obj"}
    derived_mod._STATE.clear()


def test_a_scene_where_every_object_fails_is_an_error_not_a_partial(tmp_path, cache):
    """With nothing baked there is nothing better than the preview."""
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    derived_mod._STATE.clear()
    for ref in ("cat-a", "cat-b"):
        derived_mod._STATE[ref] = {"status": "error", "reason": "no GPU"}
    got = derived_mod.status(run, start=False)
    assert got["status"] == "error" and got["objects_ready"] == 0
    derived_mod._STATE.clear()


def test_the_browser_and_the_server_agree_on_the_derived_url_prefix():
    """A cross-language contract with no import to keep it honest.

    The server names derived meshes under this prefix and the browser decides
    from it which meshes must keep the material they arrived with. If the two
    drift, a derived GLB gets its baked material thrown away and rebuilt from
    the URDF's flat gray — a silent, plausible-looking regression.
    """
    repo = Path(runs_mod.__file__).resolve().parent.parent
    src = (repo / "viewer" / "web" / "src" / "lib" / "url.ts").read_text()
    found = re.search(r'DERIVED_MESH_PREFIX = "([^"]+)"', src)
    assert found, "viewer/web/src/lib/url.ts no longer declares DERIVED_MESH_PREFIX"
    assert found.group(1) == runs_mod.DERIVED_URL_PREFIX, (
        "the browser and the server disagree about the derived URL prefix")


# ---------------------------------------------------------------------------
# Nothing to derive
# ---------------------------------------------------------------------------

def test_a_run_with_no_shader_material_is_not_baked_at_all(tmp_path, cache):
    """The bake exists for `Material(shader=...)` and nothing else.

    A scalar or textured material is already drawn correctly by `materials.ts`,
    so baking one spends 15-95 s of GPU per object reproducing an appearance
    three.js can compute itself — and on the scene layer that is the waste
    multiplied by the catalog.
    """
    ir = _object_ir()
    for part in ir["parts"]:
        part["material"] = dict(_MAT)          # scalar color, no shader
    run = _run(tmp_path, ir)
    with pytest.raises(derived_mod.DerivedUnavailable) as exc:
        derived_mod.resolve(run)
    assert "nothing to derive" in str(exc.value)
    assert derived_mod.status(run, start=False)["status"] == "unavailable"


def test_a_scene_bakes_only_its_shader_authored_objects(tmp_path, cache):
    """A scene mixes shader objects with scalar and textured ones.

    The scalar ones keep their staged meshes and their URDF materials, which is
    already the right picture for them — so they are not entries at all, and
    `objects` counts only what actually needs deriving.
    """
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    plain = json.loads((run / "catalog" / "cat-b.json").read_text())
    for part in plain["parts"]:
        part["material"] = dict(_MAT)
    (run / "catalog" / "cat-b.json").write_text(json.dumps(plain))

    assert [e.ir_id for e in derived_mod.resolve(run)] == ["cat-a"]

    _publish("cat-a", "body", "lid")
    got = derived_mod.status(run, start=False)
    assert got["status"] == "ready", got
    assert got["objects"] == 1 and got["objects_ready"] == 1

    meshes = set(re.findall(r'<mesh filename="([^"]+)"',
                            runs_mod.run_urdf(run, derived=True)))
    assert meshes == {
        "__derived__/cat-a/000_body.glb", "__derived__/cat-a/001_lid.glb",
        # cat-b was never a candidate, and keeps its catalog mesh.
        "catalog/assets/meshes/cat-b.obj"}


def test_a_scene_with_no_shader_anywhere_is_refused_for_that_reason(tmp_path, cache):
    run = _scene_run(tmp_path, instances=(("base", "cat-a"),))
    plain = json.loads((run / "catalog" / "cat-a.json").read_text())
    for part in plain["parts"]:
        part["material"] = dict(_MAT)
    (run / "catalog" / "cat-a.json").write_text(json.dumps(plain))
    with pytest.raises(derived_mod.DerivedUnavailable) as exc:
        derived_mod.resolve(run)
    assert "nothing to derive" in str(exc.value)


def test_bakes_run_one_at_a_time_until_one_has_run_on_a_cpu(monkeypatch):
    """On a GPU a second bake only slows the first; on a CPU two overlap one's
    single-threaded set-up and export with the other's sampling. The device is
    known only from a finished bake, so the first runs alone."""
    import threading

    monkeypatch.setattr(derived_mod, "_slots", {"running": 0, "limit": 1})

    def waiter():
        thread = threading.Thread(target=derived_mod._take_slot, daemon=True)
        thread.start()
        thread.join(0.2)
        return thread

    derived_mod._take_slot()
    second = waiter()
    assert second.is_alive(), "a second bake started before any had run"
    derived_mod._give_slot("GPU")
    second.join(2)
    assert not second.is_alive()
    third = waiter()
    assert third.is_alive(), "two bakes ran at once on a GPU"
    derived_mod._give_slot("CPU")
    third.join(2)
    assert not third.is_alive()
    fourth = waiter()
    assert not fourth.is_alive(), "a CPU bake did not let a second run beside it"
    assert derived_mod._slots == {"running": 2, "limit": derived_mod.CPU_BAKES}


# ---------------------------------------------------------------------------
# Flat colors: what a shader material draws as by default
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_state(monkeypatch):
    """A fresh record of color passes: fixture IRs share their ids across tests."""
    monkeypatch.setattr(derived_mod, "_FLAT_STATE", {})


def _fake_pass(monkeypatch, colours: dict, gate=None) -> list:
    """Stand in for the Blender pass, giving each object asked about the parts in `colours` (`{part: [r, g, b]}`).

    Returns the list of the batches it was asked for. With `gate`, each batch waits on it first."""
    from dycraft.render import viewer_flat as VF

    asked = []

    def flat_pass(run_dir, entries):
        entries = list(entries)
        asked.append([e.ir_id for e in entries])
        if gate is not None:
            gate.wait(5)
        out = {}
        for e in entries:
            parts = {name: {"color": c, "roughness": 0.25, "metallic": 0.75} for name, c in colours.items()}
            path = VF.flat_path(e.ir_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"ir_id": e.ir_id, "version": VF.FLAT_VERSION, "parts": parts}))
            out[e.ir_id] = parts
        return set(out)

    monkeypatch.setattr(derived_mod, "_flat_pass", flat_pass)
    return asked


def _wait_for_flat_pass(before: int) -> None:
    import time

    deadline = time.monotonic() + 5
    while derived_mod.flat_generation() == before:
        assert time.monotonic() < deadline, "the colour pass never finished"
        time.sleep(0.01)


def _link_material(urdf: str, link: str) -> dict:
    """`{"rgba", "roughness", "metalness"}` of one link's visual material, each None when absent."""
    import xml.etree.ElementTree as ET

    node = ET.fromstring(urdf).find(f"link[@name='{link}']/visual/material")
    assert node is not None, f"no visual material on {link}"
    color, rough, metal = (node.find("color"), node.find("pbr/metal/roughness"), node.find("pbr/metal/metalness"))
    return {"rgba": None if color is None else color.get("rgba"),
            "roughness": None if rough is None else float(rough.text),
            "metalness": None if metal is None else float(metal.text)}


def test_flat_colours_go_on_the_shader_parts_of_a_copy():
    from dycraft.render import viewer_flat as VF

    ir = _object_ir()
    ir["parts"][1]["material"] = dict(_MAT)
    read = {"body": {"color": [10, 20, 30], "roughness": 0.25, "metallic": 0.75},
            "lid": {"color": [1, 2, 3], "roughness": 0.1, "metallic": 0.0}}
    got = VF.flat_ir(ir, read)
    body = got["parts"][0]["material"]
    assert (body["color"], body["roughness"], body["metallic"]) == ([10, 20, 30], 0.25, 0.75)
    assert body["shader"] == _SHADER_MAT["shader"]
    # Not a shader part: it keeps what the IR says, whatever was read under its name.
    assert got["parts"][1]["material"] == _MAT
    assert "color" not in ir["parts"][0]["material"], "the IR itself changed"
    assert VF.flat_ir(ir, {}) is ir


def test_a_read_colour_is_the_srgb_byte_the_ir_holds():
    """`_blender_child._srgb_to_linear` turns an IR color into Blender's linear one; the pass turns it back."""
    from dycraft.render import viewer_flat as VF

    def srgb_to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    for byte in (0, 1, 10, 64, 128, 200, 254, 255):
        assert VF._srgb(srgb_to_linear(byte / 255)) == byte
    assert (VF._srgb(-0.5), VF._srgb(3.0)) == (0, 255)


def test_the_urdf_draws_a_shader_part_in_its_flat_colour(tmp_path):
    ir = _object_ir()
    run = _run(tmp_path, ir)
    canonical = runs_mod.run_urdf(run)
    flat = {ir["id"]: {"body": {"color": [255, 0, 0], "roughness": 0.25, "metallic": 0.75}}}
    got = runs_mod.run_urdf(run, flat=flat)
    assert _link_material(got, "body") == {"rgba": "1 0 0 1", "roughness": 0.25, "metalness": 0.75}
    assert _link_material(got, "lid") == _link_material(canonical, "lid")
    assert runs_mod.run_urdf(run, flat={}) == canonical
    assert runs_mod.run_urdf(run, flat={"another-object": flat[ir["id"]]}) == canonical


def test_a_scene_draws_each_catalog_object_in_its_own_flat_colours(tmp_path):
    run = _scene_run(tmp_path, instances=(("base", "cat-a"), ("top", "cat-b")))
    flat = {"cat-a": {"body": {"color": [0, 255, 0], "roughness": 0.25, "metallic": 0.75}}}
    got = runs_mod.run_urdf(run, flat=flat)
    canonical = runs_mod.run_urdf(run)
    assert _link_material(got, "base__body")["rgba"] == "0 1 0 1"
    assert _link_material(got, "top__body") == _link_material(canonical, "top__body")


def test_the_colours_are_read_once_in_the_background_then_from_the_cache(tmp_path, cache, flat_state,
                                                                        monkeypatch):
    import threading

    gate = threading.Event()
    asked = _fake_pass(monkeypatch, {"body": [200, 100, 50]}, gate)
    run = _run(tmp_path)
    before = derived_mod.flat_generation()
    assert derived_mod.flat_colors(run) == ("pending", {})
    # A second ask while the pass runs does not start another.
    assert derived_mod.flat_colors(run) == ("pending", {})
    gate.set()
    _wait_for_flat_pass(before)
    status, flat = derived_mod.flat_colors(run)
    assert status == "ready" and flat["objbake1"]["body"]["color"] == [200, 100, 50]
    assert asked == [["objbake1"]]
    # Another run of the same object reads the cache.
    assert derived_mod.flat_colors(_run(tmp_path, name="again"))[0] == "ready"
    assert asked == [["objbake1"]]


def test_a_run_with_no_shader_part_has_no_colour_pass(tmp_path, cache, flat_state, monkeypatch):
    asked = _fake_pass(monkeypatch, {})
    ir = _object_ir()
    for part in ir["parts"]:
        part["material"] = dict(_MAT)
    assert derived_mod.flat_colors(_run(tmp_path, ir)) == ("none", {})
    assert asked == []


def test_a_failed_pass_leaves_the_parts_grey_and_is_not_retried_per_request(tmp_path, cache, flat_state):
    """As on a machine with no Blender: conftest makes every pass fail under test."""
    run = _run(tmp_path)
    before = derived_mod.flat_generation()
    assert derived_mod.flat_colors(run)[0] == "pending"
    _wait_for_flat_pass(before)
    after = derived_mod.flat_generation()
    assert derived_mod.flat_colors(run) == ("error", {})
    assert derived_mod.flat_colors(run) == ("error", {})
    assert derived_mod.flat_generation() == after, "a failed pass was started again"


def test_the_urdf_route_says_pending_then_serves_the_colours(tmp_path, cache, flat_state, monkeypatch):
    """The viewport draws the first document gray and fetches it again once the header stops saying pending."""
    import threading

    from starlette.testclient import TestClient

    from dycraft.viewer.app import create_app

    gate = threading.Event()
    _fake_pass(monkeypatch, {"body": [255, 0, 0]}, gate)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _run(runs_dir, _object_ir(), "obj")
    url = "/api/runs/obj/files/model.urdf"
    with TestClient(create_app(runs_dir=runs_dir)) as client:
        before = derived_mod.flat_generation()
        first = client.get(url)
        assert first.headers["x-dycraft-flat-colors"] == "pending"
        assert _link_material(first.text, "body")["rgba"] != "1 0 0 1"
        again = client.get(url)
        assert again.headers["x-dycraft-flat-colors"] == "pending" and again.text == first.text
        gate.set()
        _wait_for_flat_pass(before)
        for variant in (url, url + "?derived=1"):
            done = client.get(variant)
            assert done.headers["x-dycraft-flat-colors"] == "ready"
            assert _link_material(done.text, "body")["rgba"] == "1 0 0 1"
        assert client.get("/api/runs/obj/urdf").text == client.get(url).text


@pytest.mark.expensive
def test_the_pass_reads_the_colour_the_glb_export_gives(tmp_path, cache):
    """A real pass in its own interpreter, against `flat_material`'s reading of a constant Base Color."""
    from dycraft.render import viewer_flat as VF

    _blender_or_skip()
    ir = _object_ir("objflat1")
    ir["parts"][0]["material"] = {"alpha": 1.0, "shader": {
        "name": "tan",
        "body": "def tan(nodes, links, bsdf):\n"
                "    bsdf.inputs['Base Color'].default_value = (0.5, 0.25, 0.1, 1.0)\n"
                "    bsdf.inputs['Roughness'].default_value = 0.4\n"
                "    bsdf.inputs['Metallic'].default_value = 0.2\n"}}
    run = _run(tmp_path, ir)
    assert derived_mod._flat_pass(run, derived_mod.resolve(run)) == {"objflat1"}
    body = VF.read("objflat1")["body"]
    assert body["color"] == [VF._srgb(0.5), VF._srgb(0.25), VF._srgb(0.1)]
    assert body["roughness"] == pytest.approx(0.4) and body["metallic"] == pytest.approx(0.2)
