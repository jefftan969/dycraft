"""The asset library a scene or animation run retrieves its characters and robots from.

A run is given `references/assets/` (`INDEX.md` plus the bundles) and
`load_asset("<bundle>")` stages the one it picks. These tests protect the three
things that make that safe: the index only offers what the catalog allows,
`load_asset` puts a retrieved bundle through the same checks and layout as
`--preload-asset`, and everything downstream that learns about assets from the
launch (the manifest, the IK robot list) also learns about the ones the agent
fetched itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.agent import stage_references
from dycraft.agent.assets import asset_library as AL
from dycraft.agent.cad import object_worker as W
from dycraft.agent.assets.preload_asset import (
    load_preloaded_manifest, stage_library_asset, stage_preloaded_assets)

def _sibling(name):
    # pytest runs with --import-mode=importlib, so a sibling test module is
    # loaded by path rather than imported by name.
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_bundle = _sibling("test_preload_asset").make_bundle

pytestmark_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")


def _library(tmp_path, *names):
    for n in names:
        make_bundle(tmp_path, n)
    return tmp_path / "bundles"


# ---------------------------------------------------------------- the index

def test_the_index_offers_catalogued_and_uncatalogued_bundles_but_not_hidden(tmp_path):
    lib = _library(tmp_path, "robo", "pup", "old")
    cur = {"pup": {"category": "animal", "description": "a small dog", "tags": ["dog"]},
           "old": {"hidden": "superseded by robo"}}
    rows = {r["name"]: r for r in AL.build_index(lib, cur)}
    assert set(rows) == {"robo", "pup"}, "a hidden bundle must not be offered"
    assert rows["pup"]["description"] == "a small dog"
    assert "description" not in rows["robo"], "an uncatalogued bundle is offered with its facts only"
    md = AL.render_index_md(list(rows.values()))
    assert "`pup` — a small dog" in md and "## animal" in md
    assert "`old`" not in md


def test_a_hidden_or_unknown_bundle_is_refused_with_the_reason(tmp_path):
    lib = _library(tmp_path, "old")
    with pytest.raises(AL.AssetLibraryError, match="superseded by robo"):
        AL.resolve_library_asset("old", lib, {"old": {"hidden": "superseded by robo"}})
    with pytest.raises(AL.AssetLibraryError, match="no asset named"):
        AL.resolve_library_asset("nope", lib, {})
    with pytest.raises(AL.AssetLibraryError, match="not a bundle name"):
        AL.resolve_library_asset("../old", lib, {})


def test_an_installed_library_describes_its_own_characters_and_a_hidden_entry_still_holds(tmp_path):
    """`install library` puts its version's catalog.json beside the bundles: its descriptions and caveats reach the
    index with no code change, they override the hand-written ones, and the hand-written `hidden` still holds."""
    lib = _library(tmp_path, "robo", "pup", "old")
    (lib / "catalog.json").write_text(json.dumps({"library_version": "2", "assets": {
        "robo": {"category": "robot", "description": "a new robot"},
        "pup": {"category": "animal", "description": "a small dog, corrected", "caveats": ["sits only on its haunches"]},
        "old": {"category": "person", "description": "an older build"}}}))
    tracked = tmp_path / "tracked.json"
    tracked.write_text(json.dumps({"assets": {"pup": {"category": "animal", "description": "a small dog", "tags": ["dog"]},
                                              "old": {"hidden": "superseded by robo"}}}))
    cur = AL.load_curation(tracked, lib)
    rows = {r["name"]: r for r in AL.build_index(lib, cur)}
    assert set(rows) == {"robo", "pup"}
    assert rows["robo"]["description"] == "a new robot"
    assert rows["pup"]["description"] == "a small dog, corrected" and rows["pup"]["tags"] == ["dog"]
    assert "  caveats: sits only on its haunches" in AL.render_index_md(list(rows.values()))   # what agents read


def test_every_bundle_in_the_checkouts_library_is_catalogued():
    """A newly imported bundle with no catalog entry is still offered, but with
    no description an agent cannot find it by what it is. Catch it here."""
    lib = AL.library_root()
    if lib is None:
        pytest.skip("this checkout has no assets/imported")
    cur = AL.load_curation()
    missing = [b.name for b in AL.bundles(lib) if b.name not in cur]
    assert not missing, f"add these to catalogs/asset_library.json: {missing}"
    for name, entry in cur.items():
        if not entry.get("hidden"):
            assert entry.get("description") and entry.get("category"), name


# ---------------------------------------------------------------- staging

def test_a_library_asset_is_staged_beside_preloaded_ones(tmp_path):
    lib = _library(tmp_path, "robo", "pup")
    run = tmp_path / "run"
    stage_preloaded_assets([f"pup={lib / 'pup'}"], run)
    rec = stage_library_asset("robo", lib / "robo", run)
    manifest = load_preloaded_manifest(run)
    assert set(manifest) == {"pup", "robo"}, "retrieval must merge, not replace"
    assert manifest["robo"]["retrieved"] is True and not manifest["pup"].get("retrieved")
    for rel in rec["assets"]:
        assert (run / rel).is_file()


def test_the_library_is_linked_into_every_run(tmp_path, monkeypatch):
    lib = _library(tmp_path, "robo")
    monkeypatch.setenv(AL.LIBRARY_ENV, str(lib))
    run = tmp_path / "run"
    run.mkdir()
    stage_references(run)
    link = run / "references" / "assets"
    assert link.is_symlink() and link.resolve() == lib.resolve()


def test_ik_finds_a_robot_the_agent_retrieved(tmp_path):
    """`h.ik` names each URDF robot in the run; one fetched mid-run exists only
    in the manifest of the build's source tree, not in `meta.json`."""
    from dycraft.animation.blender.solvers import robot_sources
    lib = _library(tmp_path, "robo")
    (lib / "robo" / "import.json").write_text(json.dumps(
        {"source_urdf": "/x/robo.urdf", "package_root": "/x"}))
    run = tmp_path / "run"
    stage_library_asset("robo", lib / "robo", run / "native/work/design_state/source")
    assert robot_sources(run) == {"robo": {"urdf": "/x/robo.urdf", "package_root": "/x"}}


# ---------------------------------------------------------------- load_asset

@pytestmark_cad
def test_load_asset_retrieves_a_library_bundle_on_demand(tmp_path, monkeypatch):
    lib = _library(tmp_path, "robo")
    monkeypatch.setenv(AL.LIBRARY_ENV, str(lib))
    # load_asset reads the library variable inside the CAD child, and a warm
    # child keeps the environment it was spawned with: build in a fresh one.
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    run = tmp_path / "run"
    run.mkdir()
    src = '''
r = load_asset("robo")
scene = Scene("bay")
scene.set_room(6.0, 5.0, 3.0)
scene.place(r, name="the_robot", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''
    out = W.run_object_source(src, run_dir=run, layer="scene")
    assert out["ok"], out.get("build_error") or out.get("sandbox_violation")
    manifest = load_preloaded_manifest(run)
    assert manifest["robo"]["retrieved"] is True
    assert manifest["robo"]["ir_hash"] in out["object_catalog"]


@pytestmark_cad
def test_load_asset_points_at_the_index_for_an_unknown_name(tmp_path, monkeypatch):
    lib = _library(tmp_path, "robo")
    monkeypatch.setenv(AL.LIBRARY_ENV, str(lib))
    # load_asset reads the library variable inside the CAD child, and a warm
    # child keeps the environment it was spawned with: build in a fresh one.
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    run = tmp_path / "run"
    run.mkdir()
    out = W.run_object_source('ir = load_asset("g1")', run_dir=run, layer="scene")
    assert not out["ok"]
    blob = str(out.get("build_error"))
    assert "no asset named 'g1'" in blob and "INDEX.md" in blob, blob


def test_the_library_is_found_where_runs_link_it_from(tmp_path, monkeypatch):
    """Without `$DYCRAFT_ASSET_LIBRARY` the library is `assets/imported` under the user-state root, the directory
    `stage_references` links into every run: under `$DYCRAFT_HOME`, or an installed copy's launch directory."""
    lib = tmp_path / "home" / "assets" / "imported"
    make_bundle(lib, name="fake_bundle")
    monkeypatch.delenv(AL.LIBRARY_ENV, raising=False)
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path / "home"))
    assert AL.library_root() == lib.resolve()
    run = tmp_path / "run"
    run.mkdir()
    stage_references(run)
    assert (run / "references" / "assets").resolve() == AL.library_root()
