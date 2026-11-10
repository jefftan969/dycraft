"""Everything that is given a run folder finds a run laid out for a reader (records in `src/`) as it finds an older,
flat one: the launcher, the viewer, `--prior-animation`, `--preload-asset`, the scorer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft import run_layout as L


def _w(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _sealed_film(inner: Path) -> dict:
    """A film's sealed bundle, with a manifest that validates."""
    from dycraft.animation.blender.bundle import make_manifest
    _w(inner / "meta.json", json.dumps({"layer": "unified_animation", "outcome": "committed", "run_id": "r1"}))
    _w(inner / "bundle" / "animation.blend", "blend")
    _w(inner / "bundle" / "film" / "clip.mp4", "film")
    _w(inner / "bundle" / "design_state" / "baseline.json", "{}")
    _w(inner / "bundle" / "evaluation" / "index.json", json.dumps({"n_frames": 2}))
    ir = make_manifest(inner / "bundle", scene_ref="scene", evaluation={
        "fps": 24.0, "n_frames": 2, "frame_start": 0, "blender_version": "test", "declarations": {"events": []}})
    _w(inner / "ir.json", json.dumps(ir))
    return ir


def test_ensure_run_dir_lays_a_launch_out_and_leaves_a_sub_run_flat(tmp_path):
    from dycraft.agent.agent_runner import ensure_run_dir
    _, inner = ensure_run_dir(tmp_path, "r1", reader=True)
    assert inner == (tmp_path / "r1" / "src").resolve()
    _, flat = ensure_run_dir(tmp_path, "sub", reader=False)
    assert flat == (tmp_path / "sub").resolve()
    assert not list(inner.iterdir()) and not list(flat.iterdir()), "no folder is made before it is written to"


def test_the_viewer_lists_one_entry_per_run_in_either_layout(tmp_path):
    from dycraft.viewer import runs as R
    runs = tmp_path / "runs"
    _w(runs / "old" / "meta.json", json.dumps({"layer": "object", "run_id": "old"}))
    _w(runs / "new" / "src" / "meta.json", json.dumps({"layer": "object", "run_id": "new"}))
    _w(runs / "new" / "src" / "sub_runs" / "scene_rewrite" / "meta.json", json.dumps({"layer": "plain"}))
    _w(runs / "new" / "README.md", "# new")
    listed = [r["run_id"] for r in R.list_runs(runs)]
    assert listed == ["new", "new/src/sub_runs/scene_rewrite", "old"]
    assert R.resolve_run(runs, "new") == (runs / "new" / "src").resolve()
    assert R.resolve_run(runs, "old") == (runs / "old").resolve()
    detail = R.run_detail(R.resolve_run(runs, "new"), "new")
    assert [s["run_id"] for s in detail["sub_runs"]] == ["new/src/sub_runs/scene_rewrite"]
    assert R.resolve_run(runs, detail["sub_runs"][0]["run_id"]) is not None


@pytest.mark.parametrize("given", ["folder", "top_blend", "inner"])
def test_prior_animation_takes_the_run_folder_or_its_top_level_blend(tmp_path, given):
    from dycraft.animation.blender.bundle import stage_prior
    top = tmp_path / "runs" / "r1"
    _sealed_film(top / "src")
    _w(top / "animation.blend", "blend")
    _w(top / "film.mp4", "film")
    source = {"folder": top, "top_blend": top / "animation.blend", "inner": top / "src"}[given]
    stage_prior(source, tmp_path / "staged")
    staged = sorted(p.relative_to(tmp_path / "staged").as_posix() for p in (tmp_path / "staged").rglob("*")
                    if p.is_file())
    # The bundle's own staging (its film and evaluation are left out), not the reader's folder.
    assert staged == ["animation.blend", "design_state/baseline.json"]


def test_preload_asset_finds_the_bundle_of_a_run_folder(tmp_path):
    from dycraft.agent.assets.preload_asset import parse_spec
    _w(tmp_path / "r1" / "src" / "meta.json", "{}")
    assert parse_spec(f"stool={tmp_path / 'r1'}") == ("stool", tmp_path / "r1" / "src")
    _w(tmp_path / "flat" / "meta.json", "{}")
    assert parse_spec(f"stool={tmp_path / 'flat'}") == ("stool", tmp_path / "flat")


@pytest.mark.parametrize("given", ["folder", "top_blend"])
def test_the_scorer_reads_a_run_folder_from_its_records(tmp_path, given):
    from dycraft.benchmark.score.inputs import resolve
    top = tmp_path / "runs" / "r1"
    _sealed_film(top / "src")
    _w(top / "animation.blend", "blend")
    got = resolve(top if given == "folder" else top / "animation.blend")
    assert got.blend == (top / "src" / "bundle" / "animation.blend").resolve()
    assert got.baseline is not None
    assert got.label == "runs__r1"


def test_the_export_writes_its_glb_at_the_top(tmp_path):
    from dycraft.export.glb import resolve
    top = tmp_path / "r1"
    _w(top / "src" / "meta.json", json.dumps({"layer": "object"}))
    _w(top / "src" / "ir.json", json.dumps({"id": "o", "parts": []}))
    source = resolve(top)
    assert (source.kind, source.run_dir, source.out_dir) == ("object", (top / "src").resolve(), top.resolve())
