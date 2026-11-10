"""A film installed from the published set and staged by `benchmark.stage`, as the README tells a reader to: the
viewer lists it through the link `install films` makes, and draws it from its own track with no frozen scene."""
from __future__ import annotations

import json
from pathlib import Path

from dycraft.viewer import runs as R


def _w(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _staged_film(root: Path) -> Path:
    """A native film as `benchmark.stage` leaves an installed one: its manifest and evaluation, no scene.json."""
    from dycraft.animation.blender.bundle import make_manifest
    _w(root / "bundle" / "animation.blend", "blend")
    _w(root / "bundle" / "film" / "clip.mp4", "film")
    _w(root / "bundle" / "evaluation" / "index.json", json.dumps({"n_frames": 2}))
    ir = make_manifest(root / "bundle", scene_ref="scene", evaluation={
        "fps": 24.0, "n_frames": 2, "frame_start": 0, "blender_version": "test", "declarations": {"events": []}})
    _w(root / "ir.json", json.dumps(ir))
    _w(root / "meta.json", json.dumps({"layer": "unified_animation", "run_id": root.name}))
    return root


def test_a_film_linked_into_runs_dir_is_listed_and_served(tmp_path):
    mirror = _staged_film(tmp_path / "mirror" / "films" / "B05_film")
    films = tmp_path / "films"
    films.mkdir()
    (films / "B05_film").symlink_to(mirror, target_is_directory=True)
    assert [r["run_id"] for r in R.list_runs(films)] == ["B05_film"]
    assert R.resolve_run(films, "B05_film") == films / "B05_film"


def test_a_link_is_followed_only_at_the_top_and_never_walked_through(tmp_path):
    runs = tmp_path / "runs"
    _staged_film(runs / "group" / "real")
    (runs / "group" / "again").symlink_to(runs / "group" / "real", target_is_directory=True)
    (runs / "loop").symlink_to(runs, target_is_directory=True)
    (runs / "twice").symlink_to(runs / "group" / "real", target_is_directory=True)
    listed = [r["run_id"] for r in R.list_runs(runs)]
    assert listed == ["group/real"], "a deeper link was followed, the loop walked, or a run listed twice"
    assert R.resolve_run(runs, "group/again") is None
    assert R.resolve_run(runs, "loop/group/real") is None


def test_a_staged_film_without_a_scene_has_a_timeline_and_a_bare_urdf(tmp_path):
    run = _staged_film(tmp_path / "B05_film")
    urdf = R.run_urdf(run)
    assert '<link name="world"' in urdf and "<visual" not in urdf
    timeline = R.run_animation(run)
    assert timeline["n_frames"] == 2 and timeline["native"]["version"]
    assert "film_camera" in timeline


def test_a_link_back_into_the_tree_is_neither_listed_nor_served(tmp_path):
    """The run a link into the tree names is listed and served under its own path only, so the listing and
    `resolve_run` agree."""
    runs = tmp_path / "runs"
    _staged_film(runs / "group" / "real")
    (runs / "twice").symlink_to(runs / "group" / "real", target_is_directory=True)
    (runs / "loop").symlink_to(runs, target_is_directory=True)
    assert [r["run_id"] for r in R.list_runs(runs)] == ["group/real"]
    assert R.resolve_run(runs, "twice") is None
    assert R.resolve_run(runs, "loop") is None
    assert R.resolve_run(runs, "group/real") == runs.resolve() / "group" / "real"


def test_the_walk_resolves_only_the_links_it_meets(tmp_path, monkeypatch):
    """Resolving a path costs a system call per component, so on a large runs folder the walk resolves only what
    it has to: the folder itself and the links at its top."""
    runs = tmp_path / "runs"
    for name in ("a", "b", "group/c", "group/d"):
        _staged_film(runs / name)
    (runs / "notes").mkdir()
    outside = _staged_film(tmp_path / "mirror" / "e")
    (runs / "e").symlink_to(outside, target_is_directory=True)
    resolved: list[Path] = []
    real_resolve = Path.resolve

    def spy(self, *args, **kwargs):
        resolved.append(self)
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", spy)
    listed = sorted(p.relative_to(runs).as_posix() for p in R._iter_run_dirs(runs))
    assert listed == ["a", "b", "e", "group/c", "group/d"]
    assert sorted(map(str, resolved)) == sorted([str(runs), str(runs / "e")])


# ---- A film exactly as `install films` leaves it, played with no stage step ------------------------------------------

def _installed_film(root: Path) -> Path:
    """The dataset's layout: the video, the Blender project, the plan, and `run.json`, with no run records."""
    _w(root / "bundle" / "animation.blend", "blend")
    _w(root / "bundle" / "plan.json", "{}")
    _w(root / "film.mp4", "film")
    _w(root / "prompt.md", "A robot carries a panel.")
    _w(root / "run.json", json.dumps({"brief": "Brief 5", "title": "G1 carries a panel", "agent": {"model": "m"},
                                      "film": {"file": "film.mp4", "frames": 2}}))
    return root


def _fake_prepare(calls: list, *, fail: bool = False):
    """Stands in for `benchmark.stage.prepare`, which needs Blender: writes what it writes, under `dest` only."""
    def prepare(run_dir, dest, *, timeout):
        calls.append(Path(run_dir))
        if fail:
            raise RuntimeError("the Blender child exited 1")
        from dycraft.animation.blender.bundle import make_manifest
        evaluation = Path(dest) / "bundle" / "evaluation"
        for frame in range(2):
            _w(evaluation / f"{frame:06d}.bin", "frame")
        _w(evaluation / "index.json", json.dumps({"n_frames": 2}))
        ir = make_manifest(Path(dest) / "bundle", scene_ref="scene", evaluation={
            "fps": 24.0, "n_frames": 2, "frame_start": 0, "blender_version": "test", "declarations": {"events": []}})
        ir["film"] = "film.mp4"
        ir["blend"] = "bundle/animation.blend"
        _w(Path(dest) / "ir.json", json.dumps(ir))
    return prepare


def _wait_ready(film: Path, F) -> dict:
    import time
    for _ in range(200):
        status = F.status(film)
        if status["state"] != "preparing":
            return status
        time.sleep(0.02)
    raise AssertionError("never finished preparing")


def _films_dir(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path / "home"))
    mirror = _installed_film(tmp_path / "home" / ".datasets" / "dycraft-films" / "films" / "B05_film")
    films = tmp_path / "home" / "films"
    films.mkdir(parents=True)
    (films / "B05_film").symlink_to(mirror, target_is_directory=True)
    return films, mirror


def _snapshot(folder: Path) -> list:
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in folder.rglob("*"))


def test_an_installed_film_is_listed_before_it_is_prepared_and_nothing_is_prepared_for_it(tmp_path, monkeypatch):
    from dycraft.viewer import films as F
    films, _ = _films_dir(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("dycraft.benchmark.stage.prepare", _fake_prepare(calls))
    [row] = R.list_runs(films)
    assert row["run_id"] == "B05_film" and row["label"] == "Brief 5 · G1 carries a panel"
    assert row["prepare"] == "needed" and row["ir_kind"] == "animation" and row["n_frames"] == 2
    assert row["installed"] is True, "the inspector can't tell it is a published film"
    assert R.resolve_run(films, "B05_film") is None, "served before it was prepared"
    assert R.resolve_film(films, "B05_film") == films.resolve() / "B05_film"
    assert F.status(films / "B05_film", start=False) == {"state": "unprepared"}
    assert calls == [] and not (tmp_path / "home" / F.PREPARED_DIR).exists()


def test_the_first_open_prepares_it_once_into_the_viewer_s_folder_and_never_into_the_film(tmp_path, monkeypatch):
    from dycraft.viewer import animation as A
    from dycraft.viewer import films as F
    films, mirror = _films_dir(tmp_path, monkeypatch)
    before = _snapshot(mirror)
    calls = []
    monkeypatch.setattr("dycraft.benchmark.stage.prepare", _fake_prepare(calls))
    assert F.status(films / "B05_film")["state"] == "preparing"
    assert _wait_ready(films / "B05_film", F) == {"state": "ready"}
    assert F.status(films / "B05_film") == {"state": "ready"}
    assert len(calls) == 1, "prepared more than once"
    assert _snapshot(mirror) == before, "the film's folder changed"

    run = R.resolve_run(films, "B05_film")
    assert run is not None and run.parent == (tmp_path / "home" / F.PREPARED_DIR).resolve()
    assert R.resolve_file(run, "film.mp4") == mirror.resolve() / "film.mp4"
    assert R.resolve_file(run, "ir.json") == run / "ir.json"
    _w(mirror.parent / "beside.txt", "not the film's")
    assert R.resolve_file(run, "../beside.txt") is None, "a file outside the film's folder was served"
    assert R.resolve_file(run, "../../beside.txt") is None
    assert A.clip_rel(run, json.loads((run / "ir.json").read_text())) == "film.mp4"
    [row] = R.list_runs(films)
    assert "prepare" not in row and row["label"] == "Brief 5 · G1 carries a panel" and row["model"] == "m"
    assert row["installed"] is True


def test_a_film_that_cannot_be_prepared_says_why_and_is_not_tried_again(tmp_path, monkeypatch):
    from dycraft.viewer import films as F
    films, _ = _films_dir(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("dycraft.benchmark.stage.prepare", _fake_prepare(calls, fail=True))
    F.status(films / "B05_film")
    status = _wait_ready(films / "B05_film", F)
    assert status["state"] == "failed" and "Blender child exited 1" in status["error"]
    assert F.status(films / "B05_film")["state"] == "failed"
    assert len(calls) == 1
    assert not list((tmp_path / "home" / F.PREPARED_DIR).iterdir()), "a failed preparation left a folder"
