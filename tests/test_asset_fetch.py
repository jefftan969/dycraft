"""`python -m dycraft.install library|films`: what the fetch installs, and that a later version installs in place.

A fake Hub client copies from local stand-ins for the two datasets (dycraft-assets for the library, dycraft-films for
the films), so no test reaches the network.
"""
import fnmatch
import json
import shutil
import sys
from pathlib import Path

import pytest

from dycraft.agent.assets import _fetch, asset_library


def _sibling(name):
    # pytest runs with --import-mode=importlib, so a sibling test module is loaded by path rather than by name.
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle(root: Path, name: str):
    """A valid bundle (the index reads its facts), as `import_urdf.convert` writes one."""
    _sibling("test_preload_asset").make_bundle(root, name)
    (root / "bundles" / name / "import.json").write_text("{}")


def _dataset(repos: dict, version: str, names: dict, films=()):
    """The library at the root of the assets dataset; films under films/ in the films dataset."""
    root = repos[_fetch.ASSETS_REPO]
    if (root / "bundles").exists():
        shutil.rmtree(root / "bundles")
    for n in names:
        _bundle(root, n)
    g1 = root / "robots" / "g1_description"
    g1.mkdir(parents=True, exist_ok=True)
    (g1 / "g1_29dof_mode_15_with_dex1_1.urdf").write_text("<robot/>")
    (root / "catalog.json").write_text(json.dumps(
        {"library_version": version, "assets": {n: {"category": c} for n, c in names.items()}}))
    for f in films:
        (repos[_fetch.FILMS_REPO] / "films" / f).mkdir(parents=True, exist_ok=True)
        (repos[_fetch.FILMS_REPO] / "films" / f / "film.blend").write_text("x")


class FakeHub:
    def __init__(self, repos: dict):
        self.repos, self.calls = repos, []

    def snapshot_download(self, repo_id, repo_type, allow_patterns, local_dir, revision=None):
        self.calls.append((repo_id, repo_type, list(allow_patterns), revision))
        repo = self.repos[repo_id]
        for f in repo.rglob("*"):
            rel = f.relative_to(repo).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dst = Path(local_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
        return local_dir


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    repos = {_fetch.ASSETS_REPO: tmp_path / "assets", _fetch.FILMS_REPO: tmp_path / "films"}
    for d in repos.values():
        d.mkdir()
    monkeypatch.setenv("DYCRAFT_HOME", str(home))
    hub = FakeHub(repos)
    monkeypatch.setattr(_fetch, "_hub", lambda: hub)
    return home, repos, hub


def test_library_installs_where_the_library_is_read_and_a_later_version_installs_in_place(env):
    """The bundles land where asset_library reads them, the G1 description where the bare arm looks, and a second
    version adds its new characters, reports the withdrawn one, and unlinks it only with --prune."""
    home, repos, hub = env
    _dataset(repos, "1", {"a": "person", "b": "animal"})
    assert _fetch.run_library() == 0
    lib = home / "assets" / "imported"
    assert [p.name for p in asset_library.bundles(lib)] == ["a", "b"]
    assert (home / "assets" / "robots" / "g1_description" / "g1_29dof_mode_15_with_dex1_1.urdf").is_file()
    assert json.loads((lib / "catalog.json").read_text())["library_version"] == "1"
    assert {c[0] for c in hub.calls} == {_fetch.ASSETS_REPO} and hub.calls[0][1] == "dataset"

    _dataset(repos, "2", {"a": "person", "c": "robot"})
    assert _fetch.run_library() == 0
    assert {p.name for p in asset_library.bundles(lib)} == {"a", "b", "c"}       # b reported, still linked
    assert _fetch.run_library(prune=True) == 0
    assert {p.name for p in asset_library.bundles(lib)} == {"a", "c"}


def test_a_clean_install_tells_agents_about_the_library(env, monkeypatch, capsys):
    """The prompt shows its asset-library section only when the library has an INDEX.md, and the environment check
    reads the same test: `install library` writes the index, and rewrites it after --prune."""
    from dycraft import environment_test
    from dycraft.agent import agent_runner
    monkeypatch.delenv(asset_library.LIBRARY_ENV, raising=False)
    home, repos, hub = env
    lib = home / "assets" / "imported"
    _dataset(repos, "1", {"a": "person", "b": "animal"})
    assert not agent_runner._has_asset_library()
    assert _fetch.run_library() == 0
    assert "INDEX.md lists 2 characters for agents" in capsys.readouterr().out
    assert agent_runner._has_asset_library()
    assert "### 2.2 Look in the asset library first" in agent_runner.compose_system_prompt("animation")
    assert environment_test.check_library(None).status == environment_test.PASS
    _dataset(repos, "2", {"a": "person"})
    assert _fetch.run_library(prune=True) == 0
    assert [r["name"] for r in json.loads((lib / asset_library.INDEX_JSON).read_text())["assets"]] == ["a"]
    (lib / asset_library.INDEX_MD).unlink()                     # linked, but nothing tells agents about it
    assert not agent_runner._has_asset_library()
    assert "### 2.2 Look in the asset library first" not in agent_runner.compose_system_prompt("animation")
    r = environment_test.check_library(None)
    assert r.status == environment_test.WARN and "no INDEX.md" in r.summary


def test_the_index_is_replaced_whole_so_a_reader_never_sees_part_of_it(tmp_path):
    """write_index renames a finished file over the old one: a reader holding the old INDEX.md still reads all of
    it, and no temporary file is left behind."""
    _bundle(tmp_path, "a")
    lib = tmp_path / "bundles"
    asset_library.write_index(lib)
    old = (lib / asset_library.INDEX_MD).read_text()
    with open(lib / asset_library.INDEX_MD) as reader:
        _bundle(tmp_path, "b")
        asset_library.write_index(lib)
        assert reader.read() == old                   # truncated in place, this would read the new file or nothing
    assert "`b`" in (lib / asset_library.INDEX_MD).read_text()
    assert sorted(p.name for p in lib.iterdir() if p.name.startswith(".")) == []


def test_names_and_category_choose_what_is_fetched(env, capsys):
    home, repos, hub = env
    _dataset(repos, "1", {"a": "person", "b": "animal", "c": "animal"})
    assert _fetch.run_library(category="animal") == 0
    assert sorted(p.name for p in asset_library.bundles(home / "assets" / "imported")) == ["b", "c"]
    assert _fetch.run_library(names=["a"]) == 0
    assert _fetch.run_library(names=["nope"]) == 1
    assert "not in library 1: nope" in capsys.readouterr().out


def test_your_own_bundle_of_the_same_name_is_never_replaced(env, capsys):
    home, repos, hub = env
    _dataset(repos, "1", {"a": "person"})
    mine = home / "assets" / "imported" / "a"
    mine.mkdir(parents=True)
    (mine / "ir.json").write_text(json.dumps({"id": "mine"}))
    assert _fetch.run_library() == 0
    assert not mine.is_symlink() and json.loads((mine / "ir.json").read_text())["id"] == "mine"
    out = capsys.readouterr().out
    assert "a: skipped" in out
    # The stand-in is not a readable bundle: the index leaves it out and says so, and is still written.
    assert "INDEX.md lists 0 characters" in out and (home / "assets" / "imported" / "INDEX.md").is_file()


def test_films_are_linked_under_their_dataset_names_and_a_missing_brief_fails(env, capsys):
    """The scorer reads the brief id from the film's directory name, so the name is kept."""
    home, repos, hub = env
    _dataset(repos, "1", {}, films=["B05_tea", "B13_dog"])
    assert _fetch.run_films(["b05"]) == 0
    assert hub.calls[-1][0] == _fetch.FILMS_REPO
    assert (home / "films" / "B05_tea" / "film.blend").is_file()
    assert not (home / "films" / "B13_dog").exists()
    said = capsys.readouterr().out
    # Both ways to use an installed film, and no staging step: the viewer prepares a film itself on first open.
    assert "benchmark.score" in said and "dycraft.viewer --runs-dir" in said
    assert "benchmark.stage" not in said
    assert _fetch.run_films(["B99"]) == 1
    assert "not in the dataset: B99" in capsys.readouterr().out


_REAL_HUB = _fetch._hub


def test_without_huggingface_hub_the_step_says_what_to_install(env, monkeypatch, capsys):
    monkeypatch.setattr(_fetch, "_hub", _REAL_HUB)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)      # makes the import fail
    assert _fetch.run_library() == 1
    assert "pip install huggingface_hub" in capsys.readouterr().out


def test_a_private_dataset_without_access_says_how_to_log_in(env, monkeypatch, capsys):
    """The Hub client's 401 is a traceback by itself; the step names the dataset and the login instead."""
    class NoAccess(Exception):
        response = type("R", (), {"status_code": 401})()
    hub = env[2]
    monkeypatch.setattr(hub, "snapshot_download", lambda **kw: (_ for _ in ()).throw(NoAccess("401 Unauthorized")))
    assert _fetch.run_library() == 1
    out = capsys.readouterr().out
    assert "dycraft-assets" in out and "hf auth login" in out
