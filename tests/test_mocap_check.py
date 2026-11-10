"""The environment check reports the optional motion-capture libraries, and never fails without them.

The fixture sets `DYCRAFT_HOME` rather than patching `state_path`: `environment_test` loads its own copy of `_home`
by path, while `runtime.mocap_libraries` reads `dycraft._home`, and both follow the variable.
"""
from types import SimpleNamespace

from dycraft import environment_test as E
from dycraft.animation.blender import runtime


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    return tmp_path / "assets" / "mocap"


def _write(path, text="HIERARCHY\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fetched(lib, cmu=0, dogs=0, index=True):
    """The layout `python -m dycraft.install mocap [--dog]` writes, stamps included."""
    for i in range(cmu):
        _write(lib / "cmu" / "data" / f"{i:02d}" / f"{i:02d}_01.bvh")
    if cmu:
        _write(lib / "cmu" / "README.md", "CMU\n")
        _write(lib / "cmu" / ".verified", "digest\n")
    for i in range(dogs):
        _write(lib / "mann_dog" / "data" / f"D1_{i:03d}.bvh")
    if dogs:
        _write(lib / "mann_dog" / ".verified", "digest\n")
        if index:
            _write(lib / "mann_dog" / "index.txt", "D1_000 walk 0 60\n")


def _check():
    return E.check_mocap(SimpleNamespace(facts={}))


def test_absent_data_is_an_optional_capability_not_a_failure(tmp_path, monkeypatch):
    lib = _home(tmp_path, monkeypatch)
    result = _check()
    assert result.status == E.UNAVAILABLE
    assert result.summary.startswith("no library in ")
    assert "people and dogs" not in result.lost
    assert result.fix[0].startswith("python -m dycraft.install mocap fetches the CMU library (--dog adds the MANN ")
    assert result.fix[0].endswith(f"or put any capture library in its own folder under {lib.resolve()}")


def test_an_empty_or_hidden_only_folder_is_no_library(tmp_path, monkeypatch):
    lib = _home(tmp_path, monkeypatch)
    (lib / "cmu").mkdir(parents=True)                       # a folder with no file, as a failed fetch could leave
    _write(lib / ".cmu.partial" / "01_01.bvh")              # a hidden folder: a fetch still in progress
    _write(lib / "notes" / ".DS_Store", "")                 # a folder holding only a hidden file
    assert _check().status == E.UNAVAILABLE


def test_each_library_is_reported_with_its_files_by_extension(tmp_path, monkeypatch):
    lib = _home(tmp_path, monkeypatch)
    _fetched(lib, cmu=3, dogs=2)
    result = _check()
    assert result.status == E.PASS
    assert result.summary == "cmu 3 .bvh, 1 .md; mann_dog 2 .bvh, 1 .txt"


def test_a_library_a_user_adds_counts_like_the_fetched_ones(tmp_path, monkeypatch):
    lib = _home(tmp_path, monkeypatch)
    _write(lib / "studio_takes" / "session1" / "take_01.FBX")
    _write(lib / "studio_takes" / "LICENSE")
    assert _check().summary == "studio_takes 1 .fbx, 1 other"
    _fetched(lib, cmu=2)
    assert _check().summary == "cmu 2 .bvh, 1 .md; studio_takes 1 .fbx, 1 other"


def test_dog_clips_without_their_index_are_flagged(tmp_path, monkeypatch):
    lib = _home(tmp_path, monkeypatch)
    _fetched(lib, dogs=2, index=False)
    result = _check()
    assert result.status == E.WARN and "index.txt is missing" in result.summary
    assert result.fix == ["python -m dycraft.install mocap --dog"]


def test_the_report_agrees_with_what_agents_are_shown(tmp_path, monkeypatch):
    """The check says a library is there exactly when the runtime switches mocap on for agents."""
    lib = _home(tmp_path, monkeypatch)
    layouts = [lambda: None,
               lambda: (lib / "empty").mkdir(parents=True),
               lambda: _write(lib / "user_lib" / "walk.c3d"),
               lambda: _fetched(lib, cmu=1)]
    for make in layouts:
        make()
        assert (_check().status == E.UNAVAILABLE) == (runtime.mocap_library() is None)
