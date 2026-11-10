"""What counts as a motion-capture library (`runtime.mocap_libraries`), and the gate built on it.

A library is any non-hidden subdirectory of `assets/mocap/` with at least one non-hidden file, in any format. The
gate (`runtime.mocap_library`) decides whether the agent is told about motion capture at all, so an empty or
hidden-only `assets/mocap/` must leave it off, and a user's own library must turn it on like the fetched ones.
"""
import pytest

from dycraft.agent.agent_runner import OPTIONAL
from dycraft.animation.blender import runtime

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")


@pytest.fixture
def mocap(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setenv("DYCRAFT_HOME", str(repo))
    root = repo / "assets" / "mocap"
    root.mkdir(parents=True)
    return root


def _file(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_no_directory_no_library(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path / "empty"))
    assert runtime.mocap_libraries() == {}
    assert runtime.mocap_library() is None
    assert OPTIONAL["mocap"]() is False


def test_an_empty_directory_is_off(mocap):
    """A failed fetch can leave `assets/mocap/` or an empty `cmu/` behind; neither turns mocap on."""
    assert runtime.mocap_library() is None
    (mocap / "cmu" / "data").mkdir(parents=True)
    assert runtime.mocap_libraries() == {}
    assert runtime.mocap_library() is None
    assert OPTIONAL["mocap"]() is False


def test_hidden_entries_do_not_count(mocap):
    _file(mocap / ".DS_Store")
    _file(mocap / ".partial" / "01_01.bvh")
    _file(mocap / "cmu" / ".fetch-lock")
    _file(mocap / "cmu" / ".cache" / "01_01.bvh")
    assert runtime.mocap_libraries() == {}
    assert runtime.mocap_library() is None


def test_a_loose_file_at_the_root_is_not_a_library(mocap):
    _file(mocap / "README.txt")
    assert runtime.mocap_library() is None


def test_a_users_library_in_any_format_counts(mocap):
    _file(mocap / "studio" / "takes" / "jump.fbx")
    assert runtime.mocap_libraries() == {"studio": mocap.resolve() / "studio"}
    assert runtime.mocap_library() == mocap.resolve()
    assert OPTIONAL["mocap"]() is True


def test_the_fetched_layout_counts(mocap):
    _file(mocap / "cmu" / "data" / "08" / "08_01.bvh", "HIERARCHY\n")
    _file(mocap / "mann_dog" / "data" / "D1_000.bvh", "HIERARCHY\n")
    _file(mocap / "mann_dog" / "index.txt")
    (mocap / "empty").mkdir()
    assert sorted(runtime.mocap_libraries()) == ["cmu", "mann_dog"]
    assert runtime.mocap_library() == mocap.resolve()


def test_a_linked_library_counts_and_a_link_cycle_ends(mocap, tmp_path):
    """Libraries may be symlinked in, as `assets/mocap/` itself is in run worktrees."""
    elsewhere = tmp_path / "elsewhere"
    _file(elsewhere / "clip.c3d")
    (mocap / "linked").symlink_to(elsewhere, target_is_directory=True)
    loop = mocap / "loop"
    loop.mkdir()
    (loop / "back").symlink_to(loop, target_is_directory=True)
    assert sorted(runtime.mocap_libraries()) == ["linked"]


def test_the_mount_follows_the_gate(mocap, tmp_path):
    from dycraft.render._blender_env import blender_python
    if blender_python() is None and runtime.blender_binary() is None:
        pytest.skip("needs a configured Blender")
    ws, out = tmp_path / "ws", tmp_path / "out"
    ws.mkdir(), out.mkdir()
    assert runtime.MOCAP_MOUNT not in runtime.command(ws, out, None, serve=True)
    _file(mocap / "cmu" / "data" / "08" / "08_01.bvh", "HIERARCHY\n")
    assert runtime.MOCAP_MOUNT in runtime.command(ws, out, None, serve=True)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.delenv("DYCRAFT_ASSET_LIBRARY", raising=False)
    path = tmp_path / "run"
    path.mkdir()
    return path


@pytest.mark.parametrize("contents", [(), (".partial/01_01.bvh",)])
def test_no_library_no_reference_link(mocap, run, contents):
    """The run's `references/mocap` link follows the same gate as the prompts and the mount."""
    from dycraft.agent import stage_references
    for name in contents:
        _file(mocap / name)
    assert stage_references(run) == []
    assert not (run / "references" / "mocap").exists()


def test_a_library_gets_the_reference_link(mocap, run):
    from dycraft.agent import stage_references
    _file(mocap / "studio" / "jump.fbx")
    assert stage_references(run) == [run / "references" / "mocap"]
    assert (run / "references" / "mocap" / "studio" / "jump.fbx").is_file()


@pytest.mark.parametrize("by_env", [False, True])
def test_the_asset_library_link_does_not_depend_on_mocap(mocap, run, tmp_path, monkeypatch, by_env):
    from dycraft.agent import stage_references
    if by_env:
        imported = tmp_path / "elsewhere_imported"
        monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(imported))
    else:
        imported = mocap.parent / "imported"
    imported.mkdir()
    assert stage_references(run) == [run / "references" / "assets"]
    assert (run / "references" / "assets").resolve() == imported.resolve()
