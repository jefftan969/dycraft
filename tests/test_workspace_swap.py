"""Publishing a project must not be decided by removing the tree it replaced.

On NFS, a file another process still holds open is renamed to `.nfsXXXX` in
place, so `rmtree` of the old tree fails with ENOTEMPTY after the new project
is already live. That must not report a published operation as failed, and a
leftover under the fixed name `previous` must not break the next rename onto
it. Simulated here with a directory that cannot be emptied.
"""
import os
import stat

import pytest

from dycraft.animation.jobs import swap_workspace


def _tree(path, text):
    (path / "animation_src").mkdir(parents=True)
    (path / "animation_src" / "author.py").write_text(text)
    return path


def _pin(directory):
    """Make `directory` impossible to empty, as an open file on NFS does."""
    (directory / ".nfs0000deadbeef").write_text("held open elsewhere")
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_swap_publishes_even_when_the_old_tree_cannot_be_removed(tmp_path):
    native = tmp_path / "native"
    work = _tree(native / "work", "v1")
    _pin(work / "animation_src")
    pinned = []
    try:
        swap_workspace(work, _tree(tmp_path / "cand2", "v2"), native)
        assert (work / "animation_src" / "author.py").read_text() == "v2"
        pinned = [p / "animation_src" for p in native.glob("previous*")]
        assert pinned, "the unremovable old tree should be left aside, not raise"

        # The next publish is not blocked by what the last one left behind.
        swap_workspace(work, _tree(tmp_path / "cand3", "v3"), native)
        assert (work / "animation_src" / "author.py").read_text() == "v3"
    finally:
        for p in pinned:
            p.chmod(stat.S_IRWXU)


def test_a_failed_swap_puts_the_project_back(tmp_path):
    native = tmp_path / "native"
    work = _tree(native / "work", "v1")
    with pytest.raises(OSError):
        swap_workspace(work, tmp_path / "missing", native)
    assert (work / "animation_src" / "author.py").read_text() == "v1"
    assert not list(native.glob("previous*"))
