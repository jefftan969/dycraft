"""How a job gets a frozen view of the project without paying for a copy.

A full `copytree` of the workspace before every op costs seconds on a large
project, and the copies are retained, so over a run they grow to tens of
gigabytes. Most of that buys nothing: most of the tree is agent scratch, and
the ops paying for it never write a byte of it.

Two mechanisms avoid it. `reads_only` marks the worker ops that only write under
/output so the sandbox can bind /workspace read-only, and non-mutating session
ops take their snapshot with hardlinks instead of bytes. These tests pin the
first and the property the second depends on.
"""
from __future__ import annotations

import os
import shutil

from dycraft.animation.blender.runtime import command, reads_only


def test_only_the_ops_that_never_write_the_workspace_are_read_only():
    """`worker.py` saves `animation.blend` back for exactly these three."""
    for op in ("render", "inspect", "evaluate"):
        assert reads_only({"op": op}), op
    for op in ("initialize", "edit", "update"):
        assert not reads_only({"op": op}), f"{op} saves the workspace"


def test_probe_is_not_read_only_because_it_runs_agent_code():
    """`eval` of an agent expression may legitimately write; failing that
    loudly would be a behavior change, not an enforcement of one."""
    assert not reads_only({"op": "probe"})
    assert not reads_only({})


def test_the_sandbox_binds_the_workspace_read_only_when_asked(tmp_path):
    job = tmp_path / "job.json"
    job.write_text("{}")
    ws, out = tmp_path / "ws", tmp_path / "out"
    ws.mkdir(), out.mkdir()

    def bind_flag(readonly):
        argv = command(ws, out, job, workspace_readonly=readonly)
        return argv[argv.index(str(ws)) - 1]

    assert bind_flag(True) == "--ro-bind"
    assert bind_flag(False) == "--bind"


def test_a_hardlinked_snapshot_survives_a_publish(tmp_path):
    """The property the capture change rests on.

    A publishing op swaps directories by rename and deletes the old tree. A
    hardlink into that tree keeps the content alive and unchanged, so an
    earlier snapshot still reads what it captured -- no copy required.
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "animation.blend").write_bytes(b"VERSION-1")

    capture = tmp_path / "capture"
    shutil.copytree(work, capture, copy_function=os.link)
    assert (capture / "animation.blend").stat().st_ino == (work / "animation.blend").stat().st_ino

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "animation.blend").write_bytes(b"VERSION-2")
    previous = tmp_path / "previous"
    work.rename(previous)
    candidate.rename(work)
    shutil.rmtree(previous)

    assert (work / "animation.blend").read_bytes() == b"VERSION-2"
    assert (capture / "animation.blend").read_bytes() == b"VERSION-1"


def test_mutating_ops_still_take_a_real_copy():
    """The guard, read off the source: a hardlinked capture for an op that
    writes the workspace would write straight through into the live project.

    `storyboard` and `accept_storyboard` are the deliberate exceptions and are
    asserted here as such, rather than left to widen quietly. They publish, but
    the only thing either writes into the workspace is `storyboards.json` via a
    sibling `.tmp` and `replace()`, which breaks the link with a fresh inode
    instead of writing through it, and both publish by merging boards into the
    live project rather than renaming the capture in. Both halves are
    pinned, so adding another mutating op to the exception fails this test.

    `probe` is linked too: off the live project its
    workspace is bound read-only (`processes.py::_dispatch`), so agent code
    cannot write through a link.
    """
    from dycraft.animation.blender.runtime import REPO
    source = (REPO / "animation/session.py").read_text()
    assert ('link = not mutates or operation in ("storyboard", "accept_storyboard")') in source
    assert '**({"copy_function": os.link} if link else {})' in source
    assert 'or request.get("op") == "probe"' in (REPO / "animation/processes.py").read_text()
