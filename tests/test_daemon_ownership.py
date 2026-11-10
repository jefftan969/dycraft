"""A run directory has exactly one harness daemon.

Everything the daemon writes is single-writer state, so a second daemon
corrupts rather than duplicates: `trace.jsonl` is appended and `meta.json`
rewritten without locking, `session.step` is an in-memory counter (so two
daemons re-use step numbers and overwrite each other's
`compile_reports/<step>.json` and `renders/<step>_*`), and whichever exits
last wins the `daemon_shutdown` outcome and the final `meta.json`.

An agent that wants a Blender render can relaunch the harness from inside its
own sandbox; a second daemon on the same run dir then interleaves its steps with
the first's, and `meta.json` names the wrong renderer for steps that ran on the
other.

The controlled replacement for that relaunch is the `set_render_backend` op,
covered at the bottom.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.agent.server import (
    DaemonAlreadyOwned,
    OWNER_INFO_NAME,
    OWNER_LOCK_NAME,
    acquire_run_dir,
    read_owner_info,
    write_owner_info,
)


def test_the_first_daemon_takes_ownership(tmp_path):
    handle = acquire_run_dir(tmp_path)
    try:
        assert (tmp_path / OWNER_LOCK_NAME).exists()
    finally:
        handle.close()


def test_a_second_daemon_is_refused_while_the_first_holds_it(tmp_path):
    handle = acquire_run_dir(tmp_path)
    try:
        with pytest.raises(DaemonAlreadyOwned):
            acquire_run_dir(tmp_path)
    finally:
        handle.close()


def test_ownership_is_released_when_the_owner_lets_go(tmp_path):
    """A finished run must not lock its directory out of a later relaunch."""
    acquire_run_dir(tmp_path).close()
    second = acquire_run_dir(tmp_path)
    second.close()


_HOLDER_SRC = """
import sys, time
from pathlib import Path
from dycraft.agent.server import acquire_run_dir, write_owner_info
run_dir = Path(sys.argv[1])
handle = acquire_run_dir(run_dir)
write_owner_info(run_dir, port=4242)
print("held", flush=True)
time.sleep(300)
"""


def test_a_dead_owner_does_not_leave_the_run_dir_locked(tmp_path):
    """flock is released by the kernel on process death, so a SIGKILLed
    daemon cannot strand a run dir — there is no stale-pid reaping to get
    wrong, and no window where a legitimate relaunch is refused."""
    # The repo root, **not** its parent: the parent may hold another checkout
    # named `dycraft`, which a cwd there would bind `import dycraft` to. The
    # child gets `dycraft` from the environment, so the cwd only has to not be
    # actively wrong.
    cwd=str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SRC, str(tmp_path)],
        stdout=subprocess.PIPE, text=True, cwd=cwd)
    try:
        assert proc.stdout.readline().strip() == "held"
        with pytest.raises(DaemonAlreadyOwned):
            acquire_run_dir(tmp_path)
        proc.kill()          # no atexit, no explicit close
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
    handle = acquire_run_dir(tmp_path)
    handle.close()


def test_the_refusal_names_the_incumbent(tmp_path):
    """The message has to be actionable: which pid and port to attach to."""
    handle = acquire_run_dir(tmp_path)
    write_owner_info(tmp_path, port=31337)
    try:
        with pytest.raises(DaemonAlreadyOwned) as caught:
            acquire_run_dir(tmp_path)
        assert caught.value.info["port"] == 31337
        assert caught.value.info["pid"] == os.getpid()
    finally:
        handle.close()


def test_owner_info_records_the_render_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "blender")
    handle = acquire_run_dir(tmp_path)
    try:
        write_owner_info(tmp_path, port=1)
        assert read_owner_info(tmp_path)["render_backend"] == "blender"
    finally:
        handle.close()


def test_missing_owner_info_reads_as_empty(tmp_path):
    assert read_owner_info(tmp_path) == {}
    (tmp_path / OWNER_INFO_NAME).write_text("{not json")
    assert read_owner_info(tmp_path) == {}


def test_serve_exits_zero_rather_than_failing_the_run(tmp_path, monkeypatch):
    """A refused second daemon means the desired end state already holds, so
    it must not look like a launch failure — `start_harness_daemon` aborts the
    run on a non-zero daemon exit."""
    from dycraft.agent import server as server_mod

    handle = acquire_run_dir(tmp_path)
    try:
        class _Session:
            handlers = {"compile": lambda **_: {}}

        with pytest.raises(SystemExit) as exit_info:
            server_mod.serve(_Session(), tmp_path)
        assert exit_info.value.code == 0
        # And it must not have clobbered the incumbent's port file.
        assert not (tmp_path / server_mod.PORT_NAME).exists()
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# The controlled replacement for relaunching to change renderer
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.committed = False


class _FakeSession:
    """Just enough of LayerSession to exercise the op body."""

    def __init__(self, run_dir):
        self.ctx = _FakeCtx(run_dir)
        self.traced: list[tuple] = []

    def trace(self, ev, **fields):
        self.traced.append((ev, fields))

    _do_set_render_backend = None  # bound below


@pytest.fixture
def render_session(tmp_path):
    from dycraft.agent.layers.session import LayerSession
    session = _FakeSession(tmp_path)
    session._do_set_render_backend = (
        LayerSession._do_set_render_backend.__get__(session))
    return session


@pytest.mark.parametrize("backend", ["open3d", "blender"])
def test_set_render_backend_switches_the_renderer(render_session, monkeypatch,
                                                  backend):
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    result = render_session._do_set_render_backend(backend)
    assert result["ok"] and result["render_backend"] == backend
    # `render.default_backend()` re-reads the env per request, so this is all a
    # later compile/view/snapshot/final_render needs.
    from dycraft.render import default_backend
    assert default_backend() == backend


def test_set_render_backend_rejects_an_unknown_name(render_session):
    result = render_session._do_set_render_backend("blendr")
    assert result["ok"] is False
    assert "blendr" in result["error"] and "blender" in result["error"]


def test_set_render_backend_is_traced(render_session, monkeypatch):
    """The bundle has to say which renderer produced which artifact."""
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    render_session._do_set_render_backend("blender")
    assert ("render_backend_set",
            {"backend": "blender", "previous": "open3d"}) in render_session.traced


def test_the_op_is_registered_and_forwarded_by_both_stubs():
    """Client half and daemon half must agree, or the call fails at the wire."""
    root = Path(__file__).resolve().parent.parent / "agent" / "layers"
    assert '@self.op("set_render_backend")' in (root / "session.py").read_text()
    src = (root / "_harness_tools.py").read_text()
    assert "def set_render_backend(" in src
    assert '_call("set_render_backend"' in src
    # NOT `_harness_tools_animation.py`. The op picks between the Open3D and
    # Blender renderers for the canonical stills; the animation layer has
    # neither, renders every frame through its own Blender worker at fixed
    # Eevee settings, and reads `DYCRAFT_RENDER_BACKEND` nowhere (grep
    # `animation/`). `animation/session.py::register` therefore
    # registers no such op, and `test_object_prompt_contract.py::
    # test_the_staged_harness_exposes_exactly_the_ops_the_daemon_registered`
    # holds the other side of that contract: a stub op with no handler fails at
    # the wire, which is exactly what this test exists to prevent.
    # The definition and the wire call, not the bare name: that stub carries a
    # comment recording the deliberate absence, and a guard that forbade the
    # word would forbid explaining itself.
    animation = (root / "_harness_tools_animation.py").read_text()
    assert "def set_render_backend(" not in animation
    assert '_call("set_render_backend"' not in animation


# ---------------------------------------------------------------------------
# The launcher attaches instead of racing
# ---------------------------------------------------------------------------

def test_the_launcher_attaches_to_a_live_daemon(tmp_path, monkeypatch, capsys):
    """`start_harness_daemon` must not spawn a second daemon, abort the run,
    or truncate the incumbent's `.daemon.log`.

    Before ownership existed the loop checked `.harness.port` first, so a
    second launcher silently reported "ready" against the *incumbent's* port
    while its own daemon was still starting up behind it.
    """
    from dycraft.cli import cli as cli_mod

    (tmp_path / ".daemon.log").write_text("incumbent log\n")
    handle = acquire_run_dir(tmp_path)
    write_owner_info(tmp_path, port=5555)
    spawned: list = []
    monkeypatch.setattr(cli_mod.subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or None)
    try:
        pid = cli_mod.start_harness_daemon(
            tmp_path, {}, daemon_module="dycraft.agent.layers.server")
    finally:
        handle.close()
    assert pid == os.getpid()
    assert not spawned, "a second daemon was spawned"
    assert (tmp_path / ".daemon.log").read_text() == "incumbent log\n"
    assert "attaching" in capsys.readouterr().err


def test_the_launcher_waits_for_a_daemon_slow_to_bind(tmp_path, monkeypatch):
    """On a loaded host a dispatched child's daemon can take over a minute to
    bind. A live daemon gets minutes; a dead one is still caught by its pid at
    once."""
    from types import SimpleNamespace
    from dycraft.cli import cli as cli_mod

    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 40:
            (tmp_path / ".harness.port").write_text("5555")

    monkeypatch.setattr(cli_mod, "time", SimpleNamespace(time=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(cli_mod, "_spawn_detached", lambda *a, **k: 4242)
    monkeypatch.setattr(cli_mod, "_pid_running", lambda pid: True)
    assert cli_mod.start_harness_daemon(
        tmp_path, {}, daemon_module="dycraft.agent.layers.server") == 4242
    assert 40 <= clock[0] < 41


def test_the_launcher_still_fails_loudly_on_a_real_daemon_crash(tmp_path,
                                                                monkeypatch):
    """The attach path must not swallow a genuine startup failure — a crashed
    daemon has to abort the launch rather than exec the agent into a run with
    no harness."""
    from dycraft.cli import cli as cli_mod

    # The daemon is started detached (`_spawn_detached`), so its exit status
    # is not the launcher's to read: a crash shows as a pid that is gone
    # while no other live daemon owns the run dir.
    monkeypatch.setattr(cli_mod, "_spawn_detached", lambda *a, **k: 4242)
    monkeypatch.setattr(cli_mod, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli_mod, "record_launch_failure",
                        lambda *a, **k: None)
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.start_harness_daemon(
            tmp_path, {}, daemon_module="dycraft.agent.layers.server")
    assert exit_info.value.code == 1
