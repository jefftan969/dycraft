"""Runs interrupted from outside: an agent CLI killed by the OOM killer, and a sub-run whose parent is gone.

Unhandled, an OOM-killed agent would end its run as "interrupted" with nothing to say why, and a rewriter, review or
dispatched-object sub-run whose parent was SIGKILLed would run on to its own time limit for a parent that can no longer
read its result. The launcher names the first, and a sub-run's launcher stops on the second.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dycraft import _oom
from dycraft.cli.backends import _process


def _cgroup(tmp_path: Path, kills: int) -> tuple[Path, Path]:
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice/dycraft-arm.scope\n")
    events = tmp_path / "sys" / "user.slice" / "dycraft-arm.scope" / "memory.events"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(f"low 0\nhigh 0\nmax 3\noom 1\noom_kill {kills}\noom_group_kill 0\n")
    return proc, tmp_path / "sys"


def test_the_oom_kill_count_is_read_from_this_processs_cgroup(tmp_path):
    proc, root = _cgroup(tmp_path, 2)
    assert _oom.oom_kills(proc, root) == 2
    assert _oom.oom_kills(tmp_path / "missing", root) is None


@pytest.mark.parametrize("before, after, reason, says", [
    (0, 1, "oom_kill", "FATAL: the agent CLI was killed by the kernel's OOM killer (this machine ran out of memory)."),
    (1, 1, "sigkill", "FATAL: the agent CLI was killed by SIGKILL from outside the run"),
])
def test_an_agent_killed_by_sigkill_is_named_and_recorded(tmp_path, monkeypatch, capsys, before, after, reason, says):
    monkeypatch.setattr(_oom, "oom_kills", lambda *a: after)
    status = 128 + signal.SIGKILL
    assert _process.report_agent_kill(status, tmp_path, before) == status
    assert capsys.readouterr().err.startswith(says)
    assert json.loads((tmp_path / "meta.json").read_text())["agent_stop"]["reason"] == reason


def test_any_other_exit_is_left_alone(tmp_path, capsys):
    assert _process.report_agent_kill(0, tmp_path, 0) == 0
    assert _process.report_agent_kill(143, tmp_path, 0) == 143
    assert capsys.readouterr().err == "" and not (tmp_path / "meta.json").exists()


_CHILD = textwrap.dedent("""
    import os, sys, time
    from pathlib import Path
    from dycraft.cli.backends import _process
    _process.PARENT_POLL_SECONDS = 0.1
    marker = Path(sys.argv[1])
    watching = _process.watch_parent(lambda pid: (marker.write_text(str(pid)), os._exit(_process.EXIT_PARENT_GONE)))
    marker.with_suffix(".started").write_text(str(watching))
    time.sleep(60)
""")

_PARENT = textwrap.dedent("""
    import os, subprocess, sys, time
    env = {**os.environ, "DYCRAFT_PARENT_PID": str(os.getpid())}
    child = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], env=env)
    print(child.pid, flush=True)
    time.sleep(60)
""")


def test_a_sub_run_stops_once_its_parent_is_gone(tmp_path):
    marker = tmp_path / "gone"
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, _CHILD, str(marker)], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    deadline = time.monotonic() + 30
    while not marker.with_suffix(".started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.with_suffix(".started").read_text() == "True"
    parent.kill()                                       # SIGKILL: the parent cannot tell its child anything
    parent.wait()
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == str(parent.pid)
    for _ in range(100):                                # the child exits on its own
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("the sub-run outlived its parent")


def test_a_sub_run_whose_parent_took_its_output_pipe_with_it_still_stops(tmp_path, monkeypatch):
    """A rewriter's launcher writes to a pipe its parent read; with the parent gone, writing raises."""
    from dycraft.cli import cli as C
    (tmp_path / "meta.json").write_text("{}")

    class Closed:
        def write(self, *_):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")
    monkeypatch.setattr(sys, "stderr", Closed())
    stopped = []
    monkeypatch.setattr(_process, "stop_agent_cli", lambda status: stopped.append(status) or True)
    C._parent_gone(4242, tmp_path)
    assert stopped == [_process.EXIT_PARENT_GONE]
    assert json.loads((tmp_path / "meta.json").read_text())["agent_stop"]["reason"] == "parent_gone"


def test_a_launch_with_no_parent_to_watch_does_not_watch(monkeypatch):
    monkeypatch.delenv(_process.PARENT_PID_ENV, raising=False)
    assert _process.watch_parent(lambda pid: None) is False
    monkeypatch.setenv(_process.PARENT_PID_ENV, "1")    # set, but not this process's parent: inherited from above
    assert _process.watch_parent(lambda pid: None) is False


def test_both_sub_run_spawners_tell_the_child_its_parent(tmp_path, monkeypatch):
    seen = []

    class _Popen:
        def __init__(self, argv, **kw):
            seen.append(kw.get("env") or {})
            self.returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    from dycraft.agent import oneoff
    from dycraft.agent.layers import _object_dispatch
    monkeypatch.setattr(oneoff.subprocess, "Popen", _Popen)
    monkeypatch.setattr(_object_dispatch.subprocess, "Popen", _Popen)
    oneoff.run_oneoff(prompt="hi", backend="claude", run_dir=tmp_path)
    (tmp_path / "child").mkdir()
    _object_dispatch._default_spawn(["true"], tmp_path / "child")
    assert [env.get(_process.PARENT_PID_ENV) for env in seen] == [str(os.getpid())] * 2


def test_a_claude_killed_before_it_answered_is_left_to_the_kill_report(tmp_path, monkeypatch, capsys):
    """Otherwise the session watch would call an OOM kill at startup an exit before any answer, and hide it."""
    from dycraft.cli.backends import claude as C
    monkeypatch.delenv(C.NO_SESSION_WATCH_ENV, raising=False)
    watch = C.ClaudeSessionWatch(tmp_path / "account", tmp_path)
    assert watch.after_exit(_process.SIGKILLED) is None
    assert watch.after_exit(1) == _process.EXIT_AGENT_FATAL          # an ordinary early exit is still named
    assert capsys.readouterr().err.count("FATAL") == 1


def test_a_codex_killed_before_it_answered_keeps_its_status_for_the_kill_report(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    (tmp_path / "run").mkdir()
    cli = tmp_path / "fakecodex"
    cli.write_text("#!/usr/bin/env python3\nimport os, signal, sys\nsys.stdout.write('starting'); sys.stdout.flush()\n"
                   "os.kill(os.getpid(), signal.SIGKILL)\n")
    cli.chmod(0o755)
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    try:
        status = _process._spawn_with_pty_term_shim([str(cli), "-C", str(tmp_path / "run")], dict(os.environ))
    finally:
        os.dup2(saved, 1)
        for fd in (w, saved, r):
            os.close(fd)
    assert status == _process.SIGKILLED
    assert "before it answered" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A stop reaches what the agent CLI started
# ---------------------------------------------------------------------------

def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _cli_tree(tmp_path, *, own_group: bool):
    """A stand-in CLI that starts a command in a session of its own and ignores SIGTERM, as Codex's
    `codex-linux-sandbox` → bwrap → command does; returns the CLI and the command's pid."""
    pidfile = tmp_path / "inner.pid"
    script = (f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 60' & "
              "trap '' TERM; while :; do sleep 0.1; done")
    cli = subprocess.Popen(["sh", "-c", script], start_new_session=own_group)
    deadline = time.monotonic() + 5
    while not (pidfile.is_file() and pidfile.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(0.05)
    return cli, int(pidfile.read_text())


@pytest.mark.parametrize("own_group", [True, False], ids=["its own process group", "the launcher's"])
def test_a_stop_signals_the_cli_and_the_commands_it_started(tmp_path, own_group):
    import signal
    cli, inner = _cli_tree(tmp_path, own_group=own_group)
    try:
        assert inner in _process._descendants(cli.pid)
        # A command in a session of its own is not in the CLI's group; here it is reached as a descendant, and under
        # Codex bwrap's --die-with-parent takes it once its parent goes.
        if not own_group:
            _process._signal_cli(cli.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while _alive(inner) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _alive(inner)
        assert cli.poll() is None                  # it ignores SIGTERM, so the escalation is what stops it
        _process._signal_cli(cli.pid, signal.SIGKILL)
        assert cli.wait(timeout=5) == -signal.SIGKILL
    finally:
        for pid in (cli.pid, inner):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        cli.wait()


def test_a_cli_with_its_own_group_is_signalled_as_a_group(tmp_path, monkeypatch):
    import signal
    sent = []
    monkeypatch.setattr(_process.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(_process.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(_process.os, "kill", lambda pid, sig: pytest.fail("signalled one process"))
    _process._signal_cli(4242, signal.SIGTERM)
    assert sent == [(4242, signal.SIGTERM)]
