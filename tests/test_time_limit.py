"""`--max-hours`: the launcher stops a run at its deadline, records it, and its sub-agents inherit the deadline.

The stop reuses the SIGTERM path (`backends.stop_agent_cli`), so both ways the agent CLI runs are covered: as the
launcher's child (`run_agent_cli`) and on the PTY the Codex shim drives.
"""
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from dycraft.cli import backends as B
from dycraft.cli import cli as C


@pytest.fixture(autouse=True)
def no_inherited_deadline(monkeypatch):
    monkeypatch.delenv(C.DEADLINE_ENV, raising=False)


def test_the_limit_counts_from_now_and_zero_means_none():
    deadline, record = C.run_deadline(2.0, now=1000.0)
    assert deadline == 1000.0 + 7200.0
    assert record["hours"] == 2.0 and record["reached"] is False and "inherited" not in record
    assert C.run_deadline(0, now=1000.0) == (None, {})
    assert C.run_deadline(None, now=1000.0)[0] == 1000.0 + C.DEFAULT_MAX_HOURS * 3600


def test_a_sub_agent_follows_its_parents_deadline(monkeypatch):
    monkeypatch.setenv(C.DEADLINE_ENV, "1500.0")
    deadline, record = C.run_deadline(None, now=1000.0)          # launched with no --max-hours
    assert deadline == 1500.0 and record["inherited"] is True
    assert C.run_deadline(2.0, now=1000.0)[0] == 1500.0          # an explicit limit cannot outlive it
    assert C.run_deadline(0.1, now=1000.0)[0] == 1360.0          # but can be shorter
    monkeypatch.setenv(C.DEADLINE_ENV, "inf")                    # the parent has no limit
    assert C.run_deadline(None, now=1000.0) == (None, {})


class _Backend:
    """Runs `cmd` the way a CLI backend runs the agent."""

    def __init__(self, cmd, pty=False):
        self.cmd, self.pty = cmd, pty

    def execute(self, plan):
        if self.pty:
            return B._process._spawn_with_pty_term_shim(self.cmd, dict(os.environ))
        return B.run_agent_cli(self.cmd, dict(os.environ))


def _plan(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"outcome": "interrupted",
                                                    "time_limit": {"hours": 1, "reached": False}}))
    return SimpleNamespace(run_dir=tmp_path)


def _swallow_stdout(fn):
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    try:
        return fn()
    finally:
        os.dup2(saved, 1)
        for fd in (w, saved, r):
            os.close(fd)


@pytest.mark.parametrize("pty", [False, True], ids=["child", "pty"])
def test_the_agent_is_stopped_at_the_deadline_and_it_is_recorded(tmp_path, monkeypatch, pty):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    backend = _Backend([sys.executable, "-c", "import time; time.sleep(60)"], pty=pty)
    plan = _plan(tmp_path)
    started = time.time()
    status = _swallow_stdout(lambda: C.execute_with_time_limit(backend, plan, time.time() + 1.5))
    assert status == C.EXIT_TIME_LIMIT
    assert time.time() - started < 20
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["time_limit"]["reached"] is True and meta["time_limit"]["stopped_at"]
    assert meta["outcome"] == "interrupted"                     # the daemon's record is kept


def test_the_limit_is_recorded_even_when_the_cli_exits_before_the_stopping_thread_records_it(
        tmp_path, monkeypatch):
    """Under load the thread that stops the CLI can be descheduled right after signaling it; the launcher must
    still return with `reached` recorded, not leave it to a thread that may never run again."""
    real = B.stop_agent_cli

    def preempted(status):
        stopping = real(status)
        time.sleep(1.0)
        return stopping
    monkeypatch.setattr(B, "stop_agent_cli", preempted)
    backend = _Backend([sys.executable, "-c", "import time; time.sleep(60)"])
    plan = _plan(tmp_path)
    assert C.execute_with_time_limit(backend, plan, time.time() + 1.0) == C.EXIT_TIME_LIMIT
    assert json.loads((tmp_path / "meta.json").read_text())["time_limit"]["reached"] is True


def test_a_run_that_ends_first_keeps_its_own_status(tmp_path):
    backend = _Backend([sys.executable, "-c", "raise SystemExit(3)"])
    plan = _plan(tmp_path)
    assert C.execute_with_time_limit(backend, plan, time.time() + 30) == 3
    assert json.loads((tmp_path / "meta.json").read_text())["time_limit"]["reached"] is False


def test_no_deadline_is_a_plain_execute(tmp_path):
    calls = []
    backend = SimpleNamespace(execute=lambda plan: calls.append(plan) or 0)
    assert C.execute_with_time_limit(backend, _plan(tmp_path), None) == 0
    assert len(calls) == 1


_SELF_KILL_ON_PTY = r"""
import os, sys
from pathlib import Path
from dycraft.cli import backends as B, cli as C
run_dir = Path(sys.argv[1])
C.install_self_kill_exit(run_dir)
open(run_dir / "ready", "w").close()
status = B._process._spawn_with_pty_term_shim([sys.executable, "-c", "import time; time.sleep(120)"], dict(os.environ))
print("status", status, flush=True)
sys.exit(status)
"""


def test_a_real_sigterm_reaches_a_codex_on_the_pty_through_the_self_kill_handler(tmp_path, pkgroot):
    """The launcher's SIGTERM handler forwards the stop to the CLI on the PTY and exits with its decided status
    (143: no commit) once the CLI is gone, instead of exiting at once and leaving the CLI running."""
    import signal
    import subprocess
    env = {**os.environ, "DYCRAFT_CODEX_NO_ROLLOUT_WATCH": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(pkgroot), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.Popen([sys.executable, "-c", _SELF_KILL_ON_PTY, str(tmp_path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if (tmp_path / "ready").exists():
            break
        time.sleep(0.1)
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 143, err[-500:]
    assert "status 143" in out


def test_a_stop_under_way_keeps_its_status(tmp_path, monkeypatch):
    """A committed run that stops itself exits 0; a deadline arriving while it exits must not turn that into 124."""
    monkeypatch.setattr(B._process, "_stop_status", 0)
    assert B.stop_agent_cli(C.EXIT_TIME_LIMIT) is True
    assert B.stop_status() == 0


def test_the_deadline_does_not_overwrite_a_run_that_was_already_stopping(tmp_path, monkeypatch):
    monkeypatch.setattr(B._process, "_stop_status", None)    # restored after the test whatever happens

    class Backend:
        def execute(self, plan):
            B._process._stop_status = 0                       # the run committed and is stopping itself
            time.sleep(2.0)                          # the deadline passes meanwhile
            return 0
    plan = _plan(tmp_path)
    assert C.execute_with_time_limit(Backend(), plan, time.time() + 0.5) == 0
    assert json.loads((tmp_path / "meta.json").read_text())["time_limit"]["reached"] is False


def test_an_animation_agent_is_told_when_the_run_stops():
    """The approved footer sentence, with the hours and the clock time of the deadline; other layers are not told."""
    note = C.time_limit_note(1000.0 + 8 * 3600, "animation", now=1000.0)
    assert "This run is stopped 8 hours after your session started, at " in note
    assert note.rstrip().endswith("without waiting for work in progress. Deliver before then; a run stopped "
                                  "without a delivery keeps its files, and the harness records the best film it "
                                  "rendered.")
    assert C.time_limit_note(None, "animation") == ""
    assert C.time_limit_note(1000.0 + 3600, "object", now=1000.0) == ""


def test_the_stop_time_names_a_zone_even_without_an_abbreviation(monkeypatch):
    monkeypatch.setenv("TZ", "<+0545>-05:45")
    time.tzset()
    try:
        note = C.time_limit_note(1000.0 + 3600, "animation", now=1000.0)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert " ," not in note and ("+0545" in note)


def test_a_negative_limit_is_refused(capsys):
    with pytest.raises(SystemExit) as raised:
        C.main(["--backend", "codex", "--layer", "animation", "--prompt", "x", "--max-hours", "-1"])
    assert raised.value.code == 2
    assert "--max-hours must be 0" in capsys.readouterr().err
