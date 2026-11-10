"""The self-kill SIGTERM must not report a committed run as a failure.

A batch launcher propagates the run's exit status so that
`backends.EXIT_CODEX_FATAL` can reach it. The self-kill watcher SIGTERMs the
CLI as the *normal* end of a successful run, so without this a committed run
would exit 143 and be recorded as failed. The sentinel plus the committed
artifact are the discriminator, and only the CLI process can see them.
"""
from __future__ import annotations

import signal

from dycraft.cli.cli import self_kill_exit_code

TERM = 128 + int(signal.SIGTERM)


def test_committed_self_termination_is_success(tmp_path):
    (tmp_path / ".terminate").touch()
    (tmp_path / "ir.json").write_text('{"id": "abc"}')
    assert self_kill_exit_code(tmp_path) == 0


def test_external_kill_of_a_committed_run_is_still_a_failure(tmp_path):
    # A batch launcher's stall/timeout ceiling kills a run that had already
    # committed an earlier IR. No sentinel: it did not choose to stop.
    (tmp_path / "ir.json").write_text('{"id": "abc"}')
    assert self_kill_exit_code(tmp_path) == TERM


def test_self_termination_without_a_commit_is_a_failure(tmp_path):
    # The agent dropped the sentinel but never committed. Reporting 0 here
    # would let `already_done` skip a scene that produced nothing.
    (tmp_path / ".terminate").touch()
    assert self_kill_exit_code(tmp_path) == TERM


def test_plain_sigterm_is_a_failure(tmp_path):
    assert self_kill_exit_code(tmp_path) == TERM


def test_a_stale_sentinel_does_not_kill_the_next_session(tmp_path):
    """A `.terminate` an earlier session left in the run dir would stop a
    relaunched sub-run within a second of its start. The watcher clears it at
    start, and a fresh one still stops the run."""
    import subprocess
    import sys
    import time

    from dycraft.cli.cli import start_self_kill_watcher

    (tmp_path / ".terminate").touch()
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start_self_kill_watcher(tmp_path, target.pid, log_label="test")
        assert not (tmp_path / ".terminate").exists()
        time.sleep(2.0)
        assert target.poll() is None, "the stale sentinel stopped the new session"
        (tmp_path / ".terminate").touch()
        deadline = time.monotonic() + 5
        while target.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert target.returncode == -signal.SIGTERM
    finally:
        if target.poll() is None:
            target.kill()
        target.wait()


def test_a_plain_run_that_submitted_its_answer_and_stopped_is_a_success(tmp_path):
    """A plain run's commit is its answer, `result.json`; it writes no `ir.json`."""
    (tmp_path / ".terminate").touch()
    (tmp_path / "result.json").write_text('{"text": "ready", "submitted_at": "x"}')
    assert self_kill_exit_code(tmp_path) == 0
