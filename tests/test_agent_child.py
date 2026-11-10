"""The launcher runs the agent CLI as its child and exits with the run's status.

Replacing itself with the CLI (`execvpe`) would take its SIGTERM handler along,
so a committed run that stopped itself through the watcher would exit 143, as
if killed. These tests drive a real launcher process with a stand-in CLI.
"""
import os
import signal
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="the launcher's process handling is Linux's")

#: A launcher: the self-kill watcher and exit translation `cli.main` installs,
#: then the stand-in CLI through `run_agent_cli`.
LAUNCHER = """
import os, sys
from pathlib import Path
from dycraft.cli.cli import install_self_kill_exit, start_self_kill_watcher
from dycraft.cli.backends import run_agent_cli
run_dir = Path(sys.argv[1])
start_self_kill_watcher(run_dir, os.getpid())
install_self_kill_exit(run_dir)
sys.exit(run_agent_cli([sys.executable, "-c", sys.argv[2], str(run_dir / "cli.pid")],
                       dict(os.environ)))
"""

#: The stand-in agent CLI: records its pid, then waits to be stopped, as the
#: real ones do once the agent has finished.
WAITS = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"


def _launch(run_dir, cli=WAITS):
    # The session's PYTHONPATH already leads with this checkout (`conftest._export_child_pkgroot`).
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, "-c", LAUNCHER, str(run_dir), cli],
                            env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pid_file = run_dir / "cli.pid"
    deadline = time.monotonic() + 30
    while not (pid_file.is_file() and pid_file.read_text()):
        assert proc.poll() is None, proc.communicate()[0]
        assert time.monotonic() < deadline, "the stand-in CLI never started"
        time.sleep(0.05)
    return proc, int(pid_file.read_text())


def _gone(pid, within=10.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_committed_run_that_stops_itself_exits_0(tmp_path):
    proc, cli = _launch(tmp_path)
    (tmp_path / "ir.json").write_text("{}")
    (tmp_path / ".terminate").touch()
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert "committed, self-terminated -> stopping the agent CLI, then exit 0" in out
    assert _gone(cli)


def test_a_run_killed_before_it_committed_still_exits_143(tmp_path):
    proc, cli = _launch(tmp_path)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143, out
    assert _gone(cli)


def test_a_cli_that_exits_on_its_own_sets_the_status(tmp_path):
    exits = "import os, sys; open(sys.argv[1], 'w').write(str(os.getpid())); sys.exit(3)"
    proc, _ = _launch(tmp_path, exits)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 3, out


def test_the_cli_does_not_outlive_a_launcher_killed_outright(tmp_path):
    proc, cli = _launch(tmp_path)
    proc.kill()
    proc.communicate(timeout=30)
    assert _gone(cli), "the agent CLI outlived its launcher"
