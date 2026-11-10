"""Environment problems a cluster install meets, pinned here.

(The lighting one is covered in `test_animation_blender.py`.)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from dycraft.cli import backends as B
from dycraft.cli import cli as C

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")


@pytest.fixture
def scratch():
    base = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
    root = Path(tempfile.mkdtemp(prefix="envfix_", dir=base))
    yield root
    import shutil
    shutil.rmtree(root, ignore_errors=True)


# -- 1. the collision interpreter is handed to the sandbox resolved ----------

def test_sandbox_gets_the_resolved_collision_interpreter(scratch, monkeypatch):
    """A cluster home directory can be a symlinked alias of another mount; the
    sandbox binds the resolved env, so an unresolved path named nothing inside
    it.

    Only the *directory* is resolved: the file name is kept, because a venv's
    `bin/python` is a symlink and resolving it would exec the base interpreter
    without the venv (`tests/test_sandbox_python_layouts.py`)."""
    from dycraft.animation.blender import runtime
    (scratch / "alias").symlink_to(Path(sys.executable).resolve().parents[1])
    monkeypatch.setenv("DYCRAFT_COAL_PYTHON", str(scratch / "alias" / "bin" / "python"))
    for d in ("ws", "out"):
        (scratch / d).mkdir()
    (scratch / "job.json").write_text("{}")
    cmd = runtime.command(scratch / "ws", scratch / "out", scratch / "job.json")
    given = cmd[cmd.index("DYCRAFT_COAL_PYTHON") + 1]
    assert "alias" not in given
    assert Path(given).parent == Path(sys.executable).resolve().parent
    assert Path(given).resolve() == Path(sys.executable).resolve()
    binds = [cmd[i + 2] for i, a in enumerate(cmd) if a in ("--ro-bind", "--bind")]
    assert any(given.startswith(b.rstrip("/") + "/") for b in binds)


# -- 3/4. the agent's tool shells ---------------------------------------------

def test_agent_env_puts_the_run_root_on_pythonpath_first(scratch):
    env = B.agent_shell_env({"PYTHONPATH": "/pkgroot", "PATH": "/usr/bin"}, scratch)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(scratch.resolve()), "/pkgroot"]
    again = B.agent_shell_env(dict(env), scratch)
    assert again["PYTHONPATH"] == env["PYTHONPATH"], "not idempotent"


def test_a_subdirectory_helper_can_import_harness(scratch):
    (scratch / "harness.py").write_text("X = 7\n")
    (scratch / "animation_src").mkdir()
    (scratch / "animation_src" / "helper.py").write_text("import harness; print(harness.X)\n")
    env = B.agent_shell_env(dict(os.environ), scratch)
    out = subprocess.run([sys.executable, "animation_src/helper.py"], cwd=scratch,
                         env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "7", out.stderr


def test_conda_base_auto_activation_is_off_but_an_explicit_choice_wins(scratch):
    env = B.agent_shell_env({"PATH": "/usr/bin"}, scratch)
    assert env["CONDA_AUTO_ACTIVATE_BASE"] == "false" and env["CONDA_AUTO_ACTIVATE"] == "false"
    kept = B.agent_shell_env({"PATH": "/usr/bin", "CONDA_AUTO_ACTIVATE_BASE": "true"}, scratch)
    assert kept["CONDA_AUTO_ACTIVATE_BASE"] == "true"


def test_plain_python_is_the_harness_interpreter_and_nothing_else_is_shadowed(scratch, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch / "cache"))
    env = B.agent_shell_env({"PATH": "/usr/bin:/bin", "DYCRAFT_PYTHON": sys.executable}, scratch)
    shim = Path(env["PATH"].split(os.pathsep)[0])
    assert sorted(p.name for p in shim.iterdir()) == ["python", "python3"]
    assert all((shim / n).resolve() == Path(sys.executable).resolve() for n in ("python", "python3"))
    out = subprocess.run(["bash", "-c", "command -v python"], env=env, capture_output=True, text=True)
    assert Path(out.stdout.strip()).resolve() == Path(sys.executable).resolve()
    twice = B.agent_shell_env(dict(env), scratch)
    assert twice["PATH"].split(os.pathsep).count(str(shim)) == 1


def test_no_interpreter_leaves_path_alone(scratch):
    env = B.agent_shell_env({"PATH": "/usr/bin"}, scratch)
    assert env["PATH"] == "/usr/bin"


# -- 5. the harness daemon is never the launcher's child -----------------------

def _state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return "gone"


def test_a_detached_process_is_reaped_rather_than_left_a_zombie(scratch):
    log = open(scratch / "log", "w")
    pid = C._spawn_detached([sys.executable, "-c", "import time; time.sleep(0.5)"],
                            cwd=str(scratch), log_fh=log, env=dict(os.environ))
    assert pid > 0 and pid != os.getpid()
    assert C._pid_running(pid)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _state(pid) != "gone":
        time.sleep(0.1)
    assert _state(pid) == "gone", "left behind as a zombie"
    assert not C._pid_running(pid)


def test_pid_running_says_no_for_a_zombie(scratch):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _state(child.pid) != "Z":
        time.sleep(0.05)
    assert _state(child.pid) == "Z"
    assert not C._pid_running(child.pid)       # kill -0 alone would say yes
    child.wait()


def test_start_harness_daemon_reports_an_early_exit(scratch, monkeypatch, capsys):
    fake = scratch / "pkg"
    (fake / "fakedaemon").mkdir(parents=True)
    (fake / "fakedaemon" / "__init__.py").write_text("")
    (fake / "fakedaemon" / "__main__.py").write_text("import sys; sys.exit(3)\n")
    run = scratch / "run"
    run.mkdir()
    env = dict(os.environ, PYTHONPATH=f"{fake}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    reported = []
    monkeypatch.setattr(C, "record_launch_failure", lambda run_dir, **kw: reported.append((run_dir, kw)))
    with pytest.raises(SystemExit) as exited:
        C.start_harness_daemon(run, env, daemon_module="fakedaemon")
    assert exited.value.code == 1
    assert [(d, kw["reason"]) for d, kw in reported] == [(run, "harness_daemon_exited_early")]
    assert "fakedaemon" in reported[0][1]["detail"]
    said = f"FATAL: DyCraft's daemon stopped before it was ready; see {run / '.daemon.log'}"
    assert said in capsys.readouterr().err
    assert reported[0][1]["message"] == said, "a refused run's README says it in the same words"
