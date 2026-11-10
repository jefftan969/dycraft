"""The Blender worker's sandbox where bubblewrap cannot make namespaces: Landlock with host paths.

`runtime.command` describes the sandbox as a bubblewrap argv. Where user namespaces cannot be made without sudo
(stock Ubuntu 24.04 without the system's bubblewrap), `runtime.sandbox_mode` picks Landlock, and `_host_command`
turns that argv into the `_landlock.py` launcher's spec: the same reads, writes and devices, no network, and the
host's paths in place of `/workspace`, `/tools`, `/output` and `/job.json`. These run real programs in it on this
machine, forced with DYCRAFT_BLENDER_SANDBOX.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.animation.blender import runtime


@pytest.fixture
def dirs(tmp_path):
    for d in ("ws", "out"):
        (tmp_path / d).mkdir()
    (tmp_path / "job.json").write_text('{"op": "test"}')
    return tmp_path


@pytest.fixture
def landlock(monkeypatch):
    if runtime.landlock_abi() < 1:
        pytest.skip("this kernel has no Landlock")
    monkeypatch.setenv(runtime.SANDBOX_ENV, "landlock")
    monkeypatch.setattr(runtime, "blender_binary", lambda: "/usr/bin/true")
    for name in runtime.ALLOW_UNSANDBOXED_ENVS:
        monkeypatch.delenv(name, raising=False)


PROBE = r'''
import json, os, socket, sys
res = {"cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH"), "home": os.environ.get("HOME"),
       "tmpdir": os.environ.get("TMPDIR")}
def t(name, fn):
    try:
        res[name] = fn() or "ok"
    except OSError as e:
        res[name] = f"refused: {e.strerror}"
t("job", lambda: json.load(open(os.environ["DYCRAFT_BLENDER_JOB"]))["op"])
t("tools", lambda: os.path.isfile(os.path.join(os.environ["PYTHONPATH"], "worker.py")) and "worker.py")
t("write output", lambda: open(os.path.join(os.environ["DYCRAFT_BLENDER_OUTPUT_ROOT"], "x"), "w").write("y") and None)
t("write workspace", lambda: open("made_here", "w").write("y") and None)
t("write tmp", lambda: open(os.path.join(os.environ["TMPDIR"], "t"), "w").write("y") and None)
t("write home", lambda: open(sys.argv[1], "w").write("y") and None)
t("read home", lambda: os.listdir(os.path.dirname(sys.argv[1])) and None)
def udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.close()
t("udp socket", udp)
print("PROBE", json.dumps(res))
'''


def _probe(cmd, leak: Path):
    ran = subprocess.run(runtime.with_program(cmd, [sys.executable, "-I", "-c", PROBE, str(leak)]),
                         env=runtime.worker_env(), capture_output=True, text=True, timeout=120)
    line = next((l for l in ran.stdout.splitlines() if l.startswith("PROBE ")), None)
    assert line, ran.stdout + ran.stderr
    return json.loads(line[len("PROBE "):])


def test_a_worker_sees_host_paths_and_reaches_only_what_it_is_bound(dirs, landlock, tmp_path):
    cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    assert str(runtime.LANDLOCK_LAUNCHER) in cmd and "bwrap" not in " ".join(cmd[:-1])
    home = Path.home() / ".cache" / f"dycraft_landlock_test_{os.getpid()}"
    home.mkdir(parents=True, exist_ok=True)
    try:
        got = _probe(cmd, home / "leak")
        assert not (home / "leak").exists()
    finally:
        for p in home.iterdir():
            p.unlink()
        home.rmdir()
    assert Path(got["cwd"]) == (dirs / "ws").resolve() or got["cwd"] == str(dirs / "ws")
    assert got["job"] == "test"
    assert got["tools"] == "worker.py" and got["pythonpath"].startswith(str(runtime._host_root() / "tools"))
    assert got["write output"] == "ok" and (dirs / "out" / "x").read_text() == "y"
    assert got["write workspace"] == "ok" and (dirs / "ws" / "made_here").is_file()
    # `/tmp` (and HOME, which is `/tmp`) is a directory of the worker's own.
    assert got["write tmp"] == "ok" and got["home"] == got["tmpdir"]
    assert got["tmpdir"].startswith(str(runtime._host_root() / "tmp") + "/")
    assert got["write home"].startswith("refused") and got["read home"].startswith("refused")
    assert got["udp socket"].startswith("refused")


def test_a_read_only_workspace_stays_read_only(dirs, landlock):
    cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json", workspace_readonly=True)
    got = _probe(cmd, dirs / "ws" / "leak")
    assert got["write workspace"].startswith("refused") and got["write output"] == "ok"


def test_the_mode_prefers_namespaces_then_landlock_and_unconfined_needs_the_opt_in(monkeypatch, apparmor):
    apparmor(restricted=True, bwrap=False, profiled=True)
    monkeypatch.delenv(runtime.SANDBOX_ENV, raising=False)
    for name in runtime.ALLOW_UNSANDBOXED_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "_namespaces_problem", lambda: "")
    assert runtime.sandbox_mode() == "namespaces"
    monkeypatch.setattr(runtime, "_namespaces_problem", lambda: "bwrap: setting up uid map: Permission denied")
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 6)
    assert runtime.sandbox_mode() == "landlock"
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 0)
    with pytest.raises(RuntimeError) as refused:
        runtime.sandbox_mode()
    assert "sudo apt-get install bubblewrap" in str(refused.value) and "--allow-unsandboxed" in str(refused.value)
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    assert runtime.sandbox_mode() == "unconfined"
    # The opt-in never replaces a sandbox that works.
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 4)
    assert runtime.sandbox_mode() == "landlock"


# --------------------------------------------------------------------------- what install check and a launch say

def _userns(monkeypatch, *, bwrap_rc, abi):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setattr(runtime, "_userns_prefix", lambda unshare: [])
    monkeypatch.setattr(E, "run", lambda argv, **k: E.Ran(bwrap_rc, "", "bwrap: setting up uid map: Permission denied"))
    monkeypatch.setattr(runtime, "landlock_abi", lambda: abi)
    return E, E.check_userns(SimpleNamespace(facts={"bwrap": "/opt/dycraft/bin/bwrap"}))


def test_the_check_says_blender_runs_under_landlock_where_namespaces_fail(monkeypatch):
    E, result = _userns(monkeypatch, bwrap_rc=1, abi=4)
    assert result.status == E.WARN
    assert result.summary == ("bubblewrap cannot make them here, so the Blender worker runs confined by Landlock "
                              "(ABI 4) with this machine's paths")
    assert any("does not stop a worker signalling your other processes" in line for line in result.detail)
    E, result = _userns(monkeypatch, bwrap_rc=1, abi=6)
    assert not any("signalling" in line for line in result.detail)


def test_without_landlock_blender_refuses_unless_the_opt_in_lets_it_run_unconfined(monkeypatch):
    for name in runtime.ALLOW_UNSANDBOXED_ENVS:
        monkeypatch.delenv(name, raising=False)
    E, result = _userns(monkeypatch, bwrap_rc=1, abi=0)
    assert result.status == E.FAIL and result.data == {"sandbox": True}
    assert result.fix[-1].startswith("without sudo: `--allow-unsandboxed` runs Blender without a sandbox")
    check = next(c for c in E.CHECKS if c.id == "sandbox.userns")
    assert E.launch_refusals([(check, result)], "claude") == [(check, result)]
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    E, result = _userns(monkeypatch, bwrap_rc=1, abi=0)
    assert result.summary.startswith("running Blender without a sandbox (--allow-unsandboxed): neither bubblewrap "
                                     "nor Landlock can confine it here")
    assert E.launch_refusals([(check, result)], "claude") == []


def test_a_missing_bubblewrap_is_not_a_failure_where_landlock_takes_its_place(monkeypatch):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setattr(E, "which", lambda name: None)
    monkeypatch.setattr(E.sys, "executable", "/nonexistent/bin/python")
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 6)
    result = E.check_bwrap(SimpleNamespace(facts={}))
    assert result.status == E.INFO and "runs confined by Landlock (ABI 6) instead" in result.summary
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 0)
    result = E.check_bwrap(SimpleNamespace(facts={}))
    assert result.status == E.FAIL and result.data == {"sandbox": True}


def test_a_run_whose_blender_ran_unconfined_says_so_in_its_readme(tmp_path):
    from dycraft import run_readme
    inner = tmp_path / "r1" / "src"
    inner.mkdir(parents=True)
    meta = {"layer": "animation", "runtime": "claude", "prompt": "A dog sits.",
            "blender_sandbox": {"mode": "unconfined", "reason": "bwrap: setting up uid map: Permission denied"}}
    (inner / "meta.json").write_text(json.dumps(meta))
    assert ("> **Blender ran without a sandbox** (`--allow-unsandboxed`): neither bubblewrap nor Landlock could "
            "confine it on this machine (bwrap: setting up uid map: Permission denied), so the agent's Blender code "
            "could read and write anywhere the user who launched it could.") in run_readme.render(inner, set())
    meta["blender_sandbox"] = {"mode": "landlock", "reason": "x", "landlock_abi": 6}
    (inner / "meta.json").write_text(json.dumps(meta))
    assert "without a sandbox" not in run_readme.render(inner, set())


def test_authored_code_that_names_a_sandbox_path_is_told_where_it_is(tmp_path):
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip("bpy interpreter unavailable")
    code = ("import bpy\nfrom dycraft.animation.blender import worker\n"
            "e = FileNotFoundError(2, 'No such file or directory', '/workspace/animation_src/helper.py')\n"
            "print('HINT', repr(worker._host_path_hint(e)))\n"
            "m = FileNotFoundError(2, 'No such file or directory', '/references/mocap/cmu/data/x.bvh')\n"
            "print('MOCAP', repr(worker._host_path_hint(m)))\n")
    env = dict(os.environ, DYCRAFT_BLENDER_SANDBOX_MODE="landlock", DYCRAFT_BLENDER_WORKSPACE_ROOT=str(tmp_path),
               DYCRAFT_MOCAP_ROOT="/srv/mocap")
    ran = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert ran.returncode == 0, ran.stderr
    assert (f"(Blender runs without a private filesystem on this machine, so /workspace does not exist here; it is "
            f"{tmp_path}. Use paths relative to the working directory, {tmp_path}.)") in ran.stdout
    assert ("so /references/mocap does not exist here; it is /srv/mocap. Mocap_retarget and dog_retarget accept "
            "/references/mocap/... as it is.)") in ran.stdout
    env.pop("DYCRAFT_BLENDER_SANDBOX_MODE")
    ran = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert "HINT ''" in ran.stdout
