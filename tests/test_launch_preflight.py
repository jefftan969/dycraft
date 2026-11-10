"""What a launch checks before it starts anything, and what an operator sees while it prepares the run.

`python -m dycraft.cli` runs the environment checks this launch depends on before it stages a file or starts a
rewriter (`cli._environment_preflight`), so an install that missed a step is refused with the check's own message
and fix, the same text `python -m dycraft.install check` prints. A rewrite reports its progress while it runs, and a
rewriter that stopped on a fatal error says which.

The test session sets `DYCRAFT_SKIP_PREFLIGHT=1` for every launch (conftest); these tests unset it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dycraft import environment_test as E
from dycraft._layers import LAYERS
from dycraft.agent import oneoff
from dycraft.agent.transcript_capture import project_key
from dycraft.cli import cli
from dycraft.cli.backends import BACKENDS, ClaudeBackend, CodexBackend


@pytest.fixture
def checked(monkeypatch):
    monkeypatch.delenv(cli.SKIP_PREFLIGHT_ENV, raising=False)


def _check(cid: str) -> E.Check:
    return next(c for c in E.CHECKS if c.id == cid)


def test_every_layer_names_the_checks_it_depends_on():
    """A layer missing here would launch with only the common checks, and every id must be a registered check."""
    assert set(LAYERS) <= set(E.LAUNCH_CHECKS)
    known = {c.id for c in E.CHECKS}
    named = {cid for ids in E.LAUNCH_CHECKS.values() for cid in ids} | set(E.LAUNCH_HARNESS.values())
    assert named <= known, sorted(named - known)
    assert {b().name for b in BACKENDS.values()} <= set(E.LAUNCH_HARNESS)


def test_a_failed_or_missing_dependency_refuses_and_a_warning_does_not():
    cad, claude = _check("cad.interpreter"), _check("harness.claude")
    results = [(cad, E.fail("no interpreter")), (_check("repo.shadow"), E.warn("another dycraft")),
               (claude, E.degraded("socat is missing"))]
    assert [c.id for c, _ in E.launch_refusals(results, "claude")] == ["cad.interpreter", "harness.claude"]
    # A codex whose sandbox does not work is the codex backend's to refuse.
    codex = _check("harness.codex")
    assert E.launch_refusals([(codex, E.degraded("sandbox does not work"))], "codex") == []
    assert [c.id for c, _ in E.launch_refusals([(codex, E.unavailable("not on PATH"))], "codex")] == ["harness.codex"]


def test_a_launch_runs_its_layers_checks_and_their_needs():
    seen = [c.id for c, _ in E.launch_preflight("object", "claude")]
    assert {"repo.import", "cad.interpreter", "harness.claude"} <= set(seen)
    assert seen.index("repo.import") < seen.index("cad.interpreter")
    # DyCraft's own sandbox confines the animation layers' Blender workers, and nothing an object run starts.
    assert not {"blender.animation", "sandbox.bwrap", "sandbox.userns"} & set(seen)
    assert {"sandbox.bwrap", "sandbox.userns"} <= set(E.LAUNCH_CHECKS["animation"])


def test_with_the_opt_in_a_cli_sandbox_that_cannot_start_does_not_refuse(monkeypatch):
    claude = _check("harness.claude")
    sandbox = E.degraded("its Bash sandbox cannot start here", data={"sandbox": True})
    for name in ("DYCRAFT_ALLOW_UNSANDBOXED", "DYCRAFT_ALLOW_UNSANDBOXED_CODEX"):
        monkeypatch.delenv(name, raising=False)
    assert [c.id for c, _ in E.launch_refusals([(claude, sandbox)], "claude")] == ["harness.claude"]
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    assert E.launch_refusals([(claude, sandbox)], "claude") == []
    # Only a result about the sandbox: anything else still refuses.
    other = E.degraded("not logged in")
    assert [c.id for c, _ in E.launch_refusals([(claude, other)], "claude")] == ["harness.claude"]


def _args(layer="object", backend="claude3"):
    return SimpleNamespace(layer=layer, backend=backend)


def test_a_refused_launch_names_the_failed_check_its_fix_and_the_way_round_it(checked, tmp_path, monkeypatch, capsys):
    cad = _check("cad.interpreter")
    monkeypatch.setattr(E, "launch_preflight", lambda layer, family, progress=None: [
        (cad, E.fail("no build123d interpreter found", lost="object runs refuse to start",
                     fix=["python -m dycraft.install cad"]))])
    assert cli._environment_preflight(_args(), BACKENDS["claude3"](), tmp_path) == 2
    err = capsys.readouterr().err
    assert "[cli] checking the environment for a object run on claude3 ..." in err
    assert "FATAL: build123d interpreter: no build123d interpreter found" in err
    assert "  fix: python -m dycraft.install cad" in err and "  lost: object runs refuse to start" in err
    assert "Nothing was started" in err and "DYCRAFT_SKIP_PREFLIGHT=1 launches without these checks" in err
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["launch_failure"]["reason"] == "environment"
    assert meta["launch_failure"]["detail"] == "cad.interpreter: no build123d interpreter found"
    assert meta["launch_failure"]["message"] == "FATAL: build123d interpreter: no build123d interpreter found"
    assert meta["launch_failure"]["fix"] == ["python -m dycraft.install cad"]


def test_a_healthy_environment_says_so_and_goes_on(checked, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E, "launch_preflight", lambda layer, family, progress=None: [
        (_check("repo.import"), E.ok("imports")), (_check("repo.shadow"), E.warn("another dycraft"))])
    assert cli._environment_preflight(_args(), BACKENDS["claude3"](), tmp_path) is None
    err = capsys.readouterr().err
    assert "[cli] WARNING: no other checkout shadows it: another dycraft" in err
    assert "[cli] environment ok (2 checks," in err
    assert not (tmp_path / "meta.json").exists()


def test_the_checks_can_be_skipped_with_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cli.SKIP_PREFLIGHT_ENV, "1")
    monkeypatch.setattr(E, "launch_preflight", lambda *a, **k: pytest.fail("ran the checks"))
    assert cli._environment_preflight(_args(), BACKENDS["claude3"](), tmp_path) is None
    assert "DYCRAFT_SKIP_PREFLIGHT=1: launching without the environment checks" in capsys.readouterr().err


@pytest.fixture
def no_opt_in(monkeypatch):
    for name in ("DYCRAFT_ALLOW_UNSANDBOXED", "DYCRAFT_ALLOW_UNSANDBOXED_CODEX"):
        monkeypatch.delenv(name, raising=False)


def _bwrap(tmp_path, body: str) -> str:
    path = tmp_path / "bin" / "bwrap"
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("missing, named", [({"socat"}, "socat is not on PATH"),
                                            ({"bwrap", "socat"}, "bwrap and socat are not on PATH")])
def test_claude_refuses_where_its_sandbox_cannot_start(checked, no_opt_in, tmp_path, monkeypatch, capsys, missing,
                                                       named):
    monkeypatch.setattr("shutil.which", lambda name: None if name in missing else f"/usr/bin/{name}")
    assert ClaudeBackend().preflight(tmp_path, "object") == 2
    err = capsys.readouterr().err
    assert err.startswith(f"FATAL: Claude Code's Bash sandbox cannot start here: {named}. Install bubblewrap and "
                          "socat with `python -m dycraft.install tools`")
    assert "Or, without sudo, pass --allow-unsandboxed to run Claude Code without its sandbox" in err
    assert "DYCRAFT_SKIP_PREFLIGHT=1 launches without this check." in err
    assert json.loads((tmp_path / "meta.json").read_text())["launch_failure"]["reason"] == \
        "claude_sandbox_unavailable"


def test_claude_refuses_where_its_bwrap_cannot_make_namespaces(checked, no_opt_in, tmp_path, monkeypatch, capsys,
                                                               apparmor):
    """Stock Ubuntu 24.04 with a bwrap outside /usr/bin, and a system that ships a profile for /usr/bin/bwrap."""
    apparmor(restricted=True, bwrap=False, profiled=True)
    bwrap = _bwrap(tmp_path, 'echo "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted" >&2; exit 1')
    monkeypatch.setattr("shutil.which", lambda name: bwrap if name == "bwrap" else f"/usr/bin/{name}")
    assert ClaudeBackend().preflight(tmp_path, "object") == 2
    err = capsys.readouterr().err
    assert err.startswith(f"FATAL: Claude Code's Bash sandbox cannot start here: {bwrap} cannot make a user and a "
                          "network namespace here (bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted). "
                          "With root: install the system's bubblewrap: `sudo apt-get install bubblewrap socat`; or lift "
                          "the restriction for every program")
    failure = json.loads((tmp_path / "meta.json").read_text())["launch_failure"]
    assert failure["fix"][-1].startswith("without sudo: `--allow-unsandboxed` runs Claude Code without its sandbox")


def test_claude_goes_on_where_its_sandbox_can_start(checked, no_opt_in, tmp_path, monkeypatch, capsys):
    bwrap = _bwrap(tmp_path, "exit 0")
    monkeypatch.setattr("shutil.which", lambda name: bwrap if name == "bwrap" else f"/usr/bin/{name}")
    backend = ClaudeBackend()
    assert backend.preflight(tmp_path, "object") is None
    assert backend._unsandboxed is None and capsys.readouterr().err == ""


def test_with_the_opt_in_claude_starts_without_its_sandbox_and_says_so(checked, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    bwrap = _bwrap(tmp_path, 'echo "bwrap: setting up uid map: Permission denied" >&2; exit 1')
    monkeypatch.setattr("shutil.which", lambda name: bwrap if name == "bwrap" else f"/usr/bin/{name}")
    backend = ClaudeBackend()
    assert backend.preflight(tmp_path, "object") is None
    problem = f"{bwrap} cannot make a user and a network namespace here (bwrap: setting up uid map: Permission denied)"
    assert backend._unsandboxed == problem
    assert backend.launch_meta == {"unsandboxed": {"agent": "Claude Code", "reason": problem}}
    assert capsys.readouterr().err == (f"WARNING: running Claude Code without its sandbox (--allow-unsandboxed): it "
                                       f"cannot start here ({problem}). The agent's commands can read and write "
                                       "anywhere you can.\n")
    assert not (tmp_path / "meta.json").exists()


def test_with_the_opt_in_a_working_claude_sandbox_stays_on(checked, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    bwrap = _bwrap(tmp_path, "exit 0")
    monkeypatch.setattr("shutil.which", lambda name: bwrap if name == "bwrap" else f"/usr/bin/{name}")
    backend = ClaudeBackend()
    assert backend.preflight(tmp_path, "object") is None
    assert backend._unsandboxed is None and backend.launch_meta == {}
    assert "[cli] --allow-unsandboxed: Claude Code's sandbox works here, so it stays on" in capsys.readouterr().err


def test_the_check_reports_claude_runs_only_with_the_opt_in(tmp_path, monkeypatch):
    bwrap = _bwrap(tmp_path, 'echo "bwrap: setting up uid map: Permission denied" >&2; exit 1')
    real_which = E.which
    monkeypatch.setattr("shutil.which", lambda name: bwrap if name == "bwrap" else f"/usr/bin/{name}")
    monkeypatch.setattr(E, "which", lambda name: "/usr/bin/claude" if name == "claude" else real_which(name))
    monkeypatch.setattr(E, "version_of", lambda path: "2.1.319 (Claude Code)")
    result = E.check_claude(SimpleNamespace(facts={}))
    assert result.status == E.DEGRADED and result.data == {"sandbox": True}
    assert "so Claude Code runs only with --allow-unsandboxed (unconfined)" in result.summary
    assert result.fix[-1].startswith("without sudo: `--allow-unsandboxed`")


# ---------------------------------------------------------------------------
# A rewrite in progress, and a rewrite that stopped on a fatal error
# ---------------------------------------------------------------------------

def test_a_child_launchers_reason_for_stopping_is_found_in_its_output():
    err = "[cli] runs dir: x\n[claude_plain] launching claude\nFATAL: claude is not logged in: Not logged in\nmore\n"
    assert oneoff.fatal_line(err) == "FATAL: claude is not logged in: Not logged in"
    assert oneoff.fatal_line("[cli] nothing fatal\n") is None


def test_a_steps_target_reads_as_an_operator_would_name_it():
    assert oneoff._short("cat /tmp/run/sub_runs/scene_rewrite/animation.md") == "cat animation.md"
    assert oneoff._claude_step({"name": "Read", "input": {"file_path": "/a/b/animation_24_briefs.md"}}) == \
        "read animation_24_briefs.md"
    assert len(oneoff._short("x " * 100)) == 60


def test_progress_is_read_from_a_claude_sub_agents_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "account"))
    sub = tmp_path / "run" / "sub_runs" / "scene_rewrite"
    sub.mkdir(parents=True)
    project = tmp_path / "account" / "projects" / project_key(sub)
    project.mkdir(parents=True)
    records = [{"type": "assistant", "message": {"id": "m1", "content": [
                   {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": f"{sub}/animation.md"}}]}},
               {"type": "assistant", "message": {"id": "m2", "content": [
                   {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": f"wc -w {sub}/scene.md"}}]}}]
    (project / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert oneoff.session_progress("claude", sub, time.time() - 60) == \
        {"turns": 2, "tools": 2, "last": "ran `wc -w scene.md`"}


def test_a_codex_steps_escapes_read_as_what_they_stand_for():
    """An R1 heartbeat showed `\\"$DYCRAFT_PYTHON\\" - <<'PY'\\nimport harness`: the literal's escapes, not the command."""
    payload = {"type": "custom_tool_call", "name": "exec",
               "input": 'text(await tools.exec_command({cmd:"\\"$DYCRAFT_PYTHON\\" - <<\'PY\'\\nimport harness as h"}));'}
    assert oneoff._codex_step(payload) == "ran `\"$DYCRAFT_PYTHON\" - <<'PY' import harness as h`"


def test_progress_is_read_from_a_codex_sub_agents_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(CodexBackend, "HOME", str(tmp_path / "codex"))
    sub = tmp_path / "run" / "sub_runs" / "motion_rewrite"
    sub.mkdir(parents=True)
    rollout = tmp_path / "codex" / "sessions" / "2026" / "rollout-x.jsonl"
    rollout.parent.mkdir(parents=True)
    records = [{"type": "session_meta", "payload": {"cwd": str(sub.resolve())}},
               {"type": "event_msg", "payload": {"type": "task_started"}},
               {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
                                                     "input": 'text(await tools.exec_command({cmd:"cat '
                                                              f'{sub}/scene.md"}}));'}}]
    rollout.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert oneoff.session_progress("codex", sub, time.time() - 60) == \
        {"turns": 1, "tools": 1, "last": "ran `cat scene.md`"}


def _beats(monkeypatch, tty, seconds=0.2):
    import io
    import threading
    out = io.StringIO()
    monkeypatch.setattr(oneoff.sys, "stderr", out)
    monkeypatch.setattr(oneoff, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(oneoff, "HEARTBEAT_LOG_SECONDS", 0.05)
    monkeypatch.setattr(oneoff, "session_progress", lambda *a: {"turns": 3, "tools": 5, "last": "read scene.md"})
    done = threading.Event()
    thread = threading.Thread(target=oneoff._heartbeat, args=(SimpleNamespace(poll=lambda: None), "scene rewrite",
                                                              "claude", Path("."), done, 0, tty))
    thread.start()
    time.sleep(seconds)
    done.set()
    thread.join(2)
    return out.getvalue()


def test_a_rewrite_heartbeat_says_what_the_sub_agent_did(monkeypatch):
    """In a log, one line per interval (a minute, outside this test)."""
    lines = [ln for ln in _beats(monkeypatch, tty=False).splitlines() if ln.startswith("[cli]   … scene rewrite:")]
    assert lines and lines[0].endswith(", 3 turns, 5 tool calls, last: read scene.md")


def test_on_a_terminal_the_heartbeat_is_one_line_redrawn_in_place(monkeypatch):
    """The user: the heartbeats "shouldn't pile up they can be shown in one line"."""
    out = _beats(monkeypatch, tty=True)
    assert "\n" not in out and out.count("\r\x1b[K[cli]   … scene rewrite:") >= 2
    assert out.endswith("\r\x1b[K")              # cleared, so the stage's closing line takes its place


def test_a_rewriter_that_stopped_on_a_fatal_error_says_which(tmp_path, monkeypatch):
    from dycraft.agent import scene_rewrite
    monkeypatch.setattr(oneoff, "run_oneoff", lambda **kw: {
        "ok": False, "text": None, "outcome": "no_result", "timed_out": False, "n_op_calls": 0,
        "returncode": 86, "wall_time_s": 3.0, "sub_run": "sub_runs/scene_rewrite",
        "stderr_tail": "…ugh the tail", "fatal": "FATAL: claude is not logged in: Not logged in · Please run /login"})
    with pytest.raises(RuntimeError) as err:
        scene_rewrite.rewrite("A robot waters the plants.", backend="claude3", run_dir=tmp_path)
    assert str(err.value).endswith(": FATAL: claude is not logged in: Not logged in · Please run /login")


def test_a_python_that_is_not_dycrafts_environment_is_told_so(tmp_path, pkgroot):
    """The commonest wrong setup reads as a FATAL naming the interpreter, not as a traceback. `pkgroot` names this
    checkout `dycraft` whatever its folder is called, so the child runs the code under test."""
    import subprocess
    import sys
    blocker = tmp_path / "numpy.py"
    blocker.write_text("raise ModuleNotFoundError(\"No module named 'numpy'\", name='numpy')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(pkgroot)])}
    ran = subprocess.run([sys.executable, "-m", "dycraft.cli", "--backend", "claude", "--layer", "object",
                          "--prompt", "x"], env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert ran.returncode == 2
    assert f"FATAL: {sys.executable} cannot import `numpy`, which DyCraft needs" in ran.stderr
    assert "Traceback" not in ran.stderr


# ---------------------------------------------------------------------------
# A Codex account that cannot start a session, named before the sandbox probe
# ---------------------------------------------------------------------------

def _home(tmp_path, config=None, auth=True):
    home = tmp_path / ".codex2"
    home.mkdir()
    if config is not None:
        (home / "config.toml").write_text(config)
    if auth:
        (home / "auth.json").write_text("{}")
    return home


@pytest.mark.parametrize("make, reason, says", [
    (lambda t: t / "nowhere", "codex_home_missing", "does not exist, so this account has no login"),
    (lambda t: _home(t, config="= broken\n"), "codex_config_invalid", "Codex cannot read its config"),
    (lambda t: _home(t, auth=False), "codex_not_logged_in", "Codex is not logged in on"),
], ids=["no-home", "broken-config", "no-login"])
def test_a_codex_account_that_cannot_start_is_named(tmp_path, make, reason, says):
    from dycraft.cli.backends.codex import codex_account_problem
    found = codex_account_problem(str(make(tmp_path)), env={})
    assert found is not None and found[0] == reason and says in found[1]


@pytest.mark.parametrize("make, env", [
    (lambda t: _home(t), {}),
    (lambda t: _home(t, auth=False), {"OPENAI_API_KEY": "k"}),
    (lambda t: _home(t, config='model_provider = "local"\n', auth=False), {}),
    (lambda t: _home(t, config='cli_auth_credentials_store = "keyring"\n', auth=False), {}),
], ids=["logged-in", "api-key", "own-provider", "keyring"])
def test_a_codex_account_that_can_start_passes(tmp_path, make, env):
    from dycraft.cli.backends.codex import codex_account_problem
    assert codex_account_problem(str(make(tmp_path)), env=env) is None


def test_a_codex_launch_on_a_missing_account_is_refused_before_the_sandbox_probe(checked, tmp_path, monkeypatch,
                                                                                  capsys):
    """A missing account used to be reported as a sandbox that does not start, with fixes for bubblewrap."""
    from dycraft.cli.backends import codex as X
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", lambda self: ("/bin/codex", str(tmp_path / "nowhere")))
    monkeypatch.setattr(X, "codex_sandbox_probe", lambda *a, **k: pytest.fail("ran the sandbox probe"))
    assert CodexBackend().preflight(tmp_path, "object") == 2
    err = capsys.readouterr().err
    assert err.startswith(f"FATAL: Codex's account home {tmp_path / 'nowhere'} does not exist")
    assert f"  fix: `CODEX_HOME={tmp_path / 'nowhere'} codex login`" in err and "sandbox" not in err
    failure = json.loads((tmp_path / "meta.json").read_text())["launch_failure"]
    assert failure["reason"] == "codex_home_missing" and failure["fix"][0].endswith("codex login`")


def test_a_launch_that_renders_with_blender_finds_its_interpreter_without_the_blender_checks(monkeypatch):
    """An object launch's checks include the stills renderer but not the Blender checks, which used to be the only
    ones to look the bpy interpreter up (a resume of a run that switched to Blender was refused)."""
    from dycraft.render import _blender_env as be
    monkeypatch.setattr("dycraft.render.default_backend", lambda: "blender")
    monkeypatch.setattr(be, "blender_python", lambda: "/opt/blender-env/bin/python")
    monkeypatch.setattr(be, "blender_binary", lambda: None)
    results = dict((c.id, r) for c, r in E.launch_preflight("object", "claude"))
    assert results["render.stills"].status == E.PASS, results["render.stills"].summary


@pytest.mark.parametrize("flag", ["--allow-unsandboxed", "--allow-unsandboxed-codex"])
def test_the_opt_in_and_its_old_name_set_the_switch_the_daemons_children_inherit(flag, monkeypatch):
    # Set first, so what the launch sets on this process is put back afterwards.
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "0")
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    parser = cli.build_parser()
    args = parser.parse_args(["--backend", "codex3", "--layer", "plain", "--prompt", "x", flag])
    cli._settle_args(parser, args)
    assert args.allow_unsandboxed is True and os.environ["DYCRAFT_ALLOW_UNSANDBOXED"] == "1"
    args = parser.parse_args(["--backend", "codex3", "--layer", "plain", "--prompt", "x"])
    assert args.allow_unsandboxed is False


# ---------------------------------------------------------------------------
# A Claude account that cannot start an agent, and a launch refused before its folder exists
# ---------------------------------------------------------------------------

def _claude_status(tmp_path, body: str) -> str:
    path = tmp_path / "bin" / "claude"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"#!/bin/sh\n[ \"$1 $2\" = \"auth status\" ] || exit 9\necho '{body}'\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("body, says", [
    ('{"loggedIn": false, "authMethod": "none"}', "not logged in"),
    ('{"loggedIn": true, "loginExpiresAt": "2025-01-02T03:04:05.000Z"}', "its login expired on 2025-01-02"),
    ('{"loggedIn": true, "loginExpiresAt": "2099-01-02T03:04:05.000Z"}', None),
    ("Unknown command: auth", None),
], ids=["logged-out", "expired", "fine", "old-cli"])
def test_the_claude_login_is_read_from_auth_status_without_a_model_call(tmp_path, body, says):
    from dycraft.cli.backends.claude import claude_login_problem
    assert claude_login_problem(_claude_status(tmp_path, body), str(tmp_path / "acct")) == says


def test_the_login_fix_names_the_account_a_claudeN_launch_uses():
    from dycraft.cli.backends.claude import claude_login_fix
    assert claude_login_fix(None) == "log in: `claude auth login`"
    assert claude_login_fix(str(Path("~/.claude3").expanduser())) == \
        "log in: `CLAUDE_CONFIG_DIR=~/.claude3 claude auth login`"


def test_a_claude_launch_whose_account_is_logged_out_is_refused(checked, tmp_path, monkeypatch, capsys):
    from dycraft.cli.backends import claude as C
    monkeypatch.setattr(C, "claude_login_problem", lambda exe, account: "not logged in")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    backend = BACKENDS["claude3"]()
    assert backend.preflight(None, "object") == 2
    err = capsys.readouterr().err
    assert err.startswith(f"FATAL: Claude Code cannot start an agent with the account in ~/.claude3: not logged in. "
                          "Log in: `CLAUDE_CONFIG_DIR=~/.claude3 claude auth login`, then launch again.")
    assert backend.preflight(tmp_path, "object") == 2
    failure = json.loads((tmp_path / "meta.json").read_text())["launch_failure"]
    assert (failure["reason"], failure["fix"]) == ("claude_not_logged_in",
                                                   ["log in: `CLAUDE_CONFIG_DIR=~/.claude3 claude auth login`"])


def test_a_refused_top_level_launch_leaves_no_run_behind(checked, isolated_environ, tmp_path, monkeypatch, capsys):
    cad = _check("cad.interpreter")
    monkeypatch.setattr(E, "launch_preflight", lambda layer, family, progress=None: [
        (cad, E.fail("no build123d interpreter found", fix=["python -m dycraft.install cad"]))])
    runs = tmp_path / "runs"
    assert cli.main(["--backend", "claude", "--layer", "object", "--prompt", "A stool.",
                     "--runs-dir", str(runs)]) == 2
    assert "FATAL: build123d interpreter: no build123d interpreter found" in capsys.readouterr().err
    assert not runs.exists() or not any(runs.iterdir())
