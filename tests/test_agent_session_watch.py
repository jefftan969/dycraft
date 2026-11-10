"""The claude session watch: a session that cannot go on is stopped and said so, a healthy one is left alone.

`ClaudeSessionWatch` reads the transcript Claude Code writes under its account's `projects/<run dir key>/` as the
run goes, the records shaped as recorded runs show them: a sandboxed Bash command that never started is a `user`
record whose `tool_result` is an error starting `Exit code 1` then `apply-seccomp: ...`; a usage limit or a missing
login is an `assistant` record with `isApiErrorMessage` and an `error`.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dycraft.agent.transcript_capture import project_key
from dycraft.cli.backends import _process
from dycraft.cli.backends import claude as C

SECCOMP = ("apply-seccomp: write /proc/self/setgroups (nested userns is capability-restricted; "
           "caller must provide CAP_SYS_ADMIN): Permission denied")
SESSION_LIMIT = "You've hit your session limit · resets 6:40pm (America/Los_Angeles)"


def _now(offset: float = 0.0) -> str:
    return datetime.fromtimestamp(time.time() + offset, tz=timezone.utc).isoformat()


def _bash(tool_id: str, command: str = "python harness.py status", offset: float = 0.0) -> dict:
    return {"type": "assistant", "timestamp": _now(offset),
            "message": {"id": f"msg_{tool_id}", "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_id, "name": "Bash",
                                     "input": {"command": command}}]}}


def _result(tool_id: str, text: str, *, error: bool, kind: str = "user", offset: float = 0.0) -> dict:
    return {"type": kind, "timestamp": _now(offset),
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
                                                     "is_error": error, "content": text}]}}


def _api_error(error: str, text: str, offset: float = 0.0) -> dict:
    return {"type": "assistant", "timestamp": _now(offset), "isApiErrorMessage": True, "error": error,
            "message": {"id": "msg_err", "role": "assistant", "content": [{"type": "text", "text": text}]}}


def _say(text: str = "Reading the brief.", offset: float = 0.0) -> dict:
    return {"type": "assistant", "timestamp": _now(offset),
            "message": {"id": f"msg_{text[:4]}", "role": "assistant", "content": [{"type": "text", "text": text}]}}


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A run dir, its account's project dir, and a writer for the transcript claude would keep there."""
    monkeypatch.delenv(C.NO_SESSION_WATCH_ENV, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = tmp_path / "account"
    project = config / "projects" / project_key(run_dir)
    watch = C.ClaudeSessionWatch(config, run_dir)

    def write(*records):
        project.mkdir(parents=True, exist_ok=True)
        with (project / "session.jsonl").open("a") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
    return watch, write, run_dir


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def test_a_sandbox_that_cannot_start_a_command_stops_the_run(session):
    watch, write, run_dir = session
    write(_bash("t1"), _result("t1", f"Exit code 1\n{SECCOMP}", error=True))
    status, message = watch.poll()
    assert status == _process.EXIT_AGENT_FATAL
    assert message.startswith("FATAL: Claude Code's sandbox cannot start a command on this host (apply-seccomp:")
    assert C.NO_SESSION_WATCH_ENV in message
    assert _meta(run_dir)["agent_stop"]["reason"] == "claude_sandbox_cannot_start"


@pytest.mark.parametrize("records", [
    # the same words as the agent's own text, in an echo of the conversation, and from a tool that is not Bash
    [_say(f"I will avoid {SECCOMP}")],
    [_bash("t1"), _result("t1", f"Exit code 1\n{SECCOMP}", error=True, kind="api-request-blob")],
    [{"type": "assistant", "timestamp": _now(), "message": {"id": "m", "content": [
        {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/x"}}]}},
     _result("r1", f"Exit code 1\n{SECCOMP}", error=True)],
    # a Bash command that succeeded first: the sandbox works, and a later `cat` of this file is content
    [_bash("t1"), _result("t1", "ok", error=False), _bash("t2", "cat cli/backends/claude.py"),
     _result("t2", f"Exit code 1\n{SECCOMP}", error=True)],
], ids=["agent-text", "request-echo", "not-bash", "after-a-success"])
def test_the_failure_text_alone_never_stops_a_healthy_session(session, records):
    watch, write, _ = session
    write(*records)
    assert watch.poll() is None


def test_a_session_that_never_writes_a_record_is_stopped_at_the_startup_bound(session, monkeypatch):
    watch, _, run_dir = session
    assert watch.poll() is None
    monkeypatch.setattr(C, "CLAUDE_STARTUP_BOUND_SECONDS", 0.0)
    status, message = watch.poll()
    assert status == _process.EXIT_AGENT_BLOCKED
    assert message.startswith("FATAL: claude wrote no session record in 0 s after it started")
    assert _meta(run_dir)["agent_stop"]["reason"] == "claude_never_started"


def test_on_a_terminal_the_startup_bound_waits_for_the_operator(session, monkeypatch):
    """There claude's folder-trust question, which comes before any record, is the operator's to answer (an R1 row on
    a terminal was stopped at 30 s while it stood)."""
    _, _, run_dir = session
    watch = C.ClaudeSessionWatch(run_dir.parent / "account", run_dir, interactive=True)
    monkeypatch.setattr(C, "CLAUDE_STARTUP_BOUND_SECONDS", 0.0)
    assert watch.poll() is None


def test_a_session_with_a_record_is_past_its_start(session, monkeypatch):
    watch, write, _ = session
    write(_say())
    monkeypatch.setattr(C, "CLAUDE_STARTUP_BOUND_SECONDS", 0.0)
    assert watch.poll() is None


@pytest.mark.parametrize("error, text, says", [
    ("rate_limit", SESSION_LIMIT,
     "FATAL: claude hit a usage or rate limit: You've hit your session limit · resets 6:40pm"),
    ("authentication_failed", "Not logged in · Please run /login", "FATAL: claude is not logged in: Not logged in"),
    # The server limiting requests, which is not the account's limit (an N2 matrix row), and one period after it
    ("rate_limit", "API Error: Server is temporarily limiting requests (not your usage limit) · Please try again later.",
     "FATAL: claude hit a usage or rate limit: API Error: Server is temporarily limiting requests (not your usage "
     "limit) · Please try again later. The CLI ended the session;"),
])
def test_a_usage_limit_or_no_login_is_named_when_the_cli_ends_on_it(session, capsys, error, text, says):
    watch, write, run_dir = session
    write(_say(), _api_error(error, text))
    assert watch.poll() is None                    # within its grace: Claude Code ends the turn itself
    assert watch.after_exit() == _process.EXIT_AGENT_FATAL
    assert capsys.readouterr().err.startswith(says)
    stop = _meta(run_dir)["agent_stop"]
    assert (stop["reason"], stop["detail"]) == (f"claude_{error}", text)


def test_a_usage_limit_the_cli_sits_on_is_stopped_after_its_grace(session, monkeypatch):
    watch, write, _ = session
    write(_api_error("rate_limit", SESSION_LIMIT))
    monkeypatch.setattr(C, "CLAUDE_FATAL_GRACE_SECONDS", 0.0)
    status, message = watch.poll()
    assert status == _process.EXIT_AGENT_FATAL and "The CLI is stopped" in message


def test_the_model_answering_after_an_error_clears_it(session):
    watch, write, _ = session
    write(_api_error("rate_limit", SESSION_LIMIT), _say("Back after the retry."))
    watch.poll()
    assert watch.after_exit() is None


def test_a_resumed_sessions_earlier_records_are_not_read_again(session):
    """`--resume` appends to the same file, which still holds the error that stopped the last launch."""
    watch, write, _ = session
    write(_api_error("rate_limit", SESSION_LIMIT, offset=-3600), _bash("t1", offset=-3600),
          _result("t1", f"Exit code 1\n{SECCOMP}", error=True, offset=-3600))
    assert watch.poll() is None
    assert watch.after_exit() is None


def test_the_watch_can_be_switched_off(session, monkeypatch):
    monkeypatch.setenv(C.NO_SESSION_WATCH_ENV, "1")
    _, write, run_dir = session
    watch = C.ClaudeSessionWatch(run_dir.parent / "account", run_dir)
    write(_bash("t1"), _result("t1", f"Exit code 1\n{SECCOMP}", error=True))
    assert watch.poll() is None


class _Verdict:
    """A watch that names a stop on its first poll, or names a status after the CLI exits."""
    poll_seconds = 0.05

    def __init__(self, stop=None, after=None):
        self.stop, self.after = stop, after

    def poll(self):
        return self.stop

    def after_exit(self, returncode=0):
        return self.after


def test_run_agent_cli_stops_the_cli_when_its_watch_says_so(capsys):
    started = time.monotonic()
    status = _process.run_agent_cli([sys.executable, "-c", "import time; time.sleep(60)"], {},
                                    watch=_Verdict(stop=(_process.EXIT_AGENT_BLOCKED, "FATAL: stuck")))
    assert status == _process.EXIT_AGENT_BLOCKED
    assert time.monotonic() - started < 30
    assert "FATAL: stuck" in capsys.readouterr().err


def test_run_agent_cli_reports_the_status_a_watch_names_after_exit():
    status = _process.run_agent_cli([sys.executable, "-c", "pass"], {},
                                    watch=_Verdict(after=_process.EXIT_AGENT_FATAL))
    assert status == _process.EXIT_AGENT_FATAL


def test_every_claude_launch_runs_under_the_session_watch(tmp_path, monkeypatch):
    """The watch is on for every claude launch, the arm and every sub-run alike, reading the account's own dir."""
    captured = {}
    monkeypatch.setattr(_process, "run_agent_cli",
                        lambda argv, env, watch=None: captured.update(watch=watch, env=env) or 0)
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr("os.chdir", lambda _p: None)
    from dycraft.cli.backends import BACKENDS, LaunchPlan
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    plan = LaunchPlan(layer="object", user_prompt="a cube", run_dir=run_dir, runs_dir=tmp_path, run_id="r",
                      model=None, env={"PATH": "/usr/bin"}, system_prompt_text="s", footer_text="f")
    BACKENDS["claude3"]().execute(plan)
    watch = captured["watch"]
    assert isinstance(watch, C.ClaudeSessionWatch)
    assert watch.interactive == _process._fd_is_tty(1)
    assert watch.config_dir == Path(captured["env"]["CLAUDE_CONFIG_DIR"]).expanduser()
    assert watch.run_dir == run_dir


def test_the_codex_startup_message_quotes_the_screen_it_stopped_on():
    assert _process._last_screen_line("  Sign in with ChatGPT \x07 to use Codex  ") == \
        "Sign in with ChatGPT  to use Codex"
    assert len(_process._last_screen_line("x" * 500)) == 160


def test_any_api_error_the_cli_ends_on_is_named(session, capsys):
    """Network trouble ends a claude session on an API error it retried for minutes (`server_error`)."""
    watch, write, run_dir = session
    write(_say(), _api_error("server_error", "API Error: Connection refused (ECONNREFUSED)"))
    assert watch.poll() is None                    # not stopped: claude's own retries decide
    assert watch.after_exit(1) == _process.EXIT_AGENT_FATAL
    assert capsys.readouterr().err.startswith(
        "FATAL: claude's API calls failed and it ended the session: API Error: Connection refused (ECONNREFUSED).")
    assert _meta(run_dir)["agent_stop"]["reason"] == "claude_api_error_server_error"


def test_a_cli_that_exits_before_the_first_poll_is_still_read(session, capsys):
    watch, write, _ = session
    write(_api_error("rate_limit", SESSION_LIMIT))   # written and exited before any poll
    assert watch.after_exit(1) == _process.EXIT_AGENT_FATAL
    assert "claude hit a usage or rate limit" in capsys.readouterr().err


def test_an_exit_before_any_answer_says_where_the_reason_is(session, capsys):
    """Claude Code prints "Not logged in" to its own stdout and records nothing in the transcript."""
    watch, write, run_dir = session
    write({"type": "agent-name", "timestamp": _now()})
    assert watch.after_exit(1) == _process.EXIT_AGENT_FATAL
    assert capsys.readouterr().err.startswith("FATAL: claude exited (status 1) before it answered.")
    assert _meta(run_dir)["agent_stop"]["reason"] == "claude_exited_before_answering"


def test_a_clean_exit_or_an_answered_session_says_nothing(session, capsys):
    watch, write, _ = session
    assert watch.after_exit(0) is None
    write(_say())
    assert watch.after_exit(1) is None
    assert capsys.readouterr().err == ""


def test_a_launch_from_inside_claude_code_does_not_pass_its_session_markers_on(tmp_path, monkeypatch):
    """With CLAUDE_CODE_CHILD_SESSION inherited, Claude Code saves no transcript, and the watch would see none."""
    captured = {}
    monkeypatch.setattr(_process, "run_agent_cli", lambda argv, env, watch=None: captured.update(env=env) or 0)
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr("os.chdir", lambda _p: None)
    from dycraft.cli.backends import BACKENDS, LaunchPlan
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    markers = {name: "1" for name in C.CLAUDE_SESSION_MARKERS}
    plan = LaunchPlan(layer="object", user_prompt="a cube", run_dir=run_dir, runs_dir=tmp_path, run_id="r",
                      model=None, env={"PATH": "/usr/bin", **markers}, system_prompt_text="s", footer_text="f")
    BACKENDS["claude3"]().execute(plan)
    assert not set(markers) & set(captured["env"])
