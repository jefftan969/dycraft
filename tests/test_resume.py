"""What a run needs to be carried on by a later launcher: `python -m dycraft.cli --resume`, or a daemon started again.

A new daemon numbers its steps on from the run's trace, so nothing an earlier daemon wrote under a step number is
overwritten. A daemon that stopped (killed, its machine rebooted) leaves its port file, owner record and log behind;
the next daemon's launcher sets them aside rather than reading the old port as the new daemon being ready.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dycraft.agent.agent_session import AgentSession, agent_time, highest_trace_step
from dycraft.cli import cli


def _trace(run_dir: Path, *steps: tuple[int, int]) -> None:
    with (run_dir / "trace.jsonl").open("a") as fh:
        for epoch, step in steps:
            fh.write(json.dumps({"ev": "op_call", "epoch": epoch, "step": step}, sort_keys=True) + "\n")


def test_a_new_session_numbers_its_steps_on_from_the_trace(tmp_path):
    """A paper run that restarted at 0 with every daemon kept 30 of its compile reports; the rest were overwritten."""
    _trace(tmp_path, (101, 1), (101, 7), (202, 1), (202, 3))      # an older trace, numbered again by its 2nd daemon
    assert highest_trace_step(tmp_path) == 7
    session = AgentSession(SimpleNamespace(run_dir=tmp_path, step=0))
    assert session.tick() == 8


def test_a_new_run_starts_at_step_one(tmp_path):
    assert highest_trace_step(tmp_path) == 0
    assert AgentSession(SimpleNamespace(run_dir=tmp_path, step=0)).tick() == 1


def test_a_stopped_daemons_files_are_set_aside_before_the_next_one_starts(tmp_path, monkeypatch, capsys):
    """The old port must not read as the new daemon being ready, and the old log is kept."""
    for name, text in ((".harness.port", "40001"), (".harness.owner.json", '{"pid": 1}'), (".daemon.log", "old")):
        (tmp_path / name).write_text(text)

    def spawn(argv, *, cwd, log_fh, env):
        assert not (tmp_path / ".harness.port").exists(), "the stopped daemon's port is still where readiness is read"
        (tmp_path / ".harness.port").write_text("40002")
        return 4242
    monkeypatch.setattr(cli, "_spawn_detached", spawn)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    assert cli.start_harness_daemon(tmp_path, {}, daemon_module="dycraft.agent.layers.server") == 4242
    assert (tmp_path / ".harness.port").read_text() == "40002"
    for name, text in ((".harness.port", "40001"), (".harness.owner.json", '{"pid": 1}'), (".daemon.log", "old")):
        [aside] = tmp_path.glob(f"{name}.*")
        assert aside.read_text() == text
    assert "set the stopped daemon's .harness.port, .harness.owner.json, .daemon.log aside" in capsys.readouterr().err


def _iso(offset: float) -> str:
    import time
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time() + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def test_the_claude_session_a_launch_ran_is_recorded_for_resume(tmp_path, monkeypatch):
    from dycraft.agent.transcript_capture import project_key
    from dycraft.cli.backends import claude as C
    monkeypatch.delenv(C.NO_SESSION_WATCH_ENV, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text('{"run_id": "run"}')
    project = tmp_path / "account" / "projects" / project_key(run_dir)
    project.mkdir(parents=True)
    (project / "5f0c-session.jsonl").write_text(json.dumps({"type": "user", "timestamp": _iso(0)}) + "\n")
    C.ClaudeSessionWatch(tmp_path / "account", run_dir).poll()
    assert json.loads((run_dir / "meta.json").read_text())["claude_session_id"] == "5f0c-session"


def test_a_resumed_codex_sessions_earlier_records_are_not_read_again(tmp_path):
    """`codex resume` appends to the session's rollout, which still ends on the error that stopped the last launch."""
    from dycraft.cli.backends import _process
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout = tmp_path / "home" / "sessions" / "2026" / "rollout-x.jsonl"
    rollout.parent.mkdir(parents=True)
    limit = {"type": "event_msg", "timestamp": _iso(-3600), "payload": {
        "type": "task_complete", "error": {"codex_error_info": "usage_limit_exceeded", "message": "limit"}}}
    rollout.write_text("".join(json.dumps(r) + "\n" for r in [
        {"type": "session_meta", "timestamp": _iso(-7200), "payload": {"cwd": str(run_dir.resolve())}},
        {"type": "response_item", "timestamp": _iso(-3700), "payload": {"type": "reasoning"}}, limit,
        {"type": "event_msg", "timestamp": _iso(1), "payload": {"type": "task_started"}}]))
    watch = _process._CodexRolloutWatch(["codex", "resume", "-C", str(run_dir), "sid", "go on"],
                                        {"CODEX_HOME": str(tmp_path / "home")})
    watch.poll()
    assert watch.path == rollout
    assert watch.armed_at is None and not watch.ever_answered      # the old answer and error are the last launch's


# ---------------------------------------------------------------------------
# --resume: a fake run whose claude session lives in a fake account
# ---------------------------------------------------------------------------

import fcntl
import os
import time

import pytest

from dycraft.agent.transcript_capture import project_key
from dycraft.cli import resume as R


def _ts(offset: float = 0.0) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time() + offset, tz=timezone.utc).isoformat()


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A claude object run that stopped on a usage limit after 1 h of a 3 h limit, its session in a fake account.

    Nothing here may start a daemon or an agent CLI: a refusal that let a resume through would otherwise launch one.
    The variables a resume sets on this process are set first, so they are put back afterwards."""
    from dycraft.cli import cli as C
    from dycraft.cli.backends import ClaudeBackend
    monkeypatch.delenv("DYCRAFT_ADVISORY_HOURS", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "ambient"))
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    monkeypatch.setattr(C, "start_harness_daemon", lambda *a, **k: pytest.fail("started a harness daemon"))
    monkeypatch.setattr(ClaudeBackend, "execute", lambda *a, **k: pytest.fail("started the agent CLI"))
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    account = tmp_path / "account"
    project = account / "projects" / project_key(run_dir)
    project.mkdir(parents=True)
    (project / "s-1.jsonl").write_text(json.dumps({"type": "assistant", "message": {"content": []}}) + "\n")
    meta = {"run_id": "r1", "runtime": "claude", "layer": "object", "model": "sonnet", "reasoning_effort": "low",
            "prompt": "A stool.", "outcome": "interrupted", "claude_config_dir": str(account),
            "claude_session_id": "s-1", "advisory_hours": 1.0,
            "time_limit": {"hours": 3.0, "deadline": "x", "reached": False},
            "agent_stop": {"reason": "claude_rate_limit", "detail": "You've hit your session limit"}}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    with (run_dir / "trace.jsonl").open("w") as fh:
        for rec in ({"ev": "daemon_start", "epoch": 11, "step": 0, "ts": _ts(-7200)},
                    {"ev": "op_call", "epoch": 11, "step": 4, "ts": _ts(-3600)}):
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return run_dir


def _unchanged(run_dir: Path):
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    return lambda: {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == before


def _meta_edit(run_dir: Path, **fields) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(fields)
    (run_dir / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("setup, argv, says", [
    (lambda r: None, ["--layer", "animation"], "--resume takes the layer, model, effort and render backend"),
    (lambda r: (r / "meta.json").unlink(), [], "is not a run directory: it has no readable meta.json"),
    (lambda r: _meta_edit(r, claude_session_id="gone"), [], "session gone, which this run used, is no longer in"),
    (lambda r: [*(f.unlink() for f in (r.parents[1] / "account").rglob("*.jsonl")),
                _meta_edit(r, claude_session_id=None, launch_failure={"message": "FATAL: claude is not logged in"})],
     [], "its agent never started (claude is not logged in)"),
    (lambda r: _meta_edit(r, outcome="committed"), [], 'delivered its film (outcome committed). To revise it, resume '
                                                        'with --prompt "<what to change>".'),
    (lambda r: (r / ".terminate").touch(), [], "the agent ended this run itself without delivering"),
    (lambda r: _meta_edit(r, time_limit={"hours": 1.0, "reached": True}), [],
     "this run used its 1 h limit; pass --max-hours H to continue it for H more hours"),
], ids=["other-flags", "not-a-run", "session-gone", "never-started", "delivered", "ended-itself", "time-used"])
def test_a_run_that_cannot_resume_is_refused_in_one_line_and_left_as_it_was(run, capsys, setup, argv, says):
    setup(run)
    unchanged = _unchanged(run)
    assert R.main(["--resume", str(run), *argv]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FATAL: ") and says in err and err.count("FATAL") == 1
    assert unchanged()


def test_a_run_still_served_by_its_daemon_is_refused(run, capsys):
    with (run / ".harness.owner.lock").open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        assert R.main(["--resume", str(run)]) == 2
    assert "is still running: a harness daemon holds it" in capsys.readouterr().err


def test_a_moved_claude_run_is_refused(run, capsys):
    (run / "agent_transcript.jsonl").write_text(json.dumps({"sessionId": "s-1", "cwd": "/elsewhere/r1"}) + "\n")
    assert R.main(["--resume", str(run)]) == 2
    assert "was started at /elsewhere/r1, and claude looks its session up by that path" in capsys.readouterr().err


@pytest.fixture
def launched(monkeypatch):
    """`--resume`'s launch tail faked: what it would start the agent CLI with."""
    from dycraft.cli import cli as C
    seen: dict = {}
    monkeypatch.setattr(C, "_environment_preflight", lambda *a, **k: None)
    monkeypatch.setattr("dycraft.cli.backends.ClaudeBackend.preflight", lambda self, *a, **k: None)
    monkeypatch.setattr("dycraft.agent.stage_harness_client", lambda run_dir, layer: seen.update(staged=layer))
    monkeypatch.setattr(C, "start_harness_daemon", lambda run_dir, env, **k: seen.update(env=env) or 4242)
    monkeypatch.setattr(C, "start_self_kill_watcher", lambda *a, **k: None)
    monkeypatch.setattr(C, "install_self_kill_exit", lambda *a, **k: None)
    monkeypatch.setattr(C, "run_agent", lambda backend, plan, deadline, daemon_pid: seen.update(
        backend=backend, plan=plan, deadline=deadline) or 0)
    return seen


def test_a_stopped_run_resumes_its_own_session_with_the_time_it_has_left(run, launched):
    assert R.main(["--resume", str(run)]) == 0
    plan = launched["plan"]
    assert plan.resume_session == "s-1" and plan.layer == "object" and plan.model == "sonnet"
    assert plan.footer_text == (
        "This session was interrupted (your account reached its usage limit) and has been resumed: the harness daemon "
        "was restarted and answers again, and your files are as you left them. Continue the task from where you left "
        "off.")
    assert launched["staged"] == "object"
    assert 1.9 * 3600 < launched["deadline"] - time.time() < 2.0 * 3600     # 3 h less the 1 h the agent used
    assert launched["env"]["DYCRAFT_ADVISORY_HOURS"] == repr(1 / 60)        # its 1 h budget is spent
    meta = json.loads((run / "meta.json").read_text())
    assert meta["outcome"] is None and "agent_stop" not in meta
    assert meta["resumes"][0]["from_stop"] == "claude_rate_limit" and meta["resumes"][0]["session"] == "s-1"
    assert meta["agent_time"]["seconds"] == pytest.approx(3600, abs=5)


def test_a_resume_with_a_prompt_passes_it_on_and_max_hours_sets_the_time(run, launched):
    (run / ".terminate").touch()
    assert R.main(["--resume", str(run), "--prompt", "Make the legs thicker.", "--max-hours", "0.5"]) == 0
    assert launched["plan"].footer_text == (
        "This session was interrupted (you ended the session yourself without delivering) and has been resumed: the "
        "harness daemon was restarted and answers again, and your files are as you left them. The operator adds:"
        "\n\nMake the legs thicker.")
    assert 0.49 * 3600 < launched["deadline"] - time.time() <= 0.5 * 3600
    assert not (run / ".terminate").exists() and list(run.glob(".terminate.*"))
    assert json.loads((run / "meta.json").read_text())["resumes"][0]["prompt"] == "Make the legs thicker."


def test_a_delivered_run_is_reopened_before_its_revision(run, launched, monkeypatch):
    _meta_edit(run, outcome="committed")
    note = "Your finish() in the previous session succeeded (outcome: committed). ..."
    monkeypatch.setattr(R, "_layout", lambda: (lambda r: r, lambda r: r, lambda r: {
        "prior": "prior_deliveries/1", "outcome": "committed", "note": note}))
    assert R.main(["--resume", str(run), "--prompt", "Make the legs thicker."]) == 0
    assert launched["plan"].footer_text == f"{note}\n\nMake the legs thicker."
    assert json.loads((run / "meta.json").read_text())["resumes"][0]["prior_delivery"] == "prior_deliveries/1"


def test_an_animation_resume_is_told_when_it_is_stopped(run, launched):
    _meta_edit(run, layer="animation")
    assert R.main(["--resume", str(run), "--max-hours", "2"]) == 0
    assert "\n\nThis run is stopped 2 hours after this resume, at " in launched["plan"].footer_text


@pytest.mark.parametrize("stop, why", [
    ({"reason": "oom_kill"}, "the machine ran out of memory and your CLI was stopped"),
    ({"reason": "sigkill"}, "your CLI was stopped"),
    ({"reason": "codex_fatal_error", "detail": "usage_limit_exceeded: limit"}, "your account reached its usage limit"),
    ({"reason": "codex_reconnecting"}, "your model could not be reached"),
    ({"reason": "claude_api_error_server_error"}, "your model could not be reached"),
    ({}, "for a reason that was not recorded"),
])
def test_each_stop_has_its_reason_in_the_resume_message(stop, why):
    assert R.why_it_stopped({"agent_stop": stop}, ended_itself=False) == why
    assert R.why_it_stopped({"agent_stop": stop, "time_limit": {"reached": True}}, ended_itself=False) == \
        "the run's time limit was reached"


# ---------------------------------------------------------------------------
# The launch tail both a launch and a resume end in
# ---------------------------------------------------------------------------

class _Backend:
    """Exits before it answered on its first start (recording why), then runs."""
    harness_wait_s = None

    def __init__(self, run_dir, statuses, reason="claude_exited_before_answering"):
        self.run_dir, self.statuses, self.reason, self.calls = run_dir, list(statuses), reason, 0

    def execute(self, plan):
        self.calls += 1
        status = self.statuses.pop(0)
        if status == 86:
            from dycraft.agent.agent_session import record_agent_stop
            record_agent_stop(self.run_dir, reason=self.reason, detail="exit 1")
        return status


def _plan(run_dir):
    from dycraft.cli.backends import LaunchPlan
    return LaunchPlan(layer="object", user_prompt="p", run_dir=run_dir, runs_dir=run_dir.parent, run_id="r",
                      model=None, env={}, system_prompt_text="s", footer_text="f")


def test_a_cli_that_exits_before_it_answered_is_started_once_more(run, monkeypatch, capsys):
    from dycraft.cli import cli as C
    monkeypatch.setattr(C, "STARTUP_RETRY_DELAY_S", 0.0)
    backend = _Backend(run, [86, 0])
    assert C.run_agent(backend, _plan(run), None, daemon_pid=0) == 0
    assert backend.calls == 2
    assert "starting it once more in 0 s, with the same harness daemon" in capsys.readouterr().err
    meta = json.loads((run / "meta.json").read_text())
    assert meta["startup_retries"][0]["reason"] == "claude_exited_before_answering" and "agent_stop" not in meta


def test_a_named_stop_is_not_retried_and_says_how_to_resume(run, monkeypatch, capsys):
    from dycraft.cli import cli as C
    backend = _Backend(run, [86], reason="claude_rate_limit")
    assert C.run_agent(backend, _plan(run), None, daemon_pid=0) == 86
    assert backend.calls == 1
    assert f"[cli] to carry this run on in the same agent session: python -m dycraft.cli --resume {run}" in \
        capsys.readouterr().err
    stop = json.loads((run / "meta.json").read_text())["agent_stop"]
    assert stop["resume"] == f"python -m dycraft.cli --resume {run}"


def test_the_agent_time_counts_each_launch_from_the_trace(tmp_path):
    with (tmp_path / "trace.jsonl").open("w") as fh:
        for rec in ({"ev": "daemon_start", "epoch": 1, "ts": "2026-11-08T10:00:00+00:00"},
                    {"ev": "daemon_shutdown", "epoch": 1, "ts": "2026-11-08T11:00:00+00:00"},
                    {"ev": "op_call", "epoch": 9, "ts": "2026-11-08T11:30:00+00:00"},       # not a daemon's
                    {"ev": "daemon_start", "epoch": 2, "ts": "2026-11-08T14:00:00+00:00"},  # after a 3 h wait
                    {"ev": "op_call", "epoch": 2, "ts": "2026-11-08T14:30:00+00:00"}):      # then killed
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    used = agent_time(tmp_path)
    assert used["seconds"] == 5400
    assert [launch["how"] for launch in used["launches"]] == ["daemon_shutdown", "last_event"]


def test_a_codex_resume_hands_its_session_back_with_a_cold_starts_flags(tmp_path, monkeypatch):
    from dycraft.cli.backends import CodexBackend, LaunchPlan, _process
    seen = {}
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", lambda self: ("/bin/codex", str(tmp_path / "home")))
    monkeypatch.setattr(_process, "run_agent_cli", lambda cmd, env, watch=None: seen.update(cmd=cmd) or 0)
    monkeypatch.setattr("dycraft.cli.backends.codex.mark_codex_untrusted", lambda *a, **k: (lambda: None))
    backend = CodexBackend()
    backend._preflight_done = True
    plan = LaunchPlan(layer="object", user_prompt="p", run_dir=tmp_path, runs_dir=tmp_path.parent, run_id="r",
                      model="gpt-5", env={}, system_prompt_text="SYSTEM", footer_text="carry on",
                      reasoning_effort="low", resume_session="019a-sess")
    assert backend.execute(plan) == 0
    cmd = seen["cmd"]
    assert cmd[:4] == ["/bin/codex", "resume", "-C", str(tmp_path)] and cmd[-2:] == ["019a-sess", "carry on"]
    assert "--sandbox" in cmd and "-m" in cmd and "model_reasoning_effort=low" in cmd
    assert not any("SYSTEM" in a for a in cmd)


# ---------------------------------------------------------------------------
# Resuming on another account of the same CLI
# ---------------------------------------------------------------------------

def test_a_claude_run_resumes_on_another_account_with_its_session_copied_over(run, launched, tmp_path, monkeypatch):
    other = tmp_path / "account2"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(other))           # plain `claude` follows the ambient account
    source = tmp_path / "account" / "projects" / project_key(run) / "s-1.jsonl"
    assert R.main(["--resume", str(run), "--backend", "claude"]) == 0
    copied = other / "projects" / project_key(run) / "s-1.jsonl"
    assert copied.read_bytes() == source.read_bytes() and source.exists()   # copied, not moved
    assert launched["plan"].resume_session == "s-1"
    meta = json.loads((run / "meta.json").read_text())
    assert meta["claude_config_dir"] == str(other) and meta["runtime"] == "claude"
    assert meta["resumes"][0]["from_account"] == str(tmp_path / "account")


def test_a_codex_run_resumes_on_another_account_with_its_rollout_copied_over(tmp_path, monkeypatch, launched):
    from dycraft.cli.backends import BACKENDS
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    monkeypatch.setattr("dycraft.cli.backends.CodexBackend.preflight", lambda self, *a, **k: None)
    run_dir = tmp_path / "runs" / "c1"
    run_dir.mkdir(parents=True)
    home1, home2 = tmp_path / "codex1", tmp_path / "codex2"
    rollout = home1 / "sessions" / "2026" / "11" / "08" / "rollout-2026-11-08T10-00-00-sess-9.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("".join(json.dumps(r) + "\n" for r in [
        {"type": "session_meta", "payload": {"id": "sess-9", "cwd": str(run_dir)}},
        {"type": "response_item", "payload": {"type": "reasoning"}}]))
    (run_dir / "meta.json").write_text(json.dumps({
        "runtime": "codex1", "layer": "object", "codex_home": str(home1), "codex_session_id": "sess-9",
        "outcome": "interrupted", "agent_stop": {"reason": "codex_fatal_error", "detail": "usage_limit_exceeded: x"}}))
    monkeypatch.setattr(BACKENDS["codex2"], "HOME", str(home2))
    assert R.main(["--resume", str(run_dir), "--backend", "codex2"]) == 0
    assert (home2 / rollout.relative_to(home1)).read_bytes() == rollout.read_bytes() and rollout.exists()
    assert launched["plan"].resume_session == "sess-9"
    assert "your account reached its usage limit" in launched["plan"].footer_text
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["codex_home"] == str(home2) and meta["runtime"] == "codex2"


def test_a_resume_on_the_other_cli_is_refused(run, capsys):
    unchanged = _unchanged(run)
    assert R.main(["--resume", str(run), "--backend", "codex2"]) == 2
    assert capsys.readouterr().err.startswith(
        "FATAL: --backend codex2 is not a claude account, and this run's session is a claude session")
    assert unchanged()


def test_a_delivered_object_run_is_revised_through_run_layout(run, launched):
    """The real `reopen_delivery`: the delivery moves to prior_deliveries/1/, and the session is told so first."""
    from dycraft.run_layout import NOTES
    (run / "ir.json").write_text("{}")
    _meta_edit(run, outcome="committed")
    assert R.main(["--resume", str(run), "--prompt", "Make the legs thicker."]) == 0
    assert (run / "prior_deliveries" / "1" / "ir.json").exists() and not (run / "ir.json").exists()
    note = NOTES["object"].format(outcome="committed", prior="prior_deliveries/1", layer="object")
    assert launched["plan"].footer_text == f"{note}\n\nMake the legs thicker."


def test_a_resume_prompt_that_is_a_file_is_staged_for_the_agent(run, launched, tmp_path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.7\n\x00notes")
    assert R.main(["--resume", str(run), "--prompt", str(tmp_path / "notes.pdf")]) == 0
    assert (run / "inputs" / "notes.pdf").exists()
    assert launched["plan"].footer_text.endswith(
        "The operator adds:\n\nYour brief is the file inputs/notes.pdf (PDF, 15 bytes), in your run directory. Read it "
        "before you start; it is the whole task.")
    assert json.loads((run / "meta.json").read_text())["resumes"][0]["staged"] == "inputs/notes.pdf"


def test_a_delivered_object_run_in_the_src_layout_is_revised_from_its_folder(run, launched, tmp_path):
    """A run launched today keeps its records in `src/` and its delivery at the top (`run_layout`). Resumed by its
    folder, it is reopened inside `src/`, its top level loses the delivery that was set aside, and the session runs in
    `src/` under the run's own id."""
    from dycraft.run_layout import publish
    top = tmp_path / "runs" / "r1"
    inner = top / "src"
    run.rename(tmp_path / "flat")
    inner.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "flat").rename(inner)
    account = tmp_path / "account" / "projects"
    (account / project_key(run)).rename(account / project_key(inner))
    (inner / "ir.json").write_text(json.dumps({"id": "o1", "parts": []}))
    (inner / "final_render" / "assets" / "meshes").mkdir(parents=True)
    (inner / "final_render" / "0_front.png").write_bytes(b"png")
    (inner / "final_render" / "ir.urdf").write_text('<robot><mesh filename="assets/meshes/a.obj"/></robot>')
    (inner / "final_render" / "assets" / "meshes" / "a.obj").write_text("v 0 0 0\n")
    _meta_edit(inner, outcome="committed")
    publish(inner)
    assert (top / "object.urdf").is_file() and (top / "views" / "0_front.png").is_file()

    assert R.main(["--resume", str(top), "--prompt", "Make the legs thicker."]) == 0
    assert (inner / "prior_deliveries" / "1" / "ir.json").exists() and not (inner / "ir.json").exists()
    assert launched["plan"].run_dir == inner
    assert (launched["plan"].runs_dir, launched["plan"].run_id) == (top.parent, "r1")
    for gone in ("object.urdf", "views", "meshes"):
        assert not (top / gone).exists(), f"{gone} stayed at the top after its delivery was set aside"
    assert "**Committed.**" not in (top / "README.md").read_text()


def test_a_resume_takes_the_opt_in_again_and_records_what_ran_without_a_sandbox(run, launched, monkeypatch):
    """`--allow-unsandboxed` is not carried over from the launch: a resume names it, and its session's record says
    what ran without a sandbox, as a launch's does."""
    from dycraft.cli.backends import ClaudeBackend, allow_unsandboxed
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "0")        # put back afterwards

    def preflight(self, run_dir, layer=None):
        if allow_unsandboxed():
            self.launch_meta = {"unsandboxed": {"agent": "Claude Code", "reason": "bwrap: no"}}
    monkeypatch.setattr(ClaudeBackend, "preflight", preflight)
    assert R.main(["--resume", str(run), "--allow-unsandboxed"]) == 0
    meta = json.loads((run / "meta.json").read_text())
    assert meta["unsandboxed"] == meta["resumes"][0]["unsandboxed"] == {"agent": "Claude Code", "reason": "bwrap: no"}
