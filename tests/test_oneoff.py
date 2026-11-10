"""Tests for the one-off CLI sub-agent primitive and the `plain` layer.

Covers (all fast — no real claude/codex, subprocess is faked):

1. The `plain` layer is registered + composes + stages a harness client with
   the single `submit_result` stub.
2. `PlainSession.submit_result` persists result.json, marks committed, drops
   the `.terminate` sentinel, and traces an op_call.
3. `run_oneoff` happy path / no-result / timeout / spawn-error, with the
   subprocess faked to simulate a plain sub-CLI.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


from dycraft.agent import stage_harness_client
from dycraft.agent.agent_runner import compose_footer, compose_system_prompt
from dycraft.agent import oneoff
from dycraft.agent.oneoff import run_oneoff
from dycraft.agent.plain.session import PlainContext, PlainSession


# ---------------------------------------------------------------------------
# 1. registration / composition
# ---------------------------------------------------------------------------


def test_plain_prompt_and_stage(tmp_path):
    sp = compose_system_prompt("plain")
    assert "submit_result" in sp
    # The minimal plain footer carries only the task; the submit_result
    # contract lives in the system prompt, not the footer.
    ft = compose_footer("plain", "what is 2+2?", tmp_path)
    assert "what is 2+2?" in ft
    assert "VLM" not in ft and "MAX_STEPS" not in ft  # no DyCraft boilerplate
    staged = stage_harness_client(tmp_path, "plain")
    text = staged.read_text()
    assert "def submit_result" in text
    # plain must NOT carry the SDK ops
    assert "def compile" not in text
    assert "def finish" not in text


# ---------------------------------------------------------------------------
# 2. PlainSession.submit_result
# ---------------------------------------------------------------------------


def test_submit_result_persists_and_terminates(tmp_path):
    ctx = PlainContext(run_dir=tmp_path)
    sess = PlainSession(ctx)
    assert set(sess.handlers) == {"submit_result"}

    out = sess.handlers["submit_result"](text="the answer is 4")
    assert out == {"ok": True, "result_chars": len("the answer is 4")}
    assert ctx.committed is True
    assert json.loads((tmp_path / "result.json").read_text())["text"] == \
        "the answer is 4"
    assert (tmp_path / ".terminate").exists()
    # the op went through the @op wrapper → an op_call event was traced
    trace = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert sum('"op_call"' in line for line in trace) == 1


def test_submit_result_rejects_non_string(tmp_path):
    ctx = PlainContext(run_dir=tmp_path)
    sess = PlainSession(ctx)
    out = sess.handlers["submit_result"](text=123)
    assert out["ok"] is False
    assert "string" in out["error"]
    assert ctx.committed is False
    assert not (tmp_path / "result.json").exists()


# ---------------------------------------------------------------------------
# 3. run_oneoff with a faked subprocess
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stands in for the launched `python -m dycraft.cli --layer plain` proc.

    `behavior`:
      - "ok"        → writes an op_call trace + result.json, exits 0
      - "no_result" → writes an op_call trace, but no result.json, exits 0
      - "wedged"    → writes neither (0 op_calls), exits 0
      - "crash"     → writes neither (0 op_calls), exits rc 1 (failed launch)
      - "timeout"   → first communicate() raises TimeoutExpired
    """

    def __init__(self, argv, behavior, **_kw):
        self.argv = argv
        self.behavior = behavior
        self.returncode = None
        self.terminated = False
        self.killed = False
        self._calls = 0
        a = argv
        runs_dir = Path(a[a.index("--runs-dir") + 1])
        run_id = a[a.index("--run-id") + 1]
        self.sub_run_dir = runs_dir / run_id

    def communicate(self, timeout=None):
        self._calls += 1
        self.sub_run_dir.mkdir(parents=True, exist_ok=True)
        if self.behavior == "timeout" and self._calls == 1:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.behavior in ("ok", "no_result"):
            (self.sub_run_dir / "trace.jsonl").write_text(
                '{"ev": "op_call", "op": "submit_result", "step": 1}\n')
        if self.behavior == "ok":
            (self.sub_run_dir / "result.json").write_text(
                json.dumps({"text": "hello world"}))
        if self.terminated:
            self.returncode = -15
        elif self.behavior == "crash":
            self.returncode = 1
        else:
            self.returncode = 0
        return ("stdout", "stderr-tail")

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True


def _patch_popen(monkeypatch, behavior):
    monkeypatch.setattr(
        oneoff.subprocess, "Popen",
        lambda argv, **kw: _FakePopen(argv, behavior, **kw))


def test_run_oneoff_happy_path(tmp_path, monkeypatch):
    _patch_popen(monkeypatch, "ok")
    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path,
                     sub_run_name="oneoff", timeout_s=5.0)
    assert res["ok"] is True
    assert res["text"] == "hello world"
    assert res["n_op_calls"] == 1
    assert res["outcome"] == "completed"
    assert res["timed_out"] is False
    assert res["sub_run"] == "sub_runs/oneoff"


def test_run_oneoff_hands_the_prompt_over_as_a_file(tmp_path, monkeypatch):
    """Linux refuses to start a program any one of whose arguments is over
    128 KiB. A review prompt embeds the plan and its evidence and can be that
    large, so none is passed inline."""
    seen = []
    monkeypatch.setattr(oneoff.subprocess, "Popen",
                        lambda argv, **kw: seen.append(argv) or _FakePopen(argv, "ok", **kw))
    prompt = "Review this plan.\n" + "x" * (200 * 1024)
    res = run_oneoff(prompt=prompt, backend="claude", run_dir=tmp_path,
                     sub_run_name="review", timeout_s=5.0)
    assert res["ok"]
    argv = seen[0]
    path = Path(argv[argv.index("--prompt") + 1])
    # `.md`, the suffix the CLI reads a prompt file by.
    assert path.suffix == ".md" and path.read_text() == prompt
    assert max(len(a.encode()) for a in argv) < 4096


def _as_the_cli_reads_it(argv):
    """`--prompt` resolved by `cli.py`'s rule: an existing `.md` file is read."""
    value = argv[argv.index("--prompt") + 1]
    path = Path(value)
    return path.read_text().strip() if value.endswith(".md") and path.is_file() else value


def test_a_prompt_file_reaches_the_sub_agent_as_its_contents(tmp_path, monkeypatch):
    """The scene and motion rewriters pass the path of their prompt file.
    Written into a second file, the path itself became the sub-agent's task."""
    seen = []
    monkeypatch.setattr(oneoff.subprocess, "Popen",
                        lambda argv, **kw: seen.append(argv) or _FakePopen(argv, "ok", **kw))
    prompt_file = tmp_path / "scene_rewrite_prompt.md"
    prompt_file.write_text("Rewrite this brief into a scene entry.")
    run_oneoff(prompt=str(prompt_file), backend="codex", run_dir=tmp_path,
               sub_run_name="scene_rewrite", timeout_s=5.0)
    run_oneoff(prompt="Review these frames.", backend="codex", run_dir=tmp_path,
               sub_run_name="review", timeout_s=5.0)
    assert [_as_the_cli_reads_it(argv) for argv in seen] == [
        "Rewrite this brief into a scene entry.", "Review these frames."]


def test_run_oneoff_no_result(tmp_path, monkeypatch):
    _patch_popen(monkeypatch, "no_result")
    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path,
                     timeout_s=5.0)
    assert res["ok"] is False
    assert res["text"] is None
    assert res["n_op_calls"] == 1          # it DID reach the harness
    assert res["outcome"] == "no_result"


def test_run_oneoff_codex_no_stdout_fallback(tmp_path, monkeypatch):
    # codex one-shot has no result.json AND its PTY stdout is unusable render
    # noise — so there is NO stdout fallback; it stays no_result (not the fake's
    # "stdout"). Guards against returning ANSI garbage as an answer.
    _patch_popen(monkeypatch, "wedged")  # 0 op_calls, no result.json, rc 0
    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path,
                     timeout_s=5.0)
    assert res["ok"] is False
    assert res["text"] is None
    assert res["n_op_calls"] == 0          # never reached the harness
    assert res["outcome"] == "no_result"


def test_run_oneoff_claude_stdout_fallback(tmp_path, monkeypatch):
    # claude one-shot: answers in one turn and exits without calling
    # submit_result (no result.json), but its stdout carries the bare answer.
    # run_oneoff falls back to captured stdout → a usable, completed result.
    _patch_popen(monkeypatch, "wedged")  # 0 op_calls, no result.json, rc 0
    res = run_oneoff(prompt="hi", backend="claude", run_dir=tmp_path,
                     timeout_s=5.0)
    assert res["text"] == "stdout"         # the fake's captured stdout
    assert res["n_op_calls"] == 0          # never reached the harness
    assert res["outcome"] == "completed"
    assert res["ok"] is True


def test_run_oneoff_prefers_result_json_over_stdout(tmp_path, monkeypatch):
    # When submit_result DID run, result.json wins over stdout even for claude.
    _patch_popen(monkeypatch, "ok")        # result.json={"text":"hello world"}
    res = run_oneoff(prompt="hi", backend="claude", run_dir=tmp_path,
                     timeout_s=5.0)
    assert res["text"] == "hello world"    # not the fake's "stdout"


def test_run_oneoff_no_stdout_fallback_on_error_exit(tmp_path, monkeypatch):
    # claude exits non-zero (e.g. rate-limited "session limit" notice, auth
    # error) printing to stdout. That is NOT an answer — only a clean rc-0 exit's
    # stdout is trusted, so this stays no_result rather than a false success.
    _patch_popen(monkeypatch, "crash")     # rc 1, 0 op_calls, stdout="stdout"
    res = run_oneoff(prompt="hi", backend="claude", run_dir=tmp_path,
                     timeout_s=5.0)
    assert res["text"] is None
    assert res["ok"] is False
    assert res["outcome"] == "no_result"


def test_run_oneoff_timeout(tmp_path, monkeypatch):
    captured = {}

    def _factory(argv, **kw):
        p = _FakePopen(argv, "timeout", **kw)
        captured["proc"] = p
        return p
    monkeypatch.setattr(oneoff.subprocess, "Popen", _factory)

    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path,
                     sub_run_name="oneoff", timeout_s=0.01)
    assert res["timed_out"] is True
    assert res["outcome"] == "timeout"
    assert res["ok"] is False
    assert captured["proc"].terminated is True
    # timeout path drops the .terminate sentinel to stop the PTY child
    assert (tmp_path / "sub_runs" / "oneoff" / ".terminate").exists()


def test_run_oneoff_spawn_error(tmp_path, monkeypatch):
    def _boom(argv, **kw):
        raise OSError("no such executable")
    monkeypatch.setattr(oneoff.subprocess, "Popen", _boom)
    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path)
    assert res["ok"] is False
    assert res["outcome"] == "spawn_error"
    assert "no such executable" in res["stderr_tail"]




def test_a_sub_run_keeps_its_launchers_stderr_in_its_folder(tmp_path, monkeypatch):
    """The caller reports only a FATAL line and a tail; the whole log stays beside the sub-run's others."""
    _patch_popen(monkeypatch, "crash")
    res = run_oneoff(prompt="hi", backend="codex", run_dir=tmp_path, sub_run_name="scene_rewrite", timeout_s=5.0)
    assert (tmp_path / res["sub_run"] / oneoff.LAUNCHER_LOG).read_text() == "stderr-tail"
