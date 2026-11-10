"""Viewer: transcript capture, trajectory normalization, cost, and the API.

Everything here builds a synthetic run bundle in `tmp_path`, so the suite stays
hermetic and fast — no real run dir, no external CLI config dir, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dycraft.agent import transcript_capture as tc
from dycraft.animation.blender.bundle import file_hash, write_json
from dycraft.viewer import cost as cost_mod
from dycraft.viewer import runs as runs_mod
from dycraft.viewer import scene as scene_mod
from dycraft.viewer.trajectory import build as build_trajectory

fastapi = pytest.importorskip("fastapi", reason="viewer needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dycraft.viewer.app import create_app  # noqa: E402


def _priced_total(usage, model):
    """USD for one turn's usage, or None when the model is unpriced."""
    priced = cost_mod.price_usage(usage, model)
    return None if priced is None else priced["total"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _ir() -> dict:
    return {
        "id": "abc123",
        "name": "widget",
        "root": "body",
        "schema": "object/v1",
        "parts": [
            {
                "name": "body",
                "primitive": "solid",
                "params": {
                    "mesh": "assets/meshes/deadbeef.obj",
                    "brep": "assets/brep/deadbeef.brep",
                    "source": "assets/src/cafe.py",
                    "digest": "deadbeef",
                    "aabb_min": [-0.1, -0.1, 0.0],
                    "aabb_max": [0.1, 0.1, 0.2],
                    "shape": {
                        "volume": 0.001,
                        "fill_ratio": 0.125,
                        "center": [0.0, 0.0, 0.1],
                        "inertia_diag": [1e-5, 1e-5, 1e-5],
                        "face_census": {"PLANE": 6, "CYLINDER": 1},
                    },
                },
                "material": {"color": [205, 205, 210], "alpha": 1.0,
                             "metallic": 0.9, "roughness": 0.32},
            },
            {
                "name": "lid",
                "primitive": "box",
                "params": {"width": 0.1, "length": 0.1, "height": 0.01},
                "material": {"color": [90, 90, 95], "alpha": 1.0},
            },
        ],
        "joints": [
            {
                "name": "hinge",
                "type": "revolute",
                "parent": "body",
                "child": "lid",
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 1.57],
                "default": 0.0,
                "id": 1,
                # Anchors are resolved to SE3 by the time they reach the IR:
                # quaternion is (w, x, y, z), translation in meters.
                "parent_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                                  "translation": [0.0, 0.0, 0.2]},
                "child_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                                 "translation": [0.0, 0.0, -0.005]},
            }
        ],
        "collision_allowances": [
            {"a": "body", "b": "lid", "reason": "hinge lug is embedded"}
        ],
        "assertions": [],
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A minimal but structurally real object-run bundle."""
    runs = tmp_path / "runs"
    run = runs / "260101_000000_claude_widget"
    (run / "assets" / "meshes").mkdir(parents=True)
    (run / "assets" / "src").mkdir(parents=True)
    (run / "compile_reports").mkdir()
    (run / "renders").mkdir()

    (run / "meta.json").write_text(json.dumps({
        "run_id": run.name, "layer": "object", "model": "test-model",
        "runtime": "claude_cli", "outcome": "committed", "prompt": "A widget.",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:05:00+00:00",
    }))
    (run / "ir.json").write_text(json.dumps(_ir()))
    (run / "object.py").write_text("obj = 1\n")
    (run / "assets" / "src" / "cafe.py").write_text("# build123d recipe\n")
    # A one-triangle OBJ is enough to prove the file route serves geometry.
    (run / "assets" / "meshes" / "deadbeef.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (run / "renders" / "0_front.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run / "compile_reports" / "4.json").write_text(json.dumps({
        "ok": True, "ir_hash": "abc123", "build_error": None,
        "sandbox_violation": None,
        "verifier_reports": [{
            "name": "NoSelfPenetration", "verdict": "pass",
            "evidence": {"exact_adjudication": {
                "available": True, "considered": 3, "sent": 3, "cleared": 3,
                "confirmed": 0, "unknown": 0}},
            "suggested_repairs": [],
        }],
    }))
    # latest.json duplicates the highest-numbered report and must not be listed.
    (run / "compile_reports" / "latest.json").write_text(
        (run / "compile_reports" / "4.json").read_text())
    (run / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"ev": "op_call", "op": "author_solids", "step": 1,
         "ts": "2026-01-01T00:01:00+00:00"},
        {"ev": "op_call", "op": "compile", "step": 2,
         "ts": "2026-01-01T00:02:00+00:00"},
    ]) + "\n")
    return run


def _claude_transcript(with_thinking_text: bool) -> str:
    """Two logical turns, each streamed as several records sharing message.id."""
    thinking = "I should inspect the run dir." if with_thinking_text else ""
    usage = {"input_tokens": 4, "cache_read_input_tokens": 100,
             "cache_creation_input_tokens": 20, "output_tokens": 50}
    rows = [
        {"type": "assistant", "timestamp": "2026-01-01T00:00:30+00:00",
         "effort": "high",
         "message": {"id": "msg_1", "model": "test-model", "usage": usage,
                     "stop_reason": "tool_use",
                     "content": [{"type": "thinking", "thinking": thinking,
                                  "signature": "sig"}]}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:31+00:00",
         "message": {"id": "msg_1", "model": "test-model", "usage": usage,
                     "content": [{"type": "text", "text": "Checking state."}]}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:32+00:00",
         "message": {"id": "msg_1", "model": "test-model", "usage": usage,
                     "content": [{"type": "tool_use", "id": "tu_1",
                                  "name": "Bash",
                                  "input": {"command": "ls -la"}}]}},
        {"type": "user", "timestamp": "2026-01-01T00:00:33+00:00",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_1",
              "content": "total 0", "is_error": False}]}},
        {"type": "assistant", "timestamp": "2026-01-01T00:01:00+00:00",
         "message": {"id": "msg_2", "model": "test-model",
                     "usage": {"input_tokens": 4,
                               "cache_read_input_tokens": 200,
                               "cache_creation_input_tokens": 0,
                               "output_tokens": 10},
                     "content": [
                         {"type": "tool_use", "id": "tu_2", "name": "Bash",
                          "input": {"command": 'python -c "import harness as h; '
                                               'h.author_solids(code=x)"'}},
                         # A Write whose *content* mentions the harness must not
                         # be mistaken for a harness call.
                         {"type": "tool_use", "id": "tu_3", "name": "Write",
                          "input": {"file_path": "notes.md",
                                    "content": "we call h.compile() later"}},
                     ]}},
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


# ---------------------------------------------------------------------------
# transcript capture
# ---------------------------------------------------------------------------


def test_project_key_munges_slashes_underscores_and_dots():
    key = tc.project_key(Path("/a/b_c/d.e/run_1"))
    assert key == "-a-b-c-d-e-run-1"


def test_capture_copies_newest_and_reports_status(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "260101_000000_x"
    run.mkdir(parents=True)
    config = tmp_path / "cfg"
    project = config / "projects" / tc.project_key(run)
    project.mkdir(parents=True)
    (project / "old.jsonl").write_text('{"a": 1}\n')
    newest = project / "new.jsonl"
    newest.write_text('{"a": 2}\n')
    import os
    os.utime(project / "old.jsonl", (1, 1))

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    assert tc.capture(run) == "copied"
    staged = run / tc.TRANSCRIPT_NAME
    assert staged.read_text() == '{"a": 2}\n'
    # The staged copy wins over the external one on read.
    assert tc.resolve_transcript(run) == staged
    # No leftover temp file from the write-then-replace.
    assert not list(run.glob("*.tmp"))


def test_stale_session_under_another_account_loses_to_the_newer_one(
        tmp_path, monkeypatch):
    """Fallback-list order must not outrank recency.

    Relaunching a failed run into the same run dir under a second account
    leaves two sessions keyed to that dir. First-dir-wins staged the stale one:
    a `Not logged in` stub from the aborted attempt beat the real transcript
    written 44 minutes later, and the viewer rendered the run as a single
    synthetic turn.
    """
    import os

    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    key = tc.project_key(run)

    aborted = tmp_path / "cfg-aborted" / "projects" / key
    aborted.mkdir(parents=True)
    (aborted / "stub.jsonl").write_text('{"note": "Not logged in"}\n')
    os.utime(aborted / "stub.jsonl", (1, 1))

    real = tmp_path / "cfg-real" / "projects" / key
    real.mkdir(parents=True)
    (real / "session.jsonl").write_text('{"note": "the actual run"}\n')

    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    # The aborted account is listed FIRST, so list order alone would pick it.
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS",
                        (str(tmp_path / "cfg-aborted"),
                         str(tmp_path / "cfg-real")))

    assert tc.find_claude_transcript(run) == real / "session.jsonl"
    assert tc.capture(run) == "copied"
    assert "actual run" in (run / tc.TRANSCRIPT_NAME).read_text()


def test_recorded_account_is_never_substituted_by_another(tmp_path, monkeypatch):
    """A known account with no session yet must not fall back to a leftover.

    The CLI can still be flushing its session file when the daemon shuts down.
    "Not written yet" has to read as unavailable, not as another account's stale
    session for the same run dir — that substitution is what staged a
    `Not logged in` stub over a real 964 KB transcript.
    """
    import json

    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    key = tc.project_key(run)
    mine = tmp_path / "cfg-mine"
    (mine / "projects" / key).mkdir(parents=True)      # present but empty

    leftover = tmp_path / "cfg-leftover" / "projects" / key
    leftover.mkdir(parents=True)
    (leftover / "stub.jsonl").write_text('{"note": "Not logged in"}\n')

    (run / "meta.json").write_text(json.dumps({"claude_config_dir": str(mine)}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(leftover.parent.parent))
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS",
                        (str(tmp_path / "cfg-leftover"),))

    assert tc.find_claude_transcript(run) is None
    assert tc.capture(run) == "unavailable"
    assert not (run / tc.TRANSCRIPT_NAME).exists()

    # Once our own session lands, it is found — meta outranks the env var.
    (mine / "projects" / key / "real.jsonl").write_text('{"note": "real"}\n')
    assert tc.find_claude_transcript(run) == mine / "projects" / key / "real.jsonl"


def test_configured_config_dir_outranks_a_newer_fallback(tmp_path, monkeypatch):
    """$CLAUDE_CONFIG_DIR names the account the run actually drove, so it wins
    even when another account holds a newer file for the same run dir."""
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    key = tc.project_key(run)

    configured = tmp_path / "cfg-configured" / "projects" / key
    configured.mkdir(parents=True)
    (configured / "mine.jsonl").write_text('{"note": "mine"}\n')

    other = tmp_path / "cfg-other" / "projects" / key
    other.mkdir(parents=True)
    (other / "newer.jsonl").write_text('{"note": "newer but not this run"}\n')
    import os
    os.utime(configured / "mine.jsonl", (1, 1))  # deliberately the older file

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg-configured"))
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS",
                        (str(tmp_path / "cfg-other"),))

    assert tc.find_claude_transcript(run) == configured / "mine.jsonl"


def test_fallback_globs_every_account_dir_and_skips_the_configured_one(
        tmp_path, monkeypatch):
    """The default guess list is a glob, so any account dir of the usual shape
    is searched without naming it in code: files that match are skipped, and
    the configured dir is not listed twice."""
    for name in (".claude", ".claude2", ".claude-work"):
        (tmp_path / name).mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    (tmp_path / ".codex").mkdir()
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS", (str(tmp_path / ".claude*"),))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude2"))

    assert tc._fallback_config_dirs() == [tmp_path / ".claude",
                                          tmp_path / ".claude-work"]
    assert tc._candidate_config_dirs()[0] == tmp_path / ".claude2"


def test_capture_unavailable_without_external_transcript(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS", ())
    assert tc.capture(run) == "unavailable"
    assert tc.resolve_transcript(run) is None


# ---------------------------------------------------------------------------
# transcript capture: codex
# ---------------------------------------------------------------------------


def _js_cell(command: str) -> str:
    """A 0.153.4 `exec` cell: JS source, not JSON.

    The command lives inside `tools.exec_command({cmd:"…"})` and that is the
    only place the parser looks for it — see `_codex_tool_command`. A
    `CommandExecution` event carries the argv the shell actually spawned too,
    but nothing reads those, so a fixture that put the command only there
    would be testing a path the viewer does not have.
    """
    return ('const res = await tools.exec_command({cmd:"'
            + command.replace('"', '\\"') + '", yield_time_ms: 1000});')


def _codex_rollout(cwd: Path, *, parent: str | None = None,
                   extra: list[dict] | None = None) -> str:
    """A minimal rollout: one response that runs one shell and views a render."""
    meta = {"type": "session_meta", "timestamp": "2026-09-08T17:26:13.229Z",
            "payload": {"cwd": str(cwd), "session_id": "s1"}}
    if parent:
        meta["payload"]["parent_thread_id"] = parent
    records: list[dict] = [
        meta,
        {"type": "turn_context", "timestamp": "2026-09-08T17:26:14.000Z",
         "payload": {"model": "gpt-6-astra", "effort": "xhigh"}},
        {"type": "response_item", "timestamp": "2026-09-08T17:26:15.000Z",
         "payload": {"type": "reasoning", "summary": [],
                     "encrypted_content": "gAAA"}},
        {"type": "response_item", "timestamp": "2026-09-08T17:26:16.000Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Building."}]}},
        {"type": "response_item", "timestamp": "2026-09-08T17:26:17.000Z",
         "payload": {"type": "custom_tool_call", "name": "exec",
                     "call_id": "c1", "input": _js_cell(
                         'python -c \'import harness as h; h.compile()\' '
                         '&& python -c \'import harness as h; h.snapshot()\'')}},
        {"type": "event_msg", "timestamp": "2026-09-08T17:26:19.000Z",
         "payload": {"type": "item_completed", "item": {
             "type": "CommandExecution", "exit_code": 0,
             "command": ["/bin/bash", "-lc",
                         'python -c \'h.compile()\' && python -c \'h.snapshot()\''],
             "duration": {"secs": 2, "nanos": 0},
             "aggregated_output": "ok"}}},
        {"type": "event_msg", "timestamp": "2026-09-08T17:26:19.500Z",
         "payload": {"type": "item_completed", "item": {
             "type": "ImageView", "path": "file:///run/snapshots/1_front.png"}}},
        {"type": "token_usage_record", "timestamp": "2026-09-08T17:26:20.000Z",
         "payload": {"usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                               "cache_write_input_tokens": 0,
                               "output_tokens": 50}}},
        {"type": "response_item", "timestamp": "2026-09-08T17:26:21.000Z",
         "payload": {"type": "custom_tool_call_output", "call_id": "c1",
                     "output": [{"type": "input_text", "text": "compiled"}]}},
    ]
    records.extend(extra or [])
    return "".join(json.dumps(r) + "\n" for r in records)


def _codex_run(tmp_path: Path, **meta) -> Path:
    run = tmp_path / "runs" / "helmet_gpt"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps(
        {"runtime": "codex1_cli", "model": "gpt-6-astra", **meta}))
    return run


def _codex_home(tmp_path: Path, run: Path, name: str = "sess.jsonl",
                **kwargs) -> Path:
    home = tmp_path / "codexhome"
    day = home / "sessions" / "2026" / "09" / "08"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-{name}"
    path.write_text(_codex_rollout(run, **kwargs))
    return home


def test_a_codex_run_is_recognised_from_its_runtime(tmp_path):
    """A bundle whose meta.json records only `runtime: codex1_cli`, with no
    `codex_home`, still resolves as a codex run."""
    assert tc.is_codex_run(_codex_run(tmp_path))
    plain = tmp_path / "claude_run"
    plain.mkdir()
    (plain / "meta.json").write_text('{"runtime": "claude_cli"}')
    assert not tc.is_codex_run(plain)


def test_codex_rollout_is_found_by_the_cwd_it_records(tmp_path, monkeypatch):
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    home = _codex_home(tmp_path, run)
    other = home / "sessions" / "2026" / "09" / "08" / "rollout-other.jsonl"
    other.write_text(_codex_rollout(tmp_path / "somewhere_else"))
    assert tc.find_codex_transcript(run) == home / "sessions/2026/09/08/rollout-sess.jsonl"


def test_a_reviewer_sub_thread_never_wins_over_its_parent(tmp_path):
    """Sub-threads share the run's cwd; staging one reports a fraction of it."""
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    home = _codex_home(tmp_path, run, name="sub.jsonl", parent="t0")
    assert tc.find_codex_transcript(run) is None
    (home / "sessions/2026/09/08/rollout-real.jsonl").write_text(
        _codex_rollout(run))
    assert tc.find_codex_transcript(run).name == "rollout-real.jsonl"


def test_capture_strips_inline_images_from_a_codex_rollout(tmp_path):
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    home = _codex_home(tmp_path, run)
    rollout = home / "sessions/2026/09/08/rollout-sess.jsonl"
    blob = "iVBORw0KGgo" + "A" * 4000
    rollout.write_text(rollout.read_text() + json.dumps(
        {"type": "response_item", "timestamp": "2026-09-08T17:26:22.000Z",
         "payload": {"type": "custom_tool_call_output", "call_id": "c2",
                     "output": [{"type": "input_image",
                                 "image_url": f"data:image/png;base64,{blob}"}]}}) + "\n")

    assert tc.capture(run) == "copied"
    staged = (run / tc.TRANSCRIPT_NAME).read_text()
    assert blob not in staged
    assert "inline image stripped by dycraft" in staged
    assert len(staged) < len(rollout.read_text())
    # Every other record survives intact and the file still parses.
    assert all(json.loads(line) for line in staged.splitlines())
    assert "gpt-6-astra" in staged


def test_a_short_data_uri_in_prose_is_left_alone(tmp_path):
    """The length floor keeps the stripper off text that merely mentions one."""
    data, count = tc._strip_inline_images(
        b'{"text": "data:image/png;base64,iVBORw0KGgo="}')
    assert count == 0
    assert b"iVBORw0KGgo=" in data


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------


def test_a_codex_rollout_renders_as_turns(tmp_path):
    """A codex run is one codex turn end to end; the panel wants responses."""
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    _codex_home(tmp_path, run)
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"ev": "op_call", "op": "compile", "step": 1},
        {"ev": "op_call", "op": "snapshot", "step": 2},
    ]))
    tc.capture(run)

    built = build_trajectory(run)
    assert built["source"] == "codex_cli"
    assert built["model"] == "gpt-6-astra"
    assert len(built["turns"]) == 1
    turn = built["turns"][0]
    assert turn["text"] == "Building."
    assert turn["effort"] == "xhigh"
    # Codex reports gross input; the uncached remainder is what `input` means.
    assert turn["usage"]["input"] == 200
    assert turn["usage"]["cache_read"] == 800
    assert turn["usage"]["gross_input"] == 1000
    assert turn["usage"]["output"] == 50

    (call,) = turn["tool_calls"]
    assert call["name"] == "exec"
    # The shell's argv, not the JavaScript wrapper codex takes it in.
    assert call["input"]["command"].startswith("python -c")
    # Call record -> output record: 17:26:17.000 to 17:26:21.000. NOT the
    # 2 s the rollout's `CommandExecution` item reports for the shell itself.
    # Those items are not read, so a backgrounded shell's own
    # runtime, its exit code, and the `ImageView` paths it showed the model are
    # all unavailable here — see the note below `test_codex_reasoning_...`.
    assert call["duration_ms"] == 4000.0
    assert call["started_at"] == "2026-09-08T17:26:17.000Z"
    assert call["finished_at"] == "2026-09-08T17:26:21.000Z"
    # The tool output, which is all the result is now: the snapshot path used
    # to be folded in from the `ImageView` item beside it.
    assert call["result"] == "compiled"
    assert call["dycraft_op"] == "compile"
    assert call["dycraft_step"] == 1


def test_every_op_in_one_codex_shell_claims_its_step(tmp_path):
    """A shell that ran `compile` then `snapshot` must consume both steps.

    Claiming only the first would leave the rest of `trace.jsonl` unclaimed, so
    every later shell would greedily match a stale step and the joined sequence
    would go backwards. Agents that batch commands make this the common case.
    """
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    second = [
        {"type": "response_item", "timestamp": "2026-09-08T17:27:00.000Z",
         "payload": {"type": "custom_tool_call", "name": "exec",
                     "call_id": "c2", "input": _js_cell(
                         "python -c 'import harness as h; h.snapshot()'")}},
        {"type": "event_msg", "timestamp": "2026-09-08T17:27:02.000Z",
         "payload": {"type": "item_completed", "item": {
             "type": "CommandExecution", "exit_code": 0,
             "command": ["/bin/bash", "-lc", "python -c 'h.snapshot()'"],
             "duration": {"secs": 1, "nanos": 0},
             "aggregated_output": "ok"}}},
        {"type": "token_usage_record", "timestamp": "2026-09-08T17:27:03.000Z",
         "payload": {"usage": {"input_tokens": 10, "cached_input_tokens": 0,
                               "output_tokens": 1}}},
    ]
    _codex_home(tmp_path, run, extra=second)
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"ev": "op_call", "op": "compile", "step": 1},
        {"ev": "op_call", "op": "snapshot", "step": 2},
        {"ev": "op_call", "op": "snapshot", "step": 3},
    ]))
    tc.capture(run)

    turns = build_trajectory(run)["turns"]
    first, later = turns[0]["tool_calls"][0], turns[1]["tool_calls"][0]
    # The first shell ran two ops and reports the first of them.
    assert (first["dycraft_op"], first["dycraft_step"]) == ("compile", 1)
    # Its `snapshot` consumed step 2, so the next shell's lands on 3.
    assert (later["dycraft_op"], later["dycraft_step"]) == ("snapshot", 3)


def test_a_codex_call_never_claims_an_op_from_before_it_started(tmp_path):
    """Ops a script ran stay unclaimed; a later call must skip them.

    The first shell runs a script whose `h.compile()` the command text does
    not show, so its trace line is left over. Matching on the op name alone,
    the second shell's `h.compile()` would claim that stale step 1 instead of
    its own step 2.
    """
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    second = [
        {"type": "response_item", "timestamp": "2026-09-08T17:27:00.000Z",
         "payload": {"type": "custom_tool_call", "name": "exec",
                     "call_id": "c2", "input": _js_cell(
                         "python -c 'import harness as h; h.compile()'")}},
        {"type": "token_usage_record", "timestamp": "2026-09-08T17:27:03.000Z",
         "payload": {"usage": {"input_tokens": 10, "cached_input_tokens": 0,
                               "output_tokens": 1}}},
    ]
    _codex_home(tmp_path, run, extra=second)
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        # Before any tool call in the rollout: nothing may claim it.
        {"ev": "op_call", "op": "compile", "step": 1,
         "ts": "2026-09-08T17:20:00.000000+00:00"},
        {"ev": "op_call", "op": "compile", "step": 2,
         "ts": "2026-09-08T17:26:18.000000+00:00"},
        {"ev": "op_call", "op": "snapshot", "step": 3,
         "ts": "2026-09-08T17:26:19.000000+00:00"},
        {"ev": "op_call", "op": "compile", "step": 4,
         "ts": "2026-09-08T17:27:01.000000+00:00"},
    ]))
    tc.capture(run)

    turns = build_trajectory(run)["turns"]
    first, later = turns[0]["tool_calls"][0], turns[1]["tool_calls"][0]
    assert (first["dycraft_op"], first["dycraft_step"]) == ("compile", 2)
    assert (later["dycraft_op"], later["dycraft_step"]) == ("compile", 4)


def test_a_codex_call_never_claims_an_op_from_after_it_finished(tmp_path):
    """A shell that only writes a script naming an op ran no op.

    `cat > scratch/r6_finish.py <<PY ... h.finish()` reads as a `finish` call,
    and with no upper bound it claimed the real `finish` twenty minutes later,
    stealing it from the shell that ran the script.
    """
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    second = [
        {"type": "response_item", "timestamp": "2026-09-08T17:27:00.000Z",
         "payload": {"type": "custom_tool_call", "name": "exec",
                     "call_id": "c2", "input": _js_cell(
                         "cat > s.py <<'PY'\nimport harness as h\nh.finish()\nPY")}},
        {"type": "response_item", "timestamp": "2026-09-08T17:27:02.000Z",
         "payload": {"type": "custom_tool_call_output", "call_id": "c2",
                     "output": "Process exited with code 0"}},
        {"type": "token_usage_record", "timestamp": "2026-09-08T17:27:03.000Z",
         "payload": {"usage": {"input_tokens": 10, "cached_input_tokens": 0,
                               "output_tokens": 1}}},
    ]
    _codex_home(tmp_path, run, extra=second)
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"ev": "op_call", "op": "compile", "step": 1,
         "ts": "2026-09-08T17:26:18.000000+00:00"},
        {"ev": "op_call", "op": "snapshot", "step": 2,
         "ts": "2026-09-08T17:26:19.000000+00:00"},
        {"ev": "op_call", "op": "finish", "step": 3,
         "ts": "2026-09-08T17:47:00.000000+00:00"},
    ]))
    tc.capture(run)

    later = build_trajectory(run)["turns"][1]["tool_calls"][0]
    assert later["dycraft_op"] == "finish"
    assert later["dycraft_step"] is None


def test_codex_reasoning_is_reported_as_unrecoverable(tmp_path):
    run = _codex_run(tmp_path, codex_home=str(tmp_path / "codexhome"))
    _codex_home(tmp_path, run)
    tc.capture(run)
    built = build_trajectory(run)
    assert built["reasoning_available"] is False
    assert "encrypted" in built["reasoning_note"]


def test_streamed_records_collapse_into_logical_turns(run_dir):
    """One `message.id` is one turn, and its usage is counted once.

    The CLI emits a record per content block, all repeating the turn's usage.
    Treating each as a turn would more than double a run's turns and token
    totals.
    """
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(False))
    result = build_trajectory(run_dir)

    assert result["source"] == "claude_cli"
    assert len(result["turns"]) == 2
    first = result["turns"][0]
    assert first["usage"] == {
        "input": 4, "cache_read": 100, "cache_write": 20,
        # The fixture reports only the write total, no tier breakdown.
        "cache_write_5m": 0, "cache_write_1h": 0,
        "output": 50, "gross_input": 124, "total": 174,
    }
    assert first["text"] == "Checking state."
    assert result["totals"]["turns"] == 2
    assert result["totals"]["output"] == 60
    assert result["totals"]["cache_read"] == 300


def test_empty_thinking_reports_reasoning_unavailable(run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(False))
    result = build_trajectory(run_dir)
    assert result["reasoning_available"] is False
    assert "signature" in result["reasoning_note"]
    assert result["turns"][0]["thought_summary"] is None


def test_thinking_text_when_present(run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(True))
    result = build_trajectory(run_dir)
    assert result["reasoning_available"] is True
    assert result["reasoning_note"] is None
    assert result["turns"][0]["thought_summary"] == "I should inspect the run dir."


def test_harness_op_joined_to_step_only_for_shell_calls(run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(False))
    turns = build_trajectory(run_dir)["turns"]

    plain_bash = turns[0]["tool_calls"][0]
    assert plain_bash["dycraft_op"] is None      # `ls -la` is not a harness op

    calls = {c["name"]: c for c in turns[1]["tool_calls"]}
    assert calls["Bash"]["dycraft_op"] == "author_solids"
    assert calls["Bash"]["dycraft_step"] == 1
    # Prose mentioning `h.compile()` inside a Write payload is not a call.
    assert calls["Write"]["dycraft_op"] is None


def test_tool_results_are_attached(run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(False))
    turns = build_trajectory(run_dir)["turns"]
    assert turns[0]["tool_calls"][0]["result"] == "total 0"
    assert turns[0]["tool_calls"][0]["is_error"] is False


def test_no_trajectory_is_an_empty_envelope(run_dir, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(run_dir / "nope"))
    monkeypatch.setattr(tc, "_FALLBACK_CONFIG_DIRS", ())
    result = build_trajectory(run_dir)
    assert result["source"] is None
    assert result["turns"] == []
    assert result["totals"]["turns"] == 0


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def test_cost_is_none_without_a_price_table(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_PRICES_JSON", str(tmp_path / "absent.json"))
    usage = {"input": 1000, "cache_read": 0, "cache_write": 0, "output": 1000}
    assert _priced_total(usage, "any-model") is None


def test_cost_uses_longest_matching_prefix(tmp_path, monkeypatch):
    table = tmp_path / "prices.json"
    table.write_text(json.dumps({
        "claude": {"input": 1.0, "output": 1.0},
        "claude-opus-5": {"input": 10.0, "cache_read": 1.0,
                          "cache_write": 12.5, "output": 50.0},
    }))
    monkeypatch.setenv("DYCRAFT_PRICES_JSON", str(table))
    usage = {"input": 1_000_000, "cache_read": 1_000_000,
             "cache_write": 1_000_000, "output": 1_000_000}
    # `claude-opus-5[1m]` must resolve to the opus entry, not the family one.
    assert _priced_total(usage, "claude-opus-5[1m]") == 73.5
    assert _priced_total(usage, "claude-sonnet-5") == 2.0


def test_cost_table_reloads_when_the_file_changes(tmp_path, monkeypatch):
    table = tmp_path / "prices.json"
    table.write_text(json.dumps({"m": {"output": 1.0}}))
    monkeypatch.setenv("DYCRAFT_PRICES_JSON", str(table))
    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 1_000_000}
    assert _priced_total(usage, "m") == 1.0
    table.write_text(json.dumps({"m": {"output": 2.0}}))
    import os
    os.utime(table, (10**9, 10**9))
    assert _priced_total(usage, "m") == 2.0


# ---------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------


def test_list_runs_skips_dirs_without_meta(run_dir):
    runs = run_dir.parent
    (runs / "not_a_run").mkdir()
    (runs / "_media").mkdir()
    listed = runs_mod.list_runs(runs)
    assert [r["run_id"] for r in listed] == [run_dir.name]
    assert listed[0]["n_parts"] == 2 and listed[0]["n_joints"] == 1


def test_list_runs_surfaces_reasoning_effort_from_meta(run_dir):
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["reasoning_effort"] = "xhigh"
    meta_path.write_text(json.dumps(meta))

    summary = runs_mod.list_runs(run_dir.parent)[0]
    detail = runs_mod.run_detail(run_dir)
    assert summary["reasoning_effort"] == "xhigh"
    assert detail["reasoning_effort"] == "xhigh"


def test_reasoning_effort_falls_back_to_captured_transcript(run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(True))
    summary = runs_mod.list_runs(run_dir.parent)[0]
    detail = runs_mod.run_detail(run_dir)
    assert summary["reasoning_effort"] == "high"
    assert detail["reasoning_effort"] == "high"


def test_meta_reasoning_effort_wins_over_transcript(run_dir):
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["reasoning_effort"] = "low"
    meta_path.write_text(json.dumps(meta))
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(True))
    assert runs_mod.list_runs(run_dir.parent)[0]["reasoning_effort"] == "low"


def test_compile_reports_exclude_latest_and_surface_adjudication(run_dir):
    detail = runs_mod.run_detail(run_dir)
    assert [r["step"] for r in detail["compile_reports"]] == [4]
    assert detail["compile_reports"][0]["exact_adjudication"]["cleared"] == 3


def test_part_summary_carries_shape_facts_only_for_solids(run_dir):
    parts = {p["name"]: p for p in runs_mod.run_detail(run_dir)["ir"]["parts"]}
    assert parts["body"]["fill_ratio"] == 0.125
    assert parts["body"]["source"] == "assets/src/cafe.py"
    assert "fill_ratio" not in parts["lid"]


def test_resolve_file_refuses_escapes(run_dir):
    assert runs_mod.resolve_file(run_dir, "object.py") is not None
    assert runs_mod.resolve_file(run_dir, "../../etc/passwd") is None
    assert runs_mod.resolve_file(run_dir, "nope.py") is None


def test_resolve_run_refuses_traversal(run_dir):
    runs = run_dir.parent
    assert runs_mod.resolve_run(runs, run_dir.name) is not None
    assert runs_mod.resolve_run(runs, "../secrets") is None
    assert runs_mod.resolve_run(runs, "unknown") is None
    # `/` is now legal (nested runs) but must not become an escape hatch.
    assert runs_mod.resolve_run(runs, "a/../../secrets") is None
    assert runs_mod.resolve_run(runs, "/etc") is None
    assert runs_mod.resolve_run(runs, ".hidden/x") is None


def _nest(runs: Path, rel: str, template: Path) -> Path:
    """Copy the fixture bundle to `runs/<rel>`, so it is a real run."""
    import shutil
    dest = runs / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, dest)
    return dest


def test_list_runs_finds_runs_inside_folders(run_dir):
    """`runs/` is grouped by hand into folders; a run in one must still show."""
    runs = run_dir.parent
    _nest(runs, "achieve/260102_000000_claude_kettle", run_dir)
    ids = {r["run_id"] for r in runs_mod.list_runs(runs)}
    assert ids == {run_dir.name, "achieve/260102_000000_claude_kettle"}


def test_nested_run_id_is_its_path_and_splits_into_group_and_name(run_dir):
    """The path is the identity: two folders may hold the same meta run_id, and
    the path is what every other endpoint resolves."""
    runs = run_dir.parent
    _nest(runs, "achieve/deep/kettle", run_dir)
    by_id = {r["run_id"]: r for r in runs_mod.list_runs(runs)}
    nested = by_id["achieve/deep/kettle"]
    assert nested["group"] == "achieve/deep"
    assert nested["name"] == "260101_000000_claude_widget"   # meta.json's id
    assert runs_mod.resolve_run(runs, "achieve/deep/kettle") is not None
    # a flat run keeps the old shape exactly
    assert by_id[run_dir.name]["group"] == ""


def test_walk_stops_at_a_run_and_never_enters_its_artifacts(run_dir):
    """A run's own subtree must not be scanned: `assets/`, `renders/`, and the
    rest are internals, and a stray meta.json in one would list a phantom run.
    """
    runs = run_dir.parent
    (run_dir / "renders" / "meta.json").write_text(json.dumps({"run_id": "x"}))
    assert [r["run_id"] for r in runs_mod.list_runs(runs)] == [run_dir.name]


def test_symlinked_directories_are_not_followed(run_dir):
    """A link back into `runs/` would make the walk loop forever."""
    runs = run_dir.parent
    try:
        (runs / "loop").symlink_to(runs, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert [r["run_id"] for r in runs_mod.list_runs(runs)] == [run_dir.name]


def test_runs_are_ordered_by_name_with_numbers_read_numerically(run_dir):
    """Ordered by path, not recency: a matrix run interleaves revisions, so
    newest-first scatters one prompt's cells down the list instead of lining
    them up at the same position under each revision.
    """
    runs = run_dir.parent
    for rel in ("b_rev/kettle_r2", "b_rev/kettle_r10", "a_rev/kettle_r0"):
        _nest(runs, rel, run_dir)
    ids = [r["run_id"] for r in runs_mod.list_runs(runs)]
    assert ids == [
        run_dir.name,                # leading digits sort ahead of letters
        "a_rev/kettle_r0",
        "b_rev/kettle_r2",           # numeric, so r2 precedes r10 ...
        "b_rev/kettle_r10",          # ... which plain string order would invert
    ]


def test_nested_run_is_reachable_over_http(run_dir):
    """`{run_id:path}` is greedy, so the suffixed routes have to win over the
    bare detail route — otherwise `/urdf` is read as part of the id."""
    runs = run_dir.parent
    _nest(runs, "achieve/kettle", run_dir)
    client = TestClient(create_app(runs_dir=runs))
    assert client.get("/api/runs/achieve/kettle").status_code == 200
    assert client.get("/api/runs/achieve/kettle/urdf").status_code == 200
    assert client.get("/api/runs/achieve/kettle/trajectory").status_code == 200
    assert client.get("/api/runs/achieve/kettle/text/object.py").status_code == 200
    # the folder itself is not a run
    assert client.get("/api/runs/achieve").status_code == 404


# ---------------------------------------------------------------------------
# scene bundles
#
# A scene run breaks the "one run = one ObjectIR" assumption: it commits a
# *scene* IR, its geometry lives in a sibling `catalog/` rather than in
# `ir.json`, and the runs it started are in `sub_runs/`.
# ---------------------------------------------------------------------------


def _scene_ir(object_hash: str) -> dict:
    """Two instances of one catalog object — one on the floor, one on the other.

    The stacked instance is what exercises `_resolve_world_pose`: its pose is
    stated relative to the first instance's `top` anchor, never in world
    coordinates.
    """
    identity = {"rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0]}
    return {
        "id": "scene777",
        "name": "stack",
        "schema": "dycraft.scene.v1",
        "room": {
            "width": 3.0, "depth": 3.0, "height": 2.5,
            "floor_material": {"color": [200, 194, 185], "alpha": 1.0},
            "wall_material": {"color": [170, 174, 170], "alpha": 1.0},
        },
        "groups": [],
        "assertions": [],
        "instances": [
            {"id": 0, "name": "base", "object_name": "widget",
             "object_ref": object_hash,
             "placement": {"kind": "attach", "parent": "room",
                           "parent_anchor": {**identity, "name": "bottom"},
                           "child_anchor": {**identity, "name": "bottom"}}},
            {"id": 1, "name": "stacked", "object_name": "widget",
             "object_ref": object_hash,
             "placement": {"kind": "attach", "parent": "base",
                           "parent_anchor": {**identity, "name": "top"},
                           "child_anchor": {**identity, "name": "bottom"}}},
        ],
    }


def _stage_catalog(run: Path, object_hash: str = "abc123", *,
                   joint_default: float = 0.0) -> None:
    """A per-run catalog snapshot, laid out the way the harness stages one.

    The asset lives at `catalog/assets/meshes/...` while the object IR names it
    `assets/meshes/...` — i.e. relative to the catalog root, not to the run
    root. That gap is the whole reason the viewer rebases scene asset paths.
    """
    catalog = run / "catalog"
    (catalog / "assets" / "meshes").mkdir(parents=True)
    (catalog / "assets" / "meshes" / "deadbeef.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    obj_ir = {**_ir(), "id": object_hash}
    if joint_default:
        # A non-zero authored rest pose is what makes the URDF's default-bake
        # observable: URDF q=0 is dycraft q=default, so anything driving the
        # browser's joints has to subtract it.
        obj_ir["joints"] = [{**j, "default": joint_default}
                            for j in obj_ir["joints"]]
    (catalog / f"{object_hash}.json").write_text(json.dumps(obj_ir))
    (catalog / "manifest.json").write_text(
        json.dumps({"widget": object_hash}))


@pytest.fixture
def scene_run_dir(run_dir) -> Path:
    """A scene bundle: scene IR + catalog + sub_runs."""
    runs = run_dir.parent
    run = runs / "260101_000000_claude_scene"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({
        "run_id": run.name, "layer": "scene", "model": "test-model",
        "runtime": "claude_cli", "outcome": "committed",
        "prompt": "A table and a mug.",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:40:00+00:00",
    }))
    (run / "ir.json").write_text(json.dumps(_scene_ir("abc123")))
    (run / "scene.py").write_text("scene = 1\n")
    _stage_catalog(run)

    sub = run / "sub_runs"
    sub.mkdir()
    _nest(sub, "object_widget_001", run_dir)
    # A `_`-prefixed sibling is harness bookkeeping, not work product.
    hidden = sub / "_probe"
    hidden.mkdir()
    (hidden / "meta.json").write_text(json.dumps({
        "run_id": "_probe", "layer": "plain"}))
    return run


def test_scene_urdf_resolves_meshes_through_the_catalog(
        scene_run_dir):
    """The scene's geometry is in `catalog/`, and the browser fetches every
    mesh against the run root — so a catalog-relative path has to be rebased
    or the 3D view loads a robot full of 404s."""
    urdf = runs_mod.run_urdf(scene_run_dir)
    assert 'filename="catalog/assets/meshes/deadbeef.obj"' in urdf
    # Rebasing is only correct if the rebased path actually resolves.
    assert runs_mod.resolve_file(
        scene_run_dir, "catalog/assets/meshes/deadbeef.obj") is not None
    # Both instances placed, each replicating the object's own joint tree.
    assert '<joint name="world__base" type="fixed">' in urdf
    assert '<link name="stacked__body"/>' in urdf or \
           '<link name="stacked__body">' in urdf
    assert '<joint name="stacked__hinge" type="revolute">' in urdf


def _add_catalog_texture(run: Path, rel: str, *, stage_file: bool) -> None:
    """Point the catalog object's first part at a base-color texture."""
    cat = run / "catalog"
    if stage_file:
        path = cat / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    obj = json.loads((cat / "abc123.json").read_text())
    obj["parts"][0].setdefault("material", {})["texture"] = {
        "base_color": {"image": rel, "digest": "c" * 32}}
    (cat / "abc123.json").write_text(json.dumps(obj))


def test_a_catalog_texture_is_rebased_like_a_mesh(scene_run_dir):
    """A base-color image is a catalog asset exactly like the mesh.

    The object IR names it relative to the catalog root while the browser
    fetches against the run root, so it needs the same rebase. Left unrebased
    the fetch 404s, and a 404 texture is not merely untextured — see the next
    test for why that is worse.
    """
    _add_catalog_texture(scene_run_dir, "assets/textures/cafe.png",
                         stage_file=True)
    urdf = runs_mod.run_urdf(scene_run_dir)
    assert 'filename="catalog/assets/textures/cafe.png"' in urdf
    assert runs_mod.resolve_file(
        scene_run_dir, "catalog/assets/textures/cafe.png") is not None


def test_an_unservable_texture_is_dropped_not_left_dangling(scene_run_dir):
    """A texture that cannot be fetched must not reach the browser at all.

    This is the asymmetry that makes textures different from meshes. A missing
    mesh is missing geometry and the viewer falls back wholesale. A missing
    texture renders the part **black**: three.js multiplies the material color
    by the map, an unloaded map samples as zero, and a textured material's
    color is deliberately white so it does not tint the image. Black is
    strictly worse than the flat mean color the material still carries — and
    it is silent, because nothing errors.

    So: drop the texture, keep the geometry, let the part draw in its fallback.
    """
    _add_catalog_texture(scene_run_dir, "assets/textures/never_staged.png",
                         stage_file=False)
    urdf = runs_mod.run_urdf(scene_run_dir)
    assert "<texture" not in urdf
    # The geometry is untouched — a bad texture is not a reason to lose a part.
    assert 'filename="catalog/assets/meshes/deadbeef.obj"' in urdf


def test_ir_kind_reads_the_payload_not_the_layer(scene_run_dir, run_dir):
    """The kind of run is read off `ir.json`, not the layer name: everything
    IR-shaped keys off the payload."""
    detail = runs_mod.run_detail(scene_run_dir)
    assert detail["meta"]["layer"] == "scene"
    assert detail["ir"]["kind"] == "scene"
    assert runs_mod.run_detail(run_dir)["ir"]["kind"] == "object"


def test_scene_parts_and_joints_are_prefixed_like_the_urdf_links(scene_run_dir):
    """A Parts row has to name the same thing the 3D view does, so the flatten
    reuses the URDF's `<instance>__` naming rather than inventing one."""
    detail = runs_mod.run_detail(scene_run_dir)
    assert [p["name"] for p in detail["ir"]["parts"]] == [
        "base__body", "base__lid", "stacked__body", "stacked__lid"]
    assert [j["name"] for j in detail["ir"]["joints"]] == [
        "base__hinge", "stacked__hinge"]
    # Endpoints are prefixed too — an unprefixed parent would name the wrong
    # instance's link once two copies of one object are in the scene.
    hinge = detail["ir"]["joints"][1]
    assert (hinge["parent"], hinge["child"]) == ("stacked__body",
                                                 "stacked__lid")
    # Per-part facts survive the flatten.
    assert detail["ir"]["parts"][0]["fill_ratio"] == 0.125


def test_scene_instances_carry_their_placement_parent(scene_run_dir):
    """Placement is relational, so the parent chain is the layout."""
    instances = runs_mod.run_detail(scene_run_dir)["ir"]["instances"]
    assert [(i["name"], i["parent"], i["parent_anchor"]) for i in instances] == [
        ("base", "room", "bottom"), ("stacked", "base", "top")]
    assert all(i["in_catalog"] for i in instances)
    assert runs_mod.run_detail(scene_run_dir)["ir"]["catalog_resolved"] == 2


def test_scene_summary_counts_instances_rather_than_parts(scene_run_dir):
    """A SceneIR has no `parts`/`joints` of its own, so the object counts are
    structurally 0 and the sidebar needs the instance count instead."""
    by_id = {r["run_id"]: r for r in runs_mod.list_runs(scene_run_dir.parent)}
    summary = by_id[scene_run_dir.name]
    assert summary["ir_kind"] == "scene" and summary["n_instances"] == 2
    assert summary["n_parts"] == 0 and summary["n_joints"] == 0


def test_sub_runs_are_walked_and_addressable(scene_run_dir):
    """`sub_runs/` is the one artifact subdirectory holding real runs: without
    descending, every run a parent started is unreachable.
    """
    runs = scene_run_dir.parent
    ids = {r["run_id"] for r in runs_mod.list_runs(runs)}
    assert f"{scene_run_dir.name}/sub_runs/object_widget_001" in ids
    # `_`-prefixed entries stay internal.
    assert not any("_probe" in i for i in ids)
    nested = runs_mod.resolve_run(
        runs, f"{scene_run_dir.name}/sub_runs/object_widget_001")
    assert nested is not None
    assert runs_mod.run_detail(nested)["ir"]["kind"] == "object"


def test_sub_run_ids_are_runs_dir_relative_so_they_are_linkable(scene_run_dir):
    detail = runs_mod.run_detail(scene_run_dir, scene_run_dir.name)
    assert [s["run_id"] for s in detail["sub_runs"]] == [
        f"{scene_run_dir.name}/sub_runs/object_widget_001"]
    assert detail["sub_runs"][0]["layer"] == "object"


@pytest.mark.parametrize("recorded,shown", [("unified", "scene"),
                                             ("unified_animation", "animation")])
def test_a_recorded_run_shows_its_layer_under_the_current_name(
        scene_run_dir, recorded, shown):
    """Runs recorded before the rename say `unified` or `unified_animation`;
    the sidebar, the overview badge, and the sub-run list show `scene` and
    `animation`, so old and new runs of one layer read alike."""
    meta = json.loads((scene_run_dir / "meta.json").read_text())
    (scene_run_dir / "meta.json").write_text(json.dumps({**meta, "layer": recorded}))
    child = scene_run_dir / "sub_runs" / "object_widget_001" / "meta.json"
    (child).write_text(json.dumps({**json.loads(child.read_text()), "layer": recorded}))

    listed = {r["run_id"]: r for r in runs_mod.list_runs(scene_run_dir.parent)}
    assert listed[scene_run_dir.name]["layer"] == shown
    detail = runs_mod.run_detail(scene_run_dir, scene_run_dir.name)
    assert detail["meta"]["layer"] == shown
    assert detail["sub_runs"][0]["layer"] == shown


def test_object_runs_carry_no_sub_runs(run_dir):
    detail = runs_mod.run_detail(run_dir)
    assert detail["sub_runs"] == []


def test_context_summary_is_advertised_only_when_safely_servable(run_dir):
    assert runs_mod.run_detail(run_dir)["has_context_summary"] is False
    (run_dir / "context_summary.md").write_text(
        "# Context\n\n- **built** a widget\n")
    assert runs_mod.run_detail(run_dir)["has_context_summary"] is True

    client = TestClient(create_app(runs_dir=run_dir.parent,
                                   dist_dir=Path("/nonexistent")))
    response = client.get(
        f"/api/runs/{run_dir.name}/text/context_summary.md")
    assert response.status_code == 200
    assert response.json()["text"].startswith("# Context")


def test_context_summary_detection_works_for_sub_runs(
        scene_run_dir):
    child_id = f"{scene_run_dir.name}/sub_runs/object_widget_001"
    child = runs_mod.resolve_run(scene_run_dir.parent, child_id)
    assert child is not None
    (child / "context_summary.md").write_text("## Child summary\n")
    assert runs_mod.run_detail(child, child_id)["has_context_summary"] is True


def test_scene_without_a_catalog_says_so_instead_of_drawing_nothing(
        scene_run_dir):
    """`scene_ir_to_urdf` skips instances it cannot resolve, so an absent
    catalog yields a valid URDF containing only the `world` link — an empty
    viewport with no explanation. Refuse it and name the reason."""
    import shutil
    shutil.rmtree(scene_run_dir / "catalog")
    with pytest.raises(runs_mod.UrdfUnavailable, match="catalog"):
        runs_mod.run_urdf(scene_run_dir)


def test_unrecognized_ir_shape_is_reported_as_such(run_dir):
    (run_dir / "ir.json").write_text(json.dumps({"id": "x", "name": "y"}))
    with pytest.raises(runs_mod.UrdfUnavailable, match="neither"):
        runs_mod.run_urdf(run_dir)


def test_missing_ir_still_reports_the_original_reason(run_dir):
    (run_dir / "ir.json").unlink()
    with pytest.raises(runs_mod.UrdfUnavailable, match="no ir.json"):
        runs_mod.run_urdf(run_dir)


def test_scene_run_is_fully_served_over_http(scene_run_dir):
    runs = scene_run_dir.parent
    client = TestClient(create_app(runs_dir=runs, dist_dir=Path("/nonexistent")))
    rid = scene_run_dir.name
    assert client.get(f"/api/runs/{rid}").status_code == 200
    assert client.get(f"/api/runs/{rid}/urdf").status_code == 200
    # The three.js loader fetches the URDF and its meshes off one base URL.
    assert client.get(f"/api/runs/{rid}/files/model.urdf").status_code == 200
    mesh = client.get(
        f"/api/runs/{rid}/files/catalog/assets/meshes/deadbeef.obj")
    assert mesh.status_code == 200 and mesh.content.startswith(b"v 0 0 0")
    assert client.get(f"/api/runs/{rid}/text/scene.py").status_code == 200
    sub = f"{rid}/sub_runs/object_widget_001"
    assert client.get(f"/api/runs/{sub}").status_code == 200
    assert client.get(f"/api/runs/{sub}/urdf").status_code == 200


def test_urdf_404_detail_names_the_actual_reason(scene_run_dir):
    """Not just "no ir.json": a committed scene with no catalog has an IR and
    still nothing to draw, and the 404 says which."""
    import shutil
    shutil.rmtree(scene_run_dir / "catalog")
    client = TestClient(create_app(runs_dir=scene_run_dir.parent,
                                   dist_dir=Path("/nonexistent")))
    response = client.get(f"/api/runs/{scene_run_dir.name}/urdf")
    assert response.status_code == 404
    assert "catalog" in response.json()["detail"]


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# native animation transport
# ---------------------------------------------------------------------------


def _native_bundle(run: Path, *, frames: int = 3) -> tuple[dict, str]:
    """A structurally real native bundle: two objects, one shared mesh.

    Small on purpose, and hermetic — no Blender. What it has to be *real* about
    is the manifest, because the transport's whole integrity story is "the
    manifest lists this file and the digest names its bytes".
    """
    from dycraft.animation.blender.bundle import make_manifest
    from dycraft.animation.blender.evaluation import write_mesh

    evaluation = run / "bundle" / "evaluation"
    evaluation.mkdir(parents=True)
    (run / "bundle" / "animation.blend").write_bytes(b"not really a blend")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    triangles = np.array([[0, 1, 2]], dtype="i4")
    digest = write_mesh(evaluation, vertices, triangles, set())

    for frame in range(frames):
        moved = [[1.0, 0, 0, float(frame)], [0, 1.0, 0, 0],
                 [0, 0, 1.0, 0], [0, 0, 0, 1]]
        (evaluation / f"{frame:06d}.json").write_text(json.dumps({
            "frame": frame, "transforms": {}, "joint_positions": {},
            "objects": [
                {"key": "o0", "name": "part/thing/body", "identity": "thing/body",
                 "room": False, "matrix": moved, "hidden": False,
                 "color": [0.1, 0.2, 0.3, 1.0], "mesh": digest},
                # A room surface, which the viewport has never drawn.
                {"key": "o1", "name": "room_0", "identity": None, "room": True,
                 "matrix": moved, "hidden": False,
                 "color": [1.0, 1.0, 1.0, 1.0], "mesh": digest},
            ]}))
    (evaluation / "index.json").write_text(json.dumps({"n_frames": frames}))

    ir = make_manifest(run / "bundle", scene_ref="scene", evaluation={
        "fps": 24.0, "n_frames": frames, "frame_start": 0,
        "blender_version": "test", "declarations": {"events": []}})
    (run / "ir.json").write_text(json.dumps(ir))
    return ir, digest


def _header(blob: bytes) -> dict:
    return json.loads(blob[4:4 + int.from_bytes(blob[:4], "little")])


def test_a_native_clip_ships_its_motion_once_and_its_geometry_by_digest(tmp_path):
    """The transport sends motion once per clip and geometry once per mesh.

    One whole-scene geometry request per frame would re-fetch the same meshes
    for every frame of the clip and, at 30 fps, cancel each request before it
    landed. The track carries motion only; geometry comes back once per
    distinct mesh.
    """
    from dycraft.viewer.native_animation import mesh_blob, track

    run = tmp_path / "run"
    run.mkdir()
    _ir, digest = _native_bundle(run, frames=3)

    header = _header(track(run))
    assert header["n_frames"] == 3
    # One drawable object; the room surface is dropped, as the URDF drops it.
    assert [o["name"] for o in header["objects"]] == ["part/thing/body"]
    # A mesh that never changes collapses to one index, not three.
    assert header["objects"][0]["mesh"] == 0
    assert header["objects"][0]["hidden"] is False
    assert header["meshes"] == [digest]
    assert header["objects"][0]["identity"] == "thing/body"

    # The matrices, in the order and the major the client reads them.
    blob = track(run)
    offset = 4 + int.from_bytes(blob[:4], "little")
    matrices = np.frombuffer(blob, dtype="<f4", offset=offset).reshape(3, 1, 12)
    assert list(matrices[2, 0]) == [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0]

    mesh = mesh_blob(run, digest)
    assert _header(mesh)["vertices"] == 3 and _header(mesh)["triangles"] == 1


def test_a_native_mesh_request_is_refused_unless_the_manifest_lists_it(tmp_path):
    """The digest arrives in a URL, so membership is the only thing standing
    between a visitor and any `.npz` on the box. Content addressing proves the
    bytes are the mesh they claim to be; it says nothing about whose they are."""
    from dycraft.viewer.native_animation import mesh_blob

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run)
    stranger = "0" * 64
    (run / "bundle" / "evaluation" / "meshes" / f"{stranger}.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a member"):
        mesh_blob(run, stranger)
    with pytest.raises(ValueError):
        mesh_blob(run, "../../../etc/passwd")


def test_a_deforming_frame_can_come_as_its_positions_only(tmp_path):
    """A deforming object's frames share its triangles, two thirds of a mesh's bytes, so the viewport takes them once
    and then each frame's vertices alone. They are the whole mesh's vertices exactly: the viewport works its creased
    normals out from them, and a rounding tips corners across the crease angle. The topology hash is what says the
    triangles are the same; the membership rule is the same as for a whole mesh."""
    from dycraft.viewer.native_animation import mesh_blob, mesh_positions

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run)
    digest = track_digest(run)
    blob = mesh_positions(run, digest)
    header = _header(blob)
    assert header["encoding"] == "f32" and header["vertices"] == 3
    whole = mesh_blob(run, digest)
    assert header["topology"] == _header(whole)["topology"]
    offset = 4 + int.from_bytes(blob[:4], "little")
    whole_offset = 4 + int.from_bytes(whole[:4], "little")
    assert blob[offset:] == whole[whole_offset:whole_offset + 3 * 3 * 4], "the whole mesh's vertex bytes, unrounded"
    with pytest.raises(ValueError, match="not a member"):
        mesh_positions(run, "0" * 64)


def track_digest(run):
    from dycraft.viewer.native_animation import track
    return _header(track(run))["meshes"][0]


def test_a_tampered_frame_record_is_refused(tmp_path):
    """The frame records are named by position rather than by content, so the
    manifest hash is the only thing that pins them."""
    from dycraft.viewer.native_animation import track

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run)
    record = run / "bundle" / "evaluation" / "000001.json"
    tampered = json.loads(record.read_text())
    tampered["objects"][0]["matrix"][0][3] = 99.0
    record.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash mismatch"):
        track(run)


def _binary_native_bundle(run: Path) -> dict:
    """A clip stored as binary frame records, shaped to reach every branch of
    the fast path: a body placed by its transform entry and one carrying its
    own matrix, a body that arrives late, one that hides, a mesh that changes,
    a room surface, fields kept beside the fixed layout, and two tables."""
    from dycraft.animation.blender.bundle import make_manifest
    from dycraft.animation.blender.evaluation import write_mesh, write_signed_record

    evaluation = run / "bundle" / "evaluation"
    evaluation.mkdir(parents=True)
    (run / "bundle" / "animation.blend").write_bytes(b"not really a blend")
    triangles = np.array([[0, 1, 2]], dtype="i4")
    small = write_mesh(evaluation, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "f4"), triangles, set())
    large = write_mesh(evaluation, np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], "f4"), triangles, set())

    def matrix(x, y=0.0):
        return [[1.0, 0, 0, x], [0, 1.0, 0, y], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]

    known: set = set()
    frames = 4
    for frame in range(frames):
        objects = [
            {"key": "o0", "name": "part/thing/body", "identity": "thing/body", "room": False,
             "matrix": matrix(0.5 * frame), "hidden": frame == 2, "color": [0.1, 0.2, 0.3, 1.0],
             "mesh": small if frame < 3 else large},
            {"key": "o1", "name": "room_0", "identity": None, "room": True, "matrix": matrix(0.0),
             "hidden": False, "color": [1.0, 1.0, 1.0, 1.0], "mesh": small},
        ]
        if frame >= 1:
            objects.append({"key": "o2", "name": "part/thing/lid", "identity": "thing/lid", "room": False,
                            "matrix": matrix(1.0, float(frame)), "hidden": False,
                            "color": [0.9, 0.5, 0.1, 1.0], "mesh": large, "slot": frame})
        # Odd frames carry no transform entries, so their bodies keep their own matrices.
        transforms = {} if frame % 2 else {o["name"]: o["matrix"] for o in objects}
        write_signed_record(evaluation, {"frame": frame, "transforms": transforms, "joint_positions": {},
                                         "objects": objects}, known)
    (evaluation / "index.json").write_text(json.dumps({"n_frames": frames}))
    ir = make_manifest(run / "bundle", scene_ref="scene", evaluation={
        "fps": 24.0, "n_frames": frames, "frame_start": 0,
        "blender_version": "test", "declarations": {"events": []}})
    (run / "ir.json").write_text(json.dumps(ir))
    return ir


def test_a_track_read_from_binary_records_is_the_track_read_from_whole_records(tmp_path):
    """The fast path reads each record's fixed blocks as arrays instead of
    decoding it, which makes a long film's first view seconds rather than
    minutes. It has to give the same bytes as decoding every record
    whole, the path a bundle of JSON records still takes."""
    from pathlib import PurePosixPath

    from dycraft.viewer import native_animation as NA

    run = tmp_path / "run"
    run.mkdir()
    ir = _binary_native_bundle(run)
    prefix = PurePosixPath(ir["evaluation"]).parent
    fast = NA._binary_track(run, ir, prefix, 4)
    assert fast is not None, "binary records were not read by the fast path"
    assert fast == NA._record_track(run, ir, prefix, 4)
    header = _header(fast)
    assert [o["name"] for o in header["objects"]] == ["part/thing/body", "part/thing/lid"]


def test_a_tampered_binary_record_is_refused(tmp_path):
    """The fast path checks every record against the manifest before it reads
    it, as the record path does."""
    from dycraft.viewer.native_animation import track

    run = tmp_path / "run"
    run.mkdir()
    _binary_native_bundle(run)
    record = run / "bundle" / "evaluation" / "000001.bin"
    blob = bytearray(record.read_bytes())
    blob[60] ^= 0xFF
    record.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="hash mismatch"):
        track(run)


def test_native_endpoints_cache_geometry_and_not_motion(tmp_path):
    """A track is per run and a live run rewrites it under the same URL; a mesh
    URL names its own bytes and can never go stale."""
    runs = tmp_path / "runs"
    run = runs / "260101_000000_claude_native"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({
        "run_id": run.name, "layer": "animation", "model": "test-model",
        "runtime": "claude_cli", "outcome": "committed", "prompt": "A clip.",
        "started_at": "2026-01-01T00:00:00+00:00",
    }))
    _ir, digest = _native_bundle(run)

    client = TestClient(create_app(runs_dir=runs,
                                   dist_dir=Path("/nonexistent-dist")))
    motion = client.get(f"/api/runs/{run.name}/native-track")
    assert motion.status_code == 200
    assert motion.headers["cache-control"] == "no-cache"
    assert _header(motion.content)["n_frames"] == 3

    # Gzipped once per film for a browser that takes it, every browser; the same bytes once unpacked.
    from dycraft.viewer.native_animation import track
    packed = client.get(f"/api/runs/{run.name}/native-track", headers={"Accept-Encoding": "gzip"})
    assert packed.headers["content-encoding"] == "gzip" and "Accept-Encoding" in packed.headers["vary"]
    assert packed.content == track(run)
    plain = client.get(f"/api/runs/{run.name}/native-track", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.content == track(run)

    geometry = client.get(f"/api/runs/{run.name}/native-mesh/{digest}")
    assert geometry.status_code == 200
    assert "immutable" in geometry.headers["cache-control"]
    # Its triangle indices compress, so it goes gzipped to a browser that takes it.
    assert geometry.headers["content-encoding"] == "gzip"
    assert client.get(f"/api/runs/{run.name}/native-mesh/{digest}",
                      headers={"Accept-Encoding": "identity"}).content == geometry.content

    assert client.get(f"/api/runs/{run.name}/native-mesh/{'0' * 64}").status_code == 404


@pytest.fixture
def client(run_dir) -> TestClient:
    return TestClient(create_app(runs_dir=run_dir.parent,
                                 dist_dir=Path("/nonexistent-dist")))


def test_runs_and_detail_endpoints(client, run_dir):
    listed = client.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in listed] == [run_dir.name]
    detail = client.get(f"/api/runs/{run_dir.name}").json()
    assert detail["has_ir"] is True
    assert "object.py" in detail["source_files"]
    assert detail["renders"] == ["0_front.png"]


def _bundle_urdf(*names: str) -> str:
    meshes = "".join(f'<mesh filename="{name}"/>' for name in names)
    return f"<robot>{meshes}</robot>"


def test_a_mesh_bundle_chunk_round_trips_through_tarfile(run_dir):
    """One request per visual is thousands of them on a large scene.

    The bundle replaces that with a handful of tars, so the entries have to come
    back byte-identical and named by their run-relative path — that name is how
    the browser matches an entry to the mesh it was going to fetch.
    """
    import io
    import tarfile

    from dycraft.viewer import mesh_bundle

    (run_dir / "assets" / "meshes" / "cafe0001.obj").write_text("v 1 2 3\n")
    urdf = _bundle_urdf("assets/meshes/deadbeef.obj", "assets/meshes/cafe0001.obj")

    plan = mesh_bundle.bundle_plan(run_dir, urdf)
    manifest = mesh_bundle.manifest(plan)
    assert manifest["files"] == 2
    assert len(manifest["chunks"]) == 1

    data = mesh_bundle.chunk_tar(plan, 0)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        got = {member.name: tar.extractfile(member).read() for member in tar}
    assert got == {
        "assets/meshes/deadbeef.obj":
            (run_dir / "assets" / "meshes" / "deadbeef.obj").read_bytes(),
        "assets/meshes/cafe0001.obj": b"v 1 2 3\n",
    }
    # Identical content, identical bytes: the response is served immutable.
    assert mesh_bundle.chunk_tar(plan, 0) == data
    assert mesh_bundle.chunk_tar(plan, 1) is None


def test_mesh_bundle_chunks_are_deterministic_and_bounded(run_dir, monkeypatch):
    """The manifest and the chunk requests are separate calls that must agree,
    and a chunk has to stay small enough to be an ordinary cache entry."""
    from dycraft.viewer import mesh_bundle

    monkeypatch.setattr(mesh_bundle, "BUNDLE_TARGET_BYTES", 1000)
    names = []
    for index in range(9):
        name = f"assets/meshes/{index:032x}.obj"
        (run_dir / name).write_text("v 0 0 0\n" * 50)      # 400 bytes each
        names.append(name)
    urdf = _bundle_urdf(*reversed(names))                  # order must not matter

    plan = mesh_bundle.bundle_plan(run_dir, urdf)
    assert [len(chunk) for chunk in plan] == [2, 2, 2, 2, 1]
    assert all(sum(size for _, _, size in chunk) <= 1000 for chunk in plan)
    assert plan == mesh_bundle.bundle_plan(run_dir, urdf)
    # Every mesh lands in exactly one chunk.
    packed = [name for chunk in plan for name, _, _ in chunk]
    assert sorted(packed) == sorted(names) == packed

    # A file bigger than the target is its own chunk rather than dropped.
    (run_dir / "assets" / "meshes" / "big.obj").write_text("v" * 4000)
    plan = mesh_bundle.bundle_plan(run_dir, _bundle_urdf("assets/meshes/big.obj"))
    assert [len(chunk) for chunk in plan] == [1]


def test_mesh_bundle_leaves_out_what_it_must_not_serve(run_dir):
    """A `filename` is agent-authored text, and the bundle opens files by it."""
    from dycraft.viewer import mesh_bundle

    outside = run_dir.parent / "secret.obj"
    outside.write_text("v 9 9 9\n")
    long_name = "assets/meshes/" + "d" * 120 + ".obj"
    (run_dir / long_name).write_text("v 0 0 0\n")

    plan = mesh_bundle.bundle_plan(run_dir, _bundle_urdf(
        "../secret.obj",
        "/etc/passwd",
        "assets/meshes/absent.obj",
        # A baked asset: it lives in a cache outside the run, and stays on the
        # per-file route.
        "__derived__/abc123/000_body.glb",
        # Too long for USTAR's name field.
        long_name,
        "assets/meshes/deadbeef.obj",
    ))
    assert [name for chunk in plan for name, _, _ in chunk] == [
        "assets/meshes/deadbeef.obj"]


def test_mesh_bundle_endpoints(client, run_dir):
    manifest = client.get(f"/api/runs/{run_dir.name}/mesh-bundle").json()
    assert manifest["files"] == 1
    mesh_bytes = (run_dir / "assets" / "meshes" / "deadbeef.obj").stat().st_size
    assert manifest["chunks"] == [
        {"index": 0, "files": 1, "bytes": mesh_bytes}]

    chunk = client.get(f"/api/runs/{run_dir.name}/mesh-bundle/0",
                       headers={"Accept-Encoding": "gzip"})
    assert chunk.status_code == 200
    assert chunk.headers["content-type"] == "application/x-tar"
    assert "immutable" in chunk.headers["cache-control"]
    # Gzipped by the route at level 1, in the request's worker thread: OBJ text is a quarter of its size that way,
    # which is most of a first open over a slow link. Never by the middleware, on the event loop.
    assert chunk.headers["content-encoding"] == "gzip"
    assert b"assets/meshes/deadbeef.obj" in chunk.content
    plain = client.get(f"/api/runs/{run_dir.name}/mesh-bundle/0", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.content == chunk.content

    assert client.get(f"/api/runs/{run_dir.name}/mesh-bundle/7").status_code == 404
    assert client.get("/api/runs/nope/mesh-bundle").status_code == 404


def test_assets_are_served_uncompressed_and_json_is_not(client, run_dir):
    """The middleware compresses the JSON, never the assets.

    `GZipMiddleware` app-wide at Starlette's default `compresslevel=9` would be
    the largest cost in loading a scene, about thirty times the uncompressed
    serving time, and because compression runs on the event loop it would
    serialize the whole API behind it, until the browser gives up on requests.
    The assets that compress well (the mesh bundle's chunks, native meshes, the
    native track) are compressed by their own routes instead, off the event loop.
    """
    mesh = client.get(
        f"/api/runs/{run_dir.name}/files/assets/meshes/deadbeef.obj",
        headers={"Accept-Encoding": "gzip"})
    assert mesh.status_code == 200
    assert "content-encoding" not in mesh.headers

    # The same for a run's big binary payloads, addressed under other prefixes.
    from dycraft.viewer.app import _skips_compression
    assert _skips_compression("/api/runs/x/native-track")
    assert _skips_compression("/api/runs/x/native-mesh/deadbeef")
    assert not _skips_compression("/api/runs/x/trajectory")

    # Whereas the payload that is worth compressing still is: the generated
    # URDF of a real scene is megabytes of repetitive XML.
    detail = client.get(f"/api/runs/{run_dir.name}",
                        headers={"Accept-Encoding": "gzip"})
    assert detail.status_code == 200
    assert detail.headers.get("content-encoding") == "gzip"


def test_a_regenerated_urdf_is_cached_until_the_ir_changes(client, run_dir):
    """About a second per document on a large scene, asked for on every
    re-selection and again when the baked variant lands, but a live run
    rewriting `ir.json` has to invalidate it."""
    first = client.get(f"/api/runs/{run_dir.name}/urdf").text
    assert client.get(f"/api/runs/{run_dir.name}/urdf").text == first

    (run_dir / "assets" / "meshes" / "cafe1234.obj").write_text(
        "v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n")
    ir = json.loads((run_dir / "ir.json").read_text())
    ir["parts"][0]["params"]["mesh"] = "assets/meshes/cafe1234.obj"
    (run_dir / "ir.json").write_text(json.dumps(ir))

    second = client.get(f"/api/runs/{run_dir.name}/urdf").text
    assert 'filename="assets/meshes/cafe1234.obj"' in second
    assert second != first


def test_urdf_is_generated_from_the_ir(client, run_dir):
    response = client.get(f"/api/runs/{run_dir.name}/urdf")
    assert response.status_code == 200
    assert 'filename="assets/meshes/deadbeef.obj"' in response.text
    assert '<joint name="hinge" type="revolute">' in response.text


def test_model_urdf_alias_matches_the_urdf_route(client, run_dir):
    """The three.js loader fetches `{base}/model.urdf`; the alias serves it so
    one base URL covers both the robot and its meshes."""
    alias = client.get(f"/api/runs/{run_dir.name}/files/model.urdf")
    assert alias.status_code == 200
    assert alias.text == client.get(f"/api/runs/{run_dir.name}/urdf").text
    mesh = client.get(
        f"/api/runs/{run_dir.name}/files/assets/meshes/deadbeef.obj")
    assert mesh.status_code == 200 and mesh.content.startswith(b"v 0 0 0")


def test_urdf_404s_without_an_ir(client, run_dir):
    (run_dir / "ir.json").unlink()
    assert client.get(f"/api/runs/{run_dir.name}/urdf").status_code == 404
    assert client.get(
        f"/api/runs/{run_dir.name}/files/model.urdf").status_code == 404


def test_text_endpoint_rejects_binary_suffixes(client, run_dir):
    ok = client.get(f"/api/runs/{run_dir.name}/text/object.py").json()
    assert ok["text"] == "obj = 1\n" and ok["truncated"] is False
    assert client.get(
        f"/api/runs/{run_dir.name}/text/renders/0_front.png").status_code == 415


def test_file_endpoint_blocks_traversal(client, run_dir):
    for bad in ("../../etc/passwd", "/etc/passwd"):
        assert client.get(
            f"/api/runs/{run_dir.name}/files/{bad}").status_code == 404


def test_unknown_run_404s(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/trajectory").status_code == 404


def test_trajectory_endpoint(client, run_dir):
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_claude_transcript(True))
    body = client.get(f"/api/runs/{run_dir.name}/trajectory").json()
    assert body["totals"]["turns"] == 2
    assert body["reasoning_available"] is True


# ---------------------------------------------------------------------------
# billing detail: write tier, rate modifiers, mixed models
# ---------------------------------------------------------------------------


_OPUS_RATES = {
    "claude-opus-5": {"input": 5.0, "cache_read": 0.5, "cache_write_5m": 6.25,
                      "cache_write_1h": 10.0, "output": 25.0},
}


@pytest.fixture
def opus_prices(tmp_path, monkeypatch) -> Path:
    table = tmp_path / "prices.json"
    table.write_text(json.dumps(_OPUS_RATES))
    monkeypatch.setenv("DYCRAFT_PRICES_JSON", str(table))
    return table


def test_cache_write_tier_changes_the_bill(opus_prices):
    """1 h writes cost 1.6x the 5 m rate — pricing them alike understates a run.

    dycraft runs write at the 1 h tier.
    """
    million = 1_000_000
    five_m = {"input": 0, "cache_read": 0, "cache_write_5m": million,
              "cache_write_1h": 0, "output": 0}
    one_h = {"input": 0, "cache_read": 0, "cache_write_5m": 0,
             "cache_write_1h": million, "output": 0}
    assert _priced_total(five_m, "claude-opus-5") == 6.25
    assert _priced_total(one_h, "claude-opus-5") == 10.0


def test_untiered_cache_write_falls_back_to_the_5m_rate(opus_prices):
    usage = {"input": 0, "cache_read": 0, "cache_write": 1_000_000,
             "output": 0}
    assert _priced_total(usage, "claude-opus-5") == 6.25


def test_rate_modifiers_come_from_the_recorded_billing_mode(opus_prices):
    usage = {"input": 0, "cache_read": 0, "cache_write_5m": 0,
             "cache_write_1h": 0, "output": 1_000_000}
    base = cost_mod.price_usage(usage, "claude-opus-5")
    assert base["total"] == 25.0 and base["multiplier"] == 1.0
    # us inference 1.1x, batch 0.5x, fast 2x — and they compose.
    assert cost_mod.price_usage(usage, "claude-opus-5",
                                inference_geo="us")["total"] == 27.5
    assert cost_mod.price_usage(usage, "claude-opus-5",
                                service_tier="batch")["total"] == 12.5
    assert cost_mod.price_usage(usage, "claude-opus-5",
                                speed="fast")["total"] == 50.0
    both = cost_mod.price_usage(usage, "claude-opus-5", inference_geo="us",
                                service_tier="batch")
    assert both["total"] == 13.75 and both["multiplier"] == pytest.approx(0.55)
    # "not_available" / "standard" are the common recorded values: no change.
    assert cost_mod.price_usage(usage, "claude-opus-5",
                                inference_geo="not_available",
                                service_tier="standard",
                                speed="standard")["total"] == 25.0


def _mixed_model_transcript() -> str:
    """Two fable turns then an opus turn, as a provider fallback produces."""
    def row(mid, model, out, ts):
        return {
            "type": "assistant", "timestamp": ts,
            "message": {
                "id": mid, "model": model,
                "usage": {"input_tokens": 2, "cache_read_input_tokens": 1000,
                          "cache_creation_input_tokens": 500,
                          "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                             "ephemeral_1h_input_tokens": 500},
                          "output_tokens": out,
                          "inference_geo": "not_available",
                          "service_tier": "standard", "speed": "standard"},
                "content": [{"type": "text", "text": "x"}],
            },
        }
    rows = [
        row("m1", "claude-fable-5", 100, "2026-01-01T00:00:10+00:00"),
        row("m2", "claude-fable-5", 100, "2026-01-01T00:00:20+00:00"),
        row("m3", "claude-opus-5", 1_000_000, "2026-01-01T00:00:30+00:00"),
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def test_run_model_is_the_dominant_one_not_the_last(run_dir, opus_prices):
    """A fable run with an opus tail must not be labeled an opus run.

    The provider falls back on the final turns, so taking the last `model` seen
    mis-attributes both the model and the price for the whole run.
    """
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_mixed_model_transcript())
    r = build_trajectory(run_dir)
    assert r["model"] == "claude-fable-5"
    assert r["models"] == [{"model": "claude-fable-5", "turns": 2},
                           {"model": "claude-opus-5", "turns": 1}]


def test_partial_pricing_is_reported_not_hidden(run_dir, opus_prices):
    """With fable unpriced the total covers only the opus turn — say so."""
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_mixed_model_transcript())
    t = build_trajectory(run_dir)["totals"]
    assert t["cost_complete"] is False
    assert t["unpriced_models"] == ["claude-fable-5"]
    # Only the opus turn: 1M output at $25 + its cache/input slivers.
    assert t["cost_usd"] == pytest.approx(
        25.0 + 1000 * 0.5 / 1e6 + 500 * 10.0 / 1e6 + 2 * 5.0 / 1e6)
    assert set(t["cost_buckets"]) == {"input", "cache_read", "cache_write_1h",
                                      "output"}


def test_gross_input_matches_the_articraft_vocabulary(run_dir, opus_prices):
    """`gross_input` is everything read, so it is comparable to their figure."""
    (run_dir / tc.TRANSCRIPT_NAME).write_text(_mixed_model_transcript())
    t = build_trajectory(run_dir)["totals"]
    assert t["gross_input"] == t["input"] + t["cache_read"] + t["cache_write"]
    assert t["gross_input"] == 3 * (2 + 1000 + 500)
    assert t["cache_write_1h"] == 1500 and t["cache_write_5m"] == 0


# ---------------------------------------------------------------------------
# Animation runs
#
# The layer breaks the assumption every other one shares — that a run's own
# `ir.json` is what you draw. Here `ir.json` is the native Blender animation
# record and the geometry is the SceneIR staged beside it, so each test below
# pins one place where reading the wrong file produces a picture that is
# plausible and wrong rather than an error.
# ---------------------------------------------------------------------------


_HINGE_DEFAULT = 0.4


_EVALUATION = "bundle/evaluation/index.json"


def _anim_ir(n: int = 3) -> dict:
    """A native animation record over the staged scene: three frames at 10 fps."""
    return {
        "id": "anim999",
        "schema": "dycraft.animation.blender.v1",
        "name": "slide",
        "scene_ref": "scene777",
        "fps": 10,
        "n_frames": n,
        "duration": (n - 1) / 10,
        "events": [{"name": "arrives", "predicate": {"kind": "joint_at"},
                    "window": [0.1, 0.2]}],
        "provenance": [{"label": "base.slide#1", "frames": [0, n - 1]}],
        "evaluation": _EVALUATION,
    }


@pytest.fixture
def anim_run_dir(run_dir) -> Path:
    """An animation bundle: native record + staged scene.json + catalog + film."""
    runs = run_dir.parent
    run = runs / "260101_000000_claude_anim"
    (run / "compile_reports").mkdir(parents=True)
    (run / "bundle" / "film").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({
        "run_id": run.name, "layer": "animation", "model": "test-model",
        "runtime": "claude_cli", "outcome": "committed",
        "prompt": "The base slides.", "scene_ref": "scene777",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:10:00+00:00",
    }))
    from dycraft.animation.blender.bundle import file_hash
    (run / _EVALUATION).parent.mkdir(parents=True)
    (run / _EVALUATION).write_text("{}")
    ir = _anim_ir()
    ir["files"] = {_EVALUATION: file_hash(run / _EVALUATION)}
    ir["film"] = "bundle/film/clip.mp4"
    (run / "ir.json").write_text(json.dumps(ir))
    (run / "scene.json").write_text(json.dumps(_scene_ir("abc123")))
    (run / "animation.py").write_text("anim = 1\n")
    (run / "bundle" / "film" / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    _stage_catalog(run, joint_default=_HINGE_DEFAULT)
    (run / "compile_reports" / "latest.json").write_text(json.dumps({
        "ok": True, "ir_hash": "anim999",
        "verifier_reports": [
            # AnimationWellFormed also carries an `events` key, and sorts
            # first. Reading "whichever report has one" picks this up and
            # reports every event as never satisfied.
            {"name": "AnimationWellFormed", "verdict": "pass",
             "evidence": {"actors": ["base"], "events": ["arrives"],
                          "n_frames": 3}},
            {"name": "EventsSatisfied", "verdict": "pass",
             "evidence": {"events": [{"name": "arrives",
                                      "satisfied_at": 0.2}]}},
        ],
    }))
    return run


def test_animation_ir_is_recognized_by_its_schema_tag(anim_run_dir):
    """It has neither `parts` nor `instances`, so shape-matching sees nothing
    and every geometry path would silently treat it as an unknown IR."""
    ir = json.loads((anim_run_dir / "ir.json").read_text())
    assert scene_mod.ir_kind(ir) == "animation"


def test_list_runs_counts_frames_and_the_staged_scenes_instances(anim_run_dir):
    row = next(r for r in runs_mod.list_runs(anim_run_dir.parent)
               if r["run_id"] == anim_run_dir.name)
    assert row["ir_kind"] == "animation"
    assert row["n_frames"] == 3
    # From scene.json: the animation record says nothing about how much is on
    # screen.
    assert row["n_instances"] == 2


def test_urdf_for_an_animation_run_draws_the_staged_scene(anim_run_dir):
    """Without the swap this 404s with "neither an ObjectIR nor a SceneIR" and
    the viewport shows nothing for a run that has a whole scene to draw."""
    urdf = runs_mod.run_urdf(anim_run_dir)
    assert '<joint name="world__base" type="fixed">' in urdf
    assert '<joint name="world__stacked" type="fixed">' in urdf
    assert 'filename="catalog/assets/meshes/deadbeef.obj"' in urdf


def test_a_film_without_its_staged_scene_draws_from_its_track(anim_run_dir):
    """A native film with no staged scene (an installed film, staged by
    `benchmark.stage`) draws itself from its track under a bare `world` link,
    and keeps its timeline, rather than refusing to open."""
    (anim_run_dir / "scene.json").unlink()
    urdf = runs_mod.run_urdf(anim_run_dir)
    assert '<link name="world"' in urdf and "<visual" not in urdf
    assert runs_mod.run_animation(anim_run_dir)["n_frames"]


def test_run_detail_fills_geometry_from_the_staged_scene(anim_run_dir):
    ir = runs_mod.run_detail(anim_run_dir, anim_run_dir.name)["ir"]
    assert ir["kind"] == "animation"
    assert ir["id"] == "anim999"          # the animation's own hash...
    assert ir["scene_id"] == "scene777"   # ...and the scene it animates
    assert ir["clip"] == "bundle/film/clip.mp4"
    assert ir["animation"]["scene_ref_ok"] is True
    # Parts and joints come from the scene, so the Parts panel and the joint
    # sliders — which the timeline drives — work as on any scene run.
    assert [p["name"] for p in ir["parts"]] == [
        "base__body", "base__lid", "stacked__body", "stacked__lid"]
    assert ir["catalog_resolved"] == 2


def test_run_detail_events_carry_their_verdicts_not_just_the_claim(anim_run_dir):
    """The IR states what the animation *claimed*; whether it held is in the
    verifier report. Serving the raw list makes every event of a passing run
    read as unmet — a red "unmet" beside a green `EventsSatisfied: pass`."""
    events = runs_mod.run_detail(
        anim_run_dir, anim_run_dir.name)["ir"]["animation"]["events"]
    assert [(e["name"], e["satisfied_at"]) for e in events] == [("arrives", 0.2)]
    # And the same annotation the scrubber ticks use, from one code path.
    assert events == runs_mod.run_animation(anim_run_dir)["events"]


def test_run_detail_reports_a_swapped_scene_rather_than_drawing_it(
        anim_run_dir):
    """`scene_ref_ok` is the only signal that the staged scene is not the one
    the IR was verified against — every pose would be silently misplaced."""
    (anim_run_dir / "scene.json").write_text(
        json.dumps({**_scene_ir("abc123"), "id": "someothersceneffff"}))
    ir = runs_mod.run_detail(anim_run_dir, anim_run_dir.name)["ir"]
    assert ir["animation"]["scene_ref_ok"] is False


def test_animation_endpoint_is_animation_only(anim_run_dir, run_dir):
    app = create_app(runs_dir=run_dir.parent)
    client = TestClient(app)
    assert client.get(f"/api/runs/{anim_run_dir.name}/animation").status_code == 200
    r = client.get(f"/api/runs/{run_dir.name}/animation")
    assert r.status_code == 404
    assert "not an animation run" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Animation runs: the frozen scene lives inside the bundle
#
# An animation run authors its own scene and freezes it at
# `bundle/design_state/`, so nothing lands at the run root and every geometry
# path 404s — on a run whose native clip is perfectly playable — until
# `materialize_scene` stages it.
# ---------------------------------------------------------------------------


def _frozen_design_state(run: Path) -> str:
    """A bundle's frozen design, laid out the way `compile_design` writes one.

    The one structural thing to be real about is the split that motivated the
    `source_root` override: the catalog map sits at `design_state/catalog.json`
    while the assets it names live under `design_state/source/assets/`, so an
    ObjectIR's `assets/meshes/...` does NOT resolve against the map's own
    directory. Returns the scene's content-hash id.
    """
    from dycraft.sdk.ir import content_hash

    design = run / "bundle" / "design_state"
    (design / "source" / "assets" / "meshes").mkdir(parents=True)
    (design / "source" / "assets" / "brep").mkdir(parents=True)
    (design / "source" / "assets" / "meshes" / "deadbeef.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (design / "source" / "assets" / "brep" / "deadbeef.brep").write_text("brep")

    (design / "catalog.json").write_text(json.dumps({"abc123": _ir()}))

    scene = {k: v for k, v in _scene_ir("abc123").items() if k != "id"}
    scene_id = content_hash(scene)
    (design / "scene.json").write_text(json.dumps({**scene, "id": scene_id}))
    return scene_id


@pytest.fixture
def frozen_scene_run_dir(run_dir) -> Path:
    """A committed animation run: a native bundle whose scene is inside it."""
    runs = run_dir.parent
    run = runs / "260101_000000_claude_animation"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({
        "run_id": run.name, "layer": "animation", "model": "test-model",
        "runtime": "claude_cli", "outcome": "committed",
        "prompt": "A widget that moves.",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:40:00+00:00",
    }))
    _frozen_design_state(run)
    # After the design state, so the manifest hashes it exactly as a real
    # `finish()` does — which is also what proves staging leaves it intact.
    _native_bundle(run)
    return run


def test_materialize_catalog_resolves_assets_from_an_explicit_source_root(tmp_path):
    """The override exists because a frozen catalog's assets are one level off.

    Without it `materialize_catalog` resolves `assets/meshes/...` against the
    catalog *file's* directory, which for an animation bundle holds no assets at
    all. Asserting the failure as well as the success is the point: a default
    that silently found the files would make the keyword untestable and the
    next caller would drop it.
    """
    from dycraft.agent.layers import materialize_catalog

    source = tmp_path / "design_state" / "source"
    (source / "assets" / "meshes").mkdir(parents=True)
    (source / "assets" / "brep").mkdir(parents=True)
    (source / "assets" / "meshes" / "deadbeef.obj").write_text("v 0 0 0\n")
    (source / "assets" / "brep" / "deadbeef.brep").write_text("brep")
    catalog_json = tmp_path / "design_state" / "catalog.json"
    catalog_json.write_text(json.dumps({"abc123": _ir()}))

    n = materialize_catalog(catalog_json, tmp_path / "with_root",
                            source_root=source)
    assert n == 1
    assert (tmp_path / "with_root" / "abc123.json").is_file()
    assert (tmp_path / "with_root" / "assets" / "meshes" / "deadbeef.obj").is_file()

    with pytest.raises((FileNotFoundError, SystemExit)):
        materialize_catalog(catalog_json, tmp_path / "without_root")


def test_a_staged_frozen_scene_makes_the_run_drawable(frozen_scene_run_dir):
    """The whole point: before staging the viewer refuses, after it draws.

    Both halves are asserted in one test because the before-state is the bug
    report — a run whose native timeline is fine and whose geometry 404s — and
    splitting them would let a regression fix one and lose the other.
    """
    from dycraft.agent.layers.scene_staging import (
        materialize_scene)

    # Before: a native film still plays from its own track, under a bare URDF
    # with nothing static to draw (an installed film stays like this).
    bare = runs_mod.run_urdf(frozen_scene_run_dir)
    assert '<link name="world"' in bare and "<visual" not in bare
    assert runs_mod.run_animation(frozen_scene_run_dir)["native"]

    staged = materialize_scene(
        frozen_scene_run_dir / "bundle", frozen_scene_run_dir)
    assert staged["skipped"] is False
    assert staged["n_entries"] == 1

    # The catalog is staged as a raw `<id>.json` directory with no
    # manifest.json, which `iter_catalog_ir_paths` reads as a hand-curated
    # catalog. Pinned because the viewer would otherwise silently see nothing.
    assert not (frozen_scene_run_dir / "catalog" / "manifest.json").exists()
    assert scene_mod.load_catalog(frozen_scene_run_dir).keys() == {"abc123"}

    urdf = runs_mod.run_urdf(frozen_scene_run_dir)
    assert 'filename="catalog/assets/meshes/deadbeef.obj"' in urdf

    timeline = runs_mod.run_animation(frozen_scene_run_dir)
    assert timeline["n_frames"] == 3

    # `scene_id` is the field the frontend's "no scene.json" overlay keys on.
    summary = runs_mod.run_detail(frozen_scene_run_dir, frozen_scene_run_dir.name)["ir"]
    assert summary["scene_id"] == staged["scene_ref"]
    assert summary["catalog_resolved"] == summary["catalog_total"] == 2


def test_staging_a_frozen_scene_leaves_the_bundle_manifest_valid(frozen_scene_run_dir):
    """Staging writes at the run root, never inside `bundle/`.

    The manifest hashes every file under `bundle/` and the agent's own
    `finish()` re-validates it, so a staging step that touched the bundle would
    turn a viewer convenience into a corrupted delivery.
    """
    from dycraft.agent.layers.scene_staging import (
        materialize_scene)
    from dycraft.animation.blender.bundle import validate_manifest

    materialize_scene(frozen_scene_run_dir / "bundle", frozen_scene_run_dir)
    ir = json.loads((frozen_scene_run_dir / "ir.json").read_text())
    validate_manifest(frozen_scene_run_dir, ir)


def test_restaging_a_frozen_scene_is_free(frozen_scene_run_dir):
    """Re-running over a staged corpus must not recopy the assets.

    A real scene's `catalog/assets/` is tens of megabytes; a second pass that
    re-hashed and rewrote it would make the tool something you think twice
    about running, which is how corpora stay half-migrated.
    """
    from dycraft.agent.layers.scene_staging import (
        materialize_scene)

    materialize_scene(frozen_scene_run_dir / "bundle", frozen_scene_run_dir)
    asset = frozen_scene_run_dir / "catalog" / "assets" / "meshes" / "deadbeef.obj"
    before = asset.stat().st_mtime_ns

    again = materialize_scene(frozen_scene_run_dir / "bundle", frozen_scene_run_dir)
    assert again["skipped"] is True
    assert asset.stat().st_mtime_ns == before


def test_staging_refuses_to_overwrite_a_different_scene(frozen_scene_run_dir):
    """A `scene.json` naming another scene is not ours to replace.

    That run is either a real `--layer animation` run or a stale staging, and
    quietly swapping the file would change what it draws while every recorded
    `scene_ref` went on naming the old one.
    """
    from dycraft.agent.layers.scene_staging import (
        materialize_scene)

    (frozen_scene_run_dir / "scene.json").write_text(
        json.dumps({"id": "somethingelse", "instances": []}))
    with pytest.raises(SystemExit, match="somethingelse"):
        materialize_scene(frozen_scene_run_dir / "bundle", frozen_scene_run_dir)

    forced = materialize_scene(
        frozen_scene_run_dir / "bundle", frozen_scene_run_dir, force=True)
    assert forced["skipped"] is False
    assert json.loads((frozen_scene_run_dir / "scene.json").read_text())["id"] \
        == forced["scene_ref"]


# ---------------------------------------------------------------------------
# Baked atlas UVs for native playback
#
# The bake unwraps its atlas on a part's STAGED mesh; native playback draws the
# EVALUATED one, which `EDGE_SPLIT` gives a different vertex count. What
# survives is the triangle list — count and corner order — so the sidecar's
# `(3T, 2)` corner UVs pair with evaluated geometry directly. These tests pin
# that pairing, the guard that refuses it when the counts disagree, and the
# "no bake yet" answer, which is empty rather than an error.
# ---------------------------------------------------------------------------


def _fake_bake(monkeypatch, run, *, n_tris, part="body", ir_id="abc123"):
    """Stand in for a completed bake, without running Blender.

    `manifests_for` and `atlas_uvs` are the two seams `atlas()` reaches the
    bake through; faking them keeps these tests about the pairing rather than
    about the bake cache, which `test_viewer_bake.py` already covers.
    """
    from dycraft.viewer import derived as derived_mod
    from dycraft.render import viewer_bake as VB

    # Deliberately flip-sensitive: v = 0.5 is its own mirror, so a fixture
    # built on it cannot tell a correct payload from a vertically mirrored one.
    uv = np.tile(np.array([[0.25, 0.125], [1.0, 0.0], [0.0, 0.75]], dtype="f8"),
                 (n_tris, 1))
    tri = np.arange(3 * n_tris, dtype="i8").reshape(n_tris, 3)
    monkeypatch.setattr(derived_mod, "manifests_for",
                        lambda _run: {ir_id: {"ok": True, "parts": []}})
    monkeypatch.setattr(VB, "atlas_uvs",
                        lambda _id, _m, name: (uv, tri) if name == part else None)
    (run / "scene.json").write_text(json.dumps({
        "id": "scene777", "schema": "dycraft.scene.v1", "instances": [
            {"id": 0, "name": "thing", "object_name": "widget",
             "object_ref": ir_id}]}))
    return uv


def test_a_native_clip_carries_the_bakes_own_corner_uvs(tmp_path, monkeypatch):
    """The payload that lets evaluated geometry wear a baked material.

    `n_corners == 3 * n_tris` is the whole contract: the client attaches the
    array to geometry that `toCreasedNormals` has already de-indexed, so corner
    `i` here is corner `i` there.
    """
    from dycraft.viewer.native_animation import atlas

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run, frames=3)          # one drawable object, 1 triangle
    _fake_bake(monkeypatch, run, n_tris=1)

    header = _header(atlas(run))
    assert header["dtype"] == "u16"
    assert [o["name"] for o in header["objects"]] == ["part/thing/body"]
    assert header["objects"][0]["uv_space"] == "atlas"
    entry = header["atlases"][header["objects"][0]["atlas"]]
    assert entry["n_tris"] == 1 and entry["n_corners"] == 3
    assert entry["key"] == "abc123/body"

    # Round-trip the fixed point: normalized u16 is what three divides by
    # 65535 in the shader, so the values must come back as authored — with V
    # flipped, see the next test.
    blob = atlas(run)
    offset = 4 + int.from_bytes(blob[:4], "little")
    uv = np.frombuffer(blob, dtype="<u2", offset=offset).reshape(-1, 2) / 65535.0
    assert np.allclose(uv, [[0.25, 0.875], [1.0, 1.0], [0.0, 0.25]], atol=1e-4)


def test_a_native_atlas_ships_gltfs_top_down_v_not_blenders(tmp_path,
                                                            monkeypatch):
    """The flip that decides whether a baked texture lands or is mirrored.

    The sidecar holds the layer as Blender stores it, V bottom-up, which is
    right for the *mesh* UV path: those maps load through `TextureLoader`,
    whose `flipY` defaults to true. An atlas map arrives inside a GLB instead,
    and `GLTFLoader` sets `flipY = false` because glTF measures V top-down — so
    the atlas payload has to be in glTF's space, which is exactly the UVs
    Blender's exporter already wrote into the GLB.

    Shipping Blender's V mirrors every island vertically. It still looks like a
    texture, which is why this needs a test rather than an eyeball: on the real
    kitchen it read as oak grain running the wrong way across the floor.
    """
    from dycraft.viewer.native_animation import atlas

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run, frames=3)
    sidecar = _fake_bake(monkeypatch, run, n_tris=1)

    blob = atlas(run)
    offset = 4 + int.from_bytes(blob[:4], "little")
    shipped = np.frombuffer(blob, dtype="<u2", offset=offset).reshape(-1, 2) / 65535.0
    assert np.allclose(shipped[:, 0], sidecar[:, 0], atol=1e-4), "U must not move"
    assert np.allclose(shipped[:, 1], 1.0 - sidecar[:, 1], atol=1e-4)


def test_a_native_atlas_skips_an_object_whose_triangles_disagree(tmp_path,
                                                                 monkeypatch):
    """The only check available on this path, so it has to hold.

    For the deformation path `_corner_uv_source` compares index arrays
    elementwise, because there the mesh IS the staged rest mesh. An evaluated
    mesh's indices differ by construction, so triangle count is all there is —
    and a sidecar that disagrees must be dropped, not stretched over the mesh.
    """
    from dycraft.viewer.native_animation import atlas

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run, frames=3)          # the object has 1 triangle
    _fake_bake(monkeypatch, run, n_tris=7)  # the bake claims 7

    header = _header(atlas(run))
    assert header["objects"] == [] and header["atlases"] == []


def test_a_native_atlas_is_empty_rather_than_absent_without_a_bake(tmp_path):
    """A run with no bake and no staged scene is the ordinary case.

    The client asks unconditionally and falls back to the track's flat color,
    so this must be a valid empty payload — a 404 would put an error in the
    notes strip for a run that is behaving normally.
    """
    from dycraft.viewer.native_animation import atlas

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run, frames=3)

    header = _header(atlas(run))
    assert header["objects"] == [] and header["atlases"] == []
    assert len(atlas(run)) == 4 + int.from_bytes(atlas(run)[:4], "little")


def test_two_instances_of_one_part_share_a_single_atlas(tmp_path, monkeypatch):
    """Deduplicated by `(ir_id, part)`, which is what an atlas belongs to.

    An object is an instance of a catalog part, so two stools' frames are one
    bake and one UV array. Emitting per object sent 25.5 MB for the kitchen
    where 12.5 MB is distinct.
    """
    from dycraft.viewer.native_animation import atlas

    run = tmp_path / "run"
    run.mkdir()
    _native_bundle(run, frames=3)
    _fake_bake(monkeypatch, run, n_tris=1)

    # A second object naming the same identity — the shape two instances of one
    # catalog object take in a frame record.
    evaluation = run / "bundle" / "evaluation"
    for frame in range(3):
        path = evaluation / f"{frame:06d}.json"
        record = json.loads(path.read_text())
        twin = dict(record["objects"][0])
        twin["key"], twin["name"] = "o2", "part/thing/body_twin"
        record["objects"].append(twin)
        path.write_text(json.dumps(record))
    from dycraft.animation.blender.bundle import make_manifest
    ir = make_manifest(run / "bundle", scene_ref="scene", evaluation={
        "fps": 24.0, "n_frames": 3, "frame_start": 0,
        "blender_version": "test", "declarations": {"events": []}})
    (run / "ir.json").write_text(json.dumps(ir))

    header = _header(atlas(run))
    assert len(header["objects"]) == 2
    assert len(header["atlases"]) == 1
    assert {o["atlas"] for o in header["objects"]} == {0}


def test_the_clip_tab_finds_the_film_the_manifest_names(tmp_path):
    """An animation run commits its film as `bundle/film/clip.mp4` and names it
    in `ir.json`; the clip tab serves that path and nothing outside the run."""
    import json
    from dycraft.viewer import animation as A

    run = tmp_path / "scene"
    (run / "bundle" / "film").mkdir(parents=True)
    (run / "bundle" / "film" / "clip.mp4").write_bytes(b"mp4")
    (run / "ir.json").write_text(json.dumps({"film": "bundle/film/clip.mp4"}))
    assert A.clip_rel(run) == "bundle/film/clip.mp4"

    # A manifest path outside the run is never served, and no film is None.
    outside = tmp_path / "escape"
    outside.mkdir()
    (tmp_path / "secret.mp4").write_bytes(b"x")
    (outside / "ir.json").write_text(json.dumps({"film": "../secret.mp4"}))
    assert A.clip_rel(outside) is None
    assert A.clip_rel(tmp_path / "missing") is None


def test_the_sidebar_badge_carries_a_codex_runs_sub_agent_tokens():
    from dycraft.viewer.runs import _tokens
    meta = {"usage": {"gross_input": 1000, "output": 100, "turns": 5,
                      "subagents": {"total": 900, "threads": 2, "turns": 7},
                      "total_with_subagents": 2000}}
    assert _tokens(meta) == {"gross_input": 1000, "output": 100, "turns": 5,
                             "subagents": {"total": 900, "threads": 2}}
    assert "subagents" not in _tokens({"usage": {"gross_input": 1, "output": 1}})


def _frames_project(tmp_path, events, **index):
    """A committed animation run whose declarations are in output frames."""
    fps = 24.0
    evaluation = "bundle/evaluation/index.json"
    index = {"fps": fps, "n_frames": 90, "frame_start": 1,
             "declarations": {"units": "frames", "events": events}, **index}
    write_json(tmp_path / evaluation, index)
    ir = {"schema": "dycraft.animation.blender.v1", "id": "animation", "scene_ref": "scene",
          "fps": fps, "n_frames": 90, "duration": 89/fps, "events": events,
          "evaluation": evaluation, "files": {evaluation: file_hash(tmp_path / evaluation)}}
    return ir, index


def test_event_windows_in_frames_are_placed_on_frames(tmp_path):
    """An animation project writes event windows in output frames, or as
    offsets into a clip. Without a report they are placed the way the verifier
    places them, from the evaluation index the bundle carries: an absolute
    window on its frames, a clip-relative one from the clip's span and scale,
    never frames read as seconds."""
    from dycraft.viewer.animation import summary
    events = [{"name": "lands", "predicate": "touch", "window": [10, 12]},
              {"name": "clears", "predicate": "separated", "clip": "ending", "window": [2, 5]},
              {"name": "nowhere", "predicate": "touch", "clip": "missing", "window": [0, 1]}]
    clips = {"ending": {"frames": [60, 89], "scales": [2.0], "starts": [1.0]}}
    ir, _ = _frames_project(tmp_path, events, clips=clips)
    placed = {e["name"]: e["window_frames"] for e in summary(tmp_path, ir, {"id": "scene"})["events"]}
    assert placed == {"lands": [10, 12], "clears": [62, 68], "nowhere": None}


def test_events_that_share_a_name_each_get_their_own_verdict(tmp_path):
    """Event names may repeat, one per clip, and the report's rows carry no
    clip: each event takes the row that resolved its window to the same
    frames, never another event's verdict."""
    from dycraft.viewer.animation import summary
    events = [{"name": "pin", "predicate": "touch", "clip": "first", "window": [0, 4]},
              {"name": "pin", "predicate": "touch", "clip": "second", "window": [0, 4]}]
    clips = {"first": {"frames": [0, 29]}, "second": {"frames": [40, 69]}}
    ir, index = _frames_project(tmp_path, events, clips=clips)
    fps = ir["fps"]
    write_json(tmp_path / "compile_reports/latest.json", {
        "version": "v", "evaluation": index,
        "verifier_reports": [{"name": "EventsSatisfied", "evidence": {"events": [
            {"name": "pin", "window": [0.0, 4.0], "window_frames": [40, 44], "ok": True,
             "satisfied_at": 42/fps},
            {"name": "pin", "window": [0.0, 4.0], "window_frames": [0, 4], "ok": False,
             "satisfied_at": None}]}}]})
    placed = [(e["window_frames"], e["satisfied_frame"])
              for e in summary(tmp_path, ir, {"id": "scene"})["events"]]
    assert placed == [([0, 4], None), ([40, 44], 42)]


@pytest.mark.parametrize("stale", [False, True])
def test_an_animation_run_s_report_is_matched_by_its_evaluation(tmp_path, stale):
    """An animation run's compile report carries no IR hash; it carries the
    evaluation it checked, which is the bundle's own index when it is the
    check the commit used. Matched that way its verdicts reach the scrubber;
    a report of a different evaluation is ignored."""
    from dycraft.viewer.animation import summary
    events = [{"name": "lands", "predicate": "touch", "window": [10, 12]}]
    ir, index = _frames_project(tmp_path, events)
    fps = ir["fps"]
    write_json(tmp_path / "compile_reports/latest.json", {
        "version": "v", "evaluation": {**index, "frames": [0]} if stale else index,
        "verifier_reports": [{"name": "EventsSatisfied", "evidence": {"events": [
            {"name": "lands", "window": [10.0, 12.0], "window_frames": [10, 12], "ok": True,
             "satisfied_at": 11/fps}]}}]})
    [event] = summary(tmp_path, ir, {"id": "scene"})["events"]
    assert event["window_frames"] == [10, 12]
    assert event["satisfied_frame"] == (None if stale else 11)


@pytest.mark.parametrize("change", [None, "outside_the_index", "declarations"])
def test_a_report_with_signatures_must_also_agree_on_the_declarations(tmp_path, change):
    """With per-frame signatures, the report is the commit's check when they
    match and every index field does too. The signatures do not cover the
    declarations, and `latest.json` is written only by a compile, so a commit
    after a declarations-only edit leaves a report whose signatures still
    match; its windows and verdicts are not this film's."""
    from dycraft.viewer.animation import summary
    events = [{"name": "lands", "predicate": "touch", "window": [40, 44]}]
    ir, index = _frames_project(tmp_path, events, signatures={"0": "a", "1": "b"})
    evaluation = dict(index)
    if change == "outside_the_index":
        evaluation["frames"] = [0, 1]
    elif change == "declarations":
        evaluation["declarations"] = {"units": "frames", "events": [
            {"name": "lands", "predicate": "touch", "window": [10, 12]}]}
    write_json(tmp_path / "compile_reports/latest.json", {
        "version": "v", "evaluation": evaluation,
        "verifier_reports": [{"name": "EventsSatisfied", "evidence": {"events": [
            {"name": "lands", "window": [10.0, 12.0] if change == "declarations" else [40.0, 44.0],
             "window_frames": [10, 12] if change == "declarations" else [40, 44],
             "ok": True, "satisfied_at": (11 if change == "declarations" else 42) / ir["fps"]}]}}]})
    [event] = summary(tmp_path, ir, {"id": "scene"})["events"]
    assert event["window_frames"] == [40, 44]
    assert event["satisfied_frame"] == (None if change == "declarations" else 42)


def test_a_rejected_declaration_takes_its_own_row(tmp_path):
    """The verifier's row for a declaration it rejected carries the
    declaration's index and no frames; it goes to that declaration, and the
    other event of the same name keeps its own verdict rather than both
    sharing it."""
    from dycraft.viewer.animation import summary
    events = [{"name": "pin", "predicate": "touch", "a": "nowhere", "window": [10, 12]},
              {"name": "pin", "predicate": "touch", "window": [10, 12]}]
    ir, index = _frames_project(tmp_path, events)
    write_json(tmp_path / "compile_reports/latest.json", {
        "version": "v", "evaluation": index,
        "verifier_reports": [{"name": "EventsSatisfied", "evidence": {"events": [
            {"name": "pin", "ok": False, "status": "invalid", "index": 0, "reason": "unknown"},
            {"name": "pin", "window": [10.0, 12.0], "window_frames": [10, 12], "ok": True,
             "satisfied_at": 11/ir["fps"]}]}}]})
    assert [e["satisfied_frame"] for e in summary(tmp_path, ir, {"id": "scene"})["events"]] == [None, 11]


@pytest.mark.parametrize("stale", [False, True])
def test_native_events_agree_in_overview_and_timeline(tmp_path, stale):
    from dycraft.viewer.animation import summary, timeline
    fps = 29.97
    events = [{"name": name, "predicate": "touch", "window": [0, 3], "mode": mode}
              for name, mode in (("any", "any"), ("all", "all"), ("failed", "all"),
                                 ("deferred", "all"), ("missing", "any"))]
    evaluation = "bundle/evaluation/index.json"
    write_json(tmp_path / evaluation, {})
    ir = {"schema": "dycraft.animation.blender.v1", "id": "animation", "scene_ref": "scene",
          "fps": fps, "n_frames": 90, "duration": 89/fps, "events": events,
          "evaluation": evaluation, "files": {evaluation: file_hash(tmp_path / evaluation)}}
    # The verifier resolves `satisfied_at` itself now, so the timeline reads
    # one number per event instead of zipping parallel per-frame arrays.
    rows = [
        {"name": "any", "window": [60/fps, 61/fps], "ok": True, "satisfied_at": 61/fps},
        {"name": "all", "window": [60/fps, 61/fps], "ok": True, "satisfied_at": 60/fps},
        {"name": "failed", "window": [60/fps, 61/fps], "ok": False, "satisfied_at": None,
         "unsatisfied": [[61, 61]], "unsatisfied_count": 1},
        {"name": "deferred", "window": [60/fps, 61/fps], "deferred": True},
    ]
    write_json(tmp_path / "compile_reports/latest.json", {
        "ir_hash": "old" if stale else ir["id"],
        "verifier_reports": [{"name": "EventsSatisfied", "evidence": {"events": rows}}]})
    overview = summary(tmp_path, ir, {"id": "scene"})["events"]
    playback = timeline(tmp_path, ir)["events"]
    assert overview == playback
    # A film that plays as it should has no standing notes, so no notes strip shows under the player.
    assert timeline(tmp_path, ir)["notes"] == []
    assert [e["satisfied_frame"] for e in playback] == (
        [None] * 5 if stale else [61, 60, None, None, None])
    if not stale:
        assert playback[0]["satisfied_at"] == pytest.approx(61/fps)


@pytest.mark.parametrize("value, expected", [(None, 1024), ("4096", 4096), ("16", 64), ("lots", 1024)])
def test_the_clip_budget_is_the_operators_to_set(tmp_path, monkeypatch, value, expected):
    """The page keeps at most this much built native-clip geometry (`native-stream.ts`). A machine with memory to
    spare raises it with DYCRAFT_VIEWER_CLIP_BUDGET_MB; a value too small to hold a frame, or not a number, is not
    taken."""
    from dycraft.viewer.app import CLIP_BUDGET_ENV

    if value is None:
        monkeypatch.delenv(CLIP_BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(CLIP_BUDGET_ENV, value)
    client = TestClient(create_app(runs_dir=tmp_path, dist_dir=Path("/nonexistent")))
    assert client.get("/api/config").json() == {"clip_budget_mb": expected}


def test_the_native_manifest_is_parsed_once_and_again_when_it_changes(tmp_path):
    """Every native-mesh request needs the manifest, and parsing it each time would keep the server below a long
    film's playback rate. It is cached by the file's mtime and size, so a rewritten ir.json is read again, and a run
    that is not a native clip is still refused."""
    import os

    from dycraft.animation.blender import SCHEMA
    from dycraft.viewer import native_animation as NA

    ir = tmp_path / "ir.json"
    ir.write_text(json.dumps({"schema": SCHEMA, "id": "a"}))
    first = NA.manifest(tmp_path)
    assert first["id"] == "a" and NA.manifest(tmp_path) is first
    ir.write_text(json.dumps({"schema": SCHEMA, "id": "bb"}))
    os.utime(ir, ns=(ir.stat().st_atime_ns, ir.stat().st_mtime_ns + 1_000_000))
    assert NA.manifest(tmp_path)["id"] == "bb"
    ir.write_text(json.dumps({"schema": "dycraft.scene.v1"}))
    os.utime(ir, ns=(ir.stat().st_atime_ns, ir.stat().st_mtime_ns + 2_000_000))
    with pytest.raises(ValueError, match="no native animation bundle"):
        NA.manifest(tmp_path)
    with pytest.raises(ValueError, match="no native animation bundle"):
        NA.manifest(tmp_path / "missing")
