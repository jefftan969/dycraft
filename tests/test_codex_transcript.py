"""Codex transcript capture + rollout parsing, at parity with the claude path.

Two halves, both offline:

  1. Discovery. Codex files its session under a *date* tree, so the run→session
     join is `session_meta.payload.cwd` inside the file rather than a munged
     path. Covers the account precedence, the newest-wins tie-break, and that a
     rollout belonging to a different run dir is never staged.
  2. Parsing. `viewer/trajectory._from_codex_transcript` has to fold
     `response_item` records into turns delimited by usage events, join tool
     outputs by `call_id`, and convert OpenAI-vocabulary usage (gross input,
     cached as a subset) into the viewer's billing split.

The fixtures here are synthetic but schema-faithful to real rollouts written by
codex 0.146/0.147 (`event_msg/token_count`) and 0.153+ (`token_usage_record`).
No live codex call is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.agent.transcript_capture import (
    TRANSCRIPT_NAME,
    capture,
    find_any_external_transcript,
    find_codex_transcript,
)
from dycraft.viewer.trajectory import _from_codex_transcript, _parse_cli_transcript


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _rollout(cwd: str, *, records: list[dict] | None = None,
             cli_version: str = "0.153.4") -> str:
    head = {"timestamp": "2026-09-06T09:00:00.000Z", "type": "session_meta",
            "payload": {"session_id": "s-1", "cwd": cwd,
                        "originator": "codex-tui", "cli_version": cli_version,
                        "source": "cli", "model_provider": "openai"}}
    lines = [head] + (records or [])
    return "".join(json.dumps(r) + "\n" for r in lines)


def _write_rollout(home: Path, name: str, body: str, *,
                   day: str = "2026/09/06") -> Path:
    target = home / "sessions" / day / f"rollout-{name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    return target


def _run_dir(tmp_path: Path, home: Path, *, runtime: str = "codex_cli") -> Path:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(
        {"runtime": runtime, "codex_home": str(home)}))
    return run_dir


def _turn(model: str, effort: str, *, text: str, cmd: str,
          usage: dict, call_id: str = "c1", legacy: bool = False) -> list[dict]:
    """One model response: context, assistant text, a tool call + output, usage."""
    usage_event = (
        {"timestamp": "2026-09-06T09:00:05.000Z", "type": "event_msg",
         "payload": {"type": "token_count",
                     "info": {"last_token_usage": usage}}}
        if legacy else
        {"timestamp": "2026-09-06T09:00:05.000Z", "type": "token_usage_record",
         "payload": {"usage": usage}}
    )
    return [
        {"type": "turn_context", "payload": {"model": model, "effort": effort}},
        {"timestamp": "2026-09-06T09:00:01.000Z", "type": "response_item",
         "payload": {"type": "reasoning", "summary": [],
                     "encrypted_content": "opaque"}},
        {"timestamp": "2026-09-06T09:00:02.000Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": text}]}},
        {"timestamp": "2026-09-06T09:00:03.000Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "exec_command",
                     "call_id": call_id,
                     "arguments": json.dumps({"cmd": cmd})}},
        {"timestamp": "2026-09-06T09:00:04.000Z", "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": call_id,
                     "output": "Process exited with code 0"}},
        usage_event,
    ]


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

def test_finds_the_rollout_whose_cwd_is_the_run_dir(tmp_path):
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    mine = _write_rollout(home, "mine", _rollout(str(run_dir.resolve())))
    _write_rollout(home, "elsewhere", _rollout("/some/other/place"))
    assert find_codex_transcript(run_dir) == mine


def test_a_rollout_for_another_run_dir_is_never_staged(tmp_path):
    """As on the claude path, a sibling run's session must not be mistaken for
    this run's just because it is the only one in the tree."""
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    _write_rollout(home, "sibling", _rollout(str(tmp_path / "runs" / "r2")))
    assert find_codex_transcript(run_dir) is None
    assert capture(run_dir) == "unavailable"
    assert not (run_dir / TRANSCRIPT_NAME).exists()


def test_newest_rollout_wins_when_a_run_dir_was_relaunched(tmp_path):
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    cwd = str(run_dir.resolve())
    old = _write_rollout(home, "old", _rollout(cwd), day="2026/09/05")
    new = _write_rollout(home, "new", _rollout(cwd), day="2026/09/06")
    import os
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert find_codex_transcript(run_dir) == new


def test_recorded_home_outranks_the_readers_environment(tmp_path, monkeypatch):
    """The viewer runs in a different shell than the run did, so what the run
    recorded has to win over ambient CODEX_HOME."""
    real = tmp_path / "real_home"
    decoy = tmp_path / "decoy_home"
    run_dir = _run_dir(tmp_path, real)
    cwd = str(run_dir.resolve())
    mine = _write_rollout(real, "mine", _rollout(cwd))
    _write_rollout(decoy, "decoy", _rollout(cwd))
    monkeypatch.setenv("CODEX_HOME", str(decoy))
    assert find_codex_transcript(run_dir) == mine


def test_dispatch_follows_the_recorded_runtime(tmp_path):
    """A codex run must not be searched for down the claude project tree."""
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    mine = _write_rollout(home, "mine", _rollout(str(run_dir.resolve())))
    assert find_any_external_transcript(run_dir) == mine


def test_capture_stages_the_rollout_into_the_bundle(tmp_path):
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    body = _rollout(str(run_dir.resolve()))
    _write_rollout(home, "mine", body)
    assert capture(run_dir) == "copied"
    assert (run_dir / TRANSCRIPT_NAME).read_text() == body


def test_a_missing_sessions_tree_is_not_an_error(tmp_path):
    home = tmp_path / "codex_home"
    home.mkdir()
    run_dir = _run_dir(tmp_path, home)
    assert find_codex_transcript(run_dir) is None
    assert capture(run_dir) == "unavailable"


# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------

USAGE_1 = {"input_tokens": 30000, "cached_input_tokens": 9600,
           "cache_write_input_tokens": 0, "output_tokens": 535,
           "reasoning_output_tokens": 311, "total_tokens": 30535}
USAGE_2 = {"input_tokens": 31000, "cached_input_tokens": 29056,
           "cache_write_input_tokens": 0, "output_tokens": 272,
           "reasoning_output_tokens": 100, "total_tokens": 31272}


@pytest.mark.parametrize("legacy", [False, True],
                         ids=["token_usage_record", "token_count"])
def test_both_rollout_generations_parse(tmp_path, legacy):
    """0.146/0.147 report usage as `event_msg/token_count`; 0.153+ adds
    `token_usage_record`. Either has to close a turn."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="Plan.", cmd="ls",
               usage=USAGE_1, legacy=legacy),
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert result["source"] == "codex_cli"
    assert len(result["turns"]) == 1
    assert result["turns"][0]["model"] == "gpt-5.6-sol"
    assert result["turns"][0]["effort"] == "high"


def test_openai_usage_vocabulary_is_not_double_counted(tmp_path):
    """`input_tokens` is gross and `cached_input_tokens` is a subset of it.

    Carrying it through as the viewer's `input` would bill the cache twice, so
    `gross_input` must come back equal to the raw figure and `input` must be
    the uncached remainder.
    """
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
    ]))
    usage = _from_codex_transcript(path, tmp_path)["turns"][0]["usage"]
    assert usage["gross_input"] == USAGE_1["input_tokens"]
    assert usage["cache_read"] == USAGE_1["cached_input_tokens"]
    assert usage["input"] == 30000 - 9600
    assert usage["output"] == USAGE_1["output_tokens"]


def test_a_repeated_usage_event_does_not_add_an_empty_turn(tmp_path):
    """Codex emits a duplicate `token_count` at task end. Folding it in as its
    own turn would inflate the turn count and double-count its tokens."""
    dup = {"timestamp": "2026-09-06T09:00:06.000Z", "type": "event_msg",
           "payload": {"type": "token_count",
                       "info": {"last_token_usage": USAGE_1}}}
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls",
               usage=USAGE_1, legacy=True),
        dup,
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert len(result["turns"]) == 1
    assert result["turns"][0]["usage"]["output"] == USAGE_1["output_tokens"]


def test_tool_outputs_join_their_call_by_call_id(tmp_path):
    """Codex records the call and its output as separate records, unlike
    claude's nested tool_result blocks."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="echo hi",
               usage=USAGE_1, call_id="call_abc"),
    ]))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["name"] == "exec_command"
    assert call["result"] == "Process exited with code 0"
    assert call["is_error"] is False


def test_harness_ops_are_read_out_of_the_shell_command(tmp_path):
    """The DyCraft op is only visible inside the command text, and codex's
    tool name is lowercase — which the claude-side detector would mistake for
    an in-process op name."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A",
               cmd='bash -c \'"$DYCRAFT_PYTHON" -c "import harness as h; h.compile()"\'',
               usage=USAGE_1),
    ]))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["dycraft_op"] == "compile"


def test_a_non_shell_tool_reports_no_harness_op(tmp_path):
    """`apply_patch`'s body can *mention* an op; only shell tools run one."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        {"type": "turn_context", "payload": {"model": "m", "effort": "high"}},
        {"timestamp": "2026-09-06T09:00:03.000Z", "type": "response_item",
         "payload": {"type": "custom_tool_call", "name": "apply_patch",
                     "call_id": "p1",
                     "input": "*** Begin Patch\n+# calls h.compile() later\n"}},
        {"timestamp": "2026-09-06T09:00:05.000Z", "type": "token_usage_record",
         "payload": {"usage": USAGE_1}},
    ]))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["name"] == "apply_patch"
    assert call["dycraft_op"] is None


def test_model_and_effort_carry_forward_across_turns(tmp_path):
    """`turn_context` is written when something changes, not every turn."""
    second = _turn("gpt-5.6-sol", "high", text="B", cmd="ls", usage=USAGE_2)
    second = [r for r in second if r.get("type") != "turn_context"]
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
        *second,
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert len(result["turns"]) == 2
    assert [t["model"] for t in result["turns"]] == ["gpt-5.6-sol"] * 2
    assert [t["effort"] for t in result["turns"]] == ["high"] * 2
    assert result["model"] == "gpt-5.6-sol"


def test_content_after_the_last_usage_event_is_still_reported(tmp_path):
    """A SIGTERM'd run leaves a response with no closing usage event. Dropping
    it would silently lose the work; zeroed tokens are the honest answer."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
        {"timestamp": "2026-09-06T09:00:07.000Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "cut off"}]}},
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert len(result["turns"]) == 2
    assert result["turns"][1]["text"] == "cut off"
    assert result["turns"][1]["usage"]["total"] == 0


def test_reasoning_is_reported_unavailable_with_a_reason(tmp_path):
    """Codex persists `encrypted_content` with an empty summary, so there is no
    reasoning text to show — the same ceiling as claude's signature-only
    thinking blocks. Say so rather than implying the model didn't think."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert result["reasoning_available"] is False
    assert "encrypted" in result["reasoning_note"]


def test_developer_and_user_messages_are_not_counted_as_assistant_text(tmp_path):
    """The system prompt reaches codex as a developer message in the rollout;
    echoing it back as the agent's first turn would be wrong."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        {"timestamp": "2026-09-06T09:00:00.500Z", "type": "response_item",
         "payload": {"type": "message", "role": "developer",
                     "content": [{"type": "input_text", "text": "SYSTEM"}]}},
        {"timestamp": "2026-09-06T09:00:00.600Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "TASK"}]}},
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
    ]))
    result = _from_codex_transcript(path, tmp_path)
    assert len(result["turns"]) == 1
    assert result["turns"][0]["text"] == "A"


def test_dispatcher_sniffs_the_schema_rather_than_trusting_meta(tmp_path):
    """A codex rollout handed to the claude parser yields zero turns, which
    renders like a real-but-empty run instead of like a bug — so the format is
    detected from the content, which always opens with `session_meta`."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=[
        *_turn("gpt-5.6-sol", "high", text="A", cmd="ls", usage=USAGE_1),
    ]))
    assert _parse_cli_transcript(path, tmp_path)["source"] == "codex_cli"

    claude_like = tmp_path / "c.jsonl"
    claude_like.write_text(json.dumps({
        "type": "assistant",
        "message": {"id": "m1", "model": "claude-opus-5",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2}},
    }) + "\n")
    assert _parse_cli_transcript(claude_like, tmp_path)["source"] == "claude_cli"


# ---------------------------------------------------------------------------
# 3. codex 0.153.4 shapes
# ---------------------------------------------------------------------------
#
# codex 0.153.4 writes three shapes the 0.146/0.147 rollouts do not, and a
# parser expecting only the older ones silently drops data on each:
#
#   1. the per-response usage record lands BETWEEN a tool call and its output;
#   2. the shell tool is a `custom_tool_call` named `exec` whose `input` is JS
#      source calling `tools.exec_command({cmd:…})`, not a JSON argument object,
#      and the command may be single-, double- or BACKTICK-quoted;
#   3. `custom_tool_call_output.output` is a content-block list.

def _js(cmd: str, *, quote: str = '"') -> str:
    """A 0.153.4 `exec` cell body wrapping `cmd` as a JS string literal."""
    if quote == "`":
        body = cmd
    else:
        body = (cmd.replace("\\", "\\\\").replace(quote, "\\" + quote)
                   .replace("\n", "\\n"))
    return f'text(await tools.exec_command({{cmd:{quote}{body}{quote}}}));'


def _v153_turn(cmd: str, *, usage: dict, call_id: str = "call_1",
               text: str | None = None, quote: str = '"',
               output: str = "Script completed\nOutput:\nok") -> list[dict]:
    """One 0.153.4 response, in the real record order.

    Note the usage record sits between the call and its output — that is the
    ordering, not a quirk of this fixture.
    """
    records: list[dict] = [
        {"type": "turn_context", "payload": {"model": "gpt-6-astra",
                                             "effort": "medium"}},
        {"timestamp": "2026-09-06T11:31:40.000Z", "type": "response_item",
         "payload": {"type": "reasoning", "summary": [],
                     "encrypted_content": "gAAAAA"}},
    ]
    if text is not None:
        records.append(
            {"timestamp": "2026-09-06T11:31:41.000Z", "type": "response_item",
             "payload": {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": text}]}})
    records += [
        {"timestamp": "2026-09-06T11:31:42.000Z", "type": "response_item",
         "payload": {"type": "custom_tool_call", "name": "exec",
                     "status": "completed", "call_id": call_id,
                     "input": _js(cmd, quote=quote)}},
        # <-- the response ends here, before the tool has run
        {"timestamp": "2026-09-06T11:31:43.000Z", "type": "token_usage_record",
         "payload": {"response_id": f"resp_{call_id}", "usage": usage}},
        {"timestamp": "2026-09-06T11:31:48.000Z", "type": "response_item",
         "payload": {"type": "custom_tool_call_output", "call_id": call_id,
                     "output": [{"type": "input_text", "text": output}]}},
        # 0.153.4 emits a second usage event for the same response.
        {"timestamp": "2026-09-06T11:31:49.000Z", "type": "event_msg",
         "payload": {"type": "token_count",
                     "info": {"last_token_usage": usage}}},
    ]
    return records


HARNESS_CMD = ('bash -c \'cd "$DYCRAFT_RUN_DIR" && "$DYCRAFT_PYTHON" -c '
               '"import harness as h; h.compile()"\'')


def test_tool_output_arriving_after_the_usage_record_still_attaches(tmp_path):
    """A tool's output still attaches to its call when it lands after the turn closed.

    The per-response usage record closes the turn before the tool has
    finished, so a per-turn call-id map is already cleared when the output
    lands. The map has to outlive the turn.
    """
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        HARNESS_CMD, usage=USAGE_1, text="Building it.")))
    turns = _from_codex_transcript(path, tmp_path)["turns"]
    assert len(turns) == 1
    call = turns[0]["tool_calls"][0]
    assert call["result"] == "Script completed\nOutput:\nok"


def test_content_block_list_output_is_flattened(tmp_path):
    """0.153.4 hands back `[{"type": "input_text", "text": …}]`, which the
    string/dict-only branch mapped to None."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        "ls", usage=USAGE_1, output="line one\nline two")))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["result"] == "line one\nline two"


@pytest.mark.parametrize("quote", ['"', "'", "`"],
                         ids=["double", "single", "backtick"])
def test_harness_op_is_read_out_of_the_js_cell(tmp_path, quote):
    """The command is JS source, not JSON.

    The backtick case is not hypothetical: a harness call `"$DYCRAFT_PYTHON"
    -c '…'` contains both quote kinds, so codex emits it as a template literal,
    and without the backtick case every such call is missed.
    """
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        HARNESS_CMD, usage=USAGE_1, quote=quote)))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["name"] == "exec"
    assert call["dycraft_op"] == "compile"


def test_two_usage_events_per_response_produce_one_turn(tmp_path):
    """0.153.4 emits `token_usage_record` and then `token_count` for the same
    response. Counting both would double every token in the run."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        "ls", usage=USAGE_1)))
    result = _from_codex_transcript(path, tmp_path)
    assert len(result["turns"]) == 1
    assert result["turns"][0]["usage"]["output"] == USAGE_1["output_tokens"]
    assert result["turns"][0]["usage"]["gross_input"] == USAGE_1["input_tokens"]


# --- heredoc / authored-source false positives ------------------------------

def test_a_cell_that_only_writes_source_mentioning_an_op_is_not_a_call(tmp_path):
    """`cat > object.py <<'PY' … h.compile() … PY` invokes nothing.

    This is the hazard `_SHELL_TOOLS`' comment records for the claude path (a
    context summary that *mentions* `h.author_solids(`), and 0.153.4 makes it
    common: the agent writes `object.py` through the same `exec` cell it uses
    to drive the harness.
    """
    cmd = ("cat > object.py <<'PY'\n"
           "# authored source; the harness call below is a comment\n"
           "# h.compile() would be run later by the agent\n"
           "PY")
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(cmd, usage=USAGE_1)))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["dycraft_op"] is None


def test_a_context_summary_quoting_an_op_is_not_a_call(tmp_path):
    cmd = ('cat > context_summary.md <<\'MD\'\n'
           'We repaired the part and then called h.finish() to commit.\n'
           'MD')
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(cmd, usage=USAGE_1)))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["dycraft_op"] is None


def test_a_cell_that_writes_source_then_runs_the_harness_reports_the_real_op(
        tmp_path):
    """The discriminating case: both patterns in one command.

    A plain search returns the op the heredoc *mentions* (`compile`) rather
    than the one actually invoked (`probe`), so detection anchors on the
    harness invocation and only looks after it.
    """
    cmd = ("cat > object.py <<'PY'\n"
           "# h.compile() appears here as authored text only\n"
           "PY\n"
           'bash -c \'"$DYCRAFT_PYTHON" -c "import harness as h; '
           'h.probe(\\"bbox\\")"\'')
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(cmd, usage=USAGE_1)))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["dycraft_op"] == "probe"


def test_a_plain_shell_cell_reports_no_op(tmp_path):
    """`cd "$DYCRAFT_RUN_DIR" && cat manifest.json` touches no harness — and
    `$DYCRAFT_RUN_DIR` must not be mistaken for `$DYCRAFT_PYTHON`."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        'cd "$DYCRAFT_RUN_DIR" && cat manifest.json', usage=USAGE_1)))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["dycraft_op"] is None


# --- error reporting --------------------------------------------------------

def test_an_exit_code_in_the_output_marks_the_call_as_failed(tmp_path):
    """0.146/0.147 report the shell's status in the output header."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        "false", usage=USAGE_1,
        output="Chunk ID: ab\nWall time: 0.0s\nProcess exited with code 1")))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["is_error"] is True


def test_an_unreported_status_is_not_guessed_from_the_body(tmp_path):
    """0.153.4 carries no per-call exit status, and a compile report
    legitimately contains the word "error". Marking that a failure flagged 35
    of 106 real microwave calls, so absence stays False rather than becoming a
    text heuristic."""
    path = tmp_path / "t.jsonl"
    path.write_text(_rollout("/run", records=_v153_turn(
        HARNESS_CMD, usage=USAGE_1,
        output=("Script completed\nOutput:\n"
                '{"ok": true, "build_error": null, "verifier_reports": '
                '[{"name": "Watertight", "verdict": "pass"}]}\n'
                "Traceback mentioned in a passing report"))))
    call = _from_codex_transcript(path, tmp_path)["turns"][0]["tool_calls"][0]
    assert call["is_error"] is False
    assert "verifier_reports" in call["result"]


# ---------------------------------------------------------------------------
# 4. Rename-proof rediscovery via the recorded session id
# ---------------------------------------------------------------------------

def test_discovery_records_the_session_id(tmp_path):
    """The cwd join is the only link available on the first lookup, so the id
    is captured then — that is what makes later lookups rename-proof."""
    import json as _json
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    _write_rollout(home, "2026-09-06T11-31-30-sess-abc",
                   _rollout(str(run_dir.resolve())))
    assert find_codex_transcript(run_dir) is not None
    meta = _json.loads((run_dir / "meta.json").read_text())
    assert meta["codex_session_id"] == "s-1"


def test_a_renamed_run_dir_is_still_matched_by_session_id(tmp_path):
    """A run launched under one `--exp-name` and renamed afterwards has a
    rollout whose `cwd` names no directory that exists. With only the cwd join
    the session is orphaned; with a recorded id it is found regardless of the
    directory name.
    """
    import json as _json
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    rollout = _write_rollout(home, "2026-09-06T11-31-30-s-1",
                             _rollout(str(run_dir.resolve())))
    assert find_codex_transcript(run_dir) == rollout      # records the id

    renamed = run_dir.parent / "renamed_after_the_fact"
    run_dir.rename(renamed)
    meta = _json.loads((renamed / "meta.json").read_text())
    assert meta["codex_session_id"] == "s-1"
    # cwd no longer matches anything, but the id still does.
    assert _rollout_cwd_of(rollout) != str(renamed.resolve())
    assert find_codex_transcript(renamed) == rollout


def _rollout_cwd_of(path):
    from dycraft.agent.transcript_capture import _rollout_cwd
    return _rollout_cwd(path)


def test_the_session_id_never_matches_another_runs_rollout(tmp_path):
    """An id is only authoritative because it is unique — a recorded id that
    is absent must not fall back onto a sibling run's session."""
    import json as _json
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    meta = _json.loads((run_dir / "meta.json").read_text())
    meta["codex_session_id"] = "s-gone"
    (run_dir / "meta.json").write_text(_json.dumps(meta))
    # A sibling run's rollout is present, keyed to a different cwd.
    _write_rollout(home, "2026-09-06T09-00-00-s-other",
                   _rollout(str(tmp_path / "runs" / "other")))
    assert find_codex_transcript(run_dir) is None


def test_a_pruned_session_falls_back_to_the_cwd_join(tmp_path):
    """A recorded-but-missing id must not shadow a rollout that is still
    findable the original way."""
    import json as _json
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    meta = _json.loads((run_dir / "meta.json").read_text())
    meta["codex_session_id"] = "s-pruned"
    (run_dir / "meta.json").write_text(_json.dumps(meta))
    mine = _write_rollout(home, "2026-09-06T11-00-00-s-1",
                          _rollout(str(run_dir.resolve())))
    assert find_codex_transcript(run_dir) == mine


def test_capture_also_records_the_session_id(tmp_path):
    """A staged run is the one shape that never revisits discovery, so it
    would otherwise be the only one with no recorded id."""
    import json as _json
    home = tmp_path / "codex_home"
    run_dir = _run_dir(tmp_path, home)
    _write_rollout(home, "2026-09-06T11-31-30-s-1",
                   _rollout(str(run_dir.resolve())))
    assert capture(run_dir) == "copied"
    meta = _json.loads((run_dir / "meta.json").read_text())
    assert meta["codex_session_id"] == "s-1"


def test_a_claude_run_gains_no_codex_session_key(tmp_path):
    """Discovery must not stamp codex fields onto a claude bundle."""
    import json as _json
    run_dir = tmp_path / "runs" / "claude_run"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(_json.dumps({"runtime": "claude_cli"}))
    assert capture(run_dir) == "unavailable"
    meta = _json.loads((run_dir / "meta.json").read_text())
    assert "codex_session_id" not in meta


# ---------------------------------------------------------------------------
# Shutdown during capture must not cost meta.json its outcome
# ---------------------------------------------------------------------------

def test_finalize_meta_writes_the_outcome_even_if_capture_is_interrupted(
        monkeypatch):
    """A `SystemExit` during shutdown can cut `capture`'s read short. It
    escapes `capture`'s `except OSError`, and if it also skipped the meta write
    the bundle would have neither a transcript nor an outcome.

    Only one of those is recoverable: the rollout is still in the CLI's session
    store and can be re-staged, while a run with no outcome is indistinguishable
    from one that died mid-flight. So the signal is honored one step later,
    after `meta.json` is on disk.
    """
    import pytest as _pytest

    from dycraft.agent import agent_session as S

    wrote: dict = {}

    class _TM:
        def finalize(self, outcome, **fields):
            wrote.update(outcome=outcome, fields=fields)

    class _Ctx:
        run_dir = "/nonexistent/run"
        step = 0

    sess = S.AgentSession.__new__(S.AgentSession)
    sess.ctx, sess._tm = _Ctx(), _TM()
    monkeypatch.setattr(S, "_meta_extras", lambda ctx: {})
    # Patched where `finalize_meta` holds them: they are imported at module
    # scope, so that a deleted worktree cannot fail the import.
    monkeypatch.setattr(S, "capture",
                        lambda _d: (_ for _ in ()).throw(SystemExit(0)))

    with _pytest.raises(SystemExit):
        sess.finalize_meta("committed")

    assert wrote["outcome"] == "committed"
    assert wrote["fields"]["agent_transcript"] == "interrupted"
