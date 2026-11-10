"""`agent/usage.py` — the `usage` block a bundle records about itself.

Two kinds of test, and the second kind is the one that matters.

The hermetic ones pin the three traps that make this non-obvious, each of which
would give a wrong number: a 0.153+ codex rollout reports the same response
twice (double count), a claude session repeats a turn's `usage` on every
content block (double count), and a transcript that ends mid-response has a
real turn with no usage record (undercount).

The agreement one asserts the invariant the module exists to keep: **these numbers
must equal what `viewer/trajectory.py` computes from the same file.** The two
parse the same transcripts by different routes — the viewer per turn, this
module in aggregate — and they feed two different parts of the UI, the
Trajectory panel and the sidebar badge. Two implementations of token accounting
that disagree are worse than one that is missing, so the agreement is pinned
rather than assumed: here on bundles written to carry the traps above, and in
the dev-only `test_bundle_usage_recorded_runs.py` on every recorded run.
"""

from __future__ import annotations

import json
from pathlib import Path

from dycraft.agent.usage import summarize

TOKEN_FIELDS = ("gross_input", "output", "input", "cache_read", "total")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write(run_dir: Path, records: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "agent_transcript.jsonl"
    with open(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def _codex_usage(gross: int, cached: int, out: int, write: int = 0) -> dict:
    """Codex's vocabulary: `input_tokens` is gross, cached is a subset of it."""
    return {"input_tokens": gross, "cached_input_tokens": cached,
            "cache_write_input_tokens": write, "output_tokens": out,
            "total_tokens": gross + out}


def _assistant(text: str = "ok") -> dict:
    return {"type": "response_item",
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


def _tool_call() -> dict:
    return {"type": "response_item",
            "payload": {"type": "custom_tool_call", "name": "shell"}}


def _token_count(usage: dict) -> dict:
    """The `event_msg` form. Carries the per-response figure *and* a running
    total; only `last_token_usage` is read, because the running total is a
    different number that disagrees with the records on ragged transcripts."""
    return {"type": "event_msg",
            "payload": {"type": "token_count",
                        "info": {"last_token_usage": usage,
                                 "total_token_usage": usage}}}


def _token_usage_record(usage: dict) -> dict:
    """The 0.153+ form, emitted *in addition to* `_token_count`."""
    return {"type": "token_usage_record", "payload": {"usage": usage}}


# ---------------------------------------------------------------------------
# codex
# ---------------------------------------------------------------------------


def test_a_response_reported_by_both_record_kinds_is_counted_once(tmp_path):
    """0.153+ emits `token_count` *and* `token_usage_record` per response.

    Summing whichever arrives without excluding the other doubles every figure
    in the run. The guard is that a usage record only closes a turn when
    content accumulated since the last one — the same rule
    `viewer/trajectory.py::_from_codex_transcript`'s `flush` applies.
    """
    usage = _codex_usage(gross=1000, cached=800, out=50)
    _write(tmp_path, [
        _assistant(), _tool_call(),
        _token_count(usage),            # closes the response
        _token_usage_record(usage),     # the same response again
    ])
    block = summarize(tmp_path)
    assert block["turns"] == 1, "the duplicate record opened a second turn"
    assert block["gross_input"] == 1000
    assert block["output"] == 50
    assert block["total"] == 1050


def _keyed_record(usage: dict, response_id: str, ts: str | None = None) -> dict:
    record = {"type": "token_usage_record",
              "payload": {"usage": usage, "response_id": response_id}}
    if ts:
        record["timestamp"] = ts
    return record


def test_a_compaction_is_a_response_and_is_counted(tmp_path):
    """A context compaction spends tokens and emits no content.

    A guard that counts only responses with content would drop it, tokens and
    all. Keyed on `response_id`, it is one more response, and the `compacted`
    record names it, so the block can say when it happened and what it cost.
    """
    from dycraft.viewer.trajectory import build as build_trajectory
    first, summary, after = (_codex_usage(1000, 800, 50), _codex_usage(4000, 1000, 300),
                             _codex_usage(600, 0, 20))
    _write(tmp_path, [
        {"type": "session_meta", "payload": {}},
        _assistant(), _tool_call(), _keyed_record(first, "r1"), _token_count(first),
        _keyed_record(summary, "r2", "2026-09-19T20:25:46.871Z"),
        {"type": "compacted", "timestamp": "2026-09-19T20:25:46.885Z",
         "payload": {"compaction_response_id": "r2", "message": ""}},
        # The new window's size, reported once; not a response.
        _token_count(_codex_usage(900, 0, 0)),
        _assistant(), _keyed_record(after, "r3"), _keyed_record(after, "r3"),
    ])
    block = summarize(tmp_path)
    assert block["turns"] == 3
    assert block["gross_input"] == 5600 and block["output"] == 370
    assert block["compactions"] == [{"ts": "2026-09-19T20:25:46.885Z", "total_tokens": 4300}]
    trajectory = build_trajectory(tmp_path)
    assert trajectory["totals"]["total"] == block["total"]
    assert trajectory["totals"]["turns"] == 3
    assert [bool(t.get("compaction")) for t in trajectory["turns"]] == [False, True, False]


def test_a_token_count_inside_a_response_does_not_count_it_twice(tmp_path):
    """The content guard's other trap: a `token_count` landing between a
    response's text and its tool call must not close the turn early, or the
    record that follows sees fresh content and counts the response again."""
    from dycraft.viewer.trajectory import build as build_trajectory
    usage = _codex_usage(1000, 800, 50)
    _write(tmp_path, [{"type": "session_meta", "payload": {}},
                      _assistant(), _token_count(_codex_usage(700, 0, 10)), _tool_call(),
                      _keyed_record(usage, "r1")])
    block = summarize(tmp_path)
    assert block["turns"] == 1 and block["total"] == 1050
    assert "compactions" not in block
    assert build_trajectory(tmp_path)["totals"]["total"] == 1050


def test_codex_gross_and_cached_are_split_not_added(tmp_path):
    """`input_tokens` already includes `cached_input_tokens`.

    Adding them would bill the cache twice; `input` is the uncached remainder.
    """
    _write(tmp_path, [
        _assistant(),
        _token_usage_record(_codex_usage(gross=1000, cached=800, out=50,
                                        write=50)),
    ])
    block = summarize(tmp_path)
    assert block["gross_input"] == 1000
    assert block["cache_read"] == 800
    assert block["cache_write"] == 50
    assert block["input"] == 150, "gross - cached - written"


def test_a_response_whose_usage_record_never_arrived_still_counts_as_a_turn(tmp_path):
    """A transcript that ends mid-response has a turn with no usage.

    The viewer closes it with an unconditional trailing flush, so this must
    too or the two disagree on `turns`. A run that is killed or runs out of
    credit ends this way.
    """
    _write(tmp_path, [
        _assistant(), _token_usage_record(_codex_usage(1000, 800, 50)),
        _assistant(), _tool_call(),       # second response, no usage record
    ])
    block = summarize(tmp_path)
    assert block["turns"] == 2
    assert block["gross_input"] == 1000, "the unclosed turn adds no tokens"


def test_only_assistant_messages_with_text_open_a_turn(tmp_path):
    """A rollout carries `developer` and `user` messages too, and an assistant
    message can be empty. Treating either as content closes a turn the viewer
    never opened."""
    _write(tmp_path, [
        {"type": "response_item",
         "payload": {"type": "message", "role": "developer",
                     "content": [{"type": "input_text", "text": "prompt"}]}},
        {"type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "   "}]}},
        _token_usage_record(_codex_usage(1000, 800, 50)),
    ])
    assert summarize(tmp_path) is None, "nothing was accumulated to close"


# ---------------------------------------------------------------------------
# claude
# ---------------------------------------------------------------------------


def _claude_record(mid: str, usage: dict, blocks: list[dict]) -> dict:
    return {"type": "assistant",
            "message": {"id": mid, "role": "assistant",
                        "content": blocks, "usage": usage}}


def _claude_usage(inp: int, read: int, write: int, out: int) -> dict:
    """Anthropic's vocabulary: `input_tokens` is the *uncached remainder*."""
    return {"input_tokens": inp, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write, "output_tokens": out}


def test_a_claude_turn_split_across_records_is_counted_once(tmp_path):
    """The CLI writes one record per content block, all sharing `message.id`
    and repeating the turn's `usage` verbatim. Summing naively multiplies a
    turn's tokens by its number of blocks."""
    usage = _claude_usage(inp=100, read=900, write=0, out=20)
    _write(tmp_path, [
        _claude_record("msg_1", usage, [{"type": "text", "text": "hi"}]),
        _claude_record("msg_1", usage, [{"type": "tool_use", "name": "Bash"}]),
        _claude_record("msg_1", usage, [{"type": "tool_use", "name": "Read"}]),
    ])
    block = summarize(tmp_path)
    assert block["format"] == "claude"
    assert block["turns"] == 1, "one logical turn, three records"
    assert block["gross_input"] == 1000, "100 uncached + 900 cache read"
    assert block["input"] == 100
    assert block["output"] == 20
    # Tool calls are *not* deduplicated: each record carries a different block,
    # so every `tool_use` appears exactly once across the group.
    assert block["tool_calls"] == 2


def _request_echo(blocks: list[dict]) -> dict:
    """An `api-request-blob` record: the conversation as sent with a request."""
    return {"type": "api-request-blob", "hash": "h", "sessionId": "s",
            "message": {"role": "assistant", "content": blocks}}


def test_a_request_echo_does_not_count_its_tool_calls_again(tmp_path):
    """A request's echo of the conversation carries the earlier turns' tool
    calls; they were counted when the model made them."""
    usage = _claude_usage(inp=100, read=900, write=0, out=20)
    call = {"type": "tool_use", "id": "toolu_1", "name": "Bash"}
    _write(tmp_path, [
        _claude_record("msg_1", usage, [call]),
        _request_echo([call]),
        _claude_record("msg_2", usage, [{"type": "text", "text": "done"}]),
    ])
    block = summarize(tmp_path)
    assert block["turns"] == 2
    assert block["tool_calls"] == 1


# ---------------------------------------------------------------------------
# absence
# ---------------------------------------------------------------------------


def test_a_bundle_with_no_transcript_reports_nothing_rather_than_zero(tmp_path):
    """None, not a block of zeros: `meta.json` must keep saying "unknown"
    instead of claiming the run cost nothing."""
    assert summarize(tmp_path) is None


def test_a_transcript_with_no_model_responses_reports_nothing(tmp_path):
    """A run that was launched and never got a reply (an account out of
    credit, say) has a transcript but no usage to account for."""
    _write(tmp_path, [
        {"type": "session_meta", "payload": {"session_id": "x"}},
        {"type": "event_msg", "payload": {"type": "task_started"}},
    ])
    assert summarize(tmp_path) is None


# ---------------------------------------------------------------------------
# the corpus invariant
# ---------------------------------------------------------------------------


def _disagreements(run_dir: Path) -> list[str] | None:
    """Every field `summarize` and the viewer's trajectory both report, where they differ.

    None when either view has nothing for this bundle. Shared with the dev-only
    `test_bundle_usage_recorded_runs.py`, so both apply one comparison.
    """
    from dycraft.viewer.trajectory import build as build_trajectory

    block = summarize(run_dir)
    totals = build_trajectory(run_dir).get("totals") or {}
    if block is None or not totals:
        return None
    return [f"{run_dir}: {field} usage={block.get(field)} trajectory={totals[field]}"
            for field in TOKEN_FIELDS + ("turns", "tool_calls")
            if totals.get(field) is not None and block.get(field) != totals[field]]


def test_usage_matches_the_viewers_own_arithmetic(tmp_path) -> None:
    """The sidebar badge and the Trajectory panel must not disagree.

    `viewer/trajectory.py` parses the same transcripts per turn and is what the
    Trajectory panel shows; this module aggregates them and is what lands in
    `meta.json` for the sidebar. Compared on one codex bundle (a response
    reported by both record kinds, a tool call, a second turn) and one claude
    bundle (a turn split across records, a request echoing its tool call,
    a second turn with a cache write).
    """
    codex = _codex_usage(gross=1000, cached=800, out=50)
    _write(tmp_path / "codex", [
        {"type": "session_meta", "payload": {"id": "s"}},
        _assistant(), _tool_call(), _token_count(codex), _token_usage_record(codex),
        _assistant("next"), _token_count(_codex_usage(gross=1500, cached=1200, out=70)),
    ])
    first = _claude_usage(inp=100, read=900, write=0, out=20)
    _write(tmp_path / "claude", [
        _claude_record("msg_1", first, [{"type": "text", "text": "hi"}]),
        _claude_record("msg_1", first, [{"type": "tool_use", "id": "toolu_1", "name": "Bash"}]),
        _request_echo([{"type": "tool_use", "id": "toolu_1", "name": "Bash"}]),
        _claude_record("msg_2", _claude_usage(inp=50, read=1000, write=10, out=30),
                       [{"type": "text", "text": "done"}]),
    ])
    assert {name: _disagreements(tmp_path / name) for name in ("codex", "claude")} == {
        "codex": [], "claude": []}


def test_blend_backups_are_excluded_from_hashes_and_manifests():
    """Blender's `.blend1` is never a manifest member or a workspace hash input.

    It sits in the workspace beside `animation.blend` at the same size, so if
    kept it would be copied by every job, hashed by `tree_hash` on every
    publish, copied into `bundle/` at `finish`, and hashed into `ir.json`, where
    a delivered manifest would assert the integrity of the project's
    *previous* state. Suppressed at the save and excluded here, so the manifest
    stays right either way.
    """
    import pathlib
    import tempfile
    from dycraft.animation.blender.bundle import make_manifest, tree_hash

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "animation.blend").write_bytes(b"real project")
    (d / "evaluation").mkdir()
    (d / "evaluation" / "index.json").write_text("{}")
    ev = {"fps": 24, "n_frames": 2, "frame_start": 0, "blender_version": "4.2.0"}
    before, manifest_before = tree_hash(d), make_manifest(d, scene_ref="s", evaluation=ev)
    (d / "animation.blend1").write_bytes(b"stale backup" * 10)
    after, manifest_after = tree_hash(d), make_manifest(d, scene_ref="s", evaluation=ev)
    assert before == after
    assert manifest_before["files"] == manifest_after["files"]
    assert not [k for k in manifest_after["files"] if k.endswith(".blend1")]
