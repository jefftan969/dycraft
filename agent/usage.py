"""Token totals for one run bundle, read from its own staged transcript.

Why this lives under `agent/` and not `viewer/`
-----------------------------------------------
`viewer/trajectory.py` already computes these numbers, per turn, and far more
besides — but it cannot be the source for `meta.json`, in either direction:

* **The harness cannot call it.** `viewer/trajectory.py` imports
  `..agent.transcript_capture`, so the layering runs viewer -> agent and an
  `agent` -> `viewer` import would close a cycle.
* **The viewer cannot compute it on demand.** `viewer/runs.py::list_runs` reads
  wall time and tokens "straight off meta.json and never computed here",
  because the sidebar lists every directory under `runs/`, and opening each
  transcript to draw a badge would make listing
  cost as much as reading.

So the bundle has to record its own totals, and the code that does it has to
sit at or below `agent/`. `transcript_capture` is already the module that owns
"where is this run's transcript"; this is the thin companion that owns "what
did it cost". Every bundle records it, child runs included, so dispatched and
review sub-agents' tokens are accounted for.

Agreement with the viewer
-------------------------
Two implementations of token accounting that disagree are worse than one that
is missing — they feed the sidebar badge and the Trajectory panel of the same
bundle. So the agreement is pinned by a test, not assumed:
`tests/test_bundle_usage.py::test_corpus_usage_matches_the_viewers_own_arithmetic`
compares every field both produce against
`viewer.trajectory.build(run_dir)["totals"]` over every local bundle.

The vocabulary is deliberately identical to `viewer/trajectory.py::_usage`:
`gross_input` is everything the model read including cache, while `input` is
the uncached remainder only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

#: Written into the block so a reader can tell which CLI produced the numbers
#: and therefore which vocabulary they are in.
FORMAT_CODEX = "codex"
FORMAT_CLAUDE = "claude"

_TRANSCRIPT_NAME = "agent_transcript.jsonl"


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


def _block(*, gross_input: int, cache_read: int, cache_write: int,
           output: int, total: int | None = None) -> dict:
    """The six figures, named as `viewer/trajectory.py::_usage` names them."""
    gross_input, cache_read = int(gross_input), int(cache_read)
    cache_write, output = int(cache_write), int(output)
    return {
        # The uncached remainder. Anthropic reports this directly as
        # `input_tokens`; codex reports the gross and the cached subset, so it
        # is a subtraction there.
        "input": max(gross_input - cache_read - cache_write, 0),
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": output,
        "gross_input": gross_input,
        "total": int(total) if total is not None else gross_input + output,
    }


def _from_codex(path: Path) -> dict | None:
    """Totals from a codex rollout, summed per response.

    Codex also keeps a running total — `info.total_token_usage` on the last
    `event_msg/token_count` — and reading that is tempting because it needs no
    arithmetic. But on a transcript that ends raggedly, codex's own counter and
    the records it wrote diverge by a turn or two at the tail, and a sidebar
    badge that contradicts `viewer/trajectory.py`'s panel for the same bundle
    is its own defect. So this sums the records instead and agrees with the
    viewer by construction.

    The double-counting trap that makes summing non-obvious: a 0.153+ rollout
    emits **both** an `event_msg/token_count` and a `token_usage_record` per
    response, each carrying the same figure. Such a rollout is counted from
    its `token_usage_record`s alone, once per `response_id`, and its
    `token_count`s are ignored. A content guard instead — a usage record closes
    a turn only if a tool call or assistant text arrived since the last one —
    is wrong in both directions: it drops every **context compaction**, a
    response that spends tokens and emits neither, and a `token_count` that
    lands mid-response closes the turn early, so the same response's record is
    counted twice. Keyed on the id, the sum equals codex's own cumulative
    `thread_token_usage` on the last record.

    A rollout with no `token_usage_record` (0.146/0.147) has no id to key on
    and keeps the content guard over its `token_count`s, as the viewer does.

    `compactions` names each one with the tokens its own response spent: the
    `compacted` record carries the `compaction_response_id` it was produced
    by. Nothing else in the run directory records that one happened.

    `input_tokens` in codex's vocabulary is the gross figure and
    `cached_input_tokens` is the cached subset of it.
    """
    keyed = any(r.get("type") == "token_usage_record" for r in _iter_jsonl(path))
    gross = cache_read = cache_write = output = 0
    turns = tool_calls = 0
    pending = False          # content seen since the last usage record
    found = False
    responses: dict[str, dict] = {}
    compactions: list[dict] = []

    def flush(usage: dict, response_id: str | None = None) -> None:
        nonlocal gross, cache_read, cache_write, output, turns, pending, found
        if response_id is not None:
            if response_id in responses:
                return
            responses[response_id] = usage
        elif not pending:    # a repeat of the response already counted
            return
        pending = False
        found = True
        turns += 1
        read = int(usage.get("cached_input_tokens", 0) or 0)
        write = int(usage.get("cache_write_input_tokens", 0) or 0)
        gross += int(usage.get("input_tokens", 0) or 0)
        cache_read += read
        cache_write += write
        output += int(usage.get("output_tokens", 0) or 0)

    for record in _iter_jsonl(path):
        kind = record.get("type")
        payload = record.get("payload")
        if kind == "token_usage_record":
            usage = record.get("usage")
            response_id = None
            if isinstance(payload, dict):
                usage = payload.get("usage", usage)
                response_id = payload.get("response_id") or None
            if isinstance(usage, dict):
                flush(usage, response_id)
            continue
        if not isinstance(payload, dict):
            continue
        if kind == "compacted":
            spent = responses.get(payload.get("compaction_response_id") or "") or {}
            compactions.append({
                "ts": record.get("timestamp"),
                "total_tokens": int(spent.get("input_tokens", 0) or 0)
                + int(spent.get("output_tokens", 0) or 0)})
            continue
        item = payload.get("type")
        if item == "token_count":
            if keyed:
                continue
            info = payload.get("info")
            last = info.get("last_token_usage") if isinstance(info, dict) else None
            if isinstance(last, dict):
                flush(last)
        elif item in ("function_call", "custom_tool_call"):
            tool_calls += 1
            pending = True

        elif item == "message" and payload.get("role") == "assistant":
            # Non-empty text only. The viewer accumulates `_codex_text(...)`
            # and skips the falsy result, so an assistant message carrying no
            # text is not content there and must not be content here either —
            # counting it closes a turn the viewer never opened.
            content = payload.get("content")
            if isinstance(content, list) and any(
                    isinstance(b, dict) and (b.get("text") or "").strip()
                    for b in content):
                pending = True
            elif isinstance(content, str) and content.strip():
                pending = True

    # A response whose usage record never arrived still happened. The viewer
    # closes it the same way — `_from_codex_transcript` ends with an
    # unconditional `flush({}, prev_ts)` — so a trailing flush here adds the
    # turn and no tokens, and the two agree on `turns` as well as on the token
    # figures. Without it a transcript that ends mid-response counts one turn
    # short.
    flush({})

    if not found:
        return None
    out = _block(gross_input=gross, cache_read=cache_read,
                 cache_write=cache_write, output=output,
                 total=gross + output)
    out.update({"format": FORMAT_CODEX, "turns": turns,
                "tool_calls": tool_calls})
    if compactions:
        out["compactions"] = compactions
    return out


def _from_claude(path: Path) -> dict | None:
    """Totals from a claude session transcript.

    Summed per turn, and **deduplicated on `message.id`**: the CLI writes one
    JSONL record per content block, all sharing the turn's id and repeating its
    `usage` verbatim, so a naive sum multiplies a turn's tokens by its number
    of blocks. `viewer/trajectory.py::_group_assistant_records` exists for the
    same reason.

    Tool calls are counted on `assistant` records only, as the viewer counts
    them: `api-request-blob` records echo the conversation sent with a request,
    earlier tool calls included.
    """
    gross = cache_read = cache_write = output = 0
    seen: set[str] = set()
    turns = tool_calls = 0
    found = False
    for record in _iter_jsonl(path):
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        if record.get("type") == "assistant":
            tool_calls += sum(1 for block in (message.get("content") or [])
                              if isinstance(block, dict) and block.get("type") == "tool_use")
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        key = message.get("id")
        if isinstance(key, str) and key:
            if key in seen:
                continue
            seen.add(key)
        found = True
        turns += 1
        inp = int(usage.get("input_tokens", 0) or 0)
        read = int(usage.get("cache_read_input_tokens", 0) or 0)
        write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        # Anthropic's `input_tokens` is already the uncached remainder, so the
        # gross figure is the sum of the three.
        gross += inp + read + write
        cache_read += read
        cache_write += write
        output += int(usage.get("output_tokens", 0) or 0)
    if not found:
        return None
    out = _block(gross_input=gross, cache_read=cache_read,
                 cache_write=cache_write, output=output,
                 total=gross + output)
    out.update({"format": FORMAT_CLAUDE, "turns": turns,
                "tool_calls": tool_calls})
    return out


#: The figures that add across threads.
_ADDITIVE = ("input", "cache_read", "cache_write", "output", "gross_input",
             "total", "turns", "tool_calls")


def summarize_rollouts(paths: list[Path]) -> dict | None:
    """Summed totals of several codex rollouts — a run's sub-agents.

    Each rollout is summed exactly as the run's own transcript is
    (`_from_codex`), then the threads are added together. `threads` counts the
    rollouts that recorded at least one response; None when none did.
    """
    blocks = [block for block in (_from_codex(Path(p)) for p in paths) if block]
    if not blocks:
        return None
    out = {key: sum(int(block.get(key) or 0) for block in blocks)
           for key in _ADDITIVE}
    out.update({"format": FORMAT_CODEX, "threads": len(blocks)})
    return out


def _sniff(path: Path) -> str | None:
    """Which CLI wrote this transcript, from its content.

    Sniffed rather than read off `meta.json`'s `runtime`, for the reason
    `viewer/trajectory.py::_parse_cli_transcript` gives: a bundle can be staged
    or replayed under a different runner than the one that recorded it.
    """
    for record in _iter_jsonl(path):
        if record.get("type") in ("session_meta", "turn_context",
                                  "token_usage_record", "response_item",
                                  "event_msg"):
            return FORMAT_CODEX
        if isinstance(record.get("message"), dict):
            return FORMAT_CLAUDE
    return None


def summarize(run_dir: Path | str) -> dict | None:
    """The run's `usage` block, or None when there is nothing to account for.

    None — rather than a block of zeros — so `meta.json` keeps saying "unknown"
    instead of claiming a run cost nothing. Two cases reach it: no staged
    transcript at all, and a transcript that records no model response. The
    second is a real state, not a parse failure — a run whose account has no
    credit launches, writes a `task_started`, and never gets a reply. The
    viewer reports 0 turns and 0 tokens for those same bundles.

    `wall_seconds` is deliberately absent: `meta.json` already
    carries `started_at` / `finished_at`, and `viewer/runs.py::_wall_seconds`
    prefers `usage.wall_seconds` when present, so writing a second, differently
    measured span here would quietly override the harness's own.
    """
    path = Path(run_dir) / _TRANSCRIPT_NAME
    if not path.is_file():
        return None
    fmt = _sniff(path)
    if fmt == FORMAT_CODEX:
        return _from_codex(path)
    if fmt == FORMAT_CLAUDE:
        return _from_claude(path)
    return None
