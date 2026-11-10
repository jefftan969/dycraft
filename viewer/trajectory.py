"""Normalize an agent run's LLM trajectory into one shape for the viewer.

The run's external CLI owns the LLM loop, and its session transcript is staged into the bundle by
`agent/transcript_capture.py`; `_parse_cli_transcript` tells the two formats apart from the first record rather
than trusting `meta.json`'s `runtime`.

* **`claude`**: a Claude Code session. Per-turn `usage`, every tool call, and the visible text are there, but
  **thinking blocks carry an empty `thinking` string** (only the signature is persisted), so reasoning is
  unrecoverable and `reasoning_available` is False, letting the UI say why rather than imply the model did not
  think.
* **`codex`**: a Codex *rollout*, read by `_from_codex_transcript`. A turn is one model response
  (`token_usage_record`), since a headless run is a single Codex turn from end to end. Reasoning has the same gap:
  `encrypted_content` with an empty summary.

Turn shape (see `viewer/web/src/components/inspector/trajectory-types.ts`):

    {index, ts, elapsed_ms, model, effort, thought_summary, text,
     tool_calls: [{name, input, result, is_error, dycraft_op, dycraft_step,
                   started_at, finished_at, duration_ms}],
     usage: {input, cache_read, cache_write, output, total},
     cost_usd, stop_reason}

`started_at` / `finished_at` / `duration_ms` on a tool call are what `viewer/agent_timeline.py` lays out on its time
axis. For a harness op the authoritative duration is `trace.jsonl`'s `op_result.elapsed_ms` and these are a
fallback; for every other tool (`Read`, `Edit`, `Skill`, plain shell) they are the only timing there is.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..agent.transcript_capture import resolve_transcript
from . import _trace
from .cost import price_usage, unpriced_models

# `h.compile()`, `h.probe(expression)`, `h.view("all")` — the agent drives
# the harness by shelling out to $RUN_DIR/harness.py, so the dycraft op is only
# visible inside the Bash command text.
#
# `harness` as well as `h`, and whitespace around the dot, because agents write
# both: a call written as `harness.record_deviation(` or split across a line
# continuation is still a call.
_HARNESS_OP_RE = re.compile(r"\b(?:h|harness)\s*\.\s*([a-z_]+)\s*\(")

#: Op names the harness exposes. The regex above matches any `h.<name>(`, and
#: an agent's own scripts have other receivers called `h`
#: (`h = hashlib.sha256()`, then `h.update(b)`), which would tag the script as
#: a harness op `update`. Matching the shape is not enough: the name has to be
#: one that exists.
_HARNESS_OP_NAMES = frozenset({
    "accept_storyboard", "author_mesh", "author_solids", "await_any",
    "blender", "build",
    "cancel_contribution", "cancel_dispatch", "check_motion", "compile",
    "contribution_status", "dispatch_motion", "dispatch_object",
    "dispatch_scene", "dispatch_status", "final_render", "finish", "ik",
    "inspect", "job_cancel", "job_result", "job_status", "motion_image",
    "plan", "probe", "read_file", "record_deviation", "render", "review",
    "set_render_backend", "snapshot", "storyboard", "submit_object",
    "submit_result", "view", "write_file"
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_ms(prev: datetime | None, cur: datetime | None) -> float | None:
    if prev is None or cur is None:
        return None
    return round((cur - prev).total_seconds() * 1000.0, 1)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _text_of(content: Any) -> str:
    """Flatten a tool result / message content into display text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, indent=2, default=str))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, indent=2, default=str)


def _usage(*, inp: int = 0, cache_read: int = 0, cache_write: int = 0,
           cache_write_5m: int = 0, cache_write_1h: int = 0,
           out: int = 0) -> dict:
    """One turn's tokens, split the way they are *billed*.

    `cache_write` is the provider's total; the 5 m / 1 h split matters because
    the 1 h tier costs 1.6x. When only the total is known the tiers are left at
    zero and `cost.price_usage` falls back to the 5 m rate.

    `gross_input` is the Articraft/OpenAI-style figure — everything the model
    read this turn, cache included. Anthropic reports `input_tokens` as the
    *uncached remainder* only, so the two vocabularies disagree and both are
    carried explicitly rather than one being silently relabeled.
    """
    inp, cache_read = int(inp), int(cache_read)
    cache_write = int(cache_write)
    cache_write_5m, cache_write_1h = int(cache_write_5m), int(cache_write_1h)
    out = int(out)
    return {
        "input": inp,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "cache_write_5m": cache_write_5m,
        "cache_write_1h": cache_write_1h,
        "output": out,
        "gross_input": inp + cache_read + cache_write,
        "total": inp + cache_read + cache_write + out,
    }


def _model_counts(models) -> list[dict]:
    """Turns per model, busiest first."""
    counts: dict[str, int] = {}
    for m in models:
        if m:
            counts[m] = counts.get(m, 0) + 1
    return [{"model": m, "turns": n}
            for m, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _dominant_model(models) -> str | None:
    """The model that ran the most turns.

    Not the last one seen: a `--model fable` run that falls back to opus-5 on
    its final turns is a fable run with an opus tail, and labeling the whole
    run by its last turn mis-attributes both the model and its price.
    """
    counts = _model_counts(models)
    return counts[0]["model"] if counts else None


# ---------------------------------------------------------------------------
# dycraft-step join
# ---------------------------------------------------------------------------


def _harness_ops(run_dir: Path) -> list[dict]:
    """`trace.jsonl`'s op_call records, chronological.

    Used to attach a dycraft step number to the agent tool call that triggered
    it. Both sequences are chronological, but an in-order greedy match on the
    op name alone is not enough: ops a script runs (`python fix.py` calling
    `h.compile()` forty times) show no op in the tool call, so they stay
    unclaimed, and the next call that does name `compile` would match the
    oldest of them, hours before it started. `_claim_step` therefore also
    requires the op to have been received after the call began.
    """
    out = []
    for rec in _trace.events(run_dir / "trace.jsonl", {"op_call"}):
        out.append({"op": rec.get("op"), "step": rec.get("step"),
                    "ts": rec.get("ts"), "at": _utc(_parse_ts(rec.get("ts"))),
                    "used": False})
    return out


_trace.register({"op_call"})


# The daemon stamps an op when it reads the request, after the agent's tool
# call record; this only absorbs timestamp rounding between the two writers,
# which share one host clock.
_CLAIM_SLACK = timedelta(seconds=1)


def _utc(value: datetime | None) -> datetime | None:
    """Naive timestamps are UTC: `utcnow_iso` and both CLIs write UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _claim_step(ops: list[dict], op: str, started_at: Any = None,
                finished_at: Any = None) -> int | None:
    """Claim the first unused `op` the daemon received while the call ran.

    An op older than the call cannot be its own, and since later calls start
    later still, no call will ever claim it — skipping it loses nothing. The
    upper bound matters as much: a shell that only *writes* a script naming
    `h.finish()` (`cat > scratch/r6_finish.py <<PY`) ran no op, and without it
    claimed the real `finish` an hour later, stealing it from the call that
    did. A shell that backgrounds its script loses its step to this — None, not
    a wrong one. Either bound is skipped when a timestamp is missing.
    """
    start = _utc(_parse_ts(started_at))
    end = _utc(_parse_ts(finished_at))
    for entry in ops:
        if entry["used"] or entry["op"] != op:
            continue
        at = entry["at"]
        if at is not None and start is not None and at < start - _CLAIM_SLACK:
            continue
        if at is not None and end is not None and at > end + _CLAIM_SLACK:
            # Ops are chronological: every later candidate is later still.
            return None
        entry["used"] = True
        return entry["step"]
    return None


# Only a shell tool actually runs the harness. Scanning every tool's payload
# would match prose: a context summary that *mentions* `h.author_solids(` would
# be counted as a call.
_SHELL_TOOLS = {"bash", "shell", "run_command", "terminal"}


def _detect_op(tool_name: str | None, tool_input: Any) -> str | None:
    """The dycraft harness op a tool call invoked, if any."""
    if not tool_name:
        return None
    # A lowercase tool name is taken as the harness op itself.
    if not tool_name[0].isupper():
        return tool_name
    if tool_name.lower() not in _SHELL_TOOLS:
        return None
    command = (tool_input or {}).get("command") if isinstance(
        tool_input, dict) else tool_input
    if not isinstance(command, str):
        return None
    # Same gate the codex path applies: a bare `h.<name>(` is only a harness
    # call if the command also drives the harness. See `_HARNESS_INVOCATION_RE`.
    if not _HARNESS_INVOCATION_RE.search(command):
        return None
    match = _HARNESS_OP_RE.search(command)
    return match.group(1) if match and match.group(1) in _HARNESS_OP_NAMES else None


# ---------------------------------------------------------------------------
# claude / codex CLI transcript
# ---------------------------------------------------------------------------


def _group_assistant_records(records: list[dict]) -> list[dict]:
    """Fold streamed assistant records into logical turns.

    The CLI writes one JSONL record per content block, all sharing the same
    `message.id` and repeating the turn's `usage` verbatim. Treating each
    record as a turn both inflates the turn count and multiplies token totals,
    so group on `message.id` and take `usage` once.

    Each block keeps the timestamp of the record it arrived on, as `_record_ts`.
    The group's own `timestamp` is the *first* block's, which is right for "when
    this turn started" and wrong for anything else: a turn that emits
    `[thinking, tool_use]` where the tool input is a 10 kB context summary
    finishes its thinking block a couple of minutes before it finishes emitting
    the tool call. Measuring a tool call from the group timestamp charged the
    model's generation time to the tool — `Write context_summary.md` came out at
    154 s, and the Author lane at 31% of an object run instead of 9%.

    `prev_timestamp` is the record immediately *before* the group opened, which
    is when the model was unblocked; `last_timestamp` is when it stopped. The
    two bracket the window the model held the floor for.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    prev_ts: Any = None
    for rec in records:
        # A subagent's records interleave into the same file. They are not this
        # turn's predecessor, and taking one as `prev_timestamp` would give
        # model windows that end before they begin.
        if rec.get("isSidechain"):
            continue
        if rec.get("type") != "assistant":
            if rec.get("timestamp"):
                prev_ts = rec.get("timestamp")
            continue
        message = rec.get("message") or {}
        key = (message.get("id") or rec.get("requestId")
               or f"_seq{len(order)}")
        group = groups.get(key)
        if group is None:
            group = {
                "message": message,
                "content": [],
                "timestamp": rec.get("timestamp"),
                "last_timestamp": rec.get("timestamp"),
                "prev_timestamp": prev_ts,
                "records": [],
                "effort": rec.get("effort"),
            }
            groups[key] = group
            order.append(key)
        if rec.get("timestamp"):
            group["last_timestamp"] = rec.get("timestamp")
        group["records"].append({
            "at": rec.get("timestamp"),
            "kinds": [b.get("type") for b in (message.get("content") or [])
                      if isinstance(b, dict)],
        })
        group["content"].extend(
            dict(b, _record_ts=rec.get("timestamp"))
            for b in (message.get("content") or []) if isinstance(b, dict))
        if rec.get("timestamp"):
            prev_ts = rec.get("timestamp")
    return [groups[k] for k in order]


# ---------------------------------------------------------------------------
# codex CLI rollout
# ---------------------------------------------------------------------------
#
# Codex records a different shape from claude, so this is a separate parser
# rather than a branch inside the claude one:
#
#   * one `response_item` record per content block, but no `message.id` to
#     group on — a model response is instead delimited by the
#     `token_count` / `token_usage_record` event that reports its usage;
#   * tool calls and their outputs are *separate* records joined by
#     `call_id`, not nested tool_result blocks;
#   * model and effort live on `turn_context`, which appears when either
#     changes rather than on every turn, so they carry forward;
#   * usage is in OpenAI vocabulary, where `input_tokens` is gross and
#     `cached_input_tokens` is a subset of it.

# Codex's shell/exec tools, whose arguments carry the command line. Distinct
# from `_SHELL_TOOLS` because those names are matched case-insensitively for
# claude, and codex's lowercase tool names would otherwise be mistaken for
# in-process harness op names by `_harness_op`.
#
# `exec` is 0.153.4's name for the JS-REPL cell: a `custom_tool_call` whose
# `input` is *source code* calling `tools.exec_command({cmd:"..."})` rather
# than a JSON argument object. `exec_command` is the 0.146/0.147 name, where
# the same payload arrived as JSON on a `function_call`. Both generations are
# in play, so both names stay.
_CODEX_SHELL_TOOLS = {"exec", "exec_command", "shell", "local_shell",
                      "container.exec"}

# `tools.exec_command({cmd:"…"})` / `({command:'…'})` / `({cmd:`…`})` inside a
# 0.153.4 `exec` cell. Captures the opening quote so the closing one must
# match, and allows backslash escapes (the command is a JS string literal, so
# embedded quotes and newlines arrive escaped).
#
# The backtick alternative is not optional: a command containing both quote
# kinds, such as a harness call `"$DYCRAFT_PYTHON" -c '…'`, is emitted as a
# template literal, and without it every such call is missed.
_CODEX_JS_CMD_RE = re.compile(
    r"""(?:cmd|command)\s*:\s*(["'`])((?:\\.|(?!\1).)*)\1""", re.DOTALL)

# A harness call is `h.<op>(` — but that text also appears inside authored
# source. A `cat > object.py <<'PY' … h.compile() … PY` cell writes a file that
# *mentions* the op without invoking it, and a context summary can quote one
# verbatim (the failure `_SHELL_TOOLS`' comment records). So an op only counts
# when the command also shows the marks of actually driving the harness: the
# interpreter the prompts mandate, or an import of the staged client.
#
# `h` is also just a letter. Agents name a local shape `h` and then measure it
# — `b = h - Sphere(r)`, `h.solids()`, `h.bounding_box()`, `h.edges()` — and a
# scratch script full of that reads as six harness ops that never happened.
# This gate is what tells the two apart, so both transcript paths apply it now
# rather than only the codex one (audit I5).
_HARNESS_INVOCATION_RE = re.compile(
    r"DYCRAFT_PYTHON|import\s+harness|from\s+harness\s+import|harness\.py")


def _codex_tool_command(name: str, arguments: Any) -> str | None:
    """The shell command a codex tool call ran, if it ran one.

    Handles all three shapes seen across codex generations: a JSON string, an
    already-decoded dict, and 0.153.4's JS cell source.
    """
    if name not in _CODEX_SHELL_TOOLS:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            # 0.153.4: not JSON but JS source. Pull the command out of the
            # `tools.exec_command({cmd:"…"})` call rather than scanning the
            # whole cell, so surrounding JS cannot be mistaken for the command.
            matches = _CODEX_JS_CMD_RE.findall(arguments)
            if matches:
                return "\n".join(_unescape_js(body) for _, body in matches)
            return None
    if isinstance(arguments, dict):
        command = arguments.get("cmd") or arguments.get("command")
        if isinstance(command, list):
            return " ".join(str(c) for c in command)
        return command if isinstance(command, str) else None
    return None


_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
               '"': '"', "'": "'", "0": "\0"}


def _unescape_js(body: str) -> str:
    r"""Decode the escapes a JS string literal uses (`\n`, `\"`, `\\`, …).

    Only the handful codex actually emits; an unknown escape keeps its
    character, which is what a shell would have seen anyway.
    """
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            out.append(_JS_ESCAPES.get(body[i + 1], body[i + 1]))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _codex_harness_ops(name: str, arguments: Any) -> list[str]:
    """Every dycraft op a codex tool call invoked, in order.

    Anchored on the harness *invocation* rather than searched over the whole
    command. A single cell routinely writes source and then runs the harness:

        cat > object.py <<'PY' … h.compile() …  PY
        "$DYCRAFT_PYTHON" -c 'import harness as h; h.probe(...)'

    A plain search would return `compile` there — the op the authored file
    merely mentions — instead of `probe`, the op actually called. So we find
    where the harness is driven from and only look after that point.

    **One cell drives the harness more than once as a matter of course.** codex
    batches — `python -c 'h.compile()' && python -c 'h.snapshot()'` is the
    common shape, not the corner one — and each of those is a real op with its
    own line in `trace.jsonl`. Returning only the first left the rest of the
    trace unclaimed, so every later shell greedily matched a stale step and the
    joined sequence ran backwards: the bicycle_helmet arm joined 4, 6, 7, 9, 5,
    11 in that order.

    One op per *invocation*, not every match in the command, which is what
    keeps the anchoring above honest in both directions. A window opens at each
    invocation and ends at the next, and only the first op inside it counts —
    so the write-then-run cell reports the op it ran rather than the one its
    heredoc mentions, and the run-then-write cell does not pick up an op out of
    the source it went on to author.
    """
    command = _codex_tool_command(name, arguments)
    if not isinstance(command, str):
        return []
    anchors = [m.end() for m in _HARNESS_INVOCATION_RE.finditer(command)]
    found: list[str] = []
    for i, start in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else len(command)
        match = _HARNESS_OP_RE.search(command, start, end)
        if match and match.group(1) in _HARNESS_OP_NAMES:
            found.append(match.group(1))
    return found


def _codex_usage_block(payload: dict) -> dict | None:
    """Per-response usage from either rollout generation.

    0.146/0.147 report `event_msg/token_count` with `info.last_token_usage`;
    0.153+ additionally emit a `token_usage_record` whose `usage` is the same
    per-response figure. Either one closes a turn.
    """
    if payload.get("type") == "token_count":
        info = payload.get("info")
        last = info.get("last_token_usage") if isinstance(info, dict) else None
        return last if isinstance(last, dict) else None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _codex_text(content: Any) -> str:
    """Concatenate a codex message's text blocks."""
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("output_text", "input_text", "text"):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _codex_output_text(output: Any) -> str | None:
    """Flatten a codex tool output into display text.

    Three shapes across generations: a bare string (`function_call_output` in
    every version), a content-block list (`custom_tool_call_output` in
    0.153.4 — `[{"type": "input_text", "text": …}, …]`), and a dict wrapper.
    The list form mapped to None before, so every `exec` cell's output was
    lost even once it reached its call.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        inner = output.get("content")
        return _codex_output_text(inner) if inner is not None else str(output)
    if isinstance(output, list):
        text = _text_of(output)
        return text or None
    return str(output)


# 0.146/0.147 put the shell's exit status in the output's leading status line
# ("Process exited with code 1"), which is a real signal. Zero or absent means
# success.
#
# 0.153.4 does NOT: `custom_tool_call.status` is `"completed"` for every call
# (the tool completed, not the command succeeded) and the output header is
# always `Script completed` / `Script running with cell ID N`. The exit status
# is nowhere in the rollout. We deliberately do not guess from the body —
# grepping for "Traceback"/"Error:" flags successful `compile`s whose *report*
# legitimately contains the word, and a false error is worse than an unknown one. So `is_error` stays
# False on that generation and the result text carries the truth.
_CODEX_EXIT_RE = re.compile(
    r"(?:Process exited with code|Exit code:?)\s+(\d+)", re.IGNORECASE)


def _codex_output_is_error(payload: dict) -> bool:
    """Whether a codex tool output *reports* a failure.

    False also means "not reported" — see `_CODEX_EXIT_RE`. Never inferred
    from the output body.
    """
    for key in ("success", "ok"):
        if key in payload:
            return payload.get(key) is False
    text = _codex_output_text(payload.get("output")) or ""
    match = _CODEX_EXIT_RE.search(text)
    return bool(match) and match.group(1) != "0"


def _from_codex_transcript(path: Path, run_dir: Path) -> dict:
    """Parse a codex session rollout into the shared trajectory envelope."""
    ops = _harness_ops(run_dir)
    claims: list[tuple[dict, list[str]]] = []
    turns: list[dict] = []
    models_seen: list[str | None] = []
    prev_ts: datetime | None = None

    model: str | None = None
    effort: str | None = None
    # Accumulators for the response currently being assembled.
    texts: list[str] = []
    calls: list[dict] = []
    saw_reasoning_block = False
    started: datetime | None = None
    # Call-id -> call dict, scoped to the WHOLE ROLLOUT rather than to the
    # turn. In 0.153.4 the per-response usage record lands *between* a tool
    # call and its output:
    #
    #     custom_tool_call
    #     token_usage_record      <- the response ended here
    #     custom_tool_call_output <- the tool only finished now
    #
    # so a per-turn map is already cleared when the output arrives and every
    # result would be silently dropped. The 0.146/0.147 rollouts order these the
    # other way round, so fixtures built from them alone would not show it. Turn
    # dicts hold references to these same call dicts,
    # so attaching a result after the turn was appended still updates the
    # emitted turn.
    by_call_id: dict[str, dict] = {}
    # A 0.153+ rollout is read from its `token_usage_record`s alone, once per
    # `response_id`, exactly as `agent/usage.py::_from_codex` counts it. The
    # content guard below dropped every context compaction -- a response that
    # spends tokens and emits no text or tool call -- and a `token_count`
    # landing mid-response closed a turn early, so the same response was
    # counted twice. Keyed, the sum equals codex's own `thread_token_usage` on
    # all 291 local codex bundles. Only older rollouts, which carry no id,
    # keep the guard.
    keyed = any(r.get("type") == "token_usage_record" for r in _iter_jsonl(path))
    by_response_id: dict[str, dict] = {}

    def flush(usage_raw: dict, ts: datetime | None,
              response_id: str | None = None) -> None:
        nonlocal texts, calls, prev_ts, started
        if response_id is not None:
            if response_id in by_response_id:
                return
        # A `token_count` with nothing accumulated is a repeat of the previous
        # one (codex emits a duplicate at task end); emitting it would add an
        # empty turn and double-count its tokens.
        elif not texts and not calls:
            return
        cache_read = int(usage_raw.get("cached_input_tokens", 0) or 0)
        cache_write = int(usage_raw.get("cache_write_input_tokens", 0) or 0)
        # OpenAI vocabulary: `input_tokens` is gross and cached is a subset,
        # so subtract to avoid billing the cache twice.
        uncached = max(0, int(usage_raw.get("input_tokens", 0) or 0)
                       - cache_read - cache_write)
        usage = _usage(inp=uncached, cache_read=cache_read,
                       cache_write=cache_write,
                       out=int(usage_raw.get("output_tokens", 0) or 0))
        models_seen.append(model)
        priced = price_usage(usage, model)
        opened = started or ts
        turns.append({
            "index": len(turns) + 1,
            "ts": opened.isoformat() if opened else None,
            "elapsed_ms": _elapsed_ms(prev_ts, ts),
            "model": model,
            "effort": effort,
            # Codex reports no geo/tier/speed on usage, so there is no
            # per-turn multiplier to read; leaving these None is honest
            # rather than asserting standard rates.
            "billing": {"inference_geo": None, "service_tier": None,
                        "speed": None,
                        "multiplier": priced["multiplier"] if priced else None},
            "thought_summary": None,
            "text": "\n\n".join(texts) or None,
            "tool_calls": calls,
            "usage": usage,
            "cost_usd": priced["total"] if priced else None,
            "cost_buckets": priced["buckets"] if priced else None,
            # No per-response stop reason in the rollout; only a per-task
            # `task_complete`.
            "stop_reason": None,
        })
        if response_id is not None:
            by_response_id[response_id] = turns[-1]
        texts, calls = [], []
        started = None
        if ts is not None:
            prev_ts = ts

    for record in _iter_jsonl(path):
        kind = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        ts = _parse_ts(record.get("timestamp"))

        if kind == "turn_context":
            model = payload.get("model") or model
            effort = payload.get("effort") or effort
            continue

        if kind == "compacted":
            # Written just after the response that produced the summary, and
            # naming it -- so that turn is the compaction, tokens and all.
            turn = by_response_id.get(payload.get("compaction_response_id") or "")
            if turn is not None:
                turn["compaction"] = True
            continue

        if kind == "response_item":
            item = payload.get("type")
            if started is None:
                started = ts
            if item == "reasoning":
                # `summary` is empty and `encrypted_content` is opaque, so
                # there is no reasoning text to show — same ceiling as
                # claude's signature-only thinking blocks.
                saw_reasoning_block = True
            elif item == "message" and payload.get("role") == "assistant":
                text = _codex_text(payload.get("content"))
                if text:
                    texts.append(text)
            elif item in ("function_call", "custom_tool_call"):
                name = payload.get("name") or ""
                arguments = payload.get("arguments")
                if arguments is None:
                    arguments = payload.get("input")
                in_call = _codex_harness_ops(name, arguments)
                op = in_call[0] if in_call else None
                # The shell command, not the JavaScript wrapper codex takes it
                # in. `trajectory-adapter.ts` stringifies whatever this is
                # straight into the panel, so the raw cell source put a page of
                # `tools.exec_command({cmd:"…"})` boilerplate in front of the
                # one line a reader wants. Shaped as `{"command": …}` to match
                # the claude path's `input`, which `_detect_op` already reads
                # that way — so the envelope has one tool-call shape, not two.
                command = _codex_tool_command(name, arguments)
                call = {
                    "name": name,
                    "input": ({"command": command} if command is not None
                              else arguments),
                    "result": None,
                    "is_error": False,
                    "dycraft_op": op,
                    # Claimed once the output's timestamp is known, below.
                    "dycraft_step": None,
                    # Measured call record -> output record, the same span the
                    # claude path reports, so one envelope field means one
                    # thing on both. `finished_at` and `duration_ms` are filled
                    # in when the output arrives; the codex path cannot know
                    # them here because the output is written after the
                    # `token_usage_record` that closes this response.
                    "started_at": record.get("timestamp"),
                    "finished_at": None,
                    "duration_ms": None,
                }
                if op:
                    claims.append((call, in_call))
                calls.append(call)
                call_id = payload.get("call_id")
                if isinstance(call_id, str):
                    by_call_id[call_id] = call
            elif item in ("function_call_output", "custom_tool_call_output"):
                call = by_call_id.get(payload.get("call_id") or "")
                if call is not None:
                    call["result"] = _codex_output_text(payload.get("output"))
                    call["is_error"] = _codex_output_is_error(payload)
                    call["finished_at"] = record.get("timestamp")
                    call["duration_ms"] = _elapsed_ms(
                        _parse_ts(call.get("started_at")),
                        _parse_ts(record.get("timestamp")))
            continue

        if kind == "token_usage_record" or (
                kind == "event_msg" and payload.get("type") == "token_count"
                and not keyed):
            usage_raw = _codex_usage_block(payload)
            if usage_raw is not None:
                flush(usage_raw, ts, (payload.get("response_id") or None)
                      if kind == "token_usage_record" else None)

    # A run killed mid-response leaves content with no closing usage event.
    # Keep it, with zeroed tokens, rather than dropping the work silently.
    if texts or calls:
        flush({}, prev_ts)

    # Deferred to here because a call's output, which bounds the ops it can
    # have run, is written after the response that made the call. Still in
    # call order. The call reports its first op, but the others ran too and
    # have their own `trace.jsonl` lines; claiming them is what stops the next
    # shell matching a step this one already spent — see `_codex_harness_ops`.
    for call, in_call in claims:
        call["dycraft_step"] = _claim_step(
            ops, in_call[0], call["started_at"], call["finished_at"])
        for extra in in_call[1:]:
            _claim_step(ops, extra, call["started_at"], call["finished_at"])

    return {
        "source": "codex_cli",
        "transcript_path": str(path),
        "model": _dominant_model(models_seen),
        "models": _model_counts(models_seen),
        "reasoning_available": False,
        "reasoning_note": (
            "This agent's reasoning is stored encrypted, so none of it can be shown."
            if saw_reasoning_block else None
        ),
        "turns": turns,
    }


def _from_claude_transcript(path: Path, run_dir: Path) -> dict:
    records = list(_iter_jsonl(path))

    # Tool results arrive in the *following* user record, so index them first.
    # That record's own `timestamp` is when the tool finished, and it is the
    # only end time a non-harness tool call has — `Read`, `Edit`, `Skill`, and
    # plain shell never reach `trace.jsonl`, so without it they would have no
    # duration at all and could not be laid out on a time axis.
    results: dict[str, dict] = {}
    for rec in records:
        if rec.get("type") != "user":
            continue
        content = (rec.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if tid:
                    results[tid] = dict(block, _ts=rec.get("timestamp"))

    ops = _harness_ops(run_dir)
    turns: list[dict] = []
    prev_ts: datetime | None = None
    models_seen: list[str | None] = []
    saw_thinking = False
    saw_thinking_text = False

    for group in _group_assistant_records(records):
        message = group["message"]
        content = group["content"]
        ts = _parse_ts(group["timestamp"])

        thoughts, texts, tool_calls = [], [], []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("thinking", "redacted_thinking"):
                saw_thinking = True
                thought = (block.get("thinking") or "").strip()
                if thought:
                    saw_thinking_text = True
                    thoughts.append(thought)
            elif btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    texts.append(text)
            elif btype == "tool_use":
                name = block.get("name")
                tool_input = block.get("input")
                result = results.get(block.get("id") or "")
                op = _detect_op(name, tool_input)
                ended = _parse_ts(result.get("_ts")) if result else None
                tool_calls.append({
                    "name": name,
                    "input": tool_input,
                    "result": _text_of(result.get("content")) if result else None,
                    "is_error": bool(result.get("is_error")) if result else False,
                    "dycraft_op": op,
                    "dycraft_step": (_claim_step(
                        ops, op, block.get("_record_ts") or group["timestamp"],
                        result.get("_ts") if result else None)
                        if op else None),
                    # This block's own record, not the group's first: see
                    # `_group_assistant_records`. Calls emitted in one record
                    # can run concurrently, so several may share this start.
                    "started_at": block.get("_record_ts") or group["timestamp"],
                    "finished_at": result.get("_ts") if result else None,
                    "duration_ms": _elapsed_ms(
                        _parse_ts(block.get("_record_ts")) or ts, ended),
                })

        usage_raw = message.get("usage") or {}
        creation = usage_raw.get("cache_creation") or {}
        usage = _usage(
            inp=usage_raw.get("input_tokens", 0),
            cache_read=usage_raw.get("cache_read_input_tokens", 0),
            cache_write=usage_raw.get("cache_creation_input_tokens", 0),
            cache_write_5m=creation.get("ephemeral_5m_input_tokens", 0),
            cache_write_1h=creation.get("ephemeral_1h_input_tokens", 0),
            out=usage_raw.get("output_tokens", 0),
        )
        # Billing mode is recorded per turn, so read it rather than assume
        # standard rates. "not_available" is the common value for geo.
        geo = usage_raw.get("inference_geo")
        tier = usage_raw.get("service_tier")
        speed = usage_raw.get("speed")
        turn_model = message.get("model")
        models_seen.append(turn_model)
        priced = price_usage(usage, turn_model, inference_geo=geo,
                             service_tier=tier, speed=speed)
        turns.append({
            "index": len(turns) + 1,
            "ts": group["timestamp"],
            "elapsed_ms": _elapsed_ms(prev_ts, ts),
            # The window the model actually held the floor for: unblocked by
            # the previous record, done at its own last block. `elapsed_ms` is
            # the gap between consecutive turn starts and so includes the tool
            # time in between; this does not, which is what the timeline's
            # model track needs to avoid double-counting a lane's own bars.
            "began_at": group.get("prev_timestamp"),
            "ended_at": group.get("last_timestamp"),
            # One entry per assistant record, in order. The CLI writes a record
            # per content block, so these timestamps bracket the turn's
            # internal phases: how long it thought, and how long it spent
            # composing the tool call it ended on. `agent_timeline` turns them
            # into the phases it draws.
            "records": group.get("records") or [],
            "generation_ms": _elapsed_ms(
                _parse_ts(group.get("prev_timestamp")),
                _parse_ts(group.get("last_timestamp"))),
            "model": turn_model,
            "effort": group.get("effort"),
            "billing": {"inference_geo": geo, "service_tier": tier,
                        "speed": speed,
                        "multiplier": priced["multiplier"] if priced else None},
            "thought_summary": "\n\n".join(thoughts) or None,
            "text": "\n\n".join(texts) or None,
            "tool_calls": tool_calls,
            "usage": usage,
            "cost_usd": priced["total"] if priced else None,
            "cost_buckets": priced["buckets"] if priced else None,
            "stop_reason": message.get("stop_reason"),
        })
        if ts is not None:
            prev_ts = ts

    return {
        "source": "claude_cli",
        "transcript_path": str(path),
        "model": _dominant_model(models_seen),
        "models": _model_counts(models_seen),
        "reasoning_available": saw_thinking_text,
        "reasoning_note": (
            None if saw_thinking_text else
            "This agent's reasoning is stored as a signature without its text, so none of it can be shown."
            if saw_thinking else
            "This run's transcript contains no thinking blocks."
        ),
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def _totals_only(run_dir: Path) -> dict:
    """The envelope for a run with no readable transcript.

    Empty, unless the bundle recorded its own `meta.usage`, which is how a
    run whose transcript this module cannot reach still reports its usage.

    Both staged formats are parsed (claude sessions and codex rollouts), so
    reaching here means the transcript is genuinely missing rather than merely
    unreadable — an interrupted run, or a bundle whose CLI session store was
    cleaned before it was staged.
    """
    empty = _usage() | {"turns": 0, "cost_usd": None, "cost_buckets": {},
                        "tool_calls": 0, "cost_complete": False,
                        "unpriced_models": []}
    note = ("No agent trajectory found. External-CLI backends keep it in the "
            "CLI's own session directory; it is staged into the bundle as "
            "agent_transcript.jsonl when a run finishes.")
    usage = None
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        usage = meta.get("usage") if isinstance(meta, dict) else None
    except (OSError, ValueError):
        pass
    if isinstance(usage, dict) and not usage.get("error"):
        for key, src in (("input", "input"), ("cache_read", "cache_read"),
                         ("cache_write", "cache_write"), ("output", "output"),
                         ("gross_input", "gross_input"), ("total", "total")):
            if usage.get(src) is not None:
                empty[key] = int(usage[src])
        empty["turns"] = int(usage.get("turns") or 0)
        empty["tool_calls"] = int(usage.get("tool_calls") or 0)
        note = (f"Totals only. This run's loop was recorded by "
                f"{usage.get('format') or 'another'} in a format this viewer "
                f"does not render turn by turn, so the numbers below come from "
                f"meta.json's `usage` block ({usage.get('source') or 'unknown'})"
                f" rather than from a staged transcript.")
    return {"source": None, "transcript_path": None, "model": None,
            "reasoning_available": False, "reasoning_note": note,
            "models": [], "turns": [], "totals": empty}

def _parse_cli_transcript(path: Path, run_dir: Path) -> dict:
    """Dispatch a staged transcript to the parser for the CLI that wrote it.

    Sniffed from the content rather than from `meta.json`'s `runtime`, because
    the bundle may have been staged by an older launcher (no `runtime`), and
    because a mismatch here is silent: handing a codex rollout to the claude
    parser yields zero turns and all-zero tokens, which renders exactly like a
    real-but-empty run instead of like a bug. A codex rollout always opens
    with a `session_meta` record; a claude transcript never does.
    """
    for record in _iter_jsonl(path):
        if record.get("type") == "session_meta":
            return _from_codex_transcript(path, run_dir)
        break
    return _from_claude_transcript(path, run_dir)


#: The trajectories built most recently, `run_dir -> (stamp, trajectory)`. A browser asks for a run's trajectory and
#: its timeline together, and the timeline is built on the trajectory, so without this a long run's transcript and
#: trace were each parsed twice per selection. Two, because one is tens of megabytes.
_BUILT: dict[str, tuple[tuple, dict]] = {}
_BUILT_KEEP = 2
_BUILT_LOCK = threading.Lock()
_BUILD_LOCKS: dict[str, threading.Lock] = {}


def _stamp(path: Path | None) -> tuple:
    if path is None:
        return (None,)
    try:
        st = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), st.st_ino, st.st_size, st.st_mtime_ns)


def build(run_dir: Path) -> dict:
    """The run's trajectory, or an empty envelope when no trace exists.

    Built once for each version of the files it is read from, the transcript, `trace.jsonl`, and `meta.json`, and
    shared by every caller: read it, do not change it. A caller arriving while it is being built waits for that build.
    """
    run_dir = Path(run_dir)
    key = str(run_dir)
    with _BUILT_LOCK:
        lock = _BUILD_LOCKS.setdefault(key, threading.Lock())
    with lock:
        transcript = resolve_transcript(run_dir)
        stamp = (_stamp(transcript), _stamp(run_dir / "trace.jsonl"), _stamp(run_dir / "meta.json"))
        hit = _BUILT.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        result = _build(run_dir, transcript)
        with _BUILT_LOCK:
            _BUILT.pop(key, None)
            _BUILT[key] = (stamp, result)
            while len(_BUILT) > _BUILT_KEEP:
                _BUILT.pop(next(iter(_BUILT)))
        return result


def _build(run_dir: Path, transcript: Path | None) -> dict:
    if transcript is None:
        return _totals_only(run_dir)
    result = _parse_cli_transcript(transcript, run_dir)

    turns = result["turns"]
    totals = _usage()
    for key in ("input", "cache_read", "cache_write", "cache_write_5m",
                "cache_write_1h", "output", "gross_input", "total"):
        totals[key] = sum(t["usage"][key] for t in turns)

    buckets: dict[str, float] = {}
    for turn in turns:
        for name, usd in (turn.get("cost_buckets") or {}).items():
            buckets[name] = buckets.get(name, 0.0) + usd
    costs = [t["cost_usd"] for t in turns if t["cost_usd"] is not None]
    # A partial total is worse than none: if any turn's model is unpriced the
    # sum silently understates the run, so report the gap instead of hiding it.
    missing = unpriced_models(t.get("model") for t in turns)
    totals["cost_usd"] = round(sum(costs), 6) if costs else None
    totals["cost_buckets"] = {k: round(v, 6) for k, v in buckets.items()}
    totals["cost_complete"] = bool(costs) and not missing
    totals["unpriced_models"] = missing
    totals["turns"] = len(turns)
    totals["tool_calls"] = sum(len(t["tool_calls"]) for t in turns)
    result["totals"] = totals
    result.setdefault("models", [])
    return result
