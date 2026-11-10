"""AgentSession — generic per-run agent runtime shared by every agent class.

What this module owns (agent-agnostic plumbing):

  * `TraceMeta`: byte-level
    `trace.jsonl` + `meta.json` writer with outcome-rank-aware finalize.
  * `AgentContext` (Protocol): the duck-typed shape every per-run context
    must satisfy — `run_dir`, `step`, `committed`, and an optional
    `meta_extras()` hook for agent-specific meta.json fields.
  * `AgentSession[CtxT]`: a generic base that wraps an AgentContext with
      - step counter + `tick()`
      - `TraceMeta` façade (`trace`, `write_initial_meta`, `finalize_meta`)
      - `committed` property bridging from ctx
      - `op(name)` decorator that wraps `(**kwargs) -> dict` op-fns with the
        canonical `tick → trace(op_call) → run → trace(op_result)` pattern
        and registers them in `self.handlers` for the harness daemon to
        dispatch by name.

Agent-specific subclasses (`LayerSession`, `FilmSession`, `PlainSession`)
register their own ops via `@self.op("name")` and add no further plumbing.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

# Everything `finalize_meta` calls is imported here, not where it is used: at
# shutdown the worktree this package was imported from may already be gone,
# and an import then raises `ModuleNotFoundError` and loses the record.
from .transcript_capture import approval_reviewer_usage, capture, subagent_usage
from .usage import summarize


# ---------------------------------------------------------------------------
# TraceMeta
# ---------------------------------------------------------------------------


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Any outcome not listed here ranks 0 (`.get(outcome, 0)`), which is why
#: `daemon_failed` is listed rather than left to the default: it is written
#: directly into meta.json by `record_launch_failure` (below, called from the
#: launcher) for a daemon that died before binding, and at rank 0 a later
#: `finalize("interrupted")` from a stale sweep would have overwritten it with
#: the vaguer verdict. Equal rank to `interrupted` means neither displaces the
#: other, while a genuine commit still wins: `delivered_with_exceptions`, and
#: above it a clean `committed`.
_OUTCOME_RANK = {None: 0, "interrupted": 1, "daemon_failed": 1,
                 "delivered_with_exceptions": 2, "committed": 3}


class TraceMeta:
    """Shared writer for `trace.jsonl` + `meta.json` artifacts.

    Outcome ordering on `finalize`: None < {"interrupted", "daemon_failed"}
    < "delivered_with_exceptions" < "committed". A later call with a
    higher-ranked outcome overrides an earlier one; a lower- or equal-ranked
    one keeps the outcome and only moves `finished_at`. This matters because
    an atexit sweep from a stale daemon could otherwise downgrade a committed
    run, or freeze a pre-commit "interrupted" verdict.

    **Outcomes are an open string set**, so anything absent from
    `_OUTCOME_RANK` ranks 0 and is displaceable by any real verdict.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._trace_path = run_dir / "trace.jsonl"
        self._meta_path = run_dir / "meta.json"

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    @property
    def meta_path(self) -> Path:
        return self._meta_path

    def trace(self, ev: str, *, step: int = 0, **fields) -> None:
        # `epoch` is the writing process, and `daemon_start` carries the same
        # pid. Steps carry on from daemon to daemon (`highest_trace_step`); a
        # trace written before they did can repeat a step number, which
        # `(epoch, step)` still tells apart.
        payload = {"ev": ev, "step": step, "epoch": os.getpid(),
                   "ts": utcnow_iso(), **fields}
        try:
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        except OSError:
            pass

    def write_initial(self, **fields) -> None:
        """Merge-aware initial write.

        If `meta.json` already exists (e.g. the launcher wrote a base record
        before starting the agent binary), this call adds/overrides only
        the supplied keys — `started_at`, `outcome`, `run_id`, etc. are
        preserved from the pre-existing file. This eliminates the api-mode
        double-write where the launcher clobbered the runner's superset
        (or vice versa).
        """
        existing: dict = {}
        if self._meta_path.exists():
            try:
                existing = json.loads(self._meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = {}
        meta = {
            "run_id": self.run_dir.name,
            "outcome": None,
            "started_at": utcnow_iso(),
            "finished_at": None,
            **existing,
            **fields,
        }
        self._meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def finalize(self, outcome: str, **fields) -> None:
        if not self._meta_path.exists():
            return
        try:
            meta = json.loads(self._meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        current = meta.get("outcome")
        if _OUTCOME_RANK.get(outcome, 0) > _OUTCOME_RANK.get(current, 0):
            meta["outcome"] = outcome
        # When the run last stopped, whatever it reached: a resumed run that
        # ends again moves it on.
        meta["finished_at"] = utcnow_iso()
        meta.update(fields)
        try:
            self._meta_path.write_text(
                json.dumps(meta, indent=2, sort_keys=True))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# AgentContext protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentContext(Protocol):
    """Duck-typed shape every per-run agent context must satisfy.

    Contexts (`ToolContext`, `FilmContext`, `PlainContext`) typically add many more
    fields; only these are needed by the generic AgentSession.
    """

    run_dir: Path
    step: int
    committed: bool


def _meta_extras(ctx: Any) -> dict:
    """Pull agent-specific meta.json fields off the context.

    Two conventions are accepted:
      - `ctx.meta_extras() -> dict` — the film and plain contexts.
      - `ctx.layer_cfg.layer_meta(ctx) -> dict` — used by the object/scene
        ToolContext, which carries a LayerConfig.
    Exceptions inside the impl propagate: a layer_meta crash is a bug in the
    layer, not something to swallow into an incomplete meta.json.
    """
    fn = getattr(ctx, "meta_extras", None)
    if callable(fn):
        return fn() or {}
    layer_cfg = getattr(ctx, "layer_cfg", None)
    if layer_cfg is not None and callable(getattr(layer_cfg, "layer_meta", None)):
        return layer_cfg.layer_meta(ctx) or {}
    return {}


#: How much of a failing op's reason to put on its `op_result` trace record.
#: Long enough for a verifier summary or an exception repr, short enough that a
#: 200-op run's trace stays greppable. Reasons are naturally bounded (the
#: longest is a Python traceback through the harness, and `layers/compile.py`
#: caps build tracebacks at 4096 upstream), so this matches that cap.
_REASON_MAX_CHARS = 4096

#: Keys an op payload uses to say *why* it failed, in the order they are read.
#: `error` is the house convention (49 sites across `agent/`); `error_kind` and
#: `outcome` are short enumerations that some ops carry instead of or beside
#: it — `await_any`'s `timeout` / `all_reported` / `nothing_dispatched` are
#: only in `outcome`, and they are the majority of that op's "failures".
_REASON_KEYS = ("error", "error_kind", "outcome", "build_error",
                "error_summary", "failure_class")

#: Payload keys that are not a failure *reason* and are too long or too raw for
#: `trace.jsonl`: process output, and up to 4000 chars of a reviewer's prose.
_REASON_FORBIDDEN = frozenset({"stderr_tail", "reply_text", "stdout", "stderr"})


def _run_shape(run_dir: Any, ctx: Any) -> dict:
    """How much work the run did, for `meta.json`. Never raises.

    Without it a run that solved its scene in twelve steps reads like one that
    ground through seventy, short of replaying `trace.jsonl`. There is no step
    limit to report: nothing caps the number of operations.

    `usage` comes from `agent.usage.summarize`, which reads the transcript
    `capture()` has just staged. It is written for *every* bundle, child runs
    included: the dispatched object agents and review sub-agents are most of an
    animation run's cost. A codex run's own sub-agents are added as
    `usage.subagents` (see `transcript_capture.subagent_usage`).
    """
    out: dict = {}
    step = getattr(ctx, "step", None)
    if isinstance(step, int):
        out["steps"] = step
    advisory = os.environ.get("DYCRAFT_ADVISORY_HOURS")
    if advisory:
        try:
            out["advisory_hours"] = float(advisory)
        except ValueError:
            pass
    if not run_dir:
        return out

    # Counted from the trace rather than tracked in memory, so a resumed or
    # re-entered run reports the whole bundle's history and not just this
    # process's share of it.
    attempts = failures = ops = op_failures = 0
    op_errors = check_findings = op_rejections = 0
    highest_attempt = 0
    jobs = jobs_cancelled = jobs_killed = 0
    job_errors = job_check_findings = job_rejections = 0
    try:
        with open(Path(run_dir) / "trace.jsonl", errors="replace") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                event = record.get("ev")
                # Interactive jobs too: a killed worker is the same fact
                # whichever path ran it.
                if event == "job_result" and _killed(record):
                    jobs_killed += 1
                if event == "compile_attempt":
                    attempts += 1
                    number = record.get("attempt")
                    if isinstance(number, int):
                        highest_attempt = max(highest_attempt, number)
                    # The record's own verdict. Not `build_error or
                    # verifier_summary`: `verifier_summary` is written on every
                    # attempt, passing or failing.
                    if not record.get("ok"):
                        failures += 1
                elif event == "op_result":
                    ops += 1
                    if not record.get("ok"):
                        op_failures += 1
                        kind = _failure_class(record)
                        if kind == "rejection":
                            op_rejections += 1
                        elif kind == "check_finding":
                            check_findings += 1
                        else:
                            op_errors += 1
                elif event == "job_result" and not record.get("interactive"):
                    # A batch op returns its handle successfully and reports
                    # its outcome here, later; most failures of a long run are
                    # here rather than in `op_result`. An interactive op writes
                    # one of these too, but its `op_result` already counted it.
                    jobs += 1
                    result = record.get("result")
                    result = result if isinstance(result, dict) else {}
                    if record.get("status") == "cancelled":
                        jobs_cancelled += 1
                    elif record.get("status") != "completed" or not result.get("ok"):
                        kind = _failure_class(result)
                        if kind == "rejection":
                            job_rejections += 1
                        elif kind == "check_finding":
                            job_check_findings += 1
                        else:
                            job_errors += 1
    except OSError:
        return out
    if attempts:
        out["compile_attempts"] = attempts
        out["compile_failures"] = failures
        # Not a replacement for the count. A compile that raised consumed a
        # number and wrote no record, so the two disagree exactly when the
        # ledger has a gap.
        if highest_attempt:
            out["compile_attempt_max"] = highest_attempt
    if ops:
        out["ops"] = ops
        # Three numbers, because they mean opposite things: a check that ran
        # perfectly and reported a defect in the *film* is not an error, and
        # counting both together makes the most thorough run look like the one
        # that failed most.
        out["op_errors"] = op_errors
        out["check_findings"] = check_findings
        out["op_rejections"] = op_rejections
        # Retained as their sum while readers migrate.
        out["op_failures"] = op_failures
    if jobs:
        # Under their own names, so the three above keep meaning what readers
        # already take them to mean.
        out["jobs"] = jobs
        out["job_errors"] = job_errors
        out["job_check_findings"] = job_check_findings
        out["job_rejections"] = job_rejections
        out["jobs_cancelled"] = jobs_cancelled
    if jobs_killed:
        out["jobs_killed"] = jobs_killed
    try:
        block = summarize(run_dir)
    except BaseException:                                       # noqa: BLE001
        block = None
    # The run's own transcript is one thread; a codex run's sub-agents are
    # others. Their figures go beside the run's rather than into them, so
    # `usage` itself keeps agreeing with the viewer's trajectory of the staged
    # transcript, and `total_with_subagents` is what the run cost in all.
    try:
        spawned = subagent_usage(run_dir) if run_dir else None
    except BaseException:                                       # noqa: BLE001
        spawned = None
    if spawned:
        block = dict(block or {})
        block["subagents"] = spawned
        block["total_with_subagents"] = (int(block.get("total") or 0)
                                         + int(spawned.get("total") or 0))
    if block:
        out["usage"] = block
    return out


def _elide(value: str, limit: int = _REASON_MAX_CHARS) -> str:
    """Truncate from the middle, never off the tail.

    A Python traceback puts the useful part last: the exception type, its
    message, and the value that caused it. Head-truncation keeps the call
    stack -- mostly harness frames nobody can act on -- and drops the
    diagnosis, often mid-path in the one line that names the problem. Same byte
    budget, both ends kept.
    """
    if len(value) <= limit:
        return value
    head = limit // 3
    tail = limit - head - 24
    dropped = len(value) - head - tail
    return f"{value[:head]}\n...<{dropped} characters elided>...\n{value[-tail:]}"


#: Set by the few returns that refuse to run an operation at all, rather than
#: running it and reporting what it found. See `_failure_class`.
_REJECTION_MARKERS = (
    "a project edit is pending",
    "project is committed",
    # The storyboard gates, not every text that names a storyboard: "stage
    # must be 'storyboard' or 'animation'" is the caller's error.
    "storyboard evidence",
    "before an animation edit",
    "before animation/delivery",
    "before animating this span",
    "with storyboard() first",
    "re-draw them with storyboard(",
    "wait for pending edits",
    "outstanding contributions",
    "repair candidate is stale",
    "project changed before completion",
    "job cancelled",
    # Retired texts, so that traces already on disk reclassify.
    "project changed while this candidate was running",
    "delivery frames are verified",
)

#: A worker the kernel or a signal ended: `-9` is what the OOM killer leaves.
#: The fork's form, then the resident's. A canceled job is never counted.
_KILLED = re.compile(r"^Blender worker exited -(9|15)\b|\(exit -(9|15)\)")


def _killed(record: dict) -> bool:
    """Whether a `job_result` is a job whose Blender was killed under it."""
    result = record.get("result")
    error = result.get("error") if isinstance(result, dict) else None
    return (record.get("status") == "failed" and isinstance(error, str)
            and bool(_KILLED.search(error)))


def _failure_class(record: dict) -> str:
    """Which of three things a not-ok `op_result` was.

    - **error** -- an exception or a sandbox violation. The op did not run.
      A reply declares which: `author_code`, raised in the agent's own code,
      or `harness`, a fault in the harness; both count here.
    - **check_finding** -- the op ran exactly as asked and reports that the
      *film* has a defect. Not a failure of the operation at all.
    - **rejection** -- the harness declined to run it: a pending edit, a
      committed project, an unreviewed storyboard. Also not a result, but
      unlike an error there is nothing wrong with the request except when it
      arrived.

    Read off the record's own `failure_class` where a handler set one, and
    inferred from the reason text otherwise so that traces already on disk
    reclassify. There is deliberately no fourth "gate leg" bucket: the binding
    gate does not refuse benign topology churn (`update.binding_modifiers`), so
    what it does refuse is a genuine rejection.
    """
    declared = record.get("failure_class")
    if declared in ("author_code", "harness"):
        return "error"
    if declared in ("error", "check_finding", "rejection"):
        return declared
    if record.get("verifier_summary") is not None or record.get("check_findings"):
        return "check_finding"
    reason = " ".join(str(record.get(key) or "") for key in
                      ("error", "error_summary", "outcome")).lower()
    if any(marker in reason for marker in _REJECTION_MARKERS):
        return "rejection"
    return "error"


def _failure_reason(payload: dict) -> dict:
    """Why a failing op failed, as trace-record fields.

    Without these fields a failing `op_result` says only that the op failed;
    the reason would have to be dug out of the agent transcript.

    Reads the reason off the payload the op already returns rather than asking
    ops to report twice, so no handler changes and an op that says nothing
    still records nothing. `_REASON_FORBIDDEN` is a deliberate deny-list, not
    an oversight (see its comment).
    """
    out: dict = {}
    for key in _REASON_KEYS:
        if key in _REASON_FORBIDDEN:          # defensive; _REASON_KEYS is fixed
            continue
        value = payload.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            value = repr(value)
        out[key] = _elide(value)
    return out


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------


CtxT = TypeVar("CtxT", bound=AgentContext)


#: A step in `trace.jsonl` as `TraceMeta.trace` writes it (`json.dumps(..., sort_keys=True)`).
_TRACE_STEP = re.compile(rb'"step": (\d+)')


def highest_trace_step(run_dir: Path) -> int:
    """The highest step `run_dir`'s trace records, or 0. Read with a pattern rather than parsed, since a long run's
    trace runs to tens of megabytes; the whole file, since a trace written before steps carried on can number
    the same steps more than once."""
    try:
        data = (Path(run_dir) / "trace.jsonl").read_bytes()
    except OSError:
        return 0
    return max((int(m) for m in _TRACE_STEP.findall(data)), default=0)


_TRACE_EPOCH = re.compile(rb'"epoch": (\d+)')


def agent_time(run_dir: Path) -> dict:
    """How long the run's agent sessions ran, from its trace: `{"seconds", "launches": [...]}`, one launch per daemon
    (`daemon_start` to its last event; the daemon lives as long as its launcher and starts after the rewriters).

    From the trace rather than kept by the launcher, so a launcher that was killed still counts up to its last
    harness event, and the time between launches (waiting for a usage limit to reset) never does.
    """
    first: dict[bytes, bytes] = {}
    last: dict[bytes, bytes] = {}
    try:
        with open(Path(run_dir) / "trace.jsonl", "rb") as fh:
            for line in fh:
                found = _TRACE_EPOCH.search(line)
                if not found:
                    continue
                epoch = found.group(1)
                if b'"ev": "daemon_start"' in line and epoch not in first:
                    first[epoch] = line
                last[epoch] = line
    except OSError:
        pass
    launches = []
    for epoch, start_line in first.items():
        try:
            start, end = json.loads(start_line), json.loads(last[epoch])
        except ValueError:
            continue
        began, ended = _epoch_seconds(start.get("ts")), _epoch_seconds(end.get("ts"))
        if began is None or ended is None:
            continue
        launches.append({"started_at": start.get("ts"), "ended_at": end.get("ts"),
                         "seconds": round(max(0.0, ended - began), 1),
                         "how": "daemon_shutdown" if end.get("ev") == "daemon_shutdown" else "last_event"})
    launches.sort(key=lambda launch: str(launch["started_at"]))
    return {"seconds": round(sum(launch["seconds"] for launch in launches), 1), "launches": launches}


def _epoch_seconds(value) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class AgentSession(Generic[CtxT]):
    """Generic agent-runtime base — owns plumbing, knows nothing about ops.

    Construction:
        session = AgentSession(ctx, user_prompt=..., initial_meta_fields={...})

    Used as a context manager (`with session: ...`); entering and leaving
    hold nothing.
    """

    def __init__(
        self,
        ctx: CtxT,
        *,
        user_prompt: str = "",
        initial_meta_fields: dict | None = None,
    ):
        self.ctx = ctx
        self.run_dir: Path = ctx.run_dir
        # A run's steps go on from the highest its trace holds: every daemon starts a new session, and one that
        # restarted at 0 (a resume, a daemon relaunched mid-run) would overwrite `compile_reports/<step>.json`,
        # `motion_checks/<step>.json` and the renders of the steps before it.
        ctx.step = max(ctx.step, highest_trace_step(ctx.run_dir))
        self.user_prompt = user_prompt
        self._initial_meta_fields = dict(initial_meta_fields or {})
        # `ctx.step += 1` is a read-modify-write, and the daemon runs handlers
        # on a thread pool — two ops ticking at once could otherwise land on
        # the same number. The lock also lets an op learn the step *it* was
        # given, which is not the same as the counter's later value: a handler
        # that runs for minutes will see it move underneath.
        self._step_lock = threading.Lock()
        self._op_step = threading.local()

        self._tm = TraceMeta(ctx.run_dir)

        # Op registry — populated via the @op decorator. Daemon-side servers
        # ship this dict directly to agent.server.serve().
        self.handlers: dict[str, Callable[..., dict]] = {}

    # ------------------------------------------------------------------ life

    def __enter__(self) -> "AgentSession[CtxT]":
        return self

    def __exit__(self, *exc):
        return None

    # ------------------------------------------------------------------ meta

    def write_initial_meta(self, **fields) -> None:
        """Write the initial meta.json. Merges:
            base TraceMeta fields  ←  prompt  ←  ctx.meta_extras()
            ←  self._initial_meta_fields  ←  **fields (last-wins).
        """
        extras: dict = {"prompt": self.user_prompt}
        extras.update(_meta_extras(self.ctx))
        if self._initial_meta_fields:
            extras.update(self._initial_meta_fields)
        if fields:
            extras.update(fields)
        self._tm.write_initial(**extras)

    def finalize_meta(self, outcome: str, **fields) -> None:
        """Refresh meta.json with outcome / finished_at + ctx meta_extras."""
        run_dir = getattr(self.ctx, "run_dir", None)
        # Stage the external agent CLI's session transcript (per-turn tokens +
        # conversation record) into the bundle. Overwrites on every call, so
        # the daemon's shutdown call — which runs after the agent process is
        # gone — is the one that lands a complete file; an earlier call from
        # the agent's own `finish` would capture a truncated transcript.
        #
        # `capture` copies a file that runs to megabytes, and this method is
        # called on the shutdown path — so the copy can be interrupted by the
        # very signal that is ending the process. It was: a `.daemon.log` shows
        # `finalize_meta` -> `capture` -> `source.read_bytes()` cut short by
        # `SystemExit: 0`, which escaped `capture`'s `except OSError`, skipped
        # the `finalize` below, and left the bundle with **neither** a
        # transcript nor an outcome in `meta.json`. Only one of those two is
        # recoverable afterwards — the rollout is still in the CLI's session
        # store and can be re-staged, while a missing outcome makes the run
        # indistinguishable from one that died mid-flight.
        #
        # So the exception is held, `meta.json` is written, and only then is it
        # re-raised: the shutdown still happens, one step later, having lost
        # the thing that can be recovered rather than the thing that cannot.
        transcript_fields: dict = {}
        pending: BaseException | None = None
        if run_dir:
            try:
                status = capture(run_dir)
            except BaseException as exc:                         # noqa: BLE001
                status, pending = "interrupted", exc
            if status != "unavailable":
                transcript_fields = {"agent_transcript": status}
        # What the approval reviewer cost, when there was one. Kept out of the
        # `capture` try/except above deliberately: that block holds an
        # interrupt so `meta.json` still gets written, and this must not be
        # able to introduce a second failure into the same path.
        reviewer_fields: dict = {}
        if run_dir:
            reviewer = approval_reviewer_usage(run_dir)
            if reviewer:
                reviewer_fields = {"approval_reviewer": reviewer}
        merged = {**_meta_extras(self.ctx), **reviewer_fields,
                  **transcript_fields, **_run_shape(run_dir, self.ctx),
                  **fields}
        self._tm.finalize(outcome, **merged)
        if pending is not None:
            raise pending

    def trace(self, ev: str, *, step: int | None = None, **fields) -> None:
        """Append a single trace event, stamped with `step` or the run's latest.

        The latest is right for an event about the run and wrong for the
        result of an op that others overtook while it ran: `op()` passes the
        step `tick()` handed it, as `run_interactive` already does for its
        `submission_step`.
        """
        self._tm.trace(ev, step=self.ctx.step if step is None else step, **fields)

    def tick(self) -> int:
        """Bump the per-run step counter and return the value taken.

        Both runner and daemon call this once per agent-driven action. The
        return value is this caller's own step; `ctx.step` is the run's latest
        and will have moved on by the time a long handler finishes. A handler
        that stamps artifacts wants the former — see `current_op_step`.

        """
        with self._step_lock:
            self.ctx.step += 1
            step = self.ctx.step
        self._op_step.value = step
        return step

    def current_op_step(self) -> int | None:
        """The step `tick()` handed this thread, or None outside an op."""
        return getattr(self._op_step, "value", None)

    # ------------------------------------------------------------------ ops

    def register_handler(self, name: str, fn: Callable[..., dict]) -> None:
        """Add `fn` to self.handlers under `name` WITHOUT wrapping.

        Use this when the underlying tool fn already does its own step
        bookkeeping and emits its own (richer-than-op_call/op_result)
        trace events, which wrapping with `@self.op` would double-trace. The daemon
        dispatcher calls `handlers[name](**kwargs) -> dict` either way,
        so adapters that translate from the LLM's `dict-args → list[Text]`
        tool surface live in the caller.
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"AgentSession.register_handler: invalid name {name!r}")
        if name in self.handlers:
            raise ValueError(
                f"AgentSession.register_handler: duplicate {name!r}")
        self.handlers[name] = fn

    def annotate_payload(self, op_name: str, payload: dict) -> dict:
        """Hook: add layer state to every op result. Identity by default.

        It exists for work that finishes on its own schedule. A CLI agent is
        turn-based and cannot be interrupted, so a background result has to
        reach it through something it was going to do anyway — the next
        `compile` or `probe` carries the news. Overridden by the object/scene
        Session for object dispatches.
        """
        return payload

    def op_scope(self):
        """Context entered around every op's handler. A no-op here; a layer
        overrides it for per-op state its tools keep (see
        `LayerSession.op_scope`)."""
        return nullcontext()

    def op(self, name: str) -> Callable[[Callable[..., dict]],
                                        Callable[..., dict]]:
        """Register `fn` as the handler for op `name`.

        """
        if not name or not isinstance(name, str):
            raise ValueError(f"AgentSession.op: invalid name {name!r}")
        if name in self.handlers:
            raise ValueError(f"AgentSession.op: duplicate registration {name!r}")

        def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
            def wrapped(**kwargs) -> dict:
                with self.op_scope():
                    return run(**kwargs)

            def run(**kwargs) -> dict:
                # The call's own step, on both records. Stamping `op_result`
                # from `ctx.step` gave it whatever step the run had reached by
                # the time the handler returned, so two overlapping ops could
                # file their results under one number.
                step = self.tick()
                self.trace("op_call", step=step, op=name, args=kwargs)
                t0 = time.monotonic()
                try:
                    payload = fn(**kwargs)
                except BaseException as exc:
                    # A raising op still gets its `op_result`, and still
                    # raises: the dispatcher and the transport rely on that.
                    # No `failure_class`: a raised refusal ("wait for
                    # pending edits") is a rejection, which `_failure_class`
                    # reads off the text; a declared "error" would mask it.
                    self.trace("op_result", step=step, op=name, ok=False,
                               elapsed_ms=int((time.monotonic() - t0) * 1000),
                               error_type=type(exc).__name__,
                               error=_elide(str(exc) or repr(exc)))
                    raise
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                result_fields = {
                    "op": name,
                    "ok": bool(payload.get("ok")) if isinstance(payload, dict) else False,
                    "elapsed_ms": elapsed_ms,
                }
                if isinstance(payload, dict) and "cache_hit" in payload:
                    result_fields["cache_hit"] = payload.get("cache_hit")
                if isinstance(payload, dict) and not result_fields["ok"]:
                    result_fields.update(_failure_reason(payload))
                self.trace("op_result", step=step, **result_fields)
                # A layer may fold live background state into every result. The
                # default does nothing; see `annotate_payload`.
                if isinstance(payload, dict):
                    payload = self.annotate_payload(name, payload)
                return payload
            wrapped.__name__ = name
            self.handlers[name] = wrapped
            return wrapped
        return decorator


def write_run_meta(
    run_dir: Path,
    *,
    layer: str,
    runtime: str,
    prompt: str,
    run_id: str,
    model: str | None,
    extra: dict | None = None,
) -> Path:
    """Write the base `meta.json` for a run.

    Called by the CLI launcher before starting an external backend
    (claude/codex) so the run dir has a meta.json on disk *before* the
    agent process exists to write one itself. The daemon's
    `AgentSession.write_initial_meta` merges over it (`TraceMeta.write_initial`
    is merge-aware), so the launcher's fields are preserved and only new keys
    (e.g. layer fingerprint) are overlaid.

    `runtime` is the backend selector the agent ran on (`"claude"`, `"codex3"`,
    ...); runs recorded earlier say `"claude_cli"` and the like, which readers
    accept too.
    """
    fields = {
        "prompt": prompt,
        "layer": layer,
        "runtime": runtime,
        "model": model,
        "run_id": run_id,
        # Which machine ran it, for reading runs spread over several.
        "host": socket.gethostname(),
    }
    if extra:
        fields.update(extra)
    TraceMeta(run_dir).write_initial(**fields)
    return run_dir / "meta.json"


#: How much of a failed process's log to copy into meta.json.
_LOG_TAIL_CHARS = 4000


def _log_tail(log_path: Path) -> str | None:
    """The last `_LOG_TAIL_CHARS` characters of `log_path`, or None.

    Seeks rather than reading the whole file, which can be far larger than the
    tail we keep. Reads 4x the character budget in bytes so a multi-byte tail
    cannot come up short, and decodes with `replace` because a seek can land
    mid-codepoint.
    """
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - _LOG_TAIL_CHARS * 4))
            raw = fh.read()
    except OSError:
        return None
    return raw.decode("utf-8", "replace")[-_LOG_TAIL_CHARS:]


def record_agent_stop(run_dir: Path, *, reason: str, detail: str) -> None:
    """Record in `meta.json` why the launcher stopped the agent CLI mid-run, or why it exited on an error it
    reported (a usage limit, no login): `meta["agent_stop"] = {"reason", "detail", "at"}`. The outcome is left to
    the daemon, which finalizes the run as usual. Raises nothing but `KeyboardInterrupt`, as
    `record_launch_failure` does."""
    try:
        meta_path = Path(run_dir) / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta["agent_stop"] = {"reason": reason, "detail": detail, "at": utcnow_iso()}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        raise
    except Exception:                                # noqa: BLE001
        pass


def record_launch_failure(run_dir: Path, *, reason: str, detail: str,
                          log_path: Path | None = None, message: str | None = None,
                          fix: list | None = None) -> None:
    """Stamp a launcher-side startup failure onto `meta.json`, then return.

    A run whose daemon dies before binding leaves a run dir that is
    **indistinguishable from a run still in progress**: `meta.json` says
    `outcome: null, finished_at: null`, there is no `trace.jsonl` and no
    `system_prompt.md`, and the only record of what happened is `.daemon.log`
    — a dotfile nothing reads.

    `outcome` is only written when it ranks above the current one, so this can
    never downgrade a run that committed, with exceptions or without. The log tail rides along because
    the traceback is the whole diagnosis and a dotfile is a bad place to keep
    it; the path is recorded too, since the tail is bounded.

    Raises nothing but `KeyboardInterrupt` — every caller is on its way to
    `sys.exit`, and failing to *describe* a failure must not replace it with a
    different one. Ctrl-C is the one exception, for the reason
    `compile.build_ir_with_layer` gives: absorbing operator intent makes it do
    nothing at all.
    """
    try:
        meta_path = run_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        failure: dict = {"reason": reason, "detail": detail}
        # What the operator was shown, so a reader of the run folder sees the same words: the first FATAL line, and
        # the fix lines that followed it.
        if message:
            failure["message"] = message
        if fix:
            failure["fix"] = list(fix)
        if log_path is not None:
            failure["log"] = log_path.name
            failure["log_tail"] = _log_tail(log_path)
            if failure["log_tail"] is None:
                del failure["log_tail"]
        meta["launch_failure"] = failure
        if (_OUTCOME_RANK.get(meta.get("outcome"), 0)
                < _OUTCOME_RANK["delivered_with_exceptions"]):
            meta["outcome"] = "daemon_failed"
        meta["finished_at"] = utcnow_iso()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        # Operator intent, never swallowed — the same rule
        # `compile.build_ir_with_layer` follows for the same reason.
        raise
    except BaseException:                    # noqa: BLE001 — best effort only
        pass


__all__ = [
    "AgentContext",
    "AgentSession",
    "TraceMeta",
    "record_launch_failure",
    "utcnow_iso",
    "write_run_meta",
]
