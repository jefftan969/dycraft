"""Asynchronous object delegation for the scene author.

**Why asynchronous, and not a blocking batch.** The scene author has a main
thread of work — the room shell, the cabinetry run, the counter interfaces —
and delegation exists so that a detailed object can be built *while that work
continues*. A call that blocks until its children land buys parallelism between
the children and pays for it with the author's own idle time, which is the
larger cost: an object run takes from half an hour to a couple of hours.

Nor is there a lifecycle wrapped around a dispatch — no slot manifest to keep
in step, no refine cadence, no freshness counters comparing submission numbers
rather than consumed content. `submit_object` returns an id, the author keeps
authoring, and `await_any` blocks only when the author decides it
has run out of other work.

**How a landing is noticed.** A CLI agent is turn-based and cannot be
interrupted, so "notify me" has to mean "tell me next time I ask something
else": `readiness` is folded into every op result by `Session`, so an author
learns a child landed from the compile it was going to run anyway. `await_any`
with `timeout=0` is the explicit cheap poll for an author that wants one.

**Integration is `load_object`.** A child writes an ordinary `object.py` under
`<run>/objects/<name>/`, and the parent builds it into the scene with
`load_object("objects/<name>/object.py")` — editing it first if it wants to.
Dispatch puts a program on disk; loading is a separate, ordinary build step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ... import _oom
from ..agent_runner import user_prompt_from_env

#: Children write here, so the path an author passes to `load_object` is
#: `objects/<name>/object.py` — inside the run dir, which is what
#: `confine_to_run_dir` requires of it.
DISPATCH_DIR = "objects"

#: How many children may run at once.
#:
#: The bound is the machine's, not a view about how much delegation is healthy:
#: each child is a full agent run holding its own CAD workers, and a warm one
#: costs about a gigabyte resident. The author decides which objects are worth
#: dispatching; this only decides how many can be in flight while it does.
DEFAULT_MAX_CONCURRENT = 8


def max_concurrent() -> int:
    """The in-flight cap for this run: `DYCRAFT_MAX_CONCURRENT_DISPATCHES`, else
    `DEFAULT_MAX_CONCURRENT`."""
    raw = os.environ.get("DYCRAFT_MAX_CONCURRENT_DISPATCHES")
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT

#: Terminal statuses. `committed` is the only one that produced geometry.
TERMINAL = ("committed", "failed", "spawn_error", "cancelled", "timeout")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DispatchRecord:
    """One delegated object run."""

    dispatch_id: str
    name: str
    brief: str
    sub_run_dir: Path
    status: str = "running"
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    proc: "subprocess.Popen | None" = None
    result: dict | None = None
    thread: "threading.Thread | None" = None

    def elapsed_s(self) -> float:
        end = (datetime.fromisoformat(self.finished_at) if self.finished_at
               else datetime.now(timezone.utc))
        return max(0.0, (end - datetime.fromisoformat(self.started_at))
                   .total_seconds())

    def summary(self) -> dict:
        out = {"dispatch_id": self.dispatch_id, "name": self.name,
               "status": self.status, "elapsed_s": round(self.elapsed_s(), 1)}
        if self.status == "committed":
            out["source_path"] = f"{DISPATCH_DIR}/{self.name}/object.py"
        return out


def _state(ctx) -> dict:
    """The run's dispatch table, created on first use."""
    table = getattr(ctx, "_dispatches", None)
    if table is None:
        table = {}
        ctx._dispatches = table
        ctx._dispatch_lock = threading.Lock()
        ctx._dispatch_done = threading.Condition(ctx._dispatch_lock)
    return table


def _lock(ctx):
    _state(ctx)
    return ctx._dispatch_lock


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


def compose_brief(*, name: str, brief: str, scene_request: str = "",
                  size=None, interfaces=None, materials=None,
                  states=None) -> str:
    """The child's whole task text.

    **The brief carries the context.** A child told only a slot's name, size,
    palette hint and affordances — not the global request, the room, and the
    neighboring geometry — designs against a bounding box, and the parent
    discovers the mismatch on integration. The scene author holds the whole
    design, so what it can put in a brief is exactly what makes delegation
    safe: the envelope, the surfaces this object has to meet, and what the
    scene as a whole is for.

    Every field is optional. A detail that does not constrain the object is
    noise in a brief, not diligence.
    """
    parts = [brief.strip()]
    if scene_request:
        parts += ["", "This object is one part of a larger scene, built for "
                       "this request:", f"  {scene_request.strip()}",
                  "Build the object, not the scene."]
    if size:
        parts += ["", f"Target envelope (meters, w x d x h): "
                      f"{tuple(round(float(v), 4) for v in size)}",
                  "Stay within it; say so in a deviation if the object cannot."]
    if interfaces:
        parts += ["", "Surfaces and dimensions it must meet:"]
        parts += [f"  - {line}" for line in _as_lines(interfaces)]
    if materials:
        parts += ["", "Material and finish intent:"]
        parts += [f"  - {line}" for line in _as_lines(materials)]
    if states:
        parts += ["", "States it must support:"]
        parts += [f"  - {line}" for line in _as_lines(states)]
    return "\n".join(parts)


def _as_lines(value) -> list:
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def _sub_backend(ctx) -> str:
    """Which CLI backend serves a child. Same resolution `review` uses."""
    from .review import review_backend
    return review_backend(ctx)


def _child_argv(ctx, *, name: str, prompt_path: Path, sub_runs: Path,
                advisory_hours: float | None = None) -> list:
    argv = [
        sys.executable, "-m", "dycraft.cli",
        "--backend", _sub_backend(ctx),
        "--layer", "object",
        "--runs-dir", str(sub_runs),
        "--run-id", name,
        "--prompt", str(prompt_path),
    ]
    parent_run = getattr(ctx, "run_id", None) or ctx.run_dir.name
    argv += ["--parent-run", str(parent_run)]
    # Without it a child gets the CLI's default of one hour, whatever its
    # launcher has left.
    if advisory_hours is not None:
        argv += ["--advisory-hours", repr(float(advisory_hours))]
    # The parent's own model: without it a child falls back to the CLI
    # backend's default model. Together with `review_backend` reading the
    # run's `runtime`, this keeps a dispatched object in the same family as the
    # scene that asked for it.
    from .review import run_model, run_reasoning_effort

    model = run_model(ctx)
    if model:
        argv += ["--model", model]
    # Effort is inherited too. A child left on the CLI default ran at `high`
    # under an `xhigh` parent, which is a difference in the work rather than in
    # the wiring, and it never appeared in the arm's own config.
    effort = run_reasoning_effort(ctx)
    if effort:
        argv += ["--reasoning-effort", effort]
    render_backend = os.environ.get("DYCRAFT_RENDER_BACKEND")
    if render_backend:
        argv += ["--render-backend", render_backend]
    return argv


def submit_object(ctx, *, name: str, brief: str, size=None, interfaces=None,
                  materials=None, states=None,
                  _spawn=None) -> dict:
    """Start an object agent on `brief` and return immediately.

    `_spawn` is a seam for tests: launching a real CLI child costs minutes,
    and what the tests here need to establish is the bookkeeping around it.
    """
    table = _state(ctx)
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        return {"ok": False, "error": "name must be a simple directory name"}
    if not isinstance(brief, str) or not brief.strip():
        return {"ok": False, "error": "brief must describe the object to build"}
    with _lock(ctx):
        live = [r for r in table.values() if r.status == "running"]
        cap = max_concurrent()
        if len(live) >= cap:
            return {"ok": False, "outcome": "at_concurrency_limit",
                    "error": (f"{len(live)} objects already building "
                              f"(limit {cap}). await_any "
                              f"one of them first."),
                    "dispatches": [r.summary() for r in live]}
        if any(r.name == name and r.status == "running" for r in table.values()):
            return {"ok": False, "outcome": "duplicate_name",
                    "error": f"an object named {name!r} is already building"}
        dispatch_id = f"d_{len(table) + 1:03d}"

    sub_runs = Path(ctx.run_dir) / DISPATCH_DIR
    sub_run_dir = sub_runs / name
    # `objects/<name>/` means two different things, and only one of them is
    # safe to dispatch into. A child run leaves a full bundle there — its own
    # `meta.json`, trace, and `object.py` — and redispatching over that is a
    # workflow the prompt explicitly offers ("A failed child may leave useful
    # source ... recover it, author locally, or redispatch"). But the parent
    # can also write `objects/<name>/object.py` itself and `load_object` it,
    # and that directory has no `meta.json`; authors do this often.
    #
    # Dispatching into the second kind would hand the child a directory whose
    # `object.py` the parent's own `scene.py` is loading, and the child would
    # overwrite it — silently, mid-compile. Refuse instead, since the fix (a
    # different name) costs the agent nothing and the prompt already tells it
    # to pick one.
    if (sub_run_dir / "object.py").is_file() and not (
            sub_run_dir / "meta.json").is_file():
        return {"ok": False, "outcome": "would_overwrite_local_source",
                "error": (f"objects/{name}/object.py exists without child-run "
                          f"metadata. Dispatch under a different name to "
                          f"preserve the existing source."),
                "path": f"{DISPATCH_DIR}/{name}/object.py"}
    sub_run_dir.mkdir(parents=True, exist_ok=True)
    text = compose_brief(
        name=name, brief=brief, size=size, interfaces=interfaces,
        materials=materials, states=states,
        scene_request=user_prompt_from_env())
    prompt_path = sub_run_dir / "brief.md"
    prompt_path.write_text(text, encoding="utf-8")

    record = DispatchRecord(dispatch_id=dispatch_id, name=name, brief=text,
                            sub_run_dir=sub_run_dir)
    argv = _child_argv(ctx, name=name, prompt_path=prompt_path,
                       sub_runs=sub_runs)
    spawn = _spawn or _default_spawn
    try:
        record.proc = spawn(argv, sub_run_dir)
    except BaseException as e:  # noqa: BLE001 — a spawn failure is data
        record.status = "spawn_error"
        record.finished_at = _now()
        record.result = {"ok": False, "outcome": "spawn_error",
                         "error": f"{type(e).__name__}: {e}"}
        with _lock(ctx):
            table[dispatch_id] = record
        return {"ok": False, **record.result, **record.summary()}

    with _lock(ctx):
        table[dispatch_id] = record
    record.thread = threading.Thread(
        target=_wait_for, args=(ctx, record), daemon=True,
        name=f"dispatch-{dispatch_id}")
    record.thread.start()
    return {"ok": True, "dispatch_id": dispatch_id, "name": name,
            "status": "running", "brief_path": str(prompt_path),
            "note": ("building in the background. Keep authoring; call "
                     "await_any() when you need it, or watch the `dispatches` "
                     "line on your next op result.")}


def _default_spawn(argv: list, sub_run_dir: Path):
    log = (sub_run_dir / "child.log").open("wb")
    # SUBAGENT, so the kernel takes a delegated object agent before the arm
    # that dispatched it; the child's own `dycraft.cli` entry keeps it there.
    from ...cli.backends._process import PARENT_PID_ENV
    # The child's launcher watches this pid, and stops when it is gone (`_process.watch_parent`).
    return subprocess.Popen(_oom.tiered(argv, _oom.SUBAGENT), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env={**os.environ, PARENT_PID_ENV: str(os.getpid())})


def _wait_for(ctx, record: DispatchRecord) -> None:
    """Background: wait for the child, then read what it left behind."""
    try:
        returncode = record.proc.wait()
    except BaseException as e:  # noqa: BLE001 — must resolve the record
        returncode = -1
        record.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result = _harvest(record, returncode)
    with _lock(ctx):
        record.status = result["status"]
        record.finished_at = _now()
        record.result = result
        ctx._dispatch_done.notify_all()


def _deviations_from_trace(run_dir: Path) -> list:
    """`deliberate_deviations` rebuilt from a child's `trace.jsonl`.

    `record_deviation` traces an `op_call` whose args carry the record, so a
    child that was killed before its final `meta.json` flush still reports
    what it deviated on.
    """
    try:
        text = (run_dir / "trace.jsonl").read_text()
    except OSError:
        return []
    out: list = []
    seen: set = set()
    for line in text.splitlines():
        if '"record_deviation"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("ev") != "op_call" or ev.get("op") != "record_deviation":
            continue
        args = ev.get("args") or {}
        what, why = args.get("what"), args.get("why")
        if not (isinstance(what, str) and what and isinstance(why, str) and why):
            continue
        if (what, why) in seen:
            continue
        seen.add((what, why))
        out.append({"what": what, "why": why,
                    "severity": args.get("severity"),
                    "suggested_fix": args.get("suggested_fix") or None,
                    "blocked_layer": args.get("blocked_layer"),
                    "targets": args.get("targets")})
    return out


def _read_deviations(run_dir: Path, meta: dict) -> list:
    """A child's deviations: `meta.json`'s list, else what its trace recorded."""
    devs = meta.get("deliberate_deviations") if isinstance(meta, dict) else None
    return list(devs) if devs else _deviations_from_trace(run_dir)


def _harvest(record: DispatchRecord, returncode: int) -> dict:
    """What the child produced, read off its run directory.

    Deviations come back deliberately: the parent owns the scene, so a child
    saying "I could not make this meet the counter" is telling the one agent
    that can act on it.
    """

    run_dir = record.sub_run_dir
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
    outcome = str(meta.get("outcome") or "")
    source = run_dir / "object.py"
    ir_path = run_dir / "ir.json"
    committed = (outcome == "committed" and ir_path.is_file()
                 and ir_path.stat().st_size > 0)

    if record.status == "cancelled":
        status = "cancelled"
    elif committed:
        status = "committed"
    elif returncode != 0 and not outcome:
        status = "spawn_error"
    else:
        status = "failed"

    try:
        deviations = _read_deviations(run_dir, meta)
    except BaseException:  # noqa: BLE001 — never fail the harvest on these
        deviations = []

    payload = {
        "ok": status == "committed",
        "status": status,
        "outcome": outcome or None,
        "dispatch_id": record.dispatch_id,
        "name": record.name,
        "returncode": returncode,
        # Run-relative, because that is what `load_object` takes.
        "source_path": (f"{DISPATCH_DIR}/{record.name}/object.py"
                        if source.is_file() else None),
        "ir_path": (f"{DISPATCH_DIR}/{record.name}/ir.json"
                    if ir_path.is_file() else None),
        "sub_run": str(run_dir),
        "deviations": deviations,
    }
    if status == "committed":
        payload["next"] = (
            f'load_object("{payload["source_path"]}") to build it into the '
            f"scene. Edit that file first if the result needs changing — it "
            f"is yours now.")
    else:
        payload["next"] = (
            "the child produced no committed object. Read its log and "
            f"{record.name}/context_summary.md, then author the object "
            "yourself or dispatch a sharper brief.")
    return payload


# ---------------------------------------------------------------------------
# Await, poll, cancel
# ---------------------------------------------------------------------------


def await_any(ctx, dispatch_ids=None, timeout=None) -> dict:
    """Block until one of `dispatch_ids` is terminal, or `timeout` elapses.

    `timeout=0` polls: it returns what has already landed without waiting,
    which is the cheap check an author makes between edits.
    """
    table = _state(ctx)
    with _lock(ctx):
        if dispatch_ids is None:
            watched = [r for r in table.values()]
        else:
            ids = ([dispatch_ids] if isinstance(dispatch_ids, str)
                   else list(dispatch_ids))
            unknown = [i for i in ids if i not in table]
            if unknown:
                return {"ok": False, "outcome": "unknown_dispatch_id",
                        "error": f"unknown dispatch ids: {unknown}",
                        "known": sorted(table)}
            watched = [table[i] for i in ids]
        if not watched:
            return {"ok": False, "outcome": "nothing_dispatched",
                    "error": "no object dispatches to wait for"}

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            done = [r for r in watched if r.status in TERMINAL
                    and not getattr(r, "_reported", False)]
            if done:
                landed = sorted(done, key=lambda r: r.dispatch_id)[0]
                landed._reported = True
                return {**(landed.result or {}),
                        "pending": [r.summary() for r in watched
                                    if r.status == "running"]}
            if all(r.status in TERMINAL for r in watched):
                return {"ok": False, "outcome": "all_reported",
                        "error": ("every dispatch you are waiting on has "
                                  "already been reported"),
                        "dispatches": [r.summary() for r in watched]}
            remaining = (None if deadline is None
                         else deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                # A poll that finds nothing has answered its question. It used
                # to answer `ok: False, timeout`, which `meta.json` counted as
                # a failed operation: 37 of the 59 recorded `await_any` calls,
                # and 58 of the 59 were polls. A wait the author asked for and
                # that did not deliver is still a failure -- that one is real
                # information about a stuck child.
                polling = not float(timeout or 0)
                return {"ok": polling,
                        "outcome": "nothing_ready" if polling else "timeout",
                        "pending": [r.summary() for r in watched
                                    if r.status == "running"]}
            ctx._dispatch_done.wait(timeout=remaining)


def dispatch_status(ctx) -> dict:
    table = _state(ctx)
    with _lock(ctx):
        rows = [r.summary() for r in sorted(table.values(),
                                            key=lambda r: r.dispatch_id)]
    return {"ok": True, "dispatches": rows,
            "running": sum(1 for r in rows if r["status"] == "running")}


def cancel_dispatch(ctx, dispatch_id: str) -> dict:
    table = _state(ctx)
    with _lock(ctx):
        record = table.get(dispatch_id)
        if record is None:
            return {"ok": False, "error": f"unknown dispatch {dispatch_id!r}",
                    "known": sorted(table)}
        if record.status in TERMINAL:
            return {"ok": True, "status": record.status,
                    "note": "already finished"}
        record.status = "cancelled"
    try:
        record.proc.terminate()
    except BaseException:  # noqa: BLE001 — a dead child is already canceled
        pass
    return {"ok": True, "status": "cancelled", "dispatch_id": dispatch_id}


def readiness(ctx) -> "list | None":
    """One line per live or unreported dispatch, for an op result to carry.

    This is the notification channel. A CLI agent cannot be interrupted, so a
    child landing has to reach it through something it was going to do anyway —
    the next compile, probe or snapshot carries the news.
    """
    table = getattr(ctx, "_dispatches", None)
    if not table:
        return None
    lock = getattr(ctx, "_dispatch_lock", None)
    if lock is None:
        return None
    with lock:
        rows = [r.summary() for r in sorted(table.values(),
                                            key=lambda r: r.dispatch_id)
                if r.status == "running" or not getattr(r, "_reported", False)]
    return rows or None
