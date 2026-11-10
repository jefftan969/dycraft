"""What crosses between an animation op and its caller: replies, intervals, job polling, and the workspace swap."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time


def worker_printed(output):
    """What the agent's code printed during the call, as the reply's `printed`."""
    path = Path(output) / "printed.txt"
    return {"printed": path.read_text(errors="replace")} if path.is_file() else {}


def swap_workspace(work, candidate, native):
    """Make `candidate` the project at `work`; remove the old tree if the filesystem lets us.

    The swap is two renames and is what publishes. Removing the old tree is
    housekeeping, and must not decide the outcome: on NFS a file another
    process still holds open (the resident Blender's `.pyc`, an Alembic cache)
    is renamed to `.nfsXXXX` in place, and `rmtree` then fails with ENOTEMPTY
    *after* the new project is live, so a failure there must not report the
    publish as failed, and a tree left under one fixed name would block the
    next rename. So each retired tree gets a name of its own, is removed
    best-effort, and stragglers are swept on the next swap, by which time their
    holders have usually let go.
    """
    import uuid
    for stale in native.glob("previous*"):
        shutil.rmtree(stale, ignore_errors=True)
    retired = native / f"previous.{uuid.uuid4().hex[:12]}"
    work.rename(retired)
    try:
        candidate.rename(work)
    except BaseException:
        retired.rename(work)
        raise
    shutil.rmtree(retired, ignore_errors=True)


def worker_warnings(output):
    """Advisory notes a worker left beside its result, or nothing.

    A sidecar rather than a field on the worker's return value, because
    `run_worker` hands back the author's own payload -- a probe's number, a
    render's paths -- and the harness must not reshape that to carry a note.
    `unsaved_actions` names datablocks Blender will not write to the saved
    blend, which the author can retain with `use_fake_user`. `cuts` reports
    the camera markers this operation materialized from `dycraft_cuts`, with
    the absolute frame each one resolved to.
    """
    path = Path(output) / "warnings.json"
    if not path.is_file():
        return {}
    try:
        warnings = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {"warnings": warnings} if warnings else {}

#: How an interval can arrive. `start`/`end` are inclusive zero-based output
#: frame indices — the surface's one unit. `seconds` is the `[a, b]` alternate.
#: `clip` names an NLA strip and is resolved in the worker, which is the only
#: place that can read the blend without paying a second Blender launch for
#: the privilege; it is listed here so a clip-scoped call is *partial* by the
#: same predicate as any other, and so a clip-scoped `compile` is a diagnostic
#: rather than the run's verdict. `frame_indices` rejects mixing the spellings.
INTERVAL_KEYS = ("start", "end", "seconds", "clip")


def interval_of(args):
    return {k: args.get(k) for k in INTERVAL_KEYS}


def is_partial(args):
    """True when a check was asked for a slice of the timeline rather than all of it."""
    return any(args.get(k) is not None for k in INTERVAL_KEYS)


def counts_as_compile(operation, args):
    """Whether this check is the run's record of the project's verified state.

    Only a whole-timeline `compile`/`finish` is. A partial `compile` is the
    same operation as `check_motion` — a diagnostic on an interval — and must
    not touch `n_compile_attempts`, `last_compile_payload`,
    `compile_reports/`, the `compile_attempt` trace or the validation cache,
    because `project/resume.py` reconstructs the attempt count from that trace
    and the orchestrator reads `compile_reports/latest.json` as the verdict on
    the whole animation.
    """
    return operation in ("compile", "finish") and not is_partial(args)


def slim_job_payload(payload):
    """Drop what a reading agent does not need, at every depth that carries it.

    `_slim_compile_payload_for_llm` folds passing verifiers into a flat name
    list, but it only looks at the top level. A `storyboard` reply carries its
    staged-frame check under `checks`, so that level is walked too.
    """
    from ..agent.layers.session import _slim_compile_payload_for_llm
    if not isinstance(payload, dict):
        return payload
    if "verifier_reports" in payload:
        payload = _slim_compile_payload_for_llm(payload)
    # An animation `storyboard` reply carries its one shared staged-frame check
    # here, at the top level, rather than once per state.
    if isinstance(payload.get("checks"), list):
        payload = {**payload, "checks": [slim_job_payload(c) for c in payload["checks"]]}
    return payload


def register_job_handlers(session, jobs):
    """Polling is a status question; only `job_result` hands back the payload.

    `contribution_status` deliberately does NOT go through this: contributions
    have no separate result op, so stripping `result` there would remove the
    only way to collect one.
    """
    def traced(name, fn):
        """Emit one trace record per job-control call, including a failed one.

        These three go through `register_handler` rather than `session.op`
        because `job_status` strips `result` and does its own shaping, and
        `register_handler` does not wrap or trace. Without this the run
        directory would record the jobs but not the waiting, which is about
        half of a run's harness interaction and the half that says whether the
        run was latency-bound.

        Traced flat, one record per call: the inter-poll timing is the signal,
        so consecutive polls are not collapsed.

        `job_result` names its event `job_result_op` because the runtime
        already emits `job_result` when a job finishes -- which is why this
        gap was easy to miss: grepping the trace for `job_result` finds
        plenty and none of them is a poll.
        """
        event = "job_result_op" if name == "job_result" else name

        def handler(job_id):
            started = time.monotonic()
            try:
                record = fn(job_id)
            except Exception as error:
                session.trace(event, job_id=job_id, ok=False, error=str(error),
                              elapsed_s=round(time.monotonic() - started, 3))
                raise
            session.trace(event, job_id=job_id, ok=True,
                          status=record.get("status"),
                          progress=record.get("progress"),
                          elapsed_s=round(time.monotonic() - started, 3))
            return record

        return handler

    def status(job_id):
        record = jobs.status(job_id)
        record["result_available"] = "result" in record
        record.pop("result", None)
        return record

    def result(job_id):
        record = jobs.status(job_id)
        if "result" in record:
            record["result"] = slim_job_payload(record["result"])
        return record

    session.register_handler("job_status", traced("job_status", status))
    session.register_handler("job_result", traced("job_result", result))
    session.register_handler("job_cancel", traced("job_cancel", jobs.cancel))
