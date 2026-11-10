"""Run one complete `object.py` in the CAD interpreter, from the main env.

Parent half of the object worker: JSON job in, JSON result out, bounded by a
timeout, never raising for a modeling failure.

This ships `object.py` itself and hands back its ObjectIR — one authored
program, one execution, nothing retrieved by name. The subprocess is dependency
isolation: build123d lives in its own environment.

Builds run on *warm* interpreters from `_worker_pool`, which is what makes the
separate environment affordable — `import build123d` costs ~1.4 s, paid once
per child rather than per build. A child is reused only while it is above suspicion: any
timeout, death, protocol desync, or `worker_internal` retires it, so a faulted
interpreter is never handed to the next build. `DYCRAFT_OBJECT_WORKER_PERSISTENT=0`
forces a fresh interpreter per build.

The returned dict is exactly `compile.build_ir_with_layer`'s shape, so a caller
can route to either without caring which ran.

Interpreter discovery is `_cad_env.b123d_python()` — deliberately the same
single chain (`DYCRAFT_B123D_PYTHON` > `.b123d_python` > conventional prefixes)
every other CAD consumer uses, so they can never disagree about which
environment has the kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ... import _oom
from . import _timelimit, _worker_pool
from ._cad_env import b123d_child_env, b123d_python

_CHILD_PATH = Path(__file__).resolve().parent / "_object_worker_child.py"

# How long a brand-new CAD interpreter gets to finish importing build123d.
# Separate from the build timeout on purpose — see `_Child.ensure_ready`.
_STARTUP_TIMEOUT_S = 120.0

# One CAD start (~3.5 s warm) plus whatever the source models. Generous because
# the failure mode of a too-short timeout is a lost compile that looked like a
# modeling bug.
_DEFAULT_TIMEOUT_S = _timelimit.DEFAULT_LIMIT_S
#: The scene layer's build is `scene.py` and every object it loads, under one
#: limit; the child bounds the source by the same one (`_timelimit`).
_SCENE_TIMEOUT_S = _timelimit.SCENE_LIMIT_S
_TIMEOUT_ENV_VAR = _timelimit.LIMIT_ENV_VAR

_UNAVAILABLE_MESSAGE = (
    "no build123d interpreter found, so the source cannot be built; run "
    "`python -m dycraft.install cad` to record one in .b123d_python, or set "
    "DYCRAFT_B123D_PYTHON"
)


def is_available() -> bool:
    """True when an interpreter that can run the worker was found."""
    return b123d_python() is not None


def shutdown_pool() -> None:
    """Stop every warm CAD interpreter. Idempotent.

    Registered at exit by `_worker_pool`, so this is only needed where teardown
    must be deterministic — a test asserting on spawn counts, or a daemon that
    wants its children gone before it finalises.
    """
    _worker_pool.shutdown()


def _resolve_timeout(explicit: float | None = None, layer: str = "object") -> float:
    default = _SCENE_TIMEOUT_S if layer == "scene" else _DEFAULT_TIMEOUT_S
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _timed_out(source_path: Path, limit: float) -> str:
    """The timeout message: the file that ran, and what the limit covers.

    A whole-scene build names `scene.py` and its scope, so the author of a
    scene that is only large is not sent to their CAD alone.
    """
    if source_path.name == "scene.py":
        return (f"scene.py did not finish within {limit:g}s, the limit for one build of the "
                "whole scene with every object it loads; inspect loops, expensive CAD "
                "operations and tessellation, starting with the slowest objects")
    return (f"{source_path.name} did not finish within {limit:g}s; "
            "inspect loops, expensive CAD operations, and tessellation")


#: Every worker-level reason a build can fail for, declared in one place.
#:
#: `tools._WORKER_REPAIRS` answers each of these with what the *agent* should do,
#: and the two must not drift: a kind with no entry reaches the agent with an
#: empty `suggested_repairs`, which reads as "nothing to be done" rather than as
#: a gap. `test_object_worker_routing.py` pins the two sets equal.
#:
#: `build123d_version_mismatch` and `ocp_missing` are produced by the child's
#: startup checks (`check_build123d_version`, `check_ocp_available`) and carried
#: out on `SolidCompileError.kind`, not raised here, so they are listed rather
#: than derived.
FAILURE_KINDS = (
    "worker_unavailable",       # no CAD interpreter resolved
    "timeout",                  # the child exceeded its wall clock
    "worker_crash",             # the child died without writing a result
    "worker_internal",          # the harness itself faulted inside the child
    "build123d_version_mismatch",   # the interpreter is not the pinned build123d
    "ocp_missing",              # the interpreter has build123d but no OCP
)


def worker_failure(kind: str, message: str, **extra) -> dict:
    """A build_ir_with_layer-shaped failure carrying a worker-level reason.

    Kept in the `build_error` slot rather than `sandbox_violation`: the source
    did not violate anything, the environment could not run it, and the two lead
    to completely different repairs.

    Shared with `_object_worker_child`, which reports its own faults in the same
    shape — one definition, so a consumer reading `error_kind` and
    `build_error.evidence.reason` gets the same answer whichever side produced it.
    """
    if kind not in FAILURE_KINDS:
        raise ValueError(
            f"unknown worker failure kind {kind!r}; add it to FAILURE_KINDS so "
            f"tools._WORKER_REPAIRS is required to answer it")
    out = {
        "ok": False, "ir": None, "sandbox_violation": None,
        "build_error": {
            "error": "ObjectWorkerError",
            "summary": message,
            "evidence": {"reason": kind, **extra},
            "refs": [],
            "suggested_repairs": [],
        },
        "error_kind": kind,
        "message": message,
    }
    out.update(extra)
    return out



def run_object_source(source: str, *, run_dir,
                      timeout: float | None = None,
                      python: str | None = None,
                      layer: str = "object") -> dict:
    """Execute `source` as `object.py` in the CAD interpreter.

    Returns `{ok, ir, build_error, sandbox_violation}` — the same contract as
    an in-process `build_ir_with_layer`, so the caller does not branch.

    The source is written to `<run_dir>/object.py` before execution. That is
    both what the agent's own `write_file` produces and what
    `Object._live_source_text` reads back to stage as each live part's `source`
    provenance asset, so the file on disk and the artifact agree by
    construction.

    Runs on a warm pooled interpreter when one can be had, and spawns a
    single-use one otherwise. The two produce identical payloads by
    construction — same child, same `_build_one`, same result file — so a
    caller cannot tell which ran, and neither can a test.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    interp = python or b123d_python()
    if not interp:
        return worker_failure("worker_unavailable", _UNAVAILABLE_MESSAGE)

    # The scene layer authors geometry from `scene.py`; everything else about
    # the build is identical, so the filename follows the layer rather than the
    # layer forking the path.
    source_path = run_dir / ("scene.py" if layer == "scene" else "object.py")
    source_path.write_text(source, encoding="utf-8")

    if _worker_pool.POOL.enabled():
        out = _run_pooled(interp, source_path, run_dir, timeout, layer)
        if out is not None:
            return out
    return _run_one_shot(interp, source_path, run_dir, timeout, layer)


def run_cad_snippet(body: str, *, timeout: float | None = None,
                    python: str | None = None,
                    isolate: bool = False) -> dict:
    """Run `body` inside the CAD interpreter, on a warm child when possible.

    Returns `{ok, stdout, stderr, error_kind, message, traceback}`. The body's
    **stdout is its result**: print one line of JSON and read it back from
    `stdout`. `ok` is False if the body raised, and `message`/`traceback` then
    describe what it raised — the structured form of the non-zero exit a direct
    `<b123d-python> -c ...` gave.

    `dycraft` is already imported in the child, so a body must **not** re-run
    the `spec_from_file_location` bootstrap that a direct invocation needs: that
    would bind a *second* copy of the package while existing submodules still
    referenced the first. Just `from dycraft... import ...`.

    Two things this is deliberately not. It is not a sandbox — the body gets the
    same access the `-c` form had, and the sandbox that matters for authored
    source is `_sandbox.build_globals()`, applied on the build path. And it is
    not process isolation: bodies share one interpreter, each with a fresh
    globals dict, so a body that mutates process-global state can be seen by a
    later one. Pass `isolate=True` for those, which spends ~1.5 s on a
    single-use interpreter to get the old guarantee back.

    Unlike a build, the captured stdout is **not** truncated: for a build the
    stdio is a diagnostic note, here it is the value the caller parses.
    """
    interp = python or b123d_python()
    if not interp:
        return {"ok": False, "stdout": "", "stderr": "",
                "error_kind": "worker_unavailable",
                "message": _UNAVAILABLE_MESSAGE, "traceback": None}

    job = {"kind": "snippet", "body": body}
    if not isolate and _worker_pool.POOL.enabled():
        out = _snippet_pooled(interp, job, timeout)
        if out is not None:
            return out
    return _snippet_one_shot(interp, job, timeout)


def _snippet_result(out: dict, stdout: str, stderr: str) -> dict:
    return {"ok": bool(out.get("ok")), "stdout": stdout, "stderr": stderr,
            "error_kind": out.get("error_kind"),
            "message": out.get("message"),
            "traceback": out.get("traceback")}


def _snippet_failure(kind: str, message: str, *, stdout: str = "",
                     stderr: str = "") -> dict:
    return {"ok": False, "stdout": stdout, "stderr": stderr,
            "error_kind": kind, "message": message, "traceback": None}


def _snippet_pooled(interp: str, job: dict,
                    timeout: float | None) -> dict | None:
    """One snippet on a warm child, or None to say "no warm child available"."""
    try:
        child = _worker_pool.POOL.acquire(interp, _CHILD_PATH)
    except OSError as e:
        return _snippet_failure("worker_unavailable",
                                f"could not launch the object worker: {e}")
    if child is None:
        return None

    limit = _resolve_timeout(timeout)
    try:
        child.ensure_ready(_STARTUP_TIMEOUT_S)
    except _worker_pool.WorkerTimeout:
        _worker_pool.POOL.recycle(child)
        return None
    except _worker_pool.WorkerProtocolError as e:
        _worker_pool.POOL.recycle(child)
        return _snippet_failure(
            "worker_crash", f"the object worker failed to start: {e}")

    with tempfile.TemporaryDirectory(prefix="dycraft_snippet_") as td:
        result_path = Path(td) / "result.json"
        try:
            ack = child.run_job({**job, "result": str(result_path)},
                                time.monotonic() + limit)
        except _worker_pool.WorkerTimeout:
            stdout, stderr = child.spool_tails()
            _worker_pool.POOL.recycle(child)
            return _snippet_failure(
                "timeout", f"the snippet did not finish within {limit:g}s",
                stdout=stdout, stderr=stderr)
        except _worker_pool.WorkerProtocolError as e:
            stdout, stderr = child.spool_tails()
            _worker_pool.POOL.recycle(child)
            return _snippet_failure("worker_crash", str(e),
                                    stdout=stdout, stderr=stderr)

        # cap=None: this span carries the answer, not a note about it.
        stdout, stderr = child.slice_spools(ack, cap=None)
        if not result_path.exists():
            _worker_pool.POOL.recycle(child)
            return _snippet_failure(
                "worker_crash",
                "the object worker acked without writing a result",
                stdout=stdout, stderr=stderr)
        try:
            out = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _worker_pool.POOL.recycle(child)
            return _snippet_failure("worker_crash",
                                    f"unreadable worker result: {e}",
                                    stdout=stdout, stderr=stderr)

    # A body that raised is not a worker fault — the child is fine and stays
    # warm. Only the harness faulting inside the child retires it.
    if ack.get("status") != "ok" or out.get("error_kind") == "worker_internal":
        _worker_pool.POOL.recycle(child)
    else:
        _worker_pool.POOL.release(child)
    return _snippet_result(out, stdout, stderr)


def _snippet_one_shot(interp: str, job: dict, timeout: float | None) -> dict:
    """One snippet on a single-use interpreter."""
    with tempfile.TemporaryDirectory(prefix="dycraft_snippet_") as td:
        job_path = Path(td) / "job.json"
        result_path = Path(td) / "result.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        try:
            proc = subprocess.run(
                _oom.tiered([interp, str(_CHILD_PATH), "--job", str(job_path),
                             "--result", str(result_path)], _oom.WORKER),
                capture_output=True, text=True,
                timeout=_resolve_timeout(timeout), env=b123d_child_env())
        except subprocess.TimeoutExpired:
            return _snippet_failure(
                "timeout", f"the snippet did not finish within "
                           f"{_resolve_timeout(timeout):g}s")
        except OSError as e:
            return _snippet_failure(
                "worker_unavailable",
                f"could not launch the object worker: {e}")

        stdout, stderr = proc.stdout or "", proc.stderr or ""
        if not result_path.exists():
            return _snippet_failure(
                "worker_crash",
                f"the object worker exited {proc.returncode} without writing "
                f"a result; the CAD kernel may have crashed",
                stdout=stdout, stderr=stderr)
        try:
            out = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return _snippet_failure("worker_crash",
                                    f"unreadable worker result: {e}",
                                    stdout=stdout, stderr=stderr)
    return _snippet_result(out, stdout, stderr)


def _job(source_path: Path, run_dir: Path, layer: str = "object") -> dict:
    return {"source_path": str(source_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "layer": layer}


def _validate(out) -> dict | None:
    if not isinstance(out, dict) or not isinstance(out.get("ok"), bool):
        return worker_failure("worker_crash",
                              "worker result did not carry a boolean 'ok'")
    return None


def _run_pooled(interp: str, source_path: Path, run_dir: Path,
                timeout: float | None, layer: str = "object") -> dict | None:
    """One build on a warm child, or None to say "no warm child available".

    None rather than an error: a full pool is a capacity condition, not a
    failure, and the caller answers it by spawning a single-use interpreter —
    the behavior this module had for its whole life before pooling.
    """
    try:
        child = _worker_pool.POOL.acquire(interp, _CHILD_PATH)
    except OSError as e:
        return worker_failure("worker_unavailable",
                              f"could not launch the object worker: {e}")
    if child is None:
        return None

    limit = _resolve_timeout(timeout, layer)

    def _crashed(message: str) -> dict:
        stdout, stderr = child.spool_tails()
        rc = child.proc.poll()
        _worker_pool.POOL.recycle(child)
        out = worker_failure("worker_crash", message, returncode=rc,
                             stderr=stderr[-4000:])
        out["worker_stdout"] = stdout[-8000:]
        out["worker_stderr"] = stderr[-8000:]
        return out

    try:
        child.ensure_ready(_STARTUP_TIMEOUT_S)
    except _worker_pool.WorkerTimeout:
        _worker_pool.POOL.recycle(child)
        return None
    except _worker_pool.WorkerProtocolError as e:
        # A child that cannot even start is an environment problem, and every
        # later build would hit it too. Report it rather than silently falling
        # back to a one-shot spawn that is about to fail the same way.
        return _crashed(f"the object worker failed to start: {e}")

    with tempfile.TemporaryDirectory(prefix="dycraft_objworker_") as td:
        result_path = Path(td) / "result.json"
        job = {**_job(source_path, run_dir, layer),
               "result": str(result_path)}
        try:
            ack = child.run_job(job, time.monotonic() + limit)
        except _worker_pool.WorkerTimeout:
            stdout, stderr = child.spool_tails()
            _worker_pool.POOL.recycle(child)
            out = worker_failure("timeout", _timed_out(source_path, limit))
            out["worker_stdout"] = stdout[-8000:]
            out["worker_stderr"] = stderr[-8000:]
            return out
        except _worker_pool.WorkerProtocolError as e:
            return _crashed(str(e))

        stdout, stderr = child.slice_spools(ack)
        if not result_path.exists():
            return _crashed(
                "the object worker acked without writing a result; the CAD "
                "kernel may have crashed")
        try:
            out = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return _crashed(f"unreadable worker result: {e}")

    invalid = _validate(out)
    if invalid is not None:
        _worker_pool.POOL.recycle(child)
        return invalid

    # A child that faulted in its own harness cannot be vouched for, so it is
    # dropped rather than handed to the next build. A *modeling* failure is
    # not a fault: the child is fine and stays warm.
    if ack.get("status") != "ok" or out.get("error_kind") == "worker_internal":
        _worker_pool.POOL.recycle(child)
    else:
        _worker_pool.POOL.release(child)

    out["worker_stdout"] = stdout[-8000:]
    out["worker_stderr"] = stderr[-8000:]
    return out


def _run_one_shot(interp: str, source_path: Path, run_dir: Path,
                  timeout: float | None, layer: str = "object") -> dict:
    """One build on a single-use interpreter — the pre-pool path, unchanged."""
    with tempfile.TemporaryDirectory(prefix="dycraft_objworker_") as td:
        job_path = Path(td) / "job.json"
        result_path = Path(td) / "result.json"
        job_path.write_text(json.dumps(_job(source_path, run_dir, layer)),
                            encoding="utf-8")

        cmd = _oom.tiered([interp, str(_CHILD_PATH), "--job", str(job_path),
                           "--result", str(result_path)], _oom.WORKER)
        env = b123d_child_env()
        limit = _resolve_timeout(timeout, layer)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=limit, env=env)
        except subprocess.TimeoutExpired:
            return worker_failure("timeout", _timed_out(source_path, limit))
        except OSError as e:
            return worker_failure("worker_unavailable",
                            f"could not launch the object worker: {e}")

        if not result_path.exists():
            # The child writes a result even for its own internal errors, so
            # nothing here means it died before it could — a segfault in the
            # CAD kernel is the realistic case.
            return worker_failure(
                "worker_crash",
                f"the object worker exited {proc.returncode} without writing a "
                f"result; the CAD kernel may have crashed",
                returncode=proc.returncode,
                stderr=(proc.stderr or "")[-4000:])
        try:
            out = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return worker_failure("worker_crash",
                            f"unreadable worker result: {e}",
                            returncode=proc.returncode)

    invalid = _validate(out)
    if invalid is not None:
        return invalid
    # Keep the child's stdio for diagnosis; a modeling `print` lands here and
    # is otherwise lost, and a warning from the kernel often explains a verdict.
    out["worker_stdout"] = (proc.stdout or "")[-8000:]
    out["worker_stderr"] = (proc.stderr or "")[-8000:]
    return out


__all__ = ["run_object_source", "run_cad_snippet", "is_available",
           "worker_failure", "FAILURE_KINDS", "shutdown_pool"]
