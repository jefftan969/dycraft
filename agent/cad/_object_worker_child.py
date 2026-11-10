#!/usr/bin/env python3
"""Execute `object.py`, or an arbitrary CAD snippet, inside the CAD
interpreter (child process).

Two modes, two job kinds, one dispatcher (`_answer`).

Job kinds. A `build` job (the default, and the only one for most of this
module's life) runs an authored `object.py` through `build_ir_with_layer`. A
`snippet` job execs an arbitrary body and reports whether it raised, which is
what lets code needing the CAD *environment* rather than the object *layer* —
`_solid_compile`'s phases, `BRepTools`, the `_b123d_namespace` guard table —
share the warm child instead of paying `import build123d` per call. See
`_run_snippet` for the contract and its one caveat.

*One-shot* — the historical form, still used when pooling is disabled and as
the fallback when a warm child cannot be had::

    <b123d-python> _object_worker_child.py --job JOB.json --result RESULT.json

*Serve* — a warm child that pays `import build123d` once and then builds as many
programs as it is sent, which is what `_worker_pool` drives::

    <b123d-python> _object_worker_child.py --serve --stdout-spool OUT.log

Both funnel into `_build_one`, so a build cannot behave differently depending on
how the process was started.

This child uses the dycraft package — the whole point is that the SDK and
build123d are live in the same interpreter. dycraft is not installed in the CAD
environment, so it is bootstrapped from this file's own checkout by path,
which needs no symlink, no PYTHONPATH and no install.

Result protocol mirrors ``compile.build_ir_with_layer`` so the parent can hand
the payload straight back to its caller:

    {"ok": bool, "ir": {...}|null,
     "build_error": {...}|null, "sandbox_violation": {...}|null}

plus, on a failure outside that vocabulary, ``error_kind`` / ``message`` /
``traceback``.

**Serve-mode framing.** Jobs arrive as one line of JSON on stdin; each is
answered with one line of JSON on a private duplicate of the original stdout,
taken before fd 1 is redirected. Redirecting fd 1 is what makes the ack channel
trustworthy: a modeling `print`, or an OCCT warning written from C, would
otherwise interleave with protocol traffic and desync the parent. Both fd 1 and
fd 2 are regular files for the child's whole life, so the parent can still
recover build output from a child that died mid-build — which is exactly the
case a pipe would lose. Each ack reports the byte range the build appended, and
the parent slices that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _bootstrap_dycraft():
    """Import the `dycraft` package from this file's own checkout.

    The CAD environment has build123d but not dycraft, and the checkout
    directory is not named `dycraft`, so neither a plain import nor a
    `sys.path` entry resolves it. Loading from an explicit file location does,
    and binds the package to *this* tree rather than whatever else might be
    importable.
    """
    import importlib.util
    repo_root = Path(__file__).resolve().parents[2]
    if "dycraft" in sys.modules:
        return sys.modules["dycraft"]
    spec = importlib.util.spec_from_file_location(
        "dycraft", repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not bootstrap dycraft from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dycraft"] = module
    spec.loader.exec_module(module)
    return module


class _WorkerCtx:
    """The slice of ToolContext the build overlay and the SDK actually read.

    Deliberately minimal: a full ToolContext carries an LLM client, caches and
    tool plumbing that have no meaning in a compile-only child, and
    reconstructing one across the process boundary would couple the worker to
    changes that do not concern it.
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.step = 0


def _version_failure() -> dict | None:
    """The frozen-CAD-version check, as a result payload or None if it passes.

    Checked before any `object.py` runs, so a skewed interpreter is reported as
    an environment problem rather than surfacing later as geometry that
    measures subtly wrong. In serve mode this is settled once at startup: the
    interpreter cannot change under a running process, and re-asking per build
    would reintroduce a cost the warm child exists to remove.
    """
    from dycraft.agent.cad import _solid_compile
    try:
        _solid_compile.check_build123d_version()
        # OCP is bound in `object.py`'s globals, so its absence is an
        # authoring-surface fault, not a missing optional extra.
        _solid_compile.check_ocp_available()
    except _solid_compile.SolidCompileError as e:
        # Same payload shape the parent produces, from the same constructor, so
        # `error_kind` means one thing whichever side reported it.
        from dycraft.agent.cad.object_worker import worker_failure
        return worker_failure(e.kind, e.message)
    return None


def _build_one(job: dict) -> dict:
    """Build one authored program. The single build path for both modes.

    `layer` selects the config. It exists because the scene layer authors
    geometry too: its `scene.py` calls `Object(...)` and needs build123d, so it
    has to build in this interpreter rather than in the daemon, exactly as
    `object.py` does. Everything downstream of the build — verifiers, probes,
    renderers — still runs in the parent for both.
    """
    run_dir = Path(job["run_dir"])
    source = Path(job["source_path"]).read_text(encoding="utf-8")

    from dycraft.agent.layers.compile import build_ir_with_layer
    from dycraft.agent.layers.layer import (object_layer_config,
                                                  scene_layer_config)

    cfg = (scene_layer_config() if job.get("layer") == "scene"
           else object_layer_config())
    ctx = _WorkerCtx(run_dir)
    return build_ir_with_layer(source, layer_cfg=cfg, ctx=ctx)


def _run_snippet(job: dict) -> dict:
    """Exec an arbitrary program inside this interpreter and report how it went.

    **Why this exists.** A build job can only run an authored `object.py`
    through `build_ir_with_layer`, which is the wrong shape for the tests and
    tools that need to reach *into* the CAD environment — `_solid_compile`'s
    phases, `BRepTools`, the `_b123d_namespace` guard table. Launching
    `<b123d-python> -c <body>` directly would re-pay `import build123d`
    (~1.5 s) every time; this job kind lets them share the warm child.

    **Contract.** The body's stdout *is* its result: print one line of JSON and
    the parent hands it back verbatim. `ok` reports whether the body raised,
    which is how a caller gets the failing behavior a non-zero exit would
    give it — the exception is caught here so that a body which is *meant* to
    raise does not look like a worker fault and cost the pool a warm child.

    **What the caller owes.** Bodies share one long-lived interpreter, so each
    gets a fresh globals dict but *not* a fresh process. A body that mutates
    process-global state (`sys.modules`, a monkeypatched attribute, a module
    singleton) can change what a later body sees. `run_cad_snippet(...,
    isolate=True)` forces a single-use interpreter for those.
    """
    body = job["body"]
    # A fresh mapping per body, so two snippets cannot see each other's names.
    # `__builtins__` is passed through deliberately: this is not a sandbox, it
    # is the same interpreter access the direct `-c` invocation already had.
    g = {"__name__": "__dycraft_snippet__", "__builtins__": __builtins__,
         "__file__": "<dycraft-snippet>"}
    try:
        exec(compile(body, "<dycraft-snippet>", "exec"), g)
    except BaseException as e:  # noqa: BLE001 — the body's failure is data
        return {"ok": False, "error_kind": "snippet_error",
                "message": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}
    return {"ok": True}


def _answer(job: dict) -> dict:
    """Route one job to its handler. Both modes go through here, so a job kind
    cannot be understood in serve mode and not one-shot."""
    if job.get("kind") == "snippet":
        return _run_snippet(job)
    return _build_one(job)


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------

def _run(job_path: str, result_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))

    _bootstrap_dycraft()

    failure = _version_failure()
    if failure is not None:
        Path(result_path).write_text(json.dumps(failure), encoding="utf-8")
        return 1

    out = _answer(job)
    Path(result_path).write_text(json.dumps(out), encoding="utf-8")
    # A snippet reports its outcome in the payload the parent reads, and the
    # parent never looks at the exit status, so it exits 0 either way. Only a
    # build keeps the historical "non-zero means it failed" convention.
    if job.get("kind") == "snippet":
        return 0
    return 0 if out.get("ok") else 1


# ---------------------------------------------------------------------------
# Serve mode
# ---------------------------------------------------------------------------

def _offsets() -> tuple[int, int]:
    """Current write positions of fd 1 and fd 2.

    Python's buffers are flushed first: an unflushed `print` would otherwise be
    attributed to whichever build happens to trigger the flush.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 — a closed stream must not end the run
            pass
    def _pos(fd: int) -> int:
        try:
            return os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            return 0
    return _pos(1), _pos(2)


def _serve(stdout_spool: str) -> int:
    """Answer jobs until stdin reaches EOF.

    EOF is the shutdown signal: the parent closes the pipe, and a child whose
    parent has gone away reads EOF too, so nothing is left running behind a
    daemon that died.
    """
    # Take the ack channel *before* fd 1 becomes the spool, and keep it off the
    # inheritable set so a grandchild cannot scribble on the protocol.
    proto_fd = os.dup(1)
    os.set_inheritable(proto_fd, False)

    sys.stdout.flush()
    spool_fd = os.open(stdout_spool, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(spool_fd, 1)
    os.close(spool_fd)

    def ack(payload: dict) -> None:
        os.write(proto_fd, (json.dumps(payload) + "\n").encode())

    _bootstrap_dycraft()
    version_failure = _version_failure()
    if version_failure is None:
        # Force the expensive imports now rather than on the first job, so the
        # warmth is real by the time the parent sends anything.
        from dycraft.agent.layers.layer import load_solid_compiler
        load_solid_compiler()

    # Announce warmth. The parent starts each build's clock only after this,
    # so a cold start is never charged against a build's timeout — otherwise
    # the first job on a new child could be reported as a modeling timeout
    # when what actually happened was `import build123d`.
    ack({"status": "ready"})

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        out_start, err_start = _offsets()
        try:
            job = json.loads(line)
            result = version_failure if version_failure is not None \
                else _answer(job)
            Path(job["result"]).write_text(json.dumps(result), encoding="utf-8")
            status = "ok"
            detail = None
        except BaseException as e:  # noqa: BLE001 — never propagate out
            # The child's own fault, not the authored source's, so it gets its
            # own kind rather than masquerading as a build error. The parent
            # recycles this child; reporting rather than dying keeps the
            # failure classified instead of arriving as an opaque crash.
            status = "worker_internal"
            detail = f"{type(e).__name__}: {e}"
            try:
                payload = _internal_failure(detail)
                Path(json.loads(line)["result"]).write_text(
                    json.dumps(payload), encoding="utf-8")
            except BaseException:  # noqa: BLE001 — nowhere left to write it
                pass
        out_end, err_end = _offsets()
        try:
            ack({"status": status, "detail": detail,
                 "stdout_span": [out_start, out_end],
                 "stderr_span": [err_start, err_end]})
        except OSError:
            # The parent is gone; there is no one to answer.
            return 1
        # After the ack, so the parent never waits on it.
        _release_memory()


def _release_memory() -> None:
    """Hand a finished build's memory back to the system.

    A warm child builds job after job, and memory freed by a build stays in
    glibc's arenas rather than going back to the OS. The saving is small (about
    3% of a warm child's memory over 21 builds), because most of what it holds
    is live state, but the call is cheap.
    Best effort: not glibc, or no `malloc_trim`, and nothing happens.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _internal_failure(message: str) -> dict:
    try:
        from dycraft.agent.cad.object_worker import worker_failure
        return worker_failure("worker_internal", message,
                              traceback=traceback.format_exc())
    except BaseException:  # noqa: BLE001 — the bootstrap itself failed
        return {
            "ok": False, "ir": None, "build_error": None,
            "sandbox_violation": None,
            "error_kind": "worker_internal",
            "message": message,
            "traceback": traceback.format_exc(),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="dycraft object worker child")
    parser.add_argument("--job")
    parser.add_argument("--result")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--stdout-spool")
    args = parser.parse_args(argv)

    if args.serve:
        if not args.stdout_spool:
            parser.error("--serve requires --stdout-spool")
        try:
            return _serve(args.stdout_spool)
        except BaseException:  # noqa: BLE001 — die quietly; the parent notices
            traceback.print_exc()
            return 1

    if not args.job or not args.result:
        parser.error("one-shot mode requires --job and --result")
    try:
        return _run(args.job, args.result)
    except BaseException as e:  # noqa: BLE001 — never propagate out of the child
        Path(args.result).write_text(
            json.dumps(_internal_failure(f"{type(e).__name__}: {e}")),
            encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
