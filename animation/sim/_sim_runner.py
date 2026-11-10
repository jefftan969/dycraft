"""Standalone runner for agent-authored code in a foreign interpreter (child).

Runs agent-authored code in an isolated interpreter that has a dependency the main environment does not, and
emits one JSON result. ``animation/ik/ik.py`` runs it for ``h.ik`` (a kinematics solver); nothing here names the
dependency. Invoked as::

    <foreign-python> _sim_runner.py --code CODE.py --result RESULT.json
                                    [--codename ik_code.py]
                                    [--checkpoint PARTIAL.json]
                                    [--setup SETUP.py]

with the working directory set to the exported world, so the code reaches ``world.urdf`` / ``bodies/`` /
``meshes/`` / ``scene_meta.json`` and ``import dycraft_ik`` by plain name.

The contract with the code is two injected globals. ``result`` is an empty dict, and whatever the code puts in it
comes back, with **no schema**: a script may produce a pose track or just measure something. ``checkpoint()``
snapshots ``result`` to disk, where the parent can read it if this process never returns (``_make_checkpoint``).
``dycraft_ik`` is importable but not injected, because a script that builds its own scene is a supported choice.

This file must stay importable under the foreign interpreter alone: no ``dycraft.*`` imports, stdlib only at module
scope, the rule ``agent/_harness_core.py`` follows for the staged ``harness.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import traceback


def _write_result(result_path: str, payload: dict) -> None:
    try:
        with open(result_path, "w") as f:
            json.dump(payload, f, default=_fallback)
    except (OSError, TypeError, ValueError) as e:
        # Last resort: a payload that will not serialize must still produce a
        # readable failure rather than an empty file the parent reads as
        # "runner_died".
        try:
            with open(result_path, "w") as f:
                json.dump({"ok": False, "error_kind": "unserializable_result",
                           "message": (
                               f"the script's `result` could not be written as "
                               f"JSON ({type(e).__name__}: {e}). Record plain "
                               f"numbers, strings, lists, and dicts — a tensor "
                               f"or an engine handle is not JSON. Convert with "
                               f".tolist() / float().")}, f)
        except OSError:
            pass


def _fallback(o):
    """Best-effort coercion for the near-misses: numpy scalars and arrays, and
    torch tensors, are what a simulation script reaches for first."""
    for attr in ("tolist", "item"):
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 — fall through to the next shot
                pass
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _make_checkpoint(result: dict, checkpoint_path: str | None):
    """Build the injected ``checkpoint()``: snapshot ``result`` where the parent
    can read it even if this process never returns.

    This is **not** for the script that raises. That path already works — the
    parent gets ``partial_result`` from the live ``result`` dict below, so a
    rollout that crashed at frame 300 of 400 comes back whole. It is for the two
    paths where the child never writes anything at all: killed at the timeout,
    or dead from a segfault in the engine. In both, ``--result`` points inside a
    ``TemporaryDirectory`` the parent then discards, so everything recorded is
    lost — and those are the expensive failures. One archived IK call burned its
    full 600 s budget and came back with no result of any kind.

    Atomic via ``os.replace``, so a snapshot taken while the previous one is
    still being written cannot leave a half-file the parent reads as corrupt.
    Silent on any error and a no-op when the parent passed no path: a script may
    call this unconditionally, and a failed snapshot must never be what breaks a
    rollout that was otherwise fine.

    Cost is one JSON dump of whatever ``result`` holds, so call it per phase, not
    per frame.
    """
    if not checkpoint_path:
        return lambda: None

    def checkpoint() -> None:
        try:
            tmp = checkpoint_path + ".writing"
            with open(tmp, "w") as f:
                json.dump(result, f, default=_fallback)
            os.replace(tmp, checkpoint_path)
        except (OSError, TypeError, ValueError):
            pass

    return checkpoint


def _run_setup(setup_path: str | None):
    """Run the caller's setup file before the authored code.

    In a namespace of its own, so a caller can shim its dependency's
    environment (the IK op's `_ik_setup.py`, for one) without this runner
    naming that dependency, and without prepending lines that would shift the
    authored code's traceback line numbers. A setup that fails is reported on
    stderr and otherwise ignored: it must never be what breaks a run.

    The one exception is a setup that *refuses* the run, by raising an error
    carrying an `error_kind` (the IK setup does when no GPU is visible). That
    error is returned, and the code is not run.
    """
    if not setup_path:
        return None
    try:
        runpy.run_path(setup_path, run_name="__dycraft_setup__")
    except Exception as e:  # noqa: BLE001 -- see above
        if isinstance(getattr(e, "error_kind", None), str):
            return e
        print(f"[runner] setup {setup_path} failed: {type(e).__name__}: {e}",
              file=sys.stderr)
    return None


def _fail(result_path: str, kind: str, message: str, **extra) -> int:
    _write_result(result_path, {"ok": False, "error_kind": kind,
                                "message": message, **extra})
    return 1


def _run(code_path: str, result_path: str,
         codename: str = "sim_code.py",
         checkpoint_path: str | None = None,
         setup_path: str | None = None) -> int:
    try:
        with open(code_path) as f:
            source = f.read()
    except OSError as e:
        return _fail(result_path, "runner_internal",
                     f"could not read the authored code: {e}")

    try:
        compiled = compile(source, codename, "exec")
    except SyntaxError as e:
        return _fail(result_path, "syntax_error",
                     f"{e.msg} (line {e.lineno})", lineno=e.lineno)

    result: dict = {}
    namespace = {"__name__": "__dycraft_sim__", "result": result,
                 "checkpoint": _make_checkpoint(result, checkpoint_path)}
    # cwd is the exported world; make it importable so `import dycraft_ik`
    # resolves without the script touching sys.path.
    if "" not in sys.path:
        sys.path.insert(0, "")
    refused = _run_setup(setup_path)
    if refused is not None:
        return _fail(result_path, refused.error_kind, str(refused))

    try:
        exec(compiled, namespace)  # noqa: S102 — isolated simulation subprocess
    except BaseException as e:  # noqa: BLE001 — any failure must be reported
        tb = traceback.format_exc()
        # Keep whatever the script managed to record before it died: a rollout
        # that crashed at frame 300 of 400 is still worth reading, and the
        # alternative is the agent re-running it blind.
        partial = namespace.get("result")
        return _fail(result_path, "raised",
                     f"{type(e).__name__}: {e}",
                     traceback=tb[-4000:],
                     partial_result=partial if isinstance(partial, dict) else None)

    _write_result(result_path, {"ok": True,
                                "result": namespace.get("result", result)})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--result", required=True)
    # Only reaches `compile`, so a traceback from the IK op reads `ik_code.py`
    # instead of claiming to come from a simulation. Default keeps the sim
    # caller's argv unchanged.
    parser.add_argument("--codename", default="sim_code.py")
    # Where `checkpoint()` writes. Optional: without it the injected callable is
    # a no-op, which keeps every existing caller's argv valid.
    parser.add_argument("--checkpoint", default=None)
    # A file to run before the code, in its own namespace (`_run_setup`).
    parser.add_argument("--setup", default=None)
    args = parser.parse_args()
    try:
        return _run(args.code, args.result, args.codename, args.checkpoint,
                    args.setup)
    except BaseException as e:  # noqa: BLE001 — always emit a result
        return _fail(args.result, "runner_internal",
                     f"{type(e).__name__}: {e}",
                     traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    sys.exit(main())
