"""A wall-clock bound for sandboxed code running inside the daemon.

The object layer's exec happens in the CAD worker subprocess, which
`object_worker` bounds with `subprocess.run(timeout=…)`. Scene builds and every
probe expression on both layers run in the long-lived daemon instead, where
without this bound a source of

    while True:
        pass

would hold the connection open forever.

**Why not `signal`.** `signal.setitimer` is the usual answer and it is the
wrong one here: signal handlers only run on the main thread, and `agent/server.py`
services handlers from a `ThreadPoolExecutor`. A signal-based limit would work
in tests and provide nothing in production — worse than no limit, because it
would read as covered.

So: a watchdog `Timer` asynchronously raises into the target thread through
`PyThreadState_SetAsyncExc`. That works on any thread and costs nothing while
idle.

**Known limitation, deliberate.** An asynchronous exception is delivered at the
next bytecode boundary, so it cannot interrupt a single long call inside C —
an OCCT boolean, a huge `sorted()`. That is exactly the case the CAD worker's
subprocess timeout already covers, and pure-Python runaway loops are what this
path is exposed to. Documented rather than papered over.
"""

from __future__ import annotations

import ctypes
import os
import threading

DEFAULT_LIMIT_S = 300.0
#: For the scene layer's `scene.py`, which builds every object it loads in
#: the one run. 300 s is sized for one object, and a whole-scene build often
#: needs longer.
SCENE_LIMIT_S = 900.0
LIMIT_ENV_VAR = "DYCRAFT_OBJECT_WORKER_TIMEOUT_S"


class SandboxTimeout(BaseException):
    """Raised into sandboxed code that outran its limit.

    `BaseException`, not `Exception`: it must not be swallowed by an
    `except Exception` in SDK or library code that the sandboxed source calls.
    """


def resolve_limit(explicit: float | None = None, *, scene: bool = False) -> float:
    """Seconds allowed: `SCENE_LIMIT_S` for a whole-scene build, else
    `DEFAULT_LIMIT_S`. Shares the worker's env var so both paths configure
    together; 0 or negative disables the bound."""
    default = SCENE_LIMIT_S if scene else DEFAULT_LIMIT_S
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(LIMIT_ENV_VAR)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class time_limit:
    """Context manager bounding the wall-clock time of the enclosed block.

    On expiry the block sees `SandboxTimeout`. Callers translate that into
    whatever their layer reports; this module deliberately knows nothing about
    IR or build errors.
    """

    __slots__ = ("_seconds", "_thread_id", "_timer", "_fired", "_done")

    def __init__(self, seconds: float | None = None):
        self._seconds = resolve_limit(seconds)
        self._thread_id = None
        self._timer = None
        self._fired = threading.Event()
        self._done = threading.Event()

    @property
    def expired(self) -> bool:
        return self._fired.is_set()

    def _fire(self):
        if self._done.is_set():
            return
        self._fired.set()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(self._thread_id), ctypes.py_object(SandboxTimeout))

    def __enter__(self):
        if self._seconds <= 0:
            return self
        self._thread_id = threading.get_ident()
        self._timer = threading.Timer(self._seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Order matters: mark done first so a timer that has already begun
        # running returns without arming anything.
        self._done.set()
        if self._timer is not None:
            self._timer.cancel()
        if self._fired.is_set():
            # The exception may have been queued but not yet delivered — it
            # would then surface at an arbitrary later point in this thread,
            # inside harness code. Clear any pending async exception.
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._thread_id), ctypes.c_void_p(0))
        return False
