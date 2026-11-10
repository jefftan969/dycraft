"""Warm, reusable CAD worker processes for the object layer.

A separate interpreter buys dependency isolation (build123d lives in its own
env) and crash isolation (an OCCT fault costs a compile, not the run). A fresh
one per build would also charge `import build123d` every time, ~1.4 s and
nearly all of a trivial build, while OCCT faults in a worker are rare.
Dependency isolation needs a separate interpreter, not a fresh one per build,
so this module keeps the separate interpreter and stops re-paying the import.

**Why reuse is safe for state, not just for crashes.** `compile.build_ir_with_layer`
re-establishes everything a build can dirty, on every call:
`layer_cfg.singleton_reset()` clears the one-Object guard, the mesh-asset
session and the live-Shape compiler (`layer.reset_object_layer_state`, whose
docstring contemplates "a compiler from a worker run"), and
`_sandbox.build_globals()` hands the exec a fresh globals dict. Every scene
build and every probe calls it repeatedly in one process, the long-lived
daemon, so a process per build would pay for an isolation the function already
provides.

**What a fresh process still bought, and how that is preserved.** A wedged or
faulted child cannot be reasoned about, so it is never reused: a child is
*recycled* (killed, dropped, replaced on next demand) on timeout, on death, on
a protocol desync, and on `worker_internal`. The pool degrades to
spawn-per-build under repeated faults rather than handing back a suspect
interpreter.

Transport. One line of JSON per job into the child's stdin; one line of JSON
back as an ack. The build result itself still goes through a `result.json`
written by the child and read by the parent, byte-for-byte the same contract as
the one-shot path, so nothing downstream can tell which route ran.

Stdio. The child's fd 1 and fd 2 are regular *spool* files for its whole life,
so a modeling `print` and a C-level OCCT warning are both captured — including
when the child dies, which a pipe would lose. The ack channel is a private dup
of the original stdout taken before fd 1 is redirected, so protocol traffic and
build output cannot mix. Each ack carries the spool byte range the build wrote,
and the parent slices exactly that.

Concurrency. `agent/server.py` services handlers on a 32-thread pool, so two
object compiles can be in flight at once. Children are therefore pooled and
checked out, never shared: a child is owned by one build at a time.

Memory. A warm child holds ~500 MB RSS and a compile slot needs two of them
(authoring plus adjudication), so warmth is bought with ~1 GB of resident memory
per slot rather than with time. Two things keep that bounded: a small cap on
live children per script (`DYCRAFT_CAD_POOL_MAX_CHILDREN`), and an idle TTL
(`DYCRAFT_CAD_POOL_IDLE_TTL_S`) that retires a child which has gone unused —
because an agent compiles in bursts separated by long thinking pauses, and
holding a gigabyte through a pause buys nothing.

**Two consumers, one convention.** The authoring child
(`_object_worker_child.py`) and the exact-collision adjudicator
(`_adjudicate_runner.py`) both paid a kernel start per compile, and a production
compile runs both — so warming only the first would leave half the cost in
place. Any script that accepts `--serve --stdout-spool PATH`, answers one
newline-delimited JSON job per line, and acks `{"status": ...}` can be pooled
here. Children are keyed by (interpreter, script), so the two never mix.
`_adjudicate_runner.py` is deliberately dycraft-free — it cannot import this
module — so the convention is a wire contract, not shared code.
"""

from __future__ import annotations

import atexit
import errno
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ... import _oom
from ._cad_env import b123d_child_env

#: Set to "0" to force the historical spawn-per-build path. An escape hatch for
#: bisecting a suspected reuse bug without a code change.
_PERSISTENT_ENV_VAR = "DYCRAFT_OBJECT_WORKER_PERSISTENT"

#: Cap on simultaneously live children per (interpreter, script).
#:
#: Deliberately small, because a warm child is not free: a measured one holds
#: ~500 MB RSS, and a compile slot needs two (authoring + adjudication), so each
#: slot costs ~1 GB *while idle* — where the old spawn-per-build design paid that
#: only during a build. An object agent's FSM compiles serially, so the realistic
#: demand is one slot; the corpus in `remote/` bears that out at a median of one
#: worker spawn per run. Overflow is not an error — the caller spawns a
#: single-use interpreter exactly as it always did.
_MAX_CHILDREN_DEFAULT = 2
_MAX_CHILDREN_ENV_VAR = "DYCRAFT_CAD_POOL_MAX_CHILDREN"

#: Retire a child that has gone unused this long. An agent's compiles come in
#: bursts — a repair loop hammers the worker for a minute — separated by long
#: thinking pauses, which at high reasoning effort run to 10–20 minutes. Warmth
#: is what makes a burst cheap; holding ~1 GB through the pause between bursts
#: buys nothing, and paying 1.4 s once to get it back is the better trade.
#: 0 disables reaping.
_IDLE_TTL_DEFAULT_S = 300.0
_IDLE_TTL_ENV_VAR = "DYCRAFT_CAD_POOL_IDLE_TTL_S"
_REAP_INTERVAL_S = 30.0

_SPOOL_TAIL = 8000


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _max_children() -> int:
    return max(1, int(_env_float(_MAX_CHILDREN_ENV_VAR, _MAX_CHILDREN_DEFAULT)))


def _idle_ttl() -> float:
    return _env_float(_IDLE_TTL_ENV_VAR, _IDLE_TTL_DEFAULT_S)


class WorkerProtocolError(RuntimeError):
    """The child did not speak the ack protocol — treat it as a crash."""


class WorkerTimeout(RuntimeError):
    """The child outran the build's wall clock and has been killed."""


class _Child:
    """One warm CAD interpreter, owned by whoever checked it out."""

    def __init__(self, interp: str, script: Path, env: dict):
        self.interp = interp
        self.script = Path(script)
        self._dir = tempfile.mkdtemp(prefix="dycraft_cadpool_")
        self.out_spool = Path(self._dir) / "stdout.log"
        self.err_spool = Path(self._dir) / "stderr.log"
        self.out_spool.touch()
        # fd 2 is pointed straight at the spool by Popen, so an import error at
        # child startup lands there too rather than vanishing.
        self._err_fh = open(self.err_spool, "wb")
        # Resolve the interpreter before the OOM prefix goes in front of it.
        # A missing interpreter must surface as an OSError, which
        # `_snippet_pooled` turns into `worker_unavailable` -- a configuration
        # problem, distinct from `worker_crash`. With `choom` at argv[0] Popen
        # would succeed and the failure would arrive later, and less
        # truthfully, as a child that died without speaking.
        resolved = shutil.which(interp)
        if resolved is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), interp)
        self.proc = subprocess.Popen(
            # A warm CAD interpreter is ~1 GB and entirely disposable: the pool
            # respawns one that dies, and `acquire` falls back to a single-use
            # interpreter. So it carries the same OOM tier as a Blender worker
            # -- above the daemon and the agent session that cannot be respawned
            # without a human. See `_oom`.
            [*_oom.prefix(_oom.WORKER),
             resolved, str(self.script), "--serve",
             "--stdout-spool", str(self.out_spool)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._err_fh, env=env, bufsize=0,
            # A stable cwd for the child's whole life. The one-shot adjudicator
            # ran in a per-call temp dir; nothing reads relative paths (every
            # B-rep reference is absolute), but a directory that outlives the
            # process keeps OCCT's own scratch files from landing in whatever
            # the daemon's cwd happens to be.
            cwd=self._dir)
        self._stdout_fd = self.proc.stdout.fileno()
        self._rx = bytearray()
        self._ready = False
        self.last_used = time.monotonic()

    @property
    def key(self) -> str:
        """Pool identity. Two scripts under one interpreter are not
        interchangeable, so the script is part of the key."""
        return f"{self.interp}\0{self.script}"

    # -- lifecycle ---------------------------------------------------------

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        """Terminate and reap. Safe to call repeatedly."""
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 — teardown must not raise
            pass
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._err_fh.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        """Ask the child to exit by closing its stdin, then make sure it did."""
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self.kill()

    def ensure_ready(self, timeout: float) -> None:
        """Block until the child has finished importing build123d.

        Startup is deliberately *not* part of a build's timeout. Charging the
        cold import to the first job would let a short per-build timeout report
        a modeling timeout for a run that never got as far as running.
        """
        if self._ready:
            return
        ack = self._read_ack(time.monotonic() + timeout)
        if ack.get("status") != "ready":
            raise WorkerProtocolError(
                f"expected a ready ack from a new worker, got {ack!r}")
        self._ready = True

    # -- one job -----------------------------------------------------------

    def run_job(self, job: dict, deadline: float) -> dict:
        """Send one job and return the child's ack. Raises on fault."""
        line = json.dumps(job).encode() + b"\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            raise WorkerProtocolError(f"could not send the job: {e}") from e
        return self._read_ack(deadline)

    def _read_ack(self, deadline: float) -> dict:
        """Block for one newline-terminated JSON ack, or fault.

        Reads the raw fd rather than the buffered reader: `select` cannot see
        bytes already sitting in a Python-level buffer, and the two together
        can wait forever on data that has in fact arrived.
        """
        while True:
            nl = self._rx.find(b"\n")
            if nl >= 0:
                raw = bytes(self._rx[:nl])
                del self._rx[:nl + 1]
                try:
                    ack = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise WorkerProtocolError(
                        f"unparseable ack {raw[:200]!r}: {e}") from e
                if not isinstance(ack, dict):
                    raise WorkerProtocolError(f"ack was not an object: {raw[:200]!r}")
                return ack

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout()
            ready, _, _ = select.select([self._stdout_fd], [], [],
                                        min(remaining, 0.25))
            if not ready:
                # No data yet. A child that has exited will never produce any,
                # and checking here rather than only on EOF means a segfault is
                # reported in milliseconds instead of at the deadline.
                if self.proc.poll() is not None:
                    raise WorkerProtocolError(
                        f"the worker exited {self.proc.returncode} without an ack")
                continue
            try:
                chunk = os.read(self._stdout_fd, 65536)
            except OSError as e:
                raise WorkerProtocolError(f"ack channel read failed: {e}") from e
            if not chunk:
                raise WorkerProtocolError(
                    f"the worker closed its ack channel (exit "
                    f"{self.proc.poll()})")
            self._rx.extend(chunk)

    # -- captured build output --------------------------------------------

    def slice_spools(self, ack: dict, *,
                     cap: int | None = _SPOOL_TAIL) -> tuple[str, str]:
        """The stdout/stderr the child recorded for the build just finished.

        `cap` bounds how much of each span is read, because build output is
        diagnostic and a run that logged megabytes should not be loaded whole.
        Pass `cap=None` when the span *is* the answer rather than a note about
        it — a snippet job returns its result by printing it, so truncating the
        head of that span would silently corrupt the value the caller parses.
        """
        return (_read_span(self.out_spool, ack.get("stdout_span"), cap=cap),
                _read_span(self.err_spool, ack.get("stderr_span"), cap=cap))

    def spool_tails(self) -> tuple[str, str]:
        """Everything both spools hold — for a child that died mid-build and
        so never reported a span."""
        return (_read_tail(self.out_spool), _read_tail(self.err_spool))

    def discard_dir(self) -> None:
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)


def _read_span(path: Path, span, *, cap: int | None = _SPOOL_TAIL) -> str:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return ""
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return ""
    # A build that logged megabytes gets its tail kept, matching the one-shot
    # path's cap rather than loading the whole span into memory. `cap=None`
    # keeps the whole span, for callers whose result travels this way.
    if cap is not None:
        start = max(start, end - cap)
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start).decode("utf-8", "replace")
    except OSError:
        return ""


def _read_tail(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - _SPOOL_TAIL))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


class _Pool:
    """Warm children, keyed by interpreter, checked out one build at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._idle: dict[str, list[_Child]] = {}
        self._live: dict[str, int] = {}
        self._reaper: threading.Thread | None = None

    def enabled(self) -> bool:
        return os.environ.get(_PERSISTENT_ENV_VAR, "1") != "0"

    def acquire(self, interp: str, script) -> "_Child | None":
        """A warm child if one is idle, a new one if the cap allows, else None.

        None means "do this the one-shot way" — the caller keeps working instead
        of blocking on a busy pool.
        """
        script = Path(script)
        key = f"{interp}\0{script}"
        with self._lock:
            bucket = self._idle.setdefault(key, [])
            while bucket:
                child = bucket.pop()
                if child.alive():
                    return child
                # Died while idle. It never ran our job, so it is just garbage
                # to collect, not a fault to report.
                self._live[key] = max(0, self._live.get(key, 1) - 1)
                child.kill()
                child.discard_dir()
            if self._live.get(key, 0) >= _max_children():
                return None
            self._live[key] = self._live.get(key, 0) + 1
        self._ensure_reaper()
        # Spawn outside the lock: starting a CAD interpreter takes seconds and
        # must not block every other thread's checkout.
        try:
            return _Child(interp, script, b123d_child_env())
        except OSError:
            with self._lock:
                self._live[key] = max(0, self._live.get(key, 1) - 1)
            raise

    def release(self, child: _Child) -> None:
        child.last_used = time.monotonic()
        with self._lock:
            if child.alive():
                self._idle.setdefault(child.key, []).append(child)
                return
            self._live[child.key] = max(0, self._live.get(child.key, 1) - 1)
        child.kill()
        child.discard_dir()

    def recycle(self, child: _Child) -> None:
        """Drop a child that faulted. Never returns it to the idle set."""
        with self._lock:
            self._live[child.key] = max(0, self._live.get(child.key, 1) - 1)
        child.kill()
        child.discard_dir()

    # -- idle reaping ------------------------------------------------------

    def _ensure_reaper(self) -> None:
        """Start the idle reaper once, on first demand.

        A daemon thread, so it can never keep a process alive; and started
        lazily, so importing this module costs nothing in the many processes
        that never build any geometry.
        """
        with self._lock:
            if self._reaper is not None:
                return
            self._reaper = threading.Thread(
                target=self._reap_loop, name="cad-pool-reaper", daemon=True)
            self._reaper.start()

    def _reap_loop(self) -> None:
        while True:
            time.sleep(_REAP_INTERVAL_S)
            try:
                self.reap_idle()
            except Exception:  # noqa: BLE001 — the reaper must never die
                pass

    def reap_idle(self, ttl: float | None = None) -> int:
        """Retire idle children past their TTL. Returns how many went."""
        ttl = _idle_ttl() if ttl is None else ttl
        if ttl <= 0:
            return 0
        now = time.monotonic()
        doomed: list[_Child] = []
        with self._lock:
            for key, bucket in self._idle.items():
                keep = []
                for child in bucket:
                    if now - child.last_used >= ttl:
                        doomed.append(child)
                        self._live[key] = max(0, self._live.get(key, 1) - 1)
                    else:
                        keep.append(child)
                self._idle[key] = keep
        for child in doomed:
            child.close()
            child.discard_dir()
        return len(doomed)

    def shutdown(self) -> None:
        with self._lock:
            buckets = list(self._idle.values())
            self._idle = {}
            self._live = {}
        for bucket in buckets:
            for child in bucket:
                child.close()
                child.discard_dir()


POOL = _Pool()
atexit.register(POOL.shutdown)


def shutdown() -> None:
    """Stop every warm child. Idempotent; for tests and daemon teardown."""
    POOL.shutdown()


__all__ = ["POOL", "shutdown", "WorkerProtocolError", "WorkerTimeout"]
