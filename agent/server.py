"""Generic long-lived harness daemon: the transport and lifecycle every layer shares.

It frames length-prefixed JSON (a 4-byte big-endian length, both directions, as in
`agent/_harness_core.py`), runs the accept loop, and dispatches each method name to
`session.handlers`, which each `AgentSession` subclass fills with its ops. The
per-layer server modules bind a session and call `serve()`:

    python -m dycraft.agent.layers.server serve <run_dir>
    python -m dycraft.agent.plain.server serve <run_dir>

The launcher starts one, chosen by layer, before the agent CLI
(`cli/cli.py::start_harness_daemon`).

The socket is TCP loopback on a port the kernel picks, published in
`<run_dir>/.harness.port`, so concurrent runs cannot collide. Not AF_UNIX: Claude
Code's sandbox seccomp filter blocks `socket(AF_UNIX, ...)` but allows AF_INET,
which the agent needs for its API anyway. Handlers run on a bounded thread pool,
so a slow op (`review`, an `await_any` that waits) does not block the others, and
stdout is captured per thread (`_StdoutDemux`) so their output does not mix.

The daemon also owns the parent-pid watchdog, which exits when
`DYCRAFT_AGENT_PID` disappears (retargeted from the launcher to the agent CLI's own
process); SIGTERM/SIGINT as SystemExit, so the atexit hooks
(`session.finalize_meta`) run; and the optional `session.drain_on_shutdown`,
called before `finalize_meta` so in-flight sub-runs are cleaned up.
"""
from __future__ import annotations

import atexit
import errno
import gzip
import hashlib
import io
import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PORT_NAME = ".harness.port"
LOOPBACK_HOST = "127.0.0.1"
#: Named-pipe transport, published beside the port. See `_harness_core.py`'s
#: `FIFO_PATH` for why it exists: codex's sandbox blocks every socket
#: connection from inside it -- TCP loopback, AF_UNIX and even `socketpair()`
#: all return EPERM -- so a harness call over a socket costs a sandbox
#: escalation, and every escalation was a full turn of a second model session.
#: A FIFO is ordinary file I/O in a directory the sandbox already grants write
#: on. Both transports stay up; the client picks.
FIFO_NAME = ".harness.fifo"
CALLS_DIR_NAME = ".harness.calls"
#: How long a worker waits for the calling process to open its reply pipe
#: before giving the reply up as undeliverable. A client that is alive opens it
#: within microseconds -- it is already blocked on that open -- so this only
#: bounds the case where the caller died between announcing and reading.
REPLY_OPEN_TIMEOUT_S = 30.0
#: How long a reply write may make no progress before the reader is presumed
#: gone. A bound on *stalls*, not on the whole write: a live reader drains a
#: large reply at its own pace: a storyboard result of over 100 MB takes about
#: 24 s to cross the pipe on an idle machine, and longer under load.
REPLY_STALL_TIMEOUT_S = 30.0
#: Written to the FIFO by the daemon itself to release the reader thread.
_FIFO_STOP = "__stop__"

# Single-owner lock for a run directory. Held for the daemon's whole lifetime
# via flock, so the kernel releases it on exit — including a SIGKILL — and a
# stale file can never lock a run dir out.
OWNER_LOCK_NAME = ".harness.owner.lock"
OWNER_INFO_NAME = ".harness.owner.json"


class DaemonAlreadyOwned(RuntimeError):
    """Raised when another live daemon already owns this run directory."""

    def __init__(self, run_dir: Path, info: dict):
        self.run_dir = run_dir
        self.info = info
        super().__init__(
            f"harness daemon already owns {run_dir} "
            f"(pid={info.get('pid')}, port={info.get('port')})")


def read_owner_info(run_dir: Path) -> dict:
    """The incumbent daemon's recorded identity, or `{}`."""
    try:
        data = json.loads((Path(run_dir) / OWNER_INFO_NAME).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def acquire_run_dir(run_dir: Path):
    """Take exclusive ownership of `run_dir`, or raise `DaemonAlreadyOwned`.

    A run directory has exactly one harness daemon. Everything the daemon
    writes is single-writer state that a second daemon would corrupt rather
    than merely duplicate: `trace.jsonl` and `meta.json` are appended and
    rewritten without locking, `session.step` is an in-memory counter (so two
    daemons re-use step numbers and overwrite each other's
    `compile_reports/<step>.json` and `renders/<step>_*`), and whichever one
    exits last wins the `daemon_shutdown` outcome. It is a real case: an agent
    can relaunch the daemon from inside its own sandbox, to change the render
    backend for instance.

    The lock is an flock held for the process lifetime, so it is released by
    the kernel on any exit path and a crashed daemon never leaves a run dir
    permanently locked. The returned handle must be kept alive; closing it
    releases ownership.
    """
    import fcntl

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = open(run_dir / OWNER_LOCK_NAME, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Contention means a *live* holder: flock is released on process death,
        # so a stale lock cannot get here.
        handle.close()
        raise DaemonAlreadyOwned(run_dir, read_owner_info(run_dir)) from None
    return handle


def write_owner_info(run_dir: Path, *, port: int) -> None:
    """Record who owns this run dir, for diagnostics and for a second
    launcher's refusal message."""
    info = {"pid": os.getpid(), "port": port,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "render_backend": os.environ.get("DYCRAFT_RENDER_BACKEND") or None}
    path = Path(run_dir) / OWNER_INFO_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Length-prefixed JSON framing (mirrors _harness_core.py)
# ---------------------------------------------------------------------------

def _send(conn: socket.socket, payload: dict) -> None:
    data = json.dumps(payload, default=str).encode("utf-8")
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv(conn: socket.socket) -> dict | None:
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    (n,) = struct.unpack(">I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode("utf-8"))


# ---------------------------------------------------------------------------
# Parent-process watchdog with agent-pid discovery
# ---------------------------------------------------------------------------

def _find_agent_pid(run_dir: Path, marker: str) -> int | None:
    """Scan /proc for the agent CLI process by argv match.

    Looks for a process whose raw cmdline (null-separated argv) contains:
      - `--add-dir\\0<run_dir>\\0` as an exact adjacent argv-pair, and
      - the backend's marker substring (e.g., "claude" or "codex").

    The argv-pair match (null-bounded) is stricter than a substring match:
    it disambiguates exp-names that share a prefix (`runs/cube` would
    otherwise textually match `runs/cube_large`). One small cmdline read
    per /proc entry — no /proc/*/fd traversal needed.

    Returns the first matching pid, or None if no agent is currently in /proc
    (which is the expected steady state until the launcher starts the
    agent, and after the agent exits).
    """
    if not marker:
        return None
    needle = f"--add-dir\0{str(run_dir)}\0".encode()
    marker_b = marker.encode().lower()
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if needle in cmdline and marker_b in cmdline.lower():
            return int(entry)
    return None


class _Watchdog:
    """SIGTERMs the daemon when the launcher that started it is gone.

    The launcher passes its own pid ($DYCRAFT_AGENT_PID). It runs the agent CLI
    as its child and lives as long as the run does, so it is the pid to follow:
    a CLI that exits and is started again by the same launcher (a startup
    retry) keeps this daemon, which a watchdog following the CLI's own pid
    would have shut down with the first CLI.

    A launcher that predates that became the CLI by `execvpe`, and the Claude
    Code binary then forked a worker, so the launcher pid disappeared while
    the agent ran on at another pid. So once the launcher pid is gone, the
    watchdog looks in /proc for `--add-dir\\0<run_dir>\\0` plus the backend's
    argv marker and retargets to that pid, tolerating the missing launcher for
    a grace window (default 60 s from start). Once retargeted, the agent pid's
    end shuts the daemon down at once (orphan cleanup).

    SIGTERMing our own pid (rather than os._exit) ensures the main thread's
    atexit hooks (Session.finalize_meta, trace.jsonl shutdown event) run.
    """

    def __init__(self, initial_pid: int, run_dir: Path, marker: str, *,
                 poll_s: float = 2.0, grace_s: float = 60.0) -> None:
        self._target_pid = initial_pid
        self._retargeted = False
        self._lock = threading.Lock()
        self._run_dir = run_dir
        self._marker = marker
        self._poll_s = poll_s
        self._grace_s = grace_s
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the watchdog thread to exit on its next poll without
        SIGTERMing the daemon. Useful in tests (so the watchdog doesn't
        kill pytest when test cleanup tears down the agent process) and
        for graceful daemon shutdown paths that already know they're
        exiting.
        """
        self._stop_event.set()

    @property
    def target_pid(self) -> int:
        with self._lock:
            return self._target_pid

    @property
    def retargeted(self) -> bool:
        with self._lock:
            return self._retargeted

    def _retarget(self, new_pid: int) -> None:
        with self._lock:
            previous = self._target_pid
            self._target_pid = new_pid
            self._retargeted = True
        print(f"[daemon] watchdog retargeted from launcher pid {previous} "
              f"to agent pid {new_pid}", flush=True)

    def start(self) -> None:
        threading.Thread(target=self._loop, name="harness-parent-watchdog",
                         daemon=True).start()

    def _loop(self) -> None:
        grace_deadline = time.time() + self._grace_s
        while not self._stop_event.is_set():
            target, retargeted = self.target_pid, self.retargeted
            if _pid_alive(target):
                time.sleep(self._poll_s)
                continue
            if not retargeted:
                # The launcher is gone. One that became the agent CLI by exec
                # leaves the CLI running under another pid: follow that.
                found = _find_agent_pid(self._run_dir, self._marker) if self._marker else None
                if found:
                    self._retarget(found)
                    continue
                if time.time() < grace_deadline:
                    time.sleep(self._poll_s)
                    continue
            reason = "agent" if retargeted else "launcher"
            print(f"[daemon] {reason} pid {target} gone; shutting down",
                  flush=True)
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                                      # exists, not ours to signal
    return True


# ---------------------------------------------------------------------------
# Per-thread stdout capture
# ---------------------------------------------------------------------------

class _StdoutDemux:
    """`sys.stdout` proxy that routes writes to a per-thread `StringIO`
    when one is set, falling back to the original stdout otherwise.

    Concurrent handler workers each install their own buffer for the
    duration of a call and then read it back — without this, two threads
    using `contextlib.redirect_stdout` would clobber each other's
    `sys.stdout` swap.
    """

    def __init__(self, real: object) -> None:
        self._real = real
        self._local = threading.local()

    def push(self) -> io.StringIO:
        buf = io.StringIO()
        # `_stack` is a per-thread stack — handlers nest cleanly even if
        # one accidentally re-enters (shouldn't happen, but cheap to
        # support).
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(buf)
        return buf

    def pop(self) -> str:
        stack = getattr(self._local, "stack", None) or []
        if not stack:
            return ""
        return stack.pop().getvalue()

    def _target(self):
        stack = getattr(self._local, "stack", None) or []
        return stack[-1] if stack else self._real

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        try:
            return self._target().flush()
        except Exception:  # noqa: BLE001
            return None

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

# Default worker count. Most workers are blocked in Popen.wait, so high
# concurrency is cheap.
#: The checkout this daemon was imported from: a run's worktree, or the main one.
_SOURCE_REPO = Path(__file__).resolve().parents[1]
#: Diff flags pinned against any git config, and the paths left out of the
#: patch. A launcher that rebuilds a run's tree from `{base, patch}` must use
#: the same ones, so that the patch recorded here is the one the tree was built
#: with.
_SOURCE_DIFF = ("diff", "--binary", "--full-index", "--no-renames", "--no-color",
                "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
                "--unified=3", "--diff-algorithm=myers", "--indent-heuristic", "-O/dev/null")
_SOURCE_EXCLUDE = ("--", ".", ":!claude", ":!paper", ":!tests")


def _source_record(run_dir: Path, repo: Path = _SOURCE_REPO) -> dict | None:
    """The code this daemon runs, as `{base, patch}`.

    `base` is the last pushed commit under HEAD; `patch` is the sha256 of the
    diff from it to the files on disk, kept once per distinct patch as
    `run_dir/source/<patch>.patch.gz`. A worktree launcher can rebuild the tree
    from the pair, so a resume can run exactly this code. `None` when git
    cannot say.
    """
    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, timeout=60).stdout
    try:
        remotes = git("for-each-ref", "--format=%(objectname)", "refs/remotes").split()
        try:
            base = git("merge-base", "HEAD", *(r.decode() for r in remotes)) if remotes else b""
        except subprocess.CalledProcessError:
            base = b""
        base = (base or git("rev-parse", "HEAD")).decode().strip()
        patch = git(*_SOURCE_DIFF, base, *_SOURCE_EXCLUDE)
        digest = hashlib.sha256(patch).hexdigest()
        target = Path(run_dir) / "source" / f"{digest}.patch.gz"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            partial.write_bytes(gzip.compress(patch, mtime=0))
            partial.replace(target)
    except (OSError, subprocess.SubprocessError):
        return None
    return {"base": base, "patch": digest}


def _handler_failure(error: BaseException, run_dir: Path, method: str) -> dict:
    """The reply for a handler that raised.

    An SDK error (`DycraftError`) is about the author's own source. A
    `ValueError` is a refusal, written for the agent, and its message is the
    reply. Anything else is a fault in the harness: one line, with the
    traceback appended to `harness_errors.log` rather than printed into the
    agent's terminal, where it read as something to repair in the scene.
    """
    from ..sdk._errors import DycraftError
    if isinstance(error, DycraftError):
        # About the author's own source; see `runtime.failed_result`.
        reply = {"error": str(error), "failure_class": "author_code"}
        if callable(getattr(error, "to_dict", None)):
            reply["error_detail"] = error.to_dict()
        return reply
    if isinstance(error, ValueError):
        return {"error": str(error), "failure_class": "rejection"}
    from ..animation.blender.runtime import harness_fault
    log = Path(run_dir) / "harness_errors.log"
    try:
        with log.open("a") as f:
            f.write(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} {method}\n"
                    f"{traceback.format_exc()}\n")
    except OSError:
        pass
    return {"error": harness_fault(f"{type(error).__name__}: {error}", log),
            "failure_class": "harness"}


def _committed_outcome(run_dir: Path) -> str:
    """How a committed run committed, from its `ir.json`: a commit with
    exceptions is `delivered_with_exceptions`, not a clean `committed`."""
    try:
        outcome = json.loads((Path(run_dir) / "ir.json").read_text()).get("outcome")
    except (OSError, ValueError, AttributeError):
        outcome = None
    return outcome if isinstance(outcome, str) and outcome else "committed"


def _active_render_backend() -> str:
    """The render backend this daemon will actually use.

    Read live rather than captured: `render.default_backend()` consults
    `$DYCRAFT_RENDER_BACKEND` on every request, so the `set_render_backend`
    op can change it mid-run and every subsequent render follows.
    """
    return os.environ.get("DYCRAFT_RENDER_BACKEND") or "open3d"


DEFAULT_MAX_WORKERS = 32
# Drain grace for in-flight handlers on shutdown. Handlers should be
# cancellable in well under this; the bound is to keep `.terminate` from
# hanging if a worker is wedged.
DEFAULT_SHUTDOWN_DRAIN_S = 30.0


def serve(session, run_dir: Path, *, start_fields: dict | None = None,
          max_workers: int = DEFAULT_MAX_WORKERS,
          shutdown_drain_s: float = DEFAULT_SHUTDOWN_DRAIN_S) -> None:
    """Bind a TCP loopback socket under run_dir and dispatch incoming
    requests to `session.handlers`. Blocks until killed.

    `session` is any `AgentSession` subclass; only its agent-agnostic
    surface is used here — `.handlers`, `.trace`, `.finalize_meta`,
    `.ctx.committed`, and (optionally) `.drain_on_shutdown()` for
    agent-specific in-flight cleanup. `start_fields` is merged into the
    `daemon_start` trace event so each agent class can stamp its own
    context (layer, …) without this module knowing those
    concepts.

    Concurrency: a `ThreadPoolExecutor` services handlers off the accept
    loop. The accept loop only `recv`s the request frame and submits the
    handler call to the pool — long-running handlers (`dispatch_*`)
    cannot HOL-block concurrent harness calls from the agent (`read_file`,
    `list_dispatches`, `await_dispatch`, …).

    Each worker captures its own stdout via `_StdoutDemux` so concurrent
    `print(...)` calls inside handlers don't cross-contaminate. Installs
    SIGTERM / SIGINT handlers that raise SystemExit so atexit hooks
    (session.finalize_meta) get a chance to run.
    """
    handlers = session.handlers
    if not handlers:
        raise SystemExit("daemon: session.handlers is empty; the "
                         "AgentSession subclass registered no tool fns.")

    # Ownership first: before the socket is bound, before `.harness.port` is
    # rewritten, and before a single trace event is appended. A second daemon
    # that got as far as writing the port file would already have stolen the
    # agent's connections from the incumbent.
    try:
        owner_handle = acquire_run_dir(run_dir)
    except DaemonAlreadyOwned as already:
        info = already.info
        print(f"[daemon] REFUSING: {run_dir} already has a harness daemon "
              f"(pid={info.get('pid')}, port={info.get('port')}). A run "
              f"directory has exactly one daemon; a second would corrupt "
              f"trace.jsonl, meta.json, and the step counter rather than "
              f"duplicate them. The existing daemon remains in charge and "
              f"$RUN_DIR/{PORT_NAME} still points at it, so the agent's "
              f"harness calls keep working — attach to it rather than "
              f"restarting. To change the render backend mid-run use the "
              f"`set_render_backend` op; to change it for a whole run use "
              f"`--render-backend` on `python -m dycraft.cli`.",
              file=sys.stderr, flush=True)
        # Exit 0: the desired end state (one live daemon for this run dir) is
        # already true, so this is a no-op attach, not a failure. A non-zero
        # exit here would make the launcher's `start_harness_daemon` abort a
        # perfectly healthy run.
        raise SystemExit(0)

    def _shutdown(*_args) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Install the per-thread stdout demultiplexer once for the daemon's
    # lifetime. Worker threads push/pop their own buffer; the accept loop
    # and other threads see the real stdout.
    stdout_demux = _StdoutDemux(sys.stdout)
    sys.stdout = stdout_demux  # type: ignore[assignment]

    port_path = run_dir / PORT_NAME
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LOOPBACK_HOST, 0))
    port = srv.getsockname()[1]
    # listen() backlog generous enough that an agent issuing a burst of
    # parallel submits never trips the kernel's SYN drop. Workers grab off
    # the pool fast enough that the backlog should stay near-empty in
    # practice.
    srv.listen(64)
    tmp_port_path = port_path.with_suffix(port_path.suffix + ".tmp")
    tmp_port_path.write_text(str(port))
    os.chmod(tmp_port_path, 0o600)
    os.replace(tmp_port_path, port_path)
    write_owner_info(run_dir, port=port)
    print(f"[daemon] listening on {LOOPBACK_HOST}:{port} "
          f"(pid={os.getpid()}, port_file={port_path}, "
          f"max_workers={max_workers}, "
          f"render_backend={os.environ.get('DYCRAFT_RENDER_BACKEND') or 'open3d'})",
          flush=True)

    parent_pid_env = os.environ.get("DYCRAFT_AGENT_PID")
    agent_marker = os.environ.get("DYCRAFT_AGENT_CMDLINE_MARKER", "")
    watchdog = None
    if parent_pid_env and parent_pid_env.isdigit():
        watchdog = _Watchdog(int(parent_pid_env), run_dir, agent_marker)
        watchdog.start()
        if not agent_marker:
            print("[daemon] DYCRAFT_AGENT_CMDLINE_MARKER unset; watchdog "
                  "cannot retarget on agent-pid drift (each new backend "
                  "must declare Backend.agent_cmdline_marker — see "
                  "cli/README.md)", flush=True)
    else:
        print("[daemon] no DYCRAFT_AGENT_PID set; running without "
              "parent watchdog", flush=True)

    pool = ThreadPoolExecutor(max_workers=max_workers,
                              thread_name_prefix="harness-handler")

    @atexit.register
    def _on_shutdown() -> None:
        if watchdog is not None:
            watchdog.stop()
        # Run agent-specific drain (e.g. cancel in-flight sub-dispatches)
        # BEFORE we tear down the worker pool. Handlers' Popen objects
        # live in agent state, so the drain hook is the only place that
        # can SIGTERM them. Bounded by shutdown_drain_s.
        drain = getattr(session, "drain_on_shutdown", None)
        if callable(drain):
            try:
                drain(timeout_s=shutdown_drain_s)
            except BaseException as e:  # noqa: BLE001
                print(f"[daemon] drain_on_shutdown raised "
                      f"{type(e).__name__}: {e}", flush=True)
        # Shut the pool down without waiting for stuck handlers — the
        # drain hook should have already canceled / killed everything
        # that matters. `cancel_futures=True` drops queued work; running
        # work is given the same drain budget.
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python <3.9 (cancel_futures kwarg)
            pool.shutdown(wait=False)
        outcome = (_committed_outcome(run_dir) if session.ctx.committed
                   else "interrupted")
        session.trace("daemon_shutdown", outcome=outcome,
                      render_backend=_active_render_backend())
        # `render_backend` on meta comes from the daemon that actually
        # rendered, not from the launcher's flag, so a `set_render_backend`
        # mid-run is reflected instead of contradicted.
        session.finalize_meta(outcome,
                              render_backend=_active_render_backend())
        # Release ownership last, so nothing can start writing this run dir
        # while we are still flushing meta.json and the transcript. (flock
        # would be released by process exit anyway; doing it explicitly and
        # last makes the ordering a property of the code, not of the runtime.)
        try:
            owner_handle.close()
        except OSError:
            pass

    # `render_backend` is stamped here rather than by each layer's
    # `start_fields`, so the value recorded is always the one the *owning*
    # daemon will actually render with; the launcher's value is wrong for a
    # daemon relaunched with another backend.
    session.trace("daemon_start", pid=os.getpid(),
                  render_backend=_active_render_backend(),
                  source=_source_record(run_dir),
                  **(start_fields or {}))

    def _dispatch(req: dict) -> dict:
        """Run one request and build its response.

        Shared by both transports -- the socket accept loop and the FIFO reader
        -- so that adding the pipe could not give the two paths different
        semantics for stdout capture, unknown methods or handler exceptions.
        That divergence is exactly what the `job_status` shaping comment below
        records happening the last time two paths served the same ops.
        """
        method = req.get("method", "")
        args = req.get("args") or {}
        handler = handlers.get(method)
        if handler is None:
            return {
                "ok": False,
                "error": f"unknown method: {method!r} "
                         f"(known: {sorted(handlers)})",
                "stdout": "",
            }
        stdout_demux.push()
        try:
            try:
                result = handler(**args)
                # Pretty-print so the agent's "import harness as h" shell
                # sees the same human-friendly JSON dump.
                print(json.dumps(result, indent=2,
                                 sort_keys=True, default=str))
                return {"ok": True, "result": result,
                        "stdout": stdout_demux.pop()}
            except SystemExit as e:
                # Worker-thread SystemExit must NOT propagate into the
                # executor (it would tear down the worker but not the
                # daemon, leaving a half-dead pool). We surface it as a
                # handler-level error instead.
                return {"ok": False,
                        "error": f"SystemExit: {e}",
                        "stdout": stdout_demux.pop()}
            except BaseException as e:  # noqa: BLE001
                return {"ok": False, **_handler_failure(e, run_dir, method),
                        "stdout": stdout_demux.pop()}
        except BaseException:
            # Even if the response builder itself blew up, drain the
            # per-thread stdout stack so we don't leak buffers across
            # reuses of the worker.
            while stdout_demux.pop():
                pass
            raise

    def _handle(conn: socket.socket) -> None:
        # Each worker owns one connection from recv → send → close.
        try:
            req = _recv(conn)
            if req is None:
                return
            resp = _dispatch(req)
            try:
                _send(conn, resp)
            except OSError:
                # Client may have walked away (agent timed out, sandbox
                # killed the bash); not much we can do.
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ---------------------------------------------------------------
    # Named-pipe transport
    # ---------------------------------------------------------------
    calls_dir = run_dir / CALLS_DIR_NAME
    calls_dir.mkdir(exist_ok=True)
    fifo_path = run_dir / FIFO_NAME
    fifo_tmp = run_dir / (FIFO_NAME + ".tmp")
    for stale in (fifo_path, fifo_tmp):
        try:
            stale.unlink()
        except OSError:
            pass
    os.mkfifo(fifo_tmp, 0o600)
    # Open the read end *before* the path is visible under its published name.
    # A client opens the FIFO `O_WRONLY` to announce a call, and that open
    # blocks until a reader exists -- so a published FIFO that nobody is
    # reading would hang the agent on its first harness call rather than fail
    # over to the socket.
    fifo_r = os.open(fifo_tmp, os.O_RDONLY | os.O_NONBLOCK)
    os.set_blocking(fifo_r, True)
    # And hold a writer of our own for the daemon's lifetime, so the reader
    # never sees EOF when the last client closes and the loop below is one
    # blocking read rather than a reopen-on-EOF spin.
    fifo_w = os.open(fifo_tmp, os.O_WRONLY)
    os.replace(fifo_tmp, fifo_path)

    def _write_reply(path: Path, payload: bytes) -> bool:
        """Write one length-prefixed frame to a caller's reply pipe.

        Non-blocking open, because `O_WRONLY` on a FIFO blocks until a reader
        arrives and the reader here is a client process that may have died
        between announcing its call and opening for the answer. A live client
        is already blocked on that open, so this succeeds immediately; a dead
        one costs `REPLY_OPEN_TIMEOUT_S` in one pool thread instead of wedging
        it for the life of the daemon.
        """
        deadline = time.monotonic() + REPLY_OPEN_TIMEOUT_S
        while True:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                break
            except FileNotFoundError:
                return False
            except OSError as e:
                if e.errno != errno.ENXIO:
                    raise
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.005)
        try:
            view = memoryview(struct.pack(">I", len(payload)) + payload)
            while view:
                # `deadline` bounded the open only; see REPLY_STALL_TIMEOUT_S.
                if not select.select([], [fd], [], REPLY_STALL_TIMEOUT_S)[1]:
                    return False
                try:
                    view = view[os.write(fd, view):]
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        continue
                    if e.errno == errno.EPIPE:
                        return False
                    raise
            return True
        finally:
            os.close(fd)

    def _handle_fifo(ident: str) -> None:
        """One call announced over the FIFO: read payload, dispatch, answer."""
        request = calls_dir / f"{ident}.req"
        reply = calls_dir / f"{ident}.rsp"
        try:
            try:
                req = json.loads(request.read_bytes().decode("utf-8"))
            except (OSError, ValueError) as e:
                resp = {"ok": False, "stdout": "",
                        "error": f"unreadable request {ident!r}: "
                                 f"{type(e).__name__}: {e}"}
            else:
                resp = _dispatch(req)
            try:
                _write_reply(reply, json.dumps(resp, default=str).encode("utf-8"))
            except OSError as e:
                print(f"[daemon] could not answer {ident}: {e}", flush=True)
        finally:
            try:
                request.unlink()
            except OSError:
                pass

    def _fifo_loop() -> None:
        """Read call ids off the shared pipe and fan them out to the pool.

        Only the id travels here; the payload is a file the client renamed
        into place. A pipe write is atomic only to `PIPE_BUF` (4096 bytes) and
        a `blender()` call carrying a page of author Python is bigger than
        that, so putting payloads on this pipe would let two concurrent
        callers interleave halfway through a request.
        """
        buf = b""
        while True:
            try:
                chunk = os.read(fifo_r, 4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                ident = line.decode("ascii", "replace").strip()
                if ident == _FIFO_STOP:
                    return
                if ident:
                    try:
                        pool.submit(_handle_fifo, ident)
                    except RuntimeError:
                        return      # pool shut down under us

    threading.Thread(target=_fifo_loop, name="harness-fifo",
                     daemon=True).start()
    print(f"[daemon] fifo transport at {fifo_path} "
          f"(calls in {calls_dir.name}/)", flush=True)

    try:
        while True:
            conn, _ = srv.accept()
            pool.submit(_handle, conn)
    finally:
        try:
            srv.close()
        finally:
            # Release the FIFO reader, then take both published transports out
            # of the run dir together: a client that finds one of them after
            # the daemon is gone would block on it rather than fail over.
            try:
                os.write(fifo_w, (_FIFO_STOP + "\n").encode("ascii"))
            except OSError:
                pass
            for fd in (fifo_w, fifo_r):
                try:
                    os.close(fd)
                except OSError:
                    pass
            for path in (fifo_path, port_path):
                try:
                    path.unlink()
                except OSError:
                    pass


__all__ = ["serve", "PORT_NAME", "LOOPBACK_HOST", "_StdoutDemux"]
