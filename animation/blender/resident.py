"""One long-lived sandboxed Blender per project, serving requests.

A fresh `blender -b` per operation costs a fifth of a second before any work
happens -- process start, the bubblewrap setup, and Blender's own imports -- and
opening a large project's blend adds most of a second on top, where the same
edit applied to an already-open project in a warm `bpy` costs milliseconds.

That gap is not academic: an agent that can get the same answer faster by
shelling out to a Blender binary on `PATH` will do so, and work done outside
the harness is invisible to its checks, its delivery, and any later analysis.

The sandbox is unchanged: `runtime.command(serve=True)` builds the same argv
with the same namespaces, the same `--cap-drop ALL`, and the same curated
read-only `/tools`. Residency changes how long the process lives, not what it
can reach.

This module is the process half of that design.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import threading
import time

from .runtime import WorkerError, command, harness_fault, log_tail, worker_env, worker_failure, worker_path


#: A reply is a verdict plus counters. Results still travel through
#: `<output>/result.json`, so this stays small however large the job was.
_MAX_REPLY = 16 * 1024**2


class WorkerGone(RuntimeError):
    """The resident process died, or spoke something this cannot parse."""

    failure_class = "harness"

    def __init__(self, message, reason=None):
        super().__init__(message)
        #: The log's end, kept off the reply (`runtime.log_tail`).
        self.reason = reason or message


class ResidentBlender:
    """A Blender process bound to one project for as long as the caller wants it.

    Not thread-safe by construction and deliberately so: one process has one
    scene, and two concurrent requests against one scene have no meaning. The
    lock serializes callers rather than pretending otherwise, and work that
    genuinely wants parallelism -- chunked compile -- forks separate workers as
    it already does.
    """

    def __init__(self, workspace, output_root, *, readonly=(), source_roots=(),
                 log=None, bind_root=None, writable=()):
        self.workspace = Path(workspace)
        self.bind_root = Path(bind_root) if bind_root else None
        self.output_root = Path(output_root)
        self.readonly = tuple(readonly)
        self.writable = tuple(writable)
        self.source_roots = tuple(str(Path(r)) for r in source_roots)
        self.log_path = Path(log) if log else self.output_root / "resident.log"
        self.lock = threading.RLock()
        self.process = None
        self.log = None
        self.generation = 0
        self.calls = 0
        #: Set by `reload()`; makes the next request re-read the project.
        self.stale = False
        #: Counters the child reports with each reply; see `diagnostics`.
        self.last_reply = {}
        #: Who the caller holding the process said it was (`request(holder=)`),
        #: so a caller queued behind it can say what it is waiting for.
        self.holder = None
        self.started_at = None
        self.output_root.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            self._spawn()

    def _spawn(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("ab")
        reader, writer = os.pipe()
        argv = command(self.workspace, self.output_root, None,
                       self.readonly, serve=True, bind_root=self.bind_root,
                       writable=self.writable)
        try:
            self.process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                pass_fds=(writer,),
                # `worker_env` rather than a literal: a resident process pays
                # the shader cache, the NVIDIA EGL pin, and the thread count
                # once for its whole lifetime, so a second copy of this dict
                # that drifts from `run_worker`'s would only show up as slower
                # renders.
                env={**worker_env(), "DYCRAFT_BLENDER_REPLY_FD": str(writer)})
        finally:
            os.close(writer)
        self.replies = os.fdopen(reader, "rb")
        self.generation += 1
        self.started_at = time.time()
        self.calls = 0
        # The child announces itself before any request, so a spawn that fails
        # -- a missing bind, a Blender that will not start -- is reported here
        # rather than as a confusing failure of whatever op came first.
        hello = self._receive()
        if not hello.get("ready"):
            raise WorkerGone(f"resident Blender did not start: {hello}")

    def reload(self):
        """Forget the in-memory scene without paying for a new process.

        The reload point, and it is a defined moment rather than a race: only a
        publishing op replaces the project, only it holds the edit lock while
        doing so, and it calls this when it is done. The next request re-opens
        the project from disk, which after a publish is the published one --
        the rename happened inside the bound root, so the child can see it.
        """
        with self.lock:
            self.stale = True

    def restart(self):
        """What `fresh=True` means. Re-opening the project is not a clean slate.

        Measured: a reopen clears orphaned datablocks and non-persistent
        handlers but not modified preferences and not `sys.modules`. Only a new
        process clears what actually accumulates, so the escape hatch has to be
        a process, not a reload.
        """
        with self.lock:
            self.close()
            self._spawn()

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is not None and process.poll() is None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (getattr(self, "replies", None), self.log):
                try:
                    stream.close()
                except (AttributeError, OSError, ValueError):
                    pass
            self.replies = self.log = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    # -- wire --------------------------------------------------------------

    def _send(self, value):
        data = pickle.dumps(value, protocol=4)
        try:
            self.process.stdin.write(struct.pack("!Q", len(data)))
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._died(error) from error

    def _receive(self):
        header = self.replies.read(8)
        if len(header) != 8:
            raise self._died("resident Blender closed its reply channel")
        length, = struct.unpack("!Q", header)
        if length > _MAX_REPLY:
            raise WorkerGone("resident Blender reply exceeds 16 MiB")
        body = self.replies.read(length)
        if len(body) != length:
            raise self._died("truncated resident Blender reply")
        return pickle.loads(body)

    def _died(self, reason):
        code = self.process.poll() if self.process else None
        return WorkerGone(f"{reason} (exit {code}); the next call starts a new process. "
                          f"Log: {self.log_path}", reason=log_tail(self.log_path))

    # -- requests ----------------------------------------------------------

    def request(self, job, output, *, fresh=False, cancel=None, holder=None, on_wait=None):
        """Run one job in the resident process and return its result payload.

        `output` must be inside `output_root`: the sandbox binds that root once
        at spin-up, so a directory outside it is simply not reachable from the
        child and would fail as a confusing missing path rather than as the
        programming error it is.

        A request that finds the process busy waits, and that wait is no
        longer invisible: `on_wait(holder)` is called with whatever the
        current holder passed as `holder`, and `on_wait(None)` once the
        process is this caller's. The wait is cancellable -- canceling a
        request that never reached the scene costs nothing.
        """
        output = Path(output)
        try:
            relative = output.resolve().relative_to(self.output_root.resolve())
        except ValueError:
            raise RuntimeError(f"resident output {output} is outside {self.output_root}") from None
        output.mkdir(parents=True, exist_ok=True)
        if not self.lock.acquire(blocking=False):
            if on_wait is not None:
                on_wait(self.holder or {"resident": "busy"})
            while not self.lock.acquire(timeout=0.05):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("job cancelled")
            if on_wait is not None:
                on_wait(None)
        outer, self.holder = self.holder, holder
        try:
            # Canceled before it started costs nothing and must not cost the
            # process: only a request that has already touched the scene has to
            # take the session down with it.
            if cancel is not None and cancel.is_set():
                raise RuntimeError("job cancelled")
            if self.process is None or self.process.poll() is not None:
                self._spawn()
            if fresh:
                # `fresh` is a new process, not a reload: a reopen clears the
                # scene but not preferences, not `sys.modules`, and not
                # accumulated memory, so reloading would answer a weaker
                # question than the caller asked.
                self.close()
                self._spawn()
            self.calls += 1
            self._send({"job": job, "output": worker_path(self.output_root / relative, Path("/output") / relative),
                        "source_roots": self.source_roots, "reopen": self.stale})
            self.stale = False
            reply = self._await(cancel)
            self.last_reply = reply if isinstance(reply, dict) else {}
        finally:
            self.holder = outer
            self.lock.release()
        return self._result(output, reply)

    def _await(self, cancel):
        """Read the reply, honoring cancellation the only way residency can.

        A canceled request has already begun mutating a scene the next request
        will see, and there is no way to unwind it, so canceling kills the
        process. That is more expensive than canceling a one-shot worker and
        it is the honest cost of sharing a scene.
        """
        if cancel is None:
            return self._receive()
        done, box = threading.Event(), {}
        def read():
            try:
                box["reply"] = self._receive()
            except BaseException as error:  # noqa: BLE001 -- re-raised below
                box["error"] = error
            finally:
                done.set()
        thread = threading.Thread(target=read, name="resident-reply", daemon=True)
        thread.start()
        while not done.wait(0.05):
            if cancel.is_set():
                self.close()
                done.wait(5)
                raise RuntimeError("job cancelled")
        if "error" in box:
            raise box["error"]
        return box["reply"]

    def _result(self, output, reply):
        path = output / "result.json"
        if not path.is_file():
            raise self._died(f"resident Blender wrote no result for {output}")
        body = json.loads(path.read_text())
        if not body.get("ok"):
            raise worker_failure(body, path)
        if not reply.get("ok"):
            raise WorkerError(harness_fault(reply.get("error") or "resident Blender request failed",
                                            self.log_path), "harness")
        return body["result"]

    # -- diagnostics -------------------------------------------------------

    @property
    def diagnostics(self):
        """Enough to notice drift rather than guess at it.

        Residency trades a guaranteed-clean process for speed, and the design
        accepts that trade only because the agent can see the state it is
        trading against and call for a restart itself. There is no automatic
        recycle.
        """
        process = self.process
        return {"generation": self.generation, "calls": self.calls,
                "alive": process is not None and process.poll() is None,
                # Reported by the child, because `--unshare-pid` means the pid
                # the harness holds is bubblewrap's, not Blender's.
                "resident_mb": self.last_reply.get("resident_mb"),
                "orphans": self.last_reply.get("orphans"),
                "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None}
