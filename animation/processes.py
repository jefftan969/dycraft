"""Blender processes for the session: the resident, job workers, and GPU render slots."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re


class GpuOutOfMemory(RuntimeError):
    """A Cycles render ran out of GPU memory, named as such.

    It costs one failed render and kills no process, so the useful reply is
    which other GPU renders were sharing the card when it happened: the author
    can retry with fewer in flight. `error_class` is what `Jobs` and
    `run_interactive` copy onto the failed result.
    """

    error_class = "gpu_out_of_memory"
    failure_class = "harness"

    def __init__(self, error, concurrent):
        self.concurrent = list(concurrent)
        super().__init__(
            "GPU out of memory while "
            + (f"{len(self.concurrent)} other GPU render(s) were running "
               f"({', '.join(map(str, self.concurrent))})" if self.concurrent
               else "no other GPU render of this run was running")
            + ". Retry with fewer concurrent renders, or smaller ones. "
            f"Renderer error: {renderer_line(error)}")


def renderer_line(error):
    """The line of a GPU failure that says it ran out of memory, rather than
    the whole log tail a crashed worker carries (`runtime.log_tail`)."""
    text = f"{error}\n{getattr(error, 'reason', '')}"
    return next((line.strip() for line in text.splitlines()
                 if GPU_OUT_OF_MEMORY.search(line)), str(error))


#: What Cycles prints when a device allocation fails, on the backends the
#: cluster has. Matched against the worker's error and, for a worker that
#: crashed, the end of its log, which carry the renderer's own message
#: (e.g. "Out of memory in CUDA queue enqueue", "Failed to retain CUDA
#: context (Out of memory)").
GPU_OUT_OF_MEMORY = re.compile(
    r"(?is)(?=.*out of (?:gpu )?memory)(?=.*(?:cuda|optix|gpu|hip|cumem))")


class ProcessesMixin:
    @property
    def resident(self):
        """The long-lived Blender bound to this project, started on first use.

        Bound to `native/` rather than to `work/`, so `publish()`'s rename
        happens inside the child's view and the session survives it; `publish`
        calls `reload()` at the one moment the project can have moved.

        Measured against a one-shot worker on a small test fixture: a geometry
        query 251 ms -> 9.5 ms, an edit 251 ms -> 6.9 ms. The gap grows with
        the project, because the one-shot side also re-opens the blend.
        """
        existing = self._resident
        if existing is None:
            from .blender.resident import ResidentBlender
            # The output root is `native/` itself, because that is where job
            # directories already live and a request writes its result into
            # its own one. A narrower root would reject every real call site.
            existing = self._resident = ResidentBlender(
                self.work, self.native,
                source_roots=self.source_roots, bind_root=self.native,
                log=self.native / "resident.log", writable=[self.pair_store])
            self.jobs.on_close.append(self.close_resident)
        return existing

    def close_resident(self):
        session = self._resident
        if session is not None:
            session.close()
            self._resident = None

    def resident_for(self, workspace):
        """The Blender process serving one JOB's snapshot, started on first use.

        The sibling of `resident`, and the two cover disjoint halves of the same
        cost. `resident` serves the live project, which is what makes an
        interactive op cheap. This serves the fan-out *inside* one job: every op
        runs against its own snapshot (`submit` copies `self.work`, `execute`
        renames it to `<job>/workspace`), and inside one op every subcall gets
        that same path: a `storyboard` issues a probe, a render worker per
        shot and a check worker per state against it, dozens of processes in
        one call. Binding one process to that snapshot serves all of them without changing what any of them can reach.

        Keyed by the resolved workspace, so two jobs in flight never share a
        process and the snapshot a job was given is still the only tree its
        workers see.

        `bind_root=None`, unlike `resident`: a job-scoped session never outlives
        the publish that replaces the project, so it does not need the
        parent-bind that lets a process survive a rename.
        """
        from .blender.resident import ResidentBlender
        key = str(Path(workspace).resolve())
        owned = getattr(self._owned, "keys", None)
        if owned is None:
            owned = self._owned.keys = []
        with self._resident_lock:
            existing = self._residents.get(key)
            if existing is not None:
                return existing
            created = self._residents[key] = ResidentBlender(
                workspace, self.native,
                source_roots=self.source_roots,
                log=self.native / "resident.log")
            owned.append(key)
            return created

    def close_residents(self, workspace=None):
        """Shut down the job Blender bound to `workspace`, or every one alive.

        The no-argument form is the job pool's shutdown hook. Idempotent: a job
        that never reached a worker has no process to close, and closing twice
        is a no-op. `resident` — the project-scoped one — is closed separately
        by `close_resident`, because it is not owned by any job.
        """
        keys = ([str(Path(workspace).resolve())] if workspace is not None
                else list(self._residents))
        owned = getattr(self._owned, "keys", None)
        for key in keys:
            session = self._residents.pop(key, None)
            if owned is not None and key in owned:
                owned.remove(key)
            if session is not None:
                session.close()

    def close_job_residents(self):
        """Close every job Blender this thread started, whichever tree it bound.

        A job does not only touch its own snapshot: `initialize` binds
        `<job>/initial` and then renames it away, and a repair `build` runs
        against the workspace an *earlier* job left behind. Closing by the job's
        own workspace would leak those, and closing everything would kill a
        concurrently running job's process — non-mutating ops do run together.
        """
        for key in list(getattr(self._owned, "keys", []) or []):
            self.close_residents(Path(key))

    def worker(self, workspace, output, request, cancel, *, workspace_readonly=False):
        request = {"preserve_render_settings": True,
                   "journal": self.journal.on, **request}
        if not self.gpu_render(request):
            return self._dispatch(workspace, output, request, cancel, workspace_readonly)
        with self.gpu_slot(cancel) as token:
            try:
                return self._dispatch(workspace, output, request, cancel, workspace_readonly)
            except RuntimeError as error:
                if cancel.is_set() or not GPU_OUT_OF_MEMORY.search(
                        f"{error}\n{getattr(error, 'reason', '')}"):
                    raise
                with self._gpu:
                    others = [label.get("job_id") for key, label in self._gpu_holders.items()
                              if key is not token]
                raise GpuOutOfMemory(error, others) from error

    def _dispatch(self, workspace, output, request, cancel, workspace_readonly):
        from .blender.runtime import reads_only, run_worker
        # Three destinations, and the order matters.
        #
        # The live project goes to the project-scoped session: that is the
        # interactive path, and routing on the path rather than on a flag means
        # a caller cannot accidentally send a candidate's work to the session
        # holding the published project.
        if Path(workspace).resolve() == self.work.resolve():
            # Another interactive op may hold the process; the wait is
            # reported on this op's own record rather than looking like work.
            return self.resident.request(request, Path(output), cancel=cancel,
                                         holder=getattr(self._local, "job", None),
                                         on_wait=self.mark_waiting)
        # A job's own snapshot goes to a job-scoped session, but ONLY when the
        # request wants the tree read-write. A resident binds its workspace once
        # at spawn, and `submit()` HARDLINKS the snapshot for every non-mutating
        # op and for `storyboard` — correctness there rests on the bind being
        # read-only, which is what `reads_only` decides and what a resident
        # cannot offer once it is running. So a read-only request keeps forking,
        # where the `--ro-bind` still enforces what the code already does.
        # `check_in_parallel` passes `workspace_readonly=True` for all four of
        # its chunks and so keeps its concurrency, which one process could not
        # give it: `ResidentBlender.request` holds a lock for the exchange.
        #
        # A probe off the live project is read-only too. It evals agent code,
        # which is why `reads_only` leaves it out, but a batch probe's snapshot
        # is hardlinked rather than copied, so a write through a link would land
        # in the live project. The bind refuses the write instead.
        readonly = workspace_readonly or reads_only(request) or request.get("op") == "probe"
        if not readonly:
            return self.resident_for(workspace).request(
                request, Path(output), cancel=cancel)
        return run_worker(workspace, output, request,
                          cancel=cancel, workspace_readonly=readonly,
                          writable=self.store_binds(request))

    #: Concurrent Cycles GPU renders allowed per arm. The arm has one
    #: GPU, assigned outside the harness, and several renders may share it:
    #: the cap bounds the stack, and a render past it waits as `queued`.
    #: Four until stacking throughput on one card is measured; the launcher
    #: can override it.
    GPU_RENDERS = max(1, int(os.environ.get("DYCRAFT_GPU_RENDERS") or 4))

    @staticmethod
    def gpu_render(request):
        """Whether a worker request is a Cycles render that will use the GPU.

        Workbench and the diagnostic default (no `render_settings`) are not:
        they hold no Cycles device memory, which is what has run out.
        """
        if request.get("op") != "render" or request.get("render_settings") is None:
            return False
        from .blender.render_config import final_render_settings
        from .blender.runtime import nvidia_usable
        try:
            settings = final_render_settings(request["render_settings"])
        except ValueError:
            return False      # the worker will refuse it, holding nothing
        return (settings["engine"] == "CYCLES" and settings["compute_backend"] != "CPU"
                and nvidia_usable())

    @contextlib.contextmanager
    def gpu_slot(self, cancel):
        """Hold one of the arm's `GPU_RENDERS` slots for the length of a render.

        The interactive resident's renders take a slot like any job's, since
        they draw on the same card. While waiting, the op's record says so
        (`queued`, `waiting_for: {gpu_renders: [...]}`) and stays cancellable.
        """
        token = object()
        label = dict(getattr(self._local, "job", None) or {})
        waited = False
        with self._gpu:
            while len(self._gpu_holders) >= self.GPU_RENDERS:
                if not waited:
                    waited = True
                    self.mark_waiting({"gpu_renders": [held.get("job_id") for held
                                                       in self._gpu_holders.values()]})
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("job cancelled")
                self._gpu.wait(0.1)
            self._gpu_holders[token] = label
        if waited:
            self.mark_waiting(None)
        try:
            yield token
        finally:
            with self._gpu:
                self._gpu_holders.pop(token, None)
                self._gpu.notify_all()

    def mark_waiting(self, waiting_for):
        """Report this thread's op as queued behind something, or running again."""
        job = getattr(self._local, "job", None)
        if job and job.get("job_id"):
            self.jobs.mark_waiting(job["job_id"], waiting_for)
