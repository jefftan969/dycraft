"""The animation layer's session: one author over private, revisioned native workspaces."""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import threading
from types import SimpleNamespace

from .blender.bundle import file_hash, skip_sandbox_masks, tree_hash, write_json
from .blender.render_config import SOURCE_SUBDIR
from .blender.journal import Journal, describe
from .blender.runtime import Jobs, failed_result, traced_result
from .processes import ProcessesMixin
from .checking import CheckingMixin
from .storyboard import StoryboardMixin
from .build import BuildMixin
from .commit import CommitMixin
from .jobs import (counts_as_compile, interval_of, register_job_handlers, slim_job_payload, swap_workspace,
                   worker_printed, worker_warnings)
from ..sdk.ir import content_hash
from .context import read_json, workspace_context
from .planning import evaluation_digest, missing_states, shot_interval, validate_plan


class AnimationSession(ProcessesMixin, CheckingMixin, StoryboardMixin, BuildMixin, CommitMixin):
    def __init__(self, session):
        self.session = session
        self.ctx = session.ctx
        self.root = Path(self.ctx.run_dir)
        self.native = self.root / "native"
        self.work = self.native / "work"
        self.lock = threading.RLock()
        self.edit_pending = False
        self.validation_cache = {}
        self.jobs = Jobs(self.native / "jobs", session.trace)
        self.journal = Journal(self.native / "journal")
        #: Per-thread state: each batch job runs in its own thread
        #: (`Jobs.submit`) and each interactive op in its caller's, so one
        #: thread is one op. Holds `inline_record` and the `job` label that
        #: waits report (`mark_waiting`).
        self._local = threading.local()
        self.native.mkdir(parents=True, exist_ok=True)
        self.design = self.root / "design"
        self.design.mkdir(exist_ok=True)
        self.candidates = {}
        self.inline_record = None
        #: When this daemon started: the moment `DYCRAFT_ADVISORY_HOURS` is
        #: counted from (`wall_budget`).
        self.started = time.time()
        self.latest_seq = {}
        #: The edit holding `edit_pending`, so a refused edit can name it.
        self.edit_running = None
        #: Cycles GPU renders in flight on this arm, keyed by a token per
        #: render; see `gpu_slot`.
        self._gpu = threading.Condition()
        self._gpu_holders = {}
        #: Interval checks kept to certify delivery ranges, keyed by
        #: content version and then by interval. Apart from
        #: `validation_cache`'s single whole-film entry, so an interval can
        #: never evict it.
        self.interval_checks = {}
        #: The submission (`_seq`) of the check each cache holds. Checks
        #: overlap, and one submitted on an older content version that finishes
        #: last must not evict the newer one's entry.
        self.cached_seq = {"whole": 0, "interval": 0}
        #: `{state: _seq}` of the last submission that dropped its board. A
        #: storyboard submitted before that drew the state as it was, and its
        #: board must not come back when it merges after the drop.
        self.dropped_at = {}
        #: The film's length as the last edit left it. Only an edit changes it,
        #: and edits publish one at a time, so this is current where
        #: `timeline_hint` -- set by whichever check finished last -- may not
        #: be; a parallel check sized from a stale length checked a lengthened
        #: film only as far as its old end.
        self.film_frames = None
        #: Monotonic project identity for the interactive path; see
        #: `revision_token`. Batch publishes still return a content hash.
        self.revision = 0
        # The launcher's record, then the CLI's. `contribution.json` is what
        # `dispatch()` writes into a child's run dir; `meta.json` carries the
        # role for a child launched any other way.
        contribution = read_json(self.root / "contribution.json", {})
        meta = read_json(self.root / "meta.json", {})
        self.contribution_role = contribution.get("role") or meta.get("contribution_role")
        #: Job-scoped Blender processes, keyed by resolved workspace, and the
        #: per-thread record of which keys this job created. See `resident_for`.
        #: Each job runs in its own thread (`Jobs.submit`), so thread-local
        #: ownership closes exactly the processes a job started -- not a
        #: concurrently running job's, and not one bound to a workspace the job
        #: borrowed rather than owns.
        self._residents = {}
        self._owned = threading.local()
        #: Guards get-or-create only. Deliberately not `self.lock`, which
        #: publish and the edit gate hold; nothing else takes this one, so it
        #: cannot deadlock against them.
        self._resident_lock = threading.Lock()
        self.jobs.on_close.append(self.close_residents)
        # The clip a motion contributor owns, when the launcher scoped it to
        # one. Only `require_storyboards` reads it: everything else about a
        # child is unchanged, and a child with no clip is held to the whole
        # storyboard exactly as its launcher is.
        self.contribution_clip = contribution.get("clip")
        prior = self.root / "prior_animation"
        if prior.exists() and not self.work.exists():
            if not (prior / "design_state/build.json").is_file():
                raise ValueError("refinement needs an animation bundle with its design sources")
            tree_hash(prior)
            shutil.copytree(prior, self.work)
            for path in ("evaluation", "film", "edit_history"):
                shutil.rmtree(self.work / path, ignore_errors=True)
            build = read_json(self.work / "design_state/build.json")
            for rel in build["source_files"]:
                dest = self.design / rel
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.work / "design_state/source" / rel, dest)
        self.work.mkdir(exist_ok=True)
        #: Submission order, shared by both paths. Two things read it: which
        #: check last wrote `latest.json`, and which of two storyboards that
        #: drew the same state wins the board merge. Checks run concurrently,
        #: so finish order is not submission order.
        #: Boards and the delivery receipt keep their submission on disk, so a
        #: restarted daemon, or a run started from a prior project, counts on
        #: from the highest one: numbered from 1, every new storyboard and final
        #: render would lose to a record already there. Read after the prior
        #: project is copied in, for that reason.
        boards = read_json(self.work / "storyboards.json", {}) or {}
        held = [b.get("submitted") or 0 for b in boards.values() if isinstance(b, dict)]
        held.append((read_json(self.native / "delivery.json") or {}).get("submitted") or 0)
        self._seq = itertools.count(max(held) + 1)
        # This layer's source API, staged under the name the prompt uses; the
        # asset section only when there is something to load: the library this
        # checkout links into the run, or an asset the launcher preloaded.
        from ..agent.agent_runner import _PROMPTS_DIR
        reference = (_PROMPTS_DIR / "animation_reference.md").read_text()
        if not ((self.design / "preloaded" / "manifest.json").is_file()
                or (self.root / "references" / "assets" / "INDEX.md").is_file()):
            reference = re.sub(r"(?ms)^## Asset library\n.*?(?=^## )", "", reference)
        (self.root / "source_reference.md").write_text(reference)

    def capture_journal_sources(self, entry):
        """Freeze the source tree this operation is about to read.

        Same contract as `close_journal_entry`: a source root that cannot be
        read must not stop the edit that was going to run against it.
        """
        try:
            entry["src_manifest"] = self.journal.capture_sources(self.source_roots)
        except Exception:                          # noqa: BLE001 - see above
            entry["src_manifest"] = None

    def open_journal_entry(self, operation, args, step, version):
        """Allocate this operation's place in the run's ordered record."""
        return {"seq": self.journal.next_seq(), "step": step,
                "operation": operation, "version_before": version,
                "args": describe(args, self.journal),
                # Always present, so a reader never has to distinguish "this
                # operation captured no source" from "this record predates the
                # field". Only publishing operations fill it in.
                "src_manifest": None}

    def close_journal_entry(self, entry, directory, result, failure, started):
        """Finish this operation's record. Never raises into the operation.

        This runs from the job's `finally`, so an exception here would replace
        whatever the operation was about to return or raise with a logging
        error. The journal is diagnostic: a run must be able to lose its record
        and still deliver a film.
        """
        try:
            self._close_journal_entry(entry, directory, result, failure, started)
        except Exception:                          # noqa: BLE001 - see above
            pass

    def _close_journal_entry(self, entry, directory, result, failure, started):
        entry["job_id"] = directory.name
        entry["duration_s"] = round(time.monotonic() - started, 4)
        entry["ok"] = failure is None
        if failure is not None:
            entry["error"] = str(failure)[:2000]
        if isinstance(result, dict):
            entry["ok"] = bool(result.get("ok", True))
            # Taken from the result the operation already computed. Asking
            # `self.version()` again would re-hash the whole workspace, which
            # measured ~1.3 s on a real project -- 400x this entire record.
            entry["version_after"] = result.get("version")
        # A `build` whose merge raised binding issues keeps its tree as a
        # candidate for the agent to repair, so a digest can describe a scene
        # that was never published. Say which happened rather than leaving a
        # reader to infer it from a version that did not move.
        after = entry.get("version_after")
        entry["published"] = after is not None and after != entry["version_before"]
        entry["digest"] = self.journal.store_digest(entry["seq"], directory)
        self.journal.record(**entry)

    def run_read_only(self, operation, args, workspace, directory, cancel):
        """`inspect` and `probe`: a worker call on the workspace that publishes nothing."""
        if operation == "inspect":
            clean = {k: v for k, v in args.items() if not k.startswith("_")}
            return {"ok": True, **self.worker(workspace, directory / "inspect",
                                              {"op": "inspect", **clean}, cancel)}
        if operation == "probe":
            frame, frames = args.get("frame"), args.get("frames")
            if frame is not None and frames is not None:
                raise ValueError("supply frame or frames, not both")
            result = self.worker(workspace, directory / "probe",
                                 {"op": "probe", "code": args["expression"],
                                  "frame": frame, "frames": frames}, cancel)
            return {"ok": True, "result": result,
                    **({"frames": frames} if frames is not None else {"frame": frame}),
                    **worker_printed(directory / "probe")}
        raise ValueError(f"unsupported operation {operation}")

    @property
    def inline_record(self):
        """The inline job record for the interactive op running on this thread,
        so `publish()` can say when it has passed the point where canceling
        would leave the workspace half swapped.

        Per thread, so two interactive calls in flight at once do not overwrite
        each other's record.
        """
        return getattr(self._local, "inline_record", None)

    @inline_record.setter
    def inline_record(self, value):
        self._local.inline_record = value

    @property
    def source_roots(self):
        """`design/` joins `animation_src/` here: `build` compiles scene.py, and
        an agent rewrites it between calls the same way it rewrites a helper."""
        return [self.root / "animation_src", self.design]

    @property
    def views_root(self):
        """Rendered storyboard panels, deliberately outside the workspace.

        They are evidence *about* a revision, never an input to one: no check
        reads them and no verifier depends on them, so keeping them in the
        workspace only made every op carry them. `render()` already writes
        previews to `previews/<job>/` for the same reason — looking at the
        project must not change `tree_hash(workspace)`.

        Inside the workspace, the panels and their evidence would grow
        `storyboards.json` to hundreds of megabytes in a tree that `submit()`
        copies and `publish()` hashes, slowing every later op. What stays
        versioned is the gate record, a few kB a shot (see
        `planning.gate_record`).

        `finish` copies this into the delivered bundle explicitly, so delivery
        still ships the evidence it always did.
        """
        return self.native / "storyboard_views"

    #: Workspace files outside the content identity: the plan (hashed as its
    #: document instead), the boards, and the siblings `write_json` replaces
    #: them through.
    IDENTITY_EXCLUDED = frozenset({"plan.json", "plan.json.tmp",
                                   "storyboards.json", "storyboards.json.tmp"})

    def identity(self, workspace):
        """`(content, project)`: what the film is made of, and that plus the boards.

        **Content** is the blend, `design_state/`, `animation_src/`, and the
        plan *document*: everything a frame or a verdict depends on. It keys
        the check cache, delivery records, and the receipt, and it is what a
        copying publish compares. The boards are left out
        because a board changes nothing in the film: with them in, every
        `storyboard()` and `accept_storyboard()` would discard the passing
        check and every delivery frame drawn so far.

        **Project** adds the boards. It is the `version` replies report, so
        an acceptance still reads as a new project state.

        The plan's delivery and schedule keys are in neither, because nothing
        reads them to decide what the film contains -- so recording
        `delivery.min_pixels`, which the prompt asks the author to do, rotates
        nothing and discards no check.

        One tree hash for both.
        """
        workspace = Path(workspace)
        plan = read_json(workspace / "plan.json") or {}
        document = {k: v for k, v in plan.items() if k not in self.PLAN_UNEVALUATED}
        content = content_hash({"workspace": tree_hash(workspace, exclude=self.IDENTITY_EXCLUDED),
                                "plan_document": content_hash(document)})
        boards = workspace / "storyboards.json"
        project = content_hash({"content": content,
                                "boards": file_hash(boards) if boards.is_file() else None})
        return content, project

    def content_version(self, workspace):
        return self.identity(workspace)[0]

    def content_parts(self, workspace):
        """What `identity` hashes as content, part by part: the plan document
        and each top-level entry of the workspace. Recorded with a film so a
        refusal can name what changed since it was drawn."""
        from .blender.bundle import excluded
        workspace = Path(workspace)
        plan = read_json(workspace / "plan.json") or {}
        parts = {"plan": content_hash({k: v for k, v in plan.items()
                                       if k not in self.PLAN_UNEVALUATED})}
        for entry in sorted(workspace.iterdir()):
            if entry.name in self.IDENTITY_EXCLUDED or excluded(entry, workspace):
                continue
            if entry.is_dir():
                parts[entry.name + "/"] = tree_hash(entry)
            elif entry.is_file():
                parts[entry.name] = file_hash(entry)
        return parts

    def stale_film(self, receipt, workspace):
        """`finish()`'s refusal when the film is not of this content, naming
        what changed since it was drawn where the receipt can say."""
        message = ("Render the current project with final_render() before finish(): "
                   "the film must be drawn from this project's content.")
        if not receipt:
            return message + " No film has been drawn yet."
        drawn = receipt.get("content_parts")
        if not isinstance(drawn, dict):
            return message
        now = self.content_parts(workspace)
        changed = sorted(name for name in set(drawn) | set(now) if drawn.get(name) != now.get(name))
        return message + (f" Changed since the film was drawn: {', '.join(changed)}." if changed else "")

    def project_hash(self, workspace):
        return self.identity(workspace)[1]

    def version(self):
        return self.project_hash(self.work)

    #: Plan keys that describe how the film is delivered rather than what it
    #: contains. A call carrying only these publishes nothing: it takes no
    #: edit lock and leaves the validated plan document alone. Recording
    #: `delivery.min_pixels` is the metadata update the prompt itself asks
    #: for, and it must not rotate the project version, take the lock a
    #: running edit holds, or discard the validation cache.
    PLAN_METADATA = frozenset({"delivery", "schedule", "cadence", "notes"})
    #: Plan keys outside the content identity: the metadata, the prose nothing
    #: evaluates, and the plan's `id`, which hashes all of it. Rewording
    #: `intent` or a criterion is still an edit, but it does not discard the
    #: checks and the delivery receipt.
    PLAN_UNEVALUATED = PLAN_METADATA | {"intent", "criteria", "id"}

    #: Where `animation_src/` is readable inside the workspace. Defined in
    #: `render_config`, which the worker imports too: it puts this directory
    #: on `sys.path`, and the two must name the same one.
    SOURCE_SUBDIR = SOURCE_SUBDIR

    #: Workspace names the harness owns. An author file of the same name used
    #: to replace one silently, because the source tree is copied onto the
    #: workspace root.
    HARNESS_OWNED = frozenset({"plan.json", "storyboards.json", "animation.blend",
                               "animation.blend1", "design_state", "evaluation"})

    #: Ops that change the project when they succeed. A committed project
    #: refuses them (bar `finish`, which answers that it already committed).
    #: `storyboard` and `accept_storyboard` change only the boards, and
    #: publish by merging them state by state (`merge_boards`). `final_render`
    #: is not here: it writes nothing into the project.
    PUBLISHING = frozenset({"plan", "build", "blender", "storyboard",
                            "accept_storyboard", "finish"})

    #: The ops that exclude one another, and the only ones that hold
    #: `edit_pending`. Each takes 1-3 minutes, and two edits to the
    #: blend cannot both stand. Gate calls and renders run alongside them:
    #: they do not save the blend, so holding the guard through them would
    #: only block edits.
    EDITS = frozenset({"plan", "build", "blender"})

    #: Ops that publish by building a candidate tree and renaming it in, at
    #: every setting, rather than editing the project where it stands.
    #:
    #: `build` because its dominant workflow *is* the unpublished tree: it
    #: hands back a candidate id and a blend path for review. `finish` because
    #: the delivery receipt records a content hash of the project the film was
    #: drawn from, and an in-place op has no such moment to record -- its
    #: identity is a counter. (`final_render` also runs on a pinned snapshot,
    #: for the same reason, but publishes nothing; see `PINNED`.)
    #:
    #: Everything else edits the project in place. Failure isolation survives
    #: because the worker saves only after the author's code returns, so a
    #: raising edit leaves `animation.blend` exactly as it was, and the
    #: resident session re-reads it rather than carrying a half-applied scene.
    #: The one consequence: an author whose own code calls `save_as_mainfile`
    #: and then raises keeps that write.
    COPY_TO_PUBLISH = frozenset({"build", "finish"})

    #: Ops that run on a hardlinked snapshot pinned to one content version
    #: even interactively, and publish nothing. `final_render` because its
    #: receipt names the version its frames belong to; a delivery `render`
    #: is the same case and is recognized by its argument (`pinned`).
    PINNED = frozenset({"final_render"})

    @classmethod
    def pinned(cls, operation, args):
        return operation in cls.PINNED or (
            operation in ("render", "snapshot") and bool(args.get("delivery")))

    #: Where there is no interactive story at any setting, so the handle is the
    #: better default. Each still accepts `batch=False`; the agent simply waits.
    BATCH_BY_DEFAULT = frozenset({"compile", "final_render", "finish", "review",
                                  "ik"})

    @classmethod
    def plan_metadata_only(cls, operation, args):
        """A `plan()` that declares no shots and no states, only how to deliver."""
        value = args.get("plan")
        return bool(operation == "plan" and isinstance(value, dict) and value
                    and set(value) <= cls.PLAN_METADATA)

    def revision_token(self):
        """Project identity on the interactive path.

        Deliberately not a content hash. Hashing the workspace is what a
        mutating op pays three times today, and on one recorded project
        that is 2.14 s -- which would dominate an operation whose whole point
        is to cost milliseconds. A counter is sufficient for what an
        interactive result and its trace record need, which is to say *which*
        state of this run they refer to.
        """
        return f"r{self.revision}"

    def submit(self, operation, batch=None, fresh=False, **args):
        """Run `operation`, either in this process or as a job.

        `batch=False` runs it here and returns its result; `batch=True` returns
        a handle to poll. The agent chooses: it is single-threaded and already
        blocked on whatever it asked for, so a handle only earns its keep when
        the agent has other work to overlap. The harness never inspects a
        payload to guess.

        `fresh=True` restarts the resident Blender before the call. A reopen is
        not a clean slate -- it clears the scene but not preferences, not
        `sys.modules` and not accumulated memory -- so the escape hatch has to
        be a process.
        """
        if batch is None:
            # A delivery render draws the film at delivery settings, which takes
            # tens of minutes. Nothing about it is a look, so it defaults to a
            # handle for the same reason `compile` does.
            batch = operation in self.BATCH_BY_DEFAULT or (
                operation in ("render", "snapshot") and bool(args.get("delivery")))
        if not batch or self.plan_metadata_only(operation, args):
            # A metadata-only plan writes one file and publishes nothing, so
            # there is no candidate to poll and nothing a handle would buy.
            return self.run_interactive(operation, args, fresh=fresh)
        step = self.session.current_op_step()
        mutates = operation in self.PUBLISHING
        edits = operation in self.EDITS
        claim = None
        with self.lock:
            # Narrowed with the interactive path: see `run_interactive`.
            if edits and self.edit_pending:
                return self.edit_refusal()
            if self.ctx.committed and mutates and operation != "finish":
                return {"ok": False, "failure_class": "rejection",
                        "error": self.COMMITTED}
            seq = next(self._seq)
            content, version = self.identity(self.work)
            capture = Path(tempfile.mkdtemp(prefix="snapshot_", dir=self.native))
            source_capture = None
            try:
                # An op that never writes the workspace does not need a byte
                # copy of it -- hardlinks give the same frozen view for free,
                # where a copy costs time per op and disk for every retained
                # job directory.
                #
                # Safe because nothing writes through the links: `run_worker`
                # binds /workspace read-only for `render`, `inspect`,
                # `evaluate` and, off the live project, `probe` (see
                # `_dispatch`); and publishing ops rename whole directories so
                # a later edit lands on a fresh inode rather than mutating one
                # an earlier snapshot still points at.
                #
                # `storyboard` and `accept_storyboard` publish and are still
                # linkable, and storyboard, called often against a large
                # workspace, is the op this matters most for. The only thing
                # either writes into the workspace is
                # `storyboards.json`, through `write_json`'s sibling `.tmp` and
                # `replace()`, so a link is broken by a fresh inode rather than
                # written through; and both publish by merging boards into the
                # live project, never by renaming this tree in. Their renders
                # and checks write under the job directory.
                link = not mutates or operation in ("storyboard", "accept_storyboard")
                shutil.copytree(self.work, capture, dirs_exist_ok=True,
                                **({"copy_function": os.link} if link else {}))
                if operation == "build" and not args.get("candidate"):
                    source_capture = Path(tempfile.mkdtemp(prefix="sources_", dir=self.native))
                    tree_hash(self.design)  # Reject symlink escapes before copying.
                    shutil.copytree(self.design, source_capture, dirs_exist_ok=True,
                                    ignore=skip_sandbox_masks)
            except BaseException:
                shutil.rmtree(capture, ignore_errors=True)
                if source_capture:
                    shutil.rmtree(source_capture, ignore_errors=True)
                raise
            if edits:
                # OR'd, never assigned: assigning it would let any batch look
                # -- a `check_motion`, `compile`, `snapshot`, or `render` --
                # clear the guard a running edit held.
                claim = self.claim_edit(operation, step)
        entry = self.open_journal_entry(operation, args, step, version)

        def execute(directory, cancel):
            started = time.monotonic()
            self._local.job = {"job_id": directory.name, "operation": operation}
            # In the job thread, immediately before the workspace is set up, so
            # the manifest names the source this call read rather than the
            # source it was submitted against.
            if mutates:
                self.capture_journal_sources(entry)
            workspace = directory / "workspace"
            result = caught = None
            try:
                capture.rename(workspace)
                request = {**args, "_version": version, "_content": content, "_seq": seq,
                           **({"_fresh": True} if fresh else {})}
                if source_capture:
                    request["_source"] = source_capture
                # No session-wide lock: `latest.json` is ordered by
                # submission below, and the caches it shares are keyed or
                # validated, so a race costs a miss, where serializing every
                # check behind one lock would queue them invisibly.
                result = self.execute(operation, request, workspace, directory, cancel, step)
                return result
            except BaseException as error:
                caught = error
                raise
            finally:
                self.close_journal_entry(entry, directory, result, caught, started)
                # Success, failure, or cancellation: every job-scoped process
                # this thread started goes with it. See `close_job_residents`.
                self.close_job_residents()
                self._local.job = None
                shutil.rmtree(capture, ignore_errors=True)
                if source_capture:
                    shutil.rmtree(source_capture, ignore_errors=True)
                if claim is not None:
                    self.release_edit(claim)
        try:
            record = self.jobs.submit(operation, execute, step=step, version=version)
        except BaseException:
            if claim is not None:
                self.release_edit(claim)
            shutil.rmtree(capture, ignore_errors=True)
            if source_capture:
                shutil.rmtree(source_capture, ignore_errors=True)
            raise
        if claim is not None:
            claim["job_id"] = record["job_id"]
        return record

    def claim_edit(self, operation, step):
        """Take `edit_pending` for an edit; the caller holds `self.lock`."""
        claim = {"operation": operation, "step": step}
        self.edit_pending, self.edit_running = True, claim
        return claim

    def release_edit(self, claim):
        with self.lock:
            if self.edit_running is claim:
                self.edit_pending, self.edit_running = False, None

    def edit_refusal(self):
        """Refuse a second edit, naming the one in flight."""
        running = self.edit_running or {}
        what = running.get("operation") or "an edit"
        if running.get("job_id"):
            what += f", job {running['job_id']}"
        return {"ok": False, "failure_class": "rejection",
                "error": (f"A project edit is pending ({what}); poll it before submitting "
                          "another edit. Only plan, build, and blender exclude each other: "
                          "storyboards, acceptances, checks, and renders run alongside it.")}

    def run_interactive(self, operation, args, *, fresh=False):
        """Execute here and now, against the project itself, and return the result.

        No job handle, no polling, and -- for everything outside
        `COPY_TO_PUBLISH` and `pinned` -- no snapshot of the workspace either,
        which would cost time before every op and disk for every one retained.

        An edit takes the edit guard here as a batch one does, so the
        invariant is the same whichever path an op took: at most one edit is
        in flight, and a second is refused rather than racing it. Relying on
        the agent being single-threaded would have made the guarantee true by
        accident, and it is not the only caller -- the daemon and a dispatched
        contributor reach this object too.
        """
        import threading
        import uuid

        step = self.session.current_op_step()
        metadata = self.plan_metadata_only(operation, args)
        publishes = operation in self.PUBLISHING and not metadata
        edits = operation in self.EDITS and not metadata
        claim = None
        with self.lock:
            # Only an edit waits for an edit. A read, a check, a storyboard, or
            # a render running alongside one costs nothing, and blocking them
            # would leave the agent unable even to `probe` while a long
            # storyboard ran.
            #
            # Whatever lands while a candidate is in flight, `publish()` keeps
            # it: board merges are carried across, and only a change to the
            # project's content outside the harness refuses the candidate.
            if edits and self.edit_pending:
                return self.edit_refusal()
            if self.ctx.committed and publishes and operation != "finish":
                return {"ok": False, "failure_class": "rejection",
                        "error": self.COMMITTED}
            seq = next(self._seq)
            # Claimed inside the same lock acquisition that checked it, so two
            # edits cannot both see it free. Released in the `finally` below,
            # on every path including a raised setup error.
            if edits:
                claim = self.claim_edit(operation, step)
        directory = self.native / "jobs" / uuid.uuid4().hex[:16]
        directory.mkdir(parents=True, exist_ok=True)
        if claim is not None:
            claim["job_id"] = directory.name
        cancel = threading.Event()
        # A delivery render writes the accepted evaluation into the workspace
        # for the replay check to read, so it cannot run against the live
        # project the way a look can -- that would move the revision the
        # frames are about to be pinned to. `final_render` is the same case.
        copies = operation in self.COPY_TO_PUBLISH or self.pinned(operation, args)
        # A copying op is version-pinned the way a batch one is, because it
        # really can be overtaken; an in-place op cannot be, so it carries the
        # cheap token instead.
        content = None
        if copies:
            content, version = self.identity(self.work)
        else:
            version = self.revision_token()
        # Setting the call up is not running it, and the two fail differently.
        # A symlink escaping the source tree, or a workspace that cannot be
        # copied, is a bad request: it raises here exactly as the batch path
        # raises from `submit`, so the same input fails the same way whichever
        # path it took. Only an operation that started and then failed comes
        # back as a verdict.
        workspace, source_capture = self.work, None
        try:
            if copies:
                workspace = directory / "workspace"
                # Links for a render, bytes for a publish -- `submit()`'s
                # argument exactly: a render never writes through the
                # workspace it reads, and a 95 MB blend copied per delivery
                # range is the cost of drawing them separately at all.
                link = self.pinned(operation, args)
                shutil.copytree(self.work, workspace, dirs_exist_ok=True,
                                **({"copy_function": os.link} if link else {}))
                if operation == "build" and not args.get("candidate"):
                    source_capture = Path(tempfile.mkdtemp(prefix="sources_", dir=self.native))
                    tree_hash(self.design)  # Reject symlink escapes before copying.
                    shutil.copytree(self.design, source_capture, dirs_exist_ok=True,
                                    ignore=skip_sandbox_masks)
        except BaseException:
            if source_capture:
                shutil.rmtree(source_capture, ignore_errors=True)
            shutil.rmtree(directory, ignore_errors=True)
            if claim is not None:
                self.release_edit(claim)
            raise
        entry = self.open_journal_entry(operation, args, step, version)
        started = time.monotonic()
        result = caught = None
        try:
            # Registered for the whole call, so `job_status` and `job_cancel`
            # take the id the harness has already published on disk; otherwise
            # the only way to stop one render would be killing the resident
            # Blender.
            with self.jobs.register_inline(directory.name, operation, cancel,
                                           step=step, version=version) as record:
                self.inline_record = record
                self._local.job = {"job_id": directory.name, "operation": operation}
                if publishes:
                    self.capture_journal_sources(entry)
                if fresh:
                    self.resident.restart()
                request = {**args, "_version": version, "_seq": seq,
                           **({"_content": content} if content else {}),
                           **({"_fresh": True} if fresh else {})}
                if source_capture:
                    request["_source"] = source_capture
                # No session-wide check lock; see `submit`.
                result = self.execute(operation, request, workspace, directory, cancel, step)
                if isinstance(result, dict):
                    # `status` as well as the id: the harness client waits on
                    # any reply carrying a `job_id` whose status is not
                    # terminal, and this one is already finished.
                    result = {**result, "job_id": directory.name, "status": "completed"}
                # On the record too, so the id answers `job_result` after the
                # call has returned.
                record["result"] = result
                return result
        except BaseException as error:
            caught = error
            # The same shape a polled job hands back for a failure, so an agent
            # reading a result does not have to care which path produced it.
            result = {**failed_result(error, directory, cancel.is_set()),
                      "version": version, "job_id": directory.name,
                      "status": "cancelled" if cancel.is_set() else "failed"}
            if self.inline_record is not None:
                self.inline_record["result"] = result
            return result
        finally:
            self.inline_record = None
            self._local.job = None
            if claim is not None:
                self.release_edit(claim)
            self.close_journal_entry(entry, directory, result, caught, started)
            self.close_job_residents()
            # Interactive ops write the same trace records as jobs do. Run
            # forensics is how this layer earns the right to be in the loop,
            # and an op that is fast but invisible is the failure the whole
            # exercise exists to avoid.
            #
            # The job id at the top level and the same result the caller got,
            # on both paths, so every job directory can be found from the
            # trace.
            self.session.trace("job_result", operation=operation, job_id=directory.name,
                               status=("completed" if caught is None
                                       else "cancelled" if cancel.is_set() else "failed"),
                               version=version, interactive=True, submission_step=step,
                               result=traced_result(result, directory))
            if source_capture:
                shutil.rmtree(source_capture, ignore_errors=True)

    def publish(self, workspace, args, cancel, *, drop_boards=()):
        """Make an edit's result the project. `args` carries what the op captured.

        `drop_boards` are the states the edit invalidated. They are applied to
        the live boards rather than to the candidate's copy, because the
        boards are not this edit's to own: storyboards and acceptances run
        alongside edits now and merge their boards into the live
        project while the edit is in flight.
        """
        if Path(workspace).resolve() == self.work.resolve():
            # In place: the op edited the project itself, so there is nothing
            # to swap in and no candidate that could have gone stale. Identity
            # still advances, because a result and its trace record have to say
            # which state of the project they describe.
            with self.lock:
                if cancel.is_set():
                    raise RuntimeError("job cancelled")
                if self.ctx.committed:
                    raise ValueError(self.COMMITTED)
                self.commit_inline()
                self.note_drops(drop_boards, args)
                if drop_boards:
                    boards = read_json(self.work / "storyboards.json", {})
                    write_json(self.work / "storyboards.json",
                               {k: v for k, v in boards.items() if k not in set(drop_boards)})
                self.revision += 1
                return self.revision_token()
        # Before the rename, not after: this workspace is about to become the
        # project, and a job-scoped process still holding it open would be
        # serving a tree that no longer has that name. The project-scoped
        # `resident` is the opposite case -- it is bound to `native/` precisely
        # so it survives this, and `reload()` below is its moment.
        self.close_residents(workspace)
        with self.lock:
            if cancel.is_set():
                raise RuntimeError("job cancelled")
            if self.ctx.committed:
                raise ValueError(self.COMMITTED)
            # Content, not the whole project. Edits exclude each other, so the
            # only things that can land while this candidate runs are board
            # merges and metadata-only plans, and both are carried across
            # below. Anything else moving the content was a write from outside
            # the harness. (This once compared the full project hash,
            # and an author's in-place helper edit -- hardlinked into the
            # project -- discarded 45 recorded candidates.)
            if self.content_version(self.work) != args.get("_content"):
                raise ValueError(
                    "the project changed outside the harness while this candidate was running "
                    "(for example, a direct write into native/work); retry against the "
                    "current project")
            # A metadata-only `plan()` neither publishes nor holds the edit
            # lock, so one can land while this candidate runs. The rename
            # would discard it, and it is not part of the identity this
            # candidate was checked against -- so carry it across.
            live = read_json(self.work / "plan.json") or {}
            carried = {k: v for k, v in live.items() if k in self.PLAN_METADATA}
            if carried:
                plan = read_json(workspace / "plan.json") or {}
                if any(plan.get(k) != v for k, v in carried.items()):
                    write_json(workspace / "plan.json", {**plan, **carried})
            # The live boards, not the candidate's copy of them, for the same
            # reason: a storyboard or an acceptance may have merged since.
            boards = read_json(self.work / "storyboards.json")
            if boards is not None:
                write_json(workspace / "storyboards.json",
                           {k: v for k, v in boards.items() if k not in set(drop_boards)})
            self.note_drops(drop_boards, args)
            self.commit_inline()
            swap_workspace(self.work, workspace, self.native)
            # The reload point. This is the only place the project is replaced,
            # and it holds the lock while doing it, so a resident session can
            # be told to re-read here rather than having to guess whether the
            # tree it has open is still the published one.
            session = self._resident
            if session is not None:
                session.reload()
            # Kept monotonic across both paths so an interactive token issued
            # after a batch publish cannot repeat one issued before it.
            self.revision += 1
            return self.version()

    def note_drops(self, names, args):
        """Remember which submission dropped these boards; see `dropped_at`."""
        seq = args.get("_seq", 0)
        for name in names or ():
            self.dropped_at[name] = max(seq, self.dropped_at.get(name, 0))

    def merge_boards(self, workspace, args, cancel, *, boards=None, drop=(), reviews=None,
                     panels=None):
        """Publish a gate op's boards onto the live project, one state at a time.

        `storyboard` and `accept_storyboard` change exactly one file,
        `storyboards.json`: panels are written outside the project, and the
        render they use does not save the blend. So neither renames a tree in
        and neither checks a version. Each finishes on the version it
        started from and merges what it decided into whatever the boards are
        now:

        * `boards` -- states this call drew or re-checked. Where two calls
          drew the same state, the later *submission* wins (`submitted`),
          whichever finished last. A review the live board carries is kept
          when the new board shows the same picture of the same state.
        * `drop` -- boards this call removed (states the plan no longer has).
        * `panels` -- `{state: image}` this call drew, copied to the board's
          `path` for each board that wins, so the file and the board agree.
        * `reviews` -- acceptances, `{state: {review, image_digest}}`. Applied
          only to the panel they were written about: a state redrawn since,
          or failing now, is reported in `not_applied` and keeps nothing.

        Boards are outside the content identity, so a merge never
        invalidates a check or a delivery frame.
        """
        in_place = Path(workspace).resolve() == self.work.resolve()
        seq = args.get("_seq", 0)
        superseded, applied, not_applied = [], [], {}
        with self.lock:
            if cancel.is_set():
                raise RuntimeError("job cancelled")
            if self.ctx.committed:
                raise ValueError(self.COMMITTED)
            self.commit_inline()
            live = read_json(self.work / "storyboards.json", {})
            for name in drop:
                if (live.get(name) or {}).get("submitted", 0) <= seq:
                    live.pop(name, None)
            self.note_drops(drop, args)
            for name, board in (boards or {}).items():
                current = live.get(name)
                if ((current and current.get("submitted", 0) > seq)
                        or self.dropped_at.get(name, 0) > seq):
                    superseded.append(name)
                    continue
                if (current and current.get("reviewed") and not board.get("reviewed")
                        and current.get("image_digest") == board.get("image_digest")
                        and current.get("state_revision") == board.get("state_revision")):
                    board = {**board, "reviewed": True, "review": current.get("review"),
                             "retained_from": current.get("retained_from") or current.get("design_id")}
                if name in (panels or {}):
                    (self.native / board["path"]).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(panels[name], self.native / board["path"])
                live[name] = board
            for name, review in (reviews or {}).items():
                current = live.get(name)
                if current is None:
                    not_applied[name] = "its board was removed"
                elif current.get("image_digest") != review["image_digest"]:
                    not_applied[name] = "it was redrawn since; review the new panel"
                elif current.get("checks_pass") is False:
                    not_applied[name] = "it now fails its checks"
                else:
                    current.update(reviewed=True, review=review["review"])
                    applied.append(name)
            write_json(self.work / "storyboards.json", live)
            self.revision += 1
            content = None
            if in_place:
                version = self.revision_token()
            else:
                content, version = self.identity(self.work)
        return {"version": version, "content": content, "boards": live,
                "superseded": sorted(superseded), "applied": sorted(applied),
                "not_applied": not_applied}

    def commit_inline(self):
        """Past here an inline op is swapping the project, and `job_cancel`
        refuses rather than flagging it: stopping mid-swap is what leaves the
        workspace renamed aside with nothing renamed back."""
        if self.inline_record is not None:
            self.inline_record["publishing_committed"] = True

    def wall_budget(self):
        """Elapsed and remaining seconds against the run's advisory budget, and
        `stop_in_s`, the seconds until the launcher stops the run.

        The run clock every check reply carries, so an author pricing a render
        has a denominator to price it against. `None` when the run declares no
        advisory budget (unset, unreadable, not finite, or zero or less, which is
        how a launch says "no budget"), rather than a made-up one. `stop_in_s`
        comes from the launcher's hard limit (`DYCRAFT_DEADLINE`, epoch seconds,
        `inf` for none); it is absent without a finite limit, and with no
        advisory budget, since it rides in this block.

        Counted from this daemon's start. A cold start's daemon starts as the
        session does; a resume starts a new one with the budget that is left.
        """
        hours = os.environ.get("DYCRAFT_ADVISORY_HOURS")
        if not hours:
            return None
        try:
            total = float(hours) * 3600.0
        except ValueError:
            return None
        if not (math.isfinite(total) and total > 0):
            return None
        now = time.time()
        elapsed = now - self.started
        budget = {"elapsed_s": round(elapsed), "remaining_s": round(total - elapsed),
                  "advisory_hours": float(hours)}
        try:
            deadline = float(os.environ.get("DYCRAFT_DEADLINE") or "inf")
        except ValueError:
            deadline = float("inf")
        if math.isfinite(deadline):
            budget["stop_in_s"] = max(0, round(deadline - now))
        return budget

    @staticmethod
    def image_digest(path):
        """Pixels, not file bytes: PNG metadata is not part of the evidence."""
        import hashlib
        from PIL import Image
        with Image.open(path) as image:
            return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()

    @staticmethod
    def plan_of(workspace):
        plan = read_json(workspace / "plan.json")
        if not plan:
            raise ValueError("Register the introductory plan with plan() first.")
        return plan

    @staticmethod
    def build_of(workspace):
        build = read_json(workspace / "design_state/build.json")
        if not build or not (workspace / "animation.blend").is_file():
            raise ValueError("Write design/scene.py and call build() first.")
        return build

    def resolve_shot(self, workspace, args):
        """Expand `shot='id'` into the interval that shot declared in the plan.

        Resolution only. The harness reads the interval an author registered;
        it does not create timeline markers, schedule cuts, or check that the
        blend agrees with the plan. Expanded here so everything downstream —
        including whether a `compile` counts as the run's verdict — sees an
        ordinary interval and cannot treat a shot-scoped call as a full one.
        """
        if args.get("shot") is None:
            return {k: v for k, v in args.items() if k != "shot"}
        if any(args.get(k) is not None for k in ("start", "end", "seconds", "clip")):
            raise ValueError("supply shot=, clip= or an explicit interval, not two of them")
        start, end = shot_interval(self.plan_of(workspace), args["shot"])
        return {**{k: v for k, v in args.items() if k != "shot"},
                "start": start, "end": end}

    def source_fresh(self, workspace):
        if tree_hash(self.design) != self.build_of(workspace)["source_hash"]:
            running = self.edit_running or {}
            if running.get("operation") == "build":
                # Said even while that build was running, so authors started a
                # second one rather than wait for the first.
                job = f" (job {running['job_id']})" if running.get("job_id") else ""
                raise ValueError(
                    f"design/ changed since the last build read it, and a build is running{job}; "
                    "wait for it, and run h.build() again only if design/ changed after it started")
            raise ValueError("design/ changed since build() read it; run h.build() again")

    #: What the latest evaluation says about the film, kept for planning the
    #: next check and for placing states: `state_frames` arrives with
    #: `evaluate(states=)` and is used wherever it is present; `render`
    #: is the recipe whose aspect a bare `final_render()` keeps.
    #: The refusal for any change once `finish()` has committed the project.
    COMMITTED = "The project is committed; nothing changes after finish()."

    TIMELINE_KEYS = ("n_frames", "fps", "declarations", "requires_preroll",
                     "state_frames", "render", "resolution")
    #: `TIMELINE_KEYS` from the latest evaluation, or None before the first.
    timeline_hint = None
    #: The newest whole-film check's evaluation, or None before one finished.
    last_whole = None
    #: The project's resident Blender, started on first use (`resident`).
    _resident = None

    #: The method that runs each operation, all called as `(operation, args, workspace, directory, cancel, step)`.
    OPERATIONS = {
        "plan": "op_plan",
        "build": "op_build",
        "blender": "op_blender",
        "probe": "op_look",
        "inspect": "op_look",
        "motion_image": "op_look",
        "ik": "op_ik",
        "storyboard": "op_storyboard",
        "accept_storyboard": "op_accept_storyboard",
        "review": "op_review",
        "compile": "op_check",
        "check_motion": "op_check",
        "render": "op_render",
        "snapshot": "op_render",
        "final_render": "op_final_render",
        "finish": "op_finish",
    }

    def execute(self, operation, args, workspace, directory, cancel, step):
        """Run `operation` on `workspace` through its method in `OPERATIONS`."""
        method = self.OPERATIONS.get(operation)
        if method is None:
            raise ValueError("unsupported operation: " + operation)
        if operation in ("compile", "check_motion", "render", "snapshot"):
            args = self.resolve_shot(workspace, args)
        return getattr(self, method)(operation, args, workspace, directory, cancel, step)

    def op_plan(self, operation, args, workspace, directory, cancel, step):
        if self.plan_metadata_only(operation, args):
            # Merged, never replaced: a metadata call must not drop the
            # plan document, and `validate_plan` would refuse it anyway.
            plan = read_json(workspace / "plan.json") or {}
            plan.update(args["plan"])
            write_json(workspace / "plan.json", plan)
            return {"ok": True, "plan_id": plan.get("id"),
                    "updated": sorted(args["plan"]),
                    "version": self.revision_token()}
        plan = validate_plan(args["plan"])
        # A plan document replaces the plan, but not how the film is to be
        # delivered: dropping `delivery` on a re-plan would silently take
        # a measured pixel budget back to the default.
        previous_plan = read_json(workspace / "plan.json") or {}
        kept = sorted(k for k in self.PLAN_METADATA
                      if k in previous_plan and k not in args["plan"])
        plan.update({k: previous_plan[k] for k in kept})
        write_json(workspace / "plan.json", plan)
        version = self.publish(workspace, args, cancel)
        # Not the plan back. `validate_plan` adds an `id` and returns
        # everything else verbatim, so echoing it would hand the author a
        # large copy of what they just sent. What they cannot already
        # know is the identity it registered under and the shot ids other
        # tools now accept; the stored copy is `plan.json` in the
        # workspace.
        return {"ok": True, "plan_id": plan["id"],
                "shots": [shot["id"] for shot in plan["shots"]],
                **({"kept": kept} if kept else {}),
                "version": version}

    def op_build(self, operation, args, workspace, directory, cancel, step):
        return self.build(args, workspace, directory, cancel)

    def op_blender(self, operation, args, workspace, directory, cancel, step):
        self.build_of(workspace)
        stage = args.get("stage", "animation")
        if stage not in ("storyboard", "animation"):
            raise ValueError("stage must be 'storyboard' or 'animation'")
        span = None
        if stage == "animation":
            span = self.require_span(workspace, args.get("span"), args.get("shot"),
                                     directory=directory, cancel=cancel)
        elif args.get("span") is not None or args.get("shot") is not None:
            raise ValueError("span= and shot= scope an animation edit; "
                             "pass them with stage='animation'")
        sources = self.root / "animation_src"
        if sources.exists():
            tree_hash(sources)
            self.stage_sources(sources, workspace)
        boards = read_json(workspace / "storyboards.json", {})
        plan = self.plan_of(workspace)
        # Resolved before the edit runs, not after it: a raise after the
        # worker applied the code would discard the whole publish, and on the
        # interactive path, where `blender` edits the project in place, leave
        # the live workspace mutated by an op reporting failure.
        invalidated = self.resolve_invalidations(plan, boards, args.get("invalidate_shots"))
        (directory / "source.py").write_text(args["code"])
        result = self.worker(workspace, directory / "edit", {"op": "edit", "code": args["code"], "name": args.get("name")}, cancel)
        # The worker reports what the edit's animation reaches beside the
        # author's result, never inside it.
        changed = read_json(directory / "edit/animation_changed.json")
        reported = (self.span_report(span, workspace, directory, cancel,
                                     {"animation_changed": changed} if changed else result, boards)
                    if span is not None else None)
        version = self.publish(workspace, args, cancel, drop_boards=invalidated)
        if changed and changed.get("n_frames"):
            self.film_frames = changed["n_frames"]
        payload = {"ok": True, "result": result, "version": version,
                   "blend": str(self.work / "animation.blend"),
                   **({"span": reported} if reported is not None else {}),
                   **({"invalidated": sorted(invalidated)}
                      if args.get("invalidate_shots") else {})}
        return {**payload, **worker_warnings(directory / "edit"),
                **worker_printed(directory / "edit")}

    def op_look(self, operation, args, workspace, directory, cancel, step):
        self.build_of(workspace)
        if operation == "motion_image":
            from .visual import motion_image
            return motion_image(self, args, workspace, directory, cancel)
        clean = {k: v for k, v in args.items() if not k.startswith("_")}
        return self.run_read_only(operation, clean, workspace, directory, cancel)

    def op_ik(self, operation, args, workspace, directory, cancel, step):
        from .blender.solvers import consult, requested_frames
        if requested_frames(args):
            # Only an export reads the project; a script that builds its
            # own robot and world does not wait on a build.
            self.build_of(workspace)
        ctx = workspace_context(workspace)
        # Existing solver exports bind to this immutable design and pose;
        # the agent's own `ik/` belongs to the run, not to the design.
        proxy = SimpleNamespace(worker=self.worker, native=workspace / "design_state", ctx=ctx,
                                root=ctx.run_dir, run_root=self.root, version_of=tree_hash)
        return consult(proxy, operation, args, workspace, directory, cancel)

    def op_storyboard(self, operation, args, workspace, directory, cancel, step):
        return self.storyboard(args, workspace, directory, cancel)

    def op_accept_storyboard(self, operation, args, workspace, directory, cancel, step):
        boards = read_json(workspace / "storyboards.json", {})
        reviews = args["reviews"]
        # Each condition its own message, and every offender named at once.
        empty = sorted(s for s, n in reviews.items()
                       if not isinstance(n, str) or not n.strip())
        unstaged = sorted(s for s in reviews if s not in boards)
        # A board drawn failing at its own frame is evidence, and not
        # acceptable evidence; the refusal names what fails there.
        failing = {s: boards[s].get("failing_here") or ["the required checks"]
                   for s in sorted(reviews) if s in boards
                   and boards[s].get("checks_pass") is False}
        if empty or unstaged or failing:
            kept = self.keep_review_drafts(workspace, reviews, boards)
            problems = (["empty observation for: " + ", ".join(empty)] if empty else []) + (
                ["no storyboard board for: " + ", ".join(unstaged)
                 + ". Draw it with storyboard() first; a board is removed only when its "
                 "state is invalidated or leaves the plan"] if unstaged else []) + (
                ["these states fail at their own frame and cannot be accepted: "
                 + "; ".join(f"{s} ({', '.join(v)})" for s, v in failing.items())
                 + ". Repair them and re-run storyboard() for them"] if failing else [])
            raise ValueError("; ".join(problems) + "." + (
                f" Nothing was accepted. Your observations for {kept} state(s) were kept, "
                "and storyboard() offers each back as draft_review when it draws the "
                "same panel again." if kept else " Nothing was accepted."))
        # Merged onto the live boards, and applied only to the panel each
        # review was written about: a state redrawn while this ran
        # keeps its new, unreviewed board.
        published = self.merge_boards(workspace, args, cancel, reviews={
            state: {"review": notes, "image_digest": boards[state].get("image_digest")}
            for state, notes in reviews.items()})
        self.drop_review_drafts({s: reviews[s] for s in published["applied"]})
        if published["not_applied"]:
            self.keep_review_drafts(self.work, {s: reviews[s] for s in published["not_applied"]},
                                    boards)
        build_now = self.build_of(self.work)
        reply = {"ok": not published["not_applied"], "version": published["version"],
                 "accepted": published["applied"],
                 "missing": missing_states(self.plan_of(self.work), published["boards"],
                                           build_now["design_id"],
                                           build_now.get("design_objects"))}
        if published["not_applied"]:
            reply.update(failure_class="rejection", not_applied=published["not_applied"])
        if (published["content"] is not None and args.get("_content")
                and published["content"] != args["_content"]):
            reply["changed_since"] = {"from_version": args["_content"],
                                      "to_version": published["content"]}
        return reply

    def op_review(self, operation, args, workspace, directory, cancel, step):
        from .visual import review
        return review(self, args, workspace, directory, cancel)

    def op_check(self, operation, args, workspace, directory, cancel, step):
        counts = counts_as_compile(operation, args)
        result = self.check(workspace, directory / "check", cancel, remember=True,
                            seq=args.get("_seq"), fresh=bool(args.get("_fresh")),
                            workers=args.get("workers"), **interval_of(args))
        # The staged frames were evaluated along with every other one, so
        # what the storyboard's evidence is still worth can be answered
        # from this result rather than by rendering anything.
        observed = self.observe_storyboard(workspace, result)
        if observed:
            result["storyboard"] = observed
        # What stands between this project and a clean `finish()`, from
        # the one function every readiness reply uses. `ok` stays
        # "the required checks pass over the frames checked".
        result["unmet"] = self.unmet(workspace, content=result["version"],
                                     appeared=(observed or {}).get("appeared"))
        # Partial results are recorded too, under their own directory:
        # `compile_reports/latest.json` is read elsewhere as the verdict on
        # the whole animation, and an interval check is not that.
        folder = "compile_reports" if counts else "motion_checks"
        write_json(self.root / folder / f"{step}.json", result)
        # `latest.json` is the run's verdict to the orchestrator, so it
        # follows submission order: checks run concurrently now,
        # and an older check finishing later must not overwrite a newer
        # one's verdict.
        seq = args.get("_seq", 0)
        with self.lock:
            newest = seq >= self.latest_seq.get(folder, 0)
            if newest:
                self.latest_seq[folder] = seq
                write_json(self.root / folder / "latest.json", result)
            if counts:
                self.ctx.n_compile_attempts += 1
                if newest:
                    self.ctx.last_compile_payload = result
        # The persisted report keeps the whole index; the reply carries a
        # digest of it (see `evaluation_digest`).
        return {**result, "evaluation": evaluation_digest(result.get("evaluation"))}

    def op_render(self, operation, args, workspace, directory, cancel, step):
        # `snapshot` is `render` with `encode=False` defaulted. One worker op
        # behind both names lets an interval be drawn through a diagnostic
        # camera and a frame set be encoded. The name stays because the prompt
        # uses it.
        self.build_of(workspace)
        from .delivery import preview
        if operation == "snapshot":
            # `snapshot` keeps its own two defaults: no encode, and a bare
            # call means frame 0 rather than the whole timeline. `render`
            # with no selection means the whole film, which is the right
            # default for the op whose job is a draft and the wrong one
            # for the op whose job is a look.
            args = {**args, "encode": args.get("encode", False)}
            if all(args.get(k) is None for k in ("frames", "start", "end", "seconds")):
                args["frames"] = [args.get("frame") or 0]
        return preview(self, {k: v for k, v in args.items() if k != "frame"},
                       workspace, directory, cancel)

    def op_final_render(self, operation, args, workspace, directory, cancel, step):
        from .delivery import render
        return render(self, args, workspace, directory, cancel)

    def op_finish(self, operation, args, workspace, directory, cancel, step):
        if self.contribution_role == "motion":
            return self.finish_contribution(workspace, directory, cancel)
        return self.finish(workspace, directory, cancel, args)


def register(session):
    service = AnimationSession(session)
    session.blender_animation = service
    from .dispatch import register as register_dispatch
    register_dispatch(session, service)
    for name in ("plan", "build", "blender", "probe", "inspect", "snapshot", "motion_image", "storyboard",
                 "accept_storyboard", "review", "compile", "check_motion", "render",
                 "final_render", "finish", "ik"):
        def handler(_name=name, **kwargs):
            # The same slimming `job_result` applies, so an op run here and the
            # same op polled as a job reply alike: passing verifiers fold to
            # their names, at every depth a storyboard nests them.
            return slim_job_payload(service.submit(_name, **kwargs))
        session.op(name)(handler)
    # Polled replies are slimmed as replies are: `storyboard`'s is this layer's
    # largest, and the prompt has the author poll every 5-15 s.
    register_job_handlers(session, service.jobs)
