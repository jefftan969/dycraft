"""Checks of the film: whole-timeline and interval verification, chunked across workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .blender.bundle import tree_hash, write_json
from .blender.chunking import ContextMismatch, merge_rows, plan_chunks
from .blender.evaluation import merge_chunk_indexes
from .blender.verify import advisory_gaps, aggregate
from .blender.fingerprint import implementation_fingerprint
from ..sdk.ir import content_hash
from .context import read_json, workspace_context
from .planning import required_checks_pass, source_gate

def mem_available_gib():
    """The kernel's `MemAvailable`, in GiB, or None where it cannot be read."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except (OSError, ValueError, IndexError):
        pass
    return None


class CheckingMixin:
    def check_cache_key(self, workspace):
        """Simpler than the native layer's key, because less varies here.

        Everything a check reads — the blend, `design_state/` with its scene,
        catalog, and baseline, the plan — lives inside the workspace, so one
        tree hash covers what that layer spells out field by field. No VLM
        identity either: this layer runs no judge at check time.

        The content identity, not the project's: the boards are not an input
        to any check, and keying on them discarded a passing check at every
        `storyboard()` and `accept_storyboard()`.
        """
        return self.cache_key(self.content_version(workspace))

    @staticmethod
    def cache_key(content):
        return content_hash({"workspace": content,
                             "implementation": implementation_fingerprint()})

    #: Interval checks kept per content version (see `interval_checks`).
    #: Enough for the ranges an author certifies between edits; the oldest
    #: goes first.
    INTERVAL_CHECKS = 16

    #: How many Blender processes a whole-timeline check may use. The frame
    #: loop is frame-local (see `chunking`) and the check is nearly all of a
    #: compile, so splitting it is what moves wall clock rather than CPU, with
    #: identical verdicts. Four by default because each worker holds the whole
    #: scene in memory (a large blend is gigabytes resident) and too many can
    #: trip the host's OOM guard. A call's `workers=` replaces it.
    CHECK_WORKERS = int(os.environ.get("DYCRAFT_ANIMATION_CHECK_WORKERS") or 4)
    #: The measured peak memory of one check worker, in GiB.
    CHECK_WORKER_GIB = 2.5

    def check_workers(self, requested=None):
        """How many processes a whole-timeline check runs on: the request, or
        `CHECK_WORKERS`, capped by the CPUs this process may use and by free
        memory, so a packed host degrades to fewer workers rather than to OOM
        kills."""
        if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int)
                                      or requested < 1):
            raise ValueError("workers must be a positive integer")
        workers = min(requested or self.CHECK_WORKERS, len(os.sched_getaffinity(0)))
        free = mem_available_gib()
        if free is not None:
            workers = min(workers, max(1, int(free // self.CHECK_WORKER_GIB)))
        return workers

    def chunk_plan(self, workers=None):
        """Owned ranges for a parallel check, or None to run it in one process.

        Needs the timeline and the declared events, which live in the blend, so
        the first check of a session runs whole and every later one is planned
        from what it reported. A scene that needs pre-roll is never split: its
        frames are not independent, and each chunk would step from frame zero.
        """
        hint = self.timeline_hint
        if not hint or hint.get("requires_preroll") is not False:
            return None
        workers = self.check_workers(workers)
        if workers < 2:
            return None
        return plan_chunks(range(self.film_frames or hint["n_frames"]), workers,
                           events=(hint.get("declarations") or {}).get("events", []),
                           info=hint)

    def check_in_parallel(self, workspace, directory, cancel, plan, request):
        """Run the chunks, then merge their evaluations and their findings.

        The workspace is mounted read-only: `evaluate` only reads the blend, and
        nothing else may be true while several processes share it.
        """
        merged, done = self.evaluate_in_chunks(workspace, directory, cancel, plan, request)
        # The chunks' per-frame rows, aggregated once: the reports equal one
        # process's by construction. Kept, as a single pass
        # keeps its own, for a later check to re-verify only some frames of.
        rows = merge_rows([read_json(output / "verification_rows.json") for _, _, output in done])
        (directory / "evaluated/verification_rows.json").write_text(json.dumps(rows))
        if advisory_gaps(rows):
            # An instance one chunk saw move has no advisory rows in another:
            # one more worker measures them from the merged evaluation.
            self.worker(workspace, directory / "evaluated",
                        {"op": "fill_rows", "verification": request["verification"]},
                        cancel, workspace_readonly=True)
            rows = read_json(directory / "evaluated/verification_rows.json")
        return merged, aggregate(rows)

    def evaluate_in_chunks(self, workspace, directory, cancel, plan, request):
        """Evaluate the plan's chunks in parallel and merge their evaluations.

        Returns `(merged index, [(owned, evaluation, chunk output)])`. The
        merged evaluation and index are in `evaluated/evaluation`; whatever
        else a chunk wrote stays in its own directory.
        """
        from concurrent.futures import ThreadPoolExecutor

        def run(numbered):
            index, (low, high, frames) = numbered
            output = directory / f"chunk{index}"
            evaluation = self.worker(workspace, output,
                                     {**request, "start": None, "end": None, "seconds": None,
                                      "clip": None, "frames": frames,
                                      "defer_partial_events": False, "owned": [low, high]},
                                     cancel, workspace_readonly=True)
            return (low, high), evaluation, output

        with ThreadPoolExecutor(max_workers=len(plan)) as pool:
            done = list(pool.map(run, enumerate(plan)))
        target = directory / "evaluated/evaluation"
        target.mkdir(parents=True, exist_ok=True)
        for owned, evaluation, output in done:
            low, high = owned
            for path in sorted((output / "evaluation").iterdir()):
                if path.is_dir():                      # meshes/ and tables/ are content-addressed
                    (target / path.name).mkdir(exist_ok=True)
                    for asset in path.iterdir():
                        asset.replace(target / path.name / asset.name)
                elif path.name == "index.json":
                    continue
                elif path.stem.isdigit() and not low <= int(path.stem) <= high:
                    # A context frame, evaluated only to warm the chunk up.
                    # Its owner wrote it by stepping; this one was reached by
                    # a jump, which can store -0.0 where stepping stored +0.0,
                    # so letting it overwrite the owned record made the merged
                    # evaluation differ from a single-process one, and
                    # delivery's replay check then refused the frame.
                    continue
                else:
                    path.replace(target / path.name)
        # Every chunk's frames, signatures, and render-state digests, where the
        # first chunk's index alone kept only its own.
        merged = merge_chunk_indexes([evaluation for _, evaluation, _ in done],
                                     owned=[owned for owned, _, _ in done])
        write_json(target / "index.json", merged)
        return merged, done

    def recheck_in_parallel(self, workspace, directory, cancel, plan, request, base):
        """A re-check across the plan's chunks: what `worker.reverify` does in one process.

        A repeat compile is a re-check, and its floor is evaluating every
        frame, which takes one worker minutes of CPU on a long film even with
        nothing changed. Here the chunks evaluate every frame in parallel,
        without verifying. The merged signatures against the base's name the
        changed frames, and the frames they reach (`reverify_frames`) are
        measured from the merged evaluation, again split by the chunks' owned
        ranges. A changed index field, or base rows over other frames,
        measures every frame instead, as `reverify` does. The rows, reports,
        and the `changed`/`reverified`/`carried` the reply names are one
        worker's, by the same construction as `check_in_parallel`'s.
        """
        from concurrent.futures import ThreadPoolExecutor
        from .blender.chunking import CONTEXT, merge_scoped, reverify_frames
        from .blender.continuity import frame_runs
        from .blender.evaluation import RECHECK_FIELDS
        verification = request["verification"]
        evaluate = {key: value for key, value in request.items()
                    if key not in ("verification", "reverify")}
        merged, _ = self.evaluate_in_chunks(workspace, directory, cancel, plan, evaluate)
        evaluated = directory / "evaluated"
        frames = merged["frames"]
        base_rows = read_json(base["rows"])
        whole = (any(merged.get(field) != base["index"].get(field) for field in RECHECK_FIELDS)
                 or base_rows.get("frames") != frames)
        changed = [] if whole else [
            frame for frame in frames
            if base["signatures"].get(str(frame)) != merged["signatures"].get(str(frame))]

        def measured(owned, needed):
            """`owned`'s rows, measured from the merged evaluation by the chunks that own them."""
            def run(numbered):
                index, (low, high, _) = numbered
                mine = [frame for frame in owned if low <= frame <= high]
                if not mine:
                    return None
                output = directory / f"recheck{index}"
                shutil.rmtree(output, ignore_errors=True)
                # Linked, not copied: the evaluation is written and immutable.
                shutil.copytree(evaluated / "evaluation", output / "evaluation",
                                copy_function=os.link)
                self.worker(workspace, output,
                            {"op": "measure_rows", "verification": verification,
                             "frames": [frame for frame in needed
                                        if low - CONTEXT <= frame <= high + CONTEXT],
                             "owned_frames": mine, "room_frame": frames[0]},
                            cancel, workspace_readonly=True)
                rows = read_json(output / "verification_rows.json")
                shutil.rmtree(output, ignore_errors=True)
                return rows
            with ThreadPoolExecutor(max_workers=len(plan)) as pool:
                parts = [rows for rows in pool.map(run, enumerate(plan)) if rows is not None]
            return merge_rows(parts)

        owned, needed = (frames, frames) if whole else reverify_frames(merged, changed)
        if whole:
            rows = measured(frames, frames)
        elif not owned:
            rows = {**base_rows, "reverified": [],
                    "stats": {key: 0 for key in base_rows["stats"]}}
        else:
            try:
                rows = merge_scoped(base_rows, measured(owned, needed))
            except ContextMismatch:
                # Context the index cannot show moved: sound only measured whole.
                whole, changed, owned = True, [], frames
                rows = measured(frames, frames)
        (evaluated / "verification_rows.json").write_text(json.dumps(rows))
        if (not whole and owned) or advisory_gaps(rows):
            # The base's overlap spans may now run into re-measured frames, and
            # an instance the repair set moving has no rows on the carried ones.
            self.worker(workspace, evaluated,
                        {"op": "fill_rows", "verification": verification,
                         "diagnostics": not whole},
                        cancel, workspace_readonly=True)
            rows = read_json(evaluated / "verification_rows.json")
        reports = aggregate(rows)
        if not whole:
            owned_set = set(owned)
            merged.update(changed=frame_runs(changed), reverified=frame_runs(owned),
                          carried=frame_runs([frame for frame in frames if frame not in owned_set]))
            rows["reverified"] = frame_runs(owned)
            (evaluated / "verification_rows.json").write_text(json.dumps(rows))
            scope = {"reverified": rows["reverified"], "carried": merged["carried"]}
            reports = [{**report, "scope": scope} for report in reports]
            write_json(evaluated / "evaluation/index.json", merged)
        (evaluated / "verifier_reports.json").write_text(json.dumps(reports))
        return merged, reports

    def reverify_base(self, workspace):
        """The last whole-film check, if a re-check against it is sound.

        Not when the design or the checker changed since -- a frame's verdict
        depends on the scene IR, the catalog, and the verifiers as well as on
        its content -- nor when its rows or evaluation are gone. Index-level
        fields (fps, range, declarations, ...) are compared in the worker,
        which reads them.
        """
        held = self.last_whole
        if not held or held["implementation"] != implementation_fingerprint():
            return None
        rows, index = Path(held["rows"]), Path(held["evaluation"]) / "index.json"
        if not rows.is_file() or not index.is_file():
            return None
        if held["design"] != tree_hash(workspace / "design_state"):
            return None
        info = read_json(index)
        if not info.get("signatures"):
            return None
        from .blender.evaluation import RECHECK_FIELDS
        return {"rows": rows, "signatures": info["signatures"], "job": self.job_of(held["evaluation"]),
                "index": {field: info.get(field) for field in RECHECK_FIELDS}}

    def remember_whole(self, workspace, directory, seq):
        """Make this whole-film check the base of the next re-check, in submission order."""
        rows = directory / "evaluated/verification_rows.json"
        if not rows.is_file():
            return
        seq = seq or 0
        with self.lock:
            held = self.last_whole
            if held and held["seq"] > seq:
                return
            self.last_whole = {"seq": seq, "evaluation": str(directory / "evaluated/evaluation"),
                               "rows": str(rows), "design": tree_hash(workspace / "design_state"),
                               "implementation": implementation_fingerprint()}

    #: The evaluate request a merged render carries, so the render worker can
    #: verify the frames it drew without a second Blender start.
    def verification_spec(self, workspace):
        ctx = workspace_context(workspace)
        return {"verification": {"baseline": read_json(workspace / "design_state/baseline.json"),
                                 "scene": ctx.scene_ir, "catalog": ctx.catalog,
                                 **self.pair_store_spec()}}

    @property
    def pair_store(self):
        """The directory holding this project's collision pair store.

        Every check's worker reads and writes the store itself, bound
        read-write at its own path, so pairs persist across checks without
        being shipped through every request and reply.
        """
        return self.native / "collision_pairs"

    def pair_store_spec(self):
        return {"collision_pair_store": str(self.pair_store.resolve() / "pairs.sqlite")}

    def store_binds(self, request):
        """The pair store, bound read-write only into a worker that checks.

        A check trusts the verdicts it reads there, so code an author runs must
        not reach it: a probe runs its code in a forked worker and a job's
        edits in that job's resident, and neither gets the bind. The project
        resident keeps it: interactive checks run there, beside the edits, and
        it binds all of `native/` already.
        """
        spec = (request.get("verification")
                or (request.get("evaluate") or {}).get("verification") or {})
        if request.get("op") in ("evaluate", "render", "measure_rows", "fill_rows") and spec.get("collision_pair_store"):
            return [self.pair_store]
        return []

    def check(self, workspace, directory, cancel, *, start=None, end=None,
              seconds=None, frames=None, clip=None, version=None, evaluated=None,
              remember=False, seq=None, fresh=False, workers=None):
        """`version` lets a caller that checks repeatedly hash the workspace once.
        It is the workspace's content version (`identity`), which is also the
        `version` the payload reports.

        A job owns a private copy of the workspace and nothing mutates it for
        the job's lifetime — `check` writes only under its own `directory` —
        so the hash is the same for every check in one call. `storyboard()`
        makes one per shot, and a full SHA-256 pass over the tree each time
        would cost seconds for an answer that cannot have changed.

        `evaluated=(info, output_dir)` says a worker has already evaluated and
        verified exactly these frames — `storyboard()`'s render does, in the
        same process that drew the panels — so this assembles the payload from
        that output instead of starting a second Blender. It is only ever a
        partial check, so it never enters the validation cache.

        `remember=True` keeps a partial check too, per content version and
        interval (`interval_checks`), so a clean `check_motion` over a range
        can certify that range for delivery. `compile` and
        `check_motion` pass it; a storyboard's staged-frame check does not.

        A whole-film check after an earlier one re-verifies only the frames
        the repair can have changed: the worker evaluates
        every frame, diffs their signatures against the last whole-film check
        (`last_whole`, by submission order `seq`), measures what the changes
        reach, and merges the rest of that check's rows. The reply names what
        it carried. `fresh=True`, a partial check, a changed design or
        implementation, or a base that is gone checks every frame.
        """
        if workers is not None:
            self.check_workers(workers)    # refused as such even when unused
        directory.mkdir(parents=True, exist_ok=True)
        build = self.build_of(workspace)
        # `clip` rides along unresolved: only the worker can read the NLA, and
        # resolving it here would cost a Blender launch to learn something the
        # evaluation is about to read anyway. It still makes the check partial,
        # which is what keeps a clip-scoped `compile` a diagnostic.
        interval = {"start": start, "end": end, "seconds": seconds,
                    "frames": frames, "clip": clip}
        partial = any(v is not None for v in interval.values())
        content = version if version is not None else self.content_version(workspace)
        # Whole-timeline checks are cached in `validation_cache`, one entry
        # that delivery and completion reuse. Remembered intervals go in their
        # own store, so they can never evict it. Failing results are cached
        # too: with no judge in this layer a check is a pure function of the
        # workspace, so re-asking an unchanged project why it failed should be
        # free.
        key = self.cache_key(content) if not partial or remember else None
        interval_key = (json.dumps(interval, sort_keys=True, default=list)
                        if partial and remember else None)
        if not partial:
            cached = self.validation_cache.get(key)
        elif interval_key:
            cached = self.interval_checks.get(key, {}).get(interval_key)
        else:
            cached = None
        if cached and tree_hash(Path(cached["path"])) != cached["hash"]:
            cached = None
        carried_over = ran_on = None
        if evaluated is not None:
            evaluation, source = evaluated
            reports = read_json(source / "verifier_reports.json")
            evaluation_path = str(source / "evaluation")
            self.timeline_hint = {k: evaluation.get(k) for k in self.TIMELINE_KEYS}
        elif cached:
            evaluation, reports = cached["evaluation"], list(cached["reports"])
            evaluation_path = cached["path"]
            carried_over = cached.get("carried_over")
        else:
            request = {"op": "evaluate", **interval, **self.verification_spec(workspace)}
            plan = None if partial else self.chunk_plan(workers)
            base = None if partial or fresh else self.reverify_base(workspace)
            if base is not None and plan:
                evaluation, reports = self.recheck_in_parallel(
                    workspace, directory, cancel, plan, request, base)
                if evaluation.get("carried"):
                    carried_over = {"frames": evaluation["carried"], "from_job": base["job"]}
            elif base is not None:
                output = directory / "evaluated"
                output.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base["rows"], output / "base_rows.json")
                request["reverify"] = {"base_signatures": base["signatures"],
                                       "base_index": base["index"]}
                evaluation = self.worker(workspace, output, request, cancel)
                reports = read_json(output / "verifier_reports.json")
                if evaluation.get("carried"):
                    carried_over = {"frames": evaluation["carried"], "from_job": base["job"]}
            elif plan:
                try:
                    evaluation, reports = self.check_in_parallel(
                        workspace, directory, cancel, plan, request)
                except ContextMismatch:
                    # The chunks read the scene differently (a room wall
                    # hidden on one chunk's first frame): one pass, then,
                    # over none of the chunks' merged records.
                    shutil.rmtree(directory / "evaluated", ignore_errors=True)
                    plan = None
            if base is None and not plan:
                evaluation = self.worker(workspace, directory / "evaluated", request, cancel)
                reports = read_json(directory / "evaluated/verifier_reports.json")
            ran_on = len(plan) if plan else 1
            self.timeline_hint = {k: evaluation.get(k) for k in self.TIMELINE_KEYS}
            evaluation_path = str(directory / "evaluated/evaluation")
            entry = {"path": evaluation_path, "hash": tree_hash(Path(evaluation_path)),
                     "evaluation": evaluation, "reports": list(reports),
                     **({"carried_over": carried_over} if carried_over else {})}
            if not partial:
                with self.lock:
                    if (seq or 0) >= self.cached_seq["whole"]:
                        self.validation_cache, self.cached_seq["whole"] = {key: entry}, seq or 0
                self.remember_whole(workspace, directory, seq)
            elif interval_key:
                with self.lock:
                    # One content version at a time: an edit makes every
                    # older interval unusable for certification anyway.
                    if key in self.interval_checks or (seq or 0) >= self.cached_seq["interval"]:
                        held = {k: v for k, v in self.interval_checks.get(key, {}).items()
                                if k != interval_key}
                        held[interval_key] = entry
                        self.interval_checks = {
                            key: dict(list(held.items())[-self.INTERVAL_CHECKS:])}
                        self.cached_seq["interval"] = max(seq or 0, self.cached_seq["interval"])
        frames = evaluation["frames"]
        # A check that ran and found a defect is not a failed operation, so it
        # is not counted with exceptions and refusals.
        payload = {"ok": required_checks_pass(reports),
                   "verifier_reports": reports,
                   "version": content,
                   "evaluation": evaluation,
                   "evaluation_path": evaluation_path, "partial": partial,
                   "validation_reused": cached is not None,
                   **({"carried_over": carried_over} if carried_over else {}),
                   **({"workers": ran_on} if ran_on else {}),
                   # In front of the agent at every checkpoint it already
                   # makes, so the render arithmetic has a denominator.
                   **({"budget": budget} if (budget := self.wall_budget()) else {}),
                   "frames_checked": {"start": frames[0], "end": frames[-1], "count": len(frames),
                                      "seconds": [frames[0] / evaluation["fps"],
                                                  frames[-1] / evaluation["fps"]]}}
        if not partial:
            reports.extend(build["verifier_reports"])
            # The same rule as the storyboard gate: an advisory `uncertain`
            # from the design's suite does not fail the film.
            payload["ok"] = payload["ok"] and not source_gate(build["verifier_reports"])[0]
        # Only a check that found something is a finding.
        if not payload["ok"]:
            payload["failure_class"] = "check_finding"
        write_json(directory / "report.json", payload)
        return payload
