"""Finishing the film: the commit, its refusals, and what the delivered bundle carries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .blender.bundle import make_manifest, tree_hash, validate_manifest, write_json
from ..sdk.ir import content_hash
from .context import read_json, workspace_context
from .planning import plan_states, state_revision

class CommitMixin:
    @property
    def drafts_path(self):
        """Observations from a refused `accept_storyboard`, outside the workspace
        because a refused op publishes nothing."""
        return self.native / "review_drafts.json"

    def keep_review_drafts(self, workspace, reviews, boards):
        """Keep each non-empty observation against the panel it describes.

        Keyed on the state's revision and the digest of the panel reviewed:
        the board's in `boards`, as the acceptance read them, else the panel
        last drawn for the state, which lives on in `views_root` whether or
        not its board did. Not the panel there now: a state redrawn since is
        the reason a review is refused. `eggs` resubmitted 40-41 observations
        six times in eight calls.
        """
        plan = self.plan_of(workspace)
        by_id = {s["id"]: s for s in plan_states(plan)}
        drafts, kept = read_json(self.drafts_path, {}), 0
        for state, notes in reviews.items():
            panel = self.views_root / content_hash(state) / "000.png"
            digest = (boards.get(state) or {}).get("image_digest") or (
                self.image_digest(panel) if panel.is_file() else None)
            if state not in by_id or not isinstance(notes, str) or not notes.strip() \
                    or digest is None:
                continue
            drafts[state] = {"review": notes, "state_revision": state_revision(plan, by_id[state]),
                             "image_digest": digest}
            kept += 1
        if kept:
            write_json(self.drafts_path, drafts)
        return kept

    def drop_review_drafts(self, reviews):
        drafts = read_json(self.drafts_path, {})
        if set(drafts) & set(reviews):
            write_json(self.drafts_path, {k: v for k, v in drafts.items() if k not in reviews})

    @staticmethod
    def clips_ready(rendered, boards):
        """Which clips have every state bound into them reviewed.

        Reported, never gated: the storyboard gate stays whole-film. This is
        so an author can see where motion work can start without reading the
        board file themselves.
        """
        ready = {}
        for name, evidence in rendered.items():
            clip = evidence.get("clip")
            if clip is None:
                continue
            ready[clip] = ready.get(clip, True) and bool(
                boards.get(name, {}).get("reviewed"))
        return dict(sorted(ready.items()))

    def bundle_storyboard_views(self, bundle):
        """Put the panels back for delivery, where they belong with the film.

        They left the workspace so that carrying them stopped costing every
        operation (`views_root`); a bundle is written once, so the same
        argument does not apply to it. The delivered layout is unchanged from
        when they lived in the workspace, which is what `make_manifest` then
        hashes into the receipt.
        """
        if self.views_root.is_dir():
            shutil.copytree(self.views_root, bundle / "storyboard_views", dirs_exist_ok=True)

    def record_finish(self, payload):
        """Write what `finish()` decided, on every path including a refusal.

        `finish_result.json` is what a reader of the run directory looks for, so
        the harness writes it on every path. A refusal must leave it too:
        otherwise "did completion fail, or did it never run?" is the question
        the run directory cannot answer.
        """
        from datetime import datetime, timezone
        record = {**payload, "finished_at": datetime.now(timezone.utc).isoformat()}
        record.setdefault("version", self.version())
        write_json(self.root / "finish_result.json", record)
        return payload

    def finish_contribution(self, workspace, directory, cancel):
        try:
            return self.record_finish(
                self._finish_contribution(workspace, directory, cancel))
        except BaseException as error:
            self.record_finish({"ok": False, "reason": str(error),
                                "error_type": type(error).__name__})
            raise

    def finish(self, workspace, directory, cancel, args=None):
        try:
            return self.record_finish(self._finish(workspace, directory, cancel, args or {}))
        except BaseException as error:
            self.record_finish({"ok": False, "reason": str(error),
                                "error_type": type(error).__name__})
            raise

    def _finish_contribution(self, workspace, directory, cancel):
        """A dispatched specialist returns a checked private project to its parent."""
        if self.ctx.committed:
            manifest = read_json(self.root / "ir.json")
            validate_manifest(self.root, manifest)
            return {"ok": True, "already_committed": True, "ir_hash": manifest["id"]}
        from .dispatch import running
        if running(self):
            raise ValueError("wait for outstanding contributions before finish")
        self.source_fresh(workspace)
        # A clip-scoped specialist is checked over its clip, not the film. It
        # was given one clip to author and told not to touch the rest, so a
        # whole-timeline verdict here answers a question about its launcher's
        # work -- and on a long score that check is tens of minutes. The
        # launcher re-checks after integrating, which is where the whole-film
        # claim belongs; `MotionContinuity` then names any join the swap broke.
        checked = self.check(workspace, directory / "check", cancel,
                             clip=self.contribution_clip)
        if not checked["ok"]:
            return checked
        bundle = directory / "bundle"
        shutil.copytree(workspace, bundle)
        shutil.copytree(checked["evaluation_path"], bundle / "evaluation")
        self.bundle_storyboard_views(bundle)
        animation_source_hash = self.bundle_sources(bundle)
        manifest = make_manifest(bundle, scene_ref=workspace_context(workspace).scene_ir["id"], evaluation=checked["evaluation"])
        manifest["animation_source_hash"] = animation_source_hash
        manifest["contribution_role"] = "motion"
        manifest["contribution_clip"] = self.contribution_clip
        manifest["id"] = content_hash({k: v for k, v in manifest.items() if k != "id"})
        with self.lock:
            self.source_fresh(workspace)
            if cancel.is_set() or self.content_version(workspace) != self.content_version(self.work):
                raise ValueError("contribution changed or was cancelled before publication")
            bundle.rename(self.root / "bundle")
            try:
                write_json(self.root / "ir.json", manifest)
            except BaseException:
                (self.root / "bundle").rename(bundle)
                raise
            self.ctx.committed, self.ctx.last_successful_ir = True, manifest
            self.session.trace("commit", ir_hash=manifest["id"])
            self.session.finalize_meta("committed")
        return {"ok": True, "ir_path": str(self.root / "ir.json"), "contribution_role": "motion",
                "clip": self.contribution_clip,
                "checked": checked["frames_checked"],
                "blend": str(self.root / "bundle/animation.blend"), "integration": "launcher owns integration and film delivery"}

    def _finish(self, workspace, directory, cancel, args):
        """Commit the project and its film: cleanly, or with named exceptions.

        A clean commit still means the whole film, every state, and every
        check: `unmet` is empty. `exceptions=True` is the one explicit
        decision to commit with shortfalls, as `delivered_with_exceptions`,
        usable any time and never automatic; it unlocks nothing for
        authoring. Without it, a refusal names what is unmet. Either way the
        receipt must be for this project's content -- a film drawn from
        another project state is never accepted -- and "frozen scene identity
        changed" is never waived, because it makes the check meaningless.

        `criteria` is required: a verdict with evidence for every
        criterion the plan declares. A `fail` is unmet; `uncertain` is
        recorded and is not.
        """
        if self.ctx.committed:
            manifest = read_json(self.root / "ir.json")
            validate_manifest(self.root, manifest)
            return {"ok": True, "ir_hash": manifest["id"], "already_committed": True,
                    "outcome": manifest.get("outcome", "committed")}
        from .dispatch import running
        if running(self):
            raise ValueError("Wait for or cancel outstanding contributions before finish().")
        self.source_fresh(workspace)
        plan = self.plan_of(workspace)
        criteria = self.criteria_verdicts(plan, args.get("criteria"))
        receipt = read_json(self.native / "delivery.json")
        content = args.get("_content") or self.content_version(workspace)
        if not receipt or receipt.get("version") != content:
            raise ValueError(self.stale_film(receipt, workspace))
        if tree_hash(Path(receipt["artifacts"])) != receipt["artifacts_hash"]:
            raise ValueError("render/check artifacts changed; render again")
        checks = receipt.get("checks") or {}
        if self.frozen_identity_changed(checks.get("verifier_reports") or ()):
            raise ValueError("The scene the film was checked against is not the frozen scene "
                             "(frozen scene identity changed). That is never waived: rebuild "
                             "and check again.")
        # The receipt's own check outlives a daemon restart that empties the
        # caches; it is evidence about exactly this content version.
        extra = ([self.evidence_entry("whole", {"evaluation": checks["evaluation"],
                                                "reports": checks["verifier_reports"],
                                                "path": checks.get("evaluation_path")})]
                 if checks.get("verifier_reports") and not checks.get("partial") else [])
        appeared = self.appeared_since_review(
            workspace, checks if extra else None, content=content)
        unmet = self.unmet(workspace, content=content, appeared=appeared, criteria=criteria,
                           drawing=receipt.get("drawing") or (), extra=extra)
        if unmet and not args.get("exceptions"):
            return {"ok": False, "failure_class": "rejection", "unmet": unmet,
                    "error": self.finish_refusal(unmet)}
        outcome = "delivered_with_exceptions" if unmet else "committed"
        bundle = directory / "bundle"
        shutil.copytree(workspace, bundle)
        shutil.copytree(Path(receipt["artifacts"]) / "evaluation", bundle / "evaluation")
        shutil.copytree(Path(receipt["artifacts"]) / "film", bundle / "film")
        self.bundle_storyboard_views(bundle)
        history = bundle / "edit_history"
        history.mkdir(exist_ok=True)
        for source in sorted((self.native / "jobs").glob("*/source.py")):
            shutil.copy2(source, history / (source.parent.name + ".py"))
        animation_source_hash = self.bundle_sources(bundle)
        settings = receipt.get("requested_render_settings") or receipt["render_settings"]
        self.bundle_render_settings(bundle, settings, directory, cancel)
        notes = self.ship_notes(bundle)
        (bundle / "DELIVERY.md").write_text(self.delivery_notes(
            outcome=outcome, content=content, unmet=unmet, criteria=criteria,
            settings=receipt["render_settings"], receipt=receipt, notes=notes))
        manifest = make_manifest(bundle, scene_ref=workspace_context(workspace).scene_ir["id"],
                                 evaluation=checks["evaluation"])
        manifest.update(layer="animation", film="bundle/film/clip.mp4",
                        render=receipt["render_settings"], plan="bundle/plan.json",
                        animation_source_hash=animation_source_hash,
                        outcome=outcome, unmet=unmet, criteria=criteria)
        manifest["id"] = content_hash({k: v for k, v in manifest.items() if k != "id"})
        with self.lock:
            if cancel.is_set():
                raise RuntimeError("job cancelled")
            self.source_fresh(workspace)
            if self.content_version(self.work) != receipt["version"]:
                raise ValueError("project changed before completion")
            destination = self.root / "bundle"
            if destination.exists():
                raise ValueError("a bundle already exists; inspect it before retrying completion")
            bundle.rename(destination)
            try:
                write_json(self.root / "ir.json", manifest)
            except BaseException:
                destination.rename(bundle)
                raise
            self.ctx.committed, self.ctx.last_successful_ir = True, manifest
            self.session.trace("commit", ir_hash=manifest["id"], outcome=outcome)
            self.session.finalize_meta(outcome)
        result = {"ok": True, "outcome": outcome, "unmet": unmet, "notes_shipped": notes,
                  "ir_hash": manifest["id"], "ir_path": str(self.root / "ir.json"),
                  "blend": str(destination / "animation.blend"),
                  "film": str(destination / "film/clip.mp4")}
        result["viewer_scene"] = self.stage_viewer_scene(destination)
        # The run's top level: the film, its project, and the README, for whoever opens the run folder. After the
        # commit and never able to fail it (`run_layout.publish`); the launcher writes it again when it exits.
        from ..run_layout import publish
        self.session.trace("publish", **publish(self.root))
        return result

    @staticmethod
    def criteria_verdicts(plan, supplied):
        """One `{verdict, evidence}` per declared criterion, or a refusal that is the checklist.

        `plan()` requires criteria, and declared criteria that nobody walks can
        name exactly the defect a film ships. So `finish()` asks for a verdict
        on each, judged on the finished film, and records them in `ir.json` and
        `DELIVERY.md`.
        """
        declared = {c["id"]: c for c in plan["criteria"]}
        supplied = supplied if supplied is not None else {}
        if not isinstance(supplied, dict):
            raise ValueError("criteria must map each criterion id to {'verdict', 'evidence'}")
        missing = [cid for cid in declared if cid not in supplied]
        unknown = sorted(cid for cid in supplied if cid not in declared)
        invalid = sorted(cid for cid, value in supplied.items() if cid in declared and (
            not isinstance(value, dict) or value.get("verdict") not in ("pass", "fail", "uncertain")
            or not isinstance(value.get("evidence"), str) or not value["evidence"].strip()))
        if missing or unknown or invalid:
            lines = [f"\n  {cid}: {declared[cid]['requirement']} (observe: {declared[cid]['observe']})"
                     for cid in missing]
            raise ValueError(
                "finish() needs a verdict for every plan criterion, judged on the finished film: "
                "criteria={id: {'verdict': 'pass' | 'fail' | 'uncertain', 'evidence': '...'}}. "
                "A fail is listed as unmet, so committing it needs exceptions=True."
                + (" Missing:" + "".join(lines) if missing else "")
                + (" Not in the plan: " + ", ".join(unknown) + "." if unknown else "")
                + (" Needs a verdict and non-empty evidence: " + ", ".join(invalid) + "."
                   if invalid else ""))
        return {cid: {"verdict": supplied[cid]["verdict"],
                      "evidence": supplied[cid]["evidence"].strip()} for cid in declared}

    @staticmethod
    def frozen_identity_changed(reports):
        well_formed = next((r for r in reports if r.get("name") == "AnimationWellFormed"), None)
        failures = ((well_formed or {}).get("evidence") or {}).get("failures") or ()
        return any(f.get("reason") == "frozen scene identity changed" for f in failures)

    @staticmethod
    def finish_refusal(unmet):
        """Name what stands between the film and a clean commit, and the cheapest way to clear it."""
        cheapest = {
            "unchecked": "run compile(); the film needn't be redrawn",
            "check": "repair, compile() again (only frames that changed need redrawing), then final_render(frames_from=[...])",
            "design": "repair the design source and build()",
            "storyboard": "storyboard() and accept_storyboard() the states named",
            "criterion": "repair, or commit the failed criterion with exceptions=True",
            "replay_mismatch": "redraw the frames named with render(delivery=True)",
            "draft_quality": "redraw them with render(delivery=True)",
            "unverified_frames": "redraw them with render(delivery=True)"}
        lines = []
        for item in unmet:
            where = item.get("frames") or item.get("states") or item.get("verifier") or item.get("id")
            what = item.get("verifier") or item.get("id") or ""
            lines.append(f"{item['kind']}{' ' + what if what and what != where else ''}: "
                         f"{where} -- {cheapest.get(item['kind'], 'see unmet')}")
        return ("Not a clean commit yet; unmet:\n  " + "\n  ".join(lines)
                + "\nfinish(exceptions=True) commits this film as delivered with these "
                "exceptions instead.")

    def bundle_render_settings(self, bundle, settings, directory, cancel):
        """Write the delivery settings into the bundle's blend, set to still output.

        `final_render` does not write into the project, so the delivered blend
        gets its settings here, in the bundle's copy only. Output is PNG frames
        beside the film, not FFMPEG: a recipient rendering a still from a blend
        left on video output gets "Cannot write a single file with an animation
        format selected".
        """
        self.worker(bundle, directory / "bundle_settings", {"op": "edit", "code":
            "from render_config import apply_final_render_settings\n"
            f"result = apply_final_render_settings(bpy, {settings!r})\n"
            "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
            "bpy.context.scene.render.filepath = '//film/frames/'\n"}, cancel)
        # Before the bundle is renamed into place; see `publish`.
        self.close_residents(bundle)

    #: The author's notes `finish()` ships, from the run folder. A fixed list,
    #: because the run folder also holds harness files.
    NOTE_FILES = ("README.md", "DELIVERY_NOTES.md", "EDITING.md")

    def ship_notes(self, bundle):
        """Copy the author's notes into `bundle/notes/` and say which.

        From the run folder, not only `animation_src/`, so a `README.md` or
        `DELIVERY_NOTES.md` an author writes beside the project ships too.
        """
        target, shipped = bundle / "notes", []
        sources = [(self.root / name, name) for name in self.NOTE_FILES]
        sources += [(path, f"inspection/{path.name}")
                    for path in sorted((self.root / "inspection").glob("*.md"))]
        for path, name in sources:
            if path.is_file():
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / name)
                shipped.append(f"notes/{name}")
        return shipped

    @staticmethod
    def delivery_notes(*, outcome, content, unmet, criteria, settings, receipt, notes):
        """`bundle/DELIVERY.md`: the harness's facts about what was delivered.

        The harness writes facts; the author writes prose (`notes/`).
        """
        def bullet(item):
            rest = {k: v for k, v in item.items() if k not in ("kind", "rows")}
            return f"- **{item['kind']}**: " + json.dumps(rest, sort_keys=True)

        lines = ["# Delivery", "",
                 f"- **Outcome:** {'committed' if outcome == 'committed' else 'delivered with exceptions'}",
                 f"- **Content version:** `{content}`",
                 f"- **Film:** `film/clip.mp4`, composed from "
                 f"{json.dumps(receipt.get('admitted') or {}, sort_keys=True)}",
                 f"- **Render settings:** `{json.dumps(settings, sort_keys=True)}`",
                 "- **The blend** carries these settings, with PNG output to `//film/frames/`.",
                 "", "## Unmet", ""]
        lines += [bullet(item) for item in unmet] or ["Nothing: every frame, state, and check is clean."]
        lines += ["", "## Criteria", ""]
        lines += [f"- **{cid}** -- {value['verdict']}: {value['evidence']}"
                  for cid, value in criteria.items()]
        lines += ["", "## Author notes", ""]
        lines += [f"- `{name}`" for name in notes] or ["None shipped."]
        return "\n".join(lines) + "\n"

    def stage_viewer_scene(self, bundle):
        """Stage the frozen scene at the run root so the viewer can draw this run.

        This layer authors its scene and freezes it inside the bundle, so
        nothing lands at the run root on its own — and the viewer reads an
        animation run's geometry from `<run>/scene.json` plus `<run>/catalog/`
        and nowhere else.

        **Best effort, by construction.** This runs after the commit is
        durable and outside the lock, and it writes only at the run root,
        never under `bundle/` — so it cannot invalidate the manifest
        `make_manifest` just wrote, and a failure here leaves a committed
        animation that is simply not yet viewable. Re-runnable at any time
        with::

            python -m dycraft.animation.stage_scene <run_dir>

        `SystemExit` is caught alongside `Exception` deliberately:
        `materialize_catalog` and `validate_scene_ir` both raise it (they are
        launcher-side helpers, where exiting is the right response to a bad
        input), and it is **not** an `Exception` — a bare `except Exception`
        would let a staging fault escape a run that has already written
        `ir.json` and called `finalize_meta("committed")`.
        """
        from ..agent.layers.scene_staging import materialize_scene
        try:
            staged = materialize_scene(Path(bundle), self.root)
        except (Exception, SystemExit) as exc:      # noqa: BLE001 — see above
            detail = f"{type(exc).__name__}: {exc}"
            self.session.trace("stage_viewer_scene_failed", error=detail)
            return {"ok": False, "error": detail,
                    "retry": ("python -m dycraft.animation.stage_scene "
                              + str(self.root))}
        self.session.trace("stage_viewer_scene", **staged)
        return {"ok": True, **staged}
