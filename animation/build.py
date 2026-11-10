"""Building the film's Blender project from the design, and staging the author's sources."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from .blender.bundle import skip_sandbox_masks, tree_hash, write_json
from .blender.import_scene import prepare_scene
from ..sdk.ir import content_hash
from .context import read_json, workspace_context
from .planning import PASSING, plan_states

def _fresh_copy(src, dst, *, follow_symlinks=True):
    """`shutil.copy2` onto a new inode, so a hardlink to `dst` keeps the old bytes."""
    temp = f"{dst}.staging"
    shutil.copy2(src, temp, follow_symlinks=follow_symlinks)
    os.replace(temp, dst)
    return dst


class BuildMixin:
    def compile_design(self, source, workspace):
        from ..agent.layers.layer import scene_layer_config
        from ..agent.layers.scene_context import SceneCatalog
        from ..agent.layers.tools import ToolContext, build_current_ir
        from ..verify.scene_verify import support_connected, scene_relational_assertions, room_containment
        from ..verify._per_object import PER_OBJECT_VERIFIERS
        destination = workspace / "design_state"
        shutil.rmtree(destination, ignore_errors=True)
        target = destination / "source"
        source_files = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
        source_hash = tree_hash(source)
        shutil.copytree(source, target)
        payload = SceneCatalog(target)
        ctx = ToolContext(run_dir=target, layer_cfg=scene_layer_config(), layer_payload=payload)
        payload.bind(ctx)
        raw = build_current_ir(ctx)
        if not raw.get("ok"):
            return {"ok": False, "failure_class": "author_code", "build_error": raw.get("build_error"),
                    "sandbox_violation": raw.get("sandbox_violation"), "error": raw.get("error_summary"),
                    "geometry_notes": raw.get("geometry_notes", [])}
        scene, catalog = raw["ir"], raw["object_catalog"]
        # Retain source geometry/mechanism sanity, configuration support, and
        # explicit assertions. Penetration is checked on authored output frames,
        # rather than imposing the object's unused joint-range collision sweep.
        reports = [fn(scene, ctx).to_dict() for fn in
                   (support_connected, room_containment, scene_relational_assertions)]
        for name, fn in PER_OBJECT_VERIFIERS:
            if name != "NoSelfPenetration":
                reports.append(fn(scene, ctx).to_dict())
        write_json(destination / "scene.json", scene)
        write_json(destination / "catalog.json", catalog)
        build = {"source_hash": source_hash, "source_files": source_files,
                 "design_id": content_hash({"scene": scene, "catalog": catalog}),
                 # Per-instance, so a repair to one object does not invalidate
                 # every storyboard state. `design_id` alone is one scalar over
                 # the whole build, and comparing it would re-clear every state
                 # after any repair, including states that do not show the
                 # changed object.
                 "design_objects": {inst["name"]: content_hash(
                     {"instance": inst, "object": catalog.get(inst["object_ref"])})
                     for inst in scene.get("instances", [])},
                 "verifier_reports": reports, "geometry_notes": raw.get("geometry_notes", [])}
        write_json(destination / "build.json", build)
        return {"ok": True, **build}

    def build(self, args, workspace, directory, cancel):
        self.plan_of(workspace)
        candidate_id = args.get("candidate")
        if candidate_id:
            rec = self.candidates.get(candidate_id)
            if not rec:
                raise ValueError("unknown repair candidate; rerun build after a daemon restart")
            if rec["content"] != args.get("_content"):
                raise ValueError("repair candidate is stale; rebuild from the current project")
            shutil.rmtree(workspace)
            shutil.copytree(rec["workspace"], workspace)
            self.source_fresh(workspace)
            issues = rec["result"]["issues"]
            acknowledged = set(args.get("accept_changes") or ())
            if acknowledged != {i["id"] for i in issues}:
                return {"ok": False, "failure_class": "rejection",
                        "error": "acknowledge each diagnosed binding change with accept_changes",
                        **rec["result"]}
            changed = None
            if args.get("repair_code"):
                self.worker(workspace, directory / "repair", {"op": "edit", "code": args["repair_code"]}, cancel)
                changed = read_json(directory / "repair/animation_changed.json")
            # Never publish a repaired tree whose generated mechanism is invalid.
            check = self.check(workspace, directory / "repair_check", cancel, start=0, end=0)
            binding = [r for r in check["verifier_reports"] if r["name"] in ("AnimationWellFormed", "JointLimits")]
            if not binding or any(r["verdict"] not in PASSING for r in binding):
                return {**check, "ok": False, "failure_class": "check_finding",
                        "error": "repair candidate has invalid generated bindings"}
            setting = rec.get("setting") or []
            version = self.publish(workspace, args, cancel,
                                   drop_boards=self.setting_drops(workspace, setting))
            if changed and changed.get("n_frames"):
                self.film_frames = changed["n_frames"]
            return {"ok": True, "version": version, "accepted_changes": sorted(acknowledged),
                    **self.setting_note(setting)}

        old_baseline = read_json(workspace / "design_state/baseline.json")
        old_scene = read_json(workspace / "design_state/scene.json")
        result = self.compile_design(args["_source"], workspace)
        if not result["ok"]:
            return result
        inputs = directory / "inputs"
        prepare_scene(workspace_context(workspace), inputs)
        donor = directory / "donor"
        donor.mkdir()
        # Only the generated snapshot's assets are mounted, not mutable sources.
        from .blender.runtime import run_worker
        run_worker(donor, directory / "import", {"op": "initialize", "spec": str(inputs / "scene_spec.json"),
                   "preserve_render_settings": True, "journal": self.journal.on}, cancel=cancel,
                   readonly=[inputs, workspace / "design_state/source"])
        baseline = read_json(directory / "import/baseline.json")
        baseline["allow_native_collision_allowances"] = True
        if old_baseline:
            update = run_worker(workspace, directory / "update", {
                "op": "update", "donor": str(donor / "animation.blend"),
                "old_baseline": old_baseline, "new_baseline": baseline,
                "refresh": args.get("refresh"), "preserve_render_settings": True,
                "journal": self.journal.on},
                cancel=cancel, readonly=[donor])
        else:
            shutil.copy2(donor / "animation.blend", workspace / "animation.blend")
            update = {"issues": [], "added": sorted(baseline["generated"])}
        if "generated" in update:
            baseline["generated"] = update.pop("generated")
        write_json(workspace / "design_state/baseline.json", baseline)
        # The room and the lighting are in every panel and in no state's
        # design fingerprint, which covers only the instances the panel shows,
        # so a build that changes either drops every board.
        scene = read_json(workspace / "design_state/scene.json") or {}
        refresh = args.get("refresh") if isinstance(args.get("refresh"), dict) else {}
        setting = [what for what, changed in (
            ("room", old_scene is not None and old_scene.get("room") != scene.get("room")),
            ("lighting", old_scene is not None and bool(refresh.get("lighting")))) if changed]
        if update["issues"]:
            self.candidates[directory.name] = {"workspace": workspace, "content": args.get("_content"),
                                               "result": update, "setting": setting}
            # The follow-up call, spelled out, so the author does not have to
            # learn the two-call protocol by failing.
            ids = [issue["id"] for issue in update["issues"] if isinstance(issue, dict) and "id" in issue]
            return {"ok": False, "candidate": directory.name, "blend": str(workspace / "animation.blend"),
                    "failure_class": "rejection",
                    "error": ("binding changes require explicit review/repair before publication. "
                              "Review `issues`, then accept them with "
                              f"build(candidate={directory.name!r}, accept_changes={ids!r}), adding "
                              "repair_code='...' if the native work needs a repair to match"),
                    **update}
        # Source may have changed while the CAD worker ran; don't publish an
        # unexpectedly old design over the author's newer source.
        self.source_fresh(workspace)
        version = self.publish(workspace, args, cancel,
                               drop_boards=self.setting_drops(workspace, setting))
        return {"ok": True, "version": version, "blend": str(self.work / "animation.blend"),
                "design_id": result["design_id"], "verifier_reports": result["verifier_reports"],
                **update, **self.setting_note(setting)}

    def setting_drops(self, workspace, setting):
        """Every state's board, when the build changed what every panel shows."""
        if not setting:
            return ()
        boards = read_json(self.work / "storyboards.json", {}) or {}
        return {s["id"] for s in plan_states(self.plan_of(workspace))} | set(boards)

    @staticmethod
    def setting_note(setting):
        if not setting:
            return {}
        return {"storyboards_dropped": (
            f"the {' and '.join(setting)} changed, and every panel shows "
            f"{'it' if len(setting) == 1 else 'them'}; draw the storyboard again")}

    def stage_sources(self, sources, workspace):
        """Put `animation_src/` where the code that reads it says it is.

        `/workspace/animation_src/<file>` has to exist, because that is the path an author writes; without it an author
        pastes the data into the code string instead, megabytes of motion data inlined as a JSON literal.

        Two hardlinked views of the same bytes, for the reason `submit()` links a workspace: the subdirectory is the
        path an author writes, and the flat copy keeps `import <author module>` resolving from the workspace root, which
        comes first on the worker's `sys.path`, ahead of `SOURCE_SUBDIR`.

        **Both views are the harness's own bytes, never the author's files.** Were the subdirectory a hardlink to
        `animation_src/`, an author rewriting a helper in place (an editor, `open(p, 'w')`, a script regenerating
        `pose_cache.json`) would change the live project with no harness call, every candidate in flight would be
        refused, and sealed bundles would stay tied to live files. An edit reaches the project at the next `blender()`.
        Each file lands on a fresh inode (`_fresh_copy`), because a running storyboard, check, or render holds a
        hardlinked snapshot of this tree.

        A name clash with harness state at the workspace root is refused, because copying over it would silently replace
        a file such as a validated `plan.json`.
        """
        clash = sorted(self.HARNESS_OWNED & {p.name for p in sources.iterdir()})
        if clash:
            raise ValueError(
                "animation_src/ is copied into the workspace root, so these names would "
                "overwrite harness state: " + ", ".join(clash) + ". Rename them; the same "
                f"files are also readable at {self.SOURCE_SUBDIR}/<name> inside the workspace.")

        def link_flat(src, dst, *, follow_symlinks=True):
            flat = workspace / Path(src).relative_to(sources)
            try:
                os.link(flat, dst)
            except OSError:
                shutil.copy2(flat, dst, follow_symlinks=follow_symlinks)

        # Never Claude's sandbox masks (`bundle.SANDBOX_MASKS`): a read-only
        # copy of one is what the next copy fails on.
        shutil.copytree(sources, workspace, dirs_exist_ok=True, copy_function=_fresh_copy,
                        ignore=skip_sandbox_masks)
        shutil.rmtree(workspace / self.SOURCE_SUBDIR, ignore_errors=True)
        shutil.copytree(sources, workspace / self.SOURCE_SUBDIR, dirs_exist_ok=True,
                        copy_function=link_flat, ignore=skip_sandbox_masks)

    def bundle_sources(self, bundle):
        """Re-sync `animation_src/` into a bundle about to be sealed.

        The tree is copied into the workspace only inside the `blender` op, and
        `finish()` packages the workspace verbatim -- so without this a bundle
        would ship whatever the *last edit* happened to carry, which can be a
        superseded `compile_result.json` reading `ok: false` and miss files
        written since, so a reviewer opening it cold would conclude the delivery
        had failed its checks.

        Into the bundle, never back into the workspace: re-syncing the
        workspace here would change the project hash the delivery receipt was
        written against, and `finish()` would refuse its own bundle. Returns
        the source hash for the manifest, so the bundle states which source it
        shipped instead of leaving it to two timestamps.
        """
        sources = self.root / "animation_src"
        if not sources.exists():
            return None
        self.stage_sources(sources, bundle)
        return tree_hash(sources)
