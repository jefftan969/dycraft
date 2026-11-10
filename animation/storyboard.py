"""The storyboard: drawing the planned states, their evidence, and the gate on animation edits."""
from __future__ import annotations

from pathlib import Path
import shutil

from .blender.bundle import tree_hash
from .blender.render_config import REVIEW_RENDER_SETTINGS, over_defaults, unsized
from ..sdk.ir import content_hash
from .context import read_json
from .planning import (appeared_states, check_items, deferred_frames, failing_at, failure_map, frame_spans,
                       gate_record, missing_states, plan_states, pose_digest, poses_comparable, segment_coverage,
                       stale_states, state_fingerprint, state_verdicts, source_gate, state_revision,
                       uncovered_frames)

class StoryboardMixin:
    def require_storyboards(self, workspace, check=None):
        """Every declared state reviewed, before any animation edit.

        A complete 3D storyboard precedes animation authoring by design:
        without one, a run tends to interpolate one camera across the whole
        film where it would otherwise cut between shots.

        A dispatched motion contributor is the exception: it is held to
        the states inside the clip it owns, and to any state whose clip is
        not yet known because it has never been drawn. Dispatch has no
        storyboard precondition, so the parent's acceptance cannot be
        assumed; but making a specialist repairing one clip re-accept every
        state of a film it did not plan buys nothing either. A never-drawn
        state is included because it has no board to name its clip, and
        leaving it out would give a child dispatched before any storyboard no
        gate at all.

        Design staleness is scoped per state (`missing_states`), which leaves
        one case the design cannot answer: an object *added* since a panel was
        reviewed and visible at its frame. `appeared_since_review` settles it
        from the whole-timeline check that has already run against this exact
        workspace, so the added case costs nothing and, without such a check,
        is simply not asked.
        """
        plan, build = self.plan_of(workspace), self.build_of(workspace)
        boards = read_json(workspace / "storyboards.json", {})
        missing = missing_states(plan, boards, build["design_id"],
                                 build.get("design_objects"))
        appeared = self.appeared_since_review(workspace, check)
        missing = sorted(set(missing) | set(appeared))
        owned = self.contribution_clip
        if owned is not None:
            missing = [name for name in missing
                       if boards.get(name, {}).get("clip") in (owned, None)]
        if missing:
            detail = "".join(
                f"\n  {name}: now shows {', '.join(appeared[name])}, "
                "which was not in the scene when it was reviewed"
                for name in missing if name in appeared)
            raise ValueError("these states need a passing, accepted storyboard before an "
                             "animation edit: " + ", ".join(missing) + detail + "\nDraw them "
                             "with h.storyboard() and accept them with h.accept_storyboard(), "
                             "or scope the edit with span= or shot=")

    def require_span(self, workspace, span, shot, *, directory=None, cancel=None):
        """The storyboard gate for an animation edit, scoped to a span.

        With neither argument this is the whole-film rule, `require_storyboards`.
        With `span=(start_state, end_state)` -- `None` meaning the film's start
        or end -- or `shot='id'`, the span's first state to its last, it is the
        same rule restricted to the span: every state that falls inside it, in
        time order and endpoints included, has been drawn and accepted, and
        every other planned state has been drawn at least once, passing or
        not.

        The unit is the stretch between two accepted states, because that is
        where motion is written. Waiting for the whole storyboard, or for whole
        shots, can hold a run with no accepted unit for hours, when the motion
        between two accepted states could already be written.

        States are placed where they are now: one `evaluate(states=)` in the
        worker, which reads the NLA, before the author's code runs; the result
        also becomes the latest `state_frames`. A state
        that no longer resolves is refused by name rather than placed where
        its board was drawn. Without a `directory` to evaluate into, states
        are placed by `placement`. Returns the span's description, or `None`
        for the whole film.
        """
        if span is None and shot is None:
            self.require_storyboards(workspace)
            return None
        if span is not None and shot is not None:
            raise ValueError("pass span=(start_state, end_state) or shot='id', not both")
        plan, build = self.plan_of(workspace), self.build_of(workspace)
        boards = read_json(workspace / "storyboards.json", {})
        ids = [s["id"] for s in plan_states(plan)]
        placement = self.placement(plan, boards)
        if directory is not None:
            info = self.worker(workspace, directory / "span_states",
                               {"op": "evaluate", "render_states": False, "place_only": True,
                                "states": [{"id": s["id"], "frame": s.get("frame")}
                                           for s in plan_states(plan)]}, cancel)
            unresolved = info.get("unresolved_states") or {}
            if unresolved:
                raise ValueError("These states cannot be placed in the current film; re-draw "
                                 "them with storyboard(states=[...]): "
                                 + "; ".join(f"{name}: {why}" for name, why in sorted(unresolved.items())))
            self.timeline_hint = {k: info.get(k) for k in self.TIMELINE_KEYS}
            placement = dict(info["state_frames"])
        if shot is not None:
            span = self.shot_span(plan, placement, shot)
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("span must be (start_state, end_state), with None for the "
                             "film's start or end")
        start, end = span
        unknown = [n for n in (start, end) if n is not None and n not in ids]
        if unknown:
            raise ValueError("unknown state in span: " + ", ".join(map(str, unknown)))
        undrawn = [n for n in ids if n not in boards]
        if undrawn:
            raise ValueError("A span needs every planned state drawn at least once, passing or "
                             "not, so it can be placed; draw these with storyboard() first: "
                             + ", ".join(undrawn))
        n_frames = (self.timeline_hint or {}).get("n_frames")
        unplaced = [n for n in ids if placement.get(n) is None
                    or (n_frames and placement[n] >= n_frames)]
        if unplaced:
            raise ValueError("These states cannot be placed in the current film (drawn past its "
                             "end, so probably before a retime): " + ", ".join(unplaced)
                             + ". Re-draw them with storyboard(states=[...]).")
        low = 0 if start is None else placement[start]
        high = None if end is None else placement[end]
        if high is not None and low > high:
            raise ValueError(f"span is reversed: {start} is at frame {low}, after {end} "
                             f"at frame {high}")
        inside = sorted((n for n in ids if placement[n] >= low
                         and (high is None or placement[n] <= high)), key=placement.get)
        missing = set(missing_states(plan, boards, build["design_id"],
                                     build.get("design_objects")))
        appeared = self.appeared_since_review(workspace)
        blocked = [n for n in inside if n in missing or n in appeared]
        owned = self.contribution_clip
        if owned is not None:
            blocked = [n for n in blocked if boards.get(n, {}).get("clip") in (owned, None)]
        if blocked:
            detail = "".join(
                f"\n  {name}: now shows {', '.join(appeared[name])}, "
                "which was not in the scene when it was reviewed"
                for name in blocked if name in appeared)
            raise ValueError("Accept these states before animating this span (they fall inside "
                             "it): " + ", ".join(blocked) + detail)
        return {"start_state": start, "end_state": end, "frames": [low, high],
                "states": inside,
                "unaccepted_elsewhere": sorted(n for n in missing | set(appeared)
                                               if n not in inside)}

    @staticmethod
    def shot_span(plan, placement, shot):
        """`shot='id'` as a span: that shot's first state to its last, in time.

        A nested plan's states name their shot. A current-shape plan's shots
        hold no states, so a shot that declares `frames: [a, b]` takes the
        states placed inside that interval.
        """
        shots = {s["id"]: s for s in plan["shots"]}
        if shot not in shots:
            raise ValueError("unknown shot: " + str(shot))
        states = plan_states(plan)
        members = [s["id"] for s in states if s.get("shot") == shot]
        if not members and shots[shot].get("frames"):
            low, high = shots[shot]["frames"]
            members = [s["id"] for s in states if placement.get(s["id"]) is not None
                       and low <= placement[s["id"]] <= high]
        if not members:
            raise ValueError(f"shot {shot!r} holds no state that can be placed; pass "
                             "span=(start_state, end_state) instead")
        ordered = sorted(members, key=lambda m: (placement.get(m) is None, placement.get(m) or 0))
        return ordered[0], ordered[-1]

    def span_report(self, span, workspace, directory, cancel, result, boards):
        """What a span edit did beyond its declaration. Reported, never refused.

        The span is a declaration, not a fence: the edit is arbitrary code, and
        an interactive edit applies in place, so nothing is undone after the
        fact. Two things are reported:

        * `outside_span_keys` -- the frames outside the span that the edit's
          keys reach, from the worker's `animation_changed` record;
        * `endpoints_moved` -- accepted endpoints whose staged pose the edit
          moved. They keep their acceptance: acceptance never compared poses,
          and an author driving states and motion from one pose function would
          otherwise loop on re-review.
        """
        report = dict(span)
        low, high = span["frames"]
        changed = result.get("animation_changed") if isinstance(result, dict) else None
        if isinstance(changed, dict):
            if changed.get("outside_span") is not None:
                report["outside_span_keys"] = changed["outside_span"]
            elif changed.get("changed_frames") is not None:
                frames = set()
                for item in changed["changed_frames"]:
                    frames.update(range(item[0], item[1] + 1) if isinstance(item, (list, tuple))
                                  else [item])
                outside = [f for f in frames if f < low or (high is not None and f > high)]
                if outside:
                    report["outside_span_keys"] = frame_spans(outside)
        ends = {name: frame for name, frame in ((span["start_state"], low), (span["end_state"], high))
                if name is not None and (boards.get(name) or {}).get("pose_digest")}
        if not ends:
            return report
        try:
            from .blender.evaluation import read_record
            output = directory / "span_endpoints"
            info = self.worker(workspace, output, {"op": "evaluate",
                                                   "frames": sorted(set(ends.values()))}, cancel)
            moved = []
            for name, frame in ends.items():
                current = pose_digest(read_record(output / "evaluation", frame),
                                      viewpoints=info.get("viewpoints"))
                approved = boards[name]["pose_digest"]
                if poses_comparable(current, approved) and current != approved:
                    moved.append(name)
            report["endpoints_moved"] = sorted(moved)
        except Exception:                                       # noqa: BLE001
            # A report, so it never costs the edit; `compile()` reports drift too.
            if cancel.is_set():
                raise
        return report

    def storyboard(self, args, workspace, directory, cancel):
        plan, build = self.plan_of(workspace), self.build_of(workspace)
        blocked_by, advisories = source_gate(build["verifier_reports"])
        states = plan_states(plan)
        by_id = {s["id"]: s for s in states}
        selected = self.selected_states(states, args)
        settings = over_defaults(args.get("render_settings"), REVIEW_RENDER_SETTINGS)
        boards, rendered = read_json(workspace / "storyboards.json", {}), {}
        drafts = read_json(self.drafts_path, {})
        content_at = self.content_version(workspace)
        seq = args.get("_seq", 0)
        # A board for a state the plan no longer declares is dead weight
        # nothing can clear: `missing_states` only walks the current list, so
        # a renamed state would leave its evidence behind forever.
        dropped = sorted(set(boards) - set(by_id))
        for stale in dropped:
            boards.pop(stale)
            shutil.rmtree(self.views_root / content_hash(stale), ignore_errors=True)

        # What the gate already considers current is not drawn again: it is
        # re-checked from the staged-frame evaluation instead. A
        # blocked design is the exception, since nothing can bank while it
        # fails, so the author gets the pictures; and `fresh=True` skips it,
        # the escape hatch for a retention that misjudges.
        kept, kept_frames = ({}, set()) if blocked_by or args.get("_fresh") else self.retainable(
            plan, build, boards, selected, settings, workspace, directory, cancel, content_at)

        # Rendered first, then checked, which is what lets a state be placed
        # inside a clip. Only the worker can read
        # the NLA, so a clip-relative state's absolute frame comes back from
        # the render that drew it, and the check then covers exactly those.
        #
        # One render group per declared camera override, which means one per
        # shot for a nested plan and one for the whole call for a current
        # one: with the camera derived from the timeline there is nothing
        # left to split the render on.
        render_groups = {}
        for state in states:
            if state["id"] in selected and state["id"] not in kept:
                render_groups.setdefault(state.get("shot") or "_", []).append(state)

        # Hashed before anything is rendered, because the panel a review was
        # written about lives at the path the new one is about to take. The
        # old code hashed the outgoing evidence before replacing the
        # directory; doing it afterwards compares an image with itself and
        # retains every review unconditionally, which is the worst available
        # failure — a stale observation presented as current coverage.
        outgoing = {name: self.image_digest(self.native / board["path"])
                    for name, board in boards.items()
                    if name in selected and name not in kept and board.get("path")
                    and (self.native / board["path"]).is_file()}

        # A current plan derives every panel's camera from the timeline, so
        # there is one render group and the check can ride along inside it. A
        # nested plan that overrides the camera per shot needs a worker each,
        # and its check stays one separate whole-storyboard call rather than
        # one per shot.
        panels, sources, merged = {}, {}, None
        fused = len(render_groups) == 1
        for owner, members in render_groups.items():
            key = content_hash(owner)
            request = {"op": "render", "render_settings": settings,
                       "fit_aspect": unsized(args.get("render_settings")),
                       "states": [{"id": s["id"], "frame": s.get("frame")} for s in members]}
            if owner != "_":
                shot = next(x for x in plan["shots"] if x["id"] == owner)
                request["camera_name"] = shot["camera"]
            if fused:
                request["evaluate"] = self.verification_spec(workspace)
            result = self.worker(workspace, directory / ("render_" + key), request, cancel)
            if fused and result.get("evaluation") is not None:
                merged = (result["evaluation"], directory / ("render_" + key))
            for index, state in enumerate(members):
                # The live panel is one file per state, keyed on the state id
                # -- which is what the stale-board sweep above removes -- and
                # `merge_boards` writes it from this job's own copy only if
                # this board wins. Storyboards overlap, and writing it here
                # gave the file to whichever finished last while the board
                # went to the later submission.
                dest = self.views_root / content_hash(state["id"]) / "000.png"
                sources[state["id"]] = directory / ("render_" + key) / result["paths"][index]
                panels[state["id"]] = {
                    "frame": result["state_frames"][state["id"]],
                    "clip": result["state_clips"].get(state["id"]),
                    "path": dest.relative_to(self.native).as_posix(),
                    "digest": self.image_digest(sources[state["id"]]),
                    # The whole view record, not only the camera's name: the
                    # matrix, lens, and sensor in it are what resolve which
                    # objects this panel frames.
                    "view": result["views"][index] or {},
                    "camera": (result["views"][index] or {}).get("camera")}

        # One check over every staged frame, not one per clip or shot.
        #
        # The verifier takes an explicit frame set, so a group buys nothing
        # the set does not, and a Blender start is the dominant cost of
        # checking a handful of scattered frames: starting a worker costs more
        # than many frames do. Per group that fixed cost would be paid once per
        # shot, and after frustum-scoped invalidation the common call is a one-
        # or two-state refresh, where startup is nearly the whole bill.
        #
        # Strictly more informative, too: `EventsSatisfied` defers an event
        # unless every frame of its window was checked, so widening the set
        # can only answer events that separate groups each had to defer.
        staged_frames = {panel["frame"] for panel in panels.values()}
        all_frames = sorted(staged_frames)
        shared = [self.check(workspace, directory / "checks_staged", cancel,
                             frames=all_frames, version=content_at,
                             evaluated=merged)] if all_frames else []
        reports = shared[0]["verifier_reports"] if shared else []
        film_frames = next((c["evaluation"].get("n_frames") for c in shared
                            if isinstance(c.get("evaluation"), dict)), None)
        # Each state is judged at its own frame, so a few failing frames do
        # not deny every state in the call its evidence. A failure that cannot
        # be placed at a frame still fails every state.
        placed, unplaced = failure_map(reports)
        design_blocks = [b["verifier"] for b in blocked_by]
        changed, moved = {}, {}
        for name, panel in panels.items():
            state = by_id[name]
            revision = state_revision(plan, state)
            # A review is an observation about an image. If the picture
            # this state produces has not changed, the observation is
            # still true of it and re-reading establishes nothing. Under
            # the current plan shape the state record carries no timing,
            # so a retime cannot rotate `revision` and cannot discard a
            # review of a pose that did not move.
            previous = boards.get(name)
            comparable = previous if (previous and previous.get("reviewed")
                                      and previous.get("state_revision") == revision
                                      and previous.get("render_settings") == settings) else None
            prior = outgoing.get(name) if comparable else None
            # One record read serves both: which objects this panel shows,
            # and the pose it shows them in. The pose is what a later
            # check compares against to say whether the film still passes
            # through the configuration this review was written about.
            record = self.evaluated_at(shared[0], panel["frame"])
            # What this panel *frames*, falling back to what the scene
            # merely contains when the camera cannot be resolved. The
            # difference is the whole value of per-state scoping: the
            # fallback set is usually the entire scene.
            framed = self.framed_instances(shared[0], record, panel.get("view"), settings)
            instances = framed if framed is not None else self.instances_in(record)
            failing = failing_at(placed, unplaced, panel["frame"])
            evidence = {"state_revision": revision, "design_id": build["design_id"],
                        "design_instances": instances,
                        # How `instances` was measured, and through which
                        # camera, so a later "what does this panel show
                        # now?" is answered the same way; see
                        # `observed_instances`.
                        "design_scope": "framed" if framed is not None else "scene",
                        "view": panel.get("view"),
                        "design_fingerprint": state_fingerprint(
                            build.get("design_objects") or {}, instances),
                        "pose_digest": pose_digest(
                            record, viewpoints=self.viewpoints_of(shared[0]))
                        if record else None,
                        "reviewed": False, "frame": panel["frame"], "path": panel["path"],
                        "render_settings": settings, "image_digest": panel["digest"],
                        "camera": panel["camera"], "clip": panel["clip"],
                        "purpose": state["purpose"],
                        "checks_pass": not failing and not design_blocks,
                        "failing_here": failing + design_blocks,
                        "verdicts": state_verdicts(reports, failing)}
            if prior is not None and prior == panel["digest"]:
                evidence.update(reviewed=True, review=comparable["review"],
                                retained_from=comparable["design_id"])
            # A retime is not silent: the state is reported where it
            # was drawn before and where it is now.
            if previous and previous.get("frame") is not None and previous["frame"] != panel["frame"]:
                moved[name] = [previous["frame"], panel["frame"]]
            rendered[name] = evidence
            # Every drawn state keeps a board, passing or not, so "drawn",
            # "clean", and "accepted" are three fields on one record rather
            # than the presence of a board. Retention carries the review,
            # never the verdict.
            changed[name] = boards[name] = {**gate_record(evidence, directory.name),
                                            "submitted": seq}
        for name, board in kept.items():
            # The board is the evidence. Nothing was drawn, so there is no new
            # digest to report and the panel on disk is the one the review was
            # written about; the verdict is the re-check's.
            changed[name] = boards[name] = {**board, "submitted": seq}
            rendered[name] = {**board, "redrawn": False}
        for name, evidence in rendered.items():
            # A refused acceptance's text, offered back and never applied:
            # only against the exact panel it was written about.
            draft = drafts.get(name)
            if (evidence.get("checks_pass") and not evidence.get("reviewed") and draft
                    and draft.get("state_revision") == evidence.get("state_revision")
                    and draft.get("image_digest") == evidence.get("image_digest")):
                evidence["draft_review"] = draft["review"]
        staged_frames |= kept_frames
        # A call that drew nothing still knows the film's length, from the
        # evaluation the retention pass just ran.
        film_frames = film_frames or (self.timeline_hint or {}).get("n_frames")
        published = self.merge_boards(workspace, args, cancel, boards=changed, drop=dropped,
                                      panels=sources)
        version = published["version"]
        retained = sorted(s for s, e in rendered.items()
                          if e.get("reviewed") and e.get("checks_pass")
                          and (e.get("retained_from") or e.get("redrawn") is False))
        rechecked = sorted(s for s in kept if s not in retained)
        banked = sorted(s for s, e in rendered.items() if e.get("checks_pass"))
        failing_states = {s: e["failing_here"] for s, e in sorted(rendered.items())
                          if not e.get("checks_pass")}
        unreviewed = sorted(s for s, e in rendered.items()
                            if e.get("checks_pass") and not e.get("reviewed"))
        offered = sorted(s for s, e in rendered.items() if e.get("draft_review"))
        # Both source lists ride on every reply, so a board emptied by the
        # design leg says why rather than coming back `ok: False` with every
        # state check green.
        steps = []
        if blocked_by:
            steps.append("No state can become coverage while the design's own checks fail: "
                         + ", ".join(design_blocks)
                         + ". Repair the design source and build() again (see blocked_by), "
                         "then re-run storyboard().")
        elif failing_states:
            steps.append("These states fail the required checks at their own frame: "
                         + "; ".join(f"{s} ({', '.join(v)})" for s, v in failing_states.items())
                         + ". Repair them and re-run storyboard(states=[...]) for them; every "
                         "other state kept its own evidence.")
        if unreviewed:
            steps.append("Inspect the rendered evidence for " + ", ".join(unreviewed)
                         + ", repair as needed, then accept_storyboard(reviews={state_id: observations}).")
        if not steps:
            steps.append("Every selected state reproduced its reviewed evidence; no re-review is needed.")
        next_step = " ".join(steps)
        # The staged-frame check is one result every drawn state shares, so it
        # travels once. Embedded in each state, the same large payload would be
        # repeated per state, enough to break the transport and fill the trace.
        shared_checks = [{k: v for k, v in c.items()
                          if k not in ("evaluation", "evaluation_path")} for c in shared]
        ok = all(boards.get(s, {}).get("checks_pass") for s in selected)
        reply = {"ok": ok, "version": version,
                 "states": rendered, "checks": shared_checks,
                 "view_root": str(self.native),
                 "retained": retained, "rechecked": rechecked, "banked": banked,
                 "needs_review": unreviewed,
                 "blocked_by": blocked_by, "advisories": advisories,
                 "clips_ready": self.clips_ready(rendered, published["boards"]),
                 "coverage": {**self.gate_coverage(staged_frames, film_frames),
                              **segment_coverage(plan, published["boards"])},
                 "next": next_step
                 + (" Kept without redrawing (design, pose, and render settings unchanged), "
                    "re-checked where they stand: " + ", ".join(sorted(kept))
                    + ". Name a camera or lighting change in invalidate_shots, or pass "
                    "fresh=True, to redraw them."
                    if kept else "")
                 + (" Your refused observations are attached as draft_review for "
                    + ", ".join(offered) + "; re-submit them only if they still describe the panel."
                    if offered else "")}
        if not ok:
            reply["failure_class"] = "check_finding"
        if moved:
            reply["moved"] = moved
        if published["superseded"]:
            # A later submission drew these too, or an edit dropped them; that stands.
            reply["superseded"] = published["superseded"]
        if published["content"] is not None and published["content"] != content_at:
            # Finished on the version it started from and merged onto a
            # project that moved meanwhile: kept, and flagged. The next
            # storyboard re-checks these through retention.
            reply.update(changed_since={"from_version": content_at,
                                        "to_version": published["content"]},
                         drawn_at=content_at, poses_may_have_moved=sorted(changed))
        reply["unmet"] = self.unmet(self.work, content=published["content"] or content_at,
                                    boards=published["boards"])
        return reply

    def retainable(self, plan, build, boards, selected, settings, workspace,
                   directory, cancel, version_at):
        """Selected states whose board is still current, re-checked rather than redrawn.

        Comparing a newly rendered PNG with the old one would pay the render before it could skip it, and an image
        digest depends on the whole scene, so any change would invalidate every panel. This asks the gate's own question
        instead: a board is current when `stale_states` does not list it (same `state_revision`, same design fingerprint
        over the instances the panel frames) and no object has appeared at its frame since it was drawn, so the gate and
        `storyboard()` agree about what counts as evidence.

        Unaccepted boards count too: a drawn state whose pose has not moved is not redrawn just to be re-checked. The
        re-check is one evaluation of the staged frames, with no render, and a kept board takes its verdict from it at
        its own frame, so a state that now passes banks without a redraw and one that now fails is marked failing. The
        staged pose must not have moved, because gate invalidation is blind to anything outside the design and staged
        poses often move after approval. A board with no `pose_digest`, or one digested under an earlier rule
        (`poses_comparable`), cannot be compared and is redrawn. Camera and lighting changes are native Blender state
        outside the design and `pose_digest`; `invalidate_shots` is the lever for them.

        A board's `frame` is where it was drawn, which a retime can leave past the film's end; such candidates are
        dropped, and a frame-range error from the probe keeps nothing, so every selected state is redrawn where it now
        is.
        """
        stale = set(stale_states(plan, boards, build["design_id"], build.get("design_objects")))
        n_frames = (self.timeline_hint or {}).get("n_frames")
        candidates = [name for name in sorted(selected)
                      if name not in stale and name in boards
                      and boards[name].get("render_settings") == settings
                      and type(boards[name].get("frame")) is int
                      and (not n_frames or boards[name]["frame"] < n_frames)
                      and boards[name].get("pose_digest") is not None
                      and boards[name].get("path")
                      and (self.native / boards[name]["path"]).is_file()]
        if not candidates:
            return {}, set()
        frames = sorted({boards[name]["frame"] for name in candidates})
        try:
            probe = self.check(workspace, directory / "checks_retained", cancel,
                               frames=frames, version=version_at)
        except Exception:
            if cancel.is_set():
                raise
            return {}, set()
        placed, unplaced = failure_map(probe["verifier_reports"])
        observed, poses = {}, {}
        viewpoints = self.viewpoints_of(probe)
        for name in candidates:
            record = self.evaluated_at(probe, boards[name]["frame"])
            if record is None:
                continue
            instances = self.observed_instances(probe, record, boards[name])
            if instances is not None:
                observed[name] = instances
            poses[name] = pose_digest(record, viewpoints=viewpoints)
        appeared = appeared_states(boards, observed)
        kept = {}
        for name in candidates:
            if name in appeared or poses.get(name) is None or poses[name] != boards[name]["pose_digest"]:
                continue
            failing = failing_at(placed, unplaced, boards[name]["frame"])
            kept[name] = {**boards[name], "checks_pass": not failing, "failing_here": failing,
                          "verdicts": state_verdicts(probe["verifier_reports"], failing),
                          "evidence_job": directory.name}
            if boards[name].get("reviewed"):
                # Retention carries the observation, never a verdict.
                kept[name]["retained_from"] = (boards[name].get("retained_from")
                                               or boards[name].get("design_id"))
        return kept, {boards[name]["frame"] for name in kept}

    @staticmethod
    def selected_states(states, args):
        """Which states this call covers: those `states=` names, or all of them."""
        names = {s["id"] for s in states}
        if args.get("states") is not None:
            chosen = set(args["states"])
            if chosen - names:
                raise ValueError("unknown storyboard state: " + ", ".join(sorted(chosen - names)))
            return chosen
        return names

    @staticmethod
    def evaluated_at(check, frame):
        """The evaluated record a check wrote for `frame`, or `None`.

        Read from the cache the check just produced, so it costs a file read
        rather than another Blender call. `None` when the record cannot be
        read, which every caller treats as "no observation available" rather
        than as a finding.
        """
        path = check.get("evaluation_path")
        if not path:
            return None
        try:
            from .blender.evaluation import read_record
            return read_record(Path(path), frame)
        except BaseException:                                   # noqa: BLE001
            return None

    @staticmethod
    def instances_in(record):
        """Scene instance names present and unhidden in an evaluated record.

        Generated objects are named `part/<instance>/<part>`, so the instance
        is the middle segment; anything else in the scene is native geometry
        with no design object behind it and is not scoped.
        """
        if record is None:
            return None
        names = set()
        for obj in record.get("objects", ()):
            if obj.get("hidden"):
                continue
            parts = str(obj.get("name") or "").split("/")
            if len(parts) >= 3 and parts[0] == "part":
                names.add(parts[1])
        return sorted(names)

    @staticmethod
    def framed_instances(check, record, view, settings):
        """Instance names this panel actually frames, or `None` if unanswerable.

        `instances_in` answers "present and unhidden", which is usually the
        *entire scene*: authors place objects out of shot rather than hiding
        them, so scoping design staleness by that set would let any
        single-object repair clear every reviewed state.

        What makes a panel's set narrow is the camera. Each body's world
        bounding box is tested against the view frustum built from the
        camera's own matrix, lens, and sensor, so an object standing behind the
        lens or off the side of frame does not hold a review hostage, and a
        background prop invalidates only the panels that see it.

        Conservative in the one direction that matters: an *occluded* object
        still counts, because the box test cannot see occlusion, so the set is
        a superset of what the pixels show and a review is never retained over
        a change that did reach the image. Orthographic and panoramic cameras,
        a missing view record, and unreadable geometry all return `None`, which
        the caller reads as "fall back to present-and-unhidden".
        """
        import numpy as np
        if record is None or not isinstance(view, dict):
            return None
        matrix = view.get("camera_matrix")
        if view.get("camera_type") != "PERSP" or not matrix:
            return None
        lens, sensor = view.get("lens_mm"), view.get("sensor_mm")
        if not lens or not sensor:
            return None
        path = check.get("evaluation_path")
        if not path:
            return None
        try:
            from .blender.evaluation import read_local, world_geometry
            geometry = world_geometry(record, read_local(Path(path), record))
        except BaseException:                                   # noqa: BLE001
            return None
        width = (settings or {}).get("width") or 1
        height = (settings or {}).get("height") or 1
        # Blender fits the sensor to the larger dimension under AUTO, which is
        # what every recorded project uses; an explicit fit is honored.
        fit = view.get("sensor_fit") or "AUTO"
        horizontal = fit == "HORIZONTAL" or (fit == "AUTO" and width >= height)
        half = np.arctan((sensor / 2.0) / lens)
        if horizontal:
            tan_x = np.tan(half)
            tan_y = tan_x * height / width
        else:
            tan_y = np.tan(half)
            tan_x = tan_y * width / height
        # A small margin so an object grazing the edge of frame is kept rather
        # than argued about; the cost of a false keep is one re-render.
        tan_x, tan_y = tan_x * 1.05, tan_y * 1.05
        m = np.asarray(matrix, dtype=float)
        rotation, eye = m[:3, :3], m[:3, 3]
        forward, up, right = -rotation[:, 2], rotation[:, 1], rotation[:, 0]
        seen = set()
        for obj in record.get("objects", ()):
            name = str(obj.get("name") or "")
            parts = name.split("/")
            if obj.get("hidden") or len(parts) < 3 or parts[0] != "part":
                continue
            if parts[1] in seen:
                continue
            try:
                vertices, _ = geometry[name]
            except BaseException:                               # noqa: BLE001
                return None
            if len(vertices) == 0:
                continue
            low, high = vertices.min(0), vertices.max(0)
            corners = np.array([[x, y, z] for x in (low[0], high[0])
                                for y in (low[1], high[1]) for z in (low[2], high[2])])
            relative = corners - eye
            depth = relative @ forward
            ahead = depth > 1e-6
            if not ahead.any():
                continue
            across = np.abs(relative[ahead] @ right)
            vertical = np.abs(relative[ahead] @ up)
            near = depth[ahead]
            if ((across <= near * tan_x) & (vertical <= near * tan_y)).any():
                seen.add(parts[1])
        return sorted(seen)

    @staticmethod
    def gate_coverage(staged, n_frames):
        """What fraction of the film this gate actually looked at.

        The gate checks staged frames and says nothing about the
        interpolation between them, which is where motion defects live: a
        storyboard can pass on every staged frame of a film whose in-between
        frames carry many findings.

        Reported, never gated. Staging more states costs time linearly and
        still cannot fail on a transition at any sample size; the structural
        answer is to drive staged states and continuous motion from one pose
        function, so a staged state is a *sample of the film* rather than a
        separate artifact.
        What this number fixes is narrower and worth fixing anyway: an author
        reading "storyboard passed" should be able to see, without doing
        arithmetic, how small a fraction of the frames it certifies.
        """
        checked = len(staged)
        if not n_frames:
            return {"frames_certified": checked}
        # The figure, and what it does and does not certify -- no workflow
        # advice, which belongs in the prompt.
        return {"frames_certified": checked, "frames_in_film": n_frames,
                "fraction": round(checked / n_frames, 4),
                "note": (f"{checked} of {n_frames} frames are staged "
                         f"({100 * checked / n_frames:.1f} %). Staging more states "
                         "cannot certify the frames between them; compile() does.")}

    def observe_storyboard(self, workspace, check):
        """What a full check can say about the storyboard, for free.

        Every whole-timeline check evaluates the staged frames along with all
        the others, so the pose each review was written about is already on
        disk by the time this runs. Reading it back costs one file read per
        state and answers the question the gate itself cannot: does the film
        still pass through the configurations that were approved?

        **Reported, never gated**, like `gate_coverage` and `clips_ready`
        beside it. Staged poses usually move after approval, whether or not
        staging and motion come from one pose function, so a gate here would
        fire on nearly every edit and teach authors to route around it. What it
        is for is to surface drift at the edit that caused it rather than at
        `final_render()` or never.

        `appeared` is the exception that does feed a gate, through
        `require_storyboards`: an object that was not in the scene when a
        panel was reviewed and is visible at its frame now is a change to the
        picture the review is about, and no fingerprint over the design can
        see it (see `missing_states`).
        """
        boards = read_json(workspace / "storyboards.json", {})
        if not boards or check.get("partial") or not check.get("evaluation_path"):
            return None
        drifted, unreadable, uncompared, unchanged, observed, moved = [], [], [], [], {}, {}
        viewpoints = self.viewpoints_of(check)
        for name, board in sorted(boards.items()):
            frame = self.resolved_frame(check, name, board)
            if frame is None:
                continue
            if board.get("frame") is not None and frame != board["frame"]:
                moved[name] = [board["frame"], frame]
            record = self.evaluated_at(check, frame)
            if record is None:
                unreadable.append(name)
                continue
            instances = self.observed_instances(check, record, board)
            if instances is not None:
                observed[name] = instances
            current = pose_digest(record, viewpoints=viewpoints)
            approved = board.get("pose_digest")
            if approved is None or current is None:
                continue
            if not poses_comparable(current, approved):
                uncompared.append(name)
            elif current != approved:
                drifted.append({"state": name, "frame": frame, "clip": board.get("clip")})
            else:
                unchanged.append(name)
        appeared = appeared_states(boards, observed)
        reviewed = [n for n, b in boards.items() if b.get("reviewed")]
        payload = {
            "states_reviewed": len(reviewed),
            # Only states whose pose was compared and matched: a state with no
            # frame, an unreadable record, or no digest is not known unchanged.
            "pose_unchanged": len([n for n in unchanged if boards[n].get("reviewed")]),
            "drifted": drifted,
            "segments": segment_coverage(self.plan_of(workspace), boards)}
        if appeared:
            payload["appeared"] = appeared
        if unreadable:
            payload["unreadable"] = unreadable
        if uncompared:
            # Digested under an earlier rule: neither drifted nor unchanged,
            # and re-staging the state records it under the current one.
            payload["uncompared"] = uncompared
        if moved:
            payload["moved"] = moved
        # No prose note. It opened "Advisory ... nothing here blocks", and
        # then the next delivery was refused for states it never mentioned
        # What blocks a clean commit is now `unmet`, beside this.
        return payload

    def appeared_since_review(self, workspace, check=None, content=None):
        """States showing an object added since their review.

        `check` is a whole-timeline check payload the caller has just run, and
        is the authoritative source when there is one. Delivery has to pass it:
        `render()` gates before it checks, so on that path the cache holds
        whatever the author's last `compile()` left and an author who added an
        object and went straight to `final_render()` would not be asked at all.

        With neither, the question goes unanswered and the gate behaves as it
        did before, rather than inventing an answer from a stale evaluation.
        """
        path = info = None
        if check is not None and not check.get("partial") and check.get("evaluation_path"):
            path, info = Path(check["evaluation_path"]), check.get("evaluation")
        else:
            key = (self.cache_key(content) if content is not None
                   else self.check_cache_key(workspace))
            cached = self.validation_cache.get(key)
            if cached:
                candidate = Path(cached["path"])
                if candidate.is_dir() and tree_hash(candidate) == cached["hash"]:
                    path, info = candidate, cached.get("evaluation")
        if path is None or not path.is_dir():
            return {}
        boards = read_json(workspace / "storyboards.json", {})
        check = {"evaluation_path": str(path), "evaluation": info}
        observed = {}
        for name, board in boards.items():
            frame = self.resolved_frame(check, name, board)
            if frame is None:
                continue
            instances = self.observed_instances(
                check, self.evaluated_at(check, frame), board)
            if instances is not None:
                observed[name] = instances
        return appeared_states(boards, observed)

    @staticmethod
    def resolved_frame(check, name, board):
        """Where state `name` is in the film `check` evaluated.

        The evaluation's own resolution when it carries one --
        `evaluate(states=)` returns `state_frames` -- and otherwise the frame
        the state's panel was drawn at. A
        board's `frame` never decides anything once the resolution is there:
        a retime moves the state and not that number.
        """
        info = check.get("evaluation") if isinstance(check.get("evaluation"), dict) else {}
        resolved = (info or {}).get("state_frames") or {}
        return resolved.get(name, board.get("frame"))

    def placement(self, plan, boards):
        """Every planned state's current frame, or `None` where none is known.

        From the latest evaluation's `state_frames` when it has them,
        then the frame the state's panel was drawn at, then a frame the plan
        pins. Used to place spans and delivery ranges in time.
        """
        resolved = (self.timeline_hint or {}).get("state_frames") or {}
        return {s["id"]: resolved.get(s["id"], (boards.get(s["id"]) or {}).get("frame", s.get("frame")))
                for s in plan_states(plan)}

    def evidence(self, workspace, *, content=None, extra=()):
        """Every check at this content version, as per-frame evidence.

        The cached whole-film check and the remembered interval checks
        (`check(remember=True)`), plus `extra` -- such as the check a receipt
        recorded, which outlives a daemon restart that empties the caches.
        """
        key = self.cache_key(content if content is not None else self.content_version(workspace))
        held = []
        if self.validation_cache.get(key):
            held.append(("whole", self.validation_cache[key]))
        held += [("interval", entry) for entry in list(self.interval_checks.get(key, {}).values())]
        return [self.evidence_entry(kind, entry) for kind, entry in held] + list(extra)

    def evidence_entry(self, kind, held):
        """One check as `check_items` reads it. `held` has `evaluation`, `reports`, `path`."""
        info, reports = held["evaluation"], held["reports"]
        frames = (set(range(info["n_frames"])) if kind == "whole"
                  else set(info.get("frames") or ()))
        placed, unplaced = failure_map(reports)
        return {"kind": kind, "frames": frames, "placed": placed, "unplaced": unplaced,
                "deferred": set() if kind == "whole" else deferred_frames(reports, frames),
                "n_frames": info.get("n_frames"), "path": held.get("path"),
                "evaluation": info, "reports": reports, "job": self.job_of(held.get("path"))}

    def job_of(self, path):
        """The job directory a check's evaluation lives in, by name."""
        try:
            return Path(path).resolve().relative_to(self.jobs.root.resolve()).parts[0]
        except (TypeError, ValueError, IndexError):
            return None

    def unmet(self, workspace, *, content=None, boards=None, frames=None, appeared=None,
              criteria=None, drawing=(), extra=()):
        """What stands between this project and a clean `finish()`.

        The one function every readiness reply uses -- `compile`,
        `check_motion`, `storyboard`, `render(delivery=True)`, `final_render`'s
        receipt and `finish()` -- so they cannot disagree. Records, each with
        a `kind`:

        * `check` -- a required verifier failing, with its frames (`None` when
          the failure is not placed at a frame). Advisory verifiers never
          appear: only `REQUIRED_VERIFIERS` count, never a `diagnostic_only`
          report.
        * `unchecked` -- frames no check at this content version covered.
        * `design` -- a failing design verifier.
        * `storyboard` -- unreviewed, appeared, or failing states; over a
          range (`frames`), the frames outside every accepted span.
        * `criterion` -- a criterion judged `fail` (only `finish()` passes them).
        * whatever the drawing recorded (`replay_mismatch`, `draft_quality`,
          `unverified_frames`).

        `frames` scopes the check and storyboard records to a range; the
        default is the whole film.
        """
        plan, build = self.plan_of(workspace), self.build_of(workspace)
        entries = self.evidence(workspace, content=content, extra=extra)
        n_frames = (next((e["n_frames"] for e in entries if e.get("n_frames")), None)
                    or (self.timeline_hint or {}).get("n_frames"))
        scope = (set(frames) if frames is not None
                 else set(range(n_frames)) if n_frames else None)
        items = check_items(entries, scope)
        items += [{"kind": "design", "verifier": b["verifier"], "repairs": b["repairs"]}
                  for b in source_gate(build["verifier_reports"])[0]]
        boards = read_json(workspace / "storyboards.json", {}) if boards is None else boards
        appeared = appeared or {}
        missing = set(missing_states(plan, boards, build["design_id"],
                                     build.get("design_objects"))) | set(appeared)
        if frames is None:
            if missing:
                item = {"kind": "storyboard", "states": sorted(missing)}
                failing = sorted(n for n in missing if (boards.get(n) or {}).get("checks_pass") is False)
                undrawn = sorted(n for n in missing if n not in boards)
                item.update({k: v for k, v in (("failing", failing), ("undrawn", undrawn),
                                               ("appeared", appeared)) if v})
                items.append(item)
        else:
            accepted = {s["id"] for s in plan_states(plan)} - missing
            outside, blamed = uncovered_frames(scope, self.placement(plan, boards), accepted)
            if outside:
                items.append({"kind": "storyboard", "states": blamed,
                              "frames": frame_spans(outside)})
        items += [{"kind": "criterion", "id": key, "evidence": value.get("evidence")}
                  for key, value in (criteria or {}).items() if value.get("verdict") == "fail"]
        return items + list(drawing)

    def observed_instances(self, check, record, board):
        """What `board`'s panel shows in `record`, measured the way it was recorded.

        `storyboard()` records the camera-framed set when it can resolve the
        camera and the whole scene when it cannot, and says which in
        `design_scope`. Comparing a framed set with the whole scene makes
        every object out of shot look newly appeared, which would refuse a
        `final_render` and defeat retention. So the current set goes through
        the panel's own recorded camera, and anything that cannot be answered
        like-for-like -- a board without `design_scope`, or a camera that no
        longer frames -- returns `None`, which `appeared_states` reads as "not
        answered" rather than as a finding.
        """
        if record is None:
            return None
        scope = board.get("design_scope")
        if scope == "framed":
            return self.framed_instances(check, record, board.get("view"),
                                         board.get("render_settings"))
        if scope == "scene":
            return self.instances_in(record)
        return None

    @staticmethod
    def viewpoints_of(check):
        """Camera and light names the check's evaluation recorded, or `None`.

        `None` for an evaluation from before the field, so `pose_digest`
        marks its digest as produced under the older rule instead of passing
        it off as comparable with one that left the cameras out.
        """
        info = check.get("evaluation")
        if not (isinstance(info, dict) and "viewpoints" in info):
            path = check.get("evaluation_path")
            info = read_json(Path(path) / "index.json", {}) if path else {}
        names = info.get("viewpoints")
        return names if isinstance(names, list) else None

    @staticmethod
    def resolve_invalidations(plan, boards, names):
        """The states `invalidate_shots=[...]` clears, or a message naming what exists.

        Coverage is keyed by state, so a name resolves to the states it
        covers: a shot id to the states that shot holds -- under the current
        plan shape, the states drawn through its camera or inside its
        declared frames -- a clip name to the states recorded as bound into
        it, or a state id to itself. Popping the
        bare name would silently clear nothing, which is the worst outcome for
        an argument whose whole job is to say "this evidence is no longer
        good".

        A clip is nameable only through the boards, because the binding lives
        in the blend and only the worker can read it -- the plan declares no
        clips. So a clip holding no staged state cannot be named. Every
        unresolved name is reported at once, with what each namespace
        currently holds, because the author cannot otherwise tell which of the
        three they missed.
        """
        states = plan_states(plan)
        shots = {s["id"] for s in plan["shots"]}
        by_shot, clips = {}, {}
        for state in states:
            if state.get("shot"):
                by_shot.setdefault(state["shot"], set()).add(state["id"])
        for name, board in boards.items():
            if board.get("clip"):
                clips.setdefault(board["clip"], set()).add(name)
            # A state of the current shape names no shot; where its board was
            # drawn does, so a shot id resolves through it.
            for shot in plan["shots"]:
                interval, frame = shot.get("frames"), board.get("frame")
                if shot.get("states") is None and (
                        board.get("camera") == shot["camera"]
                        or (interval and type(frame) is int
                            and interval[0] <= frame <= interval[1])):
                    by_shot.setdefault(shot["id"], set()).add(name)
        known = {s["id"] for s in states} | set(boards)
        covered, unknown = set(), []
        for name in names or ():
            hit = by_shot.get(name, set()) | clips.get(name, set())
            if name in known:
                hit.add(name)
            if not hit and name not in shots:
                unknown.append(name)
            covered |= hit
        if unknown:
            raise ValueError(
                "nothing to invalidate for: " + ", ".join(sorted(set(unknown)))
                + ". Nothing was edited. Shots: " + (", ".join(sorted(shots)) or "none")
                + "; clips with staged states: " + (", ".join(sorted(clips)) or "none")
                + "; states: " + (", ".join(sorted(known)) or "none")
                + ". A clip is nameable here only while a state is staged in it.")
        return covered
