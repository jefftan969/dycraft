"""Small explicit planning records; no attempt to infer arbitrary bpy intent."""
from ..sdk.ir import content_hash

DEFAULT_MIN_PIXELS = 960 * 540


def validate_plan(value):
    if not isinstance(value, dict):
        raise ValueError("plan must be a record with intent, criteria, and shots")
    if not isinstance(value.get("intent"), str) or not value["intent"].strip():
        raise ValueError("plan.intent must describe the intended performance")
    criteria, shots = value.get("criteria"), value.get("shots")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("register the consequential functional and visual criteria")
    for criterion in criteria:
        if not isinstance(criterion, dict) or any(
                not isinstance(criterion.get(k), str) or not criterion[k].strip()
                for k in ("id", "requirement", "why", "observe")):
            raise ValueError("each criterion needs id, requirement, why, and observe")
    if len({c["id"] for c in criteria}) != len(criteria):
        raise ValueError("criterion ids must be unique")
    if not isinstance(shots, list) or not shots:
        raise ValueError("plan every shot before animation authoring")
    # Two shapes. The current one declares `states` at the top level, ordered
    # and without frames: a state is a configuration the film passes through,
    # and where it sits is bound in the blend, clip-relative, so retiming
    # cannot invalidate a review of it. The older one nests states inside
    # shots with absolute frames; it still validates and behaves exactly as
    # before, because a project mid-flight must not have to be rewritten.
    # Declaring both is rejected rather than merged: which one located a
    # given state would then be a precedence rule nobody asked for.
    top_level = value.get("states")
    nested = [s for s in shots if isinstance(s, dict) and s.get("states") is not None]
    if top_level is not None and nested:
        raise ValueError("declare states at the top level or inside shots, not both")
    if top_level is not None:
        if not isinstance(top_level, list) or not top_level:
            raise ValueError("plan.states must be the ordered configurations the film passes through")
        for state in top_level:
            if not isinstance(state, dict) or any(
                    not isinstance(state.get(k), str) or not state[k].strip()
                    for k in ("id", "purpose")):
                raise ValueError("each state needs a unique id and a purpose saying what it exposes")
            # Optional. A state with no frame is located in the blend by
            # `dycraft_states`; one with a frame is pinned to an absolute
            # output index, which stays available for a film with no clips.
            if "frame" in state and (type(state["frame"]) is not int or state["frame"] < 0):
                raise ValueError("state.frame, when given, is a zero-based output frame index")
        if len({s["id"] for s in top_level}) != len(top_level):
            raise ValueError("state ids must be unique")
    for shot in shots:
        if not isinstance(shot, dict) or any(
                not isinstance(shot.get(k), str) or not shot[k].strip()
                for k in ("id", "intent", "camera")):
            raise ValueError("each shot needs id, intent, and a native camera name")
        if top_level is None and (not isinstance(shot.get("states"), list) or not shot["states"]):
            raise ValueError("plan.states is required: the ordered configurations the film "
                             "passes through, each with an id and a purpose")
        for state in shot.get("states") or ():
            if not isinstance(state, dict) or type(state.get("frame")) is not int or state["frame"] < 0:
                raise ValueError("state.frame must be a zero-based output frame index")
            if not isinstance(state.get("purpose"), str) or not state["purpose"].strip():
                raise ValueError("state.purpose must say what the staging exposes")
        # Optional, and only ever resolved — never enforced. Declaring it lets
        # tools take `shot='id'` instead of an interval; it does not create
        # timeline markers, schedule cuts, or assert anything about the blend.
        interval = shot.get("frames")
        if interval is not None:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(f) is not int for f in interval)
                    or interval[0] < 0 or interval[1] < interval[0]):
                raise ValueError("shot.frames must be an inclusive [start, end] pair of "
                                 "zero-based output frame indices")
            outside = [s["frame"] for s in shot.get("states") or ()
                       if not interval[0] <= s["frame"] <= interval[1]]
            if outside:
                raise ValueError(f"shot {shot['id']!r} states at {outside} fall outside its "
                                 f"declared frames {list(interval)}")
    if len({s["id"] for s in shots}) != len(shots):
        raise ValueError("shot ids must be unique")
    delivery = value.get("delivery", {})
    if not isinstance(delivery, dict):
        raise ValueError("delivery must be a record")
    pixels = delivery.get("min_pixels", DEFAULT_MIN_PIXELS)
    if type(pixels) is not int or pixels < 4:
        raise ValueError("delivery.min_pixels must be a positive pixel count")
    if pixels != DEFAULT_MIN_PIXELS and not delivery.get("reason"):
        raise ValueError("record the task's reason for a non-default delivery target")
    return {**value, "id": content_hash({k: v for k, v in value.items() if k != "id"})}


def shot_interval(plan, shot_id):
    """The inclusive output-frame interval a shot declared, for tools that take one."""
    for shot in plan["shots"]:
        if shot["id"] == shot_id:
            interval = shot.get("frames")
            if interval is None:
                raise ValueError(f"shot {shot_id!r} declares no frames interval; add "
                                 "frames: [start, end] to it in plan(), or pass start/end")
            return int(interval[0]), int(interval[1])
    raise ValueError("unknown shot: " + str(shot_id))


def plan_states(plan):
    """Every storyboard state in order, whichever shape the plan declares.

    One canonical record per state: `id`, `purpose`, an optional absolute
    `frame`, and for a nested plan the `shot` whose camera renders it. A
    nested plan's states have no ids of their own, so they take `shot/index`
    — stable under everything except reordering the states within a shot,
    which is the same thing as declaring different states.
    """
    if plan.get("states") is not None:
        return [dict(state) for state in plan["states"]]
    states = []
    for shot in plan["shots"]:
        for index, state in enumerate(shot.get("states") or ()):
            states.append({**state, "id": f"{shot['id']}/{index}", "shot": shot["id"]})
    return states


def state_revision(plan, state):
    """What a review was written about: the state record, and nothing else.

    A state under the current shape carries no timing at all, so retiming a
    clip cannot rotate this hash and cannot discard a review of a picture
    that has not changed. A nested state still carries its absolute `frame`,
    so a nested plan keeps its frame-based behavior.

    The plan's `criteria` and `intent` are left out: with them in, adding a
    criterion or rewording the intent would void every review while no panel
    changed. A review is about the panel; criteria are judged once, at
    `finish()`. The board carries what else a panel depends
    on (`design_id`, `pose_digest`, the render settings). `plan` stays in the
    signature so callers need not know.
    """
    return content_hash({"state": state})


def state_fingerprint(design_objects, instances):
    """A state's design identity, over only the objects it actually shows.

    `instances` are the scene instance names present and unhidden at the
    state's frame -- a superset of "in shot", because `hide_render` is what
    the evaluated record carries and the frustum is not. Conservative in the
    right direction: an object off the side of frame still counts.

    Returns `None` when the instance list is unknown, which reads as "fall
    back to the whole-build id".
    """
    if instances is None:
        return None
    return content_hash({"objects": sorted(
        (name, design_objects.get(name)) for name in instances)})


def missing_states(plan, boards, design_id, design_objects=None):
    """States without current reviewed evidence, in plan order.

    Design staleness is judged per state where the board recorded which
    instances its panel contained, and over the whole build otherwise.

    Scoping covers all three shapes of design change, because the fingerprint
    is taken over `design_objects.get(name)` for the state's *own* instances:

    * **modified** -- the handle under a recorded name changes, so exactly the
      states that show it go stale;
    * **removed** -- the lookup returns `None` for states that showed it and
      is untouched for every other state, so removal scopes for free;
    * **added** -- no board recorded the new instance, so nothing rotates
      here. That case cannot be answered from the design alone: whether a new
      object shows up in a panel depends on where it sits and what is hidden
      at that frame, which only an evaluation knows. `appeared_states()`
      answers it from the next check, which already evaluates those frames.

    Comparing the whole instance *set* first would make every add and remove
    clear every board, and redraw states that do not show the changed
    object, to cover only the added-object case, which the next check covers.
    """
    stale = set(stale_states(plan, boards, design_id, design_objects))

    def unaccepted(board):
        # A board drawn failing at its own frame is evidence, but not
        # acceptable evidence.
        return not board.get("reviewed") or board.get("checks_pass") is False

    return [s["id"] for s in plan_states(plan)
            if s["id"] in stale or unaccepted(boards.get(s["id"], {}))]


def stale_states(plan, boards, design_id, design_objects=None):
    """States whose board no longer describes the state or the design it shows.

    `missing_states` without its "reviewed and passing" half, which is what
    retention needs: a board that is current but not yet accepted can be
    re-checked rather than redrawn. A state with no board is stale.
    """
    def stale(state, board):
        if board.get("state_revision") != state_revision(plan, state):
            return True
        recorded = board.get("design_instances")
        if design_objects is not None and recorded is not None:
            return board.get("design_fingerprint") != state_fingerprint(design_objects, recorded)
        return board.get("design_id") != design_id

    return [s["id"] for s in plan_states(plan) if stale(s, boards.get(s["id"], {}))]


def appeared_states(boards, observed):
    """Boards whose frame now shows an instance that was not there when reviewed.

    The other half of add-scoping. `observed` maps a state id to the instance
    names present and unhidden at its frame in a fresh evaluation; a state
    missing from it is simply not
    answered here and keeps whatever `missing_states` decided.

    Only *additions* count. A disappearance is already caught by the
    fingerprint when the object left the design, and a state whose instance
    merely became hidden is a staging change the pose observation reports --
    invalidating a review over it would put this function back in the
    business of clearing boards nobody asked it to clear.
    """
    appeared = {}
    for name, board in boards.items():
        recorded = board.get("design_instances")
        seen = observed.get(name)
        if recorded is None or seen is None:
            continue
        extra = sorted(set(seen) - set(recorded))
        if extra:
            appeared[name] = extra
    return appeared


#: How finely a staged pose is recorded before it is hashed, in meters and
#: radians. Movement below this does not rotate `pose_digest`, so the
#: observation reports a pose as unchanged when nothing moved further than
#: this. Well under the millimeter an author would call a change, and well
#: above float32's resolution at room scale, so an unchanged rig re-evaluated
#: hashes the same.
POSE_QUANTUM = 1e-4


def pose_digest(record, quantum=POSE_QUANTUM, viewpoints=None):
    """The staged pose at one frame, quantized and hashed.

    Every object's placement and every joint coordinate, and deliberately
    nothing else: color, visibility, and mesh identity belong to
    `state_fingerprint` and to the image digest, and folding them in here
    would make an appearance pass read as a pose change. That separation is
    the whole point: most authored edits touch keyframes, materials, and camera
    in the same call, so a signal that cannot tell those apart reports drift on
    nearly every edit.

    `viewpoints` names the cameras and lights, which the evaluation lists in
    its `index.json` and which are left out: they frame the pose, they are not
    part of it. With them in, moving one camera would report nearly every other
    state as drifted. The version prefix says which rule produced a
    digest, so `poses_comparable` never reads a rule change as movement.

    A skinned character's pose is its bones: its object frame holds still and
    it has no joints, so a record's `bone_rotations` (each armature's evaluated
    bone quaternions) are part of the pose too, quantized the same way. A
    digest that counts bones carries its own rule (`pose1b`/`pose2b`), so a
    board digested before bones were counted reads as "no observation" rather
    than as unchanged; a record with no armature digests exactly as before.

    Quantized rather than exact so the comparison has a tolerance at all: a
    digest cannot be compared with a threshold, so the threshold has to go in
    before the hash. `None` when the record carries no placements, which reads
    as "no observation available" rather than as "unchanged".
    """
    transforms = record.get("transforms") or {}
    if not transforms:
        return None
    skip = set(viewpoints or ())
    def q(value):
        return int(round(float(value) / quantum))
    pose = {
        "transforms": sorted((name, [q(v) for v in _flatten(matrix)])
                             for name, matrix in transforms.items() if name not in skip),
        "joints": sorted((name, q(value))
                         for name, value in (record.get("joint_positions") or {}).items())}
    bones = record.get("bone_rotations") or {}
    if bones:
        pose["bones"] = sorted((name, [q(v) for v in _flatten(rows)])
                               for name, rows in bones.items())
    rule = ("pose2" if viewpoints is not None else "pose1") + ("b" if bones else "")
    return rule + ":" + content_hash(pose)


def evaluation_digest(info):
    """What a `compile()` reply says about its evaluation, instead of all of it.

    The full index -- every declaration, clip span, collision object, and
    evaluated frame -- would be tens of kilobytes on every compile, a third of a
    reply, all of it the author's own input. The index is unchanged on disk at `evaluation_path` and in the persisted
    report, so this keeps the counts and the timeline facts a reply is read
    for.
    """
    if not isinstance(info, dict):
        return info
    digest = {key: info[key] for key in ("scene_ref", "fps", "n_frames", "frame_start",
                                         "complete", "blender_version") if key in info}
    declarations = info.get("declarations")
    if isinstance(declarations, dict):
        digest["declarations"] = {
            "events": len(declarations.get("events") or ()),
            "collision_allowances": len(declarations.get("collision_allowances") or ()),
            "liquids": len(declarations.get("liquids") or ()),
            "units": declarations.get("units")}
    for key in ("clips", "collision_objects", "frames"):
        if isinstance(info.get(key), (list, dict)):
            digest[key] = len(info[key])
    digest["full_index"] = "index.json under evaluation_path"
    return digest


def poses_comparable(current, approved):
    """Whether two digests were produced by the same rule, so `!=` means movement.

    An unprefixed digest predates versioning and is its own rule. A pair from
    different rules is neither drift nor unchanged: the caller treats it as
    "no observation", and retention re-draws the state rather than keep it.
    """
    if current is None or approved is None:
        return False
    def rule(digest):
        return digest.split(":", 1)[0] if ":" in digest else "pose0"
    return rule(current) == rule(approved)


def _flatten(matrix):
    for row in matrix:
        if isinstance(row, (list, tuple)):
            yield from _flatten(row)
        else:
            yield row


def segment_coverage(plan, boards):
    """How much of the film sits between two approved states.

    The unit an author actually reasons about: motion is authored *between*
    two staged poses, so a span whose endpoints are both reviewed is a span
    whose ends were agreed. Reported next to `frames_certified`, which counts
    the staged samples themselves and so reads as a discouraging few percent
    however well the storyboard is doing.

    Consecutive in plan order, and only within one clip -- two states either
    side of a cut do not bracket a continuous motion, and pairing them would
    claim coverage across an edit nobody staged.
    """
    states = plan_states(plan)
    segments, ready = [], 0
    for first, second in zip(states, states[1:]):
        a, b = boards.get(first["id"], {}), boards.get(second["id"], {})
        if a.get("clip") is not None and a.get("clip") != b.get("clip"):
            continue
        covered = bool(a.get("reviewed") and b.get("reviewed"))
        ready += covered
        segments.append({"from": first["id"], "to": second["id"], "covered": covered,
                         "frames": sorted(f for f in (a.get("frame"), b.get("frame"))
                                          if f is not None)})
    return {"segments": len(segments), "covered": ready, "detail": segments}


#: Board fields with a reader. The first three are what `missing_states`
#: compares; the rest are what `storyboard()` needs to decide whether a
#: re-render reproduced the evidence a review was written about.
#:
#: `frame` is where the panel was drawn, and nothing decides on it: a
#: state's current time is resolved from the blend, because a retime moves the
#: state and not this number. `checks_pass` and `failing_here` are this
#: state's own verdict at that frame, not the call's.
GATE_FIELDS = ("state_revision", "design_id", "reviewed", "review",
               "frame", "path", "render_settings", "image_digest",
               "camera", "clip", "retained_from", "pose_digest",
               "design_instances", "design_fingerprint", "design_scope", "view",
               "checks_pass", "failing_here")


def gate_record(evidence, job):
    """The durable part of a state's evidence: what a later call actually reads.

    `storyboard()` returns the full check payload to its caller, because an
    author repairing a failure needs the failing pairs and frames. None of
    that is read again. The board is consulted by `missing_states`, by
    `require_storyboards` through it, and by retention — and between them they
    touch the fields above and nothing else.

    Storing the payload instead would put a full check record per planned
    state on the boards, hundreds of megabytes over a run (a single
    `EventsSatisfied` report can be a megabyte), in a file that lives in the
    workspace, so `submit()` would copy it and `publish()` hash it on every
    later operation, not just storyboard ones. `job` names the directory that still holds the whole thing, so
    nothing is lost — only moved to where every other check payload already
    lives.
    """
    record = {k: evidence[k] for k in GATE_FIELDS if k in evidence}
    record["checks_pass"] = bool(evidence["checks_pass"])
    record["failing_here"] = list(evidence.get("failing_here") or [])
    # Per state: a required verifier failing elsewhere in the call reads
    # `pass` here, because it does not fail at this frame.
    record["verdicts"] = dict(evidence.get("verdicts") or {})
    record["evidence_job"] = job
    return record


#: The verifiers whose verdicts decide a film. Everything else -- continuity,
#: support, slip, anatomical limits -- is advisory and never gates.
REQUIRED_VERIFIERS = ("AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied")

#: Verdicts a required verifier passes with. `not_applicable` is for
#: a verifier whose set to check is empty by construction (no joints, no
#: events, no enclosing walls) says so instead of reading as a pass.
PASSING = frozenset({"pass", "not_applicable"})


def required_checks_pass(reports):
    by_name = {r["name"]: r for r in reports}
    return set(REQUIRED_VERIFIERS).issubset(by_name) and all(
        by_name[n]["verdict"] in PASSING for n in REQUIRED_VERIFIERS)


def _spans(runs):
    """Frames named by inclusive `[lo, hi]` runs, or `None` if any is malformed."""
    frames = set()
    for run in runs or ():
        if (not isinstance(run, (list, tuple)) or len(run) != 2
                or any(type(f) is not int for f in run) or run[1] < run[0]):
            return None
        frames.update(range(run[0], run[1] + 1))
    return frames


def failing_frames(report):
    """The frames a failing required verifier fails at, or `None` if it cannot say.

    Read from each verifier's own evidence: `NoPenetration`'s pair spans and
    room escapes, `JointLimits`' per-subject frame runs, a failing event's
    window, and `AnimationWellFormed`'s per-frame mesh findings. A report's
    uniform `evidence.frames_failed` is preferred when it carries one.

    `None` is "this failure is not placed at a frame" -- an invalid event
    declaration, a frozen scene identity that changed, a truncated finding
    list -- and the caller treats it as failing everywhere it looked.
    """
    evidence = report.get("evidence") or {}
    if evidence.get("frames_failed") is not None:
        return _spans(evidence["frames_failed"])
    name = report["name"]
    if name == "NoPenetration":
        runs = [span.get("frames") for pair in evidence.get("pairs") or ()
                for span in pair.get("spans") or ()]
        runs += [run for row in evidence.get("out_of_room") or () for run in row.get("frames") or ()]
        return _spans(runs) or None
    if name == "JointLimits":
        subjects = [subject for finding in evidence.get("findings") or ()
                    for subject in finding.get("subjects") or ()]
        if not subjects or any(not subject.get("frames") for subject in subjects):
            return None
        return _spans([run for subject in subjects for run in subject["frames"]]) or None
    if name == "EventsSatisfied":
        rows = evidence.get("events")
        failing = [row for row in rows if row.get("ok") is False] if isinstance(rows, list) else []
        if not failing or any("window_frames" not in row for row in failing):
            return None
        return _spans([row["window_frames"] for row in failing]) or None
    if name == "AnimationWellFormed":
        failures = evidence.get("failures") or []
        if (not failures or evidence.get("failure_count", len(failures)) > len(failures)
                or any(type(f.get("frame")) is not int for f in failures)):
            return None
        return {f["frame"] for f in failures}
    return None


def failure_map(reports):
    """`(placed, unplaced)` over the required verifiers of one check.

    `placed` maps a failing verifier to the frames it fails at; `unplaced`
    names the ones whose failure no frame can be given for, which therefore
    fail every frame the check covered. A report marked `diagnostic_only`
    never counts, whatever its verdict. A required verifier missing from the
    reports is unplaced, as `required_checks_pass` already treats it.
    """
    by_name = {r["name"]: r for r in reports}
    placed, unplaced = {}, []
    for name in REQUIRED_VERIFIERS:
        report = by_name.get(name)
        if report is None:
            unplaced.append(name)
            continue
        if (report.get("evidence") or {}).get("diagnostic_only") or report["verdict"] in PASSING:
            continue
        frames = failing_frames(report) if report["verdict"] == "fail" else None
        if frames is None:
            unplaced.append(name)
        else:
            placed[name] = frames
    return placed, unplaced


def failing_at(placed, unplaced, frame):
    """The required verifiers failing at one frame, given `failure_map`'s result."""
    return sorted(set(unplaced) | {name for name, frames in placed.items() if frame in frames})


def source_gate(reports):
    """`(blocked_by, advisories)` over the design's own verifier reports.

    Only `fail` blocks. `uncertain` is the suite's advisory verdict:
    `JointOriginNearBothBodies` returns it on violation by design, "without
    blocking commit", and the object layer's own gate is `!= "fail"`. A gate
    requiring `pass` would let one advisory empty every board, refusing a
    storyboard with no failing state check and nothing in the reply to say
    why. The advisories still come back, with the verifier's own repair text, so the author sees
    the heuristic without being stopped by it.
    """
    def entry(report):
        return {"verifier": report["name"], "verdict": report["verdict"],
                "repairs": report.get("suggested_repairs") or []}
    return ([entry(r) for r in reports if r["verdict"] == "fail"],
            [entry(r) for r in reports if r["verdict"] == "uncertain"])


def state_verdicts(reports, failing):
    """One check's verdicts as they stand at one state's frame.

    A required verifier failing elsewhere in the check passes here; one in
    `failing` (from `failing_at`) fails. Advisory verdicts are the check's.
    """
    return {r["name"]: ("fail" if r["name"] in failing
                        else "pass" if r["name"] in REQUIRED_VERIFIERS and r["verdict"] == "fail"
                        else r["verdict"])
            for r in reports}


def frame_spans(frames):
    """Frame indices as inclusive `[lo, hi]` runs, the shape verifiers report."""
    spans = []
    for frame in sorted(set(frames)):
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def deferred_frames(reports, frames):
    """Frames a partial check left an event unanswered on, so it certifies them for nothing.

    `EventsSatisfied` defers an event whose window the check did not cover
    whole. Its rows name each deferred window; when every event deferred the
    report collapses to a count and names none, and then no frame of the check
    can be vouched for.
    """
    events = next((r for r in reports if r["name"] == "EventsSatisfied"), None)
    rows = ((events or {}).get("evidence") or {}).get("events")
    if isinstance(rows, dict):
        return set(frames) if rows.get("deferred") else set()
    out = set()
    for row in rows or ():
        if row.get("deferred") and row.get("window_frames"):
            lo, hi = row["window_frames"]
            out.update(f for f in frames if lo <= f <= hi)
    return out


def uncovered_frames(frames, placement, accepted):
    """The frames outside every accepted span, and the states responsible.

    A frame lies in an accepted span when every state between the nearest
    state at or before it and the nearest at or after it is accepted; the
    film's start and end count as accepted, for lead-in and lead-out. A state
    with no known frame cannot be placed, so nothing is covered until it is
    drawn. `placement` maps every planned state to its frame or `None`.
    """
    unplaced = sorted(name for name, frame in placement.items() if frame is None)
    if unplaced:
        return set(frames), unplaced
    ordered = sorted((frame, name) for name, frame in placement.items())
    out, blamed = set(), set()
    for f in frames:
        low = max((frame for frame, _ in ordered if frame <= f), default=None)
        high = min((frame for frame, _ in ordered if frame >= f), default=None)
        between = [n for frame, n in ordered
                   if (low is None or frame >= low) and (high is None or frame <= high)]
        bad = [n for n in between if n not in accepted]
        if bad:
            out.add(f)
            blamed.update(bad)
    return out, sorted(blamed)


def check_items(entries, scope):
    """`unmet` records for required-check failures and unchecked frames over `scope`.

    `entries` are the checks at one content version (`AnimationSession.evidence`).
    A frame any of them failed is listed under its verifier; a failure no
    frame can be given for is listed with `frames: None`; a frame none of
    them covered (or covered with an event deferred) is `unchecked`. `scope`
    is the frames asked about, or `None` for a film of unknown length.
    """
    items, covered, failing, unplaced = [], set(), {}, {}
    for entry in entries:
        frames = entry["frames"] if scope is None else entry["frames"] & scope
        if scope is not None and not frames:
            continue
        covered |= frames - entry["deferred"]
        for name in entry["unplaced"]:
            unplaced.setdefault(name, set()).add(entry.get("job"))
        for name, bad in entry["placed"].items():
            failing.setdefault(name, set()).update(bad & frames)
    for name in REQUIRED_VERIFIERS:
        if name in unplaced:
            items.append({"kind": "check", "verifier": name, "frames": None,
                          "jobs": sorted(job for job in unplaced[name] if job)})
        if failing.get(name):
            items.append({"kind": "check", "verifier": name, "frames": frame_spans(failing[name])})
    if scope is None:
        if not entries:
            items.append({"kind": "unchecked", "frames": "all"})
    elif scope - covered:
        items.append({"kind": "unchecked", "frames": frame_spans(scope - covered)})
    return items
