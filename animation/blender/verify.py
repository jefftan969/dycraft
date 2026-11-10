"""Checks on Blender's evaluated meshes, independent of the author's code."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path

import numpy as np

try:
    from .continuity import MotionContinuity, frame_runs
    from .evaluation import load_frame
    from ...verify._inherited_allowances import InheritedAllowances
    from .verify_declarations import (allowance_limit, allowed, clip_offset, declaration_repair, declared_units,
                                      frame_indices, inherited_allowances, resolve_declarations, resolve_liquids,
                                      validate_declarations, window_indices)
    from .verify_room import (ROOM_SURFACE_TAGS, escaped_faces, out_of_room, replaced_surfaces,
                              resolve_room_surfaces, room_containment, room_shell)
    from .verify_anatomy import (BODY_LIMITS, HYPEREXTENSION_DEG, ROLL_LIMITS, ROLL_RANGE, anatomical_measure,
                                 body_findings, body_measures, bone_anatomy, bone_anatomy_rows, current_anatomy,
                                 rig_anatomy_table)
    from .verify_support import (_footprint, _supported_margin, _supported_margin_held, contact_patch_outline,
                                 contact_patch_xy, ground_candidates_many, ground_index, part_mass_table,
                                 slip_evidence, slip_tracks, support_evidence, support_margin, support_samples,
                                 world_verts)
    from .verify_penetration import (_measure_diagnostics, _penetration_evidence, allowance_overrun_repair,
                                     allowance_overruns, allowed_contacts, deepest_vertex_m, fill_diagnostics,
                                     pairs_by_depth, penetration_pairs, penetration_repair)
except ImportError:  # Read-only modules staged into isolated Blender.
    from continuity import MotionContinuity, frame_runs
    from evaluation import load_frame
    from _inherited_allowances import InheritedAllowances
    from verify_declarations import (allowance_limit, allowed, clip_offset, declaration_repair, declared_units,
                                     frame_indices, inherited_allowances, resolve_declarations, resolve_liquids,
                                     validate_declarations, window_indices)
    from verify_room import (ROOM_SURFACE_TAGS, escaped_faces, out_of_room, replaced_surfaces,
                             resolve_room_surfaces, room_containment, room_shell)
    from verify_anatomy import (BODY_LIMITS, HYPEREXTENSION_DEG, ROLL_LIMITS, ROLL_RANGE, anatomical_measure,
                                body_findings, body_measures, bone_anatomy, bone_anatomy_rows, current_anatomy,
                                rig_anatomy_table)
    from verify_support import (_footprint, _supported_margin, _supported_margin_held, contact_patch_outline,
                                contact_patch_xy, ground_candidates_many, ground_index, part_mass_table,
                                slip_evidence, slip_tracks, support_evidence, support_margin, support_samples,
                                world_verts)
    from verify_penetration import (_measure_diagnostics, _penetration_evidence, allowance_overrun_repair,
                                    allowance_overruns, allowed_contacts, deepest_vertex_m, fill_diagnostics,
                                    pairs_by_depth, penetration_pairs, penetration_repair)


def joint_position(record, joint):
    """Recover a mechanism coordinate without discarding authored revolutions.

    Read from geometry: the joint Empty's transform relative to its origin,
    which must be a pure motion along or about `joint["axis"]` (local Z, for a
    project imported since joints became native channels). The evaluated
    channel value in `joint_positions` disambiguates turns when it agrees with
    the geometry; otherwise the limits must select exactly one equivalent
    angle.
    """
    transforms = record["transforms"]
    relative = np.linalg.solve(np.asarray(transforms[joint["origin"]]),
                               np.asarray(transforms[joint["object"]]))
    axis = np.asarray(joint["axis"], dtype=float)
    target = np.eye(4)
    if joint["type"] == "prismatic":
        q = float(relative[:3, 3] @ axis)
        target[:3, 3] = q * axis
    else:
        r = relative[:3, :3]
        sin = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0],
                        r[1, 0]-r[0, 1]]) @ axis / 2
        principal = math.atan2(sin, (np.trace(r)-1)/2)
        hint = record.get("joint_positions", {}).get(joint["object"])
        if hint is not None and (isinstance(hint, bool) or not math.isfinite(float(hint))):
            raise ValueError("nonfinite or invalid evaluated joint q")
        if hint is not None and abs(math.remainder(float(hint)-principal, math.tau)) <= 1e-5:
            q = float(hint)
        else:
            lo, hi = joint["limits"]
            first = math.ceil((lo-1e-5-principal)/math.tau)
            last = math.floor((hi+1e-5-principal)/math.tau)
            if first != last:
                raise ValueError("joint angle is outside limits or its turn count is ambiguous; key the joint's own channel (key_joint) so its value says the turn")
            q = principal + first*math.tau
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        target[:3, :3] += math.sin(q)*skew + (1-math.cos(q))*(skew@skew)
    if not np.isfinite(relative).all() or not np.allclose(relative, target, atol=1e-5):
        raise ValueError("joint transform violates its original motion axis/frame")
    return q


#: Every verifier an animation check reports, in the order `aggregate` emits
#: them in `verifier_reports`.
VERIFIER_NAMES = ("AnimationWellFormed", "JointLimits", "AnatomicalJointLimits",
                  "NoPenetration", "EventsSatisfied", "MotionContinuity",
                  "SupportMargin", "ContactSlip")


def report(name, failures, *, list_failures=True, advisory=False, repairs=(),
           not_applicable=None, **evidence):
    """A verifier report; `failures` decides the verdict whether or not it ships.

    `list_failures=False` keeps the verdict and `failure_count` but drops the
    per-item list, for a verifier whose own evidence already carries the same
    items in a better shape. `NoPenetration` is the case: its `pairs` are the
    merged, complete form of exactly what `failures` truncated at 100.

    `advisory=True` caps the verdict at `uncertain` and marks the evidence
    `diagnostic_only`, the contract `MotionContinuity` already returns by
    hand: a finding worth reading that is not a defect, and that
    `required_checks_pass` does not consult. Without it a verifier outside the
    required set still reports `fail`, which reads as blocking to everything
    that renders a verdict even though nothing gates on it.

    `not_applicable` is a reason, for a check with nothing to check by
    construction -- no joints, no events, no two adjacent frames. With no
    failures the verdict is then `not_applicable` rather than a `pass` that
    reads the same as one that looked; gates treat the two alike.
    """
    if not_applicable and not failures:
        return {"name": name, "verdict": "not_applicable",
                "evidence": {"failure_count": 0, "not_applicable": not_applicable},
                "suggested_repairs": []}
    evidence = {"failure_count": len(failures), **evidence}
    if advisory:
        evidence = {"diagnostic_only": True, **evidence}
    # An empty list beside a zero count is the count said twice, and passing
    # reports are most of what this layer emits.
    if list_failures and failures:
        evidence = {"failures": failures[:100], **evidence}
    verdict = ("uncertain" if advisory else "fail") if failures else "pass"
    # `repairs` is for a verifier whose findings imply a *different* repair
    # than its verdict suggests -- an unresolved event needs its declaration
    # fixed, not its geometry moved. Kept as a keyword rather than letting
    # callers post-patch the returned dict, so `verify()`'s return list stays
    # a list of `report(...)` calls and the suite that reads the emitted names
    # out of it by AST keeps working.
    return {"name": name, "verdict": verdict,
            "evidence": evidence, "suggested_repairs": list(repairs)}


def joint_findings(rows):
    """`JointLimits` rows regrouped: one entry per reason, naming its subjects.

    Five different findings share this verifier's list — a mechanism frame
    that changed, one that went missing or singular, an articulated visual
    moved outside its mechanism, a joint driven past its limits, and a joint
    whose angle could not be resolved at all. They arrive keyed `object`,
    `identity`, or `joint` depending on which, so a reader has to branch on
    which keys are present, and only the limit violation carries a
    measurement.

    Flat, they also repeat: six joints failing at one frame produced six rows
    carrying the same 88-character reason. Hoisting the reason and grouping
    its subjects by frame span says the same thing once, and surfaces the
    `q` against `limits` comparison where the row has one — which is the
    only part a repair can be sized from.
    """
    by_reason = {}
    for row in rows:
        subject = row.get("joint") or row.get("object") or row.get("identity") or "?"
        entry = by_reason.setdefault(row.get("reason", "unspecified"), {})
        held = entry.setdefault(subject, {"frames": [], "q": []})
        if row.get("frame") is not None:
            held["frames"].append(row["frame"])
        if "q" in row:
            held["q"].append(row["q"])
            held["limits"] = row.get("limits")
    findings = []
    counts = {}
    for reason, subjects in by_reason.items():
        items = []
        for subject, held in subjects.items():
            frames = sorted(set(held["frames"]))
            item = {"subject": subject, "frames": frame_runs(frames)}
            if held["q"]:
                item["q_range"] = [round(min(held["q"]), 6), round(max(held["q"]), 6)]
                item["limits"] = held.get("limits")
            items.append((len(frames), item))
            counts[subject] = len(frames)
        # Count stays on the finding, where it ranks one against another, and
        # not on each subject, where the spans beside it already say it.
        items.sort(key=lambda pair: (-pair[0], pair[1]["subject"]))
        listed = [item for _, item in items]
        findings.append({"reason": reason, "subjects": listed,
                         "frame_count": sum(n for n, _ in items),
                         "repair": joint_repair(reason, listed)})
    findings.sort(key=lambda f: (-f["frame_count"], f["reason"]))
    return findings


#: The repair for each `JointLimits` finding, keyed by its reason.
#:
#: Five unlike faults share this verifier, and each needs its own repair; the
#: reason string alone does not say what to change.
_JOINT_REPAIRS = {
    "original mechanism frame changed":
        "Authored motion moved a `link/*` object the mechanism owns (a constraint "
        "such as CopyTransforms on an IR link counts). Leave link transforms at "
        "their bind pose: drive the joints through their own channel (`key_joint`), "
        "or deform the meshes with an armature of your own and bind to that instead.",
    "mechanism frame missing or singular":
        "A `link/*` object the binding names was deleted, renamed, hidden from "
        "evaluation, or given a non-invertible matrix. Restore it; do not "
        "re-parent scene geometry onto it.",
    "articulated visual moved outside its mechanism":
        "A part of an articulated object moved relative to its own link. Deform "
        "the mesh vertices (armature, shape key, lattice) instead of moving the "
        "part's object transform.",
}

#: For a reason not in the table: the `str(error)` of a joint whose value
#: could not be recovered (`joint_position`), or a transform that was missing.
_UNREADABLE_JOINT_REPAIR = (
    "The joint's value could not be read from the evaluated transforms. Keep its "
    "origin and object present on every frame, and move it only through its own "
    "channel (`key_joint`: rotation_euler[2] for a revolute joint, location[2] for "
    "a prismatic one); any other rotation or location component moves it off its "
    "axis.")


def joint_repair(reason, subjects):
    """One sentence per `JointLimits` finding; the limit case quotes its numbers.

    The limit violation is the only finding with a measurement in it, and a
    repair that names the number is worth more than one that names the rule,
    so it quotes the worst subject's reach against its limits.
    """
    if reason == "joint limit or motion axis changed":
        worst = next((s for s in subjects if "q_range" in s), None)
        where = ("" if worst is None else
                 f" `{worst['subject']}` reached q in {worst['q_range']} against limits "
                 f"{worst['limits']}" + (f" (and {len(subjects) - 1} more joint(s))"
                                         if len(subjects) > 1 else "") + ".")
        return ("A joint was driven outside its declared range." + where +
                " Clamp the drive, or change the joint's limits in the object "
                "source; do not edit the evaluated pose.")
    return _JOINT_REPAIRS.get(reason, _UNREADABLE_JOINT_REPAIR)


def unresolved_evidence(rows):
    """Top-level evidence for events no frame could evaluate.

    The verdict stays `fail`: an event naming an operand that does not exist
    is an authoring error, and letting it read as `uncertain` would drop a
    typo'd event name out of the required set entirely. What changes is that
    the reason is legible from the summary line instead of inferable from a
    missing field.
    """
    if not rows:
        return {}
    refs = sorted({ref for row in rows for group in row.get("missing_refs", ())
                   for ref in group})
    return {"unresolved_count": len(rows), "unresolved_refs": refs}


def unresolved_repair(rows):
    """The repair an unresolved event needs, which is not a geometry repair."""
    if not rows:
        return []
    refs = sorted({ref for row in rows for group in row.get("missing_refs", ())
                   for ref in group})
    return [f"{len(rows)} event(s) never resolved: no checked frame could evaluate "
            f"{', '.join(refs) if refs else 'the declared operands'}. An operand is an "
            "authored part identity ('object/part'), a joint name, or a room surface "
            "such as 'room.floor'. Fix the declaration rather than the geometry."]

#: What `touch` means by default: within a millimeter, i.e. near contact. It
#: was 5 mm, against real contacts measured at 0.001-0.016 mm, so a declared
#: touch passed for parts that never met: 11 of the spring cart's 23
#: interfaces, a clock's wrong gears 2.2-3.2 mm apart, a piano's feet 5-8 mm
#: up on 1,084 of 1,085 frames. An author who means a looser contact
#: sets `tolerance`.
TOUCH_TOLERANCE_M = .001

#: Predicates whose `b` defaults to the room floor when the author omits it,
#: because the surface a support stands on is almost always that.
_DEFAULT_GROUND = "room.floor"


def _mechanism_held(lefts, rights, targets, *, singular=True):
    """`np.allclose(np.linalg.solve(left, right), target, atol=1e-5)` for each triple, batched.

    The same answers as one at a time: a stack is solved matrix by matrix by
    the same LAPACK routine, and the comparison is `isclose`'s own expression
    for finite values. Anything non-finite is left to `allclose`, which treats
    infinities apart. A singular matrix sends the batch back to one at a time,
    where it is `None` with `singular`, and raises without. A frame's mechanism
    checks were one `solve` and one `allclose` each: 250 a frame on the G1
    panel film, which made `allclose`'s Python overhead most of them.
    """
    try:
        relative = np.linalg.solve(np.asarray(lefts, dtype=np.float64),
                                   np.asarray(rights, dtype=np.float64))
    except np.linalg.LinAlgError:
        held = []
        for left, right, target in zip(lefts, rights, targets):
            try:
                actual = np.linalg.solve(np.asarray(left), np.asarray(right))
            except np.linalg.LinAlgError:
                if not singular:
                    raise
                held.append(None)
                continue
            held.append(bool(np.allclose(actual, target, atol=1e-5)))
        return held
    targets = np.asarray(targets, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        held = (np.abs(relative - targets) <= 1e-5 + 1e-5 * np.abs(targets)).all(axis=(1, 2)).tolist()
    finite = np.isfinite(relative).all(axis=(1, 2)) & np.isfinite(targets).all(axis=(1, 2))
    for k in np.flatnonzero(~finite):
        held[k] = bool(np.allclose(relative[k], targets[k], atol=1e-5))
    return held


#: What `measure` writes and `aggregate` reads: every verifier's per-frame rows
#: for the frames one process owns, and the declared context they are judged in.
ROWS_FORMAT = "dycraft.verification_rows.v1"


def verify(directory, baseline, scene_ir, catalog, *, start=None, end=None,
           seconds=None, frames=None, cancel=None, defer_partial_events=True,
           owned=None, info=None, records=None, progress=None, pair_store=None):
    """Check evaluated frames against the design, the plan, and the declarations.

    Two stages, and every caller gets the same answer from them. `measure`
    walks the frames and records what each frame showed; `aggregate` turns
    those rows into reports. One process aggregates its own rows. A parallel
    check aggregates the concatenated rows of its chunks
    (`chunking.merge_rows`), so its reports equal one process's by
    construction: a per-verifier merge could mislabel the advisory checks, keep
    one chunk's evidence, or judge an `any` or `planted` event differently from
    a single pass.

    `owned=(first, last)` narrows what is *measured* without narrowing what is
    *read*: frames outside it are still evaluated, so a step, a spike's
    neighborhood and an overlap that began earlier are judged with the
    evidence they need, but nothing outside the range becomes a row.
    """
    # Offline callers run the same native code in the configured bpy interpreter.
    try:
        import bpy  # noqa: F401
    except ImportError:
        if records is not None:
            raise RuntimeError('streamed verification requires the Blender interpreter')
        return _offline_verify(directory, baseline, scene_ir, catalog, start, end,
                               seconds, frames, cancel)
    rows = measure(directory, baseline, scene_ir, catalog, start=start, end=end,
                   seconds=seconds, frames=frames, cancel=cancel, owned=owned, info=info,
                   records=records, progress=progress, pair_store=pair_store)
    return aggregate(rows, defer_partial_events=defer_partial_events)


def measure(directory, baseline, scene_ir, catalog, *, start=None, end=None,
            seconds=None, frames=None, cancel=None, owned=None, info=None, records=None,
            progress=None, pair_store=None, owned_frames=None,
            advisories_for=None, room_frame=None):
    """`verify`'s first stage: each verifier's per-frame rows, for the owned frames.

    A row is what one frame showed -- a pair that overlapped and its box bound,
    whether an event's predicate held, where a bearing part's patch was, a
    body's support margin. Anything that depends on more than one frame
    (anchors, bearing intervals, phases, spans, verdicts) is `aggregate`'s.

    The exception is the overlap diagnostics, which need the meshes. This
    process measures the shift at each span's representative frame and the
    depth at its `DEPTH_FRAMES` highest-bound frames, over the spans it sees.
    A span split between processes has its whole-span representative and
    depth frames among its pieces' (see `_overlap_spans`), so `aggregate`
    finds every value it needs whichever process measured it.

    `owned_frames` is the scattered form of `owned`: a re-check owns the
    frames a repair could have changed (`chunking.reverify_frames`) and reads
    the context around them.

    Returns a JSON document (`ROWS_FORMAT`), already round-tripped, so the
    rows one process aggregates are exactly the rows a merge would read back.
    """
    try:
        from .collision import (BearingResolver, GeometryCache, collision, gap_predicate,
                                TOLERANCE, directional_separation, pair_keys,
                                separation_bound, bounded_box, boxes_overlap_many,
                                broad_overlap, coal_queries, backend_stats)
    except ImportError:
        from collision import (BearingResolver, GeometryCache, collision, gap_predicate,
                               TOLERANCE, directional_separation, pair_keys,
                               separation_bound, bounded_box, boxes_overlap_many,
                               broad_overlap, coal_queries, backend_stats)
    directory = Path(directory)
    info = info or json.loads((directory / "index.json").read_text())
    try:
        from .evaluation import READABLE_FORMATS
    except ImportError:
        from evaluation import READABLE_FORMATS
    if info.get('format') not in READABLE_FORMATS:
        raise ValueError('unsupported evaluation format')
    if records is None and info.get('format') and not info.get('complete'):
        raise ValueError('incomplete evaluated geometry cache')
    # An explicit set, for callers checking scattered moments rather than a
    # span — storyboard states are the case. Taken as given rather than
    # re-derived, so what is verified is exactly what was evaluated.
    if frames is None:
        frames = frame_indices(info, start, end, seconds=seconds)
    else:
        frames = [int(i) for i in frames]
        if (not frames or any(i < 0 or i >= info["n_frames"] for i in frames)
                or any(b <= a for a, b in zip(frames, frames[1:]))):
            raise ValueError("explicit frames must be increasing zero-based output indices "
                             "inside the animation")
    if owned_frames is not None:
        owned_set = set(owned_frames)
        owns = owned_set.__contains__
    else:
        owns = ((lambda frame: owned[0] <= frame <= owned[1]) if owned else (lambda frame: True))
    if records is None and 'frames' in info and not set(frames).issubset(info['frames']):
        raise ValueError('evaluation does not contain every requested frame')
    declarations = info.get("declarations", {})
    failures, mechanisms = [], []
    #: Frames where a joint left the narrower range a preloaded asset declares.
    #: Separate from `mechanisms` on purpose: a `JointLimits` failure means the
    #: mechanism was driven somewhere it cannot go, which is a broken binding;
    #: this means it was driven somewhere it *can* go but should not be seen
    #: going, which is an authoring choice. Reporting them as one verdict would
    #: make a pose note indistinguishable from a rig fault.
    anatomical = []
    #: `NoPenetration`'s rows: a pair that overlapped at a frame, an object
    #: outside the room at a frame, and the two diagnostics measured per span.
    hits, escapes, shifts, depths = [], [], [], []
    allowed_rows = []   # depth of every allowed overlap, measured though excused
    event_rows, slip_rows, support_rows, moved_rows = [], [], [], []
    expected = baseline["identities"]
    inherited = InheritedAllowances(scene_ir, catalog)
    masses = part_mass_table(scene_ir, catalog)
    #: The instances whose own balance and footing are a question: the
    #: articulated ones. Every authored CAD part has a `shape.volume`, so a
    #: mass table over a whole scene weighs the house, the stairs, and the yard
    #: as readily as the movers, and asking whether a house is balanced on its
    #: support is not a question. Joints are the cheap scene-agnostic signal
    #: for a body with its own behavior; `planted` stays declarable for any
    #: rigid prop this does not reach.
    articulated = {instance["name"] + "/"
                   for instance in (scene_ir or {}).get("instances", ())
                   if ((catalog or {}).get(instance.get("object_ref")) or {}).get("joints")}
    weighed_prefixes = {prefix for prefix in
                        {key.split("/")[0] + "/" for key in masses}
                        if prefix in articulated}
    #: Each weighed part's bounds on the previous evaluated frame. Footing is
    #: only a question for a body that moves -- a desk cannot develop a slide
    #: -- so `aggregate` reports the advisory checks for the instances seen to
    #: move anywhere in the check. Which ones moved is a per-frame fact (did a
    #: part's bounds change from the frame before), so a chunk can record it
    #: for the frames it owns; "moved since the first frame" could not be.
    previous_boxes = {}
    # A snapshot, so "the verifier does not write to its input" is a checked
    # invariant rather than a convention; compared on the way out.
    declared_before = json.dumps(declarations, sort_keys=True, default=repr)
    # Resolved once, here: the per-frame loop asks about an allowance window
    # for every candidate pair on every checked frame.
    allowance_index, resolved_events, declaration_errors = resolve_declarations(
        declarations, baseline, info)
    liquids = resolve_liquids(declarations, baseline, info)[0]
    liquid_body = {}

    def is_liquid(name):
        held = liquid_body.get(name)
        if held is None:
            held = liquid_body[name] = name in liquids or name.split("/")[0] in liquids
        return held
    # `window_indices` returns a contiguous run, so its two ends say it all.
    windows = [(indices[0], indices[-1]) for _, indices in resolved_events]
    continuity = MotionContinuity(info["fps"])
    #: Imported characters' joint limits, keyed by their armature object. Their
    #: rows join `anatomical`, one per frame; see `bone_anatomy`.
    rig_anatomy = current_anatomy(rig_anatomy_table(scene_ir, catalog), info.get("rigs"))
    try:
        from .collision_cache import PairCollisionCache
        from .collision import VERSION as COLLISION_METHOD
    except ImportError:
        from collision_cache import PairCollisionCache
        from collision import VERSION as COLLISION_METHOD
    # One cache for both lifetimes. A pair verdict keyed on the two local meshes
    # and their relative pose is reusable *within* a frame, *across* frames, and
    # *across* jobs, so it replaces the frame loop's own per-call OrderedDict as
    # well as the previous whole-scene frame cache. `pair_store` is the
    # project's on-disk store; without one the cache lives for this pass. See
    # collision_cache.py for why a whole-scene key was the wrong granularity.
    reusable = PairCollisionCache(None, method=COLLISION_METHOD, store=pair_store)
    geometry_cache = GeometryCache()
    # Per-body results keyed on the body's pose, the same argument the pair
    # cache makes for pair verdicts: a world AABB, a joint coordinate, and a
    # mechanism-frame check are pure functions of the matrices they read, and
    # most bodies hold still. Call-local, because the mapping from a joint or
    # edge name to its specification comes from *this* call's baseline.
    aabb_cache, joint_cache, edge_cache = OrderedDict(), {}, {}
    #: Last frame's candidate pairs: `(names) -> (state a, state b, verdict)`.
    #: A body's state is its mesh and its exact matrix, and a pair of bodies in
    #: the states they held on the frame before has that frame's verdict: its
    #: key, its lookup, and a recorded hit's measured bound are functions of the
    #: four. Most body-frames of a typical film are such holds.
    pair_held = {}
    aabb_limit = 200000

    def pose_key(matrix):
        return np.asarray(matrix, dtype=np.float64).tobytes()

    jointed = {name.split("__", 1)[0] for name in baseline["joints"]}
    replaced = {tag: instance for tag, instance in
                (baseline.get("surface_replacements") or {}).items()
                if tag in ROOM_SURFACE_TAGS}
    replacing = set(replaced.values())
    room_tags = containment = None
    if room_frame is not None:
        # A fill measures other frames than the check it completes; the room
        # is the one that check read, on its first frame.
        room_tags, containment = room_shell(*load_frame(directory, room_frame), baseline)
    #: `advisories_for`: only those instances' advisory rows, for `fill_advisories`.
    adv_only = advisories_for is not None
    resolver = BearingResolver()
    #: Each measured instance's last support inputs and result (`_supported_margin_held`).
    support_held = {}
    #: Weighed instances seen to move so far in this pass. The advisory rows
    #: are measured for them only -- `aggregate` reports nothing else -- from
    #: the frame each first moves; `fill_advisories` measures their earlier
    #: frames, here and after a merge. Measuring every weighed instance was
    #: most of their cost: all ten of the dog film's, of which three move.
    active = set()
    for position, frame in enumerate(frames):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("job cancelled")
        geometry_cache.next_frame()
        try:
            # A fill places bodies only: its record leaves out the transforms.
            record, geometry = (next(records) if records is not None
                                else load_frame(directory, frame, transforms=not adv_only))
        except StopIteration:
            raise ValueError('incomplete frame stream') from None
        if record['frame'] != frame:
            raise ValueError('out-of-order or incomplete frame stream')
        if room_tags is None:
            room_tags, containment = room_shell(record, geometry, baseline)
        transforms = None if adv_only else record["transforms"]
        # Each fixed edge's verdict: held, changed, or `None` for a missing or
        # singular frame. The ones not cached are solved in one batch, and
        # the rows are written in edge order, as one edge at a time did.
        frame_keys = {}

        def key_of(name):
            if name not in frame_keys:
                frame_keys[name] = pose_key(transforms[name])
            return frame_keys[name]
        verdicts, pending = [], []
        for index, edge in enumerate(() if adv_only else baseline.get("fixed_edges", [])):
            try:
                edge_key = (index, key_of(edge["parent"]), key_of(edge["child"]))
            except KeyError:
                verdicts.append((edge, None))
                continue
            held = edge_cache.get(edge_key)
            if held is None:
                pending.append((len(verdicts), edge_key))
            verdicts.append((edge, held))
        if pending:
            answers = _mechanism_held(
                [transforms[verdicts[k][0]["parent"]] for k, _ in pending],
                [transforms[verdicts[k][0]["child"]] for k, _ in pending],
                [verdicts[k][0]["matrix"] for k, _ in pending])
            for (k, edge_key), held in zip(pending, answers):
                if held is not None:
                    edge_cache[edge_key] = held
                verdicts[k] = (verdicts[k][0], held)
        for edge, held in verdicts:
            if not held and owns(frame):
                mechanisms.append({"frame": frame, "object": edge["child"],
                                   "reason": "original mechanism frame changed" if held is False
                                   else "mechanism frame missing or singular"})
        by_identity = {}
        for obj in record["objects"]:
            if obj.get("identity"):
                by_identity.setdefault(obj["identity"], []).append(obj)
        placed_in = []
        for identity, spec in ({} if adv_only else expected).items():
            objects = by_identity.get(identity, [])
            if (objects and spec["link"] in transforms
                    and identity.split("/", 1)[0] in jointed):
                placed_in.append((identity, transforms[spec["link"]], objects[0]["matrix"],
                                  spec["local_matrix"]))
        if placed_in:
            # A singular link raises here, as it did one identity at a time.
            answers = _mechanism_held([c[1] for c in placed_in], [c[2] for c in placed_in],
                                      [c[3] for c in placed_in], singular=False)
            for (identity, *_), held in zip(placed_in, answers):
                if not held and owns(frame):
                    mechanisms.append({"frame": frame, "identity": identity,
                                       "reason": "articulated visual moved outside its mechanism"})
        poses = {}
        for name, joint in ({} if adv_only else baseline["joints"]).items():
            try:
                joint_key = (name, pose_key(record["transforms"][joint["origin"]]),
                             pose_key(record["transforms"][joint["object"]]),
                             record.get("joint_positions", {}).get(joint["object"]))
                q = joint_cache.get(joint_key)
                if q is None:
                    q = joint_position(record, joint)
                    joint_cache[joint_key] = q
                lo, hi = joint["limits"]
                if not lo-1e-5 <= q <= hi+1e-5 and owns(frame):
                    mechanisms.append({"frame": frame, "joint": name, "q": q,
                                       "limits": [lo, hi], "reason": "joint limit or motion axis changed"})
                narrowed = joint.get("anatomical_limits")
                if narrowed and owns(frame):
                    a_lo, a_hi = narrowed
                    if not a_lo-1e-5 <= q <= a_hi+1e-5:
                        anatomical.append({
                            "frame": frame, "joint": name, "q": q,
                            "anatomical_limits": [a_lo, a_hi],
                            "hardware_limits": [lo, hi],
                            "reason": "pose the hardware allows but the asset's "
                                      "anatomical overlay does not"})
                poses[name.replace("__", ".", 1)] = q
            except (KeyError, ValueError, np.linalg.LinAlgError) as error:
                if owns(frame):
                    mechanisms.append({"frame": frame, "joint": name, "reason": str(error)})
        bodies = []
        body_objects = []
        #: Per body, `(mesh, exact matrix)`, or None where the frame has no
        #: local arrays: the state `pair_held` compares.
        states = []
        #: Indices into `bodies` that are room shell, not scene geometry. They
        #: are carried so an event can name `room.floor` and ask whether a thing
        #: rests on it, and skipped in the pair loop below so that resting on it
        #: is never reported as a penetration. Support and interpenetration are
        #: different questions and the floor is the surface that separates them.
        #:
        #: The prompts document `"room.floor"` as a legal event operand, so room
        #: surfaces must be bodies here, or an event naming one would be missing
        #: on every frame and fail `EventsSatisfied` though it held throughout.
        room_indices = set()
        local = getattr(geometry, 'local', None)
        world_bounds = {}
        room_objects = set(baseline.get("room_objects", []))
        seen_surfaces = {}
        for obj in record["objects"]:
            if obj.get("hidden"):
                continue
            is_room = bool(obj.get("room")) or obj["name"] in room_objects
            # The world bounds are a function of the local mesh and the
            # placement, and nothing downstream reads the world *vertices* when
            # the local arrays are present -- `body()` builds its BVH from those
            # -- so a hit here skips a full float32->float64 transform of every
            # vertex, which more than halves the time per frame on a scene of a few
            # hundred bodies.
            bounds_key = (obj.get("mesh"), pose_key(obj["matrix"])) if (
                local is not None and obj.get("mesh")) else None
            bounds = aabb_cache.get(bounds_key) if bounds_key else None
            if bounds is not None:
                aabb_cache.move_to_end(bounds_key)
                verts, tris = local[obj["name"]]
                lo, hi = bounds
            else:
                verts, tris = geometry[obj["name"]]
                if not len(verts) or not len(tris):
                    if not is_room and owns(frame):
                        failures.append({"frame": frame, "object": obj["name"], "reason": "empty or nonfinite evaluated mesh"})
                    continue
                lo, hi = verts.min(axis=0), verts.max(axis=0)
                if bounds_key is not None and np.isfinite(lo).all() and np.isfinite(hi).all():
                    while len(aabb_cache) >= aabb_limit:
                        aabb_cache.popitem(last=False)
                    aabb_cache[bounds_key] = (lo, hi)
            # NaN and +-Inf both propagate through min/max, so finite bounds are
            # exactly `np.isfinite(verts).all()` — without a second full pass
            # over the vertex array for every body on every frame.
            if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                if not is_room and owns(frame):
                    failures.append({"frame": frame, "object": obj["name"], "reason": "empty or nonfinite evaluated mesh"})
                continue
            if is_room:
                # Tagged once per call by `room_shell`, from whatever surfaces
                # exist -- not per frame against a room box that an uncapped
                # room does not have.
                tag = room_tags.get(obj["name"])
                if tag is None or tag in seen_surfaces:
                    continue
                seen_surfaces[tag] = obj["name"]
                room_indices.add(len(bodies))
                body_objects.append(obj)
                bodies.append(("room." + tag, verts, tris, lo, hi))
                states.append(bounds_key)
                continue
            identity = obj.get("identity") or obj["name"]
            # A replacement surface is *supposed* to occupy the plane the shell
            # one did, and with any thickness it straddles it; the scene
            # verifier exempts it the same way (`verify/scene_verify.py`), as
            # `replace_room_surface` documents.
            if (escaped_faces(lo, hi, containment) and not is_liquid(identity)
                    and identity.split("/")[0] not in replacing):
                # Two unlike findings share this verifier's failure count: a
                # pair that interpenetrates, and a single object that left the
                # room. Only the first becomes a span, so room escapes are kept
                # in their own rows and reported in their own shape.
                if owns(frame):
                    escapes.append({"frame": frame, "object": obj["name"],
                                    "aabb": [lo.tolist(), hi.tolist()]})
            body_objects.append(obj)
            world_bounds[obj["name"]] = (lo, hi)
            # `verts` is local on an AABB-cache hit and world on a miss, and the
            # slot cannot say which. Carry the matrix and a flag so a consumer
            # that needs world coordinates can get them: reading the slot as
            # world space compares a cached local array against a world anchor
            # and reports the body's own offset -- meters -- as motion.
            bodies.append((obj.get("identity") or obj["name"], verts, tris, lo, hi,
                           obj["matrix"], bounds is not None))
            states.append(bounds_key)
        # Reuse the bounds just reduced, and the local arrays, so continuity
        # does not make two more full passes over every body's vertices.
        if not adv_only:
            continuity.add(frame, record, geometry, local=local, bounds=world_bounds)
            continuity.add_bones(frame, record, info.get("rigs"))
            if rig_anatomy and owns(frame):
                anatomical.extend(bone_anatomy(frame, record, info.get("rigs"), rig_anatomy))
        bvh = {}
        def body(index):
            if index not in bvh:
                obj = body_objects[index]
                local = getattr(geometry, 'local', None)
                if local is not None:
                    bvh[index] = geometry_cache.placed(*local[obj['name']], obj['matrix'],
                                                      key=obj.get('mesh'))
                else:
                    bvh[index] = geometry_cache.get(bodies[index][1], bodies[index][2])
            return bvh[index]
        # No frame-level short-circuit: reuse is decided per pair, below. The
        # broad phase is cheap (an N x N AABB test plus one O(1) box per body)
        # next to the narrow-phase queries it filters, and running it always is
        # what lets a scene where one part moved still reuse every static pair.
        def local_box(obj):
            # `get` is memoized on the digest, so this reduces each unique
            # local mesh once for the whole call rather than once per frame.
            body = geometry_cache.get(*local[obj['name']], key=obj.get('mesh'))
            return bounded_box(body.lo, body.hi, obj['matrix'])
        boxes = [local_box(o) for o in body_objects] if local is not None and not adv_only else None
        frame_hits = []
        lows = np.array([b[3] for b in bodies])
        highs = np.array([b[4] for b in bodies])
        if bodies and not adv_only:
            overlap = np.minimum(highs[:, None], highs[None, :])-np.maximum(lows[:, None], lows[None, :])
            thin = np.minimum((highs-lows)[:, None], (highs-lows)[None, :]) <= 2*TOLERANCE
            broad = np.all(np.where(thin, overlap >= -TOLERANCE, overlap > TOLERANCE), axis=2)
        else:
            broad = np.zeros((0, 0), dtype=bool)
        candidates = []
        allowed_now = []
        # Per body once, rather than per pair: whether it is never a candidate,
        # and its identity split into instance and part.
        skipped = [k in room_indices or is_liquid(entry[0]) for k, entry in enumerate(bodies)]
        split = [entry[0].partition("/") for entry in bodies]
        for i, a in enumerate(() if adv_only else bodies):
            if skipped[i]:
                continue
            ai, _, ap = split[i]
            for j in np.flatnonzero(broad[i, :i]).tolist():
                if skipped[j]:
                    continue
                b = bodies[j]
                if allowed(allowance_index, a[0], b[0], frame):
                    # Excused from failing, not from being measured: an allowance's
                    # reason can say "a few mm" over an overlap of centimeters.
                    if owns(frame):
                        allowed_now.append((i, j, allowance_limit(allowance_index, a[0], b[0], frame)))
                    continue
                bi, _, bp = split[j]
                if ai == bi and inherited.allows(ai, ap, bp, poses):
                    continue
                candidates.append((i, j))
        if boxes is not None and candidates:
            # Oriented boxes, for the whole frame at once (`boxes_overlap_many`).
            keep = boxes_overlap_many([boxes[i] for i, _ in candidates],
                                      [boxes[j] for _, j in candidates])
            candidates = [pair for pair, kept in zip(candidates, keep) if kept]
        if allowed_now:
            if boxes is not None:
                keep = boxes_overlap_many([boxes[i] for i, _, _ in allowed_now],
                                          [boxes[j] for _, j, _ in allowed_now])
                allowed_now = [entry for entry, kept in zip(allowed_now, keep) if kept]
            for i, j, limit in allowed_now:
                depth = deepest_vertex_m(body(i), body(j))
                allowed_rows.append({"frame": frame, "a": bodies[i][0], "b": bodies[j][0],
                                     "depth": None if depth is None else float(depth),
                                     "limit": limit})
        names = [obj["name"] for obj in body_objects]
        held_now, reused, fresh = {}, {}, []
        for n, (i, j) in enumerate(candidates):
            held = pair_held.get((names[i], names[j]))
            if (held is not None and states[i] is not None and states[j] is not None
                    and held[0] == states[i] and held[1] == states[j]):
                reused[n] = held[2]
            else:
                fresh.append(n)
        keys = dict(zip(fresh, pair_keys([body(candidates[n][0]) for n in fresh],
                                         [body(candidates[n][1]) for n in fresh])))
        looked = {n: reusable.get(keys[n]) for n in fresh}
        # Every uncached pair's first Coal query, in one round trip (fix 6).
        # Most end there; the ones that overlap go on to their erosion and
        # containment tests one at a time, as before.
        asked = [n for n in fresh if not looked[n][1]
                 and broad_overlap(body(candidates[n][0]).lo, body(candidates[n][0]).hi,
                                   body(candidates[n][1]).lo, body(candidates[n][1]).hi)]
        raws = dict(zip(asked, coal_queries([('raw_overlap', body(candidates[n][0]),
                                               body(candidates[n][1])) for n in asked])))
        for n, (i, j) in enumerate(candidates):
            a, b = bodies[i], bodies[j]
            if n in reused:
                hit = reused[n]
                held_now[(names[i], names[j])] = (states[i], states[j], hit)
                if hit is not None:
                    frame_hits.append({"a": a[0], "b": b[0], **hit})
                continue
            key = keys[n]
            hit, cached = looked[n]
            if cached:
                if hit is not None:
                    # The pose bin is 1e-6 wide, so a recorded bound is
                    # only approximate for this exact pose. Measure it.
                    bound, direction = separation_bound(body(i), body(j))
                    hit = {**hit, 'separation_upper_bound_m': bound,
                           'separation_direction': direction.tolist()}
            else:
                try:
                    # Not asked means the boxes do not overlap, which
                    # `collision` decides before reading `raw`.
                    hit = collision(body(i), body(j), raw=raws.get(n, False))
                except ValueError as error:
                    raise ValueError(f'collision query failed at frame {frame}, {a[0]} / {b[0]}: {error}') from error
                reusable.put(key, hit)
            if states[i] is not None and states[j] is not None:
                held_now[(names[i], names[j])] = (states[i], states[j], hit)
            if hit is not None:
                frame_hits.append({"a": a[0], "b": b[0], **hit})
        pair_held = held_now
        for hit in (frame_hits if owns(frame) else ()):
            hits.append({"frame": frame, "a": hit["a"], "b": hit["b"], "kind": hit["kind"],
                         "bound": float(hit["separation_upper_bound_m"]),
                         "direction": [float(v) for v in hit["separation_direction"]]})
        lookup = {b[0]: i for i, b in enumerate(bodies)}
        lookup.update(replaced_surfaces(bodies, replaced, lookup))
        boxes_now, moved = {}, set()
        for entry in bodies:
            prefix = entry[0].split("/")[0] + "/"
            if prefix not in weighed_prefixes:
                continue
            box = (tuple(np.round(entry[3], 6)), tuple(np.round(entry[4], 6)))
            boxes_now[entry[0]] = box
            if previous_boxes.get(entry[0], box) != box:
                moved.add(prefix)
        previous_boxes = boxes_now
        if moved and owns(frame):
            moved_rows.append({"frame": frame, "moved": sorted(moved)})
        active |= moved
        measuring = (set(advisories_for) if adv_only else weighed_prefixes & active)
        if measuring and owns(frame):
            # The bodies that bear their own weight, which is what creep and
            # balance are about. A prop that slides while resting is worth
            # knowing too, but a scene has hundreds of props and two feet;
            # `planted` is declarable for any body this does not reach.
            gindex = ground_index(bodies)
            parts = [i for i, entry in enumerate(bodies) if entry[0].split("/")[0] + "/" in measuring]
            bears = resolver.resolve(parts, ground_candidates_many(bodies, parts, .01, gindex),
                                     bodies, body, .01, frame)
            lifted = [index for index, borne_on in bears.items() if borne_on is None]
            under = dict(zip(lifted, ground_candidates_many(bodies, lifted, 10.0, gindex)))
            for index, borne_on in bears.items():
                entry = bodies[index]
                row = {"frame": frame, "part": entry[0], "on": borne_on}
                if borne_on is not None:
                    row["patch"] = [float(v) for v in _footprint(contact_patch_xy, entry,
                                                                  body_objects[index])]
                elif under[index]:
                    # Height above whatever is under it, or above nothing.
                    row["clearance"] = float(entry[3][2] - max(bodies[g][4][2] for g in under[index]))
                slip_rows.append(row)
            for instance in sorted(prefix[:-1] for prefix in measuring):
                measured, why = _supported_margin_held(
                    support_held, instance, masses, bodies, body_objects, body, .01,
                    gap_predicate, bears)
                support_rows.append(
                    {"frame": frame, "instance": instance, "margin": float(measured[0]),
                     "bearing": int(measured[1])} if measured is not None
                    else {"frame": frame, "instance": instance, "why": why})
        # The frame's `touch` and `separated` gaps, in one round trip.
        spaced = [k for k, (event, _) in enumerate(() if adv_only else resolved_events)
                  if event["predicate"] in ("touch", "separated") and owns(frame)
                  and windows[k][0] <= frame <= windows[k][1]
                  and event.get("a") in lookup and event.get("b") in lookup]
        gaps = dict(zip(spaced, coal_queries([
            ("distance", body(lookup[resolved_events[k][0]["a"]]),
             body(lookup[resolved_events[k][0]["b"]])) for k in spaced])))
        for k, (event, _) in enumerate(() if adv_only else resolved_events):
            if not windows[k][0] <= frame <= windows[k][1] or not owns(frame):
                continue
            predicate = event["predicate"]
            if predicate == "supported":
                # `a` is an instance; its parts are what carry the mass, so
                # there is no single identity to resolve. The ground is.
                refs = (event.get("b") or _DEFAULT_GROUND,)
            elif predicate == "planted":
                refs = (event.get("a"), event.get("b") or _DEFAULT_GROUND)
            elif predicate in ("touch", "separated", "above"):
                refs = (event.get("a"), event.get("b"))
            else:
                refs = ()
            row = {"event": k, "frame": frame}
            event_rows.append(row)
            missing = [ref for ref in refs if ref not in lookup]
            if missing or (predicate == "joint_at" and event["joint"] not in poses):
                row["missing"] = missing or [event["joint"]]
            elif predicate in ("touch", "separated"):
                # Measured, not only thresholded: the row reports the worst
                # gap over the window, and a pass at 0.9 mm against a
                # 1 mm tolerance and one at 0.001 mm are different facts.
                gap = float(gaps[k])
                row["gap"] = gap
                row["holds"] = (gap <= event.get('tolerance', TOUCH_TOLERANCE_M)
                                if predicate == 'touch' else gap >= event.get('distance', .005))
            elif predicate == "joint_at":
                row["holds"] = bool(abs(poses[event["joint"]] - event["value"])
                                    <= event.get("tolerance", .01))
            elif predicate == "above":
                a, b = bodies[lookup[event["a"]]], bodies[lookup[event["b"]]]
                row["holds"] = bool(a[3][2] >= b[4][2] - event.get("tolerance", .005))
            elif predicate == "planted":
                ai = lookup[event["a"]]
                gi = lookup[event.get("b") or _DEFAULT_GROUND]
                # Deliberately looser than `tolerance`: the two measure
                # different things and must not share a scale. This one decides
                # what counts as bearing, and therefore what counts as a step
                # -- a foot that lifts less than `contact` never releases, so
                # its whole stride would be reported as one planted span. That
                # case is real and is reported (one long interval, near-zero
                # clearance) rather than thresholded away.
                row["touching"] = bool(gap_predicate(body(ai), body(gi),
                                                     event.get("contact", .01)))
                if row["touching"]:
                    row["patch"] = [float(v) for v in contact_patch_xy(world_verts(bodies[ai]))]
                else:
                    row["lift"] = float(bodies[ai][3][2] - bodies[gi][4][2])
            elif predicate == "supported":
                gi = lookup[event.get("b") or _DEFAULT_GROUND]
                measured, _why = _supported_margin(
                    event["a"], masses, bodies, body_objects, body, gi,
                    event.get("contact", .01), gap_predicate)
                row["margin"] = None if measured is None else float(measured[0])
            else:
                raise ValueError(f"unsupported evaluated event predicate: {predicate}")
        if progress:
            progress({'phase': 'evaluate_verify', 'frame': frame, 'done': position + 1,
                      'total': len(frames),
                      'collision_count': len(hits) + len(escapes), **reusable.stats(),
                      'bvh_builds': backend_stats().get('builds', 0),
                      'bvh_cache_hits': backend_stats().get('cache_hits', 0)})
    if adv_only:
        reusable.flush()
        return json.loads(json.dumps({"format": ROWS_FORMAT, "frames": frames,
                                      "rows": {"SupportMargin": support_rows,
                                               "ContactSlip": slip_rows},
                                      "stats": _bearing_stats(resolver)}))
    _measure_diagnostics(hits, shifts, depths, directory, geometry_cache, collision_tools=(
        directional_separation,), progress=progress, cancel=cancel, store=reusable)
    reusable.flush()
    # The frames before each instance first moved here.
    own = {"frames": [frame for frame in frames if owns(frame)],
           "rows": {"moved": moved_rows, "SupportMargin": support_rows, "ContactSlip": slip_rows}}
    _fill_advisory_rows(own, directory, baseline, scene_ir, catalog,
                        info={**info, "complete": True}, room_frame=frames[0], cancel=cancel)
    support_rows, slip_rows = own["rows"]["SupportMargin"], own["rows"]["ContactSlip"]
    if json.dumps(declarations, sort_keys=True, default=repr) != declared_before:
        raise RuntimeError("verify() modified the declarations it was checking; a verifier "
                           "must not write to its input, because the worker persists it "
                           "and replay compares it")
    # The allowance tables are declared input, not findings, and echoing them on
    # every report would dominate what this verifier emits. Counts stay so a
    # surprising exemption is still visible; the tables themselves live in the
    # design sources and the persisted declarations.
    allowances = {"declared": len(declarations.get("collision_allowances") or ()),
                  "inherited": len(inherited.evidence)}
    if inherited.evidence:
        allowances.update(inherited_allowances(inherited.evidence, catalog))
    room = None
    if containment is not None:
        lo, hi, faces = containment
        room = {"lo": None if lo is None else [float(v) for v in lo],
                "hi": None if hi is None else [float(v) for v in hi],
                "faces": [[axis, sign, tag] for (axis, sign), tag in sorted(faces.items())]}
    bundle = {
        "format": ROWS_FORMAT,
        "frames": [frame for frame in frames if owns(frame)] or frames,
        # Identical in every chunk of one check: declared input and scene
        # facts. `chunking.merge_rows` refuses chunks that disagree.
        "context": {"fps": info["fps"],
                    "identity_changed": info.get("scene_ref") != scene_ir["id"],
                    "events": [{"event": event, "window_frames": list(window)}
                               for (event, _), window in zip(resolved_events, windows)],
                    "declaration_errors": declaration_errors,
                    "allowances": allowances,
                    "liquids": sorted(liquids),
                    "room": room,
                    # What exists to be checked, for `not_applicable`.
                    "joints": bool(baseline["joints"] or baseline.get("fixed_edges")),
                    "anatomical": bool(rig_anatomy) or any(
                        joint.get("anatomical_limits") for joint in baseline["joints"].values()),
                    "weighed": sorted(prefix[:-1] for prefix in weighed_prefixes)},
        # Per process, not per frame; a merge sums them. The collision-cache
        # test reads them to prove a reuse path was actually taken.
        "stats": {"bvh_builds": backend_stats().get('builds', 0),
                  "bvh_cache_hits": backend_stats().get('cache_hits', 0),
                  **reusable.stats(), **_bearing_stats(resolver)},
        "rows": {"AnimationWellFormed": failures,
                 "JointLimits": mechanisms,
                 "AnatomicalJointLimits": anatomical,
                 "NoPenetration": {"hits": hits, "escapes": escapes,
                                   "shifts": shifts, "depths": depths,
                                   "allowed": allowed_rows},
                 "EventsSatisfied": event_rows,
                 "MotionContinuity": continuity.rows(owns, clips=info.get("clips")),
                 "moved": moved_rows,
                 "SupportMargin": support_rows,
                 "ContactSlip": slip_rows},
    }
    return json.loads(json.dumps(bundle))


def _bearing_stats(resolver):
    """The bearing resolver's counters, flat, so a merge sums them."""
    return {"bearing_" + key: value for key, value in resolver.counts.items()}


def advisory_gaps(bundle):
    """`{frame: instance prefixes}`: the advisory rows a moving instance lacks.

    `aggregate` reports the instances seen to move anywhere in the check, and
    reads their rows on every checked frame. `measure` writes them from the
    frame an instance first moves in its own pass, so a merge of chunks, or a
    re-check, can lack an instance's rows where another pass saw it move.
    """
    rows = bundle.get("rows") or {}
    moving = sorted({prefix for row in rows.get("moved", ()) for prefix in row["moved"]})
    have = {(row["instance"] + "/", row["frame"]) for row in rows.get("SupportMargin", ())}
    missing = {}
    for prefix in moving:
        for frame in bundle["frames"]:
            if (prefix, frame) not in have:
                missing.setdefault(frame, set()).add(prefix)
    return missing


def fill_advisories(bundle, directory, baseline, scene_ir, catalog, *, info=None, cancel=None):
    """Measure the advisory rows `advisory_gaps` names, in place, from the evaluation on disk.

    Exact: the rows are what one pass would have written, because they are
    what `measure` writes for those frames, against the room the check read on
    its first frame. Returns the number of frames measured.
    """
    info = info or json.loads((Path(directory) / "index.json").read_text())
    return _fill_advisory_rows(bundle, directory, baseline, scene_ir, catalog,
                               info={**info, "complete": True}, room_frame=bundle["frames"][0],
                               cancel=cancel)


def _fill_advisory_rows(bundle, directory, baseline, scene_ir, catalog, *, info, room_frame,
                        cancel=None):
    missing = advisory_gaps(bundle)
    if not missing:
        return 0
    frames = sorted(missing)
    extra = measure(directory, baseline, scene_ir, catalog, frames=frames, owned_frames=frames,
                    info=info, advisories_for=set().union(*missing.values()),
                    room_frame=room_frame, cancel=cancel)
    rows = bundle["rows"]
    # Sorted by frame and stable, which is all `aggregate` reads the order
    # for: each part's and each instance's rows in time order.
    rows["SupportMargin"] = sorted(
        rows["SupportMargin"] + [row for row in extra["rows"]["SupportMargin"]
                                 if row["instance"] + "/" in missing[row["frame"]]],
        key=lambda row: row["frame"])
    rows["ContactSlip"] = sorted(
        rows["ContactSlip"] + [row for row in extra["rows"]["ContactSlip"]
                               if row["part"].split("/")[0] + "/" in missing[row["frame"]]],
        key=lambda row: row["frame"])
    return len(frames)


def aggregate(bundle, *, defer_partial_events=True):
    """`verify`'s second stage: the reports, from `measure`'s rows.

    Pure and Blender-free, so the parent of a parallel check runs it on the
    merged rows of its chunks. Everything that spans frames is decided here,
    once: overlap spans, event satisfaction, slip anchors, and bearing
    intervals, support phases, and which instances moved.
    """
    if bundle.get("format") != ROWS_FORMAT:
        raise ValueError("not a verification rows document")
    context, rows, frames = bundle["context"], bundle["rows"], bundle["frames"]
    gaps = advisory_gaps(bundle)
    if gaps:
        # Reporting from them would read the absent frames as gaps in each
        # part's contact. A merge has to complete them first.
        raise ValueError(f"advisory rows are missing on {len(gaps)} frame(s), first {min(gaps)}, "
                         "for instances that moved elsewhere in the check; "
                         "`fill_advisories` measures them")
    fps = context["fps"]
    errors = context["declaration_errors"]
    failures = ([{"reason": "frozen scene identity changed"}] if context["identity_changed"]
                else []) + rows["AnimationWellFormed"]
    mechanisms = rows["JointLimits"]
    findings = joint_findings(mechanisms) if mechanisms else []
    exceeded = allowance_overruns(rows["NoPenetration"].get("allowed", ()))
    collisions = rows["NoPenetration"]["hits"] + rows["NoPenetration"]["escapes"] + exceeded
    penetration = _penetration_evidence(context, rows["NoPenetration"], bundle["stats"])
    events_failed, event_rows, unresolved, placed = _event_evidence(
        context, rows["EventsSatisfied"], frames, defer_partial_events)
    moving = {prefix for row in rows["moved"] for prefix in row["moved"]}
    samples, skipped = support_samples(rows["SupportMargin"], moving)
    support_flagged, support_kwargs = support_evidence(
        samples, fps, considered=sorted(prefix[:-1] for prefix in moving), skipped=skipped)
    slip_flagged, slip_kwargs = slip_evidence(slip_tracks(rows["ContactSlip"], moving), fps)
    unweighed = (None if context["weighed"] else
                 "no articulated instance with known masses to bear its own weight")
    return [report("AnimationWellFormed", failures,
                   frames_checked={"start": frames[0], "end": frames[-1],
                                   "count": len(frames)}),
            report("JointLimits", mechanisms, list_failures=False,
                   repairs=[finding["repair"] for finding in findings],
                   not_applicable=None if context["joints"] else "no joints or mechanism frames",
                   **({"findings": findings} if findings else {})),
            # Advisory, by the operator's decision: this is diagnostic only
            # and must not block.
            #
            # A `JointLimits` failure means the mechanism was driven somewhere
            # it cannot go — a broken binding. This means it was driven
            # somewhere it *can* go but the G1's anatomical overlay says it
            # should not be seen going, which is a pose judgement. Nothing
            # gated on it either way, since `required_checks_pass` omits the
            # name; reporting `fail` only made a pose note read as a defect
            # everywhere a verdict is rendered.
            # IR joints' rows stay one per frame; an imported character's bone
            # rows are grouped per joint and measure (`bone_anatomy_rows`).
            report("AnatomicalJointLimits",
                   [row for row in rows["AnatomicalJointLimits"] if "rig" not in row]
                   + bone_anatomy_rows([row for row in rows["AnatomicalJointLimits"]
                                        if "rig" in row], fps),
                   advisory=True,
                   not_applicable=None if context["anatomical"] else "no joint carries anatomical limits"),
            report("NoPenetration", collisions, list_failures=False,
                   repairs=(declaration_repair(errors, "collision_allowance")
                            + declaration_repair(errors, "liquid")
                            + allowance_overrun_repair(exceeded)
                            + penetration_repair(penetration)),
                   **penetration),
            # `failures` here was the names of the events the rows below
            # already mark `ok: false`, so it said the same thing twice.
            report("EventsSatisfied", events_failed, list_failures=False,
                   repairs=(declaration_repair(errors, "event")
                            + unresolved_repair(unresolved)),
                   not_applicable=None if event_rows else "no events declared",
                   events=event_rows, **unresolved_evidence(unresolved), **placed),
            MotionContinuity.aggregate(rows["MotionContinuity"], fps,
                                       adjacent=any(b == a + 1 for a, b in zip(frames, frames[1:]))),
            report("SupportMargin", support_flagged, advisory=True, not_applicable=unweighed,
                   **support_kwargs),
            report("ContactSlip", slip_flagged, advisory=True, not_applicable=unweighed,
                   **slip_kwargs)]


def _event_evidence(context, rows, frames, defer_partial_events):
    """`EventsSatisfied`'s `(failed, rows, unresolved, extra evidence)`, over `frames`.

    The extra evidence holds `frames_failed` when frames failed, and
    `events_checked` and `events_deferred` when any event was deferred.
    """
    fps, checked = context["fps"], set(frames)
    measured_by_event = {}
    for row in rows:
        measured_by_event.setdefault(row["event"], []).append(row)
    events_failed, event_rows, failed_frames = [], [], []
    for k, entry in enumerate(context["events"]):
        event = entry["event"]
        lo, hi = entry["window_frames"]
        answered = [frame for frame in range(lo, hi + 1) if frame in checked]
        # Deferral is "were this event's frames actually checked", not "is its
        # window nested in the requested interval". The two differ whenever an
        # interval boundary falls between frames, and the frames question is
        # the one that matters. `defer_partial_events=False` reports what was
        # seen of an event and defers only one nothing was seen of; the
        # default is the answer a lone interval check must give: an event is
        # not "unsatisfied" because the frames that would satisfy it were
        # never looked at.
        deferred = (len(answered) < hi - lo + 1 if defer_partial_events else not answered)
        # Seconds rounded: `37.708333333333336` is frame 905 over 24 fps, and
        # every digit past the fourth is float noise. Both units, always: the
        # declared window is in whichever the project uses, the resolved
        # frames are what was actually checked.
        row = {"name": event["name"],
               "window": [round(float(bound), 4) for bound in event["window"]],
               "window_frames": [lo, hi]}
        if deferred:
            event_rows.append({**row, "deferred": True, "status": "deferred"})
            continue
        measured = measured_by_event.get(k, [])
        if [m["frame"] for m in measured] != answered:
            raise ValueError(f"event {event['name']!r} has rows for {len(measured)} of the "
                             f"{len(answered)} checked frames in its window")
        predicate = event["predicate"]
        values, missing = [], []
        anchor, intervals, clearance, slip = None, 0, 0.0, None
        worst = None    # (gap or margin, frame): the frame nearest to failing
        for m in measured:
            if "missing" in m:
                values.append(False)
                missing.append(m)
                continue
            if predicate == "planted":
                # Anchored, not per-frame. A per-frame slip threshold passes a
                # support that creeps a millimeter a frame for a second, which
                # is the failure this predicate exists to catch; measuring from
                # first contact catches both that and a single lurch.
                if not m["touching"]:
                    # Airborne is not a failure; the claim is about the span
                    # where it bears. Releasing drops the anchor, so the next
                    # touchdown is measured from where it actually landed.
                    if anchor is not None:
                        intervals += 1
                    anchor = None
                    clearance = max(clearance, m["lift"])
                    values.append(True)
                    continue
                patch = np.asarray(m["patch"], dtype=float)
                if anchor is None:
                    anchor = patch
                moved = float(np.linalg.norm(patch - anchor))
                values.append(moved <= event.get("tolerance", .005))
                # Record on the first bearing frame too, not only when a later
                # one beats it: a satisfied event that reports no number
                # cannot be told from one that measured nothing.
                if slip is None or moved > slip[0]:
                    slip = (moved, m["frame"])
            elif predicate == "supported":
                # No margin: nothing of this body is bearing on that surface,
                # or it cannot be weighed. Not a failure: airborne is a phase,
                # and an unweighable body is a gap in the asset.
                values.append(m["margin"] is None or m["margin"] <= event.get("margin", 0.0))
                if m["margin"] is not None and (worst is None or m["margin"] > worst[0]):
                    worst = (m["margin"], m["frame"])
            else:
                values.append(m["holds"])
                # The gap the verdict turns on: an `all` touch fails at its
                # widest frame, an `any` touch holds or fails at its closest
                # approach; `separated` the other way round.
                wider = (predicate == "touch") == (event.get("mode", "all") == "all")
                if "gap" in m and (worst is None or (m["gap"] > worst[0] if wider
                                                     else m["gap"] < worst[0])):
                    worst = (m["gap"], m["frame"])
        ok = bool(values) and (all(values) if event.get("mode", "all") == "all" else any(values))
        if not ok:
            events_failed.append({"event": event["name"]})
        unmet = [m["frame"] for m, holds in zip(measured, values) if not holds]
        held = next((m["frame"] for m, holds in zip(measured, values) if holds), None)
        row.update(ok=ok, checked=len(answered),
                   satisfied_at=(round(held / fps, 4) if ok and held is not None else None))
        if len(answered) < hi - lo + 1:
            row["partial"] = True
        if unmet:
            row.update(unsatisfied_count=len(unmet), unsatisfied=frame_runs(unmet))
        # The measurement beside the verdict: the gap it turns on for
        # `touch` and `separated`, the margin furthest outside the support for
        # `supported`.
        if worst is not None:
            key = "worst_margin" if predicate == "supported" else "worst_gap"
            row.update({f"{key}_m": round(worst[0], 6), f"{key}_frame": worst[1]})
        if not ok:
            # An `any` event that never held fails over its whole window; an
            # `all` event fails where it did not hold.
            failed_frames.extend(range(lo, hi + 1) if event.get("mode", "all") == "any" else unmet)
        # A `planted` row carries the measurement, not only the verdict: how
        # far the patch actually crept and where. A pass at 4 mm against a 5 mm
        # tolerance and a pass at 0.01 mm are different facts about the motion,
        # and the author cannot tell them apart from `ok` alone.
        if slip is not None:
            row.update(max_slip_m=round(slip[0], 6), max_slip_frame=slip[1])
            # Slip alone cannot tell a slide from a shallow step: both are a
            # patch that moved a long way while never far off the ground. The
            # pair can. One bearing interval spanning the window with a
            # near-zero peak clearance is a foot that never lifted; many
            # intervals with real clearance between them is a walk.
            row.update(bearing_intervals=intervals + (1 if anchor is not None else 0),
                       max_clearance_m=round(clearance, 6))
        # "The predicate resolved and was false" and "the predicate never
        # resolved" are different findings with different repairs -- move the
        # geometry, versus fix the declaration -- so they must not reach the
        # author as the same `ok: false`, or an author repairs geometry that
        # is already correct.
        #
        # `unresolved` is reserved for "no checked frame could evaluate this"
        # -- the unambiguous case. An operand that resolves on some frames and
        # not others is `unsatisfied` with `missing_on` naming where it went
        # missing, because there the geometry question was genuinely asked and
        # genuinely answered somewhere.
        missing_frames = [m["frame"] for m in missing]
        if missing:
            row["missing_refs"] = [list(refs) for refs in sorted(
                {tuple(m["missing"]) for m in missing})]
            row["missing_on"] = frame_runs(missing_frames)
        row["status"] = ("satisfied" if ok
                         else "unresolved" if len(missing_frames) == len(answered)
                         else "unsatisfied")
        event_rows.append(row)
    # An event that could not be read is an unmet obligation, not an absent
    # one: dropping it would pass `EventsSatisfied` on a film whose one
    # declared requirement was mistyped. It is placed at no frame.
    invalid = [error for error in context["declaration_errors"] if error["kind"] == "event"]
    for error in invalid:
        name = error.get("name") or f"events[{error['index']}]"
        events_failed.append({"event": name})
        event_rows.append({"name": name, "ok": False, "status": "invalid",
                           "index": error["index"], "reason": error["reason"]})
    # Unmet first, worst first; then the ones nothing could say anything
    # about; then the satisfied ones in film order. A real report had its one
    # `ok: false` row fifth of nine, which is the wrong place for the only
    # part of it that needed doing something about.
    event_rows.sort(key=lambda row: (
        0 if row.get("ok") is False else (1 if row.get("deferred") else 2),
        -row.get("unsatisfied_count", 0),
        row.get("window", [0])[0], row["name"]))
    unresolved = [row for row in event_rows if row.get("status") == "unresolved"]
    # How much of the declaration this check could judge. A storyboard or
    # interval check defers every event whose window it did not cover, and a
    # `pass` over the rest read as a pass over all of them; the counts say how
    # many were checked and how many were left for a check that covers them.
    deferred_count = sum(1 for row in event_rows if row.get("deferred"))
    coverage = ({"events_checked": len(event_rows) - deferred_count,
                 "events_deferred": deferred_count} if deferred_count else {})
    # A check that covered no event's window says so once. A storyboard shot
    # check evaluates a handful of state frames, so every multi-frame event
    # defers by construction, and naming each one would fill the reply with rows
    # that say only "deferred".
    if event_rows and all(row.get("deferred") for row in event_rows):
        event_rows = {"deferred": len(event_rows), "checked": 0}
    placed = ({"frames_failed": frame_runs(sorted(set(failed_frames)))}
              if failed_frames and not invalid else {})
    return events_failed, event_rows, unresolved, {**placed, **coverage}


def _offline_verify(directory, baseline, scene_ir, catalog, start, end,
                    seconds, frames, cancel):
    import os
    import tempfile
    import sys
    from ...render._blender_env import blender_python
    from .._process import run_cancellable
    interpreter = blender_python()
    if not interpreter:
        raise RuntimeError('Native Blender collision verification requires bpy')
    with tempfile.TemporaryDirectory(prefix='native_verify_') as scratch:
        request = Path(scratch) / 'request.json'
        result = Path(scratch) / 'result.json'
        request.write_text(json.dumps([str(Path(directory).resolve()), baseline, scene_ir, catalog, start, end, seconds, frames]))
        # The child imports `verify` as a **top-level module**, which is what
        # the try/except imports at the head of this file exist for and what
        # `runtime.py` does when it binds these modules into `/tools`, so the
        # checkout's directory name does not matter (importing
        # `dycraft.animation.blender.verify` through the repo's parent works
        # only when it is literally `dycraft`). Two directories, because the top-level fallbacks reach sideways into
        # `<repo>/verify` for `_inherited_allowances`.
        script = """import bpy, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
sys.path.insert(1, sys.argv[4])
from verify import verify
p, baseline, scene, catalog, start, end, seconds, frames = json.loads(Path(sys.argv[1]).read_text())
reports = verify(p, baseline, scene, catalog, start=start, end=end,
                 seconds=seconds, frames=frames)
Path(sys.argv[2]).write_text(json.dumps({'reports': reports}))
"""
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
                   DYCRAFT_COAL_PYTHON=os.environ.get('DYCRAFT_COAL_PYTHON') or sys.executable)
        _here = Path(__file__).resolve()
        run_cancellable([interpreter, '-c', script, str(request), str(result),
                         str(_here.parents[0]), str(_here.parents[2] / 'verify')],
                        check=True, cancel=cancel, env=env)
        return json.loads(result.read_text())['reports']
