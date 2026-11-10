"""Declarations: the timeline's units and windows, collision allowances, liquids, and events, resolved."""
from __future__ import annotations

import math

try:
    from .verify_room import ROOM_SURFACE_TAGS
except ImportError:  # Read-only modules staged into isolated Blender.
    from verify_room import ROOM_SURFACE_TAGS


def frame_indices(info, start=None, end=None, *, seconds=None):
    """Resolve a checked interval to inclusive zero-based output indices.

    `start`/`end` are inclusive zero-based OUTPUT FRAME INDICES. That is the
    surface's one unit: `snapshot`, `plan` states, `ik`, `camera_path`, and
    keyframe authoring all speak it. Supply both ends.

    `seconds=[start, end]` is the alternate spelling, inclusive seconds from
    the first output frame. Event and collision-allowance windows are
    declared in it because a window persisted in the blend asserts a physical
    duration rather than an index. Never supply both spellings; neither means
    the whole timeline.
    """
    n, fps = info["n_frames"], info["fps"]
    if type(n) is not int or n < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid animation timeline")
    if seconds is not None and (start is not None or end is not None):
        raise ValueError("supply output indices (start/end) or seconds=[a, b], not both")
    if seconds is not None:
        pair = list(seconds) if isinstance(seconds, (list, tuple)) else None
        if pair is None or len(pair) != 2:
            raise ValueError("seconds must be an inclusive [start, end] pair")
        lo, hi = pair
        if (isinstance(lo, bool) or isinstance(hi, bool)
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi))
                or lo < 0 or hi < lo or hi > (n - 1) / fps + 1e-9):
            raise ValueError("interval must be finite, ordered, and inside the animation")
        frames = [i for i in range(n) if lo - 1e-9 <= i / fps <= hi + 1e-9]
        if not frames:
            raise ValueError("interval contains no output frame")
        return frames
    if start is None and end is None:
        return list(range(n))
    # `type(...) is not int` rather than isinstance: bool is an int subclass,
    # and it also rejects the seconds a caller may still be passing here out
    # of habit — 2/24 is a float and fails loudly rather than being read as
    # frame 0.
    if (type(start) is not int or type(end) is not int
            or start < 0 or end < start or end >= n):
        raise ValueError("frame interval must be ordered zero-based output indices "
                         "inside the animation, with both ends given; "
                         "use seconds=[a, b] for the seconds spelling")
    return list(range(start, end + 1))


def declared_units(declarations):
    """Which unit this project's event and allowance windows are written in.

    Seconds when the declarations name none: the native layer documents
    seconds and writes no `units`. An animation project is seeded with
    `units: 'frames'`, and an edit that drops the key gets it back
    (`worker.keep_declared_units`), so its windows stay in frames.
    """
    unit = declarations.get("units", "seconds")
    if unit not in ("frames", "seconds"):
        raise ValueError("declarations units must be 'frames' or 'seconds'")
    return unit


def clip_offset(span, offset, *, clip, subject):
    """A clip-relative offset in output frames: `offset` action frames, scaled.

    `{'clip': name, 'frame': n}` names frame n of the clip's action. A strip
    plays its action from the action's first frame `a`, at scale s, so frame n
    lands (n - a)*s output frames after the strip's start: an action keyed only
    from frame 3 still puts its frame-5 state on that pose. States, camera cuts,
    and event/allowance windows all take that spelling and must agree on it, or
    a retimed strip leaves some of them outside their clip.

    A clip is an author-chosen name unioned over every object carrying a strip
    by it (see `worker.clip_spans`), so its extent is always defined but a
    single scale is not; strips that disagree are refused rather than one of
    them being picked by iteration order; so are strips whose actions start on
    different frames. A span without `scales` or `starts` reads as unscaled,
    or as starting at frame 0.
    """
    scales = span.get("scales") or [1.0]
    starts = span.get("starts") or [0.0]
    if len(scales) > 1:
        raise ValueError(
            f"{subject} binds into clip {clip!r}, whose strips carry different scales "
            f"({', '.join(f'{v:g}' for v in scales)}), so one clip-relative offset "
            "cannot name one output frame. Give the strips a common scale, or use an "
            "absolute frame")
    if len(starts) > 1:
        raise ValueError(
            f"{subject} binds into clip {clip!r}, whose strips play actions starting on "
            f"different frames ({', '.join(f'{v:g}' for v in starts)}). Key every object's "
            "action in the clip from the same first frame, or use an absolute frame")
    if offset < starts[0] - 1e-6:
        raise ValueError(
            f"{subject} names frame {offset:g} of clip {clip!r}, whose action starts on frame "
            f"{starts[0]:g}; an offset is a frame of the clip's action, so the clip's first "
            f"frame is {starts[0]:g}")
    return int(round((offset - starts[0]) * scales[0]))


def window_indices(info, declarations, record):
    """A declared window as inclusive zero-based output frame indices.

    `record["clip"]` makes the window an offset into that clip, which is the
    form that survives a retime: move or scale the strip and the window moves
    with it. Without it the window is absolute, in whichever unit
    `declarations["units"]` names. Seconds cannot be clip-relative: a clip's
    placement is frames, and mixing the two would be asking which one wins.
    """
    window = record["window"] if isinstance(record, dict) else record
    clip = record.get("clip") if isinstance(record, dict) else None
    if clip is None:
        if declared_units(declarations) == "seconds":
            return frame_indices(info, seconds=window)
        lo, hi = window
        if type(lo) is not int or type(hi) is not int:
            # `frame_indices`' own message points at `seconds=`, an argument of
            # `check_motion` rather than of a declaration.
            raise ValueError(f"window [{lo}, {hi}] must be whole output frames: these "
                             "declarations' units are frames")
        return frame_indices(info, lo, hi)
    if declared_units(declarations) == "seconds":
        raise ValueError("a clip-relative window needs declarations units 'frames'")
    span = (info.get("clips") or {}).get(clip)
    if span is None:
        known = sorted(info.get("clips") or {})
        raise ValueError(f"window names unknown clip {clip!r}; this project has "
                         + (", ".join(known) if known else "no NLA strips"))
    lo, hi = window
    if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo:
        raise ValueError("a clip-relative window is an ordered pair of non-negative offsets")
    start = span["frames"][0]
    subject = f"window [{lo}, {hi}]"
    first = start + clip_offset(span, lo, clip=clip, subject=subject)
    last = start + clip_offset(span, hi, clip=clip, subject=subject)
    if last > span["frames"][1]:
        raise ValueError(f"window [{lo}, {hi}] runs past the end of clip {clip!r} "
                         f"{span['frames']}")
    return frame_indices(info, first, last)


def allowed(allowances, a, b, frame):
    """`frame` is a zero-based output index; windows are resolved to those.

    `allowances` is `resolve_declarations`' index, `{frozenset((a, b)):
    [(first, last, max_depth_m), ...]}`, built once before the per-frame loop,
    so this is one dict lookup and an integer comparison rather than a scan of
    every record for every candidate pair on every frame.
    """
    return any(lo <= frame <= hi for lo, hi, _ in allowances.get(frozenset((a, b)), ()))


def allowance_limit(allowances, a, b, frame):
    """The depth the allowances in force at `frame` let `a`/`b` reach.

    `None` when one of them states no `max_depth_m` (an unbounded allowance
    excuses any depth), otherwise the largest stated bound. Only meaningful
    where `allowed` is true.
    """
    limits = [limit for lo, hi, limit in allowances.get(frozenset((a, b)), ()) if lo <= frame <= hi]
    if not limits or any(limit is None for limit in limits):
        return None
    return max(limits)


#: The operands each event predicate reads. Anything else is not a predicate
#: the frame loop can evaluate.
_EVENT_OPERANDS = {"touch": ("a", "b"), "separated": ("a", "b"), "above": ("a", "b"),
                   "joint_at": ("joint", "value"), "planted": ("a",),
                   "supported": ("a",)}


def resolve_declarations(declarations, baseline, info):
    """Every declaration record validated and resolved, and the ones that fail.

    Returns `(allowances, events, errors)`: `allowances` is the index
    `allowed` reads, `events` is `[(event, frame_indices)]` for the events
    that parsed, and `errors` names each record that did not.

    **Collected, not raised.** A raise here would kill the whole check, so the
    compile would return a traceback instead of its verdicts. A bad record is
    one finding among many and is reported as one.

    The two kinds do not degrade the same way, and that is deliberate. A
    dropped **allowance** makes the check stricter — an overlap it would
    have excused is reported — so skipping it is safe. A dropped **event**
    would silently remove an obligation, so `verify()` turns each one into
    an `invalid` row that fails `EventsSatisfied`.

    Still raised: `declarations` that is not a dict, and a `units` that is
    unreadable. Neither has a per-record degradation — no window anywhere
    can be read.

    **Nothing here writes to `declarations`.** The worker writes `index.json`
    from that same object and replay compares it with a clean copy, so any
    write here would make `final_render` report changed declarations nobody
    changed.
    """
    if not isinstance(declarations, dict):
        raise ValueError("declarations must be a JSON object")
    declared_units(declarations)
    errors = []
    names = _declarable_names(baseline, info)
    if baseline.get("room_objects"):
        # `room.floor` is what the layer prompt documents and what event
        # operands resolve, so it has to be spellable here too. A room
        # allowance is inert: the pair loop never tests a room body, which is
        # why resting on the floor needs no allowance.
        names.update("room." + tag for tag in ROOM_SURFACE_TAGS)
    allowances = {}
    for index, rec in enumerate(declarations.get("collision_allowances") or ()):
        try:
            if (not isinstance(rec, dict) or not rec.get("reason")
                    or not isinstance(rec.get("a"), str) or not isinstance(rec.get("b"), str)):
                raise ValueError("a collision allowance needs string a/b identities and a reason")
            a, b = rec["a"], rec["b"]
            if "instance" in rec:
                a, b = rec["instance"] + "/" + a, rec["instance"] + "/" + b
            unknown = [ref for ref in (a, b) if ref not in names]
            if a == b or unknown:
                raise ValueError(
                    f"collision allowance {a!r} / {b!r}: "
                    + (f"unknown {', '.join(map(repr, unknown))}; " if unknown else "")
                    + "operands are two distinct part identities ('instance/part')"
                    + (", native object names" if baseline.get("allow_native_collision_allowances")
                       else "")
                    + (" or room surfaces ('room.floor')" if baseline.get("room_objects") else ""))
            span = window_indices(info, declarations, rec) if "window" in rec else None
            limit = rec.get("max_depth_m")
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                                      or not math.isfinite(limit) or limit < 0):
                raise ValueError(f"collision allowance {a!r} / {b!r}: max_depth_m must be a "
                                 f"non-negative number of meters, got {limit!r}")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(_declaration_error("collision_allowance", index, rec, error))
            continue
        allowances.setdefault(frozenset((a, b)), []).append(
            ((span[0], span[-1]) if span else (-math.inf, math.inf))
            + (None if limit is None else float(limit),))
    errors.extend(resolve_liquids(declarations, baseline, info)[1])
    events = []
    # What a `supported` event's `a` can name: the instances whose parts are
    # bodies here, from the design and from the evaluated scene's objects.
    instances = {name.split("/")[0] for name in (*baseline["identities"], *info.get("collision_objects", ()))
                 if "/" in name}
    for index, event in enumerate(declarations.get("events") or ()):
        try:
            if not isinstance(event, dict):
                raise ValueError("an event is an object with name, predicate, and window")
            if not isinstance(event.get("name"), str) or not event["name"]:
                raise ValueError("an event needs a name")
            if event.get("mode", "all") not in ("all", "any"):
                raise ValueError("event mode must be all or any")
            predicate = event.get("predicate")
            if predicate not in _EVENT_OPERANDS:
                raise ValueError(f"unsupported event predicate {predicate!r}; use one of "
                                 + ", ".join(sorted(_EVENT_OPERANDS)))
            missing = [key for key in _EVENT_OPERANDS[predicate] if event.get(key) is None]
            if missing:
                raise ValueError(f"a {predicate} event needs {', '.join(missing)}")
            # `supported` reads its instance's parts by prefix, so a name that
            # matches none would leave the margin unmeasured, which counts as
            # holding. The other operands resolve per frame and report `missing`.
            if predicate == "supported" and instances and event["a"] not in instances:
                raise ValueError(f"a {event['a']!r} names no instance of this scene")
            if "window" not in event:
                raise ValueError("an event needs a window")
            events.append((event, window_indices(info, declarations, event)))
        except (ValueError, TypeError, KeyError) as error:
            errors.append(_declaration_error("event", index, event, error))
    return allowances, events, errors


def _declarable_names(baseline, info):
    """What a declaration may name: part identities, and native objects where the design allows."""
    names = set(baseline["identities"])
    if baseline.get("allow_native_collision_allowances"):
        names.update(info.get("collision_objects", ()))
    return names


def resolve_liquids(declarations, baseline, info):
    """`(liquids, errors)`: the objects declared liquid, and the entries that name none.

    A liquid is never a `NoPenetration` candidate, against anything, and is not
    held to the room: water displaced by a float, poured into a cup, or flowing
    past a hull is not a penetration, and on recorded fluid films authors
    declared 22 and 17 water allowances to say so while the check spent 491 s
    of a cold compile proving floats did not cross water surfaces. Liquids stay
    in events and in the contact checks.

    An entry is a part identity (`instance/part`), an instance name (all its
    parts), a native object name where the design allows those, or
    `{"object": ..., "reason": ...}` naming one. Collected like a collision
    allowance: a dropped entry only makes the check stricter.
    """
    names = _declarable_names(baseline, info)
    instances = {identity.split("/")[0] for identity in baseline["identities"]}
    liquids, errors = set(), []
    entries = declarations.get("liquids") or ()
    if not isinstance(entries, (list, tuple)):
        return liquids, [_declaration_error("liquid", 0, entries, ValueError(
            "liquids is a list of part identities, instance names, or object names"))]
    for index, entry in enumerate(entries):
        ref = entry.get("object") if isinstance(entry, dict) else entry
        try:
            if not isinstance(ref, str) or not ref:
                raise ValueError("a liquid is a part identity ('instance/part'), an instance "
                                 "name, or a native object name")
            if ref not in names and ref not in instances:
                raise ValueError(f"liquid {ref!r} names no part, instance, or object of this scene")
        except ValueError as error:
            errors.append(_declaration_error("liquid", index, entry, error))
            continue
        liquids.add(ref)
    return liquids, errors


def _declaration_error(kind, index, rec, error):
    """One rejected record, named the way its author would look for it."""
    row = {"kind": kind, "index": index}
    if isinstance(rec, dict):
        if isinstance(rec.get("name"), str):
            row["name"] = rec["name"]
        for key in ("a", "b", "instance", "window", "clip", "object"):
            if key in rec:
                row[key] = rec[key]
    # `str(KeyError('x'))` is `"'x'"`, which reads as nothing.
    row["reason"] = (f"missing {error.args[0]!r}" if isinstance(error, KeyError)
                     else str(error))
    return row


def validate_declarations(declarations, baseline, info):
    """Every problem `verify()` would report about these declarations, as a list.

    Pure, and needs no evaluated frame: `info` supplies only `n_frames`,
    `fps`, `clips`, and `collision_objects`. It exists so a caller that
    *writes* declarations can reject a bad record at the moment it is written,
    rather than letting it be discovered at the next check. Today the first
    validation of any declaration is inside `verify()`.
    """
    try:
        return resolve_declarations(declarations, baseline, info)[2]
    except ValueError as error:
        return [{"kind": "declarations", "reason": str(error)}]


def declaration_repair(errors, kind):
    """The repair for rejected records of one kind: fix the record, not the scene."""
    rows = [row for row in errors if row["kind"] == kind]
    if not rows:
        return []
    noun = {"collision_allowance": "collision allowance", "liquid": "liquid"}.get(kind, "event")
    first = rows[0]
    consequence = {"collision_allowance": "It was skipped, so the overlaps it meant to excuse "
                                          "are reported.",
                   "liquid": "It was skipped, so that object is checked as a solid."}.get(
                       kind, "It is reported as an unmet `invalid` event until it parses.")
    return [f"{len(rows)} {noun} declaration(s) could not be read; the first, at index "
            f"{first['index']}, because {first['reason']}. {consequence} Fix the "
            "declaration rather than the geometry."]


#: How many object definitions `inherited_allowances` lists in full. Every
#: one that exempts its complete part graph is still *named*, uncapped, in
#: `objects_exempting_every_pair`: that list is the point.
INHERITED_GROUP_LIMIT = 12


def inherited_allowances(evidence, catalog):
    """Inherited allowances per object definition, against all possible pairs.

    Grouped by the definition that generated them and compared with
    `C(parts, 2)`. Honest adjacency lists exempt a small fraction of the
    possible pairs; a definition that exempts every pair means no part of the
    object is checked against another part of itself, and `NoPenetration`
    passes on it.

    So no threshold is needed and none is applied: at 1.00 there is a fact
    to state (`exempts_every_pair`), and below it the fraction is handed over
    rather than judged. This never changes a verdict; it makes a `pass`
    readable as the narrower claim it can be.

    `evidence` is `InheritedAllowances.evidence`: one record per allowance per
    instance, each carrying `instance` and `object_ref`.
    """
    groups = {}
    for rec in evidence:
        held = groups.setdefault(rec["object_ref"], {"instances": set(), "pairs": set()})
        held["instances"].add(rec["instance"])
        held["pairs"].add(frozenset((rec["a"], rec["b"])))
    rows, complete = [], []
    for ref, held in groups.items():
        parts = len((catalog.get(ref) or {}).get("parts") or ())
        possible = parts * (parts - 1) // 2
        row = {"object_ref": ref, "instances": sorted(held["instances"]),
               "parts": parts, "pairs_exempted": len(held["pairs"]),
               "possible_pairs": possible}
        if possible:
            row["fraction_of_all_pairs"] = round(len(held["pairs"]) / possible, 4)
            if len(held["pairs"]) >= possible:
                row["exempts_every_pair"] = True
                complete.append(ref)
        rows.append(row)
    rows.sort(key=lambda row: (-row["pairs_exempted"], row["object_ref"]))
    out = {"groups": rows[:INHERITED_GROUP_LIMIT]}
    if len(rows) > INHERITED_GROUP_LIMIT:
        out["groups_total"] = len(rows)
    if complete:
        out["objects_exempting_every_pair"] = sorted(complete)
    return out
