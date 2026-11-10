"""Split a whole-timeline check across processes, and put the findings back together.

The frame loop is frame-local apart from three things, and each one sets a rule
the split has to respect:

* **Look-back and look-around.** `MotionContinuity` compares a frame with its
  predecessor and ranks a spike against neighbors within three frames, and an
  overlap may have begun before the chunk starts. So a chunk *evaluates*
  `CONTEXT` frames either side of the range it *owns* and *measures* only the
  owned range -- `verify.measure(owned=...)` does the narrowing at the source,
  so every frame is judged with the evidence a single pass would have had and
  recorded exactly once.
* **Contiguity.** An overlap span and a room escape are runs of frames, and a
  run may cross a boundary.
* **Declared windows.** An event's window, a slip anchor, a support phase, and
  "did this body move" all span frames, and may span chunks.

The last two are why a chunk's *reports* are never merged. Each chunk returns
`verify.measure`'s per-frame rows, `merge_rows` concatenates them in frame
order, and `verify.aggregate` -- the function a single pass calls on its own
rows -- decides everything that spans frames, once. A chunked report therefore
equals the single-process report by construction. Merging per-verifier reports
instead would have to guess an event's mode, keep one chunk's evidence for an
advisory check, and anchor a `planted` slip where a chunk began.

Diagnostic counters that belong to a process rather than to a frame --
`bvh_builds`, the pair-cache hit and miss counts -- are summed, so they report
the work actually done rather than what one process would have done.
"""
from __future__ import annotations

try:
    from .verify import ROWS_FORMAT, aggregate
except ImportError:  # bound flat into the worker's /tools, for a re-check
    from verify import ROWS_FORMAT, aggregate

#: Frames either side of an owned range that a chunk evaluates for context but
#: does not report on: one for the step into the first owned frame, three more
#: for the neighborhood a spike is ranked against.
CONTEXT = 4


def plan_chunks(frames, workers, *, events=(), info=None, context=CONTEXT):
    """Owned ranges and the frames each chunk must evaluate, or None to run whole.

    Returns `[(owned_lo, owned_hi, [frames to evaluate]), ...]`. Owned ranges
    partition the timeline; the evaluated list adds `context` frames either
    side, which are read for look-back and neighborhood only.

    Boundaries are placed evenly and are free to fall anywhere, because a chunk
    reports what it saw of a declared event rather than deferring it -- the
    alternative, keeping every event window inside one chunk, cannot split this
    layer's real content at all: events are often "true from here to the
    end", which can rule out a boundary over much of a film.
    """
    frames = list(frames)
    if workers < 2 or len(frames) < 2 * (context + 1) * workers:
        return None
    if frames != list(range(frames[0], frames[-1] + 1)):
        return None                      # a scattered set has no interior to chunk
    lo, hi = frames[0], frames[-1]
    bounds = [lo + round(i * (hi + 1 - lo) / workers) for i in range(workers + 1)]
    chunks = [(start, end - 1) for start, end in zip(bounds, bounds[1:]) if end > start]
    if len(chunks) < 2:
        return None
    return [(start, end, list(range(max(lo, start - context), min(hi, end + context) + 1)))
            for start, end in chunks]


#: Context every chunk of one check must agree on: declared input, scene
#: identity, and the room shell, which each process reads on its own first
#: frame -- a wall hidden there and not elsewhere gives two processes two
#: containments. A disagreement means the chunks were not checking one
#: project the way one pass would.
_SHARED_CONTEXT = ("fps", "identity_changed", "events", "declaration_errors", "allowances",
                   "liquids", "room")


class ContextMismatch(ValueError):
    """Two measured documents that cannot be one check's: measure it whole instead."""


def merge_rows(bundles):
    """One `verify.measure` document from several chunks', as one pass would have measured it.

    Owned ranges partition the checked frames, so each list of rows
    concatenates in frame order and each count sums. The scene's room is read
    from the first chunk's first frame, which is where a single pass reads it.
    """
    if any((bundle or {}).get("format") != ROWS_FORMAT for bundle in bundles):
        raise RuntimeError("a chunk returned no verification rows")
    bundles = sorted(bundles, key=lambda bundle: bundle["frames"][0])
    first = bundles[0]
    for bundle in bundles:
        for key in _SHARED_CONTEXT:
            if bundle["context"].get(key) != first["context"].get(key):
                raise ContextMismatch(f"chunks of one check disagree on {key}")
    frames = [frame for bundle in bundles for frame in bundle["frames"]]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise RuntimeError("chunk owned ranges overlap")

    def combine(parts):
        if isinstance(parts[0], list):
            return [item for part in parts for item in part]
        if isinstance(parts[0], dict):
            # The union of keys: a row kind a chunk from an older checker lacks
            # (`NoPenetration.allowed`, a later addition) is simply empty there.
            keys = list(dict.fromkeys(key for part in parts for key in part))
            return {key: combine([part.get(key, [] if any(isinstance(p.get(key), list) for p in parts)
                                          else {} if any(isinstance(p.get(key), dict) for p in parts)
                                          else 0) for part in parts]) for key in keys}
        return sum(parts)

    return {"format": ROWS_FORMAT, "frames": frames, "context": first["context"],
            "stats": combine([bundle["stats"] for bundle in bundles]),
            "rows": combine([bundle["rows"] for bundle in bundles])}


#: How far a changed frame reaches, before and after it, in what a check
#: concludes. Every verifier's rows are per frame (`verify.measure`), except
#: that `MotionContinuity` ranks a frame's step against the steps up to three
#: frames either side, and a changed frame changes the steps into and out of
#: it -- so frames from three before it to four after it can conclude
#: differently. Read with `CONTEXT` frames around them, as a chunk is.
REACH = (3, 4)


def reverify_frames(info, changed):
    """`(owned, evaluated)`: what a re-check must re-verify after `changed` frames changed.

    `owned` is every frame whose rows the changes can reach (`REACH`); the
    re-check measures those, reading `evaluated` -- `owned` plus `CONTEXT`
    either side -- for their neighborhoods. Exact, not conservative: with
    per-frame rows, an event's window needs re-measuring only where its
    frames changed, and the merged report equals a whole-film check's.
    """
    last = info["n_frames"] - 1
    owned = sorted({frame for c in changed
                    for frame in range(max(0, c - REACH[0]), min(last, c + REACH[1]) + 1)})
    evaluated = sorted({frame for o in owned
                        for frame in range(max(0, o - CONTEXT), min(last, o + CONTEXT) + 1)})
    return owned, evaluated


def merge_scoped(base, scoped):
    """One whole-film rows document: `base`'s rows, with `scoped`'s frames replaced.

    `base` is the last whole-film check's document and `scoped` a re-check of
    the frames a repair could have changed (`reverify_frames`). Every row
    names its frame, so the rows of those frames are taken from the re-check
    and every other frame's from the base. `verify.fill_diagnostics` then
    measures what the merged overlap spans lack, and `verify.aggregate`
    reports it as a single check would.
    """
    for bundle in (base, scoped):
        if (bundle or {}).get("format") != ROWS_FORMAT:
            raise ValueError("not a verification rows document")
    for key in _SHARED_CONTEXT:
        if base["context"].get(key) != scoped["context"].get(key):
            raise ContextMismatch(f"the earlier check and the re-check disagree on {key}")
    owned = set(scoped["frames"])
    if not owned <= set(base["frames"]):
        raise RuntimeError("the re-check covers frames the earlier check did not")

    def merge(old, new):
        if isinstance(old, dict):
            # Union of keys, as `combine` does: a base checked before a row kind
            # existed has none of it, and the re-check's rows are all there are.
            return {key: merge(old.get(key, type(new.get(key, []))()), new.get(key, type(old.get(key, []))()))
                    for key in dict.fromkeys(list(old) + list(new))}
        return sorted([row for row in old if row["frame"] not in owned] + list(new),
                      key=lambda row: row["frame"])

    return {"format": ROWS_FORMAT, "frames": base["frames"], "context": scoped["context"],
            "stats": scoped["stats"], "rows": merge(base["rows"], scoped["rows"]),
            "reverified": sorted(owned)}
