"""Render the checked working revision, then let finish package that revision."""
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import shutil
import time

from ._process import run_cancellable
from .blender.bundle import file_hash, tree_hash, write_json
from .blender.render_config import (DIAGNOSTIC_RENDER_SETTINGS, final_render_settings,
                                     check_size, over_defaults, unsized)
from .context import read_json
from .planning import DEFAULT_MIN_PIXELS, frame_spans

#: What a delivery-grade `render()` leaves in its own job directory so a later
#: `final_render` can admit the frames it drew.
#:
#: Written *before* the first frame, because the case this exists for is a
#: render that does not finish: a render that dies partway leaves finished
#: frames on disk, and without this record a resubmission at the same project
#: version would draw them all again.
DELIVERY_RECORD = "delivery_frames.json"


def _runs(indices):
    """Contiguous runs of frame indices, so a gap reads as `402-509`.

    A coverage failure that lists 108 integers is a failure an author has to
    parse before acting on. The same information as ranges is one glance.
    """
    runs, indices = [], sorted(indices)
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [f"{low}" if low == high else f"{low}-{high}" for low, high in runs]


def png_complete(path):
    """Whether a PNG was finished, rather than merely started.

    The last eight bytes of a complete PNG are a zero-length `IEND` chunk and
    its constant CRC. A worker killed mid-write leaves the file short, and a
    recorded host OOM left exactly that shape: **191 files for 190 finished
    frames**, caught only by noticing the off-by-one. ffmpeg would reject the
    truncated frame later and blame
    the encode; this names the frame while it can still be re-drawn.
    """
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as handle:
            handle.seek(-8, os.SEEK_END)
            return handle.read() == b"IEND\xaeB`\x82"
    except OSError:
        return False


def encode_and_inspect(result, rendered, directory, settings, cancel):
    if settings["strategy"] == "batch":
        if result.get("clip") != "clip.mp4":
            raise RuntimeError("renderer did not produce clip.mp4")
        clip = rendered / "clip.mp4"
    else:
        clip = directory / "clip.mp4"
        command = ["ffmpeg", "-y", "-v", "error", "-framerate", str(result["fps"]),
                   "-i", str(rendered / "frames/%06d.png"), "-c:v", "libx264"]
        if "threads" in settings:
            command += ["-threads", str(settings["threads"])]
        run_cancellable([*command, "-pix_fmt", "yuv420p", str(clip)], check=True, cancel=cancel)
    probe = run_cancellable(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_read_frames,avg_frame_rate,r_frame_rate",
        "-of", "json", str(clip)], check=True, capture_output=True, text=True, cancel=cancel)
    streams = json.loads(probe.stdout).get("streams", [])
    encoded = streams[0] if len(streams) == 1 else {}
    expected = (settings["width"], settings["height"], "h264", "yuv420p", str(result["n_frames"]))
    if tuple(encoded.get(k) for k in ("width", "height", "codec_name", "pix_fmt", "nb_read_frames")) != expected:
        raise RuntimeError(f"film dimensions/codec/frame count do not match the project: {encoded}")
    for key in ("avg_frame_rate", "r_frame_rate"):
        if abs(float(Fraction(encoded[key])) - result["fps"]) > .001:
            raise RuntimeError("film FPS does not match the constant-rate project timeline")
    return clip, encoded


#: The fewest frames a whole-film estimate is priced from. Below it the first
#: frame's warm-up -- BVH build, kernel load, the first sync -- is most of the
#: sample, and one- and two-frame probes at comparable settings can differ by
#: an order of magnitude a frame.
MIN_FRAMES_FOR_ESTIMATE = 5


def render_budget(service, drawing_s, frames_drawn, n_frames=None, *,
                  rendered=None, delivery=False):
    """What this render says the whole film would cost.

    The prompt tells an author to "time one or two frames at intended settings
    and multiply by frame count"; this does the multiplication in the reply, so
    a film planned at one resolution and frame count early can be re-priced
    against what a frame actually costs.

    The rate is the render loop's own, read from the `progress.json` the
    worker leaves in `rendered`: from the second frame on where the worker
    reported it, because the first frame carries the scene's synchronization
    and, averaged into a few frames, overstates a film severalfold; the first
    frame's time is reported beside it as `first_frame_s`.
    The loop's mean, which still includes that first frame, is the next
    fallback, and job wall time the last; the basis says which was used. No
    film estimate comes from fewer than `MIN_FRAMES_FOR_ESTIMATE` frames.

    `measures` says whether this was a delivery-grade render. A draft skips
    the per-frame replay check a delivery render runs, so it is a lower bound
    on delivery at the same settings, sometimes by a factor of several.
    `frames_in_film` is the
    render's own count, falling back to the last evaluation's -- and where
    neither knows it, the rate is reported and the estimate left `None`.
    """
    if not frames_drawn or drawing_s <= 0:
        return None
    progress = read_json(Path(rendered) / "progress.json") if rendered else None
    loop = (progress if isinstance(progress, dict)
            and progress.get("phase") in ("render", "movie") else {})
    steady, loop_rate = loop.get("seconds_per_step_after_first"), loop.get("seconds_per_step_mean")
    first_frame_s = loop.get("first_step_s") if steady else None
    if steady:
        rate, basis = float(steady), ("render loop from the second frame on: Blender start-up, "
                                      "scene load and the first frame's scene sync excluded "
                                      "(first_frame_s is that first frame)")
    elif loop_rate:
        rate, basis = float(loop_rate), ("render loop mean: Blender start-up excluded, the "
                                         "first frame's scene sync included")
    else:
        rate, basis = drawing_s / frames_drawn, ("job wall time, Blender start-up included, "
                                                 "divided by the frames drawn")
    n_frames = n_frames or (getattr(service, "timeline_hint", None) or {}).get("n_frames")
    enough = frames_drawn >= MIN_FRAMES_FOR_ESTIMATE
    budget = {"measured_s_per_frame": round(rate, 3), "frames_timed": frames_drawn,
              "elapsed_s": round(drawing_s, 1), "frames_in_film": n_frames,
              "estimated_full_film_seconds": (round(rate * n_frames)
                                              if n_frames and enough else None),
              "measures": "delivery" if delivery else "draft",
              "basis": basis}
    if first_frame_s is not None:
        budget["first_frame_s"] = first_frame_s
    if not enough:
        budget["note"] = (f"{frames_drawn} frame(s) is too few to price a film: the first "
                          f"frame's warm-up dominates. Time at least {MIN_FRAMES_FOR_ESTIMATE}.")
    if not delivery:
        budget["delivery_note"] = (
            "A draft's rate is a lower bound on delivery at these settings; price delivery "
            f"with render(delivery=True, start=..., end=...) over at least "
            f"{MIN_FRAMES_FOR_ESTIMATE} frames.")
    remaining = service.wall_budget()
    if remaining:
        budget["run"] = remaining
    return budget


def delivery_defaults(service, workspace, requested, info=None):
    """Delivery settings with a bare request's size filled in at the plan's floor.

    When the request names neither dimension, it gets the plan's delivery floor
    (960x540 pixels by default, or a larger `min_pixels`) at the aspect ratio of
    the project's own render resolution, rounded up to the even sizes H.264
    needs. Anything the author did state is kept. `info` is the evaluation the
    frames are certified against; without it, the latest one this session saw.
    """
    check_size(requested)
    requested = dict(requested or {})
    if "width" in requested or "height" in requested:
        return requested
    pixels = service.plan_of(workspace).get("delivery", {}).get("min_pixels", DEFAULT_MIN_PIXELS)
    source = info or getattr(service, "timeline_hint", None) or {}
    size = source.get("resolution") or ()
    aspect = size[0] / size[1] if len(size) == 2 and size[0] and size[1] else 16 / 9

    def even(value):
        return 2 * math.ceil(value / 2 - 1e-6)

    height = even(math.sqrt(pixels / aspect))
    width = even(height * aspect)
    while width * height < pixels:
        height += 2
        width = even(height * aspect)
    return {**requested, "width": width, "height": height}


def certifier(service, workspace, directory, cancel, content, *, selection=None, frames=None):
    """The evaluation delivery frames are drawn against, and on whose authority.

    Drawing never waits on a verdict: this always returns something.
    In order of preference --

    * `whole` -- the cached whole-film check at this content version, passing
      or failing. A failing one still certifies its clean frames; its
      failures are listed in `unmet` for the frames they touch.
    * `interval` -- a remembered `check_motion`/`compile` interval at this
      version that covers every frame asked for.
    * `evaluation` -- an evaluation-only pass over the selection, run here.
      The replay check still compares every drawn frame with it, and the
      frames are listed `unchecked`: no verifier has looked at them.

    The replay is the constant. It is a render-time hook, so a finished PNG
    cannot be shown afterwards to have come from checked geometry; whatever
    certified the range is what each frame is compared with as it is drawn.
    """
    key = service.cache_key(content)
    cached = service.validation_cache.get(key)
    if cached and tree_hash(Path(cached["path"])) == cached["hash"]:
        checked = service.check(workspace, directory / "check", cancel, version=content)
        return {"kind": "whole", "evaluation": checked["evaluation"],
                "path": checked["evaluation_path"], "check": checked,
                "job": service.job_of(checked["evaluation_path"])}
    if frames is not None:
        wanted = set(frames)
        covering = [entry for entry in service.interval_checks.get(key, {}).values()
                    if wanted <= set(entry["evaluation"].get("frames") or ())
                    and tree_hash(Path(entry["path"])) == entry["hash"]]
        if covering:
            entry = min(covering, key=lambda e: len(e["evaluation"]["frames"]))
            return {"kind": "interval", "evaluation": entry["evaluation"], "path": entry["path"],
                    "job": service.job_of(entry["path"])}
    selection = {k: v for k, v in (selection or {}).items() if v is not None}
    output = directory / "evaluated"
    info = service.worker(workspace, output, {"op": "evaluate", **selection}, cancel)
    # The latest evaluation, like any check's: it knows the film's length.
    service.timeline_hint = {k: info.get(k) for k in service.TIMELINE_KEYS}
    return {"kind": "evaluation", "evaluation": info, "path": str(output / "evaluation"),
            "job": directory.name}


def identities(chosen, frames, drawn=None):
    """`({frame: [signature, render_state]}, render_state_version)` for drawn frames.

    What a later `final_render` admits a frame by, at any index: the
    renderer's own `drawn` identities when it returns them, else the
    certifying evaluation's, which the replay check just matched each drawn
    frame against. Empty when the evaluation carries no render-state digest
    -- a signature alone admits nothing, because geometry is not all a frame
    shows -- and then admission falls back to the same content version.
    """
    info = chosen["evaluation"]
    states, version = info.get("render_states"), info.get("render_state_version")
    if drawn:
        return ({str(f): [v.get("signature"), v.get("render_state")] for f, v in drawn.items()},
                version)
    if not states or not version:
        return {}, None
    from .blender.evaluation import read_record
    out = {}
    for frame in frames:
        try:
            signature = read_record(Path(chosen["path"]), frame).get("signature")
        except Exception:                                       # noqa: BLE001
            signature = None
        out[str(frame)] = [signature, states.get(str(frame))]
    return out, version


def drawing_items(mismatches=(), draft=()):
    """`unmet` records for what the drawing itself recorded."""
    items = []
    if mismatches:
        # A mismatch is recorded and the render carries on.
        items.append({"kind": "replay_mismatch",
                      "frames": frame_spans(m["frame"] for m in mismatches),
                      "rows": list(mismatches)[:50]})
    if draft:
        spans = frame_spans(draft)
        items.append({"kind": "draft_quality", "frames": spans})
        items.append({"kind": "unverified_frames", "frames": spans})
    return items


def draft_settings(requested, applied):
    """The settings a draft's frames compose under, or `None` if they cannot.

    `requested` is the request as `render` completed it from the diagnostic
    default. Its size is the one the worker drew at, which differs from the
    request's when the request named no size and the worker fitted the
    default to the project's aspect. A draft with no `render_settings` drew at
    the worker's own default, which only the worker can name; its record of
    what it applied is turned back into a request here.
    """
    applied = applied or {}
    if requested is not None:
        size = {key: applied[key] for key in ("width", "height") if applied.get(key)}
        return {**final_render_settings({**requested, **size}), "strategy": "frames"}
    base = {key: applied.get(key) for key in ("engine", "width", "height")}
    if applied.get("engine") == "CYCLES" and applied.get("samples"):
        base["samples"] = applied["samples"]
    try:
        return {**final_render_settings(base), "strategy": "frames"}
    except (TypeError, ValueError):
        return None


def replay_mismatches(job, record):
    """The frames of a delivery render that did not reproduce their check.

    A finished render's record lists them. One the box killed part way
    through has only the record written before its first frame, and the
    worker's log of each mismatch as it found it.
    """
    if "replay_mismatch" in record:
        return record["replay_mismatch"] or []
    log = job / "render" / "replay_mismatch.jsonl"
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            break                    # the line being written when it died
    return rows


def admit(service, job_ids, requested, version, *, targets=None, blend_hash=None):
    """Which frames the submitted render jobs deliver, and on whose authority.

    The certification is separate from the renderer: welded to it, a render
    that dies would be worth nothing, a film could only be drawn in one serial
    pass, and running out of clock would mean downgrading the whole film rather
    than stopping with the shots that are finished. `preview` and delivery are
    the same worker op, `render_frames`, behind two session policies.

    Takes job ids and reads records the harness wrote -- never a path, a frame
    list or a settings claim from the caller. A frame is admitted:

    * **at the same content version**, from a delivery render, which
      replay-checked it as it drew;
    * **by identity, at any index**: a frame drawn at an older
      version whose (`signature`, `render_state`) pair equals the target's
      frame here. A held or shifted frame is reused instead of redrawn. Off
      when the evaluation carries no render-state digest;
    * **as a draft**, from a plain render of this same blend
      through the film's own cameras. Listed `draft_quality` and
      `unverified_frames`: no replay check ran.

    Frames from another project state are never admitted. They are named in
    `not_admitted`, and `final_render` draws those frames itself.
    """
    if not job_ids:
        raise ValueError("frames_from takes the job ids of delivery renders, "
                         "from render(..., delivery=True); omit it to draw the film here")
    frames, views, owner, settings, applied, source = {}, {}, {}, requested, None, None
    pool, draft, not_admitted, same, older, mismatches = {}, set(), {}, 0, {}, []
    rs_version = (targets or {}).get("version")
    for job_id in job_ids:
        job = service.jobs.root / str(job_id)
        record = job / DELIVERY_RECORD
        drawn = json.loads(record.read_text()) if record.is_file() else {}
        delivery = bool(drawn.get("delivery") and drawn.get("verify_replay"))
        is_draft = drawn.get("delivery") is False and drawn.get("blend_hash") is not None
        if not (delivery or is_draft):
            raise ValueError(
                f"job {job_id} drew no frames final_render can use; draw them with "
                "render(..., delivery=True)")
        if is_draft and (not drawn.get("composable") or drawn.get("render_settings") is None):
            raise ValueError(f"job {job_id} was drawn through a diagnostic camera or with "
                             "settings that cannot be restated; it is a look, not film frames")
        if is_draft and drawn["blend_hash"] != blend_hash:
            raise ValueError(f"job {job_id} was drawn from another blend; draft frames are "
                             "admitted only from the project being delivered")
        current = is_draft or drawn.get("version") == version
        if not current and not (targets and drawn.get("render_state_version") == rs_version
                                and rs_version):
            not_admitted[str(job_id)] = (f"drawn at content version {drawn.get('version')}; "
                                         f"this delivery is {version}")
            continue
        if not current:
            older[job_id] = drawn.get("version")
        # `applied` is what the renderer resolved -- backends and devices the
        # request left open -- and is what the receipt and `camera.json`
        # record. `render_settings` is the authored request, and is what
        # admissibility compares, because an author states a request and not a
        # resolution.
        applied = applied or drawn.get("applied_render_settings")
        if settings is None:
            settings, source = drawn["render_settings"], job_id
        elif drawn["render_settings"] != settings:
            differs = sorted(k for k in set(settings) | set(drawn["render_settings"])
                             if settings.get(k) != drawn["render_settings"].get(k))
            where = f"job {source}" if source else "this call"
            raise ValueError(
                f"job {job_id} and {where} disagree on {', '.join(differs)}; one film is one "
                "set of render settings. Re-draw the odd jobs or state the settings you mean.")
        pairs = drawn.get("frames") or {}
        # A frame whose replay did not match is admitted with its mismatch,
        # which `unmet` then lists, and never by identity: the pair recorded
        # for it may be what it should have shown rather than what it shows.
        mismatched = {m["frame"]: m for m in replay_mismatches(job, drawn)}
        job_views = {int(p.stem): p for p in (job / "render" / "views").glob("*.json")}
        for image in sorted((job / "render" / "frames").glob("*.png")):
            index = int(image.stem)
            if not png_complete(image):
                # Dropped rather than refused: the truncated frame is almost
                # always the last one a killed worker started, and the author
                # wants to re-draw that one frame, not re-read this message.
                continue
            pair = pairs.get(str(index))
            if (pair and None not in pair and index not in mismatched
                    and drawn.get("render_state_version") == rs_version):
                pool.setdefault(tuple(pair), (job_id, image, job_views.get(index)))
            if not current:
                continue
            if index in owner:
                raise ValueError(
                    f"frame {index} is drawn by both job {owner[index]} and job {job_id}; "
                    "submit jobs whose frames do not overlap")
            owner[index], frames[index] = job_id, image
            same += 1
            if index in mismatched:
                mismatches.append(mismatched[index])
            if is_draft or index in set(drawn.get("draft_frames") or ()):
                draft.add(index)
            if index in job_views:
                views[index] = json.loads(job_views[index].read_text())
    by_identity = 0
    for index, pair in sorted(((targets or {}).get("frames") or {}).items()):
        index = int(index)
        if index in frames or not pair or None in pair:
            continue
        hit = pool.get(tuple(pair))
        if hit is None:
            continue
        owner[index], frames[index] = hit[0], hit[1]
        if hit[2] is not None:
            views[index] = {**json.loads(hit[2].read_text()), "frame": index}
        by_identity += 1
    # A job of another version that no frame here matched by identity is as
    # unused as one from before the digest, and is named the same way.
    for job_id, drawn_version in older.items():
        if job_id not in owner.values():
            not_admitted[str(job_id)] = (f"drawn at content version {drawn_version}; this "
                                         f"delivery is {version}, and none of its frames "
                                         "shows what a frame here does")
    return {"settings": settings, "applied": applied, "frames": frames, "views": views,
            "owner": owner, "draft": draft, "not_admitted": not_admitted,
            "same_version": same - len(draft), "by_identity": by_identity,
            "mismatches": mismatches}


def best_film(run_dir):
    """The best film a run leaves behind, read from records already on disk.

    For an interrupted run's `meta.json`, so a complete film that never
    committed is still recorded. The
    animation layer's `layer_meta` (`agent/layers/layer.py`)
    calls this; the records it reads are this module's.

    Cheap by construction, because it runs on the way out: it reads JSON and
    lists directories, and hashes nothing. So it does not say whether the film
    is current -- a caller that can afford a hash compares `version` with
    `AnimationSession.identity`'s content version (a draft's, with the blend's
    file hash).

    A finished `final_render` film (its receipt) wins. Otherwise the job whose
    frames cover the most of the film, delivery-grade before draft at equal
    coverage, latest last. Drafts through a diagnostic camera are not film and
    are skipped. `None` when nothing was drawn.
    """
    native = Path(run_dir) / "native"
    receipt = read_json(native / "delivery.json")
    film = Path((receipt or {}).get("artifacts") or "-") / "film" / "clip.mp4"
    if receipt and film.is_file():
        info = (receipt.get("checks") or {}).get("evaluation") or {}
        return {"kind": "delivery", "film": str(film), "version": receipt.get("version"),
                "frames": info.get("n_frames"),
                "unmet": sorted({item["kind"] for item in receipt.get("unmet") or ()})}
    best, rank = None, None
    for path in (native / "jobs").glob(f"*/{DELIVERY_RECORD}"):
        record = read_json(path) or {}
        delivery = bool(record.get("delivery") and record.get("verify_replay"))
        if not delivery and not record.get("composable"):
            continue
        frames = sorted(int(p.stem) for p in (path.parent / "render" / "frames").glob("*.png")
                        if p.stem.isdigit())
        if not frames:
            continue
        candidate = (len(frames), delivery, path.stat().st_mtime)
        if rank is None or candidate > rank:
            rank, best = candidate, {
                "kind": "delivery" if delivery else "draft", "job": path.parent.name,
                "version": record.get("version") if delivery else record.get("blend_hash"),
                "frames": len(frames), "spans": frame_spans(frames)}
    return best


def compose(frames, rendered):
    """Put the admitted frames where the encoder already looks.

    Hardlinked, not copied: `encode_and_inspect` reads
    `frames/%06d.png` out of one directory, and a 2,508-frame delivery is
    gigabytes. The same argument -- and the same safety argument, that nothing
    writes through the links -- is why `submit()` links a non-mutating job's
    workspace instead of copying it. Falls back to copying across a device
    boundary, which a run directory spanning two mounts can produce.
    """
    target = rendered / "frames"
    target.mkdir(parents=True, exist_ok=True)
    for index, image in sorted(frames.items()):
        destination = target / f"{index:06d}.png"
        try:
            os.link(image, destination)
        except OSError:
            shutil.copy2(image, destination)
    return target


def render(service, args, workspace, directory, cancel):
    """Draw and encode the delivery film, composing what is already drawn.

    **It always produces a film**. It uses whatever verdict exists --
    the cached whole-film check at this content version, failing or not --
    and an evaluation-only pass when there is none. It never runs a full
    check itself: the clean path is `compile()`, then
    `final_render(frames_from=[...])`, then `finish()`. Its receipt lists
    `unmet`, and a clean receipt means `unmet` is empty; only `finish()`
    decides whether to commit with exceptions.

    `frames_from=[job_id, ...]` admits the frames those renders drew (see
    `admit`), and this draws the rest itself -- the gaps a killed range left,
    frames another version drew, anything not yet drawn -- instead of
    refusing. Omitting it draws the whole film here, the one-call path.

    **It writes nothing into the project**. The settings are applied
    in this job's snapshot only, and `finish()` writes them into the bundle's
    copy of the blend. So a final render can run alongside an edit, and
    cannot be thrown away by one.
    """
    service.source_fresh(workspace)
    plan = service.plan_of(workspace)
    content = args.get("_content") or service.content_version(workspace)
    # Part by part, so a `finish()` refused for a stale film can say what changed.
    parts = service.content_parts(workspace)
    submitted = [str(job) for job in (args.get("frames_from") or ())]
    requested = args.get("render_settings")
    check_size(requested)
    chosen = certifier(service, workspace, directory, cancel, content)
    info = chosen["evaluation"]
    n_frames = info["n_frames"]
    ids, rs_version = identities(chosen, range(n_frames))
    if submitted:
        # Settings default to what the jobs already agree on rather than to a
        # default: they were declared when the frames were drawn, and deriving
        # one here would refuse a coherent submission for disagreeing with a
        # default nobody chose. Restating them is still allowed, and then they
        # have to match. `strategy` is pinned on both sides: composing reads
        # PNGs, and the frames were drawn with `strategy='frames'` for that
        # reason.
        restated = ({**final_render_settings(requested), "strategy": "frames"}
                    if requested is not None else None)
        admitted = admit(service, submitted, restated, content,
                         targets={"frames": ids, "version": rs_version},
                         blend_hash=file_hash(workspace / "animation.blend"))
        settings = admitted["settings"] or {**final_render_settings(
            delivery_defaults(service, workspace, None, info)), "strategy": "frames"}
    else:
        settings = final_render_settings(delivery_defaults(service, workspace, requested, info))
        admitted = {"frames": {}, "views": {}, "owner": {}, "draft": set(), "applied": None,
                    "not_admitted": {}, "same_version": 0, "by_identity": 0, "mismatches": []}
    if settings["width"] * settings["height"] < plan.get("delivery", {}).get("min_pixels", DEFAULT_MIN_PIXELS):
        raise ValueError("final render falls below the planned delivery target")
    drawn = {i: path for i, path in admitted["frames"].items() if i < n_frames}
    views = dict(admitted["views"])
    gaps = [i for i in range(n_frames) if i not in drawn]
    # The guard below means exactly "rendering changed the project": the
    # snapshot is never edited here, so anything that moves it was the render.
    checked_version = tree_hash(workspace)
    rendered = directory / "render"
    mismatches, drawn_seconds = list(admitted["mismatches"]), None
    shutil.copytree(Path(chosen["path"]), workspace / "evaluation")
    try:
        if not submitted:
            drawing_started = time.monotonic()
            result = service.worker(workspace, rendered, {"op": "render", "verify_replay": True,
                "render_settings": settings, "movie": settings["strategy"] == "batch"}, cancel)
            drawn_seconds = time.monotonic() - drawing_started
            mismatches = result.get("replay_mismatch") or []
            gaps = []
        else:
            if gaps:
                filled = service.worker(workspace, directory / "gaps", {
                    "op": "render", "verify_replay": True, "frames": gaps, "movie": False,
                    "render_settings": {**settings, "strategy": "frames"}}, cancel)
                mismatches += filled.get("replay_mismatch") or []
                for index in gaps:
                    drawn[index] = directory / "gaps" / "frames" / f"{index:06d}.png"
                for view in filled.get("views") or ():
                    views[view["frame"]] = view
            compose({index: drawn[index] for index in range(n_frames)}, rendered)
            result = {"fps": info["fps"], "n_frames": n_frames, "frame_start": info["frame_start"],
                      "render_settings": admitted["applied"] or settings,
                      "views": [views[i] for i in range(n_frames) if i in views]}
    finally:
        shutil.rmtree(workspace / "evaluation", ignore_errors=True)
    if tree_hash(workspace) != checked_version:
        raise ValueError("rendering changed project dependencies; bake/persist them before delivery")
    clip, encoded = encode_and_inspect(result, rendered, directory, settings, cancel)
    artifacts = directory / "delivery"
    shutil.copytree(Path(chosen["path"]), artifacts / "evaluation")
    film = artifacts / "film"
    film.mkdir()
    shutil.copy2(clip, film / "clip.mp4")
    camera = {"views": result.get("views", []), "fps": result["fps"], "n_frames": result["n_frames"],
              "frame_start": result["frame_start"], "render_settings": result["render_settings"],
              "requested_render_settings": settings, "encoded_video": encoded}
    write_json(film / "camera.json", camera)
    service.source_fresh(workspace)
    drawing = drawing_items(mismatches, sorted(admitted["draft"]))
    unmet = service.unmet(workspace, content=content, drawing=drawing)
    counts = {"same_version": admitted["same_version"], "by_identity": admitted["by_identity"],
              "draft": len(admitted["draft"]), "drawn_here": n_frames if not submitted else len(gaps)}
    check = chosen.get("check") or {"ok": False, "partial": False, "unchecked": True,
                                    "evaluation": info, "evaluation_path": chosen["path"],
                                    "verifier_reports": []}
    # `composed_from` so the receipt says what it is attesting to. A film
    # drawn in one pass and one assembled from six jobs are both delivered
    # here, and only the record distinguishes them afterwards. `version` is
    # the content version, which an acceptance does not move.
    receipt = {"version": content, "checks": check, "artifacts": str(artifacts),
               "artifacts_hash": tree_hash(artifacts), "render_settings": result["render_settings"],
               "requested_render_settings": settings, "composed_from": submitted or None,
               "certified_by": {"job": chosen.get("job"), "kind": chosen["kind"]},
               "admitted": counts, "unmet": unmet, "drawing": drawing,
               "submitted": args.get("_seq", 0), "content_parts": parts}
    with service.lock:
        # In submission order: final renders overlap edits and each other, and
        # an older one finishing last made `finish()` refuse the newer film.
        held = read_json(service.native / "delivery.json") or {}
        if receipt["submitted"] >= (held.get("submitted") or 0):
            write_json(service.native / "delivery.json", receipt)
    if (rendered / "frames").is_dir():
        # This job's frames are delivery frames too: if the project moves
        # on, `final_render(frames_from=[this job])` re-admits every frame
        # that is unchanged and draws only the rest.
        write_json(directory / DELIVERY_RECORD,
                   {"delivery": True, "verify_replay": True, "version": content,
                    "render_settings": {**settings, "strategy": "frames"},
                    "applied_render_settings": result["render_settings"],
                    "frames_root": str(rendered / "frames"), "scope": None,
                    "certified_by": receipt["certified_by"], "frames": ids,
                    "render_state_version": rs_version,
                    "draft_frames": sorted(admitted["draft"]),
                    "replay_mismatch": mismatches})
    payload = {"ok": True, "path": str(film / "clip.mp4"), "version": content, **camera,
               "unmet": unmet, "admitted": counts,
               "next": ("Inspect the film; finish() packages this project and film while its "
                        "content is unchanged." + (
                            " Nothing stands between it and a clean commit." if not unmet else
                            " `unmet` lists what a clean commit still needs; finish(exceptions=True)"
                            " commits it as delivered with those exceptions."))}
    if admitted["not_admitted"]:
        payload["not_admitted"] = admitted["not_admitted"]
    if drawn_seconds is not None:
        payload["budget"] = render_budget(service, drawn_seconds, result["n_frames"],
                                          result["n_frames"], rendered=rendered, delivery=True)
    if submitted:
        payload["composed_from"] = {
            job: ", ".join(_runs(i for i, held in admitted["owner"].items() if held == job))
            for job in submitted if any(held == job for held in admitted["owner"].values())}
        if gaps:
            payload["composed_from"]["drawn_here"] = ", ".join(_runs(gaps))
    return payload


def preview(service, args, workspace, directory, cancel):
    """Render frames to look at, and encode them when there is a sequence.

    It draws everything `final_render` can (same engines, dimensions, and samples, the whole film included) and
    guarantees nothing: no checks, no replay verification, no receipt, and nothing here satisfies `finish()`. The
    delivery path certifies that the film matches a checked project; this answers "what does it look like" while
    the project is still moving. `snapshot` is this op with encoding off and frame 0 as its bare default.

    Selection, camera, and output shape are independent arguments. Selection is `frames=[...]`, an interval
    (`start`/`end`, `seconds`, `shot`), or neither for the whole timeline. A set need not be contiguous, so a
    strided review (`frames=range(0, n, 4)`) is an ordinary call, encoded in frame order at the project's FPS.
    `encode` defaults to on whenever more than one frame was drawn. The PNG paths come back in the order asked for,
    and the clip is always chronological.

    Output lands under `previews/<job>/`, outside the workspace, so looking never changes the project's revision or
    invalidates a delivery receipt. `version` names the blend the frames were drawn from.

    **`delivery=True` is the exception.** The same worker op draws under the delivery defaults and the replay
    check, and leaves a record `final_render(frames_from=[...])` can admit at full trust. The replay check is a
    render-time hook, because a finished PNG cannot be shown afterwards to have come from checked geometry. It
    always draws: against the whole-film check at this content version if there is one, passing or failing, else a
    covering interval check, else an evaluation-only pass (`certifier`). What the range still lacks for a clean
    commit (failing or unchecked frames, frames outside every accepted span) comes back in `unmet` and stays in its
    record. Selection, the output folder and the absence of a receipt are as above.

    A plain render leaves a record too, so `final_render` can compose its frames as a draft when nothing better
    exists, but only frames drawn through the film's own cameras from the same blend.
    """
    service.build_of(workspace)
    requested = args.get("render_settings")
    for_delivery = bool(args.get("delivery"))
    if (requested or {}).get("strategy") == "batch":
        raise ValueError("batch renders the whole timeline in one native operation and "
                         "cannot take an interval or a frame set; previews use "
                         "strategy='frames', which also keeps the PNGs so a draft can "
                         "be read frame by frame")
    if for_delivery:
        # The documented delivery defaults, not the diagnostic ones: these
        # frames are the film. `strategy` is pinned because composition reads
        # PNGs and the batch path writes none.
        settings = {**final_render_settings(delivery_defaults(service, workspace, requested)),
                    "strategy": "frames"}
        plan = service.plan_of(workspace)
        if settings["width"] * settings["height"] < plan.get("delivery", {}).get(
                "min_pixels", DEFAULT_MIN_PIXELS):
            raise ValueError("delivery frames fall below the planned delivery target")
    else:
        # Omitted settings stay omitted, so the worker applies the diagnostic
        # default it already applies to every other look-at-it render: Workbench
        # at 1280x720. Filling in `final_render_settings` here instead would make
        # Cycles at 1920x1080 the default for a draft, and a full 1,800-frame
        # draft is 13.8 min in Workbench against roughly 87 at that setting.
        settings = None if requested is None else {
            **final_render_settings(over_defaults(requested, DIAGNOSTIC_RENDER_SETTINGS)),
            "strategy": "frames"}
    selection = {k: args.get(k) for k in ("frames", "start", "end", "seconds", "clip")}
    if selection["frames"] is not None and any(
            selection[k] is not None for k in ("start", "end", "seconds", "clip")):
        raise ValueError("supply frames=[...] or an interval, not both")
    request = {"op": "render", "movie": False, **selection,
               "fit_aspect": not for_delivery and unsized(requested)}
    if settings is not None:
        request["render_settings"] = settings
    for key in ("camera_name", "eye", "look_at", "up", "fov_deg"):
        if args.get(key) is not None:
            request[key] = args[key]
    rendered = directory / "render"
    if for_delivery:
        service.source_fresh(workspace)
        content = args.get("_content") or service.content_version(workspace)
        named = (selection["frames"] if selection["frames"] is not None
                 else list(range(selection["start"], selection["end"] + 1))
                 if (selection["start"] is not None and selection["end"] is not None
                     and selection["seconds"] is None and selection["clip"] is None)
                 else None)
        chosen = certifier(service, workspace, directory, cancel, content,
                           selection=selection, frames=named)
        # Sized again from the evaluation the frames are certified against: a
        # bare request takes the project's aspect, and the hint the size above
        # came from is whichever check finished last, possibly of another
        # version.
        settings = request["render_settings"] = {**final_render_settings(
            delivery_defaults(service, workspace, requested, chosen["evaluation"])),
            "strategy": "frames"}
        request["verify_replay"] = True
        # Written before the first frame, so a render the box kills half way
        # through still leaves something `final_render` can admit. That is the
        # whole point: the frames of an interrupted delivery render are on
        # disk today and there has never been a way to hand them back.
        record = {"delivery": True, "verify_replay": True, "version": content,
                  "render_settings": settings, "applied_render_settings": None,
                  "frames_root": str(rendered / "frames"),
                  "certified_by": {"job": chosen.get("job"), "kind": chosen["kind"]}}
        write_json(directory / DELIVERY_RECORD, record)
        shutil.copytree(Path(chosen["path"]), workspace / "evaluation")
    drawing_started = time.monotonic()
    try:
        result = service.worker(workspace, rendered, request, cancel)
    finally:
        drawing_s = time.monotonic() - drawing_started
        if for_delivery:
            shutil.rmtree(workspace / "evaluation", ignore_errors=True)
    frames = [int(Path(p).stem) for p in result["paths"]]
    blend_hash = None if for_delivery else file_hash(workspace / "animation.blend")
    if for_delivery:
        drawn_frames = sorted(set(frames))
        mismatches = result.get("replay_mismatch") or []
        unmet = service.unmet(workspace, content=content, frames=drawn_frames,
                              drawing=drawing_items(mismatches))
        ids, rs_version = identities(chosen, drawn_frames, result.get("drawn"))
        write_json(directory / DELIVERY_RECORD, {
            **record, "applied_render_settings": result.get("render_settings"),
            # The certified interval, `None` for the whole film, and the
            # verdict these frames were drawn under.
            "scope": None if chosen["kind"] == "whole" else [drawn_frames[0], drawn_frames[-1]],
            "checks_ok": not any(u["kind"] in ("check", "unchecked") for u in unmet),
            "unmet": unmet, "frames": ids, "render_state_version": rs_version,
            "replay_mismatch": mismatches})
    else:
        write_json(directory / DELIVERY_RECORD, {
            "delivery": False, "verify_replay": False, "blend_hash": blend_hash,
            # Through the film's own cameras, or it is a look rather than film.
            "composable": not any(args.get(k) is not None for k in ("camera_name", "eye", "look_at")),
            "render_settings": draft_settings(settings, result.get("render_settings")),
            "applied_render_settings": result.get("render_settings"),
            "frames_root": str(rendered / "frames")})
    out = service.root / "previews" / directory.name
    out.mkdir(parents=True, exist_ok=True)
    # Hardlinked, like the delivery frames above: nothing writes through
    # either copy, and a whole-timeline draft is hundreds of megabytes that a
    # copy would keep twice for the life of the run. Falls back to copying
    # across a device boundary.
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for image in sorted((rendered / "frames").glob("*.png")):
        destination = out / "frames" / image.name
        if destination.exists():
            continue
        try:
            os.link(image, destination)
        except OSError:
            shutil.copy2(image, destination)
    drawn = sorted(int(p.stem) for p in (out / "frames").glob("*.png"))
    # A directory, the indices drawn into it, and how to spell a name — not a
    # list of absolute paths. The paths are `frames_dir/(frame_file % i)` for
    # every `i` in `frame_indices`, so listing them spent about 100 bytes a
    # frame to repeat a prefix the reply already carries: ~180 KB on a
    # whole-timeline draft against ~9 KB for the indices alone. Nothing slims
    # a render reply — `slim_job_payload` walks `verifier_reports` and
    # `shots` — so the shape here is the whole budget.
    #
    # `frame_indices` keeps the caller's own order and repeats, which
    # `frames` cannot: an explicit sparse set is not recoverable from
    # start/end/count.
    payload = {"ok": True, "preview": True,
               "frames_dir": str(out / "frames"), "frame_indices": frames,
               "frame_file": "%06d.png",
               "frames": {"start": drawn[0], "end": drawn[-1], "count": len(drawn)},
               "fps": result["fps"],
               # Not `views`. The worker returns one camera record per frame,
               # which `final_render` writes into `camera.json` because a
               # delivery has to record the optics it shipped. A draft does
               # not, and nothing slims a render reply — `slim_job_payload`
               # only walks `verifier_reports` and `shots` — so a
               # whole-timeline preview would have carried ~1,800 4x4
               # matrices back to the author who asked to see a film.
               "render_settings": result.get("render_settings", settings),
               # A delivery render reports the project *revision* its frames
               # belong to, because that is the identity `final_render`
               # matches them against; a draft reports the blend it was drawn
               # from, which is all a picture needs to be tied to a state.
               "version": content if for_delivery else blend_hash,
               "budget": render_budget(service, drawing_s, len(drawn),
                                       result.get("n_frames"), rendered=rendered,
                                       delivery=for_delivery),
               "note": ("Delivery frames, verified as they were drawn against "
                        + {"whole": "the whole-film check", "interval": "an interval check"}.get(
                            chosen["kind"], "an evaluation no check has passed")
                        + ". Pass this job_id to final_render(frames_from=[...]); `unmet` lists "
                        "what these frames still lack for a clean commit."
                        if for_delivery else
                        "A look: no checks ran, and these frames reach a film only as drafts.")}
    # One grid beside the frames, on by default for a multi-frame look: authors
    # otherwise write their own contact-sheet builder, and one that does not
    # reviews frames one at a time and misses what a grid shows. Advisory by
    # construction:
    # `contact_sheet` returns [] on any failure and never raises, so a render
    # that succeeded is never lost to a sheet that could not be drawn. Not for
    # delivery frames -- those exist to be composed, not looked at.
    if not for_delivery and args.get("sheet", True):
        from .sheets import contact_sheet
        sheets = contact_sheet(((index, out / "frames" / ("%06d.png" % index))
                                for index in drawn), out / "sheet.jpg",
                               fps=result.get("fps"))
        if sheets:
            payload["sheet"] = sheets if len(sheets) > 1 else sheets[0]
        if len(sheets) > 1:
            # Paged sheets hold every frame; this is the one-page look at the
            # whole render that arms reviewing a film otherwise build by hand.
            from .sheets import overview_sheet
            overview = overview_sheet(((index, out / "frames" / ("%06d.png" % index))
                                       for index in drawn), out / "sheet_overview.jpg",
                                      fps=result.get("fps"))
            if overview:
                payload["sheet_overview"] = overview
    if for_delivery:
        payload.update(delivery_grade=True, job_id=directory.name, unmet=unmet,
                       certified_by=record["certified_by"],
                       render_settings=result.get("render_settings", settings))
    encode = args.get("encode")
    # A delivery render does not encode unless asked. Its frames exist to be
    # composed into one film by `final_render`, and encoding a 700-frame
    # fragment on the way there is minutes of ffmpeg for a clip nobody
    # watches. Asking for it is still legal: a per-shot delivery render is
    # exactly the thing an author wants to look at before drawing the rest.
    if encode is False or (encode is None and (for_delivery or len(drawn) < 2)):
        return payload
    clip = directory / "clip.mp4"
    # Glob rather than a numbered sequence, because an explicit frame set is
    # allowed to be sparse and `-start_number` with `%06d` stalls on the first
    # gap. The names are zero-padded indices, so lexical order is frame order:
    # the clip is chronological even when the frames were asked for in another
    # order, which is what a clip should be.
    command = ["ffmpeg", "-y", "-v", "error", "-framerate", str(result["fps"]),
               "-pattern_type", "glob", "-i", str(out / "frames" / "*.png"), "-c:v", "libx264"]
    if settings and "threads" in settings:
        command += ["-threads", str(settings["threads"])]
    run_cancellable([*command, "-pix_fmt", "yuv420p", str(clip)], check=True, cancel=cancel)
    probe = run_cancellable(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_read_frames,avg_frame_rate",
        "-of", "json", str(clip)], check=True, capture_output=True, text=True, cancel=cancel)
    streams = json.loads(probe.stdout).get("streams", [])
    encoded = streams[0] if len(streams) == 1 else {}
    if encoded.get("nb_read_frames") != str(len(drawn)):
        raise RuntimeError(f"preview encode dropped frames: wanted {len(drawn)}, got {encoded}")
    shutil.copy2(clip, out / "clip.mp4")
    return {**payload, "path": str(out / "clip.mp4"), "encoded_video": encoded}
