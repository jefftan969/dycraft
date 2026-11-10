"""Optional motion inspection and independent, advisory visual review."""
import json
import shlex
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .blender.render_config import (DIAGNOSTIC_RENDER_SETTINGS, REVIEW_RENDER_SETTINGS,
                                     final_render_settings, over_defaults, unsized)
from .context import workspace_context
from .planning import plan_states

#: How long the reviewer has to answer, from its launch.
REVIEW_TIMEOUT_S = 600


def rgb_temporal(images, gain=4.0):
    if len(images) not in (2, 3) or not np.isfinite(gain) or gain <= 0:
        raise ValueError("RGB motion inspection needs two or three images and positive gain")
    arrays = [np.asarray(im.convert("RGB"), dtype=np.float32) for im in images]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("motion images must have the same dimensions")
    luminance = [a @ np.array([.2126, .7152, .0722], dtype=np.float32) for a in arrays]
    if len(luminance) == 2:
        luminance.append(luminance[1])  # red / cyan, not red / blue
    base = np.mean(luminance, axis=0)
    dev = np.stack([np.clip(128 + gain * (l - base), 0, 255) for l in luminance], axis=2)
    mag = np.max([np.abs(l - base) for l in luminance], axis=0)
    weight = np.clip(mag * gain / 110., 0, 1)[..., None]
    grey = np.repeat(base[..., None], 3, axis=2) * .55 + 40
    return Image.fromarray(np.uint8(np.clip(grey * (1 - weight) + dev * weight, 0, 255)))


def motion_image(service, args, workspace, directory, cancel):
    frames = args.get("frames")
    if not isinstance(frames, list) or len(frames) not in (2, 3):
        raise ValueError("motion_image.frames must contain two or three ordered output indices")
    # Defaults to the diagnostic engine: this is a temporal difference image,
    # where flat shading is a help rather than a limitation. The override
    # exists for the case where the change being located IS a shading change.
    settings = args.get("render_settings")
    result = service.worker(workspace, directory / "frames", {"op": "render", "frames": frames,
        "fit_aspect": unsized(settings),
        **({"render_settings": final_render_settings(
            over_defaults(settings, DIAGNOSTIC_RENDER_SETTINGS))} if settings else {}),
        **{k: args[k] for k in ("eye", "look_at", "up", "fov_deg", "camera_name") if args.get(k) is not None}}, cancel)
    paths = [directory / "frames" / p for p in result["paths"]]
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.copy())
    out = directory / "motion_rgb.png"
    rgb_temporal(images, args.get("gain", 4.0)).save(out)
    return {"ok": True, "path": str(out), "source_images": [str(p) for p in paths], "frames": frames,
            "encoding": "R=earliest, G=middle, B=latest" if len(frames) == 3 else "R=earlier, cyan=later",
            "interpretation": "Neutral pixels are unchanged. Color differences can also come from camera motion, lighting, or occlusion; compare the ordinary frames."}


def review(service, args, workspace, directory, cancel):
    from ..agent.oneoff import run_oneoff
    from ..agent.layers.review import review_backend, run_model, run_reasoning_effort
    plan = service.plan_of(workspace)
    service.build_of(workspace)
    design = workspace_context(workspace)
    design_checks = {"scene": design.scene_ir.get("checks", []),
                     "objects": {key: obj.get("checks", []) for key, obj in design.catalog.items()}}
    # Review is asked about materials, lighting, and composition among other
    # things, so its evidence cannot be solid-shaded. Validate here rather
    # than inside the worker: a typo should fail before a Blender launch.
    settings = final_render_settings(over_defaults(args.get("render_settings"),
                                                   REVIEW_RENDER_SETTINGS))
    fit = unsized(args.get("render_settings"))
    seeds = []
    if plan.get("states") is not None:
        # The current plan shape. Its states carry no frame and its
        # shots no camera override: where a state sits is bound in the blend,
        # and the camera comes from the timeline. So the states are drawn by
        # id, exactly as `storyboard()` draws them, and the worker resolves
        # each one. Reading `shot["states"]` here failed every call on this
        # shape.
        states = plan_states(plan)
        result = service.worker(workspace, directory / "seed_states", {
            "op": "render", "render_settings": settings, "fit_aspect": fit,
            "states": [{"id": s["id"], "frame": s.get("frame")} for s in states]}, cancel)
        located = result.get("state_frames") or {}
        seeds.append({"states": [{"id": s["id"], "purpose": s["purpose"],
                                  "frame": located.get(s["id"], s.get("frame"))} for s in states],
                      "images": [str(directory / "seed_states" / p) for p in result["paths"]]})
    else:
        for i, shot in enumerate(plan["shots"]):
            result = service.worker(workspace, directory / f"seed_{i}", {
                "op": "render", "frames": [s["frame"] for s in shot["states"]],
                "camera_name": shot["camera"], "render_settings": settings,
                "fit_aspect": fit}, cancel)
            seeds.append({"shot": shot, "images": [str(directory / f"seed_{i}" / p) for p in result["paths"]]})
    # Frames the author wants looked at, beyond the planned states. Rendered
    # through the native timeline rather than a shot camera, so they exercise
    # the actual cuts — `focus` is prose and cannot select a frame.
    extra = args.get("frames")
    if extra is not None:
        if not isinstance(extra, list) or not 1 <= len(extra) <= 32:
            raise ValueError("review frames must be 1 to 32 zero-based output indices")
        result = service.worker(workspace, directory / "seed_requested", {
            "op": "render", "frames": extra, "render_settings": settings,
            "fit_aspect": fit}, cancel)
        seeds.append({"shot": {"id": "__requested__", "intent": (
            "Frames the author asked you to look at specifically, rendered through the "
            "native timeline cameras."), "states": [{"frame": f, "purpose": "requested"} for f in extra]},
            "images": [str(directory / "seed_requested" / p) for p in result["paths"]]})
    # The reviewer's own renders inherit the same settings, so what it chooses
    # to look at is shaded like what it was handed.
    command = shlex.join([sys.executable, "-m", "dycraft.animation.review_render",
                         "--workspace", str(workspace), "--out-dir", str(directory / "sub_runs/review/extra"),
                         "--render-settings", json.dumps(settings)])
    prompt = (
        "Give a fresh-context second opinion based on selected still frames of this project. "
        "Your findings are advisory, not a review of the complete video. You have no encoded-film playback input. "
        "Read the seed image files, then render and read other useful frames/views with the command below "
        "when a specific question needs more evidence. Judge the explicit criteria and original task only "
        "to the extent the inspected views support: action states and consequences, contacts and support, "
        "functional geometry and visible defects. Do not infer a successful action from one attractive pose. "
        "For a transition, compare relevant earlier, intermediate, and later states; distinguish the observations "
        "from assumptions about unobserved intervals. Sparse stills do not establish smoothness, convincing "
        "weight transfer, natural speed, whole-film pacing, or complete action coverage. Mark those questions "
        "uncertain when the inspected evidence is insufficient. Do not claim to have watched the film. "
        "Report the frames/views you actually opened, separately from images merely supplied or rendered, "
        "and state the important uninspected intervals and limitations. Cite evidence for findings and repairs. "
        "Return JSON with inspected_views, limitations, results "
        "[{id, verdict: pass|fail|uncertain, evidence, repairs}], and overall observations.\n\n"
        f"Original task:\n{service.ctx.user_prompt}\n\nPlan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Additional registered design checks:\n{json.dumps(design_checks, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(seeds, indent=2)}\n\n"
        f"Extra views: {command} --frames 0 10 --camera Camera\n"
        "Or replace --camera with --eye x y z --look-at x y z. Frame indices are zero-based. "
        "Use --help for options. The renderer opens an immutable copy of this project. "
        "Your renders use the same engine and dimensions as the seed images; pass a different "
        "--render-settings JSON object only when the question needs it, and say so in your report.\n"
        f"Review focus: {args.get('focus') or 'the planned criteria as supported by inspected frames; identify unverified aspects'}"
    )
    (directory / "review_prompt.txt").write_text(prompt)
    if cancel.is_set():
        raise RuntimeError("job cancelled")
    done = threading.Event()
    def stop_reviewer():
        # Use the launcher's existing clean-stop protocol; the oneoff owns its
        # child process and output recovery. Repeat through startup if needed.
        while not done.wait(.25):
            if cancel.is_set():
                sentinel = directory / "sub_runs/review/.terminate"
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.touch()
    watcher = threading.Thread(target=stop_reviewer, daemon=True)
    watcher.start()
    try:
        reply = run_oneoff(prompt=prompt, backend=review_backend(service.ctx),
                           model=run_model(service.ctx) or None,
                           reasoning_effort=run_reasoning_effort(service.ctx) or None,
                           run_dir=directory, sub_run_name="review",
                           timeout_s=REVIEW_TIMEOUT_S)
    finally:
        done.set()
        watcher.join(timeout=1)
    if cancel.is_set():
        raise RuntimeError("job cancelled")
    reviewed = bool(reply.get("ok"))
    result = {"ok": reviewed, "advisory": True, "reviewed": reviewed,
              "text": reply.get("text") or "", "outcome": reply.get("outcome"),
              "render_settings": settings, "seed_views": seeds}
    if not reviewed:
        # Not `ok` when the reviewer gave no opinion, and the reply says why: an
        # agent that reads `ok` goes on as if it had the opinion it asked for.
        result.update(failure_class="harness", error=_review_failure(reply, directory),
                      returncode=reply.get("returncode"),
                      stderr_tail=reply.get("stderr_tail") or "")
    return result


def _review_failure(reply, directory):
    """Why a review produced no opinion, in a sentence the author can act on."""
    outcome = reply.get("outcome")
    if outcome == "timeout":
        why = f"it did not answer within {REVIEW_TIMEOUT_S:g} s"
    elif outcome == "spawn_error":
        why = "it could not be started"
    else:
        why = f"it exited with status {reply.get('returncode')} without answering"
    return (f"the reviewer gave no opinion: {why}. Nothing was reviewed; you may call "
            f"review() again. Its run is {directory / 'sub_runs' / 'review'}, and "
            "`stderr_tail` holds the end of its error output.")
