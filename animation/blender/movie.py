"""One native animation render with per-frame validation and direct encoding.

Blender swallows Python handler exceptions. Signal the supervising runtime as
well as remembering the exception, so a failed replay check stops the worker
and can never publish a partially checked movie.
"""
import json
from pathlib import Path
import time
import traceback


def render_movie(bpy, info, output, prepare_frame, record_view=None, progress=None, *,
                 run=None, stills=False):
    """Render one run of output frames in a single `render(animation=True)` call.

    `run=(first, last, step)` is a contiguous or evenly strided run of output
    frame indices, the whole film by default; it becomes the scene's own
    range and `frame_step` for the call, and the scene's range is restored
    after. `stills=True` writes each frame as `frames/<index>.png`, which is
    what a draft or a delivery drawn as frames needs; otherwise the run is
    encoded to `clip.mp4`.

    One call, because a render call has a fixed setup -- Workbench measured
    130-240 ms of it per frame on this layer's scenes, and one call drew the
    chair at 62 ms a frame against 225. The handlers are what every
    frame still gets individually: `prepare_frame` (camera pinning, the
    replay check) before it, `record_view` and progress after.
    """
    scene = bpy.context.scene
    if scene.render.use_multiview:
        raise ValueError("Batch film rendering requires a single view; use strategy='frames' for multiview scenes")
    if info["n_frames"] < 1:
        raise ValueError("The animation must contain at least one frame")
    first, last, step = run if run is not None else (0, info["n_frames"] - 1, 1)
    expected = list(range(first, last + 1, step))
    if not expected or last >= info["n_frames"]:
        raise ValueError(f"Batch render run {run} is not inside the timeline")
    saved = (scene.frame_start, scene.frame_end, scene.frame_step,
             getattr(scene.render, "filepath", ""))
    frames_dir = Path(output) / "frames"
    if stills:
        scene.render.image_settings.file_format = "PNG"
        frames_dir.mkdir(exist_ok=True)
        # Blender names a frame by its native number; `after` renames it to
        # the output index every other path writes.
        scene.render.filepath = str(frames_dir / "native_######")
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"  # CRF 23, like libx264's default.
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.audio_codec = "NONE"  # The existing final-render path is video-only.
        scene.render.ffmpeg.use_autosplit = False
        scene.render.filepath = str(Path(output) / "clip.mp4")
    scene.render.use_file_extension = True
    scene.frame_start = info["frame_start"] + first
    scene.frame_end = info["frame_start"] + last
    scene.frame_step = step
    prepared, written, errors = [], [], []

    def guard(callback):
        def handler(current, *unused):
            if errors:
                return
            try:
                callback(current)
            except Exception as error:
                errors.append(error)
                pending = Path(output) / "render_error.json.tmp"
                pending.write_text(json.dumps({"error": str(error), "traceback": traceback.format_exc()}))
                pending.replace(Path(output) / "render_error.json")
        return handler

    def before(current):
        index = current.frame_current - info["frame_start"]
        if len(prepared) >= len(expected) or index != expected[len(prepared)]:
            raise ValueError(f"Unexpected batch render frame {index}; expected "
                             f"{expected[len(prepared)] if len(prepared) < len(expected) else 'none'}")
        prepare_frame(index)
        prepared.append(index)

    def after(current):
        index = current.frame_current - info["frame_start"]
        if len(written) >= len(prepared) or index != expected[len(written)]:
            raise ValueError(f"Unexpected encoded frame {index}; expected {expected[len(written)]}")
        if stills:
            Path(scene.render.frame_path(frame=current.frame_current)).replace(
                frames_dir / f"{index:06d}.png")
        written.append(index)
        # After the frame is written, which is the only moment the active
        # camera is the one that drew it: the render operator switches cameras
        # from the timeline markers itself, after `render_pre` has run.
        if record_view is not None:
            record_view(index)
        payload = {"phase": "render" if stills else "movie", "frame": index,
                   "done": len(written), "total": len(expected)}
        if progress is not None:
            # The shared writer: a rate and a projected finish, written
            # atomically: a batch film render is the longest single call on the
            # surface.
            progress(payload)
        else:
            (Path(output) / "progress.json").write_text(json.dumps(payload))

    pre, post = guard(before), guard(after)
    bpy.app.handlers.render_pre.append(pre)
    bpy.app.handlers.render_write.append(post)
    started = time.monotonic()
    try:
        status = bpy.ops.render.render(animation=True)
        if errors:
            raise RuntimeError(f"Batch render validation failed: {errors[0]}") from errors[0]
        if status != {"FINISHED"} or len(written) != len(expected):
            raise ValueError(f"Batch render incomplete: wrote {len(written)} of {len(expected)} frames")
        elapsed = time.monotonic() - started
        if stills:
            return {"paths": {i: f"frames/{i:06d}.png" for i in written},
                    "render_seconds": elapsed}
        if not (Path(output) / "clip.mp4").is_file():
            raise ValueError("Blender did not write the expected clip.mp4")
        return {"clip": "clip.mp4", "render_encode_seconds": elapsed,
                "rendered_frames_per_second": len(written) / elapsed,
                "encoding": {"codec": "H264", "crf": 23, "preset": "GOOD"}}
    finally:
        bpy.app.handlers.render_pre.remove(pre)
        bpy.app.handlers.render_write.remove(post)
        scene.frame_start, scene.frame_end, scene.frame_step, scene.render.filepath = saved
