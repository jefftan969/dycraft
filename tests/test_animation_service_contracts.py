"""The animation service's job polling and the frame order of its renders."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace
import zlib

import pytest

from dycraft.animation.blender.runtime import REPO
from dycraft.animation import delivery
from dycraft.tests._animation_fixtures import service


def test_polling_survives_workspace_publication(tmp_path, monkeypatch):
    """A job status read while a publish moves the workspace under it still
    answers from the job record instead of raising."""
    svc = service(tmp_path)
    svc.jobs.jobs["published"] = ({"status": "completed", "result": {"ok": True}},
                                  threading.Event())

    def raced_progress(self, pattern):
        raise FileNotFoundError("workspace moved during traversal")
    monkeypatch.setattr(Path, "rglob", raced_progress)
    try:
        assert svc.jobs.status("published")["result"]["ok"]
    finally:
        svc.jobs.close()


@pytest.mark.parametrize("strategy", ["batch", "frames"])
@pytest.mark.parametrize("bad_encode", [None, "dimensions", "frame_count", "fps"])
def test_an_encoded_film_that_does_not_match_the_project_is_refused(tmp_path, monkeypatch,
                                                                    strategy, bad_encode):
    """The probe of the encoded film must match the requested size, the
    rendered frame count and the timeline's rate; a batch render's own video is
    probed as it is, and only a frames render is encoded here."""
    rendered, directory = tmp_path / "rendered", tmp_path / "job"
    (rendered / "frames").mkdir(parents=True)
    directory.mkdir()
    (rendered / "clip.mp4").write_bytes(b"film")
    encoded = []

    def process(argv, **kwargs):
        if argv[0] == "ffmpeg":
            assert strategy == "frames", "a batch render's own video is not re-encoded"
            encoded.append(argv[-1])
            Path(argv[-1]).write_bytes(b"film")
            return SimpleNamespace(returncode=0)
        assert argv[0] == "ffprobe"
        rate = "12/1" if bad_encode == "fps" else "24/1"
        return SimpleNamespace(stdout=json.dumps({"streams": [{
            "width": 1280 if bad_encode == "dimensions" else 640, "height": 480,
            "codec_name": "h264", "pix_fmt": "yuv420p",
            "nb_read_frames": "2" if bad_encode == "frame_count" else "3",
            "avg_frame_rate": rate, "r_frame_rate": rate}]}))
    monkeypatch.setattr(delivery, "run_cancellable", process)
    result = {"fps": 24, "n_frames": 3, "clip": "clip.mp4"}
    settings = {"width": 640, "height": 480, "strategy": strategy}
    if bad_encode:
        with pytest.raises(RuntimeError, match="FPS" if bad_encode == "fps" else "frame count"):
            delivery.encode_and_inspect(result, rendered, directory, settings, threading.Event())
    else:
        clip, probe = delivery.encode_and_inspect(result, rendered, directory, settings,
                                                  threading.Event())
        assert clip == (rendered if strategy == "batch" else directory) / "clip.mp4"
        assert probe["width"] == 640
    assert len(encoded) == (strategy == "frames")


def _png(path):
    """A 1x1 gray PNG, enough for the frames a preview copies."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"
                           + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(b"\x00\x80"))
                           + chunk(b"IEND", b""))


def _worker_module(monkeypatch, scene, rendered):
    """The real Blender worker with its Blender collaborators replaced; no GPU."""
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "bpy_extras", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    # The sandbox puts the worker's siblings (`render_config`, ...) on the path.
    monkeypatch.syspath_prepend(str(REPO / "animation/blender"))
    spec = importlib.util.spec_from_file_location("snapshot_worker", REPO / "animation/blender/worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    def render(**kwargs):
        rendered.append(scene.render.filepath)
        _png(scene.render.filepath)
    # Takes `diagnostic=`: a render asks for the cheap diagnostic profile.
    monkeypatch.setattr(worker, "render_settings", lambda diagnostic=False: scene)
    monkeypatch.setattr(worker, "timeline", lambda: {"n_frames": 12, "fps": 24})
    monkeypatch.setattr(worker, "diagnostic_render_settings", lambda scene: None)
    monkeypatch.setattr(worker, "FrameGeometryCache", lambda: SimpleNamespace(close=lambda: None))
    # No simulation in the scene, so every frame is jumped to.
    monkeypatch.setattr(worker, "requires_preroll", lambda scene: False)
    monkeypatch.setattr(worker, "bpy", SimpleNamespace(
        context=SimpleNamespace(scene=scene),
        data=SimpleNamespace(materials=[]),
        ops=SimpleNamespace(render=SimpleNamespace(render=render))))
    return worker


@pytest.mark.parametrize("frames", [[10, 0], [5, 5], [10, 0, 10]])
def test_preview_frames_come_back_in_the_order_asked(tmp_path, monkeypatch, frames):
    """Each requested frame index names its own PNG in the reply, repeats and
    order included, and each distinct frame is drawn once, jumped to in time
    order."""
    visited, rendered = [], []
    camera = SimpleNamespace(name="Camera",
                             matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                             data=SimpleNamespace(type="PERSP", lens=50, sensor_width=36.0,
                                                  sensor_height=24.0, sensor_fit="AUTO",
                                                  ortho_scale=6.0, shift_x=0.0, shift_y=0.0))
    scene = SimpleNamespace(frame_start=7, camera=camera, timeline_markers=[],
                            # A zero size leaves the aspect fit out of it.
                            render=SimpleNamespace(filepath="", engine="BLENDER_WORKBENCH",
                                                   use_persistent_data=False,
                                                   resolution_x=0, resolution_y=0),
                            frame_set=visited.append)
    worker = _worker_module(monkeypatch, scene, rendered)
    svc = service(tmp_path / "run")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "animation.blend").write_bytes(b"blend")
    directory = tmp_path / "job"
    directory.mkdir()

    def run_worker(workspace, output, request, cancel, **kwargs):
        monkeypatch.setenv("DYCRAFT_BLENDER_WORKSPACE_ROOT", str(workspace))
        monkeypatch.setenv("DYCRAFT_BLENDER_OUTPUT_ROOT", str(output))
        Path(output).mkdir(parents=True, exist_ok=True)
        return worker.render_frames(request)
    monkeypatch.setattr(svc, "worker", run_worker)
    monkeypatch.setattr(svc, "build_of", lambda workspace: {})
    try:
        reply = delivery.preview(svc, {"frames": frames, "sheet": False, "encode": False}, workspace, directory,
                                 threading.Event())
    finally:
        svc.jobs.close()
    assert reply["frame_indices"] == frames
    assert visited == [7 + i for i in sorted(set(frames))]
    assert len(rendered) == len(set(frames))


def test_the_budget_carries_the_seconds_until_the_run_is_stopped(tmp_path, monkeypatch):
    """`stop_in_s` counts down to the launcher's hard limit (`DYCRAFT_DEADLINE`,
    epoch seconds), never goes below zero, and is absent without a finite limit
    (unset, `inf`, `-inf`, `nan`, or unreadable) and when the run has no advisory
    budget, since it rides in the budget block."""
    import time
    svc = service(tmp_path)
    monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", "1")
    try:
        monkeypatch.delenv("DYCRAFT_DEADLINE", raising=False)
        assert "stop_in_s" not in svc.wall_budget()
        for value in ("inf", "-inf", "nan", "soon"):
            monkeypatch.setenv("DYCRAFT_DEADLINE", value)
            assert "stop_in_s" not in svc.wall_budget(), value
        monkeypatch.setenv("DYCRAFT_DEADLINE", repr(time.time() + 3600.0))
        assert 3590 <= svc.wall_budget()["stop_in_s"] <= 3600
        monkeypatch.setenv("DYCRAFT_DEADLINE", repr(time.time() - 5.0))
        assert svc.wall_budget()["stop_in_s"] == 0
        monkeypatch.setenv("DYCRAFT_DEADLINE", repr(time.time() + 3600.0))
        monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", "0")
        assert svc.wall_budget() is None
    finally:
        svc.jobs.close()


@pytest.mark.parametrize("value", ["0", "-5", "", "soon", "nan", "inf"])
def test_a_run_without_an_advisory_budget_has_no_budget_block(tmp_path, monkeypatch, value):
    """Zero or less, unset, unreadable, or not finite means no budget: no
    `budget.run` in render replies, and the children it dispatches get no
    budget either."""
    from dycraft.animation.dispatch import child_hours
    svc = service(tmp_path)
    monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", value)
    try:
        assert svc.wall_budget() is None
        assert "run" not in delivery.render_budget(svc, 40.0, 10)
        assert child_hours(svc.wall_budget()) == 0
    finally:
        svc.jobs.close()


def test_a_run_from_a_prior_project_numbers_its_boards_after_the_ones_it_inherits(tmp_path):
    """A refinement or contributor run starts from a project whose storyboards
    keep their submission numbers. Its own submissions count on from them;
    numbered from 1, every board it draws would lose the merge to an inherited
    one and be recorded as superseded."""
    prior = tmp_path / "prior_animation"
    (prior / "design_state").mkdir(parents=True)
    (prior / "design_state/build.json").write_text(json.dumps({"source_files": []}))
    (prior / "storyboards.json").write_text(json.dumps({"opening": {"submitted": 50}}))
    svc = service(tmp_path)
    try:
        assert (svc.work / "storyboards.json").is_file()
        assert next(svc._seq) == 51
    finally:
        svc.jobs.close()
