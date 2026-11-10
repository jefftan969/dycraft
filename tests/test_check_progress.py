"""A job's progress counts steps, and its rates are its current phase's own."""
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    from dycraft.animation.blender.runtime import REPO
    for name in ("bpy", "bpy_extras", "_blender_child"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    spec = importlib.util.spec_from_file_location("progress_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_partial_check_projects_from_its_position_not_its_frame_index(worker, tmp_path, monkeypatch):
    """Frames 500-600 are 101 steps. Reading the absolute frame as the count put
    the job 501 steps into 101 and projected nothing left from its first frame."""
    clock = iter(range(1000, 2000))
    monkeypatch.setattr(worker.time, "time", lambda: float(next(clock)))
    stamps = {}
    for done, frame in enumerate(range(500, 511), start=1):
        worker.write_progress(tmp_path, {"phase": "evaluate", "frame": frame, "done": done,
                                         "total": 101}, 999.0, stamps)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["frame"] == 510 and progress["done"] == 11
    assert progress["seconds_per_step_recent"] == pytest.approx(1.0)
    assert progress["projected_remaining_s"] == pytest.approx(90.0)


def test_a_new_phase_is_timed_from_when_the_last_one_reported(worker, tmp_path, monkeypatch):
    """A new phase's rate is measured from its own start, not the job's."""
    now = [0.0]
    monkeypatch.setattr(worker.time, "time", lambda: now[0])
    stamps = {}
    for done in range(1, 101):                      # 100 frames at 1 s each
        now[0] += 1.0
        worker.write_progress(tmp_path, {"phase": "evaluate_verify", "done": done,
                                         "total": 100}, 0.0, stamps)
    for done in range(0, 3):                        # diagnostics at 10 s a span
        now[0] += 10.0 if done else 0.0
        for _ in (0, 1):                            # reported before and after
            worker.write_progress(tmp_path, {"phase": "penetration_diagnostics", "frame": 40,
                                             "intervals_done": done, "interval_count": 5},
                                  0.0, stamps)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["seconds_per_step_mean"] == pytest.approx(10.0)
    assert progress["seconds_per_step_recent"] == pytest.approx(10.0)
    assert progress["projected_remaining_s"] == pytest.approx(30.0)


def test_a_render_reports_its_rate_without_the_first_frames_scene_sync(worker, tmp_path, monkeypatch):
    """The first frame of a render carries the scene's synchronization, often
    several times a later frame's time, so a mean that includes it overstates
    the price of a whole film from a short pricing render."""
    now = [100.0]
    monkeypatch.setattr(worker.time, "time", lambda: now[0])
    stamps = {}
    worker.write_progress(tmp_path, {"phase": "render", "done": 0, "total": 5}, 100.0, stamps)
    for done, step in ((1, 6.5), (2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)):
        now[0] += step
        worker.write_progress(tmp_path, {"phase": "render", "done": done, "total": 5},
                              100.0, stamps)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["seconds_per_step_mean"] == pytest.approx(14.5 / 5)
    assert progress["seconds_per_step_after_first"] == pytest.approx(2.0)
    assert progress["first_step_s"] == pytest.approx(6.5)
