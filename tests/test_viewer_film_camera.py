"""`viewer/animation.py: film_camera`: the view a delivered film opens on, read from its own `camera.json`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.viewer import animation as A

MATRIX = [[-0.73, 0.14, -0.67, -1.25], [-0.68, -0.15, 0.72, 1.05], [0.0, 0.98, 0.20, 1.75], [0.0, 0.0, 0.0, 1.0]]


def _view(frame, camera="CAM_01", **over):
    view = {"frame": frame, "camera": camera, "camera_matrix": MATRIX, "camera_type": "PERSP", "lens_mm": 24.3,
            "sensor_mm": 36.0, "sensor_height_mm": 24.0, "sensor_fit": "HORIZONTAL", "ortho_scale": 6.0,
            "shift": [0.0, 0.0]}
    return {**view, **over}


def _film(run: Path, views, *, frame_start=0, width=1920, height=1080) -> dict:
    film = run / "bundle" / "film"
    film.mkdir(parents=True)
    (film / "clip.mp4").write_bytes(b"mp4")
    (film / "camera.json").write_text(json.dumps({
        "views": views, "frame_start": frame_start, "n_frames": len(views), "fps": 24.0,
        "render_settings": {"width": width, "height": height, "resolution_percentage": 50}}))
    return {"film": "bundle/film/clip.mp4"}


def test_a_delivered_film_opens_on_its_first_frame_s_camera(tmp_path):
    ir = _film(tmp_path, [_view(10, "CAM_late"), _view(0, "CAM_open")])
    got = A.film_camera(tmp_path, ir)
    assert got["camera"] == "CAM_open" and got["frame"] == 0
    assert got["matrix"] == MATRIX
    assert (got["lens_mm"], got["sensor_mm"], got["sensor_height_mm"], got["sensor_fit"]) == (24.3, 36.0, 24.0,
                                                                                              "HORIZONTAL")
    assert got["aspect"] == pytest.approx(1920 / 1080)


def test_a_film_that_starts_later_opens_on_its_own_first_frame(tmp_path):
    ir = _film(tmp_path, [_view(100, "CAM_a"), _view(101, "CAM_b")], frame_start=101)
    assert A.film_camera(tmp_path, ir)["camera"] == "CAM_b"


@pytest.mark.parametrize("change", [
    lambda run, ir: (run / "bundle" / "film" / "camera.json").unlink(),
    lambda run, ir: (run / "bundle" / "film" / "camera.json").write_text("not json"),
    lambda run, ir: ir.pop("film"),
], ids=["no-record", "unreadable", "no-film"])
def test_without_a_readable_record_the_viewer_keeps_its_own_framing(tmp_path, change):
    ir = _film(tmp_path, [_view(0)])
    change(tmp_path, ir)
    assert A.film_camera(tmp_path, ir) is None


@pytest.mark.parametrize("bad", [{"camera_type": "ORTHO"}, {"camera_matrix": [[1, 0], [0, 1]]}, {"lens_mm": 0}],
                         ids=["orthographic", "matrix", "lens"])
def test_a_camera_the_viewer_cannot_reproduce_is_left_out(tmp_path, bad):
    ir = _film(tmp_path, [_view(0, **bad)])
    assert A.film_camera(tmp_path, ir) is None


def test_the_film_view_can_follow_every_frame_s_camera(tmp_path):
    """`film_cameras`: one entry per frame from the record, the camera a cut changes to, each frame's own matrix and
    lens, and -1 where a frame's camera cannot be reproduced, so the film view holds its last shot there."""
    moved = [row[:] for row in MATRIX]
    moved[0][3] += 0.5
    views = [_view(0, "CAM_a"), _view(1, "CAM_a", camera_matrix=moved, lens_mm=30.0), _view(2, "CAM_b"),
             _view(3, "CAM_b"), _view(4, "CAM_c", camera_type="ORTHO")]
    ir = _film(tmp_path, views)
    track = A.film_cameras(tmp_path, ir)
    assert track["n_frames"] == 5 and track["frame_start"] == 0
    assert [c["name"] for c in track["cameras"]] == ["CAM_a", "CAM_b"]
    assert track["camera"] == [0, 0, 1, 1, -1]
    assert track["matrix"][:12] == [x for row in MATRIX[:3] for x in row]
    assert track["matrix"][12:24] == [x for row in moved[:3] for x in row]
    assert track["lens_mm"][:2] == [24.3, 30.0]
    assert track["aspect"] == pytest.approx(1920 / 1080)
    ortho = tmp_path / "ortho"
    assert A.film_cameras(ortho, _film(ortho, [_view(0, camera_type="ORTHO")])) is None


def test_a_record_outside_the_run_is_not_read(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _film(tmp_path / "elsewhere", [_view(0)])
    assert A.film_camera(run, {"film": "../elsewhere/bundle/film/clip.mp4"}) is None


_MAKE_BLEND = r"""
import bpy, json, sys
blend, expect = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 3, 40
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
def camera(name, location):
    obj = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (1.1, 0.0, 0.7)
    return obj
scene.camera = camera("CAM_default", (5, -5, 2))
moving = camera("CAM_marked", (1, -6, 1.5))
moving.data.lens = 35
for frame, x in ((0, -2.0), (10, 3.0)):
    moving.location.x = x
    moving.keyframe_insert("location", index=0, frame=frame)
scene.timeline_markers.new("cut", frame=1).camera = moving
if "constrained" in blend:
    target = bpy.data.objects.new("target", None)
    scene.collection.objects.link(target)
    moving.constraints.new("TRACK_TO").target = target
# A second shot from frame 20: the per-frame read must cut to it.
scene.timeline_markers.new("cut2", frame=20).camera = scene.camera
bpy.ops.wm.save_as_mainfile(filepath=blend)
scene.frame_set(scene.frame_start)
first = {"camera": moving.name, "matrix": [list(r) for r in moving.matrix_world]}
frames = {}
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    bound = [m for m in scene.timeline_markers if m.camera and m.frame <= f]
    cam = max(bound, key=lambda m: m.frame).camera
    frames[f] = {"camera": cam.name, "matrix": [list(r) for r in cam.matrix_world]}
json.dump({**first, "frames": frames}, open(expect, "w"))
"""


@pytest.mark.parametrize("constrained", [False, True], ids=["animated", "constrained"])
def test_without_a_record_the_camera_is_read_from_the_blend(tmp_path, monkeypatch, constrained):
    """A film installed from the published set has no `camera.json`. The camera the timeline markers bind at the first
    frame, here animated, is read from its `.blend` without evaluating the scene, and matches Blender's own pose; a
    camera placed by a constraint is left out. The read is cached, so Blender runs once."""
    import subprocess
    from dycraft.render import _blender_env as E
    interp = E.blender_python()
    if not interp:
        pytest.skip("no Blender interpreter")
    monkeypatch.setenv("DYCRAFT_VIEWER_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    run = tmp_path / "run"
    (run / "bundle" / "film").mkdir(parents=True)
    (run / "bundle" / "film" / "clip.mp4").write_bytes(b"mp4")
    blend = run / "bundle" / ("constrained.blend" if constrained else "animation.blend")
    expect = tmp_path / "expect.json"
    subprocess.run([interp, "-c", _MAKE_BLEND, str(blend), str(expect)], check=True, capture_output=True,
                   env=E.blender_child_env())
    ir = {"film": "bundle/film/clip.mp4", "blend": f"bundle/{blend.name}"}
    got = A.film_camera(run, ir)
    if constrained:
        assert got is None
        return
    want = json.loads(expect.read_text())
    assert got["camera"] == want["camera"] == "CAM_marked"
    assert [x for row in got["matrix"] for x in row] == pytest.approx([x for row in want["matrix"] for x in row],
                                                                      abs=1e-5)
    assert got["lens_mm"] == pytest.approx(35.0) and got["aspect"] == pytest.approx(1280 / 720)
    # Every frame, for the film view to follow: the camera each frame's markers bind, posed as Blender poses it.
    track = A.film_cameras(run, ir)
    assert track["frame_start"] == 3 and track["n_frames"] == 38
    for i, (frame, view) in enumerate(sorted(want["frames"].items(), key=lambda kv: int(kv[0]))):
        assert track["cameras"][track["camera"][i]]["name"] == view["camera"], f"frame {frame}"
        assert track["matrix"][12 * i:12 * i + 12] == pytest.approx(
            [x for row in view["matrix"][:3] for x in row], abs=1e-5), f"frame {frame}"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("read the .blend again"))
    assert A.film_camera(run, ir) == got
    assert A.film_cameras(run, ir) == track


def test_a_read_that_times_out_is_not_tried_again_by_this_server(tmp_path, monkeypatch):
    """A film whose `.blend` read always times out costs the timeout once per server, not once per `/animation`
    request. It is not cached on disk, since a busy machine can time out a read that would succeed, and a changed
    file is read again."""
    import os
    import subprocess
    from dycraft.render import _blender_env as E
    monkeypatch.setenv("DYCRAFT_VIEWER_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(E, "blender_python", lambda: "blender-python")
    calls = []

    def times_out(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", times_out)
    run = tmp_path / "run"
    blend = run / "bundle" / "animation.blend"
    blend.parent.mkdir(parents=True)
    blend.write_bytes(b"blend")
    ir = {"film": "bundle/film/clip.mp4"}
    assert A.film_camera(run, ir) is None
    assert A.film_camera(run, ir) is None
    assert len(calls) == 1, "the timed-out read was tried again"
    assert not (tmp_path / "cache" / "film_camera").exists(), "a timeout was cached on disk"
    st = blend.stat()
    os.utime(blend, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert A.film_camera(run, ir) is None
    assert len(calls) == 2, "a changed file was not read again"


def _storyboard(run: Path, entries: dict, *, images=()) -> None:
    from PIL import Image
    bundle = run / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "storyboards.json").write_text(json.dumps(entries))
    for rel in images:
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (200, 120, 40)).save(bundle / rel)


def test_the_timeline_marks_the_storyboard_s_keyframes_in_frame_order(tmp_path, monkeypatch):
    """`storyboard_keys`: the film's keyframes in frame order, a keyframe whose image is not in the run left out; a
    thumbnail per keyframe, small, made once and kept."""
    monkeypatch.setenv("DYCRAFT_VIEWER_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    run = tmp_path / "run"
    _storyboard(run, {"late": {"frame": 90, "path": "views/b.png", "camera": "CAM_b"},
                      "open": {"frame": 0, "path": "views/a.png", "camera": "CAM_a"},
                      "lost": {"frame": 40, "path": "views/missing.png"},
                      "outside": {"frame": 50, "path": "../../escape.png"}},
                images=("views/a.png", "views/b.png"))
    (tmp_path / "escape.png").write_bytes(b"not for the viewer")
    keys = A.storyboard_keys(run)
    assert keys == [{"name": "open", "frame": 0, "camera": "CAM_a", "index": 0},
                    {"name": "late", "frame": 90, "camera": "CAM_b", "index": 1}]
    from PIL import Image
    import io
    thumb = A.storyboard_thumb(run, 1)
    with Image.open(io.BytesIO(thumb)) as im:
        assert im.format == "JPEG" and im.size == (320, 180)
    assert A.storyboard_thumb(run, 2) is None and A.storyboard_thumb(run, -1) is None
    assert len(list((tmp_path / "cache" / "storyboard").glob("*.jpg"))) == 1
    assert A.storyboard_keys(tmp_path / "no-storyboard") == []
    # The film's first frame comes from its own manifest, so its keyframes land on the playhead's frames whether or
    # not its cameras can be followed (a film whose every camera a constraint places has no camera track).
    (run / "ir.json").write_text(json.dumps({"frame_start": 1}))
    assert A.storyboard(run) == {"frame_start": 1, "keys": keys}
    assert A.storyboard(tmp_path / "no-storyboard") == {"frame_start": 0, "keys": []}
