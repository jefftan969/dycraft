"""Batch encoding retains frame coverage, replay checks, and rendered appearance."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.animation.blender.movie import render_movie


@pytest.mark.parametrize("failure", [None, "replay", "missing_frame"])
def test_batch_handler_errors_cannot_be_swallowed_and_handlers_are_removed(tmp_path, failure):
    scene = SimpleNamespace(frame_start=7, frame_end=9, frame_step=3,
        render=SimpleNamespace(use_multiview=False, ffmpeg=SimpleNamespace(), image_settings=SimpleNamespace()))
    existing = lambda *args: None
    handlers = SimpleNamespace(render_pre=[existing], render_write=[])
    def native_render(**args):
        assert args == {"animation": True}
        assert scene.frame_step == 1
        for frame in range(7, 10):
            scene.frame_current = frame
            for fn in handlers.render_pre:
                fn(scene)
            if not (failure == "missing_frame" and frame == 9):
                for fn in handlers.render_write:
                    fn(scene)
        (tmp_path / "clip.mp4").write_bytes(b"movie")
        return {"FINISHED"}
    bpy = SimpleNamespace(context=SimpleNamespace(scene=scene), app=SimpleNamespace(handlers=handlers),
                          ops=SimpleNamespace(render=SimpleNamespace(render=native_render)))
    observed = []
    def prepare(index):
        if failure == "replay" and index == 1:
            raise ValueError("changed geometry")
        observed.append(index)
    if failure:
        with pytest.raises((ValueError, RuntimeError), match="changed geometry|incomplete"):
            render_movie(bpy, {"n_frames": 3, "frame_start": 7}, tmp_path, prepare)
        if failure == "replay":
            assert json.loads((tmp_path / "render_error.json").read_text())["error"] == "changed geometry"
    else:
        result = render_movie(bpy, {"n_frames": 3, "frame_start": 7}, tmp_path, prepare)
        assert observed == [0, 1, 2]
        assert result["clip"] == "clip.mp4"
    assert handlers.render_pre == [existing]
    assert handlers.render_write == []


@pytest.mark.expensive
def test_a_batch_movie_matches_frame_renders_and_draws_a_bad_replay(tmp_path):
    """The batch movie (the default Workbench delivery path) draws what per-frame
    renders draw: same settings, views and frame count, pixels within H.264
    tolerance, the compositor and a moving camera kept. A frame that does not
    replay its checked evaluation is drawn and named, not refused."""
    import shutil
    import subprocess
    from fractions import Fraction
    from PIL import Image
    from dycraft.animation.blender.bundle import file_hash
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_python
    from dycraft.tests._scene_fixtures import _stack_scene
    if blender_python() is None:
        pytest.skip("Blender interpreter not configured")
    scene, catalog = _stack_scene()
    (tmp_path / "catalog").mkdir()
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog), inputs)
    readonly = [inputs, tmp_path / "catalog"]
    def worker(label, request):
        return run_worker(work, tmp_path / label, request, readonly=readonly)
    worker("init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")})
    worker("edit", {"op": "edit", "code": '''
s = bpy.context.scene
s.frame_start = 7
s.frame_end = 10
s.frame_step = 3
s.render.fps = 30
s.render.fps_base = 1.001
s.display.render_aa = '8'
s.display.shading.show_shadows = True
s.display.shading.show_cavity = True
s.use_nodes = True
nodes = s.node_tree.nodes
nodes.clear()
layers = nodes.new('CompositorNodeRLayers')
mix = nodes.new('CompositorNodeMixRGB')
mix.blend_type = 'MULTIPLY'
mix.inputs[0].default_value = .35
mix.inputs[2].default_value = (.6, 1, .6, 1)
out = nodes.new('CompositorNodeComposite')
s.node_tree.links.new(layers.outputs['Image'], mix.inputs[1])
s.node_tree.links.new(mix.outputs[0], out.inputs[0])
root = bpy.data.objects['link/box__body']
root.keyframe_insert('location', frame=7)
root.location.z += .2
root.keyframe_insert('location', frame=10)
from mathutils import Vector
camera = s.camera
for frame, eye in ((7, (3, -4, 3)), (10, (4, -4, 3))):
    camera.location = eye
    camera.rotation_euler = (Vector((0, 0, .8)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.keyframe_insert('location', frame=frame)
    camera.keyframe_insert('rotation_euler', frame=frame)
'''})
    info = worker("evaluation", {"op": "evaluate"})
    shutil.copytree(tmp_path / "evaluation/evaluation", work / "evaluation")
    before = file_hash(work / "animation.blend")
    request = {"op": "render", "verify_replay": True,
               "render_settings": {"engine": "WORKBENCH", "width": 640, "height": 360, "threads": 2}}
    frames = worker("frames", request)
    batch = worker("batch", {**request, "movie": True})
    assert file_hash(work / "animation.blend") == before
    assert batch["render_settings"] == frames["render_settings"]
    # The saved '8'/shadows staging above is overridden by the Workbench draw
    # profile; cavity, which it does not touch, comes through as saved.
    assert batch["render_settings"]["antialiasing"] == "FXAA"
    assert batch["render_settings"]["workbench_shading"]["show_shadows"] is False
    assert batch["render_settings"]["workbench_shading"]["show_cavity"] is True
    assert [view["frame"] for view in batch["views"]] == [0, 1, 2, 3]
    assert batch["views"] == frames["views"]
    assert not (tmp_path / "batch/frames").exists()
    clip = tmp_path / "batch/clip.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(clip)],
                           check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["nb_read_frames"] == "4"
    assert float(Fraction(stream["avg_frame_rate"])) == pytest.approx(info["fps"], abs=.001)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    video = np.frombuffer(raw, dtype=np.uint8).reshape(4, 360, 640, 3)
    # Lossy H.264 has small differences; a lost compositor or wrong camera
    # produces a much larger error. Every frame is compared to its PNG render.
    for index, relative in enumerate(frames["paths"]):
        reference = np.asarray(Image.open(tmp_path / "frames" / relative).convert("RGB"))
        assert np.abs(video[index].astype(float) - reference).mean() < 3.0
    # Damage the accepted evaluation's frame 1 itself. The replay check
    # re-signs both sides from their content rather than trusting a stored
    # signature, so the damage has to be to what was recorded: one placement,
    # moved.
    from dycraft.animation.blender.evaluation import read_record, write_record
    record = read_record(work / "evaluation", 1)
    moved = sorted(record["transforms"])[0]
    record["transforms"][moved][0][3] += .5
    write_record(work / "evaluation", record)
    # Drawn and named, not refused: drawing never waits on a verdict.
    bad = worker("bad_replay", {**request, "movie": True})
    assert (tmp_path / "bad_replay/clip.mp4").is_file()
    assert not (tmp_path / "bad_replay/render_error.json").is_file()
    mismatch, = bad["replay_mismatch"]
    assert mismatch["frame"] == 1 and moved in mismatch["changed"]["transforms"]


@pytest.mark.expensive
def test_a_run_of_stills_is_drawn_in_one_call_and_matches_frame_by_frame(tmp_path):
    """An evenly spaced run of eight or more frames is one render call, which
    pays the per-call setup once; the frames it writes are the frames a call
    per frame writes. A scattered evaluation jumps to its frames and signs them
    as walking the whole film does."""
    from PIL import Image
    from dycraft.animation.blender.evaluation import read_record
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_python
    from dycraft.tests._scene_fixtures import _stack_scene
    if blender_python() is None:
        pytest.skip("Blender interpreter not configured")
    scene, catalog = _stack_scene()
    (tmp_path / "catalog").mkdir()
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog), inputs)
    readonly = [inputs, tmp_path / "catalog"]

    def worker(label, request):
        return run_worker(work, tmp_path / label, request, readonly=readonly)
    worker("init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")})
    worker("edit", {"op": "edit", "code": '''
s = bpy.context.scene
s.frame_start = 3
s.frame_end = 26
root = bpy.data.objects['link/box__body']
root.keyframe_insert('location', frame=3)
root.location.z += .3
root.location.x += .2
root.keyframe_insert('location', frame=26)
'''})
    settings = {"engine": "WORKBENCH", "width": 160, "height": 96, "threads": 2}
    strided = list(range(2, 23, 2))                     # 11 frames, step 2: one run
    batch = worker("batch", {"op": "render", "frames": strided + [23], "render_settings": settings})
    single = worker("single", {"op": "render", "frames": [2, 8, 23], "render_settings": settings})
    assert batch["paths"][:len(strided)] == [f"frames/{i:06d}.png" for i in strided]
    assert [view["frame"] for view in batch["views"]] == strided + [23]
    for frame in (2, 8, 23):
        drawn = np.asarray(Image.open(tmp_path / "batch" / f"frames/{frame:06d}.png").convert("RGB"), float)
        alone = np.asarray(Image.open(tmp_path / "single" / f"frames/{frame:06d}.png").convert("RGB"), float)
        assert np.abs(drawn - alone).mean() < .5, frame
        assert (tmp_path / "batch" / f"views/{frame:06d}.json").is_file()
    assert not list((tmp_path / "batch/frames").glob("native_*")), "frames are named by output index"

    walked = worker("walked", {"op": "evaluate"})
    jumped = worker("jumped", {"op": "evaluate", "frames": [5, 17, 22]})
    assert jumped["evaluation_start_frame"] == 5
    for frame in (5, 17, 22):
        assert (read_record(tmp_path / "jumped/evaluation", frame)["signature"]
                == read_record(tmp_path / "walked/evaluation", frame)["signature"]), frame
    assert walked["frames"] == list(range(24))
