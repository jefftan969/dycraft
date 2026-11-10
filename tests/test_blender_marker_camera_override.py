"""An explicitly requested render camera has to survive `render.render()`.

`scene.camera` assigned before a render does not stick: Blender's render
pipeline re-applies the camera bound to the timeline marker covering the
current frame. Without the override, a diagnostic render comes back in the
film's camera whenever the requested frame lies outside the requested camera's
own shot -- and `views`, recorded off `scene.camera` beforehand, reports the
camera that was *asked* for, so the two never visibly disagree. Anything
projected through the requested camera (a tracker's seed points, a trajectory
overlay) then lands on the wrong pixels.

The fakes below reproduce that switch: `frame_set` and `render.render` both
re-apply the marker camera, exactly as Blender does.
"""
import importlib.util
from types import SimpleNamespace

import pytest

from dycraft.animation.blender.runtime import REPO


def _camera(name, lens):
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # The full optics `record_view` persists on this branch: a field of view is
    # not recoverable from focal length alone, and `storyboard()` resolves which
    # objects a panel frames from exactly these numbers.
    return SimpleNamespace(name=name, type="CAMERA", matrix_world=matrix,
                           data=SimpleNamespace(type="PERSP", lens=lens,
                                                sensor_width=36.0,
                                                sensor_height=24.0,
                                                sensor_fit="AUTO",
                                                ortho_scale=6.0,
                                                shift_x=0.0, shift_y=0.0))


class _Scene:
    """Just enough bpy scene, with the marker camera switch wired in."""

    def __init__(self, markers, frame_start=1):
        self.frame_start = frame_start
        self.frame_current = frame_start
        self.timeline_markers = markers
        self.render = SimpleNamespace(filepath="", engine="FAKE",
                                      use_persistent_data=False)
        self.camera = None
        self.rendered = []
        self._apply_marker_camera()

    def _apply_marker_camera(self):
        """`BKE_scene_camera_switch_update`: last bound marker at or before now."""
        bound = [m for m in self.timeline_markers
                 if m.camera is not None and m.frame <= self.frame_current]
        if bound:
            self.camera = max(bound, key=lambda m: m.frame).camera

    def frame_set(self, frame):
        self.frame_current = frame
        self._apply_marker_camera()

    def render_still(self, **kwargs):
        # As in Blender: the render pipeline re-applies the switch, discarding
        # whatever the caller assigned to `scene.camera` moments earlier.
        self._apply_marker_camera()
        self.rendered.append((self.frame_current, self.camera.name))


def _worker(monkeypatch, scene, out_dir, objects, render_still=None):
    import sys
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils",
                        SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "marker_camera_worker", REPO / "animation/blender/worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(worker, "_output", lambda: out_dir)
    # `**_`: this branch's `render_settings` takes `config=` and a keyword-only
    # `diagnostic=`, and the double's job is to hand back the scene rather than
    # to pin a signature -- `test_object_prompt_contract` pins signatures.
    monkeypatch.setattr(worker, "render_settings", lambda *_a, **_k: scene)
    monkeypatch.setattr(worker, "apply_display_colors", lambda engine=None: None)
    monkeypatch.setattr(worker, "render_config", lambda: {"engine": "FAKE"})
    # Two more of this branch's reads. `diagnostic_render_settings` imports
    # `render_config` flat, which only resolves inside the worker's sandbox, and
    # the Cycles persistent-data decision reads `scene.render.engine`.
    monkeypatch.setattr(worker, "diagnostic_render_settings", lambda scene: None)
    monkeypatch.setattr(worker, "timeline",
                        lambda: {"n_frames": 336, "fps": 24, "frame_start": scene.frame_start})
    monkeypatch.setattr(worker, "FrameGeometryCache",
                        lambda: SimpleNamespace(close=lambda: None))
    # The frame loop asks whether the scene has history to step through
    # this one has none, so frames are jumped to.
    monkeypatch.setattr(worker, "requires_preroll", lambda scene: False)
    monkeypatch.setattr(worker, "bpy", SimpleNamespace(
        data=SimpleNamespace(objects=SimpleNamespace(get=objects.get)),
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)),
        ops=SimpleNamespace(render=SimpleNamespace(
            render=render_still or scene.render_still))))
    return worker


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """The fridge run's cut structure: CAM_Establish at 1, CAM_Grasp at 161."""
    def build(render_still=None):
        establish, grasp = _camera("CAM_Establish", 28), _camera("CAM_Grasp", 40)
        markers = [SimpleNamespace(name="CAM_Establish", frame=1, camera=establish),
                   SimpleNamespace(name="CAM_Grasp", frame=161, camera=grasp)]
        scene = _Scene(markers)
        objects = {"CAM_Establish": establish, "CAM_Grasp": grasp}
        worker = _worker(monkeypatch, scene, tmp_path / "out", objects, render_still)
        return worker, scene, markers, establish, grasp
    return build


def test_requested_camera_survives_a_cut_between_the_two_endpoints(setup):
    """Output 158 -> native 159 is inside CAM_Establish's shot, 204 -> 205 is
    inside CAM_Grasp's. A phase spanning the cut must still come back as two
    views of one camera; that is the whole point of naming one.
    """
    worker, scene, _, _, grasp = setup()
    result = worker.render_frames({"op": "render", "frames": [158, 204],
                                   "camera_name": "CAM_Grasp"})
    assert scene.rendered == [(159, "CAM_Grasp"), (205, "CAM_Grasp")]
    assert [v["lens_mm"] for v in result["views"]] == [grasp.data.lens] * 2


def test_recorded_views_describe_the_camera_that_actually_drew(setup):
    """`views` written before the render would echo the request instead of the
    result, silently. Asserting the lens against the request alone would still
    pass on that;
    what has to hold is that `views` agrees with the camera the render used.
    """
    worker, scene, _, establish, grasp = setup()
    lenses = {"CAM_Establish": establish.data.lens, "CAM_Grasp": grasp.data.lens}
    assert establish.data.lens != grasp.data.lens
    result = worker.render_frames({"op": "render", "frames": [158, 204],
                                   "camera_name": "CAM_Grasp"})
    assert [v["frame"] for v in result["views"]] == [158, 204]
    drew = [name for _, name in scene.rendered]
    assert [v["lens_mm"] for v in result["views"]] == [lenses[n] for n in drew]


def test_markers_keep_their_cameras_after_an_override_render(setup):
    """The markers are the film's cuts. Borrowing them for one diagnostic
    render must not edit the project.
    """
    worker, _, markers, establish, grasp = setup()
    worker.render_frames({"op": "render", "frames": [158, 204],
                          "camera_name": "CAM_Grasp"})
    assert [m.camera for m in markers] == [establish, grasp]


def test_markers_are_restored_when_the_render_raises(setup):
    """A scene left with its cuts stripped would silently retime every later
    render of the same project, so the restore cannot sit on the happy path.
    """
    def boom(**kwargs):
        raise RuntimeError("render failed")

    worker, _, markers, establish, grasp = setup(render_still=boom)
    with pytest.raises(RuntimeError, match="render failed"):
        worker.render_frames({"op": "render", "frames": [158],
                              "camera_name": "CAM_Grasp"})
    assert [m.camera for m in markers] == [establish, grasp]


def test_timeline_switching_is_untouched_without_an_explicit_camera(setup):
    """No camera named means "show me what the film shows", so the cuts must
    still apply and `views` must report each shot's own camera.
    """
    worker, scene, _, establish, grasp = setup()
    result = worker.render_frames({"op": "render", "frames": [158, 204]})
    assert scene.rendered == [(159, "CAM_Establish"), (205, "CAM_Grasp")]
    assert [v["lens_mm"] for v in result["views"]] == [establish.data.lens,
                                                       grasp.data.lens]


def test_an_unknown_camera_name_still_raises(setup):
    worker, _, _, _, _ = setup()
    with pytest.raises(ValueError, match="unknown native camera: CAM_Nope"):
        worker.render_frames({"op": "render", "frames": [0],
                              "camera_name": "CAM_Nope"})
