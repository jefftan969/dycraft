"""A mechanism joint is one native channel of its Empty, with no driver.

Each revolute or prismatic joint imports as `joint/<instance>__<joint>` under
an origin frame turned so the joint's axis is the Empty's local Z. Its
coordinate is `rotation_euler[2]` or `location[2]`, `key_joint` writes only
that, and anything else moves the joint off its axis, which `JointLimits`
reports. There is no `q` property or driver: deleting animation data would
remove them, a rotation-mode switch would leave them writing an ignored
channel, and a rebuild's matrix write could turn a joint the wrong way.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.animation.blender.verify import load_frame


#: Deliberately not a coordinate axis: the frame turning is what is tested.
HINGE_AXIS = (0.0, 0.6, 0.8)
SLIDE_AXIS = (0.6, 0.0, 0.8)


def _rig(lid_default=0.0):
    from dycraft.sdk import Object, ObjectHandle, Scene
    from dycraft.sdk.object_builder import _reset_object_singleton
    from dycraft.sdk.scene_builder import _reset_scene_singleton
    _reset_object_singleton()
    o = Object("rig")
    body = o.make_box("body", 0.4, 0.4, 0.4)
    lid = o.make_box("lid", 0.4, 0.4, 0.02)
    slider = o.make_box("slider", 0.1, 0.1, 0.1)
    o.declare_root(body)
    o.attach_revolute("lid", body, lid, axis=HINGE_AXIS, limits=(-720.0, 720.0),
                      default=lid_default, parent_anchor=o.anchor(body, "top_back"),
                      child_anchor=o.anchor(lid, "bottom_back"))
    o.attach_prismatic("slide", body, slider, axis=SLIDE_AXIS, limits=(0.0, 0.3),
                       default=0.0, parent_anchor=o.anchor(body, "front"),
                       child_anchor=o.anchor(slider, "back"))
    obj = o.finalize()
    _reset_scene_singleton()
    s = Scene("joints")
    s.set_room(width=4, depth=4, height=3)
    s.place(ObjectHandle(obj), name="rig", pose=((0, 0, 0), (0, 0, 0)))
    return s.finalize(), {obj["id"]: obj}


def _initialize(tmp_path):
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_python
    if blender_python() is None:
        pytest.skip("Blender interpreter not configured")
    scene, catalog = _rig()
    (tmp_path / "catalog").mkdir()
    workspace, inputs = tmp_path / "workspace", tmp_path / "inputs"
    workspace.mkdir()
    prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog), inputs)
    readonly = [inputs, tmp_path / "catalog"]

    def worker(name, request):
        return run_worker(workspace, tmp_path / name, request, readonly=readonly)
    worker("init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")})
    baseline = json.loads((tmp_path / "init/baseline.json").read_text())
    return worker, baseline, scene, catalog


def _rotation(axis, angle):
    x, y, z = np.asarray(axis) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


@pytest.mark.expensive
def test_a_joint_on_an_arbitrary_axis_moves_through_one_channel(tmp_path):
    from dycraft.animation.blender.verify import joint_position, verify
    worker, baseline, scene, catalog = _initialize(tmp_path)
    lid, slide = baseline["joints"]["rig__lid"], baseline["joints"]["rig__slide"]
    assert lid["axis"] == slide["axis"] == [0.0, 0.0, 1.0]
    assert lid["source_axis"] == pytest.approx(list(HINGE_AXIS))
    state = worker("inspect", {"op": "probe", "code": """
j = bpy.data.objects['joint/rig__lid']
result = {'q': 'q' in j, 'mode': j.rotation_mode,
          'drivers': len(j.animation_data.drivers) if j.animation_data else 0}
"""})
    assert state == {"q": False, "mode": "XYZ", "drivers": 0}

    turns = math.radians(630)
    worker("edit", {"op": "edit", "code": f"""
key_joint('rig__lid', 0.0, 0)
key_joint('rig.lid', {turns!r}, 1)
key_joint('joint/rig__slide', 0.0, 0)
key_joint('rig__slide', 0.2, 1)
bpy.context.scene.frame_end = 1
"""})
    worker("state", {"op": "evaluate"})
    rest, _ = load_frame(tmp_path / "state/evaluation", 0)
    moved, _ = load_frame(tmp_path / "state/evaluation", 1)
    assert moved["joint_positions"][lid["object"]] == pytest.approx(turns)
    assert moved["joint_positions"][slide["object"]] == pytest.approx(0.2)
    assert joint_position(moved, lid) == pytest.approx(turns, abs=1e-5)
    assert joint_position(moved, slide) == pytest.approx(0.2, abs=1e-6)

    # In world space: the lid turned about the URDF axis by 630 deg, the
    # slider traveled 0.2 m along its own, and the body did not move.
    def world(record, name):
        return np.asarray(record["transforms"][name])
    turned = world(moved, "part/rig/lid") @ np.linalg.inv(world(rest, "part/rig/lid"))
    assert turned[:3, :3] == pytest.approx(_rotation(HINGE_AXIS, turns), abs=1e-5)
    shifted = world(moved, "part/rig/slider")[:3, 3] - world(rest, "part/rig/slider")[:3, 3]
    assert shifted == pytest.approx(0.2 * np.asarray(SLIDE_AXIS), abs=1e-6)
    assert world(moved, "part/rig/body") == pytest.approx(world(rest, "part/rig/body"))

    reports = {r["name"]: r for r in verify(tmp_path / "state/evaluation", baseline, scene, catalog)}
    assert reports["JointLimits"]["verdict"] == "pass", reports["JointLimits"]


@pytest.mark.expensive
def test_writing_anything_but_the_joint_s_channel_fails_its_axis_check(tmp_path):
    from dycraft.animation.blender.verify import verify
    worker, baseline, scene, catalog = _initialize(tmp_path)
    worker("edit", {"op": "edit", "code": """
j = bpy.data.objects['joint/rig__lid']
j.rotation_euler[0] = 0.3
"""})
    worker("state", {"op": "evaluate"})
    reports = {r["name"]: r for r in verify(tmp_path / "state/evaluation", baseline, scene, catalog)}
    limits = reports["JointLimits"]
    assert limits["verdict"] == "fail"
    reasons = [finding["reason"] for finding in limits["evidence"]["findings"]]
    assert any("violates its original motion axis" in reason for reason in reasons), reasons


@pytest.mark.expensive
def test_key_joint_names_what_it_cannot_key(tmp_path):
    worker, *_ = _initialize(tmp_path)
    with pytest.raises(Exception, match="no revolute or prismatic joint named 'rig__nope'.*joint/rig__lid"):
        worker("bad_name", {"op": "edit", "code": "key_joint('rig__nope', 0.1, 0)"})
    with pytest.raises(Exception, match="rotation mode QUATERNION"):
        worker("bad_mode", {"op": "edit", "code": """
bpy.data.objects['joint/rig__lid'].rotation_mode = 'QUATERNION'
key_joint('rig__lid', 0.1, 0)
"""})


@pytest.mark.expensive
def test_a_rebuild_with_a_new_default_angle_keeps_the_joint_s_direction(tmp_path):
    """A `build()` that changes a joint's default writes only that joint's
    channel, and only for a joint the author has neither animated nor posed.
    A whole matrix written over an axis-angle Empty can bring a joint imported
    at -88 deg back at +88 deg."""
    from dycraft.tests._animation_fixtures import call, plan, service
    door = '''
obj = Object("cabinet")
body = obj.add_part("carcass", Box(0.6, 0.6, 0.9), material=Material(color=(120, 90, 60)))
door = obj.add_part("door", Box(0.6, 0.02, 0.8), material=Material(color=(200, 200, 205)))
obj.declare_root(body)
obj.attach_revolute("hinge", parent=body, child=door,
                    parent_anchor=obj.anchor(body, "front_left"),
                    child_anchor=obj.anchor(door, "back_left"),
                    axis=(0, 0, 1), limits=(-100, 100), default=DEFAULT)
scene = Scene("kitchen")
scene.set_room(4.0, 3.0, 2.6)
scene.place(obj.finalize(), name="cabinet", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''
    svc = service(tmp_path)
    probe = ("[bpy.data.objects['joint/cabinet__hinge'].rotation_euler[2], "
             "bpy.data.objects['part/cabinet/door'].matrix_world.to_euler()[2]]")
    try:
        (svc.design / "scene.py").write_text(door.replace("DEFAULT", "-88"))
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        before = call(svc, "probe", expression=probe)
        assert before["ok"] and before["result"][0] == pytest.approx(math.radians(-88))
        (svc.design / "scene.py").write_text(door.replace("DEFAULT", "-60"))
        rebuilt = call(svc, "build")
        assert rebuilt["ok"], rebuilt
        after = call(svc, "probe", expression=probe)
        assert after["ok"]
        # Untouched, the joint takes the source's new start, the same way.
        assert after["result"][0] == pytest.approx(math.radians(-60), abs=1e-6)
        assert after["result"][1] < 0, "the door must still be turned the negative way"
        # A pose the author set in Blender is theirs, keyed or not.
        assert call(svc, "blender", stage="storyboard", code=(
            "bpy.data.objects['joint/cabinet__hinge'].rotation_euler[2] = -1.0"))["ok"]
        (svc.design / "scene.py").write_text(door.replace("DEFAULT", "-55"))
        assert call(svc, "build")["ok"]
        assert call(svc, "probe", expression=probe)["result"][0] == pytest.approx(-1.0, abs=1e-6)
        # A project whose joints still carry a `q` property is refused by
        # name rather than merged into a joint that would turn off its axis.
        legacy = call(svc, "blender", stage="storyboard", code=(
            "j = bpy.data.objects['joint/cabinet__hinge']\n"
            "j.rotation_mode = 'AXIS_ANGLE'\nj['q'] = 0.5"))
        assert legacy["ok"], legacy
        (svc.design / "scene.py").write_text(door.replace("DEFAULT", "-50"))
        refused = call(svc, "build")
        assert not refused["ok"] and "predate driver-free joints" in refused["error"], refused
    finally:
        svc.jobs.close()


def test_the_worker_and_key_joint_agree_on_each_joint_type_s_channel(monkeypatch):
    """Not expensive: the worker restates the table because it is also loaded
    outside the sandbox, so the two copies are held equal here."""
    import importlib.util
    import sys
    from types import SimpleNamespace
    from dycraft.animation.blender import dycraft_joints
    from dycraft.animation.blender.runtime import REPO
    for name in ("bpy", "bpy_extras", "_blender_child"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    # The worker imports the modules beside it by name; drop any a previous test cached, so they see this bpy.
    for sibling in ("surface_colors", "frame_export", "animation_diff"):
        monkeypatch.delitem(sys.modules, sibling, raising=False)
    spec = importlib.util.spec_from_file_location("channels_worker", REPO / "animation/blender/worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert sys.modules["frame_export"].JOINT_CHANNELS == dycraft_joints.JOINT_CHANNELS
