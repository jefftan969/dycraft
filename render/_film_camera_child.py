"""In Blender: the camera a film opens on, or with `--all` its camera at every frame, read from its `.blend` without
evaluating anything.

usage: <blender-python> _film_camera_child.py <animation.blend> <result.json> [--all]

The film's scene is loaded as library data, so nothing in it is evaluated: a heavy film's file takes minutes to open
and evaluate, and this read a fraction of a second. The camera is the one the timeline markers bind at the first
frame, else the scene's own, posed from its own transform and animation curves at that frame. A camera that a
constraint or a constrained parent places cannot be posed this way, and is reported as such. The result has the
fields of a view in a delivered film's `camera.json`, plus the render resolution. With `--all`, `views` holds one such
view per frame from the scene's first to its last, each with the camera the markers bind there; a frame whose camera
a constraint places is left out.
"""
import json
import sys

import bpy  # noqa: E402 -- the whole point of this interpreter
from mathutils import Euler, Matrix, Quaternion  # noqa: E402

ALL = "--all" in sys.argv
blend, result_path = [a for a in sys.argv if a != "--all"][-2:]


def channel(idb, path, index, default, frame):
    """`idb`'s animated value of `path[index]` at `frame`, or `default` when no curve drives it."""
    ad = idb.animation_data
    fc = ad.action.fcurves.find(path, index=index) if ad and ad.action else None
    return fc.evaluate(frame) if fc else default


def local(obj, frame):
    loc = [channel(obj, "location", i, obj.location[i], frame) for i in range(3)]
    scale = [channel(obj, "scale", i, obj.scale[i], frame) for i in range(3)]
    if obj.rotation_mode == "QUATERNION":
        rot = Quaternion([channel(obj, "rotation_quaternion", i, obj.rotation_quaternion[i], frame) for i in range(4)])
    elif obj.rotation_mode == "AXIS_ANGLE":
        angle_axis = [channel(obj, "rotation_axis_angle", i, obj.rotation_axis_angle[i], frame) for i in range(4)]
        rot = Quaternion(angle_axis[1:], angle_axis[0])
    else:
        rot = Euler([channel(obj, "rotation_euler", i, obj.rotation_euler[i], frame) for i in range(3)],
                    obj.rotation_mode)
    return Matrix.LocRotScale(loc, rot, scale)


def world(obj, frame):
    m = local(obj, frame)
    return world(obj.parent, frame) @ obj.matrix_parent_inverse @ m if obj.parent else m


def bound_camera(scene, frame):
    """The camera the timeline markers bind at `frame`, else the scene's own."""
    bound = [m for m in scene.timeline_markers if m.camera and m.frame <= frame]
    return max(bound, key=lambda m: m.frame).camera if bound else scene.camera


def constrained(cam):
    chain, obj = [], cam
    while obj is not None:
        chain.append(obj)
        obj = obj.parent
    return any(o.constraints for o in chain)


def view(cam, frame):
    data = cam.data
    return {"frame": frame, "camera": cam.name, "camera_matrix": [list(row) for row in world(cam, frame)],
            "camera_type": data.type, "lens_mm": channel(data, "lens", 0, data.lens, frame),
            "sensor_mm": data.sensor_width, "sensor_height_mm": data.sensor_height, "sensor_fit": data.sensor_fit,
            "shift": [channel(data, "shift_x", 0, data.shift_x, frame), channel(data, "shift_y", 0, data.shift_y, frame)]}


def read():
    with bpy.data.libraries.load(blend, link=True) as (src, dst):
        dst.scenes = list(src.scenes[:1])
    if not dst.scenes:
        return {"ok": False, "reason": "the file has no scene"}
    scene = dst.scenes[0]
    resolution = [scene.render.resolution_x, scene.render.resolution_y]
    if ALL:
        views = []
        for frame in range(scene.frame_start, scene.frame_end + 1):
            cam = bound_camera(scene, frame)
            if cam is not None and cam.type == "CAMERA" and not constrained(cam):
                views.append(view(cam, frame))
        if not views:
            return {"ok": False, "reason": "no frame has a camera that can be posed from its own curves"}
        return {"ok": True, "frame_start": scene.frame_start, "views": views, "resolution": resolution}
    frame = scene.frame_start
    cam = bound_camera(scene, frame)
    if cam is None or cam.type != "CAMERA":
        return {"ok": False, "reason": "the film has no camera at its first frame"}
    if constrained(cam):
        return {"ok": False, "reason": f"{cam.name} is placed by a constraint"}
    return {"ok": True, **view(cam, frame), "resolution": resolution}


try:
    result = read()
except Exception as error:  # noqa: BLE001 -- the caller keeps the viewer's own framing, with the reason
    result = {"ok": False, "reason": f"{type(error).__name__}: {error}"}
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh)
