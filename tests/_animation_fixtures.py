"""Deterministic authoring fixture: actual CAD source and native Blender jobs."""
import time
from types import SimpleNamespace

from dycraft.agent.layers.film import FilmContext
from dycraft.animation.session import AnimationSession


SOURCE = '''
obj = Object("block")
obj.declare_root(obj.add_part("body", Box(WIDTH, 0.2, 0.2), material=Material(color=COLOR)))
scene = Scene("stage")
scene.set_room(4.0, 4.0, 3.0)
scene.place(obj.finalize(), name="block", pose=((0, 0, 0), (0, 0, 0)))

@lighting
def rig(L):
    key = L.new_light("key", type="AREA")
    key.data.energy = 300.0
    key.location = (1.0, -1.0, 2.0)
    L.aim(key, (0.0, 0.0, 0.0))
    L.world.color = (0.05, 0.05, 0.05)

scene.set_lighting(rig)
ir = scene.finalize()
'''
# The rig is authored rather than inherited: a v2 scene gets exactly the
# lights its `@lighting` block creates, and none when it has no block.


def source(width=.2, color=(180, 100, 30)):
    return f"WIDTH = {width!r}\nCOLOR = {color!r}\n" + SOURCE


def plan():
    return {"intent": "A small block slides across the floor in a single readable shot.",
            "criteria": [{"id": "support", "requirement": "block remains on the floor",
                          "why": "sliding contact establishes the action", "observe": "oblique view of the lower edge"}],
            "shots": [{"id": "slide", "intent": "show the whole slide", "camera": "Camera",
                       "states": [{"frame": 0, "purpose": "initial support"}, {"frame": 1, "purpose": "final support"}]}],
            "delivery": {"min_pixels": 64 * 64, "reason": "explicit tiny deterministic integration-test target"}}


def flat_plan():
    """The current plan shape: states at the top level, without frames.

    `plan()` stays nested because the nested shape is still accepted and
    `plan()` is used all over the suite. This is the shape the prompt
    documents, so the gate and `review()` are tested on it too.
    """
    base = {k: v for k, v in plan().items() if k != "shots"}
    return {**base,
            "shots": [{"id": "slide", "intent": "show the whole slide", "camera": "Camera"}],
            "states": [{"id": "start", "purpose": "initial support"},
                       {"id": "end", "purpose": "final support"}]}


def state_reviews(note):
    """One observation per state the fixture plan declares.

    Review is per state now, and the fixture's nested plan gives its two
    states the ids `slide/0` and `slide/1`. Tests say what they inspected
    once and let this spread it, so adding a state to the fixture does not
    mean editing every acceptance in the suite.
    """
    from dycraft.animation.planning import plan_states, validate_plan
    return {state["id"]: note for state in plan_states(validate_plan(plan()))}


def service(root):
    context = FilmContext(run_dir=root, user_prompt=plan()["intent"])
    session = SimpleNamespace(ctx=context, trace=lambda *a, **k: None,
                              current_op_step=lambda: len(list((root / "native/jobs").glob("*/job.json"))),
                              finalize_meta=lambda *a: None)
    return AnimationSession(session)


def call(service, operation, *, timeout=180, **kwargs):
    submission = service.submit(operation, **kwargs)
    if not submission.get("job_id"):
        return submission
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.jobs.status(submission["job_id"])
        if record["status"] not in ("queued", "running"):
            return record["result"]
        time.sleep(.1)
    service.jobs.cancel(submission["job_id"])
    raise TimeoutError(f"{operation}: {record}")


STAGE = '''
from mathutils import Vector
s = bpy.context.scene
s.frame_start = 1
s.frame_end = 2
s.render.fps = 24
camera = s.camera
camera.name = 'Camera'
camera.location = (1.4, -1.8, 1.1)
camera.rotation_euler = (Vector((0.1, 0, 0.1))-camera.location).to_track_quat('-Z', 'Y').to_euler()
s.display.shading.light = 'STUDIO'
'''

ANIMATE = '''
root = bpy.data.objects['joint/world__block']
root.location.x = 0
root.keyframe_insert('location', frame=1)
root.location.x = .05
root.keyframe_insert('location', frame=2)
'''
