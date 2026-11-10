"""Committed object/scene IRs the animation tests build motion on top of.

Not a test module. The Blender animation tests import them, and they use only
the object/scene SDK, so keep them free of animation imports.
"""

from dycraft.sdk import Object, ObjectHandle, Scene
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton


def _box_object(name, w=1.0, l=1.0, h=1.0):
    _reset_object_singleton()
    o = Object(name)
    p = o.make_box("body", w, l, h)
    o.declare_root(p)
    return o.finalize()


def _hinged_object(name="cabinet"):
    """Body plus a door on a revolute joint, so a joint pose changes geometry."""
    _reset_object_singleton()
    o = Object(name)
    body = o.make_box("body", 0.6, 0.4, 0.8)
    door = o.make_box("door", 0.02, 0.4, 0.8)
    o.declare_root(body)
    o.attach_revolute(
        "hinge", body, door, axis=(0.0, 0.0, 1.0), limits=(0.0, 90.0),
        default=0.0,
        parent_anchor=o.anchor(body, "top_front_left"),
        child_anchor=o.anchor(door, "top_front_left"),
    )
    return o.finalize()


def _stack_scene(follow=True):
    """A 1 m table with a 0.2 m box resting on its top, box attached to table."""
    _reset_scene_singleton()
    table_ir = _box_object("table", 1.0, 1.0, 0.75)
    box_ir = _box_object("box", 0.2, 0.2, 0.2)
    catalog = {table_ir["id"]: table_ir, box_ir["id"]: box_ir}
    s = Scene("stack")
    s.set_room(width=4.0, depth=4.0, height=3.0)
    th = ObjectHandle(table_ir)
    bh = ObjectHandle(box_ir)
    table = s.place(th, name="table", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    s.attach(bh, name="box", parent=table,
             parent_anchor=s.anchor(table, "top"),
             child_anchor=s.anchor(bh, "bottom"),
             follow=follow)
    return s.finalize(), catalog
