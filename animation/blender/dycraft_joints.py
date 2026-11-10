"""A mechanism joint's coordinate is one native channel; `key_joint` keys it.

Every revolute or prismatic joint of an imported mechanism is an Empty,
`joint/<instance>__<joint>`, under a frame `origin/<instance>__<joint>` that is
turned so the joint's axis is the Empty's local Z. The joint's coordinate is
therefore a single channel of that Empty, and the channel *is* the joint: a
revolute joint's angle is `rotation_euler[2]` (radians, any number of turns),
a prismatic joint's travel is `location[2]` (meters). There is no driver and
no custom property between them.

Anything else written to the Empty -- another rotation component, another
location component, a whole `matrix_basis` -- moves the joint off its axis,
and `JointLimits` fails it as "joint transform violates its original motion
axis/frame" rather than letting it render.

Bound into `/tools` and into every `blender()` edit's namespace, so author
code can call `key_joint(...)` directly or `from dycraft_joints import
key_joint`.
"""

#: The channel that is each joint type's coordinate, as `(data path, index)`.
JOINT_CHANNELS = {"revolute": ("rotation_euler", 2), "prismatic": ("location", 2)}


def joint_object(name):
    """The joint Empty for `name`: `joint/robot__knee`, `robot__knee`, or `robot.knee`."""
    import bpy
    obj = bpy.data.objects.get(name if name.startswith("joint/")
                               else "joint/" + name.replace(".", "__", 1))
    if obj is not None and obj.get("joint_type") in JOINT_CHANNELS:
        return obj
    known = sorted(obj.name for obj in bpy.data.objects
                   if obj.name.startswith("joint/") and obj.get("joint_type") in JOINT_CHANNELS)
    raise KeyError(f"no revolute or prismatic joint named {name!r}; this project has "
                   + (", ".join(known[:20]) + (f" (+{len(known) - 20} more)" if len(known) > 20 else "")
                      if known else "none"))


def key_joint(name, value, frame):
    """Set joint `name` to `value` (radians or meters) and key it at `frame`.

    `frame` is a whole frame of the joint's active action, keyed at native
    frame `scene.frame_start + frame`: the output frame while that action
    plays where it was keyed, and clip time while you author an action a
    strip will place. Only the joint's own channel is written, so the joint
    cannot leave its axis. Returns the joint Empty.
    """
    import numbers
    import bpy
    obj = joint_object(name)
    path, index = JOINT_CHANNELS[obj["joint_type"]]
    if isinstance(frame, bool) or not isinstance(frame, numbers.Integral) or frame < 0:
        raise ValueError("frame must be a non-negative whole frame of the joint's action: "
                         "the output frame, or clip time while you author a clip")
    frame = int(frame)
    if path == "rotation_euler" and obj.rotation_mode != "XYZ":
        raise ValueError(f"{obj.name} is in rotation mode {obj.rotation_mode}, so "
                         "rotation_euler is not what moves it; set its rotation_mode "
                         "back to 'XYZ'")
    getattr(obj, path)[index] = float(value)
    obj.keyframe_insert(path, index=index, frame=bpy.context.scene.frame_start + frame)
    return obj
