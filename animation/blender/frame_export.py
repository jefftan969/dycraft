"""One evaluated frame as records: object geometry (cached when static), joints, bones, and surfaces."""
from collections import OrderedDict

import bpy
import numpy as np

try:
    from .surface_colors import surface_color
except ImportError:  # Read-only modules staged into isolated Blender.
    from surface_colors import surface_color


class FrameGeometryCache:
    """Job-local cache for plain mesh data; arbitrary deformation stays evaluated."""

    def __init__(self, max_bytes=256 * 1024**2):
        self.entries = OrderedDict()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.enabled = True
        self.hits = 0
        self.builds = 0
        # Observe updates when they happen: evaluated_depsgraph_get() can clear
        # update flags before begin_frame gets to inspect them.
        self._handler = self._updated
        bpy.app.handlers.depsgraph_update_post.append(self._handler)
        #: The originals Blender re-evaluated since the last frame, gathered by
        #: both handlers, and whether either ran. `held` returns a body's last
        #: extraction only when one did and none of the body's inputs is among
        #: them. A frame change's updates are only readable inside
        #: `frame_change_post`: after `frame_set` returns, `depsgraph.updates`
        #: is empty.
        # A plain function, so it can carry the mark `render_state` reads: it
        # only reads updates, and changes nothing a frame looks like.
        def frame_changed(scene, deps=None):
            self._note(deps)
        frame_changed.dycraft_internal = True
        self._frame_handler = frame_changed
        bpy.app.handlers.frame_change_post.append(self._frame_handler)
        self._touched, self._heard, self._blind = set(), False, False
        self._touched_now, self._reusable, self._held = set(), False, {}

    def close(self):
        if self._handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(self._handler)
        if self._frame_handler in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(self._frame_handler)

    def _note(self, deps):
        self._heard = True
        if deps is None:
            # A call without the depsgraph says nothing about what was
            # evaluated: nothing is reused on the next frame.
            self._blind = True
            return
        self._touched.update(update.id.original.as_pointer() for update in deps.updates)

    def _updated(self, scene, deps):
        self._note(deps)
        changed = set()
        for update in deps.updates:
            original = update.id.original
            # `Curve` covers text and surface data too: their evaluated
            # triangles are cached the way a mesh's are, and invalidated the same way.
            if isinstance(original, (bpy.types.Mesh, bpy.types.Curve)):
                changed.add(original.as_pointer())
            elif (isinstance(original, bpy.types.Object) and original.type in self._CACHED_TYPES
                  and update.is_updated_geometry):
                changed.add(original.data.as_pointer())
        for key in list(self.entries):
            if key[0] in changed:
                arrays = self.entries.pop(key)
                self.bytes -= sum(array.nbytes for array in arrays)

    def begin_frame(self, deps):
        # Python handlers can mutate data outside the usual animation channels.
        self.enabled = not any(handler is not self._handler and handler is not self._frame_handler
            for name in (
            'frame_change_pre', 'frame_change_post',
            'depsgraph_update_pre', 'depsgraph_update_post')
            for handler in getattr(bpy.app.handlers, name))
        if not self.enabled:
            self.entries.clear()
            self.bytes = 0
            self._held.clear()
        self._reusable = self.enabled and self._heard and not self._blind
        self._touched_now, self._touched, self._heard, self._blind = self._touched, set(), False, False

    def held(self, obj):
        """`obj`'s extraction on the frame before, if Blender re-evaluated none of its inputs since.

        Its object, its data, its active material, and that material's node
        tree: what `evaluated_frame` reads besides the matrix, which it reads
        for every object every frame. Every body whose extracted values change
        is among the depsgraph's updates, and usually only a minority of bodies
        are re-evaluated on a frame, so the rest skip their reads.
        """
        if not self._reusable:
            return None
        entry = self._held.get(obj.as_pointer())
        if entry is None or not self._touched_now.isdisjoint(entry[0]):
            return None
        return entry[1]

    def hold(self, obj, value):
        material = obj.active_material
        tree = getattr(material, 'node_tree', None) if material is not None else None
        inputs = (obj.as_pointer(), obj.data.as_pointer() if obj.data is not None else 0,
                  material.as_pointer() if material is not None else 0,
                  tree.as_pointer() if tree is not None else 0)
        self._held[inputs[0]] = (inputs, value)

    #: Modifiers whose evaluated output is a pure function of the input mesh and
    #: their own settings -- never of time, the object's world transform, or any
    #: other datablock. Membership is the whole soundness argument for reusing an
    #: evaluated array across frames, so a type may be added only when all three
    #: hold, and `_MODIFIER_SETTINGS` must then name every setting that changes
    #: its result:
    #:
    #:   * no ID-pointer input (object, texture, collection) -- those make the
    #:     result depend on something this key does not observe;
    #:   * no world-space or scene-state input (Blender has no world-space bevel
    #:     width or solidify thickness; an Array with an offset *object* would
    #:     disqualify Array, for example);
    #:   * no time dependence beyond its own animated settings, which the
    #:     descriptor reads from the EVALUATED modifier and therefore captures.
    #:
    #: Vertex groups, bevel weights, and creases are inputs too, and they live in
    #: the mesh datablock, which the key identifies by pointer and the
    #: depsgraph handler invalidates on edit.
    #:
    #: Bevel and Solidify are common on authored parts; without them here every
    #: such body would be rebuilt on every frame to produce identical arrays.
    #: With them admitted every exported array is bit-identical and the
    #: rebuilds stop.
    _MODIFIER_SETTINGS = {
        'EDGE_SPLIT': ('split_angle', 'use_edge_angle', 'use_edge_sharp'),
        'BEVEL': ('width', 'width_pct', 'segments', 'limit_method', 'angle_limit',
                  'offset_type', 'profile', 'profile_type', 'affect', 'material',
                  'miter_inner', 'miter_outer', 'spread', 'vertex_group',
                  'invert_vertex_group', 'harden_normals', 'loop_slide',
                  'use_clamp_overlap', 'mark_seam', 'mark_sharp', 'face_strength_mode'),
        'SOLIDIFY': ('thickness', 'thickness_clamp', 'thickness_vertex_group', 'offset',
                     'solidify_mode', 'nonmanifold_thickness_mode', 'nonmanifold_boundary_mode',
                     'use_even_offset', 'use_rim', 'use_rim_only', 'use_flip_normals',
                     'use_quality_normals', 'bevel_convex', 'edge_crease_inner',
                     'edge_crease_outer', 'edge_crease_rim', 'material_offset',
                     'material_offset_rim', 'vertex_group', 'invert_vertex_group',
                     'shell_vertex_group', 'rim_vertex_group'),
    }

    def modifier_descriptor(self, modifier):
        """Settings that decide a local modifier's output, or None if it is not local.

        `getattr(..., None)` rather than a bare read because the property set of
        a modifier type moves between Blender versions; a name that disappears
        contributes None on both sides of a comparison, and a name that is added
        is caught by the version check the evaluation format already carries.
        """
        names = self._MODIFIER_SETTINGS.get(modifier.type)
        if names is None:
            return None
        values = []
        for name in names:
            value = getattr(modifier, name, None)
            # Vector/array properties compare by identity otherwise.
            values.append(tuple(value) if hasattr(value, '__len__') and not isinstance(value, str) else value)
        # A custom bevel profile is an embedded curve, not a setting: read its
        # control points, because two objects sharing one mesh and one settings
        # tuple can still carry different profiles.
        if modifier.type == 'BEVEL' and getattr(modifier, 'profile_type', None) == 'CUSTOM':
            profile = getattr(modifier, 'custom_profile', None)
            if profile is None:
                return None
            values.append(tuple((point.location[0], point.location[1],
                                 point.handle_type_1, point.handle_type_2)
                                for point in profile.points))
        return (modifier.type, modifier.show_viewport, modifier.show_render, tuple(values))

    #: Object types whose evaluated triangles are a function of their own data
    #: and modifiers. Text and curves are otherwise re-triangulated by
    #: `to_mesh` on every frame, which on a scene with a few hundred text
    #: objects is most of the record extraction.
    _CACHED_TYPES = ('MESH', 'CURVE', 'FONT')

    @staticmethod
    def curve_inputs(data):
        """Whether a curve's evaluated surface reads another object, or `None` if it does not."""
        for name in ('bevel_object', 'taper_object', 'follow_curve'):
            if getattr(data, name, None) is not None:
                return name
        return None

    def key(self, obj, evaluated):
        if (not self.enabled or obj.type not in self._CACHED_TYPES or obj.mode != 'OBJECT'
                or getattr(obj.data, 'shape_keys', None) or obj.data.animation_data):
            return None
        # Deform parenting moves vertices, so keep it on the evaluated path.
        # Parenting to an object or to one bone moves the whole body, which
        # `matrix_world` carries, so a bone-parented prop's arrays never change.
        if obj.parent and obj.parent_type in ('ARMATURE', 'LATTICE'):
            return None
        fonts = ()
        if obj.type != 'MESH':
            # A bevel or taper object, or a text's path, is an input the key
            # cannot see.
            if self.curve_inputs(obj.data):
                return None
            fonts = tuple((font.name_full, font.filepath) if font is not None else None
                          for font in (getattr(obj.data, name, None)
                                       for name in ('font', 'font_bold', 'font_italic',
                                                    'font_bold_italic')))
        # Settings are read from the EVALUATED modifier, so an animated or
        # driven setting changes the key on the frame it changes.
        modifiers = []
        for modifier in evaluated.modifiers:
            descriptor = self.modifier_descriptor(modifier)
            if descriptor is None:
                return None
            modifiers.append(descriptor)
        return (obj.data.as_pointer(), obj.data.name_full, tuple(modifiers)) + ((fonts,) if fonts else ())

    def get(self, key):
        if key is not None and key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key]
        return None

    def put(self, key, arrays):
        self.builds += 1
        if key is None:
            return
        size = sum(array.nbytes for array in arrays)
        while self.entries and self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= sum(array.nbytes for array in old)
        if size <= self.max_bytes:
            for array in arrays:
                array.flags.writeable = False
            self.entries[key] = arrays
            self.bytes += size


def duplicating_objects(deps):
    """Objects whose instances this exporter would miss, by name.

    `evaluated_frame` walks `scene.objects` and takes `to_mesh()` from each,
    which is the source object's evaluated geometry at the source object's
    transform. Instanced *copies* -- vertex/face instancing, collection
    instances, geometry-nodes instance-on-points -- are not covered by that
    walk, so a scene with 500 instanced dominoes would export one and the
    verifiers would silently check one. Refusing is right.

    The test must not be `any(item.is_instance ...)`, which also fires on an
    evaluated curve or text object, because Blender represents a curve's
    generated surface as a geometry instance *of the curve itself*. On 4.5
    LTS:

        FONT plain            is_instance  obj=Text        parent=Text
        CURVE bevel_depth     is_instance  obj=BézierCurve parent=BézierCurve
        MESH VERTS instancing is_instance  obj=Icosphere   parent=Cube
        EMPTY collection      is_instance  obj=Cube        parent=Inst

    and `to_mesh()` already returns the full surface for the first two, so
    that test would refuse geometry this exports correctly, and beveled text
    is the obvious way to make a title plate.

    The distinguishing property is whether the instance is a *duplicate* of
    some other object or the object's own evaluated geometry: `instance_object
    is parent` for the latter. Names are compared because the depsgraph hands
    back transient RNA wrappers.
    """
    duplicating = set()
    for item in deps.object_instances:
        # Transient StructRNA: every attribute must be read inside the loop.
        if not item.is_instance or item.parent is None:
            continue
        source = item.instance_object
        if source is None or source.name != item.parent.name:
            duplicating.add(item.parent.name)
    return sorted(duplicating)


#: The channel that is each joint type's coordinate. Mirrors
#: `dycraft_joints.JOINT_CHANNELS`, restated rather than imported because this
#: file is also loaded outside the sandbox, where `/tools` is not on the path.
JOINT_CHANNELS = {"revolute": ("rotation_euler", 2), "prismatic": ("location", 2)}


#: What `FrameGeometryCache.held` returns for a body with no surface to export.
_NO_SURFACE = object()


def evaluated_frame(i, geometry_cache=None, *, with_transforms=False):
    """`(record, local)` of the current frame; with `with_transforms`, also
    the transforms as the float32 array the record's lists were made from
    (`evaluation.write_signed_record` encodes from it)."""
    deps = bpy.context.evaluated_depsgraph_get()
    duplicating = duplicating_objects(deps)
    if duplicating:
        raise ValueError(
            "Realize these geometry instances before checking; instanced copies cannot be "
            "checked: " + ", ".join(duplicating[:12])
            + (f" (+{len(duplicating) - 12} more)" if len(duplicating) > 12 else ""))
    if geometry_cache is not None:
        geometry_cache.begin_frame(deps)
    # Every object's matrix in one read. The originals' `matrix_world` is the
    # evaluated one: the active depsgraph writes it back after evaluating, and
    # the two agree on every object and frame of the films checked. Read one
    # object at a time, through `evaluated_get` and a `Matrix` per object, it
    # would be most of extracting a frame. Stored column by column, so the
    # rows `Matrix` iterates are the transpose.
    scene_objects = bpy.context.scene.objects
    names = scene_objects.keys()
    placed = np.empty(len(names) * 16, dtype=np.float32)
    scene_objects.foreach_get("matrix_world", placed)
    placed = placed.reshape(len(names), 4, 4).transpose(0, 2, 1)
    rows = placed.tolist()
    local, objects = {}, []
    # Only the evaluation's own stream reuses a body Blender did not touch: it
    # alone steps frames with nothing between them (`FrameGeometryCache.held`).
    reuse = with_transforms and geometry_cache is not None
    for n, obj in enumerate(scene_objects):
        if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
            continue
        held = geometry_cache.held(obj) if reuse else None
        if held is not None:
            if held is _NO_SURFACE:
                continue
            arrays, identity, room, hidden, color = held
            color = list(color)
        else:
            evaluated = obj.evaluated_get(deps)
            cache_key = geometry_cache.key(obj, evaluated) if geometry_cache is not None else None
            arrays = geometry_cache.get(cache_key) if geometry_cache is not None else None
            if arrays is None:
                mesh = evaluated.to_mesh()
                if mesh is None:
                    if reuse:
                        geometry_cache.hold(obj, _NO_SURFACE)
                    continue
                try:
                    mesh.calc_loop_triangles()
                    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
                    mesh.vertices.foreach_get("co", vertices.ravel())
                    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
                    mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
                    arrays = vertices, triangles
                finally:
                    evaluated.to_mesh_clear()
                if geometry_cache is not None:
                    geometry_cache.put(cache_key, arrays)
            color = surface_color(evaluated.active_material)
            identity, room = obj.get("dycraft_identity"), bool(obj.get("dycraft_room"))
            hidden = bool(obj.hide_render)
            if reuse:
                geometry_cache.hold(obj, (arrays, identity, room, hidden, list(color)))
        key = f"o{len(objects)}"
        local[obj.name] = arrays
        objects.append({"key": key, "name": obj.name,
                            "identity": identity,
                            "room": room,
                            "matrix": [row[:] for row in rows[n]],
                            "hidden": hidden,
                            "color": color})
    transforms = dict(zip(names, rows))
    joint_positions = {}
    for n, name in enumerate(names):
        if not name.startswith("joint/"):
            continue
        obj = scene_objects[n]
        channel = JOINT_CHANNELS.get(obj.get("joint_type"))
        if channel is not None:
            path, index = channel
            joint_positions[obj.name] = float(getattr(obj.evaluated_get(deps), path)[index])
    record = {"frame": i, "objects": objects, "transforms": transforms,
              "joint_positions": joint_positions}
    bones = bone_rotations(deps)
    if bones:
        record["bone_rotations"] = bones
    return (record, local, placed) if with_transforms else (record, local)


def armature_table(scene):
    """`{armature: {"bones": [...], "parents": [...], "rest": [...]}}`, the order `bone_rotations` uses.

    In the evaluation index rather than in every frame, for the reason
    `viewpoints` is: the names do not vary, and only the numbers belong in a
    record. `parents` holds indices into `bones`, `-1` for a root. `rest` is
    each bone's rest head, tail, and frame (columns = its axes, armature space),
    the rows the importer hands `rig_anatomy.skeleton_anatomy`, so
    `verify.current_anatomy` can derive a rig's joints when the anatomy its
    content-addressed ObjectIR froze predates the current one.
    """
    def rounded(values):
        return [round(float(v), 6) for v in values]
    table = {}
    for obj in scene.objects:
        if obj.type != "ARMATURE":
            continue
        names = [b.name for b in obj.data.bones]
        slots = {n: k for k, n in enumerate(names)}
        table[obj.name] = {"bones": names,
                           "parents": [slots[b.parent.name] if b.parent else -1
                                       for b in obj.data.bones],
                           "rest": [{"head": rounded(b.head_local), "tail": rounded(b.tail_local),
                                     "deform": bool(b.use_deform),
                                     "rotation": [rounded(row) for row in b.matrix_local.to_3x3()]}
                                    for b in obj.data.bones]}
    return table


def bone_rotations(deps):
    """Each armature's evaluated bone orientations, armature space, as unit quaternions.

    What a skinned character's motion actually is. Its object frame holds still
    while the bones move, and its mesh is one body, so the rigid channels see
    nothing and the deformation channel sees a forearm flipping 175 degrees as
    under a centimeter of RMS over the whole skin. `pose.bones[].matrix`
    of the *evaluated* armature, so IK and other constraints are included,
    unlike `matrix_basis`. Canonical sign (w >= 0) and six decimals, so the
    record and its signature do not depend on which of q/-q Blender returned.
    Absent from the record when the scene has no armature, which keeps every
    armature-free frame's signature what it was.
    """
    out = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        rows = []
        for pb in obj.evaluated_get(deps).pose.bones:
            q = pb.matrix.to_quaternion().normalized()
            if q.w < 0:
                q.negate()
            # `+ 0.0` folds -0.0 into +0.0, as `evaluation._f4` does for every
            # binary field: these rows are signed as residual JSON, where the two
            # zeros differ, and which one Blender returns depends on the path.
            rows.append([round(float(v), 6) + 0.0 for v in (q.w, q.x, q.y, q.z)])
        out[obj.name] = rows
    return out
