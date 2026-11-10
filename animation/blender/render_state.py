"""What a frame's pixels depend on beyond its signature: the render-state digest.

The `signature` covers where every body is, its mesh, color, hidden flag, and
joints. The pixels also depend on materials, lights, the world, the active
camera and its optics, textures, mesh attributes such as UVs, visibility flags,
the compositor, and color management. This digests those, so a frame drawn at
an older version can be admitted at any index when nothing it shows has
changed: `final_render(frames_from)` admits by `(signature, render_state)`, and
the held-frame skip reads the same pair.

**Built by exclusion**, so it cannot go stale: one walk of every datablock's
RNA properties per evaluation, at a fixed frame, then per frame the evaluated
values of the animated look properties, and mesh attributes only for objects
whose time-varying modifiers can change them. The walk takes seconds on a film.

**Left out**, because the render path sets them and the digest would otherwise
differ between the check and the delivery: the `final_render_settings` keys,
the Workbench draw profile, the output format and frame range, persistent data,
and the display colors `render_frames` syncs from each base color
(canonicalized to what the sync produces). The walk also skips
`scene.cycles.denoiser` and `preview_denoiser`, whose first read probes devices
for 8-11 s, every read-only value, and image `pixels`: an image is digested by
its file or packed bytes, since reading pixels one float at a time takes
seconds per large image.

An ID's animated channels are its own slot's: on Blender 4.4+ one action can
hold every object's curves in hundreds of slots.

**Abstains rather than guesses** (`None`): on frames where a visible object
is of a type the evaluation does not record (volumes, point clouds, hair
curves, grease pencil, rendered particles, gas domains), and on every frame
when a Python driver drives a look property or a frame-change handler is
registered, since neither can be read off the data.

**Index-free**, so a retimed frame still matches, except where the pixels
themselves depend on the frame number: Cycles' animated seed, a stamp that
prints the frame, and image sequences and movies, which read a file per
frame. With motion blur on, a frame's digest also covers its neighbors'
signatures and digests (`finish`).

Stable across processes: floats are canonicalized as the signature's are
(-0.0 is +0.0), enum-flag sets are sorted, and custom-property groups are
read as plain data (their `repr` holds a memory address). The Blender version
is part of every digest, so two versions never compare equal.
"""
import hashlib
import json
import os

#: What the digest is; a consumer compares only digests of the same version.
RENDER_STATE_VERSION = "dycraft.render_state.v2"

_SKIP = {"rna_type", "users", "session_uid", "is_evaluated", "original", "tag", "is_runtime_data",
         "is_missing", "is_embedded_data", "library_weak_reference", "asset_data", "preview",
         "use_extra_user", "is_library_indirect", "override_library", "bl_rna", "id_data",
         "is_editmode", "is_dirty", "filepath", "filepath_raw", "packed_file", "packed_files",
         "bindcode", "is_float", "has_data", "frame_duration", "animation_data",
         "evaluated_depsgraph_get", "depsgraph", "object_instances", "updates", "ids",
         "objects_all", "name_full"}
#: Per datablock kind: what the signature already covers, or what varies per
#: frame and is read per frame instead.
_KIND_SKIP = {
    "objects": {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle",
                "scale", "delta_location", "delta_rotation_euler", "delta_rotation_quaternion",
                "delta_scale", "matrix_world", "matrix_local", "matrix_basis",
                "matrix_parent_inverse", "bound_box", "dimensions", "color", "hide_render",
                "pose", "mode", "select", "motion_path"},
    "meshes": {"vertices", "edges", "polygons", "loops", "loop_triangles",
               "loop_triangle_polygons", "vertex_normals", "polygon_normals", "corner_normals",
               "attributes", "uv_layers", "vertex_colors", "color_attributes", "skin_vertices",
               "vertex_creases", "edge_creases", "shape_keys", "normals_domain",
               "vertex_paint_mask", "uv_layer_clone", "uv_layer_stencil"},
    "scenes": {"frame_current", "frame_current_final", "frame_float", "frame_start",
               "frame_end", "frame_step", "objects", "statistics", "cursor", "tool_settings",
               "sequence_editor", "keying_sets", "keying_sets_all", "timeline_markers",
               "collection"},
}
#: Struct properties the render path sets: `render_config.final_render_settings`'
#: keys and the Workbench profile, plus output format, persistent data, and the
#: device. Skipped wherever they appear.
_RENDER_PATH = {"engine", "resolution_x", "resolution_y", "resolution_percentage", "samples",
                "taa_render_samples", "use_denoising", "denoiser", "use_adaptive_sampling",
                "adaptive_threshold", "threads", "threads_mode", "device", "use_persistent_data",
                "preview_denoiser",
                "image_settings", "ffmpeg", "render_aa", "show_shadows"}
_DEPTH = 7
_COLLECTIONS = ("materials", "node_groups", "lights", "worlds", "cameras", "textures", "images",
                "scenes", "collections", "objects", "meshes", "curves", "libraries", "linestyles",
                "particles")
_ANIMATED = ("objects", "materials", "node_groups", "lights", "worlds", "cameras", "textures",
             "scenes", "meshes", "curves", "particles", "linestyles")
_LOOK_TYPES = {"Material", "ShaderNodeTree", "World", "Camera", "Texture", "Image", "Scene",
               "PointLight", "SunLight", "SpotLight", "AreaLight", "Light", "CompositorNodeTree",
               "ParticleSettings", "FreestyleLineStyle"}
_TRANSFORM = ("location", "rotation_", "scale", "delta_", "pose.bones[", "constraints[")
_UNCOVERED = {"VOLUME", "POINTCLOUD", "CURVES", "GPENCIL", "GREASEPENCIL"}
#: Modifiers that can change attributes or topology over time. Deform-only
#: ones change positions, which the signature covers.
_ATTRIBUTE_MODIFIERS = {"MESH_SEQUENCE_CACHE", "OCEAN", "DYNAMIC_PAINT", "EXPLODE", "BUILD",
                        "PARTICLE_SYSTEM", "FLUID", "REMESH", "SKIN"}
_TIME_NODES = {"GeometryNodeInputSceneTime", "GeometryNodeSimulationInput",
               "GeometryNodeSimulationOutput"}


def _canon(value):
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value))
    return value


class RenderStates:
    """One evaluation's render-state digests. `frame(i)` after the scene is on `i`."""

    def __init__(self, bpy, info, *, display_color=None):
        self.bpy = bpy
        self.info = info
        self.display_color = display_color
        scene = bpy.context.scene
        self.abstain = self._abstain_reason()
        self.attr_names = self._attribute_node_names()
        self.channels = self._look_channels()
        self.attribute_objects = self._attribute_objects()
        self.uncovered = [obj for obj in scene.objects if self._uncovered(obj)]
        self.per_frame_index = self._frame_dependent(scene)
        self.motion_blur = bool(getattr(scene.render, "use_motion_blur", False))
        self.static = None if self.abstain else self._static_walk()

    # -- the film-wide decisions ------------------------------------------

    def _abstain_reason(self):
        handlers = self.bpy.app.handlers
        # The harness's own observers (`dycraft_internal`) read updates and
        # change nothing a frame looks like.
        if any(not getattr(handler, "dycraft_internal", False)
               for handler in (*handlers.frame_change_pre, *handlers.frame_change_post)):
            return "frame-change handlers are registered"
        for idb, curve, _ in self._animated_curves():
            driver = curve.driver if hasattr(curve, "driver") else None
            if driver is None or driver.type != "SCRIPTED" or getattr(driver, "is_simple_expression", False):
                continue
            if self._is_look(idb, curve.data_path):
                return f"a Python driver drives {type(idb).__name__} {idb.name!r} {curve.data_path}"
        return None

    def _attribute_node_names(self):
        names = set()
        trees = [m.node_tree for m in self.bpy.data.materials if m.node_tree] + list(self.bpy.data.node_groups)
        for tree in trees:
            for node in tree.nodes:
                if (node.bl_idname == "ShaderNodeAttribute"
                        and getattr(node, "attribute_type", "") in ("OBJECT", "INSTANCER", "VIEW_LAYER")):
                    names.add(node.attribute_name)
        return names

    def _is_look(self, idb, path):
        kind = type(idb).__name__
        if kind in _LOOK_TYPES:
            return True
        if kind == "Object":
            if path.startswith(_TRANSFORM) or path in ("color", "hide_render"):
                return False
            if path.startswith('["'):
                return path[2:path.index('"', 2)] in self.attr_names
            if path.startswith("data."):
                return idb.type in ("CAMERA", "LIGHT")
            return path.startswith(("visible_", "is_holdout", "is_shadow_catcher", "pass_index",
                                    "active_material"))
        if kind == "Mesh":
            return path.startswith("materials")
        return False

    def _animated_curves(self):
        """`(id, fcurve or driver, is_driver)` for every animated channel of every datablock."""
        for name in _ANIMATED:
            for idb in getattr(self.bpy.data, name, ()):
                for owner in (idb, getattr(idb, "node_tree", None)):
                    data = getattr(owner, "animation_data", None) if owner is not None else None
                    if data is None:
                        continue
                    plays = [(data.action, _slot(data))] + [
                        (strip.action, _slot(strip)) for track in data.nla_tracks
                        if not track.mute for strip in track.strips if not strip.mute]
                    for action, slot in plays:
                        for curve in _action_curves(action, slot):
                            yield owner, curve, False
                    for curve in data.drivers:
                        yield owner, curve, True

    def _look_channels(self):
        """`(id, data path, index)` of each animated look property, read per frame."""
        seen, out = set(), []
        for idb, curve, _ in self._animated_curves():
            if not self._is_look(idb, curve.data_path):
                continue
            key = (type(idb).__name__, idb.name, curve.data_path, curve.array_index)
            if key not in seen:
                seen.add(key)
                out.append((idb, curve.data_path, curve.array_index))
        return out

    def _attribute_objects(self):
        out = []
        for obj in self.bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            for modifier in obj.modifiers:
                timed = modifier.type in _ATTRIBUTE_MODIFIERS or (
                    modifier.type == "NODES" and _time_dependent(modifier.node_group))
                animated = bool(obj.animation_data) and any(
                    curve.data_path.startswith(f'modifiers["{modifier.name}"]')
                    for curve in _action_curves(obj.animation_data.action,
                                                _slot(obj.animation_data)))
                if timed or (animated and modifier.type not in ("ARMATURE", "LATTICE", "HOOK")):
                    out.append(obj)
                    break
        return out

    @staticmethod
    def _uncovered(obj):
        if obj.type in _UNCOVERED:
            return True
        if any(system.settings.render_type in ("PATH", "OBJECT", "COLLECTION")
               for system in getattr(obj, "particle_systems", ())):
            return True
        return any(m.type == "FLUID" and m.fluid_type == "DOMAIN"
                   and m.domain_settings.domain_type == "GAS" for m in obj.modifiers)

    def _frame_dependent(self, scene):
        images = any(image.source in ("SEQUENCE", "MOVIE") for image in self.bpy.data.images)
        seed = bool(getattr(getattr(scene, "cycles", None), "use_animated_seed", False))
        stamp = bool(scene.render.use_stamp and scene.render.use_stamp_frame)
        return images or seed or stamp

    # -- the static walk -------------------------------------------------

    def _static_walk(self):
        bpy = self.bpy
        digest = hashlib.blake2b(digest_size=32)
        digest.update(("blender " + bpy.app.version_string).encode())
        for name in _COLLECTIONS:
            for idb in getattr(bpy.data, name, ()):
                digest.update((name + ":" + idb.name_full).encode())
                self._walk(idb, digest, 0, set(), _KIND_SKIP.get(name, ()))
                if name == "meshes":
                    _mesh_attributes(idb, digest)
                elif name == "images":
                    digest.update(self._image_content(idb))
                elif name == "libraries":
                    digest.update(_file_digest(bpy.path.abspath(idb.filepath)))
                elif name == "materials" and self.display_color is not None:
                    digest.update(repr(tuple(_canon(float(v)) for v in self.display_color(idb))).encode())
        return digest.digest()

    def _walk(self, struct, digest, depth, seen, skip):
        bpy = self.bpy
        try:
            properties = struct.bl_rna.properties
        except Exception:                                       # noqa: BLE001
            return
        for prop in properties:
            name = prop.identifier
            if name in _SKIP or name in skip or name in _RENDER_PATH:
                continue
            if name == "diffuse_color" and isinstance(struct, bpy.types.Material) and self.display_color:
                continue                     # canonicalized in `_static_walk`
            if prop.type not in ("POINTER", "COLLECTION") and prop.is_readonly:
                # Derived from what is walked, and some of it from the render
                # engine the delivery selects: `render.use_spherical_stereo`
                # flips with it. Not even read, because reading some is costly.
                continue
            if name == "pixels" and isinstance(struct, bpy.types.Image):
                continue                     # `_image_content` digests the image
            try:
                value = getattr(struct, name)
            except Exception:                                   # noqa: BLE001
                continue
            if prop.type == "POINTER":
                if value is None:
                    digest.update(b"\x00")
                elif isinstance(value, bpy.types.ID):
                    digest.update(("@" + type(value).__name__ + ":" + value.name_full).encode())
                elif depth < _DEPTH:
                    pointer = value.as_pointer() if hasattr(value, "as_pointer") else id(value)
                    if pointer not in seen:
                        seen.add(pointer)
                        try:
                            self._walk(value, digest, depth + 1, seen, ())
                        except ReferenceError:
                            pass
            elif prop.type == "COLLECTION":
                if depth >= _DEPTH:
                    continue
                try:
                    items = list(value)
                except Exception:                               # noqa: BLE001
                    continue
                digest.update(("[%d" % len(items)).encode())
                for item in items:
                    if isinstance(item, bpy.types.ID):
                        digest.update(("@" + item.name_full).encode())
                        continue
                    pointer = item.as_pointer() if hasattr(item, "as_pointer") else id(item)
                    if pointer not in seen:
                        seen.add(pointer)
                        try:
                            self._walk(item, digest, depth + 1, seen, ())
                        except ReferenceError:
                            pass
            else:
                try:
                    if getattr(prop, "is_array", False):
                        value = tuple(_canon(v) for v in value)
                    else:
                        value = _canon(value)
                    digest.update(repr(value).encode())
                except Exception:                               # noqa: BLE001
                    pass
        if isinstance(struct, bpy.types.ID):
            for key in sorted(struct.keys()):
                # Not an add-on's storage for a property registered on the
                # struct (`scene["cycles"]` is `scene.cycles`, walked above
                # without the render path's keys, which it repeats), and not
                # the harness's own metadata.
                if key in properties or key.startswith(("dycraft_", "_")):
                    continue
                value = struct[key]
                if hasattr(value, "to_dict"):
                    value = json.dumps(value.to_dict(), sort_keys=True, default=str)
                elif hasattr(value, "to_list"):
                    value = json.dumps(value.to_list(), default=str)
                elif isinstance(value, bpy.types.ID):
                    value = "@" + value.name_full
                digest.update(("%s=%r" % (key, _canon(value))).encode())

    def _image_content(self, image):
        if image.packed_file is not None:
            data = image.packed_file.data
            return hashlib.sha256(data if isinstance(data, bytes) else bytes(data, "latin-1")).digest()
        if image.source in ("FILE", "SEQUENCE", "TILED", "MOVIE"):
            return _file_digest(self.bpy.path.abspath(image.filepath, library=image.library))
        return image.source.encode()

    # -- per frame -------------------------------------------------------

    def frame(self, index, deps):
        """This frame's digest as hex, or `None` where it abstains."""
        if self.static is None:
            return None
        scene = self.bpy.context.scene
        for obj in self.uncovered:
            if not obj.evaluated_get(deps).hide_render:
                return None
        digest = hashlib.blake2b(self.static, digest_size=32)
        camera = scene.camera
        digest.update(("camera:" + (camera.name if camera else "")).encode())
        for idb, path, slot in self.channels:
            try:
                value = idb.evaluated_get(deps).path_resolve(path)
            except Exception:                                   # noqa: BLE001
                value = None
            if hasattr(value, "__len__") and not isinstance(value, str):
                value = value[slot] if 0 <= slot < len(value) else tuple(value)
            if hasattr(value, "name_full"):
                value = value.name_full
            digest.update(repr((idb.name, path, slot, _canon(value))).encode())
        for obj in self.attribute_objects:
            evaluated = obj.evaluated_get(deps)
            mesh = evaluated.to_mesh()
            try:
                _mesh_attributes(mesh, digest)
            finally:
                evaluated.to_mesh_clear()
        if self.per_frame_index:
            digest.update(b"frame:%d" % index)
        return digest.hexdigest()

    def finish(self, states, signatures):
        """With motion blur on, each frame's digest also covers its neighbors'."""
        if not self.motion_blur:
            return states
        out = {}
        for frame, state in states.items():
            around = [(signatures.get(frame + d), states.get(frame + d)) for d in (-1, 1)
                      if 0 <= frame + d < self.info["n_frames"]]
            if state is None or any(None in pair for pair in around):
                out[frame] = None
                continue
            out[frame] = hashlib.blake2b(json.dumps([state, around]).encode(),
                                         digest_size=32).hexdigest()
        return out


#: `_slot(holder)` when the Blender has no slots: read every curve, as before.
_ANY_SLOT = object()


def _slot(holder):
    """The action slot an `AnimData` or NLA strip plays, or `_ANY_SLOT` before 4.4."""
    return getattr(holder, "action_slot", _ANY_SLOT)


def _action_curves(action, slot=_ANY_SLOT):
    """The F-Curves `slot` plays of an action: its own channelbags on 4.4+, else the flat list.

    A slot of `None` -- an action assigned to nothing in it -- animates nothing.
    """
    if action is None:
        return []
    layers = list(getattr(action, "layers", ()) or ())
    if layers and slot is not _ANY_SLOT:
        if slot is None:
            return []
        curves = []
        for layer in layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot) if hasattr(strip, "channelbag") else None
                if bag is not None:
                    curves.extend(bag.fcurves)
        return curves
    curves = []
    for layer in layers:
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()) or ():
                curves.extend(bag.fcurves)
    if curves:
        return curves
    try:
        return list(action.fcurves)
    except Exception:                                           # noqa: BLE001
        return []


def _time_dependent(tree, seen=None):
    seen = set() if seen is None else seen
    if tree is None or tree.name in seen:
        return False
    seen.add(tree.name)
    if getattr(tree, "animation_data", None) is not None:
        return True
    return any(node.bl_idname in _TIME_NODES
               or (node.bl_idname == "GeometryNodeGroup" and _time_dependent(node.node_tree, seen))
               for node in tree.nodes)


def _mesh_attributes(mesh, digest):
    """Every attribute but positions, in the signature's float32 encoding."""
    import numpy as np
    for attribute in mesh.attributes:
        if attribute.name == "position" or attribute.name.startswith("."):
            continue
        count = len(attribute.data)
        if not count:
            continue
        kind = attribute.data_type
        field = {"FLOAT2": "vector", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color",
                 "BYTE_COLOR": "color"}.get(kind, "value")
        try:
            first = getattr(attribute.data[0], field)
        except Exception:                                       # noqa: BLE001
            continue
        width = len(first) if hasattr(first, "__len__") else 1
        dtype = (np.int32 if kind in ("INT", "INT8", "INT32_2D")
                 else np.bool_ if kind == "BOOLEAN" else np.float32)
        values = np.empty(count * width, dtype=dtype)
        try:
            attribute.data.foreach_get(field, values)
        except Exception:                                       # noqa: BLE001
            continue
        if values.dtype == np.float32:
            values = values + np.float32(0)
        digest.update(attribute.name.encode() + attribute.domain.encode() + values.tobytes())


def _file_digest(path):
    if not path or not os.path.isfile(path):
        return b"missing:" + str(path).encode()
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.digest()
