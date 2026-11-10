"""Generated-data replacement inside the authored blend (isolated bpy worker).

Semantic names identify generated objects. Content hashes describe the donor
geometry, never identity. Existing objects keep their actions and references.
"""
import hashlib
import json

import bpy



def _set_materials(mesh, materials):
    """Replace `mesh`'s material slots, keeping every face on its own slot.

    `Mesh.materials.clear()` resets every face's `material_index` to 0, so a
    clear-and-append moved all of a multi-material mesh onto its first slot. An
    imported character is one such mesh: a German shepherd's three
    slots (hair cards, body, head) and a Rocketbox man's (body, head, opacity)
    all rendered with slot 0 -- hair cards over the whole dog -- after any build
    that changed geometry or refreshed materials. The indices are read before the
    clear and written back, clamped to the new slot count."""
    n = len(mesh.polygons)
    idx = [0] * n
    if n:
        mesh.polygons.foreach_get("material_index", idx)
    mesh.materials.clear()
    for material in materials:
        mesh.materials.append(material)
    if n and materials:
        top = len(materials) - 1
        mesh.polygons.foreach_set("material_index", [min(i, top) for i in idx])
        mesh.update()

def mesh_signature(mesh, *, topology=False):
    value = {"polygons": [list(p.vertices) for p in mesh.polygons],
             "vertices": len(mesh.vertices) if topology else [list(v.co) for v in mesh.vertices]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


#: Modifier types that store vertex *indices* of their own, so a change to the
#: index set silently rebinds them. `HOOK` is the one: its `vertex_indices`
#: are raw offsets into the mesh.
INDEX_BOUND_MODIFIERS = frozenset({"HOOK"})


def binding_modifiers(obj):
    """Modifiers whose bindings a topology change can actually invalidate.

    Not every modifier: what authors attach to generated parts is mostly
    appearance work (`BEVEL`, `SOLIDIFY`, `WEIGHTED_NORMAL`), which holds no
    per-vertex reference, so no topology change can break it. Refusing on any
    modifier would make a refusal, and a recompile before an `accept_changes`
    that changes nothing, the common case.

    Two kinds genuinely bind: a modifier holding vertex indices directly
    (`HOOK`), and one that resolves a vertex group by name (`ARMATURE`,
    `MASK`, `DISPLACE`, ... -- `vertex_group` is a common property rather than
    a per-type one, so it is read as one). Both are named here; the rest stop
    producing an issue at all rather than producing a benign one to be
    acknowledged.
    """
    return sorted(m.name for m in obj.modifiers
                  if m.type in INDEX_BOUND_MODIFIERS or getattr(m, "vertex_group", ""))


def generated_record(scene):
    result = {}
    for obj in scene.objects:
        # `rig/` is an imported character's armature (`worker.RIG_PREFIX`).
        if (obj.name.startswith(("link/", "origin/", "joint/", "part/", "rig/"))
                or obj.get("dycraft_room") or obj.type == "LIGHT"):
            result[obj.name] = {"type": obj.type, "parent": obj.parent.name if obj.parent else None,
                                "matrix": [list(row) for row in obj.matrix_basis]}
            if obj.type == "MESH":
                result[obj.name].update(geometry=mesh_signature(obj.data),
                                       topology=mesh_signature(obj.data, topology=True))
    return result


#: The scene settings a `@lighting` block can write besides lights and the
#: World (`L.view`, `L.display`, `L.render`, `L.cycles`).
LIGHTING_SETTINGS = ("view_settings", "display_settings", "render", "cycles")


def scene_settings(scene):
    """`{"owner.property": value}` for the plain settings a lighting block can write."""
    out = {}
    for owner in LIGHTING_SETTINGS:
        holder = getattr(scene, owner, None)
        if holder is None:
            continue
        for prop in holder.bl_rna.properties:
            if prop.is_readonly or prop.type in ("POINTER", "COLLECTION"):
                continue
            value = getattr(holder, prop.identifier)
            if not isinstance(value, (bool, int, float, str)):
                try:
                    value = list(value)
                except TypeError:
                    continue
            out[f"{owner}.{prop.identifier}"] = value
    return out


#: Written first, in this order, because each limits what the next accepts:
#: a view transform belongs to a display device, and a look to a view transform.
SETTINGS_FIRST = ("display_settings.display_device", "view_settings.view_transform")


def apply_scene_settings(scene, settings):
    """Write back what `scene_settings` read, one property at a time."""
    rank = {path: i for i, path in enumerate(SETTINGS_FIRST)}
    for path in sorted(settings, key=lambda path: rank.get(path, len(rank))):
        value = settings[path]
        owner, _, identifier = path.partition(".")
        try:
            setattr(getattr(scene, owner), identifier, value)
        except (AttributeError, TypeError, ValueError):
            pass


def references_to(target, generated):
    """Bindings whose intended replacement/removal cannot be guessed."""
    refs = []
    animation = target.animation_data
    # NLA strips are authored animation exactly as an active action is, and an
    # object animated through them has `animation_data.action` set to None. A
    # check that reads only the active action therefore reports *no bindings*
    # for a clip-authored object: `synchronize` raises no issue, the removal
    # below takes the animation with it, and `build()` returns `{"ok": true}`.
    # Same class as the donor-library leak -- a success that is not one,
    # discovered later and somewhere else.
    if animation and (animation.action or animation.drivers
                      or any(track.strips for track in animation.nla_tracks)):
        refs.append("animation on " + target.name)
    for obj in bpy.context.scene.objects:
        if obj.parent == target and obj.name not in generated:
            refs.append("parent of " + obj.name)
        for owner in [obj, *list(obj.pose.bones if obj.pose else ())]:
            for constraint in owner.constraints:
                for prop in constraint.bl_rna.properties:
                    if prop.type == "POINTER" and getattr(constraint, prop.identifier, None) == target:
                        refs.append(f"constraint {obj.name}/{constraint.name}")
        for modifier in obj.modifiers:
            for prop in modifier.bl_rna.properties:
                if prop.type == "POINTER" and getattr(modifier, prop.identifier, None) == target:
                    refs.append(f"modifier {obj.name}/{modifier.name}")
        data = obj.animation_data
        for curve in data.drivers if data else ():
            for variable in curve.driver.variables:
                if any(t.id == target for t in variable.targets):
                    refs.append(f"driver {obj.name}/{curve.data_path}")
    return sorted(set(refs))


def _source_axis(joint):
    """A baseline joint's URDF axis; `axis` held it before joints became local Z."""
    return joint.get("source_axis", joint["axis"])


#: The one channel a joint Empty moves on, as in `dycraft_joints`.
JOINT_CHANNEL = {"revolute": ("rotation_euler", 2), "prismatic": ("location", 2)}


def _animated(obj):
    data = obj.animation_data
    return bool(data and (data.action or data.drivers
                          or any(track.strips for track in data.nla_tracks)))


def synchronize(job):
    old, new = job["old_baseline"], job["new_baseline"]
    refresh = job.get("refresh") or {}
    materials = refresh.get("materials", [])
    if not isinstance(materials, list) or not all(isinstance(v, str) for v in materials):
        raise ValueError("refresh.materials must be a list of identities, instances, or '*'")
    known = set(new["identities"]) | {v.split("/")[0] for v in new["identities"]} | {"*"}
    if set(materials) - known:
        raise ValueError("unknown material refresh targets: " + str(sorted(set(materials) - known)))
    before, after = dict(old["generated"]), dict(new["generated"])
    retained_lights = {}
    if not refresh.get("lighting"):
        retained_lights = {n: s for n, s in before.items() if s["type"] == "LIGHT"}
        before = {n: s for n, s in before.items() if s["type"] != "LIGHT"}
        after = {n: s for n, s in after.items() if s["type"] != "LIGHT"}
    scene = bpy.context.scene
    existing = {name: bpy.data.objects.get(name) for name in before}
    missing = [name for name, obj in existing.items() if obj is None]
    if missing:
        raise ValueError("generated objects are missing; repair before updating: " + str(missing))
    # Joints imported before they became driver-free turn about their URDF axis
    # in axis-angle. The donor's origins are now rotated onto local Z, so
    # merging them under such a joint would turn it about the wrong axis.
    # There is deliberately no converter: refuse rather than corrupt.
    axis_angle = sorted(name for name in (joint["object"] for joint in old["joints"].values())
                        if bpy.data.objects.get(name) is not None
                        and bpy.data.objects[name].rotation_mode == "AXIS_ANGLE"
                        and "q" in bpy.data.objects[name])
    if axis_angle:
        raise ValueError("this project's joints predate driver-free joints (axis-angle "
                         "Empties driven by `q`) and cannot take a source update; re-import "
                         "the project: " + ", ".join(axis_angle[:5]))
    collisions = [name for name in set(after) - set(before) if bpy.data.objects.get(name)]
    if collisions:
        raise ValueError("new generated identities conflict with native objects: " + str(collisions))
    issues = []
    for name in sorted(set(before) - set(after)):
        refs = references_to(existing[name], before)
        if refs:
            issues.append({"id": "removed:" + name, "object": name,
                           "reason": "removed generated object has authored bindings", "bindings": refs})
    for name in sorted(set(before) & set(after)):
        obj = existing[name]
        if before[name]["type"] != after[name]["type"]:
            raise ValueError("generated object type changed: " + name)
        topology_changed = before[name].get("topology") != after[name].get("topology")
        geometry_changed = before[name].get("geometry") != after[name].get("geometry")
        bound = list(obj.vertex_groups.keys()) if obj.type == "MESH" else []
        modifiers = binding_modifiers(obj) if obj.type == "MESH" else []
        if (obj.type == "MESH" and ((geometry_changed and obj.data.shape_keys)
                or (topology_changed and (bound or modifiers)))):
            reason = ("source geometry changed under shape keys; transfer or recreate the affected keys"
                      if geometry_changed and obj.data.shape_keys else
                      "topology changed under vertex groups or index-bound modifiers; "
                      "rebuild affected bindings")
            issue = {"id": "geometry:" + name, "object": name, "reason": reason}
            # Which binding is at risk, so the author is not left to guess
            # between a vertex group and a modifier they cannot see from here.
            if not (geometry_changed and obj.data.shape_keys):
                if bound:
                    issue["vertex_groups"] = bound
                if modifiers:
                    issue["modifiers"] = modifiers
            issues.append(issue)
        if before[name]["parent"] != after[name]["parent"]:
            issues.append({"id": "parent:" + name, "object": name,
                           "reason": "mechanism parent changed; inspect dependent channels and constraints"})
    for name, joint in new["joints"].items():
        previous = old["joints"].get(name)
        if previous and (previous["type"] != joint["type"]
                         or _source_axis(previous) != _source_axis(joint)):
            issues.append({"id": "joint:" + name, "object": joint["object"],
                           "reason": "joint type or axis changed; the joint's existing keys "
                                     "need explicit review"})

    # Load by requested names: Blender may suffix names while old IDs coexist.
    with bpy.data.libraries.load(job["donor"], link=False) as (src, dst):
        names = tuple(after)
        # Blender replaces entries in the assigned list with datablocks.
        # Keep the semantic names in a separate immutable sequence.
        dst.objects = list(names)
        world_names = [new["world"]] if refresh.get("lighting") and new.get("world") else []
        dst.worlds = world_names
    donor = dict(zip(names, dst.objects))
    if any(obj is None for obj in donor.values()):
        raise ValueError("donor is missing generated objects")
    targets = dict(existing)
    added = set(after) - set(before)
    for name in sorted(added):
        obj = donor[name]
        obj.name = name
        scene.collection.objects.link(obj)
        targets[name] = obj

    joint_objects = {joint["object"] for joint in new["joints"].values()}
    for name, spec in after.items():
        obj, incoming = targets[name], donor[name]
        previous = before.get(name)
        identity = incoming.get("dycraft_identity", "")
        refresh_material = "*" in materials or identity in materials or identity.split("/")[0] in materials
        if name not in added:
            if obj.type == "MESH":
                if previous.get("geometry") != spec.get("geometry"):
                    saved_materials = list(obj.data.materials)
                    obj.data = incoming.data.copy()
                    if not refresh_material:
                        _set_materials(obj.data, saved_materials)
                elif refresh_material:
                    # Copy on write: other authored users may intentionally share this mesh.
                    obj.data = obj.data.copy()
                    _set_materials(obj.data, list(incoming.data.materials))
            elif obj.type == "LIGHT" and refresh.get("lighting"):
                obj.data = incoming.data.copy()
        obj.parent = targets.get(spec["parent"])
        # Unchanged generated bases leave authored transforms untouched. A
        # source edit of an origin/rest frame is an explicit placement update.
        #
        # Never a joint's own Empty, whose transform is the joint's
        # coordinate and the author's: writing the donor's rest matrix over
        # it would discard the author's pose. A changed start reaches an
        # unanimated joint below, through its channel alone.
        if name in joint_objects and name not in added:
            continue
        if previous is None or previous["matrix"] != spec["matrix"] or (obj.type == "LIGHT" and refresh.get("lighting")):
            obj.matrix_basis = incoming.matrix_basis.copy()

    for name, joint in new["joints"].items():
        obj, incoming = targets[joint["object"]], donor[joint["object"]]
        for key in ("axis", "limits", "joint_type"):
            obj[key] = incoming[key]
        # A changed start (a definition's `default`, an instance's
        # `initial_joints`) reaches a joint the author has not touched: no keys,
        # strips, or drivers, so it would otherwise stay at the old start. A
        # joint the author animated or posed keeps its value, and only the
        # channel is written, so it cannot turn the other way. A baseline from
        # before `default` was recorded cannot say, so it is left alone.
        previous = old["joints"].get(name) or {}
        if (joint["object"] not in added and "default" in previous
                and previous["default"] != joint.get("default") and not _animated(obj)):
            path, index = JOINT_CHANNEL[obj["joint_type"]]
            if abs(getattr(obj, path)[index] - previous["default"]) < 1e-6:
                getattr(obj, path)[index] = joint["default"]

    for name in set(before) - set(after):
        bpy.data.objects.remove(existing[name], do_unlink=True)
    # Library loading can bring dependencies beyond the explicitly listed IDs.
    # Never delete an existing authored object; clean only unused donor copies.
    for name, obj in donor.items():
        if name not in added:
            bpy.data.objects.remove(obj, do_unlink=True)
    if refresh.get("lighting") and dst.worlds:
        scene.world = dst.worlds[0]
    if refresh.get("lighting"):
        # The block's color management, film, and light-transport settings,
        # which live on the scene rather than on any object the donor lends.
        apply_scene_settings(scene, new.get("lighting_settings") or {})
    scene["dycraft_scene_ref"] = new["scene_ref"]
    scene["dycraft_native_collision_allowances"] = True
    # Appending an object drags its dependencies in with it, and the ones this
    # merge does not adopt stay behind as unused `.001` copies. Each carries a
    # `library_weak_reference` to the donor under `jobs/<id>/donor/`, which
    # `bpy.utils.blend_paths()` reports and `audit_dependencies` then rejects
    # as a path outside the workspace — so `build` succeeded and the next
    # check, compile, or storyboard failed, naming a directory the author never
    # chose. Observed as eight orphaned materials from one source update.
    # `pack_all` cannot absorb these, so they have to go where they are made.
    for library in list(bpy.data.libraries):
        if library.users == 0:
            bpy.data.libraries.remove(library)
    for collection in (bpy.data.materials, bpy.data.meshes, bpy.data.node_groups,
                       bpy.data.actions, bpy.data.images, bpy.data.armatures):
        for item in list(collection):
            if item.users == 0 and getattr(item, "library_weak_reference", None):
                collection.remove(item)
    bpy.context.view_layer.update()
    return {"issues": issues, "added": sorted(added), "removed": sorted(set(before) - set(after)),
            "updated": sorted(set(before) & set(after)), "refresh": refresh,
            "generated": {**after, **retained_lights}}
