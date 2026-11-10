"""Blender-interpreter child: joint structure of armature rigs, read-only.

    <blender_python> _rig_child.py request.json

A character animated by an armature has its parts bone-parented, and its
joints live in the rig, not in a URDF joint tree or an object parent chain.
Two bones are **connected** (their parts meet at a joint) when

1. one is the other's Blender parent, or
2. one's head coincides with the other's tail in the first animated pose (a chain built
   without parenting: upper arm -> forearm -> hand), or
3. failing both, the bone's head — its pivot — lies in or on a part of another
   bone in that pose (a shoulder in the torso, a finger base in the palm);
   the single nearest such bone is taken; failing that, another bone's free
   end (a tail no bone continues from) lying in or on this bone's parts (a
   neck over the top of the torso).

Joints are read in the animation's first pose. Separately, every pair of parts of one rig (bone-parented or skinned)
that already interpenetrates in the rig's **rest pose** (the character as modeled: layered feathers, a collar over
a neck) is listed with its deepest vertex intrusion (`rest_overlaps`); the scorer treats it as a construction
overlap.

Only rig data and part geometry are read, never names. The file is opened with `use_scripts=False`, the rest pose is
set in memory only, and the file is never saved and is hash-checked before and after. Writes
`{"rule", "object_bone": {object: "armature:bone"}, "connected": [[a, b, reason]],
"rest_overlaps": [[part, part, depth_mm]]}`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

RIG_VERSION = "rig-joints-v2"
COINCIDE_M = 1e-3       # head on tail: within 1 mm ...
COINCIDE_REL = 0.02     # ... or 2 % of the shorter bone
PIVOT_TOL_M = 0.01      # pivot within 1 cm of a part's surface (or inside it)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _mesh(o, deps):
    ev = o.evaluated_get(deps)
    mesh = ev.to_mesh()
    try:
        if mesh is None or not len(mesh.polygons):
            return None
        mw = ev.matrix_world
        return [mw @ v.co for v in mesh.vertices], [tuple(p.vertices) for p in mesh.polygons]
    finally:
        ev.to_mesh_clear()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    blend = Path(request["blend"])
    before = sha256(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False, load_ui=False)
    scene = bpy.context.scene
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    object_bone, connected = {}, []
    parts, rig_parts = {}, {}
    for o in scene.objects:
        if o.type != "MESH":
            continue
        if o.parent is not None and o.parent.type == "ARMATURE" and o.parent_type == "BONE" and o.parent_bone:
            key = f"{o.parent.name}:{o.parent_bone}"
            object_bone[o.name] = key
            parts.setdefault(key, []).append(o)
            rig_parts.setdefault(o.parent.name, set()).add(o.name)
        for m in o.modifiers:  # skinned parts deform with the rig too
            if m.type == "ARMATURE" and getattr(m, "object", None) is not None:
                rig_parts.setdefault(m.object.name, set()).add(o.name)

    # Joints, read in the animation's own first pose: bone-parented parts were
    # attached in some pose, and only the animated pose puts them where they
    # are rendered (the rest pose can misplace them).
    scene.frame_set(scene.frame_start)
    deps = bpy.context.evaluated_depsgraph_get()
    trees = {}
    for key, objs in parts.items():
        for o in objs:
            m = _mesh(o, deps)
            if m:
                trees.setdefault(key, []).append(BVHTree.FromPolygons(*m))

    def distance(point, key):
        """0 inside a part of `key`, else the distance to its nearest surface."""
        best = float("inf")
        for tree in trees.get(key, ()):
            hit = tree.find_nearest(point)
            if hit[0] is None:
                continue
            location, normal, _, dist = hit
            if (point - location).dot(normal) < 0:
                return 0.0
            best = min(best, dist)
        return best

    for arm in armatures:
        mw = arm.matrix_world
        bones = list(arm.data.bones)
        head = {b.name: mw @ arm.pose.bones[b.name].head for b in bones}
        tail = {b.name: mw @ arm.pose.bones[b.name].tail for b in bones}
        key = lambda b: f"{arm.name}:{b.name}"
        # a free end: a tail no other bone continues from (the top of a torso,
        # a fingertip), unlike an elbow, where the forearm starts
        free = {o.name for o in bones
                if not any(c is not o and (head[c.name] - tail[o.name]).length <= COINCIDE_M for c in bones)}
        pairs = set()
        for b in bones:
            if b.parent is not None:
                pairs.add((key(b.parent), key(b), "bone parent"))
                continue
            chain = [o for o in bones if o is not b and
                     (head[b.name] - tail[o.name]).length <= max(COINCIDE_M, COINCIDE_REL * min(b.length, o.length))]
            if chain:
                for o in chain:
                    pairs.add((key(o), key(b), "head on tail"))
                continue
            near = [(distance(head[b.name], key(o)), key(o)) for o in bones if o is not b and key(o) in trees]
            near = [x for x in near if x[0] <= PIVOT_TOL_M]
            if near:
                d, other = min(near)
                pairs.add((other, key(b), f"pivot in part ({d * 1000:.1f} mm)"))
                continue
            # the other way round: another bone's tail sits in this bone's parts
            # (a head whose own neck part covers the pivot, over the torso's end)
            socket = [(distance(tail[o.name], key(b)), key(o)) for o in bones if o is not b and o.name in free]
            socket = [x for x in socket if x[0] <= PIVOT_TOL_M]
            if socket:
                d, other = min(socket)
                pairs.add((other, key(b), f"tail in part ({d * 1000:.1f} mm)"))
        connected.extend(sorted([list(p) for p in pairs]))

    # Built-in overlaps: two parts of one rig that already interpenetrate in
    # the rig's rest pose — the character as modeled (layered feathers, a
    # collar over a neck) — with the deepest vertex intrusion either way.
    for arm in armatures:
        arm.data.pose_position = "REST"  # in memory only; the file is never saved
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    rays = (Vector((0, 0, 1)), Vector((1, 0, 0)), Vector((0, 1, 0)))

    def inside(tree, point):
        votes = 0
        for d in rays:
            n, o = 0, point.copy()
            for _ in range(64):
                hit = tree.ray_cast(o, d)
                if hit[0] is None:
                    break
                n, o = n + 1, hit[0] + d * 1e-5
            votes += n % 2
        return votes >= 2

    rest_overlaps = []
    for arm_name, names in sorted(rig_parts.items()):
        meshes = {}
        for n in sorted(names):
            m = _mesh(bpy.data.objects[n], deps)
            if m:
                meshes[n] = (m[0], BVHTree.FromPolygons(*m))
        names = sorted(meshes)
        for i, a in enumerate(names):
            va, ta = meshes[a]
            for b in names[i + 1:]:
                vb, tb = meshes[b]
                if not ta.overlap(tb):
                    continue
                depth = max([tb.find_nearest(p)[3] for p in va if inside(tb, p)] +
                            [ta.find_nearest(p)[3] for p in vb if inside(ta, p)] + [0.0])
                rest_overlaps.append([a, b, round(depth * 1000, 2)])

    if sha256(blend) != before:
        raise SystemExit(f"source .blend changed while reading the rig: {blend}")
    Path(request["output"]).write_text(json.dumps(
        {"rule": RIG_VERSION, "blend": str(blend), "blend_sha256": before,
         "armatures": [a.name for a in armatures], "object_bone": object_bone, "connected": connected,
         "rest_overlaps": rest_overlaps},
        indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
    # The output is written and closed. Leave without Blender's teardown, which can deadlock after opening a scene
    # with caches or threads (large blends, Alembic caches).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
