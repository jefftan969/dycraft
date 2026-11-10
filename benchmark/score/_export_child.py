"""Blender-interpreter child: export any `.blend` into the `dycraft.evaluation` format.

    <blender_python> _export_child.py request.json

The export is the animation layer's own, `worker.evaluate({})`, so a bare agent's scene and a DyCraft bundle are
measured from one definition of evaluated geometry. This wrapper adds only what a foreign file needs first, all in
memory:

* the file is opened with `use_scripts=False`, so no code it carries runs, and **never saved**; its sha256 is
  checked before and after;
* FONT objects (captions) and anything parented under a camera (screen-space overlays) are removed with their
  descendants: neither is physical geometry, and Blender 4.x evaluates text through an instance, which the exporter
  refuses;
* CURVE / SURFACE / META objects whose *local* evaluated geometry never changes are converted to meshes in place
  (object, animation, and parenting kept); one whose shape is animated cannot be converted without freezing it, so it
  is removed and reported;
* fluid domains and particle emitters are classed `fluid`, and liquid and effect meshes are classed by their
  materials (`_fluid_rule.py`);
* visibility the exporter does not record (collection or view-layer exclusion, `visible_camera` off) is captured per
  frame into `bodies.json`, with each object's type and parent (for assemblies and link adjacency).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

EXPORT_VERSION = "physical-export-v2"
GEOMETRY_TYPES = ("MESH", "CURVE", "SURFACE", "FONT", "META")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*args):
    print("[penetration-export]", *args, file=sys.stderr, flush=True)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    repo = Path(request["repo"])
    blend = Path(request["blend"])
    out = Path(request["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    os.environ["DYCRAFT_BLENDER_WORKSPACE_ROOT"] = str(Path(request.get("workspace_root") or blend.parent))
    os.environ["DYCRAFT_BLENDER_OUTPUT_ROOT"] = str(out)
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(repo / "animation" / "blender"), str(repo / "verify")]
    import bpy

    started = time.perf_counter()
    before = sha256(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False, load_ui=False)
    scene = bpy.context.scene
    warnings = []

    spec = importlib.util.spec_from_file_location("worker", repo / "animation" / "blender" / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    try:
        worker.audit_dependencies()
    except ValueError as error:
        # Missing textures or sounds do not change geometry: recorded, not fatal.
        warnings.append(f"audit_dependencies: {error}")

    objects = list(scene.objects)

    def ancestors(obj):
        p = obj.parent
        while p is not None:
            yield p
            p = p.parent

    remove, reasons = set(), {}
    for obj in objects:
        if obj.type == "FONT":
            remove.add(obj.name); reasons[obj.name] = "text"
        elif obj.type in GEOMETRY_TYPES and any(p.type == "CAMERA" for p in ancestors(obj)):
            remove.add(obj.name); reasons[obj.name] = "camera_overlay"
    for obj in objects:  # descendants of removed geometry go with it
        if obj.name not in remove and obj.type in GEOMETRY_TYPES:
            for p in ancestors(obj):
                if p.name in remove:
                    remove.add(obj.name); reasons[obj.name] = reasons[p.name]
                    break

    # Convert shape-static curves/surfaces/metaballs; drop shape-animated ones.
    frame_start, frame_end = scene.frame_start, scene.frame_end
    samples = sorted({frame_start, frame_end, *(frame_start + (frame_end - frame_start) * k // 8 for k in range(9))})
    to_convert = [o for o in objects if o.type in ("CURVE", "SURFACE", "META") and o.name not in remove]
    static_shape, animated_shape = [], []
    if to_convert:
        import numpy as np
        signatures = {o.name: set() for o in to_convert}
        for f in samples:
            scene.frame_set(f)
            deps = bpy.context.evaluated_depsgraph_get()
            for o in to_convert:
                ev = o.evaluated_get(deps)
                mesh = ev.to_mesh()
                if mesh is None:
                    signatures[o.name].add("none")
                    continue
                try:
                    v = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                    mesh.vertices.foreach_get("co", v)
                    signatures[o.name].add(hashlib.sha256(v.tobytes()).hexdigest() + f":{len(mesh.polygons)}")
                finally:
                    ev.to_mesh_clear()
        scene.frame_set(frame_start)
        for o in to_convert:
            sig = signatures[o.name]
            if sig == {"none"}:
                remove.add(o.name); reasons[o.name] = "no_geometry"
            elif len(sig) == 1:
                static_shape.append(o)
            else:
                animated_shape.append(o.name)
                remove.add(o.name); reasons[o.name] = "animated_curve_shape"
                warnings.append(f"{o.name}: {o.type} with animated local shape cannot be converted without freezing it; excluded")

    from _fluid_rule import object_class
    classes = {}
    for obj in objects:
        if obj.name in remove:
            continue
        material_cls, _ = object_class(obj) if obj.type in ("MESH", "CURVE", "SURFACE", "META") else (None, None)
        if material_cls:
            classes[obj.name] = material_cls
        for m in obj.modifiers:
            if m.type == "FLUID" and getattr(m, "fluid_type", None) in ("DOMAIN", "FLOW"):
                classes[obj.name] = "fluid"
            elif m.type in ("OCEAN",):
                classes[obj.name] = "fluid"
        if obj.particle_systems:
            classes.setdefault(obj.name, "particles")

    parents = {o.name: (o.parent.name if o.parent else None) for o in objects if o.name not in remove}
    types = {o.name: o.type for o in objects if o.name not in remove}
    removed = [{"name": n, "reason": reasons[n]} for n in sorted(remove)]
    for name in sorted(remove):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    converted = []
    for o in static_shape:
        name = o.name
        original = o.type
        for sel in bpy.context.view_layer.objects:
            sel.select_set(False)
        hidden_viewport = o.hide_viewport
        o.hide_viewport = False
        try:
            o.hide_set(False)
        except RuntimeError:
            pass
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        with bpy.context.temp_override(active_object=o, object=o, selected_objects=[o],
                                       selected_editable_objects=[o]):
            bpy.ops.object.convert(target="MESH")
        o = bpy.data.objects[name]
        o.hide_viewport = hidden_viewport
        if o.type != "MESH":
            raise RuntimeError(f"conversion of {name} ({original}) to MESH failed")
        types[name] = "MESH"
        converted.append({"name": name, "from": original})

    deps = bpy.context.evaluated_depsgraph_get()
    leftover = sorted({(i.parent.name if i.parent else "?") for i in deps.object_instances if i.is_instance})
    if leftover:
        raise SystemExit("unrealized geometry instances remain after cleanup (instancers: "
                         + ", ".join(leftover[:20]) + "); realize them before evaluation")

    # Visibility the exporter's `hidden = obj.hide_render` does not capture.
    def static_hidden(obj):
        cols = list(obj.users_collection)
        if cols and all(c.hide_render for c in cols):
            return True
        layer = bpy.context.view_layer.layer_collection

        def excluded(lc, target):
            if lc.collection == target:
                return lc.exclude
            for child in lc.children:
                r = excluded(child, target)
                if r is not None:
                    return r or lc.exclude
            return None
        return bool(cols) and all(excluded(layer, c) for c in cols)

    geometry = [o for o in scene.objects if o.type in GEOMETRY_TYPES]
    fixed = {o.name for o in geometry if static_hidden(o)}
    hidden_frames = {}

    def capture(sc, *_):
        i = sc.frame_current - sc.frame_start
        for o in geometry:
            if o.name in fixed or not o.visible_camera:
                spans = hidden_frames.setdefault(o.name, [])
                if spans and spans[-1][1] == i - 1:
                    spans[-1][1] = i
                elif not spans or spans[-1][1] != i:
                    spans.append([i, i])
    bpy.app.handlers.frame_change_post.append(capture)
    t_prep = time.perf_counter()
    info = worker.evaluate({})
    bpy.app.handlers.frame_change_post.remove(capture)
    t_eval = time.perf_counter()

    after = sha256(blend)
    if after != before:
        raise SystemExit(f"source .blend changed during export: {blend}")
    bodies = {"export_version": EXPORT_VERSION,
              "objects": {name: ({"class": classes[name]} if name in classes else {}) for name in types},
              "parents": parents, "types": types, "hidden_frames": hidden_frames,
              "removed": removed, "converted": converted}
    (out / "bodies.json").write_text(json.dumps(bodies, indent=1, sort_keys=True))
    manifest = {"export_version": EXPORT_VERSION, "blend": str(blend), "blend_sha256": before,
                "blend_bytes": blend.stat().st_size, "blender_version": bpy.app.version_string,
                "frames": len(info["frames"]), "objects_exported": len(info["collision_objects"]),
                "removed": len(removed), "removed_by_reason": _count(r["reason"] for r in removed),
                "converted": converted, "animated_curves_excluded": animated_shape,
                "warnings": warnings,
                "timing": {"prepare_s": t_prep - started, "evaluate_s": t_eval - t_prep,
                           "total_s": time.perf_counter() - started}}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    log(f"exported {manifest['frames']} frames, {manifest['objects_exported']} objects in "
        f"{manifest['timing']['total_s']:.1f}s")


def _count(items):
    out = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    main()
    # The output is written and closed. Leave without Blender's teardown, which can deadlock after opening a scene
    # with caches or threads (large blends, Alembic caches).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
