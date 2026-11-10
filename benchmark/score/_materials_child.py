"""Blender-interpreter child: classify a `.blend`'s meshes as liquid or effect by material.

    <blender_python> _materials_child.py request.json

For inputs whose evaluation is read as delivered (DyCraft bundles), which carries no material data. The `.blend` is
opened with `use_scripts=False`, never saved, and hash-checked before and after. Writes
`{"rule": ..., "objects": {name: {"class", "reason"}}}`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bpy
    from _fluid_rule import RULE_VERSION, object_class
    blend = Path(request["blend"])
    before = sha256(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False, load_ui=False)
    objects = {}
    for obj in bpy.context.scene.objects:
        if obj.type not in ("MESH", "CURVE", "SURFACE", "META"):
            continue
        cls, reason = object_class(obj)
        if cls:
            objects[obj.name] = {"class": cls, "reason": reason}
    if sha256(blend) != before:
        raise SystemExit(f"source .blend changed while reading materials: {blend}")
    Path(request["output"]).write_text(json.dumps(
        {"rule": RULE_VERSION, "blend": str(blend), "blend_sha256": before, "objects": objects},
        indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
    # The output is written and closed. Leave without Blender's teardown, which can deadlock after opening a scene
    # with caches or threads (large blends, Alembic caches).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
