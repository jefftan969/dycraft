"""In Blender: each shader part's flat color, roughness, and metallic, read from its node graph with no mesh built.

usage: <blender-python> _viewer_flat_child.py <jobs.json> <result.json>

`jobs.json` is `[{"ir_id", "job"}]`, each job packed by `viewer_bake.pack_bake_job`. The result is
`{ir_id: {part name: {"color": [r, g, b, a] linear | "image" | None, "roughness", "metallic"}}}`, with the same
reading the GLB export gives (`export/_blender/flat_material.py`).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "export" / "_blender"))

import _blender_child as C  # noqa: E402
from flat_material import flat_color, flat_value  # noqa: E402

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = {}
for entry in jobs:
    job = entry["job"]
    C._reset_scene(job)
    parts = {}
    for i, spec in enumerate(job["frames"][0]["parts"]):
        packed = spec.get("material") or {}
        if not packed.get("shader"):
            continue
        name = spec.get("ir_part") or spec.get("name")
        try:
            mat = C._build_material(f"flat_{i}_mat", packed, spec.get("alpha"),
                                    has_uv=bool(spec.get("uvs_npy") or spec.get("uv_from_obj")))
        except Exception as exc:                        # noqa: BLE001 -- one graph must not lose the others
            print(f"[flat] {name}: {exc!r}", file=sys.stderr)
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        color = flat_color(bsdf.inputs["Base Color"])
        parts[name] = {"color": list(color) if isinstance(color, tuple) else color,
                       "roughness": float(flat_value(bsdf.inputs["Roughness"])),
                       "metallic": float(flat_value(bsdf.inputs["Metallic"]))}
    result[entry["ir_id"]] = parts
Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
print("RESULT", len(result))
