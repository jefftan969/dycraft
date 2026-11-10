"""The one color each shader part comes to, for the viewer's default view.

A `Material(shader=...)` computes its surface in a Blender node graph that the viewer cannot run, so by default the
viewer draws such a part in one color, with one roughness and one metallic: the values the GLB export gives it
(`export/_blender/flat_material.py`), so the viewer and an exported GLB agree. The full appearance is opt-in (the
viewer's "Baked materials", `viewer_bake`).

One Blender process per run reads them from every shader part's graph. It builds no meshes: `flat_color` walks the
graph that feeds the Principled BSDF's Base Color. The values are cached per object by its IR hash, beside the bake's
cache (`cache_root()/flat/<ir_id>_<FLAT_VERSION>.json`), so every later view, and every run that shares the object,
reads them.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

from . import RenderRequest, RenderUnavailable
from . import viewer_bake as VB

#: Bumped when the recipe changes, which re-reads every object's colors.
FLAT_VERSION = "vf1"

_CHILD = Path(__file__).resolve().parent / "_viewer_flat_child.py"

#: One Blender start plus the graphs; a run with hundreds of shader parts takes seconds.
TIMEOUT_S = float(os.environ.get("DYCRAFT_VIEWER_FLAT_TIMEOUT_S", "600"))


def flat_path(ir_id: str) -> Path:
    return VB.cache_root() / "flat" / f"{ir_id}_{FLAT_VERSION}.json"


def read(ir_id: str) -> dict | None:
    """`{part name: {"color": [r, g, b] (sRGB, 0-255), "roughness", "metallic"}}` for one object, or None when its
    colors have not been read yet."""
    try:
        data = json.loads(flat_path(ir_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("parts") if isinstance(data, dict) else None


def has_shader(ir: dict) -> bool:
    return any((p.get("material") or {}).get("shader") for p in ir.get("parts") or ())


def _srgb(linear: float) -> int:
    """A linear Blender channel as the sRGB byte the IR's `color` holds (`_blender_child._srgb_to_linear`, inverted)."""
    c = max(0.0, min(1.0, float(linear)))
    s = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return int(round(s * 255))


def compute(entries) -> dict[str, dict]:
    """Read and cache the colors of every entry (`viewer/derived.DerivedEntry`) in one Blender process.

    Returns `{ir_id: parts}` for the entries it read. A part whose Base Color comes from an image is left out, as
    is one whose graph failed to build, so the viewer keeps drawing what the IR says for it.
    """
    from ._blender_env import blender_child_env, blender_python
    from .. import _oom

    entries = list(entries)
    if not entries:
        return {}
    interp = blender_python()
    if not interp:
        raise RenderUnavailable("no Blender interpreter: set $DYCRAFT_BLENDER_PYTHON or write one to .blender_python "
                                "(python -m dycraft.install blender)")
    with tempfile.TemporaryDirectory(prefix="dycraft-flat-") as tmp:
        scratch = Path(tmp)
        jobs = []
        for i, entry in enumerate(entries):
            out = scratch / f"o{i}"
            request = RenderRequest(layer="object", ir=entry.ir, catalog=None, target="all", out_dir=out / "renders",
                                    step=0, asset_base_dir=entry.asset_base_dir, backend="blender")
            jobs.append({"ir_id": entry.ir_id, "job": VB.pack_bake_job(request, out)})
        jobs_path, result_path = scratch / "jobs.json", scratch / "result.json"
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        try:
            proc = subprocess.run(_oom.tiered([interp, str(_CHILD), str(jobs_path), str(result_path)], _oom.WORKER),
                                  capture_output=True, text=True, timeout=TIMEOUT_S, env=blender_child_env())
        except subprocess.TimeoutExpired:
            raise RenderUnavailable(f"reading shader colors timed out after {TIMEOUT_S:.0f} s") from None
        if not result_path.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
            raise RenderUnavailable("the shader color pass died before it could report:\n  " + "\n  ".join(tail))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    read_back: dict[str, dict] = {}
    for ir_id, parts in result.items():
        kept = {name: {"color": [_srgb(c) for c in v["color"][:3]], "roughness": float(v["roughness"]),
                       "metallic": float(v["metallic"])}
                for name, v in parts.items() if isinstance(v.get("color"), list)}
        path = flat_path(ir_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps({"ir_id": ir_id, "version": FLAT_VERSION, "parts": kept}), encoding="utf-8")
        os.replace(tmp_path, path)
        read_back[ir_id] = kept
    return read_back


def flat_ir(ir: dict, parts: dict | None) -> dict:
    """A copy of an ObjectIR whose shader parts carry their read color, roughness, and metallic, so the URDF draws
    them in it. `ir` unchanged when there is nothing to fill in. Viewer-only: nothing canonical sees the copy."""
    if not parts:
        return ir
    out = copy.deepcopy(ir)
    for part in out.get("parts") or ():
        material = part.get("material") or {}
        found = parts.get(part.get("name")) if material.get("shader") else None
        if found:
            material.update(color=found["color"], roughness=found["roughness"], metallic=found["metallic"])
    return out
