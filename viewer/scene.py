"""Scene-layer artifacts: the per-run catalog, and a URDF the viewer can serve.

An **object** run is self-contained: its `ir.json` names every part, and each part's mesh path
(`assets/meshes/<digest>.obj`) is relative to the run root, which is what `/api/runs/{id}/files/` is the base URL
for, so `object_ir_to_urdf(ir)` is the whole story.

A **scene** run's `ir.json` holds only *instances*, each a content hash into `<run_dir>/catalog/`, and the catalog
objects' asset paths resolve against `catalog/`. So serving it takes two more steps: load the catalog through
`iter_catalog_ir_paths` (`catalog/manifest.json` is the allowlist of which `<hash>.json` files are IR), and rebase
every asset path onto the run root (`assets/…` → `catalog/assets/…`) so the one `files/` base URL resolves it. An
**animation** run takes the same path with its staged `scene.json` as the IR (`viewer/animation.py` supplies the
per-frame poses on top).

The rebasing happens in memory because the viewer never writes into a run it displays. `scene_ir_to_urdf`
replicates each catalog object's joint tree under an `<instance>__` prefix, so joint sliders work for articulated
objects placed in a scene.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .animation import is_animation_ir
from ..sdk._material import texture_maps_from_ir as _texture_maps_from_ir

#: The one kind this file names, and only to decide whether what is left
#: after rebasing is still a texture at all.
BASE_COLOR = "base_color"

# Mirrors `agent.layers.scene_context.CATALOG_DIR`. Duplicated as a
# literal rather than imported at module scope so importing the viewer never
# pulls in the harness; the functions below import lazily.
CATALOG_DIR = "catalog"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def ir_kind(ir: Any) -> str | None:
    """`"object"`, `"scene"`, `"animation"`, or None for none of those.

    Keyed off the IR's own shape rather than `meta.json`'s `layer`, so one
    rule reads what a run actually committed, whatever layer name its
    `meta.json` records.

    `animation` is the one kind that carries no geometry of its own — its
    scene is the `scene.json` staged next to it — so it is matched on the
    schema tag rather than on a payload field, and callers that want geometry
    must go through `viewer/animation.py` rather than through this IR.
    """
    if not isinstance(ir, dict):
        return None
    if isinstance(ir.get("instances"), list):
        return "scene"
    if isinstance(ir.get("parts"), list):
        return "object"
    if is_animation_ir(ir):
        return "animation"
    return None


def _rebase_asset(rel: Any, catalog_dir: Path, run_dir: Path) -> str | None:
    """A catalog-relative asset path as a run-root-relative one, or None.

    None means "the viewer cannot serve this file": an absolute path (an
    asset living outside the run bundle) or a path that resolves to nothing. Callers fall back rather than emit a `<mesh>` the
    browser would 404 on.
    """
    text = str(rel)
    path = Path(text)
    if path.is_absolute():
        return None
    if (catalog_dir / path).is_file():
        return (Path(CATALOG_DIR) / path).as_posix()
    # Some staging layouts land assets at the run root instead; accept those
    # unchanged rather than declaring the object unservable.
    if (run_dir / path).is_file():
        return path.as_posix()
    return None


def _rebase_parts(parts, catalog_dir: Path, run_dir: Path) -> tuple[list, bool]:
    """Rebase every asset-bearing part; also report whether all of them worked.

    Returns `(parts, complete)`. `complete` is False when any asset could not
    be rebased, so a caller can fall back wholesale rather than draw an object
    with holes in it.
    """
    from ..sdk._primitives import render_asset_key

    out: list = []
    complete = True
    for part in parts:
        if not isinstance(part, dict):
            continue
        part = _rebase_material(part, catalog_dir, run_dir)
        params = part.get("params")
        key = render_asset_key(part.get("primitive"))
        if key is None or not isinstance(params, dict) or key not in params:
            out.append(part)
            continue
        rebased = _rebase_asset(params[key], catalog_dir, run_dir)
        if rebased is None:
            complete = False
            out.append(part)
            continue
        out.append({**part, "params": {**params, key: rebased}})
    return out, complete


def _rebase_material(part: dict, catalog_dir: Path, run_dir: Path) -> dict:
    """Rebase a part's base-color texture, or drop it if it cannot be served.

    Deliberately NOT part of `complete`. A geometry asset that will not resolve
    means the object would be drawn with a hole in it, so the caller falls back
    wholesale — a texture that will not resolve means it is drawn in its
    material's own color, which is the fallback the color exists to be and
    exactly what the Open3D backend does when a decode fails
    (`render/open3d_renderer.py::_usable_texture`). Failing the whole object
    over it would make the viewer strictly worse than before textures existed.
    """
    material = part.get("material")
    if not isinstance(material, dict):
        return part
    texture = material.get("texture")
    maps = _texture_maps_from_ir(texture)
    if not maps:
        return part

    # Every map kind the material carries, whatever it is called, base color
    # included. Kinds are data; `texture_maps_from_ir` is the one place that
    # decides what counts as a map, so a new kind rebases without an edit here.
    for kind, sub in maps.items():
        rebased = _rebase_asset(sub["image"], catalog_dir, run_dir)
        if rebased is None:
            # Drop just this map. The scalar the material carries is exactly
            # the fallback it exists to be, so the surface still reads.
            texture = {k: v for k, v in texture.items() if k != kind}
        else:
            texture = {**texture, kind: {**sub, "image": rebased}}

    if BASE_COLOR not in texture:
        # No base color means no texture at all: the other maps describe a
        # surface whose color is missing, and binding roughness alone would
        # modulate a flat color the material never intended to be shiny. The
        # tile size goes with it — `Material` refuses a mapping with nothing to
        # lay out, and the IR should not say what the type would not.
        return {**part, "material": {k: v for k, v in material.items()
                                     if k not in ("texture", "mapping")}}
    return {**part, "material": {**material, "texture": texture}}


def load_catalog(run_dir: Path) -> dict[str, dict]:
    """`{object hash: ObjectIR}` from `<run_dir>/catalog/`, asset paths rebased.

    Keyed by the IR's own `id` so lookups match `inst["object_ref"]`, which is
    what `scene_query` resolves against. Returns `{}` when the run has no
    catalog — a scene run only snapshots one on a successful compile.
    """
    catalog_dir = run_dir / CATALOG_DIR
    if not catalog_dir.is_dir():
        return {}
    try:
        from ..agent.layers.scene_context import iter_catalog_ir_paths
        paths = iter_catalog_ir_paths(catalog_dir)
    except Exception:
        # A malformed or half-written manifest must not take the viewer down;
        # fall back to the raw-directory reading the loader itself uses when
        # no manifest is present.
        paths = sorted(p for p in catalog_dir.glob("*.json")
                       if p.name != "manifest.json")
    catalog: dict[str, dict] = {}
    for path in paths:
        obj_ir = _read_json(path)
        if ir_kind(obj_ir) != "object":
            continue
        parts, _ = _rebase_parts(obj_ir.get("parts") or (), catalog_dir, run_dir)
        catalog[str(obj_ir.get("id") or path.stem)] = {**obj_ir, "parts": parts}
    return catalog


def scene_urdf(ir: dict, catalog: Mapping[str, dict]) -> str:
    """URDF for a SceneIR, drawn from the run's own catalog.

    Instances whose `object_ref` is missing from the catalog are silently
    omitted by `scene_ir_to_urdf`; `catalog_coverage` is what tells a caller
    that happened, so it can say so instead of showing a half-empty room.
    """
    from ..render.urdf_export import scene_ir_to_urdf

    return scene_ir_to_urdf(ir, catalog)


def catalog_coverage(ir: dict, catalog: Mapping[str, dict]) -> tuple[int, int]:
    """`(resolvable instances, total instances)` for a SceneIR."""
    instances = [i for i in ir.get("instances") or () if isinstance(i, dict)]
    resolved = sum(1 for i in instances if i.get("object_ref") in catalog)
    return resolved, len(instances)


def instance_summaries(ir: dict, catalog: Mapping[str, dict]) -> list[dict]:
    """Per-instance rows for the inspector: what it is, and where it hangs.

    `parent` is the instance (or `"room"`) this one is attached to, which is
    the single fact that explains a scene's layout — "mug attached to table"
    says more than the mug's world coordinates do, and it is what a placement
    bug shows up in first.
    """
    rows = []
    for inst in ir.get("instances") or ():
        if not isinstance(inst, dict):
            continue
        placement = inst.get("placement") or {}
        obj_ir = catalog.get(inst.get("object_ref")) or {}
        rows.append({
            "name": inst.get("name"),
            "object_name": inst.get("object_name"),
            "object_ref": inst.get("object_ref"),
            "in_catalog": inst.get("object_ref") in catalog,
            "placement_kind": placement.get("kind"),
            "parent": placement.get("parent"),
            "parent_anchor": (placement.get("parent_anchor") or {}).get("name"),
            "child_anchor": (placement.get("child_anchor") or {}).get("name"),
            "n_parts": len(obj_ir.get("parts") or ()),
            "n_joints": len(obj_ir.get("joints") or ()),
        })
    return rows


def scene_parts_and_joints(ir: dict, catalog: Mapping[str, dict],
                           part_summary) -> tuple[list, list]:
    """Flattened `(parts, joints)` across every instance, `<instance>__`-prefixed.

    The prefix is not decoration: it is the exact link/joint naming
    `scene_ir_to_urdf` emits, so a row in the Parts panel names the same thing
    the 3D view and the joint sliders do.

    Each joint's `default` is replaced by **that instance's** committed value
    (`initial_joints`, falling back to the definition's own default), in the
    IR's internal units — the same units the URDF's joint frame uses, so the
    shell can seed its sliders from this row directly. It has to be per
    instance: the rows come from one shared ObjectIR, so two cabinets placed
    from one definition at different door angles would otherwise both draw at
    the definition's angle, and the viewer would disagree with every other
    consumer about the scene it is showing.
    """
    parts: list = []
    joints: list = []
    for inst in ir.get("instances") or ():
        if not isinstance(inst, dict):
            continue
        obj_ir = catalog.get(inst.get("object_ref"))
        if not isinstance(obj_ir, dict):
            continue
        initial = inst.get("initial_joints") or {}
        prefix = f"{inst.get('name')}__"
        for part in obj_ir.get("parts") or ():
            summary = part_summary(part)
            summary["name"] = f"{prefix}{summary.get('name')}"
            summary["instance"] = inst.get("name")
            parts.append(summary)
        for joint in obj_ir.get("joints") or ():
            if not isinstance(joint, dict):
                continue
            row = {
                **joint,
                "name": f"{prefix}{joint.get('name')}",
                "parent": (f"{prefix}{joint['parent']}"
                           if joint.get("parent") else joint.get("parent")),
                "child": (f"{prefix}{joint['child']}"
                          if joint.get("child") else joint.get("child")),
                "instance": inst.get("name"),
            }
            if joint.get("name") in initial:
                row["default"] = initial[joint["name"]]
            joints.append(row)
    return parts, joints
