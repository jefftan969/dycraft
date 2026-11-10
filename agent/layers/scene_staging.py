"""Staging an animation bundle's frozen scene, and SceneIR validation.

`materialize_scene` stages an animation bundle's frozen design
(`design_state/`) at a run root as `scene.json` + `catalog/`, so the viewer and
tools can read it. `validate_scene_ir` re-hashes a SceneIR before it is staged.
The catalog half is delegated to `materialize_catalog`, the one implementation
of the on-disk catalog contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...sdk.ir import SCENE_SCHEMA_VERSIONS, content_hash
from .scene_context import CATALOG_DIR


#: The frozen SceneIR, staged read-only into the run dir. Named `scene.json`
#: rather than `scene.py`'s neighbor `ir.json`, because `ir.json` is where the
#: run's *own* output (the AnimationIR) lands on finish — two different IRs in
#: one run dir need two different names.
SCENE_JSON = "scene.json"


# ---------------------------------------------------------------------------
# SceneIR validation
# ---------------------------------------------------------------------------


def validate_scene_ir(scene_ir, *, source: str) -> str:
    """Re-hash a SceneIR and return its `id`, or raise SystemExit.

    The scene-layer catalog is validated the same way (`validate_catalog`), and
    for the same reason: a content-hashed IR whose stated `id` no longer matches
    its bytes has been edited after freezing, and every downstream cache key,
    `scene_ref` check and anti-tamper guarantee is keyed on that hash.

    `SystemExit` rather than a `BuildError`: this is a staging/operator fault
    about an *input*, not something the agent can repair by editing
    `animation.py`. Same posture as the daemon's `empty catalog at …`.
    """
    if not isinstance(scene_ir, dict):
        raise SystemExit(
            f"FATAL: {source}: expected a SceneIR object, got "
            f"{type(scene_ir).__name__}")
    # Both scene schema versions animate. The version boundary is about how
    # lighting is *authored* (v1 symbolic keys vs a v2 `@lighting` block) and
    # says nothing about instances or joints, which is all this layer moves —
    # so refusing v1 here would strand every already-committed scene.
    schema = scene_ir.get("schema")
    # v1 legacy lighting, v2 the `@lighting` vocabulary, v3 v2 plus
    # per-instance `initial_joints`. See `sdk/ir.py`.
    accepted = SCENE_SCHEMA_VERSIONS
    if schema not in accepted:
        raise SystemExit(
            f"FATAL: {source}: schema tag is {schema!r}, expected one of "
            f"{list(accepted)}. --layer=animation animates a committed "
            f"SceneIR; point --scene at a scene run dir.")
    stated = scene_ir.get("id")
    if not isinstance(stated, str) or not stated:
        raise SystemExit(
            f"FATAL: {source}: SceneIR missing its content-hash 'id'")
    without_id = {k: v for k, v in scene_ir.items() if k != "id"}
    try:
        recomputed = content_hash(without_id)
    except (ValueError, TypeError) as e:
        raise SystemExit(
            f"FATAL: {source}: SceneIR is not JSON-canonicalizable ({e})")
    if recomputed != stated:
        raise SystemExit(
            f"FATAL: {source}: SceneIR content-hash mismatch — stated "
            f"{stated}, recomputed {recomputed}. The scene was edited after it "
            f"was frozen; re-run the scene layer rather than hand-patching "
            f"ir.json.")
    return stated


#: An animation bundle's frozen design, written by
#: `animation/build.py::compile_design`: the run authors its scene
#: mid-run and freezes it here rather than at the run root.
DESIGN_STATE = "design_state"


def materialize_scene(bundle: Path, run_dir: Path, *,
                              force: bool = False) -> dict:
    """Stage an animation bundle's frozen scene at `run_dir`. Returns meta fields.

    Writes `scene.json` and `catalog/` from `<bundle>/design_state/`, where an
    animation run freezes the scene it authors mid-run. Every viewer path that
    draws an animation run's geometry reads those two names and only those two.

    Nothing is written under `bundle/`, so the animation manifest
    (`animation/blender/bundle.py::make_manifest`, which hashes `bundle/**`)
    stays valid.

    Three behaviors worth stating, because they are what makes this safe to
    re-run over a corpus:

      * **Idempotent.** An already-staged run whose `scene.json` carries this
        scene's id is left alone — no re-hashing the catalog, no recopying
        ~86 MB of meshes.
      * **Refuses a mismatch.** A `scene.json` naming a *different* scene is a
        real `--layer animation` run or a stale staging; overwriting it would
        silently change what the run draws. `force=True` is the override.
      * **`scene.json` last.** `viewer/animation.py::load_scene_ir` finding it
        is the viewer's gate for both geometry paths, so writing it after the
        catalog makes its presence mean "the catalog completed". A crash
        part-way leaves a run that fails exactly as it did before, rather than
        one that opens onto a half-empty room.
    """
    # Local imports for the same reason the rest of this module uses them:
    # `agent.layers.__init__` imports this module's siblings, and the
    # animation package is the harness half. `agent → animation` is the
    # direction `agent/layers/server.py` already takes.
    from . import materialize_catalog
    from ...animation.context import workspace_context

    bundle = Path(bundle)
    design_state = bundle / DESIGN_STATE
    scene_path = design_state / SCENE_JSON
    if not scene_path.is_file():
        raise SystemExit(
            f"FATAL: {bundle} has no {DESIGN_STATE}/{SCENE_JSON}; only an "
            f"animation run's bundle carries its frozen scene.")

    # One definition of the design_state layout, and it is not this one:
    # `workspace_context` owns where the SceneIR, the catalog map and the asset
    # root sit relative to each other.
    ctx = workspace_context(bundle)
    scene_ref = validate_scene_ir(ctx.scene_ir, source=str(scene_path))

    staged_path = run_dir / SCENE_JSON
    if staged_path.is_file() and not force:
        staged = json.loads(staged_path.read_text())
        if staged.get("id") == scene_ref:
            return {"scene_ref": scene_ref, "n_entries": 0,
                    "source": str(scene_path), "skipped": True}
        raise SystemExit(
            f"FATAL: {staged_path} already stages a different scene "
            f"({staged.get('id')}, not {scene_ref}). Re-run with force=True "
            f"only if you are sure this run should draw the bundle's scene.")

    run_dir.mkdir(parents=True, exist_ok=True)
    n_entries = materialize_catalog(design_state / "catalog.json",
                                    run_dir / CATALOG_DIR,
                                    source_root=ctx.asset_base_dir)
    staged_path.write_text(json.dumps(ctx.scene_ir, indent=2, sort_keys=True))
    return {
        "scene_ref": scene_ref,
        "n_entries": n_entries,
        "source": str(scene_path),
        "skipped": False,
    }


__all__ = [
    "SCENE_JSON",
    "DESIGN_STATE",
    "materialize_scene",
    "validate_scene_ir",
]
