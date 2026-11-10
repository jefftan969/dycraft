"""The object, scene, and animation layers' harness: session, tools, compile.

    from dycraft.agent.layers import Session, SessionConfig
    from dycraft.agent.layers import materialize_catalog
"""

from __future__ import annotations

import json
from pathlib import Path

from .scene_context import iter_catalog_ir_paths, stage_object_assets
from .session import Session, SessionConfig


def materialize_catalog(src: Path, dst: Path, *,
                        source_root: Path | None = None) -> int:
    """Copy ObjectIRs from `src` (a JSON file or a directory of JSON files)
    into `dst/<id>.json`. Returns the number of entries written.

    The `<id>.json` layout is the scene agent's catalog/ contract — the
    scene SDK enumerates it via `sorted(objects.keys())` and looks up each
    ObjectIR by its `id` field. Lives here so a single source of truth owns
    the catalog filesystem schema; the CLI launcher just calls it.

    `source_root` overrides where an ObjectIR's relative asset paths
    (`assets/brep/<digest>.brep`) resolve from. It defaults to the catalog's
    own location — `src` for a directory, `src.parent` for a single file —
    which is right whenever the map and its assets share a tree. An *animation*
    bundle is the case that needs the override: it keeps the two in sibling
    trees (`design_state/catalog.json` beside `design_state/source/assets/`),
    so the root is off by exactly one level.
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0

    def _write(ir: dict, fallback: str, default_root: Path) -> None:
        nonlocal n
        h = ir.get("id") or fallback
        if not h:
            raise SystemExit(
                "FATAL: ObjectIR missing 'id' field and no fallback id"
        )
        (dst / f"{h}.json").write_text(json.dumps(ir, indent=2))
        stage_object_assets(ir, source_root=source_root or default_root,
                            catalog_root=dst)
        n += 1

    if src.is_dir():
        for f in iter_catalog_ir_paths(src):
            ir = json.loads(f.read_text())
            _write(ir, f.stem, src)
    else:
        data = json.loads(src.read_text())
        if not isinstance(data, dict):
            raise SystemExit(
                f"FATAL: {src}: expected an object {{hash: ObjectIR, ...}}"
            )
        for k, ir in data.items():
            _write(ir, k, src.parent)
    if n == 0:
        raise SystemExit(f"FATAL: catalog at {src} is empty")
    return n


__all__ = [
    "Session",
    "SessionConfig",
    "materialize_catalog",
]
