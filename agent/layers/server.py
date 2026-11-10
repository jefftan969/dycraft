"""The object, scene, and animation layers' harness daemon: binds the layer's session and serves it.

Thin per-agent wrapper around the generic `agent.server.serve()`. Started
by the launcher before it starts the agent CLI:

    python -m dycraft.agent.layers.server serve <run_dir>

Reads `$DYCRAFT_LAYER`: object and scene get a `LayerSession` with their LayerConfig and payload, animation a
`FilmSession`; the session's ops are then served.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..agent_runner import user_prompt_from_env
from ..server import serve
from .layer import (LayerConfig, object_layer_config, scene_layer_config)
from .scene_context import SceneCatalog
from .session import Session, SessionConfig

LAYER_NAMES = ("object", "scene", "animation")


def _build_layer(name: str, run_dir: Path) -> tuple[LayerConfig, Any]:
    """Build (LayerConfig, layer_payload) for the object or scene layer."""
    if name == "object":
        return object_layer_config(), None
    if name == "scene":
        # No catalog to load: a scene run authors the objects it places, so
        # its catalog is a product of each build rather than an input.
        return scene_layer_config(), SceneCatalog(run_dir)
    raise SystemExit(f"daemon: unknown layer: {name!r}")


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "serve":
        print("usage: python -m dycraft.agent.layers.server "
              "serve <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[2]).resolve()
    layer_name = os.environ.get("DYCRAFT_LAYER")
    if layer_name not in LAYER_NAMES:
        print(f"daemon: $DYCRAFT_LAYER must be one of {list(LAYER_NAMES)} "
              f"(got {layer_name!r})", file=sys.stderr)
        return 2
    # meta.json was already written by the launcher (cli.py::write_run_meta);
    # finalize_meta at shutdown refreshes outcome / finished_at and the layer's fields.
    if layer_name == "animation":
        from .film import FilmSession
        session = FilmSession(run_dir, user_prompt=user_prompt_from_env())
    else:
        layer_cfg, layer_payload = _build_layer(layer_name, run_dir)
        session = Session(SessionConfig(
            run_dir=run_dir,
            layer=layer_cfg,
            layer_payload=layer_payload,
            user_prompt=user_prompt_from_env(),
            initial_meta=None,
        ))
    with session:
        serve(session, run_dir, start_fields={"layer": layer_name})
    return 0


if __name__ == "__main__":
    # Inherited from `dycraft.cli` when it spawns this daemon; raised here
    # too for a daemon started any other way. See `_oom`.
    from ... import _oom
    _oom.raise_self(_oom.SESSION)
    raise SystemExit(main(sys.argv))
