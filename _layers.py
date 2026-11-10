"""Layer names, and the older names recorded runs may carry.

Stdlib-only, so a reader of recorded runs (the viewer, benchmark staging, transcript tools) can map a name without
importing `dycraft.agent`.

`--layer` accepts only the current names. `canonical_layer` maps the older names a recorded run may carry,
`unified` and `unified_animation`, to `scene` and `animation`, so readers handle every run under one name.
"""
from __future__ import annotations

#: The layers `--layer` accepts.
LAYERS = ("object", "scene", "animation", "plain")

#: Older layer names found in recorded runs, mapped to their current names.
LAYER_ALIASES = {"unified": "scene", "unified_animation": "animation"}


def canonical_layer(name: str | None) -> str | None:
    """`name` under its current spelling; unknown names and None pass through."""
    return LAYER_ALIASES.get(name, name) if name is not None else None


__all__ = ["LAYERS", "LAYER_ALIASES", "canonical_layer"]
