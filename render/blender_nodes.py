"""The shader-node vocabulary of the pinned Blender, as data.

`render/blender_nodes.json` records every shader node class this repo's Blender
can create, with its socket names in socket ORDER and its enum properties. It is
generated rather than written for its consumer:

* `sdk/_shader` validates an authored `nodes.new("...")` against it at author
  time, in the CAD interpreter, which has no `bpy`. A node that does not exist
  becomes a compile-report error naming the nearest matches instead of an
  `AttributeError` inside a render child minutes later.

**Sockets are stored as a LIST, not a set, and the order is the contract.**
`ShaderNodeMix` exposes ten inputs named `Factor, Factor, A, B, A, B, A, B, A, B`
— one pair per `data_type` — so `mix.inputs["A"]` resolves to index 2, the FLOAT
variant, and is silently wrong under `data_type='RGBA'` (which wants index 6).
A name-keyed dict would erase exactly the fact a caller needs.

Regenerate after a Blender upgrade::

    $(cat .blender_python) render/blender_nodes.py

`tests/test_blender_nodes_ref.py` re-derives it and fails if the checked-in file
and the installed `bpy` disagree, so an upgrade cannot land silently.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

#: The checked-in artifact. Beside this module so both interpreters find it by
#: the same route and neither needs a run directory.
NODES_JSON = Path(__file__).resolve().parent / "blender_nodes.json"

_CACHE: dict | None = None


# ---------------------------------------------------------------------------
# Reading (no bpy — this half runs in the CAD interpreter)
# ---------------------------------------------------------------------------

def load() -> dict:
    """The reference, parsed and cached.

    Returns `{}` when the artifact is missing rather than raising: the author-
    time node check is a convenience that turns a render-time failure into a
    compile-time one, and losing it must not make an object unbuildable.
    """
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(NODES_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _CACHE = {}
    return _CACHE


def node_types() -> frozenset:
    """Every shader node class name the pinned Blender can create."""
    return frozenset((load().get("nodes") or {}).keys())


def sockets(node_type: str) -> dict:
    """``{"inputs": [name, ...], "outputs": [name, ...]}`` in socket order."""
    entry = (load().get("nodes") or {}).get(node_type) or {}
    return {"inputs": list(entry.get("inputs") or ()),
            "outputs": list(entry.get("outputs") or ())}


def enums(node_type: str) -> dict:
    """``{property: [identifier, ...]}`` for the node's own enum properties."""
    entry = (load().get("nodes") or {}).get(node_type) or {}
    return dict(entry.get("enums") or {})


def blender_version() -> str:
    return str(load().get("blender_version") or "")


def suggest(name: str, *, limit: int = 3) -> list:
    """Node class names closest to `name`, for a repair hint.

    Case-insensitive on the cheap: an agent that writes `shaderNodeTexNoise`
    should still be pointed at the real spelling rather than told nothing looks
    like it.
    """
    known = sorted(node_types())
    if not known:
        return []
    hits = difflib.get_close_matches(name, known, n=limit, cutoff=0.6)
    if hits:
        return hits
    low = name.lower()
    return [k for k in known if k.lower() == low][:limit]


# ---------------------------------------------------------------------------
# Generation (needs bpy — this half runs in the Blender interpreter)
# ---------------------------------------------------------------------------

def generate() -> dict:
    """Introspect the installed `bpy` and return the reference dict.

    Every candidate is actually CREATED in a real shader node tree rather than
    read off `bpy.types`, because `bpy.types` also carries abstract bases and
    node classes that belong to other tree types; `nodes.new` is the only
    honest test of "can an authored shader ask for this".
    """
    import bpy  # noqa: PLC0415 — only available in the Blender interpreter

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mat = bpy.data.materials.new("__nodes_ref__")
    mat.use_nodes = True
    tree = mat.node_tree

    # Properties every node inherits. Excluded so each entry carries only the
    # enums that are actually about the node's behavior.
    inherited = set()
    for base in ("Node", "NodeInternal", "ShaderNode"):
        rna = getattr(bpy.types, base, None)
        if rna is not None:
            inherited.update(rna.bl_rna.properties.keys())

    out: dict = {}
    for attr in dir(bpy.types):
        if not attr.startswith("ShaderNode"):
            continue
        try:
            node = tree.nodes.new(attr)
        except (RuntimeError, TypeError):
            continue        # abstract base, or not creatable in a shader tree
        try:
            entry = {
                "inputs": [s.name for s in node.inputs],
                "outputs": [s.name for s in node.outputs],
            }
            node_enums = {}
            for key, prop in node.bl_rna.properties.items():
                if key in inherited or prop.is_readonly:
                    continue
                if getattr(prop, "type", None) != "ENUM":
                    continue
                node_enums[key] = [e.identifier for e in prop.enum_items]
            if node_enums:
                entry["enums"] = dict(sorted(node_enums.items()))
            out[attr] = entry
        finally:
            tree.nodes.remove(node)

    return {
        "blender_version": bpy.app.version_string.split()[0],
        "generated_by": "render/blender_nodes.py",
        "nodes": dict(sorted(out.items())),
    }


def dumps(ref: dict) -> str:
    """Canonical text form, so the checked-in file has a stable diff."""
    return json.dumps(ref, indent=1, sort_keys=True) + "\n"


def main() -> int:
    ref = generate()
    NODES_JSON.write_text(dumps(ref), encoding="utf-8")
    n = len(ref["nodes"])
    print(f"[blender-nodes] wrote {NODES_JSON} "
          f"({n} shader node types, Blender {ref['blender_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
