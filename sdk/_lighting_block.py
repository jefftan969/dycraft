"""`@lighting` — the scene's lighting, World, and photographic settings, as the
Blender Python that builds them.

The agent authors an ordinary helper in its `scene.py`::

    @lighting
    def rig(L):
        key = L.new_light("key", "AREA")
        key.data.energy = 40.0
        key.data.size = 1.2
        key.location = (1.2, -0.4, 2.3)
        key.rotation_euler = (math.radians(-35), 0.0, math.radians(20))

        sky = L.world_nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(12.0)
        sky.sun_intensity = 0.4
        bg = L.world_nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 0.6
        L.world_links.new(sky.outputs["Color"], bg.inputs["Color"])
        L.world_links.new(bg.outputs["Background"],
                          L.world_output.inputs["Surface"])

        L.view.view_transform = "AgX"          # set BEFORE look
        L.view.look = "AgX - Medium High Contrast"
        L.view.exposure = -0.7

    scene.set_lighting(rig)

The body authors Blender's own lights, World nodes and view settings, so what
it sets is what renders: no unit conversion, and no exposure it cannot see.

**The boundary.** The body is handed one object, `L`, and never `bpy`. Through
`L` it reaches lights, the World node tree, view settings, and managed images —
the whole lighting and photographic domain — and nothing else: no geometry, no
animation, no `bpy.ops`, no filesystem, no network, no process. `check_ast`
independently rejects imports, dunder attributes and `exec`/`eval`/`getattr`,
so the body cannot climb out of the façade it is given.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from ._errors import InvalidPrimitive
from ._shader import capture_source_block

#: As it must be spelled in the authored source.
DECORATOR_NAME = "lighting"

#: Pure numeric names a lighting body may use without binding them, spelled as
#: bare functions exactly as `agent/cad/sandbox._MATH_BUILTINS` does.
#:
#: Allowed here and deliberately still absent from `@shader`: a shader's
#: arithmetic belongs in Math nodes, where the graph can vary it, but a light's
#: rotation is a constant the author has to write in radians, so
#: `radians(34.0)` is the honest spelling. The render child injects exactly
#: these into the body's namespace — no `math` module, nothing importable.
#: Kept identical to that set by `tests/test_lighting_block.py`, because the
#: render child has no dycraft import and must hardcode the same names.
EXTRA_ALLOWED_FREE = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot",
    "sqrt", "exp", "log", "radians", "degrees", "floor", "ceil",
    "pi", "tau", "inf",
})

#: The single façade parameter. One object rather than several because the
#: lighting domain is not naturally three sockets the way a material's node
#: tree is, and a single namespace can grow a capability without changing every
#: authored signature in the tree.
PARAMS = ("L",)


@dataclass(frozen=True)
class LightingBlock:
    """One scene's authored lighting, carried as source text.

    Inline in the SceneIR for the same reason `Shader` is: it has no path that
    can go stale, it enters the IR's content hash directly, and the scene layer
    has no asset session to stage it through.
    """

    name: str
    body: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise InvalidPrimitive(
                f"lighting.name must be a Python identifier, got {self.name!r}",
                evidence={"param": "lighting.name", "value": repr(self.name),
                          "constraint": "identifier"})
        if not isinstance(self.body, str) or not self.body.strip():
            raise InvalidPrimitive(
                "lighting.body must be non-empty source text",
                evidence={"param": "lighting.body",
                          "constraint": "non-empty str"})
        # Parsed here, in the interpreter without bpy, so a syntax error is an
        # author-time compile report rather than a render traceback later. It
        # proves the text parses only — a wrong property name still needs
        # Blender to discover.
        try:
            builtins.compile(self.body, f"<lighting:{self.name}>", "exec")
        except SyntaxError as e:
            raise InvalidPrimitive(
                f"lighting {self.name!r} does not parse: {e.msg}",
                evidence={"param": "lighting.body", "detail": e.msg,
                          "lineno": e.lineno or 0},
                suggested_repairs=["fix the syntax error in the lighting body"],
            ) from None

    def to_ir(self) -> dict:
        return {"name": self.name, "body": self.body}


def lighting(fn):
    """Turn an authored lighting function into a `LightingBlock` value.

    Returns the value, not the function — the same discipline `@shader` uses:
    what the IR stores has to be serialisable and hashable, and the name is
    bound to something the CAD interpreter cannot call by accident (`L` does
    not exist there and never will; it is supplied by the render child).
    """
    name, body = capture_source_block(
        fn, decorator_name=DECORATOR_NAME, params=PARAMS,
        # The World tree is built from the same `ShaderNode*` classes a
        # material uses, so the version-drift check is just as useful here.
        check_node_types=True,
        extra_allowed=EXTRA_ALLOWED_FREE,
    )
    return LightingBlock(name=name, body=body)


__all__ = ["lighting", "LightingBlock",
           "DECORATOR_NAME", "PARAMS", "EXTRA_ALLOWED_FREE"]
