"""`@shader` — a procedural material the agent writes as Blender node code.

The agent authors an ordinary helper function in its `object.py`::

    @shader
    def marble(nodes, links, bsdf):
        tc    = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 3.0
        ramp  = nodes.new("ShaderNodeValToRGB")
        links.new(tc.outputs["UV"],       noise.inputs["Vector"])
        links.new(noise.outputs["Fac"],   ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"],  bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.14

    MARBLE = Material(shader=marble)

The binding carries no appearance scalars beside the shader, and that is the
agent-facing rule rather than a schema one: `Material` still accepts them, but a
value in both places has no source of truth and leaves the graph unable to stand
alone. `alpha` and `emission` are the exceptions.

**Nothing defers it, because Python already does.** A `def` binds a function and
runs none of its body; this decorator receives the function object and never
calls it. `nodes`, `links` and `bsdf` do not exist in the CAD interpreter and
never will — they are supplied by `render/_blender_child.py`, in the process
that has `bpy`, at render time.

So the only work here is turning a function into something serializable, and
then checking as much of it as can be checked without Blender. That last part
is the whole reason this is a function rather than a source string: the body is
real AST, so `agent/cad/sandbox.check_ast` walks it under the ordinary
allowlist (no imports, no dunder attributes, no banned builtins) with only the
attribute-ASSIGNMENT rule relaxed, and the checks below add signature, free
variables and node types on top. Everything they catch becomes a compile report
the repair loop already knows how to surface, instead of a traceback inside a
render child several minutes later.

What still cannot be caught here: a socket name, an enum value, an argument
count. Those need Blender, and they come back as render warnings — see
`render/_blender_child._build_material`.

The authored source is read back through the process-active
`MeshAssetSession`, so a layer can only offer `shader` if it installs one.
The object and scene layers both do (`layer._object_build_overlay` and
`layer._scene_build_overlay`), and the animation layer builds its scene through
the scene layer's. `shader` is the whole material vocabulary.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from . import _mesh_assets
from ._errors import InvalidPrimitive
from ._material import Shader

#: The decorator's own name, as it must appear in the authored source. The AST
#: exemption in `agent/cad/sandbox` keys on this same spelling, so the
#: two are a contract: renaming the overlay entry without renaming it there
#: would make every shader body fail the attribute-assignment rule again.
DECORATOR_NAME = "shader"

#: The parameters a shader body is handed, in order. Fixed rather than inspected
#: so the authored signature is a contract both sides can state: the child calls
#: `fn(nodes, links, bsdf)` positionally.
PARAMS = ("nodes", "links", "bsdf")

#: The wider signature, which additionally hands over the `Material` datablock
#: so a body can reach material-level render properties and build an output
#: graph that is not Principled. Both are accepted: an existing three-argument
#: shader keeps working unchanged, and the child inspects which form it got.
PARAMS_WITH_MATERIAL = ("nodes", "links", "bsdf", "mat")

#: Every accepted signature, widest last.
ACCEPTED_PARAMS = (PARAMS, PARAMS_WITH_MATERIAL)

#: Names a body may reference without binding them. Deliberately just the
#: builtins Python's own `exec` provides anyway — there is no injection to keep
#: in sync with the render child, because there is nothing to inject.
#:
#: No `math`. A shader's arithmetic belongs in Math nodes, which is where the
#: graph can vary it; Python-side arithmetic can only produce a constant, and a
#: constant can be written as a literal.
_ALLOWED_FREE = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float",
    "int", "len", "list", "max", "min", "range", "reversed", "round",
    "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None",
})

#: Parsed authored sources, keyed on a digest of the text. One `object.py` holds
#: every shader an object binds, and the decorator fires once per shader.
_PARSE_CACHE: dict = {}


def _authored_source() -> str:
    """The source of the `object.py` this decorator was invoked from.

    **Taken from the session, not from disk.** `compile.build_ir_with_layer`
    sets `MeshAssetSession.source_text` to the exact bytes it is about to
    `exec`, immediately before executing them, so that is the text this shader
    was actually written in. Re-reading `<run_dir>/object.py` would be a second
    read of a mutable file — it makes the answer depend on the file's state
    afterwards rather than on the execution, which is precisely why the staging
    path stopped doing it.

    The disk read remains as a fallback for the one case the session documents
    as `source_text is None`: a test constructing a `MeshAssetSession` directly,
    outside a harness build. `object_worker` writes the file before executing
    it, so there the two agree by construction anyway.

    Unreadable is fatal here, unlike in `object_builder._source_text_for_staging`
    where the source is provenance and a placeholder will do. A shader IS its
    source: with no text there is nothing to send to Blender.
    """
    session = _mesh_assets.get_active_mesh_session()
    run_dir = getattr(session, "run_dir", None)
    staged = getattr(session, "source_text", None)
    if staged is not None:
        text = staged if isinstance(staged, str) else staged.decode("utf-8")
        if not text.strip():
            raise InvalidPrimitive(
                "@shader was handed an empty authored source",
                evidence={"param": DECORATOR_NAME},
            )
        return text
    if run_dir is None:
        raise InvalidPrimitive(
            "@shader reads back the source of the file it is written in, and "
            "that needs an asset session; there is none here",
            evidence={"param": DECORATOR_NAME, "constraint": "asset session"},
            suggested_repairs=[
                "author shader functions in object.py or scene.py, where the "
                "harness installs a session; elsewhere describe the surface "
                "with Material(color=...)",
            ],
        )
    # `source_text` above is the normal path on every layer, because
    # `compile.build_ir_with_layer` sets it from the bytes it is about to exec.
    # This re-read is the fallback for a session built directly, and it has to
    # know the layer's filename: the scene layer authors `scene.py`, and a
    # shader on a room surface is as legitimate as one on a part.
    candidates = [Path(run_dir, n) for n in ("object.py", "scene.py")]
    found = next((p for p in candidates if p.exists()), candidates[0])
    try:
        text = found.read_text(encoding="utf-8")
    except OSError as e:
        raise InvalidPrimitive(
            f"@shader could not read the authored source back: {e}",
            evidence={"param": DECORATOR_NAME, "detail": str(e)},
        ) from None
    if not text.strip():
        raise InvalidPrimitive(
            "@shader read the authored source back and it was empty",
            evidence={"param": DECORATOR_NAME},
        )
    return text


def _parsed(source: str):
    key = hashlib.blake2b(source.encode("utf-8"), digest_size=16).hexdigest()
    hit = _PARSE_CACHE.get(key)
    if hit is None:
        # A SyntaxError cannot happen: `check_ast` parsed this same text before
        # anything executed. Guarded anyway so a hand-driven call reports rather
        # than propagating a bare exception.
        try:
            hit = ast.parse(source)
        except SyntaxError as e:
            raise InvalidPrimitive(
                f"@shader could not parse the authored source: {e.msg}",
                evidence={"param": DECORATOR_NAME, "lineno": e.lineno or 0},
            ) from None
        _PARSE_CACHE[key] = hit
    return hit


def _is_decorated(node: ast.FunctionDef,
                  decorator_name: str = DECORATOR_NAME) -> bool:
    return any(isinstance(d, ast.Name) and d.id == decorator_name
               for d in node.decorator_list)


def _locate(tree: ast.Module, name: str,
            decorator_name: str = DECORATOR_NAME) -> ast.FunctionDef:
    """The module-level `@<decorator> def <name>` — exactly one of them."""
    found = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if not found:
        raise InvalidPrimitive(
            f"@{decorator_name} could not find a module-level "
            f"`def {name}` in the authored source",
            evidence={"param": name, "constraint": "module-level def"},
            suggested_repairs=[
                "define the shader with `@shader` on a plain `def` at the top "
                "level of the file, not inside another function",
            ],
        )
    if len(found) > 1:
        raise InvalidPrimitive(
            f"the authored source defines {len(found)} functions named "
            f"{name!r}; @{decorator_name} cannot tell which one is meant",
            evidence={"param": name, "constraint": "unique name",
                      "value": len(found)},
            suggested_repairs=["give each shader its own name"],
        )
    node = found[0]
    if not _is_decorated(node, decorator_name):
        raise InvalidPrimitive(
            f"{name!r} is not decorated with @{decorator_name}",
            evidence={"param": name,
                      "constraint": f"@{DECORATOR_NAME} decorator"},
            suggested_repairs=[
                f"write `@{DECORATOR_NAME}` on the line above `def {name}`; "
                f"the decorator is also what lets the body set node properties",
            ],
        )
    return node


def signature_params(node: ast.FunctionDef,
                     accepted=ACCEPTED_PARAMS,
                     decorator_name: str = DECORATOR_NAME) -> tuple:
    """The accepted parameter tuple this body declares. Raises otherwise.

    Returning *which* form was used is what lets the render child call a
    three-argument body and a four-argument body correctly without inspecting
    the function object it just `exec`d.
    """
    a = node.args
    names = tuple(arg.arg for arg in a.posonlyargs) + \
        tuple(arg.arg for arg in a.args)
    extras = bool(a.vararg or a.kwarg or a.kwonlyargs or a.defaults
                  or a.kw_defaults)
    if not extras:
        for candidate in accepted:
            if names == candidate:
                return candidate
    forms = " or ".join(f"({', '.join(c)})" for c in accepted)
    raise InvalidPrimitive(
        f"a @{decorator_name} function takes {forms}; "
        f"{node.name!r} takes ({', '.join(names) or 'nothing'})",
        evidence={"param": node.name, "value": list(names),
                  "constraint": forms},
        suggested_repairs=[
            f"def {node.name}({', '.join(accepted[0])}):  — the render child "
            f"calls it positionally",
        ],
    )


def _body_nodes(node: ast.FunctionDef):
    """Every AST node in the function's body, decorators excluded.

    `ast.walk` on a `FunctionDef` descends into `decorator_list` too, so
    `@shader` itself reads as a `Name` load — which would make every shader
    fail the free-variable check for referencing `shader`. Only the body
    travels to Blender, so only the body is what these checks are about.
    """
    for stmt in node.body:
        yield from ast.walk(stmt)


def _bound_names(node: ast.FunctionDef, params=None) -> set:
    """Every name the function binds anywhere inside itself.

    Whole-subtree rather than scope-aware, matching
    `sandbox._locally_bound_names` and for the same reason: the only question is
    whether a name is the body's own or has to come from outside it.
    """
    bound = set(params if params is not None else PARAMS)
    for sub in _body_nodes(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            bound.add(sub.id)
        elif isinstance(sub, ast.arg):
            bound.add(sub.arg)
        elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
            bound.add(sub.name)
    return bound


def _check_free_names(node: ast.FunctionDef, params=None,
                      decorator_name: str = DECORATOR_NAME,
                      extra_allowed=frozenset()) -> None:
    """Refuse a body that reads a name Blender will not have.

    The body travels to the render child on its own — a module constant the
    agent naturally reaches for (`noise.inputs["Scale"].default_value =
    MARBLE_SCALE`) is simply not there, and an unbound name inside a Cycles
    render is a warning nobody reads and a gray surface.

    Rejecting is the loud, cheap answer. The alternative — snapshotting the
    constants into a preamble — captures them when the DECORATOR runs, so a
    constant reassigned further down the file would silently ship its old value.
    That trade is available later; a stale number that renders is worse than an
    error that names the line.
    """
    bound = _bound_names(node, params)
    free = sorted({
        sub.id for sub in _body_nodes(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
        and sub.id not in bound
        and sub.id not in _ALLOWED_FREE and sub.id not in extra_allowed
    })
    if free:
        raise InvalidPrimitive(
            f"{decorator_name} {node.name!r} reads "
            f"{', '.join(repr(f) for f in free)}, "
            f"which will not exist when Blender runs it: a @{decorator_name} "
            f"body travels without the rest of its source file",
            evidence={"param": node.name, "value": free,
                      "constraint": "self-contained body"},
            suggested_repairs=[
                f"move the value inside {node.name}(), or write it as a literal",
                f"a @{decorator_name} body sees only "
                f"{', '.join(params or PARAMS)}, plain builtins"
                + (", and bare math names (e.g. radians, sin, pi)"
                   if extra_allowed else ""),
            ],
        )


def _check_node_types(node: ast.FunctionDef) -> None:
    """Refuse `nodes.new("...")` naming a class this Blender does not have.

    The reference is generated from the pinned `bpy`, so this catches the whole
    version-drift class at author time — `ShaderNodeTexMusgrave` was removed in
    Blender 4.1 and is exactly what an agent working from older material
    tutorials reaches for.

    Only literal arguments are checked. A computed node type is left alone
    rather than guessed at; Blender will report it.
    """
    from ..render import blender_nodes as _nodes_ref
    known = _nodes_ref.node_types()
    if not known:
        return              # reference unavailable; Blender still checks
    for sub in _body_nodes(node):
        # Any `<receiver>.new("SomeNode...")`, not just a receiver spelled
        # `nodes`. A lighting block builds the World tree through
        # `L.world_nodes`, and the version-drift class of error is identical
        # there; keying on the receiver's *name* would have silently skipped
        # every lighting body.
        if not (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "new"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)):
            continue
        requested = sub.args[0].value
        if not requested.endswith("Node") and "Node" not in requested:
            continue                    # not a node-class literal at all
        if requested in known:
            continue
        hints = _nodes_ref.suggest(requested)
        repairs = [f"did you mean {', '.join(hints)}?"] if hints else []
        repairs.append(
            f"see render/blender_nodes.json for every node type Blender "
            f"{_nodes_ref.blender_version()} can create")
        raise InvalidPrimitive(
            f"shader {node.name!r} asks for node type {requested!r}, which "
            f"Blender {_nodes_ref.blender_version()} does not have",
            refs=[],
            evidence={"param": node.name, "value": requested,
                      "lineno": getattr(sub, "lineno", 0) or 0,
                      "constraint": "known shader node type"},
            suggested_repairs=repairs,
        )


def shader(fn):
    """Turn an authored shader function into a `Shader` value.

    Returns a `Shader`, **not the function**, and that is the point: what
    `Material(shader=...)` stores has to be serializable and hashable, and the
    function object is neither. It also means the name is bound to a value the
    CAD interpreter cannot call by accident.
    """
    if not callable(fn) or not hasattr(fn, "__name__"):
        raise InvalidPrimitive(
            f"@{DECORATOR_NAME} decorates a function; got "
            f"{type(fn).__name__}",
            evidence={"param": DECORATOR_NAME,
                      "constraint": "function"},
        )
    name = fn.__name__
    source = _authored_source()
    node = _locate(_parsed(source), name)
    params = signature_params(node)
    _check_free_names(node, params)
    _check_node_types(node)

    body = ast.get_source_segment(source, node)
    if not body or not body.strip():
        raise InvalidPrimitive(
            f"@shader could not recover the source of {name!r}",
            evidence={"param": name},
        )
    # `get_source_segment` starts at `FunctionDef.lineno`, which for a
    # decorated function is the `def` line — so the decorator is excluded and
    # the text is a self-contained `def` the child can exec on its own.
    return Shader(name=name, body=body + "\n", params=tuple(params))


def capture_source_block(fn, *, decorator_name: str, params: tuple,
                         check_node_types: bool = True,
                         extra_allowed=frozenset()) -> tuple:
    """`(name, body)` for a decorated, self-contained source block.

    The shared half of `@shader` and `@lighting`: read back the source of the
    file the decorator was invoked from, locate the function, check that it
    declares the agreed signature and reads nothing it will not have when the
    render child `exec`s it alone, and return its text.

    Factored out rather than copied because the failure modes this catches —
    a free module constant, a node class removed in a later Blender — are
    identical for a lighting block and a material block, and a second copy
    would drift.
    """
    if not callable(fn) or not hasattr(fn, "__name__"):
        raise InvalidPrimitive(
            f"@{decorator_name} decorates a function; got "
            f"{type(fn).__name__}",
            evidence={"param": decorator_name, "constraint": "function"},
        )
    name = fn.__name__
    source = _authored_source_for(decorator_name)
    node = _locate_for(_parsed(source), name, decorator_name)
    signature_params(node, accepted=(params,), decorator_name=decorator_name)
    _check_free_names(node, params, decorator_name, extra_allowed)
    if check_node_types:
        _check_node_types(node)
    body = ast.get_source_segment(source, node)
    if not body or not body.strip():
        raise InvalidPrimitive(
            f"@{decorator_name} could not recover the source of {name!r}",
            evidence={"param": name},
        )
    return name, body + "\n"


def _authored_source_for(decorator_name: str) -> str:
    return _authored_source()


def _locate_for(tree, name: str, decorator_name: str):
    return _locate(tree, name, decorator_name=decorator_name)


__all__ = ["shader", "DECORATOR_NAME", "PARAMS", "PARAMS_WITH_MATERIAL",
           "ACCEPTED_PARAMS", "capture_source_block", "signature_params"]
