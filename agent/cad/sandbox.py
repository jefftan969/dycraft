"""Agent sandbox: AST allowlist, restricted globals, and read-only views.

Three mechanisms, and they guard different things:

  * the **AST allowlist** decides which syntax may run: no imports, no dunder
    access, no `with`. It governs *reach*.
  * the **restricted globals** decide what names resolve. With the guarded
    namespaces in `_b123d_namespace`, they keep modules, the filesystem, the
    process, and the network out of scope.
  * the **read-only views** decide what may be *written*. Everything the
    harness injects that outlives one execution (the cached IR, and in a scene
    probe the object handles and catalog) is a frozen view.

The views exist because mutation needs no distinctive syntax: an alias, a
parameter, or a method call (`objects.clear()`) gets past any rule on names, and
a probe's `ir['parts'].clear()` would corrupt the shared build cache. They live
in `sdk/_readonly.py`, which explains them (the SDK's `ObjectHandle` needs them,
and the SDK cannot import this package); `freeze` is re-exported here. The
subscript rule in `check_ast` is a lint for `Object['x'] = 1`, not the boundary.

Two surfaces:

- `build_globals(overlay=...)`: for executing the agent's source file. The
  layer's overlay (the SDK names and, where build123d is installed, the curated
  build123d namespace) plus a curated builtins subset.
- `build_query_globals(overlay=...)`: for evaluating a `probe` expression. The
  same builtins, the layer's query symbols, and `ir` (on the scene layer also
  `catalog` and `objects`, once a build exists).

`check_ast(source, mode="exec" | "eval")` walks the AST before exec or eval and
raises `SandboxViolation` on any disallowed construct.
"""

from __future__ import annotations

import ast
import math
from typing import Literal

from ...sdk._errors import SandboxViolation, SourceSyntaxError
from ...sdk._readonly import freeze  # noqa: F401 — re-exported for layers


# ---------------------------------------------------------------------------
# AST allowlist
# ---------------------------------------------------------------------------

_ALLOWED_NODES: frozenset = frozenset({
    "Module", "Expression", "Interactive",
    "FunctionDef", "ClassDef", "Return",
    "Assign", "AugAssign", "AnnAssign", "Expr", "Delete",
    "For", "While", "If", "Pass", "Break", "Continue",
    "Call", "Attribute",
    # Exceptions. The build fails geometrically all the time — OCCT refuses a
    # fillet radius and the whole source dies — and without `try` the author
    # has no way to fall back to a smaller one. `raise` comes with it so a
    # handler can re-raise. The exception *names* live in `_RESTRICTED_BUILTINS`
    # (Python's) and `_b123d_namespace.CURATED` (OCCT's); without those `except`
    # would be a NameError.
    "Try", "ExceptHandler", "Raise", "Assert",
    # No capability, only expressiveness. `Match` needs its whole pattern
    # vocabulary or it dies on the first `case`.
    "NamedExpr", "Yield", "YieldFrom", "Nonlocal",
    "Match", "match_case", "MatchValue", "MatchSingleton", "MatchSequence",
    "MatchMapping", "MatchClass", "MatchStar", "MatchAs", "MatchOr",
    "Name", "Constant", "Tuple", "List", "Dict", "Set",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Subscript", "Slice", "Starred",
    "Lambda", "arguments", "arg",
    "And", "Or", "Not", "Invert",
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "USub", "UAdd", "MatMult", "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn", "Is", "IsNot",
    "Load", "Store", "Del",
    "keyword",
    "FormattedValue", "JoinedStr",
})

# Still rejected, and why — every one of these is a decision, not an oversight:
#
#   Import / ImportFrom   the authoring surface is *injected* (`bd`, `math`,
#                         `np`, the SDK overlay). Keeping imports out is what
#                         makes `object.py` replayable by `fixed_source`
#                         outside its session.
#   With                  builder mode (`BuildPart`/`BuildSketch`/`BuildLine`)
#                         is the only thing that wants it, and the prompt and
#                         Skill both teach algebra mode. Adding `With` would
#                         reopen that choice — see `_b123d_namespace.CURATED`.
#   async family          an event loop inside a pooled, reused CAD worker
#                         that a wall-clock timeout cannot reason about.
#   Global                rebinding injected globals; no use case.
#
# `Try`/`Raise`/`Assert`/`Match`/walrus/generators are allowed: the agent
# authoring `object.py` also has `Bash(*)`, so syntax is not the boundary.
# What is load-bearing is name resolution (below, plus the guarded namespaces)
# and the read-only views.
_REJECTED_NAMES: frozenset = frozenset({
    "exec", "eval", "__import__", "compile",
    "open", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "memoryview", "breakpoint",
    "input", "help", "copyright", "credits", "license",
    "exit", "quit",
})


def _src_line(source: str, lineno: int) -> str:
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


# What to tell the author when a construct is refused. The prompt deliberately
# says nothing about the sandbox — the agent is meant to assume ordinary Python
# and learn the edges here — so these strings are the only documentation of the
# boundary that exists. Name the alternative, not just the refusal.
_NODE_REPAIRS = {
    #: One text for both nodes. It has to carry real weight: the prompts say
    #: nothing about the sandbox on purpose, so an agent assuming ordinary
    #: Python writes `import numpy as np` and learns the edge from here.
    #:
    #: It describes the surface by what the source DOES rather than by naming
    #: layers. Naming them was accurate right up until a fourth layer existed
    #: and the list silently excluded it; `bd`/`math`/`np`/`OCP` are in the
    #: object and scene overlays and in neither of the other two, and that
    #: distinction is about authoring geometry, not about the layer's name.
    "Import": "imports are unavailable — the authoring surface is injected instead. "
    "Where source authors geometry, that includes `bd` (build123d), `math`, "
    "`np` (numpy) and `OCP`, alongside the SDK builders; source that only composes "
    "gets a narrower set. Drop the "
    "import and use the injected name.",
    "ImportFrom": "imports are unavailable — the authoring surface is injected instead. "
    "Where source authors geometry, that includes `bd` (build123d), `math`, "
    "`np` (numpy) and `OCP`, alongside the SDK builders; source that only composes "
    "gets a narrower set. Drop the "
    "import and use the injected name.",
    "With": "`with` is unavailable. For build123d, use algebra mode "
            "(`+`, `-`, `&`) instead of BuildPart/BuildSketch/BuildLine. "
            "For other context managers, rewrite using supported calls.",
    "AsyncFunctionDef": "async is not available; write a plain `def`.",
    "AsyncFor": "async is not available; write a plain `for`.",
    "AsyncWith": "async is not available.",
    "Await": "async is not available.",
    "Global": "`global` is not available; return the value instead.",
}

_GENERIC_REPAIR = (
    "Remove the disallowed construct. Imports, with, global, and async syntax are "
    "refused; ordinary control flow, exceptions, assertions, classes, generators, "
    "comprehensions, and := are supported. Name and mutation restrictions still apply."
)


def _repair_for_node(node_type: str) -> str:
    return _NODE_REPAIRS.get(node_type, _GENERIC_REPAIR)


def _violation(source: str, node, summary: str, suggested: str) -> SandboxViolation:
    lineno = getattr(node, "lineno", 0) or 0
    col = getattr(node, "col_offset", 0) or 0
    return SandboxViolation(
        summary,
        refs=[],
        evidence={
            "node_type": type(node).__name__,
            "lineno": lineno,
            "col": col,
            "src_line": _src_line(source, lineno),
        },
        suggested_repairs=[suggested],
    )


def _subscript_base(node: ast.AST) -> ast.AST:
    """The innermost thing a subscript chain is ultimately indexing.

    `d[k]` -> `d`; `grid[i][j]` -> `grid`; `obj.parts[0]` -> the `obj.parts`
    Attribute; `f()[0]` -> the Call.
    """
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _locally_bound_names(tree: ast.AST) -> frozenset:
    """Names the source binds itself: assignment targets, loop variables,
    comprehension targets, function/class names, parameters, lambda args.

    Used to tell an agent's own container apart from an injected global. It is
    deliberately whole-module rather than scope-aware: a name bound anywhere in
    the file is one the source controls, and the only thing riding on the
    distinction is whether a subscript write is allowed — a shadowed injected
    name is rebound locally anyway, so the injected object is never the one
    being written.
    """
    bound: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return frozenset(bound)


#: The decorator that marks a Blender shader body. Must match
#: `sdk._shader.DECORATOR_NAME` — the two spellings are one contract, and
#: renaming either alone makes every shader body fail the rule below again.
SHADER_DECORATOR = "shader"

#: The decorator that marks a Blender lighting body. Must match
#: `sdk._lighting_block.DECORATOR_NAME`, on the same contract.
LIGHTING_DECORATOR = "lighting"

#: Every decorator whose body is Blender code carried to the render child as
#: text rather than executed here. All of them get the one relaxation below.
#:
#: A lighting body is attribute assignment almost end to end
#: (`light.data.energy`, `L.view.exposure`,
#: `bg.inputs["Color"].default_value`), there is no setter API on `L`, and
#: `setattr` is rejected, so without the relaxation every `@lighting` scene
#: would fail to compile, and a scene with no lighting block renders unlit.
#: One set, so a new decorator of this kind is wired in one place.
NODE_BODY_DECORATORS = frozenset({SHADER_DECORATOR, LIGHTING_DECORATOR})


def _node_body_nodes(tree: ast.AST) -> set:
    """`id()` of every node inside a body decorated with `NODE_BODY_DECORATORS`.

    Used to relax **one** rule — the attribute-assignment rejection — and
    nothing else. Authoring a Blender node graph or a lighting rig IS attribute
    assignment: `noise.inputs["Scale"].default_value = 3.0`,
    `ramp.color_ramp.elements[0].position = 0.42`, `mix.data_type = "RGBA"`,
    `key.data.energy = 40.0`, `L.view.exposure = -0.7` are all `ast.Attribute`
    targets, so the rule as written makes the entire Blender API unreachable.

    The exemption is narrow on purpose and matches what the rule is actually
    for. Its own advice is *"do not mutate attributes on SDK objects"*, and
    these bodies hold no SDK objects: `nodes`, `links`, `bsdf`, `mat`, and `L`
    are Blender-side handles that do not exist in this interpreter at all, and
    the body is never executed here — it is carried to the render child as
    text. Everything else still applies inside the body, which is the point of
    accepting a function rather than a source string: the allowlist keeps
    `import` out, the underscore rule keeps `__class__` chains out, and
    `_REJECTED_NAMES` keeps `exec`/`eval`/`getattr` out, so what reaches the
    child has been read.

    Keyed on `id()`, which is safe only because `check_ast` holds the parsed
    tree for the whole call: every node is reachable from it, so none can be
    collected and have its address reused under a different node.
    """
    exempt: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(isinstance(d, ast.Name) and d.id in NODE_BODY_DECORATORS
                   for d in node.decorator_list):
            continue
        for sub in ast.walk(node):
            exempt.add(id(sub))
    return exempt


def check_ast(source: str, *, mode: Literal["exec", "eval"] = "exec") -> ast.AST:
    """Parse and validate `source`. Returns the parsed AST on success;
    raises SandboxViolation on the first disallowed node.

    mode="eval" additionally requires the top-level form to be an Expression
    with a single expression body (used by `probe`).
    """
    try:
        tree = ast.parse(source, mode="exec" if mode == "exec" else "eval")
    except SyntaxError as e:
        # Not `SandboxViolation`: nothing was refused, the file does not
        # parse. See `sdk/_errors.SourceSyntaxError`.
        raise SourceSyntaxError(
            f"syntax error: {e.msg}",
            refs=[],
            evidence={
                "node_type": "SyntaxError",
                "lineno": e.lineno or 0,
                "col": e.offset or 0,
                "src_line": _src_line(source, e.lineno or 0),
            },
            suggested_repairs=["fix the syntax error"],
        )

    if mode == "eval" and not isinstance(tree, ast.Expression):
        raise SandboxViolation(
            "probe expression must be a single expression",
            refs=[],
            evidence={"node_type": type(tree).__name__, "lineno": 1, "col": 0,
                      "src_line": source.splitlines()[0] if source else ""},
            suggested_repairs=["pass a single Python expression, not statements"],
        )

    local = _locally_bound_names(tree)
    node_body = _node_body_nodes(tree)

    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in _ALLOWED_NODES:
            raise _violation(
                source, node,
                f"disallowed AST construct: {name}",
                _repair_for_node(name),
            )
        # Attribute access — reject private/dunder names.
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise _violation(
                source, node,
                f"private/dunder attribute access not allowed: {node.attr!r}",
                "do not access names starting with underscore",
            )
        # `case Foo(_secret=v)` names an attribute *as an identifier*, so it
        # produces no `Attribute` node at all and the rule above never sees it.
        # Without this, allowing `Match` would have opened a bypass around the
        # underscore policy rather than just adding pattern matching.
        if isinstance(node, ast.MatchClass):
            for attr in node.kwd_attrs:
                if attr.startswith("_"):
                    raise _violation(
                        source, node,
                        f"private/dunder attribute access not allowed in a "
                        f"class pattern: {attr!r}",
                        "match on public attributes only",
                    )
        # Writes through a target. Attribute writes stay rejected outright:
        # `obj._parts = ...` mutating SDK internals is the thing this guards.
        #
        # The subscript rule below is a **lint, not the boundary**. It tests a
        # name, and a name cannot carry provenance: `h = objects; h['x'] = 1`
        # and `def f(t): t['x'] = 1; f(objects)` both bind a local, and
        # `objects.clear()` needs no subscript at all. Injected state is kept
        # safe by handing the sandbox a frozen view of it (`sdk/_readonly.py`),
        # which closes aliases and mutating methods together. What survives
        # here is the early, legible rejection of `Object['x'] = 1`.
        #
        # It must stay narrow. A blanket ban on subscript assignment would
        # protect nothing (no SDK type defines __setitem__, and `.update()` /
        # `.append()` are allowed) while refusing the plain local dicts a
        # program uses to loop over symmetric hardware:
        #     handle_parts[side] = (lower, hub)      brackets[side] = b
        #     caps[(rname, x)] = cap                 steer_j[tag] = ...
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                             ast.Delete)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.Delete):
                targets = node.targets
            else:
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and id(t) not in node_body:
                    raise _violation(
                        source, t,
                        "attribute assignment not allowed",
                        "do not mutate attributes on SDK objects (a @shader "
                        "or @lighting body may, since those set Blender node "
                        "and datablock properties)",
                    )
                if isinstance(t, ast.Subscript):
                    base = _subscript_base(t)
                    if not (isinstance(base, ast.Name) and base.id in local):
                        shown = (f"`{base.id}`" if isinstance(base, ast.Name)
                                 else "that expression")
                        raise _violation(
                            source, t,
                            "subscript assignment on a non-local container",
                            f"{shown} is not a name this source binds; item "
                            f"assignment is allowed on lists and dicts your "
                            f"own program creates, not through an injected "
                            f"global, an attribute, or a call result",
                        )
                if isinstance(t, ast.Name) and isinstance(node, ast.Delete):
                    if t.id not in local:
                        raise _violation(
                            source, t,
                            f"cannot delete the injected name {t.id!r}",
                            "del only names your own source bound",
                        )
        # Name reference to a banned builtin.
        if isinstance(node, ast.Name) and node.id in _REJECTED_NAMES:
            raise _violation(
                source, node,
                f"reference to disallowed name: {node.id!r}",
                f"do not use {node.id}",
            )
    return tree


# ---------------------------------------------------------------------------
# Restricted globals
# ---------------------------------------------------------------------------

def _no_op_print(*args, **kwargs):  # silent; bytes intentionally dropped
    return None


#: Pure numeric functions from `math`, spelled as builtins.
#:
#: `import math` is not allowed and will not be — the allowlist has no import
#: statement at all — but the *functions* carry no attack surface: they take
#: floats and return floats, reach nothing, and mutate nothing.
#:
#: They are here because leaving them out would remove capability, not API.
#: The animation layer's base channel is authored as world poses, so "rotate
#: this about the table's edge" is written by tracing the arc — which needs
#: `cos`/`sin` and nothing else. The object layer wants the same thing for
#: placing parts on a circle.
def _smoothstep(x: float) -> float:
    """Hermite ease from 0 to 1 over `x` in [0, 1], clamped outside it.

    Not from `math`, and here for the same reason the trig functions are: every
    authored deformation rig needs it and none can import it. A pure function
    of one float, so it carries no more surface than `sin`.
    """
    y = min(1.0, max(0.0, float(x)))
    return y * y * (3.0 - 2.0 * y)


def _pulse(t: float, rise0: float, rise1: float,
           fall0: float, fall1: float) -> float:
    """Ease up over `[rise0, rise1]`, hold, ease back down over `[fall0, fall1]`.

    The shape of a gust, a squeeze, a press — an amplitude envelope with a
    defined start and end rather than a wave that never stops. Deformation tracks
    want this far more often than they want a bare `sin`, because a deformation
    that does not return to rest leaves the part permanently bent.
    """
    # A zero-width ramp is a step, not a division by zero. `pulse(t, 0, 0, 1, 1)`
    # is the natural way to write "on at 0, off at 1" and must not raise
    # ZeroDivisionError from inside a curated math builtin.
    def _edge(x, a, b):
        if b <= a:
            return 0.0 if x < a else 1.0
        return _smoothstep((x - a) / (b - a))

    return _edge(t, rise0, rise1) * (1.0 - _edge(t, fall0, fall1))


_MATH_BUILTINS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "hypot": math.hypot,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "radians": math.radians, "degrees": math.degrees,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "tau": math.tau, "inf": math.inf,
    "smoothstep": _smoothstep, "pulse": _pulse,
}

# Exceptions the source may name. Needed the moment `try`/`except` entered the
# allowlist: none of these resolved before, so `except ValueError:` was a
# NameError. build123d's own failures (`Standard_Failure`, `StdFail_NotDone`,
# `Standard_ConstructionError`, `DraftAngleError`) come in through
# `_b123d_namespace.CURATED` instead, because this module must stay importable
# without build123d — it is loaded in the main env too, which has no CAD.
#
# `BaseException` is deliberately absent. `except BaseException: pass` would
# swallow `KeyboardInterrupt`, which the compile path re-raises on purpose, and
# `SystemExit`, which it reports as a policy breach.
_EXCEPTION_BUILTINS = {
    "Exception": Exception,
    "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
    "AttributeError": AttributeError, "IndexError": IndexError,
    "KeyError": KeyError, "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "TypeError": TypeError,
    "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError,
}

_RESTRICTED_BUILTINS = {
    "range": range, "len": len, "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    "int": int, "float": float, "bool": bool, "str": str,
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
    "any": any, "all": all, "sorted": sorted, "reversed": reversed,
    "isinstance": isinstance, "issubclass": issubclass,
    # Iteration and numeric builtins. Comprehensions are allowed, so leaving
    # these out would draw an arbitrary line.
    "map": map, "filter": filter, "next": next, "iter": iter,
    "divmod": divmod, "pow": pow, "repr": repr, "format": format,
    "slice": slice,
    # `ClassDef` has been allowed for local parametric helpers all along, and
    # `super` is what lets a subclass extend an inherited method:
    # `super().profile() + Rectangle(...)`. It cannot chain `__init__` —
    # `super().__init__(...)` needs dunder attribute access, and a helper class
    # cannot hold state anyway while attribute assignment is refused below.
    "super": super,
    # The primitive a `class` statement compiles to. `ClassDef` is allowed for
    # local parametric helpers, and without this every `class` dies at run time
    # with `NameError: __build_class__ not found`.
    # It grants nothing `ClassDef` does not already intend: the escapes that
    # make `type` interesting (`__mro__`, `__subclasses__`) are dunder
    # attributes, which the walker still rejects.
    "__build_class__": __build_class__,
    "True": True, "False": False, "None": None,
    "print": _no_op_print,
    **_MATH_BUILTINS,
    **_EXCEPTION_BUILTINS,
}


def _restricted_builtins() -> dict:
    """A fresh builtins table for one execution.

    A copy already, so mutating it was only ever self-harm within a single
    exec — but `__builtins__` is readable as a bare Name and `b = __builtins__;
    b.clear()` reads exactly like the attacks that were not self-harm. Handing
    out a copy keeps that story simple and costs nothing measurable.

    Deliberately a real dict rather than a frozen view: CPython resolves
    builtins through this object on every name lookup, and a Mapping subclass
    there would slow every line of every authoring program for no gain.
    """
    return dict(_RESTRICTED_BUILTINS)


def build_globals(overlay: dict | None = None) -> dict:
    """Globals for executing the source file: layer-supplied overlay plus
    curated builtins.

    The default overlay (when None) exposes the object SDK, preserving the
    original v1 behavior used by the object agent and its tests.
    """
    if overlay is None:
        from ...sdk import Object  # local import: SDK is the injected surface
        overlay = {"Object": Object}
    # `__name__` is not decoration: a class body reads it to build
    # `__qualname__`/`__module__`, so without it every `class` in `object.py`
    # dies with `NameError: name '__name__' is not defined`.
    g = {"__builtins__": _restricted_builtins(), "__name__": "object"}
    g.update(overlay)
    return g


def build_query_globals(overlay: dict) -> dict:
    """Globals for evaluating a `probe` expression.

    The overlay carries layer-specific symbols (`ir` / `catalog` / `objects`
    plus the per-layer query helpers); LayerConfig.probe_overlay_factory
    builds it. Curated builtins are always merged in on top, along with the
    same `math` / `np` handles `object.py` gets — a probe reads `part_shape`
    tables and bbox tuples, and that is the same arithmetic. The overlay wins
    if a layer binds either name itself.
    """
    g = {"__builtins__": _restricted_builtins()}
    g.update(_numeric_handles())
    g.update(overlay)
    return g


def _numeric_handles() -> dict:
    """`math` and `np`, for surfaces that get no build123d overlay.

    Imported lazily and best-effort: this module loads in the main env as well
    as the CAD one, and a probe must not fail to evaluate because numpy is
    absent from some interpreter.
    """
    import math  # noqa: PLC0415
    out: dict = {"math": math}
    try:
        import numpy  # noqa: PLC0415
    except ImportError:
        return out
    out["np"] = numpy
    return out
