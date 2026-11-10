"""Generate the build123d namespace reference from the namespace itself.

The `build123d_authoring` skill's `references/namespace_reference.md` lists
exactly what `object.py` can call. Written by hand it would drift from
`_b123d_namespace.CURATED`.

So it is generated. `render()` builds the page from the live namespace plus
`inspect.signature` under the pinned interpreter, and a test regenerates it and
diffs against the committed file. Editing the namespace without regenerating
fails that test; the reference cannot drift.

The page is a **neutral interface contract**: names, signatures and enum members
only. It deliberately carries no usage advice, no per-operation explanation and
no worked examples — the authoring policy lives in the prompts and the skill, and
this file is only the answer to "which symbols exist and what do they take".

Must run in the CAD interpreter — build123d is not importable in the main env.
Run directly to refresh the page::

    LD_LIBRARY_PATH=<env>/lib <env>/bin/python -m ..._namespace_doc > <page>

or, from a checkout that is not importable as `dycraft`, see the invocation in
`tests/test_object_namespace_reference.py`.
"""

from __future__ import annotations

import enum
import inspect
import re

# Names withdrawn from the namespace on purpose, and the reason. Rendered into
# the page so an agent that reaches for one is told why it is missing rather
# than only that it is — build123d's own documentation shows all of them.
WITHDRAWN = (
    ("BuildPart, BuildSketch, BuildLine",
     "builder mode needs `with`, which the sandbox rejects — use algebra mode"),
    ("edges(), faces(), solids(), vertices(), wires(), and their singular forms",
     "these read the *active builder*; each raises `requires a Builder context "
     "to be in scope`. Use the methods on a shape: `part.edges()`"),
    ("MM, CM, M, IN, FT",
     "these constants assume millimeters (`M` is 1000); DyCraft interprets "
     "geometry coordinates as meters"),
    ("Text, FontStyle, available_fonts",
     "depends on installed fonts rather than on the geometry asked for"),
    ("RigidJoint, RevoluteJoint, LinearJoint, CylindricalJoint, BallJoint",
     "build123d's own joint system; DyCraft owns anchors and joints"),
    ("import_step, import_stl, export_step, export_stl, export_gltf, "
     "export_brep, Mesher, ExportSVG, ExportDXF, export_to_pcbway",
     "file and network I/O; `add_part` is the only supported way geometry "
     "leaves `object.py` — the harness stages the .brep and the mesh"),
    ("DimensionLine, ExtensionLine, TechnicalDrawing, Draft, Arrow, ArrowHead",
     "dimensioned 2D drawings, which this pipeline does not produce — and "
     "they render text from a system font"),
    ("ArcArcTangentArc, ArcArcTangentLine, PointArcTangentArc, "
     "PointArcTangentLine",
     "deprecated in build123d 0.11.1; use `ConstrainedArcs` / "
     "`ConstrainedLines`, both of which are bound"),
    ("BoundBox, OrientedBoundBox, bounding_box",
     "already reachable as methods: `part.bounding_box()`, "
     "`part.oriented_bounding_box()`, `Solid.from_bounding_box(...)`"),
    ("new_edges is available; Select.LAST and Select.NEW are not usable",
     "`Select.LAST`/`Select.NEW` need a builder context to diff against — "
     "`new_edges(a, b, combined=a + b)` is the algebra-mode equivalent"),
)


#: Classmethod/staticmethod names present on every shape class through
#: `Shape`, carrying no modeling vocabulary of their own. Excluded so the
#: factory table lists what each class can *make* rather than repeating ten
#: rows of base-class plumbing per class.
_FACTORY_PLUMBING = frozenset({
    "cast", "combined_center", "compute_mass", "get_shape_list",
    "get_single_shape", "make_composite", "register_composite_factory",
    "combine", "order_chamfer_edges",
})

#: Withheld for the same reason `Text` is not in `CURATED`: it renders a
#: system font, so its success depends on what fonts the CAD environment
#: happens to have rather than on the geometry asked for.
_FACTORY_WITHHELD = frozenset({"make_text"})


def _class_factories(value) -> list[tuple[str, str]]:
    """Qualified names and signatures of declared class/static factories.

    Constructor signatures omit useful surfacing APIs such as
    `Face.make_surface_from_array_of_points`. Inspect only `vars(value)` to
    avoid repeating inherited factories on every subclass.
    """
    out = []
    for name in sorted(vars(value)):
        if name.startswith("_") or name in _FACTORY_PLUMBING:
            continue
        if name in _FACTORY_WITHHELD:
            continue
        if not isinstance(inspect.getattr_static(value, name),
                          (classmethod, staticmethod)):
            continue
        member = getattr(value, name)
        try:
            signature = _call_signature(member)
        except (ValueError, TypeError):
            signature = "(…)"
        out.append((f"{value.__name__}.{name}", signature))
    return out


def _call_signature(value) -> str:
    """Keep every parameter/default, without the verbose type annotations."""
    signature = inspect.signature(value)
    parameters = [p.replace(annotation=inspect.Parameter.empty)
                  for p in signature.parameters.values()]
    return str(signature.replace(parameters=parameters,
                                 return_annotation=inspect.Signature.empty))


def _kind_and_detail(value):
    """(kind, detail) for one injected name."""
    if isinstance(value, type) and issubclass(value, enum.Enum):
        return "enum", ", ".join(m.name for m in value)
    if isinstance(value, type):
        try:
            return "class", f"{value.__name__}{_call_signature(value)}"
        except (ValueError, TypeError):
            return "class", value.__name__
    if callable(value):
        try:
            sig = _call_signature(value)
        except (ValueError, TypeError):
            return "function", f"{getattr(value, '__name__', '?')}(…)"
        return "function", f"{getattr(value, '__name__', '?')}{sig}"
    return "value", repr(value)


def _clean(sig: str) -> str:
    """Strip fully-qualified build123d module paths and collapse whitespace.

    Deliberately does **not** truncate: this page is the whole interface
    contract and the only place the agent is told what a call takes, and a
    signature cut at a fixed length (`align: 'Align | tuple[Align, Align,
    Align]' = (<Align.CEN…`) does not say what the parameter is.
    """
    for prefix in ("build123d.topology.composite.",
                   "build123d.topology.zero_d.", "build123d.topology.one_d.",
                   "build123d.topology.two_d.", "build123d.topology.three_d.",
                   "build123d.topology.shape_core.", "build123d.geometry.",
                   "build123d.build_enums.", "build123d.build_common.",
                   "collections.abc.", "typing."):
        sig = sig.replace(prefix, "")
    # `inspect.signature` renders an enum default through build123d's own
    # `__repr__`, giving `<Mode.ADD>` — which is not an expression anyone can
    # write. `Mode.ADD` is, and `Mode` is injected, so the dotted form is both
    # what the default *is* and something the reader can put in the source.
    # The `(?:: [^<>]+)?` is this line's, not the shader-viewer line's, and
    # theirs' own `test_no_default_is_an_unwritable_repr` is what proved it
    # necessary: an enum whose members carry values reprs as
    # `<ContinuityLevel.C0: 0>`, and a pattern that only handles `<Mode.ADD>`
    # leaves the value suffix behind — an unwritable default on the page and
    # not valid Python either. Theirs never saw it because that line's CURATED
    # is 83 names; this one's is 122.
    sig = re.sub(r"<([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+)(?:: [^<>]+)?>",
                 r"\1", sig)
    return " ".join(sig.split())


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell.

    build123d's type unions are written `'Align | tuple[Align, Align]'`, and a
    raw `|` splits the row even inside a code span, turning a two-column row
    into four or more columns. GitHub-flavored markdown honors the
    backslash escape inside backticks, so the pipe survives as a pipe.
    """
    return text.replace("|", "\\|")


def render(curated, build123d_version: str) -> str:
    """The namespace reference page, as markdown."""
    import build123d as bd

    # Resolve through the same helper the live namespace uses, so a curated
    # name the pinned build123d cannot export fails here too. Resolving
    # independently would let both sides skip it in lockstep, and the
    # generated-vs-committed diff would stay clean through the exact typo it
    # exists to catch.
    try:                       # imported as a dycraft submodule (the test)
        from ._b123d_namespace import resolve_curated
    except ImportError:        # run as a bare script from this directory
        from _b123d_namespace import resolve_curated  # type: ignore

    rows = []
    factories: list[tuple[str, str]] = []
    for name, value in resolve_curated(bd, curated).items():
        kind, detail = _kind_and_detail(value)
        rows.append((name, kind, _clean(detail)))
        # The factory table is the skill page's, and it is the reason this
        # collection exists: `_kind_and_detail` renders a class as its
        # CONSTRUCTOR, so `Face` alone would read as `Face(*args, **kwargs)`
        # and say nothing about `Face.make_surface_from_array_of_points`.
        # Every freeform-surface route in build123d is a classmethod, so
        # without this table the namespace looks as if it has no surfacing API
        # and a reader reaches for `OCP` instead.
        if kind == "class":
            factories += [(m, _clean(sig)) for m, sig in _class_factories(value)]

    by_kind: dict[str, list] = {}
    for name, kind, detail in rows:
        by_kind.setdefault(kind, []).append((name, detail))

    out = [
        "# Namespace reference",
        "",
        "**Generated — do not edit.** Produced by "
        "`agent/cad/_namespace_doc.py` from the live namespace under "
        f"build123d {build123d_version}, and checked by "
        "`tests/test_object_namespace_reference.py`.",
        "",
        f"`object.py` binds **{len(rows)}** build123d names, also reachable "
        "through `bd`. This page describes DyCraft's injected namespace, "
        "not the full API available to raw build123d scripts. `math`, "
        "`np` (NumPy), and `OCP` are module handles; scalar math names such "
        "as `sin`, `cos`, `radians`, and `sqrt` are also available bare.",
        "",
        "Call signatures preserve all parameters and defaults; type "
        "annotations and enum display brackets are omitted. A bare class "
        "name or `(...)` means its signature was unavailable to "
        "introspection.",
        "",
    ]

    titles = {"class": "Classes and shape constructors",
              "function": "Operations",
              "enum": "Enums",
              "value": "Values"}
    for kind in ("class", "function", "enum", "value"):
        if kind not in by_kind:
            continue
        out += [f"## {titles[kind]}", "",
                "| name | " + {"enum": "members",
                                   "value": "definition"}.get(kind, "signature")
                + " |", "|---|---|"]
        for name, detail in sorted(by_kind[kind]):
            out.append(f"| `{name}` | `{_cell(detail)}` |")
        out.append("")

    if factories:
        out += [
            "## Class factory methods", "",
            "Call factories on the class, e.g. `Face.make_surface(...)`. "
            "These complement the constructors above: `Face.make_surface*` "
            "and `Face.make_gordon_surface` construct surfaces, "
            "`Shell.make_loft` and `Solid.make_loft` loft sections, and "
            "`Solid.thicken` offsets a face into a wall. Check these before "
            "implementing the operation through `OCP`.", "",
            "Return types vary: factories produce topology objects such as "
            "`Face`, `Shell`, or `Solid`; algebra operations can return "
            "`Curve`, `Sketch`, or `Part`. An empty `Solid & x` can return "
            "`None`, while `Part & x` returns an empty `Compound`. Before "
            "continuing a one-solid construction, check "
            "`shape is not None and len(shape.solids()) == 1`.", "",
            "| name | signature |", "|---|---|",
        ]
        # Sorted by qualified name, so each class's factories stay together:
        # "what can `Face` make" is the question this table gets asked, and a
        # sort on the bare method name answers it by scattering seven
        # `extrude` rows and splitting `make_loft` across two classes.
        for name, detail in sorted(set(factories)):
            out.append(f"| `{name}` | `{_cell(detail)}` |")
        out.append("")

    out += ["## The kernel, as `OCP`", "",
            "For lower-level control, pass `shape.wrapped` to OCCT and "
            "wrap the `TopoDS_*` result with the matching build123d topology "
            "class, such as `Face`, `Shell`, or `Solid`. Examples include "
            "variable-radius fillets through "
            "`OCP.BRepFilletAPI.BRepFilletAPI_MakeFillet` and sewing through "
            "`OCP.BRepBuilderAPI.BRepBuilderAPI_Sewing`. For algorithms that "
            "expose `IsDone()`, check it before retrieving the result.", "",
            "Wrap a bare `Geom_Surface` with "
            "`Face(OCP.BRepBuilderAPI.BRepBuilderAPI_MakeFace(surface, tol)"
            ".Face())`. For point grids, start with "
            "`Face.make_surface_from_array_of_points`; its `tol` is in "
            "model units, so the default `0.01` can be too coarse for "
            "thin sheets authored in meters. If 3-D fitting distorts X/Y "
            "enough to disrupt prism booleans or volume, the "
            "`OCP.GeomAPI.GeomAPI_PointsToBSplineSurface` **height-grid "
            "overload** accepts a `TColStd_Array2OfReal` Z grid plus "
            "`X0, dX, Y0, dY`, preserving affine X/Y parameterization.", ""]

    out += ["## Not available, and why", "",
            "build123d's documentation shows all of these. They are not in "
            "scope here.", "",
            "| what | why |", "|---|---|"]
    for what, why in WITHDRAWN:
        out.append(f"| {what} | {why} |")
    out.append("")

    out += ["## Algebra-mode arguments", "",
            "- `mode=` controls combination with an active builder; in algebra "
            "mode combine returned shapes with `+`, `-`, or `&`.",
            "- `clean=` also applies in algebra mode; it controls topology cleanup.",
            "- Supply operands explicitly where a `None` default would otherwise "
            "take input from an active builder.", ""]
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover - invoked by hand / by the test
    import importlib.metadata as _md
    import sys

    from _b123d_namespace import CURATED  # type: ignore

    # `write`, not `print`: the committed page must equal `render()` byte for
    # byte, and the test diffs it against a fresh call.
    sys.stdout.write(render(CURATED, _md.version("build123d")))
