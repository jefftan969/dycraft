"""The build123d surface injected into `object.py`'s globals.

`object.py` runs under an AST allowlist with no ``Import`` / ``ImportFrom``
node (`sandbox._ALLOWED_NODES`) and builtins with no ``__import__``. The CAD
API is injected instead: `Call`, `Attribute` and `Name` are already allowed, so
``Box(0.1, 0.2, 0.3)`` and ``bd.fillet(...)`` work and the import policy stays
as strict as it is.

Two ways in:

  * a curated set of names bound directly, so ordinary modeling reads the way
    build123d's own documentation and examples do;
  * ``bd``, a *guarded view* over exactly those names, so sources written in
    the ``bd.Box(...)`` style work too.

``bd`` is not the build123d module itself, which would be an unrestricted
escape: build123d 0.11.1 re-exports 54 modules at its top level
(``bd.os.system``, ``bd.sys.path``, ``bd.Lib3MF.ctypes.util.subprocess``), and
``bd.Path``, ``bd.import_step``, ``bd.export_stl``, ``bd.Mesher``, and
``bd.export_to_pcbway`` reach the filesystem or the network.
`_GuardedNamespace` resolves ``bd.X`` only if ``X`` is also bound as a global,
so it adds no reach over the direct bindings.

``math``, ``np``, and ``OCP`` are the real modules:

  * ``math`` exposes no module-typed attribute, so there is nothing to traverse
    to (``sandbox._MATH_BUILTINS`` also supplies bare ``sin``/``cos``/... to
    ``object.py`` and probes).
  * ``np`` re-exports modules, and no denylist closes it: with ``ctypeslib``,
    ``f2py``, ``testing``, ``random``, and ``distutils`` denied,
    ``np.ma.core.builtins`` (the real ``builtins``), ``np.lib.format.os``, and
    ``np.lib.format.pickle`` are still within three hops, and
    ``_REJECTED_NAMES`` checks ``Name`` nodes only. ``ndarray.tofile`` /
    ``.dump`` / ``.dumps`` are instance methods no namespace guard could
    intercept anyway.
  * ``OCP`` reaches what build123d's fluent layer cannot (a fillet radius that
    varies along the edge, ``BRepFilletAPI_MakeFillet.Add(r1, r2, edge)``; a
    face set ``BRepBuilderAPI_Sewing`` stitches at a tolerance), costs nothing
    to import (build123d has already loaded 410 of its submodules), and opens
    no reach ``np`` had not.

These are accepted because this sandbox is not the containment boundary: the
agent that writes ``object.py`` also runs shell commands, so containment is the
agent CLI's own sandbox and the run's scope (`_oom.enter_scope`). What the
exceptions cost is the *artifact contract*: ``object.py`` is replayed outside
its session by `fixed_source`, and a source that reaches the filesystem is not
replayable. Weigh that before widening this further. Replay works wherever the
harness rebuilds these globals (``object.py`` is not standalone Python, since
``Object`` and ``bd`` are injected too), and
``_b123d_version.check_ocp_available`` makes that checkable: a child with
build123d but no OCP fails at start with ``ocp_missing``.

**Collisions are checked, not assumed.** Injected names are merged on top of
the SDK overlay, so a build123d export called ``Object`` or ``Material`` would
silently shadow the SDK. The pinned 0.11.1 has none (473 top-level exports,
none matching the SDK overlay or a restricted builtin), but that is a property
of a version, so `assert_no_collisions` turns a future one into a loud error at
overlay construction.
"""

from __future__ import annotations


def _sandbox_violation(summary: str, evidence: dict, repair: str):
    """Build a `SandboxViolation`, importing it lazily.

    Deliberately not a module-level import: this module is loaded standalone
    (no package context) by the prompt-contract test, which reads `CURATED` to
    check the prompt teaches no name the namespace cannot bind. A package-
    relative import at module scope would break that, and the vocabulary guard
    is worth more than the one saved line.
    """
    from ...sdk._errors import SandboxViolation
    return SandboxViolation(summary, refs=[], evidence=evidence,
                            suggested_repairs=[repair])




class _GuardedNamespace:
    """A read-only view exposing exactly the names in a fixed table.

    Used for the `bd` and `math` handles. The point is that a handle must not
    be a door to anything its bound-directly siblings are not: `getattr` serves
    the table or refuses, so the reachable surface of `bd.X` is by construction
    identical to the set of injected globals, whatever build123d adds in a
    future release.

    Refusal raises `SandboxViolation` rather than `AttributeError` because that
    is what it is — a policy denial, not a typo — and `compile` already routes
    that to its own report field with the offending name attached.
    """

    __slots__ = ("_label", "_table")

    def __init__(self, label: str, table: dict):
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_table", dict(table))

    def __getattr__(self, name: str):
        table = object.__getattribute__(self, "_table")
        if name in table:
            return table[name]
        label = object.__getattribute__(self, "_label")
        raise _sandbox_violation(
            f"{label}.{name} is not available in object.py",
            {"node_type": "GuardedNamespace", "lineno": 0, "col": 0,
             "src_line": "", "handle": label, "attr": name},
            f"`{label}.{name}` is outside this namespace's exposed vocabulary. "
            f"Use an available attribute; the authoring reference describes "
            f"the supported names. The namespace is not the full module.",
        )

    def __setattr__(self, name, value):
        label = object.__getattribute__(self, "_label")
        raise _sandbox_violation(
            f"cannot assign to {label}.{name}",
            {"node_type": "GuardedNamespace", "lineno": 0, "col": 0,
             "src_line": "", "handle": label, "attr": name},
            "the injected namespaces are read-only",
        )

    def __dir__(self):
        return sorted(object.__getattribute__(self, "_table"))

    def __repr__(self):
        label = object.__getattribute__(self, "_label")
        n = len(object.__getattribute__(self, "_table"))
        return f"<{label}: {n} injected names>"


# Curated names bound directly into the authoring namespace. Chosen to cover
# the modeling vocabulary the object layer actually needs — primitives,
# placement, sketching, the standard operations, and the selector vocabulary —
# without pulling the whole top-level namespace into scope (473 names at the
# pinned version, counted in this module's docstring), where a typo would
# silently resolve to something unrelated.
CURATED = (
    # primitives / bodies
    "Box", "Cylinder", "Sphere", "Cone", "Torus", "Wedge",
    "Solid", "Compound", "Part", "Shell", "Face", "Wire", "Edge", "Vertex",
    # `ConvexPolyhedron` is the points -> solid hull; `make_hull` below is the
    # 2D one (edges -> Sketch), which is why both are here. The three hole
    # cutters default to `mode=Mode.SUBTRACT`, which is builder-only: in
    # algebra mode they construct the cutting solid and you subtract it
    # yourself (`part -= CounterBoreHole(...)`). `Hole` alone is just a
    # cylinder, kept so the family is not split across the boundary.
    "ConvexPolyhedron", "Hole", "CounterBoreHole", "CounterSinkHole",
    # sketch primitives. `Text` is deliberately absent: it renders a font, so
    # it is the one modeling name whose success depends on what fonts the CAD
    # environment happens to have rather than on the geometry asked for, and no
    # articulated object in the run history has ever needed lettering.
    "Rectangle", "Circle", "Ellipse", "Polygon", "RegularPolygon",
    "Triangle", "Trapezoid", "SlotOverall", "SlotCenterToCenter",
    "RectangleRounded", "SlotArc", "SlotCenterPoint",
    "Sketch", "Curve",
    # sketch/line geometry
    "Line", "Polyline", "Spline", "RadiusArc", "SagittaArc", "CenterArc",
    "TangentArc", "ThreePointArc", "JernArc", "Bezier", "FilletPolyline",
    # The rest of build123d's curve vocabulary. `Helix` is the reason this
    # group exists: without it there is no spring, no coil and no thread, and
    # the corpus asks for threaded parts. `Ellipse` above is the closed curve
    # only — the elliptical/parabolic/hyperbolic *arcs* are here. Every one of
    # these was verified to construct outside a builder context; the four
    # deprecated tangent-arc spellings (ArcArcTangentArc, ArcArcTangentLine,
    # PointArcTangentArc, PointArcTangentLine) are deliberately NOT here —
    # 0.11.1 warns on them and routes to ConstrainedArcs / ConstrainedLines.
    "Helix", "BSpline",
    "EllipticalCenterArc", "EllipticalStartArc",
    "ParabolicCenterArc", "HyperbolicCenterArc",
    "PolarLine", "IntersectingLine", "DoubleTangentArc", "BlendCurve",
    "ConstrainedArcs", "ConstrainedLines", "Airfoil",
    # operations
    "extrude", "revolve", "loft", "sweep", "section", "offset", "scale",
    "fillet", "chamfer", "mirror", "split", "make_face", "make_hull",
    "add", "thicken", "project", "trace",
    # `draft` is the molding/casting taper. It is *not* reachable as a method
    # in practice: algebra mode hands back a `Part`, and `.draft` lives on
    # `Solid`, so it would mean `part.solids()[0].draft(...)`. The free
    # function is the real entry point. `full_round` solves for the largest
    # tangent arc that rounds a sketch end (fillet makes you pick the radius);
    # `make_brake_formed` is sheet metal; `project_workplane` builds a plane
    # by projection rather than by hand.
    "draft", "make_brake_formed", "project_workplane", "full_round",
    # Connectivity-based selection. Everything else here selects geometrically
    # (`sort_by`, `group_by`, `filter_by` against GeomType/Axis/SortBy), which
    # cannot express "the rest of the edge loop around this pocket" — the
    # selection you want when filleting one feature without rounding the
    # whole shape. `polar` joins them because bolt circles are the same job.
    "topo_explore_connected_edges", "topo_explore_common_vertex",
    "topo_distance_to", "edges_to_wires", "polar",
    # `new_edges(*objects, combined=…)` — the edges a boolean created. Builder
    # mode gets this from `Select.NEW`, which diffs against the builder's
    # previous state; algebra mode has no context to diff against, so without
    # this there is no way to fillet a weld seam without also rounding every
    # other edge of the shape.
    "new_edges",
    # placement / frames
    "Location", "Locations", "Pos", "Rot", "Rotation", "RotationLike",
    "Plane", "Axis", "Vector", "Color",
    # `Rotation` and `Location` both take `ordering: Extrinsic | Intrinsic`
    # (default `Intrinsic.XYZ`). Binding the constructor without its enums
    # left every non-default Euler ordering unnameable. `Matrix` completes the
    # affine surface; `Plane` has an overload that takes one.
    "Matrix", "Extrinsic", "Intrinsic",
    "GridLocations", "PolarLocations", "HexLocations",
    # enums / selectors. `FontStyle` goes with `Text`.
    "Align", "Mode", "Kind", "Keep", "Select", "SortBy", "GeomType",
    "Until", "Side", "Transition", "CenterOf", "PositionMode",
    "FrameMethod", "AngularDirection", "LengthMode", "MeshType",
    # Arguments to the curve constructors added above.
    "Sagitta", "Tangency", "ContinuityLevel",
    # Exception types, so `except` can name what it is catching. These are the
    # failures worth catching — a fillet radius OCCT refuses — and without
    # them the sandbox's new `try` only reaches them as bare `Exception`.
    # Not in `build123d.__all__`, but `resolve_curated` goes through getattr,
    # and build123d re-exports them precisely so callers can catch them.
    "DraftAngleError", "Standard_Failure", "Standard_ConstructionError",
    "StdFail_NotDone",
    # NOT the builder API. `BuildPart` / `BuildSketch` / `BuildLine` require
    # `with`, which is not in the AST allowlist, so they could be constructed
    # and never entered — injected surface with no reachable use. The prompt
    # teaches algebra mode.
    #
    # NOT the unit constants either. build123d's `MM` is 1 and its `M` is 1000,
    # because the library's own unit is the millimeter; DyCraft authors in
    # meters, so `Box(0.24 * M)` is a 240-meter box and `0.24 * MM` is a
    # quarter-millimeter one. Both read as correct. Nothing in the run history
    # uses them, the prompt never taught them, and a scale factor that is right
    # in one library and wrong in this one is not worth keeping in scope.
)


class NamespaceCollision(RuntimeError):
    """A build123d export would shadow an SDK name or a restricted builtin."""


class UnresolvedCuratedName(RuntimeError):
    """A name in `CURATED` is not exported by the pinned build123d."""


def resolve_curated(bd, curated=None) -> dict:
    """Bind every curated name against `bd`, or raise naming the misses.

    The interpreter is pinned to one build123d and refused before this runs
    (`_solid_compile.check_build123d_version`), so a name that does not resolve
    is a typo in `CURATED`, not a version difference. Skipping it would put the
    name in the prompt and the reference page but not in `object.py`'s globals,
    surfacing to the agent as a bare `NameError` for something it was told to
    use — the same silent degradation the version pin exists to prevent.

    Shared with `_namespace_doc.render` on purpose: if the two resolved names
    independently, they could skip in lockstep and the generated-reference diff
    would stay clean through the very typo it is meant to catch.
    """
    names = CURATED if curated is None else curated
    out: dict = {}
    missing = []
    for name in names:
        value = getattr(bd, name, None)
        if value is None:
            missing.append(name)
        else:
            out[name] = value
    if missing:
        raise UnresolvedCuratedName(
            f"build123d {getattr(bd, '__version__', '?')} does not export "
            f"{', '.join(sorted(missing))}; fix the name in "
            f"`_b123d_namespace.CURATED` (the interpreter is version-pinned, "
            f"so this is a typo, not a version skew)")
    return out


def assert_no_collisions(names, reserved) -> None:
    """Raise if any injected name would shadow something already in scope.

    `reserved` is everything the authoring namespace already relies on: the SDK
    overlay plus the restricted builtins. Shadowing either is a silent,
    extremely confusing failure — `Object` resolving to a build123d class would
    make every `Object("name")` call fail with a type error that points nowhere
    near the cause.
    """
    clashes = sorted(set(names) & set(reserved))
    if clashes:
        raise NamespaceCollision(
            f"build123d exports would shadow {clashes} in object.py's "
            f"namespace. Rename the SDK symbol or drop the build123d name from "
            f"CURATED — do not let the injection win silently.")


def build_namespace(reserved=()) -> dict:
    """The build123d + math globals for `object.py`, or raise if build123d is
    absent.

    Returns the curated names bound directly, plus four handles this module
    owns: `bd`, a `_GuardedNamespace` view over exactly those same names, and
    `math` / `np` / `OCP` as **real modules**.

    Those three are a deliberate exception to "no module object is in scope".
    `math`
    costs nothing to expose; `np` and `OCP` buy the variable-radius fillet and
    the tolerant sew, and `np` is a genuine escape surface. What each one costs
    is written out in
    `tests/test_object_sandbox_boundary.py::test_the_only_injected_modules_are_math_np_and_ocp`.

    A curated name the pinned build123d does not export raises
    `UnresolvedCuratedName` — see `resolve_curated`.
    """
    import math as _math

    import numpy as _np

    import build123d as bd

    import OCP as _OCP

    ns: dict = resolve_curated(bd)
    b123d_table = dict(ns)

    assert_no_collisions(list(ns), reserved)
    # Added after the collision check: `bd`/`math`/`np`/`OCP` are handles this
    # module owns, not build123d exports, and shadowing is the intended
    # behavior.
    ns["bd"] = _GuardedNamespace("bd", b123d_table)
    ns["math"] = _math
    ns["np"] = _np
    ns["OCP"] = _OCP
    return ns
