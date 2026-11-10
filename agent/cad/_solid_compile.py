"""Shared build123d Shape -> artifact compilation.

The one producer of a ``solid`` part: ``sdk.object_builder.Object.add_part``
registers a live build123d Shape constructed inside ``object.py``, and compiles
it at ``finalize()``. Four answers about a Shape come from here — what it
measures (``shape_table``), what its render mesh is
(``tessellate_within_budget``), how its surface is parameterized for a texture
(``_face_uv_mapper``, reached by passing a tile size down the tessellation
path), and what bytes get staged (``compile_entry``).

The third is here rather than in a renderer because tessellation DISCARDS the
B-rep face structure: this is the last point that still knows a barrel is a
cylinder rather than a strip of unrelated triangles. See the "Texture
coordinates" section.

**Import contract.** Standard library and the sibling ``_geometry_qc`` at module
scope; ``build123d`` / ``OCP`` only *inside* functions. ``_geometry_qc`` is pure
stdlib and required, not optional: a solid whose mesh was never welded, measured
or given normals must not be stageable, so a failure to import it is an
error rather than a degraded result.

Reached from the SDK through a process-active hook installed by the agent layer
(see ``sdk._mesh_assets.set_active_solid_compiler``) rather than by importing it
from the SDK, which would invert the sdk -> agent layering.

Numeric stability: every float that leaves here is rounded to ``SIG_DIGITS``
significant digits. These land in the ObjectIR and therefore in the content
hash, so they must not wobble with an OCCT patch release. Significant digits
rather than decimal places because a small part's volume is legitimately
~1e-6 m^3 and fixed-point rounding would flatten it to zero.
"""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
from collections import OrderedDict
from pathlib import Path

from . import _geometry_qc as _gq
# The pin and the shared error type live in `_b123d_version` because
# `_adjudicate_runner` loads them standalone, in an interpreter that has no
# dycraft. Re-exported here so callers and `except` clauses see one class.
from ._b123d_version import (  # noqa: F401 — re-exported
    REQUIRED_BUILD123D,
    SOLID_COMPILE_KINDS,
    SOLID_COMPILE_REPAIRS,
    SolidCompileError,
    check_build123d_version,
    check_ocp_available,
    installed_build123d_version,
    installed_ocp_version,
    kind_repairs,
)


# Six significant digits: below the ~1e-6-relative agreement OCCT's own
# integrators give, and far below any authoring tolerance that matters at
# DyCraft's 1 mm minimum part dimension.
SIG_DIGITS = 6

#: Triangle budget for a solid's render mesh, and how far the ladder may back
#: off when it is blown. See `tessellate_within_budget` for why a single
#: tolerance needs this guard at all.
#:
#: The budget is a soft target and exceeding it is cheap: measured, 6.1x the
#: triangles is 1.10x the render time (fixed-cost dominated — context setup and
#: six 1280x720 views), parsing is 2.5% of a render, and disk runs a stable
#: ~65 bytes per triangle. Destroying the geometry to honor it is not cheap,
#: which is what sets `COARSEN_ATTEMPTS`.
MAX_TRIANGLES = 20000
COARSEN_FACTOR = 4.0

#: Probe count, which bounds wall clock. It does NOT bound how far the ladder
#: coarsens — `COARSEN_MAX_FACTOR` does — because a rung skipped for making no
#: progress still advances the tolerance without spending an attempt; one
#: number doing both jobs would let a shape whose rungs stall coarsen past the
#: cap.
COARSEN_ATTEMPTS = 4

#: The most the chord tolerance may be coarsened in total, as a multiple of
#: where the ladder started.
#:
#: Converted to pixels in a canonical render (sagitta through the camera
#: formula in `render/__init__.py`, 720 px frame), every rung up to x64 adds
#: under a pixel of silhouette error, while x256 adds 6-9 px, a visibly
#: polygonal object. A shape that cannot fit by x64 is staged over budget
#: instead, which is the right way round: an over-budget mesh costs disk, a
#: ruined one costs the object.
COARSEN_MAX_FACTOR = 64.0

COARSEN_MIN_PROGRESS = 0.05

#: How weakly the triangle count can grow as the tolerance tightens, used to
#: decide whether a finer-than-default request is worth probing at all.
#:
#: `count ~ (1 / tol) ** e`. Measured across four shape families the exponent
#: runs 0.17 (sphere) to 0.65 (lofted shell) — far below 1, because past some
#: fineness `angular_tolerance` is what binds and the chord dial stops adding
#: triangles. Taking the *smallest* observed exponent makes the estimate
#: optimistic: if even the weakest growth puts the request over budget, no
#: shape can bring it under, so the probe can be skipped without ever
#: refusing a request that would have fitted.
#:
#: The guard exists because probing is not cheap: a torus at `tolerance=1e-5`
#: takes about two minutes to mesh before the budget rejects it, which against
#: the build time limit is a lost compile. It matters whenever the default fits
#: and the request is finer, so it is independent of where `MAX_TRIANGLES`
#: sits.
COARSEN_GROWTH_EXPONENT = 0.17

#: The tolerance `add_part` defaults to, named here because the coarsening
#: ladder is anchored on it rather than on the caller's request — see
#: `tessellate_within_budget` for why that anchoring is what makes the result
#: monotone in the request.
DEFAULT_TOLERANCE = 1e-3

# Planar grid imprinting, behind `add_part(planar_max_edge=...)`. The per-face
# cap bounds a pathological request (a 2 m panel at 1 mm would ask for 2828
# divisions a side); the attempt count is how many times the requested edge is
# doubled when the imprinted mesh blows MAX_TRIANGLES. See
# `imprint_planar_grids` for why the tolerance ladder cannot defend that budget.
PLANAR_IMPRINT_MAX_N = 64
PLANAR_COARSEN_ATTEMPTS = 4




# ---------------------------------------------------------------------------
# Numeric normalization
# ---------------------------------------------------------------------------

def q(value) -> float:
    """Round to `SIG_DIGITS` significant digits; normalize -0.0 to 0.0."""
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise SolidCompileError("invalid_mesh", f"non-finite geometry value {v!r}")
    out = float(f"{v:.{SIG_DIGITS}g}")
    return 0.0 if out == 0.0 else out


def qvec(v) -> list:
    return [q(c) for c in v]


def xyz(v):
    """(x, y, z) floats from a build123d Vector / OCC point / sequence.

    Three shapes of coordinate reach this, and no single access pattern covers
    them: a build123d ``Vector`` exposes uppercase ``.X/.Y/.Z`` as **properties**
    and is not subscriptable, so indexing it raises; a raw OCC point (``gp_Pnt``,
    ``gp_Dir``) exposes the same names as *methods*, hence the callable branch;
    a plain sequence falls through to indexing.
    """
    x = getattr(v, "X", None)
    if x is not None:
        if callable(x):                      # gp_Pnt / gp_Dir: X() Y() Z()
            return (float(v.X()), float(v.Y()), float(v.Z()))
        return (float(x), float(v.Y), float(v.Z))   # build123d Vector
    return (float(v[0]), float(v[1]), float(v[2]))


# ---------------------------------------------------------------------------
# Feature extraction — the `shape` fact table the agent reads
# ---------------------------------------------------------------------------

class FaceInspectionError(Exception):
    """A face could not be described. Carries what was being read, and why.

    Exists so that "this face is genuinely unsupported" and "we called an API
    that does not exist" do not look the same: swallowing every inspection
    error would let a missing build123d method silently strip every
    cylindrical face of its axis and bore classification.
    """

    def __init__(self, what: str, cause: BaseException):
        super().__init__(f"{what}: {type(cause).__name__}: {cause}")
        self.what = what
        self.cause = cause


# Attribute and type errors mean *we* called the API wrong, or it moved between
# releases. Those must never be silently absorbed — they are the version-skew
# signal. Everything else from OCCT (a surface it declines to describe, a
# degenerate trim) is a legitimately unsupported face.
#
# `ImportError` is deliberately NOT here. These helpers are also reachable from
# an interpreter with no CAD kernel at all (unit tests, the main DyCraft env),
# where "OCP is absent" means the optional second source is unavailable — not
# that DyCraft called something wrong. Import failures are handled where they
# occur, and only the *call* is treated as misuse.
_API_MISUSE = (AttributeError, TypeError, NameError)


def _inspect_face(face, *, primary, adaptor, primary_label, adaptor_label):
    """Ask build123d about a face, then OCCT, under one exception policy.

    Every per-face inspector needs the same three-way distinction, and getting
    it wrong is how a whole column of the inventory empties without a word:

      * **API misuse** — a renamed attribute, a property that became a method, a
        malformed face with no OCCT payload. `_API_MISUSE` names those, and they
        raise ``FaceInspectionError``. This is the case that must never be
        silent: it means the code and the installed kernel disagree.
      * **A domain "no"** — this surface has no radius, is not a revolution.
        The source declines by raising something else, and the answer is None.
      * **No kernel at all** — the adaptor is unimportable, so the primary
        source's answer stands and there is nothing further to ask.

    A second source is worth having because the two disagree on real geometry:
    ``Face.radius`` returns None for some trimmed cylindrical surfaces — the
    shallow scoop left by subtracting a large cylinder from a plate — where the
    adaptor still answers 0.5 on the same face.

    The policy lives here rather than in each inspector because ordering the two
    handlers the other way round silently swallows an API mismatch, and a
    convention cannot enforce itself across copies.
    """
    try:
        value = primary(face)
        if value is not None:
            return value
    except FaceInspectionError:
        raise                           # already diagnosed; do not retry
    except _API_MISUSE as e:
        raise FaceInspectionError(primary_label, e) from e
    except Exception:  # noqa: BLE001 — a domain "no": this surface has none
        pass
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
    except ImportError:
        return None                     # no kernel here; primary answer stands
    try:
        return adaptor(BRepAdaptor_Surface(face.wrapped))
    except _API_MISUSE as e:
        raise FaceInspectionError(adaptor_label, e) from e
    except Exception:  # noqa: BLE001 — genuinely not that kind of surface
        return None


def face_radius(face):
    """Radius of a cylindrical face, or None if the surface has none.

    ``Face.radius`` answers directly under the pinned build123d; the OCCT
    adaptor is the second source (see ``_inspect_face``).
    """
    return _inspect_face(
        face,
        primary=lambda f: None if f.radius is None else float(f.radius),
        adaptor=lambda s: float(s.Cylinder().Radius()),
        primary_label="Face.radius",
        adaptor_label="BRepAdaptor_Surface(...).Cylinder()")


def face_axis(face):
    """``(axis_point, axis_dir)`` for a face of revolution, or None.

    **This is the joint-axis candidate**, and the reason authoring in CAD beats
    authoring in triangles: a bore's axis is the hinge a revolute joint should
    turn about, so losing it costs the agent a real capability rather than a
    cosmetic field.

    ``Face.axis_of_rotation`` answers under the pinned build123d, returning an
    ``Axis`` with ``.position`` and ``.direction``. A missing method must not be
    swallowed: every bore would silently lose `axis_point`, `axis_dir` and
    `concave`, which is the failure `_inspect_face` exists to make impossible.
    The OCCT adaptor is the second source; `gp_Ax1` exposes
    Location()/Direction().
    """
    def _from_build123d(f):
        axis = f.axis_of_rotation
        if axis is None:
            return None
        try:
            return xyz(axis.position), xyz(axis.direction)
        except _API_MISUSE as e:
            raise FaceInspectionError("Axis.position/.direction", e) from e

    def _from_adaptor(surface):
        ax = surface.Cylinder().Axis()
        loc, direction = ax.Location(), ax.Direction()
        return ((loc.X(), loc.Y(), loc.Z()),
                (direction.X(), direction.Y(), direction.Z()))

    return _inspect_face(
        face, primary=_from_build123d, adaptor=_from_adaptor,
        primary_label="Face.axis_of_rotation",
        adaptor_label="gp_Ax1 Location()/Direction()")


def face_tables(part):
    """Per-face inventory: type census plus the cylindrical-face table.

    The cylindrical-face table is the payoff for authoring in CAD rather than
    triangles: each entry's axis is a *candidate joint axis* (an outer cylinder
    is a pin/tube, an inner one a bore or bearing seat), which is what lets a
    joint reference real geometry instead of a bounding-box corner plus a
    hand-tuned offset. Concavity is what distinguishes the two, so it is
    reported per face.

    **Every face is processed defensively.** The inventory is advisory — the
    caller's load-bearing output is the mass properties — so one face the kernel
    describes oddly must cost at most that face's row, never the whole
    measurement.
    """
    census: dict = {}
    cylinders: list = []
    skipped = 0
    api_errors: list = []
    for f in part.faces():
        try:
            kind = str(f.geom_type).rsplit(".", 1)[-1]
        except _API_MISUSE as e:
            raise FaceInspectionError("Face.geom_type", e) from e
        except Exception:  # noqa: BLE001 — an undescribable surface
            skipped += 1
            continue
        census[kind] = census.get(kind, 0) + 1
        try:
            center = xyz(f.center())
            if kind == "CYLINDER":
                entry = {"center": qvec(center), "area": q(f.area)}
                radius = face_radius(f)
                if radius is not None:
                    entry["radius"] = q(radius)
                axis = face_axis(f)
                if axis is not None:
                    entry["axis_point"] = qvec(axis[0])
                    entry["axis_dir"] = qvec(axis[1])
                concave = is_concave_cylinder(part, f)
                if concave is not None:
                    entry["concave"] = concave
                cylinders.append(entry)
        except FaceInspectionError as e:
            # An API mismatch, not an awkward surface. Collect rather than
            # raise so one bad face does not cost the whole (advisory)
            # inventory, but make it LOUD in the notes — this is the class of
            # failure that hid the missing bore axes for an entire release.
            api_errors.append(str(e))
            skipped += 1
            continue
        except Exception:  # noqa: BLE001 — one odd face, not the whole table
            skipped += 1
            continue
    if api_errors:
        raise FaceInspectionError(
            f"{len(api_errors)} face inspection(s) failed on an API mismatch "
            f"(first: {api_errors[0]})", RuntimeError("build123d API skew"))
    # Deterministic order: the tables enter the content hash, and OCCT's face
    # iteration order is not contractually stable. Sort on keys every entry has
    # — `radius` is optional.
    cylinders.sort(key=lambda d: (d["center"], d.get("radius") or 0.0))
    return census, cylinders, skipped


def is_concave_cylinder(part, face):
    """True for a bore/seat, False for a shaft/boss, None if undecidable.

    Probes a point between the surface and the axis. If that point is **solid**
    the cylinder is filled with material, i.e. a shaft, boss, or convex fillet
    (concave=False); if it is **void** the material wraps around the outside,
    i.e. a bore, bearing seat, or concave blend (concave=True).

    Getting this backwards is easy and quiet — every face still gets a boolean,
    it is just the wrong one — so the mapping is spelled out here: a hinge bore
    must come back True, and the boss it rides in must come back False.
    """
    axis = face_axis(face)                  # raises FaceInspectionError on skew
    r = face_radius(face)
    if axis is None or r is None:
        return None
    try:
        from build123d import Vector
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_IN
    except ImportError as e:
        raise FaceInspectionError("concavity probe imports", e) from e
    try:
        ap = Vector(*axis[0])
        ad = Vector(*axis[1])
        c = Vector(*xyz(face.center()))
        radial = c - ap
        radial = radial - ad * radial.dot(ad)
        if radial.length <= 1e-12:
            # The face center sits on the axis, so there is no outward
            # direction to probe along. Undecidable, not an error.
            return None
        radial = radial / radial.length
        probe = c - radial * (r * 0.5)      # halfway toward the axis
        cls = BRepClass3d_SolidClassifier(part.wrapped)
        cls.Perform(gp_Pnt(probe.X, probe.Y, probe.Z), 1e-7)
        return cls.State() != TopAbs_IN     # void toward the axis -> a bore
    except _API_MISUSE as e:
        raise FaceInspectionError("BRepClass3d_SolidClassifier probe", e) from e
    except Exception:  # noqa: BLE001 — kernel declined to classify this point
        return None


def shape_table(part) -> tuple:
    """Exact mass properties + the feature inventory for one solid.

    ``inertia_diag`` is build123d's ``matrix_of_inertia`` diagonal, which is
    taken **about the centroid at unit density** (verified: it is invariant
    under translation and matches the closed-form box tensor). ``_primitives``
    relies on exactly that convention — it multiplies by the caller's density
    and places the body at ``center``.

    Two tiers, deliberately: the mass properties are load-bearing (``inertia``
    and the URDF inertial body read them), so a failure there is a real error.
    The feature inventory is advisory, so if it cannot be built the solid is
    still returned with an empty inventory and a note.

    Returns ``(table, aabb_min, aabb_max, notes)``.
    """
    bb = part.bounding_box()
    lo, hi = xyz(bb.min), xyz(bb.max)
    volume = float(part.volume)
    box_volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    M = part.matrix_of_inertia
    notes: list = []
    try:
        census, cylinders, skipped = face_tables(part)
        if skipped:
            notes.append(f"{skipped} face(s) could not be described; the "
                         f"feature inventory is incomplete for this solid")
    except FaceInspectionError as e:
        # An API mismatch against the pinned build123d. Still not fatal — the
        # mass properties are what callers depend on — but it is reported as a
        # HARNESS defect rather than as a property of the author's geometry, so
        # nobody goes bisecting a model over it.
        census, cylinders = {}, []
        notes.append(
            f"HARNESS DEFECT: the feature inventory could not be built because "
            f"a build123d API did not behave as expected ({e}). This is a bug "
            f"in DyCraft or a build123d version skew, NOT a problem with the "
            f"authored geometry — expected build123d "
            f"{REQUIRED_BUILD123D}, found "
            f"{installed_build123d_version() or 'none'}. Mass properties are "
            f"unaffected.")
    except Exception as e:  # noqa: BLE001 — advisory data, never fatal
        census, cylinders = {}, []
        notes.append(f"feature extraction failed ({type(e).__name__}: {e}); "
                     f"mass properties are unaffected")
    table = {
        "volume": q(volume),
        "area": q(part.area),
        "center": qvec(xyz(part.center())),
        "inertia_diag": [q(M[0][0]), q(M[1][1]), q(M[2][2])],
        # A single scalar telling the agent how badly the bounding box
        # over-states this part: ~1.0 is a brick, ~0.2 a tube, ~0.05 a bracket.
        "fill_ratio": q(volume / box_volume) if box_volume > 0.0 else 0.0,
        "face_census": {k: census[k] for k in sorted(census)},
        "topo": {
            "V": len(part.vertices()), "E": len(part.edges()),
            "F": len(part.faces()), "solids": len(part.solids()),
            "shells": len(part.shells()),
        },
        # Only `cylinders` is carried, not the plane inventory: a bore is the
        # feature an author needs to confirm, because `a - b` on non-touching
        # solids silently no-ops and leaves the source looking correct.
        "cylinders": cylinders,
    }
    return table, [q(v) for v in lo], [q(v) for v in hi], notes


# ---------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------


def tessellate(shape, tol: float, ang: float, *, scale_m=None):
    """Triangulate a Shape at a chord tolerance and an angular tolerance.

    Returns ``(verts, faces, groups, normals, uvs)``.

    * `groups` is the cumulative vertex end-offset of each source solid — the
      weld boundary `_geometry_qc` needs.
    * `normals` is one exact B-rep surface normal per **node**, parallel to
      `verts`, or None per entry where the kernel could not supply one.
    * `uvs` is one texture coordinate per **node**, in tile units, or None
      entirely when no material tile size was given.

    Both attribute streams are per node and PRE-WELD, which is the index space
    `RepairedMesh.corners` joins against — see `_shading_per_corner`. A welded
    box corner is one position carrying three different normals, so neither
    stream can be indexed by the surviving vertex.

    `scale_m` is the material's tile size in meters, and passing it is what asks
    for texture coordinates; the normals are computed either way. **This walks
    the faces unconditionally**, untextured parts included, rather than calling
    build123d's own `Shape.tessellate`: the exact normals live on the per-face
    `Poly_Triangulation` that method discards. `_tessellate_faces` reproduces
    that method's traversal exactly and `test_solid_compile_tessellation.py` asserts
    the two agree vertex for vertex, on the imprinted path as well as the plain
    one.

    Solids are tessellated one at a time rather than the whole Shape at once,
    which is what makes those offsets knowable. OCCT triangulates per face
    either way, so concatenating the solids in ``shape.solids()`` order
    reproduces the whole-Shape result vertex for vertex, in the same order.
    That equivalence is the load-bearing claim here — it is what keeps every
    staged digest and every object id still — so it is asserted directly rather
    than left as a remark.

    What it buys is that `_weld_vertices` can refuse to merge across a solid
    boundary, so two bodies authored exactly tangent stay two clean surfaces
    instead of one pinched together along their shared line.

    A shape with no solids is tessellated whole and reports a single group.
    `validate_shape` does not admit one, but `tessellate` is also reached
    through `_wrap_tds` on an imprinted copy, and a contract that holds only for
    the shapes we expect is not a contract.

    Both tolerance arguments are passed positionally and unconditionally. The
    pinned build123d accepts both, so a signature that rejects the second one
    means the interpreter is not the pinned one — a condition the version check
    already refuses. Falling back to a one-argument call instead would silently
    drop ``angular_tolerance``, which changes every triangle and therefore the
    digest of every staged asset.
    """
    # `or [shape]` is the no-solids fallback: one piece, therefore one group.
    pieces = list(shape.solids()) or [shape]
    verts: list = []
    faces: list = []
    groups: list = []
    normals: list = []
    uvs: list | None = [] if scale_m is not None else None
    for piece in pieces:
        base = len(verts)
        piece_v, piece_t, piece_n, piece_uv = _tessellate_faces(
            piece, tol, ang, None if scale_m is None else float(scale_m))
        verts.extend(piece_v)
        normals.extend(piece_n)
        if uvs is not None:
            uvs.extend(piece_uv)
        faces.extend((a + base, b + base, c + base) for a, b, c in piece_t)
        groups.append(len(verts))
    return verts, faces, groups, normals, uvs


def _tessellate_faces(piece, tol: float, ang: float, scale_m):
    """`Shape.tessellate`, reimplemented to keep what it throws away.

    Returns ``(verts, faces, normals, node_uv)`` with LOCAL vertex indices;
    `node_uv` is None when `scale_m` is.

    This mirrors build123d 0.11.1's `Shape.tessellate` exactly — `mesh()`, then
    `self.faces()`, `BRep_Tool.Triangulation_s`, nodes transformed by the face's
    location, triangle winding reversed for a `TopAbs_REVERSED` face, one
    running vertex offset. It has to: `test_solid_compile_tessellation.py`
    asserts the two agree vertex for vertex, so the attributes below are the
    only difference between this mesh and the one that method returns.

    The reimplementation exists because `Shape.tessellate` returns only points
    and indices, while the `Poly_Triangulation` it reads carries `UVNode(i)` for
    every node — OCCT has already computed the surface parameters and they are
    thrown away one line later. **Both attributes want that same read**: the UV
    mapper converts the parameter to meters along the surface, and the exact
    normal is `D1` evaluated at it. So the node is read once and serves both,
    which is why this is one walk and not two.

    **The reversed-winding flip is the part to be careful with.** `_wrap_tds`
    warns that a second copy of that rule is how a normal ends up inside-out on
    imprinted parts only; a UV bound to the wrong corner is the same class of
    fault, silent and visible only in a render.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_Orientation
    from OCP.TopLoc import TopLoc_Location

    piece.mesh(tol, ang)
    verts: list = []
    faces: list = []
    normals: list = []
    node_uv: list | None = [] if scale_m is not None else None
    offset = 0
    for face in piece.faces():
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
        if poly is None:
            # build123d would raise here. A face OCCT declined to triangulate
            # contributes no geometry, and refusing the whole solid over it
            # would be a harsher rule than the pre-merge untextured path
            # applied. Skipping without advancing `offset` keeps every later
            # triangle index correct.
            continue
        trsf = loc.Transformation()
        reverse = (face.wrapped.Orientation()
                   == TopAbs_Orientation.TopAbs_REVERSED)
        adaptor = BRepAdaptor_Surface(face.wrapped)
        has_uv = poly.HasUVNodes()
        pts = [xyz(poly.Node(i).Transformed(trsf))
               for i in range(1, poly.NbNodes() + 1)]
        tris = []
        for t in poly.Triangles():
            a, b, c = t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1
            tris.append((a, c, b) if reverse else (a, b, c))

        fn = None
        if node_uv is not None:
            needs_uv, fn = _face_uv_mapper(face, _face_mean_normal(pts, tris))
            if needs_uv and not has_uv:
                # A curved face whose triangulation carries no parameters
                # cannot be mapped through its surface. Degrade to the
                # projection rather than emitting a constant, which would
                # paint the whole face one texel.
                fn = _projection_only(pts, tris)

        for i, p in enumerate(pts):
            if has_uv:
                node = poly.UVNode(i + 1)
                u, v = float(node.X()), float(node.Y())
                normals.append(_surface_normal(adaptor, u, v, reverse))
            else:
                # No parameters, so `D1` has nothing to evaluate at. The OBJ
                # writer substitutes the triangle's geometric normal.
                u = v = 0.0
                normals.append(None)
            if node_uv is not None:
                s, t = fn(p, u, v)
                node_uv.append((s / scale_m, t / scale_m))

        verts.extend(pts)
        faces.extend((a + offset, b + offset, c + offset) for a, b, c in tris)
        offset += poly.NbNodes()
    return verts, faces, normals, node_uv


def _surface_normal(surface, u: float, v: float, reverse: bool):
    """Exact outward unit normal of `surface` at parameter (u, v), or None.

    The normal is ``dP/du x dP/dv`` evaluated on the real B-rep surface, not
    inferred from triangles. `surface` is a BRepAdaptor_Surface: its D1
    derivatives already include the face's location. Unlike triangulation
    nodes, they must NOT be transformed by that location a second time.

    `reverse` mirrors the winding flip below: a face whose topological
    orientation is REVERSED has its triangles emitted in swapped order, so its
    geometric normal must be negated to keep pointing out of the material.
    Getting that wrong is silent — every facet still gets a normal, it just
    faces inward and the surface renders black — so the equivalence test asserts
    every triangle's winding against its own exact normal.

    None where the cross product vanishes (a cone apex, a sphere pole). The OBJ
    writer substitutes the triangle's geometric normal there.
    """
    from OCP.gp import gp_Pnt, gp_Vec

    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(u, v, point, du, dv)
    n = du.Crossed(dv)
    magnitude = n.Magnitude()
    if magnitude <= 0.0:
        return None
    n = gp_Vec(n.X() / magnitude, n.Y() / magnitude,
               n.Z() / magnitude)
    return (-n.X(), -n.Y(), -n.Z()) if reverse else (n.X(), n.Y(), n.Z())


# ---------------------------------------------------------------------------
# Texture coordinates
#
# UVs are derived HERE, at staging, and written into the OBJ — not recomputed by
# each renderer. Two reasons. A CAD solid's tessellation discards the B-rep face
# structure, so this is the only place that still knows a barrel is a cylinder
# rather than a strip of unrelated triangles; and an imported glTF/URDF asset
# already arrives with UVs baked into its OBJ, so baking these makes both kinds
# of asset one kind of thing downstream.
#
# Every mapper below returns MeterS along the surface. The caller divides by the
# material's tile size, so "one tile spans scale_m of real surface" is true by
# construction rather than per-surface arithmetic.
#
# One thing happens AFTER all of this, in `compile_entry`: each part's finished
# coordinates are shifted by a deterministic per-part tile phase, so that two
# parts with the same shape and the same material stop sampling an identical
# patch of it. See `part_uv_phase`. Nothing in this section knows about it, and
# that is the point — the phase is one constant added to a finished lattice, not
# a per-face decision.
# ---------------------------------------------------------------------------

#: Axis pair to project onto, per dominant axis, for the fallback: dominant
#: X -> (Y, Z), dominant Y -> (X, Z), dominant Z -> (X, Y). These are the axes
#: of the PART frame, not the world's — the nodes reaching the mappers are the
#: face's own, transformed only by the face location, so the pattern is fixed
#: to the part rather than to where the part stands. The same table
#: `render/open3d_renderer._triplanar_uvs` uses, and for the same reason, so a
#: face this cannot parameterize looks the way it did before UVs were baked.
_TRIPLANAR_AXES = ((1, 2), (0, 2), (0, 1))


def _dot(p, d) -> float:
    return p[0] * d[0] + p[1] * d[1] + p[2] * d[2]


def _face_mean_normal(pts, tris):
    """Area-weighted normal of a face's triangles, or None if it has no area."""
    nx = ny = nz = 0.0
    for (a, b, c) in tris:
        pa, pb, pc = pts[a], pts[b], pts[c]
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        # Not normalized: the cross product's magnitude is twice the triangle's
        # area, which is exactly the weight a mean normal wants.
        nx += uy * vz - uz * vy
        ny += uz * vx - ux * vz
        nz += ux * vy - uy * vx
    return None if (nx, ny, nz) == (0.0, 0.0, 0.0) else (nx, ny, nz)


def _face_uv_mapper(face, mean_normal):
    """``(needs_surface_uv, fn)`` mapping a face's nodes to surface meters.

    `fn(point, u, v) -> (s, t)`, where `point` is the node in the PART frame and
    `(u, v)` are OCCT's own surface parameters for it. `needs_surface_uv` says
    whether `fn` reads them, so the caller can fall back when a face carries no
    UV nodes.

    **A plane is anchored at the part origin, not at its own parametric
    origin.** OCCT gives every planar face `(u, v)` in meters already, which is
    tempting — but each face's origin is its own, and `imprint_planar_grids`
    splits one panel into a grid of subfaces. Using the native parameters would
    restart the tile lattice at every cell boundary. Taking the plane's AXES
    from the B-rep and the ORIGIN from the part keeps one lattice across every
    coplanar subface, and across parallel faces too, which is what a tiled
    material means by "continuous".

    The curved surfaces do the opposite, and for the opposite reason: their
    parameterization is the thing positions cannot reconstruct. A cylinder's `u`
    is an angle, so `r * u` is arc length and the grain wraps around the barrel
    instead of being cut into three flat projections with two seams.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType as _T

    ad = BRepAdaptor_Surface(face.wrapped)
    kind = ad.GetType()

    if kind == _T.GeomAbs_Plane:
        ax = ad.Plane().Position()
        xd, yd = xyz(ax.XDirection()), xyz(ax.YDirection())
        return False, lambda p, u, v: (_dot(p, xd), _dot(p, yd))

    if kind == _T.GeomAbs_Cylinder:
        r = float(ad.Cylinder().Radius())
        return True, lambda p, u, v: (r * u, v)

    if kind == _T.GeomAbs_Cone:
        cone = ad.Cone()
        r0, half = float(cone.RefRadius()), float(cone.SemiAngle())
        # The radius grows along the generating line, so the arc length a given
        # angle spans grows with it. Using a single radius would stretch the
        # pattern toward the wide end.
        return True, lambda p, u, v: ((r0 + v * math.sin(half)) * u, v)

    if kind == _T.GeomAbs_Sphere:
        r = float(ad.Sphere().Radius())
        # Equirectangular, and knowingly distorted toward the poles — no
        # isometric map of a sphere exists. On DyCraft parts a sphere is a
        # fillet's corner patch, a few millimeters across, where the error is
        # far below one texel.
        return True, lambda p, u, v: (r * u * math.cos(v), r * v)

    if kind == _T.GeomAbs_Torus:
        tor = ad.Torus()
        maj, minr = float(tor.MajorRadius()), float(tor.MinorRadius())
        return True, lambda p, u, v: ((maj + minr * math.cos(v)) * u, minr * v)

    # Anything else — a B-spline patch, a surface of revolution, an offset —
    # gets a triplanar projection along the face's dominant normal axis. Per
    # face rather than per triangle, so one patch keeps one parameterization.
    axis = 2
    if mean_normal is not None:
        axis = max(range(3), key=lambda i: abs(mean_normal[i]))
    ua, va = _TRIPLANAR_AXES[axis]
    return False, lambda p, u, v: (p[ua], p[va])


def _projection_only(pts, tris):
    """The dominant-axis projection, for a face whose surface cannot serve."""
    normal = _face_mean_normal(pts, tris)
    axis = 2 if normal is None else max(range(3), key=lambda i: abs(normal[i]))
    ua, va = _TRIPLANAR_AXES[axis]
    return lambda p, u, v: (p[ua], p[va])


#: Salt for the per-part tile phase. It is in the hash so that "which phase does
#: this part get" is a property of this module's version rather than of the bare
#: name, and so changing it re-phases the corpus deliberately rather than by
#: accident. Bumping it rewrites every textured part's OBJ bytes and therefore
#: every textured object's id, so it is a knob, not an implementation detail.
_UV_PHASE_SALT = "dycraft.uv.phase.v1"


def part_uv_phase(name: str) -> tuple:
    """Deterministic ``(du, dv)`` tile phase for one part, each in [0, 1).

    **Why a part needs a phase at all.** The bake is a pure function of the
    part's local geometry and its tile size, so two parts with the same shape
    and the same material produce the same `vt` stream — and because staged
    assets are content-addressed, they then collapse into ONE OBJ file. A
    bench's ten identical slats would each show a pixel-identical patch of grain,
    and the seat would read as printed wallpaper rather than as ten boards.

    **Keyed on the name, and deliberately not on the material.** The name is
    unique within an object (`object_builder` validates it) and stable across
    recompiles, which is the whole requirement: the same source must stage the
    same bytes. Folding the texture's digest in would look tempting and cost
    real work — an agent re-describes a surface several times in a run, and each
    new digest would rebake every mesh in the object for no visual gain.
    Excluding it also makes the phase reconstructible from the IR, so
    nothing has to be stored to explain why two identical parts differ.

    **In tile units, and unbounded on purpose.** The maps tile, and every
    consumer wraps (`_blender_child` sets `extension = "REPEAT"`; Open3D and
    three.js use repeat addressing), so a phase is only meaningful modulo 1 and
    UVs already run well outside [0, 1] on any part more than one tile across.
    Nothing needs clamping.

    **What this does NOT fix.** A scene places N instances of one ObjectIR, and
    they share its baked meshes — so twenty copies of one tree still carry one
    phase. Per-instance variation cannot be baked into a shared mesh and would
    need a different mechanism; this fixes cloning WITHIN an object only.
    """
    digest = hashlib.blake2b(
        f"{_UV_PHASE_SALT}\n{name}".encode("utf-8"), digest_size=8).digest()
    scale = float(1 << 32)
    return (int.from_bytes(digest[:4], "big") / scale,
            int.from_bytes(digest[4:], "big") / scale)


def _phase_corner_uv(corner_uv, name: str):
    """`corner_uv` shifted by the part's tile phase, or None unchanged.

    **One shift for the WHOLE part, never per face, and that is the invariant
    this function exists to hold.** `_face_uv_mapper` anchors a plane at the
    part origin precisely so that one tile lattice runs across every coplanar
    subface `imprint_planar_grids` cuts, and across parallel faces too. A phase
    chosen per face would restart the lattice at every cell boundary — exactly
    the artifact that anchoring was written to avoid. Adding one constant to
    every corner of every face shifts the whole lattice together, so the
    continuity survives by construction.

    Faces are shifted in their own parameter space, which is the right reading
    of "one lattice": coplanar and parallel faces share both the plane axes and
    the part origin and so move together, while perpendicular faces never had
    continuity across the corner between them.
    """
    if corner_uv is None:
        return None
    du, dv = part_uv_phase(name)
    # A corner may carry no UV at all, and that is not the same as `corner_uv`
    # being None. `_shading_per_corner` returns one triple per triangle whose
    # entries are None when the tessellation carried no parameters — an
    # untextured part (`scale_m is None`) comes back as a full-length list of
    # (None, None, None). `_geometry_qc._uv_rows` reads that as "write no `vt`
    # at all" under its all-or-nothing rule, so the shift has to preserve it:
    # phasing None would crash, and substituting (du, dv) would manufacture a
    # coordinate for a face that has none and defeat that rule. Leaving them
    # alone is also what keeps an untextured part's OBJ bytes free of any phase,
    # which is the promise this function opened with.
    def _shift(c):
        return c if c is None else (c[0] + du, c[1] + dv)
    return [(_shift(a), _shift(b), _shift(c)) for a, b, c in corner_uv]
#: How many offending sub-shapes to name before summarizing. A defective solid
#: is usually one face, but can carry many.
_BREP_CHECK_MAX_ROWS = 4


def _brep_check_detail(shape) -> str:
    """What OCCT objects to, localised, as a sentence to append to the error.

    `Shape.is_valid` is a bool; `BRepCheck_Analyzer` knows which face, which
    status, and the wires on it. Typically the solid and the shell report
    `NoError` and **exactly one face** carries the defect, often with an outer
    wire of meters plus parasitic wires of a fraction of a millimeter — the
    residue of a boolean whose operand grazed the face, which nobody modeled.
    Naming the face and its wire lengths turns an unlocalisable failure into a
    coordinate the author can find in their own source.

    Best-effort by construction: this runs on an error path, so anything it
    cannot determine it simply omits. An empty string is a valid answer, and
    is what a solid gets that OCCT calls invalid while reporting `NoError` on
    every face, wire, edge, and the shell.
    """
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.TopAbs import TopAbs_ShapeEnum
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except BaseException:  # noqa: BLE001 — no kernel introspection available
        return ""
    try:
        analyzer = BRepCheck_Analyzer(shape.wrapped)
        rows = []
        explorer = TopExp_Explorer(shape.wrapped, TopAbs_ShapeEnum.TopAbs_FACE)
        index = 0
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            if not analyzer.IsValid(face):
                rows.append((index, _face_defect_summary(face, analyzer)))
            index += 1
            explorer.Next()
        if not rows:
            return (" OCCT reports no specific defect on any face, wire or "
                    "edge, so it objects to the solid as a whole and cannot "
                    "say where.")
        shown = rows[:_BREP_CHECK_MAX_ROWS]
        parts = "; ".join(f"face #{i} {detail}" for i, detail in shown)
        more = ("" if len(rows) <= len(shown)
                else f", and {len(rows) - len(shown)} more face(s)")
        return (f" OCCT localises it to {len(rows)} of {index} faces: "
                f"{parts}{more}.")
    except BaseException:  # noqa: BLE001 — never lose the error to its detail
        return ""


def _face_defect_summary(face, analyzer=None) -> str:
    """One face's OCCT status and wire inventory.

    **Do not measure wires with `GCPnts_AbscissaPoint` here.**
    `GCPnts_AbscissaPoint.Length_s(BRepAdaptor_CompCurve(wire))` **segfaults**
    on a wire OCCT has already flagged as pathological (`IntersectingWires`,
    `UnorientableShape`), typically the parasitic micro-wire this diagnosis
    exists to name. A segfault is not catchable, so the `except BaseException`
    around this is no protection: the whole worker dies and the author gets
    `worker_crash` with no part name and no diagnosis. build123d's own
    `Wire.length` measures the same wires, with the same numbers, without
    crashing.

    The status comes first because it is the most diagnostic field:
    `IntersectingWires` and `UnorientableShape` call for different repairs,
    and the wire lengths only hint at which.
    """
    bits = []
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        bits.append(str(BRepAdaptor_Surface(face).GetType()).rsplit("_", 1)[-1])
    except BaseException:  # noqa: BLE001
        pass
    if analyzer is not None:
        try:
            status = analyzer.Result(face).Status()
            names = [str(s).rsplit("_", 1)[-1] for s in status]
            named = [n for n in names if n and n != "NoError"]
            if named:
                bits.append("/".join(dict.fromkeys(named)))
        except BaseException:  # noqa: BLE001 — status is a bonus, not the point
            pass
    try:
        import build123d as bd

        lengths = sorted((float(w.length) * 1000.0
                          for w in bd.Face(face).wires()), reverse=True)
    except BaseException:  # noqa: BLE001
        lengths = []
    if lengths:
        outer, rest = lengths[0], lengths[1:]
        if rest:
            inner = ", ".join(f"{v:.2f}" for v in rest[:4])
            tail = "" if len(rest) <= 4 else f", +{len(rest) - 4} more"
            bits.append(f"outer wire {outer:.1f} mm plus {len(rest)} wire(s) "
                        f"of {inner} mm{tail}")
        else:
            bits.append(f"one wire {outer:.1f} mm")
    return f"({', '.join(bits)})" if bits else "(unreadable)"


def tessellate_within_budget(shape, name: str, tol: float, ang: float, *,
                             scale_m=None):
    """Tessellate, coarsening until the triangle count is under budget.

    ``tolerance`` is **relative, not meters**: build123d calls
    ``BRepMesh_IncrementalMesh(shape, tol, isRelative=True, ang, parallel)``, so
    the chord deflection OCCT enforces is ``tol`` times the local feature size.
    One value therefore serves every scale identically — measured, a 0.01 m and
    a 1 m sphere both land on 10 108 triangles at ``tol=1e-3`` — which is also
    why the guard is needed: the dial is a dimensionless fineness, so a value
    chosen to "compensate" for a small part just multiplies the triangle count
    (1e-4 passes 100k, a 3.5 MB OBJ, per part). Since the OBJ is only ever the
    render/URDF visual — the exact ``.brep`` is what geometry questions are
    answered from — trading visual precision for a bounded asset is right.

    Neither dial reaches a **planar** face: chord and angular error are both
    zero there at any tolerance, so OCCT triangulates it from its boundary
    alone. A pure box is 24 vertices / 12 triangles from 1e-3 down to 1e-6.
    Interior vertices on a flat face come from B-rep topology (more faces), not
    from here.

    Retrying **requires** dropping OCCT's cached triangulation first: a
    triangulation is stored on the ``TopoDS_Shape``, so a second ``tessellate``
    with a coarser tolerance silently returns the original fine mesh.

    Two properties the ladder holds:

    * **Monotone in the request.** The rungs start at
      ``max(request, DEFAULT_TOLERANCE)``, so asking for a finer mesh than the
      default can never deliver a coarser one. Rungs anchored on the request
      would take a shape over budget at ``tol=1e-4`` through 4e-4, 1.6e-3, ...
      and land it coarser than the same shape at ``tol=1e-3``.
    * **No attempt spent on a no-op.** Past some fineness the chord dial is not
      what binds — ``angular_tolerance`` is — so a x4 rung can return the
      identical mesh, and for some shapes (a torus, a filleted box) that is the
      first and most expensive rung. Such rungs do not count against
      ``COARSEN_ATTEMPTS``.

    What the ladder does **not** do is land near the budget: the count is a
    step function of the dial for a given shape (a torus may settle at 60% of
    the budget with the next step finer over it), so the headroom is the
    geometry's, not the search's.
    """
    # Probe at the DEFAULT first when the request is finer than it, and only
    # try the request if the default leaves room. The requested tolerance gives
    # the most expensive mesh the part will ever produce, thrown away whenever
    # it blows the budget: a torus at `tolerance=1e-5` takes about two minutes
    # where the default takes under a second. Against `_timelimit`'s build bound
    # that is not a slow compile but a lost one, and the author who asked for
    # more fidelity would get less.
    requested = tol
    tol = max(tol, DEFAULT_TOLERANCE)
    try:
        verts, faces, groups, normals, uvs = tessellate(
            shape, tol, ang, scale_m=scale_m)
    except BaseException as e:  # noqa: BLE001 — becomes a structured kind
        # A real OCCT triangulation failure, raised as its own kind: let
        # through, it would be classified as a generic `GeometryKernelError`
        # and the author would get advice about operands rather than about the
        # surface that would not mesh.
        raise SolidCompileError(
            "tessellation_failure",
            f"OCCT could not triangulate solid {name!r}: "
            f"{type(e).__name__}: {e}") from None
    if not verts or not faces:
        raise SolidCompileError(
            "empty_mesh",
            f"solid {name!r} tessellated to {len(verts)} vertices / "
            f"{len(faces)} triangles; both must be > 0")
    if len(faces) <= MAX_TRIANGLES:
        if requested >= tol:
            return verts, faces, groups, normals, uvs, tol
        # The default fits. Is the finer request even reachable? Triangle
        # count only grows as the tolerance tightens, so an optimistic
        # estimate that already exceeds the budget settles it without paying
        # for the mesh.
        predicted = len(faces) * (tol / requested) ** COARSEN_GROWTH_EXPONENT
        if predicted > MAX_TRIANGLES:
            return verts, faces, groups, normals, uvs, tol
        try:
            from OCP.BRepTools import BRepTools
            BRepTools.Clean_s(shape.wrapped)
            v2, f2, g2, n2, u2 = tessellate(
                shape, requested, ang, scale_m=scale_m)
        except BaseException:  # noqa: BLE001 — keep the mesh that worked
            return verts, faces, groups, normals, uvs, tol
        if v2 and f2 and len(f2) <= MAX_TRIANGLES:
            return v2, f2, g2, n2, u2, requested
        return verts, faces, groups, normals, uvs, tol

    try:
        from OCP.BRepTools import BRepTools
    except BaseException:  # noqa: BLE001 — without Clean_s a retry is a no-op
        return verts, faces, groups, normals, uvs, tol
    # `tol` is already `max(request, DEFAULT_TOLERANCE)`, which is what makes
    # the ladder monotone in the request: an author who asks for more fidelity
    # must not get less.
    attempts = 0
    # The ceiling is anchored on the DEFAULT, not on `tol`. `tol` is already
    # `max(request, DEFAULT_TOLERANCE)`, so anchoring here too would let a
    # *coarser*-than-default request widen the ladder's own reach by the same
    # factor: `tolerance=0.04` would set the ceiling at `0.04 * 64 = 2.56`, far
    # enough to come back visibly faceted, and the author's only remedy would
    # be to give up the coarser mesh they wanted in order to stop the ladder.
    #
    # The `max()` above and this constant are the same rule seen from both
    # ends: the request may move where the ladder STARTS, never how far it
    # may go. The reachable ceiling is therefore always
    # `DEFAULT_TOLERANCE * COARSEN_MAX_FACTOR` = 0.064, whatever was asked
    # for. A shape that cannot fit under it is staged over budget instead,
    # which is this module's stated preference — an over-budget mesh costs
    # disk, a ruined one costs the object.
    #
    # Consequence worth knowing: a request coarser than 0.016 leaves no rung
    # below the ceiling, so the ladder does not run for it at all. That is
    # the intended reading of the cap as an absolute quality floor, not an
    # oversight — asking for a coarse mesh is allowed, asking for license to
    # destroy the part is not.
    tol_floor = DEFAULT_TOLERANCE
    while attempts < COARSEN_ATTEMPTS:
        if tol * COARSEN_FACTOR > tol_floor * COARSEN_MAX_FACTOR * 1.0001:
            # Coarsening further would cost more than the budget is worth.
            break
        prev_count = len(faces)
        tol *= COARSEN_FACTOR
        BRepTools.Clean_s(shape.wrapped)   # else OCCT returns the cached mesh
        try:
            v2, f2, g2, n2, u2 = tessellate(
                shape, tol, ang, scale_m=scale_m)
        except BaseException:  # noqa: BLE001 — keep the last good tessellation
            break
        if not (v2 and f2):
            attempts += 1
            continue
        # Normals, UVs, and groups all travel with the mesh they describe:
        # a coarser retry renumbers every vertex and re-triangulates every
        # face, so keeping any of them from the previous pass would bound
        # the weld at the wrong places and attach one tessellation's
        # shading to another's triangles.
        verts, faces, groups, normals, uvs = v2, f2, g2, n2, u2
        if len(faces) <= MAX_TRIANGLES:
            break
        if len(faces) > prev_count * (1.0 - COARSEN_MIN_PROGRESS):
            # A dead or near-dead rung: this tolerance produced materially the
            # mesh we already had, because `angular_tolerance` — not the chord
            # dial — is what is binding at this fineness. Charging it as an
            # attempt burns a quarter of the ladder on a no-op, and the no-op
            # is common, often the first and most expensive rung. Multiply
            # again instead — there is nothing to lose, since the mesh in hand
            # is the one this rung produced.
            continue
        attempts += 1
    return verts, faces, groups, normals, uvs, tol


# ---------------------------------------------------------------------------
# Planar grid imprinting  (`planar_max_edge`)
# ---------------------------------------------------------------------------

def _faces_of(tds):
    """Every TopoDS_Face of a raw TopoDS_Shape, in exploration order."""
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(tds, TopAbs_ShapeEnum.TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current()))
        ex.Next()
    return out


def _wrap_tds(tds):
    """Wrap a raw TopoDS_Shape back into build123d so `tessellate` applies.

    Going back through build123d rather than re-deriving the triangulation here
    keeps one implementation of the per-face orientation flip: `Shape.tessellate`
    reverses the winding of a REVERSED face, and a second copy of that rule is
    how a normal ends up inside-out on imprinted parts only.
    """
    from build123d import Compound, Solid
    from OCP.TopAbs import TopAbs_ShapeEnum
    if tds.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
        return Compound(tds)
    return Solid(tds)


def _plane_key(pl) -> tuple:
    """Identity of a face's underlying plane: unit normal + signed distance.

    Used to carry a plan from pass 1 to pass 2, where the original face handles
    are stale. Splitting a face preserves both the surface and its orientation,
    so the sub-faces key back to the same plan without canonicalizing the normal
    sign — and two genuinely distinct coplanar, co-oriented faces sharing a plan
    is harmless, because the plan is the same grid either way.
    """
    ax = pl.Axis()
    d, o = ax.Direction(), ax.Location()
    n = (d.X(), d.Y(), d.Z())
    dist = n[0] * o.X() + n[1] * o.Y() + n[2] * o.Z()
    return (round(n[0], 9), round(n[1], 9), round(n[2], 9), round(dist, 9))


def _planar_plan(face, max_edge):
    """`(key, point_fn, uv_bounds, n_u, n_v)` for a planar face needing a grid.

    ``None`` for a non-planar face — chord and angular tolerance already govern
    those, and a line laid across a curved surface does not lie on it — and for
    a planar face whose own diagonal already fits inside ``max_edge``, so a 2 mm
    chamfer flat is not subdivided for nothing.

    ``n_u``/``n_v`` come from the cell **diagonal**, not the cell side: the
    longest edge of a triangle in a u x v cell is ``hypot(u, v)``, so each side
    gets ``max_edge / sqrt(2)`` and the guarantee holds on a non-square face.

    A planar face's UV parameters are meters along its own plane axes, so this
    is orientation-independent — a solid rotated by an arbitrary angle reports
    the same extents as the axis-aligned one.
    """
    import math as _m
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_SurfaceType

    ad = BRepAdaptor_Surface(face)
    if ad.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
        return None
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    du, dv = u1 - u0, v1 - v0
    if not (du > 0.0 and dv > 0.0) or _m.hypot(du, dv) <= max_edge:
        return None

    side = max_edge / _m.sqrt(2.0)
    n_u = min(PLANAR_IMPRINT_MAX_N, max(1, int(_m.ceil(du / side))))
    n_v = min(PLANAR_IMPRINT_MAX_N, max(1, int(_m.ceil(dv / side))))
    if n_u < 2 and n_v < 2:
        return None

    pos = ad.Plane().Position()
    o, xd, yd = pos.Location(), pos.XDirection(), pos.YDirection()

    def point(u, v):
        return (o.X() + u * xd.X() + v * yd.X(),
                o.Y() + u * xd.Y() + v * yd.Y(),
                o.Z() + u * xd.Z() + v * yd.Z())

    return _plane_key(ad.Plane()), point, (u0, u1, v0, v1), n_u, n_v


def imprint_planar_grids(shape, max_edge: float):
    """Imprint a grid on every planar face too coarse for ``max_edge``.

    Returns ``(TopoDS_Shape, n_faces_imprinted)``. The shape is always a
    **copy**, so the caller's live Shape — and therefore the exact ``.brep``
    serialized from it and every measurement taken off it — is untouched. This
    is deliberate: the imprint exists to give the render mesh interior vertices
    (that is what a deformation rig binds to), not to change the geometry the
    author committed. ``n_faces_imprinted == 0`` means nothing needed it.

    **Two passes per face, one face at a time.** Neither half of that is
    optional, and both were measured on a 115x58x55 box:

    * One `BRepFeat_SplitShape` handed both line families does not intersect
      them with each other — the face count grows by n-1 and the longest edge
      stays at 55 mm. So the second family goes on separately, onto the faces
      the first pass produced, where nothing competes with it.
    * Batching all six faces into one pair of passes silently drops some of the
      second family: five faces came out at the target and -Y stayed at 55 mm.
      So each face gets its own pair of `Build()` calls, with the shape rebuilt
      and its sub-faces re-found by plane key in between. It costs about 10 s on
      a six-faced box at 4 mm, which buys the guarantee actually holding.

    What bounds the triangles is **boundary vertex density**, not the grid of
    sub-faces: pass 2 splits only some strips into cells, but on the rest its
    lines still cut the long boundary edges into n segments, and `BRepMesh`
    discretizes a face's boundary before Delaunay-triangulating inside it. Every
    sub-face — cells and leftover strips alike — comes out at the same longest
    edge, `hypot(du/n_u, dv/n_v)`.

    Never raises: any OCCT refusal returns the shape as imprinted so far and the
    count reached. A solid that cannot be imprinted is still a good solid, and
    the caller reports the shortfall rather than failing the part over it.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeEdge
    from OCP.BRepFeat import BRepFeat_SplitShape
    from OCP.gp import gp_Pnt

    tds = BRepBuilderAPI_Copy(shape.wrapped).Shape()

    def seg(p0, p1):
        return BRepBuilderAPI_MakeEdge(gp_Pnt(*p0), gp_Pnt(*p1)).Edge()

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType

    from OCP.BRepTools import BRepTools

    def within(t, key, bounds, pad=1e-7):
        """Faces of `t` in plane `key` whose UV box sits inside `bounds`.

        Two things at once. The handles from the previous `Build()` belong to the
        shape it replaced, so sub-faces have to be re-found — that is what the
        plane key does. But keying on the plane ALONE conflates two distinct
        faces that happen to be coplanar, and then one face's grid lines, derived
        from its own UV bounds, do not lie on the other face and OCCT drops them.
        Measured on a U-shaped part whose notch splits the top into two coplanar
        faces: one came out at 3.93 mm and the other stayed at 50.0 mm against a
        4 mm request, silently. The UV containment test is what separates them.
        """
        u0, u1, v0, v1 = bounds
        out = []
        for f in _faces_of(t):
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
                continue
            if _plane_key(ad.Plane()) != key:
                continue
            a0, a1, b0, b1 = BRepTools.UVBounds_s(f)
            if (a0 >= u0 - pad and a1 <= u1 + pad
                    and b0 >= v0 - pad and b1 <= v1 + pad):
                out.append(f)
        return out

    # One plan per FACE, not per plane. The `within` docstring says why.
    plans = [p for p in (_planar_plan(f, max_edge) for f in _faces_of(tds))
             if p is not None]
    if not plans:
        return tds, 0

    done = 0
    for key, point, (u0, u1, v0, v1), n_u, n_v in plans:
        bounds = (u0, u1, v0, v1)
        try:
            sp = BRepFeat_SplitShape(tds)
            for f in within(tds, key, bounds):
                for i in range(1, n_u):
                    u = u0 + (u1 - u0) * i / n_u
                    sp.Add(seg(point(u, v0), point(u, v1)), f)
            sp.Build()
            mid = sp.Shape()

            sp2 = BRepFeat_SplitShape(mid)
            for f in within(mid, key, bounds):
                for j in range(1, n_v):
                    v = v0 + (v1 - v0) * j / n_v
                    sp2.Add(seg(point(u0, v), point(u1, v)), f)
            sp2.Build()
            tds = sp2.Shape()
            done += 1
        except BaseException:  # noqa: BLE001 — keep the faces already imprinted
            break
    return tds, done


def tessellate_with_planar_imprint(shape, name: str, tol: float, ang: float,
                                   max_edge: float, *, scale_m=None):
    """`tessellate_within_budget` over a grid-imprinted copy of ``shape``.

    Returns ``(verts, faces, groups, normals, uvs, tol_used, max_edge_used,
    n_faces_imprinted, max_edge_achieved)`` — the last one **measured**, not
    assumed. `groups` bounds the weld and comes from the IMPRINTED copy, which
    is the mesh it describes; imprinting splits faces and never solids, so the
    offsets mean the same thing they would have before.

    `normals` and `uvs` likewise describe the imprinted mesh. This is the case the
    part-local plane anchoring exists for: the imprint replaces one panel with a
    grid of coplanar subfaces, and a parameterization taken from each subface's
    own parametric origin would restart the tile lattice at every cell.

    The budget is defended by **doubling the requested edge**, not by coarsening
    ``tolerance``: planar triangles come from topology, so the chord ladder
    cannot rescue an over-budget imprint — it would spend four attempts wrecking
    the curved surfaces and come back over budget anyway. Only once the imprint
    itself has been backed off ``PLANAR_COARSEN_ATTEMPTS`` times does the
    ordinary tolerance ladder run, so a part that is over budget for reasons
    unrelated to the imprint still gets its usual treatment.
    """
    # Choose the target BEFORE imprinting, from arithmetic. Imprinting is the
    # expensive step and its cost grows as n_u*n_v per face — a 0.5 mm request on
    # a 115 mm block asks for 64x64 (the per-face cap) on six faces, which is
    # ~24k `Add` calls that the budget was always going to reject. Discovering
    # that by trial would pay for the worst grid first and then three more.
    baseline = len(tessellate(shape, tol, ang)[1])   # [1] is `faces`
    target = float(max_edge)
    for _ in range(PLANAR_COARSEN_ATTEMPTS):
        if baseline + _projected_imprint_triangles(shape, target) <= MAX_TRIANGLES:
            break
        target *= 2.0

    tds, n_imprinted = imprint_planar_grids(shape, target)
    verts, faces, groups, normals, uvs = tessellate(
        _wrap_tds(tds), tol, ang, scale_m=scale_m)
    if len(faces) <= MAX_TRIANGLES:
        return (verts, faces, groups, normals, uvs, tol, target, n_imprinted,
                worst_planar_edge(tds))
    # The projection was optimistic (it cannot know what the curved faces cost
    # once the flats stop dominating). Only now is the chord ladder the right
    # tool: what is left over is not the grid's doing.
    verts, faces, groups, normals, uvs, tol_used = tessellate_within_budget(
        _wrap_tds(tds), name, tol, ang, scale_m=scale_m)
    return (verts, faces, groups, normals, uvs, tol_used, target, n_imprinted,
            worst_planar_edge(tds))


def worst_planar_edge(tds) -> float | None:
    """Longest triangle edge over the **planar** faces of an already-meshed shape.

    Read back rather than trusted. Every silent failure this parameter has had
    looked like a success from the request side — a plan computed, a `Build()`
    that returned, a face count that went up — while one face stayed at its
    original fan. So the achieved number is measured and reported, and
    `n_faces_imprinted` alone is never the evidence that it worked.

    ``None`` when no planar face carries a triangulation. Call after
    `tessellate`, which leaves the triangulation attached to the faces.
    """
    import math as _m
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopLoc import TopLoc_Location

    worst = None
    for f in _faces_of(tds):
        if BRepAdaptor_Surface(f).GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
            continue
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        pts = [tri.Node(i).Transformed(trsf)
               for i in range(1, tri.NbNodes() + 1)]
        for k in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(k)
            a, b, c = (pts[t.Value(1) - 1], pts[t.Value(2) - 1],
                       pts[t.Value(3) - 1])
            for p, q in ((a, b), (b, c), (c, a)):
                e = _m.dist((p.X(), p.Y(), p.Z()), (q.X(), q.Y(), q.Z()))
                worst = e if worst is None else max(worst, e)
    return worst


def _projected_imprint_triangles(shape, max_edge: float) -> int:
    """Triangles the grid would add at ``max_edge``, without imprinting.

    Two per cell, summed over the planar faces that would be subdivided. Pure
    arithmetic over `_planar_plan`, which only reads UV bounds — no OCCT
    modeling, so this is cheap enough to call once per candidate target.
    """
    total = 0
    for f in _faces_of(shape.wrapped):
        plan = _planar_plan(f, max_edge)
        if plan is None:
            continue
        _key, _point, _bounds, n_u, n_v = plan
        total += 2 * n_u * n_v
    return total


# ---------------------------------------------------------------------------
# Shape acceptance
# ---------------------------------------------------------------------------

def _wrapped_or_none(shape):
    """`shape.wrapped`, or ``None`` when the shape is null.

    Under build123d 0.11.1 ``Shape.wrapped`` is a property that **asserts**::

        @property
        def wrapped(self): assert self._wrapped; return self._wrapped

    so reading it on a null shape raises a bare, message-less ``AssertionError``
    rather than returning ``None``. That made the ``empty_shape`` branch below
    unreachable for the case its own message names — ``a & b`` where the two do
    not intersect — and `hasattr` does not save us either, since it only
    swallows ``AttributeError``.

    What the agent saw instead: the ``AssertionError`` escaped `add_part`,
    `_classify_exec_failure` matched a ``build123d`` frame and produced a
    `GeometryKernelError` reading "the CAD kernel could not complete this
    operation: AssertionError: " whose first suggested repair — the kernel's own
    message, normally the useful part — was the empty string.
    """
    try:
        return shape.wrapped
    except AssertionError:
        return None


def _is_shape(shape) -> bool:
    """Whether `shape` looks like a build123d topology object.

    `hasattr(shape, "wrapped")` is the wrong test twice over: it is
    False for a *null* shape (the property asserts) and it would be True for any
    unrelated object carrying that attribute name. Ask for the pair of members
    every topology object has, and treat a raising `wrapped` as "yes, a shape,
    and it is null" rather than "not a shape".
    """
    if not hasattr(type(shape), "wrapped"):
        return False
    return callable(getattr(shape, "solids", None))


def validate_shape(shape, *, name: str) -> None:
    """Reject anything that cannot become a rigid Part.

    Duck-typed on the real build123d surface rather than an ``isinstance``
    check: ``Shape`` is **not** in build123d's ``__all__``, so a public base
    class is not reliably importable, while ``.wrapped`` and ``.solids()`` are
    present on every topology object the API hands back.

    A Part is a rigid body, so "contains at least one Solid" is checked
    unconditionally: a Face, a Wire, or an empty boolean result is construction
    geometry and must be rejected at the line that registered it. There is no
    permissive mode — a caller cannot end up with a lax check by omitting a
    keyword.
    """
    if not _is_shape(shape):
        # `not_a_shape`, NOT `tessellation_failure`: nothing was tessellated
        # here, and nothing can be — the argument is not a topology object at
        # all. Reporting a tessellator failure would send the author looking at
        # a surface that does not exist.
        raise SolidCompileError(
            "not_a_shape",
            f"expected a build123d shape for {name!r}, got "
            f"{type(shape).__name__}")
    if _wrapped_or_none(shape) is None:
        raise SolidCompileError(
            "empty_shape",
            f"the shape for {name!r} is null — an empty Compound, or a boolean "
            f"whose operands did not intersect")
    # `Shape.is_valid` is a PROPERTY under the pinned build123d. Reading it as a
    # property is deliberate — calling it would raise
    # `TypeError: 'bool' object is not callable`, which is exactly the kind of
    # version skew the pin exists to prevent.
    try:
        valid = shape.is_valid
    except _API_MISUSE as e:
        raise SolidCompileError(
            "invalid_shape",
            f"could not read Shape.is_valid for {name!r} "
            f"({type(e).__name__}: {e}) — expected build123d "
            f"{REQUIRED_BUILD123D}, found "
            f"{installed_build123d_version() or 'none'}") from None
    except Exception:  # noqa: BLE001 — kernel declined to check; fall through
        valid = True
    if valid is False:
        raise SolidCompileError(
            "invalid_shape",
            f"{name!r} is not a valid solid — OCCT rejects its topology."
            + _brep_check_detail(shape) +
            " Check operand validity and the failing boolean interface. "
            "Coincident or tangent faces can cause failures; adjust local "
            "overlap at the feature's scale or rebuild the interface, then "
            "recheck. Coincidence alone does not establish the cause.")
    try:
        n_solids = len(shape.solids())
    except BaseException as e:  # noqa: BLE001
        raise SolidCompileError(
            "invalid_shape",
            f"could not inspect the topology of {name!r} "
            f"({type(e).__name__}: {e}); a Part must be a Solid or a Compound "
            f"of Solids") from None
    if n_solids < 1:
        raise SolidCompileError(
            "invalid_shape",
            f"{name!r} contains no Solid ({type(shape).__name__} with "
            f"{n_solids} solids). Register a Solid, build123d Part, or "
            f"Compound containing solids; extrude/revolve/loft construction "
            f"geometry into a body first.")


# ---------------------------------------------------------------------------
# Phase 1: measure  (eager — at add_part time)
# ---------------------------------------------------------------------------

def measure_shape(shape, *, name: str, tolerance: float = DEFAULT_TOLERANCE,
                  angular_tolerance: float = 0.1,
                  planar_max_edge: float | None = None,
                  texture_scale_m: float | None = None) -> dict:
    """Validate, measure, and SERIALIZE a live Shape; return a snapshot entry.

    The immutable registration boundary. Everything a Part will ever need is
    captured here — tessellation, mass properties, feature inventory, and the
    exact B-rep bytes — and the live Shape is then dropped. The returned entry
    holds no reference to it, so a Shape mutated afterwards cannot reach any
    artifact. That is a structural guarantee, not a convention.

    Doing the work eagerly also puts geometry errors on the authoring line
    rather than at finalize, where they would name a part but not a cause.
    """
    validate_shape(shape, name=name)

    # `planar_max_edge` tessellates an imprinted COPY; `shape` itself — and so
    # the B-rep, the AABB and every fact measured below — is the geometry the
    # author wrote. See `imprint_planar_grids`.
    # `texture_scale_m` is the bound material's tile size, and passing it is
    # the only thing that asks for texture *coordinates*. It is None for every
    # untextured part; the exact normals are computed either way.
    scale_m = None if texture_scale_m is None else float(texture_scale_m)
    if planar_max_edge is None:
        (verts, faces, weld_groups, normals, uvs,
         tol_used) = tessellate_within_budget(
            shape, name, float(tolerance), float(angular_tolerance),
            scale_m=scale_m)
        pme_used, n_imprinted, pme_achieved = None, 0, None
    else:
        (verts, faces, weld_groups, normals, uvs, tol_used, pme_used,
         n_imprinted, pme_achieved) = tessellate_with_planar_imprint(
            shape, name, float(tolerance), float(angular_tolerance),
            float(planar_max_edge), scale_m=scale_m)

    # Serialize the exact B-rep NOW, not at compile time.
    #
    # This is what makes registration a true snapshot boundary. Serialized
    # later, the B-rep would track mutations that the tessellation, AABB and
    # fact table — all captured here — do not, so a Shape edited after
    # registration would produce artifacts that disagree. The exact geometry is
    # what the collision adjudicator loads, so that divergence would be
    # silently permissive: a part could be checked for collisions somewhere
    # other than where every other consumer placed it.
    #
    # `Clean_s` first, and this is NOT optional. `tessellate` attaches a
    # triangulation to the TopoDS_Shape, and `BRepTools.Write_s` serializes
    # whatever is attached — several times the clean size, for data the .brep
    # exists precisely to not carry (the OBJ is the mesh). `shape_table`'s face
    # queries happen to drop the triangulation too, but depending on that would
    # let any reordering silently change every asset digest and therefore every
    # object id. Cleaning explicitly makes the bytes a function of the geometry
    # alone.
    from OCP.BRepTools import BRepTools as _BRepTools
    _BRepTools.Clean_s(shape.wrapped)
    brep = brep_bytes(shape, name=name)

    try:
        table, aabb_min, aabb_max, measure_notes = shape_table(shape)
    except SolidCompileError:
        raise
    except BaseException as e:  # noqa: BLE001
        # Only the mass properties can land here (the advisory inventory is
        # caught inside `shape_table`), so name that so the agent does not go
        # hunting in its feature-generating geometry.
        raise SolidCompileError(
            "exec_exception",
            f"could not measure the mass properties of solid {name!r}: "
            f"{type(e).__name__}: {e}. The shape built and tessellated, so this "
            f"is a volume/centroid/inertia query failing — usually a shape that "
            f"is not a closed solid (a lone face or shell, or a boolean that "
            f"produced an empty result).") from None

    return {
        # NOTE: the live Shape is deliberately NOT retained. Every downstream
        # artifact is derived from the values captured above, so there is no
        # reference through which a later mutation could reach them. Making the
        # hazard structurally impossible beats documenting it as unsupported.
        "name": name, "verts": verts, "faces": faces, "brep_bytes": brep,
        # Vertex end-offset per source solid, bounding the weld in
        # `compile_entry` so two exactly-tangent bodies are not fused into one
        # pinched surface. A list of ints, so the entry stays JSON-safe for the
        # worker boundary it crosses.
        "weld_groups": weld_groups,
        # Per-NODE shading attributes, parallel to `verts` and PRE-WELD: the
        # exact B-rep surface normal, and the texture coordinate in tile units
        # (None entirely for an untextured part). `compile_entry` lifts both to
        # per-corner through `RepairedMesh.corners`, which is the only thing
        # that can still say which node a welded corner came from.
        #
        # Shading and texture data only — nothing derived from either reaches
        # the mass properties, the AABB or the exact B-rep.
        "normals": normals,
        "uvs": uvs,
        "table": table, "aabb_min": aabb_min, "aabb_max": aabb_max,
        "tolerance": tolerance, "tolerance_used": tol_used,
        "planar_max_edge": planar_max_edge,
        "planar_max_edge_used": pme_used,
        "planar_max_edge_achieved": (None if pme_achieved is None
                                    else q(pme_achieved)),
        "n_faces_imprinted": n_imprinted,
        "measure_notes": measure_notes,
    }


# ---------------------------------------------------------------------------
# Phase 2: compile to bytes  (at finalize / _finish_solid time)
# ---------------------------------------------------------------------------

def brep_bytes(shape, *, name: str) -> bytes:
    """Serialize the exact B-rep.

    OCCT's writer takes a filesystem path, so this round-trips through a temp
    file. The BREP format is pure geometry text with nothing path-dependent in
    it, so the bytes are identical to what writing straight to the destination
    would have produced — asserted by the extraction-equivalence test.
    """
    from OCP.BRepTools import BRepTools
    with tempfile.TemporaryDirectory(prefix="dycraft_brep_") as td:
        path = Path(td) / "shape.brep"
        if not BRepTools.Write_s(shape.wrapped, str(path)):
            raise SolidCompileError(
                "brep_write_failure",
                f"OCCT declined to write the B-rep for {name!r}")
        return path.read_bytes()


def _obj_bytes(verts, faces, corner_normals=None, corner_uvs=None) -> bytes:
    """Render mesh as OBJ bytes, through `_geometry_qc._write_obj`.

    Going through the shared writer rather than formatting here is what keeps
    the exact per-corner normals: `_write_obj` emits `vn` lines and `v//vn`
    faces, without which every loader we ship to averages normals across sharp
    CAD edges and a flat panel renders with an X-shaped crease. A textured part
    additionally carries `vt` lines and takes the `v/vt/vn` form.
    """
    with tempfile.TemporaryDirectory(prefix="dycraft_obj_") as td:
        path = Path(td) / "mesh.obj"
        _gq._write_obj(str(path), verts, faces,
                       corner_normals=corner_normals, corner_uvs=corner_uvs)
        return path.read_bytes()


def _shading_per_corner(corners, normals, uvs, *, flipped: bool):
    """Per-triangle corner triples of exact normals and UVs, or ``(None, None)``.

    `corners` is `RepairedMesh.corners` — for each surviving triangle, the
    PRE-WELD node indices of its three corners — which is the only thing that
    can still say which tessellation node a welded corner came from. `normals`
    and `uvs` are indexed by that same pre-weld numbering.

    `flipped` negates every normal. `_repair_mesh` re-winds a mesh whose shells
    all enclose volume inside-out, and that reverses which side of the surface
    is outward; the exact normals were computed against the original winding, so
    leaving them alone would point every one of them into the material.
    """
    if not corners or not normals:
        return None, None
    sign = -1.0 if flipped else 1.0

    def normal_at(i):
        n = normals[i] if 0 <= i < len(normals) else None
        return None if n is None else (sign * n[0], sign * n[1], sign * n[2])

    def uv_at(i):
        return uvs[i] if uvs and 0 <= i < len(uvs) else None

    corner_normals = [tuple(normal_at(i) for i in tri) for tri in corners]
    corner_uvs = [tuple(uv_at(i) for i in tri) for tri in corners]
    return corner_normals, corner_uvs


#: Compiled parts, keyed by the geometry that produced them.
#:
#: **Why a scene build needs this and an object build does not.** `object.py`
#: compiles one object, so nothing recurs. A scene-layer `scene.py` compiles every
#: object it authors on every build, so editing one line of one object would
#: rerun repair, QC and OBJ encoding for every other object, each of which
#: reconstructs byte-identically. `compile_entry` is about half of a part's
#: compile cost, and it is the half that is a pure function of the geometry
#: handed to it.
#:
#: Not the whole cost: tessellation happens earlier, in `measure_shape`, and is
#: the other half. Caching that too would mean serializing the B-rep before
#: triangulating rather than after, which is an ordering this file arrived at
#: deliberately (see `measure_shape`) — worth doing, not worth doing quickly.
#:
#: Bounded because each entry holds an OBJ and a B-rep. Oldest-out rather than
#: true LRU: the access pattern is a rebuild sweeping every part in order, so
#: recency and insertion order are the same thing here.
_COMPILE_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_COMPILE_CACHE_MAX = 512


def _compile_cache_key(entry: dict) -> "str | None":
    """A digest of the whole entry, or None when there is nothing to key on.

    **Everything, not the fields that look relevant.** `brep_bytes` plus the
    tessellation parameters do not determine the compile: `texture_scale_m`
    reaches the entry as mapping state and phases the UV corners per part, so
    an untextured part and a textured one off the same box with the same name
    would collide and one be served the other's bytes. Hashing the input
    rather than a summary of it keeps the next field added here from doing the
    same.

    Cost is a blake2b pass over the entry, dominated by the vertex and face
    arrays. Against the `compile_entry` it guards, that is noise; a wrong
    answer served instantly is not.
    """
    if not isinstance(entry, dict):
        return None
    brep = entry.get("brep_bytes")
    if not isinstance(brep, (bytes, bytearray)) or not brep:
        # Nothing identifies this geometry, so nothing may be reused for it.
        return None
    h = hashlib.blake2b(digest_size=16)
    for field in sorted(entry):
        value = entry[field]
        h.update(field.encode("utf-8"))
        h.update(b"\x00")
        if isinstance(value, (bytes, bytearray)):
            h.update(value)
        else:
            try:
                h.update(repr(value).encode("utf-8"))
            except Exception:  # noqa: BLE001 — an unhashable field is a miss
                return None
        h.update(b"\x01")
    return h.hexdigest()


def clear_compile_cache() -> None:
    """Drop the per-part compile cache. For tests and for measurement."""
    _COMPILE_CACHE.clear()


def compile_entry(entry: dict) -> dict:
    """Repair -> phase -> QC -> bytes, in that order, memoized on the geometry.

    A hit is deep-copied, not shallow-copied. `object_builder` builds its
    geometry note by reading `warnings` off this result and appending to it, so
    a shallow copy hands every later hit the previous caller's edits — the
    second compile of one part would report the first one's warnings. The
    payload's bulk is `obj_bytes` and `brep_bytes`, which are immutable and so
    cost a reference each; what actually gets copied is a handful of small
    lists and dicts.
    """
    key = _compile_cache_key(entry)
    if key is not None:
        hit = _COMPILE_CACHE.get(key)
        if hit is not None:
            return copy.deepcopy(hit)
    result = _compile_entry_uncached(entry)
    if key is not None:
        _COMPILE_CACHE[key] = copy.deepcopy(result)
        while len(_COMPILE_CACHE) > _COMPILE_CACHE_MAX:
            _COMPILE_CACHE.popitem(last=False)
    return result


def _compile_entry_uncached(entry: dict) -> dict:
    """Repair -> phase -> QC -> bytes, in that order.

    The order matters: the reported QC has to describe exactly the geometry
    that gets staged, not the raw tessellation.

    Operates purely on the snapshot `measure_shape` captured; it never touches
    a live Shape, because the entry does not carry one. The B-rep bytes are
    passed through unchanged from registration time.

    Returns a dict carrying ``obj_bytes`` / ``brep_bytes`` plus everything the
    result payload and the ObjectIR params need; the caller hands the bytes to
    ``MeshAssetSession.stage_solid``.
    """
    name = entry["name"]
    verts, faces = entry["verts"], entry["faces"]
    # Measurement notes (an incomplete feature inventory) ride out with the mesh
    # warnings rather than in a channel of their own — one place to look.
    warnings: list = list(entry.get("measure_notes") or ())
    repairs: list = []

    original = (verts, faces)
    corner_normals = corner_uvs = None
    try:
        # `track_corners=True` is what makes the shading lift possible: the
        # weld merges nodes that share a position but not a normal, so after it
        # only the pre-weld index can say which node a surviving corner was.
        # One join serves both attribute streams.
        repaired = _gq._repair_mesh(
            verts, faces, entry.get("weld_groups"), track_corners=True)
        verts, faces, repairs = repaired.verts, repaired.faces, repaired.actions
        corner_normals, corner_uvs = _shading_per_corner(
            repaired.corners, entry.get("normals"), entry.get("uvs"),
            flipped=repaired.flipped)
        repair_status = "ok"
        warnings.extend(repaired.notes)
    except BaseException as e:  # noqa: BLE001 — keep the unrepaired mesh
        verts, faces = original
        repairs = []
        corner_normals = corner_uvs = None
        repair_status = "failed"
        warnings.append(
            f"mesh repair failed internally ({type(e).__name__}: {e}); the "
            f"unrepaired tessellation was staged and its QC numbers may be "
            f"misleading")

    if not faces:
        raise SolidCompileError(
            "empty_mesh",
            f"every triangle of solid {name!r} was zero-area or duplicated")

    # The part's tile phase, applied once to the finished array — after repair,
    # so it lands on exactly the corners that get written, and on both the
    # repaired and the fall-back-to-unrepaired paths above. A constant shift
    # commutes with everything repair does (dropping triangles, reordering a
    # triangle's three corners to flip winding), so the order is a readability
    # choice rather than a correctness one. `None` for an untextured part, which
    # is what keeps its bytes byte-identical to a pre-phase build.
    corner_uvs = _phase_corner_uv(corner_uvs, name)

    qc = None
    qc_status = "ok"
    try:
        qc = _gq._mesh_qc(verts, faces)
    except BaseException as e:  # noqa: BLE001 — never fail a good solid over QC
        qc_status = "failed"
        warnings.append(
            f"mesh QC failed internally ({type(e).__name__}: {e}); the "
            f"solid was staged but its triangle mesh is unmeasured")
    # **Mesh QC and the coarsening note are measured but NOT reported to the
    # agent.** `warnings` reaches the agent as `geometry_notes`, and neither
    # earns its cost there:
    #
    # * Nothing downstream of the object layer reads mesh closure. The layer's
    #   own adjudicator is B-rep (`verify/_narrow_phase`), `urdf_export` gives
    #   an authored solid a `<box>` collision proxy rather than its mesh, and
    #   inertia comes from B-rep mass properties. The closure predicates that
    #   do gate something — the importers' collision decision — run on
    #   third-party assets through their own `watertight()`.
    # * An offline grader recomputes `bounds_volume` off the exported mesh
    #   under its own interpreter, so it never reads these notes and still
    #   scores them. A graded metric does not have to be an in-run warning.
    #
    # A mesh-closure note sends an agent hunting for a defect nothing reads,
    # often after the object is already committable. The coarsening note says
    # outright that the B-rep is unaffected, which is the whole reason it is
    # noise here.
    #
    # `qc` still goes to the IR as `shape.qc`, so `part_shape` answers the
    # question for an agent that asks it, and post-hoc analysis keeps every
    # number. `qc_status` keeps only the states that mean QC itself did not
    # run; a measured defect is not a status the agent sees.

    # Same rule for the planar imprint, and the same reason: an author who asked
    # for a 4 mm ceiling and silently got 8 mm would bind a deformation rig to a
    # mesh that cannot carry it, which is exactly the failure the parameter
    # exists to prevent. `n_faces_imprinted == 0` with a request set means no
    # planar face needed one, or OCCT declined — say which is not knowable here,
    # so say the fact.
    req = entry.get("planar_max_edge")
    if req is not None:
        used = entry.get("planar_max_edge_used")
        if used is not None and used != req:
            warnings.append(
                f"solid {name!r} exceeded the {MAX_TRIANGLES}-triangle render "
                f"budget at planar_max_edge {req:g}; the planar grid was "
                f"coarsened to {used:g} ({len(faces)} triangles). A rig bound "
                f"to this mesh cannot resolve detail finer than {used:g} m.")
        if not entry.get("n_faces_imprinted"):
            warnings.append(
                f"solid {name!r} requested planar_max_edge {req:g} but no "
                f"planar face was imprinted — either every planar face was "
                f"already inside it, or the shape has none (a fully curved "
                f"body). Read n_vertices before binding a rig to it.")
        # The measured outcome, not the request. Grid lines are laid across a
        # face's UV bounding box, so a face whose outline does not fill that box
        # — notched, L-shaped, holed — keeps a coarse region near the cutout
        # where OCCT refused the splitters. That is the one way this parameter
        # falls short while every other signal reads as success.
        got = entry.get("planar_max_edge_achieved")
        used = entry.get("planar_max_edge_used")
        if got is not None and used is not None and got > used:
            warnings.append(
                f"solid {name!r}: the longest edge measured on a planar face is "
                f"{got:g} m, above the {used:g} m grid. Expect this on a face "
                f"whose outline does not fill its bounding box (a notch, an "
                f"L, a hole) — the region around the cutout is still coarse. A "
                f"rig bound there cannot resolve detail finer than {got:g} m.")

    # `qc` / `repairs` / `warnings` belong in the shape table, not just in this
    # staging record: only `shape` survives into the ObjectIR params, and
    # `part_shape()` is where a caller reads `bounds_volume` / `n_shells`. Anything
    # left out here is computed and discarded.
    #
    # `qc` joins the shape table because it is a *fact about the staged
    # geometry* — closed, n_shells, n_boundary_edges, volume_m3, bbox — in the
    # same sense as `volume` and `topo`, and it has to travel in the IR to be
    # visible at all to a consumer that only ever sees the catalog (the scene
    # agent reads it through `objects[…].part_shapes()`, which has no access to
    # this run's compile reports). Always present; None only when the QC pass
    # itself failed, in which case `warnings` carries the reason.
    #
    # `repairs` and `warnings` deliberately do NOT go here. They describe what
    # happened during *this compile* — which duplicate vertices were welded, an
    # exception a stage raised, a tessellation coarsened to stay under the
    # triangle budget — not what the object is. Two reasons that matters: the IR
    # is content-hashed whole, so a warning's prose would become part of the
    # object's identity (and the coarsening note embeds a triangle count, which
    # is OCCT-version dependent); and a per-run diagnostic belongs with the
    # other per-run diagnostics. They ride out in the return value below and
    # reach the agent through the compile report — see
    # `sdk._mesh_assets.MeshAssetSession.add_geometry_note`.
    shape = dict(entry["table"])
    shape["qc"] = qc

    return {
        "repairs": repairs,
        "warnings": warnings,
        "obj_bytes": _obj_bytes(verts, faces, corner_normals, corner_uvs),
        # Captured at registration by `measure_shape`, not re-derived here:
        # the entry holds no live Shape to re-derive it from.
        "brep_bytes": entry["brep_bytes"],
        "aabb_min": entry["aabb_min"],
        "aabb_max": entry["aabb_max"],
        "shape": shape,
        "n_vertices": len(verts),
        "n_faces": len(faces),
        "repair_status": repair_status,
        "qc_status": qc_status,
        "tolerance_used": entry["tolerance_used"],
        "planar_max_edge_used": entry.get("planar_max_edge_used"),
        "planar_max_edge_achieved": entry.get("planar_max_edge_achieved"),
        "n_faces_imprinted": entry.get("n_faces_imprinted") or 0,
    }


def failure_hint(message: str):
    """Plain-language hint for a known kernel failure, or None.

    Re-exported on the compiler surface so the SDK can reach it through the
    injected compiler (``sdk._mesh_assets.set_active_solid_compiler``) instead of
    importing the agent layer. `compile.py`, which is in this package already,
    calls `_geometry_qc.failure_hint` directly.
    """
    return _gq.failure_hint(message)
