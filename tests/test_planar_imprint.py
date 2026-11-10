"""`add_part(planar_max_edge=...)` — interior vertices on flat faces.

Neither `tolerance` nor `angular_tolerance` can put a vertex inside a planar
face: chord and angular error are both zero on a plane, so OCCT triangulates one
from its boundary alone at any value (a pure box is 24 vertices from 1e-3 to
1e-6). That is not a missing flag: no `IMeshTools_Parameters` setting, alone or
combined, moves the count. So a flat
face gets its own parameter, and this file pins the four things about it that a
refactor could silently take away:

  * it actually bounds the longest triangle edge, using the cell **diagonal**
    rather than the cell side, so the guarantee holds on a non-square face;
  * the exact `.brep` and every measurement taken off the Shape are **untouched**
    — the imprint is a tessellation-time copy, because the geometry the author
    committed is what collision and mass properties are answered from;
  * a curved face is left to the tolerances, and an already-fine flat face is
    left alone rather than subdivided for nothing;
  * the triangle budget is defended by coarsening the **grid**, not the chord
    tolerance, and never silently.

Why this parameter exists at all: a deformation rig binds to the render mesh, so
a face with no interior vertices can only move as a plane. A sponge whose end
caps are two triangles each passes every object verifier and still cannot show a
localised dent under a gripper pad at any control count.
"""

from __future__ import annotations

import math

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.sdk._errors import BuildError


# ---------------------------------------------------------------------------
# Validation — fires before any CAD is needed, so it runs in the fast suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -0.004, float("nan"), float("inf"),
                                 "0.004", [], True])
def test_a_nonpositive_or_nonnumeric_planar_max_edge_is_rejected(bad):
    """Rejected at the authoring line, as a `BuildError`, not at finalize.

    In-process on purpose: the check sits before `add_part` looks for a
    compiler, so it needs no CAD interpreter and belongs in the fast suite.

    `True` is in the list because `isinstance(True, int)` — a bool that slipped
    through would be read as a 1 m ceiling, i.e. silently no grid at all. `None`
    is the one non-number that is legal and means "leave flat faces alone", so
    the check has to distinguish it from every other non-number rather than
    testing truthiness.
    """
    from dycraft.sdk._errors import InvalidPrimitive
    from dycraft.sdk.object_builder import Object, _reset_object_singleton
    _reset_object_singleton()
    try:
        obj = Object("v")
        with pytest.raises(InvalidPrimitive) as excinfo:
            obj.add_part("b", object(), planar_max_edge=bad)
        assert "planar_max_edge" in str(excinfo.value)
        assert isinstance(excinfo.value, BuildError)
        assert excinfo.value.to_dict()["evidence"]["param"] == "planar_max_edge"
    finally:
        _reset_object_singleton()


# ---------------------------------------------------------------------------
# Geometry — needs the frozen CAD interpreter
# ---------------------------------------------------------------------------

cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")

#: The sponge that motivated the parameter, 115 x 58 x 55 mm. Bare, because
#: every assertion here is about its flat faces and a 3 mm edge fillet only buys
#: 5 500 vertices of tessellation to pay for. `_FILLETED` is used where the mix
#: of planar and curved faces is the point.
_BLOCK = "Box(0.115, 0.058, 0.055)"
_FILLETED = "fillet(Box(0.115, 0.058, 0.055).edges(), 0.003)"


def _build(run_dir, shape=_BLOCK, *, pme=None, extra=""):
    arg = "" if pme is None else f", planar_max_edge={pme!r}"
    src = (f'obj = Object("t")\n'
           f'b = obj.add_part("body", {extra or shape}{arg})\n'
           f'obj.declare_root(b)\nir = obj.finalize()\n')
    out = W.run_object_source(src, run_dir=run_dir)
    assert out["ok"], out.get("build_error") or out.get("worker_stderr")
    return out


def _notes(out):
    return out["geometry_notes"][0]


def _qc(out) -> dict:
    """The staged mesh's QC facts, which own the vertex/triangle counts."""
    return out["ir"]["parts"][0]["params"]["shape"]["qc"]


def _no_notes(out) -> bool:
    """True when the compile had nothing actionable to report.

    Post-merge a per-part note is emitted only when the compiler could not fix
    something, or when the author asked for a planar imprint. A clean part that
    asked for nothing produces no entry at all — and `compile.py` omits the key
    entirely rather than carrying an empty list.
    """
    return not out.get("geometry_notes")


def _obj_mesh(run_dir, out):
    """(vertices, triangles) of the staged OBJ, as plain lists of tuples."""
    mesh = run_dir / out["ir"]["parts"][0]["params"]["mesh"]
    verts, tris = [], []
    for line in mesh.read_text().splitlines():
        if line.startswith("v "):
            verts.append(tuple(float(x) for x in line.split()[1:4]))
        elif line.startswith("f "):
            tris.append(tuple(int(t.split("/")[0]) - 1 for t in line.split()[1:4]))
    return verts, tris


def _face_max_edge(verts, tris, axis, value, tol=1e-6):
    """Longest triangle edge, in meters, over triangles lying in one plane.

    `None` when that plane carries no triangle at all, which is a distinct
    outcome from "carries coarse ones" and must not read as a pass.
    """
    longest = None
    for a, b, c in tris:
        pts = (verts[a], verts[b], verts[c])
        if any(abs(p[axis] - value) > tol for p in pts):
            continue
        for p, q in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            e = math.dist(p, q)
            longest = e if longest is None else max(longest, e)
    return longest


@cad
def test_a_flat_face_is_boundary_only_until_asked(tmp_path):
    """Without the parameter the end cap is a fan across the whole face.

    The baseline half of the guarantee. If this ever starts passing on its own,
    OCCT changed and the parameter may no longer be load-bearing — which is
    worth knowing, rather than having the paired test below silently carry it.
    """
    out = _build(tmp_path)
    verts, tris = _obj_mesh(tmp_path, out)
    cap_x = max(v[0] for v in verts)
    longest = _face_max_edge(verts, tris, 0, cap_x)
    assert longest is not None, "the +X cap carries no triangles at all"
    # A bare box cap is two triangles whose long edge IS the face diagonal,
    # hypot(58, 55) = 79.9 mm. Anything near that is a boundary-only fan.
    assert longest > 0.070, f"expected a spanning fan, got {longest * 1000:.1f} mm"
    # Nothing was requested and nothing went wrong, so there is no note to
    # read. The mesh assertion above is what carries this test; "no imprint
    # happened" is spelled by the absence of an entry.
    assert _no_notes(out), out.get("geometry_notes")


@cad
def test_the_requested_edge_bounds_every_flat_face(tmp_path):
    """The guarantee, on all six flat faces of the block at once.

    Bounded by the cell *diagonal*, not the cell side: a triangle of a u x v
    cell is at most `hypot(u, v)` long, so each side is divided against
    `max_edge / sqrt(2)`. Dividing against `max_edge` itself would overshoot by
    up to 41% and this test would fail — which is the point of asserting on
    every face rather than only the square-ish cap.
    """
    want = 0.004
    out = _build(tmp_path, pme=want)
    verts, tris = _obj_mesh(tmp_path, out)
    assert _notes(out)["planar_max_edge_used"] == want
    assert _notes(out)["n_faces_imprinted"] == 6, "all six flats, not just the caps"

    for axis, label in ((0, "X"), (1, "Y"), (2, "Z")):
        for value, side in ((min(v[axis] for v in verts), "-"),
                            (max(v[axis] for v in verts), "+")):
            longest = _face_max_edge(verts, tris, axis, value)
            assert longest is not None, f"{side}{label} carries no triangles"
            assert longest <= want + 1e-9, (
                f"{side}{label} longest edge {longest * 1000:.2f} mm exceeds the "
                f"requested {want * 1000:.1f} mm")


@cad
def test_the_brep_and_every_measurement_are_untouched(tmp_path):
    """The load-bearing invariant: only the render mesh moves.

    The imprint runs on a copy, so the `.brep` bytes, the AABB and the shape
    fact table must be identical with and without it. They are what the exact
    collision adjudicator, the mass properties and every probe answer from; an
    imprint that reached them would mean asking for a finer *visual* mesh
    silently changed what the object physically is.
    """
    plain = _build(tmp_path / "plain", shape=_FILLETED)
    fine = _build(tmp_path / "fine", shape=_FILLETED, pme=0.010)

    p_par = plain["ir"]["parts"][0]["params"]
    f_par = fine["ir"]["parts"][0]["params"]

    assert p_par["digest"] == f_par["digest"], "the B-rep digest moved"
    assert (tmp_path / "plain" / p_par["brep"]).read_bytes() == \
           (tmp_path / "fine" / f_par["brep"]).read_bytes()
    assert p_par["aabb_min"] == f_par["aabb_min"]
    assert p_par["aabb_max"] == f_par["aabb_max"]

    # Everything in the fact table except `qc` describes the exact solid — mass
    # properties, feature inventory — and must not move at all.
    p_shape = {k: v for k, v in p_par["shape"].items() if k != "qc"}
    f_shape = {k: v for k, v in f_par["shape"].items() if k != "qc"}
    assert p_shape == f_shape, "a fact about the exact solid changed"

    # `qc` is a fact about the *staged mesh*, so its counts are supposed to grow
    # — that is the feature. What must not change is whether that mesh is still
    # a sound closed body, which is the half a denser tessellation could break.
    p_qc, f_qc = p_par["shape"]["qc"], f_par["shape"]["qc"]
    # `closed` split into `bounds_volume` (edge parity — the enclosure every
    # consumer needs) and `is_manifold` (the stronger every-edge-exactly-twice
    # claim, reported and enforced nowhere). Both are checked: a denser
    # tessellation must not break either.
    for key in ("bounds_volume", "is_manifold", "winding_consistent",
                "n_shells", "n_degenerate_shells", "n_boundary_edges",
                "n_nonmanifold_edges"):
        assert p_qc[key] == f_qc[key], f"qc.{key}: {p_qc[key]} -> {f_qc[key]}"
    assert f_qc["volume_m3"] == pytest.approx(p_qc["volume_m3"], rel=1e-3)

    # And the thing that *should* differ, so the test cannot pass vacuously.
    assert f_qc["n_faces"] > p_qc["n_faces"]
    # Vertex counts describe the tessellator rather than the design, so they
    # left the per-part note and live on `qc`, which always carries them.
    assert f_qc["n_vertices"] > p_qc["n_vertices"]


@cad
def test_a_curved_body_is_left_to_the_tolerances_and_says_so(tmp_path):
    """A sphere has no planar face; the request is reported as a no-op.

    Silence here would read as "your rig has interior vertices now", which is
    exactly the wrong conclusion to let an author reach.
    """
    out = _build(tmp_path, shape="Sphere(0.05)", pme=0.004)
    notes = _notes(out)
    assert notes["n_faces_imprinted"] == 0
    assert any("no planar face was imprinted" in w for w in notes["warnings"]), \
        notes["warnings"]


@cad
def test_an_already_fine_flat_face_is_not_subdivided(tmp_path):
    """A request coarser than the face's own diagonal must do nothing.

    Otherwise a 2 mm chamfer flat gets a grid, multiplying faces for a face that
    was never going to carry a rig.
    """
    out = _build(tmp_path, shape="Box(0.01, 0.01, 0.01)", pme=0.050)
    assert _notes(out)["n_faces_imprinted"] == 0


@cad
def test_the_budget_coarsens_the_grid_not_the_chord_tolerance(tmp_path):
    """An impossible request comes back coarser, and names what it got.

    Coarsening `tolerance` cannot rescue an over-budget planar imprint — those
    triangles come from topology, not chord error — so the grid is what backs
    off. `tolerance_used` staying put is the half of this that a naive
    implementation gets wrong: it would spend four attempts wrecking the
    fillets and still be over budget.
    """
    out = _build(tmp_path, pme=0.0005)
    notes = _notes(out)
    assert notes["planar_max_edge_used"] > 0.0005, "the grid did not back off"
    # The note carries `tolerance_used` only when it differs from the request,
    # so its ABSENCE is the assertion that the chord ladder never ran — sharper
    # than an equality, which would also pass if the ladder ran and happened
    # to land back on the requested value.
    assert "tolerance_used" not in notes, notes.get("tolerance_used")
    assert _qc(out)["n_faces"] <= 20000
    assert any("planar grid was coarsened" in w for w in notes["warnings"]), \
        notes["warnings"]


@cad
def test_imprinting_is_orientation_independent(tmp_path):
    """A rotated block gets the same treatment as an axis-aligned one.

    A planar face's UV parameters are meters along its *own* plane axes, so the
    plan is computed in the face's frame. An implementation that reached for
    world X/Y/Z instead would pass every test above and fail here.
    """
    # A coarse grid is enough: what is under test is that the plan is computed
    # in the face's own frame, not how fine it is.
    rotated = f"Rotation(23, 41, 17) * ({_BLOCK})"
    plain_out = _build(tmp_path / "plain", pme=0.010)
    turned_out = _build(tmp_path / "turned", shape=rotated, pme=0.010)
    plain, turned = _notes(plain_out), _notes(turned_out)
    assert turned["n_faces_imprinted"] == plain["n_faces_imprinted"] == 6
    # Same geometry, same grid: the triangle counts should be within a few
    # percent (the fillet tessellation shifts slightly under rotation).
    # Counts come from `qc`, not from the note.
    p_n = _qc(plain_out)["n_faces"]
    t_n = _qc(turned_out)["n_faces"]
    assert abs(t_n - p_n) < 0.05 * p_n, f"rotated {t_n} vs axis-aligned {p_n}"


@cad
def test_two_coplanar_faces_are_imprinted_separately(tmp_path):
    """A notch splits one face into two coplanar ones; both must be refined.

    The plan is per-FACE. Keyed on the plane alone, the two would be
    conflated: the grid lines would come from whichever face is found first,
    would not lie on the other, and OCCT would drop them, leaving the twin
    unrefined while `n_faces_imprinted` reports success.
    """
    want = 0.004
    notched = ("Box(0.15, 0.05, 0.04) - "
               "Pos(0, 0, 0.015) * Box(0.05, 0.06, 0.02)")
    out = _build(tmp_path, shape=notched, pme=want)
    verts, tris = _obj_mesh(tmp_path, out)

    z_top = max(v[2] for v in verts)
    for label, keep in (("left", lambda x: x < 0.0), ("right", lambda x: x > 0.0)):
        longest = None
        for a, b, c in tris:
            pts = (verts[a], verts[b], verts[c])
            if any(abs(p[2] - z_top) > 1e-6 for p in pts):
                continue
            if not keep(sum(p[0] for p in pts) / 3.0):
                continue
            for p, q in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
                e = math.dist(p, q)
                longest = e if longest is None else max(longest, e)
        assert longest is not None, f"the {label} top face carries no triangles"
        assert longest <= want + 1e-9, (
            f"{label} top face longest edge {longest * 1000:.2f} mm exceeds "
            f"the requested {want * 1000:.1f} mm")


@cad
def test_the_achieved_edge_is_measured_and_reported(tmp_path):
    """`planar_max_edge_achieved` is read back off the mesh, not assumed.

    Every silent failure this parameter has had looked like success from the
    request side — a plan computed, a `Build()` that returned, a face count that
    went up — so the achieved number is the only honest evidence.
    """
    out = _build(tmp_path, pme=0.004)
    notes = _notes(out)
    assert notes["planar_max_edge_achieved"] is not None
    assert notes["planar_max_edge_achieved"] <= 0.004 + 1e-9
    # And there is no note at all when nothing was asked for.
    assert _no_notes(_build(tmp_path / "off"))


@cad
def test_a_nonrectangular_planar_face_reports_its_shortfall(tmp_path):
    """A KNOWN LIMITATION, pinned so it stays visible rather than surprising.

    Grid lines are laid across a face's UV *bounding box*. On a face whose
    outline fills that box they lie on the face and split it. On a notched,
    L-shaped, or holed face part of each line falls outside the face, OCCT
    refuses that splitter, and the region near the cutout stays coarse — the
    U-shaped side faces of this part reach 16 mm against a 4 mm request.

    What must hold is that the shortfall is *reported*: `planar_max_edge_achieved`
    exceeds the request, so a caller about to bind a deformation rig can see it.
    Clipping each line to the face outline (`BRepTopAdaptor_FClass2d`) would fix
    the underlying gap; it is not implemented. If it ever is, this test should
    start failing on its first assertion — that is the intended signal.
    """
    want = 0.004
    notched = ("Box(0.15, 0.05, 0.04) - "
               "Pos(0, 0, 0.015) * Box(0.05, 0.06, 0.02)")
    notes = _notes(_build(tmp_path, shape=notched, pme=want))
    achieved = notes["planar_max_edge_achieved"]
    assert achieved > want, (
        "the U-shaped side faces are expected to fall short; if they no longer "
        "do, line clipping landed and this test has served its purpose")
    assert notes["n_faces_imprinted"] > 0, (
        "n_faces_imprinted counts attempts, so it must not read as failure here")
