"""Mesh QC, repair, and OBJ shading data — `_geometry_qc`.

This runs on **every** `add_part`: the shared weld/repair/measure/write pass
between the CAD kernel's raw tessellation and the OBJ that reaches the renderer,
the URDF and the viewer. It is pure stdlib and O(n) on purpose, so these tests
need no CAD interpreter and no subprocess.

The subtle parts, and why each has a test:

  * orientation is decided **per shell**, never on the mesh's total signed
    volume — that sum is taken over shells sharing no edges, so a big inverted
    shell next to a small correct one makes it negative and a sum-based test
    "fixes" the wrong one;
  * shells enclosing no volume are excluded from that vote entirely, because
    the sign of a ~1e-23 volume is rounding noise rather than an orientation;
  * welding keys off the **written** form of a coordinate, so two vertices weld
    if and only if they serialize identically.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Loaded by path, deliberately: `_geometry_qc` must stay importable with no
# package context, since CAD children sit next to it in a flat directory and
# dycraft is not installed in their interpreter.
_spec = importlib.util.spec_from_file_location(
    "_geometry_qc_ut", REPO / "agent" / "cad" / "_geometry_qc.py")
GQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GQ)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _unit_cube():
    """A welded, outward-wound unit cube: 8 vertices, 12 triangles."""
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return verts, faces


def _cylinder(nseg=32, radius=0.5, height=1.0, caps=True):
    """A welded cylinder: a smooth barrel meeting two flat caps at a crease."""
    verts, faces = [], []
    for i in range(nseg):
        a = 2 * math.pi * i / nseg
        verts.append((radius * math.cos(a), radius * math.sin(a), 0.0))
        verts.append((radius * math.cos(a), radius * math.sin(a), height))
    for i in range(nseg):
        b0, b1 = 2 * i, 2 * ((i + 1) % nseg)
        faces += [(b0, b1, b1 + 1), (b0, b1 + 1, b0 + 1)]
    if caps:
        bot, top = len(verts), len(verts) + 1
        verts += [(0.0, 0.0, 0.0), (0.0, 0.0, height)]
        for i in range(nseg):
            b0, b1 = 2 * i, 2 * ((i + 1) % nseg)
            faces += [(bot, b1, b0), (top, b0 + 1, b1 + 1)]
    return verts, faces


def _measure(verts, faces):
    """Repair then measure, the order `compile_entry` uses.

    The order matters: the reported QC has to describe exactly the geometry that
    gets staged, not the raw input.
    """
    r = GQ._repair_mesh(verts, faces)
    return GQ._mesh_qc(r.verts, r.faces), r.actions, r.notes, r.verts, r.faces


# A tetrahedron, in both windings.
_TET_OUT = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]
_TET_IN = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
_TET_V = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)]

# Two disjoint tetrahedra. The second is 2x the first, so when it alone is
# inverted the mesh's TOTAL signed volume is negative even though one shell is
# perfectly fine — the exact configuration a sum-based orientation test gets
# wrong. Shell A = indices 0-3, shell B = indices 4-7.
_TWO_TETS_V = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1),
               (1, 0, 0), (1.2, 0, 0), (1, 0.2, 0), (1, 0, 0.2)]
_TET_A_OUT = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]
_TET_A_IN = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
_TET_B_IN = [(4, 5, 6), (4, 6, 7), (4, 7, 5), (5, 7, 6)]

# Shell A (indices 0-3) is a real tetrahedron; shell B (indices 4-7) has the
# same topology but all four points COPLANAR (z=0) — a flat fin enclosing
# nothing. Its four triangles all have real area, so `_drop_bad_faces` leaves
# them alone and the shell stays closed and consistently wound; only its VOLUME
# is zero. This is the excavator bucket's tooth defect in miniature: every
# vertex of a piece placed on one plane, whose signed volume then comes out at
# ~1e-23 with a sign decided by rounding.
_TET_PLUS_FLAT_V = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1),
                    (1, 0, 0), (1.2, 0, 0), (1, 0.2, 0), (1.05, 0.05, 0)]


# ---------------------------------------------------------------------------
# Shading data: exact normals and UVs, supplied per corner
# ---------------------------------------------------------------------------
#
# The normals are not derived here: `_solid_compile` evaluates them on the
# real B-rep surface and hands them in. What these tests pin is the contract
# this module owns: that supplied shading is written faithfully,
# that a missing entry degrades in a defined way, and that none of it can move
# the geometry.


def _exact_cube_normals(verts, faces):
    """Per-corner normals for `_unit_cube`, each corner carrying its own plane.

    Stands in for what `_solid_compile._surface_normal` produces on a box: one
    exact normal per face, not blended with anything.
    """
    return [tuple(GQ._unit(GQ._face_normals(verts, [tri])[0]) for _ in range(3))
            for tri in faces]


def test_supplied_normals_are_written_per_corner_not_per_vertex():
    """The whole point of `vn`: one welded position can carry several normals.

    Without per-corner normals in the file, every loader we ship to (Open3D's
    `compute_vertex_normals`, three.js's `computeVertexNormals`) averages over
    all faces at a vertex. On a WELDED mesh that rounds off every CAD edge — a
    box top renders with an X-shaped crease fanning out of its corners. A cube
    has 8 positions and 6 distinct normals, and both must survive into the file.
    """
    verts, faces = _unit_cube()
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "cube.obj"), verts, faces,
                  corner_normals=_exact_cube_normals(verts, faces))
    txt = (d / "cube.obj").read_text()

    assert txt.count("\nv ") == len(verts) == 8
    vn = [ln for ln in txt.splitlines() if ln.startswith("vn ")]
    assert len(vn) == 6, vn          # exactly the six axis directions
    assert {tuple(round(float(c)) for c in ln.split()[1:]) for ln in vn} == {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    # No UVs supplied, so the `v//vn` form.
    assert "//" in txt and "vt " not in txt
    assert len([ln for ln in txt.splitlines() if ln.startswith("f ")]) == len(faces)


def test_supplied_uvs_are_written_and_produce_the_full_reference_form():
    verts, faces = _unit_cube()
    normals = _exact_cube_normals(verts, faces)
    # A distinct UV per corner, so nothing can dedup them away.
    uvs = [tuple((0.01 * (3 * i + k), 0.02 * (3 * i + k)) for k in range(3))
           for i in range(len(faces))]
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "cube.obj"), verts, faces,
                  corner_normals=normals, corner_uvs=uvs)
    txt = (d / "cube.obj").read_text()

    assert len([ln for ln in txt.splitlines() if ln.startswith("vt ")]) == 3 * len(faces)
    face_lines = [ln for ln in txt.splitlines() if ln.startswith("f ")]
    for ln in face_lines:
        for ref in ln.split()[1:]:
            v, t, n = ref.split("/")
            assert v and t and n, ln       # the full v/vt/vn form
    assert "//" not in txt


def test_uvs_are_all_or_nothing():
    """A partial UV set is worse than none: it smears the faces that lack one.

    So one missing corner drops `vt` for the whole mesh, while the normals —
    which do have a defined per-corner fallback — are unaffected.
    """
    verts, faces = _unit_cube()
    normals = _exact_cube_normals(verts, faces)
    uvs = [tuple((0.0, 0.0) for _ in range(3)) for _ in faces]
    uvs[3] = (uvs[3][0], None, uvs[3][2])          # one corner has no parameter
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "cube.obj"), verts, faces,
                  corner_normals=normals, corner_uvs=uvs)
    txt = (d / "cube.obj").read_text()
    assert "vt " not in txt
    assert "vn " in txt and "//" in txt


def test_a_corner_with_no_exact_normal_falls_back_to_its_triangle():
    """A cone apex and a sphere pole have parallel surface derivatives, so the
    kernel yields no normal there. That corner must still get a usable unit
    vector — a zero-length normal renders black — and it must not disturb the
    corners that do have one."""
    verts, faces = _unit_cube()
    normals = _exact_cube_normals(verts, faces)
    normals[0] = (None, normals[0][1], normals[0][2])
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "cube.obj"), verts, faces, corner_normals=normals)
    txt = (d / "cube.obj").read_text()
    for ln in txt.splitlines():
        if ln.startswith("vn "):
            length = math.sqrt(sum(float(c) ** 2 for c in ln.split()[1:]))
            assert abs(length - 1.0) < 1e-6, ln
    # Face 0's own plane normal is what the missing corner fell back to, so the
    # normal count is unchanged: it deduplicates against its two siblings.
    assert len([ln for ln in txt.splitlines() if ln.startswith("vn ")]) == 6


def test_a_fully_degenerate_face_still_writes_a_unit_normal():
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "degen.obj"), [(0.1, 0.2, 0.3)], [(0, 0, 0)],
                  corner_normals=[(None, None, None)])
    degen = (d / "degen.obj").read_text()
    assert "v 0.100000 0.200000 0.300000" in degen
    assert "vn 0.000000 0.000000 1.000000" in degen
    assert "f 1//1 1//1 1//1" in degen


def test_no_exact_normals_still_ships_normals_flat_rather_than_none():
    """An OBJ with no `vn` is the melted-wax case: the loader averages across
    every sharp edge. So a mesh with no exact normals available falls back to
    flat per-triangle shading, never to an unshaded file. A cube's six planes
    give six normals either way."""
    verts, faces = _unit_cube()
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "flat.obj"), verts, faces)
    flat = (d / "flat.obj").read_text()
    assert len([ln for ln in flat.splitlines() if ln.startswith("vn ")]) == 6
    assert "//" in flat and "vt " not in flat
    assert flat.count("\nv ") == len(verts)


def test_shading_is_byte_deterministic():
    """The OBJ is content-addressed, so a dedup order that varies between runs
    renames the asset and changes the object's id."""
    verts, faces = _unit_cube()
    normals = _exact_cube_normals(verts, faces)
    d = Path(tempfile.mkdtemp())
    GQ._write_obj(str(d / "a.obj"), verts, faces, corner_normals=normals)
    GQ._write_obj(str(d / "b.obj"), verts, faces, corner_normals=normals)
    assert (d / "a.obj").read_bytes() == (d / "b.obj").read_bytes()


# ---------------------------------------------------------------------------
# Carrying shading across the repair
# ---------------------------------------------------------------------------

def test_repair_tracks_which_preweld_node_each_surviving_corner_came_from():
    """The join key for shading. Welding merges nodes that share a position but
    not a normal, so after the weld only the pre-weld index can say which node a
    corner was — and a dropped triangle must take its entry with it, or every
    later triangle gets somebody else's normals."""
    verts, faces = _unit_cube()
    # Append a zero-area triangle, which `_drop_bad_faces` removes.
    faces = list(faces) + [(0, 0, 1)]
    r = GQ._repair_mesh(verts, faces, track_corners=True)
    assert r.corners is not None
    assert len(r.corners) == len(r.faces) == len(faces) - 1
    # Each tracked triple indexes the ORIGINAL (pre-weld) vertex list.
    for tri, orig in zip(r.faces, r.corners):
        assert len(orig) == 3
        for welded_i, orig_i in zip(tri, orig):
            assert r.verts[welded_i] == verts[orig_i]


def test_repair_reports_a_flip_so_the_caller_can_negate_its_normals():
    """A whole-mesh re-wind reverses which side is outward, so exact normals
    computed against the original winding point into the material afterwards.
    The flag is the only way a caller can know to negate them."""
    r_in = GQ._repair_mesh(_TET_V, _TET_IN, track_corners=True)
    assert r_in.flipped is True
    # The corner order is reversed in lockstep with the triangles.
    for tri, orig in zip(r_in.faces, r_in.corners):
        for welded_i, orig_i in zip(tri, orig):
            assert r_in.verts[welded_i] == _TET_V[orig_i]

    r_out = GQ._repair_mesh(_TET_V, _TET_OUT, track_corners=True)
    assert r_out.flipped is False


def test_tracking_is_off_by_default():
    r = GQ._repair_mesh(*_unit_cube())
    assert r.corners is None and r.flipped is False


def test_weld_key_matches_obj_serialization():
    """Welding must key off the WRITTEN coordinate, not an arithmetic rounding.

    Two vertices weld if and only if they serialize identically. Quantizing by
    arithmetic instead disagrees near a half-way boundary, which would leave two
    coincident *written* vertices unwelded — a seam in a mesh that looks closed.
    """
    # Both format to the same text but take different arithmetic keys.
    a, b = -0.0049965, -0.004996500000000001
    assert GQ._fmt_coord(a) == GQ._fmt_coord(b)
    welded, _faces, merged = GQ._weld_vertices([(a, 0, 0), (b, 0, 0)], [(0, 1, 0)])
    assert merged == 1 and len(welded) == 1

    # A signed zero names the same point as a positive one.
    assert GQ._fmt_coord(-0.0) == GQ._fmt_coord(0.0) == "0.000000"
    _w, _f, merged2 = GQ._weld_vertices([(-0.0, 0, 0), (0.0, 0, 0)], [])
    assert merged2 == 1


def _unit_box(ox: float, oy: float):
    """A unit cube at ``(ox, oy, 0)``, wound outward. 8 verts, 12 triangles."""
    verts = [(ox + x, oy + y, z) for (x, y, z) in
             [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return verts, faces


def _two_boxes(dx: float, dy: float):
    """Two unit cubes, the second offset by ``(dx, dy)``, as one vertex list.

    Returns ``(verts, faces, groups)`` in the shape `_solid_compile.tessellate`
    produces: the second body's vertices follow the first's, and `groups` is the
    cumulative end offset of each.
    """
    av, af = _unit_box(0, 0)
    bv, bf = _unit_box(dx, dy)
    return (av + bv,
            af + [(a + 8, b + 8, c + 8) for (a, b, c) in bf],
            [8, 16])


def test_a_weld_across_two_tangent_solids_manufactures_the_defect_it_reports():
    """Two bodies authored exactly tangent must survive the weld as two bodies.

    This is the whole reason `_weld_vertices` takes bounds. Tangent solids share
    coordinates along their contact, so an unbounded weld fuses their surfaces
    into one shell pinched along it — an edge belonging to four triangles, which
    QC then reports as a mesh that is not manifold. Nothing is wrong with the
    geometry: the volume is exactly 2.0 either way, and the kernel handed over
    two clean bodies. The defect is created here and nowhere else, which is why
    the fix belongs here rather than in the QC that notices it.

    Both contacts are covered because they fail differently. Sharing a square
    fuses four vertices and yields four non-manifold edges; sharing only a line
    fuses two and yields one — the shape of the real case, a boss tangent to a
    peg along a single edge.
    """
    for label, (dx, dy), n_welds, n_nonmanifold in (
            ("face-tangent", (1, 0), 4, 4),
            ("edge-tangent", (1, 1), 2, 1)):
        verts, faces, groups = _two_boxes(dx, dy)

        # Unbounded: the bodies fuse, and QC reports a defect that is not there.
        # The fused shell still encloses its volume exactly (every edge used an
        # even number of times, `bounds_volume` stays True) — what the weld
        # actually breaks is manifoldness, since the shared line is now used by
        # four triangles instead of two.
        _v, _f, merged = GQ._weld_vertices(verts, faces)
        assert merged == n_welds, label
        fused = GQ._mesh_qc(*GQ._repair_mesh(verts, faces)[:2])
        assert fused["n_shells"] == 1, label
        assert fused["n_nonmanifold_edges"] == n_nonmanifold, label
        # `is_manifold`, not `bounds_volume`, and the difference is the point of
        # the predicate split. The pinched shell is NOT a 2-manifold — an edge
        # belongs to four triangles — but it still encloses its volume exactly,
        # so every consumer that needs enclosure (the divergence theorem,
        # `point_inside_mesh`, the tangency erosion) is unaffected. A single
        # `closed` flag would conflate the two and report the manifold failure
        # as if enclosure had broken.
        assert fused["is_manifold"] is False, label
        assert fused["bounds_volume"] is True, label

        # Bounded: nothing crosses the boundary, and both bodies stay clean.
        _v2, _f2, merged2 = GQ._weld_vertices(verts, faces, groups)
        assert merged2 == 0, label
        kept = GQ._mesh_qc(*GQ._repair_mesh(verts, faces, groups)[:2])
        assert kept["n_shells"] == 2, label
        assert kept["is_manifold"] is True, label
        assert kept["bounds_volume"] is True, label
        assert kept["n_nonmanifold_edges"] == 0, label
        assert kept["n_boundary_edges"] == 0, label

        # The geometry is the same either way; only the verdict differs.
        assert abs(fused["volume_m3"] - 2.0) < 1e-12, label
        assert abs(kept["volume_m3"] - 2.0) < 1e-12, label


def test_the_weld_still_merges_coincident_vertices_inside_one_solid():
    """Bounding must not turn the weld off, only fence it.

    The weld's original job — OCCT tessellates each surface patch separately, so
    a smooth solid arrives as disjoint patches sharing no indices — is why every
    part reports hundreds of merges. A bound that also suppressed those would
    make every solid read as dozens of open shells.
    """
    verts, faces = _unit_box(0, 0)
    dupes = verts + [verts[0]]                 # a coincident vertex, same solid
    faces2 = list(faces) + [(8, 1, 2)]
    _v, _f, merged = GQ._weld_vertices(dupes, faces2, [9])
    assert merged == 1, "a duplicate inside one group must still weld"

    # And the same duplicate is NOT welded once a bound falls between them,
    # which is the only behavior the parameter changes.
    _v2, _f2, merged2 = GQ._weld_vertices(dupes, faces2, [8, 9])
    assert merged2 == 0


def test_shading_attributes_survive_every_step_of_the_repair():
    """The per-corner join key must be dropped and permuted with its triangles.

    The repair does not carry the coordinates themselves. It carries
    `RepairedMesh.corners` — one triple of PRE-WELD node indices per surviving
    triangle — and `_solid_compile._shading_per_corner` uses that to lift BOTH
    node-indexed streams, the exact normals and the UVs, to per corner
    afterwards. One join key rather than one array per attribute, which is why
    the normals need it too: a welded box corner is one position carrying three
    different normals.

    Two of the four repair steps can disturb the key, and they fail differently:

      * welding and dropping unused vertices RENUMBER vertices, and the key is
        not indexed by vertex, so both must be no-ops for it;
      * dropping a triangle must drop its row, and the inside-out flip reverses
        each triangle's winding, so the key must take the same permutation. A
        flip that left it in place would MIRROR the texture on exactly the parts
        this repair fires for — indistinguishable downstream from a badly
        authored material — and would point every normal into the material,
        which is what `flipped` exists to prevent.
    """
    # (a) A weld that renumbers vertices leaves the key untouched. Two
    #     face-tangent boxes, welded unbounded: four vertices merge and not one
    #     triangle is dropped, so the weld is the only step under test.
    tangent_v, tangent_f, _groups = _two_boxes(1, 0)
    r = GQ._repair_mesh(tangent_v, tangent_f, None, track_corners=True)
    assert any("welded" in a for a in r.actions), "the fixture must weld"
    assert len(r.verts) == len(tangent_v) - 4
    assert len(r.faces) == len(tangent_f)
    assert r.corners == [tuple(f) for f in tangent_f], (
        "a weld renumbers vertices, not corners")
    assert r.flipped is False

    # (b) A dropped triangle takes its key row with it. The added triangle is
    #     degenerate (three collinear points), so exactly one row goes.
    verts, faces = _unit_box(0, 0)
    flat = verts + [(0.5, 0.0, 0.0)]
    faces3 = list(faces) + [(0, 1, 8)]
    r3 = GQ._repair_mesh(flat, faces3, None, track_corners=True)
    assert any("zero-area" in a for a in r3.actions)
    assert len(r3.corners) == len(r3.faces) == len(faces)
    assert r3.corners == [tuple(f) for f in faces3[:len(faces)]], (
        "the surviving rows must keep their order")

    # (c) The inside-out flip permutes each triangle's corners; the key follows.
    r4 = GQ._repair_mesh(_TET_V, list(_TET_IN), None, track_corners=True)
    assert any("inside-out" in a for a in r4.actions), "the fixture must flip"
    assert r4.faces == [(a, c, b) for (a, b, c) in _TET_IN]
    assert r4.flipped is True, "the caller must be told to negate its normals"
    assert r4.corners == [(a, c, b) for (a, b, c) in _TET_IN], (
        "corners were reversed but their join key was not")

    # (d) Tracking is opt-in: without it there is no key and no cost.
    r5 = GQ._repair_mesh(tangent_v, tangent_f)
    assert r5.corners is None


def test_the_obj_writer_emits_no_vt_without_texture_coordinates():
    """No coordinates, no `vt` stream: the *texture* parameter existing does
    not add a `vt` line to an untextured mesh. (Every OBJ carries `vn`, so the
    claim is about `vt` alone.)
    """
    import tempfile
    verts, faces = _unit_box(0, 0)
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.obj", Path(d) / "b.obj"
        GQ._write_obj(str(a), verts, faces)
        GQ._write_obj(str(b), verts, faces, corner_uvs=None)
        assert a.read_bytes() == b.read_bytes()
        assert b"vt " not in a.read_bytes()
        assert b"vn " in a.read_bytes(), "normals are unconditional now"


def test_the_obj_writer_emits_vt_and_the_three_index_face_form():
    """`f v/vt/vn` — both attribute streams over one welded vertex list.

    The `vt` table is deduplicated on the WRITTEN text and emitted sorted. Sorted
    because the staged OBJ is content-addressed: an insertion-ordered table would
    depend on OCCT's face traversal, and the same part would stage under
    different digests on different days.
    """
    import tempfile
    verts, faces = _unit_box(0, 0)
    # Two distinct coordinates, each used by many corners.
    uv = [((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)) for _ in faces]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.obj"
        GQ._write_obj(str(path), verts, faces, corner_uvs=uv)
        lines = path.read_text().splitlines()
    vt = [l for l in lines if l.startswith("vt ")]
    fs = [l for l in lines if l.startswith("f ")]
    assert vt == ["vt 0.000000 0.000000", "vt 1.000000 0.000000"], vt
    assert all(c.count("/") == 2 and "//" not in c
               for line in fs for c in line.split()[1:]), fs[0]


def test_omitting_the_bounds_preserves_the_previous_behaviour_exactly():
    """`groups=None` is the unbounded weld, for callers with no solids.

    `_geometry_qc` is loaded by path from CAD children and from tests with
    hand-built meshes that never came from `tessellate`; those have no solid
    structure to respect and must keep welding as they always did.
    """
    verts, faces, _groups = _two_boxes(1, 0)
    a = GQ._weld_vertices(verts, faces)
    b = GQ._weld_vertices(verts, faces, None)
    c = GQ._weld_vertices(verts, faces, ())
    assert a[0] == b[0] == c[0] and a[1] == b[1] == c[1]
    assert a[2] == b[2] == c[2] == 4


# ---------------------------------------------------------------------------
# Repair: what is fixed, what is only reported
# ---------------------------------------------------------------------------

def test_global_inside_out_is_fixed_local_winding_is_only_reported():
    """Global flip is repaired; local winding inconsistency is reported only."""
    # (a) Fully inverted closed tetrahedron -> repaired.
    qc, actions, _notes, _v, _f = _measure(_TET_V, _TET_IN)
    assert qc["bounds_volume"] and qc["winding_consistent"] and qc["volume_m3"] > 0
    assert any("flipped inside-out orientation" in a for a in actions)

    # (b) The same tetrahedron, correctly wound -> nothing to do.
    qc, actions, _notes, _v, _f = _measure(_TET_V, _TET_OUT)
    assert qc["volume_m3"] > 0
    assert actions == []

    # (c) One face reversed -> DOCUMENTED behavior: detected, not repaired.
    one_flipped = [(0, 2, 1), (0, 3, 2), (0, 1, 3), (2, 1, 3)]
    qc, actions, _notes, _v, _f = _measure(_TET_V, one_flipped)
    # Edge-count topology still calls it closed, which is exactly why
    # `winding_consistent` has to be reported separately.
    assert qc["bounds_volume"] is True
    assert qc["winding_consistent"] is False
    assert not any("inside-out" in a for a in actions)


def test_shells_disagreeing_on_orientation_are_not_flipped():
    """Mixed per-shell orientation must be reported, never "fixed" wholesale.

    Separate shells share no edge, so `_winding_consistent` cannot relate them
    and the mesh's total signed volume is just a sum. Flipping on that sum
    would correct the inverted shell and BREAK the correct one.
    """
    qc, actions, notes, _v, _f = _measure(_TWO_TETS_V, _TET_A_OUT + _TET_B_IN)

    # Closed and locally consistent — neither existing signal catches this.
    assert qc["bounds_volume"] is True and qc["winding_consistent"] is True
    assert qc["n_shells"] == 2
    # Both shells are real solids: neither may be written off as degenerate,
    # which would drop it out of the orientation vote entirely.
    assert qc["n_degenerate_shells"] == 0
    # The trap: the sum is negative purely because the inverted shell is bigger.
    assert qc["volume_m3"] < 0

    # No flip was applied, and the decline is explained rather than silent.
    assert actions == []
    assert any("inside-out" in w and "preserve correct shells" in w
               for w in notes)


def test_uniformly_inverted_multi_shell_mesh_is_flipped():
    """Every shell inverted is unambiguous, so the whole mesh is flipped.

    Guards against over-tightening the gate to single-shell meshes: real
    multi-piece parts (the typewriter's 14-shell type-bar fan) are uniformly
    inside-out and should still be repaired.
    """
    qc, actions, notes, v, f = _measure(_TWO_TETS_V, _TET_A_IN + _TET_B_IN)
    assert qc["n_shells"] == 2
    assert qc["volume_m3"] > 0                            # corrected
    assert any("flipped inside-out orientation" in a for a in actions)
    assert notes == []                                    # nothing declined

    # EVERY shell ends up outward — not merely the sum, which a half-flip
    # could also satisfy.
    per_shell = [GQ._signed_volume(v, s) for s in GQ._shell_partition(v, f)]
    assert len(per_shell) == 2 and all(x > 0 for x in per_shell), per_shell


def test_a_cavity_wall_is_not_reported_as_an_inverted_shell():
    """A sealed void is represented by an inward-wound shell. That is correct
    geometry, not a defect, and the volume proves it: an outer 0.2 m cube with a
    0.18 m void measures exactly 0.2^3 - 0.18^3. Counting the void's wall as
    "inside-out" would tell the author of an ordinary hollow box to fix winding
    that is already right."""
    outer = [(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0),
             (0, 0, 0.2), (0.2, 0, 0.2), (0.2, 0.2, 0.2), (0, 0.2, 0.2)]
    inner = [(0.01, 0.01, 0.01), (0.19, 0.01, 0.01), (0.19, 0.19, 0.01),
             (0.01, 0.19, 0.01), (0.01, 0.01, 0.19), (0.19, 0.01, 0.19),
             (0.19, 0.19, 0.19), (0.01, 0.19, 0.19)]

    def box(o, flip):
        f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        f = [(a + o, c + o, b + o) if flip else (a + o, b + o, c + o)
             for (a, b, c) in f]
        return f

    verts = outer + inner
    faces = box(0, False) + box(8, True)     # void wall wound the other way
    qc, actions, notes, _v, _f = _measure(verts, faces)
    assert qc["n_shells"] == 2 and qc["bounds_volume"] is True
    # The wall subtracts the void, which is the whole point of its winding.
    assert abs(qc["volume_m3"] - (0.2 ** 3 - 0.18 ** 3)) < 1e-12
    assert actions == [] and notes == []


def test_flat_shells_are_reported_as_degenerate_not_inside_out():
    """A shell with no volume is a thickness defect, not an orientation defect.

    Reading the sign of a ~0 volume as an orientation misnames the problem: a
    correctly oriented body with flat teeth would be reported as having
    inside-out shells.
    """
    qc, actions, notes, _v, _f = _measure(_TET_PLUS_FLAT_V,
                                          _TET_A_OUT + _TET_B_IN)
    assert qc["bounds_volume"] is True and qc["winding_consistent"] is True
    assert qc["n_shells"] == 2
    assert qc["n_degenerate_shells"] == 1

    # The good shell is not flipped on account of the flat one's noise sign.
    assert actions == []
    assert any("enclose no volume" in w for w in notes), notes
    # And the defect is named correctly — no orientation claim at all.
    assert not any("inside-out" in w for w in notes), notes


def test_degenerate_shells_do_not_veto_a_legitimate_flip():
    """A flat shell must not block a flip the real shells unambiguously need.

    Flipping a volume-less shell changes no geometry, so it has no stake in the
    vote. Counting it as "not inverted" would leave a genuinely inside-out mesh
    unrepaired.
    """
    qc, actions, notes, _v, _f = _measure(_TET_PLUS_FLAT_V,
                                          _TET_A_IN + _TET_B_IN)
    assert qc["n_degenerate_shells"] == 1
    assert qc["volume_m3"] > 0                            # the solid corrected
    assert any("flipped inside-out orientation" in a for a in actions)
    # The flip happened AND the unrelated defect is still reported.
    assert any("enclose no volume" in w for w in notes)


def test_lossless_cleanups_are_reported_when_they_change_nothing():
    """`actions == []` means "ran and changed nothing", never "did not run".

    `compile_entry` turns that distinction into `repair_status`, which is how a
    reader tells a clean mesh from an unmeasured one.
    """
    verts, faces = _unit_cube()
    v, f, actions, notes = GQ._repair_mesh(verts, faces)[:4]
    assert (v, f) == (verts, faces)
    assert actions == [] and notes == []

    # A duplicated vertex and a zero-area triangle are both removed, and named.
    dupe = list(verts) + [verts[0]]
    extra = list(faces) + [(0, 0, 0)]
    _v2, _f2, actions2, _n2 = GQ._repair_mesh(dupe, extra)[:4]
    assert actions2, "a weld and a degenerate-face drop must both be reported"
