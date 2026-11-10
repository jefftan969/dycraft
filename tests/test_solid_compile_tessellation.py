"""Per-solid tessellation and the weld bound it exists to supply.

`_solid_compile.tessellate` triangulates a Shape one solid at a time and returns
the cumulative vertex end-offset of each. Those offsets are the only reason it
does so: they let `_geometry_qc._weld_vertices` refuse to merge across a solid
boundary, so two bodies authored exactly tangent stay two clean surfaces instead
of one shell pinched along their contact. `test_geometry_qc.py` covers that
directly, in
`test_a_weld_across_two_tangent_solids_manufactures_the_defect_it_reports`.

**The load-bearing claim is the equivalence.** Every staged OBJ digest, and
therefore every object id, depends on the per-solid walk reproducing what the
whole-Shape call produced — the same vertices, in the same order, with the same
triangle indices. OCCT triangulates per face either way, so it does; but "so it
does" is an argument, and the digests are worth an assertion. That is what this
file is for, and it covers the planar-imprint path too, since that is where the
shape reaching `tessellate` is a rebuilt copy rather than the author's own.

These need the pinned CAD interpreter and reach it through
`object_worker.run_cad_snippet`, which runs the body on the same warm pooled
child the rest of the suite builds on.
"""

from __future__ import annotations

import json

import pytest

# Imported in-process for the pure-Python phase helpers: `_solid_compile`
# reaches OCP only inside functions, so the module imports without CAD.
from dycraft.agent.cad import _solid_compile as SC
from dycraft.agent.cad import object_worker as W

_needs_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs the pinned build123d interpreter")


def _run(body: str) -> dict:
    """Run `body` under the CAD interpreter; parse its single JSON line.

    Goes through `run_cad_snippet`, so this shares the warm pooled child with
    every other CAD caller instead of spawning an interpreter and re-paying
    `import build123d` (~1.5 s) per test. The body needs no dycraft bootstrap
    for the same reason: the child already has the package bound.
    """
    out = W.run_cad_snippet(body)
    assert out["ok"], (
        f"{out['message']}\n{out['traceback'] or ''}\n"
        f"stderr={out['stderr'][-3000:]}")
    return json.loads(out["stdout"].strip().splitlines()[-1])


# Two disjoint bodies and a curved one, so the comparison covers both the
# multi-solid concatenation and faces whose tessellation is tolerance-driven.
_SHAPE = (
    "import build123d as bd\n"
    "from dycraft.agent.cad import _solid_compile as sc\n"
    "from OCP.BRepTools import BRepTools\n"
    "shape = (bd.Box(0.4, 0.2, 0.1)\n"
    "         + bd.Pos(1, 0, 0) * bd.Cylinder(radius=0.05, height=0.3))\n"
)


@_needs_cad
def test_per_solid_tessellation_reproduces_the_whole_shape():
    """Vertex for vertex, index for index — this is what holds the digests.

    Cleaned between the two calls so each is computed rather than read back
    from the triangulation OCCT caches on the `TopoDS_Shape`; without that the
    second call would agree with the first by construction and prove nothing.
    """
    out = _run(_SHAPE + (
        "whole_v, whole_t = shape.tessellate(1e-3, 0.1)\n"
        "whole_v = [sc.xyz(v) for v in whole_v]\n"
        "whole_t = [(int(a), int(b), int(c)) for a, b, c in whole_t]\n"
        "BRepTools.Clean_s(shape.wrapped)\n"
        "verts, faces, groups, _n, _uv = sc.tessellate(shape, 1e-3, 0.1)\n"
        "import json; print(json.dumps({\n"
        "    'same_verts': verts == whole_v,\n"
        "    'same_faces': faces == whole_t,\n"
        "    'n_verts': len(verts), 'n_whole': len(whole_v),\n"
        "    'n_solids': len(shape.solids()), 'groups': groups}))\n"
    ))
    assert out["n_solids"] == 2, "the fixture must actually be multi-solid"
    assert out["n_verts"] == out["n_whole"] > 0
    assert out["same_verts"], (
        "per-solid concatenation moved a vertex or reordered one; every staged "
        "OBJ digest and every object id depends on this holding")
    assert out["same_faces"], "per-solid concatenation renumbered a triangle"


@_needs_cad
def test_the_groups_mark_one_ascending_range_per_solid():
    """The contract `_weld_vertices` reads: ascending, one per solid, ending at
    the vertex count. A short or unsorted list would silently bound the weld at
    the wrong places rather than fail."""
    out = _run(_SHAPE + (
        "verts, faces, groups, _n, _uv = sc.tessellate(shape, 1e-3, 0.1)\n"
        "import json; print(json.dumps(\n"
        "    {'groups': groups, 'n_verts': len(verts),\n"
        "     'n_solids': len(shape.solids())}))\n"
    ))
    groups = out["groups"]
    assert len(groups) == out["n_solids"]
    assert groups == sorted(groups) and len(set(groups)) == len(groups)
    assert groups[-1] == out["n_verts"]
    assert groups[0] > 0, "a solid contributing no vertices would be a bug"


@_needs_cad
def test_the_planar_imprint_keeps_one_group_per_solid():
    """Imprinting splits FACES, never solids, so the offsets keep their meaning.

    Worth pinning because the imprinted shape is a rebuilt `TopoDS_Shape` that
    goes back through `_wrap_tds`: if that ever returned something whose
    `solids()` disagreed with the original, the weld would be bounded at
    coordinates that no longer correspond to bodies.
    """
    out = _run(_SHAPE + (
        "(verts, faces, groups, _n, _uv, tol_used, pme, n_imp,\n"
        " achieved) = sc.tessellate_with_planar_imprint(\n"
        "    shape, 'p', 1e-3, 0.1, 0.05)\n"
        "import json; print(json.dumps(\n"
        "    {'groups': groups, 'n_verts': len(verts), 'n_imprinted': n_imp,\n"
        "     'n_solids': len(shape.solids())}))\n"
    ))
    assert out["n_imprinted"] > 0, "the fixture must actually be imprinted"
    assert len(out["groups"]) == out["n_solids"]
    assert out["groups"][-1] == out["n_verts"]


@_needs_cad
def test_a_shape_with_no_solids_still_reports_one_group():
    """`validate_shape` does not admit one, but `tessellate` is public and is
    also reached through `_wrap_tds`. A contract that holds only for the shapes
    we expect is not a contract, so the fallback is pinned rather than assumed.
    """
    out = _run((
        "import build123d as bd\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "face = bd.Box(0.4, 0.2, 0.1).faces()[0]\n"
        "verts, faces, groups, _n, _uv = sc.tessellate(face, 1e-3, 0.1)\n"
        "import json; print(json.dumps(\n"
        "    {'groups': groups, 'n_verts': len(verts), 'n_faces': len(faces),\n"
        "     'n_solids': len(face.solids())}))\n"
    ))
    assert out["n_solids"] == 0
    assert out["n_verts"] > 0 and out["n_faces"] > 0
    assert out["groups"] == [out["n_verts"]]


@_needs_cad
def test_located_surface_normals_follow_geometry_once():
    """Location changes positions and normals once, including hollow faces.

    Comparing against the unlocated surface also covers reversed inner faces,
    sharp planar cuts and smooth poles without blending their normals.
    """
    # No `cad_bootstrap` prefix: `_run` goes through the warm pooled child,
    # which already has the package bound — as every other call here does.
    out = _run('''
import json
import build123d as bd
from dycraft.agent.cad import _solid_compile as sc
profile = ((bd.Ellipse(.024, .032) - bd.Ellipse(.023, .031))
           & (bd.Pos(.024, 0) * bd.Rectangle(.048, .128)))
shell = bd.revolve(bd.Plane.XZ * profile, axis=bd.Axis.Z, revolution_arc=180)
worst = 0.0
checked = 0
missing = 0
for shape in [bd.Box(.04, .03, .02), bd.Sphere(.024), shell]:
    v0, f0, g0, n0, _ = sc.tessellate(shape, .001, .12)
    for rotation in [(0, 0, 90), (0, 0, -90), (31, -47, 123)]:
        rot = bd.Rotation(*rotation)
        moved = bd.Pos(.1, -.2, .3) * rot * shape
        v1, f1, g1, n1, _ = sc.tessellate(moved, .001, .12)
        assert f0 == f1 and g0 == g1 and len(v0) == len(v1)
        for a, b in zip(n0, n1):
            if a is None or b is None:
                missing += 1
                assert a is None and b is None
                continue
            expected = bd.Vector(a).transform(bd.Matrix(rot.wrapped.Transformation()))
            worst = max(worst, max(abs(x-y) for x,y in zip(expected, b)))
            checked += 1
print(json.dumps(dict(worst=worst, checked=checked, missing=missing)))
''')
    assert out["checked"] > 1000
    assert out["worst"] < 1e-9


# ---------------------------------------------------------------------------
# Texture coordinates
#
# UVs are derived at staging and written into the OBJ, so the staged file is
# the thing to check — not a renderer, which is deliberately untouched by this
# milestone. Each test below reads the bytes `stage_solid` would receive.
# ---------------------------------------------------------------------------

_STAGE = (
    "import build123d as bd, hashlib, json\n"
    "from dycraft.agent.cad import _solid_compile as sc\n"
    "def stage(shape, scale_m=None, pme=None):\n"
    "    e = sc.measure_shape(shape, name='p', planar_max_edge=pme,\n"
    "                         texture_scale_m=scale_m)\n"
    "    return sc.compile_entry(e)['obj_bytes']\n"
    "def parse(b):\n"
    "    v, vt, f = [], [], []\n"
    "    for line in b.decode().splitlines():\n"
    "        p = line.split()\n"
    "        if not p: continue\n"
    "        if p[0] == 'v': v.append(tuple(float(x) for x in p[1:4]))\n"
    "        elif p[0] == 'vt': vt.append((float(p[1]), float(p[2])))\n"
    "        elif p[0] == 'f':\n"
    "            f.append([tuple(int(i) if i else 0 for i in c.split('/'))\n"
    "                      for c in p[1:4]])\n"
    "    return v, vt, f\n"
)


@_needs_cad
def test_a_textured_part_is_the_untextured_mesh_plus_coordinates():
    """The UV walk must not perturb the geometry it decorates.

    One face walk serves both cases — the exact normals are read off the B-rep
    whether or not a texture asked for coordinates — so `scale_m` selects
    nothing but whether the `vt` stream is emitted. This pins that: passing a
    tile size adds texture coordinates and moves nothing else. Same vertices,
    same triangle indices; only the `vt` stream is new.
    """
    out = _run(_STAGE + (
        "shape = (bd.Box(0.4, 0.2, 0.1)\n"
        "         + bd.Pos(1, 0, 0) * bd.Cylinder(radius=0.05, height=0.3))\n"
        "tv, tvt, tf = parse(stage(shape, scale_m=0.1))\n"
        "uv_, uvt, uf = parse(stage(shape, scale_m=None))\n"
        "print(json.dumps({\n"
        "    'same_verts': tv == uv_,\n"
        "    'same_indices': [[c[0] for c in x] for x in tf]\n"
        "                    == [[c[0] for c in x] for x in uf],\n"
        "    'untextured_has_no_vt': uvt == [],\n"
        "    'textured_vt': len(tvt),\n"
        "    'textured_form': len(tf[0][0]), 'untextured_form': len(uf[0][0])}))\n"
    ))
    assert out["same_verts"], "the UV walk moved a vertex"
    assert out["same_indices"], "the UV walk renumbered a triangle"
    assert out["untextured_has_no_vt"] and out["textured_vt"] > 0
    # `v/vt/vn` against `v//vn`: three components either way, but the middle one
    # is empty when there is no texture, which `parse` reads back as 0.
    assert out["textured_form"] == 3 and out["untextured_form"] == 3


@_needs_cad
def test_a_cylinder_wraps_at_metric_scale():
    """The case a projection cannot do, and the reason to read OCCT's own UVs.

    A cylinder's `u` is an angle, so `r * u` is arc length: the barrel's texture
    runs once around the circumference and the tile stays square. Triplanar
    projection would cut the same barrel into three flat patches with two seams.
    """
    out = _run(_STAGE + (
        "import math\n"
        "from OCP.BRep import BRep_Tool\n"
        "from OCP.BRepAdaptor import BRepAdaptor_Surface\n"
        "from OCP.GeomAbs import GeomAbs_SurfaceType as T\n"
        "from OCP.TopLoc import TopLoc_Location\n"
        "r, h, scale = 0.05, 0.3, 0.1\n"
        "cyl = bd.Cylinder(radius=r, height=h)\n"
        "cyl.mesh(1e-3, 0.1)\n"
        "res = {}\n"
        "for face in cyl.faces():\n"
        "    if BRepAdaptor_Surface(face.wrapped).GetType() != T.GeomAbs_Cylinder:\n"
        "        continue\n"
        "    poly = BRep_Tool.Triangulation_s(face.wrapped, TopLoc_Location())\n"
        "    pts = [sc.xyz(poly.Node(i)) for i in range(1, poly.NbNodes() + 1)]\n"
        "    tris = [(t.Value(1)-1, t.Value(2)-1, t.Value(3)-1)\n"
        "            for t in poly.Triangles()]\n"
        "    _needs, fn = sc._face_uv_mapper(\n"
        "        face, sc._face_mean_normal(pts, tris))\n"
        "    uv = []\n"
        "    for i, p in enumerate(pts):\n"
        "        node = poly.UVNode(i + 1)\n"
        "        s, t = fn(p, float(node.X()), float(node.Y()))\n"
        "        uv.append((s / scale, t / scale))\n"
        "    us = [a for a, b in uv]; vs = [b for a, b in uv]\n"
        "    res = {'u_span': max(us) - min(us), 'v_span': max(vs) - min(vs),\n"
        "           'want_u': 2 * math.pi * r / scale, 'want_v': h / scale}\n"
        "print(json.dumps(res))\n"
    ))
    assert abs(out["u_span"] - out["want_u"]) < 1e-9, (
        f"the barrel spans {out['u_span']} tiles around, not {out['want_u']} — "
        f"`u` was not converted from radians to arc length")
    assert abs(out["v_span"] - out["want_v"]) < 1e-9


@_needs_cad
def test_the_plane_lattice_survives_the_planar_imprint():
    """The reason planes are anchored at the part, not at their own origin.

    `imprint_planar_grids` replaces one panel with a grid of coplanar subfaces.
    OCCT gives each subface `(u, v)` from its OWN parametric origin, so using
    those natively would restart the tile lattice at every cell — a visible grid
    of texture discontinuities exactly where the imprint added detail, and only
    on deformable parts, which is where it would be hardest to attribute.

    Taking the AXES from the B-rep plane and the ORIGIN from the part makes the
    imprinted panel carry the same coordinates the whole panel did, so the
    imprinted mesh's UV range is the plain one's and every plain coordinate
    still appears.
    """
    out = _run(_STAGE + (
        "plain = parse(stage(bd.Box(0.4, 0.2, 0.1), scale_m=0.1))[1]\n"
        "imp = parse(stage(bd.Box(0.4, 0.2, 0.1), scale_m=0.1, pme=0.05))[1]\n"
        "print(json.dumps({\n"
        "    'plain_range': [min(u for u, v in plain), max(u for u, v in plain),\n"
        "                    min(v for u, v in plain), max(v for u, v in plain)],\n"
        "    'imp_range': [min(u for u, v in imp), max(u for u, v in imp),\n"
        "                  min(v for u, v in imp), max(v for u, v in imp)],\n"
        "    'imp_is_superset': set(map(tuple, plain)) <= set(map(tuple, imp)),\n"
        "    'n_plain': len(plain), 'n_imp': len(imp)}))\n"
    ))
    assert out["imp_range"] == out["plain_range"], (
        "the imprinted panel's UVs do not span what the whole panel did — the "
        "lattice restarted per subface")
    assert out["imp_is_superset"], (
        "an imprinted subface moved a coordinate the plain face had")
    assert out["n_imp"] > out["n_plain"], "the fixture must actually imprint"


@_needs_cad
def test_the_staged_obj_is_reproducible_and_follows_the_tile_size():
    """Content addressing needs the bytes to be a function of the inputs alone.

    The `vt` table is deduplicated and SORTED for exactly this reason: an
    insertion-ordered table would depend on OCCT's face traversal, and the same
    part would stage under different digests on different days.
    """
    out = _run(_STAGE + (
        "b = lambda s: bd.Box(0.4, 0.2, 0.1) if True else None\n"
        "a1 = stage(b(0), scale_m=0.1)\n"
        "a2 = stage(b(0), scale_m=0.1)\n"
        "c = stage(b(0), scale_m=0.25)\n"
        "u = stage(b(0), scale_m=None)\n"
        "print(json.dumps({'stable': a1 == a2, 'scale_matters': a1 != c,\n"
        "                  'texture_matters': a1 != u,\n"
        "                  'untextured_has_no_vt': b'vt ' not in u}))\n"
    ))
    assert out["stable"], "the same part staged twice produced different bytes"
    assert out["scale_matters"], "changing the tile size changed nothing"
    assert out["texture_matters"] and out["untextured_has_no_vt"]


# ---------------------------------------------------------------------------
# Per-part tile phase
#
# The bake is a pure function of (local geometry, tile size), so without a
# per-part phase two parts with the same shape and the same material would
# stage the same `vt` stream, and content addressing would collapse them into
# ONE OBJ: a park bench's ten slats would be one file showing one patch of
# grain.
#
# `part_uv_phase` is pure Python, so the first three of these need no CAD.
# ---------------------------------------------------------------------------

def test_the_phase_is_deterministic_and_inside_one_tile():
    """Determinism is the whole contract: the same source must stage the same
    bytes, on every recompile and on every machine."""
    for name in ("seat_slat_0", "body", "p", "a" * 200, "unicode-π-name"):
        a = SC.part_uv_phase(name)
        b = SC.part_uv_phase(name)
        assert a == b, f"{name!r} phased differently on a second call"
        assert len(a) == 2
        for c in a:
            assert 0.0 <= c < 1.0, f"{name!r} phased outside [0, 1): {a}"


def test_distinct_part_names_get_distinct_phases():
    """Part names are unique within an object (`object_builder` validates it),
    so this is what makes repeated slats stop matching."""
    names = [f"seat_slat_{i}" for i in range(5)] + \
            [f"back_slat_{i}" for i in range(5)]
    phases = [SC.part_uv_phase(n) for n in names]
    assert len(set(phases)) == len(names), "two slats share a phase"
    # And both components vary — `v` is the one that matters most, because a
    # long thin slat spans ~0.1 tiles across its width and so currently samples
    # one narrow strip of the image.
    assert len({p[0] for p in phases}) == len(names)
    assert len({p[1] for p in phases}) == len(names)


def test_an_untextured_part_is_never_phased():
    """The gate. `None` in, `None` out, so an untextured part's bytes are
    unchanged by the phase — which is what keeps every existing untextured
    object's id stable."""
    assert SC._phase_corner_uv(None, "anything") is None


def test_the_phase_shifts_every_corner_by_one_constant():
    """The invariant behind coplanar-subface continuity, asserted directly.

    `_face_uv_mapper` anchors a plane at the part origin so one tile lattice
    runs across every subface `imprint_planar_grids` cuts. A phase chosen per
    face would restart that lattice at each cell boundary. Adding ONE constant
    to every corner of every face cannot, and this measures exactly that: the
    per-corner difference between two phasings is a single vector.
    """
    uv = [((0.0, 0.0), (1.0, 0.5), (2.0, -3.0)),
          ((7.5, 1.25), (-0.5, 9.0), (0.125, 0.375))]
    a = SC._phase_corner_uv(uv, "left")
    b = SC._phase_corner_uv(uv, "right")
    deltas = {(round(bc[0] - ac[0], 12), round(bc[1] - ac[1], 12))
              for tri_a, tri_b in zip(a, b) for ac, bc in zip(tri_a, tri_b)}
    assert len(deltas) == 1, f"the shift was not uniform: {sorted(deltas)}"
    # Non-trivial, i.e. the two names really do differ.
    assert deltas != {(0.0, 0.0)}
    # Spans are untouched by a translation, which is why the existing
    # `test_a_cylinder_wraps_at_metric_scale` still holds.
    flat_a = [c for tri in a for c in tri]
    flat_uv = [c for tri in uv for c in tri]
    for i in (0, 1):
        assert (max(c[i] for c in flat_a) - min(c[i] for c in flat_a)) == \
            pytest.approx(max(c[i] for c in flat_uv) - min(c[i] for c in flat_uv))


_PHASE_STAGE = (
    "import build123d as bd, json\n"
    "from dycraft.agent.cad import _solid_compile as sc\n"
    "def stage(shape, name, scale_m=None):\n"
    "    e = sc.measure_shape(shape, name=name, texture_scale_m=scale_m)\n"
    "    return sc.compile_entry(e)['obj_bytes']\n"
    "def corners(b):\n"
    "    vt, out = [], []\n"
    "    for line in b.decode().splitlines():\n"
    "        p = line.split()\n"
    "        if not p: continue\n"
    "        if p[0] == 'vt': vt.append((float(p[1]), float(p[2])))\n"
    "        elif p[0] == 'f':\n"
    "            for c in p[1:4]:\n"
    "                i = c.split('/')[1]\n"
    "                out.append(vt[int(i) - 1] if i else None)\n"
    "    return out\n"
)


@_needs_cad
def test_two_identical_textured_parts_do_not_share_a_mesh():
    """The per-part phase, end to end.

    Same shape, same tile size, different part name — as a bench's ten slats
    are. Without the phase these would stage byte-identical OBJs and content
    addressing would collapse them into one file.
    """
    out = _run(_PHASE_STAGE + (
        "shape = bd.Box(1.55, 0.07, 0.03)\n"
        "a = stage(shape, 'seat_slat_0', 0.55)\n"
        "b = stage(shape, 'seat_slat_1', 0.55)\n"
        "u = stage(shape, 'seat_slat_0', None)\n"
        "v = stage(shape, 'seat_slat_1', None)\n"
        "ca, cb = corners(a), corners(b)\n"
        "d = sorted({(round(y[0] - x[0], 9), round(y[1] - x[1], 9))\n"
        "            for x, y in zip(ca, cb)})\n"
        "print(json.dumps({\n"
        "    'textured_differ': a != b,\n"
        "    'untextured_identical': u == v,\n"
        "    'same_corner_count': len(ca) == len(cb) and len(ca) > 0,\n"
        "    'n_deltas': len(d), 'delta': d[0],\n"
        "    'u_spread': max(x for x, _ in d) - min(x for x, _ in d),\n"
        "    'v_spread': max(y for _, y in d) - min(y for _, y in d),\n"
        "    'stable': a == stage(shape, 'seat_slat_0', 0.55)}))\n"
    ))
    assert out["textured_differ"], (
        "two identically-shaped textured parts still stage the same bytes; "
        "they would collapse into one file and show one patch of grain")
    assert out["untextured_identical"], (
        "the name leaked into an UNTEXTURED part's bytes — every existing "
        "object id in the corpus would move")
    assert out["same_corner_count"]
    # One constant per part, measured at the OBJ's own resolution. `_fmt_uv`
    # writes six decimals, so two corners a whole tile apart can land on
    # neighboring quanta and read back 1e-6 different — which is what a
    # bit-exact comparison here would trip over. The shift's exactness is
    # asserted in memory by
    # `test_the_phase_shifts_every_corner_by_one_constant`; what matters on
    # disk is that no corner moved by a DIFFERENT amount, and one quantum
    # either way is the tightest honest bound.
    assert out["u_spread"] <= 2e-6 and out["v_spread"] <= 2e-6, (
        "the phase was not one constant per part; coplanar-subface continuity "
        f"is broken (per-corner shift spread u={out['u_spread']:.3g} "
        f"v={out['v_spread']:.3g}, over {out['n_deltas']} distinct values)")
    assert out["delta"] != [0.0, 0.0]
    assert out["stable"], "the same part staged twice produced different bytes"


@_needs_cad
def test_the_phase_survives_the_planar_imprint_as_one_lattice():
    """The imprint is where a per-face phase would show, so check it there too.

    `imprint_planar_grids` cuts one panel into a grid of coplanar subfaces. If
    the phase were applied per face, the subfaces would each restart the tile
    lattice; one constant per part keeps the grid a subset of the ungridded
    coordinates, shifted.
    """
    out = _run(_PHASE_STAGE + (
        "shape = bd.Box(0.8, 0.6, 0.05)\n"
        "def uvs(name, pme):\n"
        "    e = sc.measure_shape(shape, name=name, planar_max_edge=pme,\n"
        "                         texture_scale_m=0.2)\n"
        "    return sc.compile_entry(e)['obj_bytes']\n"
        "plain = corners(uvs('panel', None))\n"
        "imp = corners(uvs('panel', 0.05))\n"
        "sp, si = set(plain), set(imp)\n"
        "print(json.dumps({\n"
        "    'imprint_happened': len(imp) > len(plain),\n"
        "    'plain_is_subset': sp <= si,\n"
        "    'n_plain': len(sp), 'n_imp': len(si)}))\n"
    ))
    assert out["imprint_happened"], "the fixture must actually imprint"
    assert out["plain_is_subset"], (
        "the imprinted subfaces no longer lie on the ungridded lattice — a "
        "per-face phase would do exactly this")


# ---------------------------------------------------------------------------
# The vertex-weld quantum's margin
# ---------------------------------------------------------------------------

@_needs_cad
def test_the_weld_quantum_clears_occts_own_vertex_disagreement():
    """`_OBJ_COORD_DECIMALS = 6` welds at 1 micron, and this checks that it is
    enough.

    Two triangles meeting at a shared tessellation point normally carry
    byte-identical coordinates, but on a spline-bounded face OCCT emits them
    from separate surface evaluations and they differ by float noise. If that
    noise ever exceeded the weld quantum the weld would silently miss, and
    `qc.n_shells` — which the prompt teaches authors to reason against, one
    per outside plus one per sealed cavity — would count a seam as a shell.

    Measured on a filleted box: 156 distinct-but-coincident pairs, the widest
    under 1e-9 m, three orders of margin under the 1e-6 key. Asserted at 1e-7
    to leave room for a kernel that is merely worse, not broken.
    """
    out = _run(
        "import math, json\n"
        "import build123d as bd\n"
        "from collections import defaultdict\n"
        "box = bd.Box(0.20, 0.04, 0.02)\n"
        "part = bd.fillet(box.edges().filter_by(bd.Axis.X), radius=0.004)\n"
        "v, f = part.tessellate(1e-3, 0.1)\n"
        "pts = [tuple(p) for p in v]\n"
        "G = 1e-4\n"
        "buckets = defaultdict(list)\n"
        "for i, p in enumerate(pts):\n"
        "    buckets[(round(p[0]/G), round(p[1]/G), round(p[2]/G))].append(i)\n"
        "worst = 0.0\n"
        "n = 0\n"
        "for idxs in buckets.values():\n"
        "    for a in range(len(idxs)):\n"
        "        for b in range(a + 1, len(idxs)):\n"
        "            d = math.dist(pts[idxs[a]], pts[idxs[b]])\n"
        "            if 0.0 < d < 1e-3:\n"
        "                worst = max(worst, d)\n"
        "                n += 1\n"
        "print(json.dumps({'worst': worst, 'n_pairs': n, 'n_verts': len(pts)}))\n"
    )
    from dycraft.agent.cad import _geometry_qc as gq

    quantum = 10.0 ** -gq._OBJ_COORD_DECIMALS
    assert out["n_pairs"] > 0, (
        "no near-coincident pair was produced, so this shape no longer "
        "exercises the margin — pick one that does rather than deleting the "
        "assertion")
    assert out["worst"] < quantum / 10.0, (
        f"OCCT disagrees with itself by {out['worst']:.3e} m at a shared "
        f"point, against a {quantum:.0e} m weld key. The weld is at risk of "
        f"missing coincident vertices, which inflates qc.n_shells")


# ---------------------------------------------------------------------------
# The coarsening ladder
# ---------------------------------------------------------------------------

_LADDER = (
    "import json, time\n"
    "import build123d as bd\n"
    "from dycraft.agent.cad import _solid_compile as sc\n"
    "def run(shape, tol):\n"
    "    t0 = time.perf_counter()\n"
    "    v, f, g, n, u, used = sc.tessellate_within_budget(shape, 't', tol, 0.1)\n"
    "    return {'tri': len(f), 'tol_used': used,\n"
    "            'ms': (time.perf_counter() - t0) * 1000}\n"
)


@_needs_cad
def test_asking_for_a_finer_mesh_never_returns_a_coarser_one():
    """The ladder is anchored on the default, not on the request.

    Geometric rungs anchored on the request made the outcome non-monotone in
    it: a shape over budget at 1e-4 coarsened through 4e-4, 1.6e-3, ... and
    landed coarser than the same shape asked at 1e-3. An author who asks for
    more fidelity and gets less cannot reason about the dial at all.
    """
    out = _run(_LADDER + (
        "res = {}\n"
        "for label, tol in (('default', 1e-3), ('finer', 1e-4),\n"
        "                   ('finest', 1e-5)):\n"
        "    res[label] = run(bd.Torus(0.06, 0.012), tol)\n"
        "print(json.dumps(res))\n"))
    counts = {k: v["tri"] for k, v in out.items()}
    assert counts["finer"] >= counts["default"], counts
    assert counts["finest"] >= counts["default"], counts


@_needs_cad
def test_an_unviable_fine_request_is_not_paid_for():
    """A fine tolerance the triangle budget cannot afford is not paid for.

    A first probe at the requested tolerance would be the most expensive mesh
    the part will ever produce, discarded whenever it blows the budget.
    Measured on this torus, `tolerance=1e-5` would spend about two minutes on a
    mesh thrown away for the 12 096-triangle one the default reaches in under a
    second — against the build time limit, a lost compile, not a slow one.
    """
    out = _run(_LADDER + (
        "res = {'coarse': run(bd.Torus(0.06, 0.012), 1e-3),\n"
        "       'fine': run(bd.Torus(0.06, 0.012), 1e-5)}\n"
        "print(json.dumps(res))\n"))
    # Same answer, and the fine request must not cost an order of magnitude
    # more to reach it.
    assert out["fine"]["tri"] == out["coarse"]["tri"]
    assert out["fine"]["ms"] < out["coarse"]["ms"] * 4, (
        f"the unviable fine request cost {out['fine']['ms']:.0f} ms against "
        f"{out['coarse']['ms']:.0f} ms for the default — it is being probed")


@_needs_cad
def test_a_fine_request_that_fits_is_still_honoured():
    """Guard on the other side: the cheap-first probe must not cost fidelity.

    A part small or simple enough to fit the budget at the requested tolerance
    should get exactly that tolerance, one extra probe having established it.
    """
    out = _run(_LADDER + (
        "res = {'default': run(bd.Cylinder(0.02, 0.10), 1e-3),\n"
        "       'fine': run(bd.Cylinder(0.02, 0.10), 1e-5)}\n"
        "print(json.dumps(res))\n"))
    assert out["fine"]["tol_used"] == 1e-5, out
    assert out["fine"]["tri"] > out["default"]["tri"], (
        "the finer tolerance produced no more triangles, so it was not used")


@_needs_cad
def test_a_coarser_request_than_the_default_is_taken_as_given():
    """Anchoring must not silently refine. An author asking for less gets it."""
    out = _run(_LADDER + (
        "print(json.dumps({'coarse': run(bd.Torus(0.06, 0.012), 1e-2)}))\n"))
    assert out["coarse"]["tol_used"] == 1e-2


@_needs_cad
def test_a_dead_rung_does_not_consume_a_coarsening_attempt():
    """Past some fineness the chord dial is not what binds — the angular one
    is — so a x4 rung can return the identical mesh.

    Measured across five shape families, four have at least one such rung, and
    on a torus and a filleted box it is the FIRST, which is also the most
    expensive probe. Charging it as an attempt would spend a quarter of the
    ladder learning nothing, and on a shape needing three real steps that is
    the difference between landing under budget and exiting over it.
    """
    from dycraft.agent.cad import _solid_compile as SC

    # Budget forced low so the ladder runs regardless of the production
    # figure: this test is about the ladder's mechanics, not about whether
    # today's cap happens to bite this shape.
    out = _run(_LADDER + (
        "sc.MAX_TRIANGLES = 8000\n"
        "res = {}\n"
        "for tol in (1e-3, 4e-3):\n"
        "    sh = bd.Torus(0.06, 0.012)\n"
        "    from OCP.BRepTools import BRepTools\n"
        "    BRepTools.Clean_s(sh.wrapped)\n"
        "    v, f = sh.tessellate(tol, 0.1)\n"
        "    res[str(tol)] = len(f)\n"
        "res['budgeted'] = run(bd.Torus(0.06, 0.012), 1e-3)\n"
        "print(json.dumps(res))\n"))
    # The premise: x1 and x4 really are the same mesh on this shape, so the
    # first rung is a no-op.
    assert out["0.001"] == out["0.004"], (
        "this torus no longer has a dead first rung; pick a shape that does "
        "rather than deleting the test")
    # And the ladder still lands under the forced budget despite that rung,
    # which it could only do by not charging for it.
    assert out["budgeted"]["tri"] <= 8000, out["budgeted"]


@_needs_cad
def test_an_unreachable_fine_request_is_estimated_rather_than_probed():
    """The guard that holds when the default fits the budget.

    Probing the default first only protects when the default is over budget.
    At a 50 000 budget a torus fits at the default, so a `tolerance=1e-5`
    request would be probed — and that probe takes minutes. Triangle count
    only grows as tolerance tightens, so an
    optimistic estimate that already exceeds the budget settles it for free.

    The estimate uses the *smallest* growth exponent measured across shape
    families, which is what makes it safe: it can only skip a request that no
    shape could bring under budget.
    """
    out = _run(_LADDER + (
        "res = {'coarse': run(bd.Torus(0.06, 0.012), 1e-3),\n"
        "       'unreachable': run(bd.Torus(0.06, 0.012), 1e-5),\n"
        "       'reachable': run(bd.Cylinder(0.02, 0.10), 1e-5)}\n"
        "print(json.dumps(res))\n"))
    # Skipped: same mesh as the default, and no 100x cost to reach it.
    assert out["unreachable"]["tri"] == out["coarse"]["tri"]
    assert out["unreachable"]["ms"] < out["coarse"]["ms"] * 4, (
        f"the unreachable request cost {out['unreachable']['ms']:.0f} ms — it "
        f"is still being probed")
    # Not skipped: a simple part has room for the finer mesh and gets it.
    assert out["reachable"]["tol_used"] == 1e-5, out["reachable"]


@_needs_cad
def test_the_ladder_stops_before_it_ruins_the_part():
    """A x256 coarsening would destroy the geometry.

    Converted into what the judge sees — sagitta through the camera formula,
    720 px frame — the rungs cost 0.07 px (x4), 0.05 px (x16) and 0.23-0.68 px
    (x64) of silhouette error. All invisible. The fourth rung adds **5.8 to
    9.4 px**, which is a visibly polygonal object, and six recorded parts
    reached it.

    So the ladder is capped at x64 and a shape that still does not fit is
    staged over budget. That is the right way round: exceeding the budget
    costs disk (65 bytes a triangle, and 6.1x the triangles is 1.10x the
    render time), while ruining the mesh costs the object.
    """
    from dycraft.agent.cad import _solid_compile as SC

    # The cap has to be a total factor, not an attempt count: a rung skipped
    # for making no progress still advances the tolerance without spending an
    # attempt, so `COARSEN_ATTEMPTS` bounds probe count and nothing else.
    # Capping attempts at 3 still let this torus reach x256.
    assert SC.COARSEN_MAX_FACTOR == 64.0

    # A budget low enough that even x64 cannot save this torus: it must come
    # back over budget rather than coarsened past x64.
    out = _run(_LADDER + (
        "sc.MAX_TRIANGLES = 500\n"
        "res = {'r': run(bd.Torus(0.06, 0.012), 1e-3)}\n"
        "res['x64_tol'] = 1e-3 * 64\n"
        "print(json.dumps(res))\n"))
    assert out["r"]["tri"] > 500, (
        "the part fitted, so this no longer exercises the cap")
    assert out["r"]["tol_used"] <= out["x64_tol"] * 1.0001, (
        f"coarsened to {out['r']['tol_used']}, past the x64 cap")
