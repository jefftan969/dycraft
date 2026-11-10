"""FCL narrow phase: tangency policy, pose forms, and agreement with B-rep.

The cheap tests pin the behavior that decides whether this backend can stand in
for the B-rep adjudicator above the object layer:

* **Tangency.** FCL's collision query is a tolerance-free boolean, so two bodies
  exactly touching — most of every scene — come back as a hit. The filter is a
  depth threshold at `TANGENCY_DEPTH_M`, which is the same 1 µm the B-rep path
  states as a volume (`tangency_vol_eps` multiplies it by the contact patch area
  because a boolean returns a volume). Without this, every resting contact is an
  interference and agents go back to inventing clearances they cannot justify —
  the exact behavior `tangency_vol_eps` was written to end.
* **Pose forms.** Kernels here hand back `(translation, quaternion wxyz)`;
  a quaternion misread as `(x,y,z,w)` rotates the body and *still* returns
  plausible-looking depths, so the two spellings are asserted equal rather than
  assumed.

The differential is the one that matters: the same pairs through both narrow
phases, asserting they agree *and* that the shared verdict is the right one. It
builds its own objects rather than sweeping a corpus (the comment above
`_DIFFERENTIAL_CASES` says why), and skips
only when there is no CAD interpreter to compare against.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from dycraft.verify import _narrow_phase as fc
from dycraft.verify._exact_adjudicator import TANGENCY_DEPTH_M

# No skip guard: coal is a hard dependency of the scene and animation layers, so
# a machine that cannot run these cannot run those either, and a green suite here
# would be a lie about that.
assert fc.is_available(), "coal is required; see environment-*.yml"

_IDENTITY = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _box(w, l, h):
    v = np.array([[x, y, z] for x in (-w / 2, w / 2)
                  for y in (-l / 2, l / 2) for z in (-h / 2, h / 2)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    return v, f


@pytest.fixture(scope="module")
def stack():
    """A 50 mm block sitting on a 1 m table slab, both as BVHs."""
    table = fc.build_bvh(*_box(1.0, 1.0, 0.05))
    block = fc.build_bvh(*_box(0.05, 0.05, 0.05))
    return table, block


def _at(block_dz):
    """Block pose whose bottom face is `block_dz` from the table's top face."""
    return ((0.0, 0.0, 0.05 + block_dz), (1.0, 0.0, 0.0, 0.0))


def test_exact_tangency_is_not_a_penetration(stack):
    table, block = stack
    assert fc.penetration(table, _IDENTITY, block, _at(0.0)) is None


@pytest.mark.parametrize("overlap", [1e-9, 1e-8, 1e-7])
def test_sub_quantum_overlap_is_cleared(stack, overlap):
    """Below one position quantum an overlap is quantization, not interference."""
    table, block = stack
    assert fc.penetration(table, _IDENTITY, block, _at(-overlap)) is None


@pytest.mark.parametrize("overlap", [1e-4, 1e-3, 5e-3])
def test_real_overlap_is_reported_at_its_true_depth(stack, overlap):
    """Near contact the measured depth *is* the overlap, which is what makes a
    depth threshold a faithful restatement of the volume one.

    Within one position quantum, and **never over** it. The depth is measured by
    sliding the pair apart, and the pair slid is the one the verdict rests on —
    the eroded bodies — so the answer sits one erosion inset below the true
    overlap. That is the direction to be off in: this number is what a report
    calls the interference, and it should not claim more than the solids do.
    """
    table, block = stack
    hit = fc.penetration(table, _IDENTITY, block, _at(-overlap))
    assert hit is not None
    assert hit["method"] == "coal_bvh"
    assert hit["measured"] is True
    # One quantum for the erosion inset, one more because `_record` rounds the
    # record to micrometers and the band would otherwise turn on an ulp.
    budget = 2 * fc.TANGENCY_DEPTH_M
    assert overlap - budget <= hit["depth_m"] <= overlap + budget


def test_the_depth_is_measured_on_geometry_not_on_the_bounding_boxes(stack):
    """The AABB overlap is only where the search *starts*.

    A block dropped onto the corner of the table overlaps their boxes across the
    whole block, while the solids share a sliver. Reporting the box number is how
    a committed run came to tell an agent that a 100 mm ball was 230 mm inside a
    stand; the measured shift cannot exceed the geometry that has to move.
    """
    table, block = stack
    # Slid so only a corner of the block is over the table, and dropped 2 mm in.
    pose = ((0.49, 0.49, 0.05 - 2e-3), (1.0, 0.0, 0.0, 0.0))
    hit = fc.penetration(table, _IDENTITY, block, pose)
    assert hit is not None and hit["measured"] is True
    assert hit["depth_m"] == pytest.approx(2e-3, abs=fc.TANGENCY_DEPTH_M)


def test_the_measured_depth_never_exceeds_the_bound_it_replaced():
    """The one property that makes measuring safe to turn on.

    Both numbers are sound upper bounds, so the search is only ever allowed to
    *tighten*. The search slides along one world axis, which is often not a
    direction the two bodies come apart in at all: a tumbling block wants to be
    moved along a rotated face normal, and can collide at every shift out to the
    summed extent. Escalating past the bound would report numbers that are not
    depths of anything.
    """
    a = fc.build_bvh(*_box(0.20, 0.20, 0.10))
    b = fc.build_bvh(*_box(0.20, 0.20, 0.10))
    for overlap in (1e-5, 1e-3, 1e-2):
        pose = ((0.0, 0.0, 0.10 - overlap), (1.0, 0.0, 0.0, 0.0))
        hit = fc.penetration(a, _IDENTITY, b, pose)
        assert hit is not None
        # The AABB overlap is what the module reported before measuring, and
        # here it is exactly the true overlap. Never above it, by any route.
        assert hit["depth_m"] <= overlap + fc.TANGENCY_DEPTH_M, (
            f"a {overlap * 1000:g} mm overlap reported "
            f"{hit['depth_m'] * 1000:g} mm")


@pytest.mark.parametrize("gap", [1e-6, 1e-4, 2e-3])
def test_separated_pairs_report_an_exact_gap(stack, gap):
    table, block = stack
    assert fc.penetration(table, _IDENTITY, block, _at(gap)) is None
    out = fc.surface_gap(table, _IDENTITY, block, _at(gap))
    assert out["gap_m"] == pytest.approx(gap, rel=1e-6, abs=1e-9)


def test_quaternion_and_matrix_poses_agree(stack):
    """`(t, quat_wxyz)` and `(R, t)` must place a body identically."""
    table, block = stack
    ang = math.radians(37.0)
    q = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
    R = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                  [math.sin(ang), math.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    t = (0.01, 0.02, 0.05 - 1e-4)
    a = fc.penetration(table, _IDENTITY, block, (t, q))
    b = fc.penetration(table, _IDENTITY, block, (R, np.asarray(t)))
    assert (a is None) == (b is None)
    assert a["depth_m"] == pytest.approx(b["depth_m"], abs=1e-12)


def test_a_body_swallowed_whole_is_a_penetration():
    """The worst overlap there is, and the one a crossing test cannot see.

    A BVH answers triangle intersections. Nest one box entirely inside another
    and no triangle crosses, so the collision query alone reports CLEAR.
    Containment is settled by probing points, because
    the absence of crossings already means every point is on the same side.
    """
    outer = fc.build_bvh(*_box(0.40, 0.40, 0.40))
    inner = fc.build_bvh(*_box(0.10, 0.10, 0.10))
    hit = fc.penetration(outer, _IDENTITY, inner, _IDENTITY)
    assert hit is not None, "a swallowed body must not read as clear"
    assert hit["containment"] is True

    # Symmetric, and only inside: nudged out past the wall it is a crossing
    # again, and moved clear away it is neither.
    assert fc.penetration(inner, _IDENTITY, outer, _IDENTITY) is not None
    crossing = fc.penetration(outer, _IDENTITY, inner,
                                  ((0.0, 0.0, 0.22), (1.0, 0.0, 0.0, 0.0)))
    assert crossing is not None and not crossing.get("containment")
    assert fc.penetration(outer, _IDENTITY, inner,
                              ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))) is None


def test_containment_keeps_the_box_bound_and_says_it_did_not_measure():
    """No shift search for a body swallowed whole.

    "Translate by this much" is the least useful thing to say about a part
    entirely inside another, and the inner body's own extent — which is what the
    AABB overlap reduces to when one box nests in the other — is already the
    honest scale. `measured` is what tells the two cases apart, so a reader is
    never left guessing which kind of number a record holds.
    """
    outer = fc.build_bvh(*_box(0.40, 0.40, 0.40))
    inner = fc.build_bvh(*_box(0.10, 0.10, 0.10))
    hit = fc.penetration(outer, _IDENTITY, inner, _IDENTITY)
    assert hit["containment"] is True
    assert hit["measured"] is False
    assert hit["depth_is_bound"] is True
    assert hit["depth_m"] == pytest.approx(0.10)


def test_geometry_is_eroded_once_per_key_not_once_per_body():
    """The cache exists because callers make a fresh `Body` per batch and frame.

    Erosion plus the BVH rebuild it forces is the expensive half of a query and
    is pure in `(vertices, triangles, depth)`, so naming the geometry is enough
    to reuse it. Asserted on identity rather than on a timing, which is the only
    form of this that does not go flaky on a loaded machine.
    """
    fc.clear_cache()
    verts, tris = _box(0.20, 0.20, 0.10)
    first = fc.build_bvh(verts, tris, key="unit-test-box")
    again = fc.build_bvh(verts, tris, key="unit-test-box")
    assert again is first, "a named body must not be rebuilt"
    assert first.shrunk(fc.TANGENCY_DEPTH_M) is again.shrunk(fc.TANGENCY_DEPTH_M)

    # A *fresh wrapper* over the same geometry is the case that actually happens:
    # `_answer_with_mesh` allocates one per adjudication batch.
    third = fc.build_bvh(verts, tris, key="unit-test-box")
    assert third.shrunk(fc.TANGENCY_DEPTH_M) is first.shrunk(fc.TANGENCY_DEPTH_M)

    # Unnamed geometry keeps the per-instance twin — a deformed body's vertices
    # really are new each frame, and caching them under any key would be wrong.
    loose = fc.build_bvh(verts, tris)
    assert loose is not first
    assert loose.shrunk(fc.TANGENCY_DEPTH_M) is not first.shrunk(
        fc.TANGENCY_DEPTH_M)


def _ring(radius, half_h, n, phase=0.0, inner=None):
    """A tessellated cylinder, or an annulus when `inner` is given."""
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + phase
    if inner is None:
        top = np.stack([radius * np.cos(th), radius * np.sin(th),
                        np.full(n, half_h)], 1)
        bot = np.stack([radius * np.cos(th), radius * np.sin(th),
                        np.full(n, -half_h)], 1)
        v = np.vstack([top, bot, [[0, 0, half_h]], [[0, 0, -half_h]]])
        ct, cb = 2 * n, 2 * n + 1
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i], [ct, j, i], [cb, n + i, n + j]]
        return v, np.array(f, np.int64)
    pts = []
    for z in (half_h, -half_h):
        pts.append(np.stack([inner * np.cos(th), inner * np.sin(th),
                             np.full(n, z)], 1))
        pts.append(np.stack([radius * np.cos(th), radius * np.sin(th),
                             np.full(n, z)], 1))
    v = np.vstack(pts)
    it, ot, ib, ob = 0, n, 2 * n, 3 * n
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[it + i, ot + i, ot + j], [it + i, ot + j, it + j],
              [ib + i, ob + j, ob + i], [ib + i, ib + j, ob + j],
              [ot + i, ob + i, ob + j], [ot + i, ob + j, ot + j],
              [it + i, it + j, ib + j], [it + i, ib + j, ib + i]]
    return v, np.array(f, np.int64)


def test_a_mating_fit_clears_at_the_tessellations_own_resolution():
    """The case the position quantum cannot reach.

    A shaft's facets sit inside the true shaft and a bore's bulge into the true
    hole, so two independently tessellated halves of a fit interpenetrate by
    about the sum of their chord errors while the solids clear. One IR quantum is
    1 microm and the chord error here is 24 microm, so eroding by the quantum
    leaves the fit reading as a penetration — and reading as a deep one, since
    the bound is the bore's whole thickness.

    Pinned at a designed clearance of 10 microm, which is the figure a
    recorded bicycle transmission's crank journal actually needs.
    """
    n, radius = 64, 0.02
    sagitta = radius * (1.0 - math.cos(math.pi / n))
    assert sagitta > 10e-6, "the fixture must be coarser than the clearance"
    # Half a segment out of phase, which is what independent tessellation gives.
    peg = fc.build_bvh(*_ring(radius - 10e-6, 0.025, n))
    bore = fc.build_bvh(*_ring(radius * 1.6, 0.015, n, phase=math.pi / n,
                               inner=radius))
    assert fc.penetration(peg, _IDENTITY, bore, _IDENTITY) is None

    # …and it is the erosion doing it, not the pair being clear to begin with.
    saved = fc.CHORD_ERROR_CAP_M
    try:
        fc.CHORD_ERROR_CAP_M = fc.TANGENCY_DEPTH_M     # quantum-only erosion
        fc.clear_cache()
        peg = fc.build_bvh(*_ring(radius - 10e-6, 0.025, n))
        bore = fc.build_bvh(*_ring(radius * 1.6, 0.015, n, phase=math.pi / n,
                                   inner=radius))
        assert fc.penetration(peg, _IDENTITY, bore, _IDENTITY) is not None
    finally:
        fc.CHORD_ERROR_CAP_M = saved
        fc.clear_cache()


def test_chord_erosion_never_costs_a_clearance_the_quantum_would_have_given():
    """The invariant that lets the deeper erosion be switched on at all.

    Where the averaged vertex normal is not a true inward direction the eroded
    body comes out locally *bigger*, and the deeper the erosion the more of that
    there is. A bigger body can only fail to clear, never clear falsely, so both
    erosions are sound clearing tests and `penetration` accepts either. Without
    that, deepening the erosion *lost* clearances: in a recorded three-block
    topple two stacked blocks share 3 microm, the quantum
    erosion clears them and the chord erosion does not.

    Asserted as a property over a swept contact rather than on one pose, since
    the cases where the two disagree are exactly the marginal ones.
    """
    a = fc.build_bvh(*_box(0.20, 0.20, 0.10))
    b = fc.build_bvh(*_box(0.20, 0.20, 0.10))
    saved = fc.CHORD_ERROR_CAP_M
    try:
        for overlap in (1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 1e-4, 1e-3):
            pose = ((0.0, 0.0, 0.10 - overlap), (1.0, 0.0, 0.0, 0.0))
            fc.CHORD_ERROR_CAP_M = fc.TANGENCY_DEPTH_M
            fc.clear_cache()
            shallow = fc.penetration(fc.build_bvh(a.verts, a.tris), _IDENTITY,
                                     fc.build_bvh(b.verts, b.tris), pose)
            fc.CHORD_ERROR_CAP_M = saved
            fc.clear_cache()
            deep = fc.penetration(fc.build_bvh(a.verts, a.tris), _IDENTITY,
                                  fc.build_bvh(b.verts, b.tris), pose)
            if shallow is None:
                assert deep is None, (
                    f"a {overlap * 1e6:g} microm overlap the quantum erosion "
                    f"cleared came back as {deep}")
    finally:
        fc.CHORD_ERROR_CAP_M = saved
        fc.clear_cache()


def test_a_flat_face_has_no_chord_error_and_a_crease_is_not_one():
    """Why this change does not touch the contacts the module already handled.

    The estimate is per vertex and comes from the angle between facets, so a box
    — all flat faces, all 90-degree creases — asks for nothing beyond the
    quantum, and every resting contact behaves exactly as it did. A tessellated
    curve asks for its sagitta, and `L*theta/8` reproduces that exactly for an
    arc rather than approximating it.
    """
    verts, tris = _box(0.20, 0.20, 0.10)
    wv, wf, face_n, edges, eorder, estart, counts = _welded(verts, tris)
    box_chord = fc._chord_error(wv, wf, face_n, edges, eorder, estart, counts)
    assert box_chord.max() == 0.0, "a box has no curvature to resolve"

    n, radius = 48, 0.05
    verts, tris = _ring(radius, 0.03, n)
    wv, wf, face_n, edges, eorder, estart, counts = _welded(verts, tris)
    got = fc._chord_error(wv, wf, face_n, edges, eorder, estart, counts).max()
    assert got == pytest.approx(radius * (1.0 - math.cos(math.pi / n)), rel=1e-3)

    # And it is the curvature that sets it, not how big the part happens to be:
    # the axial edges of a ten-times-taller cylinder turn by the same angle but
    # run along the ruling, where nothing curves. Measuring their own length
    # instead of the span across them scaled the answer by the height.
    tall = _ring(radius, 0.30, n)
    assert fc._chord_error(*_welded(*tall)).max() == pytest.approx(got, rel=1e-6)


def _welded(verts, tris):
    """`_erode`'s intermediate state, so `_chord_error` can be tested alone."""
    v = np.asarray(verts, float)
    f = np.asarray(tris)
    key = np.ascontiguousarray(np.round(v / fc._WELD_QUANTUM_M).astype(np.int64))
    _u, first, inv = np.unique(
        key.view([("x", np.int64), ("y", np.int64), ("z", np.int64)]).ravel(),
        return_index=True, return_inverse=True)
    wf = inv.reshape(-1)[f]
    wf = wf[(wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2])
            & (wf[:, 2] != wf[:, 0])]
    wv = v[first]
    p0, p1, p2 = wv[wf[:, 0]], wv[wf[:, 1]], wv[wf[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    if float(np.einsum("ij,ij->", p0, np.cross(p1, p2))) < 0.0:
        face_n = -face_n
    edges = np.sort(np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]],
                                    wf[:, [2, 0]]]), axis=1)
    ekey = edges[:, 0] * np.int64(len(wv)) + edges[:, 1]
    eorder = np.argsort(ekey, kind="stable")
    _uniq, estart, counts = np.unique(ekey[eorder], return_index=True,
                                      return_counts=True)
    return wv, wf, face_n, edges, eorder, estart, counts


def test_the_reported_depth_does_not_depend_on_which_body_is_first():
    """A depth that changes when the caller swaps its arguments is not a depth.

    The search slides B along the axis, and out one side is a different journey
    from back the other — measured 1.002 mm against 6.000 mm on the committed
    `bike_transmission`'s crank set and chainring marker. Both directions are
    tried and the nearer wins, which makes the answer a property of the pair.
    """
    a = fc.build_bvh(*_box(0.40, 0.10, 0.10))
    b = fc.build_bvh(*_box(0.05, 0.05, 0.05))
    # Buried near one end, so the two directions out are very different lengths.
    pose = ((0.16, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    one = fc.penetration(a, _IDENTITY, b, pose)
    other = fc.penetration(b, pose, a, _IDENTITY)
    assert (one is None) == (other is None)
    assert one["depth_m"] == pytest.approx(other["depth_m"], abs=1e-9)


def test_an_unerodible_mesh_caches_that_answer_too():
    """"No inside" is a result, not a miss.

    Imported robots carry open-shell meshes, and rediscovering that would cost
    a full weld and edge-parity pass on every frame they are near anything.
    `None` therefore has to round-trip through the cache
    without being mistaken for an empty slot.
    """
    fc.clear_cache()
    # An open sheet: two triangles, four boundary edges, no volume to erode into.
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    body = fc.build_bvh(verts, tris, key="unit-test-sheet")
    assert body.shrunk(fc.TANGENCY_DEPTH_M) is None
    assert ("shrunk", "unit-test-sheet", fc.TANGENCY_DEPTH_M, True) in fc._CACHE
    assert fc.build_bvh(verts, tris,
                        key="unit-test-sheet").shrunk(fc.TANGENCY_DEPTH_M) is None


def test_coplanar_faces_are_cleared_by_the_erosion_not_by_the_margin():
    """The case `security_margin` cannot see, and the reason it is asked twice.

    Two boxes of **equal footprint** stacked have coplanar side faces, and coal's
    coplanar triangle routine calls two triangles that merely share an edge an
    intersection. Measured, it does so at exact tangency for every margin from 0
    to -0.1 mm — so on this geometry the margin alone would report every resting
    contact in a scene as a penetration. Eroding both bodies by one quantum turns
    the flush contact into the 1 microm gap coal does resolve.

    Both halves are asserted, because either alone passes for the wrong reason: a
    module that had lost the erosion would fail the first block, and one that
    eroded far too deeply would fail the second.
    """
    same = fc.build_bvh(*_box(0.2, 0.2, 0.1))
    other = fc.build_bvh(*_box(0.2, 0.2, 0.1))

    def at(overlap):
        return ((0.0, 0.0, 0.1 - overlap), (1.0, 0.0, 0.0, 0.0))

    for overlap in (0.0, 1e-9, 1e-7):
        assert fc.penetration(same, _IDENTITY, other, at(overlap)) is None, \
            f"coplanar contact at {overlap} m is tangency"
    for overlap in (1e-5, 1e-3, 1e-2):
        assert fc.penetration(same, _IDENTITY, other, at(overlap)) is not None, \
            f"{overlap} m through a coplanar face is a real overlap"


def _tri(verts):
    return fc.build_bvh(np.array(verts, float), np.array([[0, 1, 2]], np.int64))


@pytest.mark.parametrize("gap_m", [0.395e-6, 0.884e-6])
def test_a_sub_micron_gap_is_not_a_crossing(gap_m):
    """coal's default `gjk_tolerance` (1e-6) calls a sub-micron gap a hit.

    This is the two-triangle reduction of a carcass and its liner standing
    under a micron apart after erosion, which the default tolerance reports as
    intersecting although the B-rep backend clears the pair.

    The consequence is not a micron-sized error. `_separating_shift` believes
    the crossing verdict and walks the liner along the object's depth until it
    stops, which is when the liner is nearly all the way out, reporting
    hundreds of millimeters for a sub-micron gap.
    """
    a = _tri([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 1.4]])
    b = _tri([[-0.3, gap_m, 0.0], [0.3, gap_m, 0.0], [0.0, gap_m, 1.4]])
    assert not fc._crosses(a, _IDENTITY, b, _IDENTITY, 0.0), \
        f"{gap_m * 1e6:.3f} microm of clear air is not a crossing"


@pytest.mark.parametrize("pierce_m", [1e-6, 1e-4, 1e-3])
def test_tightening_gjk_still_catches_a_real_pierce(pierce_m):
    """The other half of the single-variable experiment, and the reason the
    tolerance moves *down* rather than the erosion moving up.

    A triangle that genuinely crosses another's plane is still found at 1e-9 —
    and at 1 microm the 1e-6 default *misses* it, so tighter is the safe
    direction on both sides of contact rather than a trade.
    """
    a = _tri([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 1.4]])
    b = _tri([[-0.1, -pierce_m, 0.5], [0.1, -pierce_m, 0.5], [0.0, 0.05, 0.5]])
    assert fc._crosses(a, _IDENTITY, b, _IDENTITY, 0.0), \
        f"a {pierce_m * 1e3:.3f} mm pierce is a real crossing"


def test_the_gjk_tolerance_is_set_and_not_left_to_coals_default():
    """Pinned as a value because the failure mode of forgetting it is silent:
    the query still answers, just wrongly, and the wrong answer arrives
    magnified by the shift search rather than as a 1 microm discrepancy.
    """
    import coal
    assert fc._GJK_TOLERANCE == 1e-9
    assert coal.CollisionRequest().gjk_tolerance > fc._GJK_TOLERANCE, \
        "coal's default is no longer the looser value this constant exists to override"


def test_erosion_follows_the_surface_not_the_centroid():
    """A C-shape's jaws face inward, so scaling toward its center is not erosion.

    A clamp gripping a slab touches it exactly (the CAD child says CONTACT, zero
    shared volume), and a centroid-scaled "erosion" would drive the jaws further
    into the slab and report a penetration of tens of millimeters. Every jaw
    vertex must move *away* from the
    gap.
    """
    # A C in the xz plane: a back wall with two jaws, gripping toward -x.
    verts, tris = _box(0.10, 0.10, 0.10)
    body = fc.build_bvh(verts, tris)
    moved = fc._erode(body.verts, body.tris, 1.0e-3)
    assert moved is not None
    # Every vertex of a convex body moves strictly inside its own box.
    assert np.all(moved.min(axis=0) > body.verts.min(axis=0))
    assert np.all(moved.max(axis=0) < body.verts.max(axis=0))

    # Inward winding must not turn erosion into dilation.
    flipped = fc._erode(body.verts, body.tris[:, ::-1].copy(), 1.0e-3)
    assert np.allclose(np.sort(flipped, axis=0), np.sort(moved, axis=0))

    # An open sheet has no inside to move toward, and says so.
    assert fc._erode(body.verts, body.tris[:2].copy(), 1.0e-3) is None

    # But **non-manifold is not open**. Two lumps of one `solid` meeting along an
    # edge give it four triangles, and the body still bounds a volume. Demanding
    # exactly two would refuse to erode such parts -- a crank whose journal sits
    # in a bore, say -- so the mating fit erosion exists to clear would keep its
    # conservative verdict and read as a deep penetration.
    seam = np.vstack([body.tris, body.tris])   # every edge shared by four faces
    assert fc._erode(body.verts, seam, 1.0e-3) is not None

    # **Unwelded is not open either**, and this is the case imported geometry
    # actually presents: one vertex per face-corner, so no two triangles share an
    # index and every edge looks unpaired. Measured on the downloaded Franka, all
    # four of its *collision* meshes are written this way — without welding they
    # can never be eroded, and every flush contact on an imported robot keeps its
    # conservative verdict and reads as a penetration.
    split_v = body.verts[body.tris.reshape(-1)]
    split_t = np.arange(len(split_v), dtype=np.int64).reshape(-1, 3)
    split = fc._erode(split_v, split_t, 1.0e-3)
    assert split is not None, "a closed body written with split vertices is closed"
    # Same body, so the same set of moved positions — reached through a different
    # indexing. Welding must also fix the *normals*: an unwelded corner sees one
    # face instead of three, so a per-face normal would push it out along that
    # face rather than into the body.
    assert np.allclose(np.unique(np.round(split, 9), axis=0),
                       np.unique(np.round(moved, 9), axis=0))


def test_a_body_in_a_cavity_shares_no_material_with_the_shell_around_it():
    """Containment is about *material*, not about being surrounded.

    The trap is that both readings sound alike, and the probe that decides
    containment is a winding-number test whose answer is "inside the solid",
    not "inside the outline". A block sitting in a hollow box touches its ceiling
    and is enclosed on all six sides, yet the two bodies share nothing — and a
    containment test that read the AABB gate as the answer, or that probed
    without an inside test, would report the whole block as swallowed.
    """
    # A hollow cube: the outer skin, plus an inward-wound inner skin bounding a
    # 0.20 m cavity. Two shells, one closed body.
    ov, ot = _box(0.40, 0.40, 0.40)
    iv, it = _box(0.20, 0.20, 0.20)
    shell = fc.build_bvh(np.vstack([ov, iv]),
                         np.vstack([ot, it[:, ::-1] + len(ov)]).astype(np.int64))
    block = fc.build_bvh(*_box(0.05, 0.05, 0.05))

    # Flush with the cavity ceiling: enclosed, touching, sharing nothing.
    assert fc.penetration(shell, _IDENTITY, block,
                          ((0.0, 0.0, 0.075), (1.0, 0.0, 0.0, 0.0))) is None
    # Driven up through the ceiling into the shell's material: a real overlap.
    assert fc.penetration(shell, _IDENTITY, block,
                          ((0.0, 0.0, 0.085), (1.0, 0.0, 0.0, 0.0))) is not None


def test_a_deep_narrow_sliver_is_a_penetration_however_little_volume_it_holds():
    """The one place this backend is deliberately *stricter* than the CAD child.

    B-rep spends the quantum as a volume, against `tangency_vol_eps` = one
    quantum times the smaller part's largest face — a budget sized for 1 microm
    of numerical film spread over a whole contact patch. A corner digging into a
    face spends that budget differently: in a recorded block topple, two 47 mm
    blocks share 1.05e-9 m^3 in a sliver **0.22 mm deep**, which is under the
    face-area budget and was cleared as tangency.

    Here the quantum is a depth, so the budget cannot be spread thin and a
    quarter-millimeter interference is reported whatever its volume. Four
    committed runs flip from pass to fail on exactly this, and the difference is
    a policy, not a defect on either side — but it is the direction that reports
    interference rather than hiding it, and it is pinned so the two cannot be
    quietly reconciled by loosening this one.
    """
    plate = fc.build_bvh(*_box(0.20, 0.20, 0.02))
    # A thin blade on edge, sunk 0.22 mm into the plate's top face.
    blade = fc.build_bvh(*_box(0.20, 0.0004, 0.02))
    sunk = 0.00022
    hit = fc.penetration(plate, _IDENTITY, blade,
                         ((0.0, 0.0, 0.02 - sunk), (1.0, 0.0, 0.0, 0.0)))
    assert hit is not None, "0.22 mm is 220 quanta, not quantization"
    assert hit["depth_m"] == pytest.approx(sunk, abs=1e-9)
    # Volume shared is ~1.8e-11 m^3 — three orders under the B-rep budget for a
    # pair this size, which is the whole point of asserting it here.


def test_an_unmodelled_primitive_is_refused_not_guessed():
    """The direction that matters: approximating a shape we do not model would
    invent penetrations. A part carrying an unmodeled primitive (`cylinder`)
    must reach the caller's `unchecked_pairs` rather than a wrong verdict.
    """
    assert fc.PRIMITIVES == ("box",)
    with pytest.raises(ValueError):
        fc.primitive_bvh("cylinder", {"radius": 0.05, "length": 0.4})


def test_a_primitive_is_placed_by_its_world_pose(stack):
    """A box BVH must honor the SE3 it is given rather than assume identity —
    the same property the analytic path asserted, and the one whose failure mode
    is a body measured at the origin while it sits a meter away."""
    _table, block = stack
    box = fc.primitive_bvh("box", {"width": 0.2, "length": 0.2, "height": 0.2})
    here = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    far = ((10.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    assert fc.penetration(box, here, block, _at(-1e-3)) is not None
    assert fc.penetration(box, far, block, _at(-1e-3)) is None


def test_tangency_threshold_is_the_brep_constant():
    """One quantity, one constant — not two that happen to match.

    A private copy in `_mesh_collide` would drift from the B-rep policy
    silently, and the only symptom would be resting contacts failing in one
    layer and passing in another.
    """
    assert fc.penetration.__kwdefaults__["tol"] is TANGENCY_DEPTH_M


# ---------------------------------------------------------------------------
# Through the production path, not the module surface
# ---------------------------------------------------------------------------


def test_the_answerer_filters_tangency_at_the_quantum_not_the_query_tolerance():
    """`_answer_with_mesh` must spend `TANGENCY_DEPTH_M`, not the caller's `tol`.

    The adjudicators take `tol=DEFAULT_TOL_M` (1 nm) — the *distance* tolerance
    of the B-rep boolean query — and that value was once threaded through as the
    FCL tangency depth, a thousand times tighter than the quantum. A flush
    contact then came back PENETRATING and `_is_tangency` could not downgrade it,
    because these records carry no volume.

    Asserted on the answerer rather than end to end, and the reason is worth
    keeping: the SAT broad phase only reports depths at or above
    `_object_kernel._PENETRATION_EPS`, which *is* 1 microm, so two axis-aligned
    boxes can never present this band to the narrow phase — for a box the OBB
    depth is the true depth. The band is reachable only where the OBB
    over-reports (a round or shelled part, which is why exact adjudication exists
    at all): SAT flags 5 microm, the real surfaces share 0.2. Building that
    geometry to test one threshold would test the fixture instead.
    """
    from dycraft.verify import _exact_adjudicator as adj

    box = {"primitive": "box",
           "params": {"width": 0.2, "length": 0.2, "height": 0.1}}
    ident = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def verdict(overlap):
        q = [{"tag": "t", "a": box, "a_se3": ident,
              "b": box, "b_se3": ((0.0, 0.0, 0.1 - overlap),
                                  (1.0, 0.0, 0.0, 0.0))}]
        return adj._answer_with_mesh(q)["t"]["verdict"]

    assert verdict(1.0e-7) == "CLEAR", "100 nm is quantization, not interference"
    assert verdict(0.0) == "CLEAR", "exact tangency is not interference"
    assert verdict(1.0e-3) == "PENETRATING", "1 mm is a real overlap"


# ---------------------------------------------------------------------------
# Agreement with the B-rep adjudicator — the claim this backend rests on
# ---------------------------------------------------------------------------
#
# The objects are built here, in this file, rather than read from recorded
# runs. A test pinned to one run directory rots into a permanent skip once the
# directory is gone; a sweep over `runs/` reads whatever has accumulated on the
# developer's disk (it is gitignored), differs between sessions because other
# tests write into it, and can spend minutes on a single object whose OCCT
# boolean never finishes inside the adjudicator's timeout. The cases below
# cover both directions in which the two backends could differ:
#
#   * a mesh that is too **lax** would miss `overlap_5mm` or `sliver_100um`;
#   * a mesh that is too **aggressive** would flag `clearance_fit_in_a_bore`
#     or `body_in_a_cavity`, which is the mating-fit false positive that
#     `_chord_error` and the erosion machinery exist to prevent, and the
#     behavior whose absence sent agents back to inventing clearances.
#
# It runs in a couple of seconds on the warm CAD child and is not marked
# `expensive`, so it actually runs.
#
# **Known limitation this does not cover.** The mesh backend can report a part
# seated in a socket (a chair's spindles in its seat, a haft in a head) as
# overlapping by tens of millimeters where the B-rep calls it clear, when the
# staged meshes are not watertight even after vertex merging, which makes
# `_deform_eval.point_inside_mesh`'s winding number unreliable. None of the
# synthetic cases below reproduce it, so it is an open defect rather than
# something to pin.


#: name -> (source, whether the pair is a genuine interference)
#:
#: Each builds exactly two parts that the SAT broad phase flags, so the
#: adjudicator is always asked. Exact tangency is deliberately not here: it does
#: not survive the broad phase, and `test_exact_tangency_is_not_a_penetration`
#: above covers it against the narrow phase directly.
_DIFFERENTIAL_CASES = {
    # A post sunk 5 mm into its base: unambiguous shared material.
    "overlap_5mm": ("""
obj = Object("overlap")
a = obj.add_part("base", Box(0.30, 0.30, 0.04))
b = obj.add_part("post", Box(0.05, 0.05, 0.20))
sunk = obj.add_anchor(a, "sunk", translation=(0.0, 0.0, 0.02 - 0.005))
obj.declare_root(a)
obj.attach_fixed("j", parent=a, child=b, parent_anchor=sunk,
                 child_anchor=obj.anchor(b, "bottom"))
ir = obj.finalize()
""", True),

    # 100 um: above the 1 um tangency quantum but far below anything a
    # tessellation tolerance could explain away. The mesh backend must still
    # see it, or the erosion has been widened into a blindfold.
    "sliver_100um": ("""
obj = Object("sliver")
a = obj.add_part("base", Box(0.30, 0.30, 0.04))
b = obj.add_part("post", Box(0.05, 0.05, 0.20))
sunk = obj.add_anchor(a, "sunk", translation=(0.0, 0.0, 0.02 - 0.0001))
obj.declare_root(a)
obj.attach_fixed("j", parent=a, child=b, parent_anchor=sunk,
                 child_anchor=obj.anchor(b, "bottom"))
ir = obj.finalize()
""", True),

    # A 10 mm peg through a 12 mm bore. The AABBs interpenetrate completely and
    # the solids share nothing: the false positive this module exists to clear.
    "clearance_fit_in_a_bore": ("""
obj = Object("socket")
plate = obj.add_part("plate", Box(0.20, 0.20, 0.04) - Cylinder(0.012, 0.04))
peg = obj.add_part("peg", Cylinder(0.010, 0.08))
obj.declare_root(plate)
obj.attach_fixed("j", parent=plate, child=peg,
                 parent_anchor=obj.anchor(plate, "center"),
                 child_anchor=obj.anchor(peg, "center"))
ir = obj.finalize()
""", False),

    # A core floating inside a hollow shell: one AABB swallows the other whole,
    # and still no material is shared.
    "body_in_a_cavity": ("""
obj = Object("cavity")
shell = obj.add_part("shell", Box(0.20, 0.20, 0.20) - Box(0.16, 0.16, 0.16))
core = obj.add_part("core", Box(0.10, 0.10, 0.10))
obj.declare_root(shell)
obj.attach_fixed("j", parent=shell, child=core,
                 parent_anchor=obj.anchor(shell, "center"),
                 child_anchor=obj.anchor(core, "center"))
ir = obj.finalize()
""", False),

    # A spindle bottomed out in a mortise bored too shallow for it. Reads like
    # the clearance fit to a broad phase and to the eye; it is a real overlap.
    "seated_past_the_mortise_floor": ("""
obj = Object("mortise")
seat = obj.add_part("seat", Box(0.24, 0.24, 0.03)
                    - Pos(0, 0, -0.005) * Cylinder(0.0105, 0.02))
spindle = obj.add_part("spindle", Cylinder(0.010, 0.30))
socket = obj.add_anchor(seat, "socket", translation=(0.0, 0.0, 0.005))
obj.declare_root(seat)
obj.attach_fixed("j", parent=seat, child=spindle, parent_anchor=socket,
                 child_anchor=obj.anchor(spindle, "bottom"))
ir = obj.finalize()
""", True),
}


@pytest.fixture(scope="module")
def differential_objects():
    """Build every case once, on the warm CAD child, and stage its geometry.

    Module-scoped because the builds are the whole cost: five of them share one
    `import build123d` and the per-case adjudication is milliseconds.
    """
    from dycraft.agent.cad import object_worker as W
    if not W.is_available():
        pytest.skip("needs the pinned build123d interpreter")
    import tempfile
    built = {}
    for name, (source, expected) in _DIFFERENTIAL_CASES.items():
        run_dir = Path(tempfile.mkdtemp(prefix=f"dycraft_diff_{name}_"))
        out = W.run_object_source(source, run_dir=run_dir)
        assert out["ok"], (
            f"{name} failed to build: "
            f"{(out.get('build_error') or {}).get('summary') or out.get('message')}")
        built[name] = (out["ir"], run_dir, expected)
    return built


@pytest.mark.parametrize("case", sorted(_DIFFERENTIAL_CASES))
def test_the_mesh_backend_agrees_with_the_cad_child(case, differential_objects):
    """Same SAT candidates, both narrow phases, identical verdicts.

    Both sides go through `adjudicate_object_pairs`, which is the real path, so
    this covers `geometry_spec`, the cache key, and `_answer_with_mesh`'s record
    shape rather than just the two geometry kernels.

    The expected verdict is asserted as well as the agreement. Agreement alone
    would pass if both backends broke the same way — and they share the SAT
    candidate list, the erosion cache, and the tangency constant, so that is not
    a hypothetical.
    """
    from dycraft.sdk import object_query
    from dycraft.verify import _exact_adjudicator as adj

    ir, run_dir, expected = differential_objects[case]
    pairs = object_query.penetration_pairs(ir, poses=None)
    assert pairs, f"{case}: the broad phase flagged nothing to adjudicate"
    queries = [(p, None) for p in pairs]

    kept_brep, stats = adj.adjudicate_object_pairs(
        ir, queries, run_dir=str(run_dir), cache={}, backend=adj.BACKEND_BREP)
    assert stats.get("available"), "the CAD child could not adjudicate"
    assert not stats.get("unknown") and not stats.get("not_adjudicable"), stats

    kept_mesh, _ = adj.adjudicate_object_pairs(
        ir, queries, run_dir=str(run_dir), cache={}, backend=adj.BACKEND_MESH)

    brep = {(p.a, p.b) for p in kept_brep}
    mesh = {(p.a, p.b) for p in kept_mesh}
    assert mesh == brep, (
        f"{case}: mesh-only {sorted(mesh - brep)}, "
        f"B-rep-only {sorted(brep - mesh)}")
    assert bool(brep) is expected, (
        f"{case}: expected penetrating={expected}, both backends said "
        f"{bool(brep)}")




def test_the_object_layer_never_REPORTS_a_pair_on_mesh_evidence(monkeypatch):
    """The pin that keeps `object.py`'s *reported* pairs on the CAD child.

    The object prompt teaches the agent to size a fix from `exact_volume_cm3`, a boolean intersection no mesh query
    produces, so a pair reported on mesh evidence would say it overlaps and give nothing to fix it by.

    The object layer does use meshes: `_exact_adjudicator._brep_with_mesh_prefilter` asks them first, because a boolean
    between two thin shells can take a minute where the mesh takes milliseconds, and a batch timeout would leave only
    bounding-box verdicts. The prefilter cannot report a pair, by construction: `CLEAR` and `CONTACT` are
    `_CLEARING_VERDICTS`, so a mesh answer can only drop a pair; `PENETRATING`, `UNKNOWN`, a touching mesh distance and
    a part with no bounding box all escalate to `_invoke`, so the records that reach `_enrich`, the one place
    `exact_volume` is set, come from the boolean.

    So this asserts that the verdict still resolves, that the boolean adjudicates anything reported, and that no
    reported pair carries an `exact_volume_cm3` no boolean produced, with `DYCRAFT_COLLIDE_BACKEND=mesh` set (the
    variable must not drag the object layer onto a backend that cannot answer its question; an explicit request
    still wins). The object probe overlay reaches no narrow phase; its collision probes are SAT and AABB throughout.
    """
    from dycraft.sdk import _probes, _object_kernel as ok
    from dycraft.sdk.object_builder import Object, _reset_object_singleton
    from dycraft.verify import _exact_adjudicator as adj
    from dycraft.verify import object_verify as ov

    monkeypatch.setenv("DYCRAFT_COLLIDE_BACKEND", "mesh")
    reached_mesh, sent_to_brep = [], []
    for name in ("penetration", "surface_gap", "build_bvh", "primitive_bvh"):
        real = getattr(fc, name)

        def poisoned(*a, _n=name, _r=real, **k):
            reached_mesh.append(_n)
            return _r(*a, **k)

        monkeypatch.setattr(fc, name, poisoned)
    monkeypatch.setattr(adj, "_invoke",
                        lambda q, **k: sent_to_brep.append(len(q)))

    _reset_object_singleton()
    obj = Object("clash")
    body = obj.make_box("body", 0.40, 0.40, 0.10)
    lug = obj.make_box("lug", 0.10, 0.10, 0.10)      # buried in `body`
    obj.attach_fixed("seat", body, lug,
                     parent_anchor=obj.anchor(body, "center"),
                     child_anchor=obj.anchor(lug, "center"))
    obj.declare_root(body)
    ir = obj.finalize()
    _reset_object_singleton()
    assert ok.penetration_pairs(ir), "the fixture must give SAT something to flag"

    class Ctx:
        run_dir = None
        asset_base_dir = None
        adjudication_cache: dict = {}

    report = ov.no_self_penetration(ir, Ctx())
    assert report.verdict in ("pass", "fail")
    assert sent_to_brep, "the adjudication branch was not reached at all"
    # The property: nothing REPORTED may carry an exact number the mesh cannot
    # produce. `_invoke` is stubbed to return None here, so any pair that
    # survives does so on its SAT verdict and must say so.
    for row in report.evidence.get("pairs") or []:
        assert "exact_volume_cm3" not in row, (
            f"{row['a']}~{row['b']} was reported with an exact volume while "
            f"the boolean returned nothing — mesh evidence has reached "
            f"`_enrich`")
        assert not row.get("adjudicated"), (
            f"{row['a']}~{row['b']} claims adjudication the boolean did not do")

    # The probe half is a separate claim — the probe overlay reaches no narrow
    # phase AT ALL — so it is measured from zero rather than cumulatively.
    # `no_self_penetration` above consults the meshes through
    # `mesh_prefilter=True`, which is the verifier's flag alone: the probes
    # promise "the exact surface distance, the same B-reps NoSelfPenetration
    # judges with" and must not be answered from a mesh.
    reached_mesh.clear()

    names = [p["name"] for p in ir["parts"]]
    _probes.object_penetration_pairs(ir)
    _probes.object_pair_report(ir, names[0], names[1])
    _probes.object_within_report(ir, names[1], names[0])
    _probes.object_nearest_neighbors(ir, names[0])
    _probes.object_find_clearance_risks(ir)
    _probes.object_support_report(ir)
    assert reached_mesh == [], (
        "an object-layer probe reached the mesh narrow phase "
        f"({sorted(set(reached_mesh))}) — they are SAT and AABB throughout")




def test_the_probe_and_the_verifier_read_one_reported_depth():
    """The number an agent plans against must be the number it is judged on.

    `_probes._self_penetrations_internal` reads depth through the verifier's
    `reported_depth`, not `pair.depth` (SAT's oriented-box number), so the probe
    and the verifier both answer on triangles; on a robot's limbs the box
    number can be several times the triangle one.

    Pinned as identity rather than by re-measuring a bundle, because the risk
    is two definitions of one quantity, not the arithmetic in either.
    """
    from dycraft.sdk import _probes
    from dycraft.verify import _exact_adjudicator as adj
    import inspect

    assert adj.reported_depth(_FakePair(0.230, -0.029)) == pytest.approx(0.029)
    # SAT stands alone when nothing measured the triangles (B-rep backend, or a
    # pair that kept its conservative verdict).
    assert adj.reported_depth(_FakePair(0.230, None)) == pytest.approx(0.230)
    assert adj.reported_depth(_FakePair(0.230, 0.0)) == pytest.approx(0.230)
    # The probe must route through it, not re-derive it.
    src = inspect.getsource(_probes._self_penetrations_internal)
    rigid_branch = src.split("out +=")[0]
    assert "reported_depth" in rigid_branch
    assert "round(p.depth" not in rigid_branch


class _FakePair:
    """Minimal stand-in for a `PenetrationPair`: SAT depth plus the signed
    exact distance `_enrich` attaches on the mesh backend."""

    def __init__(self, depth, exact_distance):
        self.depth = depth
        self.exact_distance = exact_distance


def test_only_the_verifier_opts_into_the_mesh_prefilter(monkeypatch):
    """The prefilter is fast because it skips the boolean; the probes cannot
    afford that, because `object.md` promises them exactness by name.

    `gap(ir, '<a>', '<b>')` is documented as "the **exact** surface distance,
    the same B-reps `NoSelfPenetration` judges with". A probe answered from a
    mesh would make that sentence quietly false, so the flag defaults off and
    `no_self_penetration` is the only caller that sets it.
    """
    import inspect

    from dycraft.verify import _exact_adjudicator as adj
    from dycraft.verify import object_verify as ov

    sig = inspect.signature(adj.adjudicate_object_pairs)
    assert sig.parameters["mesh_prefilter"].default is False, (
        "the prefilter must be opt-in, or every probe inherits it")

    # The verifier opts in; nothing else in the tree does.
    assert "mesh_prefilter=True" in inspect.getsource(ov.no_self_penetration)
    import pathlib
    root = pathlib.Path(adj.__file__).resolve().parents[1]
    setters = []
    for path in root.rglob("*.py"):
        # Run directories and scratch trees hold copied source snapshots, not
        # callers.
        if path.relative_to(root).as_posix().startswith(("runs/", "tmp_")):
            continue
        # The defining module names the flag in its own comment and cannot
        # call itself.
        if path.name.startswith("test_") or path.name == "_exact_adjudicator.py":
            continue
        try:
            body = path.read_text()
        except OSError:
            continue
        if "mesh_prefilter=True" in body:
            setters.append(path.name)
    assert setters == ["object_verify.py"], (
        f"mesh_prefilter=True is set in {setters}; only the verifier may, "
        f"because only its reports carry a boolean-derived volume")


def test_the_shift_follows_the_contact_normal_not_just_the_thinnest_axis():
    """An oblique contact extracted sideways reports a distance too long.

    Both bodies are rotated, so the direction that separates them is a face
    normal and not a world axis, and the overlap is injected at a known depth
    so there is a truth to check against rather than a regression baseline.
    Sliding only along the thinnest AABB axis reported up to 12.7 mm for an
    overlap of at most 4; with the normals among the candidates the search
    recovers the injected depth.
    """
    import math
    import numpy as np
    from dycraft.verify import _narrow_phase as NP
    if not NP.is_available():
        pytest.skip("coal unavailable")

    def box(sx, sy, sz):
        v = np.array([[x, y, z] for x in (-sx/2, sx/2)
                      for y in (-sy/2, sy/2) for z in (-sz/2, sz/2)], float)
        t = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]], np.int32)
        return NP.build_bvh(v, t)

    def rot(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))

    shelf, part = box(.40, .30, .04), box(.16, .05, .05)
    base = rot(.37, -.52, .29)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        spin = rot(*rng.uniform(-1.2, 1.2, 3))
        sink = float(rng.uniform(.0001, .004))
        half = float(np.abs((part.verts @ spin.T)[:, 2]).max())
        off = np.array([rng.uniform(-.1, .1), rng.uniform(-.08, .08),
                        .02 + half - sink])
        record = NP.penetration(shelf, (base, np.zeros(3)),
                                part, (base @ spin, base @ off))
        if not record:
            continue
        assert record["measured"], "a shallow contact has a separating shift"
        # Never under the truth: the search reports a separating translation,
        # and one shorter than the overlap would not separate anything.
        assert record["depth_m"] >= sink - 1e-4
        worst = max(worst, record["depth_m"])
    assert worst <= .0045, (
        f"reported {worst*1000:.3f} mm for an overlap of at most 4 mm; the "
        "search is extracting along an axis rather than the contact normal")
