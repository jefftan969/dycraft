"""Shared numpy-only contact erosion for mesh collision backends."""
import math
import numpy as np

_WELD_QUANTUM_M = 1e-9
_CREASE_ANGLE_RAD = math.radians(30)

def _weld_indices(vertices, quantum=_WELD_QUANTUM_M):
    """Exact quantized-coordinate groups, with a fast integer-sort candidate.

    Hashes are never trusted as equality: validate every assigned coordinate
    against its representative and use full row equality on any hash collision.
    This avoids three-column comparison sorting on the common path.
    """
    keys = np.ascontiguousarray(np.round(vertices / quantum).astype(np.int64))
    unsigned = keys.view(np.uint64)
    hashed = (unsigned[:, 0] * np.uint64(0x9E3779B185EBCA87)
              ^ unsigned[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
              ^ unsigned[:, 2] * np.uint64(0x165667B19E3779F9))
    _, first, inverse = np.unique(hashed, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if not np.array_equal(keys, keys[first[inverse]]):
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
    return first, inverse


def _erode(verts, tris, depth: float, *, chord: bool = True, chord_cap: float = 1e-3):
    """`verts` pulled inside their own surface, or None if it cannot be.

    At least `depth` everywhere, and by the mesh's own chord error where that is
    larger — see `_chord_error`. `chord=False` asks for the flat `depth`, which
    is what `penetration` uses for its second, shallower attempt.

    Erosion along **outward vertex normals**, not a scale about the centroid.
    The centroid version is wrong on exactly the geometry this exists for: a
    C-clamp gripping a slab has its jaws facing *inward*, so scaling toward the
    center drives the gripping faces further into the gripped part and turns a
    flush contact into a deeper one: the difference between agreeing with the
    CAD child that a clamp and a slab touch and reporting a centimeters-deep
    penetration where the two solids touch exactly.

    Normals are area-weighted (the cross product is left unnormalized, so a big
    triangle counts for more) and oriented by the mesh's signed volume, so an
    inward-wound import erodes inward rather than dilating. A mesh that is not
    closed — some edge shared by an odd number of triangles — gets `None`, and
    the caller keeps the conservative verdict rather than eroding in a direction
    it cannot justify. Six of the downloaded Franka's eighteen meshes are exactly
    that: high-poly *visual* shells with hundreds of boundary edges, which have no
    inside to move toward and should not pretend otherwise.

    The depth is capped at a quarter of the thinnest extent so a foil-thin part
    is not turned inside out by a micron.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(tris)
    if len(f) == 0 or len(v) < 4:
        return None
    # **Weld first, and on an imported mesh this is the whole ballgame.** Both
    # tests below are topological — which triangles share an edge, which faces
    # meet at a vertex — and a mesh written with one vertex per face-corner
    # shares nothing by index even though it is closed by geometry. An imported
    # robot's collision meshes are written that way (three vertices per
    # triangle, every edge unpaired), and they are the ones it is staged with.
    # Without this, every flush contact on an imported robot keeps its
    # conservative verdict and reads as a penetration.
    #
    # Welding also fixes the normals, which is the same bug wearing a different
    # hat: an unwelded corner accumulates one face's normal instead of all of
    # them, so eroding it would pull the corner out along a single face rather
    # than into the body.
    #
    # Integer-hash sorting proposes groups; full coordinate equality validates
    # them, with a row-sort fallback on hash collisions. Weld semantics and the
    # first original vertex representing each group remain unchanged.
    first, inv = _weld_indices(v)
    wf = inv[f]
    # Welding can collapse a sliver triangle to a line. Its normal is zero and it
    # would leave two edges unpaired, so it is dropped from the topology rather
    # than allowed to veto the whole body — the BVH still keeps it, since the
    # caller re-triangulates from the original `tris`.
    wf = wf[(wf[:, 0] != wf[:, 1]) & (wf[:, 1] != wf[:, 2])
            & (wf[:, 2] != wf[:, 0])]
    if len(wf) == 0:
        return None
    wv = v[first]
    edges = np.sort(np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]],
                                    wf[:, [2, 0]]]), axis=1)
    # One int64 per edge instead of a two-column lexsort. Exact: an index fits
    # far inside the 3e9 either side of the product that int64 allows, and the
    # pair is already sorted so `(lo, hi)` is unique per undirected edge. A
    # two-column lexsort compares column by column and would be most of
    # `_erode`'s time.
    ekey = edges[:, 0] * np.int64(len(wv)) + edges[:, 1]
    eorder = np.argsort(ekey, kind="stable")
    _uniq, estart, counts = np.unique(ekey[eorder], return_index=True,
                                      return_counts=True)
    if np.any(counts % 2):
        # An open sheet has no inside to move toward. Checked by edge parity
        # rather than by signed volume: a lone face far from the origin has a
        # perfectly healthy-looking volume, being the cone the origin subtends
        # over it, and would have been eroded in whichever direction that cone
        # happened to point.
        #
        # **Even, not exactly two.** A staged tessellation is closed but not
        # always manifold: where two lumps of one `solid` meet along an edge,
        # four triangles share it, and the body still bounds a volume. Demanding
        # exactly two would refuse to erode such a part, and a mating fit on it
        # (a crank journal in its bore) would keep its conservative verdict and
        # read as a deep penetration.
        return None
    p0, p1, p2 = wv[wf[:, 0]], wv[wf[:, 1]], wv[wf[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    if float(np.einsum("ij,ij->", p0, np.cross(p1, p2))) < 0.0:
        face_n = -face_n                  # inward-wound import; erode, not dilate
    # `np.bincount` rather than `np.add.at`, which falls back to an unbuffered
    # elementwise loop: same sum, 9.5 ms -> 3.4 ms on that body.
    idx = wf.ravel()
    rep = np.repeat(face_n, 3, axis=0)
    n = np.empty_like(wv)
    for c in range(3):
        n[:, c] = np.bincount(idx, weights=rep[:, c], minlength=len(wv))
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0.0)
    ext = v.max(axis=0) - v.min(axis=0)
    thin = ext[ext > 0.0]
    if not len(thin):
        return None
    # Per vertex, not one number for the body: `_chord_error` is zero on a flat
    # face and large on a coarse curve, and the same part routinely has both.
    # That is what keeps this change from touching the contacts the module
    # already handled — a box on a table erodes by the quantum exactly as before.
    # `max(depth, cap)` as the ceiling, not the cap alone: `np.clip` applies its
    # upper bound last, so a caller asking for more erosion than the cap would
    # have silently got less than it asked for rather than more.
    per_vertex = np.clip(_chord_error(wv, wf, face_n, edges, eorder, estart,
                                      counts) if chord else 0.0,
                         float(depth), max(float(depth), chord_cap))
    per_vertex = np.broadcast_to(per_vertex, (len(wv),)).copy()
    per_vertex = np.minimum(per_vertex, 0.25 * float(thin.min()))
    if not chord:
        # **`chord=False` is a flat depth, with no local cap and no inward
        # gate.** `penetration` clears a pair if *either* erosion clears it, so
        # adding the chord path below can only clear more pairs, never fewer. The
        # guards below both *reduce* erosion; at the quantum depth the local
        # cap would bite only sliver triangles with 1 microm edges.
        return v - per_vertex[inv, None] * n[inv]
    # Below here is the chord path only, which is why the two guards are
    # computed here rather than above: the fallback pays for neither.
    #
    # **Erosion has to be monotone, and an averaged normal is not always
    # inward.** Moving a vertex by -d*n leaves the solid across incident face k
    # unless `n . n_k >= 0`; where some incident face fails that, the average
    # points the wrong way and the vertex is held still instead. Being
    # *under*-eroded there is safe — the pair keeps its conservative verdict —
    # while moving outward invents overlap, by up to about a micron at
    # chord-error depths.
    unit_f = face_n / np.clip(np.linalg.norm(face_n, axis=1, keepdims=True),
                              1e-300, None)
    inward = np.full(len(wv), np.inf)
    np.minimum.at(inward, wf.ravel(),
                  np.einsum("ij,ij->i", n[wf.ravel()],
                            np.repeat(unit_f, 3, axis=0)))
    # **And capped locally, not just globally.** A vertex takes the largest sag
    # of the edges meeting it, so one shared between a coarse curved region and a
    # fine thin-walled one gets the coarse region's erosion applied where the
    # wall cannot absorb it, and the eroded body can come out *bigger*. A
    # vertex's shortest incident edge is the scale
    # it actually lives at, and a quarter of it is the most that can be taken off
    # without turning its neighborhood inside out. This never binds on the fits
    # this exists for: `w * theta / 8` with theta under the crease angle is
    # already at most a fifteenth of the vertex's own span.
    per_vertex = np.minimum(per_vertex, 0.25 * _shortest_edge(wv, wf))
    per_vertex = np.where(inward > 0.0, per_vertex, 0.0)
    # Scattered back to the caller's own indexing, because it rebuilds the BVH
    # with the original `tris`.
    return v - per_vertex[inv, None] * n[inv]


def _shortest_edge(wv, wf):
    """Length of the shortest triangle edge at each welded vertex.

    The local scale of the mesh, used to keep an erosion from exceeding the
    neighborhood it is applied to. `inf` for a vertex no surviving triangle
    references, which then erodes by nothing anyway — its accumulated normal is
    zero.
    """
    out = np.full(len(wv), np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        length = np.linalg.norm(wv[wf[:, a]] - wv[wf[:, b]], axis=1)
        np.minimum.at(out, wf[:, a], length)
        np.minimum.at(out, wf[:, b], length)
    return out


def _chord_error(wv, wf, face_n, edges, eorder, estart, counts):
    """Per-welded-vertex estimate of how far the real surface leaves the mesh.

    A triangulation is inscribed: its vertices lie on the surface and each facet sags inside it, which is why two
    independently tessellated parts of a fit interpenetrate where the solids clear. Below the sag, whether a pair
    touches or overlaps is decided by the phase of two tessellations rather than the geometry, so erosion by it is
    the scale at which a triangulation stops having an opinion.

    The estimate is per edge, `w * theta / 8`, from the angle `theta` between the two facets sharing it and their
    width `w` **across** the edge (each facet's opposite vertex to the edge line). For a circular arc it is exact:
    `theta = 2*pi/n` and `w = 2R sin(pi/n)` give `R(1 - cos(pi/n))`, the sagitta. Across, not along: a cylinder's
    axial edges turn by the rim's angle but run along the ruling, where nothing curves, so their own length would
    scale the answer by the cylinder's height. Edges sharper than `_CREASE_ANGLE_RAD` are the model's creases and
    contribute nothing; otherwise a cube's corners would read as a chord error of meters. A vertex takes the largest
    sag of its edges, so a part is eroded only where it is curved.

    Against exact facet-to-surface deviations, the estimate is 1.12x on cylinders (radii 20-300 mm, 16-64
    segments), just over, which is the harmless direction. On a lat/long sphere it is 0.36-0.38x: a doubly curved
    surface deviates by about the sum of its two principal contributions, and a per-edge measure sees one. That
    under-erodes spheres, which keeps more conservative verdicts; a fitted factor would over-erode everything else,
    and the fits this exists for (a shaft in a bore) are singly curved.

    A fit whose relief is within the chord error clears (a crank journal in its bore, as `backend=brep` agrees). A
    fit that wants more relief than the tessellation accounts for is still reported, a false positive: a wheel hub
    in a large rim wants about 2 mm where the rim's chord error gives under 1 mm. Raising `CHORD_ERROR_CAP_M` would
    clear it and swallow real overlaps of that size elsewhere, a policy call rather than a reason to inflate the
    estimate.
    """
    two = estart[counts == 2]
    if not len(two):
        return np.zeros(len(wv))
    unit = face_n / np.clip(np.linalg.norm(face_n, axis=1, keepdims=True),
                            1e-300, None)
    # `edges` stacks three blocks of one row per triangle, so a row's index
    # modulo the triangle count is the triangle it came from.
    tri = np.tile(np.arange(len(wf)), 3)[eorder]
    t1, t2 = tri[two], tri[two + 1]
    theta = np.arccos(np.clip(np.einsum("ij,ij->i", unit[t1], unit[t2]),
                              -1.0, 1.0))
    ea, eb = edges[eorder][two, 0], edges[eorder][two, 1]
    # The apex of a triangle is whichever of its three indices is not on the
    # edge, and they are distinct, so the sum gives it away without a search.
    apex1 = wf[t1].sum(axis=1) - ea - eb
    apex2 = wf[t2].sum(axis=1) - ea - eb
    origin = wv[ea]
    along = wv[eb] - origin
    along /= np.clip(np.linalg.norm(along, axis=1, keepdims=True), 1e-300, None)

    def across(apex):
        d = wv[apex] - origin
        return np.linalg.norm(
            d - np.einsum("ij,ij->i", d, along)[:, None] * along, axis=1)

    sag = np.maximum(across(apex1), across(apex2)) * theta / 8.0
    sag[theta > _CREASE_ANGLE_RAD] = 0.0
    out = np.zeros(len(wv))
    np.maximum.at(out, ea, sag)
    np.maximum.at(out, eb, sag)
    return out
