"""The narrow phase for every layer above the object layer.

One function answers "do these two placed bodies share material", whether either side is a rigid part, a deformed
part, or an analytic primitive: they are all triangles by the time they get here, so one narrow phase serves all of
them. A rigid body's BVH is built once and placed by an SE3; a deformed one's is rebuilt when its
vertices move.

The object layer keeps its B-rep adjudicator: `exact_volume_cm3`, the number `object.md` tells the agent to size a fix
from, is a boolean intersection only a CAD kernel produces. Above that layer B-rep is unavailable anyway, since an
imported robot description ships triangles and no `.brep`.

## The Three Questions

**Overlap.** `coal.collide` with `security_margin = -TANGENCY_DEPTH_M`, then, only for a pair it reports, the same
query with both bodies eroded (`_erode`). The margin is exact on most geometry but not on the commonest contact: for
two boxes of equal footprint, whose side faces are coplanar, coal's coplanar-triangle routine calls an edge shared by
two triangles an intersection at every margin from 0 to -0.1 mm, down to exact tangency. coal does resolve a real
gap of 1 µm on the same pair, so eroding both bodies turns a flush contact into a gap it can see and leaves a genuine
overlap overlapping. A pair survives only if the solids share material deeper than the erosion: DyCraft's tangency
rule expressed as geometry.

**Erosion depth** is one IR position quantum at least, and above it each vertex's own **chord error**
(`_chord_error`, bounded by `CHORD_ERROR_CAP_M`): a triangulation is inscribed, and the surface bulges outside it by
that much, so two independently tessellated halves of a fit interpenetrate by about the sum of their chord errors
while the solids clear. A 20 mm journal at 64 segments has a chord error of 24 µm against a 1 µm quantum, and would
read as deep overlap at a micron of design clearance without it. Both depths are tried and either clearing is
accepted: an averaged vertex normal is not always inward, so an eroded body can come out locally larger, which can
only fail to clear, so each test and their disjunction are sound.

**Containment.** A BVH answers crossings, and a body swallowed whole produces none, at any margin. It is decided by
probing points (`test_full_containment_is_confirmed_not_cleared`).

**Clearance.** `coal.distance`, exact to the tessellation in both directions. An overlapping pair returns 0.0, so
`penetration` is the other half.

## Depth

`Contact.penetration_depth` is not used: for coincident or perpendicular faces it returns the lateral extent of the
intersection, the same whole box width for a 100 nm overlap as for 1 mm. `coal.distance` has no signed distance for
BVH pairs (coal 3.0.3). So the depth is measured the way the gap is: `_separating_shift` slides one body and bisects
for the translation at which the pair comes apart. That is the negative half of `surface_gap`, one quantity through
zero, recorded as a negative `distance`. It is measured on the pair that carried the verdict (the eroded bodies when
erosion ran), so it understates the raw overlap by the sub-micron inset.

The AABB overlap is both the bracket and the ceiling: shifting by it separates the boxes, so a converged search comes
in under it. Non-convex geometry can re-engage, so bisection finds *a* separating translation, not necessarily the
least (`depth_is_bound` stays `True`; `measured` says which bound). A search that does not converge inside the bound
reports nothing and the bound stands: tumbling blocks often collide at every shift along a world axis, and widening
the interval would return the bodies' own size. The search also slides along the contact normals
(`_contact_normals`, `_ceiling_along`, the few tightest bisected, the axis pair always among them), because along a
world axis an oblique contact comes out several times too deep. Containment, and bodies that would not erode (where
the coplanar problem is unmitigated), keep the bound with nothing measured. One layer up,
`_exact_adjudicator.reported_depth` takes the smaller of this and SAT's OBB depth, which are loose in opposite cases.

## The Cache

Erosion is the expensive half of a query (hundreds of ms on a body of 150k triangles) and pure in `(vertices,
triangles, depth)`. A caller that names its geometry passes `key=` to `build_bvh`, and the body and its eroded twin are
memoized module-wide under that key, so a rigid part erodes once per process. `None` is cached too: a mesh that cannot
be eroded cannot be eroded next frame either. The key identifies the **geometry, not the part** (`_exact_adjudicator`
passes `_geometry_id`, a content digest), so parts sharing a mesh share the work. A deformed body passes no key.

## Where This Disagrees with the CAD Child

The tangency budget is a depth here and a volume there. `tangency_vol_eps` is one quantum times the smaller part's
largest face, spent against the true shared volume, so a corner digging 0.22 mm into a face clears in the CAD child
on 1e-9 m³ and is reported here
(`test_a_deep_narrow_sliver_is_a_penetration_however_little_volume_it_holds`). A volume bound from the AABB overlap
would not reconcile them: on an edge contact the true patch is a fraction of the box, so the bound stays over budget.
Whether one quantum is the right depth for a world pose, which arrives through an FK chain of quantized values, is a
policy question for the layer.

## The Limit

An overlap smaller than the tessellation's chord error is invisible, which is the unsafe direction; staged meshes
under-fill by tens of microns against a 5 mm default contact tolerance, and a coarse imported mesh by more. Eroding by
the chord error makes this consistent rather than a coin toss on the phase of two tessellations. `CHORD_ERROR_CAP_M`
is where that runs out: a mesh coarse enough to claim tens of millimeters of chord error is one this layer cannot
answer for, and a visible false positive beats quietly clearing an overlap that big. The scene and animation layers
pay this; the object layer answers from analytic geometry. A finer `tolerance` on `add_part` would shrink it at the
source, but nothing upstream tells the object agent that a fit downstream needs it.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ._exact_adjudicator import TANGENCY_DEPTH_M

try:                                    # pragma: no cover - import guard
    import coal as _coal
except Exception:                       # pragma: no cover
    _coal = None


#: Analytic primitives a part can carry (`sdk/_primitives.py`). Any other shape
#: is a `solid`, which carries its own triangulation.
PRIMITIVES = ("box",)

#: Vertices probed per body when deciding containment. One would be sound for a
#: single closed component; a `solid` authored as disjoint lumps can have one
#: lump inside while the probed vertex belongs to a lump outside, so a spread is
#: taken and the bound is stated rather than assumed.
_CONTAINMENT_PROBES = 32

#: Coordinates within this distance are the same vertex, for the topology tests
#: in `_erode`. A nanometer: far below any modeled feature, and seven orders
#: above the float noise of a meter-scale double, so it merges what an exporter
#: split and nothing else.
_WELD_QUANTUM_M = 1e-9

#: Above this dihedral angle an edge is a **crease** the model actually has, not
#: a step around a curve, and it contributes no chord error. The distinction is
#: not a judgement call on this corpus: over 1.56 million interior edges of the
#: committed meshes the angle is sharply bimodal — 86% below 15 degrees
#: (discretization) and an 11% spike at 80-100 degrees (box corners, fillet
#: boundaries), with 2.6% in the whole valley between. Anywhere in that valley
#: gives the same answer: moving the cut from 15 to 30 degrees moves the median
#: per-mesh chord error from 0.096 mm to 0.101 mm.
_CREASE_ANGLE_RAD = 30.0 * np.pi / 180.0

#: Ceiling on the chord-error erosion, whatever the mesh says.
#:
#: A coarse mesh really can have a chord error of tens of millimeters — a 24 m
#: fuselage reaches 88 mm, which is a truthful 0.4% of the part — and
#: eroding by that would silently swallow an 88 mm interpenetration. The two
#: failures are not symmetric: a false positive is visible to the agent and can
#: be argued with, a false negative is invisible. So the erosion stops at a
#: millimeter, which is a fifth of the layer's 5 mm default contact tolerance and
#: the same figure `ROOM_SLACK_M` already uses for "below this the layer does not
#: distinguish". Over a corpus of recorded meshes it binds on 3.6% of vertices,
#: across 106 of 459 meshes; where it binds the pair keeps the conservative
#: verdict it has today.
CHORD_ERROR_CAP_M = 1e-3

#: How far a containment probe is pulled off the surface, toward its own body's
#: centroid. Any positive fraction works — the point only has to leave the
#: boundary, where the winding number is 0.5 and the inside test is a coin toss.
#: Small enough that a thin body's probes stay well within it.
_PROBE_INSET = 1e-3

#: Minimum AABB overlap, on its thinnest axis, for a pair to be worth a narrow
#: phase. The rigid path has always had this — `_object_kernel._PENETRATION_EPS`
#: is the same 1 microm — while the deformable path's AABB broad phase did not,
#: so a pair touching with zero thickness reached a narrow phase the rigid path
#: never showed it. Aligning them costs nothing in coverage: a pair under this
#: floor cannot overlap by more than the floor, so `penetration` would clear it.
BROAD_PHASE_MIN_OVERLAP_M = TANGENCY_DEPTH_M

#: GJK's numerical tolerance for the crossing queries, set explicitly because
#: coal's default (1e-6) is loose enough to call a sub-micron gap a hit. This is
#: a *convergence* epsilon, not one of the clearing budgets above: tightening it
#: removes false intersections rather than admitting more, so it moves in the
#: opposite direction from `TANGENCY_DEPTH_M` / `CHORD_ERROR_CAP_M`. See
#: `_crosses_objs` for the run that pinned it.
_GJK_TOLERANCE = 1e-9

#: Bisection steps `_separating_shift` will spend. The loop also stops once the
#: bracket is under one quantum, which is what usually ends it: brackets start at
#: the AABB overlap, so a millimeter-deep contact needs ten steps, not thirty.
_SHIFT_ITERS = 30

#: How much geometry the shared cache may hold, counted in triangles across both
#: the bodies and their eroded twins. A budget rather than an entry count because
#: that is the thing that actually consumes memory: 64 entries is nothing for
#: boxes and gigabytes for imported robots. At ~90 bytes of vertices, indices and
#: BVH nodes per triangle this is on the order of 350 MB, and a whole catalog of
#: typical objects fits inside it several times over.
_CACHE_MAX_TRIS = 4_000_000

#: `(kind, key[, depth]) -> Body | None`, in least-recently-used order. `None` is
#: a cached *answer* ("this mesh has no inside"), not a cache miss, which is why
#: lookups go through a sentinel.
_CACHE: "OrderedDict[tuple, Body | None]" = OrderedDict()
_CACHE_TRIS = 0
_MISS = object()


def _cache_get(ck):
    hit = _CACHE.get(ck, _MISS)
    if hit is not _MISS:
        _CACHE.move_to_end(ck)
    return hit


def _cache_put(ck, body):
    global _CACHE_TRIS
    if ck in _CACHE:
        return
    _CACHE[ck] = body
    _CACHE_TRIS += 0 if body is None else len(body.tris)
    while _CACHE_TRIS > _CACHE_MAX_TRIS and len(_CACHE) > 1:
        _old, victim = _CACHE.popitem(last=False)
        _CACHE_TRIS -= 0 if victim is None else len(victim.tris)


def clear_cache() -> None:
    """Drop everything memoized. For tests that measure the cold cost."""
    global _CACHE_TRIS
    _CACHE.clear()
    _CACHE_TRIS = 0


def is_available() -> bool:
    """True if the backend imported. False is a configuration fact, not a
    verdict: the caller raises rather than reporting a clean pair."""
    return _coal is not None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Body:
    """A BVH plus the triangles it was built from.

    The vertices are kept because a BVH answers crossings and containment needs
    more than that. Cheap to hold: they are already in memory to build the model.

    `shrunk` is the same body eroded by one position quantum, built on first use
    and only ever on the collision path — see `penetration` for why the question
    has to be asked twice, and `_erode` for what "eroded" means on a mesh.

    `key` names the geometry, when the caller knows it. It moves the eroded twin
    out of this instance and into the module cache, which is the difference
    between eroding a rigid part once and eroding it once per `Body` wrapper —
    and callers make a fresh wrapper per batch, per frame, and per neighbor.
    """

    __slots__ = ("model", "verts", "tris", "key", "_shrunk", "_shrunk_at")

    def __init__(self, model, verts, tris, key=None):
        self.model, self.verts, self.tris = model, verts, tris
        self.key = key
        self._shrunk = None
        self._shrunk_at = None      # the depth `_shrunk` was built at, if any

    def shrunk(self, depth: float, *, chord: bool = True):
        # Keyed on the depth, so a caller that passes a different tolerance does
        # not silently get the first one's body back. In practice every caller
        # takes the default, which is the point: one tangency rule. `chord` is
        # part of the key for the same reason — the two are different bodies and
        # `penetration` asks for both.
        want = (float(depth), bool(chord))
        if self.key is not None:
            ck = ("shrunk", self.key) + want
            hit = _cache_get(ck)
            if hit is _MISS:
                v = _erode(self.verts, self.tris, depth, chord=chord)
                # No key on the twin: it is different geometry, and caching it
                # under this one's name would hand it back as the unshrunk body.
                hit = None if v is None else build_bvh(v, self.tris)
                _cache_put(ck, hit)
            return hit
        if self._shrunk_at != want:
            v = _erode(self.verts, self.tris, depth, chord=chord)
            self._shrunk = None if v is None else build_bvh(v, self.tris)
            self._shrunk_at = want
        return self._shrunk


def build_bvh(verts, tris, *, key=None) -> Body:
    """A BVH over part-**local** triangles. Place it per query with an SE3.

    Building is the expensive half and rigid geometry does not move, so such a
    caller builds once and passes poses. A deformed body rebuilds when its
    vertices move, which this module cannot avoid: measured, refitting the tree
    in place is only 1.2x faster than rebuilding it, because what dominates is
    moving vertices across the language boundary, not building the tree.

    `key` identifies the **geometry**, not the part, and opts the body and its
    eroded twin into the module cache. It has to be a content name — the
    adjudicator passes `_geometry_id`, whose mesh branch is the file's own digest
    — because handing two different meshes the same key returns the first one's
    triangles for the second one's query. Omit it and nothing is cached, which is
    the right answer for a deformed body whose vertices moved this frame.
    """
    if _coal is None:
        raise RuntimeError("coal is not installed")
    if key is not None:
        hit = _cache_get(("bvh", key))
        if hit is not _MISS:
            return hit
    v = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
    f = np.ascontiguousarray(np.asarray(tris, dtype=np.int64))
    vv = _coal.StdVec_Vec3s()
    for p in v:
        vv.append(p)
    tt = _coal.StdVec_Triangle()
    for a, b, c in f:
        tt.append(_coal.Triangle(int(a), int(b), int(c)))
    m = _coal.BVHModelOBBRSS()
    m.beginModel(len(f), len(v))
    m.addSubModel(vv, tt)
    m.endModel()
    body = Body(m, v, f, key)
    if key is not None:
        _cache_put(("bvh", key), body)
    return body


from ._mesh_erosion import _erode as _mesh_erode, _shortest_edge, _chord_error


def _erode(verts, tris, depth: float, *, chord: bool = True):
    return _mesh_erode(verts, tris, depth, chord=chord, chord_cap=CHORD_ERROR_CAP_M)


def primitive_bvh(primitive: str, params: dict) -> Body:
    """A `Body` for an analytic primitive, in the part's own local frame.

    A box's twelve triangles represent it **exactly**, so routing it through the
    mesh path costs no accuracy — which is what lets one narrow phase serve
    mesh-vs-mesh and mesh-vs-primitive alike.

    Cached on its own dimensions, since that is all a box is. Twelve triangles
    are cheap to build and cheap to erode, but a per-frame check calls this once
    per neighbor per frame, and a scene of boxes makes that the most repeated
    call in the module.
    """
    if primitive not in PRIMITIVES:
        raise ValueError(f"no geometry for primitive {primitive!r}")
    w, l, h = (float(params[k]) for k in ("width", "length", "height"))
    verts = np.array([[x, y, z] for x in (-w / 2, w / 2)
                      for y in (-l / 2, l / 2) for z in (-h / 2, h / 2)], float)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                     [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                     [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    return build_bvh(verts, tris, key=(primitive, w, l, h))


def _rotation(se3):
    """`(R, t)` from `((tx,ty,tz), (qw,qx,qy,qz))`, from `(R, t)`, or identity."""
    if se3 is None:
        return np.eye(3), np.zeros(3)
    a, b = np.asarray(se3[0], float), np.asarray(se3[1], float)
    if a.shape == (3, 3):
        return a, b
    q = b / (np.linalg.norm(b) or 1.0)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float), a


def _placed(body, se3):
    if _coal is None:
        raise RuntimeError("coal is not installed")
    R, t = _rotation(se3)
    tf = _coal.Transform3s()
    tf.setRotation(np.ascontiguousarray(R))
    tf.setTranslation(np.ascontiguousarray(t))
    return _coal.CollisionObject(body.model, tf)


def world_verts(body, se3) -> np.ndarray:
    """A body's vertices in world space."""
    R, t = _rotation(se3)
    return body.verts @ R.T + t


def world_aabb(body, se3):
    """A body's world AABB, for a caller's broad phase."""
    w = world_verts(body, se3)
    return w.min(axis=0), w.max(axis=0)


def worth_narrow_phase(box_a, box_b,
                       min_overlap: float = BROAD_PHASE_MIN_OVERLAP_M) -> bool:
    """Do two world AABBs overlap by at least the quantum on every axis?

    The broad-phase counterpart of `penetration`'s `security_margin`: a pair
    whose boxes barely graze cannot have real geometry overlapping by more than
    that graze, so the narrow phase would clear it as tangency and the call is
    wasted. Keeping this floor equal to the narrow phase's is what lets the two
    layers agree instead of one of them producing pairs the other always drops.

    The floor is lowered per axis to the thinner body's own extent, so a body
    that is flat in some direction — a cloth sheet, a rig collapsed onto a plane
    — is not filtered out on the very axis it has no thickness in. Without that
    a genuine deep intersection through a zero-thickness mesh reads as clear.

    Sound against an inflated box as well as a true one: an inflated box
    contains the real geometry, so the real overlap never exceeds the box
    overlap, and a pair dropped here would have been dropped on the real
    geometry too.
    """
    (alo, ahi), (blo, bhi) = box_a, box_b
    alo, ahi = np.asarray(alo, float), np.asarray(ahi, float)
    blo, bhi = np.asarray(blo, float), np.asarray(bhi, float)
    lo, hi = np.maximum(alo, blo), np.minimum(ahi, bhi)
    floor = np.minimum(min_overlap, np.minimum(ahi - alo, bhi - blo))
    return bool(np.all(hi - lo >= floor))


# ---------------------------------------------------------------------------
# The narrow phase
# ---------------------------------------------------------------------------


def penetration(body_a, se3_a, body_b, se3_b, *,
                tol: float = TANGENCY_DEPTH_M) -> dict | None:
    """Do two placed bodies share material? `None` when they do not.

    `tol` is the overlap a pair must exceed to count. It defaults to `TANGENCY_DEPTH_M`, one IR position quantum, the
    same constant the B-rep path spends as a volume, and callers should leave it alone. In particular they must not
    pass an adjudicator's `tol`, `DEFAULT_TOL_M` (1 nm), the *distance* tolerance of a boolean query, which would run
    this filter a thousand times too tight.

    A pair coal reports at `security_margin` is asked again with both bodies **eroded** (`_erode`), because coal's
    coplanar-triangle routine calls any flush contact an intersection at every margin, while it does resolve a 1 µm
    gap. A pair survives only if the solids share material deeper than the erosion, which is DyCraft's tangency rule
    stated in geometry; without it, two solids that touch exactly can be reported as tens of millimeters deep.

    The AABB overlap bound is consulted first, as a prefilter and as the start of `_separating_shift`'s bracket, so
    the threshold and the reported depth's upper end are one quantity. It is sound in the direction that matters:
    separating along the thinnest overlap axis separates the boxes, hence the bodies. The depth is then measured: the
    pair that carried the verdict is slid apart and bisected, the same question `surface_gap` answers on the other
    side of zero (the module docstring has the bracket's proof and the cases that keep the bound). A body that cannot
    be eroded, being open with no inside, keeps the conservative verdict.
    """
    if _coal is None:
        raise RuntimeError("coal is not installed")
    tol = abs(float(tol))
    wa = world_verts(body_a, se3_a)
    wb = world_verts(body_b, se3_b)
    if not worth_narrow_phase((wa.min(axis=0), wa.max(axis=0)),
                              (wb.min(axis=0), wb.max(axis=0)), tol):
        return None
    crossed = _crosses(body_a, se3_a, body_b, se3_b, -tol)
    # **Measured on the bodies as authored, not on the eroded proxies.** The
    # verdict comes from the eroded pair; the number describes the real ones.
    # Erosion is the mesh's own chord error and can reach a millimeter, and a
    # depth measured on the eroded pair would be that much light: no longer a
    # ceiling on the overlap, which is the one thing the report promises.
    witness = (body_a, body_b, -tol)
    if crossed:
        # **Either erosion clearing the pair clears it, and that is what makes
        # the chord-error depth safe to switch on.** Both are sound *clearing*
        # tests: each asks whether the solids, shrunk by an amount their own
        # triangulation cannot resolve, still share material. Where the averaged
        # vertex normal is not a true inward direction the eroded body can come
        # out locally *bigger*, and a bigger body only ever fails to clear — it
        # can never clear something the real one would not. So the failure mode
        # of the deeper erosion is a missed clearance, and asking the shallower
        # one too recovers it.
        #
        # Not hypothetical: two stacked blocks sharing 3 microm are cleared by
        # the quantum erosion and not by the chord erosion. Trying both keeps
        # every clearance the quantum alone gives.
        for use_chord in (True, False):
            sa = body_a.shrunk(tol, chord=use_chord)
            sb = body_b.shrunk(tol, chord=use_chord)
            if sa is None or sb is None:
                continue
            if not _crosses(sa, se3_a, sb, se3_b, 0.0):
                crossed = False
                break
    if not crossed:
        # Both ways of not crossing land here. Crossing queries are blind to
        # containment however they came to say no, and one shared exit is one
        # fewer place for that to be forgotten; the erosion branch costs nothing
        # extra, since a pair that reaches it has already failed the AABB gate
        # `_containment` opens with, or is genuinely worth the probe.
        return _containment(wa, wb, body_a.tris, body_b.tris)
    return _record(wa, wb, witness=(witness, se3_a, se3_b, tol))


def _crosses(body_a, se3_a, body_b, se3_b, margin: float) -> bool:
    """Do the two placed triangulations intersect, shrunk by `margin`?

    A negative margin is coal's own way of saying "and by more than this much".
    Contacts are not asked for: `Contact.penetration_depth` is unusable on BVH
    pairs (see the module docstring), and one contact is enough to answer a
    boolean.
    """
    return _crosses_objs(_placed(body_a, se3_a), _placed(body_b, se3_b), margin)


def _crosses_objs(obj_a, obj_b, margin: float) -> bool:
    """`_crosses` on already-placed objects, for the shift search.

    Split out so bisection places the moving body only: A's `CollisionObject` is
    the same at every step, and rebuilding it thirty times per reported pair is
    the one avoidable cost in a search that is otherwise pure queries.
    """
    req = _coal.CollisionRequest()
    req.security_margin = margin
    req.num_max_contacts = 1
    req.enable_contact = False
    # coal's own default is 1e-6, which calls a sub-micron gap an
    # intersection. After erosion, nested shells (a fridge's carcass and its
    # liner) can stand under a micron apart, and `_separating_shift` would then
    # walk the inner shell out along its whole depth and report that as the
    # penetration. At 1e-9 those gaps clear while a genuine pierce is still
    # caught down to 1 um, where the 1e-6 default *misses* it, so tighter is
    # the safe direction on both sides of contact. `epa_tolerance` does not
    # matter here: `enable_contact = False` means EPA is never entered.
    req.gjk_tolerance = _GJK_TOLERANCE
    res = _coal.CollisionResult()
    _coal.collide(obj_a, obj_b, req, res)
    return bool(res.isCollision())


#: Contact normals bisected beside the axis pair. Each costs a bisection and
#: they are screened by ceiling first, so a handful covers the useful ones.
_NORMAL_CANDIDATES = 2


def _contact_normals(obj_a, obj_b, margin: float, limit: int = 8):
    """Distinct SAT contact normals for an overlapping pair.

    Only directions are taken. ``Contact.penetration_depth`` beside them is
    the per-triangle-pair quantity the module docstring rejects; the distance
    is still bisected on ``_crosses_objs``.

    This is the one query here that asks for contacts, so it is the one that
    enters EPA — the verdict query next door keeps ``enable_contact = False``
    and its tolerance note continues to apply unchanged.
    """
    req = _coal.CollisionRequest()
    req.security_margin = margin
    req.num_max_contacts = limit
    req.enable_contact = True
    req.gjk_tolerance = _GJK_TOLERANCE
    res = _coal.CollisionResult()
    _coal.collide(obj_a, obj_b, req, res)
    out = []
    for index in range(res.numContacts()):
        normal = np.asarray(res.getContact(index).normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= 1e-12:
            continue
        unit = normal / length
        if not any(float(unit @ kept) > 0.999 for kept in out):
            out.append(unit)
    return out


def _ceiling_along(wa, wb, direction):
    """Shift of B along `direction` that provably separates the two AABBs.

    `_overlap_bound` generalized off the world axes. Separation needs only one
    axis to come apart, so the ceiling is the cheapest component to clear.
    Bounding the boxes bounds the meshes inside them, which is what keeps each
    bisection one-directional — it can only come in under its own ceiling.
    """
    alo, ahi = wa.min(axis=0), wa.max(axis=0)
    blo, bhi = wb.min(axis=0), wb.max(axis=0)
    reach = np.inf
    for i in range(3):
        if direction[i] > 1e-12:
            reach = min(reach, (ahi[i] - blo[i]) / direction[i])
        elif direction[i] < -1e-12:
            reach = min(reach, (bhi[i] - alo[i]) / -direction[i])
    return reach


def _record(wa, wb, *, witness=None, **extra) -> dict:
    """The pair's report. `witness` is what to measure the depth on, if anything.

    `measured` is carried beside `depth_is_bound` rather than replacing it: the
    number is still an upper bound (bisection on a non-convex pair finds *a*
    separating translation, not provably the least), but a reader wants to know
    whether it came from geometry or from a box, and those are different claims.
    """
    depth, axis = _overlap_bound(wa, wb)
    measured = False
    if witness is not None:
        bodies, se3_a, se3_b, tol = witness
        if bodies is not None:
            shift = _separating_shift(*bodies, se3_a, se3_b, wa, wb, axis, depth,
                                      tol)
            # `<= depth` is the invariant, asserted here rather than trusted:
            # this number replaces one that was already a sound upper bound, so
            # the only acceptable outcome is a tighter one.
            if shift is not None and shift <= depth:
                depth, measured = shift, True
    return {"method": "coal_bvh", "depth_m": round(float(depth), 6),
            "depth_is_bound": True, "measured": measured,
            "axis": "xyz"[axis], **extra}


def _separating_shift(body_a, body_b, margin, se3_a, se3_b, wa, wb, axis,
                      bound, tol):
    """How far along `axis` body B must travel before it stops touching A.

    Bisection on `_crosses`, which is the whole point: the depth then comes from
    the same triangle descent as the verdict and as `surface_gap`, instead of
    from a box that agrees with neither.

    **The AABB overlap is the bracket, and the search may never exceed it.** Two
    intervals `[alo, ahi]` and `[blo, bhi]` that partially overlap come apart
    once B moves by `ahi - blo`, which is the overlap itself. The answer is
    clamped to it either way, so it never exceeds the bound.

    **Returning None is a real outcome.** A world axis is often not a direction
    in which the bodies come apart at all: a tumbling block wants to be moved
    along a rotated face normal, which is why the contact normals are among the
    candidates. Each candidate is capped by its own ceiling, because bisecting
    past it returns roughly the bodies' size rather than a depth, and a search
    that has not converged inside one contributes nothing.

    The bracket is opened a couple of quanta past `bound` only so a pair that
    genuinely separates *at* the bound is not rejected for landing on
    exactly-touching boxes, which is coal's coplanar case again.

    **Both directions along every candidate, and the nearest wins.** Pushing B
    out the far side of A is a different translation from pulling it back the
    way it came, and on a real pair they differ a lot: the committed
    `bike_transmission`'s crank set and chainring marker separate at 1.002 mm one
    way and 6.000 mm the other. Searching only `+axis` made the answer depend on
    which body the caller happened to pass first, which is not a property a
    reported depth may have.
    """
    if bound <= 0.0:
        return None
    rb, tb = _rotation(se3_b)
    placed_a = _placed(body_a, se3_a)           # hoisted: A does not move

    def crosses(direction, distance):
        # The `(R, t)` spelling, so the shift needs no quaternion round trip.
        return _crosses_objs(placed_a,
                             _placed(body_b, (rb, tb + direction * distance)), margin)

    def bisect(direction, ceiling):
        """Least shift along `direction` that clears, or None inside `ceiling`."""
        far = ceiling + 2.0 * tol
        if crosses(direction, far):
            return None
        near = 0.0
        for _ in range(_SHIFT_ITERS):
            if far - near <= tol:
                break
            mid = 0.5 * (near + far)
            if crosses(direction, mid):
                near = mid
            else:
                far = mid
        return min(far, ceiling)

    unit = np.zeros(3)
    unit[axis] = 1.0
    screened = []
    for normal in _contact_normals(placed_a, _placed(body_b, (rb, tb)), margin):
        for signed in (normal, -normal):
            ceiling = _ceiling_along(wa, wb, signed)
            if np.isfinite(ceiling) and ceiling > 0.0:
                screened.append((ceiling, signed))
    screened.sort(key=lambda candidate: candidate[0])
    best = None
    for ceiling, direction in ([(bound, unit), (bound, -unit)]
                               + screened[:_NORMAL_CANDIDATES]):
        cleared = bisect(direction, ceiling)
        if cleared is not None and (best is None or cleared < best):
            best = cleared
    return best


def _overlap_bound(wa, wb):
    """`(depth, axis)` — the AABB overlap along its thinnest axis.

    An upper bound on how far the bodies interpenetrate, and the same *kind* of
    number as the object layer's SAT `depth`, whose prompt already tells the
    agent it overstates the correction. It stands in for
    `Contact.penetration_depth`, which for coincident or perpendicular faces
    returns the lateral extent of the intersection instead.
    """
    lo = np.maximum(wa.min(axis=0), wb.min(axis=0))
    hi = np.minimum(wa.max(axis=0), wb.max(axis=0))
    d = np.clip(hi - lo, 0.0, None)
    axis = int(np.argmin(d))
    return float(d[axis]), axis


def _containment(wa, wb, tris_a, tris_b) -> dict | None:
    """One body swallowed whole by the other, which no crossing reveals.

    The only overlap `collide` cannot see. Any partial overlap crosses a
    triangle, and the BVH descends to it; a body entirely inside another crosses
    nothing, so it has to be decided by probing points against a surface.

    **Gated on one AABB lying within the other, which is what keeps tangency
    out.** Containment implies box containment, so the gate has no false
    negatives; without it, two bodies resting exactly flush send the touching
    face's vertices into the winding-number test, where a point *on* the surface
    is ambiguous and reads as inside. That is the whole tangency false positive:
    it entered here, not through `collide`, whose `security_margin` had already
    cleared the pair.

    The probes are interior points, not surface ones: a spread of vertices, each
    pulled a little toward its own body's centroid. Vertices alone decide two
    exactly coincident bodies — a duplicated part at a duplicated pose, total
    overlap — by asking whether a point on B's surface is inside B.

    A spread rather than a single probe, because one point is only sound for a
    single-component body: a two-piece mesh with one piece inside and one
    outside is contained in neither direction, and probing only its first vertex
    answers for whichever piece that vertex happens to belong to.
    """
    from ..sdk import _deform_eval as _de

    for probe, verts, tris in ((wa, wb, tris_b), (wb, wa, tris_a)):
        if not _box_within(probe, verts):
            continue
        for p in _interior_probes(probe):
            if _de.point_inside_mesh(verts, tris, p):
                return _record(wa, wb, containment=True)
    return None


def _interior_probes(verts):
    """Probe points for the containment test, nudged off the surface.

    Each sampled vertex is drawn `_PROBE_INSET` of the way toward the body's own
    centroid. On a convex body that lands strictly inside; on a concave one it
    may not, which costs nothing — the gate above has already established that
    this body's box is inside the other's, so a probe that reads "outside" only
    forfeits its own vote and the remaining probes still answer.
    """
    pts = _spread(verts, _CONTAINMENT_PROBES)
    return pts + (verts.mean(axis=0) - pts) * _PROBE_INSET


def _spread(verts, n):
    """`n` vertices spread over the array, or all of them when there are fewer."""
    if len(verts) <= n:
        return verts
    return verts[np.linspace(0, len(verts) - 1, n).astype(int)]


def _box_within(inner, outer) -> bool:
    """Is `inner`'s world AABB inside `outer`'s? Non-strict, so equal boxes pass."""
    return bool(np.all(inner.min(axis=0) >= outer.min(axis=0))
                and np.all(inner.max(axis=0) <= outer.max(axis=0)))


def surface_gap(body_a, se3_a, body_b, se3_b) -> dict:
    """Closest approach between two placed bodies: `{gap_m, method, exact}`.

    Exact to the triangulation in both directions — coal descends the BVH to
    triangle-pair distances, so the closest points may lie in triangle interiors.
    `resolution_m` is carried as 0.0 for consumers that read it as the width of
    an interval the gap sits at the top of; triangle-triangle closes that
    interval. `gap_m` is 0.0 for an overlapping pair — `penetration` is the other
    half of contact.
    """
    if _coal is None:
        raise RuntimeError("coal is not installed")
    req = _coal.DistanceRequest()
    req.enable_signed_distance = False
    res = _coal.DistanceResult()
    d = float(_coal.distance(_placed(body_a, se3_a), _placed(body_b, se3_b),
                             req, res))
    return {"gap_m": round(max(0.0, d), 6), "resolution_m": 0.0,
            "method": "coal_bvh", "exact": True}
