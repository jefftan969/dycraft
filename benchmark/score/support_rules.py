"""Floating policy: anchors, support edges, per-frame support, free flight.

Numpy and the standard library only, so any threshold or policy change re-applies to a cached geometry pass
(`floating_raw.npz`) without touching a mesh.

1. **classes** (`classes.py`): each body is `solid`, `carrier` (a deforming solid: scored, but supported only by
   contact in any direction, since the slope and grip tests need a rigid shape), `agent` or `exempt` (a support
   source), `fluid` or `text` (structural links only).
2. **anchors**: never-moving solids with a footprint >= 4 m^2 in the lowest band of the scene (ground), plus flat
   panels (two sides >= 1.5 m, one <= 0.5 m) contact-connected to the ground, plus agents and exempt bodies.
3. **edges** per pair over time, latched with hysteresis on the measured gaps: passive `A -> B` (A rests on B),
   pivot `A -> B` (A turns about an axis through B), and the symmetric mounted, structural, carrier, and bonded edges.
4. **support** per frame: reachability from the active sources, along supporter -> supported.
5. **free flight**: every unsupported component whose membership stays the same is fitted with a gravity parabola;
   ballistic segments are legal.

    illegal(b, t)  = active, scored, dynamic body b has no support path at t and is not in a ballistic segment;
                     the sources are the anchors AND every never-moving scored body (local static support)
    Floating rate  = sum_{b,t} illegal(b, t) / sum_{b,t} [b dynamic, active, and scored at t]

A body is dynamic if the animation moves it at all (MOVING at some frame and displaced >= 1 mm); it then counts on
every frame it is unsupported, at rest or not, so an object left hovering after release still floats. Never-moving
bodies are scene fixtures: a static structure with no path to the ground (a machine base hovering above the floor)
is charged once, in `static_grounding_failure_rate`, not to every moving part mounted on it. `strict_floating_rate`
(the anchors only) and the frame-level `floating_frame_rate` are reported too.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

import numpy as np

RULES_VERSION = "physical-floating-rules-v2-pivot"

from .classes import CARRIER, CONTACT_CLASSES, SELF_SUPPORTING_CLASSES, SOLID, tokens

#: Edge kinds, as stored per frame.
PASSIVE, MOUNTED, STRUCTURAL, CARRIED, BONDED, PIVOT = 0, 1, 2, 3, 4, 5
KIND_NAMES = {PASSIVE: "passive", MOUNTED: "mounted", STRUCTURAL: "structural",
              CARRIED: "carrier", BONDED: "bonded", PIVOT: "pivot"}
#: flags in the raw pair table
EMB_AB, EMB_BA = 1, 2

DEFAULT_PARAMS = {
    #: resting / mounted / carrier contact (m). Sub-visible placement clearances of a few millimeters (a sole above
    #: a floor, furniture on a rug) are common in every method and do not read as floating on screen; larger gaps
    #: and detached parts still fail.
    "tau_p": 0.005,
    #: structural (joint / fixed / same link / parenting) contact (m)
    "tau_s": 0.005,
    #: bonded (grasp / carry) contact (m): rigid fingers cannot conform to the object they hold, so hand-object gaps
    #: during a real carry run to about a centimeter.
    "tau_b": 0.010,
    #: contact is kept while the gap stays below exit_factor * tau
    "exit_factor": 1.5,
    #: size-adaptive floors: tau_eff = min(tau, max(floor, frac * smallest extent))
    "tau_floor": 0.0005, "tau_frac": 0.5,
    "tau_b_floor": 0.002, "tau_b_frac": 1.0,
    #: bonded: max relative corner drift per transition (m), and the distance the
    #: pair must travel together inside the contact interval (m)
    "bond_drift_m": 0.002, "bond_travel_m": 0.010,
    #: pivot: an undeclared hinge, arbor, or pin, the same for every method (a bare scene declares no joints, a DyCraft
    #: scene does not declare all of its own). Over a non-structural contact (within tau_s, latched like a structural
    #: pair) the pair turns >= pivot_min_deg about one fixed axis (axes agree within pivot_axis_deg, fixed-line
    #: residual <= pivot_line_m); the body the axis runs through (its local box, inflated by the contact tolerance)
    #: is the axle and holds the other: a paddle on the axle it touches, clock hands on their boss, a door on its
    #: frame (both ways for a pin in a hole). A disc spinning about its own axis holds nothing it touches; rolling or
    #: sliding is no pivot. 0.2 deg: a clock's minute hand turns 0.3 deg in 17 s.
    "pivot_support": True,
    "pivot_min_deg": 0.2, "pivot_axis_deg": 5.0, "pivot_line_m": 0.003,
    #: anchors
    "ground_area_m2": 4.0, "ground_fallback_area_m2": 1.0, "ground_band_m": 0.25,
    "panel_long_m": 1.5, "panel_thin_m": 0.5,
    #: ballistic free flight: fitted downward acceleration within [lo, hi] * g,
    #: horizontal <= horiz * g, rms residual <= max(rms_floor, rms_frac * path)
    "g": 9.81, "g_lo": 0.5, "g_hi": 1.5, "g_horiz": 0.3,
    "rms_floor_m": 0.005, "rms_frac": 0.02,
    #: a body is dynamic (counted by the headline) if the animation displaces it
    #: by at least this much; never-moving bodies are static fixtures (reported)
    "animated_m": 0.001,
}

# ----------------------------------------------------------------- helpers

def ranges(mask_or_frames):
    frames = np.flatnonzero(mask_or_frames) if isinstance(mask_or_frames, np.ndarray) and \
        mask_or_frames.dtype == bool else sorted(mask_or_frames)
    out = []
    for f in frames:
        f = int(f)
        if out and out[-1][1] == f - 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


#: Gaps are stored at 0.01 mm resolution in float32 (`_support_child.ROUND_M`); a gap designed at exactly the
#: tolerance must not fail on storage rounding.
GAP_EPS = 1.5e-5


def latch(value, active, enter, exit_):
    """Hysteresis: in contact from a frame with value <= enter, kept while value
    <= exit_, reset by any frame that is not active (either body inactive)."""
    within = active & (value <= exit_ + GAP_EPS)
    enter_mask = within & (value <= enter + GAP_EPS)
    if not enter_mask.any():
        return np.zeros_like(within)
    n = len(value)
    # segment id of each frame: new segment where `within` starts
    starts = within & ~np.concatenate([[False], within[:-1]])
    seg = np.cumsum(starts)
    seg[~within] = 0
    cum = np.cumsum(enter_mask)
    # cumulative enters since the start of the segment
    seg_start_idx = np.zeros(n, dtype=np.int64)
    idx = np.flatnonzero(starts)
    seg_start_idx[idx] = idx
    seg_start_idx = np.maximum.accumulate(seg_start_idx)
    before = np.where(seg_start_idx > 0, cum[np.maximum(seg_start_idx - 1, 0)], 0)
    before[seg_start_idx == 0] = 0
    return within & ((cum - before) > 0)


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


# ------------------------------------------------------------------ context

class Scene:
    """Everything the rules need, from the raw geometry pass and the motion states."""

    def __init__(self, raw: dict, moving: np.ndarray, structure: list, params: dict):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.names = list(raw["names"])
        self.identities = list(raw["identities"])
        self.N = len(self.names)
        self.n = int(raw["n_frames"])
        self.fps = float(raw["fps"])
        self.M = raw["M"]                       # (N, n, 3, 4) float32
        self.hid = raw["hid"]                   # (N, n) mesh index, -1 absent
        self.active0 = raw["active"]            # (N, n) present/visible/non-degenerate
        self.mesh_lo, self.mesh_hi = raw["mesh_lo"], raw["mesh_hi"]
        self.mesh_centroid = raw["mesh_centroid"]
        self.mesh_volume = raw["mesh_volume"]
        self.classes = list(raw["classes"])
        self.pairs = raw["pairs"]               # (P, 2)
        self.runs = raw["runs"]                 # (R, 3) pair, start, end
        self.vals = raw["vals"]                 # (R, 3) gap, gapU_ab, gapU_ba
        self.flags = raw["flags"]               # (R,)
        self.moving = moving                    # (N, n) bool
        self.structure = structure              # [(a_idx, b_idx, source)]
        cls = np.array(self.classes)
        self.contact = np.isin(cls, CONTACT_CLASSES)
        self.agent = np.isin(cls, SELF_SUPPORTING_CLASSES)   # agents and buoyant bodies
        self.solid = cls == SOLID
        # Scored: solids and deforming solids (carriers). A carrier (a skinned coat, a hand, a cord, cloth) gets
        # support only from contact in any direction.
        self.scored = self.solid | (cls == CARRIER)
        self.carrier = cls == CARRIER
        self.node = self.active0.copy()     # every class is a graph node
        # smallest world extent (local box x matrix scale) at the first active frame
        self.min_extent = np.zeros(self.N)
        for b in range(self.N):
            idx = np.flatnonzero(self.active0[b])
            if len(idx):
                self.min_extent[b] = float(self.dims(b, int(idx[0])).min())
        order = np.argsort(self.runs[:, 0], kind="stable") if len(self.runs) else np.zeros(0, dtype=np.int64)
        self.runs, self.vals, self.flags = self.runs[order], self.vals[order], self.flags[order]
        P = len(self.pairs)
        self.row_start = np.searchsorted(self.runs[:, 0], np.arange(P), side="left") if P else np.zeros(0, int)
        self.row_end = np.searchsorted(self.runs[:, 0], np.arange(P), side="right") if P else np.zeros(0, int)
        self.pair_index = {(int(a), int(b)): k for k, (a, b) in enumerate(self.pairs)}
        self._centroids = {}
        self.anchor = np.zeros(self.N, dtype=bool)
        self.anchor_info = {}

    # geometry --------------------------------------------------------------
    def dense(self, k):
        """gap, gapU_ab, gapU_ba (inf where not measured / farther than the query), emb flags."""
        n = self.n
        gap = np.full(n, np.inf); gab = np.full(n, np.inf); gba = np.full(n, np.inf)
        emb = np.zeros(n, dtype=np.uint8)
        for r in range(self.row_start[k], self.row_end[k]):
            _, s, e = self.runs[r]
            gap[s:e + 1], gab[s:e + 1], gba[s:e + 1] = self.vals[r]
            emb[s:e + 1] = self.flags[r]
        return gap, gab, gba, emb

    def centroid(self, b):
        """World centroid per frame (n, 3); NaN where the body is absent."""
        if b not in self._centroids:
            h = self.hid[b]
            c = self.mesh_centroid[np.maximum(h, 0)]
            M = self.M[b].astype(np.float64)
            x = np.einsum("tij,tj->ti", M[:, :, :3], c) + M[:, :, 3]
            x[h < 0] = np.nan
            self._centroids[b] = x
        return self._centroids[b]

    def weight(self, b):
        h = self.hid[b][self.active0[b]]
        if not len(h):
            return 0.0
        h = int(h[0])
        v = float(self.mesh_volume[h])
        if v > 0:
            return v
        e = np.maximum(self.mesh_hi[h] - self.mesh_lo[h], 1e-4)
        return float(np.prod(e))

    def world_box(self, b, t):
        h = int(self.hid[b, t])
        lo, hi = self.mesh_lo[h], self.mesh_hi[h]
        M = self.M[b, t].astype(np.float64)
        c = M[:, :3] @ ((lo + hi) / 2) + M[:, 3]
        half = np.abs(M[:, :3]) @ ((hi - lo) / 2)
        return c - half, c + half

    def dims(self, b, t):
        h = int(self.hid[b, t])
        M = self.M[b, t].astype(np.float64)
        return (self.mesh_hi[h] - self.mesh_lo[h]) * np.linalg.norm(M[:, :3], axis=0)

    def tau(self, a, b, tau, floor, frac):
        e = min(self.min_extent[a], self.min_extent[b])
        return min(tau, max(floor, frac * e))


# ------------------------------------------------------------------- anchors

def find_anchors(sc: Scene):
    p = sc.p
    info = {"ground": [], "panels": [], "rejected_panels": [], "no_ground": False, "fallback_ground": False}
    never_moving = np.array([sc.solid[b] and sc.active0[b].any() and not (sc.moving[b] & sc.active0[b]).any()
                             for b in range(sc.N)], dtype=bool)
    footprint, zmin = {}, {}
    large, panels = [], []
    for b in np.flatnonzero(never_moving):
        t = int(np.flatnonzero(sc.active0[b])[0])
        lo, hi = sc.world_box(b, t)
        footprint[b] = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        zmin[b] = float(lo[2])
        d = np.sort(sc.dims(b, t))
        if footprint[b] >= p["ground_area_m2"]:
            large.append(b)
        elif d[1] >= p["panel_long_m"] and d[0] <= p["panel_thin_m"]:
            panels.append(b)
    # Ground = large static bodies at the bottom of the scene. A large body
    # higher up (an elevated course bed, a wall coping, a mezzanine) is not
    # ground: like everything else it needs a support path (contact) to it.
    ground = []
    if large:
        floor_z = min(zmin[b] for b in large)
        ground = [b for b in large if zmin[b] <= floor_z + p["ground_band_m"]]
        info["elevated_large_bodies"] = [sc.names[b] for b in large if b not in ground][:40]
    if not ground and footprint:
        best = max(footprint, key=lambda b: (footprint[b], -b))
        if footprint[best] >= p["ground_fallback_area_m2"]:
            ground = [best]
            info["fallback_ground"] = True
    if not ground:
        info["no_ground"] = True
    # self-propelled and buoyant bodies hold themselves up, like a robot on its own feet: they are support sources
    # for what they carry or touch (added after the panel flood fill, so an agent touching a wall does not make the
    # wall an anchor)
    agents = [int(b) for b in np.flatnonzero(sc.agent)]
    anchors = set(ground)
    panel_set = set(panels)
    # flood fill ground -> panels over contact (any direction, tau_s)
    near = defaultdict(set)
    for k, (a, b) in enumerate(sc.pairs):
        a, b = int(a), int(b)
        if (a in panel_set or a in anchors) and (b in panel_set or b in anchors):
            gap = sc.dense(k)[0]
            if np.nanmin(np.where(sc.active0[a] & sc.active0[b], gap, np.inf)) <= p["tau_s"]:
                near[a].add(b); near[b].add(a)
    queue = deque(sorted(anchors))
    while queue:
        u = queue.popleft()
        for v in sorted(near[u]):
            if v in panel_set and v not in anchors:
                anchors.add(v); queue.append(v)
    anchors |= set(agents)
    sc.anchor = np.zeros(sc.N, dtype=bool)
    sc.anchor[list(anchors)] = True
    info["ground"] = [{"name": sc.names[b], "footprint_m2": round(footprint[b], 3)} for b in ground]
    info["panels"] = [sc.names[b] for b in panels if b in anchors]
    info["rejected_panels"] = [sc.names[b] for b in panels if b not in anchors]
    info["agents"] = [sc.names[b] for b in agents][:40]
    info["count"] = int(sc.anchor.sum())
    sc.anchor_info = info
    return info


# --------------------------------------------------------------------- edges

def pair_edges(sc: Scene, tau_p=None, tau_b=None):
    """Per-frame edges of every measured / structural pair.

    Returns (frames, a, b, kind) arrays; passive rows mean `a` rests on `b`, pivot rows
    that `a` turns on `b` (b is the axle).
    Also per-body-frame nearest gap, nearest partner, and two cause flags.
    """
    p = sc.p
    tau_p = p["tau_p"] if tau_p is None else tau_p
    tau_b = p["tau_b"] if tau_b is None else tau_b
    xf = p["exit_factor"]
    n = sc.n
    F, A, B, K = [], [], [], []
    nearest = np.full((sc.N, n), np.inf)
    partner = np.full((sc.N, n), -1, dtype=np.int64)
    near_moving = np.zeros((sc.N, n), dtype=bool)
    lateral = np.zeros((sc.N, n), dtype=bool)
    static = ~sc.moving
    struct = {}
    for a, b, source in sc.structure:
        struct.setdefault((min(a, b), max(a, b)), source)

    def emit(mask, a, b, kind):
        f = np.flatnonzero(mask)
        if len(f):
            F.append(f); A.append(np.full(len(f), a)); B.append(np.full(len(f), b)); K.append(np.full(len(f), kind))

    for k, (a, b) in enumerate(sc.pairs):
        a, b = int(a), int(b)
        both = sc.node[a] & sc.node[b]
        if not both.any():
            continue
        gap, gab, gba, emb = sc.dense(k)
        tp = sc.tau(a, b, tau_p, p["tau_floor"], p["tau_frac"])
        ts = sc.tau(a, b, p["tau_s"], p["tau_floor"], p["tau_frac"])
        tb = sc.tau(a, b, tau_b, p["tau_b_floor"], p["tau_b_frac"])
        g = np.where(both, gap, np.inf)
        st = struct.get((min(a, b), max(a, b)))
        if not (sc.contact[a] and sc.contact[b]):
            # fluid / text: delivered structural links only, never contact support
            if st is not None:
                emit(latch(g, both, ts, ts * xf), a, b, STRUCTURAL)
            continue
        # cause bookkeeping (per body-frame)
        for x, y in ((a, b), (b, a)):
            better = g < nearest[x]
            nearest[x] = np.where(better, g, nearest[x])
            partner[x] = np.where(better, y, partner[x])
            near_moving[x] |= (g <= tb * xf) & sc.moving[y]
            lateral[x] |= g <= tp
        if sc.carrier[a] or sc.carrier[b]:
            emit(latch(g, both, tp, tp * xf), a, b, CARRIED)
            continue
        # passive, each direction
        v_ab = np.where(emb & EMB_AB, 0.0, gab)
        v_ba = np.where(emb & EMB_BA, 0.0, gba)
        emit(latch(v_ab, both, tp, tp * xf), a, b, PASSIVE)
        emit(latch(v_ba, both, tp, tp * xf), b, a, PASSIVE)
        contact_p = latch(g, both, tp, tp * xf)
        emit(contact_p & static[a] & static[b], a, b, MOUNTED)
        if st is not None:
            emit(latch(g, both, ts, ts * xf), a, b, STRUCTURAL)
        elif p.get("pivot_support"):
            on_b, on_a = pivot_mask(sc, a, b, latch(g, both, ts, ts * xf), ts * xf)
            emit(on_b, a, b, PIVOT)             # a turns on b (the axis runs through b): b holds a
            emit(on_a, b, a, PIVOT)
        bond = bonded_mask(sc, a, b, latch(g, both, tb, tb * xf))
        emit(bond, a, b, BONDED)
    if F:
        F, A, B, K = (np.concatenate(x) for x in (F, A, B, K))
        order = np.lexsort((K, B, A, F))
        F, A, B, K = F[order], A[order], B[order], K[order]
    else:
        F = A = B = K = np.zeros(0, dtype=np.int64)
    return (F, A, B, K), nearest, partner, near_moving, lateral


def bonded_mask(sc: Scene, a, b, contact):
    """Contact intervals split where the grip slips; kept where the pair moved together."""
    out = np.zeros(sc.n, dtype=bool)
    if not contact.any() or not (sc.moving[a] & contact).any() and not (sc.moving[b] & contact).any():
        return out
    p = sc.p
    ca, cb = sc.centroid(a), sc.centroid(b)
    for s, e in ranges(contact):
        idx = np.arange(s, e + 1)
        Ma = sc.M[a, idx].astype(np.float64)
        Mb = sc.M[b, idx].astype(np.float64)
        Ra, ta = Ma[:, :, :3], Ma[:, :, 3]
        Rb, tb_ = Mb[:, :, :3], Mb[:, :, 3]
        h = sc.hid[a, idx]
        corners = np.stack([_corners(sc.mesh_lo[x], sc.mesh_hi[x]) for x in h])      # (m, 8, 3)
        world = np.einsum("tij,tkj->tki", Ra, corners) + ta[:, None, :]
        try:
            Rb_inv = np.linalg.inv(Rb)
        except np.linalg.LinAlgError:          # a body flattened to zero on one axis
            Rb_inv = np.linalg.pinv(Rb)
        rel = np.einsum("tij,tkj->tki", Rb_inv, world - tb_[:, None, :])            # A corners in B frame
        # drift across transition t-1 -> t, using frame t's corners placed at t-1
        prev_world = np.einsum("tij,tkj->tki", Ra[:-1], corners[1:]) + ta[:-1, None, :]
        prev_rel = np.einsum("tij,tkj->tki", Rb_inv[:-1], prev_world - tb_[:-1, None, :])
        drift = np.linalg.norm(rel[1:] - prev_rel, axis=2).max(axis=1) if len(idx) > 1 else np.zeros(0)
        cut = np.flatnonzero(drift > p["bond_drift_m"]) + 1       # new sub-interval starts here
        bounds = [0, *cut.tolist(), len(idx)]
        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            J = idx[i0:i1]
            if len(J) < 2:
                continue
            travel = max(np.nanmax(np.linalg.norm(ca[J] - ca[J[0]], axis=1)),
                         np.nanmax(np.linalg.norm(cb[J] - cb[J[0]], axis=1)))
            if travel >= p["bond_travel_m"]:
                out[J] = True
    return out


def _homogeneous(M):
    """(m, 3, 4) placements as (m, 4, 4) float64."""
    out = np.zeros((len(M), 4, 4))
    out[:, :3] = M
    out[:, 3, 3] = 1.0
    return out


def _line_hits_box(point, direction, lo, hi):
    """Does the infinite line point + s * direction pass through the box [lo, hi]?"""
    t0, t1 = -np.inf, np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if not lo[i] <= point[i] <= hi[i]:
                return False
            continue
        u, v = (lo[i] - point[i]) / direction[i], (hi[i] - point[i]) / direction[i]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
    return t0 <= t1


def pivot_mask(sc: Scene, a, b, contact, margin):
    """(a turns on b, b turns on a): contact frames on which the pair turns about one fixed
    axis that runs through b's box, respectively a's box.

    Per contact interval: G_t = W_b(t)^-1 M_a(t) M_a(s)^-1 W_b(s) carries a point of a,
    in b's rigid frame at the interval start s, to its place in b's frame at t (W_b:
    b's placement with the rotation orthonormalised, so scale cancels). It must be a
    rotation about a fixed line, turning at least
    pivot_min_deg; the body whose local box (inflated by `margin`, m) the line crosses
    is the axle and holds the other. Remeshed bodies (a changing mesh index) are never pivots.
    """
    p = sc.p
    on_b, on_a = np.zeros(sc.n, dtype=bool), np.zeros(sc.n, dtype=bool)
    if not contact.any():
        return on_b, on_a
    for s, e in ranges(contact):
        idx = np.arange(s, e + 1)
        if len(idx) < 3 or not (sc.moving[a, idx].any() or sc.moving[b, idx].any()):
            continue
        ha, hb = sc.hid[a, idx], sc.hid[b, idx]
        if ha[0] < 0 or hb[0] < 0 or (ha != ha[0]).any() or (hb != hb[0]).any():
            continue
        Ma, Mb = _homogeneous(sc.M[a, idx]), _homogeneous(sc.M[b, idx])
        u, _, vt = np.linalg.svd(Mb[:, :3, :3])
        Wb = Mb.copy()
        Wb[:, :3, :3] = u @ vt
        try:
            G = np.linalg.inv(Wb) @ Ma @ np.linalg.inv(Ma[0]) @ Wb[0]
        except np.linalg.LinAlgError:
            continue
        u, _, vt = np.linalg.svd(G[:, :3, :3])
        Rm, pv = u @ vt, G[:, :3, 3]
        theta = np.degrees(np.arccos(np.clip((np.trace(Rm, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)))
        if theta.max() < p["pivot_min_deg"]:
            continue
        turn = theta >= 0.5 * p["pivot_min_deg"]
        axes = np.linalg.svd(Rm[turn] - np.eye(3))[2][:, -1, :]      # null vector of R - I: the axis
        axes = axes * np.sign(axes @ axes[0])[:, None]
        axis = axes.mean(axis=0)
        axis /= np.linalg.norm(axis)
        if float(np.degrees(np.arccos(np.clip(axes @ axis, -1, 1))).max()) > p["pivot_axis_deg"]:
            continue
        # a rotation about the line through q: p_t = (I - R_t) q, for every frame
        A = (np.eye(3)[None] - Rm).reshape(-1, 3)
        q, *_ = np.linalg.lstsq(A, pv.reshape(-1), rcond=None)
        if float(np.linalg.norm((A @ q - pv.reshape(-1)).reshape(-1, 3), axis=1).max()) > p["pivot_line_m"]:
            continue
        point = Wb[0, :3, :3] @ q + Wb[0, :3, 3]                   # the axis in world space at s
        direction = Wb[0, :3, :3] @ axis
        for M0, h, held in ((Mb[0], int(hb[0]), on_b), (Ma[0], int(ha[0]), on_a)):
            try:
                inv = np.linalg.inv(M0)
            except np.linalg.LinAlgError:
                continue
            pad = margin / np.maximum(np.linalg.norm(M0[:3, :3], axis=0), 1e-12)   # world margin, local units
            if _line_hits_box(inv[:3, :3] @ point + inv[:3, 3], inv[:3, :3] @ direction,
                              sc.mesh_lo[h] - pad, sc.mesh_hi[h] + pad):
                held[idx] = True
    return on_b, on_a


# --------------------------------------------------------------------- support

def _reach(N, anchors, src, dst):
    """Bodies reachable from `anchors` along src -> dst (C speed when scipy is present)."""
    if not len(anchors):
        return np.zeros(N, dtype=bool)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order
    except ImportError:  # pure-Python fallback
        nbrs = defaultdict(list)
        for a, b in zip(src.tolist(), dst.tolist()):
            nbrs[a].append(b)
        seen = np.zeros(N, dtype=bool)
        seen[anchors] = True
        queue = deque(int(x) for x in anchors)
        while queue:
            u = queue.popleft()
            for v in nbrs.get(u, ()):
                if not seen[v]:
                    seen[v] = True; queue.append(v)
        return seen
    root = N                                   # virtual root -> every active anchor
    s = np.concatenate([np.full(len(anchors), root), src])
    d = np.concatenate([anchors, dst])
    graph = csr_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(N + 1, N + 1))
    order = breadth_first_order(graph, root, directed=True, return_predecessors=False)
    seen = np.zeros(N + 1, dtype=bool)
    seen[order] = True
    return seen[:N]


def _components(nodes, a, b):
    """Connected components of `nodes` over undirected edges (a, b) inside `nodes`."""
    if not len(nodes):
        return []
    index = {int(x): i for i, x in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a.tolist(), b.tolist()):
        if x in index and y in index:
            rx, ry = find(index[x]), find(index[y])
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    groups = defaultdict(list)
    for x, i in index.items():
        groups[find(i)].append(x)
    return sorted(tuple(sorted(g)) for g in groups.values())


def support(sc: Scene, edges):
    """Per frame: unsupported evaluated bodies, their components, mounted-only bodies.

    Support is reachability from the active anchors along *supporter -> supported*:
    a passive edge `a -> b` (a rests on b) and a pivot edge (a turns on b) are
    traversed b -> a only; every other kind both ways.
    """
    F, A, B, K = edges
    n, N = sc.n, sc.N
    bounds = np.searchsorted(F, np.arange(n + 1))
    evaluated = sc.node & sc.scored[:, None] & ~sc.anchor[:, None]
    unsupported, components, mounted_only, has_edge = [], [], [], []
    last = None
    for t in range(n):
        s, e = bounds[t], bounds[t + 1]
        a, b, k = A[s:e], B[s:e], K[s:e]
        active_nodes = np.flatnonzero(sc.node[:, t])
        sig = (a.tobytes(), b.tobytes(), k.tobytes(), active_nodes.tobytes())
        if last is not None and sig == last[0]:
            unsupported.append(last[1]); components.append(last[2]); mounted_only.append(last[3])
            has_edge.append(last[4])
            continue
        anchors = np.flatnonzero(sc.anchor & sc.node[:, t])
        passive = (k == PASSIVE) | (k == PIVOT)         # directed: supporter b holds a
        sym = ~passive
        src = np.concatenate([b[passive], a[sym], b[sym]])
        dst = np.concatenate([a[passive], b[sym], a[sym]])
        sup = _reach(N, anchors, src, dst)
        ev = evaluated[:, t]
        uns = [int(x) for x in np.flatnonzero(ev & ~sup)]
        mounted = k == MOUNTED
        if mounted.any():
            keep = ~mounted
            p2, s2 = passive & keep, sym & keep
            strict = _reach(N, anchors, np.concatenate([b[p2], a[s2], b[s2]]),
                            np.concatenate([a[p2], b[s2], a[s2]]))
            mo = [int(x) for x in np.flatnonzero(ev & sup & ~strict)]
        else:
            mo = []
        comps = _components(uns, a, b)
        touched = np.zeros(N, dtype=bool)
        touched[a] = True; touched[b] = True
        he = {x for x in uns if touched[x]}
        last = (sig, uns, comps, mo, he)
        unsupported.append(uns); components.append(comps); mounted_only.append(mo); has_edge.append(he)
    return unsupported, components, mounted_only, has_edge


# ---------------------------------------------------------------- free flight

def ballistic(sc: Scene, members, start, end):
    """Fit the volume-weighted centroid over [start-1, end] (launch frame
    included; the landing frame is not: impact happens between frames)."""
    p = sc.p
    frames = [f for f in range(start - 1, end + 1)
              if 0 <= f < sc.n and all(sc.active0[b, f] for b in members)]
    w = np.array([sc.weight(b) for b in members])
    w = w / w.sum() if w.sum() > 0 else np.full(len(members), 1 / len(members))
    x = np.array([sum(wi * sc.centroid(b)[f] for wi, b in zip(w, members)) for f in frames]) if frames else np.zeros((0, 3))
    t = np.array(frames, dtype=np.float64) / sc.fps
    g = p["g"]
    res = {"samples": len(frames), "fitted_a": None, "rms_m": None, "ballistic": False, "determined": True}
    if len(frames) >= 4:
        t0 = t - t[0]
        X = np.stack([np.ones_like(t0), t0, 0.5 * t0 ** 2], axis=1)
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        a = coef[2]
        fit = X @ coef
        rms = float(np.sqrt(np.mean(np.sum((x - fit) ** 2, axis=1))))
        path = float(np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)))
        ok_rms = rms <= max(p["rms_floor_m"], p["rms_frac"] * path)
        res["rms_m"] = rms
    elif len(frames) == 3:
        a = 2 * ((x[2] - x[1]) / (t[2] - t[1]) - (x[1] - x[0]) / (t[1] - t[0])) / (t[2] - t[0])
        ok_rms = True
    else:
        res["determined"] = False
        return res
    res["fitted_a"] = [round(float(v), 4) for v in a]
    down = -float(a[2])
    res["ballistic"] = bool(p["g_lo"] * g <= down <= p["g_hi"] * g
                            and math.hypot(float(a[0]), float(a[1])) <= p["g_horiz"] * g and ok_rms)
    return res


def segments(components):
    """Maximal runs of frames in which exactly the same member set is an unsupported component."""
    open_, out = {}, []
    for t, comps in enumerate(components):
        now = set(comps)
        for c in list(open_):
            if c not in now:
                out.append((c, open_.pop(c), t - 1))
        for c in comps:
            if c not in open_:
                open_[c] = t
    for c, s in open_.items():
        out.append((c, s, len(components) - 1))
    return sorted(out, key=lambda x: (x[1], x[0]))


# ------------------------------------------------------------------- evaluate

def _legal_pass(sc: Scene, edges, anchors, detail):
    """Support from `anchors`, then the free-flight exception: per frame, the illegal bodies."""
    saved = sc.anchor
    sc.anchor = anchors
    try:
        unsupported, components, mounted_only, has_edge = support(sc, edges)
        illegal = [set(u) for u in unsupported]
        free_frames = set()
        seg_out, undetermined = [], 0
        for members, s, e in segments(components):
            fit = ballistic(sc, members, s, e)
            if not fit["determined"]:
                undetermined += 1
            if fit["ballistic"]:
                for t in range(s, e + 1):
                    illegal[t].difference_update(members)
                    free_frames.add(t)
            if detail:
                seg_out.append({"bodies": [sc.names[b] for b in members], "start": s, "end": e,
                                "duration_s": round((e - s + 1) / sc.fps, 4), **fit})
    finally:
        sc.anchor = saved
    return unsupported, components, mounted_only, has_edge, illegal, free_frames, seg_out, undetermined


def local_anchors(sc: Scene):
    """Local static support: every never-moving scored body is also a support source.
    A moving part is then charged only for its own support failure, not for the
    static structure it is mounted on failing to reach the floor; that failure is
    reported by `static_grounding_failure_rate`. Support still needs a real edge
    (resting from above, structure + contact, grip, carrier contact)."""
    return sc.anchor | (sc.scored & ~animated_bodies(sc) & sc.active0.any(axis=1))


def evaluate(sc: Scene, *, tau_p=None, tau_b=None, detail=True):
    """Headline: local static support. Every never-moving scored body is a support
    source, so a moving part is charged only for its own support failure; a static
    structure without a path to the true ground (a machine base hovering above the
    floor) is charged once, in `static_grounding_failure_rate`, not to every moving
    part mounted on it. The strict ground-path rate is reported as `strict_floating_rate`."""
    edges, nearest, partner, near_moving, lateral = pair_edges(sc, tau_p, tau_b)
    n = sc.n
    animated = animated_bodies(sc)
    # strict: support only from the true world anchors (static grounding, strict rate)
    s_uns, s_comps, s_mounted_only, s_has_edge, s_illegal, _, _, _ = _legal_pass(sc, edges, sc.anchor, False)
    # headline: local static support
    (unsupported, components, mounted_only, has_edge, illegal, free_frames, seg_out,
     undetermined) = _legal_pass(sc, edges, local_anchors(sc), detail)
    dynamic = [{b for b in x if animated[b]} for x in illegal]
    s_dynamic = [{b for b in x if animated[b]} for x in s_illegal]
    static_ = [{b for b in x if not animated[b]} for x in s_illegal]    # never-moving, no path to the ground
    floating = np.array([bool(x) for x in dynamic], dtype=bool)
    any_floating = np.array([bool(d or st) for d, st in zip(dynamic, static_)], dtype=bool)
    evaluated = sc.node & sc.scored[:, None] & ~sc.anchor[:, None]
    ev_dyn = int((evaluated & animated[:, None]).sum())
    dyn_body_frames = sum(len(x) for x in dynamic)
    strict_body_frames = sum(len(x) for x in s_dynamic)
    static_active = evaluated & ~animated[:, None]
    static_uns = sum(len(x) for x in static_)
    summary = {
        # headline: per animated body and frame, local static support
        "floating_rate": dyn_body_frames / ev_dyn if ev_dyn else None,
        "floating_body_frames": dyn_body_frames,
        "animated_body_frames": ev_dyn,
        # never-moving scored bodies without a path to the true ground (co-reported)
        "static_grounding_failure_rate": static_uns / int(static_active.sum()) if static_active.sum() else None,
        "static_ungrounded_body_frames": static_uns,
        "static_body_frames": int(static_active.sum()),
        # strict: moving parts also charged when their static structure does not reach the ground
        "strict_floating_rate": strict_body_frames / ev_dyn if ev_dyn else None,
        "strict_floating_body_frames": strict_body_frames,
        # frame-level: frames with >= 1 floating dynamic body (saturates)
        "floating_frame_rate": float(floating.sum() / n) if n else None,
        "floating_frames": int(floating.sum()),
        "total_frames": n,
        "all_floating_frame_rate": float(any_floating.sum() / n) if n else None,
        "static_floating_bodies": int(sum(1 for b in range(sc.N) if any(b in x for x in static_))),
    }
    if not detail:
        return summary

    edge_sets = {"local": has_edge, "strict": s_has_edge}
    current = ["local"]

    def cause_of(b, t):
        if b in edge_sets[current[0]][t]:
            return "no_anchor_path", None
        if near_moving[b, t]:
            return "contact_not_bonded", None
        if lateral[b, t]:
            return "lateral_only_contact", None
        g = nearest[b, t]
        return "no_contact", ("2-5mm" if g <= 0.005 else "5-20mm" if g <= 0.02 else
                              "20-30mm" if g <= 0.03 else ">30mm")

    def tally(sets, mode):
        current[0] = mode
        causes, buckets = Counter(), Counter()
        per_body = defaultdict(lambda: {"frames": [], "causes": Counter(), "gaps": [], "partners": Counter()})
        for t in range(n):
            for b in sets[t]:
                c, bucket = cause_of(b, t)
                causes[c] += 1
                if bucket:
                    buckets[bucket] += 1
                pb = per_body[b]
                pb["frames"].append(t)
                pb["causes"][c] += 1
                if np.isfinite(nearest[b, t]):
                    pb["gaps"].append(float(nearest[b, t]))
                if partner[b, t] >= 0:
                    pb["partners"][int(partner[b, t])] += 1
        bodies = []
        for b, pb in sorted(per_body.items(), key=lambda kv: (-len(kv[1]["frames"]), sc.names[kv[0]])):
            gaps = sorted(pb["gaps"])
            bodies.append({
                "name": sc.names[b], "identity": sc.identities[b], "frames": len(pb["frames"]),
                "ranges": ranges(pb["frames"])[:20], "first_frame": pb["frames"][0],
                "moving_frames": int(sum(1 for t in pb["frames"] if sc.moving[b, t])),
                "dominant_cause": pb["causes"].most_common(1)[0][0], "causes": dict(pb["causes"]),
                "median_nearest_gap_mm": round(gaps[len(gaps) // 2] * 1000, 2) if gaps else None,
                "nearest_partner": sc.names[pb["partners"].most_common(1)[0][0]] if pb["partners"] else None})
        return causes, buckets, bodies

    causes, gap_buckets, offending = tally(dynamic, "local")
    s_causes, _, s_offending = tally(s_dynamic, "strict")
    st_causes, _, static_bodies = tally(static_, "strict")
    mid = n // 2
    static_comps = [c for c in s_comps[mid] if any(not animated[b] for b in c)] if n else []
    moving_now = sum(1 for t in range(n) if any(sc.moving[b, t] for b in dynamic[t]))
    events = ranges(floating)
    longest = max(events, key=lambda r: r[1] - r[0]) if events else None
    # static bodies held to the ground only through static-static contact (unmodelled fasteners);
    # a strict-pass notion: under local support every static body is itself a source
    mounted_counter = Counter(b for mo in s_mounted_only for b in mo)
    feet = [b for b in range(sc.N) if tokens(sc.names[b], sc.identities[b]) &
            {"foot", "feet", "ankle", "sole", "toe", "toes", "paw", "paws", "hoof"}]
    kinds = Counter(int(k) for k in edges[3])
    summary.update({
        "floating_frame_ranges": events,
        "events": {"count": len(events), "per_minute": len(events) / (n / sc.fps / 60) if n else None,
                   "longest_frames": (longest[1] - longest[0] + 1) if longest else 0,
                   "longest_s": round((longest[1] - longest[0] + 1) / sc.fps, 3) if longest else 0.0,
                   "longest_range": longest},
        "dynamic_frames_by_instant_motion": {"offender_moving": moving_now,
                                             "offenders_at_rest": int(floating.sum()) - moving_now},
        "causes_body_frames": dict(sorted(causes.items())),
        "no_contact_nearest_gap": dict(sorted(gap_buckets.items())),
        "distinct_offending_bodies": len(offending),
        "offending_bodies": offending[:40],
        "static_floating": {
            "rule": "never-moving scored bodies with no path to the true ground (strict anchors)",
            "bodies": len(static_bodies),
            "frames_with_static_floating": int(sum(1 for x in static_ if x)),
            "causes_body_frames": dict(sorted(st_causes.items())),
            "ungrounded_static_groups_mid_frame": len(static_comps),
            "largest_ungrounded_static_group": max((sum(1 for b in c if not animated[b]) for c in static_comps),
                                                  default=0),
            "ungrounded_static_examples": [[sc.names[b] for b in c if not animated[b]][:6]
                                           for c in sorted(static_comps, key=len, reverse=True)[:10]],
            "top": [{k: v for k, v in x.items() if k not in ("ranges", "moving_frames")} for x in static_bodies[:40]]},
        "strict_ground_path": {
            "rule": "support only from the true world anchors: a moving part also counts when the static "
                    "structure it is mounted on does not reach the ground",
            "inherited_body_frames": strict_body_frames - dyn_body_frames,
            "distinct_offending_bodies": len(s_offending),
            "causes_body_frames": dict(sorted(s_causes.items())),
            "offending_bodies": [{k: v for k, v in x.items() if k != "ranges"} for x in s_offending[:40]]},
        "animated_bodies": int(animated.sum()),
        "unsupported_components": {
            "component_frames": sum(len(c) for c in components),
            "max_size": max((len(x) for c in components for x in c), default=0)},
        "mounted_only_supported": {
            "body_frames": int(sum(mounted_counter.values())), "bodies": len(mounted_counter),
            "top": [{"name": sc.names[b], "frames": c} for b, c in mounted_counter.most_common(30)]},
        "free_flight": {"frames": len(free_frames),
                        "segments": [x for x in seg_out if x["ballistic"]][:60],
                        "rejected_segments": [x for x in seg_out if not x["ballistic"]
                                              and any(animated[sc.names.index(m)] for m in x["bodies"])][:60],
                        "segments_total": len(seg_out),
                        "short_undetermined_segments": undetermined},
        "foot_bodies_unsupported_frames": int(sum(1 for t in range(n) if any(b in dynamic[t] for b in feet))),
        "edge_frames_by_kind": {KIND_NAMES[k]: int(v) for k, v in sorted(kinds.items())},
    })
    return summary


def animated_bodies(sc: Scene):
    """A body is *dynamic* if the animation moves it at all: MOVING in some active
    frame (`motion.py`) and displaced >= `animated_m` (1 mm) from its first
    active pose at some frame (box corners, so pure rotation counts). Never-
    moving bodies are scene fixtures: their floating is reported, not counted."""
    if getattr(sc, "_animated", None) is not None:
        return sc._animated
    out = np.zeros(sc.N, dtype=bool)
    for b in range(sc.N):
        idx = np.flatnonzero(sc.active0[b])
        if len(idx) < 2 or not (sc.moving[b] & sc.active0[b]).any():
            continue
        h = sc.hid[b, idx]
        corners = np.stack([_corners(sc.mesh_lo[x], sc.mesh_hi[x]) for x in h])
        Mb = sc.M[b, idx].astype(np.float64)
        world = np.einsum("tij,tkj->tki", Mb[:, :, :3], corners) + Mb[:, None, :, 3]
        out[b] = float(np.linalg.norm(world - world[0], axis=2).max()) >= sc.p["animated_m"]
    sc._animated = out
    return out
