"""Turn time-varying deformation (caches, lattices, hooks, geometry nodes, moving booleans, shape-key caches...) into
glTF morph targets.

glTF only carries node transforms, skins, and morph weights, so a mesh deformed by anything but its armature would
export frozen at one frame. Each such mesh is sampled at every frame (bake_morphs): what moves as a whole becomes
transform keys, the rest morphs, either principal shapes mixed by per-frame weights or keyframes crossfaded in turn,
whichever fits closer (_fit); long rich deformation is split into spans of frames (_fit_spans), a mesh whose faces
change into an object per topology (_resample). A per-frame shape-key cache is computed from its keys rather than
evaluated (KeyMix): the same numbers, in milliseconds instead of seconds a frame."""
import collections, hashlib, json, os, subprocess, sys, time
from contextlib import contextmanager
import bpy
import numpy as np
from mathutils import Matrix

DEFORM = {"MESH_CACHE", "MESH_SEQUENCE_CACHE", "LATTICE", "HOOK", "SHRINKWRAP", "NODES", "DISPLACE", "CURVE",
          "SIMPLE_DEFORM", "WAVE", "CAST", "SURFACE_DEFORM", "MESH_DEFORM", "CLOTH", "SOFT_BODY", "WARP"}
TOL, KMAX, STEP = 0.002, 48, 1
# Morph fits aim for FIT_TOL (m; layered sheets need it: a label rides a fraction of a millimeter off its backing),
# FIT_TOL_BIG for meshes of BIG vertices or more, with at most MORPH_MAX targets (a three.js material passes each
# mesh's weights as uniforms) and vertices x targets within VK_MAX (an 80k-vertex surface at about 50 targets, under
# 10 MB of GLB). A mesh is split into spans of frames only when one fit misses by more than SPAN_TOL.
FIT_TOL, FIT_TOL_BIG, BIG, MORPH_MAX, VK_MAX, SPAN_TOL = 0.0005, 0.002, 10000, 128, 4_000_000, 0.002
# The scene's deforming meshes share SCENE_VK vertices x targets, by vertex count (about 6 bytes each once
# compressed; without the cap, four cloth meshes at VK_MAX each came to about 100 MB).
SCENE_VK = 4_000_000
# a mesh whose faces change gets an object per topology up to these; beyond, one shape fitted to each frame's box.
# A topology shown in TOPO_MAIN of the frames stands in for the rare others (held over them).
TOPO_GROUPS, TOPO_VERTS, TOPO_MAIN = 48, 200_000, 0.9


def _coords(o, dg, topology=False):
    """o's evaluated vertex positions (object space); with topology, also a signature of its faces."""
    e = o.evaluated_get(dg)
    m = e.to_mesh()
    a = np.empty(len(m.vertices) * 3, np.float32)
    m.vertices.foreach_get("co", a)
    sig = _signature(m) if topology else None
    e.to_mesh_clear()
    return (a.reshape(-1, 3), sig) if topology else a.reshape(-1, 3)


def _signature(me):
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    lt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return hashlib.md5(vi.tobytes() + lt.tobytes() + str(len(me.vertices)).encode()).hexdigest()[:16]


def _curves(action, slot):
    """The F-curves an action plays for one slot (4.4+ layered actions), or for its only slot."""
    if action is None:
        return []
    try:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
        return list(cb.fcurves) if cb else []
    except (ImportError, AttributeError):
        return list(action.fcurves)


def _support(fc):
    """Action-time interval outside which F-curve fc is exactly 0, or None when it is 0 everywhere. Conservative: the
    hull of every segment with a nonzero key or Bezier handle, open-ended where the extrapolation carries a value."""
    if fc.modifiers:
        return -np.inf, np.inf
    kp = fc.keyframe_points
    if not len(kp):
        return None
    xs = [p.co[0] for p in kp]
    nz = lambda i: kp[i].co[1] != 0
    lo, hi = np.inf, -np.inf
    for i in range(len(kp) - 1):
        bez = kp[i].interpolation == 'BEZIER'
        if nz(i) or (kp[i].interpolation != 'CONSTANT' and nz(i + 1)) \
                or (bez and (kp[i].handle_right[1] != 0 or kp[i + 1].handle_left[1] != 0)):
            lo, hi = min(lo, xs[i]), max(hi, xs[i + 1])
        if nz(i + 1):                           # the key itself, where a CONSTANT segment ends
            lo, hi = min(lo, xs[i + 1]), max(hi, xs[i + 1])
    if nz(0) or (fc.extrapolation != 'CONSTANT' and lo <= hi):
        lo = -np.inf
    if nz(len(kp) - 1) or (fc.extrapolation != 'CONSTANT' and lo <= hi):
        hi = np.inf
    # padded: within 0.01 frame of a key, Blender takes that key's value
    return (lo - 0.02, hi + 0.02) if lo <= hi else None


def _evaluatable(fc):
    """Blender's is_fcurve_evaluatable: curves it skips write nothing (and aren't in the NLA's domain)."""
    return not fc.mute and fc.is_valid and not (fc.group and fc.group.mute) \
        and (len(fc.keyframe_points) or len(fc.modifiers))


class _Curves:
    """The key F-curves of one action slot: key index, F-curve, and the support intervals as arrays."""

    def __init__(self, fcurves, index):
        self.channels, self.live, self.other = set(), [], []
        for fc in fcurves:
            if not _evaluatable(fc):
                continue
            i = index.get(fc.data_path)
            if i is None:
                self.other.append(fc.data_path)    # a key's range, mute... animated: not computed here
                continue
            self.channels.add(i)
            s = _support(fc)
            if s is not None:
                self.live.append((i, fc, s[0], s[1]))
        self.lo = np.array([x[2] for x in self.live])
        self.hi = np.array([x[3] for x in self.live])

    def at(self, t):
        """(key index, value) of the curves that can be nonzero at action time t."""
        return [(self.live[k][0], self.live[k][1].evaluate(t)) for k in np.flatnonzero((self.lo <= t) & (t <= self.hi))]


class KeyMix:
    """An object's shape as Blender evaluates its relative shape keys, computed here only from the keys that are not
    zero at a frame: reference + sum of weight * (key - its relative key), in key order, in float32.

    The weights come from the Key's own animation, played the way Blender's animation system does: an action, or NLA
    tracks of action strips at full influence that replace (the strip a track plays at a frame, its time mapping and
    hold rules, the upper track winning, the channels no strip writes at that frame back at their default of 0).
    Each F-curve is evaluated by Blender's own F-curve code, at the frames it can be nonzero.

    A per-frame shape cache of thousands of keys (effect meshes with thousands of keys on several NLA tracks) makes
    Blender's frame change cost a second or more, mostly finding every curve's key_blocks["..."] by name; computed here, a frame is
    under a millisecond. KeyMix.check is None for the objects this holds for and otherwise says why not; the
    callers also compare it against Blender's evaluation before using it."""

    @staticmethod
    def check(o):
        k = o.data.shape_keys if o.type == 'MESH' else None
        if k is None:
            return 'no shape keys'
        if not k.use_relative:
            return 'absolute keys'
        if any(m.show_viewport for m in o.modifiers):
            return 'modifiers'
        if o.show_only_shape_key:
            return 'shows one key'
        if any(kb.vertex_group for kb in k.key_blocks):
            return 'key vertex groups'
        ad = k.animation_data
        if ad is None:
            return None
        if len(ad.drivers):
            return 'drivers'
        if ad.use_tweak_mode:
            return 'NLA tweak mode'
        tracks = [t for t in ad.nla_tracks if not t.mute] if ad.use_nla else []
        if any(t.is_solo for t in ad.nla_tracks):
            return 'solo NLA track'
        if tracks and ad.action:
            return 'action over NLA'
        for t in tracks:
            for s in t.strips:
                if s.type != 'CLIP' or s.blend_type != 'REPLACE' or s.use_animated_influence or s.use_animated_time \
                        or s.blend_in or s.blend_out or s.use_reverse:
                    return f'NLA strip {s.name}: {s.type} {s.blend_type}'
        return None

    def __init__(self, o):
        self.name = o.name
        k = o.data.shape_keys
        self.kbs = list(k.key_blocks)
        self.n = len(o.data.vertices)
        index = {f'key_blocks["{bpy.utils.escape_identifier(kb.name)}"].value': i for i, kb in enumerate(self.kbs)}
        self.ref = self._co(k.reference_key)
        ref = self.kbs.index(k.reference_key)
        # the keys that add to the mix, each with the key its offset is measured from (Blender's do_rel_key)
        self.rel = {i: (self.kbs.index(kb.relative_key) if kb.relative_key else ref) for i, kb in enumerate(self.kbs)
                    if i != ref and not kb.mute}
        self.clamp = [(kb.slider_min, kb.slider_max) for kb in self.kbs]
        self.delta = {}
        ad = k.animation_data
        self.action, self.tracks = None, []
        domain = set()
        if ad and ad.use_nla and any(not t.mute for t in ad.nla_tracks):
            for t in ad.nla_tracks:             # bottom to top
                if t.mute:
                    continue
                strips = []
                for s in t.strips:
                    c = _Curves(_curves(s.action, getattr(s, 'action_slot', None)), index)
                    domain |= c.channels
                    strips.append((s.frame_start, s.frame_end, s.action_frame_start, s.action_frame_end, s.scale,
                                   s.repeat, s.extrapolation, s.mute, c))
                self.tracks.append(strips)
        elif ad and ad.action:
            self.action = _Curves(_curves(ad.action, getattr(ad, 'action_slot', None)), index)
        if self.action:
            domain = self.action.channels
        self.other = sorted({p for c in [self.action] + [s[8] for t in self.tracks for s in t] if c for p in c.other})
        # keys no curve touches keep their stored value; the others are written every frame, the NLA's with its
        # default (0, through the slider range like any value) where no strip plays them
        self.static = {i: kb.value for i, kb in enumerate(self.kbs) if i not in domain and kb.value != 0}
        self.static.update({i: self._clamp(i, 0.0) for i in domain if self._clamp(i, 0.0) != 0})

    def _co(self, kb):
        a = np.empty(self.n * 3, np.float32)
        kb.data.foreach_get('co', a)
        return a.reshape(-1, 3)

    @staticmethod
    def _strip_at(strips, f):
        """The strip a track plays at frame f and its action time (Blender's nlastrips_ctime_get_strip and
        nlastrip_get_frame_actionclip), or None."""
        est, side = None, 0
        for i, s in enumerate(strips):
            start, end, ext = s[0], s[1], s[6]
            if start <= f <= end:
                est = s
                break
            if f < start:
                if i == 0:
                    est, side = (s if ext == 'HOLD' else None), -1
                else:
                    est, side = (strips[i - 1] if strips[i - 1][6] != 'NOTHING' else None), 1
                break
            if f > end and i == len(strips) - 1:
                est, side = (s if ext != 'NOTHING' else None), 1
                break
        if est is None or est[7]:
            return None
        start, end, astart, aend, scale, repeat = est[:6]
        f = start if side < 0 else end if side > 0 else f
        scale = abs(scale) or 1.0
        repeat = repeat or 1.0
        alen = (aend - astart) or 1.0
        if f == end and repeat == int(repeat):
            return est[8], astart + alen
        return est[8], astart + np.fmod(f - start, alen * scale) / scale

    def weights(self, f):
        w = dict(self.static)
        if self.action:
            w.update(self.action.at(f))
        claimed = []                            # channels of the strips above, which replace these
        for strips in reversed(self.tracks):
            hit = self._strip_at(strips, f)
            if hit is None:
                continue
            c, t = hit
            for i, v in c.at(t):
                if not any(i in u for u in claimed):
                    w[i] = v
            claimed.append(c.channels)
        return {i: self._clamp(i, v) for i, v in w.items()}

    def _clamp(self, i, v):
        return min(max(v, self.clamp[i][0]), self.clamp[i][1])

    def coords(self, f):
        co = self.ref.copy()
        for i, v in sorted(self.weights(f).items()):
            if v == 0 or i not in self.rel:
                continue
            d = self.delta.get(i)
            if d is None:
                d = self.delta[i] = self._co(self.kbs[i]) - self._co(self.kbs[self.rel[i]])
            co += np.float32(v) * d
        return co


@contextmanager
def muted_keys(objs):
    """While inside: the shape-key animation of these objects muted, so a frame change doesn't evaluate it."""
    saved = []
    for o in objs:
        ad = o.data.shape_keys.animation_data if o.data.shape_keys else None
        if ad:
            saved.append((ad, ad.action, getattr(ad, 'action_slot', None), [(t, t.mute) for t in ad.nla_tracks]))
            ad.action = None
            for t in ad.nla_tracks:
                t.mute = True
    try:
        yield
    finally:
        for ad, act, slot, tracks in saved:
            ad.action = act
            if slot is not None and getattr(ad, 'action_slot', None) != slot:
                ad.action_slot = slot
            for t, mute in tracks:
                t.mute = mute


def _refs(x):
    """Objects x's modifiers and constraints point at (shrinkwrap targets, geometry-node inputs...)."""
    out = set()
    for m in list(getattr(x, 'modifiers', ())) + list(getattr(x, 'constraints', ())):
        for p in m.bl_rna.properties:
            if p.type == 'POINTER':
                v = getattr(m, p.identifier, None)
                if isinstance(v, bpy.types.Object):
                    out.add(v.name)
        for key in (m.keys() if m.type == 'NODES' else ()):
            if isinstance(m[key], bpy.types.Object):
                out.add(m[key].name)
    return out


def check_frames(sc, mixes, n=10):
    """Frames to compare KeyMix with Blender at: spread over the film, and either side of every strip's ends."""
    fs = {round(sc.frame_start + (sc.frame_end - sc.frame_start) * i / (n - 1)) for i in range(n)}
    for m in mixes:
        for strips in m.tracks:
            for s in strips:
                for e in (s[0], s[1]):
                    fs |= {int(np.floor(e)) - 1, int(np.floor(e)), int(np.ceil(e)), int(np.ceil(e)) + 1}
    fs = sorted(f for f in fs if sc.frame_start <= f <= sc.frame_end)
    if len(fs) > 4 * n:                         # many strips: an even pick of them
        fs = [fs[round(i * (len(fs) - 1) / (4 * n - 1))] for i in range(4 * n)]
    return fs


def key_mixes(sc, objs, others=()):
    """{name: KeyMix} for the objects of objs whose shape KeyMix computes (and that nothing in others reads)."""
    used = set().union(*(_refs(x) for x in others)) if others else set()
    out = {}
    for o in objs:
        why = KeyMix.check(o) or ('read by another object' if o.name in used else None)
        m = KeyMix(o) if why is None else None
        if m and m.other:
            why = 'animates ' + m.other[0]
        if why is None:
            out[o.name] = m
        elif o.data.shape_keys and len(o.data.shape_keys.key_blocks) > KMAX + 1:
            print(f'shape keys: {o.name} ({len(o.data.shape_keys.key_blocks)} keys) evaluated by Blender: {why}',
                  flush=True)
    return out


def _rigid(P, Q):
    """4x4 rigid transform best mapping points P onto Q (Kabsch)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H.astype(np.float64))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, cq - R @ cp
    return M


def _pick(S, tol=TOL, kmax=KMAX):
    """Greedy keyframe indices so piecewise-linear interpolation of S stays within tol, and that error.

    Error is computed on a vertex sample, and each segment's worst frame is cached."""
    if S.shape[1] > 3000:
        S = S[:, np.random.default_rng(0).choice(S.shape[1], 3000, replace=False)]

    def worst(a, b):
        if b - a < 2:
            return 0.0, None
        t = (np.arange(a + 1, b) - a)[:, None, None] / (b - a)
        err = np.abs(S[a + 1:b] - ((1 - t) * S[a] + t * S[b])).max(axis=(1, 2))
        k = int(err.argmax())
        return float(err[k]), a + 1 + k

    keys = [0, len(S) - 1]
    seg = {(0, len(S) - 1): worst(0, len(S) - 1)}
    while len(keys) < kmax:
        (a, b), (e, wi) = max(seg.items(), key=lambda kv: kv[1][0])
        if e <= tol or wi is None:
            break
        del seg[(a, b)]
        seg[(a, wi)], seg[(wi, b)] = worst(a, wi), worst(wi, b)
        keys = sorted(keys + [wi])
    return keys, max(e for e, _ in seg.values())


def _pca(S, shown, kmax, tol):
    """Mean shape, principal shapes, and each frame's weights for S (frames x verts x 3), from the shown frames, with
    the fewest shapes that keep every shown frame within tol (else kmax), and that error (on a vertex sample).

    A feeding web (vertices sliding along a fixed path), a pour or cloth is close to a few shapes mixed
    in varying amounts: where keyframes crossfading in turn missed by 5 mm, 16 principal shapes miss by under 0.05."""
    F, V, _ = S.shape
    X = S.reshape(F, -1)
    mu = X[shown].mean(0)
    Xc = (X[shown] - mu).astype(np.float32)
    B = _top_basis(Xc, kmax)
    W = (X - mu) @ B.T
    cols = np.arange(X.shape[1]) if V <= 3000 else \
        (3 * np.random.default_rng(0).choice(V, 3000, replace=False)[:, None] + np.arange(3)).ravel()
    Xs, Ws, Bs = X[:, cols][shown], W[shown], B[:, cols]
    err = np.inf
    for k in [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if k < len(B)] + [len(B)]:
        err = float(np.abs(Xs - mu[cols] - Ws[:, :k] @ Bs[:k]).max())
        if err <= tol:
            break
    return mu.reshape(V, 3), B[:k].reshape(k, V, 3), W[:, :k], err


def _after_armature(o):
    """Skinned, with deformation stacked after the armature (a strap fitted by shrinkwraps and geometry nodes). A
    glTF skin carries only the armature, and the exporter applies the rest at the rest pose,
    so the part ends up meters from the body. Such meshes are resampled like the others and lose their skin."""
    t = [m.type for m in o.modifiers]
    return "ARMATURE" in t and any(x in DEFORM for x in t[t.index("ARMATURE") + 1:])


def _keys_under_modifiers(o):
    """Shape keys under modifiers other than an armature (a handle with two keys, then a solidify). The
    exporter applies the modifiers at one frame, then adds the keys' offsets on top again, matched by vertex index
    to the modified mesh (the solidify's copies take other vertices' offsets)."""
    return o.type == "MESH" and o.data.shape_keys is not None \
        and any(m.show_viewport and m.type != "ARMATURE" for m in o.modifiers) \
        and not any(m.type == "ARMATURE" for m in o.modifiers)


def _keys_animated(o):
    ad = o.data.shape_keys.animation_data
    return bool(ad and (ad.action or len(ad.nla_tracks) or len(ad.drivers)))


def _moving_booleans(sc, step=6):
    """Meshes with a boolean whose operand moves relative to them (a liquid level: a cutter rising through each
    bottle's liquid), found from object transforms every few frames. A boolean between parts moving together, or
    still, isn't deformation (and mechanisms have many)."""
    pairs = []
    for o in sc.objects:
        if o.type != "MESH" or any(m.type == "ARMATURE" for m in o.modifiers):
            continue
        for m in o.modifiers:
            if m.type == "BOOLEAN" and m.show_viewport:
                ops = [m.object] if m.operand_type == "OBJECT" else list(m.collection.all_objects) if m.collection else []
                pairs += [(o, c) for c in ops if c is not None]
    if not pairs:
        return set()
    from visibility import cheap_eval
    first, moving = {}, set()
    cur = sc.frame_current
    with cheap_eval():                          # transforms only
        for f in list(range(sc.frame_start, sc.frame_end + 1, step)) + [sc.frame_end]:
            sc.frame_set(f)
            for o, c in pairs:
                if o.name in moving:
                    continue
                rel = o.matrix_world.inverted_safe() @ c.matrix_world
                r0 = first.setdefault((o.name, c.name), rel.copy())
                if max(abs(a - b) for ra, rb in zip(rel, r0) for a, b in zip(ra, rb)) > 1e-5:
                    moving.add(o.name)
    sc.frame_set(cur)
    return moving


def _live_sims(sc):
    """Simulations evaluated frame by frame from the start (no baked cache): sampling them out of order is wrong."""
    live = [o.name for o in sc.objects for m in getattr(o, 'modifiers', ())
            if m.type in ('CLOTH', 'SOFT_BODY', 'DYNAMIC_PAINT', 'FLUID') and hasattr(m, 'point_cache')
            and not m.point_cache.is_baked]
    live += [o.name for o in sc.objects for p in getattr(o, 'particle_systems', ()) if not p.point_cache.is_baked]
    if sc.rigidbody_world and sc.rigidbody_world.enabled and not sc.rigidbody_world.point_cache.is_baked:
        live.append('rigid body world')
    return live


class Samples:
    """Per candidate and frame: evaluated vertex positions (object space), matrix_basis, a topology signature, and
    whether it is hidden from render."""

    def __init__(self, names):
        self.co, self.basis, self.topo, self.hidden = ({n: [] for n in names} for _ in range(4))

    def extend(self, other):
        for a, b in ((self.co, other.co), (self.basis, other.basis), (self.topo, other.topo), (self.hidden, other.hidden)):
            for n in a:
                a[n] += b[n]


def _sample(sc, cands, frames, mixes=None):
    """Samples of each candidate at each frame. The shapes of the objects in mixes ({name: KeyMix}) are computed
    instead, their shape-key animation muted for the frame changes."""
    mixes = mixes or {}
    out = Samples([o.name for o in cands])
    with muted_keys([o for o in cands if o.name in mixes]):
        for f in frames:
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            for o in cands:
                m = mixes.get(o.name)
                co, sig = (m.coords(f), 'keys') if m else _coords(o, dg, topology=True)
                out.co[o.name].append(co)
                out.topo[o.name].append(sig)
                out.basis[o.name].append(o.matrix_basis.copy())
                out.hidden[o.name].append(bool(o.hide_render))
    return out


def _verify_mixes(sc, cands, mixes):
    """Drop the KeyMix shapes that differ from Blender's evaluation at check_frames (and all of them if muting their
    animation changed another candidate's shape). Returns what is left."""
    checks = check_frames(sc, mixes.values())
    t0 = time.time()
    ref = _sample(sc, cands, checks)
    t1 = time.time()
    got = _sample(sc, cands, checks, mixes)
    bad = {}
    for o in cands:
        for f, a, b, ma, mb in zip(checks, ref.co[o.name], got.co[o.name], ref.basis[o.name], got.basis[o.name]):
            err = float(np.abs(a - b).max()) if a.shape == b.shape and a.size else (0.0 if a.shape == b.shape else np.inf)
            err = max(err, max(abs(x - y) for ra, rb in zip(ma, mb) for x, y in zip(ra, rb)))
            if err > 1e-5 * max(1.0, float(np.abs(a).max()) if a.size else 1.0):
                bad[o.name] = (f, err)
                break
    print(f'shape keys: {len(mixes)} computed, checked at {len(checks)} frames: Blender {(t1 - t0) / len(checks):.2f}s '
          f'a frame, computed {(time.time() - t1) / len(checks):.2f}s', flush=True)
    for n, (f, err) in bad.items():
        print(f'shape keys: {n} differs from Blender at frame {f} by {err:.2g}', flush=True)
    if any(n not in mixes for n in bad):
        return {}
    return {n: m for n, m in mixes.items() if n not in bad}


def _blender_binary():
    """The Blender executable to start sampling workers with, or '' under the bpy wheel, which has none."""
    return bpy.app.binary_path or ''


def _meminfo(field, path='/proc/meminfo'):
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(field + ':'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


#: A worker's memory as a multiple of this process's own: each worker opens the whole scene again and evaluates it
#: (sampling measured at about 1.8 times the scene, baking, with Cycles' buffers and the images it writes, at about
#: twice it).
WORKER_MEMORY = {'sampling': 2.0, 'baking': 2.5}


#: Fitting holds every sampled frame of the deforming meshes, as float32 and again in float64 while it decomposes
#: them: about 2.5 times the samples (measured 2.4 on a film with 2.9 GB of samples).
FIT_MEMORY = 2.5


def worker_count(what, cap=4, then=0):
    """How many worker processes `what` (sampling, baking) gets: `MORPH_WORKERS` / `BAKE_WORKERS` when set, else as
    many as fit, at most `cap` and half the CPUs, within 80% of the memory available. Prints the expected peak, and
    the peak of the step after it when `then` (its bytes beyond this process) is given."""
    rss, avail = _meminfo('VmRSS', '/proc/self/status'), _meminfo('MemAvailable')
    each = rss * WORKER_MEMORY[what]
    env = os.environ.get('BAKE_WORKERS' if what == 'baking' else 'MORPH_WORKERS')
    if env:
        n = max(1, int(env))
    elif not _blender_binary():
        n = 1
    else:
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        fit = int(0.8 * avail // each) if each and avail else 1
        n = max(1, min(cap, ncpu // 2, fit))
    gb = lambda b: f"{b / 1e9:.1f} GB"
    peak = rss + each * n if n > 1 else rss
    after = f", then about {gb(rss + then)} while fitting the morph targets" if then else ""
    print(f"memory: {what} in {n} process{'es' if n > 1 else ''}: the scene takes {gb(rss)}, so about {gb(peak)} "
          f"while {what}{after} ({gb(avail)} available; --jobs sets the number of {what} processes)", flush=True)
    return n


def _sample_parallel(sc, cands, frames, workers, tmp, mixes=None):
    """_sample over contiguous chunks of frames in `workers` Blender processes, from a saved copy of this state."""
    os.makedirs(tmp, exist_ok=True)
    blend = os.path.join(tmp, 'state.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True, compress=False)
    names = [o.name for o in cands]
    chunks = [frames[i * len(frames) // workers:(i + 1) * len(frames) // workers] for i in range(workers)]
    procs = []
    for i, ch in enumerate(chunks):
        spec = os.path.join(tmp, f'chunk{i}.json')
        json.dump({'names': names, 'frames': ch, 'mixes': sorted(mixes or ())}, open(spec, 'w'))
        log = open(os.path.join(tmp, f'chunk{i}.log'), 'w')
        procs.append((subprocess.Popen([_blender_binary(), '-b', '--factory-startup', blend, '--python-exit-code', '1',
                                        '--python', os.path.abspath(__file__), '--', '--sample', spec,
                                        os.path.join(tmp, f'chunk{i}.npz')], stdout=log, stderr=subprocess.STDOUT), log))
    for i, (p, log) in enumerate(procs):
        p.wait()
        log.close()
        if p.returncode:
            raise RuntimeError(f'morph sampling worker {i} failed; see {log.name}')
    out = Samples(names)
    for i in range(workers):
        with np.load(os.path.join(tmp, f'chunk{i}.npz')) as z:      # closed before the folder goes (NFS keeps open files)
            part = Samples(names)
            for k, n in enumerate(names):
                part.co[n] = np.split(z[f'c{k}'], np.cumsum(z[f'n{k}'])[:-1])
                part.topo[n] = [t.decode() for t in z[f't{k}']]
                part.hidden[n] = list(z[f'h{k}'])
                part.basis[n] = [Matrix(m.reshape(4, 4).tolist()) for m in z[f'b{k}']]
            out.extend(part)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def bake_morphs(sc, tmp=None):
    """Turn every mesh of sc that deforms other than by its armature into morph targets; return their names."""
    # Also resample meshes carrying a per-frame shape-key cache: hundreds of keys would bloat the GLB.
    # Absolute (non-relative) keys blend by evaluation time, which glTF morph targets cannot express either.
    many_keys = lambda o: o.data.shape_keys is not None and (len(o.data.shape_keys.key_blocks) > KMAX + 1
                                                             or not o.data.shape_keys.use_relative)
    booleans = _moving_booleans(sc)
    cands = [o for o in sc.objects if o.type == "MESH"
             and (_after_armature(o) or o.name in booleans
                  or (many_keys(o) or any(m.type in DEFORM for m in o.modifiers)
                      or _keys_under_modifiers(o) and _keys_animated(o))
                  and not any(m.type == "ARMATURE" for m in o.modifiers))]
    # still keys under modifiers: the shape they make, frozen
    for o in sc.objects:
        if o not in cands and _keys_under_modifiers(o) and not _keys_animated(o):
            dg = bpy.context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            _use_mesh(o, me, co.reshape(-1, 3))
            print("morphs:", o.name, "still shape keys under modifiers: frozen", flush=True)
    if not cands:
        print("morphs: none", flush=True)
        return []
    frames = list(range(sc.frame_start, sc.frame_end + 1, STEP))
    if frames[-1] != sc.frame_end:
        frames.append(sc.frame_end)
    # Sampling is a full-scene evaluation per frame. A per-frame shape-key cache (thousands of keys, which make each
    # frame change cost a second) is computed instead from the keys not zero at each frame (KeyMix), once that matches
    # Blender's evaluation at check frames. What is still slow is split over MORPH_WORKERS processes when it would
    # take over a minute and nothing in the scene needs its frames in order (a simulation without a baked cache).
    live = _live_sims(sc)
    mixes = key_mixes(sc, [o for o in cands if many_keys(o)], others=cands)
    if mixes and not live:
        mixes = _verify_mixes(sc, cands, mixes)
    elif mixes:
        mixes = {}                              # the check frames would run a live simulation out of order
    sc.frame_set(sc.frame_start)
    t0 = time.time()
    head = _sample(sc, cands, frames[:3], mixes)
    est = (time.time() - t0) / 3 * len(frames)
    parallel = tmp and not live and est > float(os.environ.get('MORPH_MIN_SECONDS', 60)) and len(frames) > 32
    sampled = sum(len(head.co[n][0]) for n in head.co if head.co[n]) * 12 * len(frames)
    workers = worker_count('sampling', then=int(FIT_MEMORY * sampled)) if parallel else 1
    if workers > 1 and tmp and not live and est > float(os.environ.get('MORPH_MIN_SECONDS', 60)) \
            and len(frames) > 8 * workers:
        print(f'morphs: sampling {len(frames)} frames in {workers} workers (about {est / 60:.0f} min in one)', flush=True)
        sc.frame_set(sc.frame_start)
        smp = _sample_parallel(sc, cands, frames, workers, tmp, mixes)
    else:
        if live and est > 60:
            print('morphs: sampling in order, live simulations:', ' '.join(live[:5]), flush=True)
        smp = head
        smp.extend(_sample(sc, cands, frames[3:], mixes))
    budget = _share(smp, [o.name for o in cands])
    names = [o.name for o in cands]
    for o in cands:
        _resample(sc, o, frames, smp, budget.get(o.name, VK_MAX))
    sc.frame_set(sc.frame_start)
    return names


def _share(smp, names):
    """SCENE_VK split among the meshes that deform: each one's need (vertices x the principal shapes that reach its
    tolerance on a vertex sample; VK_MAX for those that don't within MORPH_MAX) where the needs fit, and the rest
    shared by vertex count among those wanting more (a 36k-vertex rail needs 2 shapes; cloth straps want them all)."""
    need, verts = {}, {}
    for n in names:
        S, H = smp.co[n], smp.hidden[n]
        if len({len(c) for c in S}) > 1:
            continue                            # split by topology or boxed: _resample divides what it gets
        S = np.stack(S)
        shown = ~np.asarray(H, bool) if not all(H) else np.ones(len(S), bool)
        V = S.shape[1]
        tol = FIT_TOL if V < BIG else FIT_TOL_BIG
        if float(np.abs(S[shown] - S[shown][0]).max()) <= tol:
            continue
        X = S[:, np.random.default_rng(0).choice(V, min(V, 300), replace=False)].reshape(len(S), -1)[shown]
        Xc = (X - X.mean(0)).astype(np.float32)
        B = _top_basis(Xc, MORPH_MAX)
        R = Xc @ B.T
        k = next((k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if k <= len(B)
                  and float(np.abs(Xc - R[:, :k] @ B[:k]).max()) <= tol), None)
        # half again for what the sample misses
        need[n], verts[n] = (V * min(MORPH_MAX, -(-3 * k // 2)) if k else min(VK_MAX, V * 8 * MORPH_MAX)), V
    left, out = SCENE_VK, {}
    while need:
        per = left / max(1, sum(verts[n] for n in need))
        small = [n for n in need if need[n] <= per * verts[n]]
        if not small:
            out.update({n: min(need[n], int(per * verts[n])) for n in need})
            break
        for n in small:
            out[n] = min(VK_MAX, need.pop(n))
            left -= out[n]
    for n in names:
        if n in out:
            out[n] = max(out[n], 16 * verts[n])         # never fewer than 16 shapes
    print('morphs: budget', ' '.join(f'{n}={v // 1000}k' for n, v in sorted(out.items(), key=lambda kv: -kv[1])[:8]),
          flush=True)
    return out


def _resample(sc, o, frames, smp, vk=VK_MAX):
    """Replace o's deformation with morph targets over the frames (see _fit). A mesh whose faces change (a liquid from
    an Alembic cache) is split by topology: a copy per topology it shows, each following its frames and shown only at
    them. With too many topologies (hundreds), its most common one is fitted to each frame's box."""
    S, T, H, B = smp.co[o.name], smp.topo[o.name], smp.hidden[o.name], smp.basis[o.name]
    shown = [i for i in range(len(S)) if not H[i]] or list(range(len(S)))
    count = collections.Counter(T[i] for i in shown)
    if len(set(T)) == 1:
        sc.frame_set(sc.frame_start)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        if not _fit_spans(o, frames, np.stack(S), B, me, [not h for h in H], force=_keys_under_modifiers(o), vk=vk):
            bpy.data.meshes.remove(me)
        return
    groups = [g for g, _ in count.most_common()]
    if count[groups[0]] >= TOPO_MAIN * len(shown):
        groups = groups[:1]
    first = {g: next(i for i in shown if T[i] == g) for g in groups}
    nverts = sum(len(S[first[g]]) for g in groups)
    boxes = len(groups) > TOPO_GROUPS or nverts > TOPO_VERTS
    meshes = {}
    for g in groups[:1] if boxes else groups:
        sc.frame_set(frames[first[g]])
        dg = bpy.context.evaluated_depsgraph_get()
        meshes[g] = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    if boxes:
        g = groups[0]
        R = S[first[g]]
        rlo, rhi = R.min(0), R.max(0)
        rext = rhi - rlo
        Ms = []
        for i in range(len(S)):
            P = S[min(shown, key=lambda j: abs(j - i))]
            lo, hi = (P.min(0), P.max(0)) if len(P) else (rlo, rhi)
            k = np.where(rext > 1e-6, (hi - lo) / np.maximum(rext, 1e-12), 1.0)
            M = np.eye(4)
            M[:3, :3] = np.diag(k)
            M[:3, 3] = (lo + hi) / 2 - k * (rlo + rhi) / 2
            Ms.append(M)
        _use_mesh(o, meshes[g], R)
        _key_transforms(o, frames, B, Ms)
        print("morphs:", o.name, f"{len(groups)} topologies ({nverts} verts): its most common fitted to each frame's box",
              flush=True)
        return
    # copies first, of o as it is: fitting o gives it keys a copy taken after would inherit
    if len(groups) == 1:
        # one topology stands in for the rare others: shown whenever o is, holding its nearest shape over them
        _fit_spans(o, frames, _held(S, T, H, groups[0]), B, meshes[groups[0]], [not h for h in H], force=True, vk=vk)
        print("morphs:", o.name, f"{len(count)} topologies: the one in {count[groups[0]]} of {len(shown)} frames "
              "stands in for the others", flush=True)
        return
    xs = [o] + [_twin(o, f"{o.name}~t{k}") for k in range(1, len(groups))]
    for x, g in zip(xs, groups):
        vis = [not H[i] and T[i] == g for i in range(len(S))]
        _fit_spans(x, frames, _held(S, T, H, g), B, meshes[g], vis, force=True, keyed=True, vk=vk // len(groups))
    print("morphs:", o.name, f"{len(groups)} topologies ({nverts} verts): one object each", flush=True)


def _held(S, T, H, g):
    """Per frame, a shape of topology g: the frame's own where it shows g, else the nearest frame's that does."""
    idx = np.array([i for i in range(len(S)) if T[i] == g and not H[i]] or [i for i in range(len(S)) if T[i] == g])
    out = []
    for i in range(len(S)):
        j = int(np.searchsorted(idx, i))
        near = [idx[m] for m in (j - 1, j) if 0 <= m < len(idx)]
        out.append(S[min(near, key=lambda c: abs(c - i))])
    return np.stack(out)


def _twin(o, name):
    """A copy of o, in its collections and with its own action, for another of its topologies."""
    x = o.copy()
    x.name = name
    for c in o.users_collection:
        c.objects.link(x)
    return x


def _own_action(x):
    """Keys added to x's active action must not animate the objects sharing it (a copy, NLA batches)."""
    ad = x.animation_data
    if ad and ad.action and ad.action.users > 1:
        slot = getattr(ad, 'action_slot', None)
        ident = slot.identifier if slot else None
        ad.action = ad.action.copy()
        if ident and getattr(ad.action, 'slots', None) and ident in ad.action.slots:
            ad.action_slot = ad.action.slots[ident]


def _key_visibility(x, frames, shown):
    """x shown at the frames where shown is true, through delta_scale like visibility's channel."""
    ad = x.animation_data
    for d in [d for d in (ad.drivers if ad else ()) if d.data_path == 'delta_scale']:
        ad.drivers.remove(d)
    _own_action(x)
    prev = None
    for f, v in zip(frames, shown):
        if v != prev:
            x.delta_scale = (1.0,) * 3 if v else (1e-4,) * 3
            x.keyframe_insert('delta_scale', frame=f)
            prev = v
    for fc in _action_curves(x):
        if fc.data_path == 'delta_scale':
            for p in fc.keyframe_points:
                p.interpolation = 'CONSTANT'


def _action_curves(x):
    ad = x.animation_data
    return _curves(ad.action, getattr(ad, 'action_slot', None)) if ad and ad.action else []


def _use_mesh(o, me, co):
    """o's deformation replaced by mesh me at positions co: modifiers, shape keys and (after an armature) skin go."""
    me.vertices.foreach_set("co", co.ravel())
    skinned = _after_armature(o)
    o.modifiers.clear()
    o.data = me
    if skinned:
        # no skin left: without its groups (and an armature parent) the exporter can't bind it again
        o.vertex_groups.clear()
        if o.parent_type == "ARMATURE":
            o.parent_type = "OBJECT"
    if me.shape_keys:
        o.shape_key_clear()
    return skinned


def _key_transforms(o, frames, basis, Ms):
    """o's transform at each frame: its own (basis) times M. Every frame gets a key, overriding its animation."""
    _own_action(o)
    o.rotation_mode = "QUATERNION"
    q0 = None
    for f, B, M in zip(frames, basis, Ms):
        loc, q, scl = (B @ Matrix(M.tolist())).decompose()
        if q0 is not None and q.dot(q0) < 0:
            q.negate()
        q0 = q
        o.location, o.rotation_quaternion, o.scale = loc, q, scl
        for p in ("location", "rotation_quaternion", "scale"):
            o.keyframe_insert(p, frame=f)


def _top_basis(Xc, k):
    """The k leading principal directions (k x columns, orthonormal rows) of centered rows Xc, by SVD: a thin one when
    the matrix is small, else a randomized range finder with power iterations (Halko et al.), as close for k well below
    the frame count and built on matrix products (a thin SVD of a 1800-frame, 142k-value cloth takes over half an
    hour). Not through a Gram matrix: squaring the data's condition number costs thin layers a millimeter."""
    F, D = Xc.shape
    k = min(k, F, D)
    Xc = np.nan_to_num(Xc)
    try:
        if F * D <= 4_000_000:
            return np.linalg.svd(Xc.astype(np.float64), full_matrices=False)[2][:k].astype(np.float32)
        k2 = min(k + 12, F, D)
        Y = Xc @ np.random.default_rng(0).standard_normal((D, k2)).astype(np.float32)
        for _ in range(3):
            Y, _r = np.linalg.qr(Y)
            Y = Xc @ (Xc.T @ Y)
        Q, _r = np.linalg.qr(Y)
        return np.linalg.svd((Q.T @ Xc).astype(np.float64), full_matrices=False)[2][:k].astype(np.float32)
    except np.linalg.LinAlgError:
        # LAPACK's divide-and-conquer SVD can fail to converge on degenerate data: the eigenvectors of the smaller Gram
        # matrix instead
        if D <= F:
            w, E = np.linalg.eigh((Xc.T @ Xc).astype(np.float64))
            return E[:, np.argsort(w)[::-1][:k]].T.astype(np.float32)
        w, U = np.linalg.eigh((Xc @ Xc.T).astype(np.float64))
        order = np.argsort(w)[::-1][:k]
        w, U = w[order], U[:, order]
        keep = w > max(w[0], 1e-30) * 1e-14
        return ((U[:, keep].T @ Xc) / np.sqrt(w[keep])[:, None]).astype(np.float32)


def _span_error(S, shown, bounds, k):
    """Largest miss of k principal shapes fitted per span (frames bounds[j]..bounds[j+1]), on a vertex sample."""
    F, V, _ = S.shape
    X = S[:, np.random.default_rng(0).choice(V, min(V, 1000), replace=False)].reshape(F, -1).astype(np.float64)
    err = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = np.zeros(F, bool)
        m[a:b + 1] = True
        m &= shown
        if m.sum() < 2:
            continue
        Xc = (X[m] - X[m].mean(0)).astype(np.float32)
        B = _top_basis(Xc, k)
        err = max(err, float(np.abs(Xc - (Xc @ B.T) @ B).max()))
    return err


def _fit_spans(o, frames, S, basis, me, vis, force=False, keyed=False, vk=VK_MAX):
    """_fit, or for long, richly deforming meshes (mop yarn, straps, a canal's water) the film split into
    spans of frames, each a copy of o with its own fit, shown over its span: what moves over minutes is far from a few
    shapes, over seconds much closer. The number of spans is the one that fits closest within the same budget of
    vertices x targets; neighboring spans share their boundary frame, where one hands over to the next. vis: the
    frames o shows; keyed: its visibility already comes from delta_scale keys (_key_visibility), not its own."""
    shown = np.asarray(vis, bool)
    F, V = S.shape[0], S.shape[1]
    if not shown.any() or float(np.abs(S[shown] - S[shown][0]).max()) <= SPAN_TOL or F < 96:
        ok = _fit(o, frames, S, basis, me, force=force, shown=shown, kcap=max(1, min(MORPH_MAX, vk // V)))
        if ok and keyed:
            _key_visibility(o, frames, shown)
        return ok
    best = (np.inf, 1, None)
    for n in (1, 2, 4, 8, 16, 32):
        k = min(MORPH_MAX, vk // (V * n))
        if k < 4 or F // n < 24:
            break
        e = _span_error(S, shown, [round(j * (F - 1) / n) for j in range(n + 1)], k)
        if e < best[0] * 0.8:                       # more spans only for a clear gain
            best = (e, n, k)
        if e <= SPAN_TOL:
            break
    n = best[1]
    if n == 1:
        ok = _fit(o, frames, S, basis, me, force=force, shown=shown, kcap=max(1, min(MORPH_MAX, vk // V)))
        if ok and keyed:
            _key_visibility(o, frames, shown)
        return ok
    bounds = [round(j * (F - 1) / n) for j in range(n + 1)]
    xs = [o] + [_twin(o, f"{o.name}~s{j}") for j in range(1, n)]
    mes = [me] + [me.copy() for _ in range(1, n)]
    idx = np.arange(F)
    for j, (x, m, a, b) in enumerate(zip(xs, mes, bounds[:-1], bounds[1:])):
        Sj = S.copy()
        Sj[:a], Sj[b + 1:] = S[a], S[b]
        _fit(x, frames, Sj, basis, m, force=True, shown=shown & (idx >= a) & (idx <= b), kcap=best[2])
        _key_visibility(x, frames, shown & (idx >= a) & ((idx <= b) if j == n - 1 else (idx < b)))
    print("morphs:", o.name, f"in {n} spans of {F // n} frames", flush=True)
    return True


def _fit(o, frames, S, basis, me, force=False, shown=None, kcap=None):
    """Give o mesh me and morph targets following shapes S (frames x verts x 3); pieces that move as a whole get that
    as transform keys and only the residual as morphs. The morphs are whichever fits closer for the same budget
    (FIT_TOL, MORPH_MAX, VK_MAX): principal shapes mixed by per-frame weights (_pca), or the frames needed for linear
    interpolation crossfaded in turn (_pick). False, leaving o alone, when S doesn't change (unless force)."""
    if len(me.vertices) != S.shape[1]:
        print("morphs: vertex mismatch, skipped", o.name, flush=True)
        return False
    if not force and float(np.abs(S - S[0]).max()) <= TOL:
        return False
    # Carried pieces move as a whole: animate that on the object and keep only the residual as morphs.
    Ms = [_rigid(S[0], s) for s in S]
    moved = max(float(np.abs(M[:3, 3]).max()) for M in Ms) > 0.05
    if moved:
        S = np.stack([(s - M[:3, 3]) @ M[:3, :3] for s, M in zip(S, Ms)])
    shown = np.ones(len(S), bool) if shown is None or not np.any(shown) else np.asarray(shown)
    kcap = kcap or max(1, min(MORPH_MAX, VK_MAX // S.shape[1]))
    tol = FIT_TOL if S.shape[1] < BIG else FIT_TOL_BIG
    how, err, n = "still", 0.0, 0
    if float(np.abs(S[shown] - S[shown][0]).max()) > tol:
        keys, kerr = _pick(S, tol, kcap + 1)
        how, err, n = "keyframes", kerr, len(keys) - 1
        if kerr > tol or len(keys) - 1 > 4:
            mu, P, W, perr = _pca(S, shown, kcap, tol)
            if perr < kerr and not (kerr <= tol and len(keys) - 1 <= len(P)):
                how, err, n = "principal", perr, len(P)
    skinned = _use_mesh(o, me, mu if how == "principal" else S[0])
    if moved:
        _key_transforms(o, frames, basis, Ms)
    if how == "keyframes":
        o.shape_key_add(name="Basis", from_mix=False)
        kbs = []
        for k in keys[1:]:
            kb = o.shape_key_add(name="f%d" % frames[k], from_mix=False)
            kb.data.foreach_set("co", S[k].ravel())
            kbs.append(kb)
        # weight of key j: 1 at its frame, 0 at neighboring keys (key 0 is the basis itself)
        kf = [frames[k] for k in keys]
        for j, kb in enumerate(kbs, start=1):
            for i, f in enumerate(kf):
                kb.value = 1.0 if i == j else 0.0
                kb.keyframe_insert("value", frame=f)
        act = me.shape_keys.animation_data.action
        for fc in getattr(act, "fcurves", []):
            for p in fc.keyframe_points:
                p.interpolation = "LINEAR"
    elif how == "principal":
        # targets scaled so each weight spans [-1, 1] (shape key values are clamped to their slider range)
        span = np.maximum(np.abs(W).max(0), 1e-12)
        o.shape_key_add(name="Basis", from_mix=False)
        fr = np.asarray(frames, np.float32)
        for k in range(n):
            kb = o.shape_key_add(name="pc%d" % k, from_mix=False)
            kb.data.foreach_set("co", (mu + span[k] * P[k]).astype(np.float32).ravel())
            kb.slider_min, kb.slider_max = -1.0, 1.0
            kb.value = float(W[0, k] / span[k])
            kb.keyframe_insert("value", frame=frames[0])
            fc = next(c for c in _curves(me.shape_keys.animation_data.action,
                                         getattr(me.shape_keys.animation_data, 'action_slot', None))
                      if c.data_path == kb.path_from_id("value"))
            fc.keyframe_points.add(len(frames) - 1)
            co = np.empty(2 * len(frames), np.float32)
            co[0::2], co[1::2] = fr, (W[:, k] / span[k]).astype(np.float32)
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * len(frames))      # LINEAR
            fc.update()
    print("morphs:", o.name, "verts", S.shape[1], how, n, f"within {err * 1000:.2f} mm" if n else "",
          "rigid" if moved else "", "unskinned" if skinned else "", flush=True)
    return True


if __name__ == '__main__' and '--sample' in sys.argv:
    # worker of _sample_parallel: blender -b <state.blend> --python morphs.py -- --sample <chunk.json> <out.npz>
    args = sys.argv[sys.argv.index('--sample') + 1:]
    spec = json.load(open(args[0]))
    sc = bpy.context.scene
    cands = [bpy.data.objects[n] for n in spec['names']]
    got = _sample(sc, cands, spec['frames'], {n: KeyMix(bpy.data.objects[n]) for n in spec.get('mixes', ())})
    arrays = {}
    for k, o in enumerate(cands):
        # frames of different vertex counts (topology changes): concatenated, with the counts to split them again
        arrays[f'c{k}'] = np.concatenate(got.co[o.name]).astype(np.float32)
        arrays[f'n{k}'] = np.array([len(c) for c in got.co[o.name]], np.int64)
        arrays[f't{k}'] = np.array([t.encode() for t in got.topo[o.name]], 'S16')
        arrays[f'h{k}'] = np.array(got.hidden[o.name], bool)
        arrays[f'b{k}'] = np.array([[v for row in m for v in row] for m in got.basis[o.name]], np.float64)
    np.savez(args[1], **arrays)
    print('SAMPLE_OK', len(spec['frames']), 'frames', flush=True)
