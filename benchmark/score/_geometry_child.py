"""Blender-interpreter child: the measured penetration timeline of every candidate pair.

    <blender_python> _geometry_child.py request.json raw.json

It runs under the `bpy` interpreter because `animation/blender/collision.py` needs `mathutils`; Coal runs in the
worker that module spawns (`DYCRAFT_COAL_PYTHON`). The output is geometry only: which pairs were candidates, why
the structural ones were set aside, and for the rest a code per frame (`penetration_rules.py`). Counting policy lives
in `penetration_rules.py`, so it can be re-applied to a cached pass.

The contact predicate is the animation layer's own `collision.collision` (Coal BVH, erosion re-check, containment).
What this adds is a **penetration depth** per pair-frame: the smallest translation of one body that `collision()`
calls clear, minimized over up to 18 directions (world axes and both bodies' local axes, each sign) and bisected to
0.1 mm. That is the minimum-translation depth approximated from above by the direction set; it is exact for the
box-on-surface and peg-in-block cases that dominate, and needs no mesh offsetting. Offsetting vertices inward stops
being an erosion once the offset exceeds a feature (a 1 mm bevel): the bevel's vertices then stand proud of the moved
faces and every threshold "passes".
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

GEOMETRY_VERSION = "physical-penetration-geometry-v2"
#: A body whose world box is smaller than this on every axis is treated as hidden (scaling to zero is a common way
#: to hide an object instead of `hide_render`).
MIN_EXTENT_M = 1e-4
#: Relative-transform drift below which two bodies are rigidly coupled, plus float32 storage resolution at the bodies'
#: distance from the origin (matrices are stored as float32: about 4 um of noise at 40 m).
RIGID_TOL = 1e-5
RIGID_TOL_PER_M = 2.4e-7
#: Matrix drift below which a body counts as not moving (float32 storage).
STATIC_TOL = 1e-7
#: Depth bisection: stop at this bracket width, and after this many halvings.
DEPTH_RESOLUTION_M = 1e-4
DEPTH_STEPS = 24
#: Vertices sampled per body for the whole-object enclosure test.
ENCLOSURE_SAMPLES = 48
#: CONTAINED: the pair's first body lies inside its second; CONTAINS: the second inside the first (the pair record's
#: a / b order).
INACTIVE, CLEAR, CROSSING, CONTAINED, ERROR, CONTAINS = -1, 0, 1, 2, 4, 5


def log(*args):
    print("[penetration-geometry]", *args, file=sys.stderr, flush=True)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    repo = Path(request["repo"])
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(repo / "animation" / "blender"), str(repo / "verify")]
    import bpy  # noqa: F401  -- provides mathutils for collision.py
    import collision as C
    from collision_cache import PairCollisionCache
    from evaluation import asset_hash
    from records import check_format, read_record

    t_start = time.perf_counter()
    ev = Path(request["evaluation_dir"])
    index = json.loads((ev / "index.json").read_text())
    check_format(index)
    if not index.get("complete"):
        raise SystemExit(f"evaluation export is incomplete: {ev}")
    frames = list(index["frames"])
    n = len(frames)
    ladder = [float(x) for x in request["ladder_m"]]
    adaptive = request["adaptive"]
    sidecar = json.loads(Path(request["bodies_path"]).read_text()) if request.get("bodies_path") else None
    baseline = json.loads(Path(request["baseline_path"]).read_text()) if request.get("baseline_path") else None

    # ------------------------------------------------------------------ pass 1
    names, name_index, identity, room = [], {}, {}, {}
    hashes, hash_index = [], {}
    digest = hashlib.sha256()
    records_M, records_hid, records_hidden = [], [], []
    for p, frame in enumerate(frames):
        record, raw = read_record(ev, frame)
        digest.update((record.get("signature") or hashlib.sha256(raw).hexdigest()).encode())
        m_row, h_row, hid_row = {}, {}, {}
        for obj in record["objects"]:
            name = obj["name"]
            if name not in name_index:
                name_index[name] = len(names)
                names.append(name)
                identity[name] = obj.get("identity")
                room[name] = bool(obj.get("room"))
            key = obj["mesh"]
            if key not in hash_index:
                hash_index[key] = len(hashes)
                hashes.append(key)
            m_row[name_index[name]] = np.asarray(obj["matrix"], dtype=np.float64)[:3]
            h_row[name_index[name]] = hash_index[key]
            hid_row[name_index[name]] = bool(obj.get("hidden"))
        records_M.append(m_row)
        records_hid.append(h_row)
        records_hidden.append(hid_row)
    N = len(names)
    M = np.zeros((N, n, 3, 4), dtype=np.float64)
    M[:, :, 0, 0] = M[:, :, 1, 1] = M[:, :, 2, 2] = 1.0
    hid = np.full((N, n), -1, dtype=np.int64)
    present = np.zeros((N, n), dtype=bool)
    hidden = np.zeros((N, n), dtype=bool)
    for p in range(n):
        for b, m in records_M[p].items():
            M[b, p] = m
            present[b, p] = True
        for b, h in records_hid[p].items():
            hid[b, p] = h
        for b, h in records_hidden[p].items():
            hidden[b, p] = h
    del records_M, records_hid, records_hidden
    input_digest = digest.hexdigest()
    t_pass1 = time.perf_counter()
    log(f"read {n} frames, {N} bodies, {len(hashes)} unique meshes")

    # --------------------------------------------------------------- mesh info
    mesh_lo = np.zeros((len(hashes), 3))
    mesh_hi = np.zeros((len(hashes), 3))
    mesh_nt = np.zeros(len(hashes), dtype=np.int64)
    tri_digest = []
    arrays = OrderedDict()
    array_bytes = [0]
    ARRAY_BUDGET = 3 * 1024**3

    def load(h):
        key = hashes[h]
        if key in arrays:
            arrays.move_to_end(key)
            return arrays[key]
        with np.load(ev / "meshes" / f"{key}.npz", allow_pickle=False) as data:
            v, t = data["vertices"].copy(), data["triangles"].copy()
        if asset_hash(v, t) != key:
            raise ValueError(f"evaluated mesh content hash mismatch: {key}")
        v = v.astype(np.float64)
        t = t.astype(np.int32)
        size = v.nbytes + t.nbytes
        while arrays and array_bytes[0] + size > ARRAY_BUDGET:
            _, (ov, ot) = arrays.popitem(last=False)
            array_bytes[0] -= ov.nbytes + ot.nbytes
        arrays[key] = (v, t)
        array_bytes[0] += size
        return v, t

    for h in range(len(hashes)):
        v, t = load(h)
        if len(v):
            mesh_lo[h], mesh_hi[h] = v.min(axis=0), v.max(axis=0)
        mesh_nt[h] = len(t)
        tri_digest.append(hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest())

    # ----------------------------------------------------------- body classes
    side_objects = (sidecar or {}).get("objects", {})
    side_hidden = (sidecar or {}).get("hidden_frames", {})
    for name, spans in side_hidden.items():
        if name in name_index:
            b = name_index[name]
            for s, e in spans:
                hidden[b, s:e + 1] = True
    body_class = {}
    for b, name in enumerate(names):
        info = side_objects.get(name, {})
        if info.get("class"):
            body_class[b] = info["class"]
        else:
            seen = {tri_digest[h] for h in hid[b][present[b]]}
            if len(seen) > 1:
                body_class[b] = "topology_changing"
    eligible = np.array([b not in body_class for b in range(N)])

    # World boxes from the local box of each frame's mesh: a conservative broad phase; `collision()` re-tests on
    # vertices.
    lo_l = mesh_lo[np.maximum(hid, 0)]
    hi_l = mesh_hi[np.maximum(hid, 0)]
    center_l = (lo_l + hi_l) / 2
    half_l = (hi_l - lo_l) / 2
    R = M[:, :, :, :3]
    center_w = np.einsum("bpij,bpj->bpi", R, center_l) + M[:, :, :, 3]
    half_w = np.einsum("bpij,bpj->bpi", np.abs(R), half_l)
    LO, HI = center_w - half_w, center_w + half_w
    nonempty = mesh_nt[np.maximum(hid, 0)] > 0
    degenerate = (HI - LO).max(axis=2) < MIN_EXTENT_M
    active = present & ~hidden & nonempty & ~degenerate & eligible[:, None]

    moving = np.zeros(N, dtype=bool)
    for b in range(N):
        idx = np.flatnonzero(active[b])
        if len(idx) < 2:
            continue
        if np.abs(M[b, idx] - M[b, idx[0]]).max() > STATIC_TOL or len(set(hid[b, idx].tolist())) > 1:
            moving[b] = True

    # ---------------------------------------------------- assemblies, adjacency
    parents = (sidecar or {}).get("parents", {})
    types = (sidecar or {}).get("types", {})

    def assembly(name):
        if identity[name]:
            return identity[name].partition("/")[0]
        node = name
        seen = set()
        while parents.get(node) and node not in seen:
            seen.add(node)
            node = parents[node]
        return node

    labels = [identity[x] or x for x in names]
    assemblies = [assembly(x) for x in names]
    adjacency = set()
    if baseline:
        by_link = defaultdict(list)
        for ident, rec in baseline.get("identities", {}).items():
            by_link[rec.get("link")].append(ident)
        for joint in baseline.get("joints", {}).values():
            for a in by_link.get(joint.get("parent"), ()):
                for c in by_link.get(joint.get("child"), ()):
                    adjacency.add(frozenset((a, c)))
    if parents:
        mesh_names = set(names)
        node_of = {x: (parents.get(x) if parents.get(x) and types.get(parents.get(x)) != "MESH" else x)
                   for x in names}
        link_nodes = set(node_of.values())

        def parent_link(node):
            cur, seen = parents.get(node), set()
            while cur and cur not in seen:
                if cur in link_nodes:
                    return cur
                seen.add(cur)
                cur = parents.get(cur)
            return None
        members = defaultdict(list)
        for x in names:
            members[node_of[x]].append(x)
        for x in names:
            p = parents.get(x)
            if p in mesh_names:
                adjacency.add(frozenset((x, p)))
        for node, xs in members.items():
            up = parent_link(node)
            if up is None:
                continue
            for x in xs:
                for y in members.get(up, ()):
                    if x != y:
                        adjacency.add(frozenset((x, y)))

    # --------------------------------------------------------- candidate pairs
    candidate = defaultdict(list)
    TOL = C.TOLERANCE
    for p in range(n):
        act = np.flatnonzero(active[:, p])
        mov = act[moving[act]]
        if not len(mov):
            continue
        lo_a, hi_a = LO[act, p], HI[act, p]
        for i in mov:
            ov = np.all((np.minimum(HI[i, p], hi_a) - np.maximum(LO[i, p], lo_a)) >= -TOL, axis=1)
            for j in act[ov]:
                if j == i or (moving[j] and j < i):
                    continue  # moving-moving pairs are visited once, from the lower index
                a, c = (i, j) if i < j else (j, i)
                candidate[(int(a), int(c))].append(p)
    static_pairs_first = 0
    p0 = 0
    act0 = np.flatnonzero(active[:, p0] & ~moving)
    for start in range(0, len(act0), 512):
        chunk = act0[start:start + 512]
        ov = np.all((np.minimum(HI[chunk, p0][:, None], HI[act0, p0][None]) -
                     np.maximum(LO[chunk, p0][:, None], LO[act0, p0][None])) > TOL, axis=2)
        static_pairs_first += int(sum(np.count_nonzero(ov[k, :start + k]) for k in range(len(chunk))))

    exemptions = {k: {"pairs": 0, "aabb_pair_frames": 0, "examples": []}
                  for k in ("rigid", "adjacent_link")}
    narrow = []
    for (i, j), ps in sorted(candidate.items()):
        covis = np.flatnonzero(active[i] & active[j])
        reason = None
        if (len(set(hid[i, covis].tolist())) == 1 and len(set(hid[j, covis].tolist())) == 1):
            A = np.zeros((len(covis), 4, 4)); A[:, :3] = M[i, covis]; A[:, 3, 3] = 1
            B = np.zeros((len(covis), 4, 4)); B[:, :3] = M[j, covis]; B[:, 3, 3] = 1
            # A body flattened by a zero scale on one axis has no inverse; such a pair is measured rather than
            # declared rigidly coupled.
            invertible = np.abs(np.linalg.det(A[:, :3, :3])) > 1e-12
            if invertible.all():
                rel = np.linalg.solve(A, B)
                reach = float(max(np.abs(M[i, covis, :, 3]).max(), np.abs(M[j, covis, :, 3]).max()))
                if np.abs(rel - rel[0]).max() <= RIGID_TOL + RIGID_TOL_PER_M * reach:
                    reason = "rigid"
        pair_labels = frozenset((labels[i], labels[j]))
        if reason is None and (pair_labels in adjacency or frozenset((names[i], names[j])) in adjacency):
            reason = "adjacent_link"
        if reason:
            e = exemptions[reason]
            e["pairs"] += 1
            e["aabb_pair_frames"] += len(ps)
            if len(e["examples"]) < 12:
                e["examples"].append([labels[i], labels[j]])
            continue
        narrow.append((i, j, ps))
    t_candidates = time.perf_counter()
    log(f"{len(candidate)} candidate pairs; {len(narrow)} need narrow phase; "
        f"exempt: " + ", ".join(f"{k}={v['pairs']}" for k, v in exemptions.items()))

    # ------------------------------------------------------------ narrow phase
    geometry = C.GeometryCache(max_bytes=2 * 1024**3)
    pair_cache = PairCollisionCache(None, method=C.VERSION)
    depth_cache = OrderedDict()
    thin_cache = {}
    errors = []
    queries = Counter()
    axes = np.concatenate([np.eye(3), -np.eye(3)])

    def placed(b, p):
        v, t = load(int(hid[b, p]))
        matrix = np.eye(4)
        matrix[:3] = M[b, p]
        return geometry.placed(v, t, matrix, key=hashes[int(hid[b, p])])

    def thin(body):
        key = body.local.key
        if key not in thin_cache:
            ext = body.local.hi - body.local.lo
            ext = ext[ext > 0]
            thin_cache[key] = float(ext.min()) if len(ext) else 0.0
        return thin_cache[key]

    def directions(A, B):
        dirs = [axes]
        for body in (A, B):
            r = body.matrix[:3, :3]
            cols = r / np.maximum(np.linalg.norm(r, axis=0), 1e-300)
            dirs.append(np.concatenate([cols.T, -cols.T]))
        U = np.concatenate(dirs)
        keep = []
        for u in U:
            if all(float(u @ w) < 0.999 for w in keep):
                keep.append(u)
        return np.array(keep)

    def separation_along(A, B, u, upper):
        """Smallest translation of B along u that `collision()` calls clear."""
        lo, hi = 0.0, upper + 2 * C.TOLERANCE
        for _ in range(DEPTH_STEPS):
            if hi - lo <= DEPTH_RESOLUTION_M:
                break
            mid = (lo + hi) / 2
            queries["depth_collision"] += 1
            if C.collision(A, B.translated(u * mid)) is None:
                hi = mid
            else:
                lo = mid
        return hi

    def penetration_depth(A, B, pk):
        """min over directions of the separating translation (an MTD upper bound).

        A direction's projection interval is only an *upper* bound on its own separation, so it orders the search
        but never rules a direction out: a direction with a large bound may still separate the pair by a hair (a
        finger resting just under a panel's edge has a large bound along the panel normal and a tiny true
        separation). Every direction is therefore asked one question, "is it already clear at the best shift found
        so far?", and bisected only if it is.
        """
        encoded = pair_cache.encode(pk)
        if encoded in depth_cache:
            depth_cache.move_to_end(encoded)
            return depth_cache[encoded]
        U = directions(A, B)
        bound = (A.vertices @ U.T).max(axis=0) - (B.vertices @ U.T).min(axis=0)
        best = math.inf
        for k in np.argsort(bound, kind="stable"):
            upper = float(bound[k])
            if upper >= best:
                queries["depth_collision"] += 1
                if C.collision(A, B.translated(U[k] * best)) is not None:
                    continue           # still colliding at `best`: cannot improve on it
                upper = best
            best = min(best, separation_along(A, B, U[k], upper))
        depth_cache[encoded] = best
        if len(depth_cache) > 200000:
            depth_cache.popitem(last=False)
        return best

    def measure(A, B):
        pk = C.pair_key(A, B)
        hit, cached = pair_cache.get(pk)
        if not cached:
            queries["contact"] += 1
            hit = C.collision(A, B)
            pair_cache.put(pk, hit)
        if hit is None:
            return (CLEAR, 0, False, 0.0)
        depth = penetration_depth(A, B, pk)
        level = sum(1 for rung in ladder if depth >= rung - 1e-9)
        t_min = min(thin(A), thin(B))
        tau = max(adaptive["floor_m"], min(adaptive["cap_m"], adaptive["fraction"] * t_min))
        if hit["kind"] == "containment":
            a_in_b = bool(np.all(A.lo >= B.lo - C.TOLERANCE) and np.all(A.hi <= B.hi + C.TOLERANCE))
            kind = CONTAINED if a_in_b else CONTAINS
        else:
            kind = CROSSING
        return (kind, level, depth >= tau - 1e-9, round(depth * 1000, 2))

    by_frame = defaultdict(list)
    codes = {}
    for k, (i, j, ps) in enumerate(narrow):
        covis = active[i] & active[j]
        codes[k] = [(CLEAR, 0, False, 0.0) if covis[p] else (INACTIVE, 0, False, 0.0) for p in range(n)]
        for p in ps:
            by_frame[p].append(k)
    last_log = time.perf_counter()
    # A chunk of a split pass measures only its own frames, and the parent merges chunks by frame
    # (penetration.merge_chunks). The in-run caches are per chunk.
    lo_f, hi_f = request.get("frame_range") or (0, n)
    lo_f, hi_f = max(0, int(lo_f)), min(n, int(hi_f))
    chunked = request.get("frame_range") is not None
    for p in range(lo_f, hi_f):
        cache_bodies = {}
        for k in by_frame.get(p, ()):
            i, j, _ = narrow[k]
            try:
                A = cache_bodies.setdefault(i, placed(i, p))
                B = cache_bodies.setdefault(j, placed(j, p))
                codes[k][p] = measure(A, B)
            except (ValueError, RuntimeError) as error:
                codes[k][p] = (ERROR, 0, False, 0.0)
                if len(errors) < 200:
                    errors.append({"frame": frames[p], "a": labels[i], "b": labels[j], "error": str(error)[:300]})
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            log(f"frame {p + 1}/{n} (chunk {lo_f}-{hi_f})  contact queries {queries['contact']}  "
                f"depth queries {queries['depth_collision']}" if chunked else
                f"frame {p + 1}/{n}  contact queries {queries['contact']}  depth queries {queries['depth_collision']}")
    t_narrow = time.perf_counter()

    def pair_record(i, j, seq):
        """One pair, ordered by label, with names/assemblies/containment kept aligned."""
        if labels[j] < labels[i]:
            i, j = j, i
            seq = [((CONTAINS if c[0] == CONTAINED else CONTAINED if c[0] == CONTAINS else c[0]),) + tuple(c[1:]) for c in seq]
        return {"a": labels[i], "b": labels[j], "a_name": names[i], "b_name": names[j],
                "assembly_a": assemblies[i], "assembly_b": assemblies[j],
                "relation": "self" if assemblies[i] == assemblies[j] else "cross",
                "timeline": compress(seq)}

    # ------------------------------------------- consumption: whole-object enclosure
    # An agent is an object of several bodies whose parts move relative to each other or deform (a character, an
    # animal, a robot, an articulated machine), decided from the transforms alone. For every other object X that
    # penetrates an agent A, test at those frames whether X is *wholly* inside A: every body of X, every sampled
    # vertex, inside some body of A.
    t_enc0 = time.perf_counter()
    members = defaultdict(list)
    for b in range(N):
        if eligible[b]:
            members[assemblies[b]].append(b)
    agents = set()
    for asm, bs in members.items():
        if len(bs) < 2:
            continue
        if any(len(set(hid[b][present[b]].tolist())) > 1 for b in bs):
            agents.add(asm)
            continue
        r = bs[0]
        for b in bs[1:]:
            covis = np.flatnonzero(active[r] & active[b])
            if len(covis) < 2:
                continue
            A4 = np.zeros((len(covis), 4, 4)); A4[:, :3] = M[r, covis]; A4[:, 3, 3] = 1
            B4 = np.zeros((len(covis), 4, 4)); B4[:, :3] = M[b, covis]; B4[:, 3, 3] = 1
            ok = np.abs(np.linalg.det(A4[:, :3, :3])) > 1e-12
            if ok.sum() < 2:
                continue
            rel = np.linalg.solve(A4[ok], B4[ok])
            if np.abs(rel - rel[0]).max() > RIGID_TOL:
                agents.add(asm)
                break
    contact_frames = defaultdict(set)          # (X, A) -> frames where some X-A pair penetrates
    for k, (i, j, _) in enumerate(narrow):
        for x, a in ((i, j), (j, i)):
            X, A = assemblies[x], assemblies[a]
            if A in agents and X != A:
                for p, c in enumerate(codes[k]):
                    if c[0] in (CROSSING, CONTAINED, CONTAINS):
                        contact_frames[(X, A)].add(p)
    samples_cache = {}

    def samples(b, p):
        key = int(hid[b, p])
        if key not in samples_cache:
            v, _ = load(key)
            step = max(1, len(v) // ENCLOSURE_SAMPLES)
            samples_cache[key] = np.concatenate([v[::step], v.mean(axis=0, keepdims=True)])
        pts = samples_cache[key]
        return pts @ M[b, p, :, :3].T + M[b, p, :, 3]

    enclosures = []
    log(f"enclosure test: {len(contact_frames)} object-agent pairs, "
        f"{sum(len(fs) for fs in contact_frames.values())} contact frames")
    last_log = time.perf_counter()
    for e_i, ((X, A), fs) in enumerate(sorted(contact_frames.items())):
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            log(f"enclosure test {e_i}/{len(contact_frames)}")
        enclosed = []
        for p in sorted(fs):
            xs = [b for b in members[X] if active[b, p]]
            as_ = [b for b in members[A] if active[b, p]]
            ok = bool(xs) and bool(as_)
            for x in xs:
                if not ok:
                    break
                for pt in samples(x, p):
                    hit = False
                    for a in as_:
                        if np.all(pt >= LO[a, p]) and np.all(pt <= HI[a, p]):
                            try:
                                if placed(a, p).inside(pt):
                                    hit = True
                                    break
                            except ValueError:
                                pass
                    if not hit:
                        ok = False
                        break
            if ok:
                enclosed.append(p)
        if enclosed:
            enclosures.append({"object": X, "agent": A,
                               "object_bodies": sorted(names[b] for b in members[X]),
                               "agent_bodies": sorted(names[b] for b in members[A]),
                               "enclosed_frames": _ranges(enclosed),
                               "frames_tested": len(fs)})
    t_enc = time.perf_counter()

    # ---------------------------------------------- never-moving pairs, once
    # Two bodies that never move have one relative pose for the whole clip, so one measurement answers every
    # co-visible frame. They never count toward the rate; they are measured so static overlap in the scene is
    # reported rather than silently skipped.
    static_budget = int(request.get("max_static_pairs", 50000)) if request.get("measure_static", True) else 0
    static_bodies = np.flatnonzero(eligible & ~moving & active.any(axis=1))
    first_active = {int(b): int(np.flatnonzero(active[b])[0]) for b in static_bodies}
    static_candidates = set()
    for p in sorted(set(first_active.values())):
        new = [b for b, q in first_active.items() if q == p]
        pool = np.array([b for b in static_bodies if active[b, p]])
        if not len(pool):
            continue
        for i in new:
            ov = np.all((np.minimum(HI[i, p], HI[pool, p]) - np.maximum(LO[i, p], LO[pool, p])) > TOL, axis=1)
            for j in pool[ov]:
                if j != i:
                    static_candidates.add((min(i, int(j)), max(i, int(j))))
    static_measured, static_truncated, static_out = 0, False, []
    log(f"never-moving pairs: {len(static_candidates)} candidates")
    shard, shards = request.get("static_shard") or (0, 1)
    for s_i, (i, j) in enumerate(sorted(static_candidates)):
        if s_i % shards != shard:
            continue  # another process of a split pass measures this one
        pair_labels = frozenset((labels[i], labels[j]))
        if pair_labels in adjacency or frozenset((names[i], names[j])) in adjacency:
            continue
        if static_measured >= static_budget:
            static_truncated = True
            break
        covis = active[i] & active[j]
        if not covis.any():
            continue
        p = int(np.flatnonzero(covis)[0])
        static_measured += 1
        try:
            code = measure(placed(i, p), placed(j, p))
        except (ValueError, RuntimeError) as error:
            code = (ERROR, 0, False, 0.0)
            if len(errors) < 200:
                errors.append({"frame": frames[p], "a": labels[i], "b": labels[j], "error": str(error)[:300]})
        if code[0] == CLEAR:
            continue
        seq = [code if covis[q] else (INACTIVE, 0, False, 0.0) for q in range(n)]
        static_out.append({**pair_record(i, j, seq), "never_moving": True})
    t_static = time.perf_counter()

    # ------------------------------------------------------------- tunneling
    fast = Counter()
    for b in np.flatnonzero(moving):
        idx = np.flatnonzero(active[b])
        if len(idx) < 2:
            continue
        c = (LO[b, idx] + HI[b, idx]) / 2
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        extent = (HI[b, idx] - LO[b, idx]).min(axis=1)[1:]
        consecutive = np.diff(idx) == 1
        count = int(np.count_nonzero((step > extent) & consecutive))
        if count:
            fast[labels[b]] = count

    # ------------------------------------------------------------------ output
    pairs_out = []
    for k, (i, j, ps) in enumerate(narrow):
        seq = codes[k]
        if not chunked and not any(c[0] in (CROSSING, CONTAINED, CONTAINS, ERROR) for c in seq):
            continue
        pairs_out.append(pair_record(i, j, seq))  # a chunk emits every narrow pair, for the merge
    pairs_out.extend(static_out)
    pairs_out.sort(key=lambda x: (x["a"], x["b"]))
    open_bodies = sorted(labels[b] for b in range(N) if eligible[b] and active[b].any()
                         and not _is_closed(load, int(hid[b][active[b]][0]), C))
    class_counts = Counter(body_class.values())
    result = {
        "geometry_version": GEOMETRY_VERSION,
        "input_digest": input_digest,
        "evaluation_dir": str(ev),
        "fps": index["fps"], "frame_start": index.get("frame_start"),
        "n_frames": n, "frames": frames,
        "ladder_m": ladder, "adaptive": adaptive,
        "bodies": {
            "total": N, "eligible": int(eligible.sum()), "moving": int((moving & eligible).sum()),
            "excluded_by_class": dict(sorted(class_counts.items())),
            "excluded": sorted([[labels[b], c] for b, c in body_class.items()])[:200],
            "hidden_or_inactive_some_frame": int(((~active) & present & eligible[:, None]).any(axis=1).sum()),
            "degenerate_some_frame": int((degenerate & present).any(axis=1).sum()),
            "open_meshes": len(open_bodies), "open_mesh_examples": open_bodies[:40],
            "deforming": int(sum(1 for b in range(N) if eligible[b] and len(set(hid[b][present[b]].tolist())) > 1)),
            "assemblies_with_multiple_bodies": len([a for a, c in Counter(assemblies).items() if c > 1]),
        },
        "candidate_pairs": len(candidate),
        "narrow_pairs": len(narrow),
        "static_aabb_pairs_first_frame": static_pairs_first,
        "never_moving_pairs": {"aabb_candidates": len(static_candidates), "measured": static_measured,
                               "overlapping": len(static_out), "truncated": static_truncated},
        "exemptions": exemptions,
        "adjacency_pairs_known": len(adjacency),
        "errors": errors,
        "tunneling_risk": {"body_steps": int(sum(fast.values())),
                           "top": [[k, v] for k, v in sorted(fast.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]},
        "pairs": pairs_out,
        "agents": sorted(agents),
        "enclosures": enclosures,
        "stats": {**pair_cache.stats(), "coal": C.backend_stats(), "bvh_builds_local": geometry.builds,
                  "contact_queries": queries["contact"], "depth_collision_queries": queries["depth_collision"],
                  "depth_cache_entries": len(depth_cache)},
        "timing": {"read_s": t_pass1 - t_start, "candidates_s": t_candidates - t_pass1,
                   "narrow_phase_s": t_narrow - t_candidates, "enclosure_s": t_enc - t_enc0,
                   "never_moving_pairs_s": t_static - t_enc,
                   "total_s": time.perf_counter() - t_start},
    }
    out_path.write_text(json.dumps(result))
    log(f"done in {result['timing']['total_s']:.1f}s")


def _ranges(frames):
    out = []
    for f in sorted(frames):
        if out and out[-1][1] == f - 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def compress(codes):
    """RLE over (kind, level, tau, depth_mm) codes; see `penetration_rules.expand`."""
    out = []
    for f, (kind, level, tau, depth) in enumerate(codes):
        if (out and out[-1][1] == f - 1 and out[-1][2] == kind and out[-1][3] == level
                and out[-1][4] == bool(tau) and out[-1][5] == depth):
            out[-1][1] = f
        else:
            out.append([f, f, kind, int(level), bool(tau), depth])
    return out


def _is_closed(load, h, C):
    v, t = load(h)
    if not len(t):
        return False
    return all(closed for _, closed in C.Body(v, t).components)


if __name__ == "__main__":
    main()
    # The output is written and closed. Leave without Blender's teardown, which can deadlock after opening a scene
    # with caches or threads (large blends, Alembic caches).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
