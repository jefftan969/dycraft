"""Blender-interpreter child: contact geometry for the floating rate.

    <blender_python> _support_child.py request.json raw.npz

It runs under the `bpy` interpreter because `animation/blender/collision.py` needs `mathutils` (BVH nearest-point
queries); Coal runs in the animation layer's worker (`DYCRAFT_COAL_PYTHON`). The output is geometry only; the policy
is `support_rules.py`:

* per body and frame: world matrix, mesh id, `active` (present, visible, non-empty, not degenerate) and the body
  classes (`classes.py`); fluid and text bodies are measured only against their structural partners;
* per mesh: local box, volume, local centroid (the volume centroid of a closed mesh, else the area-weighted surface
  centroid);
* per candidate pair and frame where the surfaces are within the query distance Q: the Coal surface gap, and `gapU`
  in each direction, the smallest distance of a *qualifying contact sample* (a vertex of one body and its nearest
  point on the other, on a surface no steeper than `slope_deg`, below the resting body's centroid), plus
  "embedded below the centroid" flags for crossing surfaces.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

CHILD_VERSION = "physical-floating-geometry-v1"
MIN_EXTENT_M = 1e-4              # scale-to-zero hiding (same as penetration)
SCALE_TOL = 0.01                 # matrix scale varying by more than 1 % -> non-rigid
ROUND_M = 1e-5                   # stored gap resolution (0.01 mm), for run-length compression
#: a contact sample this close already passes every (size-adaptive) resting
#: tolerance (`support_rules` tau_floor), so embedding cannot change the edge
EMB_SKIP_M = 5e-4


def log(*args):
    print("[floating-geometry]", *args, file=sys.stderr, flush=True)


def mesh_stats(v, t):
    """(volume, centroid, closed) of a triangle mesh in its local frame."""
    if not len(t):
        return 0.0, np.zeros(3), False
    tri = v[t]
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    closed = bool(len(counts) and np.all(counts == 2))
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    if closed:
        vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
        volume = vol6.sum() / 6.0
        if abs(volume) > 1e-12:
            centroid = (vol6[:, None] * tri.sum(axis=1)).sum(axis=0) / (4.0 * vol6.sum())
            return float(abs(volume)), centroid, True
    if area.sum() > 0:
        return 0.0, (area[:, None] * tri.mean(axis=1)).sum(axis=0) / area.sum(), closed
    return 0.0, v.mean(axis=0), closed


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    repo = Path(request["repo"])
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(repo / "animation" / "blender"), str(repo / "verify")]
    import bpy  # noqa: F401  -- mathutils for collision.py
    from mathutils import Vector
    import collision as C
    from evaluation import asset_hash
    from records import check_format, read_record
    import classes as R

    t_start = time.perf_counter()
    ev = Path(request["evaluation_dir"])
    index = json.loads((ev / "index.json").read_text())
    check_format(index)
    if not index.get("complete"):
        raise SystemExit(f"evaluation export is incomplete: {ev}")
    frames = list(index["frames"])
    n = len(frames)
    Q = float(request["query_m"])
    slope_cos = float(np.cos(np.radians(request["slope_deg"])))
    max_samples = int(request.get("max_samples", 2000))
    #: contact samples only matter where a resting / mounted contact can latch
    #: (largest swept tau_p x exit factor); farther pairs keep only their gap.
    S = min(Q, float(request.get("sample_m", Q)))
    clock = Counter()
    sidecar = json.loads(Path(request["bodies_path"]).read_text()) if request.get("bodies_path") else {}

    # ------------------------------------------------------------ records
    names, where, identity, room = [], {}, {}, {}
    hashes, hash_index = [], {}
    digest = hashlib.sha256()
    rows = []
    for frame in frames:
        record, raw = read_record(ev, frame)
        digest.update((record.get("signature") or hashlib.sha256(raw).hexdigest()).encode())
        row = {}
        for obj in record["objects"]:
            name = obj["name"]
            if name not in where:
                where[name] = len(names); names.append(name)
                identity[name] = obj.get("identity"); room[name] = bool(obj.get("room"))
            if obj["mesh"] not in hash_index:
                hash_index[obj["mesh"]] = len(hashes); hashes.append(obj["mesh"])
            row[where[name]] = (obj["matrix"], hash_index[obj["mesh"]], bool(obj.get("hidden")))
        rows.append(row)
    N = len(names)
    M = np.zeros((N, n, 3, 4), dtype=np.float64)
    M[:, :, 0, 0] = M[:, :, 1, 1] = M[:, :, 2, 2] = 1.0
    hid = np.full((N, n), -1, dtype=np.int64)
    hidden = np.zeros((N, n), dtype=bool)
    for p, row in enumerate(rows):
        for b, (m, h, hd) in row.items():
            M[b, p] = np.asarray(m, dtype=np.float64)[:3]
            hid[b, p] = h
            hidden[b, p] = hd
    del rows
    present = hid >= 0
    for name, spans in (sidecar.get("hidden_frames") or {}).items():
        if name in where:
            for s, e in spans:
                hidden[where[name], s:e + 1] = True
    t_read = time.perf_counter()
    log(f"read {n} frames, {N} bodies, {len(hashes)} unique meshes")

    # ------------------------------------------------------------- meshes
    H = len(hashes)
    mesh_lo, mesh_hi = np.zeros((H, 3)), np.zeros((H, 3))
    mesh_nt = np.zeros(H, dtype=np.int64)
    mesh_volume, mesh_centroid = np.zeros(H), np.zeros((H, 3))
    mesh_closed = np.zeros(H, dtype=bool)
    tri_digest = []
    arrays = OrderedDict()
    budget = [0]

    def load(h):
        key = hashes[h]
        if key in arrays:
            arrays.move_to_end(key)
            return arrays[key]
        with np.load(ev / "meshes" / f"{key}.npz", allow_pickle=False) as data:
            v, t = data["vertices"].copy(), data["triangles"].copy()
        if asset_hash(v, t) != key:
            raise ValueError(f"evaluated mesh content hash mismatch: {key}")
        v, t = v.astype(np.float64), t.astype(np.int32)
        while arrays and budget[0] + v.nbytes + t.nbytes > 2 * 1024**3:
            _, (ov, ot) = arrays.popitem(last=False)
            budget[0] -= ov.nbytes + ot.nbytes
        arrays[key] = (v, t)
        budget[0] += v.nbytes + t.nbytes
        return v, t

    for h in range(H):
        v, t = load(h)
        if len(v):
            mesh_lo[h], mesh_hi[h] = v.min(axis=0), v.max(axis=0)
        mesh_nt[h] = len(t)
        mesh_volume[h], mesh_centroid[h], mesh_closed[h] = mesh_stats(v, t)
        tri_digest.append(hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest())

    # world boxes, active mask
    hh = np.maximum(hid, 0)
    lo_l, hi_l = mesh_lo[hh], mesh_hi[hh]
    Rm = M[:, :, :, :3]
    center = np.einsum("bpij,bpj->bpi", Rm, (lo_l + hi_l) / 2) + M[:, :, :, 3]
    half = np.einsum("bpij,bpj->bpi", np.abs(Rm), (hi_l - lo_l) / 2)
    LO, HI = center - half, center + half
    nonempty = mesh_nt[hh] > 0
    degenerate = (HI - LO).max(axis=2) < MIN_EXTENT_M
    active = present & ~hidden & nonempty & ~degenerate

    deforming, scale_changing = [], []
    for b in range(N):
        idx = np.flatnonzero(active[b])
        deforming.append(len({hid[b, i] for i in idx}) > 1)
        if len(idx) > 1:
            s = np.linalg.norm(M[b, idx, :, :3], axis=1)          # (k, 3) column norms
            ratio = s.max(axis=0) / np.maximum(s.min(axis=0), 1e-12)
            scale_changing.append(bool((ratio > 1 + SCALE_TOL).any()))
        else:
            scale_changing.append(False)
    side_classes = {k: v.get("class") for k, v in (sidecar.get("objects") or {}).items() if v.get("class")}
    identities = [identity[x] for x in names]
    classes, reasons = R.seed_classes(names, identities, sidecar_classes=side_classes, deforming=deforming,
                                      scale_changing=scale_changing, task_entry=request.get("task_entry"))
    # Every class is a graph node; fluid and text bodies only through delivered structure (a part misnamed as fluid),
    # never by contact.
    graph = np.ones(len(classes), dtype=bool)
    contact_ok = np.array([c in R.CONTACT_CLASSES for c in classes], dtype=bool)
    # Delivered structure: a structural pair needs its gap only (a structural edge with contact already supports
    # both ways).
    import structure as ST
    struct_pairs, _ = ST.pairs(names, identities, baseline_path=request.get("baseline_path"),
                               sidecar_path=request.get("bodies_path"))
    struct_set = {(min(a, b), max(a, b)) for a, b, _ in struct_pairs}

    # -------------------------------------------------------- pair geometry
    geometry = C.GeometryCache(max_bytes=2 * 1024**3)
    gap_cache, sample_cache = OrderedDict(), OrderedDict()
    queries = Counter()
    pair_ids, runs_out = {}, []          # (i,j) -> k ;  [k, start, end, gap, gab, gba, flags]
    last = {}                            # (i,j) -> (key tuple, value)
    open_run = {}

    def placed(b, p):
        v, t = load(int(hid[b, p]))
        m = np.eye(4); m[:3] = M[b, p]
        return geometry.placed(v, t, m, key=hashes[int(hid[b, p])])

    def quat(R3):
        return tuple(int(round(float(x) / 1e-6)) for x in C.rotation_quaternion(R3))

    def samples(A, B, cA, cB):
        """Qualifying contact distances for A-on-B and B-on-A (inf if none)."""
        started = time.perf_counter()
        best_ab, best_ba = np.inf, np.inf
        for X, Y in ((A, B), (B, A)):
            pts = X.vertices
            lo, hi = Y.lo - S, Y.hi + S
            sel = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
            if not len(sel):
                continue
            Ry, ty = Y.matrix[:3, :3], Y.matrix[:3, 3]
            local = (pts[sel] - ty) @ Ry
            # exact prefilter: distance to Y's local (oriented) box is a lower bound
            # of the distance to its surface. Under the cap, spread the samples
            # evenly: "nearest to the box first" ties at 0 for every vertex inside
            # a large body's box and would keep one arbitrary side of the mesh.
            box_d = np.linalg.norm(np.maximum(np.maximum(Y.local.lo - local, local - Y.local.hi), 0.0), axis=1)
            keep = np.flatnonzero(box_d <= S)
            if not len(keep):
                continue
            if len(keep) > max_samples:
                keep = keep[np.linspace(0, len(keep) - 1, max_samples).astype(np.int64)]
            sel, local = sel[keep], local[keep]
            tree = Y.local.tree
            queries["nearest"] += len(sel)
            for k, q in zip(sel, local):
                hit = tree.find_nearest(Vector(q), S)
                if hit[0] is None:
                    continue
                d = float(hit[3])
                if d >= best_ab and d >= best_ba:
                    continue
                nz = abs(float(Ry[2] @ np.asarray(hit[1])))
                if nz < slope_cos:
                    continue
                zc = 0.5 * (pts[k, 2] + float(Ry[2] @ np.asarray(hit[0]) + ty[2]))
                if zc < cA[2] - 1e-6 and d < best_ab:
                    best_ab = d
                if zc < cB[2] - 1e-6 and d < best_ba:
                    best_ba = d
        queries["sample_pairs"] += 1
        clock["samples_s"] += time.perf_counter() - started
        return best_ab, best_ba

    cross_cache = OrderedDict()

    def crossing(A, B, key):
        if key not in cross_cache:
            queries["crossing"] += 1
            t_c = time.perf_counter()
            cross_cache[key] = C.collision(A, B) is not None
            clock["crossing_s"] += time.perf_counter() - t_c
            if len(cross_cache) > 500000:
                cross_cache.popitem(last=False)
        return cross_cache[key]

    def centroid_world(b, p):
        m = M[b, p]
        return m[:, :3] @ mesh_centroid[hid[b, p]] + m[:, 3]

    # Conservative advancement: a pair measured farther than Q at frame pm stays
    # farther than Q while its bodies' combined box-corner displacement since pm
    # is below (gap - Q). Exact skip (never changes a result); no query needed.
    box_corners = np.stack([np.array([[x, y, z] for x in (mesh_lo[h][0], mesh_hi[h][0])
                                      for y in (mesh_lo[h][1], mesh_hi[h][1])
                                      for z in (mesh_lo[h][2], mesh_hi[h][2])]) for h in range(H)]) \
        if H else np.zeros((0, 8, 3))
    far = {}

    def moved(b, p0, p1):
        c = box_corners[hid[b, p1]]
        a = c @ M[b, p0][:, :3].T + M[b, p0][:, 3]
        z = c @ M[b, p1][:, :3].T + M[b, p1][:, 3]
        return float(np.sqrt(((z - a) ** 2).sum(axis=1)).max())

    def close_run(key):
        if key in open_run:
            runs_out.append(open_run.pop(key))

    last_log = time.perf_counter()
    # An optional frame chunk, merged by frame with the others (floating.merge_chunks). The in-run caches are per
    # chunk.
    f0, f1 = request.get("frame_range") or (0, n)
    for p in range(f0, f1):
        act = np.flatnonzero(active[:, p] & graph)
        placed_cache = {}
        seen = set()
        if len(act) > 1:
            lo, hi = LO[act, p] - Q, HI[act, p] + Q
            for ii in range(len(act) - 1):
                i = act[ii]
                rest = act[ii + 1:]
                ov = np.all((np.minimum(hi[ii], hi[ii + 1:]) - np.maximum(lo[ii], lo[ii + 1:])) >= 0, axis=1)
                for j in rest[ov]:
                    i_, j_ = int(i), int(j)
                    key = (i_, j_)
                    if not (contact_ok[i_] and contact_ok[j_]) and key not in struct_set:
                        continue
                    seen.add(key)
                    state = (M[i_, p].tobytes(), M[j_, p].tobytes(), int(hid[i_, p]), int(hid[j_, p]))
                    prev = last.get(key)
                    bound = far.get(key)
                    if prev is not None and prev[0] == state:
                        value = prev[1]
                    elif (bound is not None and bound[2] == state[2:]
                          and bound[0] - moved(i_, bound[1], p) - moved(j_, bound[1], p) > Q):
                        queries["distance_skipped_far"] += 1
                        value = None
                        last[key] = (state, value)
                    else:
                        for x in (i_, j_):
                            if x not in placed_cache:
                                placed_cache[x] = placed(x, p)
                        A, B = placed_cache[i_], placed_cache[j_]
                        pk = C.pair_key(A, B)
                        if pk in gap_cache:
                            gap = gap_cache[pk]; gap_cache.move_to_end(pk)
                        else:
                            queries["distance"] += 1
                            t_q = time.perf_counter()
                            gap = float(C.surface_gap(A, B)["gap_m"])
                            clock["distance_s"] += time.perf_counter() - t_q
                            gap_cache[pk] = gap
                            if len(gap_cache) > 500000:
                                gap_cache.popitem(last=False)
                        if gap > Q:
                            value = None
                            far[key] = (gap, p, state[2:])
                        else:
                            far.pop(key, None)
                            cA, cB = centroid_world(i_, p), centroid_world(j_, p)
                            sk = (pk, quat(A.matrix[:3, :3]))
                            if gap > S or key in struct_set:
                                gab, gba = np.inf, np.inf
                            elif sk in sample_cache:
                                gab, gba = sample_cache[sk]
                            else:
                                gab, gba = samples(A, B, cA, cB)
                                sample_cache[sk] = (gab, gba)
                                if len(sample_cache) > 500000:
                                    sample_cache.popitem(last=False)
                            flags = 0
                            if gap <= 1e-6 and key not in struct_set:
                                # Surfaces really cross (in collision(), touching is not crossing): embedded
                                # support if the overlap reaches below the resting body's centroid (a cup sunk
                                # into a table, a disc on an axle through its center). The costly crossing test
                                # runs only where it can change a passive edge: not for structural pairs (gap 0
                                # already gives the structural edge), not where a qualifying contact sample is
                                # already within the smallest tolerance.
                                bottom = max(LO[i_, p, 2], LO[j_, p, 2])
                                want_ab = bottom < cA[2] and gab > EMB_SKIP_M
                                want_ba = bottom < cB[2] and gba > EMB_SKIP_M
                                if (want_ab or want_ba) and crossing(A, B, sk):
                                    if bottom < cA[2]:
                                        flags |= 1
                                    if bottom < cB[2]:
                                        flags |= 2
                            r = lambda x: float(round(x / ROUND_M) * ROUND_M) if np.isfinite(x) else float("inf")
                            value = (r(gap), r(gab), r(gba), flags)
                        last[key] = (state, value)
                    if value is None:
                        close_run(key)
                        continue
                    k = pair_ids.setdefault(key, len(pair_ids))
                    run = open_run.get(key)
                    if run is not None and run[2] == p - 1 and tuple(run[3:]) == value:
                        run[2] = p
                    else:
                        close_run(key)
                        open_run[key] = [k, p, p, *value]
        for key in [k for k in open_run if k not in seen]:
            close_run(key)
        for key in [k for k in last if k not in seen]:
            del last[key]
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            log(f"frame {p + 1}/{n}  distance {queries['distance']}  sample pairs {queries['sample_pairs']}  "
                f"pairs {len(pair_ids)}")
    for key in list(open_run):
        close_run(key)
    t_pairs = time.perf_counter()

    pairs = np.zeros((len(pair_ids), 2), dtype=np.int32)
    for (i, j), k in pair_ids.items():
        pairs[k] = (i, j)
    runs = np.array([r[:3] for r in runs_out], dtype=np.int32).reshape(-1, 3)
    vals = np.array([r[3:6] for r in runs_out], dtype=np.float32).reshape(-1, 3)
    flags = np.array([r[6] for r in runs_out], dtype=np.uint8)
    meta = {
        "child_version": CHILD_VERSION, "input_digest": digest.hexdigest(), "evaluation_dir": str(ev),
        "fps": index["fps"], "frame_start": index.get("frame_start"), "n_frames": n, "frames": frames,
        "names": names, "identities": identities, "room": [room[x] for x in names],
        "classes": classes, "class_reasons": reasons,
        "deforming": [names[b] for b in range(N) if deforming[b]],
        "scale_changing": [names[b] for b in range(N) if scale_changing[b]],
        "open_meshes": int(sum(1 for b in range(N) if active[b].any() and not mesh_closed[hid[b][active[b]][0]])),
        "query_m": Q, "sample_m": S, "slope_deg": request["slope_deg"], "max_samples": max_samples,
        "stats": {"distance_queries": queries["distance"], "sample_pair_queries": queries["sample_pairs"],
                  "nearest_point_queries": queries["nearest"], "crossing_queries": queries["crossing"],
                  "distance_skipped_far": queries["distance_skipped_far"],
                  "structural_pairs": len(struct_set),
                  "distance_s": round(clock["distance_s"], 2), "samples_s": round(clock["samples_s"], 2),
                  "crossing_s": round(clock["crossing_s"], 2),
                  "pairs": len(pair_ids), "runs": len(runs_out), "bvh_builds": geometry.builds,
                  "coal": C.backend_stats()},
        "frame_range": [f0, f1],
        "timing": {"read_s": t_read - t_start, "pairs_s": t_pairs - t_read,
                   "total_s": time.perf_counter() - t_start},
    }
    np.savez_compressed(out_path, meta=np.array(json.dumps(meta)), M=M.astype(np.float32), hid=hid.astype(np.int32),
                        active=active, mesh_lo=mesh_lo, mesh_hi=mesh_hi, mesh_centroid=mesh_centroid,
                        mesh_volume=mesh_volume, mesh_closed=mesh_closed, pairs=pairs, runs=runs, vals=vals,
                        flags=flags)
    log(f"done in {meta['timing']['total_s']:.1f}s: {len(pair_ids)} pairs, {queries['distance']} distance "
        f"queries, {queries['sample_pairs']} sample passes")


if __name__ == "__main__":
    main()
