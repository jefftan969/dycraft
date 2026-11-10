"""Per-body MOVING / STATIC state per frame, from the evaluated transforms alone.

A penetrating pair counts at frame t only if one of its bodies is MOVING at t. The state comes from each body's
per-frame world matrix and mesh digest only, with no agent declaration, method identity, or task knowledge, so every
arm and any other film converted to `dycraft.evaluation` is classified the same way.

**Change.** For each transition t-1 -> t, the largest displacement of the corners of the body's local box:

    change(t) = max_c | M_t c - M_{t-1} c |

One number covers translation, rotation (a corner at distance r from the origin moves about r * angle) and scale. A
change of mesh digest (deformation) is always a change. A change counts only above the float32 resolution of the
stored matrices: `NOISE_ULPS` ulps of the corner's world coordinates, floored at `EPSILON_M`.

**States**, with K = round(0.5 s * fps) frames (at least 1):

* changed(t)   change(t) > tolerance(t); frame 0 uses its forward transition (0 -> 1).
* recent(t)    some changed(s) for s in (t - K, t].
* settled(t)   no changed(s) for s in (t, t + K]. Near the clip's end, where fewer than K future transitions exist,
               the window is completed with the most recent past ones, so a body is never called settled on less
               than 0.5 s of evidence unless the clip is shorter than that.

    MOVING(t) = changed(t) or (recent(t) and not settled(t))

So a pause shorter than 0.5 s inside a motion stays MOVING, a body becomes STATIC at the first frame after its last
change from which it stays unchanged for 0.5 s, and a body at rest before it first moves is STATIC until then.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .records import check_format, read_record

MOTION_VERSION = "physical-motion-v1"
DURATION_S = 0.5
#: Exports hold `matrix_world` as float32, so re-evaluating an unchanged pose can move a corner by a few ulps of its
#: world coordinates (about 1.2e-7 m at 1 m from the origin, 1e-6 m at 10 m). Anything above that resolution is
#: motion: strict about STATIC, permissive about MOVING.
EPSILON_M = 1e-6
NOISE_ULPS = 8


def frames_for(duration_s: float, fps: float) -> int:
    """0.5 s in frames: the nearest integer, at least one (24 fps -> 12, 30 -> 15)."""
    return max(1, int(round(duration_s * fps)))


def read_transforms(evaluation_dir):
    """(index, names, matrices (N, n, 4, 4), mesh ids (N, n; -1 absent), local box per mesh id)."""
    ev = Path(evaluation_dir)
    index = json.loads((ev / "index.json").read_text())
    check_format(index)
    frames = list(index["frames"])
    names, where, hashes, hid_of = [], {}, [], {}
    rows = []
    for frame in frames:
        record, _ = read_record(ev, frame)
        row = {}
        for obj in record["objects"]:
            if obj["name"] not in where:
                where[obj["name"]] = len(names)
                names.append(obj["name"])
            if obj["mesh"] not in hid_of:
                hid_of[obj["mesh"]] = len(hashes)
                hashes.append(obj["mesh"])
            row[where[obj["name"]]] = (obj["matrix"], hid_of[obj["mesh"]])
        rows.append(row)
    N, n = len(names), len(frames)
    M = np.tile(np.eye(4), (N, n, 1, 1))
    hid = np.full((N, n), -1, dtype=np.int64)
    for p, row in enumerate(rows):
        for b, (matrix, h) in row.items():
            M[b, p] = matrix
            hid[b, p] = h
    boxes = np.zeros((len(hashes), 2, 3))
    for h, key in enumerate(hashes):
        with np.load(ev / "meshes" / f"{key}.npz", allow_pickle=False) as data:
            v = data["vertices"]
        if len(v):
            boxes[h] = v.min(axis=0), v.max(axis=0)
    return index, names, M, hid, boxes


def corners(box):
    lo, hi = box
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def transition_changes(M, hid, boxes):
    """(change, tolerance), each (N, n), index 0 unused.

    change[b, t] is the largest corner displacement from t-1 to t (inf for a mesh change, 0 where either frame is
    absent); tolerance[b, t] is the float32 noise bound for that transition.
    """
    N, n = hid.shape
    change = np.zeros((N, n))
    tolerance = np.full((N, n), EPSILON_M)
    if n < 2:
        return change, tolerance
    box_corners = np.stack([corners(b) for b in boxes]) if len(boxes) else np.zeros((0, 8, 3))
    for b in range(N):
        h = hid[b]
        c = box_corners[np.maximum(h, 0)]                       # (n, 8, 3): each frame's own box
        world = np.einsum("tij,tkj->tki", M[b, :, :3, :3], c) + M[b, :, None, :3, 3]
        prev = np.einsum("tij,tkj->tki", M[b, :-1, :3, :3], c[1:]) + M[b, :-1, None, :3, 3]
        disp = np.linalg.norm(world[1:] - prev, axis=2).max(axis=1)
        disp[h[1:] != h[:-1]] = np.inf
        disp[(h[1:] < 0) | (h[:-1] < 0)] = 0.0
        change[b, 1:] = disp
        reach = np.maximum(np.abs(world[1:]).max(axis=(1, 2)), np.abs(prev).max(axis=(1, 2)))
        ulp = np.spacing(reach.astype(np.float32)).astype(np.float64)
        tolerance[b, 1:] = np.maximum(EPSILON_M, NOISE_ULPS * ulp)
    return change, tolerance


def classify(change_row, K, tolerance_row=None):
    """MOVING mask for one body from its per-transition changes (module docstring)."""
    n = len(change_row)
    tol = np.full(n, EPSILON_M) if tolerance_row is None else np.asarray(tolerance_row)
    changed = np.zeros(n, dtype=bool)
    if n >= 2:
        changed[1:] = change_row[1:] > tol[1:]
        changed[0] = change_row[1] > tol[1]
    moving = np.zeros(n, dtype=bool)
    if not changed.any():
        return moving
    csum = np.concatenate([[0], np.cumsum(changed[1:] if n > 1 else [])])  # csum[k] = changes in (0, k]

    def count(lo, hi):  # changed transitions s with lo < s <= hi, s >= 1
        lo, hi = max(lo, 0), min(hi, n - 1)
        return 0 if hi <= lo else int(csum[hi] - csum[lo])
    for t in range(n):
        if changed[t]:
            moving[t] = True
            continue
        if count(t - K, t) == 0:
            continue
        future = min(K, n - 1 - t)
        if future < K:
            settled = count(t, t + future) == 0 and count(t - (K - future), t) == 0
        else:
            settled = count(t, t + K) == 0
        moving[t] = not settled
    return moving


def intervals(mask):
    out, start = [], None
    for t, v in enumerate(mask):
        if v and start is None:
            start = t
        elif not v and start is not None:
            out.append([start, t - 1])
            start = None
    if start is not None:
        out.append([start, len(mask) - 1])
    return out


def compute(evaluation_dir, duration_s=DURATION_S):
    """MOVING intervals per body name, with the classifier's settings and totals."""
    started = time.perf_counter()
    index, names, M, hid, boxes = read_transforms(evaluation_dir)
    t_read = time.perf_counter()
    fps = float(index["fps"])
    K = frames_for(duration_s, fps)
    change, tolerance = transition_changes(M, hid, boxes)
    moving = {}
    total_moving = 0
    for b, name in enumerate(names):
        mask = classify(change[b], K, tolerance[b])
        total_moving += int(mask.sum())
        if mask.any():
            moving[name] = intervals(mask)
    n = hid.shape[1]
    c, tol = change[:, 1:], tolerance[:, 1:]
    fin = np.isfinite(c)
    below = fin & (c > 0) & (c <= tol)
    above = fin & (c > tol)
    return {
        "motion_version": MOTION_VERSION, "fps": fps, "duration_s": duration_s, "window_frames": K,
        "epsilon_m": EPSILON_M, "noise_ulps": NOISE_ULPS,
        "tolerance_m": {"min": float(tol.min()) if tol.size else EPSILON_M,
                        "max": float(tol.max()) if tol.size else EPSILON_M},
        "n_frames": n, "bodies": len(names),
        "bodies_ever_moving": len(moving),
        "moving_body_frames": total_moving, "static_body_frames": len(names) * n - total_moving,
        "transition_noise": {
            "transitions": int(c.size), "exactly_zero": int((fin & (c == 0)).sum()),
            "nonzero_within_tolerance": int(below.sum()),
            "max_within_tolerance_m": float(c[below].max()) if below.any() else 0.0,
            "above_tolerance": int(above.sum()),
            "min_above_tolerance_m": float(c[above].min()) if above.any() else None,
            "deformation_transitions": int(np.isinf(c).sum())},
        "moving_intervals": moving,
        "timing_s": {"read": t_read - started, "classify": time.perf_counter() - t_read},
    }


def mask_for(motion: dict, name: str, n: int) -> list:
    out = [False] * n
    for s, e in motion["moving_intervals"].get(name, ()):
        for t in range(s, e + 1):
            out[t] = True
    return out


def cache_key(evaluation_dir, duration_s=DURATION_S) -> str:
    ev = Path(evaluation_dir)
    h = hashlib.sha256()
    h.update((ev / "index.json").read_bytes())
    h.update(json.dumps([MOTION_VERSION, duration_s, EPSILON_M, NOISE_ULPS]).encode())
    return h.hexdigest()
