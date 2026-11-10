"""Which walls and ceilings the free camera may cut away, worked out from a native film's own geometry.

The viewport hides a wall while it stands between the free camera and the action, and a ceiling once the camera rises
above it (`web/src/components/viewer3d/cutaway.ts`). This module decides which objects are walls and ceilings. It
reads only the film's evaluated geometry, the same track and meshes the viewport draws, and never a name, so it
works the same on any scene. It is conservative: when in doubt, an object stays.

* **A wall** is one static object, mostly vertical faces in one direction (a slab, not a box or a cabinet), at
  least 1.2 m long, at most 0.5 m thick and at least 1.8 m tall, standing on a floor: a horizontal surface of at
  least 2 m² at or above its foot. It has almost no upward faces (not shelving), and its main face fills at least
  half of its outline, so a wall that is mostly an opening stays.
* **A ceiling** is one static object, thin, at least a meter across both ways, facing down with at least 2 m², at
  least 1.9 m above a floor, and over the action. A roof beside the action, a neighbor's garage, stays.
* **Kept:** a wall or ceiling that anything moving touches or passes through on any frame stays, always. That is
  what keeps the wall whose doorway a character walks through, and a wall something leans on.
* **Mounted:** a static object inside a wall's slab, off every floor, and untouched by anything moving (a window)
  hides with that wall. It stays if it touches anything else that stays: a door's glass touches the door leaf
  standing on the floor, a tap its sink.

Room shells (walls and floor merged into one mesh) are never cut: hiding one would take the floor with it.

`cutaway(run_dir)` returns, in Blender's world frame (Z up):

    {"version", "animation": the manifest id, "action": [[x, y, z], [x, y, z]] or null,
     "units": [{"kind": "wall", "objects": [name], "box": [cx, cy, cz, nx, ny, hn, ht, hz]},
               {"kind": "ceiling", "objects": [name], "box": [x0, y0, x1, y1], "under": z}],
     "attach": [{"objects": [name], "units": [unit index]}],
     "kept": [{"objects": [name], "why": text}]}

A wall's box is its slab: center, horizontal normal (nx, ny), and half sizes along the normal, along the wall,
and up. Cached in the viewer's asset cache by the manifest id and `VERSION`, so it is worked out once per film.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

import numpy as np

#: Bumped when the rules change, so a cached answer from older rules is not served.
VERSION = 1

_FLOOR_AREA = 2.0          # m² of upward faces at one height that make a floor
_WALL_MIN_HEIGHT = 1.8
_WALL_MIN_LENGTH = 1.2
_WALL_MAX_THICKNESS = 0.5
_WALL_MIN_FACE = 2.0       # m², one side
_CEILING_MIN_CLEARANCE = 1.9
_TOUCH = 0.05              # how close something moving may come to a wall or ceiling before it stays
_RIGID_STEP = 1            # every frame for an object that wears one mesh
_DEFORMING_STEP = 3        # every third for one whose mesh changes: a crossing lasts far longer than three frames


def _envelope(blob: bytes) -> tuple[dict, memoryview]:
    n = struct.unpack_from("<I", blob, 0)[0]
    return json.loads(blob[4:4 + n]), memoryview(blob)[4 + n:]


def _per_frame(value, n: int) -> list:
    return value if isinstance(value, list) else [value] * n


#: A heavy film's cutaway takes seconds; this bounds one that hangs.
TIMEOUT_S = 600
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
#: Films whose cutaway timed out, remembered for the server's life: no walls are cut there, and the next server tries
#: again. Never retried in the server itself, which would hold its interpreter lock for as long again.
_TIMED_OUT: set[str] = set()


def cutaway(run_dir: Path) -> dict:
    """The film's cutaway, from the cache or worked out now (see the module docstring).

    Worked out in its own interpreter, which the server waits on without holding the interpreter lock: reading every
    mesh of a long film takes seconds of Python, and inside the server that would hold up the meshes playback is
    fetching at the same time. One at a time per film; a second request waits for the first. When the interpreter
    cannot be started or fails, it is worked out here; when it runs out of time, the film gets no cutaway (every wall
    stays), and this server does not try again.
    """
    from ..render.viewer_bake import cache_root
    from . import native_animation as N

    ir = N.manifest(run_dir)
    path = cache_root() / "cutaway" / f"{ir['id']}_v{VERSION}.json"
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(str(path), threading.Lock())
    nothing = {"version": VERSION, "animation": ir["id"], "action": None, "units": [], "attach": [], "kept": []}
    with lock:
        if str(path) in _TIMED_OUT:
            return nothing
        try:
            cached = json.loads(path.read_text())
            if isinstance(cached, dict) and cached.get("animation") == ir["id"]:
                return cached
        except (OSError, ValueError):
            pass
        try:
            result = _work_out_apart(run_dir)
        except subprocess.TimeoutExpired:
            print(f"[viewer] no cutaway for {run_dir}: not worked out within {TIMEOUT_S} s", file=sys.stderr, flush=True)
            _TIMED_OUT.add(str(path))
            return nothing
        except (OSError, RuntimeError) as exc:
            print(f"[viewer] cutaway worked out in the server: {exc}", file=sys.stderr, flush=True)
            result = _work_out(run_dir, ir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(result))
            os.replace(tmp, path)
        except OSError:
            pass
        return result


def _work_out_apart(run_dir: Path) -> dict:
    """`_work_out` in a fresh interpreter (`main` below)."""
    import tempfile

    from .. import _oom

    package = __name__.split(".")[0]
    # The folder the package was imported from, unresolved, so a package reached through a symlink keeps its name.
    root = str(Path(sys.modules[package].__file__).parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
    with tempfile.TemporaryDirectory(prefix="dycraft-cutaway-") as tmp:
        out = Path(tmp) / "cutaway.json"
        proc = subprocess.run(_oom.tiered([sys.executable, "-m", f"{package}.viewer.cutaway", str(run_dir), str(out)],
                                          _oom.WORKER),
                              capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError("working out the cutaway failed:\n  " + "\n  ".join(tail))
        return json.loads(out.read_text())


def main(argv: list[str]) -> int:
    from . import native_animation as N

    run_dir, out = Path(argv[0]), Path(argv[1])
    out.write_text(json.dumps(_work_out(run_dir, N.manifest(run_dir))))
    return 0


def _work_out(run_dir: Path, ir: dict) -> dict:
    from ..animation.blender.bundle import safe_path
    from ..animation.blender.evaluation import read_local
    from . import native_animation as N

    header, body = _envelope(N.track(run_dir))
    mats = np.frombuffer(body, "<f4").reshape(header["n_frames"], len(header["objects"]), 3, 4)
    prefix = PurePosixPath(ir["evaluation"]).parent

    def geometry(k: int):
        digest = header["meshes"][k]
        N.listed_file(run_dir, ir, f"{prefix}/meshes/{digest}.npz")
        v, t = read_local(safe_path(run_dir, str(prefix)), {"frame": 0, "objects": [{"name": "m", "mesh": digest}]})["m"]
        return np.asarray(v, dtype=np.float64), np.asarray(t, dtype=np.int64).reshape(-1, 3)

    return {"version": VERSION, "animation": ir["id"], **classify(header, mats, geometry)}


def classify(header: dict, mats: np.ndarray, load) -> dict:
    """The cutaway of a clip given as the native track gives it.

    `header` is the track's header (`n_frames`, `objects` with `name`, `mesh`, and `hidden`, each one value or one per
    frame); `mats` its `[frames][objects][3][4]` placements; `load(k)` mesh `k`'s vertices and triangles, in the
    object's frame. Returns `action`, `units`, `attach`, and `kept` (see the module docstring).
    """
    nf, objs = header["n_frames"], header["objects"]
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def geometry(k: int):
        if k not in meshes:
            meshes[k] = load(k)
        return meshes[k]

    static, moving = [], []
    for i, o in enumerate(objs):
        rigid = isinstance(o["mesh"], int)
        if rigid and o["mesh"] >= 0 and o["hidden"] is False and np.abs(mats[:, i] - mats[0, i]).max() < 1e-5:
            static.append(i)
        elif any(m >= 0 for m in _per_frame(o["mesh"], nf)):
            moving.append(i)

    # ---- static objects, in world space
    S: dict[int, dict] = {}
    for i in static:
        v, t = geometry(objs[i]["mesh"])
        if not len(t):
            continue
        m = mats[0, i].astype(np.float64)
        w = v @ m[:, :3].T + m[:, 3]
        a, b, c = w[t[:, 0]], w[t[:, 1]], w[t[:, 2]]
        cross = np.cross(b - a, c - a)
        double = np.linalg.norm(cross, axis=1)
        n = np.zeros_like(cross)
        ok = double > 0
        n[ok] = cross[ok] / double[ok, None]
        S[i] = {"w": w, "n": n, "area": double / 2, "cz": (a[:, 2] + b[:, 2] + c[:, 2]) / 3,
                "lo": w.min(0), "hi": w.max(0)}

    # ---- what moves: its box on every frame (every few for a deforming mesh)
    local_box: dict[int, np.ndarray | None] = {}

    def corners(k: int):
        if k not in local_box:
            v, _ = geometry(k)
            local_box[k] = None if not len(v) else np.array(
                [[x, y, z] for x in (v[:, 0].min(), v[:, 0].max()) for y in (v[:, 1].min(), v[:, 1].max())
                 for z in (v[:, 2].min(), v[:, 2].max())])
        return local_box[k]

    los, his, samples = [], [], []
    for i in moving:
        mesh = np.asarray(_per_frame(objs[i]["mesh"], nf))
        shown = (mesh >= 0) & ~np.asarray(_per_frame(objs[i]["hidden"], nf), bool)
        step = _RIGID_STEP if isinstance(objs[i]["mesh"], int) else _DEFORMING_STEP
        frames = np.nonzero(shown)[0]
        frames = frames[(frames % step == 0) | (frames == frames[-1])] if len(frames) else frames
        for k in np.unique(mesh[frames]):
            at = frames[mesh[frames] == k]
            c = corners(int(k))
            if c is None:
                continue
            m = mats[at, i].astype(np.float64)
            w = np.einsum("cj,fij->fci", c, m[:, :, :3]) + m[:, None, :, 3]
            los.append(w.min(1)); his.append(w.max(1))
            samples.append(((w.min(1) + w.max(1)) / 2)[:: max(1, len(at) // 13)])
    boxes_lo = np.concatenate(los) if los else np.zeros((0, 3))
    boxes_hi = np.concatenate(his) if his else np.zeros((0, 3))
    centers = np.concatenate(samples) if samples else np.zeros((0, 3))

    # ---- floors: every height holding at least 2 m² of upward faces
    zs = np.concatenate([s["cz"][s["n"][:, 2] > 0.9] for s in S.values()]) if S else np.zeros(0)
    ar = np.concatenate([s["area"][s["n"][:, 2] > 0.9] for s in S.values()]) if S else np.zeros(0)
    bins = np.round(zs / 0.02).astype(int)
    floors = np.array(sorted(b * 0.02 for b in np.unique(bins) if ar[bins == b].sum() >= _FLOOR_AREA))

    # ---- the action: where what moves goes, 5th to 95th percentile per axis, at least a person's height and a meter
    action = None
    if len(centers):
        lo, hi = np.percentile(centers, 5, axis=0), np.percentile(centers, 95, axis=0)
        base = floors[floors <= lo[2] + 0.1]
        lo[2] = min(lo[2], float(base.max()) if len(base) else lo[2])
        hi[2] = max(hi[2], lo[2] + 1.6)
        mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, [0.5, 0.5, 0.0])
        action = [(mid - half).tolist(), (mid + half).tolist()]

    def touched_by_motion(lo, hi, pad=_TOUCH) -> bool:
        return bool(len(boxes_lo) and (np.all(boxes_lo <= hi + pad, 1) & np.all(boxes_hi >= lo - pad, 1)).any())

    def slab_touched(box) -> bool:
        if not len(boxes_lo):
            return False
        cx, cy, cz, nx, ny, hn, ht, hz = box
        cs = np.stack([np.stack([np.where(k & 1, boxes_hi[:, 0], boxes_lo[:, 0]),
                                 np.where(k & 2, boxes_hi[:, 1], boxes_lo[:, 1]),
                                 np.where(k & 4, boxes_hi[:, 2], boxes_lo[:, 2])], 1) for k in range(8)], 1)
        q = cs - np.array([cx, cy, cz])
        a, b = q[..., 0] * nx + q[..., 1] * ny, -q[..., 0] * ny + q[..., 1] * nx
        p = _TOUCH
        return bool(((a.min(1) <= hn + p) & (a.max(1) >= -hn - p) & (b.min(1) <= ht + p) & (b.max(1) >= -ht - p)
                     & (q[..., 2].min(1) <= hz + p) & (q[..., 2].max(1) >= -hz - p)).any())

    def wall_frame(s):
        """A slab's horizontal normal and extents, or None when its faces are not one vertical slab."""
        vertical = np.abs(s["n"][:, 2]) < 0.3
        if s["area"][vertical].sum() < 0.7 * s["area"].sum():
            return None
        nh, wts = s["n"][vertical][:, :2], s["area"][vertical]
        d = np.linalg.eigh((nh * wts[:, None]).T @ nh)[1][:, -1]
        along = np.abs(nh @ d) > 0.95
        if wts[along].sum() < 0.6 * wts.sum():
            return None                                  # faces in two directions: a box, not a wall
        t = np.array([-d[1], d[0]])
        pn, pt = s["w"][:, :2] @ d, s["w"][:, :2] @ t
        return d, t, (pn.min(), pn.max()), (pt.min(), pt.max()), wts[along].sum() / 2

    units: list[dict] = []
    kept: list[dict] = []
    walls: list[dict] = []
    for i, s in S.items():
        lo, hi = s["lo"], s["hi"]
        name = objs[i]["name"]
        down = s["area"][s["n"][:, 2] < -0.9].sum()
        if (hi[2] - lo[2] <= 0.6 and down >= _FLOOR_AREA and min(hi[0] - lo[0], hi[1] - lo[1]) >= 1.0
                and (floors <= lo[2] - _CEILING_MIN_CLEARANCE).any()
                and action is not None and lo[0] <= action[1][0] and hi[0] >= action[0][0]
                and lo[1] <= action[1][1] and hi[1] >= action[0][1]):
            unit = {"kind": "ceiling", "objects": [name], "box": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
                    "under": float(lo[2])}
            if touched_by_motion(lo, hi):
                kept.append({"objects": [name], "why": "something moving reaches it"})
            else:
                units.append(unit)
            continue
        if not ((floors >= lo[2] - 0.3) & (floors <= hi[2] - _WALL_MIN_HEIGHT)).any():
            continue                                     # not standing on a floor, or not tall enough above one
        frame = wall_frame(s)
        if frame is None:
            continue
        d, t, (n0, n1), (t0, t1), face = frame
        height = hi[2] - lo[2]
        if n1 - n0 > _WALL_MAX_THICKNESS or t1 - t0 < _WALL_MIN_LENGTH or face < _WALL_MIN_FACE:
            continue
        if s["area"][s["n"][:, 2] > 0.7].sum() > 0.15 * s["area"].sum() or face < 0.5 * (t1 - t0) * height:
            continue                                     # shelving, or more opening than wall
        center = (n0 + n1) / 2 * d + (t0 + t1) / 2 * t
        box = [float(center[0]), float(center[1]), float((lo[2] + hi[2]) / 2), float(d[0]), float(d[1]),
               float((n1 - n0) / 2 + 0.005), float((t1 - t0) / 2 + 0.005), float(height / 2 + 0.005)]
        if slab_touched(box):
            kept.append({"objects": [name], "why": "something moving touches or passes through it"})
        else:
            walls.append({"kind": "wall", "objects": [name], "box": box})
    units = walls + units

    # ---- mounted: inside a wall's slab, off every floor, untouched by motion, and touching nothing that stays
    def in_slab(box, p, pad):
        cx, cy, cz, nx, ny, hn, ht, hz = box
        q = p - np.array([cx, cy, cz])
        a, b = q[:, 0] * nx + q[:, 1] * ny, -q[:, 0] * ny + q[:, 1] * nx
        return (np.abs(a) <= hn + pad) & (np.abs(b) <= ht + pad) & (np.abs(q[:, 2]) <= hz + pad)

    in_unit = {n for u in units for n in u["objects"]}
    candidate: dict[int, int] = {}
    for i, s in S.items():
        if objs[i]["name"] in in_unit or (np.abs(floors - s["lo"][2]) <= 0.05).any():
            continue
        points = s["w"][:: max(1, len(s["w"]) // 200)]
        for k, u in enumerate(units):
            if u["kind"] == "wall" and in_slab(u["box"], points, 0.03).mean() >= 0.9:
                if not touched_by_motion(s["lo"], s["hi"], 0.02):
                    candidate[i] = k
                break
    stays = [i for i in S if i not in candidate and objs[i]["name"] not in in_unit]
    changed = True
    while changed and candidate:
        changed = False
        lo = np.array([S[j]["lo"] for j in stays]) if stays else np.zeros((0, 3))
        hi = np.array([S[j]["hi"] for j in stays]) if stays else np.zeros((0, 3))
        for i in list(candidate):
            if len(lo) and (np.all(lo <= S[i]["hi"] + 0.01, 1) & np.all(hi >= S[i]["lo"] - 0.01, 1)).any():
                stays.append(i)
                del candidate[i]
                changed = True
    attach = [{"objects": [objs[i]["name"]], "units": [k]} for i, k in sorted(candidate.items())]
    return {"action": action, "units": units, "attach": attach, "kept": kept}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
