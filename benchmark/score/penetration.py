"""Penetration rate: frames with at least one counted penetrating pair / evaluated frames.

    counted pair (A, B, t) = penetration deeper than the pair's threshold
                             and not inside the pair's initial overlap
                             and (MOVING(A, t) or MOVING(B, t))
                             and not consumed, and not an allowed self pair

The threshold is tau = max(1 mm, min(5 mm, 0.5 * the thinner body's smallest local extent)); a body wholly inside
the other counts at any depth. A fixed 5 mm would be invisible between a foot and a floor but halfway through a 4 mm
card, and the 1 mm floor keeps tau above mesh and float noise. An absolute sweep (1-50 mm) is reported too.

A pair is never measured when it is **rigidly coupled** for the whole clip (static-static, welded, or attached
parts) or **directly jointed** (adjacent links of a DyCraft joint tree, or a Blender parent chain read as URDF-style
link adjacency). Agent-authored collision allowances are never read. Liquid and effect meshes are excluded by
material (`_fluid_rule.py`).

The geometry runs in Blender children (`_export_child.py`, `_geometry_child.py`, `_materials_child.py`,
`_rig_child.py`), each spawning the animation layer's Coal worker. Every stage is cached under the result folder,
keyed by its inputs, so a re-run re-applies the rules (`penetration_rules.py`) without touching a mesh.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import motion
from . import penetration_rules as rules
from .common import HERE, REPO_ROOT, Settings, chunk_count, run_child, run_children, sha256
from .inputs import Input

SCHEMA = "dycraft.score.penetration.v1"
#: The never-moving pairs are measured once each, for a diagnostic; this bounds that work.
MAX_STATIC_PAIRS = 50000
ADAPTIVE = {"floor_m": 0.001, "cap_m": 0.005, "fraction": 0.5}


# ------------------------------------------------------------------- export

def export(inp: Input, out: Path, settings: Settings) -> Path:
    """Export `inp.blend` into `out/export/` (cached by the blend's hash and the exporter's code); returns it."""
    export_dir = out / "export"
    manifest_path = export_dir / "export_manifest.json"
    index = export_dir / "evaluation" / "index.json"
    blend_sha = sha256(inp.blend)
    recipe = {"child": sha256(HERE / "_export_child.py"),
              "worker": sha256(REPO_ROOT / "animation" / "blender" / "worker.py")}
    if not settings.recompute and manifest_path.is_file() and index.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("blend_sha256") == blend_sha and manifest.get("recipe") == recipe
                and json.loads(index.read_text()).get("complete")):
            return export_dir
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    workspace = inp.blend.parent.parent if inp.blend.parent.name == "bundle" else inp.blend.parent
    request = export_dir / "export_request.json"
    request.write_text(json.dumps({"repo": str(REPO_ROOT), "blend": str(inp.blend),
                                   "output_root": str(export_dir), "workspace_root": str(workspace)}))
    started = time.perf_counter()
    run_child(settings, "_export_child.py", [request])
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"] = recipe
    manifest["wall_s"] = time.perf_counter() - started
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return export_dir


def evaluation_of(inp: Input, out: Path, settings: Settings) -> tuple[Path, Path | None]:
    """(evaluation directory, `bodies.json` or None) for an input: delivered, or exported from its blend."""
    if inp.evaluation_dir is not None:
        sidecar = inp.evaluation_dir.parent / "bodies.json"
        return inp.evaluation_dir, sidecar if sidecar.is_file() else None
    export_dir = export(inp, out, settings)
    # A DyCraft bundle is scored with the structure of its joint tree only, as the paper scored every DyCraft film:
    # the export's record of Blender parenting would make welded parts adjacent and exempt pairs the joint tree
    # leaves measured (a wheel's tire against the rim welded to it).
    return export_dir / "evaluation", (None if inp.kind == "dycraft" else export_dir / "bodies.json")


def motion_states(evaluation_dir: Path, out: Path, settings: Settings) -> dict:
    key = motion.cache_key(evaluation_dir)
    path = out / "motion.json"
    if not settings.recompute and path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("cache_key") == key:
            return cached
    result = motion.compute(evaluation_dir)
    result["cache_key"] = key
    path.write_text(json.dumps(result))
    return result


# ----------------------------------------------------------------- geometry

def _child_version(path: Path) -> str:
    found = re.search(r'^GEOMETRY_VERSION = "([^"]+)"', path.read_text(), re.M)
    if not found:
        raise SystemExit(f"{path} declares no GEOMETRY_VERSION")
    return found.group(1)


def geometry_key(evaluation_dir: Path, inp: Input, bodies: Path | None, ladder) -> tuple[dict, str]:
    """(child request, cache key). The child is identified by its declared GEOMETRY_VERSION, bumped whenever its
    measurements change, plus the collision code it calls."""
    request = {
        "repo": str(REPO_ROOT), "evaluation_dir": str(evaluation_dir),
        "bodies_path": str(bodies) if bodies else None,
        "baseline_path": str(inp.baseline) if inp.baseline else None,
        "ladder_m": list(ladder), "adaptive": ADAPTIVE, "max_static_pairs": MAX_STATIC_PAIRS,
    }
    code = {name: sha256(path) for name, path in {
        "records": HERE / "records.py",
        "collision": REPO_ROOT / "animation" / "blender" / "collision.py",
        "coal_worker": REPO_ROOT / "animation" / "blender" / "coal_worker.py",
        "erosion": REPO_ROOT / "verify" / "_mesh_erosion.py",
        "cache": REPO_ROOT / "animation" / "blender" / "collision_cache.py"}.items()}
    code["child"] = _child_version(HERE / "_geometry_child.py")
    key_material = {**{k: v for k, v in request.items() if k != "repo"},
                    "index": sha256(evaluation_dir / "index.json"), "bodies": sha256(bodies),
                    "baseline": sha256(inp.baseline), "code": code}
    return request, hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()


def _sum_numbers(dicts):
    out = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                out[k] = _sum_numbers([out.get(k, {}), v])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = out.get(k, 0) + v
            else:
                out.setdefault(k, v)
    return out


def merge_chunks(parts: list, ranges: list) -> dict:
    """One geometry result from frame-range chunks.

    A pair's code at frame f is the one from the chunk that owns f. Within a chunk, a pair's depth for a rounded
    relative pose is computed once and reused, so in a rare near-degenerate contact a code can depend on where its
    chunk starts (README).
    Never-moving pairs come from every part (a separate process measures them); everything computed from the whole
    clip (bodies, candidates, agents, tunneling) comes from the first chunk.
    """
    n = parts[0]["n_frames"]
    merged = {}
    for part, (lo, hi) in zip(parts, ranges):
        for rec in part["pairs"]:
            if rec.get("never_moving"):
                continue
            key = (rec["a"], rec["b"])
            if key not in merged:
                merged[key] = (rec, rules.expand(rec["timeline"], n))
            codes = rules.expand(rec["timeline"], n)
            merged[key][1][lo:hi] = codes[lo:hi]
    penetrating = (rules.CROSSING, rules.CONTAINED, rules.CONTAINS, rules.ERROR)
    pairs = [{**rec, "timeline": rules.compress(codes)} for rec, codes in merged.values()
             if any(c[0] in penetrating for c in codes)]
    pairs += [rec for part in parts for rec in part["pairs"] if rec.get("never_moving")]
    pairs.sort(key=lambda x: (x["a"], x["b"]))
    enclosures = {}
    for part in parts:
        for enc in part.get("enclosures", []):
            key = (enc["object"], enc["agent"])
            e = enclosures.setdefault(key, {**enc, "enclosed_frames": [], "frames_tested": 0})
            e["enclosed_frames"] += [f for s_, e_ in enc["enclosed_frames"] for f in range(s_, e_ + 1)]
            e["frames_tested"] += enc["frames_tested"]
    for e in enclosures.values():
        e["enclosed_frames"] = rules.ranges(sorted(set(e["enclosed_frames"])))
    out = dict(parts[0])
    out["pairs"] = pairs
    out["enclosures"] = [enclosures[k] for k in sorted(enclosures)]
    out["errors"] = [err for part in parts for err in part.get("errors", [])][:200]
    static = [part.get("never_moving_pairs") or {} for part in parts]
    if any(static):
        out["never_moving_pairs"] = {**max(static, key=lambda d: d.get("aabb_candidates", 0)),
                                     **{k: sum(d.get(k, 0) for d in static) for k in ("measured", "overlapping")},
                                     "truncated": any(d.get("truncated") for d in static)}
    out["stats"] = _sum_numbers([part.get("stats") for part in parts])
    out["timing"] = {**_sum_numbers([part.get("timing") for part in parts]),
                     "chunks": len(parts), "wall_s": max(part["timing"]["total_s"] for part in parts)}
    return out


def geometry(evaluation_dir: Path, out: Path, inp: Input, bodies: Path | None, ladder, settings: Settings) -> dict:
    """The measured pair timelines (`_geometry_child.py`), cached in `penetration_raw.json`.

    With more than one chunk (`settings.chunks`, default `settings.jobs`) the frames are split over that many
    children, at most `settings.jobs` at a time, and the never-moving pairs are measured by one more; chunks are
    merged by frame.
    """
    request, cache_key = geometry_key(evaluation_dir, inp, bodies, ladder)
    raw_path = out / "penetration_raw.json"
    if not settings.recompute and raw_path.is_file():
        raw = json.loads(raw_path.read_text())
        if raw.get("cache_key") == cache_key:
            return raw
    n = json.loads((evaluation_dir / "index.json").read_text())["n_frames"]
    chunks = chunk_count(settings, n)
    if chunks == 1:
        req_path = out / "geometry_request.json"
        req_path.write_text(json.dumps(request))
        tmp = out / "penetration_raw.json.tmp"
        run_child(settings, "_geometry_child.py", [req_path, tmp])
        raw = json.loads(tmp.read_text())
        tmp.unlink()
        req_path.unlink()
    else:
        bounds = [round(n * c / chunks) for c in range(chunks + 1)]
        work = [(f"f{bounds[c]}-{bounds[c + 1]}", {"frame_range": [bounds[c], bounds[c + 1]], "measure_static": False})
                for c in range(chunks) if bounds[c + 1] > bounds[c]]
        work.append(("static", {"frame_range": [0, 0], "measure_static": True}))
        jobs = []
        for label, over in work:
            req_path = out / f"geometry_request.{label}.json"
            req_path.write_text(json.dumps({**request, **over}))
            jobs.append((label, [req_path, out / f"penetration_raw.{label}.json.tmp"]))
        run_children(settings, "_geometry_child.py", jobs, out)
        parts = [json.loads(Path(args[1]).read_text()) for _, args in jobs]
        raw = merge_chunks(parts, [tuple(over["frame_range"]) for _, over in work])
        for _, (req_path, tmp) in jobs:
            req_path.unlink(missing_ok=True)
            Path(tmp).unlink(missing_ok=True)
    raw["cache_key"] = cache_key
    raw_path.write_text(json.dumps(raw))
    return raw


def material_classes(inp: Input, out: Path, bodies: Path | None, settings: Settings) -> dict:
    """{object name: class} for liquid and effect meshes (`_fluid_rule.py`).

    An exported blend's `bodies.json` already carries the classes. A DyCraft evaluation carries no material data, so
    its `animation.blend` is read once by `_materials_child.py`, cached by the blend's identity.
    """
    classes = {}
    if bodies and Path(bodies).is_file():
        side = json.loads(Path(bodies).read_text()).get("objects", {})
        classes.update({k: v["class"] for k, v in side.items() if v.get("class") in ("fluid", "effect")})
    if inp.kind != "dycraft" or inp.blend is None:
        return classes
    blend = inp.blend
    path = out / "materials.json"
    key = {"blend": str(blend), "size": blend.stat().st_size, "mtime_ns": blend.stat().st_mtime_ns,
           "rule": sha256(HERE / "_fluid_rule.py"), "child": sha256(HERE / "_materials_child.py")}
    cached = json.loads(path.read_text()) if path.is_file() else None
    if settings.recompute or not cached or cached.get("cache_key") != key:
        request = out / "materials_request.json"
        request.write_text(json.dumps({"blend": str(blend), "output": str(path)}))
        run_child(settings, "_materials_child.py", [request])
        request.unlink()
        cached = json.loads(path.read_text())
        cached["cache_key"] = key
        path.write_text(json.dumps(cached, indent=1, sort_keys=True))
    classes.update({k: v["class"] for k, v in cached["objects"].items()})
    return classes


def rig_structure(inp: Input, out: Path, settings: Settings) -> tuple[dict, set, dict]:
    """Joint structure of armature rigs (`_rig_child.py`): ({object: bone}, {frozenset(connected bone pair)},
    {frozenset(part pair): rest-pose overlap mm}). Read once from the input's blend, cached by its identity; empty
    when there is no blend."""
    blend = inp.blend
    if blend is None or not blend.is_file():
        return {}, set(), {}
    path = out / "rig.json"
    key = {"blend": str(blend), "size": blend.stat().st_size, "mtime_ns": blend.stat().st_mtime_ns,
           "child": sha256(HERE / "_rig_child.py")}
    cached = json.loads(path.read_text()) if path.is_file() else None
    if settings.recompute or not cached or cached.get("cache_key") != key:
        request = out / "rig_request.json"
        request.write_text(json.dumps({"blend": str(blend), "output": str(path)}))
        run_child(settings, "_rig_child.py", [request])
        request.unlink()
        cached = json.loads(path.read_text())
        cached["cache_key"] = key
        path.write_text(json.dumps(cached, indent=1, sort_keys=True))
    return rig_from(cached)


def rig_from(cached: dict) -> tuple[dict, set, dict]:
    connected = {frozenset((a, b)) for a, b, _ in cached["connected"] if a != b}
    rest = {frozenset((a, b)): d for a, b, d in cached.get("rest_overlaps", [])}
    return cached["object_bone"], connected, rest


# -------------------------------------------------------------------- rules

def drop_material_bodies(raw: dict, classes: dict) -> tuple[dict, dict]:
    """Remove pairs involving liquid or effect bodies: the same as excluding the bodies up front, since no other
    pair's measurement depends on them."""
    if not classes:
        return raw, {"excluded_bodies": 0, "pairs_dropped": 0}
    kept, dropped = [], []
    for p in raw["pairs"]:
        (dropped if p["a_name"] in classes or p["b_name"] in classes else kept).append(p)
    by_class = {}
    for name, cls in classes.items():
        by_class.setdefault(cls, []).append(name)
    diag = {"excluded_bodies": len(classes),
            "by_class": {k: sorted(v)[:100] for k, v in sorted(by_class.items())},
            "pairs_dropped": len(dropped),
            "dropped_pairs": sorted([p["a"], p["b"]] for p in dropped)[:100]}
    return {**raw, "pairs": kept}, diag


def jointed_groups(baseline_path) -> tuple[dict, set]:
    """Rigid groups of links, and which pairs of groups share a moving joint.

    A DyCraft scene often gives every part its own link and welds them with fixed joints (`baseline["fixed_edges"]`);
    the weld graph's connected components are the rigid bodies, and each revolute or prismatic joint in
    `baseline["joints"]` connects two of them. Returns ({identity: group}, {frozenset((group_a, group_b))}).
    """
    if not baseline_path or not Path(baseline_path).is_file():
        return {}, set()
    baseline = json.loads(Path(baseline_path).read_text())
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for edge in baseline.get("fixed_edges", []):
        a, b = find(edge["child"]), find(edge["parent"])
        if a != b:
            parent[a] = b
    group_of = {ident: find(rec["link"]) for ident, rec in baseline.get("identities", {}).items() if rec.get("link")}
    jointed = {frozenset((find(j["parent"]), find(j["child"]))) for j in baseline.get("joints", {}).values()}
    return group_of, {g for g in jointed if len(g) == 2}


def body_enclosures(pairs: list, agents: set, whole_objects: set) -> list:
    """Single-body enclosure candidates: a body X of another object lying wholly inside one body of an agent, on the
    frames it does. The geometry child's whole-object test covers objects of one body or entirely enclosed; this
    covers a piece that leaves its own object (a morsel taken from a pouch and swallowed). Returns
    [(enclosure, member pairs)] in the whole-object format."""
    frames, members, own_bodies, asm_of = {}, {}, {}, {}
    for p in pairs:
        for side in "ab":
            name, asm = p.get(f"{side}_name", p[side]), p[f"assembly_{side}"]
            own_bodies.setdefault(asm, set()).add(name)
            asm_of[name] = asm
    for p in pairs:
        if p["relation"] != "cross":
            continue
        na, nb = p.get("a_name", p["a"]), p.get("b_name", p["b"])
        for x, a, asm_x, asm_a, code in ((na, nb, p["assembly_a"], p["assembly_b"], rules.CONTAINED),
                                          (nb, na, p["assembly_b"], p["assembly_a"], rules.CONTAINS)):
            if asm_a not in agents or asm_x in whole_objects:
                continue
            key = (x, asm_a)
            members.setdefault(key, []).append(p)
            frames.setdefault(key, set()).update(f for f, c in enumerate(p["codes"]) if c[0] == code)
    out = []
    for (x, asm_a), fs in sorted(frames.items()):
        if not fs:
            continue
        host = {}  # frame -> the agent body that contains x there
        for q in members[(x, asm_a)]:
            qa, qb = q.get("a_name", q["a"]), q.get("b_name", q["b"])
            other, code = (qb, rules.CONTAINED) if qa == x else (qa, rules.CONTAINS)
            for f, c in enumerate(q["codes"]):
                if c[0] == code:
                    host.setdefault(f, other)
        agent_bodies = sorted({q.get("b_name", q["b"]) if q.get("a_name", q["a"]) == x else q.get("a_name", q["a"])
                               for q in members[(x, asm_a)]})
        own = sorted(own_bodies.get(asm_of[x], set()) - {x})
        out.append(({"object": x, "agent": asm_a, "object_bodies": [x], "agent_bodies": agent_bodies,
                     "own_bodies": own, "enclosed_frames": rules.ranges(sorted(fs)), "single_body": True,
                     "host": host}, members[(x, asm_a)]))
    return out


def consumption(raw: dict, pairs: list, fps: float, states: dict | None = None,
                evaluation_dir: Path | None = None, window_s: float = 1.0,
                persist_s: float = 0.5) -> tuple[dict, list]:
    """Consumed frames per pair: an object an agent has taken inside itself (a dog swallowing a treat).

    An *agent* is an object whose parts move relative to each other or deform (a character, an animal, a robot, an
    articulated machine), decided from the transforms. An object X is **consumed** by agent A over a run of frames
    only when:

    1. X is wholly inside A for every frame of the run (the geometry child's whole-object test, or a single body of
       a larger object contained in one body of A, `body_enclosures`), and the run does not begin at their first
       shared frame;
    2. within `window_s` before the run, or at its first frame, some body of X partially penetrated A (ingestion);
    3. X itself moved during that window or the run: it was brought in, not swept over;
    4. X comes to ride with A: over a trailing stretch of at least `persist_s` (or all of the run, if shorter) its
       centroid stays within max(2 cm, 10 % of its size) of a fixed point of the body enclosing it; the way in (mouth
       to stomach) may move;
    5. X was separate before: at some frame before the ingestion contact every X-A pair is co-visible and clear, so
       a part built into or onto A never qualifies;
    6. the containment lasts at least `persist_s` or to the end of the clip (a limb clipping through a torso for a
       few frames is penetration, not ingestion);
    7. a single body of a larger object must have left that object: it is not fixed to any other body of its object
       from `window_s` before the entry to the end (a throat in its head, a thumb on its palm).

    It stays consumed until X leaves A or the clip ends. The ingestion contact itself still counts. Rejected runs
    are reported with the failing condition.
    """
    window = max(1, int(round(window_s * fps)))
    persist = max(1, int(round(persist_s * fps)))
    by_objects = {}
    for p in pairs:
        by_objects.setdefault(frozenset((p["assembly_a"], p["assembly_b"])), []).append(p)
    consumed, report = {}, []
    encs = raw.get("enclosures", [])
    geo = None
    if evaluation_dir is not None and (encs or any(p["relation"] == "cross" and any(
            c[0] in (rules.CONTAINED, rules.CONTAINS) for c in p["codes"]) for p in pairs)):
        _, names, M, hid, boxes = motion.read_transforms(evaluation_dir)
        geo = ({n: i for i, n in enumerate(names)}, M, hid, boxes)

    def moved(bodies, lo, hi):
        if states is None:
            return True
        for b in bodies:
            for s_, e_ in states["moving_intervals"].get(b, ()):
                if s_ <= hi and e_ >= lo:
                    return True
        return False

    def co_moving(obj_bodies, agent_bodies, s, e, host_of=None, own_bodies=()):
        """Conditions 4 and 7: (ok, drift_mm over the riding stretch, first frame of it or `attached:<body>`)."""
        if geo is None:
            return True, None, s
        where, M, hid, boxes = geo
        xs = [where[b] for b in obj_bodies if b in where]
        as_ = [where[b] for b in agent_bodies if b in where]
        if not xs or not as_:
            return False, None, None

        def centroid(f):
            pts, size = [], 0.0
            for x in xs:
                if hid[x, f] < 0:
                    continue
                lo, hi = boxes[hid[x, f]]
                pts.append(M[x, f, :3, :3] @ ((lo + hi) / 2) + M[x, f, :3, 3])
                size = max(size, float(np.max(np.abs(M[x, f, :3, :3]) @ (hi - lo))))  # world extent
            return (np.mean(pts, axis=0) if pts else None), size

        def local(a, f, c):
            return np.linalg.lstsq(M[a, f, :3, :3], c - M[a, f, :3, 3], rcond=None)[0]

        def host(f, c):
            h = where.get(host_of.get(f)) if host_of else None
            if h is not None:
                return h
            best, pick = np.inf, None  # the agent body whose box the centroid is deepest in
            for a in as_:
                if hid[a, f] < 0:
                    continue
                lo, hi = boxes[hid[a, f]]
                l = local(a, f, c)
                outside = float(np.maximum(np.maximum(lo - l, l - hi), 0).max())
                if outside < best:
                    best, pick = outside, a
            return pick
        track = []
        for f in range(s, e + 1):
            c, size = centroid(f)
            if c is not None:
                track.append((f, c, size, host(f, c)))
        if not track or track[-1][3] is None:
            return False, None, None
        f_end, c_end, size, h = track[-1]
        tol = max(0.02, 0.1 * size)
        anchor = local(h, f_end, c_end)
        drift, first = 0.0, f_end
        for f, c, _, hf in reversed(track):
            if hf != h:
                break
            d = float(np.linalg.norm(M[h, f, :3, :3] @ anchor + M[h, f, :3, 3] - c))
            if d > tol:
                break
            drift, first = max(drift, d), f
        rides = e - first + 1 >= min(persist, e - s + 1)
        approach = [(f, centroid(f)[0]) for f in range(max(0, s - window), e + 1)]
        approach = [(f, c) for f, c in approach if c is not None]
        for o in own_bodies:
            k = where.get(o)
            if k is None or not rides:
                continue
            seen = [(f, c) for f, c in approach if hid[k, f] >= 0]
            if len(seen) < 2:
                continue
            ref = local(k, seen[-1][0], seen[-1][1])
            if max(float(np.linalg.norm(M[k, f, :3, :3] @ ref + M[k, f, :3, 3] - c)) for f, c in seen) <= tol:
                return False, round(drift * 1000, 1), f"attached:{o}"
        return rides, round(drift * 1000, 1), first

    candidates = []
    for enc in encs:
        members = by_objects.get(frozenset((enc["object"], enc["agent"])), [])
        if members:
            candidates.append((enc, members))
    candidates += body_enclosures(pairs, set(raw.get("agents", ())), {e["object"] for e in encs})
    for enc, members in candidates:
        X, A = enc["object"], enc["agent"]
        first_shared = min((next((f for f, c in enumerate(p["codes"]) if c[0] != rules.INACTIVE), 10**9)
                            for p in members), default=0)
        for s, e in enc["enclosed_frames"]:
            entry = {"object": X, "agent": A, "enclosed_frames": [s, e], "length": e - s + 1}
            n_frames = len(members[0]["codes"])
            if e - s + 1 < persist and e < n_frames - 1:
                report.append({**entry, "consumed": False, "rejected": f"not persistent (< {persist_s} s, then exits)"})
                continue
            if s <= first_shared:
                report.append({**entry, "consumed": False, "rejected": "enclosed from the first shared frame"})
                continue
            # a small morsel can be bitten in one frame: inside the skull while still crossing the jaw
            contact = [f for f in range(max(0, s - window), s)
                       if any(p["codes"][f][0] in rules.PENETRATING for p in members)]
            if not contact and any(p["codes"][s][0] == rules.CROSSING for p in members):
                contact = [s]
            if not contact:
                report.append({**entry, "consumed": False, "rejected": "no ingestion contact in the preceding window"})
                continue
            if not any(all(p["codes"][f][0] in (rules.CLEAR, rules.INACTIVE) for p in members)
                       and any(p["codes"][f][0] == rules.CLEAR for p in members) for f in range(contact[0])):
                report.append({**entry, "consumed": False, "rejected": "never separate from the agent before the contact"})
                continue
            if not moved(enc["object_bodies"], max(0, s - window), e):
                report.append({**entry, "consumed": False, "rejected": "object never moved (swept over, not brought in)"})
                continue
            ok, drift, settled = co_moving(enc["object_bodies"], enc["agent_bodies"], s, e, enc.get("host"),
                                           enc.get("own_bodies", ()))
            if not ok:
                reason = ("still attached to its own object (" + settled.split(":", 1)[1] + ")"
                          if isinstance(settled, str) else "object never comes to ride with the enclosing body")
                report.append({**entry, "consumed": False, "drift_mm": drift, "rejected": reason})
                continue
            for p in members:
                consumed.setdefault(id(p), set()).update(range(s, e + 1))
            report.append({**entry, "consumed": True, "ingestion_contact_frame": contact[0], "drift_mm": drift,
                           "rides_from_frame": settled})
    for p in pairs:
        if id(p) in consumed:
            p["consumed"] = sorted(consumed[id(p)])
    return consumed, sorted(report, key=lambda r: (r["enclosed_frames"][0], r["object"]))


def attach_motion(raw: dict, states: dict, structure=({}, set()), rig=({}, set(), {})) -> list:
    """Expanded codes plus, per pair: `dynamic` (a body MOVING, per frame), `jointed` (a self pair on rigid groups
    joined by a moving joint, or on two connected bones of a rig) and `rest_overlap_mm` (a self pair of one rig
    that already interpenetrates in its rest pose, as modeled)."""
    n = raw["n_frames"]
    masks = {}

    def mask(name):
        if name not in masks:
            masks[name] = motion.mask_for(states, name, n)
        return masks[name]
    group_of, jointed = structure
    bone_of, bones_connected, rest = rig
    pairs = []
    for p in raw["pairs"]:
        a, b = mask(p["a_name"]), mask(p["b_name"])
        entry = {**p, "codes": rules.expand(p["timeline"], n), "dynamic": [x or y for x, y in zip(a, b)]}
        ga, gb = group_of.get(p["a"]), group_of.get(p["b"])
        if p["relation"] == "self" and ga is not None and gb is not None and frozenset((ga, gb)) in jointed:
            entry["jointed"] = True
        ba, bb = bone_of.get(p["a_name"]), bone_of.get(p["b_name"])
        if p["relation"] == "self" and ba is not None and bb is not None and frozenset((ba, bb)) in bones_connected:
            entry["jointed"] = True
        built = rest.get(frozenset((p["a_name"], p["b_name"])))
        if p["relation"] == "self" and built:
            entry["rest_overlap_mm"] = built
        pairs.append(entry)
    return pairs


def score_pairs(raw: dict, ladder, sweep_m, pairs, params=rules.DEFAULT_PARAMS) -> tuple[dict, list]:
    n, fps = raw["n_frames"], raw["fps"]
    primary = rules.evaluate(pairs, n, fps, rules.passes_adaptive, ladder, params)
    sweep = []
    for d in sweep_m:
        s = rules.evaluate(pairs, n, fps, rules.passes_rung(ladder.index(round(d, 9))), ladder, params, detail=False)
        sweep.append({"threshold_mm": round(d * 1000, 6), **s})
    return primary, sweep


def apply_rules(raw: dict, states: dict, classes: dict, baseline, rig, evaluation_dir: Path | None) -> dict:
    """The policy on a measured (or cached) geometry pass: {primary, sweep, diagnostics}."""
    ladder = sorted({round(x, 9) for x in (*rules.DEFAULT_LADDER_M, *rules.DEFAULT_SWEEP_M)})
    raw, material_diag = drop_material_bodies(raw, classes)
    bone_of, bones_connected, rest_overlaps = rig
    pairs = attach_motion(raw, states, jointed_groups(baseline), rig)
    _, consumption_report = consumption(raw, pairs, raw["fps"], states, evaluation_dir)
    primary, sweep = score_pairs(raw, ladder, list(rules.DEFAULT_SWEEP_M), pairs)
    if raw.get("frame_start") is not None:
        for ev in primary.get("event_list", []):
            ev["native_frames"] = [ev["start_frame"] + raw["frame_start"], ev["end_frame"] + raw["frame_start"]]
    diagnostics = {
        "bodies": raw["bodies"],
        "candidate_pairs": raw["candidate_pairs"],
        "narrow_phase_pairs": raw["narrow_pairs"],
        "pairs_with_any_hit": len(raw["pairs"]),
        "exemptions": raw["exemptions"],
        "adjacency_pairs_known": raw["adjacency_pairs_known"],
        "rig_joints": {"parts_on_bones": len(bone_of), "connected_bone_pairs": len(bones_connected),
                       "rest_overlap_part_pairs": len(rest_overlaps),
                       "self_pairs_built_overlapping": sum(1 for p in pairs if p.get("rest_overlap_mm")),
                       "self_pairs_jointed_by_rig": sum(1 for p in pairs if p.get("jointed")
                                                        and p["a_name"] in bone_of and p["b_name"] in bone_of)},
        "never_moving_pairs": raw.get("never_moving_pairs"),
        "liquid_effect_exclusion": material_diag,
        "agents": raw.get("agents", []),
        "consumption": consumption_report,
        "enclosures": raw.get("enclosures", []),
        "measurement_errors": {"count": len(raw["errors"]), "first": raw["errors"][:20]},
        "tunneling_risk": raw["tunneling_risk"],
        "motion": {k: states[k] for k in ("fps", "window_frames", "bodies", "bodies_ever_moving",
                                          "moving_body_frames", "static_body_frames", "transition_noise")},
    }
    return {"primary": primary, "sweep": sweep, "diagnostics": diagnostics,
            "config": {"threshold": "max(1 mm, min(5 mm, 0.5 * thinner body's smallest extent)); containment at any depth",
                       "ladder_mm": [round(x * 1000, 6) for x in ladder], "rules": rules.DEFAULT_PARAMS,
                       "geometry_version": raw["geometry_version"]}}


def run(inp: Input, out: Path, settings: Settings) -> dict:
    """Score one input; writes `out/penetration.json` and returns it."""
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    evaluation_dir, bodies = evaluation_of(inp, out, settings)
    ladder = sorted({round(x, 9) for x in (*rules.DEFAULT_LADDER_M, *rules.DEFAULT_SWEEP_M)})
    states = motion_states(evaluation_dir, out, settings)
    raw = geometry(evaluation_dir, out, inp, bodies, ladder, settings)
    classes = material_classes(inp, out, bodies, settings)
    rig = rig_structure(inp, out, settings)
    result = {"schema": SCHEMA,
              "input": {"path": str(inp.path), "kind": inp.kind, "evaluation_dir": str(evaluation_dir),
                        "exported": inp.evaluation_dir is None, "blend": str(inp.blend) if inp.blend else None,
                        "fps": raw["fps"], "frame_start": raw.get("frame_start"),
                        "input_digest": raw["input_digest"], "notes": inp.notes},
              **apply_rules(raw, states, classes, inp.baseline, rig, evaluation_dir)}
    result["timing"] = {"geometry_cpu_s": raw["timing"].get("total_s"), "total_s": time.perf_counter() - t0}
    (out / "penetration.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result
