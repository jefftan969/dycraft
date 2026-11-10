"""Floating rate: unsupported dynamic body-frames / dynamic body-frames.

A body is unsupported at t when no support path leads to it from a source; it is illegal unless it is inside a
ballistic free-flight segment. The sources are the ground anchors, self-propelled or buoyant bodies the brief names
(`tasks.json`), and every never-moving scored body, so a moving part counts only for its own support failure. Support
paths use only measured geometry and delivered structure:

    passive     A -> B   A rests on B: contact <= tau_p below A's centroid on a surface no steeper than 75 deg
                         (or embedded, reaching below it)
    pivot       A -> B   A turns about a fixed axis through B, in contact <= tau_s
    mounted     A -- B   both STATIC (motion.py), contact <= tau_p in any direction
    structural  A -- B   same link, joint, or weld (DyCraft baseline) or Blender parenting, contact <= tau_s
    bonded      A -- B   contact <= tau_b, a stable relative pose, moved together (a grasp or a carry)
    carrier     A -- C   contact with a deforming body (a cord, cloth, a skinned coat)

Agent declarations (touch events, allowances, assertions, placements) are never used as support. The geometry pass
runs in `_support_child.py` under the `bpy` interpreter and is cached as `floating_raw.npz`; a re-run re-applies only
`support_rules`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from . import classes as C
from . import structure
from . import support_rules as R
from .common import HERE, REPO_ROOT, Settings, chunk_count, run_child, run_children, sha256
from .inputs import Input
from .penetration import evaluation_of, motion_states

SCHEMA = "dycraft.score.floating.v1"
#: The largest gap measured; it covers the contact tolerances and their 1.5x hysteresis.
QUERY_M = 0.030
#: Contact samples are taken only where a resting or mounted contact can latch: 10 mm x the 1.5 exit factor.
SAMPLE_M = 0.015
SLOPE_DEG = 75.0
#: Contact-sample vertices per body per pair, evenly spread after an exact box prefilter.
MAX_SAMPLES = 2000


def child_version() -> str:
    m = re.search(r'^CHILD_VERSION = "([^"]+)"', (HERE / "_support_child.py").read_text(), re.M)
    return m.group(1) if m else "unknown"


def geometry_request(evaluation_dir, bodies, task_entry, inp: Input) -> dict:
    return {"repo": str(REPO_ROOT), "evaluation_dir": str(evaluation_dir),
            "bodies_path": str(bodies) if bodies else None,
            "baseline_path": str(inp.baseline) if inp.baseline else None,
            "query_m": QUERY_M, "slope_deg": SLOPE_DEG, "max_samples": MAX_SAMPLES, "sample_m": SAMPLE_M,
            "task_entry": task_entry}


def geometry_key(request, evaluation_dir, bodies, inp: Input) -> str:
    """Cache key of the geometry pass. Body classes are deliberately not in it: they only decide which pairs the
    child skips, and `classes_compatible` decides whether the cached measurements still cover the current ones."""
    key_material = {
        **{k: v for k, v in request.items() if k not in ("repo", "task_entry")},
        "index": sha256(Path(evaluation_dir) / "index.json"), "bodies": sha256(bodies),
        "baseline": sha256(inp.baseline), "child_version": child_version(),
        "code": {name: sha256(path) for name, path in {
            "records": HERE / "records.py", "structure": HERE / "structure.py",
            "collision": REPO_ROOT / "animation" / "blender" / "collision.py",
            "coal_worker": REPO_ROOT / "animation" / "blender" / "coal_worker.py"}.items()},
    }
    return hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode()).hexdigest()


def merge_chunks(paths):
    """Union of the per-pair runs of every chunk; the per-body arrays are identical in each chunk."""
    base, meta = None, None
    pair_ids, runs, vals, flags = {}, [], [], []
    stats, timing = {}, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            d = {k: data[k] for k in data.files}
        m = json.loads(str(d.pop("meta")))
        if base is None:
            base, meta = d, m
        elif m["names"] != meta["names"] or m["input_digest"] != meta["input_digest"]:
            raise SystemExit("geometry chunks disagree on the export they measured")
        remap = np.zeros(len(d["pairs"]), dtype=np.int64)
        for k, (i, j) in enumerate(d["pairs"].tolist()):
            remap[k] = pair_ids.setdefault((i, j), len(pair_ids))
        if len(d["runs"]):
            r = d["runs"].astype(np.int64).copy()
            r[:, 0] = remap[r[:, 0]]
            runs.append(r)
            vals.append(d["vals"])
            flags.append(d["flags"])
        for k, v in m["stats"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stats[k] = stats.get(k, 0) + v
        for k, v in m["timing"].items():
            timing[k] = timing.get(k, 0.0) + v
    pairs = np.array(sorted(pair_ids, key=pair_ids.get), dtype=np.int32).reshape(-1, 2)
    base["pairs"] = pairs
    base["runs"] = (np.concatenate(runs) if runs else np.zeros((0, 3))).astype(np.int32)
    base["vals"] = np.concatenate(vals).astype(np.float32) if vals else np.zeros((0, 3), dtype=np.float32)
    base["flags"] = np.concatenate(flags).astype(np.uint8) if flags else np.zeros(0, dtype=np.uint8)
    stats["pairs"] = len(pairs)
    stats["chunks"] = len(paths)
    meta.update(stats=stats, timing=timing, frame_range=[0, meta["n_frames"]])
    return base, meta


def load_raw(path):
    with np.load(path, allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    raw.update(meta)
    return raw


def resolve_classes(raw, bodies, task_entry):
    """Body classes from the cached pass's per-body facts and the current class rules."""
    side = {}
    if bodies and Path(bodies).is_file():
        side = {k: v.get("class") for k, v in (json.loads(Path(bodies).read_text()).get("objects") or {}).items()
                if v.get("class")}
    names = list(raw["names"])
    deforming, scaling = set(raw["deforming"]), set(raw["scale_changing"])
    return C.seed_classes(names, list(raw["identities"]), sidecar_classes=side,
                          deforming=[n in deforming for n in names],
                          scale_changing=[n in scaling for n in names], task_entry=task_entry)


def classes_compatible(measured_classes, classes) -> bool:
    """The cached pass skipped the contact pairs of bodies that could not take part in contact when it ran. It still
    covers the current classes unless one of those bodies can now (fluid -> solid)."""
    return not any(old not in C.CONTACT_CLASSES and new in C.CONTACT_CLASSES
                   for old, new in zip(measured_classes, classes))


def geometry(evaluation_dir, out, bodies, task_entry, inp: Input, settings: Settings):
    request = geometry_request(evaluation_dir, bodies, task_entry, inp)
    cache_key = geometry_key(request, evaluation_dir, bodies, inp)
    raw_path, key_path = out / "floating_raw.npz", out / "floating_raw.key"
    if not settings.recompute and raw_path.is_file() and key_path.is_file() and key_path.read_text() == cache_key:
        raw = load_raw(raw_path)
        classes, reasons = resolve_classes(raw, bodies, task_entry)
        if classes_compatible(raw["classes"], classes):
            raw["classes"], raw["class_reasons"] = classes, reasons
            return raw
    tmp = out / "floating_raw.tmp.npz"
    n_frames = len(json.loads((Path(evaluation_dir) / "index.json").read_text())["frames"])
    chunks = chunk_count(settings, n_frames)
    if chunks == 1:
        req_path = out / "floating_request.json"
        req_path.write_text(json.dumps(request))
        run_child(settings, "_support_child.py", [req_path, tmp])
        req_path.unlink()
    else:
        bounds = [round(n_frames * k / chunks) for k in range(chunks + 1)]
        jobs = []
        for k in range(chunks):
            if bounds[k] == bounds[k + 1]:
                continue
            rp, tp = out / f"floating_request.chunk{k}.json", out / f"floating_raw.chunk{k}.npz"
            rp.write_text(json.dumps(dict(request, frame_range=[bounds[k], bounds[k + 1]])))
            jobs.append((f"chunk{k}", [rp, tp]))
        run_children(settings, "_support_child.py", jobs, out)
        arrays, meta = merge_chunks([tp for _, (_, tp) in jobs])
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        for _, (rp, tp) in jobs:
            rp.unlink()
            tp.unlink()
    tmp.replace(raw_path)
    key_path.write_text(cache_key)
    return load_raw(raw_path)


def moving_mask(states, names, n):
    mask = np.zeros((len(names), n), dtype=bool)
    for b, name in enumerate(names):
        for s, e in states["moving_intervals"].get(name, ()):
            mask[b, s:e + 1] = True
    return mask


def apply_rules(raw, states, bodies, baseline, params=None) -> dict:
    """The support rules on a measured (or cached) geometry pass: {primary, diagnostics, config}."""
    n = int(raw["n_frames"])
    moving = moving_mask(states, raw["names"], n)
    struct, struct_info = structure.pairs(raw["names"], raw["identities"], baseline_path=baseline,
                                          sidecar_path=bodies)
    sc = R.Scene(raw, moving, struct, {**R.DEFAULT_PARAMS, **(params or {})})
    anchors = R.find_anchors(sc)
    primary = R.evaluate(sc)
    classes = raw["classes"]
    class_lists = {c: sorted(nm for nm, k in zip(raw["names"], classes) if k == c)
                   for c in (C.CARRIER, C.AGENT, C.FLUID, C.TEXT, C.EXEMPT)}
    headline = ("floating_rate", "floating_body_frames", "animated_body_frames", "static_grounding_failure_rate",
                "static_ungrounded_body_frames", "static_body_frames", "strict_floating_rate",
                "strict_floating_body_frames", "floating_frame_rate", "floating_frames", "total_frames",
                "floating_frame_ranges", "all_floating_frame_rate", "static_floating_bodies")
    diagnostics = {k: primary.pop(k) for k in list(primary) if k not in headline}
    diagnostics["anchors"] = anchors
    diagnostics["classes"] = {"source": "automatic (export material classes, name tokens, deformation, tasks.json)",
                              "counts": {c: len(v) for c, v in class_lists.items()},
                              **{c: v[:80] for c, v in class_lists.items()},
                              "reasons": dict(list(raw["class_reasons"].items())[:200])}
    diagnostics["structure"] = struct_info
    diagnostics["bodies"] = {"total": len(raw["names"]),
                             "evaluated": int((sc.scored & ~sc.anchor & sc.active0.any(axis=1)).sum()),
                             "open_meshes": raw["open_meshes"],
                             "deforming": len(raw["deforming"]), "scale_changing": len(raw["scale_changing"])}
    return {"primary": primary, "diagnostics": diagnostics,
            "config": {"params": sc.p, "slope_deg": SLOPE_DEG, "query_mm": QUERY_M * 1000,
                       "sample_mm": SAMPLE_M * 1000, "max_samples": MAX_SAMPLES,
                       "rules_version": R.RULES_VERSION, "geometry_version": raw["child_version"]},
            "stats": raw["stats"]}


def run(inp: Input, out: Path, settings: Settings) -> dict:
    """Score one input; writes `out/floating.json` and returns it."""
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    evaluation_dir, bodies = evaluation_of(inp, out, settings)
    task_entry = C.task_entry(inp.task)
    states = motion_states(evaluation_dir, out, settings)
    raw = geometry(evaluation_dir, out, bodies, task_entry, inp, settings)
    result = {"schema": SCHEMA,
              "input": {"path": str(inp.path), "kind": inp.kind, "task": inp.task, "task_entry": task_entry,
                        "evaluation_dir": str(evaluation_dir), "exported": inp.evaluation_dir is None,
                        "fps": raw["fps"], "frame_start": raw.get("frame_start"), "input_digest": raw["input_digest"]},
              **apply_rules(raw, states, bodies, inp.baseline)}
    result["timing"] = {"geometry_cpu_s": raw["timing"].get("total_s"), "total_s": time.perf_counter() - t0}
    (out / "floating.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    return result
