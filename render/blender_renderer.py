"""Blender/Cycles backend for `dycraft.render`, the photoreal alternative.

Same contract as `open3d_renderer`: `render(RenderRequest) -> RenderResult`, the same PNG filenames, cache, and
`RenderResult` fields, and a different picture (shadows, no black edge cage, an environment). Opt in with
`--render-backend blender` or `$DYCRAFT_RENDER_BACKEND=blender`.

**This module composes nothing.** World geometry, camera framing, material packing, UV projection, the room
cutaway, and the render cache all come from `_world.py`, as they do for Open3D, so the two backends can disagree
about lighting and never about where the door is. This file resolves the interpreter, hardlinks cache hits, writes
`job.json` + `.npy` sidecars, spawns `_blender_child.py` under the `bpy` env, and reads `result.json`; the child
imports `bpy` and nothing from `dycraft`.

  * **Cost.** At 1280x720 / 64 samples on an RTX 3090: six object views 4.7 s (Open3D 2.9 s), six scene views
    8.9 s (3.0 s), a 12-frame scene clip 23.0 s (4.1 s). The fixed cost per child call (interpreter, `bpy`
    import, scene build) is ~1.8 s and covers every camera and frame of a request; the marginal cost is ~0.5 s
    per object view and ~1.2 s per scene view, the difference being the scene rebuilding its cutaway walls per
    camera and its instance meshes per frame.
  * **`RenderUnavailable`** if there is no `.blender_python`. Every caller handles it (the review images report
    it, `finalize_render_result` still writes the URDF), so a box without the env degrades rather than fails.
  * **It is not byte-reproducible.** Two renders of one scene differ by at most 1/255 on a few pixels in ten
    thousand, from the OptiX denoiser's reduction order, so a test asserts on what is in the picture, not on a
    golden PNG. Open3D is byte-reproducible, which is why `RENDER_CONFIG_VERSION` stays Open3D's and this module
    versions itself with `BLENDER_CONFIG_VERSION`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import (
    RENDER_HEIGHT,
    RENDER_WIDTH,
    RenderRequest,
    RenderResult,
    RenderUnavailable,
    register_backend,
)
from .. import _oom
from ._blender_env import blender_child_env, blender_python
from ._world import (
    _materialize,
    _prepare,
    _triplanar_uvs,
    box_mesh_arrays,
    plan_outputs,
    room_cutaway_planes,
    room_cutaway_slots,
    room_wall_boxes,
)
from ..sdk import _primitives as _prim
from ..sdk.ir import is_native_scene_schema
from ..sdk._material import SHADER_UV_SCALE_M


# ---------------------------------------------------------------------------
# Render recipe
# ---------------------------------------------------------------------------

#: This backend's own cache-key version, keyed in place of the package default
#: (which is Open3D's). Bump when anything below changes the picture; Open3D's
#: stored renders are untouched because the backend name is already in the key.
BLENDER_CONFIG_VERSION = "b4-nearclip-0.1mm"

_ENGINE = "CYCLES"
_SAMPLES = int(os.environ.get("DYCRAFT_BLENDER_SAMPLES", "64"))
_DEVICE = os.environ.get("DYCRAFT_BLENDER_DEVICE", "OPTIX")
_BG_COLOR = (0.95, 0.95, 0.95)
#: Seconds for one child call. Generous: a clip hands the child every frame at
#: once, so this bounds a whole sequence, not one picture.
_TIMEOUT_S = float(os.environ.get("DYCRAFT_BLENDER_TIMEOUT_S", "3600"))
#: Upper bound on concurrent children; see `_worker_count`. Override per run
#: with `DYCRAFT_BLENDER_WORKERS`.
_MAX_WORKERS = 3

_CHILD = Path(__file__).resolve().parent / "_blender_child.py"


# ---------------------------------------------------------------------------
# Job spec assembly
# ---------------------------------------------------------------------------

def _se3(se3) -> dict:
    t, q = se3
    return {"t": [float(v) for v in t], "q": [float(v) for v in q]}


def _box_spec(spec: dict, size, center, material, scratch: Path,
              tag: str) -> dict:
    """Fill `spec` as a parent-built box: vertices, triangles, and its UVs.

    The box is generated and parameterized HERE, not in the child, and that is
    the load-bearing choice in this module. A box is the one shape the renderer
    synthesizes rather than loads — an analytic part, an asset's AABB fallback,
    a room surface — so it has no authored UVs and something must project them.
    Doing it parent-side means both backends call the same `_triplanar_uvs` on
    the same vertices: a tiled oak floor cannot come out with different grain
    depending on who drew it. A child that projected its own would be the
    second implementation this whole module exists to avoid.

    UVs only when there is something to sample. `scale_m` absent means the
    material is untextured or was never sized, and `_apply_texture_uvs` returns
    early in exactly that case — **unless the material carries a shader**,
    which has no `Mapping` to be sized by and still needs coordinates to read.
    A graph on an unparameterized box renders one flat color with nothing
    saying why, so the shader case projects at 1 m and lets the graph scale
    itself (`sdk._material.SHADER_UV_SCALE_M`).
    """
    verts, tris = box_mesh_arrays(size, center)
    vp = scratch / f"{tag}_v.npy"
    tp = scratch / f"{tag}_t.npy"
    np.save(vp, verts)
    np.save(tp, tris)
    spec["kind"] = "mesh"
    spec["verts_npy"] = str(vp)
    spec["tris_npy"] = str(tp)

    scale_m = (material.get("mapping") or {}).get("scale_m")
    if scale_m is None and material.get("shader"):
        scale_m = SHADER_UV_SCALE_M
    if scale_m is not None:
        up = scratch / f"{tag}_uv.npy"
        np.save(up, _triplanar_uvs(verts, tris, scale_m))
        spec["uvs_npy"] = str(up)
    return spec


def _instance_of(part_name: str) -> str:
    """The instance a scene-layer world part belongs to.

    Scene parts are named `"<instance>/<part>"` (`_world._materialize`);
    object-layer parts carry no instance and answer "".
    """
    name = part_name or ""
    return name.split("/", 1)[0] if "/" in name else ""


def _part_spec(part, scratch: Path, tag: str, cutaway_slots=None) -> dict:
    """One `_WorldPart` as JSON the child can build without dycraft.

    Mirrors `open3d_renderer._part_mesh`'s dispatch exactly — box, deformed
    asset, rest asset, unknown-primitive fallback — because the two backends
    must draw the same *shape*. Only the mesh library differs.
    """
    material = part.material or {}
    prim = getattr(part, "primitive", "box")
    spec = {
        "name": part.name,
        "se3": _se3(part.se3),
        "material": material,
    }
    # Authored architecture only. These are the surfaces this part's instance
    # was declared to replace; the child builds one switchable mask slot per
    # entry and `cutaway_hidden` below decides which are live per camera.
    slots = (cutaway_slots or {}).get(_instance_of(part.name))
    if slots:
        spec["cutaway_slots"] = slots

    if prim == "box":
        return _box_spec(spec, part.size, part.center, material, scratch, tag)

    asset_key = _prim.render_asset_key(prim)
    obj_path = part.params.get(asset_key) if asset_key else None
    deformed = getattr(part, "deformed", None)

    if prim in _prim.ASSET_PRIMITIVES and deformed is not None:
        verts, tris = deformed
        vp = scratch / f"{tag}_v.npy"
        tp = scratch / f"{tag}_t.npy"
        np.save(vp, np.asarray(verts, dtype=np.float64))
        np.save(tp, np.asarray(tris, dtype=np.int32))
        spec["kind"] = "mesh"
        spec["verts_npy"] = str(vp)
        spec["tris_npy"] = str(tp)
        # LBS moves vertices and leaves topology alone, so the rest mesh's
        # per-corner UVs stay valid — but they live in the OBJ this branch
        # otherwise skips, and arrays carry none. Without them a textured skin
        # samples a single texel and renders in one flat color, which reads as
        # "this asset has no texture" rather than as a bug.
        spec["uv_from_obj"] = str(obj_path) if obj_path else None
        # Normals are deliberately NOT carried from the rest asset: a solid's
        # crease-split normals describe the *rest* surface and would light a
        # bent panel as though it were still flat.
        return spec

    if prim in _prim.ASSET_PRIMITIVES and obj_path:
        spec["kind"] = "asset"
        spec["obj"] = str(obj_path)
        spec["scale"] = [float(v) for v in part.params.get("scale", (1.0, 1.0, 1.0))]
        # The AABB fallback travels WITH the asset, so the child can degrade to
        # it when the file will not load. The parent cannot decide that here —
        # a path that exists as a string may still be missing, unreadable, or
        # malformed on disk, and only the importer finds out.
        #
        # Not defensive plumbing: caught by rendering a real clip with the wrong
        # `asset_base_dir`. Open3D drew every part as its bounding box (visibly
        # wrong, obviously a scene), and Blender drew an **empty room** —
        # because a dropped part leaves nothing at all behind. Losing the object
        # is strictly worse than drawing a crude one, and it is the failure this
        # module's docstring already promised not to have.
        spec["fallback"] = _box_spec({}, part.size, part.center, material,
                                     scratch, f"{tag}_fb")
        return spec

    # Unknown primitive, or an asset tag with no file at all: the bounding box,
    # placed at its true (possibly off-center) local AABB center. Same fallback
    # Open3D takes, so a broken asset produces the same wrong-but-legible
    # picture on both backends rather than two different ones.
    return _box_spec(spec, part.size, part.center, material, scratch, tag)


def _frame_spec(plan, outputs, scratch: Path, index: int,
                request=None) -> dict | None:
    """One request's parts, room, and to-be-drawn cameras. None if fully cached."""
    todo = [o for o in outputs if o.cached_src is None]
    if not todo:
        return None

    cutaway_slots = room_cutaway_slots(plan.room_info)
    parts = [_part_spec(p, scratch, f"f{index}_p{i}", cutaway_slots)
             for i, p in enumerate(plan.world_parts)]

    cameras = []
    for out in todo:
        cam = out.camera
        # Room surfaces are boxes like any other, so they go through the same
        # parent-side generation and projection — the floor's tiling has to
        # match a floor-coloured box part's, and it only does if one function
        # laid out both.
        walls = []
        for wi, w in enumerate(room_wall_boxes(cam, plan.room_info)):
            wall = {"tag": w.tag, "se3": _se3(w.se3),
                    "material": w.material, "alpha": w.alpha,
                    # Exists for lighting, may be hidden from this camera.
                    # See `_world.WallBox`.
                    "camera_visible": w.camera_visible}
            walls.append(_box_spec(wall, w.size, (0.0, 0.0, 0.0), w.material,
                                   scratch, f"f{index}_{cam.id}_w{wi}"))
        # The authored counterpart of a WallBox's `camera_visible`: which
        # replaced surfaces stand between THIS eye and the room. Per camera for
        # the same reason the walls are.
        hidden = sorted({h["tag"]
                         for hs in room_cutaway_planes(
                             cam, plan.room_info).values()
                         for h in hs})
        cameras.append({
            "id": cam.id,
            "eye": [float(v) for v in cam.eye],
            "cutaway_hidden": hidden,
            "target": [float(v) for v in cam.target],
            "up": [float(v) for v in cam.up],
            "fov_deg": float(cam.fov_deg),
            "walls": walls,
            "out": str(out.out_path),
        })

    # The lighting rig is sized from these, per frame. `framing` is the bbox the
    # cameras were placed from, so the rig scales with whatever is actually in
    # shot; `room` (when the scene has one) switches the child to an interior
    # rig, because a studio rig sits outside its subject and the walls of a room
    # are opaque — measured, three of the four fixed lights fell outside this
    # scene's 5.4x5.0x3.0 m room and the world dome was left supplying 71% of
    # the illumination.
    lo, hi = plan.target_bbox
    out = {"parts": parts, "cameras": cameras,
           "framing": [[float(v) for v in lo], [float(v) for v in hi]]}
    # `room` selects the interior rig, so it is shipped only for a room that
    # actually encloses something. An open-air extent is a ground plane and a
    # framing box, not an interior: lighting it from three area lights hung
    # under a ceiling it does not have puts the light in the wrong place and
    # leaves the sky out.
    if plan.room_info is not None and plan.room_info.get("capped", True):
        rlo, rhi = plan.room_info["aabb"]
        out["room"] = [[float(v) for v in rlo], [float(v) for v in rhi]]

    if request is not None:
        out.update(_lighting_frame_keys(request))
    return out


def _lighting_frame_keys(request) -> dict:
    """The lighting-related frame keys for one request.

    A native scene (v2 or later) ships its authored `@lighting` block and the
    paths its managed images resolve against. Any other IR (an object) ships no
    lighting keys, and the child lights it with its automatic rig.

    Split out of `_frame_spec` so the branch is unit-testable without a render.
    """
    ir = request.ir if isinstance(request.ir, dict) else {}
    out: dict = {}
    if is_native_scene_schema(ir.get("schema")):
        # v2-OR-LATER, not `== v2`: v3 is v2 plus per-instance joint state.
        out["schema_v2"] = True
        lighting = ir.get("lighting")
        if lighting:
            out["lighting"] = dict(lighting)
        out["run_dir"] = str(request.out_dir.parent if request.out_dir else "")
        if request.asset_base_dir:
            out["asset_base_dir"] = str(request.asset_base_dir)
    return out


# ---------------------------------------------------------------------------
# Child invocation
# ---------------------------------------------------------------------------

def _run_child(job: dict, scratch: Path, tag: str = "") -> dict:
    interp = blender_python()
    if not interp:
        raise RenderUnavailable(
            "no Blender interpreter: set $DYCRAFT_BLENDER_PYTHON or write one "
            "to .blender_python (see setup_blender.sh)")

    job_path = scratch / f"job{tag}.json"
    result_path = scratch / f"result{tag}.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    try:
        proc = subprocess.run(
            _oom.tiered([interp, str(_CHILD), str(job_path), str(result_path)],
                        _oom.WORKER),
            capture_output=True, text=True, timeout=_TIMEOUT_S,
            env=blender_child_env())
    except subprocess.TimeoutExpired:
        raise RenderUnavailable(
            f"blender child timed out after {_TIMEOUT_S:.0f}s") from None
    except OSError as e:
        raise RenderUnavailable(f"blender child could not start: {e}") from e

    if not result_path.exists():
        # No result file means the child died before it could report — an
        # import error, a segfault in bpy, an OOM. Its stderr is the only
        # evidence, so it goes in the message rather than to a log nobody reads.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise RenderUnavailable(
            f"blender child exited {proc.returncode} without a result:\n"
            + "\n".join(tail))

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("ok"):
        # `detail` carries the structured diagnostic — for a failed `@lighting`
        # block it is the block name, the exception, and a short traceback.
        # Dropping it would leave the agent with "render failed" and no way to
        # repair, which is the one outcome a non-fallback lighting contract
        # must not produce.
        detail = result.get("detail")
        raise RenderUnavailable(
            f"blender render failed: {result.get('error', 'unknown')}"
            + (f"\n{detail}" if detail else ""))
    return result


def _worker_count(n_frames: int) -> int:
    """How many Blender processes to run at once.

    **The GPU is idle most of the time, which is why this exists.** Sampled at
    1 Hz through a real clip, SM utilization alternates 99% / 0% with a mean of
    44%: roughly 0.8 s of Cycles, then ~1 s of CPU-only work — OBJ import, mesh
    construction, the depsgraph evaluation Blender does before it can hand
    anything to OptiX, the orphan purge, the PNG encode. One process therefore
    leaves the card idle over half the wall clock, and a second process's CPU
    phase can fill the first's GPU phase.

    Bounded by three things, all of them real:
      * CPU. A child measures ~250% (2.5 cores), so `nproc // 3` is what fits.
      * GPU memory. ~1.1 GB resident per child against 24 GB here — not the
        binding constraint, but it is the one that fails hardest, so it caps.
      * Diminishing returns. The ceiling is 1/0.44 ≈ 2.3x, so past ~3 workers
        the extra contention costs more than the overlap wins.

    Not applied to a single frame's worth of work: one child already renders
    every camera of a request, and splitting six stills across processes pays
    the ~1.8 s bpy startup several times over to save less than that.
    """
    override = os.environ.get("DYCRAFT_BLENDER_WORKERS")
    if override:
        try:
            return max(1, min(int(override), n_frames))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(1, min(_MAX_WORKERS, cpu // 3, n_frames))


def _run_children(base: dict, frames: list, scratch: Path) -> list:
    """Render `frames` through one or more concurrent Blender processes.

    Returns the union of every child's non-fatal warnings, first-seen order,
    deduplicated across children — the same message from two workers is one
    fact about a material, not two.

    Frames are independent — each is its own scene build and its own
    `render()` — so the split is a plain contiguous partition and needs no
    coordination. Contiguous rather than round-robin so each child renders a
    run of neighboring frames, which is what a future "reuse the scene between
    frames" optimization would want.

    An exception in any child propagates once every child has been waited on,
    so a failure cannot leave orphaned Blender processes holding GPU memory.
    """
    k = _worker_count(len(frames))
    if k <= 1:
        res = _run_child({**base, "frames": frames}, scratch)
        return (list(res.get("warnings") or ()),
                dict(res.get("view_settings") or {}))

    size = math.ceil(len(frames) / k)
    chunks = [frames[i:i + size] for i in range(0, len(frames), size)]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_run_child, {**base, "frames": c}, scratch, f"_w{i}")
                   for i, c in enumerate(chunks)]
        # `.result()` on every future before re-raising: the first failure must
        # not return control while siblings are still rendering.
        errors = []
        warnings: list = []
        view_settings: dict = {}
        for f in futures:
            try:
                res = f.result()
                for w in res.get("warnings") or ():
                    if w not in warnings:
                        warnings.append(w)
                # Every child in a batch shares one set of render settings and
                # one scene build, so the last one to report wins and they
                # agree by construction.
                if res.get("view_settings"):
                    view_settings = dict(res["view_settings"])
            except BaseException as e:      # noqa: BLE001
                errors.append(e)
    if errors:
        raise errors[0]
    return warnings, view_settings


def _dispatch(requests: list[RenderRequest]) -> list[RenderResult]:
    """Plan every request, hardlink the cache hits, draw the rest in one child.

    `render` passes one request.
    """
    plans = [_prepare(r) for r in requests]
    results = [RenderResult(target_bbox=p.target_bbox, cameras=p.cameras)
               for p in plans]

    all_outputs = [plan_outputs(r, p, config_version=BLENDER_CONFIG_VERSION)
                   for r, p in zip(requests, plans)]

    # Before anything writes. `_prepare` also does this, but depending on that
    # makes the directory a side effect of *planning* — and the first thing
    # below is a hardlink into it, which fails with a bare FileNotFoundError
    # naming a path nobody asked for.
    for req in requests:
        req.out_dir.mkdir(parents=True, exist_ok=True)

    # Cache hits first, so a fully-cached call never spawns a child at all.
    for req, outs, res in zip(requests, all_outputs, results):
        for out in outs:
            res.render_paths[out.camera.id] = out.out_path
            if out.cached_src is not None:
                _materialize(out.cached_src, out.out_path)
                res.cache_hits[out.camera.id] = True
            else:
                res.cache_hits[out.camera.id] = False

    with tempfile.TemporaryDirectory(prefix="dycraft-blender-") as tmp:
        scratch = Path(tmp)
        frames = []
        for i, (req, plan, outs) in enumerate(zip(requests, plans, all_outputs)):
            spec = _frame_spec(plan, outs, scratch, i, request=req)
            if spec is not None:
                frames.append(spec)

        child_warnings: list = []
        child_view_settings: dict = {}
        if frames:
            _w, _h = (getattr(requests[0], "resolution", None)
                      or (RENDER_WIDTH, RENDER_HEIGHT))
            base = {
                "engine": _ENGINE,
                "device": _DEVICE,
                "samples": _SAMPLES,
                # Per-request, not the module default: `resolution` is part of
                # `render_cache_key`, so a 640x360 selection render keyed as
                # such must not come back full-size from this backend while the
                # Open3D one honors it. `requests[0]` matches `png_compress_level`
                # above — one child batch is one set of render settings.
                "width": _w, "height": _h,
                "background": list(_BG_COLOR),
                "png_compress_level": requests[0].png_compress_level,
                # One child batch is one set of render settings, so `any` is
                # the safe reading: if any request in the batch wants strict
                # materials, none of its images may silently fall back.
                "strict_materials": any(
                    bool(getattr(r, "strict_materials", False))
                    for r in requests),
            }
            child_warnings, child_view_settings = _run_children(
                base, frames, scratch)

    # Broadcast to every result rather than attributed per frame. A child
    # renders a contiguous run of frames and a material warning is about the
    # MATERIAL, which every frame that drew the part shares — so narrowing it
    # to one request would be a guess dressed up as precision.
    if child_warnings:
        for res in results:
            res.warnings.extend(child_warnings)
    if child_view_settings:
        for res in results:
            res.view_settings = dict(child_view_settings)

    # Only now populate the cache: a key must never point at a PNG the child
    # failed to write, or every later render of that (ir, camera, pose) would
    # hardlink a missing file.
    for req, outs in zip(requests, all_outputs):
        if req.cache is None:
            continue
        for out in outs:
            if out.cached_src is None and out.out_path.exists():
                req.cache[out.key] = out.out_path

    return results


# ---------------------------------------------------------------------------
# Backend entry points
# ---------------------------------------------------------------------------

def render(request: RenderRequest) -> RenderResult:
    """Render one request's cameras. Raises `RenderUnavailable` if bpy is absent."""
    return _dispatch([request])[0]


register_backend("blender", render)
