"""Derive a viewer-only asset from a part's real mesh and its real shader.

`Material(shader=...)` makes a Cycles node graph the sole authority for appearance, and the browser cannot evaluate
one. So this **samples** it: unwrap a packed atlas on the real mesh, evaluate the original graph at every surface
point, write the result into that point's own texel, and hand the browser a GLB it renders locally at full frame
rate. Against `blender_renderer.render()`, the baked asset's worst view differs by a few tenths of a level per
channel out of 255, while the live Blender stream differs by more than half a level from JPEG alone, so the
derived asset is closer to the canonical image than streaming it, and needs no GPU at view time.

This is the parent half: it decides what to bake and where the result lives. `_blender_child.py --bake` does the
baking (`_bake_run` there lists the three things that are silent when wrong).

**Nothing canonical moves.** The job is packed from the same `_prepare` and `_part_spec` a still uses, so the
geometry and shader are the ones the canonical render draws. The IR gains no key, so `sdk.ir.content_hash` and
every object id are unchanged; the staged `assets/meshes/<digest>.obj` are only read; `render/urdf_export.py`, the
stored `renders/ir.urdf` and the scene export `h.ik` uses are not involved. The derived assets live in a cache
**outside the run directory**, keyed by the IR's id, and only the viewer addresses them.

The bake happens on the REST shape. It also writes each part's atlas UVs as a sidecar (`BAKE_UV_SIDECAR_SUFFIX` in
`_blender_child.py`, `atlas_uvs` below), which the viewer's native playback reads
(`viewer/native_animation.py::atlas`).

**The one divergence from the canonical render.** Cycles evaluates a shader driven from object space
(`tc.outputs["Object"]`, `Generated`, `Position`) at the deformed position, so its pattern swims through a bending
surface; a baked map is pinned to the surface and travels with it, as a patterned cushion being compressed should.
Measured on a deformed frame: 1.135/255, against 0.238 for a UV-driven shader, which agrees exactly.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import RenderRequest, RenderUnavailable
from .. import _oom
from ._blender_env import blender_child_env, blender_python
from ._world import _prepare

_CHILD = Path(__file__).resolve().parent / "_blender_child.py"

#: Mirrors `_blender_child.BAKE_VERSION`. Duplicated rather than imported
#: because this module must not import anything from the bpy interpreter, and
#: asserted equal by `tests/test_viewer_bake.py` so the two cannot drift.
BAKE_VERSION = "vb3"

#: A user cache, because derived bytes are reproducible from the IR, are large,
#: and must never land in git or in a run directory.
CACHE_ENV_VAR = "DYCRAFT_VIEWER_ASSET_CACHE_DIR"
_DEFAULT_CACHE = Path.home() / ".cache" / "dycraft" / "viewer_assets"

MANIFEST_NAME = "manifest.json"

#: Generous. A 16-part object bakes in 45-95 s on an RTX 3090 (64 bakes, four
#: maps per part), and this has to survive a CPU-only host.
TIMEOUT_S = float(os.environ.get("DYCRAFT_VIEWER_BAKE_TIMEOUT_S", "3600"))


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV_VAR)
    return Path(override).expanduser() if override else _DEFAULT_CACHE


def asset_dir(ir_id: str) -> Path:
    """Where one IR's derived assets live.

    Keyed by the IR's own content hash plus `BAKE_VERSION`. The hash already
    covers geometry and materials, so it is the whole cache key an object needs;
    the version is what re-bakes when the *recipe* changes rather than the
    object.
    """
    return cache_root() / f"{ir_id}_{BAKE_VERSION}"


def read_manifest(ir_id: str) -> dict | None:
    """The manifest for a complete bake, or None if there isn't one.

    A directory with no manifest is a bake that died partway; treating it as
    absent is what makes a retry work instead of serving half an object.
    """
    path = asset_dir(ir_id) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not manifest.get("ok"):
        return None
    return manifest


def asset_path(ir_id: str, filename: str) -> Path | None:
    """Resolve one derived file, refusing anything outside the cache entry."""
    root = asset_dir(ir_id).resolve()
    if not root.is_dir():
        return None
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def atlas_uvs(ir_id: str, manifest: dict, part_name: str):
    """`(uv, tri)` for one part's atlas layer, or None.

    `uv` is `(3T, 2)` float32 in the STAGED MESH's corner order and `tri` is
    `(T, 3)` int32 of its vertex indices — see
    `_blender_child.BAKE_UV_SIDECAR_SUFFIX` for why the sidecar exists and why
    the pairing is exact.

    Native playback needs this because it draws the evaluated mesh, not the
    GLB, and so never sees the GLB's own UVs. Returns
    None rather than raising for every ordinary absence — a part with no entry,
    a bake from before the sidecar existed, a truncated file — because the
    caller's fallback (the staged mesh's own UVs, and the preview appearance)
    is always available.
    """
    import numpy as np

    entry = next((e for e in manifest.get("parts") or ()
                  if e.get("part") == part_name), None)
    if entry is None or not entry.get("atlas_uv"):
        return None
    path = asset_path(ir_id, str(entry["atlas_uv"]))
    if path is None:
        return None
    try:
        with np.load(path) as data:
            uv = np.asarray(data["uv"], dtype=np.float64)
            tri = np.asarray(data["tri"], dtype=np.int64)
    except (OSError, ValueError, KeyError):
        return None
    if uv.ndim != 2 or uv.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        return None
    if len(uv) != 3 * len(tri):
        return None
    return uv, tri


def build(request: RenderRequest, *, force: bool = False) -> dict:
    """Bake `request`'s parts into the cache and return the manifest.

    Idempotent: an existing complete manifest is returned untouched unless
    `force`. Writes into a temporary directory and renames on success, so a
    crashed or killed bake cannot leave a partial entry that later looks
    complete.
    """
    ir_id = str(request.ir.get("id") or "")
    if not ir_id:
        raise RenderUnavailable(
            "cannot derive viewer assets for an IR with no id — the id is the "
            "cache key, and an unfinalized IR has none")
    if not force:
        existing = read_manifest(ir_id)
        if existing is not None:
            return existing

    interp = blender_python()
    if not interp:
        raise RenderUnavailable(
            "no Blender interpreter: set $DYCRAFT_BLENDER_PYTHON or write one "
            "to .blender_python (see setup_blender.sh)")

    target = asset_dir(ir_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dycraft-bake-") as tmp:
        scratch = Path(tmp)
        job_path = scratch / "job.json"
        result_path = scratch / "result.json"
        staging = scratch / "out"
        staging.mkdir()
        job_path.write_text(json.dumps(pack_bake_job(request, scratch)),
                            encoding="utf-8")

        try:
            proc = subprocess.run(
                _oom.tiered([interp, str(_CHILD), "--bake", str(job_path),
                             str(staging), str(result_path)], _oom.WORKER),
                capture_output=True, text=True, timeout=TIMEOUT_S,
                env=blender_child_env())
        except subprocess.TimeoutExpired:
            raise RenderUnavailable(
                f"viewer-asset bake timed out after {TIMEOUT_S:.0f}s") from None
        except OSError as e:
            raise RenderUnavailable(f"bake child could not start: {e}") from e

        if not result_path.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            raise RenderUnavailable(
                "viewer-asset bake died before it could report:\n  "
                + "\n  ".join(tail))
        manifest = json.loads(result_path.read_text(encoding="utf-8"))
        if not manifest.get("ok"):
            raise RenderUnavailable(
                "viewer-asset bake failed: "
                + str(manifest.get("error") or "no reason given"))
        if not manifest.get("parts"):
            raise RenderUnavailable(
                "viewer-asset bake produced no parts; nothing to show")

        manifest["ir_id"] = ir_id
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1),
                                             encoding="utf-8")
        # Rename last, and only over a directory we then remove: the manifest
        # arriving is what makes the entry readable, so a half-written bake is
        # never served.
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging), str(target))
    return manifest


def pack_bake_job(request: RenderRequest, scratch: Path) -> dict:
    """The parts to bake, packed by the canonical helpers.

    Deliberately much smaller than a render job: **no cameras, no lights, no
    framing, no room.** An `EMIT` bake returns the socket's own value and a
    `NORMAL` bake returns tangent-space geometry, so neither consults the
    lighting rig — packing one would be inventing a dependency the bake does not
    have. What must be identical to a still is the *geometry and the material*,
    and those come from `_prepare` and `_part_spec` unchanged.
    """
    from .blender_renderer import _ENGINE, _part_spec

    if request.layer != "object":
        # **Not a scope limit — the object layer is the only shape a bake has.**
        # A scene is baked as its catalog objects, one object-layer request
        # each, because a catalog entry IS an ObjectIR: see
        # `viewer/derived.resolve`, which is where a run fans out into the
        # objects it draws. So nothing needs a scene-shaped bake, and a request
        # that arrives with one is a caller mistake rather than a missing
        # feature.
        raise RenderUnavailable(
            f"viewer-asset bake takes one object at a time (got layer "
            f"{request.layer!r}); a scene is baked as its catalog objects")
    # An explicit camera short-circuits `_prepare`'s default-pose world rebuild,
    # which exists only to compute a framing number nothing here reads.
    from . import CameraSpec
    plan = _prepare(dataclasses.replace(
        request, camera_ids=None,
        cameras=(CameraSpec(id="bake", eye=(1.0, 0.0, 0.0),
                            target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)),)))
    # **Only parts the URDF draws as a `<mesh>`.**
    #
    # `object_ir_to_urdf` emits native `<box>` / `<cylinder>` / `<sphere>`
    # geometry for the analytic primitives, so there is no `filename` to
    # substitute and a GLB for one could never be reached. Baking it anyway
    # would spend GPU time producing an asset the viewer cannot address.
    #
    # This costs nothing in practice: across every run on disk, all 69
    # shader-bearing parts are `solid`. An analytic part is scalar-shaded, which
    # the existing viewer path already draws correctly.
    from ..sdk import _primitives as _prim

    bakeable = [
        (i, p, ir_part) for i, (p, ir_part)
        in enumerate(zip(plan.world_parts, request.ir.get("parts") or ()))
        if _prim.render_asset_key(ir_part.get("primitive")) is not None
    ]
    parts = [_part_spec(p, scratch, f"p{i}") for i, p, _ in bakeable]

    # **Carry the IR part name, rather than parsing it back out of the world
    # part's.** `_WorldPart.name` is `f"{object_name}/{part_name}"`, and
    # recovering the second half by splitting would be a guess that happens to
    # work — object names can contain a slash and instance names collide across
    # a scene. The two lists are index-aligned by construction
    # (`_build_object_world_parts` iterates `ir["parts"]` and never skips), and
    # `_build_part` ignores `spec["name"]` entirely, so this is free.
    ir_parts = request.ir.get("parts") or ()
    if len(plan.world_parts) != len(ir_parts):
        raise RenderUnavailable(
            f"IR has {len(ir_parts)} parts but the world plan has "
            f"{len(plan.world_parts)}; the derived asset could not be matched "
            "back to the IR and would be attached to the wrong geometry")
    for spec, (_, _, ir_part) in zip(parts, bakeable):
        spec["ir_part"] = ir_part.get("name")
    if not parts:
        raise RenderUnavailable(
            "no part of this object is drawn as a mesh, so there is nothing a "
            "derived viewer asset could replace")
    return {
        "engine": _ENGINE,
        "device": os.environ.get("DYCRAFT_BLENDER_DEVICE", "OPTIX"),
        "samples": 1,
        "width": 16, "height": 16,       # unused; `_reset_scene` wants them
        "frames": [{"parts": parts, "cameras": []}],
    }


def substituted_ir(ir: dict, manifest: dict, url_prefix: str) -> dict:
    """A COPY of `ir` whose parts point at their derived assets.

    Used only to generate the viewer's URDF. The original dict is never
    mutated — it is the object the viewer also reports, hashes, and shows in the
    code panel, and a rewritten mesh path leaking into any of those is exactly
    the kind of thing that turns a display change into a data change.

    A part with no derived asset keeps its staged mesh, so a partially bakeable
    object degrades to a mixture rather than losing geometry.
    """
    by_part = {e["part"]: e for e in manifest.get("parts") or ()}
    out = dict(ir)
    parts = []
    for part in ir.get("parts") or ():
        entry = by_part.get(part.get("name"))
        if entry is None or not isinstance(part.get("params"), dict):
            parts.append(part)
            continue
        params = dict(part["params"])
        params["mesh"] = f"{url_prefix}{entry['file']}"
        parts.append({**part, "params": params})
    out["parts"] = parts
    return out
