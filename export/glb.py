"""Export a run, a published film, or a .blend to GLB (`python -m dycraft.export glb`).

A film is exported from its own Blender project. An object or scene run is first built into a Blender scene from its
committed IR, the way a still of it is drawn (`render/_blender_child.py --build-blend`), and then goes the same way.
The Blender half lives in `_blender/` and runs in the Blender that `install blender` sets up: the official binary when
there is one, since it can start parallel workers, else the bpy wheel's interpreter, one process. The optional
browser step (`--web`) runs gltf-transform under Node from `web/`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLENDER_SCRIPTS = HERE / "_blender"
WEB_DIR = HERE / "web"
#: The settings the project page's scenes use; see web/web_glb.mjs.
WEB_ARGS = ("--deflat", "--weights16", "--neardup", "--normal-size", "512", "--mr-size", "512")
#: Lines from the Blender side worth showing while it works.
_PROGRESS = re.compile(r"^(morphs:|memory:|visibility:|baking|baked|prepared|devices|\[\d+/\d+\]|hide_render|glass)")


class ExportError(Exception):
    """A refusal or failure, with the one-line reason a reader acts on."""


@dataclass
class Source:
    kind: str                   # "animation", "object", or "scene"
    name: str                   # the output's stem
    out_dir: Path               # where the GLB goes by default
    blend: Path | None = None   # a film's blend
    run_dir: Path | None = None  # an object or scene run's own folder (the one holding ir.json)
    ir: dict | None = None


def _internal(path: Path) -> Path:
    """The folder holding a run's records: `src/` in a run laid out for readers, else the run itself."""
    from ..run_layout import internal_dir
    return internal_dir(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def resolve(path: Path) -> Source:
    """What `path` exports, or an `ExportError` saying why it can't.

    Accepts a `.blend`; a run folder in either layout; a published film folder (`films/<id>/`, whose blend is at
    `bundle/animation.blend` with its caches beside it).
    """
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".blend":
            raise ExportError(f"{path} is not a .blend, a run folder, or a film folder")
        return Source("animation", path.stem, path.parent, blend=path)
    if not path.is_dir():
        raise ExportError(f"{path} does not exist")
    inner = _internal(path)
    for blend in (path / "animation.blend", inner / "bundle" / "animation.blend"):
        if blend.is_file():
            return Source("animation", "animation", path, blend=blend)
    meta = _read_json(inner / "meta.json") or {}
    if meta.get("layer") == "plain":
        raise ExportError(f"{path} is a plain run: it made no geometry to export")
    ir = _read_json(inner / "ir.json")
    if isinstance(ir, dict) and isinstance(ir.get("instances"), list):
        return Source("scene", "scene", path, run_dir=inner, ir=ir)
    if isinstance(ir, dict) and isinstance(ir.get("parts"), list):
        return Source("object", "object", path, run_dir=inner, ir=ir)
    if meta.get("layer") == "animation" or (inner / "native").is_dir():
        raise ExportError(f"{path} has no delivered film yet: an animation run is exported from the "
                          f"animation.blend it delivers")
    raise ExportError(f"{path} has nothing to export: no animation.blend and no committed ir.json")


# ---------------------------------------------------------------------------------------------------------------- Blender


def _blender_command(script: Path, args: list[str]) -> tuple[list[str], dict, bool]:
    """The command running `script` in Blender, its environment, and whether workers can be started."""
    from ..render._blender_env import blender_binary, blender_child_env, blender_python
    binary = blender_binary()
    if binary:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        return ([binary, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                 "--", *args], env, True)
    python = blender_python()
    if python:
        return [python, str(script), *args], blender_child_env(), False
    raise ExportError("no Blender found: run `python -m dycraft.install blender` first")


def _run_blender(script: Path, args: list[str], *, jobs: int | None, log, label: str) -> None:
    command, env, parallel = _blender_command(script, args)
    if jobs is not None:
        env["MORPH_WORKERS"] = env["BAKE_WORKERS"] = str(max(1, jobs))
    elif not parallel:
        env["MORPH_WORKERS"] = env["BAKE_WORKERS"] = "1"
    tail: list[str] = []
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace") as proc:
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-25:]
            if _PROGRESS.match(line):
                log(f"[export] {label}: {line}")
        code = proc.wait()
    if code:
        raise ExportError(f"Blender failed while {label} (exit {code}):\n" + "\n".join(tail[-12:]))


def _build_ir_blend(source: Source, out_blend: Path, scratch: Path) -> None:
    """An object or scene run's committed IR, built into a .blend the way a still of it is drawn."""
    from ..render import RenderRequest
    from ..render._world import _prepare
    from ..render.blender_renderer import _part_spec
    from ..render._blender_env import blender_child_env, blender_python
    catalog, base = None, source.run_dir
    if source.kind == "scene":
        from ..agent.layers.scene_context import iter_catalog_ir_paths
        catalog_dir = source.run_dir / "catalog"
        if not catalog_dir.is_dir():
            raise ExportError(f"{source.run_dir} has a scene but no catalog/ of the objects it places")
        catalog = {}
        for p in iter_catalog_ir_paths(catalog_dir):
            obj = _read_json(p)
            if isinstance(obj, dict):
                catalog[str(obj.get("id") or p.stem)] = obj
        base = catalog_dir
    request = RenderRequest(layer=source.kind, ir=source.ir, catalog=catalog, target="all",
                            out_dir=scratch / "unused", step=0, backend="blender", asset_base_dir=base)
    plan = _prepare(request)
    parts = [_part_spec(p, scratch, f"p{i}") for i, p in enumerate(plan.world_parts)]
    job = {"engine": "CYCLES", "device": "CPU", "samples": 1, "width": 64, "height": 64,
           "background": [1.0, 1.0, 1.0], "frames": [{"parts": parts, "cameras": []}]}
    job_path, result_path = scratch / "build_job.json", scratch / "build_result.json"
    job_path.write_text(json.dumps(job))
    python = blender_python()
    if not python:
        raise ExportError("no Blender found: run `python -m dycraft.install blender` first")
    child = Path(__file__).resolve().parents[1] / "render" / "_blender_child.py"
    ran = subprocess.run([python, str(child), "--build-blend", str(job_path), str(out_blend), str(result_path)],
                         env=blender_child_env(), capture_output=True, text=True)
    result = _read_json(result_path) or {}
    if not result.get("ok"):
        detail = result.get("error") or "\n".join((ran.stderr or ran.stdout or "").strip().splitlines()[-10:])
        raise ExportError(f"could not build the {source.kind} in Blender: {detail}")


# ---------------------------------------------------------------------------------------------------------------- web


def _web(glb: Path, out: Path, scratch: Path) -> None:
    node = shutil.which("node")
    if not node:
        raise ExportError("--web needs Node.js 20 or later on PATH (https://nodejs.org); the plain GLB is written")
    if not (WEB_DIR / "node_modules" / "@gltf-transform" / "core").is_dir():
        raise ExportError(f"--web needs its packages once: npm --prefix {WEB_DIR} ci   (the plain GLB is written)")
    resampled = scratch / "resampled.glb"
    for command in ([node, str(WEB_DIR / "resample_safe.mjs"), str(glb), str(resampled)],
                    [node, str(WEB_DIR / "web_glb.mjs"), str(resampled), str(out), *WEB_ARGS]):
        ran = subprocess.run(command, capture_output=True, text=True, cwd=str(WEB_DIR))
        if ran.returncode:
            raise ExportError("the browser step failed:\n" + (ran.stderr or ran.stdout).strip()[-1500:])


# ---------------------------------------------------------------------------------------------------------------- export


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s} s" if m else f"{s} s"


def export(path, out=None, *, bake_materials=False, web=False, jobs=None, log=print) -> dict:
    """Export `path` (see `resolve`) to a GLB; return what was written.

    `out` defaults to `<kind>.glb` beside the input. `--web` also writes `<out>.web.glb`.
    """
    started = time.time()
    source = resolve(Path(path))
    out = Path(out).expanduser().resolve() if out else source.out_dir / f"{source.name}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dycraft-export-") as td:
        scratch = Path(td)
        blend, original = source.blend, None
        if source.kind in ("object", "scene"):
            blend = scratch / f"{source.name}.blend"
            log(f"[export] building the {source.kind} in Blender from {source.run_dir / 'ir.json'}")
            _build_ir_blend(source, blend, scratch)
        if bake_materials:
            baked = scratch / "baked.blend"
            log("[export] baking procedural materials into textures (Cycles; a GPU is used when there is one)")
            _run_blender(BLENDER_SCRIPTS / "bake.py", [str(blend), str(baked)], jobs=jobs, log=log, label="baking")
            blend, original = baked, blend
        log(f"[export] exporting {blend if original is None else original} -> {out}")
        report_path = scratch / "report.json"
        args = [str(blend), str(out), "--report", str(report_path)] + (["--source", str(original)] if original else [])
        _run_blender(BLENDER_SCRIPTS / "export_glb.py", args, jobs=jobs, log=log, label="exporting")
        report = _read_json(report_path) or {}
        if web:
            web_out = out.with_name(out.stem + ".web.glb")
            log("[export] compressing for browsers (gltf-transform: meshopt geometry, WebP textures)")
            _web(out, web_out, scratch)
            report["web"] = str(web_out)
            report["web_bytes"] = web_out.stat().st_size
    report["seconds_total"] = round(time.time() - started, 1)
    frames = report.get("frames") or 1
    what = (f"{frames} frames at {report.get('fps', 0):g} fps, " if frames > 1 else "") + \
        f"{report.get('objects', 0)} meshes, {report.get('cameras', 0)} cameras"
    log(f"[export] wrote {out} ({_mb(out.stat().st_size)}; {what}) in {_duration(report['seconds_total'])}")
    if report.get("cuts"):
        log(f"[export] camera cuts and zooms, which glTF cannot hold: {report['cuts']}")
    if report.get("web"):
        log(f"[export] for browsers: {report['web']} ({_mb(report['web_bytes'])})")
    flat, objects = report.get("flattened_materials") or [], report.get("flattened_objects") or []
    if flat and not bake_materials:
        shown = ", ".join(objects[:5]) + (f" and {len(objects) - 5} more" if len(objects) > 5 else "")
        log(f"[export] {len(objects)} objects have procedural materials, exported as one flat color each "
            f"({shown}); --bake-materials bakes them into textures (a GPU is strongly advised)")
    return report
