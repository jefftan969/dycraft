"""Stage a bare agent's delivered `.blend` as an animation bundle like a dycraft run's, with its film.

A bare agent authors freely, so its `.blend` has no DyCraft rig and none of the layer's checks apply
to it (`animation/blender/verify.py` needs the baseline a DyCraft run builds). What the benchmark
does with it is make it watchable beside a DyCraft run:

* the blend is copied, its external files are resolved to this host and packed, and the copy passes
  the layer's dependency audit, so the bundle is portable;
* `worker.evaluate` records its evaluated geometry on every frame (`bundle/evaluation/`), which the
  viewer plays in 3D as it plays a DyCraft run;
* the bundle's film, `bundle/film/clip.mp4`, named in `ir.json` as a dycraft run's is, is the agent's
  own when it delivered one, otherwise rendered here from the blend's own camera, frame range and
  frame rate with DyCraft's final-render defaults at 960x540 (`FILM_SETTINGS`), so a run that ran
  out of time before rendering still has a film to watch.

Scoring is separate: `python -m dycraft.benchmark.score` exports the staged blend itself, recording
the visibility, parenting, and materials this evaluation does not.

The command rebuilds a DyCraft run's evaluation instead, for a run that travels without it (the
evaluation is most of a bundle's files):

    python -m dycraft.benchmark.stage RUN_DIR [--film FILE]

`rebuild` evaluates `RUN_DIR/bundle/animation.blend` where it is and replaces `bundle/evaluation/`.
The declarations (events, collision allowances) and the scene reference come from the blend, as
they did in the run. Apart from a film it may add, nothing else in the bundle changes.

The Blender children open the agent's file with `use_scripts=False`, so no agent-authored code runs,
which is why no sandbox is needed here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLEND_NAME = "animation.blend"
FILM_NAME = "film/clip.mp4"
#: The layer a DyCraft run records in its `meta.json`.
DYCRAFT_LAYER = "animation"
#: A missing bare film is rendered with DyCraft's final-render defaults (`final_render_settings`:
#: Cycles, 8 samples denoised, the GPU when there is one) at a size typical of DyCraft's own films.
FILM_SETTINGS = {"width": 960, "height": 540}

#: Runs inside the Blender interpreter, with the animation layer's Blender modules importable the way
#: `animation/blender/runtime.py` binds them into `/tools`.
_CHILD = r'''
import importlib.util, json, os, sys, traceback
from pathlib import Path


def main():
    job = json.loads(Path(sys.argv[-1]).read_text())
    repo = Path(job["repo"])
    for sub in ("animation/blender", "verify", "render"):
        sys.path.insert(0, str(repo / sub))

    spec = importlib.util.spec_from_file_location(
        "worker", repo / "animation" / "blender" / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    import bpy

    blend = Path(job["blend"])
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)

    # A bare agent saves inside its sandbox, so its external files are named as `/workspace/...` or
    # relative to where its blend was. This copy sits elsewhere: point each reference at the file on
    # this host, pack what Blender can pack, and save the copy, so the audit below sees a portable file.
    relocate = job.get("relocate")
    if relocate:
        origin = Path(relocate["origin"])
        prefixes = relocate.get("prefixes") or {}

        def _host(fp):
            if fp.startswith("//"):
                fp = str(origin / fp[2:])
            for old, new in prefixes.items():
                old = old.rstrip("/")
                if fp == old or fp.startswith(old + "/"):
                    return new + fp[len(old):]
            return fp

        for attribute in ("images", "sounds", "fonts", "movieclips", "cache_files", "volumes",
                          "libraries"):
            for item in getattr(bpy.data, attribute, ()):
                fp = getattr(item, "filepath", "")
                if (not fp or fp.startswith("<")
                        or getattr(item, "packed_file", None) is not None):
                    continue
                try:
                    item.filepath = _host(fp)
                except Exception:
                    pass
        for operator in ("pack_all", "pack_libraries"):
            try:
                getattr(bpy.ops.file, operator)()
            except Exception as error:
                print(f"[stage] {operator}: {error}", file=sys.stderr)
        bpy.ops.wm.save_mainfile(filepath=str(blend))
    # The two guards the layer applies to its own agent: an unpacked external dependency will not
    # survive being moved into a bundle, and a transient physics cache will not survive reopening.
    worker.audit_dependencies()
    info = worker.evaluate({})

    out = {"evaluation": {k: info.get(k) for k in
                          ("fps", "n_frames", "frame_start", "complete", "format")}}
    print("STAGE_OK " + json.dumps(out, default=float))


try:
    main()
    code = 0
except BaseException:
    traceback.print_exc()           # the caller reports it, missing the result line
    code = 1
# Leave without Blender's teardown, which can deadlock after a blend with caches (Alembic, fluid, cloth) was
# opened; the result line is already out.
sys.stdout.flush(); sys.stderr.flush()
os._exit(code)
'''


_RENDER_CHILD = r'''
import json, os, sys, traceback
from pathlib import Path


def main():
    job = json.loads(Path(sys.argv[-1]).read_text())
    sys.path.insert(0, str(Path(job["repo"]) / "animation" / "blender"))
    import bpy
    import render_config

    # As for staging: no agent-authored code runs.
    bpy.ops.wm.open_mainfile(filepath=job["blend"], use_scripts=False)
    scene = bpy.context.scene
    if scene.camera is None:
        cameras = sorted((o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name)
        if not cameras:
            print("RENDER_FAILED the scene has no camera", flush=True)
            return
        scene.camera = cameras[0]
    actual = render_config.apply_final_render_settings(bpy, job["settings"])
    scene.render.use_file_extension = True
    scene.render.filepath = str(Path(job["frames"]) / "f_")
    bpy.ops.render.render(animation=True)
    print("RENDER_OK " + json.dumps({
        "camera": scene.camera.name, "frame_start": scene.frame_start, "frame_end": scene.frame_end,
        "fps": scene.render.fps / scene.render.fps_base, "settings": actual}, default=str), flush=True)


try:
    main()
    code = 0
except BaseException:
    traceback.print_exc()           # the caller reports it, missing the result line
    code = 1
# Leave without Blender's teardown, which can deadlock after a blend with caches (Alembic, fluid, cloth) was
# opened; the result line is already out.
sys.stdout.flush(); sys.stderr.flush()
os._exit(code)
'''


def render_film(blend: Path, dest: Path, *, settings: dict | None = None,
                timeout: int = 4 * 3600) -> dict:
    """Render `blend`'s animation to the H.264 file `dest` and return what drew it.

    The blend's own active camera (else its first camera by name), frame range and frame rate;
    DyCraft's final-render settings, `FILM_SETTINGS` unless given. Frames are drawn as PNGs in a
    temporary folder and encoded with ffmpeg.
    """
    from ..render._blender_env import blender_python

    interpreter = blender_python()
    if not interpreter:
        raise RuntimeError("no Blender interpreter: run `python -m dycraft.install blender`")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not on PATH")
    with tempfile.TemporaryDirectory(prefix="benchmark_film_") as td:
        frames, job = Path(td) / "frames", Path(td) / "job.json"
        frames.mkdir()
        script = Path(td) / "render.py"
        script.write_text(_RENDER_CHILD)
        job.write_text(json.dumps({"repo": str(REPO), "blend": str(Path(blend).resolve()),
                                   "frames": str(frames),
                                   "settings": dict(settings or FILM_SETTINGS)}))
        try:
            done = subprocess.run([interpreter, "-B", str(script), str(job)], capture_output=True,
                                  text=True, timeout=timeout,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"the Blender render child did not finish within {timeout} s") from error
        lines = (done.stdout or "").splitlines()
        failed = next((ln for ln in lines if ln.startswith("RENDER_FAILED ")), None)
        if failed:
            raise RuntimeError(failed[len("RENDER_FAILED "):])
        ok = next((ln for ln in lines if ln.startswith("RENDER_OK ")), None)
        if ok is None:
            raise RuntimeError(f"the Blender render child exited {done.returncode}: "
                               + ((done.stdout or "") + (done.stderr or ""))[-2500:])
        drawn = json.loads(ok[len("RENDER_OK "):])
        pngs = sorted(frames.glob("f_*.png"))
        if not pngs:
            raise RuntimeError("the render wrote no frames")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            encode = subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-framerate", repr(drawn["fps"]),
                 "-start_number", str(drawn["frame_start"]), "-i", str(frames / "f_%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                 str(dest)], capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("ffmpeg did not finish encoding within 1800 s") from error
        if encode.returncode or not dest.is_file():
            raise RuntimeError(f"ffmpeg failed: {encode.stderr[-1500:]}")
    return {**drawn, "frames": len(pngs), "path": str(dest)}


def _child_env(work: Path, out: Path) -> dict:
    env = dict(os.environ)
    env["DYCRAFT_BLENDER_WORKSPACE_ROOT"] = str(work)
    env["DYCRAFT_BLENDER_OUTPUT_ROOT"] = str(out)
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _evaluate(blend: Path, work: Path, out: Path, *, relocate: dict | None, timeout: int) -> dict:
    """Run the staging child on `blend`, with `work` as the workspace whose dependency audit it
    passes, and return its report; the evaluation is written to `out/evaluation/`."""
    from ..render._blender_env import blender_python

    interpreter = blender_python()
    if not interpreter:
        raise RuntimeError("no Blender interpreter: a .blend cannot be evaluated without one. "
                           "Run `python -m dycraft.install blender`, or set DYCRAFT_BLENDER_PYTHON.")
    script, job = out.parent / "evaluate.py", out.parent / "job.json"
    script.write_text(_CHILD)
    job.write_text(json.dumps({"repo": str(REPO), "blend": str(blend), "relocate": relocate}))
    try:
        done = subprocess.run([interpreter, "-B", str(script), str(job)], capture_output=True,
                              text=True, env=_child_env(work, out), timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"the Blender child did not finish within {timeout} s") from error
    line = next((ln for ln in (done.stdout or "").splitlines() if ln.startswith("STAGE_OK ")), None)
    if line is None:
        raise RuntimeError(f"the Blender child exited {done.returncode}: "
                           + ((done.stdout or "") + (done.stderr or ""))[-2500:])
    return json.loads(line[len("STAGE_OK "):])


def stage(blend: Path, out_dir: Path, *, scene_ref: str, relocate: dict | None = None,
          film: Path | None = None, render_missing_film: bool = False,
          timeout: int = 4 * 3600) -> dict:
    """Stage `blend` as an animation bundle under `out_dir` and return what was staged.

    `<out_dir>/bundle/` holds the evaluated copy of the blend, its per-frame evaluation (what the
    viewer plays in 3D) and its film; `<out_dir>/ir.json` is the manifest
    (`dycraft.animation.blender.v1`, `scene_ref` naming the arm). The film is `film` (the agent's
    own) when it is a file, else, with `render_missing_film`, one rendered from the staged blend
    (`render_film`); a render that fails leaves the bundle without a film and says why in the
    returned `film`.

    `relocate`: `{"origin": <the blend's own folder>, "prefixes": {"/workspace": <host workspace>}}`
    -- references are resolved to this host and packed into the staged copy before the dependency
    audit. A dependency Blender cannot pack (a simulation cache, a linked library it cannot pack)
    still fails the audit, and staging fails with it.
    """
    # Resolved, because the dependency audit compares resolved paths with the workspace root.
    blend = Path(blend).resolve()
    if not blend.is_file():
        raise FileNotFoundError(f"no .blend at {blend}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    film_record = None
    with tempfile.TemporaryDirectory(prefix="benchmark_stage_") as td:
        work, out = Path(td).resolve() / "ws", Path(td).resolve() / "out"
        work.mkdir(), out.mkdir()
        # Copied, not evaluated in place: `audit_dependencies` requires every unpacked dependency to
        # sit under the workspace root, which is the check that makes a bundle portable.
        shutil.copy2(blend, work / BLEND_NAME)
        payload = _evaluate(work / BLEND_NAME, work, out, relocate=relocate, timeout=timeout)

        from ..animation.blender.bundle import make_manifest, validate_manifest
        bundle = out_dir / "bundle"
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        shutil.copy2(work / BLEND_NAME, bundle / BLEND_NAME)
        shutil.copytree(out / "evaluation", bundle / "evaluation")
        info = json.loads((bundle / "evaluation" / "index.json").read_text())
        # A bare agent declares nothing: no events, no collision allowances.
        info["declarations"] = {"events": [], "collision_allowances": []}
        if film is not None and Path(film).is_file():
            (bundle / FILM_NAME).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(film, bundle / FILM_NAME)
            film_record = {"source": "agent", "path": str(bundle / FILM_NAME)}
        elif render_missing_film:
            try:
                film_record = {"source": "harness",
                               **render_film(bundle / BLEND_NAME, bundle / FILM_NAME)}
            except Exception as error:                             # noqa: BLE001
                film_record = {"source": None, "error": f"{type(error).__name__}: {error}"}
        manifest = make_manifest(bundle, scene_ref=scene_ref, evaluation=info)
        if (bundle / FILM_NAME).is_file():
            from ..animation.blender.bundle import content_hash
            manifest["film"] = f"bundle/{FILM_NAME}"
            manifest["id"] = content_hash({k: v for k, v in manifest.items() if k != "id"})
        (out_dir / "ir.json").write_text(json.dumps(manifest, indent=1))
        validate_manifest(out_dir, manifest)

    return {"blend": str(blend), "evaluation": payload.get("evaluation"), "film": film_record}


def rebuild(run_dir: Path, *, film: Path | None = None, timeout: int = 4 * 3600) -> dict:
    """Rebuild a DyCraft run's `bundle/evaluation/` from its own blend, in place, and return what
    was written.

    The blend is evaluated where it sits, with `bundle/` as the workspace, so its caches resolve at
    their relative paths and the dependency audit checks that each is there. It is not saved. The
    film is `film`, else `RUN_DIR/film.mp4`, copied to `bundle/film/clip.mp4`; an existing
    `clip.mp4` is replaced only by an explicit `film`.

    `RUN_DIR/ir.json` is kept with every field the run recorded. Only the file list, the blend and
    evaluation entries, the film entry and the id are refreshed. An evaluation whose frame rate, frame
    count or first frame differ from the manifest's is refused: then the blend is not the film the
    manifest describes. Without an `ir.json`, one is built from the blend, as a run builds it. A
    `meta.json` is written only when the run has none, since the viewer lists a run only when it has
    one. A run without a design baseline is refused, so a bare agent's arm is not relabelled as a
    DyCraft run.

    Everything is built and checked beside the run, on its filesystem, before the run changes: the
    new evaluation and film are then moved in with `os.replace`, and `ir.json` is replaced whole, so a
    failure leaves the run as it was.
    """
    from ..animation.blender import SCHEMA
    from ..animation.blender.bundle import content_hash, excluded, file_hash, make_manifest, validate_manifest

    from ..run_layout import internal_dir
    run_dir = internal_dir(Path(run_dir).resolve())
    bundle = run_dir / "bundle"
    if not (bundle / BLEND_NAME).is_file():
        raise FileNotFoundError(f"no {BLEND_NAME} under {bundle}")
    if not any((bundle / rel).is_file() for rel in ("design_state/baseline.json", "baseline.json")):
        raise ValueError(f"{bundle} has no design_state/baseline.json: not a DyCraft run")
    clip = bundle / FILM_NAME
    source = Path(film).resolve() if film is not None else run_dir / "film.mp4"
    if film is not None and not source.is_file():
        raise FileNotFoundError(f"no film at {source}")
    copy_film = source.is_file() and (film is not None or not clip.is_file()) and source != clip.resolve()
    # A rebuild killed outright leaves its folder behind. Killed between its two renames, that folder holds the run's
    # previous evaluation, which goes back first: the run is then as it was before that rebuild.
    for stale in sorted(run_dir.glob(".rebuild_*")):
        if (stale / "previous_evaluation").is_dir() and not (bundle / "evaluation").exists():
            os.replace(stale / "previous_evaluation", bundle / "evaluation")
        shutil.rmtree(stale, ignore_errors=True)
    recorded = json.loads((run_dir / "ir.json").read_text()) if (run_dir / "ir.json").is_file() else None

    with tempfile.TemporaryDirectory(prefix=".rebuild_", dir=run_dir) as td:
        staged = Path(td) / "out"
        staged.mkdir()
        payload = _evaluate(bundle / BLEND_NAME, bundle, staged, relocate=None, timeout=timeout)
        evaluation = staged / "evaluation"
        info = json.loads((evaluation / "index.json").read_text())
        differ = [k for k in ("fps", "n_frames", "frame_start")
                  if recorded is not None and recorded.get(k) != info[k]]
        if differ:
            raise ValueError(f"the rebuilt evaluation differs from {run_dir / 'ir.json'} in "
                             + ", ".join(f"{k} ({recorded.get(k)} recorded, {info[k]} now)" for k in differ))
        if copy_film:
            shutil.copy2(source, Path(td) / "clip.mp4")

        # The manifest of the bundle as it will be: its files now, without the evaluation being replaced, plus the
        # staged evaluation and film. `make_manifest` also refuses a bundle holding a symlink.
        fresh = make_manifest(bundle, scene_ref=info.get("scene_ref") or (recorded or {}).get("scene_ref")
                              or run_dir.name, evaluation=info)
        files = {k: v for k, v in fresh["files"].items() if not k.startswith("bundle/evaluation/")}
        files.update({f"bundle/evaluation/{path.relative_to(evaluation).as_posix()}": file_hash(path)
                      for path in sorted(evaluation.rglob("*")) if path.is_file() and not excluded(path, evaluation)})
        if copy_film:
            files[f"bundle/{FILM_NAME}"] = file_hash(Path(td) / "clip.mp4")
        fresh["files"] = dict(sorted(files.items()))
        manifest = ({**fresh, "layer": DYCRAFT_LAYER} if recorded is None
                    else {**recorded, **{k: fresh[k] for k in ("files", "blend", "evaluation")}})
        if f"bundle/{FILM_NAME}" in manifest["files"]:
            manifest["film"] = f"bundle/{FILM_NAME}"
        manifest.pop("id", None)
        manifest["id"] = content_hash(manifest)
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"{run_dir / 'ir.json'} is not a {SCHEMA} manifest")
        (Path(td) / "ir.json").write_text(json.dumps(manifest, indent=1))

        # The run changes only from here, by renames within its filesystem.
        if (bundle / "evaluation").exists():
            os.replace(bundle / "evaluation", Path(td) / "previous_evaluation")
        os.replace(evaluation, bundle / "evaluation")
        if copy_film:
            clip.parent.mkdir(parents=True, exist_ok=True)
            os.replace(Path(td) / "clip.mp4", clip)
        os.replace(Path(td) / "ir.json", run_dir / "ir.json")
    validate_manifest(run_dir, manifest)

    meta_written = not (run_dir / "meta.json").is_file()
    if meta_written:
        (run_dir / "meta.json").write_text(json.dumps(
            {"run_id": run_dir.name, "layer": DYCRAFT_LAYER, "outcome": "delivered"}, indent=1))
    return {"run_dir": str(run_dir), "evaluation": payload.get("evaluation"),
            "frames": len(info.get("frames") or ()), "film": manifest.get("film"),
            "manifest": "updated" if recorded is not None else "written", "meta_written": meta_written}


def prepare(run_dir: Path, dest: Path, *, timeout: int = 4 * 3600) -> dict:
    """Make a film playable in the viewer without changing it, and return what was written.

    What `rebuild` writes into a run, written under `dest` instead: the per-frame evaluation of
    `RUN_DIR/bundle/animation.blend` in `dest/bundle/evaluation/`, which fills frame by frame as the
    blend is evaluated, so a caller can count its progress there, and then `dest/ir.json`, the
    manifest of the film as the viewer sees it: the film's own bundle and video, plus this
    evaluation. The blend is evaluated where it sits, with its bundle as the workspace, as `rebuild`
    does, and the film's folder is only read: a film that links into the dataset mirror stays as it
    was downloaded. The viewer lays `dest` over the film's folder (`viewer/films.py`).
    """
    from ..animation.blender.bundle import content_hash, excluded, file_hash, make_manifest
    from ..run_layout import internal_dir

    run_dir = internal_dir(Path(run_dir).resolve())
    bundle = run_dir / "bundle"
    if not (bundle / BLEND_NAME).is_file():
        raise FileNotFoundError(f"no {BLEND_NAME} under {bundle}")
    dest = Path(dest).resolve()
    out = dest / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    payload = _evaluate(bundle / BLEND_NAME, bundle, out, relocate=None, timeout=timeout)
    for scratch in ("evaluate.py", "job.json"):
        (dest / scratch).unlink(missing_ok=True)
    evaluation = out / "evaluation"
    info = json.loads((evaluation / "index.json").read_text())
    # The film's own bundle files, hashed where they are, apart from an evaluation a staged copy may carry, which
    # this one replaces.
    fresh = make_manifest(bundle, scene_ref=info.get("scene_ref") or run_dir.name, evaluation=info)
    files = {k: v for k, v in fresh["files"].items() if not k.startswith("bundle/evaluation/")}
    files.update({f"bundle/evaluation/{path.relative_to(evaluation).as_posix()}": file_hash(path)
                  for path in sorted(evaluation.rglob("*")) if path.is_file() and not excluded(path, evaluation)})
    manifest = {**fresh, "files": dict(sorted(files.items())), "layer": DYCRAFT_LAYER}
    # A published film keeps its video at the top as `film.mp4`; a run keeps it in its bundle.
    film = next((rel for rel in ("film.mp4", f"bundle/{FILM_NAME}") if (run_dir / rel).is_file()), None)
    if film:
        manifest["film"] = film
    manifest.pop("id", None)
    manifest["id"] = content_hash(manifest)
    (dest / "ir.json").write_text(json.dumps(manifest, indent=1))
    return {"run_dir": str(run_dir), "dest": str(dest), "evaluation": payload.get("evaluation"),
            "frames": len(info.get("frames") or ()), "film": film}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dycraft.benchmark.stage",
        description="Rebuild a dycraft run's bundle/evaluation/ from its bundle/animation.blend, in place, "
                    "so the viewer can play the run in 3D.")
    ap.add_argument("run_dir", type=Path, metavar="RUN_DIR", help="the run folder holding bundle/animation.blend")
    ap.add_argument("--film", type=Path, default=None,
                    help="the run's film, copied to bundle/film/clip.mp4 (default: RUN_DIR/film.mp4 if present)")
    args = ap.parse_args(argv)
    print(json.dumps(rebuild(args.run_dir, film=args.film), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
