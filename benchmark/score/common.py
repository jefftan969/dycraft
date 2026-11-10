"""Interpreters, child processes, and hashing shared by both metrics."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


@dataclass
class Settings:
    blender_python: str
    coal_python: str
    jobs: int = 1
    #: Pieces each film's frames are split into (default `jobs`); at most `jobs` run at once. A geometry child's
    #: memory grows with the frames it measures, so more, shorter pieces lower the peak of a long film.
    chunks: int | None = None
    quiet: bool = True
    recompute: bool = False


def sha256(path) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def blender_python(explicit: str | None = None) -> str:
    """The `bpy` interpreter dycraft's Blender layers use (`.blender_python` / `DYCRAFT_BLENDER_PYTHON`)."""
    if explicit:
        return explicit
    from dycraft.render import _blender_env
    found = _blender_env.blender_python()
    if not found:
        raise SystemExit("no Blender Python: run `python -m dycraft.install blender`, set DYCRAFT_BLENDER_PYTHON, "
                         "or pass --blender-python")
    return found


def coal_python(explicit: str | None = None) -> str:
    """An interpreter with `coal`, for the animation layer's collision worker (default: this one)."""
    if explicit:
        return explicit
    if os.environ.get("DYCRAFT_COAL_PYTHON"):
        return os.environ["DYCRAFT_COAL_PYTHON"]
    if importlib.util.find_spec("coal") is not None:
        return sys.executable
    raise SystemExit("no interpreter with coal: run under the DyCraft environment, set DYCRAFT_COAL_PYTHON, "
                     "or pass --coal-python")


def child_env(settings: Settings, threads: int = 4) -> dict:
    return dict(os.environ, DYCRAFT_COAL_PYTHON=settings.coal_python, OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS=str(threads))


def run_child(settings: Settings, script: str, args) -> None:
    """Run one of this package's Blender children to completion."""
    completed = subprocess.run([settings.blender_python, str(HERE / script), *map(str, args)],
                               env=child_env(settings),
                               stdout=subprocess.PIPE if settings.quiet else None,
                               stderr=subprocess.PIPE if settings.quiet else None, text=True)
    if completed.returncode:
        detail = (completed.stderr or "")[-4000:] if settings.quiet else ""
        raise SystemExit(f"{script} failed (exit {completed.returncode})\n{detail}")


def chunk_count(settings: Settings, n_frames: int) -> int:
    return max(1, min(settings.chunks or settings.jobs, n_frames))


def run_children(settings: Settings, script: str, jobs: list[tuple[str, list]], log_dir: Path) -> None:
    """Run several children of one script, at most `settings.jobs` at once (one per (label, args)); each logs to its
    own file. Once one fails the film cannot be scored, so no further child starts and the running ones are stopped,
    rather than spending hours on a result that will be discarded."""
    pending, running, failed = list(jobs), [], []
    try:
        while (pending and not failed) or running:
            while pending and not failed and len(running) < max(1, settings.jobs):
                label, args = pending.pop(0)
                log = open(log_dir / f"{Path(script).stem}.{label}.log", "w")
                running.append((label, log, subprocess.Popen([settings.blender_python, str(HERE / script),
                                                              *map(str, args)], env=child_env(settings, threads=1),
                                                             stdout=log, stderr=subprocess.STDOUT)))
            done = [item for item in running if item[2].poll() is not None]
            for item in done:
                label, log, proc = item
                if proc.returncode:
                    failed.append(label)
                log.close()
                running.remove(item)
            if failed:
                break
            if not done:
                time.sleep(0.2)
    finally:
        for _, _, proc in running:
            proc.terminate()
        for _, log, proc in running:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
    if failed:
        raise SystemExit(f"{script} parts {failed} failed; see {log_dir}/{Path(script).stem}.<part>.log")
    for label, _ in jobs:
        (log_dir / f"{Path(script).stem}.{label}.log").unlink(missing_ok=True)
