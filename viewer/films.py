"""Films installed from the published set, played in the viewer with no stage step.

`python -m dycraft.install films` links each film into `films/` as the dataset has it: its video
(`film.mp4`), its Blender project (`bundle/animation.blend`), its plan, and its `run.json`. It has none
of the per-frame evaluation the viewer plays in 3D, and none of the run records (`meta.json`,
`ir.json`) the viewer reads.

So the viewer lists such a film from its `run.json`, and the first time someone opens it, prepares
it (`benchmark.stage.prepare`): the evaluation and a manifest are written to the viewer's own folder
under the DyCraft home, `.viewer_films/<key>/`, keyed by the film's path and its blend's size and
modification time. From then on that folder is the film's run folder, laid over the film's own
folder for the two files it doesn't copy, the video and the blend (`locate`). Later opens are
immediate.

Nothing is written into the film's folder, which links into the dataset mirror, and nothing is
prepared for a film nobody opens. A preparation that fails is remembered for the server's life, so
a film that cannot be prepared costs the attempt once.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

from .._home import state_path

#: Under the DyCraft home: one folder per prepared film.
PREPARED_DIR = ".viewer_films"
#: In a prepared folder: the film folder it was prepared from.
SOURCE = "film.json"
#: How long one preparation may take before it is given up.
TIMEOUT_S = 2 * 3600

_lock = threading.Lock()
#: Preparations this server started and has not finished, or that failed, by key.
_jobs: dict[str, dict] = {}


def is_installed_film(path: Path) -> bool:
    """Whether `path` is a film as the dataset publishes it: a `run.json` and a Blender project, and no run records.

    A film staged by `benchmark.stage` has a `meta.json`, so it is a run like any other.
    """
    path = Path(path)
    return ((path / "run.json").is_file() and (path / "bundle" / "animation.blend").is_file()
            and not (path / "meta.json").is_file())


def root() -> Path:
    return state_path(PREPARED_DIR)


def _key(film: Path) -> str | None:
    real = Path(film).resolve()
    try:
        st = (real / "bundle" / "animation.blend").stat()
    except OSError:
        return None
    return hashlib.sha256(f"{real}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:24]


def prepared(film: Path) -> Path | None:
    """The film's prepared folder, once it is complete (`meta.json` is written last)."""
    key = _key(film)
    if key is None:
        return None
    folder = root() / key
    return folder if (folder / "meta.json").is_file() else None


def run_json(film: Path) -> dict:
    try:
        data = json.loads((Path(film) / "run.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def film_meta(film: Path) -> dict:
    """The run record the viewer lists a film by, from its `run.json` and prompt."""
    record = run_json(film)
    agent = record.get("agent") if isinstance(record.get("agent"), dict) else {}
    meta = {"run_id": Path(film).name, "layer": "animation", "outcome": "delivered",
            "model": agent.get("model"), "reasoning_effort": agent.get("reasoning_effort"),
            "runtime": agent.get("cli")}
    try:
        meta["prompt"] = (Path(film) / "prompt.md").read_text(errors="replace")
    except OSError:
        pass
    return {k: v for k, v in meta.items() if v is not None}


def film_label(film: Path) -> str | None:
    """What the run list calls a film: its brief and its title from `run.json`, joined by a middle dot."""
    record = run_json(film)
    parts = [str(record[k]) for k in ("brief", "title") if record.get(k)]
    return " · ".join(parts) or None


def film_frames(film: Path) -> int | None:
    frames = (run_json(film).get("film") or {}).get("frames")
    return frames if isinstance(frames, int) and frames > 0 else None


def source_of(run_dir: Path) -> Path | None:
    """The film folder a prepared folder is laid over, else None."""
    run_dir = Path(run_dir)
    try:
        if run_dir.resolve().parent != root().resolve():
            return None
        film = Path(json.loads((run_dir / SOURCE).read_text())["film"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return film if film.is_dir() else None


def locate(run_dir: Path, rel: str) -> Path | None:
    """The file `rel` names inside the run folder, or, for a prepared film, inside the film's own folder.

    Containment is checked in each folder on its own, so nothing outside either is reachable.
    """
    if not rel:
        return None
    for base in (Path(run_dir), source_of(run_dir)):
        if base is None:
            continue
        top = base.resolve()
        target = (top / rel).resolve()
        if target.is_relative_to(top) and target.is_file():
            return target
    return None


def status(film: Path, *, start: bool = True) -> dict:
    """Where the film's preparation stands, starting it when `start` and it is neither ready nor under way.

    `{"state": "ready"}`; `{"state": "preparing", "frames_done", "frames", "seconds"}`, where `frames_done` stays 0
    while Blender opens the file, which for a heavy film is most of the wait; `{"state": "failed", "error"}`; or,
    without `start`, `{"state": "unprepared"}`.
    """
    film = Path(film)
    key = _key(film)
    if key is None:
        return {"state": "failed", "error": "the film has no bundle/animation.blend"}
    if (root() / key / "meta.json").is_file():
        return {"state": "ready"}
    with _lock:
        job = _jobs.get(key)
        if job is None:
            if not start:
                return {"state": "unprepared"}
            job = {"state": "preparing", "started": time.monotonic(), "frames": film_frames(film),
                   "work": root() / f".{key}.{os.getpid()}"}
            _jobs[key] = job
            threading.Thread(target=_prepare, args=(film, key, job), daemon=True,
                             name=f"prepare-{film.name}").start()
    if job["state"] == "failed":
        return {"state": "failed", "error": job["error"]}
    evaluation = job["work"] / "bundle" / "evaluation"
    try:
        done = sum(1 for p in evaluation.iterdir() if p.suffix == ".bin")
    except OSError:
        done = 0
    return {"state": "preparing", "frames_done": done, "frames": job["frames"],
            "seconds": round(time.monotonic() - job["started"], 1)}


def _prepare(film: Path, key: str, job: dict) -> None:
    from ..benchmark.stage import prepare

    work: Path = job["work"]
    final = root() / key
    try:
        _clear_abandoned(key)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        prepare(film, work, timeout=TIMEOUT_S)
        (work / SOURCE).write_text(json.dumps({"film": str(film.resolve())}))
        (work / "meta.json").write_text(json.dumps(film_meta(film), indent=1))
        # A folder left without its `meta.json` was never finished, so it is replaced.
        shutil.rmtree(final, ignore_errors=True)
        os.replace(work, final)
        with _lock:
            _jobs.pop(key, None)
    except Exception as error:                                          # noqa: BLE001 - reported to the viewer
        shutil.rmtree(work, ignore_errors=True)
        with _lock:
            job.update(state="failed", error=f"{type(error).__name__}: {error}"[-800:])


def _clear_abandoned(key: str) -> None:
    """Remove this film's working folders left by servers that stopped mid-preparation."""
    for folder in root().glob(f".{key}.*"):
        try:
            pid = int(folder.name.rsplit(".", 1)[1])
            os.kill(pid, 0)
            if pid != os.getpid():
                continue                                    # another live server is preparing it
        except (ValueError, ProcessLookupError):
            pass
        except PermissionError:
            continue
        shutil.rmtree(folder, ignore_errors=True)
