"""Animation-run artifacts: the staged scene, and a timeline the viewer scrubs.

An animation run's `ir.json` is a native Blender animation record
(`dycraft.animation.blender.v1`): it holds no geometry of its own. Its scene is
the frozen `SceneIR` staged beside it as `scene.json`, plus the `catalog/` that
SceneIR's instances resolve against, so `viewer/runs.py` reads the URDF off the
staged scene and this module supplies what the inspector and the scrubber need
on top: a summary, the declared events, and whether each held, the rendered
clip, and the timeline, which `viewer/native_animation.py` builds from the
run's evaluated frames.

Runs recorded in the older AnimationIR format are not supported: the viewer
reports them as having no timeline.

Read-only, like the rest of `viewer/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Mirrors `agent/layers/scene_staging.py::SCENE_JSON`. Literals rather
# than imports so importing the viewer never pulls in the harness.
ANIMATION_SCHEMA_PREFIX = "dycraft.animation."
NATIVE_SCHEMA = "dycraft.animation.blender.v1"
SCENE_JSON = "scene.json"
#: The name the animation verifier reports declared events under.
_EVENTS_SATISFIED = "EventsSatisfied"


class TimelineUnavailable(Exception):
    """The run's `ir.json` is not one this viewer can build a timeline for."""



def is_animation_ir(ir: Any) -> bool:
    """True for an animation run's `ir.json`, keyed off its schema tag.

    Unlike an ObjectIR (`parts`) or a SceneIR (`instances`), an animation IR has
    no structural field the other two lack, and the schema string is
    authoritative and version-tagged.
    """
    return (isinstance(ir, dict)
            and str(ir.get("schema") or "").startswith(ANIMATION_SCHEMA_PREFIX))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_scene_ir(run_dir: Path) -> dict | None:
    """The frozen SceneIR staged beside the animation, or None."""
    ir = _read_json(run_dir / SCENE_JSON)
    return ir if isinstance(ir, dict) and isinstance(ir.get("instances"), list) \
        else None


def n_frames(ir: dict) -> int:
    """Frame count, as the native record states it; 0 for anything else."""
    try:
        return int(ir["n_frames"]) if ir.get("schema") == NATIVE_SCHEMA else 0
    except (KeyError, TypeError, ValueError):
        return 0


def frame_time(ir: dict, i: int) -> float:
    return i / float(ir["fps"])


def clip_rel(run_dir: Path, ir: dict | None = None) -> str | None:
    """Run-root-relative path to the rendered clip, or None if not rendered.

    An animation run commits its film inside the bundle and names it in its
    manifest (`ir.json`'s `film`, `bundle/film/clip.mp4`). The manifest's path
    is used only when it stays inside the run directory.

    None when the run has no film: it may have stopped before rendering one.
    """
    run_dir = Path(run_dir)
    if ir is None:
        try:
            ir = json.loads((run_dir / "ir.json").read_text())
        except (OSError, ValueError):
            ir = None
    film = ir.get("film") if isinstance(ir, dict) else None
    from .films import locate
    # Inside the run folder, or, for a prepared film, inside the film's own folder, where its video stays.
    return film if isinstance(film, str) and film and locate(run_dir, film) is not None else None


def summary(run_dir: Path, ir: dict, scene_ir: dict | None) -> dict:
    """The animation facts the inspector shows, without the dense arrays.

    Events come through `_events` rather than straight off the IR. The IR states
    what was *claimed*; whether each claim held is in the run's verifier report,
    and a caller shown the raw list has no way to tell the two apart — it reads
    every event of a passing run as unsatisfied.
    """
    return {
        "fps": ir.get("fps"),
        "duration": ir.get("duration"),
        "n_frames": n_frames(ir),
        "scene_ref": ir.get("scene_ref"),
        "scene_ref_ok": (None if scene_ir is None
                         else ir.get("scene_ref") == scene_ir.get("id")),
        "events": _events(run_dir, ir),
        "provenance": list(ir.get("provenance") or ()),
    }


def _events(run_dir: Path, ir: dict) -> list:
    """Declared events, each annotated with the frames it maps to and whether
    the run's own `EventsSatisfied` found it true.

    Shared by `summary` and `timeline` so the Overview list and the scrubber
    ticks cannot disagree about which events fired.
    """
    n = n_frames(ir)
    fps = float(ir.get("fps") or 0)
    index = _evaluation_index(run_dir, ir)
    rows = _event_rows(run_dir, ir, index)
    used: set[int] = set()
    out = []
    # The position in `events` is the declaration index the verifier's rows
    # for a rejected declaration carry.
    for position, ev in enumerate(ir.get("events") or ()):
        if not isinstance(ev, dict):
            continue
        w = list(ev.get("window") or [None, None])
        placed = _placed_frames(ir, ev, w, index, n, fps)
        row = _matching_row(rows.get(ev.get("name")) or [], position, placed, used)
        resolved = row.get("window_frames")
        idx = (list(range(max(resolved[0], 0), min(resolved[1], n - 1) + 1))
               if _frame_pair(resolved) else placed)
        at = row.get("satisfied_at")
        out.append({
            "name": ev.get("name"),
            "predicate": ev.get("predicate"),
            # Rounded for display: an authored 0.36333333333333334 is a frame
            # boundary, not a meaningful eleventh decimal place.
            "window": [round(v, 4) if isinstance(v, (int, float)) else v
                       for v in w],
            "window_frames": [idx[0], idx[-1]] if idx else None,
            "satisfied_at": at,
            "satisfied_frame": (None if at is None or not fps
                                else int(round(float(at) * fps))),
        })
    return out


def timeline(run_dir: Path, ir: dict) -> dict:
    """Everything needed to pose the scene at any frame (native runs only), and the camera the film opens on."""
    if ir.get("schema") != NATIVE_SCHEMA:
        raise TimelineUnavailable(
            f"no timeline for schema {ir.get('schema')!r}: this viewer reads "
            f"native Blender animation runs ({NATIVE_SCHEMA})")
    from .native_animation import timeline as native_timeline
    out = native_timeline(run_dir, ir)
    out["film_camera"] = film_camera(run_dir, ir)
    return out


#: The film's camera record, written beside the film when it is delivered (`animation/delivery.py`): the active
#: camera at every frame, with its world matrix and lens.
FILM_CAMERA = "camera.json"
_LENS_KEYS = ("lens_mm", "sensor_mm", "sensor_height_mm")


def film_camera(run_dir: Path, ir: dict) -> dict | None:
    """The view the film opens on, so the viewer can start there: the camera at the film's first frame.

    From the film's own record when it was delivered with one; otherwise, as for a film installed from the
    published set, read from its `.blend` (`_blend_camera`). `matrix` is the camera's world matrix in Blender's
    frame, rows first, the frame the timeline's poses are in; `aspect` is the film's width over height. None when
    neither gives a perspective camera with a lens.
    """
    rel = clip_rel(run_dir, ir)
    record = _read_json((Path(run_dir) / rel).parent / FILM_CAMERA) if rel is not None else None
    if isinstance(record, dict):
        views = [v for v in record.get("views") or () if isinstance(v, dict)]
        start = record.get("frame_start", 0)
        view = next((v for v in views if v.get("frame") == start), views[0]) if views else None
        settings = record.get("render_settings") or {}
        resolution = [settings.get("width"), settings.get("height")]
    else:
        view = _blend_camera(run_dir, ir)
        resolution = (view or {}).get("resolution") or [None, None]
    return _usable_view(view, resolution) if view else None


def film_cameras(run_dir: Path, ir: dict) -> dict | None:
    """The film's camera at every frame, so the film view can follow its shots as the film plays.

    From the same sources as `film_camera`: the film's own record (every frame's active camera), else the film's
    `.blend` read at every frame (`_blend_camera(all_frames=True)`). Compact, one entry per frame from `frame_start`:
    `camera` indexes `cameras` (name and sensor; -1 where the frame's camera cannot be reproduced, a constraint or an
    orthographic lens), `matrix` holds the world matrix's top three rows, rows first, twelve numbers a frame, and
    `lens_mm` and `shift` the lens. A change of `camera` is a cut. None when no frame has a usable camera.
    """
    rel = clip_rel(run_dir, ir)
    record = _read_json((Path(run_dir) / rel).parent / FILM_CAMERA) if rel is not None else None
    if isinstance(record, dict):
        views = [v for v in record.get("views") or () if isinstance(v, dict)]
        start = record.get("frame_start", 0)
        settings = record.get("render_settings") or {}
        resolution = [settings.get("width"), settings.get("height")]
    else:
        found = _blend_camera(run_dir, ir, all_frames=True)
        if not found:
            return None
        views, start, resolution = found.get("views") or [], found.get("frame_start", 0), found.get("resolution")
    return _camera_track(views, start, resolution or [None, None])


def _camera_track(views: list[dict], start, resolution) -> dict | None:
    """`film_cameras`' compact form of per-frame views, or None when none of them is usable."""
    width, height = (list(resolution) + [None, None])[:2]
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0):
        return None
    start = start if isinstance(start, int) else 0
    by_frame = {v["frame"]: v for v in views if isinstance(v.get("frame"), int)}
    if not by_frame:
        return None
    n = max(by_frame) - start + 1
    cameras: list[dict] = []
    known: dict[tuple, int] = {}
    camera, matrix, lens, shift = [], [], [], []
    for frame in range(start, start + max(n, 0)):
        usable = _usable_view(by_frame[frame], resolution) if frame in by_frame else None
        if usable is None:
            camera.append(-1)
            matrix.extend([0.0] * 12)
            lens.append(0.0)
            shift.extend([0.0, 0.0])
            continue
        key = (usable["camera"], usable["sensor_mm"], usable["sensor_height_mm"], usable["sensor_fit"])
        if key not in known:
            known[key] = len(cameras)
            cameras.append({"name": usable["camera"], "sensor_mm": usable["sensor_mm"],
                            "sensor_height_mm": usable["sensor_height_mm"], "sensor_fit": usable["sensor_fit"]})
        camera.append(known[key])
        matrix.extend(round(float(x), 6) for row in usable["matrix"][:3] for x in row)
        lens.append(round(usable["lens_mm"], 4))
        shift.extend(round(float(x), 6) for x in (list(usable["shift"]) + [0.0, 0.0])[:2])
    if not cameras:
        return None
    return {"frame_start": start, "n_frames": len(camera), "aspect": width / height, "cameras": cameras,
            "camera": camera, "matrix": matrix, "lens_mm": lens, "shift": shift}


def film_camera_track(run_dir: Path) -> dict | None:
    """`film_cameras` for a run directory, or None when it is not a native film or has no usable camera."""
    ir = _read_json(Path(run_dir) / "ir.json")
    if not isinstance(ir, dict) or ir.get("schema") != NATIVE_SCHEMA:
        return None
    return film_cameras(Path(run_dir), ir)


#: The film's storyboard, written beside its bundle: each keyframe's name, frame, camera, and rendered view.
STORYBOARDS = "bundle/storyboards.json"
#: Storyboard thumbnails: the height they are drawn at in the timeline's preview, twice over for a sharp screen.
_THUMB_SIZE = (320, 180)


def storyboard(run_dir: Path) -> dict:
    """`storyboard_keys`, and the film's first frame from its manifest, which puts the keyframes' frames (the film's own)
    on the playhead's: a film's keyframes do not depend on whether its cameras can be followed."""
    ir = _read_json(Path(run_dir) / "ir.json")
    start = ir.get("frame_start") if isinstance(ir, dict) and isinstance(ir.get("frame_start"), int) else 0
    return {"frame_start": start, "keys": storyboard_keys(run_dir)}


def storyboard_keys(run_dir: Path) -> list[dict]:
    """The film's storyboard keyframes, in frame order: `name`, `frame` (the film's own frame), `camera`, and `index`,
    the keyframe's place in `storyboard_thumb`. Keyframes whose image is not in the run are left out; a film without a
    storyboard has none (an installed film's bundle carries none).
    """
    from .films import locate

    record = _read_json(locate(run_dir, STORYBOARDS) or Path("/nonexistent"))
    if not isinstance(record, dict):
        return []
    keys = []
    for name, entry in record.items():
        if not (isinstance(entry, dict) and isinstance(entry.get("frame"), int) and isinstance(entry.get("path"), str)):
            continue
        if locate(run_dir, f"bundle/{entry['path']}") is None:
            continue
        keys.append({"name": str(name), "frame": entry["frame"],
                     "camera": entry.get("camera") if isinstance(entry.get("camera"), str) else None,
                     "path": entry["path"]})
    keys.sort(key=lambda k: (k["frame"], k["name"]))
    return [{"name": k["name"], "frame": k["frame"], "camera": k["camera"], "index": i} for i, k in enumerate(keys)]


def storyboard_thumb(run_dir: Path, index: int) -> bytes | None:
    """Keyframe `index`'s rendered view as a small JPEG, made once per image and kept in the viewer's asset cache.

    The views are full renders (a film's two dozen can be 55 MB); the timeline shows one at a time, on hover, small.
    """
    import hashlib
    import io

    from PIL import Image

    from ..render.viewer_bake import cache_root
    from .films import locate

    record = _read_json(locate(run_dir, STORYBOARDS) or Path("/nonexistent"))
    if not isinstance(record, dict):
        return None
    entries = sorted(((v["frame"], str(n), v["path"]) for n, v in record.items()
                      if isinstance(v, dict) and isinstance(v.get("frame"), int) and isinstance(v.get("path"), str)
                      and locate(run_dir, f"bundle/{v['path']}") is not None))
    if not 0 <= index < len(entries):
        return None
    image = locate(run_dir, f"bundle/{entries[index][2]}")
    st = image.stat()
    key = hashlib.sha256(f"{image}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:24]
    cached = cache_root() / "storyboard" / f"{key}.jpg"
    if cached.is_file():
        return cached.read_bytes()
    with Image.open(image) as im:
        im = im.convert("RGB")
        im.thumbnail(_THUMB_SIZE)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=80, optimize=True)
    data = out.getvalue()
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError:
        pass
    return data


def _usable_view(view: dict, resolution) -> dict | None:
    """`view` (a `camera.json` view) in the payload's shape, or None when the viewer cannot reproduce it."""
    matrix = view.get("camera_matrix")
    if not (isinstance(matrix, list) and len(matrix) == 4
            and all(isinstance(row, list) and len(row) == 4
                    and all(isinstance(x, (int, float)) for x in row) for row in matrix)):
        return None
    if view.get("camera_type") != "PERSP" or not all(isinstance(view.get(k), (int, float)) and view[k] > 0
                                                       for k in _LENS_KEYS):
        return None
    width, height = (list(resolution) + [None, None])[:2]
    if not (isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0):
        return None
    shift = view.get("shift") if isinstance(view.get("shift"), list) else [0.0, 0.0]
    return {"frame": view.get("frame"), "camera": view.get("camera"), "matrix": matrix,
            **{k: float(view[k]) for k in _LENS_KEYS},
            "sensor_fit": view.get("sensor_fit") or "AUTO", "shift": shift, "aspect": width / height}


def _frame_pair(value) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(type(v) is int for v in value))


def _placed_frames(ir: dict, ev: dict, w: list, index: dict | None, n: int,
                   fps: float) -> list[int]:
    """The output frames an event's window covers, placed as the verifier
    places it (`verify.window_indices`) from the evaluation index the bundle
    carries: the index holds the declaration units and the clip spans a
    clip-relative window is placed by. A window it refuses covers no frame.
    Without an index, the window is read in seconds."""
    if None in w:
        return []
    if index is not None:
        from ..animation.blender.verify import window_indices
        try:
            return list(window_indices(index, index.get("declarations") or {}, ev))
        except (ValueError, KeyError, TypeError):
            return []
    if not fps:
        return []
    # A frame is in the window when its time is, with a hair of slack for a
    # boundary that falls exactly on a frame.
    return [i for i in range(n)
            if float(w[0]) - 1e-9 <= frame_time(ir, i) <= float(w[1]) + 1e-9]


def _matching_row(candidates: list[dict], position: int, placed: list[int],
                  used: set[int]) -> dict:
    """The report row for one declared event, each row used at most once.

    A rejected declaration's row (`status: "invalid"`) carries the
    declaration's index, and is matched by it. Otherwise by name when one row
    is left under it. Event names may repeat (one per clip, say) and a row
    carries no clip, so then the row must also have resolved the window to the
    same frames; when none does, none is attached rather than another event's.
    """
    free = [r for r in candidates if id(r) not in used]
    by_index = [r for r in free if r.get("status") == "invalid" and r.get("index") == position]
    if by_index:
        chosen = by_index[0]
    else:
        free = [r for r in free if r.get("status") != "invalid"]
        span = [placed[0], placed[-1]] if placed else None
        if len(free) == 1:
            chosen = free[0]
        else:
            chosen = next((r for r in free if span and r.get("window_frames") == span), None)
    if chosen is None:
        return {}
    used.add(id(chosen))
    return chosen


def _evaluation_index(run_dir: Path, ir: dict) -> dict | None:
    """The evaluation index the bundle carries (`ir["evaluation"]`), if it is
    inside the run directory and readable."""
    rel = ir.get("evaluation")
    if not isinstance(rel, str) or not rel:
        return None
    path = (Path(run_dir) / rel).resolve()
    if not path.is_relative_to(Path(run_dir).resolve()):
        return None
    index = _read_json(path)
    return index if isinstance(index, dict) else None


def _report_describes(report: dict, ir: dict, index: dict | None) -> bool:
    """Whether a compile report is about this committed animation.

    A report stamped with an IR hash must carry this one. An animation run's
    report carries its evaluation instead, which is the bundle's own index
    when the report is the check the commit used: the per-frame signatures
    and every field of the index besides (the declarations among them, which
    the signatures do not cover) when the index has signatures, the whole
    index otherwise. `latest.json` is written only by a compile, so a commit
    after a declarations-only edit can leave one whose signatures still match.
    """
    if report.get("ir_hash") is not None:
        return report["ir_hash"] == ir.get("id")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict) or index is None:
        return False
    if index.get("signatures"):
        from ..animation.blender.evaluation import INDEX_FIELDS
        return (evaluation.get("signatures") == index["signatures"]
                and all(evaluation.get(k) == index.get(k) for k in INDEX_FIELDS + ("clips",)))
    return evaluation == index


def _event_rows(run_dir: Path, ir: dict, index: dict | None) -> dict[str, list[dict]]:
    """`{event name: [EventsSatisfied rows]}` from the run's own final compile report.

    Recomputing this would mean re-running `events_satisfied` over every frame
    on an HTTP request; the run already recorded the answer. Guarded by
    `_report_describes`: a report from an earlier, repaired attempt describes a
    different animation, and marking the timeline with its timings would be
    worse than marking nothing.
    """
    report = _read_json(run_dir / "compile_reports" / "latest.json")
    if not isinstance(report, dict) or not _report_describes(report, ir, index):
        return {}
    for entry in report.get("verifier_reports") or ():
        # Match the verifier by NAME, not by "which report has an `events`
        # key". `AnimationWellFormed` also carries one — it lists the declared
        # events to check each names a real actor — and it sorts first, so
        # shape-matching silently reads its rows and reports every event as
        # never satisfied.
        if not isinstance(entry, dict) or entry.get("name") != _EVENTS_SATISFIED:
            continue
        evidence = entry.get("evidence")
        rows = evidence.get("events") if isinstance(evidence, dict) else None
        if isinstance(rows, list):
            # The verifier computes `satisfied_at` and `window_frames` itself,
            # `satisfied_at` already `None` for a deferred or unmet event, so no
            # reader re-derives them.
            out: dict[str, list[dict]] = {}
            for r in rows:
                if isinstance(r, dict):
                    out.setdefault(r.get("name"), []).append(r)
            return out
    return {}


#: The Blender child that reads a film's opening camera from its `.blend`, and how long it may take.
_CAMERA_CHILD = Path(__file__).resolve().parents[1] / "render" / "_film_camera_child.py"
_CAMERA_TIMEOUT_S = 15
#: Reads that timed out, by the same key as the cache: remembered for this server's life, so a film whose read always
#: times out costs the timeout once rather than on every `/animation` request. Not written to the cache, because a
#: read can time out on a busy machine, and the next server tries again.
_CAMERA_TIMED_OUT: set[str] = set()


def _blend_camera(run_dir: Path, ir: dict, all_frames: bool = False) -> dict | None:
    """The opening camera read from the film's `.blend`, for a film delivered without a `camera.json`; with
    `all_frames`, the film's camera at every frame (`views`, `frame_start`, `resolution`), for `film_cameras`.

    A Blender child loads the file's scene as library data, so nothing is evaluated: a fraction of a second, where
    opening and evaluating a heavy film takes minutes. Cached by the file's path, size, and modification time in the
    viewer's asset cache, so it runs once per film. None, cached too, when the read fails or the camera is placed by
    a constraint. None, not cached, without a Blender interpreter, and when the read times out, which this server
    then remembers (`_CAMERA_TIMED_OUT`).
    """
    import hashlib
    import subprocess
    import tempfile

    from .. import _oom
    from ..render._blender_env import blender_child_env, blender_python
    from ..render.viewer_bake import cache_root

    from .films import locate

    rel = ir.get("blend") if isinstance(ir.get("blend"), str) else "bundle/animation.blend"
    # A prepared film's blend stays in the film's own folder (`films.locate`).
    blend = locate(run_dir, rel)
    if blend is None:
        return None
    st = blend.stat()
    key = hashlib.sha256(f"{blend}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:24]
    if all_frames:
        key += "_all"
    cached = cache_root() / "film_camera" / f"{key}.json"
    found = _read_json(cached)
    if isinstance(found, dict):
        return found if found.get("ok") else None
    if key in _CAMERA_TIMED_OUT:
        return None
    interp = blender_python()
    if not interp:
        return None
    with tempfile.TemporaryDirectory(prefix="dycraft-film-camera-") as tmp:
        out = Path(tmp) / "camera.json"
        try:
            command = [interp, str(_CAMERA_CHILD), str(blend), str(out)] + (["--all"] if all_frames else [])
            subprocess.run(_oom.tiered(command, _oom.WORKER), capture_output=True, text=True,
                           timeout=_CAMERA_TIMEOUT_S * (4 if all_frames else 1), env=blender_child_env())
        except subprocess.TimeoutExpired:
            _CAMERA_TIMED_OUT.add(key)
            return None
        except (OSError, subprocess.SubprocessError):
            return None
        found = _read_json(out)
    if not isinstance(found, dict):
        return None
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(found))
    except OSError:
        pass
    return found if found.get("ok") else None

