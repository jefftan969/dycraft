"""Read-only data access over `runs/` for the viewer.

Everything here is derived from artifacts a finished run already writes — no new
bookkeeping, no mutation. The one thing this module *generates* is the URDF: it
builds one from `ir.json` rather than serving `final_render/ir.urdf`, because
the generated form carries run-root-relative mesh paths
(`assets/meshes/<digest>.obj`) that resolve for every run, while the committed
copy resolves against its own localized `final_render/assets/` tree and only
exists once a run commits.

Everything IR-shaped below dispatches on `scene.ir_kind(ir)` — the payload —
rather than on `meta.json`'s `layer`, since a `plain` run commits no IR.
`viewer/scene.py` holds the scene-only half (catalog loading, asset rebasing).

An **animation** run breaks the "the run's IR is what you draw" assumption
object and scene runs share: its `ir.json` holds motion and no geometry, and
the scene it animates is the frozen `SceneIR` staged beside it as
`scene.json`. Every geometry path below therefore swaps in that file for such
a run and is otherwise the scene path unchanged; `viewer/animation.py` adds
the motion.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from .._layers import canonical_layer
from ..run_layout import INTERNAL, internal_dir, reader_dir
from . import animation as anim_mod
from . import films as films_mod
from . import scene as scene_mod

_COMPILE_DIR = "compile_reports"
_LATEST_REPORT = "latest.json"
_AGENT_TRANSCRIPT = "agent_transcript.jsonl"
_SUB_RUNS_DIR = "sub_runs"
_CONTEXT_SUMMARY = "context_summary.md"


class UrdfUnavailable(Exception):
    """Why a run cannot be shown in 3D — the message is the API's 404 detail.

    A missing `ir.json` is only one of the reasons, and it stopped being the
    common one once scene runs were served: a scene's geometry lives in its
    `catalog/`, so a committed SceneIR with no catalog snapshot has an IR and
    still nothing to draw. Carrying the reason as an exception keeps the two
    call sites in `app.py` from having to re-derive it.
    """

# `runs/<group>/<group>/<run>` and then some. Deep enough for any grouping a
# person would create by hand, shallow enough that a mislinked directory cannot
# send the walk off into the filesystem.
_MAX_DEPTH = 6


def _iter_run_dirs(runs_dir: Path, *, max_depth: int = _MAX_DEPTH) -> Iterator[Path]:
    """Every directory under `runs_dir` holding a `meta.json`.

    The walk stops descending the moment a directory *is* a run, so a run's own
    artifact tree (`assets/`, `renders/`, `final_render/`, ...) is never entered
    — a nested hit therefore always means a folder someone made to group runs,
    never a run's own subdirectory.

    `sub_runs/` is the single exception, because it is the one artifact
    subdirectory that holds runs: the scene and motion rewriters each write a
    full, ordinary run bundle there, and without descending their trajectories
    are unreachable from the viewer. The `.`/`_`-prefix filter still applies
    inside it.

    A symlinked directory is followed only at the top of `runs_dir`, and only
    when it is a run outside the tree: never walked through, so a link pointing
    back into the tree cannot loop, and two links to one run list it once.
    `resolve_run` serves exactly the links listed here.
    """
    stack: list[tuple[Path, int]] = [(runs_dir, 0)]
    # Real paths of the links followed. Nothing else needs resolving: the walk never passes through a link, so every
    # other entry's path under `runs_dir` is already its own and unique.
    seen: set[Path] = set()
    tree = runs_dir.resolve()
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith((".", "_")):
                continue
            # **A link is followed at the top of `runs_dir` only, and only to a run.** `install films` files each
            # film in `films/` this way, and what someone puts in their runs folder is theirs to view.
            linked = entry.is_symlink()
            if linked and depth > 0:
                continue
            if not entry.is_dir():
                continue
            if linked:
                real = entry.resolve()
                # A link into this same tree names a run the walk reaches under its own path, which is the one listed.
                if real in seen or real.is_relative_to(tree):
                    continue
                seen.add(real)
            # A run laid out for a reader is its folder, whose records are in `src/` (`run_layout`); the `src/`
            # folder itself is never listed as a run of its own.
            inner = internal_dir(entry)
            # A film installed from the published set has no run records until the viewer prepares it (`films`).
            if (inner / "meta.json").is_file() or films_mod.is_installed_film(entry):
                yield entry
                sub_runs = inner / _SUB_RUNS_DIR
                if not linked and depth + 1 < max_depth and sub_runs.is_dir():
                    stack.append((sub_runs, depth + 1))
            elif not linked and depth + 1 < max_depth:
                stack.append((entry, depth + 1))


def _wall_seconds(meta: dict) -> float | None:
    """How long the run took, preferring what its own session measured."""
    usage = meta.get("usage")
    if isinstance(usage, dict) and usage.get("wall_seconds") is not None:
        return usage["wall_seconds"]
    start, end = _iso(meta.get("started_at")), _iso(meta.get("finished_at"))
    if start and end:
        return round((end - start).total_seconds(), 1)
    return None


def _tokens(meta: dict) -> dict | None:
    """`{gross_input, output}` — the two figures worth a badge, or None.

    Deliberately not the whole usage block: the sidebar shows what the model
    read and what it wrote, and the cache split only matters once you are
    looking at cost, which is the Trajectory panel's business.

    A codex run's sub-agents are separate sessions (`usage.subagents`, see
    `agent.transcript_capture.subagent_usage`); their total rides along as
    `subagents`, so a run that spawned them does not look half its real cost.
    """
    usage = meta.get("usage")
    if not isinstance(usage, dict) or usage.get("error"):
        return None
    if usage.get("gross_input") is None and usage.get("output") is None:
        return None
    out = {"gross_input": int(usage.get("gross_input") or 0),
           "output": int(usage.get("output") or 0),
           "turns": int(usage.get("turns") or 0)}
    sub = usage.get("subagents")
    if isinstance(sub, dict) and sub.get("total") is not None:
        out["subagents"] = {"total": int(sub.get("total") or 0),
                            "threads": int(sub.get("threads") or 0)}
    return out


def _iso(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
#: Reserved first segment under the `files/` route for derived viewer assets.
#: They live in a cache outside the run directory, but the three.js loader
#: resolves every `<mesh filename>` against one base URL — so the URDF names
#: them here and `viewer/app.py` intercepts the prefix, exactly as it already
#: does for `model.urdf`. A run directory containing its own `__derived__/`
#: would shadow nothing: the interception happens first.
DERIVED_DIR = "__derived__"
DERIVED_URL_PREFIX = f"{DERIVED_DIR}/"


def derived_url_prefix(ir_id: str) -> str:
    """Where one object's derived assets are addressed, as a URL prefix.

    **The ir_id segment is load-bearing, not decorative.** A scene draws
    several objects, each with its own cache entry, and the manifests number
    their files per object — so every object has a `000_*.glb`. A flat
    `__derived__/000_body.glb` would be ambiguous across a scene's objects and
    unresolvable for all but one of them, since `viewer_bake.asset_path`
    resolves inside a single entry.
    """
    return f"{DERIVED_DIR}/{ir_id}/"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _natural_key(text: str) -> tuple:
    """Sort key that orders embedded numbers numerically.

    Plain lexicographic order files `kettle_r10` before `kettle_r2`, which reads
    as a bug in a list of repeats. Digit runs compare as ints, everything else
    casefolded; the `(0, int)` / `(1, str)` tags keep the tuple orderable when a
    numeric segment lines up against a textual one.
    """
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
                 for part in re.split(r"(\d+)", text) if part)


def _reasoning_effort(run_dir: Path, meta: dict) -> str | None:
    """Configured reasoning effort, with a fallback for older run bundles.

    New runs persist the launch setting in ``meta.json``.  Claude transcripts
    also record it on assistant events, which lets the viewer recover the
    value for existing self-contained bundles without guessing a backend
    default.  Stop at the first usable event so listing runs does not parse a
    potentially large transcript in full.
    """
    recorded = meta.get("reasoning_effort")
    if recorded is not None and str(recorded).strip():
        return str(recorded)

    transcript = run_dir / _AGENT_TRANSCRIPT
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # Most early transcript records are setup events.  Avoid a
                # JSON decode until a line could actually carry the field.
                if '"effort"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                effort = event.get("effort") if isinstance(event, dict) else None
                if effort is not None and str(effort).strip():
                    return str(effort)
    except OSError:
        pass
    return None


def list_runs(runs_dir: Path) -> list[dict]:
    """Run summaries ordered by name, discovered recursively under `runs_dir`.

    Sorted by `run_id` — the path — rather than by start time. Once runs are
    filed into folders, recency stops being a useful browsing order: a matrix
    run interleaves revisions, so newest-first scatters the same prompt's cells
    across the list. Sorting by path keeps a folder's contents together and
    lines the same prompt up at the same position under every revision, which
    is what makes two revisions comparable by eye.

    `run_id` is the run's path **relative to `runs_dir`** (`achieve/kettle`),
    not `meta.json`'s own `run_id`. Two runs filed under different folders may
    legitimately carry the same meta run_id — a copied bundle, or the same cell
    re-run into another tree — and the path is what every other endpoint
    resolves. For a run sitting directly in `runs/` the two coincide, so this
    changes nothing for a flat layout.

    `group` (parent folders, `""` at the top) and `name` (what to label it)
    split that path so the sidebar can mirror the directory tree without having
    to re-derive it.
    """
    out = []
    if not runs_dir.is_dir():
        return out
    for folder in _iter_run_dirs(runs_dir):
        entry = internal_dir(folder)
        # An installed film is listed from its prepared folder once it has one, else from its `run.json`.
        film = films_mod.is_installed_film(folder)
        if film:
            entry = films_mod.prepared(folder) or folder
        meta = _read_json(entry / "meta.json")
        if film and not isinstance(meta, dict):
            meta = films_mod.film_meta(folder)
        if not isinstance(meta, dict):
            continue
        rel = folder.relative_to(runs_dir).as_posix()
        ir = _read_json(entry / "ir.json")
        kind = scene_mod.ir_kind(ir)
        # An animation's instance count comes from the scene it animates, not
        # from its own IR — the badge should say what is on screen, and an
        # actor count says nothing about how big the scene is.
        anim_scene = (anim_mod.load_scene_ir(entry) if kind == "animation"
                      else None)
        out.append({
            "run_id": rel,
            "name": meta.get("run_id") or folder.name,
            "group": rel.rsplit("/", 1)[0] if "/" in rel else "",
            "layer": canonical_layer(meta.get("layer")),
            "model": meta.get("model"),
            "reasoning_effort": _reasoning_effort(entry, meta),
            "runtime": meta.get("runtime"),
            "outcome": meta.get("outcome"),
            "prompt": meta.get("prompt"),
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            # Wall time and tokens, when the run bundle recorded them. Read
            # straight off meta.json and never computed here: this runs for
            # every directory under `runs/`, and opening each one's transcript
            # to draw a sidebar badge would make listing cost as much as
            # reading. The per-turn breakdown stays with `/trajectory`.
            "wall_seconds": _wall_seconds(meta),
            "tokens": _tokens(meta),
            "has_ir": isinstance(ir, dict),
            # What the row's count badge means depends on the IR: an ObjectIR
            # has parts and joints, a SceneIR has instances and neither, an
            # animation run has frames. Report them all and let the
            # sidebar pick, rather than showing "0p/0j" for every scene and
            # animation run.
            "ir_kind": kind,
            "n_parts": len(ir.get("parts") or ()) if kind == "object" else 0,
            "n_joints": len(ir.get("joints") or ()) if kind == "object" else 0,
            "n_instances": (
                len(ir.get("instances") or ()) if kind == "scene"
                else len(anim_scene.get("instances") or ()) if anim_scene
                else 0),
            "n_frames": anim_mod.n_frames(ir) if kind == "animation" else 0,
        })
        if film:
            # Named by its brief and title rather than by its prompt, which is a paragraph. `installed` lets the
            # inspector leave out what a published film has none of: an agent trajectory, a context summary, parts.
            out[-1].update({"label": films_mod.film_label(folder), "installed": True})
        if film and entry == folder:
            # Not prepared yet: the viewer prepares it when it is opened (`/prepare`).
            out[-1].update({"has_ir": True, "ir_kind": "animation", "n_frames": films_mod.film_frames(folder) or 0,
                            "prepare": "needed"})
    out.sort(key=lambda r: _natural_key(r["run_id"]))
    return out


def resolve_run(runs_dir: Path, run_id: str) -> Path | None:
    """The run directory for a runs-dir-relative `run_id`, refusing escapes.

    `run_id` may name a nested folder (`achieve/kettle`), so `/` is allowed. Containment is judged by its segments,
    on the path under `runs_dir`, not by where that path resolves:

    - no absolute path, and no empty, `..`, or dot-prefixed segment, so no segment can climb out;
    - no symlink in any segment, except a link that is the whole `run_id`, a single top-level segment, as
      `install films` files a film;
    - and that link only when it leads out of the tree. A link back into the tree names a run served under its
      own path.

    These are the links `_iter_run_dirs` lists, so whatever the listing shows resolves, and nothing else through a
    link does.
    """
    candidate = _run_path(runs_dir, run_id)
    if candidate is None:
        return None
    # An installed film is served from its prepared folder, laid over the film's own (`films`); until it is
    # prepared, only `/prepare` reaches it (`resolve_film`).
    if films_mod.is_installed_film(candidate):
        return films_mod.prepared(candidate)
    # The folder holding the records, which every endpoint reads: `src/` for a run laid out for a reader.
    candidate = internal_dir(candidate)
    return candidate if (candidate / "meta.json").is_file() else None


def _run_path(runs_dir: Path, run_id: str) -> Path | None:
    """The folder `run_id` names under `runs_dir`, by `resolve_run`'s rules, whether or not it holds a run."""
    if not run_id:
        return None
    pure = PurePosixPath(run_id.replace("\\", "/"))
    if pure.is_absolute():
        return None
    parts = pure.parts
    if not parts or any(not p or p.startswith(".") for p in parts):
        return None
    root = runs_dir.resolve()
    for depth in range(1, len(parts) + 1):
        segment = root.joinpath(*parts[:depth])
        if not segment.is_symlink():
            continue
        if depth > 1 or len(parts) > 1 or segment.resolve().is_relative_to(root):
            return None
    return root.joinpath(*parts)


def resolve_film(runs_dir: Path, run_id: str) -> Path | None:
    """The installed film `run_id` names, by the same rules as `resolve_run`, else None."""
    path = _run_path(runs_dir, run_id)
    return path if path is not None and films_mod.is_installed_film(path) else None


def _compile_reports(run_dir: Path) -> list[dict]:
    """Per-step compile reports in step order, with verifier verdicts.

    `latest.json` is a duplicate of the highest-numbered report, so it is
    skipped — listing it would show the final compile twice. A verifier's
    evidence and suggested repairs stay on disk: the inspector shows each
    verdict and the exact adjudication, read out here, and the evidence was
    14.7 MB of a long film's 14.7 MB run detail, fetched on every selection.
    """
    directory = run_dir / _COMPILE_DIR
    if not directory.is_dir():
        return []
    reports = []
    for path in directory.glob("*.json"):
        if path.name == _LATEST_REPORT:
            continue
        report = _read_json(path)
        if not isinstance(report, dict):
            continue
        raw = [v for v in report.get("verifier_reports") or ()
               if isinstance(v, dict)]
        verifiers = [{"name": v.get("name"), "verdict": v.get("verdict")}
                     for v in raw]
        adjudication = next(
            (v["evidence"].get("exact_adjudication")
             for v in raw
             if isinstance(v.get("evidence"), dict)
             and v["evidence"].get("exact_adjudication")),
            None,
        )
        reports.append({
            "step": int(path.stem) if path.stem.isdigit() else None,
            "ok": report.get("ok"),
            "ir_hash": report.get("ir_hash"),
            "build_error": report.get("build_error"),
            "sandbox_violation": report.get("sandbox_violation"),
            "verifiers": verifiers,
            "exact_adjudication": adjudication,
        })
    reports.sort(key=lambda r: (r["step"] is None, r["step"]))
    return reports


def _part_summary(part: dict) -> dict:
    """What the agent knows about a part, for the Parts panel.

    `shape` is present only on `solid` parts; `fill_ratio` is the single number
    that says how badly a bounding box misrepresents the geometry.
    """
    params = part.get("params") or {}
    shape = params.get("shape") if isinstance(params, dict) else None
    out = {
        "name": part.get("name"),
        "primitive": part.get("primitive"),
        "material": part.get("material"),
        "color": part.get("color"),
        "source": params.get("source") if isinstance(params, dict) else None,
    }
    if isinstance(shape, dict):
        out.update({
            "volume": shape.get("volume"),
            "fill_ratio": shape.get("fill_ratio"),
            "face_census": shape.get("face_census"),
            "center": shape.get("center"),
        })
    return out


def _sub_runs(run_dir: Path, run_id: str) -> list[dict]:
    """The child runs in this run's `sub_runs/` (its rewriters), in name order.

    `run_id` is prefixed onto each child's directory name so the value is
    directly addressable by every other endpoint — the sub-run bundles are
    reachable exactly like top-level runs (see `_iter_run_dirs`), so the
    inspector can link straight into one.
    """
    directory = run_dir / _SUB_RUNS_DIR
    if not directory.is_dir():
        return []
    out = []
    for entry in sorted(directory.iterdir(), key=lambda p: _natural_key(p.name)):
        if entry.name.startswith((".", "_")) or not entry.is_dir():
            continue
        meta = _read_json(entry / "meta.json")
        if not isinstance(meta, dict):
            continue
        out.append({
            # Addressed through `src/` for a run laid out for a reader, which is where its sub-runs are.
            "run_id": (f"{run_id}/{INTERNAL}" if reader_dir(run_dir) != run_dir else run_id)
                      + f"/{_SUB_RUNS_DIR}/{entry.name}",
            "name": entry.name,
            "layer": canonical_layer(meta.get("layer")),
            "model": meta.get("model"),
            "outcome": meta.get("outcome"),
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            # Wall time and tokens, when the run bundle recorded them. Read
            # straight off meta.json and never computed here: this runs for
            # every directory under `runs/`, and opening each one's transcript
            # to draw a sidebar badge would make listing cost as much as
            # reading. The per-turn breakdown stays with `/trajectory`.
            "wall_seconds": _wall_seconds(meta),
            "tokens": _tokens(meta),
            "has_ir": (entry / "ir.json").is_file(),
        })
    return out


def _ir_summary(run_dir: Path, ir: dict) -> dict:
    """The IR reduced to what the inspector shows, per IR shape.

    Both shapes fill `parts` and `joints` so the Parts panel needs no special
    case; for a scene those are the per-instance parts and joints flattened
    under the same `<instance>__` prefix the URDF uses, which is what makes a
    row in the panel name the same thing as a link in the 3D view.

    An animation reports its own `id`/`name` and an `animation` block, but
    every geometry field is filled from the staged scene — so the Parts panel
    and the joint sliders work on an animation run exactly as on the scene run
    it animates, which is the point: those sliders are what the timeline drives.
    """
    kind = scene_mod.ir_kind(ir)
    summary: dict = {
        "kind": kind,
        "id": ir.get("id"),
        "name": ir.get("name"),
        "root": ir.get("root"),
        "assertions": ir.get("assertions") or [],
        "collision_allowances": ir.get("collision_allowances") or [],
    }
    if kind == "animation":
        scene_ir = anim_mod.load_scene_ir(run_dir)
        summary["animation"] = anim_mod.summary(run_dir, ir, scene_ir)
        summary["clip"] = anim_mod.clip_rel(run_dir, ir)
        # A native film draws itself from its own track, so it needs no staged scene (`run_urdf`): an installed
        # film has none, and the viewer must not report one missing.
        summary["native"] = _native_film(ir)
        if scene_ir is None:
            summary.update({"parts": [], "joints": []})
            return summary
        # Everything below is the scene branch verbatim, on the staged scene.
        # `scene_id` and the scene's own name are kept distinct from the
        # animation's so the inspector can show which scene is being animated.
        summary["scene_id"] = scene_ir.get("id")
        summary["scene_name"] = scene_ir.get("name")
        ir, kind = scene_ir, "scene"
    if kind == "scene":
        # One read of the catalog feeds all three: it is a directory of files,
        # and a scene with a large catalog would otherwise re-parse it per
        # field of a single response.
        catalog = scene_mod.load_catalog(run_dir)
        parts, joints = scene_mod.scene_parts_and_joints(
            ir, catalog, _part_summary)
        resolved, total = scene_mod.catalog_coverage(ir, catalog)
        summary.update({
            "parts": parts,
            "joints": joints,
            "instances": scene_mod.instance_summaries(ir, catalog),
            "groups": ir.get("groups") or [],
            "room": ir.get("room"),
            "catalog_resolved": resolved,
            "catalog_total": total,
        })
    else:
        summary.update({
            "parts": [_part_summary(p) for p in ir.get("parts") or ()],
            "joints": ir.get("joints") or [],
        })
    return summary


def run_detail(run_dir: Path, run_id: str | None = None) -> dict:
    """Everything the inspector shows for one run.

    `run_id` is the runs-dir-relative path the caller addressed this run by.
    It is needed only to make `sub_runs` entries linkable, so it defaults to
    the directory name — correct for a run sitting directly in `runs/`, and
    the callers that can do better (the API routes) pass the real path.
    """
    meta = _read_json(run_dir / "meta.json") or {}
    # Recorded runs may carry a layer's older name; the viewer shows the current one.
    if isinstance(meta.get("layer"), str):
        meta = {**meta, "layer": canonical_layer(meta["layer"])}
    ir = _read_json(run_dir / "ir.json")
    detail: dict = {
        "run_id": meta.get("run_id") or run_dir.name,
        "reasoning_effort": _reasoning_effort(run_dir, meta),
        "meta": meta,
        "compile_reports": _compile_reports(run_dir),
        "source_files": sorted(
            p.name for p in run_dir.glob("*.py")
            if p.name not in ("harness.py",)
        ),
        "renders": sorted(p.name for p in (run_dir / "renders").glob("*.png"))
        if (run_dir / "renders").is_dir() else [],
        "has_ir": isinstance(ir, dict),
        # Kept separate from source_files: this is prose with a dedicated
        # rendered tab, not authoring source for the code inspector.
        "has_context_summary": resolve_file(
            run_dir, _CONTEXT_SUMMARY) is not None,
        "sub_runs": _sub_runs(run_dir, run_id or run_dir.name),
    }
    if isinstance(ir, dict):
        detail["ir"] = _ir_summary(run_dir, ir)
    return detail


def run_urdf(run_dir: Path, *, derived: bool = False, flat: Mapping[str, dict] | None = None) -> str:
    """URDF text for the run's IR.

    `flat` is `{ir_id: parts}` from `derived.flat_colors`: each shader part listed there is drawn in its flat
    color, roughness, and metallic (`render.viewer_flat.flat_ir`), the default view of a shader material. A part
    not listed draws its IR color.

    `derived=True` asks for the **viewer-only** variant, whose mesh filenames
    point at the baked atlas GLBs instead of the staged OBJs — for an object
    run its own parts, for a scene run each catalog object that has a complete
    bake. Everything else about the document is identical, so the joint tree,
    the link frames, and the collision proxies are the ones the canonical
    exporter produces — articulation and Preview Motion work on the same graph,
    and only the geometry each link *draws* changes.

    The request is ignored per object (rather than refused) when there is no
    complete bake, so a scene with one un-baked object returns a **mixture**
    and the viewport falls back to the preview appearance for just that one.

    Nothing canonical is touched: this generates from a COPY of the IR
    (`render.viewer_bake.substituted_ir`), and `render/urdf_export.py`, the
    stored `renders/ir.urdf` and the IK solver export never see it. This function
    has no non-viewer callers.

    Raises `UrdfUnavailable` — never returns a URDF that would draw nothing —
    when the run has no IR, when its IR is an unrecognized shape, or when a
    SceneIR's instances cannot be resolved against a catalog. The one exception
    is a native film with no staged scene, such as an installed film: its
    playback draws the film from its own track, under a bare `world` link.

    For an animation run the URDF is the *staged* scene at rest: an AnimationIR
    has no geometry, and the frame the viewport shows is produced by posing
    this graph, not by re-exporting it per frame.
    """
    ir = _read_json(run_dir / "ir.json")
    kind = scene_mod.ir_kind(ir)
    if kind == "animation":
        scene_ir = anim_mod.load_scene_ir(run_dir)
        if scene_ir is None and _native_film(ir):
            # **The one URDF that draws nothing, on purpose.** A film installed
            # from the published set and staged by `benchmark.stage` has its
            # evaluated geometry and motion, but no frozen scene to stage: the
            # dataset ships the Blender project, not the scene's catalog. Native
            # playback draws the whole film from its track, and mounts under
            # the URDF's `world` link, so a bare root is what it needs.
            return _bare_film_urdf(ir)
        if scene_ir is None:
            raise UrdfUnavailable(
                f"animation run has no {anim_mod.SCENE_JSON} — the frozen "
                f"scene is staged into the run dir at launch, so a run "
                f"missing it was not produced by `--layer animation`")
        ir, kind = scene_ir, "scene"
    if kind is None:
        if isinstance(ir, dict):
            raise UrdfUnavailable(
                "ir.json is neither an ObjectIR (parts) nor a SceneIR "
                "(instances)")
        raise UrdfUnavailable(
            "run has no ir.json — geometry is written on commit")
    # One lookup for both layers. Empty when `derived=False`, when the layer is
    # refused, or when nothing has been baked yet — and `_derived_substitution`
    # is then the identity, which is what makes the flag a request rather than
    # an assertion.
    manifests: dict = {}
    if derived:
        from . import derived as derived_mod

        manifests = derived_mod.manifests_for(run_dir)

    flat = flat or {}

    if kind == "object":
        from ..render.urdf_export import object_ir_to_urdf

        return object_ir_to_urdf(_derived_substitution(_flat(ir, flat), manifests))
    catalog = scene_mod.load_catalog(run_dir)
    resolved, total = scene_mod.catalog_coverage(ir, catalog)
    if total and not resolved:
        raise UrdfUnavailable(
            f"scene has {total} instance(s) but none resolve against "
            f"{scene_mod.CATALOG_DIR}/ — the object catalog is snapshotted "
            f"next to the scene IR on a successful compile")
    if flat:
        catalog = {ref: _flat(obj, flat) for ref, obj in catalog.items()}
    if manifests:
        # **Substituted in the catalog, not in the SceneIR.** `scene_ir_to_urdf`
        # replicates each catalog object's own link tree and reads every mesh
        # filename straight out of that object's `params`, so rewriting the
        # catalog reaches every instance — including repeated ones, which share
        # the entry and therefore the URL — without this layer knowing anything
        # about `<instance>__<part>` link naming.
        catalog = {ref: _derived_substitution(obj, manifests)
                   for ref, obj in catalog.items()}
    return scene_mod.scene_urdf(ir, catalog)


def _flat(ir: dict, flat: Mapping[str, dict]) -> dict:
    """`ir` with its shader parts in their flat colors, or `ir` itself when none were read for it."""
    parts = flat.get(str(ir.get("id") or ""))
    if not parts:
        return ir
    from ..render.viewer_flat import flat_ir

    return flat_ir(ir, parts)


def _derived_substitution(ir: dict, manifests: Mapping[str, dict]) -> dict:
    """`ir` with its mesh paths pointed at its own derived assets.

    Returns `ir` **unchanged** when this object has no complete bake, so a
    partially baked scene degrades to a mixture of derived GLBs and staged OBJs
    rather than losing the objects that have not been baked yet.
    """
    ir_id = str(ir.get("id") or "")
    manifest = manifests.get(ir_id)
    if manifest is None:
        return ir
    from ..render.viewer_bake import substituted_ir

    return substituted_ir(ir, manifest, derived_url_prefix(ir_id))


def _native_film(ir) -> bool:
    """Whether `ir` is a native film, which draws itself from its evaluated track."""
    return isinstance(ir, dict) and ir.get("schema") == anim_mod.NATIVE_SCHEMA


def _bare_film_urdf(ir: dict) -> str:
    """A URDF holding only the `world` link a native film's playback mounts under."""
    import xml.etree.ElementTree as ET
    robot = ET.Element("robot", {"name": str(ir.get("id") or "film")})
    ET.SubElement(robot, "link", {"name": "world"})
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"


def run_animation(run_dir: Path) -> dict:
    """The scrubbable timeline for an animation run.

    Its own endpoint rather than a field on `run_detail` because it is the one
    dense payload here — per-frame poses for every instance that moves — and
    every other run kind would pay for it on selection. Raises
    `UrdfUnavailable` for the same reasons `run_urdf` does, plus "this is not
    an animation run"; the two are asked for together, so one failure vocabulary
    keeps the frontend from needing a second error path.
    """
    ir = _read_json(run_dir / "ir.json")
    if scene_mod.ir_kind(ir) != "animation":
        raise UrdfUnavailable("run is not an animation run")
    if _native_film(ir) and anim_mod.load_scene_ir(run_dir) is None:
        # The native timeline reads only the film's own records, so a staged
        # film with no frozen scene still has one (see `run_urdf`).
        try:
            return anim_mod.timeline(run_dir, ir)
        except anim_mod.TimelineUnavailable as exc:
            raise UrdfUnavailable(str(exc)) from exc
    scene_ir = anim_mod.load_scene_ir(run_dir)
    if scene_ir is None:
        raise UrdfUnavailable(
            f"animation run has no {anim_mod.SCENE_JSON}")
    catalog = scene_mod.load_catalog(run_dir)
    if not catalog:
        raise UrdfUnavailable(
            f"animation run has no {scene_mod.CATALOG_DIR}/ — the objects the "
            f"scene places are staged next to it at launch")
    try:
        return anim_mod.timeline(run_dir, ir)
    except anim_mod.TimelineUnavailable as exc:
        raise UrdfUnavailable(str(exc)) from exc


def resolve_file(run_dir: Path, rel_path: str) -> Path | None:
    """A file inside the run dir, refusing anything that escapes it.

    For a prepared film, a file it does not hold, such as the video, is looked up in the film's own folder, under
    the same rule (`films.locate`).
    """
    return films_mod.locate(run_dir, rel_path)
