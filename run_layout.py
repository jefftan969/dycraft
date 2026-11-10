"""Where a run's files are, what its top level holds, and how a delivered run is reopened for a revision.

A run launched from the command line keeps everything the agent and the harness use in `<run>/src/`, the agent's
working folder, and holds what a reader came for at its top level: the film and its Blender project, the views and
URDF of an object or scene, or a plain run's answer, with a README. `publish` writes that top level, and nothing else
does. Older runs, sub-runs and published films are one flat folder. `internal_dir` and `reader_dir` accept either, so
a command given either folder finds the same run.

`reopen_delivery` sets a delivered run's delivery aside so a resumed session can change the work and deliver again.
`finish()` refuses while a delivery is in place, and the committed state is held in memory only, so with the delivery
moved a new harness daemon accepts edits again. Nothing is deleted: each delivery goes to the next
`prior_deliveries/<n>/` with a copy of `meta.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ._layers import canonical_layer

INTERNAL = "src"
PRIOR = "prior_deliveries"

#: What each layer's delivery is, relative to the run's internal folder. Moved as a unit, in this order; a missing
#: entry is skipped. The animation layer's `scene.json` and `catalog/` are the frozen scene staged for the viewer:
#: they move too, because staging refuses a `scene.json` that names a different scene, so a revision that changes the
#: scene could not be drawn.
DELIVERY = {
    "animation": ("ir.json", "bundle", "finish_result.json", "scene.json", "catalog"),
    "object": ("ir.json", "final_render", "final_render_failed", "renders/ir.urdf"),
    "scene": ("ir.json", "final_render", "final_render_failed", "renders/ir.urdf"),
    "plain": ("result.json",),
}
#: The file whose presence means a layer has delivered.
DELIVERED_BY = {"animation": "bundle", "object": "ir.json", "scene": "ir.json", "plain": "result.json"}

#: The message a resumed session receives first, per layer: what happened to its delivery. Agent-facing.
NOTES = {
    "animation": ("Your finish() in the previous session succeeded (outcome: {outcome}). That delivery, ir.json and "
                  "bundle/, has been moved to {prior}/ so the project can change and be delivered again. Everything "
                  "else (native/work, design/, your sources, journal, contributions, renders and notes) is exactly as "
                  "you left it, and the harness accepts edits again."),
    "object": ("Your finish() in the previous session succeeded (outcome: {outcome}). That delivery (ir.json and its "
               "final renders) has been moved to {prior}/ so the {layer} can change and be delivered again. Everything "
               "else, your source and its assets included, is exactly as you left it, and the harness accepts edits "
               "again."),
    "plain": "Your previous answer has been moved to {prior}/result.json.",
}
NOTES["scene"] = NOTES["object"]


def internal_dir(path) -> Path:
    """The folder holding a run's records: `<path>/src` when it holds `meta.json` and `path` does not, else `path`."""
    path = Path(path)
    inner = path / INTERNAL
    return inner if (inner / "meta.json").is_file() and not (path / "meta.json").is_file() else path


def reader_dir(path) -> Path:
    """The folder a reader opens: the run folder above a `src/` that holds its records, else `path` itself."""
    path = Path(path)
    if path.name == INTERNAL and (path / "meta.json").is_file() and not (path.parent / "meta.json").is_file():
        return path.parent
    return path


def top_of(internal) -> Path:
    """The top level of the run whose records go in `internal`: its parent for a `src/` folder, else `internal`.

    Unlike `reader_dir`, this does not wait for `meta.json`: the launcher writes a refusal's README before the run
    has any other record.
    """
    internal = Path(internal)
    if internal.name == INTERNAL and not (internal.parent / "meta.json").is_file():
        return internal.parent
    return internal


def new_run(runs_dir, run_id: str) -> Path:
    """The internal folder for a launch: `<runs_dir>/<run_id>/src`, made, or `<runs_dir>/<run_id>` itself when it
    already holds a flat run."""
    top = Path(runs_dir) / run_id
    if (top / "meta.json").is_file():
        return top.resolve()
    inner = top / INTERNAL
    inner.mkdir(parents=True, exist_ok=True)
    return inner.resolve()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _layer(run_dir: Path) -> str:
    meta = _read_json(run_dir / "meta.json") or {}
    layer = canonical_layer(meta.get("layer"))
    if layer not in DELIVERY:
        raise RuntimeError(f"{run_dir}: meta.json names no layer this can reopen ({meta.get('layer')!r})")
    return layer


def _outcome(run_dir: Path, layer: str) -> str:
    if layer == "plain":
        return "answered"
    ir = _read_json(run_dir / "ir.json") or {}
    meta = _read_json(run_dir / "meta.json") or {}
    return ir.get("outcome") or meta.get("outcome") or "committed"


def reopen_delivery(run_dir) -> dict | None:
    """Move a delivered run's delivery to the next `prior_deliveries/<n>/`; None when the run has not delivered.

    `run_dir` is the run's internal folder (`internal_dir`). Returns `{"prior": "prior_deliveries/<n>", "outcome",
    "note"}`: where the delivery went, relative to `run_dir` (the agent's `$DYCRAFT_RUN_DIR`), how it ended, and the
    message for the resumed session. A failure part-way puts back what had moved and raises `RuntimeError`, so the
    run is either reopened or unchanged.
    """
    run_dir = Path(run_dir)
    layer = _layer(run_dir)
    if not (run_dir / DELIVERED_BY[layer]).exists():
        return None
    outcome = _outcome(run_dir, layer)
    n = 1
    while (run_dir / PRIOR / str(n)).exists():
        n += 1
    prior = f"{PRIOR}/{n}"
    target = run_dir / prior
    moved: list[tuple[Path, Path]] = []
    new_parent = not (run_dir / PRIOR).exists()
    try:
        target.mkdir(parents=True)
        shutil.copy2(run_dir / "meta.json", target / "meta.json")
        for rel in DELIVERY[layer]:
            src = run_dir / rel
            if not (src.exists() or src.is_symlink()):
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            moved.append((src, dst))
    except OSError as error:
        for src, dst in reversed(moved):
            try:
                dst.rename(src)
            except OSError:
                pass
        shutil.rmtree(target if not new_parent else run_dir / PRIOR, ignore_errors=True)
        raise RuntimeError(f"could not move the delivery to {target}: {error}") from error
    return {"prior": prior, "outcome": outcome,
            "note": NOTES[layer].format(outcome=outcome, prior=prior, layer=layer)}


# ---------------------------------------------------------------------------------------------------- the top level

#: Everything `publish` may put at a run's top level. It owns these names and no others: each is rewritten from the
#: run's records, or removed when the run no longer has it (a delivery set aside by `--resume`, for one).
TOP_LEVEL = ("README.md", "film.mp4", "animation.blend", "notes", "views", "object.urdf", "scene.urdf", "meshes",
             "answer.md")
#: What `publish` placed, kept beside the records so it can tell what is already current.
_PUBLISHED = ".published.json"


def _place(src: Path, dst: Path) -> None:
    """`dst` as `src`'s bytes: a hardlink, so it takes no space, or a copy where links are not possible."""
    try:
        if os.path.samefile(src, dst):
            return                                  # already this file; `rename` onto a link of itself does nothing
    except OSError:
        pass
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _place_tree(src: Path, dst: Path, *, rename=None) -> int:
    """`src`'s files under `dst`, each through `_place`. Returns how many."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    n = 0
    for path in sorted(src.rglob("*")):
        if path.is_file() and not path.is_symlink():
            target = tmp / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            _place(path, target)
            n += 1
    _remove(dst)
    if n:
        os.replace(tmp, dst)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return n


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


_BLEND_SCRIPT = r"""
import bpy, json, os, sys
src, dst = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.open_mainfile(filepath=src)
# Only paths that resolve: a film can carry stale relative names (a library reference from the machine it was made
# on) that open nothing from the bundle either, and remapping them would only cost a copy.
relative = sorted({p for p in bpy.utils.blend_paths(absolute=False, packed=False)
                   if p.startswith("//") and os.path.exists(bpy.path.abspath(p))})
if relative:
    bpy.ops.wm.save_as_mainfile(filepath=dst, copy=True, relative_remap=True)
print("PUBLISH_BLEND " + json.dumps(relative), flush=True)
"""


def _place_blend(src: Path, dst: Path) -> str:
    """The film's project at the top level, opening as it does in the bundle.

    A blend can name files beside it (an Alembic or Mesh Cache file, an image) by a path relative to itself, which
    would not resolve from the top level. Blender is asked: with no such file the top-level blend is a hardlink;
    with one, Blender saves a copy whose relative paths point into the bundle. Returns `"link"` or `"remapped"`.
    Without a Blender interpreter it is a hardlink.
    """
    from .render._blender_env import blender_child_env, blender_python
    interp = blender_python()
    tmp = dst.with_name(f".{dst.name}.tmp")
    if interp:
        if tmp.exists():
            tmp.unlink()
        from . import _oom
        ran = subprocess.run(_oom.tiered([interp, "-c", _BLEND_SCRIPT, str(src), str(tmp)], _oom.WORKER),
                             capture_output=True, text=True, timeout=600, env=blender_child_env())
        line = next((l for l in ran.stdout.splitlines() if l.startswith("PUBLISH_BLEND ")), None)
        if line is None:
            raise RuntimeError("Blender could not read the film's project: "
                               + ((ran.stderr or ran.stdout).strip().splitlines() or ["no output"])[-1])
        if json.loads(line[len("PUBLISH_BLEND "):]):
            os.replace(tmp, dst)
            return "remapped"
    _place(src, dst)
    return "link"


def _stamp(path: Path) -> list:
    st = path.stat()
    return [st.st_ino, st.st_size, st.st_mtime_ns]


def _rewritten_urdf(text: str) -> str:
    """`final_render/ir.urdf`'s text with its `assets/...` paths pointing into the top level's `meshes/`."""
    return text.replace('filename="assets/', 'filename="meshes/')


def publish(internal) -> dict:
    """Write a run's top level from its records: always `README.md`; the deliverables once the run has them.

    Only for a run whose records are in a `src/` folder; any other run is left untouched and `{}` returned. Never
    raises: the top level is a view of the run, and failing to write it must not fail what called it (a commit, the
    launcher's exit). Returns what was written, and `errors` for anything that could not be.
    """
    from . import run_readme
    internal = Path(internal)
    top = top_of(internal)
    if top == internal:
        return {}
    written, errors = [], []
    try:
        previous = json.loads((internal / _PUBLISHED).read_text())
    except (OSError, ValueError):
        previous = {}
    record = {}
    meta = _read_json(internal / "meta.json") or {}
    layer = canonical_layer(meta.get("layer"))
    wanted: set[str] = {"README.md"}

    def attempt(name, fn):
        try:
            fn()
            written.append(name)
        except Exception as error:                      # noqa: BLE001 -- see the docstring
            errors.append(f"{name}: {type(error).__name__}: {error}")

    ir = _read_json(internal / "ir.json") or {}
    if layer == "animation" and (internal / "bundle").is_dir() and ir.get("film"):
        film = internal / ir["film"]
        blend = internal / (ir.get("blend") or "bundle/animation.blend")
        if film.is_file():
            wanted.add("film.mp4")
            attempt("film.mp4", lambda: _place(film, top / "film.mp4"))
        if blend.is_file():
            wanted.add("animation.blend")
            stamp = _stamp(blend)
            if previous.get("blend") == stamp and (top / "animation.blend").is_file():
                record["blend"], record["blend_kind"] = stamp, previous.get("blend_kind")
            else:
                def blend_step():
                    record["blend_kind"] = _place_blend(blend, top / "animation.blend")
                    record["blend"] = stamp
                attempt("animation.blend", blend_step)
        notes = internal / "bundle" / "notes"
        if notes.is_dir():
            wanted.add("notes")
            attempt("notes", lambda: _place_tree(notes, top / "notes"))
    elif layer in ("object", "scene") and (internal / "final_render").is_dir() and ir:
        views = internal / "final_render"
        pngs = sorted(views.glob("*.png"))
        if pngs:
            wanted.add("views")

            def views_step():
                tmp = top / ".views.tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                for png in pngs:
                    _place(png, tmp / png.name)
                _remove(top / "views")
                os.replace(tmp, top / "views")
            attempt("views", views_step)
        urdf = views / "ir.urdf"
        if urdf.is_file():
            name = f"{layer}.urdf"
            wanted.add(name)
            attempt(name, lambda: (top / name).write_text(_rewritten_urdf(urdf.read_text())))
            if (views / "assets").is_dir():
                wanted.add("meshes")
                attempt("meshes", lambda: _place_tree(views / "assets", top / "meshes"))
    elif layer == "plain":
        result = _read_json(internal / "result.json")
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            wanted.add("answer.md")
            attempt("answer.md", lambda: (top / "answer.md").write_text(result["text"].rstrip() + "\n"))

    attempt("README.md", lambda: (top / "README.md").write_text(run_readme.render(internal, sorted(wanted))))
    for name in TOP_LEVEL:
        if name not in wanted and ((top / name).exists() or (top / name).is_symlink()):
            try:
                _remove(top / name)
            except OSError as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
    try:
        (internal / _PUBLISHED).write_text(json.dumps(record))
    except OSError:
        pass
    out = {"top": str(top), "written": written}
    if errors:
        out["errors"] = errors
        print(f"[dycraft] could not write all of {top}'s top level: " + "; ".join(errors), file=sys.stderr)
    return out


__all__ = ["INTERNAL", "PRIOR", "DELIVERY", "NOTES", "TOP_LEVEL", "internal_dir", "reader_dir", "top_of", "new_run",
           "reopen_delivery", "publish"]
