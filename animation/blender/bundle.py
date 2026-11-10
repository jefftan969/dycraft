"""Portable file manifests; no claim of cross-version Blender determinism."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from ...sdk.ir import content_hash
from . import SCHEMA


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"bundle path escapes root: {relative}")
    return p


#: Never part of a manifest or a workspace hash, whatever produced them.
#: Blender's own save backups: redundant with the file they sit beside, and
#: excluded here as well as suppressed at the save (`worker.py`) so a manifest
#: stays correct even if some path resets the preference.
EXCLUDED_SUFFIXES = (".blend1", ".blend2", ".pyc")
#: Directories never part of a manifest or a workspace hash: the bytecode
#: caches Python leaves wherever a script imports a module from the project.
#: One run sealed 68 `.pyc` files into its IR, and a script merely run again
#: changed the project's hash.
EXCLUDED_DIRS = ("__pycache__",)
#: The read-only `.claude/` and `.mcp.json` mask entries Claude Code's sandbox
#: leaves in any directory its shell ran in. Not the author's files: copying
#: one onto the last copy's read-only mask failed `blender()` calls, and
#: hashing one changed the design's hash.
SANDBOX_MASKS = (".claude", ".mcp.json")
#: `shutil.copytree(..., ignore=skip_sandbox_masks)` for the author's trees.
skip_sandbox_masks = shutil.ignore_patterns(*SANDBOX_MASKS)


def excluded(path, root=None):
    """Whether `path` stays out of hashes and manifests. With `root`, a path
    inside a sandbox mask or a bytecode cache under it is excluded too."""
    if path.name.endswith(EXCLUDED_SUFFIXES):
        return True
    parts = path.relative_to(root).parts if root is not None else (path.name,)
    return any(part in SANDBOX_MASKS or part in EXCLUDED_DIRS for part in parts)


def tree_hash(root, *, exclude=()):
    """Hash all workspace dependencies, rejecting paths that escape a snapshot.

    `exclude` holds workspace-relative paths the caller hashes itself:
    `check_cache_key` hashes the plan's document without the delivery
    metadata that no check reads.
    """
    root = Path(root)
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"workspace symlinks are not portable: {path}")
        if not path.is_file() or excluded(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        entries[relative] = file_hash(path)
    return content_hash(entries)


#: The recipe recorded for an evaluation record with no `render` block (one
#: written before `render_config`): the values such a record was drawn with.
#: Nothing new is rendered this way. The stored recipe is `worker._DELIVERY`,
#: and a current record carries its own `render` block.
_LEGACY_RENDER = {"engine": "BLENDER_EEVEE_NEXT", "width": 1280,
                  "height": 720, "samples": 64}


def _render_block(evaluation):
    """The render recipe to record, from the worker's own report.

    Falls back to the historical defaults for an evaluation produced before
    `worker.render_config` existed — a cached or archived record — because the
    alternative is a manifest that omits the field entirely and a consumer that
    has to special-case its absence.
    """
    reported = evaluation.get("render")
    if not isinstance(reported, dict):
        return dict(_LEGACY_RENDER)
    return {**_LEGACY_RENDER, **reported}


def make_manifest(root, *, scene_ref, evaluation, prefix="bundle"):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"bundle contains a symlink: {path}")
        if path.is_file() and not excluded(path, root):
            files[f"{prefix}/{path.relative_to(root).as_posix()}"] = file_hash(path)
    fps = evaluation["fps"]
    ir = {"schema": SCHEMA, "scene_ref": scene_ref,
          "blend": f"{prefix}/animation.blend", "files": files,
          "evaluation": f"{prefix}/evaluation/index.json",
          "fps": fps, "n_frames": evaluation["n_frames"],
          "duration": (evaluation["n_frames"] - 1) / fps,
          "frame_start": evaluation["frame_start"],
          "events": evaluation.get("declarations", {}).get("events", []),
          # What actually drew, reported by the process that drew it.
          #
          # Not rebuilt from `os.environ` here: that is the **daemon's**
          # environment, and `runtime.run_worker` gives the child a closed
          # `env=` dict, so the manifest could assert an engine the worker
          # never used, and `validate_manifest` hashes the manifest rather than
          # the pixels. `worker.render_config` travels out through the
          # evaluation record instead.
          "render": _render_block(evaluation),
          "blender_version": evaluation["blender_version"]}
    ir["id"] = content_hash(ir)
    return ir


def validate_manifest(run_dir, ir):
    if ir.get("schema") != SCHEMA:
        raise ValueError("not a native Blender animation bundle")
    if content_hash({k: v for k, v in ir.items() if k != "id"}) != ir.get("id"):
        raise ValueError("animation manifest hash mismatch")
    for rel, expected in ir["files"].items():
        if file_hash(safe_path(run_dir, rel)) != expected:
            raise ValueError(f"animation asset hash mismatch: {rel}")
    if ir["blend"] not in ir["files"] or ir["evaluation"] not in ir["files"]:
        raise ValueError("blend and evaluation must be hashed bundle members")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def stage_prior(source, destination):
    """Stage a native workspace/bundle, preserving its relative dependencies."""
    import shutil
    from ...run_layout import internal_dir
    source, destination = Path(source), Path(destination)
    # A run laid out for a reader: its folder, or the `animation.blend` at its top, stages from the sealed bundle in
    # `src/`, as an older run's folder does.
    if source.suffix == ".blend" and source.name == "animation.blend" \
            and internal_dir(source.parent) != source.parent:
        source = source.parent
    if source.is_dir():
        source = internal_dir(source)
    if source.is_dir():
        if (source / "ir.json").is_file():
            ir = json.loads((source / "ir.json").read_text())
            validate_manifest(source, ir)
            source = safe_path(source, ir["blend"]).parent
        elif (source / "bundle/animation.blend").is_file():
            source = source / "bundle"
    elif source.suffix == ".blend":
        if source.name != "animation.blend":
            raise ValueError("prior native workspace must name its main file animation.blend")
        source = source.parent
    else:
        raise ValueError("--prior-animation takes the folder, bundle, or workspace of a prior run, or its animation.blend")
    if not (source / "animation.blend").is_file():
        raise ValueError("prior workspace has no animation.blend")
    tree_hash(source)
    def ignore(directory, names):
        # An animation bundle's design_state/baseline.json is source provenance,
        # needed for incremental updates. Only a root-level baseline.json is
        # regenerated from its frozen SceneIR on import.
        return [n for n in names if Path(directory) == source and
                n in ("evaluation", "edit_history", "baseline.json", "film")]
    shutil.copytree(source, destination, ignore=ignore)
