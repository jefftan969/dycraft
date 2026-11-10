"""`load_object` and `load_asset` — definitions a `scene.py` did not construct.

Two doors, because a scene build has two ways to come by a definition it did
not write inline. `load_object` builds another file's object program: the
geometry is this run's, the source is the author's to edit, and the loaded file
joins the dependency closure. `load_asset` returns a frozen ObjectIR the
launcher staged — geometry that already existed, which the author places but
cannot rebuild. See `agent/assets/preload_asset.py` for why that second
door exists and what the launcher checks before opening it.

The rest of this file is about the first one.

**Why a loader and not `import`.** The obvious way to let one `scene.py` reuse
a `object.py` is to relax the AST ban and let the author write `import fridge`.
Four things make that the wrong mechanism here, and only the first is about
taste:

* **The warm CAD pool.** `_worker_pool` reuses a child interpreter across
  builds, and its safety argument is explicit: reuse is sound *because*
  `build_ir_with_layer` re-establishes everything a build can dirty — the
  builder singletons, the asset session, the Shape compiler, a fresh globals
  dict. `sys.modules` is the one thing it cannot re-establish, so an imported
  helper would be cached in the child and an edit to it would silently not take
  effect until the pool happened to recycle that child. An agent iterating on
  geometry would see its own edit ignored, nondeterministically, with no error
  — the worst failure this harness could hand someone.
* **The sandbox invariant.** Imports stay refused, so the boundary does not
  move. `load_object` is a *name* in the overlay, which is the mechanism the
  sandbox already relies on for `Object`, `Box` and everything else.
* **Replayability.** `fixed_source` re-runs an authored program outside its
  session. A loader names its inputs, so the closure is enumerable and can be
  staged alongside; an import graph is not.
* **Cache invalidation falls out for free.** The loader records what it read,
  so the dependency closure is a by-product of loading rather than a module
  graph someone has to walk. `build_current_ir` compares those hashes.

**What it is not.** Loading a program is an ordinary build step, not an agent
dispatch: it runs the file, it does not launch anything. Delegation puts a
child's `object.py` on disk; this is how the parent then builds and edits it.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from ...sdk._errors import BuildError
from ..tool_io import confine_to_run_dir
from ..cad import sandbox as _sandbox

#: `{run-relative path: content hash}` for every file this build loaded.
#: Drained into the build result, where the parent turns it into a cache key.
_deps_lock = threading.Lock()
_source_deps: dict = {}
#: Paths currently being loaded, innermost last. A loaded program may itself
#: load one, so a cycle is reachable and would otherwise recurse until the
#: interpreter's stack gave out — a `RecursionError` naming none of the files
#: involved.
_loading: list = []


def reset_source_deps() -> None:
    """Clear the ledger at the start of a build."""
    with _deps_lock:
        _source_deps.clear()
        _loading.clear()


def take_source_deps() -> dict:
    """Drain and return this build's `{path: hash}` dependency closure."""
    with _deps_lock:
        out = dict(_source_deps)
        _source_deps.clear()
        _loading.clear()
    return out


def source_digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _fail(message: str, path, reason: str, repairs) -> BuildError:
    return BuildError(
        message,
        refs=[],
        evidence={"reason": reason, "path": str(path)},
        suggested_repairs=list(repairs),
    )


def make_load_object(ctx, overlay: dict):
    """Build the `load_object` bound into one scene build's namespace.

    `overlay` is the very dict being assembled by the caller, so the loaded
    program executes against exactly the namespace `scene.py` has — including
    `load_object` itself, which is what lets a loaded program load another.
    Reusing the dict rather than calling the overlay factory again also matters
    for correctness: that factory installs a fresh `MeshAssetSession`, and a
    second one mid-build would strand the parts already staged.
    """
    run_dir = Path(getattr(ctx, "run_dir", ".") or ".")

    def load_object(path: str):
        """Build another file's object program; return its finalized ObjectIR.

        `path` is relative to the run directory. The file is an ordinary
        `object.py`: it constructs one Object and ends by assigning `ir`.
        """
        abs_path = confine_to_run_dir(run_dir, path)
        if abs_path is None:
            raise _fail(
                f"load_object({path!r}): path must be inside the run directory",
                path, "path_outside_run_dir",
                ["pass a path relative to the run dir, e.g. "
                 "'objects/fridge/object.py'"])
        if not abs_path.is_file():
            raise _fail(f"load_object({path!r}): no such file", path,
                        "missing_source",
                        ["check the path; a delegated object is written under "
                         "its own directory in this run"])
        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError as e:
            raise _fail(f"load_object({path!r}): could not read it ({e})",
                        path, "unreadable_source",
                        ["check the file's permissions"]) from None

        # Recorded before the exec, so a program that fails to build is still a
        # dependency: the author will edit it and expect that edit to
        # invalidate the cached failure.
        with _deps_lock:
            _source_deps[path] = source_digest(text)
            if path in _loading:
                cycle = " -> ".join(_loading + [path])
                _loading.clear()
                raise _fail(
                    f"load_object({path!r}): circular load ({cycle})",
                    path, "circular_load",
                    ["a program may not load itself, directly or through "
                     "another program it loads"])
            _loading.append(path)

        # The staged `source` asset must be the program that built the object,
        # not the program that loaded it. `compile.build_ir_with_layer` put
        # `scene.py`'s text on the session before this build started, and
        # `Object._source_text_for_staging` reads it at `finalize()` — which
        # for this object happens inside the exec below. Swapping it here is
        # what makes a loaded object's provenance point at its own file;
        # without it every object in a scene build cites `scene.py`.
        from ...sdk import _mesh_assets
        session = _mesh_assets.get_active_mesh_session()
        previous = getattr(session, "source_text", None) if session else None
        previously_owned = getattr(session, "source_owns_object", True) \
            if session else True
        try:
            _sandbox.check_ast(text, mode="exec")
            g = _sandbox.build_globals(overlay=overlay)
            if session is not None:
                session.source_text = text
                # This file IS one object's own program, unlike the `scene.py`
                # that loaded it, so the object it builds records it.
                session.source_owns_object = True
            # Compiled under its own filename so a traceback names the file the
            # author has to edit, rather than pointing at the `load_object`
            # call in scene.py.
            exec(compile(text, path, "exec"), g)
        finally:
            if session is not None:
                session.source_text = previous
                session.source_owns_object = previously_owned
            with _deps_lock:
                if path in _loading:
                    _loading.remove(path)

        ir = g.get("ir")
        if ir is None:
            raise _fail(
                f"load_object({path!r}): the program defined no `ir`",
                path, "no_ir_assigned",
                ["an object program ends with `ir = obj.finalize()`"])
        if not isinstance(ir, dict) or "parts" not in ir or "id" not in ir:
            raise _fail(
                f"load_object({path!r}): `ir` is not a finalized ObjectIR "
                f"(got {type(ir).__name__})",
                path, "bad_ir",
                ["assign the value `obj.finalize()` returned, unmodified"])
        return ir

    return load_object


def make_load_asset(ctx, overlay: dict):
    """Build the `load_asset` bound into one scene build's namespace.

    Sibling of `load_object`, and deliberately shaped like it: both answer
    "give me a definition to place", and the author should not have to think
    about which mechanism produced the geometry until they want to edit it —
    at which point the difference is the whole point. `load_object` returns an
    object this run built and whose source is the author's to change;
    `load_asset` returns one that was supplied frozen, and there is nothing in
    this run to edit.

    The staging, verification, and manifest live in
    `agent/assets/preload_asset.py`; this is only the build-time door.

    A name that was not preloaded is looked up in the asset library
    (`asset_library.py`, offered to the agent as `references/assets/INDEX.md`)
    and staged on the spot into this build's source tree, so the agent chooses
    its own characters and robots. In `animation` that tree is the
    per-build copy of `design/`, so a retrieved asset is staged again on each
    build; it is content-addressed and verified each time, so every build
    resolves the same frozen definition.
    """
    from ...sdk.object_builder import register_external_object
    from ..assets.preload_asset import (PRELOAD_DIR, PreloadAssetError, load_preloaded_manifest,
                                stage_library_asset)
    from ..assets.asset_library import AssetLibraryError, resolve_library_asset

    run_dir = Path(getattr(ctx, "run_dir", ".") or ".")

    def load_asset(name: str):
        """Return the frozen ObjectIR of a library or preloaded asset.

        `name` is a bundle name from the asset library's INDEX.md (staged into
        this run the first time it is loaded), or the name given to
        `--preload-asset <name>=<bundle>`. The result is a finalized ObjectIR:
        pass it straight to `scene.place(...)`, which accepts a finalized dict.
        """
        index_hint = ("search the asset library's INDEX.md (references/assets/ "
                      "in your run directory) for a bundle name")
        if not isinstance(name, str) or not name:
            raise _fail(
                f"load_asset({name!r}): name must be a non-empty string",
                name, "bad_asset_name", [index_hint])
        manifest = load_preloaded_manifest(run_dir)
        if name not in manifest:
            try:
                bundle = resolve_library_asset(name)
                stage_library_asset(name, bundle, run_dir)
            except AssetLibraryError as e:
                raise _fail(f"load_asset({name!r}): {e}", name, "unknown_asset",
                            [index_hint] + ([f"preloaded in this run: {sorted(manifest)}"]
                                            if manifest else [])) from None
            except PreloadAssetError as e:
                raise _fail(f"load_asset({name!r}): the library bundle failed its "
                            f"checks: {e}", name, "bad_library_asset",
                            ["choose another asset; this one cannot be trusted as staged"]) from None
            manifest = load_preloaded_manifest(run_dir)

        rel = manifest[name].get("ir_path") or f"{PRELOAD_DIR}/{name}.json"
        abs_path = confine_to_run_dir(run_dir, rel)
        if abs_path is None or not abs_path.is_file():
            raise _fail(
                f"load_asset({name!r}): staged IR missing at {rel}",
                rel, "missing_asset_ir",
                ["re-launch the run; the launcher stages this file before the "
                 "daemon starts"])
        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError as e:
            raise _fail(f"load_asset({name!r}): could not read {rel} ({e})",
                        rel, "unreadable_asset_ir",
                        ["check the file's permissions"]) from None

        # Recorded like a loaded program's source, so the build cache keys on
        # it. A preloaded IR does not change under a running agent today, but
        # the closure `fixed_source` replays and the cache `build_current_ir`
        # compares should name every input, not only the editable ones.
        with _deps_lock:
            _source_deps[rel] = source_digest(text)

        try:
            ir = json.loads(text)
        except json.JSONDecodeError as e:
            raise _fail(f"load_asset({name!r}): {rel} is not valid JSON ({e})",
                        rel, "bad_asset_json",
                        ["re-stage the bundle"]) from None
        if not isinstance(ir, dict) or "parts" not in ir or "id" not in ir:
            raise _fail(
                f"load_asset({name!r}): {rel} is not a finalized ObjectIR",
                rel, "bad_asset_ir",
                ["re-stage the bundle from a directory holding a valid "
                 "ir.json"])

        # Enter it in this execution's catalog. Without this the scene could
        # place it — `Scene.place` takes a finalized dict — and the definition
        # would then be absent from `object_catalog`, so the committed catalog
        # would name an `object_ref` it does not contain.
        # `register_external_object` recomputes the content hash rather than
        # trusting the stated one. The staging pass already checked the file
        # (`preload_asset._read_bundle`), but this is the boundary where a dict
        # assembled outside the SDK enters trusted state, and the object handed
        # over here is not necessarily the file that was checked then.
        register_external_object(ir)
        return ir

    return load_asset
