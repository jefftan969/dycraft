"""SceneAgentContext — per-run catalog state for the scene-layer agent.

The scene agent receives a `catalog: Mapping[str, dict]` of finalized
ObjectIRs (keyed by content hash). This module:

  * validates the catalog (schema tag, hash recomputation, key consistency,
    name uniqueness) and raises `InvalidCatalog` / `DuplicateObjectName`
    before any LLM call;
  * materializes one `ObjectHandle` per entry (cached for the run);
  * snapshots the *used* subset into `runs/<run_id>/catalog/` after each
    successful compile, plus a manifest mapping object names to hashes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping

from ...sdk import ObjectHandle
from ...sdk._material import (
    texture_maps_from_ir as _texture_maps_from_ir,
)
from ...sdk._errors import InvalidCatalog, DuplicateObjectName
from ...sdk._mesh_assets import SRC_SUBDIR
from ...sdk.ir import OBJECT_SCHEMA_VERSION, content_hash
from ..tool_io import atomic_write_text
from ._scene_source import source_digest


CATALOG_DIR = "catalog"
CATALOG_MANIFEST = "manifest.json"
#: Which sources and object revisions a bundle was built from.
PROVENANCE_FILE = "provenance.json"



# ---------------------------------------------------------------------------
# Catalog asset contract
# ---------------------------------------------------------------------------

def referenced_object_assets(ir: dict) -> tuple[Path, ...]:
    """Return the run-relative assets referenced by an ObjectIR.

    Object runs are self-contained: every file-backed primitive points below
    ``assets/``. Keeping this extraction next to catalog ingestion gives the
    CLI, snapshots, renderer, and exact adjudicator one shared definition of what
    has to travel with an ObjectIR.
    """
    out: set[Path] = set()

    def _collect(params: dict, keys: tuple, where: str) -> None:
        """Validate and record one holder's REQUIRED asset paths."""
        for key in keys:
            raw = params.get(key)
            if not isinstance(raw, str) or not raw:
                raise ValueError(
                    f"object {ir.get('name')!r} {where} "
                    f"is missing non-empty asset path {key!r}"
                )
            rel = Path(raw)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts \
                    or rel.parts[0] != "assets":
                raise ValueError(
                    f"unsafe object asset path {raw!r}: catalog assets must be "
                    "run-relative paths below assets/"
                )
            out.add(rel)

    # The authored program is object-level (one `object.py` builds every part),
    # so it is collected once from the top of the IR rather than from each
    # part's params. Absent for an object with no CAD-authored parts.
    if ir.get("source"):
        _collect(ir, ("source",), "object-level source")
    for part in ir.get("parts", ()):
        primitive = part.get("primitive")
        params = part.get("params") or {}
        # Per-tag asset key lists rather than one union: every key named here is
        # REQUIRED to be present and non-empty below, so a union would reject
        # every part of both tags.
        keys = ()
        if primitive == "solid":
            # No `source`: it lives on one object-level `ir["source"]` rather
            # than on every part's params, because one `object.py` builds every
            # part and N identical strings were a fossil of the retired
            # `author_solids` design. It reaches the catalog through
            # `ObjectHandle.source()` instead. Naming it here would fail
            # staging for every `solid` part.
            keys = ("brep", "mesh")
        elif primitive == "imported":
            # No `source`/`brep` (an imported asset has neither), and the
            # collision mesh is a second file that must travel too — it is what
            # the mesh backend collides, so a catalog staged without it would
            # silently fall back to the conservative SAT verdict.
            keys = ("mesh", "collision")
        _collect(params, keys, f"part {part.get('name')!r}")
        # Every image the material carries, base color included.
        # `texture_maps_from_ir` is the single definition of "what is a map", so
        # a kind added to the library travels without an edit here — see
        # `sdk/_material.py`. Optional, unlike the geometry keys above, and
        # outside the per-tag `keys` dispatch because any primitive can carry
        # one. It has to travel for the same reason a collision mesh does: the
        # renderer resolves the path against the staged catalog root, and an
        # object staged without it renders the fallback color with nothing
        # anywhere reporting that the texture was left behind.
        tex = (part.get("material") or {}).get("texture")
        for raw in [m.get("image")
                    for m in _texture_maps_from_ir(tex).values()]:
            if not raw:
                continue
            rel = Path(raw)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts \
                    or rel.parts[0] != "assets":
                raise ValueError(
                    f"unsafe object asset path {raw!r}: catalog assets must be "
                    "run-relative paths below assets/"
                )
            out.add(rel)
    # The frozen rig of an imported character (`import_gltf`). Object-level
    # rather than a part asset, because it replaces the body part wholesale in
    # the animation layer's Blender scene, but it has to travel for the same
    # reason a collision mesh does: a catalog staged without it places a statue.
    rig = ir.get("native_rig") or {}
    extra = [rig.get("blend")]
    for raw in extra:
        if not raw:
            continue
        rel = Path(raw)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts \
                or rel.parts[0] != "assets":
            raise ValueError(
                f"unsafe object asset path {raw!r}: catalog assets must be "
                "run-relative paths below assets/"
            )
        out.add(rel)
    return tuple(sorted(out, key=lambda p: p.as_posix()))


def _content_digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_object_assets(ir: dict, *, source_root: Path,
                        catalog_root: Path) -> tuple[Path, ...]:
    """Copy exactly the assets referenced by ``ir`` into a catalog root.

    Relative paths are preserved, so the ObjectIR remains byte/hash-identical.
    Existing content-identical files are reused (content-addressed Object
    assets commonly deduplicate across objects); a same-path content mismatch
    is rejected instead of silently replacing geometry.
    """
    source_root = Path(source_root).resolve()
    catalog_root = Path(catalog_root).resolve()
    copied: list[Path] = []
    for rel in referenced_object_assets(ir):
        src = (source_root / rel).resolve()
        try:
            src.relative_to(source_root)
        except ValueError:
            raise ValueError(
                f"object asset {rel.as_posix()!r} resolves outside "
                f"{source_root}"
            ) from None
        if not src.is_file():
            raise FileNotFoundError(
                f"object {ir.get('name')!r} references missing asset "
                f"{rel.as_posix()!r} under {source_root}"
            )

        dst = catalog_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not dst.is_file() or _content_digest(src) != _content_digest(dst):
                raise ValueError(
                    f"catalog asset collision at {rel.as_posix()!r}: source "
                    "and staged contents differ"
                )
        else:
            shutil.copy2(src, dst)
        copied.append(dst)
    return tuple(copied)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _catalog_ref(name=None) -> dict:
    return {"kind": "catalog", "name": name, "id": None}


def _object_ref(name, ir_hash) -> dict:
    return {"kind": "object", "name": name, "id": ir_hash}


def validate_catalog(catalog: Mapping[str, dict]) -> dict:
    """Validate `catalog` and return a name→hash mapping on success.

    Raises:
        InvalidCatalog — non-mapping catalog, non-dict entry, missing/bad
            schema tag, missing/mismatched `id`, missing `name`, or any IR
            whose recomputed content hash disagrees with its stated `id`.
        DuplicateObjectName — two entries share the same `name`.
    """
    if not isinstance(catalog, Mapping):
        raise InvalidCatalog(
            f"catalog must be a Mapping, got {type(catalog).__name__}",
            refs=[_catalog_ref()],
            evidence={"reason": "catalog_not_mapping",
                      "value": type(catalog).__name__},
            suggested_repairs=["pass a {object_ref_hash: ObjectIR_dict} mapping"],
        )

    name_to_hash: dict[str, str] = {}
    for key, ir in catalog.items():
        if not isinstance(key, str):
            raise InvalidCatalog(
                f"catalog key must be a string hash, got {type(key).__name__}",
                refs=[_catalog_ref()],
                evidence={"reason": "key_not_string",
                          "object_ref": repr(key), "value": type(key).__name__},
                suggested_repairs=[
                    "use the ObjectIR's content hash (str) as the catalog key",
                ],
            )
        if not isinstance(ir, dict):
            raise InvalidCatalog(
                f"catalog entry for {key!r} must be a dict, got "
                f"{type(ir).__name__}",
                refs=[_catalog_ref()],
                evidence={"reason": "entry_not_dict",
                          "object_ref": key, "value": type(ir).__name__},
                suggested_repairs=["pass each ObjectIR as a JSON-shaped dict"],
            )

        schema = ir.get("schema")
        if schema != OBJECT_SCHEMA_VERSION:
            raise InvalidCatalog(
                f"catalog entry {key!r} has wrong schema tag",
                refs=[_catalog_ref()],
                evidence={"reason": "wrong_schema_tag",
                          "object_ref": key, "value": schema},
                suggested_repairs=[
                    f"every catalog entry must have schema == "
                    f"{OBJECT_SCHEMA_VERSION!r}",
                ],
            )

        stated_id = ir.get("id")
        if not isinstance(stated_id, str):
            raise InvalidCatalog(
                f"catalog entry {key!r} missing or non-string 'id'",
                refs=[_catalog_ref()],
                evidence={"reason": "missing_id",
                          "object_ref": key, "value": repr(stated_id)},
                suggested_repairs=[
                    "each ObjectIR must carry its content-hash 'id'",
                ],
            )

        if stated_id != key:
            raise InvalidCatalog(
                f"catalog entry {key!r} has mismatched 'id' field {stated_id!r}",
                refs=[_catalog_ref()],
                evidence={"reason": "id_key_mismatch",
                          "object_ref": key, "value": stated_id},
                suggested_repairs=[
                    "the catalog key must equal the ObjectIR's 'id' field",
                ],
            )

        # Recompute the content hash to catch tampered IRs.
        without_id = {k: v for k, v in ir.items() if k != "id"}
        try:
            recomputed = content_hash(without_id)
        except (ValueError, TypeError) as e:
            raise InvalidCatalog(
                f"catalog entry {key!r} not JSON-canonicalizable: {e}",
                refs=[_catalog_ref()],
                evidence={"reason": "not_canonicalizable",
                          "object_ref": key, "value": str(e)},
                suggested_repairs=[
                    "ensure the ObjectIR is the dict returned by "
                    "Object.finalize() (canonical JSON, no NaN/Inf)",
                ],
            )
        if recomputed != stated_id:
            raise InvalidCatalog(
                f"catalog entry {key!r} content-hash mismatch",
                refs=[_catalog_ref()],
                evidence={"reason": "hash_mismatch",
                          "object_ref": key, "value": recomputed},
                suggested_repairs=[
                    "do not mutate ObjectIRs after Object.finalize() — the "
                    "hash is the IR's identity",
                ],
            )

        name = ir.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidCatalog(
                f"catalog entry {key!r} missing or non-string 'name'",
                refs=[_catalog_ref()],
                evidence={"reason": "missing_name",
                          "object_ref": key, "value": repr(name)},
                suggested_repairs=["every ObjectIR must carry a non-empty 'name'"],
            )

        if name in name_to_hash:
            raise DuplicateObjectName(
                f"two catalog entries share name {name!r}",
                refs=[
                    _catalog_ref(name),
                    _object_ref(name, name_to_hash[name]),
                    _object_ref(name, key),
                ],
                evidence={"name": name,
                          "hashes": [name_to_hash[name], key]},
                suggested_repairs=[
                    "rename one of the colliding objects upstream and "
                    "regenerate its ObjectIR",
                ],
            )
        name_to_hash[name] = key

    return name_to_hash


#: JSON files in a catalog dir that are NOT ObjectIRs, skipped even in the
#: no-manifest fallback branch. `import.json` is
#: `agent/assets/import_urdf.py`'s conversion recipe, which sits beside
#: the `ir.json` it produced; without this a converter bundle staged with no
#: `manifest.json` would have its recipe ingested as an ObjectIR.
IMPORT_RECIPE_FILE = "import.json"

_CATALOG_NON_IR_FILES = frozenset({
    CATALOG_MANIFEST,
    IMPORT_RECIPE_FILE,
})


def iter_catalog_ir_paths(catalog_dir: Path) -> list[Path]:
    """Enumerate the ObjectIR JSON files that belong to `catalog_dir`.

    When `<catalog_dir>/manifest.json` exists it is the authoritative
    allowlist: only the `<hash>.json` files it references are returned, in
    sorted order. Both manifest shapes this repo writes are accepted — the flat
    `{slot: hash}` and the bundle form whose allowlist is under `objects` — see
    the comment at the branch. Anything else under `catalog_dir` (the catalog
    manifest itself, debug dumps) is ignored — the loader does not assume every `*.json` is an IR.

    When no manifest is present the directory is treated as a raw,
    hand-curated catalog: every `*.json` match is returned (sorted),
    minus the named non-IR sidecars in `_CATALOG_NON_IR_FILES` so the
    same sidecar files are filtered out symmetrically.
    """
    manifest_path = catalog_dir / CATALOG_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError(
                f"{manifest_path}: expected {{name: hash, ...}}, "
                f"got {type(manifest).__name__}"
            )
        # Two manifest shapes are read: the flat `{name: hash}`, and the
        # bundle form `{"objects": {name: hash}, "placed": [...],
        # "asset_base": "."}` that bundles are written in, whose
        # allowlist is under `objects`.
        entries = manifest.get("objects")
        if not isinstance(entries, dict):
            entries = {k: v for k, v in manifest.items() if isinstance(v, str)}
        paths: list[Path] = []
        missing: list[str] = []
        for slot_name, ir_hash in entries.items():
            p = catalog_dir / f"{ir_hash}.json"
            if p.exists():
                paths.append(p)
            else:
                missing.append(f"{slot_name}→{ir_hash}.json")
        if missing:
            raise FileNotFoundError(
                f"{manifest_path}: missing IR file(s) for object(s): "
                f"{', '.join(missing)}"
            )
        return sorted(paths)
    return sorted(p for p in catalog_dir.glob("*.json")
                  if p.name not in _CATALOG_NON_IR_FILES)


__all__ = [
    "CATALOG_DIR",
    "CATALOG_MANIFEST",
    "validate_catalog",
    "iter_catalog_ir_paths",
    "referenced_object_assets",
    "stage_object_assets",
]


# ---------------------------------------------------------------------------
# Scene-layer catalog
# ---------------------------------------------------------------------------


class SceneCatalog:
    """Catalog state for a scene run, derived from the current build.

    A scene-layer `scene.py` authors its objects, so the catalog is an *output* of
    the very build whose verifiers, probes and renders need it. Nothing can
    hand it over at run start because it does not exist yet.

    **Why properties over an adopt step.** The obvious shape is a mutable
    payload plus a call that copies each build's catalog into it. That call has
    to be made on every path that produces a build — compile, probe, view,
    snapshot, and each of their cache-hit branches — and the failure mode when
    one is missed is not an error but a stale catalog: the verifier judges the
    previous revision's geometry and passes. Reading through to the build cache
    means there is no adoption to forget. The catalog is a function of the
    build, so it is written here as one.

    `asset_base_dir` is the run dir, not a staged catalog root. A scene build
    stages its parts through the run's own `MeshAssetSession`, so ObjectIR mesh
    and B-rep paths are already run-relative and resolve where they are — the
    there is no per-compile catalog staging. Assembling a *portable* bundle is a `finish` concern; see
    `stage_object_assets`.
    """

    def __init__(self, run_dir: Path, snapshot_dir: Path | None = None):
        self._run_dir = Path(run_dir)
        self._ctx = None
        self.snapshot_dir = Path(
            snapshot_dir if snapshot_dir is not None
            else self._run_dir / CATALOG_DIR)
        # Memo keyed by the identity of the catalog dict it was derived from,
        # so a rebuild that produced byte-identical objects reuses its handles
        # and a rebuild that did not cannot serve the old ones.
        self._derived_from = None
        self._handles: dict = {}
        self._name_to_hash: dict = {}

    def bind(self, ctx) -> None:
        """Attach the run's ToolContext.

        The payload is built by `server._build_layer`, before any ToolContext
        exists, so the two are joined here rather than at construction. Only a
        payload that reads live build state needs this; `Session` calls it on
        any payload that offers it.
        """
        self._ctx = ctx

    # ---- source of truth -------------------------------------------------

    def _build_result(self) -> dict:
        """This op's build, else the run's latest (`tools.pinned_build`)."""
        from .tools import pinned_build
        pinned = pinned_build(self._ctx) if self._ctx is not None else None
        if pinned is not None:
            return pinned
        cached = getattr(self._ctx, "_current_build_cache", None) or {}
        return cached.get("result") or {}

    @property
    def catalog(self) -> dict:
        result = self._build_result()
        if not result.get("ok"):
            return {}
        return result.get("object_catalog") or {}

    @property
    def asset_base_dir(self) -> Path:
        return Path(getattr(self._ctx, "run_dir", None) or self._run_dir)

    def _derive(self) -> None:
        catalog = self.catalog
        if catalog is self._derived_from:
            return
        # Names are advisory in a scene build (the catalog is hash-keyed), so
        # a collision is disambiguated rather than raised on: two definitions
        # legitimately share a name when an author parameterizes one factory
        # into a wide and a narrow cabinet. The hash suffix keeps every entry
        # addressable by *some* name for probes and `view`, and the bundle
        # manifest written at finish carries the same spelling.
        name_to_hash: dict = {}
        for h, ir in catalog.items():
            name = ir.get("name") or "object"
            if name in name_to_hash:
                name = f"{name}@{h[:8]}"
            name_to_hash[name] = h
        self._name_to_hash = name_to_hash
        self._handles = {n: ObjectHandle(catalog[h])
                         for n, h in name_to_hash.items()}
        self._derived_from = catalog

    @property
    def name_to_hash(self) -> dict:
        self._derive()
        return self._name_to_hash

    @property
    def object_handles(self) -> dict:
        self._derive()
        return self._handles

    # ---- fields the scene-layer plumbing reads ---------------------------

    @property
    def anatomical_limits(self) -> dict:
        """`{ir_hash: overlay}` for preloaded assets that narrow their ranges.

        Read per compile rather than cached at construction: the payload is
        built before the run dir is fully populated, and one small JSON read
        is cheaper than a staleness bug.
        `verify/scene_verify.py::anatomical_joint_limits` is the consumer.
        """
        from ..assets.preload_asset import anatomical_limits
        return anatomical_limits(self._run_dir)

    def refresh_snapshot(self, ir: dict, *, is_finish: bool = False) -> None:
        """Write the built catalog to `<run_dir>/catalog/` after a compile.

        Not needed to render or verify — `asset_base_dir` already resolves
        every path — but it is what makes the definitions this build produced
        readable as files: by the agent through `read_file`, and by anyone
        opening the run dir afterwards.

        **Assets are copied at finish and not before.** During iteration they
        are already resolvable where they were staged, under the run's own
        `assets/`, so copying every mesh and B-rep into `catalog/` on each
        compile would duplicate the run's geometry on every edit to buy
        nothing. At finish it becomes worth it: `catalog/` with its own
        `assets/` inside is the shape every downstream reader expects — the
        animation layer's `--scene`, and the structured handoff — and it is
        what makes the bundle survive being moved.
        """
        catalog = self.catalog
        if not catalog:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        used = {inst["object_ref"] for inst in (ir.get("instances") or ())}
        keep = {f"{h}.json" for h in catalog}
        for h, obj_ir in sorted(catalog.items()):
            atomic_write_text(self.snapshot_dir / f"{h}.json",
                              json.dumps(obj_ir, indent=2, sort_keys=True))
        manifest = {
            "objects": {n: h for n, h in sorted(self.name_to_hash.items())},
            # Which definitions the scene actually places, as opposed to which
            # ones the program built. An author inspecting a candidate before
            # placing it should not have that object disappear from the run.
            "placed": sorted(used),
            "asset_base": ".",
        }
        atomic_write_text(self.snapshot_dir / CATALOG_MANIFEST,
                          json.dumps(manifest, indent=2, sort_keys=True))
        for stale in self.snapshot_dir.glob("*.json"):
            if stale.name != CATALOG_MANIFEST and stale.name not in keep:
                stale.unlink()
        if is_finish:
            self._stage_bundle(catalog, used)

    def _stage_bundle(self, catalog: dict, used: set) -> None:
        """Make `catalog/` self-contained, and record what built it.

        Every asset an ObjectIR references is copied in with its relative path
        preserved, so the IR stays byte- and hash-identical and resolves
        against `catalog/` the way a staged scene catalog does. That includes
        each object's `source`, which is how the recipe travels: the bundle
        carries the programs that produced the geometry, not just the geometry.
        """
        for h in sorted(catalog):
            stage_object_assets(catalog[h], source_root=self.asset_base_dir,
                                catalog_root=self.snapshot_dir)

        result = self._build_result()
        entry = self._run_dir / "scene.py"
        provenance = {
            "schema": "dycraft.unified_provenance.v1",
            # The exact revisions consumed. A bundle that cannot say which
            # source produced it is a bundle whose geometry cannot be
            # regenerated or diffed against a later run.
            "entry_source": {
                "path": "scene.py",
                "digest": (source_digest(entry.read_text(encoding="utf-8"))
                           if entry.is_file() else None),
            },
            "loaded_sources": dict(sorted(
                (result.get("source_deps") or {}).items())),
            "objects": {n: h for n, h in sorted(self.name_to_hash.items())},
            "placed": sorted(used),
            "asset_base": ".",
        }
        # Which definitions this run did NOT author. Without it the bundle
        # claims `scene.py` and its loaded programs regenerate every object in
        # the catalog, which for a preloaded asset is false — no source in the
        # closure produces it, and a reader diffing the bundle against a
        # re-run would find geometry it could not account for.
        from ..assets.preload_asset import load_preloaded_manifest
        preloaded = {rec["ir_hash"]: {"name": name,
                                      "object_name": rec.get("object_name"),
                                      "source": rec.get("source")}
                     for name, rec in sorted(
                         load_preloaded_manifest(self._run_dir).items())
                     if rec.get("ir_hash") in catalog}
        if preloaded:
            provenance["preloaded"] = preloaded
        atomic_write_text(self.snapshot_dir / PROVENANCE_FILE,
                          json.dumps(provenance, indent=2, sort_keys=True))

        # The programs, copied in beside the geometry they produced.
        #
        # An object built by its own program references that program as an
        # asset, so `stage_object_assets` already brought it. An object built
        # inline in `scene.py` does not — it has no program of its own to name
        # (see `LayerConfig.entry_source_is_object_program`) — so the entry
        # source and everything it loaded are staged here explicitly. Without
        # this the bundle would carry geometry whose recipe existed only in a
        # run directory that may be gone.
        # `SRC_SUBDIR` is already "assets/src" — the path staging uses inside a run,
        # and the same one relative to a bundle root.
        src_dir = self.snapshot_dir / SRC_SUBDIR
        for rel, digest in [("scene.py", provenance["entry_source"]["digest"])] \
                + sorted(provenance["loaded_sources"].items()):
            if digest is None:
                continue
            origin = self._run_dir / rel
            if not origin.is_file():
                continue
            src_dir.mkdir(parents=True, exist_ok=True)
            # Content-addressed under the same digest `stage_source` uses, so a
            # loaded program staged by both paths lands on one file.
            dest = src_dir / f"{digest}.py"
            if not dest.exists():
                dest.write_bytes(origin.read_bytes())
