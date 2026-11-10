"""Stage externally-supplied geometry into a scene or animation run as frozen assets.

`--preload-asset <name>=<bundle>` hands a scene-layer `scene.py` geometry it may
**place** but must not **build**: a robot imported from a third-party URDF, a
character from glTF, or an object a previous run committed. The layer's premise
is that it authors every object it places, and that premise is exactly right for
the objects a brief asks it to design. It is wrong for an asset that already
exists and whose whole value is being *the* asset — a Unitree G1 the reader is
meant to recognize is not improved by an agent's impression of one.

**Why a name in the build namespace.** A scene or animation run's contract is the
build — a definition exists because `scene.py` constructed it — so the thing to
add here is a way to *obtain* a definition that the program did not construct.
That is a name in the build namespace, which is the same mechanism
`load_object` already is, and the two read as siblings:

    robot = load_asset("g1")                  # frozen; supplied to this run
    chair = load_object("objects/chair.py")   # built; this run's own source

**Where the pieces land, and why.**

* The referenced assets are copied to ``<run_dir>/assets/`` at **exactly** the
  relative paths the IR names. That is what keeps the IR byte-identical, and
  therefore keeps its ``id`` — a content hash — true. It also means no new
  resolution path: `SceneCatalog.asset_base_dir` is the run dir already, so
  the renderer, the narrow phase and the exact adjudicator find these meshes
  the same way they find a mesh the build staged. Both namespaces are
  content-addressed under blake2b, so a preloaded mesh and an authored one
  cannot collide on a name without being the same bytes.
* The IR is written to ``<run_dir>/preloaded/<name>.json``, **not** into
  ``catalog/``. The catalog directory is an output: `refresh_snapshot` rewrites
  it from the build on every compile and unlinks any `*.json` the build did not
  produce, so an input parked there would survive exactly until the first
  compile and then vanish.
* ``preloaded/manifest.json`` describes what was staged — name, hash, size,
  root, parts, joints, and their ranges — so the agent can read what it has
  before its first build, rather than discovering the shape of a 38-part robot
  by placing it and probing.

**What is checked here, and why here.** The stated ``id`` is recomputed from
the content, the same anti-tamper rule `scene_context.validate_catalog` applies
to a catalog, and every asset the IR cites must exist in the bundle. It runs in
the launcher, before the daemon starts, so a bad bundle fails with a message to
a human rather than at the agent's first compile — or worse, passes and gives
the scene geometry whose identity is a lie.

Nothing downstream learns from *this module* that the geometry was supplied
rather than authored. `verify/_per_object.py` does excuse an imported
definition from one object check, but it reads that off the definition's own
parts (`_is_imported`), not off this manifest — an asset is exempt because of
what it is, not because of how it got here.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from ...sdk._object_kernel import global_bbox
from ...sdk.ir import OBJECT_SCHEMA_VERSION, content_hash
from ..layers.scene_context import referenced_object_assets

#: Where a staged asset's frozen IR lives, relative to the run dir. Outside
#: `catalog/` because that directory is rebuilt from each compile.
PRELOAD_DIR = "preloaded"
#: Index of everything staged, for the agent and for the verifier exemption.
PRELOAD_MANIFEST = "manifest.json"
#: Schema tag of an anatomical overlay document.
ANATOMICAL_SCHEMA = "dycraft.anatomical_limits.v1"


#: Optional per-asset narrowing of the hardware joint ranges, staged beside the
#: IR. See `_load_anatomical_limits`.
ANATOMICAL_FILE = "anatomical_limits.json"
#: Shipped overlays, keyed by the ObjectIR's `name`. Tracked in the repo because
#: an overlay is our editorial judgment about how to *use* an asset, not a fact
#: the vendor shipped with it — and because the bundles themselves are tens of
#: megabytes that travel by rsync rather than by git.
ANATOMICAL_LIBRARY = Path(__file__).resolve().parents[2] / "catalogs" / "anatomical"


def _anatomical_source(bundle: Path, ir: dict) -> "Path | None":
    """Where this asset's overlay comes from, bundle first.

    A bundle may carry its own `anatomical_limits.json`, which wins — that is
    how a caller overrides the shipped judgment for one asset, or supplies one
    for an asset the library has never seen. Otherwise the library is consulted
    by the object's name, which is what makes a G1 arrive with its limits no
    matter whose copy of the bundle was passed.
    """
    local = bundle / ANATOMICAL_FILE
    if local.is_file():
        return local
    shipped = ANATOMICAL_LIBRARY / f"{ir.get('name')}.json"
    return shipped if shipped.is_file() else None


def _load_anatomical_limits(path: Path, ir: dict, name: str) -> dict:
    """Read and check an overlay against the IR it narrows.

    **The one invariant is that an overlay may only tighten.** A range wider
    than the hardware's is not an anatomical constraint — it is a claim about
    the machine that the machine does not support, and honoring it would put a
    pose in the scene that the real robot cannot reach. Refused here, in the
    launcher, rather than discovered at the first compile.

    Units are the surface convention the rest of the authoring surface uses:
    `limits_deg` for a revolute joint, `limits_m` for a prismatic one. Naming
    the unit in the key rather than inferring it means a file that disagrees
    with the mechanism is a loud error instead of a silent factor of 57.
    """
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PreloadAssetError(
            f"--preload-asset {name}: {path} unreadable: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema") != ANATOMICAL_SCHEMA:
        raise PreloadAssetError(
            f"--preload-asset {name}: {path} is not a "
            f"{ANATOMICAL_SCHEMA!r} document")
    rows = doc.get("limits")
    if not isinstance(rows, dict) or not rows:
        raise PreloadAssetError(
            f"--preload-asset {name}: {path} declares no limits")
    joints = {j["name"]: j for j in ir.get("joints") or ()}
    out = {}
    for jname, row in rows.items():
        j = joints.get(jname)
        if j is None:
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} narrows {jname!r}, which is "
                f"not a joint of {ir.get('name')!r}")
        if j["type"] not in ("revolute", "prismatic"):
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} narrows {jname!r}, which is "
                f"a {j['type']!r} joint and has no range")
        key = "limits_deg" if j["type"] == "revolute" else "limits_m"
        if not isinstance(row, dict) or key not in row:
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} entry for {jname!r} needs "
                f"{key!r} (it is a {j['type']} joint)")
        try:
            lo, hi = (float(v) for v in row[key])
        except (TypeError, ValueError):
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} entry for {jname!r} has a "
                f"malformed {key}: {row[key]!r}") from None
        if lo >= hi:
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} entry for {jname!r} is empty "
                f"({lo} >= {hi})")
        # Compare in the IR's own units so the tolerance means the same thing
        # for both joint types.
        i_lo, i_hi = (math.radians(lo), math.radians(hi)) \
            if j["type"] == "revolute" else (lo, hi)
        h_lo, h_hi = j["limits"]
        if i_lo < h_lo - 1e-9 or i_hi > h_hi + 1e-9:
            unit = "deg" if j["type"] == "revolute" else "m"
            h = ([round(math.degrees(v), 3) for v in (h_lo, h_hi)]
                 if j["type"] == "revolute" else [h_lo, h_hi])
            raise PreloadAssetError(
                f"--preload-asset {name}: {path} widens {jname!r} to "
                f"[{lo}, {hi}] {unit} beyond the asset's hardware range "
                f"{h} {unit}. An anatomical overlay may only tighten — the "
                f"hardware limits are what the machine can actually do.")
        out[jname] = {"limits": [i_lo, i_hi], "surface": [lo, hi],
                      "unit": "deg" if j["type"] == "revolute" else "m",
                      "guards": row.get("guards")}
    return {"source": str(path), "object": doc.get("object"),
            "rule": doc.get("rule"), "joints": out}


class PreloadAssetError(Exception):
    """A `--preload-asset` spec or bundle this launcher refuses."""


def parse_spec(spec: str) -> tuple[str, Path]:
    """`"g1=assets/imported/g1_with_dex1"` -> `("g1", Path(...))`."""
    name, sep, path = spec.partition("=")
    name, path = name.strip(), path.strip()
    if not sep or not name or not path:
        raise PreloadAssetError(
            f"--preload-asset {spec!r} must be NAME=BUNDLE, e.g. "
            "g1=assets/imported/g1_with_dex1")
    # The name becomes a file stem and the argument to `load_asset`, so it may
    # not be a path at all.
    if name != Path(name).name or name in (".", "..") or "/" in name:
        raise PreloadAssetError(
            f"--preload-asset name {name!r} is not a plain name")
    # A run folder laid out for a reader holds its bundle in `src/`.
    from ...run_layout import internal_dir
    return name, internal_dir(Path(path))


def _load_and_verify(bundle: Path, name: str) -> dict:
    """Read the bundle's ObjectIR and refuse anything a build could not trust."""
    ir_path = bundle / "ir.json"
    if not ir_path.is_file():
        raise PreloadAssetError(
            f"--preload-asset {name}: no ir.json in {bundle} — a bundle is a "
            "directory holding ir.json plus the assets/ its parts cite")
    try:
        ir = json.loads(ir_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PreloadAssetError(
            f"--preload-asset {name}: {ir_path} unreadable: {e}") from None
    if not isinstance(ir, dict):
        raise PreloadAssetError(
            f"--preload-asset {name}: {ir_path} is not an object")
    if ir.get("schema") != OBJECT_SCHEMA_VERSION:
        raise PreloadAssetError(
            f"--preload-asset {name}: schema tag is {ir.get('schema')!r}, "
            f"expected {OBJECT_SCHEMA_VERSION!r}")
    stated = ir.get("id")
    recomputed = content_hash({k: v for k, v in ir.items() if k != "id"})
    if stated != recomputed:
        raise PreloadAssetError(
            f"--preload-asset {name}: content hash mismatch — ir.json states "
            f"id={stated!r} but its content hashes to {recomputed!r}. The IR "
            "was edited after it was frozen; re-run the converter rather than "
            "patching the file.")
    if not ir.get("parts"):
        raise PreloadAssetError(f"--preload-asset {name}: IR has no parts")
    _verify_native_rig(bundle, ir, name)
    return ir


def _verify_native_rig(bundle: Path, ir: dict, name: str) -> None:
    """The rig blend an imported character names must be the bytes it names.

    The IR's hash covers the block, digest included, but not the file; without
    this, swapping `character.blend` under an unchanged IR would animate a
    different body than the one the scene was built around.
    """
    rig = ir.get("native_rig")
    if rig is None:
        return
    from .import_urdf import digest_bytes
    part_names = {p.get("name") for p in ir.get("parts") or ()}
    if rig.get("part") not in part_names:
        raise PreloadAssetError(
            f"--preload-asset {name}: native_rig names part "
            f"{rig.get('part')!r}, which the IR does not have")
    path = bundle / str(rig.get("blend") or "")
    if not rig.get("blend") or not path.is_file():
        raise PreloadAssetError(
            f"--preload-asset {name}: native_rig blend {rig.get('blend')!r} "
            f"missing from {bundle}")
    actual = digest_bytes(path.read_bytes())
    if actual != rig.get("digest"):
        raise PreloadAssetError(
            f"--preload-asset {name}: {rig['blend']} hashes to {actual!r}, "
            f"but the IR states {rig.get('digest')!r}; re-run import_gltf "
            "rather than replacing the blend")


def _joint_summary(ir: dict) -> list:
    """Every joint's name, type, parent/child, and range, in surface units.

    The manifest carries this because the alternative is the agent reading a
    38-joint IR to find out whether the thing it was handed can bend at the
    waist. Angles are converted to degrees to match the units every authoring
    surface uses; the IR stores radians.
    """
    import math
    out = []
    for j in ir.get("joints") or ():
        row = {"name": j.get("name"), "type": j.get("type"),
               "parent": j.get("parent"), "child": j.get("child")}
        limits = j.get("limits")
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            if j.get("type") == "revolute":
                row["limits_deg"] = [round(math.degrees(float(v)), 3)
                                     for v in limits]
            else:
                row["limits_m"] = [float(v) for v in limits]
        out.append(row)
    return out


def _describe(ir: dict) -> dict:
    """The manifest row for one staged asset: what it is, without its geometry."""
    (xmn, ymn, zmn), (xmx, ymx, zmx) = global_bbox(ir)
    return {
        "object_name": ir.get("name"),
        "ir_hash": ir["id"],
        "root": ir.get("root"),
        "size": [round(max(xmx - xmn, 1e-4), 6),
                 round(max(ymx - ymn, 1e-4), 6),
                 round(max(zmx - zmn, 1e-4), 6)],
        "n_parts": len(ir.get("parts") or ()),
        "n_joints": len(ir.get("joints") or ()),
        "parts": [p.get("name") for p in (ir.get("parts") or ())],
        "joints": _joint_summary(ir),
        **({"native_rig": _rig_summary(ir["native_rig"])}
           if ir.get("native_rig") else {}),
    }


def _rig_summary(rig: dict) -> dict:
    """What an author needs to animate an imported character's armature."""
    part = rig.get("part")
    return {
        "blender_objects": {
            "mesh": f"part/<instance>/{part}",
            "armature": f"rig/<instance>/{part}"},
        "root_bones": rig.get("root_bones") or [],
        "bones": [{k: b.get(k) for k in ("name", "parent", "head", "tail")}
                  for b in rig.get("bones") or ()],
        "actions": rig.get("actions") or [],
        "shape_keys": rig.get("shape_keys") or [],
        "frame": "bone heads/tails are in the asset's own frame at its bind "
                 "pose: meters, z-up, feet on z=0, facing +Y",
        **({"anatomy": _anatomy_summary(rig["anatomy"])} if rig.get("anatomy") else {}),
    }


def _anatomy_summary(anatomy: dict) -> dict:
    """What `AnatomicalJointLimits` checks on this rig, its ranges, and how the rig bends.

    `bend_axes` gives, for each elbow, knee, and finger joint, the axis in the child
    bone's own frame that flexes it when turned positively. It differs by rig family
    -- a Mixamo finger curls about +X, a 3ds Max Biped's about -Z, whose +X swings it
    sideways -- so curling every rig's fingers about +X splays a Biped's.
    `hands` gives each palm's normal.
    """
    joints = [{"joint": j["name"], "bones": [j["parent"], j["child"]],
               **({"flexion_deg": j["flexion_deg"], "off_plane_deg": j["off_plane_deg"],
                   "rest_flexion_deg": j["rest_flexion_deg"]} if j["kind"] == "hinge"
                  else {"max_deviation_deg": j["max_deg"]})}
              for j in anatomy.get("joints") or ()]
    out = {"skeleton": anatomy.get("skeleton")}
    body = anatomy.get("body") or {}
    if body:
        from ...animation.blender.verify import BODY_LIMITS, ROLL_RANGE
        out["checked"] = sorted(body)
        out["limits_deg"] = BODY_LIMITS
        out["note"] = (
            "AnatomicalJointLimits (advisory) reports frames where a joint leaves these ranges, "
            "measured the same way on every rig whatever the roll of its bones: elbow and knee "
            "bend, and bending backwards (nearly straight, or bent against the way the joint "
            "folds); shoulder and hip roll, whose "
            "range grows with how far the limb is raised (counted while the elbow or knee is "
            f"bent {ROLL_RANGE[0]:.0f}-{ROLL_RANGE[1]:.0f} degrees); forearm rotation from the "
            "neutral (thumb up, palm toward the body, + palm up); the wrist's flexion (+ toward "
            "the palm) and tilt (+ toward the thumb) within an ellipse; ankle flexion (+ toes up); "
            "the head's turn, tilt, and nod. Limits are a proposal, not certified.")
        if anatomy.get("hands"):
            out["hands"] = anatomy["hands"]
        if anatomy.get("bend_axes"):
            out["bend_axes"] = anatomy["bend_axes"]
    elif joints:
        out["checked_joints"] = joints
        out["note"] = ("AnatomicalJointLimits reports frames where these are exceeded: a hinge's "
                       "flexion is measured from the straight limb toward the side it folds to "
                       "(negative = bending backwards); a cone's deviation is from the rest "
                       "direction relative to the parent bone.")
    else:
        out["checked_joints"] = []
        out["note"] = "no anatomical limits declared: " + anatomy.get("reason", "")
    return out


def stage_preloaded_assets(specs, run_dir: Path, *, dest: Path = None) -> dict:
    """Stage every `NAME=BUNDLE` spec; return `{name: record}`.

    `dest` is the root the assets and `preloaded/` are written under, and
    defaults to `run_dir`. The two differ for `--layer animation`,
    whose source tree is `design/`: that layer copies `design/` wholesale into
    `design_state/source/` on every build and resolves everything — loaded
    programs, `load_asset`, and every mesh path — relative to that copy. Staging
    there rather than at the run root is what makes an asset resolve under both
    the build and the Blender import with no second resolution path, and what
    puts it inside the `tree_hash` the build is keyed on.

    Persisted to `<dest>/preloaded/manifest.json`, which is what `load_asset`
    resolves against and what the verifiers read to learn which definitions are
    supplied rather than authored.

    Raises `PreloadAssetError` on anything wrong — the caller turns it into a
    `parser.error`, so the run never starts against a bundle it cannot trust.
    """
    run_dir = Path(dest) if dest is not None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records: dict = {}
    for spec in specs:
        name, bundle = parse_spec(spec)
        if name in records:
            raise PreloadAssetError(
                f"--preload-asset {name}: named twice; each asset name may be "
                "staged once")
        records[name] = _stage_one(name, bundle, run_dir)
    if records:
        _write_manifest(run_dir, records)
    return records


def stage_library_asset(name: str, bundle: Path, dest: Path) -> dict:
    """Stage one library bundle under `dest` on demand and add it to the manifest.

    `load_asset(name)` calls this the first time a build asks for a library
    asset that was not preloaded. Same checks and same layout as
    `--preload-asset`: the IR re-hashed, the rig blend's digest checked, only
    the files the IR cites copied, the anatomical overlay staged beside it. The
    record is merged into whatever the manifest already holds, so a preloaded
    asset and a retrieved one sit side by side and read the same way.
    """
    dest = Path(dest)
    record = _stage_one(name, Path(bundle), dest)
    record["retrieved"] = True
    records = load_preloaded_manifest(dest)
    records[name] = record
    _write_manifest(dest, records)
    return record


def _write_manifest(run_dir: Path, records: dict) -> None:
    preload_dir = Path(run_dir) / PRELOAD_DIR
    preload_dir.mkdir(parents=True, exist_ok=True)
    (preload_dir / PRELOAD_MANIFEST).write_text(
        json.dumps({"schema": "dycraft.preloaded_assets.v1",
                    "assets": records}, indent=2, sort_keys=True))


def _stage_one(name: str, bundle: Path, run_dir: Path) -> dict:
    """Verify one bundle and copy what its IR cites under `run_dir`; its record."""
    preload_dir = run_dir / PRELOAD_DIR
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise PreloadAssetError(
            f"--preload-asset {name}: {bundle} is not a directory")
    ir = _load_and_verify(bundle, name)

    # Copy exactly what the IR cites, at the path it cites, so the IR stays
    # hash-true and resolves against the run dir like any staged mesh.
    try:
        wanted = referenced_object_assets(ir)
    except ValueError as e:
        raise PreloadAssetError(
            f"--preload-asset {name}: {e}") from None
    staged: list = []
    for rel in wanted:
        src = (bundle / rel).resolve()
        try:
            src.relative_to(bundle)
        except ValueError:
            raise PreloadAssetError(
                f"--preload-asset {name}: asset {rel.as_posix()!r} "
                f"resolves outside {bundle}") from None
        if not src.is_file():
            raise PreloadAssetError(
                f"--preload-asset {name}: IR references missing asset "
                f"{rel.as_posix()!r} under {bundle}")
        dst = run_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Content-addressed on both sides, so an existing file at this path
        # is the same bytes. Left alone rather than recopied.
        if not dst.exists():
            shutil.copy2(src, dst)
        staged.append(rel.as_posix())

    preload_dir.mkdir(parents=True, exist_ok=True)
    ir_rel = f"{PRELOAD_DIR}/{name}.json"
    # `sort_keys` is NOT set: the bytes are irrelevant to the hash (the IR
    # is re-hashed from its parsed content, not its file text) but writing
    # it as given keeps a diff against the source bundle readable.
    (run_dir / ir_rel).write_text(json.dumps(ir, indent=2))

    record = _describe(ir)
    record.update({"name": name, "source": str(bundle),
                   "ir_path": ir_rel, "assets": staged})

    # The overlay, if this asset has one. Staged beside the IR rather than
    # merged into it: narrowing a range would change the IR's content hash
    # and therefore its identity, and the frozen asset must stay the thing
    # the bundle says it is.
    src = _anatomical_source(bundle, ir)
    if src is not None:
        overlay = _load_anatomical_limits(src, ir, name)
        rel = f"{PRELOAD_DIR}/{name}.{ANATOMICAL_FILE}"
        (run_dir / rel).write_text(json.dumps(overlay, indent=2))
        record["anatomical_limits"] = {
            "path": rel, "source": overlay["source"],
            "narrowed": {jn: {"limits": v["surface"], "unit": v["unit"]}
                         for jn, v in overlay["joints"].items()},
        }
    return record


def load_preloaded_manifest(run_dir: Path) -> dict:
    """`{name: record}` for this run, or `{}` when nothing was preloaded.

    Absent is the overwhelmingly common case — every scene or animation run that builds
    everything it places — so a missing or unreadable manifest reads as "no
    preloaded assets" rather than raising into a build.
    """
    path = Path(run_dir) / PRELOAD_DIR / PRELOAD_MANIFEST
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    assets = blob.get("assets") if isinstance(blob, dict) else None
    return assets if isinstance(assets, dict) else {}


def anatomical_limits(run_dir: Path) -> dict:
    """`{ir_hash: {joint: {...}}}` for every preloaded asset that has an overlay.

    Keyed by the definition's content hash rather than the `load_asset` name,
    because that is what a SceneIR instance carries: the verifier reads
    `instance["object_ref"]` and has no other handle on which asset it is
    looking at.
    """
    out: dict = {}
    for rec in load_preloaded_manifest(run_dir).values():
        block = (rec or {}).get("anatomical_limits") or {}
        path = Path(run_dir) / (block.get("path") or "")
        if not block.get("path") or not path.is_file():
            continue
        try:
            overlay = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("ir_hash") and overlay.get("joints"):
            out[rec["ir_hash"]] = overlay
    return out


__all__ = [
    "PRELOAD_DIR", "PRELOAD_MANIFEST", "PreloadAssetError",
    "parse_spec", "stage_preloaded_assets", "load_preloaded_manifest",
    "anatomical_limits",
    "ANATOMICAL_FILE", "ANATOMICAL_SCHEMA", "ANATOMICAL_LIBRARY",
]
