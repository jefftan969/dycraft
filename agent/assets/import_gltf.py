"""Import a skinned glTF 2.0 / GLB (or FBX, or a native .blend) character as a native Blender rig.

    python -m dycraft.agent.assets.import_gltf \\
        --in  assets/characters/Xbot.glb \\
        --out assets/imported/xbot
    python -m dycraft.agent.assets.import_gltf --in X.glb --inspect

Sibling of `import_urdf.py` and the same kind of artifact: a self-contained
bundle (`ir.json`, `import.json`, `assets/...`) that `--preload-asset
<name>=<bundle>` stages into a scene or animation run, where `scene.py` places it with
`load_asset("<name>")`. What differs is what the character *is* downstream.

**A character is one deforming part, not a mechanism.** A URDF robot's joints
are the thing being animated, so its IR carries them and the native layer
realizes them as the generated `link/`/`joint/` Empties with a `q` each. A
character's articulation is a skinning rig -- dozens of free-rotating bones and
one mesh bound across all of them -- and the tool built to author that is a
Blender armature. So the IR here has **one part and no joints**: the bind-shape
mesh, which is what the scene layer places, anchors to and collides at layout
time. The rig itself travels frozen beside it as `character.blend` (an Armature
object `rig` and one skinned mesh `body`), named by the object-level
`native_rig` block, and `worker.initialize` swaps it in for the part when the
animation project is built: the mesh becomes `part/<instance>/body` on its
link, the armature `rig/<instance>/body` beside it, and the Armature modifier
deforms the one into the other. Every native check reads evaluated geometry,
so collision, continuity and support see the deformed body with nothing
character-specific in them; the body's *object* frame never leaves its link,
which is the only thing the mechanism checks ask of it. This is deliberately
not a "direct rig binding": no bone stands in for an IR joint, because the
character has none.

**Blender does the reading.** `_import_gltf_blender.py` runs under the Blender
interpreter, imports the file with Blender's own glTF importer and normalizes
the result -- identity object transforms, feet on z=0, centered on the origin,
facing +Y, meters -- then freezes it. The importer already resolves what a
hand-written reader would have to be told: node-level axis conversions an
exporter baked into an ancestor (CesiumMan's `Z_UP`), centimeter armatures
(Mixamo's 0.01 object scale) and bind-pose recovery. Every glTF animation is
kept as an Action named `<object>/<clip>`, which is the asset's own gait for an
author to start from. The child verifies its own normalization by posing every
clip before and after it; see its docstring for the trap that check exists for.

**Facing is measured, never read off bone names.** CesiumMan labels its
figure's right side `_L_`. `--yaw auto` (the default) turns the figure by the
measured facing -- a foot reaches further forward than a heel reaches back --
and refuses when there is nothing to measure; `--inspect` prints it without
writing anything, and `--yaw DEG` overrides it. It is recorded either way.

**What the bundle does not carry.** Joint limits: glTF declares none, and a
bone has no `q` for `JointLimits` to read. The proxy part's appearance is a
flat color (the area-weighted mean of the materials, textures included); the
real materials, textures, and shape keys live in the blend and are what the
animation layer renders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:                                    # package import
    from .import_urdf import (COLLISION_SUBDIR, MESHES_SUBDIR,
                              PRIMITIVE_IMPORTED, convex_hull, digest_bytes,
                              mass_properties, watertight, weld, write_obj)
    from ... import _oom
    from ...render._blender_env import blender_child_env, blender_python
    from ...sdk.ir import OBJECT_SCHEMA_VERSION, insert_id
    from ...animation.blender.rig_anatomy import (  # noqa: F401  (re-exported)
        ANATOMY_POLICIES, ANATOMY_VERSION, FINGERS, HUMAN_JOINTS, LIMB_ROLES, SKELETON_FAMILIES,
        _role_name, skeleton_anatomy, skeleton_roles)
except ImportError:                     # standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dycraft.agent.assets.import_urdf import (
        COLLISION_SUBDIR, MESHES_SUBDIR, PRIMITIVE_IMPORTED, convex_hull,
        digest_bytes, mass_properties, watertight, weld, write_obj)
    from dycraft import _oom
    from dycraft.render._blender_env import blender_child_env, blender_python
    from dycraft.sdk.ir import OBJECT_SCHEMA_VERSION, insert_id
    from dycraft.animation.blender.rig_anatomy import (  # noqa: F401  (re-exported)
        ANATOMY_POLICIES, ANATOMY_VERSION, FINGERS, HUMAN_JOINTS, LIMB_ROLES, SKELETON_FAMILIES,
        _role_name, skeleton_anatomy, skeleton_roles)


CONVERTER = "dycraft.agent.assets.import_gltf"
#: 1 was the ObjectIR bone-chain converter (3 revolutes + linker cubes per
#: bone, LBS weights in `skins`); 2 the native rig; 3 adds `native_rig.anatomy`.
CONVERTER_VERSION = 4
#: Where the frozen rig blend goes, content-addressed like the meshes.
RIGS_SUBDIR = "assets/rigs"
#: Schema tag of the IR's `native_rig` block.
NATIVE_RIG_SCHEMA = "dycraft.native_rig.v1"
#: The one part every character bundle has.
BODY_PART = "body"
BLENDER_CHILD = Path(__file__).with_name("_import_gltf_blender.py")
BLENDER_TIMEOUT_S = 900


class CharacterImportError(RuntimeError):
    """The file cannot become a character bundle; the message says why."""


# ---------------------------------------------------------------------------
# The Blender child
# ---------------------------------------------------------------------------


def run_blender(src: Path, *, out: Path | None, name: str,
                yaw="auto", scale: float = 1.0, clips: str = "all", up: str = "+z",
                rest: str = "bind", reparent: dict | None = None) -> dict:
    """Run `_import_gltf_blender.py` on `src`; return its report.

    `out=None` inspects: the child reads and measures but writes no blend.
    """
    interp = blender_python()
    if interp is None:
        raise CharacterImportError(
            "no Blender interpreter configured (`python -m dycraft.install "
            "blender` records one in .blender_python); the character is read by "
            "Blender's own glTF importer")
    with tempfile.TemporaryDirectory(prefix="dycraft_gltf_") as tmp:
        job_path = Path(tmp) / "job.json"
        report_path = Path(tmp) / "report.json"
        job_path.write_text(json.dumps({
            "src": str(Path(src).resolve()),
            "out": str(Path(out).resolve()) if out is not None else None,
            "name": name, "yaw_deg": yaw, "scale": float(scale), "clips": clips, "up": up, "rest": rest,
            "reparent": dict(reparent or {}), "report": str(report_path)}))
        proc = subprocess.run(
            _oom.tiered([interp, str(BLENDER_CHILD), str(job_path)], _oom.WORKER),
            env=blender_child_env(), capture_output=True, text=True,
            timeout=BLENDER_TIMEOUT_S)
        if not report_path.is_file():
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise CharacterImportError(f"the Blender import child failed "
                               f"(exit {proc.returncode}):\n{tail}")
        report = json.loads(report_path.read_text())
    if not report.pop("ok", False):
        raise CharacterImportError(report.get("error") or "import refused")
    return report


def inspect(src, *, name: str | None = None, up: str = "+z", rest: str = "bind") -> dict:
    """What is in the file, measured, with nothing decided or written."""
    src = Path(src)
    return run_blender(src, out=None, name=name or sanitize_name(src.stem), up=up, rest=rest)


# ---------------------------------------------------------------------------
# Proxy geometry
# ---------------------------------------------------------------------------


def sanitize_name(raw: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in raw).strip("_")
    return (out or "character").lower()


def cap_open_loops(verts, tris):
    """Fan each boundary loop to its centroid, so an open skin bounds a volume.

    Only the *collision* mesh is capped. A character's skin is often open where
    a garment or visor was modeled as a separate sheet (Soldier: 232 boundary
    edges after welding), and a BVH over an open shell answers containment
    about a surface with no inside. Capping is the least geometry that fixes
    that; a convex hull would, but a hull of a figure with its arms out fills
    the space between them.

    A cap triangle is `(b, a, centroid)` for each boundary directed edge
    `(a, b)`, which supplies the missing opposite direction; the caller's
    `watertight` check confirms it. A boundary that revisits a vertex (a
    figure-eight) is split into simple cycles as the walk closes each one.
    Returns `(verts, tris, n_caps)`.
    """
    tris = np.asarray(tris)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    directed = set(map(tuple, e))
    succ: dict = {}
    for a, b in directed:
        if (b, a) not in directed:
            succ.setdefault(int(a), []).append(int(b))
    verts = np.asarray(verts, dtype=np.float64)
    if not succ:
        return verts, tris.astype(np.int64), 0
    loops = []
    while any(succ.values()):
        start = next(v for v, l in succ.items() if l)
        path, at, cur = [start], {start: 0}, start
        while succ.get(cur):
            nxt = succ[cur].pop()
            if nxt in at:
                i = at[nxt]
                loops.append(path[i:])
                for v in path[i + 1:]:
                    del at[v]
                del path[i + 1:]
            else:
                at[nxt] = len(path)
                path.append(nxt)
            cur = nxt
    new_verts, new_tris = [], []
    for loop in loops:
        if len(loop) < 3:
            continue
        c = len(verts) + len(new_verts)
        new_verts.append(verts[loop].mean(axis=0))
        for k, a in enumerate(loop):
            new_tris.append([loop[(k + 1) % len(loop)], a, c])
    if not new_verts:
        return verts, tris.astype(np.int64), 0
    return (np.concatenate([verts, np.asarray(new_verts)]),
            np.concatenate([tris, np.asarray(new_tris, dtype=np.int64)]),
            len(new_verts))


def bounds_volume(tris) -> bool:
    """True when every directed edge has exactly as many reverse twins.

    That is what a closed, consistently wound surface is, and it is weaker
    than `watertight`'s `closed and oriented` in the one case that matters
    here: a **self-touching** skin. Where two closed sheets meet along a seam
    (Michelle's neck: 20 edges carrying four faces each), each edge has two
    copies in each direction, so `oriented` ("every directed edge once")
    reads it as mis-wound while nothing is. The divergence theorem and a BVH
    inside test only need the balance, which this checks.
    """
    t = np.asarray(tris, dtype=np.int64)
    if not len(t):
        return False
    d = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    fwd, fc = np.unique(d, axis=0, return_counts=True)
    rev, rc = np.unique(d[:, ::-1], axis=0, return_counts=True)
    return (fwd.shape == rev.shape and bool(np.array_equal(fwd, rev))
            and bool(np.array_equal(fc, rc)))


def collision_mesh(verts, tris, policy: str = "auto"):
    """`(verts, tris, how)` for the proxy's collision mesh.

    `auto`: the welded skin if it bounds a volume (`bounds_volume`), else the
    skin with its open loops capped if that makes it bound one, else its
    convex hull -- which for a figure with its arms out fills the space
    between them, so it is the last resort. `mesh` and `hull` force the first
    and last.
    """
    wt = watertight(verts, tris)
    touching = " (self-touching)"
    if policy == "mesh" or (policy == "auto" and bounds_volume(tris)):
        strict = wt["closed"] and wt["oriented"]
        return verts, tris, "skin" + ("" if strict or policy == "mesh" else touching)
    if policy == "auto":
        cv, ct, n = cap_open_loops(verts, tris)
        if bounds_volume(ct):
            cw = watertight(cv, ct)
            strict = cw["closed"] and cw["oriented"]
            return cv, ct, (f"skin with {n} open loop(s) capped"
                            + ("" if strict else touching))
    hv, ht = convex_hull(verts)
    return hv, ht, "convex hull"


def _srgb_byte(linear: float) -> int:
    c = min(max(float(linear), 0.0), 1.0)
    c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return int(round(255 * c))


def _tri_areas(verts, tris):
    a, b, c = (verts[tris[:, k]] for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def proxy_material(rows: list, material_index, verts, tris) -> dict:
    """The proxy part's flat material: an area-weighted mean over the skin.

    Blender reports linear colors (a texture's pixels included); the IR's
    `color` is 8-bit sRGB, like every other material in a catalog.
    """
    if not rows:
        return {"color": [200, 200, 200], "metallic": 0.0, "roughness": 0.5,
                "alpha": 1.0}
    areas = _tri_areas(verts, tris)
    idx = np.clip(np.asarray(material_index, dtype=np.int64), 0, len(rows) - 1)
    w = np.bincount(idx, weights=areas, minlength=len(rows))
    w = w / w.sum() if w.sum() > 0 else np.full(len(rows), 1.0 / len(rows))

    def mean(key):
        return np.sum([w[i] * np.asarray(r[key], dtype=float)
                       for i, r in enumerate(rows)], axis=0)

    return {"color": [_srgb_byte(c) for c in mean("color")],
            "metallic": round(float(mean("metallic")), 4),
            "roughness": round(float(mean("roughness")), 4),
            "alpha": round(float(mean("alpha")), 4)}


def _stage(out_dir: Path, subdir: str, data: bytes, suffix: str) -> tuple:
    dig = digest_bytes(data)
    rel = f"{subdir}/{dig}{suffix}"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return rel, dig


# ---------------------------------------------------------------------------
# Anatomy: in `animation/blender/rig_anatomy.py`, which the sandboxed check
# worker loads too (see its docstring).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def emit_bundle(src, out_dir, *, name: str | None = None, yaw="auto",
                scale: float = 1.0, collision: str = "auto",
                clips: str = "all", anatomy: str = "auto", up: str = "+z",
                rest: str = "bind", reparent: dict | None = None) -> dict:
    """Import `src` and write the bundle to `out_dir`; return the IR.

    `reparent` ({bone: new parent}) hangs bones the rig leaves behind its body
    -- a hoof skinned under an IK controller -- from the bones they belong to,
    keeping each clip's motion (`attached_loose_bones` in `import.json`)."""
    src, out_dir = Path(src), Path(out_dir)
    name = name or sanitize_name(src.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dycraft_gltf_out_") as tmp:
        tmp = Path(tmp)
        rep = run_blender(src, out=tmp, name=name, yaw=yaw, scale=scale, clips=clips, up=up, rest=rest,
                          reparent=reparent)
        rest = np.load(tmp / "rest.npz")
        blend_bytes = (tmp / "character.blend").read_bytes()
    blend_rel, blend_dig = _stage(out_dir, RIGS_SUBDIR, blend_bytes, ".blend")

    raw_v, raw_t = rest["verts"].astype(np.float64), rest["tris"].astype(np.int64)
    verts, tris, keep = weld(raw_v, raw_t, return_keep=True)
    uvs = rest["uvs"][keep] if "uvs" in rest.files else None
    material_index = rest["material"][keep]
    wt = watertight(verts, tris)

    vis_rel, vis_dig = _stage(out_dir, MESHES_SUBDIR,
                              write_obj(verts, tris, BODY_PART, uvs), ".obj")
    cv, ct, how = collision_mesh(verts, tris, collision)
    col_rel, col_dig = _stage(out_dir, COLLISION_SUBDIR,
                              write_obj(cv, ct, f"{BODY_PART}:collision"), ".obj")
    # Mass properties of what bounds a volume: the collision mesh is closed by
    # construction, the skin need not be.
    mp = mass_properties(cv, ct)
    part = {
        "id": 0,
        "name": BODY_PART,
        "primitive": PRIMITIVE_IMPORTED,
        "params": {
            "mesh": vis_rel, "collision": col_rel,
            "digest": vis_dig, "collision_digest": col_dig,
            "aabb_min": [float(v) for v in verts.min(axis=0)],
            "aabb_max": [float(v) for v in verts.max(axis=0)],
            "shape": {"volume": mp["volume"], "center": mp["center"],
                      "inertia_diag": mp["inertia_diag"],
                      "n_vertices": int(len(verts)),
                      "n_triangles": int(len(tris)),
                      "closed": bool(wt["closed"] and wt["oriented"])},
        },
        "material": proxy_material(rep["materials"], material_index, verts, tris),
    }
    bones = [{"name": b["name"], "parent": b["parent"], "head": b["head"],
              "tail": b["tail"], "deform": b["deform"]} for b in rep["bones"]]
    native_rig = {
        "schema": NATIVE_RIG_SCHEMA,
        "part": BODY_PART,
        "blend": blend_rel,
        "digest": blend_dig,
        "collection": rep["collection"],
        "armature": rep["armature"],
        "mesh": rep["mesh"],
        "root_bones": rep["root_bones"],
        "bones": bones,
        "actions": [{k: a[k] for k in ("name", "clip", "frame_start",
                                       "frame_end", "fps", "seconds")}
                    for a in rep["actions"]],
        "shape_keys": rep["shape_keys"],
        "anatomy": skeleton_anatomy(rep["bones"], anatomy, height=float(rep["extent_m"][2])),
    }
    ir = insert_id({
        "schema": OBJECT_SCHEMA_VERSION,
        "name": name,
        "root": BODY_PART,
        "parts": [part],
        "joints": [],
        "assertions": [],
        "collision_allowances": [],
        "native_rig": native_rig,
    })
    (out_dir / "ir.json").write_text(json.dumps(ir, indent=1))

    note = {
        "converter": CONVERTER,
        "converter_version": CONVERTER_VERSION,
        "primitive": PRIMITIVE_IMPORTED,
        "ir_id": ir["id"],
        "object_name": name,
        "source_gltf": str(src.resolve()),
        "blender": rep["blender"],
        "representation": (
            "one proxy part (the bind-shape skin) and no joints; the rig is "
            f"{blend_rel}, swapped in for part {BODY_PART!r} by the native "
            "animation import and deformed by its Armature modifier"),
        "yaw_deg": rep["yaw_deg"],
        "yaw_from": "--yaw auto (measured facing)" if yaw == "auto" else "--yaw",
        "scale": rep["scale"],
        "raw": rep["raw"],
        "facing_after": rep["facing"],
        "normalisation": rep["normalisation"],
        "bone_space_factor": rep["bone_space_factor"],
        "rescaled_location_channels": rep["rescaled_location_channels"],
        "importer": rep["importer"],
        **({"unit_scale_length": rep["unit_scale_length"]} if "unit_scale_length" in rep else {}),
        **({"blend": rep["blend"]} if "blend" in rep else {}),
        **({"pose_defaults": rep["pose_defaults"]} if "pose_defaults" in rep else {}),
        **({"attached_loose_bones": rep["attached_loose_bones"]} if rep.get("attached_loose_bones") else {}),
        **({"merged_skeleton_copies": rep["merged_skeleton_copies"]} if rep.get("merged_skeleton_copies") else {}),
        **({"object_parented_parts": rep["object_parented_parts"]} if rep.get("object_parented_parts") else {}),
        **({"material_repairs": rep["material_repairs"]} if "material_repairs" in rep else {}),
        **({"weighted_unweighted_vertices": rep["weighted_unweighted_vertices"]}
           if rep.get("weighted_unweighted_vertices") else {}),
        **({"particle_hair": rep["particle_hair"]} if rep.get("particle_hair") else {}),
        "bone_heuristic": rep["bone_heuristic"],
        "relinked_images": rep["relinked_images"],
        "missing_images": rep["missing_images"],
        "dropped_objects": rep["dropped_objects"],
        "dropped_object_transform_keys": rep["dropped_object_transform_keys"],
        "self_check": rep["self_check"],
        "clip_policy": rep["clip_policy"],
        "dropped_clips": rep["dropped_clips"],
        "extent_m": rep["extent_m"],
        "n_bones": len(bones),
        "root_bones": rep["root_bones"],
        "vertex_groups_without_bone": rep["vertex_groups_without_bone"],
        "actions": rep["actions"],
        "shape_keys": rep["shape_keys"],
        "materials": rep["materials"],
        **({"uv_maps_unified": rep["uv_maps_unified"]} if rep.get("uv_maps_unified") else {}),
        "proxy": {"vertices_before_weld": int(len(raw_v)),
                  "vertices": int(len(verts)), "triangles": int(len(tris)),
                  "skin_closed": part["params"]["shape"]["closed"],
                  "boundary_edges": int(wt["n_boundary_edges"]),
                  "collision": how,
                  "collision_triangles": int(len(ct)),
                  "material": "area-weighted mean of the skin's materials; "
                              "the blend keeps the real ones"},
        "joint_limits": (
            {"source": "authored in the source; kept on the rig in the blend as "
                       "Blender constraints and IK limits, which act on the "
                       "evaluated pose (JointLimits reads no bone q)",
             "bones": rep["authored_limits"]}
            if rep.get("authored_limits") else
            "none: glTF declares no limits, and a bone has no q for "
            "JointLimits to read"),
        "anatomy": {k: v for k, v in native_rig["anatomy"].items() if k != "joints"}
                   | {"joints": [j["name"] for j in native_rig["anatomy"]["joints"]]},
    }
    (out_dir / "import.json").write_text(json.dumps(note, indent=1))
    return ir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _yaw(value: str):
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--yaw takes 'auto' or degrees")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dycraft.agent.assets.import_gltf",
        description="Import a skinned glTF/GLB character as a native Blender "
                    "rig bundle for --preload-asset.")
    ap.add_argument("--in", dest="src", required=True,
                    help="a .glb/.gltf, .fbx, or .blend file (an FBX finds its textures "
                         "by the paths it records, so keep them beside it)")
    ap.add_argument("--out", dest="out", help="bundle directory to write")
    ap.add_argument("--name", help="object name (default: the file stem)")
    ap.add_argument("--yaw", type=_yaw, default="auto",
                    help="degrees about +Z to turn the figure so it faces +Y; "
                         "'auto' (default) measures its facing from its feet")
    ap.add_argument("--up", choices=("+z", "-z", "+x", "-x", "+y", "-y"), default="+z",
                    help="which of the file's axes is the figure's up (default +z); "
                         "e.g. -z for a figure lying on its back. Applied before --yaw")
    ap.add_argument("--rest", default="bind",
                    help="the rig's rest: its bind pose (default), or the figure as "
                         "displayed, when a bone above the whole body that the clips "
                         "also key stands a tipped-over bind up (a Y-up FBX), or "
                         "clip:<name>, the trunk as that clip's first frame stands it "
                         "(a bind modeled on end whose own clip stands it up), or "
                         "clip-pose:<name>, the whole pose at that clip's first frame "
                         "(a bind with wings spread and beak open whose clip starts perched)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="uniform scale on top of the file's own units "
                         "(Blender already resolves a centimeter armature)")
    ap.add_argument("--clips", choices=("all", "matching", "none"), default="all",
                    help="the asset's own clips: 'all' (default) refuses an "
                         "import any clip of which no longer matches its source "
                         "after normalization; 'matching' drops just those and "
                         "records them; 'none' imports the rig without clips "
                         "(also admits a non-uniformly scaled armature)")
    ap.add_argument("--anatomy", choices=ANATOMY_POLICIES, default="auto",
                    help="human joint limits for AnatomicalJointLimits: 'auto' "
                         "(default) when the skeleton is a known family and the "
                         "figure is built like a standing human; 'human' to apply "
                         "them to any known family; 'none' to declare none")
    ap.add_argument("--reparent", action="append", default=[], metavar="BONE=PARENT",
                    help="hang BONE from PARENT, keeping every clip's motion; for a "
                         "part skinned to a bone the rig leaves behind its body, e.g. "
                         "a hoof under an IK controller that stays planted when the "
                         "leg is posed (repeatable)")
    ap.add_argument("--collision", choices=("auto", "mesh", "hull"),
                    default="auto", help="the proxy part's collision mesh")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing bundle at --out")
    ap.add_argument("--inspect", action="store_true",
                    help="measure and print what is in the file; write nothing")
    args = ap.parse_args(argv)
    if args.scale <= 0:
        ap.error("--scale must be positive")
    reparent = {}
    for item in args.reparent:
        bone, sep, parent = item.partition("=")
        if not (sep and bone and parent):
            ap.error(f"--reparent takes BONE=PARENT, got {item!r}")
        reparent[bone] = parent
    try:
        if args.inspect:
            print(json.dumps(inspect(args.src, name=args.name, up=args.up, rest=args.rest), indent=1))
            return 0
        if not args.out:
            ap.error("--out is required unless --inspect")
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            if not args.force:
                ap.error(f"{out} is not empty; pass --force to replace it")
            if not (out / "import.json").is_file():
                ap.error(f"{out} is not a bundle (no import.json); refusing "
                         "to replace it even with --force")
            shutil.rmtree(out)
        ir = emit_bundle(args.src, out, name=args.name, yaw=args.yaw,
                         scale=args.scale, collision=args.collision,
                         clips=args.clips, anatomy=args.anatomy, up=args.up, rest=args.rest,
                         reparent=reparent)
    except CharacterImportError as e:
        print(f"import_gltf: {e}", file=sys.stderr)
        return 2
    note = json.loads((Path(args.out) / "import.json").read_text())
    print(json.dumps({"ir_id": ir["id"], "out": args.out,
                      "extent_m": note["extent_m"], "yaw_deg": note["yaw_deg"],
                      "bones": note["n_bones"],
                      "actions": [a["name"] for a in note["actions"]],
                      "collision": note["proxy"]["collision"],
                      "self_check_m": note["self_check"]["max_residual_m"],
                      "dropped_clips": note["dropped_clips"],
                      "anatomy": note["anatomy"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
