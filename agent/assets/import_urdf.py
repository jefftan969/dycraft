"""Convert a third-party URDF into a DyCraft ObjectIR bundle. Offline, no LLM.

    python -m dycraft.agent.assets.import_urdf \
        --urdf assets/robots/franka_panda/panda.urdf \
        --out  assets/imported/franka_panda \
        --name panda_arm

The agent-free counterpart to an object run, like `fixed_source.py` — except
that one *authors* geometry through the build123d kernel and this one *imports*
it. Nothing here runs in the sandbox and nothing here is reachable from
`object.py`: an imported part carries no `source`, so there is no recipe for the
object agent to edit, which is exactly what makes the asset frozen.

What it writes under `--out` (the layout of a committed object run, so the
scene layer's `materialize_catalog` ingests it unchanged):

    ir.json                        the ObjectIR, content-hashed
    assets/meshes/<digest>.obj     per-part VISUAL mesh
    assets/collision/<digest>.obj  per-part COLLISION mesh
    assets/textures/<digest>.png   base-color image, when the source has one
    import.json                    the conversion recipe + provenance

**Two meshes per part, not one.** The Panda's visual meshes run 6-15k faces per
link and its collision meshes 200-300; one asset serving both would either cost
40x in the coal BVH or render a robot as its own convex hull. `params["mesh"]`
is what the renderer and URDF visual read, `params["collision"]` is what the
narrow phase reads.

**Mass properties come from the mesh, not from the URDF.** Every link in
`assets/robots/franka_panda/panda.urdf` declares `ixx=iyy=izz=0.1` with zero
products — placeholders, not measurements. So the shape fact table is integrated
from the triangles (divergence theorem, unit density, about the true centroid),
which makes an imported part's `shape` the same *kind* of object as a CAD
solid's and lets `_primitives.inertia` serve both with no new branch.

Note the cost, because it is asset-dependent and nothing warns: an asset whose
inertials are real loses them. The G1 declares 34.13 kg (Unitree's spec is ~35)
and comes out at 20.90 kg at 1000 kg/m^3, its foot 299x light. `import.json`
records `urdf_inertial` per part so the two are comparable, and nothing compares
them yet.

**From the VISUAL mesh by default** (`--mass-from`). The visual mesh is the
part — it is what renders, what a reviewer judges, and what the AABB and the
27-name anchor vocabulary come from — so the `shape` table describes it and
nothing else. An open shell falls through to the hull of *itself* rather than
borrowing the collision proxy's volume: an over-estimate of the right body
instead of a measurement of the wrong one. The G1's foot shows why: a 10 mm
sole plate holding 0.7% of the visible 292 cm^3, with both
meshes closed and oriented, so the converter's own watertight check passes on
both and the 143x gap shows up in no diagnostic.

**Textures survive when the source OBJ has them.** A visual `<mesh>` that cites
an `mtllib` with a `map_Kd` gets that image staged content-addressed and named
by the part's `material.texture.base_color` — the same representation
`import_gltf` produces, so a robot and a character are one kind of thing
downstream. Two things make this the *only* honest way to carry the Panda's
look: its 22 MTLs all name one `colors.png`, and its per-corner `vt` indices are
what put the dark collars and the FRANKA logo in the right place, so the UVs are
carried through the weld as `(T,3,2)` rather than collapsed per-vertex. The URDF
`<material><color>` stays in `material.color` as a fallback for a consumer that
cannot sample (URDF export, flat shading) — it is not a tint; see
`sdk/_material.Texture`.

A link whose *visual* element points at a mesh with no readable material is
reported in `import.json` under `parts[].texture.why` rather than passed over:
the Panda's `link0` and `link7` name their collision OBJ as their visual, and
those two cite MTLs the distribution does not ship. Collision geometry drops UVs
and textures outright — containment has nothing to sample, and `--collision
auto` may swap the mesh for a hull.

**Frames.** A DyCraft joint's `parent_anchor` / `child_anchor` are raw SE3s in
their parts' local frames, and FK composes `rest = pa o inv(ca)` with the axis
in pa-local and the pivot at `pa.translation`. That is *exactly* URDF's
`<origin>` + joint-frame `<axis>`, so the conversion is term-for-term:
`parent_anchor = origin`, `child_anchor = identity`, `axis` verbatim. The 27-name
AABB anchor vocabulary is an authoring convenience that lives above the IR; it
is not involved.

No axis remap is applied by default (`--yaw-deg 0`): the IR's joint origins stay
numerically identical to the URDF's, which is what makes the import checkable
against upstream and round-trippable back out to cuRobo/Genesis. DyCraft's
"functional front faces +Y" is a soft convention with nothing recording it,
and a base yaw is a *scene-layer* placement decision
that costs nothing there — so baking one in here would trade a verifiable
property for an unverifiable one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# `sdk._rotation` and `sdk.ir` only: this module must agree with the kernel on
# quaternion order and on the canonical hash, and the way to guarantee that is
# to call the same code rather than to reimplement it.
try:
    from ...sdk import _rotation as _rot
    from ...sdk.ir import OBJECT_SCHEMA_VERSION, insert_id
except ImportError:  # standalone execution (python agent/assets/import_urdf.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dycraft.sdk import _rotation as _rot
    from dycraft.sdk.ir import OBJECT_SCHEMA_VERSION, insert_id


#: The imported-asset shape tag. Deliberately not `solid` (it has no B-rep and
#: no authored source) and deliberately not the retired `mesh` tag (that one was
#: agent-authorable, carried a `scale`, and proxied its inertia off the AABB).
PRIMITIVE_IMPORTED = "imported"

MESHES_SUBDIR = "assets/meshes"
COLLISION_SUBDIR = "assets/collision"

HASH_DIGEST_BYTES = 16          # matches sdk/_mesh_assets.HASH_DIGEST_BYTES
WELD_QUANTUM_M = 1.0e-7         # 0.1 um: below any real feature, above float noise

DEFAULT_COLOR = (176, 176, 176)  # sdk/_material.DEFAULT_PART_COLOR


class ImportError_(Exception):
    """A URDF this converter refuses to convert."""


# ---------------------------------------------------------------------------
# Rotation: URDF rpy is FIXED-AXIS (extrinsic) XYZ
# ---------------------------------------------------------------------------

def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """URDF `rpy` (radians, fixed-axis XYZ) -> WXYZ quaternion.

    `R = Rz(yaw) . Ry(pitch) . Rx(roll)`.

    Built from `quat_mul` in that order rather than through
    `_rot.euler_xyz_deg_to_quat_wxyz`, whose convention is the *intrinsic* XYZ
    `Rx . Ry . Rz` (see its docstring, and the warning in
    `render/urdf_export.py`). The two agree only when at most one angle is
    non-zero, and the Panda has six joints with a +-pi/2 roll — so reusing the
    intrinsic helper here would place most of the arm wrongly while every
    single-axis test case still passed.
    """
    qx = _rot.quat_from_axis_angle_rad((1.0, 0.0, 0.0), roll)
    qy = _rot.quat_from_axis_angle_rad((0.0, 1.0, 0.0), pitch)
    qz = _rot.quat_from_axis_angle_rad((0.0, 0.0, 1.0), yaw)
    return _rot.quat_mul(qz, _rot.quat_mul(qy, qx))


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def se3_to_matrix(se3) -> np.ndarray:
    """`((t), (quat_wxyz))` -> 4x4 homogeneous matrix."""
    t, q = se3
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def se3_to_ir(se3) -> dict:
    """Match `object_builder._se3_internal_to_ir` byte for byte."""
    (tx, ty, tz), q = se3
    q = _rot.sign_canonicalize_quat(_rot.normalize_quat(q))
    return {
        "translation": [float(tx), float(ty), float(tz)],
        "rotation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


IDENTITY_IR_SE3 = {"translation": [0.0, 0.0, 0.0],
                   "rotation": [1.0, 0.0, 0.0, 0.0]}


# ---------------------------------------------------------------------------
# OBJ I/O
# ---------------------------------------------------------------------------

def load_obj(path: Path, *, with_uvs: bool = False):
    """Parse an OBJ into `(verts[n,3], tris[m,3])`, or a triple with its UVs.

    Fan-triangulates polygons and resolves the negative (relative) index form.
    Deliberately a third reader alongside `_deform_eval.load_obj` and
    `_mesh_assets._parse_obj_aabb` rather than an import of either: this one runs
    outside the package's runtime path, on third-party files, and has to tolerate
    what they contain.

    `with_uvs=True` adds a third element: **per-corner** texture coordinates of
    shape `(T, 3, 2)`, or None when the file declares no `vt` at all. Per-corner
    is not a conversion here — it is what OBJ already stores, because `f v/vt`
    carries two independent index streams. That is also why UVs survive `weld`,
    which renumbers `v` and never touches `vt`.

    **A face that omits its texture index gets zeros, it does not void the
    file.** Real assets mix the two forms: Franka's `link6.obj` writes 20287
    faces as `v/vt/vn` and 1333 as `v//vn`, and refusing the whole mesh over
    that last 6% would discard UVs for the other 94%. OBJ cannot express "this
    face has no vt" once any face has one — `f v/vt` must be uniform — so the
    untextured corners land at (0, 0) and sample the atlas's corner. That is the
    same trade `import_gltf._extend_uv_for_caps` makes for cap fans, and for the
    same reason: a wrong-but-bounded coordinate beats losing the whole map.

    The V axis is passed through unflipped. An OBJ's origin is bottom-left, which
    is what every consumer downstream expects; only glTF needs a flip, and
    `import_gltf` does it there.
    """
    verts: list = []
    tris: list = []
    vt: list = []
    corner_vt: list = []
    n_faces_without_vt = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            fields = raw.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                verts.append((float(fields[1]), float(fields[2]),
                              float(fields[3])))
            elif fields[0] == "vt" and len(fields) >= 3:
                vt.append((float(fields[1]), float(fields[2])))
            elif fields[0] == "f" and len(fields) >= 4:
                face, face_vt = [], []
                for tok in fields[1:]:
                    bits = tok.split("/")
                    idx = int(bits[0])
                    face.append(idx - 1 if idx > 0 else len(verts) + idx)
                    if len(bits) >= 2 and bits[1]:
                        t = int(bits[1])
                        face_vt.append(t - 1 if t > 0 else len(vt) + t)
                    else:
                        face_vt.append(-1)          # -> (0, 0) below
                        n_faces_without_vt += 1
                for k in range(1, len(face) - 1):
                    tris.append((face[0], face[k], face[k + 1]))
                    corner_vt.append((face_vt[0], face_vt[k], face_vt[k + 1]))
    if not verts or not tris:
        raise ImportError_(
            f"{path}: no usable triangles ({len(verts)} v, {len(tris)} f)")
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64)
    if not with_uvs:
        return v, t
    uvs = None
    if vt and len(corner_vt) == len(tris):
        idx = np.asarray(corner_vt, dtype=np.int64)
        # A zero row appended so index -1 (a corner with no vt) resolves to
        # (0, 0) through ordinary fancy indexing rather than a branch.
        table = np.concatenate([np.asarray(vt, dtype=np.float64),
                                np.zeros((1, 2), dtype=np.float64)])
        idx = np.where((idx < 0) | (idx >= len(table) - 1), len(table) - 1, idx)
        uvs = table[idx]                         # (T, 3, 2)
    return v, t, uvs


def load_stl(path: Path):
    """Parse a binary or ASCII STL into `(verts[n,3], tris[m,3])`.

    Hand-rolled rather than delegated to Open3D for two reasons: this module
    stays numpy-only (it runs as an offline tool, not inside a render env), and
    STL shares no vertices at all — every triangle carries its own three — so
    the output must be welded regardless of who parses it. `weld` does that; a
    reader that silently returned unshared vertices would make `watertight`
    report every closed hull as an open shell.

    Binary detection is by size, not by the "solid" magic: an ASCII file always
    starts with `solid`, but so do some binary writers' 80-byte headers. A
    binary STL is exactly `84 + 50 * n_triangles` bytes, which no ASCII file
    matches by accident.
    """
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + 50 * n_tri and n_tri > 0:
            # 50 bytes/triangle: 3 float32 normal + 9 float32 verts + uint16.
            rec = np.frombuffer(data, dtype=np.dtype([
                ("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]),
                count=n_tri, offset=84)
            verts = rec["v"].reshape(-1, 3).astype(np.float64)
            tris = np.arange(3 * n_tri, dtype=np.int64).reshape(-1, 3)
            return verts, tris

    verts: list = []
    tris: list = []
    cur: list = []
    for raw in data.decode("utf-8", errors="replace").splitlines():
        f = raw.split()
        if not f:
            continue
        if f[0] == "vertex" and len(f) >= 4:
            cur.append((float(f[1]), float(f[2]), float(f[3])))
        elif f[0] == "endloop":
            base = len(verts)
            verts.extend(cur)
            for k in range(1, len(cur) - 1):
                tris.append((base, base + k, base + k + 1))
            cur = []
    if not tris:
        raise ImportError_(f"{path}: no usable triangles in STL")
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(tris, dtype=np.int64))


def load_mesh(path: Path, *, with_uvs: bool = False):
    """Dispatch on suffix. OBJ and STL only — see `ImportError_` for why.

    COLLADA (`.dae`) is deliberately unsupported rather than half-supported:
    Open3D's reader returns an empty mesh for every `.dae` in `assets/robots`,
    and a silently-empty mesh would surface as a zero-volume part rather than as
    a missing feature. A URDF citing `.dae` is refused by name.
    """
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path, with_uvs=with_uvs)
    if suffix == ".stl":
        # STL carries no texture coordinates at all — not "none in this file",
        # the format has no place for them. So an STL-only robot (`g1`, `h1`,
        # `go2`) can never be textured, and that is the asset's ceiling rather
        # than a gap here.
        v, t = load_stl(path)
        return (v, t, None) if with_uvs else (v, t)
    raise ImportError_(
        f"{path.name}: unsupported mesh format {suffix!r} — this converter "
        "reads .obj and .stl. COLLADA (.dae) needs a parser this module does "
        "not have; converting an asset whose meshes are .dae is a separate "
        "piece of work, not a flag.")


# ---------------------------------------------------------------------------
# URDF primitive collision shapes, triangulated
# ---------------------------------------------------------------------------

def _box_mesh(sx: float, sy: float, sz: float):
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy)
                      for z in (-hz, hz)], dtype=np.float64)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                     [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                     [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
                    dtype=np.int64)
    return verts, tris


def _cylinder_mesh(radius: float, length: float, n: int = 24):
    """Closed cylinder about local +Z, centred — URDF's convention."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
    hz = length / 2.0
    verts = np.concatenate([
        np.column_stack([ring, np.full(n, -hz)]),
        np.column_stack([ring, np.full(n, hz)]),
        np.array([[0.0, 0.0, -hz], [0.0, 0.0, hz]]),
    ])
    bot_c, top_c = 2 * n, 2 * n + 1
    tris = []
    for i in range(n):
        j = (i + 1) % n
        # Side, wound so the normal is radially OUTWARD. `[i, n+i, n+j]` is the
        # tempting order and gives `z_hat x tangential` = inward; the resulting
        # mesh is closed but not consistently oriented, so its
        # divergence-theorem volume partially cancels instead of failing.
        tris += [[i, j, n + j], [i, n + j, n + i]]
        tris.append([bot_c, j, i])                      # bottom cap (outward -Z)
        tris.append([top_c, n + i, n + j])              # top cap (outward +Z)
    return verts, np.asarray(tris, dtype=np.int64)


def _sphere_mesh(radius: float, n_lat: int = 12, n_lon: int = 24):
    """Closed UV sphere centred at the origin."""
    verts = [[0.0, 0.0, radius]]
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            th = 2.0 * np.pi * j / n_lon
            verts.append([radius * np.sin(phi) * np.cos(th),
                          radius * np.sin(phi) * np.sin(th),
                          radius * np.cos(phi)])
    verts.append([0.0, 0.0, -radius])
    south = len(verts) - 1

    def idx(i, j):
        return 1 + (i - 1) * n_lon + (j % n_lon)

    tris = []
    for j in range(n_lon):
        tris.append([0, idx(1, j), idx(1, j + 1)])
        tris.append([south, idx(n_lat - 1, j + 1), idx(n_lat - 1, j)])
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            tris += [[a, c, d], [a, d, b]]
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def primitive_mesh(tag: str, attrib: dict):
    """Triangulate a URDF `<box>` / `<cylinder>` / `<sphere>` collision shape.

    Every result is closed and consistently oriented, so a part whose collision
    geometry is primitives gets real mass properties instead of falling back to
    its full-detail visual mesh. That fallback is the whole reason this exists:
    the Unitree H1 and Go2 declare *all* of their collision volume as primitives
    (23 and 27 shapes, zero collision meshes), so without this the narrow phase
    would collide their visual meshes.
    """
    if tag == "box":
        sx, sy, sz = (float(v) for v in attrib["size"].split())
        return _box_mesh(sx, sy, sz)
    if tag == "cylinder":
        return _cylinder_mesh(float(attrib["radius"]), float(attrib["length"]))
    if tag == "sphere":
        return _sphere_mesh(float(attrib["radius"]))
    raise ImportError_(f"unsupported collision primitive <{tag}>")


def write_obj(verts: np.ndarray, tris: np.ndarray, name: str,
              uvs: np.ndarray | None = None) -> bytes:
    """Serialize to OBJ bytes. `%.9g` keeps micron precision without noise.

    `uvs`, when given, is **per-corner**: shape `(T, 3, 2)`, one texture
    coordinate per triangle corner rather than per vertex. That is the whole
    reason texture survives this pipeline: OBJ's `f v/vt` carries two
    independent index streams, so `weld`ing vertices renumbers `v` while
    leaving `vt` untouched. A per-*vertex* UV array could not survive, because
    a welded corner of a box has one position and three different UVs.

    Omitting `uvs` reproduces the previous output **byte for byte** — every
    already-staged mesh keeps its content digest, so no catalog is invalidated
    by this parameter existing.
    """
    out = [f"# dycraft imported part: {name}\n"]
    for v in verts:
        out.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
    if uvs is None:
        for t in tris:
            out.append(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        return "".join(out).encode("utf-8")

    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.shape != (len(tris), 3, 2):
        raise ValueError(
            f"uvs must be per-corner with shape ({len(tris)}, 3, 2), got "
            f"{uvs.shape} — one UV per triangle corner, not per vertex")
    # Deduplicate: the source is usually per-vertex UVs fanned out to corners,
    # so the corner array holds ~3x more entries than distinct values. np.unique
    # sorts, which makes the output deterministic (a dict-order dedupe would
    # not be) and therefore keeps the content digest reproducible.
    flat = uvs.reshape(-1, 2)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1, 3)
    for uv in uniq:
        out.append(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
    for t, c in zip(tris, inverse):
        out.append(f"f {t[0] + 1}/{c[0] + 1} {t[1] + 1}/{c[1] + 1} "
                   f"{t[2] + 1}/{c[2] + 1}\n")
    return "".join(out).encode("utf-8")


#: Where a base-color image goes, content-addressed like the meshes. Defined
#: here rather than in `import_gltf` because both importers stage into it: the
#: bytes reach this function the same way whether they were sliced out of a GLB's
#: BIN chunk or read off disk next to an OBJ's `.mtl`.
TEXTURES_SUBDIR = "assets/textures"

_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def stage_texture(out_dir: Path, texture: dict | None):
    """Write a base-color image content-addressed; return its relpath or None.

    `texture` is `{"base_color": <bytes>, "mime": "image/png"|"image/jpeg"}`.
    The bytes go to disk verbatim — a glTF image and a `map_Kd` file are both
    already complete PNG/JPEG files, so there is nothing to re-encode — and the
    extension follows the declared mime so ordinary tools (and Open3D's image
    reader) can open it.
    """
    if not texture:
        return None
    data = texture["base_color"]
    ext = ".jpg" if texture.get("mime") == "image/jpeg" else ".png"
    rel = f"{TEXTURES_SUBDIR}/{digest_bytes(data)}{ext}"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return rel


def base_color_texture_ir(rel: str | None) -> dict | None:
    """The `material.texture` block for a staged base-color image, or None.

    Returns the shared shape — ``{"base_color": {"image", "digest"}}`` — which
    is what `sdk._material.Texture` reads and what every consumer downstream
    enumerates through `texture_maps_from_ir`. Base color is a map kind like
    any other; an imported asset simply has only that one.

    The digest is the file's stem, not a re-hash: `stage_texture` names the file
    `<blake2b-128 of its bytes><ext>`, so the stem IS the digest by
    construction, and computing it again would be a second source of truth for
    the same fact.

    **No `mapping` accompanies it, deliberately.** A tile size says how to lay a
    texture out over a surface that has no coordinates of its own; an imported
    asset arrives with its UVs authored into the staged OBJ, so there is nothing
    to lay out. Its absence is what tells every renderer to sample the mesh's
    own coordinates rather than derive new ones.
    """
    if not rel:
        return None
    return {"base_color": {"image": rel, "digest": Path(rel).stem}}


def read_obj_texture(path: Path):
    """The base-color image an OBJ's material library points at, or None.

    Follows `mtllib` (relative to the OBJ) and then `map_Kd` (relative to the
    MTL), which is the only texture channel this importer reads — `map_Bump`,
    `map_Ks` and friends would need `Material.texture` to grow the matching
    fields first.

    Returns `(texture_or_None, note)`. `note` is a sentence when there was
    something to read and it could not be, so a caller can report the loss
    instead of silently shipping a flat color.
    """
    try:
        mtl_names = [l.split(None, 1)[1].strip()
                     for l in path.read_text(encoding="utf-8",
                                             errors="replace").splitlines()
                     if l.startswith("mtllib ")]
    except OSError as e:
        return None, f"{path.name} unreadable: {e}"
    if not mtl_names:
        return None, None                     # no material library; not a loss
    maps: list = []
    for name in mtl_names:
        mtl = (path.parent / name).resolve()
        if not mtl.is_file():
            return None, f"{path.name} cites {name!r}, which is not on disk"
        for line in mtl.read_text(encoding="utf-8",
                                  errors="replace").splitlines():
            fields = line.split()
            # `map_Kd [options] filename` — options start with '-', and a
            # filename may contain spaces, so take everything after the last
            # option token.
            if fields and fields[0] == "map_Kd" and len(fields) >= 2:
                rest = fields[1:]
                while rest and rest[0].startswith("-"):
                    rest = rest[2:] if len(rest) > 1 else []
                if rest:
                    maps.append((mtl.parent / " ".join(rest)).resolve())
    if not maps:
        return None, None                     # MTL with no map_Kd; flat color
    uniq = sorted({m for m in maps})
    if len(uniq) > 1:
        # Same reason `import_gltf` refuses a multi-atlas asset: one mesh has one
        # UV set, so sampling image B at image A's coordinates would be a
        # confident wrong color rather than a missing one.
        return None, (f"{path.name} references {len(uniq)} different map_Kd "
                      f"images, and one mesh has only one UV set")
    img = uniq[0]
    if not img.is_file():
        return None, f"{path.name} cites map_Kd {img.name!r}, not on disk"
    mime = _IMAGE_MIME.get(img.suffix.lower())
    if mime is None:
        return None, f"unsupported map_Kd image type {img.suffix!r}"
    return {"base_color": img.read_bytes(), "mime": mime}, None


def digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=HASH_DIGEST_BYTES).hexdigest()


# ---------------------------------------------------------------------------
# Mesh cleanup
# ---------------------------------------------------------------------------

def transform_verts(verts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return verts @ m[:3, :3].T + m[:3, 3]


def merge(pieces) -> tuple:
    """Concatenate `[(verts, tris)]` — or `[(verts, tris, uvs)]` — into one set.

    Returns a pair for the first form and a triple for the second. Per-corner
    UVs concatenate along the **triangle** axis and need no index offsetting,
    which is the same property that carries them through `weld`. A piece with no
    UVs contributes zeros, so the result is only meaningful when every piece had
    them — the caller checks that, because "some pieces textured" is a decision
    about materials, not about geometry.
    """
    if not pieces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    with_uvs = len(pieces[0]) == 3
    vs, ts, us, off = [], [], [], 0
    for piece in pieces:
        v, t = piece[0], piece[1]
        vs.append(v)
        ts.append(t + off)
        off += len(v)
        if with_uvs:
            u = piece[2]
            us.append(u if u is not None
                      else np.zeros((len(t), 3, 2), dtype=np.float64))
    verts, tris = np.concatenate(vs), np.concatenate(ts)
    if not with_uvs:
        return verts, tris
    return verts, tris, np.concatenate(us)


def weld(verts: np.ndarray, tris: np.ndarray, quantum: float = WELD_QUANTUM_M,
         *, return_keep: bool = False):
    """Merge coincident vertices and drop degenerate triangles.

    Required, not cosmetic: the Panda's collision OBJs store one vertex per
    face corner (`link4.obj`: 900 vertices for 300 faces), so *no* edge is
    shared by index and any watertightness test run before welding reports a
    perfectly closed hull as an open shell.
    """
    keys = np.round(verts / quantum).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    order = np.argsort(first)                      # keep original vertex order
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    new_verts = verts[first[order]]
    new_tris = remap[inverse[tris.reshape(-1)]].reshape(-1, 3)
    keep = ((new_tris[:, 0] != new_tris[:, 1])
            & (new_tris[:, 1] != new_tris[:, 2])
            & (new_tris[:, 0] != new_tris[:, 2]))
    if return_keep:
        # The mask is over the INPUT triangles and welding never reorders the
        # ones it keeps, so `per_corner_attr[keep]` is all a caller needs to hold
        # a per-corner array (texture coordinates) in step with the result.
        return new_verts, new_tris[keep], keep
    return new_verts, new_tris[keep]


def watertight(verts: np.ndarray, tris: np.ndarray) -> dict:
    """Report whether the surface is closed and consistently oriented.

    Closed: every undirected edge is shared by exactly two triangles.
    Oriented: every directed edge appears exactly once, i.e. the two triangles
    sharing an edge traverse it in opposite directions.

    Both matter, and for the same reason: the divergence-theorem volume below
    is only the enclosed volume when they hold. It does not *fail* otherwise —
    it returns a plausible wrong number, which is the failure mode this
    function exists to make loud.
    """
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    directed, dcount = np.unique(e, axis=0, return_counts=True)
    undirected, ucount = np.unique(np.sort(e, axis=1), axis=0,
                                   return_counts=True)
    return {
        "closed": bool(np.all(ucount == 2)),
        "oriented": bool(np.all(dcount == 1)),
        "n_boundary_edges": int(np.sum(ucount == 1)),
        "n_nonmanifold_edges": int(np.sum(ucount > 2)),
        "n_duplicated_directed_edges": int(np.sum(dcount > 1)),
        "n_directed": int(len(directed)),
        "euler_characteristic": int(len(verts) - len(undirected) + len(tris)),
    }


def convex_hull(verts: np.ndarray):
    """Convex hull as `(verts, tris)` with consistently OUTWARD winding.

    Qhull does not promise a consistent orientation across `simplices`, so each
    face is flipped to agree with its own outward plane normal from
    `equations`. Without that step the hull is closed but not oriented, the
    signed volume partially cancels, and `mass_properties` returns a number
    that looks reasonable and is wrong.
    """
    from scipy.spatial import ConvexHull

    h = ConvexHull(np.asarray(verts, dtype=np.float64))
    hv = verts[h.vertices]
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[h.vertices] = np.arange(len(h.vertices))
    tris = []
    for simplex, eq in zip(h.simplices, h.equations):
        a, b, c = (int(remap[i]) for i in simplex)
        n = np.cross(hv[b] - hv[a], hv[c] - hv[a])
        if float(n @ eq[:3]) < 0.0:
            b, c = c, b
        tris.append((a, b, c))
    return hv, np.asarray(tris, dtype=np.int64)


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------

#: Second-moment matrix of the canonical tetrahedron (0, e1, e2, e3):
#: integral of x x^T over its volume. Under a linear map A the integral becomes
#: det(A) * A C A^T, which is what makes the per-triangle accumulation below a
#: sum of signed tetrahedra rather than a surface integral.
_C_CANONICAL = np.array([[2.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0],
                         [1.0, 1.0, 2.0]]) / 120.0


def mass_properties(verts: np.ndarray, tris: np.ndarray) -> dict:
    """Unit-density volume, centroid, and inertia diagonal about that centroid.

    Divergence theorem as a sum over the signed tetrahedra `(0, a, b, c)`, so
    every triangle contributes and the parts outside the solid cancel. Returns
    the same three quantities the CAD kernel puts in a `solid`'s `shape` table
    (`volume`, `center`, `inertia_diag`), in the same units and about the same
    point, so `_primitives.inertia` treats both identically.

    `inertia_diag` is the diagonal of the tensor about the centroid at unit
    density. Products of inertia are computed (they are needed to shift the
    tensor) and then dropped, matching the `(mass, ixx, iyy, izz)` contract
    every consumer expects.
    """
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]

    det = np.einsum("ij,ij->i", a, np.cross(b, c))     # 6 * signed tet volume
    volume = float(det.sum() / 6.0)

    # Centroid: sum over tets of (vol_i * centroid_i) / total volume.
    tet_centroids = (a + b + c) / 4.0
    com_num = np.einsum("i,ij->j", det / 6.0, tet_centroids)

    # Second moments about the ORIGIN: sum_i det_i * A_i C A_i^T, with A_i the
    # matrix whose columns are (a, b, c).
    A = np.stack([a, b, c], axis=2)                    # (n, 3, 3), columns abc
    m2 = np.einsum("i,ijk->jk", det,
                   np.einsum("ijp,pq,ikq->ijk", A, _C_CANONICAL, A))
    m2 = 0.5 * (m2 + m2.T)                             # symmetrize float noise

    if abs(volume) <= 0.0:
        return {"volume": 0.0, "center": [0.0, 0.0, 0.0],
                "inertia_diag": [0.0, 0.0, 0.0], "inertia_full": None}

    com = com_num / volume
    # I = tr(M2) * Id - M2, then translate the tensor from the origin to the
    # centroid (parallel-axis, subtracting because we move TO the centroid).
    i_origin = np.trace(m2) * np.eye(3) - m2
    d = com
    shift = volume * ((d @ d) * np.eye(3) - np.outer(d, d))
    i_com = i_origin - shift
    return {
        "volume": abs(volume),
        "center": [float(v) for v in com],
        "inertia_diag": [float(max(0.0, i_com[0, 0])),
                         float(max(0.0, i_com[1, 1])),
                         float(max(0.0, i_com[2, 2]))],
        "inertia_full": [[float(v) for v in row] for row in i_com],
        "volume_signed": volume,
    }


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def _origin_of(elem) -> tuple:
    """`<origin xyz rpy>` -> internal SE3. Absent element = identity."""
    o = None if elem is None else elem.find("origin")
    if o is None:
        return ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
    return (tuple(xyz), rpy_to_quat(*rpy))


def _resolve_mesh(filename: str, urdf_path: Path, package_root: Path) -> Path:
    """Resolve a URDF mesh reference.

    `package://<pkg>/<rest>` resolves to `<ancestor>/<pkg>/<rest>` for the
    nearest ancestor of the URDF that actually contains a directory named
    `<pkg>` — each robot directory is named after its ROS package, so that
    ancestor is the package path. Searching for it beats deriving it from the
    URDF's depth: the Panda's URDF sits at `<pkg>/panda.urdf` and the H1's at
    `<pkg>/urdf/h1.urdf`, so any fixed number of `.parent`s is right for one and
    doubles the package name for the other
    (`h1_description/h1_description/meshes/...`).

    `package_root`, when given, is tried first — an explicit `--package-root`
    stays authoritative. Anything not a `package://` URL is relative to the
    URDF's own directory. No fallback on a miss: a dangling reference is a
    defect in the asset, not a spelling to tolerate.
    """
    if not filename.startswith("package://"):
        return (urdf_path.parent / filename).resolve()

    rel = filename[len("package://"):]
    pkg = rel.split("/", 1)[0]
    candidates = []
    if package_root is not None:
        candidates.append(package_root / rel)
    for anc in [urdf_path.parent, *urdf_path.parents]:
        if (anc / pkg).is_dir():
            candidates.append(anc / rel)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    # Return the first candidate so the error names a real attempted path.
    return (candidates[0] if candidates else package_root / rel).resolve()


def parse_urdf(urdf_path: Path, package_root: Path) -> dict:
    root = ET.parse(urdf_path).getroot()

    # Material table: URDF names a color once and references it by name after.
    colors: dict = {}
    for m in root.iter("material"):
        c = m.find("color")
        name = m.get("name")
        if name and c is not None and c.get("rgba"):
            rgba = [float(v) for v in c.get("rgba").split()]
            colors.setdefault(name, tuple(
                int(round(255 * max(0.0, min(1.0, v)))) for v in rgba[:3]))

    links: dict = {}
    for lk in root.findall("link"):
        name = lk.get("name")
        entry: dict = {"visual": [], "collision": [], "color": None,
                       "urdf_inertial": None}
        for tag in ("visual", "collision"):
            for g in lk.findall(tag):
                mesh = g.find("geometry/mesh")
                if mesh is None or not mesh.get("filename"):
                    # A primitive <box>/<cylinder>/<sphere>. Kept as geometry,
                    # not skipped: the H1 and Go2 declare all of their collision
                    # volume this way, and treating it as absent would silently
                    # promote the visual mesh into the collision role.
                    geom = g.find("geometry")
                    kids = [k for k in geom] if geom is not None else []
                    if len(kids) == 1 and kids[0].tag in (
                            "box", "cylinder", "sphere"):
                        entry[tag].append({
                            "primitive": kids[0].tag,
                            "attrib": dict(kids[0].attrib),
                            "origin": _origin_of(g),
                            "scale": [1.0, 1.0, 1.0],
                        })
                    else:
                        entry.setdefault("skipped", []).append(
                            {"tag": tag, "geometry": [k.tag for k in kids]})
                else:
                    entry[tag].append({
                        "file": _resolve_mesh(mesh.get("filename"), urdf_path,
                                              package_root),
                        "origin": _origin_of(g),
                        "scale": [float(v) for v in
                                  (mesh.get("scale") or "1 1 1").split()],
                    })
                mat = g.find("material")
                if mat is not None and entry["color"] is None:
                    entry["color"] = colors.get(mat.get("name"))
        ine = lk.find("inertial")
        if ine is not None:
            it = ine.find("inertia")
            mass = ine.find("mass")
            entry["urdf_inertial"] = {
                "origin": _origin_of(ine),
                "mass": float(mass.get("value")) if mass is not None else None,
                "inertia": {k: float(it.get(k, 0.0)) for k in
                            ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                if it is not None else None,
            }
        links[name] = entry

    joints = []
    for j in root.findall("joint"):
        axis_el = j.find("axis")
        lim = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "origin": _origin_of(j),
            "axis": ([float(v) for v in axis_el.get("xyz").split()]
                     if axis_el is not None and axis_el.get("xyz") else None),
            "limit": ({"lower": float(lim.get("lower", 0.0)),
                       "upper": float(lim.get("upper", 0.0))}
                      if lim is not None else None),
        })
    return {"name": root.get("name"), "links": links, "joints": joints}


# ---------------------------------------------------------------------------
# Tree surgery: geometry-less links
# ---------------------------------------------------------------------------

def prune_frame_links(urdf: dict) -> dict:
    """Drop links that carry no geometry, composing their transforms away.

    URDF uses geometry-less links as pure frames — the Panda has
    `panda_link8` (a flange offset) and `panda_grasptarget` (a TCP). A DyCraft
    part is a rigid *body*: it has a mesh, an AABB and an anchor vocabulary
    derived from it, so a massless frame has no honest representation as one and
    synthesising a tiny box would put fake geometry into the collision phase.

    Two cases, and only two are safe:

    * **Interior frame** reached by a `fixed` joint: splice it out and compose
      `origin(parent->frame) o origin(frame->child)` into each child's joint.
      Exact — a fixed joint is a constant transform.
    * **Leaf frame** reached by a `fixed` joint: keep it as a named semantic
      anchor on its parent part. That is what `add_anchor` is for, and it is
      where a grasp frame belongs.

    A frame reached by a *movable* joint is refused: splicing it would delete a
    degree of freedom, and no DyCraft part could carry it.
    """
    links = urdf["links"]
    joints = list(urdf["joints"])
    has_geom = {n: bool(v["visual"] or v["collision"])
                for n, v in links.items()}
    incoming = {j["child"]: j for j in joints}
    children_of = {}
    for j in joints:
        children_of.setdefault(j["parent"], []).append(j)

    anchors: dict = {}          # part name -> [anchor record]
    dropped: list = []

    for name in list(links):
        if has_geom[name]:
            continue
        inc = incoming.get(name)
        if inc is None:
            raise ImportError_(
                f"root link {name!r} carries no geometry; a DyCraft object's "
                "root part must be a real body")
        if inc["type"] != "fixed":
            raise ImportError_(
                f"link {name!r} carries no geometry but its incoming joint "
                f"{inc['name']!r} is {inc['type']!r}, not fixed; splicing it "
                "would delete a degree of freedom")
        kids = children_of.get(name, [])
        if kids:
            for k in kids:                          # interior frame: splice
                k["origin"] = _rot.se3_compose(inc["origin"], k["origin"])
                k["parent"] = inc["parent"]
                children_of.setdefault(inc["parent"], []).append(k)
            dropped.append({"link": name, "kind": "spliced",
                            "via_joint": inc["name"],
                            "rehomed": [k["name"] for k in kids]})
        else:                                        # leaf frame: keep as anchor
            t, q = inc["origin"]
            record = {
                "name": name,
                "translation": [float(v) for v in t],
                "rotation": [float(v) for v in _rot.sign_canonicalize_quat(
                    _rot.normalize_quat(q))],
                "kind": "imported_frame",
                "source": f"urdf:{name}",
            }
            anchors.setdefault(inc["parent"], []).append(record)
            # The pose travels into `import.json` too, not just the IR. The
            # recipe is the readable description of a frozen asset, and a frame
            # named there without its pose (the Panda's `panda_grasptarget`, its
            # shipped TCP 0.105 m along the hand's +Z) would leave an agent to
            # re-derive a number that is already on disk.
            dropped.append({"link": name, "kind": "anchor",
                            "on_part": inc["parent"], "via_joint": inc["name"],
                            "anchor": f"{inc['parent']}.{name}",
                            "translation": record["translation"],
                            "rotation": record["rotation"]})
        joints.remove(inc)
        del links[name]

    urdf["joints"] = joints
    urdf["anchors"] = anchors
    urdf["dropped_links"] = dropped
    return urdf


def find_root(urdf: dict) -> str:
    children = {j["child"] for j in urdf["joints"]}
    roots = [n for n in urdf["links"] if n not in children]
    if len(roots) != 1:
        raise ImportError_(f"expected exactly one root link, got {roots}")
    return roots[0]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _entry_label(e) -> str:
    """How a geometry entry is named in `import.json`.

    A mesh entry is its resolved path; a primitive entry is the URDF element it
    came from, so the recipe records `<cylinder radius=0.05 length=0.4>` rather
    than an empty string a reader would take for a missing file.
    """
    if "primitive" in e:
        attrs = " ".join(f"{k}={v}" for k, v in sorted(e["attrib"].items()))
        return f"<{e['primitive']} {attrs}>"
    return str(e["file"])


def _baked_geometry(entries, label: str):
    """Load, transform by each entry's `<origin>`/`scale`, merge, weld.

    Baking the per-`<visual>` origin into the vertices is what lets one DyCraft
    part hold a link with several visual elements — the Panda's right finger is
    the same OBJ as the left, rotated 180 degrees about Z, so the two parts
    legitimately end up with different content digests. `<mesh scale>` is baked
    here too: the imported tag carries no `scale` param, on the same argument
    `_validate_solid_params` makes for a solid, and with the extra teeth that
    the narrow phase's BVH cache keys off the asset digest alone and would
    silently share one model across two scales.
    """
    pieces = []
    textures: list = []
    notes: list = []
    for e in entries:
        if "primitive" in e:
            v, t = primitive_mesh(e["primitive"], e["attrib"])
            uv = None
        else:
            v, t, uv = load_mesh(e["file"], with_uvs=True)
            if Path(e["file"]).suffix.lower() == ".obj":
                tex, note = read_obj_texture(Path(e["file"]))
                if tex is not None:
                    textures.append(tex)
                elif note:
                    notes.append(note)
        s = np.asarray(e["scale"], dtype=np.float64)
        if not np.allclose(s, 1.0):
            v = v * s
        pieces.append((transform_verts(v, se3_to_matrix(e["origin"])), t, uv))
    verts, tris, uvs = merge(pieces)
    if len(tris) == 0:
        raise ImportError_(f"{label}: merged geometry has no triangles")
    verts, tris, keep = weld(verts, tris, return_keep=True)
    # Per-corner, so the weld's triangle mask is the whole of what they need —
    # welding renumbers `v` and leaves `vt` alone.
    uvs = uvs[keep] if uvs is not None and len(uvs) == len(keep) else None

    # One image for the merged mesh, or none. Several entries agreeing on one
    # image is the common case (Franka's 22 MTLs all name `colors.png`), and
    # disagreeing entries are refused for the same reason a multi-atlas glTF is.
    texture, note = None, None
    if textures:
        digests = {digest_bytes(t["base_color"]) for t in textures}
        if len(digests) == 1 and len(textures) == len(
                [e for e in entries if "primitive" not in e]):
            texture = textures[0]
        elif len(digests) > 1:
            note = (f"{label}: {len(digests)} different map_Kd images across "
                    f"{len(entries)} visual elements, and the merged mesh has "
                    f"one UV set")
        else:
            note = (f"{label}: only {len(textures)} of "
                    f"{len([e for e in entries if 'primitive' not in e])} mesh "
                    f"elements carry a map_Kd")
    if texture is not None and uvs is None:
        texture, note = None, (f"{label}: has a map_Kd but no usable texture "
                               f"coordinates after merging")
    return verts, tris, uvs, texture, note or (notes[0] if notes else None)


def parse_default_pose(spec: str) -> dict:
    """`"j2=-45,j4=-135,finger=0.02"` -> `{name: value}` in SURFACE units.

    Degrees for revolute, meters for prismatic — the SDK's convention at every
    authoring surface (`attach_revolute` takes degrees and converts), so the
    flag reads the way the rest of DyCraft does rather than the way the URDF
    does.
    """
    out: dict = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ImportError_(f"--default-pose entry {item!r} is not name=value")
        k, _, v = item.partition("=")
        out[k.strip()] = float(v)
    return out


#: `--self-collision-allowances` choices. See `_structural_allowances`.
SELF_COLLISION_ALLOWANCE_POLICIES = ("none", "rest-pose")


def _structural_allowances(ir_no_id: dict, out_dir: Path, policy: str) -> list:
    """Declare the overlaps that are facts about how the asset is built.

    A URDF's links are shells around a mechanism, and at a bearing a child's
    shell routinely lies inside its parent's. That is structure, not a defect,
    and every robotics stack says so somewhere: MoveIt writes
    `<disable_collisions reason="Adjacent"/>` and cuRobo's shipped configs
    ignore adjacent pairs. Declared in the bundle, the fact is written down once
    for every reader instead of being recomputed by whoever needs it.

    **Measured, not assumed.** The rule is "overlaps at the asset's rest pose",
    not "is adjacent in the joint tree". Measuring is both tighter and wider
    than adjacency, and on the G1 it is both at once: of 37 adjacent pairs only
    18 actually overlap at rest, so an adjacency rule would blanket-allow 19
    pairs that are clear and ought to stay checked — while the measurement also
    catches `torso_link`/`waist_yaw_link`, which nest through the waist stack
    two joints apart and are structural all the same.

    **Blanket, not scoped.** The allowance schema can scope to a joint and a
    `q_range`, but scoping an adjacent pair to its own connecting joint over
    that joint's full range is exactly equivalent to a blanket record and costs
    more bytes to say. A tighter range would need evidence a rest-pose
    measurement cannot supply, so the honest record is the unconditional one
    with its measured depth in the reason.

    **What this does not mask.** Only the pairs that overlap at rest. On the G1
    that leaves 157 of 190 swept rows still reported, including every
    arm-through-head case — those are two-hop or distant pairs that a hardware
    limit permits and no structure requires.

    Returns `[]` for `policy="none"`, and also whenever the exact adjudicator
    is unavailable. The fallback there would be SAT's deliberately
    over-reporting pair set, which would declare MORE pairs intended and hand
    back a quieter, less trustworthy asset — the one outcome worth refusing.
    """
    if policy == "none":
        return []
    from ...sdk._probes import _self_penetrations_internal
    try:
        hits = _self_penetrations_internal(ir_no_id, {}, exact=True,
                                           asset_base_dir=out_dir, cache={})
    except Exception as e:                              # noqa: BLE001
        print(f"[import_urdf] structural allowances skipped: the exact "
              f"adjudicator is unavailable ({type(e).__name__}: {e}). The "
              f"asset is imported without them; re-run where CAD is available "
              f"to declare them.", file=sys.stderr)
        return []
    out = []
    for h in sorted(hits, key=lambda r: (r["a"], r["b"])):
        if h.get("deformed"):
            continue
        depth_mm = float(h["depth_m"]) * 1000.0
        out.append({
            "a": h["a"], "b": h["b"],
            "reason": (f"Structural: {h['a']} and {h['b']} overlap by "
                       f"{depth_mm:.1f} mm at the asset's rest pose. Measured "
                       f"by the exact adjudicator at URDF import, not "
                       f"declared by an author."),
        })
    return out


def convert(urdf_path: Path, out_dir: Path, *, object_name: str,
            package_root: Path, mass_from: str = "visual",
            collision_hull: str = "when-open",
            self_collision_allowances: str = "rest-pose",
            default_pose: dict | None = None,
            continuous_range_deg: float | None = None) -> dict:
    urdf = prune_frame_links(parse_urdf(urdf_path, package_root))
    root_link = find_root(urdf)

    # Part order: root first, then BFS. FK walks the tree from the root, and a
    # deterministic order keeps the content hash stable across runs.
    order = [root_link]
    kids_of: dict = {}
    for j in urdf["joints"]:
        kids_of.setdefault(j["parent"], []).append(j["child"])
    frontier = [root_link]
    while frontier:
        nxt = []
        for p in frontier:
            for c in sorted(kids_of.get(p, [])):
                if c not in order:
                    order.append(c)
                    nxt.append(c)
        frontier = nxt
    if len(order) != len(urdf["links"]):
        raise ImportError_(
            f"links not reachable from root {root_link!r}: "
            f"{sorted(set(urdf['links']) - set(order))}")

    (out_dir / MESHES_SUBDIR).mkdir(parents=True, exist_ok=True)
    (out_dir / COLLISION_SUBDIR).mkdir(parents=True, exist_ok=True)

    parts = []
    report_parts = []
    for pid, link_name in enumerate(order):
        link = urdf["links"][link_name]
        vis_entries = link["visual"] or link["collision"]
        # A link with no <collision> at all falls back to its visual geometry,
        # which is not a collision mesh: the G1's `left_rubber_hand` visual is
        # 45,748 triangles, and putting that in the narrow phase costs more than
        # the rest of the robot together. The fallback is hulled below —
        # `collision_borrowed` is what says so — so a missing <collision> yields
        # a cheap conservative volume rather than an expensive exact one.
        collision_borrowed = not link["collision"]
        col_entries = link["collision"] or link["visual"]
        vv, vt, v_uvs, v_tex, v_note = _baked_geometry(
            vis_entries, f"{link_name} visual")
        # The collision mesh's UVs and texture are dropped on purpose: collision
        # asks about containment, which has nothing to sample, and `--collision
        # auto` may replace the geometry with a convex hull outright.
        cv, ct = _baked_geometry(col_entries, f"{link_name} collision")[:2]

        wt_vis = watertight(vv, vt)
        wt_col = watertight(cv, ct)
        vis_ok = wt_vis["closed"] and wt_vis["oriented"]
        col_ok = wt_col["closed"] and wt_col["oriented"]

        # COLLISION must be a closed surface. The narrow phase decides
        # penetration with erosion and interior probes, and "inside" is not
        # defined for an open shell — several of the Panda's links are tubes
        # whose ends upstream left unclosed (link6's collision mesh has 586
        # boundary edges), which pybullet gets away with because it convex-
        # decomposes before it ever queries. Hulling here is the same move, made
        # explicit and recorded: it can only ever over-report contact, which is
        # the safe direction and matches the SAT phase's own bias.
        hulled = False
        if collision_hull == "always" or collision_borrowed or (
                collision_hull == "when-open" and not col_ok):
            cv, ct = convex_hull(cv)
            wt_col = watertight(cv, ct)
            col_ok = wt_col["closed"] and wt_col["oriented"]
            hulled = True
            if not col_ok:
                raise ImportError_(
                    f"{link_name}: convex hull of the collision mesh is still "
                    f"not a closed oriented surface ({wt_col})")

        # VISUAL stays verbatim — an open shell renders correctly, and the
        # silhouette is the whole reason to keep a 12k-triangle mesh at all.
        mp_vis = mass_properties(vv, vt) if vis_ok else None
        mp_col = mass_properties(cv, ct) if col_ok else None

        # `visual` deliberately does NOT fall back to `collision`. The rule is
        # that the visual mesh is the part — it is what renders, what a reviewer
        # judges, and what the AABB and anchor vocabulary come from — so
        # borrowing the proxy's volume would make one `shape` table describe two
        # bodies per part, recorded only here in the report. An open shell falls
        # through to the hull of *itself* below: an over-estimate of the right
        # body rather than a measurement of the wrong one. On the G1 that is 18
        # of 38 parts measured directly and 20 hulled, against a foot whose
        # proxy is a 10 mm sole plate holding 0.7% of the visible volume.
        order_pref = (("visual",) if mass_from == "visual"
                      else ("collision", "visual"))
        chosen, mp = None, None
        for cand in order_pref:
            cand_mp = mp_vis if cand == "visual" else mp_col
            if cand_mp is not None and cand_mp["volume"] > 0.0:
                chosen, mp = cand + ("_hull" if cand == "collision" and hulled
                                     else ""), cand_mp
                break
        if mp is None:
            # Last resort: the hull of the visual mesh. Always closed, so the
            # volume exists; over-estimated for a concave link, and named as a
            # hull in the report so nobody reads it as measured.
            hv, ht = convex_hull(vv)
            mp = mass_properties(hv, ht)
            chosen = "visual_hull"
        if mp["volume"] <= 0.0:
            raise ImportError_(
                f"{link_name}: no positive volume from any candidate mesh "
                f"(visual={wt_vis}, collision={wt_col})")

        tex_rel = stage_texture(out_dir, v_tex)
        vis_bytes = write_obj(vv, vt, f"{link_name}:visual",
                              v_uvs if tex_rel else None)
        col_bytes = write_obj(cv, ct, f"{link_name}:collision")
        vis_digest = digest_bytes(vis_bytes)
        col_digest = digest_bytes(col_bytes)
        vis_rel = f"{MESHES_SUBDIR}/{vis_digest}.obj"
        col_rel = f"{COLLISION_SUBDIR}/{col_digest}.obj"
        (out_dir / vis_rel).write_bytes(vis_bytes)
        (out_dir / col_rel).write_bytes(col_bytes)

        # The AABB is the VISUAL mesh's: it is what the renderer draws and what
        # the anchor vocabulary is derived from. Using the coarse collision hull
        # would make anchors land off the visible surface — a landmark wants to
        # be ON the surface, and error in any direction is wrong.
        aabb_min = [float(v) for v in vv.min(axis=0)]
        aabb_max = [float(v) for v in vv.max(axis=0)]
        # A *bound* wants the opposite: to lie outside everything the narrow
        # phase can measure, where only outward error is safe. One box cannot be
        # both, and the narrow phase measures the collision mesh — which can
        # poke outside this one by millimeters (on the Panda most parts do; on
        # the G1, whose proxies are hulls of the visual, none do). So a pair
        # whose boxes are a few mm apart could be touching and never be
        # examined, breaking the prune's stated contract
        # (`_probes._gap_candidates`).
        #
        # Recorded as a scalar rather than fixed by widening the box, because
        # widening it moves every anchor. `_scene_kernel` adds it to the same
        # inflation channel deformation uses — which lives in the bound
        # producers, never in `local_bbox`, so the anchor path provably cannot
        # pick it up (`_object_kernel._inflate_local_box`).
        escape = 0.0
        for k in range(3):
            escape = max(escape,
                         float(aabb_min[k] - cv[:, k].min()),
                         float(cv[:, k].max() - aabb_max[k]))

        part = {
            "id": pid,
            "name": link_name,
            "primitive": PRIMITIVE_IMPORTED,
            "params": {
                "mesh": vis_rel,
                "collision": col_rel,
                "digest": vis_digest,
                "collision_digest": col_digest,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
                "collide_escape_m": escape,
                "shape": {
                    "volume": mp["volume"],
                    "center": mp["center"],
                    "inertia_diag": mp["inertia_diag"],
                    "n_vertices": int(len(vv)),
                    "n_triangles": int(len(vt)),
                    "closed": bool(wt_vis["closed"] and wt_vis["oriented"]),
                },
                # What the URDF declared, beside what the mesh measures. Same
                # frame: `<inertial><origin>` is relative to the link, and each
                # `<visual><origin>` is baked into the staged vertices above,
                # so this `com` and `shape.center` describe the same space.
                **({"inertial": {
                    "mass": link["urdf_inertial"]["mass"],
                    "com": [float(v) for v in link["urdf_inertial"]["origin"][0]],
                    "source": "urdf",
                }} if (link.get("urdf_inertial")
                       and link["urdf_inertial"].get("mass")
                       and link["urdf_inertial"]["mass"] > 0) else {}),
            },
            "material": {
                "color": list(link["color"] or DEFAULT_COLOR),
                "metallic": 0.0, "roughness": 0.5, "alpha": 1.0,
            },
        }
        if tex_rel:
            # `color` stays as the URDF declared it and becomes a fallback
            # rather than a tint — see `sdk/_material.Texture`. A consumer that
            # can sample the image uses a white factor; one that cannot (URDF
            # export, a flat-shaded viewer) uses the color.
            part["material"]["texture"] = base_color_texture_ir(tex_rel)
        anchors = urdf["anchors"].get(link_name)
        if anchors:
            part["anchors"] = sorted(anchors, key=lambda a: a["name"])
        parts.append(part)

        report_parts.append({
            "link": link_name, "part_id": pid,
            "texture": ({"file": tex_rel,
                         "uv_corners": int(0 if v_uvs is None else len(v_uvs))}
                        if tex_rel else {"staged": False, "why": v_note}),
            "visual": {"file": vis_rel, "n_vertices": int(len(vv)),
                       "n_triangles": int(len(vt)), **wt_vis,
                       "sources": [_entry_label(e) for e in vis_entries]},
            "collision": {"file": col_rel, "n_vertices": int(len(cv)),
                          "n_triangles": int(len(ct)), **wt_col,
                          "convex_hulled": hulled,
                          "borrowed_from_visual": collision_borrowed,
                          "sources": [_entry_label(e) for e in col_entries]},
            "mass_from": chosen,
            "volume_m3": mp["volume"],
            "volume_visual_m3": mp_vis["volume"] if mp_vis else None,
            "volume_collision_m3": mp_col["volume"] if mp_col else None,
            "center": mp["center"],
            "inertia_diag_unit_density": mp["inertia_diag"],
            "urdf_inertial": _jsonable(link["urdf_inertial"]),
            "skipped_geometry": link.get("skipped", []),
            "aabb_min": aabb_min, "aabb_max": aabb_max,
        })

    name_to_id = {p["name"]: p["id"] for p in parts}
    joints = []
    for jid, j in enumerate(sorted(urdf["joints"],
                                   key=lambda x: (order.index(x["child"])))):
        if j["type"] not in ("fixed", "revolute", "prismatic", "continuous"):
            raise ImportError_(
                f"joint {j['name']!r} has type {j['type']!r}, which no DyCraft "
                "joint can express (fixed / revolute / prismatic only)")
        record = {
            "id": jid,
            "name": j["name"],
            "type": "revolute" if j["type"] == "continuous" else j["type"],
            "parent": j["parent"],
            "child": j["child"],
            # parent_anchor = the URDF <origin>; child_anchor = identity.
            # FK computes rest = pa o inv(ca) and pivots at pa.translation with
            # the axis in pa-local, which is URDF's semantics term for term.
            "parent_anchor": se3_to_ir(j["origin"]),
            "child_anchor": dict(IDENTITY_IR_SE3),
        }
        if record["type"] != "fixed":
            axis = j["axis"] or [0.0, 0.0, 1.0]
            n = math.sqrt(sum(a * a for a in axis))
            if n <= 0.0:
                raise ImportError_(f"joint {j['name']!r} has a zero axis")
            record["axis"] = [a / n for a in axis]
            lim = j["limit"]
            if (lim is None or lim["upper"] <= lim["lower"]) \
                    and j["type"] == "continuous":
                # URDF `continuous` means unbounded rotation, which no DyCraft
                # revolute can express. A range has to be chosen, and choosing
                # one silently would fabricate a limit the asset never stated —
                # so it must be asked for explicitly.
                if continuous_range_deg is None:
                    raise ImportError_(
                        f"joint {j['name']!r} is `continuous` (unbounded) and "
                        "DyCraft has no such joint type. Pass "
                        "--continuous-range <deg> to bound every continuous "
                        "joint symmetrically, and record in your notes that the "
                        "bound is yours, not the asset's.")
                half = abs(float(continuous_range_deg)) / 2.0
                lim = {"lower": _rot.deg_to_rad(-half),
                       "upper": _rot.deg_to_rad(half)}
            if lim is None or lim["upper"] <= lim["lower"]:
                raise ImportError_(
                    f"joint {j['name']!r} ({j['type']}) has no usable limit "
                    f"{lim!r}")
            # IR units: radians for revolute, meters for prismatic — the same
            # values URDF already carries, so no conversion. The DEGREES in
            # `attach_revolute` are an authoring-surface convention only.
            record["limits"] = [lim["lower"], lim["upper"]]
            # `default` is the object's committed REST pose — what the renderer
            # draws, what a catalog thumbnail shows, and what the scene layer
            # places unless it poses the instance. The URDF has no notion of
            # one, and its zero is often a bad choice (the Panda's zero sits
            # joint4 exactly on its limit with the arm straight up), so it is
            # the fallback, not the answer.
            want = (default_pose or {}).get(j["name"])
            if want is None:
                q = min(max(0.0, lim["lower"]), lim["upper"])
            else:
                q = (_rot.deg_to_rad(want) if record["type"] == "revolute"
                     else float(want))
                if not (lim["lower"] - 1e-12 <= q <= lim["upper"] + 1e-12):
                    raise ImportError_(
                        f"--default-pose {j['name']}={want} is outside the "
                        f"URDF limit [{lim['lower']}, {lim['upper']}] "
                        f"({'radians' if record['type'] == 'revolute' else 'm'})")
            record["default"] = float(q)
        joints.append(record)

    ir_no_id = {
        "schema": OBJECT_SCHEMA_VERSION,
        "name": object_name,
        "root": root_link,
        "parts": parts,
        "joints": joints,
        "assertions": [],
        # Filled below, then the id is taken over the finished content: an
        # allowance is inside the content hash, so declaring one re-identifies
        # the asset. That is correct — a bundle that allows an overlap is not
        # the same input as one that does not — but it does mean re-importing
        # an existing asset under a new policy yields a new `ir_id`.
        "collision_allowances": [],
    }
    ir_no_id["collision_allowances"] = _structural_allowances(
        ir_no_id, out_dir, self_collision_allowances)
    ir = insert_id(ir_no_id)

    (out_dir / "ir.json").write_text(json.dumps(ir, indent=2))
    recipe = {
        "converter": "dycraft.agent.assets.import_urdf",
        # 2: emits rest-pose structural `collision_allowances`. A bundle
        # converted at version 1 carries none and hashes differently.
        "converter_version": 2,
        "primitive": PRIMITIVE_IMPORTED,
        "ir_id": ir["id"],
        "object_name": object_name,
        "source_urdf": str(urdf_path),
        "urdf_robot_name": urdf["name"],
        "package_root": str(package_root),
        "axis_remap": "identity (URDF frame preserved verbatim)",
        "mass_from_preference": mass_from,
        "collision_hull_policy": collision_hull,
        "self_collision_allowance_policy": self_collision_allowances,
        "self_collision_allowances": ir["collision_allowances"],
        "continuous_range_deg": continuous_range_deg,
        "default_pose_surface_units": default_pose or {},
        "weld_quantum_m": WELD_QUANTUM_M,
        "root_part": root_link,
        "n_parts": len(parts), "n_joints": len(joints),
        "part_order": order,
        "dropped_links": urdf["dropped_links"],
        "parts": report_parts,
        # `limits` / `default` are IR-internal: radians for a revolute joint,
        # meters for a prismatic one, straight off the URDF so the bundle stays
        # checkable against upstream. Labeled rather than converted, because
        # `limits_unit` is one key and a second convention in the same file is
        # a trap: `default_pose_surface_units` above IS converted, so the two
        # sit side by side, and on the Panda that reads `panda_joint2: -45.0`
        # next to a `default` of -0.785398 for the same joint. `scene_joint_limits`
        # and `JointRangeNonTrivial` carry the same key.
        "joints": [{"name": j["name"], "type": j["type"],
                    "parent": j["parent"], "child": j["child"],
                    "limits": j.get("limits"), "axis": j.get("axis"),
                    "default": j.get("default"),
                    "limits_unit": (None if j["type"] == "fixed"
                                    else "m" if j["type"] == "prismatic"
                                    else "rad")} for j in joints],
    }
    (out_dir / "import.json").write_text(json.dumps(recipe, indent=2))
    return {"ir": ir, "recipe": recipe}


def upgrade_bundle(bundle: Path, out_dir: Path, *,
                   policy: str = "rest-pose") -> dict:
    """Re-emit an existing bundle with rest-pose structural allowances.

    The converter is the right place to declare these, and re-running it is the
    right way to get them — when the source URDF is still reachable. Often it is
    not: a bundle is the portable artifact and outlives the checkout that made
    it (the G1 here cites a `source_urdf` on a machine this one cannot see).
    This derives the same allowances from the bundle's own geometry, which is
    all `_structural_allowances` ever reads.

    The result is a NEW bundle with a NEW `ir_id`, never an edit in place.
    Allowances are inside the content hash, so a bundle that declares them is
    a different input from one that does not, and pretending otherwise would
    make `id` a lie — exactly the tampering `preload_asset._load_and_verify`
    exists to catch.
    """
    from ..layers.scene_context import IMPORT_RECIPE_FILE, referenced_object_assets

    bundle, out_dir = Path(bundle).resolve(), Path(out_dir).resolve()
    ir = json.loads((bundle / "ir.json").read_text())
    if ir.get("collision_allowances"):
        raise ImportError_(
            f"{bundle} already declares "
            f"{len(ir['collision_allowances'])} collision allowances; refusing "
            "to recompute them over an author's or a converter's existing set")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Assets first: the allowance measurement reads the meshes, and it must read
    # them where the new bundle will keep them.
    for rel in referenced_object_assets(ir):
        src, dst = bundle / rel, out_dir / rel
        if not src.is_file():
            raise ImportError_(f"{bundle}: missing asset {rel.as_posix()}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    ir_no_id = {k: v for k, v in ir.items() if k != "id"}
    ir_no_id["collision_allowances"] = _structural_allowances(
        ir_no_id, out_dir, policy)
    new_ir = insert_id(ir_no_id)
    (out_dir / "ir.json").write_text(json.dumps(new_ir, indent=2))

    recipe = {}
    prev = bundle / IMPORT_RECIPE_FILE
    if prev.is_file():
        try:
            recipe = json.loads(prev.read_text())
        except (OSError, ValueError):
            recipe = {}
    recipe.update({
        "converter_version": 2,
        "ir_id": new_ir["id"],
        "upgraded_from": {"bundle": str(bundle), "ir_id": ir.get("id")},
        "self_collision_allowance_policy": policy,
        "self_collision_allowances": new_ir["collision_allowances"],
    })
    (out_dir / IMPORT_RECIPE_FILE).write_text(json.dumps(recipe, indent=2))
    return {"ir": new_ir, "recipe": recipe}


def _jsonable(obj):
    """Recursively make SE3 tuples / numpy scalars JSON-safe."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dycraft.agent.assets.import_urdf",
        description="Convert a third-party URDF into a DyCraft object bundle "
                    "(offline; no agent).")
    ap.add_argument("--urdf", required=True,
                    help="the robot's URDF file")
    ap.add_argument("--out", required=True,
                    help="directory to write the bundle into, e.g. "
                         "assets/imported/<name>")
    ap.add_argument("--name", required=True,
                    help="the object's name, written into its IR")
    ap.add_argument("--package-root", default=None,
                    help="root for package:// references "
                         "(default: the URDF's parent's parent)")
    ap.add_argument("--mass-from", choices=("collision", "visual"),
                    default="visual",
                    help="which mesh to integrate mass properties from. "
                         "`visual` is the part (what renders, what the AABB and "
                         "anchors come from) and hulls itself when it is an open "
                         "shell; `collision` measures the shipped contact proxy "
                         "and falls back to the visual mesh")
    ap.add_argument("--collision-hull",
                    choices=("never", "when-open", "always"),
                    default="when-open",
                    help="replace a link's collision mesh with its convex "
                         "hull; 'when-open' only for meshes that are not "
                         "closed surfaces (default)")
    ap.add_argument("--self-collision-allowances",
                    choices=SELF_COLLISION_ALLOWANCE_POLICIES,
                    default="rest-pose",
                    help="declare the link pairs that overlap at the asset's "
                         "rest pose as intended structure, so the verifiers "
                         "stop re-deciding a fact about the mechanism. "
                         "'none' declares no allowances. "
                         "Allowances are inside the content hash, so this "
                         "changes the object's id.")
    ap.add_argument("--continuous-range", type=float, default=None,
                    metavar="DEG",
                    help="bound every URDF `continuous` joint to +-DEG/2. "
                         "Required when the URDF has any, because DyCraft has "
                         "no unbounded joint and inventing a range silently "
                         "would attribute it to the asset.")
    ap.add_argument("--default-pose", default="",
                    help="committed rest pose, 'joint=value,...' — DEGREES for "
                         "revolute, meters for prismatic (the SDK's surface "
                         "convention). Unlisted joints fall back to the URDF "
                         "zero clamped into its limit.")
    args = ap.parse_args(argv)

    urdf_path = Path(args.urdf).resolve()
    out_dir = Path(args.out).resolve()
    pkg_root = (Path(args.package_root).resolve() if args.package_root
                else urdf_path.parent.parent)

    try:
        result = convert(urdf_path, out_dir, object_name=args.name,
                         package_root=pkg_root, mass_from=args.mass_from,
                         collision_hull=args.collision_hull,
                         self_collision_allowances=args.self_collision_allowances,
                         default_pose=parse_default_pose(args.default_pose),
                         continuous_range_deg=args.continuous_range)
    except ImportError_ as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    ir, rec = result["ir"], result["recipe"]
    print(f"[import_urdf] {urdf_path}")
    print(f"[import_urdf] -> {out_dir}")
    print(f"[import_urdf] object {ir['name']!r} id={ir['id']}  "
          f"root={ir['root']}  {len(ir['parts'])} parts / "
          f"{len(ir['joints'])} joints")
    for p in rec["parts"]:
        print(f"    {p['link']:<20s} vis {p['visual']['n_triangles']:>6d} tri  "
              f"col {p['collision']['n_triangles']:>5d} tri"
              f"{' [hull]' if p['collision']['convex_hulled'] else '       '}  "
              f"vol {p['volume_m3'] * 1e6:>9.1f} cm^3 (from {p['mass_from']})"
              f"{'' if p['visual']['closed'] else '  [visual open shell]'}")
    for d in rec["dropped_links"]:
        print(f"    dropped {d['link']:<20s} {d['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
