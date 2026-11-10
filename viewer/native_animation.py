"""Read-only native animation transport: one motion track, meshes by digest.

A native run's 3D playback is *evaluated* Blender geometry: `worker.py` walks the depsgraph once per frame and
writes what Blender computed, so the viewport shows modifiers, constraints, and drivers rather than a URDF's guess.

The geometry is nearly always the same in every frame: `worker.py` stores evaluated meshes content-addressed and
shares them across frames, so a handful serve hundreds of frames (121 frames of 3 objects can be 3 meshes). A
request per frame would re-send the same meshes while the user scrubs, and at 30 fps each request would cancel
the one before it. So the transport is split:

* **`track`** is the whole clip minus the geometry, one request of a few tens of KB: which objects exist, each
  one's color, which mesh it wears on each frame, and a `[frames][objects][12]` block of float32 matrices. Once it
  is loaded, advancing a frame is writing matrices, with no network in the loop.
* **`mesh_blob`** is one evaluated mesh by content digest, cacheable forever (`_IMMUTABLE_CACHE` in `app.py`)
  because the URL names the bytes.

**Integrity.** A frame record is named by position, so `checked_file` verifies it against the manifest's hash. A
mesh is named by the `sha256` of its own arrays, and `evaluation.read_local` already refuses one whose content
does not match, so a mesh is checked for *membership* in the manifest and for content by `read_local`, not hashed
twice.
"""
from __future__ import annotations

import gzip
import json
import struct
import threading
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import numpy as np

from ..animation.blender import SCHEMA
from ..animation.blender.evaluation import read_local, read_record
from ..animation.blender.bundle import file_hash, safe_path

#: Bumped when either blob's layout changes. The client refuses a version it
#: does not know rather than reading a renegotiated payload as geometry.
TRACK_VERSION = 1

#: Built tracks, keyed by `(resolved run dir, manifest id)`. Building one reads
#: and hashes every frame record in the bundle, which is milliseconds on a
#: 121-frame clip and approaching a second on a long one with a crowded scene —
#: and the viewer asks for it again every time the run is re-selected. The
#: manifest id is a content hash of the manifest, so a recompiled bundle gets a
#: new key rather than a stale answer.
_TRACK_CACHE: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_TRACK_CACHE_BYTES = 0
_TRACK_CACHE_LIMIT = 128 * 1024 * 1024
#: `track_gzip`'s results, by the same key: a few MB a film.
_TRACK_GZIP: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_TRACK_GZIP_LIMIT = 64 * 1024 * 1024
_TRACK_LOCK = threading.RLock()

#: Rows 0..2 of a 4x4 identity, flattened — one frame of "this object is not
#: here". Copied rather than shared, since a frame that does have the object
#: overwrites its own entry.
_IDENTITY_ROWS = [1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0]


#: Parsed manifests by `(path, mtime, size)`. Every mesh request needs the manifest, and parsing a long film's
#: `ir.json` on each one would be half its cost, and the server would fall behind playback. Callers only read it.
_MANIFESTS: dict[tuple, dict] = {}


def manifest(run_dir):
    path = Path(run_dir) / "ir.json"
    try:
        stat = path.stat()
    except OSError:
        raise ValueError("no native animation bundle") from None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    ir = _MANIFESTS.get(key)
    if ir is None:
        ir = json.loads(path.read_text())
        if len(_MANIFESTS) >= 16:
            _MANIFESTS.clear()
        _MANIFESTS[key] = ir
    if isinstance(ir, dict) and ir.get("schema") == SCHEMA:
        return ir
    raise ValueError("no native animation bundle")


def listed_file(run_dir, ir, relative):
    """A bundle member's path: inside the run, and named by the manifest.

    The membership test is what makes this safe to hand a digest that arrived
    in a URL — an npz the manifest does not list is not this run's, whatever it
    hashes to. It deliberately does NOT hash the file; see the module docstring
    for which check belongs where.
    """
    path = safe_path(run_dir, relative)
    if relative not in ir["files"]:
        raise ValueError(f"not a member of this animation bundle: {relative}")
    return path


def checked_file(run_dir, ir, relative):
    """A bundle member verified byte-for-byte against the manifest."""
    path = listed_file(run_dir, ir, relative)
    if file_hash(path) != ir["files"][relative]:
        raise ValueError(f"evaluated animation file hash mismatch: {relative}")
    return path


def timeline(run_dir, ir):
    from .animation import _events, clip_rel
    checked_file(run_dir, ir, ir["evaluation"])
    return {"fps": ir["fps"], "n_frames": ir["n_frames"], "duration": ir["duration"],
            "joints": [], "bodies": [], "rigs": [], "deform": [],
            "events": _events(Path(run_dir), ir), "provenance": [],
            "clip": clip_rel(Path(run_dir), ir),
            "bounds": None, "native": {"version": ir["id"]},
            "notes": []}


def _envelope(header: dict, buffers) -> bytes:
    """`uint32` length, a space-padded JSON header, then the arrays.

    The padding is not cosmetic: the client reads the arrays as typed-array
    *views* on the response buffer, and `Float32Array` refuses an offset that is
    not a multiple of four. Padding the header to a multiple of four keeps every
    array that follows aligned, since each one's element size divides four.
    """
    text = json.dumps(header, allow_nan=False).encode()
    text += b" " * (-len(text) % 4)
    return struct.pack("<I", len(text)) + text + b"".join(buffers)


def _compact(values):
    """One value when every frame agrees, the per-frame list when they do not.

    Which a clip does, overwhelmingly: an object wears one mesh and keeps one
    visibility for its whole life, and spelling that out per frame would be the
    bulk of the header on a long clip for no information at all.
    """
    first = values[0]
    return first if all(v == first for v in values) else values


def track(run_dir) -> bytes:
    """The whole clip's motion, with the geometry left out.

    Objects are the union over every frame, ordered by first appearance, so a
    scene that gains or loses an object mid-clip still has one stable column per
    object. An object absent from a frame gets mesh index `-1` and an identity
    matrix, which the client draws as hidden — the alternative, a ragged
    per-frame object list, would put the bookkeeping in the browser and cost the
    fixed-width matrix block that makes playback allocation-free.

    Room surfaces are dropped, as they were per frame: `scene_ir_to_urdf` omits
    room geometry, so the viewport has never drawn it and drawing it only in
    native playback would make the two paths disagree about what the scene is.
    """
    ir = manifest(run_dir)
    key = (str(Path(run_dir).resolve()), ir["id"])
    with _TRACK_LOCK:
        if key in _TRACK_CACHE:
            _TRACK_CACHE.move_to_end(key)
            return _TRACK_CACHE[key]

    prefix = PurePosixPath(ir["evaluation"]).parent
    n_frames = int(ir["n_frames"])
    blob = _binary_track(run_dir, ir, prefix, n_frames)
    if blob is None:
        blob = _record_track(run_dir, ir, prefix, n_frames)

    global _TRACK_CACHE_BYTES
    with _TRACK_LOCK:
        while _TRACK_CACHE and _TRACK_CACHE_BYTES + len(blob) > _TRACK_CACHE_LIMIT:
            _, evicted = _TRACK_CACHE.popitem(last=False)
            _TRACK_CACHE_BYTES -= len(evicted)
        if len(blob) <= _TRACK_CACHE_LIMIT:
            _TRACK_CACHE[key] = blob
            _TRACK_CACHE_BYTES += len(blob)
    return blob


def track_gzip(run_dir) -> bytes:
    """`track`, gzipped once per film and kept.

    Most of a track is the still objects' matrices, the same every frame, so it compresses about 25 times (a long
    film's: 63 MB to 2.5 MB). It is the largest single download of a film's first open, which over a slow link is most
    of the wait. Compressed here once, at level 6, rather than by the middleware on every request at the level 1 it
    can afford.
    """
    ir = manifest(run_dir)
    key = (str(Path(run_dir).resolve()), ir["id"])
    with _TRACK_LOCK:
        if key in _TRACK_GZIP:
            _TRACK_GZIP.move_to_end(key)
            return _TRACK_GZIP[key]
    packed = gzip.compress(track(run_dir), compresslevel=6, mtime=0)
    # A live run can recompile between the two reads; then this is the newer track, and not kept under the older id.
    if manifest(run_dir)["id"] != ir["id"]:
        return packed
    with _TRACK_LOCK:
        _TRACK_GZIP[key] = packed
        while sum(len(v) for v in _TRACK_GZIP.values()) > _TRACK_GZIP_LIMIT and len(_TRACK_GZIP) > 1:
            _TRACK_GZIP.popitem(last=False)
    return packed


def _binary_track(run_dir, ir, prefix, n_frames):
    """`track`'s bytes read straight from binary frame records, or None for a bundle stored otherwise.

    Decoding every frame into Python dicts and lists would make a long film's first view take minutes: most of a
    record is transforms the track does not use, and each float would become a Python object. Here each record's fixed
    blocks are viewed as arrays, each table is verified and parsed once, and the placements land in one array, so a
    2,222-frame film with 400 bodies builds in seconds. Every record and table is still checked against the manifest
    before it is read. Fields a record keeps beside its fixed layout never name, place, color, mesh, or hide a body,
    so the result is byte-identical to reading the records whole.
    """
    from ..animation.blender.evaluation import (_FLAG_BODY_MATRICES, _RECORD_HEADER, RECORD_MAGIC,
                                                RECORD_VERSION, _table_id)
    if any(f"{prefix}/{frame:06d}.bin" not in ir["files"] for frame in range(n_frames)):
        return None
    tables: dict[str, tuple] = {}
    columns: dict[str, int] = {}
    rows: list[dict] = []
    meshes: dict[bytes, int] = {}
    frames = []
    for frame in range(n_frames):
        buffer = checked_file(run_dir, ir, f"{prefix}/{frame:06d}.bin").read_bytes()
        magic, version, flags, n_objects, n_bodies, n_joints, raw_id = _RECORD_HEADER.unpack_from(buffer, 0)
        if magic != RECORD_MAGIC or version != RECORD_VERSION:
            raise ValueError("not a DyCraft evaluated frame record")
        table_id = raw_id.hex()
        if table_id not in tables:
            table = json.loads(checked_file(run_dir, ir, f"{prefix}/tables/{table_id}.json").read_text())
            if _table_id(table) != table_id:
                raise ValueError("evaluated frame record does not match its table")
            slots = {name: i for i, name in enumerate(table["objects"])}
            tables[table_id] = (table, np.asarray([slots.get(name, -1) for name, _, _ in table["bodies"]]))
        table, body_slots = tables[table_id]
        if (len(table["objects"]) != n_objects or len(table["bodies"]) != n_bodies
                or len(table["joints"]) != n_joints):
            raise ValueError("evaluated frame record disagrees with its table")
        offset = _RECORD_HEADER.size
        transforms = np.frombuffer(buffer, "<f4", n_objects * 16, offset).reshape(n_objects, 4, 4)
        offset += n_objects * 64
        if flags & _FLAG_BODY_MATRICES:
            matrices = np.frombuffer(buffer, "<f4", n_bodies * 16, offset).reshape(n_bodies, 4, 4)
            offset += n_bodies * 64
        else:
            matrices = transforms[body_slots]
        digests = np.frombuffer(buffer, np.uint8, n_bodies * 32, offset).reshape(n_bodies, 32)
        offset += n_bodies * 32
        colors = np.frombuffer(buffer, "<f4", n_bodies * 4, offset).reshape(n_bodies, 4)
        offset += n_bodies * 16 + n_joints * 4
        hidden = np.frombuffer(buffer, np.uint8, n_bodies, offset)
        cols, picks, mesh_index = [], [], []
        for i, (name, identity, room) in enumerate(table["bodies"]):
            if room:
                continue
            column = columns.get(name)
            if column is None:
                column = columns[name] = len(rows)
                rows.append({"name": name, "identity": identity,
                             "color": [float(c) for c in colors[i].tolist()]})
            digest = digests[i].tobytes()
            if digest not in meshes:
                meshes[digest] = len(meshes)
            cols.append(column)
            picks.append(i)
            mesh_index.append(meshes[digest])
        frames.append((np.asarray(cols, dtype=np.int64), matrices[picks, :3, :].reshape(-1, 12),
                       np.asarray(mesh_index, dtype=np.int64), hidden[picks].astype(bool)))
    n = len(rows)
    block = np.zeros((n_frames, n, 12), dtype="<f4")
    block[:, :, [0, 5, 10]] = 1.0                      # an absent object: identity, as `grow` leaves it
    mesh_ids = np.full((n, n_frames), -1, dtype=np.int64)
    hidden_flags = np.ones((n, n_frames), dtype=bool)
    for frame, (cols, mats, mesh_index, hid) in enumerate(frames):
        block[frame, cols] = mats
        mesh_ids[cols, frame] = mesh_index
        hidden_flags[cols, frame] = hid
    for column, row in enumerate(rows):
        row["mesh"] = _compact(mesh_ids[column].tolist())
        row["hidden"] = _compact(hidden_flags[column].tolist())
    return _envelope(
        {"version": TRACK_VERSION, "animation": ir["id"], "n_frames": n_frames,
         "fps": ir["fps"], "meshes": [d.hex() for d in meshes], "objects": rows},
        [np.ascontiguousarray(block).tobytes()])


def _record_track(run_dir, ir, prefix, n_frames):
    """`track`'s bytes from whole decoded records: frames stored as JSON, or binary ones read the slow way."""
    columns: dict[str, int] = {}
    rows: list[dict] = []
    meshes: dict[str, int] = {}
    mesh_ids: list[list[int]] = []
    hidden: list[list[bool]] = []
    # Column-major while filling, `[frames][objects]` once emitted. Appending a
    # column to a Python list is what keeps a late-arriving object cheap: the
    # obvious `np.concatenate` onto a `[frames][objects]` array would recopy the
    # whole block per new object, which on a crowded scene is quadratic in the
    # object count for no reason.
    per_object: list[list[list[float]]] = []

    def grow(name, obj):
        columns[name] = len(rows)
        rows.append({"name": name, "identity": obj.get("identity"),
                     "color": [float(c) for c in obj.get("color") or (0.8, 0.8, 0.8, 1.0)]})
        mesh_ids.append([-1] * n_frames)
        hidden.append([True] * n_frames)
        # An object that appears late is *absent*, not at the origin: its
        # matrix stays identity and its mesh index -1 until it shows up.
        per_object.append([_IDENTITY_ROWS[:] for _ in range(n_frames)])

    for frame in range(n_frames):
        record = read_record(safe_path(run_dir, str(prefix)), frame,
                             checked=lambda relative: checked_file(run_dir, ir, f"{prefix}/{relative}"))
        for obj in record["objects"]:
            if obj.get("room"):
                continue
            name = obj["name"]
            if name not in columns:
                grow(name, obj)
            column = columns[name]
            digest = obj.get("mesh")
            if isinstance(digest, str):
                if digest not in meshes:
                    meshes[digest] = len(meshes)
                mesh_ids[column][frame] = meshes[digest]
            hidden[column][frame] = bool(obj.get("hidden"))
            # Rows 0..2 only: the fourth row of an affine matrix is [0,0,0,1]
            # in every frame of every clip, and dropping it is a quarter of the
            # track's bytes.
            per_object[column][frame] = [c for row in obj["matrix"][:3] for c in row]

    for column, row in enumerate(rows):
        row["mesh"] = _compact(mesh_ids[column])
        row["hidden"] = _compact(hidden[column])

    matrices = np.asarray(per_object, dtype="<f4").reshape(len(rows), n_frames, 12)
    return _envelope(
        {"version": TRACK_VERSION, "animation": ir["id"], "n_frames": n_frames,
         "fps": ir["fps"], "meshes": list(meshes), "objects": rows},
        [np.ascontiguousarray(matrices.transpose(1, 0, 2)).tobytes()])


def _track_header(run_dir) -> dict:
    """The track's JSON header, without re-deriving what the track knows.

    Reusing `track` rather than re-reading frame records is what keeps this
    honest about a clip it did not author: the track already resolves the
    object table over **every** frame, drops room surfaces, and collapses the
    per-frame mesh column. Reading frame 0 here instead would quietly assume
    that every object exists in it and that no object ever changes mesh,
    which is not true of clips in general.
    The build is memoized in `_TRACK_CACHE`, and the client fetches the track
    before the atlas anyway, so this is normally free.
    """
    blob = track(run_dir)
    return json.loads(blob[4:4 + int.from_bytes(blob[:4], "little")])


def _atlas_sources(run_dir, ir):
    """`{object name: (ir_id, part, uv, n_tris)}` for objects with usable UVs.

    The pairing this rests on, and why it is safe: the bake unwraps its atlas
    on the part's *staged* mesh, while native playback draws the *evaluated*
    one — and those differ. `_build_part` puts an unconditional `EDGE_SPLIT`
    modifier on every part, so a box's 8 vertices become 24. What EDGE_SPLIT
    does **not** change is the triangle list: count and corner order survive
    it, which is the same invariant `_uvs_from_rest_obj`, the sidecar itself,
    and `viewer/animation.py::_corner_uvs` already rely on. The sidecar is
    `(3T, 2)` in corner order, so it pairs with the evaluated triangles
    directly and no vertex correspondence is ever needed.

    Resolved per **object**, never per mesh digest. Evaluated geometry is
    deduplicated by content, so ten robot parts can share one mesh while owning
    ten different atlas islands; keying by digest would draw one part's texture
    on all of them.

    An object with no `identity`, no bake, or no sidecar is simply absent from
    the result and keeps the track's flat color — that is the ordinary state
    of a run whose bake has not finished, not an error.
    """
    from . import animation as anim_mod
    from . import derived as derived_mod
    from ..render import viewer_bake as VB

    scene_ir = anim_mod.load_scene_ir(Path(run_dir))
    if scene_ir is None:
        return {}
    manifests = derived_mod.manifests_for(Path(run_dir))
    if not manifests:
        return {}
    refs = {i.get("name"): i.get("object_ref")
            for i in scene_ir.get("instances") or ()}

    header = _track_header(run_dir)
    digests = header["meshes"]
    prefix = PurePosixPath(ir["evaluation"]).parent
    counted: dict[str, int] = {}

    def triangles(digest: str) -> int:
        """Triangle count of one evaluated mesh, read once per digest."""
        if digest not in counted:
            local = read_local(safe_path(run_dir, str(prefix)),
                               {"frame": 0, "objects": [{"name": "m", "mesh": digest}]})
            counted[digest] = len(local["m"][1])
        return counted[digest]

    out: dict[str, tuple] = {}
    for row in header["objects"]:
        identity = row.get("identity")
        if not isinstance(identity, str) or not identity:
            continue
        instance, _, part = identity.partition("/")
        ir_id = refs.get(instance)
        manifest_for = manifests.get(ir_id or "")
        if manifest_for is None or not part:
            continue
        found = VB.atlas_uvs(ir_id, manifest_for, part)
        if found is None:
            continue
        uv, tri = found
        # The mesh column is a scalar when the object never changes mesh and a
        # per-frame list when it does; `-1` is "absent in that frame". Taking
        # the first one it actually wears means an object that appears late in
        # the clip is handled like any other.
        column = row["mesh"]
        worn = [column] if isinstance(column, int) else list(column)
        first = next((i for i in worn if isinstance(i, int) and i >= 0), None)
        if first is None:
            continue
        # The only check available here. For the deformation path
        # `_corner_uv_source` can compare the index arrays elementwise, because
        # there the mesh IS the staged rest mesh; an evaluated mesh's indices
        # differ by construction (EDGE_SPLIT), so triangle count is the whole
        # of what can be verified. The client re-checks per digest before it
        # binds anything, which is what covers an object whose mesh changes
        # mid-clip to one the atlas does not fit.
        if len(tri) != triangles(digests[first]):
            continue
        # **V is flipped here, and it is the whole difference between a baked
        # texture that lands and one that is mirrored inside every island.**
        # The sidecar holds the layer as Blender stores it, V bottom-up, which
        # is right for the *mesh* UV path: those textures load through
        # `TextureLoader`, whose `flipY` defaults to true, so bottom-up V
        # samples correctly (`viewer/animation.py::_corner_uvs` says as much).
        # An atlas is different because its map arrives inside a GLB, and
        # `GLTFLoader` sets `flipY = false` on every texture it makes — glTF
        # measures V top-down. Blender's own exporter writes `1 - v` into the
        # GLB for exactly that reason, and those exported UVs are what the
        # static scene draws correctly today. Verified rather than reasoned:
        # the GLB's `TEXCOORD_0` equals `(u, 1 - v_sidecar)` to 2.4e-08.
        flipped = np.asarray(uv, dtype="<f4").copy()
        flipped[:, 1] = 1.0 - flipped[:, 1]
        out[row["name"]] = (ir_id, part, flipped, len(tri))
    return out


def atlas(run_dir) -> bytes:
    """Per-object baked atlas UVs, so native playback can wear its real maps.

    Its own route rather than a field on the track for two reasons: it is
    useless without a completed bake, which arrives minutes after the clip and
    on a different schedule; and a client with baked materials switched off
    should not pay for it at all. Shaped like `track` — one request for the
    whole clip — because the payload is per object and static for the run, and
    153 separate fetches would buy nothing.

    **Deduplicated by `(ir_id, part)`, which is what the atlas actually belongs
    to.** An object is an *instance* of a catalog part, so two stools' frames
    are one bake and one UV array; emitting per object sent the kitchen's
    25.5 MB as 3.2 M corners when 2.0 M are distinct. The client keys its
    geometry the same way, because that is the granularity at which a UV
    attribute is correct.

    An **empty payload is the normal answer**, not an error: a run with no
    shader materials, no bake yet, or no staged scene has nothing to send, and
    the client's fallback (the track's flat per-object color) is what it
    already draws today.
    """
    ir = manifest(run_dir)
    sources = _atlas_sources(run_dir, ir)

    atlases: dict[tuple, int] = {}
    entries: list[dict] = []
    arrays: list = []
    objects: list[dict] = []
    for name in sorted(sources):
        ir_id, part, uv, n_tris = sources[name]
        key = (ir_id, part)
        index = atlases.get(key)
        if index is None:
            index = atlases[key] = len(entries)
            entries.append({"key": f"{ir_id}/{part}", "n_tris": n_tris})
            arrays.append(np.ascontiguousarray(uv.reshape(-1, 2)))
        objects.append({"name": name, "uv_space": "atlas", "atlas": index})

    # `uint16` normalized, not float32. An atlas unwrap is bounded in [0, 1] by
    # construction, so the fixed-point step is 1/65535 — about thirty times
    # finer than one texel of the largest atlas the baker will emit
    # (`BAKE_MAX_SIZE = 2048`), while halving a payload that is 25 MB on this
    # corpus. `dtype` is still declared, and a bake that somehow lands outside
    # [0, 1] falls back to float32 rather than being silently wrapped: three
    # would tile such a UV, which draws a plausible, wrong texture.
    dtype = "u16"
    for array in arrays:
        if array.size and (array.min() < 0.0 or array.max() > 1.0):
            dtype = "f32"
            break

    buffers = []
    offset = 0
    for entry, array in zip(entries, arrays):
        packed = (array.astype("<f4") if dtype == "f32"
                  else np.rint(array * 65535.0).astype("<u2"))
        entry["offset"] = offset
        entry["n_corners"] = len(array)
        buffers.append(np.ascontiguousarray(packed).tobytes())
        offset += len(array)
    return _envelope({"version": TRACK_VERSION, "animation": ir["id"],
                      "dtype": dtype, "atlases": entries,
                      "objects": objects}, buffers)


#: Each mesh's topology hash, by digest: the triangles' bytes hashed, the same for every frame of a deforming object.
_TOPOLOGY: OrderedDict[str, str] = OrderedDict()
_TOPOLOGY_LOCK = threading.Lock()


def _topology(digest: str, triangles: np.ndarray) -> str:
    """A short hash of a mesh's triangles, which says whether two meshes can share one index and one normal plan."""
    import hashlib

    with _TOPOLOGY_LOCK:
        known = _TOPOLOGY.get(digest)
        if known is not None:
            _TOPOLOGY.move_to_end(digest)
            return known
    value = hashlib.sha256(np.ascontiguousarray(triangles, dtype="<u4").tobytes()).hexdigest()[:16]
    with _TOPOLOGY_LOCK:
        _TOPOLOGY[digest] = value
        while len(_TOPOLOGY) > 65536:
            _TOPOLOGY.popitem(last=False)
    return value


def _mesh_arrays(run_dir, digest: str):
    ir = manifest(run_dir)
    prefix = PurePosixPath(ir["evaluation"]).parent
    # Rejected on the manifest, not on the filesystem: `read_local` validates
    # the digest's *shape* and its content, but only the manifest says whether
    # this run has anything to do with it.
    listed_file(run_dir, ir, f"{prefix}/meshes/{digest}.npz")
    local = read_local(safe_path(run_dir, str(prefix)),
                       {"frame": 0, "objects": [{"name": "mesh", "mesh": digest}]})
    vertices, triangles = local["mesh"]
    return np.asarray(vertices, dtype="<f4"), np.asarray(triangles, dtype="<u4")


def mesh_positions(run_dir, digest: str) -> bytes:
    """One evaluated mesh's vertices only, exactly as `mesh_blob` sends them, with its topology hash.

    A deforming object's meshes share their triangles from frame to frame, two thirds of a mesh's bytes. The viewport
    takes them once, from the object's first mesh (`mesh_blob`), and then only these, checking `topology` to know the
    triangles are the same. The vertices are not rounded: the viewport works out creased normals from them, and on a
    character a rounding of even a hundredth of a millimeter tips hundreds of corners across the crease angle, a
    different few each frame, which shows as flicker.
    """
    vertices, triangles = _mesh_arrays(run_dir, digest)
    return _envelope(
        {"version": TRACK_VERSION, "digest": digest, "vertices": len(vertices),
         "topology": _topology(digest, triangles), "encoding": "f32"},
        [np.ascontiguousarray(vertices, dtype="<f4").tobytes()])


def mesh_blob(run_dir, digest: str) -> bytes:
    """One evaluated mesh, addressed by the digest of its own arrays.

    `read_local` is what reads it, rather than `np.load` here, so that this
    shares the process-wide decompressed-mesh cache with the verifier and the
    solvers — and so that the content-hash check lives in exactly one place.
    """
    vertices, triangles = _mesh_arrays(run_dir, digest)
    return _envelope(
        {"version": TRACK_VERSION, "digest": digest,
         "vertices": len(vertices), "triangles": len(triangles), "topology": _topology(digest, triangles)},
        [vertices.tobytes(), triangles.tobytes()])
