"""Versioned evaluated geometry storage shared by worker, verifier, and viewer."""
from collections import OrderedDict
import threading
import hashlib
import zipfile
import json
from pathlib import Path
import re
import struct
import numpy as np

#: What this module writes. `v2` (one JSON document per frame) is still read:
#: every bundle and accepted animation produced before v3 is in it, and the
#: replay check compares against those caches.
FORMAT = 'dycraft.evaluation.v3'
READABLE_FORMATS = (None, 'dycraft.evaluation.v2', 'dycraft.evaluation.v3')

#: A v3 frame is a fixed-layout binary record beside a shared table of the
#: things that do not vary from frame to frame.
#:
#: JSON would print float64 text for values Blender holds as float32 and repeat
#: every object's name on every frame; on a film with a thousand objects that is
#: about five times the bytes per frame and tens of times the time to write or
#: read one.
#:
#: float32 is not a precision compromise. `matrix_world` is float32 inside
#: Blender, so storing the source width loses nothing, and it makes the bytes
#: canonical, which is what lets a frame key be a digest of them.
#:
#: One file per frame, still: a partial check evaluates a slice of the timeline
#: and must write only those frames, and the viewer verifies each frame it
#: reads against the bundle manifest.
RECORD_MAGIC = b'DYCF'
RECORD_VERSION = 3
_RECORD_HEADER = struct.Struct('<4sHHIII16s')
#: Set when a body's placement is not simply its entry in the transform table.
#: `evaluated_frame` reads both from one `matrix_world`, so it writes none, but
#: the reader must not have to assume that.
_FLAG_BODY_MATRICES = 1
#: Set when the record carries anything outside the schema below, which is then
#: appended as canonical JSON. The signature digests the whole encoding, so a
#: field this layout does not know about still changes it, and a fixed binary
#: schema protects every key a record carries.
_FLAG_RESIDUAL = 2
#: Every per-body field the binary layout carries, with the value assumed when a
#: record omits it -- `surface_color(None)`'s gray, and "not hidden, not room".
_BODY_FIELDS = {'identity': None, 'room': False, 'hidden': False,
                'color': (0.6, 0.6, 0.6, 1.0)}
_BODY_KEYS = frozenset({'key', 'name', 'matrix', 'mesh', *_BODY_FIELDS})
_RECORD_KEYS = frozenset({'frame', 'objects', 'transforms', 'joint_positions', 'signature'})
#: The index fields a frame's identity and verdict depend on beside its own
#: record. A replay that changed one is another project state, and a re-check
#: across one is unsound: adding an allowance changes no frame's signature but
#: changes `NoPenetration`.
INDEX_FIELDS = ('fps', 'n_frames', 'frame_start', 'scene_ref', 'declarations', 'blender_version')
#: What a re-check also compares. Clip spans place clip-relative event and
#: allowance windows and MotionContinuity's transitions, and live in the
#: blend's NLA, where neither the design nor a frame's signature sees them.
#: They change no drawn frame, so a replay does not compare them. `rigs` is the
#: bone order every frame's `bone_rotations` is read in (the `bone` channel and
#: the per-bone anatomy rows), so a re-check across a changed table is unsound.
RECHECK_FIELDS = INDEX_FIELDS + ('clips', 'rigs')
_MESH_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LOCK = threading.RLock()


def asset_hash(vertices, triangles):
    h = hashlib.sha256()
    for array in (np.asarray(vertices, dtype='<f4'), np.asarray(triangles, dtype='<i4')):
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


#: Deflate level for mesh assets. `np.savez_compressed` uses 6; level 1 writes
#: several times faster to within a few percent of the size and reads back as
#: fast, while storing uncompressed costs about twice the disk.
MESH_DEFLATE_LEVEL = 1


def _save_npz(path, **arrays):
    """`np.savez_compressed` at `MESH_DEFLATE_LEVEL`: an ordinary `.npz`, read by `np.load`."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=MESH_DEFLATE_LEVEL) as archive:
        for name, array in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)


class MeshWriter:
    """Writes mesh assets on a background thread while the next frame evaluates.

    A new deforming mesh is compressed and written every frame: 30 ms of the
    drawbridge's river against 1.6 ms to hash it. zlib and the file write
    release the GIL, so on another thread they overlap Blender's evaluation of
    the next frame. The digest stays synchronous: the record, and so the
    signature, name it. At most `limit` writes wait, holding their arrays.
    `flush` waits for every write and raises the first one's error, so a reader
    of the evaluation sees whole files.
    """

    def __init__(self, limit=8):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mesh-writer')
        self._pending, self._limit = deque(), limit

    def submit(self, path, vertices, triangles):
        while len(self._pending) >= self._limit:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(_save_npz, path, vertices=vertices,
                                               triangles=triangles))

    def flush(self):
        while self._pending:
            self._pending.popleft().result()

    def close(self):
        """Wait for the writes and stop the thread; never raises.

        On the way out of a failed evaluation an error here would hide the
        one that ended it. A finished evaluation has flushed before writing
        its index, and that flush raises.
        """
        try:
            self.flush()
        except Exception:
            pass
        finally:
            self._pool.shutdown(wait=True)


def write_mesh(directory, vertices, triangles, known, memo=None, writer=None):
    """Store a local mesh under the digest of its own arrays; return the digest.

    `memo` maps the identity of an array pair to the digest already computed for
    it. The evaluated-geometry cache hands the *same* arrays back on every frame
    a body's geometry does not change, so without this the digest is recomputed
    from scratch per body per frame, megabytes of SHA-256 per frame. The arrays are
    retained by the memo so CPython cannot recycle an id under a live key, and
    the cache only ever returns read-only arrays, so an entry cannot go stale
    under a mutation.
    """
    fingerprint = (id(vertices), id(triangles))
    if memo is not None and fingerprint in memo:
        return memo[fingerprint][0]
    key = asset_hash(vertices, triangles)
    if key not in known:
        path = Path(directory) / 'meshes' / (key+'.npz')
        path.parent.mkdir(exist_ok=True)
        if writer is not None:
            writer.submit(path, vertices, triangles)
        else:
            _save_npz(path, vertices=vertices, triangles=triangles)
        known.add(key)
    if memo is not None and not (vertices.flags.writeable or triangles.flags.writeable):
        memo[fingerprint] = (key, vertices, triangles)
    return key


def _read_asset(asset, key):
    """One v2 mesh asset, validated against its content key, through the process cache."""
    global _CACHE_BYTES
    if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('invalid evaluated mesh reference')
    asset = Path(asset)
    stat = asset.stat()
    # `str(asset)`, not `asset.resolve()`: st_dev + st_ino already
    # identify the file uniquely, and the content hash `key` is in
    # the tuple as well, so resolving adds no discriminating power.
    # It did add a syscall per body per frame — 23,520 `resolve()`
    # calls costing 1.83 s over 193 frames of a 114-body clip, 7% of
    # the pass, spent proving a cache entry valid that the inode
    # already identified.
    fingerprint = (str(asset), key, stat.st_dev, stat.st_ino,
                   stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _CACHE_LOCK:
        if fingerprint in _MESH_CACHE:
            _MESH_CACHE.move_to_end(fingerprint)
            return _MESH_CACHE[fingerprint]
        with np.load(asset, allow_pickle=False) as data:
            v, t = data['vertices'].copy(), data['triangles'].copy()
        if asset_hash(v, t) != key:
            raise ValueError('evaluated mesh content hash mismatch')
        v.flags.writeable = t.flags.writeable = False
        size = v.nbytes+t.nbytes
        # Bounded by bytes. A small count bound thrashes on a scene with more
        # meshes than the bound, even when they total a few MB, and every
        # frame read back then re-loads and re-hashes them.
        while _MESH_CACHE and (_CACHE_BYTES+size > 64*1024**2 or len(_MESH_CACHE) >= 8192):
            _, (old_v, old_t) = _MESH_CACHE.popitem(last=False)
            _CACHE_BYTES -= old_v.nbytes+old_t.nbytes
        if size <= 64*1024**2:
            _MESH_CACHE[fingerprint] = v, t
            _CACHE_BYTES += size
        return v, t


def read_local(directory, record, checked=None):
    """Read a record's meshes: shared assets, or a per-frame NPZ in older bundles.

    Recorded films store each frame's meshes in `<frame>.npz` beside the record;
    they are read so those films still replay.

    checked(relative) optionally verifies each file against a committed manifest.
    Asset ids are always validated, including offline replay and solver reads.
    """
    directory = Path(directory)
    def path(relative):
        return checked(relative) if checked else directory / relative
    result = {}
    per_frame = None
    try:
        for obj in record['objects']:
            if 'mesh' in obj:
                key = obj['mesh']
                if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key):
                    raise ValueError('invalid evaluated mesh reference')
                v, t = _read_asset(path('meshes/'+key+'.npz'), key)
            else:
                if per_frame is None:
                    per_frame = np.load(path(f"{record['frame']:06d}.npz"), allow_pickle=False)
                v, t = per_frame[obj['key']+'_v'].copy(), per_frame[obj['key']+'_t'].copy()
            result[obj['name']] = (v, t)
    finally:
        if per_frame is not None:
            per_frame.close()
    return result


class _LazyLocal(dict):
    """`read_local`'s mapping, read one mesh at a time, on first access.

    A pass that needs a few bodies of a frame -- an advisory fill, a span's
    diagnostics -- reads those, not the frame's hundreds.
    """

    def __init__(self, directory, record):
        super().__init__()
        self._directory = Path(directory)
        self._objects = {obj['name']: obj for obj in record['objects']}

    def __missing__(self, name):
        key = self._objects[name]['mesh']
        value = self[name] = _read_asset(self._directory / 'meshes' / (str(key) + '.npz'), key)
        return value


class WorldGeometry(dict):
    """World arrays for existing consumers, plus exact local arrays for BVH reuse.

    Materialized **on first access**, not for every object in the record. The
    transform is a full pass over the vertex array plus a float32->float64 copy,
    and the frame loop that consumes this skips hidden objects, room objects and
    anything in `baseline['room_objects']` outright — so building them eagerly
    paid for arrays nobody read, once per frame. Only `__getitem__` populates:
    no consumer iterates this mapping (checked across animation/, verify/,
    viewer/, render/), and one that starts to must materialize explicitly.
    """

    def __init__(self, local, matrices):
        super().__init__()
        self.local = local
        self._matrices = matrices

    def __missing__(self, name):
        vertices, triangles = self.local[name]
        matrix = np.asarray(self._matrices[name])
        value = (vertices.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3], triangles)
        self[name] = value
        return value


def world_geometry(record, local):
    return WorldGeometry(local, {obj['name']: obj['matrix'] for obj in record['objects']})


def _frame_table(record):
    """The part of a record that does not vary from frame to frame.

    Names, identities, room membership, and the object order are properties of
    the scene, not of the moment. Hoisting them leaves the frame file holding
    only numbers. A frame whose table differs -- an object added, a body whose
    mesh stopped evaluating -- simply writes a new table and names it in its
    header, so the layout never has to assume the scene is fixed.
    """
    return {'objects': list(record['transforms']),
            'bodies': [[obj['name'], obj.get('identity', _BODY_FIELDS['identity']),
                        bool(obj.get('room', _BODY_FIELDS['room']))]
                       for obj in record['objects']],
            'joints': list(record.get('joint_positions') or {})}


def _table_id(table):
    return hashlib.blake2b(json.dumps(table, sort_keys=True, separators=(',', ':')).encode(),
                           digest_size=16).hexdigest()


def write_record(directory, record, known_tables=None):
    """One evaluated frame as a fixed-layout binary record. Returns its path."""
    payload, table = encode_record(record)
    return _write_encoded(directory, record['frame'], payload, table, record['signature'],
                          known_tables)


def write_signed_record(directory, record, known_tables=None, *, transforms=None):
    """Sign an evaluated frame and write it: `frame_signature`, then `write_record`.

    One encoding serves both, since encoding is most of the cost of storing a
    frame. Sets `record['signature']` and returns it.
    `transforms` is `encode_record`'s. The record's bodies all carry `mesh`.
    """
    payload, table = encode_record(record, transforms=transforms)
    record['signature'] = hashlib.sha256(payload).hexdigest()
    _write_encoded(directory, record['frame'], payload, table, record['signature'], known_tables)
    return record['signature']


def _write_encoded(directory, frame, payload, table, signature, known_tables):
    directory = Path(directory)
    table_id = _table_id(table)
    if known_tables is None or table_id not in known_tables:
        path = directory / 'tables' / (table_id + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table))
        if known_tables is not None:
            known_tables.add(table_id)
    path = directory / f'{frame:06d}.bin'
    path.write_bytes(payload + bytes.fromhex(signature))
    return path


def decode_record(buffer, frame, table, *, transforms=True, table_digest=None):
    """A v3 record as the dict every consumer already reads.

    `transforms=False` leaves out `transforms` and `joint_positions`, for a
    reader that only places bodies: an advisory fill. Most of a frame's
    numbers are transforms of objects that are not bodies, 1,654 against 406
    on the chair film. Leaving the keys out, rather than empty, makes a reader
    that needed them fail instead of seeing nothing. `table_digest` is
    `_table_id(table)` when the caller already computed it from this table.
    """
    magic, version, flags, n_objects, n_bodies, n_joints, table_id = \
        _RECORD_HEADER.unpack_from(buffer, 0)
    if magic != RECORD_MAGIC or version != RECORD_VERSION:
        raise ValueError('not a DyCraft evaluated frame record')
    if table_id.hex() != (table_digest if table_digest is not None else _table_id(table)):
        raise ValueError('evaluated frame record does not match its table')
    offset = _RECORD_HEADER.size
    def take(dtype, count, shape=None):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        offset += array.nbytes
        return array.reshape(shape) if shape else array
    matrix_rows = take('<f4', n_objects * 16, (n_objects, 4, 4))
    placements = (take('<f4', n_bodies * 16, (n_bodies, 4, 4))
                  if flags & _FLAG_BODY_MATRICES else None)
    digests = take(np.uint8, n_bodies * 32, (n_bodies, 32))
    colors = take('<f4', n_bodies * 4, (n_bodies, 4))
    joints = take('<f4', n_joints)
    hidden = take(np.uint8, n_bodies)
    offset += -n_bodies % 4
    residual = {'objects': [{}] * n_bodies, 'record': {}}
    if flags & _FLAG_RESIDUAL:
        end = len(buffer) - 32
        residual = json.loads(bytes(buffer[offset:end]).decode())
        offset = end
    signature = bytes(buffer[offset:offset + 32]).hex()
    names = table['objects']
    if (len(names) != n_objects or len(table['bodies']) != n_bodies
            or len(table['joints']) != n_joints):
        raise ValueError('evaluated frame record disagrees with its table')
    slots = {name: i for i, name in enumerate(names)}
    # `tolist` gives each float32 as the Python float `float(v)` does, one call
    # per array rather than one per element: the per-element form was most of
    # a frame read on a 650-body film. A body's matrix is copied row by row, so
    # it is not the same list as its transform's, as before.
    if placements is not None:
        matrices = placements.tolist()
    elif not transforms:
        matrices = matrix_rows[[slots[name] for name, _, _ in table['bodies']]].tolist()
    else:
        transform_rows = matrix_rows.tolist()
        matrices = [[row[:] for row in transform_rows[slots[name]]] for name, _, _ in table['bodies']]
    color_rows, hidden_flags = colors.tolist(), hidden.tolist()
    objects = []
    for i, (name, identity, room) in enumerate(table['bodies']):
        objects.append({'key': f'o{i}', 'name': name, 'identity': identity, 'room': room,
                        'matrix': matrices[i],
                        'hidden': bool(hidden_flags[i]), 'color': color_rows[i],
                        'mesh': bytes(digests[i]).hex(), **residual['objects'][i]})
    record = {'frame': frame, 'objects': objects, **residual['record']}
    if transforms:
        if placements is not None:
            transform_rows = matrix_rows.tolist()
        record['transforms'] = dict(zip(names, transform_rows))
        record['joint_positions'] = dict(zip(table['joints'], joints.tolist()))
    record['signature'] = signature
    return record


#: Frame tables already parsed and digested, by file identity, as `_read_asset`
#: holds meshes. Every frame of a film names the same table or a few, and
#: parsing and digesting it again for each was a fifth of a frame read.
_TABLES = OrderedDict()


def _read_table(path, table_id):
    """`(table, its digest)` for a table file, through the process cache."""
    stat = path.stat()
    fingerprint = (str(path), table_id, stat.st_dev, stat.st_ino,
                   stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _CACHE_LOCK:
        held = _TABLES.get(fingerprint)
        if held is not None:
            _TABLES.move_to_end(fingerprint)
            return held
    table = json.loads(path.read_text())
    held = (table, _table_id(table))
    with _CACHE_LOCK:
        _TABLES[fingerprint] = held
        while len(_TABLES) > 64:
            _TABLES.popitem(last=False)
    return held


def read_record(directory, frame, checked=None, *, transforms=True):
    """One evaluated frame, in either storage format.

    `checked(relative)` is the viewer's manifest gate: a bundle member is
    verified before it is read, and the table is a bundle member too, so a
    gated read takes no table from the cache. `transforms` is
    `decode_record`'s; a JSON frame is returned whole.
    """
    directory = Path(directory)
    def path(relative):
        return Path(checked(relative)) if checked else directory / relative
    try:
        buffer = path(f'{frame:06d}.bin').read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return json.loads(path(f'{frame:06d}.json').read_text())
    except ValueError:
        # A v2 bundle: `checked` rejects the binary name because the manifest
        # has never heard of it. Not finding a frame in the new format is not
        # an integrity failure, so ask for the one the bundle does list.
        return json.loads(path(f'{frame:06d}.json').read_text())
    table_id = _RECORD_HEADER.unpack_from(buffer, 0)[6].hex()
    if checked:
        table = json.loads(path(f'tables/{table_id}.json').read_text())
        return decode_record(buffer, frame, table, transforms=transforms)
    table, digest = _read_table(directory / 'tables' / f'{table_id}.json', table_id)
    return decode_record(buffer, frame, table, transforms=transforms, table_digest=digest)


def load_frame(directory, frame, *, transforms=True):
    record = read_record(directory, frame, transforms=transforms)
    if all('mesh' in obj for obj in record['objects']):
        return record, world_geometry(record, _LazyLocal(directory, record))
    return record, world_geometry(record, read_local(directory, record))


def _f4(values, shape):
    """`values` as canonical little-endian float32: -0.0 is written as +0.0.

    Blender's stored bits for a zero depend on how the scene reached the frame.
    It skips writing an animated property whose new value equals the old one
    in float32, and when it does write, the NLA blend `old + (value - old) *
    influence` turns a curve's -0.0 into +0.0. So a jump to a frame keeps the
    file's -0.0 while stepping through the film writes +0.0, and two
    evaluations of the same film would sign differently over sign-of-zero
    alone. Adding +0.0 maps -0.0 to +0.0 and leaves every other value, NaN
    included, unchanged.
    """
    return (np.asarray(values, dtype='<f4').reshape(shape) + np.float32(0)).astype('<f4', copy=False)


def encode_record(record, table=None, *, transforms=None):
    """The canonical bytes of one evaluated frame, without its signature.

    This is both what `write_record` stores and what `frame_signature` digests,
    so a frame has exactly one canonical form and the signature costs a hash of
    ~70 KB rather than a second canonical JSON rendering of the whole record.
    Zeros are canonical (`_f4`), so the form does not depend on the evaluation
    path.

    `transforms`, when given, is `record['transforms']` as the float32 array
    its lists were made from, in the table's order. The evaluation holds it
    already, and converting the nested lists back to float32 would be most of
    an encoding on a scene of a thousand objects. The lists are `tolist()` of that array, so
    both give the same float32 values.
    """
    table = table if table is not None else _frame_table(record)
    names = table['objects']
    bodies = record['objects']
    joints = record.get('joint_positions') or {}
    raw = transforms
    if raw is None:
        transforms = _f4([record['transforms'][name] for name in names], (len(names), 4, 4))
    else:
        if len(raw) != len(names):
            raise ValueError('transform array does not match the record table')
        transforms = _f4(raw, (len(names), 4, 4))
    authored = _f4([obj['matrix'] for obj in bodies], (len(bodies), 4, 4))
    # A body is normally placed by its entry in the transform table -- both come
    # from one `matrix_world` read -- so storing the placement twice is waste. A
    # record that does not carry every body in its table (hand-built records and
    # older bundles' per-frame NPZ records do not) simply keeps the body matrices.
    flags = _FLAG_BODY_MATRICES
    if raw is not None:
        slots = {name: i for i, name in enumerate(names)}
        if all(obj['name'] in slots for obj in bodies):
            placed = np.asarray(raw, dtype='<f4')[[slots[obj['name']] for obj in bodies]]
            if np.array_equal(placed.reshape(len(bodies), 4, 4), authored):
                flags = 0
    else:
        placements = [record['transforms'].get(obj['name']) for obj in bodies]
        if all(placement is not None for placement in placements):
            if np.array_equal(np.asarray(placements, dtype='<f4').reshape(len(bodies), 4, 4), authored):
                flags = 0
    table_id = bytes.fromhex(_table_id(table))
    blocks = [_RECORD_HEADER.pack(RECORD_MAGIC, RECORD_VERSION, flags, len(names),
                                  len(bodies), len(joints), table_id),
              transforms.tobytes()]
    if flags & _FLAG_BODY_MATRICES:
        blocks.append(authored.tobytes())
    blocks += [b''.join(bytes.fromhex(obj['mesh']) for obj in bodies),
               _f4([obj.get('color', _BODY_FIELDS['color']) for obj in bodies],
                   (len(bodies), 4)).tobytes(),
               _f4([joints[name] for name in table['joints']], (len(table['joints']),)).tobytes(),
               np.asarray([bool(obj.get('hidden', _BODY_FIELDS['hidden'])) for obj in bodies],
                          dtype=np.uint8).tobytes(),
               b'\x00' * (-len(bodies) % 4)]
    residual = [{k: v for k, v in obj.items() if k not in _BODY_KEYS} for obj in bodies]
    extra = {k: v for k, v in record.items() if k not in _RECORD_KEYS}
    if extra or any(residual):
        flags |= _FLAG_RESIDUAL
        blocks.append(json.dumps({'objects': residual, 'record': extra},
                                 sort_keys=True, separators=(',', ':')).encode())
    blocks[0] = _RECORD_HEADER.pack(RECORD_MAGIC, RECORD_VERSION, flags, len(names),
                                    len(bodies), len(joints), table_id)
    return b''.join(blocks), table


def frame_signature(record, local=None):
    """Geometry-and-properties identity of one evaluated frame.

    Storage references and key ordering are deliberately outside it: the digest
    is of the *canonical encoding* of what was evaluated -- every object's
    placement, every body's mesh identity, color, and visibility, and every
    joint coordinate -- so the same frame in either storage format signs the
    same, and a v2 cache and a v3 cache can be compared directly.

    `obj['mesh']` **is** `asset_hash(vertices, triangles)`: `write_mesh` returns
    exactly that and `read_local` refuses any asset whose content does not hash
    to the id it was requested under, so the geometry is covered without a
    second pass over the vertex arrays. `local` is accepted for older bundles'
    per-frame NPZ records, which carry no id, and ignored otherwise.
    """
    if any('mesh' not in obj for obj in record['objects']):
        record = {**record, 'objects': [{**obj, 'mesh': obj.get('mesh') or asset_hash(*local[obj['name']])}
                                        for obj in record['objects']]}
    return hashlib.sha256(encode_record(record)[0]).hexdigest()


def replay_differences(reference, record):
    """What differs between two records of one frame, by field, with names.

    Compared in the signature's own encoding (`_f4`), so a field is named here
    exactly when it is what changed the signature. `objects` names what one
    side has and the other does not. Both records carry `mesh` digests.
    """
    def differing(old, new, value):
        return sorted({name for name in old.keys() & new.keys()
                       if not np.array_equal(value(old[name]), value(new[name]))})
    old = {obj['name']: obj for obj in reference['objects']}
    new = {obj['name']: obj for obj in record['objects']}
    old_q, new_q = reference.get('joint_positions') or {}, record.get('joint_positions') or {}
    old_b, new_b = reference.get('bone_rotations') or {}, record.get('bone_rotations') or {}
    changed = {
        'transforms': differing(reference['transforms'], record['transforms'],
                                lambda matrix: _f4(matrix, (4, 4))),
        'joints': sorted(set(differing(old_q, new_q, lambda q: _f4([q], (1,))))
                         | (old_q.keys() ^ new_q.keys())),
        # Armatures whose bone orientations differ, compared as the residual
        # block stores them (canonical JSON of the six-decimal rows).
        'bones': sorted({name for name in old_b.keys() & new_b.keys()
                         if old_b[name] != new_b[name]} | (old_b.keys() ^ new_b.keys())),
        'color': differing(old, new, lambda obj: _f4(obj.get('color', _BODY_FIELDS['color']), (4,))),
        'mesh': differing(old, new, lambda obj: obj.get('mesh')),
        'hidden': differing(old, new, lambda obj: bool(obj.get('hidden', False))),
        'objects': sorted((reference['transforms'].keys() ^ record['transforms'].keys())
                          | (old.keys() ^ new.keys())),
    }
    return {field: names for field, names in changed.items() if names}


def replay_frame(directory, frame, record, local, details=None):
    """`(signature, changed)`: this evaluation's signature and what differs from the accepted one.

    Both sides are signed **now**, rather than trusting the digest the reference
    cache stored. The comparison then does not depend on which storage format
    the accepted animation is in, nor on the signature definition in force the
    day it was accepted -- only on whether the two evaluations agree. It costs
    one encode of a record already read, and it is what lets an accepted v2
    cache be re-checked by a v3 run. `changed` is `replay_differences`' map,
    empty when the frame reproduces. Given a dict as `details`, a changed
    color also leaves its values there (`replay_colors`); `changed` itself
    stays names, which is what the bake hint and the delivery record read.
    """
    reference = read_record(directory, frame)
    # Old accepted caches did not record turn counts; retain their existing
    # geometry-only replay contract rather than fabricating missing evidence.
    if 'joint_positions' not in reference:
        record = {k: v for k, v in record.items() if k != 'joint_positions'}
    # The same contract for bone orientations, which caches accepted before
    # `worker.bone_rotations` existed do not carry: their rigged characters
    # replay on geometry alone, as they were accepted.
    if 'bone_rotations' not in reference:
        record = {k: v for k, v in record.items() if k != 'bone_rotations'}
    def with_meshes(rec, arrays):
        if all('mesh' in obj for obj in rec['objects']):
            return rec
        arrays = arrays() if callable(arrays) else arrays
        return {**rec, 'objects': [{**obj, 'mesh': obj.get('mesh') or asset_hash(*arrays[obj['name']])}
                                   for obj in rec['objects']]}
    reference = with_meshes(reference, lambda: read_local(directory, reference))
    record = with_meshes(record, local)
    signature = frame_signature(record)
    if signature == frame_signature(reference):
        return signature, {}
    changed = (replay_differences(reference, record)
               or {'record': ['fields outside the schema']})
    if details is not None and changed.get('color'):
        details['color'] = replay_colors(reference, record, changed['color'])
    return signature, changed


def replay_colors(reference, record, names):
    """`{name: [old rgb, new rgb]}` for the bodies `names`, three decimals.

    A refusal that named only "color" sent one agent to unbind a skin-bound
    pouch to fix a color change on a wall; the values say what changed.
    """
    old = {obj['name']: obj for obj in reference['objects']}
    new = {obj['name']: obj for obj in record['objects']}
    def rgb(obj):
        return [round(float(c), 3) for c in _f4(obj.get('color', _BODY_FIELDS['color']), (4,))[:3]]
    return {name: [rgb(old[name]), rgb(new[name])] for name in names if name in old and name in new}


def replay_message(frame, changed, simulated=(), details=None):
    """A refusal naming what differed, and a bake hint only where a cache explains it.

    A scene with nothing to bake gets no bake hint, which would point the author
    at the wrong cause.
    """
    colors = (details or {}).get('color') or {}
    def listed(field, names):
        shown = [f'{name} {colors[name][0]} -> {colors[name][1]}'
                 if field == 'color' and name in colors else name for name in names[:12]]
        return f'{field}: {shown}' + (f' (+{len(names) - 12} more)' if len(names) > 12 else '')
    named = '; '.join(listed(field, names) for field, names in changed.items())
    baked = sorted({name for names in changed.values() for name in names} & set(simulated))
    return (f'accepted animation replay changed geometry/properties at frame {frame}; '
            f'changed {named}'
            + (f'; {", ".join(baked[:6])} carry simulation caches: bake them' if baked else ''))


def check_replay_frame(directory, frame, record, local, simulated=()):
    """`replay_frame`, raising when the frame does not reproduce.

    For a caller that must refuse a mismatch. A delivery render records
    mismatches and draws the frame instead.
    """
    details = {}
    _, changed = replay_frame(directory, frame, record, local, details)
    if changed:
        raise ValueError(replay_message(frame, changed, simulated, details))


def merge_chunk_indexes(indexes, owned=None):
    """One `index.json` from a parallel check's chunks.

    The chunks evaluated one film, so everything but their frames is the
    same, and the first chunk's index is taken for it -- but each chunk
    carries only its own frames' `signatures` and `render_states`, which a
    merge that kept the first index whole dropped. A chunk that disagrees on
    `render_state_version` makes every digest incomparable, so they are all
    withdrawn rather than mixed.

    `owned`, one `(low, high)` per index, takes each frame's values from the
    chunk that owns it. A chunk also indexes its context frames, and its first
    one can lack what its owner has: with motion blur on, a digest needs the
    frame before, which that chunk never evaluated.
    """
    pairs = [(index, span) for index, span in zip(indexes, owned or [None] * len(indexes))
             if index is not None]
    indexes = [index for index, _ in pairs]
    merged = dict(indexes[0])
    merged['frames'] = sorted({frame for index in indexes for frame in index.get('frames', ())})
    for key in ('signatures', 'render_states'):
        if any(key in index for index in indexes):
            merged[key] = {frame: value for index, span in pairs
                           for frame, value in (index.get(key) or {}).items()
                           if span is None or span[0] <= int(frame) <= span[1]}
    versions = {index.get('render_state_version') for index in indexes}
    if len(versions) > 1 and 'render_states' in merged:
        merged['render_states'] = dict.fromkeys(merged['render_states'])
        merged['render_state_version'] = None
    merged.update(evaluation_start_frame=0, complete=True)
    return merged
