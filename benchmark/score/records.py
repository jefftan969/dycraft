"""Read one evaluated frame in either `dycraft.evaluation` storage format.

* **v2**: `NNNNNN.json`, one document per frame.
* **v3**: `NNNNNN.bin`, a fixed-layout binary record (header `<4sHHIII16s`: magic `DYCF`, version 3, flags,
  n_objects, n_bodies, n_joints, table id), plus `tables/<id>.json` holding the scene's names.

The layout mirrors `animation/blender/evaluation.decode_record` but is decoded here, so a change to the producer's
reader cannot silently change what is scored. Only the fields the metrics use are decoded: per body its name,
identity, matrix, hidden flag, and mesh digest, plus the frame signature.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

READABLE_FORMATS = (None, "dycraft.evaluation.v2", "dycraft.evaluation.v3")
_HEADER = struct.Struct("<4sHHIII16s")
_MAGIC, _VERSION = b"DYCF", 3
_FLAG_BODY_MATRICES = 1
_tables: dict = {}


def check_format(index: dict) -> None:
    if index.get("format") not in READABLE_FORMATS:
        raise ValueError(f"unsupported evaluation format {index.get('format')!r}; "
                         f"readable: {READABLE_FORMATS[1:]}")


def _table_id(table) -> str:
    return hashlib.blake2b(json.dumps(table, sort_keys=True, separators=(",", ":")).encode(),
                           digest_size=16).hexdigest()


def read_record(directory, frame: int) -> tuple[dict, bytes]:
    """(record, raw bytes) for one frame; the raw bytes feed the input digest."""
    directory = Path(directory)
    binary = directory / f"{frame:06d}.bin"
    if not binary.is_file():
        raw = (directory / f"{frame:06d}.json").read_bytes()
        return json.loads(raw), raw
    buffer = binary.read_bytes()
    magic, version, flags, n_objects, n_bodies, n_joints, table_id = _HEADER.unpack_from(buffer, 0)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError(f"not a DyCraft v3 evaluated frame record: {binary}")
    tid = table_id.hex()
    if tid not in _tables:
        table = json.loads((directory / "tables" / f"{tid}.json").read_text())
        if _table_id(table) != tid:
            raise ValueError(f"evaluated frame table does not match its id: {tid}")
        _tables[tid] = table
    table = _tables[tid]
    if len(table["objects"]) != n_objects or len(table["bodies"]) != n_bodies:
        raise ValueError(f"evaluated frame record disagrees with its table: {binary}")
    offset = _HEADER.size

    def take(dtype, count, shape=None):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        offset += array.nbytes
        return array.reshape(shape) if shape else array

    transforms = take("<f4", n_objects * 16, (n_objects, 4, 4))
    placements = take("<f4", n_bodies * 16, (n_bodies, 4, 4)) if flags & _FLAG_BODY_MATRICES else None
    digests = take(np.uint8, n_bodies * 32, (n_bodies, 32))
    take("<f4", n_bodies * 4)            # colors
    take("<f4", n_joints)                # joint positions
    hidden = take(np.uint8, n_bodies)
    signature = bytes(buffer[-32:]).hex()
    slots = {name: i for i, name in enumerate(table["objects"])}
    objects = []
    for i, (name, identity, room) in enumerate(table["bodies"]):
        matrix = placements[i] if placements is not None else transforms[slots[name]]
        objects.append({"name": name, "identity": identity, "room": bool(room),
                        "matrix": matrix.astype(np.float64).tolist(), "hidden": bool(hidden[i]),
                        "mesh": bytes(digests[i]).hex()})
    return {"frame": frame, "objects": objects, "signature": signature}, buffer
