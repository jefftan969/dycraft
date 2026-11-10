"""Staged-mesh reading and point-in-mesh tests for the exact mesh backend.

`load_obj` reads a staged OBJ into numpy arrays, and `point_inside_mesh` /
`winding_number` answer containment by the generalized winding number, which
stays robust on meshes that are not perfectly watertight.
`verify/_narrow_phase.py` and `verify/_exact_adjudicator.py` read meshes
through here.

**Why numpy, when the rest of `sdk/` is pure Python.** `_rotation.py` and the
two kernels work on SE3s and AABBs, a handful of floats each. Here the natural
unit is a dense per-vertex array, and pure Python is about 100x slower at the
mesh sizes this sees, so this module imports numpy and says so.

The module name dates from the deformation rigs (since removed) that first
read meshes this way.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._errors import InvalidRig


#: Generalized-winding-number threshold for "inside". 0.5 is the exact value
#: for a closed surface; anything watertight lands within 1e-6 of 0 or 1.
INSIDE_THRESHOLD = 0.5


def _ref(kind: str, name) -> dict:
    return {"kind": kind, "name": name, "id": None}


# ---------------------------------------------------------------------------
# Rest meshes
# ---------------------------------------------------------------------------


def load_obj(path: Path) -> tuple:
    """Parse an OBJ into `(vertices[n,3], triangles[m,3])`.

    Fan-triangulates polygons and drops texture/normal indices, which is all the
    containment test needs. `_mesh_assets._parse_obj_aabb` reads the same files
    but only their `v` lines — it is computing an AABB and never needs
    connectivity — so this is a second reader rather than a shared one.
    """
    verts: list = []
    tris: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            fields = raw.split()
            if not fields:
                continue
            tag = fields[0]
            if tag == "v" and len(fields) >= 4:
                verts.append((float(fields[1]), float(fields[2]),
                              float(fields[3])))
            elif tag == "f" and len(fields) >= 4:
                face = [int(f.split("/", 1)[0]) - 1 for f in fields[1:]]
                for k in range(1, len(face) - 1):
                    tris.append((face[0], face[k], face[k + 1]))
    if not verts or not tris:
        raise InvalidRig(
            f"{path.name} has no usable triangles "
            f"({len(verts)} vertices, {len(tris)} triangles)",
            refs=[_ref("part", path.name)],
            evidence={"mesh": str(path), "n_vertices": len(verts),
                      "n_triangles": len(tris)},
            suggested_repairs=[
                "check that this part's staged mesh has faces: the exact mesh "
                "backend needs triangles to measure against",
            ],
        )
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(tris, dtype=np.int32))


# ---------------------------------------------------------------------------
# Closest point on a triangle set  (vectorised over triangles)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inside / outside
# ---------------------------------------------------------------------------


def winding_number(verts, tris, point) -> float:
    """Generalized winding number of the mesh about `point`.

    Van Oosterom-Strackee signed solid angle summed over triangles. Chosen over
    a parity ray cast because it degrades gracefully: a ray cast needs a
    watertight mesh and a lucky direction, while this returns ~1 inside and ~0
    outside for anything close to closed, and something in between (rather than
    a coin flip) for a mesh that genuinely is not.
    """
    tri = verts[tris]
    p = np.asarray(point, dtype=np.float64)
    a, b, c = tri[:, 0, :] - p, tri[:, 1, :] - p, tri[:, 2, :] - p
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc
           + np.einsum("ij,ij->i", a, b) * lc
           + np.einsum("ij,ij->i", a, c) * lb
           + np.einsum("ij,ij->i", b, c) * la)
    return float(np.sum(2.0 * np.arctan2(num, den)) / (4.0 * math.pi))


def point_inside_mesh(verts, tris, point) -> bool:
    return abs(winding_number(verts, tris, point)) > INSIDE_THRESHOLD
