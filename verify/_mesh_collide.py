"""Mesh geometry shared by the collision paths, and the exact room test.

A deformed part has no B-rep: LBS moves vertices without producing a new solid, and the exact adjudicator works on
B-rep booleans. Handing it the rest solid would be worse than not asking, because `adjudicate_*` only downgrades:
a misplaced pair shares no volume, comes back CLEAR, and a real interpenetration is dropped. So deformed geometry
is measured on its triangles, and a penetration check excludes it rather than adjudicating the wrong shape.

**The narrow phase is triangle against triangle** (`verify/_narrow_phase.py`, on `coal`). A vertex-inside test
misses a crossing in which no vertex of either body is inside the other (two thin sheets meeting in an X, a
gripper's fingers against its hand), which is the normal case for the cloth and panels this path is for. `coal` is
a hard dependency: a narrow phase that silently misses gripper contacts is worse than none.

What remains here is what a BVH does not answer better:

* **`room_escape`**, exact: a room face is an axis-aligned plane, so the deepest excursion is a coordinate extremum
  over the real vertices.
* **Mesh utilities**: `transform_to_world` and `world_aabb`, used across the deformation path.

"Touching" is `_exact_adjudicator.TANGENCY_DEPTH_M` on both paths, one constant `_narrow_phase` imports so the two
cannot drift; `worth_narrow_phase` is the narrow phase's broad-phase filter.

**Resolution is a limit** for anything measured on these triangles. They come from the render tessellation the
object layer staged, whose chord tolerance was chosen for pictures; against their B-reps they under-fill by a
fraction of a percent of volume, up to tens of microns of surface. That hides an overlap rather than inventing
one, which is why the object layer does not measure this way.
"""

from __future__ import annotations


import numpy as np



def face_clearance(verts, room_lo, room_hi, axis: int, side: str) -> float:
    """Signed clearance from a world-space mesh to one room face.

    Positive is inside the room, negative is material through the face. The one
    computation both directions of the room question reduce to — how much room
    is left (`_exact_adjudicator._mesh_face_gap`, backing `room_gap_at`) and how
    far out it has gone (`room_escape`, backing `DeformedNoPenetration`) — which
    is why it is one function rather than two that agree by inspection.

    Exact, and cheaply so: a room face is an axis-aligned plane, so the extreme
    excursion is a coordinate minimum or maximum over the real vertices. No
    boolean, no BVH, and no blind spot. `axis` is 0/1/2 and `side` is `"below"`
    (the low face on that axis) or `"above"`, matching
    `_exact_adjudicator.ROOM_FACES`.
    """
    v = np.asarray(verts)
    if side == "below":
        return float(v[:, axis].min()) - float(room_lo[axis])
    return float(room_hi[axis]) - float(v[:, axis].max())


