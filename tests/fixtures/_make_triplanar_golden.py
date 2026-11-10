"""Regenerate `triplanar_uv_golden.json`, the fixture the browser is checked on.

    python tests/fixtures/_make_triplanar_golden.py

Base-color textures are parameterized procedurally rather than stored, so the
Open3D backend and the browser each run their OWN implementation of the rule.
Nothing forces them to agree, and a disagreement is silent: both draw wood, on
different parts of the plank, and the viewer stops being an honest preview of
what the renderer draws.

`viewer/web/scripts/check-triplanar-uv.mjs` runs the shipped TypeScript against
this file; `tests/test_texture_render.py::test_the_triplanar_golden_is_what_the
_renderer_produces` asserts the file is what `_triplanar_uvs` produces. So
neither end can drift without one of the two failing.

Both halves are load-bearing. Keep the Python half in a file that has nothing
to do with whichever producer supplies the textures, so removing a producer
cannot remove it.

Three kinds of case. The axis cases are small enough to read and pin the
dominant-axis choice on each of the six face directions. The oblique case pins
the tie-break behavior where no axis dominates cleanly. The real slice is
lifted from a committed solid's OBJ — the same argument the LBS fixture makes:
real CAD triangles have magnitudes and normal distributions a hand-written
triangle does not reproduce, and the fillets are where the dominant axis
actually flips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
for _path in (REPO, REPO.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
OUT = Path(__file__).resolve().parent / "triplanar_uv_golden.json"
REAL_OBJ = (REPO / "runs" / "earthquake_judge_on_02" / "catalog" / "assets"
            / "meshes" / "72bb54e54869ecc86085a1d0d7376a58.obj")
N_SAMPLE_TRIS = 24


def _case(name, verts, tris, scale_m):
    from dycraft.render.open3d_renderer import _triplanar_uvs

    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    uvs = _triplanar_uvs(verts, tris, scale_m)
    return {
        "name": name,
        "scale_m": scale_m,
        "n_verts": int(len(verts)),
        "n_tris": int(len(tris)),
        "vertices": [round(float(v), 9) for v in verts.reshape(-1)],
        "triangles": [int(t) for t in tris.reshape(-1)],
        "uvs": [round(float(u), 9) for u in np.asarray(uvs).reshape(-1)],
    }


def _axis_cases():
    """One triangle facing each of +-X, +-Y, +-Z, at a distinctive scale.

    Winding is reversed for the negative directions so the normal really does
    point the other way — `argmax(|n|)` is sign-blind on purpose (a face and its
    back take the same projection), and this is what proves that rather than
    assuming it.
    """
    out = []
    unit = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]     # +Z
    out.append(_case("face_pos_z", unit, [[0, 1, 2]], 0.5))
    out.append(_case("face_neg_z", unit, [[0, 2, 1]], 0.5))
    yz = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]       # +-X
    out.append(_case("face_pos_x", yz, [[0, 1, 2]], 1.0))
    out.append(_case("face_neg_x", yz, [[0, 2, 1]], 1.0))
    xz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]       # +-Y
    out.append(_case("face_pos_y", xz, [[0, 1, 2]], 0.25))
    out.append(_case("face_neg_y", xz, [[0, 2, 1]], 0.25))
    return out


def _oblique_case():
    """A 45-degree face, where two axes tie and `argmax` takes the first.

    The seam the whole approach trades away lives exactly here, so the two
    implementations have to break the tie identically or the seam lands in a
    different place in each.
    """
    verts = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _case("oblique_45", verts, [[0, 1, 2]], 0.3)


def _degenerate_case():
    """A zero-area triangle: no normal to normalize, and no pixels either."""
    verts = [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]
    return _case("degenerate", verts, [[0, 1, 2]], 0.5)


def _real_case():
    """A slice of a committed solid's render mesh: a cafe table's top."""
    if not REAL_OBJ.is_file():
        return None
    verts, tris = [], []
    for line in REAL_OBJ.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            tris.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
        if len(tris) >= N_SAMPLE_TRIS:
            break
    if not tris:
        return None
    used = sorted({i for t in tris for i in t})
    remap = {old: new for new, old in enumerate(used)}
    return _case("real_table_top",
                 [verts[i] for i in used],
                 [[remap[i] for i in t] for t in tris],
                 0.22)


def main() -> None:
    cases = _axis_cases() + [_oblique_case(), _degenerate_case()]
    real = _real_case()
    if real is not None:
        cases.append(real)
    else:
        print(f"note: {REAL_OBJ} absent, wrote synthetic cases only")
    # Exact, not approximate. Both sides do the same finite arithmetic in
    # float64 -> float32; a real disagreement is a different RULE, not drift,
    # and a loose tolerance would hide the projection axis being swapped.
    OUT.write_text(json.dumps(
        {"tolerance": 1e-6, "cases": cases}, indent=1) + "\n")
    print(f"wrote {OUT} with {len(cases)} cases")


if __name__ == "__main__":
    main()
