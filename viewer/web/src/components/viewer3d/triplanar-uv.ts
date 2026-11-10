import * as THREE from 'three';

/**
 * Triplanar UV generation — the browser half of dycraft's texture parameterization.
 *
 * A base-color texture is not stored with the mesh: no OBJ this project writes
 * carries `vt`, and adding it would change every solid's mesh digest and with it
 * the id of every object that has one. UVs are DERIVED instead, from the same
 * rule in both renderers, which is why this file exists at all rather than the
 * browser reading what Python wrote.
 *
 * The rule: project each triangle onto whichever world plane its own normal
 * points most strongly along — a +Z-facing face takes (x, y), a +X-facing face
 * takes (y, z) — and divide by `scaleM`, so one tile of the image spans
 * `scaleM` meters of surface whatever the part's size.
 *
 * Two properties this has to preserve, both of which the golden fixture pins:
 *
 *  * **`argmax(|n|)` is sign-blind.** A face and its back take the same
 *    projection, so a reversed winding must not move the texture.
 *  * **A tie goes to the lowest axis index.** At 45 degrees two axes are equally
 *    dominant and the seam lands wherever the tie-break puts it. If the two
 *    implementations broke it differently the seam would sit in a different
 *    place in the viewport than in the canonical render.
 *
 * **The two APIs disagree about what a UV belongs to, and that decides the
 * whole port.** Open3D's `triangle_uvs` is per CORNER — 3 per triangle, so a
 * shared vertex carries a different UV on each face it belongs to. three's `uv`
 * attribute is per VERTEX, which cannot represent that: on indexed geometry the
 * last triangle to touch a corner wins, and every face along a seam inherits its
 * neighbor's projection. The two layouts coincide only when every triangle owns
 * its vertices, so `applyTriplanarUVs` de-indexes first. OBJLoader output already
 * is non-indexed; three's own primitives are not.
 *
 * Kept in sync with `render/open3d_renderer.py::_triplanar_uvs` by
 * `viewer/web/scripts/check-triplanar-uv.mjs`, which runs this code against
 * `tests/fixtures/triplanar_uv_golden.json`.
 */

/** The two world axes that are not `axis`: X -> (Y, Z), Y -> (X, Z), Z -> (X, Y). */
const U_AXIS = [1, 0, 0];
const V_AXIS = [2, 2, 1];

/**
 * Per-corner UVs for an indexed or non-indexed triangle soup.
 *
 * `positions` is a flat xyz array in the part's LOCAL frame — the same frame the
 * Python side uses, and the reason both call sites here run before the object's
 * world transform is applied. Derived from world coordinates instead, the same
 * table would show different grain depending on where it stands in a room.
 *
 * Returns a flat uv array of `2 * (positions.length / 3)` entries, aligned to
 * the position array, so it drops straight into a `uv` BufferAttribute whether
 * the geometry is indexed or not.
 */
export function triplanarUVs(
  positions: ArrayLike<number>,
  indices: ArrayLike<number> | null,
  scaleM: number,
): Float32Array {
  const nVerts = Math.floor(positions.length / 3);
  const uvs = new Float32Array(nVerts * 2);
  if (nVerts === 0 || !(scaleM > 0)) {
    return uvs;
  }

  const nTris = indices ? Math.floor(indices.length / 3) : Math.floor(nVerts / 3);
  for (let t = 0; t < nTris; t += 1) {
    const i0 = indices ? indices[t * 3] : t * 3;
    const i1 = indices ? indices[t * 3 + 1] : t * 3 + 1;
    const i2 = indices ? indices[t * 3 + 2] : t * 3 + 2;

    const ax = positions[i0 * 3], ay = positions[i0 * 3 + 1], az = positions[i0 * 3 + 2];
    const bx = positions[i1 * 3], by = positions[i1 * 3 + 1], bz = positions[i1 * 3 + 2];
    const cx = positions[i2 * 3], cy = positions[i2 * 3 + 1], cz = positions[i2 * 3 + 2];

    // n = (b - a) x (c - a). Not normalized: `argmax(|n|)` is scale-invariant,
    // and a zero-area triangle would otherwise divide by zero here rather than
    // simply falling through to axis 0 with no pixels to cover.
    const nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    const nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

    const lx = Math.abs(nx), ly = Math.abs(ny), lz = Math.abs(nz);
    // `>` not `>=` at both comparisons: ties fall to the earlier axis, matching
    // numpy's `argmax`. This is the 45-degree seam's placement.
    let axis = 0;
    if (ly > lx) axis = 1;
    if (lz > (axis === 0 ? lx : ly)) axis = 2;

    const ua = U_AXIS[axis];
    const va = V_AXIS[axis];
    for (const idx of [i0, i1, i2]) {
      uvs[idx * 2] = positions[idx * 3 + ua] / scaleM;
      uvs[idx * 2 + 1] = positions[idx * 3 + va] / scaleM;
    }
  }
  return uvs;
}

/**
 * Give `mesh` triplanar UVs, de-indexing its geometry first if it has to.
 *
 * Takes the mesh rather than the geometry because de-indexing produces a NEW
 * BufferGeometry, and handing back a detached one is the kind of API that gets
 * called for its side effect and silently does nothing. See the header for why
 * de-indexing is required rather than merely tidy.
 *
 * Replacing whatever UVs arrived matters for three's own primitives:
 * `BoxGeometry` ships per-face 0..1 UVs, which tile one whole image onto every
 * face regardless of its size — a 2 m tabletop and a 5 cm knob would show the
 * same single plank.
 */
export function applyTriplanarUVs(mesh: THREE.Mesh, scaleM: number): void {
  let geometry = mesh.geometry;
  if (!geometry?.getAttribute('position')) {
    return;
  }
  if (geometry.getIndex()) {
    const deindexed = geometry.toNonIndexed();
    geometry.dispose();
    mesh.geometry = deindexed;
    geometry = deindexed;
  }
  const position = geometry.getAttribute('position');
  const uvs = triplanarUVs(position.array as ArrayLike<number>, null, scaleM);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
}
