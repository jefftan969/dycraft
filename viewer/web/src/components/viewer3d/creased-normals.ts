// Creased normals for an indexed triangle mesh: three's `toCreasedNormals` method, with vertices merged only where
// they coincide.
//
// Each corner's normal is the normalized sum of the face normals that meet at its position within the crease angle,
// summed in three's order. The corners that share a vertex and that normal are then drawn as one vertex, so the mesh
// comes back indexed: about half the memory of one vertex per corner, with every corner's normal unchanged. three
// merges positions on a 1 cm grid, which on a part a few centimeters across merges thousands of unrelated vertices:
// their normals average across surfaces that do not touch, which washes out the shading, and the per-corner comparison
// inside such a group is quadratic (seconds for one 40k-triangle mesh). Here the grid is 0.01 mm, so a group is the
// vertices at one point, and each vertex is hashed once into an integer table rather than each corner into a string.
//
// Dependency-free, so the loader worker and `check-native-track.mjs` import it without three.

/** Positions closer than 0.01 mm (in scene units of meters) count as one point. */
const HASH_MULTIPLIER = (1 + 1e-10) * 1e5;

export interface CreasedMesh {
  /** One position per output vertex: an input vertex, once per distinct normal its corners take. */
  position: Float32Array;
  /** One unit normal per output vertex, or zero where it serves only degenerate triangles. */
  normal: Float32Array;
  /** Three output vertices per input triangle, in the input's triangle and corner order. */
  index: Uint32Array;
}

export function creasedNormals(
  positions: Float32Array,
  indices: Uint32Array,
  creaseAngle: number,
): CreasedMesh {
  const creaseDot = Math.cos(creaseAngle);
  const nVertices = positions.length / 3;
  const nTriangles = indices.length / 3;

  // Each vertex's group: the vertices sharing its quantized position.
  const group = new Int32Array(nVertices);
  let capacity = 16;
  while (capacity < nVertices * 2) capacity *= 2;
  const mask = capacity - 1;
  const keys = new Int32Array(capacity * 3);
  const slots = new Int32Array(capacity).fill(-1);
  let nGroups = 0;
  for (let v = 0; v < nVertices; v += 1) {
    const x = ~~(positions[3 * v] * HASH_MULTIPLIER);
    const y = ~~(positions[3 * v + 1] * HASH_MULTIPLIER);
    const z = ~~(positions[3 * v + 2] * HASH_MULTIPLIER);
    let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
    for (;;) {
      const slot = slots[h];
      if (slot === -1) {
        slots[h] = nGroups;
        keys[3 * h] = x;
        keys[3 * h + 1] = y;
        keys[3 * h + 2] = z;
        group[v] = nGroups;
        nGroups += 1;
        break;
      }
      if (keys[3 * h] === x && keys[3 * h + 1] === y && keys[3 * h + 2] === z) {
        group[v] = slot;
        break;
      }
      h = (h + 1) & mask;
    }
  }

  // Face normals, as `Vector3.crossVectors(c - b, a - b).normalize()`.
  const face = new Float64Array(nTriangles * 3);
  for (let t = 0; t < nTriangles; t += 1) {
    const a = 3 * indices[3 * t];
    const b = 3 * indices[3 * t + 1];
    const c = 3 * indices[3 * t + 2];
    const ux = positions[c] - positions[b];
    const uy = positions[c + 1] - positions[b + 1];
    const uz = positions[c + 2] - positions[b + 2];
    const vx = positions[a] - positions[b];
    const vy = positions[a + 1] - positions[b + 1];
    const vz = positions[a + 2] - positions[b + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    // `normalize()` multiplies by the reciprocal, which can differ from dividing in the last bit.
    const inverse = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
    face[3 * t] = nx * inverse;
    face[3 * t + 1] = ny * inverse;
    face[3 * t + 2] = nz * inverse;
  }

  // Each group's corners, in triangle-then-corner order, which is the order three sums them in.
  const start = new Int32Array(nGroups + 1);
  for (let i = 0; i < indices.length; i += 1) start[group[indices[i]] + 1] += 1;
  for (let g = 0; g < nGroups; g += 1) start[g + 1] += start[g];
  const fill = start.slice(0, nGroups);
  const faces = new Int32Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    const g = group[indices[i]];
    faces[fill[g]] = (i / 3) | 0;
    fill[g] += 1;
  }

  // Output vertices, at most one per corner: per input vertex, a chain of the normals its corners have taken.
  const position = new Float32Array(nTriangles * 9);
  const normal = new Float32Array(nTriangles * 9);
  const index = new Uint32Array(nTriangles * 3);
  const head = new Int32Array(nVertices).fill(-1);
  const next = new Int32Array(nTriangles * 3);
  const corner = new Float32Array(3);
  let nOut = 0;
  for (let t = 0; t < nTriangles; t += 1) {
    const fx = face[3 * t];
    const fy = face[3 * t + 1];
    const fz = face[3 * t + 2];
    for (let k = 0; k < 3; k += 1) {
      const v = indices[3 * t + k];
      const g = group[v];
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let j = start[g]; j < start[g + 1]; j += 1) {
        const f = 3 * faces[j];
        if (fx * face[f] + fy * face[f + 1] + fz * face[f + 2] > creaseDot) {
          sx += face[f];
          sy += face[f + 1];
          sz += face[f + 2];
        }
      }
      const inverse = 1 / (Math.sqrt(sx * sx + sy * sy + sz * sz) || 1);
      // Rounded to float32 first, so two corners match exactly when their stored normals would.
      corner[0] = sx * inverse;
      corner[1] = sy * inverse;
      corner[2] = sz * inverse;
      let out = head[v];
      while (out !== -1 && (normal[3 * out] !== corner[0] || normal[3 * out + 1] !== corner[1]
                            || normal[3 * out + 2] !== corner[2])) {
        out = next[out];
      }
      if (out === -1) {
        out = nOut;
        nOut += 1;
        position[3 * out] = positions[3 * v];
        position[3 * out + 1] = positions[3 * v + 1];
        position[3 * out + 2] = positions[3 * v + 2];
        normal[3 * out] = corner[0];
        normal[3 * out + 1] = corner[1];
        normal[3 * out + 2] = corner[2];
        next[out] = head[v];
        head[v] = out;
      }
      index[3 * t + k] = out;
    }
  }
  return { position: position.slice(0, 3 * nOut), normal: normal.slice(0, 3 * nOut), index };
}
