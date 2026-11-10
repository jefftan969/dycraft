// The free camera's cutaway: which walls and ceilings to hide from where the camera stands.
//
// `viewer/cutaway.py` decides which of a film's objects are walls and ceilings, from its geometry alone. This decides,
// each time the camera moves, which of them are in the way: a ceiling once the camera rises above its underside; a
// wall once the line from the camera to the action crosses it, or the camera stands in it or just behind it. The action
// is 13 points over the region the film's motion covers, plus the orbit pivot kept within it, and only the points the
// camera looks toward count, so turning around inside a room hides nothing behind or beside it. A hidden piece needs
// wider margins to come back (its slab grown by half a meter, the points up to 60% beyond the frame's edge, 25 cm below
// a ceiling), so nothing blinks while the camera sits near an edge. What is mounted on a wall (a window) hides while
// every wall it hangs on does.
//
// Ported from the project page's viewer. Everything here is in Blender's world frame, Z up: the caller carries the
// camera into it.
//
// Pure, apart from the `THREE` types, so `npm run check:cutaway` can pin it without a DOM.

import * as THREE from 'three';

export type CutawayWall = {
  kind: 'wall';
  objects: string[];
  /** The slab: center (x, y, z), horizontal normal (nx, ny), half sizes along the normal, along the wall, and up. */
  box: [number, number, number, number, number, number, number, number];
};

export type CutawayCeiling = {
  kind: 'ceiling';
  objects: string[];
  /** Its extent from above: x0, y0, x1, y1. */
  box: [number, number, number, number];
  /** Its underside's height. */
  under: number;
};

export type CutawayUnit = CutawayWall | CutawayCeiling;

/** `/api/runs/<id>/cutaway`. */
export type Cutaway = {
  version: number;
  animation: string;
  /** The region the film's motion covers, [min, max]; null when nothing moves. */
  action: [number[], number[]] | null;
  units: CutawayUnit[];
  attach: { objects: string[]; units: number[] }[];
  kept: { objects: string[]; why: string }[];
};

type P3 = [number, number, number];

/** The center, the eight corners, and the six face centers of the action's box, each at 80% of the way out. */
export function actionPoints(action: [number[], number[]], pivot?: P3 | null): P3[] {
  const [lo, hi] = action;
  const mid = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const half = [0, 1, 2].map((k) => (hi[k] - lo[k]) / 2);
  const at = (x: number, y: number, z: number): P3 => [mid[0] + x * half[0], mid[1] + y * half[1], mid[2] + z * half[2]];
  const points: P3[] = [at(0, 0, 0)];
  for (let i = 0; i < 8; i += 1) points.push(at(i & 1 ? 0.8 : -0.8, i & 2 ? 0.8 : -0.8, i & 4 ? 0.8 : -0.8));
  points.push(at(0.8, 0, 0), at(-0.8, 0, 0), at(0, 0.8, 0), at(0, -0.8, 0));
  if (pivot) points.push([0, 1, 2].map((k) => Math.min(hi[k], Math.max(lo[k], pivot[k]))) as P3);
  return points;
}

/** Whether `unit` is in the way from `eye` toward `points`; `hidden`, its state now, widens the margins to return. */
export function inTheWay(unit: CutawayUnit, eye: P3, points: P3[], hidden: boolean): boolean {
  if (unit.kind === 'ceiling') return eye[2] > unit.under + (hidden ? -0.25 : 0.05);
  const [cx, cy, cz, nx, ny, hn, ht, hz] = unit.box;
  const m = hidden ? 0.5 : 0.05;
  const H = [hn + m, ht + m, hz + m];
  const frame = (p: P3) => {
    const dx = p[0] - cx, dy = p[1] - cy;
    return [dx * nx + dy * ny, -dx * ny + dy * nx, p[2] - cz];
  };
  const P = frame(eye);
  // In it, or just behind it.
  if (Math.abs(P[0]) <= hn + (hidden ? 0.4 : 0.15) && Math.abs(P[1]) <= H[1] && Math.abs(P[2]) <= H[2]) return true;
  for (const q of points) {
    const Q = frame(q);
    if (Math.abs(Q[0]) <= hn && Math.abs(Q[1]) <= ht && Math.abs(Q[2]) <= hz) continue;   // on or in it
    // The segment from the camera to the point, clipped against the grown slab.
    let t0 = 0, t1 = 1;
    for (let k = 0; k < 3 && t0 <= t1; k += 1) {
      const d = Q[k] - P[k];
      if (Math.abs(d) < 1e-9) {
        if (Math.abs(P[k]) > H[k]) t0 = 2;
        continue;
      }
      let ta = (-H[k] - P[k]) / d, tb = (H[k] - P[k]) / d;
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta);
      t1 = Math.min(t1, tb);
    }
    if (t0 <= t1) return true;
  }
  return false;
}

/**
 * Which units and mounted objects to hide, seen through `camera` (in the viewport) from where it stands.
 *
 * `toViewer` carries Blender's world into the viewport (the clip's group's matrixWorld); `pivot` is the orbit target,
 * in the viewport. `hidden` is each unit's state now, for the margins; the result is the next state.
 */
export function cutawayState(
  cutaway: Cutaway,
  camera: THREE.Camera,
  toViewer: THREE.Matrix4,
  pivot: THREE.Vector3 | null,
  hidden: readonly boolean[],
): { units: boolean[]; attach: boolean[] } {
  const toBlender = toViewer.clone().invert();
  const eyeV = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld).applyMatrix4(toBlender);
  const eye: P3 = [eyeV.x, eyeV.y, eyeV.z];
  const piv = pivot ? pivot.clone().applyMatrix4(toBlender) : null;
  const points = cutaway.action ? actionPoints(cutaway.action, piv ? [piv.x, piv.y, piv.z] : null) : [];
  // Only the points the camera looks toward: within the frame (near), or a little beyond it (far, to come back).
  const near: P3[] = [], far: P3[] = [];
  const v = new THREE.Vector3();
  for (const p of points) {
    v.set(p[0], p[1], p[2]).applyMatrix4(toViewer).project(camera);
    if (!(v.z > -1 && v.z < 1)) continue;                 // behind the camera
    const r = Math.max(Math.abs(v.x), Math.abs(v.y));
    if (r <= 1.6) far.push(p);
    if (r <= 1.15) near.push(p);
  }
  const units = cutaway.units.map((u, i) => inTheWay(u, eye, hidden[i] ? far : near, Boolean(hidden[i])));
  const attach = cutaway.attach.map((a) => a.units.length > 0 && a.units.every((i) => units[i]));
  return { units, attach };
}
