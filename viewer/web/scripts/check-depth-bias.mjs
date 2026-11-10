// Pin the per-visual depth bias so a scene cannot be drawn in the wrong order.
//
//     npm run check:depth-bias
//
// The viewer gives every visual a polygon offset, as a deterministic tie-break
// between coplanar surfaces. Two properties have to hold, and neither did:
//
//   1. The bias must not grow with how many visuals the model has. It was
//      `1 + ordinal * 0.25` over a running count, so the SAME refrigerator got
//      1.00..4.25 in the Object Viewer and 15.25..18.50 as instance 5 of a
//      129-visual kitchen, with the last instance reaching 33.
//   2. It must go to `polygonOffsetUnits` only. `polygonOffsetFactor`
//      multiplies the polygon's depth slope in window space, which is a
//      function of the camera, so a non-zero factor makes the bias slide as
//      the view turns -- unboundedly, as a surface goes edge-on.
//
// A depth-buffer unit is not a fixed distance: it is worth
// `z**2 * (far - near) / (near * far) * 2**-24` of world depth. The kitchen
// scene's camera ran at near = 0.001 / far = 20, because the clip box includes
// the ground grid and the grid is three times the model footprint, so one unit
// was ~3 mm at the model and 33 of them ~10 cm — on surfaces `ir.json` asserts
// are coplanar. The same object alone got near = 0.80 / far = 6.8. So this
// pins the near plane too: the grid must not drag it to a millimeter.
//
// The far plane never comes nearer than DEFAULT_CLIP_FAR (1000 m), so near:far
// can be large without costing resolution at the model: `(far - near) / far` is
// within a thousandth of 1 from far = 20 m. So this checks what matters
// directly: the near plane's floor, and the worst-case displacement at the
// model (section 4).
//
// Transpiled through the esbuild vite already installs, like check:uv.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-depth-bias-"));
const failures = [];

const check = (ok, msg) => { if (!ok) failures.push(msg); };

/** World depth a polygon offset of `units` buys, under the viewer's reversed-Z. */
function worldShiftMm(units, z, near, far) {
  return units * 2 ** -24 * z * z * ((far - near) / (near * far)) * 1000;
}

function bundle(source, name) {
  const out = join(work, name);
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [source, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${out}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  return import(pathToFileURL(out).href);
}

try {
  const V3 = join(WEB, "src", "components", "viewer3d");
  const materials = await bundle(join(V3, "materials.ts"), "materials.mjs");
  const cameraUtils = await bundle(join(V3, "camera-utils.ts"), "camera.mjs");
  const THREE = await import(pathToFileURL(
    join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  const { depthBiasForOrdinal, applyDepthBias, createMaterial } = materials;

  // ---- 1. bounded, and independent of the model's size ------------------
  //
  // The kitchen that produced the bug has 129 visuals; the ceiling has to hold
  // well past that, because nothing stops a scene being bigger.
  let maxBias = 0;
  for (let i = 0; i < 10000; i += 1) maxBias = Math.max(maxBias, depthBiasForOrdinal(i));
  check(maxBias <= 4, `depth bias reaches ${maxBias} over 10000 visuals; it must stay small ` +
                      `(it was 1 + ordinal*0.25, i.e. 2501 here and 33 on the kitchen)`);
  check(depthBiasForOrdinal(5) === depthBiasForOrdinal(5),
        "depthBiasForOrdinal is not a pure function of the ordinal");

  // Adjacent visuals -- one part's own surfaces, which is the case the bias
  // exists for -- must still get distinct values.
  for (let i = 0; i < 8; i += 1) {
    check(depthBiasForOrdinal(i) !== depthBiasForOrdinal(i + 1),
          `visuals ${i} and ${i + 1} share a depth bias, so coplanar surfaces in one ` +
          `part have no tie-break`);
  }

  // ---- 2. units only: nothing view-dependent -----------------------------
  const spec = {
    name: "probe", color: new THREE.Color(0.5, 0.5, 0.5), opacity: 1, metalness: 0,
    roughness: 0.5, transmission: 0, thickness: 0, ior: 1.45, clearcoat: 0,
    clearcoatRoughness: 0.2, envMapIntensity: 0.8,
  };
  for (const ordinal of [0, 1, 7, 42, 128]) {
    const bias = depthBiasForOrdinal(ordinal);
    const mat = createMaterial({ ...spec }, { depthBias: bias });
    check(mat.polygonOffsetFactor === 0,
          `createMaterial(ordinal ${ordinal}) set polygonOffsetFactor=${mat.polygonOffsetFactor}; ` +
          `the factor term scales with the view and must stay 0`);
    check(mat.polygonOffsetUnits === bias,
          `createMaterial(ordinal ${ordinal}) put ${mat.polygonOffsetUnits} in units, expected ${bias}`);
  }
  // The derived-GLB path writes through the same helper; that is the whole
  // point of the helper existing, so check it directly.
  const loaded = new THREE.MeshPhysicalMaterial();
  applyDepthBias(loaded, depthBiasForOrdinal(128));
  check(loaded.polygonOffsetFactor === 0,
        `applyDepthBias set polygonOffsetFactor=${loaded.polygonOffsetFactor}, expected 0`);
  check(loaded.polygonOffset === true, "applyDepthBias left polygonOffset disabled");
  applyDepthBias(loaded, 0);
  check(loaded.polygonOffset === false, "a zero bias should not enable polygonOffset");

  // ---- 3. the ground grid must not set the near plane --------------------
  //
  // Reproduces the kitchen's geometry: a 4.0 x 2.25 x 2.94 m model with the
  // grid scaled to three times its footprint, viewed from 7.77 m.
  const { updateCameraClipping } = cameraUtils;
  const model = new THREE.Mesh(new THREE.BoxGeometry(4.0, 2.25, 2.94));
  model.position.set(0, 1.125, 0);
  const grid = new THREE.Mesh(new THREE.BoxGeometry(12, 0.001, 12));
  for (const o of [model, grid]) o.updateMatrixWorld(true);

  const camera = new THREE.PerspectiveCamera(45, 431 / 299, 0.001, 1000);
  camera.position.set(4.65, 4.07, 4.71);
  camera.lookAt(0, 1.125, 0);
  camera.updateMatrixWorld(true);
  updateCameraClipping(camera, [model, grid]);

  const dist = camera.position.distanceTo(new THREE.Vector3(0, 1.125, 0));
  const ratio = camera.far / camera.near;
  check(camera.near >= 0.01,
        `near is ${camera.near} m; the grid must not be able to drag it to a millimeter and ` +
        `collapse depth resolution`);
  check(camera.far >= cameraUtils.DEFAULT_CLIP_FAR,
        `far is ${camera.far} m; the render distance must not stop at the scene's own box`);
  // The grid still has to be inside the frustum, and the floor on `near` must
  // clip nothing a viewer needs: the nearest point of a 12 m grid under this
  // camera is 4.07 m away, and the near plane has to stay well in front of it.
  check(camera.far >= dist + 6,
        `far is ${camera.far}, too short to still cover the 12 m grid`);
  const nearestGridPoint = Math.hypot(0, camera.position.y, 0);
  check(camera.near < nearestGridPoint,
        `near ${camera.near} would clip the grid passing under the camera at ` +
        `${nearestGridPoint.toFixed(2)} m`);

  // ---- 4. the property all of the above exists to buy --------------------
  //
  // Worst-case world displacement of a surface at the model, from the bias
  // alone.
  const worst = worldShiftMm(maxBias, dist, camera.near, camera.far);
  check(worst < 1.0,
        `worst-case depth displacement is ${worst.toFixed(3)} mm at ${dist.toFixed(2)} m; ` +
        `the kitchen IR asserts surfaces coplanar to +/-5 mm, so the tie-break has to ` +
        `stay well inside that (it was 103 mm)`);

  console.log(`depth bias: max ${maxBias} over 10000 visuals, units-only`);
  console.log(`clip planes with a 3x grid: near ${camera.near.toFixed(3)} far ${camera.far.toFixed(3)} ` +
              `(ratio 1:${Math.round(ratio)}, grid under the camera at ${nearestGridPoint.toFixed(2)} m)`);
  console.log(`worst-case world depth displacement at ${dist.toFixed(2)} m: ${worst.toFixed(4)} mm`);
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-depth-bias: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-depth-bias: ok");
