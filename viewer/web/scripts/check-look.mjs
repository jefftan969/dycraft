// Pin the viewport's look (`lighting.ts` `FILM_LOOK`): where its sun's shadow reaches, and what the room may not shade.
//
//     npm run check:look
//
//   1. **The sun's shadow covers the scene, whatever its size** (`fitSunToBox`): every corner of the drawn scene lies
//      inside the shadow camera's box, for a part a few centimeters across and for a whole building.
//   2. **The room casts no shadow on what is in it** (`roomCastsNoShadow`): the walls and ceilings the cutaway knows,
//      what is mounted on them, and anything at or above a ceiling's underside cast none; a table and a lamp hung
//      below the ceiling still do. In the viewport's frame, which the film's (Z up) is carried into.
//
// Transpiled through the esbuild vite already installs, like check:cutaway.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
// Under node_modules, so the bundles and this script share one three (see check-native-track).
const work = mkdtempSync(join(WEB, "node_modules", ".dycraft-look-"));
const failures = [];
const check = (ok, msg) => { if (!ok) failures.push(msg); };

function bundle(source, name) {
  const out = join(work, name);
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [source, "--bundle", "--format=esm", "--platform=node", "--log-level=error", "--external:three",
     "--define:import.meta.env={}",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${out}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  return import(pathToFileURL(out).href);
}

try {
  const V3 = join(WEB, "src", "components", "viewer3d");
  const L = await bundle(join(V3, "lighting.ts"), "lighting.mjs");
  const C = await bundle(join(V3, "useCutaway.ts"), "useCutaway.mjs");
  const THREE = await import(pathToFileURL(join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  for (const [label, min, max] of [["a part", [0.1, 0, -0.05], [0.18, 0.06, 0.02]],
                                   ["a room", [-3, 0, -4], [5, 2.6, 4]],
                                   ["a building", [-60, -1, -40], [60, 30, 45]]]) {
    const rig = L.createLightingRig();
    const box = new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
    L.fitSunToBox(rig, box);
    const sun = rig.getObjectByName("key-light");
    const cam = sun.shadow.camera;
    cam.position.copy(sun.position);
    cam.lookAt(sun.target.position);
    cam.updateMatrixWorld();
    const inv = cam.matrixWorld.clone().invert();
    let inside = true;
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
      const p = new THREE.Vector3(x, y, z).applyMatrix4(inv);
      inside &&= p.x >= cam.left - 1e-9 && p.x <= cam.right + 1e-9 && p.y >= cam.bottom - 1e-9
        && p.y <= cam.top + 1e-9 && -p.z >= cam.near - 1e-9 && -p.z <= cam.far + 1e-9;
    }
    check(inside, `${label}: a corner of the scene is outside the sun's shadow`);
    const dir = sun.position.clone().sub(sun.target.position).normalize();
    check(dir.distanceTo(L.FILM_LOOK.sun.direction) < 1e-9, `${label}: the sun's direction moved`);
    check(sun.shadow.normalBias > 0 && sun.shadow.bias <= 0, `${label}: shadow bias not scaled to the shadow's texel`);
  }

  // The film's frame (Z up) carried into the viewport's (Y up), as the robot group does.
  const group = new THREE.Group();
  group.rotation.x = -Math.PI / 2;
  group.updateMatrixWorld(true);
  const box = (name, min, max) => {
    const geometry = new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2])
      .translate((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const mesh = new THREE.Mesh(geometry);
    mesh.name = name;
    mesh.castShadow = true;
    group.add(mesh);
    return mesh;
  };
  const meshes = {
    wall: box("wall", [-2, -2, 0], [2, -1.9, 2.5]),
    ceiling: box("ceiling", [-2, -2, 2.5], [2, 2, 2.6]),
    window: box("window", [-0.5, -2, 1], [0.5, -1.9, 2]),
    beam: box("beam", [-2, -0.1, 2.45], [2, 0.1, 2.5]),
    table: box("table", [-0.5, -0.5, 0], [0.5, 0.5, 0.75]),
    lamp: box("lamp", [-0.1, -0.1, 1.9], [0.1, 0.1, 2.4]),
  };
  group.updateMatrixWorld(true);
  const data = {
    units: [{ kind: "wall", objects: ["wall"] }, { kind: "ceiling", objects: ["ceiling"], under: 2.5 }],
    attach: [{ objects: ["window"], units: [0] }],
  };
  C.roomCastsNoShadow(data, group, new Map(group.children.map((m) => [m.name, m])));
  for (const name of ["wall", "ceiling", "window", "beam"]) {
    check(!meshes[name].castShadow, `the room's ${name} still casts a shadow on what is in it`);
  }
  for (const name of ["table", "lamp"]) check(meshes[name].castShadow, `the ${name} in the room stopped casting`);
  console.log("look: the sun's shadow fitted to 3 scenes; the room's 4 pieces cast none, its 2 contents still do");
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-look: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-look: ok");
