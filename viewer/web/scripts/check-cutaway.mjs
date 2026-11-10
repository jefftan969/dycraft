// Pin the free camera's cutaway: what it hides from where the camera stands.
//
//     npm run check:cutaway
//
// `cutaway.ts` decides, each time the camera moves, which of a film's walls and ceilings (`viewer/cutaway.py`) are in
// the way. What has to hold, on a 4 x 4 m room with a wall on each side, a ceiling, and a window in the south wall:
//
//   1. **The wall in the way goes, and only it.** From outside the south wall, looking at the room's middle, the south
//      wall is hidden and the north one is not.
//   2. **Inside, nothing goes.** From the room's middle the camera sees the action without crossing a wall.
//   3. **A ceiling goes once the camera is above it,** and comes back 25 cm below it, not at once.
//   4. **Hidden walls need wider margins to come back,** so a camera sitting near an edge does not make it blink.
//   5. **What is mounted goes with its wall,** and stays while any wall it hangs on stays.
//   6. **Only what the camera looks toward counts:** looking away from the room, nothing is hidden.
//   7. **The carry into the viewport:** the same answers through a z-up rotation and an offset.
//
// Transpiled through the esbuild vite already installs, like check:film-camera.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-cutaway-"));
const failures = [];
const check = (ok, msg) => { if (!ok) failures.push(msg); };

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
  const C = await bundle(join(WEB, "src", "components", "viewer3d", "cutaway.ts"), "cutaway.mjs");
  const THREE = await import(pathToFileURL(join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  // Blender's frame, Z up. Walls 10 cm thick, 2.6 m tall, around x, y in [-2, 2]; the ceiling's underside at 2.6 m.
  const wall = (name, cx, cy, nx, ny, length) => ({ kind: "wall", objects: [name], box: [cx, cy, 1.3, nx, ny, 0.05, length / 2, 1.3] });
  const cut = {
    version: 1, animation: "a",
    action: [[-1, -1, 0], [1, 1, 1.6]],
    units: [wall("south", 0, -2, 0, 1, 4), wall("north", 0, 2, 0, 1, 4), wall("east", 2, 0, 1, 0, 4),
            wall("west", -2, 0, 1, 0, 4),
            { kind: "ceiling", objects: ["ceiling"], box: [-2, -2, 2, 2], under: 2.6 }],
    attach: [{ objects: ["window"], units: [0] }, { objects: ["corner_shelf"], units: [0, 3] }],
    kept: [],
  };
  const names = cut.units.map((u) => u.objects[0]);

  function look(toViewer, eye, at, hidden = []) {
    const camera = new THREE.PerspectiveCamera(50, 1.5, 0.01, 1000);
    const e = new THREE.Vector3(...eye).applyMatrix4(toViewer);
    const a = new THREE.Vector3(...at).applyMatrix4(toViewer);
    camera.position.copy(e);
    camera.up.set(0, 0, 1).transformDirection(toViewer);   // Blender's up, in the viewport
    camera.lookAt(a);
    camera.updateMatrixWorld(true);
    return C.cutawayState(cut, camera, toViewer, a, hidden);
  }
  const hiddenNames = (s) => names.filter((_, i) => s.units[i]);

  for (const [label, toViewer] of [
    ["Blender's frame", new THREE.Matrix4()],
    ["carried into the viewport", new THREE.Matrix4().makeRotationX(-Math.PI / 2)
      .premultiply(new THREE.Matrix4().makeTranslation(0.4, -0.2, 1.1))],
  ]) {
    let s = look(toViewer, [0, -6, 1.6], [0, 0, 0.8]);
    check(JSON.stringify(hiddenNames(s)) === '["south"]', `${label}: from the south, hid ${hiddenNames(s)}`);
    check(s.attach[0] === true && s.attach[1] === false,
          `${label}: the window should go with the south wall, the corner shelf stay with the west one (${s.attach})`);

    s = look(toViewer, [0, 0, 1.6], [0, 1.5, 1.0]);
    check(hiddenNames(s).length === 0, `${label}: inside the room, hid ${hiddenNames(s)}`);

    s = look(toViewer, [0.3, -0.5, 3.4], [0, 0, 0.8]);
    check(hiddenNames(s).includes("ceiling"), `${label}: above the ceiling, it should go (${hiddenNames(s)})`);
    const above = s.units;
    s = look(toViewer, [0.3, -0.5, 2.5], [0, 0, 0.8], above);
    check(hiddenNames(s).includes("ceiling"), `${label}: 10 cm under the ceiling, it should still be gone`);
    s = look(toViewer, [0.3, -0.5, 2.3], [0, 0, 0.8], above);
    check(!hiddenNames(s).includes("ceiling"), `${label}: 30 cm under the ceiling, it should be back`);

    // Just inside the south wall's plane, beside it: hidden once, it stays while within half a meter.
    s = look(toViewer, [2.25, -6, 1.6], [2.25, 0, 1.0]);
    const first = hiddenNames(s);
    s = look(toViewer, [2.3, -6, 1.6], [2.3, 0, 1.0], cut.units.map((u) => u.objects[0] === "south"));
    check(hiddenNames(s).includes("south"), `${label}: a hidden wall came back at the edge of its margin (${first} -> ${hiddenNames(s)})`);

    s = look(toViewer, [0, -6, 1.6], [0, -12, 1.6]);
    check(hiddenNames(s).length === 0, `${label}: looking away from the room, hid ${hiddenNames(s)}`);
  }
  console.log(`cutaway: ${names.length} units, 2 frames, 7 situations each`);
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-cutaway: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-cutaway: ok");
