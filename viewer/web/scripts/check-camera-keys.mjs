// Pin the viewport's creative-mode flight: what the mouse, the keys, and the wheel do to the camera.
//
//     npm run check:camera-keys
//
// The camera flies as in Minecraft's creative mode (`camera-keys.ts`). What has to hold, most of it silent when
// broken, since a camera that drifts or speeds up looks like the scene's fault:
//
//   1. **The bindings.** WASD and the arrows fly, Space/E rise, Shift/Q sink; opposites cancel, doubles do not add.
//   2. **W follows the horizontal heading.** Looking down, W and S keep their height; A and D strafe level and to
//      the right sides; the vertical keys are plumb.
//   3. **One speed wherever the camera is.** The keys' velocity is the scene's speed whatever the camera's distance
//      from anything, which is what the user asked for; diagonals are no faster than an axis.
//   4. **A little easing, never past the keys' speed,** and a long frame (a backgrounded tab) does not teleport.
//   5. **The mouse looks.** Dragging right turns the view right, up turns it up; pitch stops short of vertical and
//      the camera never rolls.
//   6. **The wheel steps a fixed distance,** back for a positive delta, in proportion to it (a trackpad glides),
//      at most three notches at once.
//   7. **Degenerate views give numbers.** Looking straight down still has a horizontal heading.
//
// Transpiled through the esbuild vite already installs, like check:depth-bias.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-camera-keys-"));
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
  const V3 = join(WEB, "src", "components", "viewer3d");
  const keys = await bundle(join(V3, "camera-keys.ts"), "camera-keys.mjs");
  const THREE = await import(pathToFileURL(
    join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  const {
    easeVelocity, EASE_SECONDS, flySpeed, heading, isMoveKey, keyVelocity, lookQuaternion, moveAxesFromKeys,
    stepDisplacement, wheelStep,
  } = keys;

  /** A camera looking at `target` from `position`. */
  const view = (position, target = [0, 0, 0]) => {
    const camera = new THREE.PerspectiveCamera(45, 1.4, 0.05, 50);
    camera.position.set(...position);
    camera.lookAt(...target);
    camera.updateMatrixWorld(true);
    return camera;
  };
  const SPEED = 2.5;

  // ---- 1. the bindings ---------------------------------------------------
  const expected = {
    KeyW: "forward+", ArrowUp: "forward+",
    KeyS: "forward-", ArrowDown: "forward-",
    KeyA: "right-", ArrowLeft: "right-",
    KeyD: "right+", ArrowRight: "right+",
    KeyE: "up+", Space: "up+",
    KeyQ: "up-", ShiftLeft: "up-", ShiftRight: "up-",
  };
  for (const [code, want] of Object.entries(expected)) {
    check(isMoveKey(code), `${code} is not bound`);
    const axes = moveAxesFromKeys([code]);
    const got = ["forward", "right", "up"]
      .filter((axis) => axes[axis] !== 0)
      .map((axis) => `${axis}${axes[axis] > 0 ? "+" : "-"}`)
      .join(",");
    check(got === want, `${code} maps to ${got || "nothing"}, expected ${want}`);
  }
  check(!isMoveKey("KeyR") && !isMoveKey("KeyK") && !isMoveKey("Tab"),
        "a key that is not a flight key claims to be one, so the handler would eat it");
  const still = moveAxesFromKeys(["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"]);
  check(still.forward === 0 && still.right === 0 && still.up === 0,
        `holding every direction moves ${JSON.stringify(still)}, expected a standstill`);
  check(moveAxesFromKeys(["KeyW", "ArrowUp"]).forward === 1, "W and Up together are faster than W");

  // ---- 2. the heading ----------------------------------------------------
  const camera = view([4, 3, 5]);                         // three-quarter view, looking down
  const flat = camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
  const w = keyVelocity(camera, moveAxesFromKeys(["KeyW"]), SPEED);
  check(Math.abs(w.y) < 1e-12, `W looking down moved y=${w.y}; it must keep its height, as in creative mode`);
  check(w.clone().normalize().dot(flat) > 1 - 1e-9, "W does not follow the horizontal heading");
  check(keyVelocity(camera, moveAxesFromKeys(["KeyS"]), SPEED).add(w).length() < 1e-9, "S is not the opposite of W");
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const d = keyVelocity(camera, moveAxesFromKeys(["KeyD"]), SPEED);
  check(Math.abs(d.y) < 1e-12 && Math.abs(d.clone().normalize().dot(flat)) < 1e-9, "D does not strafe level");
  check(d.dot(right) > 0, "D moves to the camera's left");
  for (const [code, sign] of [["KeyE", 1], ["Space", 1], ["KeyQ", -1], ["ShiftLeft", -1]]) {
    const v = keyVelocity(camera, moveAxesFromKeys([code]), SPEED);
    check(Math.abs(v.x) < 1e-12 && Math.abs(v.z) < 1e-12 && Math.sign(v.y) === sign,
          `${code} moved ${v.toArray()}; it must be plumb, sign ${sign}`);
  }

  // ---- 3. one speed --------------------------------------------------------
  for (const position of [[0.4, 0.3, 0.5], [4, 3, 5], [40, 30, 50]]) {
    const v = keyVelocity(view(position), moveAxesFromKeys(["KeyW"]), SPEED).length();
    check(Math.abs(v - SPEED) < 1e-9, `W at ${position} flies ${v.toFixed(3)} m/s, expected ${SPEED} wherever it is`);
  }
  const diagonal = keyVelocity(camera, moveAxesFromKeys(["KeyW", "KeyD"]), SPEED).length();
  check(diagonal <= SPEED + 1e-9, `a diagonal flies ${(diagonal / SPEED).toFixed(3)}x the speed`);
  check(flySpeed(0.08) >= 0.05 && flySpeed(40) <= 20 && flySpeed(4) < flySpeed(40) && flySpeed(NaN) === 1,
        "the scene's speed is out of its bounds or not monotone");

  // ---- 4. easing -----------------------------------------------------------
  const target = new THREE.Vector3(SPEED, 0, 0);
  let v = new THREE.Vector3();
  v = easeVelocity(v, target, EASE_SECONDS);
  check(Math.abs(v.x / SPEED - (1 - Math.exp(-1))) < 1e-9, `after ${EASE_SECONDS} s the velocity is ${v.x}`);
  for (let i = 0; i < 60; i += 1) v = easeVelocity(v, target, 1 / 60);
  check(v.x > SPEED * 0.999 && v.x <= SPEED + 1e-12, `after a second the velocity is ${v.x}, never past ${SPEED}`);
  check(stepDisplacement(target, 5).length() <= SPEED * 0.1 + 1e-9, "a 5 s frame moved more than 0.1 s of flight");
  check(stepDisplacement(target, -1).length() === 0, "a negative frame moved the camera");

  // ---- 5. the mouse looks -------------------------------------------------
  const before = camera.getWorldDirection(new THREE.Vector3());
  const turned = camera.clone();
  turned.quaternion.copy(lookQuaternion(camera, 40, 0, 600));
  turned.updateMatrixWorld(true);
  check(turned.getWorldDirection(new THREE.Vector3()).dot(right) > 0, "dragging right did not turn the view right");
  const raised = camera.clone();
  raised.quaternion.copy(lookQuaternion(camera, 0, -40, 600));
  raised.updateMatrixWorld(true);
  check(raised.getWorldDirection(new THREE.Vector3()).y > before.y, "dragging up did not turn the view up");
  const vertical = camera.clone();
  vertical.quaternion.copy(lookQuaternion(camera, 0, -1e6, 600));
  vertical.updateMatrixWorld(true);
  const up = vertical.getWorldDirection(new THREE.Vector3());
  check(up.y < 1 && up.y > 0.99, `pitch went to ${up.y}; it must stop short of straight up`);
  for (const c of [turned, raised, vertical]) {
    const r = new THREE.Vector3().setFromMatrixColumn(c.matrixWorld, 0);
    check(Math.abs(r.y) < 1e-9, `looking rolled the camera (its right vector has y=${r.y})`);
  }

  // ---- 6. the wheel --------------------------------------------------------
  const back = wheelStep(camera, 100, SPEED);
  check(Math.abs(back.length() - SPEED * 0.25) < 1e-9, `a notch moved ${back.length()}, expected a quarter second`);
  check(back.dot(before) < 0, "a positive wheel delta moved forward");
  check(wheelStep(view([40, 30, 50]), 100, SPEED).length() === back.length(), "the wheel's step depends on where it is");
  check(wheelStep(camera, 10000, SPEED).length() <= SPEED * 0.75 + 1e-9, "a huge wheel delta stepped more than three notches");
  check(Math.abs(wheelStep(camera, 2, SPEED).length() - SPEED * 0.25 * 0.02) < 1e-9,
        "a trackpad's 2-pixel delta must move a fiftieth of a notch, not a whole one");

  // ---- 7. degenerate views -------------------------------------------------
  const overhead = view([0, 6, 0]);
  const h = heading(overhead);
  check([h.x, h.y, h.z].every(Number.isFinite) && Math.abs(h.y) < 1e-12 && h.length() > 0.99,
        `looking straight down gave the heading ${h.toArray()}`);
  for (const combo of [["KeyW"], ["KeyD"], ["Space"]]) {
    const step = keyVelocity(overhead, moveAxesFromKeys(combo), SPEED);
    check(step.toArray().every(Number.isFinite) && step.length() > 0,
          `${combo.join("+")} looking straight down gave ${step.toArray()}`);
  }
  console.log("camera keys: bindings, heading, one speed, easing, look, wheel, and the nose-down view hold");
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-camera-keys: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-camera-keys: ok");
