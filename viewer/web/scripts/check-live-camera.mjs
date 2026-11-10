// Pin the live Blender viewport's change of basis.
//
//     npm run check:live-camera
//
// The streamed frame has to land on the viewport *exactly*, or replacing the
// preview with it produces a visible framing jump — and the browser is the
// only place that knows the conversion, because the robot group carries both
// the z-up rotation (`useUrdfLoader.ts`, `rotation.x = -PI/2`) and the
// ground-normalizing offset. A sign error there is silent: the render is
// plausible, of the wrong viewpoint.
//
// Nothing in the Python suite can catch this, and there is no browser here, so
// this drives `cameraToDycraftWorld` under node against three's own math.
// Transpiled through the esbuild vite already installs, like check:uv.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const SOURCE = join(WEB, "src", "components", "viewer3d", "useBlenderStream.ts");

const TOL = 1e-6;
const work = mkdtempSync(join(tmpdir(), "dycraft-live-cam-"));
const failures = [];

function near(label, got, want, tol = TOL) {
  const d = Math.max(...got.map((v, i) => Math.abs(v - want[i])));
  if (!(d <= tol)) {
    failures.push(`${label}: got [${got.map((v) => v.toFixed(6))}], ` +
                  `want [${want.map((v) => v.toFixed(6))}] (|d| = ${d.toExponential(2)})`);
  }
  return d;
}

try {
  const bundle = join(work, "cam.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [SOURCE, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${bundle}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );

  const { cameraToDycraftWorld } = await import(pathToFileURL(bundle).href);
  const THREE = await import(pathToFileURL(
    join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  /** The group the URDF loader builds: z-up rotation, then a ground offset. */
  function robotGroup(offset) {
    const group = new THREE.Group();
    group.rotation.x = -Math.PI / 2;
    group.position.set(offset[0], offset[1], offset[2]);
    group.updateMatrixWorld(true);
    return group;
  }

  // ---- 1. The axis convention, stated in dycraft terms ------------------
  //
  // A three.js camera above and in front of the model, with three's default
  // +Y up, must come back as a dycraft camera with +Z up. If the rotation were
  // applied the other way this would read -Z and every streamed frame would be
  // upside down — which is exactly the kind of thing that looks like "the
  // renderer is broken" rather than like a basis error.
  {
    const group = robotGroup([0, 0, 0]);
    const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.01, 100);
    camera.position.set(0, 2, 3);       // three: up 2, toward the viewer 3
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    const got = cameraToDycraftWorld(camera, new THREE.Vector3(0, 0, 0), group);
    // Perpendicular to the view direction and on the +Z side, which is what
    // "up" means in a z-up world.
    if (!(got.up[2] > 0.7)) {
      failures.push(`up is not predominantly +Z in dycraft: ` +
                    `[${got.up.map((v) => v.toFixed(4))}]`);
    }
    // three (x, y, z) -> dycraft (x, -z, y) under a -90deg rotation about X.
    near("eye maps three(0,2,3) -> dycraft(0,-3,2)", got.eye, [0, -3, 2]);
    near("target at the origin stays there", got.target, [0, 0, 0]);
    if (got.fovDeg !== 45) {
      failures.push(`fovDeg passed through as ${got.fovDeg}, want 45 ` +
                    `(both ends mean VERTICAL fov)`);
    }
  }

  // ---- 2. The ground-normalizing offset is undone ------------------------
  //
  // `normalizeRobotGroupToGroundOrigin` shifts the group so the model sits on
  // the grid. Ignoring it would send a camera offset by the model's own
  // centering — small, plausible, and wrong by exactly the amount that makes a
  // framing jump visible.
  {
    const offset = [0.31, -0.07, 0.44];
    const group = robotGroup(offset);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(0.31, 2, 3.44);
    camera.up.set(0, 1, 0);
    camera.lookAt(0.31, -0.07, 0.44);
    const got = cameraToDycraftWorld(camera, new THREE.Vector3(0.31, -0.07, 0.44), group);
    near("the offset is subtracted from the eye", got.eye, [0, -3, 2.07]);
    near("a target at the group origin is the dycraft origin", got.target, [0, 0, 0]);
  }

  // ---- 3. Round trip, over a spread of poses -----------------------------
  //
  // The general invariant, and the one that would survive a refactor of the
  // loader: pushing the answer back through the group's own matrix must
  // reproduce the three.js camera it came from.
  let worst = 0;
  for (const offset of [[0, 0, 0], [0.4, 0.2, -0.9], [-2, 1.5, 0.05]]) {
    const group = robotGroup(offset);
    for (const yaw of [0, 0.7, 1.9, 3.4, 5.1]) {
      for (const pitch of [-0.9, 0.0, 0.6, 1.2]) {
        const r = 2.4;
        const camera = new THREE.PerspectiveCamera(37.5, 4 / 3, 0.01, 100);
        camera.position.set(
          r * Math.cos(pitch) * Math.cos(yaw),
          r * Math.sin(pitch),
          r * Math.cos(pitch) * Math.sin(yaw),
        );
        camera.up.set(0, 1, 0);
        const controlsTarget = new THREE.Vector3(0.1, 0.2, -0.3);
        camera.lookAt(controlsTarget);
        camera.updateMatrixWorld(true);
        const expectedUp = new THREE.Vector3()
          .setFromMatrixColumn(camera.matrixWorld, 1).normalize();
        const got = cameraToDycraftWorld(camera, controlsTarget, group);

        const backEye = new THREE.Vector3(...got.eye).applyMatrix4(group.matrixWorld);
        const backTarget = new THREE.Vector3(...got.target).applyMatrix4(group.matrixWorld);
        const normal = new THREE.Matrix3().setFromMatrix4(group.matrixWorld);
        const backUp = new THREE.Vector3(...got.up).applyMatrix3(normal).normalize();

        worst = Math.max(
          worst,
          near("round-trip eye", [backEye.x, backEye.y, backEye.z],
               [camera.position.x, camera.position.y, camera.position.z], 1e-5),
          near("round-trip target", [backTarget.x, backTarget.y, backTarget.z],
               [controlsTarget.x, controlsTarget.y, controlsTarget.z], 1e-5),
          near("round-trip up", [backUp.x, backUp.y, backUp.z],
               [expectedUp.x, expectedUp.y, expectedUp.z], 1e-5),
        );
      }
    }
  }

  // ---- 4. Straight down, where `camera.up` would have gone degenerate ----
  {
    const group = robotGroup([0, 0, 0]);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(0, 3, 0);       // three +Y == dycraft +Z
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    const got = cameraToDycraftWorld(camera, new THREE.Vector3(0, 0, 0), group);
    near("a top-down eye is straight up in dycraft", got.eye, [0, 0, 3]);
    // The property that actually matters, stated the way the child uses it:
    // `_aim_camera` builds `right = forward x up` and falls back to an
    // arbitrary perpendicular when that vector has no length. So the test is
    // the CONDITIONING of that cross product, not exact orthogonality —
    // three's own `lookAt` nudges the axis by 1e-4 in this degenerate case and
    // that is fine. Sending `camera.up` instead would put this at ~0.
    const forward = new THREE.Vector3(...got.eye).negate().normalize();
    const cross = forward.clone().cross(new THREE.Vector3(...got.up)).length();
    if (!(cross > 0.99)) {
      failures.push(`looking straight down, |forward x up| = ` +
                    `${cross.toExponential(3)}; _aim_camera would take its ` +
                    `degenerate fallback and pick an arbitrary image rotation`);
    }
    const length = Math.hypot(...got.up);
    if (!(Math.abs(length - 1) < 1e-5)) {
      failures.push(`up is not a unit vector (|up| = ${length})`);
    }
  }

  if (failures.length > 0) {
    console.error("check:live-camera FAILED — the streamed frame would not " +
                  "match the viewport:");
    for (const line of failures) console.error(`  ${line}`);
    process.exit(1);
  }
  console.log(`check:live-camera OK, worst round-trip error = ` +
              `${worst.toExponential(3)} over 60 poses`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
