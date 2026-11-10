// Pin the film camera: a film run opens in the viewport on exactly the view its film does.
//
//     npm run check:film-camera
//
// `film-camera.ts` turns a delivered film's camera (`bundle/film/camera.json`: a Blender world matrix and a lens) into
// a three.js view. What has to hold, and is silent when broken, since a wrong camera still shows the scene:
//
//   1. **Same view.** Points in front of the camera project to the same place in the film's frame through Blender's
//      camera model (written out here, independently of the module) and through the three.js camera, when the canvas
//      has the film's shape, for horizontal, vertical, and automatic sensor fits.
//   2. **Nothing cropped.** On a wider or narrower canvas the film's whole frame stays inside the canvas, touching
//      its edges along one axis.
//   3. **The model's own carry.** The camera goes through the same matrix as the model (the robot group's), here a
//      z-up rotation plus an offset.
//   4. **A target on the line of sight**, so orbiting turns about what the shot looks at.
//   5. **The viewer's own field of view** is the module's default, which a non-film selection returns to.
//
// Transpiled through the esbuild vite already installs, like check:camera-keys.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-film-camera-"));
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
  const F = await bundle(join(V3, "film-camera.ts"), "film-camera.mjs");
  const THREE = await import(pathToFileURL(join(WEB, "node_modules", "three", "build", "three.module.js")).href);

  // A delivered film's opening camera, as its camera.json records it.
  const opening = {
    frame: 0, camera: "CAM_01_SCULLERY",
    matrix: [[-0.7314673662185669, 0.13798077404499054, -0.6677700281143188, -1.25],
             [-0.6818764209747314, -0.14801551401615143, 0.7163350582122803, 1.0499999523162842],
             [1.5206627779207338e-07, 0.9793123602867126, 0.20235435664653778, 1.75],
             [0, 0, 0, 1]],
    lens_mm: 24.325603485107422, sensor_mm: 36, sensor_height_mm: 24, sensor_fit: "HORIZONTAL",
    shift: [0, 0], aspect: 1920 / 1080,
  };
  const cases = [
    ["horizontal fit, 16:9", opening],
    ["automatic fit, landscape", { ...opening, sensor_fit: "AUTO", lens_mm: 16.8, aspect: 1280 / 720 }],
    ["automatic fit, portrait", { ...opening, sensor_fit: "AUTO", lens_mm: 50, aspect: 720 / 1280 }],
    ["vertical fit", { ...opening, sensor_fit: "VERTICAL", lens_mm: 35, aspect: 4 / 3 }],
  ];

  /** Blender's projection: a world point to the film frame, in [-1, 1] across each dimension. */
  function blenderNdc(film, point) {
    const m = new THREE.Matrix4().set(...film.matrix.flat());
    const local = point.clone().applyMatrix4(m.clone().invert());
    // Blender's sensor fit: AUTO measures `sensor_width` along the larger dimension.
    let fit = film.sensor_fit;
    let size = fit === "VERTICAL" ? film.sensor_height_mm : film.sensor_mm;
    if (fit === "AUTO") { fit = film.aspect >= 1 ? "HORIZONTAL" : "VERTICAL"; size = film.sensor_mm; }
    const half = size / 2 / film.lens_mm;
    const tanH = fit === "HORIZONTAL" ? half : half * film.aspect;
    const tanV = fit === "HORIZONTAL" ? half / film.aspect : half;
    return [local.x / -local.z / tanH, local.y / -local.z / tanV];
  }

  // The model's carry into the viewport: the z-up rotation the robot group applies, and a ground offset.
  const toViewer = new THREE.Matrix4().makeRotationX(-Math.PI / 2)
    .premultiply(new THREE.Matrix4().makeTranslation(0.3, -0.12, 0.7));
  const box = new THREE.Box3(new THREE.Vector3(-2, 0, -2), new THREE.Vector3(2, 2.5, 2));

  let worst = 0;
  for (const [label, film] of cases) {
    const m = new THREE.Matrix4().set(...film.matrix.flat());
    const position = new THREE.Vector3().setFromMatrixPosition(m);
    const basis = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    m.extractBasis(...basis);
    const [right, up, back] = basis;
    // Points spread across the film's frame at several depths, in Blender's world.
    const points = [];
    for (const depth of [0.5, 2, 6]) {
      for (const a of [-0.6, 0, 0.5]) {
        for (const b of [-0.4, 0, 0.3]) {
          points.push(position.clone().addScaledVector(back, -depth)
            .addScaledVector(right, a * depth * 0.3).addScaledVector(up, b * depth * 0.3));
        }
      }
    }
    for (const canvasAspect of [film.aspect, film.aspect * 1.6, film.aspect / 1.6]) {
      const camera = new THREE.PerspectiveCamera(45, canvasAspect, 0.01, 100);
      const target = F.placeAtFilmCamera(camera, film, toViewer, box);
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld(true);
      camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
      for (const p of points) {
        const [bx, by] = blenderNdc(film, p);
        const v = p.clone().applyMatrix4(toViewer).project(camera);
        // The film frame on this canvas: one axis matches the canvas, the other is inset by the aspect ratio.
        const sx = canvasAspect >= film.aspect ? film.aspect / canvasAspect : 1;
        const sy = canvasAspect >= film.aspect ? 1 : canvasAspect / film.aspect;
        const err = Math.max(Math.abs(v.x - bx * sx), Math.abs(v.y - by * sy));
        worst = Math.max(worst, err);
        check(err < 1e-4, `${label}, canvas ${canvasAspect.toFixed(2)}: a point lands at (${v.x.toFixed(4)}, ` +
              `${v.y.toFixed(4)}) instead of (${(bx * sx).toFixed(4)}, ${(by * sy).toFixed(4)})`);
      }
      // The frame's corners stay on the canvas.
      check(Math.abs(F.filmFovDegrees(film, canvasAspect) - camera.fov) < 1e-9, `${label}: fov not applied`);
      // The target is on the line of sight.
      const sight = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      const toTarget = target.clone().sub(camera.position).normalize();
      check(sight.dot(toTarget) > 1 - 1e-9, `${label}: the orbit target is off the line of sight`);
    }
  }

  check(F.isUsableFilmCamera(opening) && !F.isUsableFilmCamera({ ...opening, matrix: [[1, 0], [0, 1]] })
        && !F.isUsableFilmCamera(null), "isUsableFilmCamera accepts or refuses the wrong records");
  const scene = readFileSync(join(V3, "useThreeScene.ts"), "utf8");
  check(scene.includes(`new THREE.PerspectiveCamera(${F.DEFAULT_FOV_DEGREES},`),
        `DEFAULT_FOV_DEGREES (${F.DEFAULT_FOV_DEGREES}) is not the viewport camera's own field of view`);
  console.log(`projection: ${cases.length} cameras x 3 canvas shapes x 27 points, worst error ${worst.toExponential(2)}`);

  // The film's camera at every frame (`/film-cameras`): what the film view follows as the film plays.
  {
    const rows = opening.matrix.slice(0, 3).flat();
    const moved = rows.map((x, k) => (k === 3 ? x + 0.5 : x));
    const track = {
      frame_start: 10, n_frames: 3, aspect: opening.aspect,
      cameras: [{ name: "CAM_a", sensor_mm: 36, sensor_height_mm: 24, sensor_fit: "HORIZONTAL" }],
      camera: [0, 0, -1], matrix: [...rows, ...moved, ...new Array(12).fill(0)], lens_mm: [24.3, 30, 0],
      shift: [0, 0, 0.1, 0, 0, 0],
    };
    const second = F.filmCameraAt(track, 1);
    check(second && second.camera === "CAM_a" && second.frame === 11 && second.lens_mm === 30,
          `frame 1 of the track: ${JSON.stringify(second)}`);
    check(second && second.matrix[0][3] === opening.matrix[0][3] + 0.5 && second.matrix[3].join() === "0,0,0,1",
          "frame 1's own matrix did not come through, with its fourth row");
    check(second && second.shift[0] === 0.1, "frame 1's lens shift did not come through");
    check(F.filmCameraAt(track, 2) === null, "a frame whose camera cannot be reproduced must give none");
    check(F.filmCameraAt(track, 99)?.lens_mm === undefined || F.filmCameraAt(track, 99) === null,
          "past the end the last frame stands");
    check(F.filmCameraAt(track, -5)?.frame === 10, "before the start the first frame stands");
    // Shots: a change of camera is a cut, a camera returned to starts a shot of its own, and a frame without a
    // camera the viewer can reproduce belongs to the shot before it.
    const shotsTrack = {
      ...track, n_frames: 7,
      cameras: [{ name: "A", sensor_mm: 36, sensor_height_mm: 24, sensor_fit: "AUTO" },
                { name: "B", sensor_mm: 36, sensor_height_mm: 24, sensor_fit: "AUTO" }],
      camera: [-1, 0, 0, 1, -1, 0, 0],
    };
    const shots = F.shotsOf(shotsTrack);
    check(JSON.stringify(shots) === JSON.stringify([{ start: 1, end: 2, name: "A" }, { start: 3, end: 4, name: "B" },
                                                     { start: 5, end: 6, name: "A" }]),
          `shots: ${JSON.stringify(shots)}`);
    check(F.shotsOf(null).length === 0, "no track, no shots");
    const a = new THREE.Matrix4().makeTranslation(1, 2, 3);
    check(F.sameCameraPose(a, a.clone()) && !F.sameCameraPose(a, new THREE.Matrix4().makeTranslation(1, 2, 3.001)),
          "the film view's pose must match to a tenth of a millimeter, and a millimeter off is not it");
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-film-camera: ${failures.length} failure(s)`);
  for (const f of failures.slice(0, 20)) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-film-camera: ok");
