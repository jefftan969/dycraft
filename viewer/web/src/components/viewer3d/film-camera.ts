// The camera a delivered film opens on, as a three.js view: where a film run starts in the viewport.
//
// The server reads it from the film's own record (`bundle/film/camera.json`, see `viewer/animation.py`
// `film_camera`): the camera's world matrix in Blender's frame and its lens. Blender and three.js cameras both look
// down their local -Z with +Y up, so the camera's pose is its world matrix carried into the viewport by the same
// matrix that carries the model there (the robot group's: the z-up rotation and the ground offset).
//
// Pure math, apart from the `THREE` types, so `npm run check:film-camera` can pin it against Blender's own
// projection without a DOM.

import * as THREE from 'three';

/** The viewport camera's own vertical field of view (`useThreeScene`), to which every non-film selection returns. */
export const DEFAULT_FOV_DEGREES = 45;

/** `film_camera` in the `/animation` payload. */
export type FilmCamera = {
  frame: number | null;
  camera: string | null;
  /** World matrix in Blender's frame, rows first. */
  matrix: number[][];
  lens_mm: number;
  sensor_mm: number;
  sensor_height_mm: number;
  /** Blender's `sensor_fit`: which film dimension the sensor size is measured along. */
  sensor_fit: 'AUTO' | 'HORIZONTAL' | 'VERTICAL' | string;
  shift: number[];
  /** The film's width over its height. */
  aspect: number;
};

/** Half-angle tangents of the film's horizontal and vertical fields of view, as Blender computes them. */
export function filmHalfTangents(film: FilmCamera): { h: number; v: number } {
  const aspect = film.aspect;
  // AUTO measures the sensor width along whichever dimension is larger.
  const fit = film.sensor_fit === 'AUTO' ? (aspect >= 1 ? 'HORIZONTAL' : 'VERTICAL') : film.sensor_fit;
  if (fit === 'VERTICAL') {
    const size = film.sensor_fit === 'AUTO' ? film.sensor_mm : film.sensor_height_mm;
    const v = size / 2 / film.lens_mm;
    return { h: v * aspect, v };
  }
  const h = film.sensor_mm / 2 / film.lens_mm;
  return { h, v: h / aspect };
}

/**
 * The vertical field of view, in degrees, that shows all of the film's frame on a canvas of `canvasAspect`.
 *
 * A canvas wider than the film keeps the film's height and shows more at the sides; a narrower one keeps its width
 * and shows more above and below. Either way nothing the film shows is cropped.
 */
export function filmFovDegrees(film: FilmCamera, canvasAspect: number): number {
  const { h, v } = filmHalfTangents(film);
  const tan = canvasAspect >= film.aspect ? v : h / canvasAspect;
  return THREE.MathUtils.radToDeg(2 * Math.atan(tan));
}

/** The film camera's world matrix in the viewport: `toViewer` (the robot group's matrixWorld) times Blender's. */
export function filmCameraMatrix(film: FilmCamera, toViewer: THREE.Matrix4): THREE.Matrix4 {
  const m = film.matrix;
  const blender = new THREE.Matrix4().set(
    m[0][0], m[0][1], m[0][2], m[0][3],
    m[1][0], m[1][1], m[1][2], m[1][3],
    m[2][0], m[2][1], m[2][2], m[2][3],
    m[3][0], m[3][1], m[3][2], m[3][3],
  );
  return new THREE.Matrix4().multiplyMatrices(toViewer, blender);
}

/**
 * The film's camera at every frame (`/film-cameras`, `viewer/animation.py` `film_cameras`), which the film view
 * follows as the film plays. One entry per frame: `camera` indexes `cameras` (-1 where the viewer cannot reproduce
 * that frame's camera), `matrix` holds twelve numbers a frame (the world matrix's top three rows, rows first), and
 * `lens_mm` and `shift` the lens. A change of `camera` is a cut.
 */
export type FilmCameraTrack = {
  frame_start: number;
  n_frames: number;
  aspect: number;
  cameras: { name: string; sensor_mm: number; sensor_height_mm: number; sensor_fit: string }[];
  camera: number[];
  matrix: number[];
  lens_mm: number[];
  shift: number[];
};

/** The film's camera at the playhead's `frame`, or null where the track has none the viewer can reproduce. */
export function filmCameraAt(track: FilmCameraTrack, frame: number): FilmCamera | null {
  const i = Math.max(0, Math.min(track.n_frames - 1, Math.round(frame)));
  const index = track.camera[i];
  const camera = index >= 0 ? track.cameras[index] : undefined;
  if (!camera) return null;
  const m = track.matrix.slice(12 * i, 12 * i + 12);
  const film: FilmCamera = {
    frame: track.frame_start + i,
    camera: camera.name,
    matrix: [m.slice(0, 4), m.slice(4, 8), m.slice(8, 12), [0, 0, 0, 1]],
    lens_mm: track.lens_mm[i],
    sensor_mm: camera.sensor_mm,
    sensor_height_mm: camera.sensor_height_mm,
    sensor_fit: camera.sensor_fit,
    shift: [track.shift[2 * i], track.shift[2 * i + 1]],
    aspect: track.aspect,
  };
  return isUsableFilmCamera(film) ? film : null;
}

/** One shot of a film, in the playhead's frames, both ends included, named by its camera. */
export type Shot = { start: number; end: number; name: string };

/**
 * The film's shots from its camera track: a change of camera is a cut, and a camera the film returns to later starts a
 * shot of its own. Frames whose camera the viewer cannot reproduce belong to the shot before them.
 */
export function shotsOf(track: FilmCameraTrack | null): Shot[] {
  if (!track) return [];
  const shots: (Shot & { camera: number })[] = [];
  for (let i = 0; i < track.n_frames; i += 1) {
    const camera = track.camera[i];
    const last = shots[shots.length - 1];
    if (last && (camera < 0 || camera === last.camera)) {
      last.end = i;
    } else if (camera >= 0) {
      shots.push({ start: i, end: i, name: track.cameras[camera].name, camera });
    }
  }
  return shots.map(({ start, end, name }) => ({ start, end, name }));
}

/** Whether two camera poses are the same, to a tenth of a millimeter: the film view is still the film's. */
export function sameCameraPose(a: THREE.Matrix4, b: THREE.Matrix4, tolerance = 1e-4): boolean {
  for (let k = 0; k < 16; k += 1) if (Math.abs(a.elements[k] - b.elements[k]) > tolerance) return false;
  return true;
}

/** Whether `film` is something this viewer can open on. */
export function isUsableFilmCamera(film: FilmCamera | null | undefined): film is FilmCamera {
  return Boolean(film && Array.isArray(film.matrix) && film.matrix.length === 4
    && film.matrix.every((row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite))
    && film.lens_mm > 0 && film.sensor_mm > 0 && film.aspect > 0);
}

/**
 * Put `camera` where the film opens, aimed as it is, with the film's field of view; return the orbit target.
 *
 * The target is on the camera's line of sight, at the point nearest the middle of `box` (the scene, in viewport
 * coordinates), so orbiting turns about what the shot is looking at rather than about the camera. When the scene
 * is behind or level with the camera, it is half the scene's size ahead instead.
 *
 * The orbit controls' next update aims the camera at that target with the viewport's up, which drops any roll: a
 * film camera tilted about its own axis (a dutch angle) shows level here. That is the controls, not the placement.
 */
export function placeAtFilmCamera(
  camera: THREE.PerspectiveCamera,
  film: FilmCamera,
  toViewer: THREE.Matrix4,
  box: THREE.Box3,
): THREE.Vector3 {
  const world = filmCameraMatrix(film, toViewer);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  world.decompose(position, quaternion, new THREE.Vector3());
  camera.position.copy(position);
  camera.quaternion.copy(quaternion);
  camera.fov = filmFovDegrees(film, camera.aspect);
  camera.updateMatrixWorld(true);

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
  const along = center.clone().sub(position).dot(forward);
  const distance = along > radius * 0.05 ? along : radius;
  return position.clone().addScaledVector(forward, distance);
}
