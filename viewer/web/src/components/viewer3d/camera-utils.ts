// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import * as THREE from 'three';

export interface FitResult {
  position: THREE.Vector3;
  target: THREE.Vector3;
  near: number;
  far: number;
}

const MIN_NEAR = 0.001;
const MIN_FAR_GAP = 1;
/** The nearest the far plane ever comes: the render distance is not capped at the scene's own box. */
export const DEFAULT_CLIP_FAR = 1000;
/**
 * Tightest near:far the viewport will run at.
 *
 * `near` is not just a clip plane: it sets what a depth-buffer step is worth in
 * world units, which is `z^2 * (far - near) / (near * far) * 2^-24` on the
 * 24-bit default framebuffer. Anything denominated in depth-buffer steps —
 * `polygonOffsetUnits`, see `depthBiasForOrdinal` — scales with that.
 *
 * Without a floor a scene bottoms out at `MIN_NEAR`, because the clip box
 * includes the ground grid and `positionGroundHelpers` scales the grid to three
 * times the model footprint: on a room-sized model the bounding sphere is wider
 * than the camera's distance from it, so `centerDepth - radius - padding` goes
 * negative. A near plane at a millimeter against a far plane at 20 m makes one
 * depth step millimeters wide at the model, hundreds of times an object view's,
 * enough to draw coplanar surfaces in the wrong order.
 *
 * The floor is a ratio rather than a fixed distance so it follows the model's
 * scale, and it is loose enough (2 cm in front of the eye at far = 20 m) to clip
 * nothing a viewer needs — in particular the grid passing under the camera,
 * which driving `near` off the model alone would cut.
 */
const MAX_NEAR_FAR_RATIO = 1000;

function getBoundingBox(
  object: THREE.Object3D | readonly THREE.Object3D[],
): THREE.Box3 | null {
  const objects = Array.isArray(object) ? object : [object];
  const box = new THREE.Box3();

  for (const entry of objects) {
    if (!entry.visible) {
      continue;
    }
    box.expandByObject(entry);
  }

  return box.isEmpty() ? null : box;
}

export function computeClipPlanesForBox(
  camera: THREE.PerspectiveCamera,
  box: THREE.Box3,
): { near: number; far: number } {
  camera.updateMatrixWorld(true);

  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const centerDepth = -sphere.center.clone().applyMatrix4(camera.matrixWorldInverse).z;
  const nearPadding = Math.max(sphere.radius * 0.1, 0.02);
  const farPadding = Math.max(sphere.radius * 0.5, 0.5);

  // A bounding sphere produces stable clip planes even when the model bounds
  // straddle the camera plane during pans or close orbit moves.
  const wanted = Math.max(MIN_NEAR, centerDepth - sphere.radius - nearPadding);
  const far = Math.max(wanted + MIN_FAR_GAP, centerDepth + sphere.radius + farPadding);
  // Depth resolution, not visibility: see MAX_NEAR_FAR_RATIO. Only ever raises
  // `near`, and only when the box reaches so far past the camera that `near`
  // would otherwise sit on its floor.
  const near = Math.max(wanted, far / MAX_NEAR_FAR_RATIO);
  // Nothing past the scene's box is clipped either: the far plane goes out to at least DEFAULT_CLIP_FAR. Depth
  // precision is set by `near`, which keeps the box's own ratio above, so pushing `far` out costs next to nothing:
  // a depth step is `z^2 * (far - near) / (near * far) * 2^-24`, and `(far - near) / far` is already close to 1.
  return { near, far: Math.max(far, DEFAULT_CLIP_FAR) };
}

/**
 * Compute a camera position that fits the given object within the camera's field of view.
 * Returns the ideal position, target (center of bounding box), and clipping planes.
 */
export function computeFit(
  camera: THREE.PerspectiveCamera,
  object: THREE.Object3D,
): FitResult {
  const box = getBoundingBox(object);
  if (!box) {
    return {
      position: camera.position.clone(),
      target: new THREE.Vector3(),
      near: camera.near,
      far: camera.far || DEFAULT_CLIP_FAR,
    };
  }
  return computeFitForBox(camera, box);
}

/**
 * `computeFit` against an explicit box rather than the current geometry.
 *
 * An animation needs this: the scene's bounds at any one frame are not the
 * bounds of the clip, so fitting on what is currently drawn puts anything that
 * starts in the air — or ends on the floor — outside the viewport.
 */
export function computeFitForBox(
  camera: THREE.PerspectiveCamera,
  box: THREE.Box3,
): FitResult {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  // Distance needed so the object fits in view, with a small padding factor
  const distance = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

  // Position camera along its current direction from center, or default diagonal
  const direction = new THREE.Vector3();
  if (camera.position.lengthSq() > 0) {
    direction.copy(camera.position).sub(center).normalize();
  }
  if (direction.lengthSq() < 0.001) {
    direction.set(1, 0.8, 1).normalize();
  }

  const position = center.clone().add(direction.multiplyScalar(distance));
  const fitCamera = camera.clone();
  fitCamera.position.copy(position);
  fitCamera.lookAt(center);
  const { near, far } = computeClipPlanesForBox(fitCamera, box);

  return { position, target: center, near, far };
}

/**
 * Preserve the camera's current viewing angle and relative distance when switching
 * to a new model. Adjusts position so the new object is centered and fits in view.
 */
export function preserveViewAcrossModelSwitch(
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; update: () => void },
  newObject: THREE.Object3D,
): void {
  const oldTarget = controls.target.clone();
  const oldOffset = camera.position.clone().sub(oldTarget);
  const oldDistance = oldOffset.length();
  const direction = oldDistance > 0.001
    ? oldOffset.normalize()
    : new THREE.Vector3(1, 0.8, 1).normalize();

  const box = new THREE.Box3().setFromObject(newObject);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const fitDistance = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

  // Use the old viewing direction but scale to fit the new object
  const newPosition = center.clone().add(direction.multiplyScalar(fitDistance));

  camera.position.copy(newPosition);
  controls.target.copy(center);
  controls.update();

  updateCameraClipping(camera, newObject);
}

/**
 * Update the camera's near and far clipping planes based on the object's bounding box size.
 * Ensures the full object is visible without z-fighting artifacts.
 */
export function updateCameraClipping(
  camera: THREE.PerspectiveCamera,
  object: THREE.Object3D | readonly THREE.Object3D[],
): void {
  const box = getBoundingBox(object);
  if (!box) {
    return;
  }

  const { near, far } = computeClipPlanesForBox(camera, box);
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();
}
