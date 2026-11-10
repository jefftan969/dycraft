// Creative-mode flight for the viewport, as in Minecraft: the mouse looks, the keys fly.
//
// Orbiting is the right control for looking *at* one object and the wrong one for being *inside* a room, and it
// made the camera's motion depend on how far out it was: dragging swung a far camera across the scene, and the keys
// moved it a fraction of its distance from the target each second, so the further out, the faster. Here nothing
// depends on that distance:
//
//   1. **Dragging looks around where the camera stands**, by yaw and pitch, the way the mouse turns the head in
//      Minecraft: drag right and the view turns right. Pitch stops short of straight up and down.
//   2. **WASD flies along the horizontal heading.** W and S keep their height when the camera looks down; A and D
//      strafe level; Space and E rise, Shift and Q sink, plumb. The arrows mirror WASD.
//   3. **At one speed for the scene,** a quarter of its diagonal a second (an 8 cm mug and a 40 m store both stay
//      usable), whatever the zoom; twice that while sprinting (W tapped twice, as in Minecraft: Ctrl+W would close
//      the tab before the page saw it).
//   4. **With a little easing:** the velocity follows the keys over about a tenth of a second, so a tap is a nudge.
//   5. **The wheel steps along the view by a fixed distance,** a quarter second of flight a notch, in proportion to
//      the delta so a trackpad glides.
//
// The math lives here, apart from the hook that binds it (`useCameraKeys`), so `npm run check:camera-keys` can pin
// it without a DOM.

import * as THREE from 'three';

/** How far each axis is deflected this frame; -1, 0, or 1. */
export interface MoveAxes {
  /** Along the horizontal heading. */
  forward: number;
  /** Across it, horizontally. */
  right: number;
  /** World vertical. */
  up: number;
}

/**
 * Physical keys, by `KeyboardEvent.code`.
 *
 * `code` rather than `key` on purpose: `key` reports the *character*, so a
 * capital `W` arrives while Shift is held for "down" and would have to be
 * case-folded, and on a non-QWERTY layout it names whichever letter sits under
 * the key instead of the WASD diamond the hand is on.
 *
 * The two schemes are unified rather than kept separate — Space and Shift work
 * alongside WASD, and Q/E alongside the arrows — because there is no reason to
 * make a user remember which vertical pair belongs to which hand.
 */
export const MOVE_KEYS: Readonly<Record<string, Partial<MoveAxes>>> = {
  KeyW: { forward: 1 },
  ArrowUp: { forward: 1 },
  KeyS: { forward: -1 },
  ArrowDown: { forward: -1 },
  KeyA: { right: -1 },
  ArrowLeft: { right: -1 },
  KeyD: { right: 1 },
  ArrowRight: { right: 1 },
  KeyE: { up: 1 },
  Space: { up: 1 },
  KeyQ: { up: -1 },
  ShiftLeft: { up: -1 },
  ShiftRight: { up: -1 },
};

export function isMoveKey(code: string): boolean {
  return code in MOVE_KEYS;
}

/**
 * Sum the held keys into one deflection per axis.
 *
 * Opposites cancel rather than fight (holding W and S stands still), and two
 * keys bound to the same direction do not double it.
 */
export function moveAxesFromKeys(pressed: Iterable<string>): MoveAxes {
  const axes: MoveAxes = { forward: 0, right: 0, up: 0 };
  for (const code of pressed) {
    const binding = MOVE_KEYS[code];
    if (!binding) continue;
    axes.forward += binding.forward ?? 0;
    axes.right += binding.right ?? 0;
    axes.up += binding.up ?? 0;
  }
  return {
    forward: Math.sign(axes.forward),
    right: Math.sign(axes.right),
    up: Math.sign(axes.up),
  };
}

/** World up. The robot group is rotated into three.js's Y-up on load. */
const WORLD_UP = new THREE.Vector3(0, 1, 0);

/** Seconds a single step may integrate: a backgrounded tab resumes with a long delta, which must not teleport. */
const MAX_STEP_SECONDS = 0.1;

/** How long the velocity takes to close about two thirds of the way to the keys' (Minecraft's flight drifts a little). */
export const EASE_SECONDS = 0.08;

/** Sprinting, twice the speed. */
export const SPRINT_FACTOR = 2;

/** A second press of W within this many milliseconds sprints, until W is let go. */
export const SPRINT_TAP_MS = 300;

/** Pitch stops this short of straight up or down, so the heading never degenerates. */
const PITCH_LIMIT = Math.PI / 2 - 0.02;

/** The scene's flight speed, in meters a second: a quarter of its diagonal, within sane bounds. */
export function flySpeed(sceneDiagonal: number): number {
  if (!Number.isFinite(sceneDiagonal) || sceneDiagonal <= 0) return 1;
  return Math.min(20, Math.max(0.05, sceneDiagonal * 0.25));
}

/** The horizontal heading: where the camera looks, flattened, so W keeps its height when looking down. */
export function heading(camera: THREE.Camera): THREE.Vector3 {
  const forward = camera.getWorldDirection(new THREE.Vector3());
  forward.y = 0;
  if (forward.lengthSq() < 1e-12) {
    // Straight down or up: the camera's own up vector points where it faces.
    forward.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion));
    forward.y = 0;
    if (forward.lengthSq() < 1e-12) forward.set(0, 0, -1);
  }
  return forward.normalize();
}

/** The velocity the held keys ask for, in meters a second, in the world frame. */
export function keyVelocity(camera: THREE.Camera, axes: MoveAxes, speed: number): THREE.Vector3 {
  const forward = heading(camera);
  const right = new THREE.Vector3().crossVectors(forward, WORLD_UP).normalize();
  const horizontal = new THREE.Vector3().addScaledVector(forward, axes.forward).addScaledVector(right, axes.right);
  // Diagonals are no faster than the axes they are made of.
  if (horizontal.lengthSq() > 1) horizontal.normalize();
  return horizontal.addScaledVector(WORLD_UP, axes.up).multiplyScalar(speed);
}

/** `current` moved toward `target` over `deltaSeconds`, exponentially, with time constant `EASE_SECONDS`. */
export function easeVelocity(current: THREE.Vector3, target: THREE.Vector3, deltaSeconds: number): THREE.Vector3 {
  const dt = Math.min(Math.max(deltaSeconds, 0), MAX_STEP_SECONDS);
  const k = 1 - Math.exp(-dt / EASE_SECONDS);
  return current.clone().lerp(target, k);
}

/** One step's displacement at `velocity`: no more than `MAX_STEP_SECONDS` of it. */
export function stepDisplacement(velocity: THREE.Vector3, deltaSeconds: number): THREE.Vector3 {
  if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return new THREE.Vector3();
  return velocity.clone().multiplyScalar(Math.min(deltaSeconds, MAX_STEP_SECONDS));
}

/**
 * The camera's orientation after a drag of `dx`, `dy` pixels on a canvas `height` pixels tall: a drag across the
 * canvas's height turns the view by its vertical field of view. Drag right to look right, up to look up.
 */
export function lookQuaternion(
  camera: THREE.PerspectiveCamera, dx: number, dy: number, height: number,
): THREE.Quaternion {
  const k = THREE.MathUtils.degToRad(camera.fov) / Math.max(1, height);
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
  euler.y -= dx * k;
  euler.x = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, euler.x - dy * k));
  euler.z = 0;
  return new THREE.Quaternion().setFromEuler(euler);
}

/**
 * The wheel's step along the view, in meters: a quarter second of flight per 100 of `deltaY` (one notch of a mouse
 * wheel), back for a positive delta, at most three notches an event. Proportional, so a trackpad's many small deltas
 * glide rather than each jumping a notch.
 */
export function wheelStep(camera: THREE.Camera, deltaY: number, speed: number): THREE.Vector3 {
  if (!Number.isFinite(deltaY) || deltaY === 0) return new THREE.Vector3();
  const notches = Math.max(-3, Math.min(3, deltaY / 100));
  return camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(-notches * speed * 0.25);
}
