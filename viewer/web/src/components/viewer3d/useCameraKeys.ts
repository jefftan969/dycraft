// Binds `camera-keys` to the viewport: creative-mode flight, as in Minecraft. Dragging looks around where the camera
// stands, the keys fly at the scene's speed with a little easing, the wheel steps along the view, and R goes back to
// the film's view.
//
// OrbitControls stays for what it does well, a right-drag pan, and for the events the rest of the viewport listens
// to ('change', 'end'). Its rotate and zoom are off: they moved the camera by fractions of its distance from the
// target, so the further out it was, the faster everything went. The target is kept a fixed distance straight ahead
// (the scene's speed, one second of flight), so the pan's scale is the scene's too, and `controls.update()` rebuilds
// exactly the pose set here.

import { useEffect, useRef, type RefObject } from 'react';
import * as THREE from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import {
  easeVelocity, isMoveKey, keyVelocity, lookQuaternion, moveAxesFromKeys, SPRINT_FACTOR, SPRINT_TAP_MS,
  stepDisplacement, wheelStep,
} from './camera-keys';

/** Elements that own the keyboard while they have it. */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
}

interface CameraKeysOptions {
  /**
   * The viewport element. Keys count while it is hovered or focused.
   *
   * A ref rather than the element: it is populated after the first render and
   * nothing re-renders on that, so an element read during render would be null
   * forever. The effect reads it when `camera` and `controls` arrive, which is
   * a state change and therefore a re-render.
   */
  containerRef: RefObject<HTMLElement | null>;
  camera: THREE.PerspectiveCamera | null;
  controls: OrbitControls | null;
  /** The canvas renders on demand; every moved frame has to ask for a draw. */
  invalidate: () => void;
  /** The scene's flight speed in meters a second (`flySpeed`), read on every step. */
  speedRef: RefObject<number>;
  /** R: back to the film's view (or the scene's framing when the run has no film camera). */
  onReset?: () => void;
  enabled?: boolean;
}

/**
 * Fly the camera while the viewport has the pointer or focus.
 *
 * **Keys are gated on hover or focus, not on the window.** The inspector is full of scrollable panels and the arrow
 * keys scroll them; a global listener would fly the camera away while someone was reading a trajectory. And never
 * while a text field has the keyboard.
 */
export function useCameraKeys({
  containerRef,
  camera,
  controls,
  invalidate,
  speedRef,
  onReset,
  enabled = true,
}: CameraKeysOptions): void {
  const pressedRef = useRef<Set<string>>(new Set());
  const hoveredRef = useRef(false);
  const resetRef = useRef(onReset);
  resetRef.current = onReset;

  useEffect(() => {
    const container = containerRef.current;
    if (!enabled || !container || !camera || !controls) {
      return;
    }
    const pressed = pressedRef.current;
    const element = controls.domElement as HTMLElement;
    const was = { rotate: controls.enableRotate, zoom: controls.enableZoom };
    controls.enableRotate = false;
    controls.enableZoom = false;

    const speed = () => (speedRef.current && speedRef.current > 0 ? speedRef.current : 1);
    /** Keep the target one second of flight straight ahead, so the controls rebuild this pose and pan at scale. */
    const settleTarget = () => {
      const ahead = camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(speed());
      controls.target.copy(camera.position).add(ahead);
      controls.update();
      invalidate();
    };

    // ---- keys
    let velocity = new THREE.Vector3();
    let frame = 0;
    let last: number | null = null;
    let sprinting = false;
    let lastW = -Infinity;

    const stop = () => {
      if (frame !== 0) cancelAnimationFrame(frame);
      frame = 0;
      last = null;
      velocity = new THREE.Vector3();
      // `useUrdfLoader` refits on the controls' own `end`, once per burst, as after a drag.
      controls.dispatchEvent({ type: 'end' });
    };

    const step = (now: number) => {
      frame = 0;
      const dt = last == null ? 1 / 60 : (now - last) / 1000;
      last = now;
      const target = keyVelocity(camera, moveAxesFromKeys(pressed), speed() * (sprinting ? SPRINT_FACTOR : 1));
      velocity = easeVelocity(velocity, target, dt);
      if (pressed.size === 0 && velocity.length() < speed() * 0.01) {
        stop();
        return;
      }
      camera.position.add(stepDisplacement(velocity, dt));
      camera.updateMatrixWorld();
      settleTarget();
      frame = requestAnimationFrame(step);
    };
    const start = () => {
      if (frame === 0) frame = requestAnimationFrame(step);
    };

    const isActive = () => hoveredRef.current
      || (document.activeElement !== null && container.contains(document.activeElement));

    const onKeyDown = (event: KeyboardEvent) => {
      // Leave every browser and OS shortcut alone. Shift is excluded on purpose: it is a binding here.
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (!isActive() || isTypingTarget(event.target)) return;
      if (event.code === 'KeyR' && !event.repeat) {
        resetRef.current?.();
        return;
      }
      if (!isMoveKey(event.code)) return;
      // The arrow keys scroll the page and Space activates whatever is focused; both would happen as well.
      event.preventDefault();
      if (event.repeat) return;
      if (event.code === 'KeyW' || event.code === 'ArrowUp') {
        const now = performance.now();
        if (now - lastW <= SPRINT_TAP_MS) sprinting = true;
        lastW = now;
      }
      pressed.add(event.code);
      start();
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'KeyW' || event.code === 'ArrowUp') sprinting = false;
      if (pressed.delete(event.code) && isMoveKey(event.code)) event.preventDefault();
    };
    // A key held while focus leaves the page never sends its keyup, nor a button its pointerup.
    const release = () => {
      pressed.clear();
      sprinting = false;
      endLook();
    };
    const onPointerEnter = () => {
      hoveredRef.current = true;
    };
    const onPointerLeave = () => {
      hoveredRef.current = false;
      if (!isActive()) release();
    };

    // ---- the mouse looks: a left drag turns the camera where it stands
    let look: { x: number; y: number; id: number; moved: boolean } | null = null;
    // The clip planes are refit on 'end' (`useUrdfLoader` walks every mesh), so only after a look that moved.
    function endLook() {
      if (look?.moved) controls?.dispatchEvent({ type: 'end' });
      look = null;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 || !event.isPrimary) return;
      look = { x: event.clientX, y: event.clientY, id: event.pointerId, moved: false };
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!look || event.pointerId !== look.id) return;
      // A release this page never saw (outside the window, a context menu): the button is up, so the look is over.
      if ((event.buttons & 1) === 0) {
        endLook();
        return;
      }
      look.moved = true;
      const dx = event.clientX - look.x;
      const dy = event.clientY - look.y;
      look.x = event.clientX;
      look.y = event.clientY;
      if (!dx && !dy) return;
      camera.quaternion.copy(lookQuaternion(camera, dx, dy, element.clientHeight));
      camera.updateMatrixWorld();
      settleTarget();
    };
    const onPointerUp = (event: PointerEvent) => {
      if (look && event.pointerId === look.id) endLook();
    };

    // ---- the wheel steps along the view, and the clip planes are refit once it has been still for 150 ms
    let wheelEnd = 0;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY;
      camera.position.add(wheelStep(camera, delta, speed()));
      camera.updateMatrixWorld();
      settleTarget();
      window.clearTimeout(wheelEnd);
      wheelEnd = window.setTimeout(() => controls.dispatchEvent({ type: 'end' }), 150);
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', release);
    document.addEventListener('visibilitychange', release);
    container.addEventListener('pointerenter', onPointerEnter);
    container.addEventListener('pointerleave', onPointerLeave);
    element.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    element.addEventListener('wheel', onWheel, { passive: false });

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', release);
      document.removeEventListener('visibilitychange', release);
      container.removeEventListener('pointerenter', onPointerEnter);
      container.removeEventListener('pointerleave', onPointerLeave);
      element.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      element.removeEventListener('wheel', onWheel);
      window.clearTimeout(wheelEnd);
      controls.enableRotate = was.rotate;
      controls.enableZoom = was.zoom;
      pressed.clear();
      if (frame !== 0) cancelAnimationFrame(frame);
    };
  }, [camera, containerRef, controls, enabled, invalidate, speedRef]);
}
