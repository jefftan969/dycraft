// Binds the cutaway (`cutaway.ts`) to the viewport: fetches a native film's walls and ceilings, and each time the
// camera moves, hides the ones in its way.
//
// A hidden object goes to layer 1, which the camera, the shadow pass, and picking leave out, rather than having
// `visible` cleared: native playback writes `visible` every frame (`applyNativeFrame`), and the two must not fight.
//
// The film's own view shows every wall, as the film does; the cutaway starts once the camera leaves it (an orbit, a
// pan, a key). With no film camera, the viewport opens free and the cutaway is on from the first frame.

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { fetchCutaway } from '@/lib/api';
import { type Cutaway, cutawayState } from './cutaway';

/** The layer a cut-away object is moved to: the camera renders layer 0 only. */
export const CUT_LAYER = 1;
/** `buildNativeScene`'s group; its meshes are named by object. */
const NATIVE_GROUP = 'native-animation';

interface CutawayOptions {
  runId: string | null;
  /** The native clip's version (`timeline.native.version`); null when the run has no native clip. */
  nativeVersion: string | null | undefined;
  /**
   * The first picture is up. The cutaway is asked for only then, so working it out (a pass over every mesh, on the
   * server) never competes with what the first picture needs.
   */
  ready: boolean;
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  controls: OrbitControls | null;
  /** Walls cut away (the default) or all shown: the viewport's toggle and the H key. */
  enabled: boolean;
  /**
   * The camera's world matrix where the film's view put it, or null when it did not. While the camera is still
   * there, every wall shows.
   */
  filmPose: RefObject<THREE.Matrix4 | null>;
  /**
   * Set while the viewport itself moves the camera onto the film's view (`SceneCanvas`): the controls' 'change' then
   * comes before the new pose is recorded, and the one after it is the one to decide on.
   */
  placing?: RefObject<boolean>;
  invalidate: () => void;
}

export interface CutawayStatus {
  /** The film has walls or ceilings to cut away. */
  available: boolean;
  /** The camera is in free view, where the cutaway applies. */
  free: boolean;
  /** How many walls and ceilings are hidden now. */
  hiddenCount: number;
}

/** Whether two world matrices put a camera in the same place, looking the same way. */
function samePose(a: THREE.Matrix4, b: THREE.Matrix4): boolean {
  const x = a.elements, y = b.elements;
  for (let i = 0; i < 16; i += 1) if (Math.abs(x[i] - y[i]) > 1e-4) return false;
  return true;
}

/**
 * The room casts no shadow on what is in it. The look's sun (`FILM_LOOK`) comes from above, and a ceiling or a wall
 * between it and the action would put the whole film in one shadow, which reads as no shadow at all. So the walls and
 * ceilings this cutaway knows, what is mounted on them, and anything at or above a ceiling's underside (beams, joists,
 * the lamps hung there) cast none; they still take the shadows of what is in the room.
 *
 * Decided once per clip, on the frame on screen when the cutaway arrives: an object that starts at or above a ceiling
 * and comes down later (a lowered lamp, a drone) casts nothing for the whole clip. One not built yet (an empty box)
 * keeps casting.
 */
export function roomCastsNoShadow(data: Cutaway, group: THREE.Object3D, byName: Map<string, THREE.Object3D>): void {
  for (const u of [...data.units, ...data.attach]) {
    for (const name of u.objects) {
      const object = byName.get(name);
      if (object) object.castShadow = false;
    }
  }
  const unders = data.units.flatMap((u) => (u.kind === 'ceiling' && typeof u.under === 'number' ? [u.under] : []));
  if (unders.length === 0) return;
  // The lowest ceiling's underside in the viewport: the group's matrix carries the film's frame (Z up) into it.
  const underY = new THREE.Vector3(0, 0, Math.min(...unders)).applyMatrix4(group.matrixWorld).y;
  const box = new THREE.Box3();
  for (const child of group.children) {
    if (!child.castShadow) continue;
    box.setFromObject(child);
    if (!box.isEmpty() && box.min.y >= underY - 0.05) child.castShadow = false;
  }
}

function setCut(object: THREE.Object3D | undefined, cut: boolean): void {
  if (!object) return;
  if (cut) object.layers.set(CUT_LAYER);
  else object.layers.set(0);
}

export function useCutaway({
  runId, nativeVersion, ready, scene, camera, controls, enabled, filmPose, placing, invalidate,
}: CutawayOptions): CutawayStatus {
  const [data, setData] = useState<Cutaway | null>(null);
  const [status, setStatus] = useState<CutawayStatus>({ available: false, free: false, hiddenCount: 0 });
  const hidden = useRef<boolean[]>([]);
  const attached = useRef<boolean[]>([]);

  useEffect(() => {
    setData(null);
    hidden.current = [];
    attached.current = [];
  }, [runId, nativeVersion]);
  const asked = useRef<string | null>(null);
  useEffect(() => {
    if (!runId || !nativeVersion || !ready) return;
    const key = `${runId}@${nativeVersion}`;
    if (asked.current === key) return;
    asked.current = key;
    fetchCutaway(runId)
      .then((c) => { if (asked.current === key && c.animation === nativeVersion) setData(c); })
      // No cutaway: every wall stays, which is what the viewer did before. Asked again on the next first picture.
      .catch(() => { if (asked.current === key) asked.current = null; });
  }, [runId, nativeVersion, ready]);

  const update = useCallback(() => {
    if (!scene || !camera) return;
    const group = scene.getObjectByName(NATIVE_GROUP);
    if (!data || !group) {
      setStatus((s) => (s.available || s.hiddenCount ? { available: false, free: s.free, hiddenCount: 0 } : s));
      return;
    }
    camera.updateMatrixWorld();
    group.updateWorldMatrix(true, false);
    const pose = filmPose.current;
    const free = !pose || !samePose(camera.matrixWorld, pose);
    const byName = new Map<string, THREE.Object3D>();
    for (const child of group.children) byName.set(child.name, child);
    // A clip built since the last look (a new run, or the same one reloaded) starts with every mesh on layer 0.
    if (group.userData.cutawayFor !== data.animation) {
      group.userData.cutawayFor = data.animation;
      for (const u of [...data.units, ...data.attach]) for (const name of u.objects) setCut(byName.get(name), false);
      hidden.current = [];
      attached.current = [];
      roomCastsNoShadow(data, group, byName);
      invalidate();
    }
    let next: { units: boolean[]; attach: boolean[] };
    if (enabled && free) {
      next = cutawayState(data, camera, group.matrixWorld, controls ? controls.target : null, hidden.current);
    } else {
      next = { units: data.units.map(() => false), attach: data.attach.map(() => false) };
    }
    let changed = false;
    next.units.forEach((h, i) => {
      if (h === Boolean(hidden.current[i])) return;
      changed = true;
      for (const name of data.units[i].objects) setCut(byName.get(name), h);
    });
    next.attach.forEach((h, i) => {
      if (h === Boolean(attached.current[i])) return;
      changed = true;
      for (const name of data.attach[i].objects) setCut(byName.get(name), h);
    });
    hidden.current = next.units;
    attached.current = next.attach;
    if (changed) invalidate();
    const count = next.units.filter(Boolean).length;
    setStatus((s) => (s.available && s.free === free && s.hiddenCount === count ? s
      : { available: data.units.length > 0, free, hiddenCount: count }));
  }, [camera, controls, data, enabled, filmPose, invalidate, scene]);

  // Whenever the camera moves (an orbit, a pan, the fly keys, which all go through the controls), and whenever what
  // it decides from changes.
  useEffect(() => {
    if (!controls) return;
    const onChange = () => {
      if (!placing?.current) update();
    };
    controls.addEventListener('change', onChange);
    return () => controls.removeEventListener('change', onChange);
  }, [controls, placing, update]);
  useEffect(() => { update(); }, [update]);

  // The clip's meshes can arrive after the cutaway does, and a reloaded clip's start on layer 0: look again once a
  // second, which costs one walk of the scene.
  useEffect(() => {
    if (!data) return;
    const id = window.setInterval(update, 1000);
    return () => window.clearInterval(id);
  }, [data, update]);

  return status;
}
