// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import { useState, useCallback, useMemo, useRef, useEffect, type RefObject } from 'react';
import * as THREE from 'three';
import type { UrdfSpec, UrdfJoint } from './urdf-parser';

export interface JointControllerState {
  jointValues: Map<string, number>;
  /**
   * The pose actually applied to the graph, live.
   *
   * `jointValues` is React state and lags on purpose — Preview Motion and clip
   * playback pose the graph every frame but only sync the inspector ~10x a
   * second, because re-rendering the sliders at 60 fps is pure waste. Anything
   * that has to agree with what is *on screen* right now (the live Blender
   * viewport streams the pose it is told, so a lagging pose would render the
   * object visibly behind the three.js graph) reads this instead.
   */
  posedJointValues: RefObject<Map<string, number>>;
  setJointValue: (name: string, value: number) => void;
  applyJointValues: (values: Map<string, number>, options?: { commit?: boolean }) => void;
  resetAll: () => void;
}

interface JointRuntime {
  spec: UrdfJoint;
  axis: THREE.Vector3;
}

const DEFAULT_AXIS: [number, number, number] = [0, 0, 1];
const tempQuaternion = new THREE.Quaternion();
const tempOffset = new THREE.Vector3();

/**
 * Resolve every joint to a value.
 *
 * `restValues` is the scene's committed rest configuration and is the fallback
 * for any joint nobody has set — NOT a one-time seed. That distinction is the
 * whole fix: the URDF is in dycraft's joint frame and cannot express a default
 * pose, so an unset joint must resolve to its authored default rather than to
 * 0, no matter which caller is asking. Seeding once instead was tried and lost
 * a race — the preview-motion effect restores `jointValuesRef.current`, a ref
 * synced one render late, and wiped the seeded pose back to zero.
 */
function resolveJointValues(
  explicitValues: Map<string, number>,
  runtimeMap: Map<string, JointRuntime>,
  restValues?: Map<string, number> | null,
): Map<string, number> {
  const resolved = new Map<string, number>();
  const resolving = new Set<string>();

  const resolveJoint = (name: string): number => {
    const cached = resolved.get(name);
    if (cached !== undefined) {
      return cached;
    }

    const runtime = runtimeMap.get(name);
    if (!runtime) {
      return explicitValues.get(name) ?? restValues?.get(name) ?? 0;
    }
    if (resolving.has(name)) {
      return 0;
    }

    resolving.add(name);
    const mimic = runtime.spec.mimic;
    let value = explicitValues.get(name) ?? restValues?.get(name) ?? 0;
    if (mimic) {
      value = resolveJoint(mimic.joint) * mimic.multiplier + mimic.offset;
    }
    resolving.delete(name);
    resolved.set(name, value);
    return value;
  };

  for (const name of runtimeMap.keys()) {
    resolveJoint(name);
  }

  return resolved;
}

function applyResolvedJointValues(
  jointNodes: Map<string, THREE.Object3D> | null,
  runtimeMap: Map<string, JointRuntime>,
  values: Map<string, number>,
): void {
  if (!jointNodes) {
    return;
  }

  for (const [name, node] of jointNodes) {
    const runtime = runtimeMap.get(name);
    if (!runtime) {
      continue;
    }
    applyJointValue(node, runtime.spec, runtime.axis, values.get(name) ?? 0);
  }
}

/**
 * What the controller can drive, as a string.
 *
 * Compared instead of the `UrdfSpec` object because a spec object is new on
 * every load — including the ones that reload the *same* model in another
 * variant, which is what the baked-material swap is. Keying the pose reset on
 * object identity there would throw the user's pose away: every door they had
 * opened would shut the moment the bake landed. Two specs with the same joints, types
 * and limits are interchangeable as far as a pose is concerned.
 */
function jointSignature(urdfSpec: UrdfSpec | null): string {
  return (urdfSpec?.joints ?? [])
    .map((joint) => [
      joint.name, joint.type,
      joint.limit?.lower ?? '', joint.limit?.upper ?? '',
      joint.mimic?.joint ?? '',
    ].join(':'))
    .join('|');
}

/**
 * Hook for managing joint values and applying them to the Three.js scene graph.
 *
 * Maintains a Map<string, number> of resolved joint values. Mimic joints are
 * derived from their source joint and are not independently controllable.
 */
export function useJointController(
  jointNodes: Map<string, THREE.Object3D> | null,
  urdfSpec: UrdfSpec | null,
  restValues?: Map<string, number> | null,
  /**
   * Called after the graph has been posed, so the canvas repaints.
   *
   * The viewport renders **on demand**: mutating the three.js graph without
   * this sets the transforms and never draws them.
   */
  onPosed?: () => void,
): JointControllerState {
  const [jointValues, setJointValues] = useState<Map<string, number>>(new Map());
  const jointRuntimeMap = useRef<Map<string, JointRuntime>>(new Map());
  const explicitJointValuesRef = useRef<Map<string, number>>(new Map());
  const posedJointValuesRef = useRef<Map<string, number>>(new Map());
  const signature = useMemo(() => jointSignature(urdfSpec), [urdfSpec]);
  const signatureRef = useRef<string | null>(null);

  // The one place a resolved pose reaches the graph, so it is also the one
  // place that has to record it. Four callers resolve a pose; none of them
  // should have to remember to publish it.
  const pose = useCallback(
    (resolved: Map<string, number>) => {
      posedJointValuesRef.current = resolved;
      applyResolvedJointValues(jointNodes, jointRuntimeMap.current, resolved);
    },
    [jointNodes],
  );

  useEffect(() => {
    const map = new Map<string, JointRuntime>();
    if (urdfSpec) {
      for (const joint of urdfSpec.joints) {
        map.set(joint.name, {
          spec: joint,
          axis: new THREE.Vector3(...(joint.axis ?? DEFAULT_AXIS)).normalize(),
        });
      }
    }
    jointRuntimeMap.current = map;
    // Seed from the scene's committed rest configuration, not from 0. The URDF
    // is in dycraft's joint frame and has no notion of a default pose (URDF
    // cannot express one), so a joint whose authored default is non-zero would
    // otherwise draw at 0 — a door hanging open, an arm folded — and look
    // deliberate. `restValues` comes from each JointSummary's `default`.
    //
    // The pose the user has set is kept when the *same* joints come back under
    // a new spec object — the baked-material swap, or a re-selected run — and
    // dropped only when the joints themselves change, which is a different
    // model and a pose that would mean nothing on it.
    if (signatureRef.current !== signature) {
      signatureRef.current = signature;
      explicitJointValuesRef.current = new Map();
    }
    const seeded = resolveJointValues(explicitJointValuesRef.current, map,
                                      restValues);
    posedJointValuesRef.current = seeded;
    setJointValues(seeded);
  }, [urdfSpec, restValues, signature]);

  // Pose the graph as soon as its nodes exist, with whatever the current pose
  // is. Separate from the effect above because the nodes arrive *after* the
  // spec — the URDF has to be parsed and its meshes loaded before they can be
  // collected — so the seed alone would leave the model drawn at zero. On a
  // variant swap this is what carries the pose onto the new graph: the joint
  // names are the same, so the same values apply.
  useEffect(() => {
    if (!jointNodes || !urdfSpec) {
      return;
    }
    const resolved = resolveJointValues(explicitJointValuesRef.current,
                                        jointRuntimeMap.current, restValues);
    pose(resolved);
    setJointValues(resolved);
    onPosed?.();
  }, [jointNodes, onPosed, pose, restValues, urdfSpec]);

  const setJointValue = useCallback(
    (name: string, value: number) => {
      const runtime = jointRuntimeMap.current.get(name);
      if (!runtime || runtime.spec.mimic) {
        return;
      }

      const nextExplicit = new Map(explicitJointValuesRef.current);
      nextExplicit.set(name, value);
      explicitJointValuesRef.current = nextExplicit;

      const resolved = resolveJointValues(nextExplicit, jointRuntimeMap.current,
                                          restValues);
      pose(resolved);
      setJointValues(resolved);
    },
    [pose, restValues],
  );

  const applyJointValues = useCallback(
    (values: Map<string, number>, options?: { commit?: boolean }) => {
      const nextExplicit = new Map<string, number>();
      for (const [name, value] of values) {
        if (!Number.isFinite(value)) {
          continue;
        }
        const runtime = jointRuntimeMap.current.get(name);
        if (runtime?.spec.mimic) {
          continue;
        }
        nextExplicit.set(name, value);
      }

      explicitJointValuesRef.current = nextExplicit;
      const resolved = resolveJointValues(nextExplicit, jointRuntimeMap.current,
                                          restValues);
      pose(resolved);

      if (options?.commit) {
        setJointValues(new Map(resolved));
      }
    },
    [pose, restValues],
  );

  // "Reset" means back to the pose the scene committed, which is what the
  // renders and every geometric query use — not back to a URDF zero nobody
  // authored.
  const resetAll = useCallback(() => {
    explicitJointValuesRef.current = new Map();
    const resolved = resolveJointValues(new Map(), jointRuntimeMap.current,
                                        restValues);
    pose(resolved);
    setJointValues(resolved);
  }, [pose, restValues]);

  return { jointValues, posedJointValues: posedJointValuesRef, setJointValue,
           applyJointValues, resetAll };
}

/**
 * Apply a joint value to a Three.js node.
 *
 * For revolute/continuous joints, a quaternion rotation around the joint axis is applied.
 * For prismatic joints, a translation along the joint axis is applied.
 */
function applyJointValue(
  node: THREE.Object3D,
  spec: UrdfJoint,
  axis: THREE.Vector3,
  value: number,
): void {
  switch (spec.type) {
    case 'revolute':
    case 'continuous': {
      tempQuaternion.setFromAxisAngle(axis, value);
      node.quaternion.copy(tempQuaternion);
      break;
    }
    case 'prismatic': {
      tempOffset.copy(axis).multiplyScalar(value);
      node.position.copy(tempOffset);
      break;
    }
    default:
      break;
  }
}
