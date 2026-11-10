// Pose the loaded URDF graph at one frame of an AnimationIR.
//
// The kinematics are NOT recomputed here. `viewer/animation.py` resolves every
// instance's world pose per frame with the real scene kernel and ships the
// answers; this file's whole job is to put a given world matrix onto the right
// node. That split is deliberate — base deltas propagate through the SceneIR's
// attach forest in ways the URDF does not model (a `follow=false` child is
// parented to `world` at its rest pose yet still rides its parent's motion), so
// a second implementation here would agree on the easy cases and be quietly
// wrong on those.
//
// Where the pose is applied: `scene-graph-builder` wires every URDF joint as
//
//     parent link -> jointFrame (the joint origin) -> motion node -> child link
//
// and the joint controller writes the motion node for revolute/prismatic
// joints only. An instance is placed by a *fixed* joint, so that joint's motion
// node is an untouched identity — exactly the hook a whole-body pose needs, and
// it composes with the joint sliders instead of fighting them.

import * as THREE from "three";
import type { AnimationBody, AnimationTimeline } from "@/lib/types";
import type { UrdfSpec } from "./urdf-parser";

/** A body's placement joint, resolved against the loaded graph. */
export interface PlacementTarget {
  body: AnimationBody;
  /** The fixed joint's motion node — what we write. */
  node: THREE.Object3D;
  /** Its parent, the joint-origin frame — what we measure against. */
  frame: THREE.Object3D;
  /**
   * The URDF root group. Poses arrive in **dycraft world coordinates**, and
   * three.js world is not that: `useUrdfLoader` wraps the robot in a group it
   * rotates -90° about X (z-up to y-up) and then translates to sit the model on
   * the ground plane. This node's frame is dycraft world, so every pose is
   * expressed through it. Composing against `matrixWorld` directly instead
   * lays every clip on its side, a meter or so off the origin.
   */
  root: THREE.Object3D;
  /** Depth in the URDF link tree; parents must be posed before children. */
  depth: number;
}

/** `scene-graph-builder` names the URDF root group `robot:<name>`. */
const ROOT_NAME_PREFIX = "robot:";

const _target = new THREE.Matrix4();
const _local = new THREE.Matrix4();
const _inverse = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _one = new THREE.Vector3(1, 1, 1);

/**
 * The URDF root group above `node`, or the topmost ancestor if the naming
 * convention ever changes — a wrong-but-consistent reference frame degrades to
 * a visibly misplaced model, where returning null would silently disable
 * playback instead.
 */
function findUrdfRoot(node: THREE.Object3D): THREE.Object3D {
  let cursor: THREE.Object3D = node;
  let fallback: THREE.Object3D = node;
  while (cursor.parent) {
    if (cursor.name.startsWith(ROOT_NAME_PREFIX)) return cursor;
    fallback = cursor;
    cursor = cursor.parent;
  }
  return fallback;
}

/**
 * `[tx,ty,tz,qw,qx,qy,qz]` (dycraft, w-first) as a THREE.Matrix4.
 *
 * The reorder to three.js's x-first quaternion happens here and nowhere else.
 */
export function se3ToMatrix4(row: number[], out: THREE.Matrix4): THREE.Matrix4 {
  _pos.set(row[0], row[1], row[2]);
  _quat.set(row[4], row[5], row[6], row[3]);
  return out.compose(_pos, _quat, _one);
}

/**
 * Resolve each moving body onto the node that places it, parent-first.
 *
 * The placement joint is found by its *child link* rather than by rebuilding
 * its name: `scene_ir_to_urdf` calls it `world__<instance>` for a room-placed
 * instance but `<parent link>__<instance>` for a kinematic-follow attach, and
 * the child link is the same either way.
 *
 * Returns `{targets, missing}`; `missing` names bodies with no such joint in
 * the loaded graph, which means the URDF and the timeline disagree about the
 * scene and is worth showing rather than swallowing.
 */
export function resolvePlacementTargets(
  timeline: AnimationTimeline,
  spec: UrdfSpec,
  jointNodes: Map<string, THREE.Object3D>,
): { targets: PlacementTarget[]; missing: string[] } {
  const jointByChild = new Map<string, { name: string; parent: string }>();
  for (const joint of spec.joints) {
    jointByChild.set(joint.child, { name: joint.name, parent: joint.parent });
  }

  const depthOf = (link: string): number => {
    let depth = 0;
    let cursor: string | undefined = link;
    // Bounded by the link count: a malformed URDF with a parent cycle would
    // otherwise spin here forever.
    for (let guard = 0; guard <= spec.links.length; guard += 1) {
      const joint: { name: string; parent: string } | undefined =
        cursor === undefined ? undefined : jointByChild.get(cursor);
      if (!joint) return depth;
      depth += 1;
      cursor = joint.parent;
    }
    return depth;
  };

  const targets: PlacementTarget[] = [];
  const missing: string[] = [];
  for (const body of timeline.bodies) {
    const joint = jointByChild.get(body.link);
    const node = joint ? jointNodes.get(joint.name) : undefined;
    if (!node || !node.parent) {
      missing.push(body.instance);
      continue;
    }
    targets.push({
      body,
      node,
      frame: node.parent,
      root: findUrdfRoot(node),
      depth: depthOf(body.link),
    });
  }
  targets.sort((a, b) => a.depth - b.depth);
  return { targets, missing };
}

/** Joint values at `frame`, as the joint controller wants them. */
export function jointValuesAtFrame(
  timeline: AnimationTimeline,
  frame: number,
): Map<string, number> {
  const values = new Map<string, number>();
  for (const track of timeline.joints) {
    const value = track.values[frame];
    if (value !== undefined) values.set(track.name, value);
  }
  return values;
}

/**
 * Place every body at `frame`. Call AFTER the joint values for that frame have
 * been applied — a follow-attached child hangs off a parent link whose pose
 * depends on the parent's joints, so the frame matrices read below must
 * already reflect them.
 */
export function applyBodyPoses(targets: PlacementTarget[], frame: number): void {
  for (const { body, node, frame: jointFrame, root } of targets) {
    node.visible = body.exists[frame] ?? true;
    const row = body.frames[frame];
    if (!row) continue;
    // Recompute the origin frame's world matrix from the root each time: an
    // earlier target in this loop may be this one's ancestor, and `targets` is
    // ordered parent-first precisely so that its pose is already in place.
    jointFrame.updateWorldMatrix(true, false);
    root.updateWorldMatrix(true, false);
    // dycraft world -> three.js world, then down into the joint's own frame.
    // The z-up rotation and ground-normalizing offset appear in both matrices
    // and cancel, which is the point of going through `root` rather than
    // hard-coding either.
    se3ToMatrix4(row, _target);
    _target.premultiply(root.matrixWorld);
    _inverse.copy(jointFrame.matrixWorld).invert();
    _local.multiplyMatrices(_inverse, _target);
    // Decompose rather than assigning `node.matrix`: leaving matrixAutoUpdate
    // on keeps every other consumer of this graph (the joint controller, the
    // overlay, the snapshot exporter) working off position/quaternion as usual.
    _local.decompose(node.position, node.quaternion, node.scale);
    node.updateMatrixWorld(true);
  }
}

/** Return every placed body to the rest pose the URDF was exported at. */
export function clearBodyPoses(targets: PlacementTarget[]): void {
  for (const { node } of targets) {
    node.visible = true;
    node.position.set(0, 0, 0);
    node.quaternion.identity();
    node.scale.set(1, 1, 1);
    node.updateMatrixWorld(true);
  }
}
