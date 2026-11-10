// Vendored from Articraft (Apache-2.0), extracted: this logic lived inline in
// their `ViewerShell.tsx`, which is not vendored (it is built around their
// record store). Pulling it into its own module is the only change — the motion
// math is theirs, unmodified.
//
// Drives the "Preview motion" render option: every movable joint sweeps its own
// range on its own cycle, phase-offset so a multi-joint object does not move as
// one rigid block.

import * as THREE from "three";

import type { UrdfJoint } from "@/components/inspector/JointSlider";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";

export const PREVIEW_UI_SYNC_MS = 100;
const PREVIEW_MIN_CYCLE_SECONDS = 2.6;
const PREVIEW_LINEAR_SPEED_MPS = 0.08;
const PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 4;
const PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 5;
const PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS = 0.24;
const PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS = (Math.PI * 2) / 3;

export type PreviewJointMotion = {
  joint: UrdfJoint;
  cycleSeconds: number;
  phaseOffset: number;
};

function isPreviewJoint(joint: UrdfJoint): boolean {
  const mimic = (joint as UrdfJoint & { mimic?: unknown }).mimic;
  return (
    !mimic &&
    (joint.type === "revolute" ||
      joint.type === "continuous" ||
      joint.type === "prismatic")
  );
}

function buildRootLinks(joints: UrdfJoint[]): string[] {
  const parents = new Set<string>();
  const children = new Set<string>();
  for (const joint of joints) {
    parents.add(joint.parent);
    children.add(joint.child);
  }
  const roots = Array.from(parents).filter((name) => !children.has(name));
  return roots.length > 0
    ? roots
    : Array.from(new Set(joints.map((joint) => joint.parent)));
}

/** Movable joints in kinematic order, root outward. */
function buildPreviewJointSequence(urdfSpec: UrdfSpec): UrdfJoint[] {
  const jointsByParent = new Map<string, UrdfJoint[]>();
  for (const joint of urdfSpec.joints) {
    const current = jointsByParent.get(joint.parent) ?? [];
    current.push(joint);
    jointsByParent.set(joint.parent, current);
  }

  const orderedJoints: UrdfJoint[] = [];
  const seenLinks = new Set<string>();

  const visitLink = (linkName: string): void => {
    if (seenLinks.has(linkName)) return;
    seenLinks.add(linkName);
    for (const joint of jointsByParent.get(linkName) ?? []) {
      if (isPreviewJoint(joint)) orderedJoints.push(joint);
      visitLink(joint.child);
    }
  };

  for (const rootLink of buildRootLinks(urdfSpec.joints)) visitLink(rootLink);

  // Joints unreachable from any root (a malformed tree) still get to move.
  const seenJointNames = new Set(orderedJoints.map((joint) => joint.name));
  for (const joint of urdfSpec.joints) {
    if (isPreviewJoint(joint) && !seenJointNames.has(joint.name)) {
      orderedJoints.push(joint);
    }
  }
  return orderedJoints;
}

function previewJointTravelSpan(joint: UrdfJoint): number {
  if (joint.type === "continuous") return Math.PI * 2;
  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number" &&
    Number.isFinite(lower) &&
    typeof upper === "number" &&
    Number.isFinite(upper) &&
    upper > lower;
  if (hasRange) return upper - lower;
  if (joint.type === "prismatic") {
    return PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS;
  }
  return PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS;
}

function previewJointCycleSeconds(joint: UrdfJoint): number {
  const span = previewJointTravelSpan(joint);
  if (joint.type === "continuous") {
    return span / PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND;
  }
  const speed =
    joint.type === "prismatic"
      ? PREVIEW_LINEAR_SPEED_MPS
      : PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND;
  return Math.max(PREVIEW_MIN_CYCLE_SECONDS, (span * 2) / speed);
}

/** Deterministic per-joint phase, so joints do not swing in lockstep. */
function previewJointPhaseOffset(jointName: string, index: number): number {
  let hash = 0;
  for (const character of jointName) {
    hash = (hash * 33 + character.charCodeAt(0)) % 4096;
  }
  return THREE.MathUtils.euclideanModulo(
    hash / 4096 + index * 0.61803398875,
    1,
  );
}

export function buildPreviewJointMotions(
  urdfSpec: UrdfSpec,
): PreviewJointMotion[] {
  return buildPreviewJointSequence(urdfSpec).map((joint, index) => ({
    joint,
    cycleSeconds: previewJointCycleSeconds(joint),
    phaseOffset: previewJointPhaseOffset(joint.name, index),
  }));
}

export function previewJointValue(joint: UrdfJoint, phase: number): number {
  if (joint.type === "continuous") {
    return (
      THREE.MathUtils.euclideanModulo(phase * Math.PI * 2 + Math.PI, Math.PI * 2) -
      Math.PI
    );
  }
  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number" &&
    Number.isFinite(lower) &&
    typeof upper === "number" &&
    Number.isFinite(upper) &&
    upper > lower;
  if (hasRange) {
    // Cosine ease so the joint slows at both limits instead of snapping.
    const normalized = 0.5 - 0.5 * Math.cos(phase * Math.PI * 2);
    return THREE.MathUtils.lerp(lower, upper, normalized);
  }
  const wave = Math.sin(phase * Math.PI * 2);
  if (joint.type === "prismatic") return wave * 0.12;
  return wave * (Math.PI / 3);
}
