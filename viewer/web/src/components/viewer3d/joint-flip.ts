// "Flip joints": mirror every movable joint to the opposite side of its own
// range, in one pose.
//
// What it is for is inspection, not animation. A scene commits with every joint
// at its authored rest value — which for a door, a drawer, or a lid is almost
// always shut — so the rest pose says nothing about whether the articulation is
// right. One flip swings the lot the other way at once, and that is where the
// mistakes are visible: a door hinged on the wrong edge, a drawer that slides
// into its own carcass, a lid whose limit lets it pass through the wall behind
// it.
//
// Distinct from the Render tab's "Preview motion", which sweeps each joint on
// its own phase-offset cycle: that shows one joint at a time moving smoothly
// and never shows the whole mechanism displaced at once. There is no driver
// here and nothing animates — it is a pose, and the second click undoes it.

import type { UrdfJoint } from "@/components/inspector/JointSlider";

/**
 * A continuous joint has no limits at all; one full turn is the range the
 * sliders already show it over (`JointSlider.jointRange`), centered on where the
 * scene rests it.
 */
const CONTINUOUS_HALF_TURN = Math.PI;

/**
 * The travel a joint can actually be posed across, or null when it has none.
 *
 * Null covers three kinds of joint that a flip must leave alone: `fixed` and
 * the shapes the viewer cannot pose (`floating`, `planar`); a `mimic` joint,
 * whose value is a function of its source and is resolved rather than set; and
 * a revolute or prismatic joint whose URDF limits are absent or degenerate,
 * which is a joint that cannot move at all rather than one free to swing.
 */
export function flipRange(joint: UrdfJoint): [number, number] | null {
  if (joint.mimic) return null;
  if (joint.type === "continuous") {
    return [-CONTINUOUS_HALF_TURN, CONTINUOUS_HALF_TURN];
  }
  if (joint.type !== "revolute" && joint.type !== "prismatic") return null;
  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  if (typeof lower !== "number" || !Number.isFinite(lower)) return null;
  if (typeof upper !== "number" || !Number.isFinite(upper)) return null;
  return upper > lower ? [lower, upper] : null;
}

/** The joints a flip would move — the count the button reports. */
export function flippableJoints(joints: UrdfJoint[]): UrdfJoint[] {
  return joints.filter((joint) => flipRange(joint) !== null);
}

/** Where a joint sits when nothing has posed it: the scene's committed default. */
function restOf(joint: UrdfJoint, rest?: Map<string, number> | null): number {
  const value = rest?.get(joint.name);
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/**
 * Where one joint goes: **the mirror of where it is, within its own range**.
 *
 *     flipped = lower + upper - current
 *
 * On a range of 10..90 that sends 20 to 80, 45 to 55, 10 to 90, and leaves 50
 * where it is. Two properties follow from the arithmetic rather than from any
 * bookkeeping, and they are the whole reason for the button:
 *
 *   1. **It is its own inverse.** `f(f(q)) === q` exactly, for every starting
 *      pose — not just for one that happens to sit at a limit. So the second
 *      click returns the model to the pose the first click departed from, and
 *      there is no "flipped" flag for the per-joint sliders to contradict.
 *   2. **It cannot leave the limits.** The reflection of a point inside
 *      `[lower, upper]` is inside `[lower, upper]`. A pose a clip wrote may sit
 *      outside the range, so `current` is clamped in first; everything written
 *      is then in range by construction.
 *
 * **A mirror, not "the far end of the range".** The two coincide only for a
 * joint sitting *at* a limit, as a cabinet's door, drawer, and lid do. On a rig
 * whose rest pose is mid-range, which is every articulated robot, sending each
 * joint to its far end would throw the model to one extreme on the first click
 * and to the *other* on the second, never coming back.
 *
 * A continuous joint has no limits to mirror within, so it mirrors about its
 * rest value instead, over the same one-turn range the sliders show it on. One
 * sitting exactly at rest therefore stays put, which is the same thing that
 * happens to a bounded joint sitting exactly mid-range: there is no "other
 * side" of a midpoint.
 */
export function flippedValue(
  joint: UrdfJoint,
  current: number,
  rest?: Map<string, number> | null,
): number | null {
  const range = flipRange(joint);
  if (!range) return null;
  if (joint.type === "continuous") {
    // Mirror about rest: the range is notionally rest +- half a turn, and
    // `lower + upper` for that range is twice rest.
    return 2 * restOf(joint, rest) - current;
  }
  const [lower, upper] = range;
  const clamped = Math.min(Math.max(current, lower), upper);
  return lower + upper - clamped;
}

/**
 * The flipped pose.
 *
 * Seeded with `current` so that joints with no range to flip, and any pose a
 * driver has written, are carried through rather than being dropped back to
 * their authored defaults by the caller's whole-pose apply.
 */
export function buildFlippedJointValues(
  joints: UrdfJoint[],
  current: Map<string, number>,
  rest?: Map<string, number> | null,
): Map<string, number> {
  const next = new Map(current);
  for (const joint of joints) {
    const value = flippedValue(joint, current.get(joint.name) ?? 0, rest);
    if (value !== null) next.set(joint.name, value);
  }
  return next;
}
