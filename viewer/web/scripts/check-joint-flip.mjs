// Pin "Flip joints" to the two properties the button is only useful under.
//
//     npm run check:joint-flip
//
// The Parts tab's flip mirrors every joint to the opposite side of its own
// range at once, to make articulation mistakes visible: a scene commits with
// its doors, drawers, and lids shut, so the rest pose it renders in says nothing
// about whether they open the right way.
//
//   1. **Every value stays inside the joint's own limits.** The point of the
//      pose is to look for interpenetration, so a value past a limit would
//      manufacture exactly the fault being looked for — and the current pose it
//      flips from can legitimately be outside the range (a clip wrote it, or a
//      continuous joint spun past one turn).
//   2. **Two clicks are exactly a round trip, from any pose.** `flipped =
//      lower + upper - current` is a reflection, so it is its own inverse for
//      every starting value, and no "flipped" flag exists for the per-joint
//      sliders to contradict.
//
// **Why a mirror and not the far end of the range.** The two are the same only
// for a joint sitting *at* a limit, as a cabinet door, a drawer, and a lid do.
// On a rig resting mid-range, the far end would throw the model to one extreme
// and then to the *other*, so the second click would never come back.
//
// Plus the joints a flip must not touch: `fixed`, the poses the viewer cannot
// drive, a `mimic` joint (resolved from its source by the controller, not set),
// and a revolute or prismatic joint whose URDF limits are missing or
// degenerate, which cannot move at all.
//
// Transpiled through the esbuild vite already installs, like check:depth-bias.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-joint-flip-"));
const failures = [];

const check = (ok, msg) => { if (!ok) failures.push(msg); };
const near = (a, b) => Math.abs(a - b) < 1e-12;

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

/** A kitchen's worth of joint shapes, as `urdf-parser` produces them. */
const JOINTS = [
  // A cabinet door: shut at zero, opening one way only.
  { name: "door", type: "revolute", parent: "carcass", child: "door",
    axis: [0, 0, 1], limit: { lower: 0, upper: 1.92 } },
  // A drawer, in meters of travel.
  { name: "drawer", type: "prismatic", parent: "carcass", child: "drawer",
    axis: [1, 0, 0], limit: { lower: 0, upper: 0.46 } },
  // A symmetric joint whose rest pose sits mid-range.
  { name: "faucet", type: "revolute", parent: "sink", child: "spout",
    axis: [0, 0, 1], limit: { lower: -0.7, upper: 0.7 } },
  // An asymmetric one resting mid-range, which is what a robot's joints are,
  // and which a "far end" rule would get wrong.
  { name: "elbow", type: "revolute", parent: "upper", child: "fore",
    axis: [0, 1, 0], limit: { lower: -1.04, upper: 0.83 } },
  // No limits at all.
  { name: "wheel", type: "continuous", parent: "cart", child: "wheel",
    axis: [0, 1, 0] },
  // The three that must not move.
  { name: "shelf", type: "fixed", parent: "carcass", child: "shelf" },
  { name: "door_twin", type: "revolute", parent: "carcass", child: "twin",
    axis: [0, 0, 1], limit: { lower: 0, upper: 1.92 },
    mimic: { joint: "door", multiplier: -1, offset: 0 } },
  { name: "pinned", type: "revolute", parent: "carcass", child: "pinned",
    axis: [0, 0, 1], limit: { lower: 0, upper: 0 } },
];

const MOVABLE = ["door", "drawer", "faucet", "elbow", "wheel"];
const FROZEN = ["shelf", "door_twin", "pinned"];

try {
  const flip = await bundle(
    join(WEB, "src", "components", "viewer3d", "joint-flip.ts"),
    "joint-flip.mjs",
  );
  const { buildFlippedJointValues, flippableJoints, flipRange, flippedValue } = flip;

  // ---- which joints the button claims ------------------------------------
  const flippable = flippableJoints(JOINTS).map((j) => j.name);
  check(JSON.stringify(flippable) === JSON.stringify(MOVABLE),
        `flippableJoints reports [${flippable}], expected [${MOVABLE}] — the count ` +
        `beside the button is what tells the user the pose will do anything`);
  for (const name of FROZEN) {
    const joint = JOINTS.find((j) => j.name === name);
    check(flipRange(joint) === null,
          `${name} (${joint.type}${joint.mimic ? ", mimic" : ""}) reports a flip range; ` +
          `it cannot be posed`);
  }

  // ---- the rule itself, on the range it was specified against ------------
  //
  // "for joint limits of 10-90, a joint at 20 would go 20 => 80 => 20. a joint
  // at 45 would go to 55. 10 would go to 90 and 50 would stay put."
  const spec = { name: "spec", type: "revolute", parent: "a", child: "b",
                 axis: [0, 0, 1], limit: { lower: 10, upper: 90 } };
  for (const [from, to] of [[20, 80], [80, 20], [45, 55], [10, 90], [90, 10], [50, 50]]) {
    const got = flippedValue(spec, from);
    check(near(got, to), `on 10..90, ${from} flipped to ${got}, expected ${to}`);
  }

  // ---- 1. inside the limits, from any starting pose ----------------------
  const rest = new Map([
    ["door", 0], ["drawer", 0], ["faucet", 0], ["elbow", 0], ["wheel", 0],
    ["shelf", 0], ["door_twin", 0], ["pinned", 0],
  ]);
  // Poses a driver can leave behind: past both ends, and a wheel spun round.
  const poses = [
    rest,
    new Map([...rest, ["door", 1.92], ["drawer", 0.46], ["faucet", 0.7], ["elbow", -1.04]]),
    new Map([...rest, ["door", 5], ["drawer", -3], ["wheel", 11.4], ["elbow", -9]]),
    new Map([...rest, ["faucet", -0.34], ["door", 0.96], ["elbow", 0.2]]),
  ];
  for (const [index, pose] of poses.entries()) {
    const flipped = buildFlippedJointValues(JOINTS, pose, rest);
    for (const joint of JOINTS) {
      const range = flipRange(joint);
      if (!range) {
        check(flipped.get(joint.name) === pose.get(joint.name),
              `pose ${index}: ${joint.name} has no range but moved to ` +
              `${flipped.get(joint.name)}`);
        continue;
      }
      if (joint.type === "continuous") continue;   // no limits to be inside of
      const [lower, upper] = range;
      const value = flipped.get(joint.name);
      check(value >= lower - 1e-12 && value <= upper + 1e-12,
            `pose ${index}: ${joint.name} flipped to ${value}, outside ` +
            `[${lower}, ${upper}] — a value past a limit manufactures the very ` +
            `interpenetration the pose is meant to reveal`);
    }
  }

  // A shut door opens; a joint resting mid-range mirrors across its midpoint;
  // a joint sitting exactly on the midpoint has nowhere to go.
  const once = buildFlippedJointValues(JOINTS, rest, rest);
  check(near(once.get("door"), 1.92), `a shut door flipped to ${once.get("door")}, not 1.92`);
  check(near(once.get("drawer"), 0.46), `the drawer flipped to ${once.get("drawer")}, not 0.46`);
  check(near(once.get("faucet"), 0),
        `a symmetric joint resting on its own midpoint flipped to ` +
        `${once.get("faucet")}; a midpoint has no other side`);
  check(near(once.get("elbow"), -0.21),
        `an asymmetric joint resting mid-range flipped to ${once.get("elbow")}, ` +
        `not the mirror -0.21 (= -1.04 + 0.83 - 0)`);
  check(near(once.get("wheel"), 0),
        `a continuous joint resting at its centre flipped to ${once.get("wheel")}; ` +
        `it mirrors about rest, so rest is a fixed point`);

  // ---- 2. two clicks are exactly a round trip ----------------------------
  //
  // From *any* pose, not only one that sits at a limit. This is what makes
  // the button its own undo.
  for (const [index, pose] of poses.entries()) {
    const there = buildFlippedJointValues(JOINTS, pose, rest);
    const back = buildFlippedJointValues(JOINTS, there, rest);
    for (const name of MOVABLE) {
      const started = pose.get(name);
      const joint = JOINTS.find((j) => j.name === name);
      const range = flipRange(joint);
      // A pose outside the limits is clamped on the way out, so the round trip
      // returns to the clamped value — the nearest pose the mechanism has.
      const expected = joint.type === "continuous"
        ? started
        : Math.min(Math.max(started, range[0]), range[1]);
      check(near(back.get(name), expected),
            `pose ${index}: ${name} went ${started} -> ${there.get(name)} -> ` +
            `${back.get(name)}; two clicks must land back on ${expected}`);
    }
  }

  // ---- the caller's whole-pose apply must not lose anything -------------
  //
  // `applyJointValues` replaces the explicit pose wholesale, so a joint left
  // out of this map falls back to its authored default rather than staying put.
  for (const [name] of rest) {
    check(once.has(name),
          `${name} is missing from the flipped pose; the caller's apply would reset it`);
  }

  // An object with nothing movable leaves the pose exactly as it was.
  const inert = buildFlippedJointValues(
    JOINTS.filter((j) => FROZEN.includes(j.name)), rest, rest);
  check([...inert].every(([name, value]) => value === rest.get(name)),
        "a model with no movable joint still changed pose");

  console.log(`flip: ${flippable.length} of ${JOINTS.length} joints movable ` +
              `(${flippable.join(", ")})`);
  console.log(`rest -> flip: door ${once.get("door")} rad, drawer ${once.get("drawer")} m, ` +
              `elbow ${once.get("elbow").toFixed(4)} rad, faucet ${once.get("faucet")} rad`);
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-joint-flip: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-joint-flip: ok");
