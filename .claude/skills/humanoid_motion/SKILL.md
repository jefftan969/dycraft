---
name: humanoid_motion
description: Plan, solve, and review humanoid motion when authoring animation — walking, turning, reaching, working postures, grasping, and carrying — and set up the solvers that produce it: cuRobo through `h.ik` where it is available, or your own footstep, IK, posture, and grasp solvers. Read before choosing body trajectories or writing any motion solver; applies across humanoid mechanisms rather than prescribing robot-specific joint curves.
---

# Humanoid motion

Every pose you deliver comes out of a solver you set up or wrote, and most bad motion is a solver doing exactly what it was written to do: a root dragged along a path, IK that only knows its targets, durations taken from story beats. The rules below replace those defaults.

Start by mapping anatomical roles onto the actual rig — pelvis/root, spine, hips, knees, ankles, feet, shoulders, elbows, wrists, hands, head — with their axes and limits. Do not assume G1 names, human proportions, a flexible spine, a neck, or articulated toes, and preserve the supplied mechanism.

For a robot, where the harness provides `h.ik`, cuRobo can solve the joints: a whole-body, floating-base solve over every frame of a clip in one call, with joint limits, self-collision, and scene obstacles, from the pelvis, foot, and hand targets you plan (for a Unitree G1, `references/ik/examples/g1_sequence.py` in your run directory runs as it stands, on the scene's own G1 model; `g1_arm_plan.py` beside it plans one arm around obstacles while the body holds still). It is a fast first pass that avoids collisions: smooth its output and solve contacts yourself. Use it or a solver of your own, whichever suits the motion; the rules below decide the targets either way, and balance, support, and grasp timing stay yours, since cuRobo does not model them.

Where the harness provides `key_joint(name, value, frame)`, apply each solved joint value with it. It keys the joint's own channel, in radians or meters, at a frame of the joint's active action, which is clip time while you author a clip and the output frame otherwise.

**1. Place footsteps first and move the body over them.**
Decide where and when each foot lands, keep a planted foot fixed, and move the pelvis over the supporting leg. Cover distance with more steps, not longer ones, and keep stride and stance in proportion to the legs. A stepping foot rises before it travels and stops traveling before it lands. A wheeled body plans its base path the same way.
*Look for:* planted feet sliding, feet gliding without steps, soles skimming the floor through a step, feet still traveling as they land or lift, feet hovering above or sinking into the floor, heels raised in a squat the task does not call for, strides or stances out of proportion to the legs, and legs trailing behind the body.

**2. Solve every limb within anatomy, toward a natural pose.**
Enforce each joint's anatomical range as a hard limit in every solve: hinges such as knees and elbows bend one way and never past straight. Define the ranges if the rig has none, and check declared ones on the rendered body. Targets never fix the whole pose, so give each solve a natural pose for the posture being held, and let it, rather than the solver, settle everything the targets leave open. Constrain only what the task needs of the hands and feet. Nothing checks a skinned body against itself: test the forearms and hands against the torso and thighs, the legs against each other in turns and crossing steps, and the armpit and elbow where a limb folds, and fix the pose or the weights rather than accept the overlap.
*Look for:* joints bent backward or locked straight; elbows winged during work below the shoulder, or knees caving or splaying; limbs flipping between frames; wrists or shoulders twisted to meet a forced orientation; and an arm or hand sinking into the chest or thigh, or legs passing through each other.

**3. Put the body where the task is, and choose the posture.**
Position the body so each target lies in comfortable reach. For low or far work, choose a squat, kneel, or lunge suited to the task and the body's limits, and return to standing when it is done. If a natural posture still cannot reach, move the body or change the grip; never stretch, clamp, or excuse a limb, and never let the solver lower the pelvis until the targets happen to be met.
*Look for:* reaches made by the arm alone while the torso stays frozen, unintended frog squats and stiff-legged stoops, crouches held after the task or used for walking, and hands stopping short of their targets.

**4. Time motion from distance, rotation, and effort.**
Derive each duration from how far the body travels, how much it turns, how much its posture changes, and what it carries, at speeds this body could produce. Never speed up finished motion to fit a runtime. Loads change balance, posture, and pace.
*Look for:* dashes, spins in a fraction of a second, snapping into or out of a squat, held objects flipping, heavy loads handled like light ones, and poses that could not balance on their supports.

**5. Face where you travel, and turn through the body.**
Heading follows the direction of travel: to reposition, turn toward the destination, walk, and turn back. Turns unfold over steps, led by the head, or the chest if there is no neck, with the pelvis following. Travel backward or sideways only when the task calls for it.
*Look for:* shuffling sideways or backward between nearby spots, spinning on planted feet, the body rotating as one rigid block, and a head that never turns relative to the hips.

**6. Held objects move with the hand.**
Grip from the outside: a hand holds an object with its palm toward the object and the back of the hand facing away from the body, so the hand closes on the object's outer side; the back of a hand almost never faces the body. A thin or small object — a pen or marker, a card, a cap, a phone held up to read, a lid by its rim — is held in the fingertips, between the pad of the thumb and the index and middle fingers, not clenched in the fist. Derive the object's motion from the grip, with the fingers closed on it and one shared carrying pose for two-handed holds. Keep transforms continuous when picking up and releasing.
*Look for:* the back of a holding hand turned toward the body, a hand reaching around to the object's near side, a pen, card, or cap clutched in a closed fist, gaps between hand and object, objects resting on an open palm or drifting in the grip, one hand of a two-handed hold detaching, and jumps at pickup or release.

**7. Keep motion continuous and alive.**
Hold each pose until its next deliberate change, keep rotations continuous, and choose interpolation on purpose. Stagger onsets across the body. A character at rest stands with its arms hanging relaxed at its sides, hands beside the thighs; idle characters settle and follow the action, free arms hang or assist, and repeated actions vary. Do not add a fixed sinusoid to make a channel move.
*Look for:* poses drifting between distant keys, stop-and-go at every key, rotation flips, frozen characters, free arms held in fixed poses, hands parked on the hips, in front of the body, or in the air while nothing is being done with them, and exact repeats.

**8. Worn items deform with the body.**
A pouch, belt, bag, strap, or garment is part of the character's surface, not a prop placed near it. Bind it to the same rig: parent a rigid item (a watch, a helmet) to the bone it rides on; give a soft item an Armature modifier on the character's armature, with weights transferred from the nearest skin (a Data Transfer of vertex groups), and a Shrinkwrap kept just outside the skin so it neither sinks in nor floats. Then it follows the hip, thigh, and chest as they move, and stays in contact by construction.
*Look for:* a pouch or belt floating off the body or sinking into it when the hips or thighs move, an item that keeps one rigid pose while the body under it bends, and a worn item kept clear of the body only by a broad collision allowance.

**Measure and repair.** Check these symptoms on every frame of the baked motion, not in stills or from a verifier pass. When an interval fails, change the plan or solver that produced it and re-solve the interval and its neighbors rather than adding frame-specific patches. After any repair, smoothing pass, or change of technique, re-measure every hold, touch, and planted foot in the interval; one that lowers penetration often does it by pulling hands off their holds or feet off the floor.

The rules are written for humanoids; for other bodies, translate the terms (feet, hands, head) to that body and apply the same principles. Stylized timing or an intended failure is fine as long as supports, limits, and contacts stay coherent.

## Reference

[In depth](references/in_depth.md) has one section per rule: the solver designs that break it, the coordination detail behind it, rough defaults, and how to watch, measure, and repair. Read it before writing any footstep, IK, posture, grasp, or repair code, and again before submitting.
