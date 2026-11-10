---
name: mocap_reference
description: Prefer real motion capture as the basis of a character's motion, from the capture libraries installed under `references/mocap/`, for example human (the CMU library, ~2,500 clips of walking, running, turning, sitting, standing up, bending, reaching, carrying, gestures, sports, and more) and, where its optional captures are installed, dog (the MANN library: walking, trotting, galloping, sitting, lying, sniffing). Read when animating a person, a human-like character, or an animal, before writing any locomotion or full-body motion: start from a captured clip and edit it, and write by hand only what no clip covers. Not for robots: solve a robot's joints with `h.ik`.
---

# Motion-capture reference

Hand-authored walking and full-body motion tends to come out stiff, with feet barely leaving the ground, the body bobbing and lunging, and no arm swing. Recorded motion has the timing, weight transfer, and secondary motion right. Bring it onto your character and edit it into the performance.

**Start from a capture by default.** For locomotion and whole-body motion — walking, stepping backward or sideways, turning, crouching and rising, sitting down and standing up, a dog's sitting, lying, rising, walking, and following — begin from a clip: retarget it, then edit it into the performance (path, timing, head and hands, contacts). Write motion by hand only for what no clip covers, such as a grasp on a particular object or a contact with your scene, layered on top of captured body motion, and say in your notes which motions you authored by hand and why. Taking only a few captured key poses and interpolating between them does not count as using the capture: the transitions — the weight shift, the timing, the secondary motion — are what the capture is for. How you bring a clip in, and what you change, stays your decision, and you still own the result and its checks.

## What is there

The libraries installed for this run are the directories under `references/mocap/` in your run directory (in Blender, `/references/mocap/`). List them before you search.

Every library DyCraft installs is laid out the same way, and one you add can follow the same layout:

```
references/mocap/<library>/
  NOTES.md        what it holds, its facts, its index and its helper, in the sections below
  index.txt       one line per clip or span: <clip> TAB <start>-<end>s TAB <label> TAB <notes>
  JOINT_MAPS.md   which of its joints drives which bone, for each rig its helper maps
  data/           the clips
```

**Read a library's `NOTES.md` before you search it.** Its sections are the same in every library:
- **What it holds:** the bodies and performers, and the terms to credit it under.
- **Facts:** where its clips are in your shell and in Blender, their file format, frame rate, up axis and import settings, units, any rest frame that is not part of the performance, and the skeleton's joints.
- **Index:** how its labels were made, and which ones mislead.
- **Helper:** the module in Blender's `/tools` that brings its clips onto a rig, when it has one: the rigs it maps (with `JOINT_MAPS.md`), how to call it, what it does, its report, what to check after it, its edits, and its known limits.

A directory without `NOTES.md` is a library the user added in its own form: read whatever came with it, and bring its clips in yourself (see "Doing it yourself" below), since each helper reads only its own library's files. A library that is not there is not installed: use another installed library that has the body, or animate it by hand, as you would without this skill.

**Finding a clip.** Every index has the same line format, so one search covers every installed library: `grep -i -E "walk.*back|backward" references/mocap/*/index.txt`. Read the notes field as well as the label: it names the performer where the library records one, and the performer's body (a child, a pregnant woman, a dancer) shows in every frame, a plain standing still included. Many clips hold several actions in sequence ("walk forward, turn around, walk back"), so find the part you want by its motion, not by assuming it fills the clip. Labels are coarse, and some mislead: a library's notes name the misleading ones it knows of.

**Look before you use it.** Import the clip alone into a scratch scene (or call its helper's `inspect_bvh`). Read its frame range and root path, and draw a few frames or a strided Workbench clip. Pick the frames you need.

## Using a clip

How a clip's motion reaches your character is your decision. A library's helper does the mechanical part for the rigs its notes name, and you can also do it yourself. Using clips only as visual reference (timing, stride, weight shift, arm swing) and authoring the motion yourself is the fallback for a motion no clip covers well, not the default.

**Doing it yourself.** Possible approaches:
- compute each target bone's rotation from the source pose with `bpy`/`mathutils`;
- drive the target from the imported source skeleton with constraints (Copy Rotation / Copy Location, IK toward source joint positions), then bake to keys;
- read the capture file yourself and work from joint positions rather than rotations.

Start from the library's facts: import with its up axis, convert its units to meters, resample its frame rate to the scene's, and leave out a rest frame that is not part of the performance.

Whatever you choose, look at the result: render it and measure it.

**Differences between the capture's skeleton and your rig's that any retarget has to account for.** A helper handles each of these for the rigs it knows, and a hand-written retarget must too. Each one, missed, gives a visibly wrong result:
- **Rest poses differ.** Captures often rest in a T-pose, as CMU's do; most imported characters rest in an A-pose. A rotation copied from one rest pose to the other lands the arms somewhere else.
- **Bone axes differ.** The same anatomical bone can point a different way in each rig. An FBX-imported Biped head bone points at the jaw, forward and down, where CMU's points up. Anything that aligns bone axes can tip the head back by nearly 90°.
- **Rest facing can differ from the clip's facing.** If rotations are measured against a rest pose that faces another way, a forward lean comes out as a backward or sideways one.
- **Proportions differ.** Hip translation copied at the actor's scale makes the feet slide or over-reach. The leg-length ratio (hip → knee → ankle) is the usual scale, not a constant: libraries record in different units.
- **Ground contact is not preserved.** Different ankle-to-sole depth and leg lengths leave the feet hovering or sinking by centimeters. Lower the pelvis by the skin's lowest point on each frame, then pin the stance feet (foot locking), because a lowered stance foot still slides.

**Editing into the performance, by any means.** Every result is an ordinary Action. Edit its keys directly, layer corrections on NLA tracks (Add/Combine), retime strips, or write your own tools, whichever serves the shot; a helper's edits are conveniences, not the only way.
- Cut the frames you need.
- Turn and translate the root to place the motion on your path.
- Retime with care: more than about ±20 % tends to read as wrong, and a different clip reads better than a stretched one.
- Join clips at matching gait phases.
- Mocap and authored motion can be combined, for example a mocap gait with the reach, gesture, or grasp solved for the task.
- Keep the result keyed or in NLA strips like any other motion, so it stays editable.

**Check it like any other motion.**
- Feet planted while bearing weight.
- No limb flips, and joints in range.
- Contacts and reaches met.
- Hands and feet. Captures rarely record fingers, and a retarget can leave a planted sole pitched or rolled: look at the hands in close-up and the soles on planted frames, and pose what the capture lacks.
- No high-frequency jitter. Marker noise and per-frame corrections flicker in hands, feet, head, and root: smooth before you accept the motion (with a helper's smoothing, or filter the F-curves yourself), then look at a slowed render.
- No self-collision. A skinned character is one part, so the harness's penetration check never tests it against itself. On a character bulkier than the performer, check arms against the torso and legs crossing, measured on the skinned meshes.
- The character doing what the brief says, rather than what the clip happened to do.
- Held poses that stay alive. Hold with a captured standing, waiting, or idle span, looped if needed, not with one frame held still, and do not slow a capture below about half speed: a frozen frame reads as a statue, and a capture slowed to a crawl jolts where it rejoins live motion.

[humanoid_motion](../humanoid_motion/SKILL.md) still applies to everything you change.

## What it is not

- It does not cover every body. A library holds the bodies it recorded (DyCraft's installs hold people and one dog): another animal needs an installed library that has it, or one whose notes say it fits that body, and a robot or non-human proportions need a deliberate mapping.
- It is not a substitute for the task's specific actions. A clip of "pick up box" does not reach *your* box: retarget the gait and posture, then solve the contact.
- The libraries have little for task work: handling equipment, lifting a loaded bar, carrying on stairs, a dog's low pickup. Search for the motion once; when no clip fits, retarget the body around the task (gait, posture) and solve the task yourself rather than searching on.
- It is not proof of quality. A badly retargeted clip is still a bad animation, so look at it.
