# The CMU library (people)

The skill `mocap_reference` covers what applies to any library: finding and looking at a clip, retargeting it yourself, editing it into the performance, and checking it.

## What it holds

CMU Graphics Lab Motion Capture Database, converted to BVH: 2,548 clips from 113 subjects (numbered 1–144, with gaps). It is free for research and for use in products, but the data itself may not be resold, even converted; credit "CMU Graphics Lab Motion Capture Database" in your notes. `README.md` and `READMEFIRST.txt` here are the BVH conversion's own notes.

## Facts

- **Clips:** `data/<SSS>/<SS>_<CC>.bvh`, read-only, at two paths:
  - **In your own shell**: `references/mocap/cmu/data/<SSS>/<SS>_<CC>.bvh` under your run directory (`$DYCRAFT_RUN_DIR/references/mocap/...`), for browsing, grepping, and reading files.
  - **Inside Blender** (`h.blender`, `h.probe`, your scripts): `/references/mocap/cmu/data/<SSS>/<SS>_<CC>.bvh`. This path exists only in the Blender sandbox; it is normal for your shell not to see it.
  - Example: subject 8, clip 1 is `data/008/08_01.bvh` under either root.
- **Format:** BVH.
- **Frame rate:** 120 fps.
- **Up axis:** Y-up; import with `axis_forward="-Z", axis_up="Y"` so the result is Z-up.
- **Units:** arbitrary (roughly 0.056 m per unit). Scale by leg length, not by a constant.
- **Rest frame:** **frame 1 is a T-pose**, not part of the performance.
- **Skeleton:** 31 joints: `Hips`, `LHipJoint`, `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase` (and Right), `LowerBack`, `Spine`, `Spine1`, `Neck`, `Neck1`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand` (and Right), plus finger stubs. "Left" is the actor's own left.

## Index

`index.txt`, one line per clip: the whole clip's span, CMU's own label, and its subject's description (`08_01<TAB>   0.0-   2.3s<TAB>walk<TAB>subject 8: walk`). The install writes it from CMU's own `cmu-mocap-index-text.txt`, kept beside it, and adds the 113 clips that file leaves out, with no label. Grep it for the action you need, for example `grep -i -E "walk.*back|backward" references/mocap/cmu/index.txt`, and read the subject: subjects 111 and 114 are a pregnant woman. Many clips hold several actions in sequence ("walk forward, turn around, walk back"), so find the part you want by its motion, not by assuming it fills the clip. The labels are CMU's own and coarse, and some mislead: 131_01 and 131_02 ("start walk stop") are gestures, 77_01 ("looking around"), 26_09 ("bend, pick up") and 02_06 ("bend over, scoop up") are crouches, and 79_93 ("weight lifting") is overhead. Look at a clip before you build on it.

## Helper: `mocap_retarget`

Inside Blender (`h.blender` code, your scripts, `h.probe`), `import mocap_retarget as mr`:

```python
import mocap_retarget as mr
mr.inspect_bvh('/references/mocap/cmu/data/008/08_01.bvh')
# -> {'frames', 'fps', 'seconds', 'root_travel_m', 'root_displacement_m'}  (scene untouched)
rep = mr.retarget(bpy.data.objects['rig/trainer/body'],
                  '/references/mocap/cmu/data/008/08_01.bvh',
                  start_s=0.5, end_s=4.0, name='trainer_walk', heading_deg=0)
# rep['action'] names a new Action keyed on frames 1..N at the scene fps.
```

- **What it does.** It maps the CMU joints onto the rig (3ds Max Biped, Mixamo/ReadyPlayerMe, or Unreal-style names; `mr.bone_map(armature)` shows the mapping, `mapping={cmu_joint: bone}` overrides it). It also handles the rest-pose and bone-axis differences and scales root travel by leg length. It reads the CMU library's files: their joint names, their unit (about 0.056 m) and their T-pose first frame. Retarget a clip from another library yourself (the skill's "Doing it yourself").
- **Facing.** The actor starts facing the character's own front; `heading_deg` turns the performance from there.
- **Ground and feet.** Contacts are read off the source skeleton, each stance against its own neighborhood (CMU's floor is not level). `ground=True` lowers or raises the body on those frames so the planted sole sits at the rest pose's sole height, and interpolates in between, so a run keeps its flight. The sole is the skin's foot vertices carried by the foot bones, so clothes never set the ground. `foot_lock=True` holds the contact point where it touched down: it follows heel-to-toe roll, sets the planted sole on the floor and lifts any sole that would pass through it. It shifts the root first, then applies two-bone leg IK, easing over 2 frames either side of each stance. Every edit below reuses the input's contact timing and re-locks.
- **What it leaves alone.** The new Action is left *unassigned*. Your armature's own action and NLA are restored, so nothing you already placed changes. Put it on an NLA strip where it belongs, at the frame and position you choose (the root starts near the rest position), and edit it like any other motion.
- **The report.** It returns `leg_scale`, `ground_offset_m`, `stance_slip_p90_mps` (the contact point's speed while planted, before and after locking), `planted_fraction` (each foot's share of frames planted; a jog is ~0.3, a walk ~0.6), `stance_sole_gap_mm` (how far planted soles sit off the floor), `sole_below_ground_mm`, `sole_calibration_frames` (per foot; under 2 means that foot's angle is the rest pose's, which can be off by tens of degrees: check its sole on contact, or take another clip) and `root_travel_m`, plus `warnings`. Read them, and read the warnings: one foot planted far less than the other, soles off the floor, or an uncalibrated foot each mean the clip and the character disagree there, and a different clip or range may fit better.
- **Limits.** Wrists follow the clip, aligned by the palm (hand → middle-finger base, hand → thumb), because CMU's hand joint has no length of its own; a rig without middle-finger and thumb bones, or `hands=False`, keeps the hands at rest relative to the forearm. Fingers, face, and twist bones keep their rest pose: CMU records no finger motion, so grips and gestures are yours to pose. The helper does not reach *your* props: solve hand contacts, gaze, and task-specific actions on top. It applies to humanoids only (not quadrupeds or birds). On a rig it cannot map it raises an error naming the CMU joints it found no bone for.

**What to check and clean after a retarget.** The helper gets the body right; treat the ends of the limbs and the high frequencies as yours to inspect:
- **Hands and below the wrist may be wrong.** CMU records one coarse hand joint, a finger stub, and a thumb, and no fingers. The wrist is aligned by the palm and its twist is moved into the forearm (capped at about ±90°), but a fast or extreme forearm turn can leave the palm off, and fingers stay at rest. Look at the hands in close-up, and pose grips, gestures, and palm direction yourself where the shot needs them.
- **Ankles and feet may be wrong.** The foot is calibrated so a planted sole is flat, from the clip's own stance frames; a clip with few or no planted frames (jumps, sitting, kneeling) falls back to the rest pose and can leave the foot pitched or rolled. Check the sole against the floor on planted frames and the ankle angle in side view, and correct the foot (or pin it with IK) where it is off.
- **Expect high-frequency jitter, and smooth it.** Optical mocap carries marker noise, and the per-frame wrist, foot, and IK corrections can add frame-to-frame flicker, most visible in hands, feet, head, and the root. Retargeted motion is *not* filtered for you. Smooth it before you accept the motion:
  ```python
  sm = mr.smooth(arm, walk['action'], sigma_frames=1.0)   # Gaussian over time, feet re-pinned
  # sm['jitter_p95_deg'] -> {'before': 5.2, 'after': 2.3} on a dance clip
  ```
  About 1 frame (a 3–5 frame window at 24 fps) removes flicker without softening real motion; 2 or more starts to round off fast accents (a stamp, a clap). `bones=[...]` limits it to some bones, and `root_sigma_frames` sets the root separately. Or smooth the F-curves yourself; either way, look at a slowed render of hands and feet afterwards.

**Known limits of the helper.** Plan around these rather than discovering them in review:
- **Fingers never move.** CMU records no finger motion. Grips, pointing, and gestures are yours to pose.
- **Extreme arm twists are only approximate.** The forearm takes the palm's twist up to about ±90° from the upper arm; beyond that, the rest stays at the wrist, so in fast whirling arm moves (a dance) the palm can be off by tens of degrees.
- **A flat, level floor is assumed.** Grounding, foot locking, and the foot calibration all take the floor as a horizontal plane at the rest pose's height. Stairs, slopes, steps up, sitting on a chair, kneeling, or lying down need your own root placement and foot/hip contacts. A clip with no planted frames (jumps, sitting, kneeling) falls back to the rest pose for the foot, which can leave the sole pitched or rolled.
- **Deep squats** can leave about 10° of twist at the knee, from the foot-locking IK.
- **No self-collision, and nothing checks it.** On a character much bulkier than the actor (armour, a heavy build), arms and hands can pass into the torso or thighs, and one leg through the other in crossing steps or turns. A skinned character is one part, so the harness's penetration check never tests it against itself: a passing check says nothing about a forearm in the chest. Check those frames yourself, measured on the skinned meshes rather than on bones or proxies, and look from the film's camera and from an angle that shows the gap; fix it in the pose.
- **The thumb sits at a constant ~18° from the actor's.** The two skeletons splay the thumb differently. It does not move over time.
- **`follow_path` bends a straight clip.** It warns past about 90°/s of turning; a sharp turn wants a turning clip spliced in.
- **Untested:** characters with clearly non-human proportions (the chimp), jumps, flips, crawling, and rolling on the ground. Check these especially carefully.

**Joint names.** [JOINT_MAPS.md](JOINT_MAPS.md), beside these notes, lists which CMU joint corresponds to which bone in the three namings `mr.bone_map` knows: 3ds Max Biped, Mixamo/ReadyPlayerMe, and Unreal-style. Your character's `native_rig.anatomy.skeleton` says which naming its rig uses. Imported characters come with others too (`rigify`, `character_creator`, `unity`, and more) or with none the importer recognizes (`unknown`); for those, map by anatomical role yourself and pass `mapping={cmu_joint: bone}`. It replaces the automatic map, so name every joint you want driven.

**Editing the result.** These are conveniences, not the only way. Every result is an ordinary Action. Edit its keys directly, layer corrections on NLA tracks (Add/Combine), retime strips, or write your own tools, whichever serves the shot. The helper also makes the standard small edits, each returning a *new* Action and a report:

```python
loop = mr.cycle(arm, walk['action'], seconds=12)              # one gait cycle repeated, travel accumulated;
                                                               # loop['step_frames'] = the two steps, should be about equal
route = mr.follow_path(arm, loop['action'], [(0, 0), (0, 4), (3, 7)])   # world x, y; corners rounded
slow = mr.adjust(arm, route['action'], speed=0.9, stride=1.05)  # retime / scale step length, feet re-pinned
joined = mr.splice(arm, walk['action'], stop['action'], blend_s=0.25)   # join at matching foot landing
```

- **`cycle`** finds two consecutive landings of one foot and repeats the stride between them, spreading the small mismatch at the seam. It needs a steady, straight stretch; `cycle_turn_deg` says how much the cycle turns, and `step_frames` gives the two steps inside it. A warning about uneven steps means a limp or skip that will repeat every cycle: take another stretch or clip. `symmetry` lists every full stride in the clip with each foot's contact length and the two step intervals (frames) and an `uneven` score (0 is perfectly even); judge evenness by the contacts too, not the steps alone (CMU 16_56 steps 9/9 but holds the left foot down 7 frames and the right 5). `stride=<index>` takes another stride. At 24 fps a jog's contact is 3-5 frames, so a score of 1-2 is one frame of rounding; compare a few clips and look at the run in motion. Even jogs measured on a humanoid: 35_22 and 35_17 (uneven 1), 16_36 (2).
- **`follow_path`** lays the clip's forward progress along your path, keeps its sideways sway, turns the body with the path, leans it into the turns, and re-pins the feet. It drops frames past the path's end; if the clip is shorter than the path, lengthen it with `cycle` first. The lean balances the turn (tan = speed² × curvature / g, capped at 15°, reported as `lean_deg`); `lean=0` turns it off, `lean=0.5` halves it. If you lean a body yourself, take the speed and curvature averaged over a stride: a runner's pelvis speeds up and slows down inside every step, and a lean from the per-frame speed pumps 5° each step and shakes the upper body sideways. Lean the body, not the feet: soles stay flat and planted feet stay put.
- **`adjust`** resamples time (`speed` > 1 is faster) and scales horizontal travel (`stride`), re-pinning the feet with leg IK.
- **`splice`** cuts the first clip at a left-foot landing and enters the second at its first left-foot landing. It moves and turns the second clip to continue from the first, and crossfades for `blend_s`.

These are standard edits, and they hold up for small changes. The report's `warnings` say when you are past that range:
- speed beyond about ±20 %;
- stride beyond about ±15 %;
- turns faster than about 90°/s;
- a path longer than the clip.

Past those limits a different clip reads better than a stretched one: a slower walk, a turning clip (`grep -i turn`), or a start/stop clip spliced in.
