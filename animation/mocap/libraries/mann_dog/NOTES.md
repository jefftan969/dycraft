# The MANN dog captures

The skill `mocap_reference` covers what applies to any library: finding and looking at a clip, retargeting it yourself, editing it into the performance, and checking it.

## What it holds

The dog captures of Zhang et al., "Mode-Adaptive Neural Networks for Quadruped Motion Control" (SIGGRAPH 2018): 51 clips, 41 minutes of one dog at 60 fps. CC BY-NC 4.0; credit the paper in your notes.

## Facts

- **Clips:** `data/<clip>.bvh`, read-only. Shell: `references/mocap/mann_dog/data/<clip>.bvh`; Blender: `/references/mocap/mann_dog/data/`.
- **Format:** BVH.
- **Frame rate:** 60 fps.
- **Up axis:** Y-up.
- **Units:** centimeters.
- **Rest frame:** none; every frame is performance. `dog_retarget` takes the dog's neutral stand from `D1_001_KAN01_001`.
- **Skeleton:** 21 joints: `Hips`, `Spine`, `Spine1`, `Neck`, `Head`, `Tail`, `Tail1`; each foreleg `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand` (and Right); each hind leg `LeftUpLeg`, `LeftLeg`, `LeftFoot` (and Right); plus end sites at the paws, head, and tail.

## Index

`index.txt`, DyCraft's own, one line per labeled span: `<clip><TAB><start>-<end>s<TAB><label><TAB><notes>`. Labels: walk, trot, canter/gallop, stand, sit, lie, stand or walk with nose down (sniff), turn in place, jump, with speed, turning and transitions (`starts with: sit down`). They are automatic: look at a span before relying on it.

## Helper: `dog_retarget`

`import dog_retarget as dr` in Blender, for a four-legged rig standing on all fours in its rest pose. A rig with the rigify dog metarig's names (`spine.004`, `thigh.L`, `shin.L`, `foot.L`, `toe.L`, `front_thigh.L` … `front_toe.L`), such as the asset library's German shepherds, is mapped by those names. Any other is mapped by its skeleton's shape: the legs are the two mirrored pairs of chains that reach the floor, the front pair at the head's end, and each leg's paw is where it meets the floor, with thigh, shin, and foot above it. `dr.profile(arm)` returns the bones it chose for each role, or says why it found none (not standing on four legs, left and right legs not mirror images, no clear head end). Read it before you retarget. When a role is wrong, pass your own map, in the form `profile` returns, as `roles=` to `retarget`. [JOINT_MAPS.md](JOINT_MAPS.md) lists which capture segment drives which bone. It reads the MANN clips, by name, from `/references/mocap/mann_dog/`. Edit the capture first, then solve it once:

```python
import dog_retarget as dr
dr.inspect_bvh("D1_008_KAN01_002")                 # length, travel and the index's spans
c = dr.clip("D1_008_KAN01_002", 0.7, 6.5)          # a span; every edit returns a new clip
c = c.then(dr.clip("D1_009_KAN01_001", 3.3, 9.0))  # splice, cross-faded over blend_s
c = c.follow([(0, 0), (0, 3), (2, 5)])             # walk it along a path, armature space
rep = dr.retarget(bpy.data.objects["rig/dog/body"], c, name="dog_fetch")
```

Edits: `c.section(a, b)`, `c.speed(f)`, `c.stride(f)`, `c.in_place()` (a sit, lie-down, or stand-up where the dog stands: the capture's own steps and turns removed), `c.cycle(seconds=...)` (repeats one gait period of a steady span), `c.smooth(sigma_frames)`, `c.info()`. To walk or trot longer than a span, use `cycle`, which loops at two landings of one paw: wrapping a frame index (`f % n`) or replaying a stretch from its start jumps the legs at every seam, up to 150 deg in one frame.

Steps, a porch, a slope: pass the ground with `floor=`. `dr.floor_from_objects([porch, steps, path], arm)` samples the tops of those meshes into one; `floor=` also takes any `f(x, y) -> z` on arrays, in the armature's space. Each planted paw then stands on its tread, a swing climbs over the edge it crosses, and the body rises and pitches with the footing. Walk the whole route in one followed clip, up and down the steps, rather than splicing short pieces around them. A captured stride is longer than a tread is deep, so slow the dog on the steps (`c.speed`, `c.stride`) to about one tread per step: a stride that carries a forepaw two treads up folds the leg against the shoulder and the paw falls short.

The captured dog often looks up at its handler. Turn the head where the shot needs it with `head_pitch_deg=` (positive lowers the nose) and `head_yaw_deg=` (positive turns it to the dog's left): a number, a value per frame, or a function of time in seconds, e.g. `head_pitch_deg=lambda t: 35 * min(1, t / 0.8)` to lower the nose onto a scent. The neck bends over its length.

- **What `retarget` does.** The capture is smoothed in time first (`smooth_s=0.05` seconds; the raw tracks jitter). Each bone takes the rotation its captured segment makes away from the dog's neutral stand. Strides and travel are scaled to the rig's leg lengths. The dog is placed as one body, eased over a quarter second: pitched about the hips so its fore and hind paws reach the ground alike, tilted with the ground under its front and hind paws on steps, and set at the height its planted paws need; skin in the floor raises the whole dog. Each paw is then pinned to one point of the ground for each contact the capture shows, swings from one footing to the next along the capture's own arc, and the leg is re-solved at its own bone lengths (nothing is stretched), with the knee (elbow) kept on the side the joint can bend to. The toe lies flat while planted. As with `mr.retarget`, the new Action is left unassigned. The report gives the contacts, `scale`, `stride_scale`, `lift_max_m` (how far skin in the floor raised the dog), `paw_shortfall_m` (how far each paw ever missed its target) and `below_ground_max_m`.
- **Carrying something in the mouth.** `retarget` poses no grip, so it is yours, and two things decide whether it reads:
  - *Across the jaws.* A dog picks a roll, a stick, or a toy up crosswise, held behind the canines. Before the bite, turn the object where it lies so its long axis runs along the head's left-right axis at the bite frame (or bring the head round to it), then fix it to the head at the grip point. Fixed to the head as it lay, in line with the muzzle, it hangs end-on from the corner of the mouth and stands upright when the head lifts.
  - *The jaw opens only a little on these rigs.* Size what is carried to the gape (about 6 cm across for the German shepherds), and let the object follow the head, not the jaw, so it does not swing as the jaw moves.
  - At a handover, pass it while both the mouth and the hand hold it, with the hand closing on the end nearest it, and only then open the jaw.
- **Turning before sitting or lying down.** Turn the dog while it is standing or walking (a turning span, or a curve in `follow`), then sit and lie it down in place. Turning an in-place sit or lie-down about the hips spins the dog on the floor: the capture's dog lies down straight, and turned on the spot it pivots on its belly.
- **Other four-legged bodies.** The captures are one dog, so any other body moves as a dog would. Graded on the library's animals, a walk or trot reads well on dogs, cats and big cats, bears, cattle, horses, pigs, sheep and goats, rodents, and the beaver; on elephants it passes with caveats. Sitting is the weakest: on several horses and cows, and on a cat, a dog's sit reads as rearing or as a crouch, and a lie-down can end in a crouch. For a horse's or a cow's sit or lie-down, key the pose or take it from another source, and look at every sit and lie-down in the film.
- **Known limits.** One dog and one handler: in many spans the dog looks up at the handler, so check head pitch and choose spans where it looks where your shot needs. Gallops retarget cleanly; jumps are not reliable: limbs tucked in flight can still turn fast, and a jump onto a raised surface needs that surface as `floor`. Hold a pose with a captured standing or lying span, not a single frame repeated: a frozen dog reads as a statue. No mouth, jaw, or ear motion (and no grip, above). A sitting dog's tail or thigh skin can still dip a few centimeters. `follow` turns the whole dog about its hips each frame; a sharp turn wants a turning span spliced in. On descents a forepaw can stop several centimeters short of the lower tread, so check descents.
