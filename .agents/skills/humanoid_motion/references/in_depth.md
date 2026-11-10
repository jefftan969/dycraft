# Humanoid motion in depth

Defaults are for a realistic adult-proportioned body. Scale them to the character.

## 1. Footsteps and support

- **Designs that break it:** feet that chase a pelvis path and step only when a leg is about to overextend; feet that move with the root without stepping; a collision search free to put a foot wherever is clear; a single step clamp so large that one stride covers what should be three; hard-coded foot positions.
- **Weight transfer:** in a slow step the body shifts toward the supporting foot as the other unloads, and the receiving leg accepts the body before the next leg leaves. The body need not stay centered over one foot in brisk walking.
- **Swing:** the swinging foot clears the floor and arrives without a snap, traveling under the body about as narrow as the stance. If it cannot pass the stance leg, shorten the step or the swing rather than widening the path.
- **Pelvis:** its height follows the legs; avoid both a level glide and decorative bouncing. Keep the stance leg short of full extension so IK does not snap the knee.
- **Starting and stopping:** prepare support and unload the first stepping foot before moving off. Stop with shorter steps, finish with the feet under the pelvis, and let the body settle; never stop the root while the feet keep cycling.
- **Feet:** preserve the contact patch while planted. Rigid robot feet do not roll heel to toe, and any rotation of a planted rigid foot is slip.
- **Defaults:** walking pace about 1 m/s for an adult, slower when carrying or in tight space, scaled with leg length. A heavy lift may widen the stance for that interval. A run needs a running gait, not a sped-up walk.

## 2. Limbs within anatomy

- **Designs that break it:** a posture term negligible next to target error; a posture term that only pulls toward the previous frame; a solve restarted from a neutral seed every frame; elbow or knee direction chosen by collision cost, residual ranking, flip retries, or a pole fixed in world or armature space, which flips the limb when the body turns or bends; derive the pole from the torso instead. Warm-starting from the previous frame is fine as long as the posture term, not the seed, decides.
- **Collision repulsion is not a posture:** pushing the arm away from the torso is exactly what raises and flares the elbows.
- **Declared limits can be wrong:** a robot's declared ranges can allow backward elbows or back splits and still forbid a deep squat. Render the body at each bound before relying on a range.
- **Natural pose depends on the posture:** for work below the shoulder, upper arms stay near the torso with elbows bent and pointing down or back; overhead work, climbing, and pushing wide loads need a deliberate choice instead.
- **Orientation:** constrain the axes the task needs: a level tray, a pour, a screwdriver's axis. Forcing a full world orientation elsewhere makes the shoulder and elbow twist and lift.
- **Pelvis and thorax:** keep pelvis position, pelvis orientation, and thorax orientation relative to the pelvis as separate controls. When the pelvis turns, account for it in the chest's target so the rotation is not applied twice. Missing waist axes are absorbed by hips and shoulders, not by breaking the mechanism.

## 3. Reach and posture

- **Reach failures that get hidden:** clamping a limb short and logging the miss, letting IK stretch a bone, scaling a segment, inventing a mechanism the character does not have, declaring a collision allowance to excuse a limb. A strong posture term will expose misses; replan the body or grip rather than weakening it.
- **Low work:** choosing a posture from the target height is correct; letting the solver sink the pelvis until fixed hand and foot targets are met is not. For low work lasting more than a few seconds prefer a kneel or an upright-torso squat, if the mechanism's range allows. Walk crouched only under a low obstacle; otherwise stand, step, then lower.
- **Heavy lifts:** hips back and knees bent, load close to the body, back near neutral.

## 4. Timing and effort

- **Designs that break it:** durations from story beats or fixed frame budgets, then a global retime that raises every speed at once, footsteps included; every segment eased to a stop, so the path stops and starts at each waypoint.
- **Defaults:** turns spread over several steps at roughly 90 degrees per second or less; rising from a deep squat on the order of a second, longer under load; hands moving at the speed a person reaches.
- **Balance and effort:** in slow, static, or loaded phases keep the center of mass over the supports. Push the hips back when bending forward, lean away from a load held in front or to one side and let the free arm counterbalance, and shift weight before lifting a foot. Heavy objects move more slowly, driven by the legs and torso rather than the arms alone.

## 5. Travel and turns

- **Designs that break it:** a single heading held for a whole film; a carry at a fixed yaw; heading authored independently of the route.
- **Turns:** step through any turn larger than about 45 degrees. The head, or the chest if there is no neck, leads and the pelvis follows; a whole body rotating as one block at constant speed reads as robotic.
- **Gaze:** the head looks at a target before the hand reaches it, and at the path before and during a turn or walk.
- **Deliberate backward or sideways travel** — pulling a cart, easing a panel through a doorway — uses short, explicit steps with the head checking the path. A side-step needs lateral placements of its own; a forward gait with a rotated chest is not one.
- **Wheeled bases** may turn in place, but a differential drive cannot move sideways.

## 6. Held objects

- **Designs that break it:** an object animated along its own world path with hand IK absorbing every rise, sway, and turn of the body, so the object floats on a rail while the character moves under it. Object-driven hand IK is fine when the object's targets were derived from coordinated body and arm motion; avoid cyclic hand-object constraints.
- **Carrying pose:** choose the grasp and a carrying region relative to the chest or pelvis. Below-shoulder carries keep the elbows low; carry a wide or heavy object lower or closer rather than raising the elbows. The hands and object inherit the torso's motion, softened but not canceled by the arms, with some lag and settling as the body accelerates, turns, and stops. The free arm and torso may adapt asymmetrically; do not mirror a walking arm swing through a grasp. A heavy load is borne from below or by its handles — palms up under an edge, or a firm grip on a handle — with the elbows low and in front; hands resting on top of a load or pressed flat against it do not carry it.
- **Grip side:** a person holds a cup, bottle, jug, tool, or box from its outer side: the palm wraps the side facing away from the body or the side, the back of the hand faces outward or up, and the thumb lies on the near side. A grip whose back of the hand faces the body (the palm turned outward, the hand hooked around to the object's near side) is a reversed grip people use almost never; it twists the forearm toward the end of its rotation. Set the hand's grasp frame from this rule before solving, rather than taking whatever orientation the solver or the object's axes give. When the object's outer side is out of reach, turn the body, step, or change hands instead of reversing the grip. Check it from the side and from above: on every held object the back of the hand should face away from the torso.
- **Fingertip grip:** thin and small objects are held in a precision grip at the fingertips — a pen or marker between the thumb, index, and middle fingers as for writing, a card or a sheet pinched between the thumb pad and the side or pad of the index finger, a cap or small lid pinched by its rim, a phone cradled in the fingers with the thumb on its face. The palm stays off the object and the other fingers curl loosely. Use a power grip (the whole hand wrapped around) only for handles, bottles, cups, jugs, and tools that take force. Place the contact at the finger pads, and check that the object sits in the fingers rather than buried in the palm.
- **Grasp:** it constrains the hand to the object, not the object to the world. The fingers do not slide on the held surface. Two-handed holds solve both arms against compatible grasp frames on one carrying pose; never drive the hands independently and let one grip stretch or detach.
- **Pickup and set-down:** settle a usable stance, approach along the grasp direction without sweeping through the object, close the grip, transfer support to the hands, then lift. At set-down, decelerate and make contact before releasing. Keep transforms continuous when changing parenting or constraint weights. Insertion or deliberate placement can need world-space precision during that phase only.

## 7. Continuity and life

- **Keying:** a channel keyed only where it changes drifts across unrelated beats when eased between distant keys — a crouch sinks back during a later gesture, fingers open during a strike. Key holds as well as transitions, or blend explicit windows back to a neutral pose. Blender's default Bezier eases to a stop at every key; bake solved motion per frame or set interpolation explicitly. Unwrap angles so a heading never jumps through 360 degrees, and keep quaternions sign-continuous.
- **Overlap:** start a movement with the gaze, then the torso, then the limbs, with a small preparation before and a settle after. Secondary motion follows its cause and settles when the cause ends.
- **Rest pose:** standing at rest, both arms hang relaxed at the sides, the hands beside the thighs with the palms toward them and the fingers softly curled, the elbows nearly straight. Return to it between tasks and whenever a hand has nothing to do, rather than leaving the hands on the hips, folded in front, raised, or hovering over the last object. A rig's T- or A-pose bind pose is not a rest pose.
- **Idle:** a free arm hangs relaxed, braces, or assists; it does not sit in a constant decorative pose. A robot's idle is small balance settling and attention, not noise. Repeated actions vary slightly and carry state such as fatigue.

## 8. Worn items

- **Designs that break it:** a worn item placed as a separate object and parented (or copy-transformed) to the pelvis as one rigid piece, so the thigh swings through it; an item positioned by hand per clip; a whole-film allowance between the item and the whole body to silence what that produces. A bound item lying against the skin can still overlap it slightly; declare that narrowly rather than separating the item from the body.
- **Binding:** rigid items take a bone parent at the bone they sit on. Soft items take the character's Armature modifier with vertex-group weights transferred from the nearest skin of the character's mesh, so every vertex follows the bones that move the skin under it. A Shrinkwrap after the Armature modifier, set to stay outside the surface with a few millimeters of offset, removes residual sinking. Apply the binding at the rest pose, and check it at the extremes of the performance: deep squat, high knee, twist.
- **What it does not fix:** a hand reaching into a pouch, or fingers at a dog's mouth, is a pose problem. Put the hand target outside the surface it touches and solve the pose to it; do not exempt the overlap.

## Measure and repair

- **Watching:** review with fixed diagnostic cameras, front and side, at playback speed, over a full gait cycle and the entry and exit of every transition; camera motion can hide foot slip or imitate sway, and sparse stills cannot establish timing. Inspect world pelvis motion separately from waist motion relative to the pelvis.
- **Measuring:** check every frame of the baked motion for the symptoms in SKILL.md. A hard-check pass or a `JointLimits` pass does not show natural motion or balance. Use `check_motion` for local geometric checks and keep the full acceptance workflow.
- **Feet:** for each foot, record sole height and horizontal travel on every frame. A sole within a few millimeters of the floor should not travel, and a walking step should lift the foot clear of the floor. Measure before and after any smoothing, retime, or re-solve of a walk: smoothing a stepping trajectory flattens its swing into a glide.
- **Repairing:** frame-numbered special cases, per-frame root corrections accepted or rejected independently of their neighbors, and joint-space splices over solved intervals reintroduce snaps, teleports, and flips. A repair sized to be safe everywhere is usually too large for the interval that needed it; check that it did not add a crouch, shrug, widened stance, bowing swing, raised elbow, or lost contact nearby. Adding oscillation is not a repair.
