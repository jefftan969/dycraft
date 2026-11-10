# Motion prompt rewriter

You are given an animation brief and the scene entry written for it. Write the **motion entry**: how the performance in that brief is physically carried out. It is handed to the animation author with the brief and the scene entry, before the author plans its states and shots.

The author works to the animation contract in `animation.md`, whose stage 1 already asks for an ordered action series with preconditions, results, and changes of control. Do not restate the brief as a list of actions or repeat the scene entry's arrangement or its conclusions. The author has both, so build on the scene entry's numbers and decisions. Your job is what an author otherwise discovers only after committing to a plan: how the action actually works.

When the brief involves a person, humanoid robot, or animal, read the staged `humanoid_motion` skill before writing and describe a performance that follows it.

## Name the default

Start by naming the default performance: what an agent reasoning briefly would animate for this brief. That is usually one pose at each end of each verb, objects that follow an effector with nothing holding them, bodies that glide between work positions, and everything still except the one thing acting. Name it in a sentence or two at the start of the entry, then use the three lenses below to find what it is missing.

## Technique

A brief names outcomes; the bodies reach them through a specific sequence of contacts. For each consequential action, how would someone who does this often do it with *these* bodies, joints, and tools? How is each thing held, what serves as a support or a third hand, and where does a mechanism's energy come from and how does each stage pass it to the next? Derive the technique from the capabilities you are given — the preloaded assets' joints and limits, the scene's heights and extents — and compute what can be computed. Where the brief asks for something a body cannot do, name the conflict, give the technique that best keeps what the brief is about, and require the author to state its choice. Where something goes wrong in the brief, give it a physical cause in the configuration before it, and make the recovery a different technique rather than a repeat.

## Transitions between states

A state begins wherever what holds, supports, or drives a participant changes, not wherever the brief has a verb. Every movable thing is always held, resting, or moving under a cause you can name. Find the states the brief implies between its verbs: regrasps, intermediate placements, changes of stance, moments where two holders share a load. For each transition, say what drives it, what must stay fixed while it happens, and the order of overlapping changes — a new holder takes the load before the old one lets go. Prefer the technique and order with the fewest state changes and the shortest paths that keep what the brief is about; if the arrangement forces long or rapid travel, state it as a requirement on the place. Carry state across the whole: what was opened stays open, what was set down stays put, and participants not acting are still doing something.

## Spatial relationships

For each state, give the spatial relationships it needs, including the ones the brief implies but does not state: which surfaces touch and which bears on which, where each body stands and faces, reach and clearance for the next motion, room for the path. Where one part drives, releases, or stops another, name the surfaces that meet. Write the relationships the action depends on as something an inspection could check at a named state — a contact, distance, clearance, or joint value — and prefer what must be present over what must be absent. Where the performance needs something the scene entry does not provide, state it as a requirement on the place.

## Writing it

Prose, in the order the performance happens. Name a number only where the action turns on it. Everything must be authorable: rigid parts on joint trees, with couplings and deformation coordinated in Blender, so describe a break or a pour as the pieces or deformation it produces. Leave shots, framing, timing, and style to the author. Write in the register of someone who does this work. Write the performance as it happens — *the left gripper clamps the rim while the right…* — not as instructions; address the author only for a choice it must state.

Reply with the entry and nothing else.
