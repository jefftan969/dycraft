# The Animation Layer

`--layer animation` gives one CLI author the scene SDK and a persistent native Blender project, and needs no input
scene. It requires CAD, the configured Blender interpreter, bubblewrap or user namespaces, ffmpeg, and ffprobe.

```bash
python -m dycraft.cli --backend codex --layer animation --prompt task.md
```

The author-facing contract is [animation.md](../agent/prompts/animation.md). The source SDK reference is staged
beside the run's harness as `source_reference.md`.

`--prior-animation <animation-run-or-bundle>` refines an earlier animation, whose generated baseline and editable
sources travel with the bundle. `--prior-scene scene.py` seeds a new design. Copy any helper or object sources it needs
into `design/`.

## Assets

People, animals, and robots come from the asset library in `assets/imported/`, described by the generated `INDEX.md`
and the hand-written `catalogs/asset_library.json`. Rebuild both with
`python -m dycraft.agent.assets.asset_library [--thumbnails]`. Every run gets the library read-only at
`references/assets/`. `load_asset("<bundle>")` stages the bundle the agent picks into the build's copy of `design/`,
with the same checks `--preload-asset` runs, and after a build it is listed in
`native/work/design_state/source/preloaded/manifest.json`.

`--preload-asset <name>=<bundle>` stages external geometry the design may place but not build, such as an imported
robot or character. It lands under `design/`, so it travels into `design_state/source/` with the rest of the source
tree and resolves for both the build and the Blender import. Reach it from `scene.py` with `load_asset("<name>")`, and
read `design/preloaded/manifest.json` for its size, root, parts, and joint ranges.

If an asset declares anatomical limits, `AnatomicalJointLimits` checks every authored frame against them alongside
`JointLimits`. An imported character (`agent/assets/import_gltf.py`) declares them per bone instead, in
`native_rig.anatomy`, when its skeleton uses a naming the importer recognizes
(`animation/blender/rig_anatomy.SKELETON_FAMILIES`) and its rest figure is a standing human: elbow and knee hinges
(bending backwards, swinging out of the hinge plane) and wrist, ankle, and head cones. The evaluation records every
armature's bone orientations (`bone_rotations`, with bone order in the index's `rigs`). They feed those checks and
`MotionContinuity`'s `bone` channel, a per-joint single-frame spike detector for the limb flips the body channels
cannot see on a skinned mesh.

## Operations

The prompt orders the work in four stages, which the table's Stage column refers to:

1. Action and shot plan.
2. Complete 3D storyboard: the shared scene, its assets and every shot's key moments, verified and inspected.
3. Continuous motion, with cameras, checked at the intended pace.
4. Film and delivery.

| Operation | Stage | Result and scope |
| --- | --- | --- |
| `plan(plan)` | 1; later revisions | Stores intent, criteria, and all shots/states; does not generate or semantically verify the plan |
| `build(refresh=None)` | 2; source revisions later | Compiles `design/scene.py` and synchronizes a private Blender candidate; source reports are separate from import success |
| `blender(code, stage='storyboard')` | 2 | Author-supplied static staging, including pose keys/IK |
| `storyboard(states=None, render_settings=None)` | 2; revised evidence later | Checks and renders the planned states through the native timeline cuts, recording evidence per state; does not schedule cuts, create poses, or inspect intervening motion |
| `accept_storyboard(reviews)` | End of 2 | Records author observations by state id after image inspection |
| `blender(code, invalidate_shots=None)` | 3–4 | Native motion, camera, and appearance authoring after storyboard coverage; a name to invalidate may be a shot, a clip, or a state |
| `probe`, `snapshot`, `motion_image` | 2–4 after build | Queries Bpy, renders views or creates optional temporal RGB images; no qualitative verdict |
| `review(focus=None)` | 2–4 | Advisory second opinion on selected still frames, including frames from authored animation; does not replace whole-film inspection |
| `compile`, `check_motion(start,end,shot=,clip=)` | Primarily 3–4 | Full timeline or interval checks; `clip='name'` resolves an NLA strip's extent in the worker. Neither rebuilds sources nor runs visual review |
| `ik` | 2 for feasibility; 3 for motion | Solver consultation; returned data does not become authored motion. Blender's own solvers are authored directly with `blender()` |
| `final_render(render_settings=None)` | 4 | Checks and renders the native camera sequence and saves final settings in the still-editable project; video only |
| `finish()` | End of 4 | Packages the same project revision and its ready film |
| `dispatch_object/scene/motion(name,brief)` | As useful within the stage order | Private optional contributors; explicit launcher integration |

All of these return job ids. Poll `job_result` for the verdict, and `contribution_status` for contributor jobs.
`job_cancel` and `cancel_contribution` request cancellation. Project edits serialize. Inspection and check jobs
capture immutable snapshots. Failed or stale candidates cannot replace current work.

Harness frame indices are zero-based, and native Bpy frames are `scene.frame_start + output_index`. Planned shot
cameras are render overrides, not a camera schedule. The author makes the cuts in Blender.

For a draft sequence, `render()` takes a named frame set, an interval given as output indices, seconds, or a shot id,
or the whole timeline. It draws through the native cuts, a named camera, or a placed diagnostic one, and returns the
PNGs under `previews/<job>/`, outside the workspace, plus an encoded clip whenever more than one frame was drawn. It
runs no checks and writes no receipt, so a draft is available at any point and satisfies no part of delivery. A named
set need not be contiguous, so a strided review pass is an ordinary call. `snapshot` is the same operation with
encoding off.

`inspect()` returns structured evaluated facts (per-object world position and bounds, camera, elapsed time) for up to
four frames, paginated, with no image. `storyboard()` carries a shot's review forward when its images come back
pixel-identical, so a design repair costs a fresh review only where the picture moved.

`review()` renders fresh seeds from the current blend at the planned shot-state frames and gives the reviewer their
paths to open. It can request other frames or views, 1–32 frames per additional render call. `focus` is free text
naming a specific uncertainty, and selects no frames. The reviewer is asked to name the images it inspected and its
observation limits, which is self-reported, not machine-verified. Returned `seed_views` identifies generated images.
`reviewed: true` means the reviewer returned, not that every image was read or the whole film assessed. The reviewer
does not receive the encoded MP4.

## Rendering

A bare `final_render()` draws Cycles with 8 denoised samples at the plan's delivery floor, 960x540 pixels by default
or a larger `delivery.min_pixels`, in the project's aspect ratio. The blend itself is stored with Cycles at 1920x1080.
Storyboard and review share an appearance profile of Cycles at 640x360, because both are judged for materials,
lighting and composition. Drafts, diagnostic snapshots, and motion images default to Workbench at 1280x720. Denoised
Cycles is both cheaper and closer to a 256-sample reference than Eevee, and Workbench answers the geometry and framing
questions a diagnostic asks. Pass the intended engine and dimensions explicitly rather than assuming native settings
are retained. Neither final render strategy writes audio, so a brief's requested audio stays an unmet delivery
requirement.

## Checks

Temporal events go in the `events` list of the scene's `dycraft_declarations` JSON. The predicates are `touch`,
`separated`, `joint_at`, and `above`, with inclusive time windows in seconds, and `mode='all'` (default) or `'any'`.
Plan criteria written as text and visual claims in the source do not create checks. A local check defers an event
unless its entire window is included, and an empty event list passes without proving an interaction.

`hide_render` meshes are excluded from penetration checks; off-camera and occluded geometry takes part. Collision
declarations accept original identities and exact native geometry names, including mixed pairs and time windows, and
reports list exclusions. Unused joint ranges are not swept. `SupportConnected`, room containment, source geometry and
mechanism sanity, and relational assertions check the authored source configuration; there is no dynamic force or
support solver. Numerical or unsupported failures in required scope block delivery. Continuity and visual findings are
advisory.

Storyboard states are the configurations the film passes through, bound to a clip in `scene['dycraft_states']`, so a
retime moves them rather than discarding their reviews. Coverage and review are per state. Changing the plan or the
synchronized design invalidates the recorded revision. A state's review goes stale when an instance its board
recorded changes or is removed, a board that recorded no instances goes stale on any design change, and an added
object is caught by the next check that draws the state. After arbitrary Blender edits the author names the affected
shots.

## Rebuilding Sources

Generated Blender objects are named `link/`, `origin/`, `joint/`, and `part/`. New source geometry replaces generated
mesh data while keeping compatible object ids and animation. Appearance imports once;
`build(refresh={'materials': ['instance/part'], 'lighting': True})` refreshes an explicit scope. Shape keys affected by
geometry changes, topology-dependent bindings, changed joint axes or types, reparenting, and referenced removals produce
inspectable private candidates: pass their issue ids as `accept_changes`, with optional `repair_code`, on a later
`build(candidate=...)`. Acknowledging a change records intent, and the final full-frame checks still apply.

## Delivery

The final bundle holds `animation.blend`, the film and its camera and render receipt, evaluated geometry,
sources/catalog/baseline, the plan, storyboard evidence, portable resources, and edit history. Manifest hashes cover
the film as well as the project. Rendering validates replay and the encoded dimensions, H.264, frame count and FPS. An
edit or source change makes an older film ineligible for completion. A dispatched motion contributor returns a
checked private project without its own film; only the main author completes the deliverable.

Blender's export limits apply: realize geometry instances before verification, bake required native caches, and keep
dependencies packed or relative to the workspace.

## Reading a Run Afterwards

Every operation appends to `native/journal/` the code it ran, the authored source tree it ran against, and, for the
ones that save, a fingerprint of the project they produced. The journal is in the run's internal folder: `src/` for a
run launched from the command line, the run folder itself for a sub-run. `Journal.timeline()` joins the records into
"operation N changed this much", which finds the call that introduced a regression without opening a `.blend` or
re-running anything.

```python
from dycraft.animation.blender.journal import Journal
from dycraft.run_layout import internal_dir
journal = Journal(internal_dir(run_dir) / "native/journal")
for row in journal.timeline():
    print(row["seq"], row["operation"], row["changed"])      # None = scene untouched
record = journal.records()[-1]
journal.resolve_arg(record["args"]["code"])                  # what that call ran
journal.read_manifest(record["src_manifest"])["files"]       # ...and its helpers
```

It is diagnostic only. Nothing reads it back into a run, it lives outside `native/work/` so it never enters `version`
or a delivered bundle, and `DYCRAFT_ANIMATION_JOURNAL=0` turns it off. Keyframe values are not stored: each curve is
reduced to a hash and a few summary numbers, which localize a change without reconstructing it, at an order of
magnitude less than storing the values for an author who regenerates the animation on every call. Replaying the
journal's captured source recovers them; [scene_digest.py](blender/scene_digest.py) has the measurements.

## Tests and Smoke Artifacts

```bash
python -m pytest -q tests/test_animation.py
python -m pytest -q tests/test_animation_blender.py --run-expensive
python -m pytest -q tests/test_animation_daemon.py --run-expensive
python -m dycraft.tests.animation_smoke --out /tmp/animation-smoke
```

The smoke command uses the real session registration, isolated CAD and Blender workers, source compilation,
verification, the native camera, and the H.264 encoder. It creates a two-frame synthetic slide at 1920x1080 by default
and leaves every artifact on disk; use a new output directory each time. `--width 64 --height 64` selects a small
target. Its storyboard review is scripted for this fixture.

The tests cover planning and coverage, required versus advisory checks, RGB encoding, atomic failure and staleness,
source synchronization, preservation of authored appearance and actions, explicit refresh, private incompatible
candidates, collision exceptions, both movie strategies, final-film freshness, relocated reopening, refinement, and
contributor plumbing, with no model calls.
