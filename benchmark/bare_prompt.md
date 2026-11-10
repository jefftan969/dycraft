# Animation design and delivery guidance

Create a complete, polished Blender animation realizing the task above. Author
the scene and animated performance, and deliver the editable project, complete
rendered film, and supporting evidence. Reuse any explicitly supplied assets as
directed, preserving the identity, geometry, and articulation the task requires.

The task and execution instructions specify any required assets, duration,
frame rate, resources, deadline, and output locations. Those explicit requirements
take precedence over the defaults below. Resolve unspecified details coherently
and proceed independently.

## Your decisions

You own the creative direction, visual style, designs, staging, choreography,
duration, pacing, frame rate, aspect ratio, cameras, editing, sound, rigging,
animation techniques, renderer, production workflow, and validation methods,
subject to the task and these delivery requirements.

Choose how to allocate your time and compute. No production stages, review
cycles, critics, planning documents, animation algorithms, or shot counts are
prescribed. Choose methods that produce a convincing, editable final result
within the resources actually available.

## Scope and completeness

Show the complete event, including the consequential actions and intermediate
states needed to make the requested outcome understandable and credible.
Essential interactions must occur in the animated scene and be visible in the
finished film. Descriptions, attractive stills, and an already completed result
do not establish that an omitted action occurred.

Develop the setting, subjects, and supporting assets to the extent required by
the task. The scene must feel complete and intentional at establishing,
working, and close inspection distances. Detail should serve the task and the
chosen visual language. Maintain a coherent scene beyond the principal view.

## Motion and causality

Motion must convey credible timing, weight, balance, forces, and material
behavior appropriate to the subject and the task. Articulated mechanisms must
move in ways their construction permits. Contacts, grasps, releases, support
changes, tool engagement, and transfers must be consistent with the action and
the load. Choose the means of achieving this, including authored motion,
simulation, or a combination.

Objects retain their identity and state through time. Changes in position,
attachment, shape, contents, or appearance must have a cause consistent with
the task. There must be no unintended sliding, floating, penetration,
teleportation, popping, duplication, disappearance, or reset. Contents and
materials that are transferred must have credible origins and destinations.
These requirements apply throughout transitions and between key poses, as well
as across edits. State changes and contact must remain consistent with what
the acting subject is actually doing.

## Visual finish and presentation

Achieve a professional level of modeling, materials, lighting, composition,
and animation in the visual style you choose. Important forms, actions, and
contacts must remain readable. The finished sequence must be temporally stable
and free of distracting geometry, shading, rendering, and encoding defects.

Unless the task specifies the timing, choose a duration and presentation that
communicate the complete event with readable timing and an appropriate sense
of effort. The film must contain the full animated performance. Camera
placement, cuts, darkness, motion blur, depth of field, or effects must not
conceal missing actions or unresolved defects. Other relevant viewpoints must
reveal a coherent underlying scene.

## Required delivery

All paths below are relative to your assigned task directory. Use these paths
unless the task explicitly supplies different delivery locations.

- `output/blend/final.blend`: the complete, meaningfully editable Blender scene
  and its actual finished animation. Preserve useful controls, mechanisms,
  object identities, and scene structure so the performance can be edited and
  retimed. Set the scene frame range and frame rate to the intended performance.
- `output/film/final.mp4`: one complete film rendered from that final scene,
  encoded as H.264 MP4 at a constant frame rate. Unless the task specifies a
  different resolution, deliver at least 2,073,600 pixels per frame (the pixel
  count of 1920 x 1080); choose the aspect ratio. Choose an encoding quality
  that preserves the required visual detail. The film must have complete frame
  coverage and no missing, black, corrupt, unintentionally duplicated, or
  obsolete frames. Still-image slideshows and viewport captures do not meet
  this requirement.
- `output/evidence/`: enough additional rendered visual evidence to assess
  essential interactions and the underlying scene where the film alone is
  insufficient. Choose its format, viewpoints, and amount. Identify which
  requirements the evidence establishes; no fixed still or alternate-film
  count is required. If the film sufficiently establishes a requirement,
  reference its relevant time interval in the notes instead of duplicating it.
- `output/NOTES.md`: delivered paths; Blender version; exact commands and any
  included scripts needed to reproduce the film from the saved scene; enough
  information to edit its principal controls and timing; substantive task
  assumptions; validation evidence; and any remaining limitations.
- Include the scripts, textures, dependencies, and persistent caches actually
  needed to edit, reopen, and reproduce the scene and film, using portable
  paths or packed resources. Include the source scripts used to author the
  result if authoring was scripted.

The saved scene must reproduce the delivered motion and appearance after a
fresh reopening in the documented environment. It must not depend on unsaved
state, transient authoring-session caches, playback history, or an undocumented
external process. Persist or bake generated motion and simulation results as
needed. Describe any script execution required for reproduction. Save the
project with `bpy.ops.wm.save_as_mainfile`. Renders and evidence must correspond
to the delivered final scene.

## Environment and boundaries

This is a standalone task. Keep authored files, caches, temporary data, and
outputs inside the assigned task directory, except for runtime locations
explicitly provided by the execution environment. Do not read, search, copy,
or modify other solutions, experiment arms, repository content, or unrelated
user files. Use only the supplied task inputs and permitted runtime resources.

Installed tools and their documentation may be used. External factual or API
documentation may be consulted if available tools and permissions permit it.
Do not download an existing scene, character rig, or animation to substitute
for the assignment. Explicitly supplied starting assets are permitted and
must be reused when the task requires them. Do not load or invoke skills,
prior-session memories, or instructions from other experiments.

Use the interpreter, Blender version, hardware, and tool access described by
the execution environment. Choose rendering methods and settings within the
resources actually available.

## Validation, completion, and available time

Completion means satisfying the task and the quality and delivery requirements.
Successful execution, saving, rendering, encoding, or passing a structural
check alone does not establish completion. Establish the quality of the actual
saved result using methods you choose and report the evidence honestly in
`output/NOTES.md`. Inspect the resulting animation and presentation, including
transitions and essential interactions; isolated successful poses do not
establish that motion between them is correct. Distinguish what you inspected
or tested from what you inferred or left unverified. Do not claim that
uninspected frames, unsatisfied requirements, or unrun checks passed.

Use the wall-clock budget and absolute deadline if supplied with the execution.
Run `date` to read the current time and pace yourself against that deadline.
The budget includes all authoring, rendering, encoding, and validation. Work
autonomously within it and allocate your own time and compute; finish earlier
when the requirements are met. At the deadline, preserve and report the actual
result with unmet requirements identified. A time limit does not convert an
unfinished artifact into a successful delivery. This guidance does not itself
set or enforce a wall-clock limit.

Before the session ends, write `context_summary.md` in the task directory.
Include the final status, delivered scene, film, evidence and documentation
paths, checks actually run and their results, and any unmet requirements.
For unfinished or incorrect work, describe attempted repairs and a useful next
step. Reference files and relevant lines instead of reproducing the task,
complete source files, or tool logs. Write this summary directly; no skill is
required.
