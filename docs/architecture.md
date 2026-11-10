# Architecture

This guide follows the method section of the paper, step by step, and points to the code for each step.

| Step | Harness operations | Code |
|---|---|---|
| Prompt rewriting | (before the agent starts) | `agent/scene_rewrite.py`, `agent/motion_rewrite.py`, run by `agent/oneoff.py`; prompts in `agent/prompts/` |
| Planning | `plan` | `animation/planning.py` |
| Scene construction in CAD | `build`, `probe`, `snapshot` | `sdk/`, `animation/build.py`, `animation/blender/import_scene.py` |
| 3D storyboard | `blender(stage='storyboard')`, `storyboard`, `accept_storyboard` | `animation/storyboard.py` |
| Motion authoring | `blender`, `ik`, `render`, `snapshot`, `motion_image`, `inspect`, `dispatch_*` | `animation/blender/`, `animation/ik/`, `animation/dispatch.py` |
| Verifiers | `compile`, `check_motion`, and every `build` and `storyboard` | `verify/`, `animation/blender/verify*.py` |
| Independent review | `review` | `animation/review_render.py`, `verify/review_images.py` |
| Film | `final_render`, `finish` | `animation/delivery.py`, `animation/blender/movie.py` |

## How a Run Works

DyCraft runs an existing coding agent, Claude Code or Codex, with a system prompt and a set of harness operations for
building, checking, and rendering scenes and motion. The agent keeps all of its own tools: it edits files, runs shell
commands and Python, and calls the harness when it needs one of DyCraft's operations.

```
python -m dycraft.cli --backend claude --layer <layer> --prompt ...
  │
  ├─ prepares runs/<run_id>/src/: meta.json, the system prompt, harness.py, skills, references/
  ├─ starts the harness daemon for this run (one per run, on localhost)
  └─ runs the agent CLI (claude / codex) in runs/<run_id>/src/, sandboxed
        │
        │  the agent edits object.py / scene.py / Blender code, then calls
        │  python harness.py <op> ...      ─────►  daemon: build, check, render, ...
        │                                  ◄─────  JSON result: errors, verdicts, image paths
        ▼
     finish()  ─►  the daemon commits only when no blocking check fails
```

The launcher (`cli/`) parses the options, runs the prompt rewriters, prepares the run directory and its environment,
and runs the agent CLI as a child process. The harness daemon (`agent/`) holds the run's state and serves the harness
operations. `harness.py` is a small client, using only the standard library, that sends each operation to the daemon
and prints the JSON reply. The state is kept by the daemon and on disk, so a run can be inspected from its directory at
any time.

Before anything is staged, the launcher runs the setup checks the layer and backend need, the same checks and messages
as `python -m dycraft.install check`. If one fails, it prints the check's fix and exits with status 2 without starting
anything. `DYCRAFT_SKIP_PREFLIGHT=1` launches without them.

The agent works in the run's `src/`, which holds everything the agent and the harness write. Only the harness writes the
top level of the run directory: the film, the Blender project, the agent's notes, and a `README.md` on the run's current
state, which it rewrites at launch, at `finish`, when the launcher exits, and when `--resume` reopens a delivered run. Paths in this guide are inside `src/` unless they say otherwise.

Every operation in an animation run is also recorded in `native/journal/`: the code it ran, the source files it read,
and a fingerprint of the project it saved. `Journal.timeline()` in `animation/blender/journal.py` lists what each
operation changed, which helps to find the call that introduced a problem.

## Layers

`--layer` selects the agent type. The animation agent is DyCraft as described in the paper. The scene and object
agents use the same CAD tools and verifiers on their own, and the animation agent can hand work to them.

| Layer | The agent writes | Main harness operations |
|---|---|---|
| `animation` | A scene and its animation, in a Blender project kept for the whole run | `plan`, `build`, `storyboard`, `accept_storyboard`, `blender`, `render`, `check_motion`, `ik`, `review`, `final_render`, `finish`; `dispatch_object`, `dispatch_scene`, and `dispatch_motion` for helper agents |
| `scene` | One `scene.py` that builds its objects and places them | `compile`, `probe`, `snapshot`, `review`, `final_render`, `finish`; `submit_object` for helper agents; `load_asset` in `scene.py` |
| `object` | `object.py`, build123d solids registered as parts and connected by joints | `compile`, `probe`, `snapshot`, `review`, `final_render`, `finish` |
| `plain` | Text | None |

Each agent type's instructions are its system prompt in `agent/prompts/` (`animation.md`, `scene.md`, `object.md`),
with the SDK reference for its source files (`animation_reference.md` for the animation agent). The skills in
`.claude/skills/` and `.agents/skills/` are copied into each run, and the agent reads them when it needs them:
`build123d_authoring` for CAD pitfalls, `humanoid_motion` for character motion, and `mocap_reference` for motion
capture, copied only when a motion-capture library is installed.

## Prompt Rewriting

User prompts are often short and leave most of the scene and the motion unspecified, and agents rarely add detail
beyond what the prompt states. Before the animation agent starts, the launcher therefore runs two helper agents with
fresh context, on the same backend:

1. The scene rewriter (`agent/prompts/scene_rewriter.md`) reads the brief and writes `scene.txt`, a description of the
   place where the action happens. It first names the default scene an agent would build without thinking further,
   then chooses one organizing idea, such as how the place has been used over time, and describes the place from it.
2. The motion rewriter (`agent/prompts/motion_rewriter.md`) reads the brief and `scene.txt` and writes `motion.txt`, a
   description of how the action is physically carried out. For a person, humanoid robot, or animal, it follows the
   `humanoid_motion` skill.

Both files are staged in the run directory beside the brief and the system prompt. When `--max-hours H` is given, the
motion rewriter sizes the performance to fit in H hours. `--scene-brief FILE` and `--motion-brief FILE` use your own
files, `--no-scene-brief` and `--no-motion-brief` skip a rewriter, and `--on-rewrite-failure` says what to do when a
rewriter fails (`ask`, `abort`, or `continue`). While a rewriter runs, the launcher shows what it is doing, as one line
updated in place on a terminal or a line a minute in a log.

## Planning

The agent's first stage is a plan of the whole film. It works out the actions in order, each with its visible
result, the objects and spatial relationships it needs, and the contacts and state changes it causes, in the plan or in
its own notes. It then registers the plan with `h.plan({...})`, stored by `animation/planning.py`. A registered plan
has:

- an intent for the film;
- the shots, each with an id, its intent, and the name of the Blender camera that films it, with optional, provisional
  frames;
- the states, one ordered list of the film's key moments, each with an id and a purpose. Their frames are not part of
  the plan; they are set in the Blender project, relative to a clip;
- the criteria the finished film must meet, each of which needs a verdict with evidence before `finish`.

The harness stores the plan and checks its structure. It does not generate or judge the plan's content. The agent
revises the plan whenever later work changes it, and a revision makes the whole-film checks and the final render run
again.

## Scene Construction in CAD

### Articulated Objects in build123d

Agents build objects as code. Each object is a [build123d](https://github.com/gumyr/build123d) program, `object.py`,
whose shapes are exact boundary-representation (B-rep) solids from the OpenCascade kernel. Extrusions, revolutions,
lofts, fillets, and boolean operations give precise solid geometry, without hand-written vertex lists.

The SDK (`sdk/object_builder.py`) adds what animation needs on top of build123d:

- parts: `add_part` registers a solid as a rigid part with a material;
- joints: `attach_revolute`, `attach_prismatic`, and `attach_fixed` connect a child part to its parent part, with an
  axis, a pivot, and limits;
- named coordinate frames: `add_anchor` names a point and direction on a part, so other code can refer to a place such as
  the hinge side of a door by name;
- assertions: `assert_distance`, `assert_contact`, `assert_parallel`, `assert_flush`, and others state relationships
  that the verifiers then test;
- queries: `probe` and the object handle report sizes, frames, and the joint table, including each joint's limits.

The program runs in a separate process under the pinned build123d version, in a restricted Python environment.
Each part keeps its exact B-rep and a triangle mesh for rendering and collision.

### ObjectIR and SceneIR

An articulated object is a tree: parts are its nodes and joints are its edges, from a root part down to every moving
part. Meshes alone cannot hold this structure, so the SDK writes it as an intermediate representation (`sdk/ir.py`):

- `ObjectIR` holds one object's parts, its joint tree (each joint's type, parent and child, axis, pivot, limits, and
  default value), its anchors, and its assertions. Meshes and B-reps are stored as files referenced by content hash.
- `SceneIR` holds the room and the placed instances. Each instance names an object definition by its hash, gives its
  placement, and can give its own initial joint values, so two cabinets from one definition can have their doors at
  different angles.

An object's identity is the hash of its content, so a scene records which version of each object it placed,
and a changed object becomes a new definition. The verifiers, renderers, viewer, and Blender import all read the same
documents.

### Scenes

A scene agent writes `scene.py`, which builds or loads its objects and places them in a room with
`sdk/scene_builder.py`. The animation agent writes the same kind of source under `design/`, and `build` compiles it.
People, animals, and robots usually come from the asset library rather than being built in CAD. `load_asset(name)` stages
the named bundle into the run and returns its definition, which `scene.place` places like any other object. See the
[usage guide](usage.md#asset-library) for more details. Native Blender tools remain available for what CAD does not
suit, such as deformable characters, cloth, and liquids.

### From CAD to Blender

`build` compiles the design, then imports the SceneIR into the run's Blender project (`animation/blender/import_scene.py`):

- each part's mesh becomes a Blender object at its placed pose, with its material;
- each revolute or prismatic joint becomes an Empty, `joint/<instance>__<joint>`, under a frame turned so that the
  joint's axis is the Empty's local Z. The joint's value is a single channel of that Empty, the Z rotation for a
  revolute joint or the Z location for a prismatic joint;
- an imported character keeps its native armature and skinned mesh;
- the scene's joint tree is also written as URDF, which the inverse kinematics solver reads.

Agents key a joint with `key_joint(name, value, frame)` (`animation/blender/dycraft_joints.py`). Anything else written
to a joint Empty moves the joint off its axis, and the `JointLimits` verifier fails it. `build` replaces the imported
objects when the design changes and keeps what the agent authored in Blender.

## 3D Storyboard

Agents asked to animate tend to script the whole motion in one pass, so layout and staging problems appear late,
mixed with motion errors. DyCraft requires a storyboard first. Each planned state is staged as a static 3D scene,
checked and looked at before any continuous motion is written.

1. The agent poses each state with `blender(code, stage='storyboard')`, using keyed poses and, for robots, `ik`.
2. `storyboard(states=None)` runs the verifiers on each state and renders it through the film's cameras, with
   the same appearance settings as the final film at lower resolution (Cycles, 640x360). The images and results are
   recorded per state (`animation/storyboard.py`).
3. The agent opens the images, and `accept_storyboard(reviews)` records what it saw in each state.

Animation can be authored only after every state of the film passes the required verifiers and has been reviewed.
A motion helper agent is held only to the states of its own clip. Storyboard states are cheap to render and check, so
the agent can settle layout, reach, clearance, and camera framing before any motion exists. Reviews are kept per state.
A redrawn state whose image is unchanged keeps its review, and a state whose shown objects did not change is not
redrawn.

## Motion Authoring

The agent writes motion as Python run inside Blender with `blender(code)`, against the project that `build` made
(`animation/blender/worker.py`). The worker runs in a sandbox with no network, and the project is saved after each
call. The storyboard states anchor the motion, and the finished animation must pass through them.

DyCraft guides the agent toward methods that keep motion plausible:

- Characters: the `humanoid_motion` skill covers footstep planning before the body moves, balance, and joints that
  bend the right way. The `mocap_reference` skill has the agent start motion from captured clips in the
  libraries installed under `assets/mocap/`, each with its own notes and index, bring a clip onto the character with
  that library's retarget helper or its own retarget, and edit it into the performance.
- Robots: `ik` asks the cuRobo solver (`animation/ik/`) for joint values that reach a pose, which the agent keys.
  Without an NVIDIA GPU, the launcher says IK is unavailable, the prompt leaves `ik` out, and the agent solves robot
  joints with its own solvers, following the `humanoid_motion` skill.
- Physical consequences: falling, collisions, cloth, and liquids are simulated with Blender's own solvers and baked,
  with the authored bodies as colliders or inflows. The system prompt asks the agent to author the intention and
  simulate the consequence.
- Mechanisms: `key_joint` keys each joint along its own axis.

### Visual Inspection

The agent can look at the scene at any time:

- `render` draws a frame set, an interval, or the full timeline through the film's cameras or another camera, and
  returns the images and a short clip. It runs no checks.
- `snapshot` renders selected frames to look at closely.
- `motion_image` combines frames over time into one image, for judging motion at a glance.
- `inspect` returns positions, bounds, and camera data for up to four frames, without images.

Drafts render with Workbench at 1280x720, which is fast and enough for questions about geometry and framing.

### Delegation

Once the plan is set, parts of the work can proceed in parallel. `dispatch_object`, `dispatch_scene`, and
`dispatch_motion` start helper agents (`animation/dispatch.py`) in `contributions/<name>/`. A helper is an object
agent, a scene agent, or a motion agent that works on one clip inside the main project. The main agent stays responsible for bringing their work
into the film.

## Verifiers

Verifiers are numerical tests of the geometry and the motion. Each returns `pass`, `fail`, or `uncertain`, with a report
that names the parts, the frames and the measurements, such as the depth, and direction of an overlap. Fast verifiers
run first, and slower ones run only after those pass. Collision tests use [coal](https://github.com/coal-library/coal)
on the parts' meshes for scenes and animations, and exact B-rep booleans for objects, whose repair advice gives the
overlap volume. Before testing overlap, each closed mesh is shrunk by its own tessellation error, so parts that only
touch pass.

### Animation Verifiers

`animation/blender/verify.py` evaluates the Blender project frame by frame and tests every output frame. These four
are required, and a failure blocks `finish`:

- `AnimationWellFormed`: every evaluated mesh is non-empty and finite on every frame.
- `JointLimits`: every mechanism joint stays within its limits and moves only along its own axis.
- `NoPenetration`: no two parts intersect on any frame, and nothing leaves the room, apart from declared allowances.
  A skinned character is one part, so its limbs are not tested against its own body.
- `EventsSatisfied`: every event the agent declared holds over its time window.

These four are advisory. They report at most `uncertain` and point the agent to frames to inspect:

- `AnatomicalJointLimits`: the bones of an imported character stay within anatomical ranges at the elbows, knees,
  wrists, ankles, head, and spine.
- `MotionContinuity`: no part or bone jumps between consecutive frames, such as a limb that flips for one frame.
- `SupportMargin`: a moving body's center of mass stays over the area it stands on.
- `ContactSlip`: a part that bears weight, such as a foot, does not slide while it is planted.

The source design is also tested for its assertions, support, and room containment, and with the object verifiers.

### Declarations

The agent declares what the verifiers should test over time in the scene's `dycraft_declarations`:

- events, each a predicate over a window of frames: `touch`, `separated`, `joint_at`, `above`, `supported` (the
  center of mass within the footprint it bears on) and `planted` (a bearing contact that does not creep);
- collision allowances for intended overlaps, such as an object held in a hand or a worn item, limited to the named
  parts, an optional time window, and a maximum depth that the verifier enforces;
- liquids, which are exempt from penetration and room tests.

Text in the plan does not create checks. Only declared events are tested.

### Finishing

`finish` commits the run only when the required verifiers pass. For the animation agent, it also needs a verdict, with
evidence, for each criterion in the plan. When the agent cannot meet every criterion, `finish(exceptions=True)` delivers
the film as `delivered_with_exceptions` and records what is unmet. Object and scene agents can record a planned
departure from the brief with `record_deviation`, which writes it to `meta.json` but does not let a failing verifier
pass.

### Scene Verifiers

`verify/scene_verify.py` runs the object verifiers on every placed object, except `SupportConnected`, which the scene
version replaces, and `NoSelfPenetration` for imported assets. It runs these on the placed instances:

- `NoCrossInstancePenetration`: no two placed objects intersect, at their placed joint values and at sampled ones.
- `RoomContainment`: every object stays inside the room.
- `SupportConnected`: every object rests on the floor, a wall, or the ceiling, directly or through other objects. An
  attachment declared in code does not count without contact.
- `SceneRelationalAssertions`: the agent's assertions between objects hold.
- `AnatomicalJointLimits`: placed robots or characters that declare anatomical ranges stay within them.

### Object Verifiers

`verify/object_verify.py` runs these on every build of an object:

- `NoSelfPenetration`: no two parts of the object intersect, at rest or at positions sampled across each joint's range.
- `MinPartDimension`: no part is thinner than a minimum set by the object's size.
- `JointRangeNonTrivial`: every revolute joint can turn at least 1 degree, and every prismatic joint can slide at least
  1 mm.
- `JointOriginNearGeometry`: each joint's pivot lies on or near the geometry it connects.
- `JointOriginNearBothBodies` (advisory): the pivot is near both the parent part and the child part.
- `SupportConnected`: every part is connected to the object's base through a chain of contacts.
- `ObjectRelationalAssertions`: the agent's assertions hold at the rest pose.

## Independent Review

No automatic visual judge runs. The agent judges its own renders, and `review(focus=None)` asks for a second opinion
from a new agent on the same backend, with no access to the main agent's reasoning. The harness renders fresh images
at the planned states (Cycles, 640x360), and the reviewer reads them with the main agent's claims. The reviewer can
request other frames or views, from 1 to 32 frames per request. `focus` names the question to look at. The reviewer
does not watch the film, and its opinion is advisory.

## Film

`final_render` renders the film through the project's cameras and cuts, by default with Cycles at 1920x1080 and 8
denoised samples, and saves the render settings in the project, which stays editable. The agent then watches the
encoded film, repairs and renders again if needed, and calls `finish`. `finish` packages the same project revision
with its film in `bundle/`: `film/clip.mp4`, `animation.blend`, and `design_state/` with the design's source files and
the frozen scene. It then puts the same two files at the top of the run directory as `film.mp4` and `animation.blend`,
as hard links or, where the file system has none, copies. A project that reads cache files beside it is saved there as
a copy whose paths point into `bundle/`, so it opens the same. When a resumed run delivers again, both are replaced, and
earlier deliveries stay in `prior_deliveries/<n>/`. Films have no sound.

## Sandbox Settings

DyCraft runs agents unattended, so each CLI runs without approval prompts, inside its sandbox. These settings apply to
both:

- Writes are limited to `src/`. It is the agent's working directory, and its shell commands can write only there,
  plus temporary directories on Codex. A command the sandbox refuses fails.
- Reads are not limited. The agent can read any file your user can read, because it needs the installed packages,
  documentation and tools.
- The agent sees the launcher's environment, including any exported API keys.
- `object.py` and `scene.py` run in a restricted Python interpreter with no imports, no file access, and a fixed set of
  names. This keeps agent programs to what the harness can track. It is not a security boundary, since the agent also
  has a shell.
- Blender code runs in a `bubblewrap` container with no network, a read-only view of the system, no capabilities, and
  only the project directory writable.
  Where `bubblewrap` cannot create user namespaces without sudo, Landlock and a filter on sockets confine the worker
  to the same reads and writes, with no network, on this machine's real paths.
- The launcher stops the agent CLI and every helper agent after `--max-hours` (default 12), records the stop in
  `meta.json` under `time_limit`, and exits with code 124. A resumed run gets its allowance less the agent time it has
  used, or `--max-hours H` more hours.
- An agent CLI that writes no session record within 30 seconds of starting is stopped, unless the launcher runs in a
  terminal, where you can see and answer what it waits on.

### Claude Code

DyCraft starts `claude` with:

- Claude Code's sandbox on for Bash (`bubblewrap` and `socat` on Linux) and required. If the sandbox cannot start, the
  agent does not run, unless `--allow-unsandboxed` is given, which turns the sandbox off for that run. Bash commands
  run inside it without prompts, and the model's `dangerouslyDisableSandbox` option
  is off.
- Bash writes only in `src/`. The `Edit` and `Write` tools work only under `src/`, except its `.claude/`
  directories. `Read` works everywhere.
- Network from the agent's shell to `localhost` only, where the harness daemon listens. The CLI's own connection to
  the model API is not affected.
- `--permission-mode dontAsk`, so anything not allowed is refused at once instead of waiting for a prompt.
- `--strict-mcp-config`, so your own MCP servers are not loaded, and hooks off.

On machines that restrict nested user namespaces, such as Ubuntu 24.04, Claude Code's filter for Unix sockets cannot
start. There DyCraft allows Unix sockets in the sandbox, and makes the sockets it knows about unreadable to the agent:
container runtimes, the system and session buses, tmux and screen, the X11 display, ssh agents, and the Claude Code
daemon. Other sockets stay reachable. `DYCRAFT_CLAUDE_UNIX_SOCKETS=allow` or `=block` overrides the detection.

### Codex

DyCraft starts `codex` with:

- Account checks before anything else: the account's folder exists, its `config.toml` parses, and it holds a login.
  Each failure is refused with its own fix.
- `-c check_for_update_on_startup=false` and `-c notice.hide_rate_limit_model_nudge=true`, so no notice screen waits
  for an answer.
- `--sandbox workspace-write --ask-for-approval never`, where the account allows it. A command the sandbox refuses
  fails and the model is told. Some organizations' ChatGPT workspaces do not allow `never`, and Codex then runs with
  `on-request`, where Codex's automatic reviewer can approve a command outside the sandbox. DyCraft warns at launch and
  records the policy in `meta.json` under `codex_sandbox.approval_effective`.
- A sandbox test before each launch. DyCraft runs one command under `codex sandbox`, which takes a fraction of a second
  and makes no model call. If the sandbox does not start or does not limit writes, the launch is refused with the cause
  and the fixes. `--allow-unsandboxed` starts Codex anyway. Its commands then go to Codex's automatic reviewer and,
  once approved, run with no limit on writes.
- `src/` as its working root (`-C`), so `workspace-write` limits writes to it, plus `/tmp` and `$TMPDIR`.
- Network access in the sandbox, so the agent's shell can reach the harness daemon. Unlike with Claude Code, this is
  not limited to `localhost`.
- Your own `~/.codex`. Settings in `~/.codex/config.toml` apply to DyCraft's agents too, including settings that widen
  the sandbox, such as extra `writable_roots`. Use a separate account (`codex1` to `codex9`) if you keep such settings.
- A temporary trust entry. Codex run without a terminal in a directory with no trust decision waits forever on its "Do
  you trust this directory?" screen. While Codex runs, DyCraft records `src/` as `untrusted` in
  `~/.codex/config.toml` and removes the entry afterwards. This skips the screen and keeps Codex from loading
  configuration, hooks, or rules from `src/`, where an agent could write them.

## Version Constraints

DyCraft pins a version only where the code enforces it, where a version does not run on the hardware, or where no
package exists. Everything else floats, and `python -m dycraft.install check` reports where an installed version
differs from the validated one. Before you loosen a constraint, read its row: it says what breaks and where the
constraint is enforced.

Main environment (`environment-main.yml` and `requirements-main.txt`, generated from the `REQUIREMENTS` table in
`environment_test.py`: edit the table and run `python environment_test.py --write-env`):

| Constraint | Kind | What breaks otherwise |
|---|---|---|
| Python `>=3.10,<3.15` | Wheels | Validated on 3.10, 3.13, and 3.14. Python 3.15 has no open3d, coal, or OCP wheels yet. Lift the bound once they do. |
| `build123d==0.11.1` | API; enforced by the CAD process, which refuses any other version | 0.12.0 removes `Face.make_plane`, which the build123d_authoring skill's namespace reference offers the agent. It also changes the `material` default from `''` to `None`, and renders a reference part to a different image. 0.9.1 lacks `Face.axis_of_rotation`. To move the pin, change `REQUIRED_BUILD123D`, regenerate the namespace reference (`agent/cad/_namespace_doc.py`), and recheck the skill's pitfalls. |
| `coal>=3.0.1` | Enforced by `animation/blender/coal_worker.py` | Nothing older is on PyPI. 3.0.1 to 3.0.3 are validated. |
| NumPy, SciPy, Pillow, Open3D, OpenCV, the viewer's packages | None | NumPy 2 comes in through build123d (`numpy>=2,<3`) and coal's cmeel-boost, not from DyCraft. |

Blender environment (`environment-blender.yml`, `python -m dycraft.install blender`):

| Constraint | Kind | What breaks otherwise |
|---|---|---|
| `bpy==4.5.*` (Blender 4.5 LTS) | API | Blender 5.0 moved pose selection from `Bone.select` to `PoseBone.select`. Importing a rigged glTF then fails with `AttributeError: 'Bone' object has no attribute 'select'`. 4.5.13 and 4.5.15 are validated. |
| Python `3.11` | Wheels | bpy 4.5 publishes cp311 wheels only. On other Pythons pip reports "no matching distribution". |
| NumPy `<2` (declared by bpy) | bpy's own | This is why Blender has its own environment. coal, in the main environment, needs NumPy 2. |
| Official Blender 4.5.13 binary | Integrity | `install blender` checks the published SHA-256 of that file. The binary is needed for fluids only, because Mantaflow does not run in the bpy module. Another 4.5 LTS build can be recorded in `.blender_binary`; only 4.5.13 is validated. |

IK environment (`environment-ik.yml`, `python -m dycraft.install ik`; optional, NVIDIA GPU only):

| Constraint | Kind | What breaks otherwise |
|---|---|---|
| PyTorch: no version; the index is chosen by GPU | Hardware | No single PyTorch build has kernels from Pascal to Blackwell. The CUDA 12.6 builds carry sm_50 to sm_90, and the CUDA 12.8 builds sm_75 to sm_120. PyPI's default build is CUDA 13 and starts at sm_75. A build without kernels for the GPU still imports and reports `torch.cuda.is_available() == True`, then fails its first kernel with "no kernel image is available for execution on the device". `install ik` reads the compute capability (`nvidia-smi --query-gpu=compute_cap`) and picks the index: cu126 below sm_100, cu128 for Blackwell. `--torch-index` overrides it. |
| `warp-lang<1.18` | Hardware, driver | cuRobo imports Warp. From 1.18, Warp's PyPI wheels are CUDA 13 builds, which need sm_75 and an R580 driver. On a GTX 1070, cuRobo fails with "CUDA 13.4 requires sm_75 or higher". On a Turing or newer GPU with an R580 or newer driver, any Warp works. Before Volta (sm_70), the IK process runs `animation/ik/_ik_setup.py` before the agent's code, which turns off Warp's cuBLASDx path (`warp.config.enable_mathdx_gemm/solver = False`). cuBLASDx has no kernels before Volta, and cuRobo's LM step fails with "libmathdx cuBLASDx error" without the switch. |
| `cuda-core[cu12]` | Consistency | cuRobo's kernel backend, matching the CUDA 12 PyTorch. Without a backend, the first forward kinematics call fails with "No curobo kernel backend available!". |
| cuRobo at a pinned commit (`CUROBO_COMMIT` in `install.py`) | API, reproducibility | The v0.8.0 tag lacks `curobo.types.ToolPoseCriteria`, which `animation/ik/CUROBO.md` teaches. A newer upstream commit also solves IK on Blackwell, so newer commits probably work. To bump `CUROBO_COMMIT`, rerun `tests/test_ik_adapter.py --run-expensive`, which checks CUROBO.md's claims against the build. |
| Python `>=3.10,<3.14` | Validated range | cuRobo needs 3.10 or later. 3.10, 3.12, and 3.13 are validated. |

`install ik` records `.ik_python` only after cuRobo solves its bundled Franka arm back to a forward-kinematics pose on
the GPU. `install check` runs a kernel, and `install check --deep` runs that solve again.

## Running the Tests

```bash
pip install -e ".[test]"                              # once: adds pytest
python -m pytest tests/ -q -p no:randomly             # from the repo root
python -m pytest tests/ -q -p no:randomly --run-expensive   # also the slow tests: Blender, CAD builds
```

None of the expensive tests starts an agent CLI or spends account usage. Pass `tests` explicitly, since it points
pytest at `tests/pytest.ini`. From the repo root without it, pytest collects the wrong files. Tests that need the CAD
interpreter or Blender skip themselves when it is not configured.

## Module READMEs

| Module | README |
|---|---|
| The launcher, its transport, and adding an agent type or backend | [cli/README.md](../cli/README.md) |
| The animation agent: stages, operations, render defaults | [animation/README.md](../animation/README.md) |
| The `ik` operation and cuRobo | [animation/ik/README.md](../animation/ik/README.md) |
| The benchmark and the scorer | [benchmark/README.md](../benchmark/README.md), [benchmark/score/README.md](../benchmark/score/README.md) |
| The viewer | [viewer/README.md](../viewer/README.md) |
