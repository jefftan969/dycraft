# Usage Guide

```bash
python -m dycraft.cli --backend <claude|codex> --layer <animation|scene|object|plain> --prompt <text or file> [options]
```

The run directory, `runs/<run_id>/`, is in the clone or in `$DYCRAFT_HOME` (see the
[install guide](install.md#where-dycraft-keeps-its-files)), unless you pass `--runs-dir`. After one hour of a run
(`--advisory-hours`), the agent is told how long it has been running. The README covers the film command, the common
options, and resuming a run.

## Films: `--layer animation`

Before the agent starts, two helper agents expand the brief into a description of the scene (`scene.txt`) and a
description of how the action is carried out (`motion.txt`). `--scene-brief FILE` and `--motion-brief FILE` use your
own files instead. `--no-scene-brief` and `--no-motion-brief` skip them. With `--max-hours H`, the motion
description is sized to fit in H hours.

The agent builds the scene and animates it in one Blender project, in four stages:

1. a plan of the action and the shots;
2. a 3D storyboard of each shot's key moments, which it renders and inspects before it animates;
3. the motion between those moments;
4. the final film.

The agent takes people, animals, and robots from the asset library, and can hand objects, parts of the scene, or motion
to helper agents.

### Writing a Brief

Say who acts and why, what the scene contains at the start, the actions in order, and how the scene ends. The camera,
the duration and the style can be left to the agent. Films have no sound. The 24 benchmark briefs are examples. See the
[benchmark guide](benchmark.md) for more details.

## More Options

| Option | Description |
|---|---|
| `--advisory-hours H` | When the agent starts being told how long it has run (default 1; 0 turns it off) |
| `--render-backend open3d\|blender` | The renderer for object and scene images. `open3d` (default) is fast; `blender` needs the Blender environment |
| `--exp-name` | Text added to the default run directory name |
| `--allow-unsandboxed` | Start Claude Code or Codex without its sandbox where the sandbox cannot start. The agent's commands can then read and write anywhere you can (see the [install guide](install.md#sandbox-fails-to-start)) |

## Models and Accounts

`--backend claude` uses the account the `claude` command would use in your shell: `$CLAUDE_CONFIG_DIR` if set,
otherwise `~/.claude`. `--backend codex` uses `~/.codex`. To run on one of several accounts, `claude1` to `claude9` and
`codex1` to `codex9` use `~/.claudeN` or `~/.codexN`. Log in to each one once, for example with
`CLAUDE_CONFIG_DIR=~/.claude2 claude auth login`. `--resume runs/<run_id> --backend codex2` continues a run on
another account of the same CLI, for example after a usage limit. DyCraft copies the run's session into that account
first.

The agent inherits the launcher's environment. If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set there, the CLI may
bill that key instead of your subscription, and the agent can read it.

The paper's runs used Claude Code 2.1.316–2.1.319 and Codex CLI 0.153.4. Newer releases are expected to work.

## What a Run Directory Holds

A run that delivered a film holds:

| Path | Contents |
|---|---|
| `film.mp4` | The film |
| `animation.blend` | The Blender 4.5 project that renders it |
| `README.md` | What happened: the brief, the outcome, each criterion's verdict, what is unmet, and how to view, export, or resume the run |
| `notes/` | The agent's notes on the film, if it wrote any, such as how to edit the project |
| `src/` | The agent's working directory, with everything else |

A run that has not delivered, because it is still running, stopped, or failed, holds only `README.md` and `src/`. An
object or scene run holds `README.md`, its final renders in `views/`, `object.urdf` or `scene.urdf` with its
`meshes/`, and `src/`. A plain run holds `README.md`, `answer.md`, and `src/`.

`src/` is the only place the agent can write. It holds:

| Path | Contents |
|---|---|
| `meta.json` | How the run was launched and how it ended |
| `object.py`, `scene.py`, `design/` | The agent's code |
| `ir.json`, `assets/` | The finished object or scene, with its meshes and CAD solids |
| `compile_reports/` | Every build, with each check's result |
| `renders/`, `final_render/` | The images the agent looked at, and the final renders |
| `native/` | The live Blender project and the record of every operation on it (animation runs) |
| `bundle/` | The delivered film, its project, plan, and design sources (animation runs) |
| `prior_deliveries/` | Earlier deliveries, when a resumed run delivered again |
| `sub_runs/` | The prompt rewriters' and reviewers' runs |
| `contributions/` | The helper agents' work (animation runs) |
| `trace.jsonl` | Every tool call and its result |
| `agent_transcript.jsonl` | The agent's transcript |

The interactive viewer shows all of this for any run.

## Asset Library

Agents build furniture, tools, and machines as build123d programs. Rigged characters (people and animals)
and robots usually come from an asset library of imported models, which the animation agent can cast. A brief with the
Unitree G1 robot, a barista, a dog, or a film crew needs the library.

The library is downloaded separately:

```bash
python -m dycraft.install library
```

It downloads the library from the [dycraft-assets](https://huggingface.co/datasets/jefftan969/dycraft-assets) dataset
on Hugging Face into `assets/` in the clone or in `$DYCRAFT_HOME`, and links each bundle into
`assets/imported/`. Run it again to update the library in place. Only new or changed files are downloaded.
`--category person`, `animal`, or `robot` and `--names <bundle>,<bundle>` fetch part of it, `--revision` pins a version,
and `--prune` unlinks characters that a newer version no longer lists. A bundle you imported yourself under the same name
is never replaced.

### What Is in It

The library holds people, animals, and robots. The robots include the Unitree G1 with Dex1 grippers and the Franka
Panda arm. The people include the cast of the paper's films, under the names the runs used, and the characters that
briefs ask for by name, such as a barista, a customer, an older man, and a trainer. Every character was reviewed on films
of its motion and on a test of its rig before it entered the library. `catalog.json` beside the bundles lists each one
with its description, caveats, license, and credit.

### How Agents Use It

Each run links the library at `src/references/assets/`. Its `INDEX.md` lists each bundle's size, skeleton, bone count, and
animation clips, and for a robot its joints. The agent can read the library but not change it. In scene and
animation runs, the agent picks assets. `load_asset("<bundle>")` in its `scene.py` stages a bundle and returns its
definition, and `scene.place` places it. Assets are not passed on the command line.

To give a run a model that is not in the library, such as a finished object run or a scanned object:

```bash
python -m dycraft.cli --backend codex --layer animation \
    --preload-asset robot=path/to/bundle --prompt brief.md      # scene.py: load_asset("robot")
```

The bundle can be a finished object or scene run (`runs/<run_id>`), or any directory with an `ir.json` and the
`assets/` it uses. A preloaded asset takes the place of a library
bundle with the same name.

### Licenses

Each bundle records its license and credit. Most characters are CC BY. Some are CC BY-SA or CC0, and some are CC BY-NC
or CC BY-NC-SA, which allow non-commercial use only. Characters whose license forbids derivative works are not
included, because an imported bundle is a modified version of the original model.

The library's `catalog.json` records each character's license, author, and source, and its `CREDITS.md` lists them by
license. The license texts are in the dataset's `LICENSES/` folder. The Unitree G1 description is under the BSD
3-Clause license in its own `LICENSE`, the Franka Panda description is under Apache-2.0, and the Microsoft Rocketbox
characters are under the MIT license.

### Where It Lives

DyCraft looks for the library at `assets/imported/` in the clone or in `$DYCRAFT_HOME`, or at
`$DYCRAFT_ASSET_LIBRARY`. Each entry is a bundle, a directory with
`ir.json`, `import.json` (how it was imported), and the meshes, textures, and rig it uses. Without a library, every agent
type still runs, but briefs that need a character or a robot will not find one.

`catalogs/asset_library.json` adds what the bundles do not record: what each asset is, a one-line description, search
words, and whether it is offered. DyCraft builds an index from the bundles and that file:

```bash
python -m dycraft.agent.assets.asset_library               # writes INDEX.md and index.json into the library
python -m dycraft.agent.assets.asset_library --thumbnails  # also renders a thumbnail per bundle (needs Blender)
```

### Importing Your Own

Two importers write bundles in the library's format. Neither calls a model.

A URDF robot:

```bash
python -m dycraft.agent.assets.import_urdf \
    --urdf path/to/robot.urdf --out assets/imported/my_robot --name my_robot
```

Each link becomes a part with a visual mesh and a simpler collision mesh, and each joint keeps its limits.
`--package-root` resolves `package://` mesh paths. `--help` lists the options for collision hulls, allowed
self-collisions, and default poses.

A rigged character (glTF/GLB, FBX, or a `.blend`):

```bash
python -m dycraft.agent.assets.import_gltf --in character.glb --inspect          # report only, write nothing
python -m dycraft.agent.assets.import_gltf --in character.glb --out assets/imported/my_character
```

The character becomes a Blender rig with its skinned mesh and its animation clips (`--clips`). `--up`, `--yaw`, and
`--scale` fix the orientation and size of models made in other conventions. `--reparent BONE=PARENT` attaches a bone
that the rig leaves behind when the body moves, such as a hoof skinned to an IK controller, to the limb it belongs to.
Files that give each part its own copy of the skeleton import as one rig. A rig driven through a separate control rig
(Rigify-style constraints) imports from its bind pose, so its size can be wrong and its clips can be lost.

When the skeleton uses a naming scheme the importer knows (Mixamo, 3ds Max Biped, Unreal, Rigify, and eight others), the
importer also records the direction of each palm and the bend axis of each joint. For a standing human figure,
the animation checks then compare every frame with anatomical joint limits for the shoulders, hips, elbows, knees,
forearms, wrists, ankles, head, and spine. This check reports problems but does not block the run.

After importing, rebuild the index so agents can find the new bundle.

## Motion Capture Libraries

`python -m dycraft.install mocap` fetches the CMU library of people and, with `--dog`, the MANN dog captures into
`assets/mocap/`. Agents can use any library placed as a directory there. Each fetched library comes with notes, an
index of its clips, and a retarget helper for agents. A library you add can include the same two files: a `NOTES.md`
describing it (the fetched libraries' notes, in `animation/mocap/libraries/`, are examples), and an `index.txt` with one
tab-separated line per clip or span (`<clip>`, `<start>-<end>s`, `<label>`, `<notes>`). Agents bring in its clips
themselves.

## Scenes: `--layer scene`

The agent writes one `scene.py` that builds the objects and places them, and can hand objects to helper agents. The
checks test that objects do not intersect, that everything stays inside the room and is supported, and that the
agent's assertions hold. The agent looks at its renders and can ask a separate agent to review them. The result is
`scene.urdf` with its `meshes/` and `views/`, at the top of the run directory. `--preload-asset` places a model the agent should not rebuild (see
[How Agents Use It](#how-agents-use-it)).

## Objects: `--layer object`

The agent writes `object.py`, a build123d program that adds rigid parts and connects them with joints. The checks test
that parts do not intersect at sampled joint positions, that the parts form one supported object, that the joints have
working ranges and sit on the parts they connect, and that the agent's assertions hold. The result is `object.urdf`
with its `meshes/` and `views/`, at the top of the run directory.

## One-Off Prompts: `--layer plain`

```bash
python -m dycraft.cli --backend codex --layer plain --prompt "summarize agent/prompts/object.md in ten lines"
```

The agent answers in text, without DyCraft's tools.

## Exporting to GLB

```
python -m dycraft.export glb RUN [-o OUT.glb] [--bake-materials] [--web] [--jobs N]
```

`RUN` is a run directory, one of the paper's films as downloaded (`films/<Bnn_…>/`), or a `.blend`. The output goes
beside it unless `-o` names another file: `animation.glb`, `scene.glb`, or `object.glb` in a run directory or film,
and `<name>.glb` for `<name>.blend`. It needs only the Blender install.

The file is plain glTF 2.0, with no required extensions. Blender 4.5 imports it and three.js loads it. Other glTF
viewers should open it, but we have not tested them.

A film keeps its motion. Parts that move rigidly become animated nodes, characters become skinned meshes, and other
deforming meshes, such as cloth, become morph targets. Every camera is included. glTF cannot hold camera cuts and
zooms, so they go in `<name>.cuts.json` beside the GLB. An object or scene is exported at rest, and lights are not
exported.

Procedural materials become one flat color each. The command lists the objects that have them. `--bake-materials`
bakes them into textures with Cycles instead. On a CPU, a scene of 69 objects took 6.5 minutes and 7.7 GB of memory,
so use a GPU for a film.

`--web` also writes `<out>.web.glb`, a compressed copy for web pages. It needs Node.js 20 or later and, once,
`npm --prefix export/web ci` from the clone. If that step is missing, `--web` prints the exact command. Blender cannot
import this copy, because it uses meshopt compression.

The command prints the memory it expects before the heavy steps. `--jobs` sets the number of processes. The default is
up to 4, as memory allows. On a 16-core CPU with no GPU:

| Film | Time | Memory | GLB size |
|---|---|---|---|
| B24, 32 s | 18 s | 0.9 GB | 26 MB (4.2 MB with `--web`) |
| B11, 93 s, with cloth | 7 to 14 min | 13.8 GB | 117 MB |
| B11 with `--jobs 1` | 16 min | 8.7 GB | 117 MB |

With 16 GB of memory, use `--jobs 1` for a long film.

## Fixing Problems While Running

### A Launch Stops With "Nothing Was Started"

The launcher runs the setup checks first. Each failing check prints `FATAL: <check>: …` with its fix. Apply the fix, or
run `python -m dycraft.install check` for the full report. `DYCRAFT_SKIP_PREFLIGHT=1` launches without the checks.

### A Run Stopped at Its Time Limit

The launcher exits with code 124, and `src/meta.json` records the stop under `time_limit`. The limit counts wall-clock
time, including waits for an account's usage limit to reset. To continue, resume the run with `--max-hours`. For long
briefs, pass a larger `--max-hours` (or 0 for none) at launch.

### Stopping a Run Early

Create `runs/<run_id>/src/.terminate`, and a process outside the sandbox stops the agent. Interrupting the launcher in its
terminal also stops the run. The run directory keeps everything up to that point, and `--resume` continues it.

### The Agent CLI Opened No Session in 30 Seconds

DyCraft starts Codex without its update and rate-limit notices, and answers its "Do you trust this directory?" screen
with a temporary entry in `~/.codex/config.toml`. A CLI that still opens no session within 30 seconds is stopped with
"FATAL: codex opened no session in 30 s …" or "FATAL: claude wrote no session record in 30 s …". Launched from a
terminal, it is left running so you can see the screen. Otherwise, run `codex` or `claude` once by hand in the run's
`src/` and answer the screen it shows.

### An Animation Launch Stops With "No Terminal to Ask Whether to Continue"

One of the two helpers that write `scene.txt` and `motion.txt` failed, and there was no terminal to ask whether to go
on without it. Pass your own files (`--scene-brief FILE`, `--motion-brief FILE`), skip the helpers
(`--no-scene-brief`, `--no-motion-brief`), or pass `--on-rewrite-failure continue`. The failed attempt is kept in the
run directory.

### The Prompt Asks for Blender Renders but the Images Come From Open3D

Naming a renderer in the prompt does not select it. Pass `--render-backend blender`, or set
`DYCRAFT_RENDER_BACKEND=blender`. The launcher warns when the two disagree.

### Surfaces Render Flat Grey

Open3D does not render materials. Use `--render-backend blender` when appearance matters.

### The First Blender Render of Each Run Is Slow

Blender workers share their compiled shader and kernel caches in `$TMPDIR/dycraft-blender-cache-<uid>`, so only the
first worker compiles them. Where `$TMPDIR` is cleared after each login or job, as on many clusters, set
`DYCRAFT_BLENDER_CACHE` to a local directory that persists.

### "--prompt foo.md: No Such Prompt File"

A single word ending in `.md` is read as a file path. Check the path, relative to the directory you launch from.

### "--prompt Is Empty"

The prompt was blank. This usually means `--prompt "$(cat file)"` ran on a missing file.

