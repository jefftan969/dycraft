# DyCraft Run Viewer

We provide an interactive viewer for inspecting finished runs and agent trajectories. It reads runs of every layer
from `runs/` and writes nothing there.

The frontend is vendored from [Articraft](https://github.com/mattzh72/articraft) under Apache-2.0, as the top-level
`NOTICE` records. Nothing here is imported by the harness or the build path.

## Setup

The server's packages (FastAPI, uvicorn, websockets) install with DyCraft. Building the frontend needs Node.js ≥ 20.19,
once:

```bash
npm --prefix viewer/web ci
npm --prefix viewer/web run build
```

## Run

From the repo root:

```bash
python -m dycraft.viewer                        # API + SPA on http://127.0.0.1:8770
python -m dycraft.viewer --runs-dir /other/runs
python -m dycraft.viewer --dev                  # API only; `npm run dev` serves :5173
```

`--runs-dir` defaults to `$DYCRAFT_RUNS_DIR`, else `runs/` under `$DYCRAFT_HOME`, else under the checkout, else under
the current directory for a regular install.

## Layout

```
viewer/
  app.py           FastAPI routes + SPA mount
  runs.py          read runs/ — meta, IR summary, compile reports, URDF, files
  scene.py         scene-layer half: per-run catalog + asset-path rebasing
  animation.py     animation-layer half: staged scene, summary, events, and the
                   native timeline (from native_animation.py)
  live.py          the interactive Cycles stream: target resolution, camera and
                   pose extraction, frame framing (WS /live)
  derived.py       shader parts' flat colors and baked-atlas viewer assets, and
                   their status
  trajectory.py    normalize the LLM trajectory (two producers, one shape)
  cost.py          optional USD estimation from prices.json ($DYCRAFT_PRICES_JSON overrides)
  __main__.py      python -m dycraft.viewer
  web/             React 19 + three.js + Vite frontend
```

## Navigating the Viewport

Drag to orbit and scroll to zoom. The keyboard flies the camera, which a large scene needs: in a 40 m hardware store
the interesting parts are behind other parts, and orbiting cannot get the camera between two shelves.

| Keys | Moves |
|---|---|
| `W` `A` `S` `D` | Forward, left, back, right |
| `↑` `←` `↓` `→` | The same, for the other hand |
| `E` / `Q` | Up / down |
| `Space` / `Shift` | Up / down |

`Space`/`Shift` work alongside WASD and `Q`/`E` alongside the arrows. The scheme is in
`web/src/components/viewer3d/camera-keys.ts`, pinned by `npm run check:camera-keys`:

* Forward follows the view direction, pitch included, as in other 3D editors, while strafing and the vertical keys
  stay in the world frame, so `A`/`D` never drift up and `Q`/`E` are always plumb. That keeps a nose-down view
  recoverable.
* Speed is 0.9 of the orbit distance per second rather than a fixed speed, because the viewer draws both an 8 cm mug
  and a 40 m store. Each step is clamped to 0.1 s of travel, so a backgrounded tab does not resume by teleporting.
* The camera and `controls.target` translate together. OrbitControls rebuilds the camera from `position - target` on
  every `update()`, so moving one without the other is either undone on the next frame or silently becomes a
  rotation. The orbit pivot therefore travels with you: orbiting after flying rotates around what is now in front of
  you.

The keys are live while the pointer is over the viewport, or after clicking it (the viewport takes focus). They are not
global, because the inspector's scrollable panels use the arrow keys.

## Scene Runs

An object run commits an `ObjectIR` and a scene run a `SceneIR`. Everything IR-shaped keys off `scene.ir_kind(ir)`, the
payload, rather than off `meta.json`'s `layer`.

A scene's geometry is not in its `ir.json`. That file holds only *instances*, each a content hash into
`<run_dir>/catalog/`, and the catalog objects' asset paths (`assets/meshes/<digest>.obj`) resolve against `catalog/`
rather than against the run root. So the viewer loads the catalog and **rebases** those paths in memory
(`catalog/assets/…`) so the one `files/` base URL resolves them, instead of copying meshes into a run it only displays.

* The Parts panel flattens a scene's parts and joints across instances, prefixed `<instance>__`, the same names
  `scene_ir_to_urdf` gives the URDF's links and joints, so a row names the same thing the 3D view does. Joint sliders
  work for an articulated object placed in a scene.
* The scene and motion rewriters a run started sit in its `sub_runs/` and are addressed as `<parent>/sub_runs/<name>`.

## Animation Runs

An animation run carries a `dycraft.animation.blender.v1` manifest in `ir.json` and a hashed `bundle/` containing the
accepted `.blend` and evaluated frame geometry. The staged `scene.json`/`catalog/` provide scene context; playback uses
the evaluated meshes through `viewer/native_animation.py` and `useNativeAnimation.ts`. Overview event results and
timeline markers share the committed run's `EventsSatisfied` evidence, including fractional FPS. Failed, deferred, or
stale event results do not produce success markers.

Transport is split the way the data is: geometry barely changes between frames, and motion changes every frame.
`native-track` is one request for the whole clip, the object table plus a `[frames][objects][12]` block of matrices,
and `native-mesh/{digest}` is one request per *distinct* evaluated mesh, served `immutable` because the URL names the
bytes. Advancing a frame is writing matrices. A frame record is hashed against the manifest, and a mesh is checked for
*membership* in it and for content by the digest it is named after.

The meshes load in the background (`native-stream.ts`), because a long film's deforming objects carry one mesh per
frame, which can be thousands of meshes and gigabytes once built. The first picture needs every mesh an object keeps
for the whole clip, plus the current frame's deforming meshes. If the whole clip's built size (estimated from the first
picture) then fits the budget, the rest of the clip loads from the playhead on. Otherwise a window from 1 s behind the
playhead to 4 s ahead fills, nearest first. The budget is 1024 MB by default, set on the server with
`DYCRAFT_VIEWER_CLIP_BUDGET_MB` (`/api/config` passes it to the page). Past the budget, the meshes least recently
wanted are dropped, never one in the window or on screen. Playback starts, and resumes after a stall, once the next 2 s
are loaded, as a video player buffers. An object whose mesh has not arrived keeps its nearest loaded one. The server
keeps each run's parsed manifest, so a mesh costs it one `.npz` read.

Object colors come from the material's Principled BSDF Base Color (`worker.surface_color`), not from
`Material.diffuse_color`, which is Blender's viewport swatch: nothing in this repo writes it, so it sits at the default
0.8 gray and would draw whole scenes in one flat color. A staged part's own material wins over both when its
`identity` resolves to a URDF link; only the scalars are borrowed, since an evaluated mesh carries no UVs. Exact Blender
shaders are visible in the rendered film, `bundle/film/clip.mp4`, which `ir.json`'s `film` names.

Runs in the earlier AnimationIR format (dense per-frame deltas over a staged scene) show no timeline.

Native evaluations (`dycraft.evaluation.v3`, and v2) store mesh arrays under `evaluation/meshes/<sha256>.npz` and
reference them from frame JSON records. Bundles that store each frame's meshes in `<frame>.npz` stay readable through
the shared evaluation reader.

## Shader Materials in the Viewer

`Material(shader=...)` makes a Cycles node graph the only authority for a surface's appearance, and a browser cannot
evaluate one. A shader material also carries no `color`, so `render/urdf_export.py` writes the URDF fallback
`128,128,128` for it.

The viewer draws a shader part in one of three ways. All three are viewer-only: no IR key is added, no object id moves,
and the staged assets, the canonical URDF, and the IK solver export are untouched.

### Flat Colors

The default. Each shader part draws in one color, roughness, and metallic value, the ones the GLB export gives it
(`export/_blender/flat_material.py`), so the viewer and an exported GLB agree. `render/viewer_flat.py` reads them from
every shader part's node graph in one Blender process per run, which takes seconds because it builds no meshes. The
process runs in its own interpreter (`viewer/_flat_pass.py`), is started by the first URDF request (`viewer/derived.py`
`flat_colors`), and caches the values per object by IR id under `flat/` in the viewer asset cache. Until they are read,
the parts draw gray: `files/model.urdf` sends `X-Dycraft-Flat-Colors: pending`, and `useUrdfLoader` asks again every
1.5 s and reloads the model in place once the header says `ready`. A failed pass (no Blender, for example) leaves the
parts gray and is not retried until the server restarts.

### Baked Atlas Assets

`bakedMaterials`, off by default. `render/viewer_bake.py` unwraps a packed non-overlapping
atlas on the real mesh, evaluates the original graph at every surface point, and exports a per-part GLB.
`GET /api/runs/{id}/derived` reports status and starts the bake; `files/model.urdf?derived=1` returns a variant whose
`<mesh filename>` entries point at those GLBs, under the reserved prefix `__derived__/<ir_id>/<file>.glb`, which
`app.py` intercepts and serves out of `~/.cache/dycraft/viewer_assets` (override with
`DYCRAFT_VIEWER_ASSET_CACHE_DIR`). Only the mesh filenames change: the joint tree, the link frames, and the collision
proxies are the canonical exporter's, so articulation and Preview Motion work on the same graph.

The `<ir_id>` segment is load-bearing. A **scene** needs one asset per catalog object it instances, each with its own
cache entry, and their manifests number files per object, so every object has a `000_*.glb`. The substitution happens
in the *catalog*, not in the SceneIR: `scene_ir_to_urdf` replicates each object's own link tree and reads mesh
filenames out of that object's `params`, so rewriting the catalog reaches every instance, including repeated ones,
which share a single entry and therefore a single download.

Only shader-authored objects are candidates. A run with no `Material(shader=...)` reports `unavailable` and is never
baked, since three.js already draws scalar and textured materials correctly. A scene that mixes the two bakes only the
shader objects; the rest keep their staged meshes and their URDF materials.

Status is aggregate for a scene: `ready` only when every candidate has landed, `partial` when some baked and the rest
permanently failed. The URDF is then a **mixture**, and `useUrdfLoader` decides *per visual*, from the filename,
whether to keep the material the file arrived with (a derived GLB) or build one from the URDF spec (a staged OBJ).

With "Baked materials" on, opening an un-baked run starts the bake as a side effect of a GET, at 15–95 s per object,
one at a time process-wide. A six-object scene is therefore minutes of GPU on first open, and cached from then on. That
cost is why baking is off by default.

### Live Cycles Stream

`blenderRender`, off by default. `viewer/live.py` plus `WS /api/runs/{id}/live` drive a
resident `_blender_child.py --serve` session with `use_persistent_data`, and paint its frames over the WebGL canvas
while three.js keeps owning interaction underneath. It runs at 18–28 fps orbiting at 640×360, with a floor of ~20–32 ms
a frame. The baked path is *closer* to the canonical image (0.05–0.37/255 against the stream's 0.56, of which JPEG
costs 0.53) and needs no GPU at view time, so it is the one to turn on for the full appearance.

## Loading Large Scenes

A scene is served as one generated URDF plus one mesh file per visual, from tens of meshes to over 5,000. Five choices
keep that fast:

1. **Selective compression.** `app.py`'s `_SelectiveGZip` passes asset routes (`files/`, `native-track`,
   `native-mesh/`) through uncompressed and compresses everything else at level 1. Starlette's default level 9 takes
   about thirty times the CPU of level 1 for a file about six percent smaller, and it runs on the event loop, so every
   mesh would queue behind the one thread compressing. The content-addressed routes (`native-mesh/`, hashed files
   under `assets/`) are served `immutable`, so the browser fetches each once.
2. **Blocking handlers off the event loop.** `list_runs`, `run_urdf`, and every transcript parse go through
   `run_in_threadpool`, so none of them stalls the mesh requests in flight.
3. **A cached URDF.** `_urdf` keeps a small cache keyed by the size and mtime of `ir.json`, `catalog/manifest.json`,
   and `scene.json`, so a re-selection or a baked-variant reload of a big scene costs milliseconds instead of a second,
   and a live run rewriting its IR still invalidates it.
4. **A bounded client fan-out.** `geometry-loader.ts` caps mesh downloads at 12 in flight: thousands at once queue
   behind six usable connections, and Chrome starts failing them (`ERR_INSUFFICIENT_RESOURCES`, which surfaces as the
   bare message "Failed to fetch"). `attachVisualMeshGeometry` uses `allSettled`, so a scene where one part in a
   thousand has not arrived shows the other parts and a note.
5. **Bundled meshes.** `viewer/mesh_bundle.py` packs the meshes a document references into 16 MB tar chunks, so a
   scene is a manifest plus a handful of archives: 3 requests for a 5,403-mesh scene. Chunks rather than one archive,
   because a 169 MB response is past what a browser will keep in its cache. The client half (`mesh-bundle.ts`) is a
   **prefetch**: every entry carries its run-relative path and is primed into the template cache the per-file loader
   reads, so a server with no such route, a chunk that fails, a `.glb` baked asset, or a path too long for a USTAR
   header all fall back to their own request. `npm run check:mesh-bundle` pins the tar reader against an archive
   `mesh_bundle.py` wrote, plus hand-built edge cases (non-multiple sizes, `prefix`-split names, truncated archives).

On the largest scenes `OBJLoader.parse` takes up to a few seconds in the browser. Each load logs its phases, such as
`[dycraft] 562 links in 412 ms — urdf fetch 12 ms, urdf parse 31 ms, graph 18 ms, bundle 6/6 chunks …`, so a slow
scene can be attributed. A large scene's `/api/runs/<id>` detail payload is also megabytes of per-instance and
per-part facts.

### What Makes a Scene Heavy

Most of a heavy scene's bytes are often a few props with many separately tessellated repeated parts (a hose of a
hundred ribs, a dozen helical spring coils), not what the brief is about. A per-object breakdown shows this quickly:

```bash
python - <<'PY'
import json; from pathlib import Path
run = Path("runs/<id>")
per = {}
for cat in (run / "catalog").glob("*.json"):
    if cat.stem == "manifest": continue
    ir = json.loads(cat.read_text())
    for part in ir.get("parts") or []:
        mesh = (part.get("params") or {}).get("mesh")
        if mesh and (run / "catalog" / mesh).is_file():
            per[ir["name"]] = per.get(ir["name"], 0) + (run / "catalog" / mesh).stat().st_size
for name, size in sorted(per.items(), key=lambda kv: -kv[1])[:5]:
    print(f"{size/1e6:7.2f} MB  {name}")
PY
```

### Replacing a Drawn Scene

`useUrdfLoader` distinguishes a **first load** for a selection from a **variant swap**, the same run reloaded because
its baked-material asset landed or its IR was recompiled. A first load covers the canvas. A variant swap keeps the
existing scene on screen, orbitable and posable, until the replacement is built and swapped in one commit (the old
group is removed only after the new geometry is attached).

* The camera is preserved by `preserveViewAcrossModelSwitch`, so the swap reads as the appearance sharpening rather
  than as a reload.
* The pose is preserved by `useJointController`, which resets the joints only when the joint set itself changes, not
  whenever a new spec object arrives, so a door the user opened stays open when the bake lands.
* A failed variant swap keeps drawing what is already there and says so in a chip, rather than blanking the viewport.

## Staying Responsive While a Run Loads

The page's heavy parsing runs in loader workers (`web/src/workers/`):

- the trajectory and agent-timeline JSON
- the mesh bundle's OBJs, with their normals and bounds
- the native clip's meshes and their creased normals

`creased-normals.ts` is three's `toCreasedNormals` method, changed to merge only vertices that coincide. Three's own
merges positions on a 1 cm grid, which on a part a few centimeters across averages the normals of surfaces that do not
touch and takes seconds per mesh.

Posting a parsed payload back whole would cost the page a deserialization about as long as the parse, so a JSON payload
comes back as its small fields first, then each large top-level array in slices. Meshes come back as transferred typed
arrays. On the page, building the model's visuals yields to the event loop every 8 ms (`lib/yield.ts`). The
agent-timeline strip draws events too narrow to tell apart as one segment, which opens the first event's turn and says
how many events it holds. The Trajectory tab renders its turns as it is scrolled.

The viewport's loading tray (`LoadingTray.tsx`) has one row per item still loading (geometry, colors, animation
including buffering, trajectory, and timeline), with progress where it is known. It is a DOM overlay, so snapshots never
include it.

## Where the Trajectory Comes From

The trajectory is read from the CLI's session transcript (`claude` or `codex`). External-CLI transcripts record
thinking blocks with an empty `thinking` string (only the signature is persisted), so their reasoning is
unrecoverable, and the UI says so. What those runs do carry: every turn's visible narration, the exact commands, tool
results, and the `input` / `cache_read` / `cache_write` / `output` token split.

`agent/transcript_capture.py` stages that transcript into `runs/<id>/agent_transcript.jsonl` when a run finishes, so
bundles stay self-contained. For a run without one, the viewer reads `$CLAUDE_CONFIG_DIR/projects/<munged-run-dir>/`.

## Cost

Rates ship in `viewer/prices.json` (USD per 1M tokens; its `_source` says which published rates and as of when). Point
`$DYCRAFT_PRICES_JSON` at your own file to override them. Token counts are exact either way, and a model with no rate
renders as `--` rather than a guessed dollar figure.

## Notes

* 3D needs `ir.json`, which is written when a run commits (plus, for a scene, the `catalog/` snapshot beside it). An
  interrupted run shows its prompt, trajectory, and tokens with an empty viewport.
* **Refresh**, top right of the run list, re-reads `runs/` and reloads whatever is selected. The list is otherwise
  fetched once on mount, so this shows a run that finished, or a corpus that was staged, while the viewer was open,
  without a restart that would lose the selection and every expanded folder.
* **Flip joints**, at the top of the Parts tab, mirrors every movable joint to the opposite side of its own range at
  once (`web/src/components/viewer3d/joint-flip.ts`, pinned by `npm run check:joint-flip`). A scene commits with its
  doors, drawers, and lids shut, so its rest pose says nothing about whether they open the right way; one click swings
  the lot, and the next undoes it. Disabled while Preview motion or a clip is driving the joints, since those rewrite
  every joint per frame. Preview motion, in the Render tab, sweeps each joint on its own phase-offset cycle and so
  never shows the whole mechanism displaced at once.
* The URDF is generated on demand from `ir.json` rather than served from `final_render/ir.urdf`, because the generated
  form's mesh paths resolve against the run root for every run, which is what `files/` is the base URL for. It is
  exposed as `files/model.urdf`, the filename the vendored three.js loader appends to its base URL.
