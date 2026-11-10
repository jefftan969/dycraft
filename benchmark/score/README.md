# The Physical-Plausibility Scorer

```bash
python -m dycraft.benchmark.score runs/bench_long                   # every arm of a benchmark directory
python -m dycraft.benchmark.score runs/x/dycraft_codex other_method/b13_dog_fetches_newspaper.blend
```

This computes the paper's two automatic metrics (Table 2) for any film delivered as a `.blend` or as a DyCraft bundle:

- the **penetration rate**: the share of frames in which some moving mesh penetrates another;
- the **floating rate**: the share of moving body-frames with no support path.

Lower is better for both. A film's rate covers its own frames; the paper's numbers are unweighted means over the 24
briefs.

| Path | Geometry |
|---|---|
| A DyCraft run, arm or bundle, or its `bundle/animation.blend` | `bundle/evaluation/` as delivered, or, for a film shipped without it, exported from `animation.blend`; the joint tree from `design_state/baseline.json` (never the export's Blender parenting), and materials and armature rigs from `animation.blend` |
| A bare agent's arm (`bundle/animation.blend`) or any `.blend` | Exported frame by frame with the animation layer's own exporter (`worker.evaluate`). The file is opened with `use_scripts=False` and never saved. Captions and camera overlays are removed, static curves are converted to meshes, and visibility, Blender parenting and material classes are recorded |
| A benchmark output directory | Every arm in it; each arm also gets `benchmark/score.json` for `python -m dycraft.benchmark.report` |

A bare arm's `bundle/evaluation/` from `stage.py` is not used for scoring: it lacks the visibility, parenting and
material records, and it keeps captions and overlays.

**Briefs.** Name a `.blend` after its brief (`b14_...`), or pass `--task B14`. Three briefs exempt a body from the
floating rate (`tasks.json`): the self-propelled crow in B14, and the buoyant float in B20 and boat in B22. Every other
film needs no brief.

**Results.** Results go to `--out` (default `<runs dir>/.scores/`), one folder per film. Each folder holds
`penetration.json` and `floating.json`: the headline under `primary`, plus offending pairs and bodies, causes and
events under `diagnostics`; `penetration.json` also has the rate at fixed depth thresholds under `sweep`. Each stage's
geometry is cached there too, keyed by its inputs and the code that measured it, so a re-run reuses them.

**Requirements and cost.** It needs a Blender Python (`python -m dycraft.install blender`) and `coal`, both of which
the animation layer already requires. Geometry dominates the cost. On the paper's 24 DyCraft films it took from a few
CPU minutes to about 3 CPU-hours per film, about 19 CPU-hours in all; the 24 bare films took about 8. `--jobs N`
splits a film's frames over N processes. A geometry process grows with the frames it measures (about 20 MB per frame
on a busy film), so for a long film use `--chunks M` with M larger than N: M pieces, N at a time.

## Penetration Rate

    penetration rate = frames with >= 1 counted pair / evaluated frames

A pair-frame is counted only if it survives every step below.

1. **Bodies.** Every visible mesh object is a body. These are left out:
   - hidden objects: `hide_render`, collection, or view-layer exclusion, `visible_camera` off, or scaled below
     0.1 mm on every axis;
   - text, and objects parented under a camera;
   - fluid domains and particle emitters;
   - **meshes whose triangulation changes between frames** (remeshed liquids, cloth re-tessellated per frame);
   - **liquid and effect meshes, judged by material**. A body is liquid when every material slot has an IOR of
     1.30–1.36 or a volume shader. It is an effect when every slot has its alpha driven by a node, or uses only
     transparent, emission or holdout shaders. Glass (IOR ~1.47, constant alpha) stays solid.
2. **Pairs.** A box broad phase finds candidate pairs, keeping those where at least one body ever moves. Some pairs are
   never measured:
   - **rigidly coupled** pairs, whose relative pose never changes (welded parts, static-static pairs);
   - **directly jointed** pairs: adjacent links of a DyCraft joint tree, or adjacent links of a Blender parent chain
     read as a URDF-style link tree.

   Agent-authored collision allowances are never read.
3. **Contact and depth.** The animation layer's `collision.collision` decides contact: a Coal BVH, an erosion
   re-check, and winding-number containment. Depth is the smallest translation that separates the pair. It is
   minimized over up to 18 directions (world axes and both bodies' axes, each sign) and bisected to 0.1 mm, so it is
   an upper bound on the true minimum.
4. **Threshold.** A pair-frame passes when its depth is at least τ = max(1 mm, min(5 mm, ½ × the thinner body's
   smallest extent)). A body wholly inside the other passes at any depth. A sweep at fixed thresholds (1–50 mm) is
   reported too.
5. **Initial overlap.** A pair already penetrating at its first shared frame is exempt until it separates (2 clear
   frames) or deepens by 2 rungs of the 1/2/5/10/20/50/100/200 mm ladder. This holds whether or not the pair moves.
6. **Motion.** At least one of the two bodies must be MOVING at that frame. A body is MOVING when a corner of its box
   moved more than float32 resolution, or moved within the last 0.5 s and has not yet stayed still for 0.5 s.
7. **Self pairs** (two parts of one object, meaning one DyCraft instance or one top-level Blender parent):
   - never counted: parts on rigid groups joined by a moving joint, and parts on connected bones of an armature rig
     (bone parent, a head on a tail, or a pivot inside a part);
   - counted only beyond max(2 × the starting depth, starting depth + 20 mm): parts that overlap in ≥80% of their
     first 0.5 s, or in the rig's rest pose.
8. **Consumption.** Frames on which an object is *consumed* are not counted. An object is consumed when it is wholly
   inside an *agent* (an object whose parts move relative to each other or deform), and only if all of these hold:
   - it entered from outside through a penetrating contact;
   - it moved;
   - it comes to ride with the body that encloses it;
   - it stays inside for ≥0.5 s;
   - it was a separate object before.

   The bite itself still counts.

Self and cross-object pairs both count; `primary` also reports each separately.

## Floating Rate

    floating rate = unsupported dynamic body-frames / dynamic body-frames

**Counted units.** A body is *dynamic* if the animation moves it at all: it is MOVING at some frame and displaced
≥1 mm from its first pose. It then counts once per frame on every frame it is visible, at rest or not. Each mesh is a
body.

**Support.** Support is reachability in a per-frame graph whose sources are:
- the **ground**: never-moving solids with a footprint ≥4 m² in the lowest 0.25 m band, plus wall panels touching it;
- the bodies a brief exempts;
- **every never-moving body**, so a moving part is charged only for its own support failure. A static structure with
  no path to the ground is reported once, as `static_grounding_failure_rate`, and `strict_floating_rate` uses the
  ground alone.

**Edges.** Every edge needs measured contact within a size-adaptive tolerance (min(tol, max(floor, fraction × the
smaller body's smallest extent))). Contact starts at the tolerance and holds until the gap exceeds 1.5 × the
tolerance.

| Edge | Rule | Tolerance |
|---|---|---|
| Passive A → B | A rests on B: a contact sample on a surface no steeper than 75°, below A's centroid, or A embedded below its centroid | 5 mm |
| Pivot A → B | A turns ≥0.2° about one fixed axis that runs through B (an undeclared hinge, arbor or pin) | 5 mm |
| Mounted A — B | Both still, contact in any direction | 5 mm |
| Structural A — B | A DyCraft rigid group or joint, or Blender parenting | 5 mm |
| Bonded A — B | ≤2 mm relative drift per frame while traveling ≥10 mm together (a grasp or a carry) | 10 mm |
| Carrier A — C | Contact with a deforming body (a cord, cloth, a skinned coat) | 5 mm |

**Free flight.** An unsupported group whose centroid follows a gravity arc is legal: downward acceleration of
0.5–1.5 g, ≤0.3 g sideways, and an rms residual ≤ max(5 mm, 2% of the path).

**Body classes** are automatic:
- a part named as liquid or text ('Water / reservoir', 'title') is not scored and supports nothing by contact;
- a deforming mesh is a carrier.

## Against the Paper

On the paper's final films, the 24 DyCraft and 24 bare films behind Table 2 (VIGA's films are not part of the
release), the scorer reproduces Table 2 exactly. Scored from scratch, every film gives the paper's per-film
penetration and floating rates with the same counted frames and body-frames, so the means are the paper's: 13.0% and
2.7% for DyCraft, 80.1% and 15.7% for the bare agent. This was checked with Blender 4.5.13 LTS (`bpy`) and coal 3.0.3,
the versions of the paper's runs.

The paper scored the DyCraft films from their delivered evaluations (B01, which had none, from its blend). The films
release ships each DyCraft film as its blend and baseline, without its evaluation; scored from the blend, a film gives
the same rates, verified on B05, B17, B20, B23, and B24 and on four of the released films. B06 is the exception: its
Table 2 row was scored on its 8 fps source motion (651 frames), which is not in the films release; the released film is
its 24 fps re-timing (1,953 frames), so it is scored on different frames.

Exact reproduction uses the paper's chunk partition, `--chunks` of `clamp(frames // 60, 8, 16)` for DyCraft and
`clamp(frames // 150, 2, 12)` for the bare agent, because in a rare near-degenerate contact a pair's depth, computed
once per chunk for each rounded relative pose, depends on where a chunk starts (of the 48 films, only DyCraft B16).

The paper's §4 summarizes these definitions in two sentences. The exact rules above differ from that summary in
places a re-implementation would get wrong:

**Penetration**
- The 5 mm tolerance is adaptive, and full containment counts at any depth.
- Only frames where a body moves are counted.
- The initial-overlap exemption covers moving pairs as well as the static scene.
- Remeshing meshes are not measured.

**Floating**
- It is a share of body-frames, not of frames.
- Every never-moving body is a support source.
- Ballistic flight is legal.

## Limitations

- Only rendered frames are checked, so a body can tunnel between two frames (`tunneling_risk` reports fast bodies).
- Depth is an upper bound over the direction set.
- An overlap present from the first frame that never separates or deepens is not counted.
- A single deforming mesh cannot penetrate itself.
- Body classes come from names and materials, so an opaque effect mesh counts as solid, and a solid part named like
  a liquid is excluded.
- Hinges are known from declared joints in a DyCraft film and from parenting in a Blender scene. The pivot rule
  credits undeclared hinges in both.
