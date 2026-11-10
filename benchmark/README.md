# Benchmark: DyCraft Against a Bare Agent

Run the animation benchmark's briefs through DyCraft and through a bare CLI agent with Blender, then
compare what each delivered.

| Command | What it does |
|---|---|
| `python -m dycraft.benchmark.run` | One `python -m dycraft.cli --layer animation` run per brief |
| `python -m dycraft.benchmark.bare` | One sandboxed bare Codex or Claude Code session per brief, then stages its `.blend` and film for the viewer |
| `python -m dycraft.benchmark.score` | The paper's penetration and floating rates for every arm (or any `.blend`); [score/README.md](score/README.md) |
| `python -m dycraft.benchmark.report` | A side-by-side table and `report.html` for a results directory |
| `python -m dycraft.benchmark.sizing` | Each brief's planned and predicted film length and expected runtime (`--beats FILE` sizes a brief of your own) |
| `python -m dycraft.benchmark.stage RUN_DIR` | Rebuilds a DyCraft run's `bundle/evaluation/` from its blend, in place, so the viewer can play a run that travels without it |

The briefs are `agent/prompts/animation_24_briefs.md`: 24 long briefs, 13 short cuts (B01S–B12S, B15S)
and 10 medium cuts. `--briefs` takes a set or ids. A set takes each brief's cut where there is one; where
there is none, in the short and medium sets the DyCraft arm takes the variant that casts its characters
from the asset library (`animation_briefs_imported_characters.md`), and the bare arm, which has no
library, the brief as written:

| `--briefs` | Briefs |
|---|---|
| `long` | B01–B24 |
| `short` | B01S–B12S and B15S; B13, B14 and B16 (imported characters for the DyCraft arm); B17–B24 |
| `medium` | B01M–B04M, B06M, B08M–B12M; B05 and B07; B13–B16 (imported characters for the DyCraft arm); B17–B24 |
| `B01S,B13` | Those ids |

A listed id runs as itself: `--briefs B13` is B13 as written with its own set's budget (24 hours), where the short
set gives the DyCraft arm the imported-character B13 and 8 hours. To repeat one member of a set exactly, run the set,
or add `--imported-characters` and that set's `--budget-hours` and `--film-seconds`. Each record names the set it ran
in (`horizon`).

`--prompt TEXT` (or a `.md` file) runs one brief of your own instead, wrapped the same way; its directory is named
from its first words. For the bare arm, `--g1` mounts the Unitree G1 at `/assets/g1` and adds the sentence the paper's
G1 briefs gave their bare runs, so a prompt about a humanoid robot gets the robot the DyCraft arm casts from the
library.

## Example: Short and Medium Briefs, Codex and Claude, Both Arms

```bash
for set in short medium; do
  for agent in codex claude; do
    out=runs/bench_${set}_${agent}
    python -m dycraft.benchmark.run  --briefs $set --agent $agent --budget-hours 2 --out $out
    python -m dycraft.benchmark.bare --briefs $set --agent $agent --budget-hours 2 \
        --film-seconds default --out $out
    python -m dycraft.benchmark.score $out
    python -m dycraft.benchmark.report $out
  done
done
```

Here both arms are told the same film length and the same 2-hour budget, and each is stopped 20
minutes after it. Without `--budget-hours` and `--film-seconds`, the bare arm gets the paper's form
instead (below). Add `--model` and `--effort` to pick the agent model and reasoning effort; both are
passed through verbatim. `--parallel N` runs N briefs at once. `--dry-run` writes the prompts and prints the commands
without starting anything. Both arms of a brief land under the same `--out`:

    <out>/<brief>_<staged id>/
      prompt_dycraft_<agent>.md      the DyCraft arm's prompt
      dycraft_<agent>/               the DyCraft run (bundle/, finish_result.json, …)
        benchmark/                   record.json for the report
      bare_<agent>/                  control/ (prompt, config, events, sessions), workspace/,
                                     bundle/ + ir.json (the staged blend and its film), meta.json,
                                     benchmark/record.json
    <out>/report.html

`python -m dycraft.viewer --runs-dir <out>` lists both arms; the bare arm plays from its staged
bundle like a DyCraft run. The bundle's film (`bundle/film/clip.mp4`) is the agent's own
`output/film/final.mp4`, or, when it delivered none (a hard stop before its render, say), one the
harness renders from the blend's own camera, frame range and frame rate with DyCraft's final-render
defaults at 960x540; `--no-render` skips that render. The record says which (`film_source`).

## What Each Arm Is Given

The DyCraft arm receives the brief text unchanged, and the bare arm the same text with two sentences adapted (below).
Around it, each gets what the paper's runs got:

- **DyCraft**: the brief, then a film-length paragraph ("Expect a film of no more than about 60–75
  seconds …"; 120–150 s for the medium set), then a wall-clock budget paragraph ("Work to a wall-clock
  budget of N hours …"), passed with `--advisory-hours`. The scene and motion rewriters then add
  `scene.txt` and `motion.txt`. Default budgets: 8 hours for every brief in the short set and 24 in the
  medium and long sets (the paper's long runs had 8); a listed id takes its own set's, and a brief of your
  own 8. `--film-seconds` and `--budget-hours 0` change or drop the paragraphs; `--budget-hours 0` also
  turns the countdown off (`--advisory-hours 0`). The short and medium sets
  take the imported-character variant of B10 and B13–B16 where there is no cut; `--imported-characters`
  does the same for the long set and listed ids.
- **bare**: the paper's task wrapper ("Use Blender to create the following animation from scratch …")
  around the brief, then `---`, then `bare_prompt.md`, the paper's bare guidance. The bare agent has no
  asset library, so it never gets an imported-character variant, and two sentences of a brief are
  adapted: a G1 brief's "take it from the asset library" sentence becomes a pointer to the robot
  description at `/assets/g1`, and a people cut's casting sentence ("Take them from the asset library
  rather than creating them.") is dropped, while the cut's description of its cast stays. By default there is
  no length or budget paragraph, only a hard stop (`--hours`, default 72, the paper's limit). For the
  24 long briefs this is byte-for-byte the prompt the paper's bare runs received
  (`tests/test_benchmark_briefs.py`) when the installed Blender is 4.5.13, as theirs was; the
  environment sentence names the installed version. Opt-in `--budget-hours H` and `--film-seconds N|LOW-HIGH|default`
  add the DyCraft arm's two paragraphs, verbatim, after the brief. The guidance already says to use
  a budget "if supplied with the execution". With a budget, the hard stop defaults to the budget plus
  20 minutes. The paper has no medium bare runs, and its short bare runs used
  a later wrapper that deferred the film render to a separate step; here every brief gets the same
  wrapper, and the agent renders its own film as the guidance requires; the harness renders one only
  when the agent delivered none.

The bare agent runs inside `unshare -Ur bwrap`. System directories, the Blender environment, and the
agent binary are mounted read-only. `/workspace` is its only writable project path. Its private home
holds only its credentials, with no skills and no `AGENTS.md`. The GPU is passed through. Codex runs
`codex exec` with the paper's flags (user config, rules, skills, plugins, apps, native sub-agents, and
web search off), with approvals as in DyCraft's own Codex launch (`--approve-for-me`) rather than the
paper's bypass. Claude Code runs `claude -p` with sub-agent and web tools disallowed.

## Comparing the Arms

By their films, and by the paper's two physical metrics. `benchmark/stage.py` stages a bare agent's
`.blend` as an animation bundle like a DyCraft run's: packed so it is portable, evaluated frame by frame
(`animation/blender/worker.py`) so the viewer plays it in 3D, and given its film.

`python -m dycraft.benchmark.score <out>` computes the paper's Table 2 metrics for every arm: the
penetration rate (frames with a moving mesh penetrating another) and the floating rate (moving
body-frames with no support path). It writes each arm's `benchmark/score.json`; the definitions, and
the check that it reproduces the paper's numbers, are in [score/README.md](score/README.md). It takes
any `.blend` too, so another method's films can be scored the same way.

The report shows each brief's films side by side with each arm's outcome (the DyCraft arm's own
verdict from `finish_result.json`), the film's length, where a bare film came from, the wall time,
the two rates where they were scored with each arm's mean over its scored briefs, and the agent-hours
per minute of film (main session / with sub-agents, `benchmark/cost.py`).

## Requirements

- Linux with `bwrap` and user namespaces (as DyCraft's Blender layer needs; where only bwrap may
  create them, as on stock Ubuntu 24.04, it is used directly).
- A Blender Python with `bpy` (`python -m dycraft.install blender`), and `ffmpeg`.
- The `codex` and/or `claude` CLI, logged in (credentials in `~/.codex` / `~/.claude`, or
  `--agent-config DIR`, or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). `--agent-bin` names the binary
  if it is not on `PATH`. npm installs `codex` as a Node script; the runner runs the native binary the
  package ships instead, since the sandbox has no Node outside the system directories.
- For B01–B08 (the Unitree G1 briefs) and `--g1`: the robot description at `assets/robots/g1_description/`
  (the URDF `g1_29dof_mode_15_with_dex1_1.urdf`, the meshes it references and its LICENSE), which
  is not in git. The DyCraft arm takes the G1 from the asset library instead.

`python -m dycraft.benchmark.bare --briefs B13 --agent codex --out /tmp/check --check` builds the
sandbox and runs its preflight and a tool probe without starting an agent.
