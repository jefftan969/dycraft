# Benchmark Guide

The paper's benchmark has 24 long-horizon animation briefs. Each brief is one paragraph that says who acts and why,
what the scene contains at the start, a dozen or more actions in order, and how the scene ends. Most briefs include a
complication the performer has to handle, such as a shelf that rocks, a parcel that skews, or a name called twice. The
duration, frame rate, camera, and style are left to the system that takes the brief.

We wrote the briefs in this format for the paper, but DyCraft does not need it. Your own prompts can be shorter and
less prescribed, and the prompt rewriters fill in the details of the scene and the motion before the agent starts.

| Briefs | Performer |
|---|---|
| B01–B08 | The Unitree G1 humanoid robot, from the asset library |
| B09–B16 | People and animals |
| B17–B24 | Machines |

The briefs are in `agent/prompts/animation_24_briefs.md`, with shorter and medium-length versions of most of them.
B01–B08 need the Unitree G1 from the [asset library](usage.md#asset-library), for both DyCraft and the bare agent.
`python -m dycraft.install library --names g1_with_dex1_v3` installs only the G1. B20 and B21 involve liquid, so they
need the official Blender binary (see the [install guide](install.md#5-install-blender)).

## Comparing DyCraft With a Bare Agent

The paper compares DyCraft with a bare coding agent that has Blender and no DyCraft tools. These commands run both
agents on the same briefs and write their results to one directory. Then they score every film and build a report that
shows each brief's films side by side:

```bash
python -m dycraft.benchmark.run    --briefs short --agent codex --out runs/bench_short   # DyCraft
python -m dycraft.benchmark.bare   --briefs short --agent codex --out runs/bench_short   # the bare agent
python -m dycraft.benchmark.score  runs/bench_short
python -m dycraft.benchmark.report runs/bench_short                # writes runs/bench_short/report.html
```

`--agent claude` runs Claude Code instead. `--briefs` takes a set or a list of brief ids:

| `--briefs` | Briefs | Budget per brief |
|---|---|---|
| `short` | Short versions of B01–B12 and B15, the others as written | 8 hours |
| `medium` | Medium versions of most of B01–B12, the others as written | 24 hours |
| `long` | B01–B24 as written | 24 hours |
| `B01S,B13` | Those briefs | The budget of their set |
| `--prompt TEXT` | One brief of your own (text, or a `.md` file) | 8 hours |

DyCraft is told its budget and a film length, and is stopped 20 minutes after the budget. The bare agent gets the
paper's bare-agent prompt and is stopped after 72 hours. It runs in a sandbox with Blender and its own login, and
nothing else. `--budget-hours H` and `--film-seconds S` give both the same budget and film length. Start with the
short set, since even it takes many hours of agent time per brief. Some of the long briefs run for much longer: with
GPT-6-Astra, several of the paper's films took more than 12 hours. `--parallel N` runs N briefs at once, and
`--dry-run` prints the commands without running them.

The report also lists each agent's outcome, run time, scores, and agent-hours per minute of film.
`python -m dycraft.viewer --runs-dir runs/bench_short` plays both in 3D. [benchmark/README.md](../benchmark/README.md)
lists the output files and every option.

## Scoring

`python -m dycraft.benchmark.score` computes the two physical metrics of the paper's Table 2:

- penetration rate, the share of frames in which a moving mesh intersects another mesh;
- floating rate, the share of frames in which a moving body has no support.

It accepts a DyCraft run, a bare agent's run, a benchmark directory, or any `.blend`, so films from other methods can be
scored the same way. A film takes from a few CPU-minutes to about three CPU-hours. On the paper's 48 films, it
reproduces Table 2. [benchmark/score/README.md](../benchmark/score/README.md) has the definitions.

## The Paper's Films

The 24 films DyCraft made for the paper are in the
[dycraft-films dataset](https://huggingface.co/datasets/jefftan969/dycraft-films). Each film comes as a bundle with the
video, the Blender project that renders it, the plan, and the brief. To download them:

```bash
python -m dycraft.install films                    # all 24 films, 13.6 GB
python -m dycraft.install films --brief B05,B13    # only those briefs
```

Each film goes in `films/` in the clone, or in `$DYCRAFT_HOME` if it is set, under the name it has in the dataset, such
as `films/B05_g1_carries_panel_through_doorway/`. A film takes from 90 MB to 2.6 GB. Running the command again
downloads only what changed, and `--revision TAG` pins a version of the dataset.

`python -m dycraft.benchmark.score` scores the films like any other film. To play them in the viewer:

```bash
python -m dycraft.viewer --runs-dir films
```

The first time you open a film, the viewer takes a moment to prepare it, about a minute for the larger films. Later
openings are immediate. The viewer keeps what it prepares in `.viewer_films/` in the clone or `$DYCRAFT_HOME`, about as
much space as the film itself.

To re-render a film so that it matches the published video, render with OptiX, which needs an NVIDIA RTX GPU. On a CPU
or with CUDA, B09 and B22 show black surfaces where two faces lie in the same plane, on B09's right wall and door and
on B22's canal brick walls.

The films are CC BY 4.0, except for the third-party characters and motion capture they contain. B11, B13, and B15
include material for non-commercial use only.

The paper ([arXiv:2610.40754](https://arxiv.org/abs/2610.40754)) reports the benchmark results, and the
[project page](https://dy-craft.github.io) shows the films of DyCraft and the baselines for each brief.

## Differences From the Paper's Runs

This release differs from the code that made the paper's films in ways that can change what you get from a brief:

- Agents can use motion capture. `python -m dycraft.install mocap [--dog]` downloads two libraries, CMU (people) and
  MANN (dogs), and agents have retargeting helpers for both.
- Runs stop after 12 hours by default. Several of the paper's films took longer. For a long brief, pass a larger
  `--max-hours`, or 0 for no limit. The benchmark runner sets the limit from each set's budget. A long run can also be
  continued with several `--resume` calls in a row, on a different account each time one reaches its usage limit:
  `python -m dycraft.cli --resume runs/<run_id> --backend codex2`, then `--backend codex3`, and so on.
- The short briefs are shorter, and the budgets differ. The short versions were cut to fit an 8-hour run, keeping each
  brief's point and its complication. The default budgets are 8 hours for the short set and 24 hours for the medium and
  long sets. The paper's runs were told 8 hours.
- Agents cannot leave their sandbox. Codex cannot approve a command outside its sandbox, and a Codex launch is refused
  where the sandbox does not work. Claude Code's option to run a command outside its sandbox is turned off. Commands
  that the paper's agents sometimes ran outside the sandbox now fail, and the agent works around them.
- The tools and the agents' instructions were fixed and tightened. Some checks that missed problems in the paper's
  runs now catch them, and the motion instructions ask agents to check more of the body. The same brief will not
  reproduce the paper's film.
- One character from the paper's films did not pass review: the grip in B11's crew, whose mesh has a defect. B11's
  briefs take the crew from the library, so an agent casts someone else in that role.
- Some characters have known limits, listed as caveats in the library's catalog. For example, the crow's wings fold
  cleanly only through intermediate poses.

## Cost

We report cost as agent-hours per minute of film, the hours the agent sessions were active divided by the length of
the film. Over the paper's 24 films, made with GPT-6-Astra in Codex, the median was 11 agent-hours per minute of film
for the main agent, and 15 with its helper agents. Runs that finished within one 8-hour budget needed about 8 and 11.
Other models may take more or less time.
`python -m dycraft.benchmark.cost <run_dir>` reports it for any run.

