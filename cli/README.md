# `dycraft.cli` — The Launcher

`python -m dycraft.cli` starts one agent run: it prepares a run directory, starts the harness daemon the agent talks
to, and runs a coding-agent CLI (Claude Code or Codex) on the task.

```
python -m dycraft.cli --backend claude --layer animation --prompt brief.md
python -m dycraft.cli --backend codex  --layer object --prompt "A desk lamp with a two-link arm" --model gpt-6-astra
python -m dycraft.cli --backend claude2 --layer scene --prompt scene.md --max-hours 4
```

## Layers

| `--layer` | What the agent does |
|---|---|
| `animation` | The film agent: plans, storyboards, animates, and renders a complete sequence in Blender |
| `scene` | Builds a scene, authoring its own objects and arranging them |
| `object` | Builds one articulated object |
| `plain` | A one-off text answer; used internally by `review()` and the brief rewriters |

## Backends and Accounts

- `claude` runs Claude Code, `codex` runs the Codex CLI. Both use their own login; DyCraft reads no API keys. Export
  anything else a CLI needs in the shell you launch from.
- `claude1`…`claude9` select the account in `~/.claude1`…`~/.claude9`, and `codex1`…`codex9` the one in
  `~/.codex1`…`~/.codex9`. Plain `claude` uses `CLAUDE_CONFIG_DIR` from the shell, or `~/.claude`. Every sub-agent
  a run starts uses the same account.
- Claude agents run in Claude Code's sandbox: shell commands write only inside the run directory and reach only
  `localhost`, file edits are allowed only inside the run directory, and hooks are off. Codex agents run in Codex's
  `workspace-write` sandbox. Where the agent's sandbox cannot start on the machine, the launch is refused (exit 2)
  unless `--allow-unsandboxed` is given: the agent then runs without it, so its commands can read and write anywhere
  you can, and the launch says so, as do `meta.json` (`unsandboxed`) and the run's README. Where the sandbox works,
  the flag leaves it on. `--resume` does not carry the flag over: pass it again.

## Flags You Will Use

- `--prompt`: the task. A path to an existing `.md` file is read as the task.
- `--model`, `--reasoning-effort` (default `xhigh`): passed to the agent CLI.
- `--max-hours` (default 12; 0 for no limit): stops the agent and every sub-agent that long after the agent starts;
  the launcher then exits 124 and `meta.json` records `time_limit`.
- `--runs-dir`: where runs are written (default `runs/` under the checkout, or under the launch directory for an
  installed package; `$DYCRAFT_HOME` overrides both).
- Film runs: `--scene-brief FILE` / `--motion-brief FILE` supply the scene and motion descriptions; without them a
  rewriter generates each one (`--no-scene-brief`, `--no-motion-brief` turn that off, `--on-rewrite-failure` says
  what to do when a rewrite fails). `--preload-asset NAME=BUNDLE` stages geometry the run may place but not edit.
- `--render-backend {open3d,blender}`: the renderer for object and scene stills (default `open3d`, or
  `$DYCRAFT_RENDER_BACKEND`).

`python -m dycraft.cli --help` lists the rest, grouped.

## What a Launch Does

1. Creates the run directory and checks the backend can start (a Codex launch refuses here if its sandbox does not
   work).
2. For a film run, generates the scene and motion descriptions unless given or turned off.
3. Writes `meta.json`, stages the harness client (`harness.py`) and the repository's skills (`.claude/skills/`,
   `.agents/skills/`), and writes the system prompt (`system_prompt.md`).
4. Starts the `--max-hours` clock, the harness daemon (it writes `.harness.port` and logs to `.daemon.log`) and a
   watcher that stops the agent when `.terminate` appears.
5. Runs the agent CLI. When it exits, the daemon records the outcome in `meta.json`.

A run directory then holds `meta.json` (task, backend, model, outcome, time limit), `trace.jsonl` (every harness
operation), `agent_transcript.jsonl` (the agent CLI's own session), the agent's sources and results, and the renders
and reports it made.

## Code Map

- `cli.py`: argument parsing and `main()`, the launch sequence above, and `build_env_dict` (the environment the
  daemon and the agent inherit).
- `backends/`: the agent CLIs behind one interface. `_base.py` holds `Backend` and `LaunchPlan`, `claude.py`
  `ClaudeBackend` and `codex.py` `CodexBackend` (each with its `claudeN`/`codexN` selectors), and `_process.py`
  runs and stops the agent CLI process. A backend implements `execute(plan)` and sets `agent_cmdline_marker`, a
  substring of its agent's command line that the daemon's watchdog uses to find the agent process after the CLI
  forks; `preflight`, `prepare_shared_env` (what the daemon must share with the agent, such as the account) and
  `credential_meta` are optional.
- The harness itself is in `dycraft.agent`: `LAYERS` maps each layer to its daemon module and harness stubs,
  `agent_runner.py` composes prompts and footers, and `agent/server.py` is the daemon. Each layer's staged
  `harness.py` exposes exactly the operations its daemon registers.
