"""Single entrypoint for the DyCraft agent harness.

    python -m dycraft.cli --backend claude --layer object    --prompt "<prompt>" --model opus
    python -m dycraft.cli --backend codex  --layer animation --prompt brief.md

Launch sequence (`main`; `--resume` hands the whole launch to `resume.main` instead):
    1. `_settle_args`           check the arguments, read a prompt file, pin the renderer
    2. `ensure_run_dir`         (`runs/<id>/src/` for a launch, `run_layout`; one folder for a sub-run)
    3. `_environment_preflight`, then `backend.preflight`
                                the environment checks, then the backend's own (a Codex launch
                                refuses here when Codex's sandbox does not work)
    4. `_stage_prior_work`, `_stage_preloaded_assets`, `_stage_entries`
                                --prior-{scene,animation}, --preload-asset bundles, and
                                scene.txt / motion.txt (the rewriters run here)
    5. write the initial meta.json (`_launch_meta`, with the CAD preflight; for an animation run,
       `_decide_ik` records whether `h.ik` is available)
    6. `stage_harness_client`   (agent assembles _harness_core + per-layer stubs)
       / `stage_skills`         (this checkout's .claude/.agents skill pages, so the
                                 prompt's relative path resolves inside the run dir)
       and the system prompt (`_system_prompt`); start the `--max-hours` clock
    7. `_start_agent`: compose the user-task footer, build the env dict (DYCRAFT_*,
       EGL/Mesa fixes), start the harness daemon (LAYERS[layer].server_module via
       `python -m … serve <run_dir>`) and the self-kill watcher (touches
       .terminate → SIGTERM agent), then `execute_with_time_limit` →
       `backend.execute(plan)` runs the agent CLI as the launcher's child

cli/ owns the launcher concerns: argparse, process spawn (daemon,
watcher), env wiring, exec handoff. Agent-domain contracts (run-dir layout,
catalog schema, harness client assembly, per-layer daemon module) live in
`dycraft.agent`; this module imports them via `LAYERS`, `ensure_run_dir`,
`stage_harness_client`, and
`stage_skills`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .. import _home
from .._layers import LAYERS as LAYER_NAMES
from ..agent import (LAYERS,
                     stage_harness_client)
from ..agent.agent_runner import (
    TASK_FILE_ENV,
    compose_footer,
    compose_system_prompt,
    stage_long_task,
    stage_system_prompt,
    ensure_run_dir,
)
from ..agent.agent_session import record_launch_failure, write_run_meta
# Imported rather than re-hardcoded: the default renderer and the name of its
# env var have exactly one home, so `--render-backend`'s help text, this
# module's pin and `render.default_backend()` cannot drift apart.
from ..render import DEFAULT_RENDER_BACKEND, RENDER_BACKEND_ENV
from ..run_layout import INTERNAL, publish, top_of
from .backends import ALLOW_UNSANDBOXED_ENV, BACKENDS, Backend, LaunchPlan, _process


#: `--max-hours` default: longer than any session of the paper's runs that worked to a budget, so an 8-hour run is
#: never cut, and short enough that a forgotten or wedged run stops within the day.
DEFAULT_MAX_HOURS = 12.0
#: The launcher's exit status when `--max-hours` stopped the run (as `timeout`'s).
EXIT_TIME_LIMIT = 124
#: The run's deadline in epoch seconds, exported to the daemon and the agent, so every sub-agent the daemon starts
#: stops by its parent's deadline.
DEADLINE_ENV = "DYCRAFT_DEADLINE"


_HERE = Path(__file__).resolve().parent
_DYCRAFT_DIR = _HERE.parent

#: How long a live daemon may take to bind. Long, because on a loaded host a
#: dispatched child's daemon can take over a minute to bind; a daemon that dies
#: is still caught at once, by its pid.
HARNESS_DAEMON_TIMEOUT_S = 300
HARNESS_DAEMON_POLL_S    = 0.2


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


# Renderer names as they show up in a natural-language prompt. Matched on word
# boundaries so "blender" in "blenderized" does not count; `cycles` is
# Blender's engine name and is what a prompt asking for photoreal output tends
# to say.
_RENDERER_IN_PROMPT = {
    "blender": re.compile(r"\b(?:blender|cycles)\b", re.IGNORECASE),
}

#: The agent CLI's reasoning effort unless `--reasoning-effort` says otherwise.
#: The benchmark's bare arm takes the same default, so both arms run alike.
DEFAULT_REASONING_EFFORT = "xhigh"


def warn_render_backend_mismatch(prompt: str, *, effective: str,
                                 flag_given: bool) -> str | None:
    """Warn once when the prompt names a renderer the run will not use.

    Returns the warning text (for tests), or None when there is nothing to say.
    Silent when `--render-backend` was passed explicitly: the operator has
    already made the choice and does not need second-guessing by a keyword
    match. Silent, too, when the prompt names the renderer that is actually
    active — the common case once the default is set.
    """
    if flag_given or not prompt:
        return None
    for backend, pattern in _RENDERER_IN_PROMPT.items():
        if backend == effective or not pattern.search(prompt):
            continue
        message = (
            f"[cli] WARNING: the prompt asks for {backend!r} but this run "
            f"starts with {effective!r}. Naming a renderer in prose does not "
            f"set it. Pass `--render-backend {backend}` "
            f"for this run, or `export {RENDER_BACKEND_ENV}={backend}` to make "
            f"it your default. Object and scene agents can also call "
            f"h.set_render_backend({backend!r}). Continuing with {effective!r}."
        )
        print(message, file=sys.stderr)
        return message
    return None


def _rewrite_failed(staged_name: str, kind: str, exc: BaseException,
                    sub_run: Path, started: float,
                    file_flag: str, off_flag: str, policy: str = "ask") -> None:
    """Report a rewriter that produced no entry, then continue or stop.

    Neither fixed answer is right. Always stopping throws away a long run over
    one rewriter that hit its bound; always continuing hides the opposite
    failure, a whole batch of rewriters stuck on a startup screen nobody can
    answer, every run then working from the bare brief without anyone being
    asked.

    So it asks. `policy`:

      ask       the default. With a terminal on stdin, a y/N question; only an
                explicit yes continues. With none (a headless `RUN.sh`, a
                redirected launch) there is nobody to ask, and it stops.
      continue  run on from the brief alone, loudly, as a degraded run.
      abort     stop.

    Stopping raises SystemExit(2) before the daemon starts and before
    `meta.json` exists, so a launcher reads the run dir as never started and
    cold-starts it next time. The failed attempt stays in `sub_run`.
    """
    print(f"[cli] WARNING: generating {staged_name} failed after "
          f"{time.time() - started:.0f}s ({type(exc).__name__}: {exc}).",
          file=sys.stderr)
    choices = (f"Pass {file_flag} FILE to supply an entry, {off_flag} to run "
               f"without one deliberately, or --on-rewrite-failure continue.")
    if policy == "ask":
        if sys.stdin is not None and sys.stdin.isatty():
            print(f"[cli] Continue WITHOUT a {kind} entry? The arm would work "
                  f"from the brief alone. [y/N] ", end="", file=sys.stderr,
                  flush=True)
            try:
                answer = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt, OSError):
                answer = ""
            policy = "continue" if answer in ("y", "yes") else "abort"
        else:
            print(f"[cli] no terminal to ask whether to continue without "
                  f"{staged_name} (stdin is not a tty); stopping.",
                  file=sys.stderr)
            policy = "abort"
    if policy == "abort":
        print(f"[cli] stopped before the agent started: no {staged_name}. The "
              f"rewriter's attempt is in {sub_run}. {choices} Move the run "
              f"directory aside first for a clean retry.", file=sys.stderr)
        raise SystemExit(2)
    print(f"[cli] WARNING: continuing WITHOUT a {kind} entry -- the prompt "
          f"carries no pointer to {staged_name} and the arm works from the "
          f"brief alone. This is a degraded run, not a failed one; it is "
          f"recorded here and in {sub_run}.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """The launcher's arguments. Internal flags (set by the launcher for the
    sub-runs it starts) are accepted but left out of `--help`."""
    parser = argparse.ArgumentParser(
        prog="python -m dycraft.cli",
        description="Run one dycraft agent: a coding-agent CLI (Claude Code or Codex) on a layer's task.",
        epilog="To carry a stopped run on in its own agent session, or revise a delivered one: "
               "python -m dycraft.cli --resume RUN_DIR [--prompt TEXT|FILE] [--max-hours H] "
               "(python -m dycraft.cli --resume x --help).",
    )
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS.keys()), metavar="BACKEND",
                        help="claude (Claude Code) or codex (Codex CLI). claude1..claude9 and codex1..codex9 "
                             "run on the account in ~/.claudeN or ~/.codexN.")
    parser.add_argument("--layer", required=True, choices=LAYER_NAMES, metavar="LAYER",
                        help="animation (the film agent: plans, builds the place, storyboards, animates "
                             "and delivers a film), scene (builds a scene, authoring its own objects), "
                             "object (one articulated object) or plain (a one-off text answer).")
    parser.add_argument("--prompt", required=True,
                        help="The task: text, or the path to a file. A text file (.md, .txt, .json, ...) is read as "
                             "the task; any other file (a PDF, an image, a model) is copied into the run's inputs/ "
                             "and is the brief the agent opens.")

    agent = parser.add_argument_group("agent")
    agent.add_argument("--model", default=None,
                       help="The agent's model: passed to `claude --model` or `codex -m`.")
    agent.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                       help="Passed to `claude --effort` (low|medium|high|xhigh|max) or as codex's "
                            f"`model_reasoning_effort`. Default: {DEFAULT_REASONING_EFFORT}.")

    limits = parser.add_argument_group("time")
    limits.add_argument("--max-hours", type=float, default=None,
                        help="Stop the run this many hours after the agent session starts "
                             f"(default {DEFAULT_MAX_HOURS:g}; 0: no limit). The agent CLI and every "
                             "sub-agent are stopped, meta.json records it under `time_limit`, and the "
                             f"launcher exits {EXIT_TIME_LIMIT}.")
    limits.add_argument("--advisory-hours", type=float, default=1.0,
                        help="Hours after which the agent's replies report how long it has run (the "
                             "animation agent's carry its budget). Advisory: nothing is stopped. "
                             "0 disables. Default: 1.")

    where = parser.add_argument_group("run directory")
    where.add_argument("--runs-dir", default=None,
                       help="Where runs are written. Default: runs/ under the checkout, or under the "
                            "launch directory for an installed package; $DYCRAFT_HOME overrides both.")
    where.add_argument("--run-id", default=None,
                       help="The run directory's name. Default: a timestamp and the backend.")
    where.add_argument("--exp-name", default=None,
                       help="Appended to the default run directory name.")

    staged = parser.add_argument_group("scene and animation runs")
    staged.add_argument("--preload-asset", action="append", default=[], metavar="NAME=BUNDLE",
                        help="Geometry the run may place but not edit, reachable from scene.py as "
                             "`load_asset(\"NAME\")`. BUNDLE is a directory holding ir.json and the "
                             "assets/ it cites (what `python -m dycraft.agent.assets.import_urdf` writes, "
                             "or a committed object run). Repeatable. Not needed for the asset library "
                             "(assets/imported/), which the agent searches itself.")
    # Revision is internal: out of the docs, it refuses the published films, and it is not validated end to end.
    staged.add_argument("--prior-scene", default=None, metavar="SCENE_PY", help=argparse.SUPPRESS)

    film = parser.add_argument_group("animation runs")
    film.add_argument("--scene-brief", default=None, metavar="FILE",
                      help="A description of the place the task happens in, staged as `scene.txt` and "
                           "pointed at from the task. Without it a rewriter generates one.")
    film.add_argument("--no-scene-brief", dest="scene_brief_auto", action="store_false", default=True,
                      help="Do not generate a scene description.")
    film.add_argument("--motion-brief", default=None, metavar="FILE",
                      help="How the performance is physically carried out (technique, the states "
                           "between the brief's actions, the spatial relationships they need), staged "
                           "as `motion.txt`. Without it a rewriter generates one from the scene "
                           "description.")
    film.add_argument("--no-motion-brief", dest="motion_brief_auto", action="store_false", default=True,
                      help="Do not generate a motion description.")
    film.add_argument("--on-rewrite-failure", default="ask", choices=("ask", "continue", "abort"),
                      help="When a rewriter produces nothing: `ask` (default) asks y/N on the terminal "
                           "and stops without one, `abort` stops, `continue` runs from the brief alone.")
    film.add_argument("--prior-animation", default=None, metavar="PROJECT", help=argparse.SUPPRESS)

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument("--render-backend", default=None, choices=("open3d", "blender"),
                          help="Renderer for object and scene stills (snapshot, review images, "
                               f"final_render): {DEFAULT_RENDER_BACKEND!r} by default, or "
                               "$DYCRAFT_RENDER_BACKEND. `blender` renders Cycles and needs the "
                               "Blender environment; `open3d` does not evaluate @shader graphs.")
    advanced.add_argument("--allow-unsandboxed", "--allow-unsandboxed-codex", dest="allow_unsandboxed",
                          action="store_true",
                          help="Start the agent even where its CLI's sandbox (Codex's, or Claude Code's Bash "
                               "sandbox) cannot start on this machine, without that sandbox: its commands can "
                               "then read and write anywhere you can. By default such a launch is refused. Where "
                               "the sandbox works, it stays on. Not kept by --resume: pass it again.")

    # Set by the launcher for the sub-runs it starts.
    parser.add_argument("--contribution-role", choices=("motion",), help=argparse.SUPPRESS)
    parser.add_argument("--parent-run", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    """Launch one agent run, in the order the module docstring lists; `--resume` carries a stopped one on."""
    argv = sys.argv[1:] if argv is None else list(argv)
    # Before any check or child: the tools `install tools` set up are where every later lookup expects them.
    _home.tools_on_path()
    if any(a == "--resume" or a.startswith("--resume=") for a in argv):
        from .resume import main as resume
        return resume(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _settle_args(parser, args)
    backend: Backend = BACKENDS[args.backend]()

    runs_dir = (Path(args.runs_dir).resolve() if args.runs_dir
                else _home.default_runs_dir())
    print(f"[cli] runs dir: {runs_dir}", file=sys.stderr)
    effective_run_id = args.run_id or _gen_run_id(args.backend, args.exp_name)
    # A run someone launches is laid out for a reader: its records and the agent's working folder in `src/`, the
    # deliverables and a README at the top (`run_layout`). A sub-run, whose reader is its parent, stays one folder:
    # a contributor names its parent, and a brief rewriter, launched before its parent has any record, is filed in
    # the parent's `sub_runs/`.
    from ..agent.oneoff import SUB_RUNS_DIR
    sub_run = bool(args.parent_run) or runs_dir.name == SUB_RUNS_DIR
    if not sub_run:
        # Refused before its folder exists, so a launch that cannot start leaves no run behind. A sub-run's are
        # refused inside it instead, where its parent reads why (`_launch`).
        refusal = _environment_preflight(args, backend, None)
        if refusal is None and not backend.preflight_needs_run_dir:
            refusal = backend.preflight(None, args.layer)
        if refusal is not None:
            return refusal
    run_id, run_dir = ensure_run_dir(runs_dir, effective_run_id, reader=not sub_run)
    top = top_of(run_dir)
    if top != run_dir:
        print(f"[cli] run directory: {top}/ (the agent works in {INTERNAL}/)", file=sys.stderr)
    try:
        return _launch(parser, args, backend, runs_dir=runs_dir, run_id=run_id, run_dir=run_dir, sub_run=sub_run)
    finally:
        # Whatever ended the launch: the README says how, and the top level holds what was delivered.
        publish(run_dir)


def _launch(parser, args, backend, *, runs_dir: Path, run_id: str, run_dir: Path, sub_run: bool = True) -> int:
    """`main` once the run's folder exists."""
    _process.watch_parent(lambda pid: _parent_gone(pid, run_dir))
    # Before anything else is started for this launch, the rewriters included: a broken install is refused with the
    # environment check's own message (`_environment_preflight`), and a backend that cannot start its agent here
    # (`Backend.preflight`). A top-level launch ran those before making its folder (`main`), except a preflight
    # that needs the folder: Codex's sandbox probe runs in it.
    if sub_run:
        refusal = _environment_preflight(args, backend, run_dir)
        if refusal is not None:
            return refusal
    if sub_run or backend.preflight_needs_run_dir:
        refusal = backend.preflight(run_dir, args.layer)
        if refusal is not None:
            return refusal

    if args.prompt_file is not None:
        from . import prompt_files
        staged = args.prompt_staged = prompt_files.stage(args.prompt_file, run_dir)
        args.prompt = args.prompt_file.task(staged)
        # A rewriter works from its own sub-run, so it is pointed at the brief by its full path.
        args.rewriter_prompt = args.prompt_file.task(str(run_dir / staged), in_run_dir=False)
    _stage_prior_work(parser, args, run_dir)
    preloaded_assets = _stage_preloaded_assets(parser, args, run_dir)
    _stage_entries(parser, args, run_dir)

    extra_meta = _launch_meta(args, backend, preloaded_assets)
    if args.layer == "animation":
        extra_meta["ik"] = _decide_ik()
    write_run_meta(
        run_dir, layer=args.layer,
        runtime=args.backend,
        prompt=args.prompt, run_id=run_id, model=args.model,
        extra=extra_meta or None,
    )
    publish(run_dir)

    # Stage the harness client (and, through it, the revision's own skill
    # pages; see `agent.stage_skills`) and the system prompt.
    stage_harness_client(run_dir, args.layer)
    system_prompt = _system_prompt(args)
    stage_system_prompt(run_dir, system_prompt)

    # The clock starts here, as close to the agent's own start as the footer allows: after the rewriters, before the
    # daemon.
    deadline, time_limit = run_deadline(args.max_hours)
    if time_limit:
        _merge_meta(run_dir, lambda meta: meta.update(time_limit=time_limit))
    if deadline is not None and deadline <= time.time():
        print(f"[cli] the run's time limit ({time_limit['deadline']}) has already passed; "
              "not starting the agent", file=sys.stderr)
        _record_time_limit_reached(run_dir, time.time())
        return EXIT_TIME_LIMIT

    return _start_agent(args, backend, run_dir=run_dir, runs_dir=runs_dir,
                        run_id=run_id, system_prompt=system_prompt,
                        deadline=deadline)


def _parent_gone(pid: int, run_dir: Path) -> None:
    """A sub-run whose parent is gone has nobody to read its result: say so, stop its agent if it has one, and
    exit (`backends._process.watch_parent`)."""
    from ..agent.agent_session import record_agent_stop
    record_agent_stop(run_dir, reason="parent_gone", detail=f"parent pid {pid}")
    try:
        print(f"FATAL: this sub-run's parent (pid {pid}) is gone, so nothing will read its result; the sub-run is "
              "stopped.", file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass                       # a parent that read this output through a pipe took the pipe with it
    if not _process.stop_agent_cli(_process.EXIT_PARENT_GONE):
        os._exit(_process.EXIT_PARENT_GONE)


#: Launches without the environment checks (`_environment_preflight`), for a setup they misjudge.
SKIP_PREFLIGHT_ENV = "DYCRAFT_SKIP_PREFLIGHT"


def _environment_preflight(args: argparse.Namespace, backend: Backend, run_dir: Path) -> int | None:
    """Refuse a launch this environment cannot run, before anything is staged or a rewriter starts.

    Runs the checks `environment_test.LAUNCH_CHECKS` names for this layer and backend, the same checks and messages
    as `python -m dycraft.install check`, so an install that missed a step is told which, and how to fix it, before
    any agent spends usage. Exit status 2, recorded as a launch failure. `DYCRAFT_SKIP_PREFLIGHT=1` launches
    without them.
    """
    if os.environ.get(SKIP_PREFLIGHT_ENV) == "1":
        print(f"[cli] WARNING: {SKIP_PREFLIGHT_ENV}=1: launching without the environment checks.",
              file=sys.stderr, flush=True)
        return None
    from .. import environment_test as E
    started = time.monotonic()
    print(f"[cli] checking the environment for a {args.layer} run on {args.backend} ...",
          file=sys.stderr, flush=True)
    results = E.launch_preflight(args.layer, backend.name)
    refusals = E.launch_refusals(results, backend.name)
    for check, result in results:
        # A sandbox that cannot start does not refuse with --allow-unsandboxed (`launch_refusals`), and is said here.
        if (result.status in (E.WARN, E.DEGRADED) or result.data.get("sandbox")) and (check, result) not in refusals:
            print(f"[cli] WARNING: {check.title}: {result.summary}", file=sys.stderr)
    if not refusals:
        print(f"[cli] environment ok ({len(results)} checks, {time.monotonic() - started:.1f} s)",
              file=sys.stderr, flush=True)
        return None
    for check, result in refusals:
        print(f"FATAL: {check.title}: {result.summary}", file=sys.stderr)
        if result.lost:
            print(f"  lost: {result.lost}", file=sys.stderr)
        for line in result.fix:
            print(f"  fix: {line}", file=sys.stderr)
    print(f"  Nothing was started. `python -m dycraft.install check` gives the full report; "
          f"{SKIP_PREFLIGHT_ENV}=1 launches without these checks.", file=sys.stderr, flush=True)
    if run_dir is None:
        return 2
    first, result = refusals[0]
    record_launch_failure(run_dir, reason="environment",
                          detail="; ".join(f"{check.id}: {result.summary}" for check, result in refusals),
                          message=f"FATAL: {first.title}: {result.summary}",
                          fix=[line for _, r in refusals for line in r.fix])
    return 2


def _settle_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the arguments, read a prompt file, and pin the environment
    decisions the arguments carry. A bad argument exits through `parser.error`."""
    if args.max_hours is not None and args.max_hours < 0:
        parser.error("--max-hours must be 0 (no limit) or a positive number of hours")
    if args.allow_unsandboxed:
        # In the environment, so the daemon and every child it dispatches inherit the decision.
        os.environ[ALLOW_UNSANDBOXED_ENV] = "1"

    # Pin the renderer on this process's environment, which `build_env_dict`
    # copies to the daemon and every sub-run inherits.
    #
    # Precedence is flag > ambient env > `render.DEFAULT_RENDER_BACKEND`, which
    # is what lets an operator set a machine-wide default with
    # `export DYCRAFT_RENDER_BACKEND=blender` instead of remembering the flag on
    # every invocation.
    #
    # Deliberately pinned rather than left unset: the daemon and every sub-run
    # must agree on one value for the whole run, and `build_env_dict`
    # re-exports this key unconditionally.
    os.environ[RENDER_BACKEND_ENV] = (
        args.render_backend
        or os.environ.get(RENDER_BACKEND_ENV)
        or DEFAULT_RENDER_BACKEND)

    # `--prompt` is text, or a file of any type (`prompt_files`): a text file is read as the prompt, any other file is
    # the brief the agent opens itself, staged into the run once its directory exists (`main`).
    from . import prompt_files
    text, brief, note, error = prompt_files.read(args.prompt)
    if error:
        parser.error(error)
    args.prompt_file = brief
    args.prompt = text if brief is None else brief.task(brief.staged_name())
    if note:
        print(note, file=sys.stderr)

    # Refused on every layer: each one hands the prompt to its agent as the
    # task, and none has a meaningful empty-task mode. It is the last line of
    # defense when a launcher runs `--prompt "$(cat prompt.txt)"` on a checkout
    # without the file: bash substitutes "", and the run would last hours on
    # no brief. Whitespace-only is the same prompt: `$(cat)` of a blank file.
    if not args.prompt.strip():
        parser.error("--prompt is empty. Every layer takes it as the agent's "
                     "task; if it came from `$(cat <file>)`, that file is "
                     "missing or blank on this checkout")

    # The prompt cannot select a renderer, and operators reasonably expect it
    # to: "render the final result in Blender" in natural language runs on
    # Open3D unless `--render-backend` is passed, and the agent burns steps
    # working around neutral-gray shader previews. Warn rather than act: a
    # keyword match must not silently take over a run-wide setting that costs
    # real render time, and a prompt can mention a renderer in passing.
    warn_render_backend_mismatch(
        args.prompt, effective=os.environ[RENDER_BACKEND_ENV],
        flag_given=args.render_backend is not None)

    if args.preload_asset and args.layer not in ("scene", "animation"):
        # The layers that build what they place; `--preload-asset` is how they
        # are handed geometry whose value is being the real thing.
        parser.error("--preload-asset is only valid with --layer=scene or "
                     "--layer=animation")
    if args.contribution_role and args.layer != "animation":
        parser.error("--contribution-role requires --layer=animation")
    if args.prior_scene and args.layer not in ("scene", "animation"):
        parser.error("--prior-scene is only valid with --layer=scene or "
                     "--layer=animation")
    if args.prior_animation and args.layer != "animation":
        parser.error("--prior-animation is only valid with --layer=animation")


def _stage_prior_work(parser: argparse.ArgumentParser, args: argparse.Namespace,
                      run_dir: Path) -> None:
    """Pre-stage `--prior-scene` / `--prior-animation` for update/refine mode."""
    # Pre-stage source files for update/refine mode. The harness reads
    # scene.py from run_dir like any other file; this is purely a
    # CLI-stage copy. Done BEFORE write_run_meta so the file is present
    # when the agent boots.
    if args.prior_scene:
        src = Path(args.prior_scene).resolve()
        if not src.exists():
            parser.error(f"--prior-scene path does not exist: {src}")
        if src.is_dir():
            # A run folder, in either layout: its scene source, where a scene run or a film writes it.
            from ..run_layout import internal_dir
            inner = internal_dir(src)
            src = next((p for p in (inner / "scene.py", inner / "design" / "scene.py") if p.is_file()), src)
            if src.is_dir():
                parser.error(f"--prior-scene: {args.prior_scene} holds no scene.py or design/scene.py")
        target = run_dir / ("design/scene.py" if args.layer == "animation" else "scene.py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src.read_text())
        print(f"[cli] pre-staged scene.py from {src} (refine mode)",
              file=sys.stderr)
    if args.prior_animation:
        src = Path(args.prior_animation).resolve()
        if not src.exists():
            parser.error(f"--prior-animation path does not exist: {src}")
        from ..animation.blender.bundle import stage_prior
        stage_prior(src, run_dir / "prior_animation")
        print(f"[cli] pre-staged native animation from {src}", file=sys.stderr)


def _stage_preloaded_assets(parser: argparse.ArgumentParser, args: argparse.Namespace,
                            run_dir: Path) -> dict:
    """Stage `--preload-asset` bundles; returns what was staged, by name."""
    # Scene and animation: stage external geometry as frozen definitions `scene.py` reaches
    # with `load_asset(...)`. Before the daemon starts, so the agent's first
    # read of `preloaded/manifest.json` already describes what it was given.
    #
    # And before the brief rewriters below, because a bad bundle path is
    # argument validation: rejecting it after them costs a scene rewrite plus
    # a motion rewrite — two live sub-agent calls and minutes of the run's
    # budget — to tell an operator that a path is misspelled. The rewriters
    # take `args.preload_asset`, the specs, not anything staged here.
    preloaded_assets: dict = {}
    if args.preload_asset:
        from ..agent.assets.preload_asset import (
            PreloadAssetError, stage_preloaded_assets)
        # `animation` authors under `design/` and rebuilds from a copy
        # of that tree, so its assets belong inside it; `scene` builds in the
        # run dir itself. See `stage_preloaded_assets`.
        dest = (run_dir / "design" if args.layer == "animation"
                else run_dir)
        try:
            preloaded_assets = stage_preloaded_assets(args.preload_asset,
                                                      run_dir, dest=dest)
        except PreloadAssetError as e:
            parser.error(str(e))
        for name, rec in sorted(preloaded_assets.items()):
            print(f"[cli] preloaded asset {name!r}: {rec['ir_hash']} from "
                  f"{rec['source']} ({rec['n_parts']} parts, "
                  f"{rec['n_joints']} joints, "
                  f"{len(rec['assets'])} files, size {rec['size'][0]:.3f} x "
                  f"{rec['size'][1]:.3f} x {rec['size'][2]:.3f} m, FROZEN)",
                  file=sys.stderr)
    return preloaded_assets


def _stage_entries(parser: argparse.ArgumentParser, args: argparse.Namespace,
                   run_dir: Path) -> None:
    """Stage the scene and motion entries (`scene.txt`, `motion.txt`) and point
    the task at each one that exists."""
    from ..agent import motion_rewrite as _mr
    from ..agent import scene_rewrite as _sr
    scene_txt = _stage_scene_entry(parser, args, run_dir)
    motion_txt = _stage_motion_entry(parser, args, run_dir, scene_txt)

    # Scene first, then motion: the order they are meant to be read in.
    for staged, pointer in ((scene_txt, _sr.POINTER), (motion_txt, _mr.POINTER)):
        if staged.is_file() and pointer not in args.prompt:
            args.prompt = args.prompt.rstrip() + "\n\n" + pointer


def _stage_scene_entry(parser: argparse.ArgumentParser, args: argparse.Namespace,
                       run_dir: Path) -> Path:
    """`scene.txt`: the `--scene-brief` file, or one the scene rewriter writes."""
    # Before the daemon starts, for the same reason the assets above are: the
    # agent's first read of its own brief should find the file that brief
    # points at already on disk.
    # Explicit file, or generated for a fresh authoring animation run. Either
    # way the run ends up with `scene.txt` and the prompt ends up pointing at
    # it, so the agent reads one thing whichever path produced it.
    from ..agent import oneoff as _oneoff
    from ..agent import scene_rewrite as _sr
    scene_txt = run_dir / _sr.STAGED_NAME
    if args.scene_brief:
        src = Path(args.scene_brief)
        if not src.is_file():
            parser.error(f"--scene-brief: no such file: {src}")
        shutil.copy(src, scene_txt)
        print(f"[cli] staged scene brief {src} -> {_sr.STAGED_NAME} "
              f"({len(src.read_text().split())} words)", file=sys.stderr)
    elif (args.scene_brief_auto
            and args.layer == "animation"
            # A contributor works inside its launcher's project, and a prior
            # animation is a place that exists: neither is a new place to
            # describe.
            and not args.contribution_role
            and not args.prior_animation
            # Re-entering a run dir must not regenerate what it already has.
            and not scene_txt.is_file()):
        # Announced, because this blocks the launch on a live sub-agent for
        # as long as its wall clock allows, and an operator watching a silent
        # terminal could not tell a rewrite in progress from a wedged start.
        # Before the `try`, so a failure is
        # attributable to the step that was announced.
        print(f"[cli] rewriting the brief into {_sr.STAGED_NAME} "
              f"(scene_rewrite sub-agent: backend={args.backend}, "
              f"model={_sr.MODEL_OVERRIDE or args.model or 'backend default'}, "
              f"effort={args.reasoning_effort or 'backend default'}, "
              f"timeout {_sr.DEFAULT_TIMEOUT_S:.0f}s) ...", file=sys.stderr)
        t_rewrite = time.time()
        entry = None
        try:
            # A one-off CLI sub-agent on this run's own backend, model,
            # account, and effort -- the same subscription the run itself
            # uses, and the same launch path the object/scene dispatches
            # ride. The sub-run lands in `<run_dir>/sub_runs/scene_rewrite/`
            # and is bounded by its own wall clock, so a wedged CLI costs
            # that bound rather than the whole run.
            entry = _sr.rewrite(getattr(args, "rewriter_prompt", None) or args.prompt, preload_specs=args.preload_asset,
                                backend=args.backend, run_dir=run_dir,
                                model=args.model,
                                reasoning_effort=args.reasoning_effort)
        except Exception as exc:                            # noqa: BLE001
            _rewrite_failed(_sr.STAGED_NAME, "scene", exc,
                            run_dir / _oneoff.SUB_RUNS_DIR / _sr.SUB_RUN,
                            t_rewrite, "--scene-brief", "--no-scene-brief",
                            args.on_rewrite_failure)
        if entry:
            scene_txt.write_text(entry + "\n")
            print(f"[cli] generated {_sr.STAGED_NAME} "
                  f"({len(entry.split())} words, "
                  f"{time.time() - t_rewrite:.0f}s)", file=sys.stderr)
    return scene_txt


def _stage_motion_entry(parser: argparse.ArgumentParser, args: argparse.Namespace,
                        run_dir: Path, scene_txt: Path) -> Path:
    """`motion.txt`: the `--motion-brief` file, or one the motion rewriter writes."""
    # The motion entry, the same two ways, and after the scene entry because
    # it is written against it: the performance it describes happens in that
    # place. Same guards and for the same reasons -- a staged scene, a
    # contributor and a prior animation are each not a fresh performance in a
    # new place. The rewriter is handed the brief before either pointer is
    # appended, as the scene rewriter is.
    from ..agent import oneoff as _oneoff
    from ..agent import motion_rewrite as _mr
    motion_txt = run_dir / _mr.STAGED_NAME
    if args.motion_brief:
        src = Path(args.motion_brief)
        if not src.is_file():
            parser.error(f"--motion-brief: no such file: {src}")
        shutil.copy(src, motion_txt)
        print(f"[cli] staged motion brief {src} -> {_mr.STAGED_NAME} "
              f"({len(src.read_text().split())} words)", file=sys.stderr)
    elif (args.motion_brief_auto
            and args.layer == "animation"
            and not args.contribution_role
            and not args.prior_animation
            and not motion_txt.is_file()):
        # Announced for the reason the scene rewrite above is.
        print(f"[cli] rewriting the brief into {_mr.STAGED_NAME} "
              f"(motion_rewrite sub-agent: backend={args.backend}, "
              f"model={_mr.MODEL_OVERRIDE or args.model or 'backend default'}, "
              f"effort={args.reasoning_effort or 'backend default'}, "
              f"timeout {_mr.DEFAULT_TIMEOUT_S:.0f}s) ...", file=sys.stderr)
        t_rewrite = time.time()
        entry = None
        # Sized only to a limit given on this launch: the default limit is a
        # safety stop, not a target, and an inherited deadline does not reach
        # here (children do not run the rewriter).
        time_limit_hours = args.max_hours if args.max_hours else None
        try:
            entry = _mr.rewrite(getattr(args, "rewriter_prompt", None) or args.prompt,
                                scene=(scene_txt.read_text()
                                       if scene_txt.is_file() else None),
                                preload_specs=args.preload_asset,
                                backend=args.backend, run_dir=run_dir,
                                model=args.model,
                                reasoning_effort=args.reasoning_effort,
                                time_limit_hours=time_limit_hours)
        except Exception as exc:                            # noqa: BLE001
            _rewrite_failed(_mr.STAGED_NAME, "motion", exc,
                            run_dir / _oneoff.SUB_RUNS_DIR / _mr.SUB_RUN,
                            t_rewrite, "--motion-brief", "--no-motion-brief",
                            args.on_rewrite_failure)
        if entry:
            motion_txt.write_text(entry + "\n")
            print(f"[cli] generated {_mr.STAGED_NAME} "
                  f"({len(entry.split())} words, "
                  f"{time.time() - t_rewrite:.0f}s)", file=sys.stderr)
    return motion_txt


def blender_sandbox_meta() -> dict:
    """`meta.json`'s `blender_sandbox`: how this machine confines the Blender worker (`runtime.sandbox_mode`), and
    why not with namespaces when it does not."""
    from ..animation.blender import runtime
    try:
        mode = runtime.sandbox_mode()
    except RuntimeError as exc:
        return {"mode": "unavailable", "reason": str(exc)}
    record = {"mode": mode}
    if mode != "namespaces":
        record["reason"] = runtime._namespaces_problem()
    if mode == "landlock":
        record["landlock_abi"] = runtime.landlock_abi()
    return record


def _launch_meta(args: argparse.Namespace, backend: Backend, preloaded_assets: dict) -> dict:
    """The launch's own keys for the initial `meta.json`."""
    extra_meta: dict = {}

    # Unconditional, unlike every other key here: which renderer drew a run's
    # pictures is a property of every run. Without it the absence would be
    # indistinguishable from `open3d`, and the only way to tell what a finished
    # run used would be forensic — Blender stamps `cycles.ViewLayer.samples`
    # into its PNG metadata and Open3D writes a bare PNG, which identifies a
    # Cycles run but can never positively identify a Filament one.
    #
    # Read back from the environment rather than from `args.render_backend`,
    # because the pin in `_settle_args` (`os.environ["DYCRAFT_RENDER_BACKEND"]`)
    # is what every later `RenderRequest` actually resolves against. Recording
    # `args` instead would write
    # `None` for a defaulted run and re-derive the default a third time.
    extra_meta["render_backend"] = os.environ["DYCRAFT_RENDER_BACKEND"]
    if args.contribution_role:
        extra_meta["contribution_role"] = args.contribution_role
    if args.layer == "animation":
        extra_meta["animation_authoring"] = "blender"
        extra_meta["blender_sandbox"] = blender_sandbox_meta()

    # Persist the value that will actually be forwarded to the agent CLI. The
    # viewer should not have to reconstruct launch flags from logs, and
    # recording it here also covers interrupted runs whose transcript was
    # never captured.
    if args.reasoning_effort:
        extra_meta["reasoning_effort"] = args.reasoning_effort
    # `--resume` restates the advisory budget less the time the agent used.
    extra_meta["advisory_hours"] = args.advisory_hours

    # Which account this run drives. Transcript capture needs it: the external
    # CLI files its own session under that home, and when the same run dir has
    # been launched twice under different accounts (a failed attempt, then a
    # relaunch) the only way to tell the real session from the leftover is to
    # know which account was ours. Recording it also makes the bundle readable
    # from a shell with a different — or no — CLAUDE_CONFIG_DIR/CODEX_HOME,
    # which is exactly the viewer's situation. See
    # agent/transcript_capture.py::find_claude_transcript.
    #
    # Asked of the backend rather than duck-typed here: probing for a method
    # name would file any backend that happened to define it under
    # `claude_config_dir`, mislabelling its home as a claude account.
    try:
        extra_meta.update(backend.credential_meta())
    except BaseException:          # noqa: BLE001 — never block a launch
        pass
    extra_meta.update(backend.launch_meta)

    # CAD preflight, for the layers that execute build123d source.
    if args.layer in ("object", "scene", "animation"):
        extra_meta["cad_interpreter"] = _cad_preflight(args.layer)

    if args.parent_run:
        extra_meta["parent_run_id"] = args.parent_run
    brief = getattr(args, "prompt_file", None)
    if brief is not None:
        extra_meta["prompt_source"] = {"path": str(brief.path), "type": brief.kind, "bytes": brief.size,
                                       "staged": args.prompt_staged}
    if args.prior_scene:
        extra_meta["prior_scene"] = str(Path(args.prior_scene).resolve())
    if args.prior_animation:
        extra_meta["prior_animation"] = str(
            Path(args.prior_animation).resolve())
    if preloaded_assets:
        # Which bundle arrived under which `load_asset` name is what makes the
        # run reproducible.
        extra_meta["preloaded_assets"] = {
            name: {"ir_hash": rec["ir_hash"], "source": rec["source"],
                   "object_name": rec["object_name"]}
            for name, rec in sorted(preloaded_assets.items())
        }
    return extra_meta


def _decide_ik() -> dict:
    """Whether this launch has `h.ik`, as `meta.json` records it under `ik`.

    Decided once, before the prompt and the staging read it, and pinned in the
    environment the daemon, the agent, and the sub-runs inherit
    (`animation.ik.ik.decide_for_launch`). Said on stderr when unavailable.
    """
    from ..animation.ik.ik import decide_for_launch, launch_line
    found = decide_for_launch()
    line = launch_line(found)
    if line:
        print(line, file=sys.stderr)
    return found


def _cad_preflight(layer: str) -> dict:
    """The CAD interpreter this launch will use, as recorded in `meta.json`,
    with a warning on stderr when it cannot run build source."""
    from ..agent.cad._b123d_version import REQUIRED_BUILD123D
    from ..agent.cad._cad_env import b123d_python, probe_build123d_version, probe_ocp
    # Without a working CAD interpreter these layers cannot run their
    # build source at all — the session refuses to start. The session's own
    # check is the hard gate; this is the earlier, friendlier one, in the
    # one place a human is watching.
    #
    # It asks the interpreter for its version rather than trusting that a
    # resolved path implies a usable environment — a stale `.b123d_python`
    # resolves fine and fails every compile. Recorded in meta.json as the
    # version itself, so a later reader can tell a healthy run from a coarse
    # one without re-deriving it.
    solid_python = b123d_python()
    found = probe_build123d_version(solid_python) if solid_python else None
    # Build source is handed `OCP` as a global, so an interpreter with
    # build123d and no OCP fails at namespace construction rather than on
    # some later geometry call. Asked here for the same reason the version
    # is: this is the one place a human is watching.
    has_ocp = probe_ocp(solid_python) if solid_python else False
    if solid_python is None:
        print(f"[cli] WARNING: no build123d interpreter found — the "
              f"{layer} layer runs build123d source and will refuse "
              f"to start. Point DYCRAFT_B123D_PYTHON (or .b123d_python) at "
              f"a Python with build123d installed.",
              file=sys.stderr)
    elif found is None:
        print(f"[cli] WARNING: {solid_python} resolved as the build123d "
              f"interpreter but has no importable build123d — every "
              f"compile will fail there. Point DYCRAFT_B123D_PYTHON (or "
              f".b123d_python) at a Python with "
              f"build123d=={REQUIRED_BUILD123D} installed.",
              file=sys.stderr)
    elif found != REQUIRED_BUILD123D:
        print(f"[cli] WARNING: {solid_python} carries build123d {found}, "
              f"but DyCraft is frozen at {REQUIRED_BUILD123D}. The worker "
              f"refuses a skewed interpreter, so every compile will fail "
              f"with a version mismatch.",
              file=sys.stderr)
    elif not has_ocp:
        print(f"[cli] WARNING: {solid_python} carries build123d {found} "
              f"but no importable OCP, the OCCT binding build123d is "
              f"built on. Build source is given `OCP` as a global, so "
              f"every compile will fail at namespace construction. Point "
              f"DYCRAFT_B123D_PYTHON (or .b123d_python) at the pinned "
              f"environment — see environment-b123d.yml.",
              file=sys.stderr)
    return {"path": solid_python, "build123d": found, "ocp": has_ocp}


def _system_prompt(args: argparse.Namespace) -> str:
    """The layer's prompt file, plus the motion-contribution section for a
    motion contributor."""
    system_prompt = compose_system_prompt(args.layer)
    if args.contribution_role == "motion":
        system_prompt += (
            "\n## This run is a motion contribution\n\n"
            "You are a private motion contributor working on a copy of the launcher's project. "
            "Your task names the clip you own; author inside it and leave every other clip alone, "
            "since the launcher takes only that clip. Without one, you own the whole film. Your "
            "storyboard gate and your checks cover only what you own. `finish()` checks and "
            "packages your project for the launcher to integrate; you do not render the film, "
            "and stage 5's delivery is the launcher's.\n")
    return system_prompt


def _start_agent(args: argparse.Namespace, backend: Backend, *, run_dir: Path,
                 runs_dir: Path, run_id: str, system_prompt: str,
                 deadline: float | None) -> int:
    """Compose the task and footer, build the env, start the harness daemon and
    the self-kill watcher, then run the agent CLI until it exits or the
    deadline passes."""
    def footer_for(task):
        return compose_footer(
            args.layer, task, run_dir,
            contribution_role=args.contribution_role,
        ) + time_limit_note(deadline, args.layer)
    # A task too long for one argument reaches the agent as a file, and the
    # daemon reads the same file (`user_prompt_from_env`); `meta.json` keeps
    # the task itself either way.
    agent_task, task_file = stage_long_task(run_dir, args.prompt, system_prompt, footer_for)
    footer = footer_for(agent_task)
    if task_file:
        print(f"[cli] the task is {len(args.prompt.encode())} bytes, too long for one "
              f"argument; the agent is pointed at {task_file}", file=sys.stderr)
    env = build_env_dict(
        agent_task,
        run_dir=run_dir, runs_dir=runs_dir, run_id=run_id,
        layer=args.layer, runtime=args.backend,
        advisory_hours=args.advisory_hours,
        render_backend=args.render_backend,
    )
    if task_file:
        env[TASK_FILE_ENV] = str(task_file)
    # `inf` when there is no limit, so a sub-agent follows its parent rather than taking the default.
    env[DEADLINE_ENV] = repr(deadline) if deadline is not None else "inf"
    # How long a harness call waits for its job before returning it unfinished
    # (`_harness_core.WAIT_S`), for a backend whose shell kills commands sooner.
    if backend.harness_wait_s:
        env.setdefault("DYCRAFT_HARNESS_WAIT_S", backend.harness_wait_s)
    # The daemon uses this to find the agent's host pid by argv-scanning /proc
    # (the launcher pid the daemon initially watches dies early when claude
    # internally forks; this marker lets the watchdog retarget). See
    # `agent/server.py::_find_agent_pid`.
    if backend.agent_cmdline_marker:
        env["DYCRAFT_AGENT_CMDLINE_MARKER"] = backend.agent_cmdline_marker
    # Let the backend pin what the daemon also needs (the account, so every
    # sub-run the daemon starts authenticates as the agent does). Must happen
    # before the daemon is spawned; `execute` re-applies the same resolution
    # to the agent's own copy.
    backend.prepare_shared_env(env)
    daemon_pid = start_harness_daemon(run_dir, env,
                         daemon_module=LAYERS[args.layer].server_module,
                         log_label=f"{args.backend}_{args.layer}")
    start_self_kill_watcher(run_dir, os.getpid(),
                            log_label=f"{args.backend}_{args.layer}")
    # Must follow the watcher: it is the thing whose SIGTERM this reinterprets.
    install_self_kill_exit(run_dir, log_label=f"{args.backend}_{args.layer}")

    plan = LaunchPlan(
        layer=args.layer, user_prompt=args.prompt,
        run_dir=run_dir, runs_dir=runs_dir, run_id=run_id,
        model=args.model,
        env=env,
        system_prompt_text=system_prompt,
        footer_text=footer,
        reasoning_effort=args.reasoning_effort,
    )
    return run_agent(backend, plan, deadline, daemon_pid=daemon_pid)


# ---------------------------------------------------------------------------
# Run-id naming (CLI labeling convention — backend / layer / exp_name)
# ---------------------------------------------------------------------------

def _gen_run_id(backend: str, exp_name: str | None) -> str:
    """CLI-side run-id label: `YYMMDD_HHMMSS_<backend>[_<exp>]`."""
    base = datetime.now().strftime("%y%m%d_%H%M%S") + f"_{backend}"
    if exp_name:
        base += "_" + exp_name
    return base


# ---------------------------------------------------------------------------
# Harness daemon spawn
# ---------------------------------------------------------------------------

#: Forks the daemon, reports its pid on the inherited pipe fd, and exits.
_DETACH_STARTER = (
    "import os, sys\n"
    "wfd = int(sys.argv[1]); argv = sys.argv[2:]\n"
    "pid = os.fork()\n"
    "if pid == 0:\n"
    "    os.close(wfd); os.setsid(); os.execv(argv[0], argv)\n"
    "os.write(wfd, str(pid).encode()); os._exit(0)\n")


def _spawn_detached(argv: list, *, cwd: str, log_fh, env: dict) -> int:
    """Start `argv` as a grandchild in its own session; return its pid.

    NOT a child of this process, on purpose. The launcher goes on to exec the
    agent CLI (claude always, codex whenever a terminal is attached) or to run
    the PTY shim, and none of those ever reaps a child it did not start. A
    daemon that stopped before the agent did would linger as a zombie, and
    anything that tests liveness by `/proc/<pid>` or `kill -0` -- an agent's
    own daemon-restart script, say -- would read it as still running. A
    starter that forks the daemon and exits at once leaves it with init (or
    the nearest subreaper), which reaps it.
    """
    r, w = os.pipe()
    try:
        starter = subprocess.Popen(
            [sys.executable, "-c", _DETACH_STARTER, str(w), *argv],
            cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, pass_fds=(w,))
    finally:
        os.close(w)
    try:
        starter.wait()
        data = b""
        while chunk := os.read(r, 64):
            data += chunk
    finally:
        os.close(r)
    if starter.returncode != 0 or not data.strip().isdigit():
        raise RuntimeError(f"could not start {argv[3] if len(argv) > 3 else argv}: "
                           f"starter rc={starter.returncode}")
    return int(data)


def _pid_running(pid: int) -> bool:
    """Alive and not a zombie. `kill -0` alone answers yes for a zombie."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True                      # no /proc here: kill -0 is all we have


# ---------------------------------------------------------------------------
# Time limit
# ---------------------------------------------------------------------------

def _iso(epoch: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")


def run_deadline(max_hours: float | None, now: float | None = None) -> tuple[float | None, dict]:
    """This launch's deadline (epoch seconds, or None for no limit) and its `meta.json` `time_limit` record.

    `max_hours` counts from now, which is when the agent session starts (the rewriters have already run); 0 means
    no limit. A sub-agent inherits its parent's deadline (`DEADLINE_ENV`, `inf` for none): given no `--max-hours`
    of its own, it follows the parent; given one, the earlier deadline wins. A top-level launch given none gets
    `DEFAULT_MAX_HOURS`.
    """
    now = time.time() if now is None else now
    raw = os.environ.get(DEADLINE_ENV)
    try:
        inherited = float(raw) if raw else None
    except ValueError:
        inherited = None
    if max_hours is None and inherited is not None:
        hours, own = None, float("inf")
    else:
        hours = DEFAULT_MAX_HOURS if max_hours is None else max_hours
        own = now + hours * 3600.0 if hours > 0 else float("inf")
    deadline = min(own, inherited if inherited is not None else float("inf"))
    if deadline == float("inf"):
        return None, {}
    record = {"hours": hours or None, "deadline": _iso(deadline), "reached": False}
    if deadline != own:
        record["inherited"] = True
    return deadline, record


def time_limit_note(deadline: float | None, layer: str, now: float | None = None,
                    since: str = "your session started") -> str:
    """The footer sentence that tells an `animation` agent when the run is stopped, or "" without a limit. `since`
    names what the hours count from: the session's start, or `--resume`'s "this resume".

    Only that layer: the sentence speaks of the film it delivers. Other layers' runs are stopped all the same.
    """
    if deadline is None or layer != "animation":
        return ""
    from datetime import datetime
    now = time.time() if now is None else now
    hours = round(max(0.0, deadline - now) / 3600.0, 1)
    local = datetime.fromtimestamp(deadline).astimezone()
    zone = local.strftime("%Z") or local.strftime("UTC%z")       # some zones have no abbreviation
    at = f"{local.strftime('%H:%M')} {zone}"
    return (f"\n\nThis run is stopped {hours:g} hours after {since}, at {at}, without waiting for "
            "work in progress. Deliver before then; a run stopped without a delivery keeps its files, and the "
            "harness records the best film it rendered.\n")


def execute_with_time_limit(backend, plan, deadline: float | None, *, daemon_pid: int = 0) -> int:
    """`backend.execute(plan)`, stopped at `deadline`.

    At the deadline the agent CLI is stopped as for a SIGTERM (`backends.stop_agent_cli`: SIGTERM, then SIGKILL
    after the grace), unless a stop is already under way (a committed run stopping itself exits 0, and keeps it).
    `meta.json` records `time_limit.reached` at once, and again once the daemon, which cancels its children and
    finalizes `meta.json` as usual, has exited; the launcher then returns `EXIT_TIME_LIMIT`.
    """
    if deadline is None:
        return backend.execute(plan)
    import threading
    from . import backends as _backends
    fired: dict = {}
    done = threading.Event()
    run_dir = Path(plan.run_dir)

    def stop() -> None:
        if done.wait(max(0.0, deadline - time.time())):
            return
        # The CLI may not have started yet; keep asking until it has, the launch is over, or another stop began.
        while not done.is_set() and _backends.stop_status() is None:
            if _backends.stop_agent_cli(EXIT_TIME_LIMIT):
                if _backends.stop_status() == EXIT_TIME_LIMIT:
                    fired["at"] = time.time()
                    print(f"[cli] time limit reached ({_iso(deadline)}): stopping the agent CLI",
                          file=sys.stderr, flush=True)
                    _record_time_limit_reached(run_dir, fired["at"])
                return
            done.wait(1.0)

    stopper = threading.Thread(target=stop, name="time-limit", daemon=True)
    stopper.start()
    try:
        status = backend.execute(plan)
    finally:
        done.set()
    # Decided by the stop status, not by `fired`: the CLI can exit before the stopping thread gets to record it.
    if status != EXIT_TIME_LIMIT or _backends.stop_status() != EXIT_TIME_LIMIT:
        return status
    stopper.join(timeout=30)
    stop_daemon(daemon_pid)
    _record_time_limit_reached(run_dir, fired.get("at", time.time()))
    return EXIT_TIME_LIMIT


#: How long a stopped run's daemon gets to cancel its children and finalize `meta.json`: its drain bound and
#: transcript capture, with room to spare.
_DAEMON_FINALIZE_S = 180.0


#: The harness daemons this launcher started (not one it attached to), which it stops once its agent is done.
_spawned_daemons: set[int] = set()


def stop_daemon(pid: int) -> None:
    """SIGTERM a daemon this launcher started and wait for it to finalize `meta.json` and capture the transcript.

    The daemon follows the launcher (`agent/server.py::_Watchdog`) and would stop on its own once the launcher is
    gone; stopping it here instead means `meta.json` is final when the launcher returns.
    """
    if pid not in _spawned_daemons:
        return
    _spawned_daemons.discard(pid)
    if not _pid_running(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_for_exit(pid, timeout=_DAEMON_FINALIZE_S)


#: The stops `run_agent` starts the CLI once more for: an exit before any answer that named no cause.
_RETRIED_STOPS = ("claude_exited_before_answering", "codex_exited_before_answering")

#: How long `run_agent` waits before that one retry.
STARTUP_RETRY_DELAY_S = 10.0

#: Stops a resume cannot get past: the environment, or a sub-run's parent, decides those.
_NOT_RESUMABLE = ("claude_sandbox_cannot_start", "claude_never_started", "codex_blocked_screen",
                  "codex_never_started", "parent_gone")


def run_agent(backend, plan, deadline: float | None, *, daemon_pid: int) -> int:
    """Run the agent CLI to its end (`execute_with_time_limit`), then stop the daemon and record the run's agent time.

    A CLI that exits before it answered, for no reason it named (the claude and codex watches name a login, a
    usage limit, a blocked screen), is started once more after `STARTUP_RETRY_DELAY_S`, with the same daemon.
    A run that can be carried on says how: `--resume`.
    """
    from .backends import stop_status
    run_dir = Path(plan.run_dir)
    status = execute_with_time_limit(backend, plan, deadline, daemon_pid=daemon_pid)
    if status == _process.EXIT_AGENT_FATAL and stop_status() is None and \
            (_read_meta(run_dir).get("agent_stop") or {}).get("reason") in _RETRIED_STOPS:
        print(f"[cli] the agent CLI exited before it answered; starting it once more in "
              f"{STARTUP_RETRY_DELAY_S:g} s, with the same harness daemon", file=sys.stderr, flush=True)
        _merge_meta(run_dir, lambda meta: meta.setdefault("startup_retries", []).append(meta.pop("agent_stop")))
        time.sleep(STARTUP_RETRY_DELAY_S)
        status = execute_with_time_limit(backend, plan, deadline, daemon_pid=daemon_pid)
    stop_daemon(daemon_pid)
    if (run_dir / "meta.json").exists():
        from ..agent.agent_session import agent_time
        _merge_meta(run_dir, lambda meta: meta.update(agent_time=agent_time(run_dir)))
        _say_how_to_resume(run_dir, status)
    return status


def _read_meta(run_dir: Path) -> dict:
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _say_how_to_resume(run_dir: Path, status: int) -> None:
    """After a stop the run can be carried on from, the command that does it, also kept as `agent_stop.resume`."""
    meta = _read_meta(run_dir)
    stop = meta.get("agent_stop") or {}
    timed_out = status == EXIT_TIME_LIMIT
    if status == 0 or meta.get("parent_run_id") or stop.get("reason") in _NOT_RESUMABLE:
        return
    if not (meta.get("claude_session_id") or meta.get("codex_session_id")):
        return
    from .resume import _layout
    command = (f"python -m dycraft.cli --resume {_layout()[1](Path(run_dir))}"
               + (" --max-hours H" if timed_out else ""))
    print(f"[cli] to carry this run on in the same agent session: {command}"
          + (" (H more hours)" if timed_out else ""), file=sys.stderr, flush=True)
    if stop:
        _merge_meta(run_dir, lambda meta: meta.setdefault("agent_stop", {}).update(resume=command))


def _wait_for_exit(pid: int, timeout: float) -> None:
    end = time.time() + timeout
    while pid and time.time() < end and _pid_running(pid):
        time.sleep(0.5)


def _merge_meta(run_dir: Path, change) -> None:
    """Apply `change(meta)` to `meta.json` in place, written atomically (temporary file, then rename)."""
    path = Path(run_dir) / "meta.json"
    try:
        meta = json.loads(path.read_text())
        change(meta)
        tmp = path.with_name(f".meta.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        print(f"[cli] could not update {path}: {exc}", file=sys.stderr)


def _record_time_limit_reached(run_dir: Path, stopped_at: float) -> None:
    _merge_meta(run_dir, lambda meta: meta.setdefault("time_limit", {}).update(
        reached=True, stopped_at=_iso(stopped_at)))


def start_harness_daemon(
    run_dir: Path, env: dict, *, daemon_module: str, log_label: str = "cli",
) -> int:
    """Spawn `python -m <daemon_module> serve <run_dir>` detached.

    `daemon_module` is the per-agent harness daemon module (e.g.
    `dycraft.agent.layers.server`), picked by the caller from the
    layer via `_DAEMON_MODULE`.

    The daemon's watchdog follows the launcher's pid (DYCRAFT_AGENT_PID):
    the launcher runs the agent CLI as its child, so the daemon outlives a
    CLI the launcher starts again, and stops on its own once the launcher is
    gone (`agent/server.py::_Watchdog`). A dead daemon's port, owner record
    and log are set aside first.
    """
    from ..agent.server import (DaemonAlreadyOwned, acquire_run_dir,
                                read_owner_info)

    port_path = run_dir / ".harness.port"
    log_path = run_dir / ".daemon.log"

    # Attach rather than race. A run directory has exactly one harness daemon
    # (`agent/server.py::acquire_run_dir`), so if one is already live here the
    # right move is to leave it in charge — not to spawn a second that will
    # refuse, and certainly not to overwrite its `.harness.port`. Checking
    # before spawning also keeps `.daemon.log` intact: opening it "w" below
    # would truncate the incumbent's own log.
    try:
        acquire_run_dir(run_dir).close()
    except DaemonAlreadyOwned:
        info = read_owner_info(run_dir)
        print(f"[{log_label}] harness daemon already owns {run_dir} "
              f"(pid={info.get('pid')}, port={info.get('port')}); attaching "
              f"to it instead of starting a second one.", file=sys.stderr)
        return int(info.get("pid") or 0)

    # No daemon owns the run dir (the flock above), so a `.harness.port` here is a stopped daemon's, and would read
    # below as the new daemon being ready before it has bound. It is kept aside, with that daemon's owner record and
    # its log, which the new daemon's would otherwise overwrite.
    stale = [name for name in (".harness.port", ".harness.owner.json", ".daemon.log") if (run_dir / name).exists()]
    if stale:
        stamp = time.strftime("%y%m%d_%H%M%S")
        for name in stale:
            os.replace(run_dir / name, run_dir / f"{name}.{stamp}")
        print(f"[{log_label}] set the stopped daemon's {', '.join(stale)} aside (*.{stamp})", file=sys.stderr)
    log_fh = open(log_path, "w")
    daemon_env = dict(env)
    # The launcher lives exactly as long as the agent CLI it runs as its child
    # (`backends.run_agent_cli`); the daemon retargets to the CLI's own pid
    # once it appears.
    daemon_env.setdefault("DYCRAFT_AGENT_PID", str(os.getpid()))
    pid = _spawn_detached(
        [sys.executable, "-u", "-m", daemon_module, "serve", str(run_dir)],
        cwd=str(run_dir), log_fh=log_fh, env=daemon_env)

    deadline = time.time() + HARNESS_DAEMON_TIMEOUT_S
    while time.time() < deadline:
        if port_path.exists():
            port = port_path.read_text().strip()
            print(f"[{log_label}] harness daemon ready at 127.0.0.1:{port} "
                  f"(pid={pid})", file=sys.stderr)
            _spawned_daemons.add(pid)
            return pid
        if not _pid_running(pid):
            # Lost the ownership race to a daemon that started between our
            # pre-check and our spawn: ours refused and the incumbent is
            # serving. The end state we wanted already holds, so this is an
            # attach, not a launch failure. (Detached, the daemon's exit code
            # is not ours to read; the incumbent's liveness is the test.)
            info = read_owner_info(run_dir)
            owner = int(info.get("pid") or 0)
            if owner and owner != pid and _pid_running(owner):
                print(f"[{log_label}] harness daemon already owned by "
                      f"pid={owner} (port={info.get('port')}); "
                      f"attaching.", file=sys.stderr)
                return owner
            why = f"FATAL: DyCraft's daemon stopped before it was ready; see {log_path}"
            print(why, file=sys.stderr)
            record_launch_failure(
                run_dir, reason="harness_daemon_exited_early",
                detail=f"{daemon_module} exited before binding a port",
                message=why, log_path=log_path)
            sys.exit(1)
        time.sleep(HARNESS_DAEMON_POLL_S)
    why = (f"FATAL: DyCraft's daemon was not ready after {HARNESS_DAEMON_TIMEOUT_S} s; "
           f"see {log_path}")
    print(why, file=sys.stderr)
    record_launch_failure(
        run_dir, reason="harness_daemon_never_bound",
        detail=f"{daemon_module} did not write .harness.port within "
               f"{HARNESS_DAEMON_TIMEOUT_S}s",
        message=why, log_path=log_path)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Self-kill watcher
# ---------------------------------------------------------------------------

def self_kill_exit_code(run_dir: Path) -> int:
    """What a SIGTERM means for this run, as an exit status.

    SIGTERM arrives here for two opposite reasons and the default disposition
    cannot tell them apart:

      * **The run asked to stop.** `start_self_kill_watcher` SIGTERMs this
        process the moment the agent drops `$run_dir/.terminate`, which is the
        documented *final* step of a successful run — the agent CLI it drives
        does not exit on its own. That is a success, and it exited 143.
      * **Something killed it.** A batch launcher's stall or timeout
        ceiling, an operator, an OOM policy. That is a
        failure, and it also exited 143.

    A launch script propagates the status (so that `backends.EXIT_CODEX_FATAL`
    can reach the launcher), so without the translation a committed run would
    be recorded as `failed`. The sentinel is the discriminator, and only
    this process can see it, so the translation belongs here rather than in the
    shell that reads the code.

    Gated on the artifact as well as the sentinel: an agent that terminates
    without committing has not succeeded, and the daemon writes the commit's
    artifact, `ir.json` for an object, scene or film and `result.json` for a
    plain run's answer. A `stat` and not a JSON parse, because this runs in a
    signal handler — the richer check (`meta.outcome == "committed"`) belongs
    to a caller that is not in a handler and can afford it.
    """
    if not (run_dir / ".terminate").exists():
        return 128 + int(signal.SIGTERM)
    if not any((run_dir / name).is_file() for name in ("ir.json", "result.json")):
        return 128 + int(signal.SIGTERM)
    return 0


def install_self_kill_exit(run_dir: Path, log_label: str = "cli") -> None:
    """Translate the self-kill SIGTERM into an honest exit status.

    While the launcher waits on the agent CLI (`backends.run_agent_cli`), the
    SIGTERM is passed on to it and the launcher exits with this status once
    the CLI is gone. Otherwise, as while pumping a headless Codex's PTY,
    `os._exit`: the signal lands inside `backend.execute`, unwinding through
    that has nowhere good to go, and the default disposition did not run
    finalizers either.
    """
    from .backends import stop_agent_cli

    def _on_sigterm(signum, frame):                            # noqa: ANN001
        code = self_kill_exit_code(run_dir)
        why = ("committed, self-terminated" if code == 0
               else "no commit or no sentinel")
        if stop_agent_cli(code):
            print(f"[{log_label}] SIGTERM: {why} -> stopping the agent CLI, "
                  f"then exit {code}", file=sys.stderr, flush=True)
            return
        print(f"[{log_label}] SIGTERM: {why} -> exit {code}",
              file=sys.stderr, flush=True)
        # `os._exit` skips the launch's `finally`, which is what brings the top level up to date.
        publish(run_dir)
        os._exit(code)

    signal.signal(signal.SIGTERM, _on_sigterm)


def start_self_kill_watcher(
    run_dir: Path, target_pid: int, *, log_label: str = "cli",
) -> int:
    """Spawn a detached watcher that SIGTERMs `target_pid` once the sentinel
    file `$run_dir/.terminate` appears.

    Why this exists: codex's `workspace-write` sandbox runs each tool shell
    in a fresh PID namespace, so the agent CLI's host pid is invisible from
    inside the sandbox — a `kill` from a tool shell can never reach the
    parent. The watcher lives outside the sandbox and delivers SIGTERM
    when the agent drops the sentinel from inside.

    Both ClaudeBackend and CodexBackend use this so the termination contract
    is the same across both CLIs.

    It watches until the sentinel appears or the target exits, with no time
    limit: a fixed cutoff would leave a run that finishes later (a long film
    across resumes) waiting after its commit for a SIGTERM nobody sends.
    """
    sentinel = run_dir / ".terminate"
    log_path = run_dir / ".self_kill_watcher.log"
    # A sentinel an earlier session of this run dir left behind is not a
    # request to stop this one: it killed a relaunched sub-run within a second.
    if sentinel.exists():
        sentinel.unlink(missing_ok=True)
        print(f"[{log_label}] removed a stale {sentinel.name} left by an earlier "
              "session", file=sys.stderr)
    bootstrap = (
        "import os, signal, sys, time\n"
        f"sentinel = {str(sentinel)!r}\n"
        f"target   = {int(target_pid)}\n"
        "print(f'[self-kill] watching {sentinel} for pid {target}',\n"
        "      flush=True)\n"
        "while True:\n"
        "    if os.path.exists(sentinel):\n"
        "        time.sleep(1.0)\n"
        "        try:\n"
        "            os.kill(target, signal.SIGTERM)\n"
        "            print(f'[self-kill] SIGTERM -> {target}', flush=True)\n"
        "        except ProcessLookupError:\n"
        "            print('[self-kill] target already gone', flush=True)\n"
        "        sys.exit(0)\n"
        "    try:\n"
        "        os.kill(target, 0)\n"
        "    except ProcessLookupError:\n"
        "        print('[self-kill] target exited; watcher stopping',\n"
        "              flush=True)\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.5)\n"
    )
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", bootstrap],
        cwd=str(run_dir),
        stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[{log_label}] self-kill watcher pid={proc.pid} "
          f"sentinel={sentinel}", file=sys.stderr)
    return proc.pid


# ---------------------------------------------------------------------------
# Env + prompt + footer composition
# ---------------------------------------------------------------------------

def build_env_dict(
    user_prompt: str,
    *,
    run_dir: Path,
    runs_dir: Path,
    run_id: str,
    layer: str,
    runtime: str,
    advisory_hours: float = 1.0,
    render_backend: str | None = None,
) -> dict[str, str]:
    """Build the env dict the agent CLI + harness daemon inherit.

    Sets the DYCRAFT_* keys the daemon and system prompt reference, plus
    EGL/Mesa fixes that we've empirically needed across nearly every Codex
    run and at least one Claude run (Open3D's Filament OffscreenRenderer
    talks to EGL; Codex's bubblewrap sandbox unshares /run/user/<uid>, so
    the default XDG_RUNTIME_DIR + EGL platform fail and the renderer
    segfaults). Setting them unconditionally costs nothing and avoids
    agents wasting turns rediscovering the combo.
    """
    env = os.environ.copy()
    env["DYCRAFT_USER_PROMPT"]         = user_prompt
    # A parent's task file is not this run's task; the launcher sets it again
    # when this run's own task went to a file.
    env.pop(TASK_FILE_ENV, None)
    env["DYCRAFT_LAYER"]               = layer
    env["DYCRAFT_RUNTIME"]             = runtime
    env["DYCRAFT_RUN_DIR"]             = str(run_dir)
    env["DYCRAFT_RUNS_DIR"]            = str(runs_dir)
    env["DYCRAFT_RUN_ID"]              = run_id
    env["DYCRAFT_ADVISORY_HOURS"]      = repr(float(advisory_hours))
    # The launcher's own python (the chord conda env when invoked from
    # `conda activate chord`) has all the harness deps. Codex's sandbox
    # tool shells run under `bash -lc`, which sources .bashrc and ends up
    # pointing `python` at the base conda env instead — that env lacks
    # httpx/numpy. Both prompts always invoke the harness via
    # `$DYCRAFT_PYTHON` to dodge this; claude doesn't strictly need it but
    # paying the same price keeps the prompts identical.
    env["DYCRAFT_PYTHON"]              = sys.executable
    # The CAD interpreter, resolved once here rather than per child. Distinct
    # from `DYCRAFT_PYTHON`: only this one carries build123d, and on a host
    # where the launcher runs from a different env the two differ. Exported so
    # the object prompts can name an interpreter the agent can actually run a
    # scratch geometry script with, and so every child resolves the same one
    # the daemon does. Skipped when unresolvable — set-but-empty is the
    # "deliberately unconfigured" sentinel in `_cad_env._resolve_b123d_python`.
    from ..agent.cad._cad_env import b123d_python as _resolve_cad_python
    _b123d_interp = _resolve_cad_python()
    if _b123d_interp:
        env["DYCRAFT_B123D_PYTHON"]    = _b123d_interp
    # Put the launcher interpreter's `bin/` on PATH. The agent CLIs sandbox themselves with `bwrap` and `socat`, which they
    # look up on PATH — and a launcher started by absolute path
    # (`/…/envs/dycraft/bin/python -m dycraft.cli`) never puts its own env on
    # PATH, so claude refuses to start with "sandbox required but unavailable:
    # bubblewrap not installed" while `bwrap` sits unreferenced in that very
    # env. Appended, not prepended: a tool the caller deliberately put earlier
    # on PATH still wins.
    _launcher_bin = str(Path(sys.executable).resolve().parent)
    _path = env.get("PATH", "")
    if _launcher_bin not in _path.split(os.pathsep):
        env["PATH"] = f"{_path}{os.pathsep}{_launcher_bin}" if _path else _launcher_bin
    # Which renderer the daemon and every sub-run draw with. Exported unconditionally rather than only when the flag
    # was passed, so an inherited `DYCRAFT_RENDER_BACKEND` from the launching
    # shell cannot quietly override an explicit `--render-backend open3d`.
    #
    # `render_backend` is the *flag*, which is None whenever it was not passed —
    # so falling back to a literal here would export `open3d` to the daemon even
    # on a run whose renderer was set by the environment, and undo `main()`'s
    # resolution for every CLI backend. Read `main()`'s already-resolved pin
    # instead; it is the single place the flag > env > default order lives.
    env[RENDER_BACKEND_ENV] = (
        render_backend
        or os.environ.get(RENDER_BACKEND_ENV)
        or DEFAULT_RENDER_BACKEND)
    env.setdefault("EGL_PLATFORM", "surfaceless")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    _xdg_dir = run_dir / ".xdg_runtime"
    _xdg_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(_xdg_dir, 0o700)
    except OSError:
        pass
    env.setdefault("XDG_RUNTIME_DIR", str(_xdg_dir))
    env.setdefault("MESA_SHADER_CACHE_DIR", str(_xdg_dir))
    return env


if __name__ == "__main__":
    from .. import _oom
    _oom.arm_entry()
    raise SystemExit(main())
