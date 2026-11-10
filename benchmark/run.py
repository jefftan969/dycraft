"""Run benchmark briefs through DyCraft — one `python -m dycraft.cli` run per brief.

    python -m dycraft.benchmark.run --briefs short --agent codex --model gpt-6-astra \\
        --effort medium --budget-hours 2 --out runs/bench_short_codex
    python -m dycraft.benchmark.run --briefs B01S,B13 --agent claude --model claude-opus-5-5 \\
        --effort high --budget-hours 4 --out runs/bench_mixed --dry-run

Each brief becomes one `--layer animation` run, which authors the scene and the motion in a
single session, at `<out>/<brief>_<staged id>/dycraft_<agent>/`. Its prompt is the brief, then the
film-length paragraph, then the wall-clock budget paragraph — the form the paper's DyCraft runs
received (`benchmark/briefs.py`). The CLI's scene and motion rewriters then write `scene.txt` and
`motion.txt` and point the prompt at them, as in the paper's runs.

Unless `--budget-hours` is given, each brief takes its set's budget: 8 hours for every brief in the
short set and 24 in the medium and long sets; a listed id takes its own set's (B01S 8, B13 24), and a
brief of your own (`--prompt`) 8. The budget is told to the agent and passed as `--advisory-hours`,
which the harness counts down in every reply. The launcher stops the run at the budget plus 20 minutes
(`--max-hours`, counted from the agent session's start, after the scene and motion rewriters). With
`--budget-hours 0` there is no budget: no paragraph, no countdown (`--advisory-hours 0`), and the
launcher's default limit. The stop is for a run that ignored its deadline or wedged, not how the
budget is met. This runner keeps its own backstop 15 minutes past that, in case the launcher itself
wedges.

When a run ends, `<run>/benchmark/record.json` records its outcome (from `finish_result.json`), its
film and the film's length. `python -m dycraft.benchmark.report <out>` puts both arms' films side by
side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import briefs as B

#: Past the stated budget before the launcher stops the run (the arm is asked to wrap up at the
#: deadline, and that wrap-up is work).
GRACE_MINUTES = 20
#: Past the launcher's own limit before this runner stops it.
BACKSTOP_MINUTES = 15
#: Each rewriter (scene, motion) is bounded at 900 s and runs before the launcher's clock starts.
REWRITE_ALLOWANCE_S = 2 * 900
#: The budget each set gets unless `--budget-hours` is given. The short and medium cuts are sized to fit in
#: 6-8 and 12-24 hours (`benchmark/sizing.py`); the long briefs get 24.
DEFAULT_BUDGET_HOURS = {"short": 8.0, "medium": 24.0, "long": 24.0}
#: A brief of your own (`--prompt`).
PROMPT_BUDGET_HOURS = 8.0


def arm_name(agent: str) -> str:
    return f"dycraft_{agent}"


def budget_for(brief: B.Brief, horizon: str | None, given: float | None) -> float | None:
    """The budget a brief runs with: `given` when set (0: none), else its set's or a brief of your own's."""
    if given is not None:
        return given or None
    if brief.id == B.ADHOC_ID:
        return PROMPT_BUDGET_HOURS
    return DEFAULT_BUDGET_HOURS[horizon or brief.horizon]


def command(brief_dir: Path, prompt_file: Path, args, budget_hours: float | None) -> list[str]:
    cmd = [sys.executable, "-m", "dycraft.cli",
           "--backend", args.agent,
           "--layer", "animation",
           "--on-rewrite-failure", "abort",
           "--runs-dir", str(brief_dir),
           "--run-id", arm_name(args.agent),
           "--prompt", str(prompt_file)]
    if args.model:
        cmd += ["--model", args.model]
    if args.effort:
        cmd += ["--reasoning-effort", args.effort]
    if budget_hours:
        cmd += ["--advisory-hours", f"{budget_hours:g}",
                "--max-hours", f"{max_hours(budget_hours):g}"]
    else:
        cmd += ["--advisory-hours", "0"]
    return cmd + list(args.cli_arg or ())


def max_hours(budget_hours: float | None) -> float | None:
    """The launcher's limit for a budget: the budget plus `GRACE_MINUTES`."""
    return budget_hours + GRACE_MINUTES / 60 if budget_hours else None


def backstop_seconds(budget_hours: float | None) -> float:
    """When this runner stops a launcher that outlived its own limit (the rewriters run before that clock)."""
    from ..cli.cli import DEFAULT_MAX_HOURS
    hours = max_hours(budget_hours) or DEFAULT_MAX_HOURS
    return hours * 3600 + BACKSTOP_MINUTES * 60 + REWRITE_ALLOWANCE_S


def _run_one(brief: B.Brief, horizon: str | None, args) -> dict:
    out = Path(args.out).resolve()
    brief_dir = out / brief.slug
    brief_dir.mkdir(parents=True, exist_ok=True)
    film = args.film_seconds if args.film_seconds_given else B.FILM_SECONDS[horizon or brief.horizon]
    budget = budget_for(brief, horizon, args.budget_hours)
    prompt = B.dycraft_prompt(brief, film_seconds=film, budget_hours=budget)
    prompt_file = brief_dir / f"prompt_{arm_name(args.agent)}.md"
    prompt_file.write_text(prompt + "\n")
    cmd = command(brief_dir, prompt_file, args, budget)
    run_dir = brief_dir / arm_name(args.agent)
    record = {"brief": brief.id, "slug": brief.slug, "arm": arm_name(args.agent),
              "agent": args.agent, "model": args.model, "effort": args.effort,
              "horizon": horizon or brief.horizon, "budget_hours": budget, "film_seconds": film,
              "prompt": str(prompt_file), "command": cmd, "run_dir": str(run_dir)}
    if args.dry_run:
        print(" ".join(cmd))
        return record
    log = brief_dir / f"{arm_name(args.agent)}.log"
    limit = backstop_seconds(budget)
    started = time.time()
    print(f"[benchmark.run] {brief.id}: {arm_name(args.agent)} -> {run_dir}", flush=True)
    with log.open("a") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            record["hard_stop"] = True
    record.update(returncode=rc, wall_seconds=round(time.time() - started, 1), log=str(log))
    # The launcher lays a run out for a reader, with its records in `src/` (`run_layout`).
    from ..run_layout import internal_dir
    run_dir = internal_dir(run_dir)
    meta = run_dir / "meta.json"
    if meta.is_file():
        limit_record = json.loads(meta.read_text()).get("time_limit") or {}
        if limit_record.get("reached"):
            record["time_limit"] = limit_record
    finish = run_dir / "finish_result.json"
    if finish.is_file():
        result = json.loads(finish.read_text())
        record["outcome"] = result.get("outcome")
        record["unmet"] = len(result.get("unmet") or ())
    blend = run_dir / "bundle" / "animation.blend"
    record["blend"] = str(blend) if blend.is_file() else None
    film_path = run_dir / "bundle" / "film" / "clip.mp4"
    record["film"] = str(film_path) if film_path.is_file() else None
    manifest = run_dir / "ir.json"
    if manifest.is_file():
        ir = json.loads(manifest.read_text())
        if ir.get("fps") and ir.get("n_frames"):
            record["seconds"] = round(ir["n_frames"] / ir["fps"], 1)
    (run_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    (run_dir / "benchmark" / "record.json").write_text(json.dumps(record, indent=1))
    return record


def add_common(ap: argparse.ArgumentParser, *, imported_characters: bool = True) -> None:
    ap.add_argument("--briefs", default=None,
                    help="a set (long, short, medium) or comma-separated ids (B01S,B13)")
    ap.add_argument("--prompt", default=None,
                    help="instead of --briefs: one brief of your own, as text or a .md file")
    ap.add_argument("--agent", required=True, choices=("codex", "claude"))
    ap.add_argument("--model", default=None, help="passed to the agent CLI")
    ap.add_argument("--effort", default=None, help="reasoning effort, passed verbatim")
    ap.add_argument("--out", required=True, help="results directory, shared by both arms")
    if imported_characters:
        ap.add_argument("--imported-characters", action="store_true",
                        help="the long set and listed ids: B10 and B13-B16 take their characters from the "
                             "asset library (animation_briefs_imported_characters.md); the short and medium "
                             "sets already do where they have no cut")
    ap.add_argument("--parallel", type=int, default=1, help="briefs run at once (default 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the prompts and print the commands; run nothing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.run",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap)
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="wall-clock budget told to the agent (default per set: 8 for short, 24 for medium "
                         "and long, whose paper runs had 8; 8 for --prompt); the launcher stops the run 20 "
                         "minutes after it. 0: no budget paragraph, no advisory countdown (--advisory-hours 0), "
                         "and the launcher's default --max-hours")
    ap.add_argument("--film-seconds", default=None,
                    help="film-length ceiling, N or LOW-HIGH or none (default: 60-75 for short and "
                         "long briefs, 120-150 for medium)")
    ap.add_argument("--cli-arg", action="append",
                    help="extra argument for `python -m dycraft.cli` (repeatable)")
    args = ap.parse_args(argv)
    if args.budget_hours is not None and args.budget_hours < 0:
        raise SystemExit("--budget-hours must be 0 (none) or more")
    args.film_seconds_given = args.film_seconds is not None
    if args.film_seconds_given:
        args.film_seconds = B.parse_film_seconds(args.film_seconds)
    chosen, horizon = B.choose(args, imported_characters=args.imported_characters, library=True)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(pool.map(lambda b: _run_one(b, horizon, args), chosen))
    if not args.dry_run:
        for r in records:
            print(f"[benchmark.run] {r['brief']}: rc={r.get('returncode')} "
                  f"outcome={r.get('outcome')} film={'yes' if r.get('film') else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
