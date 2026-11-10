"""Score films with the paper's two physical-plausibility metrics: penetration rate and floating rate.

    python -m dycraft.benchmark.score runs/bench_long                 # every arm of a benchmark directory
    python -m dycraft.benchmark.score runs/bench_long/B13_*/dycraft_codex  path/to/film.blend --jobs 8

Each path is a DyCraft run (or its bundle), a bare agent's arm, a benchmark output directory (every arm in it), or
any `.blend`, so another method's films can be scored the same way. A `.blend` needs nothing else: it is exported
frame by frame, read-only, with the animation layer's own exporter. Name it after its brief (`b14_...blend`), or pass
`--task`, for the briefs whose floating rate exempts a flying or floating body (B14, B20, B22).

Results go to `--out` (default `<runs dir>/.scores/`), one folder per input with `penetration.json`, `floating.json`
and the cached geometry; a re-run with the same inputs reuses the cache. A benchmark arm also gets
`benchmark/score.json`, which `python -m dycraft.benchmark.report` shows. The definitions are in README.md beside
this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import floating, penetration
from .common import Settings, blender_python, coal_python
from .inputs import Input, benchmark_arms, resolve


def expand(paths, task) -> list[Input]:
    out = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        arms = benchmark_arms(path) if path.is_dir() and not (path / "bundle").is_dir() else []
        for p in (arms or [path]):
            try:
                out.append(resolve(p, task))
            except FileNotFoundError as error:
                if not arms:
                    raise SystemExit(str(error))
                print(f"skipped {p}: {error}", file=sys.stderr)
    return out


def result_dir(root: Path, inp: Input) -> Path:
    return root / f"{inp.label}-{hashlib.sha256(str(inp.path).encode()).hexdigest()[:8]}"


def summary(inp: Input, pen: dict | None, flo: dict | None, where: Path) -> dict:
    row = {"input": str(inp.path), "kind": inp.kind, "task": inp.task, "results": str(where)}
    if pen is not None:
        p = pen["primary"]
        row.update(penetration_rate=p["dynamic_penetration_frame_rate"], frames=p["evaluated_frames"],
                   penetrating_frames=p["dynamic_invalid_frames"],
                   cross_object_rate=p["cross_object_dynamic_frame_rate"],
                   self_penetration_rate=p["self_penetration_dynamic_frame_rate"])
    if flo is not None:
        f = flo["primary"]
        row.update(floating_rate=f["floating_rate"], floating_body_frames=f["floating_body_frames"],
                   dynamic_body_frames=f["animated_body_frames"])
    return row


def table(rows) -> str:
    pct = lambda v: "" if v is None else f"{100 * v:.1f}"
    lines = [("input", "frames", "penetration %", "floating %")]

    def rate(r, metric):
        value = r.get(f"{metric}_rate")
        return "failed" if value is None and f"{metric}:" in r.get("error", "") else pct(value)

    lines += [(Path(r["input"]).name if Path(r["input"]).name != "bundle" else Path(r["input"]).parent.name,
               str(r.get("frames", "")), rate(r, "penetration"), rate(r, "floating")) for r in rows]
    widths = [max(len(row[i]) for row in lines) for i in range(4)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.score", description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 2)[2], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", metavar="PATH", help="a run, arm, bundle, benchmark directory or .blend")
    ap.add_argument("--out", type=Path, help="results root (default <runs dir>/.scores)")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)),
                    help="parallel geometry processes per film (its frames are split between them)")
    ap.add_argument("--chunks", type=int, default=None,
                    help="split each film's frames into this many pieces, --jobs at a time (default: --jobs). A "
                         "geometry process grows with the frames it measures, so more pieces lower the peak memory "
                         "of a long film")
    ap.add_argument("--task", help="the brief every input answers (B01..B24); without it, each input's brief is "
                                   "read from its path")
    ap.add_argument("--metric", choices=("both", "penetration", "floating"), default="both")
    ap.add_argument("--json", type=Path, help="also write the summary rows here")
    ap.add_argument("--recompute", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--blender-python", help="the Blender (bpy) interpreter (default: the one `python -m "
                                             "dycraft.install blender` set up, or DYCRAFT_BLENDER_PYTHON)")
    ap.add_argument("--coal-python", help="an interpreter with coal (default: DYCRAFT_COAL_PYTHON, else this one)")
    ap.add_argument("--verbose", action="store_true", help="show the Blender children's output")
    args = ap.parse_args(argv)

    from dycraft import _home
    root = (args.out or _home.default_runs_dir() / ".scores").expanduser().resolve()
    settings = Settings(blender_python(args.blender_python), coal_python(args.coal_python),
                        jobs=max(1, args.jobs), chunks=args.chunks, quiet=not args.verbose, recompute=args.recompute)
    inputs = expand(args.paths, args.task)
    rows = []
    for inp in inputs:
        where = result_dir(root, inp)
        started = time.perf_counter()
        print(f"scoring {inp.path} ({inp.kind}{', task ' + inp.task if inp.task else ''}) -> {where}",
              file=sys.stderr, flush=True)
        # A metric that cannot be scored (a bare agent's blend with instances the exporter refuses, say) is recorded
        # with its error and traceback; the other metric and the other films are still scored.
        results, errors = {}, []
        for name, metric in (("penetration", penetration), ("floating", floating)):
            if args.metric not in ("both", name):
                continue
            try:
                results[name] = metric.run(inp, where, settings)
            except (SystemExit, Exception) as error:                 # noqa: BLE001
                errors.append((name, f"{type(error).__name__}: {error}", traceback.format_exc()))
                print(f"{name} failed for {inp.path}: {errors[-1][1]}", file=sys.stderr, flush=True)
        row = summary(inp, results.get("penetration"), results.get("floating"), where)
        if errors:
            row["error"] = "; ".join(f"{name}: {message}" for name, message, _ in errors)
            row["traceback"] = "\n".join(trace for _, _, trace in errors)
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        if inp.arm is not None:
            (inp.arm / "benchmark" / "score.json").write_text(json.dumps(row, indent=1))
    print(table(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
