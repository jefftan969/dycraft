"""Stage a committed animation run's frozen scene so the viewer can draw it.

    python -m dycraft.animation.stage_scene <run_dir> [...]

An animation run authors its scene mid-run and freezes it inside the bundle at
`bundle/design_state/`. The viewer reads an animation run's geometry from
`<run>/scene.json` plus `<run>/catalog/` and nowhere else, so it refuses the
run until the scene is staged there. A commit stages it
(`AnimationSession.stage_viewer_scene`); this re-runs that for a run whose
staging failed or predates it.

A repair tool rather than a launch mode, and deliberately not lazy work inside
the viewer: the viewer is read-only over `runs/` (its one exception, the
`/derived` bake, writes to a cache *outside* the run dir and says so), whereas
this copies tens of megabytes into the run itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..agent.layers.scene_staging import materialize_scene
from .blender import SCHEMA


def eligible(run_dir: Path) -> str | None:
    """None when this run can be staged, else why it cannot.

    Checked before staging rather than inside it so a sweep can report a
    skipped run in the same voice as a staged one.
    """
    ir_path = run_dir / "ir.json"
    if not ir_path.is_file():
        return "no ir.json — geometry is written on commit"
    try:
        ir = json.loads(ir_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable ir.json: {exc}"
    if not isinstance(ir, dict) or ir.get("schema") != SCHEMA:
        return f"ir.json is not a {SCHEMA} bundle"
    if not (run_dir / "bundle/design_state/scene.json").is_file():
        return "bundle carries no design_state/scene.json"
    return None


def stage_run(run_dir: Path, *, force: bool = False) -> dict:
    """Stage one run. Raises SystemExit on a refusal, as its callees do."""
    run_dir = Path(run_dir).resolve()
    why = eligible(run_dir)
    if why is not None:
        raise SystemExit(f"FATAL: {run_dir.name}: {why}")
    return materialize_scene(run_dir / "bundle", run_dir, force=force)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dycraft.animation.stage_scene",
        description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="restage over a scene.json naming a different scene")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what each run would do, write nothing")
    args = parser.parse_args(argv)

    failures = 0
    for run_dir in args.run_dir:
        run_dir = Path(run_dir).resolve()
        name = run_dir.name
        if args.dry_run:
            why = eligible(run_dir)
            print(f"{name}: {'would stage' if why is None else f'skip: {why}'}")
            failures += why is not None
            continue
        # One run's refusal must not abort the rest of a sweep: these are
        # independent repairs, and the caller wants the ones that can be done.
        try:
            staged = stage_run(run_dir, force=args.force)
        except (Exception, SystemExit) as exc:      # noqa: BLE001
            print(f"{name}: failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{name}: {'skipped (already staged)' if staged['skipped'] else 'staged'} "
              f"{staged['scene_ref']} + {staged['n_entries']} ObjectIR entries")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
