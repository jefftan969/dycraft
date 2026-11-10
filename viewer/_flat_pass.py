"""The viewer's shader color pass (`render/viewer_flat.compute`), in its own interpreter.

usage: python -m <package>.viewer._flat_pass <run_dir> <ir_id>...

Started by `derived._flat_pass`, so that a viewer busy serving a run's other requests does not slow the pass: its
packing and its cache writes are Python, and inside the server they wait on the threads serializing the run's
trajectory and track, several times longer than the Blender process takes. Prints the ids it read as a JSON list.
"""
import json
import sys
from pathlib import Path

from ..render import RenderUnavailable
from ..render import viewer_flat as VF
from . import derived


def main(argv: list[str]) -> int:
    run_dir, wanted = Path(argv[0]), set(argv[1:])
    entries = [e for e in derived.resolve(run_dir) if e.ir_id in wanted]
    try:
        read = VF.compute(entries)
    except RenderUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(sorted(read)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
