"""Size a brief: how long its film will be and how long a run takes to deliver it.

    python -m dycraft.benchmark.sizing                 # every shipped brief's plan, predicted film and runtime
    python -m dycraft.benchmark.sizing --write         # regenerate the sizing tables in animation_24_briefs.md
    python -m dycraft.benchmark.sizing --beats FILE    # size a brief of your own from its beats

A brief's film is the sum of every state it implies, each timed at a pace the body could produce: the plan. The
agent never speeds motion up to fit, so the plan is a floor; finished films run longer (`PACE`). The run's time
follows the predicted film (`runtime`). The beats of the shipped briefs are in `sizing_beats.py`.

Rate card for timing a beat (the `humanoid_motion` skill's rules; the G1's 0.6 m legs walk slower than an adult's,
and its parallel grippers cost extra beats, such as set-downs and handovers, rather than slower ones):

    walk                     ~1-1.2 m/s adult, ~0.8 m/s G1; 0.45-0.9 m/s loaded, in tight space, or carrying liquid
    start and stop a walk    ~0.5 s each
    turn                     <= 90 deg/s, stepped past 45 deg: 90 deg ~1.3 s, 180 deg ~2.6 s
    reach and grasp          ~1 s (the gaze arrives first)
    place, release           ~1 s; a precise insert ~1.5 s
    two-handed lift or set   ~2 s
    squat or kneel           1-3 s each way, longer under load
    a check or a reaction    1-2 s held, long enough for the camera to read
    establish and end hold   4-8 s in total
    a process (melting, filling): the shortest span that still reads as the change happening

A beats file has one beat per line, `LOW-HIGH label` or `SECONDS label`, in seconds; blank lines and lines
starting with `#` are skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .briefs import BRIEFS_FILE
from .sizing_beats import BEATS

#: Finished films run this much longer than the plan's physical-rate total: authors pace motion a little slower
#: than the fastest the body could produce, and hold beats for the camera.
PACE = 1.28

#: Working hours to the first delivered film, against the predicted film in minutes: a line fitted to runs that
#: delivered one film covering their brief, with the characters taken from the asset library, the G1 or none, on
#: GPT-6 at `xhigh`. `BAND` is the line's 10th-90th percentile as a ratio. Beyond about 1.5 minutes of predicted
#: film the line is an extrapolation. Characters authored from nothing take longer and vary more.
HOURS_AT_ZERO, HOURS_PER_MINUTE = 2.9, 3.9
BAND = (0.7, 1.4)


def plan_seconds(beats) -> float:
    """The plan: the sum of every beat's midpoint."""
    return sum((lo + hi) / 2 for _, _, lo, hi in beats)


def predicted_film(beats) -> float:
    return plan_seconds(beats) * PACE


def runtime(film_seconds: float) -> tuple[float, float, float]:
    """Expected working hours to the first film, with the low and high ends of the band."""
    mid = HOURS_AT_ZERO + HOURS_PER_MINUTE * film_seconds / 60
    return mid, mid * BAND[0], mid * BAND[1]


def mmss(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m}:{s:02d}"


#: The sizing tables in `animation_24_briefs.md`, in file order: the long set's three groups, the short set and
#: the medium set. A row is (label, ids); a row over several ids gives their range.
TABLES = [
    [(f"B{n:02d}", [f"B{n:02d}"]) for n in range(1, 9)],
    [(f"B{n:02d}", [f"B{n:02d}"]) for n in range(9, 17)],
    [(f"B{n:02d}", [f"B{n:02d}"]) for n in range(17, 25)],
    [(f"B{n:02d}S", [f"B{n:02d}S"]) for n in range(1, 13)]
    + [("B13 (library)", ["B13"]), ("B14 (library)", ["B14"]), ("B15S", ["B15S"]), ("B16 (library)", ["B16"]),
       ("B17-B24", [f"B{n}" for n in range(17, 25)])],
    [(i, [i]) for i in ("B01M", "B02M", "B03M", "B04M", "B06M", "B08M", "B09M", "B10M", "B11M", "B12M")]
    + [("B05, B07", ["B05", "B07"]), ("B13-B16 (library)", ["B13", "B14", "B15", "B16"]),
       ("B17-B24", [f"B{n}" for n in range(17, 25)])],
]
HEADER = "| | plan | predicted film | expected runtime |"
_TABLE = re.compile(r"^\| \| plan \| predicted film \| [^\n]*\|\n(?:\|[^\n]*\n)+", re.M)


def _span(values, f) -> str:
    lo, hi = f(min(values)), f(max(values))
    return lo if lo == hi else f"{lo}-{hi}"


def table(rows) -> str:
    out = [HEADER, "|---|---|---|---|"]
    for label, ids in rows:
        plans = [plan_seconds(BEATS[i]) for i in ids]
        films = [p * PACE for p in plans]
        hours = [runtime(f) for f in films]
        mids = _span([h[0] for h in hours], lambda v: f"{v:.0f}")
        cells = (_span(plans, mmss), _span(films, mmss),
                 f"{mids} h ({min(h[1] for h in hours):.0f}-{max(h[2] for h in hours):.0f})")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_tables(path: Path = BRIEFS_FILE) -> str:
    """The brief file with every sizing table replaced by the generated one."""
    text = path.read_text()
    blocks = list(_TABLE.finditer(text))
    if len(blocks) != len(TABLES):
        raise SystemExit(f"{path.name}: found {len(blocks)} sizing tables, expected {len(TABLES)}")
    for m, rows in reversed(list(zip(blocks, TABLES))):
        text = text[:m.start()] + table(rows) + "\n" + text[m.end():]
    return text


_BEAT = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s+(.+?)\s*$")


def read_beats(path: Path):
    beats = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _BEAT.match(line)
        if not m:
            raise SystemExit(f"{path}:{n}: expected 'LOW-HIGH label' or 'SECONDS label', got {line!r}")
        lo = float(m.group(1))
        beats.append(("", m.group(3), lo, float(m.group(2) or lo)))
    return beats


def describe(beats) -> str:
    film = predicted_film(beats)
    mid, lo, hi = runtime(film)
    return (f"plan {mmss(plan_seconds(beats))}, predicted film {mmss(film)}, "
            f"expected runtime {mid:.0f} h ({lo:.0f}-{hi:.0f})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.sizing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"regenerate the sizing tables in {BRIEFS_FILE.name}")
    ap.add_argument("--beats", type=Path, metavar="FILE", help="size a brief of your own from its beats")
    args = ap.parse_args(argv)
    if args.beats:
        print(describe(read_beats(args.beats)))
    elif args.write:
        BRIEFS_FILE.write_text(write_tables())
        print(f"wrote the sizing tables in {BRIEFS_FILE}")
    else:
        for bid, beats in BEATS.items():
            print(f"{bid:5} {describe(beats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
