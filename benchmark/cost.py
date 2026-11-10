"""Agent-hours per minute of delivered film.

    python -m dycraft.benchmark.cost RUN_DIR [RUN_DIR ...] [--json]

A run's cost is the agent time it took per minute of the film it delivered, counted two ways:

* **main**: the main agent's sessions only (the run directory itself). A resumed run's sessions are summed, and the
  time between them is not counted.
* **total**: the main sessions plus every sub-agent and child run below the run directory: the scene and motion
  rewriters, `review()`, and the object, scene and motion runs the agent dispatched. Sessions that ran at the same
  time are summed, as CPU-hours are.

A session is one harness daemon's lifetime, from its `daemon_start` to its `daemon_shutdown` event in the run
directory's `trace.jsonl`. The daemon starts with the agent CLI and stops when the CLI exits. A session with no
recorded shutdown (a killed launcher) ends at its last trace event. Agent-hours are the sessions' *active* time:
whenever neither the trace nor the agent's transcript records anything for more than `IDLE_CUTOFF_S`, the time
beyond the cutoff is not counted. This keeps a CLI that sits on an error (a usage limit) until someone notices out
of the cost. The raw session lifetimes are reported as `session_hours_*`.

A bare-agent arm (`benchmark/bare.py`) is one session with no sub-agents: it starts at the runner's `started_at`,
lasts the runner's `wall_seconds`, and its activity is read from the session transcripts the runner copies out.

Only time up to the delivery counts: the `commit` event in the run's trace that recorded the delivered manifest
(`ir.json`'s `id`). Work after it (a wrap-up, or a later round reopened on the same run that did not replace the
delivery) is not part of that film's cost. A run that delivered without committing is counted to its end.

The film's length is the manifest's `n_frames / fps` (`ir.json`, or a harness-packaged `bundle/ir.uncommitted.json`),
else the duration of `bundle/film/clip.mp4`. Token counts come from each session's `meta.json` `usage`.

Pure standard library and read-only, so `benchmark/report.py` can call `cost` for every arm it lists.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import struct
import sys
from datetime import datetime
from pathlib import Path

#: A stretch with no recorded activity counts only up to this long. Long enough that a model's own silent work
#: (a long reasoning turn, a wait on a render it started) is never cut, short enough that a CLI stuck on an error
#: costs at most this.
IDLE_CUTOFF_S = 30 * 60

#: Directories that hold copies, caches or media, never a session's own run directory.
_NOT_RUNS = frozenset({"bundle", "prior_deliveries", "partial_delivery", "model_sessions", "renders",
                       "previews", "catalog", "compile_reports", "vlm_calls", "references", "film",
                       "cache", ".git", "__pycache__"})

#: `usage` fields summed into the token totals.
_TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "total")

_TRACE_TS = re.compile(r'"ts":\s*"([^"]+)"')
_TRANSCRIPT_TS = re.compile(r'"timestamp":\s*"([^"]+)"')


def _ts(value) -> float | None:
    """Seconds since the epoch for an ISO-8601 timestamp (local time when it names no zone), or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _line_ts(line: str, pattern: re.Pattern) -> float | None:
    found = pattern.findall(line)
    return _ts(found[-1]) if found else None


def sessions(run_dir: Path) -> tuple[list[tuple[float, float]], list[str]]:
    """The agent sessions recorded in `run_dir/trace.jsonl`, as `(start, end)` epoch seconds, and notes."""
    trace = Path(run_dir) / "trace.jsonl"
    out: list[tuple[float, float]] = []
    notes: list[str] = []
    if not trace.is_file():
        return out, notes
    start: float | None = None
    last: float | None = None

    def close_open() -> None:
        out.append((start, last if last is not None and last >= start else start))
        notes.append(f"{run_dir.name}: a session has no daemon_shutdown; it ends at its last trace event")

    with trace.open(errors="replace") as fh:
        for line in fh:
            if '"daemon_start"' in line or '"daemon_shutdown"' in line:
                try:
                    rec = json.loads(line)
                except ValueError:
                    rec = {}
                event, when = rec.get("ev"), _ts(rec.get("ts"))
                if event == "daemon_start" and when is not None:
                    if start is not None:
                        close_open()
                    start, last = when, when
                    continue
                if event == "daemon_shutdown" and when is not None and start is not None:
                    out.append((start, max(when, start)))
                    start = last = None
                    continue
            if start is not None:
                when = _line_ts(line, _TRACE_TS)
                if when is not None:
                    last = when
    if start is not None:
        close_open()
    return out, notes


def delivery_time(run_dir: Path, ir_id: str | None = None) -> float | None:
    """When `run_dir` committed the manifest `ir_id` (default: its `ir.json`'s `id`), or None."""
    run_dir = Path(run_dir)
    if ir_id is None:
        try:
            ir_id = json.loads((run_dir / "ir.json").read_text()).get("id")
        except (OSError, ValueError, AttributeError):
            return None
    trace = run_dir / "trace.jsonl"
    if not ir_id or not trace.is_file():
        return None
    found = None
    with trace.open(errors="replace") as fh:
        for line in fh:
            if '"commit"' in line and ir_id in line:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("ev") == "commit" and rec.get("ir_hash") == ir_id:
                    found = _ts(rec.get("ts"))
    return found


def _clip(spans: list[tuple[float, float]], until: float | None) -> list[tuple[float, float]]:
    if until is None:
        return spans
    return [(s, min(e, until)) for s, e in spans if s < until]


def activity(paths) -> list[float]:
    """Sorted epoch seconds of every record in the given trace and transcript files."""
    times: list[float] = []
    for path in paths:
        pattern = _TRACE_TS if path.name == "trace.jsonl" else _TRANSCRIPT_TS
        try:
            with path.open(errors="replace") as fh:
                for line in fh:
                    when = _line_ts(line[:4096] if pattern is _TRANSCRIPT_TS else line, pattern)
                    if when is not None:
                        times.append(when)
        except OSError:
            continue
    times.sort()
    return times


def active_seconds(start: float, end: float, times: list[float], cutoff: float = IDLE_CUTOFF_S) -> float:
    """`end - start`, less what each gap between recorded events in that interval runs past `cutoff`."""
    lo, hi = bisect.bisect_right(times, start), bisect.bisect_left(times, end)
    points = [start, *times[lo:hi], end]
    return sum(min(b - a, cutoff) for a, b in zip(points, points[1:]))


def _meta(run_dir: Path) -> dict:
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _is_session_dir(path: Path, meta: dict) -> bool:
    return bool(meta.get("runtime")) and (path / "trace.jsonl").is_file()


def _run_hours(run_dir: Path, meta: dict, until: float | None = None) -> tuple[float, float, int, list[str]]:
    """Session hours, active hours, and the session count of one run directory up to `until`, and notes."""
    spans, notes = sessions(run_dir)
    if not spans:
        start, end = _ts(meta.get("started_at")), _ts(meta.get("finished_at"))
        if start is None or end is None or end < start:
            return 0.0, 0.0, 0, notes + [f"{run_dir.name}: no session times recorded"]
        spans = [(start, end)]
        notes.append(f"{run_dir.name}: no daemon events; timed from meta.json started_at to finished_at")
    spans = _clip(spans, until)
    if not spans:
        return 0.0, 0.0, 0, notes
    times = activity([run_dir / "trace.jsonl", run_dir / "agent_transcript.jsonl"])
    raw = sum(e - s for s, e in spans)
    active = sum(active_seconds(s, e, times) for s, e in spans)
    return raw / 3600.0, active / 3600.0, len(spans), notes


def _bare_hours(arm_dir: Path, meta: dict) -> tuple[float, float] | None:
    try:
        record = json.loads((arm_dir / "benchmark" / "record.json").read_text())
    except (OSError, ValueError):
        return None
    seconds = record.get("wall_seconds") if isinstance(record, dict) else None
    start = _ts(meta.get("started_at"))
    if not isinstance(seconds, (int, float)):
        return None
    if start is None:
        return seconds / 3600.0, seconds / 3600.0
    times = activity(sorted((arm_dir / "control" / "sessions").rglob("*.jsonl")))
    if not times:
        return seconds / 3600.0, seconds / 3600.0
    return seconds / 3600.0, active_seconds(start, start + seconds, times) / 3600.0


def _tokens(meta: dict) -> dict:
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {k: usage[k] for k in _TOKEN_FIELDS if isinstance(usage.get(k), (int, float))}


def _add(into: dict, more: dict) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


def child_runs(run_dir: Path):
    """Every sub-agent and child run directory below `run_dir`, depth first, each once."""
    top = os.path.realpath(run_dir)
    seen: set[str] = set()
    for root, dirs, _files in os.walk(run_dir, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:                    # reached again through a symlink
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = sorted(d for d in dirs if d not in _NOT_RUNS and not d.startswith("rerender"))
        if real == top:
            continue
        here = Path(root)
        meta = _meta(here)
        if _is_session_dir(here, meta):
            yield here, meta


def _mp4_seconds(path: Path) -> float | None:
    """Duration from an MP4's `moov/mvhd` box, or None."""
    try:
        with path.open("rb") as fh:
            def boxes(start: int, end: int):
                pos = start
                while pos + 8 <= end:
                    fh.seek(pos)
                    size, kind = struct.unpack(">I4s", fh.read(8))
                    header = 8
                    if size == 1:
                        size, header = struct.unpack(">Q", fh.read(8))[0], 16
                    elif size == 0:
                        size = end - pos
                    if size < header:
                        return
                    yield kind, pos + header, pos + size
                    pos += size

            for kind, body, end in boxes(0, path.stat().st_size):
                if kind != b"moov":
                    continue
                for sub, sub_body, _sub_end in boxes(body, end):
                    if sub != b"mvhd":
                        continue
                    fh.seek(sub_body)
                    version = fh.read(4)[0]
                    if version == 1:
                        fh.read(16)
                        scale, duration = struct.unpack(">IQ", fh.read(12))
                    else:
                        fh.read(8)
                        scale, duration = struct.unpack(">II", fh.read(8))
                    return duration / scale if scale else None
    except (OSError, struct.error, IndexError):
        return None
    return None


def film_seconds(run_dir: Path) -> float | None:
    """Length of the film `run_dir` delivered, in seconds, or None."""
    run_dir = Path(run_dir)
    for manifest in (run_dir / "ir.json", run_dir / "bundle" / "ir.uncommitted.json",
                     run_dir / "bundle" / "ir.json"):
        try:
            ir = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(ir, dict) and ir.get("fps") and ir.get("n_frames"):
            return float(ir["n_frames"]) / float(ir["fps"])
    for film in (run_dir / "bundle" / "film" / "clip.mp4", run_dir / "film.mp4"):
        if film.is_file():
            seconds = _mp4_seconds(film)
            if seconds:
                return seconds
    return None


def cost(arm_dir: Path, *, film_seconds_override: float | None = None,
         delivered_ir_id: str | None = None) -> dict | None:
    """Agent-hours per minute of delivered film for a run directory or benchmark arm, or None.

    None when the run recorded no session or delivered no film whose length can be read.
    `film_seconds_override` replaces the film length, and `delivered_ir_id` names the delivered manifest, for a
    film that is not the run directory's current `ir.json`.
    """
    from ..run_layout import internal_dir
    arm_dir = internal_dir(Path(arm_dir))
    meta = _meta(arm_dir)
    notes: list[str] = []
    until = None
    if meta.get("layer") == "bare_agent":
        hours = _bare_hours(arm_dir, meta)
        if hours is None:
            return None
        raw_main, main = hours
        raw_total, total, n_main, n_total = raw_main, main, 1, 1
        tok_main, tok_total = {}, {}
    else:
        if not _is_session_dir(arm_dir, meta):
            return None
        until = delivery_time(arm_dir, delivered_ir_id)
        if delivered_ir_id and until is None:
            notes.append(f"no commit of {delivered_ir_id} in the trace; counted to the run's end")
        raw_main, main, n_main, more = _run_hours(arm_dir, meta, until)
        notes += more
        if not n_main:
            return None
        tok_main = _tokens(meta)
        raw_total, total, n_total, tok_total = raw_main, main, n_main, dict(tok_main)
        for child, child_meta in child_runs(arm_dir):
            raw, active, n, child_notes = _run_hours(child, child_meta, until)
            raw_total, total, n_total = raw_total + raw, total + active, n_total + n
            notes += child_notes
            _add(tok_total, _tokens(child_meta))
    seconds = film_seconds_override or film_seconds(arm_dir)
    if not seconds:
        return None
    minutes = seconds / 60.0
    return {
        "agent_hours_main": round(main, 3),
        "agent_hours_total": round(total, 3),
        "film_minutes": round(minutes, 3),
        "hours_per_film_minute_main": round(main / minutes, 2),
        "hours_per_film_minute_total": round(total / minutes, 2),
        "session_hours_main": round(raw_main, 3),
        "session_hours_total": round(raw_total, 3),
        "n_sessions_main": n_main,
        "n_sessions_total": n_total,
        "tokens_main": tok_main,
        "tokens_total": tok_total,
        "idle_cutoff_minutes": IDLE_CUTOFF_S / 60,
        "counted_until": (datetime.fromtimestamp(until).astimezone().isoformat(timespec="seconds")
                          if until else None),
        "notes": notes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.cost", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories or benchmark arm directories")
    ap.add_argument("--json", action="store_true", help="print one JSON object per run")
    args = ap.parse_args(argv)
    status = 0
    for run in args.runs:
        result = cost(Path(run))
        if args.json:
            print(json.dumps({"run": run, **(result or {"error": "no sessions or no film length"})}))
            continue
        if result is None:
            print(f"{run}: no recorded sessions or no film length")
            status = 1
            continue
        print(f"{run}: {result['hours_per_film_minute_main']} agent-hours per film minute (main), "
              f"{result['hours_per_film_minute_total']} (with sub-agents); "
              f"{result['agent_hours_main']:.2f} h main, {result['agent_hours_total']:.2f} h total over "
              f"{result['n_sessions_total']} sessions, {result['film_minutes']:.2f} min of film")
    return status


if __name__ == "__main__":
    sys.exit(main())
