"""`benchmark/cost.py`: agent-hours per minute of delivered film, from a run directory's own records.

Each test builds a small run tree with known session times, so the expected hours are exact.
"""
import json
import struct
from datetime import datetime, timedelta, timezone

import pytest

from dycraft.benchmark import cost as C

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def at(hours: float) -> str:
    return (T0 + timedelta(hours=hours)).isoformat()


def write_run(run_dir, sessions, *, events=(), commits=(), layer="animation", usage=None,
              film=None, ir_id="ir-final", shutdown=True):
    """A run directory: `sessions` are (start_h, end_h) daemon lifetimes; `events` extra trace times (hours)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (start, end) in enumerate(sessions):
        lines.append({"epoch": i, "ev": "daemon_start", "ts": at(start)})
        lines += [{"epoch": i, "ev": "op_call", "ts": at(t)} for t in events if start < t < end]
        lines += [{"epoch": i, "ev": "commit", "ir_hash": h, "ts": at(t)} for h, t in commits
                  if start < t < end]
        if shutdown or i < len(sessions) - 1:
            lines.append({"epoch": i, "ev": "daemon_shutdown", "ts": at(end)})
        else:
            lines.append({"epoch": i, "ev": "op_result", "ts": at(end)})
    (run_dir / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    meta = {"layer": layer, "runtime": "codex_cli", "started_at": at(sessions[0][0])}
    if usage:
        meta["usage"] = usage
    (run_dir / "meta.json").write_text(json.dumps(meta))
    if film:
        (run_dir / "ir.json").write_text(json.dumps({"id": ir_id, "fps": 24, "n_frames": int(film * 24)}))
    return run_dir


def dense(start, end, step=0.25):
    """Event times every `step` hours, so no stretch is idle."""
    n = int(round((end - start) / step))
    return [start + step * k for k in range(1, n)]


def test_resumed_sessions_are_summed_without_the_time_between_them(tmp_path):
    run = write_run(tmp_path / "run", [(0, 2), (5, 6)], events=dense(0, 2) + dense(5, 6), film=60)
    r = C.cost(run)
    assert r["agent_hours_main"] == pytest.approx(3.0)
    assert r["n_sessions_main"] == 2
    assert r["film_minutes"] == pytest.approx(1.0)
    assert r["hours_per_film_minute_main"] == pytest.approx(3.0)


def test_a_session_without_a_shutdown_ends_at_its_last_event(tmp_path):
    run = write_run(tmp_path / "run", [(0, 1.5)], events=dense(0, 1.5), film=60, shutdown=False)
    r = C.cost(run)
    assert r["session_hours_main"] == pytest.approx(1.5)
    assert any("no daemon_shutdown" in n for n in r["notes"])


def test_idle_time_beyond_the_cutoff_is_not_agent_time(tmp_path):
    # Active for the first hour, then silent for nine hours (a CLI stuck on a usage limit) until shut down.
    run = write_run(tmp_path / "run", [(0, 10)], events=dense(0, 1.25), film=60)  # last event at 1 h
    r = C.cost(run)
    assert r["session_hours_main"] == pytest.approx(10.0)
    assert r["agent_hours_main"] == pytest.approx(1.0 + C.IDLE_CUTOFF_S / 3600)


def test_transcript_records_count_as_activity(tmp_path):
    run = write_run(tmp_path / "run", [(0, 3)], film=60)
    (run / "agent_transcript.jsonl").write_text(
        "".join(json.dumps({"timestamp": at(t).replace("+00:00", "Z"), "type": "response_item"}) + "\n"
                for t in dense(0, 3)))
    assert C.cost(run)["agent_hours_main"] == pytest.approx(3.0)


def test_total_adds_every_sub_agent_and_child_run_once(tmp_path):
    run = write_run(tmp_path / "run", [(0, 4)], events=dense(0, 4), film=120,
                    usage={"input": 10, "output": 1, "total": 11})
    write_run(run / "sub_runs" / "scene_rewrite", [(0, 0.25)], usage={"input": 5, "output": 2, "total": 7})
    child = write_run(run / "contributions" / "walk", [(1, 2)], events=dense(1, 2))
    write_run(child / "native" / "jobs" / "j1" / "sub_runs" / "review", [(1.5, 1.75)])
    # Copies and caches are not runs, and a run reached twice through a symlink counts once.
    write_run(run / "bundle" / "contributions" / "walk", [(1, 2)])
    (run / "contributions" / "walk_again").symlink_to(child, target_is_directory=True)
    r = C.cost(run)
    assert r["agent_hours_main"] == pytest.approx(4.0)
    assert r["agent_hours_total"] == pytest.approx(4.0 + 0.25 + 1.0 + 0.25)
    assert r["n_sessions_total"] == 4
    assert r["hours_per_film_minute_total"] == pytest.approx(5.5 / 2)
    assert r["tokens_main"] == {"input": 10, "output": 1, "total": 11}
    assert r["tokens_total"] == {"input": 15, "output": 3, "total": 18}


def test_time_after_the_delivered_commit_is_not_counted(tmp_path):
    # Delivered at 3 h; the run was reopened for another round (and a child) that did not replace it.
    run = write_run(tmp_path / "run", [(0, 4), (10, 12)], events=dense(0, 4) + dense(10, 12),
                    commits=[("ir-final", 3.0), ("ir-later", 11.0)], film=60)
    write_run(run / "contributions" / "late", [(10.5, 11.5)], events=dense(10.5, 11.5))
    write_run(run / "contributions" / "early", [(1, 2)], events=dense(1, 2))
    r = C.cost(run)                                         # ir.json names ir-final
    assert r["agent_hours_main"] == pytest.approx(3.0)
    assert r["agent_hours_total"] == pytest.approx(4.0)
    assert r["counted_until"] is not None
    r = C.cost(run, delivered_ir_id="ir-later")
    assert r["agent_hours_main"] == pytest.approx(4.0 + 1.0)


def test_an_uncommitted_run_counts_to_its_end(tmp_path):
    run = write_run(tmp_path / "run", [(0, 2)], events=dense(0, 2), film=60)
    assert C.cost(run)["agent_hours_main"] == pytest.approx(2.0)
    assert C.cost(run)["counted_until"] is None


def mp4(seconds: float, timescale: int = 1000) -> bytes:
    mvhd_body = struct.pack(">B3xII", 0, 0, 0) + struct.pack(">II", timescale, int(seconds * timescale))
    mvhd_body += b"\0" * 80
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom\0\0\0\1"
    mdat = struct.pack(">I4s", 12, b"mdat") + b"\0" * 4
    return ftyp + mdat + moov


def test_film_length_falls_back_to_the_film_file(tmp_path):
    run = write_run(tmp_path / "run", [(0, 1)], events=dense(0, 1))
    (run / "bundle" / "film").mkdir(parents=True)
    (run / "bundle" / "film" / "clip.mp4").write_bytes(mp4(45.0))
    assert C.film_seconds(run) == pytest.approx(45.0)
    assert C.cost(run)["film_minutes"] == pytest.approx(0.75)


def test_no_film_or_no_sessions_gives_none(tmp_path):
    assert C.cost(write_run(tmp_path / "nofilm", [(0, 1)])) is None
    (tmp_path / "empty").mkdir()
    assert C.cost(tmp_path / "empty") is None


def test_a_bare_arm_is_one_session_timed_by_the_runner(tmp_path):
    arm = tmp_path / "bare_codex"
    (arm / "benchmark").mkdir(parents=True)
    start = datetime(2026, 10, 1, 9, 0)                      # the runner writes local time
    (arm / "meta.json").write_text(json.dumps({"layer": "bare_agent", "runtime": "codex",
                                               "started_at": start.isoformat()}))
    (arm / "benchmark" / "record.json").write_text(json.dumps({"wall_seconds": 7200}))
    sessions = arm / "control" / "sessions" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text("".join(
        json.dumps({"timestamp": (start + timedelta(minutes=m)).astimezone(timezone.utc).isoformat()}) + "\n"
        for m in range(0, 61, 10)))                          # active for an hour, then silent
    (arm / "ir.json").write_text(json.dumps({"fps": 20, "n_frames": 1200}))
    r = C.cost(arm)
    assert r["session_hours_main"] == r["session_hours_total"] == pytest.approx(2.0)
    assert r["agent_hours_main"] == r["agent_hours_total"] == pytest.approx(1.5)
    assert r["n_sessions_total"] == 1


def test_the_command_prints_one_line_per_run(tmp_path, capsys):
    run = write_run(tmp_path / "run", [(0, 2)], events=dense(0, 2), film=60)
    assert C.main([str(run)]) == 0
    assert "2.0 agent-hours per film minute (main)" in capsys.readouterr().out
    assert C.main([str(tmp_path / "missing"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["error"]
