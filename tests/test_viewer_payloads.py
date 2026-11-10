"""The trajectory and agent timeline, built apart from the server and served as stored JSON (`viewer/_payload_pass`).

Building them in the server held the interpreter lock against everything else it serves, a film's meshes included.
These pin that the stored bytes are what the routes used to send, that they are reused until a file they are read
from changes, and that a run still going gets a fresh timeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dycraft.viewer import _payload_pass as PP


def _run(root: Path, *, finished: bool = True) -> Path:
    run = root / "run1"
    run.mkdir(parents=True)
    meta = {"run_id": "run1", "layer": "object", "model": "test-model", "prompt": "A box.",
            "started_at": "2026-01-01T00:00:00+00:00"}
    if finished:
        meta.update(outcome="committed", finished_at="2026-01-01T00:10:00+00:00")
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "trace.jsonl").write_text("")
    return run


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_VIEWER_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_the_stored_payload_is_what_the_route_sent_and_is_built_in_its_own_interpreter(tmp_path, cache):
    run = _run(tmp_path)

    def same(kind):
        # The timeline stamps the moment it was built (`now`); everything else is the route's output, byte for byte.
        got, here = json.loads(PP.cached_payload(kind, run)), json.loads(PP._build_here(kind, run))
        got.pop("now", None)
        here.pop("now", None)
        return got == here

    assert all(same(kind) for kind in PP.KINDS)
    assert PP.cached_payload("trajectory", run) == PP._build_here("trajectory", run)
    assert len(list((cache / "payloads").glob("*.json"))) == 2


def test_a_payload_is_reused_until_a_file_it_is_read_from_changes(tmp_path, cache, monkeypatch):
    run = _run(tmp_path)
    built = []
    monkeypatch.setattr(PP, "_build_apart", lambda kind, run_dir: built.append(kind) or PP._build_here(kind, run_dir))
    first = PP.cached_payload("trajectory", run)
    assert PP.cached_payload("trajectory", run) == first and built == ["trajectory"]
    meta = json.loads((run / "meta.json").read_text())
    (run / "meta.json").write_text(json.dumps({**meta, "prompt": "A taller box."}))
    st = (run / "meta.json").stat()
    os.utime(run / "meta.json", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    PP.cached_payload("trajectory", run)
    assert built == ["trajectory", "trajectory"]


def test_a_run_still_going_gets_a_fresh_timeline_every_time(tmp_path, cache, monkeypatch):
    run = _run(tmp_path, finished=False)
    built = []
    monkeypatch.setattr(PP, "_build_apart", lambda kind, run_dir: built.append(kind) or PP._build_here(kind, run_dir))
    PP.cached_payload("agent_timeline", run)
    PP.cached_payload("agent_timeline", run)
    assert built == ["agent_timeline", "agent_timeline"]


def test_when_its_interpreter_cannot_run_the_payload_is_built_in_the_server(tmp_path, cache, monkeypatch):
    run = _run(tmp_path)

    def refuse(kind, run_dir):
        raise RuntimeError("no interpreter")

    monkeypatch.setattr(PP, "_build_apart", refuse)
    assert PP.cached_payload("trajectory", run) == PP._build_here("trajectory", run)


def test_the_routes_serve_the_stored_json(tmp_path, cache):
    from fastapi.testclient import TestClient

    from dycraft.viewer.app import create_app
    from dycraft.viewer.trajectory import build as build_trajectory

    runs = tmp_path / "runs"
    run = _run(runs)
    client = TestClient(create_app(runs_dir=runs, dist_dir=Path("/nonexistent")))
    got = client.get("/api/runs/run1/trajectory")
    assert got.status_code == 200 and got.headers["content-type"].startswith("application/json")
    assert got.json() == json.loads(PP._encode(build_trajectory(run)))
    assert client.get("/api/runs/run1/agent_timeline").json()["events"] == []


def test_a_build_that_times_out_is_a_503_and_never_rebuilt_in_the_server(tmp_path, cache, monkeypatch):
    """Building it in the server after the child ran out of time would hold the server's interpreter lock for as long
    again, which is what building apart is for. The route answers 503, and this server does not try again."""
    import subprocess

    from fastapi.testclient import TestClient

    from dycraft.viewer.app import create_app

    runs = tmp_path / "runs"
    _run(runs)
    calls = []

    def times_out(kind, run_dir):
        calls.append("apart")
        raise subprocess.TimeoutExpired("child", PP.TIMEOUT_S)

    monkeypatch.setattr(PP, "_TIMED_OUT", set())         # this test's, so what it remembers goes with it
    monkeypatch.setattr(PP, "_build_apart", times_out)
    monkeypatch.setattr(PP, "_build_here", lambda *a: calls.append("here") or b"{}")
    client = TestClient(create_app(runs_dir=runs, dist_dir=Path("/nonexistent")))
    for _ in range(2):
        got = client.get("/api/runs/run1/trajectory")
        assert got.status_code == 503 and "longer than" in got.json()["detail"]
    assert calls == ["apart"], calls
