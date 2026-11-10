"""What code a run ran: the daemon records its checkout's source on `daemon_start`.

`agent/server._source_record` notes the commit the checkout is at and a patch of its uncommitted changes, and the
daemon writes that pair onto the `daemon_start` event every resume path already reads. A tree with no git records
nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from dycraft.agent.server import _source_record

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def test_no_git_records_no_source(tmp_path):
    assert _source_record(tmp_path / "run", repo=tmp_path) is None


_DAEMON = r"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
from dycraft.agent import server

run_dir = Path(sys.argv[1])
class Session:
    ctx = SimpleNamespace(committed=False, run_dir=run_dir)
    handlers = {"ping": lambda: {"ok": True}}
    def trace(self, event, **fields):
        with (run_dir / "trace.jsonl").open("a") as f:
            f.write(json.dumps({"ev": event, **fields}) + "\n")
    def finalize_meta(self, outcome, **fields):
        pass

server.serve(Session(), run_dir)
"""


def test_daemon_start_carries_the_source_of_this_checkout(tmp_path):
    """The record rides on the event every resume path already reads."""
    import sys
    import time

    proc = subprocess.Popen([sys.executable, "-c", _DAEMON, str(tmp_path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    trace = tmp_path / "trace.jsonl"
    try:
        deadline = time.monotonic() + 30
        while not (trace.exists() and "daemon_start" in trace.read_text()):
            assert proc.poll() is None and time.monotonic() < deadline, proc.stderr.read()
            time.sleep(0.05)
    finally:
        proc.terminate()
        proc.communicate(timeout=10)
    start = next(json.loads(line) for line in trace.read_text().splitlines()
                 if json.loads(line)["ev"] == "daemon_start")
    source = start["source"]
    assert len(source["base"]) == 40
    assert (tmp_path / "source" / f"{source['patch']}.patch.gz").is_file()
