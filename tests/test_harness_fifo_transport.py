"""The named-pipe transport, which is what removes the sandbox escalation.

Codex's workspace-write sandbox blocks every socket connection from inside it
-- TCP loopback, AF_UNIX and even `socketpair()` all return EPERM, so it is a
syscall filter rather than an address rule. Over a socket every harness call
would have to be escalated, and every escalation is a full turn of a second
model session adjudicating it. A FIFO is ordinary file I/O in a directory the
sandbox already grants write on.

These tests run the real daemon in a subprocess and the real staged client, with
the socket path deliberately sabotaged, so anything that answers proves the pipe
carried it.
"""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

DAEMON = '''
import sys, time, pathlib
from dycraft.agent import server as srv
run = pathlib.Path(sys.argv[1]); run.mkdir(parents=True, exist_ok=True)
# Short, so a reply write still bounded by the *open* deadline -- the defect
# the large-reply test pins -- fails there deterministically rather than only
# on a loaded box. A live client opens its reply pipe within microseconds.
srv.REPLY_OPEN_TIMEOUT_S = 0.5
class Ctx: committed = False
class Session:
    ctx = Ctx()
    def __init__(self):
        self.handlers = {
            "plan": lambda **kw: {"ok": True, "echo": kw},
            "blender": lambda **kw: {"ok": True, "len": len(kw.get("code") or "")},
            "boom": lambda **kw: (_ for _ in ()).throw(ValueError("kaboom")),
            "slow": lambda **kw: (time.sleep(kw.get("s", 0.4)), {"ok": True})[1],
            "big": lambda **kw: {"ok": True, "blob": "x" * kw["n"]},
        }
    def trace(self, *a, **k): pass
    def finalize_meta(self, *a, **k): pass
srv.serve(Session(), run)
'''


@pytest.fixture
def harness(tmp_path):
    from dycraft.agent import stage_harness_client
    run = tmp_path / "run"
    script = tmp_path / "daemon.py"
    script.write_text(DAEMON)
    proc = subprocess.Popen([sys.executable, str(script), str(run)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(200):
            if (run / ".harness.fifo").exists() and (run / ".harness.port").exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail("daemon never published its transports")
        stage_harness_client(run, "animation")
        spec = importlib.util.spec_from_file_location("staged_h", run / "harness.py")
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        # Sabotage the socket rung: whatever still answers came over the pipe.
        h._read_port = lambda: pytest.fail("the socket path must not be used")
        yield h, run
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_daemon_publishes_a_fifo_and_the_client_prefers_it(harness):
    h, run = harness
    assert (run / ".harness.fifo").is_fifo()
    assert h._fifo_available()
    assert h.plan({"intent": "x"}, batch=False)["echo"]["plan"] == {"intent": "x"}


def test_payload_far_larger_than_pipe_buf_round_trips(harness):
    """Only the call id goes down the shared pipe; the payload is a file.

    A pipe write is atomic only to PIPE_BUF (4096 bytes) and a `blender()` call
    carrying a page of author Python is bigger, so a payload on the shared pipe
    would let two concurrent callers interleave mid-request.
    """
    h, _ = harness
    assert h.blender("x" * 200_000, batch=False)["len"] == 200_000


def test_replies_are_per_call_so_the_pool_stays_concurrent(harness):
    """A single shared return pipe would serialise a 32-thread daemon."""
    h, _ = harness
    started = time.time()
    threads = [threading.Thread(target=lambda: h._call("slow", {"s": 0.4}))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started
    assert elapsed < 1.0, f"{elapsed:.2f}s for 3x0.4s — replies are serialising"


def test_failures_behave_as_they_do_on_the_socket_path(harness):
    h, run = harness
    with pytest.raises(SystemExit):
        h._call("boom", {})
    with pytest.raises(SystemExit):
        h._call("nope", {})
    # Neither path may leave request payloads or reply pipes behind.
    assert not list((run / ".harness.calls").iterdir())


def test_a_reply_far_larger_than_the_pipe_arrives_whole_and_in_linear_time(harness):
    """A storyboard result can run past a hundred megabytes.

    A write bounded by the 30 s a caller has to *open* its reply pipe, or a
    client that re-copies its buffer on every 64 KB read, would abandon such a
    reply mid-frame under load. A write is bounded only by stalls, and the
    client reads in linear time.
    """
    h, _ = harness
    size = 128 * 1024 * 1024
    started = time.time()
    reply = h._call("big", {"n": size})
    elapsed = time.time() - started
    assert len(reply["blob"]) == size
    # Far beyond the daemon's 0.5 s open deadline, which the write outlives.
    assert elapsed < 20.0, f"{elapsed:.1f}s to move {size >> 20} MB"
