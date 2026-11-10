"""Real staged client and daemon RPC; local sockets, no model calls."""
import importlib.util
import os
import subprocess
import sys
import time

import pytest

from dycraft.agent import stage_harness_client
from dycraft.tests._animation_fixtures import plan


@pytest.mark.expensive
def test_the_animation_daemon_registers_its_client(tmp_path):
    harness = stage_harness_client(tmp_path, "animation")
    env = dict(os.environ, DYCRAFT_LAYER="animation", DYCRAFT_AGENT_PID="",
               DYCRAFT_USER_PROMPT=plan()["intent"])
    with (tmp_path / "daemon.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "dycraft.agent.layers.server", "serve", str(tmp_path)],
                                env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / ".harness.port").exists() and proc.poll() is None:
                assert time.monotonic() < deadline, "daemon startup timed out"
                time.sleep(.05)
            assert proc.poll() is None, (tmp_path / "daemon.log").read_text()
            spec = importlib.util.spec_from_file_location("animation_test_harness", harness)
            client = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(client)
            submitted = client.plan(plan())
            assert submitted["ok"]
            deadline = time.monotonic() + 5
            while True:
                record = client.job_result(submitted["job_id"])
                if record["status"] not in ("queued", "running"):
                    break
                assert time.monotonic() < deadline
                time.sleep(.05)
            assert record["result"]["ok"], record
            assert (tmp_path / "native/work/plan.json").exists()
            assert (tmp_path / "source_reference.md").exists()
        finally:
            proc.terminate()
            proc.wait(timeout=10)
