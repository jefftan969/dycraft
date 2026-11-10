"""`python -m dycraft.cli` refuses a prompt it cannot use before it starts anything.

An empty or blank `--prompt`, or a `--prompt` naming a file that is not there, exits with status 2 and creates no
run directory, on every layer. Otherwise a launcher whose prompt file went missing hands the CLI an empty brief, and
the run authors a film nobody asked for.
"""
from __future__ import annotations

import os
import subprocess
import sys


def test_cli_refuses_an_empty_prompt_on_every_layer(pkgroot, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(pkgroot)}
    for layer, prompt in (("animation", ""), ("animation", " \n\t"),
                          ("plain", ""), ("object", "   ")):
        proc = subprocess.run(
            [sys.executable, "-m", "dycraft.cli", "--layer", layer,
             "--backend", "codex", "--runs-dir", str(tmp_path),
             "--run-id", "r", "--prompt", prompt],
            capture_output=True, text=True, timeout=120, env=env)
        assert proc.returncode == 2, (layer, proc.stderr[-500:])
        assert "--prompt is empty" in proc.stderr
    assert not (tmp_path / "r").exists()


def test_cli_refuses_a_prompt_file_that_is_not_there(pkgroot, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "dycraft.cli", "--layer", "plain",
         "--backend", "codex", "--runs-dir", str(tmp_path), "--run-id", "r",
         "--prompt", str(tmp_path / "brief.md")],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(pkgroot)})
    assert proc.returncode == 2
    assert f"--prompt {tmp_path / 'brief.md'}: no such file (relative to " in proc.stderr
    assert "Pass the brief's text in quotes, or the path to an existing file." in proc.stderr
