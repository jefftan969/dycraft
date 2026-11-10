"""`h.ik` without a GPU: one decision per launch, the op's reply, the prompt and the staging that follow it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.agent import agent_runner
import importlib
ik_mod = importlib.import_module("dycraft.animation.ik.ik")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    """No launch decision and no IK interpreter override inherited from the shell running the tests."""
    for name in (ik_mod.UNAVAILABLE_ENV, "DYCRAFT_IK_PYTHON", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _smi(tmp_path, monkeypatch, listing: str | None):
    """Put an `nvidia-smi` on PATH that prints `listing`, or none at all."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if listing is not None:
        smi = bin_dir / "nvidia-smi"
        smi.write_text(f"#!/bin/sh\nprintf '%s' '{listing}'\n")
        smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))      # nothing else: this machine may have an nvidia-smi of its own


@pytest.mark.parametrize("devices, why", [("", "CUDA_VISIBLE_DEVICES is empty"),
                                          ("-1", "CUDA_VISIBLE_DEVICES is -1"),
                                          ("NoDevFiles", "CUDA_VISIBLE_DEVICES is NoDevFiles")])
def test_a_cuda_visible_devices_that_hides_every_gpu_is_decisive(clean_env, tmp_path, devices, why):
    _smi(tmp_path, clean_env, "GPU 0: NVIDIA TITAN RTX (UUID: GPU-1)\n")
    clean_env.setenv("CUDA_VISIBLE_DEVICES", devices)
    assert ik_mod.gpu_visible() == (False, why)


@pytest.mark.parametrize("listing, expected", [
    (None, (False, "nvidia-smi is not installed")),
    ("", (False, "nvidia-smi lists no GPU")),
    ("No devices were found\n", (False, "nvidia-smi lists no GPU")),
    ("GPU 0: NVIDIA TITAN RTX (UUID: GPU-1)\n", (True, "")),
])
def test_otherwise_nvidia_smi_has_to_list_a_gpu(clean_env, tmp_path, listing, expected):
    _smi(tmp_path, clean_env, listing)
    assert ik_mod.gpu_visible() == expected


def test_availability_names_why(clean_env, tmp_path):
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: None)
    assert ik_mod.availability()["reason"] == ik_mod.NOT_INSTALLED
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    _smi(tmp_path, clean_env, None)
    assert ik_mod.availability() == {"available": False, "reason": ik_mod.NO_GPU,
                                     "detail": "nvidia-smi is not installed"}
    _smi(tmp_path, clean_env, "GPU 0: NVIDIA TITAN RTX (UUID: GPU-1)\n")
    assert ik_mod.availability()["available"] is True
    clean_env.setenv(ik_mod.UNAVAILABLE_ENV, ik_mod.NO_GPU)
    assert ik_mod.availability()["reason"] == ik_mod.NO_GPU


def test_the_launch_pins_its_decision_for_everything_it_starts(clean_env):
    resolve = ik_mod._resolve_ik_python
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "")
    env = {}
    found = ik_mod.decide_for_launch(env)
    assert found == {"available": False, "reason": ik_mod.NO_GPU, "detail": "CUDA_VISIBLE_DEVICES is empty"}
    assert env == {"DYCRAFT_IK_PYTHON": "", ik_mod.UNAVAILABLE_ENV: ik_mod.NO_GPU}
    assert ik_mod.launch_line(found) == (
        "[dycraft] IK (cuRobo): unavailable — no NVIDIA GPU is visible (CUDA_VISIBLE_DEVICES is empty). "
        "The agent solves kinematics itself; nothing else changes.")
    # With the decision pinned, the interpreter no longer resolves, and the reason survives.
    for name, value in env.items():
        clean_env.setenv(name, value)
    assert resolve() is None
    clean_env.delenv("CUDA_VISIBLE_DEVICES")
    assert ik_mod.availability()["reason"] == ik_mod.NO_GPU


def test_an_available_launch_changes_nothing(clean_env, tmp_path):
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    _smi(tmp_path, clean_env, "GPU 0: NVIDIA TITAN RTX (UUID: GPU-1)\n")
    env = {}
    assert ik_mod.decide_for_launch(env)["available"] is True
    assert env == {} and ik_mod.launch_line({"available": True}) is None


def test_the_op_says_so_without_starting_a_process(clean_env, tmp_path):
    """No GPU visible: `unavailable` at once, in the agent-facing words, and the interpreter never runs."""
    marker = tmp_path / "started"
    stub = tmp_path / "python"
    stub.write_text(f"#!/bin/sh\ntouch {marker}\nexec {sys.executable} \"$@\"\n")
    stub.chmod(0o755)
    clean_env.setenv("DYCRAFT_IK_PYTHON", str(stub))
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "")
    out = ik_mod.ik("result['x'] = 1", ik_dir=tmp_path / "ik", require_scene=False)
    assert out == {"ok": False, "error_kind": "unavailable", "message": (
        "`h.ik` is not available on this machine: cuRobo needs an NVIDIA GPU and none is visible. Solve the joints "
        "with your own solvers, as the `humanoid_motion` skill describes; the rest of the animation layer works as "
        "usual.")}
    assert not marker.exists()
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: None)
    clean_env.delenv("CUDA_VISIBLE_DEVICES")
    assert ik_mod.ik("", ik_dir=tmp_path / "ik", require_scene=False)["message"].startswith(
        "`h.ik` is not available on this machine: cuRobo is not installed. ")


def test_a_gpu_lost_after_launch_is_unavailable_too(clean_env, tmp_path):
    """The child's backstop: the launch saw a GPU, torch does not. Told in the op's words, not cuRobo's."""
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    clean_env.setattr(ik_mod, "gpu_visible", lambda: (True, ""))
    ik_dir = tmp_path / "ik"
    (ik_dir / "torch").mkdir(parents=True)        # the child's cwd comes first on its path
    (ik_dir / "torch" / "__init__.py").write_text("class cuda:\n    is_available = staticmethod(lambda: False)\n")
    out = ik_mod.ik("result['x'] = 1", ik_dir=ik_dir, require_scene=False)
    assert out["ok"] is False and out["error_kind"] == "unavailable"
    assert out["message"] == ik_mod.unavailable_message(ik_mod.NO_GPU)
    assert "result" not in out


def _render(monkeypatch, available: bool) -> str:
    monkeypatch.setattr(ik_mod, "is_available", lambda: available)
    return agent_runner.compose_system_prompt("animation")


def test_with_ik_the_prompt_is_unchanged_word_for_word(monkeypatch):
    """The blocks only add markers to the file: rendered with IK, it is the prompt of the commit before them."""
    before = subprocess.run(["git", "show", "2be2716c:agent/prompts/animation.md"], cwd=ROOT,
                            capture_output=True, text=True)
    if before.returncode != 0:
        pytest.skip("not a git checkout with 2be2716c")
    assert _render(monkeypatch, True) == agent_runner.render_prompt(before.stdout)


def test_without_ik_only_the_three_ik_places_change(monkeypatch):
    with_ik = _render(monkeypatch, True).splitlines()
    without = _render(monkeypatch, False).splitlines()
    assert "h.ik(" not in "\n".join(without)
    removed = [line for line in with_ik if line not in without]
    added = [line for line in without if line not in with_ik]
    assert len(removed) == 3 and len(added) == 2
    assert removed[1].startswith("| `h.ik(")
    assert added == [
        "For a humanoid robot or any other articulated robot, solve the joints with a solver of your own "
        "(see Kinematics below).",
        "**Kinematics.** `h.ik` (cuRobo) is not available in this run. Solve a robot's "
        "joints with your own solvers, as the `humanoid_motion` skill describes. Blender's IK constraints do not "
        "avoid obstacles, so check clearances yourself.",
    ]


def test_an_else_branch_replaces_a_block_and_keeps_the_paragraphs(monkeypatch):
    monkeypatch.setitem(agent_runner.OPTIONAL, "probe", lambda: held)
    text = "A.\n\n<!-- probe -->\nwith\n<!-- else -->\nwithout\n<!-- /probe -->\n\nB.\n"
    held = True
    assert agent_runner.render_prompt(text) == "A.\n\nwith\n\nB.\n"
    held = False
    assert agent_runner.render_prompt(text) == "A.\n\nwithout\n\nB.\n"
    held = False
    assert agent_runner.render_prompt("A.\n\n<!-- probe -->\nwith\n<!-- /probe -->\n\nB.\n") == "A.\n\nB.\n"


def test_the_ik_docs_are_staged_only_with_ik(monkeypatch, tmp_path):
    from dycraft.agent import stage_harness_client
    import dycraft.agent as agent_pkg
    for available in (False, True):
        run = tmp_path / str(available)
        run.mkdir()
        monkeypatch.setattr(agent_pkg, "_ik_available", lambda: available)
        stage_harness_client(run, "animation")
        assert (run / "references" / "ik" / "README.md").is_file() is available


def test_the_animation_op_answers_before_evaluating_anything(clean_env, tmp_path):
    """`h.ik(frames=...)` with no IK: no Blender evaluation, no export, the same reply as the op's."""
    from dycraft.animation.blender import solvers

    class Service:
        root = tmp_path

        def worker(self, *a, **k):
            raise AssertionError("evaluated a frame for a call that cannot run")

        def version_of(self, workspace):
            raise AssertionError("read the project for a call that cannot run")

    clean_env.setenv("CUDA_VISIBLE_DEVICES", "")
    clean_env.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    out = solvers.consult(Service(), "ik", {"code": "result['x'] = 1", "frames": [0, 10]}, tmp_path, tmp_path, None)
    assert out == {"ok": False, "error_kind": "unavailable", "message": ik_mod.unavailable_message(ik_mod.NO_GPU)}
    assert not (tmp_path / ik_mod.IK_DIR).exists()
