"""The `h.ik` child's setup file and the runner hook that runs it."""
from __future__ import annotations

import importlib.machinery
import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "animation" / "sim" / "_sim_runner.py"
SETUP = ROOT / "animation" / "ik" / "_ik_setup.py"


def _run(tmp_path, code: str, setup: str | None) -> tuple[dict, str]:
    # Not `code.py`: the child's cwd is on its path, and a file of that name
    # would stand in for the stdlib's `code`, which importing torch reaches.
    (tmp_path / "ik_code.py").write_text(code)
    argv = [sys.executable, str(RUNNER), "--code", str(tmp_path / "ik_code.py"),
            "--result", str(tmp_path / "result.json"), "--codename", "ik_code.py"]
    if setup is not None:
        (tmp_path / "setup.py").write_text(setup)
        argv += ["--setup", str(tmp_path / "setup.py")]
    proc = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    return json.loads((tmp_path / "result.json").read_text()), proc.stderr


def test_setup_runs_first_in_its_own_namespace(tmp_path):
    out, _ = _run(tmp_path, "import os\nresult['seen'] = os.environ.get('SETUP_RAN')\n"
                  "result['leaked'] = 'marker' in globals()\n",
                  "import os\nos.environ['SETUP_RAN'] = 'yes'\nmarker = 1\n")
    assert out == {"ok": True, "result": {"seen": "yes", "leaked": False}}


def test_a_failing_setup_never_breaks_the_run(tmp_path):
    out, err = _run(tmp_path, "result['x'] = 1\n", "raise RuntimeError('no gpu')\n")
    assert out == {"ok": True, "result": {"x": 1}}
    assert "setup" in err and "no gpu" in err


def test_the_authored_code_keeps_its_line_numbers(tmp_path):
    out, _ = _run(tmp_path, "a = 1\nb = 2\nraise ValueError('here')\n", "x = 1\ny = 2\n")
    assert out["ok"] is False
    assert 'File "ik_code.py", line 3' in out["traceback"]


def test_the_ik_op_passes_its_setup():
    import importlib
    ik = importlib.import_module("dycraft.animation.ik.ik")
    assert ik._SETUP_PATH == SETUP and SETUP.is_file()


def _fake(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.mark.parametrize("capability, disabled", [
    ((6, 1), True),     # Pascal: cuBLASDx has no kernels, use Warp's own
    ((7, 0), False),    # Volta and up: unchanged
    ((12, 0), False),
])
def test_pre_volta_gpus_turn_off_warps_mathdx_paths(monkeypatch, capability, disabled):
    cuda = types.SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda i: capability)
    config = types.SimpleNamespace(enable_mathdx_gemm=True, enable_mathdx_solver=True)
    monkeypatch.setitem(sys.modules, "torch", _fake("torch", cuda=cuda))
    monkeypatch.setitem(sys.modules, "warp", _fake("warp", config=config))
    runpy.run_path(str(SETUP), run_name="__dycraft_setup__")
    assert config.enable_mathdx_gemm is (not disabled)
    assert config.enable_mathdx_solver is (not disabled)


def test_with_no_cuda_device_the_setup_refuses_and_nothing_runs(tmp_path):
    """The backstop for a GPU hidden or lost after launch: the call reports
    `unavailable` instead of running into cuRobo's CUDA traceback."""
    # The child's cwd comes first on its path, so this torch is the one it finds.
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text(
        "class cuda:\n    is_available = staticmethod(lambda: False)\n")
    out, _ = _run(tmp_path, "result['ran'] = True\n", SETUP.read_text())
    assert out["ok"] is False and out["error_kind"] == "unavailable"
    assert "result" not in out


def test_a_setup_refusal_needs_an_error_kind(tmp_path):
    """Only an error that names its kind refuses the call; any other setup
    failure is reported and the code still runs."""
    out, _ = _run(tmp_path, "result['x'] = 1\n",
                  "class Refused(Exception):\n    error_kind = 'busy'\nraise Refused('later')\n")
    assert out == {"ok": False, "error_kind": "busy", "message": "later"}


def test_the_real_setup_runs_cleanly_in_this_interpreter(tmp_path):
    """Whatever this interpreter has (torch, warp, a GPU, or none of them),
    the setup either runs the code or, with torch and no CUDA device, refuses
    it as unavailable; it never fails, and never touches Warp from sm_70 up."""
    import importlib.util
    out, err = _run(tmp_path, "import sys\nresult['ran'] = True\n", SETUP.read_text())
    no_gpu = False
    if importlib.util.find_spec("torch") is not None:
        import torch
        no_gpu = not torch.cuda.is_available()
    if no_gpu:
        assert out["ok"] is False and out["error_kind"] == "unavailable"
    else:
        assert out == {"ok": True, "result": {"ran": True}}
    assert "[runner] setup" not in err
