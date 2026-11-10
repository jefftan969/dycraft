"""A job result holding numpy values comes back as plain numbers, with the coercion recorded.

Probes and edits often return `np.isclose(...)` or a float32 norm. `worker._plain` turns them into Python values and
lists each coercion, as it does for Blender containers, instead of failing the job after its work is done.
`worker.py` imports `bpy`, so this runs in the Blender interpreter.
"""
from pathlib import Path
import subprocess

import pytest

WORKER = Path(__file__).resolve().parents[1] / "animation" / "blender" / "worker.py"

SCRIPT = r'''
import importlib.util, json, math
import numpy as np
spec = importlib.util.spec_from_file_location("plain_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)

coerced, nonfinite = [], []
value = {"close": np.isclose(1.0, 1.0), "norm": np.float32(0.5), "count": np.int64(3),
         "f64": np.float64(0.25), "grid": np.arange(4, dtype=np.int64).reshape(2, 2),
         "bad": np.float32("nan")}
plain = worker._plain(value, "result", coerced, nonfinite)
assert plain["close"] is True and plain["norm"] == 0.5 and plain["count"] == 3, plain
assert plain["f64"] == 0.25 and plain["grid"] == [[0, 1], [2, 3]], plain
assert all(isinstance(v, (bool, int, float, list)) and not isinstance(v, np.generic) or k == "f64"
           for k, v in plain.items()), plain   # np.float64 is a float subclass and passes as it is
assert math.isnan(plain["bad"]) and nonfinite == ["result.bad"], nonfinite
assert set(coerced) == {"result.close (bool_ -> bool)", "result.norm (float32 -> float)",
                        "result.count (int64 -> int)", "result.grid (ndarray -> list)",
                        "result.bad (float32 -> float)"}, coerced
json.dumps({k: v for k, v in plain.items() if k != "bad"}, allow_nan=False)
print("ok")
'''


def test_numpy_values_become_plain_and_are_recorded():
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip("Blender interpreter unavailable")
    script = SCRIPT.replace("WORKER_PATH", repr(str(WORKER)))
    result = subprocess.run([interpreter, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok"), result.stdout
