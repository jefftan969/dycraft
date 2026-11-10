"""`collision.rotation_quaternion`: the public name the benchmark scorer's cache key reads.

The scorer keys contact samples on these quaternions, so the public function must return exactly what the
pair-cache bins use, under one name, with no private copy for a caller to drift onto.
`collision` imports `mathutils`, so the checks run in the Blender interpreter.
"""
import subprocess

import pytest

SCRIPT = r'''
import bpy  # makes mathutils importable
import math
import numpy as np
from dycraft.animation.blender import collision as C

def matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])

assert not hasattr(C, '_rotation_quaternion')

c, s = math.cos(math.pi/4), math.sin(math.pi/4)
known = [
    (np.eye(3), [1.0, 0.0, 0.0, 0.0]),
    (matrix(np.array([c, 0.0, 0.0, s])), [c, 0.0, 0.0, s]),   # positive trace: 90 degrees about z
    (np.diag([1.0, -1.0, -1.0]), [0.0, 1.0, 0.0, 0.0]),       # trace -1, the pivot branch: 180 about x
    (np.diag([-1.0, -1.0, 1.0]), [0.0, 0.0, 0.0, 1.0]),       # 180 about z
]
for m, expected in known:
    np.testing.assert_allclose(C.rotation_quaternion(m), expected, atol=1e-12)

rng = np.random.default_rng(0)
quats = rng.normal(size=(200, 4))
quats[:50, 0] = rng.normal(scale=1e-3, size=50)   # near 180 degrees: the non-positive-trace branch
stack = np.stack([matrix(q) for q in quats])
batch = C._rotation_quaternions(stack)
for q, m, b in zip(quats, stack, batch):
    single = C.rotation_quaternion(m)
    assert single[np.flatnonzero(single)[0]] > 0, single
    assert np.isclose(abs(np.dot(single, q / np.linalg.norm(q))), 1.0, atol=1e-9), (single, q)
    np.testing.assert_allclose(single, b, rtol=0, atol=1e-15)   # the vectorized arithmetic rounds differently
print("ok")
'''


def test_public_rotation_quaternion(tmp_path, pkgroot):
    """One public name, canonical on both trace branches, and equal to the stack version to rounding."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    source = tmp_path/'rotation_quaternion.py'
    source.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+SCRIPT)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.strip().endswith("ok"), result.stdout
