"""A source update must not move a multi-material mesh onto its first slot.

`Mesh.materials.clear()` zeroes every face's `material_index`, so an update
path that clears and re-appends slots whenever geometry changes or materials
are refreshed puts an imported character's hair cards over its whole body.
Run against real Blender on the German shepherd's skin.
"""
import subprocess

import pytest

from dycraft.animation.blender.runtime import REPO
from dycraft.render._blender_env import blender_python

RIG = REPO / "assets/imported/sf_dog_gsd_puppyclub_rig/assets/rigs"

_SCRIPT = r'''
import bpy, sys, collections
sys.path.insert(0, sys.argv[-2])
import update
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1])
o = next(x for x in bpy.data.objects if x.type == "MESH")
count = lambda: sorted(collections.Counter(p.material_index for p in o.data.polygons).items())
before = count()
update._set_materials(o.data, list(o.data.materials))          # same slots, re-set
same = count()
o.data = o.data.copy()
update._set_materials(o.data, list(o.data.materials)[:2])      # one slot fewer: clamp
fewer = count()
print("RESULT", before, same, fewer)
'''


def test_resetting_slots_keeps_each_face_on_its_slot(tmp_path):
    py = blender_python()
    blends = list(RIG.glob("*.blend"))
    if py is None or not blends:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    script = tmp_path / "t.py"
    script.write_text(_SCRIPT)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/blender"), str(blends[0])],
                         capture_output=True, text=True, timeout=600)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, out.stdout[-1500:] + out.stderr[-1500:]
    before, same, fewer = _parse(line)
    assert len(before) == 3, before
    assert same == before, "re-setting the same slots must keep every face's slot"
    assert dict(fewer)[1] == dict(before)[1] + dict(before)[2], "indices past the last slot clamp to it"


def _parse(line):
    import ast
    parts = line[7:].split("] [")
    return [ast.literal_eval(("[" if i else "") + p + ("]" if i < len(parts) - 1 else "")) for i, p in enumerate(parts)]
