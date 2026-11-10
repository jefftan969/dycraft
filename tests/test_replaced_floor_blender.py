"""`room.floor` names a modeled floor declared with `replace_room_surface`.

The prompt has authors model an interior's floor and declare it as the shell
floor's replacement, and documents `room.floor` as an event operand and as
the ground of `supported` and `planted`. The replacement removed the shell
floor, so `room.floor` resolved to nothing on every frame and its events were
never checked: 18 events in one audited film, 48 of 126 in another.
"""
import json

import pytest

from dycraft.tests._animation_fixtures import (
    STAGE, call, plan, service, state_reviews)

pytestmark = pytest.mark.expensive

SOURCE = '''
obj = Object("block")
obj.declare_root(obj.add_part("body", Box(0.2, 0.2, 0.2), material=Material(color=(180, 100, 30))))
slab = Object("slab")
slab.declare_root(slab.add_part("top", Box(3.0, 3.0, 0.05), material=Material(color=(90, 90, 90))))
scene = Scene("stage")
scene.set_room(4.0, 4.0, 3.0)
scene.place(obj.finalize(), name="block", pose=((0, 0, 0), (0, 0, 0)))
floor = scene.place(slab.finalize(), name="floor", pose=((0, 0, -0.05), (0, 0, 0)))
scene.replace_room_surface("floor", floor)

@lighting
def rig(L):
    key = L.new_light("key", type="AREA")
    key.data.energy = 300.0
    key.location = (1.0, -1.0, 2.0)
    L.aim(key, (0.0, 0.0, 0.0))

scene.set_lighting(rig)
ir = scene.finalize()
'''

ON_THE_FLOOR = '''
import json
s = bpy.context.scene
d = json.loads(s['dycraft_declarations'])
d['events'] = [{'name': 'on_floor', 'predicate': 'touch', 'a': 'block/body', 'b': 'room.floor',
                'tolerance': 0.06, 'window': [0, 1]},
               {'name': 'stands', 'predicate': 'supported', 'a': 'block', 'window': [0, 1]}]
s['dycraft_declarations'] = json.dumps(d)
'''


def test_room_floor_resolves_to_the_modeled_floor(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(SOURCE)
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        baseline = json.loads((svc.work / "design_state/baseline.json").read_text())
        assert baseline["surface_replacements"] == {"floor": "floor"}
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings={"engine": "WORKBENCH", "width": 128,
                                                        "height": 96, "threads": 2})["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("fixture"))["ok"]
        assert call(svc, "blender", code=ON_THE_FLOOR)["ok"]
        report = call(svc, "compile", timeout=600)
        events = next(r for r in report["verifier_reports"] if r["name"] == "EventsSatisfied")
        rows = {row["name"]: row for row in events["evidence"]["events"]}
        for name in ("on_floor", "stands"):
            assert "missing_refs" not in rows[name], rows[name]
            assert rows[name]["status"] == "satisfied", rows[name]
    finally:
        svc.jobs.close()
