"""Frame identity beyond the signature, as drawn, and what an edit reaches.

Real Blender, on the stack fixture: a box that slides over ten frames.
"""
import json
import shutil
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.expensive


@pytest.fixture
def film(tmp_path):
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_python
    from dycraft.tests._scene_fixtures import _stack_scene
    if blender_python() is None:
        pytest.skip("Blender interpreter not configured")
    scene, catalog = _stack_scene()
    (tmp_path / "catalog").mkdir()
    work, inputs = tmp_path / "work", tmp_path / "inputs"
    work.mkdir()
    prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog), inputs)
    readonly = [inputs, tmp_path / "catalog"]

    def worker(label, request):
        return run_worker(work, tmp_path / label, request, readonly=readonly)
    worker("init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")})
    worker("edit", {"op": "edit", "code": '''
s = bpy.context.scene
s.frame_start = 0
s.frame_end = 9
root = bpy.data.objects['link/box__body']
root.keyframe_insert('location', frame=0)
root.location.x += .3
root.keyframe_insert('location', frame=9)
light = bpy.data.lights.new('fill', 'POINT')
lamp = bpy.data.objects.new('fill', light)
s.collection.objects.link(lamp)
lamp.location = (2, -2, 3)
for frame, energy in ((0, 300), (4, 300), (6, 500), (9, 500)):
    light.energy = energy
    light.keyframe_insert('energy', frame=frame)
'''})
    return SimpleNamespace(worker=worker, work=work, root=tmp_path)


def states(info):
    return {int(frame): value for frame, value in info["render_states"].items()}


def test_the_digest_is_stable_and_changes_exactly_where_the_look_does(film):
    first = film.worker("e1", {"op": "evaluate"})
    again = film.worker("e2", {"op": "evaluate"})
    assert first["render_state_version"] and None not in states(first).values()
    assert states(first) == states(again), "two evaluations of one blend must agree"
    # Jumped to, scattered: the same digests as walking the film.
    jumped = film.worker("e3", {"op": "evaluate", "frames": [2, 7]})
    assert {f: states(first)[f] for f in (2, 7)} == states(jumped)
    assert jumped["signatures"] == {k: first["signatures"][k] for k in ("2", "7")}

    # An animated look changed on frames 5-9 only: those digests move, the
    # rest do not, and no signature does -- geometry is unchanged.
    film.worker("brighter", {"op": "edit", "code": '''
light = bpy.data.lights['fill']
for frame, energy in ((6, 900), (9, 900)):
    light.energy = energy
    light.keyframe_insert('energy', frame=frame)
'''})
    brighter = film.worker("e4", {"op": "evaluate"})
    assert brighter["signatures"] == first["signatures"]
    moved = sorted(f for f in range(10) if states(brighter)[f] != states(first)[f])
    assert moved == [5, 6, 7, 8, 9], moved

    # A static look change -- the camera's lens -- moves every frame.
    film.worker("lens", {"op": "edit", "code": "bpy.context.scene.camera.data.lens += 7"})
    lens = film.worker("e5", {"op": "evaluate"})
    assert all(states(lens)[f] != states(brighter)[f] for f in range(10))
    film.worker("holdout", {"op": "edit", "code": "bpy.data.objects['part/table/body'].is_holdout = True"})
    holdout = film.worker("e6", {"op": "evaluate"})
    assert all(states(holdout)[f] != states(lens)[f] for f in range(10))

    # What the evaluation does not record, the digest does not vouch for.
    film.worker("points", {"op": "edit", "code": '''
cloud = bpy.data.objects.new('spray', bpy.data.pointclouds.new('spray'))
bpy.context.scene.collection.objects.link(cloud)
'''})
    assert set(states(film.worker("e7", {"op": "evaluate"})).values()) == {None}


def test_a_delivery_render_draws_the_identities_the_check_recorded(film):
    """The render signs what it draws. Delivery admits a frame later by this
    pair, so it must equal the check's when nothing changed."""
    checked = film.worker("check", {"op": "evaluate"})
    shutil.copytree(film.root / "check/evaluation", film.work / "evaluation")
    settings = {"engine": "WORKBENCH", "width": 64, "height": 64, "threads": 2}
    drawn = film.worker("draw", {"op": "render", "verify_replay": True,
                                 "render_settings": settings})
    assert drawn["replay_mismatch"] == []
    assert drawn["render_state_version"] == checked["render_state_version"]
    assert {int(f): (v["signature"], v["render_state"]) for f, v in drawn["drawn"].items()} == {
        f: (checked["signatures"][str(f)], states(checked)[f]) for f in range(10)}

    # A frame that does not reproduce is drawn, and named, rather than
    # refused.
    from dycraft.animation.blender.evaluation import read_record, write_record
    record = read_record(film.work / "evaluation", 3)
    record["transforms"]["part/box/body"][0][3] += .5
    write_record(film.work / "evaluation", record)
    redrawn = film.worker("redraw", {"op": "render", "verify_replay": True,
                                     "render_settings": settings, "frames": [2, 3, 4]})
    assert len(redrawn["paths"]) == 3
    mismatch, = redrawn["replay_mismatch"]
    assert mismatch["frame"] == 3 and "part/box/body" in mismatch["changed"]["transforms"]


def test_an_edit_reports_the_frames_its_animation_reaches(film):
    """An edit's new keys reach from the key before them to the key after."""
    film.worker("rekey", {"op": "edit", "code": '''
root = bpy.data.objects['link/box__body']
root.location.z += .1
root.keyframe_insert('location', index=2, frame=5)
'''})
    changed = json.loads((film.root / "rekey/animation_changed.json").read_text())
    # And the film's length after the edit, which sizes the next parallel check.
    assert changed == {"changed_frames": [[0, 9]], "n_frames": 10}
    film.worker("nothing", {"op": "edit", "code": "bpy.context.scene.camera.data.lens += 1"})
    assert json.loads((film.root / "nothing/animation_changed.json").read_text())[
        "changed_frames"] == []
    film.worker("late", {"op": "edit", "code": '''
light = bpy.data.lights['fill']
light.energy = 650
light.keyframe_insert('energy', frame=8)
'''})
    # Inserting a key at 8 also re-derives key 6's automatic handles, which
    # shape the curve back to key 4: those frames changed too.
    changed = json.loads((film.root / "late/animation_changed.json").read_text())
    assert changed["changed_frames"] == [[4, 9]]

    # A deleted key reaches as far as an added one.
    film.worker("unkey", {"op": "edit", "code": '''
bpy.data.lights['fill'].keyframe_delete('energy', frame=8)
'''})
    changed = json.loads((film.root / "unkey/animation_changed.json").read_text())
    assert changed["changed_frames"] == [[4, 9]]
    # So does a curve modifier's setting, not only its kind.
    for label, code in (("noise", "curve.modifiers.new('NOISE')"),
                        ("stronger", "curve.modifiers[0].strength = 3")):
        film.worker(label, {"op": "edit", "code": (
            "curve = bpy.data.lights['fill'].animation_data.action.fcurves.find('energy')\n"
            + code)})
        changed = json.loads((film.root / label / "animation_changed.json").read_text())
        assert changed["changed_frames"] == [[0, 9]], label



def test_placing_states_names_a_pin_past_the_end_and_needs_no_bake(film):
    """`require_span`'s placement pass certifies nothing: an unbaked cloth that
    a check refuses must not stop the edit that would bake it, and a state
    pinned past a shortened film is refused by name, not as a bad selection."""
    film.worker("cloth", {"op": "edit", "code": (
        "bpy.data.objects['part/table/body'].modifiers.new('drape', 'CLOTH')")})
    with pytest.raises(Exception, match="bake"):
        film.worker("check", {"op": "evaluate"})
    placed = film.worker("place", {"op": "evaluate", "render_states": False, "place_only": True,
                                   "states": [{"id": "early", "frame": 3},
                                              {"id": "late", "frame": 40}]})
    # Placed, not evaluated: no frame is visited, so no simulation is stepped.
    assert placed["state_frames"] == {"early": 3} and placed["frames"] == []
    assert not (film.root / "place/evaluation").exists()
    assert "past the timeline [0, 9]" in placed["unresolved_states"]["late"]


SETTINGS_SCRIPT = r'''
import bpy, sys, json
sys.path.insert(0, sys.argv[-2])
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1], load_ui=False)
import render_config, render_state, worker
def static():
    worker.prepare_evaluation()
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    return render_state.RenderStates(bpy, worker.timeline(), display_color=worker.display_color).static.hex()
before = static()
out = {}
for settings in ({"engine": "CYCLES", "width": 64, "height": 64, "samples": 4,
                  "compute_backend": "CPU", "denoise": False, "adaptive_threshold": None},
                 {"engine": "EEVEE", "width": 96, "height": 64, "samples": 2},
                 {"engine": "WORKBENCH", "width": 640, "height": 360, "threads": 3}):
    render_config.apply_final_render_settings(bpy, settings)
    worker.apply_display_colors()
    out[settings["engine"]] = static()
print("RESULT " + json.dumps({"before": before, "after": out}))
'''


def test_no_delivery_render_setting_reaches_the_digest(film, tmp_path):
    """`final_render` applies its settings only in its own snapshot, so
    a digest that read any of them would differ between the check and every
    delivery. Engine-derived read-only flags and an add-on's storage of its
    own settings are excluded for that reason."""
    import subprocess
    from dycraft.animation.blender.runtime import REPO
    from dycraft.render._blender_env import blender_binary
    film.worker("check", {"op": "evaluate"})
    script = tmp_path / "settings.py"
    script.write_text(SETTINGS_SCRIPT)
    done = subprocess.run([blender_binary() or "blender", "-b", "--factory-startup", "--python",
                           str(script), "--", str(REPO / "animation/blender"),
                           str(film.work / "animation.blend")],
                          capture_output=True, text=True, timeout=300)
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    assert set(result["after"].values()) == {result["before"]}, result


SLOTS_SCRIPT = r'''
import bpy, sys, json
sys.path.insert(0, sys.argv[-2])
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1], load_ui=False)
import render_state, worker
fill = bpy.data.lights['fill']
other = bpy.data.lights.new('rim', 'POINT')
bpy.context.scene.collection.objects.link(bpy.data.objects.new('rim', other))
# One action, two slots: the rim light plays its own slot of the fill's action.
other.animation_data_create().action = fill.animation_data.action
for frame, value in ((0, (1, 1, 1)), (9, (1, .5, .2))):
    other.color = value
    other.keyframe_insert('color', frame=frame)
assert other.animation_data.action == fill.animation_data.action
assert len(fill.animation_data.action.slots) == 2
worker.prepare_evaluation()
states = render_state.RenderStates(bpy, worker.timeline(), display_color=worker.display_color)
print("RESULT " + json.dumps(sorted({(idb.name, path) for idb, path, _ in states.channels})))
'''


def test_an_id_s_channels_are_its_own_slot_s(film, tmp_path):
    """One action can hold several IDs' curves in slots. Each ID's channels are
    its own slot's: reading every slot for every user visits millions of curves
    on a large film, and gives a light the channels of every other ID in the
    action."""
    import subprocess
    from dycraft.animation.blender.runtime import REPO
    from dycraft.render._blender_env import blender_binary
    film.worker("check", {"op": "evaluate"})
    script = tmp_path / "slots.py"
    script.write_text(SLOTS_SCRIPT)
    done = subprocess.run([blender_binary() or "blender", "-b", "--factory-startup", "--python",
                           str(script), "--", str(REPO / "animation/blender"),
                           str(film.work / "animation.blend")],
                          capture_output=True, text=True, timeout=300)
    line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, done.stdout[-2000:] + done.stderr[-2000:]
    assert json.loads(line[len("RESULT "):]) == [["fill", "energy"], ["rim", "color"]]
