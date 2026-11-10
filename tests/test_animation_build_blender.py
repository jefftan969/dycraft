"""What `h.build()` imports is what the source says, in real Blender.

Three behaviors it pins:

* an instance's `initial_joints` starts its joint channel there, so a keyed
  value is the joint's value rather than an offset from it;
* `refresh={'lighting': True}` brings back everything the `@lighting` block
  set, its color management and render settings as well as its lights; and
* a `@shader` that raises fails the build, naming itself, rather than
  importing a neutral gray part under a successful build.

Also what a build does to the storyboard when it changes what every panel
shows: the room or the lighting.
"""
import json
import math

import pytest

from dycraft.tests._animation_fixtures import call, plan, service

pytestmark = pytest.mark.expensive


def _built(tmp_path, source):
    svc = service(tmp_path)
    (svc.design / "scene.py").write_text(source)
    assert call(svc, "plan", plan=plan())["ok"]
    return svc, call(svc, "build", timeout=600)


CABINETS = '''
def make_cab():
    obj = Object("cab")
    body = obj.add_part("body", Box(0.4, 0.4, 0.6), material=Material(color=(120, 120, 120)))
    door = obj.add_part("door", Box(0.4, 0.02, 0.6), material=Material(color=(200, 60, 60)))
    obj.declare_root(body)
    obj.attach_revolute("hinge", body, door, axis="z", limits=(0, 120), default=0,
                        parent_anchor=obj.anchor(body, "front_right"),
                        child_anchor=obj.anchor(door, "back_right"))
    return obj.finalize()
cab = make_cab()
scene = Scene("stage")
scene.set_room(4.0, 4.0, 3.0)
scene.place(cab, name="shut", pose=((-1, 0, 0), (0, 0, 0)))
scene.place(cab, name="open", pose=((1, 0, 0), (0, 0, 0)), initial_joints={"hinge": 60})
ir = scene.finalize()
'''

DOOR_YAW = ("[round(math.degrees(bpy.data.objects['part/' + n].matrix_world.to_euler()[2]), 3) "
            "for n in ('shut/door', 'open/door')]")


def test_a_keyed_joint_is_its_value_whatever_the_instance_started_at(tmp_path):
    svc, built = _built(tmp_path, CABINETS)
    try:
        assert built["ok"], built
        channel = call(svc, "probe", expression=(
            "round(math.degrees(bpy.data.objects['joint/open__hinge'].rotation_euler[2]), 3)"))
        assert channel["result"] == pytest.approx(60.0)
        assert call(svc, "probe", expression=DOOR_YAW)["result"] == pytest.approx([0.0, 60.0])
        keyed = call(svc, "blender", stage="storyboard", code=(
            "key_joint('shut.hinge', math.radians(90), 0)\n"
            "key_joint('open.hinge', math.radians(90), 0)\n"))
        assert keyed["ok"], keyed
        assert call(svc, "probe", expression=DOOR_YAW, frame=0)["result"] == pytest.approx([90.0, 90.0])
    finally:
        svc.jobs.close()


def test_a_changed_start_reaches_a_joint_nothing_animates(tmp_path):
    svc, built = _built(tmp_path, CABINETS)
    try:
        assert built["ok"], built
        assert call(svc, "blender", stage="storyboard", code=(
            "key_joint('shut.hinge', math.radians(30), 0)\n"))["ok"]
        (svc.design / "scene.py").write_text(CABINETS.replace(
            'initial_joints={"hinge": 60}', 'initial_joints={"hinge": 45}').replace(
            'name="shut", pose=((-1, 0, 0), (0, 0, 0))',
            'name="shut", pose=((-1, 0, 0), (0, 0, 0)), initial_joints={"hinge": 10}'))
        assert call(svc, "build", timeout=600)["ok"]
        # `open` moved to its new start; `shut` is keyed, and keeps its key.
        assert call(svc, "probe", expression=DOOR_YAW, frame=0)["result"] == pytest.approx(
            [30.0, 45.0])
    finally:
        svc.jobs.close()


def _lit(energy, transform, exposure, strength):
    return f'''
obj = Object("block")
obj.declare_root(obj.add_part("body", Box(0.2, 0.2, 0.2), material=Material(color=(180, 100, 30))))
scene = Scene("stage")
scene.set_room(4.0, 4.0, 3.0)
scene.place(obj.finalize(), name="block", pose=((0, 0, 0), (0, 0, 0)))

@lighting
def rig(L):
    key = L.new_light("key", type="AREA")
    key.data.energy = {energy}
    key.location = (1.0, -1.0, 2.0)
    L.aim(key, (0.0, 0.0, 0.0))
    bg = L.world_nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = {strength}
    L.world_links.new(bg.outputs["Background"], L.world_output.inputs["Surface"])
    L.view.view_transform = "{transform}"
    L.view.exposure = {exposure}

scene.set_lighting(rig)
ir = scene.finalize()
'''


LIGHTING = ("[bpy.context.scene.view_settings.view_transform, "
            "round(bpy.context.scene.view_settings.exposure, 3), "
            "[o.data.energy for o in bpy.data.objects if o.type == 'LIGHT']]")


def test_a_lighting_refresh_brings_back_everything_the_block_set(tmp_path):
    svc, built = _built(tmp_path, _lit(100.0, "AgX", -0.3, 0.5))
    try:
        assert built["ok"], built
        assert call(svc, "probe", expression=LIGHTING)["result"] == ["AgX", -0.3, [100.0]]
        (svc.design / "scene.py").write_text(_lit(500.0, "Standard", 1.0, 2.0))
        assert call(svc, "build", timeout=600)["ok"]
        # Without a refresh the project keeps its lighting, as before.
        assert call(svc, "probe", expression=LIGHTING)["result"] == ["AgX", -0.3, [100.0]]
        assert call(svc, "build", refresh={"lighting": True}, timeout=600)["ok"]
        assert call(svc, "probe", expression=LIGHTING)["result"] == ["Standard", 1.0, [500.0]]
    finally:
        svc.jobs.close()


def test_a_lighting_refresh_restores_what_the_block_set_to_the_default(tmp_path):
    """Exposure 0 is Blender's default, so the block's write changes nothing
    on import and a before/after diff never saw it."""
    svc, built = _built(tmp_path, _lit(100.0, "AgX", 0.0, 0.5))
    try:
        assert built["ok"], built
        assert call(svc, "blender", stage="storyboard", code=(
            "bpy.context.scene.view_settings.view_transform = 'Standard'\n"
            "bpy.context.scene.view_settings.exposure = 2.0\n"))["ok"]
        assert call(svc, "build", refresh={"lighting": True}, timeout=600)["ok"]
        assert call(svc, "probe", expression=LIGHTING)["result"][:2] == ["AgX", 0.0]
    finally:
        svc.jobs.close()


def test_a_build_that_changes_the_room_or_the_lighting_drops_every_board(tmp_path):
    from dycraft.tests._animation_fixtures import STAGE, source
    svc, built = _built(tmp_path, source())
    try:
        assert built["ok"], built
        assert call(svc, "blender", stage="storyboard", code=STAGE)["ok"]
        assert call(svc, "storyboard", timeout=600)["states"]
        boards = svc.work / "storyboards.json"
        unchanged = call(svc, "build", timeout=600)
        assert unchanged["ok"] and "storyboards_dropped" not in unchanged, unchanged
        assert len(json.loads(boards.read_text())) == 2
        relit = call(svc, "build", refresh={"lighting": True}, timeout=600)
        assert "lighting" in relit["storyboards_dropped"], relit
        assert json.loads(boards.read_text()) == {}
        assert call(svc, "storyboard", timeout=600)["states"]
        (svc.design / "scene.py").write_text(source().replace("set_room(4.0", "set_room(5.0"))
        moved = call(svc, "build", timeout=600)
        assert "room" in moved["storyboards_dropped"], moved
        assert json.loads(boards.read_text()) == {}
    finally:
        svc.jobs.close()


BAD_SHADER = '''
@shader
def bad(nodes, links, bsdf):
    bsdf.inputs["Base Colour"].default_value = (1.0, 0.0, 0.0, 1.0)

obj = Object("block")
obj.declare_root(obj.add_part("body", Box(0.2, 0.2, 0.2), material=Material(shader=bad)))
scene = Scene("stage")
scene.set_room(4.0, 4.0, 3.0)
scene.place(obj.finalize(), name="block", pose=((0, 0, 0), (0, 0, 0)))
ir = scene.finalize()
'''


def test_a_shader_that_raises_fails_the_build_naming_itself(tmp_path):
    svc, built = _built(tmp_path, BAD_SHADER)
    try:
        assert not built["ok"], built
        assert built["failure_class"] == "author_code", built
        assert "shader 'bad'" in built["error"] and "Base Colour" in built["error"], built["error"]
    finally:
        svc.jobs.close()


def test_an_unsized_look_follows_the_project_s_aspect(tmp_path):
    """Storyboards, reviews, and looks default to a pixel count, drawn at the
    project's own aspect ratio; a vertical film's panels were drawn 16:9."""
    from PIL import Image
    from dycraft.tests._animation_fixtures import STAGE, source

    def size(reply):
        index = reply["frame_indices"][0]
        with Image.open(f"{reply['frames_dir']}/{reply['frame_file'] % index}") as image:
            return image.size

    svc, built = _built(tmp_path, source())
    try:
        assert built["ok"], built
        assert call(svc, "blender", stage="storyboard", code=STAGE)["ok"]
        assert call(svc, "blender", stage="storyboard", code=(
            "bpy.context.scene.render.resolution_x = 1080\n"
            "bpy.context.scene.render.resolution_y = 1920\n"))["ok"]
        width, height = size(call(svc, "snapshot", frame=0))
        assert height > width and abs(width * height - 1280 * 720) < .02 * 1280 * 720
        assert size(call(svc, "snapshot", frame=0, render_settings={
            "engine": "WORKBENCH", "width": 320, "height": 180})) == (320, 180)
    finally:
        svc.jobs.close()
