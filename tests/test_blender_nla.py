"""NLA-authored animation must be visible to the harness, not silently lost.

Clip-local keying is the authoring shape a retimeable timeline needs: keys live
in an Action's own frame space and a strip says where that Action plays, so
moving or stretching one clip touches no other clip's keyframes. Blender 4.5
supports it directly -- slotted Actions carry every object in a clip at once.
A harness that reads only `animation_data.action` misses it.

Each test here pins one place that has to read the NLA. They are failure modes
rather than features: in every case a harness reading only the active action
returns success and the animation is gone somewhere else, later.
"""
import pytest

from dycraft.animation.blender.scene_digest import diff
from dycraft.tests._animation_fixtures import (call, plan, service,
                                                       source)

#: The integration render profile: small, fast, and shaded enough to differ
#: when the subject does.
_CHEAP = {"engine": "WORKBENCH", "width": 128, "height": 96, "threads": 2}


#: Two clips on one track, keyed 0..10 in their own Actions and placed at 1 and
#: 21. The object's active action is cleared after each: an NLA-authored object
#: has `animation_data.action is None`, which the readers below must not treat
#: as "not animated".
def clips(target):
    return f'''
o = bpy.data.objects[{target!r}]
o.animation_data_create()

def clip(name, dx):
    action = bpy.data.actions.new(name)
    o.animation_data.action = action
    o.animation_data.action_slot = action.slots.new(id_type='OBJECT', name=o.name)
    o.location.x = 0.0
    o.keyframe_insert('location', frame=0)
    o.location.x = dx
    o.keyframe_insert('location', frame=10)
    o.animation_data.action = None
    return action

track = o.animation_data.nla_tracks.new()
track.name = 'sequence'
for i, (name, dx) in enumerate((('approach', .05), ('settle', .09))):
    strip = track.strips.new(name, 1 + i * 20, clip(name, dx))
    strip.action_slot = strip.action.slots[0]
result = {{'active_action': o.animation_data.action,
           'strips': [[s.name, s.frame_start, s.frame_end] for s in track.strips]}}
'''


def built(root):
    """A compiled project, with no storyboard: `stage='storyboard'` needs none."""
    svc = service(root)
    (svc.design / "scene.py").write_text(source())
    assert call(svc, "plan", plan=plan())["ok"]
    assert call(svc, "build")["ok"]
    return svc


@pytest.mark.expensive
def test_a_clip_authored_object_is_not_silently_unbound_by_a_source_change(tmp_path):
    """`references_to` decides whether a removed object had authored bindings.

    Reading only the active action, it answers "none" for an object animated
    through strips -- so `synchronize` raises no issue, `build()` returns
    `{"ok": true}`, and the object and its clips are deleted together. The
    author is told nothing, and finds out at the next render.
    """
    svc = built(tmp_path)
    try:
        edit = call(svc, "blender", stage="storyboard", code=clips("part/block/body"))
        assert edit["ok"], edit
        # Precondition, not decoration: if the clips had left an active action
        # behind, an active-action check alone would catch them and the test
        # would pass for the wrong reason.
        assert edit["result"]["active_action"] is None
        assert [s[0] for s in edit["result"]["strips"]] == ["approach", "settle"]

        # Rename the part. `part/block/body` and `link/block__body` leave the
        # generated set; the animation on the first of them is the binding.
        (svc.design / "scene.py").write_text(source().replace('"body"', '"shell"'))
        rebuilt = call(svc, "build")
        assert not rebuilt["ok"], rebuilt
        removed = [i for i in rebuilt["issues"] if i["id"] == "removed:part/block/body"]
        assert removed, rebuilt["issues"]
        assert "animation on part/block/body" in removed[0]["bindings"]
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_the_digest_records_a_clip_moving_without_a_keyframe_changing(tmp_path):
    """Retiming is strip placement, so a curve-only fingerprint sees nothing.

    This is the whole point of clip-local keying, and the reason the journal
    has to know about it: shifting a clip re-cuts the film and leaves every
    keyframe where it was. A digest that reported `None` for that operation
    would be answering "what did this call change" with "nothing".
    """
    svc = built(tmp_path)
    try:
        assert call(svc, "blender", stage="storyboard",
                    code=clips("joint/world__block"))["ok"]
        authored = svc.journal.records()[-1]["seq"]

        moved = call(svc, "blender", stage="storyboard", code='''
track = bpy.data.objects['joint/world__block'].animation_data.nla_tracks['sequence']
strip = track.strips['settle']
before = [c.co.x for bag in strip.action.layers[0].strips[0].channelbags
          for c in bag.fcurves for c in c.keyframe_points]
strip.frame_start_ui = 41
strip.scale = 2.0
after = [c.co.x for bag in strip.action.layers[0].strips[0].channelbags
         for c in bag.fcurves for c in c.keyframe_points]
result = {'keys_unchanged': before == after,
          'placement': [strip.frame_start, strip.frame_end]}
''')
        assert moved["ok"], moved
        # The premise: the clip now plays later and twice as slowly, and not
        # one of its keyframes moved.
        assert moved["result"]["keys_unchanged"]
        assert moved["result"]["placement"][0] == pytest.approx(41.0)

        changed = diff(svc.journal.read_digest(authored),
                       svc.journal.read_digest(svc.journal.records()[-1]["seq"]))
        assert changed is not None, "a retime that the journal reports as a no-op"
        assert "joint/world__block" in changed["objects"]["modified"]
        # `diff` omits a field it found nothing in, so this says the retime
        # reached the object record and nothing else -- which is the claim:
        # placement moved, content did not.
        assert "curves" not in changed, changed["curves"]
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_an_action_that_will_not_survive_the_save_is_reported_while_it_exists(tmp_path):
    """Blender does not write a zero-user datablock, and says nothing about it.

    An alternative take the author stashes rather than places is absent when
    the next job reopens the file. Nothing raises -- not Blender, not the
    harness -- so the loss surfaces much later as a clip that cannot be found.
    """
    svc = built(tmp_path)
    try:
        stashed = call(svc, "blender", stage="storyboard", code='''
alternate = bpy.data.actions.new('alternate_take')
retained = bpy.data.actions.new('retained_take')
retained.use_fake_user = True
result = {'users': [alternate.users, retained.users]}
''')
        assert stashed["ok"], stashed
        # A fake user *is* a user, which is the whole mechanism: it is what
        # takes the retained action above zero and so into the saved file.
        assert stashed["result"]["users"] == [0, 1]
        # Only the one Blender is about to drop. A fake user is the author
        # saying "keep this", and warning about it would be noise.
        assert stashed["warnings"]["unsaved_actions"] == ["alternate_take"]

        # `fresh=True` restarts the resident Blender, so this reads the saved
        # file rather than the session that made it. That distinction is the
        # whole point: the stashed action is still in memory until something
        # reopens the project, and every fresh worker, every bundle, and the
        # delivered blend are reopens.
        survived = call(svc, "probe", fresh=True, expression=
                        "sorted(a.name for a in bpy.data.actions)")
        assert survived["result"] == ["retained_take"], survived
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_a_clip_can_be_checked_and_drawn_by_name_without_working_out_where_it_sits(tmp_path):
    """`clip='settle'` is the addressing half of clip-local authoring.

    Keys being clip-local buys nothing if looking at a clip still means
    computing its absolute span by hand through every retime. The worker
    resolves the name against
    the NLA, so moving the clip first changes the answer and costs the author
    nothing.
    """
    svc = built(tmp_path)
    try:
        assert call(svc, "blender", stage="storyboard",
                    code=clips("joint/world__block"))["ok"]
        # The scene has to be long enough to contain the strips.
        assert call(svc, "blender", stage="storyboard",
                    code="bpy.context.scene.frame_end = 40\n"
                         "result = {'end': bpy.context.scene.frame_end}")["ok"]

        checked = call(svc, "check_motion", clip="settle")
        assert checked["ok"] is not None, checked
        # `settle` was placed at native 21 and runs ten frames; frame_start is
        # 0 in a new project, so those are output indices unchanged.
        assert checked["frames_checked"]["start"] == 21
        assert checked["frames_checked"]["end"] == 31
        assert checked["partial"]

        # Move it, and the same name now answers somewhere else.
        assert call(svc, "blender", stage="storyboard", code='''
strip = bpy.data.objects['joint/world__block'].animation_data.nla_tracks['sequence'].strips['settle']
strip.frame_start_ui = 28
result = {'placement': [strip.frame_start, strip.frame_end]}
''')["ok"]
        moved = call(svc, "check_motion", clip="settle")
        assert moved["frames_checked"]["start"] == 28, moved["frames_checked"]

        unknown = call(svc, "check_motion", clip="nosuchclip")
        assert not unknown["ok"] and "unknown clip" in str(unknown)
    finally:
        svc.jobs.close()


#: A plan in the current shape: ordered states with no frames, and shots that
#: are camera segments. Where a state sits is bound in the blend below.
def clip_plan():
    base = {k: v for k, v in plan().items() if k != "shots"}
    return {**base,
            "shots": [{"id": "slide", "intent": "show the whole slide", "camera": "Camera"}],
            "states": [{"id": "start", "purpose": "initial support"},
                       {"id": "settled", "purpose": "final support"}]}


#: Two clips, a camera the timeline makes active, and each state bound to the
#: first frame of a clip. Nothing here names an absolute frame.
STAGE_CLIPS = '''
import json
from mathutils import Vector
s = bpy.context.scene
s.frame_start = 0
s.frame_end = 40
s.render.fps = 24
camera = s.camera
camera.name = 'Camera'
camera.location = (1.4, -1.8, 1.1)
camera.rotation_euler = (Vector((0.1, 0, 0.1)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
s.display.shading.light = 'STUDIO'

o = bpy.data.objects['joint/world__block']
o.animation_data_create()
track = o.animation_data.nla_tracks.new()
track.name = 'sequence'
for i, (name, dx) in enumerate((('approach', .0), ('settle', .05))):
    action = bpy.data.actions.new(name)
    o.animation_data.action = action
    o.animation_data.action_slot = action.slots.new(id_type='OBJECT', name=o.name)
    o.location.x = dx
    o.keyframe_insert('location', frame=0)
    o.location.x = dx + .02
    o.keyframe_insert('location', frame=8)
    o.animation_data.action = None
    strip = track.strips.new(name, 1 + i * 20, action)
    strip.action_slot = action.slots[0]

s['dycraft_states'] = json.dumps({'start': {'clip': 'approach', 'frame': 0},
                                  'settled': {'clip': 'settle', 'frame': 0}})
result = {'strips': [[x.name, x.frame_start] for x in track.strips]}
'''


#: `STAGE_CLIPS` with the 'approach' action keyed only from frame 3, and the
#: bindings, a cut, and the declarations assigned as plain data, not JSON.
STAGE_PLAIN = (STAGE_CLIPS
               .replace("o.keyframe_insert('location', frame=0)",
                        "o.keyframe_insert('location', frame=3 if name == 'approach' else 0)")
               .replace("""s['dycraft_states'] = json.dumps({'start': {'clip': 'approach', 'frame': 0},
                                  'settled': {'clip': 'settle', 'frame': 0}})""",
                        """s['dycraft_states'] = {'start': {'clip': 'approach', 'frame': 5},
                       'settled': {'clip': 'settle', 'frame': 0}}
s['dycraft_cuts'] = [{'camera': 'Camera', 'from': {'clip': 'approach', 'frame': 3}}]
s['dycraft_declarations'] = {'events': [], 'collision_allowances': []}"""))


@pytest.mark.expensive
def test_bindings_may_be_plain_data_and_clip_time_is_the_action_s_own(tmp_path):
    """A dict or list assigned to `dycraft_states`, `dycraft_cuts` or
    `dycraft_declarations` is read as the JSON it stands for. And an action
    keyed only from frame 3 puts its frame 5 two frames into its strip, not
    five: the strip plays the action from its first key."""
    assert STAGE_PLAIN != STAGE_CLIPS and "json.dumps" not in STAGE_PLAIN.split("dycraft_states")[1]
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=clip_plan())["ok"]
        assert call(svc, "build")["ok"]
        staged = call(svc, "blender", stage="storyboard", code=STAGE_PLAIN)
        assert staged["ok"], staged
        assert staged["warnings"]["cuts"] == [{"frame": 1, "camera": "Camera"}]
        assert "declaration_errors" not in staged.get("warnings", {})
        board = call(svc, "storyboard", render_settings=_CHEAP)
        assert board["ok"], board
        assert board["states"]["start"]["frame"] == 3
        assert board["states"]["settled"]["frame"] == 21
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_a_retime_moves_a_bound_state_and_keeps_the_review_it_already_had(tmp_path):
    """The property the whole design exists for, end to end.

    A state identified by an absolute frame would rotate its revision when
    moved, discarding a review of a picture that had not changed. A bound state
    has no frame in the plan at all: the clip moves, the state moves with it, the pose is the same pose,
    and the observation still describes it.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=clip_plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code=STAGE_CLIPS)["ok"]

        first = call(svc, "storyboard", render_settings=_CHEAP)
        assert first["ok"], first
        # Resolved from the strips, not from anything the plan said.
        assert first["states"]["start"]["frame"] == 1
        assert first["states"]["settled"]["frame"] == 21
        assert first["states"]["settled"]["clip"] == "settle"
        # Rendered through the timeline, so the panel names the film's camera.
        assert first["states"]["start"]["camera"] == "Camera"
        assert first["clips_ready"] == {"approach": False, "settle": False}

        assert call(svc, "accept_storyboard", reviews={
            "start": "Block rests on the floor at the head of the approach.",
            "settled": "Block rests on the floor where the settle begins."})["ok"]
        ready = call(svc, "storyboard", render_settings=_CHEAP)
        assert ready["clips_ready"] == {"approach": True, "settle": True}

        # Retime: the settle clip plays later and twice as slowly.
        assert call(svc, "blender", stage="storyboard", code='''
strip = bpy.data.objects['joint/world__block'].animation_data.nla_tracks['sequence'].strips['settle']
strip.frame_start_ui = 28
strip.scale = 2.0
result = {'placement': [strip.frame_start, strip.frame_end]}
''')["ok"]

        after = call(svc, "storyboard", render_settings=_CHEAP)
        assert after["ok"], after
        # The state moved with its clip...
        assert after["states"]["settled"]["frame"] == 28
        # ...and its review came with it. This is the whole claim.
        assert after["needs_review"] == []
        assert sorted(after["retained"]) == ["settled", "start"]
        assert after["states"]["settled"]["review"].startswith("Block rests on the floor where")
        # Coverage never lapsed, so authoring was never re-blocked.
        assert call(svc, "blender", code="result = {}")["ok"]
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_a_discontinuity_at_a_cut_is_reported_as_the_join_it_is(tmp_path):
    """Ten thousand rows saying "frames 503/504" cannot answer "did it cut badly".

    Thousands of speed and spike warnings on a first check go unread, and a
    teleport at a clip boundary is indistinguishable from mid-clip jitter
    among them, so the boundary is named.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=clip_plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code=STAGE_CLIPS)["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews={
            "start": "Head of the approach.", "settled": "Head of the settle."})["ok"]
        # A clip that begins a long way from where the previous one ended is
        # a jump at the cut, which is what this has to surface.
        assert call(svc, "blender", code='''
action = bpy.data.actions['settle']
moved = 0
for bag in action.layers[0].strips[0].channelbags:
    for curve in bag.fcurves:
        if curve.data_path == 'location' and curve.array_index == 0:
            for key in curve.keyframe_points:
                key.co.y += 1.5
                moved += 1
            curve.update()
result = {'moved': moved}
''')["ok"]

        checked = call(svc, "compile")
        rows = next(r for r in checked["verifier_reports"]
                    if r["name"] == "MotionContinuity")["evidence"].get("boundaries")
        assert rows, checked
        join = next(r for r in rows if r["enters"] == "settle")
        assert join["leaves"] == "approach" and join["frame"] == 21
        assert join["steps"]["translation"]["magnitude"] > 1.0, join
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_a_declared_cut_is_materialised_and_follows_its_clip(tmp_path):
    """The one coordinate that cannot stay relative, kept in step anyway.

    Blender reads `timeline_markers`, so a cut is an absolute frame in the
    file and something has to write it. Without that, authors make cameras
    static because they are "immune to retiming". Declaring the cut against a
    clip and projecting it here means a retime moves it for free.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=clip_plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code=STAGE_CLIPS)["ok"]
        declared = call(svc, "blender", stage="storyboard", code='''
import json
s = bpy.context.scene
second = s.camera.copy()
second.name = 'CAM_settle'
s.collection.objects.link(second)
# An author's own marker, which the harness must leave alone.
s.timeline_markers.new('my_note', frame=5)
s['dycraft_cuts'] = json.dumps([
    {'camera': 'Camera', 'from': {'clip': 'approach', 'frame': 0}},
    {'camera': 'CAM_settle', 'from': {'clip': 'settle', 'frame': 0}}])
result = {}
''')
        assert declared["ok"], declared
        assert declared["warnings"]["cuts"] == [
            {"frame": 1, "camera": "Camera"}, {"frame": 21, "camera": "CAM_settle"}]

        markers = call(svc, "probe", expression=
                       "sorted((m.frame, m.name, m.camera.name if m.camera else None)"
                       " for m in bpy.context.scene.timeline_markers)")["result"]
        assert [m[0] for m in markers] == [1, 5, 21]
        assert markers[1][1] == "my_note", markers

        # Retime the settle clip; the cut follows it with no marker edit.
        moved = call(svc, "blender", stage="storyboard", code='''
strip = bpy.data.objects['joint/world__block'].animation_data.nla_tracks['sequence'].strips['settle']
strip.frame_start_ui = 30
result = {}
''')
        assert moved["warnings"]["cuts"] == [
            {"frame": 1, "camera": "Camera"}, {"frame": 30, "camera": "CAM_settle"}]
        after = call(svc, "probe", expression=
                     "sorted(m.frame for m in bpy.context.scene.timeline_markers)")["result"]
        assert after == [1, 5, 30], after

        # Scaling the clip scales a cut's offset too, as it does a state's: 4
        # action frames into a clip at scale 2 is 8 output frames past its
        # start. Unscaled, cuts would fall outside a retimed clip.
        scaled = call(svc, "blender", stage="storyboard", code='''
import json
s = bpy.context.scene
close = s.camera.copy()
close.name = 'CAM_close'
s.collection.objects.link(close)
strip = bpy.data.objects['joint/world__block'].animation_data.nla_tracks['sequence'].strips['settle']
strip.scale = 2.0
s['dycraft_cuts'] = json.dumps([
    {'camera': 'Camera', 'from': {'clip': 'approach', 'frame': 0}},
    {'camera': 'CAM_settle', 'from': {'clip': 'settle', 'frame': 0}},
    {'camera': 'CAM_close', 'from': {'clip': 'settle', 'frame': 4}}])
result = {}
''')
        assert scaled["ok"], scaled
        assert scaled["warnings"]["cuts"] == [
            {"frame": 1, "camera": "Camera"}, {"frame": 30, "camera": "CAM_settle"},
            {"frame": 38, "camera": "CAM_close"}]
    finally:
        svc.jobs.close()


@pytest.mark.expensive
@pytest.mark.parametrize("batch", [True, False])
def test_a_helper_is_reachable_at_the_documented_path_on_either_execution_path(tmp_path, batch):
    """`/workspace` is author-facing, so both paths have to have it.

    A resident session binds `native/` and puts the project at `/native/work`
    so it survives `publish()` renaming the project directory, which on its own
    would remove `/workspace`. The prompt documents `/workspace`, and authors
    use `exec(open('/workspace/x.py'))`, so the same code must work batched and
    resident.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        helpers = svc.root / "animation_src"
        helpers.mkdir(exist_ok=True)
        (helpers / "motion.py").write_text("SLIDE_TO = 0.05\n")
        edit = call(svc, "blender", stage="storyboard", batch=batch,
                    code="exec(open('/workspace/motion.py').read())\n"
                         "result = {'slide_to': SLIDE_TO}")
        assert edit["ok"], edit
        assert edit["result"] == {"slide_to": 0.05}
        # And by relative path, after that edit has published. `publish()`
        # renames the project aside and deletes it, so a resident worker that
        # entered it once at launch would be left in the deleted copy, and
        # relative reads would raise FileNotFoundError.
        again = call(svc, "blender", stage="storyboard", batch=batch,
                     code="result = {'flat': open('motion.py').read(), "
                          "'nested': open('animation_src/motion.py').read()}")
        assert again["ok"], again
        assert again["result"] == {"flat": "SLIDE_TO = 0.05\n", "nested": "SLIDE_TO = 0.05\n"}
    finally:
        svc.jobs.close()


@pytest.mark.expensive
def test_states_listed_out_of_time_order_still_storyboard(tmp_path):
    """A plan lists states in the author's order, not the timeline's.

    An author may list a clip's `_engage` and `_result` before its mid-clip
    states. Every panel must render and the fused check must accept the frames,
    rather than refusing them after the renders for not increasing.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        reordered = clip_plan()
        reordered["states"] = list(reversed(reordered["states"]))
        assert call(svc, "plan", plan=reordered)["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code=STAGE_CLIPS)["ok"]
        board = call(svc, "storyboard", render_settings=_CHEAP)
        assert "increasing" not in str(board), board
        assert board["states"]["settled"]["frame"] == 21
        assert board["states"]["start"]["frame"] == 1
    finally:
        svc.jobs.close()
