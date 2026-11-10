"""Actual CAD -> Blender -> staged motion -> H.264 -> portable bundle tests."""
import json
from pathlib import Path
import shutil

import pytest

from dycraft.animation.blender.bundle import stage_prior, validate_manifest, write_json
from dycraft.sdk.ir import content_hash
from dycraft.tests._animation_fixtures import (
    ANIMATE, STAGE, call, plan, service, source, state_reviews)

pytestmark = pytest.mark.expensive

#: The fixture plan's one criterion, judged -- `finish()` requires it.
CRITERIA = {"support": {"verdict": "pass", "evidence": "The block stays on the floor in every frame."}}


def test_a_complete_animation(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        built = call(svc, "build")
        assert built["ok"], built
        assert not call(svc, "blender", code="pass")["ok"]
        staged = call(svc, "blender", code=STAGE, stage="storyboard")
        assert staged["ok"], staged
        boards = call(svc, "storyboard", render_settings={"engine": "WORKBENCH", "width": 128, "height": 96, "threads": 2})
        assert boards["ok"], boards
        assert call(svc, "accept_storyboard", reviews=state_reviews("Test fixture inspects the known camera/support arrangement."))["ok"]
        animated = call(svc, "blender", code=ANIMATE)
        assert animated["ok"], animated
        assert not call(svc, "finish")["ok"]
        film = call(svc, "final_render", render_settings={"engine": "WORKBENCH", "width": 64, "height": 64, "threads": 2})
        assert film["ok"], film
        # Authored sources are outside the native snapshot. A current movie
        # cannot certify a source revision that was never synchronized.
        (svc.design / "scene.py").write_text(source() + "\n# unsynchronized change\n")
        stale = call(svc, "finish")
        assert not stale["ok"] and "changed since build()" in stale["error"]
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "blender", code="bpy.context.scene.camera.data.lens = 42")["ok"]
        assert not call(svc, "finish", criteria=CRITERIA)["ok"]
        # A final render draws without a verdict, and its receipt says
        # the frames are unchecked; the clean path checks first.
        film = call(svc, "final_render", render_settings={"engine": "WORKBENCH", "width": 64, "height": 64, "threads": 2, "strategy": "frames"})
        assert film["ok"], film
        assert any(item["kind"] == "unchecked" for item in film["unmet"]), film["unmet"]
        refused = call(svc, "finish", criteria=CRITERIA)
        assert not refused["ok"] and "run compile()" in refused["error"], refused
        assert call(svc, "compile")["ok"]
        # A criterion is required: the refusal is the checklist.
        missing = call(svc, "finish")
        assert not missing["ok"] and "support" in missing["error"], missing
        film = call(svc, "final_render", render_settings={"engine": "WORKBENCH", "width": 64, "height": 64, "threads": 2, "strategy": "frames"})
        assert film["ok"] and film["unmet"] == [], film
        # final_render writes nothing into the project; the settings reach
        # the delivered blend at finish() instead.
        live = call(svc, "probe", expression="bpy.context.scene.render.resolution_x")
        assert live["ok"] and live["result"] != 64, live
        done = call(svc, "finish", criteria=CRITERIA)
        assert done["ok"] and done["outcome"] == "committed" and done["unmet"] == [], done
        manifest = json.loads((tmp_path / "ir.json").read_text())
        validate_manifest(tmp_path, manifest)
        assert manifest["film"] in manifest["files"]
        assert manifest["n_frames"] == 2
        assert manifest["render"]["width"] == 64
        assert manifest["outcome"] == "committed" and manifest["criteria"] == CRITERIA
        assert "bundle/DELIVERY.md" in manifest["files"]
        # The panels ship with the film even though they no longer live in the
        # workspace, at the path the board already named.
        boards = json.loads((svc.work / "storyboards.json").read_text())
        panels = [board["path"] for board in boards.values()]
        assert panels and all((tmp_path / "bundle" / p).is_file() for p in panels)
        assert not call(svc, "blender", code="pass")["ok"]
        # Reopen a relocated copy, with the selected final camera and settings.
        from dycraft.animation.blender.runtime import run_worker
        relocated = tmp_path / "relocated"
        stage_prior(tmp_path, relocated)
        reopened = run_worker(relocated, tmp_path / "reopened", {"op": "probe", "code":
            "{'lens': bpy.context.scene.camera.data.lens, 'width': bpy.context.scene.render.resolution_x, "
            "'format': bpy.context.scene.render.image_settings.file_format, "
            "'output': bpy.context.scene.render.filepath}"})
        # A still-friendly output: a recipient's render writes PNGs beside the film.
        assert reopened == {"lens": 42., "width": 64, "format": "PNG", "output": "//film/frames/"}
        refinement_root = tmp_path / "refinement"
        stage_prior(tmp_path, refinement_root / "prior_animation")
        refinement = service(refinement_root)
        try:
            refinement.source_fresh(refinement.work)
            assert (refinement.work / "design_state/baseline.json").is_file()
            (refinement.design / "scene.py").write_text(source(width=.25))
            update = call(refinement, "build")
            assert update["ok"], update
        finally:
            refinement.jobs.close()
    finally:
        svc.jobs.close()


def test_source_updates_preserve_native_work_and_explicit_appearance(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        # The native import runs the scene's own `@lighting` block rather than
        # installing a fixed key/fill/top rig and a default World.
        rig = call(svc, "probe", expression="{'key': bpy.data.objects['key'].data.type, 'legacy': [n for n in ('fill', 'top') if n in bpy.data.objects], 'world': list(bpy.context.scene.world.color)}")
        assert rig["ok"], rig
        assert rig["result"]["key"] == "AREA" and rig["result"]["legacy"] == []
        assert rig["result"]["world"] == pytest.approx([0.05, 0.05, 0.05])
        edited = call(svc, "blender", stage="storyboard", code=STAGE + ANIMATE + '''
controller = bpy.data.objects.new('authored_control', None)
bpy.context.collection.objects.link(controller)
controller.location = (.4, .5, .6)
material = bpy.data.materials.new('authored_material')
material.diffuse_color = (.1, .7, .3, 1)
body = bpy.data.objects['part/block/body']
body.data.materials.clear()
body.data.materials.append(material)
bpy.data.objects['key'].data.energy = 7
bpy.data.objects['key'].location = (.4, .5, 2)
bpy.context.scene.world.color = (.02, .04, .08)
''')
        assert edited["ok"], edited
        (svc.design / "scene.py").write_text(source(width=.3, color=(20, 30, 230)))
        rebuilt = call(svc, "build")
        assert rebuilt["ok"], rebuilt
        inspected = call(svc, "probe", expression="{'control': list(bpy.data.objects['authored_control'].location), 'material': bpy.data.objects['part/block/body'].active_material.name, 'action': bpy.data.objects['joint/world__block'].animation_data.action is not None, 'light': bpy.data.objects['key'].data.energy, 'light_location': list(bpy.data.objects['key'].location), 'width': bpy.data.objects['part/block/body'].dimensions.x}")
        assert inspected["ok"], inspected
        values = inspected["result"]
        assert values["control"] == pytest.approx([.4, .5, .6])
        assert values["material"] == "authored_material" and values["action"]
        assert values["light"] == 7 and values["light_location"] == pytest.approx([.4, .5, 2])
        assert values["width"] == pytest.approx(.3)
        refreshed = call(svc, "build", refresh={"materials": ["block/body"], "lighting": True})
        assert refreshed["ok"], refreshed
        inspected = call(svc, "probe", expression="[bpy.data.objects['part/block/body'].active_material.name, bpy.data.objects['key'].data.energy]")
        assert inspected["result"][0] != "authored_material"
        assert inspected["result"][1] != 7
        before = svc.version()
        (svc.design / "scene.py").write_text("this is broken source")
        assert not call(svc, "build")["ok"]
        assert svc.version() == before
    finally:
        svc.jobs.close()


def test_shape_key_update_is_private_until_acknowledged(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code="bpy.data.objects['part/block/body'].shape_key_add(name='Basis')")["ok"]
        before = svc.version()
        (svc.design / "scene.py").write_text(source(width=.3))
        candidate = call(svc, "build")
        assert not candidate["ok"] and candidate["candidate"], candidate
        # The refusal spells out the call that accepts it.
        ids = [issue["id"] for issue in candidate["issues"]]
        assert f"build(candidate={candidate['candidate']!r}, accept_changes={ids!r})" in candidate["error"]
        assert svc.version() == before
        assert not call(svc, "build", candidate=candidate["candidate"], accept_changes=[])["ok"]
        accepted = call(svc, "build", candidate=candidate["candidate"],
                        accept_changes=[issue["id"] for issue in candidate["issues"]],
                        repair_code="bpy.data.objects['part/block/body'].shape_key_add(name='Basis')")
        assert accepted["ok"], accepted
        assert svc.version() != before
    finally:
        svc.jobs.close()


def test_native_pair_exclusions_are_explicit_and_typos_fail(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        code = '''
import json
for name in ('liquid', 'shell'):
    bpy.ops.mesh.primitive_cube_add(size=.1, location=(0,0,.1))
    bpy.context.object.name = name
pairs = [('liquid', 'shell'), ('liquid','block/body'), ('shell','block/body')]
bpy.context.scene['dycraft_declarations'] = json.dumps({'collision_allowances': [
    {'a': a, 'b': b, 'reason': 'intentional physical-content fixture'} for a,b in pairs]})
'''
        assert call(svc, "blender", stage="storyboard", code=code)["ok"]
        checked = call(svc, "compile")
        assert checked["ok"], checked
        report = next(r for r in checked["verifier_reports"] if r["name"] == "NoPenetration")
        # The allowance table itself is declared input and is not echoed back
        # on every report; the count is what proves it was picked up.
        assert report["evidence"]["allowances"]["declared"] == 3
        assert call(svc, "blender", stage="storyboard", code="bpy.context.scene['dycraft_declarations'] = '{\"collision_allowances\": []}'")["ok"]
        assert not call(svc, "compile")["ok"]
        typo = call(svc, "blender", stage="storyboard", code="import json\nbpy.context.scene['dycraft_declarations'] = json.dumps({'collision_allowances': [{'a':'liquid','b':'typo','reason':'test'}]})")
        # Saved, and named by the call that wrote it rather than at the next check.
        assert typo["ok"], typo
        warned, = typo["warnings"]["declaration_errors"]
        assert "unknown 'typo'" in warned["reason"]
        failed = call(svc, "compile")
        # A typo'd allowance does not abort the whole check. It is one finding
        # among the rest: skipped, which makes the check stricter -- the overlap
        # it meant to excuse is reported -- and named.
        assert not failed["ok"]
        report = next(r for r in failed["verifier_reports"] if r["name"] == "NoPenetration")
        assert report["verdict"] == "fail"
        invalid, = report["evidence"]["invalid_allowances"]
        assert "unknown 'typo'" in invalid["reason"]
        assert report["suggested_repairs"]
    finally:
        svc.jobs.close()


def test_generated_joint_outputs_keep_external_drivers_across_source_updates(tmp_path):
    from dycraft.tests.test_scene_layer import _KITCHEN
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(_KITCHEN)
        assert call(svc, "plan", plan=plan())["ok"]
        built = call(svc, "build")
        assert built["ok"], built
        code = '''
control = bpy.data.objects.new('rig_control', None)
bpy.context.collection.objects.link(control)
control['angle'] = .2
control.keyframe_insert('["angle"]', frame=1)
control['angle'] = .4
control.keyframe_insert('["angle"]', frame=2)
joint = bpy.data.objects['joint/cabinet_left__hinge']
driver = joint.driver_add('rotation_euler', 2).driver
variable = driver.variables.new()
variable.name = 'angle'
variable.type = 'SINGLE_PROP'
variable.targets[0].id = control
variable.targets[0].data_path = '["angle"]'
driver.expression = 'angle'
bpy.context.scene.frame_end = 2
'''
        assert call(svc, "blender", code=code, stage="storyboard")["ok"]
        (svc.design / "scene.py").write_text(_KITCHEN.replace("make_cabinet(0.6)", "make_cabinet(0.65)"))
        updated = call(svc, "build")
        assert updated["ok"], updated
        result = call(svc, "probe", expression="[(bpy.context.scene.frame_set(f), bpy.data.objects['joint/cabinet_left__hinge'].evaluated_get(bpy.context.evaluated_depsgraph_get()).rotation_euler[2])[1] for f in (1,2)]")
        assert result["ok"], result
        assert result["result"] == pytest.approx([.2, .4])
        result = call(svc, "probe", expression="[d.driver.variables[0].targets[0].id.name for d in bpy.data.objects['joint/cabinet_left__hinge'].animation_data.drivers if d.data_path == 'rotation_euler'][0]")
        assert result["result"] == "rig_control"
    finally:
        svc.jobs.close()


def test_motion_specialist_finishes_a_private_checked_project_without_the_parent_film(tmp_path):
    write_json(tmp_path / "meta.json", {"contribution_role": "motion"})
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", stage="storyboard", code=STAGE)["ok"]
        done = call(svc, "finish")
        assert done["ok"] and done["contribution_role"] == "motion", done
        manifest = json.loads((tmp_path / "ir.json").read_text())
        validate_manifest(tmp_path, manifest)
        assert "film" not in manifest
        assert (tmp_path / "bundle/animation.blend").is_file()
        assert (tmp_path / "bundle/evaluation/index.json").is_file()
    finally:
        svc.jobs.close()


_CHEAP = {"engine": "WORKBENCH", "width": 128, "height": 96, "threads": 2}


def test_a_shot_that_reproduces_its_images_keeps_the_review_it_already_had(tmp_path):
    """Re-rendering identical evidence must not cost a second review.

    A review is an observation about an image. If the images come back
    pixel-identical there is nothing new to observe, so re-reading them
    establishes nothing and the existing observation still holds. Otherwise
    calling `storyboard()` twice would discard an accepted review and block
    animation authoring again, with no change to the project.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        first = call(svc, "storyboard", render_settings=_CHEAP)
        assert first["ok"] and first["retained"] == []
        assert first["needs_review"] == ["slide/0", "slide/1"]
        digests = {name: e["image_digest"] for name, e in first["states"].items()}
        assert digests and all(isinstance(d, str) for d in digests.values())
        assert call(svc, "accept_storyboard", reviews=state_reviews("Block sits on the floor."))["ok"]

        again = call(svc, "storyboard", render_settings=_CHEAP)
        assert again["ok"], again
        assert again["retained"] == ["slide/0", "slide/1"] and again["needs_review"] == []
        board = again["states"]["slide/0"]
        assert board["reviewed"] and board["review"] == "Block sits on the floor."
        assert {n: e["image_digest"] for n, e in again["states"].items()} == digests
        # Retention carries the observation, never a verdict about this design.
        assert board["retained_from"]
        # Coverage holds, so authoring is not re-blocked.
        assert call(svc, "blender", code=ANIMATE)["ok"]
    finally:
        svc.jobs.close()


def test_a_design_change_that_alters_a_shot_still_demands_a_fresh_review(tmp_path):
    """The subject changes size; the old observation is no longer about it."""
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        first = call(svc, "storyboard", render_settings=_CHEAP)
        assert first["ok"]
        before = {n: e["image_digest"] for n, e in first["states"].items()}
        assert call(svc, "accept_storyboard", reviews=state_reviews("A narrow block."))["ok"]

        (svc.design / "scene.py").write_text(source(width=.6))
        rebuilt = call(svc, "build")
        assert rebuilt["ok"], rebuilt
        # The source update alone re-blocks authoring, as it did before.
        assert not call(svc, "blender", code=ANIMATE)["ok"]

        after = call(svc, "storyboard", render_settings=_CHEAP)
        assert after["ok"], after
        assert after["retained"] == [] and after["needs_review"] == ["slide/0", "slide/1"]
        board = after["states"]["slide/0"]
        assert not board["reviewed"] and "review" not in board
        assert {n: e["image_digest"] for n, e in after["states"].items()} != before
        # And it stays blocked until the changed shot is actually re-reviewed.
        assert not call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("A wider block."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]
    finally:
        svc.jobs.close()


def test_a_source_update_leaves_no_reference_to_the_donor_it_merged_from(tmp_path):
    """`build` succeeded and the next check failed, naming a temp directory.

    Appending the donor's objects drags their dependencies in, and the ones
    the merge does not adopt stay as unused copies carrying a
    `library_weak_reference` to `jobs/<id>/donor/`. `bpy.utils.blend_paths()`
    reports those, so `audit_dependencies` rejected the published workspace at
    the next evaluate — a failure attributed to the check, caused by the build.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        (svc.design / "scene.py").write_text(source(width=.35))
        assert call(svc, "build")["ok"]
        stray = call(svc, "probe", expression="list(bpy.utils.blend_paths(absolute=True, packed=False))")
        assert stray["result"] == [], stray
        assert call(svc, "compile")["ok"]
    finally:
        svc.jobs.close()


def _image(reply, frame):
    """The one path a render reply does not spell out, joined as documented."""
    return Path(reply["frames_dir"]) / (reply["frame_file"] % frame)


def test_a_preview_render_draws_anything_and_delivers_nothing(tmp_path):
    """The look of the film, before the project is ready to certify it.

    `final_render` answers "does this checked project encode correctly"; it
    needs storyboard coverage, passing checks, and a verified replay, and it
    writes the receipt `finish()` reads. None of that helps an author who
    wants to see whether the lighting works. This draws the same pixels and
    promises none of it.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        board = plan()
        board["shots"][0]["frames"] = [0, 1]
        assert call(svc, "plan", plan=board)["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]

        # No storyboard coverage and no passing checks yet: available anyway.
        whole = call(svc, "render", render_settings=_CHEAP)
        assert whole["ok"] and whole["preview"], whole
        assert whole["frames"] == {"start": 0, "end": 1, "count": 2}
        assert Path(whole["path"]).is_file() and Path(whole["frames_dir"]).is_dir()
        assert not (svc.native / "delivery.json").exists()

        assert call(svc, "render", start=1, end=1,
                    render_settings=_CHEAP)["frames"] == {"start": 1, "end": 1, "count": 1}
        assert call(svc, "render", seconds=[0, 0],
                    render_settings=_CHEAP)["frames"]["count"] == 1

        # Renders outside the workspace, so looking never moves the revision.
        version = svc.version()
        by_shot = call(svc, "render", shot="slide", render_settings=_CHEAP)
        assert by_shot["frames"] == {"start": 0, "end": 1, "count": 2}
        assert svc.version() == version

        # Batch cannot take an interval, and says so rather than silently
        # rendering the whole timeline instead.
        refused = call(svc, "render", shot="slide",
                       render_settings={**_CHEAP, "strategy": "batch"})
        assert not refused["ok"] and "batch" in refused["error"]

        # A named set need not be contiguous, which is what makes a strided
        # review an ordinary call: the clip follows frame order regardless of
        # the order asked for, and the encode must not stall on the gap.
        strided = call(svc, "render", frames=[1, 0], render_settings=_CHEAP)
        assert strided["frame_indices"] == [1, 0], strided
        assert strided["frames"] == {"start": 0, "end": 1, "count": 2}
        assert Path(strided["path"]).is_file()
        # The reply names a directory and a spelling, not a path per frame.
        assert "paths" not in strided
        assert [_image(strided, i).name for i in strided["frame_indices"]] == [
            "000001.png", "000000.png"]
        assert all(_image(strided, i).is_file() for i in strided["frame_indices"])

        # An interval through a diagnostic camera: neither op could draw this
        # before the fold, and it is the ordinary way to look at a contact.
        flythrough = call(svc, "render", start=0, end=1, eye=[1, 1, 1], look_at=[0, 0, 0],
                          render_settings=_CHEAP)
        assert flythrough["ok"] and Path(flythrough["path"]).is_file()
        # The placed camera has to change what was *drawn*, not merely what was
        # reported: compare the pixels against the same frames through the
        # native cuts.
        assert (_image(flythrough, 0).read_bytes() != _image(whole, 0).read_bytes())

        # `encode` is a choice, and one frame does not make a clip.
        stills = call(svc, "render", start=0, end=1, encode=False, render_settings=_CHEAP)
        assert stills["ok"] and "path" not in stills
        assert stills["frames"]["count"] == 2
        assert "path" not in call(svc, "render", frames=[0], render_settings=_CHEAP)

        # `snapshot` is the same op with the encode defaulted off, and still
        # takes what it always took plus everything `render` added.
        shot = call(svc, "snapshot", frames=[0, 1], render_settings=_CHEAP)
        assert shot["ok"] and "path" not in shot
        assert [_image(shot, i).name for i in shot["frame_indices"]] == [
            "000000.png", "000001.png"]
        from dycraft.animation.blender.bundle import file_hash
        assert shot["version"] == file_hash(svc.work / "animation.blend")
        # A grid beside the frames, and the frames themselves are hardlinks
        # into the job's own render directory rather than a second copy of a
        # draft that can be hundreds of megabytes. And the reply prices the
        # whole film from what it just drew.
        assert Path(shot["sheet"]).is_file()
        assert _image(shot, 0).stat().st_nlink > 1
        assert shot["budget"]["frames_timed"] == 2
        assert shot["budget"]["measured_s_per_frame"] > 0
        # Two frames are too few to price the film from, and the rate is the
        # render loop's own rather than one that includes Blender start-up.
        assert shot["budget"]["estimated_full_film_seconds"] is None
        assert "Blender start-up, scene load and the first frame's scene sync excluded" \
            in shot["budget"]["basis"]
        assert shot["budget"]["measures"] == "draft"
        assert call(svc, "snapshot", frame=1, render_settings=_CHEAP)["frame_indices"] == [1]
        # A bare call is still frame 0, not the whole film: `render` defaults
        # to everything because a draft wants everything, and `snapshot` must
        # not inherit that.
        assert call(svc, "snapshot", render_settings=_CHEAP)["frame_indices"] == [0]
        assert call(svc, "snapshot", shot="slide",
                    render_settings=_CHEAP)["frames"]["count"] == 2

        # And none of it satisfies completion.
        assert not call(svc, "finish")["ok"]
    finally:
        svc.jobs.close()


def test_a_delivery_render_certifies_frames_given_out_of_order_and_twice(tmp_path):
    """The evaluation that certifies a delivery render walks its frames in
    order, once each, and compares what it walked with what it was asked for;
    asked for `[1, 0, 1]`, it would evaluate them all and then refuse its own
    result as an incomplete or unordered evaluation."""
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Staged."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]
        drawn = call(svc, "render", frames=[1, 0, 1], delivery=True, render_settings=_CHEAP)
        assert drawn["ok"], drawn
        assert drawn["certified_by"]["kind"] == "evaluation", drawn
    finally:
        svc.jobs.close()


def test_every_workbench_render_draws_material_colour_at_the_draft_profile(tmp_path):
    """Default or explicit, a Workbench frame shows the author's colors cheaply.

    Workbench MATERIAL shading reads the `diffuse_color` swatch, which the SDK
    never writes, so the color sync and the FXAA/no-shadow profile must run
    for an explicit `render_settings` too, or a render that picks a resolution
    comes back gray at 8x AA.
    """
    import numpy as np
    from PIL import Image
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())   # an orange block
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        for settings in ({}, {"render_settings": _CHEAP}):
            reply = call(svc, "render", frames=[0], **settings)
            assert reply["ok"], reply
            applied = reply["render_settings"]
            assert applied["engine"] == "BLENDER_WORKBENCH", applied
            assert applied["antialiasing"] == "FXAA", applied
            assert applied["workbench_shading"]["show_shadows"] is False, applied
            rgb = np.asarray(Image.open(_image(reply, 0)).convert("RGB"), dtype=float)
            # The factory swatch is neutral gray, so a render that never
            # synced it peaks at a chroma of a few levels. The block is the
            # only colored thing in shot and covers ~0.3 % of the frame.
            chroma = rgb.max(axis=2) - rgb.min(axis=2)
            assert (chroma > 60).mean() > .001, (settings, float((chroma > 60).mean()))
    finally:
        svc.jobs.close()


def test_probe_reads_a_curve_in_one_pass_and_reports_the_frame_that_failed(tmp_path):
    """One launch per question, not one per sample.

    The launch is the cost of a probe; the eval is not. Without `frames=`,
    asking the same expression at several frames would mean a call each, or a
    comprehension like `[(scene.frame_set(f), ...)[1] for f in (1, 2)]`, which
    side-effects the frame inside an expression and pre-rolls nothing.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Fixture."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]

        read = "bpy.data.objects['part/block/body'].evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.translation.x"
        curve = call(svc, "probe", expression=read, frames=[0, 1])
        assert curve["ok"], curve
        assert [row["frame"] for row in curve["result"]] == [0, 1]
        # The same values one call each, so the batched pass is not merely
        # self-consistent: it agrees with the op it replaces.
        one_at_a_time = [call(svc, "probe", expression=read, frame=i)["result"] for i in (0, 1)]
        assert [row["value"] for row in curve["result"]] == pytest.approx(one_at_a_time)

        # A frame that raises is a row, not a lost call: which frame it turned
        # on is the finding, and the frames that answered still answer.
        partial = call(svc, "probe", frames=[0, 1],
                       expression="1/0 if bpy.context.scene.frame_current > bpy.context.scene.frame_start else 7")
        assert [row.get("value") for row in partial["result"]] == [7, None]
        assert "ZeroDivisionError" in partial["result"][1]["error"]

        both = call(svc, "probe", expression="1", frame=0, frames=[0])
        assert not both["ok"] and "not both" in both["error"]
        backwards = call(svc, "probe", expression="1", frames=[1, 0])
        assert not backwards["ok"] and "increasing" in backwards["error"]
    finally:
        svc.jobs.close()


def test_shot_scoped_checks_resolve_the_interval_the_plan_declared(tmp_path):
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        board = plan()
        board["shots"][0]["frames"] = [0, 1]
        assert call(svc, "plan", plan=board)["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]

        fragment = call(svc, "check_motion", shot="slide")
        assert fragment["partial"] and fragment["frames_checked"]["count"] == 2

        # A shot-scoped compile is still only a diagnostic.
        attempts = svc.ctx.n_compile_attempts
        scoped = call(svc, "compile", shot="slide")
        assert scoped["partial"] and svc.ctx.n_compile_attempts == attempts

        both = call(svc, "check_motion", shot="slide", start=0, end=1)
        assert not both["ok"] and "not two of them" in both["error"]
        unknown = call(svc, "check_motion", shot="nope")
        assert not unknown["ok"] and "unknown shot" in unknown["error"]
    finally:
        svc.jobs.close()


def test_inspect_answers_where_everything_is_without_rendering_anything(tmp_path):
    """The author's structured read of the evaluated scene.

    It existed in the worker from the start, reachable only by the native
    layer's VLM judge. The author's only numeric query was a single Bpy
    expression, which is why runs measured by rendering pictures and looking
    at them, or hand-built their own capture scripts.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "blender", code=ANIMATE, stage="storyboard")["ok"]

        seen = call(svc, "inspect", frames=[0, 1], overview=True)
        assert seen["ok"], seen
        assert [f["frame"] for f in seen["frames"]] == [0, 1]
        assert seen["timeline"]["n_frames"] == 2 and seen["frames"][1]["time"] > 0
        assert seen["frames"][0]["camera"] == "Camera"
        body = [o for o in seen["frames"][0]["objects"] if o["name"] == "part/block/body"]
        assert len(body) == 1 and len(body[0]["bounds"]) == 2
        assert seen["frames"][0]["overview"]["visible_groups"]

        # It reports evaluated motion, which is the point of asking per frame.
        at = [[o for o in f["objects"] if o["name"] == "part/block/body"][0]["position"][0]
              for f in seen["frames"]]
        assert at[1] != at[0]

        # `names` narrows it; the caps belong to the worker and still hold.
        one = call(svc, "inspect", frames=[0], names=["part/block/body"])
        assert [o["name"] for o in one["frames"][0]["objects"]] == ["part/block/body"]
        assert not call(svc, "inspect", frames=[0, 1, 0, 1, 0])["ok"]
    finally:
        svc.jobs.close()


def test_a_storyboard_evaluates_every_staged_frame_in_one_worker(tmp_path):
    """One Blender start for the whole storyboard, whatever it is made of.

    One start per call rather than per state, shot, or clip, because the
    verifier takes an explicit frame set and a start is what actually costs:
    starting a worker costs more than a handful of frames, and after
    frustum-scoped invalidation the common call refreshes one or two states,
    where startup is nearly the whole bill.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        states = plan()["shots"][0]["states"]
        assert len(states) > 1, "fixture must have several states for this to mean anything"

        original, ops = svc.worker, []
        svc.worker = lambda ws, out, req, cancel, **kw: (
            ops.append(req.get("op")) or original(ws, out, req, cancel, **kw))
        boards = call(svc, "storyboard", render_settings=_CHEAP)
        assert boards["ok"], boards
        # One Blender start for the whole call: the render evaluates and
        # verifies the frames it drew, so there is no separate `evaluate`.
        assert ops == ["render"], ops
        # And it really did check every state, not just one -- a fused
        # evaluation still has to be a real one.
        # The shared staged-frame check travels once, not once per state.
        assert "checks" not in boards["states"]["slide/0"]
        assert boards["states"]["slide/0"]["checks_pass"] is True
        assert boards["checks"][0]["frames_checked"]["count"] == len(states)
        assert boards["checks"][0]["partial"] is True
        verdicts = boards["checks"][0]["verifier_reports"]
        assert {r["name"] for r in verdicts} >= {
            "AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied"}
    finally:
        svc.worker = original
        svc.jobs.close()


def test_storyboard_evidence_leaves_the_workspace_and_the_board_keeps_only_the_gate(tmp_path):
    """The panels and their check payloads are not carried by every later op.

    In the workspace, which `submit()` copies and `publish()` hashes on every
    operation, they would make `storyboards.json` hundreds of megabytes of which
    the gate reads a few kB, so rendering a storyboard would make every
    subsequent `snapshot`, `probe`, and `build` slower.

    What must stay true: the gate still works and the images are still
    findable. The revision does still move — acceptance has to be versioned
    with the project or a stale review would survive a design change — but it
    moves by a few kB of gate record rather than by the panels and their
    evidence.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]

        result = call(svc, "storyboard", render_settings=_CHEAP)
        assert result["ok"], result

        # The panels are outside the workspace, and that is where the reply points.
        assert result["view_root"] == str(svc.native)
        assert not (svc.work / "storyboard_views").exists()
        images = [svc.native / e["path"] for e in result["states"].values()]
        assert images and all(p.is_file() for p in images)
        # Spelled as they always were, so the bundle layout and anything that
        # addresses a panel by this list are unaffected by the move.
        assert all(e["path"].startswith("storyboard_views/") for e in result["states"].values())

        # The board carries the gate's fields and a verdict, not the evidence.
        board = json.loads((svc.work / "storyboards.json").read_text())["slide/0"]
        assert board["checks_pass"] is True and board["verdicts"]["NoPenetration"] == "pass"
        assert board["evidence_job"] and "checks" not in board
        assert {"state_revision", "design_id", "reviewed", "image_digest"} <= set(board)
        # Rendered through the timeline, so the panel names the camera the
        # film would use at that frame rather than one the caller chose.
        assert board["camera"] == "Camera"
        # Small enough that copying and hashing it is not a cost of its own.
        assert (svc.work / "storyboards.json").stat().st_size < 16_384

        # The author still gets the full payload in the reply that reports it.
        assert result["checks"][0]["verifier_reports"]

        assert call(svc, "accept_storyboard", reviews=state_reviews("Block on the floor."))["ok"]
        # Retention still works across a call, reading the images from their new home.
        again = call(svc, "storyboard", render_settings=_CHEAP)
        assert again["retained"] == ["slide/0", "slide/1"] and again["needs_review"] == []
    finally:
        svc.jobs.close()


def test_a_board_for_a_shot_the_plan_dropped_does_not_outlive_it(tmp_path):
    """A board for a state the plan no longer declares is dropped.

    `missing_storyboards` only walks the current shot list, so without this a
    renamed or split shot's old board would stay in `storyboards.json` forever,
    reviewed and unreachable.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Block on the floor."))["ok"]
        assert "slide/0" in json.loads((svc.work / "storyboards.json").read_text())
        panels = svc.views_root / content_hash("slide/0")
        assert panels.is_dir()

        renamed = plan()
        renamed["shots"][0]["id"] = "traverse"
        assert call(svc, "plan", plan=renamed)["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]

        boards = json.loads((svc.work / "storyboards.json").read_text())
        assert set(boards) == {"traverse/0", "traverse/1"}, boards
        assert not panels.exists(), "the dropped shot's panels are gone too"
    finally:
        svc.jobs.close()


def test_delivery_reuses_the_compile_it_already_passed_and_still_guards_the_render(tmp_path):
    """Delivery reuses the cached whole-film check, and the render is still guarded.

    `final_render` writes nothing into the project, so its content version
    is the one `compile()` just checked, the cache hits, and the timeline is
    not evaluated a second time for one delivery.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Staged."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Animated."))["ok"]

        compiled = call(svc, "compile")
        assert compiled["ok"], compiled
        film = call(svc, "final_render", render_settings={**_CHEAP, "width": 64, "height": 64})
        assert film["ok"], film
        receipt = json.loads((svc.native / "delivery.json").read_text())
        assert receipt["checks"]["validation_reused"], "delivery re-evaluated an unchanged project"

        # The guard: a render that changes the project must still fail.
        original = svc.worker
        def mutating(ws, out, req, cancel, **kw):
            result = original(ws, out, req, cancel, **kw)
            if req.get("op") == "render" and req.get("verify_replay"):
                (Path(ws) / "stray_bake.txt").write_text("written while rendering")
            return result
        svc.worker = mutating
        again = call(svc, "final_render", render_settings={**_CHEAP, "width": 64, "height": 64})
        assert not again["ok"] and "rendering changed project dependencies" in again["error"]
    finally:
        svc.worker = original
        svc.jobs.close()


def test_a_long_timeline_checked_in_parallel_agrees_with_one_process(tmp_path, monkeypatch):
    """Four processes over a split timeline must reach the single-process verdict.

    The split is only sound because each chunk *evaluates* a few frames either
    side of what it *owns* and reports on the owned range alone, so a step, a
    spike's neighborhood and a contact that began earlier are all judged with
    the evidence a whole-timeline pass would have had, and every frame is
    counted once. What legitimately differs is per-process bookkeeping -- BVH
    builds and pair-cache hits -- because there are now four caches and the
    shared frames are evaluated twice.
    """
    from dycraft.animation import checking as animation_checking
    # Four workers on any host: the split is what this compares.
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: None)
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        staged = call(svc, "blender", code=STAGE.replace("s.frame_end = 2", "s.frame_end = 60"),
                      stage="storyboard")
        assert staged["ok"], staged
        boards = call(svc, "storyboard", render_settings={"engine": "WORKBENCH", "width": 128,
                                                          "height": 96, "threads": 2})
        assert boards["ok"], boards
        assert call(svc, "accept_storyboard", reviews=state_reviews("Fixture inspects the slide."))["ok"]
        animated = call(svc, "blender", code='''
root = bpy.data.objects['joint/world__block']
for frame, x in ((1, 0.), (20, .05), (40, -.05), (60, .05)):
    root.location.x = x
    root.keyframe_insert('location', frame=frame)
''')
        assert animated["ok"], animated

        svc.CHECK_WORKERS = 1
        single = call(svc, "compile", timeout=600)
        assert single["frames_checked"]["count"] == 60, single["frames_checked"]
        assert svc.chunk_plan() is None, "one worker must not split anything"

        svc.CHECK_WORKERS = 4
        svc.validation_cache = {}
        chunks = svc.chunk_plan()
        assert chunks and len(chunks) == 4, chunks
        assert [owned for owned, _, _ in chunks] == [0, 15, 30, 45]
        assert chunks[1][2][0] == 11 and chunks[1][1] == 29, chunks[1]
        # `fresh`: the same content again would otherwise re-check against the
        # single-process check and carry every frame.
        parallel = call(svc, "compile", fresh=True, timeout=600)
        assert "carried_over" not in parallel

        assert parallel["ok"] == single["ok"]
        assert parallel["frames_checked"] == single["frames_checked"]
        evaluated = Path(parallel["evaluation_path"])
        assert sorted(int(p.stem) for p in evaluated.glob("[0-9]*.bin")) == list(range(60))
        assert json.loads((evaluated / "index.json").read_text())["frames"] == list(range(60))

        # `stats` is the per-process bookkeeping the docstring above says must
        # differ — BVH builds and pair-cache hits, because four workers hold
        # four caches and evaluate the shared context frames twice. It nests
        # under one key now; naming the four old top-level keys stripped
        # nothing, and this fixture only passed because they were all zero.
        noise = {"stats"}
        def comparable(reports):
            return [{**report, "evidence": {k: v for k, v in report["evidence"].items()
                                            if k not in noise}} for report in reports]
        assert comparable(parallel["verifier_reports"]) == comparable(single["verifier_reports"])
    finally:
        svc.jobs.close()


def test_a_delivery_render_hands_its_frames_to_the_next_one(tmp_path):
    """A film drawn in pieces, certified as one.

    Drawing and certification are separate, so a render that dies part-way
    keeps the frames it finished: a resubmission against the same project
    version draws only the frames still missing.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Staged."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Animated."))["ok"]

        # Drawing never waits on a verdict. With no check at this
        # version the frames are replay-verified against an evaluation-only
        # pass, and listed as unchecked.
        early = call(svc, "render", start=0, end=0, delivery=True, render_settings=_CHEAP)
        assert early["ok"] and early["certified_by"]["kind"] == "evaluation", early
        assert {"kind": "unchecked", "frames": [[0, 0]]} in early["unmet"], early
        assert call(svc, "compile")["ok"]

        first = call(svc, "render", start=0, end=0, delivery=True, render_settings=_CHEAP)
        assert first["ok"] and first["delivery_grade"], first
        assert first["frames"] == {"start": 0, "end": 0, "count": 1}
        # No clip: these frames exist to be composed, and encoding a fragment
        # on the way there is ffmpeg time for something nobody watches.
        assert "path" not in first

        # Half a film is finished here: the missing range is drawn, not refused.
        short = call(svc, "final_render", frames_from=[first["job_id"]])
        assert short["ok"], short
        assert short["composed_from"] == {first["job_id"]: "0", "drawn_here": "1"}, short
        assert short["admitted"] == {"same_version": 1, "by_identity": 0, "draft": 0, "drawn_here": 1}

        # A job that drew no delivery frames is refused by name, and so is one
        # that never existed: admissibility is read from what the harness
        # recorded, never from a path or a claim.
        unknown = call(svc, "final_render", frames_from=[first["job_id"], "0" * 16])
        assert not unknown["ok"] and "drew no frames final_render can use" in unknown["error"], unknown

        second = call(svc, "render", start=1, end=1, delivery=True, render_settings=_CHEAP)
        assert second["ok"] and second["version"] == first["version"], second
        overlap = call(svc, "final_render",
                       frames_from=[first["job_id"], first["job_id"], second["job_id"]])
        assert not overlap["ok"] and "drawn by both" in overlap["error"], overlap

        film = call(svc, "final_render", frames_from=[first["job_id"], second["job_id"]])
        assert film["ok"], film
        assert Path(film["path"]).is_file() and film["n_frames"] == 2
        assert film["composed_from"] == {first["job_id"]: "0", second["job_id"]: "1"}
        # The optics of every delivered frame, recovered from the trail each
        # render wrote as it drew rather than from a result only a job that
        # returns can produce.
        assert [view["frame"] for view in film["views"]] == [0, 1]
        receipt = json.loads((svc.native / "delivery.json").read_text())
        assert receipt["composed_from"] == [first["job_id"], second["job_id"]]
        assert receipt["checks"]["validation_reused"], "composition re-evaluated the project"
        # Same receipt a one-pass render writes, so completion cannot tell.
        assert call(svc, "finish", criteria=CRITERIA)["ok"]
    finally:
        svc.jobs.close()


def test_frames_drawn_against_another_revision_are_named_and_redrawn(tmp_path):
    """The identity that makes composition safe, and the one that breaks it.

    A frame belongs to the content version it was drawn from. Nothing about a
    finished PNG says which one that was, which is why admission reads the
    job's record instead of the file -- and why an edit between drawing and
    submitting keeps those frames out of the film. `final_render` draws them
    again itself and names the job it could not use.
    """
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Staged."))["ok"]
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "storyboard", render_settings=_CHEAP)["ok"]
        assert call(svc, "accept_storyboard", reviews=state_reviews("Animated."))["ok"]
        assert call(svc, "compile")["ok"]
        drawn = call(svc, "render", delivery=True, render_settings=_CHEAP)
        assert drawn["ok"], drawn

        # An ordinary appearance edit, and now the frames describe a project
        # that no longer exists.
        assert call(svc, "blender", code="bpy.context.scene.camera.data.lens = 41")["ok"]
        assert call(svc, "compile")["ok"]
        stale = call(svc, "final_render", frames_from=[drawn["job_id"]])
        assert stale["ok"], stale
        assert drawn["version"] in stale["not_admitted"][drawn["job_id"]], stale
        assert stale["admitted"]["same_version"] == 0 and stale["admitted"]["drawn_here"] == 2

        # Settings are the other half of the identity, and a disagreement
        # names the keys rather than the dicts.
        again = call(svc, "render", delivery=True, render_settings=_CHEAP)
        wider = call(svc, "final_render", frames_from=[again["job_id"]],
                     render_settings={**_CHEAP, "width": 160})
        assert not wider["ok"] and "disagree on width" in wider["error"], wider
        assert "strategy" not in wider["error"], "composition pins strategy; it is not a disagreement"
        assert call(svc, "final_render", frames_from=[again["job_id"]])["ok"]
    finally:
        svc.jobs.close()


def test_dispatch_reads_the_projects_clips_from_real_blender(tmp_path):
    """`project_clips` against the real worker: the clip names come from the
    NLA, and a project without strips says so in the refusal."""
    from dycraft.animation.dispatch import project_clips
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        assert project_clips(svc) == {}
        clip = ("track = root.animation_data.nla_tracks.new()\n"
                "track.strips.new('lift', 1, root.animation_data.action)\n")
        edited = call(svc, "blender", code=STAGE + ANIMATE + clip, stage="storyboard")
        assert edited["ok"], edited
        assert "lift" in project_clips(svc)
    finally:
        svc.jobs.close(timeout_s=10)
