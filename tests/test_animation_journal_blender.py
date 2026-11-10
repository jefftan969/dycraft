"""The journal, driven through real Blender jobs rather than stubs.

`tests/test_animation_journal.py` pins the store's own rules. This pins the
thing they exist for: after a run, can an inspection agent recover the source a
call ran and say what that call changed in the project — without opening a
`.blend` and without re-running anything.
"""
import json

import pytest

from dycraft.animation.blender.scene_digest import diff
from dycraft.tests._animation_fixtures import (
    ANIMATE, STAGE, call, plan, service, source, state_reviews)

pytestmark = pytest.mark.expensive

#: What an agent actually does: put the logic in a helper and point at it, so
#: the call records only the pointer.
HELPER = "SLIDE_TO = 0.05\n"
VIA_HELPER = ("exec(open('/workspace/motion.py').read())\n"
              "root = bpy.data.objects['joint/world__block']\n"
              "root.location.x = 0\n"
              "root.keyframe_insert('location', frame=1)\n"
              "root.location.x = SLIDE_TO\n"
              "root.keyframe_insert('location', frame=2)\n")


#: The animation layer refuses an animation-stage edit until every planned shot
#: has been staged, rendered, and reviewed, so reaching one takes the whole
#: preamble. It also gives these tests four different operation kinds to record.
def _setup(svc):
    (svc.design / "scene.py").write_text(source())
    assert call(svc, "plan", plan=plan())["ok"]
    assert call(svc, "build")["ok"]
    assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
    assert call(svc, "storyboard", render_settings={
        "engine": "WORKBENCH", "width": 128, "height": 96, "threads": 2})["ok"]
    assert call(svc, "accept_storyboard",
                reviews=state_reviews("Fixture: the block rests on the floor in both states."))["ok"]


def test_the_journal_reconstructs_what_each_operation_ran_and_changed(tmp_path):
    svc = service(tmp_path)
    try:
        _setup(svc)
        helpers = svc.root / "animation_src"
        helpers.mkdir(exist_ok=True)
        # Written with the agent's own tools, outside any harness operation --
        # which is exactly why the call boundary is the only place to catch it.
        (helpers / "motion.py").write_text(HELPER)
        assert call(svc, "blender", code=VIA_HELPER, name="slide.py")["ok"]

        journal = svc.journal
        records = journal.records()
        assert [r["seq"] for r in records] == list(range(1, len(records) + 1))
        assert [r["operation"] for r in records] == \
            ["plan", "build", "blender", "storyboard", "accept_storyboard", "blender"]
        assert all(r["ok"] for r in records), records

        edit = records[-1]
        assert edit["job_id"] and edit["duration_s"] > 0
        assert edit["published"] and edit["version_after"] != edit["version_before"]

        # 1. What source ran -- the call string, byte for byte.
        assert edit["args"]["name"] == "slide.py"
        assert journal.resolve_arg(edit["args"]["code"]) == VIA_HELPER

        # 2. ...including the helper the call string only points at. The session
        # copies the contents of `animation_src/` to the workspace root, so the
        # call reads `/workspace/motion.py` while the manifest keeps the
        # authored path -- which is the one a reader is looking for.
        manifest = journal.read_manifest(edit["src_manifest"])
        entry = manifest["files"]["animation_src/motion.py"]
        assert entry["stored"] and journal.read_blob(entry["sha"]).decode() == HELPER
        # `design/` is versioned by the same manifest: `build` compiles it.
        assert "design/scene.py" in manifest["files"]

        # 3. What it changed, without opening a blend. The immediately
        # preceding operation was `accept_storyboard`, which publishes a
        # version but never touches the scene and so writes no digest -- which
        # is why the comparison walks back to the last one that did.
        previous, before = journal.latest_digest(before=edit["seq"])
        assert previous < edit["seq"]
        after = journal.read_digest(edit["seq"])
        changed = diff(before, after)
        assert changed is not None
        assert "joint/world__block" in changed["objects"]["modified"]
        # Keys carry the action, the slot that owns the channel, and the
        # channel itself, so one action shared between two objects stays
        # unambiguous.
        assert any(c.startswith("joint/world__blockAction|")
                   and c.endswith("location[0]")
                   for c in changed["curves"]["added"]), changed["curves"]

        # And the curve's own summary says how it moved, not merely that it did.
        moved = next(c for c in changed["curves"]["added"] if c.endswith("location[0]"))
        first_frame, last_frame, low, high, _mean = after["curves"][moved]["stats"]
        assert (first_frame, last_frame) == (1.0, 2.0)
        assert low == pytest.approx(0.0) and high == pytest.approx(0.05)
    finally:
        svc.jobs.close()


def test_an_operation_that_reads_the_project_records_no_version_boundary(tmp_path):
    """`probe` and `snapshot` neither publish nor save, so they fingerprint nothing.

    Capturing source for them would cost a tree hash per call for an answer
    that cannot differ from the previous boundary's.
    """
    svc = service(tmp_path)
    try:
        _setup(svc)
        assert call(svc, "probe", expression="len(bpy.data.objects)")["ok"]
        probe = svc.journal.records()[-1]
        assert probe["operation"] == "probe"
        assert probe["ok"] and probe["digest"] is None
        assert probe["src_manifest"] is None and not probe["published"]
        # The scene is still readable at that point in the timeline: the last
        # boundary's digest carries forward.
        seq, digest = svc.journal.latest_digest(before=probe["seq"])
        assert seq < probe["seq"] and digest["totals"]["objects"] > 0
    finally:
        svc.jobs.close()


def test_a_failed_edit_is_recorded_with_its_error_and_leaves_the_project_alone(tmp_path):
    svc = service(tmp_path)
    try:
        _setup(svc)
        current = svc.journal.latest_digest()
        assert not call(svc, "blender", code="raise RuntimeError('deliberate')")["ok"]
        failed = svc.journal.records()[-1]
        assert failed["operation"] == "blender" and failed["ok"] is False
        assert "deliberate" in failed["error"]
        assert not failed["published"] and failed["digest"] is None
        # Its source is still captured: a failed call is the one you most want
        # to read afterwards.
        assert svc.journal.resolve_arg(failed["args"]["code"]) == \
            "raise RuntimeError('deliberate')"
        assert failed["src_manifest"]
        # The project did not move, so the previous digest is still current.
        assert svc.journal.latest_digest() == current
    finally:
        svc.jobs.close()


def test_an_edit_that_changes_nothing_is_visible_as_such(tmp_path):
    """Many operations in a run change no curve at all.

    Telling those apart from the ones that do is what makes bisecting a
    regression cheap.
    """
    svc = service(tmp_path)
    try:
        _setup(svc)
        assert call(svc, "blender", code=ANIMATE)["ok"]
        moved = svc.journal.records()[-1]
        assert call(svc, "blender", code="result = {'noop': True}")["ok"]
        idle = svc.journal.records()[-1]

        assert idle["published"], "it still republished the project"
        assert diff(svc.journal.read_digest(moved["seq"]),
                    svc.journal.read_digest(idle["seq"])) is None
    finally:
        svc.jobs.close()


def test_the_timeline_is_the_join_an_inspection_agent_would_write(tmp_path):
    """One call gives the whole run as "operation N changed this much"."""
    svc = service(tmp_path)
    try:
        _setup(svc)
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "blender", code="result = {'noop': True}")["ok"]

        rows = svc.journal.timeline()
        assert [r["operation"] for r in rows] == \
            ["plan", "build", "blender", "storyboard", "accept_storyboard",
             "blender", "blender"]
        # The operations that never touch the scene report no change at all...
        assert [r["operation"] for r in rows if r["changed"] is None] == \
            ["plan", "build", "storyboard", "accept_storyboard", "blender"]
        # ...and the one that animated the block says what it moved.
        animated = rows[-2]
        assert animated["operation"] == "blender" and animated["published"]
        assert animated["changed"]["objects"] >= 1
        assert animated["changed"]["curves"] >= 3
        assert "joint/world__block" in animated["detail"]["objects"]["modified"]
    finally:
        svc.jobs.close()


def test_a_broken_journal_cannot_break_the_run(tmp_path, monkeypatch):
    """It runs from the job's `finally`, so a failure there would replace
    whatever the operation was about to return. A run must be able to lose its
    record and still deliver a film."""
    svc = service(tmp_path)
    try:
        _setup(svc)

        def explode(*a, **k):
            raise RuntimeError("journal is having a bad day")

        monkeypatch.setattr(svc.journal, "store_digest", explode)
        monkeypatch.setattr(svc.journal, "capture_sources", explode)
        monkeypatch.setattr(svc.journal, "record", explode)
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert call(svc, "probe", expression="len(bpy.data.objects)")["ok"]
    finally:
        svc.jobs.close()


def test_turning_the_journal_off_costs_the_worker_nothing(tmp_path, monkeypatch):
    """The toggle has to reach the worker, or a disabled journal still pays
    40-170 ms per saving operation to fingerprint a scene nobody will read."""
    monkeypatch.setenv("DYCRAFT_ANIMATION_JOURNAL", "0")
    svc = service(tmp_path)
    try:
        _setup(svc)
        assert call(svc, "blender", code=ANIMATE)["ok"]
        assert not (svc.native / "journal" / "journal.jsonl").exists()
        assert not list((svc.native / "jobs").rglob("digest.json"))
    finally:
        svc.jobs.close()
