"""After a repair, re-check only what it changed, and say so.

A whole-film `compile` after an earlier one evaluates every frame, diffs the
signatures against that check, re-verifies the frames the change can reach,
and carries every other frame's rows. These are the four cases, on the real
fixture project.
"""
import json

import pytest

from dycraft.tests._animation_fixtures import (
    STAGE, call, plan, service, source, state_reviews)

pytestmark = pytest.mark.expensive


@pytest.fixture(autouse=True)
def _workers_not_bound_by_free_memory(monkeypatch):
    """The worker counts set here are what the cases compare, on any host."""
    from dycraft.animation import checking as animation_checking
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: None)

SLIDE = '''
root = bpy.data.objects['joint/world__block']
for frame, x in ((0, 0.), (12, .04), (24, .08), (36, .12)):
    root.location.x = x
    root.keyframe_insert('location', frame=frame)
'''

#: Fails on frames 0-5 and only there: the block rests on the floor.
APART = '''
import json
s = bpy.context.scene
d = json.loads(s['dycraft_declarations'])
d['events'] = [{'name': 'lifted', 'predicate': 'separated', 'a': 'block/body', 'b': 'room.floor',
                'distance': .01, 'window': [0, 5]}]
s['dycraft_declarations'] = json.dumps(d)
'''

#: Lifts the block on its last frames only: a new channel, flat before its
#: first key. (Re-keying the slide's last key would re-derive every earlier
#: key's automatic handles, and so change nearly every frame.)
LATE = '''
root = bpy.data.objects['joint/world__block']
z = root.location.z
root.keyframe_insert('location', index=2, frame=30)
root.location.z = z + .05
root.keyframe_insert('location', index=2, frame=36)
'''


@pytest.fixture
def project(tmp_path):
    svc = service(tmp_path)
    (svc.design / "scene.py").write_text(source())
    assert call(svc, "plan", plan=plan())["ok"]
    assert call(svc, "build")["ok"]
    staged = call(svc, "blender", code=STAGE.replace("s.frame_end = 2", "s.frame_end = 37"),
                  stage="storyboard")
    assert staged["ok"], staged
    assert call(svc, "storyboard", render_settings={"engine": "WORKBENCH", "width": 128,
                                                    "height": 96, "threads": 2})["ok"]
    assert call(svc, "accept_storyboard", reviews=state_reviews("fixture"))["ok"]
    assert call(svc, "blender", code=SLIDE)["ok"]
    yield svc
    svc.jobs.close()


def comparable(reports):
    return [{**r, "evidence": {k: v for k, v in r["evidence"].items() if k != "stats"}}
            for r in reports]


def strip(reports):
    return [{k: v for k, v in r.items() if k != "scope"} for r in comparable(reports)]


def test_a_merged_re_check_equals_a_full_one_and_names_what_it_carried(project):
    first = call(project, "compile", timeout=600)
    assert first["frames_checked"]["count"] == 37 and "carried_over" not in first
    assert call(project, "blender", code=LATE)["ok"]
    merged = call(project, "compile", timeout=600)
    carried = merged["carried_over"]
    assert carried["from_job"] and carried["frames"][0][0] == 0
    # The late keys reach frames from the key before them (24) on; the merge
    # re-verified those, with their neighborhood, and carried the rest.
    scope = merged["verifier_reports"][0]["scope"]
    assert scope["reverified"][0][0] >= 20 and scope["carried"] == carried["frames"]
    project.validation_cache = {}     # the same content: re-check it from scratch
    full = call(project, "compile", fresh=True, timeout=600)
    assert "carried_over" not in full
    assert strip(merged["verifier_reports"]) == strip(full["verifier_reports"])
    assert merged["ok"] == full["ok"]


def test_a_failure_outside_the_changed_frames_stays_reported(project):
    assert call(project, "blender", code=APART)["ok"]
    first = call(project, "compile", timeout=600)
    events = next(r for r in first["verifier_reports"] if r["name"] == "EventsSatisfied")
    assert events["verdict"] == "fail"
    assert call(project, "blender", code=LATE)["ok"]
    merged = call(project, "compile", timeout=600)
    assert "carried_over" in merged
    events = next(r for r in merged["verifier_reports"] if r["name"] == "EventsSatisfied")
    row, = events["evidence"]["events"]
    assert events["verdict"] == "fail" and row["unsatisfied"] == [[0, 5]]


def test_fresh_and_a_declaration_change_each_force_a_full_check(project):
    call(project, "compile", timeout=600)
    assert call(project, "blender", code=LATE)["ok"]
    assert "carried_over" not in call(project, "compile", fresh=True, timeout=600)
    # An event changes no frame's signature, and changes a verdict.
    assert call(project, "blender", code=APART)["ok"]
    redeclared = call(project, "compile", timeout=600)
    assert "carried_over" not in redeclared
    assert "scope" not in redeclared["verifier_reports"][0]


def test_a_parallel_re_check_equals_one_worker_s(project):
    """With chunk workers, a repeat compile evaluates its chunks in parallel and
    measures what changed from the merged evaluation. Against the same base it
    re-checks exactly as one worker does: the same reports, the same scope, the
    same frames carried. (37 frames take two workers, not four.)"""
    call(project, "compile", timeout=600)
    base = dict(project.last_whole)
    assert call(project, "blender", code=LATE)["ok"]
    project.CHECK_WORKERS = 2
    assert project.chunk_plan(), "the fixture is long enough to split in two"
    parallel = call(project, "compile", timeout=600)
    project.CHECK_WORKERS, project.last_whole, project.validation_cache = 1, base, {}
    single = call(project, "compile", timeout=600)
    assert parallel["carried_over"]["frames"] == single["carried_over"]["frames"]
    assert comparable(parallel["verifier_reports"]) == comparable(single["verifier_reports"])
    assert parallel["ok"] == single["ok"]
    project.validation_cache = {}
    full = call(project, "compile", fresh=True, timeout=600)
    assert strip(parallel["verifier_reports"]) == strip(full["verifier_reports"])


def test_a_parallel_re_check_after_a_declaration_change_measures_every_frame(project):
    call(project, "compile", timeout=600)
    assert call(project, "blender", code=APART)["ok"]
    project.CHECK_WORKERS = 2
    redeclared = call(project, "compile", timeout=600)
    assert "carried_over" not in redeclared and "scope" not in redeclared["verifier_reports"][0]
    project.CHECK_WORKERS, project.validation_cache = 1, {}
    full = call(project, "compile", fresh=True, timeout=600)
    assert comparable(redeclared["verifier_reports"]) == comparable(full["verifier_reports"])
