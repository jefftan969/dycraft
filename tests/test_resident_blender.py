"""One Blender process, many requests -- and the same answers as many processes."""
import threading
import time

import pytest

from dycraft.tests._animation_fixtures import STAGE, call, plan, service, source

pytestmark = pytest.mark.expensive


def _built(root):
    svc = service(root)
    (svc.design / "scene.py").write_text(source())
    assert call(svc, "plan", plan=plan())["ok"]
    assert call(svc, "build")["ok"]
    assert call(svc, "blender", code=STAGE, stage="storyboard")["ok"]
    return svc


def test_a_resident_session_answers_what_separate_processes_answer(tmp_path):
    """The equivalence residency has to earn before it is worth anything.

    Same project, same requests, one process instead of N. Anything that
    differs here is state leaking between calls, which is the whole risk of
    keeping a scene in memory.
    """
    from dycraft.animation.blender.resident import ResidentBlender
    from dycraft.animation.blender.runtime import run_worker

    svc = _built(tmp_path)
    try:
        asks = [{"op": "probe", "code": "bpy.context.scene.frame_end"},
                {"op": "probe", "code": "len(bpy.data.objects)"},
                {"op": "probe", "code": "bpy.context.scene.render.fps"},
                {"op": "inspect", "frames": [0]},
                {"op": "probe", "code": "bpy.context.scene.frame_start"}]
        fresh = [run_worker(svc.work, tmp_path / f"one_{i}", dict(ask))
                 for i, ask in enumerate(asks)]

        session = ResidentBlender(svc.work, tmp_path / "resident_out",
                                  source_roots=[svc.work])
        try:
            session.start()
            resident = [session.request(dict(ask), tmp_path / "resident_out" / f"r_{i}")
                        for i, ask in enumerate(asks)]
        finally:
            session.close()

        assert resident == fresh
        # And it really was one process for all of them.
        assert session.calls == len(asks) and session.generation == 1
    finally:
        svc.jobs.close()


def test_a_failed_request_does_not_poison_the_next_one(tmp_path):
    """Today the process dies and its successor re-reads the file.

    A resident session has to reach the same end state -- the last successful
    save -- without a new process, or one bad edit quietly corrupts every
    answer after it.
    """
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    try:
        with ResidentBlender(svc.work, tmp_path / "out") as session:
            before = session.request({"op": "probe", "code": "len(bpy.data.objects)"},
                                     tmp_path / "out" / "a")
            with pytest.raises(RuntimeError):
                session.request({"op": "edit", "code": "bpy.data.objects.new('x', None)\n1/0"},
                                tmp_path / "out" / "b")
            after = session.request({"op": "probe", "code": "len(bpy.data.objects)"},
                                    tmp_path / "out" / "c")
            assert after == before, "the failed edit's object survived into the next request"
            assert session.generation == 1, "recovery must not have cost a new process"
    finally:
        svc.jobs.close()


def test_an_edit_is_visible_to_the_next_request_without_reopening(tmp_path):
    """The point of residency, stated as behavior rather than as a timing."""
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    try:
        with ResidentBlender(svc.work, tmp_path / "out") as session:
            session.request({"op": "edit", "code": "bpy.context.scene.camera.data.lens = 41"},
                            tmp_path / "out" / "a")
            lens = session.request({"op": "probe", "code": "bpy.context.scene.camera.data.lens"},
                                   tmp_path / "out" / "b")
            assert lens == 41
            # The saved project agrees, because `edit` saves -- which is what
            # makes skipping the reopen safe rather than merely fast.
            from dycraft.animation.blender.runtime import run_worker
            assert run_worker(svc.work, tmp_path / "check",
                              {"op": "probe", "code": "bpy.context.scene.camera.data.lens"}) == 41
    finally:
        svc.jobs.close()


def test_fresh_restarts_the_process_rather_than_reloading(tmp_path):
    """A reopen clears the scene; only a new process clears the interpreter."""
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    try:
        with ResidentBlender(svc.work, tmp_path / "out") as session:
            session.request({"op": "probe", "code": "1"}, tmp_path / "out" / "a")
            first = session.diagnostics
            assert first["alive"] and first["generation"] == 1 and first["calls"] == 1
            session.restart()
            session.request({"op": "probe", "code": "1"}, tmp_path / "out" / "b")
            assert session.diagnostics["generation"] == 2
            assert session.diagnostics["calls"] == 1, "a restart resets the call count"
    finally:
        svc.jobs.close()


def test_a_session_survives_the_publish_that_replaces_the_project(tmp_path):
    """The reason the session binds the project's parent rather than the project.

    `publish()` swaps the project by renaming one directory over another. A
    child holding the project directory would keep the replaced inode and go
    on answering about a tree nobody can reach any more -- fast, and wrong. A
    child holding the parent sees the swap, and `reload()` is the defined
    moment at which it re-reads.
    """
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    try:
        session = ResidentBlender(svc.work, tmp_path / "out",
                                  bind_root=svc.native)
        with session:
            assert session.request({"op": "probe", "code": "bpy.context.scene.camera.data.lens"},
                                   tmp_path / "out" / "a") != 43
            # A publishing op, through the ordinary path, which renames the
            # project directory out from under the running child.
            published = call(svc, "blender", stage="storyboard",
                             code="bpy.context.scene.camera.data.lens = 43")
            assert published["ok"], published
            session.reload()
            assert session.request({"op": "probe", "code": "bpy.context.scene.camera.data.lens"},
                                   tmp_path / "out" / "b") == 43
            assert session.generation == 1, "surviving a publish must not need a new process"
    finally:
        svc.jobs.close()


def test_an_output_outside_the_bound_root_is_refused_rather_than_silently_missing(tmp_path):
    """The sandbox binds one output root; anything else is unreachable."""
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    try:
        session = ResidentBlender(svc.work, tmp_path / "out")
        with pytest.raises(RuntimeError, match="outside"):
            session.request({"op": "probe", "code": "1"}, tmp_path / "elsewhere")
    finally:
        svc.jobs.close()


def test_cancelling_a_resident_request_ends_the_process_it_was_mutating(tmp_path):
    """There is no unwind for a half-applied edit in a shared scene.

    Two cases, and they differ: a request canceled before it is sent has
    touched nothing, so it costs nothing; a request canceled in flight has
    already begun mutating the scene the next request would see, and the only
    honest recovery is to end the process.
    """
    from dycraft.animation.blender.resident import ResidentBlender

    svc = _built(tmp_path)
    session = ResidentBlender(svc.work, tmp_path / "out")
    try:
        session.start()
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            session.request({"op": "probe", "code": "1"}, tmp_path / "out" / "a", cancel=cancel)
        assert session.diagnostics["alive"], "nothing ran, so nothing had to be torn down"

        # In flight: cancel once the request is already with the child.
        late = threading.Event()
        threading.Timer(.05, late.set).start()
        with pytest.raises(RuntimeError, match="cancelled"):
            session.request({"op": "probe", "code": "__import__('time').sleep(30)"},
                            tmp_path / "out" / "b", cancel=late)
        assert not session.diagnostics["alive"]
    finally:
        session.close()
        svc.jobs.close()


def test_an_interactive_op_returns_its_result_without_polling(tmp_path):
    """The API change residency exists to enable.

    An interactive op runs in the caller's thread against the project itself
    and hands back the result, with no polling. It carries the id of the record
    it ran under, already terminal: that id is what `job_status`/`job_result`
    answer afterwards and what `job_cancel` accepts.
    """
    svc = _built(tmp_path)
    try:
        direct = svc.submit("probe", expression="len(bpy.data.objects)")
        assert direct["ok"] and isinstance(direct["result"], int)
        assert direct["status"] == "completed", direct
        assert svc.jobs.status(direct["job_id"])["status"] == "completed"

        # The same op, asked for as a job, still behaves exactly as before.
        handle = svc.submit("probe", expression="len(bpy.data.objects)", batch=True)
        assert handle["job_id"] and handle["status"] == "queued"
        assert call(svc, "probe", expression="len(bpy.data.objects)")["result"] == direct["result"]
    finally:
        svc.jobs.close()


def test_an_interactive_edit_lands_in_the_project_and_advances_its_revision(tmp_path):
    """In place, with a counter for identity rather than a hash of the tree."""
    svc = _built(tmp_path)
    try:
        first = svc.submit("blender", stage="storyboard",
                           code="bpy.context.scene.camera.data.lens = 44")
        assert first["ok"], first
        assert first["version"].startswith("r"), first["version"]
        assert svc.submit("probe", expression="bpy.context.scene.camera.data.lens")["result"] == 44

        second = svc.submit("blender", stage="storyboard",
                            code="bpy.context.scene.camera.data.lens = 45")
        assert int(second["version"][1:]) > int(first["version"][1:])
    finally:
        svc.jobs.close()


def test_a_failing_interactive_edit_leaves_the_project_as_it_was(tmp_path):
    """A failing op leaves the project intact: the worker saves only after success.

    A failing `build` must never corrupt the project, though in-place ops do not
    work on a copy.
    """
    svc = _built(tmp_path)
    try:
        svc.submit("blender", stage="storyboard", code="bpy.context.scene.camera.data.lens = 46")
        broken = svc.submit("blender", stage="storyboard",
                            code="bpy.context.scene.camera.data.lens = 99\nraise RuntimeError('boom')")
        assert not broken["ok"] and "boom" in broken["error"]
        assert svc.submit("probe", expression="bpy.context.scene.camera.data.lens")["result"] == 46
        # Raised in the agent's code: its traceback, from its own first frame.
        assert broken["failure_class"] == "author_code"
        assert broken["error"].startswith("Traceback") and "worker.py" not in broken["error"]
        missing = svc.submit("probe", expression="bpy.data.objects['no such object']")
        assert missing["failure_class"] == "author_code" and "KeyError" in missing["error"]
        # Refused by the harness: its message, nothing else.
        outside = svc.submit("probe", expression="1", frame=10**6)
        assert outside["failure_class"] == "rejection" and "Traceback" not in outside["error"]
        # What the agent's code prints comes back with the reply, and with its failure.
        said = svc.submit("blender", stage="storyboard", code="print('lens', 46)\nresult = 1")
        assert said["ok"] and said["printed"] == "lens 46\n"
        asked = svc.submit("probe", expression="print('probing')\nresult = 2")
        assert asked["result"] == 2 and asked["printed"] == "probing\n"
        loud = svc.submit("blender", stage="storyboard", code="print('before')\nraise KeyError('x')")
        assert "Printed before it raised:\nbefore" in loud["error"]
    finally:
        svc.jobs.close()


def test_build_keeps_its_candidate_tree_even_when_run_interactively(tmp_path):
    """`build`'s dominant workflow is the mutated-but-unpublished tree.

    Most builds go through the candidate protocol, which in-place mutation has
    no state to offer.
    """
    svc = _built(tmp_path)
    try:
        before = svc.version()
        (svc.design / "scene.py").write_text("this is broken source")
        failed = svc.submit("build")
        assert not failed["ok"]
        assert svc.version() == before, "a failed build must not touch the project"
    finally:
        svc.jobs.close()


def test_a_check_runs_alongside_a_publish_but_a_second_publish_does_not(tmp_path):
    """The edit lock covers publishing, not the whole surface.

    Holding it for a whole `storyboard` would leave the agent unable even to
    `probe`. What it has to prevent is two publishes racing.
    """
    svc = _built(tmp_path)
    try:
        held = svc.submit("blender", stage="storyboard", batch=True,
                          code="import time; time.sleep(2)")
        assert held["job_id"], held
        # A read gets through while the publish is in flight.
        assert svc.submit("probe", expression="1")["ok"]
        # A second publish does not.
        refused = svc.submit("plan", plan=plan())
        assert not refused["ok"] and "edit is pending" in refused["error"]

        # And once it lands, publishing is available again.
        while svc.jobs.status(held["job_id"])["status"] in ("queued", "running"):
            time.sleep(.05)
        assert svc.submit("plan", plan=plan())["ok"]
    finally:
        svc.jobs.close()


def test_a_rewritten_helper_takes_effect_on_the_next_call(tmp_path):
    """The failure residency is most likely to cause, and it is silent.

    A fresh process re-reads a helper from disk every call. A resident one
    serves `sys.modules` unless something purges it -- so an author who edits
    `animation_src/helper.py` and calls again gets the previous version, with
    no error and no symptom except the edit appearing not to take.

    The purge must compare the child's own paths: the session's *host* paths
    can never match them, which would leave it inert.
    """
    svc = _built(tmp_path)
    try:
        helper = svc.root / "animation_src" / "helper.py"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text("VALUE = 1\n")
        first = svc.submit("blender", stage="storyboard",
                           code="import helper\nresult['v'] = helper.VALUE")
        assert first["ok"] and first["result"]["v"] == 1, first

        helper.write_text("VALUE = 2\n")
        second = svc.submit("blender", stage="storyboard",
                            code="import helper\nresult['v'] = helper.VALUE")
        assert second["ok"], second
        assert second["result"]["v"] == 2, "the resident session served a stale helper"
    finally:
        svc.jobs.close()


def test_sys_path_does_not_grow_with_the_number_of_calls(tmp_path):
    """`run_job` inserts the workspace per call; the process now outlives it."""
    svc = _built(tmp_path)
    try:
        length = "len(__import__('sys').path)"
        svc.submit("probe", expression=length)
        before = svc.submit("probe", expression=length)["result"]
        for _ in range(6):
            svc.submit("probe", expression="1")
        assert svc.submit("probe", expression=length)["result"] == before
    finally:
        svc.jobs.close()


def test_an_interactive_publish_holds_the_edit_lock_too(tmp_path):
    """The invariant must not depend on which path an op took.

    Interactive ops run in the caller's thread, so a single-threaded agent can
    never observe this. The daemon and a dispatched contributor reach the same
    object, though, and "at most one publish in flight" should be true because
    it is enforced, not because of who happened to be calling.
    """
    svc = _built(tmp_path)
    try:
        started, seen = threading.Event(), {}

        def slow_publish():
            started.set()
            seen["result"] = svc.submit(
                "blender", stage="storyboard",
                code="import time; time.sleep(1.5)\nbpy.context.scene.camera.data.lens = 47")

        worker = threading.Thread(target=slow_publish)
        worker.start()
        try:
            started.wait(5)
            time.sleep(.4)  # let it get past the lock acquisition
            refused = svc.submit("plan", plan=plan())
            assert not refused["ok"] and "edit is pending" in refused["error"], refused
        finally:
            worker.join(60)
        assert seen["result"]["ok"], seen["result"]
        # Released, so the surface is usable again.
        assert not svc.edit_pending
        assert svc.submit("plan", plan=plan())["ok"]
    finally:
        svc.jobs.close()
