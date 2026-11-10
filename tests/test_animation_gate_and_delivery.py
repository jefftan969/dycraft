"""The gate and delivery loop, without a model call or Blender.

Concurrency and project integrity, per-state storyboard evidence, readiness,
spans, and delivery and commit. Each test drives the real session with the
worker faked at its boundary.
"""
import json
import threading
import time

import pytest
from PIL import Image

from dycraft.animation.blender.bundle import write_json
from dycraft.animation.planning import (failing_frames, failure_map, missing_states,
                                                plan_states, state_revision,
                                                uncovered_frames, validate_plan)
from dycraft.tests._animation_fixtures import call, flat_plan, plan, service, state_reviews

REQUIRED = ("AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied")


def passing():
    return [{"name": n, "verdict": "pass", "evidence": {}, "suggested_repairs": []} for n in REQUIRED]


def colliding(*spans):
    """Required reports with one `NoPenetration` pair failing over `spans`."""
    reports = passing()
    reports[2] = {"name": "NoPenetration", "verdict": "fail", "suggested_repairs": [],
                  "evidence": {"failure_count": len(spans),
                               "pairs": [{"a": "part/a/body", "b": "part/b/body", "kind": "x",
                                          "spans": [{"frames": list(s)} for s in spans]}]}}
    return reports


class Fake:
    """A worker that draws one flat panel per state and evaluates with `reports`."""

    def __init__(self, svc, *, n_frames=4, frames=None):
        self.svc, self.n_frames = svc, n_frames
        self.reports = passing()
        self.colors, self.requests = {}, []
        #: State id -> frame, as the blend would resolve it.
        self.frames = dict(frames or {})
        svc.worker = self

    def __call__(self, ws, output, request, cancel, **kw):
        self.requests.append(request)
        output.mkdir(parents=True, exist_ok=True)
        if request["op"] == "render" and request.get("states") is not None:
            states = request["states"]
            for index, state in enumerate(states):
                Image.new("RGB", (4, 4), self.colors.get(state["id"], (0, 0, 0))).save(
                    output / f"{index}.png")
            located = {s["id"]: self.frames.get(s["id"], s["frame"]) for s in states}
            return {"paths": [f"{i}.png" for i in range(len(states))],
                    "state_frames": located, "state_clips": {}, "views": [None] * len(states)}
        if request["op"] == "edit":
            return {"edited": True}
        if request["op"] == "evaluate" and request.get("states") is not None:
            # `evaluate(states=)`: where each state sits now.
            located = {s["id"]: self.frames.get(s["id"], s.get("frame")) for s in request["states"]}
            return {"fps": 24.0, "n_frames": self.n_frames, "frame_start": 0,
                    "state_frames": {k: v for k, v in located.items() if v is not None},
                    **({"unresolved_states": {k: "no binding" for k, v in located.items() if v is None}}
                       if None in located.values() else {}),
                    "frames": sorted({v for v in located.values() if v is not None})}
        frames = request.get("frames")
        if frames is None and request.get("start") is not None:
            frames = list(range(request["start"], request["end"] + 1))
        if frames is None:
            frames = list(range(self.n_frames))
        timeline = {"fps": 24.0, "n_frames": self.n_frames, "frame_start": 0,
                    "blender_version": "4.5", "scene_ref": "scene"}
        if request["op"] == "render":
            (output / "frames").mkdir(parents=True, exist_ok=True)
            (output / "views").mkdir(parents=True, exist_ok=True)
            for frame in frames:
                Image.new("RGB", (4, 4), (frame, 0, 0)).save(output / "frames" / f"{frame:06d}.png")
                write_json(output / "views" / f"{frame:06d}.json", {"frame": frame})
            return {**timeline, "paths": [f"frames/{f:06d}.png" for f in frames],
                    "views": [{"frame": f} for f in frames],
                    "render_settings": request.get("render_settings")}
        (output / "evaluation").mkdir(parents=True, exist_ok=True)
        (output / "evaluation/index.json").write_text("{}")
        if request.get("verification"):
            write_json(output / "verifier_reports.json", self.reports)
            write_json(output / "collision_pair_cache.json", {})
        return {**timeline, "frames": frames}


def project(svc, shape=plan, reports=()):
    assert call(svc, "plan", plan=shape())["ok"]
    (svc.work / "design_state").mkdir(parents=True, exist_ok=True)
    (svc.work / "animation.blend").write_bytes(b"blend")
    from dycraft.animation.blender.bundle import tree_hash
    write_json(svc.work / "design_state/build.json",
               {"source_hash": tree_hash(svc.design), "source_files": [], "design_id": "d",
                "verifier_reports": list(reports)})
    write_json(svc.work / "design_state/baseline.json", {"identities": {}, "joints": {}})
    write_json(svc.work / "design_state/scene.json", {"id": "scene"})


def storyboard(svc, tmp_path, name, **args):
    return svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"}, **args,
                                      "_version": svc.revision_token()},
                       svc.work, tmp_path / name, threading.Event(), 0)


def accept(svc, tmp_path, name, reviews):
    return svc.execute("accept_storyboard", {"reviews": reviews, "_version": svc.revision_token()},
                       svc.work, tmp_path / name, threading.Event(), 0)


def boards(svc):
    return json.loads((svc.work / "storyboards.json").read_text())


def wait(svc, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = svc.jobs.status(job_id)
        if record["status"] not in ("queued", "running"):
            return record
        time.sleep(.02)
    raise TimeoutError(job_id)


@pytest.fixture
def svc(tmp_path):
    service_ = service(tmp_path / "run")
    yield service_
    service_.jobs.close(timeout_s=5)


# -- Who excludes whom --------------------------------------------------------


def test_a_batch_look_does_not_switch_the_edit_guard_off(svc, tmp_path):
    """Only an edit sets `edit_pending`, so a batch look never clears the guard a
    running edit holds."""
    assert call(svc, "plan", plan=plan())["ok"]
    gate, real = threading.Event(), svc.execute

    def execute(operation, args, workspace, directory, cancel, step):
        if operation == "blender":
            gate.wait(10)
            return {"ok": True}
        if operation in ("compile", "storyboard", "accept_storyboard", "final_render"):
            return {"ok": True, "ran": operation}
        return real(operation, args, workspace, directory, cancel, step)

    svc.execute = execute
    edit = svc.submit("blender", batch=True, code="pass", stage="storyboard")
    look = svc.submit("compile", batch=True)
    assert wait(svc, look["job_id"])["result"]["ran"] == "compile"
    refused = svc.submit("plan", plan=plan())
    assert not refused["ok"] and "edit is pending" in refused["error"]
    # The refusal names the edit that holds it.
    assert f"blender, job {edit['job_id']}" in refused["error"]
    # Gate calls and renders are not edits: they run alongside it.
    for operation in ("storyboard", "final_render"):
        assert svc.submit(operation, batch=True)["ok"], operation
    gate.set()
    assert wait(svc, edit["job_id"])["status"] == "completed"
    assert call(svc, "plan", plan=plan())["ok"]


def test_an_edit_keeps_a_board_merge_that_landed_while_it_ran(svc, tmp_path):
    """The live boards are carried into the edit's candidate, minus the states
    the edit invalidated, so neither result is thrown away."""
    project(svc)
    fake = Fake(svc)
    storyboard(svc, tmp_path, "s1")
    gate, worker = threading.Event(), svc.worker

    def blocking(ws, output, request, cancel, **kw):
        if request["op"] == "edit":
            gate.wait(10)
        return worker(ws, output, request, cancel, **kw)

    svc.worker = blocking
    edit = svc.submit("blender", batch=True, code="pass", stage="storyboard",
                      invalidate_shots=["slide/1"])
    time.sleep(.2)
    assert accept(svc, tmp_path, "a", {"slide/0": "Rests on the floor."})["ok"]
    gate.set()
    assert wait(svc, edit["job_id"])["result"]["ok"], svc.jobs.status(edit["job_id"])
    live = boards(svc)
    assert live["slide/0"]["reviewed"] and live["slide/0"]["review"] == "Rests on the floor."
    assert "slide/1" not in live
    del fake


def test_an_edit_is_refused_only_when_content_moved_outside_the_harness(svc, tmp_path):
    project(svc)
    Fake(svc)
    gate, worker = threading.Event(), svc.worker

    def blocking(ws, output, request, cancel, **kw):
        if request["op"] == "edit":
            gate.wait(10)
        return worker(ws, output, request, cancel, **kw)

    svc.worker = blocking
    edit = svc.submit("blender", batch=True, code="pass", stage="storyboard")
    time.sleep(.2)
    (svc.work / "stray.txt").write_text("written straight into native/work")
    gate.set()
    record = wait(svc, edit["job_id"])
    assert not record["result"]["ok"]
    assert "changed outside the harness" in record["result"]["error"]


def test_board_merges_follow_submission_order_and_reviews_bind_to_their_panel(svc, tmp_path):
    project(svc)
    write_json(svc.work / "storyboards.json", {})
    cancel = threading.Event()
    newer = {"image_digest": "B", "state_revision": "r", "submitted": 5}
    svc.merge_boards(svc.work, {"_seq": 5}, cancel, boards={"s": newer})
    # An older submission finishing later does not overwrite the newer board.
    merged = svc.merge_boards(svc.work, {"_seq": 3}, cancel,
                              boards={"s": {"image_digest": "A", "submitted": 3}})
    assert merged["superseded"] == ["s"] and boards(svc)["s"]["image_digest"] == "B"
    # A review of a panel that has since been redrawn is not applied.
    merged = svc.merge_boards(svc.work, {"_seq": 6}, cancel,
                              reviews={"s": {"review": "about A", "image_digest": "A"}})
    assert merged["applied"] == [] and "redrawn" in merged["not_applied"]["s"]
    merged = svc.merge_boards(svc.work, {"_seq": 7}, cancel,
                              reviews={"s": {"review": "about B", "image_digest": "B"}})
    assert merged["applied"] == ["s"] and boards(svc)["s"]["review"] == "about B"
    # A redraw of the same picture keeps the review the live board carries.
    svc.merge_boards(svc.work, {"_seq": 8}, cancel,
                     boards={"s": {"image_digest": "B", "state_revision": "r", "submitted": 8}})
    assert boards(svc)["s"]["reviewed"] and boards(svc)["s"]["review"] == "about B"


def test_boards_are_not_part_of_the_content_identity(svc, tmp_path):
    """A board changes nothing in the film, so it voids no check."""
    project(svc)
    content, project_version = svc.identity(svc.work)
    write_json(svc.work / "storyboards.json", {"s": {"reviewed": True}})
    after_content, after_project = svc.identity(svc.work)
    assert after_content == content and after_project != project_version
    (svc.work / "animation.blend").write_bytes(b"edited")
    assert svc.content_version(svc.work) != content


def test_latest_json_follows_submission_order_not_finish_order(svc, tmp_path):
    project(svc)
    Fake(svc)
    for seq, label in ((2, "newer"), (1, "older")):
        svc.execute("compile", {"_seq": seq, "_version": "v"}, svc.work,
                    tmp_path / label, threading.Event(), seq)
    latest = json.loads((svc.root / "compile_reports/latest.json").read_text())
    assert latest["frames_checked"]["count"] == 4
    assert svc.latest_seq["compile_reports"] == 2
    assert {p.name for p in (svc.root / "compile_reports").iterdir()} == {"1.json", "2.json", "latest.json"}


def test_a_batch_probe_cannot_write_through_its_hardlinked_snapshot(svc, tmp_path, monkeypatch):
    """Batch probes link their snapshot and bind it read-only."""
    from dycraft.animation.blender import runtime
    seen = {}
    monkeypatch.setattr(runtime, "run_worker",
                        lambda ws, out, req, **kw: seen.update(req=req, **kw) or {"ok": True})
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    svc.worker(snapshot, tmp_path / "out", {"op": "probe", "code": "1"}, threading.Event())
    assert seen["workspace_readonly"] is True


# -- GPU renders --------------------------------------------------------------


def test_a_gpu_render_past_the_cap_queues_and_out_of_memory_is_named(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(type(svc), "gpu_render", staticmethod(lambda request: request["op"] == "render"))
    monkeypatch.setattr(svc, "GPU_RENDERS", 1)
    release, started = threading.Event(), threading.Event()

    def dispatch(ws, out, request, cancel, readonly):
        if request.get("hold"):
            started.set()
            release.wait(10)
            return {"ok": True}
        raise RuntimeError("Error: Out of memory in CUDA queue enqueue (integrator_shade_surface)")

    svc._dispatch = dispatch

    def job(request):
        def run(directory, cancel):
            svc._local.job = {"job_id": directory.name, "operation": "render"}
            return svc.worker(tmp_path, directory / "r", request, cancel)
        return run

    holder = svc.jobs.submit("render", job({"op": "render", "hold": True}), step=0, version="v")
    assert started.wait(5)
    queued = svc.jobs.submit("render", job({"op": "render"}), step=1, version="v")
    deadline = time.monotonic() + 5
    while svc.jobs.status(queued["job_id"]).get("waiting_for") is None:
        assert time.monotonic() < deadline
        time.sleep(.02)
    status = svc.jobs.status(queued["job_id"])
    assert status["status"] == "queued"
    assert status["waiting_for"] == {"gpu_renders": [holder["job_id"]]}
    time.sleep(.1)                    # a wait long enough for `queued_s` to show it
    release.set()
    failed = wait(svc, queued["job_id"])
    assert failed["status"] == "failed" and failed["result"]["error_class"] == "gpu_out_of_memory"
    assert "Out of memory in CUDA" in failed["result"]["error"]
    assert failed["queued_s"] > 0
    # A render that is not on the GPU never takes a slot.
    assert not type(svc).__dict__["gpu_render"].__func__({"op": "probe"})


def test_a_worker_that_crashed_out_of_gpu_memory_is_still_named(svc, tmp_path, monkeypatch):
    """The crash reply names only the exit code and the log; the renderer's
    line is in the log's end, which the error keeps as its `reason`."""
    from dycraft.animation.blender.runtime import WorkerError
    from dycraft.animation.processes import GpuOutOfMemory
    monkeypatch.setattr(type(svc), "gpu_render", staticmethod(lambda request: True))

    def dispatch(ws, out, request, cancel, readonly):
        raise WorkerError("Blender worker exited -6 during render; the next call starts a new "
                          "process. Log: worker.log", "harness",
                          reason="Fra:3 Sample 2/8\nError: Out of memory in CUDA queue enqueue\n")

    svc._dispatch = dispatch
    with pytest.raises(GpuOutOfMemory) as raised:
        svc.worker(tmp_path, tmp_path / "r", {"op": "render"}, threading.Event())
    assert str(raised.value).endswith("Renderer error: Error: Out of memory in CUDA queue enqueue")


def test_only_a_cycles_render_on_a_gpu_counts_against_the_cap(monkeypatch):
    from dycraft.animation.blender import runtime
    from dycraft.animation.session import AnimationSession
    monkeypatch.setattr(runtime, "nvidia_usable", lambda: True)
    gpu = AnimationSession.gpu_render
    assert gpu({"op": "render", "render_settings": {"engine": "CYCLES", "width": 640, "height": 360}})
    assert not gpu({"op": "render", "render_settings": {"engine": "WORKBENCH"}})
    assert not gpu({"op": "render"})                                   # the diagnostic default
    assert not gpu({"op": "render", "render_settings": {"compute_backend": "CPU"}})
    assert not gpu({"op": "evaluate", "render_settings": {"engine": "CYCLES"}})
    monkeypatch.setattr(runtime, "nvidia_usable", lambda: False)
    assert not gpu({"op": "render", "render_settings": {"engine": "CYCLES"}})


def test_the_arm_s_gpu_reaches_every_blender(monkeypatch):
    """`worker_env` is closed, so the GPU the operator or the scheduler gives a
    run must be passed to the forked workers and the resident explicitly.

    The host is pinned to a device cgroup that grants one of eight nodes, where
    the variable is numbered within the grant and passes verbatim. Without
    isolation an index list is renumbered to the nodes the sandbox binds
    (`5` becomes `0`); `tests/test_sandbox_gpu_devices.py` covers that. Unpinned,
    this test would read the real `/dev` and fail on any host where every GPU
    node opens."""
    from pathlib import Path
    from dycraft.animation.blender import runtime
    monkeypatch.setattr(runtime, "_dev_nodes", lambda: [Path(f"/dev/nvidia{i}") for i in range(8)])
    monkeypatch.setattr(runtime, "_openable", lambda dev: dev.name == "nvidia5")
    monkeypatch.setattr(runtime, "_NVIDIA_USABLE", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = runtime.worker_env()
    assert "CUDA_VISIBLE_DEVICES" not in env and "CUDA_DEVICE_ORDER" not in env
    for value in ("", "5", "GPU-1a2b,2"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        env = runtime.worker_env()
        assert env["CUDA_VISIBLE_DEVICES"] == value and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert runtime.worker_env()["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_a_request_queued_behind_the_resident_says_so_and_can_be_cancelled(tmp_path, monkeypatch):
    """Interactive calls wait on the shared resident Blender; the wait was invisible."""
    from dycraft.animation.blender.resident import ResidentBlender
    resident = ResidentBlender(tmp_path / "ws", tmp_path / "out")
    resident.process = type("Alive", (), {"poll": lambda self: None})()
    monkeypatch.setattr(resident, "_send", lambda value: None)
    monkeypatch.setattr(resident, "_await", lambda cancel: {"ok": True})
    monkeypatch.setattr(resident, "_result", lambda output, reply: "done")
    held, release = threading.Event(), threading.Event()

    def hold():
        with resident.lock:
            resident.holder = {"job_id": "first", "operation": "storyboard"}
            held.set()
            release.wait(10)
            resident.holder = None

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)
    waits, out = [], {}
    worker = threading.Thread(target=lambda: out.update(result=resident.request(
        {"op": "probe"}, tmp_path / "out/a", holder={"job_id": "second"}, on_wait=waits.append)))
    worker.start()
    time.sleep(.2)
    assert waits == [{"job_id": "first", "operation": "storyboard"}]
    release.set()
    worker.join(5)
    holder.join(5)
    assert waits[-1] is None and out["result"] == "done"
    # Canceling while queued costs nothing and does not touch the process.
    release.clear()
    held.clear()
    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)
    cancel = threading.Event()
    errors = []

    def queued():
        try:
            resident.request({"op": "probe"}, tmp_path / "out/b", cancel=cancel)
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=queued)
    worker.start()
    time.sleep(.1)
    cancel.set()
    worker.join(5)
    release.set()
    holder.join(5)
    assert errors == ["job cancelled"]


# -- Per-state evidence -------------------------------------------------------


def test_one_failing_frame_does_not_deny_every_state_its_evidence(svc, tmp_path):
    """A state that is clean where it stands keeps its evidence when another frame fails."""
    project(svc)
    fake = Fake(svc)
    fake.reports = colliding((1, 1))                  # slide/1 sits at frame 1
    reply = storyboard(svc, tmp_path, "s")
    assert not reply["ok"] and reply["failure_class"] == "check_finding"
    assert reply["banked"] == ["slide/0"] and reply["needs_review"] == ["slide/0"]
    assert reply["states"]["slide/0"]["checks_pass"] is True
    assert reply["states"]["slide/1"]["failing_here"] == ["NoPenetration"]
    assert "checks_ok" not in reply["states"]["slide/0"]
    live = boards(svc)
    assert live["slide/0"]["checks_pass"] and not live["slide/1"]["checks_pass"]
    assert live["slide/0"]["verdicts"]["NoPenetration"] == "pass"
    assert live["slide/1"]["verdicts"]["NoPenetration"] == "fail"
    assert accept(svc, tmp_path, "a", {"slide/0": "Rests flat."})["ok"]
    with pytest.raises(ValueError, match=r"slide/1 \(NoPenetration\)"):
        accept(svc, tmp_path, "b", {"slide/1": "Rests flat."})
    assert "slide/1" in missing_states(validate_plan(plan()), boards(svc), "d")


def test_a_failure_that_cannot_be_placed_still_fails_every_state(svc, tmp_path):
    project(svc)
    fake = Fake(svc)
    fake.reports = passing()
    fake.reports[0] = {"name": "AnimationWellFormed", "verdict": "fail", "suggested_repairs": [],
                       "evidence": {"failure_count": 1,
                                    "failures": [{"reason": "frozen scene identity changed"}]}}
    reply = storyboard(svc, tmp_path, "s")
    assert reply["banked"] == []
    assert all(e["failing_here"] == ["AnimationWellFormed"] for e in reply["states"].values())


def test_failing_frames_reads_each_verifier_s_own_evidence():
    assert failing_frames(colliding((2, 4), (9, 9))[2]) == {2, 3, 4, 9}
    joints = {"name": "JointLimits", "verdict": "fail", "evidence": {"findings": [
        {"reason": "r", "subjects": [{"subject": "j", "frames": [[5, 6]]}]}]}}
    assert failing_frames(joints) == {5, 6}
    events = {"name": "EventsSatisfied", "verdict": "fail", "evidence": {"events": [
        {"name": "e", "ok": False, "window_frames": [10, 12]}, {"name": "f", "ok": True}]}}
    assert failing_frames(events) == {10, 11, 12}
    invalid = {"name": "EventsSatisfied", "verdict": "fail", "evidence": {"events": [
        {"name": "e", "ok": False, "status": "invalid"}]}}
    assert failing_frames(invalid) is None
    # The uniform field wins once it is there.
    assert failing_frames({"name": "NoPenetration", "verdict": "fail",
                           "evidence": {"frames_failed": [[0, 1]]}}) == {0, 1}
    # Advisory reports never count, even mislabeled `fail` by a chunked merge.
    reports = passing() + [{"name": "ContactSlip", "verdict": "fail",
                            "evidence": {"diagnostic_only": True}}]
    assert failure_map(reports) == ({}, [])


def test_an_unaccepted_state_whose_pose_held_is_rechecked_not_redrawn(svc, tmp_path):
    """Otherwise about a quarter of redraws would be re-checks of an unchanged pose."""
    project(svc)
    fake = Fake(svc)
    poses = {0: 0.0, 1: 0.0}

    def evaluated_at(check, frame):
        matrix = [[1, 0, 0, poses[frame]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return {"objects": [{"name": "part/block/body", "hidden": False, "matrix": matrix}],
                "transforms": {"part/block/body": matrix}}

    svc.evaluated_at = evaluated_at
    fake.reports = colliding((1, 1))
    first = storyboard(svc, tmp_path, "s1")
    assert first["banked"] == ["slide/0"]
    drawn = len([r for r in fake.requests if r["op"] == "render"])
    # The collision was an allowance away: same pose, now passing. Nothing is
    # redrawn; slide/1 banks from the re-check, still waiting for review.
    fake.reports = passing()
    second = storyboard(svc, tmp_path, "s2")
    assert len([r for r in fake.requests if r["op"] == "render"]) == drawn
    assert second["rechecked"] == ["slide/0", "slide/1"] and second["retained"] == []
    assert second["banked"] == ["slide/0", "slide/1"]
    assert second["states"]["slide/1"]["checks_pass"] and boards(svc)["slide/1"]["checks_pass"]
    assert second["ok"]
    # `fresh=True` skips retention: every selected state is drawn again.
    third = storyboard(svc, tmp_path, "s3", _fresh=True)
    assert len([r for r in fake.requests if r["op"] == "render"]) == drawn + 1
    assert third["rechecked"] == []


def test_a_retime_past_a_board_s_frame_redraws_instead_of_crashing(svc, tmp_path):
    """Three recorded crashes came from retention asking for frames past the end."""
    project(svc, flat_plan)
    fake = Fake(svc, frames={"start": 0, "end": 3})
    storyboard(svc, tmp_path, "s1")
    assert accept(svc, tmp_path, "a", {"start": "x", "end": "y"})["ok"]
    svc.evaluated_at = lambda check, frame: {"transforms": {"a": [[1, 0, 0, 0]]}}
    live = boards(svc)
    for name in live:
        live[name]["pose_digest"] = "pose2:" + "0" * 8
    write_json(svc.work / "storyboards.json", live)
    # The film shrank to 3 frames: `end` now sits at frame 2.
    svc.timeline_hint = {"n_frames": 3}
    fake.n_frames, fake.frames["end"] = 3, 2
    reply = storyboard(svc, tmp_path, "s2")
    assert reply["moved"] == {"end": [3, 2]}
    assert reply["states"]["end"]["frame"] == 2
    # And a probe that raises keeps nothing rather than failing the call.
    svc.check, real = (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("frame index outside"))), svc.check
    try:
        assert svc.retainable(validate_plan(flat_plan()), {"design_id": "d"}, boards(svc),
                              {"start"}, {"engine": "WORKBENCH"}, svc.work, tmp_path / "r",
                              threading.Event(), "v") == ({}, set())
    finally:
        svc.check = real


def test_editing_criteria_or_intent_does_not_void_reviews():
    """A review is versioned by its state, so an edit to the plan's prose voids none."""
    value = validate_plan(plan())
    state = plan_states(value)[0]
    revised = {**value, "intent": "Something else entirely.",
               "criteria": value["criteria"] + [{"id": "x", "requirement": "r", "why": "w", "observe": "o"}]}
    assert state_revision(value, state) == state_revision(revised, state)
    assert state_revision(value, state) != state_revision(value, {**state, "purpose": "other"})


def test_rewording_intent_or_criteria_keeps_the_content_version(svc):
    """Nothing evaluates the prose, so rewording it does not discard the
    whole-film check and the delivery receipt; the states and shots do count."""
    project(svc)
    before = svc.content_version(svc.work)
    value = json.loads((svc.work / "plan.json").read_text())
    # Through `validate_plan`, as `plan()` stores it: its `id` hashes the prose.
    reworded = validate_plan({**value, "intent": "Something else entirely.",
                              "criteria": value["criteria"] + [
                                  {"id": "x", "requirement": "r", "why": "w", "observe": "o"}]})
    assert reworded["id"] != value["id"]
    write_json(svc.work / "plan.json", reworded)
    assert svc.content_version(svc.work) == before
    shots = [{**shot, "intent": "a different shot"} for shot in value["shots"]]
    write_json(svc.work / "plan.json", validate_plan({**reworded, "shots": shots}))
    assert svc.content_version(svc.work) != before


def test_the_coverage_note_states_the_figure_and_no_workflow_advice(svc):
    note = svc.gate_coverage({0, 5}, 50)["note"]
    assert note.startswith("2 of 50 frames are staged (4.0 %)")
    assert "compile() does" in note and "pose function" not in note


# -- Readiness ----------------------------------------------------------------


def test_compile_reports_what_stands_between_the_project_and_a_clean_finish(svc, tmp_path):
    project(svc)
    fake = Fake(svc)
    storyboard(svc, tmp_path, "s")
    fake.reports = colliding((2, 3))
    reply = svc.execute("compile", {"_seq": 1, "_version": "v"}, svc.work,
                        tmp_path / "c", threading.Event(), 1)
    assert not reply["ok"]
    unmet = reply["unmet"]
    assert {"kind": "check", "verifier": "NoPenetration", "frames": [[2, 3]]} in unmet
    assert any(u["kind"] == "storyboard" and u["states"] == ["slide/0", "slide/1"] for u in unmet)
    assert not any(u["kind"] == "unchecked" for u in unmet)
    assert "note" not in reply.get("storyboard", {})


def test_a_clean_interval_check_certifies_its_frames_and_nothing_else(svc, tmp_path):
    project(svc)
    Fake(svc, n_frames=10)
    svc.execute("check_motion", {"_seq": 1, "_version": "v", "start": 2, "end": 5}, svc.work,
                tmp_path / "c", threading.Event(), 1)
    svc.timeline_hint = {"n_frames": 10}
    items = svc.unmet(svc.work, boards={})
    assert {"kind": "unchecked", "frames": [[0, 1], [6, 9]]} in items
    assert not any(i["kind"] == "check" for i in items)
    # Over a range inside it, the interval is certified.
    assert not any(i["kind"] in ("check", "unchecked")
                   for i in svc.unmet(svc.work, boards={}, frames=range(2, 6)))


def test_a_frame_lies_in_an_accepted_span_only_between_accepted_states():
    placement = {"a": 0, "b": 5, "c": 9}
    outside, blamed = uncovered_frames(range(10), placement, {"a", "b"})
    assert outside == {6, 7, 8, 9} and blamed == ["c"]
    outside, blamed = uncovered_frames(range(10), {**placement, "c": None}, {"a", "b", "c"})
    assert outside == set(range(10)) and blamed == ["c"]
    assert uncovered_frames(range(10), placement, {"a", "b", "c"}) == (set(), [])


# -- Spans and the clip gate --------------------------------------------------


def three_states():
    return {**flat_plan(), "states": [{"id": "a", "purpose": "start"},
                                      {"id": "b", "purpose": "middle"},
                                      {"id": "c", "purpose": "end"}]}


def edit(svc, tmp_path, name, **args):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return svc.execute("blender", {"code": "pass", "_version": svc.revision_token(), **args},
                       svc.work, tmp_path / name, threading.Event(), 0)


def test_a_span_between_accepted_states_can_be_animated_before_the_whole_storyboard(svc, tmp_path):
    project(svc, three_states)
    Fake(svc, n_frames=10, frames={"a": 0, "b": 5, "c": 9})
    storyboard(svc, tmp_path, "s")
    assert accept(svc, tmp_path, "acc", {"a": "start pose", "b": "middle pose"})["ok"]
    with pytest.raises(ValueError, match="need a passing, accepted storyboard before an animation edit"):
        edit(svc, tmp_path, "whole", stage="animation")
    reply = edit(svc, tmp_path, "span", stage="animation", span=["a", "b"])
    assert reply["ok"]
    assert reply["span"] == {"start_state": "a", "end_state": "b", "frames": [0, 5],
                             "states": ["a", "b"], "unaccepted_elsewhere": ["c"]}
    # Lead-in: `None` is the film's start.
    assert edit(svc, tmp_path, "lead", stage="animation", span=[None, "b"])["span"]["frames"] == [0, 5]
    with pytest.raises(ValueError, match="Accept these states before animating this span.*: c"):
        edit(svc, tmp_path, "bc", stage="animation", span=["b", "c"])
    with pytest.raises(ValueError, match="reversed"):
        edit(svc, tmp_path, "rev", stage="animation", span=["b", "a"])
    with pytest.raises(ValueError, match="with stage='animation'"):
        edit(svc, tmp_path, "label", stage="storyboard", span=["a", "b"])


def test_a_span_is_placed_where_its_states_resolve_now(svc, tmp_path):
    """`evaluate(states=)` wiring: placed by the blend as it is, before
    the edit runs, not where the boards were drawn; and a state that no
    longer resolves is refused by name rather than placed by its board."""
    project(svc, three_states)
    fake = Fake(svc, n_frames=10, frames={"a": 0, "b": 5, "c": 9})
    storyboard(svc, tmp_path, "s")
    assert accept(svc, tmp_path, "acc", {"a": "x", "b": "y", "c": "z"})["ok"]
    fake.frames["b"] = 7                       # retimed since its board was drawn
    reply = edit(svc, tmp_path, "e", stage="animation", span=["a", "b"])
    assert reply["span"]["frames"] == [0, 7]
    assert svc.timeline_hint["state_frames"] == {"a": 0, "b": 7, "c": 9}
    del fake.frames["c"]
    with pytest.raises(ValueError, match="cannot be placed.*c: no binding"):
        edit(svc, tmp_path, "e2", stage="animation", span=["a", "b"])


def test_a_span_needs_every_state_drawn_once_and_shot_is_shorthand(svc, tmp_path):
    project(svc)
    fake = Fake(svc)
    storyboard(svc, tmp_path, "s", states=["slide/0"])
    with pytest.raises(ValueError, match="drawn at least once.*slide/1"):
        edit(svc, tmp_path, "e1", stage="animation", shot="slide")
    fake.reports = colliding((1, 1))
    storyboard(svc, tmp_path, "s2", states=["slide/1"])       # drawn, failing: counts as drawn
    assert accept(svc, tmp_path, "a", {"slide/0": "fine"})["ok"]
    with pytest.raises(ValueError, match="slide/1"):
        edit(svc, tmp_path, "e2", stage="animation", shot="slide")
    reply = edit(svc, tmp_path, "e3", stage="animation", span=["slide/0", "slide/0"])
    assert reply["span"]["states"] == ["slide/0"]


def test_a_span_edit_reports_keys_outside_it_and_moved_endpoints(svc, tmp_path, monkeypatch):
    from dycraft.animation.blender import evaluation
    from dycraft.animation.planning import pose_digest
    project(svc, three_states)
    fake = Fake(svc, n_frames=10, frames={"a": 0, "b": 5, "c": 9})
    storyboard(svc, tmp_path, "s")
    assert accept(svc, tmp_path, "acc", {"a": "x", "b": "y", "c": "z"})["ok"]
    still = {"transforms": {"part/a/body": [[1, 0, 0, 0]]}}
    moved = {"transforms": {"part/a/body": [[1, 0, 0, 0.5]]}}
    live = boards(svc)
    live["a"]["pose_digest"] = live["b"]["pose_digest"] = pose_digest(still)
    write_json(svc.work / "storyboards.json", live)
    monkeypatch.setattr(evaluation, "read_record",
                        lambda directory, frame: still if frame == 0 else moved)
    real = fake.__call__

    def worker(ws, output, request, cancel, **kw):
        if request["op"] == "edit":
            return {"animation_changed": {"changed_frames": [[3, 8]]}}
        return real(ws, output, request, cancel, **kw)

    svc.worker = worker
    reply = edit(svc, tmp_path, "e", stage="animation", span=["a", "b"])
    assert reply["span"]["outside_span_keys"] == [[6, 8]]
    # Reported, never voided: `b` keeps its acceptance.
    assert reply["span"]["endpoints_moved"] == ["b"]
    assert boards(svc)["b"]["reviewed"]


def test_a_clip_owning_contributor_with_no_boards_is_still_gated(svc, tmp_path):
    """Never-drawn states have no board, so the clip filter dropped them all."""
    project(svc)
    Fake(svc)
    svc.contribution_clip = "transfer"
    with pytest.raises(ValueError, match="slide/0, slide/1"):
        edit(svc, tmp_path, "e", stage="animation")


def test_a_contribution_returns_however_its_child_ended(svc, tmp_path, monkeypatch):
    """A motion dispatch's `ok` does not require a committed `ir.json`, since most
    are probes that never call `finish()`."""
    from dycraft.animation.blender.bundle import write_json as write
    from dycraft.animation import dispatch as module
    handlers = {}
    registration = type("Registration", (), {
        "trace": lambda *a, **k: None,
        "op": lambda self, name: lambda fn: handlers.setdefault(name, fn),
        "register_handler": lambda self, name, fn: handlers.setdefault(name, fn),
    })()
    module.register(registration, svc)
    finishes = {"probe": None, "refused": {"ok": False, "error": "unmet: storyboard"},
                "finisher": {"ok": True}}
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        name = argv[argv.index("--run-id") + 1]
        if finishes[name] is not None:
            write(svc.root / "contributions" / name / "finish_result.json", finishes[name])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(module, "run_cancellable", run)
    monkeypatch.setenv("DYCRAFT_SUB_BACKEND", "codex")
    write(svc.root / "meta.json", {"runtime": "codex1_cli", "model": "gpt-6"})
    (svc.design / "scene.py").write_text("source")
    records = {}
    for name in finishes:
        # `max_steps` as a client staged before it was dropped still sends it.
        result = handlers["dispatch_scene"](name=name, brief="try it", max_steps=64)
        svc.contributions.threads[-1].join(timeout=5)
        record = records[name] = svc.contributions.status(result["job_id"])["result"]
        assert record["ok"] and not record["committed"] and record["returncode"] == 0
        # `finished` is that `finish()` succeeded; a refusal writes
        # `finish_result.json` too, and says so.
        assert record["finished"] is (name == "finisher")
        assert record["ended_by"] == {"probe": "never_reached_harness", "refused": "finish_refused",
                                      "finisher": "finished"}[name]
    assert records["refused"]["finish_refused"] == "unmet: storyboard"
    assert "--max-steps" not in seen[0]
    registration.drain_on_shutdown(timeout_s=1)



def test_a_motion_contributor_cannot_own_a_clip_the_project_lacks(svc, monkeypatch):
    """A clip-owning contributor gates itself on the states inside its clip, so
    an unknown name gated it on no drawn state. Dispatch refuses it, naming the
    project's clips, from an evaluation that only places states."""
    from dycraft.animation import dispatch as module
    handlers = {}
    registration = type("Registration", (), {
        "trace": lambda *a, **k: None,
        "op": lambda self, name: lambda fn: handlers.setdefault(name, fn),
        "register_handler": lambda self, name, fn: handlers.setdefault(name, fn),
    })()
    module.register(registration, svc)
    asked = []

    def worker(workspace, output, request, cancel, **kwargs):
        asked.append(request)
        return {"clips": {"transfer": {"frames": [0, 40]}, "pour": {"frames": [41, 90]}}}

    svc.worker = worker
    with pytest.raises(ValueError, match="'lift' is not an NLA strip.*pour, transfer"):
        handlers["dispatch_motion"](name="m1", brief="lift the jug", clip="lift")
    assert asked and asked[0]["place_only"] is True and asked[0]["states"] == []
    assert not (svc.root / "contributions" / "m1").exists()

    # A known clip passes the check and reaches the launch.
    monkeypatch.setattr(module, "run_cancellable",
                        lambda argv, **k: type("Result", (), {"returncode": 0})())
    monkeypatch.setattr(svc, "build_of", lambda workspace: {}, raising=False)
    result = handlers["dispatch_motion"](name="m2", brief="pour it", clip="pour")
    svc.contributions.threads[-1].join(timeout=5)
    assert (svc.root / "contributions" / "m2" / "contribution.json").is_file()
    assert result["role"] == "motion"
    registration.drain_on_shutdown(timeout_s=1)


# -- Delivery and commit ------------------------------------------------------


CRITERIA = {"support": {"verdict": "pass", "evidence": "The block stays on the floor."}}


@pytest.fixture
def no_encode(monkeypatch):
    """The encode is ffmpeg's job and is tested with real frames elsewhere."""
    from dycraft.animation import delivery

    def encode(result, rendered, directory, settings, cancel):
        clip = directory / "clip.mp4"
        clip.write_bytes(b"film:" + ",".join(sorted(p.name for p in (rendered / "frames").glob("*.png"))).encode()
                         if (rendered / "frames").is_dir() else b"film")
        return clip, {"width": settings["width"], "height": settings["height"]}

    monkeypatch.setattr(delivery, "encode_and_inspect", encode)


def final_render(svc, tmp_path, name, **args):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return svc.execute("final_render", {"_content": svc.content_version(svc.work),
                                        "_version": "v", **args},
                       svc.work, tmp_path / name, threading.Event(), 0)


def delivery_render(svc, tmp_path, name, **args):
    directory = svc.jobs.root / name
    directory.mkdir(parents=True, exist_ok=True)
    return svc.execute("render", {"delivery": True, "_content": svc.content_version(svc.work),
                                  "_version": "v", **args},
                       svc.work, directory, threading.Event(), 0)


def finish(svc, tmp_path, name, **args):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / name / "workspace"
    import shutil
    shutil.copytree(svc.work, workspace)
    return svc.execute("finish", {"_content": svc.content_version(svc.work), "_version": "v", **args},
                       workspace, tmp_path / name, threading.Event(), 0)


def compile_(svc, tmp_path, name):
    return svc.execute("compile", {"_seq": 1, "_version": "v"}, svc.work,
                       tmp_path / name, threading.Event(), 1)


def accepted(svc, tmp_path):
    """A project whose every state is drawn, clean, and accepted."""
    project(svc)
    fake = Fake(svc)
    storyboard(svc, tmp_path, "s")
    assert accept(svc, tmp_path, "a", state_reviews("Rests on the floor."))["ok"]
    return fake


def test_a_bare_final_render_draws_at_the_plan_s_floor(svc, tmp_path):
    """A bare call came from an author who never thought about resolution."""
    from dycraft.animation.delivery import delivery_defaults
    project(svc, lambda: {**plan(), "delivery": {"min_pixels": 960 * 540}})
    assert delivery_defaults(svc, svc.work, None) == {"width": 960, "height": 540}
    assert delivery_defaults(svc, svc.work, {"engine": "WORKBENCH"})["width"] == 960
    assert delivery_defaults(svc, svc.work, {"width": 1280, "height": 720}) == {
        "width": 1280, "height": 720}
    # One dimension alone kept the other's default, at an aspect nobody chose.
    with pytest.raises(ValueError, match="width and height together"):
        delivery_defaults(svc, svc.work, {"width": 1280})
    # The project's own frame size sets the aspect, not the worker's 16:9
    # diagnostic recipe, which drew a vertical film's bare delivery at 16:9.
    svc.timeline_hint = {"render": {"width": 1280, "height": 720}, "resolution": [1000, 1000]}
    square = delivery_defaults(svc, svc.work, None)
    assert square["width"] == square["height"] and square["width"] * square["height"] >= 960 * 540
    assert square["width"] % 2 == 0
    upright = delivery_defaults(svc, svc.work, None, {"resolution": [1080, 1920]})
    assert upright["height"] > upright["width"] and upright["width"] * upright["height"] >= 960 * 540


def test_final_render_always_draws_and_lists_what_is_unmet(svc, tmp_path, no_encode):
    """A run can end without a finished whole-film check, so `final_render` draws
    anyway and lists the unchecked frames as unmet."""
    project(svc)
    Fake(svc)
    film = final_render(svc, tmp_path, "f")
    assert film["ok"] and film["admitted"]["drawn_here"] == 4
    assert {"kind": "unchecked", "frames": [[0, 3]]} in film["unmet"]
    assert any(u["kind"] == "storyboard" for u in film["unmet"])
    receipt = json.loads((svc.native / "delivery.json").read_text())
    assert receipt["version"] == svc.content_version(svc.work)
    assert receipt["certified_by"]["kind"] == "evaluation"
    # It writes nothing into the project.
    assert film["version"] == svc.content_version(svc.work)
    assert not (svc.work / "film").exists()


def test_a_failing_whole_film_check_certifies_its_clean_ranges(svc, tmp_path, no_encode):
    """One failure need not repeat the whole check; the clean frames stand."""
    fake = accepted(svc, tmp_path)
    fake.reports = colliding((2, 2))
    assert not compile_(svc, tmp_path, "c")["ok"]
    clean = delivery_render(svc, tmp_path, "r0", start=0, end=1)
    assert clean["ok"] and clean["certified_by"]["kind"] == "whole"
    assert clean["unmet"] == []
    record = json.loads((svc.jobs.root / "r0" / "delivery_frames.json").read_text())
    assert record["scope"] is None and record["checks_ok"] is True
    dirty = delivery_render(svc, tmp_path, "r1", start=2, end=3)
    assert {"kind": "check", "verifier": "NoPenetration", "frames": [[2, 2]]} in dirty["unmet"]


def test_compile_then_accept_then_deliver_keeps_the_check(svc, tmp_path, no_encode):
    """Accepting a storyboard after a compile must not discard the check, or the
    delivery render is refused."""
    project(svc)
    fake = Fake(svc)
    storyboard(svc, tmp_path, "s")
    assert compile_(svc, tmp_path, "c")["ok"]
    worked = len([r for r in fake.requests if r.get("verification")])
    assert accept(svc, tmp_path, "a", state_reviews("Rests on the floor."))["ok"]
    film = final_render(svc, tmp_path, "f")
    assert film["ok"] and film["unmet"] == [], film["unmet"]
    assert len([r for r in fake.requests if r.get("verification")]) == worked, "the check was redone"
    receipt = json.loads((svc.native / "delivery.json").read_text())
    assert receipt["certified_by"]["kind"] == "whole" and receipt["checks"]["validation_reused"]


def test_final_render_composes_what_was_drawn_and_draws_the_rest(svc, tmp_path, no_encode):
    accepted(svc, tmp_path)
    assert compile_(svc, tmp_path, "c")["ok"]
    piece = delivery_render(svc, tmp_path, "piece", start=0, end=1,
                            render_settings={"engine": "WORKBENCH", "width": 64, "height": 64})
    film = final_render(svc, tmp_path, "f", frames_from=["piece"])
    assert film["ok"], film
    assert film["composed_from"] == {"piece": "0-1", "drawn_here": "2-3"}
    assert film["admitted"] == {"same_version": 2, "by_identity": 0, "draft": 0, "drawn_here": 2}
    # The final render's own frames are delivery frames for the next one.
    record = json.loads((tmp_path / "f" / "delivery_frames.json").read_text())
    assert record["delivery"] and record["version"] == piece["version"]


def test_frames_are_admitted_by_identity_at_any_index_once_the_digest_exists(svc, tmp_path):
    """A held or shifted frame is reused, not redrawn."""
    from dycraft.animation.delivery import DELIVERY_RECORD, admit
    job = svc.jobs.root / "old"
    (job / "render/frames").mkdir(parents=True)
    for frame in (0, 1):
        Image.new("RGB", (4, 4)).save(job / "render/frames" / f"{frame:06d}.png")
    settings = {"engine": "BLENDER_WORKBENCH", "width": 64, "height": 64, "strategy": "frames"}
    write_json(job / DELIVERY_RECORD, {
        "delivery": True, "verify_replay": True, "version": "older", "render_settings": settings,
        "render_state_version": "rs1", "frames": {"0": ["sigA", "rsA"], "1": ["sigB", "rsB"]}})
    targets = {"version": "rs1", "frames": {"0": ["sigA", "rsA"], "1": ["sigA", "rsA"],
                                            "2": ["sigB", "rsB"], "3": ["sigC", None]}}
    admitted = admit(svc, ["old"], None, "current", targets=targets)
    # Frame 1 is a hold of frame 0 and frame 2 is frame 1 shifted by a retime.
    assert sorted(admitted["frames"]) == [0, 1, 2] and admitted["by_identity"] == 3
    assert admitted["frames"][1].name == "000000.png"
    # A frame whose render-state digest abstained (None), or a digest of
    # another version, admits nothing.
    assert 3 not in admitted["frames"]
    assert admit(svc, ["old"], None, "current",
                 targets={**targets, "version": "rs2"})["frames"] == {}
    assert "old" not in admitted["not_admitted"]
    # A job none of whose frames matches is named, as an unused job always was.
    unused = admit(svc, ["old"], None, "current",
                   targets={"version": "rs1", "frames": {"0": ["sigZ", "rsZ"]}})
    assert unused["frames"] == {} and "older" in unused["not_admitted"]["old"]
    # Without render-state targets the record is simply of another version:
    # named, not admitted.
    assert "older" in admit(svc, ["old"], None, "current")["not_admitted"]["old"]


def test_a_draft_of_the_final_blend_is_deliverable_with_exceptions(svc, tmp_path, no_encode):
    """With one GPU per run, a run may reach its deadline with only a draft of the
    final blend, which is deliverable with its exceptions listed."""
    accepted(svc, tmp_path)
    assert compile_(svc, tmp_path, "c")["ok"]
    directory = svc.jobs.root / "draft"
    directory.mkdir(parents=True)
    draft = svc.execute("render", {"_version": "v", "render_settings":
                                   {"engine": "WORKBENCH", "width": 64, "height": 64}},
                        svc.work, directory, threading.Event(), 0)
    assert draft["ok"] and not draft.get("delivery_grade")
    film = final_render(svc, tmp_path, "f", frames_from=["draft"])
    assert film["ok"] and film["admitted"]["draft"] == 4
    assert {"kind": "draft_quality", "frames": [[0, 3]]} in film["unmet"]
    refused = finish(svc, tmp_path, "done", criteria=CRITERIA)
    assert not refused["ok"] and "draft_quality" in refused["error"]
    done = finish(svc, tmp_path, "done2", criteria=CRITERIA, exceptions=True)
    assert done["ok"] and done["outcome"] == "delivered_with_exceptions"
    # A draft through a diagnostic camera is a look, never film frames.
    look = svc.jobs.root / "look"
    look.mkdir()
    svc.ctx.committed = False
    svc.execute("render", {"_version": "v", "eye": [1, 1, 1], "look_at": [0, 0, 0],
                           "render_settings": {"engine": "WORKBENCH", "width": 64, "height": 64}},
                svc.work, look, threading.Event(), 0)
    with pytest.raises(ValueError, match="diagnostic camera"):
        final_render(svc, tmp_path, "f2", frames_from=["look"])


def test_finish_commits_cleanly_only_when_nothing_is_unmet(svc, tmp_path, no_encode, monkeypatch):
    fake = accepted(svc, tmp_path)
    outcomes = []
    svc.session.finalize_meta = lambda outcome, **k: outcomes.append(outcome)
    (svc.root / "README.md").write_text("How to open the blend.")
    (svc.root / "inspection").mkdir()
    (svc.root / "inspection/contacts.md").write_text("Frame 2 contact checked.")
    fake.reports = colliding((2, 2))
    assert not compile_(svc, tmp_path, "c")["ok"]
    assert final_render(svc, tmp_path, "f")["ok"]
    # The criteria are the checklist.
    with pytest.raises(ValueError, match=r"support: block remains on the floor \(observe"):
        finish(svc, tmp_path, "n0")
    with pytest.raises(ValueError, match="non-empty evidence: support"):
        finish(svc, tmp_path, "n1", criteria={"support": {"verdict": "pass", "evidence": " "}})
    refused = finish(svc, tmp_path, "n2", criteria=CRITERIA)
    assert not refused["ok"] and not svc.ctx.committed
    assert {"kind": "check", "verifier": "NoPenetration", "frames": [[2, 2]]} in refused["unmet"]
    assert "exceptions=True" in refused["error"]
    failed = {"support": {"verdict": "fail", "evidence": "It lifts at frame 2."}}
    done = finish(svc, tmp_path, "n3", criteria=failed, exceptions=True)
    assert done["ok"] and done["outcome"] == "delivered_with_exceptions" and outcomes == [done["outcome"]]
    assert {"kind": "criterion", "id": "support", "evidence": "It lifts at frame 2."} in done["unmet"]
    manifest = json.loads((svc.root / "ir.json").read_text())
    assert manifest["outcome"] == "delivered_with_exceptions" and manifest["criteria"] == failed
    bundle = svc.root / "bundle"
    assert done["notes_shipped"] == ["notes/README.md", "notes/inspection/contacts.md"]
    assert (bundle / "notes/inspection/contacts.md").is_file()
    notes = (bundle / "DELIVERY.md").read_text()
    assert "delivered with exceptions" in notes and "It lifts at frame 2." in notes
    # The delivery settings went into the bundle's blend, set to still output.
    edit = [r for r in fake.requests if r["op"] == "edit"][-1]
    assert "file_format = 'PNG'" in edit["code"] and "'//film/frames/'" in edit["code"]


def test_a_frozen_scene_identity_change_is_never_waived(svc, tmp_path, no_encode):
    fake = accepted(svc, tmp_path)
    fake.reports = passing()
    fake.reports[0] = {"name": "AnimationWellFormed", "verdict": "fail", "suggested_repairs": [],
                       "evidence": {"failure_count": 1,
                                    "failures": [{"reason": "frozen scene identity changed"}]}}
    compile_(svc, tmp_path, "c")
    assert final_render(svc, tmp_path, "f")["ok"]
    with pytest.raises(ValueError, match="never waived"):
        finish(svc, tmp_path, "n", criteria=CRITERIA, exceptions=True)


def test_finish_refuses_a_film_drawn_from_other_content(svc, tmp_path, no_encode):
    accepted(svc, tmp_path)
    assert compile_(svc, tmp_path, "c")["ok"]
    assert final_render(svc, tmp_path, "f")["ok"]
    (svc.work / "animation.blend").write_bytes(b"edited")
    with pytest.raises(ValueError, match="drawn from this project's content"):
        finish(svc, tmp_path, "n", criteria=CRITERIA, exceptions=True)


def test_a_chunked_check_keeps_each_frame_s_record_from_the_chunk_that_owns_it(svc, tmp_path, monkeypatch):
    """Chunk k+1's context frames, reached by a jump, must not overwrite chunk
    k's owned records, reached by stepping."""
    from dycraft.animation import checking as module
    monkeypatch.setattr(module, "merge_rows", lambda bundles: {})
    monkeypatch.setattr(module, "aggregate", lambda rows: passing())

    def worker(ws, output, request, cancel, **kw):
        low, high = request["owned"]
        (output / "evaluation").mkdir(parents=True)
        (output / "evaluation/index.json").write_text("{}")
        for frame in request["frames"]:
            (output / "evaluation" / f"{frame:06d}.bin").write_text(f"chunk {low}-{high}")
        write_json(output / "verifier_reports.json", passing())
        return {"fps": 24.0, "n_frames": 8, "frames": request["frames"]}

    svc.worker = worker
    plan = [(0, 3, [0, 1, 2, 3, 4, 5]), (4, 7, [2, 3, 4, 5, 6, 7])]
    svc.check_in_parallel(tmp_path, tmp_path / "out", threading.Event(), plan, {"op": "evaluate"})
    merged = tmp_path / "out/evaluated/evaluation"
    assert {f: (merged / f"{f:06d}.bin").read_text() for f in range(8)} == {
        **{f: "chunk 0-3" for f in range(4)}, **{f: "chunk 4-7" for f in range(4, 8)}}


def test_an_interactive_op_s_trace_names_its_job_and_result_on_every_path(tmp_path):
    """A failed interactive op's job directory was unreachable from the trace."""
    from types import SimpleNamespace
    svc = service(tmp_path / "run")
    traces = []
    svc.session = SimpleNamespace(ctx=svc.session.ctx, current_op_step=lambda: 0,
                                  trace=lambda name, **kw: traces.append((name, kw)),
                                  finalize_meta=lambda *a: None)
    try:
        assert svc.submit("plan", plan=plan())["ok"]
        failed = svc.submit("plan", plan={})
        records = [kw for name, kw in traces if name == "job_result"]
        assert [r["status"] for r in records] == ["completed", "failed"]
        assert records[1]["job_id"] == failed["job_id"]
        assert records[1]["result"]["job_id"] == failed["job_id"] and not records[1]["result"]["ok"]
    finally:
        svc.jobs.close(timeout_s=5)


def test_an_interrupted_run_names_the_best_film_it_left(svc, tmp_path, no_encode):
    """An interrupted run names the best film it left, a complete draft included."""
    from dycraft.animation.delivery import best_film
    accepted(svc, tmp_path)
    assert best_film(svc.root) is None
    for name, args in (("range", {"delivery": True, "start": 0, "end": 1}),
                       ("whole", {}), ("look", {"eye": [1, 1, 1], "look_at": [0, 0, 0]})):
        directory = svc.jobs.root / name
        directory.mkdir(parents=True)
        svc.execute("render", {"_version": "v", "_content": svc.content_version(svc.work), **args,
                               "render_settings": {"engine": "WORKBENCH", "width": 64, "height": 64}},
                    svc.work, directory, threading.Event(), 0)
    # The whole-film draft covers more than the delivery range; the look is not film.
    assert best_film(svc.root) == {"kind": "draft", "job": "whole", "frames": 4, "spans": [[0, 3]],
                                   "version": json.loads((svc.jobs.root / "whole/delivery_frames.json")
                                                         .read_text())["blend_hash"]}
    assert compile_(svc, tmp_path, "c")["ok"]
    assert final_render(svc, tmp_path, "f")["ok"]
    film = best_film(svc.root)
    assert film["kind"] == "delivery" and film["frames"] == 4 and film["unmet"] == []


def test_a_verifier_with_nothing_to_check_passes_the_gate():
    """`not_applicable` (no joints, no events...) is a pass, not a failure."""
    from dycraft.animation.planning import required_checks_pass
    reports = passing()
    reports[1] = {"name": "JointLimits", "verdict": "not_applicable", "evidence": {}}
    assert required_checks_pass(reports) and failure_map(reports) == ({}, [])


# -- Concurrency seams --------------------------------------------------------


def test_a_composed_film_carries_the_replay_mismatches_of_the_frames_it_admits(
        svc, tmp_path, no_encode):
    """A delivery range recorded its mismatch, and `final_render(frames_from=)`
    composed the frame without it: the film then committed clean."""
    fake = accepted(svc, tmp_path)
    assert compile_(svc, tmp_path, "c")["ok"]

    def mismatching(ws, output, request, cancel, **kw):
        result = fake(ws, output, request, cancel, **kw)
        if request["op"] == "render" and request.get("verify_replay") and request.get("start") == 0:
            result = {**result, "replay_mismatch": [{"frame": 1, "changed": {"transforms": ["x"]}}]}
        return result

    svc.worker = mismatching
    delivery_render(svc, tmp_path, "piece", start=0, end=1,
                    render_settings={"engine": "WORKBENCH", "width": 64, "height": 64})
    film = final_render(svc, tmp_path, "f", frames_from=["piece"])
    assert [u["frames"] for u in film["unmet"] if u["kind"] == "replay_mismatch"] == [[[1, 1]]]
    assert not finish(svc, tmp_path, "done", criteria=CRITERIA)["ok"]


def test_an_older_check_finishing_last_keeps_the_newer_whole_film_check(svc, tmp_path, no_encode):
    """Checks overlap; the cache followed finish order, so a stale check
    finishing last cost a re-evaluation and a `finish` refusal."""
    import shutil
    accepted(svc, tmp_path)
    old = tmp_path / "old_snapshot"
    shutil.copytree(svc.work, old)
    (svc.work / "animation.blend").write_bytes(b"blend-edited")
    new = svc.execute("compile", {"_seq": 2, "_version": "v"}, svc.work, tmp_path / "c2",
                      threading.Event(), 1)
    assert new["ok"]
    svc.execute("compile", {"_seq": 1, "_version": "v"}, old, tmp_path / "c1", threading.Event(), 1)
    final_render(svc, tmp_path, "f")
    assert json.loads((svc.native / "delivery.json").read_text())["certified_by"]["kind"] == "whole"


def test_overlapping_storyboards_leave_the_panel_the_winning_board_names(svc, tmp_path):
    """Every storyboard wrote one shared file per state, so the board could name
    a picture the agent never saw: the later submission's board with the file
    of whichever call finished last."""
    import shutil
    project(svc)
    fake = Fake(svc)
    snapshot = tmp_path / "snap_seq1"
    shutil.copytree(svc.work, snapshot)
    target = plan_states(json.loads((svc.work / "plan.json").read_text()))[0]["id"]
    fake.colors[target] = (255, 0, 0)
    svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"}, "_seq": 2,
                               "_version": "r"}, svc.work, tmp_path / "s2", threading.Event(), 0)
    fake.colors[target] = (0, 0, 255)
    reply = svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"}, "_seq": 1,
                                       "_version": "r"}, snapshot, tmp_path / "s1",
                        threading.Event(), 0)
    assert target in reply["superseded"]
    board = boards(svc)[target]
    assert board["submitted"] == 2
    assert svc.image_digest(svc.native / board["path"]) == board["image_digest"]


def test_an_older_storyboard_does_not_bring_back_a_board_an_edit_dropped(svc, tmp_path):
    import shutil
    accepted(svc, tmp_path)
    target = sorted(boards(svc))[0]
    snapshot = tmp_path / "snap_seq5"
    shutil.copytree(svc.work, snapshot)
    (tmp_path / "e").mkdir()
    assert svc.execute("blender", {"code": "pass", "stage": "storyboard",
                                   "invalidate_shots": [target], "_seq": 6, "_version": "r"},
                       svc.work, tmp_path / "e", threading.Event(), 0)["ok"]
    reply = svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"}, "_seq": 5,
                                       "_version": "r"}, snapshot, tmp_path / "s5",
                        threading.Event(), 0)
    assert target in reply["superseded"] and target not in boards(svc)


def test_a_refused_review_is_kept_against_the_panel_it_was_written_about(svc, tmp_path):
    import shutil
    project(svc)
    fake = Fake(svc)
    target = plan_states(json.loads((svc.work / "plan.json").read_text()))[0]["id"]

    def draw(name, seq):
        return svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"},
                                          "_seq": seq, "_version": "r"}, svc.work,
                           tmp_path / name, threading.Event(), 0)

    fake.colors[target] = (0, 0, 255)
    draw("s1", 1)
    seen = tmp_path / "accept_view"
    shutil.copytree(svc.work, seen)
    fake.colors[target] = (255, 0, 0)
    draw("s2", 2)
    (tmp_path / "a").mkdir()
    refused = svc.execute("accept_storyboard", {"reviews": {target: "Blue pose."}, "_seq": 3,
                                                "_version": "r"}, seen, tmp_path / "a",
                          threading.Event(), 0)
    assert target in refused["not_applied"]
    assert draw("s3", 4)["states"][target].get("draft_review") is None
    fake.colors[target] = (0, 0, 255)
    assert draw("s4", 5)["states"][target].get("draft_review") == "Blue pose."


def test_an_older_final_render_finishing_last_keeps_the_newer_receipt(svc, tmp_path, no_encode):
    import shutil
    accepted(svc, tmp_path)
    old = tmp_path / "old"
    shutil.copytree(svc.work, old)
    old_content = svc.content_version(old)
    (svc.work / "animation.blend").write_bytes(b"blend-edited")
    assert compile_(svc, tmp_path, "c")["ok"]
    assert final_render(svc, tmp_path, "new", _seq=3)["ok"]
    (tmp_path / "old_render").mkdir()
    svc.execute("final_render", {"_content": old_content, "_version": "v", "_seq": 2},
                old, tmp_path / "old_render", threading.Event(), 0)
    assert json.loads((svc.native / "delivery.json").read_text())["submitted"] == 3
    assert finish(svc, tmp_path, "done", criteria=CRITERIA)["ok"]


def test_a_parallel_check_is_sized_by_the_last_edit(svc, tmp_path, monkeypatch):
    """The plan came from the length the last check saw, so a film an edit had
    lengthened was checked only to its old end, and the result cached clean."""
    project(svc)
    fake = Fake(svc, n_frames=300)
    svc.timeline_hint = {"n_frames": 240, "requires_preroll": False, "declarations": {}}

    def lengthening(ws, output, request, cancel, **kw):
        result = fake(ws, output, request, cancel, **kw)
        if request["op"] == "edit":
            write_json(output / "animation_changed.json", {"changed_frames": [], "n_frames": 300})
        return result

    svc.worker = lengthening
    assert edit(svc, tmp_path, "e", stage="storyboard")["ok"]
    monkeypatch.setattr(svc, "check_workers", lambda workers=None: 4)
    assert svc.chunk_plan()[-1][1] == 299


def test_a_span_edit_only_places_its_states(svc, tmp_path):
    """Placing the states evaluated every state frame too, which on a scene
    with a simulation stepped it to the last state before every span edit."""
    project(svc, three_states)
    fake = Fake(svc, n_frames=10, frames={"a": 0, "b": 5, "c": 9})
    storyboard(svc, tmp_path, "s")
    assert accept(svc, tmp_path, "acc", {"a": "start pose", "b": "middle pose"})["ok"]
    fake.requests.clear()
    assert edit(svc, tmp_path, "span", stage="animation", span=["a", "b"])["ok"]
    placed = [r for r in fake.requests if r["op"] == "evaluate"]
    assert placed and all(r.get("place_only") for r in placed)


def test_render_settings_change_only_what_they_name():
    """Fields a request leaves out keep that operation's own defaults: a sized
    draft stays Workbench, a review asked for more samples stays 640x360."""
    from dycraft.animation.blender.render_config import (
        DIAGNOSTIC_RENDER_SETTINGS, REVIEW_RENDER_SETTINGS, final_render_settings, over_defaults)
    draft = final_render_settings(over_defaults({"width": 640, "height": 360},
                                                DIAGNOSTIC_RENDER_SETTINGS))
    assert (draft["engine"], draft["width"], draft["height"]) == ("BLENDER_WORKBENCH", 640, 360)
    review = final_render_settings(over_defaults({"samples": 16}, REVIEW_RENDER_SETTINGS))
    assert (review["engine"], review["width"], review["height"], review["samples"]) == (
        "CYCLES", 640, 360, 16)
    # Another engine keeps the default size and none of the old engine's settings.
    look = final_render_settings(over_defaults({"engine": "WORKBENCH"}, REVIEW_RENDER_SETTINGS))
    assert (look["engine"], look["width"], look["height"]) == ("BLENDER_WORKBENCH", 640, 360)
