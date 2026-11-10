"""Policy and state boundaries, without a model call or Blender dependency."""
import json
from pathlib import Path
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dycraft.animation.blender.bundle import tree_hash, write_json
from dycraft.animation.planning import (missing_states, plan_states,
                                                required_checks_pass,
                                                state_revision, validate_plan)
from dycraft.animation.visual import rgb_temporal
from dycraft.tests._animation_fixtures import call, flat_plan, plan, service


def test_layer_registration_and_prompt(tmp_path):
    from dycraft.agent import LAYERS, stage_harness_client, _HARNESS_CORE_SRC
    from dycraft.agent.agent_runner import compose_system_prompt
    from dycraft.agent.layers.server import LAYER_NAMES
    assert "animation" in LAYER_NAMES
    prompt = compose_system_prompt("animation")
    assert "An animation edit needs the whole storyboard accepted" in prompt
    assert "Commit with `h.finish(criteria=" in prompt
    spec = LAYERS["animation"]
    # Exec the client the way the launcher assembles it -- transport core first,
    # then the stubs -- because the stubs now call `_batch_wait` from the core.
    calls = []
    # `__file__` because the core resolves `RUN_DIR` from it, as it does in a run dir.
    namespace = {"__file__": str(tmp_path / "harness.py")}
    exec(_HARNESS_CORE_SRC.read_text().split("# Operations:")[0], namespace)
    namespace["_call"] = lambda op, args, **kw: calls.append((op, args, kw)) or (op, args, kw)
    exec(spec.tools_src.read_text(), namespace)
    assert namespace["final_render"]()[0] == "final_render"
    assert namespace["blender"]("pass")[1]["stage"] == "animation"
    assert all(name in namespace for name in ("build", "plan", "storyboard", "review", "dispatch_scene", "dispatch_motion"))

    # `batch=True` must hand back the handle rather than wait for it, or `True`
    # and `False` are the same call and an author gets no concurrency.
    assert namespace["compile"](batch=True)[2]["wait"] == 0
    assert namespace["compile"](batch=False)[2]["wait"] is None
    assert namespace["compile"]()[2]["wait"] is None

    # Every op in the op table takes `batch=`/`fresh=`, as the prompt says.
    import inspect as _inspect
    for name in ("finish", "probe", "inspect", "ik"):
        params = _inspect.signature(namespace[name]).parameters
        assert "batch" in params and "fresh" in params, name
    assert namespace["probe"]("1", batch=True)[2]["wait"] == 0

    # A dispatch returns a *contribution* job id, which lives in a different
    # store from the one `job_result` reads -- so auto-waiting on it raised
    # `unknown job_id` and exited the client after the child had launched.
    assert namespace["dispatch_object"]("n", "b")[2]["wait"] == 0
    assert "contribution_status" in namespace["_JOB_METHODS"]


def test_storyboard_coverage_and_revisions():
    """Coverage is per state now; a nested plan's states take `shot/index` ids."""
    value = validate_plan(plan())
    states = plan_states(value)
    assert [s["id"] for s in states] == ["slide/0", "slide/1"]
    boards = {s["id"]: {"state_revision": state_revision(value, s),
                        "design_id": "d", "reviewed": True} for s in states}
    assert not missing_states(value, boards, "d")
    assert missing_states(value, boards, "new") == ["slide/0", "slide/1"]
    # A nested state still carries its absolute frame, so moving it rotates
    # the revision exactly as it did before.
    value["shots"][0]["states"][0]["frame"] = 5
    assert missing_states(value, boards, "d") == ["slide/0"]
    with pytest.raises(ValueError, match="unique"):
        validate_plan({**plan(), "shots": plan()["shots"] * 2})


def test_a_top_level_state_carries_no_timing_so_a_retime_cannot_discard_its_review():
    """The property the whole clip model exists for.

    Under the nested shape a state is identified by an absolute frame, so
    retiming rotates its revision and discards a review of a picture that did
    not change. A top-level state has no frame at all.
    """
    base = {k: v for k, v in plan().items() if k != "shots"}
    value = validate_plan({**base,
                           "shots": [{"id": "slide", "intent": "show it", "camera": "Camera"}],
                           "states": [{"id": "start", "purpose": "initial support"},
                                      {"id": "end", "purpose": "final support"}]})
    states = plan_states(value)
    assert [s["id"] for s in states] == ["start", "end"]
    assert all("frame" not in s for s in states)
    boards = {s["id"]: {"state_revision": state_revision(value, s),
                        "design_id": "d", "reviewed": True} for s in states}
    assert not missing_states(value, boards, "d")
    # There is no timing in the record to change, so nothing a retime does
    # can reach these revisions.
    assert state_revision(value, states[0]) == state_revision(value, dict(states[0]))
    with pytest.raises(ValueError, match="not both"):
        validate_plan({**value, "shots": [{"id": "slide", "intent": "i", "camera": "Camera",
                                           "states": [{"frame": 0, "purpose": "p"}]}]})


def test_required_checks_are_complete_but_review_and_continuity_are_advisory():
    reports = [{"name": n, "verdict": "pass"} for n in
               ("AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied")]
    assert required_checks_pass(reports + [{"name": "MotionContinuity", "verdict": "fail"}])
    assert not required_checks_pass(reports[:-1])
    reports[2]["verdict"] = "uncertain"
    assert not required_checks_pass(reports)


def test_rgb_unchanged_pixels_neutral_and_two_frame_color_is_red_cyan():
    first = Image.new("RGB", (8, 8), (30, 50, 70))
    neutral = np.asarray(rgb_temporal([first, first, first]))
    assert np.array_equal(neutral[..., 0], neutral[..., 1])
    assert np.array_equal(neutral[..., 1], neutral[..., 2])
    changed = np.asarray(rgb_temporal([first, Image.new("RGB", (8, 8), "white")]))
    assert np.array_equal(changed[..., 1], changed[..., 2])
    assert not np.array_equal(changed[..., 0], changed[..., 1])


def test_failed_plan_and_stale_publication_preserve_project(tmp_path):
    svc = service(tmp_path)
    before = svc.version()
    content = svc.content_version(svc.work)
    assert not call(svc, "plan", plan={})["ok"]
    assert svc.version() == before
    assert call(svc, "plan", plan=plan())["ok"]
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="changed outside the harness"):
        svc.publish(candidate, {"_content": content}, threading.Event())
    assert candidate.exists()
    svc.jobs.close()


def test_missing_build_cannot_animate_or_finish(tmp_path):
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    assert not call(svc, "blender", code="pass")["ok"]
    assert not call(svc, "finish")["ok"]
    svc.jobs.close()


def test_symlink_source_rejection_does_not_latch_edit_lock(tmp_path):
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    (svc.design / "outside.py").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        svc.submit("build")
    assert not svc.edit_pending
    (svc.design / "outside.py").unlink()
    assert call(svc, "plan", plan=plan())["ok"]
    svc.jobs.close()


def test_private_contributions_inherit_runtime_and_never_integrate(tmp_path, monkeypatch):
    from dycraft.animation import dispatch as module
    svc = service(tmp_path)
    (svc.design / "scene.py").write_text("parent source")
    handlers = {}
    registration = type("Registration", (), {
        "trace": lambda *a, **k: None,
        "op": lambda self, name: lambda fn: handlers.setdefault(name, fn),
        "register_handler": lambda self, name, fn: handlers.setdefault(name, fn),
    })()
    module.register(registration, svc)
    captured = []
    def run(argv, **kwargs):
        captured.append(argv)
        root = tmp_path / "contributions" / argv[argv.index("--run-id") + 1]
        (root / "scene.py").write_text("child source")
        write_json(root / "ir.json", {"id": "private"})
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(module, "run_cancellable", run)
    monkeypatch.setenv("DYCRAFT_SUB_BACKEND", "codex")
    # Same metadata route used by real CLI descendants.
    write_json(tmp_path / "meta.json", {"runtime": "codex1_cli", "model": "gpt-6", "reasoning_effort": "high"})
    version = svc.version()
    result = handlers["dispatch_scene"](name="layout", brief="Revise the block arrangement")
    svc.contributions.threads[-1].join(timeout=5)
    record = svc.contributions.status(result["job_id"])
    assert record["result"]["ok"], record
    assert (svc.design / "scene.py").read_text() == "parent source"
    assert svc.version() == version
    assert captured[0][captured[0].index("--layer") + 1] == "scene"
    assert captured[0][captured[0].index("--model") + 1] == "gpt-6"
    assert (tmp_path / "contributions/layout/input_sources/scene.py").read_text() == "parent source"
    with pytest.raises(ValueError, match="already exists"):
        handlers["dispatch_scene"](name="layout", brief="Do it again")
    registration.drain_on_shutdown(timeout_s=1)


def _contribution(tmp_path, **files):
    base = tmp_path / "contribution"
    base.mkdir()
    for name, text in files.items():
        (base / name.replace("__", ".")).write_text(text)
    return base


@pytest.mark.parametrize("files, returncode, committed, want", [
    ({"finish_result__json": '{"ok": true}', "trace__jsonl": '{"ev": "op_call"}\n'}, 0, True,
     {"finished": True, "ended_by": "finished"}),
    ({"finish_result__json": '{"ok": false, "error": "Render the current project first."}',
      "trace__jsonl": '{"ev": "op_call"}\n'}, 0, False,
     {"finished": False, "ended_by": "finish_refused",
      "finish_refused": "Render the current project first."}),
    ({"trace__jsonl": '{"ev": "op_call"}\n',
      "child__log": "[codex] session rollout recorded a fatal error with no activity for 180s: "
                    "usage_limit_exceeded: You've hit your usage limit\n"}, 86, False,
     {"finished": False, "ended_by": "cli_fatal_error",
      "cli_error": "usage_limit_exceeded: You've hit your usage limit"}),
    ({}, 87, False, {"finished": False, "ended_by": "blocked_at_startup", "reached_harness": False}),
    ({"trace__jsonl": '{"ev": "daemon_start"}\n'}, 1, False,
     {"finished": False, "ended_by": "never_reached_harness"}),
    ({"trace__jsonl": '{"ev": "op_call"}\n'}, 143, False, {"finished": False, "ended_by": "terminated"}),
    ({"trace__jsonl": '{"ev": "op_call"}\n', "ir__json": "{}"}, 0, True,
     {"finished": True, "ended_by": "finished"}),
], ids=["finished", "finish_refused", "usage_limit", "blocked", "never_started", "terminated",
        "object_committed"])
def test_a_contribution_says_how_it_ended(tmp_path, files, returncode, committed, want):
    """`finished` must not mean only that `finish_result.json` exists, which a
    refusal writes too, and why a child stopped must not live only in
    `child.log`, or a launcher reads a usage-limit stop as a crash."""
    from dycraft.animation.dispatch import how_it_ended
    base = _contribution(tmp_path, **files)
    got = how_it_ended(base, returncode, committed=committed)
    assert {k: got[k] for k in want} == want


def test_dispatch_refusals_say_which_wait_is_worth_it(tmp_path, monkeypatch):
    from dycraft.animation import dispatch as module
    svc = service(tmp_path)
    handlers = {}
    registration = type("Registration", (), {
        "trace": lambda *a, **k: None,
        "op": lambda self, name: lambda fn: handlers.setdefault(name, fn),
        "register_handler": lambda self, name, fn: handlers.setdefault(name, fn),
    })()
    module.register(registration, svc)
    svc.edit_pending = True
    with pytest.raises(ValueError, match="an edit is still running; dispatch again once"):
        handlers["dispatch_scene"](name="a", brief="b")
    svc.edit_pending = False
    svc.ctx.committed = True
    with pytest.raises(ValueError, match="the project is committed"):
        handlers["dispatch_scene"](name="a", brief="b")
    registration.drain_on_shutdown(timeout_s=1)


def test_a_cancelled_contribution_frees_its_slot(tmp_path, monkeypatch):
    """Its process takes a moment to stop; a dispatch refused in that moment
    was refused for a slot that was already free."""
    from dycraft.animation import dispatch as module
    svc = service(tmp_path)
    (svc.design / "scene.py").write_text("parent source")
    handlers = {}
    registration = type("Registration", (), {
        "trace": lambda *a, **k: None,
        "op": lambda self, name: lambda fn: handlers.setdefault(name, fn),
        "register_handler": lambda self, name, fn: handlers.setdefault(name, fn),
    })()
    module.register(registration, svc)
    release = threading.Event()
    seen = []

    def run(argv, cancel=None, **kwargs):
        seen.append(argv)
        release.wait(10)        # still stopping after its cancel was flagged
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(module, "run_cancellable", run)
    write_json(tmp_path / "meta.json", {"runtime": "codex1_cli", "model": "gpt-6"})
    monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", "2")
    jobs = [handlers["dispatch_scene"](name=f"c{i}", brief="b")["job_id"] for i in range(3)]
    with pytest.raises(ValueError, match="three contributions are already running"):
        handlers["dispatch_scene"](name="c3", brief="b")
    handlers["cancel_contribution"](jobs[0])
    assert handlers["dispatch_scene"](name="c3", brief="b")["job_id"]
    # A contributor gets what is left of its launcher's budget, not the CLI's default hour.
    hours = float(seen[0][seen[0].index("--advisory-hours") + 1])
    assert 1.97 <= hours <= 2.0
    release.set()
    registration.drain_on_shutdown(timeout_s=5)


def test_a_contributor_gets_what_is_left_of_the_budget():
    """A child gets the launcher's remaining hours (at least a minute's worth),
    and no budget (0, not the CLI default) when the launcher has none."""
    from dycraft.animation.dispatch import child_hours
    assert child_hours(None) == 0
    assert child_hours({"advisory_hours": 4.0, "remaining_s": 3 * 3600}) == 3.0
    assert child_hours({"advisory_hours": 1.0, "remaining_s": -600}) == 60 / 3600


@pytest.mark.parametrize("shape", [plan, flat_plan], ids=["nested", "flat"])
def test_independent_review_receives_claims_and_view_command_but_cannot_gate_completion(
        tmp_path, monkeypatch, shape):
    from dycraft.agent import oneoff
    svc = service(tmp_path)
    assert call(svc, "plan", plan=shape())["ok"]
    (svc.work / "animation.blend").write_bytes(b"fixture")
    write_json(svc.work / "design_state/build.json", {"design_id": "fixture"})
    write_json(svc.work / "design_state/scene.json", {"id": "scene", "checks": [{"id": "extra", "text": "extra source claim"}]})
    captured = {}
    def reviewer(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "text": "The visual criterion fails in this test.", "outcome": "completed"}
    monkeypatch.setattr(oneoff, "run_oneoff", reviewer)
    rendered = []
    monkeypatch.setattr(svc, "worker",
                        lambda ws, out, request, cancel, **k: rendered.append(request) or {"paths": ["frame.png"]})
    version = svc.version()
    result = call(svc, "review")
    assert result["ok"] and result["advisory"] and result["reviewed"]
    assert svc.version() == version
    assert "extra source claim" in captured["prompt"]
    assert "support" in captured["prompt"]
    assert "dycraft.animation.review_render" in captured["prompt"]
    assert "--workspace" in captured["prompt"]

    # This reviewer is asked about materials, lighting, and composition, so its
    # evidence must not be solid-shaded: seeds render in the shared appearance
    # profile, not the Workbench diagnostic default.
    assert rendered and all(r["render_settings"]["engine"] == "CYCLES" for r in rendered)
    assert all(r["render_settings"]["denoise"] for r in rendered)
    assert result["render_settings"]["width"] == 640
    # And the reviewer's own renders inherit it, so what it chooses to look at
    # is shaded like what it was handed.
    assert "--render-settings" in captured["prompt"]
    assert '"engine": "CYCLES"' in captured["prompt"]

    rendered.clear()
    assert call(svc, "review", render_settings={"engine": "CYCLES", "width": 960, "height": 540})["ok"]
    assert [r["render_settings"]["width"] for r in rendered] == [960]
    with pytest.raises(ValueError, match="Unknown final render settings"):
        svc.execute("review", {"render_settings": {"nonsense": 1}, "_version": svc.version()},
                    svc.work, tmp_path / "bad", threading.Event(), 0)
    svc.jobs.close()


@pytest.mark.parametrize("reply, why", [
    ({"ok": False, "outcome": "timeout", "timed_out": True, "returncode": -15,
      "stderr_tail": "killed"}, "did not answer within 600 s"),
    ({"ok": False, "outcome": "spawn_error", "returncode": None,
      "stderr_tail": "Popen failed: OSError: [Errno 7] Argument list too long"}, "could not be started"),
    ({"ok": False, "outcome": "no_result", "returncode": 1, "stderr_tail": "Traceback"},
     "exited with status 1 without answering"),
], ids=["timeout", "spawn_error", "no_result"])
def test_a_review_that_produced_no_opinion_fails_and_says_why(tmp_path, monkeypatch, reply, why):
    """A review that reviewed nothing must not come back `ok: true` without a
    reason, or the agent goes on without the opinion it asked for."""
    from dycraft.agent import oneoff
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    (svc.work / "animation.blend").write_bytes(b"fixture")
    write_json(svc.work / "design_state/build.json", {"design_id": "fixture"})
    monkeypatch.setattr(oneoff, "run_oneoff", lambda **kwargs: dict(reply, text=None))
    monkeypatch.setattr(svc, "worker", lambda ws, out, request, cancel, **k: {"paths": ["frame.png"]})
    result = call(svc, "review")
    assert result["ok"] is False and result["reviewed"] is False
    assert result["failure_class"] == "harness"
    assert why in result["error"] and "you may call review() again" in result["error"]
    assert result["stderr_tail"] == reply["stderr_tail"] and result["text"] == ""
    svc.jobs.close()


def test_motion_image_stays_diagnostic_unless_asked_otherwise(tmp_path, monkeypatch):
    """A temporal difference image wants flat shading; the override is for when
    the change being located is itself a shading change."""
    from dycraft.animation import visual
    svc = service(tmp_path)
    requests = []

    def worker(ws, out, request, cancel, **k):
        requests.append(request)
        (out).mkdir(parents=True, exist_ok=True)
        for i in range(len(request["frames"])):
            Image.new("RGB", (4, 4), (i * 40, 0, 0)).save(out / f"{i}.png")
        return {"paths": [f"{i}.png" for i in range(len(request["frames"]))]}

    svc.worker = worker
    cancel = threading.Event()
    visual.motion_image(svc, {"frames": [0, 1]}, svc.work, tmp_path / "a", cancel)
    assert "render_settings" not in requests[-1]
    visual.motion_image(svc, {"frames": [0, 1], "render_settings": {"engine": "WORKBENCH", "width": 320, "height": 180}},
                        svc.work, tmp_path / "b", cancel)
    assert requests[-1]["render_settings"]["engine"] == "BLENDER_WORKBENCH"
    assert requests[-1]["render_settings"]["width"] == 320
    svc.jobs.close()


def _checkable(svc, tmp_path):
    """Minimum on-disk shape `check()` reads, without a CAD build or Blender."""
    workspace = tmp_path / "ws"
    (workspace / "design_state").mkdir(parents=True)
    (workspace / "animation.blend").write_bytes(b"blend")
    write_json(workspace / "design_state/build.json",
               {"source_hash": "h", "source_files": [], "design_id": "d", "verifier_reports": []})
    write_json(workspace / "design_state/baseline.json", {"identities": {}, "joints": {}})
    calls = []

    def worker(ws, output, request, cancel, **kw):
        calls.append(request)
        (output / "evaluation").mkdir(parents=True)
        (output / "evaluation/index.json").write_text("{}")
        write_json(output / "verifier_reports.json",
                   [{"name": n, "verdict": "pass", "evidence": {}, "suggested_repairs": []}
                    for n in ("AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied")])
        write_json(output / "collision_pair_cache.json", {})
        frames = request.get("frames") or list(range(4))
        if request.get("start") is not None:
            frames = list(range(request["start"], request["end"] + 1))
        return {"fps": 24.0, "n_frames": 4, "frames": frames}

    svc.worker = worker
    return workspace, calls


def test_full_checks_are_cached_and_partial_ones_never_are(tmp_path):
    """The second whole-timeline check of an unchanged project is free.

    Partial checks are deliberately excluded: a repair loop edits between
    them so an entry could never be hit, and in a one-entry cache it would
    evict the full result that delivery and completion reuse.
    """
    svc = service(tmp_path)
    workspace, calls = _checkable(svc, tmp_path)
    cancel = threading.Event()

    first = svc.check(workspace, tmp_path / "a", cancel)
    assert not first["validation_reused"] and not first["partial"] and len(calls) == 1
    assert first["frames_checked"] == {"start": 0, "end": 3, "count": 4, "seconds": [0.0, 3 / 24]}

    second = svc.check(workspace, tmp_path / "b", cancel)
    assert second["validation_reused"] and len(calls) == 1
    assert second["ok"] == first["ok"]
    assert second["verifier_reports"] == first["verifier_reports"]

    # An interval always re-runs, and says so.
    partial = svc.check(workspace, tmp_path / "c", cancel, start=1, end=2)
    assert partial["partial"] and not partial["validation_reused"] and len(calls) == 2
    assert partial["frames_checked"] == {"start": 1, "end": 2, "count": 2, "seconds": [1 / 24, 2 / 24]}
    svc.check(workspace, tmp_path / "d", cancel, start=1, end=2)
    assert len(calls) == 3
    # The interval reaches the worker in frames; seconds is the alternate.
    assert [c.get("start") for c in calls] == [None, 1, 1]
    assert all(c.get("seconds") is None for c in calls)

    # The seconds spelling reaches the worker as seconds, untranslated: the
    # evaluator resolves it, because only it knows the scene's frame rate.
    svc.check(workspace, tmp_path / "s", cancel, seconds=[0, 1])
    assert calls[-1]["seconds"] == [0, 1] and calls[-1]["start"] is None

    # Editing the project invalidates it.
    (workspace / "animation.blend").write_bytes(b"edited")
    assert not svc.check(workspace, tmp_path / "e", cancel)["validation_reused"]
    assert len(calls) == 5
    svc.jobs.close()


def test_a_check_s_workers_are_capped_by_the_cpus_and_the_free_memory(tmp_path, monkeypatch):
    """`workers=N` asks; the harness caps it by the CPUs this process may use
    and by free memory at 2.5 GiB a worker, and the reply says what ran."""
    from dycraft.animation import checking as animation_checking
    svc = service(tmp_path)
    svc.CHECK_WORKERS = 4
    monkeypatch.setattr(animation_checking.os, "sched_getaffinity", lambda pid: set(range(6)))
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: 60.)
    assert svc.check_workers() == 4
    assert svc.check_workers(16) == 6
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: 3.)
    assert svc.check_workers(16) == 1
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: None)
    assert svc.check_workers(16) == 6
    for bad in (0, -1, 2.5, True, "4"):
        with pytest.raises(ValueError, match="positive integer"):
            svc.check_workers(bad)

    svc.timeline_hint = {"n_frames": 120, "fps": 24, "declarations": {"events": []},
                         "requires_preroll": False}
    assert len(svc.chunk_plan(3)) == 3
    monkeypatch.setattr(animation_checking, "mem_available_gib", lambda: 3.)
    assert svc.chunk_plan(3) is None, "one worker's memory runs the check whole"

    workspace, _ = _checkable(svc, tmp_path)
    reply = svc.check(workspace, tmp_path / "a", threading.Event(), workers=3)
    assert reply["workers"] == 1
    with pytest.raises(ValueError, match="positive integer"):
        svc.check(workspace, tmp_path / "i", threading.Event(), start=0, end=1, workers=0)
    assert "workers" not in svc.check(workspace, tmp_path / "b", threading.Event())
    svc.jobs.close()


def test_a_compile_with_an_interval_is_a_diagnostic_not_the_verdict(tmp_path):
    from dycraft.animation.jobs import counts_as_compile, is_partial
    for args in ({}, {"start": None, "end": None}):
        assert counts_as_compile("compile", args) and counts_as_compile("finish", args)
        assert not counts_as_compile("check_motion", args) and not is_partial(args)
    for args in ({"start": 0, "end": 1}, {"seconds": [0, 1]}):
        assert is_partial(args)
        assert not counts_as_compile("compile", args)
        assert not counts_as_compile("finish", args)


def test_a_shot_may_declare_an_interval_and_it_is_resolved_not_enforced():
    from dycraft.animation.planning import shot_interval
    base = {"intent": "x", "criteria": [{"id": "c", "requirement": "r", "why": "w", "observe": "o"}],
            "shots": [{"id": "s1", "intent": "i", "camera": "Camera", "frames": [0, 10],
                       "states": [{"frame": 0, "purpose": "p"}, {"frame": 10, "purpose": "q"}]}]}
    assert shot_interval(validate_plan(base), "s1") == (0, 10)

    # Optional: a plan without one is still valid, it just cannot be addressed.
    bare = json.loads(json.dumps(base))
    del bare["shots"][0]["frames"]
    with pytest.raises(ValueError, match="declares no frames interval"):
        shot_interval(validate_plan(bare), "s1")

    for broken, match in (
            ([10, 0], "inclusive"), ([0], "inclusive"), (["a", "b"], "inclusive")):
        bad = json.loads(json.dumps(base))
        bad["shots"][0]["frames"] = broken
        with pytest.raises(ValueError, match=match):
            validate_plan(bad)

    # The plan has to be internally coherent: a state outside its own shot.
    stray = json.loads(json.dumps(base))
    stray["shots"][0]["states"][1]["frame"] = 11
    with pytest.raises(ValueError, match="outside its declared frames"):
        validate_plan(stray)


@pytest.mark.parametrize("shape", [plan, flat_plan], ids=["nested", "flat"])
def test_review_renders_the_frames_the_author_names_as_well_as_the_planned_states(
        tmp_path, monkeypatch, shape):
    """`focus` is prose and cannot select a frame; `frames` can."""
    from dycraft.agent import oneoff
    svc = service(tmp_path)
    try:
        assert call(svc, "plan", plan=shape())["ok"]
        (svc.work / "animation.blend").write_bytes(b"fixture")
        write_json(svc.work / "design_state/build.json", {"design_id": "fixture"})
        monkeypatch.setattr(oneoff, "run_oneoff",
                            lambda **kw: {"ok": True, "text": "", "outcome": "completed"})
        requests = []
        monkeypatch.setattr(svc, "worker", lambda ws, out, req, cancel, **k: (
            requests.append(req) or {"paths": ["frame.png"]}))

        assert call(svc, "review")["ok"]
        if shape is plan:
            assert all(r.get("camera_name") for r in requests), "planned states use their shot camera"
        else:
            # A flat plan's states are drawn by id, as storyboard() draws
            # them: the worker resolves each one's frame and camera.
            assert len(requests) == 1 and "camera_name" not in requests[0]
            assert [s["id"] for s in requests[0]["states"]] == ["start", "end"]

        requests.clear()
        assert call(svc, "review", frames=[7, 9])["ok"]
        extra = [r for r in requests if r.get("frames") == [7, 9]]
        assert len(extra) == 1 and extra[0].get("camera_name") is None

        assert not call(svc, "review", frames=[])["ok"]
        assert not call(svc, "review", frames=list(range(33)))["ok"]
    finally:
        svc.jobs.close()


def test_the_animation_layer_slims_what_polling_hands_back(tmp_path):
    """The layer bound both job ops to the raw status call, so nothing was slimmed.

    It matters most here. `storyboard` is this surface's largest reply and it
    carries its check under `checks`, with no top-level `verifier_reports` for
    the flat slimmer to notice, and the prompt has the author poll every 5-15 s.
    """
    from dycraft.animation.jobs import slim_job_payload

    bulky = {"states": ["slide"], "checks": [{
        "ok": True,
        "verifier_reports": [
            {"name": "NoPenetration", "verdict": "pass",
             "evidence": {"pairs": [{"a": "x", "b": "y"}] * 500}},
            {"name": "EventsSatisfied", "verdict": "fail",
             "evidence": {"events": [{"id": "grasp", "ok": False}]}},
        ]}]}

    slim = slim_job_payload(bulky)
    check = slim["checks"][0]
    # The passing verifier folds to a name; the failing one keeps its evidence,
    # because that is the half an author repairs from.
    assert "NoPenetration" in json.dumps(check)
    assert len(json.dumps(check)) < len(json.dumps(bulky["checks"][0]))
    assert any(r["name"] == "EventsSatisfied" for r in check.get("verifier_reports", []))
    # The original is untouched: polling must not consume the stored result.
    assert len(bulky["checks"][0]["verifier_reports"]) == 2

    # And the layer really registers those wrappers, rather than binding the
    # raw status call under both names as it did before.
    from dycraft.agent.layers.film import FilmSession

    session = FilmSession(tmp_path)
    try:
        raw = session.blender_animation.jobs.status
        assert session.handlers["job_status"] is not raw
        assert session.handlers["job_result"] is not raw
        # All three are traced wrappers, `job_cancel` included: registered
        # through `register_handler`, they emit no trace record of their own,
        # so without the wrappers the run directory holds the jobs but not the
        # waiting.
        # It is still the cancel call underneath, which the unknown-id refusal
        # below checks behaviorally rather than by identity.
        assert session.handlers["job_cancel"] is not raw
        assert session.handlers["job_cancel"] is not session.blender_animation.jobs.cancel
        with pytest.raises(ValueError, match="unknown job_id"):
            session.handlers["job_cancel"]("no-such-job")
        # `job_status` answers whether a result exists; it does not carry one.
        job = session.blender_animation.jobs
        job.jobs["fixture"] = ({"job_id": "fixture", "status": "completed",
                                "result": bulky}, threading.Event())
        assert session.handlers["job_status"]("fixture")["result_available"] is True
        assert "result" not in session.handlers["job_status"]("fixture")
        assert session.handlers["job_result"]("fixture")["result"] != bulky
    finally:
        session.blender_animation.jobs.close()


def test_a_source_repair_only_invalidates_states_that_show_the_changed_object():
    """Design staleness is per state, not per build.

    Comparing one whole-build `design_id` against every board would make any
    source change invalidate all of them, re-rendering states that do not show
    the repaired object.
    """
    from dycraft.animation.planning import missing_states, state_fingerprint

    plan = {"intent": "i", "criteria": ["c"],
            "shots": [{"id": "s1", "camera": "Cam", "states": [{"purpose": "a"}, {"purpose": "b"}]}]}
    objects = {"chair": "h_chair", "bolt": "h_bolt", "table": "h_table"}
    shown = {"s1/0": ["chair", "bolt"], "s1/1": ["table"]}

    def boards_for(design_objects):
        from dycraft.animation.planning import plan_states, state_revision
        return {s["id"]: {"state_revision": state_revision(plan, s), "reviewed": True,
                          "design_id": "build0",
                          "design_instances": shown[s["id"]],
                          "design_fingerprint": state_fingerprint(design_objects, shown[s["id"]])}
                for s in plan_states(plan)}

    boards = boards_for(objects)
    assert missing_states(plan, boards, "build0", objects) == []

    # Repairing `bolt` invalidates only the state that shows it.
    repaired = {**objects, "bolt": "h_bolt_v2"}
    assert missing_states(plan, boards, "build1", repaired) == ["s1/0"]

    # Removing it scopes for free: the fingerprint looks the object up by the
    # state's own instance names, so it rotates for the state that showed it
    # and is untouched for every other one.
    removed = {k: v for k, v in objects.items() if k != "bolt"}
    assert missing_states(plan, boards, "build1", removed) == ["s1/0"]

    # Adding one cannot be answered from the design: no board recorded it, so
    # nothing rotates here. Whether it is *visible* in a panel is a question
    # for an evaluation, and `appeared_states` answers it from the next check.
    added = {**objects, "lamp": "h_lamp"}
    assert missing_states(plan, boards, "build1", added) == []

    # A board with no recorded instances falls back to the whole-build id.
    legacy = {k: {**v, "design_instances": None} for k, v in boards.items()}
    assert missing_states(plan, legacy, "build1", objects) == ["s1/0", "s1/1"]
    assert missing_states(plan, legacy, "build0", objects) == []


def test_an_object_added_since_review_is_caught_by_the_next_evaluation():
    """The other half of add-scoping, and the only part that gates.

    `missing_states` cannot see an added object, because no board recorded an
    instance that did not exist. What settles it is a fresh evaluation of the
    state's own frame: an instance visible there that was not visible when the
    panel was reviewed changed the picture the review is about.
    """
    from dycraft.animation.planning import appeared_states

    boards = {"s1/0": {"design_instances": ["chair", "bolt"], "reviewed": True},
              "s1/1": {"design_instances": ["table"], "reviewed": True}}

    # Nothing new in either frame.
    assert appeared_states(boards, {"s1/0": ["chair", "bolt"], "s1/1": ["table"]}) == {}

    # A lamp placed into the first state's frame; the second is untouched.
    assert appeared_states(boards, {"s1/0": ["chair", "bolt", "lamp"],
                                    "s1/1": ["table"]}) == {"s1/0": ["lamp"]}

    # A disappearance is not this function's business: the fingerprint catches
    # a removal from the design, and a merely hidden object is a staging
    # change the pose observation reports.
    assert appeared_states(boards, {"s1/0": ["chair"]}) == {}

    # A state the evaluation could not answer keeps whatever the gate decided.
    assert appeared_states(boards, {}) == {}


def test_pose_digest_separates_movement_from_appearance_and_has_a_tolerance():
    """What the drift observation is computed from.

    Color, visibility, and mesh identity are deliberately outside it: an
    appearance pass must not read as a pose change, or the observation fires
    on nearly every authored edit, since most touch keyframes, materials, and
    camera together.
    """
    from dycraft.animation.planning import POSE_QUANTUM, pose_digest

    base = {"transforms": {"a": [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            "joint_positions": {"j": 0.25},
            "objects": [{"name": "part/a/body", "hidden": False, "color": (1, 0, 0, 1)}]}
    assert pose_digest(base) == pose_digest(dict(base))

    # Appearance and visibility do not enter it.
    recoloured = {**base, "objects": [{"name": "part/a/body", "hidden": True,
                                       "color": (0, 1, 0, 1)}]}
    assert pose_digest(recoloured) == pose_digest(base)

    # Movement well under the quantum does not rotate it; movement over does.
    nudged = {**base, "transforms": {"a": [[1, 0, 0, 0.5 + POSE_QUANTUM / 10],
                                           [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
    assert pose_digest(nudged) == pose_digest(base)
    moved = {**base, "transforms": {"a": [[1, 0, 0, 0.51], [0, 1, 0, 0],
                                          [0, 0, 1, 0], [0, 0, 0, 1]]}}
    assert pose_digest(moved) != pose_digest(base)

    # A joint turning is movement too.
    turned = {**base, "joint_positions": {"j": 0.9}}
    assert pose_digest(turned) != pose_digest(base)

    # No placements at all reads as "no observation", not as "unchanged".
    assert pose_digest({"transforms": {}}) is None


def test_a_camera_move_is_not_a_pose_change_and_digest_rules_never_compare():
    """Cameras and lights frame the pose; they are not part of it.

    With them in the digest, moving a single camera would report nearly every
    reviewed state as drifted.
    """
    from dycraft.animation.planning import pose_digest, poses_comparable

    still = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    panned = [[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    base = {"transforms": {"a": still, "Cam": still}, "joint_positions": {"j": .25}}
    moved_camera = {**base, "transforms": {"a": still, "Cam": panned}}
    assert pose_digest(moved_camera, viewpoints=["Cam"]) == pose_digest(base, viewpoints=["Cam"])
    assert pose_digest(moved_camera) != pose_digest(base)
    # The body itself moving is still movement.
    moved_body = {**base, "transforms": {"a": panned, "Cam": still}}
    assert pose_digest(moved_body, viewpoints=["Cam"]) != pose_digest(base, viewpoints=["Cam"])

    # Digests from different rules are "not comparable", never "drifted".
    current = pose_digest(base, viewpoints=["Cam"])
    assert poses_comparable(current, pose_digest(moved_body, viewpoints=[]))
    assert not poses_comparable(current, pose_digest(base))           # pre-viewpoint evaluation
    assert not poses_comparable(current, "0123abcd")                  # pre-versioning board
    assert not poses_comparable(current, None)


def test_a_compile_reply_carries_a_digest_of_its_evaluation_not_the_index():
    """The compile reply does not echo the author's own input back.

    The index -- declarations, clip spans, collision objects, every frame --
    would dominate each reply. It stays on disk; the reply keeps the counts and
    the timeline facts.
    """
    from dycraft.animation.planning import evaluation_digest

    info = {"scene_ref": "s", "fps": 24.0, "n_frames": 480, "frame_start": 0,
            "complete": True, "frames": list(range(480)),
            "clips": {"a": {"frames": [0, 99]}, "b": {"frames": [100, 479]}},
            "collision_objects": ["x"] * 30,
            "declarations": {"units": "frames", "events": [{}] * 7,
                             "collision_allowances": [{}] * 40}}
    digest = evaluation_digest(info)
    assert digest["n_frames"] == 480 and digest["fps"] == 24.0
    assert digest["declarations"] == {"events": 7, "collision_allowances": 40,
                                      "liquids": 0, "units": "frames"}
    assert (digest["clips"], digest["collision_objects"], digest["frames"]) == (2, 30, 480)
    assert len(json.dumps(digest)) < len(json.dumps(info)) / 5
    assert evaluation_digest(None) is None


def test_a_long_render_gets_a_one_page_overview_beside_its_paged_sheets(tmp_path):
    """A whole-film draft gets an overview sheet too.

    The paged sheets keep every frame at a readable size, so a long draft comes
    back as many pages; one overview page lets a whole film be judged at a
    glance.
    """
    from dycraft.animation.sheets import contact_sheet, overview_sheet

    frames = []
    for index in range(96):
        path = tmp_path / f"{index:06d}.png"
        Image.new("RGB", (64, 36), (index, 0, 0)).save(path)
        frames.append((index, path))
    pages = contact_sheet(frames, tmp_path / "sheet.jpg", fps=24)
    assert len(pages) > 1
    overview = overview_sheet(frames, tmp_path / "sheet_overview.jpg", fps=24)
    assert overview and Path(overview).is_file()
    # One page, at the same cell width as the paged sheets.
    with Image.open(overview) as page, Image.open(pages[0]) as first:
        assert page.width == first.width and page.height <= 2048
    # A render that already fits one page needs no overview.
    assert overview_sheet(frames[:8], tmp_path / "small.jpg") is None


def test_a_panel_is_compared_with_what_its_own_camera_sees_now(monkeypatch):
    """The delivery gate and retention compare like with like.

    `storyboard()` records what a panel's camera frames; reading "present and
    unhidden" -- the whole scene -- on the other side would make every object
    out of shot look newly appeared, refusing `final_render` and defeating
    retention. Each side is measured the way the board was.
    """
    from dycraft.animation.planning import appeared_states
    from dycraft.animation.session import AnimationSession

    in_frame = ["block"]
    monkeypatch.setattr(AnimationSession, "framed_instances",
                        staticmethod(lambda check, record, view, settings: list(in_frame)))
    monkeypatch.setattr(AnimationSession, "instances_in",
                        staticmethod(lambda record: ["block", "floor", "prop"]))
    svc = AnimationSession.__new__(AnimationSession)
    boards = {"close": {"design_instances": ["block"], "design_scope": "framed",
                        "view": {"camera": "Close"}, "render_settings": {}},
              "wide": {"design_instances": ["block", "floor", "prop"], "design_scope": "scene"},
              "legacy": {"design_instances": ["block"]}}

    def appeared():
        observed = {name: svc.observed_instances({}, {"objects": []}, board)
                    for name, board in boards.items()}
        return appeared_states(boards, {n: o for n, o in observed.items() if o is not None})

    # Objects outside the close-up's frame are not "new" to it.
    assert appeared() == {}
    # Something that does enter its frame still costs the review.
    in_frame.append("lamp")
    assert appeared() == {"close": ["lamp"]}
    # A board that does not say how it was recorded is not answered at all.
    assert svc.observed_instances({}, {"objects": []}, boards["legacy"]) is None
    assert svc.observed_instances({}, None, boards["wide"]) is None


def test_segment_coverage_pairs_approved_states_and_never_spans_a_cut():
    """The unit an author reasons about: motion authored *between* two poses."""
    from dycraft.animation.planning import segment_coverage

    value = validate_plan({**{k: v for k, v in plan().items() if k != "shots"},
                           "shots": [{"id": "s", "intent": "i", "camera": "Cam"}],
                           "states": [{"id": "a", "purpose": "p"}, {"id": "b", "purpose": "p"},
                                      {"id": "c", "purpose": "p"}]})
    boards = {"a": {"reviewed": True, "clip": "one", "frame": 0},
              "b": {"reviewed": True, "clip": "one", "frame": 10},
              "c": {"reviewed": False, "clip": "one", "frame": 20}}
    cover = segment_coverage(value, boards)
    assert cover["segments"] == 2 and cover["covered"] == 1
    assert [s["covered"] for s in cover["detail"]] == [True, False]

    # Two states either side of a cut do not bracket a continuous motion, so
    # the pair spanning it is dropped rather than counted as uncovered.
    across = {**boards, "c": {"reviewed": True, "clip": "two", "frame": 20}}
    cut = segment_coverage(value, across)
    assert cut["segments"] == 1 and cut["covered"] == 1
    assert [s["from"] for s in cut["detail"]] == ["a"]


def test_a_panel_is_scoped_to_what_its_camera_frames_not_what_the_scene_contains():
    """A panel's instance set comes from what its camera frames.

    `instances_in` answers "present and unhidden", which is nearly the whole
    scene for every state, so scoped by it, a one-object repair would clear
    every reviewed panel.
    """
    import numpy as np
    from dycraft.animation.session import AnimationSession

    # Camera at the origin looking down -Z, Blender's convention.
    view = {"camera": "Cam", "camera_type": "PERSP",
            "camera_matrix": np.eye(4).tolist(),
            "lens_mm": 50.0, "sensor_mm": 36.0, "sensor_fit": "AUTO"}
    settings = {"width": 640, "height": 360}

    def scene(positions):
        objects, transforms = [], {}
        for name, (x, y, z) in positions.items():
            full = f"part/{name}/body"
            matrix = np.eye(4)
            matrix[:3, 3] = (x, y, z)
            objects.append({"name": full, "hidden": False, "matrix": matrix.tolist()})
            transforms[full] = matrix.tolist()
        return {"objects": objects, "transforms": transforms}

    record = scene({"infront": (0, 0, -5),      # dead center of frame
                    "behind": (0, 0, 5),        # behind the lens
                    "offside": (40, 0, -5)})    # far outside the horizontal fov

    # A tiny cube of geometry per body, so the box test has something to use.
    cube = (np.array([[-.1, -.1, -.1], [.1, .1, .1]]), np.zeros((0, 3), dtype=int))
    class Local(dict):
        def __missing__(self, key):
            return cube

    import dycraft.animation.blender.evaluation as ev
    real_local, real_world = ev.read_local, ev.world_geometry
    ev.read_local = lambda directory, rec: Local()
    try:
        framed = AnimationSession.framed_instances(
            {"evaluation_path": "/nonexistent"}, record, view, settings)
    finally:
        ev.read_local, ev.world_geometry = real_local, real_world

    assert framed == ["infront"], framed

    # An orthographic or missing camera cannot be resolved, and says so rather
    # than guessing -- the caller then falls back to present-and-unhidden.
    assert AnimationSession.framed_instances(
        {"evaluation_path": "/x"}, record, {**view, "camera_type": "ORTHO"}, settings) is None
    assert AnimationSession.framed_instances({"evaluation_path": "/x"}, record, None, settings) is None
    assert AnimationSession.framed_instances({}, record, view, settings) is None


def _storyboardable(svc, source_reports=()):
    """What `storyboard()` reads, and a worker that draws one flat panel per state.

    `colors` is what a test changes to alter a panel's picture; `requests`
    records every worker call, so a test can see which states were drawn.
    """
    assert call(svc, "plan", plan=plan())["ok"]
    workspace = svc.work
    (workspace / "design_state").mkdir(parents=True, exist_ok=True)
    (workspace / "animation.blend").write_bytes(b"blend")
    write_json(workspace / "design_state/build.json",
               {"source_hash": "h", "source_files": [], "design_id": "d",
                "verifier_reports": list(source_reports)})
    write_json(workspace / "design_state/baseline.json", {"identities": {}, "joints": {}})
    colors, requests = {}, []

    def worker(ws, output, request, cancel, **kw):
        requests.append(request)
        output.mkdir(parents=True, exist_ok=True)
        if request["op"] == "render":
            states = request["states"]
            for index, state in enumerate(states):
                Image.new("RGB", (4, 4), colors.get(state["id"], (0, 0, 0))).save(
                    output / f"{index}.png")
            return {"paths": [f"{i}.png" for i in range(len(states))],
                    "state_frames": {s["id"]: s["frame"] for s in states},
                    "state_clips": {}, "views": [None] * len(states)}
        (output / "evaluation").mkdir(parents=True)
        (output / "evaluation/index.json").write_text("{}")
        write_json(output / "verifier_reports.json",
                   [{"name": n, "verdict": "pass", "evidence": {}, "suggested_repairs": []}
                    for n in ("AnimationWellFormed", "JointLimits", "NoPenetration", "EventsSatisfied")])
        write_json(output / "collision_pair_cache.json", {})
        return {"fps": 24.0, "n_frames": 4, "frames": request.get("frames") or list(range(4))}

    svc.worker = worker
    return colors, requests


def _storyboard(svc, tmp_path, name, **args):
    return svc.execute("storyboard", {"render_settings": {"engine": "WORKBENCH"}, **args,
                                      "_version": svc.revision_token()},
                       svc.work, tmp_path / name, threading.Event(), 0)


_ADVISORY = {"name": "JointOriginNearBothBodies", "verdict": "uncertain", "evidence": {},
             "suggested_repairs": ["Advisory: this bounds heuristic does not establish a missing mechanism."]}


def test_an_advisory_source_verdict_is_reported_but_does_not_empty_the_boards(tmp_path):
    """Only a design verifier that *fails* keeps evidence from becoming coverage.

    Requiring `pass` from every design verifier would let the one that returns
    `uncertain` by design empty every board, refusing storyboards with no
    failing state check and nothing in the reply saying why.
    """
    from dycraft.animation.planning import source_gate

    failing = {**_ADVISORY, "name": "SupportConnected", "verdict": "fail",
               "suggested_repairs": ["Connect the part."]}
    passing = {**_ADVISORY, "verdict": "pass", "suggested_repairs": []}
    blocked, advisories = source_gate([passing, _ADVISORY, failing])
    assert blocked == [{"verifier": "SupportConnected", "verdict": "fail",
                        "repairs": ["Connect the part."]}]
    assert advisories == [{"verifier": "JointOriginNearBothBodies", "verdict": "uncertain",
                           "repairs": _ADVISORY["suggested_repairs"]}]

    svc = service(tmp_path / "advisory")
    _storyboardable(svc, [_ADVISORY])
    reply = _storyboard(svc, tmp_path, "a")
    assert reply["ok"] and reply["blocked_by"] == [] and reply["advisories"] == advisories
    assert set(json.loads((svc.work / "storyboards.json").read_text())) == {"slide/0", "slide/1"}
    assert "accept_storyboard" in reply["next"]
    svc.jobs.close()

    svc = service(tmp_path / "failing")
    _storyboardable(svc, [_ADVISORY, failing])
    reply = _storyboard(svc, tmp_path, "b")
    assert not reply["ok"] and reply["blocked_by"] == blocked
    assert reply["failure_class"] == "check_finding"
    # Every state check passed; the reply now says what did not.
    assert all(c["ok"] for c in reply["checks"])
    assert all(not e["checks_pass"] and e["failing_here"] == ["SupportConnected"]
               for e in reply["states"].values())
    # Drawn, so each keeps a board -- marked failing, which nothing can accept.
    boards = json.loads((svc.work / "storyboards.json").read_text())
    assert set(boards) == {"slide/0", "slide/1"}
    assert not any(b["checks_pass"] for b in boards.values())
    assert reply["banked"] == [] and reply["needs_review"] == []
    assert "SupportConnected" in reply["next"] and "accept_storyboard" not in reply["next"]
    svc.jobs.close()


def test_a_whole_film_check_is_not_failed_by_an_advisory_source_verdict(tmp_path):
    """`compile` applies the same rule to the design's verdicts."""
    svc = service(tmp_path)
    workspace, _ = _checkable(svc, tmp_path)
    build = json.loads((workspace / "design_state/build.json").read_text())
    write_json(workspace / "design_state/build.json", {**build, "verifier_reports": [_ADVISORY]})
    assert svc.check(workspace, tmp_path / "a", threading.Event())["ok"]
    write_json(workspace / "design_state/build.json",
               {**build, "verifier_reports": [{**_ADVISORY, "verdict": "fail"}]})
    assert not svc.check(workspace, tmp_path / "b", threading.Event())["ok"]
    svc.jobs.close()


def _accept(svc, tmp_path, name, reviews):
    return svc.execute("accept_storyboard", {"reviews": reviews, "_version": svc.revision_token()},
                       svc.work, tmp_path / name, threading.Event(), 0)


def test_a_refused_acceptance_names_its_cause_and_keeps_the_observations(tmp_path):
    """Each refusal condition has its own message, naming every offender. The
    refused text is kept and offered back, never applied, and only against the
    panel it was written about."""
    from dycraft.tests._animation_fixtures import state_reviews

    svc = service(tmp_path)
    colors, _ = _storyboardable(svc, [{**_ADVISORY, "name": "SupportConnected", "verdict": "fail"}])
    _storyboard(svc, tmp_path, "a")
    reviews = state_reviews("The block rests flat on the floor.")
    with pytest.raises(ValueError, match="fail at their own frame and cannot be accepted: "
                                         r"slide/0 \(SupportConnected\); slide/1") as error:
        _accept(svc, tmp_path, "b", reviews)
    assert "kept" in str(error.value) and "empty" not in str(error.value)
    with pytest.raises(ValueError, match="empty observation for: slide/1"):
        _accept(svc, tmp_path, "c", {"slide/0": "fine", "slide/1": "  "})

    # The design is repaired; the panels come back identical, so the drafts do.
    build = json.loads((svc.work / "design_state/build.json").read_text())
    write_json(svc.work / "design_state/build.json", {**build, "verifier_reports": []})
    colors["slide/1"] = (200, 0, 0)            # this one's picture changed
    reply = _storyboard(svc, tmp_path, "d")
    assert reply["ok"] and reply["needs_review"] == ["slide/0", "slide/1"]
    # The second refusal kept "fine" for slide/0 over the first call's text.
    assert reply["states"]["slide/0"]["draft_review"] == "fine"
    assert "draft_review" not in reply["states"]["slide/1"]
    assert "draft_review for slide/0" in reply["next"]
    assert not json.loads((svc.work / "storyboards.json").read_text())["slide/0"]["reviewed"]

    assert _accept(svc, tmp_path, "e", reviews)["ok"]
    assert json.loads(svc.drafts_path.read_text()) == {}
    svc.jobs.close()


def test_author_sources_are_readable_at_their_own_path_and_cannot_overwrite_the_harness(tmp_path):
    """The path an author writes, `/workspace/animation_src/x.json`, exists.

    Without it, an author's data ends up pasted into code strings. A top-level
    name clash must not replace harness state, such as a validated
    `plan.json`."""
    svc = service(tmp_path)
    sources = svc.root / "animation_src"
    (sources / "data").mkdir(parents=True)
    (sources / "motion.json").write_text('{"fps": 20}')
    (sources / "data/helper.py").write_text("VALUE = 1\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "plan.json").write_text('{"id": "validated"}')

    svc.stage_sources(sources, workspace)
    # Both views, and the flat one keeps `import helper` working.
    assert (workspace / "motion.json").read_text() == '{"fps": 20}'
    assert (workspace / "animation_src/motion.json").read_text() == '{"fps": 20}'
    assert (workspace / "animation_src/data/helper.py").is_file()
    # The harness's own bytes, never a link to the author's file: an
    # in-place edit to the helper must not reach the project until the next
    # blender() call. The two views share one copy.
    assert (workspace / "animation_src/motion.json").stat().st_ino != (sources / "motion.json").stat().st_ino
    assert (workspace / "animation_src/motion.json").stat().st_ino == (workspace / "motion.json").stat().st_ino
    (sources / "motion.json").write_text('{"fps": 99}')
    assert (workspace / "animation_src/motion.json").read_text() == '{"fps": 20}'
    # Restaging lands on a fresh inode, so a snapshot linked to the old copy keeps it.
    snapshot = tmp_path / "snapshot_motion.json"
    import os
    os.link(workspace / "motion.json", snapshot)
    svc.stage_sources(sources, workspace)
    assert (workspace / "motion.json").read_text() == '{"fps": 99}'
    assert snapshot.read_text() == '{"fps": 20}'
    (sources / "motion.json").write_text('{"fps": 20}')
    assert json.loads((workspace / "plan.json").read_text())["id"] == "validated"

    (sources / "plan.json").write_text('{"id": "the author\'s own"}')
    with pytest.raises(ValueError, match="plan.json"):
        svc.stage_sources(sources, workspace)
    assert json.loads((workspace / "plan.json").read_text())["id"] == "validated"
    svc.jobs.close()


def _mask(directory):
    """What Claude Code's sandbox leaves in a directory its shell ran in."""
    import os
    (directory / ".claude").mkdir()
    (directory / ".claude/settings.json").write_text("")
    (directory / ".mcp.json").write_text("")
    os.chmod(directory / ".claude/settings.json", 0o444)
    os.chmod(directory / ".claude", 0o555)
    os.chmod(directory / ".mcp.json", 0o444)


def test_claude_s_sandbox_masks_are_never_copied_or_hashed(tmp_path):
    """The second `blender()` call's copy of `animation_src/` failed on the
    read-only copy of a mask the first had made (12 sandbox escapes removed
    them by hand), and a mask in `design/` changed the design's hash."""
    import os
    svc = service(tmp_path)
    sources = svc.root / "animation_src"
    sources.mkdir()
    (sources / "helper.py").write_text("VALUE = 1\n")
    before = tree_hash(sources)
    _mask(sources)
    try:
        assert tree_hash(sources) == before
        workspace = tmp_path / "ws"
        workspace.mkdir()
        for _ in range(2):
            svc.stage_sources(sources, workspace)
        assert (workspace / "helper.py").is_file()
        assert not (workspace / ".claude").exists() and not (workspace / ".mcp.json").exists()
        assert not (workspace / "animation_src/.claude").exists()
        design_hash = tree_hash(svc.design)
        _mask(svc.design)
        assert tree_hash(svc.design) == design_hash
        # A link is still refused, whatever it is called.
        os.symlink("/", workspace / ".mcp.json")
        with pytest.raises(ValueError, match="symlinks"):
            tree_hash(workspace)
    finally:
        for root in (sources, svc.design):
            if (root / ".claude").exists():
                os.chmod(root / ".claude", 0o755)
    svc.jobs.close()


def test_bytecode_caches_are_never_hashed_or_sealed(tmp_path):
    """Bytecode caches stay out of the IR, so a script merely run again does
    not change the project's hash."""
    from dycraft.animation.blender.bundle import make_manifest
    root = tmp_path / "bundle"
    (root / "evaluation").mkdir(parents=True)
    (root / "animation.blend").write_bytes(b"blend")
    (root / "evaluation/index.json").write_text("{}")
    (root / "helper.py").write_text("VALUE = 1\n")
    before = tree_hash(root)
    (root / "__pycache__").mkdir()
    (root / "__pycache__/helper.cpython-311.pyc").write_bytes(b"\x00bytecode")
    (root / "stray.pyc").write_bytes(b"\x00")
    assert tree_hash(root) == before
    manifest = make_manifest(root, scene_ref="s", evaluation={
        "fps": 24, "n_frames": 2, "frame_start": 0, "blender_version": "4.5.13"})
    assert not any(".pyc" in name or "__pycache__" in name for name in manifest["files"])
    assert "bundle/helper.py" in manifest["files"]


def test_a_large_job_result_is_traced_as_a_digest_and_a_file(tmp_path):
    """One trace reached 1.65 GB, with single lines up to 101 MB: whole job
    results, probe output included, written into `trace.jsonl`."""
    from dycraft.animation.blender.runtime import TRACE_RESULT_LIMIT, traced_result
    small = {"ok": True, "version": "v1", "value": [1, 2, 3]}
    assert traced_result(small, tmp_path) is small
    big = {"ok": False, "failure_class": "rejection", "version": "v2",
           "rows": ["x" * 1000] * (TRACE_RESULT_LIMIT // 1000 + 10)}
    digest = traced_result(big, tmp_path / "job")
    assert digest["ok"] is False and digest["failure_class"] == "rejection"
    assert "rows" not in digest
    held = Path(digest["traced_digest"]["file"])
    assert held == tmp_path / "job" / "result.json"
    assert json.loads(held.read_text()) == big
    # A batch job's record is already on disk: it is named, not written again.
    record = tmp_path / "batch" / "job.json"
    record.parent.mkdir()
    record.write_text(json.dumps({"result": big}))
    digest = traced_result(big, record.parent, persisted=record)
    assert digest["traced_digest"]["file"] == str(record)
    assert not (record.parent / "result.json").exists()


def test_a_stale_film_refusal_names_what_changed(tmp_path):
    """"Render the current project with final_render() before finish()" left
    an author to work out for itself what had moved since the film."""
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    (svc.work / "animation.blend").write_bytes(b"drawn")
    (svc.work / "animation_src").mkdir(exist_ok=True)
    (svc.work / "animation_src/helper.py").write_text("A = 1\n")
    receipt = {"version": "drawn", "content_parts": svc.content_parts(svc.work)}
    assert svc.stale_film(receipt, svc.work).endswith("this project's content.")
    (svc.work / "animation.blend").write_bytes(b"edited")
    (svc.work / "animation_src/helper.py").write_text("A = 2\n")
    message = svc.stale_film(receipt, svc.work)
    assert message.endswith("Changed since the film was drawn: animation.blend, animation_src/.")
    assert svc.stale_film(None, svc.work).endswith("No film has been drawn yet.")
    # A receipt from before parts were recorded keeps the plain refusal.
    assert svc.stale_film({"version": "old"}, svc.work).endswith("this project's content.")
    svc.jobs.close()


def test_a_stale_design_during_a_build_says_to_wait_for_it(tmp_path):
    """"design/ changed since build() read it; run h.build() again" also came
    while the build of the new sources was running, and authors started
    another."""
    svc = service(tmp_path)
    (svc.design / "scene.py").write_text("changed")
    write_json(svc.work / "design_state/build.json", {"design_id": "d", "source_hash": "older"})
    (svc.work / "animation.blend").write_bytes(b"blend")
    with pytest.raises(ValueError, match=r"run h.build\(\) again$"):
        svc.source_fresh(svc.work)
    svc.edit_running = {"operation": "build", "job_id": "j1"}
    with pytest.raises(ValueError, match=r"a build is running \(job j1\); wait for it"):
        svc.source_fresh(svc.work)
    svc.edit_running = None
    svc.jobs.close()


def test_a_penetration_comes_with_a_repair():
    """Every penetration failure carries a repair, not only those involving an
    allowance or a liquid."""
    from dycraft.animation.blender.verify import penetration_repair
    assert penetration_repair({}) == []
    repair, = penetration_repair({"pairs": [{"a": "hand/palm", "b": "cup/body"}]})
    assert "Separate each pair" in repair and "max_depth_m" in repair


def test_an_unknown_name_to_invalidate_is_refused_before_the_edit_runs(tmp_path):
    """Raised after the worker had applied the code, the refusal would discard
    the whole publish, and leave the live workspace mutated by an op that
    reports failure."""
    from dycraft.animation.planning import validate_plan
    from dycraft.animation.session import AnimationSession

    value = validate_plan(plan())
    boards = {"slide/0": {"clip": "transfer_09", "reviewed": True},
              "slide/1": {"clip": None, "reviewed": True}}
    resolve = AnimationSession.resolve_invalidations
    assert resolve(value, boards, ["slide"]) == {"slide/0", "slide/1"}      # a shot
    assert resolve(value, boards, ["transfer_09"]) == {"slide/0"}           # a clip
    assert resolve(value, boards, ["slide/1"]) == {"slide/1"}               # a state
    assert resolve(value, boards, None) == set()

    with pytest.raises(ValueError) as error:
        resolve(value, boards, ["transfer", "transfer_08", "slide/0"])
    message = str(error.value)
    # Every offender at once, and what each namespace actually holds.
    assert "nothing to invalidate for: transfer, transfer_08" in message
    assert "Shots: slide" in message and "clips with staged states: transfer_09" in message
    assert "states: slide/0, slide/1" in message and "Nothing was edited" in message


def test_an_edit_that_names_an_unknown_clip_leaves_the_project_untouched(tmp_path):
    """The same finding, end to end: the code never reaches the worker."""
    svc = service(tmp_path)
    _storyboardable(svc)
    called = []
    worker = svc.worker
    svc.worker = lambda *a, **k: (called.append(a), worker(*a, **k))[1]
    with pytest.raises(ValueError, match="nothing to invalidate for: transfer"):
        svc.execute("blender", {"code": "raise SystemExit", "stage": "storyboard",
                                "invalidate_shots": ["transfer"],
                                "_version": svc.revision_token()},
                    svc.work, tmp_path / "edit", threading.Event(), 0)
    assert called == []
    svc.jobs.close()


def test_recording_delivery_metadata_neither_takes_the_lock_nor_costs_the_cache(tmp_path):
    """`plan()` publishes, but the metadata update the prompt asks for,
    `delivery.min_pixels`, must not take the project edit lock, rotate the
    version and throw away a whole-timeline check. The plan *document* still
    publishes; a call that declares only how to deliver does not."""
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    workspace, calls = _checkable(svc, tmp_path)
    write_json(workspace / "plan.json", json.loads((svc.work / "plan.json").read_text()))
    cancel = threading.Event()
    assert not svc.check(workspace, tmp_path / "a", cancel)["validation_reused"]

    # The metadata write does not move the key the check is cached under.
    before = svc.check_cache_key(workspace)
    document = json.loads((workspace / "plan.json").read_text())
    write_json(workspace / "plan.json",
               {**document, "delivery": {"min_pixels": 2 * 1024 * 1024, "reason": "measured"},
                "schedule": {"render_reservation_min": 40}})
    assert svc.check_cache_key(workspace) == before
    assert svc.check(workspace, tmp_path / "b", cancel)["validation_reused"] and len(calls) == 1
    # Changing what the document stages still does; its prose does not.
    write_json(workspace / "plan.json", {**document, "intent": "something else"})
    assert svc.check_cache_key(workspace) == before
    shots = [{**shot, "intent": "a different shot"} for shot in document["shots"]]
    write_json(workspace / "plan.json", {**document, "shots": shots})
    assert svc.check_cache_key(workspace) != before

    # And it is not a publishing op: no lock, no version rotation.
    assert not svc.plan_metadata_only("plan", {"plan": plan()})
    assert svc.plan_metadata_only("plan", {"plan": {"delivery": {"min_pixels": 64}}})
    version = svc.version()
    svc.edit_pending = True
    try:
        reply = svc.submit("plan", plan={"delivery": {"min_pixels": 64 * 64, "reason": "tiny"}})
    finally:
        svc.edit_pending = False
    assert reply["ok"] and reply["updated"] == ["delivery"]
    assert json.loads((svc.work / "plan.json").read_text())["delivery"]["min_pixels"] == 64 * 64
    assert svc.version() == version, "a metadata write is not a new project revision"
    svc.jobs.close()


def test_the_bundle_ships_the_current_author_sources_not_the_last_edit_s(tmp_path):
    """`animation_src/` reaches the workspace only inside `blender()`, and
    `finish()` packages the workspace, so without a re-sync a bundle could
    ship a superseded `compile_result.json` and files that post-date the last
    edit. The re-sync goes into the bundle, never back into the workspace,
    whose hash the delivery receipt was written against."""
    svc = service(tmp_path)
    sources = svc.root / "animation_src"
    sources.mkdir(parents=True)
    (sources / "notes.md").write_text("first")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "plan.json").write_text("{}")

    first = svc.bundle_sources(bundle)
    (sources / "notes.md").write_text("second, written after the last blender() call")
    second = svc.bundle_sources(bundle)
    assert first != second
    assert (bundle / "notes.md").read_text().startswith("second")
    assert (bundle / "animation_src/notes.md").read_text().startswith("second")
    assert second == tree_hash(sources), "the manifest states which source shipped"
    svc.jobs.close()


def test_an_interactive_op_is_pollable_and_cancellable_by_the_id_it_returns(tmp_path):
    """Wiring `register_inline`. An interactive op mints a job directory and
    publishes its name, so `job_status` and `job_cancel` must know that id, or
    the only way to stop one render is killing the resident Blender."""
    svc = service(tmp_path)
    reply = svc.submit("plan", plan=plan())
    assert reply["ok"] and reply["status"] == "completed" and reply["job_id"]

    # The id still answers after the call returned, as a batch id does.
    record = svc.jobs.status(reply["job_id"])
    assert record["status"] == "completed" and record["inline"]
    assert record["result"]["ok"]

    # It published, so canceling it is refused rather than flagged: stopping
    # mid-swap is what leaves the workspace renamed aside.
    with pytest.raises(ValueError, match="begun publishing"):
        svc.jobs.cancel(reply["job_id"])
    svc.jobs.close()


def test_a_render_prices_the_whole_film_and_says_what_the_price_rests_on(tmp_path, monkeypatch):
    """The prompt says to time a frame and multiply by the frame count, so the
    harness does the multiplication and reports what the film will cost."""
    from dycraft.animation import delivery

    svc = service(tmp_path)
    assert delivery.render_budget(svc, drawing_s=0, frames_drawn=0) is None
    svc.timeline_hint = {"n_frames": 1728}
    budget = delivery.render_budget(svc, drawing_s=82.9, frames_drawn=10)
    assert budget["measured_s_per_frame"] == 8.29 and budget["frames_timed"] == 10
    assert budget["estimated_full_film_seconds"] == round(8.29 * 1728)
    assert "start-up included" in budget["basis"]
    assert "run" not in budget, "a run with no advisory budget invents no denominator"

    # A draft says it is a lower bound on delivery, and how to price delivery.
    assert budget["measures"] == "draft" and "delivery=True" in budget["delivery_note"]
    assert "delivery_note" not in delivery.render_budget(svc, 82.9, 10, delivery=True)

    # One or two frames are mostly warm-up: the rate is reported, the film is not
    # priced from it, since such short probes vary by an order of magnitude.
    few = delivery.render_budget(svc, drawing_s=16.58, frames_drawn=2)
    assert few["measured_s_per_frame"] == 8.29
    assert few["estimated_full_film_seconds"] is None and "at least 5" in few["note"]

    # The render loop's own rate, when the worker left only its mean, excludes
    # start-up but still carries the first frame's scene sync, and says so.
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    write_json(rendered / "progress.json", {"phase": "render", "frame": 9, "total": 10,
                                            "seconds_per_step_mean": 2.0})
    loop = delivery.render_budget(svc, drawing_s=60.0, frames_drawn=10, rendered=rendered)
    assert loop["measured_s_per_frame"] == 2.0 and "start-up excluded" in loop["basis"]
    assert "first frame's scene sync included" in loop["basis"]
    assert loop["estimated_full_film_seconds"] == 2 * 1728

    # With the rate from the second frame on, the film is priced from that, and
    # the first frame's time is reported beside it rather than spread over the
    # film: a recorded price that averaged a 6.5 s first frame into five frames
    # of 1.96 s overstated the film 1.53x.
    write_json(rendered / "progress.json", {"phase": "render", "frame": 4, "total": 5,
                                            "seconds_per_step_mean": 2.9,
                                            "seconds_per_step_after_first": 1.96,
                                            "first_step_s": 6.5})
    steady = delivery.render_budget(svc, drawing_s=20.0, frames_drawn=5, rendered=rendered)
    assert steady["measured_s_per_frame"] == 1.96
    assert steady["first_frame_s"] == 6.5
    assert "second frame on" in steady["basis"]
    assert steady["estimated_full_film_seconds"] == round(1.96 * 1728)

    # An unchecked project has no frame count to multiply by, and says so
    # rather than guessing one.
    svc.timeline_hint = None
    assert delivery.render_budget(svc, 40.0, 10)["estimated_full_film_seconds"] is None

    monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", "5")
    # The run first started days ago; this daemon -- a resume, handed the 300
    # minutes that are left -- started now. Counted from the run's start, a
    # resumed run would read a remaining time hours below zero.
    write_json(svc.root / "meta.json", {"started_at": "2026-09-20T00:00:00+00:00"})
    run = svc.wall_budget()
    assert run["advisory_hours"] == 5.0
    assert run["elapsed_s"] + run["remaining_s"] == 300 * 60
    assert run["remaining_s"] > 299 * 60
    svc.jobs.close()


def test_the_staged_source_reference_is_this_layer_s_own(tmp_path):
    """Its own source, not sections cut from the scene layer's prompt, whose
    probe helpers, paths, and renderers this layer does not have; the asset
    section only when there is something to load -- the asset library linked
    into the run, or an asset the launcher preloaded."""
    import re
    plain = service(tmp_path / "plain")
    plain.jobs.close()
    reference = (plain.root / "source_reference.md").read_text()
    assert "## Asset library" not in reference
    assert "## File skeleton" in reference and "## Coordinate convention" in reference
    for absent in ("anchor_pose", "part_material", "poses=", "Open3D", "room_gap"):
        assert absent not in reference, absent
    preloaded = tmp_path / "preloaded_run"
    (preloaded / "design" / "preloaded").mkdir(parents=True)
    (preloaded / "design" / "preloaded" / "manifest.json").write_text("{}")
    with_asset = service(preloaded)
    with_asset.jobs.close()
    staged = (with_asset.root / "source_reference.md").read_text()
    assert "## Asset library" in staged
    assert len(re.findall(r"(?m)^## ", staged)) == len(re.findall(r"(?m)^## ", reference)) + 1
    library = tmp_path / "library_run"
    (library / "references" / "assets").mkdir(parents=True)
    (library / "references" / "assets" / "INDEX.md").write_text("# Asset library\n")
    with_library = service(library)
    with_library.jobs.close()
    assert "## Asset library" in (with_library.root / "source_reference.md").read_text()


def test_the_object_layer_advisory_is_not_added_here(tmp_path, monkeypatch):
    """The `budget` block is this layer's one clock. The object and scene
    layers' advisory points at a deviation tool this layer does not have, so
    the film agent's session leaves its replies as they are, however long the
    run has been going."""
    import time

    from dycraft.agent.agent_session import AgentSession
    from dycraft.agent.layers.film import FilmSession

    assert FilmSession.annotate_payload is AgentSession.annotate_payload
    monkeypatch.setenv("DYCRAFT_ADVISORY_HOURS", repr(1 / 60))
    session = FilmSession(tmp_path)
    try:
        # Through the session's own op wrapper, which is where a layer's
        # annotation would be folded into a reply.
        session.op("noop")(lambda: {"ok": True})
        two_hours_later = time.monotonic() + 7200.0
        with monkeypatch.context() as clock:
            clock.setattr(time, "monotonic", lambda: two_hours_later)
            reply = session.handlers["noop"]()
        assert reply == {"ok": True}
    finally:
        session.blender_animation.jobs.close()


def test_a_state_the_gate_still_accepts_is_not_drawn_again(tmp_path):
    """Retention asks the gate's own question, plus the staged pose, which the
    gate cannot see. Comparing the newly drawn PNG with the old one would pay
    the render before the comparison could skip it, and a whole-scene image
    digest would let any change anywhere invalidate every panel."""
    from dycraft.tests._animation_fixtures import state_reviews

    svc = service(tmp_path)
    colors, requests = _storyboardable(svc)
    poses = {"slide/0": 0.0, "slide/1": 0.0}

    # The fixture worker evaluates; teach it a per-state pose it can move.
    def evaluated_at(check, frame):
        matrix = [[1, 0, 0, poses[f"slide/{frame}"]],
                  [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        return {"objects": [{"name": "part/block/body", "hidden": False, "matrix": matrix}],
                "transforms": {"part/block/body": matrix}}
    svc.evaluated_at = evaluated_at

    first = _storyboard(svc, tmp_path, "a")
    assert first["needs_review"] == ["slide/0", "slide/1"] and first["retained"] == []
    assert _accept(svc, tmp_path, "b", state_reviews("The block rests on the floor."))["ok"]

    drawn_before = len([r for r in requests if r["op"] == "render"])
    again = _storyboard(svc, tmp_path, "c")
    assert again["ok"] and sorted(again["retained"]) == ["slide/0", "slide/1"]
    assert again["needs_review"] == []
    assert len([r for r in requests if r["op"] == "render"]) == drawn_before, (
        "a state the gate still accepts must not be re-drawn")
    assert again["states"]["slide/0"]["redrawn"] is False
    assert "Kept without redrawing" in again["next"]

    # Re-posing one state is outside the gate, and is exactly what the pose
    # check is for: that state is drawn again -- and, its picture having
    # changed with the pose, asks to be re-read. The other is untouched.
    poses["slide/1"], colors["slide/1"] = 0.5, (200, 0, 0)
    after = _storyboard(svc, tmp_path, "d")
    assert after["retained"] == ["slide/0"] and after["needs_review"] == ["slide/1"]
    assert len([r for r in requests if r["op"] == "render"]) == drawn_before + 1
    last_render = [r for r in requests if r["op"] == "render"][-1]
    assert [s["id"] for s in last_render["states"]] == ["slide/1"], (
        "only the state that moved is drawn")
    svc.jobs.close()


def test_completion_records_what_it_decided_even_when_it_refuses(tmp_path):
    """The harness writes `finish_result.json` on every path, a refusal
    included, so the run directory can answer "did completion fail or never
    run?"."""
    svc = service(tmp_path)
    assert call(svc, "plan", plan=plan())["ok"]
    refused = call(svc, "finish")
    assert not refused["ok"]
    record = json.loads((svc.root / "finish_result.json").read_text())
    assert record["ok"] is False and record["finished_at"] and record["version"]
    assert "build()" in record["reason"] or record["reason"]
    assert record["error_type"]
    svc.jobs.close()


def _recording_service(tmp_path):
    """A service whose session records traces, for the staging posture tests."""
    from types import SimpleNamespace
    (tmp_path / "native/jobs").mkdir(parents=True)
    svc = service(tmp_path)
    traces = []
    svc.session = SimpleNamespace(ctx=svc.session.ctx,
                                  trace=lambda name, **kw: traces.append((name, kw)),
                                  current_op_step=lambda: 0,
                                  finalize_meta=lambda *a: None)
    return svc, traces


def test_finish_stages_the_viewer_scene_from_the_committed_bundle(tmp_path, monkeypatch):
    """The whole point of the hook: a committed run is viewable without a
    second manual step.

    Only the wiring is asserted here — that the committed bundle and the run
    root are what get passed, and that the outcome reaches the caller.
    `materialize_scene` has its own tests against a real design_state
    in `tests/test_viewer.py`; re-staging assets here would test that twice
    and this once.
    """
    import dycraft.agent.layers.scene_staging as ac

    svc, traces = _recording_service(tmp_path)
    seen = {}
    monkeypatch.setattr(ac, "materialize_scene",
                        lambda bundle, run_dir, **kw: seen.update(
                            bundle=Path(bundle), run_dir=Path(run_dir))
                        or {"scene_ref": "abc", "n_entries": 3,
                            "source": "s", "skipped": False})

    result = svc.stage_viewer_scene(tmp_path / "bundle")
    assert seen == {"bundle": tmp_path / "bundle", "run_dir": tmp_path}
    assert result == {"ok": True, "scene_ref": "abc", "n_entries": 3,
                      "source": "s", "skipped": False}
    assert [name for name, _ in traces] == ["stage_viewer_scene"]


@pytest.mark.parametrize("failure", [SystemExit("bad catalog"), RuntimeError("disk full")])
def test_a_failed_viewer_staging_never_uncommits_the_animation(tmp_path, monkeypatch,
                                                               failure):
    """Staging runs after `ir.json` is written and `finalize_meta("committed")`
    has been called, so it must not be able to raise.

    `SystemExit` is the case worth parametrizing: `materialize_catalog` and
    `validate_scene_ir` raise it rather than an `Exception`, so a bare
    `except Exception` here would let a bad catalog escape a finished run and
    turn a committed animation into a failed op.
    """
    import dycraft.agent.layers.scene_staging as ac

    svc, traces = _recording_service(tmp_path)

    def boom(*a, **k):
        raise failure
    monkeypatch.setattr(ac, "materialize_scene", boom)

    result = svc.stage_viewer_scene(tmp_path / "bundle")
    assert result["ok"] is False
    assert type(failure).__name__ in result["error"]
    # The operator is told how to finish the job, and the tool that does it is
    # the one that already exists.
    assert result["retry"].startswith(
        "python -m dycraft.animation.stage_scene")
    assert str(tmp_path) in result["retry"]
    assert [name for name, _ in traces] == ["stage_viewer_scene_failed"]


def test_the_client_sends_ranges_sets_and_numpy_values_as_json(tmp_path):
    """`frames=range(0, n, 4)` raised `TypeError` in the client, before the
    call left the agent's process."""
    import numpy as np
    from dycraft.agent import _HARNESS_CORE_SRC
    namespace = {"__file__": str(tmp_path / "harness.py")}
    exec(_HARNESS_CORE_SRC.read_text().split("# Operations:")[0], namespace)
    sent = json.loads(json.dumps({"frames": range(0, 9, 4), "states": {"b", "a"},
                                  "frame": np.int64(5), "eye": np.array([1.0, 2.0, 3.0])},
                                 default=namespace["_jsonable"]))
    assert sent == {"frames": [0, 4, 8], "states": ["a", "b"], "frame": 5, "eye": [1.0, 2.0, 3.0]}


def test_the_client_waits_inside_the_agent_s_shell_timeout(tmp_path, monkeypatch):
    """A Claude shell kills a command at two minutes by default, and a call
    killed while it waits for its job looks like a dead one."""
    from dycraft.agent import _HARNESS_CORE_SRC
    core = _HARNESS_CORE_SRC.read_text().split("# Operations:")[0]
    namespace = {"__file__": str(tmp_path / "harness.py")}
    exec(core, namespace)
    assert namespace["WAIT_S"] == 540.0
    monkeypatch.setenv("DYCRAFT_HARNESS_WAIT_S", "100")
    namespace = {"__file__": str(tmp_path / "harness.py")}
    exec(core, namespace)
    assert namespace["WAIT_S"] == 100.0


def test_pose_digest_sees_a_skinned_character_reposed():
    """A skinned character moves its bones, not its object frame or a joint.

    Evaluation records carry each armature's evaluated `bone_rotations`; a
    digest that ignored them reported a re-posed character as unchanged, so
    storyboard drift on a character went undetected.
    """
    from dycraft.animation.planning import (POSE_QUANTUM, pose_digest,
                                                    poses_comparable)
    eye = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    still = {"transforms": {"rig_dog": eye, "Cam": eye}, "joint_positions": {},
             "bone_rotations": {"rig_dog": [[1.0, 0.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 0.0]]}}
    bent = {**still, "bone_rotations": {"rig_dog": [[1.0, 0.0, 0.0, 0.0],
                                                    [0.92388, 0.38268, 0.0, 0.0]]}}
    jitter = {**still, "bone_rotations": {"rig_dog": [
        [1.0, 0.0, 0.0, 0.0], [1.0, POSE_QUANTUM / 10, 0.0, 0.0]]}}
    view = ["Cam"]
    assert pose_digest(bent, viewpoints=view) != pose_digest(still, viewpoints=view)
    assert pose_digest(jitter, viewpoints=view) == pose_digest(still, viewpoints=view)
    # A rig-free record digests exactly as before, under the same rule, so
    # boards approved before bones were counted stay comparable.
    rigid = {k: v for k, v in still.items() if k != "bone_rotations"}
    assert pose_digest(rigid, viewpoints=view).startswith("pose2:")
    # A digest that counts bones is a different rule from one that did not,
    # so an old board for a character reads as "no observation", not as
    # unchanged.
    assert not poses_comparable(pose_digest(still, viewpoints=view),
                                pose_digest(rigid, viewpoints=view))
