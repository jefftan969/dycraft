"""`meta.json`, the run's machine-readable record."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import dycraft.agent.agent_session as S

REPO = Path(__file__).resolve().parents[1]


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


_DRIVER = r'''
import json, shutil, sys
from pathlib import Path
tree, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
import dycraft
assert Path(dycraft.__file__).resolve().parent == tree.resolve(), dycraft.__file__
from dycraft.agent.layers import server as srv
from dycraft.agent.layers.film import FilmSession
session = FilmSession(run_dir)
shutil.rmtree(tree)                     # what the worktree's EXIT trap does
session.finalize_meta("interrupted")
print("finalized")
'''


def test_the_record_survives_its_worktree_being_deleted_first(tmp_path):
    """At shutdown, the worktree a daemon was imported from can be deleted by
    the launcher's trap about a second before `finalize_meta` runs. Any import
    made on that path then raises `ModuleNotFoundError`, and the run would lose
    its whole record."""
    tree = tmp_path / "pkgroot" / "dycraft"
    tree.parent.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("agent", "animation", "sdk", "verify", "cli", "render"):
        shutil.copytree(REPO / name, tree / name, ignore=ignore)
    for path in REPO.glob("*.py"):
        shutil.copy2(path, tree / path.name)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    S.write_run_meta(run_dir, layer="animation", runtime="codex_cli",
                     prompt="p", run_id="r", model=None)
    records = [{"ev": "op_result", "op": "compile", "ok": True},
               {"ev": "job_result", "status": "failed",
                "result": {"ok": False, "error": "Blender worker exited -9: "}}]
    (run_dir / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    env = {**os.environ, "PYTHONPATH": str(tree.parent), "HOME": str(tmp_path),
           "CODEX_HOME": str(tmp_path / ".codex")}
    done = subprocess.run([sys.executable, "-c", _DRIVER, str(tree), str(run_dir)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "finalized" in done.stdout, done.stderr[-3000:]
    meta = _meta(run_dir)
    assert meta["outcome"] == "interrupted" and meta["finished_at"]
    assert meta["ops"] == 1 and meta["jobs"] == 1 and meta["jobs_killed"] == 1
    assert meta["best_film"] is None and meta["host"]


def test_finished_at_moves_on_every_finalize_and_outcomes_rank(tmp_path, monkeypatch):
    """A resumed run that ends again is finished later; a lower outcome never
    displaces a higher one, and a clean commit outranks one with exceptions."""
    clock = iter(f"2026-10-03T00:00:0{i}+00:00" for i in range(10))
    monkeypatch.setattr(S, "utcnow_iso", lambda: next(clock))
    meta = S.TraceMeta(tmp_path)
    meta.write_initial(prompt="p")
    meta.finalize("interrupted")
    first = _meta(tmp_path)["finished_at"]
    meta.finalize("interrupted")
    assert _meta(tmp_path)["finished_at"] > first
    meta.finalize("delivered_with_exceptions")
    meta.finalize("interrupted")
    assert _meta(tmp_path)["outcome"] == "delivered_with_exceptions"
    meta.finalize("committed")
    meta.finalize("delivered_with_exceptions")
    assert _meta(tmp_path)["outcome"] == "committed"


def test_the_shutdown_outcome_is_the_one_ir_json_committed(tmp_path):
    from dycraft.agent.server import _committed_outcome
    assert _committed_outcome(tmp_path) == "committed"
    (tmp_path / "ir.json").write_text(json.dumps({"outcome": "delivered_with_exceptions"}))
    assert _committed_outcome(tmp_path) == "delivered_with_exceptions"
    (tmp_path / "ir.json").write_text(json.dumps({"id": "abc"}))
    assert _committed_outcome(tmp_path) == "committed"


def test_a_raising_best_film_still_leaves_a_record(tmp_path, monkeypatch):
    from dycraft.agent.layers import film

    def broken(run_dir):
        raise KeyError("frames")

    monkeypatch.setattr(film, "best_film", broken)
    ctx = film.FilmContext(run_dir=tmp_path, step=3)
    meta = S.TraceMeta(tmp_path)
    meta.write_initial(prompt="p")
    meta.finalize("interrupted", **S._meta_extras(ctx))
    assert _meta(tmp_path)["animation_authoring"] == "blender"
    assert _meta(tmp_path)["outcome"] == "interrupted"
    # A committed run names its film in `ir.json`; nothing stale stays here.
    monkeypatch.setattr(film, "best_film", lambda run_dir: {"kind": "draft"})
    assert S._meta_extras(ctx)["best_film"] == {"kind": "draft"}
    ctx.committed = True
    assert S._meta_extras(ctx)["best_film"] is None


def test_jobs_killed_counts_signalled_workers_and_never_a_cancellation(tmp_path):
    def job(status, error, **extra):
        return {"ev": "job_result", "status": status, "result": {"ok": False, "error": error},
                **extra}

    records = [
        job("failed", "Blender worker exited -9: Killed"),
        job("failed", "Blender worker exited -15: "),
        job("failed", "resident Blender closed its reply channel (exit -9)\n...",
            interactive=True),
        job("failed", "Blender worker exited 1: Traceback"),
        job("failed", "Blender worker exited -11: "),
        job("cancelled", "Blender worker exited -15: "),
    ]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert S._run_shape(tmp_path, None)["jobs_killed"] == 3


def test_a_raised_refusal_is_a_rejection(tmp_path):
    """U44. The op wrapper stamped every raise `error`, which no text could
    then reclassify: "wait for pending edits" counted as a failure."""
    session = S.AgentSession(SimpleNamespace(run_dir=tmp_path, step=0))

    @session.op("dispatch_motion")
    def _refuse() -> dict:
        raise ValueError("wait for pending edits; contributions cannot start after completion")

    with pytest.raises(ValueError):
        session.handlers["dispatch_motion"]()
    record = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert S._failure_class(record) == "rejection"
    for text in ("repair candidate is stale; rebuild from the current project",
                 "Wait for or cancel outstanding contributions before finish().",
                 "project changed before completion", "job cancelled",
                 "project changed while this candidate was running",
                 "Stage, check, render and review these states before animation/delivery: a",
                 "these states need a passing, accepted storyboard before an animation edit: a",
                 "The project is committed; nothing changes after finish().",
                 "Accept these states before animating this span (they fall inside it): a",
                 "every declared state needs reviewed storyboard evidence"):
        assert S._failure_class({"error": text}) == "rejection", text
    # Naming a storyboard does not make a caller's mistake a refusal.
    for text in ("stage must be 'storyboard' or 'animation'", "unknown storyboard shot"):
        assert S._failure_class({"error": text}) == "error", text


def test_a_launch_failure_never_downgrades_a_commit(tmp_path):
    """A relaunched run dir whose daemon then fails to bind keeps its outcome."""
    for outcome, after in (("delivered_with_exceptions", "delivered_with_exceptions"),
                           ("committed", "committed"), ("interrupted", "daemon_failed")):
        (tmp_path / "meta.json").write_text(json.dumps({"outcome": outcome}))
        S.record_launch_failure(tmp_path, reason="harness_daemon_never_bound", detail="d")
        assert _meta(tmp_path)["outcome"] == after, outcome


def _rollout(home: Path, name: str, head: dict) -> None:
    path = home / "sessions/2026/10/02" / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "session_meta", "payload": head}) + "\n")


def test_only_the_approval_reviewer_counts_as_the_reviewer(tmp_path, monkeypatch):
    """A sub-agent the run spawns is a sub-thread too, and its spend is not the
    reviewer's."""
    from dycraft.agent import transcript_capture as tc

    home = tmp_path / ".codex"
    monkeypatch.setattr(tc, "_codex_homes", lambda run_dir: [home])
    _rollout(home, "driver", {"id": "drive"})
    _rollout(home, "reviewer", {"id": "r1", "parent_thread_id": "drive",
                                "thread_source": "guardian_review"})
    _rollout(home, "helper", {"id": "s1", "parent_thread_id": "drive",
                              "thread_source": "subagent"})
    found = tc._reviewer_rollouts(tmp_path, "drive")
    assert [p.name for p in found] == ["rollout-reviewer.jsonl"]


def _spent(home: Path, name: str, head: dict, *responses: tuple[str, int, int]) -> None:
    """A rollout whose responses each spent (`gross` input, `output`) tokens."""
    _rollout(home, name, head)
    path = home / "sessions/2026/10/02" / f"rollout-{name}.jsonl"
    with open(path, "a") as fh:
        for response_id, gross, out in responses:
            fh.write(json.dumps({"type": "token_usage_record", "payload": {
                "response_id": response_id,
                "usage": {"input_tokens": gross, "cached_input_tokens": 0,
                          "output_tokens": out}}}) + "\n")


def _spawning_run(tmp_path, monkeypatch):
    """A codex run `drive` that spawned `s1`, which spawned `s2`; `drive` and
    `s1` each have an approval reviewer; `other` belongs to another run."""
    from dycraft.agent import transcript_capture as tc

    home = tmp_path / ".codex"
    monkeypatch.setattr(tc, "_codex_homes", lambda run_dir: [home])
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"codex_session_id": "drive"}))
    _spent(home, "driver", {"id": "drive"}, ("d1", 1000, 100))
    _spent(home, "reviewer", {"id": "r1", "parent_thread_id": "drive",
                              "thread_source": "guardian_review"}, ("r", 500, 5))
    _spent(home, "helper", {"id": "s1", "parent_thread_id": "drive",
                            "thread_source": "subagent"}, ("a", 300, 30), ("b", 200, 20))
    _spent(home, "grandhelper", {"id": "s2", "parent_thread_id": "s1",
                                 "thread_source": "subagent"}, ("c", 100, 10))
    _spent(home, "helper-reviewer", {"id": "r2", "parent_thread_id": "s1",
                                     "thread_source": "guardian_review"}, ("e", 40, 4))
    _spent(home, "other", {"id": "o1", "parent_thread_id": "elsewhere",
                           "thread_source": "subagent"}, ("f", 9000, 900))
    return tc, run


def test_subagent_usage_sums_every_thread_the_run_spawned(tmp_path, monkeypatch):
    """Sub-agents at any depth count, and so does a sub-agent's own reviewer;
    the run's own approval reviewer is accounted elsewhere, and another run's
    sub-agent is not this run's."""
    tc, run = _spawning_run(tmp_path, monkeypatch)
    block = tc.subagent_usage(run)
    assert block["threads"] == 3
    assert sorted(block["sessions"]) == ["r2", "s1", "s2"]
    assert block["gross_input"] == 300 + 200 + 100 + 40
    assert block["output"] == 30 + 20 + 10 + 4
    assert block["total"] == block["gross_input"] + block["output"]
    assert block["turns"] == 4


def test_a_run_that_spawned_nothing_records_no_subagents(tmp_path, monkeypatch):
    from dycraft.agent import transcript_capture as tc

    home = tmp_path / ".codex"
    monkeypatch.setattr(tc, "_codex_homes", lambda run_dir: [home])
    _spent(home, "driver", {"id": "drive"}, ("d1", 1000, 100))
    (tmp_path / "meta.json").write_text(json.dumps({"codex_session_id": "drive"}))
    assert tc.subagent_usage(tmp_path) is None


def test_meta_usage_records_subagents_beside_the_runs_own(tmp_path, monkeypatch):
    """`usage` stays the staged transcript's own figures, so it keeps agreeing
    with the viewer's trajectory; the sub-agents sit beside it, and
    `total_with_subagents` is what the run cost in all."""
    tc, run = _spawning_run(tmp_path, monkeypatch)
    shutil.copy(tmp_path / ".codex/sessions/2026/10/02/rollout-driver.jsonl",
                run / "agent_transcript.jsonl")
    (run / "trace.jsonl").write_text("")
    usage = S._run_shape(run, SimpleNamespace())["usage"]
    assert usage["total"] == 1100
    assert usage["subagents"]["total"] == 704
    assert usage["total_with_subagents"] == 1804
