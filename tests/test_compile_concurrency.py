"""One compile at a time per session, and artifacts filed under the right step.

The daemon services handlers on a thread pool on purpose, so a long `compile`
cannot head-of-line block `probe` / `read_file`. Nothing stopped a *second*
compile starting while the first is still running would be a problem: an
agent whose foreground call is killed by its own tool timeout has every reason
to re-issue it while the daemon keeps working. Two compiles of the same source
would each run the full verifier suite, and if both resolved
`compile_reports/<step>.json` *after* the counter moved, the first would write
its report over the second's filename and lose the record of what the agent
actually read.

Two properties:
  * compile bodies for one session never overlap;
  * a compile files its artifacts under the step it started with, whatever the
    counter does while it runs.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from dycraft.agent.layers.layer import object_layer_config
from dycraft.agent.layers.session import LayerSession, SessionConfig


def _session(tmp_path):
    return LayerSession(SessionConfig(
        run_dir=tmp_path, layer=object_layer_config(), user_prompt=""))


class _Recorder:
    """Stands in for the compile body and reports what overlapped."""

    def __init__(self, hold: float = 0.15):
        self.hold = hold
        self.live = 0
        self.max_live = 0
        self.entries = []          # (artifact_step, step_at_entry)
        self._guard = threading.Lock()

    def __call__(self, session, *, freeze: bool) -> dict:
        with self._guard:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            self.entries.append(
                (session.ctx.artifact_step, session.ctx.step))
        time.sleep(self.hold)
        with self._guard:
            self.live -= 1
        return {"ok": True, "verifier_reports": []}


# --- the lock --------------------------------------------------------------


def test_two_compiles_on_one_session_never_overlap(tmp_path, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(LayerSession, "_compile_locked",
                        lambda self, *, freeze: rec(self, freeze=freeze))
    sess = _session(tmp_path)
    with sess:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: sess.handlers["compile"](), range(4)))
    assert rec.max_live == 1, (
        f"{rec.max_live} compile bodies ran at once; they must serialise")
    assert len(rec.entries) == 4


def test_finish_serialises_against_compile(tmp_path, monkeypatch):
    """`compile` and `finish` share one body, so the lock has to cover both —
    a `finish` racing a straggling `compile` is the commit-time version of the
    same overlap."""
    rec = _Recorder()
    monkeypatch.setattr(LayerSession, "_compile_locked",
                        lambda self, *, freeze: rec(self, freeze=freeze))
    sess = _session(tmp_path)
    with sess:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(sess.handlers["compile"])
            f2 = pool.submit(sess.handlers["finish"])
            f1.result(); f2.result()
    assert rec.max_live == 1


def test_independent_sessions_still_run_in_parallel(tmp_path):
    """The lock is per session. Orchestrator-parallel object agents each hold
    their own, and must not queue behind one another."""
    recs = [_Recorder(hold=0.3), _Recorder(hold=0.3)]
    sessions = [_session(tmp_path / f"run{i}") for i in range(2)]
    for s, r in zip(sessions, recs):
        s._compile_locked = (lambda *, freeze, _s=s, _r=r: _r(_s, freeze=freeze))
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda s: s.handlers["compile"](), sessions))
    elapsed = time.monotonic() - t0
    for s in sessions:
        s.__exit__(None, None, None)
    assert elapsed < 0.55, (
        f"independent sessions serialised ({elapsed:.2f}s for two 0.3s "
        f"compiles); the lock must not be shared")


# --- the frozen artifact step ---------------------------------------------


def test_a_running_compile_keeps_the_step_it_started_with(tmp_path, monkeypatch):
    """The `@op` wrapper ticks `ctx.step` before the body runs, so a queued
    second compile advances the counter while the first is still working.
    Reading it late is what made two compiles resolve the same filename."""
    rec = _Recorder(hold=0.2)
    monkeypatch.setattr(LayerSession, "_compile_locked",
                        lambda self, *, freeze: rec(self, freeze=freeze))
    sess = _session(tmp_path)
    with sess:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: sess.handlers["compile"](), range(2)))
    frozen = [e[0] for e in rec.entries]
    # Distinct, not ordered: which compile wins the lock is not which ticked
    # first, and it does not need to be. What matters is that each keeps its
    # own number so neither writes over the other's report.
    assert len(set(frozen)) == 2, f"both compiles filed under {frozen}"
    assert set(frozen) == {1, 2}, f"unexpected steps {frozen}"
    # And the counter had already moved past the earlier one by then, which is
    # exactly the condition that would collapse them onto one filename.
    assert max(e[1] for e in rec.entries) == 2


def test_the_frozen_step_is_what_the_report_path_uses(tmp_path):
    """`compile_artifact_step()` is the single reader; every compile-path
    artifact goes through it."""
    sess = _session(tmp_path)
    with sess:
        ctx = sess.ctx
        ctx.step = 7
        assert ctx.artifact_step is None
        assert ctx.compile_artifact_step() == 7      # outside a compile
        ctx.artifact_step = 3
        ctx.step = 9                                 # a later op ticks past us
        assert ctx.compile_artifact_step() == 3
        assert ctx.report_path(ctx.compile_artifact_step()).name == "3.json"


def test_the_freeze_is_cleared_even_when_the_compile_raises(tmp_path, monkeypatch):
    """A build that blows up must not leave the context pinned to a stale
    step — every later artifact would then be filed under it."""
    def _boom(self, *, freeze):
        raise RuntimeError("kernel died")
    monkeypatch.setattr(LayerSession, "_compile_locked", _boom)
    sess = _session(tmp_path)
    with sess:
        with pytest.raises(RuntimeError):
            sess.handlers["compile"]()
        assert sess.ctx.artifact_step is None
        # And the lock is released, so the session is still usable.
        assert not sess._compile_lock.locked()


def test_the_lock_is_not_shared_between_sessions(tmp_path):
    a, b = _session(tmp_path / "a"), _session(tmp_path / "b")
    assert a._compile_lock is not b._compile_lock
    a.__exit__(None, None, None); b.__exit__(None, None, None)


# --- the symptom on disk ---------------------------------------------------


def test_two_compiles_write_two_report_files(tmp_path, monkeypatch):
    """End to end: two compiles that resolved `compile_reports/<step>.json`
    after the counter moved would land on one name, replacing the report the
    agent had read with one it never saw."""
    from dycraft.agent.layers import session as sess_mod

    written = []

    def _fake_persist(ctx, *, is_finish):
        # Mirrors run_compile_and_persist's artifact write, including the
        # deliberate late read of the counter that the freeze now shields.
        time.sleep(0.15)
        path = ctx.report_path(ctx.compile_artifact_step())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ok": True, "verifier_reports": []}))
        written.append(path.name)
        return {"ok": True, "verifier_reports": []}

    monkeypatch.setattr(sess_mod, "run_compile_and_persist", _fake_persist)
    sess = _session(tmp_path)
    with sess:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: sess.handlers["compile"](), range(2)))

    assert sorted(written) == ["1.json", "2.json"], written
    on_disk = sorted(p.name for p in (tmp_path / "compile_reports").glob("*.json")
                     if p.name != "latest.json")
    assert on_disk == ["1.json", "2.json"], (
        f"one compile overwrote the other: {on_disk}")


# --- the step counter itself -----------------------------------------------


def test_concurrent_ticks_never_collide(tmp_path):
    """`ctx.step += 1` is a read-modify-write, and the daemon dispatches on a
    thread pool. Two ops landing on one step number would put two different
    actions' artifacts under the same name — the same failure as the compile
    overwrite, arriving by a different route."""
    sess = _session(tmp_path)
    with sess:
        seen, guard = [], threading.Lock()

        def _tick(_):
            v = sess.tick()
            with guard:
                seen.append(v)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_tick, range(200)))
    assert len(set(seen)) == 200, (
        f"{200 - len(set(seen))} step numbers were handed out twice")
    assert sess.ctx.step == 200


def test_each_thread_sees_its_own_op_step(tmp_path):
    """`current_op_step()` is per thread: a handler asks what *it* was given,
    not what the run has reached since."""
    sess = _session(tmp_path)
    with sess:
        assert sess.current_op_step() is None      # nothing ticked here yet
        mine = sess.tick()
        assert sess.current_op_step() == mine

        other = {}

        def _elsewhere():
            other["step"] = sess.tick()
            other["seen"] = sess.current_op_step()

        t = threading.Thread(target=_elsewhere)
        t.start(); t.join()

        assert other["seen"] == other["step"] != mine
        # The other thread moved the counter; ours still reports its own.
        assert sess.ctx.step == other["step"]
        assert sess.current_op_step() == mine


# --- the attempt ledger ----------------------------------------------------


def test_every_attempt_number_is_used_exactly_once(tmp_path, monkeypatch):
    """Every attempt number is used exactly once: a refusal records its own
    number rather than the previous call's, and a compile that raises still
    records the number it consumed, so the numbering has no gaps."""
    from dycraft.agent.layers import session as sess_mod

    calls = []

    def _persist(ctx, *, is_finish):
        # The real one numbers the attempt before anything can return or raise.
        ctx.n_compile_attempts += 1
        attempt = ctx.n_compile_attempts
        calls.append(is_finish)
        if len(calls) == 2:
            raise RuntimeError("kernel died")
        return {"ok": True, "verifier_reports": [], "attempt": attempt}

    monkeypatch.setattr(sess_mod, "run_compile_and_persist", _persist)
    sess = _session(tmp_path)
    with sess:
        sess.handlers["compile"]()
        with pytest.raises(RuntimeError):
            sess.handlers["compile"]()
        sess.handlers["compile"]()
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
               if json.loads(line)["ev"] == "compile_attempt"]
    assert [r["attempt"] for r in records] == [1, 2, 3]
    assert [r["ok"] for r in records] == [True, False, True]
    assert records[1]["build_error"] == "RuntimeError"
