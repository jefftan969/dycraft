"""The wall-clock bound the rewriters are told about, and what a miss costs.

A rewriter that is not told its wall-clock bound can spend it on one long
reasoning turn and be stopped before it calls `submit_result`. So the bound is
in the prompt (`deadline_note`, shared by both rewriters, with a reserve held
back for the submission itself), and a rewriter that returns nothing is
reported and the operator decides (`--on-rewrite-failure`) rather than the
launch failing outright and losing the entry already paid for. The bound
itself is wall clock out of the run's own budget.
"""
import io
import re
import sys
import time
from pathlib import Path

import pytest

from dycraft.agent import motion_rewrite as mr
from dycraft.agent import oneoff
from dycraft.agent import scene_rewrite as sr

#: A fixed epoch, so the rendered clock times are deterministic. Local time,
#: because `date` in the sub-agent's shell prints local time and the note is
#: written to be compared against it.
T0 = time.mktime((2026, 9, 22, 11, 0, 0, 0, 0, -1))


def _hhmmss(offset: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(T0 + offset))


def test_the_note_names_the_bound_the_hard_stop_and_a_soft_deadline():
    note = sr.deadline_note(900.0, started=T0)
    assert "15 minutes" in note
    assert _hhmmss(0) in note, "the start is not stated"
    assert _hhmmss(900.0) in note, "the hard stop is not stated"
    assert _hhmmss(675.0) in note, "the soft deadline is not stated"
    assert "`submit_result`" in note and "`date`" in note


def test_the_reserve_is_a_fraction_with_a_floor():
    """Long bounds get the fraction; short ones get the floor.

    225 s of 900 s is a little under the longest single reasoning turn seen in
    the run that motivated this, which is the reserve's whole job: room to
    notice the deadline and stop. On a short bound the fraction would be
    smaller than one turn, so the floor takes over.
    """
    assert _hhmmss(900.0 - 225.0) in sr.deadline_note(900.0, started=T0)
    assert _hhmmss(300.0 - 120.0) in sr.deadline_note(300.0, started=T0)
    # Degenerate rather than negative: a bound under the floor reserves all of
    # it, and the note still reads as an instruction to submit immediately.
    assert _hhmmss(0.0) in sr.deadline_note(60.0, started=T0)


def test_the_note_permits_a_shorter_entry():
    """The sentence that actually changes the outcome.

    Everything above this in the prompt asks for a thorough entry. An agent
    that knows it will be killed still needs to be told that a delivered short
    entry beats an undelivered better one, or it will keep refining.
    """
    note = sr.deadline_note(900.0, started=T0).lower()
    assert "stop" in note and "as it stands" in note
    assert "delivered is worth more" in note


@pytest.mark.parametrize("module", [sr, mr])
def test_both_rewriters_put_the_deadline_last_in_the_request(module, tmp_path,
                                                             monkeypatch):
    """The request on disk is what the sub-agent is handed; check that.

    `deadline_note` after `SUBMIT_INSTRUCTION` on purpose: it is the constraint
    on everything above it, including the instruction to submit.
    """
    captured = {}

    def fake_run_oneoff(**kw):
        captured.update(kw)
        return {"ok": True, "text": "an entry", "outcome": "completed"}

    monkeypatch.setattr(oneoff, "run_oneoff", fake_run_oneoff)
    kwargs = dict(backend="codex", run_dir=tmp_path, timeout_s=900.0)
    if module is mr:
        kwargs["scene"] = "a place"
    assert module.rewrite("a brief", **kwargs) == "an entry"

    text = Path(captured["prompt"]).read_text()
    assert captured["timeout_s"] == 900.0
    assert "15 minutes of wall clock" in text
    assert text.index(sr.SUBMIT_INSTRUCTION.strip()[:40]) < text.index("15 minutes")
    assert text.rstrip().endswith("worth more to the author than a better one "
                                  "that is not.")


@pytest.mark.parametrize("module", [sr, mr])
def test_the_stated_deadline_follows_the_timeout_actually_used(module, tmp_path,
                                                              monkeypatch):
    """A caller that overrides `timeout_s` must not get the default's clock.

    The note is generated from the same value passed to `run_oneoff`, so the
    two cannot disagree — which is the only way the sub-agent's `date` check
    means anything.
    """
    seen = {}
    monkeypatch.setattr(oneoff, "run_oneoff",
                        lambda **kw: seen.update(kw) or {"ok": True, "text": "e"})
    kwargs = dict(backend="codex", run_dir=tmp_path, timeout_s=600.0)
    if module is mr:
        kwargs["scene"] = None
    module.rewrite("a brief", **kwargs)
    text = Path(seen["prompt"]).read_text()
    assert "10 minutes of wall clock" in text
    assert seen["timeout_s"] == 600.0


@pytest.mark.parametrize("module", [sr, mr])
def test_a_rewriter_that_returns_nothing_still_raises(module, tmp_path,
                                                      monkeypatch):
    """The layering is unchanged: `rewrite()` raises, the caller decides.

    A timeout is reported in the message, because that is the field that
    distinguishes running out of time from a crash.
    """
    monkeypatch.setattr(oneoff, "run_oneoff", lambda **kw: {
        "ok": False, "text": "", "outcome": "timeout", "timed_out": True,
        "n_op_calls": 0, "returncode": -15, "wall_time_s": 900.0,
        "sub_run": str(tmp_path / "sub_runs" / module.SUB_RUN)})
    kwargs = dict(backend="codex", run_dir=tmp_path)
    if module is mr:
        kwargs["scene"] = None
    with pytest.raises(RuntimeError, match=r"did not return an entry"):
        module.rewrite("a brief", **kwargs)


_KINDS = [("scene.txt", "scene", "--scene-brief", "--no-scene-brief"),
          ("motion.txt", "motion", "--motion-brief", "--no-motion-brief")]


class _Stdin(io.StringIO):
    def __init__(self, text, tty):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize("staged, kind, flag, off", _KINDS)
def test_continue_policy_warns_and_returns_rather_than_exiting(
        staged, kind, flag, off, capsys, tmp_path):
    """`--on-rewrite-failure continue`: run on without the entry, on request.

    Losing a long run because an entry could not be generated in time is a
    trade to accept explicitly; `continue` accepts it, and it is not the
    default.
    """
    from dycraft.cli.cli import _rewrite_failed

    sub_run = tmp_path / "sub_runs" / f"{kind}_rewrite"
    assert _rewrite_failed(staged, kind, RuntimeError("timed out"), sub_run,
                           time.time() - 900.0, flag, off, "continue") is None

    err = capsys.readouterr().err
    assert "WARNING" in err and staged in err
    assert "RuntimeError: timed out" in err
    assert re.search(r"failed after 9\d\ds", err), "the cost is not reported"
    assert str(sub_run) in err, "the evidence is not pointed at"
    assert "degraded run, not a failed one" in err


@pytest.mark.parametrize("staged, kind, flag, off", _KINDS)
@pytest.mark.parametrize("answer, tty, continues", [
    ("y\n", True, True), ("yes\n", True, True),
    ("\n", True, False), ("n\n", True, False),
    ("y\n", False, False),          # no terminal: nobody was asked, so it stops
])
def test_ask_policy_continues_only_on_an_explicit_yes(
        staged, kind, flag, off, answer, tty, continues,
        capsys, tmp_path, monkeypatch):
    """The other direction: a run whose rewriters fail would otherwise go on
    from the bare brief with nobody told. By default the operator is asked,
    and anything but a yes -- including having no terminal -- stops the launch
    before the daemon and `meta.json` exist."""
    from dycraft.cli.cli import _rewrite_failed

    monkeypatch.setattr(sys, "stdin", _Stdin(answer, tty))
    sub_run = tmp_path / "sub_runs" / f"{kind}_rewrite"
    if continues:
        assert _rewrite_failed(staged, kind, RuntimeError("timed out"), sub_run,
                               time.time() - 900.0, flag, off) is None
    else:
        with pytest.raises(SystemExit) as stop:
            _rewrite_failed(staged, kind, RuntimeError("timed out"), sub_run,
                            time.time() - 900.0, flag, off)
        assert stop.value.code == 2
    err = capsys.readouterr().err
    assert ("[y/N]" in err) == tty, "asked without a terminal, or not asked with one"
    if not continues:
        assert flag in err and off in err and "--on-rewrite-failure continue" in err
        assert str(sub_run) in err


def test_the_absent_entry_leaves_the_prompt_alone():
    """What a warning costs downstream: one pointer that is not appended.

    `cli._stage_entries` appends each pointer only `if staged.is_file()`, so a rewrite
    that wrote nothing is indistinguishable to the run from one that was never
    asked for. Pinned here because it is the property that makes the warning
    safe, and it lives in a loop that is easy to tighten by accident.
    """
    import inspect

    from dycraft.cli import cli
    src = inspect.getsource(cli._stage_entries)
    loop = src.split("# Scene first, then motion", 1)[1][:400]
    assert "staged.is_file()" in loop
    assert "pointer not in args.prompt" in loop
