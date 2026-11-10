"""The launcher must not hold a wedged codex open forever.

Pointed at a credit-exhausted account, codex's TUI prints the error, stops
answering terminal probes and never exits, while its session rollout records
`usage_limit_exceeded` within seconds. `_spawn_with_pty_term_shim` mirrors its
output and waits, so without the watch an unattended run would hold the full
time limit on every remaining scene.

The watch reads the rollout rather than the mirrored terminal stream on
purpose: a text match on mirrored output can fire on the agent's own words,
and a JSONL record cannot.

Record shapes here are copied from real rollouts in `~/.codex1`/`~/.codex2`.
"""

from __future__ import annotations

import json

import pytest

from dycraft.cli.backends._process import (_CodexRolloutWatch, _rollout_fatal_reason,
                                           _ROLLOUT_DISCOVERY_SECONDS, _ROLLOUT_FATAL_GRACE_SECONDS)

_FATAL = {
    "timestamp": "2026-09-09T02:00:27.208Z", "type": "event_msg",
    "payload": {
        "type": "task_complete", "turn_id": "01a083e5", "last_agent_message": None,
        "error": {"message": "Your workspace is out of credits. Ask your "
                             "workspace owner to refill in order to continue.",
                  "codex_error_info": "usage_limit_exceeded"},
        "duration_ms": 2840}}

_UNAUTHORIZED = {
    "timestamp": "2026-09-12T11-08-02Z", "type": "event_msg",
    "payload": {"type": "task_complete", "last_agent_message": None,
                "error": {"message": "Your access token could not be refreshed",
                          "codex_error_info": "unauthorized"}}}

_ANSWERED = {
    "type": "event_msg",
    "payload": {"type": "task_complete", "last_agent_message": "done",
                "error": {"codex_error_info": "usage_limit_exceeded"}}}

_WORK = {"type": "token_usage_record",
         "payload": {"usage": {"input_tokens": 10, "output_tokens": 2}}}

#: A new turn and the model's first output in it: what a recovered session
#: writes. Shapes from codex 0.153.4 rollouts.
_NEW_TURN = {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t2"}}
_REASONING = {"type": "response_item", "payload": {"type": "reasoning", "summary": []}}

#: What a session that is *not* working still writes: a command the failed turn
#: left running completes, and the token count is refreshed. The first is the
#: record that kept one child up 858 minutes after its usage limit.
_COMMAND_DONE = {"type": "event_msg", "payload": {
    "type": "item_completed", "item": {"type": "CommandExecution", "id": "c1",
                                       "command": "python scripts/run.py pivot"}}}
_TOKENS = {"type": "event_msg", "payload": {"type": "token_count", "info": {}}}


# --- the record classifier -------------------------------------------------

@pytest.mark.parametrize("rec,want", [
    (_FATAL, True),
    (_UNAUTHORIZED, True),
    # A turn that produced an answer finished, whatever else it recorded.
    (_ANSWERED, False),
    (_WORK, False),
    ({"type": "event_msg", "payload": {"type": "task_complete"}}, False),
    # `payload.type == "error"` is a different, recoverable record: codex goes
    # on working after it.
    ({"type": "event_msg", "payload": {"type": "error",
                                       "message": "transient"}}, False),
])
def test_only_an_unanswered_fatal_task_complete_counts(rec, want):
    assert (_rollout_fatal_reason(rec) is not None) is want


def test_the_reason_names_the_code_and_the_message():
    why = _rollout_fatal_reason(_FATAL)
    assert "usage_limit_exceeded" in why and "out of credits" in why


# --- the watch ------------------------------------------------------------

def _session(tmp_path, cwd, records, *, home_name=".codexT"):
    """A `$CODEX_HOME` holding one rollout for `cwd`."""
    home = tmp_path / home_name
    day = home / "sessions" / "2026" / "09" / "13"
    day.mkdir(parents=True, exist_ok=True)
    path = day / "rollout-2026-09-13T00-00-00-abc.jsonl"
    head = {"type": "session_meta",
            "payload": {"cwd": str(cwd), "session_id": "abc"}}
    path.write_text("".join(json.dumps(r) + "\n" for r in [head] + records))
    return home, path


def _watch(tmp_path, home, cwd):
    w = _CodexRolloutWatch(["codex", "-C", str(cwd), "--add-dir", str(cwd)],
                           {"CODEX_HOME": str(home)})
    assert w.enabled, "watch should arm for a codex launch with -C and a home"
    return w


def test_a_wedged_session_is_killed_after_the_grace(tmp_path, monkeypatch):
    cwd = tmp_path / "run"
    cwd.mkdir()
    home, _ = _session(tmp_path, cwd, [_WORK, _FATAL])
    w = _watch(tmp_path, home, cwd)

    assert w.poll() is None, "must not fire the instant it sees the record"
    # Nothing more is written; the grace elapses.
    w.armed_at -= _ROLLOUT_FATAL_GRACE_SECONDS + 1
    w._next_poll = 0
    why = w.poll()
    assert why and "usage_limit_exceeded" in why


@pytest.mark.parametrize("recovery", [_NEW_TURN, _REASONING], ids=["new_turn", "model_output"])
def test_a_session_whose_model_works_again_is_left_alone(tmp_path, recovery):
    """The 24-second `unauthorized` token refresh, which recovered and went on
    to record 251 more entries. Killing it would have thrown away a live run."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    home, path = _session(tmp_path, cwd, [_UNAUTHORIZED])
    w = _watch(tmp_path, home, cwd)
    assert w.poll() is None
    assert w.armed_at is not None, "should be armed after the error"

    with path.open("a") as fh:
        fh.write(json.dumps(recovery) + "\n")
    w._next_poll = 0
    assert w.poll() is None
    assert w.armed_at is None, "the model working again must disarm the watch"

    w._next_poll = 0
    assert w.poll() is None


def test_records_written_without_the_model_do_not_disarm(tmp_path):
    """B12: a child hit its usage limit, a background command's completion
    record followed 4.6 s later, and that record kept the child up 858 min."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    home, path = _session(tmp_path, cwd, [_FATAL])
    w = _watch(tmp_path, home, cwd)
    assert w.poll() is None and w.armed_at is not None
    with path.open("a") as fh:
        for rec in (_COMMAND_DONE, _TOKENS, _WORK):
            fh.write(json.dumps(rec) + "\n")
    w._next_poll = 0
    assert w.poll() is None
    assert w.armed_at is not None, "a record the model did not write must not disarm"
    w.armed_at -= _ROLLOUT_FATAL_GRACE_SECONDS + 1
    w._next_poll = 0
    assert "usage_limit_exceeded" in w.poll()


def _past_discovery(w):
    w.started -= _ROLLOUT_DISCOVERY_SECONDS + 1
    w._next_poll = 0
    assert w.poll() is None


def test_a_launch_that_never_starts_a_session_is_stalled(tmp_path):
    """No rollout and no harness call after the discovery window: codex is
    stuck at startup, on a screen nobody knows or on nothing at all."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    (cwd / "trace.jsonl").write_text('{"ev": "daemon_start"}\n')
    w = _watch(tmp_path, tmp_path / ".codexT", cwd)
    assert not w.stalled_at_startup()
    _past_discovery(w)
    assert w.stalled_at_startup()


def test_a_run_that_reached_the_harness_is_not_stalled_without_a_rollout(tmp_path):
    """A rollout this watch failed to find is not a stuck codex. The harness
    call is the check, so a healthy run is never stopped for it."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    (cwd / "trace.jsonl").write_text('{"ev": "op_call", "op": "probe"}\n')
    w = _watch(tmp_path, tmp_path / ".codexT", cwd)
    _past_discovery(w)
    assert not w.stalled_at_startup()
    assert not w.stalled_at_startup()


def test_another_run_s_rollout_is_not_watched(tmp_path):
    """Codex keys sessions by date across every project, so cwd is the only
    thing tying a rollout to this launch."""
    cwd, other = tmp_path / "mine", tmp_path / "theirs"
    cwd.mkdir()
    other.mkdir()
    home, _ = _session(tmp_path, other, [_FATAL])
    w = _watch(tmp_path, home, cwd)
    for _ in range(3):
        w._next_poll = 0
        assert w.poll() is None
    assert w.path is None


def test_a_sub_thread_is_not_watched(tmp_path):
    """A reviewer sub-thread shares the run's cwd but is not the session that
    hangs, and it is written later, so a newest-first scan would prefer it."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    home = tmp_path / ".codexT"
    day = home / "sessions" / "2026" / "09" / "13"
    day.mkdir(parents=True)
    (day / "rollout-2026-09-13T00-00-01-sub.jsonl").write_text(
        json.dumps({"type": "session_meta",
                    "payload": {"cwd": str(cwd), "parent_thread_id": "p"}})
        + "\n" + json.dumps(_FATAL) + "\n")
    w = _watch(tmp_path, home, cwd)
    w._next_poll = 0
    assert w.poll() is None
    assert w.path is None


def test_a_half_written_record_is_not_parsed(tmp_path):
    """The rollout is appended to while we read it."""
    cwd = tmp_path / "run"
    cwd.mkdir()
    home, path = _session(tmp_path, cwd, [_WORK])
    w = _watch(tmp_path, home, cwd)
    assert w.poll() is None

    blob = json.dumps(_FATAL)
    with path.open("a") as fh:
        fh.write(blob[:40])          # torn write, no newline
    w._next_poll = 0
    assert w.poll() is None
    assert w.armed_at is None

    with path.open("a") as fh:
        fh.write(blob[40:] + "\n")   # completed
    w._next_poll = 0
    assert w.poll() is None
    assert w.armed_at is not None, "the completed record must arm the watch"


def test_the_watch_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    cwd = tmp_path / "run"
    cwd.mkdir()
    home, _ = _session(tmp_path, cwd, [_FATAL])
    w = _CodexRolloutWatch(["codex", "-C", str(cwd)], {"CODEX_HOME": str(home)})
    assert not w.enabled
    assert w.poll() is None


def test_a_launch_with_no_codex_home_is_inert(tmp_path, monkeypatch):
    """Nothing to read, so nothing to act on — and never a guess at which
    account's sessions tree to search."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    w = _CodexRolloutWatch(["codex", "-C", str(tmp_path)], {})
    assert w.home is None
    assert not w.enabled
    assert w.poll() is None


def test_a_launch_with_no_run_dir_is_inert(tmp_path, monkeypatch):
    """`-C <run_dir>` is the only link between a launch and its rollout."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    w = _CodexRolloutWatch(["codex", "--add-dir", str(tmp_path)],
                           {"CODEX_HOME": str(tmp_path)})
    assert w.cwd == ""
    assert not w.enabled
