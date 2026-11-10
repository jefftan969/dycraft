"""A review whose client was killed must still be recoverable and usable.

The harness client is stateless RPC: when the agent's shell hits its timeout
the daemon still completes the review and writes the result, but the reply dies
with the client. The answer the daemon wrote must still be recoverable, and
read as an answer rather than as a reviewer that said nothing.
"""
from __future__ import annotations

from dycraft.agent.layers.review import _ANSWERED_BUT_SKIPPED, _SILENCE_REASONS, _silence_reason


def test_a_recovered_answer_is_not_treated_as_silence():
    # The failing shape: `completed_after_timeout` falling through to
    # `_SILENCE_REASONS` would label every claim of a review that *did* answer
    # "reviewer timed out; no verdict available".
    assert _silence_reason("completed_after_timeout") == _ANSWERED_BUT_SKIPPED
    assert _silence_reason("completed") == _ANSWERED_BUT_SKIPPED


def test_a_genuinely_dead_reviewer_still_reads_as_silence():
    for outcome, reason in _SILENCE_REASONS.items():
        assert _silence_reason(outcome) == reason
        assert "no verdict available" in reason


def test_oneoff_reports_an_answer_recovered_after_a_kill():
    import inspect

    from dycraft.agent import oneoff

    src = inspect.getsource(oneoff.run_oneoff)
    # The regression is a one-token change, so assert the property rather than
    # the text: `ok` must not be conjoined with `not timed_out`.
    assert '"ok": (not timed_out) and text is not None' not in src
    assert '"ok": text is not None' in src
    assert "completed_after_timeout" in src


def test_a_reviewer_that_answered_nothing_is_a_failed_review_not_n_abstentions():
    """A reviewer that ran out its time, answered no claim and returned no JSON
    is one failure, labeled as the timeout, not a set of abstentions."""
    from dycraft.agent.layers.review import _review_verdicts
    ids = ("scene:a", "scene:b", "piano_dolly:c")
    verdicts, silence, evaluated, silent = _review_verdicts(
        "I looked at the renders and will now write my verdicts.",
        "completed_after_timeout", ids)
    assert silent and evaluated == 0
    assert silence == _SILENCE_REASONS["timeout"]
    assert {v["reason"] for v in verdicts.values()} == {_SILENCE_REASONS["timeout"]}


def test_a_reviewer_that_skipped_some_claims_still_reviewed():
    """A reviewer that answered some claims keeps the per-claim label for the
    ones it skipped."""
    from dycraft.agent.layers.review import _review_verdicts
    ids = ("scene:a", "scene:b")
    reply = '{"results": [{"id": "scene:a", "verdict": "met", "view": "front", "reason": "ok"}]}'
    verdicts, silence, evaluated, silent = _review_verdicts(reply, "completed_after_timeout", ids)
    assert not silent and evaluated == 1
    assert verdicts["scene:b"]["reason"] == _ANSWERED_BUT_SKIPPED
