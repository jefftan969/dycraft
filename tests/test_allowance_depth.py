"""A collision allowance excuses an overlap from failing, not from being measured.

Skipping an allowed pair outright would let a reason saying "a few mm" excuse a
4 cm overlap. So each allowed overlap's depth is measured and reported
(`allowed_contacts`),
and an allowance may state `max_depth_m`, past which `NoPenetration` fails.
"""
import pytest

from dycraft.animation.blender.chunking import merge_scoped
from dycraft.animation.blender.verify import (ROWS_FORMAT, allowance_limit, allowance_overrun_repair,
                                              allowance_overruns, allowed, allowed_contacts,
                                              resolve_declarations)

INFO = {"fps": 24.0, "n_frames": 240, "frame_start": 0}
BASELINE = {"identities": {"dog/body": {}, "roll/paper": {}, "owner/body": {}}}


def _decl(*allowances):
    return {"units": "frames", "collision_allowances": list(allowances)}


def test_max_depth_is_optional_and_read_in_metres():
    index, _, errors = resolve_declarations(_decl(
        {"a": "dog/body", "b": "roll/paper", "reason": "held in the mouth", "window": [10, 20],
         "max_depth_m": 0.005},
        {"a": "owner/body", "b": "dog/body", "reason": "stroking"}), BASELINE, INFO)
    assert not errors
    assert allowed(index, "dog/body", "roll/paper", 15) and not allowed(index, "dog/body", "roll/paper", 30)
    assert allowance_limit(index, "roll/paper", "dog/body", 15) == 0.005
    assert allowance_limit(index, "owner/body", "dog/body", 100) is None


@pytest.mark.parametrize("bad", [-0.01, "5mm", True, float("nan")])
def test_a_bad_max_depth_is_a_declaration_error_not_a_crash(bad):
    index, _, errors = resolve_declarations(_decl(
        {"a": "dog/body", "b": "roll/paper", "reason": "held", "max_depth_m": bad}), BASELINE, INFO)
    assert errors and "max_depth_m" in errors[0]["reason"]
    assert not allowed(index, "dog/body", "roll/paper", 0), "a dropped allowance makes the check stricter"


def test_an_unbounded_allowance_in_force_leaves_the_pair_unbounded():
    index, _, _ = resolve_declarations(_decl(
        {"a": "dog/body", "b": "roll/paper", "reason": "grip", "window": [0, 50], "max_depth_m": 0.004},
        {"a": "dog/body", "b": "roll/paper", "reason": "carry", "window": [40, 90]}), BASELINE, INFO)
    assert allowance_limit(index, "dog/body", "roll/paper", 20) == 0.004
    assert allowance_limit(index, "dog/body", "roll/paper", 45) is None


ROWS = [
    {"frame": 10, "a": "dog/body", "b": "roll/paper", "depth": 0.003, "limit": 0.005},
    {"frame": 11, "a": "dog/body", "b": "roll/paper", "depth": 0.041, "limit": 0.005},
    {"frame": 12, "a": "roll/paper", "b": "dog/body", "depth": 0.038, "limit": 0.005},
    {"frame": 50, "a": "owner/body", "b": "dog/body", "depth": 0.027, "limit": None},
    {"frame": 51, "a": "owner/body", "b": "dog/body", "depth": None, "limit": None},
]


def test_every_allowed_overlap_reports_its_deepest_measured_depth():
    contacts = {(c["a"], c["b"]): c for c in allowed_contacts(ROWS)}
    roll = contacts[("dog/body", "roll/paper")]
    assert roll["max_depth_m"] == 0.041 and roll["max_depth_frame"] == 11 and roll["frames"] == 3
    assert roll["limit_m"] == 0.005
    hand = contacts[("dog/body", "owner/body")]
    assert hand["max_depth_m"] == 0.027 and "limit_m" not in hand
    assert list(contacts)[0] == ("dog/body", "roll/paper"), "deepest first"


def test_only_a_stated_bound_fails_and_the_repair_names_the_depth():
    over = allowance_overruns(ROWS)
    assert [row["frame"] for row in over] == [11, 12], "within the bound, or unbounded, never fails"
    repair = allowance_overrun_repair(over)
    assert len(repair) == 1 and "41 mm" in repair[0] and "5 mm" in repair[0]


def test_the_repair_quotes_the_bound_the_deepest_overrun_broke():
    """One pair with two allowances of different depth, both overrun: the repair and the evidence pair the deepest
    overrun with its own allowance's bound, not the first one seen."""
    rows = [{"frame": 10, "a": "a", "b": "b", "depth": 0.015, "limit": 0.010},
            {"frame": 40, "a": "a", "b": "b", "depth": 0.060, "limit": 0.050}]
    over = allowance_overruns(rows)
    assert allowance_overrun_repair(over) == [
        "a / b went 60 mm deep at frame 40, past its allowance's max_depth_m of 50 mm: change the pose or the "
        "parts so they meet as the allowance says, or raise max_depth_m if that depth is what is meant"]
    assert allowance_overrun_repair(list(reversed(over)))[0].startswith(
        "a / b went 60 mm deep at frame 40, past its allowance's max_depth_m of 50 mm")


def test_a_recheck_merges_onto_a_base_checked_before_allowed_rows_existed():
    base = {"format": ROWS_FORMAT, "frames": [0, 1, 2], "context": {}, "stats": {},
            "rows": {"NoPenetration": {"hits": [{"frame": 0}], "escapes": []}}}
    scoped = {"format": ROWS_FORMAT, "frames": [1], "context": {}, "stats": {},
              "rows": {"NoPenetration": {"hits": [], "escapes": [],
                                         "allowed": [{"frame": 1, "a": "x", "b": "y", "depth": 0.01, "limit": None}]}}}
    merged = merge_scoped(base, scoped)
    assert merged["rows"]["NoPenetration"]["allowed"] == scoped["rows"]["NoPenetration"]["allowed"]
    assert merged["rows"]["NoPenetration"]["hits"] == [{"frame": 0}]
