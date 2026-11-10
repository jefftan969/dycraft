"""`pose_unchanged` in a check's storyboard observation counts only states whose pose was compared and matched.

A reviewed state skipped for having no frame, an unreadable evaluation record or no pose digest is not known to be
unchanged, so it must not be reported as unchanged.
"""
import pytest

from dycraft.animation import session as S
from dycraft.animation import storyboard as storyboard_module


@pytest.fixture
def observe(monkeypatch, tmp_path):
    def run(boards, records, frames=None):
        monkeypatch.setattr(storyboard_module, "read_json", lambda path, default: boards)
        monkeypatch.setattr(storyboard_module, "pose_digest", lambda record, viewpoints=None: record.get("digest"))
        monkeypatch.setattr(storyboard_module, "poses_comparable", lambda current, approved: True)
        monkeypatch.setattr(storyboard_module, "appeared_states", lambda boards, observed: [])
        monkeypatch.setattr(storyboard_module, "segment_coverage", lambda plan, boards: {})
        session = S.AnimationSession.__new__(S.AnimationSession)
        frames = frames or {name: board.get("frame") for name, board in boards.items()}
        session.resolved_frame = lambda check, name, board: frames[name]
        session.evaluated_at = lambda check, frame: records.get(frame)
        session.observed_instances = lambda check, record, board: None
        session.viewpoints_of = lambda check: None
        session.plan_of = lambda workspace: {}
        return session.observe_storyboard(tmp_path, {"evaluation_path": "evaluation"})
    return run


def test_an_unreadable_state_is_not_counted_unchanged(observe):
    boards = {name: {"frame": i, "reviewed": True, "pose_digest": name} for i, name in enumerate("abcd")}
    records = {0: {"digest": "a"}, 1: {"digest": "b"}, 2: {"digest": "c"}}       # frame 3 unreadable
    payload = observe(boards, records)
    assert payload["states_reviewed"] == 4 and payload["unreadable"] == ["d"]
    assert payload["pose_unchanged"] == 3 and payload["drifted"] == []


def test_only_compared_matching_reviewed_states_count(observe):
    boards = {"same": {"frame": 1, "reviewed": True, "pose_digest": "s"},
              "moved": {"frame": 2, "reviewed": True, "pose_digest": "m"},
              "gone": {"frame": 3, "reviewed": True, "pose_digest": "g"},
              "unframed": {"frame": None, "reviewed": True, "pose_digest": "u"},
              "undigested": {"frame": 5, "reviewed": True},
              "unreviewed": {"frame": 6, "pose_digest": "r"}}
    records = {1: {"digest": "s"}, 2: {"digest": "m2"}, 5: {"digest": "x"}, 6: {"digest": "r"}}
    payload = observe(boards, records)
    assert payload["states_reviewed"] == 5
    assert [d["state"] for d in payload["drifted"]] == ["moved"] and payload["unreadable"] == ["gone"]
    assert payload["pose_unchanged"] == 1
