"""An animation op's reply is slimmed whether it ran interactively or was polled as a job.

`job_result` folds passing verifiers into `verifiers_passed`. The ops registered for the agent return the same shape,
so the default interactive `check_motion` and `storyboard` do not carry every passing report's evidence.
"""
from dycraft.animation import session as S


class _Service:
    def __init__(self, session):
        self.calls, self.jobs = [], None

    def submit(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return {"ok": True, "verifier_reports": [
                    {"name": "NoPenetration", "verdict": "pass", "evidence": {"pairs": [{"a": "x", "b": "y"}] * 50}},
                    {"name": "EventsSatisfied", "verdict": "fail", "evidence": {"events": [{"id": "grasp"}]}}],
                "checks": [{"ok": True, "verifier_reports": [
                    {"name": "ContactSlip", "verdict": "pass", "evidence": {"rows": [1] * 50}}]}]}


class _Session:
    def __init__(self):
        self.ops = {}

    def op(self, name):
        def bind(fn):
            self.ops[name] = fn
            return fn
        return bind


def test_registered_ops_return_slimmed_replies(monkeypatch):
    monkeypatch.setattr(S, "AnimationSession", _Service)
    monkeypatch.setattr("dycraft.animation.dispatch.register", lambda session, service: None)
    monkeypatch.setattr(S, "register_job_handlers", lambda session, jobs: None)
    session = _Session()
    S.register(session)
    reply = session.ops["check_motion"](start=0, end=10)
    assert [r["name"] for r in reply["verifier_reports"]] == ["EventsSatisfied"]
    assert "NoPenetration" in reply["verifiers_passed"]
    nested = reply["checks"][0]
    assert nested["verifier_reports"] == [] and "ContactSlip" in nested["verifiers_passed"]
    assert session.blender_animation.calls == [("check_motion", {"start": 0, "end": 10})]
