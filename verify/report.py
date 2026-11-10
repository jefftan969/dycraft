"""VerifierReport — pure-data report emitted by each verifier."""

from dataclasses import dataclass, field
from typing import Literal


#: `not_applicable` is a check that had nothing to check by construction --
#: no joints, no enclosing walls, no declared assertions -- and says so rather
#: than reporting a `pass` that reads the same as one that looked. Never for
#: things that exist but could not be evaluated: that stays `fail` or
#: `uncertain`.
Verdict = Literal["pass", "fail", "uncertain", "not_applicable"]

#: The verdicts a gate lets through.
PASSING = frozenset({"pass", "not_applicable"})


def not_applicable(name: str, ir_hash: str, reason: str, **evidence) -> "VerifierReport":
    """A report for a check with nothing to check, naming why."""
    return VerifierReport(name=name, verdict="not_applicable", ir_hash=ir_hash,
                          evidence={"not_applicable": reason, **evidence},
                          suggested_repairs=[])


@dataclass
class VerifierReport:
    name: str
    verdict: Verdict
    ir_hash: str
    evidence: dict = field(default_factory=dict)
    suggested_repairs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "ir_hash": self.ir_hash,
            "evidence": dict(self.evidence),
            "suggested_repairs": list(self.suggested_repairs),
        }
