"""Consulting a kinematics solver while authoring an animation.

`h.ik` runs agent-authored code with cuRobo in an isolated interpreter
(`animation/sim/_sim_runner.py`) over the committed scene, and returns whatever
the script records. It **never produces the committed motion**: the agent reads
the answer and authors.

A batched, collision-aware solver picks between the branches of a 7-DOF null
space, which a hand-written solver has no way to do, and so can put a wrist
inside its own forearm. `animation/ik/README.md` is the agent-facing account;
`animation/ik/CUROBO.md` is the API reference checked against the installed
build.

The export is `animation/sim/scene_export.py`'s, unchanged. What this package
adds on top is `ik_meta.json`: the self-collision pairs that
are structural rather than a fault, and the world poses of every static part,
both computed from dycraft's exact CAD kernel because the child interpreter
cannot import `dycraft.*`.
"""

from .ik import IK_DIR, IK_RUNS_DIR, ik, is_available, stage_helpers

__all__ = ["IK_DIR", "IK_RUNS_DIR", "ik", "is_available",
           "stage_helpers"]
