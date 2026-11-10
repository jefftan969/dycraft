"""dycraft.verify — verifier suites over finalized IR dicts.

Two suites are shipped:

  * OBJECT_VERIFIERS — object-layer suite (7 verifiers:
    NoSelfPenetration, MinPartDimension, JointRangeNonTrivial,
    JointOriginNearGeometry, JointOriginNearBothBodies, SupportConnected,
    ObjectRelationalAssertions).
  * SCENE_VERIFIERS  — scene-layer suite (`verify/scene_verify.py`):
    PER_OBJECT_VERIFIERS, which is the object suite less SupportConnected
    looped over every definition the scene places (`verify/_per_object.py`),
    then INSTANCE_VERIFIERS (NoCrossInstancePenetration, RoomContainment,
    SupportConnected, SceneRelationalAssertions), then AnatomicalJointLimits.

The animation layer is verified by native Blender checks
(`animation/blender/verify.py`), not by a suite here.

Each suite is a tuple of (name, fn) pairs. fn signature is `(ir, ctx)`; the
ctx argument is ignored by object-layer verifiers.

No object or scene suite runs an automatic image critique: an agent asks for a
visual critique with `review()`, whose images `verify/review_images.py` renders.
"""

from .report import VerifierReport
from .object_verify import OBJECT_VERIFIERS
from ._per_object import PER_OBJECT_VERIFIERS
from .scene_verify import INSTANCE_VERIFIERS, SCENE_VERIFIERS


def run_suite(suite, ir, ctx=None, *, cache=None, cache_key_extra=None) -> list:
    """Run every (name, fn) in `suite` against `ir` and return a list of
    VerifierReports.

    If `cache` is provided, results are memoized by (name, ir_hash, *extra)
    where `extra` is the tuple returned by cache_key_extra(ir_hash, ctx) — or
    () when cache_key_extra is None.
    """
    ir_hash = ir["id"]
    extra = cache_key_extra(ir_hash, ctx) if cache_key_extra else ()
    reports = []
    for name, fn in suite:
        key = (name, ir_hash) + tuple(extra)
        if cache is not None and key in cache:
            reports.append(cache[key])
            continue
        report = fn(ir, ctx)
        if cache is not None:
            cache[key] = report
        reports.append(report)
    return reports


__all__ = [
    "VerifierReport",
    "OBJECT_VERIFIERS",
    "SCENE_VERIFIERS",
    "INSTANCE_VERIFIERS",
    "PER_OBJECT_VERIFIERS",
    "run_suite",
]
