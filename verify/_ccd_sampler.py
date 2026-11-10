"""Joint-pose sampling for continuous-collision verifiers.

Used by `NoSelfPenetration` (object layer) and `NoCrossInstancePenetration` (scene layer) to check penetration over
each joint's full range, not just the rest pose.

Each joint is swept linearly across `[lo, hi]` with the others at their authored defaults, since
interpenetrations usually show up one joint at a time, plus a small random multi-joint hypercube of
`random_samples` poses (default 24) for the rare pair that clips only together (a folding chair, a scissor lift).
Both ends of every range are always sampled: `N = max(2, ceil((hi - lo) / step) + 1)` points by `linspace(lo, hi,
N)`. A joint whose travel is below `JointRangeNonTrivial`'s floor fails that verifier instead.

**Step size is a resolution question.** A sweep steps over an obstacle when the swept point moves further between
samples than the obstacle is thick, so the step that cannot miss a feature of size `t` at radius `r` is `t / r`
radians (`adaptive_revolute_step_rad`). A fixed 15° step is coarser than the thinnest part of the same object
subtends for almost every revolute joint (the median needs about 2.7°), so "no pair found" would be much weaker
than it reads.

Refining is affordable because generating and SAT-testing a pose is ~0.03 ms against ~5 ms to adjudicate one
flagged (pair, pose) exactly. So the sweep is refined freely and the candidate count is bounded:
`MAX_SAMPLES_PER_JOINT` caps the cheap phase against a pathological range (a joint can span 3600°), and the caller
bounds adjudications per part pair. Both bounds are reported, so a `pass` says at what resolution it holds.
"""

from __future__ import annotations

import math
import os
import random


_DEG_TO_RAD = math.pi / 180.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def per_joint_only_default() -> bool:
    """Shared default for `per_joint_only` across object & scene verifiers.

    Reads `DYCRAFT_CCD_PER_JOINT_ONLY` (default True). When True, the
    multi-joint random hypercube is skipped — only the per-joint sweep
    (each joint walked while others stay at default) is sampled. When
    False, the hypercube is added (catches simultaneous-multi-joint
    clashes like adjacent doors both opened to intermediate angles).
    """
    return _env_bool("DYCRAFT_CCD_PER_JOINT_ONLY", True)


def scene_static_only_default() -> bool:
    """Default for the SCENE layer's `static_only`: `DYCRAFT_SCENE_STATIC_ONLY`, default True.

    When True, `NoCrossInstancePenetration` samples the **rest pose only**, every joint at its authored `default`,
    instead of walking each joint across its range. It is a separate switch from `DYCRAFT_CCD_PER_JOINT_ONLY`, which
    the object layer's `NoSelfPenetration` shares: an object's own articulation is still swept, because a door passing
    through its own carcass is a defect in the object that no later layer would notice.

    The sweep assumes every value in a joint's range is a pose the object is meant to hold. That holds for a door or
    a drawer, where a drawer that cannot open clear of a neighbor is a real layout defect. It is false for a
    manipulator, whose joint limits are a mechanical envelope almost none of which anyone will command: sweeping it
    makes the canonical robot cell (an arm bolted to a bench, whose elbow can reach below its own mounting plane)
    unsatisfiable, short of narrowing the arm's ranges or moving it. A hand-authored box arm with a +-105 deg shoulder
    fails the same way as an imported Franka, so this is a property of the check, not of an asset. "Can this motion
    collide" is answered by the animation layer's `NoPenetration`, frame by frame on the authored trajectory. Set
    this to 0 to sweep a scene of furniture, where the premise holds.
    """
    return _env_bool("DYCRAFT_SCENE_STATIC_ONLY", True)


def rest_pose(joints: list[dict]) -> list[dict[str, float]]:
    """The single-element sample list holding every joint at its `default`.

    Same shape `per_joint_samples` / `all_samples` return, so a caller can
    swap one for another without special-casing the result.
    """
    movable = [j for j in joints if j["type"] != "fixed"]
    if not movable:
        return [{}]
    return [{j["name"]: float(j["default"]) for j in movable}]


def revolute_step_rad() -> float:
    """Coarsest sampling step for revolute joints, in radians. Default 15°.

    The ceiling on `adaptive_revolute_step_rad`, and the step used when no
    geometry is supplied. Kept as the ceiling rather than the value so a
    caller that knows nothing about the object samples exactly as before.
    """
    deg = _env_float("DYCRAFT_CCD_REVOLUTE_STEP_DEG", 15.0)
    return deg * _DEG_TO_RAD


def prismatic_step_m() -> float:
    """Coarsest sampling step for prismatic joints, in meters. Default 0.10."""
    return _env_float("DYCRAFT_CCD_PRISMATIC_STEP_M", 0.10)


def max_samples_per_joint() -> int:
    """Cap on samples from one joint's sweep. Default 256.

    Bounds the cheap phase against a range no step size can cover densely:
    the corpus contains a revolute joint spanning 3600°, where the step the
    geometry asks for would ask for ~15 000 samples. At 0.03 ms a sample the
    cap is worth ~8 ms, and reaching it is reported rather than silent.
    """
    return _env_int("DYCRAFT_CCD_MAX_SAMPLES_PER_JOINT", 256)


def adaptive_revolute_step_rad(feature_m: float, radius_m: float) -> float:
    """The angular step at which a feature of size `feature_m` cannot be
    stepped over at radius `radius_m`, clamped to a usable range.

    `feature_m / radius_m` is the angle the feature subtends there: sample
    any coarser and the swept point crosses it between samples. Clamped
    above by `revolute_step_rad()` so this only ever refines, and below by
    a 64th of it so a single hair-thin part cannot drive the whole sweep to
    a resolution the sample cap would truncate anyway.

    Degenerate inputs fall back to the coarse step: a zero radius means the
    child straddles the pivot and has no swept arc to resolve.
    """
    coarse = revolute_step_rad()
    if not (feature_m > 0.0) or not (radius_m > 0.0):
        return coarse
    return min(max(feature_m / radius_m, coarse / 64.0), coarse)


def adaptive_prismatic_step_m(feature_m: float) -> float:
    """The linear step at which a feature of size `feature_m` cannot be
    stepped over, clamped the same way.

    A slide has no lever arm, so the feature size *is* the step: travel more
    than the obstacle's thickness between samples and it is missed.
    """
    coarse = prismatic_step_m()
    if not (feature_m > 0.0):
        return coarse
    return min(max(feature_m, coarse / 64.0), coarse)


def random_samples_count() -> int:
    """Number of random multi-joint hypercube samples. Default 24."""
    return _env_int("DYCRAFT_CCD_RANDOM_SAMPLES", 24)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """End-inclusive linspace: returns [lo, lo+step, ..., hi] with len == n.

    `n` must be >= 2.
    """
    if n < 2:
        raise ValueError(f"linspace requires n >= 2, got {n}")
    if n == 2:
        return [float(lo), float(hi)]
    step = (float(hi) - float(lo)) / (n - 1)
    return [float(lo) + i * step for i in range(n)]


def sample_joint_range(lo: float, hi: float, step: float) -> list[float]:
    """End-inclusive uniform samples across [lo, hi].

    `N = max(2, ceil((hi - lo) / step) + 1)` followed by linspace
    end-inclusive. `lo` and `hi` are always present in the output
    regardless of step size.

    Returns a list of floats; len >= 2.
    """
    if hi < lo:
        raise ValueError(f"sample_joint_range: hi {hi} < lo {lo}")
    if step <= 0:
        raise ValueError(f"sample_joint_range: step must be > 0, got {step}")
    span = float(hi) - float(lo)
    if span <= 0:
        # Degenerate range (caller should already have filtered this out
        # via JointRangeNonTrivial). Return both endpoints anyway.
        return [float(lo), float(hi)]
    n = max(2, math.ceil(span / step) + 1)
    return _linspace(lo, hi, n)


def _joint_step_for(joint: dict, steps: dict | None = None) -> float:
    """Pick the sampling step for a joint.

    `steps` maps joint name to a step the caller derived from geometry (see
    `adaptive_revolute_step_rad`); a joint absent from it, or a caller that
    passes none at all, gets the type's coarse default.
    """
    if steps:
        s = steps.get(joint["name"])
        if s is not None and float(s) > 0.0:
            return float(s)
    jtype = joint["type"]
    if jtype == "revolute":
        return revolute_step_rad()
    if jtype == "prismatic":
        return prismatic_step_m()
    return 0.0


def per_joint_samples(joints: list[dict],
                      steps: dict | None = None,
                      truncated: list | None = None,
                      ) -> list[dict[str, float]]:
    """Build the per-joint sweep pose list.

    For each non-fixed joint, generate `sample_joint_range(lo, hi, step)`
    samples; for every sample, build a pose dict where THAT joint is at
    the sample value and all others are at their authored defaults. The
    first sample of each joint at its `default` is included (this is
    the rest pose), so callers may de-duplicate if they want exactly
    one rest sample.

    `steps` optionally supplies a geometry-derived step per joint name.
    A joint whose step would exceed `max_samples_per_joint()` samples is
    sampled at that many instead, and its name appended to `truncated` if
    the caller passed a list — the achieved resolution is then coarser than
    the geometry asked for, which is a fact about the verdict.
    """
    movable = [j for j in joints if j["type"] != "fixed"]
    if not movable:
        return [{}]  # rest pose only
    defaults = {j["name"]: float(j["default"]) for j in movable}
    cap = max_samples_per_joint()
    out: list[dict[str, float]] = []
    for j in movable:
        lo, hi = j["limits"]
        step = _joint_step_for(j, steps)
        if step <= 0.0:
            continue
        span = abs(float(hi) - float(lo))
        if span > 0.0 and math.ceil(span / step) + 1 > cap:
            step = span / (cap - 1)
            if truncated is not None:
                truncated.append(j["name"])
        for q in sample_joint_range(float(lo), float(hi), step):
            pose = dict(defaults)
            pose[j["name"]] = q
            out.append(pose)
    # Ensure rest pose is included exactly once at the front.
    if not out:
        return [dict(defaults)]
    return out


def random_multi_samples(joints: list[dict], n: int,
                         seed: int = 0xC0DE) -> list[dict[str, float]]:
    """Random Latin-hypercube-ish samples across all non-fixed joints.

    Uniform draws from each joint's `[lo, hi]`. Seeded for determinism;
    consumers should call `random_samples_count()` to pick `n`.
    """
    movable = [j for j in joints if j["type"] != "fixed"]
    if not movable or n <= 0:
        return []
    rng = random.Random(seed)
    out: list[dict[str, float]] = []
    for _ in range(int(n)):
        pose = {}
        for j in movable:
            lo, hi = j["limits"]
            pose[j["name"]] = rng.uniform(float(lo), float(hi))
        out.append(pose)
    return out


def all_samples(joints: list[dict],
                 random_count: int | None = None,
                 seed: int = 0xC0DE,
                 steps: dict | None = None,
                 truncated: list | None = None,
                 ) -> list[dict[str, float]]:
    """Combine the per-joint sweep with `random_count` hypercube samples.

    Returns a deduplicated list of pose dicts. The first element is the
    rest pose (all defaults).

    `steps` and `truncated` pass through to `per_joint_samples`; both
    default to None so a caller with no geometry to hand samples exactly as
    it did before.
    """
    movable = [j for j in joints if j["type"] != "fixed"]
    if not movable:
        return [{}]
    defaults = {j["name"]: float(j["default"]) for j in movable}
    seen: set[tuple] = set()
    out: list[dict[str, float]] = []

    def _push(p: dict[str, float]):
        key = tuple(sorted(p.items()))
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    _push(defaults)
    for p in per_joint_samples(joints, steps=steps, truncated=truncated):
        _push(p)
    n_random = (random_samples_count() if random_count is None
                else int(random_count))
    for p in random_multi_samples(joints, n_random, seed=seed):
        _push(p)
    return out
