"""Canonical pose helpers for the dual-pose review renders.

For an articulated object the review renders, in addition to the six
default-pose views, two extra pose configurations:

    lo: every non-fixed joint set to its `limits.lo` value.
    hi: every non-fixed joint set to its `limits.hi` value.

Static objects (no non-fixed joints) render the six default views only.

Scene helpers project the same rule across every instance and return
pose dicts keyed `"<instance>.<joint>"` for consumption by the scene
kernel's `_object_poses_for_instance` projection.
"""

from __future__ import annotations

from typing import Mapping


def has_articulation(ir: dict) -> bool:
    """True iff `ir` has at least one non-fixed joint."""
    for j in ir.get("joints", ()):
        if j.get("type") != "fixed":
            return True
    return False


def lo_pose(ir: dict) -> dict:
    """Pose dict: every non-fixed joint at its `limits.lo`."""
    out: dict = {}
    for j in ir.get("joints", ()):
        if j.get("type") == "fixed":
            continue
        lo, _hi = j["limits"]
        out[j["name"]] = float(lo)
    return out


def hi_pose(ir: dict) -> dict:
    """Pose dict: every non-fixed joint at its `limits.hi`."""
    out: dict = {}
    for j in ir.get("joints", ()):
        if j.get("type") == "fixed":
            continue
        _lo, hi = j["limits"]
        out[j["name"]] = float(hi)
    return out


def scene_has_articulation(ir: dict, catalog: Mapping[str, dict]) -> bool:
    """True iff any instance's catalog object has a non-fixed joint."""
    seen: set[str] = set()
    for inst in ir.get("instances", ()):
        ref = inst.get("object_ref")
        if ref in seen or ref not in catalog:
            continue
        seen.add(ref)
        if has_articulation(catalog[ref]):
            return True
    return False


def _scene_pose(ir: dict, catalog: Mapping[str, dict], *,
                limit_idx: int) -> dict:
    """Build a scene pose dict by walking every instance and picking
    each non-fixed joint's lo (idx=0) or hi (idx=1).
    """
    out: dict = {}
    for inst in ir.get("instances", ()):
        ref = inst.get("object_ref")
        if ref not in catalog:
            continue
        for j in catalog[ref].get("joints", ()):
            if j.get("type") == "fixed":
                continue
            q = float(j["limits"][limit_idx])
            out[f"{inst['name']}.{j['name']}"] = q
    return out


def scene_lo_pose(ir: dict, catalog: Mapping[str, dict]) -> dict:
    """Scene-level pose dict: every instance's non-fixed joints at lo."""
    return _scene_pose(ir, catalog, limit_idx=0)


def scene_hi_pose(ir: dict, catalog: Mapping[str, dict]) -> dict:
    """Scene-level pose dict: every instance's non-fixed joints at hi."""
    return _scene_pose(ir, catalog, limit_idx=1)
