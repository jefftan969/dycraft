"""Pose scopes and instance boundaries for inherited object contact permissions."""
import math

from dycraft.verify._inherited_allowances import InheritedAllowances


def _permissions():
    obj = {
        "joints": [
            {"name": "hinge", "type": "revolute", "default": math.pi / 2},
            {"name": "slide", "type": "prismatic", "default": 0.015},
        ],
        "collision_allowances": [
            {"a": "housing", "b": "logo", "reason": "fixed assembly"},
            {"a": "door", "b": "housing", "joint": "hinge",
             "q_range": [80, 100], "reason": "hinge contact"},
            {"a": "door", "b": "housing", "joint": "hinge",
             "q_range": [170, 180], "reason": "second contact region"},
            {"a": "slider", "b": "housing", "joint": "slide",
             "q_range": [0.01, 0.02], "reason": "sliding contact"},
        ],
    }
    return InheritedAllowances(
        {"instances": [{"name": "one", "object_ref": "asset"},
                       {"name": "two", "object_ref": "asset"}]},
        {"asset": obj})


def test_inherited_ranges_use_surface_units_defaults_and_all_scopes():
    p = _permissions()
    assert p.allows("one", "housing", "door", {})
    assert p.allows("one", "slider", "housing", {})
    assert p.allows("one", "door", "housing", {"one.hinge": math.pi})
    assert p.allows("one", "door", "housing", {"one.hinge": math.radians(80)})
    assert not p.allows("one", "door", "housing", {"one.hinge": 1.0})
    assert not p.allows("one", "slider", "housing", {"one.slide": 0.03})
    assert p.allows("one", "door", "housing", {"two.hinge": 0.0})
    assert not p.allows("one", "slider", "door", {})
    assert not p.allows("unknown", "housing", "logo", {})


