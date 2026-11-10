"""Diagnostics an agent gets back from the scene kernel and the authoring sandbox.

A catalog miss names the missing ref and says a retry may help; a syntax error reports its own code yet routes like
a refused construct; an unresolvable entity name lists what does exist, names a wrong part, or says when a query
cannot take the dotted form.
"""
from __future__ import annotations

import pytest


# --- #2 a catalog miss must be diagnosable -------------------------------

def test_catalog_miss_names_the_ref_and_says_a_retry_may_help():
    from dycraft.sdk._errors import DanglingReference
    from dycraft.sdk._scene_kernel import _object_ir_for

    with pytest.raises(DanglingReference) as exc:
        _object_ir_for({"abc": {"name": "o"}}, "d964e017")
    d = exc.value.to_dict()
    assert d["evidence"]["object_ref"] == "d964e017"
    assert d["evidence"]["reason"] == "catalog_miss"
    assert d["evidence"]["n_catalog_entries"] == 1
    # The transient case is the one an agent cannot infer for itself.
    assert any("concurrently" in r for r in d["suggested_repairs"])


# --- #8 a syntax error is not a refused construct ------------------------

def test_a_syntax_error_reports_its_own_code_but_still_routes():
    from dycraft.agent.cad.sandbox import check_ast
    from dycraft.sdk._errors import SandboxViolation, SourceSyntaxError

    # A shape agents write.
    src = "x = compound([box(1), box(2) for y in []])"
    with pytest.raises(SourceSyntaxError) as exc:
        check_ast(src, mode="exec")
    d = exc.value.to_dict()
    assert d["code"] == "SourceSyntaxError"
    assert d["evidence"]["node_type"] == "SyntaxError"
    # Subclass on purpose: the worker and the pool route by this class into
    # the report's `sandbox_violation` slot, and that slot is right for a
    # source rejected before it ran; a sibling class would misfile it.
    assert isinstance(exc.value, SandboxViolation)


def test_a_genuinely_refused_construct_keeps_its_own_code():
    from dycraft.agent.cad.sandbox import check_ast
    from dycraft.sdk._errors import SandboxViolation

    with pytest.raises(SandboxViolation) as exc:
        check_ast("import build123d\n", mode="exec")
    assert exc.value.to_dict()["code"] == "SandboxViolation"


# --- #9 an unresolvable entity name says what is available ---------------

def _tiny_scene():
    obj = {"id": "h1", "name": "tea_table",
           "parts": [{"name": "body"}, {"name": "runners"}]}
    ir = {"instances": [{"name": "tea_table", "object_ref": "h1",
                         "placement": {}}], "groups": []}
    return ir, {"h1": obj}


def test_unknown_instance_lists_the_instances():
    from dycraft.sdk._scene_kernel import _unknown_entity
    ir, cat = _tiny_scene()
    msg = str(_unknown_entity(ir, "no_such_thing", catalog=cat))
    assert "no such instance or group" in msg
    assert "tea_table" in msg


def test_a_dotted_name_whose_part_is_wrong_names_the_parts():
    from dycraft.sdk._scene_kernel import _unknown_entity
    ir, cat = _tiny_scene()
    msg = str(_unknown_entity(ir, "tea_table.nonesuch", catalog=cat))
    assert "has no part 'nonesuch'" in msg
    assert "runners" in msg and "body" in msg


def test_a_dotted_name_a_query_cannot_take_says_so():
    from dycraft.sdk._scene_kernel import _unknown_entity
    ir, cat = _tiny_scene()
    # The recorded case: the instance and the part both exist, so the failure
    # is the query not accepting the dotted form.
    msg = str(_unknown_entity(ir, "tea_table.runners", catalog=cat))
    assert "does not accept the dotted" in msg
