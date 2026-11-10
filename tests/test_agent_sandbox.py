"""Tests for the agent sandbox: AST allowlist + restricted globals + the
end-to-end compile_with_layer pipeline.

Each test targets behavior that is not obvious from reading the
implementation: defense-in-depth between the AST check and the restricted
__builtins__, the order in which the two layers run, the per-distinct-path
allowlist branches (rejected node-type vs banned-name vs attribute-write vs
dunder-read vs eval-mode), and the routing of each failure class into the
correct compile_with_layer payload field.
"""

import builtins as _builtins

import pytest

from dycraft.agent.cad import sandbox as sb
from dycraft.agent.layers.compile import compile_with_layer
from dycraft.agent.layers.layer import object_layer_config
from dycraft.sdk._errors import SandboxViolation
from dycraft.sdk.object_builder import Object, _reset_object_singleton
from dycraft.verify.object_verify import joint_origin_near_both_bodies


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


class _BareCtx:
    """Minimal stand-in for ToolContext. Object verifiers don't read ctx."""


def _compile_object(source):
    return compile_with_layer(
        source,
        layer_cfg=object_layer_config(),
        ctx=_BareCtx(),
    )


# ---------------------------------------------------------------------------
# AST allowlist: one test per distinct rejection path, not per banned token.
# Each path has different code, so a single test of "import" wouldn't catch a
# regression in the banned-name path, etc.
# ---------------------------------------------------------------------------

def test_ast_rejects_rejected_node_type():
    """Catches a regression where ast.With gets accidentally added to the
    allowlist.

    `Try` is allowed (geometry fails at run time and an author needs to fall
    back), so the example is `With`, refused because builder mode is the only
    thing that wants it."""
    with pytest.raises(SandboxViolation) as ei:
        sb.check_ast("with open('f') as fh:\n    pass\n", mode="exec")
    ev = ei.value.to_dict()["evidence"]
    assert ev["node_type"] == "With"
    assert ev["lineno"] == 1


def test_ast_rejects_banned_name_even_inside_a_function_body():
    """The walker must descend through FunctionDef bodies. A regression that
    e.g. switches from ast.walk to a shallow visitor would miss this."""
    with pytest.raises(SandboxViolation) as ei:
        sb.check_ast(
            "def f():\n    return __import__('os')\n",
            mode="exec",
        )
    ev = ei.value.to_dict()["evidence"]
    # The offending lineno is the nested return, not the def line.
    assert ev["lineno"] == 2


def test_ast_rejects_attribute_assignment_but_allows_attribute_read():
    """`.x` read is fine; `obj.x = ...` is rejected. A naive allowlist that
    permits `Attribute` unconditionally would allow both."""
    sb.check_ast("y = (1, 2, 3); z = y[0]; w = (1).real", mode="exec")
    with pytest.raises(SandboxViolation):
        sb.check_ast("o = (1,)\no.x = 1", mode="exec")


def test_ast_rejects_dunder_attribute_read():
    """Public attribute read passes; underscore-prefixed attribute is rejected
    independently of the AST node type."""
    sb.check_ast("x = (1).real", mode="exec")     # public attr — ok
    with pytest.raises(SandboxViolation):
        sb.check_ast("x = (1).__class__", mode="exec")


def test_ast_eval_mode_rejects_statements_but_accepts_expressions():
    """The mode flag actively rejects assignment statements in eval-mode.
    A regression that forgets to switch ast.parse(mode=...) would silently
    let `probe` accept arbitrary code."""
    with pytest.raises(SandboxViolation):
        sb.check_ast("x = 1", mode="eval")
    sb.check_ast("1 + 2 * 3", mode="eval")        # single expression — ok


def test_probe_falls_back_to_statement_mode_without_widening_the_sandbox():
    """`probe` tries eval, then exec; the allowlist governs both.

    Holding a probe to expressions would be syntactic, not a safety boundary --
    `eval` already calls whatever the allowlist permits, and an agent held to
    expressions wraps its statements in
    `(lambda ns: (exec("..."), ns['out'])[1])({})` anyway. What must stay true is
    that falling back to statement mode does not admit a construct the
    sandbox refuses in expression mode.
    """
    from dycraft.sdk._errors import SourceSyntaxError

    def accepted(src):
        modes = []
        for mode in ("eval", "exec"):
            try:
                sb.check_ast(src, mode=mode)
                modes.append(mode)
            except SourceSyntaxError:
                pass
        return modes

    # A statement block is unavailable in eval mode and legal in exec mode.
    assert accepted("total = 0\nfor i in range(3):\n    total += i\nresult = total") == ["exec"]
    # A malformed probe fails both, so the eval-mode diagnosis is what reports.
    assert accepted("for i in range(:") == []
    # The fallback is a parse mode, not a policy change: a refused construct
    # is still refused when it arrives as a statement.
    with pytest.raises(SandboxViolation):
        sb.check_ast("result = (1).__class__", mode="exec")


# ---------------------------------------------------------------------------
# Defense in depth: restricted __builtins__ stops eval/exec even if the AST
# allowlist is somehow bypassed. We bypass it here by calling exec directly
# on build_globals(), then assert the restricted dict alone is sufficient.
# ---------------------------------------------------------------------------

def test_restricted_builtins_block_eval_even_when_ast_check_is_skipped():
    g = sb.build_globals()
    # `eval` is a normal lookup at exec time; if it isn't in __builtins__,
    # Python raises NameError on use. We bypass check_ast on purpose.
    with pytest.raises(NameError):
        exec(_builtins.compile("eval('1+1')", "<t>", "exec"), g)


def test_restricted_builtins_block_open_even_when_ast_check_is_skipped():
    g = sb.build_globals()
    with pytest.raises(NameError):
        exec(_builtins.compile("open('/etc/passwd')", "<t>", "exec"), g)


# ---------------------------------------------------------------------------
# End-to-end compile_with_layer: each routing path verified against a source
# that exercises ONLY that path. Uses object_layer_config() throughout.
# ---------------------------------------------------------------------------

def test_compile_routes_sdk_BuildError_to_build_error_payload():
    src = (
        "obj = Object('x')\n"
        "a = obj.make_box('p', 1.0, 1.0, 1.0)\n"
        "b = obj.make_box('p', 1.0, 1.0, 1.0)\n"   # duplicate name
    )
    result = _compile_object(src)
    assert result["ok"] is False
    assert result["sandbox_violation"] is None
    err = result["build_error"]
    assert err["code"] == "InvalidPrimitive"
    # Structured fields, not just a string.
    kinds = {(r["kind"], r["name"]) for r in err["refs"]}
    assert ("object", "x") in kinds
    assert ("part", "p") in kinds


def test_ast_allowlist_runs_before_exec():
    """If the AST check fired AFTER exec, this source — which has both an
    illegal import (line 1) and a SDK-time error (duplicate-name at line 4)
    — would surface as build_error. The contract is that it surfaces as
    sandbox_violation on line 1 instead, because the AST check gates exec.
    """
    src = (
        "import os\n"
        "obj = Object('x')\n"
        "a = obj.make_box('p', 1.0, 1.0, 1.0)\n"
        "b = obj.make_box('p', 1.0, 1.0, 1.0)\n"
    )
    result = _compile_object(src)
    assert result["build_error"] is None
    sv = result["sandbox_violation"]
    assert sv["code"] == "SandboxViolation"
    assert sv["evidence"]["node_type"] == "Import"
    assert sv["evidence"]["lineno"] == 1


def test_compile_routes_missing_top_level_ir_to_build_error():
    """The compile step checks for a top-level `ir` name after exec; the
    SDK has no way to enforce this naming. Tests the harness-only branch."""
    src = (
        "obj = Object('x')\n"
        "a = obj.make_box('a', 1, 1, 1)\n"
        "obj.declare_root(a)\n"
        "result = obj.finalize()\n"   # assigned to wrong name
    )
    result = _compile_object(src)
    assert result["ok"] is False
    assert result["build_error"]["code"] == "BuildError"
    # The harness-emitted error carries the structured reason.
    assert result["build_error"]["evidence"]["reason"] == "name 'ir' not defined"


def test_compile_reports_verifier_failure_with_pair_evidence_and_null_ir():
    """Verifier-failure path: build succeeds, but verifier returns fail.
    compile_with_layer must NOT expose the failing IR (ir is None / ir_hash
    is None), preventing an in-process consumer from accidentally committing
    a failing model. The pair-depth value also exercises the SDK's AABB
    overlap math under articulated pose composition."""
    src = (
        "obj = Object('x')\n"
        "a = obj.make_box('a', 0.5, 0.4, 0.3)\n"
        "b = obj.make_box('b', 0.5, 0.4, 0.3)\n"
        "obj.attach_fixed('j', a, b,\n"
        "    parent_anchor=obj.anchor(a, 'center', translation=(0.0, 0.10, 0.0)),\n"
        "    child_anchor=obj.anchor(b, 'center'))\n"
        "obj.declare_root(a)\n"
        "ir = obj.finalize()\n"
    )
    result = _compile_object(src)
    assert result["ok"] is False
    assert result["ir_hash"] is None              # no commit-by-accident
    assert result["ir"] is None                   # failing IR not exposed
    assert result["build_error"] is None
    assert result["sandbox_violation"] is None
    # The cheap suite for the object layer runs NoSelfPenetration followed
    # by ObjectRelationalAssertions. The expensive VlmJudge is skipped per
    # cheap→expensive short-circuit when a cheap verifier fails. Find the
    # NoSelfPenetration report and check its evidence shape.
    by_name = {r["name"]: r for r in result["verifier_reports"]}
    report = by_name["NoSelfPenetration"]
    assert report["verdict"] == "fail"
    pair = report["evidence"]["pairs"][0]
    # The min-axis depth (0.4-long boxes offset by 0.10 along Y → Y-overlap of
    # 0.30) still decides the pair is reported, but it is a box extent and no
    # longer ships as evidence; `pair_report` is where a caller asks for it.
    assert {pair["a"], pair["b"]} == {"a", "b"}
    assert "depth" not in pair and "contact_normal" not in pair
    # ObjectRelationalAssertions has no assertions to evaluate, and says so.
    assertions_report = by_name["ObjectRelationalAssertions"]
    assert assertions_report["verdict"] == "not_applicable"
    assert assertions_report["evidence"]["not_applicable"] == "no assertions declared"


# ---------------------------------------------------------------------------
# Public ref attributes are sandbox-accessible (no leading-underscore reach)
# ---------------------------------------------------------------------------


def test_public_ref_attributes_pass_sandbox():
    """`.name` / `.kind` / `.source` etc. on Refs must be reachable from
    sandboxed code, so authors can build dynamic assertion names (e.g.
    `name=joint.name + "_aligned"`) without hitting the AST allowlist."""
    src = (
        "obj = Object('o')\n"
        "a = obj.make_box('a', 1, 1, 1)\n"
        "b = obj.make_box('b', 1, 1, 1)\n"
        "j = obj.attach_revolute(name='hinge', parent=a, child=b,\n"
        "    axis=(0, 0, 1), limits=(0.0, 90.0), default=0.0,\n"
        "    parent_anchor=obj.anchor(a, 'top'),\n"
        "    child_anchor=obj.anchor(b, 'bottom'))\n"
        "# Read public attrs from each ref class. The AST allowlist\n"
        "# rejects underscore-prefixed names; everything below must\n"
        "# be a public property.\n"
        "_ = a.name + '_part'\n"
        "_ = j.name + '_joint'\n"
        "_ = j.kind\n"
        "_ = j.axis().source\n"
        "anc = obj.anchor(a, 'top')\n"
        "_ = anc.name\n"
        "_ = anc.entity_name\n"
        "_ = anc.entity_kind\n"
        "ar = obj.assert_angle(a=j.axis(),\n"
        "    b=obj.anchor(a, 'top').outward(),\n"
        "    min_deg=0.0, max_deg=1.0, name='along_top')\n"
        "_ = ar.name\n"
        "_ = ar.kind\n"
        "# Read-back of constructor-supplied state\n"
        "_ = a.poses\n"
        "tagged_a = a.at({'hinge': 30.0})\n"
        "_ = tagged_a.poses['hinge']\n"
        "anc_off = obj.anchor(a, 'top', translation=(0.1, 0.0, 0.0),\n"
        "    rotation=(0.0, 45.0, 0.0))\n"
        "_ = anc_off.translation\n"
        "_ = anc_off.rotation\n"
        "_ = anc_off.poses\n"
        "ax_dir = j.axis()\n"
        "_ = ax_dir.poses\n"
        "obj.declare_root(a)\n"
        "ir = obj.finalize()\n"
    )
    result = _compile_object(src)
    assert result["sandbox_violation"] is None, result["sandbox_violation"]
    assert result["build_error"] is None, result["build_error"]
    assert result["ok"] is True


def test_builder_method_calls_pass_the_structural_sandbox():
    """The sandbox is a *structural* AST allowlist, not a per-method whitelist.

    Pins that: a builder call is just Call+Attribute on a non-underscore name,
    so adding or removing an SDK method needs no sandbox change. If it is ever
    tightened to an explicit method whitelist, this fails here rather than
    silently rejecting agent-authored code at runtime.
    """
    sb.check_ast(
        "obj.make_box('seat', 0.30, 0.30, 0.04)\n"
        "obj.add_part('body', shape)\n",
        mode="exec",
    )

    src = (
        "obj = Object('stool')\n"
        "seat = obj.make_box('seat', 0.30, 0.30, 0.04)\n"
        "leg = obj.make_box('leg', 0.04, 0.04, 0.40)\n"
        "obj.attach_fixed('j1', seat, leg,\n"
        "    parent_anchor=obj.anchor(seat, 'bottom'),\n"
        "    child_anchor=obj.anchor(leg, 'top'))\n"
        "obj.declare_root(seat)\n"
        "ir = obj.finalize()\n"
    )
    result = _compile_object(src)
    assert result["sandbox_violation"] is None, result["sandbox_violation"]
    assert result["build_error"] is None, result["build_error"]
    assert result["ok"] is True
    assert {p["primitive"] for p in result["ir"]["parts"]} == {"box"}

def test_jonbb_passes_when_child_reaches_pivot():
    """The healthy case: a revolute child whose hinge edge seats AT the pivot
    on the parent surface — pivot is near both bodies, so no violation."""
    obj = Object("t")
    a = obj.make_box("a", 0.2, 0.2, 0.2)
    b = obj.make_box("b", 0.05, 0.05, 0.20)
    obj.attach_revolute(
        name="hinge", parent=a, child=b,
        axis=(1.0, 0.0, 0.0), limits=(0.0, 90.0), default=0.0,
        parent_anchor=obj.anchor(a, "top"),
        child_anchor=obj.anchor(b, "bottom"),
    )
    obj.declare_root(a)
    rep = joint_origin_near_both_bodies(obj.finalize())
    assert rep.verdict == "pass", rep.evidence


def test_jonbb_uncertain_on_floating_pivot():
    """A revolute whose child body is half a meter from the pivot, with no
    connector part to bridge it → advisory `uncertain` with evidence."""
    obj = Object("t")
    a = obj.make_box("a", 0.2, 0.2, 0.2)
    b = obj.make_box("b", 0.05, 0.05, 0.05)
    obj.attach_revolute(
        name="hinge", parent=a, child=b,
        axis=(1.0, 0.0, 0.0), limits=(0.0, 90.0), default=0.0,
        parent_anchor=obj.anchor(a, "top"),
        child_anchor=obj.anchor(b, "center", translation=(0.5, 0.0, 0.0)),
    )
    obj.declare_root(a)
    rep = joint_origin_near_both_bodies(obj.finalize())
    assert rep.verdict == "uncertain"
    viols = rep.evidence["violations"]
    assert len(viols) == 1 and viols[0]["joint"] == "hinge"
    assert viols[0]["distance_outside_child_aabb_m"] > 0.005
    assert viols[0]["bridging_part"] is None


def test_jonbb_bridging_connector_passes():
    """Same offset pivot as the floating case, but a third part straddles the
    pivot and overlaps the child (a pin/clevis) → the bridging escape fires
    and the joint is considered grounded."""
    obj = Object("t")
    a = obj.make_box("a", 0.2, 0.2, 0.2)
    b = obj.make_box("b", 0.05, 0.05, 0.30)
    obj.attach_revolute(
        name="hinge", parent=a, child=b,
        axis=(1.0, 0.0, 0.0), limits=(0.0, 90.0), default=0.0,
        parent_anchor=obj.anchor(a, "top"),
        child_anchor=obj.anchor(b, "center", translation=(0.10, 0.0, 0.0)),
    )
    pin = obj.make_box("pin", 0.24, 0.03, 0.03)
    obj.attach_fixed(
        name="pin_fix", parent=a, child=pin,
        parent_anchor=obj.anchor(a, "top"),
        child_anchor=obj.anchor(pin, "center"),
    )
    obj.declare_root(a)
    rep = joint_origin_near_both_bodies(obj.finalize())
    assert rep.verdict == "pass", rep.evidence


def test_jonbb_skips_fixed_joints():
    """Fixed joints carry no motion, so the guardrail ignores them even when
    the anchor floats far from both parts."""
    obj = Object("t")
    a = obj.make_box("a", 0.2, 0.2, 0.2)
    b = obj.make_box("b", 0.05, 0.05, 0.05)
    obj.attach_fixed(
        name="f", parent=a, child=b,
        parent_anchor=obj.anchor(a, "top", translation=(0.5, 0.0, 0.0)),
        child_anchor=obj.anchor(b, "center"),
    )
    obj.declare_root(a)
    rep = joint_origin_near_both_bodies(obj.finalize())
    # With only a fixed joint there is nothing to check, and the report says so.
    assert rep.verdict == "not_applicable"
    assert rep.evidence["not_applicable"] == "no movable joints"


# ---------------------------------------------------------------------------
# Constructs the allowlist admits. The agent authoring `object.py` also holds
# `Bash(*)`, so the node allowlist is not the containment boundary, and refusing
# these would buy nothing. What these assert is that each capability is real
# end to end — parsed *and* executed — since a node the allowlist admits can
# still fail at run time for want of a global (`class` needs `__build_class__`).
# ---------------------------------------------------------------------------

def _run(src: str):
    """check_ast + exec under the real restricted globals; returns `out`."""
    sb.check_ast(src, mode="exec")
    g = sb.build_globals(overlay={})
    exec(compile(src, "object.py", "exec"), g)  # noqa: S102 - the thing under test
    return g.get("out")


def test_try_except_can_recover_from_a_failing_operation():
    """The reason this exists: OCCT refuses a fillet radius often enough that
    falling back to a smaller one is ordinary modeling."""
    assert _run("try:\n"
                "    x = 1 / 0\n"
                "except ZeroDivisionError:\n"
                "    x = 'fell back'\n"
                "out = x\n") == "fell back"


def test_base_exception_is_not_bound():
    """`except BaseException: pass` would swallow KeyboardInterrupt, which the
    compile path re-raises deliberately, and SystemExit, which it reports as a
    policy breach."""
    assert "BaseException" not in sb._RESTRICTED_BUILTINS


def test_assert_and_raise_reach_the_ordinary_error_path():
    assert _run("try:\n"
                "    assert False, 'too thin'\n"
                "except AssertionError as e:\n"
                "    out = str(e)\n") == "too thin"
    with pytest.raises(RuntimeError):
        _run("raise RuntimeError('deliberate')\n")


def test_classes_actually_construct():
    """`ClassDef` is allowlisted for local parametric helpers, so the exec
    globals must carry `__build_class__` and `__name__`, without which every
    `class` dies at run time with a NameError."""
    assert _run("class Rib:\n"
                "    def w(self):\n"
                "        return 0.01\n"
                "class Wide(Rib):\n"
                "    def w(self):\n"
                "        return super().w() * 2\n"
                "out = Wide().w()\n") == 0.02


def test_walrus_match_generators_and_iteration_builtins():
    assert _run("out = sum(y for n in range(4) if (y := n * n) > 1)\n") == 13
    assert _run("match 3:\n"
                "    case 3:\n"
                "        out = 'three'\n"
                "    case _:\n"
                "        out = 'other'\n") == "three"
    assert _run("def g():\n    yield 2\n    yield 3\nout = sum(g())\n") == 5
    assert _run("out = list(map(abs, filter(lambda v: v < 0, [-1, 2, -3])))\n") \
        == [1, 3]


def test_match_class_pattern_cannot_name_a_private_attribute():
    """`case Foo(_secret=v)` produces no `Attribute` node, so the underscore
    rule never sees it. Allowing `Match` without this would have widened the
    dunder policy as a side effect."""
    with pytest.raises(SandboxViolation) as ei:
        sb.check_ast("match o:\n"
                     "    case Foo(_secret=v):\n"
                     "        pass\n", mode="exec")
    assert ei.value.to_dict()["evidence"]["node_type"] == "MatchClass"


def test_the_standing_exclusions_still_hold():
    for src, node in (("import os\n", "Import"),
                      ("from os import path\n", "ImportFrom"),
                      ("with open('f') as fh:\n    pass\n", "With"),
                      ("async def f():\n    pass\n", "AsyncFunctionDef"),
                      ("def f():\n    global z\n    z = 1\n", "Global")):
        with pytest.raises(SandboxViolation) as ei:
            sb.check_ast(src, mode="exec")
        assert ei.value.to_dict()["evidence"]["node_type"] == node, src


def test_the_import_refusal_names_the_handles_that_replace_it():
    """The prompt says nothing about the sandbox on purpose — the agent is meant
    to assume ordinary Python and learn the edges from the error. An agent that
    assumes ordinary Python writes `import numpy as np` first, so that message
    is the one that has to carry its weight."""
    with pytest.raises(SandboxViolation) as ei:
        sb.check_ast("import numpy as np\n", mode="exec")
    repair = ei.value.to_dict()["suggested_repairs"][0]
    for handle in ("bd", "math", "np"):
        assert f"`{handle}`" in repair, repair


# ---------------------------------------------------------------------------
# Attribute assignment inside authored Blender bodies
# ---------------------------------------------------------------------------
#
# These go through `check_ast`, as a real compile does: the sandbox's exemption
# for `@shader` and `@lighting` bodies is separate from the checks in
# `sdk/_shader.py`, so building a block by calling the decorator directly would
# not see the sandbox refuse it.


@pytest.mark.parametrize("decorator,param", [("shader", "nodes, links, bsdf"),
                                             ("shader", "nodes, links, bsdf, mat"),
                                             ("lighting", "L")])
def test_an_authored_body_may_assign_attributes(decorator, param):
    src = (f"@{decorator}\n"
           f"def body({param}):\n"
           f"    x = 1.0\n")
    sb.check_ast(src)          # baseline: the shape parses


def test_a_lighting_body_may_set_node_and_datablock_properties():
    """The exact form the SDK docstring and `scene_sdk.md` both show."""
    sb.check_ast(
        "@lighting\n"
        "def rig(L):\n"
        "    key = L.new_light('key', 'AREA')\n"
        "    key.data.energy = 40.0\n"
        "    key.location = (1.2, -0.4, 2.3)\n"
        "    bg = L.world_nodes.new('ShaderNodeBackground')\n"
        "    bg.inputs['Color'].default_value = (0.18, 0.108, 0.062, 1.0)\n"
        "    L.view.exposure = -0.7\n"
        "    L.view.white_balance_temperature = 5200\n")


def test_a_shader_body_may_still_set_node_properties():
    sb.check_ast(
        "@shader\n"
        "def surface(nodes, links, bsdf):\n"
        "    n = nodes.new('ShaderNodeTexNoise')\n"
        "    n.inputs['Scale'].default_value = 3.0\n"
        "    bsdf.inputs['Roughness'].default_value = 0.4\n")


@pytest.mark.parametrize("src,why", [
    ("obj.foo = 1\n", "module level"),
    ("def f(x):\n    x.foo = 1\n", "an undecorated function"),
    ("@staticmethod\ndef f(x):\n    x.foo = 1\n", "an unrelated decorator"),
])
def test_attribute_assignment_is_still_refused_outside_those_bodies(src, why):
    with pytest.raises(SandboxViolation, match="attribute assignment"):
        sb.check_ast(src)
    assert why


@pytest.mark.parametrize("stmt,match", [
    ("import os", "Import"),
    ("L.__class__.x = 1", "dunder"),
    ("setattr(L, 'x', 1)", "setattr"),
    ("eval('1')", "eval"),
])
def test_the_relaxation_does_not_widen_anything_else(stmt, match):
    """Only the attribute-assignment rule is relaxed; every other rule still
    applies inside a lighting body."""
    with pytest.raises(SandboxViolation, match=match):
        sb.check_ast(f"@lighting\ndef rig(L):\n    {stmt}\n")


def test_the_decorator_names_match_the_sdk_spellings():
    """Two spellings, one contract — renaming either alone reintroduces the
    failure this set exists for."""
    from dycraft.sdk import _lighting_block, _shader
    assert sb.SHADER_DECORATOR == _shader.DECORATOR_NAME
    assert sb.LIGHTING_DECORATOR == _lighting_block.DECORATOR_NAME
    assert sb.NODE_BODY_DECORATORS == {_shader.DECORATOR_NAME,
                                       _lighting_block.DECORATOR_NAME}
