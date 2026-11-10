"""Three failure kinds, three answers.

`compile` reports a failure in one of `sandbox_violation`, `build_error` or
`verifier_reports`, and the agent reads that field to decide what to change.
A policy refusal and a bug in the agent's own program are different failures
needing opposite responses, and a kernel failure's own advice ("try a smaller
value or use max_fillet()") belongs in `suggested_repairs`, not only in
`evidence.traceback`.

Three classes:

  * `SandboxViolation` — the allowlist or an injected guard refused. A policy
    decision; the program must change shape.
  * `SourceRuntimeError` — ordinary Python raised. A bug in the program.
  * `GeometryKernelError` — build123d/OCCT could not build the solid. Valid
    Python, valid call, impossible geometry; the operands must change.

The last two are `BuildError` subclasses, so they land in `build_error`
alongside the SDK's own validation failures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.agent.layers.compile import build_ir_with_layer
from dycraft.agent.layers.layer import object_layer_config
from dycraft.sdk._errors import (BuildError, GeometryKernelError,
                                 SourceRuntimeError)

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="no build123d interpreter")


class _Ctx:
    def __init__(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.step = 0


def _build(src: str) -> dict:
    return build_ir_with_layer(src, layer_cfg=object_layer_config(), ctx=_Ctx())


# ---------------------------------------------------------------------------
# Ordinary Python bugs are not sandbox violations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src,exc", [
    # A name the program never defined: a typo.
    ("obj = Object('t')\nq = milk_fridge\nir = obj.finalize()", "NameError"),
    ("x = 1 / 0\nir = None", "ZeroDivisionError"),
    ("x = 'a' + 1\nir = None", "TypeError"),
    ("xs = []\ny = xs[3]\nir = None", "IndexError"),
])
def test_a_source_bug_is_a_build_error_not_a_sandbox_violation(src, exc):
    out = _build(src)
    assert out["sandbox_violation"] is None, (
        "a typo is not an attempt to escape the sandbox")
    be = out["build_error"]
    assert be["code"] == "SourceRuntimeError"
    assert exc in be["summary"]
    assert be["evidence"]["exception"] == exc


def test_a_source_bug_reports_the_agents_own_failing_line():
    """The line number is the whole point; `lineno: 0` sends the author nowhere."""
    out = _build("obj = Object('t')\n\nq = milk_fridge\nir = obj.finalize()")
    ev = out["build_error"]["evidence"]
    assert ev["lineno"] == 3
    assert ev["src_line"].strip() == "q = milk_fridge"


def test_a_source_bug_leads_with_the_interpreter_message():
    out = _build("obj = Object('t')\nq = milk_fridge\nir = obj.finalize()")
    repairs = out["build_error"]["suggested_repairs"]
    assert "milk_fridge" in repairs[0]
    joined = " ".join(repairs)
    assert "BuildError" not in joined, (
        "the repair asks the source to raise BuildError; `raise` is not "
        "in the allowlist, so that can never be done")


def test_the_traceback_is_still_carried():
    """Classifying the failure must not drop the traceback, the evidence that
    makes it diagnosable."""
    ev = _build("x = 1 / 0\nir = None")["build_error"]["evidence"]
    assert "ZeroDivisionError" in ev["traceback"]


# ---------------------------------------------------------------------------
# SDK errors are untouched
# ---------------------------------------------------------------------------

def test_sdk_validation_errors_keep_their_own_codes():
    """The new classes are a fallback for what escaped, not a replacement for
    the SDK's structured errors."""
    out = _build("obj = Object('t')\nir = obj.finalize()")
    assert out["build_error"]["code"] == "RootNotDeclared"


def test_a_genuine_sandbox_violation_still_reports_as_one():
    out = _build("import os\nir = None")
    assert out["build_error"] is None
    assert out["sandbox_violation"]["code"] == "SandboxViolation"


# ---------------------------------------------------------------------------
# Kernel failures, through the real CAD interpreter
# ---------------------------------------------------------------------------

# A fillet radius larger than the stock it is applied to.
_FILLET_TOO_BIG = """
st = Box(0.10, 0.10, 0.02)
st = fillet(st.edges().filter_by(Axis.X), 0.030)
obj = Object("k")
p = obj.add_part("b", st, material=Material(color=(128, 128, 128)))
obj.declare_root(p)
ir = obj.finalize()
"""


@_needs_cad
@pytest.mark.expensive
def test_an_occt_failure_is_a_geometry_error_not_a_sandbox_violation(tmp_path):
    raw = W.run_object_source(_FILLET_TOO_BIG, run_dir=tmp_path)
    assert not raw["ok"]
    assert raw["sandbox_violation"] is None, (
        "a fillet the kernel cannot build is not a policy breach")
    be = raw["build_error"]
    assert be["code"] == "GeometryKernelError"
    assert be["evidence"]["reason"] == "kernel_operation_failed"


@_needs_cad
@pytest.mark.expensive
def test_the_kernels_own_advice_reaches_the_agent(tmp_path):
    """build123d says "try a smaller value or use max_fillet()". That sentence
    is the fix, so it reaches the agent outside the traceback."""
    raw = W.run_object_source(_FILLET_TOO_BIG, run_dir=tmp_path)
    repairs = raw["build_error"]["suggested_repairs"]
    assert "max_fillet" in repairs[0] or "smaller value" in repairs[0]
    assert "line 3" in " ".join(repairs), "point at the agent's own call site"


@_needs_cad
@pytest.mark.expensive
def test_a_kernel_failure_and_a_source_bug_are_told_apart(tmp_path):
    """Both are exceptions escaping exec; only the traceback's frames
    distinguish them, so check the discrimination and not just one side."""
    bug = W.run_object_source(
        "obj = Object('t')\nq = milk_fridge\nir = obj.finalize()",
        run_dir=tmp_path)
    assert bug["build_error"]["code"] == "SourceRuntimeError"

    kernel = W.run_object_source(_FILLET_TOO_BIG, run_dir=tmp_path)
    assert kernel["build_error"]["code"] == "GeometryKernelError"


# ---------------------------------------------------------------------------
# `expect_solids`: the author states the intended body count, because the
# compiler can only check ">= 1". A fuse whose operands do not touch returns a
# valid two-body Compound and registers silently, so without a stated count
# every author writes its own `_one` / `_single` guard.
# ---------------------------------------------------------------------------

_TWO_BODIES = ("Box(.1,.1,.1) + Pos(0,0,.5)*Box(.1,.1,.1)")


def _one_part_source(shape_expr: str, kwargs: str = "") -> str:
    return (
        "W = Material(color=(174, 113, 59))\n"
        "obj = Object('t')\n"
        f"a = obj.add_part('a', {shape_expr}, material=W{kwargs})\n"
        "obj.declare_root(a)\n"
        "obj.add_check('A test body.')\n"
        "ir = obj.finalize()\n"
    )


@_needs_cad
def test_expect_solids_accepts_the_intended_count():
    assert _build(_one_part_source("Box(.1,.1,.1)",
                                   ", expect_solids=1"))["ok"]
    assert _build(_one_part_source(_TWO_BODIES, ", expect_solids=2"))["ok"]


@_needs_cad
def test_omitting_expect_solids_still_accepts_several_bodies():
    """The default must not change: a Compound of rigidly-fixed pieces — a
    hinge's two leaves — is a legitimate part, so this cannot become a
    blanket rejection."""
    assert _build(_one_part_source(_TWO_BODIES))["ok"]


@_needs_cad
def test_a_failed_fuse_is_caught_with_the_measured_count_in_evidence():
    out = _build(_one_part_source(_TWO_BODIES, ", expect_solids=1"))
    assert not out["ok"]
    be = out["build_error"]
    assert be["code"] == "InvalidPrimitive"
    assert be["evidence"]["reason"] == "solid_count_mismatch"
    assert be["evidence"]["expected"] == 1
    assert be["evidence"]["measured"] == 2
    # The first repair is what the agent acts on, so it has to name the cause
    # rather than restate the count.
    assert "fuse" in be["suggested_repairs"][0]


@_needs_cad
@pytest.mark.parametrize("bad", ["0", "-1", "True", "1.0", "'1'"])
def test_expect_solids_rejects_a_nonsense_count(bad):
    """`True` is the one worth pinning: `bool` is an `int` subclass, so
    `expect_solids=True` would otherwise mean "exactly one solid" by accident.
    """
    out = _build(_one_part_source("Box(.1,.1,.1)", f", expect_solids={bad}"))
    assert not out["ok"]
    assert out["build_error"]["evidence"]["param"] == "expect_solids"


def test_both_new_classes_are_build_errors():
    """So every existing consumer that branches on BuildError keeps working —
    `fixed_source`, the repair loop, and the tool result all read that field."""
    assert issubclass(SourceRuntimeError, BuildError)
    assert issubclass(GeometryKernelError, BuildError)


# ---------------------------------------------------------------------------
# The diagnosis table
#
# A raw modeling error names a symptom, not a cause: "brep_api: command not
# done" says an operation refused, not that the fillet radius was larger than
# the surrounding faces can absorb. `_geometry_qc.failure_hint` maps the
# recognizable ones to the move that resolves them, and this is the seam where
# the mapping reaches the agent — first in the repair list, since it is more
# actionable than the kernel's own wording or the generic advice.
# ---------------------------------------------------------------------------

def test_a_recognized_kernel_message_leads_with_its_diagnosis():
    """Driven through `_classify_exec_failure` directly.

    The kernel messages the table recognizes come from OCCT, and provoking a
    real one costs a CAD subprocess (see the expensive fillet case above). What
    needs pinning here is cheaper and more specific: that a recognized message
    puts the diagnosis first and keeps the raw text.
    """
    from dycraft.agent.layers.compile import _classify_exec_failure

    exc = RuntimeError("BRep_API: command not done")
    err = _classify_exec_failure(exc, "ir = None\n", "object.py")
    repairs = err.to_dict()["suggested_repairs"]
    assert "subtracting an inner body from an outer one" in repairs[0], repairs
    assert any("command not done" in r for r in repairs[1:]), repairs


def test_an_unrecognized_message_adds_no_diagnosis():
    """Silence is the right answer when the table has nothing to say.

    A hint invented for an unmatched message would be advice about a cause
    nobody established, which is worse than the raw error.
    """
    out = _build("obj = Object('t')\nq = milk_fridge\nir = obj.finalize()")
    repairs = out["build_error"]["suggested_repairs"]
    assert "milk_fridge" in repairs[0], (
        "the interpreter's own message must still lead when nothing matched")


def test_the_hint_table_is_reachable_from_the_compiler_surface():
    """`object_builder` consults it through the injected compiler, not by
    importing the agent layer, so the SDK -> agent direction stays one-way."""
    from dycraft.agent.cad import _solid_compile as SC

    assert SC.failure_hint("BRep_API: command not done") is not None
    assert SC.failure_hint("something nobody has seen before") is None


# ---------------------------------------------------------------------------
# The geometry-kind registry
#
# A `SolidCompileError.kind` that is a free string, with no registry, no
# validation and no consumer, reaches the agent only as an `evidence.reason`
# label, gives every kind the same advice, and lets a kind be raised for a
# failure it does not describe. These pin the properties that make the
# classification worth
# keeping — it selects the advice, and it cannot be a typo.
# ---------------------------------------------------------------------------

def test_every_kind_answers_what_the_author_should_do():
    """A kind with no advice reaches the agent as 'nothing to be done'.

    Same rule `test_object_worker_routing` applies to `FAILURE_KINDS`: the table
    of kinds and the table of repairs are one table, so they cannot drift.
    """
    from dycraft.agent.cad import _b123d_version as V

    for kind in V.SOLID_COMPILE_KINDS:
        assert V.kind_repairs(kind), f"kind {kind!r} carries no repair advice"
        for line in V.kind_repairs(kind):
            assert isinstance(line, str) and line.strip()


def test_an_unregistered_kind_is_refused_at_construction():
    """A kind that selects advice must not be allowed to be a typo."""
    from dycraft.agent.cad import _b123d_version as V

    with pytest.raises(ValueError, match="unknown SolidCompileError kind"):
        V.SolidCompileError("tesselation_failure", "one 'l' too few")


def test_every_kind_the_compiler_raises_is_registered():
    """Catches a new `raise SolidCompileError(...)` with an unlisted kind.

    Reads the source rather than exercising every failure path: several kinds
    need a CAD kernel to provoke, and the property under test is a static one.
    """
    import re
    from pathlib import Path

    from dycraft.agent.cad import _b123d_version as V
    from dycraft.agent.cad import _solid_compile

    text = Path(_solid_compile.__file__).read_text(encoding="utf-8")
    raised = set(re.findall(r'SolidCompileError\(\s*"([a-z_]+)"', text))
    assert raised, "no SolidCompileError kinds found — did the call shape change?"
    unregistered = raised - set(V.SOLID_COMPILE_KINDS)
    assert not unregistered, f"unregistered kinds raised: {sorted(unregistered)}"


def test_not_a_shape_is_not_reported_as_a_tessellation_failure():
    """An argument that is not a topology object was never tessellated.

    Calling it a tessellation failure would send the author to a phantom, and
    `design_solid_authoring.md`'s error table gives it its own kind.
    """
    from dycraft.agent.cad import _solid_compile as SC

    with pytest.raises(SC.SolidCompileError) as caught:
        SC.validate_shape(object(), name="x")
    assert caught.value.kind == "not_a_shape"
    assert SC.kind_repairs("not_a_shape")


def test_the_kind_selects_the_advice_the_agent_reads():
    """The kind is computed to choose the advice.

    Two different kinds must produce different advice, or the classification
    is computed and discarded.
    """
    from dycraft.sdk.object_builder import _solid_failure_repairs
    from dycraft.agent.cad import _solid_compile as SC

    empty = SC.SolidCompileError("empty_shape", "no overlap")
    mesh = SC.SolidCompileError("invalid_mesh", "non-finite")
    a = _solid_failure_repairs(SC, empty)
    b = _solid_failure_repairs(SC, mesh)
    assert a and b and a != b
    assert "overlap" in " ".join(a).lower()
    assert "non-finite" in " ".join(b).lower()


# ---------------------------------------------------------------------------
# Hints at the point of failure
#
# Most geometry-kernel failures are met with wrong or absent advice rather
# than hard geometry. These pin the advice.
# ---------------------------------------------------------------------------

def test_the_fillet_hint_contradicts_the_kernels_own_advice():
    """`compile.py` forwards the raw kernel message into `suggested_repairs`,
    and that message says "try a smaller value".

    Measured, that is often wrong: a fillet can succeed at one radius and fail
    at every smaller one (on a helmet shell, 3.85 mm succeeds where 3, 2 and
    1 mm fail), so an agent that halves the radius fails again. The hint ships
    alongside that message and has to argue against it explicitly, or the
    author reads two pieces of advice and follows the wrong one.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    hint = gq.failure_hint(
        "ValueError: Failed creating a fillet with radius of 0.005, try a "
        "smaller value or use max_fillet()")
    assert hint, "the canonical fillet message must match a hint"
    low = hint.lower()
    assert "not monotone" in low, "the hint must name the reason"
    assert "max_fillet" in hint and "presumes" in low, (
        "max_fillet binary-searches, so it cannot answer here; the hint used "
        "to recommend it")
    # The three moves that do work, in place of halving.
    assert "r1 + r2" in hint, "the pairwise face-width rule"
    assert "bisect the selection" in low, "find the binding edge"


def test_stdfail_notdone_names_every_cause_measured():
    """`StdFail_NotDone` names no cause, so the hint is the only place one
    exists. Five operations are known to raise it, and the hint names all five.

    A missing cause is not neutral — the author reads the causes that ARE
    listed and rebuilds a wall or shrinks a radius when their actual mistake
    is a revolve axis or a duplicate point.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    hint = gq.failure_hint("StdFail_NotDone: BRep_API: command not done")
    low = hint.lower()
    assert "crosses the axis" in low
    assert "not in the profile's plane" in low or "not in" in low
    assert "duplicate consecutive point" in low
    assert "periodic-spline sections" in low
    assert ".shell()" in hint and ".fillet()" in hint


def test_an_untranslated_occt_internal_still_gets_domain_meaning():
    """`Standard_NoSuchObject: NCollection_DataMap::Find` is a raw OCCT map
    error build123d does not translate, so the author received a C++ internal
    with no domain meaning and no hint at all.

    Recorded cause: `loft` to an apex `Vertex` over many sections on a curved,
    twisting spine — the same stack raises `StdFail_NotDone` at 5 sections and
    this at 9. `loft(sections)` without the vertex succeeds.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    hint = gq.failure_hint("Standard_NoSuchObject: NCollection_DataMap::Find")
    assert hint, "an untranslated kernel internal must still carry advice"
    low = hint.lower()
    assert "vertex" in low and "loft" in low
    assert "without the vertex" in low, "the measured workaround"


def test_the_periodic_loft_advice_does_not_condemn_a_working_idiom():
    """Periodic-spline sections loft; the advice must not say they cannot.

    They are the normal idiom and loft reliably. The trigger is two periodic
    sections that DIFFER IN SHAPE: the same section at different sizes, or an
    identical section repeated, lofts fine.

    Advice that pushes an author off a working construction is worse than
    silence, which is what this module's own docstring says about inventing
    causes.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    hint = gq.failure_hint("StdFail_NotDone: BRep_API: command not done")
    low = hint.lower()
    assert "differing shape" in low, "the qualifier is the whole claim"
    assert "normal idiom" in low or "do not abandon" in low, (
        "the hint has to say periodic sections are fine, or an author reads "
        "it as a prohibition")
    assert "repeated or scaled sections can loft correctly" in low
    assert "matching seam vertices" in low, "retain the conditional repair"


def test_the_map_error_needle_is_specific_to_the_recorded_cause():
    """`NCollection_DataMap` — N-collection. A needle that matches nothing
    leaves the entry firing on its generic `standard_nosuchobject` needle,
    which shadows the selector-matched-nothing entry (`failure_hint` returns
    the first match) and hands loft-to-`Vertex` advice to any OCCT map error:
    a `sweep` raising `NCollection_IndexedDataMap::FindFromKey` would be told
    to "drop the apex" it does not have.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    specific = gq.failure_hint(
        "Standard_NoSuchObject: NCollection_DataMap::Find")
    assert specific and "apex" in specific.lower(), (
        "the recorded loft-to-Vertex cause must still be reachable")

    for other in ("Standard_NoSuchObject: NCollection_IndexedDataMap::FindFromKey",
                  "Standard_NoSuchObject: no such object",
                  "IndexError: list index out of range"):
        hint = gq.failure_hint(other)
        assert hint, other
        assert "apex" not in hint.lower(), (
            f"{other!r} was given loft-to-Vertex advice it has no basis for")
        assert "selector matched nothing" in hint.lower(), other


def test_the_downstream_subtraction_error_points_at_the_earlier_line():
    """This exception lands one iteration AFTER the boolean that annihilated
    the part, so the line the compiler blames is not the line that broke."""
    from dycraft.agent.cad import _geometry_qc as gq

    hint = gq.failure_hint(
        "ValueError: Dimensions of objects to subtract from are inconsistent")
    assert hint, "the most valuable uncovered signature in the corpus"
    assert "EARLIER line" in hint
    assert "solids()" in hint, "name the check that finds it"
