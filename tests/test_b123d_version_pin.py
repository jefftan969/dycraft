"""The build123d pin, and the feature inventory it exists to protect.

DyCraft is frozen at exactly one build123d. This file guards both halves of
that decision:

  * the pin is enforced — a skewed interpreter is refused rather than producing
    subtly different geometry;
  * the thing the pin buys is present — `Face.axis_of_rotation` does not
    exist in 0.9.1, so under that release every cylindrical face would lose its
    axis, its direction and its bore classification, and silently wherever a
    broad handler swallowed the AttributeError.

The kernel-free tests run everywhere. The inventory tests need the pinned CAD
interpreter and drive it through the worker, as the rest of the suite does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.agent.cad import _cad_env as SA

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# `_b123d_version` is loaded BY PATH on purpose: that is exactly how
# `_adjudicate_runner` loads it, in an interpreter with no dycraft installed, so
# testing it this way also proves it stays standalone-loadable. A plain import
# would pull in the package and hide a regression that only bites in the child.
import importlib.util as _u

_spec = _u.spec_from_file_location(
    "_bv_pin", REPO / "agent" / "cad" / "_b123d_version.py")
BV = _u.module_from_spec(_spec)
_spec.loader.exec_module(BV)

# The feature-inventory helpers, by contrast, are an ordinary submodule.
from dycraft.agent.cad import _solid_compile as SC  # noqa: E402


# ---------------------------------------------------------------------------
# The pin itself (no CAD kernel needed)
# ---------------------------------------------------------------------------


def _in_cad(prog: str, *, timeout: float = 300) -> str:
    """Run `prog` in the CAD interpreter and return its stdout.

    On the warm pooled child, so these probes do not each re-pay
    `import build123d`. No dycraft bootstrap needed — the child has it bound.
    """
    out = W.run_cad_snippet(prog, timeout=timeout)
    assert out["ok"], (
        f"{out['message']}\n{out['traceback'] or ''}\n"
        f"stderr={out['stderr'][-2000:]}")
    return out["stdout"]


def test_environment_spec_agrees_with_the_code_pin():
    """A spec file that drifts from the runtime check is worse than none."""
    spec = (REPO / "environment-b123d.yml").read_text()
    assert f"build123d=={BV.REQUIRED_BUILD123D}" in spec


def test_version_check_raises_a_structured_error(monkeypatch):
    monkeypatch.setattr(BV, "installed_build123d_version", lambda: "0.9.1")
    with pytest.raises(BV.SolidCompileError) as exc:
        BV.check_build123d_version()
    assert exc.value.kind == "build123d_version_mismatch"
    assert "0.11.1" in exc.value.message and "0.9.1" in exc.value.message


def test_the_raised_kind_is_one_the_agent_gets_repairs_for(monkeypatch):
    """Close the loop between the producer of this error and its repair text.

    `tools._worker_failure_repairs` dispatches on `error_kind`, which the worker
    child copies from `SolidCompileError.kind` — but the child ALSO writes the
    class name `Build123dVersionMismatch` into `build_error.error` right beside
    it. Keying the repair table on that class name instead looks correct and
    yields no repairs at all, so a skewed CAD interpreter sends the agent off
    rewriting geometry that was never wrong.

    The parametrized table test in `test_object_worker_routing.py` cannot catch
    that: it feeds the table's own keys back in as `error_kind`, so it agrees
    with itself whatever either side is spelled. This one asks the real producer.
    """
    from dycraft.agent.layers import tools as T

    monkeypatch.setattr(BV, "installed_build123d_version", lambda: "0.9.1")
    with pytest.raises(BV.SolidCompileError) as exc:
        BV.check_build123d_version()
    assert exc.value.kind in T._WORKER_REPAIRS, (
        f"check_build123d_version raises kind={exc.value.kind!r}, which "
        f"_WORKER_REPAIRS does not answer; the agent would get no repairs")


def test_version_check_reports_a_missing_install(monkeypatch):
    monkeypatch.setattr(BV, "installed_build123d_version", lambda: None)
    with pytest.raises(BV.SolidCompileError) as exc:
        BV.check_build123d_version()
    assert "no build123d" in exc.value.message


def test_non_strict_check_does_not_raise(monkeypatch):
    monkeypatch.setattr(BV, "installed_build123d_version", lambda: "0.9.1")
    assert BV.check_build123d_version(strict=False) == "0.9.1"


# ---------------------------------------------------------------------------
# OCP — bound in `object.py`'s globals, so its absence is the same class of
# fault as a build123d skew and has to be reported the same way.
# ---------------------------------------------------------------------------

def test_ocp_is_present_in_this_environment():
    """The gate's premise: OCP ships under build123d, so a CAD env has both.

    If this ever fails, the environment is broken rather than the test — the
    namespace binds `OCP` unconditionally, so `build_namespace` would raise
    ImportError before any authored line ran.
    """
    assert BV.ocp_is_importable()


def test_a_missing_ocp_raises_a_kind_the_agent_gets_repairs_for(monkeypatch):
    """Same contract `build123d_version_mismatch` has, for the same reason.

    A kind the worker can carry but `_WORKER_REPAIRS` cannot answer reaches the
    agent with an empty `suggested_repairs`, which reads as "nothing to be
    done" — and this one must additionally say *environment*, or the agent
    spends repair attempts rewriting geometry that was never the problem.
    """
    from dycraft.agent.cad import object_worker as OW
    from dycraft.agent.layers import tools as T

    monkeypatch.setattr(BV, "ocp_is_importable", lambda: False)
    with pytest.raises(BV.SolidCompileError) as exc:
        BV.check_ocp_available()
    assert exc.value.kind == "ocp_missing"
    assert exc.value.kind in OW.FAILURE_KINDS, (
        "the child carries this kind out on SolidCompileError.kind, so "
        "worker_failure would reject it")
    assert exc.value.kind in T._WORKER_REPAIRS, (
        "check_ocp_available raises a kind _WORKER_REPAIRS does not answer")
    assert "no importable OCP" in exc.value.message


def test_a_missing_ocp_is_not_reported_as_a_geometry_fault(monkeypatch):
    """The repair text must send the author at the environment, not the shape.

    `worker_crash` says "simplify the operation that produced it", which is
    right for a kernel death and wrong here. The two are one table apart.
    """
    from dycraft.agent.layers import tools as T

    advice = " ".join(T._WORKER_REPAIRS["ocp_missing"]).lower()
    assert "environment fault" in advice
    assert "not a fault in your source" in advice


def test_non_strict_ocp_check_does_not_raise(monkeypatch):
    monkeypatch.setattr(BV, "ocp_is_importable", lambda: False)
    assert BV.check_ocp_available(strict=False) == "(not installed)"


def test_the_ocp_version_is_read_for_the_record_not_gated_on(monkeypatch):
    """Presence is the gate; the version is a label.

    OCP ships as `cadquery-ocp`, `cadquery-ocp-novtk` and others, so pinning a
    distribution version would fail on a legitimate environment. An OCP with no
    metadata at all still authorises the run.
    """
    monkeypatch.setattr(BV, "installed_ocp_version", lambda: None)
    assert BV.check_ocp_available() == "(unknown version)"


# ---------------------------------------------------------------------------
# Narrow exception handling — the bug class, not just the bug
# ---------------------------------------------------------------------------

def test_api_misuse_is_not_swallowed():
    """AttributeError from a moved API must surface, not vanish.

    Under a release without `Face.axis_of_rotation`, a bare handler would
    catch the AttributeError and the column would silently empty.
    """
    class _NoAxis:
        @property
        def axis_of_rotation(self):
            raise AttributeError("'Face' object has no attribute 'axis_of_rotation'")

    with pytest.raises(SC.FaceInspectionError) as exc:
        SC.face_axis(_NoAxis())
    assert "axis_of_rotation" in str(exc.value)


def test_a_face_with_no_occt_payload_is_api_misuse_not_a_missing_axis():
    """Nothing to inspect is not the same claim as nothing to report.

    Both are answered by returning from `face_axis`, so the distinction has to
    be in *how*: a real face whose surface is not a revolution yields None (see
    the kernel-backed test below), while an object that cannot be inspected at
    all raises. Collapsing the two would let a malformed face empty the
    cylinder column silently, which is the defect this whole section exists for.
    """
    class _NoPayload:
        @property
        def axis_of_rotation(self):
            raise ValueError("surface is not a cylinder")
        wrapped = None                      # no OCCT face to fall back to

    with pytest.raises(SC.FaceInspectionError):
        SC.face_axis(_NoPayload())


def test_face_radius_surfaces_api_misuse():
    class _Bad:
        @property
        def radius(self):
            raise TypeError("'float' object is not callable")

    with pytest.raises(SC.FaceInspectionError):
        SC.face_radius(_Bad())


# ---------------------------------------------------------------------------
# The feature inventory, under the real pinned kernel
# ---------------------------------------------------------------------------

_needs_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs the pinned build123d interpreter")


@_needs_cad
def test_the_configured_interpreter_carries_the_pinned_version():
    """The interpreter DyCraft will actually launch must be the pinned one.

    Through `probe_build123d_version`, which is what the CLI preflight asks, so
    this covers the preflight's answer and not a second copy of the question.
    """
    assert SA.probe_build123d_version() == SC.REQUIRED_BUILD123D


@_needs_cad
def test_a_null_shape_is_rejected_as_empty_not_as_an_assertion():
    """`a & b` on solids that do not touch must say so.

    Like `Face.axis_of_rotation`, an API whose *shape* differs between releases
    and fails silently rather than loudly.
    Under 0.11.1 `Shape.wrapped` is a property that asserts, so reading it on a
    null shape raises a message-less `AssertionError` instead of returning
    `None` — and `hasattr` does not shield it, since that only swallows
    `AttributeError`.

    Uncaught, the `AssertionError` escapes `add_part`,
    `_classify_exec_failure` sees a build123d frame, and the agent gets a
    `GeometryKernelError` whose first suggested repair — normally the kernel's
    own sentence, the actionable part — is the empty string.
    """
    prog = (
        "import build123d as bd\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "empty = bd.Box(1, 1, 1) & bd.Pos(9, 0, 0) * bd.Box(1, 1, 1)\n"
        "try:\n"
        "    sc.validate_shape(empty, name='p')\n"
        "except sc.SolidCompileError as e:\n"
        "    print(e.kind)\n"
    )
    stdout = _in_cad(prog, timeout=300)
    assert stdout.strip() == "empty_shape", (
        "an empty boolean must reach the branch whose message names it")


@_needs_cad
def test_a_genuinely_unsupported_surface_returns_none_not_an_error():
    """A real face that is not a surface of revolution is skippable.

    Driven through the kernel rather than a stub, because the behavior under
    test *is* the second source declining: OCCT raises its own exception from
    `.Cylinder()` on a plane, and that has to read as "no axis here" rather than
    as a broken API. A hand-rolled fake cannot produce that exception, so it
    would be testing the mock instead.
    """
    prog = (
        "import build123d as bd\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "face = bd.Box(1, 1, 1).faces()[0]\n"
        "print(sc.face_axis(face), sc.face_radius(face))\n"
    )
    stdout = _in_cad(prog, timeout=300)
    assert stdout.strip() == "None None", (
        "a planar face has neither an axis of revolution nor a radius, and "
        "asking must not raise")


@_needs_cad
@pytest.mark.expensive
def test_hinge_bores_report_axis_and_concavity(tmp_path):
    """The articulated door fixture's two 6 mm hinge bores, end to end.

    Without the pinned kernel's API the six cylindrical faces would carry no
    axis and no concavity at all. The assertion that matters is not "some
    cylinders exist" but that the two BORES are identified as concave and
    carry a usable axis — that is the joint-axis candidate the CAD
    representation exists to provide.
    """
    source = (FIXTURES / "m1_live_shape_articulated.py").read_text()
    out = W.run_object_source(source, run_dir=tmp_path)
    assert out["ok"], out.get("build_error") or out.get("message")

    door = next(p for p in out["ir"]["parts"] if p["name"] == "door")
    cylinders = door["params"]["shape"]["cylinders"]
    assert len(cylinders) == 6, cylinders

    # Every cylindrical face must carry a full description now.
    for c in cylinders:
        assert "radius" in c, c
        assert "axis_point" in c and "axis_dir" in c, c
        assert "concave" in c, c

    bores = [c for c in cylinders if c["concave"] is True]
    assert len(bores) == 2, f"expected the two hinge bores, got {bores}"
    for bore in bores:
        # 6 mm bore -> 3 mm radius, drilled through the knuckle along Y.
        assert bore["radius"] == pytest.approx(0.003, abs=1e-6)
        assert abs(bore["axis_dir"][1]) == pytest.approx(1.0, abs=1e-6)
        assert bore["axis_point"][0] == pytest.approx(0.245, abs=1e-3)
    # The knuckles they are drilled through are convex, not bores.
    assert sum(1 for c in cylinders if c["concave"] is False) == 4


@_needs_cad
@pytest.mark.expensive
def test_the_ir_shape_table_carries_exactly_the_documented_fields(tmp_path):
    """The table an author probes, end to end through the real kernel.

    Two failure modes, opposite directions: a documented field the table lacks
    is a KeyError mid-probe, and an undocumented one is payload carried in
    every IR forever.
    """
    source = (FIXTURES / "m1_live_shape_articulated.py").read_text()
    out = W.run_object_source(source, run_dir=tmp_path)
    assert out["ok"], out.get("build_error") or out.get("message")

    for part in out["ir"]["parts"]:
        table = part["params"]["shape"]
        # No `repairs` or `warnings`: they describe what one compile *did*
        # (which vertices were welded, whether the render mesh was coarsened),
        # not what the part *is*, and the IR is hashed whole, so carrying them
        # would make a diagnostic's prose part of the object's identity. They
        # travel in the compile report's `geometry_notes`; see
        # `sdk/_object_kernel.part_shape_table`.
        assert set(table) == {
            "volume", "area", "center", "inertia_diag", "fill_ratio",
            "face_census", "topo", "cylinders", "qc",
        }, (part["name"], sorted(table))
        # qc is None only when the QC pass itself failed; these solids are fine.
        assert table["qc"]["bounds_volume"] is True
        assert table["qc"]["n_shells"] == 1


@_needs_cad
@pytest.mark.expensive
def test_feature_inventory_notes_stay_clean(tmp_path):
    """No 'HARNESS DEFECT' note means no API skew went unnoticed."""
    source = (FIXTURES / "m1_live_shape_single.py").read_text()
    out = W.run_object_source(source, run_dir=tmp_path)
    assert out["ok"]
    blob = json.dumps(out["ir"])
    assert "HARNESS DEFECT" not in blob


# ---------------------------------------------------------------------------
# Inertia round-off
# ---------------------------------------------------------------------------

def test_a_round_off_negative_inertia_is_clamped_not_refused():
    """A moment a hair below zero is round-off, not an error.

    Spline-bounded solids — a lofted visor, a swept hose, a shelled saddle —
    integrate their mass properties numerically, and a thin one can come back
    at -1e-11. Refusing it would stop a build with advice the author cannot act
    on.
    """
    from dycraft.sdk import _primitives as P

    # Measured, not invented. Three spline-bounded thin solids built through
    # the pinned kernel and integrated with `VolumePropertiesGK_s` return
    # smallest principal moments of -2.85e-55, -5.15e-26 and -2.02e-28 against
    # largest moments of 8.6e-8 and 1.5e-8 — ratios of 3e-48 to 1e-18, since
    # the moment being integrated is a near-total cancellation.
    for negative, largest in ((-2.85e-55, 8.6e-8),
                              (-5.15e-26, 1.5e-8),
                              (-2.02e-28, 1.5e-8)):
        diag = P._clamp_inertia_roundoff((negative, largest / 2, largest), "s")
        assert diag[0] == 0.0, f"{negative} against {largest} was refused"
        assert diag[1:] == (largest / 2, largest), "sound components changed"


def test_a_negative_the_size_of_the_tensor_still_refuses():
    """The other half of the split, and the reason it is not just a clamp.

    A negative that is a significant fraction of the tensor means the
    integration did not converge. Zeroing it would hand the physics engine a
    fabricated mass property, which is worse than a refusal — so this one
    still raises, and now says which case it is.
    """
    from dycraft.sdk import _primitives as P

    with pytest.raises(ValueError) as exc:
        P._clamp_inertia_roundoff((-4.0e-6, 5.0e-6, 8.0e-6), "solid")
    # "which case it is" is carried by the round-off clause: this negative is
    # too large a fraction of the tensor to be round-off, as against the
    # clamped case above. The message never uses the words "did not converge"
    # — that is this docstring's diagnosis, not its text.
    assert "exceeding the round-off allowance" in str(exc.value)
    assert "0.5" in str(exc.value), "the message should quote the ratio"


def test_the_clamp_tolerance_is_relative_to_the_tensor():
    """Absolute would mean something different at every scale.

    A 3 m x 0.1 mm rod's smallest principal moment is 2.2e-9 of its largest,
    so a fixed epsilon either misses round-off on large parts or zeroes real
    values on small ones. The threshold is a millionth of the largest moment,
    which clears every negative the pinned kernel can be made to produce by
    twelve orders and is still far below anything a rigid body notices.
    """
    from dycraft.sdk import _primitives as P

    # Same ratio, two scales: both clamp.
    assert P._clamp_inertia_roundoff((-1.0e-13, 1.0e-6), "s")[0] == 0.0
    assert P._clamp_inertia_roundoff((-1.0e-7, 1.0), "s")[0] == 0.0
    # And the same absolute value refuses when the tensor is small enough.
    with pytest.raises(ValueError):
        P._clamp_inertia_roundoff((-1.0e-7, 1.0e-6), "s")
