"""Live build123d Shape registration — the parts that need no CAD kernel.

`Object.add_part` takes a live OCCT Shape, so exercising it end to end needs an
interpreter with build123d. Those cases live in `test_object_worker.py`, which
drives the worker as a subprocess.

What is testable *without* the kernel is everything guarding the seam — the
failure modes that would otherwise pass silently:

  * the capability probe answers for this interpreter, including when build123d
    is installed but broken, and the no-CAD overlay stays `Object` + `Material`;
  * `add_part` refuses cleanly where no compiler is installed, rather than
    raising ImportError;
  * registered snapshots never enter the serializable `_parts` records;
  * the injected build123d namespace cannot shadow the SDK;
  * `compile_entry` reports a stage that broke as broken, and never passes a
    coarsened render mesh off as the requested one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dycraft.agent.cad import _b123d_namespace as NS
from dycraft.agent.layers import layer as L
from dycraft.sdk import Object, _mesh_assets as _ma
from dycraft.sdk._errors import InvalidPrimitive
from dycraft.sdk.ir import canonical_json_bytes
from dycraft.sdk.object_builder import _reset_object_singleton


class _Ctx:
    """Minimal stand-in for ToolContext — the overlay only reads `run_dir`."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)


@pytest.fixture()
def ctx(tmp_path):
    _reset_object_singleton()
    _ma.clear_active_mesh_session()
    _ma.clear_active_solid_compiler()
    yield _Ctx(tmp_path)
    L.reset_object_layer_state()


# ---------------------------------------------------------------------------
# The capability probe
#
# The overlay asks whether this interpreter can construct build123d geometry,
# and installs the authoring surface only if it can. These tests pin the probe
# and the no-CAD branch it selects — the branch the main DyCraft env takes.
# ---------------------------------------------------------------------------

_no_cad_here = pytest.mark.skipif(
    L.build123d_available(),
    reason="build123d is importable here, so the overlay correctly takes the "
           "CAD branch; these assert what the main dycraft env gets")


def test_the_probe_agrees_with_importing_build123d():
    """It must answer for *this* interpreter, not for a configured one."""
    try:
        import build123d  # noqa: F401
        importable = True
    except BaseException:  # noqa: BLE001
        importable = False
    assert L.build123d_available() is importable


def test_the_probe_reports_false_for_a_broken_install(monkeypatch):
    """Present-but-unimportable must read as unavailable.

    `find_spec` would say yes here. A library clash in the CAD environment —
    what `_cad_env.b123d_child_env` exists to prevent — is exactly this shape:
    an installed package whose import raises. An overlay trusting `find_spec`
    would then fail deep inside `build_namespace` instead of taking the no-CAD
    branch.
    """
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name == "build123d":
            raise ImportError("undefined symbol: XML_SetAllocTrackerActivation")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert L.build123d_available() is False


@_no_cad_here
def test_the_no_cad_overlay_is_exactly_the_sdk_names(ctx):
    """What the main DyCraft env gets, and what the scene layer runs against.

    If build123d names leaked into this overlay, a source written for it would
    execute against a different global namespace than it was written for. The
    SDK names themselves are allowed and expected: `add_part` then refuses with
    a structured "no compiler" error (the test below), which is a better
    failure than a `NameError` on `Object`.

    `Material` is the only way to author a surface with scalars, and
    `Emission` gives a part light emission. Pinned as an exact set rather than
    a subset because the point is to catch a build123d name arriving, and a
    subset check would not.

    `shader` is in the set unconditionally. It is bound above the CAD check in
    `_object_build_overlay` precisely so a box-only source can author a graph,
    so the one surface vocabulary is `shader` + `Material` + `Emission`.
    `@shader` never executes here either way: it is carried to the
    render child as text (`sdk/_shader`), so binding it in a CAD-less env costs
    nothing.

    Adding a name here is a deliberate act: the overlay IS the vocabulary an
    agent-authored `object.py` can call, so a name the prompt teaches but the
    overlay omits is a `NameError` in the agent's face.
    """
    overlay = L._object_build_overlay(ctx)
    assert sorted(overlay) == ["Direction", "Emission", "Material", "Object",
                               "shader"]
    assert _ma.get_active_solid_compiler() is None


@_no_cad_here
def test_add_part_refuses_without_a_compiler(ctx):
    """A clear structured error, not an ImportError from deep in the SDK."""
    L._object_build_overlay(ctx)
    obj = Object("no_cad")
    with pytest.raises(InvalidPrimitive) as exc:
        obj.add_part("body", object())
    assert exc.value.evidence["reason"] == "no_active_solid_compiler"
    assert "object worker" in " ".join(exc.value.suggested_repairs)


# ---------------------------------------------------------------------------
# Serialization safety
# ---------------------------------------------------------------------------

def test_registered_shapes_stay_outside_the_serializable_records(ctx):
    """A live OCCT handle inside `_parts` would break content hashing.

    `canonical_json_bytes` runs `json.dumps(..., allow_nan=False)` over the
    part records, so this is not a style preference — it is what makes the
    object hashable at all while shapes are still registered.
    """
    obj = Object("store")
    assert obj._registered_shapes == {}

    class _FakeShape:                       # not JSON-serializable
        pass

    # Simulate what add_part stores, without needing a kernel.
    obj._parts.append({"id": 0, "name": "p", "primitive": "solid",
                       "params": {"aabb_min": [0, 0, 0], "aabb_max": [1, 1, 1],
                                  "shape": {}},
                       "material": {}})
    obj._registered_shapes[0] = {"name": "p", "shape": _FakeShape()}

    # The records serialize; the live store is what holds the unserializable bit.
    canonical_json_bytes({"parts": list(obj._parts)})
    with pytest.raises(TypeError):
        json.dumps(obj._registered_shapes)


def test_staging_is_a_noop_without_registered_shapes():
    """An object with no `add_part` parts must finalize without a session."""
    obj = Object("legacy")
    assert obj._registered_shapes == {}
    obj._stage_registered_parts()               # must not raise, must not need a session
    assert obj._registered_shapes == {}


def test_singleton_reset_clears_the_compiler(ctx):
    """Otherwise a compiler installed by one exec leaks into the next."""
    _ma.set_active_solid_compiler(object())
    assert _ma.get_active_solid_compiler() is not None
    L.reset_object_layer_state()
    assert _ma.get_active_solid_compiler() is None


# ---------------------------------------------------------------------------
# compile_entry's fail-soft reporting
#
# QC must never fail an otherwise-good solid, but a stage that broke must never
# pass itself off as a stage that ran and found nothing. `repair_status` /
# `qc_status` carry exactly that distinction into the result payload.
# ---------------------------------------------------------------------------

def _entry(*, tolerance=1e-3, tolerance_used=None):
    """A `measure_shape`-shaped snapshot: an outward-wound unit cube."""
    return {
        "name": "p",
        "verts": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        "faces": [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                  (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                  (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)],
        "brep_bytes": b"opaque; nothing in this layer parses it",
        "table": {"volume": 1.0},
        "aabb_min": [0.0, 0.0, 0.0], "aabb_max": [1.0, 1.0, 1.0],
        "tolerance": tolerance,
        "tolerance_used": (tolerance if tolerance_used is None
                           else tolerance_used),
        "measure_notes": [],
    }


def test_compile_entry_reports_a_clean_mesh_as_clean():
    from dycraft.agent.cad import _solid_compile as SC

    out = SC.compile_entry(_entry())
    assert out["repair_status"] == "ok" and out["qc_status"] == "ok"
    # These three ride in the shape table, which is the only part of this
    # record that survives into the ObjectIR and so the only place
    # `part_shape()` can read them from.
    assert out["repairs"] == []                # ran, changed nothing
    assert out["warnings"] == []
    assert out["shape"]["qc"]["bounds_volume"] is True
    assert out["brep_bytes"] == _entry()["brep_bytes"]   # passed through
    # The OBJ carries per-corner normals, which is the whole reason the
    # writer is shared with the geometry module rather than inlined here.
    assert b"vn " in out["obj_bytes"] and b"//" in out["obj_bytes"]


def test_compile_entry_reports_a_broken_repair_stage_instead_of_hiding_it(monkeypatch):
    from dycraft.agent.cad import _geometry_qc as GQ
    from dycraft.agent.cad import _solid_compile as SC

    # `**kw` rather than naming the keywords: this stub only has to be
    # callable, and spelling the real signature is what made it conflict when
    # both lines widened `_repair_mesh` in the same fortnight.
    def boom(v, f, groups=None, **kw):
        raise RuntimeError("boom-repair")

    monkeypatch.setattr(GQ, "_repair_mesh", boom)
    out = SC.compile_entry(_entry())
    assert out["repair_status"] == "failed"
    assert out["repairs"] == []                # distinct from "ok" + []
    assert any("boom-repair" in w for w in out["warnings"])
    # QC still ran
    assert out["qc_status"] == "ok" and out["shape"]["qc"] is not None
    assert out["n_faces"] == 12                # the unrepaired mesh was kept


def test_compile_entry_reports_a_broken_qc_stage_instead_of_hiding_it(monkeypatch):
    from dycraft.agent.cad import _geometry_qc as GQ
    from dycraft.agent.cad import _solid_compile as SC

    def boom(v, f):
        raise RuntimeError("boom-qc")

    monkeypatch.setattr(GQ, "_mesh_qc", boom)
    out = SC.compile_entry(_entry())
    # The key is always present so reading it is never a KeyError; None is how
    # "QC could not run" is said, and the warning carries the reason.
    assert out["qc_status"] == "failed" and out["shape"]["qc"] is None
    assert out["repair_status"] == "ok"
    assert any("boom-qc" in w for w in out["warnings"])
    assert out["obj_bytes"]                    # the solid still staged


# ---------------------------------------------------------------------------
# Namespace injection
# ---------------------------------------------------------------------------

def test_the_sdk_names_the_prompt_teaches_are_all_injected():
    """Every SDK name `object.py` may use has to resolve inside the sandbox.

    `Direction` is the easy one to drop: it is only reachable through
    `assert_parallel` / `assert_antiparallel` against a world axis, which is the
    sole way to constrain a joint axis by *sign* (`assert_axis_world_aligned` is
    unsigned). Taught but not injected, that advice is a NameError the author
    cannot work around.
    """
    from types import SimpleNamespace

    from dycraft.agent.layers.layer import _object_build_overlay

    overlay = _object_build_overlay(SimpleNamespace(run_dir=None))
    for name in ("Object", "Material", "Direction"):
        assert name in overlay, f"{name} is taught but not injected"


def test_collision_assertion_rejects_shadowing():
    with pytest.raises(NS.NamespaceCollision) as exc:
        NS.assert_no_collisions(["Box", "Object"], ["Object", "Material"])
    assert "Object" in str(exc.value)


def test_collision_assertion_passes_when_disjoint():
    NS.assert_no_collisions(["Box", "Cylinder"], ["Object", "Material"])


def test_curated_list_does_not_name_sdk_symbols():
    """A static guard: CURATED is a literal, so this catches an editing slip
    without needing build123d installed."""
    NS.assert_no_collisions(NS.CURATED, ["Object", "Material"])


def test_curated_list_does_not_shadow_restricted_builtins():
    from dycraft.agent.cad.sandbox import _RESTRICTED_BUILTINS
    NS.assert_no_collisions(NS.CURATED, list(_RESTRICTED_BUILTINS))


def test_sandbox_still_rejects_imports():
    """Injection exists so the import policy need not move. Prove it did not."""
    from dycraft.agent.cad import sandbox
    from dycraft.sdk._errors import SandboxViolation
    for src in ("import build123d", "from build123d import Box"):
        with pytest.raises(SandboxViolation):
            sandbox.check_ast(src, mode="exec")


def test_compile_entry_refuses_a_mesh_repair_emptied():
    """Every triangle zero-area is an empty solid, not a silently blank asset."""
    from dycraft.agent.cad import _solid_compile as SC

    flat = dict(_entry(), verts=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                faces=[(0, 1, 2)])
    with pytest.raises(SC.SolidCompileError) as exc:
        SC.compile_entry(flat)
    assert exc.value.kind == "empty_mesh"


def test_a_coarsened_render_mesh_is_silent_to_the_agent():
    """Coarsening is recorded on the result and reported to nobody.

    Only the visual is coarsened and the exact `.brep` is untouched, so there
    is nothing for the agent to act on. `tolerance_used` stays on the result
    for the compiler and for post-hoc analysis; `warnings` carries no sentence
    about it.
    """
    from dycraft.agent.cad import _solid_compile as SC

    out = SC.compile_entry(_entry(tolerance=1e-3, tolerance_used=4e-3))
    assert out["tolerance_used"] == 4e-3
    assert out["warnings"] == [], out["warnings"]

    clean = SC.compile_entry(_entry())
    assert clean["warnings"] == []


def test_a_measured_mesh_defect_is_recorded_and_not_reported():
    """An open mesh reaches `shape.qc` and reaches the agent nowhere.

    Nothing downstream of the object layer reads mesh closure — the layer's own
    adjudicator is B-rep, an authored solid gets a `<box>` collision proxy, and
    an offline grader recomputes the predicate off the exported mesh — so a
    defect here is a measurement, not a finding. It stays out of `warnings`, and
    out of `qc_status`, which keeps only the states meaning QC did not run.
    """
    from dycraft.agent.cad import _solid_compile as SC

    open_box = _entry()
    # Drop the top face: 12 verts of boundary, so the surface cannot enclose.
    open_box["faces"] = [f for f in open_box["faces"]
                         if f not in ((4, 5, 6), (4, 6, 7))]
    out = SC.compile_entry(open_box)

    assert out["shape"]["qc"]["bounds_volume"] is False
    assert out["shape"]["qc"]["n_boundary_edges"] > 0
    assert out["warnings"] == [], out["warnings"]
    assert out["qc_status"] == "ok"


# ---------------------------------------------------------------------------
# Facts in the IR, events in the report
#
# The IR is content-hashed whole, so anything `compile_entry` puts in the shape
# table becomes part of the object's identity. `qc` earns that (a property of the
# staged geometry, and the only channel to a consumer that sees just the
# catalog); `repairs` and `warnings` do not (they describe one run, and one of
# them embeds a triangle count that moves with the OCCT version).
# ---------------------------------------------------------------------------

def test_qc_is_a_shape_fact_and_the_events_are_not():
    from dycraft.agent.cad import _solid_compile as SC

    out = SC.compile_entry(_entry())
    shape = out["shape"]
    assert isinstance(shape["qc"], dict) and shape["qc"]["bounds_volume"] is True
    for event in ("repairs", "warnings", "repair_status", "qc_status"):
        assert event not in shape, (
            f"{event} describes the compile, not the part; hashing it would put "
            f"a per-run diagnostic in the object id")
    # …and they are still reported, just on the other channel.
    assert out["repairs"] == [] and out["warnings"] == []
    assert out["repair_status"] == "ok" and out["qc_status"] == "ok"


def test_a_coarsened_mesh_never_touches_the_shape_table():
    """The case the split exists for.

    `tolerance_used` moves with OCCT's tessellator, so the same authored
    geometry would hash two ways if it rode in the IR, whether or not the
    coarsening is reported anywhere.
    """
    from dycraft.agent.cad import _solid_compile as SC

    out = SC.compile_entry(_entry(tolerance=1e-3, tolerance_used=4e-3))
    assert out["warnings"] == []
    assert "warnings" not in out["shape"]
    assert "tolerance_used" not in out["shape"]


# ---------------------------------------------------------------------------
# `qc_status` tells three things apart, not two
# ---------------------------------------------------------------------------

def test_qc_findings_name_the_defects_and_ignore_the_non_defects():
    """`ok` means the mesh has no defect, not that the QC function returned;
    an open mesh reports its finding.

    Severity is domain-aware on purpose. `is_manifold` is documented in
    `_mesh_qc` as reported-never-required, and a closed carcass legitimately
    has two shells — so neither is a defect here. Calling them defects would
    fail correct geometry.
    """
    from dycraft.agent.cad import _geometry_qc as gq

    sound = {"bounds_volume": True, "winding_consistent": True,
             "n_degenerate_shells": 0, "n_boundary_edges": 0,
             "is_manifold": False, "n_shells": 2}
    assert gq.qc_findings(sound) == [], (
        "a non-manifold two-shell carcass is valid geometry and must not "
        "read as defective")

    holed = dict(sound, bounds_volume=False, n_boundary_edges=12)
    findings = gq.qc_findings(holed)
    assert len(findings) == 1 and "does not close" in findings[0]
    assert "qc.bounds_volume" in findings[0]

    wound = dict(sound, winding_consistent=False)
    # Phrasing tracks `_QC_DEFECTS`. Matching on the `qc.` key as well keeps
    # the pin about the finding rather than the prose.
    assert any("inconsistent winding relative to their neighbors" in f
               and "qc.winding_consistent" in f
               for f in gq.qc_findings(wound))

    degenerate = dict(sound, n_degenerate_shells=3)
    assert any("enclose no volume" in f for f in gq.qc_findings(degenerate))

    # An open rim on a mesh that still closes is a different statement from a
    # hole, and must not be reported as one.
    rim = dict(sound, n_boundary_edges=8)
    assert any("open rim rather than a hole" in f for f in gq.qc_findings(rim))
    assert not any("does not close" in f for f in gq.qc_findings(rim))


# ---------------------------------------------------------------------------
# Errors that localise themselves
# ---------------------------------------------------------------------------

def test_an_invalid_solid_names_the_face_occt_objects_to():
    """`is_valid` is a bool; the author also gets the face OCCT objects to.

    `BRepCheck_Analyzer` knows which face, which status and which wires. In a
    typical `invalid_shape` the solid and shell report `NoError` and one face
    carries the defect, often a large outer wire plus parasitic millimeter
    wires, the residue of a boolean that grazed it.
    """
    from dycraft.agent.cad import object_worker as W

    if not W.is_available():
        pytest.skip("needs the pinned build123d interpreter")
    out = W.run_cad_snippet(
        "import json\n"
        "import build123d as bd\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "bowtie = bd.make_face(bd.Polyline((0,0), (0.05,0.05), (0.05,0),\n"
        "                                  (0,0.05), (0,0)))\n"
        "s = bd.extrude(bowtie, amount=0.02)\n"
        "print(json.dumps({'valid': bool(s.is_valid),\n"
        "                  'detail': sc._brep_check_detail(s),\n"
        "                  'box': sc._brep_check_detail(bd.Box(0.1,0.1,0.1))}))\n")
    assert out["ok"], out.get("message")
    import json
    d = json.loads(out["stdout"].strip().splitlines()[-1])
    assert d["valid"] is False, "this shape no longer exercises the localiser"
    assert "face #" in d["detail"], (
        f"the defect was not localised to a face: {d['detail']!r}")
    assert "wire" in d["detail"], "the wire inventory is where the tell is"
    # A solid OCCT cannot localise (invalid, with NoError on every subshape)
    # gets an honest sentence rather than silence.
    assert "cannot say where" in d["box"]


def test_the_face_localiser_never_measures_wires_with_gcpnts():
    """A property test, because the crash cannot be caught or reproduced here.

    `GCPnts_AbscissaPoint.Length_s(BRepAdaptor_CompCurve(wire))`
    **segfaults** on a wire OCCT has already flagged as pathological
    (`IntersectingWires`, `UnorientableShape`), typically the parasitic
    micro-wire the diagnosis exists to name. A segfault is not catchable, so an
    `except BaseException` around it is no protection: the worker dies and the
    author gets `worker_crash` with no part name and no diagnosis. build123d's
    `Wire.length` measures every such wire.

    Asserted on the source because the synthetic shapes available here are not
    pathological enough to crash either implementation.
    """
    import ast
    import inspect

    from dycraft.agent.cad import _solid_compile as SC

    src = inspect.getsource(SC._face_defect_summary)
    # The docstring names the banned call in order to explain why it is
    # banned, so scan the executable body only.
    tree = ast.parse(src.lstrip())
    fn = tree.body[0]
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           ) else fn.body
    code = "\n".join(ast.unparse(node) for node in body)
    for banned in ("GCPnts", "BRepAdaptor_CompCurve"):
        assert banned not in code, (
            f"{banned} is back in the wire measurement; it segfaults on the "
            f"pathological wires this function is called for")
    assert "bd.Face(face).wires()" in code, (
        "wire lengths must come from build123d, which survives these wires")


def test_the_face_localiser_reports_the_occt_status():
    """The status separates repairs that the wire lengths cannot.

    `IntersectingWires` (two wires cross) and `UnorientableShape` (a face is
    inside out) call for different fixes, and `analyzer.Result(face).Status()`
    already knows which.
    """
    from dycraft.agent.cad import object_worker as W

    if not W.is_available():
        pytest.skip("needs the pinned build123d interpreter")
    out = W.run_cad_snippet(
        "import json\n"
        "import build123d as bd\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "bowtie = bd.make_face(bd.Polyline((0,0), (0.05,0.05), (0.05,0),\n"
        "                                  (0,0.05), (0,0)))\n"
        "s = bd.extrude(bowtie, amount=0.02)\n"
        "print(json.dumps({'detail': sc._brep_check_detail(s)}))\n")
    assert out["ok"], out.get("message")
    import json
    detail = json.loads(out["stdout"].strip().splitlines()[-1])["detail"]
    assert "face #" in detail and "wire" in detail
    # Some OCCT status name, not just the surface kind.
    assert any(word in detail for word in
               ("UnorientableShape", "IntersectingWires",
                "BadOrientationOfSubshape", "InvalidImbricationOfWires")), (
        f"no OCCT status in {detail!r}; the status is the most diagnostic "
        f"field and is available from analyzer.Result(face).Status()")
