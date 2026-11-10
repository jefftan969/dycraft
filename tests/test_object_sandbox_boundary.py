"""What `object.py` can and cannot reach.

The AST allowlist decides which *syntax* runs; the injected globals decide what
that syntax can touch, so the boundary is both. build123d 0.11.1 re-exports 54
modules at its top level, so if `bd` were the build123d module itself this
would compile and run through the real worker with no violation:

    pid = bd.os.getpid()
    cwd = bd.os.getcwd()

and `bd.os.environ`, `bd.sys.path`, `bd.Path` (which is `pathlib.Path`),
`bd.export_to_pcbway` (a network upload) and `bd.Lib3MF.ctypes.util.subprocess`
would be reachable the same way, all through `Name` + non-underscore
`Attribute` nodes the allowlist has no reason to reject. `bd` is therefore a
guarded namespace serving only the names bound directly.

The file is in three parts, because the guarantees have different reach:

  * `_GuardedNamespace` itself, which needs nothing installed and so runs in
    every environment — this is where the invariants live;
  * the real injected vocabulary, which needs build123d;
  * the end-to-end proof through the CAD worker, which needs the interpreter
    and is marked expensive.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from dycraft.agent.cad import _b123d_namespace as NS
from dycraft.agent.cad import object_worker as W
from dycraft.sdk._errors import SandboxViolation

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="no build123d interpreter")


def _namespace():
    try:
        return NS.build_namespace(reserved={"Object", "Material"})
    except ImportError:
        pytest.skip("build123d is not importable in this interpreter")


def _in_cad_interpreter(body: str) -> str:
    """Run `body` under the pinned CAD interpreter and return its stdout.

    build123d lives in its own environment and the main DyCraft env cannot
    import it, so assertions about the *real* injected namespace cannot run
    in-process here. The worker crosses that boundary for compiles; these
    tests cross it the same way rather than skipping — a namespace guard that
    is only checked where the namespace cannot be built is not a guard.

    The body runs with `NS` bound to the loaded `_b123d_namespace` module and
    `ns` bound to a freshly built namespace. It should `print` its findings;
    any assertion it makes fails this test, because `run_cad_snippet` reports a
    raising body as `ok=False`.

    Runs on the warm pooled CAD child via `run_cad_snippet` rather than a fresh
    `-c` interpreter, so the nine tests here share one `import build123d`
    instead of paying it nine times. No dycraft bootstrap is needed in the
    preamble — the child already has the package bound, and re-running the
    bootstrap would load a second copy alongside it.
    """
    import textwrap

    if not W.is_available():
        pytest.skip("no build123d interpreter")
    # `sys`/`types` are part of the preamble's contract, not incidental: the
    # bodies below walk the injected namespace looking for module objects.
    prog = textwrap.dedent("""
        import sys, types
        from dycraft.agent.cad import _b123d_namespace as NS
        from dycraft.sdk._errors import SandboxViolation
        ns = NS.build_namespace(reserved={"Object", "Material"})
    """) + textwrap.dedent(body)
    out = W.run_cad_snippet(prog)
    assert out["ok"], (
        f"{out['message']}\n{out['traceback'] or ''}\n"
        f"stdout={out['stdout']}\nstderr={out['stderr']}")
    return out["stdout"]


# ---------------------------------------------------------------------------
# The guard itself — no build123d required
# ---------------------------------------------------------------------------

def test_a_guarded_namespace_serves_only_its_table():
    g = NS._GuardedNamespace("bd", {"Box": object()})
    assert g.Box is not None
    with pytest.raises(SandboxViolation) as e:
        g.os
    assert "bd.os" in str(e.value)


def test_refusal_is_a_sandbox_violation_not_an_attribute_error():
    """`compile` routes SandboxViolation to its own report field.

    Raising AttributeError instead would land the denial in the generic
    runtime-error channel, where it reads as the agent's typo rather than as
    the policy decision it is.
    """
    g = NS._GuardedNamespace("bd", {})
    with pytest.raises(SandboxViolation) as e:
        g.anything
    d = e.value.to_dict()
    assert d["code"] == "SandboxViolation"
    assert d["evidence"]["handle"] == "bd"
    assert d["evidence"]["attr"] == "anything"
    assert d["suggested_repairs"], "a refusal must say what is available instead"


def test_a_guarded_namespace_is_read_only():
    g = NS._GuardedNamespace("bd", {"Box": 1})
    with pytest.raises(SandboxViolation):
        g.Box = 2
    with pytest.raises(SandboxViolation):
        g.new_name = 2


def test_a_guarded_namespace_hides_its_own_internals():
    """`_table` would be a way back to everything the table holds by another
    name; the AST rejects underscore attributes, but the guard should not be
    relying on that alone."""
    g = NS._GuardedNamespace("bd", {"Box": 1})
    assert dir(g) == ["Box"]
    assert "1" not in repr(g) or "injected names" in repr(g)


# ---------------------------------------------------------------------------
# The real injected vocabulary
# ---------------------------------------------------------------------------

# None of these may resolve. `os`/`sys`/`ctypes` are the
# process and filesystem; `Path` is `pathlib.Path`; the import_/export_ helpers
# read and write files; `export_to_pcbway` uploads over the network.
_FORBIDDEN_ON_BD = (
    "os", "sys", "ctypes", "tempfile", "io", "logging", "re", "json",
    "requests", "webbrowser", "warnings", "inspect", "functools", "itertools",
    "copy_module", "importers", "exporters", "exporters3d", "mesher",
    "Lib3MF", "ET", "PT", "np", "Path", "PathLike", "Mesher",
    "import_step", "import_stl", "import_svg", "import_brep", "import_dxf",
    "export_step", "export_stl", "export_brep", "export_gltf",
    "export_to_pcbway", "ExportSVG", "ExportDXF",
    "BRepTools", "RWStl", "StlAPI_Writer", "STEPCAFControl_Reader",
)


@_needs_cad
def test_bd_does_not_reach_modules_or_io():
    """No name here resolves to a live module or capability."""
    out = _in_cad_interpreter(f"""
        forbidden = {_FORBIDDEN_ON_BD!r}
        served = []
        for attr in forbidden:
            try:
                getattr(ns["bd"], attr)
            except SandboxViolation:
                continue
            served.append(attr)
        assert not served, "bd still serves: " + repr(served)
        print(len(forbidden))
    """)
    assert int(out.strip()) == len(_FORBIDDEN_ON_BD)


@_needs_cad
def test_the_only_injected_modules_are_math_np_and_ocp():
    """A module is a door to every module *it* imported, so the namespace holds
    exactly three, all deliberate. `math` costs nothing — it exposes
    no module-typed attribute, so there is nothing to traverse to. `np` is the
    real numpy and *is* a door: `np.ma.core.builtins` is the builtins module,
    reachable through non-underscore attributes the walker has no rule against.
    That was accepted knowingly (see `_b123d_namespace`'s docstring) because
    the agent that writes `object.py` already holds `Bash(*)` — the sandbox is
    not the containment boundary and pretending otherwise bought nothing.

    `OCP` is the OCCT binding build123d is built on, bound because build123d
    exposes a subset of its own kernel — a variable-radius fillet and a
    tolerant sew are constructors no name binding reaches. It opens no door
    `np` had not already opened.

    The assertion is therefore *which* modules, not whether any. A fourth one
    appearing is a real event and should fail here.
    """
    _in_cad_interpreter("""
        leaked = sorted(k for k, v in ns.items()
                        if isinstance(v, types.ModuleType))
        assert leaked == ["OCP", "math", "np"], \
            "unexpected module objects in object.py globals: " + repr(leaked)
        print("ok")
    """)


@_needs_cad
def test_bd_serves_nothing_beyond_the_directly_bound_names():
    """`bd.X` must add no reachable surface over the globals themselves.

    This is what makes the boundary auditable: whatever `bd` can hand back is
    already in the namespace, so reviewing the namespace reviews `bd` too.
    """
    _in_cad_interpreter("""
        direct = {k for k in ns if k not in ("bd", "math", "np")}
        served = set(dir(ns["bd"]))
        assert served <= direct, "not bound directly: " + repr(served - direct)
        print("ok")
    """)


@_needs_cad
@pytest.mark.expensive
def test_injected_values_reach_no_module_transitively():
    """Guarding the first hop is only enough if the values themselves are closed.

    Measured against build123d 0.11.1: zero of the 419 non-module public names
    reach `os`/`sys`/`subprocess`/`ctypes`/… within four attribute hops. This
    test is what would notice if that stopped being true. Expensive: it walks
    the whole injected surface.
    """
    _in_cad_interpreter("""
        danger = {"os", "sys", "subprocess", "shutil", "importlib", "builtins",
                  "socket", "pathlib", "tempfile", "ctypes", "runpy", "pickle",
                  "multiprocessing"}
        seen = set()
        found = []

        def walk(obj, path, depth):
            if depth > 3 or found:
                return
            try:
                attrs = dir(obj)
            except Exception:
                return
            for a in attrs:
                if a.startswith("_"):
                    continue       # the AST allowlist rejects these outright
                try:
                    v = getattr(obj, a)
                except Exception:
                    continue
                if isinstance(v, types.ModuleType):
                    if v.__name__.split(".")[0] in danger:
                        found.append(path + "." + a + " -> " + v.__name__)
                        return
                elif (isinstance(v, type) or callable(v)) and id(v) not in seen:
                    seen.add(id(v))
                    walk(v, path + "." + a, depth + 1)

        for name, value in ns.items():
            if name in ("bd", "math"):
                continue
            walk(value, name, 1)
        assert not found, "reaches a dangerous module: " + repr(found)
        print("ok")
    """)


@_needs_cad
def test_the_geometry_vocabulary_still_resolves_both_ways():
    _in_cad_interpreter("""
        for name in ("Box", "Cylinder", "Pos", "Rotation", "fillet", "extrude",
                     "Axis", "Plane", "Align", "Mode", "Color"):
            assert name in ns, name + " is no longer bound directly"
            assert getattr(ns["bd"], name) is ns[name], \\
                "bd." + name + " must be the same object as " + name
        print("ok")
    """)


@_needs_cad
def test_math_and_np_are_the_real_modules():
    """`math` and `np` are the real modules. `math` needs no guard — it exposes no
    module-typed attribute, so the traversal the guard existed to stop does not
    exist. That is asserted here rather than assumed, because it is the whole
    reason `math` can be handed over whole while `bd` cannot.

    The bare scalar names (`sin`, `cos`, …) are bound independently: they come
    from `sandbox._MATH_BUILTINS`, which also serves probe expressions.
    """
    _in_cad_interpreter("""
        import math as real_math
        assert ns["math"] is real_math
        assert not [a for a in dir(real_math)
                    if isinstance(getattr(real_math, a), types.ModuleType)], \
            "math grew a module attribute; it now needs a guard like bd's"
        assert ns["math"].e == real_math.e, "the full module is bound"
        import numpy as real_np
        assert ns["np"] is real_np
        print("ok")
    """)


@_needs_cad
def test_bare_trig_still_resolves_without_a_prefix():
    """`sin(...)` with no `math.` prefix is what every authored source and
    Skill recipe uses. Dropping `MATH_NAMES` from the namespace must not have
    taken it away — it comes from the restricted builtins instead."""
    _in_cad_interpreter("""
        import math as real_math
        from dycraft.agent.cad import sandbox as sb
        g = sb.build_globals(overlay=ns)
        b = g["__builtins__"]
        for name in ("pi", "sin", "cos", "atan2", "sqrt", "radians", "degrees"):
            assert name in b, name + " is no longer reachable unprefixed"
        assert abs(b["cos"](b["pi"]) + 1.0) < 1e-12
        print("ok")
    """)


@_needs_cad
def test_the_builder_api_is_not_injected():
    """`BuildPart` and friends need `with`, which the allowlist rejects.

    Binding them would be surface with no reachable use, and the prompt
    teaches algebra mode only.
    """
    _in_cad_interpreter("""
        for name in ("BuildPart", "BuildSketch", "BuildLine"):
            assert name not in ns, name + " is injected but cannot be used"
            try:
                getattr(ns["bd"], name)
            except SandboxViolation:
                continue
            raise AssertionError("bd serves " + name)
        print("ok")
    """)


@_needs_cad
@pytest.mark.expensive
def test_build123d_class_tables_are_reachable_and_that_is_accepted():
    """A documented, bounded acceptance — not a passing grade.

    build123d keeps ~259 mutable dicts as class attributes (`Box.shape_LUT`,
    `Box.downcast_LUT`, …). `Box.shape_LUT['x'] = 1` is rejected by the AST
    lint, but `Box.shape_LUT.clear()` is a method call on a third-party class,
    which `freeze` cannot cover without breaking the library.

    It is contained rather than closed: object exec runs in a CAD worker
    subprocess, so a corrupted table lives only as long as that child. A pooled
    child carries it into its later builds in the same run until the child is
    retired; `DYCRAFT_OBJECT_WORKER_PERSISTENT=0` gives every build a fresh
    one. build123d is not injected into the scene layer. This
    test exists so the reachability stays *known*; if it ever starts failing
    because the surface shrank, that is good news worth noticing.
    """
    out = _in_cad_interpreter("""
        count = 0
        for name, val in ns.items():
            if name in ("bd", "math"):
                continue
            for a in dir(val):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(val, a)
                except Exception:
                    continue
                if isinstance(v, (dict, list, set)):
                    count += 1
        print(count)
    """)
    reachable = int(out.strip())
    assert reachable > 0, (
        "build123d no longer exposes mutable class containers — the accepted "
        "limitation in design_object_agent.md can be removed")


def test_the_namespace_still_refuses_to_shadow_the_sdk():
    with pytest.raises(NS.NamespaceCollision):
        NS.assert_no_collisions(["Box", "Object"], {"Object", "Material"})


# ---------------------------------------------------------------------------
# The invariant, applied to every namespace the sandbox hands out
# ---------------------------------------------------------------------------

def _modules_in(mapping) -> list:
    return sorted(k for k, v in mapping.items() if isinstance(v, types.ModuleType))


# The two modules that are handed over on purpose. Every other module in any
# namespace the sandbox builds is a leak. See `_b123d_namespace`'s docstring
# for why `np` is on this list despite being a door to `builtins`.
#: Modules a *probe* expression may see. A probe reads `part_shape` tables and
#: bbox tuples, so it gets the arithmetic handles and nothing else.
_INTENDED_MODULES = ["math", "np"]

#: Modules a *build* namespace may see. `OCP` is here and not above because it
#: is a construction surface: `object.py` reaches OCCT algorithm classes
#: build123d does not wrap, and a probe expression never constructs anything.
_INTENDED_BUILD_MODULES = ["OCP", "math", "np"]


def test_no_layer_exposes_an_unintended_module_anywhere():
    """Object build, object probe, scene build, and the builtins table.

    A bound module opens every module it imported, in any overlay, and a
    per-overlay test is the only thing that would notice. Runs in the main env,
    in process — the live CAD namespace is covered
    by `test_the_only_injected_modules_are_math_np_and_ocp` inside the CAD
    interpreter.

    `math` and `np` are expected in *both* the build and the probe namespaces:
    a probe reads `part_shape` tables and bbox tuples, which is the same
    arithmetic `object.py` does, so it gets the same handles. `OCP` is expected
    in the build namespace only.
    """
    import tempfile

    from dycraft.agent.cad import sandbox as sb
    from dycraft.agent.layers.layer import object_layer_config

    class _Payload:
        object_handles: dict = {}

    class _Ctx:
        run_dir = Path(tempfile.mkdtemp())
        step = 0
        layer_payload = _Payload()

    for cfg in (object_layer_config(),):
        g = sb.build_globals(overlay=cfg.build_overlay_factory(_Ctx()))
        unexpected = [m for m in _modules_in(g)
                      if m not in _INTENDED_BUILD_MODULES]
        assert not unexpected, f"{cfg.name} build globals: {unexpected}"
        assert not _modules_in(g["__builtins__"]), cfg.name

    ir = {"schema": "1", "id": "x", "name": "n", "parts": [], "joints": []}
    overlay = object_layer_config().probe_overlay_factory(_Ctx(), ir)
    q = sb.build_query_globals(overlay)
    assert _modules_in(q) == _INTENDED_MODULES, \
        f"object probe globals: {_modules_in(q)}"
    assert not _modules_in(q["__builtins__"])


# ---------------------------------------------------------------------------
# End to end, through the real worker
# ---------------------------------------------------------------------------

_ESCAPE_SOURCE = """
pid = bd.os.getpid()
obj = Object("escape_" + str(pid))
p = obj.add_part("b", Box(0.1, 0.1, 0.1), material=Material(color=(128, 128, 128)))
obj.declare_root(p)
ir = obj.finalize()
"""

_CLEAN_SOURCE = """
side = 0.12
body = bd.Box(side, side, side) - Pos(0, 0, side / 2) * Cylinder(0.02, side)
obj = Object("guarded_ok")
p = obj.add_part("body", body, material=Material(color=(160, 160, 165)))
obj.add_anchor(p, "bore_mouth", translation=(0.0, 0.0, side / 2))
obj.declare_root(p)
ir = obj.finalize()
"""


@_needs_cad
@pytest.mark.expensive
def test_the_documented_escape_is_refused_by_a_real_run(tmp_path):
    """The documented escape, run the way a run runs it, is refused."""
    raw = W.run_object_source(_ESCAPE_SOURCE, run_dir=tmp_path)
    assert not raw["ok"]
    sv = raw["sandbox_violation"]
    assert sv is not None, "bd.os must be refused, not silently evaluated"
    assert sv["code"] == "SandboxViolation"
    assert "bd.os" in sv["summary"]


@_needs_cad
@pytest.mark.expensive
def test_ordinary_bd_style_authoring_still_builds(tmp_path):
    """The other half: closing the module must not close the API.

    `bd.Box` is the style build123d's own examples use, and sources in the wild
    are written that way; it has to keep working.
    """
    raw = W.run_object_source(_CLEAN_SOURCE, run_dir=tmp_path)
    assert raw["ok"], (raw["build_error"], raw["sandbox_violation"])
    ir = raw["ir"]
    assert len(ir["parts"]) == 1
    assert ir["parts"][0]["primitive"] == "solid"
    assert ir["parts"][0]["anchors"], "anchors must survive the guarded namespace"
