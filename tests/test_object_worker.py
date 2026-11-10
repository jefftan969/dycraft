"""The live build123d path end to end, through the object worker.

These need a real CAD interpreter, so the module skips wholesale when none is
configured. They run from the main DyCraft env: the worker subprocesses into the
build123d env, which is the whole point of it existing.

What this file is for, specifically: proving that `object.py` alone produced
real geometry. Two failure modes make that non-obvious, so each gets an
explicit assertion rather than an "it didn't crash" inference:

  * `open3d_renderer._part_mesh` falls back to `_box_mesh` on ANY exception, so
    an unloadable asset renders as a bounding box and everything downstream
    looks fine;
  * an exported URDF can carry run-relative mesh paths that resolve from the
    run root but not from the URDF, so the object loads here and nowhere else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.render import CANONICAL_CAMERA_IDS
from dycraft.sdk import _primitives as _prim

pytestmark = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent / "fixtures"

# An AABB proxy box is 12 triangles. Anything at or below a small multiple of
# that is a fallback, not a tessellated CAD solid.
_BOX_TRIANGLES = 12


def _run_fixture(name: str, run_dir: Path, pkgroot: Path,
                 poses: str = "") -> dict:
    """Drive the fixed-source runner exactly as a human would."""
    cmd = [sys.executable, "-m", "dycraft.agent.layers.fixed_source",
           "--source", str(EXAMPLES / name), "--run-dir", str(run_dir)]
    if poses:
        cmd += ["--poses", poses]
    proc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(pkgroot)},
                          capture_output=True, text=True, timeout=900)
    meta_path = run_dir / "meta.json"
    assert meta_path.is_file(), (
        f"runner wrote no meta.json (exit {proc.returncode})\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Worker contract
# ---------------------------------------------------------------------------

def test_worker_builds_ir_from_live_shapes(tmp_path):
    src = (
        'obj = Object("w")\n'
        'b = obj.add_part("body", Box(0.1, 0.2, 0.3))\n'
        'obj.declare_root(b)\n'
        'ir = obj.finalize()\n'
    )
    out = W.run_object_source(src, run_dir=tmp_path)
    assert out["ok"], out.get("build_error") or out.get("message")
    part = out["ir"]["parts"][0]
    assert part["primitive"] == "solid"
    _prim.validate_params("solid", part["params"])


def test_worker_and_in_process_agree_on_a_legacy_source(tmp_path):
    """The worker must be a faithful execution environment, not a variant.

    A primitive-only source touches none of the new machinery, so the two paths
    should produce not merely equivalent but *identical* IR — same content hash.
    Any divergence here would mean the worker changes results for sources that
    have nothing to do with live Shapes.
    """
    from dycraft.agent.layers.compile import build_ir_with_layer
    from dycraft.agent.layers.layer import object_layer_config

    src = (
        'obj = Object("equiv")\n'
        'base = obj.make_box("base", width=0.4, length=0.3, height=0.04)\n'
        'post = obj.make_box("post", width=0.04, length=0.04, height=0.5)\n'
        'obj.declare_root(base)\n'
        'obj.attach_fixed("j", parent=base, child=post,\n'
        '                 parent_anchor=obj.anchor(base, "top"),\n'
        '                 child_anchor=obj.anchor(post, "bottom"))\n'
        'ir = obj.finalize()\n'
    )

    class _Ctx:
        def __init__(self, d):
            self.run_dir = Path(d)
            self.step = 0

    in_proc = build_ir_with_layer(src, layer_cfg=object_layer_config(),
                                  ctx=_Ctx(tmp_path / "a"))
    worker = W.run_object_source(src, run_dir=tmp_path / "b")
    assert in_proc["ok"] and worker["ok"]
    assert in_proc["ir"]["id"] == worker["ir"]["id"]
    assert in_proc["ir"] == worker["ir"]


def test_worker_writes_the_source_it_executed(tmp_path):
    """`object.py` on disk must be what ran — the source artifact cites it."""
    src = 'obj = Object("w")\nb = obj.add_part("b", Box(.1,.1,.1))\nobj.declare_root(b)\nir = obj.finalize()\n'
    out = W.run_object_source(src, run_dir=tmp_path)
    assert out["ok"]
    assert (tmp_path / "object.py").read_text() == src
    # Object-level: one program builds every part. And the staged bytes are the
    # bytes that were EXECUTED (handed over by `build_ir_with_layer`), not a
    # re-read of the file afterwards.
    staged = tmp_path / out["ir"]["source"]
    assert staged.read_text() == src


def test_the_staged_source_is_the_executed_source_not_a_re_read(tmp_path):
    """Provenance must depend on the execution, not on the file afterwards.

    `finalize` must not re-read `<run_dir>/object.py` to stage the `source`
    asset: a second read of a mutable file can put *different* text on disk
    than what was executed, which makes a provenance artifact a lie: an object
    whose recorded recipe never built it.
    Same argument as `measure_shape` serializing the B-rep eagerly rather than
    holding the Shape.
    """
    from dycraft.agent.layers.compile import build_ir_with_layer
    from dycraft.agent.layers.layer import object_layer_config

    class _Ctx:
        def __init__(self, rd):
            self.run_dir = rd
            self.step = 0

    executed = ('obj = Object("t")\n'
                'a = obj.add_part("a", Box(0.2, 0.2, 0.1))\n'
                'obj.declare_root(a)\n'
                'ir = obj.finalize()\n')
    stale = "# this text was never executed\n"
    (tmp_path / "object.py").write_text(stale)

    out = build_ir_with_layer(executed, layer_cfg=object_layer_config(),
                              ctx=_Ctx(tmp_path))
    assert out["ok"], out["build_error"]
    staged = (tmp_path / out["ir"]["source"]).read_text()
    assert staged == executed
    assert staged != stale


@pytest.mark.parametrize("src,slot,needle", [
    ("obj = Object('x'\n", "sandbox_violation", "syntax error"),
    ("import build123d\n", "sandbox_violation", "Import"),
    ("obj = Object('x')\nobj.add_part('b', Box(1,1,1).faces()[0])\n",
     "build_error", "contains no Solid"),
    ("obj = Object('x')\nobj.add_part('b', Box(.1,.1,.1))\nir = obj.finalize()\n",
     "build_error", "declare_root"),
])
def test_worker_classifies_failures(tmp_path, src, slot, needle):
    """A generic 'compile failed' would make repair guesswork."""
    out = W.run_object_source(src, run_dir=tmp_path)
    assert out["ok"] is False
    assert out[slot] is not None, out
    assert needle in json.dumps(out[slot])


def test_worker_traceback_points_at_object_py(tmp_path):
    src = 'obj = Object("x")\nb = obj.add_part("b", Box(.1,.1,.1))\nc = no_such_name\n'
    out = W.run_object_source(src, run_dir=tmp_path)
    assert out["ok"] is False
    blob = json.dumps(out)
    assert "object.py" in blob and '", line 3' in blob


def test_worker_times_out(tmp_path):
    out = W.run_object_source(
        "obj = Object('x')\nwhile True:\n    pass\n",
        run_dir=tmp_path, timeout=5)
    assert out["ok"] is False
    assert out["error_kind"] == "timeout"


def test_a_whole_scene_build_has_its_own_limit(monkeypatch):
    """`scene.py` builds every object it loads under one limit, so it cannot be
    the 300 s sized for one object: whole-scene builds routinely take longer."""
    from dycraft.agent.cad import _timelimit
    monkeypatch.delenv(W._TIMEOUT_ENV_VAR, raising=False)
    assert W._resolve_timeout(None, "scene") == 900 == _timelimit.resolve_limit(scene=True)
    assert W._resolve_timeout(None, "object") == 300 == _timelimit.resolve_limit()
    monkeypatch.setenv(W._TIMEOUT_ENV_VAR, "1800")
    assert W._resolve_timeout(None, "scene") == 1800 == _timelimit.resolve_limit(scene=True)


def test_a_scene_build_timeout_names_scene_py_and_what_the_limit_covers(tmp_path):
    out = W.run_object_source("while True:\n    pass\n", run_dir=tmp_path, timeout=5,
                              layer="scene")
    assert out["ok"] is False and out["error_kind"] == "timeout"
    assert out["message"].startswith("scene.py did not finish within 5s, the limit for one "
                                     "build of the whole scene")
    assert "object.py" not in out["message"]


def test_worker_reports_a_missing_interpreter(tmp_path):
    out = W.run_object_source("obj = Object('x')", run_dir=tmp_path,
                              python="/nonexistent/python")
    assert out["ok"] is False
    assert out["error_kind"] == "worker_unavailable"


# ---------------------------------------------------------------------------
# Case A — one rigid part
# ---------------------------------------------------------------------------

@pytest.mark.expensive
def test_case_a_single_part(tmp_path, pkgroot):
    meta = _run_fixture("m1_live_shape_single.py", tmp_path, pkgroot)
    assert meta["outcome"] == "committed", meta.get("error")

    ir = json.loads((tmp_path / "ir.json").read_text())
    assert len(ir["parts"]) == 1
    params = ir["parts"][0]["params"]
    _prim.validate_params("solid", params)

    # --- real artifacts, not stubs ---
    brep = (tmp_path / params["brep"]).read_bytes()
    obj = (tmp_path / params["mesh"]).read_bytes()
    assert brep.startswith(b"DBRep_DrawableShape"), "not an OCCT B-rep"
    assert obj.count(b"\nvn ") > 0, "no vertex normals in the OBJ"
    n_tris = obj.count(b"\nf ")
    assert n_tris > 100, f"{n_tris} triangles — that is a fallback, not a solid"

    # --- not the bounding box ---
    ext = [hi - lo for hi, lo in zip(params["aabb_max"], params["aabb_min"])]
    aabb_volume = ext[0] * ext[1] * ext[2]
    fill = params["shape"]["fill_ratio"]
    assert n_tris > _BOX_TRIANGLES * 4
    assert fill < 0.6, f"fill_ratio {fill} — the test shape must be hollow"
    assert 0.0 < params["shape"]["volume"] < aabb_volume

    # --- consumers ---
    assert meta["verifiers"]["NoSelfPenetration"] == "pass"
    assert meta["verifiers_ok"] is True
    assert (len(meta.get("final_render_paths") or [])
            == len(CANONICAL_CAMERA_IDS))


@pytest.mark.expensive
def test_case_a_urdf_meshes_resolve_from_the_urdf(tmp_path, pkgroot):
    """An exported URDF's meshes resolve from the URDF itself, or a solid-only
    export is silently unloadable."""
    _run_fixture("m1_live_shape_single.py", tmp_path, pkgroot)
    urdf = tmp_path / "final_render" / "ir.urdf"
    assert urdf.is_file()
    import xml.etree.ElementTree as ET
    refs = [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")]
    assert refs, "no <mesh> in the exported URDF"
    for ref in refs:
        assert (urdf.parent / ref).is_file(), f"{ref} does not resolve from the URDF"


# ---------------------------------------------------------------------------
# Case B — minimal articulated object
# ---------------------------------------------------------------------------

@pytest.mark.expensive
def test_case_b_articulated(tmp_path, pkgroot):
    meta = _run_fixture("m1_live_shape_articulated.py", tmp_path, pkgroot,
                        poses="closed:door_hinge=0;open:door_hinge=-75")
    assert meta["outcome"] == "committed", meta.get("error")

    ir = json.loads((tmp_path / "ir.json").read_text())
    assert len(ir["parts"]) == 2 and len(ir["joints"]) == 1
    for part in ir["parts"]:
        assert part["primitive"] == "solid"
        _prim.validate_params("solid", part["params"])
        assert (tmp_path / part["params"]["brep"]).stat().st_size > 0

    # Existing joint semantics, unchanged: free axis vector, revolute, radians
    # in the IR.
    joint = ir["joints"][0]
    assert joint["type"] == "revolute"
    assert joint["axis"] == [0.0, 0.0, 1.0]
    assert "parent_anchor" in joint and "child_anchor" in joint

    # The source used neither the second-program path nor the registry.
    source = (tmp_path / "object.py").read_text()
    for banned in ("author_solids", "make_solid", "emit_solid"):
        assert banned not in source

    assert meta["verifiers_ok"] is True


@pytest.mark.expensive
def test_case_b_fk_moves_the_door(tmp_path, pkgroot):
    """A pose-dependent world AABB is the proof articulation is real."""
    from dycraft.sdk import _probes as P
    _run_fixture("m1_live_shape_articulated.py", tmp_path, pkgroot)
    ir = json.loads((tmp_path / "ir.json").read_text())
    closed = P.object_part_world_aabb(ir, "door", poses={"door_hinge": 0.0})
    opened = P.object_part_world_aabb(ir, "door", poses={"door_hinge": -75.0})
    assert closed != opened
    # Swinging outward extends the door's reach along +Y well past the carcass.
    assert opened[1][1] > closed[1][1] + 0.2


@pytest.mark.expensive
def test_case_b_pose_renders_differ(tmp_path, pkgroot):
    meta = _run_fixture("m1_live_shape_articulated.py", tmp_path, pkgroot,
                        poses="closed:door_hinge=0;open:door_hinge=-75")
    renders = {p["label"]: p["paths"] for p in meta["pose_renders"]}
    assert set(renders) == {"closed", "open"}
    for paths in renders.values():
        assert len(paths) == len(CANONICAL_CAMERA_IDS)
    a = Path(renders["closed"][0]).read_bytes()
    b = Path(renders["open"][0]).read_bytes()
    assert a != b, "the two joint states rendered identically"


@pytest.mark.expensive
def test_exact_adjudication_clears_a_bounding_box_false_positive(tmp_path,
                                                                 pkgroot):
    """The payoff of the solid representation, on a live-Shape part.

    A shelf inside a shelled carcass: the carcass's OBB is its full outer box,
    so SAT reports a deep overlap. The exact B-rep pass must look at the real
    surfaces and drop it. `not_adjudicable: 0` is the load-bearing assertion —
    it is what proves the adjudicator actually read the B-rep this path emitted
    rather than skipping the pair and keeping SAT's conservative verdict, which
    would also produce a "pass" for entirely the wrong reason.
    """
    source = (
        'obj = Object("adj")\n'
        'carcass = obj.add_part("carcass",\n'
        '    Box(0.50, 0.40, 0.60) - Pos(0, 0.011, 0) * Box(0.478, 0.40, 0.578))\n'
        'shelf = obj.add_part("shelf", Box(0.470, 0.36, 0.018))\n'
        'obj.declare_root(carcass)\n'
        'obj.attach_fixed("j", parent=carcass, child=shelf,\n'
        '                 parent_anchor=obj.anchor(carcass, "center"),\n'
        '                 child_anchor=obj.anchor(shelf, "center"))\n'
        'ir = obj.finalize()\n'
    )
    src_path = tmp_path / "src.py"
    src_path.write_text(source)
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "dycraft.agent.layers.fixed_source",
         "--source", str(src_path), "--run-dir", str(run_dir), "--skip-render"],
        env={**os.environ, "PYTHONPATH": str(pkgroot)},
        capture_output=True, text=True, timeout=900)
    assert (run_dir / "meta.json").is_file(), proc.stderr[-2000:]

    report = json.loads(
        (run_dir / "compile_reports" / "latest.json").read_text())
    penetration = next(v for v in report["verifiers"]
                       if v["name"] == "NoSelfPenetration")
    stats = (penetration.get("evidence") or {}).get("exact_adjudication") or {}
    assert stats.get("available") is True, "the adjudicator did not run"
    assert stats.get("sent", 0) >= 1, "SAT flagged nothing — the fixture is wrong"
    assert stats.get("not_adjudicable", 0) == 0, (
        "a live-Shape part was not adjudicable, so its B-rep was unreadable")
    assert stats.get("cleared", 0) >= 1
    assert penetration["verdict"] == "pass"


def test_the_child_releases_memory_after_each_job(monkeypatch):
    """`gc.collect()` then glibc's `malloc_trim(0)`, best effort."""
    import ctypes
    import gc
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "agent" / "cad" / "_object_worker_child.py"
    spec = importlib.util.spec_from_file_location("_child_under_test", path)
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc") or 0)

    class Libc:
        def malloc_trim(self, pad):
            calls.append(("trim", pad))
    monkeypatch.setattr(ctypes, "CDLL", lambda name: Libc())
    child._release_memory()
    assert calls == ["gc", ("trim", 0)]

    def missing(name):
        raise OSError("not glibc")
    monkeypatch.setattr(ctypes, "CDLL", missing)
    child._release_memory()                     # no libc: nothing raised
