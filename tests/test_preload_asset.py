"""`--preload-asset`: geometry a scene or animation run places but did not build.

The scene layer is defined by authoring every object it places, and that is
the right contract for the objects a brief describes. It is the wrong one for an
asset that already exists and whose value is being the real one — a Unitree G1
the viewer is meant to recognize is not improved by an agent's impression of
one. These tests protect the seam that lets such an asset in without letting
anything else in with it: the launcher verifies the bundle before the run
starts, `load_asset` is the only door at build time, and the geometry it returns
is indistinguishable from authored geometry everywhere except the one place the
difference is real — the object verifier suite, which asks a question about a
program that does not exist in this run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.agent.assets.preload_asset import (
    PRELOAD_DIR, PRELOAD_MANIFEST, PreloadAssetError, load_preloaded_manifest,
    parse_spec, stage_preloaded_assets)
from dycraft.sdk.ir import OBJECT_SCHEMA_VERSION, content_hash

pytestmark_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")


# ---------------------------------------------------------------------------
# A bundle fixture: two imported parts on one revolute joint
# ---------------------------------------------------------------------------

#: A unit tetrahedron. Small on purpose — what is under test is the staging and
#: the hashing, and a real 23 MB robot would make these tests a download.
_OBJ = ("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")


def _shape(volume=0.1666):
    """The derived mass-property table every asset part must carry."""
    return {"volume": volume, "center": [0.25, 0.25, 0.25],
            "inertia_diag": [0.01, 0.01, 0.01]}


def _imported_part(name, mesh, collision, *, digest, collision_digest):
    return {
        "name": name,
        "primitive": "imported",
        "params": {
            "mesh": mesh, "collision": collision,
            "digest": digest, "collision_digest": collision_digest,
            "aabb_min": [0.0, 0.0, 0.0], "aabb_max": [1.0, 1.0, 1.0],
            "shape": _shape(),
        },
        "material": {"color": [90, 90, 90]},
    }


def make_bundle(tmp_path: Path, name="rig", *, tamper=False,
                drop_asset=False) -> Path:
    """Write a valid (or deliberately broken) preload bundle and return it.

    Mirrors what `import_urdf.convert` writes: `ir.json` beside the `assets/`
    its parts cite, at the run-relative paths the IR names.
    """
    bundle = tmp_path / "bundles" / name
    (bundle / "assets" / "meshes").mkdir(parents=True, exist_ok=True)
    (bundle / "assets" / "collision").mkdir(parents=True, exist_ok=True)
    for i in ("a", "b"):
        (bundle / "assets" / "meshes" / f"{i}.obj").write_text(_OBJ)
        (bundle / "assets" / "collision" / f"{i}.obj").write_text(_OBJ)

    parts = [
        _imported_part("base", "assets/meshes/a.obj",
                       "assets/collision/a.obj",
                       digest="d_a", collision_digest="c_a"),
        _imported_part("arm", "assets/meshes/b.obj",
                       "assets/collision/b.obj",
                       digest="d_b", collision_digest="c_b"),
    ]
    ir = {
        "schema": OBJECT_SCHEMA_VERSION,
        "name": f"{name}_asset",
        "root": "base",
        "parts": parts,
        "joints": [{
            "id": 0, "name": "shoulder", "type": "revolute",
            "parent": "base", "child": "arm",
            "parent_anchor": {"translation": [0.0, 0.0, 1.0],
                              "rotation": [1.0, 0.0, 0.0, 0.0]},
            "child_anchor": {"translation": [0.0, 0.0, 0.0],
                             "rotation": [1.0, 0.0, 0.0, 0.0]},
            "axis": [0.0, 1.0, 0.0], "limits": [-1.5708, 1.5708],
            "default": 0.0,
        }],
    }
    ir["id"] = content_hash(ir)
    if tamper:
        # Edited after freezing: the content no longer hashes to the stated id.
        ir["parts"][0]["params"]["aabb_max"] = [2.0, 2.0, 2.0]
    (bundle / "ir.json").write_text(json.dumps(ir, indent=2))
    if drop_asset:
        (bundle / "assets" / "meshes" / "b.obj").unlink()
    return bundle


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

def test_a_spec_is_name_equals_bundle():
    name, path = parse_spec(" g1 = assets/imported/g1_with_dex1 ")
    assert (name, path) == ("g1", Path("assets/imported/g1_with_dex1"))


@pytest.mark.parametrize("spec", ["g1", "=bundle", "g1=", ""])
def test_a_malformed_spec_is_refused(spec):
    with pytest.raises(PreloadAssetError, match="NAME=BUNDLE"):
        parse_spec(spec)


@pytest.mark.parametrize("spec", ["../g1=bundle", "a/b=bundle", ".=bundle"])
def test_a_name_that_is_a_path_is_refused(spec):
    """The name becomes a file stem and an argument to `load_asset`."""
    with pytest.raises(PreloadAssetError, match="not a plain name"):
        parse_spec(spec)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def test_staging_copies_the_assets_the_ir_cites(tmp_path):
    """At the paths the IR names, which is what keeps its hash true."""
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    recs = stage_preloaded_assets([f"rig={bundle}"], run)

    rec = recs["rig"]
    assert rec["n_parts"] == 2 and rec["n_joints"] == 1
    assert sorted(rec["assets"]) == [
        "assets/collision/a.obj", "assets/collision/b.obj",
        "assets/meshes/a.obj", "assets/meshes/b.obj"]
    for rel in rec["assets"]:
        assert (run / rel).is_file(), rel
    staged = json.loads((run / rec["ir_path"]).read_text())
    assert staged["id"] == rec["ir_hash"]
    assert content_hash({k: v for k, v in staged.items() if k != "id"}) \
        == staged["id"], "staging must not disturb the frozen identity"


def test_the_manifest_describes_what_was_staged(tmp_path):
    """The agent reads this before its first build, instead of discovering the
    shape of a 38-part robot by placing it."""
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    stage_preloaded_assets([f"rig={bundle}"], run)

    blob = json.loads((run / PRELOAD_DIR / PRELOAD_MANIFEST).read_text())
    assert blob["schema"] == "dycraft.preloaded_assets.v1"
    rec = blob["assets"]["rig"]
    assert rec["object_name"] == "rig_asset"
    assert rec["parts"] == ["base", "arm"]
    assert rec["root"] == "base"
    # Degrees, because that is the unit every authoring surface uses; the IR
    # stores radians.
    assert rec["joints"][0]["limits_deg"] == [-90.0, 90.0]
    assert rec["size"] == pytest.approx([1.0, 1.0, 2.0], abs=1e-6)


def test_a_tampered_ir_is_refused(tmp_path):
    """The same anti-tamper rule a catalog gets, applied before the run starts."""
    bundle = make_bundle(tmp_path, tamper=True)
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="content hash mismatch"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_a_bundle_missing_an_asset_is_refused(tmp_path):
    """Caught in the launcher, where a human is watching, not at first compile."""
    bundle = make_bundle(tmp_path, drop_asset=True)
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="missing asset"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_a_name_staged_twice_is_refused(tmp_path):
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="named twice"):
        stage_preloaded_assets([f"rig={bundle}", f"rig={bundle}"], run)


def test_a_bundle_that_is_not_a_directory_is_refused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="not a directory"):
        stage_preloaded_assets([f"rig={tmp_path / 'nope'}"], run)


def test_no_preload_means_no_manifest(tmp_path):
    """The overwhelmingly common case reads as empty rather than raising."""
    run = tmp_path / "run"
    run.mkdir()
    assert stage_preloaded_assets([], run) == {}
    assert not (run / PRELOAD_DIR).exists()
    assert load_preloaded_manifest(run) == {}


# ---------------------------------------------------------------------------
# `load_asset`, through the CAD worker
# ---------------------------------------------------------------------------

_PLACE = '''
rig = load_asset("rig")
crate = Object("crate")
crate.declare_root(crate.make_box("body", 0.6, 0.4, 0.4))

scene = Scene("bay")
scene.set_room(6.0, 5.0, 3.0)
scene.place(rig, name="the_rig", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
scene.place(crate, name="carton", pose=((2.0, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''


def _staged_run(tmp_path):
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    recs = stage_preloaded_assets([f"rig={bundle}"], run)
    return run, recs["rig"]


@pytestmark_cad
def test_a_preloaded_asset_can_be_placed_and_reaches_the_catalog(tmp_path):
    """The whole point. `Scene.place` already takes a finalized dict; what this
    protects is that the definition also lands in `object_catalog`, without
    which the committed catalog would name an `object_ref` it does not hold."""
    run, rec = _staged_run(tmp_path)
    out = W.run_object_source(_PLACE, run_dir=run, layer="scene")
    assert out["ok"], out.get("build_error") or out.get("sandbox_violation")

    catalog = out["object_catalog"]
    assert rec["ir_hash"] in catalog, "the asset must be in the built catalog"
    assert catalog[rec["ir_hash"]]["name"] == "rig_asset"
    refs = {i["name"]: i["object_ref"] for i in out["ir"]["instances"]}
    assert refs["the_rig"] == rec["ir_hash"]
    assert refs["carton"] in catalog and refs["carton"] != rec["ir_hash"]


@pytestmark_cad
def test_the_staged_ir_is_a_declared_dependency(tmp_path):
    """The closure `fixed_source` replays should name every input, not only
    the editable ones."""
    run, _ = _staged_run(tmp_path)
    out = W.run_object_source(_PLACE, run_dir=run, layer="scene")
    assert out["ok"], out.get("build_error")
    assert f"{PRELOAD_DIR}/rig.json" in out["source_deps"]


@pytestmark_cad
def test_loading_the_same_asset_twice_is_one_definition(tmp_path):
    """Keyed by content hash, exactly as placing one authored definition twice
    contributes one entry."""
    run, rec = _staged_run(tmp_path)
    src = '''
a = load_asset("rig")
b = load_asset("rig")
scene = Scene("bay")
scene.set_room(6.0, 5.0, 3.0)
scene.place(a, name="left", pose=((-1.0, 0.0, 0.0), (0, 0, 0)))
scene.place(b, name="right", pose=((1.0, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''
    out = W.run_object_source(src, run_dir=run, layer="scene")
    assert out["ok"], out.get("build_error")
    assert list(out["object_catalog"]) == [rec["ir_hash"]]
    assert len(out["ir"]["instances"]) == 2


@pytestmark_cad
def test_an_unknown_asset_name_lists_what_is_available(tmp_path, monkeypatch):
    # An empty library, so "g1" is unknown everywhere.
    monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(tmp_path))
    # load_asset reads the library variable inside the CAD child, and a warm
    # child keeps the environment it was spawned with: build in a fresh one.
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    run, _ = _staged_run(tmp_path)
    out = W.run_object_source('ir = load_asset("g1")', run_dir=run,
                              layer="scene")
    assert not out["ok"]
    blob = str(out.get("build_error"))
    assert "no asset named 'g1'" in blob
    assert "'rig'" in blob, blob


@pytestmark_cad
def test_load_asset_without_any_preload_points_at_the_library(tmp_path, monkeypatch):
    """With nothing preloaded the run still has the asset library, so the
    message points into it rather than at a relaunch."""
    monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(tmp_path))
    # load_asset reads the library variable inside the CAD child, and a warm
    # child keeps the environment it was spawned with: build in a fresh one.
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    run = tmp_path / "run"
    run.mkdir()
    out = W.run_object_source('ir = load_asset("g1")', run_dir=run,
                              layer="scene")
    assert not out["ok"]
    assert "INDEX.md" in str(out.get("build_error"))


# ---------------------------------------------------------------------------
# The one place supplied geometry is treated differently
# ---------------------------------------------------------------------------

def _exemption_fixture(tmp_path):
    """A scene placing one staged asset and one authored crate."""
    run, rec = _staged_run(tmp_path)
    asset_ir = json.loads((run / rec["ir_path"]).read_text())
    catalog = {rec["ir_hash"]: asset_ir,
               "beef": {"id": "beef", "name": "crate",
                        "parts": [{"name": "box", "primitive": "box"}]}}
    scene_ir = {"id": "scene", "instances": [
        {"name": "the_rig", "object_ref": rec["ir_hash"]},
        {"name": "carton", "object_ref": "beef"}]}

    class Ctx:
        class layer_payload:
            pass
    Ctx.layer_payload.catalog = catalog
    return rec, scene_ir, Ctx()


def _recording_check(checked, verdict="fail"):
    from dycraft.verify.report import VerifierReport

    def check(obj_ir, ctx=None):
        checked.append(obj_ir["name"])
        return VerifierReport(name="X", verdict=verdict, ir_hash=obj_ir["id"],
                              evidence={}, suggested_repairs=["fix"])
    return check


def test_an_imported_definition_is_skipped_only_where_it_cannot_be_judged(
        tmp_path):
    """The exemption is per check, not per asset.

    `NoSelfPenetration` is unpassable and unfixable on an imported definition
    — no B-rep to adjudicate, and its repair needs a builder that does not
    exist here. Every other object check has something to say about a
    converter's output, and measured on `g1_with_dex1_v2` they say it and
    pass. So the same definition is skipped by a `skip_imported` check and
    judged by every other one.
    """
    from dycraft.verify._per_object import per_object

    rec, scene_ir, ctx = _exemption_fixture(tmp_path)

    skipping, judging = [], []
    lenient = per_object("X", _recording_check(skipping), skip_imported=True)
    strict = per_object("X", _recording_check(judging))

    report = lenient(scene_ir, ctx)
    assert skipping == ["crate"], "the asset must not be judged by this check"
    assert report.evidence["definitions_checked"] == 1

    assert strict(scene_ir, ctx).verdict == "fail"
    assert judging == ["rig_asset", "crate"], \
        "every other check still covers the asset"


def test_a_skipped_definition_is_named_rather_than_silently_dropped(tmp_path):
    """An exemption nobody can see is indistinguishable from a check that
    never ran, and "1 definition checked" in a scene that places two is a
    number the author should be able to account for."""
    from dycraft.verify._per_object import per_object

    rec, scene_ir, ctx = _exemption_fixture(tmp_path)
    report = per_object("X", _recording_check([], verdict="pass"),
                        skip_imported=True)(scene_ir, ctx)

    skipped = report.evidence["imported_definitions_skipped"]
    assert [s["object_ref"] for s in skipped] == [rec["ir_hash"]]
    assert skipped[0]["object"] == "rig_asset"
    assert skipped[0]["instances"] == ["the_rig"], \
        "named by instance, which is how the author reads the scene"


def test_authored_geometry_is_never_exempt(tmp_path):
    """The exemption is driven by the definition's own parts, so it cannot
    quietly grow to cover geometry this run built."""
    from dycraft.verify._per_object import per_object

    catalog = {"beef": {"id": "beef", "name": "crate",
                        "parts": [{"name": "box", "primitive": "box"}]}}
    scene_ir = {"id": "s", "instances": [{"name": "c", "object_ref": "beef"}]}

    class Ctx:
        class layer_payload:
            pass
    Ctx.layer_payload.catalog = catalog

    checked = []
    report = per_object("X", _recording_check(checked, verdict="pass"),
                        skip_imported=True)(scene_ir, Ctx())
    assert checked == ["crate"]
    assert "imported_definitions_skipped" not in report.evidence


# ---------------------------------------------------------------------------
# The CLI gate
# ---------------------------------------------------------------------------

def _cli_preload(layer):
    """Run the launcher far enough to see how it treats `--preload-asset`.

    `--no-scene-brief` and `--no-motion-brief` matter: on `animation`
    the launcher generates `scene.txt` and then `motion.txt` with live one-off
    sub-agents (`agent/scene_rewrite.py`, `agent/motion_rewrite.py`). The
    bundle check runs ahead of both, so this would pass without the flags —
    but only by depending on that order, and a test of argument validation
    should not spend a real API call to find out. What is under test is the
    layer gate and the bundle check, both of which are offline.
    """
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, "-m", "dycraft.cli", "--layer", layer,
         "--no-scene-brief", "--no-motion-brief",
         "--backend", "codex", "--preload-asset", "g1=/tmp/nope",
         "--prompt", "x"],
        capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("layer", ["object", "plain"])
def test_preload_asset_is_refused_outside_the_scene_and_animation_layers(layer):
    """Only the two layers that build what they place accept `--preload-asset`."""
    proc = _cli_preload(layer)
    assert proc.returncode != 0
    assert "--preload-asset is only valid with --layer=scene" in proc.stderr


@pytest.mark.parametrize("layer", ["scene", "animation"])
def test_preload_asset_is_accepted_by_the_scene_and_animation_layers(layer):
    """Reaching the bundle check means the layer gate let it through. The two
    differ only in where the bundle is staged — the run dir for `scene`, the
    `design/` source tree for `animation`."""
    proc = _cli_preload(layer)
    assert proc.returncode != 0
    assert "is not a directory" in proc.stderr, proc.stderr
    assert "only valid with" not in proc.stderr


# ---------------------------------------------------------------------------
# Anatomical overlay: narrowing a hardware range for a character
# ---------------------------------------------------------------------------

def _overlay(tmp_path, bundle, limits, *, schema=None):
    doc = {"schema": schema or "dycraft.anatomical_limits.v1",
           "object": "rig_asset", "limits": limits}
    (bundle / "anatomical_limits.json").write_text(json.dumps(doc))
    return bundle


def test_a_bundle_local_overlay_is_staged_and_narrows(tmp_path):
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, {"shoulder": {"limits_deg": [0, 45],
                                             "guards": "no backward swing"}})
    run = tmp_path / "run"
    run.mkdir()
    rec = stage_preloaded_assets([f"rig={bundle}"], run)["rig"]

    block = rec["anatomical_limits"]
    assert block["narrowed"]["shoulder"] == {"limits": [0.0, 45.0],
                                             "unit": "deg"}
    assert (run / block["path"]).is_file()


def test_an_overlay_may_only_tighten(tmp_path):
    """A range wider than the hardware's is a claim about the machine the
    machine does not support — refused in the launcher, not discovered later."""
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, {"shoulder": {"limits_deg": [-120, 120]}})
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="may only tighten"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_an_overlay_naming_an_absent_joint_is_refused(tmp_path):
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, {"knee": {"limits_deg": [0, 90]}})
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="not a joint of"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_an_overlay_must_use_the_joint_type_s_unit(tmp_path):
    """`limits_m` on a revolute joint would otherwise be read as radians and
    silently narrow the range by a factor of 57."""
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, {"shoulder": {"limits_m": [0, 1]}})
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="limits_deg"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_a_wrong_schema_tag_is_refused(tmp_path):
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, {"shoulder": {"limits_deg": [0, 45]}},
             schema="something.else.v1")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(PreloadAssetError, match="anatomical_limits.v1"):
        stage_preloaded_assets([f"rig={bundle}"], run)


def test_no_overlay_anywhere_is_the_ordinary_case(tmp_path):
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    rec = stage_preloaded_assets([f"rig={bundle}"], run)["rig"]
    assert "anatomical_limits" not in rec
    from dycraft.agent.assets.preload_asset import anatomical_limits
    assert anatomical_limits(run) == {}


# --- the verifier ----------------------------------------------------------

def _ctx(catalog, overlays):
    class Payload:
        pass
    Payload.catalog = catalog
    Payload.anatomical_limits = overlays

    class Ctx:
        layer_payload = Payload()
    return Ctx()


def _rig_scene(ir_hash, initial_deg=None):
    """A SceneIR fragment. `initial_joints` is stored in the IR's own units —
    **radians** for a revolute joint — while every authoring surface takes
    degrees. Converting here rather than writing radians inline keeps the test
    readable and pins the boundary the verifier has to get right.
    """
    inst = {"name": "rig", "object_ref": ir_hash}
    if initial_deg is not None:
        inst["initial_joints"] = {k: math.radians(v)
                                  for k, v in initial_deg.items()}
    return {"id": "scene", "instances": [inst]}


def _staged_with_overlay(tmp_path, limits):
    bundle = make_bundle(tmp_path)
    _overlay(tmp_path, bundle, limits)
    run = tmp_path / "run"
    run.mkdir()
    rec = stage_preloaded_assets([f"rig={bundle}"], run)["rig"]
    from dycraft.agent.assets.preload_asset import anatomical_limits
    ir = json.loads((run / rec["ir_path"]).read_text())
    return run, rec, {rec["ir_hash"]: ir}, anatomical_limits(run)


def test_a_committed_pose_inside_the_overlay_passes(tmp_path):
    from dycraft.verify.scene_verify import anatomical_joint_limits
    _, rec, catalog, overlays = _staged_with_overlay(
        tmp_path, {"shoulder": {"limits_deg": [0, 45]}})
    scene = _rig_scene(rec["ir_hash"], {"shoulder": 30})   # degrees
    report = anatomical_joint_limits(scene, _ctx(catalog, overlays))
    assert report.verdict == "pass"
    assert report.evidence["joints_checked"] == 1


def test_a_committed_pose_the_hardware_allows_but_the_overlay_does_not_fails(tmp_path):
    """The whole point: the mechanism really does move this way, and that is
    exactly why nothing else objects."""
    from dycraft.verify.scene_verify import anatomical_joint_limits
    _, rec, catalog, overlays = _staged_with_overlay(
        tmp_path, {"shoulder": {"limits_deg": [0, 45], "guards": "backward"}})
    scene = _rig_scene(rec["ir_hash"], {"shoulder": -60})  # deg; hardware allows -90
    report = anatomical_joint_limits(scene, _ctx(catalog, overlays))
    assert report.verdict == "fail"
    off = report.evidence["offenders"][0]
    assert off["joint"] == "shoulder" and off["instance"] == "rig"
    assert off["committed"] == pytest.approx(-60.0)
    assert off["allowed"] == [0.0, 45.0] and off["unit"] == "deg"
    assert off["guards"] == "backward"


def test_the_definition_default_is_what_is_judged_when_unposed(tmp_path):
    """An instance that sets no `initial_joints` still commits a pose — the
    definition's default — and that is what ships."""
    from dycraft.verify.scene_verify import anatomical_joint_limits
    # The fixture's `shoulder` defaults to 0.0; forbid it.
    _, rec, catalog, overlays = _staged_with_overlay(
        tmp_path, {"shoulder": {"limits_deg": [30, 45]}})
    report = anatomical_joint_limits(_rig_scene(rec["ir_hash"]),
                                     _ctx(catalog, overlays))
    assert report.verdict == "fail"
    assert report.evidence["offenders"][0]["committed"] == pytest.approx(0.0)


def test_the_check_does_not_apply_without_any_overlay(tmp_path):
    """Every run that preloads nothing, which is nearly all of them. Nothing
    to check is `not_applicable`, which passes every gate, rather than a
    `pass` that reads as though something was checked."""
    from dycraft.verify.scene_verify import anatomical_joint_limits
    report = anatomical_joint_limits({"id": "s", "instances": []},
                                     _ctx({}, {}))
    assert report.verdict == "not_applicable"
    assert "overlay" in report.evidence["not_applicable"]


def test_the_check_is_in_the_scene_suite():
    from dycraft.agent.layers.layer import scene_layer_config
    from dycraft.verify.scene_verify import ANATOMICAL_JOINT_LIMITS
    assert ANATOMICAL_JOINT_LIMITS in [n for n, _ in
                                       scene_layer_config().verifiers]


# --- the shipped G1 overlay ------------------------------------------------

def test_the_shipped_g1_overlay_is_a_valid_narrowing():
    """A guard on the data file itself. It is hand-authored editorial
    judgment, and the one rule it must obey — only ever tighten — is not
    something a reader can check by eye across 11 joints.
    """
    from dycraft.agent.assets.preload_asset import (
        ANATOMICAL_LIBRARY, ANATOMICAL_SCHEMA)
    path = ANATOMICAL_LIBRARY / "g1_humanoid_dex1.json"
    doc = json.loads(path.read_text())
    assert doc["schema"] == ANATOMICAL_SCHEMA
    assert doc["object"] == "g1_humanoid_dex1"
    for joint, row in doc["limits"].items():
        lo, hi = row["limits_deg"]
        h_lo, h_hi = row["hardware_deg"]
        assert h_lo <= lo < hi <= h_hi, joint
        assert row.get("guards"), f"{joint} must say what it guards against"
    # The joints the overlay narrows, and no others.
    assert set(doc["limits"]) == {
        f"{side}_{joint}_joint" for side in ("left", "right")
        for joint in ("knee", "elbow", "hip_pitch", "hip_roll", "hip_yaw",
                      "shoulder_pitch", "shoulder_yaw", "wrist_yaw")
    } | {"waist_yaw_joint"}
    # The G1's joint signs: knee 0 is straight and
    # positive flexes; elbow flexes toward negative and is straight near 80,
    # so its upper bound is the hyperextension guard; hip pitch positive swings
    # the leg backwards.
    assert doc["limits"]["left_knee_joint"]["limits_deg"][0] == 0
    assert doc["limits"]["left_elbow_joint"]["limits_deg"] == [-60, 80]
    assert doc["limits"]["left_hip_pitch_joint"]["limits_deg"][1] == 30


def test_a_dest_stages_away_from_the_run_dir(tmp_path):
    """`animation` rebuilds from a copy of `design/`, so its assets have
    to live inside that tree rather than beside it."""
    bundle = make_bundle(tmp_path)
    run = tmp_path / "run"
    design = run / "design"
    run.mkdir()
    rec = stage_preloaded_assets([f"rig={bundle}"], run, dest=design)["rig"]

    assert (design / rec["ir_path"]).is_file()
    assert (design / "assets").is_dir()
    assert not (run / "preloaded").exists(), "nothing at the run root"
    assert not (run / "assets").exists()
    # And the manifest readers resolve against that same root.
    assert set(load_preloaded_manifest(design)) == {"rig"}
    assert load_preloaded_manifest(run) == {}


# ---------------------------------------------------------------------------
# AnatomicalJointLimits over an animation, per frame
# ---------------------------------------------------------------------------
#
# These live here rather than beside the other native-animation tests because
# they belong to the preload feature, and keeping them in this file means they
# do not depend on `tests/test_blender_animation.py`'s fixtures or on whichever
# revision of `animation/blender/verify.py` that file currently expects.

def _joint_record(degrees, limits, hint=None):
    """One evaluated frame holding a single revolute joint at `degrees`."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    record = {"frame": 0, "objects": [],
              "transforms": {"origin": np.eye(4).tolist(), "joint": m.tolist()}}
    if hint is not None:
        record["joint_positions"] = {"joint": math.radians(hint)}
    joint = {"origin": "origin", "object": "joint", "axis": [0, 0, 1],
             "type": "revolute", "limits": [math.radians(v) for v in limits]}
    return record, joint


def _anatomical_frames(tmp_path, degrees, *, hardware, anatomical):
    """A one-frame evaluation directory plus the baseline/scene/catalog trio."""
    import numpy as np
    record, joint = _joint_record(degrees, hardware)
    if anatomical is not None:
        joint["anatomical_limits"] = [math.radians(v) for v in anatomical]
    (tmp_path / "index.json").write_text(json.dumps(
        {"fps": 1, "n_frames": 1, "scene_ref": "scene",
         "declarations": {"events": []}}))
    (tmp_path / "000000.json").write_text(json.dumps(record))
    np.savez(tmp_path / "000000.npz")
    scene = {"id": "scene",
             "instances": [{"name": "robot", "object_ref": "robot"}]}
    obj = {"joints": [{"name": "knee", "type": "revolute", "default": 0}],
           "collision_allowances": []}
    baseline = {"identities": {}, "joints": {"robot__knee": joint}}
    return baseline, scene, {"robot": obj}


def test_a_frame_the_hardware_allows_but_the_overlay_does_not_is_flagged(tmp_path):
    """The case the overlay exists for: a backward knee the mechanism really
    can reach, which is exactly why `JointLimits` has nothing to say about it.

    Reported as `uncertain`, not `fail`. The overlay describes how a G1 should
    be *seen* moving, not what its mechanism can do, so a frame outside it is
    a pose note rather than a defect — and `required_checks_pass` never
    consulted the name, so the hard verdict blocked nothing and only made the
    note read as a fault.
    """
    from dycraft.animation.blender.verify import verify
    from dycraft.animation.planning import required_checks_pass
    args = _anatomical_frames(tmp_path, -4, hardware=(-5, 165),
                              anatomical=(0, 165))
    reports = list(verify(tmp_path, *args))
    by_name = {r["name"]: r for r in reports}
    assert by_name["JointLimits"]["verdict"] == "pass", \
        "the mechanism was not driven outside its own range"
    anatomical = by_name["AnatomicalJointLimits"]
    assert anatomical["verdict"] == "uncertain"
    assert anatomical["evidence"]["diagnostic_only"] is True
    row = anatomical["evidence"]["failures"][0]
    assert row["joint"] == "robot__knee" and row["frame"] == 0
    assert row["hardware_limits"] != row["anatomical_limits"]
    # And it does not block: the required set is unaffected by the overlay.
    assert required_checks_pass(reports)


def test_a_frame_inside_the_overlay_passes(tmp_path):
    from dycraft.animation.blender.verify import verify
    args = _anatomical_frames(tmp_path, 90, hardware=(-5, 165),
                              anatomical=(0, 165))
    reports = {r["name"]: r for r in verify(tmp_path, *args)}
    assert reports["AnatomicalJointLimits"]["verdict"] == "pass"
    assert reports["JointLimits"]["verdict"] == "pass"


def test_a_joint_with_no_overlay_is_never_reported(tmp_path):
    """Every asset that declares nothing, which is the ordinary case — the
    check must cost them a pass, not a false positive."""
    from dycraft.animation.blender.verify import verify
    args = _anatomical_frames(tmp_path, -4, hardware=(-5, 165),
                              anatomical=None)
    reports = {r["name"]: r for r in verify(tmp_path, *args)}
    assert reports["AnatomicalJointLimits"]["verdict"] == "not_applicable"
    assert reports["AnatomicalJointLimits"]["evidence"]["failure_count"] == 0


def test_the_native_suite_declares_the_new_report():
    from dycraft.animation.blender.verify import VERIFIER_NAMES
    assert "AnatomicalJointLimits" in VERIFIER_NAMES
    # Beside JointLimits, which is the one it must not be confused with.
    assert VERIFIER_NAMES.index("AnatomicalJointLimits") == \
        VERIFIER_NAMES.index("JointLimits") + 1


def _bundle_with_overlay(tmp_path):
    bundle = make_bundle(tmp_path)
    (bundle / "anatomical_limits.json").write_text(json.dumps({
        "schema": "dycraft.anatomical_limits.v1", "object": "rig_asset",
        "limits": {"shoulder": {"limits_deg": [0, 45], "guards": "backward"}}}))
    return bundle


def test_the_anatomical_spec_keys_joints_by_instance(tmp_path):
    """`prepare_scene` must produce the `<instance>__<joint>` spelling the
    worker and `verify` both index by, and one overlay can be placed twice."""
    from types import SimpleNamespace
    from dycraft.animation.blender.import_scene import _anatomical_spec

    root = tmp_path / "src"
    root.mkdir()
    rec = stage_preloaded_assets(
        [f"rig={_bundle_with_overlay(tmp_path)}"], root)["rig"]
    scene = {"id": "s", "instances": [
        {"name": "left", "object_ref": rec["ir_hash"]},
        {"name": "right", "object_ref": rec["ir_hash"]},
        {"name": "crate", "object_ref": "other"}]}
    spec = _anatomical_spec(SimpleNamespace(
        asset_base_dir=root, run_dir=root, scene_ir=scene, catalog={}))
    assert set(spec) == {"left__shoulder", "right__shoulder"}, spec
    assert spec["left__shoulder"] == pytest.approx(
        [0.0, math.radians(45)]), "the spec is in IR units, not degrees"


def test_the_anatomical_spec_is_empty_without_an_overlay(tmp_path):
    from types import SimpleNamespace
    from dycraft.animation.blender.import_scene import _anatomical_spec
    assert _anatomical_spec(SimpleNamespace(
        asset_base_dir=tmp_path, run_dir=tmp_path,
        scene_ir={"id": "s", "instances": []}, catalog={})) == {}
