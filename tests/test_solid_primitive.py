"""Tests for the ``solid`` primitive — the build123d-authored B-rep part.

``solid`` is the shape tag every authored part carries, and the reason every
part has real geometry an exact collision adjudicator can consult. Three
properties these tests pin:

  * it carries **three** content-addressed assets (authored source, exact
    ``.brep``, tessellated ``.obj``), and all three participate in the object id;
  * it has **no** ``scale`` — the authored source is authoritative about size,
    so a post-hoc scale would desync the exact mass properties from geometry;
  * its inertia is the CAD kernel's **exact** tensor about the **true
    centroid**, not a box proxy about the AABB center.

The fake ``.brep`` payload below is deliberately opaque bytes: nothing in this
layer parses a B-rep, it only content-addresses and forwards it. Producing real
B-reps is the CAD worker's job and is covered by the build123d-dependent suites
(`test_live_shape_parts.py`, `test_object_worker.py`, `test_frame_contract.py`).

Parts are registered through `add_part` against a **stub compiler** rather than
a live kernel, which keeps these tests fast and kernel-free while still
exercising the real registration → staging → validation path.

Run: `pytest dycraft/tests/test_solid_primitive.py -v`
"""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dycraft.sdk import Object
from dycraft.sdk import _mesh_assets as _ma
from dycraft.sdk import _object_kernel as _ok
from dycraft.sdk import _primitives as _prim
from dycraft.sdk import _rotation as _rot
from dycraft.sdk import _scene_kernel as _sk
from dycraft.verify import _support as _sup
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.render import urdf_export as U


# Geometry: x∈[0.1,0.3], y∈[-0.2,0.2], z∈[0,0.5]. AABB center (0.2, 0, 0.25);
# the "true" centroid below is deliberately a different point (0.2, 0, 0.24) so
# a test can tell the two apart.
_OBJ = "\n".join(
    f"v {x} {y} {z}"
    for x in (0.1, 0.3) for y in (-0.2, 0.2) for z in (0.0, 0.5)
) + "\nf 1 2 4 3\n"
_BREP = b"CASCADE Topology V3 (fake payload; nothing here parses it)"
_SRC = "p = Box(0.2, 0.4, 0.5)"
_AABB_MIN = [0.1, -0.2, 0.0]
_AABB_MAX = [0.3, 0.2, 0.5]
_AABB_CENTER = (0.2, 0.0, 0.25)
_CENTROID = (0.2, 0.0, 0.24)

# volume is deliberately *not* the AABB box volume (0.2*0.4*0.5 = 0.04) — the
# whole point of a solid's inertia is that it uses the kernel's real volume
# rather than the enclosing box, so a fixture where the two coincide would let
# a proxy-fallback regression pass unnoticed.
_SHAPE = {
    "volume": 0.025,
    "area": 0.86,
    "center": list(_CENTROID),
    "inertia_diag": [0.001, 0.002, 0.003],
    "fill_ratio": 0.625,
    "face_census": {"PLANE": 6, "CYLINDER": 2},
    "topo": {"V": 8, "E": 12, "F": 6, "solids": 1, "shells": 1},
    "cylinders": [{"radius": 0.004, "axis_point": [0.2, 0.0, 0.25],
                   "axis_dir": [0.0, 1.0, 0.0], "concave": True}],
}


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_object_singleton()
    yield
    _reset_object_singleton()
    _ma.clear_active_mesh_session()
    _ma.clear_active_solid_compiler()


class _StubCompiler:
    """The `_solid_compile` surface `add_part` and `finalize` actually call.

    Two methods and one exception type — the same two-stage split the real
    compiler has: `measure_shape` needs the (here notional) Shape, and
    `compile_entry` needs only numbers. Canned bytes stand in for tessellation
    and B-rep serialization, which is the part a CAD kernel is needed for and
    the part nothing in this layer inspects.
    """

    SolidCompileError = RuntimeError

    def __init__(self, *, obj=_OBJ, brep=_BREP, shape=None):
        self._obj, self._brep = obj, brep
        self._shape = dict(_SHAPE if shape is None else shape)

    def measure_shape(self, shape, *, name, tolerance=1e-3,
                      angular_tolerance=0.1, planar_max_edge=None,
                      texture_scale_m=None):
        # The stub tracks the real `measure_shape` signature deliberately: it
        # stands in for the compiler, so a parameter the real one grows has to
        # arrive here too, or a compiler that silently ignores it would pass.
        self.planar_max_edge = planar_max_edge
        self.texture_scale_m = texture_scale_m
        return {"name": name, "table": self._shape,
                "aabb_min": list(_AABB_MIN), "aabb_max": list(_AABB_MAX)}

    def compile_entry(self, entry):
        return {"obj_bytes": self._obj.encode("utf-8"),
                "brep_bytes": self._brep,
                "aabb_min": entry["aabb_min"], "aabb_max": entry["aabb_max"],
                "shape": entry["table"]}


def _build(run_dir, *, source=_SRC, **kw) -> dict:
    """One solid part named `panel`, registered and finalized in `run_dir`."""
    _reset_object_singleton()
    # `_source_text_for_staging` stages the run's own object.py as the
    # source asset, so the file has to exist.
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "object.py").write_text(source)
    _ma.set_active_mesh_session(_ma.MeshAssetSession(run_dir))
    _ma.set_active_solid_compiler(_StubCompiler(**kw))
    try:
        o = Object("panel_obj")
        o.declare_root(o.add_part("panel", object()))
        return o.finalize()
    finally:
        _ma.clear_active_mesh_session()
        _ma.clear_active_solid_compiler()


def _solid_params(ir):
    return ir["parts"][0]["params"]


def _build_with_material(run_dir, material):
    """`_build`, but binding a material — and handing back the stub to inspect.

    Separate rather than a keyword on `_build` because what it exists to observe
    is the compiler CALL, not the resulting IR, and every other test here reads
    the IR.
    """
    _reset_object_singleton()
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "object.py").write_text(_SRC)
    _ma.set_active_mesh_session(_ma.MeshAssetSession(run_dir))
    stub = _StubCompiler()
    _ma.set_active_solid_compiler(stub)
    try:
        o = Object("panel_obj")
        o.declare_root(o.add_part("panel", object(), material=material))
        o.finalize()
        return stub
    finally:
        _ma.clear_active_mesh_session()
        _ma.clear_active_solid_compiler()


def test_the_materials_tile_size_reaches_the_compiler(tmp_path):
    """`add_part` binds the material; the compiler derives the UVs.

    Texture coordinates are baked into the staged OBJ, which means the compiler
    needs one fact about the surface it is otherwise indifferent to: how much
    real geometry one tile of the image spans. This is the only channel for it,
    and a silent None would not fail anything — the part would simply stage
    without coordinates and render untextured, which looks exactly like a
    material that was never bound.
    """
    from dycraft.sdk import Mapping, Material, Texture, TextureMap

    textured = Material(
        color=(150, 105, 70),
        texture=Texture(map_items=(("base_color", TextureMap(
            image="assets/textures/" + "a" * 32 + ".jpg", digest="a" * 32)),)),
        mapping=Mapping(scale_m=0.25))
    assert _build_with_material(tmp_path / "tex", textured).texture_scale_m == 0.25

    # An untextured part passes None, which keeps its staged bytes free of
    # UVs.
    plain = Material(color=(150, 105, 70))
    assert _build_with_material(tmp_path / "plain", plain).texture_scale_m is None


# ---------------------------------------------------------------------------
# Registry: validation, AABB, inertia, inertia origin
# ---------------------------------------------------------------------------

def _bare_params():
    return {"brep": "assets/brep/b.brep",
            "mesh": "assets/meshes/c.obj", "digest": "b",
            "aabb_min": list(_AABB_MIN), "aabb_max": list(_AABB_MAX),
            "shape": dict(_SHAPE)}


def test_solid_rejects_scale():
    """A solid must never carry `scale` — it would desync `shape` from geometry."""
    with pytest.raises(ValueError, match="must not carry 'scale'"):
        _prim.validate_params("solid", dict(_bare_params(), scale=[1.0, 1.0, 1.0]))
    # ...even an identity scale, which is the tempting "harmless" case.
    with pytest.raises(ValueError, match="must not carry 'scale'"):
        _prim.validate_params("solid", dict(_bare_params(), scale=[2.0, 1.0, 1.0]))


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.pop("brep"), "brep"),
    (lambda p: p.update(mesh=""), "mesh"),
    (lambda p: p.pop("shape"), "shape"),
    (lambda p: p["shape"].pop("inertia_diag"), "inertia_diag"),
    (lambda p: p["shape"].update(volume=-1.0), "volume"),
    (lambda p: p["shape"].update(inertia_diag=[1.0, -2.0, 3.0]), "inertia_diag"),
    (lambda p: p.update(aabb_max=[0.0, 0.2, 0.5]), "aabb_max"),
])
def test_solid_param_validation_rejects_malformed(mutate, match):
    params = _bare_params()
    mutate(params)
    with pytest.raises(ValueError, match=match):
        _prim.validate_params("solid", params)


def test_solid_shape_table_passes_extra_keys_through():
    """Only the mass properties are validated; the feature inventory rides along."""
    out = _prim.validate_params("solid", _bare_params())
    assert out["shape"]["face_census"] == {"PLANE": 6, "CYLINDER": 2}
    assert out["shape"]["cylinders"][0]["radius"] == 0.004


def test_solid_aabb_is_stored_verbatim_and_may_be_offcenter():
    params = _bare_params()
    assert _prim.local_aabb("solid", params) == (tuple(_AABB_MIN), tuple(_AABB_MAX))


def test_solid_inertia_is_exact_not_a_box_proxy():
    params = _bare_params()
    mass, ixx, iyy, izz = _prim.inertia("solid", params, 100.0)
    # mass = density * the CAD kernel's exact volume, not the AABB box volume.
    assert mass == pytest.approx(100.0 * _SHAPE["volume"])
    aabb_box_mass = 100.0 * 0.2 * 0.4 * 0.5
    assert mass != pytest.approx(aabb_box_mass), (
        "a solid must not fall back to the mesh-style AABB box proxy")
    assert (ixx, iyy, izz) == pytest.approx((0.1, 0.2, 0.3))


def test_zero_volume_solid_falls_back_to_a_clamped_box_proxy():
    """A shell/sheet body still needs non-zero mass or the URDF is unusable."""
    params = _bare_params()
    params["shape"] = dict(_SHAPE, volume=0.0)
    mass, ixx, iyy, izz = _prim.inertia("solid", params, 100.0)
    assert mass == pytest.approx(100.0 * 0.2 * 0.4 * 0.5)
    assert min(ixx, iyy, izz) > 0.0
    # ...and the fallback body is about the AABB center, matching `inertia`.
    assert _prim.inertia_origin("solid", params) == pytest.approx(_AABB_CENTER)


def test_inertia_origin_is_per_tag():
    """The two tags put their inertial body at different points, which is what
    a caller emitting a URDF body has to offset its `<origin>` by."""
    assert _prim.inertia_origin("solid", _bare_params()) == pytest.approx(_CENTROID)
    # A solid's inertial sits at the true centroid, NOT the AABB center.
    assert _prim.inertia_origin("solid", _bare_params()) != pytest.approx(
        _AABB_CENTER)
    assert _prim.inertia_origin("box", {"width": 1, "length": 2, "height": 3}) \
        == (0.0, 0.0, 0.0)


def test_render_asset_key_maps_each_asset_tag():
    assert _prim.render_asset_key("solid") == "mesh"
    assert _prim.render_asset_key("imported") == "mesh"
    assert _prim.render_asset_key("box") is None
    assert set(_prim.ASSET_PRIMITIVES) == {"solid", "imported"}
    assert _prim.is_known("solid") and _prim.is_known("box")
    assert _prim.is_known("imported")
    # `mesh` and `cylinder` are retired tags, not renamed ones — `imported` is a
    # new tag with a different params shape, so neither name may come back.
    assert not _prim.is_known("cylinder") and not _prim.is_known("mesh")


def test_collide_asset_key_diverges_from_render_only_for_imported():
    """A solid collides the mesh it renders; an imported part does not.

    Pinned because the divergence is invisible at runtime — colliding an
    imported part's visual mesh answers *almost* the same question, just ~38x
    slower and wrong wherever that mesh is an open shell.
    """
    assert _prim.collide_asset_key("solid") == _prim.render_asset_key("solid")
    assert _prim.collide_asset_key("imported") == "collision"
    assert _prim.collide_asset_key("imported") != _prim.render_asset_key("imported")
    assert _prim.collide_asset_key("box") is None


# ---------------------------------------------------------------------------
# Builder + staging + identity
# ---------------------------------------------------------------------------

def test_add_part_stages_three_content_addressed_assets():
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
        params = _solid_params(ir)
        assert params["brep"].startswith("assets/brep/")
        assert params["mesh"].startswith("assets/meshes/")
        # The authored program is ONE asset for the whole object, so it hangs
        # off the IR rather than off each part: one `object.py` builds them all.
        assert "source" not in params
        assert ir["source"].startswith("assets/src/")
        assert (Path(run) / ir["source"]).is_file()
        for key in ("brep", "mesh"):
            assert (Path(run) / params[key]).is_file(), f"{key} not staged"
        # `digest` is the B-rep digest — the exact geometry is the identity the
        # collision adjudicator's cache keys off.
        assert params["brep"] == f"assets/brep/{params['digest']}.brep"
        assert "scale" not in params


def test_solid_identity_is_content_based_across_all_three_assets():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        base = _build(a)
        same = _build(b)                                   # different run dir
    assert base["id"] == same["id"], (
        "identical content in a different run dir must hash the same")

    for label, kw in (
        ("source",       dict(source=_SRC.replace("0.5", "0.6"))),
        ("brep",         dict(brep=_BREP + b" perturbed")),
        ("tessellation", dict(obj=_OBJ.replace("0.3", "0.30001"))),
        ("shape",        dict(shape=dict(_SHAPE, volume=0.05))),
    ):
        with tempfile.TemporaryDirectory() as d:
            other = _build(d, **kw)
        assert other["id"] != base["id"], (
            f"a {label}-only change must perturb the object id")


def test_solid_flows_through_the_kernel_like_any_other_part():
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
    assert _ok.local_bbox(ir, "panel") == (tuple(_AABB_MIN), tuple(_AABB_MAX))
    # obb_of_part places an off-center part's collider at t + R·c.
    center, _R, half = _ok.obb_of_part(ir, "panel")
    assert center == pytest.approx(_AABB_CENTER)
    assert half == pytest.approx((0.1, 0.2, 0.25))


# ---------------------------------------------------------------------------
# Exposing real geometry: probes and the scene-layer handle
# ---------------------------------------------------------------------------

def test_part_shape_and_source_path_are_solid_only():
    from dycraft.sdk import _probes
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
    table = _probes.object_part_shape(ir, "panel")
    assert table["fill_ratio"] == _SHAPE["fill_ratio"]
    assert table["cylinders"][0]["concave"] is True
    assert _probes.object_source_path(ir).startswith("assets/src/")

    # A primitive part has no derived facts — its params say everything.
    _reset_object_singleton()
    o = Object("boxy")
    o.declare_root(o.make_box("b", 1.0, 1.0, 1.0))
    box_ir = o.finalize()
    assert _probes.object_part_shape(box_ir, "b") is None
    assert _probes.object_source_path(box_ir) is None
    with pytest.raises(KeyError):
        _probes.object_part_shape(box_ir, "nope")


def test_object_source_probe_returns_text_not_a_path():
    """The overlay closure reads the staged file; a bare path would make the
    agent spend a `read_file` round trip to see what it just authored."""
    from dycraft.agent.layers.layer import _object_probe_overlay

    class _Ctx:
        def __init__(self, run_dir):
            self.run_dir = Path(run_dir)

    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
        overlay = _object_probe_overlay(_Ctx(run), ir)
        assert overlay["object_source"]() == _SRC
        # Non-solid parts and unreadable assets must return, not raise.
        assert overlay["object_source"]() is not None
        broken = _object_probe_overlay(_Ctx("/nonexistent"), ir)
        assert broken["object_source"]().startswith("assets/src/")


def test_object_handle_exposes_shape_facts_to_the_scene_layer():
    """The scene layer sees shape facts, not only AABBs, so it can tell a mesh
    part from a box, and see that most of a carcass's box is air."""
    from dycraft.sdk import ObjectHandle
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
    handle = ObjectHandle(ir)
    assert handle.part_shape("panel")["volume"] == _SHAPE["volume"]
    assert sorted(handle.part_shapes) == ["panel"]
    # The handle exposes the recipe's PATH, never its text: a scene places
    # objects rather than re-authoring them, so inlining a couple of hundred
    # lines per placed object would dominate its context. The path costs
    # nothing and makes `read_file` an opt-in decision.
    assert not hasattr(handle, "part_source")
    assert handle.source().startswith("assets/src/")
    assert handle.source().endswith(".py")


# ---------------------------------------------------------------------------
# URDF export
# ---------------------------------------------------------------------------

def _link_bodies(ir):
    xml = U.object_ir_to_urdf(ir)
    root = ET.fromstring(xml) if isinstance(xml, str) else xml
    link = next(iter(root.iter("link")))
    return {tag: link.find(tag) for tag in ("visual", "collision", "inertial")}


def test_solid_urdf_visual_is_the_obj_collision_is_a_box_inertial_is_exact():
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
    bodies = _link_bodies(ir)

    vis_geom = list(bodies["visual"].find("geometry"))[0]
    assert vis_geom.tag == "mesh"
    assert vis_geom.attrib["filename"] == _solid_params(ir)["mesh"]
    assert vis_geom.attrib["scale"] == "1 1 1"
    # The OBJ vertices already carry the offset, so the visual sits at the
    # part origin while the two proxy bodies are offset.
    assert [float(v) for v in bodies["visual"].find("origin").attrib["xyz"].split()] \
        == pytest.approx((0.0, 0.0, 0.0))

    coll_geom = list(bodies["collision"].find("geometry"))[0]
    assert coll_geom.tag == "box", "collision stays the cheap AABB proxy"
    assert [float(v) for v in bodies["collision"].find("origin").attrib["xyz"].split()] \
        == pytest.approx(_AABB_CENTER)

    inertial = bodies["inertial"]
    assert float(inertial.find("mass").attrib["value"]) == pytest.approx(
        U._DEFAULT_DENSITY * _SHAPE["volume"])
    # The distinguishing assertion: the inertial origin is the true centroid,
    # NOT the AABB center the collision proxy uses.
    xyz = [float(v) for v in inertial.find("origin").attrib["xyz"].split()]
    assert xyz == pytest.approx(_CENTROID)
    assert xyz != pytest.approx(_AABB_CENTER)


def test_has_localizable_assets_covers_every_asset_tag():
    """The guard that gates localization must not drift from what it gates.

    Keep this function and its call sites together. Spelled inline as a
    comparison against one tag, a guard like this stops covering any tag added
    later: the object then skips localization and exports a URDF whose mesh
    paths resolve to a directory that was never created next to it, and the
    parts simply fail to load.
    """
    with tempfile.TemporaryDirectory() as run:
        solid_ir = _build(run)
    assert U.has_localizable_assets(solid_ir) is True

    _reset_object_singleton()
    o = Object("boxy")
    o.declare_root(o.make_box("b", 1.0, 1.0, 1.0))
    assert U.has_localizable_assets(o.finalize()) is False

    assert U.has_localizable_assets(
        {"parts": [{"primitive": "solid", "params": {}}]}) is True
    assert U.has_localizable_assets({"parts": []}) is False
    # An unknown tag must not be treated as asset-backed.
    assert U.has_localizable_assets(
        {"parts": [{"primitive": "cylinder", "params": {}}]}) is False


def test_exported_urdf_mesh_paths_resolve_next_to_it():
    """End to end for the guard: every <mesh filename> in an exported URDF must
    exist relative to the URDF's own directory, or the file is unloadable."""
    import xml.etree.ElementTree as ET

    with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as out:
        ir = _build(run)
        out = Path(out)
        ir_for_urdf = ir
        if U.has_localizable_assets(ir):
            ir_for_urdf = U.localize_object_meshes(
                ir, source_base_dir=run,
                dest_dir=out / "assets" / "meshes", urdf_dir=out)
        (out / "ir.urdf").write_text(U.object_ir_to_urdf(ir_for_urdf))

        root = ET.fromstring((out / "ir.urdf").read_text())
        names = [m.attrib["filename"] for m in root.iter("mesh")]
        assert names, "fixture sanity: a solid part must emit a <mesh> visual"
        for rel in names:
            assert (out / rel).is_file(), (
                f"{rel} does not resolve relative to the URDF — the export is "
                f"not self-contained")


def test_localize_object_meshes_rewrites_a_solids_obj_path():
    with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as out:
        ir = _build(run)
        missing: list = []
        localized = U.localize_object_meshes(
            ir, source_base_dir=run, dest_dir=Path(out) / "meshes",
            urdf_dir=out, missing=missing)
        assert not missing
        new_rel = _solid_params(localized)["mesh"]
        assert new_rel != _solid_params(ir)["mesh"]
        assert (Path(out) / new_rel).is_file()
        # The B-rep and source are deliberately NOT copied: nothing downstream
        # of a URDF reads them.
        assert _solid_params(localized)["brep"] == _solid_params(ir)["brep"]
        assert localized["id"] == ir["id"], "export rewrite must not re-hash"


# ---------------------------------------------------------------------------
# Off-center parts in both collision kernels
#
# `local_aabb` returns a solid's stored AABB verbatim, so a solid may sit away
# from its part origin. Every collider placement therefore carries the
# `t + R·c` term, where `c` is the part-local AABB midpoint. A `box` is
# origin-centered, so `c` is zero and the term is invisible for it — these
# tests are what keep it correct for the tag every run actually produces.
# ---------------------------------------------------------------------------

def test_object_kernel_places_an_offcenter_collider_at_t_plus_rc():
    with tempfile.TemporaryDirectory() as run:
        solid = _build(run)["parts"][0]
    ir = {"parts": [
        solid,
        {"id": 1, "name": "b", "primitive": "box",
         "params": {"width": 0.2, "length": 0.2, "height": 0.2},
         "material": {"color": [1, 1, 1]}},
    ]}
    name = solid["name"]
    ident = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    cs, _, hs = _ok.obb_of_part(ir, name, frames={name: ident, "b": ident})
    cb, _, _ = _ok.obb_of_part(ir, "b", frames={name: ident, "b": ident})
    assert all(abs(a - b) < 1e-12 for a, b in zip(cs, (0.2, 0.0, 0.25)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(hs, (0.1, 0.2, 0.25)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(cb, (0.0, 0.0, 0.0)))

    # Rotated frame: the center offset rotates with the part.
    q = _rot.quat_from_axis_angle_rad((0, 0, 1), math.pi / 2)
    cs2, _, _ = _ok.obb_of_part(ir, name, frames={name: ((1.0, 2.0, 3.0), q)})
    assert all(abs(a - b) < 1e-9 for a, b in zip(cs2, (1.0, 2.2, 3.25)))


def test_support_scores_the_true_gap_for_an_offcenter_part():
    """A point to one side of an off-center part must score the real distance.

    Measuring from the part origin with a half-extent (`abs(d) - half`) scores
    zero here, because the origin is outside the geometry.
    """
    with tempfile.TemporaryDirectory() as run:
        solid = _build(run)["parts"][0]
    name = solid["name"]
    d = _sup._joint_anchor_offset_distance(
        {"parts": [solid]},
        {"parent_anchor": {"translation": [0.0, 0.0, 0.25],
                           "rotation": [1, 0, 0, 0]},
         "child_anchor": {"translation": [0.2, 0.0, 0.25],
                          "rotation": [1, 0, 0, 0]},
         "parent": name, "child": name})
    assert abs(d - 0.1) < 1e-12

    box = {"id": 0, "name": "b", "primitive": "box",
           "params": {"width": 0.2, "length": 0.2, "height": 0.2},
           "material": {}}
    d_box = _sup._joint_anchor_offset_distance(
        {"parts": [box]},
        {"parent_anchor": {"translation": [0.3, 0, 0],
                           "rotation": [1, 0, 0, 0]},
         "child_anchor": {"translation": [0.0, 0, 0],
                          "rotation": [1, 0, 0, 0]},
         "parent": "b", "child": "b"})
    assert abs(d_box - 0.2) < 1e-12


def test_scene_kernel_agrees_with_the_object_kernel_on_an_offcenter_part():
    """The scene layer composes the instance pose onto the same `t + R·c`.

    Dropping the rotated AABB-center term returns the bare composed
    translation, which is only wrong when the instance is rotated — so the
    instance pose here is deliberately not the identity.
    """
    with tempfile.TemporaryDirectory() as run:
        ir = _build(run)
    name = ir["parts"][0]["name"]
    obj_ir = {"root": name, "joints": [], "parts": [ir["parts"][0]]}
    catalog = {"panel": obj_ir}
    scene_ir = {"instances": [{"name": "inst", "object_ref": "panel",
                               "placement": None}]}
    q = _rot.quat_from_axis_angle_rad((0, 0, 1), math.pi / 2)
    inst_pose = ((10.0, 0.0, 0.0), q)
    # Pre-seed the memo so `_resolve_world_pose` returns this pose directly and
    # the test exercises only the OBB assembly.
    obbs = _sk._instance_part_obbs(scene_ir, catalog, "inst",
                                   {"inst": inst_pose})
    (_pname, (center, _axes, half)), = obbs

    ident = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    c_obj, _, _ = _ok.obb_of_part(obj_ir, name, frames={name: ident})
    expected = tuple(t + rc for t, rc in
                     zip(inst_pose[0], _rot.quat_rotate_vec(q, c_obj)))
    assert all(abs(a - b) < 1e-9 for a, b in zip(center, expected)), center
    # Concretely (10, 0.2, 0.25) — not the composed translation (10, 0, 0).
    assert all(abs(a - b) < 1e-9 for a, b in zip(center, (10.0, 0.2, 0.25)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(half, (0.1, 0.2, 0.25)))


def test_two_parts_sharing_geometry_stage_the_assets_once():
    """Content addressing dedups, the way mirrored parts do in real runs."""
    with tempfile.TemporaryDirectory() as run:
        _reset_object_singleton()
        Path(run, "object.py").write_text(_SRC)
        _ma.set_active_mesh_session(_ma.MeshAssetSession(run))
        _ma.set_active_solid_compiler(_StubCompiler())
        try:
            o = Object("pair")
            left = o.add_part("left", object())
            right = o.add_part("right", object())
            o.declare_root(left)
            o.attach_fixed("j", parent=left, child=right,
                           parent_anchor=o.anchor(left, "right"),
                           child_anchor=o.anchor(right, "left"))
            ir = o.finalize()
        finally:
            _ma.clear_active_mesh_session()
            _ma.clear_active_solid_compiler()

        a, b = (p["params"] for p in ir["parts"])
        assert a["brep"] == b["brep"] and a["mesh"] == b["mesh"]
        assert len(list(Path(run, "assets", "brep").iterdir())) == 1
        assert len(list(Path(run, "assets", "meshes").iterdir())) == 1


