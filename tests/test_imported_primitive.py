"""The `imported` primitive: params contract, geometry facts, asset routing.

`imported` is the tag `agent/assets/import_urdf.py` writes for a
third-party asset's meshes — `solid` minus `brep` and `source`, plus a separate
collision triangulation. It is deliberately *not* authorable: no SDK surface
produces it (`Object.add_part` takes a live build123d Shape and nothing else),
so these tests exercise the consumer side only.

The fixtures are built inline rather than read from `assets/imported/`, which is
gitignored — the suite must stay hermetic. A real bundle is exercised
end-to-end separately.
"""

from pathlib import Path

import pytest

from dycraft.sdk import _object_kernel as _ok
from dycraft.sdk import _primitives as _prim
from dycraft.agent.layers.scene_context import referenced_object_assets
from dycraft.verify import _exact_adjudicator as _adj


_AABB_MIN = (-0.04, -0.03, 0.00)          # deliberately off-center in z, like a
_AABB_MAX = (0.04, 0.03, 0.21)            # real link whose origin is at a joint

_SHAPE = {
    "volume": 2.0e-4,
    "center": [0.001, -0.002, 0.105],     # true centroid, NOT the AABB midpoint
    "inertia_diag": [3.0e-7, 4.0e-7, 1.0e-7],
    "n_vertices": 152,
    "n_triangles": 300,
    "closed": True,
}


def _imported_params(**over):
    p = {
        "mesh": "assets/meshes/aaaa.obj",
        "collision": "assets/collision/bbbb.obj",
        "digest": "aaaa",
        "collision_digest": "bbbb",
        "aabb_min": list(_AABB_MIN),
        "aabb_max": list(_AABB_MAX),
        "shape": dict(_SHAPE),
    }
    p.update(over)
    return p


def _imported_ir():
    return {
        "schema": "dycraft.object.v1",
        "id": "deadbeef",
        "name": "panda_arm",
        "root": "link0",
        "parts": [
            {"id": 0, "name": "link0", "primitive": "imported",
             "params": _imported_params(),
             "material": {"color": [255, 255, 255], "metallic": 0.0,
                          "roughness": 0.5, "alpha": 1.0}},
            {"id": 1, "name": "link1", "primitive": "imported",
             "params": _imported_params(mesh="assets/meshes/cccc.obj",
                                        collision="assets/collision/dddd.obj",
                                        digest="cccc", collision_digest="dddd"),
             "material": {"color": [40, 40, 40], "metallic": 0.0,
                          "roughness": 0.5, "alpha": 1.0}},
        ],
        # A tree, not two loose parts: the URDF exporter walks parts from the
        # root through the joints and KeyErrors on an unreachable one.
        "joints": [
            {"id": 0, "name": "j1", "type": "revolute",
             "parent": "link0", "child": "link1",
             # `parent_anchor` = the joint's pose in the parent part's frame,
             # `child_anchor` = identity, which is exactly what import_urdf
             # writes (a URDF <origin> maps term-for-term onto this pair).
             "parent_anchor": {"translation": [0.0, 0.0, 0.21],
                               "rotation": [1.0, 0.0, 0.0, 0.0]},
             "child_anchor": {"translation": [0.0, 0.0, 0.0],
                              "rotation": [1.0, 0.0, 0.0, 0.0]},
             "axis": [0.0, 0.0, 1.0],
             "limits": [-2.9671, 2.9671],
             "default": 0.0},
        ],
        "assertions": [],
        "collision_allowances": [],
    }


# ---------------------------------------------------------------------------
# Params contract
# ---------------------------------------------------------------------------

def test_well_formed_imported_params_validate():
    out = _prim.validate_params("imported", _imported_params())
    assert out["mesh"] == "assets/meshes/aaaa.obj"
    assert out["collision"] == "assets/collision/bbbb.obj"
    assert out["aabb_min"] == _AABB_MIN and out["aabb_max"] == _AABB_MAX
    assert out["shape"]["volume"] == pytest.approx(2.0e-4)


def test_imported_shape_table_passes_extra_keys_through():
    """As for a solid: only the mass properties are validated, the rest rides."""
    out = _prim.validate_params("imported", _imported_params())
    assert out["shape"]["n_triangles"] == 300
    assert out["shape"]["closed"] is True


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.pop("mesh"), "mesh"),
    (lambda p: p.pop("collision"), "collision"),
    (lambda p: p.update(collision=""), "collision"),
    (lambda p: p.pop("digest"), "digest"),
    (lambda p: p.pop("collision_digest"), "collision_digest"),
    (lambda p: p.pop("aabb_min"), "aabb_min"),
    (lambda p: p.pop("shape"), "shape"),
    (lambda p: p["shape"].pop("inertia_diag"), "inertia_diag"),
    (lambda p: p["shape"].update(volume=-1.0), "volume"),
    # x: -0.05 < aabb_min's -0.04, so the box is inverted on that axis.
    (lambda p: p.update(aabb_max=[-0.05, 0.03, 0.21]), "aabb_max"),
])
def test_imported_param_validation_rejects_malformed(mutate, match):
    params = _imported_params()
    mutate(params)
    with pytest.raises(ValueError, match=match):
        _prim.validate_params("imported", params)


def test_collision_is_required_not_defaulted_to_mesh():
    """The one default that must not exist.

    Falling back to `mesh` would make the narrow phase collide the full-detail
    visual mesh — for the Panda, 102,752 triangles instead of 2,680, and for the
    links whose visual mesh is an open shell, a containment query about a surface
    with no inside. A loud error is the only acceptable behavior.
    """
    params = _imported_params()
    del params["collision"]
    with pytest.raises(ValueError):
        _prim.validate_params("imported", params)


@pytest.mark.parametrize("key,value", [
    ("brep", "assets/brep/x.brep"),
    ("source", "assets/src/x.py"),
    ("scale", [1.0, 1.0, 1.0]),          # even identity, the tempting case
    ("scale", [2.0, 1.0, 1.0]),
])
def test_imported_rejects_solid_only_and_scale_keys(key, value):
    with pytest.raises(ValueError, match=f"must not carry '{key}'"):
        _prim.validate_params("imported", _imported_params(**{key: value}))


# ---------------------------------------------------------------------------
# Geometry facts: one expression for both asset tags
# ---------------------------------------------------------------------------

def test_local_aabb_is_the_stored_box_including_off_center():
    assert _prim.local_aabb("imported", _imported_params()) == (_AABB_MIN, _AABB_MAX)


def test_inertia_and_origin_match_a_solid_with_the_same_shape_table():
    """The payoff of keeping the `shape` fact table on the imported tag.

    Same table in, same numbers out — asserted rather than assumed, because the
    alternative design (an AABB box proxy, as the retired `mesh` tag used) would
    still return plausible numbers here and differ silently.
    """
    solid_params = {"source": "assets/src/a.py", "brep": "assets/brep/b.brep",
                    "mesh": "assets/meshes/c.obj", "digest": "c",
                    "aabb_min": list(_AABB_MIN), "aabb_max": list(_AABB_MAX),
                    "shape": dict(_SHAPE)}
    imported_params = _imported_params()
    for density in (1.0, 1000.0):
        assert (_prim.inertia("imported", imported_params, density)
                == _prim.inertia("solid", solid_params, density))
    assert (_prim.inertia_origin("imported", imported_params)
            == _prim.inertia_origin("solid", solid_params))
    # ...and it is the true centroid, not the AABB midpoint.
    assert _prim.inertia_origin("imported", imported_params) == (0.001, -0.002, 0.105)


def test_zero_volume_imported_part_falls_back_to_its_own_aabb_box():
    """The fallback must bound *this* part, not `PRIMITIVE_SOLID`'s params."""
    params = _imported_params(shape=dict(_SHAPE, volume=0.0))
    mass, ixx, iyy, izz = _prim.inertia("imported", params, 1000.0)
    w, l, h = (0.08, 0.06, 0.21)
    assert mass == pytest.approx(1000.0 * w * l * h)
    assert ixx == pytest.approx(mass / 12.0 * (l * l + h * h))


# ---------------------------------------------------------------------------
# The two asymmetric kernel accessors — these ARE the "no brep / no refine"
# semantics, so they are pinned rather than left to the implementation.
# ---------------------------------------------------------------------------

def test_part_shape_is_exposed_but_object_source_path_is_not():
    ir = _imported_ir()
    assert _ok.part_shape(ir, "link0")["volume"] == pytest.approx(2.0e-4)
    assert _ok.object_source_path(ir) is None


# ---------------------------------------------------------------------------
# Asset routing
# ---------------------------------------------------------------------------

def _stage(tmp_path: Path) -> Path:
    for rel in ("assets/meshes/aaaa.obj", "assets/collision/bbbb.obj"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return tmp_path


def test_mesh_backend_resolves_the_collision_asset_not_the_visual(tmp_path):
    part = _imported_ir()["parts"][0]
    spec = _adj.geometry_spec(part, _stage(tmp_path), backend=_adj.BACKEND_MESH)
    assert spec is not None
    assert Path(spec["mesh"]).name == "bbbb.obj"
    assert Path(spec["mesh"]).parent.name == "collision"


def test_brep_backend_declines_an_imported_part(tmp_path):
    """No exact B-rep adjudication, so the conservative SAT verdict stands.

    This is what keeps an imported asset out of the object layer, which asks for
    the brep backend explicitly.
    """
    part = _imported_ir()["parts"][0]
    assert _adj.geometry_spec(part, _stage(tmp_path),
                             backend=_adj.BACKEND_BREP) is None


def test_solid_mesh_backend_routing_is_unchanged(tmp_path):
    """Regression guard for the `_COLLIDE_ASSET_KEY` indirection.

    A solid must still collide `params["mesh"]` — the table exists to let the
    imported tag diverge, not to change what a solid does.
    """
    (tmp_path / "assets/meshes").mkdir(parents=True)
    (tmp_path / "assets/meshes/c.obj").write_text("v 0 0 0\nf 1 1 1\n")
    part = {"primitive": "solid",
            "params": {"source": "assets/src/a.py", "brep": "assets/brep/b.brep",
                       "mesh": "assets/meshes/c.obj", "digest": "c",
                       "aabb_min": list(_AABB_MIN), "aabb_max": list(_AABB_MAX),
                       "shape": dict(_SHAPE)}}
    spec = _adj.geometry_spec(part, tmp_path, backend=_adj.BACKEND_MESH)
    assert Path(spec["mesh"]).name == "c.obj"


def test_referenced_assets_carry_both_meshes():
    got = referenced_object_assets(_imported_ir())
    assert set(p.as_posix() for p in got) == {
        "assets/meshes/aaaa.obj", "assets/collision/bbbb.obj",
        "assets/meshes/cccc.obj", "assets/collision/dddd.obj",
    }


@pytest.mark.parametrize("bad", [
    "/abs/meshes/x.obj",
    "../outside/x.obj",
    "meshes/x.obj",              # not under assets/
])
def test_referenced_assets_rejects_unsafe_collision_paths(bad):
    ir = _imported_ir()
    ir["parts"][0]["params"]["collision"] = bad
    with pytest.raises(ValueError):
        referenced_object_assets(ir)


def test_referenced_assets_rejects_a_missing_collision_key():
    ir = _imported_ir()
    del ir["parts"][0]["params"]["collision"]
    with pytest.raises(ValueError, match="collision"):
        referenced_object_assets(ir)


# ---------------------------------------------------------------------------
# URDF export
# ---------------------------------------------------------------------------

def test_urdf_emits_the_shipped_collision_mesh_and_a_centroid_inertial():
    """Visual mesh, the asset's own collision mesh, centroid inertial.

    `_emit_link_body` takes its asset branch for an imported part too. The
    analytic fallback would emit (a) its full-detail VISUAL mesh as a
    non-convex `<collision>` and (b) an `<inertial>` at the part origin instead
    of the true centroid, putting the mass where the geometry is not.

    **The collision body is the asset's shipped proxy**, not the `<box>` AABB a
    `solid` gets. The proxy is convex, closed where the visual often is not,
    and far lighter, and it is the body DyCraft's own narrow phase measures, so
    emitting anything else would make `h.ik` reason about a different robot
    from the verifiers. Two origins, not three: a mesh needs no centering
    offset because its vertices carry one.
    """
    import xml.etree.ElementTree as ET

    from dycraft.render import urdf_export as U

    ir = _imported_ir()
    root = ET.fromstring(U.object_ir_to_urdf(ir))
    link = root.findall("link")[0]

    assert [g.tag for g in link.find("visual/geometry")] == ["mesh"]
    assert [g.tag for g in link.find("collision/geometry")] == ["mesh"]

    params = ir["parts"][0]["params"]
    assert link.find("visual/geometry/mesh").get("filename") == params["mesh"]
    assert (link.find("collision/geometry/mesh").get("filename")
            == params["collision"])

    inert = [float(v) for v in link.find("inertial/origin").get("xyz").split()]
    assert inert == pytest.approx(_prim.inertia_origin("imported", params))
    # Neither mesh carries an offset — the OBJ vertices already do. Reusing the
    # box proxy's AABB-center origin here would apply the centering twice; see
    # `_promote_mesh_collisions`, which measured that as a mug settling 46.3 mm
    # into a ground plane with nothing raised.
    for body in ("visual", "collision"):
        assert [float(v) for v in link.find(f"{body}/origin").get("xyz").split()] \
            == pytest.approx([0.0, 0.0, 0.0])


def test_urdf_references_both_of_an_imported_parts_meshes():
    """The collision OBJ travels; a solid's `.brep` still does not.

    The two cases are not symmetric: nothing downstream of a URDF can read a
    `.brep`, while a collision mesh is exactly what a URDF's `<collision>` slot
    is for.
    """
    from dycraft.render import urdf_export as U

    xml = U.object_ir_to_urdf(_imported_ir())
    assert "assets/collision" in xml
    assert "assets/meshes" in xml
    assert ".brep" not in xml


def test_import_recipe_is_not_mistaken_for_an_object_ir(tmp_path):
    """A converter bundle can be handed straight to `--catalog <dir>`."""
    from dycraft.agent.layers.scene_context import iter_catalog_ir_paths

    (tmp_path / "abc123.json").write_text("{}")
    (tmp_path / "import.json").write_text("{}")
    assert [p.name for p in iter_catalog_ir_paths(tmp_path)] == ["abc123.json"]


# ---------------------------------------------------------------------------
# Base-color textures
# ---------------------------------------------------------------------------

def test_a_material_without_a_texture_omits_the_key_entirely():
    """The absent key is a content-hash invariant, not a style choice.

    An ObjectIR's `id` is `blake2b` over `json.dumps(..., sort_keys=True)`, so
    writing `"texture": null` would change the canonical bytes of every material
    that has no texture, which is most of them. That would silently invalidate
    every cached verifier report, and un-pin the example runs the animation
    tests resolve by hash (they would *skip*, not fail).
    """
    from dycraft.sdk._material import Material, Texture, TextureMap

    plain = Material(color=(176, 176, 176)).to_ir()
    assert set(plain) == {"color", "metallic", "roughness", "alpha"}

    # An imported asset's texture: base color and nothing else, and NO
    # `mapping` — its UVs are authored into the staged OBJ, so there is no tile
    # size to record. That absence is what tells a renderer to sample the
    # mesh's own coordinates.
    def _tex():
        return Texture(map_items=(("base_color", TextureMap(
            image="assets/textures/a.png", digest="a" * 32)),))

    textured = Material(color=(150, 110, 70), texture=_tex())
    assert textured.to_ir()["texture"] == {
        "base_color": {"image": "assets/textures/a.png", "digest": "a" * 32}}
    assert "mapping" not in textured.to_ir()
    # Still a frozen value object, so it stays reusable across parts.
    assert hash(textured) == hash(Material(color=(150, 110, 70), texture=_tex()))


def test_material_from_ir_degrades_instead_of_raising_on_a_bad_texture():
    """`import_gltf` merges report fields into the same `material` dict, and the
    renderer must never crash on a material it merely does not understand."""
    from dycraft.sdk._material import material_from_ir

    for bad in ({"base_color": None}, {"base_color": {}}, {}, "not-a-dict",
                None):
        mat = material_from_ir({"color": [1, 2, 3], "texture": bad},
                               default_color=(0, 0, 0))
        assert mat.texture is None
        assert mat.color == (1, 2, 3)      # the flat fallback still works


def test_per_corner_uvs_survive_a_weld_that_renumbers_vertices():
    """This is the property the whole texture path rests on.

    A welded corner of a box has ONE position and THREE different UVs, so a
    per-vertex array cannot represent it. OBJ's `f v/vt` carries two independent
    index streams: welding renumbers `v` and leaves `vt` alone.
    """
    import numpy as np
    from dycraft.agent.assets.import_urdf import weld, write_obj

    # Two triangles sharing an edge, with the shared vertices duplicated as an
    # exporter does at a UV seam — the duplicates carry different UVs.
    verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                      [1., 0, 0], [0, 1, 0], [1, 1, 0]])
    tris = np.array([[0, 1, 2], [3, 5, 4]])
    corner_uv = np.array([[[0., 0], [1, 0], [0, 1]],
                          [[0., 0], [1, 1], [0.5, 0.5]]])

    wv, wt = weld(verts, tris)
    assert len(wv) == 4 and len(wt) == 2        # duplicates really did merge
    obj = write_obj(wv, wt, "t", corner_uv).decode()

    # Every face indexes a vt, and the distinct UVs all survived.
    faces = [l for l in obj.splitlines() if l.startswith("f ")]
    assert len(faces) == 2 and all("/" in l for l in faces)
    assert len([l for l in obj.splitlines() if l.startswith("vt ")]) == 5


def test_write_obj_without_uvs_is_byte_identical_to_the_pre_texture_output():
    """A staged mesh's content digest must not move because the parameter exists
    — otherwise every catalog is invalidated by a no-op."""
    import numpy as np
    from dycraft.agent.assets.import_urdf import write_obj

    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    t = np.array([[0, 1, 2]])
    assert write_obj(v, t, "x") == (
        b"# dycraft imported part: x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")


def test_a_textured_part_stages_its_image_and_exports_it():
    """The image has to travel with the mesh, through both consumers.

    A catalog staged without it does not fail — it renders the flat mean color
    the material also carries, which reads as a design choice rather than a
    missing file.
    """
    from dycraft.render import urdf_export as U

    ir = _imported_ir()
    ir["parts"][0]["material"]["texture"] = {
        "base_color": {"image": "assets/textures/eeee.png",
                       "digest": "e" * 32}}

    assert Path("assets/textures/eeee.png") in referenced_object_assets(ir)

    xml = U.object_ir_to_urdf(ir)
    assert '<texture filename="assets/textures/eeee.png"' in xml
    # The untextured sibling must not have acquired one.
    assert xml.count("<texture") == 1
