"""`Material.emission` — a PART property, and the two backends' honest split.

The schema half is exhaustive here because it is cheap. The render half is two
`expensive` tests, because the only way to know whether an emissive surface
*illuminates* is to render something next to it and look at the pixels.
"""

from __future__ import annotations

import pytest

from dycraft.sdk._material import (
    EMISSION, Emission, Material, Texture, TextureMap, material_from_ir,
)

_C = (200, 200, 200)
_MAP = TextureMap(image="assets/e.png", digest="0" * 32)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_absent_emission_is_omitted_from_the_ir():
    """The content-hash rule. An unconditional key would re-id every object
    that has ever been built."""
    assert "emission" not in Material(color=_C).to_ir()


def test_emission_round_trips():
    m = Material(color=_C, emission=Emission(color=(255, 240, 214), strength=12.0))
    assert material_from_ir(m.to_ir(), default_color=_C).emission == m.emission


def test_a_material_with_emission_is_still_hashable():
    """`Material` is frozen so one instance can be shared across parts."""
    m = Material(color=_C, emission=Emission(color=_C, strength=1.0))
    assert len({m, m}) == 1


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), "x", None])
def test_a_non_positive_strength_is_refused(bad):
    """Zero is refused on purpose: `emission=None` already means "not a light
    source", and a second spelling of it would put `if strength > 0` in every
    consumer — including two renderers that would each decide separately."""
    with pytest.raises(Exception):
        Emission(color=_C, strength=bad)


def test_an_emission_map_without_the_scalar_is_refused():
    """Same rule as `mapping` without `texture`: the map has nothing to scale
    and no fallback for a consumer that cannot sample it."""
    with pytest.raises(Exception, match="emission"):
        Material(color=_C, texture=Texture(map_items=((EMISSION, _MAP),)))


def test_an_emission_map_with_the_scalar_is_accepted():
    m = Material(color=_C, texture=Texture(map_items=((EMISSION, _MAP),)),
                 emission=Emission(color=_C, strength=4.0))
    assert EMISSION in m.texture.maps and m.to_ir()["emission"]["strength"] == 4.0


def test_a_malformed_emission_block_degrades_rather_than_raising():
    """This runs inside the renderers; a material nobody can parse must not
    take the picture down."""
    m = material_from_ir({"color": list(_C), "emission": {"color": "nope"}},
                         default_color=_C)
    assert m.emission is None and m.color == _C


def test_a_dropped_emission_scalar_takes_its_map_with_it():
    """Otherwise `Material.__post_init__` would reject what the IR round-trip
    produced — a parse that cannot rebuild its own output."""
    ir = {"color": list(_C),
          "emission": {"strength": "bad"},
          "texture": {EMISSION: {"image": "assets/e.png", "digest": "0" * 32}}}
    m = material_from_ir(ir, default_color=_C)
    assert m.emission is None
    assert m.texture is None or EMISSION not in m.texture.maps


def test_emission_reaches_the_renderer_side_dict():
    from dycraft.render._world import _material_to_render

    packed = _material_to_render(
        Material(color=_C, emission=Emission(color=(255, 0, 0), strength=9.0)))
    assert packed["emission"]["strength"] == 9.0
    assert packed["emission"]["color"] == (1.0, 0.0, 0.0)   # floats, like color


def test_no_emission_means_no_key_in_the_packed_dict():
    from dycraft.render._world import _material_to_render

    assert "emission" not in _material_to_render(Material(color=_C))


# ---------------------------------------------------------------------------
# Open3D: shows it, does not cast it
# ---------------------------------------------------------------------------

def test_open3d_draws_an_emissive_part_unlit():
    """NOT via `MaterialRecord.emissive_color`.

    That field exists in Open3D's Python API and is a trap: `defaultLit.filamat`
    declares baseColor / baseMetallic / baseRoughness / reflectance / clearCoat
    / anisotropy and no emissive uniform, so setting it succeeds, changes
    nothing, and reports nothing. Measured: zero pixels differed.
    """
    from dycraft.render.open3d_renderer import _pbr_material

    lit = _pbr_material({"color": (0.5, 0.5, 0.5), "metallic": 0.0,
                         "roughness": 0.8, "alpha": 1.0})
    emissive = _pbr_material({"color": (0.5, 0.5, 0.5), "metallic": 0.0,
                              "roughness": 0.8, "alpha": 1.0,
                              "emission": {"color": (1.0, 0.9, 0.7),
                                           "strength": 60.0}})
    assert lit.shader == "defaultLit"
    assert emissive.shader == "defaultUnlit"
    # brighter than the surface it replaced, and toward white at high strength
    assert min(emissive.base_color[:3]) > min(lit.base_color[:3])


def test_a_translucent_emissive_part_keeps_the_transparency_shader():
    from dycraft.render.open3d_renderer import _pbr_material

    m = _pbr_material({"color": (0.5, 0.5, 0.5), "metallic": 0.0,
                       "roughness": 0.8, "alpha": 0.4,
                       "emission": {"color": (1.0, 1.0, 1.0), "strength": 5.0}})
    assert m.shader == "defaultUnlitTransparency"
    assert m.base_color[3] == pytest.approx(0.4)


def test_strength_saturates_rather_than_wrapping():
    from dycraft.render.open3d_renderer import _pbr_material

    def bright(s):
        return _pbr_material({"color": (0.2, 0.2, 0.2), "metallic": 0.0,
                              "roughness": 0.8, "alpha": 1.0,
                              "emission": {"color": (0.4, 0.2, 0.1),
                                           "strength": s}}).base_color[:3]

    dim, hot, absurd = bright(1.0), bright(50.0), bright(10_000.0)
    assert max(dim) < max(hot)
    assert all(c <= 1.0 for c in absurd), "an over-driven emitter must clamp"
    assert hot == pytest.approx(absurd, abs=1e-6), "should already be saturated"


# ---------------------------------------------------------------------------
# The behavior that only a render can answer
# ---------------------------------------------------------------------------

def _emissive_pair_ir(emissive: bool) -> dict:
    lamp = {"name": "lamp", "primitive": "box",
            "params": {"width": 0.30, "length": 0.06, "height": 0.30},
            "material": {"color": [255, 244, 214], "roughness": 0.5,
                         "metallic": 0.0, "alpha": 1.0}}
    if emissive:
        lamp["material"]["emission"] = {"color": [255, 236, 190],
                                        "strength": 60.0}
    return {"id": f"emit_{emissive}", "name": "emit", "schema": "object/1",
            "root": "wall",
            "parts": [{"name": "wall", "primitive": "box",
                       "params": {"width": 1.4, "length": 0.05, "height": 1.0},
                       "material": {"color": [210, 210, 210], "roughness": 0.9,
                                    "metallic": 0.0, "alpha": 1.0}},
                      lamp],
            "joints": [{"id": 0, "name": "j", "type": "fixed",
                        "parent": "wall", "child": "lamp",
                        "parent_anchor": {"translation": [-0.35, 0.22, -0.28],
                                          "rotation": [1.0, 0.0, 0.0, 0.0]},
                        "child_anchor": {"translation": [0.0, 0.0, 0.0],
                                         "rotation": [1.0, 0.0, 0.0, 0.0]}}]}


def _changed_fraction(backend, tmp_path):
    """Fraction of the frame that changes when the lamp part becomes emissive."""
    import numpy as np
    from PIL import Image
    from dycraft.render import RenderRequest, render_request

    imgs = []
    for emissive in (False, True):
        r = render_request(RenderRequest(
            layer="object", ir=_emissive_pair_ir(emissive), catalog=None,
            target="all", out_dir=tmp_path / f"{backend}_{emissive}", step=0,
            camera_ids=("front",), backend=backend, cache={}))
        imgs.append(np.asarray(
            Image.open(r.render_paths["front"]).convert("RGB"), dtype=float))
    return float((np.abs(imgs[1] - imgs[0]).max(axis=2) > 1).mean())


@pytest.mark.expensive
def test_open3d_lights_only_the_emissive_part_itself(tmp_path):
    """A rasteriser gathers no light from geometry."""
    changed = _changed_fraction("open3d", tmp_path)
    assert changed > 0.001, "the emissive part did not even change its own pixels"
    assert changed < 0.05, (
        f"{changed:.1%} of the frame changed — Open3D appears to be "
        "illuminating the scene, which it cannot do")


@pytest.mark.expensive
def test_cycles_lets_an_emissive_part_illuminate_its_surroundings(tmp_path):
    """The capability the Blender backend exists for: a plain wall beside the
    lamp brightens across nearly all of it when the lamp emits."""
    from dycraft.render import _blender_env

    if not _blender_env.is_available():
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")
    changed = _changed_fraction("blender", tmp_path)
    assert changed > 0.10, (
        f"only {changed:.1%} of the frame changed — the emissive part is not "
        "acting as a light source")
