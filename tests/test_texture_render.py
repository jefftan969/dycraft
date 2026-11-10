"""What binding a texture does, and does not do, to the authoritative render.

Everything that draws needs a GL context and is skipped where Open3D cannot
open an OffscreenRenderer. The claim those pin is the one that let this change
land WITHOUT bumping `RENDER_CONFIG_VERSION`: adding UVs is invisible to a part
that has no texture, so no committed render, render-cache entry or stored
judgment went stale.

The last section is the exception and is deliberately ungated: it pins the UV
projection itself, which is arithmetic over numpy in `render/_world.py` and
reaches neither open3d nor a display.
"""

import base64
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dycraft.render import RenderUnavailable  # noqa: F401

# Two PNGs, inline and written to a temp dir at import.
#
# **Inline rather than a committed asset, and that is the point of them.** These
# tests are about what the RENDERER does with an image, so the image should be
# the smallest thing that decodes — not a real material. A real texture would
# couple every renderer test to a texture corpus, and a producer must not be
# able to break these by changing what it produces.
#
# Base color is an 8x8 two-tone checker, so a bound texture is visibly not a
# flat fill. Roughness is 8x8 greyscale at a uniform 128, i.e. 0.5 — the tests
# below only need it to DECODE, since the pixel-level roughness claim is made
# against synthetic in-memory extremes further down.
_BASE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAH0lEQVR42mOYluEE"
    "RHF2akCEzGbAKYEpBGHjlqCDHQAsv0BBcWOG6wAAAABJRU5ErkJggg==")
_ROUGH_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAAAAADhZOFXAAAADklEQVR42mNogAIG"
    "yhgAgIQgARDoahcAAAAASUVORK5CYII=")

_FIXTURES = Path(tempfile.mkdtemp(prefix="dycraft-texture-test-"))
LIB = _FIXTURES / "base_color.png"
ROUGH_LIB = _FIXTURES / "roughness.png"
LIB.write_bytes(_BASE_PNG)
ROUGH_LIB.write_bytes(_ROUGH_PNG)


def _renderer(w=320, h=240):
    from dycraft.render.open3d_renderer import (
        _select_egl_platform, _LIGHT_RIG, _normalize)
    _select_egl_platform()
    import open3d as o3d
    r = o3d.visualization.rendering.OffscreenRenderer(w, h)
    r.scene.set_background(np.asarray([0.95] * 3 + [1.0], dtype=np.float32))
    for name, pos, inten in _LIGHT_RIG:
        r.scene.scene.add_directional_light(
            name, np.array([1.0, 1.0, 1.0], np.float32),
            np.array(_normalize((-pos[0], -pos[1], -pos[2])), np.float32),
            inten, False)
    return r


try:
    _renderer(16, 16)
    _HAVE_GL = True
except BaseException:
    _HAVE_GL = False

_needs_gl = pytest.mark.skipif(not _HAVE_GL, reason="no Open3D GL context")


def _shoot(material):
    """Render one unit box through the real `_part_mesh` / `_pbr_material`."""
    from dycraft.render.open3d_renderer import (
        _part_mesh, _pbr_material, _WorldPart)
    part = _WorldPart(name="p", size=(1.0, 0.6, 0.4),
                      se3=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                      material=material, primitive="box")
    mesh, _problem = _part_mesh(part, material["color"],
                                material["alpha"])
    r = _renderer()
    r.scene.add_geometry("p", mesh, _pbr_material(material))
    r.setup_camera(60.0, [0.0, 0.0, 0.0], [1.6, -1.5, 1.1], [0, 0, 1])
    arr = np.asarray(r.render_to_image())
    del r
    return arr


_GREY = {"color": (176 / 255.0, 176 / 255.0, 176 / 255.0),
         "metallic": 0.0, "roughness": 0.9, "alpha": 1.0}
_WOOD = dict(_GREY,
             texture={"base_color": {"path": str(LIB), "digest": "wood-test"}},
             mapping={"scale_m": 0.25})


@_needs_gl
def test_a_part_with_no_texture_reaches_filament_unchanged():
    """The guard on NOT bumping RENDER_CONFIG_VERSION.

    `_apply_texture_uvs` is a no-op without a texture, so an untextured part's
    geometry reaches Filament unchanged. If this ever fails, every committed render is stale and the
    config version has to move with it.
    """
    a = hashlib.blake2b(_shoot(_GREY).tobytes(), digest_size=16).hexdigest()
    b = hashlib.blake2b(_shoot(_GREY).tobytes(), digest_size=16).hexdigest()
    assert a == b


@_needs_gl
def test_adding_uvs_to_an_untextured_mesh_changes_nothing():
    """The same claim one level down, where it is actually decided."""
    import open3d as o3d
    from dycraft.render.open3d_renderer import _apply_texture_uvs, _triplanar_uvs

    m = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    _apply_texture_uvs(m, None, 0.5)
    assert len(np.asarray(m.triangle_uvs)) == 0

    _apply_texture_uvs(m, _WOOD["texture"], 0.5)
    assert len(np.asarray(m.triangle_uvs)) == 3 * len(np.asarray(m.triangles))


@_needs_gl
def test_uvs_survive_the_world_transform_and_shading_calls():
    """`_box_mesh` sets UVs and then transforms, recomputes normals and paints.
    If any of those dropped them the texture would silently vanish."""
    import open3d as o3d
    from dycraft.render.open3d_renderer import _box_mesh

    m = _box_mesh((1.0, 1.0, 1.0), ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)),
                  (0.5, 0.5, 0.5), texture=_WOOD["texture"], scale_m=0.5)
    assert len(np.asarray(m.triangle_uvs)) == 3 * len(np.asarray(m.triangles))


@_needs_gl
def test_a_bound_texture_actually_reaches_the_pixels():
    """Wood is chromatic and mid-gray is not, so a channel spread separates
    them without pinning any particular pixel."""
    grey, wood = _shoot(_GREY), _shoot(_WOOD)
    assert grey.shape == wood.shape
    fg = np.abs(wood.astype(int) - grey.astype(int)).sum(axis=2) > 12
    assert fg.mean() > 0.05, "the texture changed almost nothing"
    lit = wood[fg].astype(float)
    assert (lit[:, 0].mean() - lit[:, 2].mean()) > 15, (
        "a wood albedo should read warmer than it reads blue")


@_needs_gl
def test_the_authored_colour_does_not_tint_a_textured_surface():
    """The prompt tells the agent `color` is a fallback that does not change
    the textured render. That is true only because `_pbr_material` binds white
    alongside the albedo — Filament MULTIPLIES its sample by base_color, and
    measured, the same wood over a blue base renders gray (lit mean RGB
    (28.7, 29.2, 30.7) against (73.2, 46.4, 29.4) over the library's own
    brown). Without the white this test fails and the taught contract is a
    lie, so it is pinned here rather than left to the docstring.
    """
    warm = _shoot(_WOOD)
    for other in ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.1, 0.1, 0.1)):
        assert np.array_equal(warm, _shoot(dict(_WOOD, color=other))), (
            f"base_color {other} changed a textured render")


@_needs_gl
def test_a_translucent_textured_surface_still_selects_the_transparency_shader():
    """The white swap must keep alpha, or binding a texture would silently
    make a glass panel opaque."""
    from dycraft.render.open3d_renderer import _pbr_material

    mat = _pbr_material(dict(_WOOD, alpha=0.4))
    assert mat.shader == "defaultLitTransparency"
    assert mat.base_color[3] == pytest.approx(0.4)
    assert mat.base_color[:3] == pytest.approx([1.0, 1.0, 1.0])


@_needs_gl
def test_an_unreadable_texture_falls_back_to_the_flat_colour():
    """Fail-soft, like a missing OBJ: a broken texture must not take the
    render down, because the render is how every verdict is formed."""
    from dycraft.render.open3d_renderer import (_load_texture, _TEXTURE_CACHE,
                                                _FAILED_TEXTURES)

    _TEXTURE_CACHE.pop(("missing-test", "RGB"), None)
    _FAILED_TEXTURES.discard(("/nonexistent/nope.png", "missing-test", "RGB"))
    assert _load_texture("/nonexistent/nope.png", "missing-test") is None
    assert ("/nonexistent/nope.png", "missing-test", "RGB") in _FAILED_TEXTURES, \
        "a miss should be cached too — by PATH, since that is what failed"
    assert ("missing-test", "RGB") not in _TEXTURE_CACHE, \
        "a miss must not land in the digest-keyed hit cache"
    broken = dict(_GREY,
                  texture={"base_color": {"path": "/nonexistent/nope.png",
                                          "digest": "missing-test"}},
                  mapping={"scale_m": 0.25})
    assert np.array_equal(_shoot(broken), _shoot(_GREY))


@_needs_gl
def test_the_decoded_texture_is_cached_by_digest():
    from dycraft.render.open3d_renderer import _load_texture, _TEXTURE_CACHE

    _TEXTURE_CACHE.pop(("cache-test", "RGB"), None)
    first = _load_texture(str(LIB), "cache-test")
    assert first is not None
    assert _load_texture("/a/different/path.png", "cache-test") is first, (
        "the cache keys on digest, not path — a scene render resolves the "
        "same staged image under a different catalog root")


@_needs_gl
def test_a_failed_path_does_not_poison_the_same_digest_elsewhere():
    """A decode failure is a property of the PATH, not of the content.

    The same staged image resolves under a different root in a scene render
    than in an object one. Caching the miss by digest alone let one bad root
    make every later, correct lookup of that image return None for the rest of
    the process — drawing it untextured with nothing reporting why.
    """
    from dycraft.render.open3d_renderer import (_load_texture, _TEXTURE_CACHE,
                                                _FAILED_TEXTURES)

    _TEXTURE_CACHE.pop(("poison-test", "RGB"), None)
    _FAILED_TEXTURES.discard(("/wrong/root/wood.png", "poison-test", "RGB"))

    assert _load_texture("/wrong/root/wood.png", "poison-test") is None
    good = _load_texture(str(LIB), "poison-test")
    assert good is not None, (
        "the correct path must still decode after a bad root failed on the "
        "same digest")


# ---------------------------------------------------------------------------
# Roughness maps (v2)
# ---------------------------------------------------------------------------

_WOOD_ROUGH = dict(_WOOD, texture={**_WOOD["texture"],
                                   "roughness": {"path": str(ROUGH_LIB),
                                                 "digest": "rough-test"}})


@_needs_gl
def test_a_roughness_map_neutralises_the_scalar_it_would_be_multiplied_by():
    """Filament MULTIPLIES `roughness_img` by `base_roughness`, exactly as it
    multiplies the albedo by `base_color`.

    With a uniform 0.5 map, the rendered brightness still follows
    `base_roughness`, which a map that REPLACED the scalar could not do. So the
    renderer binds 1.0 and lets the map be authoritative, and the authored
    scalar stays the fallback every consumer without the map uses. If this
    regresses, a wooden surface is rendered with its own roughness applied
    twice and reads glossier than the viewer shows it.
    """
    from dycraft.render.open3d_renderer import _pbr_material

    mat = _pbr_material(_WOOD_ROUGH)
    assert mat.roughness_img is not None, "the roughness map was not bound"
    assert mat.base_roughness == pytest.approx(1.0)

    # Without a map the authored scalar is untouched.
    plain = _pbr_material(_WOOD)
    assert plain.base_roughness == pytest.approx(_WOOD["roughness"])


@_needs_gl
def test_an_unreadable_roughness_map_leaves_the_scalar_alone():
    """Fail-soft, per-map: the base color survives and the surface keeps the
    roughness it authored, rather than being neutralised to 1.0 with nothing
    bound to take over."""
    from dycraft.render.open3d_renderer import _pbr_material, _TEXTURE_CACHE

    _TEXTURE_CACHE.pop(("rough-missing", "L"), None)
    broken = dict(_WOOD, texture={**_WOOD["texture"],
                                  "roughness": {"path": "/nonexistent/r.png",
                                                "digest": "rough-missing"}})
    mat = _pbr_material(broken)
    assert mat.albedo_img is not None, "the base colour should still bind"
    assert getattr(mat, "roughness_img", None) is None
    assert mat.base_roughness == pytest.approx(_WOOD["roughness"])


@_needs_gl
def test_the_roughness_map_reaches_the_pixels():
    """An extreme synthetic map, because a plausible roughness map's own
    variation is small and would not separate from render noise."""
    import open3d as o3d

    glossy = np.zeros((64, 64), np.uint8)          # 0.0 everywhere
    matte = np.full((64, 64), 255, np.uint8)       # 1.0 everywhere
    seen = []
    for arr in (glossy, matte):
        from dycraft.render.open3d_renderer import _TEXTURE_CACHE
        digest = f"synth-{int(arr[0, 0])}"
        # Primed under the mode the roughness binding asks for: the cache is
        # keyed on `(digest, mode)` so a greyscale map and a color one sharing
        # content cannot hand each other the wrong channel count.
        _TEXTURE_CACHE[(digest, "L")] = o3d.geometry.Image(
            np.ascontiguousarray(arr))
        spec = dict(_GREY, mapping={"scale_m": 0.25},
                    texture={**_WOOD["texture"],
                             "roughness": {"path": "unused",
                                           "digest": digest}})
        seen.append(_shoot(spec).astype(float))
    a, b = seen
    assert not np.array_equal(a, b), (
        "a fully glossy and a fully matte roughness map rendered identically")
    assert abs(a.mean() - b.mean()) > 0.5, (
        f"means {a.mean():.2f} vs {b.mean():.2f} — the map is bound but inert")


# ---------------------------------------------------------------------------
# Staged UVs win; projection is the fallback
#
# A staged OBJ carries its own per-corner coordinates — baked from the B-rep for
# a CAD solid, authored into the asset for an imported one. The renderer's job
# is to consume them, not to re-derive a worse parameterization over the top.
# Projection survives only where there is no staged asset to have baked into.
# ---------------------------------------------------------------------------

# A unit quad whose two triangles carry deliberately NON-triplanar UVs: a
# projection of a +Z-facing quad would take (x, y), so a v that runs backwards
# against y is something only the file can be the source of.
_QUAD_OBJ = """o quad
v 0.000000 0.000000 0.000000
v 1.000000 0.000000 0.000000
v 1.000000 1.000000 0.000000
v 0.000000 1.000000 0.000000
vt 0.000000 1.000000
vt 1.000000 1.000000
vt 1.000000 0.000000
vt 0.000000 0.000000
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


@_needs_gl
def test_a_mesh_that_arrived_with_uvs_keeps_them(tmp_path):
    """The rule, at the one function that decides it.

    Overwriting authored UVs is silent: a re-projected surface still looks
    plausibly textured, and you only notice by knowing what it should have
    looked like. So this asserts the coordinates are the FILE's, not merely that
    some coordinates are present.
    """
    import open3d as o3d
    from dycraft.render.open3d_renderer import _apply_texture_uvs, _triplanar_uvs

    path = tmp_path / "quad.obj"
    path.write_text(_QUAD_OBJ)
    mesh = o3d.io.read_triangle_mesh(str(path))
    assert mesh.has_triangle_uvs(), "fixture must carry vt for this to mean anything"
    before = np.asarray(mesh.triangle_uvs).copy()

    _apply_texture_uvs(mesh, _WOOD["texture"], 0.25)
    after = np.asarray(mesh.triangle_uvs)
    assert np.array_equal(before, after), "the file's UVs were overwritten"

    # And the projection genuinely would have differed, so the test is not
    # passing because the two happen to agree.
    projected = _triplanar_uvs(np.asarray(mesh.vertices),
                               np.asarray(mesh.triangles), 0.25)
    assert not np.allclose(before, projected)


@_needs_gl
def test_a_staged_solid_reaches_filament_with_the_uvs_from_its_obj(tmp_path):
    """End to end through `_part_mesh`, the path a CAD part actually takes.

    `_part_mesh` loads the staged OBJ and then applies the part's world SE3 and
    recomputes normals; this pins that the coordinates that survive all of it
    are the ones the staging step baked.
    """
    from dycraft.render.open3d_renderer import _part_mesh, _WorldPart

    path = tmp_path / "quad.obj"
    path.write_text(_QUAD_OBJ)
    part = _WorldPart(name="p", size=(1.0, 1.0, 0.01),
                      se3=((2.0, -1.0, 0.5), (1.0, 0.0, 0.0, 0.0)),
                      material=_WOOD, primitive="solid",
                      params={"mesh": str(path)})
    mesh, _problem = _part_mesh(part, _WOOD["color"], 1.0)
    uvs = np.asarray(mesh.triangle_uvs)
    assert np.array_equal(uvs, np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
                                         [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))


@_needs_gl
def test_runtime_geometry_with_no_staged_asset_still_gets_projected_uvs():
    """The other half of the same rule, and the reason projection stays.

    A room's floor and walls are synthesized per camera from the room bounds and
    an analytic box part has no mesh file at all — neither has anything to have
    baked into, so `create_box` emits no UVs and the projection is the only
    thing that can parameterize them.
    """
    import open3d as o3d
    from dycraft.render.open3d_renderer import _box_mesh

    raw = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    assert not raw.has_triangle_uvs(), "create_box must not supply UVs"

    m = _box_mesh((2.0, 1.0, 0.5), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                  (0.5, 0.5, 0.5), texture=_WOOD["texture"], scale_m=0.25)
    assert len(np.asarray(m.triangle_uvs)) == 3 * len(np.asarray(m.triangles))


# ---------------------------------------------------------------------------
# The projection rule the browser reimplements
#
# Ungated: `_world._triplanar_uvs` is numpy over vertices and triangles, so it
# needs no renderer and no display. Imported from `_world` rather than through
# `open3d_renderer`'s re-export for the same reason — the pin must not become
# skippable on a host with no GL, because a skipped golden is a golden nobody
# checks.
#
# **Do not drop these two.** Without them `triplanar_uv_golden.json` is pinned
# from the TypeScript side alone, which would let the renderer drift away from
# both the fixture and the browser with nothing failing, the one outcome the
# fixture exists to prevent.
# ---------------------------------------------------------------------------

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "triplanar_uv_golden.json"


def test_the_triplanar_golden_is_what_the_renderer_produces():
    """Pins `tests/fixtures/triplanar_uv_golden.json` to `_triplanar_uvs`.

    The fixture is the contract `viewer/web/scripts/check-triplanar-uv.mjs`
    holds the browser to. Pinning it here rather than letting the two
    implementations be compared to each other is what makes the renderer the
    authority: the viewer is a preview of the frames the renderer draws, so a
    drift has to fail on the side that is wrong.

    Regenerate with `python tests/fixtures/_make_triplanar_golden.py`.
    """
    from dycraft.render._world import _triplanar_uvs

    golden = json.loads(_GOLDEN.read_text())
    assert golden["cases"], "the golden fixture is empty"
    for case in golden["cases"]:
        verts = np.asarray(case["vertices"], float).reshape(-1, 3)
        tris = np.asarray(case["triangles"], int).reshape(-1, 3)
        got = _triplanar_uvs(verts, tris, case["scale_m"]).reshape(-1)
        want = np.asarray(case["uvs"], float)
        assert got.shape == want.shape, case["name"]
        assert np.abs(got - want).max() <= golden["tolerance"], case["name"]


def test_a_reversed_winding_does_not_move_the_texture():
    """`argmax(|n|)` is sign-blind, so a face and its back project alike.

    Load-bearing because the browser reimplements this rule; the golden carries
    both windings for the same reason.
    """
    from dycraft.render._world import _triplanar_uvs

    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    front = _triplanar_uvs(v, np.array([[0, 1, 2]]), 0.5)
    back = _triplanar_uvs(v, np.array([[0, 2, 1]]), 0.5)
    assert np.allclose(sorted(map(tuple, front)), sorted(map(tuple, back)))
