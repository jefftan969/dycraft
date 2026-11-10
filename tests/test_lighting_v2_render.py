"""The v2 lighting contract, against real Blender.

Three questions that only a render can answer:

  * does the authored `@lighting` block actually light the frame;
  * does a v2 scene with **no** block render unlit — proving the automatic rig,
    the renderer's default World and the gray studio dome are all genuinely
    gone rather than merely bypassed;
  * does the color management the block sets come back in `view_settings`, so
    the visual-review record hashes the image that exists.

Marked `expensive`: each case spawns bpy.
"""

from __future__ import annotations

import pytest

from dycraft.sdk import Scene, lighting
from dycraft.sdk import scene_builder as _sb
from dycraft.sdk import _mesh_assets
from dycraft.sdk._mesh_assets import MeshAssetSession
from dycraft.sdk.scene_builder import ObjectHandle

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_BLOCK = {"id": "b" * 32, "name": "blk", "schema": "object/1", "root": "b",
          "joints": [],
          "parts": [{"name": "b", "primitive": "box", "id": 0,
                     "params": {"width": .6, "length": .6, "height": .6},
                     "material": {"color": [200, 180, 150]}, "anchors": {}}]}

LIT = '''
@lighting
def rig(L):
    key = L.new_light("key_area", "AREA")
    key.data.energy = 300.0
    key.data.size = 2.0
    key.location = (1.2, -1.2, 2.2)
    L.aim(key, (0.0, 0.0, 0.5))
    L.view.view_transform = "AgX"
    L.view.look = "AgX - Medium High Contrast"
    L.view.exposure = -0.6
    L.view.use_white_balance = True
    L.view.white_balance_temperature = 5200
'''

NO_LIGHTS = '''
@lighting
def rig(L):
    L.view.view_transform = "AgX"
'''

RAISES = '''
@lighting
def rig(L):
    L.new_light("bad", "NOT_A_LIGHT_TYPE")
'''


def _block_from(src: str, tmp_path):
    session = MeshAssetSession(run_dir=tmp_path)
    session.source_text = src
    _mesh_assets.set_active_mesh_session(session)
    try:
        ns = {"lighting": lighting}
        exec(compile(src, "<scene.py>", "exec"), ns)     # noqa: S102
        return ns["rig"]
    finally:
        _mesh_assets.clear_active_mesh_session()


def _scene_ir(src: str, tmp_path):
    block = _block_from(src, tmp_path)
    _sb._reset_scene_singleton()
    s = Scene("v2")
    s.set_room(4.0, 4.0, 2.7)
    s.place(ObjectHandle(_BLOCK), name="blk", pose=((0, 0, 0), (0, 0, 0)))
    s.set_lighting(block)
    return s.finalize()


def _render(ir, tmp_path, sub="out"):
    from dycraft.render import RenderRequest, render_request
    return render_request(RenderRequest(
        layer="scene", ir=ir, catalog={_BLOCK["id"]: _BLOCK}, target="front",
        out_dir=tmp_path / sub, step=1, cache={}, backend="blender"))


def _mean(path) -> float:
    from PIL import Image
    import statistics
    with Image.open(path) as im:
        g = im.convert("L")
        px = list(g.getdata())
    return statistics.fmean(px)


def _require_blender():
    from dycraft.render import _blender_env as E
    if not E.is_available():
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")


@pytest.mark.expensive
def test_authored_lighting_lights_the_frame(tmp_path):
    _require_blender()
    res = _render(_scene_ir(LIT, tmp_path), tmp_path, "lit")
    path = next(iter(res.render_paths.values()))
    assert _mean(path) > 40.0, "the authored area light must light the room"
    assert not res.warnings


@pytest.mark.expensive
def test_a_v2_scene_with_no_lights_renders_unlit(tmp_path):
    """The decisive check that nothing is added behind the author's back.

    If the automatic interior rig, the studio rig, `_setup_world`'s gray dome
    or a renderer-added sun survived for v2, this frame could not be black.
    """
    _require_blender()
    res = _render(_scene_ir(NO_LIGHTS, tmp_path), tmp_path, "dark")
    path = next(iter(res.render_paths.values()))
    assert _mean(path) < 5.0, (
        "a v2 scene that authored no light must render unlit; a non-black "
        "frame means the renderer supplied lighting nobody asked for")


@pytest.mark.expensive
def test_view_settings_come_back_from_the_block(tmp_path):
    _require_blender()
    res = _render(_scene_ir(LIT, tmp_path), tmp_path, "vs")
    vs = res.view_settings
    assert vs["view_transform"] == "AgX"
    assert vs["look"] == "AgX - Medium High Contrast"
    assert round(vs["exposure"], 3) == -0.6
    assert vs["use_white_balance"] is True
    assert round(vs["white_balance_temperature"]) == 5200


@pytest.mark.expensive
def test_a_failing_block_is_a_diagnostic_and_no_image(tmp_path):
    """No fallback rig. A misleading render of lighting the author never wrote
    is worse than a failed one."""
    _require_blender()
    from dycraft.render import RenderUnavailable
    ir = _scene_ir(RAISES, tmp_path)
    with pytest.raises(RenderUnavailable) as e:
        _render(ir, tmp_path, "boom")
    msg = str(e.value)
    assert "lighting_failed" in msg
    assert "rig" in msg and "NOT_A_LIGHT_TYPE" in msg
    assert not list((tmp_path / "boom").glob("*.png"))


BAD_SHADER_SCENE_NOTE = """A shader that names a socket Blender does not have."""


@pytest.mark.expensive
def test_strict_materials_refuses_to_fall_back(tmp_path):
    """A failing `@shader` must abort a review/delivery render, not quietly
    draw its scalars — those images are what appearance is judged from."""
    _require_blender()
    from dycraft.render import (RenderRequest, RenderUnavailable,
                                render_request)
    from dycraft.sdk import Material
    from dycraft.sdk._material import Shader

    bad = Shader(name="broken",
                 body=("def broken(nodes, links, bsdf):\n"
                       "    bsdf.inputs['NoSuchSocket'].default_value = 1.0\n"))
    block = dict(_BLOCK)
    block["id"] = "c" * 32
    block["parts"] = [{**_BLOCK["parts"][0],
                       "material": Material(shader=bad).to_ir()}]

    _sb._reset_scene_singleton()
    s = Scene("strict")
    s.set_room(4.0, 4.0, 2.7)
    s.place(ObjectHandle(block), name="blk", pose=((0, 0, 0), (0, 0, 0)))
    s.set_lighting(_block_from(LIT, tmp_path))
    ir = s.finalize()

    def _go(strict):
        return render_request(RenderRequest(
            layer="scene", ir=ir, catalog={block["id"]: block},
            target="front", out_dir=tmp_path / f"s{int(strict)}", step=1,
            cache={}, backend="blender", strict_materials=strict))

    # A v2 scene never falls back, so BOTH forms must fail — `strict_materials`
    # is the per-caller request and v2 makes it unconditional. A scalar stand-in
    # is not a degraded version of an authored graph, it is a different surface.
    for strict in (False, True):
        with pytest.raises(RenderUnavailable) as e:
            _go(strict)
        assert "shader_failed" in str(e.value), strict
        assert "broken" in str(e.value), strict
        assert not list((tmp_path / f"s{int(strict)}").glob("*.png")), strict


# --- the ordering contract, checked structurally ---------------------------

def _child_src() -> str:
    from pathlib import Path
    from dycraft.render import blender_renderer as B
    return Path(B._CHILD).read_text()


def test_lighting_is_installed_after_geometry_in_both_callers():
    """The execution order is the contract: geometry and cameras, then
    materials, then `@lighting`, then render."""
    src = _child_src()
    for fn in ("_render_frame", "_serve_build"):
        start = src.index(f"def {fn}(")
        end = src.index("\ndef ", start + 1)
        body = src[start:end]
        assert "_install_frame_lighting" in body, fn
        assert body.index("_build_part(") < body.index(
            "_install_frame_lighting"), (
            f"{fn} installs lighting before it builds geometry")


def test_nothing_writes_lighting_or_colour_management_after_the_block():
    """`_apply_lighting` must be the last thing that touches lights, the World
    or view settings. Checked structurally because there is no runtime hook
    that would notice a later write."""
    src = _child_src()
    start = src.index("def _render_frame(")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    after = body[body.index("_install_frame_lighting"):]
    for forbidden in ("_setup_world(", "_setup_rig(", "view_settings."):
        assert forbidden not in after, (
            f"{forbidden} appears after the lighting block in _render_frame")


def test_the_automatic_rig_is_unreachable_from_a_native_scene():
    """A native scene is lit by its `@lighting` block or not at all; the
    automatic rig is only the fallback after both native branches return."""
    src = _child_src()
    callers = []
    for fn in ("_install_frame_lighting", "_render_frame", "_serve_build"):
        start = src.index(f"def {fn}(")
        end = src.index("\ndef ", start + 1)
        if "_setup_rig(" in src[start:end]:
            callers.append(fn)
    assert callers == ["_install_frame_lighting"], callers
    start = src.index("def _install_frame_lighting(")
    body = src[start:src.index("\ndef ", start + 1)]
    assert (body.index('frame.get("lighting")') < body.index('frame.get("schema_v2")')
            < body.index("_setup_rig(")), "the rig is reachable before a native branch returns"
