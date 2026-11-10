"""Which lighting a still's frame carries, decided by the IR's schema.

`blender_renderer._lighting_frame_keys` sends a native scene's authored
`@lighting` block and nothing else; any other IR gets no lighting keys and is
lit by the child's automatic rig. These pin that branch without a render.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dycraft.sdk import Scene
from dycraft.sdk import scene_builder as _sb
from dycraft.sdk.ir import (SCENE_SCHEMA_VERSION, SCENE_SCHEMA_VERSION_V1,
                            SCENE_SCHEMA_VERSION_V3)
from dycraft.sdk.scene_builder import ObjectHandle

_BLOCK = {"id": "b" * 32, "name": "blk", "schema": "object/1", "root": "b",
          "joints": [],
          "parts": [{"name": "b", "primitive": "box", "id": 0,
                     "params": {"width": .4, "length": .4, "height": .8},
                     "material": {"color": [200, 200, 200]}, "anchors": {}}]}

_RIG = {"name": "rig", "body": "def rig(L):\n    pass\n"}


@pytest.fixture(autouse=True)
def _fresh():
    _sb._reset_scene_singleton()
    yield
    _sb._reset_scene_singleton()


def _scene_ir(schema: str) -> dict:
    """A one-instance scene IR with its schema tag set to `schema`."""
    s = Scene("lit")
    s.set_room(4.0, 3.0, 2.6)
    s.place(ObjectHandle(_BLOCK), name="blk", pose=((0.0, 0.0, 0.0),
                                                    (0.0, 0.0, 0.0)))
    ir = dict(s.finalize())
    ir["schema"] = schema
    return ir


class _Req:
    """The few `RenderRequest` fields `_lighting_frame_keys` reads."""

    def __init__(self, ir, catalog, out_dir=None, asset_base_dir=None):
        self.ir = ir
        self.catalog = catalog
        self.out_dir = out_dir
        self.asset_base_dir = asset_base_dir


def _frame(ir):
    from dycraft.render.blender_renderer import _lighting_frame_keys
    return _lighting_frame_keys(
        _Req(ir, {_BLOCK["id"]: _BLOCK}, out_dir=Path("/tmp/x/renders")))


def test_a_v2_scene_ships_its_lighting_block():
    ir = _scene_ir(SCENE_SCHEMA_VERSION)
    ir["lighting"] = dict(_RIG)
    out = _frame(ir)
    assert out["schema_v2"] is True
    assert out["lighting"]["name"] == "rig"


def test_a_v3_scene_is_native_too_and_keeps_its_authored_lighting():
    """The native branch is v2-or-later, not `== v2`.

    v3 is v2 plus per-instance `initial_joints`. An equality test would send
    every scene with a posed instance to the automatic rig and drop its
    authored `@lighting` block, while still producing a plausible render.
    """
    ir = _scene_ir(SCENE_SCHEMA_VERSION_V3)
    ir["lighting"] = dict(_RIG)
    out = _frame(ir)
    assert out["schema_v2"] is True
    assert out["lighting"]["name"] == "rig"


def test_a_v1_scene_ships_no_lighting_keys():
    """A v1 scene on disk renders under the automatic rig; its old symbolic
    lighting keys are ignored rather than half-applied."""
    ir = _scene_ir(SCENE_SCHEMA_VERSION_V1)
    ir["lights"] = [{"name": "lamp", "kind": "point"}]
    ir["environment"] = {"color": [235, 237, 242], "strength": 0.6}
    assert _frame(ir) == {}


def test_an_unknown_schema_takes_the_conservative_branch():
    """A tag this tree never emitted is not assumed to be the newest one."""
    ir = _scene_ir("dycraft.scene.v9")
    ir["lighting"] = dict(_RIG)
    assert not _frame(ir).get("schema_v2")
