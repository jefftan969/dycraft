"""`@lighting`: source capture, IR round-trip, and the removed symbolic API.

The capture half mirrors `@shader`'s, so what is worth pinning here is the
boundary: which signature is accepted, which free names are allowed (and that
the allowed set matches what the render child injects), and that the v1
symbolic lighting surface is genuinely gone from the agent namespace.
"""

from __future__ import annotations

import pytest

from dycraft.sdk import Scene
from dycraft.sdk import scene_builder as _sb
from dycraft.sdk._errors import InvalidPrimitive, InvalidScene
from dycraft.sdk._lighting_block import EXTRA_ALLOWED_FREE, LightingBlock
from dycraft.sdk._mesh_assets import MeshAssetSession


@pytest.fixture(autouse=True)
def _fresh():
    _sb._reset_scene_singleton()
    yield
    _sb._reset_scene_singleton()


def _capture(src: str, name: str = "rig", tmp_path=None):
    """Run `src` as if it were the authored `scene.py` and return the block.

    **Validates through the sandbox first**, exactly as a real compile does.
    Skipping that step is what let `@lighting` ship unable to compile at all:
    every test here built its block by calling the decorator directly, so the
    sandbox's attribute-assignment rule — which no lighting body can satisfy
    without an exemption — was never exercised on this path.
    """
    from dycraft.agent.cad.sandbox import check_ast
    check_ast(src)
    from dycraft.sdk import _mesh_assets
    session = MeshAssetSession(run_dir=tmp_path)
    session.source_text = src
    _mesh_assets.set_active_mesh_session(session)
    try:
        ns: dict = {}
        from dycraft.sdk._lighting_block import lighting
        ns["lighting"] = lighting
        exec(compile(src, "<scene.py>", "exec"), ns)     # noqa: S102
        return ns[name]
    finally:
        _mesh_assets.clear_active_mesh_session()


GOOD = '''
@lighting
def rig(L):
    key = L.new_light("key", "AREA")
    key.data.energy = 40.0
    key.location = (1.0, -0.5, 2.2)
    key.rotation_euler = (radians(-35.0), 0.0, radians(20.0))
    bg = L.world_nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.6
    L.view.view_transform = "AgX"
    L.view.exposure = -0.7
'''


def test_captures_body_and_name(tmp_path):
    block = _capture(GOOD, tmp_path=tmp_path)
    assert isinstance(block, LightingBlock)
    assert block.name == "rig"
    assert block.body.startswith("def rig(L):")
    assert "ShaderNodeBackground" in block.body
    # the decorator line is excluded, so the body is a self-contained def
    assert "@lighting" not in block.body


def test_wrong_signature_is_rejected(tmp_path):
    src = '''
@lighting
def rig(nodes, links):
    pass
'''
    with pytest.raises(InvalidPrimitive) as e:
        _capture(src, tmp_path=tmp_path)
    assert "(L)" in str(e.value)


def test_free_module_constant_is_rejected(tmp_path):
    """A body travels alone; a module constant will not be there."""
    src = '''
ENERGY = 40.0

@lighting
def rig(L):
    L.new_light("key", "AREA").data.energy = ENERGY
'''
    with pytest.raises(InvalidPrimitive) as e:
        _capture(src, tmp_path=tmp_path)
    assert "ENERGY" in str(e.value)


def test_math_builtins_are_allowed(tmp_path):
    src = '''
@lighting
def rig(L):
    L.new_light("k", "SUN").rotation_euler = (radians(34.0), 0.0, pi / 4.0)
'''
    assert _capture(src, tmp_path=tmp_path).name == "rig"


def test_unknown_node_class_is_rejected_at_author_time(tmp_path):
    src = '''
@lighting
def rig(L):
    L.world_nodes.new("ShaderNodeTexMusgrave")
'''
    with pytest.raises(InvalidPrimitive):
        _capture(src, tmp_path=tmp_path)


def test_allowed_free_names_match_the_sandbox_math_set():
    """The render child hardcodes the same names — it has no dycraft import —
    so a drift here is a body that passes author-time checks and then fails
    inside Blender."""
    from dycraft.agent.cad.sandbox import _MATH_BUILTINS
    sandbox = set(_MATH_BUILTINS) - {"smoothstep", "pulse"}
    assert set(EXTRA_ALLOWED_FREE) == sandbox


# --- the removed symbolic surface -----------------------------------------


def test_set_lighting_rejects_a_plain_function():
    s = Scene("x")
    with pytest.raises(InvalidScene):
        s.set_lighting(lambda L: None)


_BLOCK_IR = {"id": "o" * 32, "name": "blk", "schema": "object/1",
             "root": "b", "joints": [],
             "parts": [{"name": "b", "primitive": "box", "id": 0,
                        "params": {"width": .4, "length": .4, "height": .8},
                        "material": {"color": [200, 200, 200]},
                        "anchors": {}}]}


def _minimal_scene(name: str):
    from dycraft.sdk.scene_builder import ObjectHandle
    s = Scene(name)
    s.set_room(3.0, 3.0, 2.6)
    s.place(ObjectHandle(_BLOCK_IR), name="blk",
            pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    return s


def test_v2_schema_and_lighting_in_ir(tmp_path):
    block = _capture(GOOD, tmp_path=tmp_path)
    s = _minimal_scene("lit")
    s.set_lighting(block)
    ir = s.finalize()
    assert ir["schema"] == "dycraft.scene.v2"
    assert ir["lighting"]["name"] == "rig"
    for v1key in ("lights", "sky", "environment", "auto_lighting"):
        assert v1key not in ir, f"a v2 scene must not carry {v1key}"


def test_lighting_enters_the_content_hash(tmp_path):
    block = _capture(GOOD, tmp_path=tmp_path)
    id_unlit = _minimal_scene("a").finalize()["id"]
    _sb._reset_scene_singleton()
    s2 = _minimal_scene("a")
    s2.set_lighting(block)
    id_lit = s2.finalize()["id"]
    assert id_lit != id_unlit, "the lighting source must change the scene id"


# --- agent-facing guidance -------------------------------------------------

def _agent_facing_docs():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "agent" / "prompts").rglob("*.md"))
    paths += list((root / ".claude" / "skills").rglob("*.md"))
    paths += list((root / ".agents" / "skills").rglob("*.md"))
    return [(p, p.read_text()) for p in paths]


@pytest.mark.parametrize("anchor", [
    "400–1600 lm", "400-1600 lm",        # bulb lumen range
    "10 000 lux", "10000 lux",           # overcast daylight
    "100 000",                            # direct sun
])
def test_no_numeric_lighting_anchors_remain(anchor):
    """These are the values two independent agents converged on `sun_lux=17000`
    from. An agent has no way to know what they mean against this renderer's
    exposure, so the guidance anchored a guess instead of informing one."""
    hits = [str(p) for p, text in _agent_facing_docs() if anchor in text]
    assert not hits, f"{anchor!r} is still anchoring lighting choices in {hits}"


@pytest.mark.parametrize("gone", ["scene.add_light(", "scene.set_sky(",
                                  "scene.set_environment(",
                                  "scene.set_auto_lighting("])
def test_no_doc_tells_an_agent_to_call_the_removed_api(gone):
    hits = [str(p) for p, text in _agent_facing_docs() if gone in text]
    assert not hits, f"{gone} is still documented as callable in {hits}"


def test_the_docs_document_the_lighting_mechanics_neutrally():
    joined = "\n".join(t for _p, t in _agent_facing_docs())
    for mechanic in ("@lighting", "L.new_light", "L.world_nodes", "L.view",
                     "L.image", "emits along its local −Z",
                     "Execution order is fixed"):
        assert mechanic in joined, f"{mechanic} is not documented"


# --- end to end through the real scene compile -----------------------------

