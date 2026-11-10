"""The scene prompt's worked program must still compile and verify.

Without it the prompt could promise a name it does not bind (`Texture`), or an
anchor form its SDK does not accept, and nothing would say so.

What it pins, all of it derived from the prompt rather than restated here:

  * `## File skeleton`'s program executes under the real scene build overlay
    and AST allowlist, and its SceneIR passes the whole scene suite. That one
    assertion covers the sandbox rules, `@shader`, `@lighting`,
    `replace_room_surface`, one-definition-two-instances, `scene.attach`, and
    the object checks lifted over authored definitions.
  * Every `h.<tool>` the prompt tells the agent to call is a tool the daemon
    registers for this layer.
  * Every SDK name the prompt claims is injected is in the build overlay, and
    the overlay binds nothing the prompt does not mention — the direction that
    catches a dead binding as well as a missing one.
"""
from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path

import pytest

from dycraft.agent import stage_harness_client
from dycraft.agent.agent_runner import compose_system_prompt
from dycraft.agent.layers.layer import scene_layer_config
from dycraft.agent.cad import object_worker as W

PROMPT = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "scene.md"
#: The animation layer's own source reference, which teaches the same
#: SDK and carries its own skeleton.
ANIMATION_REFERENCE = PROMPT.with_name("animation_reference.md")

pytestmark_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")


def _skeleton_source(source: Path = PROMPT) -> str:
    """The one fenced program under `## File skeleton`."""
    text = source.read_text()
    section = re.search(r"\n## File skeleton\n(.*?)(?=\n## )", text, re.S)
    assert section, f"{source.name} lost its `## File skeleton` section"
    fences = re.findall(r"```python\n(.*?)```", section.group(1), re.S)
    assert len(fences) == 1, (
        f"expected exactly one program in the skeleton, found {len(fences)}")
    return fences[0]


@pytest.mark.parametrize("source", [PROMPT, ANIMATION_REFERENCE], ids=lambda p: p.stem)
def test_the_skeleton_is_a_complete_program(source):
    """It has to be the whole shape, not an excerpt: that is its job."""
    src = _skeleton_source(source)
    ast.parse(src)                       # syntactically valid Python
    for needle in ("@shader", "@lighting", "Scene(", "Object(",
                   "scene.set_room(", "scene.place(", "scene.attach(",
                   "scene.replace_room_surface(", "scene.set_lighting(",
                   "ir = scene.finalize()"):
        assert needle in src, f"the skeleton no longer shows {needle}"


@pytestmark_cad
@pytest.mark.parametrize("source", [PROMPT, ANIMATION_REFERENCE], ids=lambda p: p.stem)
def test_the_skeleton_compiles_and_passes_the_scene_suite(tmp_path, source):
    """The example an agent copies must be one that works.

    Through `build_current_ir`, not `build_ir_with_layer`: the routed entry
    point is what sends authored source to the frozen build123d interpreter.
    Calling the unrouted one executes in the test process, where the object
    overlay has no geometry names and every `Box(...)` is a `NameError` that
    says nothing about the prompt.
    """
    from dycraft.agent.layers import tools as T

    (tmp_path / "compile_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scene.py").write_text(_skeleton_source(source))
    ctx = T.ToolContext(run_dir=tmp_path, layer_cfg=scene_layer_config())

    out = T.build_current_ir(ctx)
    assert out.get("ok"), (out.get("error_summary") or out.get("build_error")
                           or out.get("sandbox_violation") or out)
    ir = out["ir"]
    assert ir.get("instances"), "the skeleton placed nothing"
    assert ir.get("lighting"), "the skeleton's @lighting block did not reach the IR"
    assert (ir.get("room") or {}).get("surface_replacements"), (
        "the skeleton's replace_room_surface did not reach the IR")


def test_every_tool_the_prompt_names_is_registered_for_this_layer():
    text = compose_system_prompt("scene")
    named = set(re.findall(r"\bh\.([a-z_][a-z0-9_]*)\s*\(", text))
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(stage_harness_client(Path(tmp), "scene")).read_text()
        stubs = {n.name for n in ast.parse(staged).body
                 if isinstance(n, ast.FunctionDef)
                 and not n.name.startswith("_")}
    assert named <= stubs, f"prompt promises unstaged tools: {sorted(named - stubs)}"


def test_the_injected_namespace_and_the_prompt_agree_both_ways():
    """A dead binding is as much a defect as a missing one.

    A binding nothing consumes (an `Environment` with no method to pass it to)
    is dead surface. `Texture`/`TextureMap`/`Mapping` are the other
    direction: scene-only by decision, so this layer must not claim them.
    """
    class _Payload:
        object_handles: dict = {}
        catalog: dict = {}
        name_to_hash: dict = {}

    class _Ctx:
        run_dir = None
        step = 0
        layer_payload = _Payload()

    overlay = set(scene_layer_config().build_overlay_factory(_Ctx()))
    # The curated build123d geometry namespace is a separate surface with a
    # separate page and a separate guard (`test_object_namespace_reference`),
    # and it is 100 names deep — `Airfoil`, `Torus`, `PolarLocations`. This
    # test is about the SDK vocabulary the scene prompt itself teaches, so
    # subtract what `_b123d_namespace` supplies rather than asserting a prompt
    # must name every solid build123d exports.
    from dycraft.agent.cad import _b123d_namespace
    geometry = set(_b123d_namespace.build_namespace(reserved=set()))
    sdk_names = ({n for n in overlay if n[:1].isupper()} - geometry) | {
        "shader", "lighting", "load_object", "objects"}
    sdk_names &= overlay
    text = PROMPT.read_text()
    for name in sorted(sdk_names):
        assert re.search(rf"`{re.escape(name)}`", text), (
            f"{name} is injected into scene source and the prompt never "
            f"mentions it")
    for absent in ("Texture", "TextureMap", "Mapping", "Environment"):
        assert absent not in overlay, f"{absent} is bound again"
        assert not re.search(rf"`{absent}` is injected", text), (
            f"the prompt claims {absent} is injected, and it is not")
