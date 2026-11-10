"""What the default Object Agent prompt is allowed to teach.

The prompt is the agent's entire picture of the SDK, and the two can drift in
opposite directions: the prompt can teach an API that does not exist, or the
runtime can gain one the prompt never mentions. Either way the agent writes
against something that is not there, and the failure surfaces as authored code
that cannot run rather than as a broken test.

These tests pin the contract in the direction that matters: what the agent is
*told*. Note that being absent from the prompt is a different claim from being
absent from the SDK — `make_box` exists, and must never be taught: it makes an
analytic `box` primitive, a second authoring path beside the build123d solids
the prompt's contract is about.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PROMPTS = _REPO / "agent" / "prompts"
_OBJECT_PROMPTS = ("object.md",)

# Withdrawn from the agent's visible guidance, whether or not the SDK still
# carries the implementation.
_WITHDRAWN = (
    "make_box", "make_cylinder", "make_sphere", "make_solid",
    "author_solids", "emit_solid", "author_mesh", "make_mesh",
)


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text()


@pytest.mark.parametrize("name", _OBJECT_PROMPTS)
@pytest.mark.parametrize("api", _WITHDRAWN)
def test_no_object_prompt_teaches_a_withdrawn_api(name, api):
    text = _prompt(name)
    assert api not in text, (
        f"{name} teaches {api!r}, which is not on the agent's surface "
        f"(the implementation stays for compatibility)")


def test_the_prompt_names_the_file_the_object_layer_compiles():
    """The agent writes the file `object.md` tells it to, and the layer compiles
    its `source_filename`. Kept in step, or a rename on either side has the agent
    author a file nothing builds."""
    from dycraft.agent.layers.layer import object_layer_config

    name = object_layer_config().source_filename
    assert f"`{name}`" in _prompt("object.md"), f"object.md never names `{name}`"


def test_the_prompt_carries_no_build123d_worked_example():
    """The object prompt states the authoring contract but shows no geometry
    to copy: no file skeleton and no worked example. The contract is stated in
    prose."""
    sdk = _prompt("object.md")
    for banned in ("## File skeleton", "## Worked example"):
        assert banned not in sdk, f"{banned} is back in the object prompt"
    voc = re.search(r"### build123d vocabulary\n(.*?)\n### ", sdk, re.DOTALL)
    assert voc, "build123d vocabulary section missing"
    assert "```" not in voc.group(1), (
        "the vocabulary section carries a code example again")
    for required in ("obj.add_part(", "obj.add_anchor(", "obj.declare_root(",
                     "ir = obj.finalize()"):
        assert required in sdk, f"the prompt no longer states {required}"


def _namespace_module():
    spec = importlib.util.spec_from_file_location(
        "_b123d_ns", _REPO / "agent" / "cad" / "_b123d_namespace.py")
    ns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ns)
    return ns


def test_the_prompt_defers_the_namespace_to_the_generated_reference():
    """The vocabulary section does not enumerate a hand-picked subset.

    A partial list under-reports the namespace and makes the listed names the
    de-facto vocabulary. The complete list is in the staged skill's generated
    reference; the project additionally carries `object_namespace.md`, checked
    name-for-name against `CURATED` in `tests/test_object_namespace_reference.py`.
    This guard keeps a partial list from growing back.
    """
    # Asserted as a property of the whole prompt rather than of a named
    # section: both lines removed the inline list, but they point at DIFFERENT
    # generated pages and neither owns the other's headings. This line's object
    # prompt sends the agent to the Skill's `references/namespace_reference.md`;
    # the shader-viewer line's project prompt gets `object_namespace.md`
    # appended by `compose_system_prompt`. The invariant is the same either way.
    sdk = _prompt("object.md")
    assert re.search(r"namespace_reference\.md|Namespace reference", sdk), (
        "the object prompt no longer points at a generated namespace page")
    # A partial list shows up as a run of bullets full of backticked
    # `Name`-shaped tokens.
    runs = re.findall(r"(?m)^(?:\s*[*-] .*`[A-Z]\w+`.*\n){4,}", sdk)
    assert not runs, (
        f"the prompt enumerates build123d names again:\n{runs[0][:300]}")


def test_the_material_constructors_are_injected_where_the_prompt_teaches_them():
    """The material constructors are taught in the Materials section, not the
    build123d vocabulary, so `test_every_name_the_prompt_teaches_is_in_the_namespace`
    (which scopes to the vocabulary bullets) would not catch one going missing.

    The AST sandbox rejects `import`, so a name absent from the build overlay
    is unreachable rather than merely inconvenient — exactly the NameError the
    vocabulary test exists to prevent, one section further down the file.

    There is one material vocabulary, so the test has no mode to parametrize
    over.
    """
    from dycraft.agent.layers.layer import _object_build_overlay

    text = _prompt("object.md")
    assert "Material(" in text, "the prompt no longer teaches Material()"
    assert "Emission(" in text, "the prompt no longer teaches Emission()"
    assert "@shader" in text, "the prompt no longer teaches @shader"

    class _Ctx:
        run_dir = None

    overlay = _object_build_overlay(_Ctx())
    for name in ("Material", "Emission", "shader"):
        assert name in overlay, (
            f"the prompt teaches {name} but the object build overlay does not "
            f"bind it")
        assert callable(overlay[name])
    assert "material_from_text" not in overlay


def test_no_image_backed_material_constructor_is_reachable_or_taught():
    """Neither `material()` (a curated library of surfaces) nor
    `material_from_text()` (a text-to-PBR endpoint) may come back half-landed.

    A scalar `Material` is the whole vocabulary. A prompt teaching a
    constructor the overlay does not bind is a `NameError` in the agent's face,
    and an overlay binding one the prompt never mentions is dead surface.

    `Texture` / `TextureMap` / `Mapping` are checked too. They survive in
    `sdk/_material.py` for the importers, which is exactly why the guard has to
    be the overlay rather than the module's absence.
    """
    from dycraft.agent.layers.layer import _object_build_overlay

    class _Ctx:
        run_dir = None

    overlay = _object_build_overlay(_Ctx())
    for name in ("material", "material_from_text", "Texture", "TextureMap",
                 "Mapping"):
        assert name not in overlay, f"{name} is bound in the build overlay"

    for name in _OBJECT_PROMPTS:
        text = _prompt(name)
        assert "material_catalog" not in text, (
            f"{name} still points the agent at the withdrawn catalog")
        assert "material_from_text" not in text, (
            f"{name} still teaches the withdrawn material_from_text()")
        assert not re.search(r"(?<![_a-zA-Z])material\(", text), (
            f"{name} still teaches the withdrawn material() constructor")


def test_the_prompt_carries_the_pbr_guidance():
    """The values an agent types are the whole material vocabulary.

    The prompt is the only material teaching in the repository, so it is pinned
    here — the emission sizing rule in particular is one an agent gets wrong
    silently, by reaching for a familiar-looking number instead of dividing
    watts by area.

    `metallic` and `roughness` are deliberately not pinned here: they are
    backup scalars for renderers that do not execute a shader graph, and the
    shader is *the* authoring interface for a visible surface, so a preview
    value is not part of what the agent is taught to type. What is pinned is
    the vocabulary with no shader equivalent: `alpha` and `Emission`, both
    Material fields the harness applies, and the emission unit.
    """
    text = _prompt("object.md")
    realism = text[text.index("### Materials"):]
    realism = realism[:realism.index("\n## ")]
    for needle in ("shader", "alpha", "W/m"):
        assert needle in realism, (
            f"the Materials section no longer teaches {needle!r}")
    assert "Emission(" in realism, (
        "the Materials section no longer shows how to author emission")


def test_the_prompt_does_not_teach_the_builder_api():
    """`BuildPart` and friends need `with`, which the allowlist rejects.

    An agent whose build123d prior is builder mode loses a compile to them
    (`disallowed AST construct: With`), so the prompt names them only to rule
    them out.
    """
    ns = _namespace_module()
    reference = (_REPO / ".claude" / "skills" / "build123d_authoring" / "references"
                 / "namespace_reference.md").read_text()
    for name in ("BuildPart", "BuildSketch", "BuildLine"):
        assert name not in ns.CURATED, f"{name} is injected but needs `with`"
        assert f"| `{name}` |" not in reference, (
            f"{name} is listed as available but cannot be used without `with`")
    sdk = _prompt("object.md")
    assert "Algebra mode only" in sdk, (
        "if the builder API is unavailable the prompt has to say so — silence "
        "leaves the agent to discover it by failing a compile")


def test_the_prompt_does_not_promise_the_raw_module():
    """`bd` is a guarded view, not build123d itself.

    A prompt saying that `bd.<anything>` reaches the rest of the module would
    send the agent looking for a way out of the sandbox.
    """
    sdk = _prompt("object.md")
    for claim in ("the module is also bound as `bd`", "reaches the rest"):
        assert claim not in sdk, f"prompt still advertises the module: {claim!r}"
    assert "`bd` is **not** the build123d module" in sdk, (
        "say plainly what `bd` is, or an agent assumes the module")


def test_the_prompt_documents_that_print_is_a_no_op():
    """Otherwise an agent print-debugs into a void and cannot tell why."""
    sdk = _prompt("object.md")
    assert re.search(r"`print\(\)`.*?(does nothing|discarded)", sdk), (
        "print() is silently dropped by the sandbox; say so")


def test_the_prompt_teaches_the_local_container_rule():
    """The rule is "not on things you did not create"; a blanket "no subscript
    assignment" would forbid what the sandbox allows."""
    sdk = _prompt("object.md")
    assert "Your own containers are yours" in sdk
    assert "subscript assignment (`lst[i] = ...`)" not in sdk


def test_the_prompt_documents_the_new_error_codes():
    sdk = _prompt("object.md")
    for code in ("SourceRuntimeError", "GeometryKernelError", "SourceTimeout"):
        assert code in sdk, f"{code} is a routine compile outcome; document it"


# Marks a sentence as ruling a name *out* rather than offering it. Checked per
# line, not per document — see `_lines_mentioning`.
_RULED_OUT = ("not in scope", "unavailable", "not available", "is not",
              "are not", "cannot", "deliberately absent")


def _lines_mentioning(text: str, name: str) -> list[str]:
    return [l for l in text.splitlines() if f"`{name}`" in l]


def _assert_prompts_rule_out(name: str) -> None:
    """Every mention of `name` in an object prompt must rule it out.

    Per *line*. A document-wide check would be vacuous: `object.md` contains
    the phrase "not in scope" somewhere, so `... or "not in scope" in prompt`
    would pass wherever the name appeared, including in a vocabulary bullet
    offering it. A name may be discussed, but only to say it
    is not there.
    """
    for prompt in _OBJECT_PROMPTS:
        for line in _lines_mentioning(_prompt(prompt), name):
            assert any(m in line for m in _RULED_OUT), (
                f"{prompt} mentions `{name}` without ruling it out, on:\n"
                f"    {line.strip()}")


def test_the_builder_api_is_neither_taught_nor_injected():
    """Prompt and namespace must agree in both directions.

    A name the namespace binds and the prompt rules out is dead surface; one
    the prompt offers and the namespace lacks is a NameError.
    """
    ns = _namespace_module()
    for name in ("BuildPart", "BuildSketch", "BuildLine"):
        assert name not in ns.CURATED, f"{name} is injected but needs `with`"
        _assert_prompts_rule_out(name)


# Withdrawn from the runtime namespace, each for a reason the agent cannot see
# from the name alone. Kept as a test so a future edit that re-adds one has to
# argue for it here.
_WITHDRAWN_FROM_NAMESPACE = {
    "Text": "renders a font; success depends on the CAD env, not the geometry",
    "FontStyle": "goes with Text",
    "MM": "build123d's millimetre unit; dycraft authors in metres",
    "CM": "as MM",
    "M": "build123d's M is 1000 — `Box(0.24 * M)` is a 240-metre box",
    "IN": "as MM",
    "FT": "as MM",
}


@pytest.mark.parametrize("name", sorted(_WITHDRAWN_FROM_NAMESPACE))
def test_withdrawn_modelling_names_are_neither_injected_nor_taught(name):
    """The unit constants and `Text` are out of scope, in both directions.

    The units are the dangerous half: `MM` and `M` are real build123d exports
    that a model with a build123d prior will reach for, and they are silently
    off by 1000× against this codebase's meter convention — geometry that
    compiles, validates, and renders at the wrong scale.
    """
    ns = _namespace_module()
    assert name not in ns.CURATED, (
        f"{name} is injected: {_WITHDRAWN_FROM_NAMESPACE[name]}")
    _assert_prompts_rule_out(name)


@pytest.mark.parametrize("name", ("object.md",))
def test_both_prompts_say_injected_state_is_read_only(name):
    """An agent that believes it may edit `ir` or `objects` spends a compile
    finding out otherwise, and the message it gets is a refusal rather than a
    hint."""
    text = (_PROMPTS / name).read_text()
    assert "read-only" in text.lower()
    assert "SourceTimeout" in text, "the wall-clock limit is a real outcome"


def test_the_vocabulary_is_marked_provisional():
    """Guards against the temporary list quietly becoming the whole API.

    The prompt teaches a deliberately small core, and all of build123d is
    reachable as `bd`. If that framing disappears, the next reader has no way
    to tell a starting vocabulary from a boundary.
    """
    sdk = _prompt("object.md")
    m = re.search(r"### build123d vocabulary\n(.*?)\n### ", sdk, re.DOTALL)
    assert m
    section = m.group(1)
    assert "`bd`" in section, (
        "the section must say the whole module is reachable as `bd`, or the "
        "curated list reads as the entire available API")


def test_snapshot_is_staged_and_registered():
    """`snapshot` is in the staged client and the daemon registers it."""
    stub = (_REPO / "agent" / "layers" / "_harness_tools.py").read_text()
    assert "def snapshot(" in stub
    # ...and the daemon really does register it, or the stub calls into
    # nothing. Asked of a live session rather than by grepping the source: a
    # grep for the registration call breaks whenever it gains a keyword, while
    # the op itself is fine.
    from dycraft.agent.layers.session import (LayerSession,
                                                    SessionConfig)
    from dycraft.agent.layers.layer import object_layer_config
    import tempfile
    from pathlib import Path as _Path
    session = LayerSession(SessionConfig(
        run_dir=_Path(tempfile.mkdtemp()) / "run",
        layer=object_layer_config(), user_prompt="x"))
    assert "snapshot" in session.handlers


def test_the_staged_harness_exposes_exactly_the_ops_the_daemon_registered():
    """Each layer's staged `harness.py` exposes exactly the ops its daemon
    registers (`cli/README.md`, "Code map").

    A client stub with no handler fails at the wire with `unknown method`; a
    handler with no stub is unreachable and silently dead. Both halves are
    hand-maintained in separate files — that is the design (a layer whose
    surface differs ships its own stub file rather than branching inside a
    shared one), which is exactly why the parity needs a test.

    The object layer is excluded because constructing its Session requires a
    build123d interpreter, which a plain checkout does not have; it shares
    `_harness_tools.py` with scene, so the file is covered either way.

    `animation` is here specifically to hold the absence of
    `record_deviation` on both sides at once. That layer omits
    `deliberate_deviations` from its `layer_meta`, so the op wrote nothing;
    dropping only the stub would leave a dead handler, and dropping only the
    handler would leave a stub that fails at the wire.
    """
    import ast
    import tempfile

    from dycraft.agent import stage_harness_client
    from dycraft.agent.layers.film import FilmSession
    from dycraft.agent.layers.layer import scene_layer_config
    from dycraft.agent.layers.session import (LayerSession,
                                                   SessionConfig)

    def scene_session(run):
        return LayerSession(SessionConfig(run_dir=run, layer=scene_layer_config()))

    for layer, build in (("scene", scene_session), ("animation", FilmSession)):
        run = Path(tempfile.mkdtemp())
        stage_harness_client(run, layer)
        src = (run / "harness.py").read_text()
        client = {n.name for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef)
                  and not n.name.startswith("_")}
        handlers = set(build(run).handlers)
        assert client == handlers, (
            f"{layer}: stub-only={sorted(client - handlers)} "
            f"handler-only={sorted(handlers - client)}")


def test_a_run_directory_holds_exactly_the_folders_a_run_fills():
    """A run dir should describe what happened in it.

    An empty directory is a question a reader has to answer before they can
    trust the rest of the bundle, so `ToolContext` creates only the two a run
    actually writes into.
    """
    import tempfile

    from dycraft.agent.layers.layer import object_layer_config
    from dycraft.agent.layers.tools import ToolContext

    run = Path(tempfile.mkdtemp()) / "run"
    ToolContext(run_dir=run, layer_cfg=object_layer_config())
    created = sorted(p.name for p in run.iterdir())
    assert created == ["compile_reports", "renders"], created


def test_the_prompt_documents_the_exact_collision_probe():
    """A probe the overlay binds but the prompt never names is unreachable.

    `gap` is the object layer's only exact collision answer. Every other
    collision probe here is a bounding volume while `NoSelfPenetration` judges
    with B-rep booleans, so an agent that cannot find this name is left reading
    a box against a verdict that did not come from one.

    Checked on the composed prompt, which is what the agent receives.
    """
    from dycraft.agent.agent_runner import compose_system_prompt
    from dycraft.agent.layers.layer import _OBJECT_EXACT_PROBES

    text = compose_system_prompt("object")
    for name in _OBJECT_EXACT_PROBES:
        assert f"`{name}`" in text or f"`{name}(" in text, (
            f"exact-collision probe {name!r} is not documented in the object prompt")


def test_the_prompt_marks_its_bounding_box_collision_probes_as_such():
    """The prompt has to say which answers are surfaces and which are boxes.

    A `collided` that came from a box reads identically to one that came from a
    surface, and the prompt routes the agent to these probes after a
    `NoSelfPenetration` failure. `within_report` is the one that is a box by
    definition; the rest are measured unless the caller passes `exact=False`.
    """
    text = _prompt("object.md")
    for probe in ("penetration_pairs", "pair_report", "within_report"):
        assert probe in text, f"{probe} is not documented at all"
    lowered = text.lower()
    assert "over-report" in lowered, (
        "the prompt must say where a box answer over-reports")
    assert "bounding proxy" in lowered or "bounding volume" in lowered, (
        "the prompt must name what the box probes are")
    assert "exact=false" in lowered.replace(" ", ""), (
        "the prompt must name the way back to a box answer")


# ---------------------------------------------------------------------------
# The source-style sections against the sandbox they describe
# ---------------------------------------------------------------------------

#: One authoring prompt per layer, and the heading its departures live under.
#: All three layers share `sandbox.check_ast`, so all three make the same claim
#: and can be wrong about it independently.
_SOURCE_STYLE_SECTIONS = (
    ("object.md", "## Source-style constraints — must follow"),
    ("scene.md", "## Source-style constraints — must follow"),
    # `animation_sdk.md` dropped: the animation layer authors bpy through
    # `h.blender` and has no injected SDK or source-style section.
)

#: Python the agent might reach for, one snippet per construct. Keyed by the
#: token a prompt writes when it names the construct, so the two directions
#: below can be checked by searching for `` `token` ``.
_CONSTRUCTS = {
    "import": "import math\n",
    "with": "with 1 as x:\n    pass\n",
    "async": "async def f():\n    pass\n",
    "global": "x = 1\ndef f():\n    global x\n    x = 2\n",
    "try": "try:\n    x = 1\nexcept ValueError:\n    x = 2\n",
    "except": "try:\n    x = 1\nexcept ValueError:\n    x = 2\n",
    "raise": "raise ValueError('x')\n",
    "assert": "assert 1 == 1\n",
    "match": "match 1:\n    case 1:\n        x = 2\n",
    "yield": "def f():\n    yield 1\n",
    "nonlocal": "def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2\n    g()\n",
    ":=": "if (n := 3) > 2:\n    x = n\n",
}


def _rejects(construct: str) -> bool:
    from dycraft.agent.cad import sandbox as SB
    try:
        SB.check_ast(_CONSTRUCTS[construct], mode="exec")
    except SB.SandboxViolation:
        return True
    return False


def _source_style_section(prompt: str, heading: str) -> str:
    text = _prompt(prompt)
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


@pytest.mark.parametrize("prompt,heading", _SOURCE_STYLE_SECTIONS)
@pytest.mark.parametrize("construct", sorted(_CONSTRUCTS))
def test_the_source_style_section_names_exactly_what_the_sandbox_rejects(
        construct, prompt, heading):
    """A section that lists departures must list the real ones and no others.

    Both directions cost the agent a run. Naming a construct the sandbox
    accepts is the expensive one — an obedient author writes around syntax that
    would have worked, and no test that only exercises the sandbox can see it.
    Omitting one the sandbox rejects spends a compile on a `sandbox_violation`.

    Keyed on the backticked token, because that is how a prompt names a
    construct; prose that merely contains the word does not trip it.
    """
    section = _source_style_section(prompt, heading)
    named = f"`{construct}`" in section
    if _rejects(construct):
        assert named, (
            f"{prompt} does not name `{construct}`, which the sandbox rejects")
    else:
        assert not named, (
            f"{prompt} names `{construct}` among its departures, but the "
            f"sandbox accepts it")


def test_the_staged_object_client_exposes_exactly_the_registered_ops():
    """A stub that calls an unregistered method gets `unknown method`, which
    tells the agent nothing about why."""
    import ast
    import tempfile
    from pathlib import Path
    from dycraft.agent import stage_harness_client

    staged = Path(tempfile.mkdtemp())
    stage_harness_client(staged, "object")
    tree = ast.parse((staged / "harness.py").read_text())
    stubs = {n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    # `set_render_backend` arrived with the shader-viewer line: without an op,
    # an agent that wants Blender relaunches the daemon with
    # DYCRAFT_RENDER_BACKEND set, and a second daemon on one run dir corrupts
    # trace.jsonl, meta.json and the step counter.
    assert stubs == {"compile", "finish", "probe", "snapshot",
                     "set_render_backend",
                     "final_render", "record_deviation", "review"}



@pytest.mark.parametrize("layer", ["object", "scene", "plain"])
def test_composed_prompt_has_no_duplicate_headings(layer):
    from collections import Counter
    from dycraft.agent.agent_runner import compose_system_prompt

    headings, fence = [], None
    for line in compose_system_prompt(layer).splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None and re.match(r"^#{1,6}\s", line):
            headings.append(re.sub(r"^#+\s*", "", line).strip().casefold())
    assert not {h: n for h, n in Counter(headings).items() if n > 1}


def test_animation_stubs_forward_every_parameter_they_accept():
    """A stub that takes an argument and drops it fails silently, forever.

    An agent asking `review(frames=...)` for a specific interval would get the
    planned states and no error. The layer-parity tests compare op *names*,
    and the handler tests drive `submit` directly, so neither would see it.
    """
    import ast
    import tempfile

    from dycraft.agent import stage_harness_client

    for layer in ("animation", "scene", "object"):
        run = Path(tempfile.mkdtemp())
        stage_harness_client(run, layer)
        tree = ast.parse((run / "harness.py").read_text())
        for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and not n.name.startswith("_")]:
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            call = next((n for n in ast.walk(fn)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "_call"), None)
            if call is None:
                continue
            forwarded = {c.value for n in ast.walk(call)
                         for c in ([n] if isinstance(n, ast.Constant) else [])
                         if isinstance(c.value, str)}
            forwarded |= {kw.arg for kw in call.keywords if kw.arg}
            for n in ast.walk(call):
                if isinstance(n, ast.keyword) and n.arg:
                    forwarded.add(n.arg)
            # Or consumed by the client itself: `job_result(wait=)` waits here.
            forwarded |= {n.id for c in ast.walk(fn)
                          if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                          and c.func.id == "_await"
                          for n in ast.walk(c) if isinstance(n, ast.Name)}
            dropped = sorted(a for a in args if a not in forwarded)
            assert not dropped, f"{layer}:{fn.name} accepts but never forwards {dropped}"
