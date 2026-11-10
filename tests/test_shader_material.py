"""`Material.shader` and the `@shader` decorator.

Three things are being pinned, in rising order of how badly they fail.

The **corpus guard** is first: a material with no shader must still serialize
to exactly the four scalar keys it always did. `sdk/ir.py::content_hash` hashes
the whole IR, so one extra or missing key re-ids every stored object, every
catalog ref, every render-cache key, and every stored VLM judgment.

Then the **new value semantics**: what a shader-only material is allowed to
omit, and the two pairs it may not carry together.

Then the **author-time checks**, which are the whole reason a shader is an
authored function rather than a source string. Every failure they catch becomes
a compile report; everything they miss becomes a render warning several minutes
later, so what they cover is a design decision rather than an implementation
detail.
"""

from __future__ import annotations

import pytest

from dycraft.agent.cad import sandbox as SB
from dycraft.sdk import _mesh_assets as _ma
from dycraft.sdk._errors import InvalidPrimitive, SandboxViolation
from dycraft.sdk._material import (DEFAULT_ALPHA, DEFAULT_METALLIC,
                                   DEFAULT_PART_COLOR, DEFAULT_ROUGHNESS,
                                   Emission, Mapping, Material, Shader,
                                   Texture, TextureMap, material_from_ir)
from dycraft.sdk import _shader as SH


@pytest.mark.parametrize("form", ["legacy", "three", "four"])
def test_render_child_calls_the_authored_shader_signature(session, form):
    """Exercise the real dispatch without requiring a Blender installation."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from dycraft.render._world import _material_to_render

    params = "nodes, links, bsdf" + (", mat" if form == "four" else "")
    source = (f"@shader\ndef surface({params}):\n"
              "    bsdf.append(nodes)\n    bsdf.append(links)\n"
              + ("    bsdf.append(mat)\n" if form == "four" else ""))
    sh = _decorate(session, source, "surface")
    material_ir = Material(shader=sh).to_ir()
    if form == "legacy":
        material_ir["shader"].pop("params", None)
    packed = _material_to_render(material_from_ir(
        material_ir, default_color=DEFAULT_PART_COLOR))

    # _blender_child is a standalone bpy script. Compile its actual dispatch
    # function with node-tree stand-ins: this tests source capture, IR packing,
    # and the call boundary, which must pass all four arguments through.
    path = Path(__file__).resolve().parents[1] / "render" / "_blender_child.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_run_shader")
    ns = {"_STRICT_MATERIALS": True, "MaterialError": RuntimeError}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    nodes, links, received = object(), object(), []
    mat = SimpleNamespace(node_tree=SimpleNamespace(nodes=nodes, links=links))
    assert ns["_run_shader"]("surface", mat, received,
                             packed["shader"], packed, 1.0)
    assert received == [nodes, links] + ([mat] if form == "four" else [])


_BODY = ('def marble(nodes, links, bsdf):\n'
         '    n = nodes.new("ShaderNodeTexNoise")\n'
         '    links.new(n.outputs["Fac"], bsdf.inputs["Base Color"])\n')


def _map(kind="base_color"):
    return TextureMap(image="assets/textures/" + "a" * 32 + ".png",
                      digest="a" * 32)


@pytest.fixture
def session(tmp_path):
    """A run dir with the asset session installed, as a real compile has.

    Deliberately WITHOUT `source_text`, which exercises `@shader`'s fallback:
    the disk read of `<run_dir>/object.py`. A real compile sets `source_text`
    on the session before it execs, and that is the path
    `test_the_session_source_is_preferred_over_the_file_on_disk` covers.
    """
    run = tmp_path / "run"
    run.mkdir()
    _ma.set_active_mesh_session(_ma.MeshAssetSession(run))
    try:
        yield run
    finally:
        _ma.clear_active_mesh_session()


def _author(run, source: str):
    """Write an `object.py` and hand back a stub with the authored name."""
    (run / "object.py").write_text(source, encoding="utf-8")


def test_the_session_source_is_preferred_over_the_file_on_disk(tmp_path):
    """`compile.build_ir_with_layer` hands the session the bytes it execs.

    Those are the bytes the shader was written in. Re-reading `object.py` makes
    the answer depend on the file's state *after* execution instead — which is
    why the staging path stopped doing it — so a divergent file must lose. The
    two only differ when something rewrote the file mid-build; this pins which
    one wins so a regression to the old re-read is visible.
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "object.py").write_text(_STALE_SOURCE, encoding="utf-8")
    session = _ma.MeshAssetSession(run)
    session.source_text = _LIVE_SOURCE
    _ma.set_active_mesh_session(session)
    try:
        out = SH.shader(_stub("live"))
    finally:
        _ma.clear_active_mesh_session()
    assert out.name == "live"
    # The values, not a marker comment: `ast.get_source_segment` ends at the
    # statement's last token, so a trailing comment never reaches `body`.
    assert "(1.0, 1.0, 1.0, 1.0)" in out.body
    assert "(0.0, 0.0, 0.0, 1.0)" not in out.body


_STALE_SOURCE = """\
@shader
def live(nodes, links, bsdf):
    bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # STALE
"""

_LIVE_SOURCE = """\
@shader
def live(nodes, links, bsdf):
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # LIVE
"""


def _stub(name: str):
    """A real function with the authored name, as `exec` would have bound.

    The decorator only reads `__name__` off it — the body it ships comes from
    the source text, not from this object — but it must genuinely be callable,
    because that is the first thing `@shader` checks.
    """
    def fn(nodes, links, bsdf):     # pragma: no cover - never invoked here
        raise AssertionError("a shader body must not run in this interpreter")
    fn.__name__ = name
    return fn


def _decorate(run, source: str, name: str):
    """Run `@shader` against `source` the way an exec of it would."""
    _author(run, source)
    return SH.shader(_stub(name))


# ---------------------------------------------------------------------------
# The corpus guard
# ---------------------------------------------------------------------------

def test_a_material_with_no_shader_serializes_to_exactly_four_keys():
    """The one assertion that protects every stored object id.

    `content_hash` covers the whole canonical IR, so if this material grew a
    key — or lost one to the conditional emission — every object in the corpus
    would re-id and take its catalog refs, render-cache keys and verifier cache
    with it.
    """
    d = Material(color=(1, 2, 3)).to_ir()
    assert d == {"color": [1, 2, 3], "metallic": DEFAULT_METALLIC,
                 "roughness": DEFAULT_ROUGHNESS, "alpha": DEFAULT_ALPHA}


def test_the_scalars_are_still_filled_from_defaults_without_a_shader():
    """Optional in the signature, required in the value.

    The fields gained `None` defaults so a shader material can omit them. A
    material with no shader must not notice: it still gets the module defaults,
    which is what keeps the line above true.
    """
    m = Material(color=(9, 9, 9))
    assert (m.metallic, m.roughness, m.alpha) == (
        DEFAULT_METALLIC, DEFAULT_ROUGHNESS, DEFAULT_ALPHA)


def test_a_material_with_no_shader_still_requires_a_color():
    with pytest.raises(InvalidPrimitive) as e:
        Material()
    assert "color or a shader" in str(e.value)


# ---------------------------------------------------------------------------
# Shader value semantics
# ---------------------------------------------------------------------------

def test_a_shader_alone_is_a_complete_material():
    """No colour, no roughness, no metallic — the shader computes all three."""
    m = Material(shader=Shader(name="marble", body=_BODY))
    assert m.color is None and m.roughness is None and m.metallic is None
    assert m.to_ir() == {"shader": {"name": "marble", "body": _BODY}}


def test_a_shader_material_emits_only_the_scalars_its_author_wrote():
    """Nothing is invented on the way out.

    Authored scalars supply backups for renderers without shader support.
    Writing one must not invent the others; their absence survives the IR.
    """
    m = Material(color=(10, 20, 30), shader=Shader(name="m", body=_BODY))
    d = m.to_ir()
    assert set(d) == {"color", "shader"}


def test_the_shader_round_trips_through_the_ir():
    sh = Shader(name="marble", body=_BODY)
    m = Material(shader=sh)
    back = material_from_ir(m.to_ir(), default_color=DEFAULT_PART_COLOR)
    assert back.shader == sh


def test_reading_a_shader_material_back_preserves_scalar_ABSENCE():
    """`material_from_ir` must not invent scalars the author never wrote.

    Filling them here made the reader non-idempotent — see the round-trip test
    below — and it contradicted the one thing `shader` means. The fallback a
    non-Cycles backend needs is supplied at the packing boundary instead.
    """
    back = material_from_ir(
        Material(shader=Shader(name="m", body=_BODY)).to_ir(),
        default_color=DEFAULT_PART_COLOR)
    assert back.shader is not None
    assert back.color is None
    assert back.roughness is None
    assert back.metallic is None
    assert back.alpha is None


def test_reading_a_shader_material_keeps_the_scalars_that_WERE_written():
    """The distinction is authored-or-not, not shader-or-not."""
    m = Material(color=(9, 8, 7), roughness=0.2,
                 shader=Shader(name="m", body=_BODY))
    back = material_from_ir(m.to_ir(), default_color=DEFAULT_PART_COLOR)
    assert back.color == (9, 8, 7)
    assert back.roughness == 0.2
    assert back.metallic is None and back.alpha is None


@pytest.mark.parametrize("material", [
    "shader_only", "shader_and_color", "shader_and_roughness",
    "legacy", "legacy_with_emission",
])
def test_reading_an_ir_and_writing_it_back_changes_nothing(material):
    """**`material_from_ir(d).to_ir() == d`, and that is a contract.**

    An IR whose canonical bytes change just by being parsed and re-serialized
    is a trap for everything that round-trips one — a catalog rebase, a
    migration, a regression oracle comparing two implementations. The bytes are
    what `content_hash` covers, so "unchanged" has to mean byte-unchanged.

    Legacy materials are in the same test on purpose: their scalars ARE filled
    from the defaults on the way in, and that stays idempotent only because
    `to_ir()` then writes all four back.
    """
    sh = Shader(name="m", body=_BODY)
    cases = {
        "shader_only": Material(shader=sh),
        "shader_and_color": Material(color=(9, 8, 7), shader=sh),
        "shader_and_roughness": Material(roughness=0.2, shader=sh),
        "legacy": Material(color=(1, 2, 3)),
        "legacy_with_emission": Material(
            color=(1, 2, 3),
            emission=Emission(color=(255, 200, 150), strength=12.0)),
    }
    d = cases[material].to_ir()
    once = material_from_ir(d, default_color=DEFAULT_PART_COLOR).to_ir()
    twice = material_from_ir(once, default_color=DEFAULT_PART_COLOR).to_ir()
    assert once == d, f"parsing and re-writing changed the IR: {d} -> {once}"
    assert twice == once
    # Byte-level, because that is what the content hash actually sees.
    import json
    assert (json.dumps(once, sort_keys=True)
            == json.dumps(d, sort_keys=True))


def test_a_sparse_legacy_ir_still_reads_with_the_module_defaults():
    """Older IR that omits a scalar has to keep working."""
    m = material_from_ir({"color": [5, 5, 5]},
                         default_color=DEFAULT_PART_COLOR)
    assert (m.metallic, m.roughness, m.alpha) == (
        DEFAULT_METALLIC, DEFAULT_ROUGHNESS, DEFAULT_ALPHA)
    assert material_from_ir(None, default_color=DEFAULT_PART_COLOR).color == (
        DEFAULT_PART_COLOR)


def test_part_color_still_answers_for_a_shader_part():
    """A documented query the object probe and the scene layer both read.

    `part_material(...).color` reports the absence honestly; `part_color`
    answers the question it was asked, with the value such a part will actually
    be drawn in by anything that cannot evaluate a graph.
    """
    from dycraft.sdk._object_kernel import part_color, part_material

    ir = {"parts": [
        {"name": "shaded", "material": Material(shader=Shader(
            name="m", body=_BODY)).to_ir()},
        {"name": "scalar", "material": Material(color=(1, 2, 3)).to_ir()},
        {"name": "bare"}]}
    assert part_color(ir, "shaded") == DEFAULT_PART_COLOR
    assert part_material(ir, "shaded").color is None
    assert part_color(ir, "scalar") == (1, 2, 3)
    assert part_color(ir, "bare") is None


def test_a_shader_and_texture_maps_cannot_describe_one_surface():
    with pytest.raises(InvalidPrimitive) as e:
        Material(color=(1, 2, 3), shader=Shader(name="m", body=_BODY),
                 texture=Texture(map_items=(("base_color", _map()),)))
    assert "both a shader and texture" in str(e.value)


def test_malformed_ir_carrying_both_drops_the_maps_rather_than_raising():
    """`material_from_ir` runs inside the renderers and must not raise.

    `Material` refuses the pair, so the reader has to resolve it. The shader
    owns the sockets the maps would have driven, so the maps go — and `mapping`
    follows, because it sizes maps that no longer exist.
    """
    d = {"color": [1, 2, 3],
         "shader": {"name": "m", "body": _BODY},
         "texture": {"base_color": {"image": "assets/textures/" + "a" * 32
                                             + ".png", "digest": "a" * 32}},
         "mapping": {"scale_m": 0.2}}
    m = material_from_ir(d, default_color=DEFAULT_PART_COLOR)
    assert m.shader is not None
    assert m.texture is None and m.mapping is None


def test_a_shader_block_that_will_not_parse_degrades_to_the_scalars():
    """Same posture as a malformed texture: no exception inside a render."""
    for bad in ({"name": "m"}, {"body": _BODY}, {"name": "1nope",
                                                 "body": _BODY},
                {"name": "m", "body": "def m(:\n"}, "nonsense", None):
        m = material_from_ir({"color": [4, 5, 6], "shader": bad},
                             default_color=DEFAULT_PART_COLOR)
        assert m.shader is None, bad
        assert m.color == (4, 5, 6)


def test_emission_survives_beside_a_shader():
    """A lit shade has both. `Emission` stays a harness field either way."""
    m = Material(shader=Shader(name="m", body=_BODY),
                 emission=Emission(color=(255, 200, 150), strength=23.0))
    d = m.to_ir()
    assert set(d) == {"shader", "emission"}


def test_a_shader_material_is_hashable_and_reusable_across_parts():
    """One constant bound to many parts has to keep comparing equal."""
    a = Material(shader=Shader(name="m", body=_BODY))
    b = Material(shader=Shader(name="m", body=_BODY))
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_a_mapping_still_needs_maps_even_with_a_shader():
    with pytest.raises(InvalidPrimitive):
        Material(shader=Shader(name="m", body=_BODY),
                 mapping=Mapping(scale_m=0.2))


@pytest.mark.parametrize("bad", [
    "not a shader", 3, {"name": "m", "body": _BODY},
])
def test_the_shader_field_takes_a_shader_and_nothing_else(bad):
    with pytest.raises(InvalidPrimitive) as e:
        Material(color=(1, 2, 3), shader=bad)
    assert "must be a Shader" in str(e.value)


def test_a_shader_names_a_callable_and_carries_parseable_source():
    with pytest.raises(InvalidPrimitive):
        Shader(name="not an identifier", body=_BODY)
    with pytest.raises(InvalidPrimitive):
        Shader(name="m", body="   ")
    with pytest.raises(InvalidPrimitive) as e:
        Shader(name="m", body="def m(nodes, links, bsdf:\n    pass\n")
    assert "does not parse" in str(e.value)


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

_GOOD_SOURCE = '''MARBLE_UNUSED = 1

@shader
def marble(nodes, links, bsdf):
    tc = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    links.new(tc.outputs["UV"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bsdf.inputs["Base Color"])

MARBLE = Material(shader=marble)
'''


def test_the_decorator_recovers_the_exact_def_and_nothing_else(session):
    """The body has to be a self-contained `def` the render child can exec.

    `ast.get_source_segment` starts at `FunctionDef.lineno`, which for a
    decorated function is the `def` line — so the decorator is excluded, and
    neither the module constant above nor the binding below comes along.
    """
    sh = _decorate(session, _GOOD_SOURCE, "marble")
    assert sh.name == "marble"
    assert sh.body.startswith("def marble(nodes, links, bsdf):")
    assert "@shader" not in sh.body
    assert "MARBLE_UNUSED" not in sh.body and "Material(" not in sh.body
    # And it is executable on its own, which is what the child relies on.
    ns: dict = {}
    exec(compile(sh.body, "<t>", "exec"), ns)      # noqa: S102
    assert callable(ns["marble"])


def test_the_body_is_what_lands_in_the_ir(session):
    m = Material(shader=_decorate(session, _GOOD_SOURCE, "marble"))
    assert m.to_ir()["shader"]["body"].startswith("def marble(")


def test_a_shader_must_declare_an_accepted_signature(session):
    for sig in ("(nodes, links)", "(nodes, links, bsdf, extra)",
                "(a, b, c)", "(nodes, links, bsdf=None)",
                "(nodes, links, bsdf, *rest)", "()"):
        src = f"@shader\ndef s{sig}:\n    pass\n"
        with pytest.raises(InvalidPrimitive) as e:
            _decorate(session, src, "s")
        assert "(nodes, links, bsdf)" in str(e.value), sig


def test_a_free_variable_is_refused_and_named(session):
    """The failure an agent will actually hit.

    Module constants are how this codebase's object files are written, so
    `noise.inputs["Scale"].default_value = MARBLE_SCALE` is the natural thing
    to reach for — and the body travels to Blender without the rest of the
    file. Refused loudly here rather than shipped as an unbound name inside a
    Cycles render.
    """
    src = ('MARBLE_SCALE = 3.0\n\n'
           '@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    n = nodes.new("ShaderNodeTexNoise")\n'
           '    n.inputs["Scale"].default_value = MARBLE_SCALE\n')
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    msg = str(e.value)
    assert "MARBLE_SCALE" in msg and "will not exist when Blender runs it" in msg


def test_names_the_body_binds_itself_are_not_free(session):
    """Locals, loop variables, comprehensions, and nested helpers all count."""
    src = ('@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    scale = 3.0\n'
           '    def tweak(node, v):\n'
           '        node.inputs["Scale"].default_value = v\n'
           '    made = [nodes.new("ShaderNodeTexNoise") for _ in range(2)]\n'
           '    for i, n in enumerate(made):\n'
           '        tweak(n, scale * (i + 1))\n'
           '    links.new(made[0].outputs["Fac"], bsdf.inputs["Base Color"])\n')
    sh = _decorate(session, src, "s")
    assert sh.name == "s"


def test_an_unknown_node_type_is_caught_before_blender_ever_runs(session):
    """The version-drift class, caught at author time.

    `ShaderNodeTexMusgrave` was removed in Blender 4.1 and is exactly what an
    agent working from older material tutorials asks for. The generated node
    reference is what makes this answerable in an interpreter with no bpy.
    """
    src = ('@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    m = nodes.new("ShaderNodeTexMusgrave")\n')
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    msg = str(e.value)
    assert "ShaderNodeTexMusgrave" in msg
    assert "does not have" in msg
    assert any("blender_nodes.json" in r
               for r in e.value.to_dict()["suggested_repairs"])


def test_a_misspelled_node_type_gets_a_nearest_match(session):
    src = ('@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    m = nodes.new("ShaderNodeTexNoize")\n')
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    assert any("ShaderNodeTexNoise" in r
               for r in e.value.to_dict()["suggested_repairs"])


def test_a_computed_node_type_is_left_for_blender(session):
    """Only literals are checked; a computed one is not guessed at."""
    src = ('@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    kind = "ShaderNode" + "TexNoise"\n'
           '    m = nodes.new(kind)\n')
    assert _decorate(session, src, "s").name == "s"


def test_two_shaders_of_one_name_are_refused_rather_than_guessed(session):
    src = ('@shader\ndef s(nodes, links, bsdf):\n    pass\n\n'
           '@shader\ndef s(nodes, links, bsdf):\n    pass\n')
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    assert "cannot tell which one" in str(e.value)


def test_a_shader_must_be_defined_at_module_level(session):
    """A nested `def`'s source segment is indented and would not compile
    standalone, so the shape is refused rather than silently mangled."""
    src = ('def outer():\n'
           '    @shader\n'
           '    def s(nodes, links, bsdf):\n'
           '        pass\n')
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    assert "module-level" in str(e.value)


def test_the_decorator_must_actually_be_the_decorator(session):
    """`shader(fn)` called by hand skips the sandbox exemption its body needs,
    so the shape that cannot work is refused with the reason."""
    src = 'def s(nodes, links, bsdf):\n    pass\n'
    with pytest.raises(InvalidPrimitive) as e:
        _decorate(session, src, "s")
    assert "not decorated" in str(e.value)


def test_shader_authoring_needs_an_asset_session():
    """No asset session means no authored source to read back.

    Not "needs the object layer" any more: the scene layer installs a session
    too, because a room's floor and walls get the same material vocabulary as
    a part.
    """
    _ma.clear_active_mesh_session()
    with pytest.raises(InvalidPrimitive) as e:
        SH.shader(_stub("s"))
    assert "asset session" in str(e.value)


def test_the_decorator_takes_a_function(session):
    _author(session, _GOOD_SOURCE)
    with pytest.raises(InvalidPrimitive) as e:
        SH.shader("marble")
    assert "decorates a function" in str(e.value)


# ---------------------------------------------------------------------------
# The sandbox exemption
# ---------------------------------------------------------------------------

_ATTR_WRITES = (
    'n.inputs["Scale"].default_value = 3.0',
    'ramp.color_ramp.elements[0].position = 0.42',
    'mix.data_type = "RGBA"',
    'node.location = (100, 0)',
)


@pytest.mark.parametrize("line", _ATTR_WRITES)
def test_a_shader_body_may_set_node_properties(line):
    """Shader authoring IS attribute assignment.

    Every one of these is an `ast.Attribute` target, which the sandbox rejects
    everywhere else — so without the exemption the entire Blender node API is
    unreachable and the design does not exist.
    """
    SB.check_ast(f"@shader\ndef s(nodes, links, bsdf):\n    {line}\n")


@pytest.mark.parametrize("line", _ATTR_WRITES)
def test_the_same_line_outside_a_shader_body_is_still_refused(line):
    """The exemption is scoped, not a relaxation of the rule."""
    with pytest.raises(SandboxViolation) as e:
        SB.check_ast(line + "\n")
    assert "attribute assignment not allowed" in str(e.value)


def test_an_undecorated_function_gets_no_exemption():
    with pytest.raises(SandboxViolation):
        SB.check_ast("def s(nodes, links, bsdf):\n"
                     "    nodes.location = 1\n")


@pytest.mark.parametrize("line, expected", [
    ("import bpy", "disallowed AST construct: Import"),
    ("from bpy import ops", "disallowed AST construct: ImportFrom"),
    ("x = nodes._private", "private/dunder attribute access"),
    ("x = ().__class__", "private/dunder attribute access"),
    ("x = eval('1')", "reference to disallowed name: 'eval'"),
    ("x = getattr(nodes, 'new')", "reference to disallowed name: 'getattr'"),
    ("x = open('/etc/passwd')", "reference to disallowed name: 'open'"),
])
def test_everything_else_still_applies_inside_a_shader_body(line, expected):
    """Only ONE rule is relaxed.

    This is what makes an authored function better than an authored string: the
    body the render child `exec`s has been walked by the same allowlist as the
    rest of `object.py`, so `import`, dunder chains and the banned builtins are
    all still out of reach.
    """
    with pytest.raises(SandboxViolation) as e:
        SB.check_ast(f"@shader\ndef s(nodes, links, bsdf):\n    {line}\n")
    assert expected in str(e.value)


def test_a_violation_inside_a_shader_body_reports_its_own_line():
    """A real AST carries a position; a source string would not."""
    src = ("@shader\n"
           "def s(nodes, links, bsdf):\n"
           "    a = 1\n"
           "    import bpy\n")
    with pytest.raises(SandboxViolation) as e:
        SB.check_ast(src)
    assert e.value.to_dict()["evidence"]["lineno"] == 4


# ---------------------------------------------------------------------------
# End to end: an authored object.py through the sandbox and overlay into the IR
# ---------------------------------------------------------------------------

_OBJECT_PY = '''@shader
def brushed_steel(nodes, links, bsdf):
    tc = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 220.0
    noise.inputs["Detail"].default_value = 2.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.55, 0.56, 0.58, 1.0)
    ramp.color_ramp.elements[1].color = (0.78, 0.79, 0.81, 1.0)
    links.new(tc.outputs["UV"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.34

STEEL = Material(shader=brushed_steel)

obj = Object("bracket")
body = obj.make_box("body", 0.2, 0.1, 0.05, material=STEEL)
obj.declare_root(body)
ir = obj.finalize()
'''


def test_an_authored_object_carries_its_shader_into_the_ir(tmp_path):
    """The whole author-side path in one go.

    `check_ast` accepts the body's property writes, the overlay resolves
    `shader`, the decorator recovers the source, `Material` accepts it with no
    scalars, and `finalize` puts it in the part. Uses `make_box` so this needs
    no CAD interpreter.

    Note the source has to exist on disk: `object_worker.run_object_source`
    writes `<run_dir>/object.py` before executing it, which is the same file
    `@shader` reads back.
    """
    from dycraft.agent.layers.compile import build_ir_with_layer
    from dycraft.agent.layers.layer import object_layer_config
    from dycraft.sdk.object_builder import _reset_object_singleton

    _reset_object_singleton()
    (tmp_path / "object.py").write_text(_OBJECT_PY, encoding="utf-8")

    ctx = type("C", (), {"run_dir": tmp_path, "step": 0})()
    out = build_ir_with_layer(_OBJECT_PY, layer_cfg=object_layer_config(),
                              ctx=ctx)
    assert out["ok"], out
    part = out["ir"]["parts"][0]
    material = part["material"]
    assert set(material) == {"shader"}, material
    assert material["shader"]["name"] == "brushed_steel"
    assert material["shader"]["body"].startswith("def brushed_steel(")
    assert "ShaderNodeValToRGB" in material["shader"]["body"]
    # And it survives packing for the renderer, which is what Cycles reads.
    from dycraft.render._world import _material_to_render
    packed = _material_to_render(
        material_from_ir(material, default_color=DEFAULT_PART_COLOR))
    assert packed["shader"]["name"] == "brushed_steel"
    # The scalars the author never wrote are supplied on the way in, so the
    # non-Cycles backends and the Cycles fallback both have a surface.
    assert packed["color"] == pytest.approx(
        tuple(c / 255.0 for c in DEFAULT_PART_COLOR))


def test_a_bad_shader_in_an_authored_object_is_a_compile_report(tmp_path):
    """What the agent actually sees when it gets a shader wrong.

    A structured build error in the compile report, naming the node type and
    the nearest match — not a render-child traceback, and not a gray surface.
    """
    from dycraft.agent.layers.compile import build_ir_with_layer
    from dycraft.agent.layers.layer import object_layer_config
    from dycraft.sdk.object_builder import _reset_object_singleton

    _reset_object_singleton()
    src = ('@shader\n'
           'def s(nodes, links, bsdf):\n'
           '    m = nodes.new("ShaderNodeTexMusgrave")\n'
           '\n'
           'M = Material(shader=s)\n'
           'obj = Object("o")\n'
           'b = obj.make_box("b", 0.1, 0.1, 0.1, material=M)\n'
           'obj.declare_root(b)\n'
           'ir = obj.finalize()\n')
    (tmp_path / "object.py").write_text(src, encoding="utf-8")
    ctx = type("C", (), {"run_dir": tmp_path, "step": 0})()
    out = build_ir_with_layer(src, layer_cfg=object_layer_config(), ctx=ctx)
    assert not out["ok"]
    err = out["build_error"]
    assert "ShaderNodeTexMusgrave" in err["summary"]
    assert any("ShaderNode" in r for r in err["suggested_repairs"])


def test_the_packed_render_dict_always_carries_all_four_scalars():
    """The IR omits absent scalars; the packed dict must not.

    Two opposite jobs. `to_ir()` omits them because the IR is content-hashed
    and an unconditional key re-ids the corpus. `_material_to_render` feeds
    `_pbr_material` and `_build_material`, which SUBSCRIPT these keys directly
    in six places — so an absent one would be a `KeyError` inside a renderer
    rather than the fallback it is supposed to be.
    """
    from dycraft.render._world import _material_to_render

    sh = Shader(name="m", body=_BODY)
    for material in (Material(shader=sh),
                     Material(color=(9, 9, 9), shader=sh),
                     Material(color=(1, 2, 3))):
        packed = _material_to_render(material)
        assert {"color", "metallic", "roughness", "alpha"} <= set(packed), (
            f"{material} packed to {sorted(packed)}")

    # A shader material whose author wrote nothing still packs the defaults.
    packed = _material_to_render(Material(shader=sh))
    assert packed["roughness"] == DEFAULT_ROUGHNESS
    assert packed["metallic"] == DEFAULT_METALLIC
    assert packed["alpha"] == DEFAULT_ALPHA
    assert packed["color"] == pytest.approx(
        tuple(c / 255.0 for c in DEFAULT_PART_COLOR))
    assert packed["shader"] == {"name": "m", "body": _BODY}

    # And a material with no shader packs no `shader` key at all.
    assert "shader" not in _material_to_render(Material(color=(1, 2, 3)))


# ---------------------------------------------------------------------------
# A shader still needs UVs, and it gets none for free
# ---------------------------------------------------------------------------

def test_a_shader_material_asks_the_uv_baker_for_coordinates():
    """The defect this guards is silent, which is why it is pinned here.

    `_solid_compile` bakes per-corner UVs only when `add_part` hands it a tile
    size, and `add_part` took that from `Mapping.scale_m` — which a shader
    material has none of, because there is no image to lay out. The result was
    an OBJ with zero `vt` lines, a `ShaderNodeTexCoord` `UV` output reading a
    layer that was never written, and every UV-driven graph rendering one flat
    color with nothing anywhere saying why.
    """
    from dycraft.sdk._material import SHADER_UV_SCALE_M
    from dycraft.sdk.object_builder import _texture_scale_m

    sh = Shader(name="m", body=_BODY)
    assert _texture_scale_m(Material(shader=sh)) == SHADER_UV_SCALE_M
    assert _texture_scale_m(Material(color=(1, 2, 3), shader=sh)) == (
        SHADER_UV_SCALE_M)
    # An image-backed material is still sized by its own tile size.
    textured = Material(color=(1, 2, 3),
                        texture=Texture(map_items=(("base_color", _map()),)),
                        mapping=Mapping(scale_m=0.22))
    assert _texture_scale_m(textured) == 0.22
    # **And a bare material still passes None**, which is what keeps its mesh
    # bytes — and therefore every stored object's id — unchanged.
    assert _texture_scale_m(Material(color=(1, 2, 3))) is None
    assert _texture_scale_m(None) is None


def test_the_coordinates_are_metres_so_a_graph_can_size_itself():
    """1.0, deliberately: `Scale` of 1 in the graph is one feature per meter.

    `object.md` and `animation_reference.md` tell the agent shader inputs are
    in local meters, so the number is a contract with the prompts rather than
    an arbitrary default.
    """
    from dycraft.sdk._material import SHADER_UV_SCALE_M
    assert SHADER_UV_SCALE_M == 1.0


def test_an_analytic_box_under_a_shader_gets_projected_uvs(tmp_path):
    """The other half of the same defect.

    A `box` part has no staged mesh to bake into, so the Blender parent projects
    its UVs — gated on the same absent `Mapping`. Without this a shader on a
    `make_box` part is flat for the identical reason.
    """
    import numpy as np
    from dycraft.render import blender_renderer as B
    from dycraft.sdk._material import SHADER_UV_SCALE_M

    def spec_for(material):
        return B._box_spec({}, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), material,
                           tmp_path, f"t{id(material)}")

    bare = spec_for({"color": (0.5, 0.5, 0.5), "alpha": 1.0})
    assert "uvs_npy" not in bare, "an untextured box gained UVs it never had"

    shaded = spec_for({"color": (0.5, 0.5, 0.5), "alpha": 1.0,
                       "shader": {"name": "m", "body": _BODY}})
    assert "uvs_npy" in shaded, "a shader on a box has no coordinates to read"
    uvs = np.load(shaded["uvs_npy"])
    assert uvs.shape == (36, 2)          # per corner: 12 triangles x 3
    # Meters: a 1 m box spans 1.0 in UV at this scale.
    assert uvs.max() - uvs.min() == pytest.approx(1.0 / SHADER_UV_SCALE_M)

    # An image-backed box is still sized by its own tile size, not by this.
    tiled = spec_for({"color": (0.5, 0.5, 0.5), "alpha": 1.0,
                      "mapping": {"scale_m": 0.25}})
    tiled_uvs = np.load(tiled["uvs_npy"])
    assert tiled_uvs.max() - tiled_uvs.min() == pytest.approx(4.0)


_CAD_SHADER_OBJECT = '''@shader
def oak(nodes, links, bsdf):
    tc = nodes.new("ShaderNodeTexCoord")
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 20.0
    ramp = nodes.new("ShaderNodeValToRGB")
    links.new(tc.outputs["UV"], grain.inputs["Vector"])
    links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

OAK = Material(shader=oak)
BARE = Material(color=(150, 152, 158), metallic=0.9, roughness=0.3)

obj = Object("plank")
body = obj.add_part("body", Box(0.4, 0.2, 0.03), material=OAK)
pin = obj.add_part("pin", Pos(0.15, 0, 0.03) * Cylinder(0.005, 0.02),
                   material=BARE)
obj.declare_root(body)
obj.add_anchor(body, "seat", translation=(0.15, 0.0, 0.015))
obj.add_anchor(pin, "base", translation=(0.0, 0.0, -0.01))
obj.attach_fixed("pin_fix", parent=body, child=pin,
                 parent_anchor=obj.anchor(body, "seat"),
                 child_anchor=obj.anchor(pin, "base"))
ir = obj.finalize()
'''


def test_a_cad_part_under_a_shader_stages_an_obj_with_uvs(tmp_path):
    """The composition, through the interpreter that actually runs `object.py`.

    `_texture_scale_m` returning 1.0 and `_solid_compile` baking when handed a
    scale are unit-tested apart; this is the only thing that catches the two
    ceasing to compose. The bare-material part in the same object is the
    control: it must still stage an OBJ with no `vt` at all.
    """
    from dycraft.agent.cad import object_worker as W

    if not W.is_available():
        pytest.skip("no build123d interpreter")

    (tmp_path / "object.py").write_text(_CAD_SHADER_OBJECT, encoding="utf-8")
    raw = W.run_object_source(_CAD_SHADER_OBJECT, run_dir=tmp_path)
    assert raw["ok"], raw.get("build_error") or raw.get("sandbox_violation")

    parts = {p["name"]: p for p in raw["ir"]["parts"]}
    assert set(parts["body"]["material"]) == {"shader"}
    assert set(parts["pin"]["material"]) == {"color", "metallic", "roughness",
                                             "alpha"}

    def vt_count(part):
        obj = tmp_path / part["params"]["mesh"]
        return sum(1 for line in obj.read_text().splitlines()
                   if line.startswith("vt "))

    assert vt_count(parts["body"]) > 0, (
        "the shader part staged no texture coordinates, so its graph would "
        "read an absent UV layer and render one flat colour")
    assert vt_count(parts["pin"]) == 0, (
        "the bare part gained UVs; its mesh bytes are supposed to be "
        "byte-identical to a pre-texture build")


def test_passing_the_shader_itself_says_what_to_type():
    """`@shader` binds its name to a `Shader`, so `material=marble` reads
    perfectly and is one wrapper short.

    It is the likeliest mistake in shader mode, and generic advice would send
    the author to a flat scalar, away from the representation the skill spends
    a page teaching.

    The repair names the bare `Material(shader=...)` form and nothing else,
    since `check_material` refuses scalar backups alongside a shader.
    """
    from dycraft.sdk._material import check_material

    sh = Shader(name="marble", body=_BODY)
    with pytest.raises(InvalidPrimitive) as e:
        check_material(sh, "part.material")
    assert e.value.to_dict()["suggested_repairs"] == [
        "wrap it: material=Material(shader=marble)"]


def test_the_generic_material_advice_points_only_at_the_shader_route():
    """Whatever the mode, the advice must not point at a hidden constructor.

    It must also not offer scalars as an equivalent choice. It did, and that
    was advice the `finish` material gate then refused: a visible surface
    carrying only scalars has no authored material. The shader form, plus the
    two fields a shader cannot express, is the whole authoring vocabulary.
    """
    from dycraft.sdk._material import check_material

    with pytest.raises(InvalidPrimitive) as e:
        check_material("wood", "part.material")
    repair = e.value.to_dict()["suggested_repairs"][0]
    assert "Material(shader=" in repair
    assert "Material(color=" not in repair, (
        "the generic advice offers a scalar surface the finish gate refuses")
    assert "alpha" in repair and "emission" in repair
    assert "material_from_text" not in repair, (
        "the advice names a constructor that is absent from the default "
        "overlay")


# --- shader exclusivity at the authoring boundary --------------------------


@pytest.mark.xfail(strict=True, reason=(
    "The refusal is written and deliberately not armed: 326 of 326 "
    "`Material(shader=...)` constructions in the authored corpus carry the "
    "scalars, so arming it makes replaying any of them a build failure. The "
    "rule is taught on every agent-facing surface, which is the half that "
    "stops the corpus growing. `strict` so this reports the moment "
    "`check_material`'s block is uncommented -- see it for the two conditions "
    "that release it."))
@pytest.mark.parametrize("field,value", [("color", (1, 2, 3)),
                                         ("metallic", 1.0),
                                         ("roughness", 0.4)])
def test_authoring_a_shader_with_the_scalars_it_owns_is_refused(field, value):
    """One description of a surface, not two.

    A shader owns base color, metalness and roughness, so a material carrying
    both representations leaves nothing to arbitrate them — the graph decides
    under Cycles and the scalars decide everywhere else, which is two different
    surfaces wearing one name.
    """
    from dycraft.sdk._material import check_material

    sh = Shader(name="marble", body=_BODY)
    with pytest.raises(InvalidPrimitive) as e:
        check_material(Material(shader=sh, **{field: value}), "part.material")
    ev = e.value.to_dict()["evidence"]
    assert ev["conflict"] == [field]
    assert field in e.value.to_dict()["summary"]


def test_alpha_and_emission_are_the_two_allowed_shader_companions():
    """The harness applies both outside the graph, so a shader cannot own them.

    `_apply_emission` runs in the render child and alpha drives the
    transparency path; neither is a socket the node tree writes. They are the
    reason this rule names three fields rather than "no scalars at all".
    """
    from dycraft.sdk._material import check_material

    sh = Shader(name="marble", body=_BODY)
    em = Emission(color=(255, 236, 190), strength=1900.0)
    for kwargs in ({}, {"alpha": 0.35}, {"emission": em},
                   {"alpha": 0.35, "emission": em}):
        check_material(Material(shader=sh, **kwargs), "part.material")


def test_a_scalar_material_with_no_shader_is_still_accepted():
    """The rule is about the pairing, not about scalars existing.

    Imported assets and the renderer fallback both still need them, and
    `Material` requires a color when there is no shader to compute one.
    """
    from dycraft.sdk._material import check_material

    check_material(Material(color=(1, 2, 3), metallic=0.2, roughness=0.7),
                   "part.material")


def test_the_rule_does_not_reach_legacy_ir_reads():
    """Reading a stored shader+scalar material must stay lossless.

    `material_from_ir` deliberately does not go through `check_material`, and
    `material_from_ir(d).to_ir() == d` is a promise the content hash depends
    on. Enforcing exclusivity in `Material.__post_init__` instead would have
    made the stored corpus unreadable.
    """
    sh = Shader(name="marble", body=_BODY)
    stored = Material(shader=sh, color=(9, 8, 7), metallic=1.0,
                      roughness=0.3).to_ir()
    back = material_from_ir(stored, default_color=(128, 128, 128))
    assert back.color == (9, 8, 7) and back.shader is not None
    assert back.to_ir() == stored
