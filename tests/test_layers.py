"""Layer names: one list, the aliases recorded runs need, and the prompt each layer is given."""
from __future__ import annotations

import pytest

from dycraft._layers import LAYER_ALIASES, LAYERS, canonical_layer


@pytest.mark.parametrize("recorded,current", [
    ("unified", "scene"), ("unified_animation", "animation"),
    ("scene", "scene"), ("animation", "animation"), ("object", "object"), ("plain", "plain"),
    ("bare_agent", "bare_agent"), (None, None),
])
def test_a_recorded_layer_name_maps_to_its_current_name(recorded, current):
    """Recorded runs may name a layer `unified`/`unified_animation`; readers
    see the current names, and anything else passes through unchanged."""
    assert canonical_layer(recorded) == current


def test_every_layer_list_agrees():
    """`--layer`, the staged harness clients, the prompts, and the daemon each
    name the layers; a layer missing from one fails at launch, not here."""
    from dycraft.agent import LAYERS as SPECS
    from dycraft.agent.agent_runner import ROLE_PROMPTS
    from dycraft.agent.layers.server import LAYER_NAMES as DAEMON_LAYERS

    assert set(SPECS) == set(LAYERS) == set(ROLE_PROMPTS)
    assert set(DAEMON_LAYERS) == set(LAYERS) - {"plain"}
    assert not set(LAYER_ALIASES) & set(LAYERS)


def test_the_launcher_accepts_exactly_the_current_names():
    from dycraft.cli import cli as C

    with pytest.raises(SystemExit):
        C.main(["--backend", "codex", "--layer", "unified_animation", "--prompt", "x"])


@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer_has_its_prompt(layer):
    from dycraft.agent.agent_runner import compose_system_prompt

    assert compose_system_prompt(layer).strip()


def test_the_rewriters_read_the_animation_authors_sdk_reference_without_its_asset_section():
    """The rewriters write for the animation author, so they read the reference that
    author is given (`animation_reference.md`), not the scene agent's prompt. Its
    asset-library section is dropped: the rewriters get the library from
    `library_block`."""
    from dycraft.agent import scene_rewrite as sr

    whole = sr.SOURCE_REFERENCE.read_text()
    assert sr.SOURCE_REFERENCE.name == "animation_reference.md"
    assert whole.count("\n## Asset library\n") == 1
    reference = sr.source_reference()
    assert "## Asset library" not in reference
    assert "## Source-style constraints" in reference and "## File skeleton" in reference
    assert [name for name, _, _ in sr.REFERENCES] == ["animation.md", "source_reference.md"]


def test_help_shows_what_an_operator_sets_and_hides_the_internal_flags(capsys):
    from dycraft.cli.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    shown = capsys.readouterr().out
    for flag in ("--backend", "--layer", "--prompt", "--model", "--reasoning-effort", "--max-hours",
                 "--advisory-hours", "--runs-dir", "--run-id", "--exp-name", "--preload-asset",
                 "--scene-brief"):
        assert flag in shown, flag
    for flag in ("--contribution-role", "--parent-run"):
        assert flag not in shown, flag
    args = build_parser().parse_args(["--backend", "codex3", "--layer", "plain", "--prompt", "x",
                                      "--contribution-role", "motion", "--parent-run", "p",
                                      "--exp-name", "e"])
    assert (args.backend, args.layer, args.contribution_role) == ("codex3", "plain", "motion")


def test_the_launcher_accepts_every_argv_it_builds_for_its_own_children(tmp_path):
    """A flag removed from the CLI but still forwarded to children makes every
    review or dispatch exit 2 at argument parsing, before its run exists."""
    from types import SimpleNamespace

    from dycraft.agent.layers._object_dispatch import _child_argv
    from dycraft.agent.oneoff import _build_argv
    from dycraft.cli.cli import build_parser

    oneoff = _build_argv(backend="codex2", prompt="p", sub_runs_dir=tmp_path, sub_run_name="review",
                         model="m", reasoning_effort="high", parent_run_id="parent",
                         python_executable="python")
    ctx = SimpleNamespace(run_dir=tmp_path, run_id="parent", layer_cfg=None)
    child = _child_argv(ctx, name="chair", prompt_path=tmp_path / "brief.md", sub_runs=tmp_path,
                        advisory_hours=1.5)
    for argv in (oneoff, child):
        assert argv[1:3] == ["-m", "dycraft.cli"]
        build_parser().parse_args(argv[3:])
