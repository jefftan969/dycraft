"""Skill pages must come from the checkout under test, not from the bench.

The object prompt tells the agent to read `.claude/skills/<name>/SKILL.md`, a
path relative to its cwd — and its cwd is the run dir. Without a skills tree
there, the agent resolves it the only way it can: `--runs-dir` defaults to
`<repo>/runs`, so its own `$DYCRAFT_RUN_DIR` names the checkout, and it walks
back up to read the Skill out of the main working tree.

That is invisible in a normal run and wrong for a run of another revision.
Skills are tracked, so they are part of the revision, and such a run would read
the main working tree's copy instead of its own revision's.

So the property under test is not "a file was copied". It is that the copy comes
from the tree `dycraft` was *imported* from, and that the path the prompt prints
is the path that then exists.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dycraft import agent as A
from dycraft.agent import LAYERS, stage_harness_client
from dycraft.agent.agent_runner import compose_footer

REPO = Path(__file__).resolve().parents[1]
PROMPT = REPO / "agent" / "prompts" / "object.md"


def _fake_checkout(root: Path, marker: str) -> Path:
    """A minimal stand-in for a revision worktree carrying one Skill page."""
    for tool_dir in (".claude", ".agents"):
        page = root / tool_dir / "skills" / "build123d_authoring" / "SKILL.md"
        page.parent.mkdir(parents=True)
        page.write_text(marker)
    return root


def test_stage_skills_copies_both_tool_trees(tmp_path):
    """claude reads `.claude/skills`, codex reads `.agents/skills`; a matrix
    runs both backends, so staging one of them would silently blind the other."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    staged = A.stage_skills(run_dir)

    assert {p.parent.name for p in staged} == {".claude", ".agents"}
    for tool_dir in (".claude", ".agents"):
        page = run_dir / tool_dir / "skills" / "build123d_authoring" / "SKILL.md"
        assert page.is_file(), f"{tool_dir}/skills was not staged"
        assert page.read_text() == (
            REPO / tool_dir / "skills" / "build123d_authoring" / "SKILL.md"
        ).read_text()


def test_the_copy_comes_from_the_checkout_dycraft_was_imported_from(
        tmp_path, monkeypatch):
    """The whole point. An ablation cell imports `dycraft` through a pkgroot
    symlink into a git worktree, so `_REPO_ROOT` is that worktree — and the
    Skill it stages has to be the worktree's, not the bench's."""
    worktree = _fake_checkout(tmp_path / "wt", "PAGE FROM THE WORKTREE")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(A, "_REPO_ROOT", worktree)

    A.stage_skills(run_dir)

    staged = (run_dir / ".claude" / "skills" / "build123d_authoring"
              / "SKILL.md").read_text()
    assert staged == "PAGE FROM THE WORKTREE"
    live = (REPO / ".claude" / "skills" / "build123d_authoring"
            / "SKILL.md").read_text()
    assert staged != live, "the working tree's copy leaked into the run dir"


def test_the_path_the_prompt_prints_is_the_path_that_exists(tmp_path):
    """A broken relative path is as bad as a missing file, so assert on the
    prompt's own spelling of it rather than on a path this test invents."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    A.stage_skills(run_dir)

    printed = re.findall(r"\$DYCRAFT_RUN_DIR/(\S+/SKILL\.md)", PROMPT.read_text())
    assert printed, ("object.md no longer prints a $DYCRAFT_RUN_DIR-rooted "
                     "Skill path — if it went back to a bare relative path, the "
                     "agent will resolve it against the checkout again")
    for rel in printed:
        assert (run_dir / rel).is_file(), (
            f"the prompt tells the agent to read $DYCRAFT_RUN_DIR/{rel}, "
            f"which staging did not create")


def test_restaging_refreshes_skills_without_touching_cli_state(tmp_path):
    """`.claude/` is also where the agent CLI keeps its own per-run state, and a
    run dir gets re-staged on replay. Blowing away the parent would take that
    with it; leaving the tree alone would keep a previous sweep's Skill."""
    run_dir = tmp_path / "run"
    (run_dir / ".claude").mkdir(parents=True)
    (run_dir / ".claude" / ".cc-writes").write_text("cli state")
    stale = run_dir / ".claude" / "skills" / "withdrawn_skill" / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier sweep")

    A.stage_skills(run_dir)

    assert (run_dir / ".claude" / ".cc-writes").read_text() == "cli state"
    assert not stale.exists(), "a withdrawn Skill survived a re-stage"
    assert (run_dir / ".claude" / "skills" / "build123d_authoring"
            / "SKILL.md").is_file()


def test_a_revision_with_no_skill_tree_is_reported_not_fatal(
        tmp_path, monkeypatch, capsys):
    """A commit predating the Skills is still a revision a run may use.
    Refusing to launch it would lose the run; staging nothing quietly would
    leave the prompt pointing at a path that does not exist with no trace of
    why."""
    monkeypatch.setattr(A, "_REPO_ROOT", tmp_path / "bare")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert A.stage_skills(run_dir) == []
    assert "no Skill tree" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `stage_harness_client` is the call site, not `cli.py`, so every layer —
# including a nested `sub_runs/<layer>` workspace, a `run_oneoff` reviewer and
# a dispatched child — gets a tree. A copy, not a link: the source can be a
# temporary worktree that is deleted afterwards, and a link would leave every
# archived bundle pointing at nothing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", sorted(LAYERS))
def test_a_nested_workspace_gets_its_own_copy_of_every_skill(tmp_path, layer):
    run = tmp_path / "parent" / "sub_runs" / layer
    run.mkdir(parents=True)
    stage_harness_client(run, layer)
    for namespace in (".claude", ".agents"):
        skills = run / namespace / "skills"
        assert skills.is_dir() and not skills.is_symlink(), (
            "staged as a link — an archived bundle would point at a worktree "
            "that no longer exists")
        assert (skills / "save_context" / "SKILL.md").read_text()
        # `material_selection` is deliberately absent: it prescribed material
        # recipes and numeric anchors, and both lines deleted it.
        assert not (skills / "material_selection").exists()
    if layer != "plain":
        footer = compose_footer(layer, "test", run)
        assert "$DYCRAFT_RUN_DIR/.claude/skills/" in footer
        assert "$DYCRAFT_RUN_DIR/.agents/skills/" in footer


def test_restaging_leaves_the_cli_s_own_settings_alone(tmp_path):
    """The narrower half of the shader-viewer line's restaging test, which is
    compatible with refreshing: `settings.json` sits BESIDE `skills/`, and
    `stage_skills` scopes its rmtree to `skills/` precisely so per-run CLI
    configuration survives a replay. A user file placed *inside* `skills/` does
    not survive, and must not — that is what refreshing means."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"existing": true}')

    A.stage_skills(tmp_path)
    A.stage_skills(tmp_path)

    assert settings.read_text() == '{"existing": true}'
    assert (tmp_path / ".agents" / "skills").is_dir()
