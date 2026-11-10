"""Keep both skill trees identical and discoverable, and the context skill on the prompts' path."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLAUDE_TREE = REPO / ".claude" / "skills"
AGENTS_TREE = REPO / ".agents" / "skills"

#: Every skill DyCraft ships, discovered rather than hardcoded — a skill added
#: to one tree and not the other must fail, and a hardcoded list would let it
#: through unnoticed.
SKILL_NAMES = sorted(
    {p.name for p in CLAUDE_TREE.iterdir() if p.is_dir()}
    | {p.name for p in AGENTS_TREE.iterdir() if p.is_dir()}
)

CONTEXT_SKILL = "save_context"
PROMPTS = REPO / "agent" / "prompts"


def _files(tree: Path, name: str) -> dict[str, bytes]:
    root = tree / name
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_both_trees_ship_the_same_skills():
    """A skill in one tree only reaches one backend, silently."""
    claude = sorted(p.name for p in CLAUDE_TREE.iterdir() if p.is_dir())
    agents = sorted(p.name for p in AGENTS_TREE.iterdir() if p.is_dir())
    assert claude == agents == SKILL_NAMES


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_every_skill_is_byte_identical_across_trees(name):
    """Covers supporting files too, not just SKILL.md.

    A skill may carry supporting files beside `SKILL.md`, and the agent
    resolves them relative to the skill directory — so a supporting file that
    exists in one tree only is the same class of bug as a missing skill.
    """
    claude, agents = _files(CLAUDE_TREE, name), _files(AGENTS_TREE, name)
    assert sorted(claude) == sorted(agents), (
        f"{name}: file sets differ between .claude and .agents")
    differing = [rel for rel in claude if claude[rel] != agents[rel]]
    assert not differing, (
        f"{name}: byte-differing files between the two trees: {differing}. "
        f"The trees are one skill set; edit both or neither.")


@pytest.mark.parametrize("tree", [CLAUDE_TREE, AGENTS_TREE],
                         ids=["claude", "agents"])
@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_has_frontmatter_codex_requires(tree, name):
    text = (tree / name / "SKILL.md").read_text()
    assert text.startswith("---\n"), f"{name}: no YAML frontmatter"
    end = text.index("\n---\n", 3)
    front = text[4:end]
    assert re.search(r"^name:\s*\S", front, re.M), f"{name}: no name field"
    assert re.search(r"^description:\s*\S", front, re.M), (
        f"{name}: no description field")


@pytest.mark.parametrize("tree", [CLAUDE_TREE, AGENTS_TREE],
                         ids=["claude", "agents"])
def test_the_context_skill_writes_where_the_prompts_read(tree):
    """The prompts that end a run with `save_context` name the run dir's
    `context_summary.md`, and an agent reads it back from there after a
    compaction; a summary written outside the run dir is lost when the bundle
    is the only thing kept."""
    text = (tree / CONTEXT_SKILL / "SKILL.md").read_text()
    assert "./context_summary.md" in text, "default path missing"
    assert "$DYCRAFT_RUN_DIR/context_summary.md" in text, "dycraft run-dir override missing"
    users = [p for p in sorted(PROMPTS.glob("*.md")) if CONTEXT_SKILL in p.read_text()]
    assert users, "no prompt names the context skill"
    for prompt in users:
        assert "$DYCRAFT_RUN_DIR/context_summary.md" in prompt.read_text(), (
            f"{prompt.name} uses {CONTEXT_SKILL} without naming the run dir's summary")
