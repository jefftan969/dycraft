"""Both CLI surfaces must resolve the complete humanoid motion guidance."""

from pathlib import Path
import re

from dycraft.agent import stage_skills


ROOT = Path(__file__).resolve().parents[1]


def test_both_cli_copies_are_identical():
    trees = [ROOT / namespace / "skills/humanoid_motion"
             for namespace in (".agents", ".claude")]
    contents = [{p.relative_to(tree): p.read_bytes()
                 for p in tree.rglob("*") if p.is_file()} for tree in trees]
    assert Path("SKILL.md") in contents[0]
    assert contents[0] == contents[1]


def test_staged_prompt_skill_and_local_references_resolve(tmp_path):
    """The animation prompt names the skill, and the staged copy under each
    CLI namespace resolves its local references."""
    stage_skills(tmp_path)
    prompt = (ROOT / "agent/prompts/animation.md").read_text()
    assert "`humanoid_motion` skill" in prompt
    for namespace in (".agents", ".claude"):
        relative = f"{namespace}/skills/humanoid_motion/SKILL.md"
        entry = tmp_path / relative
        assert entry.is_file(), relative
        links = re.findall(r"\]\((references/[^)]+)\)", entry.read_text())
        assert set(links) == {"references/in_depth.md"}
        for link in links:
            assert (entry.parent / link).is_file()
