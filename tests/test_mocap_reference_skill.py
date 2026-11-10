"""The motion-capture reference: skill, the libraries' notes, prompt pointer, and the Blender sandbox mount.

The library itself (`assets/mocap/cmu/`, ~3 GB of CMU BVH) is gitignored and lives
only on boxes that fetched it; tests that need it skip without it, and tests about
what the agent is told pin its presence. The library is found under
`assets/mocap` in the user-state root (`dycraft._home.home()`), so those tests
point `DYCRAFT_HOME` at a fake checkout.
"""
from pathlib import Path
import re

import pytest

from dycraft.agent import stage_references, stage_skills
from dycraft.animation.blender import runtime
from dycraft.animation.mocap import _fetch
from dycraft.render._blender_env import blender_python

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")

ROOT = Path(__file__).resolve().parents[1]
SKILL = "mocap_reference"
LIBRARY = ROOT / "assets" / "mocap" / "cmu"
NOTES = {name: _fetch.LIBRARIES / name / "NOTES.md" for name in _fetch.SHIPPED}
#: The sections every library's NOTES.md has, and the fields of its Facts, as the skill describes them.
SECTIONS = ("What it holds", "Facts", "Index", "Helper")
FACTS = ("Clips", "Format", "Frame rate", "Up axis", "Units", "Rest frame", "Skeleton")


def test_both_cli_copies_are_identical():
    trees = [ROOT / ns / "skills" / SKILL for ns in (".agents", ".claude")]
    contents = [{p.relative_to(t): p.read_bytes() for p in t.rglob("*") if p.is_file()}
                for t in trees]
    assert Path("SKILL.md") in contents[0]
    assert contents[0] == contents[1]


def _fetched(repo):
    """One capture in the fetched layout, enough for `runtime.mocap_libraries` to count `cmu/` as a library."""
    clip = repo / "assets" / "mocap" / "cmu" / "data" / "08" / "08_01.bvh"
    clip.parent.mkdir(parents=True)
    clip.write_text("HIERARCHY\n")


def _library(monkeypatch, tmp_path, present):
    repo = tmp_path / ("repo" if present else "empty_repo")
    if present:
        _fetched(repo)
    monkeypatch.setenv("DYCRAFT_HOME", str(repo))


def _links(path):
    return set(re.findall(r"\]\(([^)#:]+)\)", path.read_text()))


def test_staged_skill_links_resolve(tmp_path, monkeypatch):
    _library(monkeypatch, tmp_path, present=True)
    stage_skills(tmp_path)
    for ns in (".agents", ".claude"):
        entry = tmp_path / ns / "skills" / SKILL / "SKILL.md"
        assert _links(entry)
        for link in _links(entry):
            assert (entry.parent / link).is_file(), link


def test_each_librarys_notes_link_only_to_files_installed_beside_them():
    for name, notes in NOTES.items():
        for link in _links(notes):
            assert link in _fetch.SHIPPED[name] and (notes.parent / link).is_file(), (name, link)


def test_every_library_follows_the_layout_the_skill_describes():
    """A library dropped in with this layout needs no edit to the skill: each shipped library's notes have the
    sections and Facts fields the skill names, and the files the skill's layout lists are the ones installed."""
    skill = (ROOT / ".claude/skills" / SKILL / "SKILL.md").read_text()
    for section in SECTIONS:
        assert f"- **{section}:**" in skill
    for f in {"index.txt", *(f for files in _fetch.SHIPPED.values() for f in files)}:
        assert f"  {f} " in skill, f
    for name, notes in NOTES.items():
        text = notes.read_text()
        assert re.findall(r"^## ([^:\n]+)", text, re.M)[:len(SECTIONS)] == list(SECTIONS), name
        facts = text.split("## Facts", 1)[1].split("\n## ", 1)[0]
        assert re.findall(r"^- \*\*([^*:]+):\*\*", facts, re.M) == list(FACTS), name


def test_the_dog_index_has_the_line_format_every_library_shares():
    """`<clip> TAB <start>-<end>s TAB <label> [TAB <notes>]`, the format CMU's generated index has too."""
    for line in (_fetch.LIBRARIES / "mann_dog" / "index.txt").read_text().splitlines():
        fields = line.split("\t")
        assert len(fields) in (3, 4) and re.fullmatch(r"\s*\d+\.\d-\s*\d+\.\ds", fields[1]), line


def test_the_skill_and_prompts_name_the_sandbox_mount():
    assert runtime.MOCAP_MOUNT in (ROOT / ".claude/skills" / SKILL / "SKILL.md").read_text()
    assert runtime.MOCAP_MOUNT + "/cmu/data/" in NOTES["cmu"].read_text()
    assert runtime.MOCAP_MOUNT + "/mann_dog/" in NOTES["mann_dog"].read_text()
    for prompt in ("animation.md",):
        body = (ROOT / "agent/prompts" / prompt).read_text()
        assert SKILL in body and runtime.MOCAP_MOUNT in body


def _argv(tmp_path):
    ws, out = tmp_path / "ws", tmp_path / "out"
    ws.mkdir(), out.mkdir()
    return runtime.command(ws, out, None, serve=True)


def test_the_sandbox_mounts_the_library_read_only_when_present(tmp_path, monkeypatch):
    if blender_python() is None and runtime.blender_binary() is None:
        pytest.skip("needs a configured Blender")
    repo = tmp_path / "repo"
    _fetched(repo)
    monkeypatch.setenv("DYCRAFT_HOME", str(repo))
    argv = _argv(tmp_path)
    i = argv.index(runtime.MOCAP_MOUNT)
    assert argv[i - 2:i] == ["--ro-bind", str((repo / "assets" / "mocap").resolve())]


def test_the_skill_offers_the_retarget_helper_without_requiring_it():
    """The helper lives in the sandbox's /tools, not in the skill tree; the CMU notes teach it, and the
    skill leaves the decision -- helper, own method or reference only -- to the author."""
    root = ROOT / ".claude/skills" / SKILL
    assert not list(root.rglob("*.py"))
    notes = NOTES["cmu"].read_text()
    assert "import mocap_retarget" in notes and "mr.retarget(" in notes
    text = (root / "SKILL.md").read_text()
    assert "your decision" in text and "Doing it yourself" in text


@pytest.mark.parametrize("present", [True, False])
def test_the_sandbox_binds_the_retarget_helpers_with_or_without_the_library(tmp_path, monkeypatch, present):
    """The helpers are bound whether or not the library is present: `mocap_retarget.bone_map`
    maps a rig's bones by role with no capture at all."""
    if blender_python() is None and runtime.blender_binary() is None:
        pytest.skip("needs a configured Blender")
    _library(monkeypatch, tmp_path, present)
    argv = _argv(tmp_path)
    for name in ("mocap_retarget", "dog_retarget"):
        i = argv.index(f"/tools/{name}.py")
        assert argv[i - 2] == "--ro-bind" and argv[i - 1].endswith(f"animation/mocap/{name}.py")


def test_editing_a_helper_leaves_cached_verifications_valid():
    """`implementation_fingerprint` keys every cached verification, so the helpers live
    outside `animation/blender/`: editing one must not invalidate the certificates of runs
    that never import it."""
    from dycraft.animation.blender.fingerprint import implementation_fingerprint
    fingerprint = implementation_fingerprint()
    assert any(path.startswith("animation/blender/") for path in fingerprint), fingerprint
    assert not [path for path in fingerprint if path.startswith("animation/mocap/")]
    assert {p.name for p in runtime.HELPERS.glob("[!_]*.py")} == {"mocap_retarget.py", "dog_retarget.py"}


def _what_the_agent_reads(tmp_path):
    from dycraft.agent import scene_rewrite
    from dycraft.agent.agent_runner import compose_system_prompt
    run = tmp_path / "run"
    run.mkdir()
    stage_skills(run)
    scene_rewrite.stage_references(run / "rewrite")
    prompts = [compose_system_prompt("animation")]
    prompts.append((run / "rewrite" / "animation.md").read_text())
    return run, prompts


def test_without_the_library_the_agent_hears_nothing_of_it(tmp_path, monkeypatch):
    """A box without `assets/mocap` gets no mocap prompt text and no `mocap_reference`
    Skill; on one with it, the prompt carries its mocap paragraph."""
    _library(monkeypatch, tmp_path, present=False)
    run, prompts = _what_the_agent_reads(tmp_path)
    for text in prompts:
        assert "mocap" not in text and "dog_retarget" not in text and "<!--" not in text
        assert "humanoid_motion" in text and "\n\n\n" not in text
    for ns in (".agents", ".claude"):
        assert not (run / ns / "skills" / SKILL).exists()
        assert (run / ns / "skills" / "humanoid_motion" / "SKILL.md").is_file()


def test_with_the_library_the_prompts_keep_its_blocks_without_markers(tmp_path, monkeypatch):
    _library(monkeypatch, tmp_path, present=True)
    run, prompts = _what_the_agent_reads(tmp_path)
    for text in prompts:
        assert SKILL in text and runtime.MOCAP_MOUNT in text and "mocap_retarget" in text
        assert "<!--" not in text and "\n\n\n" not in text
    for ns in (".agents", ".claude"):
        assert (run / ns / "skills" / SKILL / "SKILL.md").is_file()


def test_runs_get_a_shell_visible_link_to_the_library(tmp_path, monkeypatch):
    """A run links the library into its own `references/mocap`, because the agent's shell cannot see
    the Blender mount and would take the library for not installed."""
    import dycraft.agent as agent
    repo = tmp_path / "repo"
    _fetched(repo)
    monkeypatch.setenv("DYCRAFT_HOME", str(repo))
    monkeypatch.delenv("DYCRAFT_ASSET_LIBRARY", raising=False)
    run = tmp_path / "run"
    run.mkdir()
    assert stage_references(run) == [run / "references" / "mocap"]
    link = run / "references" / "mocap"
    assert link.is_symlink() and (link / "cmu").is_dir()
    assert stage_references(run) == [link]          # idempotent on re-stage
    assert "`references/mocap/` in your run directory" in (ROOT / ".claude/skills" / SKILL / "SKILL.md").read_text()
    notes = NOTES["cmu"].read_text()
    assert "references/mocap/cmu/data/" in notes and "only in the Blender sandbox" in notes


def test_the_joint_maps_say_the_toes_are_not_mapped_automatically():
    """The joint table lists the toes beside the rows `mr.bone_map` builds, but `bone_map` has no toe
    names, so a retarget leaves the toes at rest unless the author's own mapping names them; the
    table says so, and the test fails if `bone_map` ever learns toes and the sentence goes stale."""
    import ast
    tree = ast.parse((ROOT / "animation/mocap/mocap_retarget.py").read_text())
    candidates = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                      and any(getattr(t, "id", None) == "CANDIDATES" for t in n.targets))
    assert not any("Toe" in k.value for k in candidates.keys)
    text = (_fetch.LIBRARIES / "cmu" / "JOINT_MAPS.md").read_text()
    assert "`mr.bone_map` finds every row but `LeftToeBase`/`RightToeBase`" in text
