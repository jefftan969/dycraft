"""The motion rewriter's request, and how `dycraft.cli` stages its result.

The request reaches `codex` as one argv element, exactly as the scene
rewriter's does (`test_scene_rewrite_argv_cap.py`), and it carries one thing
that one does not: the scene entry, inlined. So the size test runs every brief
of the corpus against a scene entry at least as long as the longest written
for these briefs, with the G1 summary on the eight briefs that are handed it.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.agent import motion_rewrite as mr
from dycraft.agent import scene_rewrite as sr

MAX_ARG_STRLEN = 131_072
BUDGET = MAX_ARG_STRLEN // 2

PROMPTS = Path(mr.__file__).resolve().parents[1] / "agent" / "prompts"
G1_SPECS = ["g1=assets/imported/g1_with_dex1_v2"]


def _briefs() -> dict[str, str]:
    text = (PROMPTS / "animation_24_briefs.md").read_text()
    return {m.group(1): m.group(2).strip() for m in re.finditer(
        r"^### (B\d\d)\.[^\n]*\n+(.+?)(?=\n#|\Z)", text, re.M | re.S)}


#: At least as long as the longest scene entry written for these briefs (about
#: 5.8 kB), so the size test below covers the worst case for every brief.
SCENE_BYTES = 6_000


def _scene_entry(n_bytes: int = SCENE_BYTES) -> str:
    """A scene entry of `n_bytes` or a little more: paragraphs of prose, the way
    the scene rewriter writes one."""
    sentence = ("The workbench stands against the north wall at hip height, with the "
                "parts tray to its left and the finished pieces on a shelf above it. ")
    paragraphs, text = [], ""
    while len(text.encode()) < n_bytes:
        paragraphs.append(sentence * 4)
        text = "\n\n".join(p.strip() for p in paragraphs)
    return text


def test_the_briefs_parse():
    assert len(_briefs()) == 24 and len(_scene_entry().encode()) >= SCENE_BYTES


@pytest.mark.parametrize("bid", sorted(_briefs()))
def test_prompt_fits_one_argv_element(bid):
    assets = sr.asset_summary(G1_SPECS if bid <= "B08" else [])
    prompt = mr.build_prompt(_briefs()[bid], _scene_entry(), assets)
    size = len((prompt + sr.SUBMIT_INSTRUCTION
                + sr.deadline_note(mr.DEFAULT_TIMEOUT_S)).encode())
    assert size <= BUDGET, (
        f"{bid}: motion-rewrite prompt is {size:,} B, over the {BUDGET:,} B "
        f"budget. Stage what grew as a file, as the references are.")


def test_references_are_named_not_inlined():
    prompt = mr.build_prompt(_briefs()["B01"], _scene_entry(),
                             sr.asset_summary(G1_SPECS))
    for name, read, _ in sr.REFERENCES:
        body = read()
        assert f"`{name}`" in prompt
        assert body[len(body) // 2:len(body) // 2 + 200] not in prompt


def test_scene_entry_is_inlined_after_the_brief():
    brief, scene = _briefs()["B09"], _scene_entry()
    prompt = mr.build_prompt(brief, scene)
    assert prompt.index(brief) < prompt.index("# The scene entry") \
        < prompt.index(scene)


@pytest.mark.parametrize("scene", [None, "", "  \n"])
def test_absent_scene_entry_is_said_not_left_empty(scene):
    """`--no-scene-brief` still gets a motion entry, from the brief alone."""
    prompt = mr.build_prompt(_briefs()["B09"], scene)
    body = prompt.split("# The scene entry", 1)[1]
    assert body.strip().startswith(mr.NO_SCENE)


def test_pointer_names_its_file_and_not_the_scene():
    """The pointer is appended with or without `scene.txt`."""
    assert f"`{mr.STAGED_NAME}`" in mr.POINTER
    assert "scene.txt" not in mr.POINTER


def _cli(*extra):
    """Up to argument handling only; both rewrites off so nothing goes live."""
    return subprocess.run(
        [sys.executable, "-m", "dycraft.cli", "--layer", "animation",
         "--backend", "codex", "--no-scene-brief", *extra, "--prompt", "x"],
        capture_output=True, text=True, timeout=120)


def test_missing_motion_brief_file_is_refused_before_anything_starts(tmp_path):
    proc = _cli("--runs-dir", str(tmp_path), "--run-id", "r",
                "--motion-brief", str(tmp_path / "nope.txt"))
    assert proc.returncode != 0
    assert "--motion-brief: no such file" in proc.stderr
    assert not (tmp_path / "r" / mr.STAGED_NAME).exists()


# ---------------------------------------------------------------------------
# Sizing to a time limit given on this launch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hours,seconds", [(2, 18), (8, 72), (24, 150), (100, 150), (0.05, 1)])
def test_the_limit_sizes_the_film_and_caps_it(hours, seconds):
    from dycraft.agent import motion_rewrite as mr

    block = mr.time_limit_block(hours)
    assert block.startswith("\n\n# The time limit\n\n")
    assert f"This run stops after {hours:g} hours" in block
    assert f"keep the total to about {seconds} seconds of film or less" in block


@pytest.mark.parametrize("hours", [None, 0, -1])
def test_without_a_limit_the_request_says_nothing_about_one(hours):
    from dycraft.agent import motion_rewrite as mr

    assert mr.time_limit_block(hours) == ""
    assert mr.build_prompt("A dog fetches a stick.", None, None, hours) == mr.build_prompt(
        "A dog fetches a stick.", None)


def test_the_limit_reaches_the_rewriters_request(tmp_path, monkeypatch):
    from dycraft.agent import motion_rewrite as mr
    from dycraft.agent import oneoff

    monkeypatch.setattr(oneoff, "run_oneoff", lambda **kw: {"ok": True, "text": "entry"})
    mr.rewrite("A dog fetches a stick.", scene=None, backend="claude", run_dir=tmp_path,
               time_limit_hours=8)
    request = (tmp_path / "motion_rewrite_prompt.md").read_text()
    assert request.index("# The brief") < request.index("# The time limit") < request.index("# The scene entry")
    assert "about 72 seconds of film or less" in request


@pytest.mark.parametrize("flag,expected", [([], None), (["--max-hours", "0"], None),
                                           (["--max-hours", "8"], 8.0)])
def test_only_a_limit_given_on_this_launch_sizes_the_motion_entry(tmp_path, monkeypatch, isolated_environ,
                                                                  flag, expected):
    """The default limit is a safety stop, not a target, so a launch without
    `--max-hours` (or with 0) is not sized."""
    from dycraft.agent import motion_rewrite as mr
    from dycraft.cli import cli as C

    seen = {}

    def stop(brief, **kw):
        seen.update(kw)
        raise RuntimeError("stopped here")

    monkeypatch.setattr(mr, "rewrite", stop)
    with pytest.raises(SystemExit):
        C.main(["--backend", "claude", "--layer", "animation", "--prompt", "A dog fetches a stick.",
                "--no-scene-brief", "--on-rewrite-failure", "abort", "--runs-dir", str(tmp_path), *flag])
    assert seen["time_limit_hours"] == expected
