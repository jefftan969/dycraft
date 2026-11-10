"""The scene rewriter's prompt must fit in one argv element.

The prompt reaches `codex` as a single positional argument -- `cli/backends/codex.py`
does `cmd.append(plan.system_prompt_text + "\n\n" + plan.footer_text)`, and
`codex [OPTIONS] [PROMPT]` has no file or stdin form for the TUI -- and Linux
caps one argv element at MAX_ARG_STRLEN = 32 pages = 131,072 bytes. The two
references together come close to the cap, so `build_prompt` names them (they are
staged in the sub-agent's own directory) instead of inlining them. Exceeding
the cap fails silently and deterministically: `execve` returns E2BIG, argparse
exits 2, and the sub-run never exists. It is reachable by editing a prompt file
in another directory, and both references are edited regularly.
"""
import re
from pathlib import Path

import pytest

from dycraft.agent import scene_rewrite as sr
from dycraft.agent.agent_runner import render_prompt
from dycraft.agent.oneoff import SUB_RUNS_DIR

#: Linux `MAX_ARG_STRLEN`: 32 pages, the cap on ONE argv element. Not
#: `ARG_MAX`, which bounds the whole vector and is far larger -- the failure
#: here was a single oversized argument.
MAX_ARG_STRLEN = 131_072

#: What the prompt is allowed to take of that. The prompt is not the whole
#: argument: `cli/backends/codex.py` prepends the composed system prompt and wraps
#: the task in `compose_footer`'s harness preamble. Half the cap leaves room
#: for both to grow and still fails loudly long before execve would.
BUDGET = MAX_ARG_STRLEN // 2

CORPUS = Path(sr.__file__).resolve().parents[1] / "agent" / "prompts" / "animation_24_briefs.md"

#: B01-B08 are handed the G1; B09-B24 are handed nothing. The asset summary is
#: the 8 KB that pushed the G1 section over, so a test that omitted it would
#: pass on exactly the briefs that failed.
G1_SPECS = ["g1=assets/imported/g1_with_dex1_v2"]


def _briefs() -> dict[str, str]:
    """`{"B01": "A humanoid cooks...", ...}` from the corpus document."""
    text = CORPUS.read_text()
    out = {}
    for m in re.finditer(r"^### (B\d\d)\.[^\n]*\n+(.+?)(?=\n#|\Z)",
                         text, re.M | re.S):
        out[m.group(1)] = m.group(2).strip()
    return out


def test_corpus_parses():
    briefs = _briefs()
    assert len(briefs) == 24, sorted(briefs)
    assert briefs["B01"].startswith("A Unitree G1 humanoid robot cooks")


@pytest.mark.parametrize("bid", sorted(_briefs()))
def test_prompt_fits_one_argv_element(bid):
    brief = _briefs()[bid]
    assets = sr.asset_summary(G1_SPECS if bid <= "B08" else [])
    # Everything `rewrite()` concatenates, in its order — the deadline note
    # included, since that is appended per call and reaches codex in the same
    # argv element as the rest.
    size = len((sr.build_prompt(brief, assets) + sr.SUBMIT_INSTRUCTION
                + sr.deadline_note(sr.DEFAULT_TIMEOUT_S)).encode())
    assert size <= BUDGET, (
        f"{bid}: scene-rewrite prompt is {size:,} B, over the {BUDGET:,} B "
        f"budget ({MAX_ARG_STRLEN:,} B argv cap / 2). Something was inlined "
        f"into build_prompt that should be staged as a file — see "
        f"stage_references().")


def test_big_references_are_not_inlined():
    """Both references are named, neither pasted in."""
    prompt = sr.build_prompt(_briefs()["B01"], sr.asset_summary(G1_SPECS))
    for name, read, _ in sr.REFERENCES:
        body = read()
        assert f"`{name}`" in prompt, f"{name} is not named in the prompt"
        # A distinctive 200-char slice from the middle: cheaper than comparing
        # whole bodies and immune to a stray shared heading matching.
        probe = body[len(body) // 2:len(body) // 2 + 200]
        assert probe not in prompt, (
            f"{name} appears to be inlined again — that is what made every "
            f"G1 brief exceed the argv cap")


def test_stage_references_writes_both_into_the_sub_agents_workspace(tmp_path):
    """`run_oneoff` gives codex `<run>/sub_runs/scene_rewrite/` as its cwd.

    A relative path in the prompt resolves against that and nothing else, so
    the files have to be there rather than in the run directory above it. The
    contract is copied as the author's system prompt has it (`render_prompt`).
    """
    workspace = tmp_path / SUB_RUNS_DIR / "scene_rewrite"
    written = sr.stage_references(workspace)
    assert written == [name for name, _, _ in sr.REFERENCES]
    for name, read, _ in sr.REFERENCES:
        assert (workspace / name).read_text() == render_prompt(read())


def test_staging_is_idempotent(tmp_path):
    """`rewrite()` may run against a directory a previous attempt left."""
    workspace = tmp_path / SUB_RUNS_DIR / "scene_rewrite"
    sr.stage_references(workspace)
    (workspace / "leftover.txt").write_text("from an earlier attempt")
    sr.stage_references(workspace)
    assert (workspace / "leftover.txt").is_file()   # additive, never a wipe
    for name, read, _ in sr.REFERENCES:
        assert (workspace / name).read_text() == render_prompt(read())
