"""The build123d namespace reference must stay true to the runtime.

The object agent cannot introspect its own globals — `dir`, `globals`, `vars`
and `getattr` are all in `sandbox._REJECTED_NAMES` — and the refusal a bad name
earns does not enumerate the alternatives. So the skill page
`build123d_authoring/references/namespace_reference.md` is the interface
contract for the injected namespace: it must list exactly
`_b123d_namespace.CURATED`, and every prompt that authors build123d must point
at it. The exact regenerate-and-diff check lives in `test_build123d_skill.py`.
"""

from __future__ import annotations

import ast
import importlib.util as _u
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / ".claude" / "skills" / "build123d_authoring" / "references"
        / "namespace_reference.md")

_spec = _u.spec_from_file_location(
    "_b123d_ns_page", REPO / "agent" / "cad" / "_b123d_namespace.py")
NS = _u.module_from_spec(_spec)
_spec.loader.exec_module(NS)



def _page_names() -> set[str]:
    """The undotted names in the page's tables (factory rows are `Class.method`)."""
    return set(re.findall(r"^\| `([A-Za-z_][A-Za-z0-9_]*)` \|", PAGE.read_text(), re.M))


def test_the_page_lists_exactly_the_injected_namespace():
    """No CAD interpreter needed, so this guard runs everywhere. A name added
    to `CURATED` without regenerating the page is a name the agent is never
    told about, and one removed is a name it will be told to use and cannot."""
    b123d = _page_names()
    assert b123d == set(NS.CURATED), (
        f"page vs CURATED: missing {sorted(set(NS.CURATED) - b123d)}, "
        f"extra {sorted(b123d - set(NS.CURATED))}")


def test_the_page_states_its_own_counts_correctly():
    """The prose count is what the agent reads first; a stale one contradicts
    the table below it."""
    assert f"**{len(NS.CURATED)}** build123d names" in PAGE.read_text()


def test_object_prompt_uses_the_staged_skill_reference():
    from dycraft.agent.agent_runner import compose_system_prompt
    prompt = compose_system_prompt("object")
    assert PAGE.read_text().strip() not in prompt
    assert "namespace_reference.md" in prompt
    assert "$DYCRAFT_RUN_DIR/.claude/skills/build123d_authoring" in prompt


def _table_rows() -> list[tuple[int, str, str]]:
    """(line number, name, signature) for every table row, pipes unescaped."""
    out = []
    for i, l in enumerate(PAGE.read_text().split("\n"), 1):
        if not l.startswith("| `"):
            continue
        m = re.match(r"^\| `([^`]+)` \| `(.*)` \|$", l)
        assert m, f"line {i}: unparseable table row: {l[:100]}"
        out.append((i, m.group(1), m.group(2).replace("\\|", "|")))
    return out


def test_every_row_is_a_two_column_markdown_row():
    """build123d writes type unions as `'Align | tuple[Align, Align]'`, and a
    raw pipe splits the row even inside a code span, so a row's only unescaped
    pipes are its three column bars."""
    for i, line in enumerate(PAGE.read_text().split("\n"), 1):
        if not line.startswith("| `"):
            continue
        bars = line.count("|") - line.count("\\|")
        assert bars == 3, f"line {i}: {bars} unescaped pipes, expected 3"


def test_no_signature_is_truncated():
    """The page is the whole interface contract, so a signature that stops
    mid-expression tells the agent nothing about the parameter it cut."""
    txt = PAGE.read_text()
    assert "\u2026" not in txt, "a signature is truncated with an ellipsis"


def test_every_signature_is_balanced_and_parses_as_python():
    """Balanced brackets and quotes, and — for anything with a parameter list —
    something Python can actually parse. This is what catches a truncation or a
    mangled default that the eye slides over."""
    pairs = {")": "(", "]": "[", "}": "{"}
    checked = 0
    for i, name, sig in _table_rows():
        depth = {"(": 0, "[": 0, "{": 0}
        for ch in sig:
            if ch in depth:
                depth[ch] += 1
            elif ch in pairs:
                depth[pairs[ch]] -= 1
        assert not any(depth.values()), (
            f"line {i} ({name}): unbalanced brackets {depth}")
        assert sig.count("'") % 2 == 0 and sig.count('"') % 2 == 0, (
            f"line {i} ({name}): unbalanced quotes")
        if "(" not in sig:
            continue          # enum members / type-alias rows
        checked += 1
        params = sig[sig.index("(") + 1:sig.rindex(")")]
        try:
            ast.parse(f"def _f({params}): pass")
        except SyntaxError as e:
            raise AssertionError(
                f"line {i} ({name}): signature is not a valid parameter "
                f"list ({e.msg}): {params[:120]}") from None
    assert checked >= 60, f"only {checked} signatures had a parameter list"


def test_no_default_is_an_unwritable_repr():
    """`inspect.signature` renders an enum default as `<Mode.ADD>`, which is
    not an expression anyone can type. `Mode.ADD` is, and `Mode` is injected,
    so `_clean` rewrites it — otherwise the page shows a default the reader
    cannot reproduce."""
    leftovers = re.findall(r"<(?!name>)[^>]{1,60}>", PAGE.read_text())
    assert not leftovers, f"unwritable reprs on the page: {sorted(set(leftovers))}"


def test_the_rows_are_unique():
    names = [n for _, n, _ in _table_rows()]
    assert len(names) == len(set(names)), "the page lists a name twice"


