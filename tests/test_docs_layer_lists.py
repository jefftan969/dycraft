"""The layers the shipped docs name are the layers `python -m dycraft.cli --layer` accepts.

The complete lists (the README's and the CLI README's layer tables, docs/architecture.md's layer table,
docs/usage.md's synopsis and section headings, and `--layer`'s own help) must name every layer and no other.
Every other table keyed by layer, and every `--layer NAME` in a shipped document, must use a real one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dycraft._layers import LAYERS
from dycraft.cli.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
SHIPPED_DOCS = [REPO / "README.md", REPO / "cli" / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _layer_tables(text: str) -> list[tuple[int, list[str]]]:
    """(line number, first-column names) for each markdown table whose first header cell is `layer` or `--layer`,
    in any case."""
    tables, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        head = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|") or head[0].lower() not in ("layer", "--layer"):
            continue
        names = []
        for row in lines[i + 2:]:
            if not row.lstrip().startswith("|"):
                break
            names.append(row.strip().strip("|").split("|")[0].strip().strip("`"))
        tables.append((i + 1, names))
    return tables


def _first_table_after(text: str, heading: str | None) -> list[str]:
    """The first layer table after `heading`, or in the whole document when `heading` is None."""
    tail = text[text.index(heading):] if heading else text
    return _layer_tables(tail)[0][1]


def test_the_cli_offers_exactly_the_layers():
    action = next(a for a in build_parser()._actions if "--layer" in a.option_strings)
    assert tuple(action.choices) == LAYERS
    for layer in LAYERS:
        assert re.search(rf"\b{layer}\b", action.help), f"--layer's help does not name {layer}"


@pytest.mark.parametrize("doc, heading", [
    # The README's first layer table, wherever its section is: the README's headings are still being reviewed.
    ("README.md", None),
    ("cli/README.md", "## Layers"),
    ("docs/architecture.md", "## Layers"),
])
def test_each_layer_table_lists_every_layer(doc, heading):
    names = _first_table_after((REPO / doc).read_text(), heading)
    assert sorted(names) == sorted(LAYERS), f"{doc}: {names}"


def test_the_usage_page_names_every_layer():
    text = (REPO / "docs" / "usage.md").read_text()
    synopsis = re.search(r"--layer <([^>]+)>", text)
    assert synopsis and sorted(synopsis.group(1).split("|")) == sorted(LAYERS), synopsis
    sections = re.findall(r"^## .*`--layer (\w+)`", text, re.M)
    assert sorted(sections) == sorted(LAYERS), sections


@pytest.mark.parametrize("doc", SHIPPED_DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_layer_a_doc_names_is_a_real_one(doc):
    text = doc.read_text()
    for line, names in _layer_tables(text):
        unknown = [n for n in names if n not in LAYERS]
        assert not unknown, f"{doc.name}:{line}: table names {unknown}, not --layer choices {LAYERS}"
    used = set(re.findall(r"--layer[ =]([A-Za-z_]\w*)", text))
    assert used <= set(LAYERS), f"{doc.name}: --layer {sorted(used - set(LAYERS))} is not a choice"
