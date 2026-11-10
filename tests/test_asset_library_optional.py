"""The `<!-- assets -->` prompt block shows only with an installed asset library (`agent_runner.OPTIONAL["assets"]`).

animation.md sends the agent to `references/assets/INDEX.md`, which a run has only when `python -m dycraft.install
library` has put the library, with its index, where `asset_library.library_root()` finds it.
"""
from __future__ import annotations

from dycraft.agent.agent_runner import OPTIONAL, render_prompt

BLOCK = "before\n<!-- assets -->\nRead references/assets/INDEX.md.\n<!-- /assets -->\n\nafter\n"


def test_no_library_no_block(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(tmp_path / "nowhere"))
    assert OPTIONAL["assets"]() is False
    assert render_prompt(BLOCK) == "before\nafter\n"


def test_a_library_folder_without_its_index_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(tmp_path))
    assert OPTIONAL["assets"]() is False


def test_an_installed_library_shows_the_block(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_ASSET_LIBRARY", str(tmp_path))
    (tmp_path / "INDEX.md").write_text("# Assets\n")
    assert OPTIONAL["assets"]() is True
    assert render_prompt(BLOCK) == "before\nRead references/assets/INDEX.md.\n\nafter\n"
