"""Claude Code's settings files in a run directory cannot give an agent hooks, settings or permission rules.

A dispatched child starts in a directory inside its parent's run directory, which the parent can write, and Claude
Code reads `.claude/settings.json` from its working directory; its hooks run outside the sandbox. DyCraft's
`--settings` therefore switch hooks off and deny the file tools any `.claude/` in the run directory.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dycraft.cli.backends import claude_agent_settings


def test_hooks_are_off_and_no_file_tool_may_write_a_claude_directory():
    s = claude_agent_settings("/runs/r1")
    assert s["disableAllHooks"] is True
    assert s["permissions"]["deny"][:2] == ["Edit(///runs/r1/.claude/**)", "Edit(///runs/r1/**/.claude/**)"]
    assert "Edit(///runs/r1/**)" in s["permissions"]["allow"]       # the rest of the run directory stays editable


def test_a_launch_s_own_deny_rules_are_kept_alongside():
    s = claude_agent_settings("/runs/r1", deny=["Read(//x/**)"])
    assert s["permissions"]["deny"][-1] == "Read(//x/**)" and len(s["permissions"]["deny"]) == 3


def test_without_the_sandbox_only_the_sandbox_changes():
    """`--allow-unsandboxed` where Claude Code's sandbox cannot start: Bash runs unconfined, and hooks, the edit
    rules and the denied `.claude/` stay as they were."""
    on, off = claude_agent_settings("/runs/r1"), claude_agent_settings("/runs/r1", sandboxed=False)
    assert on["sandbox"]["enabled"] is True and on["sandbox"]["failIfUnavailable"] is True
    assert off["sandbox"] == {"enabled": False}
    assert {k: v for k, v in on.items() if k != "sandbox"} == {k: v for k, v in off.items() if k != "sandbox"}


def _hook_ran(tmp_path, settings: dict) -> bool:
    """Start the real claude in a run directory holding a planted SessionStart hook, with no login (no model
    call), and report whether the hook ran."""
    run = tmp_path / "run"
    (run / ".claude").mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "HOOK_RAN"
    marker.unlink(missing_ok=True)
    (run / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
        {"type": "command", "command": f"touch {marker}"}]}]}}))
    (tmp_path / "account").mkdir(exist_ok=True)
    # No credential of any kind (API key, auth token, OAuth token, Bedrock, or Vertex switches), so no model call.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "account")
    subprocess.run(["claude", "--permission-mode", "dontAsk", "--strict-mcp-config",
                    "--settings", json.dumps(settings), "--", "say hi"],
                   cwd=run, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    return marker.exists()


@pytest.mark.skipif(not shutil.which("claude"), reason="Claude Code is not installed")
def test_the_real_claude_runs_no_planted_hook_under_dycraft_s_settings(tmp_path):
    """The control shows the planted hook does run without the setting, so the check is meaningful."""
    settings = claude_agent_settings(str(tmp_path / "run"))
    control = {k: v for k, v in settings.items() if k != "disableAllHooks"}
    assert _hook_ran(tmp_path, control), "the planted hook did not run even without disableAllHooks"
    assert not _hook_ran(tmp_path, settings)
