"""The environment check reports each agent CLI's installed version and path without judging the version, and runs
the launch's own probe of Codex's sandbox.

The versions the paper's runs used are a developer note in `environment_test.py`, not a requirement, so no version
warns or fails.
"""
from types import SimpleNamespace

import pytest

from dycraft.cli.backends.codex import codex_account_problem as _real_account_problem

from dycraft import environment_test as E


def _fake_cli(tmp_path, name, version_line):
    fake = tmp_path / name
    fake.write_text(f"#!/bin/sh\necho '{version_line}'\n" if version_line else "#!/bin/sh\n")
    fake.chmod(0o755)
    return fake


@pytest.mark.parametrize("version_line", ["2.1.319 (Claude Code)", "2.1.400 (Claude Code)", "2.0.9 (Claude Code)", ""])
def test_any_claude_version_passes_with_its_version_and_path(monkeypatch, tmp_path, version_line):
    from dycraft.cli.backends import ClaudeBackend
    fake = _fake_cli(tmp_path, "claude", version_line)
    monkeypatch.setattr(ClaudeBackend, "_resolve_executable", staticmethod(lambda: (str(fake), None)))
    monkeypatch.setattr(E, "which", lambda name: str(fake) if name == str(fake) else "/usr/bin/socat")
    result = E.check_claude(SimpleNamespace(facts={}))
    assert result.status == E.PASS
    assert result.summary == f"{version_line or '?'} — {fake}"


@pytest.fixture(autouse=True)
def codex_logged_in(monkeypatch):
    """The account's login, which `check_codex` reads from its home, taken as present unless a test says not."""
    from dycraft.cli.backends import codex as X
    monkeypatch.setattr(X, "codex_account_problem", lambda home, env=None: None)


@pytest.mark.parametrize("version_line", ["codex-cli 0.153.4", "codex-cli 0.166.0", "codex-cli 0.99.1", ""])
def test_any_codex_version_passes_with_its_version_and_path(monkeypatch, tmp_path, version_line):
    from dycraft.cli.backends import CodexBackend
    fake = _fake_cli(tmp_path, "codex", version_line)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(tmp_path))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: (None, None, None, []))
    result = E.check_codex(SimpleNamespace(facts={}))
    assert result.status == E.PASS
    assert result.summary == f"{version_line or '?'} — {fake}"


def test_the_codex_check_reports_a_sandbox_that_does_not_work(monkeypatch, tmp_path):
    """The check runs the launch's own sandbox probe, so it says up front that Codex agents would be refused."""
    from dycraft.cli.backends import CodexBackend
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\necho codex-cli 0.153.4\n")
    fake.chmod(0o755)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(tmp_path))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: ("the sandbox failed to start: bwrap: x", None, None, []))
    result = E.check_codex(SimpleNamespace(facts={}))
    assert result.status == E.DEGRADED
    assert "bwrap: x" in result.summary and "--allow-unsandboxed" in result.lost
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: (None, None, None, []))
    assert E.check_codex(SimpleNamespace(facts={})).status == E.PASS


def test_a_missing_cli_is_installed_from_its_vendors_own_installer(monkeypatch, tmp_path):
    """Checked against the vendors' pages on 11-08: Claude Code's native installer, and Codex's standalone one."""
    from dycraft.cli.backends import ClaudeBackend, CodexBackend
    monkeypatch.setattr(E, "which", lambda name: None)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (None, str(tmp_path))))
    monkeypatch.setattr(ClaudeBackend, "_resolve_executable", classmethod(lambda cls: ("claude", None)))
    codex = E.check_codex(SimpleNamespace(facts={}))
    assert codex.fix[0] == ("install the Codex CLI: `curl -fsSL https://chatgpt.com/codex/install.sh | sh` "
                            "(https://learn.chatgpt.com/docs/codex/cli)")
    claude = E.check_claude(SimpleNamespace(facts={}))
    assert claude.fix[0] == ("install Claude Code: `curl -fsSL https://claude.ai/install.sh | bash` "
                             "(https://code.claude.com/docs/en/setup)")


def test_the_codex_check_names_the_account_a_launch_uses_and_its_login(monkeypatch, tmp_path):
    """--backend codex runs on ~/.codex whatever CODEX_HOME says; a missing login is degraded, and does not refuse a
    codexN launch, which checks its own account."""
    from dycraft.cli.backends import CodexBackend
    from dycraft.cli.backends import codex as X
    fake = _fake_cli(tmp_path, "codex", "codex-cli 0.167.0")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(home))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: (None, None, None, []))
    monkeypatch.setattr(X, "codex_account_problem", _real_account_problem)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    result = E.check_codex(SimpleNamespace(facts={}))
    assert result.status == E.DEGRADED and result.data == {"login": True}
    assert f"cannot start an agent: Codex is not logged in on {home}: it holds no auth.json" in result.summary
    assert result.fix == [f"`CODEX_HOME={home} codex login`"]
    assert any(line.startswith(f"CODEX_HOME={tmp_path / 'elsewhere'} is not used") for line in result.detail)
    check = next(c for c in E.CHECKS if c.id == "harness.codex")
    assert E.launch_refusals([(check, result)], "codex") == []
