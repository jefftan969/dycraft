"""The `--backend` surface.

`BACKENDS` is not an internal lookup table: its keys *are* the `--backend` and
`--sub-backend` choices, built as `sorted(BACKENDS.keys())` at parser
construction. Adding a key adds a public CLI value; removing one removes it.
That makes the registry worth pinning explicitly rather than leaving it to be
read off whichever backend happens to be under test elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.cli.backends import (ACCOUNT_SLOTS, BACKENDS,
                                  CLAUDE_CONFIG_DIR_ALIASES,
                                  ClaudeBackend, CodexBackend)


def test_only_the_claude_selectors_shorten_the_harness_wait():
    """Claude's shell kills a command at two minutes, so every claude selector,
    and no codex one, asks the harness to return a waiting call at 100 s."""
    for selector, cls in BACKENDS.items():
        expected = "100" if selector.startswith("claude") else None
        assert cls.harness_wait_s == expected, selector


def test_every_numbered_selector_pins_its_own_account():
    """`claude7` must authenticate as `~/.claude7`, never `~/.claude6`.

    The selectors are generated from `backends.ACCOUNT_SLOTS`, so one slot
    number reaches the class name, the alias table and the account dir. This
    pins the property that generation exists to guarantee — and it is the check
    a hand-written family needs most, because a copy-pasted slot still passes
    every test that only asserts *some* account got pinned.
    """
    home = Path.home()
    for slot in ACCOUNT_SLOTS:
        claude = BACKENDS[f"claude{slot}"]
        binary, config_dir = claude._resolve_executable()
        assert claude.name == "claude", slot      # name-based conditioning
        assert binary == "claude", slot           # account switch, not binary
        assert config_dir == str(home / f".claude{slot}"), slot

        codex = BACKENDS[f"codex{slot}"]
        assert codex.name == "codex", slot
        assert codex._resolve_codex_cli()[1] == str(home / f".codex{slot}"), slot


def test_a_generated_selector_identifies_itself_as_this_module():
    """Nothing binds these classes at module level, so a traceback or a `repr`
    is all a reader gets — it has to be right.

    `type()` builds them inside `abc.py`, because the base is an ABC, and so
    they inherited `__module__ == "abc"`: `<class 'abc.Claude7Backend'>`, a
    repr pointing at the standard library for a class that lives here.
    """
    for slot in ACCOUNT_SLOTS:
        for selector, expected in ((f"claude{slot}", f"Claude{slot}Backend"),
                                   (f"codex{slot}", f"Codex{slot}Backend")):
            cls = BACKENDS[selector]
            assert cls.__name__ == expected
            assert cls.__module__ == f"dycraft.cli.backends.{cls.name}", selector


def test_the_alias_table_covers_exactly_the_claude_selectors():
    """`_resolve_executable` maps each `claudeN` selector to `claude` on that
    account's config dir through this table; a selector missing from it would
    be launched as a binary named `claudeN`."""
    assert sorted(CLAUDE_CONFIG_DIR_ALIASES) == sorted(
        name for name in BACKENDS if name.startswith("claude") and name != "claude")


# ---------------------------------------------------------------------------
# The daemon authenticates as the agent does
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot", ACCOUNT_SLOTS)
def test_prepare_shared_env_pins_the_account(slot, monkeypatch):
    """`execute` pins the account only on its private env copy; the daemon
    starts `review()`, dispatched children and rewriters from the env
    `prepare_shared_env` edits, so a `claudeN` run must pin `~/.claudeN` there
    too, whatever the launching shell exported."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/wrong/dir")
    env: dict[str, str] = {}
    BACKENDS[f"claude{slot}"]().prepare_shared_env(env)
    assert env["CLAUDE_CONFIG_DIR"].endswith(f"/.claude{slot}")


def test_plain_claude_backend_does_not_pin_an_account(monkeypatch):
    """`--backend claude` leaves CLAUDE_CONFIG_DIR to the launching shell
    (or the CLI's own `~/.claude`); only `claude{1..9}` pin an account."""
    monkeypatch.delenv("DYCRAFT_CLAUDE_EXECUTABLE", raising=False)
    env: dict[str, str] = {}
    ClaudeBackend().prepare_shared_env(env)
    assert "CLAUDE_CONFIG_DIR" not in env


def test_prepare_shared_env_leaves_an_explicit_config_dir_alone(monkeypatch):
    """`DYCRAFT_CLAUDE_EXECUTABLE=claude` targets a named account through the
    ambient CLAUDE_CONFIG_DIR, which the hook must not overwrite."""
    monkeypatch.setenv("DYCRAFT_CLAUDE_EXECUTABLE", "claude")
    env = {"CLAUDE_CONFIG_DIR": "/home/u/.claude-work"}
    ClaudeBackend().prepare_shared_env(env)
    assert env["CLAUDE_CONFIG_DIR"] == "/home/u/.claude-work"


def test_codex_prepare_shared_env_pins_codex_home(monkeypatch):
    """Codex's `execute` sets CODEX_HOME after the daemon is spawned, so the
    daemon's env needs it for its sub-runs to use the agent's account."""
    monkeypatch.setenv("CODEX_HOME", "/wrong/home")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    env: dict[str, str] = {}
    CodexBackend().prepare_shared_env(env)
    assert env["CODEX_HOME"] == str(Path(CodexBackend.HOME).expanduser())


def test_credential_meta_records_the_codex_home(monkeypatch):
    """meta.json names the account a run used: transcript capture looks for the
    rollout under `$CODEX_HOME/sessions/`, and a Codex home must never be filed
    under the claude key, which is read back as a claude account."""
    monkeypatch.setattr("shutil.which", lambda _: "/bin/codex")
    meta = CodexBackend().credential_meta()
    assert meta["codex_home"] == str(Path(CodexBackend.HOME).expanduser())
    assert "claude_config_dir" not in meta
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/home/u/.claude2")
    assert ClaudeBackend().credential_meta() == {"claude_config_dir": "/home/u/.claude2"}


@pytest.mark.parametrize("selector,family", [
    ("claude", "claude"), ("claude1", "claude"), ("claude9", "claude"),
    ("codex", "codex"), ("codex1", "codex"), ("codex9", "codex"),
    # Not selectors: a two-digit or zero-suffixed name is not in
    # `ACCOUNT_SLOTS`, so it is left alone rather than given an account.
    ("claude10", "claude10"), ("codex0", "codex0"),
])
def test_backend_family_collapses_the_account_selectors(selector, family):
    """`run_oneoff` sees only the selector string, and every one of its
    result-recovery branches conditions on the family; a selector read as an
    unknown backend loses the sub-agent's answer."""
    from dycraft.agent.oneoff import backend_family

    assert backend_family(selector) == family


# ---------------------------------------------------------------------------
# The model the agent CLI will run (reported by `install check`)
# ---------------------------------------------------------------------------

def test_claude_agent_model_comes_from_the_account_settings(tmp_path, monkeypatch):
    home = tmp_path / "acct"
    home.mkdir()
    (home / "settings.json").write_text(json.dumps({"model": "opus[1m]"}))
    monkeypatch.setattr(ClaudeBackend, "_config_dir", classmethod(lambda cls: home))
    assert ClaudeBackend.resolve_agent_model(None) == "opus[1m]"
    assert ClaudeBackend.resolve_agent_model("fable") == "fable"


def test_claude_agent_model_survives_a_broken_settings_file(tmp_path, monkeypatch):
    """A malformed user config must not be able to block a launch."""
    home = tmp_path / "acct"
    home.mkdir()
    (home / "settings.json").write_text("{not json")
    monkeypatch.setattr(ClaudeBackend, "_config_dir", classmethod(lambda cls: home))
    assert ClaudeBackend.resolve_agent_model(None) is None


def test_codex_agent_model_comes_from_config_toml(tmp_path, monkeypatch):
    home = tmp_path / "codex_home"
    home.mkdir()
    (home / "config.toml").write_text(
        'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "medium"\n'
        '[projects."/repo"]\ntrust_level = "trusted"\n')
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli",
                        classmethod(lambda cls: ("/bin/codex", str(home))))
    assert CodexBackend.resolve_agent_model(None) == "gpt-5.6-sol"
    assert CodexBackend.resolve_agent_model("gpt-6-astra") == "gpt-6-astra"


@pytest.mark.parametrize("body", ["", "not = [valid toml", 'other = "x"\n'])
def test_codex_agent_model_is_none_without_a_readable_model(tmp_path, monkeypatch, body):
    """A missing, unparseable, or model-less config.toml must not block a launch."""
    home = tmp_path / "codex_home"
    home.mkdir()
    if body:
        (home / "config.toml").write_text(body)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli",
                        classmethod(lambda cls: ("/bin/codex", str(home))))
    assert CodexBackend.resolve_agent_model(None) is None


# ---------------------------------------------------------------------------
# `meta.json`'s `runtime`: the backend selector, with the older `*_cli` form still read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("runtime,backend", [
    ("codex1", "codex1"), ("claude", "claude"), ("codex1_cli", "codex1"), ("claude_cli", "claude"),
    ("bare_agent", ""), ("hand_authored", ""), ("api_cli", ""), ("", ""), (None, ""),
])
def test_the_reviewer_and_children_follow_the_run_s_recorded_backend(runtime, backend):
    """A codex run's reviewer and dispatched children must run on codex and its
    account, whichever form the run recorded."""
    from dycraft.agent.layers.review import _backend_from_runtime

    assert _backend_from_runtime(runtime) == backend


@pytest.mark.parametrize("runtime,codex", [
    ("codex1", True), ("codex1_cli", True), ("codex", True), ("claude2", False), ("claude_cli", False),
])
def test_a_codex_run_is_recognised_from_either_runtime_form(tmp_path, runtime, codex):
    from dycraft.agent import transcript_capture as tc

    (tmp_path / "meta.json").write_text(json.dumps({"runtime": runtime}))
    assert tc.is_codex_run(tmp_path) is codex


def test_the_launcher_records_the_backend_selector_as_runtime(tmp_path, monkeypatch):
    from dycraft.agent.agent_session import write_run_meta

    write_run_meta(tmp_path, layer="animation", runtime="codex3", prompt="p", run_id="r", model=None)
    assert json.loads((tmp_path / "meta.json").read_text())["runtime"] == "codex3"
