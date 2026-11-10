"""Backends — the agent CLIs behind one interface.

    ClaudeBackend   → runs the `claude` CLI; agent drives via $RUN_DIR/harness.py
    CodexBackend    → runs the `codex`  CLI; same daemon socket as ClaudeBackend

Both consume a `LaunchPlan` that `cli.py` populates after it has started the
harness daemon and the self-kill watcher.

Package layout: `_base.py` (the interface, the launch plan, the agent's shell
env), `claude.py`, `codex.py`, and `_process.py` (running and stopping the
agent CLI process; the process state lives there, read through `stop_status`).
`__all__` is what code outside the package imports from here; a test patches a
name in the module that defines it.
"""
from __future__ import annotations

from ._base import (
    ACCOUNT_SLOTS,
    ALLOW_UNSANDBOXED_ENV,
    Backend,
    LaunchPlan,
    agent_shell_env,
    allow_unsandboxed,
)
from .claude import (
    CLAUDE_CONFIG_DIR_ALIASES,
    ClaudeBackend,
    claude_agent_settings,
    claude_sandbox_problem,
    claude_sandbox_socket_fallback,
)
from .codex import (
    ALLOW_UNSANDBOXED_CODEX_ENV,
    CODEX_HEADLESS_ARGS,
    CodexBackend,
    codex_allowed_approval_policies,
    codex_sandbox_probe,
    codex_sandbox_refusal,
    mark_codex_untrusted,
)
from ._process import (
    AGENT_STOP_GRACE_SECONDS,
    EXIT_CODEX_BLOCKED,
    EXIT_CODEX_FATAL,
    run_agent_cli,
    stop_agent_cli,
    stop_status,
)
from .claude import _claude_account_backend
from .codex import _codex_account_backend


# Keyed by CLI `--backend` selector, and the only handle on the generated
# account selectors. `claudeN`/`codexN` are distinct selectors resolving to
# subclasses that share their family's `name`, so the account differs but all
# name-based conditioning still sees "claude"/"codex" — which is also how a
# caller asks for one family's selectors: `cls.name == "claude"`.
BACKENDS: dict[str, type[Backend]] = {
    ClaudeBackend.name: ClaudeBackend,
    **{f"claude{slot}": _claude_account_backend(slot)
       for slot in ACCOUNT_SLOTS},
    CodexBackend.name:  CodexBackend,
    **{f"codex{slot}": _codex_account_backend(slot)
       for slot in ACCOUNT_SLOTS},
}


__all__ = [
    "ACCOUNT_SLOTS",
    "AGENT_STOP_GRACE_SECONDS",
    "ALLOW_UNSANDBOXED_CODEX_ENV",
    "ALLOW_UNSANDBOXED_ENV",
    "BACKENDS",
    "Backend",
    "CLAUDE_CONFIG_DIR_ALIASES",
    "CODEX_HEADLESS_ARGS",
    "ClaudeBackend",
    "CodexBackend",
    "EXIT_CODEX_BLOCKED",
    "EXIT_CODEX_FATAL",
    "LaunchPlan",
    "agent_shell_env",
    "allow_unsandboxed",
    "claude_agent_settings",
    "claude_sandbox_problem",
    "claude_sandbox_socket_fallback",
    "codex_allowed_approval_policies",
    "codex_sandbox_probe",
    "codex_sandbox_refusal",
    "mark_codex_untrusted",
    "run_agent_cli",
    "stop_agent_cli",
    "stop_status",
]
