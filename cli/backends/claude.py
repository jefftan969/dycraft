"""`ClaudeBackend`: runs the `claude` CLI (Claude Code) on a run directory, with its
sandbox settings and the `claudeN` account selectors.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from . import _process
from ...agent.agent_session import record_agent_stop
from ._base import ACCOUNT_SLOTS, Backend, LaunchPlan, _fmt_cmd_for_log, agent_shell_env, allow_unsandboxed


# `claude1`..`claude9` are bash aliases that pick an account by exporting
# CLAUDE_CONFIG_DIR before invoking the real `claude` binary. Aliases don't
# survive execvpe, so the mapping is reproduced here; `ACCOUNT_SLOTS` is the
# one place a slot is added.
CLAUDE_CONFIG_DIR_ALIASES: dict[str, str] = {
    f"claude{slot}": f"~/.claude{slot}" for slot in ACCOUNT_SLOTS
}


#: 1 on a stock Ubuntu 24.04: AppArmor denies user namespaces to every binary
#: without a profile that grants `userns`.
_USERNS_RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")


def claude_sandbox_socket_fallback(sandbox: dict) -> dict:
    """`sandbox`, a Claude Code `sandbox` settings block, with the keys that
    let its Bash sandbox start on a host that restricts nested user namespaces;
    unchanged where it starts as it is. Every launcher of a sandboxed claude
    should pass its block through here, as `ClaudeBackend` does.

    Claude Code >= 2.1.305 blocks Unix sockets inside a sandboxed command with
    a seccomp filter, installed by an `apply-seccomp` helper that needs a user
    namespace nested inside bubblewrap's. With
    `kernel.apparmor_restrict_unprivileged_userns = 1` AppArmor denies that
    namespace (the bwrap profile covers bubblewrap's own namespace, not the
    nested one), so every sandboxed command fails before it starts:
    `apply-seccomp: write /proc/self/setgroups (nested userns is
    capability-restricted ...): Permission denied`. The agent then cannot run
    harness.py at all.

    `network.allowAllUnixSockets` makes Claude Code skip that helper. The
    write confinement and the localhost-only network are unchanged; what goes
    is the socket block itself, and on a desktop host that matters: from
    inside the sandbox a command could reach the tmux server, the X11 display,
    the systemd user manager and the session bus, and each of those runs
    commands outside the sandbox. `filesystem.denyRead` masks them (bubblewrap
    >= 0.5 masks socket files as well as directories): the container and bus
    sockets Claude Code itself masks in its hardened profile, plus the
    per-user ones under /tmp.

    `DYCRAFT_CLAUDE_UNIX_SOCKETS` overrides the detection: `block` keeps
    Claude Code's default (for a host where an AppArmor profile lets the
    helper run), `allow` applies this anywhere.
    """
    mode = os.environ.get("DYCRAFT_CLAUDE_UNIX_SOCKETS", "").strip().lower()
    if mode == "block":
        return sandbox
    if mode != "allow":
        try:
            if _USERNS_RESTRICT_SYSCTL.read_text().strip() != "1":
                return sandbox
        except OSError:
            return sandbox
    uid = os.getuid()
    masks = [
        # Claude Code's own list: container runtimes and the system and
        # per-user buses (`/run/user/<uid>` holds the session bus and the
        # systemd user manager).
        "/run/docker.sock", "/run/containerd/containerd.sock",
        "/run/podman/podman.sock", "/run/buildkit/buildkitd.sock",
        "/run/dbus", "/run/user",
        # Terminal multiplexers, the X11 display, a session bus started
        # outside systemd, ssh agents, Chrome and Chrome Remote Desktop, and
        # the Claude Code daemon that controls other sessions.
        "/run/screen", f"/tmp/tmux-{uid}", "/tmp/.X11-unix", "/tmp/.ICE-unix",
        "/tmp/dbus-*", "/tmp/ssh-*", "/tmp/.com.google.Chrome.*",
        "/tmp/chromoting.*", f"/tmp/cc-daemon-{uid}",
    ]
    filesystem = sandbox.get("filesystem", {})
    return {
        **sandbox,
        "network": {**sandbox.get("network", {}), "allowAllUnixSockets": True},
        "filesystem": {**filesystem,
                       "denyRead": [*filesystem.get("denyRead", []), *masks]},
    }


def claude_sandbox_problem() -> str | None:
    """None where Claude Code's Bash sandbox can start with the `bwrap` and `socat` on this PATH; otherwise why not.

    Two causes: either tool missing, and a bwrap that cannot make a user and a network namespace here, such as one
    outside /usr/bin on a host whose AppArmor policy (`kernel.apparmor_restrict_unprivileged_userns=1`, stock Ubuntu
    24.04) grants them to /usr/bin/bwrap only."""
    import subprocess
    missing = [name for name in ("bwrap", "socat") if shutil.which(name) is None]
    if missing:
        return f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not on PATH"
    from ... import _oom
    bwrap = shutil.which("bwrap")
    try:
        ran = subprocess.run(_oom.tiered([bwrap, "--unshare-user", "--unshare-net", "--ro-bind", "/", "/", "true"],
                                         _oom.WORKER), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{bwrap} could not run: {exc}"
    if ran.returncode != 0:
        why = (ran.stderr or ran.stdout).strip().splitlines()
        return f"{bwrap} cannot make a user and a network namespace here ({why[-1] if why else f'exit {ran.returncode}'})"
    return None


def claude_login_problem(executable: str, config_dir: str | None) -> str | None:
    """Why the Claude Code account in `config_dir` (CLAUDE_CONFIG_DIR; None: the CLI's own default) cannot start an
    agent, from `claude auth status`, which asks no model: not logged in, or a login past its expiry. None when it
    can, and when the CLI does not say (an older one with no `auth status`): the session watch still catches a login
    error at the start."""
    import json
    import subprocess
    from datetime import datetime, timezone

    from ... import _oom
    env = {**os.environ, **({"CLAUDE_CONFIG_DIR": config_dir} if config_dir else {})}
    try:
        ran = subprocess.run(_oom.tiered([executable, "auth", "status"], _oom.WORKER), capture_output=True, text=True,
                             timeout=30, env=env, stdin=subprocess.DEVNULL)
        status = json.loads(ran.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(status, dict) or "loggedIn" not in status:
        return None
    if not status.get("loggedIn"):
        return "not logged in"
    expires = status.get("loginExpiresAt")
    try:
        when = datetime.fromisoformat(str(expires).replace("Z", "+00:00")) if expires else None
    except ValueError:
        when = None
    if when is not None and when <= datetime.now(timezone.utc):
        return f"its login expired on {when:%Y-%m-%d}"
    return None


def claude_login_fix(config_dir: str | None) -> str:
    """How to log the account in `config_dir` in, as the user types it."""
    home = str(Path("~/.claude").expanduser())
    if config_dir and str(Path(config_dir).expanduser()) != home:
        shown = str(config_dir).replace(str(Path.home()), "~", 1)
        return f"log in: `CLAUDE_CONFIG_DIR={shown} claude auth login`"
    return "log in: `claude auth login`"


def claude_userns_fixes() -> list[str]:
    """With root, what lets Claude Code's bubblewrap make its user and network namespaces on this system
    (`_userns.fixes`): the package, a profile, or lifting AppArmor's restriction, as this system needs."""
    from ... import _userns
    return _userns.fixes() or ["the system's bubblewrap: `sudo apt-get install bubblewrap socat`"]


def claude_agent_settings(run_dir: str, *, deny=(), sandboxed: bool = True) -> dict:
    """The `--settings` DyCraft starts Claude Code with (`permission-mode dontAsk`).

    Bash runs inside Claude Code's sandbox, writing only `run_dir` and reaching only localhost; the
    sandbox is required, and the model's `dangerouslyDisableSandbox` escape is off. Reads are allowed
    everywhere, and edits only under `run_dir`, outside any `.claude/` there. Hooks are off.
    `install check`'s confinement probe launches with these same settings, so its verdict is about
    DyCraft's agents.

    `sandboxed=False` (`--allow-unsandboxed`, where `claude_sandbox_problem` finds the sandbox cannot start)
    turns the sandbox off, so Bash runs with no confinement; the rest is unchanged.
    """
    return {
        # Claude Code runs hooks itself, outside the sandbox, and reads them from settings files in its
        # working directory, which a parent agent can write for the children it dispatches. DyCraft sets
        # no hooks of its own.
        "disableAllHooks": True,
        "sandbox": {"enabled": False} if not sandboxed else claude_sandbox_socket_fallback({
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "failIfUnavailable": True,
            # No escape hatch. Otherwise the model may rerun any command
            # with `dangerouslyDisableSandbox: true`, which `Bash(*)` below
            # approves, and agents do so after a refused write.
            "allowUnsandboxedCommands": False,
            "filesystem": {"allowWrite": [run_dir]},
            "network": {"allowedDomains": ["localhost"]},
        }),
        "permissions": {
            "allow": [
                "Bash(*)",
                # The bare tool name allows reads of every path; `Read(*)`
                # means only the working directories, which `dontAsk` would
                # turn into refused reads of the catalog, sibling runs and docs.
                "Read",
                # Edit rules cover every file-editing tool, Write included;
                # Claude Code ignores a `Write(...)` rule and warns.
                f"Edit(//{run_dir}/**)",
            ],
            # No agent needs to write a `.claude/` directory in a run directory (the launcher stages
            # skills there), and one written by a parent would give a child it dispatches settings,
            # permission rules or hooks. Deny rules beat the allow above. They cover the file tools; the
            # shell sandbox cannot deny a directory created after it starts.
            "deny": [f"Edit(//{run_dir}/.claude/**)", f"Edit(//{run_dir}/**/.claude/**)", *deny],
        },
    }


#: How long claude may run before it has written one record to its session
#: transcript. The first record comes within seconds of the launch (at most
#: 3.2 s over 180 recorded launches); 30 s is about nine times that, for a slow
#: node or a cold cache, and an operator is not left waiting minutes on a start
#: that never finishes.
CLAUDE_STARTUP_BOUND_SECONDS = 30.0

#: How long a usage-limit or login error may stand as the session's last record
#: before the CLI is stopped. With its stdin at EOF Claude Code ends the turn on
#: such an error and exits by itself; this covers a release that waits instead.
CLAUDE_FATAL_GRACE_SECONDS = 60.0

#: Switches the session watch off, for a setup it misjudges.
NO_SESSION_WATCH_ENV = "DYCRAFT_CLAUDE_NO_SESSION_WATCH"

#: An API-error record's `error`, for the errors a session cannot go on from by
#: itself, and how the message names it.
_FATAL_API_ERRORS = {
    "authentication_failed": "claude is not logged in",
    # Both the account's own limit ("You've hit your session limit · resets 6:40pm") and the server limiting
    # requests ("... (not your usage limit)"); the quoted text says which.
    "rate_limit": "claude hit a usage or rate limit",
    "billing_error": "claude's account cannot be billed",
}

#: How a sandboxed Bash command that never started reads: Claude Code's sandbox
#: helpers print these before the command runs.
_SANDBOX_START_FAILURES = ("apply-seccomp:", "bwrap:")


def _block_text(content) -> str:
    """A tool_result's content as text: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


class ClaudeSessionWatch:
    """Stops a claude session that cannot go on, reading its transcript as it is written.

    Three cases, each a run that would otherwise sit until its time limit or end
    with nothing said:

    * no record at all within `CLAUDE_STARTUP_BOUND_SECONDS`: claude never
      started its first turn (a screen it waits on, a start it never finishes);
    * a Bash command that failed to start in Claude Code's sandbox, before any
      Bash command has succeeded: the agent cannot run the harness on this host;
    * a usage-limit, login, or billing error as the session's last record, for
      `CLAUDE_FATAL_GRACE_SECONDS`, or when the CLI exits on it.

    Reads the session file under `$CLAUDE_CONFIG_DIR/projects/<run dir key>/`,
    only records written since this launch (a resumed session's file holds the
    old ones), and only `user` and `assistant` records: `api-request-blob`
    records echo earlier messages, the same error text included.
    """

    poll_seconds = 2.0

    def __init__(self, config_dir: Path, run_dir: Path, interactive: bool = False):
        import time
        self.config_dir = Path(config_dir).expanduser()
        self.run_dir = Path(run_dir)
        #: On a terminal the operator sees claude's screens (its folder-trust question for a new run directory comes
        #: before any record) and answers them, so the startup bound, which is for headless launches, is not applied.
        self.interactive = interactive
        self.started = time.monotonic()
        self.wall_started = time.time()
        self.enabled = not os.environ.get(NO_SESSION_WATCH_ENV)
        self.path: Path | None = None
        self.offset = 0
        self.records = 0
        self.bash_ids: set = set()
        self.sandbox_proven = False
        self.sandbox_failure: str | None = None
        self.armed: tuple | None = None          # (monotonic time, error, text)
        self.last_api_error: tuple | None = None # (error, text) of any API error the session ended on
        self.answered = False                    # the model has answered at least once

    def _find(self) -> Path | None:
        from ...agent.transcript_capture import _normalized, _transcripts_under, project_key
        key = project_key(self.run_dir)
        fresh = []
        for f in _transcripts_under(self.config_dir, key, _normalized(key)):
            try:
                if f.stat().st_mtime >= self.wall_started - 5:
                    fresh.append((f.stat().st_mtime, f))
            except OSError:
                continue
        return max(fresh)[1] if fresh else None

    def _found(self, path: Path | None) -> None:
        """Take `path` as this launch's session file, and record its session id in `meta.json` for `--resume`."""
        self.path = path
        if path is not None:
            from ...agent.transcript_capture import remember_claude_session
            remember_claude_session(self.run_dir, path.stem)

    def _records(self):
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return []
        cut = chunk.rfind(b"\n")                  # a half-written last line waits for the next poll
        if cut == -1:
            return []
        self.offset += cut + 1
        out = []
        for line in chunk[:cut + 1].splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def _take(self, rec: dict, now: float) -> None:
        stamp = _process.iso_seconds(rec.get("timestamp"))
        if stamp is not None and stamp < self.wall_started - 1:
            return                                 # an earlier launch's record, in a resumed session's file
        self.records += 1
        kind, message = rec.get("type"), rec.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if kind == "assistant":
            error = rec.get("error")
            if rec.get("isApiErrorMessage"):
                self.last_api_error = (error, _block_text(content).strip())
                if error in _FATAL_API_ERRORS:
                    self.armed = (now, error, self.last_api_error[1])
                return
            self.armed = self.last_api_error = None   # the model answered: whatever went before passed
            self.answered = True
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash":
                    self.bash_ids.add(block.get("id"))
        elif kind == "user" and not self.sandbox_proven:
            for block in content if isinstance(content, list) else []:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"
                        and block.get("tool_use_id") in self.bash_ids):
                    continue
                if not block.get("is_error"):
                    self.sandbox_proven = True
                    return
                for line in _block_text(block.get("content")).splitlines()[:4]:
                    if line.strip().startswith(_SANDBOX_START_FAILURES):
                        self.sandbox_failure = line.strip()
                        return

    def poll(self):
        """`(status, message)` when the CLI should be stopped, else None."""
        import time
        if not self.enabled:
            return None
        now = time.monotonic()
        if self.path is None:
            self._found(self._find())
        if self.path is not None:
            for rec in self._records():
                self._take(rec, now)
        if self.sandbox_failure:
            self.enabled = False
            record_agent_stop(self.run_dir, reason="claude_sandbox_cannot_start", detail=self.sandbox_failure)
            return (_process.EXIT_AGENT_FATAL,
                    f"FATAL: Claude Code's sandbox cannot start a command on this host ({self.sandbox_failure}), "
                    "so the agent cannot run the harness; the CLI is stopped. On a host that restricts nested "
                    "user namespaces, set DYCRAFT_CLAUDE_UNIX_SOCKETS=allow: Claude Code then runs without its "
                    "Unix-socket filter, which needs a nested user namespace, and DyCraft hides the sockets of "
                    "container runtimes, D-Bus, tmux, X11, and SSH agents from the agent instead. "
                    f"{NO_SESSION_WATCH_ENV}=1 disables this watch.")
        if self.records == 0 and not self.interactive and now - self.started > CLAUDE_STARTUP_BOUND_SECONDS:
            self.enabled = False
            record_agent_stop(self.run_dir, reason="claude_never_started",
                              detail=f"no session record in {CLAUDE_STARTUP_BOUND_SECONDS:g} s")
            return (_process.EXIT_AGENT_BLOCKED,
                    f"FATAL: claude wrote no session record in {CLAUDE_STARTUP_BOUND_SECONDS:g} s after it started: "
                    "it is stuck before its first turn (a login, trust or onboarding screen, or a start it never "
                    "finishes). The CLI is stopped instead of waiting out the run's time limit. "
                    f"{NO_SESSION_WATCH_ENV}=1 disables this watch.")
        if self.armed and now - self.armed[0] >= CLAUDE_FATAL_GRACE_SECONDS:
            self.enabled = False
            return (_process.EXIT_AGENT_FATAL, self._fatal_message(stopped=True))
        return None

    def after_exit(self, returncode: int = 0) -> int | None:
        """After the CLI exits: say why when it ended on an API error, or exited non-zero before the model ever
        answered, and return the launcher's exit status for that; None when there is nothing to say. Reads the
        transcript one last time first, finding it if the CLI exited before the first poll."""
        import time
        if self.enabled:
            if self.path is None:
                self._found(self._find())
            if self.path is not None:
                for rec in self._records():
                    self._take(rec, time.monotonic())
        if self.armed:
            print(self._fatal_message(stopped=False), file=sys.stderr, flush=True)
            return _process.EXIT_AGENT_FATAL
        if self.last_api_error:
            error, text = self.last_api_error
            record_agent_stop(self.run_dir, reason=f"claude_api_error_{error or 'unknown'}", detail=text)
            print(f"FATAL: claude's API calls failed and it ended the session: "
                  f"{text.rstrip('. ') or error or 'no message'}. "
                  "Check this machine's network and proxy settings; the run directory keeps everything this run "
                  "committed.", file=sys.stderr, flush=True)
            return _process.EXIT_AGENT_FATAL
        if returncode not in (0, _process.SIGKILLED) and not self.answered and self.enabled:
            record_agent_stop(self.run_dir, reason="claude_exited_before_answering", detail=f"exit {returncode}")
            print(f"FATAL: claude exited (status {returncode}) before it answered. Its own reason is in its output "
                  "just above, for example \"Not logged in · Please run /login\": log in with this backend's "
                  "account (e.g. run `claude3` and /login) and launch again.", file=sys.stderr, flush=True)
            return _process.EXIT_AGENT_FATAL
        return None

    def _fatal_message(self, *, stopped: bool) -> str:
        _, error, text = self.armed
        record_agent_stop(self.run_dir, reason=f"claude_{error}", detail=text)
        what = _FATAL_API_ERRORS[error]
        how = "The CLI is stopped" if stopped else "The CLI ended the session"
        return (f"FATAL: {what}: {text.rstrip('. ') or 'no message'}. {how}; the run directory keeps everything "
                "this run committed.")


#: What Claude Code sets in the shells it starts, which a nested `claude` would take as its parent's.
CLAUDE_SESSION_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_CHILD_SESSION")


class ClaudeBackend(Backend):
    """Spawns the `claude` CLI with --append-system-prompt + the user-task
    footer as the initial user message. The agent drives the harness via
    $RUN_DIR/harness.py over the daemon socket.

    Sandboxing: claude code's bubblewrap-based sandbox is enabled with
    `autoAllowBashIfSandboxed` so harness bash calls don't prompt, and
    `filesystem.allowWrite` is pinned to the absolute run dir. The launcher
    also chdirs to the run dir so the sandbox's (non-disableable) implicit
    cwd-tree default coincides with that single allowed path — net effect
    for Bash subprocesses: writes only inside the run dir, everywhere else
    read-only.

    Edit/Write/Read are NOT covered by the bwrap sandbox — per Claude Code
    docs, "Read, Edit, and Write use the permission system directly rather
    than running through the sandbox." So those tools are scoped via
    permission rules instead: `Edit`/`Write` allow-listed only under the
    run dir (`//<run_dir>/**`), and `permission-mode` is `dontAsk` (not
    `acceptEdits`) so out-of-scope edits don't blanket-auto-approve. An
    Edit/Write call inside run_dir matches the allow rule and runs without
    prompting; any call no rule allows is denied at once and the agent is
    told so. Under `default` it would instead wait on a permission prompt
    that no one in this unattended run will answer. Read is allow-listed
    wildcard — reads are intentionally unrestricted (consistent with the
    bwrap posture of read-everywhere / write-narrowly).

    Network: bash-subprocess traffic is whitelisted to `localhost` only via
    `network.allowedDomains`. The harness daemon lives on loopback
    (127.0.0.1:<.harness.port>), so that single entry is all the agent's tool
    shells need. Claude's own API call
    to api.anthropic.com is unaffected because it happens in the main
    process — the sandbox only constrains Bash subprocesses (see
    https://code.claude.com/docs/en/sandboxing). This makes the network
    posture mirror the filesystem posture: minimum-permission, defense in
    depth against exfiltration from a compromised tool shell.

    On a host that restricts nested user namespaces (stock Ubuntu 24.04),
    Claude Code's Unix-socket block cannot start and takes every sandboxed
    command with it; there the settings trade it for path masks instead (see
    `claude_sandbox_socket_fallback`).

    `DYCRAFT_CLAUDE_PERMISSION_MODE` overrides the permission-mode flag for
    testing. `DYCRAFT_CLAUDE_EXECUTABLE=claude{1..9}` switches
    CLAUDE_CONFIG_DIR.

    Selecting `claude1`..`claude9` directly as the backend (rather than via the
    env var) is also supported — see `_claude_account_backend`, whose classes
    behave exactly as if `DYCRAFT_CLAUDE_EXECUTABLE` were set to that name.

    Account pinning is opt-in: only those names touch CLAUDE_CONFIG_DIR. A
    plain `--backend claude` leaves it exactly as the launching shell had it,
    so the account is the ambient CLAUDE_CONFIG_DIR, or the CLI's own
    `~/.claude` default when the shell sets nothing. Note the asymmetry that
    follows — DyCraft does not *force* `~/.claude` here, it declines to
    override; a shell exporting CLAUDE_CONFIG_DIR keeps winning.
    """

    name = "claude"
    agent_cmdline_marker = "claude"
    # Claude's Bash tool kills a command at two minutes unless told otherwise,
    # and a waiting call killed there looks like a dead one.
    harness_wait_s = "100"
    DEFAULT_PERMISSION_MODE = "dontAsk"
    DEFAULT_EXECUTABLE = "claude"
    # Overridden by the `claude{1..9}` subclasses to pin the executable
    # regardless of `DYCRAFT_CLAUDE_EXECUTABLE`. `None` = honor the env var.
    FORCED_EXECUTABLE: str | None = None
    #: Why Claude Code's sandbox cannot start here, when the launch runs without it (`--allow-unsandboxed`).
    _unsandboxed: str | None = None

    @classmethod
    def _resolve_executable(cls) -> tuple[str, str | None]:
        """Return `(binary, CLAUDE_CONFIG_DIR | None)` for this backend.

        A `claude1`..`claude9` backend is shorthand for `--backend claude`
        with `DYCRAFT_CLAUDE_EXECUTABLE` pinned to that name: the subclass
        forces the executable so the choice wins over (and stands in for) the
        env var, exactly as if the env var had been updated. Those names are
        the bash aliases in `CLAUDE_CONFIG_DIR_ALIASES`.

        A classmethod so the account can be resolved without constructing a
        backend (`install check` does); still callable on an instance, which
        is how `cli.py` reaches it when recording the account into meta.json.
        """
        executable = cls.FORCED_EXECUTABLE or os.environ.get(
            "DYCRAFT_CLAUDE_EXECUTABLE", cls.DEFAULT_EXECUTABLE)
        if executable in CLAUDE_CONFIG_DIR_ALIASES:
            return "claude", os.path.expanduser(
                CLAUDE_CONFIG_DIR_ALIASES[executable])
        return executable, None

    @classmethod
    def _config_dir(cls) -> Path:
        """The account dir the agent will authenticate against.

        Mirrors `execute`'s precedence exactly: the backend's own pin (a
        `claude{1..9}` selector) first, then whatever the launching shell
        exported, then the CLI's own default. Plain `--backend claude`
        deliberately does not override an ambient CLAUDE_CONFIG_DIR.
        """
        _, pinned = cls._resolve_executable()
        if pinned:
            return Path(pinned)
        ambient = os.environ.get("CLAUDE_CONFIG_DIR")
        return Path(ambient).expanduser() if ambient else Path("~/.claude").expanduser()

    @classmethod
    def resolve_agent_model(cls, model: str | None) -> str | None:
        """`--model`, else the account's own configured default model.

        With no `--model` the launcher emits no `--model` flag, so the agent
        runs whatever `settings.json` in its config dir says. Best-effort: a
        missing or unreadable settings file returns None.
        """
        if model:
            return model
        try:
            settings = json.loads((cls._config_dir() / "settings.json").read_text())
        except (OSError, ValueError):
            return None
        found = settings.get("model") if isinstance(settings, dict) else None
        return str(found) if found else None

    def credential_meta(self) -> dict[str, str]:
        """Which claude account this run drove, when one is determinable.

        Deliberately records nothing for a plain `--backend claude` with no
        ambient CLAUDE_CONFIG_DIR: DyCraft did not choose an account there, so
        claiming one in the bundle would be an invention, and transcript
        capture has its own fallback search for exactly that case.
        """
        _, pinned = self._resolve_executable()
        config_dir = pinned or os.environ.get("CLAUDE_CONFIG_DIR")
        return {"claude_config_dir": config_dir} if config_dir else {}

    def prepare_shared_env(self, env: dict[str, str]) -> None:
        """Pin the account for the daemon as well as the agent.

        `execute` resolves the config dir only on its private env copy, so
        without this a `claudeN` run would leave the daemon, and every sub-run
        it starts, on whatever CLAUDE_CONFIG_DIR the launching shell had.
        """
        _, config_dir = self._resolve_executable()
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir

    def preflight(self, run_dir: Path, layer: str | None = None) -> int | None:
        """Refuse (exit status 2) where Claude Code's Bash sandbox cannot start (`claude_sandbox_problem`): with
        `failIfUnavailable` it would refuse every command. With `--allow-unsandboxed`, launch without it instead."""
        if os.environ.get("DYCRAFT_SKIP_PREFLIGHT") == "1":
            return None
        executable, pinned = self._resolve_executable()
        account = pinned or os.environ.get("CLAUDE_CONFIG_DIR")
        login = claude_login_problem(executable, account) if shutil.which(executable) else None
        if login:
            where = str(self._config_dir()).replace(str(Path.home()), "~", 1)
            message = (f"FATAL: Claude Code cannot start an agent with the account in {where}: {login}. "
                       f"{claude_login_fix(account)[0].upper()}{claude_login_fix(account)[1:]}, then launch again.")
            print(message, file=sys.stderr, flush=True)
            if run_dir is not None:
                from ...agent.agent_session import record_launch_failure
                record_launch_failure(Path(run_dir), reason="claude_not_logged_in", detail=login, message=message,
                                      fix=[claude_login_fix(account)])
            return 2
        problem = claude_sandbox_problem()
        if problem is None:
            if allow_unsandboxed():
                print("[cli] --allow-unsandboxed: Claude Code's sandbox works here, so it stays on", file=sys.stderr,
                      flush=True)
            return None
        if allow_unsandboxed():
            self._unsandboxed = problem
            print(f"WARNING: running Claude Code without its sandbox (--allow-unsandboxed): it cannot start here "
                  f"({problem}). The agent's commands can read and write anywhere you can.", file=sys.stderr,
                  flush=True)
            self.launch_meta = {"unsandboxed": {"agent": "Claude Code", "reason": problem}}
            return None
        missing = problem.endswith("not on PATH")
        steps = claude_userns_fixes()
        message = (f"FATAL: Claude Code's Bash sandbox cannot start here: {problem}. "
                   + ("Install bubblewrap and socat with `python -m dycraft.install tools` (or your distribution's "
                      "`sudo apt-get install bubblewrap socat`)" if missing else
                      "With root: " + "; ".join(steps))
                   + ", then confirm with `python -m dycraft.install check`. Or, without sudo, pass "
                     "--allow-unsandboxed to run Claude Code without its sandbox: the agent's commands can then "
                     "read and write anywhere you can. DYCRAFT_SKIP_PREFLIGHT=1 launches without this check.")
        print(message, file=sys.stderr, flush=True)
        if run_dir is None:
            return 2
        from ...agent.agent_session import record_launch_failure
        record_launch_failure(Path(run_dir), reason="claude_sandbox_unavailable", detail=problem, message=message,
                              fix=[*(["`python -m dycraft.install tools`, or your distribution's `sudo apt-get "
                                      "install bubblewrap socat`"] if missing else steps),
                                   "then `python -m dycraft.install check`",
                                   "without sudo: `--allow-unsandboxed` runs Claude Code without its sandbox, so "
                                   "its commands can read and write anywhere you can"])
        return 2

    def execute(self, plan: LaunchPlan) -> int:
        env = agent_shell_env(dict(plan.env), plan.run_dir)
        # A launch from a shell Claude Code started inherits its session markers. The agent is a session of its own,
        # and with CLAUDE_CODE_CHILD_SESSION set Claude Code saves no transcript, which the session watch reads.
        for marker in CLAUDE_SESSION_MARKERS:
            env.pop(marker, None)
        executable, config_dir = self._resolve_executable()
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir
        if shutil.which(executable) is None:
            print(f"FATAL: `{executable}` not on PATH.", file=sys.stderr)
            return 2
        permission_mode = os.environ.get("DYCRAFT_CLAUDE_PERMISSION_MODE",
                                         self.DEFAULT_PERMISSION_MODE)
        # `//<abs>` is Claude Code's permission-rule syntax for an absolute
        # path in Edit/Read/Write rules (single-slash is project-relative).
        run_dir_abs = str(plan.run_dir)
        sandbox_settings = json.dumps(claude_agent_settings(run_dir_abs, sandboxed=self._unsandboxed is None))
        cmd = [
            executable,
            "--add-dir",              str(plan.run_dir),
            "--permission-mode",      permission_mode,
            "--settings",             sandbox_settings,
            # With no `--mcp-config`, this means "no MCP servers": the agent
            # does not inherit the user's own.
            "--strict-mcp-config",
            "--append-system-prompt", plan.system_prompt_text,
            "--name",                 f"dycraft:{plan.run_id}",
        ]
        # Always interactive mode -- never `--print`, for the run and for every
        # sub-agent alike. Interactive does not need a terminal: with stdin at
        # EOF claude runs one full agentic turn (any number of tool calls),
        # exits, and keeps banners on stderr so stdout is the bare answer.
        if plan.model:
            cmd += ["--model", plan.model]
        if plan.reasoning_effort:
            # claude's `--effort` accepts low|medium|high|xhigh|max (see
            # `claude --help`). Forwarded verbatim — invalid values surface
            # as a claude-side error rather than a silent DyCraft override.
            cmd += ["--effort", plan.reasoning_effort]
        resume = plan.resume_session
        if resume:
            # Same flags as the first launch, so the resumed session gets the
            # same sandbox, permissions and system prompt it was started with;
            # claude looks the id up under this cwd's project key, which is why
            # the chdir below must land on the run dir as first launched.
            cmd += ["--resume", resume]
        # `--` so an initial-msg starting with `---` (markdown HR from the
        # footer) isn't mistaken for a flag.
        cmd += ["--", plan.footer_text]

        print(f"[claude_{plan.layer}] run_dir={plan.run_dir}", file=sys.stderr)
        print(f"[claude_{plan.layer}] launching claude "
              f"(permission-mode={permission_mode}, sandbox=on, "
              f"bash-writable={plan.run_dir}, "
              f"edit/write-allow=//{run_dir_abs}/**, "
              f"model={plan.model}, effort={plan.reasoning_effort})",
              file=sys.stderr)
        # The argv string is the audit trail for "did model + reasoning
        # actually reach the agent CLI". Logged once with the footer
        # truncated (its full body is the user prompt + harness preamble,
        # already captured in meta.json + the prompt file). Tests parse
        # this line out of captured stderr to verify pass-through.
        print(f"[claude_{plan.layer}] "
              f"argv={_fmt_cmd_for_log(cmd)}",
              file=sys.stderr, flush=True)

        # chdir to run_dir so claude's sandbox cwd-tree default matches the
        # one path in `allowWrite`. Without this, the implicit cwd-tree
        # default would broaden the writable set to the launcher's cwd, which
        # settings cannot disable.
        os.chdir(plan.run_dir)
        # Headless means one turn and done. Without a terminal on stdout there
        # is nobody to type a next message, but a stdin still attached to one
        # (a `RUN.sh` started from a shell, its stdout sent to a log) could
        # leave claude waiting on it for the rest of its timeout. EOF on stdin
        # is what ends the turn cleanly; with a terminal on stdout it is left
        # alone and the TUI is the operator's.
        if not _process._fd_is_tty(1):
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
        watch = ClaudeSessionWatch(
            Path(env.get("CLAUDE_CONFIG_DIR") or "~/.claude"), plan.run_dir, interactive=_process._fd_is_tty(1))
        from ... import _oom
        oom_before = _oom.oom_kills()
        status = _process.run_agent_cli(cmd, env, watch=watch)
        return _process.report_agent_kill(status, plan.run_dir, oom_before)


def _claude_account_backend(slot: int) -> type[ClaudeBackend]:
    """Build `ClaudeNBackend`: `--backend claudeN` == `--backend claude` with
    `DYCRAFT_CLAUDE_EXECUTABLE=claudeN` (CLAUDE_CONFIG_DIR=~/.claudeN).

    Keeps `name = "claude"` (inherited) so anything that conditions on the
    backend name still treats it as a claude backend; only the executable is
    pinned. The `claudeN` selector is the BACKENDS dict key below.

    Generated rather than hand-written because the slots differ in exactly one
    character: a typed-out set is where `Claude7Backend` ends up pinned to
    `~/.claude6`, and every test that only asserts *some* account got pinned
    would still pass. Here the slot number cannot disagree with itself. A slot
    sets only its docstring and `FORCED_EXECUTABLE`; everything else is
    inherited.

    The result is reachable as `BACKENDS["claudeN"]`, not as a module-level
    `ClaudeNBackend`: nothing imports these by name, and binding nine names
    just to keep that door open is a second thing to maintain. The identity
    still reads as the slot in a traceback or `repr`, which is what the name
    was load-bearing for.

    `__module__` is set explicitly: `Backend` is an ABC, so the class is
    actually built inside `abc.py` and picks up *that* frame's module name,
    giving `<class 'abc.Claude7Backend'>` — a repr that points at the standard
    library for a class that lives here.
    """
    return type(f"Claude{slot}Backend", (ClaudeBackend,), {
        "__module__": __name__,
        "__doc__": (f'`--backend claude{slot}` == `--backend claude` with '
                    f'`DYCRAFT_CLAUDE_EXECUTABLE=claude{slot}` '
                    f'(CLAUDE_CONFIG_DIR=~/.claude{slot}). Inherits '
                    f'`name = "claude"`; only the executable is pinned.'),
        "FORCED_EXECUTABLE": f"claude{slot}",
    })
