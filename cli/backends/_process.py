"""Running and stopping the agent CLI process.

`run_agent_cli` runs it as the launcher's child; `_spawn_with_pty_term_shim`
runs Codex under a pseudo-terminal when the launcher has none, watching its
screen and rollout for states it cannot leave on its own. `stop_agent_cli`
stops whichever of the two is running, and `stop_status` reports the status a
stop has set. Their state lives in this module only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


# Terminal-capability probe → canned reply. codex's TUI emits these at startup
# (cursor-position report, primary device attributes, OSC fg/bg color queries,
# kitty-keyboard flags) and BLOCKS until a terminal answers. On the bare PTY we
# wrap it in, nothing answers, so codex deadlocks during startup and never
# reaches its first tool call. Answering with minimal canned replies lets it
# finish the handshake and reach the agent loop.
_TERM_PROBE_REPLIES: tuple[tuple[bytes, bytes], ...] = (
    (b"\x1b[6n",   b"\x1b[40;1R"),                        # DSR cursor position
    (b"\x1b[c",    b"\x1b[?1;2c"),                         # DA1 (VT100 + AVO)
    (b"\x1b]10;?", b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\"),   # OSC-10 fg color
    (b"\x1b]11;?", b"\x1b]11;rgb:0000/0000/0000\x1b\\"),   # OSC-11 bg color
    (b"\x1b[?u",   b"\x1b[?0u"),                           # kitty-keyboard flags
)


#: Exit status when codex reported a fatal error and then refused to exit.
#: Distinct so a caller can name the failure instead of guessing at a generic 1.
EXIT_CODEX_FATAL = 86


#: Exit status when codex stopped on an interactive startup screen, or never
#: started a session at all. The shim only runs when nobody is at a terminal,
#: so such a screen can never be answered; waiting on it only spends the
#: caller's whole wall-clock bound.
EXIT_CODEX_BLOCKED = 87


#: The same two outcomes for any agent CLI: a fatal error it reported (a usage
#: limit, no login) and a start it never finished. The claude session watch
#: (`claude.ClaudeSessionWatch`) exits with these as codex's watchdogs do.
EXIT_AGENT_FATAL = EXIT_CODEX_FATAL
EXIT_AGENT_BLOCKED = EXIT_CODEX_BLOCKED


#: Text of the startup screens that wait for a keypress, as it reads once
#: escape codes are stripped and whitespace collapsed. Checked on 0.153.4.
_BLOCKING_SCREENS: tuple[tuple[str, str], ...] = (
    ("Do you trust the contents of this directory", "trust this directory"),
    ("Update available!", "update available"),
    # A damaged `state_*.sqlite` in the shared `$CODEX_HOME`: codex says so,
    # moves it aside, rebuilds it and waits for Enter, which no one answers.
    ("its local database appears to be damaged", "damaged local database"),
    ("Codex rebuilt its local database", "rebuilt local database"),
)


#: Startup screens the shim answers itself, as `(text, keys, what)`: Codex 0.167 asks how to open a folder saved as
#: untrusted (`mark_codex_untrusted`) when approvals are not `never` (an organization that allows only on-request),
#: and its first choice, "Open restricted", keeps it untrusted ("Opening will not change saved trust"), which is
#: what the entry is for. `1` highlights it (codex-rs/tui/src/onboarding/trust_directory.rs), and Enter, pressed
#: shortly after, confirms it: Codex redraws only what changed, so the screen is not drawn again to wait for.
_ANSWERED_SCREENS: tuple[tuple[str, bytes, str], ...] = (
    ("Opening will not change saved trust", b"1", "open the untrusted run directory restricted"),
)

#: How long after an answer's keys Enter confirms it.
_ANSWERED_SCREEN_ENTER_SECONDS = 0.5
#: Enter again, this often and this many times, while no session has opened since the answer: a key that came before
#: the screen took input is lost, and Enter on Codex's empty composer does nothing.
_ANSWERED_SCREEN_RETRY_SECONDS = 3.0
_ANSWERED_SCREEN_RETRIES = 3

#: How long a blocking screen may stand before the child is stopped.
_BLOCKING_SCREEN_GRACE_SECONDS = 5.0


#: The scan stops once codex has opened its session rollout -- past its startup
#: screens by definition -- or after this long, whichever is first. Past that
#: point the same words can only be content: an agent `cat`-ing this very file
#: would otherwise be killed for displaying them.
_BLOCKING_SCREEN_SCAN_SECONDS = 180.0


def _record_codex_stop(watch, reason: str, detail: str) -> None:
    """`meta.json`'s `agent_stop` for a codex the PTY loop stopped, in its run dir (the `-C` the watch read)."""
    if watch.cwd:
        from ...agent.agent_session import record_agent_stop
        record_agent_stop(Path(watch.cwd), reason=reason, detail=detail)


def iso_seconds(value) -> float | None:
    """An ISO 8601 timestamp, as session records carry one (`2026-11-09T02:25:26.635Z`), in epoch seconds."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _made_a_harness_call(run_dir: Path) -> bool:
    """Whether the run's trace holds a harness call: proof the agent ran, for a rollout the watch never found."""
    try:
        with open(run_dir / "trace.jsonl", "rb") as fh:
            return any(b'"op_call"' in line for line in fh)
    except OSError:
        return False


def _last_screen_line(screen: str, width: int = 160) -> str:
    """The end of what the TUI last drew, for a message: printable text only, at most `width` characters."""
    text = "".join(ch for ch in screen if ch.isprintable()).strip()
    return text[-width:].strip()


#: A text style escape (SGR), which changes how letters look but never moves the cursor.
_STYLE = re.compile(rb"\x1b\[[0-9;]*m")


def _screen_text(data: bytes) -> str:
    """PTY output as plain text: escape sequences out, whitespace collapsed."""
    # A CSI's parameter bytes are 0x30-0x3F, `<`, `=` and `>` included (codex sends `\x1b[<1u`, `\x1b[>4;0m`).
    text = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b.",
                  b" ", data).decode("utf-8", "replace")
    return re.sub(r"\s+", " ", text)


#: How often the pump re-reads the rollout. Cheap: an incremental read from a
#: saved offset, and the file only grows.
_ROLLOUT_POLL_SECONDS = 5.0


#: How long a fatal record must stand as the *last* thing in the rollout before
#: the child is killed.
#:
#: Not zero, because the error is not always terminal: an `unauthorized` error
#: can be followed by a token refresh that resumes the session within seconds,
#: and three minutes clears that with a wide margin. A `usage_limit_exceeded`
#: that only clears when someone refills the account hours later is let go on
#: purpose: an unattended run should not hold its slot for it.
_ROLLOUT_FATAL_GRACE_SECONDS = 180.0


#: Stop looking for the rollout after this long. A session that has not opened
#: one by now is failing some other way, and the scan is the only part of this
#: watch that costs anything. If it has not reached the harness either, it is
#: stuck at startup (`_CodexRolloutWatch.stalled_at_startup`) and is stopped.
#:
#: Codex opens its rollout within seconds of starting (at most 3.3 s over 289
#: recorded launches), so 30 s, about nine times that, keeps room for a slow node
#: or a cold cache and stops a wedged start without minutes of waiting.
_ROLLOUT_DISCOVERY_SECONDS = 30.0


#: Records that only the model writes: a new turn, its output, a call it makes.
#: Only these show a session recovered from a fatal error. A command the dead
#: turn left running still completes and is recorded (`item_completed`), and so
#: are token counts, so those records arriving after a fatal error do not mean
#: the session recovered.
_MODEL_RECORDS = frozenset({"turn_context"})


_MODEL_EVENTS = frozenset({"task_started", "user_message", "agent_message",
                           "agent_reasoning"})


_MODEL_ITEMS = frozenset({"message", "reasoning", "function_call", "custom_tool_call",
                          "local_shell_call", "web_search_call"})


def _model_is_working(rec: dict) -> bool:
    """Whether a rollout record shows the model at work (`_MODEL_RECORDS`)."""
    kind = rec.get("type")
    if kind in _MODEL_RECORDS:
        return True
    payload = rec.get("payload") or {}
    if not isinstance(payload, dict):
        return False
    if kind == "event_msg":
        return payload.get("type") in _MODEL_EVENTS
    if kind == "response_item":
        return payload.get("type") in _MODEL_ITEMS
    return False


#: What codex 0.153's TUI shows once it gives up a turn it could not stream: after five reconnects it prints this
#: and goes back to its input prompt, waiting for a person, with nothing in the rollout to say so.
_LOST_STREAM = "Stream disconnected before completion"

#: How long that may stand, with no model output in the turn, before the CLI is stopped.
_LOST_STREAM_GRACE_SECONDS = 60.0

#: codex 0.153's status line while it retries a model connection it lost, as its first frame draws it: "Reconnecting...
#: waiting for network (3s • esc to interrupt)". It retries without end, and records nothing in the rollout meanwhile.
_RECONNECTING = re.compile(r"Reconnecting\.\.\.[^(]{0,60}\(\d+[smh][^)]{0,30}esc to interrupt\)")

#: How long codex may sit reconnecting, with no new rollout record, before the CLI is stopped. Longer than a network
#: blip; the run can be resumed once the network is back.
_RECONNECT_BOUND_SECONDS = 300.0


def _model_answered(rec: dict) -> bool:
    """Whether a rollout record is the model's own output: an assistant message, reasoning, a tool call, or the
    token count a finished response reports. (User and developer messages are `message` items too.)"""
    payload = rec.get("payload") or {}
    if not isinstance(payload, dict):
        return False
    if rec.get("type") == "event_msg":
        return payload.get("type") in ("token_count", "agent_message")
    if rec.get("type") == "response_item":
        kind = payload.get("type")
        return kind in _MODEL_ITEMS - {"message"} or (kind == "message" and payload.get("role") == "assistant")
    return False


#: How the fatal errors codex records read in a message, by their `codex_error_info`.
_CODEX_FATAL_ERRORS = {
    "usage_limit_exceeded": "codex hit its usage limit",
    "unauthorized": "codex's login was refused",
}


def _rollout_fatal_reason(rec: dict) -> str | None:
    """The error a `task_complete` record carries, if it is a fatal one.

    Codex's TUI prints this and then stops answering — it does not exit, and
    the pump loop below cannot tell "printed an error and gave up" from
    "thinking" by watching the mirrored stream. The rollout says so
    structurally, which is why this reads the file rather than scraping the
    terminal: a text match on mirrored output can fire on the agent's own
    words, and this cannot.

    Requires `last_agent_message` to be absent: a `task_complete` that carries
    an answer is a turn that finished, whatever else it also recorded.
    """
    if rec.get("type") != "event_msg":
        return None
    payload = rec.get("payload") or {}
    if payload.get("type") != "task_complete":
        return None
    if payload.get("last_agent_message"):
        return None
    err = payload.get("error") or {}
    info = err.get("codex_error_info")
    if not info:
        return None
    return f"{info}: {err.get('message') or 'no message'}"


class _CodexRolloutWatch:
    """Watches this launch's codex rollout for a fatal error it will sit on.

    codex records its own session as JSONL under `$CODEX_HOME/sessions/`, keyed
    to the working directory it was launched in. `CodexBackend` always launches
    with `-C <run_dir>`, so that directory identifies the session exactly.

    The failure being caught: pointed at a credit-exhausted account, the TUI
    prints the error, stops answering terminal probes, and never exits, so
    `dycraft.cli` would hold it open until the run's time limit.

    Arm-and-wait rather than kill-on-sight: see `_ROLLOUT_FATAL_GRACE_SECONDS`.
    A record of the model working again (`_model_is_working`) disarms the
    watch, because that session recovered. Other records do not: a command
    the failed turn left running is still recorded when it completes.
    """

    def __init__(self, cmd: list[str], env: dict) -> None:
        self.cwd = ""
        for i, a in enumerate(cmd):
            if a == "-C" and i + 1 < len(cmd):
                self.cwd = str(Path(str(cmd[i + 1])).resolve())
                break
        home = env.get("CODEX_HOME") or os.environ.get("CODEX_HOME") or ""
        self.home = Path(home).expanduser() if home else None
        self.started = time.monotonic()
        self.wall_started = time.time()
        #: `codex resume <id>` appends to the session's own rollout, whose earlier records are the last launch's.
        self.resumed = len(cmd) > 1 and cmd[1] == "resume"
        self.path: Path | None = None
        self.offset = 0
        self.armed_at: float | None = None
        self.reason: str | None = None
        self._next_poll = self.started
        #: Set once `_ROLLOUT_DISCOVERY_SECONDS` passed with no rollout found.
        self.never_found = False
        #: The approval policy the session runs under, from its first `turn_context`; None until seen.
        self.approval_policy: str | None = None
        #: Whether the model has produced any output since the current turn started.
        self.turn_answered = False
        #: Whether the model has produced any output in this session, and when (wall clock) the rollout was last
        #: written: its mtime, not when this watch read it, which can be a poll later.
        self.ever_answered = False
        self.last_record = 0.0
        self.asked_never = any(a == "--ask-for-approval" and b == "never" for a, b in zip(cmd, cmd[1:]))
        self.enabled = bool(
            self.cwd and self.home
            and not os.environ.get("DYCRAFT_CODEX_NO_ROLLOUT_WATCH"))

    def _find(self) -> Path | None:
        """The rollout whose `session_meta` names our run dir.

        Newest-first, and only files touched since we launched: the sessions
        tree holds one file per codex session ever run on this machine, and the
        one we want was created seconds ago. A sub-thread carries
        `parent_thread_id` and is skipped — it shares our cwd but is not the
        session that would hang.
        """
        if self.home is None or not (self.home / "sessions").is_dir():
            return None
        wall_start = time.time() - (time.monotonic() - self.started) - 60.0
        best: list = []
        try:
            for p in self.home.glob("sessions/**/rollout-*.jsonl"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_mtime >= wall_start:
                    best.append((st.st_mtime, p))
        except OSError:
            return None
        for _, p in sorted(best, reverse=True):
            try:
                with p.open(encoding="utf-8", errors="replace") as fh:
                    head = json.loads(fh.readline() or "{}")
            except (OSError, ValueError):
                continue
            if head.get("type") != "session_meta":
                continue
            payload = head.get("payload") or {}
            if payload.get("parent_thread_id"):
                continue
            if str(payload.get("cwd") or "") == self.cwd:
                return p
        return None

    def _note_approval_policy(self, context: dict) -> None:
        """Record the policy the session runs under in `meta.json` (`codex_sandbox.approval_effective`), and warn
        when `never` was overridden."""
        policy = context.get("approval_policy")
        self.approval_policy = policy if isinstance(policy, str) else json.dumps(policy)
        if self.asked_never and self.approval_policy != "never":
            print(f"[codex] the session runs with `{self.approval_policy}` approvals, not `never` (an "
                  "organization policy): Codex's automatic reviewer can approve a command outside the sandbox.",
                  file=sys.stderr, flush=True)
        path = Path(self.cwd) / "meta.json"
        try:
            meta = json.loads(path.read_text())
            meta.setdefault("codex_sandbox", {})["approval_effective"] = self.approval_policy
            tmp = path.with_name(".meta.json.codex.tmp")
            tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
            os.replace(tmp, path)
        except (OSError, ValueError, AttributeError):
            pass

    def stalled_at_startup(self) -> bool:
        """No rollout in `_ROLLOUT_DISCOVERY_SECONDS`, and no harness call in
        the run either: codex never started a session, so it is stuck on
        something at startup, a screen `_BLOCKING_SCREENS` does not know or
        none at all. The harness call is the check against a rollout this
        watch merely failed to find, which would otherwise stop healthy runs.
        """
        if not self.never_found:
            return False
        self.never_found = False                     # decided once, either way
        return not _made_a_harness_call(Path(self.cwd))

    def poll(self) -> str | None:
        """The reason to kill the child, or None. Never raises."""
        if not self.enabled:
            return None
        now = time.monotonic()
        if now < self._next_poll:
            return None
        self._next_poll = now + _ROLLOUT_POLL_SECONDS
        try:
            if self.path is None:
                if now - self.started > _ROLLOUT_DISCOVERY_SECONDS:
                    self.enabled = False
                    self.never_found = True
                    return None
                self.path = self._find()
                if self.path is None:
                    return None
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
                self.offset = fh.tell()
                if chunk:
                    self.last_record = os.fstat(fh.fileno()).st_mtime
            # A trailing partial line is re-read next poll: rewind to the last
            # newline so a half-written record is never parsed.
            cut = chunk.rfind(b"\n")
            if cut == -1:
                self.offset -= len(chunk)
                chunk = b""
            else:
                self.offset -= len(chunk) - (cut + 1)
                chunk = chunk[:cut + 1]
            for line in chunk.splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                stamp = iso_seconds(rec.get("timestamp")) if self.resumed else None
                if stamp is not None and stamp < self.wall_started - 1:
                    continue                       # an earlier launch's record, in a resumed session's file
                if self.approval_policy is None and rec.get("type") == "turn_context":
                    self._note_approval_policy(rec.get("payload") or {})
                if rec.get("type") == "event_msg" and (rec.get("payload") or {}).get("type") == "task_started":
                    self.turn_answered = False
                elif _model_answered(rec):
                    self.turn_answered = self.ever_answered = True
                why = _rollout_fatal_reason(rec)
                if why:
                    self.armed_at, self.reason = now, why
                elif self.armed_at is not None and _model_is_working(rec):
                    self.armed_at, self.reason = None, None
        except OSError:
            return None
        if self.armed_at is not None and now - self.armed_at >= (
                _ROLLOUT_FATAL_GRACE_SECONDS):
            return self.reason
        return None


def _spawn_with_pty_term_shim(cmd: list[str], env: dict) -> int:
    """Run `cmd` on a PTY, mirroring its output and answering the startup
    terminal-capability probes codex blocks on (a `pty.spawn` replacement).

    Forks the child onto a PTY slave, copies the master's output to our stdout,
    and writes canned replies whenever a probe escape appears in that output.
    Sets a window size so the TUI has sensible geometry. Returns the child's
    exit status mapped like `pty.spawn` (128 + signal when signaled).
    """
    import fcntl
    import pty
    import select
    import struct
    import termios

    global _pty_child_pid, _stop_status
    _stop_status = None
    pid, master_fd = pty.fork()
    if pid == 0:
        # Child: replace with codex on the PTY slave (stdin/out/err wired by
        # pty.fork). os.environ was already updated by the caller; pass env
        # explicitly too so a fork-time copy isn't relied on.
        #
        # Force the slave to become our CONTROLLING terminal. `pty.fork`
        # already setsid()s us, but it does not guarantee the slave is acquired
        # as the controlling tty — and when the launcher was started from a
        # session with no controlling terminal (a harness daemon runs its
        # children under `start_new_session=True`, i.e. setsid), it isn't.
        # codex then comes up on a PTY with no controlling terminal and
        # deadlocks at startup, before creating a session: every
        # daemon-dispatched `codex` sub-agent would hang. TIOCSCTTY makes the
        # slave our controlling tty so codex's TUI has the terminal it waits
        # on. EPERM/ENOTTY are non-fatal.
        try:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        except OSError:
            pass
        try:
            os.execvpe(cmd[0], cmd, env)
        except OSError:
            os._exit(127)

    # Parent: give the TUI a real-looking geometry, then pump + answer probes.
    _pty_child_pid = pid
    try:
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ,
                    struct.pack("HHHH", 40, 120, 0, 0))
    except OSError:
        pass
    watch = _CodexRolloutWatch(cmd, env)
    fatal: str | None = None
    blocked: str | None = None
    blocked_since: float | None = None
    stopped_blocked = False
    stopped_lost = False
    lost_since: float | None = None
    screen_tail = ""
    late_tail = ""                                 # what the TUI drew lately, for a turn it gave up (`_LOST_STREAM`)
    reconnecting_since: float | None = None        # when (wall clock) codex drew `_RECONNECTING`
    reconnecting = carry = ""
    answered: dict[str, float] = {}                # screens the shim answered (`_ANSWERED_SCREENS`), and when
    entered: dict[str, tuple[int, float]] = {}     # how many Enters followed each answer, and when the last was
    try:
        while True:
            try:
                # A short wait while an answer awaits its first Enter, so it follows the keys promptly.
                ready, _, _ = select.select([master_fd], [], [], 0.2 if len(entered) < len(answered) else 1.0)
            except (OSError, ValueError):
                break
            # A keypress screen at startup: nobody is at this PTY to answer it.
            # Every stop below prints its FATAL on a line of its own: codex's screen leaves the cursor mid-line.
            if blocked_since is not None and \
                    time.monotonic() - blocked_since > _BLOCKING_SCREEN_GRACE_SECONDS:
                print(f"\nFATAL: codex stopped at startup on its '{blocked}' screen, which "
                      f"nobody can answer in a headless launch; the CLI is stopped instead "
                      f"of waiting out the run's time limit. Answer it once in a terminal "
                      f"(`CODEX_HOME={watch.home or '~/.codex'} codex` in {watch.cwd or 'the run directory'}), "
                      "then launch again.",
                      file=sys.stderr, flush=True)
                _record_codex_stop(watch, "codex_blocked_screen", blocked)
                _terminate_child(pid)
                stopped_blocked = True
                break
            # Checked every pass, not only when there is output — the whole
            # point is that a wedged codex produces neither output nor an exit.
            fatal = watch.poll()
            if not fatal and watch.path is not None and not watch.turn_answered and _LOST_STREAM in late_tail:
                lost_since = lost_since or time.monotonic()
                if time.monotonic() - lost_since > _LOST_STREAM_GRACE_SECONDS:
                    at = late_tail.rfind(_LOST_STREAM)
                    said = late_tail[at:at + 160].split("›")[0].strip()
                    print(f"\nFATAL: codex could not reach its model and gave up the turn ({said!r}); it is waiting "
                          "for input nobody will give. The CLI is stopped. Check this machine's network and proxy "
                          "settings. DYCRAFT_CODEX_NO_ROLLOUT_WATCH=1 disables this watchdog.",
                          file=sys.stderr, flush=True)
                    _record_codex_stop(watch, "codex_lost_stream", said)
                    _terminate_child(pid)
                    stopped_lost = True
                    break
            elif watch.turn_answered:
                lost_since = None
            if reconnecting_since is not None and watch.last_record > reconnecting_since:
                reconnecting_since = None              # the session moved on: the connection came back
            if not fatal and reconnecting_since is not None and \
                    time.time() - reconnecting_since > _RECONNECT_BOUND_SECONDS:
                print(f"\nFATAL: codex has been reconnecting to its model for {_RECONNECT_BOUND_SECONDS:g} s "
                      f"({reconnecting!r}) and its session recorded nothing meanwhile: this machine cannot reach "
                      "the model. The CLI is stopped. Check the network and proxy settings. "
                      "DYCRAFT_CODEX_NO_ROLLOUT_WATCH=1 disables this watchdog.",
                      file=sys.stderr, flush=True)
                _record_codex_stop(watch, "codex_reconnecting", reconnecting)
                _terminate_child(pid)
                stopped_lost = True
                break
            if not fatal and watch.stalled_at_startup():
                last = _last_screen_line(screen_tail)
                print(f"\nFATAL: codex opened no session in {_ROLLOUT_DISCOVERY_SECONDS:g} s after it "
                      "started and made no harness call: it is stuck before its first turn"
                      + (f" (its screen last showed: {last!r})" if last else "")
                      + ". The CLI is stopped instead of waiting out the run's time limit. "
                      "DYCRAFT_CODEX_NO_ROLLOUT_WATCH=1 disables this watchdog.",
                      file=sys.stderr, flush=True)
                _record_codex_stop(watch, "codex_never_started", last or (
                    f"no screen after answering its startup screen ({', '.join(answered)})" if answered
                    else "no screen"))
                _terminate_child(pid)
                stopped_blocked = True
                break
            if fatal:
                kind, _, said = fatal.partition(": ")
                what = _CODEX_FATAL_ERRORS.get(kind, f"codex's session recorded a fatal error ({kind})")
                print(f"\nFATAL: {what}: {said.rstrip('. ') or 'no message'}. No activity followed for "
                      f"{_ROLLOUT_FATAL_GRACE_SECONDS:g} s, so the CLI is stopped. "
                      "DYCRAFT_CODEX_NO_ROLLOUT_WATCH=1 disables this watchdog.",
                      file=sys.stderr, flush=True)
                _record_codex_stop(watch, "codex_fatal_error", fatal)
                _terminate_child(pid)
                break
            # Enter confirms an answered screen's highlighted choice; again, a few times, while no session opens.
            for _marker, _keys, what in _ANSWERED_SCREENS:
                count, last_enter = entered.get(what, (0, answered.get(what, 0.0)))
                due = (_ANSWERED_SCREEN_ENTER_SECONDS if count == 0 else _ANSWERED_SCREEN_RETRY_SECONDS)
                if what in answered and watch.path is None and count <= _ANSWERED_SCREEN_RETRIES and \
                        time.monotonic() - last_enter > due:
                    try:
                        os.write(master_fd, b"\r")
                    except OSError:
                        pass
                    entered[what] = (count + 1, time.monotonic())
            if not ready:
                continue
            try:
                data = os.read(master_fd, 65536)
            except OSError:
                break  # PTY closed (child exited / was SIGTERM'd)
            if not data:
                break
            try:
                os.write(1, data)  # mirror codex's output to our stdout
            except OSError:
                pass
            if watch.enabled:
                # Styles out first, without a space: codex restyles single letters of its status line.
                fresh = _screen_text(_STYLE.sub(b"", data))
                # The last read's end comes along, for a status line split across reads, but not one already seen.
                found = _RECONNECTING.search(carry + fresh)
                carry = "" if found else fresh[-120:]
                if found and reconnecting_since is None:
                    reconnecting_since, reconnecting = time.time(), found.group(0)
                late_tail = (late_tail + fresh)[-4000:]
            if blocked_since is None and watch.path is None and \
                    time.monotonic() - watch.started < _BLOCKING_SCREEN_SCAN_SECONDS:
                screen_tail = (screen_tail + _screen_text(data))[-8000:]
                for marker, name in _BLOCKING_SCREENS:
                    if marker in screen_tail:
                        blocked, blocked_since = name, time.monotonic()
                        break
                for marker, keys, what in _ANSWERED_SCREENS:
                    if marker in screen_tail and what not in answered:
                        try:
                            os.write(master_fd, keys)
                        except OSError:
                            pass
                        answered[what] = time.monotonic()
                        print(f"\n[codex] answered codex's startup screen: {what}", file=sys.stderr, flush=True)
                        # Only what is drawn after the answer can show the screen still standing.
                        screen_tail = ""
            reply = b"".join(rep for probe, rep in _TERM_PROBE_REPLIES
                             if probe in data)
            if reply:
                try:
                    os.write(master_fd, reply)
                except OSError:
                    pass
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    if fatal:
        # Returned ahead of any wait: `_terminate_child` has already reaped the
        # child, so a blocking `waitpid` here would raise ECHILD. The exit
        # status would say "SIGTERM" in any case, which is not the reason
        # anyone needs to read in a run log.
        _pty_child_pid = None
        return EXIT_CODEX_FATAL
    if stopped_blocked:
        _pty_child_pid = None
        return EXIT_CODEX_BLOCKED                  # reaped, as above
    if stopped_lost:
        _pty_child_pid = None
        return EXIT_CODEX_FATAL                    # reaped, as above
    _, status = os.waitpid(pid, 0)
    _pty_child_pid = None
    if _stop_status is not None:                   # asked to stop (`stop_agent_cli`)
        return _stop_status
    code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else \
        128 + os.WTERMSIG(status) if os.WIFSIGNALED(status) else 1
    if code not in (0, SIGKILLED) and watch.cwd and watch.home and not os.environ.get("DYCRAFT_CODEX_NO_ROLLOUT_WATCH"):
        watch._next_poll = 0.0                     # read the rollout to its end: an answer may be seconds old
        watch.poll()
        if not watch.ever_answered and not _made_a_harness_call(Path(watch.cwd)):
            # codex prints why (a configuration it could not load, a login) on its own screen and exits.
            last = _last_screen_line(late_tail or screen_tail)
            print(f"\nFATAL: codex exited (status {code}) before it answered"
                  + (f"; its screen last showed: {last!r}" if last else "") + ".",
                  file=sys.stderr, flush=True)
            _record_codex_stop(watch, "codex_exited_before_answering", last or f"status {code}")
            return EXIT_CODEX_FATAL
    return code


def _descendants(pid: int) -> list[int]:
    """The processes `pid` started, and theirs, read from /proc."""
    children: dict[int, list[int]] = {}
    try:
        entries = [e.name for e in os.scandir("/proc") if e.name.isdigit()]
    except OSError:
        return []
    for name in entries:
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                stat = fh.read()
            # The command name, in parentheses, may hold spaces and parentheses: the fields after the last ')' are
            # the state, then the parent's pid.
            ppid = int(stat[stat.rindex(b")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(name))
    out, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            out.append(child)
            stack.append(child)
    return out


def _signal_cli(pid: int, sig) -> None:
    """Send `sig` to the agent CLI and to every process it started; ProcessLookupError when the CLI is gone.

    A signal to the CLI alone is not enough: Codex runs each command through `codex-linux-sandbox`, which starts
    bwrap with `--die-with-parent` tied to itself, not to codex, so the sandbox and its command outlive a codex that
    exits. Where the CLI leads a process group of its own (the PTY shim's), the group is signalled; the sandbox's
    command, in a session of its own, goes with bwrap. Where the CLI shares the launcher's group (a terminal launch),
    the CLI and its descendants are.
    """
    if os.getpgid(pid) == pid:
        os.killpg(pid, sig)
        return
    tree = _descendants(pid)
    os.kill(pid, sig)
    for child in tree:
        try:
            os.kill(child, sig)
        except ProcessLookupError:
            pass


def _terminate_child(pid: int, grace: float = 10.0) -> None:
    """SIGTERM the child and what it started (`_signal_cli`), then SIGKILL if it ignores it.

    codex is wedged by hypothesis when this is called, so the SIGKILL path is
    the expected one rather than the exception.
    """
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            _signal_cli(pid, sig)
        except OSError:
            return            # already gone
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 5.0)
        while time.monotonic() < deadline:
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except OSError:
                return        # someone else reaped it
            if done:
                return        # reaped here; the caller must not wait again
            time.sleep(0.1)


#: The agent CLI this launcher is waiting on, and the exit status the launcher
#: owes once it has been asked to stop it. See `run_agent_cli`.
_agent_child: "subprocess.Popen | None" = None


_stop_status: int | None = None


#: The codex `_spawn_with_pty_term_shim` is pumping, when that is how the CLI runs.
_pty_child_pid: int | None = None


#: How long the agent CLI has to exit after the SIGTERM it is forwarded.
AGENT_STOP_GRACE_SECONDS = 30.0


def run_agent_cli(cmd: list[str], env: dict, watch=None) -> int:
    """Run the agent CLI as this launcher's child; return the run's exit status.

    `watch`, when given, is polled on a thread while the CLI runs (`watch.poll()`
    every `watch.poll_seconds`): a `(status, message)` it returns is printed and
    the CLI stopped with that status. After the CLI exits, `watch.after_exit()`
    may name a status for a failure the CLI reported and exited on by itself.

    The launcher stays the CLI's parent rather than exec-ing it, because the
    SIGTERM handler `cli.install_self_kill_exit` installs is what tells a
    committed run that stopped itself (exit 0) from a killed one (143). It
    forwards the SIGTERM (`stop_agent_cli`) and exits with the status the
    handler decided once the CLI is gone.

    The CLI keeps the launcher's terminal and process group, so its interface
    works as in a direct launch; Ctrl-C reaches it directly, and the launcher
    ignores its own copy. If the launcher dies without forwarding (SIGKILL),
    the kernel sends the CLI a SIGTERM (`PR_SET_PDEATHSIG`).
    """
    import signal
    import subprocess

    import threading

    global _agent_child, _stop_status
    _stop_status = None
    proc = subprocess.Popen(cmd, env=env, preexec_fn=_parent_death_signal())
    _agent_child = proc
    done = threading.Event()
    if watch is not None:
        threading.Thread(target=_run_watch, args=(watch, done), name="agent-watch", daemon=True).start()
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        returncode = proc.wait()
    finally:
        done.set()
        signal.signal(signal.SIGINT, previous)
        _agent_child = None
    if _stop_status is not None:
        return _stop_status
    if watch is not None:
        status = watch.after_exit(returncode if returncode >= 0 else 128 - returncode)
        if status is not None:
            return status
    return returncode if returncode >= 0 else 128 - returncode


#: Set by whatever starts a sub-run's launcher (`oneoff.run_oneoff`, the object dispatch) to its own pid, so the
#: launcher can tell when that parent is gone (`watch_parent`).
PARENT_PID_ENV = "DYCRAFT_PARENT_PID"

#: Exit status of a sub-run stopped because its parent is gone.
EXIT_PARENT_GONE = 88

#: A CLI's exit status when it died of SIGKILL. `report_agent_kill` names that death (the OOM killer or a `kill -9`),
#: so the "exited before it answered" messages leave it alone.
SIGKILLED = 128 + 9


def report_agent_kill(status: int, run_dir, oom_before: int | None) -> int:
    """Say so, and record it, when the agent CLI died of SIGKILL: by the kernel's OOM killer when this cgroup's
    `oom_kill` count rose while it ran, else by something outside the run. Returns `status`."""
    if status != SIGKILLED:
        return status
    from ... import _oom
    from ...agent.agent_session import record_agent_stop
    after = _oom.oom_kills()
    oom = oom_before is not None and after is not None and after > oom_before
    if oom:
        what = "the kernel's OOM killer (this machine ran out of memory)"
    else:
        what = "SIGKILL from outside the run (the OOM killer when memory ran short, or a `kill -9`)"
    record_agent_stop(run_dir, reason="oom_kill" if oom else "sigkill", detail=what)
    print(f"FATAL: the agent CLI was killed by {what}. The run directory keeps everything this run committed.",
          file=sys.stderr, flush=True)
    return status


def watch_parent(on_gone) -> bool:
    """For a sub-run's launcher: when `$DYCRAFT_PARENT_PID` names this process's parent, watch it on a thread
    and call `on_gone(pid)` once it is gone (this process re-parented). False when there is no parent to watch."""
    import threading
    try:
        parent = int(os.environ.get(PARENT_PID_ENV) or 0)
    except ValueError:
        return False
    if not parent or os.getppid() != parent:
        return False

    def watch():
        while os.getppid() == parent:
            time.sleep(PARENT_POLL_SECONDS)
        on_gone(parent)
    threading.Thread(target=watch, name="parent-watch", daemon=True).start()
    return True


#: How often a sub-run's launcher checks that its parent is still there.
PARENT_POLL_SECONDS = 2.0


def _run_watch(watch, done) -> None:
    """Poll `watch` until the CLI exits or the watch names a reason to stop it."""
    while not done.wait(watch.poll_seconds):
        try:
            verdict = watch.poll()
        except Exception as exc:                     # noqa: BLE001 -- a watch must never take the run down
            print(f"[cli] the agent watch stopped itself: {exc!r}", file=sys.stderr, flush=True)
            return
        if verdict is not None:
            status, message = verdict
            # On a line of its own: a CLI drawing on this terminal leaves the cursor mid-line.
            print("\n" + message, file=sys.stderr, flush=True)
            stop_agent_cli(status)
            return


def stop_agent_cli(status: int) -> bool:
    """Stop the agent CLI `run_agent_cli` is waiting on, and make `status` the
    launcher's exit status, unless a stop is already under way, whose status
    stands. For the SIGTERM handler; returns False when there is no such child,
    and the handler exits on its own.

    Only signals: the handler runs on the main thread, which is inside the
    child's `wait` (or the PTY shim's loop), and returns to it. Also called
    from the time-limit thread (`cli.execute_with_time_limit`).
    """
    import signal
    import threading

    global _stop_status
    if _stop_status is not None:      # a stop is under way: its status stands (a committed run keeps its 0)
        return True
    proc = _agent_child
    if proc is not None and proc.returncode is None:
        pid = proc.pid

        def running() -> bool:
            return _agent_child is proc
    elif _pty_child_pid is not None:
        pid = _pty_child_pid

        def running() -> bool:
            return _pty_child_pid == pid
    else:
        return False
    _stop_status = status
    try:
        _signal_cli(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True

    def escalate():
        if running():
            try:
                _signal_cli(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    timer = threading.Timer(AGENT_STOP_GRACE_SECONDS, escalate)
    timer.daemon = True
    timer.start()
    return True


def _parent_death_signal():
    """A `preexec_fn` asking the kernel to SIGTERM the child if this process
    dies first, or None off Linux. libc is loaded here, before the fork."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    import signal
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None
    pr_set_pdeathsig = 1

    def preexec():
        libc.prctl(pr_set_pdeathsig, int(signal.SIGTERM), 0, 0, 0)
    return preexec


def _fd_is_tty(fd: int) -> bool:
    """Return whether the given stdio fd is a controlling terminal.
    Wrapped so tests can monkeypatch the per-fd helpers below."""
    try:
        return os.isatty(fd)
    except OSError:
        return False


def _stdin_is_tty() -> bool:
    return _fd_is_tty(0)


def _stdout_is_tty() -> bool:
    return _fd_is_tty(1)


def _stderr_is_tty() -> bool:
    return _fd_is_tty(2)


def _stdio_is_tty() -> bool:
    """Codex's interactive CLI expects a terminal on its standard streams.
    Nested harness launches often pipe stdout/stderr even when stdin still
    points at a tty, so treat any non-tty stdio fd as requiring PTY wrap."""
    return _stdin_is_tty() and _stdout_is_tty() and _stderr_is_tty()


def stop_status() -> int | None:
    """The exit status a stop (`stop_agent_cli`) has set for the running agent
    CLI, or None while nothing has stopped it."""
    return _stop_status
