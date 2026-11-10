"""One-off CLI sub-agent calls: a text prompt in, a text answer out.

`run_oneoff` hands a CLI sub-agent (`claude` / `codex`) a text prompt and reads
its answer back, on the same launch path (daemon, harness, PTY-wrapped CLI,
sandbox, staged home, no-progress watchdog) as every other run. The answer
channel is the `plain` layer's `submit_result` op, read back from
`<sub_run>/result.json`. `review()` and the scene and motion rewriters use it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import _oom

#: Where a one-off's run directory goes, under the parent run.
SUB_RUNS_DIR = "sub_runs"

# The numbered account selectors (`cli/backends/_base.py::ACCOUNT_SLOTS`).
# Matched rather than imported: `cli` imports `agent` and not the other
# way round, and the shape — family plus one digit — is the stable half
# of that table anyway.
_ACCOUNT_SELECTOR_RE = re.compile(r"\A(claude|codex)[1-9]\Z")

def backend_family(backend: str) -> str:
    """`"claude3"` -> `"claude"`, `"codex1"` -> `"codex"`; anything else as-is.

    The numbered selectors are account switches, not different CLIs — their
    backend classes deliberately keep `name = "claude"` / `"codex"` so that
    name-based conditioning never has to enumerate the slots. This is that same
    rule for the code that only ever sees the selector *string*: `run_oneoff`
    is handed one by `review_backend`, which derives it from a run bundle's
    `runtime`, so the class is not in reach.

    Without it a numbered selector reads as an unknown backend, and the
    family-specific answer recovery below (claude's stdout, codex's rollout)
    is skipped.
    """
    match = _ACCOUNT_SELECTOR_RE.match(backend or "")
    return match.group(1) if match else backend


def _codex_last_agent_message(sub_run_dir: Path) -> str | None:
    """The final assistant message codex recorded for this sub-run, or None.

    `result.json` is written only by the `submit_result` op, i.e. only if the
    agent reached the harness. A one-shot answer that never shells out leaves
    no result file — for claude that is recovered from stdout, and this is the
    codex equivalent, read from the session rollout the CLI writes itself.

    Best-effort: a missing or partial rollout returns None and the caller
    reports `no_result`, which is the honest outcome.
    """
    try:
        from .transcript_capture import find_codex_transcript
        rollout = find_codex_transcript(sub_run_dir)
        if rollout is None:
            return None
        last: str | None = None
        with open(rollout, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or "task_complete" not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "task_complete":
                    continue
                message = payload.get("last_agent_message")
                if isinstance(message, str) and message.strip():
                    last = message.strip()
        return last
    except OSError:
        return None


def _count_op_calls(sub_run_dir: Path) -> int:
    """Number of `op_call` events in a sub-run's trace.jsonl. Zero means the
    sub-agent never reached its first harness op — the wedged-launch
    signature. (Same definition as dispatch._count_op_calls.)"""
    tp = sub_run_dir / "trace.jsonl"
    if not tp.exists():
        return 0
    try:
        return sum(1 for line in tp.read_text().splitlines()
                   if '"op_call"' in line)
    except OSError:
        return 0


def _build_argv(*, backend: str, prompt: str, sub_runs_dir: Path,
                sub_run_name: str, model: str | None,
                reasoning_effort: str | None, parent_run_id: str | None,
                python_executable: str) -> list[str]:
    # The agent ends itself by calling submit_result; the bound is the
    # wall-clock timeout in run_oneoff.
    argv = [
        python_executable, "-m", "dycraft.cli",
        "--backend", backend,
        "--layer", "plain",
        "--runs-dir", str(sub_runs_dir),
        "--run-id", sub_run_name,
        "--prompt", prompt,
    ]
    if model:
        argv += ["--model", model]
    if reasoning_effort:
        argv += ["--reasoning-effort", reasoning_effort]
    if parent_run_id:
        argv += ["--parent-run", parent_run_id]
    return argv


#: How often `run_oneoff(heartbeat=...)` redraws a working sub-agent's progress on a terminal, where it is one line
#: updated in place until the stage's own closing line replaces it.
HEARTBEAT_SECONDS = 5.0
#: How often it prints a line where stderr is not a terminal (a log), so a long rewrite leaves a few lines, not dozens.
HEARTBEAT_LOG_SECONDS = 60.0


def _short(text: str, width: int = 60) -> str:
    """A command or path as an operator reads it: paths cut to their last component, one line, at most `width`."""
    text = re.sub(r"(?:/[^\s/'\"]+)+/([^\s/'\"]+)", r"\1", " ".join(str(text).split()))
    return text if len(text) <= width else text[:width - 1] + "…"


def _claude_step(block: dict) -> str:
    name, args = block.get("name") or "a tool", block.get("input") or {}
    if name == "Bash":
        return f"ran `{_short(args.get('command', ''))}`"
    if name == "Read":
        return f"read {_short(args.get('file_path', ''))}"
    if name in ("Write", "Edit", "MultiEdit"):
        return f"wrote {_short(args.get('file_path', ''))}"
    return name


def _codex_step(payload: dict) -> str:
    name = payload.get("name") or payload.get("type") or "a tool"
    raw = payload.get("arguments") or payload.get("input") or ""
    cmd = None
    if isinstance(raw, str):
        found = re.search(r"cmd\s*:\s*[`'\"](.*?)[`'\"]\s*[,}]", raw, re.S)
        if found:                    # a string literal in codex's own call: its escapes read as what they stand for
            cmd = re.sub(r"\\n|\\t", " ", found.group(1)).replace('\\"', '"').replace("\\'", "'")
        else:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                cmd = parsed.get("cmd") or parsed.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(map(str, cmd))
    return f"ran `{_short(cmd)}`" if cmd else name


def session_progress(backend: str, sub_run_dir: Path, since: float) -> dict | None:
    """`{"turns", "tools", "last"}` read from a sub-agent's own session log, or None when it is not written yet.

    The claude transcript under its account's `projects/<run dir key>/`, or the codex rollout whose
    `session_meta` names `sub_run_dir`; only files touched since `since` (wall time).
    """
    try:
        from ..cli.backends import BACKENDS
        cls = BACKENDS[backend]
        if backend_family(backend) == "claude":
            from .transcript_capture import _normalized, _transcripts_under, project_key
            import os
            _, config_dir = cls._resolve_executable()
            root = Path(config_dir or os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser()
            key = project_key(sub_run_dir)
            files = [f for f in _transcripts_under(root, key, _normalized(key)) if f.stat().st_mtime >= since]
            if not files:
                return None
            turns, tools, last = set(), 0, None
            for line in max(files, key=lambda f: f.stat().st_mtime).read_text(errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict) or rec.get("type") != "assistant":
                    continue
                message = rec.get("message") or {}
                turns.add(message.get("id"))
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tools, last = tools + 1, _claude_step(block)
            return {"turns": len(turns), "tools": tools, "last": last}
        _, home = cls._resolve_codex_cli()
        want = str(Path(sub_run_dir).resolve())
        for p in sorted(Path(home).expanduser().glob("sessions/**/rollout-*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            if p.stat().st_mtime < since:
                break
            lines = p.read_text(errors="replace").splitlines()
            head = json.loads(lines[0]) if lines else {}
            if (head.get("payload") or {}).get("cwd") != want:
                continue
            turns, tools, last = 0, 0, None
            for line in lines[1:]:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                payload = rec.get("payload") or {}
                if rec.get("type") == "event_msg" and payload.get("type") == "task_started":
                    turns += 1
                if rec.get("type") == "response_item" and payload.get("type") in ("function_call", "custom_tool_call"):
                    tools, last = tools + 1, _codex_step(payload)
            return {"turns": turns, "tools": tools, "last": last}
    except Exception:                                    # noqa: BLE001 -- progress must never fail the sub-run
        return None
    return None


def _heartbeat(proc, label: str, backend: str, sub_run_dir: Path, done, since: float,
               tty: bool | None = None) -> None:
    """While `proc` runs, what the sub-agent is doing, on stderr. On a terminal it is one line, redrawn in place every
    `HEARTBEAT_SECONDS` and cleared when the stage ends, so the stage's closing line takes its place; elsewhere (a log)
    a line every `HEARTBEAT_LOG_SECONDS`."""
    tty = sys.stderr.isatty() if tty is None else tty
    t0 = time.monotonic()
    drawn = False
    try:
        while not done.wait(HEARTBEAT_SECONDS if tty else HEARTBEAT_LOG_SECONDS):
            if proc.poll() is not None:
                return
            line = _progress_line(label, backend, sub_run_dir, since, t0)
            if tty:                  # within the terminal's width: a wrapped line could not be redrawn in place
                width = shutil.get_terminal_size((120, 24)).columns - 1
                print("\r\x1b[K" + line[:width], end="", file=sys.stderr, flush=True)
                drawn = True
            else:
                print(line, file=sys.stderr, flush=True)
    finally:
        if drawn:
            print("\r\x1b[K", end="", file=sys.stderr, flush=True)


def _progress_line(label: str, backend: str, sub_run_dir: Path, since: float, t0: float) -> str:
    line = f"[cli]   … {label}: {time.monotonic() - t0:.0f} s"
    progress = session_progress(backend, sub_run_dir, since)
    if not progress:
        return line + ", starting"
    turns, tools = progress["turns"], progress["tools"]
    line += f", {turns} turn{'' if turns == 1 else 's'}, {tools} tool call{'' if tools == 1 else 's'}"
    return line + (f", last: {progress['last']}" if progress["last"] else "")


#: The sub-run launcher's stderr, in its folder (`run_oneoff`), at most `STDERR_KEPT` characters, its end.
LAUNCHER_LOG = ".launcher.log"
STDERR_KEPT = 1_000_000


def fatal_line(stderr: str) -> str | None:
    """The first `FATAL:` line a child launcher printed, which says why it stopped."""
    for line in (stderr or "").splitlines():
        if line.startswith("FATAL:"):
            return line.strip()
    return None


def run_oneoff(
    *,
    prompt: str,
    backend: str,
    run_dir: Path,
    sub_run_name: str = "oneoff",
    model: str | None = None,
    reasoning_effort: str | None = None,
    parent_run_id: str | None = None,
    timeout_s: float = 120.0,
    python_executable: str | None = None,
    heartbeat: str | None = None,
) -> dict:
    """Launch one CLI sub-agent on `prompt` and return its text result.

    The sub-run lands in `<run_dir>/sub_runs/<sub_run_name>/`. Blocks until
    the sub-agent submits a result, exits, or `timeout_s` elapses (whichever
    first). Unlike the SDK dispatches — which lean on step caps + the 20-min
    daemon watchdog — this enforces an explicit wall-clock bound and SIGTERMs
    the sub-CLI on expiry.

    The answer comes from `submit_result` (→ `result.json`) when the agent
    calls it; otherwise — for claude — it falls back to the sub-CLI's captured
    **stdout**. This covers the one-shot case where a model answers a simple
    prompt in a single turn and exits without ever calling the harness (no
    `result.json`): claude's launcher routes banners to stderr, so stdout is the
    bare answer. The fallback is claude-only — codex's PTY-wrapped stdout is
    ANSI/redraw noise, and codex can't one-shot anyway (it answers via
    `submit_result` or wedges). stdin is closed (`DEVNULL`) so a one-shotting
    claude hits EOF and exits instead of hanging for more input.

    Returns a dict (never raises):
        {
          "ok": bool,          # got a text answer; see `timed_out` separately
          "text": str | None,  # submit_result answer, else captured stdout
          "n_op_calls": int,   # 0 ⇒ never reached harness (one-shot or wedge)
          "outcome": "completed" | "completed_after_timeout" | "timeout"
                     | "no_result" | "spawn_error",
          "timed_out": bool,
          "returncode": int | None,
          "sub_run": str,      # "sub_runs/<sub_run_name>"
          "wall_time_s": float,
          "stderr_tail": str,
        }
    """
    python_executable = python_executable or sys.executable
    sub_runs_dir = run_dir / SUB_RUNS_DIR
    sub_runs_dir.mkdir(parents=True, exist_ok=True)
    sub_run_dir = sub_runs_dir / sub_run_name
    sub_run_rel = f"{SUB_RUNS_DIR}/{sub_run_name}"

    # The prompt goes as a file the CLI reads (`--prompt <path>.md`), not as
    # an argument: Linux refuses to start a program any one of whose arguments
    # is over 128 KiB, and a review's prompt embeds the plan and its seed
    # evidence.
    #
    # A prompt that already is such a file goes as it is, by the CLI's own rule
    # (`--prompt` naming an existing `.md` file is read): the scene and motion
    # rewriters hand theirs over that way, and writing the path into a second
    # file would make the path itself the sub-agent's task.
    if (prompt.endswith(".md") and not any(c.isspace() for c in prompt)
            and Path(prompt).is_file()):
        prompt_arg = prompt
    else:
        prompt_path = sub_runs_dir / f"{sub_run_name}.prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        prompt_arg = str(prompt_path)
    argv = _build_argv(
        backend=backend, prompt=prompt_arg, sub_runs_dir=sub_runs_dir,
        sub_run_name=sub_run_name, model=model,
        reasoning_effort=reasoning_effort, parent_run_id=parent_run_id,
        python_executable=python_executable,
    )

    t0 = time.monotonic()
    try:
        # Close stdin: a one-shotting claude (answers in one turn, never calls
        # the harness) needs an EOF to exit rather than block waiting for the
        # next message; without this it hangs to the timeout (a false wedge).
        import os
        from ..cli.backends._process import PARENT_PID_ENV
        # The sub-run's launcher watches this pid, and stops when it is gone (`_process.watch_parent`).
        proc = subprocess.Popen(_oom.tiered(argv, _oom.SUBAGENT),
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True,
                                 env={**os.environ, PARENT_PID_ENV: str(os.getpid())})
    except (OSError, ValueError) as e:
        return {
            "ok": False, "text": None, "n_op_calls": 0,
            "outcome": "spawn_error", "timed_out": False,
            "returncode": None, "sub_run": sub_run_rel,
            "wall_time_s": round(time.monotonic() - t0, 2),
            "stderr_tail": f"Popen failed: {type(e).__name__}: {e}",
        }

    timed_out = False
    stdout = ""
    stderr = ""
    # A launcher-side caller (the rewriters) names the step, and an operator sees it progress.
    import threading
    beat_done = threading.Event()
    beat = None
    if heartbeat:
        beat = threading.Thread(target=_heartbeat, name="oneoff-heartbeat", daemon=True,
                                args=(proc, heartbeat, backend, sub_run_dir, beat_done, time.time() - 5))
        beat.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        # Signal a clean stop two ways: the self-kill sentinel (terminates the
        # PTY-wrapped CLI child the launcher spawned) and SIGTERM on the
        # launcher process itself.
        try:
            (sub_run_dir / ".terminate").touch()
        except OSError:
            pass
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                stderr = stderr or ""
        rc = proc.returncode

    beat_done.set()
    if beat is not None:
        beat.join(timeout=5)         # its line cleared before the caller prints the stage's closing line
    wall = round(time.monotonic() - t0, 2)
    # The child launcher's own stderr, kept beside its other logs: the caller reports only its FATAL line and a tail,
    # and what the CLI did before it (a startup screen it answered, say) is how a stop is told apart.
    try:
        sub_run_dir.mkdir(parents=True, exist_ok=True)
        (sub_run_dir / LAUNCHER_LOG).write_text((stderr or "")[-STDERR_KEPT:], encoding="utf-8")
    except OSError:
        pass
    n_op_calls = _count_op_calls(sub_run_dir)

    text: str | None = None
    result_path = sub_run_dir / "result.json"
    if result_path.exists():
        try:
            text = json.loads(result_path.read_text()).get("text")
        except (OSError, json.JSONDecodeError):
            text = None
    # Fallback: a one-shot claude answers in a single turn and exits on EOF
    # without writing result.json. Its stdout is the bare answer (banners go to
    # stderr). claude-only — codex's stdout is unusable PTY render noise, so
    # codex is handled by the rollout branch just below instead.
    # Gated on a clean exit (rc 0): a non-zero exit means claude produced NO
    # answer — e.g. a rate-limit notice ("session limit · resets …"), an auth
    # error, or a crash dumped to stdout — which must not be passed off as the
    # result. Those stay no_result (ok=False).
    if (text is None and not timed_out and backend_family(backend) == "claude"
            and rc == 0 and (stdout or "").strip()):
        text = stdout.strip()
    # The codex analog, and a strictly better one: codex records its own
    # final answer in the session rollout as
    # `event_msg/task_complete.last_agent_message`, so there is no need to
    # parse stdout (which for a PTY-wrapped TUI is redraw noise — the reason
    # the branch above is claude-only). Not gated on rc: a codex run that was
    # SIGTERM'd after answering still has the message on disk, and the rollout
    # is appended per event so even a killed run leaves a valid prefix.
    if text is None and backend_family(backend) == "codex":
        text = _codex_last_agent_message(sub_run_dir)

    # `timed_out` says the *shell* was killed; `text` says whether the agent
    # answered. They are independent, and treating a timeout as fatal threw
    # away the answer the codex branch above had just gone to the trouble of
    # recovering ("a codex run that was SIGTERM'd after answering still has
    # the message on disk"). A recorded scooter review is the case: it
    # answered in 145 s, the shell hit its ceiling, and the verdict was on
    # disk the whole time.
    if timed_out and text is not None:
        outcome = "completed_after_timeout"
    elif timed_out:
        outcome = "timeout"
    elif text is not None:
        outcome = "completed"
    else:
        outcome = "no_result"

    return {
        # An answer is an answer. `timed_out` is still reported below, so a
        # caller that cares about the shell's fate can still see it.
        "ok": text is not None,
        "text": text,
        "n_op_calls": n_op_calls,
        "outcome": outcome,
        "timed_out": timed_out,
        "returncode": rc,
        "sub_run": sub_run_rel,
        "wall_time_s": wall,
        # The last 2 KiB of the child's stderr; a review's tool result carries
        # it to the model.
        "stderr_tail": (stderr or "")[-2048:],
        # Why the child launcher stopped, in its own words, when it said so (`FATAL:`).
        "fatal": fatal_line(stderr),
    }


__all__ = ["run_oneoff", "backend_family", "SUB_RUNS_DIR"]
