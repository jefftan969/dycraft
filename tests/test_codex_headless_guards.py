"""The guards that keep a codex nobody is watching from waiting on a keypress.

Codex's startup screens wait for a keypress nobody can give in a headless run. The trust screen is answered by
recording the run directory as untrusted for the launch (`mark_codex_untrusted`), the update notice is switched off
on every launch (`CODEX_HEADLESS_ARGS`), and a codex still stuck on such a screen is stopped within seconds.
"""
from __future__ import annotations

import os
import time

import pytest

from dycraft.cli import backends as B


def _cfg(home):
    return (home / "config.toml").read_text()


def test_a_run_dir_is_recorded_untrusted_by_its_resolved_path_and_removed_after(tmp_path):
    """Codex matches its trust table against the resolved path; the entry lasts one launch."""
    real = tmp_path / "shared" / "runs" / "arm"
    real.mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "shared")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-x"\n')
    undo = B.mark_codex_untrusted(home, tmp_path / "link" / "runs" / "arm")
    assert f'[projects."{real.resolve()}"]\ntrust_level = "untrusted"' in _cfg(home)
    undo()
    assert _cfg(home) == 'model = "gpt-x"\n'


def test_a_trusted_entry_for_the_run_dir_becomes_untrusted_and_stays(tmp_path):
    """dycraft never grants trust: an existing entry is turned to untrusted, and not restored."""
    run = tmp_path / "run"
    run.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    other = f'[projects."{tmp_path}"]\ntrust_level = "trusted"\n'
    (home / "config.toml").write_text(f'{other}\n[projects."{run.resolve()}"]\ntrust_level = "trusted"\n')
    B.mark_codex_untrusted(home, run)()
    text = _cfg(home)
    assert f'[projects."{run.resolve()}"]\ntrust_level = "untrusted"' in text
    assert text.startswith(other)                     # a trusted checkout above is left alone
    assert text.count("[projects.") == 2


def _churn(home, run_dir, cycles):
    for _ in range(cycles):
        B.mark_codex_untrusted(home, run_dir)()


def test_concurrent_launches_never_expose_a_partial_file_and_leave_it_as_found(tmp_path):
    """Several launches add and remove their entries at once while a reader (as Codex reads, without the lock) checks
    every version it sees: the user's settings are always there, and the file ends as it began."""
    import multiprocessing
    import threading
    home = tmp_path / "home"
    home.mkdir()
    original = 'model = "gpt-x"\n' + "".join(f'[projects."/keep/{i}"]\ntrust_level = "trusted"\n' for i in range(300))
    (home / "config.toml").write_text(original)
    runs = []
    for i in range(4):
        (tmp_path / f"run{i}").mkdir()
        runs.append(tmp_path / f"run{i}")
    bad, stop = [], threading.Event()

    def read():
        while not stop.is_set():
            text = (home / "config.toml").read_text()
            if not text.startswith(original):
                bad.append(len(text))

    reader = threading.Thread(target=read)
    reader.start()
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_churn, args=(home, run, 25)) for run in runs]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    stop.set()
    reader.join()
    assert all(p.exitcode == 0 for p in procs)
    assert bad == []
    assert _cfg(home) == original


_KILLED_MID_EDIT = r"""
import os, signal, sys
from pathlib import Path
from dycraft.cli import backends as B
mode, home, run = sys.argv[1], sys.argv[2], sys.argv[3]
real_replace = os.replace
if mode == "kill":
    B.codex.os.replace = lambda *a: os.kill(os.getpid(), signal.SIGKILL)
else:
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    def replace_with_a_sigterm_first(*a):
        os.kill(os.getpid(), signal.SIGTERM)
        real_replace(*a)
    B.codex.os.replace = replace_with_a_sigterm_first
B.mark_codex_untrusted(home, run)
print("returned", flush=True)
"""


@pytest.mark.parametrize("mode", ["kill", "term"])
def test_a_launch_stopped_mid_edit_never_damages_the_file(tmp_path, pkgroot, mode):
    """SIGKILL between writing the new file and renaming it leaves the old file whole; a SIGTERM waits for the
    rename, so the edit completes before the handler runs."""
    import subprocess
    import sys
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-x"\n')
    run = tmp_path / "run"
    run.mkdir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(pkgroot), os.environ.get("PYTHONPATH")]))}
    ran = subprocess.run([sys.executable, "-c", _KILLED_MID_EDIT, mode, str(home), str(run)],
                         env=env, capture_output=True, text=True, timeout=60)
    if mode == "kill":
        assert ran.returncode == -9
        assert _cfg(home) == 'model = "gpt-x"\n'
    else:
        assert ran.returncode == 143 and "returned" not in ran.stdout
        assert f'[projects."{run.resolve()}"]\ntrust_level = "untrusted"' in _cfg(home)


def test_a_symlinked_config_is_edited_at_its_target(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "dotfiles").mkdir()
    (tmp_path / "dotfiles" / "codex.toml").write_text('model = "gpt-x"\n')
    (home / "config.toml").symlink_to(tmp_path / "dotfiles" / "codex.toml")
    B.mark_codex_untrusted(home, tmp_path)
    assert (home / "config.toml").is_symlink()
    assert "untrusted" in (tmp_path / "dotfiles" / "codex.toml").read_text()


def test_an_unwritable_account_does_not_stop_the_launch(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").mkdir()                    # cannot be opened as a file
    B.mark_codex_untrusted(home, tmp_path)()
    assert "could not record" in capsys.readouterr().err


def _fake_cli(tmp_path, body):
    script = tmp_path / "fakecodex"
    script.write_text("#!/usr/bin/env python3\nimport sys, time\n" + body)
    script.chmod(0o755)
    return str(script)


def _run_shim(argv):
    """Run the shim with stdout swallowed; return (rc, seconds)."""
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    started = time.monotonic()
    try:
        rc = B._process._spawn_with_pty_term_shim(argv, dict(os.environ))
    finally:
        os.dup2(saved, 1)
        os.close(w)
        os.close(saved)
        os.close(r)
    return rc, time.monotonic() - started


TRUST_SCREEN = ("sys.stdout.write('\\x1b[2J> You are in /x\\r\\n"
                "Do you trust the \\x1b[1mcontents\\x1b[0m of this directory?\\r\\n"
                " 1. Yes, continue\\r\\n Press enter to continue'); sys.stdout.flush()\n")


def test_a_cli_stuck_on_a_startup_screen_is_stopped_fast(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    rc, took = _run_shim([_fake_cli(tmp_path, TRUST_SCREEN + "time.sleep(120)\n")])
    assert rc == B.EXIT_CODEX_BLOCKED
    assert took < B._process._BLOCKING_SCREEN_GRACE_SECONDS + 10
    # On a line of its own, not after the screen's "Press enter to continue".
    err = capsys.readouterr().err
    assert "\nFATAL: codex stopped at startup on its 'trust this directory' screen" in err
    assert "Answer it once in a terminal (`CODEX_HOME=" in err and "codex` in " in err


#: codex 0.153.4 on a damaged `state_5.sqlite`, as one child printed it before
#: waiting 38 minutes for an Enter nobody could press.
DATABASE_SCREEN = ("sys.stdout.write(\"\\x1b[2J Codex couldn't start because its local database "
                   "appears to be damaged. Moving the damaged local database aside so Codex can "
                   "rebuild it from saved data.\\r\\n Codex rebuilt its local database. Codex "
                   "detected a damaged local database, moved it into a backup folder, and will "
                   "continue startup with a fresh database.\\r\\n Press Enter to continue.\"); "
                   "sys.stdout.flush()\n")


def test_a_cli_stuck_on_the_damaged_database_screen_is_stopped_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    rc, took = _run_shim([_fake_cli(tmp_path, DATABASE_SCREEN + "time.sleep(120)\n")])
    assert rc == B.EXIT_CODEX_BLOCKED
    assert took < B._process._BLOCKING_SCREEN_GRACE_SECONDS + 10


def test_the_same_words_after_startup_are_content_not_a_screen(tmp_path, monkeypatch):
    """An agent that `cat`s this repo's source prints these exact strings.
    Past the startup window they must not get it killed."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    monkeypatch.setattr(B._process, "_BLOCKING_SCREEN_SCAN_SECONDS", 0.0)
    rc, _ = _run_shim([_fake_cli(tmp_path, TRUST_SCREEN + "time.sleep(7)\nsys.exit(3)\n")])
    assert rc == 3


def test_an_ordinary_exit_keeps_its_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    rc, _ = _run_shim([_fake_cli(tmp_path, "print('hello'); sys.exit(5)\n")])
    assert rc == 5



def _rollout_cli(tmp_path, records, screen, exit_after, draw_after=0.0, then=None):
    """A stand-in codex that opens a rollout for its `-C` directory, draws `screen` `draw_after` s later, appends the
    records `then` (if any) 1 s after that, and exits after `exit_after` s more."""
    script = tmp_path / "fakecodex"
    script.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys, time\nfrom pathlib import Path\n"
        "run = sys.argv[sys.argv.index('-C') + 1]\n"
        "out = Path(os.environ['CODEX_HOME']) / 'sessions' / '2026' / 'rollout-x.jsonl'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        f"records = {records!r}\n"
        "records[0]['payload']['cwd'] = run\n"
        "out.write_text(''.join(json.dumps(r) + '\\n' for r in records))\n"
        f"time.sleep({draw_after}); sys.stdout.write({screen!r}); sys.stdout.flush()\n"
        f"then = {then!r}\n"
        "if then:\n"
        "    time.sleep(1.0)\n"
        "    with out.open('a') as fh: fh.write(''.join(json.dumps(r) + '\\n' for r in then))\n"
        f"time.sleep({exit_after}); sys.exit(3)\n")
    script.chmod(0o755)
    return [str(script), "-C", str(tmp_path / "run")]


_LOST = "\x1b[2J• Reconnecting... 5 / 5\r\n└ Stream disconnected before completion: Connection refused (os error 111)\r\n› Ask Codex to do anything"
_STARTED = [{"type": "session_meta", "payload": {}}, {"type": "event_msg", "payload": {"type": "task_started"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}}]


@pytest.fixture
def lost_stream_watch(tmp_path, monkeypatch):
    monkeypatch.delenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(B._process, "_ROLLOUT_POLL_SECONDS", 0.1)
    monkeypatch.setattr(B._process, "_LOST_STREAM_GRACE_SECONDS", 0.5)
    (tmp_path / "run").mkdir()


def test_a_codex_that_gave_up_a_turn_it_could_not_stream_is_stopped(tmp_path, lost_stream_watch):
    """Codex prints the reconnect failure and waits at its prompt; the rollout records nothing more."""
    rc, took = _run_shim(_rollout_cli(tmp_path, _STARTED, _LOST, exit_after=60))
    assert rc == B.EXIT_CODEX_FATAL
    assert took < 20
    import json
    stop = json.loads((tmp_path / "run" / "meta.json").read_text())["agent_stop"]
    assert stop["reason"] == "codex_lost_stream" and stop["detail"].startswith("Stream disconnected before completion")


def test_the_same_words_after_the_model_answered_are_not_a_lost_stream(tmp_path, lost_stream_watch):
    answered = _STARTED + [{"type": "response_item", "payload": {"type": "reasoning", "summary": []}}]
    rc, _ = _run_shim(_rollout_cli(tmp_path, answered, _LOST, exit_after=4))
    assert rc == 3


#: codex 0.153.4's status line while it retries a connection, as its first frame draws it (an N2 matrix row).
_RECONNECT = ("\x1b[2J\x1b[2m◦\x1b[22m \x1b[2mReconnecting... wai\x1b[22mt\x1b[1ming f\x1b[22mo\x1b[2mr network"
              "\x1b[22m \x1b[2m(3s • esc to interrupt)\r\n  └ Connection failed: error sending request\r\n"
              "› Ask Codex to do anything")


@pytest.fixture
def reconnect_watch(lost_stream_watch, monkeypatch):
    monkeypatch.setattr(B._process, "_RECONNECT_BOUND_SECONDS", 1.0)


@pytest.mark.parametrize("records", [_STARTED, _STARTED + [{"type": "response_item",
                                                           "payload": {"type": "reasoning", "summary": []}}]],
                         ids=["at-startup", "mid-run"])
def test_a_codex_that_keeps_reconnecting_with_nothing_recorded_is_stopped(tmp_path, reconnect_watch, capsys,
                                                                          records):
    """Codex retries a lost connection without end and records nothing meanwhile: the CLI is stopped and says why."""
    import json
    rc, took = _run_shim(_rollout_cli(tmp_path, records, _RECONNECT, exit_after=60, draw_after=0.5))
    assert rc == B.EXIT_CODEX_FATAL
    assert took < 20
    assert "\nFATAL: codex has been reconnecting to its model for 1 s ('Reconnecting... waiting for network (3s" \
        in capsys.readouterr().err
    stop = json.loads((tmp_path / "run" / "meta.json").read_text())["agent_stop"]
    assert stop["reason"] == "codex_reconnecting"


def test_a_codex_that_reconnected_and_went_on_is_left_alone(tmp_path, reconnect_watch, monkeypatch):
    """A record written after the status line means the connection came back."""
    monkeypatch.setattr(B._process, "_RECONNECT_BOUND_SECONDS", 2.0)
    went_on = [{"type": "response_item", "payload": {"type": "reasoning", "summary": []}}]
    rc, _ = _run_shim(_rollout_cli(tmp_path, _STARTED, _RECONNECT, exit_after=3, draw_after=0.5, then=went_on))
    assert rc == 3


#: What codex 0.153.4 printed before it exited, with this machine's proxy down (the N1 matrix row).
_NO_CONFIG = "\x1b[2J› Ask Codex to do anything\r\n\x1b[<1u\x1b[>4;0mError loading configuration: Failed to load cloud " \
             "config bundle (workspace-managed policies)."


def test_a_codex_that_exits_before_it_answered_says_what_its_screen_showed(tmp_path, lost_stream_watch, capsys):
    import json
    rc, took = _run_shim(_rollout_cli(tmp_path, _STARTED, _NO_CONFIG, exit_after=0.3))
    assert rc == B.EXIT_CODEX_FATAL
    err = capsys.readouterr().err
    assert "\nFATAL: codex exited (status 3) before it answered; its screen last showed: " in err
    assert "<1u" not in err and ">4;0m" not in err             # escapes with private parameters are taken out whole
    stop = json.loads((tmp_path / "run" / "meta.json").read_text())["agent_stop"]
    assert stop["reason"] == "codex_exited_before_answering"
    assert stop["detail"].endswith("Failed to load cloud config bundle (workspace-managed policies).")


def test_a_codex_that_exits_with_no_session_at_all_is_named_too(tmp_path, lost_stream_watch, capsys):
    cli = _fake_cli(tmp_path, f"sys.stdout.write({_NO_CONFIG!r}); sys.stdout.flush(); sys.exit(1)\n")
    rc, _ = _run_shim([cli, "-C", str(tmp_path / "run")])
    assert rc == B.EXIT_CODEX_FATAL
    assert "Error loading configuration" in capsys.readouterr().err


def test_a_codex_that_answered_keeps_its_own_exit_status(tmp_path, lost_stream_watch, capsys):
    answered = _STARTED + [{"type": "response_item", "payload": {"type": "reasoning", "summary": []}}]
    rc, _ = _run_shim(_rollout_cli(tmp_path, answered, "bye", exit_after=0.3))
    assert rc == 3
    assert "FATAL" not in capsys.readouterr().err


def test_a_usage_limit_codex_sits_on_is_named_as_one(tmp_path, lost_stream_watch, monkeypatch, capsys):
    monkeypatch.setattr(B._process, "_ROLLOUT_FATAL_GRACE_SECONDS", 0.5)
    limit = _STARTED + [{"type": "event_msg", "payload": {"type": "task_complete", "error": {
        "codex_error_info": "usage_limit_exceeded", "message": "You've hit your usage limit. Try again later."}}}]
    rc, _ = _run_shim(_rollout_cli(tmp_path, limit, "› Ask Codex to do anything", exit_after=60))
    assert rc == B.EXIT_CODEX_FATAL
    assert "\nFATAL: codex hit its usage limit: You've hit your usage limit. Try again later. No activity followed " \
        "for 0.5 s, so the CLI is stopped." in capsys.readouterr().err


#: Codex 0.167's screen for a folder saved as untrusted, under an organization that allows only on-request approvals
#: (`mark_codex_untrusted` records every run directory so), as a FarmShare and an SC launch stopped on it. The fake
#: reads keys raw, as a TUI does, and exits 0 once option 1 is chosen; with nothing chosen it sits.
RESTRICTED_SCREEN = (
    "import tty, termios, os\n"
    "tty.setraw(0)\n"
    "sys.stdout.write('\\x1b[2J This folder is untrusted: project config is skipped, and tools follow your "
    "permission settings. Opening will not change saved trust.\\r\\n \\u203a 1. Open restricted\\r\\n "
    "2. Back to Agent Command Center\\r\\n enter continue \\u00b7 esc back'); sys.stdout.flush()\n"
    "keys = b''\n"
    "while True:\n"
    "    keys += os.read(0, 16)\n"
    "    if b'1' in keys or b'\\r' in keys:\n"
    "        sys.exit(0 if b'2' not in keys else 4)\n")


def test_the_untrusted_folder_screen_is_answered_open_restricted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    rc, took = _run_shim([_fake_cli(tmp_path, RESTRICTED_SCREEN)])
    assert rc == 0 and took < 10
    assert "[codex] answered codex's startup screen: open the untrusted run directory restricted" in \
        capsys.readouterr().err


def test_the_answer_is_confirmed_with_enter_without_waiting_for_a_redraw(tmp_path, monkeypatch):
    """Codex 0.167: `1` only highlights "Open restricted" and Enter confirms it, and Codex redraws only the cells
    that changed, so the screen's words never come again. Exits 0 only on `1` then Enter."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    body = RESTRICTED_SCREEN.replace("    if b'1' in keys or b'\\r' in keys:\n        sys.exit(0 if b'2' not in keys else 4)\n",
                                     "    if b'1' in keys and b'\\r' in keys[keys.index(b'1'):]:\n        sys.exit(0)\n")
    assert "keys.index" in body
    rc, took = _run_shim([_fake_cli(tmp_path, body)])
    assert rc == 0 and took < 10


def test_a_lost_enter_is_pressed_again_while_no_session_opens(tmp_path, monkeypatch):
    """The answer's keys can come before the screen takes input; Enter follows again until Codex goes on."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_ROLLOUT_WATCH", "1")
    monkeypatch.setattr(B._process, "_ANSWERED_SCREEN_RETRY_SECONDS", 0.3)
    # Ignores the first Enter it reads, as a screen not yet taking input would; goes on at the second.
    body = RESTRICTED_SCREEN.replace("    if b'1' in keys or b'\\r' in keys:\n        sys.exit(0 if b'2' not in keys else 4)\n",
                                     "    if keys.count(b'\\r') >= 2:\n        sys.exit(0)\n")
    assert "count(b" in body
    rc, took = _run_shim([_fake_cli(tmp_path, body)])
    assert rc == 0 and took < 10
