"""The CLI backends' daemon discovery, watchdog, harness transport, and sandbox settings.

What's covered (and why):

1. `_find_agent_pid` — the /proc cmdline scan used by the watchdog to switch
   from the launcher pid to the real agent pid after Claude Code's binary
   internally forks. We verify it matches the `--add-dir\\0<run_dir>\\0`
   argv-pair as adjacent null-separated tokens (the tightening that prevents
   `runs/cube_large` from shadowing `runs/cube`) and that it requires the
   backend marker substring as well.

2. `_Watchdog` — it follows a live launcher even when the CLI exits; once the
   launcher is gone, discovery flips the target to the agent; and the
   orphan-cleanup property (post-retarget ESRCH shuts the daemon down).

3. TCP harness wire protocol — the AF_INET loopback IPC, used because
   Claude Code's bash sandbox seccomp-blocks `socket(AF_UNIX, ...)`.
   Spins up a real daemon socket + the
   `_harness_client._call` path and asserts round-trip works on a free
   ephemeral port.

4. Backend / sandbox config invariants — `agent_cmdline_marker` set for
   both CLI backends, claude's `--settings` JSON shape (sandbox enabled,
   autoAllowBashIfSandboxed, failIfUnavailable, filesystem.allowWrite
   pinned to the run dir), claude's permission rules for Edit/Write are
   path-scoped to the run dir (the bwrap sandbox does NOT cover Edit/Write
   per Claude Code docs, so scoping is enforced via the permission system
   instead), claude's default permission-mode is `dontAsk` (not
   `acceptEdits`, which would blanket-auto-approve Edit/Write everywhere
   and defeat the path scoping), and codex's argv carries `-s
   workspace-write -a never -c
   sandbox_workspace_write.network_access=true` with its working root
   pinned to the run dir, `CODEX_HOME` pointed at the user's real
   `~/.codex` (write-confinement comes from `-C`/`-s`/`--add-dir`, not
   from the config home), and no launcher-level bypass flag.

No test here launches a real agent CLI. A sandbox that cannot start a
command is caught at run time instead (`claude.ClaudeSessionWatch`, tested in
`test_agent_session_watch.py`). The model/effort argv
mapping is pinned without one by
`test_{claude,codex}_execute_forwards_model_and_reasoning_effort`, and a
committing build by `fixed_source` in `test_object_worker.py`.

**The sandbox escape hatch.** When a harness bash call fails on a sandbox error
(`apply-seccomp`, on a host that cannot nest a user namespace), Claude Code's
own system prompt tells the agent to retry with `dangerouslyDisableSandbox:
true`, and the agent then uses that flag for every later harness call while the
run still commits. `cli/backends/claude.py` allows `Bash(*)`, which would
approve such a call, and `failIfUnavailable: True` only covers the sandbox
being unavailable at *startup*. So the settings set
`allowUnsandboxedCommands: False`, which takes the flag away, and
`test_claude_settings_take_away_the_sandbox_escape_hatch` checks it.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from dycraft.agent import _harness_core as _harness_client
from dycraft.cli.backends import (ClaudeBackend, claude_agent_settings,
                                  CodexBackend, LaunchPlan)
from dycraft.agent.server import _Watchdog, _find_agent_pid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spawn_fake_agent(argv: list[str]) -> subprocess.Popen:
    """Spawn a long-sleep subprocess whose /proc/<pid>/cmdline equals
    `argv[0]\\0-c\\0<sleep>\\0argv[1:]` — i.e. argv[0] is preserved and the
    caller's remaining tokens become the script's sys.argv. We splice
    `-c <script>` in second so Python's option parser consumes it and
    treats the rest as positional (it would reject a stray `--add-dir`).

    Without that ordering trick the child errors at startup ("unknown
    option --add-dir") and /proc/<pid>/cmdline reads back empty.
    """
    args = [argv[0], "-c", "import time; time.sleep(300)", *argv[1:]]
    proc = subprocess.Popen(args, executable=sys.executable,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    # Wait until the kernel publishes the new argv (post-execve). Without
    # this the test is racy on slow hosts.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            if b"--add-dir" in _read_proc_cmdline(proc.pid):
                return proc
        except (FileNotFoundError, ProcessLookupError):
            pass
        time.sleep(0.02)
    proc.kill(); proc.wait(timeout=2)
    raise RuntimeError(
        f"fake agent never populated cmdline (argv={argv}); "
        f"executable=python -c likely choked on a leading --flag")


def _read_proc_cmdline(pid: int) -> bytes:
    return Path(f"/proc/{pid}/cmdline").read_bytes()


def _free_local_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# _find_agent_pid — argv-pair matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["claude", "codex"])
def test_find_agent_pid_matches_real_agent_argv(tmp_path, marker):
    """A subprocess with argv `<marker> --add-dir <run_dir>` is found by
    _find_agent_pid scanning real /proc. Parameterized over both backends
    so a marker plumbing bug for either is caught.
    """
    run_dir = tmp_path / "runs" / f"myrun_{marker}_cube"
    run_dir.mkdir(parents=True)
    proc = _spawn_fake_agent([marker, "--add-dir", str(run_dir)])
    try:
        assert _find_agent_pid(run_dir, marker) == proc.pid
    finally:
        proc.kill(); proc.wait(timeout=5)


def test_find_agent_pid_argv_pair_not_substring(tmp_path):
    """A query for `runs/cube` must NOT match a
    process whose argv contains `--add-dir runs/cube_large`. The needle
    is `--add-dir\\0<run_dir>\\0` with bounding nulls, so a longer path
    that contains the shorter as a prefix is correctly rejected.
    """
    short_run_dir = tmp_path / "runs" / "cube"
    long_run_dir = tmp_path / "runs" / "cube_large"
    short_run_dir.mkdir(parents=True)
    long_run_dir.mkdir(parents=True)
    long_proc = _spawn_fake_agent(["claude", "--add-dir", str(long_run_dir)])
    try:
        # Sanity: the substring DOES appear naively (so the test would fail
        # if we accidentally weakened the match back to a substring scan).
        cmdline = _read_proc_cmdline(long_proc.pid)
        assert str(short_run_dir).encode() in cmdline
        # But the null-bounded argv-pair match rejects the shadow.
        assert _find_agent_pid(short_run_dir, "claude") is None
        # And the long one is still findable for sanity.
        assert _find_agent_pid(long_run_dir, "claude") == long_proc.pid
    finally:
        long_proc.kill(); long_proc.wait(timeout=5)


def _marker_free_tmpdir() -> str:
    """Create a temp dir whose absolute path contains no backend marker.

    `_find_agent_pid` scans the whole `/proc/<pid>/cmdline` (which includes
    the run_dir, via `--add-dir`) for the marker substring. If the ambient
    temp root itself carries a marker — e.g. a `/tmp/claude-*` sandbox set as
    `TMPDIR` — a non-agent process whose only crime is sharing the run_dir
    would match, which is exactly the false-positive the requires-marker test
    is asserting against. Pick a base carrying neither "claude" nor "codex"
    so the test isolates the program-name marker check from an incidental
    path hit, regardless of the host's TMPDIR.
    """
    for base in (tempfile.gettempdir(), "/tmp",
                 os.path.expanduser("~"), os.getcwd()):
        low = str(base).lower()
        if "claude" in low or "codex" in low:
            continue
        try:
            return tempfile.mkdtemp(prefix="dycraft_pidtest_", dir=base)
        except OSError:
            continue
    pytest.skip("no marker-free temp directory available")


def test_find_agent_pid_requires_marker():
    """A process whose argv contains `--add-dir <run_dir>` but is NOT an
    agent CLI must not be misattributed. Covers the case where some
    unrelated tool (a make rule, a shell one-liner) shares the same
    --add-dir pair.
    """
    base = Path(_marker_free_tmpdir())
    try:
        run_dir = base / "runs" / "myrun"
        run_dir.mkdir(parents=True)
        proc = _spawn_fake_agent(["not-an-agent", "--add-dir", str(run_dir)])
        try:
            assert _find_agent_pid(run_dir, "claude") is None
            assert _find_agent_pid(run_dir, "codex") is None
            # Empty marker = caller didn't declare one — discovery disabled.
            assert _find_agent_pid(run_dir, "") is None
        finally:
            proc.kill(); proc.wait(timeout=5)
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# _Watchdog — retarget + orphan-cleanup
# ---------------------------------------------------------------------------


def test_watchdog_stays_on_a_live_launcher_when_the_cli_exits(tmp_path):
    """The launcher runs the CLI as its child, so the daemon follows it: a CLI that exits, to be started again by the
    same launcher, does not take the daemon with it. Run in a subprocess, which a regressed watchdog SIGTERMs."""
    run_dir = tmp_path / "runs" / "myrun_claude_retry"
    run_dir.mkdir(parents=True)
    dycraft_root = str(Path(__file__).resolve().parents[1])
    code = f"""
import os, signal, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, {dycraft_root!r})
from dycraft.agent.server import _Watchdog
run_dir = Path({str(run_dir)!r})
got_sigterm = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: got_sigterm.set())
agent = subprocess.Popen(["claude", "-c", "import time; time.sleep(60)", "--add-dir", str(run_dir)],
                         executable=sys.executable, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.3)
wd = _Watchdog(os.getpid(), run_dir, "claude", poll_s=0.05, grace_s=10.0)
wd.start()
time.sleep(0.5)
agent.kill(); agent.wait()
time.sleep(0.5)
sys.exit(0 if not got_sigterm.is_set() and not wd.retargeted else 2)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, f"the daemon followed the CLI: rc={proc.returncode} {proc.stderr!r}"


def test_watchdog_retargets_to_discovered_agent(tmp_path):
    """Once the launcher pid is gone (a launcher that exec'd the CLI, whose binary then forked), the watchdog finds
    the agent in /proc and follows it instead. The launcher here is fake (sleep, doesn't match the marker) and the
    agent the real subprocess that does match.
    """
    run_dir = tmp_path / "runs" / "myrun_claude_test"
    run_dir.mkdir(parents=True)
    launcher = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    agent = _spawn_fake_agent(["claude", "--add-dir", str(run_dir)])
    wd = _Watchdog(launcher.pid, run_dir, "claude",
                   poll_s=0.05, grace_s=10.0)
    try:
        assert wd.target_pid == launcher.pid
        assert not wd.retargeted
        wd.start()
        time.sleep(0.3)
        assert not wd.retargeted, "retargeted while the launcher was alive"
        launcher.kill(); launcher.wait(timeout=5)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if wd.retargeted:
                break
            time.sleep(0.05)
        assert wd.retargeted, "watchdog never retargeted within 5 s"
        assert wd.target_pid == agent.pid
    finally:
        # Stop the watchdog BEFORE tearing down the subprocesses, else it'd
        # detect the agent's death and SIGTERM the pytest process.
        wd.stop()
        launcher.kill(); launcher.wait(timeout=5)
        agent.kill(); agent.wait(timeout=5)


def test_watchdog_shuts_down_when_retargeted_pid_dies(tmp_path):
    """Once retargeted, the watchdog SIGTERMs its own process the moment
    the agent pid disappears. This is the orphan-cleanup property — if the
    agent crashes, the daemon doesn't linger.

    Run inside a subprocess so we can observe the SIGTERM-induced exit
    cleanly. The subprocess ALSO spawns the fake agent (rather than the
    pytest process spawning it) so that when the subprocess SIGKILLs +
    waits the agent, the zombie is reaped — without that reap,
    `os.kill(agent_pid, 0)` keeps returning success (zombies still occupy
    /proc), the watchdog never sees ESRCH, and the SIGTERM-on-orphan path
    we're trying to verify never fires.
    """
    run_dir = tmp_path / "runs" / "myrun_claude_orphan"
    run_dir.mkdir(parents=True)
    dycraft_root = str(Path(__file__).resolve().parents[1])
    code = f"""
import sys, time, threading, signal, os, subprocess
from pathlib import Path
sys.path.insert(0, {dycraft_root!r})
from dycraft.agent.server import _Watchdog

run_dir = Path({str(run_dir)!r})

# Spawn fake agent FROM HERE so we're its parent and can reap it later.
# We splice `-c <script>` before the extra args so python's option parser
# doesn't choke on `--add-dir` — see _spawn_fake_agent comment in the
# test module for the full reasoning.
agent = subprocess.Popen(
    ["claude", "-c", "import time; time.sleep(60)", "--add-dir", str(run_dir)],
    executable=sys.executable,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
deadline = time.time() + 2.0
while time.time() < deadline:
    try:
        if b"--add-dir" in Path(f"/proc/{{agent.pid}}/cmdline").read_bytes():
            break
    except (FileNotFoundError, ProcessLookupError):
        pass
    time.sleep(0.02)
else:
    agent.kill(); agent.wait()
    sys.exit(4)  # fake-agent setup failure (separate from the SIGTERM test)

got_sigterm = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: got_sigterm.set())
# A launcher that is already gone, as one that exec'd the CLI: the watchdog then follows the agent.
gone = subprocess.Popen([sys.executable, "-c", "pass"]); gone.wait()
wd = _Watchdog(gone.pid, run_dir, "claude", poll_s=0.05, grace_s=30.0)
wd.start()

# Wait for the watchdog to retarget to the agent.
for _ in range(200):
    if wd.retargeted: break
    time.sleep(0.05)
if not wd.retargeted:
    agent.kill(); agent.wait()
    sys.exit(3)  # never retargeted — different bug, distinct rc

# Kill the agent AND reap it (we're the parent), then wait for the
# watchdog to detect ESRCH and SIGTERM ourselves.
agent.kill(); agent.wait()
deadline = time.time() + 5.0
while time.time() < deadline and not got_sigterm.is_set():
    time.sleep(0.05)
sys.exit(0 if got_sigterm.is_set() else 2)
"""
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=20)
    # rc=0 → watchdog SIGTERMed us within 5 s of agent death (success).
    # rc=2 → no SIGTERM (orphan-cleanup regression).
    # rc=3 → never retargeted (different bug, surfaces clearly).
    # rc=4 → couldn't spawn the fake agent (test-fixture issue).
    assert proc.returncode == 0, (
        f"watchdog orphan-cleanup regression: rc={proc.returncode} "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}")


def test_watchdog_grace_period_tolerates_launcher_death(tmp_path):
    """If the launcher pid dies before discovery completes (the symptom that
    motivated this refactor), the watchdog tolerates ESRCH on the initial
    pid for `grace_s` seconds and uses that window to find the real agent.
    """
    run_dir = tmp_path / "runs" / "myrun_claude_grace"
    run_dir.mkdir(parents=True)
    # Launcher exits immediately so its pid is gone before the watchdog
    # gets its second poll.
    launcher = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"])
    launcher.wait()
    # Agent comes up before grace expires.
    agent = _spawn_fake_agent(["claude", "--add-dir", str(run_dir)])
    wd = _Watchdog(launcher.pid, run_dir, "claude",
                   poll_s=0.05, grace_s=5.0)
    try:
        wd.start()
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if wd.retargeted:
                break
            time.sleep(0.05)
        assert wd.retargeted, (
            "watchdog didn't tolerate launcher death long enough to discover")
        assert wd.target_pid == agent.pid
    finally:
        wd.stop()  # same reason as above — don't SIGTERM pytest at teardown
        agent.kill(); agent.wait(timeout=5)


# ---------------------------------------------------------------------------
# TCP harness wire protocol — AF_INET roundtrip
# ---------------------------------------------------------------------------


def test_harness_client_tcp_roundtrip(tmp_path, monkeypatch):
    """`_harness_client._call` issues a length-prefixed JSON request over
    TCP loopback and parses the response, end to end: port file → connect →
    frame → reply → demux.

    Integration-style rather than a mock, because the protocol invariants live
    in both halves (`agent/server.py` and the harness client core). A
    regression in either (wrong address family, off-by-one on the length
    header) only shows up when the two halves talk.
    """
    port = _free_local_port()
    # Bind a real socket with the wire protocol the daemon would serve.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def _serve_once():
        conn, _ = srv.accept()
        try:
            hdr = conn.recv(4)
            (n,) = struct.unpack(">I", hdr)
            buf = b""
            while len(buf) < n:
                buf += conn.recv(n - len(buf))
            req = json.loads(buf.decode())
            resp = {"ok": True, "result": {"echoed": req}, "stdout": ""}
            data = json.dumps(resp).encode()
            conn.sendall(struct.pack(">I", len(data)) + data)
        finally:
            conn.close()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    try:
        # Wire `_harness_client` to our temp run dir + port file.
        (tmp_path / ".harness.port").write_text(str(port))
        monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
        monkeypatch.setattr(_harness_client, "PORT_PATH",
                            tmp_path / ".harness.port")
        result = _harness_client._call("compile", {"freeze": False})
        assert result == {"echoed": {"method": "compile",
                                     "args": {"freeze": False}}}
    finally:
        srv.close()
        t.join(timeout=2)


def test_harness_client_missing_port_file_errors(tmp_path, monkeypatch):
    """If `.harness.port` is missing, the client raises a SystemExit with a
    pointer to the daemon log — preferable to a confusing `ConnectionError`
    when the agent runs before the launcher has finished starting the
    daemon.
    """
    monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
    monkeypatch.setattr(_harness_client, "PORT_PATH",
                        tmp_path / ".harness.port")
    with pytest.raises(SystemExit) as excinfo:
        _harness_client._call("compile", {})
    assert "daemon port file missing" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP CONNECT tunnel — Claude bash-sandbox netns workaround
# ---------------------------------------------------------------------------
#
# Claude Code's bash sandbox runs each tool call in a fresh network
# namespace, so a direct TCP connect to the host daemon at
# `127.0.0.1:<port>` from inside the sandbox lands in an empty loopback
# and returns ConnectionRefused. The client recovers by tunneling
# through Claude's HTTP proxy: open a socket to `$HTTP_PROXY`, send
# `CONNECT 127.0.0.1:<port> HTTP/1.1`, and use the upgraded socket as
# a transparent byte tunnel — daemon-side wire format unchanged.
#
# These tests pin: (a) the parser handles every proxy URL shape we'll
# see in real envs, (b) the direct path stays the fast common case
# (codex shares host netns + replay/debug from the host), (c) the
# refused→CONNECT fallback works against a faithful CONNECT proxy
# stand-in, and (d) the error path surfaces the proxy's status line
# verbatim when allowedDomains is misconfigured (an agent that sees
# only a bare refusal diagnoses the wrong cause).


@pytest.mark.parametrize("url, expected", [
    ("http://localhost:3128",        ("localhost", 3128)),
    ("http://127.0.0.1:3128",        ("127.0.0.1", 3128)),
    ("http://proxy.internal:8080/",  ("proxy.internal", 8080)),
    ("localhost:3128",               ("localhost", 3128)),
    ("proxy",                        ("proxy", 80)),
    # Claude Code's in-sandbox proxy is now authenticated: the `user:pass@`
    # userinfo MUST be stripped from the host or connect() raises gaierror.
    ("http://srt:tok@localhost:3128", ("localhost", 3128)),
    ("user:pass@proxy.internal:8080", ("proxy.internal", 8080)),
])
def test_parse_proxy_url_forms(url, expected):
    """$HTTP_PROXY varies by env (Claude sets `http://localhost:3128`,
    corporate proxies use `http://host.tld:port/`, some users export bare
    `host:port`). The parser is intentionally lax on shape but strict on
    scheme — `https://` and `socks5://` aren't supported and must raise
    so the caller doesn't silently downgrade the handshake.
    """
    assert _harness_client._parse_proxy_url(url) == expected


@pytest.mark.parametrize("url", [
    "https://proxy:443",                  # we don't speak TLS to the proxy
    "socks5h://localhost:1080",           # different handshake protocol
    "socks5://localhost:1080",
])
def test_parse_proxy_url_rejects_non_http_schemes(url):
    with pytest.raises(ValueError, match="unsupported proxy scheme"):
        _harness_client._parse_proxy_url(url)


@pytest.mark.parametrize("url, expected", [
    ("http://localhost:3128",         None),
    ("localhost:3128",                None),
    ("http://srt:tok@localhost:3128",
     "Proxy-Authorization: Basic " + base64.b64encode(b"srt:tok").decode()),
    ("user:pass@proxy:8080",
     "Proxy-Authorization: Basic " + base64.b64encode(b"user:pass").decode()),
])
def test_proxy_auth_header(url, expected):
    """Claude Code's in-sandbox proxy requires Basic auth; the CONNECT must
    carry `Proxy-Authorization` built from the URL's `user:pass@` userinfo,
    or the proxy answers `407` and the agent falls back to unsandboxed bash.
    """
    assert _harness_client._proxy_auth_header(url) == expected


def test_mask_proxy_redacts_credentials():
    masked = _harness_client._mask_proxy("http://srt:secrettoken@localhost:3128")
    assert "srt" not in masked and "secrettoken" not in masked
    assert masked == "http://***:***@localhost:3128"


def _spawn_fake_connect_proxy(daemon_port: int, *,
                              reply: bytes = b"HTTP/1.1 200 Connection "
                                             b"Established\r\n\r\n",
                              forward: bool = True) -> tuple[int, socket.socket]:
    """Bind a one-shot CONNECT proxy on 127.0.0.1, return `(port, srv)`.

    The proxy reads a single CONNECT request, sends `reply`, and (if
    `forward=True`) splices the client socket to a fresh TCP connection
    on `daemon_port` so the test can assert end-to-end framing through
    the tunnel. Setting `forward=False` lets a caller exercise the
    proxy-refused branch by passing a non-200 `reply`.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    proxy_port = srv.getsockname()[1]

    def _run():
        conn, _ = srv.accept()
        try:
            hdr = b""
            while b"\r\n\r\n" not in hdr:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                hdr += chunk
            conn.sendall(reply)
            if not forward:
                return
            upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            upstream.connect(("127.0.0.1", daemon_port))
            # Two-way splice. Threads die when either side closes.
            def _pump(a, b):
                try:
                    while True:
                        data = a.recv(65536)
                        if not data:
                            break
                        b.sendall(data)
                except OSError:
                    pass
                finally:
                    try: b.shutdown(socket.SHUT_WR)
                    except OSError: pass
            tA = threading.Thread(target=_pump, args=(conn, upstream),
                                  daemon=True)
            tB = threading.Thread(target=_pump, args=(upstream, conn),
                                  daemon=True)
            tA.start(); tB.start()
            tA.join(); tB.join()
            upstream.close()
        finally:
            conn.close()

    threading.Thread(target=_run, daemon=True).start()
    return proxy_port, srv


def test_open_daemon_socket_direct_when_loopback_listens(tmp_path):
    """When the host loopback actually has the daemon (codex backend,
    or replay/debug from the host shell), the direct path returns the
    connected socket without ever consulting $HTTP_PROXY. The
    `CONNECT_TIMEOUT_S` cap on the first attempt keeps the
    common case snappy.
    """
    port = _free_local_port()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    try:
        # HTTP_PROXY set but should be ignored on the success path.
        os.environ["HTTP_PROXY"] = "http://localhost:1"
        try:
            s = _harness_client._open_daemon_socket(port)
            # Accept side proves the tunnel target is the host loopback.
            conn, _ = srv.accept()
            conn.close()
            s.close()
        finally:
            os.environ.pop("HTTP_PROXY", None)
    finally:
        srv.close()


def test_harness_client_uses_connect_tunnel_when_loopback_refused(
        tmp_path, monkeypatch):
    """End-to-end of the Claude-sandbox recovery path. We simulate the
    sandbox's empty loopback by reserving a free port and *not* binding
    the daemon there — direct connect refuses. A separate CONNECT
    proxy on a different port forwards to the real daemon bound on a
    *third* port (mimicking how Claude's proxy reaches the host's
    127.0.0.1 from inside the sandbox netns).

    Then we point `_harness_client` at the refused port and
    `$HTTP_PROXY` at the proxy, and verify the round-trip succeeds. If
    the helper ever silently fell back to direct connect when the
    proxy was set, this test would still see the refused-port
    behavior and the assertion would fail.
    """
    # The "refused" port — kernel will RST on connect since nothing is
    # bound here for the duration of the test.
    refused_port = _free_local_port()

    # The actual daemon — bound on a different port so the proxy has
    # something to forward to. Wire-compatible echo server.
    daemon_port = _free_local_port()
    daemon = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    daemon.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    daemon.bind(("127.0.0.1", daemon_port))
    daemon.listen(1)

    def _serve_one():
        conn, _ = daemon.accept()
        try:
            (n,) = struct.unpack(">I", conn.recv(4))
            buf = b""
            while len(buf) < n:
                buf += conn.recv(n - len(buf))
            req = json.loads(buf.decode())
            resp = {"ok": True, "result": {"echoed": req}, "stdout": ""}
            data = json.dumps(resp).encode()
            conn.sendall(struct.pack(">I", len(data)) + data)
        finally:
            conn.close()
    threading.Thread(target=_serve_one, daemon=True).start()

    proxy_port, proxy_srv = _spawn_fake_connect_proxy(daemon_port)
    try:
        # Wire client at the refused port; rely on CONNECT to reach
        # daemon_port via the proxy.
        (tmp_path / ".harness.port").write_text(str(refused_port))
        monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
        monkeypatch.setattr(_harness_client, "PORT_PATH",
                            tmp_path / ".harness.port")
        # _connect_via_proxy issues `CONNECT 127.0.0.1:<refused_port>`,
        # but our fake proxy ignores the target and forwards to
        # daemon_port unconditionally — which models exactly what
        # Claude's proxy does (it resolves "localhost" on the host
        # side, so `127.0.0.1:<refused>` in the request maps to
        # whatever's listening on the host).
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy_port}")
        result = _harness_client._call("compile", {"freeze": True})
        assert result == {"echoed": {"method": "compile",
                                     "args": {"freeze": True}}}
    finally:
        proxy_srv.close()
        daemon.close()


def test_harness_client_proxy_refused_surfaces_status_line(tmp_path,
                                                            monkeypatch):
    """If `allowedDomains` doesn't include `localhost`, Claude's proxy
    returns `403 Forbidden` (or similar) on CONNECT. The client must
    surface the exact status line so the operator can fix the sandbox
    settings without guessing; a bare `ConnectionRefused` sends an agent
    after the wrong cause.
    """
    refused_port = _free_local_port()
    proxy_port, proxy_srv = _spawn_fake_connect_proxy(
        daemon_port=0,
        reply=b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n",
        forward=False,
    )
    try:
        (tmp_path / ".harness.port").write_text(str(refused_port))
        monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
        monkeypatch.setattr(_harness_client, "PORT_PATH",
                            tmp_path / ".harness.port")
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy_port}")
        with pytest.raises(SystemExit) as excinfo:
            _harness_client._call("compile", {})
        msg = str(excinfo.value)
        assert "403 Forbidden" in msg
        assert "allowedDomains" in msg
    finally:
        proxy_srv.close()


def test_harness_client_connect_target_is_localhost_not_ip(tmp_path,
                                                            monkeypatch):
    """The CONNECT request-line hostname must be the literal string
    `localhost`, not the IP `127.0.0.1`. Claude's proxy enforces
    `allowedDomains` as exact-string match on the request-line
    hostname, and Claude's settings docs explicitly forbid IP literals
    in `allowedDomains` — so `CONNECT 127.0.0.1:<port>` always returns
    HTTP 403 even when the sandbox is otherwise correctly configured.

    So the CONNECT line uses `PROXY_TARGET_HOST` (`localhost`), while
    direct connects keep `LOOPBACK_HOST` (`127.0.0.1`); with the IP
    literal, every harness call through the proxy would fail.
    """
    refused_port = _free_local_port()
    captured = {}
    proxy_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy_srv.bind(("127.0.0.1", 0))
    proxy_srv.listen(1)
    proxy_port = proxy_srv.getsockname()[1]

    def _run_capturing_proxy():
        conn, _ = proxy_srv.accept()
        try:
            hdr = b""
            while b"\r\n\r\n" not in hdr:
                chunk = conn.recv(4096)
                if not chunk: return
                hdr += chunk
            captured["request"] = hdr
            # Reply with 200 so the client treats CONNECT as succeeded
            # and tries to do a length-prefixed request; we don't have
            # to forward anywhere since the test asserts on the
            # captured request bytes before the client errors out on
            # the truncated tunnel.
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            conn.shutdown(socket.SHUT_WR)
        finally:
            conn.close()
    threading.Thread(target=_run_capturing_proxy, daemon=True).start()

    (tmp_path / ".harness.port").write_text(str(refused_port))
    monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
    monkeypatch.setattr(_harness_client, "PORT_PATH",
                        tmp_path / ".harness.port")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy_port}")
    # The call will fail downstream (proxy closes after the empty 200)
    # — we only care about the bytes the client put on the wire.
    try:
        _harness_client._call("compile", {})
    except (SystemExit, OSError):
        pass
    finally:
        proxy_srv.close()

    assert b"request" in captured.get("request", b"") or \
           captured.get("request"), \
        "proxy never received CONNECT request"
    req_line = captured["request"].split(b"\r\n", 1)[0]
    assert req_line.startswith(b"CONNECT localhost:"), (
        f"CONNECT must target 'localhost' (not an IP) so the proxy's "
        f"allowedDomains string match succeeds. Got: {req_line!r}"
    )
    assert b"CONNECT 127.0.0.1" not in req_line, (
        f"IP literal in CONNECT will be 403'd by Claude proxy. "
        f"Got: {req_line!r}"
    )
    # Host header must match the request-line target (per RFC 7230).
    host_lines = [l for l in captured["request"].split(b"\r\n")
                  if l.lower().startswith(b"host:")]
    assert host_lines, "CONNECT must include a Host header"
    assert b"localhost:" in host_lines[0]


def test_harness_client_no_proxy_fallback_errors_helpfully(tmp_path,
                                                            monkeypatch):
    """Direct-refused + `HTTP_PROXY` unset is a configuration bug (the
    launcher should always inherit Claude's `HTTP_PROXY=...` into the
    sandbox env). When it happens, we want a SystemExit pointing at
    `.daemon.log` rather than a stack trace from the missing-env path.
    """
    refused_port = _free_local_port()
    (tmp_path / ".harness.port").write_text(str(refused_port))
    monkeypatch.setattr(_harness_client, "RUN_DIR", tmp_path)
    monkeypatch.setattr(_harness_client, "PORT_PATH",
                        tmp_path / ".harness.port")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _harness_client._call("compile", {})
    msg = str(excinfo.value)
    assert "no $HTTP_PROXY set" in msg
    assert ".daemon.log" in msg


# ---------------------------------------------------------------------------
# Backend.execute argv/env construction — captured via intercepted run_agent_cli
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path, *, layer: str = "object",
               run_id: str = "260517_000000_test",
               model: str = "claude-opus-4-8",
               reasoning_effort: str | None = "high"):
    """Build a LaunchPlan for driving Backend.execute under an intercepted
    launch: the real dataclass, so a field `execute()` reads cannot be missing
    here and present in a launch."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    return LaunchPlan(
        layer=layer, user_prompt="a cube", run_dir=run_dir,
        runs_dir=tmp_path / "runs", run_id=run_id,
        model=model,
        env={"PATH": os.environ.get("PATH", "")},
        system_prompt_text="(stub system prompt)",
        footer_text="(stub footer)",
        reasoning_effort=reasoning_effort,
    )


@pytest.fixture(autouse=True)
def _codex_home_in_tmp(tmp_path, monkeypatch):
    """Codex launches here write their untrusted entry into a scratch account, never the user's `~/.codex`."""
    monkeypatch.setattr(CodexBackend, "HOME", str(tmp_path / "codex_home"))


def _intercept_launch(monkeypatch, sandbox_problem=None):
    """Replace `run_agent_cli` with a recorder that starts nothing.

    Returns the dict that will hold (file, argv, env) after Backend.execute
    runs. Also stubs `os.chdir` (so a missing run_dir doesn't break us),
    `shutil.which` (so the backend believes the binary is on PATH) and Codex's
    sandbox probe (`sandbox_problem` is what it reports; None: it confines).
    """
    captured: dict = {}
    monkeypatch.setattr("dycraft.cli.backends.codex.codex_sandbox_probe",
                        lambda *a, **k: sandbox_problem)
    # A failing probe tries the fallback TMPDIRs, one of them under the user's cache: kept off the real home.
    import tempfile
    monkeypatch.setenv("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="dycraft-test-xdg-"))

    def _fake_launch(argv, env, watch=None):
        captured["file"] = argv[0]
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        captured["watch"] = watch
        return 0

    monkeypatch.setattr("dycraft.cli.backends._process.run_agent_cli", _fake_launch)
    monkeypatch.setattr("os.chdir", lambda _p: None)
    monkeypatch.setattr("shutil.which", lambda _name: f"/fake/bin/{_name}")
    return captured


def test_claude_execute_uses_dont_ask_permission_mode(tmp_path, monkeypatch):
    """Default permission-mode must be `dontAsk`, not `acceptEdits` or `default`.

    `acceptEdits` blanket-auto-approves Edit/Write everywhere, which
    would defeat the path-scoped allow rules that are this backend's
    only line of defense against arbitrary writes (bwrap doesn't cover
    Edit/Write per Claude Code's sandboxing docs). Under `dontAsk`,
    Edit/Write run ONLY when an allow rule matches — i.e. only inside the
    run dir — and anything else is denied at once; `default` would leave
    it waiting on a prompt no one in an unattended run answers.
    `test_claude_runtime_blocks_write_outside_run_dir` proves the runtime
    consequence against the real binary.
    """
    captured = _intercept_launch(monkeypatch)
    ClaudeBackend().execute(_make_plan(tmp_path))
    argv = captured["argv"]
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"


def test_claude_execute_forwards_model_and_reasoning_effort(
        tmp_path, monkeypatch):
    """`--model` / `--reasoning-effort` reach the claude argv as `--model` /
    `--effort`.

    `execvpe` is the last thing the launcher does, so capturing its arguments
    is the whole claim, with no real agent needed;
    what the OS does afterwards is not DyCraft's to test.
    """
    captured = _intercept_launch(monkeypatch)
    ClaudeBackend().execute(
        _make_plan(tmp_path, model="claude-opus-4-8", reasoning_effort="high"))
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert argv[argv.index("--effort") + 1] == "high"


def test_claude_execute_inherits_none_of_the_users_mcp_servers(tmp_path, monkeypatch):
    """`--strict-mcp-config` with no `--mcp-config` means "no MCP servers":
    without it the agent would load whatever servers the user's own account
    configures, outside the run's sandbox and permission rules."""
    captured = _intercept_launch(monkeypatch)
    ClaudeBackend().execute(_make_plan(tmp_path))
    argv = captured["argv"]
    assert "--strict-mcp-config" in argv
    assert "--mcp-config" not in argv


@pytest.mark.parametrize("layer", ["object", "scene"])
def test_the_agent_is_told_dycrafts_interpreter_and_no_development_environment(layer, tmp_path):
    """The agent runs `import harness as h` with `$DYCRAFT_PYTHON`; text naming
    a developer's conda env would send it to an interpreter that may not exist
    on the user's machine."""
    from dycraft.agent.agent_runner import compose_footer, compose_system_prompt

    system, footer = compose_system_prompt(layer), compose_footer(layer, "t", tmp_path)
    assert '"$DYCRAFT_PYTHON"' in system
    assert f"`$DYCRAFT_RUN_DIR={tmp_path}`" in footer
    assert "conda" not in system and "conda" not in footer


def test_claude_execute_omits_effort_when_unset(tmp_path, monkeypatch):
    """No `--reasoning-effort` from the user means no `--effort` on the argv —
    an empty or literal `None` value would be forwarded as a real setting."""
    captured = _intercept_launch(monkeypatch)
    ClaudeBackend().execute(_make_plan(tmp_path, reasoning_effort=None))
    assert "--effort" not in captured["argv"]


def _claude_sandbox_on(tmp_path, monkeypatch, sysctl: str | None) -> tuple[dict, str]:
    """The `sandbox` block and run dir `ClaudeBackend.execute` hands claude on
    a host whose `apparmor_restrict_unprivileged_userns` reads `sysctl`
    (None: the file does not exist, as on a kernel without AppArmor)."""
    import dycraft.cli.backends as bmod
    knob = tmp_path / "apparmor_restrict_unprivileged_userns"
    if sysctl is not None:
        knob.write_text(sysctl + "\n")
    monkeypatch.setattr(bmod.claude, "_USERNS_RESTRICT_SYSCTL", knob)
    captured = _intercept_launch(monkeypatch)
    plan = _make_plan(tmp_path)
    ClaudeBackend().execute(plan)
    argv = captured["argv"]
    return json.loads(argv[argv.index("--settings") + 1])["sandbox"], str(plan.run_dir)


@pytest.mark.parametrize("sysctl", ["0", None])
def test_claude_sandbox_keeps_the_socket_block_where_userns_is_open(
        tmp_path, monkeypatch, sysctl):
    """Where nested user namespaces work, Claude Code's own seccomp block on
    Unix sockets starts, so the settings must not trade it away."""
    monkeypatch.delenv("DYCRAFT_CLAUDE_UNIX_SOCKETS", raising=False)
    sandbox, _ = _claude_sandbox_on(tmp_path, monkeypatch, sysctl)
    assert "allowAllUnixSockets" not in sandbox["network"]
    assert "denyRead" not in sandbox["filesystem"]


def test_claude_sandbox_masks_escape_sockets_where_userns_is_restricted(
        tmp_path, monkeypatch):
    """On a host with `apparmor_restrict_unprivileged_userns = 1` the seccomp
    helper cannot start and every sandboxed command fails, so the settings
    skip it (`allowAllUnixSockets`) and mask the sockets that would run
    commands outside the sandbox instead. Write confinement and the
    localhost-only network must come through unchanged.
    `test_claude_runtime_sandboxed_bash_runs_without_escape_sockets` checks
    the same against the real binary.
    """
    monkeypatch.delenv("DYCRAFT_CLAUDE_UNIX_SOCKETS", raising=False)
    sandbox, run_dir = _claude_sandbox_on(tmp_path, monkeypatch, "1")
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]
    assert sandbox["allowUnsandboxedCommands"] is False
    assert sandbox["filesystem"]["allowWrite"] == [run_dir]
    assert sandbox["network"]["allowedDomains"] == ["localhost"]
    assert sandbox["network"]["allowAllUnixSockets"] is True
    uid = os.getuid()
    for path in ("/run/user", "/run/dbus", "/run/docker.sock", "/tmp/.X11-unix",
                 f"/tmp/tmux-{uid}", f"/tmp/cc-daemon-{uid}"):
        assert path in sandbox["filesystem"]["denyRead"], path


@pytest.mark.parametrize("mode, sysctl, applied", [("block", "1", False),
                                                    ("allow", "0", True)])
def test_claude_sandbox_socket_fallback_override(
        tmp_path, monkeypatch, mode, sysctl, applied):
    """`DYCRAFT_CLAUDE_UNIX_SOCKETS` beats the detection both ways: `block` for
    a host where an AppArmor profile lets the helper run, `allow` for one the
    sysctl does not describe."""
    monkeypatch.setenv("DYCRAFT_CLAUDE_UNIX_SOCKETS", mode)
    sandbox, _ = _claude_sandbox_on(tmp_path, monkeypatch, sysctl)
    assert ("allowAllUnixSockets" in sandbox["network"]) is applied
    assert ("denyRead" in sandbox["filesystem"]) is applied


def test_codex_execute_forwards_model_and_reasoning_effort(
        tmp_path, monkeypatch):
    """Codex spells the same two settings `-m` and
    `-c model_reasoning_effort=<v>`. Pinned here rather than through a real
    codex run for the same reason as the claude case above — and codex
    *silently ignores* an unknown `-c` key, so a typo in that config name
    cannot be caught by observing the run's behavior at all.
    """
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    captured = _intercept_launch(monkeypatch)
    CodexBackend().execute(
        _make_plan(tmp_path, model="gpt-5.5", reasoning_effort="high"))
    argv = captured["argv"]
    assert argv[argv.index("-m") + 1] == "gpt-5.5"
    assert "model_reasoning_effort=high" in argv


def test_codex_execute_is_sandboxed_with_no_way_out_and_network_access(
        tmp_path, monkeypatch):
    """Where the sandbox works, Codex runs `--sandbox workspace-write
    --ask-for-approval never`: a command the sandbox refuses fails instead of
    going to Codex's automatic reviewer, which could approve it outside the
    sandbox. Network access is on, so the harness shells reach the daemon."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    captured = _intercept_launch(monkeypatch)
    CodexBackend().execute(_make_plan(tmp_path))

    argv = captured["argv"]
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert argv[argv.index("--ask-for-approval") + 1] == "never"
    assert "--approve-for-me" not in argv
    c_args = [argv[i + 1] for i, t in enumerate(argv) if t == "-c"]
    assert "sandbox_workspace_write.network_access=true" in c_args
    # Where an organization overrides `never` with on-request, the reviewer answers, never a prompt.
    assert 'approvals_reviewer="auto_review"' in c_args
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv


def test_codex_refuses_to_start_where_its_sandbox_does_not_work(tmp_path, monkeypatch, capsys):
    """Fail closed: no agent starts, and the message names what failed and the opt-in."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.delenv("DYCRAFT_ALLOW_UNSANDBOXED_CODEX", raising=False)
    monkeypatch.delenv("DYCRAFT_ALLOW_UNSANDBOXED", raising=False)
    captured = _intercept_launch(monkeypatch, sandbox_problem="the sandbox failed to start: bwrap: no")
    assert CodexBackend().execute(_make_plan(tmp_path)) == 2
    assert "argv" not in captured
    err = capsys.readouterr().err
    # Every setup tried is named with what it reported.
    assert "result: it failed in every setup tried: TMPDIR=" in err and "the sandbox failed to start: bwrap: no" in err
    assert "a /tmp of its own (unshare -r --mount, TMPDIR=" in err
    assert "pass --allow-unsandboxed to run Codex without its sandbox" in err


@pytest.mark.parametrize("name", ["DYCRAFT_ALLOW_UNSANDBOXED", "DYCRAFT_ALLOW_UNSANDBOXED_CODEX"])
def test_the_opt_in_starts_codex_with_its_reviewer_and_records_it(name, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setenv(name, "1")
    captured = _intercept_launch(monkeypatch, sandbox_problem="the sandbox failed to start: bwrap: no")
    backend = CodexBackend()
    assert backend.execute(_make_plan(tmp_path)) == 0
    assert "--approve-for-me" in captured["argv"] and "--ask-for-approval" not in captured["argv"]
    # Unwrapped: the namespace did not make the sandbox work either.
    assert captured["argv"][0].endswith("codex")
    recorded = backend.launch_meta["codex_sandbox"]
    assert (recorded["mode"], recorded["approval"]) == ("unsandboxed (--allow-unsandboxed)",
                                                        "on-request (automatic reviewer)")
    assert recorded["probe"].startswith("it failed in every setup tried: TMPDIR=")
    assert "the sandbox failed to start: bwrap: no" in recorded["probe"]
    # The README's one line gets the first setup's failure; meta.json keeps every setup tried.
    assert backend.launch_meta["unsandboxed"] == {"agent": "Codex",
                                                  "reason": "the sandbox failed to start: bwrap: no"}
    assert "WARNING: running Codex without a sandbox (--allow-unsandboxed): its sandbox does not work here (" \
        in capsys.readouterr().err


def test_the_opt_in_leaves_a_working_codex_sandbox_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED", "1")
    captured = _intercept_launch(monkeypatch, sandbox_problem=None)
    backend = CodexBackend()
    assert backend.execute(_make_plan(tmp_path)) == 0
    assert "workspace-write" in captured["argv"] and "unsandboxed" not in backend.launch_meta
    assert "[cli] --allow-unsandboxed: Codex's sandbox works here, so it stays on" in capsys.readouterr().err


@pytest.mark.parametrize("sandbox_problem", [None, "the sandbox failed to start: bwrap: no"],
                         ids=["sandboxed", "opted-in reviewer"])
def test_every_codex_launch_switches_its_keypress_notices_off(sandbox_problem, tmp_path, monkeypatch):
    """Codex's "Update available" screen and its mid-run "switch model for lower credit usage?" prompt both wait for
    an Enter that nobody presses in a headless run. Both launch modes switch them off. Codex ignores an unknown `-c`
    key, so the keys' spelling is what this pins (both are in codex 0.153.4's binary)."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setenv("DYCRAFT_ALLOW_UNSANDBOXED_CODEX", "1")
    captured = _intercept_launch(monkeypatch, sandbox_problem=sandbox_problem)
    assert CodexBackend().execute(_make_plan(tmp_path)) == 0
    argv = captured["argv"]
    c_args = [argv[i + 1] for i, t in enumerate(argv[:-1]) if t == "-c"]
    assert "check_for_update_on_startup=false" in c_args
    assert "notice.hide_rate_limit_model_nudge=true" in c_args


def test_a_preflight_already_run_is_not_repeated(tmp_path, monkeypatch):
    """`cli.main` runs the preflight before the rewriters; `execute` probes only when nobody did."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    _intercept_launch(monkeypatch)
    calls = []
    monkeypatch.setattr("dycraft.cli.backends.codex.codex_sandbox_probe",
                        lambda *a, **k: calls.append(a) or None)
    backend = CodexBackend()
    plan = _make_plan(tmp_path)
    assert backend.preflight(plan.run_dir) is None
    assert backend.launch_meta == {"codex_sandbox": {"probe": "ok", "mode": "workspace-write",
                                                     "approval": "never"}}
    backend.execute(plan)
    assert len(calls) == 1


def test_codex_records_the_run_dir_untrusted_only_while_it_runs(tmp_path, monkeypatch):
    """The trust screen is answered with `untrusted`, so Codex loads no project config from the run directory,
    and the entry this launch added is gone afterwards."""
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    _intercept_launch(monkeypatch)
    cfg = tmp_path / "codex_home" / "config.toml"
    seen = {}
    monkeypatch.setattr("dycraft.cli.backends._process.run_agent_cli",
                        lambda argv, env, watch=None: seen.setdefault("cfg", cfg.read_text()) and 0)
    plan = _make_plan(tmp_path)
    CodexBackend().execute(plan)
    assert f'[projects."{plan.run_dir.resolve()}"]\ntrust_level = "untrusted"' in seen["cfg"]
    assert "projects" not in cfg.read_text()


def test_codex_execute_scopes_working_root_to_run_dir(
        tmp_path, monkeypatch):
    """Codex workspace-write writes to its working directory (`cwd`), not
    just to `--add-dir` paths. Therefore the launcher must set Codex's
    working root to the run dir itself; pointing `-C` at the repo root
    would silently broaden writes across the whole checkout.

    We also keep `--add-dir <run_dir>` because the harness watchdog
    discovers the live agent pid by scanning /proc for that exact argv
    pair. Pin both properties here.
    """
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    captured = _intercept_launch(monkeypatch)
    plan = _make_plan(tmp_path)
    CodexBackend().execute(plan)

    argv = captured["argv"]
    assert argv[argv.index("-C") + 1] == str(plan.run_dir)
    assert argv[argv.index("--add-dir") + 1] == str(plan.run_dir)


def test_codex_execute_uses_user_codex_home(tmp_path, monkeypatch):
    """Codex is launched with CODEX_HOME pointed at the user's real
    `~/.codex` (CodexBackend.HOME), NOT a per-run staged home.

    Write-confinement is enforced by `-C <run_dir>` + `-s workspace-write`
    + `--add-dir <run_dir>` (asserted in the argv tests) and is independent
    of CODEX_HOME. Reusing the real home carries over the user's trust
    entries and model-migration acknowledgements, so codex's interactive
    TUI does not wedge on a trust/migration prompt at startup. The tradeoff
    is intentional: a sandbox-widening key in the user's config now applies.
    """
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    user_home = tmp_path / "user_codex_home"
    user_home.mkdir()
    (user_home / "auth.json").write_text('{"token":"secret"}')
    monkeypatch.setattr(CodexBackend, "HOME", str(user_home))

    captured = _intercept_launch(monkeypatch)
    plan = _make_plan(tmp_path)
    CodexBackend().execute(plan)

    assert captured["env"]["CODEX_HOME"] == str(user_home)
    # No per-run staged home is created inside the run dir.
    assert not (plan.run_dir / ".codex_cli_home").exists()


def test_codex_execute_resolves_codex_via_which_when_env_unset(
        tmp_path, monkeypatch):
    """With DYCRAFT_CODEX_EXECUTABLE unset, execute() must resolve the
    codex binary via `shutil.which("codex")`. Pinning this behavior
    prevents regressions to a hardcoded per-machine path that would FATAL
    on any other host.
    """
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    captured = _intercept_launch(monkeypatch)
    monkeypatch.delenv("DYCRAFT_CODEX_EXECUTABLE", raising=False)
    CodexBackend().execute(_make_plan(tmp_path))
    # The argv's argv[0] is the binary the CLI is started as.
    # With the env override unset, execute() resolves through
    # `shutil.which("codex")` which our intercept returns as "/fake/bin/codex".
    assert captured["argv"][0] == "/fake/bin/codex"


def test_codex_execute_pty_spawns_when_stdin_is_not_tty(tmp_path, monkeypatch):
    """When `os.isatty(stdin)` is False, codex bails with `Error: stdin is
    not a terminal`. CodexBackend.execute compensates by calling
    `_spawn_with_pty_term_shim(cmd, env)` instead of `run_agent_cli`, which
    allocates a PTY for the codex child, pumps master ↔ launcher-stdout, and
    answers the startup terminal-capability probes codex would otherwise wedge
    on.

    We verify the wrap fires by capturing the argv the shim is called
    with when `_stdin_is_tty` is monkeypatched to False, AND that
    `run_agent_cli` is NOT called (the wrap path is mutually exclusive with the
    no-wrap exec path; otherwise the launcher would try to do both).
    """
    monkeypatch.delenv("DYCRAFT_CODEX_NO_PTY_WRAP", raising=False)
    monkeypatch.setattr("dycraft.cli.backends._process._stdin_is_tty",
                        lambda: False)
    monkeypatch.setattr("dycraft.cli.backends._process._stdout_is_tty",
                        lambda: True)
    monkeypatch.setattr("dycraft.cli.backends._process._stderr_is_tty",
                        lambda: True)
    captured = _intercept_launch(monkeypatch)
    spawned = {}
    def _fake_shim(cmd, env):
        spawned["argv"] = list(cmd)
        return 0  # mimic clean exit
    monkeypatch.setattr("dycraft.cli.backends._process._spawn_with_pty_term_shim",
                        _fake_shim)

    rc = CodexBackend().execute(_make_plan(tmp_path))
    assert rc == 0
    assert "argv" in spawned, "PTY shim was not invoked"
    assert "argv" not in captured, \
        "wrap path must not also run the CLI directly"
    argv = spawned["argv"]
    assert "codex" in argv[0] or argv[0].endswith("codex")
    assert argv[argv.index("--ask-for-approval") + 1] == "never"
    assert "sandbox_workspace_write.network_access=true" in argv


def test_codex_execute_pty_spawns_when_stdout_is_not_tty(
        tmp_path, monkeypatch):
    """Nested harness launches use subprocess.run(capture_output=True),
    which leaves stdin alone but replaces stdout/stderr with pipes.
    Codex rejects that with `Error: stdout is not a terminal`, so the
    PTY wrap must also fire when stdout lacks a tty even if stdin is
    still interactive.
    """
    monkeypatch.delenv("DYCRAFT_CODEX_NO_PTY_WRAP", raising=False)
    monkeypatch.setattr("dycraft.cli.backends._process._stdin_is_tty",
                        lambda: True)
    monkeypatch.setattr("dycraft.cli.backends._process._stdout_is_tty",
                        lambda: False)
    monkeypatch.setattr("dycraft.cli.backends._process._stderr_is_tty",
                        lambda: True)
    captured = _intercept_launch(monkeypatch)
    spawned = {}
    def _fake_shim(cmd, env):
        spawned["argv"] = list(cmd)
        return 0
    monkeypatch.setattr("dycraft.cli.backends._process._spawn_with_pty_term_shim",
                        _fake_shim)

    rc = CodexBackend().execute(_make_plan(tmp_path))
    assert rc == 0
    assert "argv" in spawned, "PTY shim was not invoked for stdout pipe case"
    assert "argv" not in captured, \
        "wrap path must not also run the CLI directly"


def test_codex_execute_pty_wrap_disabled_when_stdin_is_tty(
        tmp_path, monkeypatch):
    """Inverse case: when stdin is a tty (normal interactive terminal),
    the wrap is skipped and we go straight through `run_agent_cli`. The
    extra Python process from `pty.spawn` is unnecessary when codex
    can inherit the launcher's tty.
    """
    monkeypatch.delenv("DYCRAFT_CODEX_NO_PTY_WRAP", raising=False)
    monkeypatch.setattr("dycraft.cli.backends._process._stdin_is_tty",
                        lambda: True)
    monkeypatch.setattr("dycraft.cli.backends._process._stdout_is_tty",
                        lambda: True)
    monkeypatch.setattr("dycraft.cli.backends._process._stderr_is_tty",
                        lambda: True)
    captured = _intercept_launch(monkeypatch)
    spawned = {}
    monkeypatch.setattr(
        "dycraft.cli.backends._process._spawn_with_pty_term_shim",
        lambda cmd, env: spawned.setdefault("argv", list(cmd)))
    CodexBackend().execute(_make_plan(tmp_path))
    assert "argv" not in spawned, "PTY shim should be skipped on tty stdin"
    assert captured["argv"][0].endswith("codex") or "codex" in captured["argv"][0]


def test_codex_execute_pty_wrap_opt_out_env_var(tmp_path, monkeypatch):
    """`DYCRAFT_CODEX_NO_PTY_WRAP=1` disables the wrap even without a tty.
    Escape hatch for environments where something upstream already
    provides a PTY (or where codex tolerates non-tty stdin).
    """
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setattr("dycraft.cli.backends._process._stdin_is_tty",
                        lambda: False)
    monkeypatch.setattr("dycraft.cli.backends._process._stdout_is_tty",
                        lambda: False)
    monkeypatch.setattr("dycraft.cli.backends._process._stderr_is_tty",
                        lambda: False)
    captured = _intercept_launch(monkeypatch)
    spawned = {}
    monkeypatch.setattr(
        "dycraft.cli.backends._process._spawn_with_pty_term_shim",
        lambda cmd, env: spawned.setdefault("argv", list(cmd)))
    CodexBackend().execute(_make_plan(tmp_path))
    assert "argv" not in spawned
    assert captured["argv"][0].endswith("codex") or "codex" in captured["argv"][0]


def test_pty_term_shim_answers_startup_probe(tmp_path):
    """The shim must answer the terminal-capability probes codex blocks on at
    startup. We run a child that emits a cursor-position probe (`ESC[6n`) then
    blocks reading its stdin — exactly codex's deadlock. Without the shim the
    read never returns; with it, the child receives the canned reply and can
    proceed. This is the behavior that un-wedges codex.
    """
    import os
    import sys

    from dycraft.cli.backends._process import _spawn_with_pty_term_shim

    got = tmp_path / "reply.bin"
    # raw mode so the single-byte-ish reply isn't held by line discipline
    child = (
        "import os,sys,tty;"
        "tty.setraw(0);"
        "os.write(1, b'\\x1b[6n');"           # emit DSR cursor probe
        "data = os.read(0, 64);"              # blocks until the shim answers
        "open(sys.argv[1], 'wb').write(data)"
    )
    rc = _spawn_with_pty_term_shim(
        [sys.executable, "-c", child, str(got)], dict(os.environ))
    assert rc == 0
    # the shim's canned cursor-position reply reached the child
    assert got.read_bytes() == b"\x1b[40;1R"


def test_launcher_sets_agent_cmdline_marker_env_for_daemon(monkeypatch):
    """The watchdog can only retarget when the daemon receives the marker.
    The launcher reads it from `backend.agent_cmdline_marker` and stuffs
    it into the daemon's env via `env["DYCRAFT_AGENT_CMDLINE_MARKER"]`.
    A new daemon-backed backend that forgets to declare a marker silently
    breaks the watchdog (see `cli/README.md`, "Code map") — this test
    catches that regression by running the launcher's env-population logic
    on the real backend objects and asserting the result.
    """
    # Replicate cli.py main's marker injection on each backend.
    for cls in (ClaudeBackend, CodexBackend):
        env: dict[str, str] = {}
        if cls.agent_cmdline_marker:
            env["DYCRAFT_AGENT_CMDLINE_MARKER"] = cls.agent_cmdline_marker
        # Production daemon must be able to find this on os.environ.get()
        assert env.get("DYCRAFT_AGENT_CMDLINE_MARKER") == cls.name, (
            f"{cls.__name__} marker doesn't match its name; the launcher "
            f"would pass {env.get('DYCRAFT_AGENT_CMDLINE_MARKER')!r} but "
            f"the daemon's /proc scan needs to match {cls.name!r}")


# ---------------------------------------------------------------------------
# Sub-CLI propagation — model + backend pass-through
# ---------------------------------------------------------------------------


def _drive_cli_main_to_launch(argv, monkeypatch, tmp_path):
    """Drive `cli.main(argv)` through to the agent CLI launch with all
    heavy I/O stubbed; return the env dict that would have been handed
    to the agent CLI binary.

    What's stubbed:
      - harness daemon / self-kill watcher launches (each
        normally spawns a subprocess; here they're no-ops).
      - os.chdir / shutil.which / backends.run_agent_cli (replaced with capture).
      - The harness client stager (writes a real file, harmless but
        unnecessary).
      - compose_system_prompt / compose_footer (read prompt files on
        disk; we don't care about their contents and stubbing keeps
        the test hermetic).

    What's NOT stubbed: argument parsing, the sub_model/sub_backend
    fallback expressions, build_env_dict, the LaunchPlan construction,
    and ClaudeBackend.execute up to the launch — exactly the
    code path that decides what DYCRAFT_SUB_* values cross the
    process boundary.
    """
    from dycraft.cli import cli as cli_mod

    # **Isolate the real environment first.** `cli.main` exports into
    # `os.environ` on its way to `execvpe` — the CAD interpreter,
    # DYCRAFT_RENDER_BACKEND, and the rest — which is correct in
    # production, where the exec happens microseconds later and the process is
    # replaced. Here the exec is stubbed, so those writes outlive the test and
    # leak into every test that follows in the same process.
    #
    # A leaked `DYCRAFT_RENDER_BACKEND=blender`, for one, would make later
    # tests render through Cycles, find no Blender interpreter, and fail with
    # `rendering unavailable` while passing in isolation. Swapping in a plain
    # dict keeps `main`'s writes visible to the code under test (everything
    # reads `os.environ` at call time) and lets monkeypatch put the real
    # mapping back afterwards.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    captured: dict = {}

    def _fake_launch(argv, env, watch=None):
        captured["file"] = argv[0]
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        captured["watch"] = watch
        return 0

    monkeypatch.setattr("dycraft.cli.backends._process.run_agent_cli", _fake_launch)
    monkeypatch.setattr("os.chdir", lambda _p: None)
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(cli_mod, "stage_harness_client",
                        lambda *a, **kw: None)
    monkeypatch.setattr(cli_mod, "start_harness_daemon",
                        lambda *a, **kw: None)
    monkeypatch.setattr(cli_mod, "start_self_kill_watcher",
                        lambda *a, **kw: None)
    monkeypatch.setattr(cli_mod, "compose_system_prompt",
                        lambda *a, **kw: "(stub system prompt)")
    monkeypatch.setattr(cli_mod, "compose_footer",
                        lambda *a, **kw: "(stub footer)")
    # Redirect runs to tmp_path so we don't pollute the repo.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    rc = cli_mod.main(argv)
    assert "env" in captured, (
        f"cli.main did not reach the launch (rc={rc}); the stubs may have "
        f"diverged from cli.py's call graph.")
    return captured


def test_cli_run_meta_records_reasoning_effort(tmp_path, monkeypatch):
    """The viewer's durable source is the exact external-CLI launch setting."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _drive_cli_main_to_launch([
        "--backend", "claude",
        "--layer", "object",
        "--model", "claude-opus-4-8",
        "--reasoning-effort", "xhigh",
        "--runs-dir", str(runs_dir),
        "--run-id", "reasoning_meta",
        "--prompt", "a prompt",
    ], monkeypatch, tmp_path)

    from dycraft.run_layout import internal_dir
    meta = json.loads((internal_dir(runs_dir / "reasoning_meta") / "meta.json").read_text())
    assert meta["reasoning_effort"] == "xhigh"


def test_the_launch_env_keeps_the_shells_proxy(tmp_path, monkeypatch):
    """A host that reaches the model APIs only through `HTTPS_PROXY` must hand
    it to the agent CLI, or the agent cannot connect at all."""
    from dycraft.cli.cli import build_env_dict
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    env = build_env_dict("p", run_dir=tmp_path, runs_dir=tmp_path.parent,
                         run_id="r", layer="object", runtime="claude")
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"


def test_sub_cli_without_render_backend_inherits_the_env(
        tmp_path, monkeypatch):
    """An inherited `DYCRAFT_RENDER_BACKEND` is now honored, deliberately.

    Resolution is flag > ambient env > `DEFAULT_RENDER_BACKEND` (see
    `cli.py::main`), so `export DYCRAFT_RENDER_BACKEND=blender` sets a
    machine-wide default and an explicit flag still wins.
    """
    from dycraft.cli.cli import build_env_dict
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "blender")
    env = build_env_dict(
        "a prompt", run_dir=tmp_path, runs_dir=tmp_path.parent, run_id="r",
        layer="scene", runtime="claude", render_backend=None)
    assert env["DYCRAFT_RENDER_BACKEND"] == "blender"
    # An explicit flag still wins, so `--render-backend open3d` cannot be
    # quietly overridden by the launching shell.
    explicit = build_env_dict(
        "a prompt", run_dir=tmp_path, runs_dir=tmp_path.parent, run_id="r",
        layer="scene", runtime="claude", render_backend="open3d")
    assert explicit["DYCRAFT_RENDER_BACKEND"] == "open3d"


@pytest.mark.parametrize("flag,ambient,expected", [
    (["--render-backend", "blender"], None, "blender"),
    (["--render-backend", "open3d"], None, "open3d"),
    # No flag and no ambient value: the code default, recorded concretely
    # rather than as null.
    ([], None, "open3d"),
    # No flag but an ambient value: honored (see
    # `test_sub_cli_without_render_backend_inherits_the_env` for why).
    ([], "blender", "blender"),
    # An explicit flag is never overridden by the shell.
    (["--render-backend", "open3d"], "blender", "open3d"),
])
def test_meta_json_records_the_render_backend(flag, ambient, expected, tmp_path,
                                              monkeypatch):
    """Every run must say in `meta.json` which renderer drew its pictures.

    An absent key is indistinguishable from `open3d`, so the only way to tell
    what a finished run used would be forensic. Blender stamps
    `cycles.ViewLayer.samples` into its PNG metadata and Open3D writes a bare
    IHDR+IDAT stream, which identifies a Cycles run and can never positively
    identify a Filament one. Comparing two runs across backends — the whole
    point of having two — needs the answer recorded, not inferred.

    The defaulted case is the one that matters most and is therefore
    parametrized alongside the explicit ones: `--render-backend` defaults to
    `None`, so writing `args.render_backend` straight through would put `null`
    in the file and leave the ambiguity exactly where it was.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # meta.json must agree with whatever `cli.main` resolved — flag first,
    # then the ambient shell, then the code default — never with `None`.
    if ambient is None:
        monkeypatch.delenv("DYCRAFT_RENDER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", ambient)
    argv = [
        "--backend", "claude",
        "--layer", "object",
        "--model", "claude-opus-4-8",
        "--runs-dir", str(runs_dir),
        "--exp-name", "meta_render_backend",
        *flag,
        "--prompt", "a prompt",
    ]
    _drive_cli_main_to_launch(argv, monkeypatch, tmp_path)

    metas = list(runs_dir.glob("*/src/meta.json"))           # a launch's records are in `src/` (`run_layout`)
    assert len(metas) == 1, f"expected exactly one run dir, got {metas}"
    meta = json.loads(metas[0].read_text())
    assert meta.get("render_backend") == expected, (
        f"meta.json must record the renderer this run selected: expected "
        f"{expected!r}, got {meta.get('render_backend')!r}. Without it a "
        f"finished run cannot be attributed to a backend after the fact.")


# Watchdog retargeting is tested in isolation, by
# `test_watchdog_retargets_to_discovered_agent` and
# `test_watchdog_shuts_down_when_retargeted_pid_dies`, which spin up both
# ends of the watchdog/agent relationship inside one test process tree. A
# test that launched a real agent could not assert it: the daemon's /proc
# sees the agent only when the test runner shares its PID namespace, and a
# pytest run nested in another bwrap (Claude Code's bash tool, some
# container CI runners) does not.


# ---------------------------------------------------------------------------
# Waiting on a job: what the client does so an agent need not
# ---------------------------------------------------------------------------

def _stub_exchange(monkeypatch, script):
    """Replace the socket round trip with a scripted sequence of responses."""
    from dycraft.agent import _harness_core as core
    calls = []

    def exchange(method, args, timeout):
        calls.append((method, args))
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(core, "_exchange", exchange)
    monkeypatch.setattr(core, "POLL_S", 0.0)
    monkeypatch.setattr(core, "RETRY_S", 0.0)
    return core, calls


def test_an_operation_waits_for_its_own_job_instead_of_returning_an_id(monkeypatch):
    """Without it, an agent writes `while True: job_result(...); sleep(n)`
    around every long operation. The wait belongs here, so the caller gets an
    outcome."""
    core, calls = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "queued"}},
        {"ok": True, "result": {"job_id": "abc", "status": "running"}},
        {"ok": True, "result": {"job_id": "abc", "status": "completed",
                                "result": {"ok": True}}},
    ])
    record = core._call("storyboard", {})
    assert record["status"] == "completed" and record["result"]["ok"] is True
    assert [m for m, _ in calls] == ["storyboard", "job_result", "job_result"]


def test_a_job_outliving_the_wait_comes_back_pending_to_be_polled(monkeypatch):
    """The fallback: a render longer than the bound must not block forever,
    and its id has to survive so polling still works."""
    # A wait that expires: the submit lands, one poll says `running`, the
    # budget is gone, and the pending record comes back with its id intact.
    core, calls = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "queued"}},
        {"ok": True, "result": {"job_id": "abc", "status": "running"}},
    ])
    # Drive the clock rather than racing it: one poll, then the budget is up.
    ticks = iter([0.0, 100.0, 100.0])
    monkeypatch.setattr(core.time, "monotonic", lambda: next(ticks))
    record = core._call("storyboard", {}, wait=1.0)
    assert record["status"] == "running" and record["job_id"] == "abc"
    assert [m for m, _ in calls] == ["storyboard", "job_result"]
    # `WAIT_S = 0` opts out entirely and hands back the submit record.
    core2, calls2 = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "queued"}}])
    monkeypatch.setattr(core2, "WAIT_S", 0.0)
    assert core2._call("storyboard", {})["status"] == "queued"
    assert len(calls2) == 1


def test_polling_a_job_directly_does_not_recurse(monkeypatch):
    """`job_result` returns a record carrying a `job_id`; waiting on that
    would call `job_result` again, forever."""
    core, calls = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "running"}}])
    assert core._call("job_result", {"job_id": "abc"})["status"] == "running"
    assert len(calls) == 1


class _FakeSocket:
    """Answers one framed request with a canned response."""

    def __init__(self, payload):
        self._out = struct.pack(">I", len(payload)) + payload
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        chunk, self._out = self._out[:n], self._out[n:]
        return chunk

    def close(self):
        pass


def _stub_socket(monkeypatch, script):
    """Script the socket layer itself, so `_exchange`'s own retry is exercised."""
    from dycraft.agent import _harness_core as core
    opened = []

    def open_socket(port, timeout=None):
        opened.append(port)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return _FakeSocket(json.dumps(step).encode("utf-8"))

    monkeypatch.setattr(core, "_read_port", lambda: 4242)
    monkeypatch.setattr(core, "_open_daemon_socket", open_socket)
    monkeypatch.setattr(core, "RETRY_S", 0.0)
    monkeypatch.setattr(core, "WAIT_S", 0.0)
    return core, opened


def test_a_dropped_socket_is_retried_rather_than_raised(monkeypatch):
    """A dropped daemon socket is retried, so an agent does not wrap every
    call in its own `except ConnectionError: sleep(...)` loop."""
    core, opened = _stub_socket(monkeypatch, [
        ConnectionResetError("daemon went away"),
        {"ok": True, "result": {"ok": True}},
    ])
    assert core._call("probe", {}) == {"ok": True}
    assert len(opened) == 2


def test_a_read_timeout_is_not_retried(monkeypatch):
    """A timeout means the daemon is still working. Retrying would submit the
    same expensive operation a second time."""
    core, opened = _stub_socket(monkeypatch, [socket.timeout("read timed out")])
    with pytest.raises(socket.timeout):
        core._call("storyboard", {})
    assert len(opened) == 1


def test_operation_stdout_is_kept_rather_than_printed(monkeypatch, capsys, tmp_path):
    """Twelve agent-authored files wrap every harness call in
    `redirect_stdout` because a Blender script's prints landed in the middle
    of whatever the caller was printing."""
    core, _ = _stub_exchange(monkeypatch, [
        {"ok": True, "stdout": "bake: 120 frames", "result": {"ok": True}}])
    monkeypatch.setattr(core, "RUN_DIR", tmp_path)
    core._call("blender", {})
    assert capsys.readouterr().out == ""
    assert core.last_stdout == "bake: 120 frames"
    assert "bake: 120 frames" in (tmp_path / "harness_stdout.log").read_text()


def test_a_poll_that_times_out_keeps_waiting_instead_of_failing_the_call(monkeypatch):
    """A poll asks "is it done yet". If the daemon does not answer, the job is
    unaffected, so the wait should ask again rather than fail a call whose
    work is still running — and it must not spend the submitting call's
    budget on each attempt, which stalled a whole test run for ten minutes.
    """
    core, calls = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "queued"}},
        socket.timeout("daemon busy"),
        {"ok": True, "result": {"job_id": "abc", "status": "completed",
                                "result": {"ok": True}}},
    ])
    assert core._call("storyboard", {})["status"] == "completed"
    assert [m for m, _ in calls] == ["storyboard", "job_result", "job_result"]


def test_a_poll_uses_the_short_timeout_not_the_operation_budget(monkeypatch):
    core, _ = _stub_exchange(monkeypatch, [
        {"ok": True, "result": {"job_id": "abc", "status": "queued"}},
        {"ok": True, "result": {"job_id": "abc", "status": "completed"}},
    ])
    seen = []
    inner = core._exchange
    monkeypatch.setattr(core, "_exchange",
                        lambda m, a, t: (seen.append((m, t)), inner(m, a, t))[1])
    core._call("storyboard", {}, timeout=600.0)
    assert seen[0] == ("storyboard", 600.0), "the submit keeps its budget"
    assert seen[1][1] == core.POLL_TIMEOUT_S, "the poll does not"


def test_the_proxy_handshake_cannot_block_for_the_operation_budget():
    """A proxy that accepts the connection and never sends a status line used
    to hold the `recv` for `TIMEOUT_S`. Observed hanging a full suite there.
    """
    import inspect
    from dycraft.agent import _harness_core as core
    src = inspect.getsource(core._connect_via_proxy)
    handshake = src.split("s.connect(")[0]
    assert "s.settimeout(CONNECT_TIMEOUT_S)" in handshake, (
        "the CONNECT handshake must use the short timeout")
    assert "s.settimeout(timeout)" in src.split("return s")[0], (
        "and the tunnel must be restored to the data-phase timeout")


def test_claude_settings_take_away_the_sandbox_escape_hatch(tmp_path, monkeypatch):
    """With the escape hatch on, the model can rerun any command with
    `dangerouslyDisableSandbox: true`, and the `Bash(*)` allow rule approves it."""
    for flag in ("0", "1"):
        (tmp_path / flag).mkdir()
        sandbox, _ = _claude_sandbox_on(tmp_path / flag, monkeypatch, flag)
        assert sandbox["allowUnsandboxedCommands"] is False


def test_claude_agent_settings_confine_writes_with_one_edit_rule():
    s = claude_agent_settings("/runs/r1", deny=["Read(//x/**)"])
    allow = s["permissions"]["allow"]
    assert allow == ["Bash(*)", "Read", "Edit(///runs/r1/**)"]
    assert not any(r.startswith("Write(") for r in allow)  # Edit covers writes; Write draws a warning
    assert s["permissions"]["deny"] == ["Edit(///runs/r1/.claude/**)", "Edit(///runs/r1/**/.claude/**)",
                                        "Read(//x/**)"]
    assert s["sandbox"]["filesystem"]["allowWrite"] == ["/runs/r1"]
    assert s["sandbox"]["allowUnsandboxedCommands"] is False


def test_a_launch_works_in_src_and_writes_the_top_level(tmp_path, monkeypatch, capsys):
    """The agent's cwd, its sandbox root and `$DYCRAFT_RUN_DIR` are the run's `src/`; the top holds the README, which
    the launcher writes at launch and again when it exits (`run_layout.publish`)."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    captured = _drive_cli_main_to_launch([
        "--backend", "claude", "--layer", "object", "--model", "claude-opus-4-8",
        "--runs-dir", str(runs_dir), "--run-id", "stool", "--prompt", "A stool.",
    ], monkeypatch, tmp_path)
    top = runs_dir / "stool"
    assert captured["env"]["DYCRAFT_RUN_DIR"] == str((top / "src").resolve())
    assert (top / "src" / "meta.json").is_file()
    assert sorted(p.name for p in top.iterdir()) == ["README.md", "src"]
    readme = (top / "README.md").read_text()
    assert readme.startswith("# A stool\n") and "DyCraft object run `stool`" in readme
    assert f"[cli] run directory: {top.resolve()}/ (the agent works in src/)" in capsys.readouterr().err


def test_a_sub_run_is_launched_flat(tmp_path, monkeypatch):
    runs_dir = tmp_path / "parent" / "sub_runs"
    runs_dir.mkdir(parents=True)
    captured = _drive_cli_main_to_launch([
        "--backend", "claude", "--layer", "plain", "--model", "claude-opus-4-8",
        "--runs-dir", str(runs_dir), "--run-id", "scene_rewrite", "--parent-run", "p1", "--prompt", "Expand it.",
    ], monkeypatch, tmp_path)
    assert captured["env"]["DYCRAFT_RUN_DIR"] == str((runs_dir / "scene_rewrite").resolve())
    assert (runs_dir / "scene_rewrite" / "meta.json").is_file()
    assert not (runs_dir / "scene_rewrite" / "README.md").exists()


def test_a_brief_rewriter_filed_in_sub_runs_is_launched_flat(tmp_path, monkeypatch):
    """The scene and motion rewriters run before their parent has any record and name no parent, so the folder they
    are filed in is what says they are sub-runs; their parent reads `sub_runs/<name>/result.json`."""
    runs_dir = tmp_path / "parent" / "src" / "sub_runs"
    runs_dir.mkdir(parents=True)
    captured = _drive_cli_main_to_launch([
        "--backend", "claude", "--layer", "plain", "--model", "claude-opus-4-8",
        "--runs-dir", str(runs_dir), "--run-id", "scene_rewrite", "--prompt", "Expand it.",
    ], monkeypatch, tmp_path)
    assert captured["env"]["DYCRAFT_RUN_DIR"] == str((runs_dir / "scene_rewrite").resolve())
    held = {p.name for p in (runs_dir / "scene_rewrite").iterdir()}
    assert "meta.json" in held and not held & {"src", "README.md"}
