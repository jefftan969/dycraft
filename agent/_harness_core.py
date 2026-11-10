"""Transport core of the staged harness client, `$RUN_DIR/harness.py`.

`import harness as h; h.<op>(...)` forwards each call to the run's harness
daemon over the first transport that works: a named pipe at
`$RUN_DIR/.harness.fifo`, a TCP connect to `127.0.0.1:<port>` (published in
`$RUN_DIR/.harness.port`), then an HTTP `CONNECT` tunnel through the sandbox's
proxy (`$HTTP_PROXY`). No harness call may need a sandbox escalation: codex's
workspace-write sandbox refuses every socket syscall, so it uses the pipe;
Claude's sandbox has its own network namespace and reaches the daemon through
its proxy, which allows `localhost`. Stdlib only, so a run directory replays
without the dycraft tree on the path.

Wire format, both directions: a 4-byte big-endian length, then JSON.

    request:  {"method": "compile", "args": {...}}
    response: {"ok": true, "result": <handler return>, "stdout": "<captured>"}
              {"ok": false, "error": "<message>", "stdout": "<captured>"}
"""
from __future__ import annotations

import base64
import errno
import json
import os
import select
import socket
import struct
import sys
import time
import uuid
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PORT_PATH = RUN_DIR / ".harness.port"
#: Named-pipe transport, published by the daemon beside `.harness.port`: plain
#: file I/O in the run directory, so it needs no socket syscall.
FIFO_PATH = RUN_DIR / ".harness.fifo"
#: Per-call request payloads and reply pipes live here.
CALLS_DIR = RUN_DIR / ".harness.calls"
LOOPBACK_HOST = "127.0.0.1"
# The hostname string we put in the CONNECT request line when tunneling
# through Claude's HTTP proxy. The proxy enforces `allowedDomains` on
# the literal hostname; Claude Code's docs explicitly state IP literals
# (`127.0.0.1`, CIDR ranges) cannot be added to `allowedDomains`, so the
# only working entry for loopback is the string `"localhost"`. We
# therefore use that here even though `LOOPBACK_HOST` is `127.0.0.1`
# for direct connects — the proxy resolves `localhost` on the host side
# (where it points at the real daemon's 127.0.0.1) before tunneling.
PROXY_TARGET_HOST = "localhost"
TIMEOUT_S = 600.0
CONNECT_TIMEOUT_S = 5.0

#: Seconds a call waits for the job it started before returning it unfinished,
#: still collectable with `job_result`. Kept inside the agent's shell timeout,
#: so a wait is never cut off at the tool boundary: the launcher sets
#: `DYCRAFT_HARNESS_WAIT_S` for a backend whose shell default is shorter.
#: `batch=True` returns the job at once (`_batch_wait`).
WAIT_S = float(os.environ.get("DYCRAFT_HARNESS_WAIT_S") or 540.0)
#: How often that wait asks, and the read timeout of one ask.
POLL_S = 5.0
POLL_TIMEOUT_S = 30.0
#: Attempts for one exchange before giving up; the connection can drop.
ATTEMPTS = 3
RETRY_S = 2.0
#: Ops whose own result is a job record, so waiting on them would recurse.
#:
#: `contribution_status` and `cancel_contribution` belong here for the same
#: reason: they report on a *contribution* job, whose ids live in a different
#: store from the one `job_result` reads (see `_batch_wait`).
_JOB_METHODS = ("job_status", "job_result", "job_cancel",
                "contribution_status", "cancel_contribution")
#: Terminal job states, as the daemon spells them.
_TERMINAL = ("completed", "failed", "cancelled", "interrupted")
#: stdout the daemon captured for the last call, and whether to echo it. It is
#: also appended to `harness_stdout.log` beside the run.
ECHO_STDOUT = False
last_stdout = ""


def _jsonable(value):
    """A call argument JSON cannot spell, as one it can: ranges, sets and
    tuples become lists, and numpy values plain numbers or lists."""
    if isinstance(value, (range, set, frozenset)):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"harness arguments must be JSON values; got {type(value).__name__}")


def _readn(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise IOError("daemon closed connection mid-frame")
        buf += chunk
    return buf


def _read_port() -> int:
    if not PORT_PATH.exists():
        raise SystemExit(
            f"[harness] daemon port file missing at {PORT_PATH}; the launcher "
            f"should have started the daemon. See {RUN_DIR / '.daemon.log'}."
        )
    txt = PORT_PATH.read_text().strip()
    if not txt.isdigit():
        raise SystemExit(
            f"[harness] daemon port file {PORT_PATH} has non-numeric content: "
            f"{txt!r}. See {RUN_DIR / '.daemon.log'}."
        )
    return int(txt)


def _parse_proxy_url(url: str) -> tuple[str, int]:
    """Parse a proxy URL string into (host, port).

    Accepts `http://[user:pass@]host:port[/...]`, `host:port`, or bare
    `host` (defaults to port 80). Any `user:pass@` userinfo is stripped
    here so the host resolves — credentials are applied separately via
    `_proxy_auth_header` (the CONNECT request's `Proxy-Authorization`).
    Claude Code's in-sandbox proxy is `http://<user>:<token>@localhost:3128`;
    without stripping the userinfo, the host parses as `user:pass@localhost`
    and the connect fails to resolve. Schemes other than
    `http://` raise — we deliberately don't try to speak HTTPS to the
    proxy or SOCKS (`ALL_PROXY=socks5h://...` would need a separate
    handshake).
    """
    raw = url.strip()
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() != "http":
            raise ValueError(
                f"unsupported proxy scheme {scheme!r} in {url!r}; "
                f"only http:// is implemented (SOCKS handshake not "
                f"wired up — would need separate code path)"
            )
        raw = rest
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    if "@" in raw:                       # drop `user:pass@` userinfo
        raw = raw.rsplit("@", 1)[1]
    if ":" in raw:
        host, port_s = raw.rsplit(":", 1)
        return host, int(port_s)
    return raw, 80


def _proxy_auth_header(url: str) -> str | None:
    """Build the `Proxy-Authorization: Basic ...` header line for `url`.

    Returns None when the proxy URL carries no `user:pass@` userinfo.
    Claude Code's in-sandbox proxy requires Basic auth (`GIT_CONFIG_PARAMETERS`
    in the sandbox env sets `http.proxyAuthMethod=basic`); without this
    header the proxy answers the CONNECT with `407 Proxy Authentication
    Required`. The token is `base64(user:pass)` verbatim — Claude's
    credentials are plain (`srt:<hex>`), so we don't percent-decode.
    """
    raw = url.strip()
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    if "@" not in raw:
        return None
    userinfo = raw.rsplit("@", 1)[0]
    token = base64.b64encode(userinfo.encode("utf-8")).decode("ascii")
    return f"Proxy-Authorization: Basic {token}"


def _mask_proxy(url: str) -> str:
    """Redact `user:pass@` credentials from a proxy URL for logging."""
    import re
    return re.sub(r"://[^@/\s]+@", "://***:***@", url)


def _connect_via_proxy(port: int, proxy_url: str,
                       timeout: float = TIMEOUT_S) -> socket.socket:
    """Open a CONNECT tunnel through `proxy_url` to host
    `LOOPBACK_HOST:<port>` and return the upgraded socket.

    `timeout` is the read timeout applied to the upgraded tunnel (the
    data phase); long-blocking tools pass a larger value.

    **The CONNECT handshake itself uses `CONNECT_TIMEOUT_S`, not that:** a
    proxy that has not sent a status line in five seconds is not going to.

    Raises SystemExit on any handshake failure with the proxy's status
    line included in the message — useful when the proxy refuses
    because `localhost` isn't in `allowedDomains` (the Claude sandbox
    settings under `sandbox.network.allowedDomains` must include
    `"localhost"` for this to work).
    """
    proxy_host, proxy_port = _parse_proxy_url(proxy_url)
    auth = _proxy_auth_header(proxy_url)
    masked = _mask_proxy(proxy_url)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TIMEOUT_S)
    s.connect((proxy_host, proxy_port))
    # CONNECT target uses PROXY_TARGET_HOST ("localhost"), not
    # LOOPBACK_HOST ("127.0.0.1"). The proxy matches the request-line
    # hostname against `allowedDomains` as a literal string, and
    # `allowedDomains` cannot contain IP literals, so
    # `CONNECT 127.0.0.1:<port>` is refused with HTTP 403.
    #
    # Claude Code's in-sandbox proxy requires Basic auth; the
    # `Proxy-Authorization` header (when the URL carries `user:pass@`)
    # avoids a `407 Proxy Authentication Required` on the CONNECT.
    lines = [f"CONNECT {PROXY_TARGET_HOST}:{port} HTTP/1.1",
             f"Host: {PROXY_TARGET_HOST}:{port}"]
    if auth:
        lines.append(auth)
    req = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    s.sendall(req)
    hdr = b""
    while b"\r\n\r\n" not in hdr:
        chunk = s.recv(4096)
        if not chunk:
            s.close()
            raise SystemExit(
                f"[harness] proxy {masked!r} closed during CONNECT "
                f"handshake; received {hdr!r}"
            )
        hdr += chunk
    status_line = hdr.split(b"\r\n", 1)[0]
    if b" 200 " not in status_line:
        s.close()
        hint = (" — proxy wants auth; the $HTTP_PROXY URL has no "
                "`user:pass@` for the Proxy-Authorization header"
                if b" 407 " in status_line and not auth
                else f" — check that `network.allowedDomains` in the "
                     f"sandbox settings includes \"{PROXY_TARGET_HOST}\".")
        raise SystemExit(
            f"[harness] proxy {masked!r} refused CONNECT to "
            f"{PROXY_TARGET_HOST}:{port}: "
            f"{status_line.decode(errors='replace')!r}{hint}"
        )
    # The proxy MUST NOT send body bytes after the 200 reply (those would
    # belong to the tunneled protocol). If it did, our daemon's first
    # length-prefix would be misaligned and the call would hang on _readn.
    # Detecting this here is much friendlier than silently mis-framing.
    extra = hdr.split(b"\r\n\r\n", 1)[1]
    if extra:
        s.close()
        raise SystemExit(
            f"[harness] proxy {masked!r} sent {len(extra)} bytes after "
            f"the CONNECT headers; tunnel cannot be reliably framed. "
            f"Trailer={extra[:200]!r}"
        )
    # Handshake over; the data phase is the one the caller sized.
    s.settimeout(timeout)
    return s


def _open_daemon_socket(port: int,
                        timeout: float = TIMEOUT_S) -> socket.socket:
    """Return a socket connected to the daemon at `LOOPBACK_HOST:<port>`.

    `timeout` is the read timeout applied once connected (the data phase);
    the initial connect always uses the short `CONNECT_TIMEOUT_S`.

    Tries direct connect first. If that returns `ConnectionRefusedError`
    (the steady state inside Claude's bash sandbox, where the netns has
    its own empty loopback), fall back to HTTP CONNECT through the
    proxy advertised in `$HTTP_PROXY` (`$http_proxy` lowercased as
    fallback). If neither path works, raise SystemExit with a pointer
    explaining what changed in the environment.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TIMEOUT_S)
    try:
        s.connect((LOOPBACK_HOST, port))
    except ConnectionRefusedError:
        s.close()
    else:
        s.settimeout(timeout)
        return s

    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not proxy:
        raise SystemExit(
            f"[harness] cannot reach daemon at {LOOPBACK_HOST}:{port}: "
            f"direct connect refused and no $HTTP_PROXY set for the "
            f"CONNECT fallback. If you're inside Claude Code's bash "
            f"sandbox, the launcher should have routed $HTTP_PROXY to "
            f"the in-sandbox proxy on localhost:3128 — see "
            f"`{RUN_DIR / '.daemon.log'}`."
        )
    return _connect_via_proxy(port, proxy, timeout=timeout)


def _fifo_available() -> bool:
    """Whether the daemon published a FIFO to talk over."""
    try:
        return FIFO_PATH.is_fifo()
    except OSError:
        return False


def _read_reply_fifo(path: Path, timeout: float) -> bytes:
    """Read one length-prefixed frame from `path`, waiting up to `timeout`.

    Opened `O_NONBLOCK` so the open itself cannot wedge: a FIFO opened for
    reading blocks until a writer arrives, and the writer here is the daemon
    answering a call that may take an hour. Non-blocking turns that into a
    `select` we can bound, which is what lets a caller time out a slow job
    without leaving a file descriptor stuck in the kernel.

    Before any writer opens, `select` simply reports not-readable; once the
    daemon has written and closed, we read the frame and then EOF. A reply is
    never partial in practice -- the daemon writes it in one `write` after the
    handler returns -- but the length prefix is honoured properly anyway, since
    a reply larger than the pipe buffer is delivered in several chunks.
    """
    deadline = time.monotonic() + timeout
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        # Chunks are collected and joined once, so a large reply is read in
        # linear time rather than recopied on every read.
        chunks, received, want = [], 0, None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout("timed out waiting for the harness reply")
            if not select.select([fd], [], [], min(remaining, 1.0))[0]:
                continue
            try:
                chunk = os.read(fd, 1 << 20)
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                raise
            if not chunk:
                if want is not None and received >= want + 4:
                    break
                raise IOError("daemon closed the reply pipe mid-frame")
            chunks.append(chunk)
            received += len(chunk)
            if want is None and received >= 4:
                chunks = [b"".join(chunks)]
                (want,) = struct.unpack(">I", chunks[0][:4])
            if want is not None and received >= want + 4:
                break
        return b"".join(chunks)[4:4 + want]
    finally:
        os.close(fd)


def _exchange_fifo(method: str, args: dict, timeout: float):
    """One request/response over the named-pipe transport.

    The payload goes in a regular file and only its id goes down the shared
    pipe. That split is the whole design: several callers write to one FIFO,
    and a pipe write is atomic only up to `PIPE_BUF` (4096 bytes), while a
    `blender()` call carrying a page of author Python is routinely larger. An
    id is 33 bytes, so the notification can never interleave; the payload is
    written to a temporary file and renamed, so the daemon cannot observe it
    half-written either.

    The reply is a per-call FIFO rather than one shared return pipe, because
    the daemon answers on a 32-thread pool and a single return pipe would
    serialize it -- the concurrency is the thing `batch=` exists to buy.
    """
    CALLS_DIR.mkdir(exist_ok=True)
    ident = uuid.uuid4().hex
    request = CALLS_DIR / f"{ident}.req"
    tmp = CALLS_DIR / f"{ident}.req.tmp"
    reply = CALLS_DIR / f"{ident}.rsp"
    try:
        tmp.write_bytes(json.dumps({"method": method, "args": args}, default=_jsonable).encode("utf-8"))
        os.replace(tmp, request)
        os.mkfifo(reply, 0o600)
        # Open for write last, and only to announce: the daemon holds a writer
        # on its own FIFO for its whole life, so this open never blocks waiting
        # for a reader to appear.
        fd = os.open(FIFO_PATH, os.O_WRONLY)
        try:
            os.write(fd, (ident + "\n").encode("ascii"))
        finally:
            os.close(fd)
        return json.loads(_read_reply_fifo(reply, timeout).decode("utf-8"))
    finally:
        for path in (tmp, request, reply):
            try:
                path.unlink()
            except OSError:
                pass


def _exchange(method: str, args: dict, timeout: float):
    """One request/response, retried over a dropped socket."""
    last = None
    for attempt in range(ATTEMPTS):
        # Rung one: the FIFO, when the daemon published one. It is tried ahead
        # of the socket because on the surface that has it, the socket costs a
        # sandbox escalation and the FIFO costs nothing.
        if _fifo_available():
            try:
                return _exchange_fifo(method, args, timeout)
            except socket.timeout:
                raise
            except (OSError, ValueError, struct.error) as error:
                if attempt == ATTEMPTS - 1:
                    raise
                last = error
                time.sleep(RETRY_S)
                continue
        port = _read_port()
        try:
            s = _open_daemon_socket(port, timeout=timeout)
            try:
                payload = json.dumps({"method": method, "args": args}, default=_jsonable).encode("utf-8")
                s.sendall(struct.pack(">I", len(payload)) + payload)
                (n,) = struct.unpack(">I", _readn(s, 4))
                return json.loads(_readn(s, n).decode("utf-8"))
            finally:
                s.close()
        except (OSError, ConnectionError, struct.error, ValueError) as error:
            # A read timeout means the daemon is working, not broken: retrying
            # would start a second copy of the same request.
            if isinstance(error, socket.timeout) or attempt == ATTEMPTS - 1:
                raise
            last = error
            time.sleep(RETRY_S)
    raise last


def _record_stdout(text: str) -> None:
    global last_stdout
    last_stdout = text
    if not text:
        return
    try:
        with (RUN_DIR / "harness_stdout.log").open("a") as log:
            log.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass
    if ECHO_STDOUT:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()


def _unwrap(resp):
    if not resp.get("ok"):
        sys.stderr.write(f"[harness] {resp.get('error', 'call failed')}\n")
        tb = resp.get("traceback")
        if tb:
            sys.stderr.write(tb)
            if not tb.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.flush()
        raise SystemExit(1)
    return resp.get("result")


def _batch_wait(batch) -> float | None:
    """The `wait` a `batch=` argument implies: `0` for truthy, `WAIT_S` otherwise.

    `None` is the op's own default, then wait for it; `True` returns the job at
    once; `False` runs it here.
    """
    return 0.0 if batch else None


def _call(method: str, args: dict, *, timeout: float = TIMEOUT_S, wait: float | None = None):
    resp = _exchange(method, args, timeout)
    _record_stdout(resp.get("stdout") or "")
    result = _unwrap(resp)
    # Wait on the job this call started, so the caller gets an outcome rather
    # than an id and a polling loop to write. A job that outlives the wait
    # comes back as its pending record.
    budget = WAIT_S if wait is None else wait
    if (budget > 0 and method not in _JOB_METHODS
            and isinstance(result, dict) and result.get("job_id")
            and result.get("status") not in _TERMINAL):
        return _await(result["job_id"], budget, timeout)
    return result


def _await(job_id: str, budget: float, timeout: float):
    deadline = time.monotonic() + budget
    record = {"job_id": job_id, "status": "submitted"}
    while True:
        try:
            resp = _exchange("job_result", {"job_id": job_id},
                             min(POLL_TIMEOUT_S, timeout))
        except socket.timeout:
            # The daemon did not answer this one. The job is unaffected, so
            # try again while there is budget rather than failing a call
            # whose work is still running.
            if time.monotonic() >= deadline:
                return record
            continue
        _record_stdout(resp.get("stdout") or "")
        record = _unwrap(resp)
        if not isinstance(record, dict) or record.get("status") in _TERMINAL:
            return record
        if time.monotonic() >= deadline:
            return record
        time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# Operations: `import harness as h; h.<fn>()`. Each is a thin `_call` wrapper.
# ---------------------------------------------------------------------------
