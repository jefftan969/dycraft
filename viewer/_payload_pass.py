"""A run's trajectory and agent timeline, built in their own interpreter and served as stored JSON.

usage: python -m <package>.viewer._payload_pass <trajectory|agent_timeline> <run_dir> <out.json>

Building a long run's trajectory parses tens of megabytes of transcript in Python, which takes seconds, and the
response is tens of megabytes of JSON to encode. Inside the server both held the interpreter lock against everything
else it serves, a film's meshes included, and playback ran dry. So the server runs the build here, keeps the JSON in
the viewer's asset cache keyed on the files it was read from and on this code, and serves those bytes as they are.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

KINDS = ("trajectory", "agent_timeline")

#: A long run's transcript takes seconds to parse; this bounds a build that hangs.
TIMEOUT_S = 900

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_CODE_DIGEST: list[str] = []
#: Payloads whose build ran out of time, remembered for the server's life: building one in the server instead would
#: hold its interpreter lock for as long again, and asking the child again would wait as long again.
_TIMED_OUT: set[str] = set()


class PayloadTimeout(RuntimeError):
    """The build ran out of time; the route answers 503 with this message."""


def _stamp(path: Path | None) -> list | None:
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return [str(path), None]
    return [str(path), st.st_size, st.st_mtime_ns]


def _code_digest() -> str:
    """The builders' source, so a changed builder never serves an old payload."""
    if not _CODE_DIGEST:
        here = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for name in ("trajectory.py", "agent_timeline.py", "_payload_pass.py"):
            h.update((here / name).read_bytes())
        _CODE_DIGEST.append(h.hexdigest()[:16])
    return _CODE_DIGEST[0]


def _finished(run_dir: Path) -> bool:
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and bool(meta.get("finished_at") or meta.get("outcome"))


def _encode(payload) -> bytes:
    """The JSON a FastAPI route returning `payload` sends."""
    from fastapi.encoders import jsonable_encoder

    return json.dumps(jsonable_encoder(payload), ensure_ascii=False, allow_nan=False, indent=None,
                      separators=(",", ":")).encode("utf-8")


def _build_here(kind: str, run_dir: Path) -> bytes:
    if kind == "trajectory":
        from .trajectory import build
    else:
        from .agent_timeline import build
    return _encode(build(run_dir))


def _build_apart(kind: str, run_dir: Path) -> bytes:
    """`_build_here` in a fresh interpreter, which the server waits on without holding the interpreter lock."""
    import tempfile

    from .. import _oom

    package = __name__.split(".")[0]
    # The folder this package was imported from, unresolved, so a package reached through a symlink keeps its name.
    root = str(Path(sys.modules[package].__file__).parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
    with tempfile.TemporaryDirectory(prefix="dycraft-payload-") as tmp:
        out = Path(tmp) / f"{kind}.json"
        proc = subprocess.run(
            _oom.tiered([sys.executable, "-m", f"{package}.viewer._payload_pass", kind, str(run_dir), str(out)],
                        _oom.WORKER),
            capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(f"building the {kind} failed:\n  " + "\n  ".join(tail))
        return out.read_bytes()


def cached_payload(kind: str, run_dir: Path) -> bytes:
    """The run's `kind` as the JSON its route sends, built apart and cached.

    Cached by the files the build reads (the transcript, `trace.jsonl` and `meta.json`) and the builders' code. The
    timeline of a run still going is built every time instead: what it calls running depends on the clock. One build
    per payload at a time; a second request for it waits for the first. When the interpreter cannot be started or the
    build fails there, it runs here as before; when it runs out of time, `PayloadTimeout`, and this server does not
    try that payload again.
    """
    from ..agent.transcript_capture import resolve_transcript
    from ..render.viewer_bake import cache_root

    if kind not in KINDS:
        raise ValueError(f"unknown payload {kind!r}")
    run_dir = Path(run_dir)
    sources = [_stamp(resolve_transcript(run_dir)), _stamp(run_dir / "trace.jsonl"), _stamp(run_dir / "meta.json")]
    key = json.dumps({"kind": kind, "run": str(run_dir.resolve()), "sources": sources, "code": _code_digest()})
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    cacheable = kind == "trajectory" or _finished(run_dir)
    path = cache_root() / "payloads" / f"{digest}.json"
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(digest, threading.Lock())
    with lock:
        if digest in _TIMED_OUT:
            raise PayloadTimeout(f"building the {kind} took longer than {TIMEOUT_S} s; it is not tried again until "
                                 "the viewer restarts")
        if cacheable:
            try:
                return path.read_bytes()
            except OSError:
                pass
        try:
            data = _build_apart(kind, run_dir)
        except subprocess.TimeoutExpired as exc:
            _TIMED_OUT.add(digest)
            raise PayloadTimeout(f"building the {kind} took longer than {TIMEOUT_S} s") from exc
        except (OSError, RuntimeError) as exc:
            print(f"[viewer] {kind} built in the server: {exc}", file=sys.stderr, flush=True)
            data = _build_here(kind, run_dir)
        if cacheable:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            except OSError:
                pass
        return data


def main(argv: list[str]) -> int:
    kind, run_dir, out = argv[0], Path(argv[1]), Path(argv[2])
    if kind not in KINDS:
        print(f"unknown payload {kind!r}", file=sys.stderr)
        return 2
    out.write_bytes(_build_here(kind, run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
