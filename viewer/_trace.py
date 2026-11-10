"""Reading a run's `trace.jsonl` fast enough for a first view.

A long run's trace is gigabytes, almost all of it results the viewer never reads: a probe or an edit can return
megabytes of data, and every one is a `job_result` line. Decoding every line would make a film's trajectory and
timeline take most of a minute each. `events` decodes only the lines a reader asks for. A short line is decoded as
it is; a long one is first classified by its `ev` field, read from its bytes, and skipped unread when no one wants it.
An `op_call`'s arguments, which come first in its line, are decoded only for the ops a reader names, and only the
fields it names are kept: a film's probe and compile arguments come to 1.4 GB.

The writer emits sorted keys with `json.dumps`'s default separators, so a line's `ev` field is `"ev": "<name>"`; a
line this cannot classify is decoded whole, so an unexpected layout costs time, never a record.

Even skipped, a long run's lines have to be read to find where they end, and the trajectory and the timeline, which a
browser asks for together, both want the trace (the timeline twice, through the trajectory). So `events` reads a file
once for every reader: each module `register`s what it will ask for when it is imported, a scan collects the records
all of them want, and later calls are served from those while the file is unchanged. A trace still being written is
read on from where the last scan stopped.
"""
from __future__ import annotations

import json
import mmap
import os
import re
import threading
from pathlib import Path
from typing import Iterator

try:                                    # optional: several times faster on the large lines a label still needs
    import orjson as _orjson
    _loads = _orjson.loads
except ImportError:                     # pragma: no cover
    _loads = json.loads

#: A line shorter than this is decoded outright: classifying it would cost about as much.
LONG = 16 * 1024
_EV_HEAD = re.compile(rb'"ev": ?"([A-Za-z_]+)"')


def _event_at(mm, start: int, end: int) -> tuple[str | None, int]:
    """The `ev` of the line `mm[start:end]` and where its key starts, or `(None, -1)`. Keys are sorted, so `ev` is near
    the start of most records and after the arguments of an `op_call`; the last `"ev":` in such a line is the record's
    own, since every key after it holds a short scalar."""
    head = mm[start:min(end, start + 512)]
    if not head.startswith(b'{"args":'):         # an op_call's arguments, first, can hold `ev` keys of their own
        match = _EV_HEAD.search(head)
        if match:
            return match.group(1).decode(), start + match.start()
    pos = max(mm.rfind(b'"ev": "', start, end), mm.rfind(b'"ev":"', start, end))
    if pos < 0:
        return None, -1
    match = _EV_HEAD.match(mm[pos:min(end, pos + 128)])
    return (match.group(1).decode(), pos) if match else (None, -1)


def _decode(raw: bytes):
    try:
        return _loads(raw)
    except ValueError:
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return None


#: The longest string an argument field keeps: enough for any label, and a bound on what a cached record holds.
ARG_CHARS = 200


def _project(args, keys) -> dict | None:
    """The fields `keys` of an op's arguments, each string cut to `ARG_CHARS`."""
    if not keys or not isinstance(args, dict):
        return None
    return {k: (v[:ARG_CHARS] if isinstance(v, str) else v) for k, v in args.items() if k in keys}


def _scan(mm, start: int, stop: int, wanted, args_keep) -> Iterator[dict]:
    """The records of `mm[start:stop]` whose `ev` is in `wanted`. An `op_call` keeps, as `args`, only the fields
    `args_keep` names for its op, and None for any other op."""
    while start < stop:
        end = mm.find(b"\n", start, stop)
        if end < 0:
            end = stop
        line_start, start = start, end + 1
        if end - line_start >= LONG:
            ev, pos = _event_at(mm, line_start, end)
            if ev is not None and ev not in wanted:
                continue
            if ev == "op_call" and pos > line_start:
                tail = _decode(b"{" + mm[pos:end].strip())
                if isinstance(tail, dict) and tail.get("op") not in args_keep:
                    yield {**tail, "args": None}
                    continue
        raw = mm[line_start:end].strip()
        if not raw:
            continue
        record = _decode(raw)
        if isinstance(record, dict) and record.get("ev") in wanted:
            if record.get("ev") == "op_call":
                record["args"] = _project(record.get("args"), args_keep.get(record.get("op")))
            yield record


#: Everything a registered reader asks for, which is what a scan collects: event names, and per op the argument
#: fields kept.
_WANTED: set[str] = set()
_ARGS: dict[str, frozenset] = {}
#: `path -> entry` for the files read most recently, and the lock that admits one scan of a path at a time.
_FILES: dict[str, dict] = {}
_KEEP = 4
_LOCK = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}


def _add_interest(wanted, args) -> None:
    _WANTED.update(wanted)
    for op, keys in (args or {}).items():
        _ARGS[op] = _ARGS.get(op, frozenset()) | frozenset(keys)


def register(wanted, args=None) -> None:
    """Declare the events, and per op the argument fields, a reader will ask `events` for. Called when the reader's
    module is imported, so that the first scan of a file already collects what every reader wants."""
    with _LOCK:
        _add_interest(wanted, args)


def _records(path: Path, wanted, args) -> list[dict]:
    """The collected records of `path`, scanning only what has not been scanned with this interest."""
    key = str(path)
    with _LOCK:
        _add_interest(wanted, args)
        interest = (frozenset(_WANTED), dict(_ARGS))
        lock = _PATH_LOCKS.setdefault(key, threading.Lock())
    with lock:
        try:
            fh = open(path, "rb")
        except OSError:
            _FILES.pop(key, None)
            return []
        with fh:
            st = os.fstat(fh.fileno())
            entry = _FILES.get(key)
            # The trace is only ever appended to. The same file at least as long as at the last scan is read on from
            # where that scan stopped; any other change, or a reader wanting more than was collected, starts over.
            if (entry is None or entry["interest"] != interest or entry["ino"] != st.st_ino
                    or st.st_size < entry["size"]):
                entry = {"interest": interest, "ino": st.st_ino, "size": 0, "offset": 0, "records": [],
                         "tail": []}
            if st.st_size != entry["size"] or entry["size"] == 0:
                if st.st_size == 0:
                    entry.update(size=0, offset=0, records=[], tail=[])
                else:
                    try:
                        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                    except (ValueError, OSError):
                        return []
                    with mm:
                        size = len(mm)
                        # Only whole lines are kept: a last line without its newline may still be being written, so
                        # its record is served this once and read again next time.
                        complete = mm.rfind(b"\n", entry["offset"], size) + 1
                        if complete > entry["offset"]:
                            entry["records"].extend(_scan(mm, entry["offset"], complete, *interest))
                            entry["offset"] = complete
                        entry["tail"] = list(_scan(mm, max(complete, entry["offset"]), size, *interest))
                        entry["size"] = size
            with _LOCK:
                _FILES.pop(key, None)
                _FILES[key] = entry
                while len(_FILES) > _KEEP:
                    _FILES.pop(next(iter(_FILES)))
            return entry["records"] + entry["tail"]


def events(path: Path, wanted, *, args=None) -> Iterator[dict]:
    """The records of `path` whose `ev` is in `wanted`, in file order.

    `args` maps an op to the fields of its arguments the caller reads; an `op_call` carries those fields as `args`
    (strings cut to `ARG_CHARS`), or None for an op it does not name. Malformed lines are skipped, as a reader of a
    trace still being written must. Records are shared between callers: read them, do not change them.
    """
    wanted = frozenset(wanted)
    args = {op: frozenset(keys) for op, keys in (args or {}).items()}
    for record in _records(Path(path), wanted, args):
        if record.get("ev") not in wanted:
            continue
        if record.get("ev") == "op_call":
            keys = args.get(record.get("op"))
            held = record.get("args")
            if not keys:
                if held is not None:
                    record = {**record, "args": None}
            elif held is not None and set(held) - keys:
                record = {**record, "args": {k: v for k, v in held.items() if k in keys}}
        yield record


__all__ = ["events", "register", "LONG", "ARG_CHARS"]
