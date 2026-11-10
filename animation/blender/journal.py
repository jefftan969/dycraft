"""Append-only record of what every harness operation did to the project.

`trace.jsonl` records *that* an operation ran, with its full arguments. This
answers the two questions a diagnostic agent asks afterwards:

* **"What source ran?"** For an agent that keeps its logic in helper files, the
  recorded `code` is a pointer, not a program: an `h.blender` call is often a
  line of `exec(open('animation_src/...').read())` over hundreds of helper
  files the harness does not otherwise version. Source reaches
  `animation_src/` through the agent's own shell, never through a harness
  operation, so there is no write to hook — the call boundary is the only place
  the tree can be captured, and it is also the only state replay needs.
* **"What changed in the project?"** A successful edit renames its workspace
  into place, which destroys the pre-state.

Both cost about 17 MB and 21 s over a 33-hour run.

Layout under `<run>/native/journal/`::

    journal.jsonl            one record per operation, in submission order
    blobs/<ab>/<sha>         content-addressed source files and code strings
    manifests/<hash>.json    the source tree as of one operation
    digests/<seq>.json.gz    `scene_digest` output, at every version boundary

Nothing here is inside `native/work/`, so none of it enters `version`,
`publish`, or a delivered bundle.

**Ordering.** The resident Blender keeps module state between calls, so what a
call did can depend on the calls before it; `seq` records their order.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path

#: Source files whose *content* is worth keeping. Everything else in a source
#: root is noted by size and mtime, from the stat, without ever being read. One
#: run kept 42 MB of `.json.gz` motion caches beside 1.9 MB of Python and
#: another kept 497 MB of harness-result dumps beside 173 kB of it; reading
#: those to hash them would have cost 519 MB per operation and 156 GB across
#: that run, while size and mtime still let a replay say "the cache differed"
#: instead of diverging in silence.
TEXT_SUFFIXES = frozenset({".py", ".json", ".txt", ".md", ".sh", ".csv",
                           ".yaml", ".yml", ".cfg", ".ini", ".toml"})

#: Above this, a "text" file is treated as data and noted rather than stored.
#: One run's harness-result dumps were 7 MB of `.json` each.
MAX_BLOB_BYTES = 256 * 1024


def enabled():
    """`DYCRAFT_ANIMATION_JOURNAL=0` turns the whole thing off."""
    return (os.environ.get("DYCRAFT_ANIMATION_JOURNAL") or "1") not in ("0", "false", "no")


#: An argument longer than this is stored as a blob and referenced. One run's
#: median `h.blender` call was 829 bytes and another's was 50 kB; inlining both
#: would have made the journal 7 MB of mostly-duplicate code.
MAX_INLINE_ARG = 1024


def describe(args, journal):
    """The part of an operation's arguments worth keeping in the record.

    `trace.jsonl` already carries every argument inline, so this exists to make
    the journal self-contained rather than to duplicate it: a reader that has
    only `native/journal/` can still recover the exact code a call ran.
    """
    out = {}
    for key, value in (args or {}).items():
        if key.startswith("_") or value is None:
            continue
        if isinstance(value, str):
            payload, inline = value, value
        else:
            try:
                payload = json.dumps(value, default=str, sort_keys=True)
            except (TypeError, ValueError):
                payload = repr(value)
            inline = value
        if len(payload) > MAX_INLINE_ARG:
            out[key] = {"blob": journal.store_blob(payload), "bytes": len(payload)}
        else:
            out[key] = inline
    return out


class Journal:
    """Thread-safe. Non-mutating operations run concurrently with each other."""

    def __init__(self, root, *, on=None):
        self.root = Path(root)
        self.on = enabled() if on is None else on
        self.lock = threading.Lock()
        self.path = self.root / "journal.jsonl"
        self.blobs = self.root / "blobs"
        self.manifests = self.root / "manifests"
        self.digests = self.root / "digests"
        self._seq = 0
        if not self.on:
            return
        for directory in (self.root, self.blobs, self.manifests, self.digests):
            directory.mkdir(parents=True, exist_ok=True)
        # A restarted daemon continues the run's numbering rather than
        # overwriting it; `Jobs` already reloads its records the same way.
        if self.path.is_file():
            with self.path.open(errors="replace") as handle:
                for line in handle:
                    try:
                        self._seq = max(self._seq, json.loads(line)["seq"])
                    except (ValueError, KeyError):
                        continue

    # -- writing ---------------------------------------------------------

    def next_seq(self):
        if not self.on:
            return None
        with self.lock:
            self._seq += 1
            return self._seq

    def store_blob(self, payload):
        """Content-address one blob. Returns its hash, or None when disabled."""
        if not self.on:
            return None
        if isinstance(payload, str):
            payload = payload.encode("utf8", "surrogateescape")
        digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
        target = self.blobs / digest[:2] / digest
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: two concurrent ops may store the same blob.
            temporary = target.with_suffix(".part-%d" % threading.get_ident())
            temporary.write_bytes(payload)
            temporary.replace(target)
        return digest

    def capture_sources(self, roots):
        """Store the authored source tree as it stands right now.

        Returns the manifest hash, which is also the identity of that tree —
        two operations sharing it ran against identical source.
        """
        if not self.on:
            return None
        files = {}
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            label = root.name
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    # The edit path already refuses these; recording one would
                    # mean following it out of the run directory.
                    continue
                if not path.is_file():
                    continue
                relative = f"{label}/{path.relative_to(root).as_posix()}"
                try:
                    stat = path.stat()
                except OSError:
                    continue
                # Decided from the stat, before any read. Hashing the content of
                # everything would mean reading the whole source root per
                # operation, hundreds of MB of harness-result dumps nobody
                # replays.
                # Bulk files are identified by size and mtime instead, which is
                # enough for a replay to say "the cache differed".
                if path.suffix in TEXT_SUFFIXES and stat.st_size <= MAX_BLOB_BYTES:
                    try:
                        payload = path.read_bytes()
                    except OSError:
                        continue
                    files[relative] = {"bytes": len(payload), "stored": True,
                                       "sha": self.store_blob(payload)}
                else:
                    files[relative] = {"bytes": stat.st_size, "stored": False,
                                       "mtime_ns": stat.st_mtime_ns}
        body = json.dumps({"files": files}, sort_keys=True).encode()
        digest = hashlib.blake2b(body, digest_size=16).hexdigest()
        target = self.manifests / f"{digest}.json"
        if not target.exists():
            temporary = target.with_suffix(".part-%d" % threading.get_ident())
            temporary.write_bytes(body)
            temporary.replace(target)
        return digest

    def store_digest(self, seq, directory):
        """Move a worker's `digest.json` into the journal, gzipped.

        Searched rather than named: which output subdirectory holds it depends
        on the operation, and `Jobs.status` already locates `progress.json` the
        same way. Returns the digest's own content hash, so a reader can tell an
        operation that changed the scene from one that only republished it.

        The newest wins, because `build` writes two: one for the donor it
        imports and one for the workspace after the merge. The merge runs
        second and is the state the operation would publish.
        """
        if not self.on or seq is None:
            return None
        try:
            found = sorted(Path(directory).rglob("digest.json"),
                           key=lambda p: p.stat().st_mtime)
        except OSError:
            return None
        if not found:
            return None
        try:
            payload = found[-1].read_bytes()
        except OSError:
            return None
        target = self.digests / f"{seq:06d}.json.gz"
        try:
            target.write_bytes(gzip.compress(payload, 6))
        except OSError:
            return None
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def record(self, **fields):
        """Append one operation record. Never raises into the caller."""
        if not self.on:
            return
        fields.setdefault("ts", time.time())
        line = json.dumps(fields, default=str, sort_keys=True) + "\n"
        try:
            with self.lock, self.path.open("a") as handle:
                handle.write(line)
        except OSError:
            pass

    # -- reading ---------------------------------------------------------

    def records(self):
        """Every operation, in sequence order."""
        if not self.path.is_file():
            return []
        out = []
        with self.path.open(errors="replace") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return sorted(out, key=lambda r: r.get("seq") or 0)

    def resolve_arg(self, value):
        """One recorded argument, whether it was inlined or stored as a blob.

        `describe` inlines what is short enough to read in the journal itself
        and offloads the rest, so a reader that wants the value should not have
        to know which happened.
        """
        if isinstance(value, dict) and set(value) == {"blob", "bytes"}:
            payload = self.read_blob(value["blob"])
            return None if payload is None else payload.decode("utf8", "replace")
        return value

    def read_blob(self, sha):
        path = self.blobs / sha[:2] / sha
        return path.read_bytes() if path.is_file() else None

    def read_manifest(self, digest):
        path = self.manifests / f"{digest}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def read_digest(self, seq):
        path = self.digests / f"{seq:06d}.json.gz"
        if not path.is_file():
            return None
        return json.loads(gzip.decompress(path.read_bytes()))

    def timeline(self):
        """The run as a list of "operation N changed this much", in order.

        The join an inspection agent would otherwise write every time: each
        record, carrying what the project looked like before it and what it
        changed. `changed` is `None` for an operation that left the scene
        untouched, which is the distinction that makes bisecting a change
        cheap: many operations change no curve at all.

        Operations that publish without touching the blend (`plan`,
        `accept_storyboard`) write no digest, so the comparison walks back to
        the last one that did rather than reporting a phantom change.
        """
        from .scene_digest import diff
        out, previous = [], None
        for record in self.records():
            row = {k: record.get(k) for k in
                   ("seq", "step", "operation", "ok", "published", "job_id",
                    "duration_s", "src_manifest", "error")}
            current = self.read_digest(record["seq"]) if record.get("digest") else None
            row["changed"] = row["detail"] = None
            if current is not None:
                changed = diff(previous, current) if previous is not None else None
                if changed is not None:
                    row["changed"] = {field: len(value["added"]) + len(value["removed"])
                                                + len(value["modified"])
                                      for field, value in changed.items()
                                      if isinstance(value, dict) and "added" in value}
                    row["detail"] = changed
                previous = current
            out.append(row)
        return out

    def latest_digest(self, before=None):
        """The scene as of the most recent operation that changed it.

        Operations that publish without touching the blend — `plan`,
        `accept_storyboard` — write no digest, so the timeline is continuous
        only if a reader carries the last one forward. This does that.
        """
        candidates = sorted(self.digests.glob("*.json.gz")) if self.digests.is_dir() else []
        for path in reversed(candidates):
            seq = int(path.name.split(".")[0])
            if before is not None and seq >= before:
                continue
            return seq, json.loads(gzip.decompress(path.read_bytes()))
        return None, None
