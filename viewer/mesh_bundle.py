"""Serve a run's meshes as a few tar chunks instead of one request each.

A scene is one mesh file per visual, and a large scene names thousands. Fetched one by one that is ~0.5 ms of
per-request overhead each on the server and an XHR each in the browser: seconds of overhead for 5,000 meshes, and
enough queued requests that Chrome starts failing them. The bytes are read off disk in one pass anyway, so this
packs them into tar chunks, a few requests for thousands of meshes.

* **Chunks, not one archive.** One bundle can exceed 150 MB, past what a browser keeps in its HTTP cache, while
  the per-file route is cacheable; at `BUNDLE_TARGET_BYTES` each chunk is an ordinary cache entry, and a handful
  fetch in parallel.
* **Addressed by content, not by position.** Each entry carries its run-relative path and goes into the cache the
  per-file loader reads, so the bundle is a *prefetch*: if it 404s, if the packing shifts between the manifest and
  the chunk requests (a live run recompiling), or if an entry is missing, the client fetches that mesh on its own.
* **USTAR, and only the text meshes.** The client's tar reader is thirty lines of header arithmetic, and GNU long
  names would double it, so a path that does not fit USTAR's 100-byte name field takes the per-file route. Baked
  `.glb` assets are left out: they live in a cache outside the run, are compact, and number one per object at most.
"""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

from . import runs as runs_mod

#: Bytes per chunk, before tar's 512-byte-per-entry padding.
#:
#: Big enough that the request count collapses (169 MB becomes 11 requests) and
#: small enough to stay an ordinary browser cache entry, so that re-selecting a
#: run is still served from cache rather than re-downloaded.
BUNDLE_TARGET_BYTES = 16 * 1024 * 1024

#: USTAR's name field is 100 bytes including the terminator.
_MAX_NAME_BYTES = 99

#: What may travel in a bundle. OBJ is the staged mesh format and the only one
#: the client can parse out of an archive without also resolving side files.
_BUNDLED_SUFFIXES = (".obj",)

_MESH_FILENAME_RE = re.compile(r'filename="([^"]+)"')


def referenced_meshes(urdf: str) -> list[str]:
    """Every mesh path the document names, de-duplicated and sorted.

    Taken from the generated URDF rather than walked out of the IR so that the
    bundle contains exactly what this variant's document asks for — including
    its collision proxies, and excluding anything the exporter chose not to
    reference. `app.py` caches that document, so this costs a regex.
    """
    return sorted(set(_MESH_FILENAME_RE.findall(urdf)))


def bundle_plan(run_dir: Path, urdf: str) -> list[list[tuple[str, Path, int]]]:
    """Pack the referenced meshes into chunks of `(name, path, size)`.

    Deterministic — sorted input, greedy fill — because the manifest and the
    chunk requests are separate calls that have to agree. A file larger than
    the target becomes a chunk of its own rather than being skipped.

    Every path goes through `runs_mod.resolve_file`, so a `filename` that tries
    to escape the run directory is dropped here rather than opened.
    """
    chunks: list[list[tuple[str, Path, int]]] = []
    current: list[tuple[str, Path, int]] = []
    current_bytes = 0

    for name in referenced_meshes(urdf):
        if not name.lower().endswith(_BUNDLED_SUFFIXES):
            continue
        if len(name.encode()) > _MAX_NAME_BYTES:
            continue
        target = runs_mod.resolve_file(run_dir, name)
        if target is None:
            continue
        try:
            size = target.stat().st_size
        except OSError:
            continue
        if current and current_bytes + size > BUNDLE_TARGET_BYTES:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append((name, target, size))
        current_bytes += size

    if current:
        chunks.append(current)
    return chunks


def manifest(chunks: list[list[tuple[str, Path, int]]]) -> dict:
    """What the client needs to know before asking for anything.

    Only counts and sizes: the entries name themselves inside each tar, so the
    client does not have to be told which mesh is in which chunk — and cannot
    be told wrong.

    Takes the plan rather than building one, because the manifest request and
    every chunk request need the same plan and building it means a `stat` per
    mesh — 222 ms across the 5403-mesh scene's meshes. `app.py` caches it
    beside the URDF it is derived from.
    """
    return {
        "chunk_bytes": BUNDLE_TARGET_BYTES,
        "files": sum(len(chunk) for chunk in chunks),
        "bytes": sum(size for chunk in chunks for _, _, size in chunk),
        "chunks": [
            {"index": index, "files": len(chunk),
             "bytes": sum(size for _, _, size in chunk)}
            for index, chunk in enumerate(chunks)
        ],
    }


def chunk_tar(chunks: list[list[tuple[str, Path, int]]],
              index: int) -> bytes | None:
    """One chunk as a USTAR archive, or None if there is no such chunk.

    Built in memory: a chunk is bounded by `BUNDLE_TARGET_BYTES`, and streaming
    it would buy nothing on top of that while making the size header — which
    tar writes *before* each entry — depend on a file that could change under
    it. Entry metadata is zeroed so that identical content produces identical
    bytes.
    """
    if index < 0 or index >= len(chunks):
        return None

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w",
                      format=tarfile.USTAR_FORMAT) as tar:
        for name, path, _ in chunks[index]:
            try:
                data = path.read_bytes()
            except OSError:
                # Vanished mid-bundle. Leaving it out is the whole point of
                # addressing entries by name: the client fetches it per file.
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()
