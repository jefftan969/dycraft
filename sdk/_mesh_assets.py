"""Run-local, content-addressed geometry staging for the DyCraft object SDK.

``Object.add_part`` freezes a live build123d Shape as one ``solid`` part, and
every such part carries three files: the tessellated render mesh, the exact
B-rep, and the ``object.py`` source that built it. This module owns writing
them into the run, so the run is self-contained and relocatable, and names each
file by the digest of its own bytes, so identical content deduplicates and the
object's identity stays content-based.

This module is **trusted SDK code**: it runs during the builder's execution
(outside the AST sandbox, which gates only the agent's source), so its stdlib
file I/O is permitted. Stdlib only — no Open3D, no trimesh. Every payload
arrives as bytes already produced by the CAD kernel, so nothing here parses
geometry.

Ownership / lifecycle: the object-run tooling creates one ``MeshAssetSession``
per run (rooted at the run dir) and installs it as the process-active session
via ``set_active_mesh_session`` before executing ``object.py``. ``finalize()``
compiles the registered Shapes and consults ``get_active_mesh_session`` to
stage each part's assets. The session is cleared alongside the
one-Object-per-exec singleton reset.

Stored layout::

    <run_dir>/assets/meshes/<digest>.obj      # tessellated render mesh
    <run_dir>/assets/brep/<digest>.brep       # exact B-rep
    <run_dir>/assets/src/<digest>.py          # the authored build123d source
    <run_dir>/assets/textures/<digest>.png    # a material's base-color image

Every file is named by the digest of **its own** bytes, so nothing can collide
and re-staging identical content is a no-op. The IR stores each path **relative
to the run dir**; consumers resolve it against the run dir (renderer
``asset_base_dir``) or copy it next to the exported URDF
(``urdf_export.localize_object_meshes``).
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path


HASH_DIGEST_BYTES = 16   # blake2b-128 -> 32 hex chars, matching ir.content_hash
MESHES_SUBDIR = os.path.join("assets", "meshes")
BREP_SUBDIR = os.path.join("assets", "brep")
SRC_SUBDIR = os.path.join("assets", "src")
TEXTURES_SUBDIR = os.path.join("assets", "textures")


class MeshStagingError(Exception):
    """Raised when an asset payload is not usable bytes."""


class MeshAssetSession:
    """Per-run staging of a part's assets under ``<run_dir>/assets/``.

    ``run_dir`` is the base every stored path is made relative to, so the IR
    records run-relative filenames and a run directory stays movable.
    """

    def __init__(self, run_dir, source_text: str | None = None):
        self.run_dir = Path(run_dir)
        #: The exact text that is being / was executed to build this object.
        #:
        #: Set by `compile.build_ir_with_layer`, which is the one place that
        #: holds the bytes it is about to `exec`. `finalize` stages this as the
        #: object's `source` asset, so the staged provenance is the program that
        #: actually ran rather than a re-read of a file that may have moved on.
        #: None only outside a harness build (a test constructing a session
        #: directly), where the staging path falls back to reading the run dir.
        self.source_text = source_text
        #: Per-part diagnostics from this run's compiles, in registration order.
        #: What the compile *did* — vertices welded, a stage that raised, a mesh
        #: coarsened to stay under the triangle budget — as opposed to what the
        #: part *is*, which goes in the IR. The agent layer drains this after
        #: executing the source and folds it into the compile report, so a
        #: diagnostic about one run never becomes part of the object's identity.
        self.geometry_notes: list = []

    def add_geometry_note(self, note: dict) -> None:
        """Record one part's compile diagnostics.

        Called by the builder at staging time. Kept on the session rather than in
        the IR because the session is per-run and the IR is content-hashed: a
        note is about this compile, and the same geometry compiled twice must
        keep one identity even if one of the two runs coarsened its render mesh.
        """
        self.geometry_notes.append(dict(note))

    def _stage_bytes(self, data: bytes, subdir: str, ext: str, label: str):
        """Write `data` to ``<run_dir>/<subdir>/<digest><ext>``; return (rel, digest).

        Content-addressed and idempotent: the file is named by the digest of
        its own bytes, so re-staging identical content reuses the existing file
        and distinct content can never collide.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise MeshStagingError(f"{label} must be bytes, got {type(data).__name__}")
        data = bytes(data)
        if not data:
            raise MeshStagingError(f"{label} is empty")
        digest = hashlib.blake2b(data, digest_size=HASH_DIGEST_BYTES).hexdigest()
        out_dir = self.run_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{digest}{ext}"
        if not dst.exists():
            dst.write_bytes(data)
        rel = os.path.relpath(dst, self.run_dir).replace(os.sep, "/")
        return rel, digest

    def stage_texture(self, image_bytes, ext: str = ".png"):
        """Stage a material's base-color image; return ``(rel_path, digest)``.

        Content-addressed like everything else here, which is what makes the
        common case cheap: one library material bound to twelve parts writes
        one file and yields twelve identical `Texture` values (and therefore
        twelve `Material`s that compare equal).

        The bytes are staged VERBATIM. The library ships a pre-baked image for
        exactly this reason — resizing or re-encoding here would put the
        installed Pillow version into the digest, and therefore into the id of
        every object that used a texture.
        """
        return self._stage_bytes(image_bytes, TEXTURES_SUBDIR, ext,
                                 "material texture")

    def stage_source(self, source_text) -> str:
        """Stage the authored program once; return its run-relative path.

        **One per object, not one per part.** A single `object.py` builds every
        part, so "which program built this" is a fact about the object. It used
        to be stored in every part's params — N identical strings, and a
        `part_source(part_name)` probe whose docstring had to explain that it
        ignores its own argument — because the retired `author_solids` design
        had a per-solid record to hang it on. That design is gone; this is the
        shape the fact actually has.
        """
        data = (source_text.encode("utf-8")
                if isinstance(source_text, str) else source_text)
        rel, _digest = self._stage_bytes(data, SRC_SUBDIR, ".py",
                                        "authored source")
        return rel

    def stage_solid(self, *, obj_bytes, brep_bytes,
                    aabb_min, aabb_max, shape) -> dict:
        """Stage a build123d-authored solid's two per-part assets.

        Returns the ``solid`` params fragment
        ``{brep, mesh, digest, aabb_min, aabb_max, shape}``. The authored
        program is staged once per object by `stage_source`, not here.

        ``digest`` is the **B-rep** digest — the exact geometry is the part's
        identity, and it is what the collision adjudicator's cache keys off.
        The OBJ and the source get their own digests in their own filenames, so
        a tessellation-tolerance change renames the OBJ without colliding with
        the old one, and every asset stays independently content-addressed.

        ``aabb_min``/``aabb_max`` come from the CAD kernel, NOT from parsing the
        OBJ: tessellation inscribes curved surfaces, so an OBJ-derived box would
        be slightly *smaller* than the true solid and the broad phase could miss
        a real contact. The kernel's bound errs outward, which is the safe
        direction for a conservative prefilter.
        """
        mesh_rel, _mesh_digest = self._stage_bytes(
            obj_bytes, MESHES_SUBDIR, ".obj", "solid OBJ")
        brep_rel, brep_digest = self._stage_bytes(
            brep_bytes, BREP_SUBDIR, ".brep", "solid BREP")
        return {
            "brep": brep_rel,
            "mesh": mesh_rel,
            "digest": brep_digest,
            "aabb_min": [float(v) for v in aabb_min],
            "aabb_max": [float(v) for v in aabb_max],
            "shape": shape,
        }


# ---------------------------------------------------------------------------
# Process-active session (installed by the object-run tooling per exec)
# ---------------------------------------------------------------------------

_active_lock = threading.Lock()
_active_session: "MeshAssetSession | None" = None


def set_active_mesh_session(session: "MeshAssetSession | None") -> None:
    """Install (or clear, with ``None``) the process-active staging session."""
    global _active_session
    with _active_lock:
        _active_session = session


def get_active_mesh_session() -> "MeshAssetSession | None":
    """Return the process-active staging session, or ``None`` if unset."""
    with _active_lock:
        return _active_session


def clear_active_mesh_session() -> None:
    set_active_mesh_session(None)


def drain_geometry_notes() -> list:
    """Take this run's per-part compile diagnostics, emptying the buffer.

    The agent layer calls this once, straight after executing the authored
    source, so the notes reach the compile report without the builder needing a
    return channel out of ``finalize()``, which yields only the IR.

    Draining rather than reading: a second compile in the same run must report
    its own notes, not accumulate the previous attempt's. Returns ``[]`` when no
    session is installed, as in a test that compiles a box-only source.
    """
    session = get_active_mesh_session()
    if session is None:
        return []
    notes, session.geometry_notes = session.geometry_notes, []
    return notes



# ---------------------------------------------------------------------------
# Process-active solid compiler (installed per exec by the object-run tooling)
#
# `Object.add_part` takes a LIVE build123d Shape and has to measure it at
# registration and compile it to bytes at finalize. That work lives in
# `agent/cad/_solid_compile.py`, and the SDK must not import it
# directly: `sdk` never imports `agent` (checked: `_probes` reaching into
# `verify` is the only cross-package import in the whole SDK), and the compiler
# is only importable where build123d is installed.
#
# So it is injected, exactly like the session above. Two
# consequences worth stating, since both are load-bearing:
#
#   * it keeps the sdk -> agent layering intact;
#   * where build123d is absent — the main DyCraft env, most of the test suite
#     — `add_part` raises a structured `no_active_solid_compiler` error rather
#     than an ImportError from deep inside the SDK.
#
# The injected object must expose `measure_shape(shape, *, name, tolerance,
# angular_tolerance, planar_max_edge, texture_scale_m)` and
# `compile_entry(entry)`, plus a `SolidCompileError` with `.kind` / `.message`.
#
# `texture_scale_m` is the bound material's tile size, and it is the only thing
# about a part's SURFACE the compiler is told. It is there because texture
# coordinates are derived at staging — a CAD tessellation discards the B-rep
# face structure, so nothing downstream still knows a barrel is a cylinder —
# and the tile size is what turns a surface parameter into a tile count.
# ---------------------------------------------------------------------------

_compiler_lock = threading.Lock()
_active_compiler = None


def set_active_solid_compiler(compiler) -> None:
    """Install (or clear, with ``None``) the process-active solid compiler."""
    global _active_compiler
    with _compiler_lock:
        _active_compiler = compiler


def get_active_solid_compiler():
    """Return the process-active solid compiler, or ``None`` if unset."""
    with _compiler_lock:
        return _active_compiler


def clear_active_solid_compiler() -> None:
    set_active_solid_compiler(None)
