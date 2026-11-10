"""Shared tool-I/O primitives.

  * `round_floats`               — deep float-rounding (1 micron by default)
  * `atomic_write_text`          — temp-file + os.replace, force 0o644
  * `confine_to_run_dir`         — resolve an agent-supplied path inside the run
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# JSON / Text-block helpers
# ---------------------------------------------------------------------------


def round_floats(value, ndigits: int = 6):
    """Deep copy of `value` with floats rounded to `ndigits` (default: 1 micron).

    FP noise from verifier math (bbox sums, overlap depths) otherwise bloats
    on-disk reports and the bytes the agent sees with values like
    `0.37999999...`.

    **Six, not three.** A 1 mm display (three digits) is too coarse for
    quantities the agent reads and then authors against:

      * **Quaternions would come back denormalised.** `pose_at` and
        `part_se3_at` return `(qw,qx,qy,qz)` and the prompt tells the author
        to "read one, write one"; at three decimals a unit quaternion's norm is
        off by ~5e-4, against a stored `QUAT_QUANTUM` of 1e-9.
      * **A gap could not resolve its own tolerance.** `touch`'s default is
        5 mm, and 0.0049 / 0.0050 / 0.0051 would all display as `0.005`.
      * **Ratios near one would collapse.** A deformation's `volume_ratio` of
        0.9995 or 1.0003 would read as exactly `1.0`, and "is this squash
        volume-preserving" lives entirely inside that band.
      * **Micron-scale depths would read as zero.** A deformed-mesh
        penetration of 1.4e-6 m would display as `depth_m: 0.0` — a failure
        verdict whose evidence says the overlap is nothing, which leaves an
        agent guessing at the contact tolerance.

    Six decimals of a meter is exactly 1 µm, the quantum the IR stores
    positions at, so the display agrees with the storage about what is
    representable. FP noise is at the 1e-16 level, so `0.37999999999999995`
    and `0.1 + 0.2` still round to `0.38` and `0.3`.

    `exact_volume_cm3` is in cm^3 rather than m^3: an interference volume
    spans 1e-6..1e-2 m^3, a dynamic range no fixed number of
    decimals covers, and cm^3 is also simply the readable unit for it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return tuple(round_floats(v, ndigits) for v in value)
    return value


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, data: str) -> int:
    """Write `data` to `path` via temp-file + os.replace. Forces 0o644 so
    `runs/` artifacts are world-readable regardless of umask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(data.encode("utf-8"))


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------


def confine_to_run_dir(run_dir: Path, path: str) -> Path | None:
    """Resolve an agent-supplied relative `path` inside `run_dir`, or None.

    The single gate on "an agent named a path and we are about to touch it".
    Returns None — never raises, never a partial answer — for a non-string, an
    empty string, an absolute path, any path containing `..`, and anything that
    resolves outside the run dir.

    The `.resolve()`-then-`relative_to` ordering is the part that matters: the
    textual `..` check alone misses a symlink inside the run dir pointing out of
    it, and resolving first is what collapses that case into a plain prefix
    comparison.

    Confinement only. Whether the resolved path is *allowed* — on a read
    whitelist, or an acceptable output directory — is the caller's policy.
    """
    if not isinstance(path, str) or not path:
        return None
    if path.startswith("/") or ".." in Path(path).parts:
        return None
    abs_path = (run_dir / path).resolve()
    try:
        abs_path.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return abs_path


__all__ = [
    "round_floats",
    "atomic_write_text",
    "confine_to_run_dir",
]
