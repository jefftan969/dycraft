"""The frozen build123d pin, and the structured error every CAD child raises.

The stdlib-only foundation of the CAD layer. Deliberately tiny and dependency-
free so that any child process can load it **standalone** — by path, with no
package context and no ``dycraft`` on ``sys.path`` — which is exactly what
``_adjudicate_runner`` does before it decides whether a reported collision is
real.

``SolidCompileError`` lives here rather than in ``_solid_compile`` because
``check_build123d_version`` raises it and the version check is the one thing
that must be reachable without importing the compiler. ``_solid_compile``
re-exports it, so callers and ``except`` clauses see a single class.
"""

from __future__ import annotations

import sys


#: Every kind `SolidCompileError` may carry, with what the author should DO
#: about it. A declared table rather than free text, for the same reason
#: ``object_worker.FAILURE_KINDS`` is one: a kind is a wire value that crosses a
#: process boundary and selects the advice the agent acts on, so an unknown kind
#: must be a loud programming error rather than a label that silently reaches
#: the agent with nothing attached. `test_object_failure_classification.py`
#: asserts the two stay in step.
#:
#: The advice is deliberately about the *geometry*, not about the program: every
#: kind here means the shape that was registered is not a body, so the fix is
#: always in how it was constructed.
SOLID_COMPILE_REPAIRS: dict = {
    "invalid_shape": (
        "Register a Solid, Part, or Compound of Solids. Extrude, revolve, or loft a "
        "Face, Wire, Edge, or Sketch into a body first.",
        "For a boolean result, check operand validity, placement, and intended "
        "overlap; rebuild self-intersecting or non-manifold geometry.",
    ),
    "not_a_shape": (
        "pass the build123d shape itself, not a number, a list, or the result "
        "of a call that returns None",
    ),
    "empty_shape": (
        "The result is empty. Check operand positions and whether subtraction removed "
        "the whole body or intersection found no overlap.",
        "`a - b` on non-touching solids silently returns `a`; `a & b` on "
        "non-touching shapes returns an empty shape or None (notably Solid & x)",
    ),
    "empty_mesh": (
        "No usable triangles remained after mesh cleanup. Check for collapsed "
        "dimensions or degenerate surfaces.",
    ),
    "invalid_mesh": (
        "A mesh coordinate is non-finite. Check upstream arithmetic, including zero "
        "dimensions and zero-length rotation axes.",
    ),
    "tessellation_failure": (
        "OCCT could not triangulate the shape. Check its validity and simplify the "
        "failing surface; a primitive and boolean may replace a difficult loft or "
        "sweep.",
    ),
    "brep_write_failure": (
        "OCCT could not serialize the shape. Check its topology and rebuild the body "
        "with simpler operations if invalid.",
    ),
    "exec_exception": (
        "The shape built and tessellated, but mass-property measurement failed. Check "
        "that it contains nonempty, closed solids.",
    ),
    "ocp_missing": (
        "Environment fault: OCP, the OCCT binding required by build123d, is not "
        "importable in the CAD interpreter. Repair the environment; do not rewrite "
        "the geometry.",
    ),
    "build123d_version_mismatch": (
        "Environment fault: use the pinned build123d version in the CAD interpreter; "
        "do not rewrite the geometry.",
    ),
}

#: The kinds themselves, derived so the two can never disagree.
SOLID_COMPILE_KINDS = tuple(SOLID_COMPILE_REPAIRS)


def kind_repairs(kind: str) -> list:
    """What to do about a `SolidCompileError` of this `kind`.

    Empty for an unrecognized kind rather than raising: this is consulted on an
    error path, where a lookup that itself fails would replace a specific
    geometry error with a confusing internal one.
    """
    return list(SOLID_COMPILE_REPAIRS.get(kind, ()))


class SolidCompileError(Exception):
    """A structured, LLM-repairable failure raised during Shape compilation.

    ``kind`` is the wire-level error kind: a short stable token from
    ``SOLID_COMPILE_KINDS`` that crosses the subprocess boundary in the result
    JSON and **selects the repair advice the agent acts on** (see
    ``SOLID_COMPILE_REPAIRS``), so it is part of the contract rather than a
    free-text label.

    An unknown kind raises. That is the same rule ``object_worker.worker_failure``
    applies to ``FAILURE_KINDS``, and for the same reason: a kind selects
    advice, so it cannot be allowed to be a free string or a typo.
    """

    def __init__(self, kind: str, message: str):
        if kind not in SOLID_COMPILE_REPAIRS:
            raise ValueError(
                f"unknown SolidCompileError kind {kind!r}; add it to "
                f"SOLID_COMPILE_REPAIRS so it is required to answer "
                f"'what should the author do about this?'")
        super().__init__(message)
        self.kind = kind
        self.message = message


# ---------------------------------------------------------------------------
# The frozen CAD version — single source of truth for the whole project.
#
# DyCraft targets exactly one build123d. Supporting a range would mean carrying
# compatibility branches through every geometry call, and the 0.9.1 -> 0.11.1
# move showed why that is not worth it: `Face.axis_of_rotation` does not exist
# in 0.9.1 and `Shape.is_valid` changed from a method to a property, so
# "works on both" quietly means "silently degraded on one".
#
# Checked with `importlib.metadata.version`, which reads the installed
# distribution rather than trusting a `__version__` attribute the package may
# not define.
# ---------------------------------------------------------------------------

REQUIRED_BUILD123D = "0.11.1"


def installed_build123d_version():
    """The installed build123d version string, or None if it is absent."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — Python < 3.8
        return None
    try:
        return version("build123d")
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a broken metadata dir is "unknown"
        return None


def check_build123d_version(*, strict: bool = True) -> str:
    """Fail fast unless the interpreter carries exactly the pinned build123d.

    Called at the top of every CAD child process. A version skew otherwise
    surfaces as a wrong-looking measurement or a silently missing feature
    inventory — far more expensive to diagnose than an explicit refusal.
    """
    found = installed_build123d_version()
    if found == REQUIRED_BUILD123D:
        return found
    if not strict:
        return found or "(not installed)"
    raise SolidCompileError(
        "build123d_version_mismatch",
        f"DyCraft is frozen at build123d=={REQUIRED_BUILD123D} but this "
        f"interpreter has {found or 'no build123d'} "
        f"({sys.executable}). Point DYCRAFT_B123D_PYTHON (or .b123d_python) at "
        f"the pinned environment — see environment-b123d.yml.")


# ---------------------------------------------------------------------------
# OCP, the OCCT binding underneath build123d.
#
# `object.py` may import it directly, because build123d's fluent layer does not
# expose every control its own kernel has — a variable-radius fillet and a
# tolerant sew are constructors on OCCT algorithm classes with no build123d
# spelling. That makes OCP part of the authoring contract rather than an
# implementation detail of a dependency, so its absence has to fail the same
# way a build123d skew does: once, loudly, at the start.
#
# Presence, not a version. OCP ships under several distribution names
# (`cadquery-ocp`, `cadquery-ocp-novtk`), so `find_spec` asks the only question
# that matters to an author — does `import OCP` work — and the version is read
# separately for the record.
# ---------------------------------------------------------------------------

#: Distributions that provide the `OCP` package, most specific first. Only used
#: to label what was found; the gate is `find_spec`.
_OCP_DISTRIBUTIONS = ("cadquery-ocp-novtk", "cadquery-ocp", "ocp")


def installed_ocp_version():
    """The installed OCP distribution version, or None if it cannot be read.

    None is not a failure — a vendored or path-installed OCP imports fine with
    no distribution metadata. `check_ocp_available` gates on importability.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — Python < 3.8
        return None
    for dist in _OCP_DISTRIBUTIONS:
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — a broken metadata dir is "unknown"
            return None
    return None


def ocp_is_importable() -> bool:
    """Whether `import OCP` would resolve in this interpreter.

    `find_spec` rather than a real import: this runs at every CAD child start,
    and importing OCP costs a second and hundreds of megabytes that the child
    only pays for if the authored source actually asks.
    """
    try:
        from importlib.util import find_spec
        return find_spec("OCP") is not None
    except BaseException:  # noqa: BLE001 — an unimportable OCP is "absent"
        return False


def check_ocp_available(*, strict: bool = True) -> str:
    """Fail fast unless the interpreter carries OCP.

    Called at the top of every CAD child, beside `check_build123d_version`.
    An `object.py` that imports OCP would otherwise fail with a bare
    `ModuleNotFoundError` attributed to the authored line that imported it,
    which reads as a geometry mistake and is not one.
    """
    if ocp_is_importable():
        return installed_ocp_version() or "(unknown version)"
    if not strict:
        return "(not installed)"
    raise SolidCompileError(
        "ocp_missing",
        f"this interpreter has no importable OCP ({sys.executable}). OCP is "
        f"the OCCT binding build123d is built on, so an environment with "
        f"build123d=={REQUIRED_BUILD123D} and no OCP is broken. Point "
        f"DYCRAFT_B123D_PYTHON (or .b123d_python) at the pinned environment "
        f"— see environment-b123d.yml.")
