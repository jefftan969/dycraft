"""Finding the frozen build123d interpreter, and launching a child under it.

`agent/cad/_cad_env.py` is the single resolution chain for the CAD
environment. Three things depend on getting the same answer from it:

  * `object_worker` — builds every `object.py`;
  * `verify/_exact_adjudicator` — rebuilds exact B-rep geometry to clear SAT
    false positives;
  * `session` / `cli` — preflight, so a run refuses to start rather than
    discovering the shortfall mid-way.

A run whose parts were built by one build123d and whose collisions were judged
by another produces verdicts about geometry that does not exist, and nothing
downstream would notice. That is what these tests are for: the chain, its
precedence, and the fact that all three consumers walk it.

No CAD interpreter required: resolution is filesystem and environment logic.

Run: `pytest dycraft/tests/test_cad_env.py -v`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from dycraft.agent.cad import _b123d_version as SC
from dycraft.agent.cad import _cad_env as E

_ENV_VAR = "DYCRAFT_B123D_PYTHON"


# ---------------------------------------------------------------------------
# Candidate probing
# ---------------------------------------------------------------------------

def test_an_explicitly_named_interpreter_is_taken_even_if_not_executable():
    """`launch_failed` and `unavailable` are opposite diagnoses.

    "Your interpreter is broken" sends you to fix a path; "nothing was
    configured" sends you to set one. Rejecting a caller's explicit choice for
    a missing exec bit would report the second when the first is true, so a
    named candidate is handed on and allowed to fail at launch.
    """
    # This test file: a real, regular, non-executable file.
    here = __file__
    assert not os.access(here, os.X_OK), "fixture is unexpectedly executable"
    assert E._usable(here) == here

    # An auto-discovered candidate is a guess, so the exec bit IS required.
    assert E._usable(here, require_exec=True) is None


def test_a_bare_name_resolves_through_path_and_a_missing_one_does_not():
    assert E._usable("python3") == __import__("shutil").which("python3")
    assert E._usable("definitely-not-an-interpreter-xyz") is None
    assert E._usable(None) is None
    assert E._usable("") is None


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

def test_an_explicit_argument_outranks_the_environment():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, "/nonexistent/from-env")
        assert E._resolve_b123d_python(sys.executable) == sys.executable


def test_the_environment_outranks_the_pointer_file(tmp_path):
    """The pointer file is the recorded default; the env var is the override.

    The override is a path of this test's own making, not `sys.executable`: a
    checkout whose `.b123d_python` happens to record the interpreter running the
    suite would make the two indistinguishable and the assertion vacuous.
    """
    assert E._PYTHON_POINTER_FILE.is_file(), "fixture: .b123d_python is missing"
    override = tmp_path / "python-from-env"
    override.write_text("")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, str(override))
        assert E._resolve_b123d_python() == str(override)
        assert E._resolve_b123d_python() != E._pointer_file_python()


def test_the_pointer_file_is_used_when_the_environment_is_unset():
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(_ENV_VAR, raising=False)
        recorded = E._pointer_file_python()
        if not recorded or not Path(recorded).is_file():
            pytest.skip("no usable interpreter recorded in .b123d_python")
        assert E._resolve_b123d_python() == recorded


def test_the_pointer_file_ignores_comments_and_blank_lines():
    """It is a human-edited file; the header explains the version freeze."""
    text = E._PYTHON_POINTER_FILE.read_text()
    assert text.lstrip().startswith("#"), "fixture: no comment header to skip"
    recorded = E._pointer_file_python()
    assert recorded and not recorded.startswith("#")
    assert recorded.strip() == recorded


def test_set_but_empty_is_a_deliberate_no_interpreter_sentinel():
    """The only way to prove a run fails loudly instead of degrading.

    `test_object_worker_routing` monkeypatches exactly this to show that a
    missing CAD environment is a build failure and that the session refuses to
    start. Both rest on the empty string suppressing every fallback — including
    the pointer file and the conventional prefixes, which exist on this host.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, "")
        assert E._resolve_b123d_python() is None
        assert E.b123d_python() is None
        # Whitespace is the same statement, not a path.
        mp.setenv(_ENV_VAR, "   ")
        assert E.b123d_python() is None


def test_the_ambient_interpreter_rescues_a_checkout_with_no_pointer_file(
        monkeypatch):
    """The last resort, and why it exists.

    Every candidate ahead of it is repo-relative or host-conventional, so all of
    them can be absent in a context that is otherwise fine: `_REPO_ROOT` is this
    module's grandparent, which inside a git worktree is the *worktree*, where
    the untracked `.b123d_python` does not exist, while the interpreter
    running the suite may carry the pinned build123d all along.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.setattr(E, "_PYTHON_POINTER_FILE",
                        Path("/nonexistent/.b123d_python"))
    monkeypatch.setattr(E, "_conventional_pythons",
                        lambda: ["/nonexistent/conda_envs/x/bin/python"])
    found = E._resolve_b123d_python()
    if found is None:
        pytest.skip("the interpreter running the suite has no build123d")
    assert E.probe_build123d_version(found) == SC.REQUIRED_BUILD123D


def test_the_ambient_fallback_is_gated_on_the_pinned_version(monkeypatch):
    """Presence is not the test; the version is.

    The rollback env carries 0.9.1 and the worker child refuses it with
    `build123d_version_mismatch`. An ambient interpreter that merely *has*
    build123d would therefore trade one clear startup failure for a confusing
    one on every compile.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.setattr(E, "_PYTHON_POINTER_FILE",
                        Path("/nonexistent/.b123d_python"))
    monkeypatch.setattr(E, "_conventional_pythons",
                        lambda: ["/nonexistent/conda_envs/x/bin/python"])
    monkeypatch.setattr(E, "_ambient_pythons", lambda: ["/usr/bin/python3"])
    E._has_pinned_build123d.cache_clear()
    try:
        assert E._resolve_b123d_python() is None
    finally:
        E._has_pinned_build123d.cache_clear()


def test_an_explicit_interpreter_is_never_version_gated():
    """`launch_failed` must not silently become a run somewhere else.

    Gating the caller's own choice would mean a typo in `DYCRAFT_B123D_PYTHON`
    produces a *successful* run against an interpreter nobody selected — strictly
    worse than the error it replaces.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, "/usr/bin/python3")
        assert E._resolve_b123d_python() == "/usr/bin/python3"


def test_the_pinned_env_is_searched_before_the_rollback_baseline():
    """0.11.1 first; the unsuffixed env carries 0.9.1 and must not win by luck."""
    assert E._ENV_NAMES[0] == "dycraft-b123d-0.11.1"
    candidates = E._conventional_pythons()
    first_pinned = min(i for i, c in enumerate(candidates)
                       if "dycraft-b123d-0.11.1" in c)
    first_rollback = min(i for i, c in enumerate(candidates)
                         if c.endswith("/dycraft-b123d/bin/python"))
    assert first_pinned < first_rollback, candidates


# ---------------------------------------------------------------------------
# Child environment
# ---------------------------------------------------------------------------

def test_no_interpreter_means_the_child_env_is_just_the_base():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, "")
        assert E.b123d_child_env({"FOO": "bar"}) == {"FOO": "bar"}


def test_the_child_env_prepends_the_interpreters_own_lib_dir():
    """The expat 2.2-vs-2.8 fix, without which every build123d import dies.

    `cadquery-ocp` links the system libexpat; once OCP loads it, the conda
    `pyexpat` fails with `undefined symbol:
    XML_SetAllocTrackerActivationThreshold`. build123d 0.11.1 imports IPython at
    module scope, which reaches `xml.dom.minidom`, so *every* import hits it.
    """
    interp = E.b123d_python()
    if not interp:
        pytest.skip("no build123d interpreter configured")
    lib = Path(interp).resolve().parent.parent / "lib"
    if not lib.is_dir():
        pytest.skip(f"{lib} does not exist on this host")

    env = E.b123d_child_env({})
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(lib)

    # An existing search path is preserved, behind ours.
    env = E.b123d_child_env({"LD_LIBRARY_PATH": "/opt/prior"})
    assert env["LD_LIBRARY_PATH"] == f"{lib}{os.pathsep}/opt/prior"


def test_the_child_env_copies_rather_than_mutates_its_base():
    base = {"FOO": "bar"}
    env = E.b123d_child_env(base)
    assert env["FOO"] == "bar"
    assert "LD_LIBRARY_PATH" not in base, "the caller's dict was mutated"


# ---------------------------------------------------------------------------
# Where the adjudication child lives
# ---------------------------------------------------------------------------

def test_the_adjudication_runner_sits_beside_this_module():
    """`_runner_path` locates the child relative to the agent package."""
    from dycraft.verify import _exact_adjudicator as ADJ
    path = ADJ._runner_path()
    assert path.is_file(), path
    assert path.parent == Path(E.__file__).parent


# ---------------------------------------------------------------------------
# Asking the interpreter, not just resolving a path
# ---------------------------------------------------------------------------

def test_an_unusable_interpreter_probes_as_no_version(tmp_path):
    """A resolved path is not a working CAD environment.

    A `.b123d_python` recorded months ago can point at an interpreter that has
    since lost build123d. Resolution still succeeds, so the preflight would
    report the environment healthy and every compile would fail later, inside
    the worker. The probe is what makes that visible at launch.
    """
    not_python = tmp_path / "not-a-python"
    not_python.write_text("#!/bin/sh\nexit 127\n")
    not_python.chmod(0o755)
    assert E.probe_build123d_version(str(not_python)) is None


def test_no_interpreter_probes_as_no_version():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ENV_VAR, "")
        assert E.probe_build123d_version() is None
