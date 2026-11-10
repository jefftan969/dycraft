"""How a run picks its renderer, and why the prompt cannot.

A run launched with the prompt "…and render the final result in Blender" ran
entirely on Open3D, and its agent then burned steps working around neutral-gray
shader previews — including relaunching the harness daemon twice to force
Blender. The prompt never had any bearing on the renderer: it is a launcher
setting.

Three behaviors pinned here:

  1. Resolution order — flag > `$DYCRAFT_RENDER_BACKEND` > the code default.
     The env tier is what makes `export DYCRAFT_RENDER_BACKEND=blender` a usable
     machine-wide default, so nothing may assign a hardcoded "open3d" over it.
  2. A warning when the prompt names a renderer the run will not use. A warning
     and not an action: a keyword match must not silently take over a run-wide
     setting that costs real render time.
  3. `set_render_backend` refusing a backend this machine cannot run, at the
     switch rather than once per render several steps later.
"""

from __future__ import annotations

import os

import pytest

from dycraft.cli.cli import (
    DEFAULT_RENDER_BACKEND,
    RENDER_BACKEND_ENV,
    warn_render_backend_mismatch,
)
from dycraft.render import default_backend

BLENDER_PROMPT = ("Use build123d to create a detailed microwave oven with "
                  "material textures, and render the final result in Blender.")


# ---------------------------------------------------------------------------
# 1. Resolution order
# ---------------------------------------------------------------------------

def _resolve(flag: str | None) -> str:
    """The launcher's pin, extracted so it can be exercised without main()."""
    return (flag or os.environ.get(RENDER_BACKEND_ENV)
            or DEFAULT_RENDER_BACKEND)


def test_the_flag_wins_over_everything(monkeypatch):
    monkeypatch.setenv(RENDER_BACKEND_ENV, "blender")
    assert _resolve("open3d") == "open3d"


def test_an_exported_env_var_sets_the_default(monkeypatch):
    """A machine-wide default without the flag: the exported variable is honoured."""
    monkeypatch.setenv(RENDER_BACKEND_ENV, "blender")
    assert _resolve(None) == "blender"


def test_the_code_default_applies_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv(RENDER_BACKEND_ENV, raising=False)
    assert _resolve(None) == DEFAULT_RENDER_BACKEND


def test_the_pinned_value_is_what_the_renderer_reads(monkeypatch):
    """`render.default_backend()` re-reads the env per RenderRequest, so the
    launcher's pin is the whole mechanism — nothing else has to be threaded."""
    monkeypatch.setenv(RENDER_BACKEND_ENV, _resolve("blender"))
    assert default_backend() == "blender"


# ---------------------------------------------------------------------------
# 2. Prompt/renderer mismatch warning
# ---------------------------------------------------------------------------

def test_a_prompt_asking_for_blender_on_an_open3d_run_warns():
    message = warn_render_backend_mismatch(
        BLENDER_PROMPT, effective="open3d", flag_given=False)
    assert message is not None
    # Actionable: name both the per-run flag and the persistent export.
    assert "--render-backend blender" in message
    assert f"export {RENDER_BACKEND_ENV}=blender" in message
    # And be explicit that nothing was changed on the prompt's behalf.
    assert "Naming a renderer in prose does not set it" in message
    assert "h.set_render_backend('blender')" in message


def test_an_explicit_flag_silences_the_warning():
    """The operator already chose; a keyword match must not second-guess them.

    This is the case where the prompt mentions Blender but the run was
    deliberately launched on Open3D for speed.
    """
    assert warn_render_backend_mismatch(
        BLENDER_PROMPT, effective="open3d", flag_given=True) is None


def test_no_warning_when_the_prompt_matches_the_active_renderer():
    assert warn_render_backend_mismatch(
        BLENDER_PROMPT, effective="blender", flag_given=False) is None


@pytest.mark.parametrize("prompt", [
    "",
    "A sturdy dining chair",
    # Word-boundary matched: a substring must not trip it.
    "a blenderized mesh with recycled topology",
])
def test_no_warning_for_a_prompt_that_names_no_renderer(prompt):
    assert warn_render_backend_mismatch(
        prompt, effective="open3d", flag_given=False) is None


def test_cycles_counts_as_asking_for_blender():
    """`cycles` is Blender's engine, and is what a prompt asking for photoreal
    output tends to say."""
    message = warn_render_backend_mismatch(
        "Render the final frame with Cycles at 256 samples",
        effective="open3d", flag_given=False)
    assert "asks for 'blender'" in message and "--render-backend blender" in message


def test_the_warning_does_not_change_the_renderer(monkeypatch):
    """Warn, never act — the whole point of choosing this over auto-select."""
    monkeypatch.setenv(RENDER_BACKEND_ENV, "open3d")
    warn_render_backend_mismatch(BLENDER_PROMPT, effective="open3d",
                                 flag_given=False)
    assert os.environ[RENDER_BACKEND_ENV] == "open3d"
    assert default_backend() == "open3d"


def test_the_daemon_env_carries_the_resolved_renderer(tmp_path, monkeypatch):
    """`build_env_dict` gets the raw *flag*, which is None when unset.

    Falling back to a literal there would export `open3d` to the daemon even on
    a run whose renderer came from the environment, undoing the resolution.
    """
    from dycraft.cli.cli import build_env_dict

    monkeypatch.setenv(RENDER_BACKEND_ENV, "blender")
    env = build_env_dict(
        "p", run_dir=tmp_path, runs_dir=tmp_path, run_id="r", layer="object",
        runtime="codex_cli", render_backend=None)
    assert env[RENDER_BACKEND_ENV] == "blender"


def test_an_explicit_flag_still_beats_the_environment_for_the_daemon(
        tmp_path, monkeypatch):
    from dycraft.cli.cli import build_env_dict

    monkeypatch.setenv(RENDER_BACKEND_ENV, "blender")
    env = build_env_dict(
        "p", run_dir=tmp_path, runs_dir=tmp_path, run_id="r", layer="object",
        runtime="codex_cli", render_backend="open3d")
    assert env[RENDER_BACKEND_ENV] == "open3d"


# ---------------------------------------------------------------------------
# `set_render_backend` — the op an agent calls mid-run
# ---------------------------------------------------------------------------

def _setter(cls):
    """The bound `_do_set_render_backend` of a session that was never started.

    `object.__new__` rather than a real session: this method reads and writes
    one env var and traces, so standing up a daemon to exercise it would test
    the daemon.
    """
    stub = object.__new__(cls)
    stub.trace = lambda *a, **k: None
    return lambda backend: cls._do_set_render_backend(stub, backend)


def _session_classes():
    from dycraft.agent.layers.session import LayerSession
    return [LayerSession]


@pytest.mark.parametrize("cls", _session_classes(), ids=lambda c: c.__name__)
def test_switching_to_blender_is_refused_where_blender_cannot_run(cls, monkeypatch):
    """The registry knows the name exists; it does not know this machine.

    An empty `$DYCRAFT_BLENDER_PYTHON` is the documented "no interpreter"
    sentinel (`_blender_env._resolve_blender_python`). Without this check the
    op would return ok and the refusal would arrive once per render, several
    steps on, each one a round trip out to `blender` and back.
    """
    monkeypatch.setenv("DYCRAFT_BLENDER_PYTHON", "")
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")

    result = _setter(cls)("blender")

    assert result["ok"] is False
    assert "DYCRAFT_BLENDER_PYTHON" in result["error"]
    assert os.environ["DYCRAFT_RENDER_BACKEND"] == "open3d", (
        "a refused switch must not leave the run pointed at the backend it "
        "just refused")


@pytest.mark.parametrize("cls", _session_classes(), ids=lambda c: c.__name__)
def test_switching_to_an_available_backend_still_works(cls, monkeypatch):
    monkeypatch.setenv("DYCRAFT_BLENDER_PYTHON", "")
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "blender")

    result = _setter(cls)("open3d")

    assert result["ok"] is True
    assert result["render_backend"] == "open3d"
    assert result["previous"] == "blender"
    assert os.environ["DYCRAFT_RENDER_BACKEND"] == "open3d"


@pytest.mark.parametrize("cls", _session_classes(), ids=lambda c: c.__name__)
def test_an_unknown_backend_name_is_still_refused_first(cls, monkeypatch):
    """The availability check is additional, not a replacement: a typo and a
    missing dependency need different advice, and the name check is what
    distinguishes them."""
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")

    result = _setter(cls)("raytracer9000")

    assert result["ok"] is False
    assert "unknown render backend" in result["error"]
