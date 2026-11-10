"""One verification/review policy, applied to every authoring layer.

The policy, stated once:

  * **geometry is checked automatically**, and a failure blocks completion;
  * **the author inspects its own renders whenever it likes** — `snapshot`
    everywhere, and `view` where the layer has it;
  * **an independent critique is available on demand** — `review()`, which
    reads the author's own `add_check(...)` claims — and is *never*
    automatically invoked, and never a precondition for `finish`.

What this file catches is the same question answered differently depending on
which layer authored the geometry, or on whether the author chose to delegate.
"""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

from dycraft.agent import stage_harness_client
from dycraft.agent.layers.layer import _LAYER_CONFIGS, scene_layer_config


def _names(cfg) -> list:
    return [n for n, _ in (cfg.verifiers or ())]


# --------------------------------------------------------------- the suites


def test_geometry_checks_survive_on_every_layer():
    """The half of the policy that must NOT have been relaxed."""
    obj = _names(_LAYER_CONFIGS["object"]())
    scene = _names(_LAYER_CONFIGS["scene"]())
    assert "NoSelfPenetration" in obj and "SupportConnected" in obj
    assert "NoSelfPenetration" in scene
    assert "NoCrossInstancePenetration" in scene
    # And the relational assertions, which are the author's own numbers.
    assert "ObjectRelationalAssertions" in obj
    assert "SceneRelationalAssertions" in scene


# ------------------------------------------------- the on-demand reviewer


@pytest.mark.parametrize("layer", ["object", "scene"])
def test_review_is_registered_wherever_claims_can_be_declared(layer):
    assert _LAYER_CONFIGS[layer]().supports_review, (
        f"{layer} accepts add_check claims but cannot buy the review that "
        "reads them")


def test_review_is_staged_and_served_on_every_layer_that_offers_it():
    """Stub set and handler set, both directions, for the layers changed here.

    A stub the daemon never registered is an advertised tool that cannot run;
    a handler with no stub is a capability the agent cannot reach.
    """
    from dycraft.agent.layers.session import (LayerSession,
                                                    SessionConfig)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(stage_harness_client(Path(tmp) / "scene", "scene"))
        stubs = {n.name for n in ast.parse(staged.read_text()).body
                 if isinstance(n, ast.FunctionDef)
                 and not n.name.startswith("_")}
        sess = LayerSession(SessionConfig(
            run_dir=Path(tmp) / "run", layer=scene_layer_config(),
            user_prompt="x"))
        assert stubs == set(sess.handlers), (
            f"stub-only={stubs - set(sess.handlers)} "
            f"handler-only={set(sess.handlers) - stubs}")
        assert "review" in stubs and "view" not in stubs


# ------------------------------------------------------- self-inspection

