"""Prompt-drift detector for the object-agent prompts.

Mirrors `test_agent_scene_prompt_drift.py` for what remains checkable: every
probe the object prompts name must be bound on the object layer, and the poses
claim must match the real probe signatures. The agent prompt and the SDK must
stay aligned, and a renamed or unbound probe is a call the agent cannot make.

The object prompt carries no build123d geometry for an agent to copy, so there
is nothing here to execute; `tests/test_object_namespace_reference.py` guards
the build123d surface the prompt does describe.
"""

import re
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as _worker

_needs_cad = pytest.mark.skipif(
    not _worker.is_available(), reason="no build123d interpreter")


_PROMPT = Path(__file__).resolve().parents[1] / "agent" / "prompts" / "object.md"




def _object_bound_names() -> set:
    """Short names the object probe overlay binds, read from its source.

    There is no registry for this layer — `_object_probe_overlay` writes the
    dict out inline — so the source is the list.
    """
    src = (Path(__file__).resolve().parents[1]
           / "agent" / "layers" / "layer.py").read_text()
    body = re.search(r"def _object_probe_overlay\(.*?\n\ndef ", src,
                     re.DOTALL).group(0)
    return set(re.findall(r'^\s*"([a-z_]+)":\s*(?:_probes\.|_bind|_object_source)',
                          body, re.MULTILINE))


def test_the_object_poses_claim_matches_the_real_probe_signatures():
    """The prompt's two `poses=` lists must match the signatures exactly.

    A name in the first list that lacks the kwarg costs a turn and a
    `TypeError`; a name in the second that accepts one hides a pose-tagged
    answer the agent could have had.
    """
    import inspect

    from dycraft.sdk import _probes

    text = _PROMPT.read_text()
    block = text.split("**Pose-tagged probes.**")[1].split("\n\n")
    takes = set(re.findall(r"`([a-z_]+)`", block[0]))
    takes_none = set(re.findall(r"`([a-z_]+)`", block[1]))

    bound = _object_bound_names()
    real = {n for n in bound
            if hasattr(_probes, f"object_{n}")
            and "poses" in inspect.signature(
                getattr(_probes, f"object_{n}")).parameters}

    assert takes & bound == real, (
        "the prompt's pose-tagged list disagrees with the signatures: "
        f"claimed-not-real {sorted((takes & bound) - real)}, "
        f"real-not-claimed {sorted(real - takes)}")
    for name in sorted(takes_none & bound):
        assert name not in real, (
            f"the prompt says `{name}` takes no `poses=`, but it does")


@pytest.mark.parametrize("prompt", ["object.md"])
def test_every_probe_the_object_prompts_call_is_bound_on_this_layer(prompt):
    """A probe named in call position must be bound here, or it is a `NameError`."""
    from dycraft.sdk import _probes

    text = (Path(__file__).resolve().parents[1]
            / "agent" / "prompts" / prompt).read_text()
    bound = _object_bound_names()
    every = set()
    for attr in dir(_probes):
        for prefix in ("object_", "scene_", "animation_"):
            if attr.startswith(prefix) and callable(getattr(_probes, attr)):
                every.add(attr[len(prefix):])
    for name in sorted(every - bound):
        assert f"`{name}(" not in text, (
            f"{prompt} calls `{name}(...)`, which is a probe on another layer "
            f"and is not bound on the object layer")
