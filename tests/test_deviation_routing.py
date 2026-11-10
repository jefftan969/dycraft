"""`record_deviation`'s addressing fields: validation, the daemon op, and the trace fallback.

`blocked_layer` and `targets` are optional and advisory. They address a
deviation (which layer, which instances) without structuring its diagnosis,
and a parent reading them back through `_object_dispatch._read_deviations` is
free to act on them or not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.agent.layers import _object_dispatch as _disp
from dycraft.agent.layers.tools import normalize_deviation

PROMPTS = Path(_disp.__file__).resolve().parent.parent / "prompts"


# ---------------------------------------------------------------------------
# normalize_deviation — the shared validator
# ---------------------------------------------------------------------------


def _ok(**kw):
    record, err = normalize_deviation(
        what="w", why="y", severity="high", **kw)
    assert err is None, err
    return record


def test_the_addressing_fields_default_to_null():
    """An unchanged call keeps working and records both fields as null, so
    every consumer sees one shape regardless of how old the caller is."""
    r = _ok()
    assert r["blocked_layer"] is None
    assert r["targets"] is None


def test_a_bare_string_target_is_accepted():
    """Naming one instance is the common case and `targets="fridge"` is the
    natural thing to type."""
    assert _ok(targets="fridge")["targets"] == ["fridge"]


def test_targets_keeps_a_pair_because_a_conflict_has_two_ends():
    """The fix for 'the cabinet door hits the fridge' could be either one, so
    dropping to a single name would throw away half the report."""
    assert _ok(targets=["cabinet", "fridge"])["targets"] == \
        ["cabinet", "fridge"]


def test_targets_are_stripped_and_deduped_in_order():
    assert _ok(targets=[" a ", "b", "a", "", "  "])["targets"] == ["a", "b"]


def test_blocked_layer_null_is_legitimate():
    """The animation agent can measure that a door binds without knowing
    whether the fix is to move the neighbor or re-hinge the door."""
    assert _ok(blocked_layer=None)["blocked_layer"] is None
    assert _ok(blocked_layer="scene")["blocked_layer"] == "scene"


def test_blocked_layer_rejects_anything_outside_the_two_values():
    _, err = normalize_deviation(what="w", why="y", severity="high",
                                 blocked_layer="animation")
    assert err is not None and "blocked_layer" in err["error"]


def test_too_many_targets_is_rejected_as_a_design_problem():
    _, err = normalize_deviation(what="w", why="y", severity="high",
                                 targets=[f"i{n}" for n in range(20)])
    assert err is not None and "cap" in err["error"]


def test_a_non_string_target_is_rejected():
    record, err = normalize_deviation(what="w", why="y", severity="high",
                                      targets=["a", 7])
    assert record is None
    assert err["ok"] is False
    assert "targets" in err["error"] and err["evidence"]["got"] == "7"


@pytest.mark.parametrize("bad, names", [({"what": ""}, "what must be"),
                                        ({"why": ""}, "why must be"),
                                        ({"severity": "medium"}, "severity must be one of")])
def test_each_required_field_is_rejected_by_name(bad, names):
    record, err = normalize_deviation(**{"what": "w", "why": "y", "severity": "high", **bad})
    assert record is None
    assert err["ok"] is False and err["error"].startswith(names), err


# ---------------------------------------------------------------------------
# Both entry points must agree
# ---------------------------------------------------------------------------


def test_the_daemon_op_uses_the_one_validator(tmp_path):
    """The daemon op validates through `normalize_deviation` rather than a copy."""
    import inspect
    from dycraft.agent.layers import session as sess_mod

    src = inspect.getsource(sess_mod)
    # Both call the one validator rather than re-implementing it.
    assert "normalize_deviation(" in src
    tools_src = inspect.getsource(
        __import__("dycraft.agent.layers.tools", fromlist=["x"]))
    assert tools_src.count("def normalize_deviation") == 1
    # And the daemon op accepts the two new kwargs.
    assert "blocked_layer" in src and "targets" in src


def test_the_trace_fallback_carries_the_addressing_fields(tmp_path):
    """Deviations are harvested from trace.jsonl when a sub-run died before
    its meta flush — which is exactly the run whose report matters most, so
    the new fields have to survive that path too."""
    sub = tmp_path / "sub_runs" / "animation_001"
    sub.mkdir(parents=True)
    (sub / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"ev": "op_call", "op": "record_deviation", "args": {
            "what": "door binds on fridge", "why": "62 of 105 deg",
            "severity": "high", "suggested_fix": None,
            "blocked_layer": "scene", "targets": ["cabinet_1", "fridge_1"]}},
    ]) + "\n")
    got = _disp._deviations_from_trace(sub)
    assert got[0]["blocked_layer"] == "scene"
    assert got[0]["targets"] == ["cabinet_1", "fridge_1"]
    # meta.json's list wins when the child flushed one.
    assert _disp._read_deviations(sub, {"deliberate_deviations": [{"what": "m"}]}) == [{"what": "m"}]
    assert _disp._read_deviations(sub, {})[0]["what"] == "door binds on fridge"


def test_the_daemon_op_round_trips_the_new_kwargs(tmp_path):
    """The socket path is where a CLI-driven sub-agent actually files these.
    The stub sends both keys unconditionally, so the handler has to accept
    them by name — a signature mismatch would surface as a runtime error on
    every record_deviation call from claude/codex, and on none from the api
    backend."""
    from dycraft.agent.layers.layer import object_layer_config
    from dycraft.agent.layers.session import (
        LayerSession, SessionConfig,
    )

    sess = LayerSession(SessionConfig(
        run_dir=tmp_path, layer=object_layer_config(), user_prompt=""))
    with sess:
        out = sess.handlers["record_deviation"](
            what="counter has no cutout for me", why="measured 0 mm relief",
            severity="high", suggested_fix=None,
            blocked_layer="object", targets="counter")
        assert out["ok"] is True
        assert out["recorded"]["blocked_layer"] == "object"
        assert out["recorded"]["targets"] == ["counter"]

        bad = sess.handlers["record_deviation"](
            what="x", why="y", severity="high", blocked_layer="animation")
        assert bad["ok"] is False
        assert "blocked_layer" in bad["error"]
        # The rejected call must not have landed in the accumulator.
        assert len(sess.ctx.deliberate_deviations) == 1


# ---------------------------------------------------------------------------
# System-prompt snapshot
#
# Run bundles must be identical in shape across backends, so the snapshot has
# to be written by every surface that composes a prompt — not just the CLI
# one. These pin that, and the reason it exists at all: the prompt files are
# edited far more often than they are committed, so nothing else in the
# bundle identifies what a run actually ran against.
# ---------------------------------------------------------------------------


def test_the_snapshot_round_trips_the_composed_prompt(tmp_path):
    from dycraft.agent.agent_runner import (
        SYSTEM_PROMPT_FILE, compose_system_prompt, stage_system_prompt,
    )

    text = compose_system_prompt("object")
    stage_system_prompt(tmp_path, text)
    assert (tmp_path / SYSTEM_PROMPT_FILE).read_text() == text
    # It is the real thing, not a digest or a pointer.
    assert len(text) > 10_000
    assert "# Object agent" in text


def test_every_compose_call_site_stages_the_snapshot():
    """A launcher path that composes a prompt without snapshotting it would
    silently produce a bundle missing one file. Counted over the AST rather
    than the source text: a docstring that mentions the call is not a call."""
    import ast
    import inspect
    import dycraft.cli.cli as cli_mod

    def _count(mod, name):
        tree = ast.parse(inspect.getsource(mod))
        return sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name
        )

    for mod in (cli_mod,):
        n_compose = _count(mod, "compose_system_prompt")
        n_stage = _count(mod, "stage_system_prompt")
        assert n_stage >= n_compose, (
            f"{mod.__name__}: {n_compose} compose_system_prompt call(s) but "
            f"{n_stage} stage_system_prompt call(s) — every path that "
            f"composes a prompt must also snapshot it")
    assert _count(cli_mod, "compose_system_prompt") >= 1


def test_the_scene_builder_does_not_forbid_reusing_a_handle():
    """`place`/`attach` take handle and name as separate arguments and only
    the name must be unique. If a duplicate-handle guard is ever added, the
    prompt guidance above becomes a lie."""
    import inspect
    from dycraft.sdk import scene_builder

    for fn in (scene_builder.Scene.place, scene_builder.Scene.attach):
        params = inspect.signature(fn).parameters
        assert "handle" in params and "name" in params, fn.__name__
    src = inspect.getsource(scene_builder.Scene.attach)
    assert "already placed" not in src.lower()


def test_the_scene_prompt_warns_that_bare_pose_keys_broadcast():
    """A bare `poses` key applies to every instance carrying that joint, and
    instances of one definition share joint names. The failure is silent in a
    probe, so the scene prompt has to say so.
    """
    text = (PROMPTS / "scene.md").read_text()
    # Pinned as the claim, not as one sentence: the warning has been worded
    # "A bare joint key broadcasts ..." and "Instances of one catalog object
    # share joint names. A bare key therefore affects all of them." A verbatim
    # match on either one turns a reword into a failure and tells the next
    # reader nothing about which half actually matters.
    assert "bare" in text and "broadcast" in text, (
        "scene.md no longer warns that a bare pose key broadcasts")
    assert '"<instance>.<joint>"' in text


def test_the_broadcast_semantic_is_still_what_the_kernel_does():
    """If the kernel ever stops broadcasting bare keys, the prompt warning
    above becomes wrong in the other direction."""
    import inspect
    from dycraft.sdk import _scene_kernel

    src = inspect.getsource(_scene_kernel._object_poses_for_instance)
    # qualified keys are filtered to the matching instance; bare keys fall
    # through to every instance.
    assert 'partition(".")' in src
    assert "out[key] = val" in src
