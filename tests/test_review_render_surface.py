"""The independent reviewer's view tool must be as capable as `snapshot`.

A view tool with framing plus azimuth and elevation and nothing else leaves a
claim about a surface the model covers one honest answer, `cant_tell`. Most
`cant_tell` verdicts are the model occluding itself, an unsampled joint pose,
or crop and resolution, so the reviewer gets what `snapshot` has: `only`
("names the parts to draw and hides the rest"), joint poses and resolution.
"""
from __future__ import annotations

import inspect

from dycraft.agent.layers import _review_render
from dycraft.agent.layers import review, snapshot


def _parser_flags() -> set[str]:
    import argparse

    seen: set[str] = set()
    real_init = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kw):
        for a in args:
            if isinstance(a, str) and a.startswith("--"):
                seen.add(a)
        return real_init(self, *args, **kw)

    argparse.ArgumentParser.add_argument = spy
    try:
        try:
            _review_render.main(["_review_render", "--help"])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.add_argument = real_init
    return seen


def test_the_reviewer_can_hide_occluders_and_pose_joints():
    flags = _parser_flags()
    for flag in ("--only", "--poses", "--resolution", "--fov-deg"):
        assert flag in flags, f"{flag} missing; reviewer has {sorted(flags)}"


def test_it_reuses_snapshot_validation_rather_than_copying_it():
    # A second copy of the degrees-to-radians conversion is a second thing
    # that can stop agreeing with the author's `snapshot`.
    src = inspect.getsource(_review_render.main)
    assert "_snapshot_only" in src
    assert "_snapshot_poses" in src
    assert "_snapshot_resolution" in src
    for name in ("_snapshot_only", "_snapshot_poses", "_snapshot_resolution"):
        assert callable(getattr(snapshot, name)), name


def test_the_new_inputs_reach_the_render_request():
    src = inspect.getsource(_review_render.main)
    assert "only=only_parts" in src
    assert "poses=ir_poses" in src
    assert "resolution=res" in src


def test_the_reviewer_prompt_documents_the_whole_surface():
    # A capability the reviewer is not told about is not a capability.
    src = inspect.getsource(review)
    i = src.index("If a different view would settle one of these")
    block = src[i:i + 2000]
    assert "--only" in block
    assert "--poses" in block
    assert "--resolution" in block
    assert "--fov-deg" in block

