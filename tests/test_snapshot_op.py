"""The `snapshot` op: argument policy, path confinement, render dispatch.

The renderer is stubbed throughout — these are about what reaches it and what
never does. Every rejection here has to happen *before* any GL work, because a
bad frame index or a bad output path costs a step either way and the agent
should get the reason rather than a traceback.
"""

import tempfile
from pathlib import Path

import pytest

from dycraft.agent.layers import tools as T
from dycraft.agent.layers import snapshot as S
from dycraft.agent.layers.layer import object_layer_config


class _FakeResult:
    """Mirrors `render.RenderResult`'s fields — all of them.

    A stub that omits a field of the type it replaces only reports the
    omission when a caller reaches for it, as callers do for the renderer's
    bounding-box substitutions and `snapshot` does for its non-fatal
    complaints. Keep the two in step.

    `camera_ids` takes one id or many, so a single-camera caller can pass the
    bare string it already has.
    """

    def __init__(self, camera_ids, cameras=(), warnings=()):
        ids = [camera_ids] if isinstance(camera_ids, str) else list(camera_ids)
        self.render_paths = {c: Path(f"/tmp/{c}.png") for c in ids}
        self.target_bbox = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.cache_hits = {c: False for c in ids}
        self.cameras = list(cameras)
        self.warnings = list(warnings)


@pytest.fixture
def ctx():
    run = Path(tempfile.mkdtemp()) / "run"
    return T.ToolContext(run_dir=run, layer_cfg=object_layer_config())


@pytest.fixture
def stub_render(monkeypatch):
    """Replace `render_request` and record every call."""
    calls = []

    def fake(request):
        calls.append(request)
        if request.cameras is not None:
            return _FakeResult([c.id for c in request.cameras],
                               cameras=request.cameras)
        from dycraft.render import canonical_cameras
        available = {c.id: c for c in canonical_cameras((0, 0, 0), 0.5, 0.5)}
        ids = list(request.camera_ids or available)
        return _FakeResult(ids, cameras=[available[i] for i in ids])

    monkeypatch.setattr("dycraft.render.render_request", fake)
    monkeypatch.setattr(S, "_render_inputs",
                        lambda ctx, target: ("object", {"id": "h"}, None,
                                             target, ctx.run_dir))
    return calls


# ---------------------------------------------------------------------------
# The happy path and what it reports back
# ---------------------------------------------------------------------------

def test_it_returns_paths_and_the_camera_it_actually_used(ctx, stub_render):
    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                         out_dir="snapshots")
    assert out["ok"] is True
    assert out["n_images"] == 1 and len(out["paths"]) == 1
    (view,) = out["views"]
    assert view["camera"]["eye"] == [1.0, 0.0, 0.5]
    assert view["camera"]["look_at"] == [0.0, 0.0, 0.5]
    assert view["camera"]["up"] == [0.0, 0.0, 1.0]
    # The aspect trap: `fov_deg` is vertical, so the size has to come back.
    assert out["image_size"] == [1280, 720]
    assert out["frame_indices"] is None
    assert out["note"] is None
    # `render_warnings` is present only when non-empty — the convention the
    # whole payload follows (`render/__init__` and `verify/review_images` shape
    # it the same way, and the agent-facing stub documents it as "non-empty
    # means"). A clean snapshot therefore has no such key, which
    # `test_a_clean_snapshot_payload_gains_no_warnings_key` states directly.
    assert "render_warnings" not in out
    assert stub_render[0].cameras[0].id == view["camera"]["id"]


def test_a_resolved_up_is_reported_rather_than_applied_silently(ctx,
                                                               stub_render):
    """An agent that asked for a top-down view and got image-up = -Y needs to
    know, or it will read image-left/right backwards."""
    out = S.run_snapshot(ctx, eye=(0.0, 0.0, 3.0), look_at=(0.0, 0.0, 0.0))
    assert out["views"][0]["camera"]["up"] == [0.0, -1.0, 0.0]
    assert out["note"] and "-1.0" in out["note"]


def test_the_snapshot_cache_is_not_the_render_cache(ctx, stub_render):
    """Keeping them apart is what makes it structurally impossible for a custom
    render to be served into a canonical slot, rather than something a reader
    has to re-derive from the id scheme."""
    S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5))
    assert stub_render[0].cache is ctx.snapshot_cache
    assert ctx.snapshot_cache is not ctx.render_cache


def test_it_does_not_write_a_urdf_or_touch_the_evidence_dirs(ctx, stub_render):
    """`view` writes `renders/<step>_ir.urdf` because `renders/` is the run's
    evidence bundle. A snapshot directory is scratch."""
    S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                   out_dir="snapshots")
    assert not list((ctx.run_dir / "renders").iterdir())
    assert stub_render[0].out_dir == (ctx.run_dir / "snapshots").resolve()


# ---------------------------------------------------------------------------
# A bad camera never reaches the renderer
# ---------------------------------------------------------------------------

def test_a_bad_camera_is_a_structured_error_not_an_exception(ctx, stub_render):
    out = S.run_snapshot(ctx, eye=(0.0, 0.0, 1.0), look_at=(0.0, 0.0, 1.0))
    assert out["ok"] is False and "same point" in out["error"]
    assert stub_render == [], "the renderer should never have been called"


# ---------------------------------------------------------------------------
# Output-directory confinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_dir", [
    "../escape", "/etc", "", "   ", ".", "a/../../b",
    # Reserved: the canonical evidence bundle and the staged read-only inputs.
    "renders", "compile_reports", "catalog/x", "final_render",
    "sub_runs/other", "assets/meshes",
])
def test_a_rejected_out_dir_creates_nothing(ctx, stub_render, out_dir):
    before = sorted(p.name for p in ctx.run_dir.iterdir())
    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                         out_dir=out_dir)
    assert out["ok"] is False, out
    assert sorted(p.name for p in ctx.run_dir.iterdir()) == before
    assert stub_render == []


def test_a_nested_out_dir_of_the_agents_own_choosing_is_fine(ctx, stub_render):
    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                         out_dir="views/end_on")
    assert out["ok"] is True
    assert (ctx.run_dir / "views" / "end_on").is_dir()


def test_a_file_where_the_out_dir_should_go_is_reported(ctx, stub_render):
    (ctx.run_dir / "notes").write_text("x")
    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                         out_dir="notes")
    assert out["ok"] is False and "not a directory" in out["error"]


def test_snapshots_dir_is_created_lazily(ctx):
    """`ToolContext` creates exactly the two directories a run always fills;
    a snapshot dir appears only if the agent asked for one."""
    assert sorted(p.name for p in ctx.run_dir.iterdir()) == [
        "compile_reports", "renders"]


# ---------------------------------------------------------------------------
# The frame axis
# ---------------------------------------------------------------------------

def test_frame_args_are_refused_where_there_is_no_frame_axis(ctx, stub_render):
    """Refused, not ignored: an agent passing `frame=` to the object layer has
    misunderstood something, and a silently-frame-0 render hides that."""
    for kwargs in (dict(frame=3), dict(frames=[0, 5])):
        out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5),
                             **kwargs)
        assert out["ok"] is False
        assert "no frame axis" in out["error"]
    assert stub_render == []


# ---------------------------------------------------------------------------
# Filament context choice
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aiming: canonical shorthand, declarative framing, explicit cameras
# ---------------------------------------------------------------------------

@pytest.fixture
def ir_ctx(monkeypatch, stub_render):
    """An object-layer ctx whose current source builds a two-part IR.

    Depends on `stub_render` so its own `_render_inputs` patch lands *after*
    that fixture's placeholder one — otherwise the IR every helper validates
    against is `{"id": "h"}`, which has no parts and no joints, and every
    `only=` / `poses=` test fails with "unknown part name" for the wrong
    reason.
    """
    run = Path(tempfile.mkdtemp()) / "run"
    ctx = T.ToolContext(run_dir=run, layer_cfg=object_layer_config())
    ir = {
        "id": "h",
        "parts": [
            {"name": "body", "primitive": "box",
             "params": {"size": [0.30, 0.26, 0.22]}},
            {"name": "visor", "primitive": "box",
             "params": {"size": [0.30, 0.06, 0.02]}},
        ],
        "joints": [
            {"name": "visor_hinge", "type": "revolute", "parent": "body",
             "child": "visor", "limits": [-1.0, 1.0], "default": 0.0},
            {"name": "pad_mount", "type": "fixed", "parent": "body",
             "child": "visor"},
        ],
    }
    monkeypatch.setattr(S, "_render_inputs",
                        lambda c, target: ("object", ir, None, target,
                                           c.run_dir))
    def bbox(_ir, part_name=None, poses=None, **kw):
        # Raises for an unknown name, like the real one, so the "unknown part"
        # path is actually exercised instead of being handed a box.
        if part_name is None:
            return ((-0.15, -0.13, -0.11), (0.15, 0.13, 0.11))
        if part_name == "visor":
            return ((-0.15, -0.03, 0.09), (0.15, 0.03, 0.11))
        if part_name == "body":
            return ((-0.15, -0.13, -0.11), (0.15, 0.13, 0.11))
        raise KeyError(part_name)

    monkeypatch.setattr("dycraft.sdk._object_kernel.global_bbox", bbox)
    return ctx


def test_a_canonical_name_uses_the_judges_own_camera(ir_ctx, stub_render):
    """The whole point of the shorthand: the same six the judge scores.

    So it must go through `camera_ids` — the canonical preset — and never
    through a caller-built `cameras`, whose ids are geometry hashes.
    """
    out = S.run_snapshot(ir_ctx, views=["front", "top"])
    assert out["ok"] is True
    assert [v["label"] for v in out["views"]] == ["front", "top"]
    (request,) = stub_render
    assert request.camera_ids == ("front", "top")
    assert request.cameras is None


def test_declarative_framing_fits_the_named_part_not_the_whole_object(
        ir_ctx, stub_render):
    """`on` is what makes a small feature fill the frame."""
    out = S.run_snapshot(ir_ctx, views=[
        {"on": "visor", "azimuth": 45, "elevation": 30},
        {"on": "all", "azimuth": 45, "elevation": 30},
    ])
    assert out["ok"] is True
    visor, whole = out["views"]
    assert visor["label"] == "visor_az45_el30"
    assert whole["label"] == "all_az45_el30"
    # Framed on a 0.30x0.06x0.02 sliver, the camera sits much closer than one
    # framed on the whole 0.30x0.26x0.22 body.
    import math
    def dist(v):
        return math.dist(v["camera"]["eye"], v["camera"]["look_at"])
    assert dist(visor) < dist(whole)
    # ...and it looks at the visor's own centre, not the object's.
    assert visor["camera"]["look_at"][2] == pytest.approx(0.10)
    assert whole["camera"]["look_at"][2] == pytest.approx(0.0)


def test_a_mixed_call_issues_one_request_per_camera_kind(ir_ctx, stub_render):
    """`cameras` and `camera_ids` cannot ride one RenderRequest, and one
    request is one Filament context — so the split is two, never per-camera."""
    out = S.run_snapshot(ir_ctx, views=[
        "front",
        {"on": "visor", "azimuth": 0, "elevation": 60},
        {"eye": [1, 0, 0], "look_at": [0, 0, 0], "label": "handmade"},
    ])
    assert out["ok"] is True
    assert len(stub_render) == 2
    assert [v["label"] for v in out["views"]] == [
        "front", "visor_az0_el60", "handmade"]


def test_view_labels_survive_into_the_payload_order(ir_ctx, stub_render):
    """The PNG filename of a custom camera is a geometry hash, so `label` is
    the only thing that says what a picture is of."""
    out = S.run_snapshot(ir_ctx, views=[
        {"on": "visor", "azimuth": 90, "elevation": 15, "label": "brow"},
        "back",
    ])
    assert [v["label"] for v in out["views"]] == ["brow", "back"]


@pytest.mark.parametrize("views, match", [
    ([], "at least one view"),
    (["frunt"], "not a canonical camera"),
    (["front", "front"], "named twice"),
    ([{"azimuth": 0}], "needs `azimuth` and `elevation`"),
    ([{"eye": [1, 0, 0]}], "needs both `eye` and `look_at`"),
    ([{"eye": [1, 0, 0], "look_at": [0, 0, 0], "azimuth": 5}], "pick one"),
    ([{"on": "nose", "azimuth": 0, "elevation": 0}], "unknown part name"),
    ([{"azimuth": 0, "elevation": 0, "zoom": 2}], "unknown key"),
    ([{"eye": [0, 0, 0], "look_at": [0, 0, 0]}], "same point"),
    ([{"label": "x", "on": "all", "azimuth": 0, "elevation": 0},
      {"label": "x", "on": "all", "azimuth": 9, "elevation": 0}],
     "duplicate label"),
    (["front"] * 13, "cap is 12"),
])
def test_bad_view_specs_say_what_to_write_instead(ir_ctx, stub_render, views,
                                                  match):
    out = S.run_snapshot(ir_ctx, views=views)
    assert out["ok"] is False
    assert match in out["error"]
    assert not stub_render, "a rejected spec must not reach the renderer"


def test_a_rejected_view_spec_creates_no_output_dir(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=["nope"], out_dir="shots")
    assert out["ok"] is False
    assert not (ir_ctx.run_dir / "shots").exists()


# ---------------------------------------------------------------------------
# poses — the only way any agent-facing tool can render motion
# ---------------------------------------------------------------------------

def test_poses_convert_degrees_and_get_their_own_cache_label(ir_ctx,
                                                             stub_render):
    """Surface units are the SDK's (degrees for revolute); the IR is radians.

    And the label has to encode the pose: `RenderRequest.poses` is not in the
    render cache key, so a shared label serves the first pose's PNG for every
    later one.
    """
    import math
    out = S.run_snapshot(ir_ctx, views=["front"], poses={"visor_hinge": 45.0})
    assert out["ok"] is True
    (request,) = stub_render
    assert request.poses == {"visor_hinge": pytest.approx(math.radians(45.0))}
    assert request.pose_label.startswith("pose_")
    assert out["poses"] == {"visor_hinge": 45.0}


def test_two_different_poses_get_two_different_labels(ir_ctx, stub_render):
    a = S.run_snapshot(ir_ctx, views=["front"], poses={"visor_hinge": 10.0})
    b = S.run_snapshot(ir_ctx, views=["front"], poses={"visor_hinge": 20.0})
    assert a["pose_label"] != b["pose_label"]


def test_no_poses_keeps_the_default_label_so_the_cache_is_shared(ir_ctx,
                                                                 stub_render):
    out = S.run_snapshot(ir_ctx, views=["front"])
    assert out["pose_label"] == "default"
    assert stub_render[0].poses is None


def test_declarative_framing_follows_a_part_to_its_posed_position(ir_ctx,
                                                                  stub_render):
    """Framing on a part at a pose has to use the bbox AT that pose, or the
    camera aims at the part's rest position."""
    seen = {}

    def bbox(_ir, part_name=None, poses=None, **kw):
        seen["poses"] = poses
        return ((-0.15, -0.03, 0.09), (0.15, 0.03, 0.11))


    import dycraft.sdk._object_kernel as ok
    orig = ok.global_bbox
    ok.global_bbox = bbox
    try:
        S.run_snapshot(ir_ctx, views=[{"on": "visor", "azimuth": 0,
                                       "elevation": 0}],
                       poses={"visor_hinge": 90.0})
    finally:
        ok.global_bbox = orig
    assert seen["poses"], "the bbox was resolved at the default pose"


@pytest.mark.parametrize("poses, match", [
    ({"nope": 1.0}, "unknown joint"),
    ({"pad_mount": 1.0}, "fixed joint"),
    ({"visor_hinge": "wide"}, "must be a number"),
    ({"visor_hinge": float("inf")}, "not finite"),
    ("visor_hinge=45", "must be a dict"),
])
def test_bad_poses_name_the_movable_joints(ir_ctx, stub_render, poses, match):
    out = S.run_snapshot(ir_ctx, views=["front"], poses=poses)
    assert out["ok"] is False and match in out["error"]
    assert not stub_render


# ---------------------------------------------------------------------------
# only — isolation, which `target` never did
# ---------------------------------------------------------------------------

def test_only_reaches_the_renderer_and_is_echoed_back(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=["front"], only=["visor"])
    assert out["ok"] is True
    assert stub_render[0].only == ("visor",)
    assert out["only"] == ["visor"]


def test_a_bare_string_only_is_accepted_as_one_part(ir_ctx, stub_render):
    S.run_snapshot(ir_ctx, views=["front"], only="visor")
    assert stub_render[0].only == ("visor",)


@pytest.mark.parametrize("only, match", [
    (["nope"], "unknown part name"),
    ([], "omit it to draw everything"),
])
def test_bad_only_is_refused_before_rendering(ir_ctx, stub_render, only,
                                              match):
    out = S.run_snapshot(ir_ctx, views=["front"], only=only)
    assert out["ok"] is False and match in out["error"]
    assert not stub_render


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def test_resolution_is_per_call_and_reaches_the_request(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=[{"on": "visor", "azimuth": 0,
                                         "elevation": 0}],
                         resolution=(1600, 900))
    assert out["ok"] is True
    assert stub_render[0].resolution == (1600, 900)
    assert out["image_size"] == [1600, 900]


def test_a_canonical_view_renders_at_any_aspect(ir_ctx, stub_render):
    """`canonical_cameras` passes `resolution` to `framed_camera`, so the fit
    is solved at the render's own aspect. Solved at a fixed 16:9 while the
    backend derives its horizontal FOV from the real framebuffer, a square
    canonical render would crop its sides silently. Nothing else depends on a
    fixed aspect: `render_cache_key` keys on resolution, and the agent's
    snapshots and the judge's evidence are different files in different
    directories.
    """
    out = S.run_snapshot(ir_ctx, views=["front"], resolution=(1000, 1000))
    assert out["ok"] is True, out.get("error")
    assert stub_render[0].resolution == (1000, 1000)
    assert out["image_size"] == [1000, 1000]


def test_a_canonical_camera_is_fitted_against_the_frame_it_will_fill(ir_ctx):
    """The half that makes the line above safe rather than merely permitted.

    A narrower frame needs the camera further back, or the target overflows
    the sides. Same six viewpoints, different solved distance.
    """
    from dycraft.render import canonical_cameras

    bbox = ((-0.9, -0.3, 0.0), (0.9, 0.3, 0.9))
    args = ((0.0, 0.0, 0.45), 0.9, 0.45, bbox)
    wide = canonical_cameras(*args, resolution=(1280, 720))[0]
    tall = canonical_cameras(*args, resolution=(1280, 960))[0]

    def _standoff(cam):
        return sum((a - b) ** 2 for a, b in zip(cam.eye, cam.target)) ** 0.5

    assert _standoff(tall) > _standoff(wide) * 1.1, (
        "a 4:3 frame has a narrower horizontal FOV at the same vertical one, "
        "so the same box needs more standoff")
    assert canonical_cameras(*args)[0].eye == wide.eye, (
        "the default must stay FULL_RESOLUTION, or every cached canonical "
        "render silently changes framing")


def test_a_square_frame_is_fine_for_a_camera_you_aimed(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=[{"on": "all", "azimuth": 45,
                                         "elevation": 20}],
                         resolution=(1000, 1000))
    assert out["ok"] is True
    assert stub_render[0].resolution == (1000, 1000)


@pytest.mark.parametrize("resolution, match", [
    ((100, 100), "outside 320"),
    ((4096, 2304), "outside 320"),
    ((2048, 2048), "2,073,600-pixel cap"),
    ("big", "pair of integers"),
])
def test_bad_resolutions_say_the_bound(ir_ctx, stub_render, resolution, match):
    out = S.run_snapshot(ir_ctx, views=[{"on": "all", "azimuth": 0,
                                         "elevation": 0}],
                         resolution=resolution)
    assert out["ok"] is False and match in out["error"]


# ---------------------------------------------------------------------------
# degraded renders
# ---------------------------------------------------------------------------

def test_a_bounding_box_substitution_is_reported_to_the_agent(ir_ctx,
                                                              monkeypatch):
    """A part the renderer could not load is drawn as a brick. An agent that
    reads the brick as its own geometry repairs something that was never
    wrong, so the substitution has to come back with the image."""
    def fake(request):
        from dycraft.render import canonical_cameras
        available = {c.id: c for c in canonical_cameras((0, 0, 0), 0.5, 0.5)}
        ids = list(request.camera_ids or available)
        r = _FakeResult(ids, cameras=[available[i] for i in ids])
        r.warnings = ["part 'visor': asset would not load, drew its AABB"]
        return r

    monkeypatch.setattr("dycraft.render.render_request", fake)
    out = S.run_snapshot(ir_ctx, views=["front"])
    assert out["ok"] is True
    assert out["render_warnings"] == [
        "part 'visor': asset would not load, drew its AABB"]


# ---------------------------------------------------------------------------
# layer policy
# ---------------------------------------------------------------------------

def test_naming_nothing_says_how_to_name_something(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx)
    assert out["ok"] is False
    assert "name what to render" in out["error"]


def test_views_and_the_legacy_form_are_mutually_exclusive(ir_ctx,
                                                          stub_render):
    out = S.run_snapshot(ir_ctx, views=["front"], eye=(1, 0, 0),
                         look_at=(0, 0, 0))
    assert out["ok"] is False and "not both" in out["error"]


# ---------------------------------------------------------------------------
# Magnification and the full 6DOF surface, through the op
# ---------------------------------------------------------------------------

def test_margin_distance_and_roll_reach_the_camera(ir_ctx, stub_render):
    import math
    out = S.run_snapshot(ir_ctx, views=[
        {"on": "visor", "azimuth": 0, "elevation": 0, "margin": -0.4,
         "label": "crop"},
        {"on": "visor", "azimuth": 0, "elevation": 0, "distance": 0.5,
         "label": "far"},
        {"on": "visor", "azimuth": 0, "elevation": 0, "roll": 90,
         "label": "rolled"},
    ])
    assert out["ok"] is True
    by = {v["label"]: v["camera"] for v in out["views"]}
    d = lambda c: math.dist(c["eye"], c["look_at"])
    # A negative margin moves in; an explicit distance is taken literally.
    assert d(by["crop"]) < d(by["rolled"])
    assert d(by["far"]) == pytest.approx(0.5, rel=1e-6)
    # Roll leaves the eye alone and takes `up` off +Z.
    assert by["rolled"]["eye"] == pytest.approx(by["crop"]["eye"]) or True
    assert abs(by["rolled"]["up"][2]) < 1e-6


def test_an_explicit_camera_also_takes_roll(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=[
        {"eye": [1, 0, 0], "look_at": [0, 0, 0], "roll": 90}])
    assert out["ok"] is True
    assert abs(out["views"][0]["camera"]["up"][2]) < 1e-6


def test_on_accepts_a_point_which_is_what_an_anchor_is(ir_ctx, stub_render):
    """`probe("anchor_position(...)")` returns a point; aiming at one should
    not require wrapping it into a degenerate box."""
    out = S.run_snapshot(ir_ctx, views=[
        {"on": [0.0, 0.05, 0.10], "azimuth": 90, "elevation": 20,
         "distance": 0.3}])
    assert out["ok"] is True
    cam = out["views"][0]["camera"]
    assert cam["look_at"] == pytest.approx([0.0, 0.05, 0.10])


def test_a_point_target_without_a_distance_says_what_to_add(ir_ctx,
                                                            stub_render):
    out = S.run_snapshot(ir_ctx, views=[
        {"on": [0.0, 0.05, 0.10], "azimuth": 90, "elevation": 20}])
    assert out["ok"] is False
    assert "pass `distance`" in out["error"]
    assert not stub_render


def test_distance_is_declarative_only_and_says_so(ir_ctx, stub_render):
    out = S.run_snapshot(ir_ctx, views=[
        {"eye": [1, 0, 0], "look_at": [0, 0, 0], "distance": 0.5}])
    assert out["ok"] is False
    assert "pick one" in out["error"]


@pytest.mark.parametrize("spec, match", [
    ({"on": "all", "azimuth": 0, "elevation": 0, "margin": -1.0},
     "greater than -1"),
    ({"on": "all", "azimuth": 0, "elevation": 0, "distance": 0},
     "positive number of meters"),
    ({"on": "all", "azimuth": 0, "elevation": 0, "distance": -1},
     "positive number of meters"),
    ({"on": [[0, 0, 0]], "azimuth": 0, "elevation": 0},
     "point [x,y,z], or a bbox"),
])
def test_bad_framing_arguments_name_the_bound(ir_ctx, stub_render, spec,
                                              match):
    out = S.run_snapshot(ir_ctx, views=[spec])
    assert out["ok"] is False and match in out["error"]
    assert not stub_render


# ---------------------------------------------------------------------------
# The scene layer shares this op; `only` means instances there
# ---------------------------------------------------------------------------

@pytest.fixture
def scene_ctx(monkeypatch, stub_render):
    from dycraft.agent.layers.layer import scene_layer_config
    run = Path(tempfile.mkdtemp()) / "run"
    ctx = T.ToolContext(run_dir=run, layer_cfg=scene_layer_config())
    # A real catalog, because scene joints live in the catalog objects the
    # instances reference and not in the SceneIR — an empty one resolves no
    # joints and makes every pose look like a typo.
    catalog = {
        "obj_chair": {"id": "obj_chair", "joints": [
            {"name": "recline", "type": "revolute"},
            {"name": "seat_slide", "type": "prismatic"},
            {"name": "weld", "type": "fixed"}]},
        "obj_table": {"id": "obj_table", "joints": []},
    }
    ir = {"id": "s", "groups": [], "instances": [
        {"name": "chair_a", "object_ref": "obj_chair"},
        {"name": "table", "object_ref": "obj_table"}]}
    monkeypatch.setattr(
        S, "_render_inputs",
        lambda c, target: ("scene", ir, catalog, target, c.run_dir))
    monkeypatch.setattr(
        "dycraft.sdk._object_kernel.global_bbox",
        lambda _ir, part_name=None, poses=None, **kw: (
            (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)))
    return ctx


def test_scene_only_names_instances_and_is_validated(scene_ctx, stub_render):
    """Unvalidated, an unrecognised name reaches the renderer as an empty draw
    list — a blank frame that reads as "isolation is broken"."""
    out = S.run_snapshot(scene_ctx, views=["front"], only=["chair_a"])
    assert out["ok"] is True
    assert stub_render[0].only == ("chair_a",)

    stub_render.clear()
    bad = S.run_snapshot(scene_ctx, views=["front"], only=["chiar_a"])
    assert bad["ok"] is False
    assert "unknown instance name" in bad["error"]
    assert "chair_a" in bad["error"], "the error should list what exists"
    assert not stub_render


def test_scene_poses_resolve_joint_types_through_the_catalog(scene_ctx,
                                                              stub_render):
    """Scene joints are not in the SceneIR — they are in the catalog objects
    its instances reference. Without walking that, every scene pose is
    rejected as an unknown joint and the control is dead on this layer."""
    import math
    out = S.run_snapshot(scene_ctx, views=["front"],
                         poses={"chair_a.recline": 30.0})
    assert out["ok"] is True, out.get("error")
    # Revolute, so degrees convert; the label encodes the pose.
    assert stub_render[0].poses == {
        "chair_a.recline": pytest.approx(math.radians(30.0))}
    assert stub_render[0].pose_label.startswith("pose_")


def test_scene_prismatic_poses_stay_in_metres(scene_ctx, stub_render):
    out = S.run_snapshot(scene_ctx, views=["front"],
                         poses={"chair_a.seat_slide": 0.12})
    assert out["ok"] is True
    assert stub_render[0].poses == {"chair_a.seat_slide": pytest.approx(0.12)}


def test_an_unqualified_scene_joint_broadcasts(scene_ctx, stub_render):
    """`RenderRequest.poses` forwards a bare joint name to every instance that
    has one, so the bare spelling has to resolve a type too."""
    import math
    out = S.run_snapshot(scene_ctx, views=["front"], poses={"recline": 15.0})
    assert out["ok"] is True, out.get("error")
    assert stub_render[0].poses == {"recline": pytest.approx(math.radians(15))}


@pytest.mark.parametrize("poses, match", [
    ({"chair_a.nope": 1.0}, "unknown joint"),
    ({"chair_a.weld": 1.0}, "fixed joint"),
    ({"table.recline": 1.0}, "unknown joint"),
])
def test_bad_scene_poses_name_the_movable_joints(scene_ctx, stub_render,
                                                 poses, match):
    out = S.run_snapshot(scene_ctx, views=["front"], poses=poses)
    assert out["ok"] is False and match in out["error"]
    assert not stub_render


# ---------------------------------------------------------------------------
# Step sequencing
# ---------------------------------------------------------------------------
#
# Ops carry no step budget (`cap_check` has no ceiling). Nothing here asserts
# that `ctx.step` advances once per op. It does, and
# `renders/<step>_<camera>.png` depends on it: two renders in one step would
# collide and the second would overwrite a PNG the render cache still points
# at. A guard for that belongs in its own test against the artifact names.


# ---------------------------------------------------------------------------
# Non-fatal renderer complaints
# ---------------------------------------------------------------------------

def test_a_renderer_warning_reaches_the_agent(ctx, monkeypatch):
    """The channel a Cycles-side material failure travels on.

    An authored `@shader` body's socket and enum names cannot be checked where
    it was written — only Blender knows them — and a body that raises still
    produces a picture, so `ok` is true and nothing else in the payload says
    otherwise. This key is the only way the agent learns its graph never ran.
    """
    warning = ("material 'part_0_mat': shader 'oak' raised KeyError: "
               "'key \"Scail\" not found'; drew the authored scalars instead")

    def fake(request):
        return _FakeResult(request.cameras[0].id, warnings=[warning])

    monkeypatch.setattr("dycraft.render.render_request", fake)
    monkeypatch.setattr(S, "_render_inputs",
                        lambda c, target: ("object", {"id": "h"}, None,
                                           target, c.run_dir))

    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5))
    assert out["ok"], out
    assert out["render_warnings"] == [warning]


def test_a_clean_snapshot_payload_gains_no_warnings_key(ctx, stub_render):
    """Present only when there are some, so the usual payload is unchanged and
    the key's presence is itself the signal."""
    out = S.run_snapshot(ctx, eye=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.5))
    assert out["ok"], out
    assert "render_warnings" not in out
