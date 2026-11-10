"""Caller-supplied cameras: `look_at_camera` + `RenderRequest.cameras`.

All GL-free. Several things downstream key on a camera's *id*: the render
cache keys on the id alone, the id becomes the PNG filename, `ordered_paths()`
filters to the canonical set, and the room cutaway answers canonical ids from a
table. These tests pin each of those so a caller-supplied camera cannot produce
a silently wrong image.
"""

import math
from pathlib import Path

import pytest

from dycraft.render import (CANONICAL_CAMERA_IDS, CameraSpec, RenderRequest,
                            RenderResult, canonical_cameras, framed_camera,
                            look_at_camera, render_cache_key)
from dycraft.render.open3d_renderer import _floor_only_for


def _request(**kw) -> RenderRequest:
    base = dict(layer="object", ir={"id": "h"}, catalog=None, target="all",
                out_dir="/tmp", step=0)
    base.update(kw)
    return RenderRequest(**base)


def _resolve(request):
    from dycraft.render import _cameras_for_request
    return _cameras_for_request(request, (0.0, 0.0, 0.0), 1.0, 1.0)


# ---------------------------------------------------------------------------
# Camera resolution
# ---------------------------------------------------------------------------

def test_supplied_cameras_replace_the_preset_and_ignore_target():
    """`target` normally narrows to one canonical camera when it names one.
    An explicit camera has to win over that, or a caller who framed on an
    entity called "top" silently gets the top preset instead.
    """
    cam = look_at_camera((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert _resolve(_request(cameras=(cam,), target="top")) == [cam]


def test_setting_both_cameras_and_camera_ids_is_an_error():
    """`camera_ids` filters the preset; `cameras` replaces it. A request with
    both cannot be honored as written, so it raises rather than picking one.
    """
    cam = look_at_camera((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="both"):
        _resolve(_request(cameras=(cam,), camera_ids=("front",)))


def test_the_preset_is_untouched_when_no_cameras_are_given():
    ids = [c.id for c in _resolve(_request())]
    assert ids == list(CANONICAL_CAMERA_IDS)
    assert ids == ["front", "right", "back", "left", "top", "bottom"]


# ---------------------------------------------------------------------------
# look_at_camera
# ---------------------------------------------------------------------------

def test_the_id_is_a_hash_of_the_camera_geometry():
    """The load-bearing property.

    `render_cache_key` takes the camera *id* and none of the geometry it stands
    for, and the id is also the PNG filename. A constructor handing out a fixed
    id like "custom" would be filesystem-safe and still serve one camera's PNG
    for another's request. Two cameras differing only in `eye` must therefore
    produce different keys for the same IR, target and pose label.
    """
    a = look_at_camera((1.0, 0.0, 0.5), (0.0, 0.0, 0.5))
    b = look_at_camera((1.1, 0.0, 0.5), (0.0, 0.0, 0.5))
    assert a.id != b.id
    key = dict(ir_hash="h", target="all", pose_label="default")
    assert render_cache_key(camera_id=a.id, **key) != \
        render_cache_key(camera_id=b.id, **key)
    # ...and an identical camera must still hit, or nothing ever caches.
    again = look_at_camera((1.0, 0.0, 0.5), (0.0, 0.0, 0.5))
    assert again.id == a.id


def test_the_id_is_filesystem_safe_and_never_a_canonical_id():
    """It becomes `<step>_<id>.png`, and a collision with a canonical id would
    let a custom render be served into a canonical view's slot."""
    cam = look_at_camera((1.0, -2.0, 0.5), (0.0, 0.0, 0.5), fov_deg=42.5)
    assert cam.id not in CANONICAL_CAMERA_IDS
    assert cam.id.startswith("cam_")
    assert cam.id.isascii() and cam.id.replace("_", "").isalnum()


def test_float_noise_does_not_mint_a_new_id():
    """Coordinates round-trip through JSON on the way from the agent, so an id
    that changed on the last bits would defeat the cache for free."""
    a = look_at_camera((1.0, 0.0, 0.5), (0.0, 0.0, 0.5))
    b = look_at_camera((1.0 + 1e-13, 0.0, 0.5), (0.0, 0.0, 0.5))
    assert a.id == b.id


@pytest.mark.parametrize("kwargs,match", [
    (dict(eye=(0, 0, 1), look_at=(0, 0, 1)), "same point"),
    (dict(eye=(float("nan"), 0, 0), look_at=(0, 0, 0)), "non-finite"),
    (dict(eye=(float("inf"), 0, 0), look_at=(0, 0, 0)), "non-finite"),
    (dict(eye=(1, 0), look_at=(0, 0, 0)), "3 components"),
    (dict(eye="abc", look_at=(0, 0, 0)), "sequence of 3"),
    (dict(eye=None, look_at=(0, 0, 0)), "sequence of 3"),
    (dict(eye=(1, 0, 0), look_at=(0, 0, 0), up=(0, 0, 0)), "zero length"),
    (dict(eye=(0, 0, 3), look_at=(0, 0, 0), up=(0, 0, 1)), "view direction"),
    (dict(eye=(1, 0, 0), look_at=(0, 0, 0), fov_deg=0.5), "in \\(1, 179\\)"),
    (dict(eye=(1, 0, 0), look_at=(0, 0, 0), fov_deg=180), "in \\(1, 179\\)"),
])
def test_bad_cameras_are_refused_with_a_reason(kwargs, match):
    with pytest.raises(ValueError, match=match):
        look_at_camera(**kwargs)


def test_a_near_parallel_up_is_refused_before_it_degenerates():
    """Filament's basis construction degenerates smoothly rather than at a
    point, so an `up` half a degree off the view direction gives an arbitrary
    roll rather than an error. The floor is what makes that loud."""
    eye = (0.0, 0.0, 3.0)
    off = 3.0 * math.tan(math.radians(0.5))
    with pytest.raises(ValueError, match=r"0\.5\d?°|0\.4\d°"):
        look_at_camera(eye, (0.0, 0.0, 0.0), up=(off, 0.0, 3.0))


def test_up_defaults_and_reports_which_way_is_image_up():
    """`up=None` has to work for the most obvious request an agent will make —
    a top-down view — and the fallbacks match the canonical top/bottom cameras
    so image orientation stays consistent with them."""
    assert look_at_camera((1, 0, 0), (0, 0, 0)).up == (0.0, 0.0, 1.0)
    assert look_at_camera((0, 0, 3), (0, 0, 0)).up == (0.0, -1.0, 0.0)
    assert look_at_camera((0, 0, -3), (0, 0, 0)).up == (0.0, 1.0, 0.0)
    cams = {c.id: c for c in canonical_cameras((0, 0, 0), 1.0, 1.0)}
    assert look_at_camera((0, 0, 3), (0, 0, 0)).up == cams["top"].up
    assert look_at_camera((0, 0, -3), (0, 0, 0)).up == cams["bottom"].up


# ---------------------------------------------------------------------------
# The three downstream assumptions
# ---------------------------------------------------------------------------

def test_ordered_paths_drops_a_custom_camera_on_purpose():
    """Asserted deliberately, not tolerated.

    Every caller of `ordered_paths()` means "the canonical set, in canonical
    order". A custom id vanishing from it is correct — which is why
    `run_snapshot` reads `render_paths` directly, and why a future refactor must
    not "fix" it to use the tidier-looking accessor.
    """
    cam = look_at_camera((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = RenderResult(render_paths={cam.id: "a.png", "front": "b.png"})
    assert result.ordered_paths() == ["b.png"]


def test_the_wall_cutaway_is_unchanged_for_every_canonical_camera():
    """The canonical branch of `_floor_only_for` is byte-compatibility.

    A purely geometric rule would also fire for `bottom`, whose horizontal
    offset is exactly zero and which gets walls drawn — changing that moves the
    bytes of a canonical view.
    """
    cams = {c.id: c for c in canonical_cameras((0, 0, 0.5), 1.0, 0.5)}
    assert _floor_only_for(cams["top"]) is True
    for cid in ("front", "back", "left", "right", "bottom"):
        assert _floor_only_for(cams[cid]) is False, cid


def test_a_custom_near_vertical_camera_gets_no_walls():
    """It cannot be named "top" — the id is a hash — so the id test alone would
    stand a wall between the camera and the scene."""
    assert _floor_only_for(look_at_camera((0, 0, 3), (0, 0, 0))) is True
    assert _floor_only_for(look_at_camera((0.1, 0.1, 3), (0, 0, 0))) is True
    # ...and an oblique or level one still gets its far wall.
    assert _floor_only_for(look_at_camera((1.4, 1.4, 1.9), (0, 0, 0.5))) is False
    assert _floor_only_for(look_at_camera((2, 0, 0.5), (0, 0, 0.5))) is False


def test_camera_spec_still_takes_a_hand_built_camera():
    """`look_at_camera` is the recommended constructor, not the only one — the
    dataclass stays open so a test or a one-off script can pin an exact eye."""
    cam = CameraSpec(id="probe0", eye=(1, 0, 0), target=(0, 0, 0), up=(0, 0, 1))
    assert _resolve(_request(cameras=(cam,))) == [cam]


# ---------------------------------------------------------------------------
# framed_camera: fit the projected extent, and clip planes that survive it
# ---------------------------------------------------------------------------

def _frustum_span(bbox, cam, resolution=(1280, 720)):
    """Fraction of the frame the bbox spans on each axis, and containment.

    Reimplements the projection rather than reading it off a render, so the
    fit can be checked over hundreds of directions without a GL context.
    """
    import math
    (x0, y0, z0), (x1, y1, z1) = bbox
    width, height = resolution
    tan_v = math.tan(math.radians(cam.fov_deg / 2.0))
    tan_h = (width / height) * tan_v

    def _sub(a, b):
        return [a[i] - b[i] for i in range(3)]

    def _cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    def _norm(v):
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return [c / n for c in v]

    fwd = _norm(_sub(cam.target, cam.eye))
    right = _norm(_cross(fwd, list(cam.up)))
    up = _norm(_cross(right, fwd))
    us, vs, inside = [], [], True
    for cx in (x0, x1):
        for cy in (y0, y1):
            for cz in (z0, z1):
                d = _sub((cx, cy, cz), cam.eye)
                depth = sum(a * b for a, b in zip(d, fwd))
                if depth <= 1e-9:
                    inside = False
                    continue
                u = sum(a * b for a, b in zip(d, right)) / (depth * tan_h)
                w = sum(a * b for a, b in zip(d, up)) / (depth * tan_v)
                us.append(u)
                vs.append(w)
                if abs(u) > 1.0 + 1e-6 or abs(w) > 1.0 + 1e-6:
                    inside = False
    return (max(us) - min(us)) / 2.0, (max(vs) - min(vs)) / 2.0, inside


_HELMET = ((-0.15, -0.13, -0.11), (0.15, 0.13, 0.11))


@pytest.mark.parametrize("azimuth", range(0, 360, 15))
@pytest.mark.parametrize("elevation", [-90, -89, -60, -15, 0, 15, 60, 89, 90])
def test_a_framed_camera_contains_the_box_from_every_direction(azimuth,
                                                               elevation):
    """The fit has to hold at the poles too.

    The box must be fitted in the basis `look_at_camera` renders it in. Two
    up-vector thresholds near the pole (`|dz| > 0.999`, elevation ~87.4,
    against 1 degree from the view axis, elevation 89) would fit it in one
    basis and render it in another, and between them it overflows the frame.
    """
    from dycraft.render import framed_camera
    cam = framed_camera(_HELMET, azimuth_deg=azimuth,
                        elevation_deg=elevation)
    span_u, span_v, inside = _frustum_span(_HELMET, cam)
    assert inside, f"box escapes the frame at az={azimuth} el={elevation}"
    assert max(span_u, span_v) > 0.65, "fitted, but far too loose"


def _v8_front_camera(center, extent_xy, extent_z):
    """A front camera from a center-depth formula, the baseline for the exact fit.

    `max(rxy, rz)` floored at 0.1 m, fitted at the bbox's Center depth with a
    `1.25` margin, plus an additive 0.1 m standoff. It lives here rather than
    in the source because nothing renders with it.
    """
    rxy, rz = max(extent_xy, 0.1), max(extent_z, 0.1)
    r = max(rxy, rz) * 1.25 / math.tan(math.radians(30.0)) + 0.1
    elev = math.radians(15.0)
    return CameraSpec(
        id="front",
        eye=(center[0], center[1] + r * math.cos(elev),
             center[2] + r * math.sin(elev)),
        target=tuple(center), up=(0.0, 0.0, 1.0))


def test_the_exact_fit_beats_a_centre_depth_formula():
    """Both terms of the center-depth formula cost frame area, worst on small
    objects.

    A fixed `+ 0.1 m` is a fixed cost against a variable size, and the 0.1 m
    extent floor frames anything smaller as if it were 200 mm across.
    """
    knob = ((-0.025, -0.025, -0.015), (0.025, 0.025, 0.015))
    center, exy, ez = _metrics(knob)
    old = _frustum_span(knob, _v8_front_camera(center, exy, ez))
    new = _frustum_span(knob, framed_camera(knob, azimuth_deg=90,
                                            elevation_deg=15))
    assert new[2], "the exact fit must contain the box"
    assert (new[0] * new[1]) > 20 * (old[0] * old[1])


def test_the_canonical_preset_is_now_that_same_fit():
    """`canonical_cameras` delegates rather than carrying a second formula.

    Two framing implementations is how they drift; this pins that there is one.
    The canonical camera should land within a hair of a bespoke `framed_camera`
    call from the same direction with the same margin.
    """
    for name in ("cube", "tiny", "long_narrow", "huge"):
        bbox = _SHAPES[name]
        center, exy, ez = _metrics(bbox)
        canonical = {c.id: c for c in canonical_cameras(center, exy, ez, bbox)}
        bespoke = framed_camera(bbox, azimuth_deg=90, elevation_deg=15,
                                margin=0.06)
        assert canonical["front"].eye == pytest.approx(bespoke.eye, rel=1e-9)
        assert canonical["front"].near == pytest.approx(bespoke.near, rel=1e-9)


def test_margin_loosens_the_fit_monotonically():
    from dycraft.render import framed_camera
    spans = []
    for margin in (0.0, 0.05, 0.25):
        cam = framed_camera(_HELMET, azimuth_deg=90, elevation_deg=15,
                            margin=margin)
        spans.append(max(_frustum_span(_HELMET, cam)[:2]))
    assert spans[0] > spans[1] > spans[2]


def test_the_frame_aspect_is_honoured():
    """`resolution` supplies the aspect. Fit a landscape box against a portrait
    frame without it and the sides fall off."""
    from dycraft.render import framed_camera
    portrait = (720, 1280)
    wide = ((-0.80, -0.10, -0.10), (0.80, 0.10, 0.10))
    cam = framed_camera(wide, azimuth_deg=90, elevation_deg=0,
                        resolution=portrait)
    assert _frustum_span(wide, cam, resolution=portrait)[2]


def test_a_close_fit_sets_clip_planes_instead_of_inheriting_the_scenes():
    """Open3D derives near/far from the geometry's bounding box when they are
    not passed, which puts the near plane in front of a tight close-up and
    returns an EMPTY frame."""
    from dycraft.render import canonical_cameras, framed_camera
    dial = ((-0.022, -0.140, -0.042), (0.022, -0.130, 0.002))
    cam = framed_camera(dial, azimuth_deg=-90, elevation_deg=10)
    import math
    eye_to_near = min(
        math.dist(cam.eye, (cx, cy, cz))
        for cx in (dial[0][0], dial[1][0])
        for cy in (dial[0][1], dial[1][1])
        for cz in (dial[0][2], dial[1][2]))
    assert 0.0 < cam.near < eye_to_near, "near plane would clip the subject"
    assert cam.far > math.dist(cam.eye, (0.0, 0.0, 0.0))
    # The canonical preset sets them too: it has no additive standoff, so
    # nothing else clears the derived near plane. See
    # `test_the_canonical_preset_carries_its_own_clip_planes`.
    for c in canonical_cameras((0, 0, 0), 0.15, 0.11):
        assert c.near > 0.0 and c.far > c.near


def test_clip_planes_enter_the_camera_id():
    """`camera_id` is the whole camera identity in the render cache key, so a
    field that changes the picture and not the id aliases two renders."""
    from dycraft.render import look_at_camera
    base = look_at_camera((1, 0, 0), (0, 0, 0))
    clipped = look_at_camera((1, 0, 0), (0, 0, 0), near=0.2, far=5.0)
    assert base.id != clipped.id
    # An unset clip plane must not perturb the id, or every PNG cached for a
    # camera without clip planes is orphaned.
    assert look_at_camera((1, 0, 0), (0, 0, 0), near=-1.0, far=-1.0).id == base.id


# ---------------------------------------------------------------------------
# Scale invariance, and the six degrees of freedom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
def test_the_fit_is_scale_invariant(scale):
    """The same shape at any size must frame identically.

    This is the property that catches an absolute length hiding in the solve,
    such as an additive standoff or a millimeter floor on the distance, which
    is larger than the entire fit for anything sub-millimeter.
    """
    from dycraft.render import framed_camera
    unit = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    scaled = tuple(tuple(v * scale for v in corner) for corner in unit)
    ref = _frustum_span(unit, framed_camera(
        unit, azimuth_deg=35, elevation_deg=25))
    got = _frustum_span(scaled, framed_camera(
        scaled, azimuth_deg=35, elevation_deg=25))
    assert got[2], "the box escaped the frame at this scale"
    assert got[0] == pytest.approx(ref[0], rel=1e-3)
    assert got[1] == pytest.approx(ref[1], rel=1e-3)


@pytest.mark.parametrize("scale", [1e-4, 1e-2, 1.0, 10.0])
def test_the_clip_planes_are_scale_invariant_too(scale):
    """Near/far derived from the fit, not from absolute lengths — otherwise a
    tight close-up on a small part gets a frustum that cannot hold it."""
    from dycraft.render import framed_camera
    box = ((-0.5 * scale,) * 3, (0.5 * scale,) * 3)
    cam = framed_camera(box, azimuth_deg=35, elevation_deg=25)
    dist = math.dist(cam.eye, cam.target)
    assert 0.0 < cam.near < dist < cam.far
    # The ratio is what governs depth precision, and it must not drift with
    # scale or a small part renders black.
    assert cam.near / dist == pytest.approx(
        framed_camera(((-0.5,) * 3, (0.5,) * 3),
                      azimuth_deg=35, elevation_deg=25).near
        / math.dist(*[framed_camera(((-0.5,) * 3, (0.5,) * 3),
                                    azimuth_deg=35,
                                    elevation_deg=25).__getattribute__(a)
                      for a in ("eye", "target")]), rel=1e-6)


def test_a_negative_margin_crops_in_and_a_positive_one_pads_out():
    """`margin` is the magnify knob. The default fit contains the whole box,
    which for a squat part means its near corner sets the distance and the
    silhouette cannot fill the frame — measured, a 20 mm boss holds ~52% of
    frame height at the default and ~76% at -0.5."""
    from dycraft.render import framed_camera
    squat = ((-0.010, -0.010, -0.004), (0.010, 0.010, 0.004))
    spans = [max(_frustum_span(squat, framed_camera(
        squat, azimuth_deg=35, elevation_deg=25, margin=m))[:2])
        for m in (0.25, 0.05, 0.0, -0.25, -0.5)]
    assert spans == sorted(spans), "more negative margin must magnify"


def test_the_camera_never_hugs_the_nearest_corner():
    """A standoff proportional to the target keeps the frustum usable.

    Without it the near plane collapses to `0.001 * |t_min|` against a far
    plane four times the far corner, and at `margin=-0.6` the 0.25 mm cube
    rendered an EMPTY frame — a magnify request that silently returned
    nothing.
    """
    from dycraft.render import framed_camera
    for scale in (1e-4, 1e-2, 1.0):
        box = ((-0.5 * scale,) * 3, (0.5 * scale,) * 3)
        cam = framed_camera(box, azimuth_deg=35, elevation_deg=25,
                            margin=-0.8)
        nearest = min(
            math.dist(cam.eye, (cx, cy, cz))
            for cx in (box[0][0], box[1][0])
            for cy in (box[0][1], box[1][1])
            for cz in (box[0][2], box[1][2]))
        assert cam.near < nearest, "near plane would eat the subject"
        assert nearest / scale > 1e-3, "camera is inside the box"


def test_an_explicit_distance_overrides_the_solve():
    from dycraft.render import framed_camera
    box = ((-0.01,) * 3, (0.01,) * 3)
    for d in (0.05, 0.5, 5.0):
        cam = framed_camera(box, azimuth_deg=0, elevation_deg=0, distance=d)
        assert math.dist(cam.eye, cam.target) == pytest.approx(d, rel=1e-9)


def test_a_point_target_needs_a_distance_and_says_so():
    """An anchor is a point. There is no extent to fit, so the caller has to
    say how far away to stand rather than get an invented scale."""
    from dycraft.render import framed_camera
    with pytest.raises(ValueError, match="no extent in frame"):
        framed_camera((0.1, 0.2, 0.3), azimuth_deg=0, elevation_deg=0)
    cam = framed_camera((0.1, 0.2, 0.3), azimuth_deg=0, elevation_deg=0,
                        distance=0.25)
    assert cam.target == (0.1, 0.2, 0.3)
    assert math.dist(cam.eye, cam.target) == pytest.approx(0.25)


def test_roll_is_the_sixth_degree_of_freedom():
    """Aim fixes five; roll is the one no amount of aiming can express.

    Rotating about the view axis must leave the eye and the look direction
    untouched, and must not trip `look_at_camera`'s degenerate-up check, since
    the rotation preserves the angle to that axis.
    """
    from dycraft.render import framed_camera, look_at_camera
    box = ((-0.03, -0.03, -0.35), (0.03, 0.03, 0.35))
    base = framed_camera(box, azimuth_deg=90, elevation_deg=0)
    rolled = framed_camera(box, azimuth_deg=90, elevation_deg=0, roll_deg=90)
    assert rolled.eye == pytest.approx(base.eye)
    assert rolled.target == pytest.approx(base.target)
    assert rolled.id != base.id
    # +Z up becomes horizontal, so a tall thin box renders on its side. It goes
    # from 4.5% x 95% of frame to 53% x 8% through the real renderer.
    assert abs(rolled.up[2]) < 1e-6
    # And roll composes with an explicit up on the direct constructor.
    r90 = look_at_camera((1, 0, 0), (0, 0, 0), (0, 0, 1), roll_deg=90)
    assert abs(r90.up[2]) < 1e-6


@pytest.mark.parametrize("roll", [0, 45, 90, 180, 270, -90])
def test_roll_keeps_the_up_vector_unit_and_usable(roll):
    from dycraft.render import framed_camera
    cam = framed_camera(((-1, -1, -1), (1, 1, 1)), azimuth_deg=35,
                        elevation_deg=25, roll_deg=roll)
    assert math.sqrt(sum(v * v for v in cam.up)) == pytest.approx(1.0)
    fwd = [cam.target[i] - cam.eye[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in fwd))
    cosang = abs(sum(a * b / n for a, b in zip(fwd, cam.up)))
    assert cosang < 0.999, "roll drove up parallel to the view axis"


def test_final_render_reports_a_bounding_box_substitution(tmp_path,
                                                          monkeypatch):
    """`final_render/` is the deliverable. A brick standing in for a part that
    would not load has to be labeled there, or the directory is
    indistinguishable from a run that modeled a box on purpose."""
    import dycraft.render as R

    class _Result:
        render_paths = {"front": tmp_path / "0_front.png"}
        cache_hits = {"front": False}
        cameras: list = []
        target_bbox = ((0, 0, 0), (1, 1, 1))
        warnings = ["part 'lid': asset would not load, drew its AABB"]

        def ordered_paths(self):
            return [tmp_path / "0_front.png"]

    monkeypatch.setattr(R, "render_final_to_run_dir",
                        lambda *a, **k: _Result())
    out = R.finalize_render_result({"id": "h"}, tmp_path, layer="object")
    assert out["ok"] is True
    assert out["render_warnings"] == [
        "part 'lid': asset would not load, drew its AABB"]

    # Absent, not empty, on the clean path — so its presence is the signal.
    class _Clean(_Result):
        warnings: list = []

    monkeypatch.setattr(R, "render_final_to_run_dir",
                        lambda *a, **k: _Clean())
    assert "render_warnings" not in R.finalize_render_result(
        {"id": "h"}, tmp_path, layer="object")


# ---------------------------------------------------------------------------
# canonical_cameras: clip planes in, additive 0.1 m out
# ---------------------------------------------------------------------------

_SHAPES = {
    "cube": ((-0.15, -0.15, -0.15), (0.15, 0.15, 0.15)),
    "tiny": ((-0.03, -0.02, -0.0125), (0.03, 0.02, 0.0125)),
    "flat_wide": ((-0.80, -0.225, -0.02), (0.80, 0.225, 0.02)),
    "tall_thin": ((-0.03, -0.03, -0.35), (0.03, 0.03, 0.35)),
    "long_narrow": ((-1.10, -0.20, -0.30), (1.10, 0.20, 0.30)),
    "huge": ((-2.5, -0.9, -0.7), (2.5, 0.9, 0.7)),
}


def _metrics(bbox):
    lo, hi = bbox
    center = tuple(0.5 * (a + b) for a, b in zip(lo, hi))
    exy = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
    ez = 0.5 * (hi[2] - lo[2])
    return center, exy, ez


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_every_canonical_camera_contains_the_whole_bbox(name):
    """Every canonical view holds every corner of the bbox.

    A fit at the bbox's Center depth ignores that a nearer corner projects
    larger and lets corners leave the frame; the exact fit accounts for every
    corner.
    """
    bbox = _SHAPES[name]
    center, exy, ez = _metrics(bbox)
    for cam in canonical_cameras(center, exy, ez, bbox):
        span_u, span_v, inside = _frustum_span(bbox, cam)
        assert inside, f"{name}: bbox escapes the {cam.id} frame"


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_the_canonical_preset_carries_its_own_clip_planes(name):
    """The pair is load-bearing: planes and no additive standoff, never one
    alone.

    Without explicit planes Open3D derives near/far from the scene bbox, which
    puts the near plane in front of a close subject; with no standoff to clear
    it, the frame renders EMPTY.
    """
    bbox = _SHAPES[name]
    center, exy, ez = _metrics(bbox)
    lo, hi = bbox
    for cam in canonical_cameras(center, exy, ez, bbox):
        assert cam.near > 0.0, f"{cam.id} left its near plane to the backend"
        assert cam.far > cam.near
        depths = []
        fwd = [cam.target[i] - cam.eye[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in fwd)) or 1.0
        fwd = [c / n for c in fwd]
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                for cz in (lo[2], hi[2]):
                    d = [cx - cam.eye[0], cy - cam.eye[1], cz - cam.eye[2]]
                    depths.append(sum(a * b for a, b in zip(d, fwd)))
        assert cam.near < min(depths), f"{cam.id} near plane clips the target"
        assert cam.far > max(depths), f"{cam.id} far plane clips the target"


def test_canonical_framing_holds_no_absolute_length():
    """A fixed `+ 0.1 m` is a fixed cost against a variable size, so it hurts
    small objects most. Scale invariance is the property that keeps it out."""
    unit = ((-0.5,) * 3, (0.5,) * 3)
    ref = None
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        bbox = tuple(tuple(v * scale for v in corner) for corner in unit)
        center, exy, ez = _metrics(bbox)
        cams = {c.id: c for c in canonical_cameras(center, exy, ez, bbox)}
        span = _frustum_span(bbox, cams["front"])[:2]
        if ref is None:
            ref = span
        else:
            assert span[0] == pytest.approx(ref[0], rel=1e-6)
            assert span[1] == pytest.approx(ref[1], rel=1e-6)


def test_the_real_bbox_beats_reconstructing_one_from_two_extents():
    """`target_extent_xy` is a max over BOTH horizontal axes, so rebuilding a
    box from it squares off the footprint and an exact fit then backs off as
    far as the LONG axis demands in every direction. On a 2.2 m long, narrow
    object that cost more frame than the formula it replaced."""
    bbox = _SHAPES["long_narrow"]
    center, exy, ez = _metrics(bbox)
    with_bbox = {c.id: c for c in canonical_cameras(center, exy, ez, bbox)}
    without = {c.id: c for c in canonical_cameras(center, exy, ez)}
    near_area = math.prod(_frustum_span(bbox, with_bbox["front"])[:2])
    far_area = math.prod(_frustum_span(bbox, without["front"])[:2])
    assert near_area > far_area * 1.5
    # Both must still contain it — the fallback is conservative, not wrong.
    assert _frustum_span(bbox, without["front"])[2]


def test_a_degenerate_target_still_yields_a_usable_camera():
    """A zero-extent bbox has no distance to derive. It must not raise, and it
    must not put the eye on the target — `look_at_camera` refuses that."""
    for cam in canonical_cameras((0.0, 0.0, 0.0), 0.0, 0.0):
        assert math.dist(cam.eye, cam.target) > 0.0
        assert cam.near > 0.0


# ---------------------------------------------------------------------------
# Camera ordering — one order, so `.values()` and `ordered_paths()` agree
# ---------------------------------------------------------------------------

def test_canonical_cameras_are_returned_in_canonical_id_order():
    """`canonical_cameras()` order *is* `CANONICAL_CAMERA_IDS`, so
    `RenderResult.render_paths` (insertion-ordered from that list) and
    `ordered_paths()` agree about where every view sits."""
    cameras = canonical_cameras((0.0, 0.0, 0.0), 0.5, 0.5)
    assert [c.id for c in cameras] == list(CANONICAL_CAMERA_IDS)


def test_render_paths_values_matches_ordered_items():
    """The two spellings of "canonical order" agree, so misuse is harmless."""
    result = RenderResult(render_paths={
        c.id: Path(f"/tmp/{c.id}.png")
        for c in canonical_cameras((0.0, 0.0, 0.0), 1.0, 1.0)
    })
    assert list(result.render_paths.values()) == result.ordered_paths()
    assert [label for label, _p in result.ordered_items()] == \
        list(CANONICAL_CAMERA_IDS)
