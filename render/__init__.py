"""dycraft.render — offscreen rendering of object/scene IRs into PNGs.

Public API:

    from dycraft.render import (
        render_request,             # main entry point; dispatches to a backend
        RenderRequest, RenderResult,
        RenderUnavailable,
        CANONICAL_CAMERA_IDS,
        CameraSpec,
        RENDER_CONFIG_VERSION,
        render_cache_key,
    )

Two backends implement the same `render(request) -> RenderResult`: `open3d`
(Filament's OffscreenRenderer, the default) and `blender` (Cycles in a child
Blender process, `blender_renderer.py`).

The contract: six canonical cameras at 1280x720, per-part PBR colors + edges,
room cutaway for scenes, per-run cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Mapping


# ---------------------------------------------------------------------------
# Config-version sentinel
# ---------------------------------------------------------------------------

#: Cache-key version of the canonical render recipe. Bump it whenever what a
#: canonical camera draws changes: a canonical camera's id is its NAME, not a
#: hash of its geometry, so nothing else in the render cache key would notice
#: and a warm cache would serve the old pixels under the new recipe.
RENDER_CONFIG_VERSION = "v11"

#: Framebuffer for images a human or a VLM looks at directly.
FULL_RESOLUTION = (1280, 720)

#: A smaller framebuffer, for the all-limit views a scene review adds beside its
#: six full-size default views (`verify/review_images.py`).
#:
#: Aspect must match `FULL_RESOLUTION`: Open3D derives aspect from the
#: framebuffer while the camera rig is FOV-based off the default-pose AABB, so
#: same-ratio sizes frame identically and a different ratio would silently
#: reframe every view.
CARD_RESOLUTION = (640, 360)
                              #     edge cage (v3: cylinder/sphere meshes +
                              #     shape-matched edge overlays)


# ---------------------------------------------------------------------------
# Camera spec + the 6 canonical cameras
# ---------------------------------------------------------------------------

# All six sides, for every layer. The underside is the one view no other camera
# can infer, and it carries real information at both layers: feet, contact
# faces and wheel wells on an object; anything that sank through the floor in a
# scene.
CANONICAL_CAMERA_IDS = ("front", "right", "back", "left", "top", "bottom")


@dataclass(frozen=True)
class CameraSpec:
    """One render camera, derived per-request from the target bbox.

    Coordinates are in world frame: meters and +Z up.

    Viewer-relative convention (shared with sdk/_anchors.py):
        +X = viewer's LEFT,   -X = viewer's RIGHT
        +Y = FRONT,           -Y = BACK
        +Z = TOP,             -Z = BOTTOM
    A camera at +X therefore looks at the object's left face, so id="left"
    sits at +X and id="right" sits at -X.

    Image-plane orientation (verified empirically).
    Open3D / Filament cameras follow the OpenGL convention where image-right
    = `cross(image_up, -forward)`; for our setup that means each side camera
    rotates 180° relative to the "facing +Y → image-right = +X" intuition:

        front:  image-up = +Z, image-right = -X  (camera at +Y, looks -Y)
        left:   image-up = +Z, image-right = +Y  (camera at +X, looks -X)
        back:   image-up = +Z, image-right = +X  (camera at -Y, looks +Y)
        right:  image-up = +Z, image-right = -Y  (camera at -X, looks +X)
        top:    image-up = -Y, image-right = -X  (looking down -Z; aligned with `front`)
        bottom: image-up = +Y, image-right = -X  (looking up +Z; the underside)

    Equivalently: in the front camera image, things on world +X appear on
    the LEFT of the image (which matches "+X = viewer's left") and things
    on -X appear on the right. This is a consequence of the 180° turn the
    viewer makes to look back at the origin.

    """

    id: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_deg: float = 60.0           # vertical FOV
    width: int = 720
    height: int = 480
    # Clip planes, in meters. `-1.0` means "let the backend derive them from
    # the bounding box of the geometry it is about to draw", which is Open3D's
    # documented behavior for `setup_camera` when `near_clip`/`far_clip` are
    # omitted.
    #
    # Deriving them from the scene is right for a camera framed on the whole
    # model and wrong for a close-up: isolate a 44 mm dial and frame tightly on
    # it, and the auto near plane lands in front of the part, so the render
    # comes back EMPTY rather than close, until the eye is so far off that the
    # dial is a few pixels across. `framed_camera` therefore computes
    # its own, and anything that sets them explicitly must fold them into the
    # camera id (see `look_at_camera`), because a clip plane changes the
    # picture and `camera_id` is the whole camera identity in the render cache
    # key.
    near: float = -1.0
    far: float = -1.0


#: Border `canonical_cameras` leaves around the target, as a fraction of the
#: fitted extent. Applied to the exact fit, so it means what it says.
_CANONICAL_MARGIN = 0.06

#: Fallback half-extent for a target with none at all.
#:
#: Only reachable when a bbox is exactly degenerate on the axis being fitted —
#: a zero-thickness part, or a single-point target. `canonical_cameras` is
#: handed extents rather than a bbox, so when both are zero there is nothing to
#: be relative to and an absolute fallback is unavoidable. It exists to stop
#: `r = 0` producing `eye == target`, which `look_at_camera` refuses.
_DEGENERATE_EXTENT_M = 0.01


def _safe_extent(extent: float) -> float:
    """The half-extent to fit, guarding only the degenerate case.

    No fixed floor: a 0.1 m floor would frame anything smaller as if it were
    200 mm across. The cameras carry their own clip planes, so a small extent
    needs no padding.
    """
    return extent if extent > 0.0 else _DEGENERATE_EXTENT_M


def canonical_cameras(
    target_center: tuple[float, float, float],
    target_extent_xy: float,
    target_extent_z: float,
    target_bbox: tuple | None = None,
    resolution: tuple[int, int] | None = None,
) -> list[CameraSpec]:
    """Return the 6 canonical cameras, in `CANONICAL_CAMERA_IDS` order.

    `target_extent_xy` is the max XY half-extent of the target bbox; used to
    place the side cameras. `target_extent_z` is the Z half-extent; used to
    place the top/bottom cameras.

    `resolution` is the frame these cameras will be rendered into, and it is
    part of the framing rather than a detail of it: `framed_camera` fits the
    target against BOTH frame axes, and the horizontal half-angle is
    `aspect * tan(fov/2)`. The backend is handed only a vertical FOV and
    derives the horizontal one from the real framebuffer, so a fit solved at
    one aspect and rendered at another loses the sides silently — vertical
    framing is untouched, so the picture looks plausible with its edges gone
    (a 16:9 fit rendered at 1:1 cuts a 1.8 m bench's ends off).

    Defaults to `FULL_RESOLUTION`, so every caller that renders at 16:9 or at
    `CARD_RESOLUTION` (640x360, the same aspect) gets a byte-identical camera
    and no cache generation changes.

    The return order **is** `CANONICAL_CAMERA_IDS`: `RenderResult.render_paths`
    is insertion-ordered from this list, so any other order (geometric azimuth
    order, say) would make a consumer reaching for `.values()` transpose
    `left` and `right` relative to every legend and prompt.
    """
    cx, cy, cz = target_center
    rxy = _safe_extent(target_extent_xy)
    rz = _safe_extent(target_extent_z)

    # Framing is delegated to `framed_camera`, the exact per-corner fit, rather
    # than re-derived here. A center-depth fit (`max(rxy, rz) * k / tan(fov/2)`)
    # ignores that a nearer corner projects larger, so a roughly cubic object
    # clips unless the camera is padded back, and padding shrinks a small part
    # to a few percent of the frame. `framed_camera` solves the distance against
    # all eight corners, so containment holds by construction
    # (`test_a_framed_camera_contains_the_box_from_every_direction` sweeps 1 080
    # directions), and derives its own clip planes from that fit.
    #
    # Prefer the real bbox. `target_extent_xy` is a max over BOTH horizontal
    # axes (`_world._bbox_metrics`), so a box rebuilt from it squares off the
    # footprint, and an exact fit on that square backs the camera off as far as
    # the LONG axis demands in every direction: a long narrow object then fills
    # much less of the frame. The two extents are all a caller with no bbox can
    # give, so they stay the fallback.
    fit_bbox = target_bbox if target_bbox is not None else (
        (cx - rxy, cy - rxy, cz - rz), (cx + rxy, cy + rxy, cz + rz))

    def _canonical(camera_id: str, azimuth_deg: float,
                   elevation_deg: float) -> CameraSpec:
        cam = framed_camera(fit_bbox, azimuth_deg=azimuth_deg,
                            elevation_deg=elevation_deg, fov_deg=60.0,
                            margin=_CANONICAL_MARGIN,
                            resolution=resolution)
        # Keep the canonical NAME as the id. For these six the id is both a
        # render-cache key component and the PNG filename, and is deliberately
        # not a geometry hash — `RENDER_CONFIG_VERSION` is what invalidates
        # them when this recipe changes. See `render_cache_key`.
        return replace(cam, id=camera_id)

    # +Y, +X, -Y, -X azimuth (front/left/back/right): a camera at +X looks at
    # the +X face, which under the viewer-relative convention is the viewer's
    # LEFT face -- so id="left" sits at +X. Sides sit 15 deg above center.
    #
    # top/bottom look straight down/up. `framed_camera` resolves `up` for a
    # near-vertical view through the same fallback the hand-written -Y / +Y
    # used, so image-up still matches the front view's "far side at top".
    by_id = {
        "front":  _canonical("front",   90.0,  15.0),
        "left":   _canonical("left",     0.0,  15.0),
        "back":   _canonical("back",   270.0,  15.0),
        "right":  _canonical("right",  180.0,  15.0),
        "top":    _canonical("top",      0.0,  90.0),
        "bottom": _canonical("bottom",   0.0, -90.0),
    }
    return [by_id[camera_id] for camera_id in CANONICAL_CAMERA_IDS]


#: Reject an authored `up` this close to the view direction. Filament's basis
#: construction degenerates *smoothly* rather than at a point: by 0.5° the roll
#: is already arbitrary and the picture unpredictable rather than wrong, so a
#: floor is more useful than an exact-parallel test.
_UP_MIN_SIN = math.sin(math.radians(1.0))


def _finite_vec3(value, name: str) -> tuple[float, float, float]:
    """Coerce `value` to a 3-tuple of finite floats or raise `ValueError`.

    Deliberately strict about length and finiteness: a `None`, a 2-vector, or a
    NaN reaches `setup_camera` as a silently black or silently mis-framed image
    rather than an error, which is the failure class this whole module's
    comments keep guarding against.
    """
    try:
        parts = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a sequence of 3 numbers, got {value!r}") from None
    if len(parts) != 3:
        raise ValueError(
            f"{name} must have 3 components, got {len(parts)}: {value!r}")
    if not all(math.isfinite(v) for v in parts):
        raise ValueError(f"{name} has a non-finite component: {parts}")
    return (parts[0], parts[1], parts[2])


def look_at_camera(eye, look_at, up=None, *,
                   fov_deg: float = 60.0,
                   near: float = -1.0, far: float = -1.0,
                   roll_deg: float = 0.0) -> CameraSpec:
    """A fully-specified camera from `eye` + `look_at` (+ optional `up`).

    The second `CameraSpec` constructor in the package, and the counterpart to
    `canonical_cameras` above — read the two together. That one derives six
    fixed viewpoints from a bbox; this one takes a viewpoint the caller already
    decided on, which is what the agent-facing `snapshot` op needs.

    `up=None` means "pick one": `+Z` normally, and for a near-vertical view the
    same fallbacks the canonical `top` / `bottom` cameras use (`-Y` looking
    down, `+Y` looking up) so image orientation stays consistent with them. An
    *explicitly supplied* degenerate `up` is refused instead of corrected — a
    caller that stated it should hear that it does not work.

    **The id is a hash of the camera's own geometry**, and that is load-bearing
    rather than decorative. `CameraSpec.id` is simultaneously the PNG filename
    (`open3d_renderer._draw_plan`) and the entire camera identity inside
    `render_cache_key`, so a fixed id like `"custom"` would be filesystem-safe
    and still hardlink two different cameras to the same PNG. Hashing every
    field that changes the picture makes collisions impossible by construction,
    keeps the id filesystem-safe, and can never equal a member of
    `CANONICAL_CAMERA_IDS` — so a custom render can never be served into a
    canonical view's slot.
    """
    eye_t = _finite_vec3(eye, "eye")
    target_t = _finite_vec3(look_at, "look_at")

    fwd = (target_t[0] - eye_t[0], target_t[1] - eye_t[1],
           target_t[2] - eye_t[2])
    dist = math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2)
    if dist < 1e-6:
        raise ValueError(
            f"eye and look_at are the same point (distance {dist:.3g} m); "
            "the camera has no view direction")
    fwd_hat = (fwd[0] / dist, fwd[1] / dist, fwd[2] / dist)

    try:
        fov = float(fov_deg)
    except (TypeError, ValueError):
        raise ValueError(f"fov_deg must be a number, got {fov_deg!r}") from None
    if not math.isfinite(fov) or not (1.0 < fov < 179.0):
        raise ValueError(
            f"fov_deg must be a finite angle in (1, 179), got {fov}")

    def _sin_to_fwd(vec) -> float:
        norm = math.sqrt(sum(v ** 2 for v in vec))
        if norm < 1e-9:
            return 0.0
        cross = (vec[1] * fwd_hat[2] - vec[2] * fwd_hat[1],
                 vec[2] * fwd_hat[0] - vec[0] * fwd_hat[2],
                 vec[0] * fwd_hat[1] - vec[1] * fwd_hat[0])
        return math.sqrt(sum(c ** 2 for c in cross)) / norm

    if up is None:
        # `-fwd_hat[2] > 0` is "the eye is above the target", i.e. looking down.
        candidates = ((0.0, 0.0, 1.0),
                      (0.0, -1.0, 0.0) if fwd_hat[2] < 0 else (0.0, 1.0, 0.0))
        for cand in candidates:
            if _sin_to_fwd(cand) >= _UP_MIN_SIN:
                up_t = cand
                break
        else:
            # Unreachable for any real view direction: +Z and ±Y cannot both be
            # parallel to it. Kept so the loop has no silent fall-through.
            raise ValueError(
                "cannot derive an `up` for view direction "
                f"{fwd_hat}; pass one explicitly")
    else:
        up_t = _finite_vec3(up, "up")
        norm = math.sqrt(sum(v ** 2 for v in up_t))
        if norm < 1e-9:
            raise ValueError(f"up has zero length: {up_t}")
        sin_a = _sin_to_fwd(up_t)
        if sin_a < _UP_MIN_SIN:
            raise ValueError(
                f"up {up_t} is {math.degrees(math.asin(sin_a)):.2f}° from the "
                "view direction; the image has no defined orientation. For a "
                "near-vertical view pass up=(0,-1,0) looking down or "
                "up=(0,1,0) looking up, or omit up entirely.")

    # Roll last, about the view axis, so it composes with both the derived
    # and the explicitly-supplied up. Rotating about that axis preserves the
    # angle to it, so the degenerate-up check above still holds.
    if roll_deg:
        ang = math.radians(float(roll_deg))
        if not math.isfinite(ang):
            raise ValueError(f"roll_deg is not finite: {roll_deg!r}")
        c_a, s_a = math.cos(ang), math.sin(ang)
        k, u = fwd_hat, up_t
        cross = (k[1] * u[2] - k[2] * u[1], k[2] * u[0] - k[0] * u[2],
                 k[0] * u[1] - k[1] * u[0])
        dot = sum(a * b for a, b in zip(k, u))
        rot = tuple(u[i] * c_a + cross[i] * s_a + k[i] * dot * (1.0 - c_a)
                    for i in range(3))
        norm = math.sqrt(sum(v * v for v in rot)) or 1.0
        up_t = tuple(v / norm for v in rot)

    # Rounded before hashing so JSON round-trip float noise cannot mint a
    # spurious id for a camera the caller considers unchanged.
    key = json.dumps(
        {"eye": [round(v, 9) for v in eye_t],
         "target": [round(v, 9) for v in target_t],
         "up": [round(v, 9) for v in up_t],
         "fov_deg": round(fov, 9),
         # Only present when set, so every camera minted before clip planes
         # were passable keeps the id it had and its cached PNGs stay valid.
         **({"near": round(float(near), 9)} if near > 0 else {}),
         **({"far": round(float(far), 9)} if far > 0 else {})},
        sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=5).hexdigest()

    return CameraSpec(id=f"cam_{digest}", eye=eye_t, target=target_t,
                      up=up_t, fov_deg=fov, near=float(near), far=float(far))


def framed_camera(
    bbox,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float = 60.0,
    margin: float = 0.05,
    resolution: tuple[int, int] | None = None,
    up=None,
    roll_deg: float = 0.0,
    distance: float | None = None,
) -> CameraSpec:
    """A camera at (azimuth, elevation) that fits `bbox` to the frame.

    The constructor to reach for: `canonical_cameras` fixes six viewpoints and accepts whatever framing falls out,
    and `look_at_camera` leaves framing to the caller. This takes the thing to look at and the direction to look
    from, and solves for the distance. A preset distance fitted against the short axis of a 16:9 frame plus a fixed
    offset leaves a 5 cm knob about 2% of the frame; fitting the projected extent per camera does not.

    The fit is exact: each bbox corner, written in the camera's basis as off-axis half-extents `h` (vertical) and `w`
    (horizontal) at signed depth `t` in front of the target, is inside the frustum when `D + t >= h / tan(vfov/2)`
    and `D + t >= w / tan(hfov/2)`. The max over the eight corners is the smallest `D` that contains the box, so a
    near corner of a deep object pushes the camera back instead of being clipped.

    `azimuth_deg` is measured in the XY plane from +X toward +Y and `elevation_deg` above it, so azimuth 90 /
    elevation 15 is the canonical `front` (+Y, 15 deg up) and elevation 90 is straight down. `margin` pads the fitted
    extents as a fraction (0.05 is a 5% border). `resolution` supplies the aspect ratio (default `FULL_RESOLUTION`):
    pass the one you will render at, or a portrait box is fitted against a landscape frame. `up` passes through to
    `look_at_camera`, `None` included, so a near-vertical view matches the canonical `top` / `bottom` orientation;
    `roll_deg` spins it about the view axis.

    To override the solved distance: a **negative `margin`** crops (-0.3 pulls in until the extent overflows the
    frame by 30%, to magnify a squat part whose near corner holds the camera back), and an explicit `distance`
    replaces the solve, the only way to frame a point target. `bbox` is `((lo3), (hi3))`, or a bare 3-vector for a
    point.
    """
    flat = [v for v in bbox]
    if len(flat) == 3 and not hasattr(flat[0], "__len__"):
        lo = hi = tuple(float(v) for v in flat)          # point target
    else:
        lo = tuple(float(v) for v in bbox[0])
        hi = tuple(float(v) for v in bbox[1])
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError(
            f"bbox must be ((x0,y0,z0),(x1,y1,z1)) or a 3-vector point, "
            f"got {bbox!r}")
    if any(b < a for a, b in zip(lo, hi)):
        raise ValueError(f"bbox is inverted: {bbox!r}")
    center = tuple(0.5 * (a + b) for a, b in zip(lo, hi))

    fov = float(fov_deg)
    if not math.isfinite(fov) or not (1.0 < fov < 179.0):
        raise ValueError(f"fov_deg must be a finite angle in (1, 179), got {fov}")
    margin = float(margin)
    if not math.isfinite(margin) or margin <= -1.0:
        raise ValueError(
            f"margin must be a fraction greater than -1, got {margin}. "
            f"Positive pads the fit, negative crops into it.")
    if distance is not None:
        distance = float(distance)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                f"distance must be a positive number of meters, got "
                f"{distance}")

    az = math.radians(float(azimuth_deg))
    elev = math.radians(float(elevation_deg))
    # Unit vector from the target toward the eye.
    dir_to_eye = (math.cos(elev) * math.cos(az),
                  math.cos(elev) * math.sin(az),
                  math.sin(elev))

    # The camera's own basis. `forward` runs eye -> target, so a corner nearer
    # the eye has a negative depth and correctly demands a larger distance.
    forward = tuple(-v for v in dir_to_eye)

    def _cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def _norm(v):
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return tuple(c / n for c in v)

    # Learn the up vector from `look_at_camera` rather than re-deriving one
    # here, then fit against THAT basis and pass it back explicitly.
    #
    # Deriving it twice is what this two-pass prevents: two up-vector rules
    # with different near-vertical thresholds would, between those thresholds,
    # solve the fit in one basis and render in another, and a box fitted to the
    # frame would come out clipped.
    provisional = look_at_camera(
        tuple(c + d for c, d in zip(center, dir_to_eye)), center, up,
        fov_deg=fov)
    up_resolved = provisional.up

    right = _norm(_cross(forward, up_resolved))
    up_cam = _norm(_cross(right, forward))

    width, height = tuple(resolution or FULL_RESOLUTION)
    aspect = (width / height) if height else 1.0
    tan_v = math.tan(math.radians(fov / 2.0))
    tan_h = aspect * tan_v

    pad = 1.0 + margin
    needed = 0.0
    t_min = 0.0
    t_max = 0.0
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                v = (cx - center[0], cy - center[1], cz - center[2])
                w = abs(sum(a * b for a, b in zip(v, right))) * pad
                h = abs(sum(a * b for a, b in zip(v, up_cam))) * pad
                t = sum(a * b for a, b in zip(v, forward))
                t_min = min(t_min, t)
                t_max = max(t_max, t)
                needed = max(needed, h / tan_v - t, w / tan_h - t)

    diag = math.sqrt(sum((b - a) ** 2 for a, b in zip(lo, hi)))
    if distance is None:
        # Every constant here is RELATIVE to the target, because this function
        # has to hold over the whole range of things an object can contain: an
        # absolute floor such as `diag / 2 + 1e-3` is larger than the entire
        # fit for anything sub-millimeter, and would put the camera several
        # times too far back.
        #
        # `-t_min` is how far the nearest corner sits in front of the center,
        # so this keeps every corner strictly in front of the camera without
        # naming a length.
        # Stand clear of the nearest corner by a fraction of the target's own
        # size. `-t_min * 1.001` alone lets the camera hug that corner, which
        # collapses the frustum: the near plane derived below goes to
        # `0.001 * |t_min|`, and against a far plane four times the far corner
        # the depth range is wide enough to lose the geometry entirely, and a
        # magnify request (`margin` below zero) returns an empty frame.
        # Relative, so it means the same thing at every scale.
        if diag <= 0.0:
            # A point target — an anchor position, typically. There is no
            # extent to fit, so there is no distance to derive. Say so, rather
            # than inventing a scale or letting `look_at_camera` report the
            # downstream symptom ("eye and look_at are the same point"), which
            # names neither the cause nor the fix.
            raise ValueError(
                "cannot derive a distance for a target with no extent in "
                "frame — pass `distance` to say how far away to stand")
        standoff = diag * 0.05
        distance = max(needed, -t_min + standoff)

    eye = tuple(c + distance * d for c, d in zip(center, dir_to_eye))

    # Roll: the sixth degree of freedom. Rotating `up` about the view axis
    # leaves its angle to that axis unchanged, so `look_at_camera`'s
    # degenerate-up check still means what it meant.
    # Clip planes from the fit, not from the scene: Open3D derives them from
    # the geometry's bounding box when they are not given, which puts the near
    # plane in front of a tight close-up (see `CameraSpec.near`). Both are
    # relative to the fitted depths for the same reason the distance is.
    depth_near = distance + t_min
    depth_far = distance + t_max
    near = max(depth_near * 0.5, depth_far * 1e-6)
    far = depth_far * 4.0 + diag + distance
    return look_at_camera(eye, center, up_resolved, fov_deg=fov,
                          near=near, far=far, roll_deg=roll_deg)


# ---------------------------------------------------------------------------
# Output size
# ---------------------------------------------------------------------------

# Output image resolution (pixels) for each rendered view. Shared by every
# backend rather than owned by one: it is a property of the artifact contract
# (`runs/<id>/renders/<step>_<view>.png` is 1280x720 whoever drew it), not of a
# renderer. `CameraSpec.width` / `.height` are dead fields — nothing reads them.
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


def render_image_size() -> tuple[int, int]:
    """`(width, height)` in pixels of every PNG any backend writes.

    Public because a caller aiming a camera needs it: `CameraSpec.fov_deg` is
    the *vertical* field of view, so sizing a distance from a horizontal extent
    without the aspect ratio frames roughly 44% too tight at 16:9.
    """
    return (RENDER_WIDTH, RENDER_HEIGHT)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

DEFAULT_RENDER_BACKEND = "open3d"
RENDER_BACKEND_ENV = "DYCRAFT_RENDER_BACKEND"


def default_backend() -> str:
    """The backend a `RenderRequest` gets when its caller does not name one.

    Read from the environment on every construction rather than captured at
    import: the harness daemon, the in-process runner, and every orchestrator
    sub-run inherit `DYCRAFT_RENDER_BACKEND` through `cli.py::build_env_dict`,
    and a module-level snapshot would freeze whichever value happened to be set
    when `dycraft.render` was first imported.

    An unknown name is NOT rejected here — `render_request` raises with the list
    of registered backends, which is a better error than one from a dataclass
    default, and a name can legitimately be set before its module is imported.
    """
    return os.environ.get(RENDER_BACKEND_ENV) or DEFAULT_RENDER_BACKEND


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RenderUnavailable(Exception):
    """Raised when the GL / Filament context cannot be created.

    Callers that render for an agent (snapshots, review images) catch this and
    return a structured error rather than crashing the run.
    """


# ---------------------------------------------------------------------------
# Request / Result
# ---------------------------------------------------------------------------

@dataclass
class RenderRequest:
    """Input to a renderer backend.

    Fields:
      layer            : "object" or "scene"
      ir               : finalized ObjectIR or SceneIR dict
      catalog          : catalog mapping for scene layer; None for object layer
      target           : "all" | <entity_name> | <camera_id> — see
                         `_world._resolve_target_bbox`
      out_dir          : directory under runs/<id>/renders/
      step             : runner step index; PNG filenames are
                         "<step>_<view>.png"
      cache            : optional in-memory dict for hash-keyed PNG reuse
      config_version   : bumped when the renderer recipe changes; enters the
                         cache key
    """

    layer: Literal["object", "scene"]
    ir: dict
    catalog: Mapping[str, dict] | None
    target: str
    out_dir: Path
    step: int
    cache: dict | None = None
    config_version: str = RENDER_CONFIG_VERSION
    # Resolved per construction from `$DYCRAFT_RENDER_BACKEND`; see
    # `default_backend`. A caller that names one explicitly always wins.
    backend: str = field(default_factory=default_backend)

    #: Abort the render if an authored `@shader` or `@lighting` block fails,
    #: instead of falling back to the material's scalars. Set for any image
    #: someone judges appearance from — the project agent's visual review and
    #: the delivery render — because judging a material the agent did not
    #: author is worse than being told to fix it. Off for previews, where a
    #: picture with one flat surface beats no picture.
    strict_materials: bool = False

    # Base directory for resolving an object mesh part's run-relative
    # `params["filename"]` (e.g. "assets/meshes/<digest>.obj"). Set to the run
    # dir by object tooling and to the staged catalog root by scene tooling.
    # `None` (or an absolute filename) means the filename is used as-is.
    asset_base_dir: Path | None = None

    # Override the room's enclosure for THIS RENDER ONLY: "interior",
    # "open_top", or "open_air". `None` uses whatever the IR says.
    #
    # This exists for runs authored before enclosure did. A scene's `set_room`
    # declares an extent, and until now an extent implied a sealed box — so an
    # outdoor scene that used the room as its ground plane and sky-well got a
    # ceiling and four walls it never asked for, which shut out the world dome
    # and turned an alley into a tunnel. Those runs cannot be re-authored, and
    # editing an `ir.json` on disk to fix a picture would falsify the run. So
    # the correction is a render-time argument that changes nothing stored.
    #
    # Never inferred. Not from the run's name, not from its dimensions, not
    # from whether the scene happens to have placed walls of its own — all
    # three would be a guess that fails silently on the scene that breaks the
    # pattern.
    room_enclosure: str | None = None

    # Joint pose for the rendered geometry. `None` renders at the IR's
    # authored joint defaults. When supplied, the dict's joint values
    # override defaults along each part's ancestor chain. Camera framing
    # is always computed from the default-pose AABB so lo/hi renders
    # share the framing of the default views.
    #
    # IR-internal units: radians for a revolute joint, meters for a prismatic
    # one. Convert a surface number before passing it.
    #
    # Object layer: keys are joint names (unqualified).
    # Scene layer: keys are "<instance>.<joint>" (or unqualified, applied
    #              to every instance whose object has that joint), per
    #              sdk/_scene_kernel.py:effective_object_poses — which also
    #              lays each instance's committed `initial_joints` underneath,
    #              so an omitted joint draws at the state the scene selected
    #              rather than at its definition's default.
    poses: dict | None = None

    # Whole-body rigid deltas, `{instance: ((tx,ty,tz), (qw,qx,qy,qz))}`, in
    # world axes — the dynamics layer's per-frame actor motion. Scene layer
    # only. See sdk/_scene_kernel.py:_resolve_world_pose.
    base_deltas: dict | None = None

    # Deformation-induced motion of a joint's mount frame, in the PARENT PART's
    # local frame: `{instance: {joint: ((tx,ty,tz), (qw,qx,qy,qz))}}` (scene
    # layer) or `{joint: se3}` (object layer). This is what makes a child part
    # ride its parent's deformation; see sdk/_object_kernel.py:_apply_mount.
    mount_deltas: dict | None = None

    # Per-part deformed geometry, replacing what the part's asset file holds:
    # `{"<instance>.<part>": (vertices[n,3], triangles[m,3])}` on the scene
    # layer, `{"<part>": (v, t)}` on the object layer. Vertices are in the
    # part's LOCAL frame — the renderer applies that part's FK on top, exactly
    # as it does for the rest mesh, so a deformed part is placed by the same
    # code path as a rigid one.
    #
    # Only `mesh` / `solid` parts can be overridden. An analytic primitive has
    # no vertices to replace and is left alone rather than silently ignored
    # somewhere deeper.
    deformed: dict | None = None

    # Scene instances physically present for this request. ``None`` means the
    # whole frozen SceneIR; a set (including the empty set) is the animation
    # frame's effective ``exists`` set after attachment inheritance.
    include_instances: set[str] | None = None

    # Entities to DRAW, everything else hidden. `None` draws the whole model.
    #
    # Distinct from `target`, which only re-frames: a part-name target moves the
    # cameras to fit that part and still draws every other part, so a bore, a
    # contact face or an interior fit stays buried behind the shell that
    # encloses it. This hides the rest outright.
    #
    # Object layer: part names. Scene layer: instance names. An unknown name is
    # an error rather than a silent no-op — a typo that quietly drew everything
    # would read as "isolation does not work".
    #
    # In the render cache key, because two requests that agree on every other
    # field but differ here produce genuinely different pixels.
    only: tuple[str, ...] | None = None

    # Like `poses`/`base_deltas`, the frame-specific geometry inputs above are
    # not in the render cache key — vary `pose_label` per frame or every frame
    # hardlinks to the first one's PNG.


    # zlib level for the written PNG. `None` keeps Pillow's default (6), which
    # every existing artifact was written with — leave it alone for anything
    # whose bytes are compared, cached, or judged.
    #
    # It is a knob because PNG encoding dominates rendering: measured at
    # 1280x720, level 6 costs ~400 ms/frame against ~15 ms for the actual GL
    # draw. Level 1 costs ~74 ms for files ~25% larger. That trade is wrong for
    # the five canonical stills a run keeps forever and right for the tens of
    # frames of a clip that exist only to be muxed into an mp4.
    png_compress_level: int | None = None

    # Label for the pose configuration. Included in the render-cache
    # key and used as the PNG filename suffix when not "default".
    # Conventionally "default" / "lo" / "hi".
    #
    # NOTE: the render cache key covers this label, NOT `poses`/`base_deltas`
    # themselves. A caller rendering many distinct poses must give each one a
    # distinct label or every frame will hardlink to the first one's PNG.
    pose_label: str = "default"

    # Optional explicit camera selection. None means all six canonical
    # cameras.
    camera_ids: tuple[str, ...] | None = None

    # Fully-specified cameras, bypassing the canonical preset entirely.
    # Honored ahead of both `camera_ids` and `target`; setting it together with
    # `camera_ids` raises rather than silently preferring one.
    #
    # Each spec's `id` becomes BOTH the render-cache key component and the PNG
    # filename, so it must be filesystem-safe AND must encode the camera's
    # geometry — two different cameras sharing an id hardlink to the first
    # one's PNG. `look_at_camera()` derives such an id; prefer it over building
    # a `CameraSpec` by hand.
    cameras: tuple[CameraSpec, ...] | None = None

    # Framebuffer size for this request. Enters the cache key, because two
    # requests that agree on every other field but differ here produce
    # different images — without it a low-resolution selection render would be
    # served to a caller that asked for a full-resolution one.
    resolution: tuple[int, int] = FULL_RESOLUTION


@dataclass
class RenderResult:
    """Output from a renderer backend."""

    render_paths: dict[str, Path] = field(default_factory=dict)
    # camera_id -> Path of the written PNG
    target_bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    cache_hits: dict[str, bool] = field(default_factory=dict)
    cameras: list[CameraSpec] = field(default_factory=list)
    #: Color management actually in force when these images were written —
    #: view transform, look, exposure, white balance. Reported by the Blender
    #: child after the authored `@lighting` block ran, so the visual-review
    #: record can hash what the image really is rather than what was asked
    #: for. Empty on backends that do not report it.
    view_settings: dict = field(default_factory=dict)
    # Substitutions the backend had to make to produce these images: an
    # asset-backed part whose file would not load, or a primitive this renderer
    # does not draw, either of which is rendered as a bounding BOX. Empty on the
    # normal path. Non-empty means the pixels do not depict the model, so a
    # caller showing them to a model or a human should say so rather than let a
    # brick be read as the authored shape.
    warnings: list[str] = field(default_factory=list)

    def ordered_items(self) -> list[tuple[str, Path]]:
        """`(camera_id, path)` in canonical camera order, missing ids dropped.

        Prefer this over `ordered_paths()` anywhere the images are handed to a
        model: the camera id is the label the prompt legend needs, and reading
        both out of one list is what keeps a legend from drifting off its
        images (see `verify.review_images.ImageSet`).
        """
        return [(c, self.render_paths[c]) for c in CANONICAL_CAMERA_IDS
                if c in self.render_paths]

    def ordered_paths(self) -> list[Path]:
        """Paths in canonical camera order, filtering missing ids.

        A `RenderRequest.cameras` camera is filtered out here **by design**:
        every caller of this accessor means "the canonical set, in canonical
        order". A caller that supplied its own
        cameras must read `render_paths` directly, keyed by the ids it built.
        """
        return [p for _label, p in self.ordered_items()]


def _cameras_for_request(
    request: RenderRequest,
    target_center: tuple[float, float, float],
    target_extent_xy: float,
    target_extent_z: float,
    target_bbox: tuple | None = None,
) -> list[CameraSpec]:
    """Resolve the cameras for one request.

    All six canonical cameras, unless an explicit `camera_ids` or a `target`
    naming one camera narrows them — or `cameras` replaces them outright.

    `cameras` wins before the preset is even built, since with a caller-supplied
    viewpoint the bbox no longer positions anything.
    """
    if request.cameras is not None:
        if request.camera_ids is not None:
            raise ValueError(
                "RenderRequest sets both `cameras` and `camera_ids`; pick one "
                "— `cameras` replaces the canonical preset that `camera_ids` "
                "filters, so a request with both is a bug")
        return list(request.cameras)

    available = canonical_cameras(
        target_center, target_extent_xy, target_extent_z, target_bbox,
        # The frame these will actually be rendered into — see
        # `canonical_cameras`, which fits against it.
        resolution=request.resolution)
    by_id = {camera.id: camera for camera in available}

    requested = request.camera_ids
    if requested is not None:
        unknown = [camera_id for camera_id in requested
                   if camera_id not in by_id]
        if unknown:
            raise ValueError(f"unknown camera ids: {unknown}")
        return [by_id[camera_id] for camera_id in requested]
    if request.target in by_id:
        return [by_id[request.target]]
    return available


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

def render_cache_key(
    *,
    ir_hash: str,
    target: str,
    camera_id: str,
    config_version: str = RENDER_CONFIG_VERSION,
    catalog_fingerprint: str | None = None,
    pose_label: str = "default",
    backend: str = DEFAULT_RENDER_BACKEND,
    resolution: tuple[int, int] = FULL_RESOLUTION,
    only: tuple[str, ...] | None = None,
) -> tuple:
    """Build the in-memory render-cache key.

    `catalog_fingerprint` is meaningful only for scene-layer renders; pass None for the object layer.

    `backend` is in the key because the cache dict is per session (`ctx.render_cache`), not per backend, and two
    backends draw the same IR from the same camera into different pixels: without it, switching backends in one
    process would serve the other's PNG, silently, since a hit is a hardlink and the picture is plausible. Each
    backend also passes its own recipe version as `config_version` (`_world.plan_outputs`), so bumping Blender's
    settings does not invalidate Open3D's renders.

    `pose_label` distinguishes renders of one IR at different joint poses ("default" / "lo" / "hi").

    `camera_id` is the whole camera identity here; none of the eye/target/up/fov it stands for enters the key. That
    is exact for the canonical preset. A caller supplying its own `RenderRequest.cameras` owes the key an id that
    encodes the camera's geometry, or two viewpoints hardlink to one PNG (`look_at_camera` hashes it).

    `resolution` is in the key so a small selection render is never served to a caller that asked for full size, and
    `only` so an isolated render of one part is never served for the whole model; it is sorted, so one set written in
    two orders is one entry.
    """
    return (
        ir_hash,
        target,
        camera_id,
        config_version,
        catalog_fingerprint or "",
        pose_label,
        backend,
        tuple(resolution),
        tuple(sorted(only)) if only else (),
    )


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Callable[["RenderRequest"], "RenderResult"]] = {}

#: Backends that have to be imported before they can appear in `_BACKENDS`.
#: Import is lazy and by name, so `dycraft.render` stays importable with
#: neither open3d nor bpy installed (most of the test suite has neither).
_BACKEND_MODULES = {
    "open3d": "open3d_renderer",
    "blender": "blender_renderer",
}


def register_backend(
    name: str,
    fn: Callable[["RenderRequest"], "RenderResult"],
) -> None:
    """Register a renderer backend by name; `fn` renders one request."""
    _BACKENDS[name] = fn


def _load_backend(name: str) -> None:
    """Import the module that registers `name`, if we know of one."""
    if name in _BACKENDS:
        return
    mod = _BACKEND_MODULES.get(name)
    if mod is None:
        return
    import importlib
    try:
        importlib.import_module(f"{__name__}.{mod}")
    except ImportError as e:
        raise RenderUnavailable(
            f"render backend {name!r} could not be imported: {e}") from e


def _resolve_backend(name: str):
    _load_backend(name)
    fn = _BACKENDS.get(name)
    if fn is None:
        known = sorted(set(_BACKENDS) | set(_BACKEND_MODULES))
        raise ValueError(f"unknown render backend: {name!r} (known: {known})")
    return fn


def _open3d_backend(request: RenderRequest) -> RenderResult:
    # Lazy import: the Open3D module must not be required at package import
    # time (the surrounding test suite must run in environments without GL).
    from . import open3d_renderer
    return open3d_renderer.render(request)


register_backend("open3d", _open3d_backend)


def render_request(request: RenderRequest) -> RenderResult:
    """Main entry point. Dispatches to the named backend."""
    return _resolve_backend(request.backend)(request)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _build_scene_urdf_xml(ir, catalog, out_dir, *, asset_base_dir=None):
    """Build the scene ir.urdf XML.

    `scene_ir_to_urdf` emits each catalog object's full link/joint tree, so a
    scene of articulated objects exports as articulated.

    When `asset_base_dir` is given, each catalog object's mesh assets are copied
    next to the URDF and its paths rewritten relative to it, so the export
    survives the run dir moving.
    """
    from . import urdf_export
    catalog_local = catalog
    if asset_base_dir is not None:
        catalog_local = {}
        missing: list = []
        for ref, obj_ir in catalog.items():
            if urdf_export.has_localizable_assets(obj_ir):
                obj_ir = urdf_export.localize_object_meshes(
                    obj_ir,
                    source_base_dir=asset_base_dir,
                    dest_dir=out_dir / "assets" / "catalog" / ref,
                    urdf_dir=out_dir,
                    missing=missing,
                )
            catalog_local[ref] = obj_ir
        if missing:
            print(f"[final_render] urdf: {len(missing)} catalog mesh "
                  "asset(s) missing at source; left as original paths")
    return urdf_export.scene_ir_to_urdf(ir, catalog_local)


def render_final_to_run_dir(
    ir: dict,
    run_dir: Path,
    *,
    layer: Literal["object", "scene"],
    catalog: Mapping[str, dict] | None = None,
    asset_base_dir: Path | None = None,
    out_subdir: str = "final_render",
) -> "RenderResult | None":
    """Render the 6 canonical views of `ir` into `<run_dir>/<out_subdir>/`.

    Used by `python -m dycraft.cli` (api backend) and Session.final_render
    to give the user a final visual
    summary after the agent loop terminates. Returns the RenderResult on
    success, or None if rendering is unavailable. Errors are caught and
    printed; no exception is raised.
    """
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_arg = (catalog or {}) if layer == "scene" else None
    result: "RenderResult | None"
    try:
        result = render_request(RenderRequest(
            layer=layer, ir=ir, catalog=catalog_arg,
            target="all", out_dir=out_dir, step=0, cache={},
            asset_base_dir=(run_dir if layer == "object"
                            else asset_base_dir),
        ))
    except RenderUnavailable as e:
        print(f"[final_render] skipped: rendering unavailable ({e})")
        result = None
    except BaseException as e:  # noqa: BLE001
        print(f"[final_render] skipped: {type(e).__name__}: {e}")
        result = None
    if result is not None:
        print("[final_render] saved 6 views:")
        for cid, p in sorted(result.render_paths.items()):
            print(f"   {cid}: {p}")
    # URDF write runs independently of the PNG render outcome: a user with
    # a broken GL stack should still get a useful kinematic export.
    try:
        from . import urdf_export
        if layer == "scene":
            xml = _build_scene_urdf_xml(
                ir, catalog_arg or {}, out_dir,
                asset_base_dir=asset_base_dir)
        else:
            # Self-contain any object mesh assets: copy them next to the URDF
            # and rewrite their paths relative to it, so the export survives the
            # run dir moving. Object mesh filenames are stored relative to the
            # run dir (assets/meshes/<digest>.obj), so run_dir is the source base.
            ir_for_urdf = ir
            if urdf_export.has_localizable_assets(ir):
                missing: list = []
                ir_for_urdf = urdf_export.localize_object_meshes(
                    ir, source_base_dir=run_dir,
                    dest_dir=out_dir / "assets" / "meshes",
                    urdf_dir=out_dir, missing=missing)
                if missing:
                    print(f"[final_render] urdf: {len(missing)} mesh asset(s) "
                          f"missing at source; left as original paths")
            xml = urdf_export.object_ir_to_urdf(ir_for_urdf)
        urdf_path = out_dir / "ir.urdf"
        urdf_path.write_text(xml, encoding="utf-8")
        print(f"[final_render] saved urdf: {urdf_path}")
    except BaseException as e:  # noqa: BLE001
        print(f"[final_render] urdf skipped: {type(e).__name__}: {e}")
    return result


FINAL_RENDER_DIR = "final_render"
FINAL_RENDER_FAILED_DIR = "final_render_failed"


def finalize_render_result(
    ir: dict | None,
    run_dir: Path,
    *,
    layer: Literal["object", "scene"],
    catalog: Mapping[str, dict] | None = None,
    asset_base_dir: Path | None = None,
    failed: bool = False,
    build_error: str | None = None,
) -> dict:
    """Render `ir` to `<run_dir>/final_render/`, or `final_render_failed/` when `failed=True`.

    On the success path, an unavailable renderer returns `ok:True, paths:[], note` (the URDF is still attempted by
    `render_final_to_run_dir`) and other exceptions propagate. On the failure path every exception is swallowed, an
    empty output directory is still created so tooling can list it, and the call returns `ok:True, paths:[], note`,
    so a caller in a termination sequence needs no try/except.

    A caller that could not build an `ir` passes `ir=None, build_error=<message>`, which is recorded in `note` with an
    empty output directory. A success-path caller usually returns its own build-failure dict instead; this is for a
    uniform `ok` + `out_dir` + `paths` shape whatever failed.

    Returns a JSON-serializable dict: `ok` (True except a success-path build failure with `ir=None`), `out_dir` (an
    absolute path), `paths` (the rendered PNGs, possibly empty), `urdf` when the URDF was attempted,
    `render_warnings` when the renderer substituted a bounding box for something it could not draw, and `note` /
    `error` on degraded or failed outcomes.
    """
    out_subdir = FINAL_RENDER_FAILED_DIR if failed else FINAL_RENDER_DIR
    out_dir = run_dir / out_subdir
    try:
        if ir is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            if failed:
                return {"ok": True, "out_dir": str(out_dir), "paths": [],
                        "note": build_error or "no IR to render"}
            return {"ok": False, "out_dir": str(out_dir), "paths": [],
                    "error": build_error or "no IR to render"}
        catalog_arg = catalog if layer == "scene" else None
        result = render_final_to_run_dir(
            ir, run_dir, layer=layer, catalog=catalog_arg,
            asset_base_dir=asset_base_dir, out_subdir=out_subdir,
        )
        payload: dict = {
            "ok": True,
            "out_dir": str(out_dir),
            "urdf": str(out_dir / "ir.urdf"),
        }
        if result is not None:
            payload["paths"] = [str(p) for p in result.ordered_paths()]
            # Substitutions the renderer had to make — an asset that would not
            # load or a primitive it cannot draw, either of which is rendered
            # as a BOUNDING BOX. This is the deliverable render, so a brick
            # standing in for a part is exactly what a reader must be told
            # about; without it `final_render/` is indistinguishable from a run
            # that modeled a box on purpose.
            if result.warnings:
                payload["render_warnings"] = list(result.warnings)
        else:
            payload["paths"] = []
            payload["note"] = ("PNG render skipped (renderer unavailable); "
                               "URDF still written.")
        return payload
    except BaseException as e:  # noqa: BLE001
        if failed:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except BaseException:  # noqa: BLE001
                pass
            return {"ok": True, "out_dir": str(out_dir), "paths": [],
                    "note": f"render skipped: {type(e).__name__}: {e}"}
        raise


__all__ = [
    "render_request",
    "register_backend",
    "default_backend",
    "DEFAULT_RENDER_BACKEND",
    "RENDER_BACKEND_ENV",
    "render_image_size",
    "RENDER_WIDTH",
    "RENDER_HEIGHT",
    "render_final_to_run_dir",
    "finalize_render_result",
    "FINAL_RENDER_DIR",
    "FINAL_RENDER_FAILED_DIR",
    "RenderRequest",
    "RenderResult",
    "RenderUnavailable",
    "CameraSpec",
    "CANONICAL_CAMERA_IDS",
    "canonical_cameras",
    "look_at_camera",
    "RENDER_CONFIG_VERSION",
    "render_cache_key",
    "FULL_RESOLUTION",
    "CARD_RESOLUTION",
]
