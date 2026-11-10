"""`snapshot`: render the current source from chosen cameras, for the agent to look at.

Render-only. `run_snapshot` builds the IR from the current source
(`tools.build_current_ir`) and hands back PNG paths for the agent to read with
its own tooling; it writes nothing a compile or a review reads.
`_review_render.py` reuses the resolution, pose and `only` helpers, so a
reviewer's extra cameras render the way a snapshot does.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..tool_io import confine_to_run_dir
from .tools import (
    COMPILE_REPORTS_DIR,
    RENDERS_DIR,
    ToolContext,
    build_current_ir,
)


#: Fast zlib for snapshots. The opposite trade from the canonical stills: those
#: are kept forever and their bytes are compared and judged, while a snapshot is
#: looked at once and abandoned. Measured at 1280x720, level 6 costs ~400 ms per
#: frame against ~15 ms of actual GL; level 1 costs ~74 ms for files ~25% larger.
SNAPSHOT_PNG_COMPRESS_LEVEL = 1


# ---------------------------------------------------------------------------
# Render inputs
# ---------------------------------------------------------------------------


def _resolve_view_target(ctx: ToolContext, target: str):
    """Return (layer, ir, catalog, target_for_request) or raise.

    Builds the IR from the CURRENT source file via `build_current_ir`; no
    prior compile is required.

    Object layer: requires the current source to build cleanly.
    Scene layer:
      * target == a built object's handle name → render that ObjectIR in
        isolation.
      * otherwise: requires the current scene.py to build cleanly.

    Raises PermissionError when the build is required but fails (caller
    converts to a structured tool error).
    """
    layer = ctx.layer_cfg.name

    if layer == "scene":
        handles = getattr(ctx.layer_payload, "object_handles", {}) or {}
        name_to_hash = getattr(ctx.layer_payload, "name_to_hash", {}) or {}
        catalog = getattr(ctx.layer_payload, "catalog", {}) or {}
        if target in handles:
            obj_ir = catalog[name_to_hash[target]]
            return "object", obj_ir, None, "all"

    build = build_current_ir(ctx)
    if not build["ok"]:
        summary = build.get("error_summary") or "current source did not build"
        raise PermissionError(summary)

    current_ir = build["ir"]
    if layer == "object":
        return "object", current_ir, None, target
    catalog = getattr(ctx.layer_payload, "catalog", {}) or {}
    return "scene", current_ir, catalog, target


def _render_inputs(ctx: ToolContext, target: str):
    """`_resolve_view_target` plus the `asset_base_dir` that goes with it.

    The base dir keys off the run's LAYER, not off the IR kind
    `_resolve_view_target` returns: an object rendered in isolation from a
    scene run still resolves its assets from that run's payload.
    """
    layer, ir, catalog, request_target = _resolve_view_target(ctx, target)
    asset_base_dir = (getattr(ctx.layer_payload, "asset_base_dir", None)
                      if ctx.layer_cfg.name == "scene"
                      else ctx.run_dir)
    return layer, ir, catalog, request_target, asset_base_dir


# ---------------------------------------------------------------------------
# snapshot (pure operation) — render-only
# ---------------------------------------------------------------------------

#: Directory components a snapshot may never be written into. Filename
#: collision is already impossible (the camera id is a geometry hash and the
#: step prefix is unique), so this is about not polluting directories other
#: code enumerates: the canonical evidence bundle, the read-only staged
#: catalog, and the sub-run tree.
_SNAPSHOT_RESERVED_DIRS = frozenset({
    COMPILE_REPORTS_DIR, RENDERS_DIR, "catalog", "assets",
    "final_render", "final_render_failed", "sub_runs",
})


def _snapshot_out_dir(ctx: ToolContext, out_dir: str) -> "Path | str":
    """Resolve and vet the snapshot output dir, or return an error string.

    Does NOT create it. Every rejection anywhere in `run_snapshot` — including a
    frame index resolved much later — has to leave the run dir untouched, and an
    empty directory nobody asked for is a question a reader then has to answer
    (the same reason `ToolContext.__post_init__` creates only the three a run
    always fills). So the caller mkdirs once, immediately before rendering.
    """
    if not isinstance(out_dir, str) or not out_dir.strip():
        return "out_dir must be a non-empty run-relative directory name"
    resolved = confine_to_run_dir(ctx.run_dir, out_dir)
    if resolved is None:
        return (f"out_dir not allowed: {out_dir!r} — it must be a relative "
                "path inside the run directory, with no '..' component")
    if resolved == ctx.run_dir.resolve():
        return "out_dir must name a subdirectory, not the run root"
    head = Path(out_dir).parts[0]
    if head in _SNAPSHOT_RESERVED_DIRS:
        return (f"out_dir may not write into {head!r}: that directory is part "
                "of the run's evidence bundle or staged inputs. Pick a name of "
                "your own, e.g. 'snapshots'.")
    if resolved.exists() and not resolved.is_dir():
        return f"out_dir exists and is not a directory: {out_dir}"
    return resolved


#: Most views one `snapshot` call may render. The binding cost is the agent's
#: own context — it `Read`s every PNG — so the prompt asks for the two or three
#: that answer the question and this is the backstop.
_SNAPSHOT_MAX_VIEWS = 12

#: Per-image pixel bounds for `resolution`. The floor is about legibility: a
#: detail shot below this is not a detail shot. The ceiling is the agent's
#: context again, plus the fact that nothing downstream of a snapshot resamples.
_SNAPSHOT_MIN_PX = 320
_SNAPSHOT_MAX_PX = 2048
_SNAPSHOT_MAX_TOTAL_PX = 1920 * 1080

def _snapshot_resolution(resolution) -> "tuple | str":
    """Validate `resolution`, or return an error string.

    A canonical view may use any aspect: `canonical_cameras` fits the camera
    to the requested resolution, and `render_cache_key` includes it, so
    renders at different sizes never alias. What is checked is the pixel
    budget.
    """
    if resolution is None:
        from ...render import FULL_RESOLUTION
        return FULL_RESOLUTION
    try:
        w, h = (int(v) for v in resolution)
    except (TypeError, ValueError):
        return (f"resolution must be a (width, height) pair of integers, got "
                f"{resolution!r}")
    if not (_SNAPSHOT_MIN_PX <= w <= _SNAPSHOT_MAX_PX
            and _SNAPSHOT_MIN_PX <= h <= _SNAPSHOT_MAX_PX):
        return (f"resolution {w}x{h} is outside {_SNAPSHOT_MIN_PX}.."
                f"{_SNAPSHOT_MAX_PX} px per side")
    if w * h > _SNAPSHOT_MAX_TOTAL_PX:
        return (f"resolution {w}x{h} exceeds the "
                f"{_SNAPSHOT_MAX_TOTAL_PX:,}-pixel cap "
                f"(~{_SNAPSHOT_MAX_TOTAL_PX / 1e6:.1f} MP). Reduce width or height.")
    return (w, h)


def _joint_types(ir: dict, catalog, layer: str) -> dict:
    """`{key: joint_type}` for every joint the layer's `poses` may name.

    Object layer: joint names, straight off the IR. Scene layer: the joints
    live in the *catalog objects* the instances reference, not in the SceneIR,
    and `RenderRequest.poses` accepts both a qualified `"<instance>.<joint>"`
    key and an unqualified `"<joint>"` broadcast to every instance that has one
    (`sdk/_scene_kernel.py::effective_object_poses`). Both spellings are
    reported here so the unit conversion below can find a type for either.

    A joint the caller does not name is not unposed: it holds that instance's
    committed `initial_joints` value, or its definition's default. So a
    snapshot with `poses=None` shows the scene as authored, and one that names
    a single joint changes that joint only.
    """
    if layer != "scene":
        return {j["name"]: j.get("type") for j in (ir.get("joints") or ())}
    out: dict = {}
    for inst in (ir.get("instances") or ()):
        obj = (catalog or {}).get(inst.get("object_ref")) or {}
        for j in (obj.get("joints") or ()):
            out[f"{inst['name']}.{j['name']}"] = j.get("type")
            # An unqualified name broadcasts. Two instances could in principle
            # disagree on the type behind one name; mark it rather than pick.
            prev = out.get(j["name"], j.get("type"))
            out[j["name"]] = (j.get("type") if prev == j.get("type")
                              else "__ambiguous__")
    return out


def _snapshot_poses(ir: dict, poses, *, catalog=None,
                    layer: str = "object") -> "tuple | str":
    """Convert a surface pose dict to IR units + a cache label, or error.

    Surface units follow the SDK: degrees for a revolute joint, meters for a
    prismatic one. The IR is radians/meters, so the degrees have to be
    converted here — the same conversion `attach_revolute` does on the way in.

    The label is not decoration. `RenderRequest.poses` is deliberately NOT part
    of the render cache key; `pose_label` is (see `render.render_cache_key`), so
    two poses sharing a label hardlink to the first one's PNG. Hashing the pose
    means the label cannot drift from what it names.
    """
    if poses is None:
        return None, "default"
    if not isinstance(poses, dict):
        return (f"poses must be a dict of joint name -> value, got "
                f"{type(poses).__name__}")
    types = _joint_types(ir, catalog, layer)
    movable = {k: t for k, t in types.items()
               if t not in ("fixed", None)}
    fixed = {k for k, t in types.items() if t == "fixed"}
    out: dict = {}
    for name, value in poses.items():
        if name in fixed:
            return (f"poses: {name!r} is a fixed joint and has nothing to "
                    f"pose. Movable joints: {sorted(movable)}")
        if name not in movable:
            return (f"poses: unknown joint {name!r}. Movable joints: "
                    f"{sorted(movable)}")
        if movable[name] == "__ambiguous__":
            return (f"poses: {name!r} is a revolute joint on one instance and "
                    f"prismatic on another, so its unit is ambiguous — "
                    f"qualify it as '<instance>.{name}'")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"poses[{name!r}] must be a number, got {value!r}"
        if not math.isfinite(v):
            return f"poses[{name!r}] is not finite: {value!r}"
        out[name] = (math.radians(v) if movable[name] == "revolute" else v)
    if not out:
        return None, "default"
    label = "pose_" + hashlib.blake2b(json.dumps(
        {k: round(v, 9) for k, v in sorted(out.items())},
        sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=8).hexdigest()[:10]
    return out, label


def _snapshot_only(ir: dict, only, *, layer: str) -> "tuple | str":
    """Validate `only` against the entities the layer draws, or error.

    Object layer: part names. Scene layer: instance names. Validated on both,
    because an unrecognised name reaches the renderer as an empty draw list —
    so a typo renders a blank frame and reads as "isolation does not work"
    rather than as a misspelling.
    """
    if only is None:
        return None
    if isinstance(only, str):
        only = [only]
    try:
        names = [str(v) for v in only]
    except TypeError:
        return f"only must be a list of names, got {only!r}"
    if not names:
        return "only is empty — omit it to draw everything"
    if layer == "scene":
        known = {i["name"] for i in (ir.get("instances") or ())}
        kind = "instance"
    else:
        known = {p["name"] for p in (ir.get("parts") or ())}
        kind = "part"
    unknown = sorted(set(names) - known)
    if unknown:
        return (f"only: unknown {kind} name(s) {unknown}. "
                f"{kind.capitalize()}s: {sorted(known)}")
    return tuple(names)


def _scene_snapshot_bbox(ir: dict, on, ir_poses, catalog):
    """`on` against a SceneIR: `all`, an instance name, or a group name."""
    from ...sdk import _scene_kernel as _sk

    catalog = catalog or {}
    if on is None or on == "all":
        # The union over placed instances, which is what the scene draws. The
        # room is deliberately not included: it is a synthetic camera-dependent
        # cutaway, and framing on it would push every scene into the same
        # room-sized shot.
        return _sk.scene_bbox(ir, catalog)
    if isinstance(on, str):
        try:
            return _sk.instance_world_aabb(ir, catalog, on, poses=ir_poses)
        except KeyError:
            pass
        try:
            return _sk.group_bbox(ir, catalog, on)
        except KeyError:
            instances = sorted(i["name"] for i in (ir.get("instances") or ()))
            groups = sorted(g["name"] for g in (ir.get("groups") or ()))
            return (f"on: unknown instance or group name {on!r}. "
                    f"Instances: {instances}. Groups: {groups}")
    return None


def _snapshot_bbox(ir: dict, on, ir_poses, *, layer: str = "object",
                   catalog=None) -> "tuple | str":
    """Resolve a view spec's `on` to a world AABB, or return an error string.

    **The layer matters here.** Resolved through the object kernel, a scene's
    named `on` would ask a SceneIR for the `parts` it does not have and report
    an instance that is right there as unknown, and `on="all"` would fail in
    `global_bbox`. A scene's entities are instances and groups, and its boxes come from the
    scene kernel, which needs the catalog to reach the objects behind them.
    """
    from ...sdk import _object_kernel as _ok

    if layer == "scene":
        resolved = _scene_snapshot_bbox(ir, on, ir_poses, catalog)
        if resolved is not None:
            return resolved
    if on is None or on == "all":
        return _ok.global_bbox(ir, poses=ir_poses)
    if isinstance(on, str):
        try:
            return _ok.global_bbox(ir, part_name=on, poses=ir_poses)
        except KeyError:
            known = sorted(p["name"] for p in (ir.get("parts") or ()))
            return f"on: unknown part name {on!r}. Parts: {known}"
    # An explicit box `[[x0,y0,z0], [x1,y1,z1]]`, or a bare point `[x,y,z]`.
    # A point is what an anchor is: probe `anchor_position` and aim at it.
    try:
        items = list(on)
    except TypeError:
        return (f"on must be 'all', a part name, a point [x,y,z], or a bbox "
                f"[[x0,y0,z0],[x1,y1,z1]]; got {on!r}")
    try:
        if len(items) == 3 and not hasattr(items[0], "__len__"):
            pt = tuple(float(v) for v in items)
            return (pt, pt)
        lo = tuple(float(v) for v in items[0])
        hi = tuple(float(v) for v in items[1])
    except (TypeError, ValueError, IndexError):
        return (f"on must be 'all', a part name, a point [x,y,z], or a bbox "
                f"[[x0,y0,z0],[x1,y1,z1]]; got {on!r}")
    if len(lo) != 3 or len(hi) != 3:
        return f"on: an explicit bbox needs two 3-vectors, got {on!r}"
    return (lo, hi)


def _snapshot_views(ctx: ToolContext, ir: dict, views, *, ir_poses,
                    resolution, layer: str = "object",
                    catalog=None) -> "tuple | str":
    """Resolve view specs to (canonical_ids, custom_cameras, labels) or error.

    Returns `(canonical_ids, customs, order)` where `customs` is a list of
    `(label, CameraSpec)` and `order` is the caller's requested labels in
    order, so the payload can be assembled the way it was asked for rather
    than the way the two render requests happen to come back.
    """
    from ...render import CANONICAL_CAMERA_IDS, framed_camera, look_at_camera

    if isinstance(views, (str, dict)):
        views = [views]
    try:
        views = list(views)
    except TypeError:
        return f"views must be a view spec or a list of them, got {views!r}"
    if not views:
        return "views is empty — name at least one view"
    if len(views) > _SNAPSHOT_MAX_VIEWS:
        return (f"{len(views)} views requested, cap is {_SNAPSHOT_MAX_VIEWS}. "
                f"You read every image you ask for, so two or three that "
                f"answer the question beat a sweep.")

    canonical: list = []
    customs: list = []
    order: list = []
    for i, spec in enumerate(views):
        if isinstance(spec, str):
            if spec not in CANONICAL_CAMERA_IDS:
                return (f"views[{i}]: {spec!r} is not a canonical camera. "
                        f"Use one of {list(CANONICAL_CAMERA_IDS)}, or a dict "
                        f"with `on`/`azimuth`/`elevation`, or one with "
                        f"`eye`/`look_at`.")
            if spec in canonical:
                return f"views[{i}]: {spec!r} is named twice"
            canonical.append(spec)
            order.append(spec)
            continue
        if not isinstance(spec, dict):
            return (f"views[{i}] must be a canonical camera name or a dict, "
                    f"got {type(spec).__name__}")

        unknown = sorted(set(spec) - {"on", "azimuth", "elevation", "margin",
                                      "distance", "roll", "fov_deg", "eye",
                                      "look_at", "up", "label"})
        if unknown:
            return (f"views[{i}]: unknown key(s) {unknown}. A declarative "
                    f"view takes on/azimuth/elevation/margin/distance/roll/"
                    f"fov_deg/label; an explicit one takes "
                    f"eye/look_at/up/roll/fov_deg/label.")
        fov = float(spec.get("fov_deg", 60.0))

        if "eye" in spec or "look_at" in spec:
            if "eye" not in spec or "look_at" not in spec:
                return (f"views[{i}]: an explicit view needs both `eye` and "
                        f"`look_at`")
            if any(k in spec for k in ("on", "azimuth", "elevation",
                                       "margin", "distance")):
                return (f"views[{i}]: mixes an explicit camera "
                        f"(eye/look_at) with declarative framing "
                        f"(on/azimuth/elevation/margin) — pick one. The "
                        f"declarative form solves for the distance; giving an "
                        f"eye means you already did.")
            try:
                cam = look_at_camera(spec["eye"], spec["look_at"],
                                     spec.get("up"), fov_deg=fov,
                                     roll_deg=float(spec.get("roll", 0.0)))
            except ValueError as e:
                return f"views[{i}]: {e}"
            label = str(spec.get("label") or f"view{i + 1}")
        else:
            if "azimuth" not in spec or "elevation" not in spec:
                return (f"views[{i}]: a declarative view needs `azimuth` and "
                        f"`elevation` in degrees (azimuth 0 = +X, 90 = +Y "
                        f"which is the front; elevation 90 = straight down). "
                        f"Add `on` to frame on one part.")
            on = spec.get("on", "all")
            bbox = _snapshot_bbox(ir, on, ir_poses, layer=layer,
                                  catalog=catalog)
            if isinstance(bbox, str):
                return f"views[{i}]: {bbox}"
            dist = spec.get("distance")
            try:
                cam = framed_camera(
                    bbox,
                    azimuth_deg=float(spec["azimuth"]),
                    elevation_deg=float(spec["elevation"]),
                    fov_deg=fov,
                    margin=float(spec.get("margin", 0.05)),
                    resolution=resolution,
                    up=spec.get("up"),
                    roll_deg=float(spec.get("roll", 0.0)),
                    distance=(None if dist is None else float(dist)))
            except ValueError as e:
                return f"views[{i}]: {e}"
            base = on if isinstance(on, str) else "box"
            label = str(spec.get("label")
                        or f"{base}_az{float(spec['azimuth']):g}"
                           f"_el{float(spec['elevation']):g}")
        if label in order:
            return f"views[{i}]: duplicate label {label!r}"
        customs.append((label, cam))
        order.append(label)
    return canonical, customs, order


def run_snapshot(ctx: ToolContext, *, views=None, poses=None, only=None,
                 resolution=None, out_dir: str = "snapshots",
                 eye=None, look_at=None, up=None, fov_deg: float = 60.0,
                 frame=None, frames=None) -> dict:
    """Render the current geometry from cameras the caller chooses.

    The agent's own sight channel: it returns PNG paths and nothing looks at
    them but the caller. A custom camera's id is a hash of its geometry, so it
    never collides with a canonical view's (`look_at_camera`).

    `views` is one spec or a list, in three forms:

      * `"front"` — a canonical camera, framed exactly as `review()` frames it.
      * `{"on": "visor", "azimuth": 45, "elevation": 30}` — aim from a
        direction and let the harness solve for the distance that fits `on`.
        This is the normal case; see `render.framed_camera`.
      * `{"eye": [...], "look_at": [...]}` — a camera in world meters, for
        what the declarative form cannot say.

    `poses` moves joints for the render, in SDK units (degrees for revolute,
    meters for prismatic). Nothing else in the agent's tool surface can do
    this, so a collision's `witnessing_pose` is otherwise visible only through
    `probe`. `only` names the parts to draw and hides the rest, which is what
    a contact face or an interior fit needs. `resolution` is per call.

    `frame` / `frames` select frames of a committed animation and are refused
    on the other layers rather than ignored. That path takes the single-camera
    `eye`/`look_at` form.
    """
    from ...render import (RenderRequest, RenderUnavailable, render_request)

    # Frame-indexed (animation) layers serve snapshots from their own native
    # runtime and never reach this function, so a frame is always an error here.
    if frame is not None or frames is not None:
        return {"ok": False,
                "error": (f"the {ctx.layer_cfg.name} layer has no frame axis; "
                          "drop frame=/frames=")}

    legacy = eye is not None or look_at is not None
    if legacy and views is not None:
        return {"ok": False,
                "error": ("pass either `views` or the single-camera "
                          "`eye`/`look_at` form, not both")}
    if not legacy and views is None:
        return {"ok": False,
                "error": ("name what to render: `views=['front']`, or "
                          "`views=[{'on': '<part>', 'azimuth': 45, "
                          "'elevation': 30}]`, or an explicit "
                          "`eye`/`look_at`.")}
    resolved = _snapshot_out_dir(ctx, out_dir)
    if isinstance(resolved, str):
        return {"ok": False, "error": resolved}

    # ---- object / scene --------------------------------------------------
    # `_render_inputs` builds the IR from the CURRENT source and raises
    # `PermissionError` when that source does not build. Structured error, not
    # a traceback: the agent has just edited the file and needs to be told
    # which line broke, and the step is spent either way.
    try:
        layer, ir, catalog, _t, asset_base_dir = _render_inputs(ctx, "all")
    except PermissionError as e:
        return {"ok": False,
                "error": f"current source did not build: {e}"}
    except KeyError as e:
        return {"ok": False, "error": f"unknown target: {e}"}

    ir_poses, pose_label = (None, "default")
    if poses is not None:
        got = _snapshot_poses(ir, poses, catalog=catalog, layer=layer)
        if isinstance(got, str):
            return {"ok": False, "error": got}
        ir_poses, pose_label = got

    only_parts = _snapshot_only(ir, only, layer=layer)
    if isinstance(only_parts, str):
        return {"ok": False, "error": only_parts}

    if legacy:
        views = [{"eye": eye, "look_at": look_at, "up": up,
                  "fov_deg": fov_deg}]

    res = _snapshot_resolution(resolution)
    if isinstance(res, str):
        return {"ok": False, "error": res}

    resolved_views = _snapshot_views(ctx, ir, views, ir_poses=ir_poses,
                                     layer=layer, catalog=catalog,
                                     resolution=res)
    if isinstance(resolved_views, str):
        return {"ok": False, "error": resolved_views}
    canonical_ids, customs, order = resolved_views

    # Everything that can be refused has been. Only now touch the filesystem.
    resolved.mkdir(parents=True, exist_ok=True)

    def _request(**kw):
        return RenderRequest(
            layer=layer, ir=ir, catalog=catalog, target="all",
            out_dir=resolved, step=ctx.step, cache=ctx.snapshot_cache,
            asset_base_dir=asset_base_dir,
            poses=ir_poses, pose_label=pose_label,
            only=only_parts, resolution=res,
            png_compress_level=SNAPSHOT_PNG_COMPRESS_LEVEL, **kw)

    # At most two requests: one for the canonical preset, one for the custom
    # cameras. `RenderRequest` refuses both `cameras` and `camera_ids` in one
    # request by design, and each request is one Filament context — which is
    # also why every camera of a kind goes in a single request rather than one
    # request per camera (context construction is ~1030ms of a ~1160ms render).
    paths: dict = {}
    cams: dict = {}
    hits: list = []
    warnings: list = []
    bbox = None
    try:
        if canonical_ids:
            r = render_request(_request(camera_ids=tuple(canonical_ids)))
            for cid in canonical_ids:
                paths[cid] = r.render_paths[cid]
                cams[cid] = next(c for c in r.cameras if c.id == cid)
            hits.extend(r.cache_hits.values())
            warnings.extend(r.warnings)
            bbox = r.target_bbox
        if customs:
            r = render_request(_request(
                cameras=tuple(cam for _l, cam in customs)))
            for label, cam in customs:
                paths[label] = r.render_paths[cam.id]
                cams[label] = cam
            hits.extend(r.cache_hits.values())
            warnings.extend(r.warnings)
            bbox = bbox or r.target_bbox
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    except RenderUnavailable as e:
        return {"ok": False, "error": f"rendering unavailable: {e}"}
    except KeyboardInterrupt:
        raise
    except (ValueError, KeyError) as e:
        return {"ok": False, "error": f"snapshot failed: {e}",
                "evidence": {"detail": type(e).__name__}}

    note = None
    if legacy:
        note = _up_note(up, cams[order[0]])
    return _snapshot_payload(
        resolved, [(label, paths[label]) for label in order],
        cameras=cams, resolution=res, bbox=bbox, hits=hits,
        frame_indices=None, note=note,
        poses=(dict(poses) if poses else None), pose_label=pose_label,
        only=(list(only_parts) if only_parts else None),
        warnings=warnings)


def _up_note(up, camera) -> "str | None":
    """The one thing an agent must be told when it let `up` default."""
    if up is not None or tuple(camera.up) == (0.0, 0.0, 1.0):
        return None
    return (f"up was not given and +Z is degenerate for this near-vertical "
            f"view, so image-up is world {camera.up} — matching the "
            f"canonical {'top' if camera.up[1] < 0 else 'bottom'} camera.")


def _snapshot_payload(out_dir, labelled_paths, *, cameras, resolution, bbox,
                      hits, frame_indices, note, poses=None,
                      pose_label="default", only=None, warnings=None) -> dict:
    """One payload shape for every snapshot path.

    `views` carries the per-image detail and `paths` the bare list, in the
    order the caller asked for. The PNG filename is a camera-geometry hash for
    a custom camera, so it is `label` that says what a picture is of — read
    them together.
    """
    return {
        "ok": True,
        "paths": [str(p) for p in (p for _l, p in labelled_paths)],
        "n_images": len(labelled_paths),
        "out_dir": str(out_dir),
        "views": [
            {"label": label, "path": str(path),
             # 9 dp, not the 6 the rest of the harness rounds to for
             # legibility. A close-up on a sub-millimeter feature stands
             # 1e-4 m away, where rounding the eye to a micrometre is a 1%
             # error in the one number the next call aims from — measured at
             # 0.66% over a sweep of random poses down to 1e-4 m.
             "camera": {"id": cameras[label].id,
                        "eye": [round(v, 9) for v in cameras[label].eye],
                        "look_at": [round(v, 9) for v in cameras[label].target],
                        "up": list(cameras[label].up),
                        "fov_deg": cameras[label].fov_deg}}
            for label, path in labelled_paths
        ],
        # `fov_deg` is VERTICAL against this frame. An agent sizing a distance
        # from a horizontal extent against 60 deg frames ~44% too tight.
        "image_size": list(resolution),
        "poses": poses,
        "pose_label": pose_label,
        "only": only,
        "frame_indices": frame_indices,
        "target_bbox": ([list(bbox[0]), list(bbox[1])] if bbox else None),
        "cache_hit": bool(hits) and all(hits),
        # Non-fatal problems the renderer hit. TWO producers on this tree:
        # an asset that would not load or a primitive it cannot draw, either
        # rendered as a BOUNDING BOX — so an agent that reads a brick as its
        # own geometry goes off repairing something that was never wrong; and
        # an authored `@shader` body that raised, where the child reset to the
        # scalars and rendered anyway, so the picture is plausible and nothing
        # else says the graph never ran.
        #
        # Present only when there are some, which both lines' comments asked
        # for and neither implemented here. **Absence is therefore ambiguous:**
        # a fully cached render spawns no child and reports nothing, which
        # reads identically to a clean one.
        **({"render_warnings": list(warnings)} if warnings else {}),
        "note": note,
    }


__all__ = ["run_snapshot"]
