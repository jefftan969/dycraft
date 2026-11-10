"""Render the image evidence a visual review is shown.

`render_review_images` renders an object or scene IR at the canonical cameras
and, for an articulated object, packs its articulation evidence into contact
sheets. The review tool in `agent/layers/tools.py` sends the result to a
reviewing model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReviewImage:
    """One rendered image, with the label and optional note that say what it shows."""

    label: str
    path: Path
    note: str | None = None


@dataclass(frozen=True)
class ImageSet:
    """The ordered images a visual review is shown."""

    images: tuple[ReviewImage, ...] = ()

    @classmethod
    def of(cls, items: Iterable) -> "ImageSet":
        """Build from `(label, path)` or `(label, path, note)` tuples."""
        return cls(images=tuple(
            ReviewImage(str(item[0]), Path(item[1]), item[2] if len(item) > 2 else None)
            for item in items))

    @classmethod
    def from_render_result(cls, result) -> "ImageSet":
        """Build from a `render.RenderResult`, one image per camera in canonical order."""
        return cls.of(result.ordered_items())

    def extend(self, items: Iterable) -> "ImageSet":
        """A copy with `items` appended."""
        return ImageSet(images=self.images + ImageSet.of(items).images)

    def paths(self) -> list[Path]:
        return [image.path for image in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __bool__(self) -> bool:
        return bool(self.images)


#: Cameras carried into the articulation sheet, one row each at lo and hi.
#: `front` and `left` are the two most-cited views (44.1% and 27.3% of
#: defects); `top` is next after `back`, and `back` is dropped here because it
#: duplicates `front` for a pose comparison while costing a row.
_SHEET_POSE_CAMERAS = ("front", "top", "left")

# Every one of these renders is intermediate evidence that reaches the model
# only through a composited sheet. PIL's default `compress_level=6` costs
# ~235 ms/frame against ~37 ms at level 1 for ~25% more bytes, and
# `render_cache_key` is keyed on the request, not on the image bytes.
_JUDGE_PNG_COMPRESS_LEVEL = 1


def _sheet_evidence(evidence_dir, default_result, articulated,
                    lo_poses, hi_poses, run):
    """The two-sheet payload: default views, and articulation if there is any.

    Sheet A is the canonical six at the authored pose. Sheet B is
    `_SHEET_POSE_CAMERAS` at both all-limits poses, laid out so each ROW is one
    camera at lo beside the same camera at hi — the comparison the reviewer is
    actually being asked to make. Grouping by pose instead would put the two
    halves of every comparison in different rows.

    Framing stays fixed to the default-pose silhouette for sheet B: a part that
    swings past the resting outline crowds its panel, and that is informative
    rather than a fault (a drawer floating clear of its cabinet at `hi` means
    the prismatic limit exceeds the bin depth). The three panels of a row are only comparable if
    they share a camera.

    `evidence_dir` is supplied by the caller rather than derived from the step
    here, because the step does not identify a render when one op produces
    several — see `render_review_images`.

    Returns the same 4-tuple as `render_review_images`.
    """
    from ._vlm_evidence import build_view_sheet

    sheet_a = build_view_sheet(
        [(cid, path) for cid, path in default_result.ordered_items()],
        evidence_dir / "default_views.png")
    images = ImageSet.of([(
        "default_views", sheet_a,
        "one sheet, six panels labelled front / right / back / left / top / "
        "bottom, at the model's authored joint pose")])
    layout = {"kind": "view_sheets", "default_views": 6}

    if not articulated:
        return images, False, layout, None

    # Only the cameras that reach the sheet are rendered at the limits.
    lo = run("lo", lo_poses, camera_ids=_SHEET_POSE_CAMERAS)
    hi = run("hi", hi_poses, camera_ids=_SHEET_POSE_CAMERAS)
    rows: list = []
    for cid in _SHEET_POSE_CAMERAS:
        if cid in lo.render_paths and cid in hi.render_paths:
            rows.append((f"{cid}_lo", lo.render_paths[cid]))
            rows.append((f"{cid}_hi", hi.render_paths[cid]))
    if rows:
        sheet_b = build_view_sheet(rows,
                                   evidence_dir / "articulation_views.png")
        images = images.extend([(
            "articulation_views", sheet_b,
            "one sheet, "
            + ", ".join(lbl for lbl, _ in rows)
            + " — every non-fixed joint at the matching limit simultaneously, "
              "each row one camera at lo beside hi")])
        layout["articulation_views"] = [lbl for lbl, _ in rows]
    return images, True, layout, None


def render_review_images(ir: dict, ctx, layer: str, *, evidence_key: str = ""):
    """Render the model for a visual review.

    `evidence_key` namespaces every file this call writes, and a caller that renders more than one model under one op
    must pass it. Without it the output is named by step alone (`renders/<step>_<camera>.png` from
    `_world.plan_outputs`, and `renders/<step>_vlm_views/` here), and a step is one op, not one render: `run_review`
    loops its targets with one `ctx`, so each target would overwrite the previous one's PNGs and sheets, and the
    reviewer would judge one model's pictures under every target's name. `ctx.step` can also advance mid-review, so
    the step is not a stable name even for one target.

    Returns `(images, articulated, evidence_layout, render_unavailable_reason | None)`. `images` is an `ImageSet`
    whose every image carries the label the prompt legend uses, so the legend is generated from it.

    Both layers render the six canonical views (front, right, back, left, top, bottom). An object's views are packed
    into sheets (`_sheet_evidence`): the six defaults and, for an articulated object, `_SHEET_POSE_CAMERAS` at both
    all-limit poses. An articulated scene instead gets four extra all-joint limit views, because its joints live
    across a catalog of instanced ObjectIRs rather than one object graph.

    `evidence_layout["render_warnings"]` lists anything non-fatal the renderer reported (an authored shader that
    raised, an AABB substituted for an asset that would not load), which the caller lifts into the report so the agent
    reads it beside the critique it probably caused.
    """
    from ..render import (CARD_RESOLUTION, FULL_RESOLUTION, RenderRequest,
                          RenderUnavailable, render_request)
    from ..agent.layers.tools import RENDERS_DIR
    from . import _render_pose as _rp

    # A subdirectory per target when one is named, so the `<step>_<camera>`
    # filenames underneath cannot collide across targets of the same op.
    out_dir = ctx.run_dir / RENDERS_DIR
    if evidence_key:
        out_dir = out_dir / f"target_{evidence_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = (ctx.compile_artifact_step()
            if hasattr(ctx, "compile_artifact_step")
            else getattr(ctx, "step", 0))
    sheet_dir = out_dir / f"{step}_vlm_views"
    if layer == "object":
        catalog = None
        articulated = _rp.has_articulation(ir)
        lo_poses = _rp.lo_pose(ir) if articulated else None
        hi_poses = _rp.hi_pose(ir) if articulated else None
    else:
        catalog = getattr(ctx.layer_payload, "catalog", {}) or {}
        articulated = _rp.scene_has_articulation(ir, catalog)
        lo_poses = _rp.scene_lo_pose(ir, catalog) if articulated else None
        hi_poses = _rp.scene_hi_pose(ir, catalog) if articulated else None

    render_cache = getattr(ctx, "render_cache", None)
    #: Non-fatal renderer complaints, accumulated across every pose render. The
    #: review is usually what sends an agent back to a material, so a shader
    #: body that raised has to be legible in the same report as the critique it
    #: caused — otherwise the agent is told the surface looks flat and given no
    #: way to learn that its shader never ran.
    render_warnings: list = []

    def _run(pose_label: str, poses: dict | None,
             resolution=FULL_RESOLUTION, camera_ids=None):
        result = render_request(RenderRequest(
            layer=layer, ir=ir, catalog=catalog, target="all",
            out_dir=out_dir, step=ctx.step,
            cache=render_cache,
            png_compress_level=_JUDGE_PNG_COMPRESS_LEVEL,
            # Object-layer `solid` parts store run-relative asset paths
            # (`assets/meshes/<digest>.obj`). Without this the renderer resolves
            # them against the process cwd, fails, and silently falls back to
            # each part's bounding box — so the reviewer would see boxes and
            # fault objects that are correct.
            asset_base_dir=(
                ctx.run_dir if layer == "object" else
                getattr(ctx.layer_payload, "asset_base_dir", None)
            ),
            poses=poses, pose_label=pose_label,
            resolution=resolution,
            # Six canonical cameras unless a caller narrows it. The sheet path
            # renders only the cameras that reach the sheet.
            camera_ids=camera_ids,
        ))
        for w in result.warnings:
            if w not in render_warnings:
                render_warnings.append(w)
        return result

    try:
        default_result = _run("default", None)
        images = ImageSet.from_render_result(default_result)

        if layer == "object":
            sheet_images, sheet_art, sheet_layout, sheet_err = _sheet_evidence(
                sheet_dir, default_result, articulated,
                lo_poses, hi_poses, _run)
            # Stamped here because `_sheet_evidence` has no access to the
            # accumulator; its own lo/hi renders go through `_run`, so their
            # warnings are already in the list. Without it a shader that never
            # ran would be invisible to the reviewer.
            if render_warnings and sheet_err is None:
                sheet_layout = {**sheet_layout,
                                "render_warnings": render_warnings}
            return sheet_images, sheet_art, sheet_layout, sheet_err

        if articulated:
            lo_result = _run("lo", lo_poses, CARD_RESOLUTION)
            hi_result = _run("hi", hi_poses, CARD_RESOLUTION)
            # Pose-grouped extras: front_lo, top_lo, front_hi, top_hi.
            extras = []
            for pose_label, result in (("lo", lo_result),
                                       ("hi", hi_result)):
                for cam_id in ("front", "top"):
                    if cam_id in result.render_paths:
                        extras.append((
                            f"{cam_id}_{pose_label}",
                            result.render_paths[cam_id],
                            f"{cam_id} camera with every non-fixed joint "
                            f"at its `{pose_label}` limit",
                        ))
            images = images.extend(extras)
            layout = {"kind": "all_limit_views"}
        else:
            layout = {"kind": "default_views"}
        if render_warnings:
            # Rides in `layout` rather than as a fifth return value: the
            # review tool and several tests unpack this four-tuple, and a
            # shape change would break every one of them. `layout` is also the
            # sanctioned channel for extra structured facts about a render — it
            # is copied verbatim into the report evidence — so this rides along
            # for free.
            layout = {**layout, "render_warnings": render_warnings}
        return images, articulated, layout, None
    except RenderUnavailable as e:
        return ImageSet(), False, {}, f"rendering unavailable: {e}"
    except BaseException as e:  # noqa: BLE001
        return ImageSet(), False, {}, f"render error: {type(e).__name__}: {e}"


__all__ = ["render_review_images"]
