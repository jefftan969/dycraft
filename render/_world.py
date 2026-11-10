"""Backend-free half of the renderer: geometry, framing, materials, cache.

Everything here answers a question that has nothing to do with *how* pixels get
made — where each part ends up in world space, which cameras look at it, what
surface it wears, which PNG a given (ir, camera, pose) already wrote. It was
extracted verbatim from `open3d_renderer.py` when the Blender backend arrived,
because a second backend that re-derived any of it would drift from the first
and the two pictures would disagree about the scene rather than about its
lighting.

The split line is exact: nothing in this module imports `open3d` or `bpy`.
`open3d_renderer` re-exports the names its importers (several test modules)
reach for.

Three things a backend must route through rather than reimplement:

  `_prepare(request) -> _Plan`   world parts + room + cameras + framing bbox
  `plan_outputs(request, plan)`  the per-camera cache key, PNG filename, and
                                 hit/miss decision — get this wrong and a
                                 sequence of frames hardlinks to frame 0
  `room_wall_boxes(camera, room)` which room surfaces this camera may see

See `cli/README.md`'s co-update checklist for what else moves when this does.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np

from ..sdk import _rotation as _rot
from ..sdk import _primitives as _prim
from ..sdk import object_query as _oq
from ..sdk import scene_query as _sq
from ..sdk import _scene_kernel as _sk
from ..sdk import scene_builder as _sb
from ..sdk._material import (
    Material,
    material_from_ir,
    DEFAULT_ALPHA,
    DEFAULT_METALLIC,
    DEFAULT_PART_COLOR,
    DEFAULT_ROUGHNESS,
    DEFAULT_FLOOR_COLOR,
    DEFAULT_WALL_COLOR,
)
from . import (
    CANONICAL_CAMERA_IDS,
    CameraSpec,
    RenderRequest,
    _cameras_for_request,
    render_cache_key,
)


#: Thickness of a synthesized shell surface, meters. The surface's inner face
#: is flush with the room AABB, so the clear interior volume is exactly the
#: authored width x depth x height whatever this is.
_SHELL_THICKNESS_M = 0.01


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _color_255_to_float(c):
    """Convert an IR [r, g, b] (ints in [0, 255]) to (r, g, b) floats in
    [0, 1]."""
    return (float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0)


def _material_to_render(mat: Material, base_dir=None):
    """Pack a Material into the renderer-side dict form (color floats in
    [0, 1] plus PBR scalars, and a resolved texture when it has one).

    The single packing point for every material the renderer draws — parts and
    the room's floor/walls both come through here — so the texture rides in
    this dict rather than in a new `_WorldPart` slot, and every consumer that
    already receives a material receives it for free.

    `base_dir` resolves each map's run-relative path, exactly as
    `_resolve_mesh_params` resolves a mesh part's. Scene renders pass the
    staged catalog root, so a placed instance finds its images by the same
    route its geometry does.

    `texture` stays a plain dict of kind to map, mirroring the IR: base color
    is a kind like any other, and the kinds are carried through verbatim.
    `_pbr_material` is the only place that knows what a kind MEANS, because
    that is Filament's vocabulary rather than ours — a kind it does not know
    travels everywhere and is simply not sampled, which is the right
    degradation.

    `mapping` rides beside it rather than inside, because a tile size is not a
    property of any image. It reaches exactly one consumer, the projection, and
    only for geometry that has no staged mesh.

    `shader` rides beside both, and only Cycles reads it. The four scalars are
    emitted unconditionally even under a shader, precisely so the backends that
    cannot evaluate one — and the Cycles fallback when a shader raises — still
    have a surface to draw.
    """
    # **All four scalars, always — the packed dict is a total contract even
    # though the IR's is not.** `Material.to_ir()` omits a scalar a shader
    # material's author never wrote, and `material_from_ir` preserves that
    # absence so an IR survives a round trip unchanged. Here the job is the
    # opposite: `_pbr_material` and `_build_material` subscript these keys
    # directly, so an absent one would be a `KeyError` inside a renderer rather
    # than the fallback it is meant to be.
    #
    # **So this is where the renderer's scalar approximation is decided**, and
    # that is the right boundary for it: what a backend draws when it cannot
    # evaluate a shader — and what Cycles draws when one raises — is a renderer
    # question, not a statement about what the author wrote.
    out = {
        "color": _color_255_to_float(
            mat.color if mat.color is not None else DEFAULT_PART_COLOR),
        "metallic": (mat.metallic if mat.metallic is not None
                     else DEFAULT_METALLIC),
        "roughness": (mat.roughness if mat.roughness is not None
                      else DEFAULT_ROUGHNESS),
        "alpha": mat.alpha if mat.alpha is not None else DEFAULT_ALPHA,
    }
    texture = getattr(mat, "texture", None)
    if texture is not None:
        out["texture"] = {
            kind: {"path": _resolve_asset_path(m.image, base_dir),
                   "digest": m.digest,
                   **({"channel": m.channel} if m.channel else {}),
                   **({"strength": m.strength} if m.strength is not None else {})}
            for kind, m in texture.maps.items()
        }
    mapping = getattr(mat, "mapping", None)
    if mapping is not None:
        out["mapping"] = {"scale_m": mapping.scale_m}
    shader = getattr(mat, "shader", None)
    if shader is not None:
        # Cycles-only, and it rides here rather than in a new `_WorldPart` slot
        # for the same reason `texture` does: every consumer that already
        # receives a material receives it for free, and a backend that cannot
        # evaluate a node graph simply does not look — the same degradation an
        # unknown map kind gets.
        out["shader"] = shader.to_ir()
    emission = getattr(mat, "emission", None)
    if emission is not None:
        # Color packed to floats like `color`, so a backend never has to know
        # that one of the two was authored in bytes and the other was not.
        out["emission"] = {"color": _color_255_to_float(emission.color),
                           "strength": emission.strength}
    return out


# -------------------------------------------------------------------------
# Request planning
# -------------------------------------------------------------------------

class _Plan(NamedTuple):
    """Everything a request needs decided before Filament is involved."""

    world_parts: list
    room_info: dict | None
    cameras: list
    target_bbox: tuple
    ir_hash: str
    catalog_fp: object


def _prepare(request: RenderRequest) -> _Plan:
    """Steps 1–3: world geometry, camera framing, camera list.

    Pure CPU — deliberately *not* on the Filament thread, which is a single
    shared worker for the whole process and should only ever be doing GL work.
    """
    request.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the world-space part box-list + room (for scenes), at
    # the request's joint pose.
    world_parts, room_info = _build_world_geometry(request)

    # 2. Compute the camera-framing bbox from the IR's default pose so
    # that renders at non-default poses share the framing of the
    # default-pose views.
    #
    # `base_deltas` counts as a non-default pose too. It did not when this guard
    # was written — the field did not exist — so an animation frame driven purely
    # by base deltas (no joints) framed the camera on the *moving* geometry, and
    # every frame of a clip got its own camera. Visible as the whole set drifting
    # and rescaling under an actor that leaves the scene's resting bbox: a mug
    # tossed 0.5 m up pulled the camera back and tilted it, frame by frame.
    # With caller-supplied `cameras` the framing bbox positions nothing, so the
    # default-pose world would be rebuilt purely to compute a number only
    # `target_bbox` reporting reads.
    if (request.cameras is not None
            or (request.poses is None and request.base_deltas is None)):
        default_world_parts = world_parts
    else:
        default_world_parts, _ = _build_world_geometry_at_defaults(request)
    target_bbox = _resolve_target_bbox(request, default_world_parts)
    target_center, target_extent_xy, target_extent_z = _bbox_metrics(target_bbox)

    # 3. Compute cameras. The default remains the universal five-view set;
    # callers such as the review renders can explicitly opt into bottom.
    cameras = _cameras_for_request(
        request, target_center, target_extent_xy, target_extent_z,
        target_bbox)

    return _Plan(world_parts, room_info, cameras, target_bbox,
                 request.ir["id"], _catalog_fingerprint(request))


# -------------------------------------------------------------------------
# World geometry composition
# -------------------------------------------------------------------------

class _WorldPart:
    __slots__ = ("name", "size", "se3", "material", "primitive", "params",
                 "center", "deformed")

    def __init__(self, name, size, se3, material, primitive="box", params=None,
                 center=(0.0, 0.0, 0.0), deformed=None):
        # `deformed` is `(vertices, triangles)` in the part's LOCAL frame,
        # replacing whatever its asset file holds. `se3` still places it, so a
        # deformed part travels the same FK path as a rigid one.
        self.deformed = deformed
        self.name = name
        # `size` is the full AABB extents (w, l, h) of the primitive's bounding
        # box: box -> (w, l, h); solid -> its stored AABB extents.
        # Used for framing/AABB. `primitive`+`params` drive exact mesh shape.
        self.size = size
        self.se3 = se3              # ((tx,ty,tz),(qw,qx,qy,qz))
        self.primitive = primitive
        self.params = params or {}
        # `center` is the part-local center of the AABB whose extents are
        # `size` (the scaled local-AABB midpoint). It is (0,0,0) for the
        # origin-centered analytic primitives, and non-zero only for a mesh
        # authored away from its local origin. Framing (`_world_box_aabb`) and the missing-asset fallback box
        # (`_box_mesh`) use it so both sit where the real geometry actually is,
        # rather than assuming the box straddles the part origin.
        self.center = center
        # material is the renderer-side packed dict from
        # `_material_to_render`: keys color (floats), metallic, roughness, alpha.
        self.material = material


def _build_world_geometry(request: RenderRequest):
    """Return (world_parts: list[_WorldPart], room_info_or_None).

    The geometry is composed at `request.poses` if provided, else at
    the IR's authored joint defaults.

    room_info, when present, is a dict
    {aabb, floor, wall, ceiling, suppressed}. `floor`, `wall`, and `ceiling`
    are renderer-side material dicts (see `_material_to_render`); `ceiling` is
    None for a room that never named one, and `suppressed` names the surfaces
    the scene replaced with authored geometry.
    """
    if request.layer == "object":
        return _build_object_geometry(
            request.ir, poses=request.poses,
            base_dir=request.asset_base_dir,
            mount_deltas=request.mount_deltas,
            deformed=request.deformed,
            only=request.only), None
    if request.layer == "scene":
        # On the scene layer `only` names instances, which is what
        # `include_instances` already filters — so it composes with, rather than
        # duplicates, the animation layer's exists-set. Intersecting means an
        # isolated render of an instance that this frame has already removed
        # stays empty instead of resurrecting it.
        include = request.include_instances
        if request.only is not None:
            include = (set(request.only) if include is None
                       else set(include) & set(request.only))
        return _build_scene_geometry(request.ir, request.catalog,
                                     poses=request.poses,
                                     base_deltas=request.base_deltas,
                                     base_dir=request.asset_base_dir,
                                     mount_deltas=request.mount_deltas,
                                     deformed=request.deformed,
                                     include_instances=include,
                                     enclosure_override=getattr(
                                         request, "room_enclosure", None))
    raise ValueError(f"unknown layer: {request.layer!r}")


def _build_world_geometry_at_defaults(request: RenderRequest):
    """Same as `_build_world_geometry` but with no pose or delta channels.

    Used to compute the camera-framing bbox so that lo/hi pose renders share
    the framing of the default views. "Defaults" means the object layer's joint
    defaults and, at the scene layer, each instance's **committed**
    configuration — the same thing every other `poses=None` query answers, so
    a lo/hi pair of an articulated instance is framed against the body as the
    scene delivers it.
    """
    if request.layer == "object":
        return _build_object_geometry(
            request.ir, base_dir=request.asset_base_dir,
            only=request.only), None
    if request.layer == "scene":
        include = request.include_instances
        if request.only is not None:
            include = (set(request.only) if include is None
                       else set(include) & set(request.only))
        return _build_scene_geometry(request.ir, request.catalog,
                                     base_dir=request.asset_base_dir,
                                     include_instances=include,
                                     enclosure_override=getattr(
                                         request, "room_enclosure", None))
    raise ValueError(f"unknown layer: {request.layer!r}")


def _resolve_asset_path(rel, base_dir):
    """Resolve one run-relative asset path against ``base_dir``.

    The rule `_resolve_mesh_params` has always applied, lifted out because a
    material is not a params dict and must not grow its own second rule:
    returns the value unchanged when there is no base dir, the path is already
    absolute, or it is not a string.
    """
    if base_dir is None or not isinstance(rel, str) or os.path.isabs(rel):
        return rel
    return str(Path(base_dir) / rel)


def _resolve_mesh_params(params: dict, base_dir, key: str = "filename") -> dict:
    """Resolve an asset part's run-relative geometry path against ``base_dir``.

    The object layer stores the path relative to the run dir; the renderer
    reads it from ``base_dir`` (the run dir). ``key`` is the params key holding
    it — ``"filename"`` for a ``mesh`` part, ``"mesh"`` for a ``solid`` (see
    ``_prim.render_asset_key``). Returns ``params`` unchanged when there is no
    base dir, the path is already absolute, or it is not a string; otherwise a
    shallow copy with an absolute path. Never mutates the IR's params dict.
    """
    fn = params.get(key)
    if base_dir is None or not isinstance(fn, str) or os.path.isabs(fn):
        return params
    out = dict(params)
    out[key] = str(Path(base_dir) / fn)
    return out


def _build_object_geometry(ir: dict, world_pose=None,
                            poses: dict | None = None,
                            base_dir=None,
                            mount_deltas: dict | None = None,
                            deformed: dict | None = None,
                            only: tuple | None = None) -> list[_WorldPart]:
    """List of per-part WorldParts, optionally composed with a world_pose
    and evaluated at a given joint pose.

    `poses=None` evaluates at the IR's authored joint defaults; a dict
    overrides individual joint values along each part's ancestor chain.
    `base_dir` resolves a mesh part's run-relative filename (see
    `_resolve_mesh_params`).

    `mount_deltas` (`{joint: se3}`) and `deformed` (`{part: (verts, tris)}`)
    carry the animation layer's deformation. They are independent: a part can
    deform without moving any child frame (nothing attached to it), and a mount
    can move without that part being the one drawn deformed.

    `only` restricts which parts are DRAWN, by name. Forward kinematics still
    runs over the whole tree — a hidden parent has to keep placing its visible
    children — so this filters the draw list and nothing else. An unknown name
    raises: a silently-ignored typo would draw the whole object and read as
    "isolation does not work".
    """
    frames = _oq._compute_frames(ir, poses=poses, mount_deltas=mount_deltas)
    if only is not None:
        wanted = set(only)
        known = {p["name"] for p in ir["parts"]}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(
                f"only: unknown part name(s) {unknown}; this object has "
                f"{sorted(known)}")
    parts = []
    base = world_pose or _rot.IDENTITY_SE3
    for p in ir["parts"]:
        if only is not None and p["name"] not in wanted:
            continue
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _prim.local_aabb(
            p["primitive"], p["params"])
        size = (xmx - xmn, ymx - ymn, zmx - zmn)
        # Part-local AABB center: (0,0,0) for the origin-centered analytic
        # primitives, the scaled-AABB midpoint for an off-center mesh.
        center = (0.5 * (xmn + xmx), 0.5 * (ymn + ymx), 0.5 * (zmn + zmx))
        local_se3 = frames[p["name"]]
        world_se3 = _rot.se3_compose(base, local_se3)
        mat = material_from_ir(p.get("material"),
                               default_color=DEFAULT_PART_COLOR)
        params = p["params"]
        _asset_key = _prim.render_asset_key(p["primitive"])
        if _asset_key is not None:
            params = _resolve_mesh_params(params, base_dir, _asset_key)
        parts.append(_WorldPart(
            name=f"{ir.get('name', 'object')}/{p['name']}",
            size=size,
            se3=world_se3,
            material=_material_to_render(mat, base_dir),
            primitive=p["primitive"],
            params=params,
            center=center,
            deformed=(deformed or {}).get(p["name"]),
        ))
    return parts


def _instance_deformed(deformed, inst_name: str):
    """Narrow a scene-wide `{"<inst>.<part>": (v, t)}` map to one instance.

    Returns `{part: (v, t)}` or None. Unqualified keys are not accepted: unlike
    a `poses` dict, which is meaningfully broadcast to every instance carrying
    that joint name, deformed vertices belong to exactly one part of exactly
    one instance and a bare part name would silently draw one body's geometry
    on another's.
    """
    if not deformed:
        return None
    prefix = inst_name + "."
    out = {k[len(prefix):]: v for k, v in deformed.items()
           if k.startswith(prefix)}
    return out or None


def _build_scene_geometry(ir: dict, catalog: Mapping[str, dict],
                           poses: dict | None = None,
                           base_dir=None, base_deltas: dict | None = None,
                           mount_deltas: dict | None = None,
                           deformed: dict | None = None,
                           include_instances: set[str] | None = None,
                           enclosure_override: str | None = None):
    """Build per-instance WorldParts and a room AABB.

    `poses` is a scene-level pose dict; each instance receives only the
    subset of keys that apply to it (qualified `"<inst>.<joint>"` keys
    matching this instance, plus any unqualified keys that match a
    joint name in its catalog object).

    `poses` and `base_deltas` both reach `_resolve_world_pose`. Passing `poses`
    there matters beyond this instance's own articulation: it is what makes an
    instance attached to a *moving part* of its parent ride that part
    (kinematic-follow). Until this was threaded, the renderer resolved every
    instance world pose at rest while drawing each object's own parts posed —
    so the kernel and the picture disagreed about where an attached child was.

    `base_dir` resolves run-relative mesh filenames on catalog objects (a
    native-mesh object placed into a scene), forwarded to
    `_build_object_geometry`. Scene callers pass the staged catalog root, whose
    content-addressed ``assets/`` tree contains the referenced object files.
    """
    parts: list[_WorldPart] = []
    memo: dict = {}
    for inst in ir["instances"]:
        if (include_instances is not None
                and inst["name"] not in include_instances):
            continue
        world_pose = _sq._resolve_world_pose(ir, catalog, inst["name"], memo,
                                             poses=poses,
                                             base_deltas=base_deltas)
        ref = inst["object_ref"]
        obj_ir = catalog[ref]
        # `effective_object_poses`, not the bare projection: a joint this clip
        # does not drive is drawn at the configuration the scene committed for
        # THIS instance, so the picture shows the cabinet the author placed
        # open rather than the definition's rest angle. With no saved state and
        # no poses it still answers None, exactly as before.
        obj_poses = _sk.effective_object_poses(ir, inst["name"], poses)
        # Both deformation channels are keyed by instance up here and by joint /
        # part name down there, so each object only ever sees its own — passing
        # the scene-wide dict through would trip `_compute_frames`'s
        # unknown-joint check, which is exactly what that check is for.
        obj_mounts = (mount_deltas or {}).get(inst["name"]) or None
        obj_deformed = _instance_deformed(deformed, inst["name"])
        sub_parts = _build_object_geometry(obj_ir, world_pose=world_pose,
                                            poses=obj_poses, base_dir=base_dir,
                                            mount_deltas=obj_mounts,
                                            deformed=obj_deformed)
        # Re-tag with instance name for trace clarity.
        for sp in sub_parts:
            sp.name = f"{inst['name']}/{sp.name.split('/', 1)[-1]}"
        parts.extend(sub_parts)
    room = ir.get("room")
    if room is None:
        return parts, None
    w = float(room["width"])
    d = float(room["depth"])
    h = float(room["height"])
    room_aabb = ((-w / 2, -d / 2, 0.0), (w / 2, d / 2, h))
    floor_mat = material_from_ir(room.get("floor_material"),
                                 default_color=DEFAULT_FLOOR_COLOR)
    wall_mat = material_from_ir(room.get("wall_material"),
                                default_color=DEFAULT_WALL_COLOR)
    # `base_dir`, like every other material packed here. The room's surfaces
    # carry run-relative texture paths exactly as a part's do. Without the base
    # they would resolve against the process CWD instead of the run or catalog
    # root, and fail the way a missing texture always fails here:
    # `_usable_texture` returns None, the surface falls back to its flat color,
    # and nothing says so. `_material_to_render` is the single packing point for parts and room
    # alike, so it takes the base at every call or at none.
    # A ceiling material is optional in the IR: a room authored before ceilings
    # existed has none, and `room_wall_boxes` falls back to the wall's.
    # Enclosure: the scene's own, or the render-time override for a run
    # authored before enclosure existed. Never inferred — see
    # `RenderRequest.room_enclosure`.
    enclosure = enclosure_override or room.get("enclosure")
    suppressed = tuple(dict.fromkeys(
        tuple(room.get("suppressed_surfaces") or ())
        + tuple(_sb.room_surfaces_suppressed_by(enclosure))))

    ceiling_ir = room.get("ceiling_material")
    ceiling_mat = (material_from_ir(ceiling_ir, default_color=DEFAULT_WALL_COLOR)
                   if ceiling_ir is not None else None)
    return parts, {"aabb": room_aabb,
                   "floor": _material_to_render(floor_mat, base_dir),
                   "wall": _material_to_render(wall_mat, base_dir),
                   "ceiling": (_material_to_render(ceiling_mat, base_dir)
                               if ceiling_mat is not None else None),
                   # Surfaces that are not drawn at all, visibly or
                   # otherwise — see `room_wall_boxes`. Two sources, unioned:
                   # what the scene replaced with authored geometry, and what
                   # its enclosure never had in the first place.
                   "suppressed": suppressed,
                   # Surface -> the instance the scene declared as its
                   # authored replacement. `suppressed` says a shell surface
                   # is not drawn; THIS says what stands there instead, which
                   # is what `room_cutaway_planes` needs to cut the authored
                   # geometry away from a camera the way the shell's own
                   # `camera_visible` cuts the shell.
                   "replacements": dict(room.get("surface_replacements") or {}),
                   # Whether anything caps the room. The interior rig assumes
                   # walls and a lid to bounce off; an open-air extent has
                   # neither, so it is lit like a subject instead.
                   "capped": "ceiling" not in suppressed}


# -------------------------------------------------------------------------
# Target framing
# -------------------------------------------------------------------------

def _resolve_target_bbox(request: RenderRequest, world_parts: list[_WorldPart]):
    """Compute the bbox used to position cameras."""
    target = request.target
    # camera_id targets: use 'all' framing.
    if target in CANONICAL_CAMERA_IDS or target == "all":
        return _aabb_from_world_parts(world_parts)
    if request.layer == "object":
        # part name
        try:
            return _oq.global_bbox(request.ir, part_name=target)
        except KeyError:
            raise ValueError(f"unknown part name: {target!r}")
    if request.layer == "scene":
        # try instance, then group
        try:
            return _sq.instance_bbox(request.ir, request.catalog, target)
        except KeyError:
            pass
        try:
            return _sq.group_bbox(request.ir, request.catalog, target)
        except KeyError:
            raise ValueError(f"unknown instance/group name: {target!r}")
    raise ValueError(f"unable to resolve target: {target!r}")


def _aabb_from_world_parts(world_parts: list[_WorldPart]):
    if not world_parts:
        return ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5))
    xs_min, ys_min, zs_min = [], [], []
    xs_max, ys_max, zs_max = [], [], []
    for p in world_parts:
        bb = _world_box_aabb(p)
        xs_min.append(bb[0][0]); ys_min.append(bb[0][1]); zs_min.append(bb[0][2])
        xs_max.append(bb[1][0]); ys_max.append(bb[1][1]); zs_max.append(bb[1][2])
    return ((min(xs_min), min(ys_min), min(zs_min)),
            (max(xs_max), max(ys_max), max(zs_max)))


def _world_box_aabb(p: _WorldPart):
    w, l, h = p.size
    # Corners of the part-local AABB [center ± size/2] — for an off-center mesh
    # this is its true local AABB, not a box assumed centered at the origin.
    cx, cy, cz = p.center
    corners_local = (
        (cx - w / 2, cy - l / 2, cz - h / 2), (cx + w / 2, cy - l / 2, cz - h / 2),
        (cx - w / 2, cy + l / 2, cz - h / 2), (cx + w / 2, cy + l / 2, cz - h / 2),
        (cx - w / 2, cy - l / 2, cz + h / 2), (cx + w / 2, cy - l / 2, cz + h / 2),
        (cx - w / 2, cy + l / 2, cz + h / 2), (cx + w / 2, cy + l / 2, cz + h / 2),
    )
    t, q = p.se3
    xs, ys, zs = [], [], []
    for c in corners_local:
        rx, ry, rz = _rot.quat_rotate_vec(q, c)
        xs.append(t[0] + rx); ys.append(t[1] + ry); zs.append(t[2] + rz)
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _bbox_metrics(bbox):
    (xmn, ymn, zmn), (xmx, ymx, zmx) = bbox
    center = (0.5 * (xmn + xmx), 0.5 * (ymn + ymx), 0.5 * (zmn + zmx))
    extent_xy = 0.5 * max(xmx - xmn, ymx - ymn)
    extent_z = 0.5 * (zmx - zmn)
    return center, extent_xy, extent_z


# -------------------------------------------------------------------------
# UV projection for geometry that carries none
# -------------------------------------------------------------------------

#: Triangulation of an axis-aligned box, as 12 triangles over the 8 corners
#: `(±x, ±y, ±z)` in the bit order `i = 4*x + 2*y + z`. Outward-facing winding.
_BOX_TRIS = np.array([
    (0, 2, 1), (1, 2, 3),        # -X
    (4, 5, 6), (5, 7, 6),        # +X
    (0, 1, 4), (1, 5, 4),        # -Y
    (2, 6, 3), (3, 6, 7),        # +Y
    (0, 4, 2), (2, 4, 6),        # -Z
    (1, 3, 5), (3, 5, 7),        # +Z
], dtype=np.int32)


def box_mesh_arrays(size, center=(0.0, 0.0, 0.0)):
    """`(vertices, triangles)` for a box of `size`, centered at `center`.

    Pure numpy so a backend that has no mesh library in the parent process can
    still build the one shape the renderer synthesizes rather than loads: an
    analytic `box` part, an asset's AABB fallback, and the room's floor and
    walls. In the part's LOCAL frame — the caller applies the world SE3, which
    is the only correct order for anything that then gets UVs (see
    `_triplanar_uvs`).

    Open3D does NOT use this; it keeps `TriangleMesh.create_box`, whose exact
    vertex order every stored render was drawn with. The two agree on the
    shape, which is all a second backend needs — and all a UV projection needs,
    since each face of an axis-aligned box has an unambiguous dominant axis and
    so takes the same UVs under any triangulation of it.
    """
    w, l, h = (float(v) for v in size)
    cx, cy, cz = (float(v) for v in center)
    verts = np.array([
        (cx + sx * w / 2, cy + sy * l / 2, cz + sz * h / 2)
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ], dtype=np.float64)
    return verts, _BOX_TRIS.copy()


def _triplanar_uvs(vertices, triangles, scale_m):
    """Per-corner UVs by per-triangle dominant-axis planar projection.

    Each triangle is projected onto whichever axis-aligned plane OF THE FRAME
    IT IS CALLED IN its own normal points most strongly along — a +Z-facing
    face takes (x, y), a +X-facing face takes (y, z) — and the coordinates are
    divided by `scale_m`, so one
    tile of the image spans `scale_m` meters of surface whatever the part's
    size. Returns `(3 * n_triangles, 2)`, which is the layout Open3D's legacy
    `TriangleMesh.triangle_uvs` takes: UVs are per CORNER, not per vertex, so
    a shared vertex can carry a different UV on each face it belongs to and
    the projection needs no seam handling.

    **Must be called in the part's LOCAL frame.** Both call sites build
    geometry locally and apply the part's world SE3 afterwards, which is the
    only correct order: derived from world coordinates instead, the same table
    would show different grain depending on where it stands in a room, and an
    animated part's texture would swim across its own surface as it moved.
    A texture is a property of the part, so its parameterization has to be too.

    Chosen over a chart atlas (xatlas and friends) because a library material
    TILES. An atlas packs charts into the unit square at a per-chart rotation
    and scale, which for a plank texture means grain running a different way
    and at a different size on every chart; and a chart layout is not derivable
    from geometry, so it would have to be persisted in the OBJ, changing every
    solid's mesh digest. This is derivable, world-scale-correct, and stored
    nowhere. Its cost is a seam where the dominant axis flips, at 45-degree
    normals; on CAD furniture — flat faces and fillets — that reads as a soft
    band rather than a break.
    """
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    if len(t) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    corners = v[t]                                   # (T, 3, 3)
    n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    # A degenerate (zero-area) triangle has no normal; project it on XY rather
    # than dividing by zero. It covers no pixels either way.
    n = n / np.where(ln == 0.0, 1.0, ln)
    axis = np.argmax(np.abs(n), axis=1)               # 0=X, 1=Y, 2=Z
    # The two axes that are NOT the dominant one:
    #   dominant X -> (Y, Z),  dominant Y -> (X, Z),  dominant Z -> (X, Y)
    u_axis = np.array([1, 0, 0])[axis]
    v_axis = np.array([2, 2, 1])[axis]
    flat = corners.reshape(-1, 3)                     # (3T, 3)
    idx = np.arange(len(flat))
    scale = float(scale_m)
    return np.stack([flat[idx, np.repeat(u_axis, 3)] / scale,
                     flat[idx, np.repeat(v_axis, 3)] / scale], axis=1)


# -------------------------------------------------------------------------
# Camera / room visibility
# -------------------------------------------------------------------------

def _floor_only_for(camera: CameraSpec) -> bool:
    """Whether `camera` gets the floor and no walls at all.

    Pure, so it is testable with no GL context.

    The canonical ids answer from the table rather than from geometry, and that
    is byte-compatibility rather than laziness: a purely geometric rule would
    also fire for `bottom`, whose horizontal offset is exactly zero
    (`render/__init__.py`'s `bottom_eye`) and which today *does* get walls
    drawn. Changing that would move the bytes of a canonical view, forcing a
    `RENDER_CONFIG_VERSION` bump and invalidating every stored render.

    A caller-supplied camera has no entry in that table — and cannot be named
    `"top"`, since `look_at_camera` hashes its id — so it needs the geometric
    rule: steeper than 45° from horizontal means walls would stand between the
    camera and the scene rather than behind it. The four canonical side cameras
    clear that threshold by a wide margin (at 15° elevation, horiz/|dz| = 3.73).
    """
    if camera.id == "top":
        return True
    if camera.id in CANONICAL_CAMERA_IDS:
        return False
    dx = camera.eye[0] - camera.target[0]
    dy = camera.eye[1] - camera.target[1]
    dz = camera.eye[2] - camera.target[2]
    return math.hypot(dx, dy) < abs(dz)


# ---------------------------------------------------------------------------
# Per-camera outputs: cache key, filename, hit/miss
# ---------------------------------------------------------------------------

class CameraOutput(NamedTuple):
    """Where one camera's PNG goes, and whether it already exists somewhere.

    `cached_src` non-None means the cache already holds a byte-identical image
    and the backend must NOT draw this camera — it materializes the hit instead
    (`_materialize`). That decision has to be made identically by every
    backend, which is the whole reason this is a function and not a loop body:
    the rule it encodes is subtle in exactly one place, `pose_label`. The cache
    key covers the LABEL, not the `poses` / `base_deltas` that
    produced it, so a caller rendering many poses under one label gets a
    hardlink farm pointing at the first frame — a clip that plays, and holds
    perfectly still: vary it per frame.
    """

    camera: CameraSpec
    key: tuple
    out_path: Path
    cached_src: Path | None


def plan_outputs(request: RenderRequest, plan: "_Plan",
                 *, config_version: str | None = None) -> list[CameraOutput]:
    """Resolve every camera in `plan` to its cache key, PNG path, and hit/miss.

    `config_version` overrides the request's, so a backend can key on its own
    recipe version: bumping Blender's render settings must not invalidate the
    Open3D PNGs sitting in the same in-memory cache, and vice versa. The
    backend name is already a key component (`render_cache_key`), so the two
    never collide — this is only about each backend versioning itself.
    """
    outputs = []
    suffix = ("" if request.pose_label == "default"
              else f"_{request.pose_label}")
    for cam in plan.cameras:
        key = render_cache_key(
            ir_hash=plan.ir_hash, target=request.target, camera_id=cam.id,
            config_version=(request.config_version if config_version is None
                            else config_version),
            catalog_fingerprint=plan.catalog_fp,
            pose_label=request.pose_label,
            backend=request.backend,
            # Per-request framebuffer size is part of the identity: a 640x360
            # selection render must never be served to a caller that asked for
            # a full-size one.
            resolution=request.resolution,
            only=request.only,
        )
        out_path = request.out_dir / f"{request.step}_{cam.id}{suffix}.png"
        cached = ((request.cache or {}).get(key)
                  if request.cache is not None else None)
        cached_src = (Path(cached)
                      if cached is not None and Path(cached).exists() else None)
        outputs.append(CameraOutput(cam, key, out_path, cached_src))
    return outputs


# ---------------------------------------------------------------------------
# Room surfaces, per camera
# ---------------------------------------------------------------------------

class WallBox(NamedTuple):
    """One room surface to draw for one camera, as a plain box.

    `alpha` is an override, or None to use the material's own. Nothing sets
    one: an unconditional override would discard an authored translucent room
    material. The slot stays because a cutaway that fades rather than hides
    would set it.

    `camera_visible` is the cutaway, and it is a statement about the CAMERA,
    not about the room. **A surface that is not camera-visible still exists**:
    it occludes nothing the camera can see, and it still bounces, shadows, and
    contributes to every indirect path. That distinction is the whole point —
    a room whose near wall and ceiling are simply absent is missing two of its
    six bounce surfaces, and reads flat under any lighting. A surface that
    should not exist at all is not emitted (see `suppressed` in `room_info`).
    """

    tag: str
    size: tuple
    se3: tuple
    material: dict
    alpha: float | None
    camera_visible: bool = True


#: Room surfaces, and the outward direction of each. The sign picks which side
#: of the room's AABB the surface sits on along its axis; `None` is the axis it
#: spans. Order is draw order and is stable, because `blender_renderer` names
#: each wall's scratch mesh by index.
_ROOM_SURFACES = (
    ("floor", 2, -1),
    ("ceiling", 2, +1),
    ("wall_xmin", 0, -1),
    ("wall_xmax", 0, +1),
    ("wall_ymin", 1, -1),
    ("wall_ymax", 1, +1),
)

#: Every tag `room_wall_boxes` can emit. Exported because a backend that keeps
#: one persistent scene across cameras has to REMOVE the previous camera's
#: surfaces, and it can only do that by naming them.
#:
#: Filament refuses a duplicate geometry id, warns, and IGNORES the add, so a
#: tag missing from this list is added once and never removed, and every camera
#: after the first renders the first camera's cutaway. One definition, so the
#: backends cannot drift.
ROOM_SURFACE_TAGS = tuple(tag for tag, _axis, _sign in _ROOM_SURFACES)


def _surface_bound(room_info: dict, axis: int, sign: int) -> float:
    """The room AABB face this surface sits on, along its own axis."""
    lo, hi = room_info["aabb"]
    return hi[axis] if sign > 0 else lo[axis]


#: Shift a cutaway plane this far INTO the room, in meters.
#:
#: **Without it the camera ray gets trapped inside the wall.** An authored wall
#: is a slab, and its INNER face sits exactly on the room's interior bound —
#: `set_room(4.4, 4.6, ...)` gives y = -2.30 and `room_shell` puts its plaster
#: face at y = -2.30. A strict `y < -2.30` is therefore true on the outer face
#: and false on the inner one, so a camera ray passed through the outside of
#: the wall, hit the inside of the same wall from behind, and rendered its
#: unlit backface: a room that read as a solid black block with the window as
#: the only opening. Biasing the plane 0.1 mm inward puts that inner face
#: inside the half-space, which is what makes the whole slab see-through.
#:
#: Far below any modeling tolerance (`MinPartDimension` is millimeters), and
#: it can only ever affect the shell parts that carry the mask — furniture
#: standing flush against a wall has no mask to trigger.
_CUTAWAY_PLANE_BIAS_M = 1.0e-4


def _cutaway_bound(room_info: dict, axis: int, sign: int) -> float:
    """The half-space plane for this surface: its face, biased inward.

    One helper so `room_cutaway_slots` (which ships the number to the renderer)
    and `room_cutaway_planes` (which reports it) can never disagree.
    """
    return (_surface_bound(room_info, axis, sign)
            - sign * _CUTAWAY_PLANE_BIAS_M)


def _eye_is_outward_of(camera: CameraSpec, bound: float,
                       axis: int, sign: int) -> bool:
    """Is the eye on the OUTWARD side of this surface, i.e. behind it?

    The one geometric predicate the cutaway rests on, shared so the automatic
    shell and authored architecture can never disagree about which surface
    stands between the eye and the room.
    """
    return (camera.eye[axis] > bound if sign > 0
            else camera.eye[axis] < bound)


def room_cutaway_planes(camera: CameraSpec,
                        room_info: dict | None) -> dict[str, list[dict]]:
    """Per replacement instance, the half-spaces `camera` must see through.

    The authored counterpart of `room_wall_boxes`' `camera_visible` flag, and
    it exists because those two mechanisms had a gap between them. A scene that
    calls `replace_room_surface` takes a shell surface out of
    `room_wall_boxes` entirely (`suppressed`), so there is nothing left to
    carry a visibility flag — while the authored object standing in its place
    is an ordinary instance and gets no cutaway at all. A room whose six
    surfaces are all authored therefore rendered as a **closed box** from the
    outside, which is exactly where the canonical views stand.

    Returns `{instance_name: [{"tag", "axis", "sign", "bound"}, ...]}`, one
    entry per surface that (a) the scene replaced with that instance and (b)
    the eye is outward of. Each is the half-space `sign * (p[axis] - bound) >
    0` in WORLD coordinates: the slab zone the suppressed shell surface used
    to occupy.

    **A half-space rather than a part list, because the geometry is fused.** An
    authored shell routinely carries all four walls as one solid, so there is no
    part to hide and no bounding box that isolates the near wall from the far
    one. Testing the shading point instead cuts the same slab the shell's own
    box did — the near wall plus the corner returns of the flanking walls —
    whatever the part breakdown.

    Pure, like `room_wall_boxes`, so which surfaces a camera must see through
    stays a statement about the scene and is testable without a renderer.
    """
    if room_info is None:
        return {}
    replacements = room_info.get("replacements") or {}
    if not replacements:
        return {}
    out: dict[str, list[dict]] = {}
    for tag, axis, sign in _ROOM_SURFACES:
        instance = replacements.get(tag)
        if instance is None:
            continue
        bound = _surface_bound(room_info, axis, sign)
        if not _eye_is_outward_of(camera, bound, axis, sign):
            continue
        out.setdefault(instance, []).append(
            {"tag": tag, "axis": axis, "sign": sign,
             "bound": float(_cutaway_bound(room_info, axis, sign))})
    return out


def room_cutaway_slots(room_info: dict | None) -> dict[str, list[dict]]:
    """Every half-space a replacement instance could ever need, camera-free.

    The static half of `room_cutaway_planes`: which surfaces each instance
    replaces, and where their planes are. A renderer builds the mask from this
    ONCE per part and then only switches slots on and off per camera, which is
    why this is separated out — rebuilding shader nodes per camera would mean
    re-finding and re-linking whatever the authored graph left wired to the
    material output.
    """
    if room_info is None:
        return {}
    out: dict[str, list[dict]] = {}
    for tag, axis, sign in _ROOM_SURFACES:
        instance = (room_info.get("replacements") or {}).get(tag)
        if instance is None:
            continue
        out.setdefault(instance, []).append(
            {"tag": tag, "axis": axis, "sign": sign,
             "bound": float(_cutaway_bound(room_info, axis, sign))})
    return out


def room_wall_boxes(camera: CameraSpec, room_info: dict | None) -> list[WallBox]:
    """The room's surfaces for `camera`, in draw order.

    Pure, so it is testable with no GL context and shareable between backends
    — the cutaway is a statement about the scene, not about a renderer.

    **All six surfaces are emitted; the cutaway is a visibility flag.** Every
    surface exists as geometry and `camera_visible` decides whether the camera
    may see it: an absent surface bounces nothing, a camera-invisible one
    bounces normally. Omitting the wall behind the camera and the ceiling would
    leave an interior lit inside an open box, flat however it is lit.

    A surface is hidden from the camera when the eye is on its OUTWARD side,
    because then it stands between the eye and the room. Being geometric and
    per-surface rather than per-axis, this also covers an eye placed diagonally
    outside two spans at once: both walls it is outside of are hidden.

    `suppressed` names surfaces the scene has REPLACED with authored geometry.
    Those are not emitted at all — not even invisibly — because a replaced wall
    must not leave a hidden slab shadowing the room or blocking the opening cut
    in the object that replaced it.
    """
    if room_info is None:
        return []

    (xmn, ymn, zmn), (xmx, ymx, zmx) = room_info["aabb"]
    lo = (xmn, ymn, zmn)
    hi = (xmx, ymx, zmx)
    extent = (xmx - xmn, ymx - ymn, zmx - zmn)
    centre = tuple(0.5 * (lo[i] + hi[i]) for i in range(3))

    wall_mat = room_info["wall"]
    surface_mat = {
        "floor": room_info["floor"],
        # A room that never named a ceiling material gets the wall's, which is
        # what a painted ceiling usually is and what the pre-ceiling renders
        # effectively showed at the top of frame (the gray backdrop).
        "ceiling": room_info.get("ceiling") or wall_mat,
    }
    suppressed = set(room_info.get("suppressed") or ())

    # `top` and `bottom` look along the room's short axis, where even a
    # correctly-hidden shell leaves the four walls standing edge-on around the
    # frame. Keeping them camera-invisible keeps those two canonical views a
    # clean plan of the floor.
    plan_view = _floor_only_for(camera)

    boxes: list[WallBox] = []
    for tag, axis, sign in _ROOM_SURFACES:
        if tag in suppressed:
            continue
        size = list(extent)
        size[axis] = _SHELL_THICKNESS_M
        # Outer face flush with the room AABB, so the clear interior volume is
        # exactly the authored width x depth x height.
        face = _surface_bound(room_info, axis, sign)
        position = list(centre)
        position[axis] = face + sign * 0.5 * _SHELL_THICKNESS_M

        visible = not _eye_is_outward_of(camera, face, axis, sign)
        if plan_view and tag != "floor":
            visible = False

        boxes.append(WallBox(
            tag=tag,
            size=tuple(size),
            se3=(tuple(position), (1.0, 0.0, 0.0, 0.0)),
            material=surface_mat.get(tag, wall_mat),
            # None: the surface's own authored alpha applies.
            alpha=None,
            camera_visible=visible,
        ))
    return boxes


# -------------------------------------------------------------------------
# PNG output
# -------------------------------------------------------------------------

#: zlib level for render PNGs. PIL defaults to 6, which costs about six times
#: as long as level 1 per 1280x720 render, most of all render time, for about a
#: fifth fewer bytes. These are intermediate evidence frames read by an agent and
#: kept for inspection, not a distribution format.
#:
#: Nothing depends on the exact bytes: `render_cache_key` is content-addressed on
#: (ir_hash, target, camera, config_version, pose, resolution), not on pixels.
#:
#: Lives here rather than in `open3d_renderer` because the writer moved here with
#: the rest of the shared scene composition — both backends save through it, so
#: both get the same level. `open3d_renderer` re-exports the name.
_PNG_COMPRESS_LEVEL = 1


def _save_png(arr: np.ndarray, path: Path, compress_level=None) -> None:
    from PIL import Image
    # `compress_level=None` means the module default, which is 1 rather than
    # Pillow's 6 — see `_PNG_COMPRESS_LEVEL` for the measurement. A caller that
    # wants a different level per request passes it through
    # `RenderRequest.png_compress_level`.
    level = _PNG_COMPRESS_LEVEL if compress_level is None else compress_level
    Image.fromarray(arr).save(path, format="PNG", compress_level=level)
    # Match the rest of the runs/ artifacts: world-readable.
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def _materialize(src: Path, dst: Path) -> None:
    """Cache hit: copy/hardlink src → dst so the step-indexed PNG exists.

    A hit whose cached path IS the destination is a no-op, and has to be
    checked before the unlink below: `dst.unlink()` would delete the very file
    `src` names, and the copy that follows would then fail with the cached PNG
    already gone. Reachable whenever two requests at one step agree on the
    whole cache key — same IR, camera, pose, resolution — which a caller
    naming a stable camera id twice in a step does by construction.
    """
    if src == dst or (src.exists() and dst.exists()
                      and src.samefile(dst)):
        return
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass


# -------------------------------------------------------------------------
# Catalog fingerprinting (for scene-layer cache keys)
# -------------------------------------------------------------------------

def _catalog_fingerprint(request: RenderRequest) -> str | None:
    """Fingerprint the catalog subset actually used by this scene IR.

    Object renders or non-scene calls return None.
    """
    if request.layer != "scene" or request.catalog is None:
        return None
    import hashlib
    refs = sorted({inst["object_ref"] for inst in request.ir.get("instances", [])})
    h = hashlib.blake2b(("|".join(refs)).encode("utf-8"), digest_size=8)
    return h.hexdigest()
