"""dycraft.render.urdf_export -- convert dycraft ObjectIR / SceneIR to URDF.

Used by render_final_to_run_dir to dump an ``ir.urdf`` alongside the
canonical PNG renders. Pure stdlib + dycraft sdk; no external deps.

Conventions
-----------
* A non-root part's link frame is the joint frame of its incoming joint. The
  part frame is offset from it by ``inv(child_anchor)``, which the visual,
  collision and inertial ``<origin>`` carry, so the geometry sits at the part
  frame at q = 0. The root link frame is the part frame.
* URDF ``rpy`` is fixed-axis (extrinsic) XYZ, NOT dycraft's intrinsic-XYZ
  Euler helper; ``_quat_to_urdf_rpy`` reads it from the rotation matrix.
* A ``box`` part emits a centered URDF ``<box>``. Each link's mass and diagonal
  inertia come from ``sdk._primitives.inertia`` at ``_DEFAULT_DENSITY``,
  1000 kg/m^3. Materials are inline per link (``<link>_mat``), with no dedup.
* A scene is one URDF with a synthetic ``world`` link: each instance is
  replicated with names prefixed ``<instance>__`` and attached to ``world`` by a
  fixed joint at its resolved world pose. The room shell is omitted.
* **URDF joint values ARE dycraft joint values.** ``<limit>`` carries the
  authored range verbatim, and URDF ``q`` actuates the dycraft joint to ``q``.

  So **URDF ``q = 0`` is not the rest pose**: a joint whose authored
  ``default`` is non-zero loads at a configuration the scene never committed,
  and URDF has no field for a default configuration (that is SRDF's job). A
  consumer applies the defaults itself from ``scene_meta.json``:
  ``animation/ik/dycraft_ik.py::default_q`` for cuRobo, and the viewer's
  ``useJointController``, which seeds its sliders the same way. A per-instance
  ``initial_joints`` is the same limitation one level down: two instances at
  different door angles replicate the same links and ``<limit>``, and
  ``scene_export`` records each instance's committed value as its
  ``scene_meta.json`` ``default``. The fixed joint pinning each instance is
  exact: its ``<origin>`` is resolved at the committed state, including a
  kinematic-follow attach to a part of an articulated parent.

  Defaults are not baked into the joint ``<origin>``: that would make URDF zero
  the rest pose at the cost of every consumer converting
  ``q_dycraft = q_urdf + default`` on read and write, and a gripper finger with
  default 0.04 and limits ``[0, 0.04]`` would get ``[-0.04, 0]``, open negative,
  where a sign slip is invisible.
* Asset parts (``solid``, ``imported``) emit a ``<mesh filename=.../>`` visual
  from the triangulation ``_primitives.render_asset_key`` names. Collision:

  * a ``solid`` gets a ``<box>`` from its (possibly off-center) AABB: URDF
    consumers handle a mesh visual far better than a non-convex mesh collider,
    and the box matches the kernel's AABB/OBB approximation. Its ``.brep`` is
    not exported.
  * an ``imported`` part gets the collision mesh the asset shipped, copied
    beside the URDF by ``localize_object_meshes``: it is convex, closed where
    the visual often is not, and the body DyCraft's own narrow phase measures,
    so ``h.ik`` reasons about the same robot as the verifiers. Its ``<origin>``
    is the part frame, since the OBJ vertices carry the offset.

  The ``<inertial>`` is the exact tensor from the part's ``shape`` table about
  the true centroid. Each body's ``<origin>`` is offset by the point its
  geometry is about (``_primitives.inertia_origin``), so a box proxy wraps the
  real geometry even when authored away from the local origin.
* Any other primitive raises ``ValueError``: dropping links would produce a
  kinematically incomplete URDF.
"""

from __future__ import annotations

import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from ..sdk import _rotation as _rot
from ..sdk import scene_query as _scene_query
from ..sdk import _primitives as _prim
from ..sdk._material import texture_maps_from_ir as _texture_maps_from_ir

#: The one kind this file names, and only to keep it out of the `map=` loop —
#: standard URDF's bare `<texture>` IS the diffuse.
_BASE_COLOR = "base_color"

#: SDF `<metal>` child tag -> the `Material.to_ir` key holding it. The two
#: names differ (`metalness` vs `metallic`) because one is Gazebo's schema and
#: the other is the SDK's field; keeping the pairing in one place is what lets
#: the shader branch below ask "did the author write this?" per tag.
_SHADER_SCALAR_KEYS = {"metalness": "metallic", "roughness": "roughness"}


# 1 g/cm^3, i.e. water — and close enough to most of what these objects are made
# of (ABS 1.04, PLA 1.25, hardwood 0.7, aluminum 2.7) that a body's mass is the
# right order of magnitude without per-part material data. The previous 100 was a
# tenth of that: lighter than balsa, so every exported mass and inertia was off by
# 10x. Nothing in the IR depends on it — `shape.inertia_diag` is stored at unit
# density and scaled here — so this changes the exported URDF only.
_DEFAULT_DENSITY = 1000.0  # kg/m^3
# URDF requires `effort` and `velocity` on a revolute/prismatic `<limit>`, so
# both have to be emitted; neither carries information. Nothing downstream reads
# them: cuRobo takes its ranges from `scene_meta.json`, and the viewer's parser
# stores both and uses only `lower`/`upper`. The joint's real content is the
# range above them.
_DEFAULT_EFFORT = 100.0
_DEFAULT_VELOCITY = 1.0
_GIMBAL_EPS = 1.0e-9
_BOX_PROXY_MIN = 1.0e-4    # clamp for a box-proxy extent, so mass stays non-zero


def _box_proxy_params(size) -> dict:
    """Box ``<geometry>`` params for a solid's collision proxy.

    ``size`` is the part's full AABB extents. Each extent is clamped to
    ``_BOX_PROXY_MIN`` so a flat or degenerate body still yields a box with
    non-zero mass — a zero-mass link is rejected or silently ignored by most
    URDF consumers.
    """
    return {
        "width": max(float(size[0]), _BOX_PROXY_MIN),
        "length": max(float(size[1]), _BOX_PROXY_MIN),
        "height": max(float(size[2]), _BOX_PROXY_MIN),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def object_ir_to_urdf(ir: dict, *, robot_name: str | None = None) -> str:
    """Return a URDF XML string representing the given ObjectIR.

    The single robot tree mirrors the IR's joint tree, rooted at
    ``ir["root"]``. A ``box`` part is emitted as native URDF geometry; a
    ``solid`` part emits a ``<mesh>`` visual plus a ``<box>`` collision/inertia
    proxy (see module docstring); any other primitive raises ValueError.

    Note: a solid part's ``params["mesh"]`` is emitted verbatim. To produce a
    self-contained, relocatable URDF, run the IR through
    ``localize_object_meshes`` first so the OBJ files are copied next to the
    ``.urdf`` and their paths rewritten relative to it.
    """
    name = robot_name or ir.get("name") or "object"
    root = ET.Element("robot", {"name": str(name)})
    _emit_object_into(root, ir, name_prefix="")
    return _serialize(root)


def scene_ir_to_urdf(ir: dict, catalog: Mapping[str, dict],
                     *, robot_name: str | None = None) -> str:
    """Return a URDF XML string representing the given SceneIR.

    Single ``<robot>`` with a synthetic ``world`` root link. For each
    instance, the catalog object's links/joints are replicated with names
    prefixed ``<instance>__`` for global uniqueness, then a fixed joint
    connects ``world`` to the instance's root link at the resolved world
    pose. Room geometry is omitted.

    Each catalog object contributes its **full** link/joint tree (see
    ``_emit_object_into``), so a scene of articulated objects exports as
    articulated rather than as a set of rigid bodies.
    """
    name = robot_name or ir.get("name") or "scene"
    root = ET.Element("robot", {"name": str(name)})
    ET.SubElement(root, "link", {"name": "world"})

    instances = {i["name"]: i for i in ir["instances"]}
    memo: dict = {}
    for inst in ir["instances"]:
        inst_name = inst["name"]
        object_ref = inst["object_ref"]
        if object_ref not in catalog:
            continue
        obj_ir = catalog[object_ref]
        prefix = f"{inst_name}__"
        _emit_object_into(root, obj_ir, name_prefix=prefix)
        root_link_name = f"{prefix}{obj_ir['root']}"

        # Kinematic-follow: when an instance is attached to a *part* of
        # another instance with `follow` (default), parent its root link
        # directly to that parent part's link so it rides the parent's
        # joints in the URDF (e.g. a workpiece on a trunnion plate). The
        # fixed-joint origin is the child root expressed in the parent
        # part's link frame at the default (URDF q=0) configuration.
        kin = _kinematic_parent(ir, catalog, inst, instances)
        if kin is not None:
            parent_link_name, origin = kin
            joint = ET.SubElement(root, "joint", {
                "name": f"{parent_link_name}__{inst_name}",
                "type": "fixed",
            })
            ET.SubElement(joint, "parent", {"link": parent_link_name})
            ET.SubElement(joint, "child", {"link": root_link_name})
            ET.SubElement(joint, "origin", _se3_to_origin_attrs(origin))
            continue

        world_pose = _scene_query._resolve_world_pose(
            ir, catalog, inst_name, memo)
        joint = ET.SubElement(root, "joint", {
            "name": f"world__{inst_name}",
            "type": "fixed",
        })
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": root_link_name})
        ET.SubElement(joint, "origin", _se3_to_origin_attrs(world_pose))
    return _serialize(root)


def _kinematic_parent(ir, catalog, inst, instances):
    """If `inst` is a `follow` attach to a part-scoped anchor on another
    instance, return `(parent_part_link_name, origin_se3)` so its root
    link can be parented to that part link in the URDF (riding the
    parent's joints). Otherwise return None (caller parents to `world` at
    the rest world pose — room-placed, whole-object-anchor, group, or
    `follow=False` placements).
    """
    placement = inst.get("placement")
    if not placement or placement.get("kind") != "attach":
        return None
    if placement.get("follow", True) is False:
        return None
    parent_name = placement["parent"]
    if parent_name not in instances:
        return None  # parent is the room / a group — keep the world attach
    pa_name = placement["parent_anchor"]["name"]
    if "." not in pa_name:
        return None  # whole-object anchor: static overall placement
    parent_part = pa_name.partition(".")[0]
    p_obj_ir = catalog[instances[parent_name]["object_ref"]]

    # Both of the transforms below are taken at the PARENT INSTANCE's committed
    # configuration — its `initial_joints` over the definition's defaults — not
    # at the definition's defaults alone. They are two halves of one frame
    # change (`inverse(link_in_p_root) ∘ child_root_in_p_root`), so evaluating
    # them at different configurations would not merely offset the child, it
    # would place it at a pose neither state describes.
    p_poses = _scene_query.instance_initial_joints(ir, parent_name)

    # Child root pose in the parent object's root frame.
    pa_in_p_root = _scene_query._part_anchor_pose_in_object_root(
        p_obj_ir, placement["parent_anchor"], poses=p_poses)
    ca_in_c_root = _scene_query._anchor_local_pose(
        ir, catalog, inst["name"], placement["child_anchor"])
    child_root_in_p_root = _rot.se3_compose(
        pa_in_p_root, _rot.se3_inverse(ca_in_c_root))

    # Parent part's URDF link frame in the parent object's root frame.
    # link_frame = part_frame ∘ inverse(part_in_link), where
    # part_in_link = inverse(child_anchor) of the part's incoming joint
    # (identity for the root part); see `_emit_object_into`.
    frames = _scene_query._object_query._compute_frames(p_obj_ir,
                                                        poses=p_poses)
    if parent_part not in frames:
        return None
    part_frame = frames[parent_part]
    incoming = next((j for j in p_obj_ir["joints"]
                     if j["child"] == parent_part), None)
    if incoming is None:
        part_in_link = _rot.IDENTITY_SE3  # root part
    else:
        part_in_link = _rot.se3_inverse(_se3_from_anchor(incoming["child_anchor"]))
    link_in_p_root = _rot.se3_compose(part_frame, _rot.se3_inverse(part_in_link))

    origin = _rot.se3_compose(
        _rot.se3_inverse(link_in_p_root), child_root_in_p_root)
    parent_link_name = f"{parent_name}__{parent_part}"
    return parent_link_name, origin


# ---------------------------------------------------------------------------
# Object emission
# ---------------------------------------------------------------------------

def _emit_object_into(robot_elem: ET.Element, ir: dict,
                      *, name_prefix: str) -> None:
    """Append ``<link>`` / ``<joint>`` children for one ObjectIR.

    Link and joint names are prefixed with ``name_prefix`` for global
    uniqueness in scene-layer URDFs (pass ``""`` for object-layer).
    """
    joints_by_child = {j["child"]: j for j in ir["joints"]}
    root_part_name = ir["root"]

    # part-in-link SE3 per part. Root: identity (link frame = part frame).
    # Non-root: inv(child_anchor) of the incoming joint.
    #
    # Under the anchor-offset design:
    #   rest_pose = parent_anchor ∘ inverse(child_anchor)
    # and the joint pivot in parent local = parent_anchor.translation
    # (including any anchor offset translation). The URDF link frame
    # coincides with the joint frame, so we set the joint origin at
    # parent_anchor and use inverse(child_anchor) as the child's
    # part-in-link transform.
    part_in_link: dict = {root_part_name: _rot.IDENTITY_SE3}
    for child_name, joint in joints_by_child.items():
        ca = _se3_from_anchor(joint["child_anchor"])
        part_in_link[child_name] = _rot.se3_inverse(ca)

    for part in ir["parts"]:
        link_name = f"{name_prefix}{part['name']}"
        link = ET.SubElement(robot_elem, "link", {"name": link_name})
        _emit_link_body(link, part, part_in_link[part["name"]], link_name)

    for joint in ir["joints"]:
        parent_part = joint["parent"]
        child_part = joint["child"]
        parent_anchor = _se3_from_anchor(joint["parent_anchor"])
        # Joint frame expressed in the parent's link frame.
        if parent_part == root_part_name:
            joint_in_parent_link = parent_anchor
        else:
            joint_in_parent_link = _rot.se3_compose(
                part_in_link[parent_part], parent_anchor)

        jname = f"{name_prefix}{joint['name']}"
        jelem = ET.SubElement(robot_elem, "joint",
                              {"name": jname, "type": joint["type"]})
        ET.SubElement(jelem, "parent",
                      {"link": f"{name_prefix}{parent_part}"})
        ET.SubElement(jelem, "child",
                      {"link": f"{name_prefix}{child_part}"})

        if joint["type"] in ("revolute", "prismatic"):
            axis_pa = joint["axis"]
            n2 = sum(a * a for a in axis_pa)
            inv_n = (1.0 / math.sqrt(n2)) if n2 > 0.0 else 1.0
            # IR axis is in parent-anchor-local frame. URDF axis is in joint
            # frame, which coincides with parent_anchor's frame, so the
            # axis is already expressed in joint frame.
            axis_joint = (axis_pa[0] * inv_n,
                          axis_pa[1] * inv_n,
                          axis_pa[2] * inv_n)
            lo, hi = joint["limits"]
            ET.SubElement(jelem, "origin",
                          _se3_to_origin_attrs(joint_in_parent_link))
            ET.SubElement(jelem, "axis", {"xyz": _fmt_vec3(axis_joint)})
            ET.SubElement(jelem, "limit", {
                "lower": _fmt_num(lo),
                "upper": _fmt_num(hi),
                "effort": _fmt_num(_DEFAULT_EFFORT),
                "velocity": _fmt_num(_DEFAULT_VELOCITY),
            })
        else:
            ET.SubElement(jelem, "origin",
                          _se3_to_origin_attrs(joint_in_parent_link))


def _emit_geometry(geom_parent: ET.Element, primitive: str, params: dict,
                   *, asset_key: str | None = None) -> None:
    """Emit the URDF ``<box>`` or ``<mesh>`` element under a ``<geometry>``
    parent. A box is centered at the origin — matching the dycraft part frame,
    so no extra origin offset is needed beyond the link-frame transform already
    applied.

    ``asset_key`` names which of an asset part's meshes to emit, defaulting to
    the visual one. The collision slot passes ``collide_asset_key`` for an
    ``imported`` part, whose proxy is a different file — see `_emit_link_body`.
    """
    if primitive == "box":
        size_str = (f"{_fmt_num(float(params['width']))} "
                    f"{_fmt_num(float(params['length']))} "
                    f"{_fmt_num(float(params['height']))}")
        ET.SubElement(geom_parent, "box", {"size": size_str})
    elif primitive in _prim.ASSET_PRIMITIVES:
        # An asset part's URDF visual is its triangle mesh (URDF has no B-rep
        # geometry element). A solid's exact `.brep` sits beside it for the
        # collision adjudicator and is not for consumers of this file. Scale is
        # always identity: neither asset tag carries a `scale` param, by
        # construction.
        key = asset_key or _prim.render_asset_key(primitive)
        ET.SubElement(geom_parent, "mesh", {
            "filename": str(params[key]),
            "scale": "1 1 1",
        })
    else:
        raise ValueError(
            f"urdf_export: unsupported primitive {primitive!r}; "
            f"supported: box, {', '.join(_prim.ASSET_PRIMITIVES)}"
        )


def _emit_link_body(link_elem: ET.Element, part: dict, part_in_link,
                    link_name: str) -> None:
    """Add ``<visual>``, ``<collision>``, ``<inertial>`` to a link element.

    ``part_in_link`` is the SE3 of the dycraft part frame expressed in the
    link frame (identity for root; ``inv(child_anchor)`` otherwise). An analytic
    primitive is centered at the part-frame origin, so the same origin serves
    visual, collision, and inertial; an asset part gets three origins (see
    below), because its geometry may sit away from that origin.
    """
    primitive = part["primitive"]
    params = part["params"]
    # Carry the part's Material into URDF:
    #   - <color rgba="r g b a"/>  (standard URDF)
    #   - <pbr><metal><metalness>m</metalness><roughness>r</roughness></metal></pbr>
    #     (Gazebo SDF PBR schema, embedded inline inside <material>). Standard
    #     URDF parsers (urdfdom, PyBullet, rviz) ignore the <pbr> element; PBR-
    #     aware consumers (Gazebo Sim, our vcv_server viewer) read it. We only
    #     emit <pbr> when at least one field differs from default to keep
    #     non-PBR URDFs byte-identical to the pre-PBR exporter.
    material = part.get("material") or {}
    # A shader material computes its surface in a Blender node graph that no
    # URDF consumer can run, and `Material.to_ir` omits every scalar its author
    # did not write — so for these the four `.get(...)` defaults below are a
    # guess, not the value. `_shader_name` is what tells the consumer which of
    # the two it is holding; see the marker emitted beside `<pbr>`.
    shader = material.get("shader") or None
    color = material.get("color", [128, 128, 128])
    r = int(color[0]) / 255.0
    g = int(color[1]) / 255.0
    b = int(color[2]) / 255.0
    a = float(material.get("alpha", 1.0))
    metallic = float(material.get("metallic", 0.0))
    roughness = float(material.get("roughness", 0.9))
    # Every map the material carries, base color included. `None` when there
    # is nothing to emit, which is what keeps an untextured part's URDF
    # byte-identical to what it was before any of this existed.
    maps = _texture_maps_from_ir(material.get("texture"))
    texture = maps if _BASE_COLOR in maps else None

    origin_attrs = _se3_to_origin_attrs(part_in_link)

    # An asset part's collision/inertia use an axis-aligned box proxy from its
    # AABB, placed at the AABB center (which may be off the part origin). The
    # visual keeps the real mesh at the part origin (the OBJ vertices already
    # carry the offset). Analytic primitives are centered, so they share one
    # origin for all three bodies (byte-identical to the pre-mesh exporter).
    #
    # The inertial body is NOT the box proxy: it is the exact mass properties
    # from the part's `shape` table about the true centroid — a different point
    # from the AABB center, hence a third origin. Getting that offset wrong puts
    # the mass somewhere the geometry is not.
    #
    # Both asset tags take this branch. An `imported` part left in the `else`
    # emitted its full-detail VISUAL mesh as a non-convex `<collision>` (which
    # the module docstring explains URDF consumers handle badly, and which is
    # not even the collision mesh the asset ships) and placed its `<inertial>`
    # at the part origin rather than the centroid.
    def _offset_attrs(local_point):
        return _se3_to_origin_attrs(_rot.se3_compose(
            part_in_link, (local_point, (1.0, 0.0, 0.0, 0.0))))

    inert_primitive, inert_params = primitive, params
    coll_asset_key = None
    if primitive in _prim.ASSET_PRIMITIVES:
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _prim.local_aabb(primitive, params)
        collide_key = _prim.collide_asset_key(primitive)
        ships_a_proxy = (collide_key != _prim.render_asset_key(primitive)
                         and params.get(collide_key))
        if ships_a_proxy:
            # An `imported` part ships a purpose-built collision mesh, so the
            # box proxy is a downgrade of data we already have: convex where it
            # needs to be, closed where the visual often is not, and 2,680
            # triangles against the Panda's 102,752. It is also the body
            # dycraft's own narrow phase measures, which is what makes emitting
            # it a fidelity fix rather than a cost one — otherwise `h.ik`
            # would reason about a different robot from the verifiers.
            #
            # **The origin is the part frame, not the AABB center.** The proxy
            # sits at the center because that is where a box has to be to stand
            # in for the mesh; an OBJ's vertices already carry the offset, so
            # reusing the proxy's origin applies the centering twice, and a body
            # then settles into the ground by its own half-height with no error
            # raised (`sim/scene_export._promote_mesh_collisions`).
            coll_primitive, coll_params = primitive, params
            coll_asset_key = collide_key
            proxy_origin_attrs = origin_attrs
        else:
            # A `solid`'s two asset keys name one OBJ — the tessellation of its
            # exact B-rep — so "emit the collision mesh" would emit the
            # full-detail visual as a non-convex collider, which is the thing
            # the box proxy exists to avoid.
            coll_primitive = "box"
            coll_params = _box_proxy_params(
                (xmx - xmn, ymx - ymn, zmx - zmn))
            proxy_origin_attrs = _offset_attrs(
                ((xmn + xmx) * 0.5, (ymn + ymx) * 0.5, (zmn + zmx) * 0.5))
        inertial_origin_attrs = _offset_attrs(
            _prim.inertia_origin(primitive, params))
    else:
        coll_primitive, coll_params = primitive, params
        proxy_origin_attrs = origin_attrs
        inertial_origin_attrs = origin_attrs

    visual = ET.SubElement(link_elem, "visual")
    ET.SubElement(visual, "origin", origin_attrs)
    vgeom = ET.SubElement(visual, "geometry")
    _emit_geometry(vgeom, primitive, params)
    mat = ET.SubElement(visual, "material", {"name": f"{link_name}_mat"})
    ET.SubElement(mat, "color",
                  {"rgba": f"{_fmt_num(r)} {_fmt_num(g)} {_fmt_num(b)} "
                           f"{_fmt_num(a)}"})
    if texture is not None:
        # Standard URDF's own element, and the only PBR-ish field here that is:
        # `<texture filename>` is in the spec, so a stock parser reads it and an
        # indifferent one ignores it. The filename is the run-relative staged
        # path, resolved by the consumer against the same base as `<mesh
        # filename>` — for the viewer that is the `files/` route, which serves
        # anything under the run dir.
        #
        # `scale_m` has nowhere to live in URDF's vocabulary. It is carried as
        # an attribute on the same element rather than invented as a sibling
        # tag: an unknown ATTRIBUTE is skipped by every XML reader, whereas an
        # unknown child element is what `<pbr>` already is and is read by
        # nobody here. A consumer without it falls back to whatever tiling it
        # would have chosen; the two renderers agree only because both read it.
        base_attrs = {"filename": str(maps[_BASE_COLOR]["image"])}
        # Only when the material actually has one. `mapping` is absent for a
        # surface sampled through UVs its asset already had, and inventing a
        # tile size there would tell a consumer to re-project an authored
        # atlas — the exact failure the renderers now guard against.
        scale_m = (material.get("mapping") or {}).get("scale_m")
        if scale_m is not None:
            base_attrs["scale_m"] = _fmt_num(float(scale_m))
        ET.SubElement(mat, "texture", base_attrs)
        # Extra maps are further <texture> elements carrying a `map` attribute
        # naming their kind. The base-color one above deliberately has none:
        # standard URDF's `<texture>` IS the diffuse, so a stock parser taking
        # the first element gets the right one, and an object with only a base
        # color exports as plain URDF.
        #
        # Kinds are not enumerated here. `texture_maps_from_ir` decides what
        # counts as a map and the attribute carries its name through, so adding
        # one to the library needs no change in this file — the consumer that
        # knows what a kind MEANS is the renderer, not the transport.
        for kind, m in sorted(maps.items()):
            if kind == _BASE_COLOR:
                continue
            attrs = {"filename": str(m["image"]), "map": kind}
            if m.get("channel"):
                attrs["channel"] = m["channel"]
            if m.get("strength") is not None:
                attrs["strength"] = _fmt_num(float(m["strength"]))
            ET.SubElement(mat, "texture", attrs)
    if shader is not None:
        # **The marker, and it is load-bearing.** Without it the consumer reads
        # a suppressed `<pbr>` as the SDK defaults actually authored — see
        # `EXPORTED_DEFAULT_*` in
        # `viewer/web/src/components/viewer3d/materials.ts`, whose reasoning is
        # sound for a scalar material (all four fields are always filled by
        # `Material.__post_init__`) and false for this one, where absence means
        # "the graph decides". The symptom was a shader surface drawing matte
        # gray in the viewport while *claiming* the author asked for matte gray.
        #
        # The body is deliberately NOT carried. It is Python for Blender's node
        # API, of no use to a URDF consumer, and often larger than the rest of
        # the file; the name is enough to say "computed" and to give the
        # viewer's name heuristics something to read.
        attrs = {}
        if shader.get("name"):
            attrs["name"] = str(shader["name"])
        ET.SubElement(mat, "shader", attrs)
        # Only the scalars whose author actually wrote them. A shader material
        # that wrote none emits no block at all, which reads as "unknown"
        # rather than as the defaults.
        authored = [k for k in ("metalness", "roughness")
                    if _SHADER_SCALAR_KEYS[k] in material]
        if authored:
            pbr = ET.SubElement(mat, "pbr")
            metal = ET.SubElement(pbr, "metal")
            for tag in authored:
                ET.SubElement(metal, tag).text = _fmt_num(
                    float(material[_SHADER_SCALAR_KEYS[tag]]))
    elif metallic > 0.0 or roughness != 0.9:
        pbr = ET.SubElement(mat, "pbr")
        metal = ET.SubElement(pbr, "metal")
        ET.SubElement(metal, "metalness").text = _fmt_num(metallic)
        ET.SubElement(metal, "roughness").text = _fmt_num(roughness)

    coll = ET.SubElement(link_elem, "collision")
    ET.SubElement(coll, "origin", proxy_origin_attrs)
    cgeom = ET.SubElement(coll, "geometry")
    _emit_geometry(cgeom, coll_primitive, coll_params,
                   asset_key=coll_asset_key)

    mass, ixx, iyy, izz = _prim.inertia(inert_primitive, inert_params,
                                        _DEFAULT_DENSITY)
    inertial = ET.SubElement(link_elem, "inertial")
    ET.SubElement(inertial, "origin", inertial_origin_attrs)
    ET.SubElement(inertial, "mass", {"value": _fmt_num(mass)})
    ET.SubElement(inertial, "inertia", {
        "ixx": _fmt_num(ixx), "iyy": _fmt_num(iyy), "izz": _fmt_num(izz),
        "ixy": "0", "ixz": "0", "iyz": "0",
    })


# ---------------------------------------------------------------------------
# Asset localization
# ---------------------------------------------------------------------------

def _copy_asset_once(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`, skipping the write when `dst` is already there.

    Sound because these filenames are **content digests**: `_mesh_assets` names
    every asset `blake2b(bytes, digest_size=16).hexdigest()`, so a name that
    already exists at the destination cannot hold different bytes. If asset
    naming ever stops being content-addressed, this guard has to go with it.

    Worth the guard because the callers repeat: `animation/sim/scene_export.py`
    localizes per *instance*, so an object placed N times copies its meshes N
    times into the same directory, plus once more for the static catalog — and
    every `h.ik` consultation re-exports the whole scene. On an Articraft
    kitchen that is tens to hundreds of MB of pointless I/O per agent step.

    Written through a temporary in the destination directory and `os.replace`d,
    which is atomic on the same filesystem. Without that, an interrupted copy
    would leave a truncated file under a digest name and every later run would
    skip it — the guard's own failure mode, and a silent one.
    """
    if dst.exists():
        return
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def has_localizable_assets(ir: dict) -> bool:
    """True if `ir` references any file the exported URDF will point at.

    The predicate that gates ``localize_object_meshes`` lives next to it on
    purpose: a guard spelled inline at each call site can fall behind the
    function when a primitive tag is added, and skip the call for an object
    whose URDF then points at ``assets/meshes/…`` files never copied next to
    it. Keep the two together.

    Material images count for the same reason, and the failure mode they add is
    subtler: an object whose parts are all analytic primitives has no geometry
    to localize at all, so the gate would skip the call entirely and its
    ``<texture filename>`` would dangle with nothing else looking wrong.
    """
    return any(_prim.render_asset_key(p.get("primitive")) is not None
               or _material_image_paths(p)
               for p in ir.get("parts", ()))


def _material_image_paths(part: dict) -> list[str]:
    """Every image path a part's material references, base color included.

    Kinds are library data, so this asks `texture_maps_from_ir` rather than
    naming any — see `sdk/_material.py`. Returns paths in the IR's own
    spelling; the caller resolves and rewrites them.
    """
    texture = (part.get("material") or {}).get("texture")
    if not isinstance(texture, dict):
        return []
    return [m["image"] for m in _texture_maps_from_ir(texture).values()
            if m.get("image")]


def localize_object_meshes(ir: dict, *, source_base_dir, dest_dir, urdf_dir,
                           texture_dest_dir=None,
                           missing: list | None = None) -> dict:
    """Copy an ObjectIR's referenced files next to the URDF and relativize them.

    Returns a new IR (the ``parts`` list shallow-copied, affected parts'
    ``params`` and ``material`` rewritten) whose ``filename``s are **relative to
    ``urdf_dir``**, so the emitted URDF is self-contained and survives the run
    dir moving. That promise covers every file the URDF names — meshes and the
    images a material carries alike; a `<texture filename>` left pointing into
    the run dir is exactly as broken as a `<mesh filename>` would be, and
    quieter, because the geometry still draws.

    The object layer stores each path relative to the run dir
    (``assets/meshes/<digest>.obj``, ``assets/textures/<digest>.jpg``); pass
    ``source_base_dir=run_dir`` so a relative path resolves to the staged file.
    A ``solid`` part is handled through its ``mesh`` key (its ``.brep``/source
    assets are not copied — nothing downstream of a URDF reads them).

    Images go to ``texture_dest_dir``, defaulting to a ``textures`` sibling of
    ``dest_dir`` — so a caller passing ``…/assets/meshes`` gets the
    ``…/assets/textures`` layout the run dir already uses, and no call site has
    to learn a second directory. One shared directory across objects is safe
    for the same reason ``_copy_asset_once`` is: the filenames are content
    digests.

    Parts that reference nothing pass through untouched; a file missing at
    source keeps its original path and is appended to ``missing`` (if given).
    The IR ``id`` is left unchanged — the rewritten IR is export-only and never
    re-hashed.
    """
    source_base_dir = Path(source_base_dir)
    dest_dir = Path(dest_dir)
    urdf_dir = Path(urdf_dir)
    tex_dir = (Path(texture_dest_dir) if texture_dest_dir is not None
               else dest_dir.parent / "textures")
    parts_out = []
    changed = False

    def _localize(rel, into: Path):
        """Copy one referenced file and return its path relative to the URDF,
        or None when it is absent at source."""
        src = Path(str(rel))
        if not src.is_absolute():
            src = source_base_dir / str(rel)
        if not src.exists():
            if missing is not None:
                missing.append(str(src))
            return None
        into.mkdir(parents=True, exist_ok=True)
        dst = into / src.name
        _copy_asset_once(src, dst)
        return os.path.relpath(dst, urdf_dir).replace(os.sep, "/")

    for p in ir.get("parts", ()):
        primitive = p.get("primitive")
        # Both asset keys, when they name different files. For a `solid` they
        # are the same OBJ and the second lookup is a no-op; for an `imported`
        # part `collision` is a separate, purpose-built mesh, and leaving it
        # behind is what made `_emit_link_body` unable to emit it and
        # `sim/scene_export._promote_mesh_collisions` promote the *visual* mesh
        # into the collision slot instead — 102,752 triangles for the Panda in
        # place of the 2,680 it ships, five of them open shells.
        keys = [k for k in (_prim.render_asset_key(primitive),
                            _prim.collide_asset_key(primitive)) if k]
        for key in dict.fromkeys(keys):
            rel = (p.get("params") or {}).get(key)
            if not rel:
                continue
            new_rel = _localize(rel, dest_dir)
            if new_rel is not None:
                p = {**p, "params": {**p["params"], key: new_rel}}
                changed = True

        texture = (p.get("material") or {}).get("texture")
        maps = _texture_maps_from_ir(texture)
        if maps:
            # Every kind the same way, base color included. Nothing here names
            # one — the transport does not need to know what a kind means.
            tex_out = dict(texture)
            for kind, m in maps.items():
                map_rel = _localize(m["image"], tex_dir)
                if map_rel is not None:
                    tex_out[kind] = {**m, "image": map_rel}
            if tex_out != texture:
                p = {**p, "material": {**p["material"], "texture": tex_out}}
                changed = True

        parts_out.append(p)
    if not changed:
        return ir
    return {**ir, "parts": parts_out}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _se3_from_anchor(anchor: dict):
    t = anchor["translation"]
    q = anchor["rotation"]
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])))


def _quat_to_urdf_rpy(q):
    """WXYZ unit quaternion -> (roll, pitch, yaw) for URDF fixed-axis XYZ.

    URDF rpy is fixed-axis XYZ: ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``.
    Rotation matrix entries from a WXYZ quaternion:

        R[2][0] = 2*(x*z - w*y) = -sin(pitch)
        R[2][1] = 2*(y*z + w*x) =  cos(pitch)*sin(roll)
        R[2][2] = 1 - 2*(x*x + y*y) = cos(pitch)*cos(roll)
        R[1][0] = 2*(x*y + w*z) = cos(pitch)*sin(yaw)
        R[0][0] = 1 - 2*(y*y + z*z) = cos(pitch)*cos(yaw)

    so::

        pitch = asin(-R[2][0])
        roll  = atan2(R[2][1], R[2][2])
        yaw   = atan2(R[1][0], R[0][0])

    At |cos(pitch)| ~ 0 (gimbal lock, pitch ~ +/- 90 deg) ``yaw`` is pinned
    to 0 and ``roll`` is recovered from R[0][1] = sin(pitch)*sin(roll) and
    R[1][1] = cos(roll). Output is in radians.
    """
    w, x, y, z = q
    r20 = 2.0 * (x * z - w * y)
    if r20 > 1.0:
        r20 = 1.0
    elif r20 < -1.0:
        r20 = -1.0
    sin_pitch = -r20
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) >= 1.0 - _GIMBAL_EPS:
        # Gimbal lock: pin yaw = 0, recover roll.
        sign_sp = 1.0 if sin_pitch >= 0.0 else -1.0
        r01 = 2.0 * (x * y - w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        roll = math.atan2(sign_sp * r01, r11)
        yaw = 0.0
        return roll, pitch, yaw
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(r21, r22)
    r10 = 2.0 * (x * y + w * z)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def _se3_to_origin_attrs(se3) -> dict:
    t, q = se3
    r, p, y = _quat_to_urdf_rpy(q)
    return {
        "xyz": _fmt_vec3(t),
        "rpy": f"{_fmt_num(r)} {_fmt_num(p)} {_fmt_num(y)}",
    }


def _fmt_num(x: float) -> str:
    if x == 0.0:
        return "0"
    # 17 sig figs preserves a double round-trip; consumers parsing the URDF
    # with `float()` recover the exact value, so callers can keep tight
    # tolerances when comparing URDF output to FK ground truth.
    return f"{float(x):.17g}"


def _fmt_vec3(v) -> str:
    return f"{_fmt_num(v[0])} {_fmt_num(v[1])} {_fmt_num(v[2])}"


def _serialize(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0"?>\n' + body
