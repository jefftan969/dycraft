"""Export a committed SceneIR into a world a kinematics or physics library can load.

`render/urdf_export.py` is **not modified** by any of this: it serves the canonical renders and the URDF viewer,
and widening it for a solver's needs would put both at risk. Everything here composes the two functions it
exports.

The split
---------
`scene_ir_to_urdf` welds every instance to a synthetic ``world`` link with a fixed joint, which is right for
rendering and wrong for a body that should move. `object_ir_to_urdf` emits a free body: one robot rooted at
``ir["root"]``, no ``world`` link, no weld. So:

  * the **static world** is `scene_ir_to_urdf` over the SceneIR with the freed instances filtered out, loaded
    with a fixed base;
  * each **free body** is `object_ir_to_urdf` over its catalog ObjectIR, loaded with a free base at the world pose
    the dycraft kernel resolves.

URDF loaders expect one root per file, and a free rigid body is its own robot, so this is more correct than one
multi-root URDF.

Collision geometry
------------------
A mesh part emits a ``<mesh>`` visual and a ``<box>`` collision proxy (`render/urdf_export.py`), which is fine for
a picture and wrong for collision reasoning about a curved part: a mug collides nothing like its box. So
`_promote_mesh_collisions` rewrites each link's collision geometry to the mesh its visual names, on the emitted
XML, leaving `render/`'s proxy untouched. Concavity is the consumer's concern.

The room
--------
`scene_ir_to_urdf` omits the room, and so does this. A floor and walls are primitives every engine has, so
`scene_meta.json` carries the room's dimensions and a consumer builds them engine-side, with a proper infinite
plane instead of a slab.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from ...sdk import _rotation as _rot
from ...sdk import _scene_kernel as _sk
from ...sdk import _probes as _probes
from ...sdk import scene_query as _q
from ...render.urdf_export import (
    localize_object_meshes,
    object_ir_to_urdf,
    scene_ir_to_urdf,
)

#: Subdir of the run dir the export lands in.
SIM_DIR = "sim"
SCENE_META = "scene_meta.json"
WORLD_URDF = "world.urdf"
BODIES_DIR = "bodies"
MESHES_DIR = "meshes"


class SimExportError(Exception):
    """The scene cannot be exported as asked. Operator/authoring fault, not a
    verifier verdict — it means the request is ill-formed, e.g. freeing an
    instance that something else hangs off."""


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _validate_free(scene_ir: dict, free) -> list:
    """Resolve and sanity-check the set of instances to make free bodies.

    Two refusals, both because the alternative is a silently broken world:

    * freeing an instance something else is attached to would orphan the child,
      whose placement resolves through a parent that is no longer in the static
      scene. Detected and named rather than transitively freed — automatic
      transitive freeing is a bigger semantic call than this needs to make, and
      the common case (a mug on a table, free the mug) has no children.
    * freeing a group member or root, for the same reason one level up.
    """
    known = {i["name"] for i in scene_ir.get("instances", ())}
    free = list(dict.fromkeys(free or ()))
    missing = [n for n in free if n not in known]
    if missing:
        raise SimExportError(
            f"free names not in the scene: {sorted(missing)} "
            f"(instances: {sorted(known)})")
    freeset = set(free)

    orphans = []
    for inst in scene_ir.get("instances", ()):
        if inst["name"] in freeset:
            continue
        placement = inst.get("placement") or {}
        if placement.get("parent") in freeset:
            orphans.append((inst["name"], placement["parent"]))
    if orphans:
        detail = ", ".join(f"{c} hangs off {p}" for c, p in sorted(orphans))
        raise SimExportError(
            f"cannot free an instance that something is attached to: {detail}. "
            f"Free the child too, or simulate the parent fixed.")

    for g in scene_ir.get("groups", ()) or ():
        members = set(g.get("members") or ())
        members.add(g.get("root"))
        clash = sorted(members & freeset)
        if clash:
            raise SimExportError(
                f"cannot free {clash}: they belong to group {g['name']!r}. "
                f"Groups move as a unit; free the whole group or none of it.")
    return free


def _static_scene(scene_ir: dict, freeset: set) -> dict:
    """A shallow copy with the freed instances removed. Never re-hashed — this
    IR is export-only, so its stale `id` is deliberate and harmless."""
    if not freeset:
        return scene_ir
    return {**scene_ir,
            "instances": [i for i in scene_ir["instances"]
                          if i["name"] not in freeset]}


# ---------------------------------------------------------------------------
# The mesh-collision promotion
# ---------------------------------------------------------------------------


def _promote_mesh_collisions(urdf_xml: str) -> tuple:
    """Give every link whose visual is a mesh the same mesh as its collision.

    Returns `(xml, n_promoted)`. Links whose visual is already a primitive are
    untouched — a box part should stay a box, and its proxy is exact.

    **The visual's origin comes with the mesh, and that is not cosmetic.** The
    two origins are not the same frame: a mesh visual is emitted at the part
    frame the mesh file is authored in, while the box proxy's origin is the
    *center of the mesh's AABB*, because that is where a box has to sit to
    stand in for it. Swapping the geometry and keeping the proxy's origin
    therefore applies the centering offset twice.

    Without it a mesh whose visual origin is its base but whose box proxy is
    centered settles into a ground plane by its own half-height, with no error
    raised: nothing that compares body poses looks at collision geometry.
    """
    root = ET.fromstring(urdf_xml)
    promoted = 0
    for link in root.findall("link"):
        visual = link.find("visual")
        vis_mesh = None if visual is None else visual.find("geometry/mesh")
        if vis_mesh is None:
            continue
        # An `imported` part's collision is already a mesh, and it is the one
        # the asset shipped for contact — convex where the visual is not, and
        # the body dycraft's own narrow phase measures. Promoting over it would
        # substitute the visual mesh, which is the opposite of the point: this
        # step exists to replace a *box* proxy, not to prefer the visual body.
        # Checked on the geometry rather than on the tag, because this function
        # sees XML and has no IR to ask.
        if link.find("collision/geometry/mesh") is not None:
            continue
        vis_origin = visual.find("origin")
        for col in link.findall("collision"):
            geom = col.find("geometry")
            if geom is None:
                continue
            for child in list(geom):
                geom.remove(child)
            ET.SubElement(geom, "mesh", dict(vis_mesh.attrib))
            col_origin = col.find("origin")
            if col_origin is not None:
                col.remove(col_origin)
            if vis_origin is not None:
                col.insert(0, ET.fromstring(
                    ET.tostring(vis_origin, encoding="unicode")))
            promoted += 1
    return ET.tostring(root, encoding="unicode"), promoted


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _pose_record(scene_ir, catalog, name: str) -> dict:
    """World pose of an instance at the committed rest scene, in both forms.

    Quaternion for the engine, euler degrees for a human reading the meta file.
    """
    (tx, ty, tz), quat = _sk._resolve_world_pose(scene_ir, catalog, name, {})
    rx, ry, rz = _rot.quat_to_euler_xyz_deg(quat)
    return {"translation": [tx, ty, tz],
            "quaternion_wxyz": list(quat),
            "rotation_euler_xyz_deg": [rx, ry, rz]}


def _instance_record(scene_ir, catalog, name: str) -> dict:
    # No attach parent is recorded: `world_pose` is absolute for parent and
    # child alike, so a consumer places either without knowing the other.
    lo, hi = _q.instance_world_aabb(scene_ir, catalog, name)
    inst = _sk._index_instances(scene_ir)[name]
    return {"instance": name,
            "object_ref": inst["object_ref"],
            "world_pose": _pose_record(scene_ir, catalog, name),
            "aabb": [list(lo), list(hi)]}


def export_sim_scene(scene_ir: dict, catalog: Mapping[str, dict], out_dir,
                     *, free=(), asset_base_dir) -> dict:
    """Write a loadable world into `out_dir` and return the meta dict.

    `free` names the instances that become free bodies. `asset_base_dir` is the
    root a catalog part's relative `assets/...` path resolves against — the
    staged catalog dir for an animation run.

    Layout written:

        <out_dir>/world.urdf            the static scene, fixed base
        <out_dir>/bodies/<inst>.urdf    one per free body
        <out_dir>/meshes/...            localized assets, paths relativized
        <out_dir>/scene_meta.json       the manifest + fidelity ground truth
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies_dir = out_dir / BODIES_DIR
    meshes_dir = out_dir / MESHES_DIR
    asset_base_dir = Path(asset_base_dir)

    free = _validate_free(scene_ir, free)
    freeset = set(free)
    missing: list = []
    promoted = 0

    # -- static world ------------------------------------------------------
    static_ir = _static_scene(scene_ir, freeset)
    static_catalog = dict(catalog)
    # `localize_object_meshes` handles a `solid` through its `mesh` key, which
    # is where a committed catalog's geometry actually hangs.
    for ref, obj in list(static_catalog.items()):
        static_catalog[ref] = localize_object_meshes(
            obj, source_base_dir=asset_base_dir,
            dest_dir=meshes_dir / ref, urdf_dir=out_dir, missing=missing)
    world_xml, n = _promote_mesh_collisions(
        scene_ir_to_urdf(static_ir, static_catalog,
                         robot_name="dycraft_world"))
    promoted += n
    (out_dir / WORLD_URDF).write_text(world_xml)

    # -- one URDF per instance, free or not --------------------------------
    #
    # Every instance gets a standalone body, not just the freed ones, so a
    # script can compose whatever subset of the scene it wants instead of
    # taking `world.urdf` whole. That is the difference between the agent
    # choosing what to import and this function choosing for it; a 20-instance
    # kitchen costs a mesh convexification per body on every load, and a
    # rollout about a mug and a table has no use for the other eighteen.
    #
    # Verified equivalent: loading `bodies/table.urdf` at its recorded
    # `world_pose` puts the geometry in exactly the same place as the welded
    # `world.urdf` does — 0.000 mm apart on the worked scene, across two
    # completely separate code paths (`scene_ir_to_urdf`'s weld origin versus
    # `object_ir_to_urdf` plus a spawn pose).
    #
    # What hand-composing costs is stated in `animation/sim/README.md`: `world.urdf`
    # resolves the attach forest into URDF joints, so a cup bolted to a cabinet
    # door rides that door's joint. Load them as separate fixed bodies and each
    # one is simply pinned at its rest pose.
    bodies_dir.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for inst in scene_ir["instances"]:
        name = inst["name"]
        rec = _instance_record(scene_ir, catalog, name)
        obj = localize_object_meshes(
            catalog[rec["object_ref"]], source_base_dir=asset_base_dir,
            dest_dir=meshes_dir / rec["object_ref"], urdf_dir=bodies_dir,
            missing=missing)
        xml, n = _promote_mesh_collisions(
            object_ir_to_urdf(obj, robot_name=name))
        promoted += n
        rel = f"{BODIES_DIR}/{name}.urdf"
        (out_dir / rel).write_text(xml)
        rec["urdf"] = rel
        all_records[name] = rec
    free_records = [all_records[n] for n in free]

    # -- meta --------------------------------------------------------------
    room = scene_ir.get("room") or {}
    room_lo, room_hi = _q.room_world_aabb(scene_ir)
    meta = {
        "scene_ref": scene_ir.get("id"),
        "world_urdf": WORLD_URDF,
        "room": {"width": room.get("width"), "depth": room.get("depth"),
                 "height": room.get("height"),
                 "aabb": [list(room_lo), list(room_hi)]},
        "free_bodies": free_records,
        "static": [all_records[i["name"]] for i in static_ir["instances"]],
        "joints": {i["name"]: _probes.scene_joint_limits(
            scene_ir, catalog, i["name"])
                   for i in scene_ir["instances"]},
        "mesh_collisions_promoted": promoted,
        "missing_assets": sorted(set(missing)),
    }
    (out_dir / SCENE_META).write_text(json.dumps(meta, indent=2,
                                                 sort_keys=True))
    return meta


__all__ = ["SIM_DIR", "SCENE_META", "WORLD_URDF", "BODIES_DIR", "MESHES_DIR",
           "SimExportError", "export_sim_scene"]
