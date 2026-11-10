"""Export evaluated Blender state for the cuRobo IK consultation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .bundle import write_json
from .verify import load_frame, joint_position


def pose(matrix):
    matrix = np.asarray(matrix)
    rotation = Rotation.from_matrix(matrix[:3, :3])
    q = rotation.as_quat()
    return {"translation": matrix[:3, 3].tolist(),
            "quaternion_wxyz": [float(q[3]), *q[:3].tolist()],
            "rotation_euler_xyz_deg": rotation.as_euler("xyz", degrees=True).tolist()}


def write_obj(path, vertices, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        np.savetxt(stream, vertices, fmt="v %.9g %.9g %.9g")
        np.savetxt(stream, triangles + 1, fmt="f %d %d %d")


def export_state(scene_ir, catalog, asset_base_dir, baseline, evaluation_dir,
                 frame, destination, *, free=(), version=None):
    from ..sim.scene_export import export_sim_scene
    record, geometry = load_frame(evaluation_dir, frame)
    destination = Path(destination)
    meta = export_sim_scene(scene_ir, catalog, destination,
                            free=free, asset_base_dir=asset_base_dir)
    transforms = record["transforms"]
    internal_poses = {}
    meshes, obstacles, bounds = {}, [], {}
    by_identity = {}
    for obj in record['objects']:
        if obj.get('identity') and not obj.get('hidden'):
            by_identity.setdefault(obj['identity'], []).append(obj)
    for ident, spec in baseline["identities"].items():
        inst, part = ident.split("/", 1)
        visible = by_identity.get(ident, [])
        if not visible:
            meshes[f"{inst}__{part}"] = None
            continue
        if len(visible) != 1:
            raise ValueError(f'IK/simulation snapshot has ambiguous identity {ident}')
        vertices, triangles = geometry[visible[0]['name']]
        link = np.asarray(transforms[spec["link"]])
        local = (vertices - link[:3, 3]) @ np.linalg.inv(link[:3, :3]).T
        rel = f"evaluated/{len(meshes):05d}.obj"
        write_obj(destination / rel, local, triangles)
        meshes[f"{inst}__{part}"] = rel
        obstacles.append({"instance": inst, "part": part, "mesh": rel,
                          "pose": pose(link)})
        bounds.setdefault(inst, []).append(vertices)
    for obj in record["objects"]:
        if obj.get("hidden") or obj.get("room") or obj.get("identity") in baseline["identities"] or obj["name"] in baseline.get("room_objects", []):
            continue
        vertices, triangles = geometry[obj["name"]]
        rel = f"evaluated/extra_{len(obstacles):05d}.obj"
        write_obj(destination / rel, vertices, triangles)
        obstacles.append({"instance": "__animation_helpers__", "part": obj["name"],
                          "mesh": rel, "pose": pose(np.eye(4))})
    roots = {inst["name"]: f"{inst['name']}__{catalog[inst['object_ref']]['root']}"
             for inst in scene_ir["instances"]}
    meta['static'] = [rec for rec in meta['static'] if rec['instance'] in bounds]
    meta['free_bodies'] = [rec for rec in meta['free_bodies'] if rec['instance'] in bounds]
    for rec in meta["static"] + meta["free_bodies"]:
        inst = rec["instance"]
        rec["world_pose"] = pose(transforms["link/" + roots[inst]])
        verts = np.concatenate(bounds[inst])
        rec["aabb"] = [verts.min(axis=0).tolist(), verts.max(axis=0).tolist()]
    for name, joint in baseline["joints"].items():
        q = joint_position(record, joint)
        lo, hi = joint["limits"]
        if not lo-1e-5 <= q <= hi+1e-5:
            raise ValueError(f"joint {name} is outside its original limits")
        inst, jname = name.split("__", 1)
        internal_poses.setdefault(inst, {})[jname] = q
        meta["joints"][inst][jname]["default"] = math.degrees(q) if joint["type"] == "revolute" else q
    # Replace all visual/collision geometry with the same evaluated triangles.
    for path in [destination / "world.urdf", *sorted((destination / "bodies").glob("*.urdf"))]:
        tree = ET.parse(path)
        root = tree.getroot()
        body_name = path.stem if path.parent.name == "bodies" else None
        for link in root.findall("link"):
            name = link.get("name")
            key = f"{body_name}__{name}" if body_name else name
            if key not in meshes:
                continue
            for tag in ("visual", "collision"):
                for old in link.findall(tag):
                    link.remove(old)
                if meshes[key] is None:
                    continue
                element = ET.SubElement(link, tag)
                geom = ET.SubElement(element, "geometry")
                ET.SubElement(geom, "mesh", {"filename": ("../" if body_name else "") + meshes[key]})
        if body_name is None:
            for joint in root.findall("joint"):
                child = joint.find("child").get("link")
                if child not in roots.values():
                    continue
                mat = np.asarray(transforms["link/" + child])
                parent_name = joint.find("parent").get("link")
                parent_world = (np.eye(4) if parent_name == "world" else
                                np.asarray(transforms["link/" + parent_name]))
                mat = np.linalg.solve(parent_world, mat)
                old = joint.find("origin")
                if old is not None:
                    joint.remove(old)
                p = pose(mat)
                ET.SubElement(joint, "origin", {"xyz": " ".join(map(str, p["translation"])),
                    "rpy": " ".join(str(math.radians(v)) for v in p["rotation_euler_xyz_deg"])})
            for i, obstacle in enumerate(obstacles):
                if obstacle["instance"] != "__animation_helpers__":
                    continue
                name = f"animation_extra_{i}"
                link = ET.SubElement(root, "link", {"name": name})
                for tag in ("visual", "collision"):
                    geom = ET.SubElement(ET.SubElement(link, tag), "geometry")
                    ET.SubElement(geom, "mesh", {"filename": obstacle["mesh"]})
                joint = ET.SubElement(root, "joint", {"name": name + "_fixed", "type": "fixed"})
                ET.SubElement(joint, "parent", {"link": "world"})
                ET.SubElement(joint, "child", {"link": name})
        tree.write(path, encoding="unicode")
    meta["animation_snapshot"] = {"version": version, "frame": frame,
        "obstacles": "evaluated static meshes at this frame; future deformation is not simulated"}
    write_json(destination / "scene_meta.json", meta)
    instances = {}
    from ...verify._inherited_allowances import InheritedAllowances
    inherited = InheritedAllowances(scene_ir, catalog)
    for inst in scene_ir["instances"]:
        obj = catalog[inst["object_ref"]]
        ignores = {p["name"]: [] for p in obj["parts"]}
        for joint in obj["joints"]:
            ignores[joint["parent"]].append(joint["child"])
            ignores[joint["child"]].append(joint["parent"])
        for pair in inherited.at_pose(inst["name"], internal_poses.get(inst["name"], {})):
            a, b = sorted(pair)
            ignores[a].append(b)
            ignores[b].append(a)
        instances[inst["name"]] = {"object_ref": obj["id"], "urdf": f"bodies/{inst['name']}.urdf",
            "root_link": obj["root"], "active_joints": [j["name"] for j in obj["joints"] if j["type"] != "fixed"],
            "links": list(ignores), "self_collision_ignore": {k: sorted(set(v)) for k, v in ignores.items()},
            "exact_adjudication": False,
            "self_collision_ignore_source": "joint adjacency and inherited object allowances at snapshot pose"}
    write_json(destination / "ik_meta.json", {"scene_ref": scene_ir["id"], "instances": instances,
        "obstacles": obstacles, "animation_snapshot": meta["animation_snapshot"]})
    return meta


def requested_frames(args):
    """The frames to export, as zero-based output frame indices; `[]` for none.

    Exporting is optional: a script that builds its own robot and world needs
    no snapshot of the project, and asking for none skips the Blender pass."""
    frames = args.get("frames")
    if frames is None:
        frame = args.get("frame")
        if frame is None:
            return []
        frames = [frame]
    if (not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= 16
            or any(type(f) is not int or f < 0 for f in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("frames must be 1 to 16 distinct zero-based output frame indices")
    return list(frames)


def authored_code(args, run_root):
    """`(source, label)` from exactly one of `code` (a string) and `script`
    (a file under the run dir, read now so the record holds what ran)."""
    code, script = args.get("code"), args.get("script")
    if (code is None) == (script is None):
        raise ValueError("pass exactly one of code (Python source) or script (a path under the run dir)")
    if code is not None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be non-empty Python source")
        return code, None
    root = Path(run_root).resolve()
    path = Path(script)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"script must be an existing file under the run dir {root}: {script}")
    return path.read_text(), str(path.relative_to(root))


def robot_sources(run_root):
    """`{asset name: {urdf, package_root}}` for each asset in this run imported
    from a URDF. The run stages only the asset's ObjectIR, so this is how a
    script reaches the robot's original description to build its own model.

    Assets come from two places: those preloaded at launch (`meta.json`), and
    those the agent retrieved from the library with `load_asset`, which exist
    only in the manifest of the source tree the current design was built from."""
    run_root = Path(run_root)
    preloaded = {}
    try:
        preloaded.update(json.loads((run_root / "meta.json").read_text()).get("preloaded_assets") or {})
    except (OSError, ValueError):
        pass
    for manifest in (run_root / "native/work/design_state/source/preloaded/manifest.json",
                     run_root / "preloaded/manifest.json"):
        try:
            assets = json.loads(manifest.read_text()).get("assets") or {}
        except (OSError, ValueError):
            continue
        for name, rec in assets.items():
            preloaded.setdefault(name, rec)
    out = {}
    for name, rec in preloaded.items():
        try:
            info = json.loads((Path(rec["source"]) / "import.json").read_text())
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if info.get("source_urdf"):
            out[name] = {"urdf": info["source_urdf"], "package_root": info.get("package_root")}
    return out


def consult(service, operation, args, workspace, directory, cancel):
    """Run the agent's cuRobo code on the GPU, in the run's persistent `ik/`.

    The code builds its own robot and world: the original URDFs, cuRobo's
    bundled robot configs, or the project's state at `frame`/`frames`, which is
    exported to `ik/exports/<version>_f<frame>/` (named to the code in
    `DYCRAFT_IK_EXPORTS`) only when asked for. The directory outlives the call,
    so solvers' caches, intermediate results, and the agent's own modules carry
    over; the agent's shell writes there too, which is how `script=` works.
    """
    if operation != "ik":
        raise ValueError(f"unsupported solver consultation {operation!r}")
    from ..ik.ik import IK_DIR, NOT_INSTALLED, availability, ik, unavailable_message
    run_root = Path(getattr(service, "run_root", None) or service.root)
    source, script = authored_code(args, run_root)
    frames = requested_frames(args)
    # Before anything is evaluated or exported: with no IK, the answer is the
    # same whatever the frames, and an export costs a Blender evaluation.
    found = availability()
    if not found["available"]:
        return {"ok": False, "error_kind": "unavailable",
                "message": unavailable_message(found["reason"] or NOT_INSTALLED)}
    workdir = run_root / IK_DIR
    workdir.mkdir(parents=True, exist_ok=True)
    version = service.version_of(workspace)
    free = tuple(sorted(args.get("free") or ()))
    # An export is a function of (version, frame, free), so one already on
    # disk is reused rather than re-evaluated.
    exports = {str(f): workdir / "exports" / (f"{version[:12]}_f{f:06d}"
                                              + (f"_free_{'-'.join(free)}" if free else ""))
               for f in frames}
    missing = [f for f in frames if not (exports[str(f)] / "scene_meta.json").is_file()]
    if missing:
        # One evaluation pass for every frame still to export. Same seek rule
        # as every other frame-addressed op: a stateless scene need not be
        # stepped from frame 0 to export one configuration.
        service.worker(workspace, directory / "state",
                       {"op": "evaluate", "frames": sorted(missing)}, cancel)
        baseline = json.loads((service.native / "baseline.json").read_text())
        for frame in missing:
            if cancel.is_set():
                raise RuntimeError("job cancelled")
            export_state(service.ctx.scene_ir, service.ctx.catalog,
                         getattr(service.ctx, "asset_base_dir", None) or service.root / "catalog",
                         baseline, directory / "state/evaluation", frame, exports[str(frame)],
                         free=free, version=version)
    exports = {f: str(d) for f, d in exports.items()}
    robots = robot_sources(run_root)
    result = ik(source, ik_dir=workdir, provenance_dir=directory / "provenance",
                name=args.get("name") or "ik", timeout_s=args.get("timeout_s"),
                # Per design version, shared across frames: a rigid robot's
                # link-local geometry is the same at every frame.
                cache_dir=service.native / "ik_cache" / version,
                cancel=cancel, require_scene=False,
                extra_env={"DYCRAFT_IK_EXPORTS": json.dumps(exports),
                           "DYCRAFT_IK_ROBOTS": json.dumps(robots),
                           "DYCRAFT_RUN_DIR": str(run_root)})
    return {**result, "workdir": str(workdir), "script": script, "exports": exports,
            "robots": robots}
