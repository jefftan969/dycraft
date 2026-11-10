"""Staged into the exported world; imported by the agent's IK script.

Everything here is **optional**: a script may ignore this module and drive
cuRobo directly, and that is why the op has no output contract. What it exists
for is the part of the job that is pure re-derivation, and the part that is
*measurably easy to get wrong in a way nothing reports*.

Seven such traps were measured while establishing that cuRobo can solve this
scene at all. Each one produces a plausible arm in the wrong place, or a
correct arm reported as a failure, and none of them raises:

1. **Targets are in the arm's base frame.** The URDF is a standalone robot
   rooted at the origin; the instance sits somewhere in the world. On the
   worked scene that is 622 mm of error if you hand it a world-frame target.
   `world_to_base`.
2. **URDF `q = 0` is not the rest pose.** `render/urdf_export.py` carries
   DyCraft's joint values verbatim — which is the good news, there is no
   conversion — but it has nowhere to record a default configuration, so a
   freshly loaded arm sits at zero on every joint. 17% of the corpus, and every
   interesting Panda joint, has a non-zero default. `default_q` and
   `robot_config`'s `cspace.default_joint_position`.
3. **`result.success` is unreliable.** The tolerances do not propagate onto the
   result object (`position_tolerance` reads back `0.0`), so a solve with 9e-8 m
   of error reports `success=False` — and on another call the same build
   returned `success=True` at 2.18e-08. Compare `position_error` and
   `rotation_error` against your own tolerances; do not read the flag.
4. **`cspace` is not built for you.** `RobotBuilder.build()` leaves
   `joint_names` empty and `default_joint_position` None, and the loader raises
   if any active URDF joint is missing from the list. `joint_names` collects
   them from the URDF, `cam_tilt` and the fingers included.
5. **`compute_metrics` is off by default**, and a thin plate degenerates
   silently: the wrist band (78 x 78 x 12 mm) fits to 4 spheres at coverage
   0.037 unless you ask. `sphere_report()`.
6. **Sphere density must be >= 12** — above upstream's stated practical range.
   At the default the sphere model called a pose DyCraft's exact CAD calls
   clean a 7.8 mm penetration; at 12 it converges to 1-2 mm conservative.
7. **`asset_root_path` is `<export>/bodies/`, not `<export>/`.** The URDF's
   mesh paths are relative to the URDF, so rooting at the export dir resolves
   to `<export>/../meshes/` and every load fails.

**This is a translation layer, not an IK API.** You build the solver: cuRobo is
right there, and `robot_config()` / `scene_config()` hand you the two arguments
every solver and planner in it takes. What lives here is only the two things you
cannot write yourself without getting them silently wrong — facts cuRobo has no
way to know (which self-collisions are structural, where every part is, what
configuration the scene committed) and conversions whose failure mode is a
plausible answer in the wrong place.

There is no solve wrapper here: build the solver yourself and batch targets
through it. A wrapper that constructs an `InverseKinematics` per target pays
CUDA graph capture per target and throws away the batching that is the whole
point.

Nothing here produces the committed animation. A solved `q` is a number the
agent reads, judges, and keys with `key_joint`, or discards.

Runs under the *IK* interpreter, so: stdlib, `_rotation` (staged beside this
file straight from `sdk/_rotation.py`, because a second copy of the unit
conversion would be a silent divergence), and
cuRobo, which is imported **inside** the functions that need it. That last part
is deliberate: the frames-and-units half of this module then works under any
interpreter with no GPU, which is where its tests live.
"""

from __future__ import annotations

import json
from pathlib import Path

import _rotation as _rot

#: Written by `animation/sim/scene_export.py`.
SCENE_META = "scene_meta.json"
#: Written beside the export by `animation/blender/solvers.py` — the facts only
#: dycraft can supply.
IK_META = "ik_meta.json"


# ---------------------------------------------------------------------------
# The two manifests
# ---------------------------------------------------------------------------

def load_meta(ik_dir: str = ".") -> dict:
    """`scene_meta.json` — rest poses, world AABBs, joint limits, and defaults.

    The IK export writes it with the shared scene exporter, `export_sim_scene`.
    """
    return json.loads((Path(ik_dir) / SCENE_META).read_text())


def load_ik_meta(ik_dir: str = ".") -> dict:
    """`ik_meta.json` — what cuRobo needs and the URDF cannot say.

    Self-collision pairs that are structural rather than a fault, and the world
    poses of every static part for the obstacle model. Both are computed by the
    parent, because they come from DyCraft's exact CAD kernel and this
    interpreter cannot import `dycraft.*`.
    """
    return json.loads((Path(ik_dir) / IK_META).read_text())


def _rec(meta: dict, instance: str) -> dict:
    for key in ("free_bodies", "static"):
        for rec in meta.get(key, ()):
            if rec.get("instance") == instance:
                return rec
    known = sorted(r["instance"] for k in ("free_bodies", "static")
                   for r in meta.get(k, ()))
    raise KeyError(f"no instance {instance!r} in {SCENE_META}; it has {known}")


# ---------------------------------------------------------------------------
# Frames  [trap 1]
# ---------------------------------------------------------------------------

def base_pose(instance: str, *, meta: dict):
    """`(translation, quaternion_wxyz)` of the instance's root, in world."""
    p = _rec(meta, instance)["world_pose"]
    return tuple(p["translation"]), tuple(p["quaternion_wxyz"])


def world_to_base(instance: str, position, quaternion_wxyz=None, *, meta: dict):
    """Express a world-frame pose in the instance's base frame.

    A full SE3 inverse, **not** a translation subtraction. Every arm placed so
    far happens to be axis-aligned, so the subtraction passes the only case
    anyone tries by hand and then silently mislocates the first rotated one.
    """
    inv = _rot.se3_inverse((base_pose(instance, meta=meta)))
    t, q = _rot.se3_compose(inv, (tuple(position),
                                  tuple(quaternion_wxyz or (1.0, 0.0, 0.0, 0.0))))
    return (tuple(t), tuple(q)) if quaternion_wxyz is not None else tuple(t)


def base_to_world(instance: str, position, quaternion_wxyz=None, *, meta: dict):
    """The inverse of `world_to_base`; same shape rules."""
    t, q = _rot.se3_compose(base_pose(instance, meta=meta),
                            (tuple(position),
                             tuple(quaternion_wxyz or (1.0, 0.0, 0.0, 0.0))))
    return (tuple(t), tuple(q)) if quaternion_wxyz is not None else tuple(t)


# ---------------------------------------------------------------------------
# Joints: names, units, and the rest pose the URDF cannot carry  [traps 2, 4]
# ---------------------------------------------------------------------------

def joint_records(instance: str, *, meta: dict) -> dict:
    """`{joint: {type, limits, default, axis, unit}}` for the driveable joints.

    Straight out of `scene_meta.json`, which took it from `scene_joint_limits`
    — the same record the agent reads through `probe`. Surface units: degrees
    for revolute, meters for prismatic. Fixed joints are dropped here (they
    appear in the meta so that "missing" and "fixed" can be told apart, but
    they are not part of a configuration vector).
    """
    return {n: r for n, r in (meta.get("joints", {}).get(instance) or {}).items()
            if r.get("type") not in (None, "fixed")}


def joint_names(instance: str, *, meta: dict, ik_meta: dict | None = None,
                ik_dir: str = ".") -> list:
    """Every driveable joint, in the order a `q` vector must use.

    cuRobo's loader raises if an active URDF joint is missing from
    `cspace.joint_names`, and `RobotBuilder` does not fill that list in. The
    joint it is easiest to forget is the one nobody thinks of as part of the
    arm: on the Panda that is `cam_tilt`, plus both fingers.

    **Order is the URDF's, not alphabetical**, and it is read from
    `ik_meta.json` — loaded from `ik_dir` when the caller does not pass one, so
    every call inside a real export gets it whether or not the caller thought
    about it. `active_joints` matches `kinematics.joint_names` exactly.

    That default is not politeness. Sorting puts `cam_tilt` first on the Panda,
    and a `q` built in sorted order and read back in URDF order puts the hand
    **1.02 m** from where the caller asked while every self-consistency check
    still passes — measured, by writing exactly that bug into this module's own
    first end-to-end script. The only place the sorted fallback survives is a
    bare `meta` dict with no export beside it, i.e. a unit test.
    """
    if ik_meta is None:
        try:
            ik_meta = load_ik_meta(ik_dir)
        except (OSError, ValueError):
            ik_meta = None
    if ik_meta is not None and instance in ik_meta.get("instances", {}):
        order = ik_meta["instances"][instance]["active_joints"]
        known = joint_records(instance, meta=meta)
        return [n for n in order if n in known]
    return sorted(joint_records(instance, meta=meta))


def default_q(instance: str, *, meta: dict, ik_meta: dict | None = None,
              ik_dir: str = ".") -> list:
    """The committed rest configuration, in **internal** units, in `joint_names`
    order.

    This is trap 2 in one function. The URDF loads the arm at zero on every
    joint; the scene committed something else. Feeding this to cuRobo as
    `cspace.default_joint_position` is what stops the solver seeding and
    null-space-biasing from a configuration nobody authored — which on a 7-DOF
    arm means exploring the wrong branch while reporting a perfect position
    error.
    """
    recs = joint_records(instance, meta=meta)
    return [_rot.convert_pose_value_to_internal(recs[n]["type"],
                                                float(recs[n].get("default") or 0.0))
            for n in joint_names(instance, meta=meta, ik_meta=ik_meta, ik_dir=ik_dir)]


def joint_limits_internal(instance: str, *, meta: dict) -> dict:
    """`{joint: (lo, hi)}` in internal units — radians, or meters.

    The URDF carries the authored range verbatim, so this and `<limit>`
    agree. Both differ from the meta's own
    numbers only by the unit.
    """
    recs = joint_records(instance, meta=meta)
    out = {}
    for n, r in recs.items():
        lo, hi = r["limits"]
        conv = _rot.convert_pose_value_to_internal
        out[n] = (conv(r["type"], lo), conv(r["type"], hi))
    return out


def q_to_surface(instance: str, q, *, meta: dict, ik_meta: dict | None = None,
                 ik_dir: str = ".") -> dict:
    """An internal-unit `q` vector -> `{joint: surface value}`.

    Surface units are the source's: degrees for a revolute joint and meters
    for a prismatic one, as in `limits`, `default`, and `initial_joints`.
    `key_joint` takes the internal units, radians, so key a solved `q` as it
    is. Getting this backwards is a factor of 57.
    """
    recs = joint_records(instance, meta=meta)
    names = joint_names(instance, meta=meta, ik_meta=ik_meta, ik_dir=ik_dir)
    if len(q) != len(names):
        raise ValueError(f"q has {len(q)} values, {instance} has {len(names)} "
                         f"driveable joints ({names})")
    return {n: _rot.convert_pose_value_to_surface(recs[n]["type"], float(v))
            for n, v in zip(names, q)}


def q_from_surface(instance: str, poses: dict, *, meta: dict,
                   ik_meta: dict | None = None, ik_dir: str = ".") -> list:
    """`{joint: surface value}` -> an internal-unit `q` vector.

    Any joint you leave out takes its committed default, not zero: the omission
    a caller actually means is "leave that one where the scene put it".
    """
    recs = joint_records(instance, meta=meta)
    names = joint_names(instance, meta=meta, ik_meta=ik_meta, ik_dir=ik_dir)
    unknown = sorted(set(poses) - set(names))
    if unknown:
        raise KeyError(f"{instance} has no driveable joint(s) {unknown}; "
                       f"it has {names}")
    conv = _rot.convert_pose_value_to_internal
    defaults = dict(zip(names, default_q(instance, meta=meta, ik_meta=ik_meta,
                                        ik_dir=ik_dir)))
    return [conv(recs[n]["type"], float(poses[n])) if n in poses else defaults[n]
            for n in names]


def to_authoring_joint_track(instance: str, qs, *, meta: dict,
                             ik_meta: dict | None = None,
                             ik_dir: str = ".") -> dict:
    """A list of internal-unit `q` vectors -> `{joint: [surface values]}`.

    Degrees and meters, for writing a pose into `design/`; do not key these
    with `key_joint`, which takes radians. This is the most dangerous function
    in the file: get the unit or the order wrong and the arm goes somewhere
    plausible while every verifier still passes.
    """
    names = joint_names(instance, meta=meta, ik_meta=ik_meta, ik_dir=ik_dir)
    out = {n: [] for n in names}
    for q in qs:
        for n, v in q_to_surface(instance, q, meta=meta, ik_meta=ik_meta,
                                 ik_dir=ik_dir).items():
            out[n].append(v)
    return out


# ---------------------------------------------------------------------------
# Judging a solve  [trap 3]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# cuRobo. Everything below imports it; everything above does not.
# ---------------------------------------------------------------------------

_SPHERE_DENSITY = 12.0
#: Above upstream's stated practical range (0.1-10.0), and deliberately. At the
#: default density the sphere model called a pose dycraft's exact CAD calls
#: clean a 7.8 mm penetration; the arm's wrist band and forearm are only
#: 16.3 mm apart at rest, so the error swamped the clearance. At 12 the
#: disagreement converges to 1-2 mm and stays on the conservative side.


def _cache_key(instance: str, *, ik_meta: dict, sphere_density: float,
               fit_type: str, tool_links, lock_joints, extra_ignore) -> str:
    import hashlib
    rec = ik_meta["instances"][instance]
    payload = json.dumps({
        "object_ref": rec["object_ref"],          # already a content hash
        "sphere_density": sphere_density,
        "fit_type": fit_type,
        "tool_links": sorted(tool_links or []),
        "lock_joints": sorted((lock_joints or {}).items()),
        "extra_ignore": sorted((k, sorted(v))
                               for k, v in (extra_ignore or {}).items()),
        "ignore": {k: sorted(v) for k, v in
                   sorted(rec["self_collision_ignore"].items())},
        "curobo": _curobo_version(),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _curobo_version() -> str:
    try:
        import curobo
        return str(getattr(curobo, "__version__", "unknown"))
    except Exception:                              # noqa: BLE001
        return "unknown"


def _cache_dir():
    import os
    d = os.environ.get("DYCRAFT_IK_CACHE_DIR")
    return Path(d) if d else (Path.home() / ".cache" / "dycraft" / "curobo_robots")


def sphere_report(instance: str, *, ik_dir: str = ".", **kw) -> dict:
    """Per-link `{n, coverage, protrusion}` for the fitted collision model.

    `compute_metrics` is **off** in cuRobo by default, and a degenerate fit is
    otherwise invisible: the wrist band (78 x 78 x 12 mm) comes back as four
    spheres at coverage 0.037 and nothing says so. Read this once when you
    adopt a new object; `protrusion` is the number that matters, because it is
    how far the model sticks out past the real surface and therefore how large
    a clearance it can falsely call a collision.
    """
    return _robot_bundle(instance, ik_dir=ik_dir, **kw)["metrics"]


def _robot_bundle(instance: str, *, ik_dir: str = ".",
                  sphere_density: float = _SPHERE_DENSITY,
                  fit_type: str = "MORPHIT", tool_links=None,
                  lock_joints=None, extra_ignore=None,
                  refit: bool = False) -> dict:
    """The fitted spheres, the self-collision matrix, and the fit metrics.

    Cached on disk under `$DYCRAFT_IK_CACHE_DIR`, keyed on the object's own
    content hash plus every knob that changes the answer. The cache is a
    **correctness** mechanism before it is a speed one: the fit has no seed and
    is observably non-deterministic — 178 spheres on one run and 143 on the
    next, same mesh — so without it two calls over one frozen scene would
    solve against different robots.

    Only the fit is cached. `urdf_path` and `asset_root_path` are filled in per
    call by `robot_config`, so a cache entry can never carry a stale absolute
    path from the run that produced it.
    """
    ik_meta = load_ik_meta(ik_dir)
    rec = ik_meta["instances"][instance]
    key = _cache_key(instance, ik_meta=ik_meta, sphere_density=sphere_density,
                     fit_type=fit_type, tool_links=tool_links,
                     lock_joints=lock_joints, extra_ignore=extra_ignore)
    cache = _cache_dir() / f"{instance}-{key}.json"
    if not refit and cache.is_file():
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass

    from curobo.robot_builder import RobotBuilder
    from curobo._src.geom.sphere_fit.types import SphereFitType

    urdf = str((Path(ik_dir) / rec["urdf"]).resolve())
    # The URDF's mesh paths are relative to the URDF itself, so the asset root
    # is the directory holding it -- rooting at the export dir resolves them to
    # `<export>/../meshes/` and every load fails with "string is not a file".
    b = RobotBuilder(urdf, str((Path(ik_dir) / "bodies").resolve()) + "/",
                     tool_frames=list(tool_links or []) or None)
    b.fit_collision_spheres(sphere_density=sphere_density,
                            fit_type=getattr(SphereFitType, fit_type),
                            use_collision_mesh=True, compute_metrics=True)

    ignore = {k: list(v) for k, v in rec["self_collision_ignore"].items()}
    for k, v in (extra_ignore or {}).items():
        ignore.setdefault(k, [])
        ignore[k] = sorted(set(ignore[k]) | set(v))
    # `prune_collisions=False`: the ignore list is dycraft's own rest-pose
    # answer from the exact CAD kernel, not a 1000-sample guess -- and the
    # sampling path's default batch asks for 66 GiB at this sphere count.
    b.compute_collision_matrix(prune_collisions=False, custom_ignore=ignore)

    bundle = {
        "spheres": b.collision_spheres,
        "self_collision_ignore": {k: list(v) for k, v in b.collision_matrix.items()},
        "metrics": {k: (v if isinstance(v, dict) else str(v))
                    for k, v in (b.link_metrics or {}).items()},
        "n_spheres": int(b.num_spheres),
        "sphere_density": sphere_density,
        "fit_type": fit_type,
        "curobo": _curobo_version(),
        "exact_adjudication": rec.get("exact_adjudication"),
    }
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(bundle))
    except OSError:
        pass
    return bundle


def robot_config(instance: str, *, ik_dir: str = ".",
                 sphere_density: float = _SPHERE_DENSITY,
                 fit_type: str = "MORPHIT", tool_links=None,
                 lock_joints=None, extra_ignore=None,
                 refit: bool = False) -> dict:
    """A cuRobo robot dict for one instance, ready for `InverseKinematicsCfg`.

    Assembles `cspace` by hand, because `RobotBuilder.build()` does not:
    measured, it leaves `joint_names` empty and `default_joint_position` None,
    and the loader raises if any active URDF joint is missing from that list —
    including the ones nobody thinks of as part of the arm, like a camera tilt.

    `cspace.default_joint_position` carries the scene's **committed** rest
    configuration, not zeros. URDF q = 0 is not the rest pose (the exporter has
    nowhere to record one), and a solver seeded and null-space-biased from a
    configuration the scene never chose explores the wrong branch of a 7-DOF
    null space while reporting a perfect position error.
    """
    meta = load_meta(ik_dir)
    ik_meta = load_ik_meta(ik_dir)
    rec = ik_meta["instances"][instance]
    bundle = _robot_bundle(instance, ik_dir=ik_dir,
                           sphere_density=sphere_density, fit_type=fit_type,
                           tool_links=tool_links, lock_joints=lock_joints,
                           extra_ignore=extra_ignore, refit=refit)

    names = joint_names(instance, meta=meta, ik_meta=ik_meta)
    defaults = default_q(instance, meta=meta, ik_meta=ik_meta)
    locked = dict(lock_joints or {})
    links = list(rec["links"])
    tools = list(tool_links or [])
    if not tools:
        raise ValueError(
            f"tool_links is required: name the link(s) you are solving for, "
            f"e.g. tool_links=['hand']. {instance} has {links}")

    return {"robot_cfg": {"kinematics": {
        "format_version": 2.0,
        # Both absolute. A relative `urdf_path` is resolved against cuRobo's
        # own bundled asset root, not the cwd, and fails with "is not a file"
        # naming a directory inside the cuRobo checkout.
        "urdf_path": str((Path(ik_dir) / rec["urdf"]).resolve()),
        "asset_root_path": str((Path(ik_dir) / "bodies").resolve()) + "/",
        "base_link": rec["root_link"],
        "tool_frames": tools,
        "collision_link_names": [n for n in links if bundle["spheres"].get(n)],
        "collision_spheres": bundle["spheres"],
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": bundle["self_collision_ignore"],
        "self_collision_buffer": {n: 0.0 for n in links},
        "mesh_link_names": links,
        "use_global_cumul": True,
        "lock_joints": locked,
        "cspace": {
            "joint_names": names,
            "default_joint_position": defaults,
            "null_space_weight": [1.0] * len(names),
            "cspace_distance_weight": [1.0] * len(names),
            "max_acceleration": 15.0,
            "max_jerk": 500.0,
        },
    }}}


def scene_config(instance: str, *, ik_dir: str = ".", exclude=()) -> dict:
    """The static world as a cuRobo mesh scene, in `instance`'s base frame.

    **Meshes, not bounding boxes.** A table's AABB is a solid block that fills
    the legroom, so an AABB world makes reaching under the tabletop unsolvable
    for a reason that has nothing to do with the arm.

    The solving instance is always excluded — its own links are handled by the
    self-collision model, and feeding them in twice would make every pose a
    collision. `exclude` names anything else to leave out; the usual case is a
    body the arm is deliberately about to touch.
    """
    meta = load_meta(ik_dir)
    ik_meta = load_ik_meta(ik_dir)
    drop = {instance} | set(exclude)
    meshes = []
    for ob in ik_meta.get("obstacles", ()):
        if ob["instance"] in drop:
            continue
        p, q = world_to_base(instance, ob["pose"]["translation"],
                             ob["pose"]["quaternion_wxyz"], meta=meta)
        meshes.append({"name": f"{ob['instance']}__{ob['part']}",
                       "file_path": str((Path(ik_dir) / ob["mesh"]).resolve()),
                       "pose": list(p) + list(q)})
    return {"mesh": {m["name"]: {"file_path": m["file_path"], "pose": m["pose"]}
                     for m in meshes}}


# ---------------------------------------------------------------------------
# A Unitree G1: cuRobo's own set-up, on the scene's robot
# ---------------------------------------------------------------------------

#: cuRobo's whole-body G1 config: floating base, 14 tool frames, collision spheres.
G1_CONFIG = "unitree_g1_29dof_retarget.yml"
#: Frames `G1_CONFIG` names that a pelvis-rooted G1 URDF lacks, placed where cuRobo's
#: own G1 has them (joint, parent, child, xyz). The palm frames carry its hand
#: spheres, which then sit in a gripper's base.
_G1_FRAMES = (("base_fixed", "base_link", "pelvis", "0 0 0"),
              ("pelvis_contour_joint", "pelvis", "pelvis_contour_link", "0 0 0"),
              ("left_hand_palm_joint", "left_wrist_yaw_link", "left_rubber_hand", "0.0415 0.003 0"),
              ("right_hand_palm_joint", "right_wrist_yaw_link", "right_rubber_hand", "0.0415 -0.003 0"))


def g1_robot(urdf: str, *, out_dir: str = ".") -> dict:
    """cuRobo's G1 config (`G1_CONFIG`) on another G1 URDF -- the scene's, from
    `DYCRAFT_IK_ROBOTS` -- as the `robot=` dict every cuRobo solver takes.

    The config's spheres, self-collision table, and floating base are keyed by
    link name, and every `g1_29dof_*` URDF uses the same names, so only the URDF
    changes. Two things make it load: the floating base re-parents `pelvis`, so
    `pelvis` needs a parent (`base_link`), and the config names three frames such
    a URDF lacks (`_G1_FRAMES`). Both are added to a copy written to `out_dir`;
    meshes still resolve beside the original. Measured on the Dex1 G1 against
    cuRobo's own: every tracked frame identical but the wrists (5 mm further out
    on the Dex1's), which a solve on this model then hits (0.7 mm). The Dex1
    fingers past the wrist get no spheres of their own.
    """
    import os
    import re
    import yaml
    import curobo
    src = Path(urdf)
    text = src.read_text()
    live = re.sub(r"<!--.*?-->", "", text, flags=re.S)       # commented-out links do not count
    have = set(re.findall(r'<link\s+name="([^"]+)"', live))
    # A frame is missing when the link it creates is: `base_link` for the base joint
    # (whose child, `pelvis`, exists), the child for the rest.
    add = [f for f in _G1_FRAMES if (f[1] if f[1] == "base_link" else f[2]) not in have]
    links = "".join(f'\n  <link name="{f[1] if f[1] == "base_link" else f[2]}"/>' for f in add)
    joints = "".join(f'\n  <joint name="{j}" type="fixed"><origin xyz="{o}" rpy="0 0 0"/>'
                     f'<parent link="{pa}"/><child link="{c}"/></joint>' for j, pa, c, o in add)
    at = text.index(">", text.index("<robot")) + 1
    dst = Path(out_dir).resolve() / f"{src.stem}_curobo.urdf"
    dst.write_text(text[:at] + links + joints + "\n" + text[at:])
    content = os.path.join(os.path.dirname(curobo.__file__), "content")
    cfg = yaml.safe_load(open(os.path.join(content, "configs", "robot", G1_CONFIG)))
    cfg = cfg.get("robot_cfg", cfg)
    cfg["kinematics"]["urdf_path"] = str(dst)
    cfg["kinematics"]["asset_root_path"] = str(src.resolve().parent)
    return {"robot_cfg": cfg}


__all__ = [
    "SCENE_META", "IK_META", "load_meta", "load_ik_meta",
    "G1_CONFIG", "g1_robot",
    "base_pose", "world_to_base", "base_to_world",
    "joint_records", "joint_names", "default_q", "joint_limits_internal",
    "q_to_surface", "q_from_surface", "to_authoring_joint_track",
    "robot_config", "sphere_report", "scene_config",
]
