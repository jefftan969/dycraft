"""Internal rotation and SE3 helpers for the DyCraft object SDK.

All internal angle math is in radians. The only degree → radian conversion
happens at the API boundary via deg_to_rad and euler_xyz_deg_to_quat_wxyz,
which are called from object_builder before values are stored in the IR.

SE3 representation everywhere in this module:

    se3 = ((tx, ty, tz), (qw, qx, qy, qz))

Quaternions are WXYZ. Functions that produce quaternions intended for
storage in the IR call sign_canonicalize_quat to enforce qw >= 0 (and a
deterministic tie-break when qw == 0).

Pure-Python; no numpy dependency.
"""

import math


# ---------------------------------------------------------------------------
# Scalar / vector helpers
# ---------------------------------------------------------------------------

def deg_to_rad(deg: float) -> float:
    return float(deg) * (math.pi / 180.0)


def rad_to_deg(rad: float) -> float:
    return float(rad) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Pose-dict unit conversion (surface ↔ IR-internal)
# ---------------------------------------------------------------------------
#
# The SDK surface convention is the same as for joint `limits` / `default`:
# revolute pose values are in DEGREES; prismatic pose values are in METERS.
# The IR + kernel + CCD sampler operate exclusively in IR-internal units
# (radians for revolute, meters for prismatic). Conversion happens at every
# boundary where a pose dict crosses between the two domains:
#
#   * Builder serialization of pose-tagged refs and assertion-level `poses=`
#     shorthand into the IR (surface → internal).
#   * Probe entry points that accept `poses=` from the agent
#     (surface → internal).
#   * Verifier evidence (e.g. NoSelfPenetration `witnessing_pose`) which
#     originates from the radians-domain CCD sampler (internal → surface).
#
# `joint_types` is a mapping `{joint_name: "revolute"|"prismatic"|"fixed"}`.
# Pose keys absent from the mapping are passed through unchanged (matches
# the universal "unrelated keys are silently ignored" rule for pose dicts).


def convert_pose_value_to_internal(joint_type: str, q: float) -> float:
    """Convert one surface-unit pose value to IR-internal units."""
    if joint_type == "revolute":
        return deg_to_rad(q)
    return float(q)


def convert_pose_value_to_surface(joint_type: str, q: float) -> float:
    """Convert one IR-internal pose value to surface units."""
    if joint_type == "revolute":
        return rad_to_deg(q)
    return float(q)


def _convert_pose_dict(poses, joint_types, *, to_internal: bool):
    """Internal: per-key dispatch for surface↔internal pose-dict conversion.

    `poses` may be None, an empty dict, a `{joint_name: value}` mapping,
    or the SDK's frozen `((name, value), ...)` tuple form. Returns a
    new dict (or None if input was None). Keys absent from `joint_types`
    are passed through unchanged. Keys whose joint is `fixed` are also
    passed through (no articulation value to convert).
    """
    if poses is None:
        return None
    if isinstance(poses, tuple):
        poses = dict(poses)
    if not poses:
        return {}
    out = {}
    for name, q in poses.items():
        jt = joint_types.get(name)
        if jt is None or jt == "fixed":
            out[name] = float(q)
        elif to_internal:
            out[name] = convert_pose_value_to_internal(jt, q)
        else:
            out[name] = convert_pose_value_to_surface(jt, q)
    return out


def convert_poses_to_internal(poses, joint_types):
    """Convert a surface-unit pose dict to IR-internal units in one pass."""
    return _convert_pose_dict(poses, joint_types, to_internal=True)


def convert_poses_to_surface(poses, joint_types):
    """Convert an IR-internal pose dict to surface units in one pass."""
    return _convert_pose_dict(poses, joint_types, to_internal=False)


def object_joint_types_from_ir(ir) -> dict:
    """Build a `{joint_name: joint_type}` map from an object IR."""
    return {j["name"]: j["type"] for j in ir.get("joints", [])}


def scene_joint_types_from_catalog(catalog) -> dict:
    """Build a `{joint_name_or_qualified: joint_type}` map from a scene catalog.

    Scene-layer pose keys are either bare (`"hinge"` — applies to every
    instance with a `hinge` joint) or qualified (`"<instance>.<hinge>"`).
    For a bare name, every instance that has it must share the joint type
    (LLM-authored objects all author the same `hinge` as revolute / etc.);
    if instances disagree we keep the first-seen type — the conversion
    layer is best-effort here, since the conflict would surface as a
    verifier error elsewhere.

    `catalog` is the scene's `objects: {object_ref: ObjectHandle-like}`
    mapping; this helper reads `.ir` / `["joints"]` on each entry.
    """
    out: dict = {}
    if catalog is None:
        return out
    for object_ref, handle in catalog.items():
        ir = getattr(handle, "ir", None)
        if ir is None and isinstance(handle, dict):
            ir = handle
        if not ir:
            continue
        for j in ir.get("joints", []):
            jt = j["type"]
            name = j["name"]
            out.setdefault(name, jt)
    return out


def scene_joint_type(name: str, ir, catalog, bare_types=None):
    """The joint type for a scene-layer pose key, or `None` if unresolvable.

    A key is either qualified (`"<instance>.<joint>"`) or bare (`"hinge"`).
    A qualified key is resolved on **that instance's own object**, which is the
    only correct answer when two objects in one scene share a joint name with
    different types — `scene_joint_types_from_catalog` keys by bare name and
    keeps the first-seen type, so asking it about `"cart.slide"` returns
    nothing and asking it about `"slide"` may return the *other* object's type.

    `None` means "do not convert". Returning it rather than guessing is the
    point: a caller that defaulted to `"revolute"` would scale a prismatic
    joint's meters by 180/pi, which is a factor of 57 in an error message
    nobody can trace back to here.

    `bare_types` is an optional prebuilt `scene_joint_types_from_catalog(...)`,
    so a loop over many poses does not rebuild it per key.
    """
    if "." not in name:
        if bare_types is None:
            bare_types = scene_joint_types_from_catalog(catalog)
        return bare_types.get(name)
    inst_name, joint_name = name.split(".", 1)
    inst = {i["name"]: i for i in (ir or {}).get("instances", [])}.get(inst_name)
    if inst is None:
        return None
    handle = catalog.get(inst.get("object_ref")) if catalog else None
    if handle is None:
        return None
    inst_ir = getattr(handle, "ir", None) or handle
    for j in inst_ir.get("joints", []):
        if j["name"] == joint_name:
            return j["type"]
    return None


def convert_scene_poses_to_internal(poses, ir, catalog):
    """Convert scene-layer surface poses to IR-internal units.

    Handles qualified keys `"<instance>.<joint>"` by looking up the joint
    type on that specific instance's object_ref, and bare keys via the
    first-matching joint across the catalog. Accepts dict or frozen
    `((name, value), ...)` tuple input.
    """
    if poses is None:
        return None
    if isinstance(poses, tuple):
        poses = dict(poses)
    if not poses:
        return {}
    bare_types = scene_joint_types_from_catalog(catalog)
    out: dict = {}
    for name, q in poses.items():
        jt = scene_joint_type(name, ir, catalog, bare_types)
        if jt is None or jt == "fixed":
            out[name] = float(q)
        else:
            out[name] = convert_pose_value_to_internal(jt, q)
    return out


def convert_scene_poses_to_surface(poses, ir, catalog):
    """Inverse of convert_scene_poses_to_internal."""
    if poses is None:
        return None
    if isinstance(poses, tuple):
        poses = dict(poses)
    if not poses:
        return {}
    bare_types = scene_joint_types_from_catalog(catalog)
    out: dict = {}
    for name, q in poses.items():
        jt = scene_joint_type(name, ir, catalog, bare_types)
        if jt is None or jt == "fixed":
            out[name] = float(q)
        else:
            out[name] = convert_pose_value_to_surface(jt, q)
    return out


# ---------------------------------------------------------------------------
# Quaternion (wxyz) operations
# ---------------------------------------------------------------------------

def normalize_quat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    return (w / n, x / n, y / n, z / n)


def sign_canonicalize_quat(q):
    """Force qw >= 0; tie-break by sign of qx, then qy, then qz when qw == 0.

    This makes two equivalent unit quaternions produce identical tuples,
    which is required for byte-stable IR hashing.
    """
    w, x, y, z = q
    if w < 0.0:
        return (-w, -x, -y, -z)
    if w == 0.0:
        if x < 0.0:
            return (w, -x, -y, -z)
        if x == 0.0:
            if y < 0.0:
                return (w, x, -y, -z)
            if y == 0.0 and z < 0.0:
                return (w, x, y, -z)
    return (w, x, y, z)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_rotate_vec(q, v):
    """Rotate a 3-vector v by unit quaternion q (wxyz)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(qvec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w * t + cross(qvec, t)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_from_axis_angle_rad(axis_unit, angle_rad: float):
    """Build a WXYZ quaternion from a *unit* axis and angle in radians."""
    half = 0.5 * angle_rad
    s = math.sin(half)
    return (math.cos(half), axis_unit[0] * s, axis_unit[1] * s, axis_unit[2] * s)


def quat_angle_deg(a, b) -> float:
    """Geodesic angle in degrees between two rotations.

    `q` and `-q` are the same rotation, so the sign is taken out before the
    arccos — otherwise a quaternion that happened to be negated reads as
    "almost 360 degrees away" from an identical one.
    """
    d = abs(sum(a[k] * b[k] for k in range(4)))
    return rad_to_deg(2.0 * math.acos(min(1.0, d)))


def euler_xyz_deg_to_quat_wxyz(rx_deg: float, ry_deg: float, rz_deg: float):
    """Intrinsic Euler XYZ in degrees -> WXYZ quaternion.

    Convention matches Blender's rotation_mode='XYZ':
        R_total = Rx(rx) * Ry(ry) * Rz(rz)
    where each Ri is a rotation about the respective local axis.
    """
    rx = deg_to_rad(rx_deg)
    ry = deg_to_rad(ry_deg)
    rz = deg_to_rad(rz_deg)
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    qx = (cx, sx, 0.0, 0.0)
    qy = (cy, 0.0, sy, 0.0)
    qz = (cz, 0.0, 0.0, sz)
    return quat_mul(quat_mul(qx, qy), qz)


# Gimbal-lock threshold: if |sin(ry)| >= 1 - _GIMBAL_EPS we pin rz = 0 and
# solve rx from the remaining degree of freedom. This is the documented
# convention for quat -> intrinsic-XYZ Euler recovery.
_GIMBAL_EPS = 1.0e-9


def quat_to_euler_xyz_deg(q):
    """WXYZ unit quaternion -> intrinsic Euler XYZ in degrees.

    Inverse of euler_xyz_deg_to_quat_wxyz up to gimbal-lock ambiguity. At
    the singularity (|sin(ry)| ~ 1, i.e. |ry| ~ 90 deg) the decomposition
    is not unique; we pin rz = 0 and solve rx from the remaining DOF.

    The rotation matrix entries used below come from
        R = Rx(rx) * Ry(ry) * Rz(rz)
    which gives, for unit quaternion (w, x, y, z):

        R[0][2] = +sin(ry)              = 2*(x*z + w*y)
        R[1][2] = -sin(rx)*cos(ry)      = 2*(y*z - w*x)
        R[2][2] = +cos(rx)*cos(ry)      = 1 - 2*(x*x + y*y)
        R[0][0] = +cos(ry)*cos(rz)      = 1 - 2*(y*y + z*z)
        R[0][1] = -cos(ry)*sin(rz)      = 2*(x*y - w*z)

    so rx = atan2(-R[1][2], R[2][2]), ry = asin(R[0][2]),
    rz = atan2(-R[0][1], R[0][0]). At the singularity cos(ry) = 0 we
    fall back to rz = 0, rx = atan2(R[2][1], R[1][1]).
    """
    w, x, y, z = q
    # Rotation matrix entries (only the ones we need).
    sin_ry = 2.0 * (x * z + w * y)
    # Clamp into [-1, 1] to absorb tiny float drift.
    if sin_ry > 1.0:
        sin_ry = 1.0
    elif sin_ry < -1.0:
        sin_ry = -1.0
    if abs(sin_ry) >= 1.0 - _GIMBAL_EPS:
        # Gimbal lock: ry = +/- 90 deg. Pin rz = 0 and solve rx from R[1][*].
        # With R = Rx(rx)*Ry(+/-90)*Rz(0):
        #   sin_ry = +1: R[1][0] = sin(rx),  R[1][1] = cos(rx)
        #   sin_ry = -1: R[1][0] = -sin(rx), R[1][1] = cos(rx)
        # So rx = atan2(sign(sin_ry) * R[1][0], R[1][1]).
        ry = math.asin(sin_ry)
        r10 = 2.0 * (x * y + w * z)          # R[1][0]
        r11 = 1.0 - 2.0 * (x * x + z * z)    # R[1][1]
        sign = 1.0 if sin_ry >= 0.0 else -1.0
        rx = math.atan2(sign * r10, r11)
        rz = 0.0
    else:
        ry = math.asin(sin_ry)
        # rx = atan2(-R[1][2], R[2][2])
        rx = math.atan2(-2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
        # rz = atan2(-R[0][1], R[0][0])
        rz = math.atan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (y * y + z * z))
    deg = 180.0 / math.pi
    return (rx * deg, ry * deg, rz * deg)


# ---------------------------------------------------------------------------
# SE3 operations: se3 = (translation_tuple3, quat_wxyz_tuple4)
# ---------------------------------------------------------------------------

IDENTITY_SE3 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def se3_compose(a, b):
    """Return a o b. Semantics: maps b-frame points to a-parent frame.

    c.R = a.R * b.R
    c.t = a.t + a.R * b.t
    """
    at, aq = a
    bt, bq = b
    rotated_bt = quat_rotate_vec(aq, bt)
    new_t = (at[0] + rotated_bt[0], at[1] + rotated_bt[1], at[2] + rotated_bt[2])
    new_q = quat_mul(aq, bq)
    return (new_t, new_q)


def se3_inverse(se3):
    t, q = se3
    q_inv = quat_conj(q)  # unit quat assumed
    neg_t = (-t[0], -t[1], -t[2])
    new_t = quat_rotate_vec(q_inv, neg_t)
    return (new_t, q_inv)
