"""Pure per-primitive geometry registry for the DyCraft object SDK.

Single source of truth for the geometry *facts* of each primitive shape:
parameter schema, local-frame AABB, solid inertia, and size metrics.

Dependency-light by design: **stdlib (math) only**. No Open3D, no XML, no
SDK-builder imports. Renderer-specific mesh construction lives in
``render/open3d_renderer.py`` and URDF XML lives in ``render/urdf_export.py``;
both call this module for geometry facts rather than re-deriving them.

Shape set: ``box`` (an analytic primitive, centered at the part-local origin),
``solid`` (a CAD-authored B-rep, the native shape for build123d-authored parts)
and ``imported`` (a third-party asset's triangle meshes). Both asset tags carry
their local AABB and MAY be off-center — their geometry need not straddle the
part origin — so URDF/Open3D consume these without re-framing, and the existing
``inv(child_anchor)`` link-frame offset logic is shape-independent.

Solid params hold ``{source, brep, mesh, digest, aabb_min, aabb_max, shape}``
— run-relative paths to the authored build123d source, the exact B-rep, and a
tessellated OBJ for rendering, plus the derived ``shape`` fact table (exact
volume / centroid / unit-density inertia diagonal, and the feature inventory
the agent reads). A solid has **no** ``scale``: the source is authoritative
about size, and a post-hoc scale would silently desync the exact mass
properties in ``shape`` from the geometry.

Imported params hold ``{mesh, collision, digest, collision_digest, aabb_min,
aabb_max, shape}`` — the same table, minus the two things a third-party asset
cannot have (an exact B-rep and an authored source) and plus a separate
collision triangulation. See ``PRIMITIVE_IMPORTED`` for why each of those three
differences exists. No ``scale`` here either, for the solid's reason and one
more: the narrow phase keys its BVH cache off the asset digest alone.

``box`` (``Object.make_box``) is the tests' fixture geometry, not an authoring
option: the agent's only geometry mechanism is ``add_part`` with a build123d
Shape.

Conservative collision note: ``local_aabb`` bounds an asset part by its
(axis-aligned) AABB, so it is treated conservatively (a slight over-report where
the true surface does not fill the box). That over-report is adjudicated away
downstream in ``verify/_exact_adjudicator.py`` — the AABB/OBB result stays the
cheap broad+mid phase. Which narrow phase answers differs by tag: a ``solid``
can be adjudicated exactly from its B-rep, while an ``imported`` part has none
and is answered by the mesh backend from ``collide_asset_key``'s asset.

Adding a shape later = one new branch in each dispatch function below.
"""

import math


PRIMITIVE_BOX = "box"
PRIMITIVE_SOLID = "solid"
#: Geometry that came from outside DyCraft — a third-party URDF's meshes, staged
#: by ``agent/assets/import_urdf.py``. ``solid`` minus ``brep`` and
#: ``source``, plus its own collision asset:
#:
#:   * no ``brep``  -> no exact B-rep adjudication (see ``_exact_adjudicator``,
#:     whose brep branch still tests ``== PRIMITIVE_SOLID``). The mesh backend
#:     answers for it instead, from ``params["collision"]``.
#:   * no ``source`` -> ``_object_kernel.part_source_path`` returns None, so
#:     there is no authored recipe for an agent to read back or edit. That is
#:     what makes an imported asset frozen, and it is enforced structurally
#:     rather than by a flag: no SDK surface produces this tag at all
#:     (``Object.add_part`` takes a live build123d Shape and nothing else), so
#:     an authored ``object.py`` cannot contain one.
#:   * a real ``shape`` fact table, unlike the retired ``mesh`` tag's AABB box
#:     proxy — integrated from the triangles at import time. That is what lets
#:     ``inertia`` / ``inertia_origin`` serve both asset tags
#:     through one expression instead of two.
PRIMITIVE_IMPORTED = "imported"

# Minimum box-proxy extent used for a solid's collision/inertia so a flat or
# degenerate part still yields a non-zero mass.
_MESH_MIN_EXTENT = 1.0e-4

# Required param keys per *analytic* primitive, in canonical order. Solid params
# are heterogeneous (path strings + AABB vectors + a digest), so they are NOT
# listed here; they are validated by their own branch in ``validate_params``.
PARAM_KEYS = {
    PRIMITIVE_BOX: ("width", "length", "height"),
}

# Required keys of a solid's derived ``shape`` fact table. Extra keys (the
# feature inventory: face census, plane/cylinder tables, topology counts) are
# carried through untouched — this module only needs the mass properties.
_SHAPE_REQUIRED = ("volume", "center", "inertia_diag")

# All registered shape tags. ``is_known`` / ``_require_known`` gate on this so
# every tag flows through every geometry consumer (kernel, anchors, support,
# URDF, renderer) that dispatches on it.
_KNOWN_PRIMITIVES = set(PARAM_KEYS) | {PRIMITIVE_SOLID, PRIMITIVE_IMPORTED}

# Tags whose geometry lives in an external asset rather than scalar dimensions,
# and whose local AABB is stored rather than derived. Consumers asking "is this
# an authored asset?" test membership here rather than comparing against
# ``PRIMITIVE_SOLID`` directly, so adding a second asset-backed tag is a
# one-line change and no call site can be missed.
ASSET_PRIMITIVES = (PRIMITIVE_SOLID, PRIMITIVE_IMPORTED)

# Which params key holds the run-relative *triangle-mesh* asset each asset-
# backed tag renders from. Path-rewriting consumers — the renderer's base-dir
# resolution, the URDF exporter's asset localization — look the key up here
# rather than hardcoding it.
_RENDER_ASSET_KEY = {
    PRIMITIVE_SOLID: "mesh",
    PRIMITIVE_IMPORTED: "mesh",
}

# Which params key holds the triangle mesh the NARROW PHASE collides, which is
# not always the one the renderer draws. A ``solid`` has one triangulation
# serving both (its exact geometry is the ``.brep`` beside it, and the OBJ is
# what the mesh backend falls back to). An ``imported`` part has two, because a
# third-party asset ships them separately and they differ by more than detail:
# the Panda's visual meshes total 102,752 triangles against 2,680 for its
# collision meshes, and several of the visual ones are open shells that no
# containment probe can answer about. Colliding the visual mesh would therefore
# be both ~38x slower and less correct.
_COLLIDE_ASSET_KEY = {
    PRIMITIVE_SOLID: "mesh",
    PRIMITIVE_IMPORTED: "collision",
}


def render_asset_key(primitive):
    """Params key of `primitive`'s triangle-mesh asset, or None if it has none.

    Returns ``"mesh"`` for both asset tags, and ``None`` for the analytic
    primitives (whose geometry is scalar dimensions).
    """
    return _RENDER_ASSET_KEY.get(primitive)


def collide_asset_key(primitive):
    """Params key of the triangle mesh the mesh backend should collide, or None.

    Sibling of `render_asset_key`, and deliberately a separate table: for
    ``solid`` the two agree, so this returns exactly what the adjudicator used
    to hardcode, and for ``imported`` they must not.
    """
    return _COLLIDE_ASSET_KEY.get(primitive)


def is_known(primitive) -> bool:
    """True if `primitive` is a registered shape tag (box/solid/imported)."""
    return primitive in _KNOWN_PRIMITIVES


def _require_known(primitive) -> None:
    if primitive not in _KNOWN_PRIMITIVES:
        raise ValueError(f"unsupported primitive {primitive!r}")


def _validate_vec3(value, label: str):
    """Return a 3-tuple of finite floats from `value`, or raise ValueError."""
    try:
        seq = tuple(value)
    except TypeError:
        raise ValueError(f"mesh param {label!r} must be a 3-sequence, got {value!r}")
    if len(seq) != 3:
        raise ValueError(f"mesh param {label!r} must have length 3, got {value!r}")
    out = []
    for i, c in enumerate(seq):
        v = float(c)
        if not math.isfinite(v):
            raise ValueError(f"mesh param {label}[{i}] must be finite, got {v!r}")
        out.append(v)
    return tuple(out)


#: How small a negative moment has to be, against the largest moment in the
#: same tensor, to be integrator round-off rather than a broken integral.
#:
#: Relative because the round-off scales with the integral. 1e-6 because that
#: clears, by twelve orders, every negative the pinned kernel can be made to
#: produce: three spline-bounded thin solids integrated with
#: `VolumePropertiesGK_s` return smallest moments of -2.85e-55, -5.15e-26 and
#: -2.02e-28 against largest moments of 8.6e-8 and 1.5e-8, ratios of 3e-48 to
#: 1e-18, because the moment being integrated is a near-total cancellation.
#:
#: Nothing physical is lost by zeroing a value this small either: a
#: legitimately slender part can have a smallest principal moment a billionth
#: of its largest (a 3 m x 0.1 mm rod measures 2.2e-9), and such a moment is
#: zero for any purpose a rigid body has.
#:
#: **What this does NOT fix, measured.** Of the three `finalize` rejections in
#: a recorded 24-object batch, none is round-off and none is reachable by any
#: relative tolerance: a helmet visor's negative moment is 9.15e-3 of its
#: largest — nine thousand times this figure — and for a saddle shell and a
#: sprayer hose the negative moment IS the largest. The hose's
#: off-diagonal terms exceed its diagonal and its trace is near zero, so the
#: tensor is not positive-definite at all. Those are non-physical tensors, not
#: noisy ones, and widening the tolerance to admit them would hand the physics
#: engine a fabricated mass property. They get a diagnosis instead — see the
#: message below.
#:
#: A negative that is a *significant* fraction of the tensor is not round-off —
#: the integration failed — so it still raises. Clamping that would hand the
#: physics engine a fabricated mass property, which is worse than a refusal.
_INERTIA_ROUNDOFF_REL = 1.0e-6


def _clamp_inertia_roundoff(diag, label: str):
    """Zero negative inertia components that are integrator noise; raise on
    the rest.

    Spline-bounded solids — a lofted visor, a swept hose, a shelled saddle —
    integrate numerically, and a thin one can come back with a principal
    moment a hair below zero. Refusing those cost four recorded builds a
    compile each, with advice ("this is an internal inconsistency; report
    it") that an author could act on in no way at all.
    """
    scale = max(abs(v) for v in diag) if diag else 0.0
    tol = _INERTIA_ROUNDOFF_REL * scale
    out = []
    for i, v in enumerate(diag):
        if v >= 0.0:
            out.append(v)
            continue
        if -v <= tol:
            out.append(0.0)
            continue
        raise ValueError(
            f"{label} shape inertia_diag[{i}] must be >= 0, got {v!r}. It is "
            f"{abs(v) / scale:.3g} of the largest moment in the same tensor "
            f"({scale!r}), exceeding the round-off allowance. Check the "
            f"shape and mass-property calculation. Thin spline-bounded "
            f"solids can integrate poorly: inspect `n_solids` for unintended "
            f"disconnected bodies, then wall thickness and surface complexity. "
            f"Where the design permits, thicken the wall, use analytic "
            f"extruded/revolved faces, or split the body into simpler pieces. "
            f"For parts that register, `fill_ratio` and `face_census` help "
            f"diagnose thinness and spline complexity. Do not patch the "
            f"reported inertia to hide the failure."
            if scale > 0.0 else
            f"{label} shape inertia_diag[{i}] must be >= 0, got {v!r}, and "
            f"every moment in the tensor is zero, so the shape has no "
            f"measurable extent.")
    return tuple(out)


def _validate_shape_table(shape, label: str = "solid") -> dict:
    """Structural validation for an asset part's derived ``shape`` fact table.

    `label` names the tag in the error messages only, and defaults to ``solid``
    so every message this raised before is unchanged. Both asset tags carry the
    same table — that is the point of integrating an imported mesh's mass
    properties at import time rather than proxying them off its AABB.

    Requires the mass properties this module dispatches on: ``volume`` (finite,
    >= 0), ``center`` (the true centroid in part-local — NOT the AABB center),
    and ``inertia_diag`` (the inertia tensor's diagonal about that centroid at
    **unit density**, so callers scale by their own density). Every other key —
    the feature inventory the agent reads — is carried through untouched.
    """
    if not isinstance(shape, dict):
        raise ValueError(f"{label} param 'shape' must be a dict, got {shape!r}")
    for key in _SHAPE_REQUIRED:
        if key not in shape:
            raise ValueError(f"{label} shape table missing {key!r}")
    volume = float(shape["volume"])
    if not math.isfinite(volume):
        raise ValueError(f"{label} shape 'volume' must be finite, got {volume!r}")
    if volume < 0.0:
        raise ValueError(f"{label} shape 'volume' must be >= 0, got {volume!r}")
    out = dict(shape)
    out["volume"] = volume
    out["center"] = _validate_vec3(shape["center"], "shape.center")
    diag = _validate_vec3(shape["inertia_diag"], "shape.inertia_diag")
    diag = _clamp_inertia_roundoff(diag, label)
    out["inertia_diag"] = diag
    return out


def _validate_imported_params(params) -> dict:
    """Structural validation for an ``imported`` part's params.

    Requires two non-empty run-relative asset paths — ``mesh`` (the visual
    triangulation the renderer and the URDF visual read) and ``collision`` (what
    the mesh backend collides) — a content ``digest`` for each, an
    ``aabb_max >= aabb_min`` pair, and the derived ``shape`` fact table.

    ``collision`` is REQUIRED rather than defaulted to ``mesh``. A default would
    make the expensive mistake silent: the narrow phase would build BVHs over
    the full-detail visual mesh and, for the links whose visual mesh is an open
    shell, answer containment queries about a surface that has no inside. A
    missing key is a loud error; a 38x slowdown that also changes the answer is
    not.

    ``brep``, ``source`` and ``scale`` are REJECTED, not ignored:

    * ``brep`` — an imported triangle soup has no exact geometry, and accepting
      the key would let a caller believe the exact adjudicator can answer about
      this part when its brep branch deliberately does not match this tag.
    * ``source`` — the tag exists precisely because there is no authored recipe.
      A part claiming one would be offering the object agent something to edit.
    * ``scale`` — same argument ``_validate_solid_params`` makes (the asset is
      authoritative about its own size, and a post-hoc scale desyncs the exact
      volume/centroid/inertia in ``shape``), with extra teeth here: the narrow
      phase keys its BVH cache off the asset digest alone, so one OBJ used at
      two scales would silently share one collision model.
    """
    out = {}
    for key in ("mesh", "collision", "digest", "collision_digest"):
        val = params.get(key)
        if not isinstance(val, str) or not val:
            raise ValueError(
                f"imported param {key!r} must be a non-empty string")
        out[key] = val
    for key in ("brep", "source", "scale"):
        if key in params:
            raise ValueError(
                f"imported params must not carry {key!r} — an imported asset "
                "has no exact B-rep and no authored source, and its geometry "
                "is authoritative about its own size; bake any transform into "
                "the staged mesh at import time")
    if "aabb_min" not in params or "aabb_max" not in params:
        raise ValueError("imported params missing 'aabb_min'/'aabb_max'")
    aabb_min = _validate_vec3(params["aabb_min"], "aabb_min")
    aabb_max = _validate_vec3(params["aabb_max"], "aabb_max")
    for i in range(3):
        if aabb_max[i] < aabb_min[i]:
            raise ValueError(
                f"imported aabb_max[{i}]={aabb_max[i]!r} must be >= "
                f"aabb_min[{i}]={aabb_min[i]!r}")
    if "shape" not in params:
        raise ValueError("imported params missing 'shape' fact table")
    out["aabb_min"] = aabb_min
    out["aabb_max"] = aabb_max
    out["shape"] = _validate_shape_table(params["shape"], "imported")
    # Optional: how far the collision mesh escapes the visual AABB above, in
    # meters. The two bodies differ, and the box is the visual one's so that
    # anchors land on the visible surface — which leaves the broad phase
    # bounding a body it does not measure. Bundles converted before this key
    # existed simply carry no margin and behave as they did.
    escape = params.get("collide_escape_m")
    if escape is not None:
        escape = float(escape)
        if not (escape >= 0.0) or escape == float("inf"):
            raise ValueError(
                f"imported collide_escape_m must be a finite, non-negative "
                f"number, got {params['collide_escape_m']!r}")
        out["collide_escape_m"] = escape
    # Optional: what the source asset *declared* about its own mass, as
    # distinct from what `shape` measures off its geometry. An asset with real
    # `<inertial>` data knows its mass and center of mass; uniform density over
    # a mesh drawn to be seen does not. The G1 declares 34.13 kg against
    # Unitree's ~35 and measures 20.90 kg, with one foot 299x light because
    # that link's collision proxy is 2 cm^3 — so a consumer that needs a center
    # of mass should prefer this and fall back to `shape.center`.
    #
    # `com` is in the part's own frame, the frame `shape.center` uses: a URDF
    # `<inertial><origin>` is relative to the link, and the importer bakes each
    # `<visual><origin>` into the staged mesh, so the two are comparable
    # without a transform. Bundles converted before this key existed carry no
    # `inertial` and fall back as they did.
    #
    # Mass and center only. The full tensor has its own documented lossiness
    # (`shape.inertia_diag` keeps a diagonal) and its own deferred fix; adding
    # a second half-faithful copy of it here would muddy both.
    inertial = params.get("inertial")
    if inertial is not None:
        if not isinstance(inertial, dict):
            raise ValueError(f"imported inertial must be an object, got {inertial!r}")
        if inertial.get("mass") is None or inertial.get("com") is None:
            raise ValueError("imported inertial needs both mass and com")
        mass = float(inertial["mass"])
        if not (mass > 0.0) or mass == float("inf"):
            raise ValueError(
                f"imported inertial.mass must be a finite positive number, "
                f"got {inertial['mass']!r}")
        out["inertial"] = {"mass": mass,
                           "com": _validate_vec3(inertial["com"], "inertial.com"),
                           "source": str(inertial.get("source") or "declared")}
    return out


def _validate_solid_params(params) -> dict:
    """Structural validation for a ``solid`` part's params.

    Requires two non-empty run-relative asset paths (``brep`` = the exact
    B-rep consumed by the collision adjudicator, ``mesh`` = a tessellated OBJ
    for the renderer and the URDF visual), a content ``digest``, an
    ``aabb_max >= aabb_min`` pair, and the derived ``shape`` fact table.

    The authored program is deliberately **not** here. One `object.py` builds
    every part, so it is an object-level fact and lives at ``ir["source"]``;
    requiring it per part is what the retired per-solid authoring design left
    behind.

    Unlike ``mesh`` there is deliberately no ``scale``: the source is
    authoritative about size, and scaling after the fact would desync the exact
    volume/centroid/inertia in ``shape`` from the geometry they describe.
    """
    out = {}
    for key in ("brep", "mesh", "digest"):
        val = params.get(key)
        if not isinstance(val, str) or not val:
            raise ValueError(f"solid param {key!r} must be a non-empty string")
        out[key] = val
    if "aabb_min" not in params or "aabb_max" not in params:
        raise ValueError("solid params missing 'aabb_min'/'aabb_max'")
    aabb_min = _validate_vec3(params["aabb_min"], "aabb_min")
    aabb_max = _validate_vec3(params["aabb_max"], "aabb_max")
    for i in range(3):
        if aabb_max[i] < aabb_min[i]:
            raise ValueError(
                f"solid aabb_max[{i}]={aabb_max[i]!r} must be >= "
                f"aabb_min[{i}]={aabb_min[i]!r}")
    if "scale" in params:
        raise ValueError(
            "solid params must not carry 'scale' — the authored source is "
            "authoritative about size; a post-hoc scale would desync the "
            "exact mass properties in 'shape'")
    if "shape" not in params:
        raise ValueError("solid params missing 'shape' fact table")
    out["aabb_min"] = aabb_min
    out["aabb_max"] = aabb_max
    out["shape"] = _validate_shape_table(params["shape"])
    return out


def validate_params(primitive, params) -> dict:
    """Structural validation of a primitive's params.

    Analytic primitives (box): every required key must be present, finite, and
    strictly positive; returns a dict of the validated float values (canonical
    keys only). The asset tags have their own branches, because their params are
    heterogeneous (path strings + AABB vectors + a digest + a nested table)
    rather than scalar dimensions: ``solid`` ->
    ``source``/``brep``/``mesh``/``digest``, and ``imported`` ->
    ``mesh``/``collision`` and their digests, with no ``brep`` and no
    ``source``. Both then take an AABB pair and the ``shape`` fact table. Raises
    ``ValueError`` on the first violation.

    Note: the SDK builder (``object_builder.make_*``) does its own validation
    with rich error refs for the authoring path; this function is the shared
    structural guard for downstream consumers that read IR (kernel, render,
    verify) and for any IR loaded from disk.
    """
    _require_known(primitive)
    if primitive == PRIMITIVE_SOLID:
        return _validate_solid_params(params)
    if primitive == PRIMITIVE_IMPORTED:
        return _validate_imported_params(params)
    out = {}
    for key in PARAM_KEYS[primitive]:
        if key not in params:
            raise ValueError(f"{primitive} primitive missing param {key!r}")
        v = float(params[key])
        if not math.isfinite(v):
            raise ValueError(f"{primitive} param {key!r} must be finite, got {v!r}")
        if v <= 0.0:
            raise ValueError(f"{primitive} param {key!r} must be > 0, got {v!r}")
        out[key] = v
    return out


def local_aabb(primitive, params):
    """Axis-aligned bounding box in the part-local frame.

    Returns ``((xmn, ymn, zmn), (xmx, ymx, zmx))``.

    - box              -> half-extents (width/2, length/2, height/2),
                          origin-centered
    - solid / imported -> the stored full local AABB verbatim. **May be
                          off-center** (geometry authored, or imported, away
                          from its local origin), unlike the analytic
                          primitives. For an imported part this is the VISUAL
                          mesh's box: it is what the renderer draws, what the
                          27-name anchor vocabulary is derived from, and what
                          the SAT broad phase bounds.
    """
    _require_known(primitive)
    if primitive in ASSET_PRIMITIVES:
        amn = params["aabb_min"]
        amx = params["aabb_max"]
        return ((float(amn[0]), float(amn[1]), float(amn[2])),
                (float(amx[0]), float(amx[1]), float(amx[2])))
    hx = float(params["width"]) / 2.0
    hy = float(params["length"]) / 2.0
    hz = float(params["height"]) / 2.0
    return ((-hx, -hy, -hz), (hx, hy, hz))


def inertia(primitive, params, density):
    """Solid-body mass + diagonal inertia tensor about the shape's own COM.

    Returns ``(mass, ixx, iyy, izz)`` for a uniform solid of `density`. The
    tensor is taken about the shape's center of mass: the part-local origin for
    the (origin-centered) box, or the true centroid for a solid — which may be
    off-origin, so a caller placing the body (e.g. URDF) offsets its
    ``<origin>`` by ``inertia_origin``. Box matches the prior
    ``urdf_export._box_inertia`` exactly (byte-identical output preserved).

    - box              : ixx=m(l²+h²)/12, iyy=m(w²+h²)/12, izz=m(w²+l²)/12
    - solid / imported : the mass properties from the part's ``shape`` table,
              not a proxy — ``mass = density * shape.volume`` and the diagonal
              is ``density * shape.inertia_diag`` about the true centroid
              ``shape.center``. One expression for both tags: a solid's table
              comes from the CAD kernel and an imported part's is integrated
              from its triangles at import time, but they are the same
              quantities in the same units about the same point. Products of
              inertia are dropped, matching the ``(mass, ixx, iyy, izz)``
              contract every caller expects; a degenerate (zero-volume) part
              falls back to a clamped box proxy from its AABB so it still has
              non-zero mass.
    """
    _require_known(primitive)
    if primitive in ASSET_PRIMITIVES:
        shape = params["shape"]
        volume = float(shape["volume"])
        if volume > 0.0:
            mass = density * volume
            dxx, dyy, dzz = (float(v) for v in shape["inertia_diag"])
            return (mass, density * dxx, density * dyy, density * dzz)
        # Zero-volume asset (a shell or sheet body): fall back to a clamped
        # box proxy, so the link still has non-zero mass.
        (xmn, ymn, zmn), (xmx, ymx, zmx) = local_aabb(primitive, params)
        w = max(xmx - xmn, _MESH_MIN_EXTENT)
        l = max(ymx - ymn, _MESH_MIN_EXTENT)
        h = max(zmx - zmn, _MESH_MIN_EXTENT)
        mass = density * w * l * h
        return (mass,
                mass / 12.0 * (l * l + h * h),
                mass / 12.0 * (w * w + h * h),
                mass / 12.0 * (w * w + l * l))
    w = float(params["width"])
    l = float(params["length"])
    h = float(params["height"])
    mass = density * w * l * h
    return (mass,
            mass / 12.0 * (l * l + h * h),
            mass / 12.0 * (w * w + h * h),
            mass / 12.0 * (w * w + l * l))


def inertia_origin(primitive, params):
    """Part-local point the inertial body returned by ``inertia`` is about.

    ``inertia`` documents its tensor as being about "the shape's own COM", but
    which point that is differs per tag, and a caller emitting a body (URDF)
    must offset its ``<origin>`` by exactly this vector or the mass ends up in
    the wrong place:

    - box              -> the part-local origin ``(0, 0, 0)``
    - solid / imported -> the true centroid ``shape.center`` (or, for a
               degenerate zero-volume part, the AABB center of the fallback box
               proxy, matching ``inertia``)
    """
    _require_known(primitive)
    if primitive in ASSET_PRIMITIVES:
        shape = params["shape"]
        if float(shape["volume"]) > 0.0:
            c = shape["center"]
            return (float(c[0]), float(c[1]), float(c[2]))
    else:
        return (0.0, 0.0, 0.0)
    (xmn, ymn, zmn), (xmx, ymx, zmx) = local_aabb(primitive, params)
    return (0.5 * (xmn + xmx), 0.5 * (ymn + ymx), 0.5 * (zmn + zmx))


