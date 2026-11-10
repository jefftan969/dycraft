"""Exact-geometry adjudication of SAT-reported penetration pairs.

The OBB/SAT phase in ``sdk/_object_kernel.py`` is deliberately conservative: it
bounds a solid by its AABB (see ``sdk/_primitives.py``). That makes it cheap and
never miss a real contact, but it over-reports — a rod sliding through a tube's
bore, a door swinging through a shelled carcass's open front, two skew round
tubes passing each other: all come back as deep penetrations. Those false
positives are what the object agent then burns repair cycles "fixing", usually
by pushing correct geometry somewhere visibly wrong.

This module re-checks the flagged pairs against real geometry in the CAD
interpreter and drops the ones that are not touching.

Adjudication always runs; it is part of what a collision verdict means, not an
optional refinement.

**The load-bearing invariant: adjudication only ever DOWNGRADES.** Pairs SAT did
not flag are never examined, so no new failure can appear, and anything the
adjudicator cannot answer — no interpreter, a timeout, a crashed boolean —
leaves the conservative SAT verdict standing. A compile can only go from fail to
pass, never the reverse. When that happens ``stats["available"]`` is False, and
the caller is expected to say so rather than present the surviving rows as
confirmed: they were checked only against bounding boxes.

Cost model: only SAT survivors are sent (typically 0–30 per compile), batched
into one subprocess, so a CAD kernel start (seconds) plus 1–100 ms per pair is
affordable. Using B-rep queries as the primary narrow phase — hundreds to tens
of thousands of calls per compile — would not be.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import numpy as np
from pathlib import Path

from .. import _oom
from ..sdk import _object_kernel as _ok
from ..sdk import _primitives as _prim


# Absolute meters. A pair whose exact separation exceeds this is not touching.
DEFAULT_TOL_M = 1.0e-9
# Shared volume (m^3) above which "touching" becomes "interpenetrating". 1e-12
# is a cubic 10 µm — below any real interference, above boolean noise on a
# tangential contact.
BACKEND_BREP = "brep"
BACKEND_MESH = "mesh"
BACKEND_AUTO = "auto"

#: Overrides the resolved backend for a whole process. Set to `brep` to put the
#: scene and animation layers back on the CAD child — the comparison the mesh
#: path was accepted on is reproducible that way, without editing anything.
_BACKEND_ENV_VAR = "DYCRAFT_COLLIDE_BACKEND"

DEFAULT_VOL_EPS = 1.0e-12

# Depth (m) below which an overlap is quantization rather than interference.
# One IR position quantum (1 µm): an authored
# contact cannot be placed more precisely than that, so no exactly-tangent pair
# can be held responsible for less.
TANGENCY_DEPTH_M = 1.0e-6

_DEFAULT_TIMEOUT_S = 120.0
_TIMEOUT_ENV_VAR = "DYCRAFT_ADJUDICATE_TIMEOUT_S"

# Pose quantum for the cache key. Two queries whose relative pose agrees to
# within a micron / a micro-radian are the same question; the CCD sweep asks the
# same one repeatedly as it steps other joints.
_POSE_QUANTUM_M = 1.0e-6

_RUNNER_NAME = "_adjudicate_runner.py"

# A cold CAD interpreter's import budget, kept off the batch's own timeout.
_STARTUP_TIMEOUT_S = 120.0

# Where a scene run stages its catalog. Kept as a literal rather than imported
# from `agent/layers/scene_context.py`: the verify package deliberately
# does not depend on the agent layer (the one exception is the lazy interpreter
# lookup below, which has no alternative home).
_CATALOG_SUBDIR = "catalog"

# Verdicts that mean "SAT was wrong, drop this pair". UNKNOWN deliberately is
# not among them.
_CLEARING_VERDICTS = frozenset({"CLEAR", "CONTACT"})


def _resolve_interpreter():
    """The build123d interpreter, or None.

    Imported lazily from the agent layer: that module owns the interpreter
    discovery chain (``DYCRAFT_B123D_PYTHON`` > ``.b123d_python`` > conventional
    prefixes) and duplicating it here would let the two drift apart, which shows
    up as "authoring works but adjudication silently doesn't".
    """
    try:
        from ..agent.cad._cad_env import b123d_python
    except Exception:  # noqa: BLE001 — verifiers must not fail on an import
        return None
    try:
        return b123d_python()
    except Exception:  # noqa: BLE001
        return None


def _runner_path() -> Path:
    from ..agent.cad import _cad_env
    return Path(_cad_env.__file__).with_name(_RUNNER_NAME)


def _child_env():
    """Environment for the adjudication child, or plain `os.environ`.

    Delegates to the agent layer for the same reason `_resolve_interpreter`
    does: that module owns everything about how the CAD interpreter is
    launched, and a second copy of the loader-path fix here would drift.
    """
    try:
        from ..agent.cad._cad_env import b123d_child_env
        return b123d_child_env()
    except Exception:  # noqa: BLE001 — verifiers must not fail on an import
        return dict(os.environ)


def _resolve_timeout() -> float:
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# Geometry specs
# ---------------------------------------------------------------------------

def geometry_spec(part: dict, run_dir, *, backend: str = BACKEND_BREP):
    """How to rebuild `part`'s exact geometry for `backend`, or None.

    B-rep backend: ``solid`` -> its staged ``.brep``. ``box`` -> rebuilt
    analytically from params, which is exact (not the SAT proxy) and is what
    lets a solid-vs-box pair be adjudicated at all. Anything else -> None, and
    such pairs keep their SAT verdict.

    Mesh backend: an asset part -> the triangulation named by
    ``_prim.collide_asset_key``. For a ``solid`` that is the OBJ it already ships
    beside the B-rep (``params["mesh"]``), so the mesh backend needs no new asset
    and no new content hash. For an ``imported`` part it is ``params["collision"]``,
    NOT the visual mesh: a third-party asset ships the two separately and they
    differ by more than detail — the Panda's visual meshes total 102,752
    triangles against 2,680 for its collision meshes, and several of the visual
    ones are open shells that no containment probe can answer about. ``box``
    stays analytic — the mesh path triangulates it locally, which is exact for a
    box and keeps the two backends answering about the same body.

    B-rep backend + ``imported`` -> None, and such pairs keep their SAT verdict.
    That is deliberate and load-bearing rather than an omission: an imported
    triangle soup has no exact geometry, and the object layer asks for ``brep``
    explicitly (see `resolve_backend`), so the one layer that must not
    adjudicate an imported asset cannot.

    Returning ``None`` is the honest answer, not a safe one: the caller keeps
    an unjudged SAT candidate, which reads as a failure. That is why
    ``verify/_per_object.py`` exempts imported definitions from
    ``NoSelfPenetration`` outright rather than relying on this to be benign —
    a scene run can place an imported asset, so this path is reachable
    from a layer whose object suite would otherwise fail on every one of them.
    """
    primitive = part.get("primitive")
    params = part.get("params") or {}
    if backend == BACKEND_MESH and primitive in _prim.ASSET_PRIMITIVES:
        rel = params.get(_prim.collide_asset_key(primitive))
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            if run_dir is None:
                return None
            path = Path(run_dir) / rel
        if not path.is_file():
            return None
        return {"mesh": str(path.resolve())}
    if primitive == _prim.PRIMITIVE_SOLID:
        rel = params.get("brep")
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            if run_dir is None:
                return None
            path = Path(run_dir) / rel
        if not path.is_file():
            return None
        # Absolutize: this string crosses a process boundary, and `_invoke`
        # runs the child with `cwd` set to a scratch dir. A relative path that
        # `is_file()` accepts here is unreadable there, and the only symptom is
        # every pair coming back UNKNOWN — i.e. exact adjudication silently
        # switching itself off and leaving SAT proxy verdicts standing.
        return {"brep": str(path.resolve())}
    if primitive == _prim.PRIMITIVE_BOX:
        keys = _prim.PARAM_KEYS[primitive]
        try:
            return {"primitive": primitive,
                    "params": {k: float(params[k]) for k in keys}}
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _geometry_id(spec) -> str:
    if "mesh" in spec:
        # Same argument as the brep branch: the filename is the content digest.
        return "mesh:" + Path(spec["mesh"]).stem
    if "brep" in spec:
        # The filename is the content digest, so its stem identifies the
        # geometry independently of which run staged it.
        return "brep:" + Path(spec["brep"]).stem
    params = spec.get("params") or {}
    return spec.get("primitive", "?") + ":" + ",".join(
        f"{k}={params[k]!r}" for k in sorted(params))


def _quantize(value: float) -> int:
    return int(round(float(value) / _POSE_QUANTUM_M))


def _relative_pose_key(se3_a, se3_b):
    """Key for the pose of B *relative to* A.

    Relative rather than absolute so the same joint configuration examined from
    a differently-placed root still hits the cache.
    """
    rel = _ok._rot.se3_compose(_ok._rot.se3_inverse(se3_a), se3_b)
    (tx, ty, tz), (qw, qx, qy, qz) = rel
    return (_quantize(tx), _quantize(ty), _quantize(tz),
            _quantize(qw), _quantize(qx), _quantize(qy), _quantize(qz))


def _largest_face_area(part) -> float:
    """Area of the biggest face of a part's local AABB, or 0 if unknown."""
    params = (part or {}).get("params") or {}
    lo, hi = params.get("aabb_min"), params.get("aabb_max")
    if not lo or not hi or len(lo) != 3 or len(hi) != 3:
        return 0.0
    ext = sorted(abs(float(hi[k]) - float(lo[k])) for k in range(3))
    return ext[1] * ext[2]          # the two largest extents


def tangency_vol_eps(part_a, part_b, floor: float = DEFAULT_VOL_EPS) -> float:
    """Shared volume below which an overlap is quantization, not interference.

    One constant cannot serve both a 2 mm pin and a 50 mm face. The same 1 µm
    of numerical wobble shares 4e-12 m^3 across the pin and 2.5e-9 m^3 across
    the face, so a fixed 1e-12 clears the pin and calls the face a
    penetration — which is exactly the knife edge an agent authoring a flush
    grasp lands on, and why every animation run so far invented a 0.1–0.8 mm
    clearance it could not justify instead of writing the contact it meant.

    The patch two parts can share is bounded by the smaller one's largest face,
    so that area times `TANGENCY_DEPTH_M` is the most an exactly tangent pair
    can accidentally share. Never returns less than `floor`.
    """
    areas = [_largest_face_area(part_a), _largest_face_area(part_b)]
    areas = [a for a in areas if a > 0.0]
    if not areas:
        return floor
    return max(floor, TANGENCY_DEPTH_M * min(areas))


def _part_spec_lookup(ir, catalog, catalog_dir, *, backend=BACKEND_BREP):
    """`(instance, part_name) -> (part_dict | None, spec | None)`, memoized.

    Replaces two identical `_part_of` closures that each rebuilt the whole
    instance index per lookup and then linear-scanned `obj_ir["parts"]`, with
    `geometry_spec` re-`resolve()`ing the same brep path on top. That was two
    calls per pair, which mattered little when a call handled one pair —
    batching now sends thousands through a single call, so it became a main
    cost of the parent side.

    Three things are hoisted or cached: the instance index (built once), a
    `{part_name: part}` map per object_ref, and the spec itself — which for a
    `solid` does `is_file()` + `resolve()`, i.e. two syscalls on a path that
    repeats across every pair touching that part.

    Keyed on `object_ref` rather than `id(obj_ir)`: the ref is the object's
    content hash, so two catalog entries that are equal cannot disagree.
    """
    from ..sdk import _scene_kernel as _sk

    instances = _sk._index_instances(ir)
    by_ref: dict = {}
    memo: dict = {}

    def lookup(instance_name, part_name):
        key = (instance_name, part_name)
        hit = memo.get(key)
        if hit is not None:
            return hit
        inst = instances.get(instance_name)
        ref = inst.get("object_ref") if inst else None
        parts = by_ref.get(ref)
        if parts is None:
            obj_ir = catalog.get(ref) if ref is not None else None
            parts = {p["name"]: p for p in (obj_ir or {}).get("parts", ())}
            by_ref[ref] = parts
        part = parts.get(part_name)
        out = (part, geometry_spec(part, catalog_dir, backend=backend)
               if part else None)
        memo[key] = out
        return out

    return lookup


def cache_key(spec_a, se3_a, spec_b, se3_b):
    return (_geometry_id(spec_a), _geometry_id(spec_b),
            _relative_pose_key(se3_a, se3_b))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_backend(requested: str = BACKEND_AUTO) -> str:
    """Which narrow phase answers, given a request and what is installed.

    `auto` means mesh: it agrees with the CAD child pair for pair on a corpus of
    recorded objects and runs its cold sweep faster than the child's warm cache
    (see `_narrow_phase`). There is **no fallback** — an absent `coal` is an
    install problem and reads as one, because the alternative is handing back
    the conservative SAT proxy as if it had been adjudicated.

    The **object layer asks for `brep` explicitly** and gets it: that layer
    authors the analytic geometry and its repair advice is written against
    `exact_volume_cm3`, which is a boolean intersection no mesh query produces.

    An explicit request wins over the environment. The variable is a debugging
    override for callers that did not choose — reading it *after* the argument
    let `DYCRAFT_COLLIDE_BACKEND=fcl` push the object layer onto a backend that
    cannot answer the question its prompt is written around.
    """
    requested = (requested or BACKEND_AUTO).strip().lower()
    if requested == BACKEND_AUTO:
        env = (os.environ.get(_BACKEND_ENV_VAR) or "").strip().lower()
        requested = env or BACKEND_MESH
    if requested == BACKEND_BREP:
        return BACKEND_BREP
    if requested != BACKEND_MESH:
        # Not a silent default: a misspelled backend would otherwise get mesh
        # answers with no sign that its request had been dropped.
        raise ValueError(
            f"unknown collision backend {requested!r}; expected one of "
            f"{BACKEND_AUTO!r}, {BACKEND_BREP!r}, {BACKEND_MESH!r}")
    from . import _narrow_phase as _fc
    if not _fc.is_available():
        raise RuntimeError(
            "coal is required for scene/animation collision checking; "
            f"install it, or set {_BACKEND_ENV_VAR}=brep to use the CAD child")
    return BACKEND_MESH


#: The mesh's worst measured disagreement with the exact surfaces, as a
#: fraction of the larger part's bounding diagonal.
#:
#: The worst disagreement found between the mesh and the exact surfaces is
#: about 3e-5 of the larger part's diagonal (0.16 mm on a 5 m car body), so a
#: clear at more than this is a clear.
_MESH_AGREEMENT_REL = 3.0e-5

#: How much margin to demand over that, before trusting a mesh clear: ten
#: times that worst disagreement.
_MESH_CLEAR_MARGIN = 10.0


def _mesh_specs(queries, parts, run_dir):
    """The same queries addressed to the staged meshes, or None.

    None if any part has no staged mesh, because a prefilter that silently
    skipped some pairs would report a clear it never tested.
    """
    out = []
    for q in queries:
        tag = q["tag"]
        try:
            _, names = tag.split(":", 1)
            a_name, b_name = names.split("|", 1)
        except ValueError:
            return None
        spec_a = geometry_spec(parts.get(a_name, {}), run_dir,
                              backend=BACKEND_MESH)
        spec_b = geometry_spec(parts.get(b_name, {}), run_dir,
                              backend=BACKEND_MESH)
        if spec_a is None or spec_b is None:
            return None
        out.append({"tag": tag, "a": spec_a, "a_se3": q["a_se3"],
                    "b": spec_b, "b_se3": q["b_se3"]})
    return out


def _brep_with_mesh_prefilter(queries, parts, run_dir, stats, *, tol: float,
                              vol_eps: float):
    """Clear what the meshes can clear; spend the boolean on the rest.

    The B-rep backend exists because the object layer's repair advice is
    written against `exact_volume_cm3`, which only a boolean intersection
    produces. But **you need a volume only for pairs that actually overlap**,
    and clearing a pair needs a distance.

    The costs are not comparable: a boolean on a pair of large curved parts
    takes seconds to a minute, the mesh BVH query tens of milliseconds, with
    the same verdict. A batch of booleans that runs out its timeout is
    discarded whole and leaves the conservative SAT verdict standing, which
    reports bounding-box artifacts as penetrations.

    What makes the mesh safe to clear on: it runs its own containment check
    (`_narrow_phase._containment`), so it is not fooled by a part swallowed
    whole — the failure mode that makes a bare distance query wrong and that
    `_answer` documents. And its disagreement with the exact surfaces is
    small (`_MESH_AGREEMENT_REL`); a clear is only accepted with ten times that
    margin.

    What it costs: an overlap thinner than the mesh's own error could be
    cleared here. `mesh_cleared` records how many verdicts came this way so
    the claim in the report can stay honest.
    """
    # Opt-in, and only the verifier opts in. The object-layer PROBES share
    # this adjudicator, and `gap()` / `pair_report()` promise the agent "the
    # exact surface distance, the same B-reps `NoSelfPenetration` judges with"
    # in `object.md`. A probe answered from a mesh would quietly make that
    # sentence false, so `no_self_penetration` passes `mesh_prefilter=True`
    # and nothing else does.
    mesh_queries = _mesh_specs(queries, parts, run_dir)
    if mesh_queries is None:
        return _invoke(queries, tol=tol, vol_eps=vol_eps)
    mesh_answers = _answer_with_mesh(mesh_queries)
    if not mesh_answers:
        return _invoke(queries, tol=tol, vol_eps=vol_eps)

    cleared, escalate = {}, []
    for q in queries:
        rec = mesh_answers.get(q["tag"])
        if not isinstance(rec, dict) or rec.get("verdict") == "UNKNOWN":
            escalate.append(q)
            continue
        if rec.get("verdict") == "PENETRATING":
            # Needs a volume, which only the boolean gives.
            escalate.append(q)
            continue
        distance = rec.get("distance")
        margin = _clear_margin(parts, q["tag"])
        if distance is None or float(distance) < margin:
            # Touching, or too close to call at mesh resolution.
            escalate.append(q)
            continue
        cleared[q["tag"]] = rec
    stats["mesh_cleared"] = len(cleared)
    if not escalate:
        return cleared
    stats["sent_to_brep"] = len(escalate)
    exact = _invoke(escalate, tol=tol, vol_eps=vol_eps)
    if exact is None:
        # The boolean could not run. Anything the mesh cleared still stands —
        # losing those too is what turned three recorded attempts into 112
        # false-positive pairs.
        return cleared or None
    return {**cleared, **exact}


def _clear_margin(parts, tag: str) -> float:
    """How far apart the meshes must read before a clear is trusted."""
    try:
        _, names = tag.split(":", 1)
        a_name, b_name = names.split("|", 1)
        span = 0.0
        for name in (a_name, b_name):
            params = (parts.get(name) or {}).get("params") or {}
            lo, hi = params.get("aabb_min"), params.get("aabb_max")
            if not lo or not hi:
                return float("inf")     # unmeasurable: never clear on mesh
            span = max(span, sum((float(hi[i]) - float(lo[i])) ** 2
                                 for i in range(3)) ** 0.5)
        return span * _MESH_AGREEMENT_REL * _MESH_CLEAR_MARGIN
    except BaseException:  # noqa: BLE001 — unmeasurable: never clear on mesh
        return float("inf")


def _unavailable_reason() -> str:
    """Why nothing could be adjudicated.

    `_invoke` collapses five distinct causes into one `None` — no interpreter,
    no runner file, a launch failure, a timeout, an unparsable result — and
    the message said "no build123d interpreter, or the adjudicator failed".
    Telling them apart after the fact otherwise takes loading the staged
    B-reps by hand and timing them. The interpreter is the one cause that can
    be distinguished cheaply and after the fact, so it is.
    """
    if not _resolve_interpreter():
        return "no build123d interpreter is configured"
    return (f"the adjudicator returned no usable result: possible launch "
            f"failure, unparsable output, or {_resolve_timeout():.0f} s batch "
            f"timeout. Large spline-bounded parts can make booleans slow; "
            f"inspect worker diagnostics before changing geometry.")


def _answer_with_mesh(queries):
    """`_invoke`'s contract, answered by BVH queries in this process.

    **One answerer for both callers, as a correctness requirement.** `adjudicate_*` and `exact_part_gap` share
    `ctx.adjudication_cache` and key it identically, so a record written by one is read by the other, and every
    answered record carries a verdict and a distance, as the CAD child's do. A record without a `verdict` would count
    as `unknown` and keep a pair measured clear; one without a `distance` would be dropped from the minimum, so a
    `separated()` claim would pass over a pair nobody measured.

    Material sharing is reported as `shares_material`, not as a `common_volume`: `_enrich` would copy a volume out as
    `exact_volume`, and a marker standing in for a measurement would show "1 m³ of shared material" to an agent taught
    to size fixes from that number. A mesh query has no volume, so the scene and animation evidence omit
    `exact_volume_cm3` on this backend, which `_pair_evidence` and `_interval_entry` read as absent.

    The order matters too: a boundary distance does not report containment (a swallowed part has a positive
    nearest-face distance), so the crossing-and-containment test runs first and only a pair it clears gets a
    distance. Tangency is applied inside `penetration` as coal's `security_margin` against `TANGENCY_DEPTH_M`, so a
    flush contact arrives here already CLEAR and never needs `_is_tangency`, which with no volume could not rescue it.

    **No `tol` parameter.** The callers' `tol` is `DEFAULT_TOL_M` (1 nm), the distance tolerance of the B-rep boolean,
    not a tangency depth; passed through, it would run the filter a thousand times too tight and report a flush
    contact with sub-micron numerical overlap as PENETRATING.
    """
    from . import _narrow_phase as _fc
    from ..sdk import _deform_eval as _de

    models: dict = {}

    def model(spec):
        # `_geometry_id` is a content name — a digest for a staged mesh, the
        # dimensions for a box — which is exactly what `build_bvh(key=)` wants,
        # and why this dict is now only a first-level lookup in front of the
        # module cache. Without the key every batch rebuilt the BVH *and* its
        # eroded twin from scratch; a clip is many batches.
        key = _geometry_id(spec)
        hit = models.get(key)
        if hit is None:
            if "mesh" in spec:
                verts, tris = _de.load_obj(Path(spec["mesh"]))
                hit = _fc.build_bvh(verts, tris, key=key)
            else:
                # Straight to `primitive_bvh`, which owns the box's twelve
                # triangles and its own cache key; going through
                # `_box_triangles` built the body once to unpack it and once
                # more to use it.
                hit = _fc.primitive_bvh(_prim.PRIMITIVE_BOX,
                                        spec.get("params") or {})
            models[key] = hit
        return hit

    out: dict = {}
    for q in queries:
        try:
            ba, bb = model(q["a"]), model(q["b"])
            hit = _fc.penetration(ba, q["a_se3"], bb, q["b_se3"])
            if hit is not None:
                # The narrow phase's own vocabulary, carried through rather
                # than renamed: `depth_m` is an upper bound and `depth_is_bound`
                # is what says so, and a reader grepping for either should find
                # this record too.
                #
                # **`distance` is signed here, and that is the point.** It used
                # to be a flat 0.0, so the only magnitude an overlapping pair
                # could offer a report was SAT's box depth — which
                # a verifier duly printed as "overlap by 230.0 mm" for a
                # 100 mm ball. The narrow phase now measures the depth on the
                # triangles (`_narrow_phase._separating_shift`), so the pair can
                # carry one continuous quantity through zero: `+gap` when they
                # are apart, `-depth` when they are not. `_enrich` copies it into
                # `exact_distance` either way, and the sign is what tells a
                # reader which side of contact it is on.
                #
                # Nothing on the gap path can read the negative value by
                # mistake: `exact_part_gap` returns 0.0 on `shares_material`
                # before it looks at `distance`, and the room paths never see a
                # mesh record at all.
                out[q["tag"]] = {"verdict": "PENETRATING",
                                 "shares_material": True,
                                 "distance": -float(hit["depth_m"]),
                                 "depth_m": hit["depth_m"],
                                 "depth_is_bound": hit["depth_is_bound"],
                                 "depth_measured": hit["measured"],
                                 "method": hit["method"]}
            else:
                gap = _fc.surface_gap(ba, q["a_se3"], bb, q["b_se3"])
                out[q["tag"]] = {"verdict": "CLEAR",
                                 "shares_material": False,
                                 "distance": gap["gap_m"],
                                 "method": gap["method"]}
        except Exception as exc:          # a bad mesh is not a clean pair
            # No distance either: a record with neither is what makes the gap
            # path count `not_adjudicable` instead of quietly omitting the pair.
            out[q["tag"]] = {"verdict": "UNKNOWN", "reason": str(exc)[:200]}
    return out


def _box_triangles(spec):
    """A `box` spec as `(verts, tris)`, from the one place that builds them."""
    from . import _narrow_phase as _fc
    body = _fc.primitive_bvh(_prim.PRIMITIVE_BOX, spec.get("params") or {})
    return body.verts, body.tris


def _invoke(queries, *, tol: float, vol_eps: float):
    """Run one batch through the CAD subprocess; return tag -> record, or None.

    ``None`` means "could not adjudicate at all" (no interpreter, launch
    failure, timeout, unparsable result) and the caller must leave every SAT
    verdict untouched.
    """
    interp = _resolve_interpreter()
    if not interp:
        return None
    runner = _runner_path()
    if not runner.is_file():
        return None
    payload = _invoke_pooled(interp, runner, queries, tol=tol, vol_eps=vol_eps)
    if payload is None:
        payload = _invoke_one_shot(interp, runner, queries, tol=tol,
                                   vol_eps=vol_eps)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    out = {}
    for rec in payload.get("results", []):
        if isinstance(rec, dict) and rec.get("tag") is not None:
            out[rec["tag"]] = rec
    return out


def _job_payload(queries, *, tol: float, vol_eps: float) -> dict:
    return {"queries": queries, "tol": tol, "vol_eps": vol_eps}


def _invoke_pooled(interp, runner, queries, *, tol: float, vol_eps: float):
    """One batch on a warm adjudicator, or None to fall back to a fresh one.

    Adjudication runs on every object compile that SAT flags anything on, and
    a fresh child would pay its own CAD kernel start each time, a second and a
    half on top of the authoring child's. Reuse is safe here for a simpler reason than it is
    for authoring: this child holds no state between batches at all, and the one
    thing that *would* be state — the shape cache — is rebuilt per batch inside
    `_adjudicate` precisely so a rewritten B-rep path cannot be answered from a
    previous compile.

    Any fault retires the child, so a wedged or crashed kernel is never reused.
    """
    from ..agent.cad import _worker_pool
    if not _worker_pool.POOL.enabled():
        return None
    try:
        child = _worker_pool.POOL.acquire(interp, runner)
    except OSError:
        return None
    if child is None:
        return None
    try:
        child.ensure_ready(_STARTUP_TIMEOUT_S)
    except (_worker_pool.WorkerTimeout, _worker_pool.WorkerProtocolError):
        _worker_pool.POOL.recycle(child)
        return None

    work = Path(tempfile.mkdtemp(prefix="dycraft_adj_"))
    try:
        result_path = work / "result.json"
        job = {**_job_payload(queries, tol=tol, vol_eps=vol_eps),
               "result": str(result_path)}
        try:
            ack = child.run_job(job, time.monotonic() + _resolve_timeout())
        except (_worker_pool.WorkerTimeout, _worker_pool.WorkerProtocolError):
            _worker_pool.POOL.recycle(child)
            return None
        if ack.get("status") != "ok":
            _worker_pool.POOL.recycle(child)
            return None
        _worker_pool.POOL.release(child)
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def _invoke_one_shot(interp, runner, queries, *, tol: float, vol_eps: float):
    """One batch on a single-use interpreter — the pre-pool path, unchanged."""
    work = Path(tempfile.mkdtemp(prefix="dycraft_adj_"))
    try:
        job_path = work / "job.json"
        result_path = work / "result.json"
        job_path.write_text(
            json.dumps(_job_payload(queries, tol=tol, vol_eps=vol_eps)),
            encoding="utf-8")
        try:
            subprocess.run(
                _oom.tiered([interp, str(runner), "--job", str(job_path),
                             "--result", str(result_path)], _oom.WORKER),
                cwd=str(work), capture_output=True, text=True,
                timeout=_resolve_timeout(),
                # Same loader fix the authoring child needs: the OCP wheel
                # links the system libexpat, which breaks the conda pyexpat
                # that build123d reaches through IPython.
                env=_child_env(),
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def adjudicate_object_pairs(ir: dict, pairs, *, run_dir=None, cache=None,
                            tol: float = DEFAULT_TOL_M,
                            vol_eps: float = DEFAULT_VOL_EPS,
                            backend: str = BACKEND_AUTO,
                            mesh_prefilter: bool = False):
    """Re-check SAT-reported object-layer `pairs` against exact geometry.

    `pairs` is a list of ``(PenetrationPair, internal_poses_or_None)``, or
    ``(PenetrationPair, poses, mount_deltas)`` from the animation layer: the
    pose matters because a CCD-witnessed pair was flagged at a specific joint
    configuration, and that is the configuration whose geometry has to be
    adjudicated. The mount deltas matter for the same reason one layer down —
    a deforming parent part moves its children's frames, and a pair flagged at
    a deformed mount must be adjudicated there, not at the rest one.

    Returns ``(kept, stats)``. ``kept`` holds the pairs that survived, each
    enriched via ``PenetrationPair._replace`` with ``adjudicated=True`` plus
    ``exact_distance`` / ``exact_volume`` / ``closest_points`` when available.
    ``stats`` reports what happened, for the verifier's evidence.

    Pairs that could not be adjudicated are kept verbatim — see the module
    docstring's downgrade-only invariant.
    """
    # Width-normalized before anything else, as `adjudicate_scene_pairs` does: an
    # object-layer caller has no mounts to speak of and omits the element. It has
    # to happen ahead of the early returns below, which unpack the tuples too.
    pairs = [(t[0], t[1], t[2] if len(t) > 2 else None) for t in pairs]
    stats = {"considered": len(pairs), "sent": 0, "cleared": 0,
             "confirmed": 0, "unknown": 0, "cache_hits": 0,
             "not_adjudicable": 0, "available": True}
    backend = resolve_backend(backend)
    stats["backend"] = backend
    if not pairs:
        return [], stats

    parts = {p["name"]: p for p in ir.get("parts", ())}
    if cache is None:
        cache = {}

    # Plan: resolve geometry + FK pose for each pair, hit the cache, and batch
    # what is left. Frames are computed once per distinct pose, not per pair.
    frames_by_pose: dict = {}

    def _frames(poses, mount_deltas=None):
        # Keyed on the mounts as well as the poses. Keying on poses alone made
        # two frames with identical joint values but different deformation
        # share one FK walk, so a pair flagged at a deformed mount was
        # adjudicated against whichever frame happened to populate the memo.
        key = (None if not poses else tuple(sorted(poses.items())),
               None if not mount_deltas else
               tuple(sorted((j, tuple(t), tuple(q))
                            for j, (t, q) in mount_deltas.items())))
        if key not in frames_by_pose:
            frames_by_pose[key] = _ok._compute_frames(
                ir, poses=poses, mount_deltas=mount_deltas)
        return frames_by_pose[key]

    plans = []       # (index, pair, key, query_tag_or_None, tangency_eps)
    queries = []
    # Expanding CCD adjudication from one SAT witness per part pair to every
    # SAT-positive pose exposes repeated exact questions: moving an unrelated
    # joint changes the pose dict but not this pair's relative transform.  The
    # persistent cache cannot help until the batch returns, so deduplicate keys
    # within this batch as well.
    pending_keys = set()
    for i, (pair, poses, mount_deltas) in enumerate(pairs):
        spec_a = geometry_spec(parts.get(pair.a, {}), run_dir, backend=backend)
        spec_b = geometry_spec(parts.get(pair.b, {}), run_dir, backend=backend)
        eps = tangency_vol_eps(parts.get(pair.a), parts.get(pair.b), vol_eps)
        if spec_a is None or spec_b is None:
            # Counting it is not enough. The verifier already declines to
            # claim on an unadjudicated pair (`object_verify` checks the
            # per-pair `adjudicated` flag), so the verdict is safe — but the
            # *adjudicator* is then unfixable, because nothing says what it
            # choked on. A 0914 object agent met six unadjudicated pairs, did
            # its own raw-CAD intersection to settle them, and had no way to
            # report the cause upward. `geometry_spec` returns None when a
            # part's B-rep asset does not resolve, so name the part.
            stats["not_adjudicable"] += 1
            blocked = stats.setdefault("not_adjudicable_parts", [])
            for name, spec in ((pair.a, spec_a), (pair.b, spec_b)):
                if spec is None and name not in blocked and len(blocked) < 8:
                    blocked.append(name)
            plans.append((i, pair, None, None, eps))
            continue
        frames = _frames(poses, mount_deltas)
        se3_a, se3_b = frames[pair.a], frames[pair.b]
        key = cache_key(spec_a, se3_a, spec_b, se3_b)
        if key in cache:
            stats["cache_hits"] += 1
            plans.append((i, pair, key, None, eps))
            continue
        if key in pending_keys:
            plans.append((i, pair, key, None, eps))
            continue
        tag = f"{i}:{pair.a}|{pair.b}"
        queries.append({"tag": tag, "a": spec_a, "a_se3": se3_a,
                        "b": spec_b, "b_se3": se3_b})
        plans.append((i, pair, key, tag, eps))
        pending_keys.add(key)

    if queries:
        stats["sent"] = len(queries)
        if backend == BACKEND_MESH:
            answers = _answer_with_mesh(queries)
        elif mesh_prefilter:
            answers = _brep_with_mesh_prefilter(
                queries, parts, run_dir, stats, tol=tol, vol_eps=vol_eps)
        else:
            answers = _invoke(queries, tol=tol, vol_eps=vol_eps)
        if answers is None:
            stats["available"] = False
            stats["reason"] = _unavailable_reason()
            return [p for p, _, _ in pairs], stats
        for q in queries:
            rec = answers.get(q["tag"])
            if rec is not None:
                for _i, _pair, key, tag, _eps in plans:
                    if tag == q["tag"] and key is not None:
                        cache[key] = rec
                        break

    kept = []
    for _i, pair, key, _tag, eps in plans:
        rec = cache.get(key) if key is not None else None
        if rec is None:
            kept.append(pair)          # not adjudicable / no answer -> keep
            continue
        verdict = rec.get("verdict")
        if verdict in _CLEARING_VERDICTS:
            stats["cleared"] += 1
            continue
        if verdict != "PENETRATING":
            stats["unknown"] += 1
            kept.append(pair)
            continue
        if _is_tangency(rec, eps):
            stats["cleared"] += 1
            stats["tangency_cleared"] = stats.get("tangency_cleared", 0) + 1
            continue
        stats["confirmed"] += 1
        kept.append(_enrich(pair, rec))
    return kept, stats


def exact_part_gap(ir: dict, catalog, part_pairs, *, run_dir=None,
                   asset_base_dir=None, cache=None,
                   tol: float = DEFAULT_TOL_M,
                   backend: str = BACKEND_AUTO):
    """Smallest exact boundary distance over `part_pairs`, in meters.

    `part_pairs` is `[(inst_a, part_a, inst_b, part_b, poses, base_deltas), ...]`, optionally with a 7th element
    carrying that frame's `mount_deltas` (a deforming parent moves a child part's frame, and parts are placed by SE3, so
    without it the gap is measured at the rest mount) and an 8th carrying its `deform_bounds`. Returns `(gap, stats)`;
    `gap` is `None` when nothing could be adjudicated.

    This answers what `adjudicate_*_pairs` cannot: they take pairs SAT already flagged and only downgrade them, so they
    say whether a pair overlaps, never how far apart it is. `touch`/`separated` need the distance, and an AABB gap is
    only a lower bound on it, loose by tens of millimeters for a tilted gripper, the geometry a grasp claim is about.
    Pairs that share material report `0.0`, checked first, because `distance_to_with_closest_points` measures between
    boundaries and a part swallowed whole reports a positive distance.

    **A pair naming a part that is deforming on that frame is not adjudicable here**, and is counted rather than
    answered: the geometry is B-reps, and LBS moves vertices without producing a new solid. The caller falls back to
    the bounding-box bound with `exact=False`, which is sound because `instance_part_world_aabbs` inflates it by the
    same `deform_bounds`.

    **The caller must check `stats["not_adjudicable"]`.** A pair with no exact geometry (an `imported` part on the
    B-rep backend, an asset whose collision mesh is not staged) is skipped, so `gap` is the minimum over the pairs that
    could be answered and may overestimate the true gap: the unsafe direction for both predicates, failing a real
    `touch` and passing a `separated` that is not real. `stats["not_adjudicable_parts"]` names up to eight of the parts.
    """
    stats = {"considered": len(part_pairs), "sent": 0, "cache_hits": 0,
             "not_adjudicable": 0, "available": True}
    backend = resolve_backend(backend)
    stats["backend"] = backend
    if not part_pairs:
        return None, stats
    if cache is None:
        cache = {}
    catalog_dir = _catalog_dir(run_dir, asset_base_dir)

    from ..sdk import _scene_kernel as _sk

    part_spec = _part_spec_lookup(ir, catalog, catalog_dir, backend=backend)

    plans, queries, pending = [], [], set()
    tag_of: dict = {}          # cache key -> this call's query tag
    part_pairs = [tuple(t) + (None,) * (8 - len(t)) for t in part_pairs]
    for i, (ia, pa, ib, pb, poses, deltas, mounts, bounds) in enumerate(
            part_pairs):
        if (pa in ((bounds or {}).get(ia) or {})
                or pb in ((bounds or {}).get(ib) or {})):
            stats["not_adjudicable"] += 1
            continue
        _pa, spec_a = part_spec(ia, pa)
        _pb, spec_b = part_spec(ib, pb)
        if spec_a is None or spec_b is None:
            stats["not_adjudicable"] += 1
            continue
        memo: dict = {}
        try:
            se3_a = _sk.instance_part_world_se3(ir, catalog, ia, pa, poses=poses,
                                                memo=memo, base_deltas=deltas,
                                                mount_deltas=mounts)
            se3_b = _sk.instance_part_world_se3(ir, catalog, ib, pb, poses=poses,
                                                memo=memo, base_deltas=deltas,
                                                mount_deltas=mounts)
        except KeyError:
            stats["not_adjudicable"] += 1
            continue
        key = cache_key(spec_a, se3_a, spec_b, se3_b)
        if key in cache or key in pending:
            stats["cache_hits"] += key in cache
            plans.append(key)
            continue
        tag = f"g{i}:{ia}.{pa}|{ib}.{pb}"
        queries.append({"tag": tag, "a": spec_a, "a_se3": se3_a,
                        "b": spec_b, "b_se3": se3_b})
        plans.append(key)
        pending.add(key)
        tag_of[key] = tag

    if queries:
        stats["sent"] = len(queries)
        answers = (_answer_with_mesh(queries) if backend == BACKEND_MESH
                   else _invoke(queries, tol=tol, vol_eps=DEFAULT_VOL_EPS))
        if answers is None:
            stats["available"] = False
            stats["reason"] = _unavailable_reason()
            return None, stats
        for key, tag in tag_of.items():
            if tag in answers:
                cache[key] = answers[tag]

    best = None
    for key in plans:
        rec = cache.get(key)
        # A record that carries neither a volume nor a distance answered
        # nothing — an `UNKNOWN` from a failed boolean, or a query that never
        # came back. It must raise `not_adjudicable`, because the caller
        # (`_instance_gap`) reads that counter to decide whether to label the
        # gap `exact`. Skipping it silently reports the minimum over the pairs
        # that *did* answer as if it were the minimum over all of them: with one
        # pair CLEAR at 0.4 m and one UNKNOWN, the gap came back
        # `(0.4, exact=True)`, so a `separated(tol=0.005)` claim passed over a
        # pair that might have been touching.
        if rec is None:
            stats["not_adjudicable"] += 1
            continue
        # `shares_material` first: it is what the mesh backend reports, and it
        # is deliberately not a `common_volume`, because `_enrich` would copy a
        # marker volume out as an `exact_volume` measurement.
        vol = rec.get("common_volume")
        if rec.get("shares_material") or (vol is not None and float(vol) > 0.0):
            return 0.0, stats
        d = rec.get("distance")
        if d is None:
            if vol is None:
                stats["not_adjudicable"] += 1
            continue
        best = float(d) if best is None else min(best, float(d))
    return best, stats


def exact_object_part_gaps(ir: dict, part_pairs, *, poses=None,
                           mount_deltas=None, run_dir=None, cache=None,
                           tol: float = DEFAULT_TOL_M,
                           backend: str = BACKEND_BREP):
    """Exact surface distance for each `(part_a, part_b)` of **one ObjectIR**.

    The object-layer counterpart of `exact_part_gap`: one object at one joint configuration, so a pair is two bare
    part names and `poses` is shared. `adjudicate_object_pairs` only takes pairs SAT already flagged and can only
    downgrade them, so it says whether a pair overlaps, never how far apart it is.

    Returns `(gaps, stats)` with one entry per input pair, in order:

      * a float: meters between the two parts' boundaries;
      * `0.0`: they share material, checked before the distance because `distance_to_with_closest_points` measures
        between boundaries, so a part swallowed whole reports a positive distance;
      * `None`: nothing could answer (no interpreter, a part `geometry_spec` cannot describe for this backend, a
        query that came back UNKNOWN). `None` is not "far away"; reading it as one turns an infrastructure fault into
        a geometric claim.

    **`backend` defaults to B-rep, unlike every other entry point here.** Everything the object layer builds has
    exact B-rep (`add_part` makes a `solid` with a `.brep`, `make_box` a `box` rebuilt analytically), and its repair
    advice is written in B-rep's terms. An *imported* definition placed in a scene has no `.brep`, so its pairs come
    back `not_adjudicable` and the caller keeps its box verdict; `build_object_adjacency` takes the default and sees
    this, and `object_support_report` says so in `exact_contact`. The scene path does not come through here:
    `_exact_scene_contacts` calls `exact_part_gap`, whose `BACKEND_AUTO` resolves to the mesh backend and measures an
    imported part against its shipped collision mesh. Pass `backend=BACKEND_MESH` for that here.
    """
    stats = {"considered": len(part_pairs), "sent": 0, "cache_hits": 0,
             "not_adjudicable": 0, "available": True}
    backend = resolve_backend(backend)
    stats["backend"] = backend
    if not part_pairs:
        return [], stats
    if cache is None:
        cache = {}
    parts = {p["name"]: p for p in ir.get("parts", ())}
    # One FK walk for the whole batch: every pair shares this configuration,
    # which is the difference from the scene form and the reason this is cheap
    # enough to call over a whole adjacency graph.
    frames = _ok._compute_frames(ir, poses=poses, mount_deltas=mount_deltas)

    plans: list = []            # cache key per input pair, or None
    queries, pending, tag_of = [], set(), {}
    for i, (a, b) in enumerate(part_pairs):
        spec_a = geometry_spec(parts.get(a) or {}, run_dir, backend=backend)
        spec_b = geometry_spec(parts.get(b) or {}, run_dir, backend=backend)
        if spec_a is None or spec_b is None or a not in frames or b not in frames:
            stats["not_adjudicable"] += 1
            plans.append(None)
            continue
        key = cache_key(spec_a, frames[a], spec_b, frames[b])
        plans.append(key)
        if key in cache:
            stats["cache_hits"] += 1
            continue
        if key in pending:
            continue            # the same relative pose twice in one batch
        tag = f"og{i}:{a}|{b}"
        queries.append({"tag": tag, "a": spec_a, "a_se3": frames[a],
                        "b": spec_b, "b_se3": frames[b]})
        pending.add(key)
        tag_of[key] = tag

    if queries:
        stats["sent"] = len(queries)
        answers = (_answer_with_mesh(queries) if backend == BACKEND_MESH
                   else _invoke(queries, tol=tol, vol_eps=DEFAULT_VOL_EPS))
        if answers is None:
            stats["available"] = False
            stats["reason"] = _unavailable_reason()
            return [None] * len(part_pairs), stats
        for key, tag in tag_of.items():
            if tag in answers:
                cache[key] = answers[tag]

    gaps: list = []
    for key in plans:
        rec = cache.get(key) if key is not None else None
        if rec is None:
            # Already counted above when the spec was missing; count it here
            # only for a pair that was sent and came back with nothing.
            if key is not None:
                stats["not_adjudicable"] += 1
            gaps.append(None)
            continue
        vol = rec.get("common_volume")
        if rec.get("shares_material") or (vol is not None and float(vol) > 0.0):
            gaps.append(0.0)
            continue
        d = rec.get("distance")
        if d is None:
            stats["not_adjudicable"] += 1
            gaps.append(None)
            continue
        gaps.append(float(d))
    return gaps, stats


# ---------------------------------------------------------------------------
# Room containment
# ---------------------------------------------------------------------------


def _room_exterior_box(room_lo, room_hi, k: int, side: str,
                       reach: float = 0.0):
    """Spec + world SE3 for a solid slab filling the space outside one room face.

    Oversized on the two in-face axes so a part escaping through a corner or an
    edge still lands inside the slab.

    **`reach` sizes the thickness and is not optional in practice.** A fixed
    slab is a trap: a part that clears it entirely shares no volume, comes back
    `CLEAR`, and the escape is *cleared* — an object 5.55 m under the floor of a
    3 m room read as contained, and at higher fps the deepest frames dropped out
    of the middle of an interval so the evidence claimed the object came back.
    It also silently clamps the measured depth to the slab. Pass the AABB
    excursion that flagged the frame: it bounds the true one, so a slab sized
    from it always contains the escape.
    """
    ext = [room_hi[i] - room_lo[i] for i in range(3)]
    thick = max(2.0, ext[k], 2.0 * float(reach) + 1.0)
    pad = max(1.0, float(reach))
    dims, ctr = [], []
    for i in range(3):
        if i == k:
            dims.append(thick)
            ctr.append(room_lo[k] - thick / 2.0 if side == "below"
                       else room_hi[k] + thick / 2.0)
        else:
            dims.append(ext[i] + 2.0 * pad)
            ctr.append((room_lo[i] + room_hi[i]) / 2.0)
    spec = {"primitive": _prim.PRIMITIVE_BOX,
            "params": {"width": dims[0], "length": dims[1], "height": dims[2]}}
    return spec, ((ctr[0], ctr[1], ctr[2]), (1.0, 0.0, 0.0, 0.0))


def _catalog_dir(run_dir, asset_base_dir):
    """The asset root exact geometry resolves against.

    `asset_base_dir` when the caller supplies one, else `<run_dir>/catalog` as
    the compatibility fallback for older callers. Extracted because the same
    ternary was written out at five call sites, and it is the input a
    downgrade-only path fails silently on: get it wrong and every pair comes
    back unadjudicated, with a counter in `stats` as the only symptom.
    """
    if asset_base_dir is not None:
        return Path(asset_base_dir)
    return (Path(run_dir) / _CATALOG_SUBDIR) if run_dir is not None else None


#: Room faces in both vocabularies the codebase already uses: the `(axis, side)`
#: pair `NoPenetration`'s `left_the_room` evidence reports, and the name the
#: viewer-relative axis convention implies (`+X` = left, `+Y` = front, `+Z` = up).
#: Named here rather than at the probe so the verifier and the probe cannot drift
#: into disagreeing about which wall is which.
ROOM_FACES = {
    "floor":   (2, "below"),
    "ceiling": (2, "above"),
    "back":    (1, "below"),
    "front":   (1, "above"),
    "right":   (0, "below"),
    "left":    (0, "above"),
}
_AXIS_NAME = ("x", "y", "z")


def _face_queries(ir, catalog, instance, poses, base_deltas, part_boxes,
                  bound, *, axis, side, catalog_dir, tol, cache, tag_prefix,
                  mount_deltas=None, deform_bounds=None):
    """Plan the room-face gap queries for one pose.

    `axis`/`side` select the face the way `_room_exterior_box` does. The floor is
    the common case and has its own wrapper below; the generalization exists
    because the floor was the *only* face any measurement could reach, so an
    agent asking "does this clear the wall" got a bounding box and acted on it.

    **`part_boxes` must already be deformation-inflated** — the caller computes
    it, and the `clear > tol` prune below is only sound if the box actually
    contains the part. `deform_bounds` is taken here as well, for the other half
    of the same fact: a part being deformed on this frame has no B-rep that
    describes its geometry, so it is counted `unmeasurable` rather than queried.
    Asking about its rest solid would return a confident number about a shape
    that is not there.
    """
    from ..sdk import _scene_kernel as _sk

    room_lo, room_hi = _sk.room_world_aabb(ir)
    inst = _sk._index_instances(ir).get(instance)
    obj_ir = catalog.get(inst["object_ref"]) if inst else None
    if obj_ir is None:
        return [], [], {}, None
    parts = {p["name"]: p for p in obj_ir.get("parts", ())}
    deformed_now = (deform_bounds or {}).get(instance) or {}
    b_spec, b_se3 = _room_exterior_box(room_lo, room_hi, axis, side,
                                       max(1.0, abs(bound)))
    plans, queries, tag_of, unmeasurable = [], [], {}, 0
    for pname, (plo, phi) in part_boxes:
        # How far this part's own box already is from the face. Above `tol` it
        # cannot be the nearest thing to it, so it costs no CAD query.
        clear = ((plo[axis] - room_lo[axis]) if side == "below"
                 else (room_hi[axis] - phi[axis]))
        if clear > tol:
            continue
        if pname in deformed_now:
            # Deformed on this frame: the B-rep is the rest solid. The caller
            # falls back to the (inflated, and therefore sound) box bound with
            # `exact=False`, and `DeformedNoPenetration` measures the real thing
            # against room faces exactly.
            unmeasurable += 1
            continue
        spec_a = geometry_spec(parts.get(pname) or {}, catalog_dir)
        if spec_a is None:
            unmeasurable += 1
            continue
        try:
            se3_a = _sk.instance_part_world_se3(
                ir, catalog, instance, pname, poses=poses, memo={},
                base_deltas=base_deltas, mount_deltas=mount_deltas)
        except KeyError:
            unmeasurable += 1
            continue
        key = cache_key(spec_a, se3_a, b_spec, b_se3)
        # Carrying the part name is what lets a caller say *which* part is the
        # nearest thing to the face. Without it an instance-level answer is
        # min-over-parts and nothing else, which for an appliance standing on the
        # floor is 0.0 at every joint angle — true, useless, and exactly how the
        # door-clearance question stayed unanswerable.
        plans.append((key, pname))
        if key in cache or key in tag_of:
            continue
        tag = f"{tag_prefix}{len(tag_of)}:{instance}.{pname}"
        queries.append({"tag": tag, "a": spec_a, "a_se3": se3_a,
                        "b": b_spec, "b_se3": b_se3})
        tag_of[key] = tag
    return plans, queries, tag_of, unmeasurable


def _face_bound(part_boxes, room_lo, room_hi, axis, side) -> float:
    """The world-AABB lower bound on the gap to one face."""
    if side == "below":
        return min(plo[axis] for _n, (plo, _phi) in part_boxes) - room_lo[axis]
    return room_hi[axis] - max(phi[axis] for _n, (_plo, phi) in part_boxes)


def exact_room_gap(ir: dict, catalog, instance: str, poses=None,
                   base_deltas=None, *, mount_deltas=None,
                   deform_bounds=None, faces="all",
                   run_dir=None, asset_base_dir=None, cache=None,
                   tol: float = 0.005):
    """Exact surface gap between an instance and each room face, in meters.

    `instance` is `"name"` or `"name.part"`; the part form is what makes a
    swinging door answerable, since the instance form reduces over every part and
    an appliance standing on the floor reports 0.0 at every joint angle.

    Returns `{face: {"gap": float|None, "exact": bool, "part": str|None,
    "axis": "z", "side": "below"}}` over `faces`, where `part` is the nearest
    part to that face — a face name from `ROOM_FACES`, an iterable
    of them, or `"all"`. `gap` is 0.0 when material is outside the face, and
    `exact=False` carries the same meaning as everywhere else in this module: no
    boolean answered, so the number is the world-AABB lower bound.

    This exists because the floor was the only face any *measurement* could
    reach. `RoomContainment` and `NoPenetration` decide containment exactly, but
    they decide it — an agent asking "how much room does this door have before it
    touches the floor" had `room_containment_margins`, which is
    `within_report` on bounding boxes, and a rotating panel's box is not the
    panel, so an author planning a swing from boxes gives up travel the
    verifier would have cleared.

    Cost is not six exact queries: a face whose AABB bound already
    exceeds `tol` sends no query, so a normally-placed instance pays for the one
    or two faces it is actually near.
    """
    from ..sdk import _scene_kernel as _sk

    # `"<instance>.<part>"` scopes the question to one part, the same spelling the
    # anchor vocabulary uses. Without it the answer is min-over-parts, and for
    # anything standing on the floor that is 0.0 forever — which is how "does the
    # DOOR clear the floor" stayed unanswerable while the floor gap was already
    # exact.
    instance, _sep, part_scope = instance.partition(".")
    if faces == "all":
        wanted = list(ROOM_FACES)
    elif isinstance(faces, str):
        wanted = [faces]
    else:
        wanted = list(faces)
    unknown = [f for f in wanted if f not in ROOM_FACES]
    if unknown:
        raise KeyError(f"unknown room face(s) {unknown}; "
                       f"known: {sorted(ROOM_FACES)}")

    out: dict = {}
    room_lo, room_hi = _sk.room_world_aabb(ir)
    # Both deformation channels reach the boxes, not only the queries: `bound`
    # below is read off them and is what a non-exact answer returns.
    _box_kw = dict(poses=poses, base_deltas=base_deltas,
                   mount_deltas=mount_deltas, deform_bounds=deform_bounds)
    try:
        part_boxes = _sk.instance_part_world_aabbs(
            ir, catalog, instance, **_box_kw)
    except KeyError:
        part_boxes = None
    if part_boxes and part_scope:
        part_boxes = [pb for pb in part_boxes if pb[0] == part_scope]
        if not part_boxes:
            known = sorted(n for n, _b in _sk.instance_part_world_aabbs(
                ir, catalog, instance, **_box_kw))
            raise KeyError(f"{instance!r} has no part {part_scope!r}; "
                           f"known: {known}")
    if cache is None:
        cache = {}
    for face in wanted:
        axis, side = ROOM_FACES[face]
        entry = {"axis": _AXIS_NAME[axis], "side": side}
        if not part_boxes:
            out[face] = {**entry, "gap": None, "exact": False, "part": None}
            continue
        bound = _face_bound(part_boxes, room_lo, room_hi, axis, side)
        gap, exact, part = _exact_face_gap(
            ir, catalog, instance, poses, base_deltas, part_boxes, bound,
            axis=axis, side=side, face=face, tol=tol, cache=cache,
            catalog_dir=_catalog_dir(run_dir, asset_base_dir),
            mount_deltas=mount_deltas, deform_bounds=deform_bounds)
        out[face] = {**entry, "gap": gap, "exact": exact, "part": part}
    return out


def _spec_verts(spec, loaded: dict | None = None):
    """Part-local vertices for a mesh-backend spec, or None.

    Handles both shapes `geometry_spec(..., backend=fcl)` returns: a staged
    triangulation for a `solid`, and an analytic `box` rebuilt locally. Missing
    the box branch is not a small bug — it makes every box part *unmeasurable*,
    so one box in an instance downgrades the whole room answer to its bounding
    bound even when every other part measured fine.
    """
    if spec is None:
        return None
    if "mesh" in spec:
        key = spec["mesh"]
        if loaded is None:
            loaded = {}
        if key not in loaded:
            from ..sdk import _deform_eval as _de
            try:
                loaded[key] = np.asarray(_de.load_obj(Path(key))[0], float)
            except (OSError, ValueError):
                loaded[key] = None
        return loaded[key]
    if spec.get("primitive") == _prim.PRIMITIVE_BOX:
        return np.asarray(_box_triangles(spec)[0], float)
    return None


def _mesh_room_escapes(ir, catalog, requests, stats, *, catalog_dir,
                       slack: float = 0.0):
    """`adjudicate_room_escapes` from the parts' own vertices.

    The B-rep path answers this by intersecting each part with a huge
    exterior slab and reading the shared solid's extreme coordinate. That is
    sound, and it is a boolean in a subprocess for a question whose exact answer
    is `min()` over an array: a room face is an axis-aligned plane, so the
    material past it is a coordinate extremum over the real triangles. Same
    argument as `_mesh_face_gap` and `_mesh_collide.room_escape`, and the three
    share `face_clearance` so they cannot disagree.

    What this fixes is unchanged and is the whole reason the exact pass exists:
    `_aabb_of_transformed_corners` bounds a part by its *rest* box carried
    through its world pose, so a wheel spinning about its own axis inflates by
    `R*(sqrt(2)-1)` while its silhouette never moves. The bound is not the
    excursion; this measures the excursion.
    """
    from ..sdk import _scene_kernel as _sk
    from . import _mesh_collide as _mc

    room_lo, room_hi = _sk.room_world_aabb(ir)
    instances = _sk._index_instances(ir)
    loaded: dict = {}
    confirmed: dict = {}

    for frame, instance, cands, poses, deltas, mounts, bounds in requests:
        inst = instances.get(instance)
        obj_ir = catalog.get(inst.get("object_ref")) if inst else None
        if obj_ir is None:
            for k, side, _a in cands:                 # bound stands
                confirmed[(frame, instance, k, side)] = None
                stats["not_adjudicable"] += 1
            continue
        parts = {p["name"]: p for p in obj_ir.get("parts", ())}
        deformed_now = (bounds or {}).get(instance) or {}

        worlds, missing = {}, False
        for pname in parts:
            if pname in deformed_now:
                missing = True            # a rig moved it; no rest mesh to trust
                continue
            spec = geometry_spec(parts[pname], catalog_dir, backend=BACKEND_MESH)
            verts = _spec_verts(spec, loaded)
            if verts is None:
                missing = True
                continue
            try:
                se3 = _sk.instance_part_world_se3(
                    ir, catalog, instance, pname, poses=poses, memo={},
                    base_deltas=deltas, mount_deltas=mounts)
            except KeyError:
                missing = True
                continue
            worlds[pname] = _place(verts, se3)

        for k, side, _amount in cands:
            key = (frame, instance, k, side)
            if not worlds:
                confirmed[key] = None
                stats["not_adjudicable"] += 1
                continue
            out = max(-_mc.face_clearance(w, room_lo, room_hi, k, side)
                      for w in worlds.values())
            # `slack` is what the triangulation can actually resolve, and a
            # bare `> 0` cannot stand in for it: a tessellated surface of
            # revolution is a polygon, so its silhouette *wobbles* as it turns
            # while the true surface is invariant: a 90 mm disc at 500
            # triangles swings its extreme coordinate by tens of microns over a
            # quarter turn. The caller passes the containment tolerance its
            # broad phase flagged the candidate with (1 mm), well above that.
            if out > slack:
                confirmed[key] = round(float(out), 6)
                stats["confirmed"] += 1
            elif missing:
                # Some part could not be measured, so "no escape" is only true
                # of the ones that could. Keep the bound rather than clearing.
                confirmed[key] = None
                stats["not_adjudicable"] += 1
            else:
                stats["cleared"] += 1
    return confirmed, stats


def _mesh_face_gap(ir, catalog, instance, poses, base_deltas, part_boxes,
                   bound, *, axis, side, tol, catalog_dir,
                   mount_deltas=None, deform_bounds=None):
    """`(gap, exact, part)` for one room face, from the parts' own vertices.

    A room face is an **axis-aligned plane**, so this needs no narrow phase at
    all — not even a BVH. The exact gap is a coordinate extremum over the
    triangulation, which is the same argument `_mesh_collide.room_escape` makes
    for the other direction (material *outside* a face) and the reason that one
    is the single measurement in the deformation suite with no stated blind spot.

    The B-rep path answers this by building a huge exterior box and asking the
    CAD child for a boolean distance to it — sound, and about fifty times the
    cost of `min()` over an array.

    The residual error is the tessellation's own: it is inscribed in the B-rep,
    so the extremum sits *inside* the true surface and the gap comes back
    marginally large: staged meshes under-fill by a fraction of a percent of
    volume, tens of microns of surface, against a `tol` that defaults to 5 mm.
    """
    from ..sdk import _scene_kernel as _sk
    from . import _mesh_collide as _mc

    room_lo, room_hi = _sk.room_world_aabb(ir)
    inst = _sk._index_instances(ir).get(instance)
    obj_ir = catalog.get(inst.get("object_ref")) if inst else None
    if obj_ir is None:
        return None, False, None
    parts = {p["name"]: p for p in obj_ir.get("parts", ())}
    deformed_now = (deform_bounds or {}).get(instance) or {}

    # One memo for the whole face, not one per part: `_spec_verts` defaults to a
    # fresh dict, so calling it bare re-reads the same OBJ off disk for every
    # part near the face, every frame.
    loaded: dict = {}
    best, best_part, skipped = None, None, 0
    for pname, (plo, phi) in part_boxes:
        clear = (plo[axis] - room_lo[axis] if side == "below"
                 else room_hi[axis] - phi[axis])
        if clear > tol:
            continue                      # its own box already clears the face
        if pname in deformed_now:
            skipped += 1                  # a rig moved it; the box bound stands
            continue
        spec = geometry_spec(parts.get(pname), catalog_dir, backend=BACKEND_MESH)
        verts = _spec_verts(spec, loaded)
        if verts is None:
            skipped += 1
            continue
        try:
            se3 = _sk.instance_part_world_se3(
                ir, catalog, instance, pname, poses=poses, memo={},
                base_deltas=base_deltas, mount_deltas=mount_deltas)
        except KeyError:
            skipped += 1
            continue
        world = _place(verts, se3)
        # Shared with `room_escape`, which asks the same question with the sign
        # flipped; one implementation so the two cannot drift.
        gap = max(0.0, _mc.face_clearance(world, room_lo, room_hi, axis, side))
        if best is None or gap < best:
            best, best_part = gap, pname
    if best is None:
        # Nothing measurable near the face: the box bound is the honest answer,
        # and `exact=False` is how the caller is told so.
        return (bound if skipped else max(0.0, bound)), False, None
    if skipped:
        # Some part near this face could not be measured, so the minimum over the
        # ones that *could* is not the minimum. The unmeasured part may be nearer
        # to the face, or through it. The B-rep sibling refuses for the same
        # reason (`if best is None or stats["not_adjudicable"]`), and dropping
        # that turned a bound into a claim: an instance whose one unmeasurable
        # part is 5 mm below the floor would report the others' 40 mm as exact.
        return min(best, bound), False, best_part
    return round(best, 9), True, best_part


def _place(verts, se3):
    """Part-local vertices into world, from `((t), (qw,qx,qy,qz))`."""
    (tx, ty, tz), (qw, qx, qy, qz) = se3
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return verts @ R.T + np.array([tx, ty, tz])


def _exact_face_gap(ir, catalog, instance, poses, base_deltas, part_boxes,
                    bound, *, axis, side, face, tol, cache, catalog_dir,
                    mount_deltas=None, deform_bounds=None,
                    backend=BACKEND_AUTO):
    """`(gap, exact, part)` for one face — `part` is the nearest one, or None."""
    if resolve_backend(backend) == BACKEND_MESH:
        return _mesh_face_gap(
            ir, catalog, instance, poses, base_deltas, part_boxes, bound,
            axis=axis, side=side, tol=tol, catalog_dir=catalog_dir,
            mount_deltas=mount_deltas, deform_bounds=deform_bounds)
    stats = {"not_adjudicable": 0}
    plans, queries, tag_of, unmeasurable = _face_queries(
        ir, catalog, instance, poses, base_deltas, part_boxes, bound,
        mount_deltas=mount_deltas, deform_bounds=deform_bounds,
        axis=axis, side=side, catalog_dir=catalog_dir, tol=tol, cache=cache,
        tag_prefix=face)
    if unmeasurable is None:
        return bound, False, None
    stats["not_adjudicable"] += unmeasurable
    if queries:
        answers = _invoke(queries, tol=DEFAULT_TOL_M, vol_eps=DEFAULT_VOL_EPS)
        if answers is None:
            return bound, False, None
        for key, tag in tag_of.items():
            if tag in answers:
                cache[key] = answers[tag]
    best, best_part = None, None
    for key, pname in plans:
        rec = cache.get(key)
        if rec is None:
            stats["not_adjudicable"] += 1
            continue
        vol = rec.get("common_volume")
        if vol is not None and float(vol) > 0.0:
            return 0.0, True, pname        # material outside the face
        d = rec.get("distance")
        if d is None:
            if vol is None:
                stats["not_adjudicable"] += 1
            continue
        if best is None or float(d) < best:
            best, best_part = float(d), pname
    if best is None or stats["not_adjudicable"]:
        return bound, False, None
    return best, True, best_part


def adjudicate_room_escapes(ir: dict, catalog, requests, *, run_dir=None,
                            asset_base_dir=None, cache=None,
                            tol: float = DEFAULT_TOL_M,
                            vol_eps: float = DEFAULT_VOL_EPS,
                            backend: str = BACKEND_AUTO,
                            slack: float = 0.0):
    """Re-check AABB-reported room escapes against exact geometry.

    `requests` is `[(frame, instance, [(axis_index, side, amount_m), ...], poses, base_deltas), ...]`, every flagged
    frame at once, because a rolling wheel trips the broad phase on every frame and one child launch per frame is
    unaffordable. Returns `(confirmed, stats)`: `confirmed` maps each `(frame, instance, axis_index, side)` a real
    boolean stands behind to the **measured** excursion in meters, or to `None` when only the AABB bound is
    available (no interpreter, a part `geometry_spec` cannot describe for the backend, an unanswered query).

    The measurement comes from the shared solid: `part ∩ exterior-slab` is exactly the material outside that face,
    so its extreme coordinate along `axis_index` is how far past the face the part reaches. A rotated box inflates;
    the shared solid cannot.

    **Why the AABB test needs this.** `_aabb_of_transformed_corners` carries the corners of a part's rest box through
    its world pose, so a surface of revolution spinning about its axis inflates although its silhouette is invariant:
    a cylinder of radius R gains `R*(sqrt(2)-1)` at 45 deg, 37.3 mm for a 90 mm tyre against a 1 mm containment
    slack, and zero only at multiples of 90 deg. No rolling schedule avoids it, so a correct wheel roll would read as
    "left the room below". No test pins this case.

    The room is not an instance and owns no geometry, but the space beyond a room face is a box, `_load_shape`
    rebuilds a box analytically, and "did this part cross that face" is the shared-volume question `_answer` already
    answers, so no new child protocol is needed. A wheel resting flush on the floor shares no volume and comes back
    `CONTACT`. Like the pair adjudicators this can only *clear* a candidate: no interpreter, an unstaged collision
    mesh, an unanswered query or a failed boolean leaves the AABB verdict standing.
    """
    # `(frame, instance, cands, poses, base_deltas)` or the same plus that
    # frame's `mount_deltas` and `deform_bounds`.
    requests = [tuple(r) + (None,) * (7 - len(r)) for r in requests]
    requests = [r for r in requests if r[2]]
    all_cands = {(f, inst, k, s)
                 for f, inst, cands, _p, _d, _m, _b in requests
                 for k, s, _a in cands}
    # `confirmed`/`cleared`/`unknown`/`tangency_cleared` mirror the pair
    # adjudicators: without them a compile that quietly cleared forty flagged
    # frames is indistinguishable in `compile_reports/` from one where nothing
    # ever left the room.
    stats = {"considered": len(all_cands), "sent": 0, "cache_hits": 0,
             "not_adjudicable": 0, "confirmed": 0, "cleared": 0,
             "unknown": 0, "tangency_cleared": 0, "available": True}
    backend = resolve_backend(backend)
    stats["backend"] = backend
    if backend == BACKEND_MESH:
        return _mesh_room_escapes(
            ir, catalog, requests, stats, slack=slack,
            catalog_dir=_catalog_dir(run_dir, asset_base_dir))
    if not requests:
        return {}, stats
    if cache is None:
        cache = {}
    catalog_dir = _catalog_dir(run_dir, asset_base_dir)

    from ..sdk import _scene_kernel as _sk

    room_lo, room_hi = _sk.room_world_aabb(ir)
    instances = _sk._index_instances(ir)
    plans: dict = {}          # candidate -> [(cache_key, part_dict)]
    unresolved = set()        # candidates no boolean can clear
    queries, tag_of = [], {}

    # One slab per face, sized from the deepest excursion *any* frame reports on
    # it. Sizing per-candidate would give the same face different geometry
    # across frames and defeat the cache; sizing from a constant lets a deep
    # escape pass clean through and be cleared.
    reach: dict = {}
    for _f, _i, cands, _p, _d, _m, _b in requests:
        for k, side, amount in cands:
            key = (k, side)
            reach[key] = max(reach.get(key, 0.0), float(amount))
    faces = {key: _room_exterior_box(room_lo, room_hi, key[0], key[1], r)
             for key, r in reach.items()}

    for frame, instance, cands, poses, deltas, mounts, bounds in requests:
        inst = instances.get(instance)
        obj_ir = catalog.get(inst["object_ref"]) if inst else None
        if obj_ir is None:
            unresolved.update((frame, instance, k, s) for k, s, _a in cands)
            continue  # nothing to measure with; the AABB bound is all there is
        parts = {p["name"]: p for p in obj_ir.get("parts", ())}
        try:
            part_boxes = _sk.instance_part_world_aabbs(
                ir, catalog, instance, poses=poses, base_deltas=deltas,
                mount_deltas=mounts, deform_bounds=bounds)
        except KeyError:
            unresolved.update((frame, instance, k, s) for k, s, _a in cands)
            continue  # nothing to measure with; the AABB bound is all there is
        deformed_now = (bounds or {}).get(instance) or {}
        for k, side, _amount in cands:
            cand = (frame, instance, k, side)
            b_spec, b_se3 = faces[(k, side)]
            plans.setdefault(cand, [])
            # A part whose own box is inside the face cannot be responsible.
            # This also absorbs the union-box slack: `instance_world_aabb`
            # bounds the object's *global* rest box, which is looser than the
            # per-part boxes, so a candidate can arrive with no part behind it.
            for pname, (plo, phi) in part_boxes:
                if pname in deformed_now:
                    # Not this module's business: a deformed part's containment
                    # belongs to `DeformedNoPenetration`, which measures it
                    # exactly, and the caller has already excluded it from the
                    # candidate box. Skipped rather than marked unresolved,
                    # because it did not cause this candidate — marking it would
                    # make a rigid sibling's real escape unclearable for the
                    # wrong reason, and asking CAD about its rest solid would
                    # spend a query on a shape nobody is claiming anything about.
                    continue
                out = ((room_lo[k] - plo[k]) if side == "below"
                       else (phi[k] - room_hi[k]))
                if out <= 0.0:
                    continue
                spec_a = geometry_spec(parts.get(pname) or {}, catalog_dir)
                if spec_a is None:
                    stats["not_adjudicable"] += 1
                    unresolved.add(cand)
                    continue
                try:
                    se3_a = _sk.instance_part_world_se3(
                        ir, catalog, instance, pname, poses=poses, memo={},
                        base_deltas=deltas, mount_deltas=mounts)
                except KeyError:
                    stats["not_adjudicable"] += 1
                    unresolved.add(cand)
                    continue
                key = cache_key(spec_a, se3_a, b_spec, b_se3)
                plans[cand].append((key, parts.get(pname)))
                if key in cache:
                    stats["cache_hits"] += 1
                    continue
                if key in tag_of:
                    continue
                tag = f"room{len(tag_of)}:{instance}.{pname}"
                queries.append({"tag": tag, "a": spec_a, "a_se3": se3_a,
                                "b": b_spec, "b_se3": b_se3})
                tag_of[key] = tag

    if queries:
        stats["sent"] = len(queries)
        # Room escapes stay on the CAD child: the query asks how much material
        # lies outside an axis-aligned face, which is a different shape from a
        # pair query and has no mesh form yet. `_mesh_collide.room_escape` is
        # exact for this case and is the obvious place to take it next.
        answers = _invoke(queries, tol=tol, vol_eps=vol_eps)
        if answers is None:
            stats["available"] = False
            stats["reason"] = ("no build123d interpreter, or the adjudicator "
                               "failed")
            return {c: None for c in all_cands}, stats
        for key, tag in tag_of.items():
            if tag in answers:
                cache[key] = answers[tag]

    confirmed: dict = {}
    for cand, entries in plans.items():
        _frame, _inst, k, side = cand
        for key, part in entries:
            rec = cache.get(key)
            # Only an affirmative clearing verdict may clear. `UNKNOWN` — a
            # failed boolean, an unreadable `.brep`, a per-query exception — and
            # an unanswered query are *not* clearances: treating them as such
            # inverted this module's whole contract, and a box driven 1 m below
            # the floor came back `pass` with empty evidence.
            verdict = None if rec is None else rec.get("verdict")
            if verdict in _CLEARING_VERDICTS:
                continue
            if verdict == "PENETRATING" and _is_tangency(
                    rec, tangency_vol_eps(part, None)):
                # Flush contact quantizes into a sliver of shared volume, and
                # one global constant cannot serve both a caster and a skirt.
                stats["tangency_cleared"] += 1
                continue
            depth = None
            if verdict == "PENETRATING":
                # Depth from the shared solid: it lies wholly outside the face,
                # so its extreme coordinate along `k` is how far the part
                # reaches past it. Deepest part owns the number.
                bb = rec.get("common_bbox")
                if bb:
                    depth = ((room_lo[k] - bb[0][k]) if side == "below"
                             else (bb[1][k] - room_hi[k]))
                stats["confirmed"] += 1
            else:
                stats["unknown"] += 1
            if cand in unresolved:
                # A part here could not be measured at all and may reach
                # further than any that could; another part's number would be
                # an understatement wearing a measurement's label.
                confirmed[cand] = None
                continue
            prev = confirmed.get(cand, "absent")
            if prev == "absent" or depth is None:
                confirmed[cand] = depth
            elif prev is not None:
                confirmed[cand] = max(prev, depth)
    for cand in unresolved:
        confirmed[cand] = None
    stats["cleared"] = len(all_cands) - len(confirmed)
    return confirmed, stats


def _is_tangency(rec, eps: float) -> bool:
    """Is this "penetration" just quantization across a flush contact?

    The worker already decided PENETRATING against its own job-wide `vol_eps`;
    this re-decides against the pair's own scale. Only ever *downgrades*, and
    only when the worker actually reported a volume — a missing one leaves the
    conservative verdict standing.
    """
    vol = rec.get("common_volume")
    return vol is not None and float(vol) <= eps


def adjudicate_scene_pairs(ir: dict, catalog, pairs, *, run_dir=None,
                           asset_base_dir=None,
                           cache=None, tol: float = DEFAULT_TOL_M,
                           vol_eps: float = DEFAULT_VOL_EPS,
                           backend: str = BACKEND_AUTO):
    """Re-check SAT-reported cross-instance `pairs` against exact geometry.

    ``a_sub``/``b_sub`` identify the concrete part pair to place. File-backed
    geometry is resolved against the explicit ``asset_base_dir`` supplied by
    the scene payload. ``run_dir/catalog`` remains as a compatibility fallback
    for older callers.

    Each element of `pairs` is ``(PenetrationPair, poses)`` or, from the
    dynamics layer, ``(PenetrationPair, poses, base_deltas)`` or
    ``(PenetrationPair, poses, base_deltas, mount_deltas)`` — the extra
    elements carry that frame's whole-body actor deltas so the exact
    geometry is placed where the animation actually put it. Mixed-arity
    input is accepted; a 2-tuple means "no deltas".

    Same downgrade-only contract as the object layer.
    """
    stats = {"considered": len(pairs), "sent": 0, "cleared": 0,
             "confirmed": 0, "unknown": 0, "cache_hits": 0,
             "not_adjudicable": 0, "available": True}
    backend = resolve_backend(backend)
    stats["backend"] = backend
    if not pairs:
        return [], stats
    # Normalize (pair, poses), (pair, poses, base_deltas) and
    # (pair, poses, base_deltas, mount_deltas) to one arity so every path below
    # — including the early bail-outs — is delta- and mount-aware. The mounts
    # have to ride each entry rather than the call, because one batch spans
    # frames whose deformation differs; without them this places a part at its
    # rest mount even though the SAT hit came from a deformed one.
    pairs = [(t[0], t[1],
              t[2] if len(t) > 2 else None,
              t[3] if len(t) > 3 else None) for t in pairs]
    if cache is None:
        cache = {}
    catalog_dir = _catalog_dir(run_dir, asset_base_dir)

    from ..sdk import _scene_kernel as _sk

    part_spec = _part_spec_lookup(ir, catalog, catalog_dir, backend=backend)

    plans = []       # (pair, key, query_tag_or_None, tangency_eps)
    queries = []
    pending_keys = set()
    for i, (pair, poses, base_deltas, mount_deltas) in enumerate(pairs):
        part_a, spec_a = part_spec(pair.a, pair.a_sub) if pair.a_sub \
            else (None, None)
        part_b, spec_b = part_spec(pair.b, pair.b_sub) if pair.b_sub \
            else (None, None)
        eps = tangency_vol_eps(part_a, part_b, vol_eps)
        if spec_a is None or spec_b is None:
            stats["not_adjudicable"] += 1
            plans.append((pair, None, None, eps))
            continue
        # Per-pair memo: each pair carries its own (poses, base_deltas), so a
        # memo shared across pairs would leak one frame's poses into the next.
        memo: dict = {}
        try:
            se3_a = _sk.instance_part_world_se3(ir, catalog, pair.a, pair.a_sub,
                                                poses=poses, memo=memo,
                                                base_deltas=base_deltas,
                                                mount_deltas=mount_deltas)
            se3_b = _sk.instance_part_world_se3(ir, catalog, pair.b, pair.b_sub,
                                                poses=poses, memo=memo,
                                                base_deltas=base_deltas,
                                                mount_deltas=mount_deltas)
        except KeyError:
            stats["not_adjudicable"] += 1
            plans.append((pair, None, None, eps))
            continue
        key = cache_key(spec_a, se3_a, spec_b, se3_b)
        if key in cache:
            stats["cache_hits"] += 1
            plans.append((pair, key, None, eps))
            continue
        if key in pending_keys:
            plans.append((pair, key, None, eps))
            continue
        tag = f"{i}:{pair.a}.{pair.a_sub}|{pair.b}.{pair.b_sub}"
        queries.append({"tag": tag, "a": spec_a, "a_se3": se3_a,
                        "b": spec_b, "b_se3": se3_b})
        plans.append((pair, key, tag, eps))
        pending_keys.add(key)

    if queries:
        stats["sent"] = len(queries)
        answers = (_answer_with_mesh(queries)
                   if backend == BACKEND_MESH
                   else _invoke(queries, tol=tol, vol_eps=vol_eps))
        if answers is None:
            stats["available"] = False
            stats["reason"] = _unavailable_reason()
            return [p for p, _, _, _ in pairs], stats
        for pair, key, tag, _eps in plans:
            if tag is not None and key is not None and tag in answers:
                cache[key] = answers[tag]

    kept = []
    for pair, key, _tag, eps in plans:
        rec = cache.get(key) if key is not None else None
        if rec is None:
            kept.append(pair)
            continue
        verdict = rec.get("verdict")
        if verdict in _CLEARING_VERDICTS:
            stats["cleared"] += 1
            continue
        if verdict != "PENETRATING":
            stats["unknown"] += 1
            kept.append(pair)
            continue
        if _is_tangency(rec, eps):
            stats["cleared"] += 1
            stats["tangency_cleared"] = stats.get("tangency_cleared", 0) + 1
            continue
        stats["confirmed"] += 1
        kept.append(_enrich(pair, rec))
    return kept, stats


def _enrich(pair, rec):
    """Attach exact measurements to a confirmed pair, if the NT allows it.

    ``depth`` / ``contact_normal`` / ``contact_point`` stay as SAT computed
    them, because they are what the SAT phase measured. The B-rep backend adds
    what SAT cannot give: ``separating_shift``, a translation of ``b`` bisected
    on the exact boolean until the intersection is empty. That is the field
    repair advice should quote — ``depth * contact_normal`` is a box quantity
    and overstates the real one, so a part moved by it lands somewhere wrong.
    The rest of the exact numbers remain additive evidence.

    On the **mesh** backend ``exact_distance`` arrives *signed* — negative by the
    measured overlap — so a reader that wants a depth rather than SAT's box
    number can take ``-exact_distance``. `reported_depth` below is that reader,
    and is the only one anything agent-facing should use; the object layer,
    which is B-rep and gets ``0.0``, is unaffected.
    """
    extra = {}
    for field, key in (("exact_distance", "distance"),
                       ("exact_volume", "common_volume")):
        if field in pair._fields and rec.get(key) is not None:
            extra[field] = float(rec[key])
    if "closest_points" in pair._fields:
        ca, cb = rec.get("closest_a"), rec.get("closest_b")
        if ca and cb:
            extra["closest_points"] = (tuple(ca), tuple(cb))
    if "separating_shift" in pair._fields:
        shift = rec.get("separating_shift")
        if shift:
            extra["separating_shift"] = tuple(float(v) for v in shift)
    if "adjudicated" in pair._fields:
        extra["adjudicated"] = True
    if not extra:
        return pair
    try:
        return pair._replace(**extra)
    except (TypeError, ValueError):
        return pair


def reported_depth(pair) -> float:
    """How deep this pair overlaps, preferring the measurement to the box.

    ``pair.depth`` is SAT's, measured on **oriented bounding boxes**, and above the object layer it is the wrong number
    to put in front of an agent because it is not bounded by the parts: a 100 mm ball can read 230 mm inside a stand.
    The mesh narrow phase measures the overlap on the triangles and returns it as a negative ``exact_distance``
    (`_enrich`), the quantity the `gap` probe reports positive when the pair is apart.

    **The smaller of the two**, because both are sound upper bounds (separating two OBBs separates the solids inside
    them, and so does a translation that pulls the triangles apart) and they are loose in opposite situations: SAT
    when the boxes are much bigger than the parts, the measurement when its direction is poor for a part mid-topple
    (`_narrow_phase._separating_shift` also slides along the contact normals for that reason). The minimum is tighter
    than either.

    SAT's number stands alone when the backend is B-rep (the object layer's, or ``DYCRAFT_COLLIDE_BACKEND=brep``), and
    when a pair could not be adjudicated and keeps its conservative verdict.

    It lives here, beside `_enrich`, because two surfaces have to agree on it: private to a verifier, it would leave
    `_probes._self_penetrations_internal` reporting `pair.depth`, so the verifier would answer on triangles and the
    probe an agent plans with on boxes, several times deeper on a humanoid's links, which that probe's docstring
    forbids ("a probe and the verifier it predicts must not answer differently").
    """
    d = getattr(pair, "exact_distance", None)
    if d is None or d >= 0.0:
        return float(pair.depth)
    return min(float(pair.depth), -float(d))
