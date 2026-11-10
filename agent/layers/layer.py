"""LayerConfig — per-layer configuration consumed by the shared agent harness.

One harness (sandbox/compile/tools/session) drives the object and scene layers;
the differences are captured here. Each layer ships a LayerConfig instance via a
small factory function:

    object_layer_config()   -> LayerConfig
    scene_layer_config()    -> LayerConfig

The animation layer authors a Blender project through its own operations and
session (`film.FilmSession`), and has no LayerConfig.
"""

from __future__ import annotations

import functools
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..cad import sandbox as _sandbox




# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerConfig:
    """Per-layer configuration.

    The fields are all that a layer needs to differ from the other layer.
    Anything not parameterized here is layer-agnostic and shared verbatim
    (AST allowlist, restricted builtins, FSM, run-dir layout, trace events).
    """

    name: str                                          # "object" | "scene"
    source_filename: str                               # "<name>.py"
    #: Schema tag(s) for finalized artifacts; historical IR readers own their version ladder.
    schema_tag: "str | tuple"                          # OBJECT_/SCENE_SCHEMA_VERSION

    # Reset the SDK's one-builder-per-exec singleton.
    singleton_reset: Callable[[], None]

    # Sandbox-overlay factories. They return the *additional* globals the
    # layer injects on top of the curated builtins (which the harness applies
    # itself). The factory takes the per-run ToolContext so it can reach
    # layer-specific state (e.g. the catalog for the scene layer).
    build_overlay_factory: Callable[[Any], dict]
    # The probe overlay factory takes (ctx, last_ir). If `last_ir` is None
    # and the layer disallows probing pre-compile, return None to signal
    # "not available yet". The harness gates `probe` on this.
    probe_overlay_factory: Callable[[Any, "dict | None"], "dict | None"]

    # Verifier suite. Functions accept (ir, ctx) and return VerifierReport.
    # Each tuple entry is (name, fn). The harness collects the reports and
    # places them in compile_reports/<step>.json.
    verifiers: tuple
    verifier_cache_key: Callable[[str, Any], tuple]

    # Layer-specific addenda for meta.json (alongside the base fields).
    layer_meta: Callable[[Any], dict]

    # Whether the layer offers `review` — one nested sub-agent that looks at
    # the render and judges the author's declared claims. On for the layers
    # that author geometry and declare claims about it.
    supports_review: bool = False
    # Whether the layer may delegate objects to background agent runs.
    #
    # Only the scene layer. The object layer authors one object and has
    # nothing to delegate; the scene layer cannot author geometry, so a child's
    # object would arrive somewhere it could not be edited. Delegation is only
    # coherent for an author that could also have built the thing itself.
    supports_dispatch: bool = False
    # Whether the layer offers `snapshot` — render caller-chosen cameras and
    # hand back the PNG paths. True for every layer; a flag rather than
    # an unconditional registration so a layer can decline the op and have the
    # daemon reject the call outright, rather than failing somewhere inside it.
    supports_snapshot: bool = True
    # Optional `() -> dict` called once after a successful build, whose keys
    # are merged into the build payload. Only the scene layer sets it: a scene
    # build's object definitions have to reach the *parent* process — the build
    # runs in the CAD worker and the verifiers, probes and renderers do not — so
    # they travel in the result payload, which is the only thing that crosses.
    drain_build_extras: Any = None
    # Optional `() -> dict` called on EVERY build path, success or failure,
    # whose keys are merged into the build payload. Only the scene layer
    # sets it, to return `source_deps`.
    #
    # Separate from `drain_build_extras` because the two differ on exactly one
    # question: whether a failed build should contribute it. `object_catalog`
    # must not — a rejected build contributes no objects. The dependency
    # closure must, and for the opposite reason: it is the record of what the
    # author will EDIT to fix the failure. `_cached_build_if_fresh` re-reads
    # those files to decide whether a cached result is still current, so a
    # failed build that carries none is a failure cached against its entry
    # source alone — and editing a loaded `object.py` does not change
    # `scene.py`'s hash. The cached failure would then answer every later
    # probe, snapshot and compile until `scene.py` changed.
    #
    # `_scene_source.load_object` records the dependency BEFORE the exec so
    # that a program that fails to build still invalidates. This field drains
    # it.
    drain_build_deps: Any = None
    #: Whether this layer's entry source is one object's own program.
    #:
    #: True for `object.py`, which builds exactly one object — so staging that
    #: file as the object's `source` records the recipe that produced it. False
    #: for a scene-layer `scene.py`, which builds many: staging it on each of them
    #: would claim that one file is fifteen different objects' recipe, and
    #: because `source` is inside the content hash, every edit anywhere in the
    #: file would re-id every object in it.
    #:
    #: Objects that DO have their own program still get one — `load_object`
    #: sets it for the file it is building. What a scene build drops is the
    #: false claim, not the provenance: `catalog/provenance.json` records which
    #: sources produced the bundle, and the bundle carries them.
    entry_source_is_object_program: bool = True


# ---------------------------------------------------------------------------
# Object-layer factory
# ---------------------------------------------------------------------------


def reset_object_layer_state():
    """Reset the one-Object-per-exec guard AND clear the run-local asset state.

    Distinct from `sdk.object_builder._reset_object_singleton`, which this calls:
    that one resets the builder singleton, this one also clears the run-local
    asset session and compiler.

    The mesh-asset session and the live-Shape compiler are both (re)installed
    per build by `_object_build_overlay`; clearing them here means nothing stale
    (a session pointing at a prior run dir, a compiler from a worker run)
    survives between execs.
    """
    from ...sdk.object_builder import _reset_object_singleton
    from ...sdk import _mesh_assets
    _reset_object_singleton()
    _mesh_assets.clear_active_mesh_session()
    _mesh_assets.clear_active_solid_compiler()


def load_solid_compiler():
    """The shared build123d Shape compiler, or None if it cannot be loaded.

    Imported lazily because this module is imported by every run, including
    scene runs in the main DyCraft env where build123d does not exist.
    `_solid_compile` is stdlib-only at import time, so the import succeeds there
    too and only its build123d-touching functions would fail.

    Returning None rather than propagating lets the single caller
    (`_object_build_overlay`, which asks only once build123d has been proven
    importable) turn it into a structured BuildError naming the incomplete
    checkout.
    """
    try:
        from ..cad import _solid_compile
        return _solid_compile
    except BaseException:  # noqa: BLE001 — reported as a structured BuildError
        return None


def build123d_available() -> bool:
    """True when this interpreter can construct build123d geometry.

    A capability probe rather than a switch, so there is no knob that can
    disagree with reality in either direction: claimed where build123d is absent
    (a confusing ImportError part-way through building the overlay) or denied
    where it is present (an `add_part` refusal on an interpreter that could have
    served it).

    Both answers are wanted. In the CAD worker child build123d imports and the
    live authoring surface is installed. In the main dycraft env, where the
    kernel-free tests compile box-only sources, it does not, so the overlay is
    just the SDK names (`Object`, `Material`, `Emission`, `Direction`,
    `shader`) and `add_part` refuses with a message naming the reason.

    By import, not `find_spec`: a package that is present but broken must read
    as unavailable, and only an import finds that out. `_cad_env.b123d_child_env`
    exists to prevent one such breakage (a library clash in the child's
    environment), which is exactly the shape `find_spec` would call importable.
    """
    try:
        import build123d  # noqa: F401
    except BaseException:  # noqa: BLE001 — unimportable for any reason is "no"
        return False
    return True


def _object_build_overlay(ctx):
    from ...sdk import Direction, Emission, Object, Material
    from ...sdk import _mesh_assets
    from ...sdk._shader import shader
    # Install this run's asset-staging session so `finalize` can write each
    # `add_part` solid's OBJ, B-rep and source under <run_dir>/assets/. The
    # object run tooling owns the run dir, so it owns asset staging. A ctx
    # without a run_dir (e.g. a minimal test stub compiling a box-only source)
    # simply leaves the session cleared — finalize then raises a clear "no
    # active session" error, while box-only sources compile fine.
    run_dir = getattr(ctx, "run_dir", None)
    if run_dir is not None:
        _mesh_assets.set_active_mesh_session(
            _mesh_assets.MeshAssetSession(run_dir))
    # `Direction` is the only way to name a world axis as a DirectionRef, which
    # is what `assert_parallel` / `assert_antiparallel` need to constrain a
    # joint axis by sign — `assert_axis_world_aligned` is unsigned.
    #
    # **This dict IS the material vocabulary**: `shader` + `Material` +
    # `Emission`. The AST sandbox rejects `import`, so
    # a name absent from here is unreachable rather than merely inconvenient.
    #
    # `@shader` marks a function the agent writes as ordinary Blender
    # shader-node Python. It never runs here — see `sdk/_shader` — and is
    # carried to the render child as text. `Material` still describes a surface
    # with scalars, which is what imported assets, room defaults and every part
    # too small or too hidden to be worth a graph use.
    #
    # `Emission` is not a second kind of surface but a field on that one:
    # `Material(..., emission=Emission(...))` is a bulb, which has no surface
    # worth describing. It is a physical quantity in W/m² that a photograph
    # cannot describe and a shader is not asked to compute, so the harness binds
    # it in Cycles either way (`render/_blender_child._apply_emission`).
    #
    # **An IMAGE-backed `Material` cannot be authored here.** `Texture` /
    # `TextureMap` / `Mapping` are absent from this dict, so the only textures
    # in a run are the ones importers stage from a glTF or URDF asset. The UV
    # bake in `_solid_compile` (`part_uv_phase`) serves those, and a producer
    # that returns a `Texture` plus a `Mapping` needs no changes below.
    #
    # All of these go in before the CAD check below so a box-only source can
    # bind one too.
    overlay: dict = {"Object": Object, "Material": Material,
                     "Emission": Emission, "Direction": Direction}
    overlay["shader"] = shader
    if not build123d_available():
        # No CAD here. The SDK names only, with no build123d: `add_part` then
        # refuses with a message naming the missing compiler, rather than the
        # source dying on a NameError for `Box`. Only the tests that compile
        # box-only sources in the main DyCraft env land here: the object and
        # scene layers always build in the CAD worker child
        # (`tools._build_ir_routed`).
        return overlay

    # CAD is available. Installing the compiler is what makes `add_part` work;
    # the namespace is what lets the source construct a Shape to hand it. Both
    # are required, so a half-configured environment fails loudly here rather
    # than producing a confusing NameError inside object.py.
    from ...sdk._errors import BuildError
    from ..cad import _b123d_namespace
    compiler = load_solid_compiler()
    if compiler is None:
        raise BuildError(
            "build123d is importable here but the shared solid compiler could "
            "not be loaded; the checkout is incomplete",
            refs=[], evidence={"reason": "solid_compiler_unavailable"},
            suggested_repairs=[
                "check that agent/cad/_solid_compile.py and its "
                "sibling _geometry_qc.py are present"])
    _mesh_assets.set_active_solid_compiler(compiler)
    try:
        overlay.update(_b123d_namespace.build_namespace(reserved=overlay))
    except ImportError as e:
        # The probe above already imported build123d, so reaching here means the
        # *namespace* could not be assembled from it — a version skew, not an
        # absent package.
        raise BuildError(
            f"build123d imports in this interpreter but its authoring "
            f"namespace could not be built ({e}); the interpreter is probably "
            f"not the pinned version",
            refs=[], evidence={"reason": "build123d_namespace_unavailable"},
            suggested_repairs=["point the worker at an interpreter with the "
                               "pinned build123d installed"]) from None
    return overlay


#: Exact-collision probes on the object layer, bound by hand for the same
#: reason `_SCENE_EXACT_PROBES` are — they reach the adjudicator and so need
#: state the agent cannot pass.
#:
#: Bound so the probes answer as `NoSelfPenetration` does, from B-rep
#: booleans. Unbound, `penetration_pairs`, `pair_report`, `nearest_neighbors`
#: and `find_clearance_risks` answer from OBBs or AABBs.
_OBJECT_EXACT_PROBES = ("gap", "support_report", "penetration_pairs",
                        "pair_report", "nearest_neighbors",
                        "find_clearance_risks")


def _bind_object_exact_probe(fn, ctx):
    """Inject the CAD run dir and the shared adjudication cache.

    **Not `_bind_self_probe`**, and the difference is where a `.brep` lives. On
    the scene and animation layers the object catalog is staged somewhere else
    and named by `asset_base_dir`; on the object layer the agent is authoring
    the geometry and `_stage_registered_parts` puts it under the run dir, which
    is what `object_verify.no_self_penetration` already passes as
    `adjudicate_object_pairs(run_dir=...)`. Handing this probe an
    `asset_base_dir` instead would resolve every brep path to nothing —
    `geometry_spec` returns None — and the probe would answer from the bounding
    box with no error and nothing but `exact=False` to say so.
    """
    run_dir = getattr(ctx, "run_dir", None)
    cache = getattr(ctx, "adjudication_cache", None)

    def bound(*args, **kwargs):
        kwargs.setdefault("run_dir", run_dir)
        kwargs.setdefault("cache", cache)
        return fn(*args, **kwargs)
    return bound


def _object_probe_overlay(ctx, last_ir):
    if last_ir is None:
        return None
    from ...sdk import _probes

    def _object_source():
        """This object's authored `object.py`, as TEXT.

        A closure rather than a plain probe alias because reading the file needs
        the run dir, which a stateless probe over the IR does not have. Returns
        None for an object with no authored parts, and the bare path if the
        asset is unreadable — a probe must not raise mid-expression.

        Takes no argument: one program builds every part, so a part name would
        select nothing.
        """
        rel = _probes.object_source_path(last_ir)
        if rel is None:
            return None
        run_dir = getattr(ctx, "run_dir", None)
        if run_dir is None:
            return rel
        try:
            return (run_dir / rel).read_text(encoding="utf-8")
        except OSError:
            return rel

    return {
        # Frozen, not the cached dict itself. `probe` evaluates in eval
        # mode, which forbids assignment but permits method calls, and `ir` is
        # the live object held in ctx._current_build_cache — so
        # `ir['parts'].clear()` would corrupt the cache, and a later compile on
        # unchanged source would report ok=True with the pre-mutation ir_hash.
        "ir": _sandbox.freeze(last_ir, "ir"),
        # Real geometry, for the parts that have any: the derived fact table and
        # the authored recipe. Everything else here bottoms out in an AABB.
        "part_shape": _probes.object_part_shape,
        "object_source": _object_source,
        # Migrated query symbols
        "list_parts": _probes.object_list_parts,
        "list_joints": _probes.object_list_joints,
        "joint_table": _probes.object_joint_table,
        "local_bbox": _probes.object_local_bbox,
        "frame_of": _probes.object_frame_of,
        "global_bbox": _probes.object_global_bbox,
        "centroid": _probes.object_centroid,
        "part_color": _probes.object_part_color,
        # Adjudicated now, so it needs the run dir and cache: without them
        # every candidate is unanswerable and the agent gets the SAT proxy
        # back with only `adjudicated=False` to say so.
        "penetration_pairs": _bind_object_exact_probe(
            _probes.object_penetration_pairs, ctx),
        "anchor_position": _probes.object_anchor_position,
        "outward": _probes.object_outward,
        "ir_snapshot": _probes.object_ir_snapshot,
        # Resolver helpers (shipped with verifier impl; exposed in overlay)
        "joint_axis_world": _probes.object_joint_axis_world,
        "part_world_aabb": _probes.object_part_world_aabb,
        "resolve_entity_aabb": _probes.object_resolve_entity_aabb,
        "resolve_direction": _probes.object_resolve_direction,
        # Collision probes. `gap` is the only exact one — everything else here
        # is an oriented or axis-aligned box, and the two answer different
        # questions rather than nearby ones.
        "gap": _bind_object_exact_probe(_probes.object_gap, ctx),
        # The three that make a contact claim are bound; `within_report` is
        # box algebra by definition and needs nothing.
        "pair_report": _bind_object_exact_probe(
            _probes.object_pair_report, ctx),
        "within_report": _probes.object_within_report,
        "nearest_neighbors": _bind_object_exact_probe(
            _probes.object_nearest_neighbors, ctx),
        "find_clearance_risks": _bind_object_exact_probe(
            _probes.object_find_clearance_risks, ctx),
        # Assertion-mirror probes
        "distance_between": _probes.object_distance_between,
        "overlap_along": _probes.object_overlap_along,
        "angle_between": _probes.object_angle_between,
        "motion_displacement": _probes.object_motion_displacement,
        # Verifier mirrors. `support_report` is bound rather than passed
        # straight through: its contact test is exact, so it needs the same run
        # dir and cache `gap` does or it would predict the verdict from boxes
        # the verifier does not use.
        "support_report": _bind_object_exact_probe(
            _probes.object_support_report, ctx),
    }


def _object_cache_key(ir_hash, ctx):
    return ()


def _object_layer_meta(ctx):
    return {
        "layer": "object",
        "deliberate_deviations": list(
            getattr(ctx, "deliberate_deviations", None) or []),
    }


def object_layer_config() -> LayerConfig:
    """Factory: assemble the LayerConfig for the object agent."""
    from ...sdk.ir import OBJECT_SCHEMA_VERSION
    from ...verify import OBJECT_VERIFIERS

    return LayerConfig(
        name="object",
        source_filename="object.py",
        schema_tag=OBJECT_SCHEMA_VERSION,
        singleton_reset=reset_object_layer_state,
        build_overlay_factory=_object_build_overlay,
        probe_overlay_factory=_object_probe_overlay,
        verifiers=OBJECT_VERIFIERS,
        verifier_cache_key=_object_cache_key,
        layer_meta=_object_layer_meta,
        supports_review=True,
    )


# ---------------------------------------------------------------------------
# Scene probes (the scene layer's probe surface)
# ---------------------------------------------------------------------------


#: Exact-collision probes need the adjudicator's asset root and
#: shared cache, so a plain `SCENE_PROBES` entry would be called with
#: `asset_base_dir=None` and answer from bounding boxes without saying so.
#:
#: They exist because every other collision probe on this layer is a box while
#: the layer's own verifiers are exact. `NoCrossInstancePenetration` adjudicates
#: every SAT candidate against real geometry and `RoomContainment` is the room's
#: verdict, but unbound `penetration_pairs` / `pair_report` are the SAT and
#: AABB broad phases, so the agent would plan against one number and be judged
#: on another.
#:
#: Bound individually so the scene asset root and cache reach every query.
_SCENE_EXACT_PROBES = ("gap", "room_gap", "penetration_pairs",
                       "support_report", "pair_report",
                       "nearest_neighbors", "find_clearance_risks")


def _scene_probe_overlay(ctx, last_ir):
    """Scene-mode probe globals.

    The probe model:
      * `objects` is bound from step 1 (always available).
      * `ir` and `catalog` are bound only after a successful prior compile.
      * `Scene` is NOT in probe globals (probe is read-only).
      * scene-layer probe symbols are always available.
    """
    from ...sdk import _probes
    overlay: dict = {
        "objects": _sandbox.freeze(ctx.layer_payload.object_handles, "objects"),
    }
    overlay.update(_probes.SCENE_PROBES)
    # Written out rather than looped over `_SCENE_EXACT_PROBES` with a
    # `getattr(_probes, f"scene_{short}")`: that spelling makes the functions
    # in `sdk/_probes.py` look unreferenced to static analysis (see the comment
    # above `SCENE_PROBES`), since a name in an f-string is not a reference.
    # The tuple stays as the documented list of what is bound here.
    overlay["gap"] = _bind_self_probe(_probes.scene_gap, ctx)
    overlay["room_gap"] = _bind_self_probe(_probes.scene_room_gap, ctx)
    # Same reason: its contact test is measured on surfaces, so it needs
    # the asset root and cache to predict what `SupportConnected` will say.
    overlay["support_report"] = _bind_self_probe(
        _probes.scene_support_report, ctx)
    # Adjudicated too, for the same reason. Absent from `SCENE_PROBES` rather
    # than overridden here: a registry exists to be bulk-bound, and a name that
    # needs state an agent cannot pass cannot be.
    overlay["penetration_pairs"] = _bind_self_probe(
        _probes.scene_penetration_pairs, ctx)
    overlay["pair_report"] = _bind_self_probe(_probes.scene_pair_report, ctx)
    overlay["nearest_neighbors"] = _bind_self_probe(
        _probes.scene_nearest_neighbors, ctx)
    overlay["find_clearance_risks"] = _bind_self_probe(
        _probes.scene_find_clearance_risks, ctx)
    if last_ir is not None:
        overlay["ir"] = _sandbox.freeze(last_ir, "ir")
        overlay["catalog"] = _sandbox.freeze(ctx.layer_payload.catalog, "catalog")
    return overlay


# ---------------------------------------------------------------------------
# Scene-layer factory
# ---------------------------------------------------------------------------
#
# One `scene.py` that authors its own objects and places them, so a shared
# dimension can drive a counter's cutout, the sink that sits in it, and where
# the sink goes, in one program that can be revised as one thing.
#
# The overlay is the object layer's plus the scene vocabulary, not written
# fresh: the geometry vocabulary an author gets here has to be the same one
# `object.py` gets, or the object prompt and the build123d skill stop
# describing what this layer can do.


def reset_scene_layer_state():
    """Reset both builders' per-execution state for a scene build.

    The object reset is called with `limit=None`: a scene-layer `scene.py` is
    *expected* to construct many Objects, so the one-per-execution guard that
    protects `object.py` from a duplicated builder would forbid this layer's
    entire purpose. Everything else it clears — the asset session, the live
    Shape compiler, the finalized-object registry — is cleared for exactly the
    reasons the object layer clears it.
    """
    from ...sdk.scene_builder import _reset_scene_singleton
    from ...sdk.object_builder import _reset_object_singleton
    from ...sdk import _mesh_assets
    from . import _scene_source
    _reset_object_singleton(limit=None)
    _reset_scene_singleton()
    _mesh_assets.clear_active_mesh_session()
    _mesh_assets.clear_active_solid_compiler()
    _scene_source.reset_source_deps()


def _scene_build_overlay(ctx):
    """The object vocabulary plus the scene vocabulary, in one namespace.

    `_object_build_overlay` does the load-bearing half: it installs this run's
    `MeshAssetSession` and the live-Shape compiler, and binds the curated
    build123d namespace. Calling it rather than reproducing it is what keeps
    the two layers' geometry surface identical by construction.

    What is added is the scene vocabulary, minus `objects`. There is no
    injected catalog here — the objects are whatever this program builds, so a
    frozen dict of pre-existing handles would be empty and misleading.
    """
    from ...sdk import Scene
    from ...sdk._lighting_block import lighting
    from . import _scene_source

    overlay = _object_build_overlay(ctx)
    overlay["Scene"] = Scene
    # `@lighting`: the AST sandbox rejects `import`, `Scene.set_lighting` is the
    # only lighting surface there is, and a scene with no block renders unlit.
    # `Environment` is deliberately not bound.
    overlay["lighting"] = lighting
    # Bound last and given the overlay itself: a loaded program executes
    # against this same namespace, so it can load one in turn — and, through
    # `load_asset` beside it, reach a preloaded asset from a loaded program
    # rather than only from `scene.py`.
    overlay["load_object"] = _scene_source.make_load_object(ctx, overlay)
    overlay["load_asset"] = _scene_source.make_load_asset(ctx, overlay)
    return overlay


def _scene_layer_probe_overlay(ctx, last_ir):
    """Scene-layer probes over the catalog this build produced.

    `objects` is bound from the built catalog rather than from a run-start
    payload, so before the first successful build it is empty — which is
    accurate, since no object exists yet.
    """
    overlay = _scene_probe_overlay(ctx, last_ir)
    overlay["objects"] = _sandbox.freeze(
        dict(getattr(ctx.layer_payload, "object_handles", {}) or {}), "objects")
    return overlay


def _scene_cache_key(ir_hash, ctx):
    """No extra key material: the SceneIR hash already covers the geometry.

    The objects are outputs of the build. A SceneIR names each instance's
    definition by content hash, so any change to any object's geometry changes
    an `object_ref`, which changes the scene's own hash. One hash identifies
    the whole build.
    """
    return ()


def _scene_layer_meta(ctx):
    catalog = getattr(ctx.layer_payload, "catalog", {}) or {}
    return {
        "layer": "scene",
        "n_objects": len(catalog),
        "deliberate_deviations": list(
            getattr(ctx, "deliberate_deviations", None) or []),
    }


def _scene_drain_extras():
    """The definitions a SUCCESSFUL scene build authored.

    Success-only, and deliberately: a rejected build contributes no objects.
    The dependency closure is drained separately by `_scene_drain_deps`,
    which runs on every path — see `LayerConfig.drain_build_deps`.
    """
    from ...sdk.object_builder import take_finalized_objects
    return {"object_catalog": take_finalized_objects()}


def _scene_drain_deps():
    """Every file `load_object` read, on success or failure.

    This is what lets the parent tell whether a cached build is still current
    when only a loaded program changed, including a cached FAILURE. See
    `LayerConfig.drain_build_deps`.

    Draining is also what clears the module-level ledger, so a path that
    skipped it would leave one build's dependencies visible to the next.
    """
    from . import _scene_source
    return {"source_deps": _scene_source.take_source_deps()}


def scene_layer_config() -> LayerConfig:
    """Factory: assemble the LayerConfig for the scene author."""
    from ...sdk.ir import SCENE_SCHEMA_VERSIONS
    from ...verify import SCENE_VERIFIERS

    return LayerConfig(
        name="scene",
        # Both scene schema tags: `Scene.finalize` emits v2 exactly when some
        # instance carries `initial_joints` and v1 otherwise, so an author's
        # choice of feature must not decide whether their build is accepted.
        schema_tag=SCENE_SCHEMA_VERSIONS,
        source_filename="scene.py",
        singleton_reset=reset_scene_layer_state,
        build_overlay_factory=_scene_build_overlay,
        probe_overlay_factory=_scene_layer_probe_overlay,
        verifiers=SCENE_VERIFIERS,
        verifier_cache_key=_scene_cache_key,
        layer_meta=_scene_layer_meta,
        drain_build_extras=_scene_drain_extras,
        drain_build_deps=_scene_drain_deps,
        supports_review=True,
        supports_dispatch=True,
        entry_source_is_object_program=False,
    )


# ---------------------------------------------------------------------------
# Probe binding and the layer table
# ---------------------------------------------------------------------------


def _bind_self_probe(fn, ctx, *leading):
    """Bind the leading IR/catalog args plus the adjudicator's asset root, the
    shared cache and the run dir, which the plain `_bind_*` helpers do not carry.

    The run dir is where a run's staged assets resolve from.
    """
    root = getattr(ctx, "asset_base_dir", None)
    cache = getattr(ctx, "adjudication_cache", None)
    run_dir = getattr(ctx, "run_dir", None)

    accepts_run_dir = "run_dir" in inspect.signature(fn).parameters

    def bound(*args, **kwargs):
        kwargs.setdefault("asset_base_dir", root)
        kwargs.setdefault("cache", cache)
        if accepts_run_dir:
            kwargs.setdefault("run_dir", run_dir)
        return fn(*leading, *args, **kwargs)
    return bound


#: `--layer <name>` -> its LayerConfig factory. `animation` and `plain` are
#: absent: neither has a LayerConfig (see `film.FilmSession` and `agent.plain`).
_LAYER_CONFIGS = {
    "object": object_layer_config,
    "scene": scene_layer_config,
}


__all__ = [
    "LayerConfig",
    "object_layer_config",
    "scene_layer_config",
]
