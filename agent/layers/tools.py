"""The harness operations behind the object and scene layers' daemon ops.

They are stateful: they read and write the run directory and the verifier cache
through a shared `ToolContext` that carries a `LayerConfig`. The socket daemon
dispatches method names to Session ops, which call these:

    run_compile_and_persist(ctx, *, is_finish) -> dict      (here)
    run_probe(ctx, expression)                 -> dict      (here)
    run_final_render(ctx, *, failed=False)     -> dict      (here)
    run_snapshot(ctx, *, views, poses, only, ...) -> dict   (`snapshot.py`)
    run_review(ctx, claim, ...)                -> dict      (`review.py`)

This module holds what they share: `ToolContext`, building the IR from the
current source (`build_current_ir`), and the verifier cache.
"""

from __future__ import annotations

import builtins
import json
import os
import threading
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cad import _timelimit
from . import compile as _compile_mod
from ..cad import sandbox as _sandbox
from . import _scene_source
from ...sdk._errors import SandboxViolation, SourceSyntaxError
from ..tool_io import (
    atomic_write_text,
    confine_to_run_dir,
    round_floats,
)


COMPILE_REPORTS_DIR = "compile_reports"
LATEST_REPORT = "latest.json"
IR_JSON = "ir.json"
PROBE_REPR_MAX = 4096
RENDERS_DIR = "renders"


# ---------------------------------------------------------------------------
# Where an object.py or scene.py executes
# ---------------------------------------------------------------------------

_WORKER_REPAIRS = {
    "worker_unavailable": [
        "run `python -m dycraft.install cad` to record a build123d interpreter in "
        ".b123d_python, or set DYCRAFT_B123D_PYTHON to a Python that has build123d "
        "installed",
        "this is an environment fault, not a fault in your source; do not "
        "rewrite the geometry in response to it",
    ],
    "timeout": [
        "look for a modeling loop that never terminates",
        "Check expensive booleans and tessellation; simplify small fillets or coarsen "
        "mesh settings where detail allows.",
    ],
    "worker_crash": [
        "The CAD worker crashed. Isolate the failing operation and check for invalid "
        "or zero-thickness geometry before simplifying it.",
    ],
    "worker_internal": [
        "this is a harness fault, not a fault in your source; report it rather "
        "than rewriting the geometry",
    ],
    # Keyed on `error_kind`, which for this one is `SolidCompileError.kind`
    # (`_solid_compile.check_build123d_version`) — NOT the `build_error.error`
    # class name `Build123dVersionMismatch` the worker child writes alongside
    # it. Keying on the class name silently yields no repairs at all.
    "build123d_version_mismatch": [
        "the CAD interpreter is not the pinned build123d version; this is an "
        "environment fault, not a fault in your source",
    ],
    "ocp_missing": [
        "Environment fault: OCP, which your source can use, is not importable in the "
        "CAD interpreter. Repair the environment; this is not a fault in your source.",
    ],
}


def _worker_failure_repairs(raw: dict) -> dict:
    """Attach actionable repairs to a worker-level failure.

    The worker already returns a `build_ir_with_layer`-shaped dict on every
    failure path (`object_worker._failure`), so nothing needs re-wrapping. What
    it cannot know is what the *agent* should do about it, and the distinction
    that matters most is not visible from the message alone: a timeout or a
    crash is about the geometry, while an unavailable interpreter or a version
    mismatch is about the environment and must not send the agent off rewriting
    a perfectly good `object.py`.
    """
    err = raw.get("build_error")
    kind = raw.get("error_kind")
    if isinstance(err, dict) and kind in _WORKER_REPAIRS and not err.get(
            "suggested_repairs"):
        err["suggested_repairs"] = list(_WORKER_REPAIRS[kind])
    return raw


def _build_ir_routed(ctx: "ToolContext", source: str) -> dict:
    """Build `source` into an IR, in the process that can execute it.

    The object and scene layers' sources (`object.py`, `scene.py`) construct
    build123d geometry, and build123d lives in its own frozen environment, so
    both layers build in the CAD worker subprocess. Everything downstream
    (verifiers, probes, renderers, persistence) stays here. The result has
    `compile.build_ir_with_layer`'s dict shape either way, so no caller
    branches on which path ran.

    There is no fallback. Without a CAD interpreter this returns the worker's
    `worker_unavailable` failure rather than building primitive-only geometry
    in-process: a silently degraded build produces an object that looks
    committed and is wrong.
    """
    if ctx.layer_cfg.name not in ("object", "scene"):
        return _compile_mod.build_ir_with_layer(
            source, layer_cfg=ctx.layer_cfg, ctx=ctx)

    from ..cad import object_worker
    raw = object_worker.run_object_source(source, run_dir=ctx.run_dir,
                                          layer=ctx.layer_cfg.name)
    # The child's stdio is diagnosis material, not part of the IR contract;
    # strip it here so the payload keeps `build_ir_with_layer`'s exact shape.
    stdout = raw.pop("worker_stdout", "") or ""
    stderr = raw.pop("worker_stderr", "") or ""
    if stdout or stderr:
        ctx.last_worker_stdio = {"stdout": stdout, "stderr": stderr}
    return _worker_failure_repairs(raw)


# ---------------------------------------------------------------------------
# Build-from-current-source helper (shared by probe, snapshot, review, and compile)
# ---------------------------------------------------------------------------

def build_current_ir(ctx: "ToolContext") -> dict:
    """Read the current source file, build the IR (no verifier suite), and
    cache the result by source hash so repeated probe/snapshot calls don't
    re-exec.

    Returns a dict {ok, ir, build_error, sandbox_violation, error_summary}.
    error_summary is a one-line string suitable for showing to the agent.
    On a fresh empty source file, ok=False with a missing-source error.
    """
    import hashlib
    src_path = ctx.source_path()
    src_name = ctx.layer_cfg.source_filename
    if not src_path.exists():
        return {
            "ok": False, "ir": None, "build_error": None,
            "sandbox_violation": None,
            "error_summary": f"{src_name} does not exist yet — call write_file first",
        }
    source = src_path.read_text()
    source_hash = hashlib.blake2b(source.encode("utf-8"), digest_size=16).hexdigest()
    fresh = _cached_build_if_fresh(ctx, source_hash)
    if fresh is not None:
        return fresh

    raw = _build_ir_routed(ctx, source)
    return _cache_build(ctx, source_hash, raw)


#: The build each thread's current op is working from, per context. Read by
#: `pinned_build`; set wherever a build is produced or reused; cleared at the
#: start and end of every op (`build_pin_scope`).
_PINNED = threading.local()


def _pin(ctx: "ToolContext", result: dict) -> dict:
    # Keyed by `id(ctx)` because a context is not hashable; the entry keeps the
    # context itself, so its address cannot be reused while the pin exists, and
    # a read checks it is the same object.
    pins = getattr(_PINNED, "by_ctx", None)
    if pins is None:
        pins = _PINNED.by_ctx = {}
    pins[id(ctx)] = (ctx, result)
    return result


def pinned_build(ctx) -> "dict | None":
    """The build this thread's current op produced or reused, if any.

    `ctx._current_build_cache` is shared by every op of the run, and the
    daemon serves ops on a thread pool, so it can change in the middle of an
    op: a snapshot that rebuilds edited source replaces it while a compile of
    the previous source is still verifying. Anything that reads the build's
    products more than once per op -- the scene layer's catalog, read by
    each verifier and probe -- must read this op's build, or an instance's
    `object_ref` resolves against another build's catalog and misses.
    """
    pins = getattr(_PINNED, "by_ctx", None)
    entry = pins.get(id(ctx)) if pins else None
    return entry[1] if entry is not None and entry[0] is ctx else None


@contextmanager
def build_pin_scope():
    """Scope the pins to one op, so a pooled thread never carries an earlier
    op's build into a later one."""
    _PINNED.by_ctx = {}
    try:
        yield
    finally:
        _PINNED.by_ctx = {}


def _cached_build_if_fresh(ctx: "ToolContext", source_hash: str) -> "dict | None":
    """The cached build for this source, if nothing it depends on has changed.

    The entry source hash alone is not enough. `load_object` makes a build
    depend on files the entry source does not contain, and editing one of those
    does not change the entry hash, so a cache keyed on it alone would answer
    every probe, snapshot and compile with the geometry from before the edit.
    That failure is silent, which is what makes it worth a second read of a
    handful of small files.
    """
    cached = ctx._current_build_cache
    if cached is None or cached.get("source_hash") != source_hash:
        return None
    result = cached.get("result") or {}
    for path, digest in (result.get("source_deps") or {}).items():
        dep = confine_to_run_dir(ctx.run_dir, path)
        if dep is None or not dep.is_file():
            return None
        try:
            if _scene_source.source_digest(
                    dep.read_text(encoding="utf-8")) != digest:
                return None
        except OSError:
            return None
    return _pin(ctx, result)


#: Failure kinds that describe the machine rather than the source. Kept as a
#: set rather than a `not in` test against the geometry kinds so a new worker
#: kind has to be classified deliberately — `object_worker.FAILURE_KINDS` is
#: the registry, and `test_object_worker_routing` pins the two in step.
_ENVIRONMENT_FAILURE_KINDS = frozenset({
    "worker_unavailable",
    "worker_crash",
    "worker_internal",
    "timeout",
    "build123d_version_mismatch",
    "ocp_missing",
})


def _is_environment_failure(raw: dict) -> bool:
    """Whether this failure is about the environment, not the authored source.

    A `timeout` is here on purpose even though a runaway loop in `object.py`
    can cause one: the same source can time out on a loaded machine and build
    on an idle one, so the result is not a function of the source and has no
    business in a source-keyed cache.
    """
    if raw.get("ok"):
        return False
    return raw.get("error_kind") in _ENVIRONMENT_FAILURE_KINDS


def _cache_build(ctx: "ToolContext", source_hash: str, raw: dict) -> dict:
    """Store one build result under its source hash and return the slim form.

    Shared by `build_current_ir` and `run_compile_and_persist` so a compile
    followed by a probe or a snapshot on unchanged source reuses one build
    instead of launching the CAD worker twice, which costs seconds per call.
    """
    result = {
        "ok": raw["ok"],
        "ir": raw["ir"],
        "build_error": raw["build_error"],
        "sandbox_violation": raw["sandbox_violation"],
        # Carried, not dropped. `compile_with_layer` reads these off the
        # `prebuilt` result to put `geometry_notes` in the compile payload, the
        # only channel the agent has for what the compile did to its geometry:
        # a coarsened render mesh, a stage that raised, shells that disagree on
        # orientation. Every object and scene build passes through here.
        "geometry_notes": raw.get("geometry_notes"),
        # The object definitions a scene build authored. Carried for the same
        # reason `geometry_notes` is — the slim form is what every caller sees,
        # so a key dropped here is a key that does not exist. This one is load
        # bearing rather than diagnostic: `SceneCatalog` reads the
        # catalog back out of this cache, so dropping it would leave every
        # probe, render and verifier on the scene layer with no objects.
        "object_catalog": raw.get("object_catalog"),
        # Every file `load_object` read. Carried so `_cached_build_if_fresh`
        # can tell whether a build whose entry source is unchanged is still
        # current — without it, editing a loaded object's program would leave
        # the author staring at the previous geometry.
        "source_deps": raw.get("source_deps"),
        "error_summary": None,
    }
    if not raw["ok"]:
        err = raw["build_error"] or raw["sandbox_violation"]
        result["error_summary"] = (
            err.get("summary") if isinstance(err, dict) else "build failed"
        )
    if _is_environment_failure(raw):
        # Not cached. This cache answers "the source has not changed, so the
        # build has not either" — true of a geometry error, which is a
        # property of the source, and false of a dead worker or a missing
        # interpreter, which are properties of the machine. Caching those
        # would let one transient crash answer every later probe, snapshot and
        # compile of an unchanged source.
        ctx._current_build_cache = None
        return _pin(ctx, result)
    ctx._current_build_cache = {"source_hash": source_hash, "result": result}
    return _pin(ctx, result)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Per-run agent state. Mutable; shared by all tool fns within a run."""

    run_dir: Path
    layer_cfg: Any                                # LayerConfig
    layer_payload: Any = None                     # SceneCatalog on the scene layer
    step: int = 0
    #: The step a running compile stamps onto its artifacts, frozen at the
    #: moment that compile takes the session lock. `step` itself keeps moving —
    #: the `@op` wrapper ticks it before a handler body runs, so a second
    #: `compile()` arriving mid-flight advances it while the first is still
    #: working, and two compiles reading it late would both resolve the same
    #: `compile_reports/<step>.json`. `None` outside a compile, where the live
    #: counter is the right answer.
    artifact_step: int | None = None
    n_compile_attempts: int = 0
    last_successful_ir: dict | None = None
    last_compile_payload: dict | None = None
    verifier_cache: dict = field(default_factory=dict)
    committed: bool = False

    # Cache for the build-from-current-source helper used by probe and
    # snapshot. Keyed on source-hash so repeated probe/snapshot calls without
    # an intervening edit reuse the IR without re-exec. compile/finish share
    # it too, so an object-layer run pays for one CAD subprocess per edit
    # rather than one per tool call.
    _current_build_cache: dict | None = None
    # Last CAD worker's stdout/stderr. A modeling `print` and any kernel
    # warning land here and are otherwise lost across the process boundary.
    last_worker_stdio: dict | None = None
    user_prompt: str = ""
    render_cache: dict = field(default_factory=dict)
    # Separate from `render_cache` on purpose. A snapshot's camera id is a hash
    # of its geometry, so a collision with a canonical view is already
    # impossible — but keeping the dicts apart makes that isolation structural
    # instead of something a reader has to re-derive from the id scheme before
    # trusting that a custom render can never be served as a canonical view.
    snapshot_cache: dict = field(default_factory=dict)
    # Both layers: structured punts the sub-agent made (a feature
    # skipped, a spec simplified, a prompt reinterpreted). Surfaced via
    # record_deviation. Flushed into meta.json's `deliberate_deviations`
    # field by Session on commit.
    deliberate_deviations: list = field(default_factory=list)
    # Object layer: memo for the exact-collision adjudicator, keyed by
    # (geometry-a, geometry-b, quantized relative pose). Run-lifetime because the
    # CCD sweep asks the same question repeatedly as it steps other joints, and
    # because repair iterations usually leave most part pairs untouched — unlike
    # `verifier_cache`, which is keyed by IR hash and so misses on every edit.
    adjudication_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        # Only the directories a run actually fills — a run dir should describe
        # what happened in it, so an empty directory is a question a reader has
        # to answer before they can trust the rest.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / COMPILE_REPORTS_DIR).mkdir(exist_ok=True)
        (self.run_dir / RENDERS_DIR).mkdir(exist_ok=True)

    @property
    def catalog(self) -> Any:
        """The {hash → ObjectIR} mapping from the layer payload, or None.

        Object-layer payload is None; the scene layer's is a
        SceneCatalog.
        """
        if self.layer_payload is None:
            return None
        return getattr(self.layer_payload, "catalog", None)

    @property
    def scene_ir(self) -> Any:
        """The frozen SceneIR an animation is authored against; None elsewhere.

        Read by every animation verifier (`animation_well_formed` compares it
        against the IR's `scene_ref`; the rest resolve geometry through it).
        """
        if self.layer_payload is None:
            return None
        return getattr(self.layer_payload, "scene_ir", None)

    @property
    def asset_base_dir(self) -> Any:
        """Root for resolving an ObjectIR's `assets/brep/<digest>.brep` paths.

        A caller forwards this to
        `_exact_adjudicator.adjudicate_scene_pairs`, which without it falls back
        to `run_dir/catalog`. Passing it keeps exact collision adjudication on a
        stated input instead of the staging layout.

        A wrong root fails quietly: the adjudicator is downgrade-only, so an
        unresolvable BREP leaves the conservative SAT proxy verdict standing (a
        cylinder judged as a square prism), and the only symptom is a counter in
        the report's `adjudication` stats.
        """
        if self.layer_payload is None:
            return None
        return getattr(self.layer_payload, "asset_base_dir", None)

    def source_path(self) -> Path:
        return self.run_dir / self.layer_cfg.source_filename

    def report_path(self, step: int) -> Path:
        return self.run_dir / COMPILE_REPORTS_DIR / f"{step}.json"

    def compile_artifact_step(self) -> int:
        """The step number a compile-path artifact should be filed under.

        The frozen value while a compile holds the session lock, and the live
        counter otherwise, for callers outside a compile and tests that build a
        bare context.
        """
        return self.step if self.artifact_step is None else self.artifact_step

    def latest_report_path(self) -> Path:
        return self.run_dir / COMPILE_REPORTS_DIR / LATEST_REPORT

    def ir_path(self) -> Path:
        return self.run_dir / IR_JSON


# ---------------------------------------------------------------------------
# Path whitelist (object/scene)
# ---------------------------------------------------------------------------

def _invalidate_build_cache(ctx: ToolContext) -> None:
    """Called when the source file is mutated. Forces probe/snapshot to rebuild
    on their next invocation."""
    ctx._current_build_cache = None


# ---------------------------------------------------------------------------
# compile / finish (pure operations)
# ---------------------------------------------------------------------------

def run_compile_and_persist(ctx: ToolContext, *, is_finish: bool) -> dict:
    """The operation behind Session.compile / Session.finish. Updates ctx
    state and writes artifacts to disk (compile_reports/<ctx.step>.json,
    renders/ir.urdf, ir.json on finish); returns the JSON-serializable payload.
    """
    # Numbered before anything can return, and reported on the payload, so
    # the trace record is stamped with the attempt it describes, a refusal
    # included, and no number goes unused.
    ctx.n_compile_attempts += 1
    attempt = ctx.n_compile_attempts

    # A commit while a delegated object is still building would ship a bundle
    # that excludes work in flight and orphan the child process. The author
    # either waits for it or says explicitly that it is not wanted.
    if is_finish and ctx.layer_cfg.supports_dispatch:
        from . import _object_dispatch as _dispatch
        running = [r for r in _dispatch.dispatch_status(ctx)["dispatches"]
                   if r["status"] == "running"]
        if running:
            names = ", ".join(f"{r['name']} ({r['dispatch_id']}, "
                              f"{r['elapsed_s']:.0f}s)" for r in running)
            return {
                "ok": False, "ir_hash": None, "attempt": attempt,
                "build_error": {
                    "code": "BuildError",
                    "summary": f"{len(running)} object dispatch(es) still "
                               f"running: {names}",
                    "refs": [], "verifier_reports": [],
                    "evidence": {"reason": "dispatch_in_flight",
                                 "running": running},
                    "suggested_repairs": [
                        "await_any() to take each result, then place or "
                        "integrate what came back",
                        "cancel_dispatch(<id>) for work you have decided not "
                        "to wait for — the scene commits without it",
                    ],
                },
                "sandbox_violation": None, "verifier_reports": [],
            }

    path = ctx.source_path()
    src_name = ctx.layer_cfg.source_filename
    if not path.exists():
        payload = {
            "ok": False,
            "ir_hash": None,
            "build_error": {
                "code": "BuildError",
                "summary": f"{src_name} does not exist yet",
                "refs": [],
                "evidence": {"reason": "missing_source_file"},
                "suggested_repairs": [
                    f"call write_file with file_path={src_name!r} and old_string=\"\" first",
                ],
            },
            "sandbox_violation": None,
            "verifier_reports": [],
        }
    else:
        source = path.read_text()
        # Build where the source can run (the CAD worker for the object and
        # scene layers), then verify here. Reuses a probe/snapshot build of the
        # same source rather than paying for the subprocess twice.
        import hashlib
        source_hash = hashlib.blake2b(
            source.encode("utf-8"), digest_size=16).hexdigest()
        build = _cached_build_if_fresh(ctx, source_hash)
        if build is None:
            build = _cache_build(
                ctx, source_hash, _build_ir_routed(ctx, source))
        payload = _compile_mod.compile_with_layer(
            source,
            layer_cfg=ctx.layer_cfg,
            ctx=ctx,
            verifier_cache=ctx.verifier_cache,
            prebuilt=build,
        )
        # Strip the IR dict from the persisted/tool payload to keep size
        # bounded; full IR is in ir.json on finish.
        payload = dict(payload)
        ir = payload.pop("ir", None)
        if payload["ok"] and isinstance(ir, dict):
            ctx.last_successful_ir = ir
            _post_successful_compile(ctx, ir, is_finish=is_finish)
            if is_finish:
                atomic_write_text(
                    ctx.ir_path(),
                    json.dumps(ir, indent=2, sort_keys=True),
                )
                ctx.committed = True
            _write_layer_urdf(ctx, ir, ctx.run_dir / RENDERS_DIR / "ir.urdf")

    payload["attempt"] = attempt
    rounded_payload = round_floats(payload)
    step_path = ctx.report_path(ctx.compile_artifact_step())
    atomic_write_text(step_path, json.dumps(rounded_payload, indent=2, sort_keys=True))
    latest = ctx.latest_report_path()
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(step_path.name, latest)
    except OSError:
        atomic_write_text(latest, json.dumps(rounded_payload, indent=2, sort_keys=True))

    ctx.last_compile_payload = payload
    return payload


def _write_layer_urdf(ctx: ToolContext, ir: dict, path: Path,
                      *, layer: str | None = None) -> str | None:
    """Write the URDF export of `ir` to `path`. Returns the path on success
    or None if export was skipped (broken IR, missing deps, etc).

    `layer` ("object" or "scene") picks the exporter; it defaults to the
    layer's own, which names its IR's shape.
    """
    try:
        from ...render import _build_scene_urdf_xml, urdf_export
        path.parent.mkdir(parents=True, exist_ok=True)
        effective_layer = layer or ctx.layer_cfg.name
        if effective_layer == "scene":
            catalog = getattr(ctx.layer_payload, "catalog", {}) or {}
            xml = _build_scene_urdf_xml(
                ir, catalog, path.parent,
                asset_base_dir=getattr(
                    ctx.layer_payload, "asset_base_dir", None),
            )
        else:
            # Self-contain object mesh assets next to this URDF so it loads even
            # after the run dir moves. Mesh filenames are stored relative to the
            # run dir (assets/meshes/<digest>.obj), so run_dir is the source base.
            ir_for_urdf = ir
            if urdf_export.has_localizable_assets(ir):
                source_base = (getattr(ctx.layer_payload, "asset_base_dir", None)
                               if ctx.layer_cfg.name == "scene"
                               else ctx.run_dir)
                ir_for_urdf = urdf_export.localize_object_meshes(
                    ir, source_base_dir=source_base or ctx.run_dir,
                    dest_dir=path.parent / "assets" / "meshes",
                    urdf_dir=path.parent)
            xml = urdf_export.object_ir_to_urdf(ir_for_urdf)
        path.write_text(xml, encoding="utf-8")
        return str(path)
    except BaseException as e:  # noqa: BLE001
        print(f"[urdf] skipped {path}: {type(e).__name__}: {e}")
        return None


def _post_successful_compile(ctx: ToolContext, ir: dict, *, is_finish: bool) -> None:
    """Layer-specific hook called after a clean compile.

    For the scene layer, refresh the catalog snapshot from the IR's
    `instances` so the on-disk snapshot mirrors what the scene references.
    """
    payload = ctx.layer_payload
    refresh = getattr(payload, "refresh_snapshot", None) if payload else None
    if refresh is not None:
        refresh(ir, is_finish=is_finish)


# ---------------------------------------------------------------------------
# probe (pure operation)
# ---------------------------------------------------------------------------

def run_probe(ctx: ToolContext, expression: str) -> dict:
    """Pure operation backing Session.probe. Evaluates `expression` in the
    layer's read-only probe sandbox.

    Read-only is enforced, not assumed. `eval` mode forbids assignment but not
    method calls, and the injected `ir` is the very dict held in the build
    cache — so `ir['parts'].clear()` would corrupt it, and a later compile on
    unchanged source would report `ok: True` with an `ir_hash` computed before
    the mutation. The overlay hands out frozen views (`sdk/_readonly.py`), and
    evaluation is bounded in wall-clock time because it runs in the daemon.
    """
    if not isinstance(expression, str) or not expression:
        return {"ok": False, "error": "expression must be a non-empty string"}
    build = build_current_ir(ctx)
    current_ir = build["ir"] if build["ok"] else None
    overlay = ctx.layer_cfg.probe_overlay_factory(ctx, current_ir)
    if overlay is None:
        err = {"ok": False,
               "error": "could not build IR from current source",
               "error_summary": build.get("error_summary")}
        if build.get("build_error"):
            err["build_error"] = build["build_error"]
        if build.get("sandbox_violation"):
            err["sandbox_violation"] = build["sandbox_violation"]
        return err
    # Expression first, statements second -- the same two-mode probe the
    # animation layer takes. Statement mode lets a probe use a loop, an
    # assignment or a `try`. The sandbox, not the mode, bounds what it can call.
    # Statement mode answers with `result`, matching the name the editing
    # surfaces already use.
    mode = "eval"
    try:
        _sandbox.check_ast(expression, mode="eval")
    except SourceSyntaxError as expression_error:
        # `check_ast(mode="eval")` refuses anything that is not a single
        # expression, so a statement block arrives here rather than at the
        # sandbox. Re-check it as one. A genuinely malformed probe fails both
        # modes, and the *expression* diagnosis is the one reported: reporting
        # the statement-mode error for a mistyped expression names the wrong
        # problem.
        try:
            _sandbox.check_ast(expression, mode="exec")
            mode = "exec"
        except SourceSyntaxError:
            return {"ok": False, "error": str(expression_error)}
        except Exception as e:
            err = {"ok": False, "error": str(e)}
            if hasattr(e, "to_dict"):
                err["sandbox_violation"] = e.to_dict()
            return err
    except Exception as e:
        err = {"ok": False, "error": str(e)}
        if hasattr(e, "to_dict"):
            err["sandbox_violation"] = e.to_dict()
        return err
    limit = _timelimit.resolve_limit()
    started = _time.monotonic()
    try:
        with _timelimit.time_limit(limit):
            code = builtins.compile(expression, "<probe>", mode)
            namespace = _sandbox.build_query_globals(overlay)
            if mode == "eval":
                value = eval(code, namespace)
            else:
                exec(code, namespace)
                value = namespace.get("result")
    except _timelimit.SandboxTimeout:
        # A probe runs in the daemon, so an unbounded comprehension here is as
        # good a hang as one in a source file.
        #
        # The elapsed time is reported beside the limit because they differ.
        # The limit lands at a bytecode boundary, so one long native call (a
        # whole-instance `pair_report` over thousands of part pairs) runs past
        # it, and an elapsed time well over the limit says the query is too
        # large, where the limit alone reads as "retry with more patience".
        elapsed = _time.monotonic() - started
        return {"ok": False,
                "error": (f"probe expression did not finish within {limit:g}s "
                          f"(stopped after {elapsed:.1f}s); narrow the query "
                          "rather than retrying it"),
                "limit_s": limit, "elapsed_s": round(elapsed, 1),
                "note": ("the limit is enforced at Python bytecode boundaries, so "
                         "a single long native call runs past it before it stops")}
    except KeyboardInterrupt:
        raise
    except SandboxViolation as e:
        # The read-only views raise this: `ir['parts'].clear()` is a policy
        # denial, and the agent should see it in the same field a rejected
        # construct lands in rather than as an opaque error string.
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "sandbox_violation": e.to_dict()}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    rendered = repr(value)
    truncated = False
    if len(rendered) > PROBE_REPR_MAX:
        rendered = rendered[:PROBE_REPR_MAX] + "...<truncated>"
        truncated = True
    return {"ok": True, "repr": rendered, "truncated": truncated}


def run_final_render(ctx: ToolContext, *, failed: bool = False,
                     camera: str | None = None, focus=None) -> dict:
    """Build the IR from the current source and render its 6 canonical
    views + URDF into `<run_dir>/final_render/` (success) or
    `<run_dir>/final_render_failed/` (when `failed=True`).

    `camera` and `focus` are ignored: these layers always render all six
    canonical views. (The animation layer renders its clip through its own
    `final_render`.)

    On a source-build failure during the success path, returns the
    raw `build_current_ir` failure dict (minus the helper-internal `ir` field).
    On the failure path the build failure is folded into the standard
    `finalize_render_result` shape so the termination ceremony has a uniform
    `{ok:True, out_dir, paths, note}` to react to.
    """
    from ...render import finalize_render_result
    build = build_current_ir(ctx)
    if not build["ok"]:
        if failed:
            return finalize_render_result(
                None, ctx.run_dir, layer=ctx.layer_cfg.name,
                failed=True, build_error=build.get("error_summary"))
        return {k: v for k, v in build.items() if k != "ir"}
    is_scene = ctx.layer_cfg.name == "scene"
    catalog = (getattr(ctx.layer_payload, "catalog", None)
               if is_scene else None)
    return finalize_render_result(
        build["ir"], ctx.run_dir, layer=ctx.layer_cfg.name,
        catalog=catalog,
        asset_base_dir=(getattr(ctx.layer_payload, "asset_base_dir", None)
                        if is_scene else None),
        failed=failed)


# ---------------------------------------------------------------------------
# record_deviation — structured "I deliberately departed from the spec" log
# ---------------------------------------------------------------------------

_DEVIATION_SEVERITIES = ("low", "high")

#: Which *other* layer the reporter believes has to change for this to be
#: fixed. Deliberately only two values plus null, and deliberately NOT a
#: taxonomy of problem kinds: a fixed enum of kinds cannot keep up with what
#: actually goes wrong, so the diagnosis stays free text. What is structured
#: here is the *address* (which layer, which instances), never the diagnosis.
_DEVIATION_BLOCKED_LAYERS = ("object", "scene")

#: Cap on `targets`. A deviation naming more than a handful of instances is
#: describing a design problem, not a located one, and belongs in prose.
_DEVIATION_MAX_TARGETS = 8


def normalize_deviation(*, what, why, severity, suggested_fix=None,
                        blocked_layer=None, targets=None):
    """Validate one `record_deviation` call. Returns `(record, error)` —
    exactly one is non-None. Called by the daemon `@op` in `session.py`.
    """
    if not isinstance(what, str) or not what:
        return None, {"ok": False, "error": "what must be a non-empty string"}
    if not isinstance(why, str) or not why:
        return None, {"ok": False, "error": "why must be a non-empty string"}
    if severity not in _DEVIATION_SEVERITIES:
        return None, {"ok": False,
                      "error": f"severity must be one of "
                               f"{list(_DEVIATION_SEVERITIES)}",
                      "evidence": {"got": severity}}
    if blocked_layer is not None and blocked_layer not in _DEVIATION_BLOCKED_LAYERS:
        return None, {"ok": False,
                      "error": f"blocked_layer must be null or one of "
                               f"{list(_DEVIATION_BLOCKED_LAYERS)}",
                      "evidence": {"got": blocked_layer}}

    # `targets` is forgiving about a bare string: naming one instance is the
    # common case and `targets="fridge"` is the natural thing to type.
    if targets is None:
        norm_targets = None
    else:
        raw = [targets] if isinstance(targets, str) else targets
        if not isinstance(raw, (list, tuple)):
            return None, {"ok": False,
                          "error": "targets must be a string or a list of "
                                   "strings",
                          "evidence": {"got": type(targets).__name__}}
        norm_targets = []
        for t in raw:
            if not isinstance(t, str):
                return None, {"ok": False,
                              "error": "every entry in targets must be a "
                                       "string",
                              "evidence": {"got": repr(t)}}
            t = t.strip()
            if t and t not in norm_targets:
                norm_targets.append(t)
        if len(norm_targets) > _DEVIATION_MAX_TARGETS:
            return None, {"ok": False,
                          "error": f"targets holds {len(norm_targets)} names; "
                                   f"cap is {_DEVIATION_MAX_TARGETS}. A "
                                   "deviation naming more than a few "
                                   "instances is a design problem, not a "
                                   "located one — say that in `why`.",
                          "evidence": {"n": len(norm_targets)}}
        norm_targets = norm_targets or None

    return {
        "what": what,
        "why": why,
        "severity": severity,
        "suggested_fix": (suggested_fix if isinstance(suggested_fix, str)
                          and suggested_fix else None),
        "blocked_layer": blocked_layer,
        "targets": norm_targets,
    }, None


__all__ = [
    "COMPILE_REPORTS_DIR",
    "LATEST_REPORT",
    "IR_JSON",
    "RENDERS_DIR",
    "ToolContext",
    "normalize_deviation",
    "run_compile_and_persist",
    "build_current_ir",
    "run_probe",
    "run_final_render",
]

