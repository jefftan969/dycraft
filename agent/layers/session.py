"""LayerSession — object/scene-layer agent session.

Generic plumbing (step counter, trace.jsonl, meta.json,
op-call trace pattern, daemon handler registry) lives in
`agent.agent_session.AgentSession`. This module only carries the canonical
object/scene ops (`compile / finish / probe / snapshot / review /
final_render / record_deviation`) and the commit-time
`deliberate_deviations` flush onto meta.json.

The compile op returns the slimmed payload — verifier_reports filtered to
non-pass entries; passing names moved into a flat `verifiers_passed` list —
so every consumer (LLM tool surface AND daemon shell surface) sees a
size-bounded result. The full payload is persisted to
`compile_reports/<step>.json` regardless.

`Session` and `SessionConfig` are kept as back-compat aliases.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...verify.report import PASSING
from ..agent_session import AgentSession
from .layer import LayerConfig
from . import _object_dispatch as _dispatch
from .review import run_review
from .snapshot import run_snapshot
from .tools import (
    ToolContext,
    normalize_deviation,
    run_compile_and_persist,
    run_final_render,
    run_probe,
)


@dataclass
class SessionConfig:
    """Inputs needed to construct an LayerSession."""

    run_dir: Path
    layer: LayerConfig
    layer_payload: Any = None
    user_prompt: str = ""
    initial_meta: dict | None = None


def _slim_compile_payload_for_llm(payload: dict) -> dict:
    """Filter verifier_reports to the entries the agent has to read; the rest
    of the passing names move into a flat `verifiers_passed` list. The full
    payload is what gets persisted to `compile_reports/<step>.json`; this slim
    version is what the harness daemon ships back to the agent's
    `import harness as h` shell.

    One payload, one level. A `storyboard` reply nests a check payload per
    planned state per shot, and those need slimming too — but the walking
    lives in `animation.jobs.slim_job_payload`, which knows that
    shape. Both of them recursing means the nested checks get visited twice.
    """
    slimmed = dict(payload)
    reports = payload.get("verifier_reports")
    if reports is not None:
        kept = [r for r in reports if r.get("verdict") not in PASSING]
        kept_names = {id(r) for r in kept}
        slimmed["verifier_reports"] = kept
        slimmed["verifiers_passed"] = [r.get("name") for r in reports
                                       if r.get("verdict") == "pass"
                                       and id(r) not in kept_names]
        # Named apart from the passes, so the pass count never includes checks
        # that had nothing to check.
        idle = [r.get("name") for r in reports
                if r.get("verdict") == "not_applicable" and id(r) not in kept_names]
        if idle:
            slimmed["verifiers_not_applicable"] = idle
    return slimmed


#: How long between repeats once the advisory starts firing. Every op would
#: nag — a probe-heavy run issues dozens a minute — and once per run is too
#: easy to lose in a long transcript.
_ADVISORY_REPEAT_MIN = 15.0


def _advisory_limit() -> float:
    """Minutes before the advisory starts, or 0 when it is switched off."""
    try:
        return max(0.0, float(os.environ.get("DYCRAFT_ADVISORY_HOURS") or 0) * 60.0)
    except ValueError:
        return 0.0


def require_cad_worker(session) -> None:
    """Refuse to start a run with no CAD interpreter.

    Every authoring layer executes its source in the frozen build123d
    environment (`tools._build_ir_routed`), so without an interpreter there is
    nothing that can run the agent's source at all. Failing here, before the
    agent has been handed a prompt telling it to author build123d geometry, is
    far kinder than failing at the first build, and the one thing that must
    never happen is quietly building primitive-only geometry instead: that
    commits an object that looks fine and is not what was asked for.
    """
    from ..cad import object_worker
    if object_worker.is_available():
        return
    msg = (
        "no build123d interpreter found, so the design source cannot be "
        "built. Run `python -m dycraft.install cad` to record one in "
        ".b123d_python, or set DYCRAFT_B123D_PYTHON. There is no in-process "
        "fallback."
    )
    session.trace("cad_worker_unavailable", detail=msg)
    print(f"[harness] FATAL: {msg}", file=sys.stderr)
    raise RuntimeError(msg)


class LayerSession(AgentSession[ToolContext]):
    """Object/scene session: the canonical ops + commit-time deviations flush.

    Ops are registered via `AgentSession.op(name)` decorator so they go
    through the standard `tick → op_call → run → op_result` trace
    pattern; the daemon dispatches socket method names via
    `session.handlers[name](**kwargs)`.
    """

    def op_scope(self):
        """Pin each op to the build it produced or reused, so a concurrent op's
        rebuild cannot swap the catalog under it (`tools.pinned_build`)."""
        from .tools import build_pin_scope
        return build_pin_scope()

    def __init__(self, cfg: SessionConfig):
        ctx = ToolContext(
            run_dir=cfg.run_dir,
            layer_cfg=cfg.layer,
            layer_payload=cfg.layer_payload,
            user_prompt=cfg.user_prompt,
        )
        # A payload that derives its answers from the run's live build state
        # needs the context that holds it. Built before any ToolContext exists
        # (`server._build_layer` has only the run dir), so the two are joined
        # here. Duck-typed rather than layer-checked: a payload either offers
        # `bind` or does not.
        _bind = getattr(cfg.layer_payload, "bind", None)
        if callable(_bind):
            _bind(ctx)
        super().__init__(
            ctx,
            user_prompt=cfg.user_prompt,
            initial_meta_fields=cfg.initial_meta,
        )
        #: One compile at a time per session. The daemon services handlers on
        #: a thread pool (`agent/server.py`, so a long op cannot head-of-line
        #: block `probe`/`read_file`), and nothing else stops a second
        #: `compile()` starting while the first is still running. An agent
        #: whose foreground call is killed by its own tool timeout has every
        #: reason to re-issue it, and two compiles of the same IR would each
        #: pay for a full verifier pass.
        #:
        #: Per session, not global: independent runs are separate daemons, so
        #: none of them serialize against each other.
        self._compile_lock = threading.Lock()
        self._t_start = time.monotonic()
        self._advised_at = 0.0
        self._register_ops()
        self._require_cad_worker()


    def annotate_payload(self, op_name: str, payload: dict) -> dict:
        """Fold live object dispatches into every op result.

        The notification channel for background work, in a harness where the
        agent cannot be interrupted: whatever it does next tells it that a
        child landed. `dispatch_status` and `await_any` are excluded — they
        already answer this question, and repeating it inside their own
        payloads reads as a second, disagreeing copy.
        """
        if op_name in ("dispatch_status", "await_any", "submit_object"):
            return payload
        rows = _dispatch.readiness(self.ctx)
        if rows:
            payload = dict(payload)
            payload["dispatches"] = rows
        payload = self._maybe_advise(payload)
        return payload

    def _maybe_advise(self, payload: dict) -> dict:
        """Tell the agent how long it has been running, past the advisory mark.

        Advisory, not a cap: it consumes no budget, blocks nothing, and names
        the exit that already exists rather than inventing one. The wording
        tracks `agent/prompts/object.md` §Deviations so the agent is being
        pointed at a route it was already told about.
        """
        limit = _advisory_limit()
        if not limit:
            return payload
        mins = (time.monotonic() - self._t_start) / 60.0
        if mins < limit or mins - self._advised_at < _ADVISORY_REPEAT_MIN:
            return payload
        self._advised_at = mins
        payload = dict(payload)
        payload["elapsed_hours"] = round(mins / 60.0, 1)
        payload["advisory"] = (
            f"This session has run for {mins / 60.0:.1f} hours. If the last "
            "several operations have not moved the geometry towards the "
            "request, the remaining gap might be difficult for this "
            "representation to meet: record a deviation, declare intended "
            "consequences in the model, and finish with what builds.")
        return payload

    def _require_cad_worker(self) -> None:
        require_cad_worker(self)

    # ------------------------------------------------------------------ ops

    def _register_ops(self) -> None:
        ctx = self.ctx
        self._register_deviation()

        @self.op("compile")
        def _compile(*, freeze: bool = False) -> dict:
            return self._do_compile(freeze=bool(freeze))

        @self.op("finish")
        def _finish() -> dict:
            return self._do_compile(freeze=True)

        # Cheap: ~0 s median over a traced corpus. Inspecting the geometry
        # you just wrote must not compete with compiling it.
        @self.op("probe")
        def _probe(*, expression: str) -> dict:
            return run_probe(ctx, expression)

        # `snapshot` needs only the renderer, which is camera-general. The frame
        # args ride the wire because the shared stub carries them; `run_snapshot`
        # refuses them (frames are the animation layer's own `snapshot`).
        if ctx.layer_cfg.supports_snapshot:
            @self.op("snapshot")
            def _snapshot(*, views=None, poses=None, only=None,
                          resolution=None, out_dir: str = "snapshots",
                          eye=None, look_at=None, up=None,
                          fov_deg: float = 60.0,
                          frame=None, frames=None) -> dict:
                return run_snapshot(ctx, views=views, poses=poses, only=only,
                                    resolution=resolution, out_dir=out_dir,
                                    eye=eye, look_at=look_at, up=up,
                                    fov_deg=fov_deg,
                                    frame=frame, frames=frames)

        # Scene layer only: `submit_object` delegates one object to a
        # background agent run. `finish` refuses while a dispatch is still
        # running (`tools.run_compile_and_persist`).
        if ctx.layer_cfg.supports_dispatch:
            @self.op("submit_object")
            def _submit_object(**kwargs) -> dict:
                return _dispatch.submit_object(ctx, **kwargs)

            # `timeout=0` is a poll. A longer timeout spends wall clock waiting.
            @self.op("await_any")
            def _await_any(dispatch_ids=None, timeout=None) -> dict:
                return _dispatch.await_any(ctx, dispatch_ids=dispatch_ids,
                                           timeout=timeout)

            @self.op("dispatch_status")
            def _dispatch_status() -> dict:
                return _dispatch.dispatch_status(ctx)

            @self.op("cancel_dispatch")
            def _cancel_dispatch(dispatch_id: str) -> dict:
                return _dispatch.cancel_dispatch(ctx, dispatch_id)

        if ctx.layer_cfg.supports_review:
            @self.op("review")
            def _review(targets=None) -> dict:
                return run_review(ctx, targets=targets)

        # The controlled path for changing renderer mid-run. It exists because
        # the alternative is worse: with no op, an agent that wants Blender
        # relaunches the harness daemon with DYCRAFT_RENDER_BACKEND set, and a
        # second daemon on one run dir corrupts trace.jsonl, meta.json and the
        # step counter (see `agent/server.py::acquire_run_dir`).
        # `render.default_backend()`
        # re-reads the env per request, so setting it here is all that a
        # subsequent `compile` / `view` / `snapshot` / `final_render` needs.
        @self.op("set_render_backend")
        def _set_render_backend(*, backend: str) -> dict:
            return self._do_set_render_backend(backend)

        @self.op("final_render")
        def _final_render(*, failed: bool = False, camera: str | None = None,
                          focus=None) -> dict:
            # `camera` and `focus` are accepted and ignored: these layers
            # render all five canonical views of everything.
            result = run_final_render(ctx, failed=bool(failed), camera=camera,
                                      focus=focus)
            if result.get("ok") and not failed:
                # The views and the URDF at the run's top level, for whoever
                # opens the run folder. Never fails the render.
                from ...run_layout import publish
                publish(ctx.run_dir)
            return result

    def _register_deviation(self):
        ctx = self.ctx
        # `h.record_deviation(...)`; validated by `tools.normalize_deviation`.
        # Free, and declaring a deviation honestly should never be rationed.
        @self.op("record_deviation")
        def _record_deviation(*, what: str, why: str, severity: str,
                              suggested_fix: str | None = None,
                              blocked_layer: str | None = None,
                              targets=None) -> dict:
            record, err = normalize_deviation(
                what=what, why=why, severity=severity,
                suggested_fix=suggested_fix,
                blocked_layer=blocked_layer, targets=targets)
            if err is not None:
                return err
            ctx.deliberate_deviations.append(record)
            return {"ok": True,
                    "n_deliberate_deviations": len(ctx.deliberate_deviations),
                    "recorded": record}

    def _do_set_render_backend(self, backend: str) -> dict:
        """Point subsequent renders at `backend`, for the rest of this run.

        Validated against the registry rather than accepted blindly: an
        unknown name would otherwise surface much later, as a failed render
        several steps on, and the agent would have no way to tell a typo from
        a missing dependency. The change is recorded in `trace.jsonl` and
        `meta.json` so the bundle says which renderer produced which artifact;
        the launcher's value alone would credit Open3D with images Blender made.
        """
        from ...render import _BACKEND_MODULES, _BACKENDS, RENDER_BACKEND_ENV

        known = sorted(set(_BACKENDS) | set(_BACKEND_MODULES))
        if not isinstance(backend, str) or backend not in known:
            return {"ok": False,
                    "error": (f"unknown render backend {backend!r}; "
                              f"known: {known}")}
        previous = os.environ.get(RENDER_BACKEND_ENV) or "open3d"
        # **Availability, not just spelling.** The registry says the name
        # exists; it says nothing about whether this machine can run it. A
        # Blender render needs an interpreter resolved from
        # `$DYCRAFT_BLENDER_PYTHON`, `.blender_python` or a conventional conda
        # prefix. Accepting the switch without one would surface the refusal
        # once per render, several steps later, and send the agent on a round
        # trip out to `blender` and back. The same sentence, said at the
        # switch, costs one op.
        if backend == "blender":
            from ...render import _blender_env
            if not _blender_env.is_available():
                return {"ok": False, "previous": previous,
                        "error": ("no Blender interpreter configured; backend remains "
                                  f"{previous!r}. It is set up outside the run, with "
                                  "`python -m dycraft.install blender` or "
                                  "DYCRAFT_BLENDER_PYTHON set before launch.")}
        os.environ[RENDER_BACKEND_ENV] = backend
        self.trace("render_backend_set", backend=backend, previous=previous)
        return {"ok": True, "render_backend": backend, "previous": previous}

    def _do_compile(self, *, freeze: bool) -> dict:
        """compile + finish share the same path. The outer @op wrapper
        already ticked the step counter and emitted `op_call`; this body
        emits `compile_attempt` and (on a clean finish) `commit` +
        commit-time `deliberate_deviations` flush. Returns the slim payload.

        Serialized on `_compile_lock`. A second call waits rather than starting
        a parallel verifier pass, and by the time it runs the build and
        verifier caches are warm — so an unchanged IR comes back fast.

        The artifact step is the one this op was handed by `tick()`, not the
        counter's value when the lock is finally acquired — a queued compile
        ticks while the running one waits, so by then `ctx.step` belongs to
        somebody else, and two compiles would resolve the same
        `compile_reports/<step>.json`, one overwriting the other.
        """
        my_step = self.current_op_step()
        with self._compile_lock:
            self.ctx.artifact_step = (
                self.ctx.step if my_step is None else my_step)
            try:
                return self._compile_locked(freeze=freeze)
            finally:
                self.ctx.artifact_step = None

    def _compile_locked(self, *, freeze: bool) -> dict:
        """The compile body proper. Runs with `_compile_lock` held."""
        was_committed = self.ctx.committed
        t0 = time.monotonic()
        try:
            payload = run_compile_and_persist(self.ctx, is_finish=freeze)
        except BaseException as error:
            # The number was consumed; without a record the ledger has a gap
            # and neither the count nor the highest number is the truth.
            self.trace(
                "compile_attempt", attempt=self.ctx.n_compile_attempts, ok=False,
                build_error=type(error).__name__, verifier_summary=[],
                is_finish=freeze, elapsed_ms=int((time.monotonic() - t0) * 1000),
                error=str(error))
            raise
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        verifier_summary = [
            {"name": r.get("name"), "verdict": r.get("verdict")}
            for r in (payload.get("verifier_reports") or [])
        ]
        self.trace(
            "compile_attempt",
            attempt=payload.get("attempt", self.ctx.n_compile_attempts),
            ok=payload.get("ok", False),
            build_error=(payload.get("build_error") or {}).get("code"),
            verifier_summary=verifier_summary,
            is_finish=freeze,
            elapsed_ms=elapsed_ms,
        )
        if freeze and payload.get("ok") and not was_committed:
            ir_hash = (self.ctx.last_successful_ir or {}).get("id")
            self.trace("commit", ir_hash=ir_hash)
            # `deliberate_deviations` is flushed onto meta.json by the layer
            # meta hook (_object/_scene_layer_meta), which finalize_meta
            # merges on every terminal write — committed OR failed — so it
            # needs no explicit pass here. (An empty list still means
            # "committed cleanly, no punts".)
            self.finalize_meta("committed")
        return _slim_compile_payload_for_llm(payload)


# The name the package and the daemon import (`layers/__init__.py`, `layers/server.py`).
Session = LayerSession


__all__ = ["LayerSession", "Session", "SessionConfig"]
