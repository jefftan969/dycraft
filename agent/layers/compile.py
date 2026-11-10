"""compile.py — load the source file, run it in the sandbox, run the
verifier suite.

Layer-parameterized. The shape returned here is the JSON-serializable
payload that the agent harness writes to compile_reports/<step>.json and
that the `compile` / `finish` tools emit as their tool result.
"""

from __future__ import annotations

import builtins
import traceback
from typing import Any

from ...sdk import _mesh_assets
from ...sdk import _object_kernel
from ...sdk._errors import (BuildError, GeometryKernelError, SandboxViolation,
                            SourceRuntimeError, SourceTimeout, _Structured)
from ..cad import _geometry_qc
from ..cad import _timelimit
from ..cad import sandbox as _sandbox
from ..cad.sandbox import _src_line


# Frames from these packages mean the kernel refused the operation, not that
# the program is malformed. Matched on the traceback's file paths.
_KERNEL_PACKAGES = ("build123d", "OCP", "ocp_", "cadquery")


def _failing_source_line(exc: BaseException, source: str,
                         source_filename: str) -> tuple[int, str]:
    """The line of the *agent's own file* that failed, not the deepest frame.

    A fillet failure's deepest frame is inside build123d; what the agent needs
    is which of its own lines called it.
    """
    lineno = 0
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == source_filename:
            lineno = frame.lineno or 0
    return lineno, _src_line(source, lineno)


def _with_source_location(payload: dict, exc: BaseException, source: str,
                          source_filename: str) -> dict:
    """Add the author's failing line to a structured error's evidence.

    The SDK's own errors are raised deep inside a builder method, so they know
    the *entity* at fault (`refs`) and what to do about it
    (`suggested_repairs`) but not the line that called them. Every raw exception
    got a line number from `_classify_exec_failure` and every structured error
    did not, which across the run history meant 19 of 23 build errors told the
    author what was wrong and left them to find where — on a 400-line
    `object.py` with a dozen similar `assert_distance` calls, that is the
    expensive half of the answer.

    An existing `lineno` is never overwritten: a `SandboxViolation` from the AST
    pass already carries the offending node's position, which is more precise
    than anything a traceback can say.
    """
    evidence = dict(payload.get("evidence") or {})
    if not evidence.get("lineno"):
        lineno, src_line = _failing_source_line(exc, source, source_filename)
        if lineno:
            evidence["lineno"] = lineno
            evidence["src_line"] = src_line
            payload["evidence"] = evidence
    return payload


def _timeout_error(source_filename: str, seconds: float) -> SourceTimeout:
    return SourceTimeout(
        f"{source_filename} did not finish within {seconds:g}s",
        refs=[],
        evidence={"reason": "wall_clock_limit", "limit_s": seconds,
                  "lineno": 0, "src_line": ""},
        suggested_repairs=[
            "Inspect where execution stalled: a nonterminating loop, expensive "
            "CAD operation or tessellation can exhaust the source time limit. "
            "Isolate that work before simplifying geometry.",
        ],
    )


def _classify_exec_failure(exc: BaseException, source: str,
                           source_filename: str):
    """Sort an exception escaping the agent's source into the right class.

    Not everything is a `SandboxViolation`: its repair, "fix the source so it
    raises only BuildError or runs cleanly", is advice the sandbox itself makes
    impossible, since `raise` and `try` are not in the allowlist, and most such
    exceptions are not sandbox events at all but typos (`NameError`) and OCCT
    failures (a fillet).

    Both classes are `BuildError` subclasses, so they land in `build_error`
    alongside the SDK's own validation failures — where an agent already knows
    to look for something it can fix by editing geometry.
    """
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except BaseException:  # noqa: BLE001 — the error path must not raise
        tb = f"<traceback unavailable for {type(exc).__name__}>"
    if len(tb) > 4096:
        tb = tb[:4096] + "...<truncated>"
    try:
        detail = str(exc)
    except BaseException:  # noqa: BLE001 — a __str__ that raises must not win
        detail = "<exception message unavailable>"
    lineno, src_line = _failing_source_line(exc, source, source_filename)  # noqa: E501
    kernel = any(
        any(pkg in (frame.filename or "") for pkg in _KERNEL_PACKAGES)
        for frame in traceback.extract_tb(exc.__traceback__)
    )
    evidence = {"exception": type(exc).__name__, "lineno": lineno,
                "src_line": src_line, "traceback": tb}

    # A known kernel message translates into the move that resolves it, which is
    # more actionable than either the raw text or generic advice, so it leads.
    # `failure_hint` returns None for anything it does not recognize.
    hint = _geometry_qc.failure_hint(f"{type(exc).__name__}: {detail}")

    if kernel:
        return GeometryKernelError(
            f"the CAD kernel could not complete this operation: "
            f"{type(exc).__name__}: {detail}",
            refs=[],
            evidence={**evidence, "reason": "kernel_operation_failed"},
            # The kernel's own text is actionable in its own right — build123d
            # says things like "try a smaller value or use max_fillet()" — and
            # would otherwise stay buried in the traceback.
            suggested_repairs=([hint] if hint else []) + [
                detail,
                f"the failing call is on line {lineno} of {source_filename}"
                if lineno else
                "check which operation on the failing shape is out of range",
                "Check the failing operation and its operands: radius and adjacent "
                "faces, loft section compatibility, or boolean overlap. Use the "
                "diagnostic above to choose the change.",
            ],
        )

    return SourceRuntimeError(
        f"{source_filename} raised {type(exc).__name__}: {detail}",
        refs=[],
        evidence={**evidence, "reason": "source_raised"},
        suggested_repairs=([hint] if hint else []) + [
            detail,
            f"fix line {lineno} of {source_filename}" if lineno else
            "fix the statement that raised",
        ],
    )


def _ir_not_produced(reason: str) -> BuildError:
    return BuildError(
        f"source did not produce a valid `ir` at module top level: {reason}",
        refs=[],
        evidence={"reason": reason},
        suggested_repairs=[
            "assign the finalize() result to a top-level name: ir = builder.finalize()",
        ],
    )


def build_ir_with_layer(
    source: str,
    *,
    layer_cfg,
    ctx: Any,
) -> dict:
    """Execute `source` under the layer's sandbox and return either the IR
    or a structured error. Does NOT run the verifier suite.

    Returned dict keys:
        ok                : bool — true iff the source built a valid IR
        ir                : dict | None
        build_error       : dict | None — StructuredError.to_dict()
        sandbox_violation : dict | None

    This is the build-only half of `compile_with_layer`. `probe` and
    `snapshot` build through it on every call so they operate on the
    CURRENT source rather than the last-successfully-compiled IR.
    """
    out = {"ok": False, "ir": None, "build_error": None, "sandbox_violation": None}

    # Object and scene source always executes under the AST allowlist.
    try:
        _sandbox.check_ast(source, mode="exec")
    except SandboxViolation as e:
        out["sandbox_violation"] = e.to_dict()
        return out

    # 2. Reset SDK singleton(s) + build globals.
    layer_cfg.singleton_reset()
    overlay = layer_cfg.build_overlay_factory(ctx)
    # Hand the staging session the bytes we are about to execute.
    #
    # `finalize` stages the authored program as the object's `source` asset, and
    # this is the only place that holds the exact text being run. Without it the
    # builder re-read `<run_dir>/object.py` at finalize — a second read of a
    # mutable file, which makes a provenance artifact depend on the file's state
    # afterwards rather than on the execution, and silently substitutes a
    # placeholder comment when that file cannot be read. Same argument as
    # `measure_shape` serializing the B-rep eagerly instead of holding the Shape.
    _session = _mesh_assets.get_active_mesh_session()
    if _session is not None:
        _session.source_text = source
        # Whether that text is one object's own recipe, or a program that
        # happens to build several. See `LayerConfig.entry_source_is_object_program`.
        _session.source_owns_object = layer_cfg.entry_source_is_object_program
    g = _sandbox.build_globals(overlay=overlay)
    # 3. Exec, under a wall-clock bound. The object layer reaches this inside
    # the CAD worker, which subprocess-times-out anyway; scene builds reach it
    # inside the long-lived daemon, where an unbounded loop would otherwise hold
    # the connection open forever.
    limit = _timelimit.resolve_limit(scene=layer_cfg.name == "scene")
    try:
        with _timelimit.time_limit(limit):
            code = builtins.compile(source, layer_cfg.source_filename, "exec")
            exec(code, g)
    except _timelimit.SandboxTimeout:
        out["build_error"] = _timeout_error(
            layer_cfg.source_filename, limit).to_dict()
        return out
    except KeyboardInterrupt:
        # Operator intent. Absorbing this into a build error made Ctrl-C during
        # an in-process compile do nothing at all.
        raise
    except SystemExit as e:
        # Not a bug in the program — an attempt to end the host process.
        # `SystemExit` does not come from the source raising it: `exit` and
        # `quit` are refused names, so reaching this means something the
        # source *called* tried to exit, and that is a policy breach.
        out["sandbox_violation"] = SandboxViolation(
            "the source attempted to exit the process",
            refs=[],
            evidence={"node_type": "SystemExit", "lineno": 0, "col": 0,
                      "src_line": "", "code": repr(getattr(e, "code", None))},
            suggested_repairs=[
                f"{layer_cfg.source_filename} must assign `ir` and return normally; "
                f"do not terminate the interpreter",
            ],
        ).to_dict()
        return out
    except BuildError as e:
        out["build_error"] = _with_source_location(
            e.to_dict(), e, source, layer_cfg.source_filename)
        return out
    except SandboxViolation as e:
        out["sandbox_violation"] = _with_source_location(
            e.to_dict(), e, source, layer_cfg.source_filename)
        return out
    except _Structured as e:
        out["build_error"] = _with_source_location(
            e.to_dict(), e, source, layer_cfg.source_filename)
        return out
    except BaseException as e:  # noqa: BLE001 — sandbox boundary
        # A source bug or a kernel refusal, not a policy breach. Both are
        # BuildError subclasses, so they surface where an agent already looks
        # for something it can fix.
        out["build_error"] = _classify_exec_failure(
            e, source, layer_cfg.source_filename).to_dict()
        return out
    finally:
        # Per-part compile diagnostics, on every path the source ran: a build
        # that failed at part three is exactly when the warnings from parts one
        # and two are worth reading. `finally` rather than a line after the
        # block, because each handler above returns.
        #
        # These do NOT go in the IR. They describe this compile — vertices
        # welded, a stage that raised, a render mesh coarsened to stay under the
        # triangle budget — and the IR is content-hashed whole, so putting them
        # there would make a diagnostic's prose part of the object's identity.
        # Empty for the scene layer, which installs no staging session.
        _notes = _mesh_assets.drain_geometry_notes()
        if _notes:
            out["geometry_notes"] = _notes
        # The dependency closure, on every path the source ran — and on the
        # paths it did not, where draining an empty ledger is the point.
        # `drain_build_extras` below is success-only and correct to be; this
        # is not, because a FAILED build's dependencies are what the author is
        # about to edit. Without it `_cached_build_if_fresh` sees no deps on a
        # cached failure and serves it until the entry source itself changes.
        # In the `finally` for the same reason the notes are: every handler
        # above returns.
        if layer_cfg.drain_build_deps is not None:
            out.update(layer_cfg.drain_build_deps())

    # 4. Require top-level `ir` with the right schema + id.
    ir = g.get("ir")
    if ir is None:
        out["build_error"] = _ir_not_produced("name 'ir' not defined").to_dict()
        return out
    if not isinstance(ir, dict):
        out["build_error"] = _ir_not_produced(
            f"'ir' must be a dict, got {type(ir).__name__}"
        ).to_dict()
        return out
    allowed = layer_cfg.schema_tag
    allowed = (allowed,) if isinstance(allowed, str) else tuple(allowed)
    if ir.get("schema") not in allowed:
        expected = repr(allowed[0]) if len(allowed) == 1 else repr(allowed)
        out["build_error"] = _ir_not_produced(
            f"'ir' schema tag mismatch "
            f"(got {ir.get('schema')!r}, expected {expected})"
        ).to_dict()
        return out
    if "id" not in ir:
        out["build_error"] = _ir_not_produced("'ir' missing content-hash id").to_dict()
        return out

    # Semantic-anchor provenance is checked ONCE here, where an ObjectIR
    # enters the system — not inside FK, which runs per pose across a CCD
    # sweep and would repeat a whole-object structural check thousands of
    # times for an answer that cannot change. A no-op for objects with no
    # semantic anchors.
    try:
        _object_kernel.validate_anchor_provenance(ir)
    except _object_kernel.AnchorProvenanceMismatch as e:
        out["build_error"] = _ir_not_produced(str(e)).to_dict()
        return out

    # The object definitions this build produced, for a layer that authors
    # them. After the structural checks, so a rejected build contributes
    # nothing, and always drained even when the value is unused — leaving a
    # builder's registry populated would spill this build's objects into the
    # next one's catalog.
    if layer_cfg.drain_build_extras is not None:
        out.update(layer_cfg.drain_build_extras())

    out["ok"] = True
    out["ir"] = ir
    return out


def compile_with_layer(
    source: str,
    *,
    layer_cfg,
    ctx: Any,
    verifier_cache: dict | None = None,
    prebuilt: dict | None = None,
) -> dict:
    """Execute `source` under the layer's sandbox and run the verifier suite.

    Returns a dict with keys:
        ok                : bool — true iff exec succeeded AND every verifier passed
        ir_hash           : str | None
        ir                : dict | None — the finalized IR; only present on full pass
        build_error       : dict | None — StructuredError.to_dict()
        sandbox_violation : dict | None
        verifier_reports  : list[dict]

    `prebuilt` is a `build_ir_with_layer`-shaped result the caller already has.
    It exists because the object layer builds its IR in a separate CAD process
    (`tools._build_ir_routed`), and that build must not be repeated here — the
    subprocess costs seconds, and re-running it in *this* interpreter would
    execute `object.py` without build123d and fail. Verifiers deliberately stay
    in this process either way: only the build moves.
    """
    build_result = (prebuilt if prebuilt is not None
                    else build_ir_with_layer(source, layer_cfg=layer_cfg, ctx=ctx))
    out: dict = {
        "ok": False,
        "ir_hash": None,
        "ir": None,
        "build_error": build_result["build_error"],
        "sandbox_violation": build_result["sandbox_violation"],
        "verifier_reports": [],
    }
    # Per-part compile diagnostics, carried whether or not the build succeeded —
    # a failure at part three is when the warnings from parts one and two matter
    # most. Absent for the scene layer and for a build that never reached the
    # exec. `prebuilt` is how these survive the object layer's process boundary:
    # the worker child put them in its result payload.
    if build_result.get("geometry_notes"):
        out["geometry_notes"] = build_result["geometry_notes"]
    _attach_worker_stdio(out, ctx)
    if not build_result["ok"]:
        return out
    ir = build_result["ir"]

    # 5. Verifier suite.
    reports = _run_verifiers(layer_cfg, ir, ctx, verifier_cache)
    out["verifier_reports"] = [r.to_dict() for r in reports]

    # `uncertain` does not block finish. Only `fail` is a blocker.
    if all(r.verdict != "fail" for r in reports):
        out["ok"] = True
        out["ir_hash"] = ir["id"]
        out["ir"] = ir

    return out


#: How much of the CAD child's stderr to carry. The tail, because OCCT prints
#: progress and the failure is what came last.
_WORKER_STDIO_TAIL = 4000


def _attach_worker_stdio(out: dict, ctx) -> None:
    """Put the CAD child's stderr in the compile payload when it said anything.

    `tools._build_ir_routed` captures the child's stdio onto
    `ctx.last_worker_stdio`, and this is its one reader. The worker's output is
    the channel OCCT uses for the diagnostics it does not raise, so without it
    an author whose build "succeeded" could not see the kernel complaining
    about it, and an author whose worker segfaulted would get an exit status
    with no cause attached.

    Only when non-empty: a clean build prints nothing at all (measured), so
    the field's presence is itself the signal that something spoke.
    """
    stdio = getattr(ctx, "last_worker_stdio", None)
    if not isinstance(stdio, dict):
        return
    payload = {}
    for stream in ("stderr", "stdout"):
        text = (stdio.get(stream) or "").strip()
        if not text:
            continue
        if len(text) > _WORKER_STDIO_TAIL:
            dropped = len(text) - _WORKER_STDIO_TAIL
            text = f"...[{dropped} earlier characters omitted]...\n" \
                   + text[-_WORKER_STDIO_TAIL:]
        payload[stream] = text
    if payload:
        # Named for where it came from, not for what it means: this is the CAD
        # kernel talking, and it is as likely to be a harmless note as the
        # explanation for a failure. `geometry_notes` is DyCraft's own reading
        # of the build; this is unmediated.
        out["worker_stdio"] = payload


def _run_verifiers(layer_cfg, ir, ctx, cache):
    """Run the layer's verifier suite, reusing cached reports for this IR."""
    reports = []
    cache_key_fn = layer_cfg.verifier_cache_key
    for name, fn in layer_cfg.verifiers:
        # CRITICAL: include ir["id"] in the cache key. Without it the cache
        # returns a stale report for any later compile whose IR happens to
        # differ — the agent then sees the same "fail" verdict across edits
        # that materially changed the geometry, and gives up.
        cache_key = (name, ir["id"]) + cache_key_fn(ir["id"], ctx)
        if cache is not None and cache_key in cache:
            report = cache[cache_key]
        else:
            report = fn(ir, ctx)
            if cache is not None:
                cache[cache_key] = report
        reports.append(report)
    return reports
