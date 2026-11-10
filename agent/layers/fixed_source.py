"""Run a hand-written `object.py` end to end into a persistent run bundle.

    python -m dycraft.agent.layers.fixed_source \
        --source tests/fixtures/m1_live_shape_single.py \
        --run-dir runs/m1_live_shape_single

The agent-free counterpart to a normal object run: same run-directory layout,
same worker, same verifier suite, same renderer and exporter — only the LLM is
missing, because the source is supplied.

**Why this exists.** It pins the pipeline without an LLM in it. A failure
here is a harness or geometry failure, never a prompting one, and it produces a
real run bundle to look at rather than an assertion that passed inside a temp
directory. It is also how a specific source can be replayed across revisions,
which a prompt-driven run cannot do.

What it writes under `--run-dir`:

    object.py                 the source, verbatim
    ir.json                   the ObjectIR
    assets/{brep,meshes,src}/ the staged artifacts
    compile_reports/0.json    verifier verdicts (+ latest.json)
    final_render/             6 canonical views + URDF + localized meshes
    poses/<label>/            renders at named joint configurations
    worker_result.json        the worker's raw payload, minus the IR
    worker_stdout.log         the child's stdio
    worker_stderr.log
    meta.json                 outcome, provenance, timings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


class _RunCtx:
    """The slice of ToolContext the verifiers and renderer actually read."""

    def __init__(self, run_dir, layer_cfg):
        self.run_dir = Path(run_dir)
        self.layer_cfg = layer_cfg
        self.step = 0
        # Run-lifetime cache the exact-collision adjudicator uses to avoid
        # re-asking the CAD kernel the same question across CCD samples.
        self.adjudication_cache: dict = {}
        self.render_cache: dict = {}
        self.user_prompt = "fixed-source integration run"
        self.layer_payload = None
        self.deliberate_deviations: list = []


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_poses(ctx, ir, run_dir: Path, poses_arg: str) -> list:
    """Render the object at named joint configurations.

    `--poses 'closed:door_hinge=0;open:door_hinge=75'` — the articulated case's
    proof that FK actually moves geometry, which a single rest-pose render
    cannot show.

    Revolute values are DEGREES, per the frame contract's "degrees at every
    surface, radians in the IR and kernel" rule; the
    kernel is handed radians below. Prismatic values are meters and pass
    through. Without the conversion `door_hinge=75` above renders 75 *radians*
    (= 4297 deg) and the posed render silently shows the wrong configuration.
    """
    import math

    from ...render import RenderRequest, RenderUnavailable, render_request

    out = []
    for spec in [s for s in poses_arg.split(";") if s.strip()]:
        label, _, assignments = spec.partition(":")
        label = label.strip() or "pose"
        joint_types = {j.get("name"): j.get("type") for j in (ir.get("joints") or [])}
        poses = {}
        authored = {}
        for kv in assignments.split(","):
            if "=" in kv:
                joint, _, value = kv.partition("=")
                joint = joint.strip()
                v = float(value)
                authored[joint] = v
                poses[joint] = (math.radians(v) if joint_types.get(joint) == "revolute"
                                else v)
        out_dir = run_dir / "poses" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = render_request(RenderRequest(
                layer="object", ir=ir, catalog=None, target="all",
                out_dir=out_dir, step=0, cache=ctx.render_cache,
                poses=poses, pose_label=label,
                asset_base_dir=run_dir,
            ))
            paths = [str(p) for p in result.ordered_paths()]
        except (RenderUnavailable, ValueError, KeyError) as e:
            paths = []
            _write(out_dir / "error.txt", f"{type(e).__name__}: {e}\n")
        out.append({"label": label, "poses_deg": authored,
                    "poses_rad": poses, "paths": paths})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dycraft.agent.layers.fixed_source")
    ap.add_argument("--source", required=True,
                    help="path to a hand-written object.py")
    ap.add_argument("--run-dir", required=True,
                    help="persistent run bundle directory (created if absent)")
    ap.add_argument("--poses", default="",
                    help="extra renders, e.g. "
                         "'closed:hinge=0;open:hinge=75'")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--skip-render", action="store_true",
                    help="skip the renderer (useful where no GL is available)")
    a = ap.parse_args(argv)

    from ..cad import object_worker
    from .layer import object_layer_config

    source_path = Path(a.source).resolve()
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding="utf-8")

    meta = {
        "runner": "fixed_source",
        "source_path": str(source_path),
        "run_dir": str(run_dir),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worker_python": object_worker.b123d_python(),
        "driver_python": sys.executable,
    }

    # ---- 1. execute object.py in the worker -------------------------------
    t0 = time.time()
    result = object_worker.run_object_source(
        source, run_dir=run_dir, timeout=a.timeout)
    meta["worker_seconds"] = round(time.time() - t0, 2)

    _write(run_dir / "worker_stdout.log", result.pop("worker_stdout", "") or "")
    _write(run_dir / "worker_stderr.log", result.pop("worker_stderr", "") or "")
    _write(run_dir / "worker_result.json",
           json.dumps({k: v for k, v in result.items() if k != "ir"},
                      indent=2, default=str))

    if not result.get("ok"):
        meta["outcome"] = "build_failed"
        err = result.get("build_error") or result.get("sandbox_violation") or {}
        meta["error"] = err.get("summary") or result.get("message")
        _write(run_dir / "meta.json", json.dumps(meta, indent=2, default=str))
        print(f"[fixed_source] BUILD FAILED: {meta['error']}")
        print(f"[fixed_source] bundle -> {run_dir}")
        return 1

    ir = result["ir"]
    _write(run_dir / "ir.json", json.dumps(ir, indent=2))
    meta["ir_id"] = ir["id"]
    meta["n_parts"] = len(ir.get("parts") or [])
    meta["n_joints"] = len(ir.get("joints") or [])
    print(f"[fixed_source] built {meta['n_parts']} part(s), "
          f"{meta['n_joints']} joint(s); ir={ir['id'][:12]}")

    # ---- 2. verifiers ------------------------------------------------------
    from .compile import _run_verifiers
    layer_cfg = object_layer_config()
    ctx = _RunCtx(run_dir, layer_cfg)
    t0 = time.time()
    reports = _run_verifiers(layer_cfg, ir, ctx, None)
    meta["verify_seconds"] = round(time.time() - t0, 2)
    report_dicts = [r.to_dict() for r in reports]
    payload = {"step": 0, "ok": all(r.verdict != "fail" for r in reports),
               "ir_hash": ir["id"], "verifiers": report_dicts}
    _write(run_dir / "compile_reports" / "0.json",
           json.dumps(payload, indent=2, default=str))
    _write(run_dir / "compile_reports" / "latest.json",
           json.dumps(payload, indent=2, default=str))
    meta["verifiers"] = {r.name: r.verdict for r in reports}
    meta["verifiers_ok"] = payload["ok"]
    for r in reports:
        print(f"[fixed_source]   {r.verdict:9} {r.name}")

    # ---- 3. renders + URDF -------------------------------------------------
    if not a.skip_render:
        from ...render import finalize_render_result
        t0 = time.time()
        try:
            render = finalize_render_result(ir, run_dir, layer="object")
            meta["final_render"] = {k: v for k, v in render.items()
                                    if k in ("ok", "out_dir", "note")}
            meta["final_render_paths"] = [str(p) for p in
                                          (render.get("paths") or [])]
        except Exception as e:  # noqa: BLE001 — a render failure is data
            meta["final_render"] = {"ok": False,
                                    "error": f"{type(e).__name__}: {e}"}
        meta["render_seconds"] = round(time.time() - t0, 2)

        if a.poses:
            meta["pose_renders"] = _render_poses(ctx, ir, run_dir, a.poses)

    meta["outcome"] = "committed" if meta.get("verifiers_ok") else "verify_failed"
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(run_dir / "meta.json", json.dumps(meta, indent=2, default=str))
    print(f"[fixed_source] outcome={meta['outcome']}")
    print(f"[fixed_source] bundle -> {run_dir}")
    return 0 if meta.get("verifiers_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
