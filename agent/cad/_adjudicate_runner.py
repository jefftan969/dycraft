#!/usr/bin/env python3
"""Standalone exact-collision adjudicator (child process).

Answers, for a batch of already-posed part pairs, whether they *really* touch.
Invoked by ``verify/_exact_adjudicator.py`` as::

    <b123d-python> _adjudicate_runner.py --job JOB.json --result RESULT.json

**Why a batch, and why only some pairs.** The OBB/SAT phase in
``sdk/_object_kernel.py`` bounds a solid by its AABB, so it over-reports: a rod
sliding cleanly through a tube's bore, or a door swinging through a shelled
carcass's open front, both come back as deep penetrations. Only the pairs SAT already
flagged are sent here — typically 0–30 per compile — so exact B-rep queries at
1–100 ms each are affordable, whereas using them as the primary narrow phase
(hundreds to tens of thousands of calls per compile) would not be.

**This file is deliberately self-contained**: standard library plus build123d /
OCP only. It must NOT import anything from the ``dycraft`` package, because the
CAD interpreter is a different environment where dycraft is not installed.

Geometry sources per side of a pair:
  * ``{"brep": "<abs path>"}`` — a ``solid`` part's exact B-rep.
  * ``{"primitive": "box", "params": {...}}`` — rebuilt analytically.
    ``Box(width, length, height)`` gives X/Y/Z extents about the origin,
    matching the DyCraft convention 1:1 (verified). Rebuilding rather than
    skipping is what lets a solid-vs-box pair be adjudicated at all, and it is
    exact — not a proxy.

Verdicts: ``CLEAR`` (positive distance), ``CONTACT`` (touching within
tolerance, no material shared), ``PENETRATING`` (shared volume above
``vol_eps``). Only ``CLEAR``/``CONTACT`` let the caller drop a SAT report, which
is why the two are distinguished rather than merged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _load_shape(spec, cache: dict):
    """Build one side of a pair, memoized on its geometry identity."""
    from build123d import Box, Compound, Solid
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopoDS import TopoDS_Shape

    brep = spec.get("brep")
    if brep:
        key = ("brep", brep)
        if key in cache:
            return cache[key]
        sh = TopoDS_Shape()
        if not BRepTools.Read_s(sh, brep, BRep_Builder()) or sh.IsNull():
            raise RuntimeError(f"unreadable BREP: {brep}")
        # `Shape.cast()` returns None for a TopAbs_COMPOUND, and every boolean
        # result serializes as one — dispatch on the topology type instead.
        out = (Solid(sh) if sh.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID
               else Compound(sh))
        cache[key] = out
        return out

    prim = spec.get("primitive")
    p = spec.get("params") or {}
    key = ("prim", prim, tuple(sorted((k, float(v)) for k, v in p.items())))
    if key in cache:
        return cache[key]
    if prim == "box":
        out = Box(float(p["width"]), float(p["length"]), float(p["height"]))
    else:
        raise RuntimeError(f"cannot rebuild geometry for primitive {prim!r}")
    cache[key] = out
    return out


def _place(shape, se3):
    """Apply a DyCraft SE3 ``((tx,ty,tz), (qw,qx,qy,qz))`` to a shape.

    Note the quaternion component order: DyCraft stores ``(w, x, y, z)`` while
    ``gp_Quaternion`` takes ``(x, y, z, w)``. Getting this wrong produces a
    plausible-looking but systematically rotated answer, which is the single
    easiest way to make this whole layer quietly wrong.

    **`moved`, never `located`.** A staged ``.brep`` can carry a non-identity
    ``TopoDS_Shape`` location of its own: authoring ``Pos(...) * Cylinder(...)``
    yields a *located* shape, and ``BRepTools.Write_s`` preserves that location
    rather than baking it into the vertices. ``located()`` **replaces** a
    shape's location and would therefore discard it; ``moved()`` composes with
    it, which is what placing part-local geometry into the object frame means.

    Using `located` here put every such part at the wrong pose — off by exactly
    the location that was thrown away — and produced spurious shared volume. In
    the run history that reported a revolving door's guide panes as overlapping
    the facade by 3372 cm3 when the true intersection is empty; the author
    checked it by hand, correctly concluded the harness was wrong, and
    permanently suppressed four collision checks with `allow_collision` to get
    past it. Because this layer may only downgrade a SAT verdict, the same fault
    in the other direction is worse: a part displaced away from a genuine
    overlap gets silently cleared.
    """
    from build123d import Location
    from OCP.gp import gp_Quaternion, gp_Trsf, gp_Vec

    (tx, ty, tz), (qw, qx, qy, qz) = se3
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Quaternion(qx, qy, qz, qw))
    trsf.SetTranslationPart(gp_Vec(float(tx), float(ty), float(tz)))
    return shape.moved(Location(trsf))


#: Bisection steps spent looking for a translation that ends an overlap.
#:
#: Each step is one B-rep boolean, ~5 ms measured, so eight is ~40 ms — and it
#: is only spent on a pair already confirmed penetrating, i.e. on a compile
#: that is going to fail anyway. Eight halvings resolve the bracket to 0.4% of
#: the part's own extent along the escape axis, which is finer than any
#: clearance an author would author by hand.
_SHIFT_BISECT_STEPS = 8


def _separating_translation(a, b, common_bbox, vol_eps: float):
    """A translation of `b` that was TESTED to end the overlap, or None.

    SAT's `depth` times `contact_normal` is not a usable repair: SAT measures
    depth on the *enclosing boxes*, so on anything rotated or concave it
    overstates, and a part moved by the full box depth leaves the overlap and
    lands somewhere visibly wrong, costing a repair cycle.

    So this measures instead of inferring. The direction is the **thinnest axis
    of the shared solid**, which is the cheapest way out of it: the two bodies
    interpenetrate least along that axis by construction. The distance is
    bisected on the exact boolean, so the number that comes back has been
    checked — at that translation the intersection really is empty.

    Returning None is a real answer, not a failure: it means no single-axis
    translation cleared the pair within the bracket, so the geometry has to
    change rather than move. It is the common outcome when the escape
    direction is a rotated face normal rather than a world axis.
    """
    if not common_bbox:
        return None
    try:
        (lo, hi) = common_bbox
        extents = [float(hi[i]) - float(lo[i]) for i in range(3)]
        axis = min(range(3), key=lambda i: extents[i])
        # The shared solid's extent along the escape axis is the lower bound on
        # the shift; twice it is a generous upper bound, since past that the
        # two boxes cannot still overlap along that axis.
        bound = extents[axis] * 2.0
        if not (bound > 0.0):
            return None

        def clear_at(distance: float) -> bool:
            from build123d import Location, Pos
            delta = [0.0, 0.0, 0.0]
            delta[axis] = distance
            moved = b.moved(Location(Pos(*delta)))
            inter = a.intersect(moved)
            if inter is None:
                return True
            return sum(float(s.volume) for s in inter.solids()) <= vol_eps

        if not clear_at(bound):
            # Not reachable along this axis inside the bracket. Say nothing
            # rather than report a translation that does not work.
            return None
        low, high = 0.0, bound
        for _ in range(_SHIFT_BISECT_STEPS):
            mid = 0.5 * (low + high)
            if clear_at(mid):
                high = mid
            else:
                low = mid
        out = [0.0, 0.0, 0.0]
        out[axis] = high
        return out
    except BaseException:  # noqa: BLE001 — evidence only; the verdict stands
        return None


def _answer(a, b, tol: float, vol_eps: float) -> dict:
    """Shared volume decides; distance is only evidence.

    It is tempting to short-circuit on ``distance > tol`` and call the pair
    clear, but that is WRONG for containment: ``distance_to_with_closest_points``
    measures between the two *boundaries*, so a part swallowed whole inside
    another reports a comfortably positive distance while sharing all of its
    volume — the worst penetration there is, silently cleared. So the boolean
    runs first, unconditionally. It costs nothing on the common disjoint case
    (OCCT rejects on bounding boxes, ~1 ms) and is the only test that cannot be
    fooled by topology.
    """
    try:
        inter = a.intersect(b)
        vol = 0.0
        if inter is not None:
            vol = sum(float(s.volume) for s in inter.solids())
    except Exception:  # noqa: BLE001 — a failed boolean must not lose the pair
        # UNKNOWN rather than CLEAR: refusing to adjudicate leaves the
        # conservative SAT verdict standing, which is the safe direction.
        return {"verdict": "UNKNOWN", "distance": None, "common_volume": None}

    rec = {"common_volume": vol}
    # The intersection solid, bounded. Callers that ask "how far past this face
    # is the part" get their answer from the shared material itself rather than
    # from either input's bounding box — which is the whole point, since an
    # input's box can be inflated by rotation while the shared solid cannot.
    # Reported only when there is material to bound; a failure here is evidence
    # lost, not a verdict changed.
    # `intersect` hands back a ``ShapeList``, which has ``solids()`` but no
    # ``bounding_box()`` — so union the per-solid bounds rather than asking the
    # list. The error is reported rather than swallowed: a silent miss here
    # degrades a measured depth back to a bounding-box bound, and the caller
    # cannot tell that from "nothing was measurable".
    if vol > 0.0 and inter is not None:
        try:
            bbs = [s.bounding_box() for s in inter.solids()]
            if bbs:
                rec["common_bbox"] = [
                    [min(b.min.X for b in bbs), min(b.min.Y for b in bbs),
                     min(b.min.Z for b in bbs)],
                    [max(b.max.X for b in bbs), max(b.max.Y for b in bbs),
                     max(b.max.Z for b in bbs)]]
        except BaseException as e:  # noqa: BLE001
            rec["common_bbox_error"] = f"{type(e).__name__}: {e}"
    try:
        dist, pa, pb = a.distance_to_with_closest_points(b)
        rec["distance"] = float(dist)
        rec["closest_a"] = [pa.X, pa.Y, pa.Z]
        rec["closest_b"] = [pb.X, pb.Y, pb.Z]
    except Exception:  # noqa: BLE001 — evidence only; the verdict stands without it
        dist = None
        rec["distance"] = None

    if vol > vol_eps:
        rec["verdict"] = "PENETRATING"
        shift = _separating_translation(a, b, rec.get("common_bbox"), vol_eps)
        if shift is not None:
            rec["separating_shift"] = shift
    elif dist is not None and dist <= tol:
        # Boundaries meet but share no material: flush contact, not a defect.
        rec["verdict"] = "CONTACT"
    else:
        rec["verdict"] = "CLEAR"
    return rec


def _check_version() -> str | None:
    """Refuse a skewed CAD interpreter. Returns an error message, or None.

    Loads the sibling `_b123d_version` by path rather than importing dycraft:
    this file runs in an interpreter where dycraft need not be installed. That
    module is stdlib-only and holds nothing but the pin, which is exactly why
    the pin lives there instead of in `_solid_compile`.

    A version skew matters more here than almost anywhere: this process decides
    whether reported collisions are real, so a subtly different kernel would
    silently change verdicts rather than fail.
    """
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import _b123d_version as _bv
        _bv.check_build123d_version()
        return None
    except Exception as e:  # noqa: BLE001
        return getattr(e, "message", None) or f"{type(e).__name__}: {e}"


def _adjudicate(job: dict) -> dict:
    """Answer one batch. The single adjudication path for both modes.

    The shape cache is built fresh here, per batch, and that is load-bearing
    rather than incidental: it is keyed on a geometry *spec*, and a spec names a
    B-rep by path. Those paths are rewritten by each compile, so a cache shared
    across batches would answer a later compile with an earlier compile's
    geometry — a wrong collision verdict with nothing to indicate it.
    """
    tol = float(job.get("tol", 1e-9))
    vol_eps = float(job.get("vol_eps", 1e-12))
    cache: dict = {}
    results = []
    for q in job.get("queries", []):
        rec = {"tag": q.get("tag")}
        try:
            a = _place(_load_shape(q["a"], cache), q["a_se3"])
            b = _place(_load_shape(q["b"], cache), q["b_se3"])
            rec.update(_answer(a, b, tol, vol_eps))
        except BaseException as e:  # noqa: BLE001 — one bad pair must not sink the batch
            rec.update({"verdict": "UNKNOWN",
                        "error": f"{type(e).__name__}: {e}"})
        results.append(rec)
    return {"ok": True, "results": results}


def _mismatch_payload(mismatch: str) -> dict:
    return {"ok": False, "error_kind": "build123d_version_mismatch",
            "message": mismatch}


def _run(job_path: str, result_path: str) -> int:
    mismatch = _check_version()
    if mismatch:
        Path(result_path).write_text(json.dumps(_mismatch_payload(mismatch)),
                                     encoding="utf-8")
        return 1
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    Path(result_path).write_text(json.dumps(_adjudicate(job)), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Serve mode
# ---------------------------------------------------------------------------
#
# A cold adjudicator would cost a full CAD kernel start per compile, on top of
# the authoring child's, and a production compile runs both. `_worker_pool`
# keeps warm children for either, driven by the wire contract implemented below:
# `--serve --stdout-spool PATH`, one JSON job per line on stdin, one JSON ack
# per job on a private duplicate of the original stdout.
#
# The framing is written out here rather than shared with the pool because this
# file must stay importable in an environment where dycraft does not exist (see
# the module docstring). Duplicating ~30 lines is the price of that guarantee.

def _offsets() -> "tuple[int, int]":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    def _pos(fd: int) -> int:
        try:
            return os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            return 0
    return _pos(1), _pos(2)


def _serve(stdout_spool: str) -> int:
    """Answer batches until stdin reaches EOF."""
    proto_fd = os.dup(1)
    os.set_inheritable(proto_fd, False)
    sys.stdout.flush()
    spool_fd = os.open(stdout_spool, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(spool_fd, 1)
    os.close(spool_fd)

    def ack(payload: dict) -> None:
        os.write(proto_fd, (json.dumps(payload) + "\n").encode())

    # Settled once: the interpreter cannot change under a running process, and
    # re-checking per batch would reintroduce the cost this mode removes.
    mismatch = _check_version()
    if not mismatch:
        # Force the kernel import now so the child is genuinely warm before the
        # parent sends anything.
        try:
            _load_shape({"primitive": "box",
                         "params": {"width": 0.001, "length": 0.001,
                                    "height": 0.001}}, {})
        except BaseException:  # noqa: BLE001 — warming is best-effort
            pass
    ack({"status": "ready"})

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        out_start, err_start = _offsets()
        try:
            job = json.loads(line)
            payload = (_mismatch_payload(mismatch) if mismatch
                       else _adjudicate(job))
            Path(job["result"]).write_text(json.dumps(payload),
                                           encoding="utf-8")
            status = "ok"
        except BaseException as e:  # noqa: BLE001 — never propagate out
            status = "runner_internal"
            try:
                Path(json.loads(line)["result"]).write_text(json.dumps({
                    "ok": False, "error_kind": "runner_internal",
                    "message": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }), encoding="utf-8")
            except BaseException:  # noqa: BLE001
                pass
        out_end, err_end = _offsets()
        try:
            ack({"status": status,
                 "stdout_span": [out_start, out_end],
                 "stderr_span": [err_start, err_end]})
        except OSError:
            return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dycraft exact-collision adjudicator")
    parser.add_argument("--job")
    parser.add_argument("--result")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--stdout-spool")
    args = parser.parse_args(argv)

    if args.serve:
        if not args.stdout_spool:
            parser.error("--serve requires --stdout-spool")
        try:
            return _serve(args.stdout_spool)
        except BaseException:  # noqa: BLE001 — die quietly; the parent notices
            traceback.print_exc()
            return 1

    if not args.job or not args.result:
        parser.error("one-shot mode requires --job and --result")
    try:
        return _run(args.job, args.result)
    except BaseException as e:  # noqa: BLE001 — always emit a structured result
        try:
            Path(args.result).write_text(json.dumps({
                "ok": False,
                "error_kind": "runner_internal",
                "message": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }), encoding="utf-8")
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
