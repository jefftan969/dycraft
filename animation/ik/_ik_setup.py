"""Setup the `h.ik` child runs before the authored code (`_sim_runner.py --setup`).

Environment shims the solver's dependencies need on some machines. Kept here,
beside the op, rather than in the runner: the runner is engine-agnostic and
names no dependency, and a setup prepended to the authored code would shift its
traceback line numbers. Stdlib at module scope; anything else is imported
inside a shim and only when the interpreter has it.
"""

from __future__ import annotations


class Unavailable(RuntimeError):
    """Refuses the call: the runner reports it as `error_kind` and runs nothing."""
    error_kind = "unavailable"


def _require_gpu() -> None:
    """Refuse the call when torch sees no CUDA device.

    The launcher already leaves IK out when no GPU is visible, so this is the
    backstop for a GPU hidden or lost after launch (a driver update, a job
    moved to a CPU node). Without it every call fails seconds later with
    cuRobo's "No CUDA GPUs are available" traceback instead of saying that IK
    is unavailable. Only when the interpreter has torch.
    """
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    if not torch.cuda.is_available():
        raise Unavailable("torch sees no CUDA device")


def _pre_volta_warp() -> None:
    """Use Warp's own matmul and Cholesky below Volta.

    Warp's tile matmul and Cholesky go through cuBLASDx (libmathdx), which has
    no kernels before Volta: on a GTX 1070 (sm_61), cuRobo's first IK solve
    fails with "libmathdx cuBLASDx error" while Warp builds the LM step. Warp's
    scalar fallbacks (`enable_mathdx_gemm`, `enable_mathdx_solver`) solve the
    same IK there, measured through `h.ik`'s own path (0.0 mm error). From
    sm_70 up this changes nothing. Only when the interpreter already has warp
    and torch.
    """
    import importlib.util
    if importlib.util.find_spec("warp") is None or importlib.util.find_spec("torch") is None:
        return
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) >= (7, 0):
            return
        import warp
        warp.config.enable_mathdx_gemm = False
        warp.config.enable_mathdx_solver = False
    except Exception:  # noqa: BLE001 -- a probe that fails leaves Warp as it was
        return


_require_gpu()
_pre_volta_warp()
