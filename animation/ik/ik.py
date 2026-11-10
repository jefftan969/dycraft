"""The `h.ik` op: run agent-authored code against a kinematics solver.

The parent half: interpreter resolution, the isolated child, the
`{ok, error_kind, ...}` vocabulary and the provenance triple. The child runner
is `animation/sim/_sim_runner.py` (stdlib-only and engine-agnostic by
construction), run by path.

The world the child loads is written by `animation/blender/solvers.py::
export_state` from the evaluated Blender frame: `animation/sim/scene_export.py`'s
URDF export, plus `ik_meta.json`, the facts cuRobo needs that the URDF cannot
carry: which self-collisions to ignore (joint adjacency and inherited object
allowances at the snapshot pose), and the static obstacles as meshes with world
poses rather than bounding boxes, since a table's AABB is a solid block that
fills the legroom.

Nothing this op returns becomes motion by itself. A solved configuration is a
number the agent reads, judges, and writes into the Blender project.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ... import _home, _interpreters

#: Export subdir in the *run* dir. Its own dir rather than one shared with
#: another consumer of `export_sim_scene`: `ik_meta.json` is cached on the
#: scene hash, so a co-tenant that re-exported per call could rewrite
#: `scene_meta.json` underneath a cached entry.
IK_DIR = "ik"
#: Per-call provenance, the `sim_runs/` convention.
IK_RUNS_DIR = "ik_runs"
#: The child's `checkpoint()` target, written inside the provenance dir.
CHECKPOINT_NAME = "result.partial.json"
#: What `animation/blender/solvers.py::export_state` writes and
#: `dycraft_ik.load_ik_meta` reads.
IK_META = "ik_meta.json"

#: A sibling lookup inside `animation/` — `ik` and `sim` move together.
_RUNNER_PATH = Path(__file__).resolve().parents[1] / "sim" / "_sim_runner.py"
#: Run by the child before the authored code: the solver's environment shims.
_SETUP_PATH = Path(__file__).resolve().with_name("_ik_setup.py")
_HELPER_PATH = Path(__file__).with_name("dycraft_ik.py")
_ROTATION_PATH = Path(__file__).resolve().parents[2] / "sdk" / "_rotation.py"
#: Higher than the sim default: the first call on a robot pays a sphere fit
#: (tens of seconds for a 17-link arm) plus a self-collision matrix (minutes),
#: and only then starts solving.
_DEFAULT_TIMEOUT_S = 600.0
#: The most an agent may ask for with `timeout_s`. A call runs as a job, so a
#: long solve blocks nothing, but an unbounded one would hold the GPU forever.
MAX_TIMEOUT_S = 3600.0
_LOG_TAIL_CHARS = 4000

# Repo root (…/dycraft-dev): this file is at <root>/animation/ik/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Written by setup_ik.sh; holds the interpreter path, one line.
_POINTER = _home.state_path(".ik_python")
_CONVENTIONAL = (
    Path.home() / "conda_envs" / "dycraft-ik" / "bin" / "python",
)


def _resolve_ik_python() -> str | None:
    """The interpreter to run IK code in, or None for "unavailable".

    `DYCRAFT_IK_PYTHON` (set-but-empty forces None) > the `.ik_python` pointer
    file > conventional prefixes. None is a supported outcome: the layer keeps
    working and the agent is told to author the configuration itself, exactly
    as it is when no physics engine is configured.
    """
    return _interpreters.resolve(env_var="DYCRAFT_IK_PYTHON",
                                 pointer=_POINTER,
                                 conventional=_CONVENTIONAL)


#: Why `h.ik` cannot run in this launch, set by the launcher when it decides so
#: (`decide_for_launch`), so the daemon, its sub-runs, and the op all agree:
#: `not_installed` or `no_gpu`. With it the launcher also empties
#: `DYCRAFT_IK_PYTHON`, which on its own already forces "no interpreter".
UNAVAILABLE_ENV = "DYCRAFT_IK_UNAVAILABLE"
NOT_INSTALLED = "not_installed"
NO_GPU = "no_gpu"

#: What the agent is told when it calls `h.ik` and IK is unavailable, by reason.
_WHY = {NOT_INSTALLED: "cuRobo is not installed",
        NO_GPU: "cuRobo needs an NVIDIA GPU and none is visible"}


def unavailable_message(reason: str) -> str:
    """The op's agent-facing reply when IK is unavailable for `reason`."""
    return (f"`h.ik` is not available on this machine: {_WHY.get(reason, _WHY[NOT_INSTALLED])}. "
            "Solve the joints with your own solvers, as the `humanoid_motion` skill describes; "
            "the rest of the animation layer works as usual.")


def gpu_visible() -> tuple[bool, str]:
    """Whether a GPU is visible to a child of this process, and if not, why.

    cuRobo has no CPU path. Answered without importing torch, so a launch pays
    a few milliseconds: a `CUDA_VISIBLE_DEVICES` that hides every device (empty,
    `-1`, or Slurm's `NoDevFiles`) is decisive on its own; otherwise
    `nvidia-smi` has to list at least one GPU.
    """
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is not None and devices.strip() in ("", "-1", "NoDevFiles"):
        return False, ("CUDA_VISIBLE_DEVICES is empty" if not devices.strip()
                       else f"CUDA_VISIBLE_DEVICES is {devices.strip()}")
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False, "nvidia-smi is not installed"
    try:
        listed = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False, "nvidia-smi did not answer"
    if not any(line.startswith("GPU ") for line in (listed.stdout or "").splitlines()):
        return False, "nvidia-smi lists no GPU"
    return True, ""


def availability() -> dict:
    """`{"available", "reason", "detail"}`: whether `h.ik` can run here.

    The launcher's decision when it made one (`UNAVAILABLE_ENV`); otherwise an
    interpreter has to resolve and a GPU has to be visible.
    """
    forced = os.environ.get(UNAVAILABLE_ENV)
    if forced:
        return {"available": False, "reason": forced, "detail": ""}
    if _resolve_ik_python() is None:
        return {"available": False, "reason": NOT_INSTALLED,
                "detail": "not installed (python -m dycraft.install ik)"}
    visible, detail = gpu_visible()
    if not visible:
        return {"available": False, "reason": NO_GPU, "detail": detail}
    return {"available": True, "reason": None, "detail": ""}


def is_available() -> bool:
    """Whether `h.ik` can run here: an interpreter and a visible GPU."""
    return availability()["available"]


def decide_for_launch(env=None) -> dict:
    """Decide once, at launch, whether this run has IK, and pin the answer.

    Unavailable: `DYCRAFT_IK_PYTHON` is emptied and `UNAVAILABLE_ENV` set, in
    `env` (default `os.environ`), so everything the launch starts agrees. The
    result is what `meta.json` records under `ik`.
    """
    env = os.environ if env is None else env
    found = availability()
    if not found["available"]:
        env["DYCRAFT_IK_PYTHON"] = ""
        env[UNAVAILABLE_ENV] = found["reason"]
    return found


def launch_line(found: dict) -> str | None:
    """The launcher's one line on stderr when IK is unavailable, else None."""
    if found.get("available"):
        return None
    if found.get("reason") == NO_GPU:
        why = "no NVIDIA GPU is visible" + (f" ({found['detail']})" if found.get("detail") else "")
    else:
        why = "cuRobo is not installed (optional: python -m dycraft.install ik)"
    return (f"[dycraft] IK (cuRobo): unavailable — {why}. "
            "The agent solves kinematics itself; nothing else changes.")


def _timeout_s() -> float:
    raw = os.environ.get("DYCRAFT_IK_TIMEOUT_S")
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def _helper_hashes() -> dict:
    """Short sha256 of every file `stage_helpers` copied into the export dir.

    Recorded on the payload so a result says which adapter produced it. Needed
    because `stage_helpers` rewrites those files on **every** call, including a
    call made against an archived run dir later, so the staged copy beside an
    old result is not necessarily the copy that produced it. A hash in
    `result.json` is per-call and immutable, so it cannot be overwritten the way
    the file can.

    Best-effort, like `_persist`: an unreadable helper must not be what fails a
    call that was otherwise fine.
    """
    out = {}
    for path in (_HELPER_PATH, _ROTATION_PATH):
        try:
            out[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        except OSError:
            pass
    return out


def _checkpoint_path(provenance_dir) -> "str | None":
    """Where the child's `checkpoint()` writes, or `None` if unavailable.

    Inside the provenance dir, because that is the only directory in play that
    outlives the call — `--result` lives in a `TemporaryDirectory` that is gone
    by the time a timeout is reported. Created here so the child's first
    snapshot cannot fail on a missing parent.
    """
    if provenance_dir is None:
        return None
    try:
        d = Path(provenance_dir)
        d.mkdir(parents=True, exist_ok=True)
        return str(d / CHECKPOINT_NAME)
    except OSError:
        return None


def _read_checkpoint(path) -> "dict | None":
    """The child's last `checkpoint()`, for the paths where it wrote no result.

    A timeout kill and a `runner_died` CUDA-OOM kill are the two failures that
    otherwise return nothing at all, and on this op they are the expensive ones:
    one archived call burned its whole 600 s budget and came back with no result
    of any kind. Anything unreadable is treated as absent.
    """
    if not path:
        return None
    try:
        f = Path(path)
        if not f.is_file():
            return None
        v = json.loads(f.read_text())
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def _fail(kind: str, message: str, **extra) -> dict:
    return {"ok": False, "error_kind": kind, "message": message, **extra}


def _tail(stream) -> str:
    """Last `_LOG_TAIL_CHARS` of a captured stream, whatever shape it arrives in.

    `subprocess.run(text=True)` gives `str`, but `TimeoutExpired` is populated
    by the `communicate()` inside `run`'s kill path and can carry `bytes` on
    some platforms — a timeout is exactly when we least want a
    `TypeError` on top.
    """
    if not stream:
        return ""
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", "replace")
    return stream[-_LOG_TAIL_CHARS:]


def stage_helpers(ik_dir) -> None:
    """Put `dycraft_ik.py` and `_rotation.py` in the exported world.

    The child runs with `ik_dir` as its cwd, so this is what makes
    `import dycraft_ik` resolve with no path juggling.

    `_rotation.py` goes verbatim from `sdk/` for the same reason the sim export
    does it: the adapter converts joint values between surface and internal
    units, and a second implementation of that conversion living in the IK env
    would be a factor-of-57 divergence nothing could detect.

    Exposed rather than inlined so a test can stage into a tmp dir and exercise
    the frames-and-units half with no GPU — which is where the traps that fail
    silently are pinned.
    """
    ik_dir = Path(ik_dir)
    ik_dir.mkdir(parents=True, exist_ok=True)
    (ik_dir / "dycraft_ik.py").write_text(_HELPER_PATH.read_text())
    (ik_dir / "_rotation.py").write_text(_ROTATION_PATH.read_text())


# ---------------------------------------------------------------------------
# ik_meta.json — the facts only this side of the boundary can compute
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------

def ik(code: str, *, ik_dir, provenance_dir=None, name: str = "ik",
       timeout_s: float | None = None, cache_dir=None, cancel=None,
       require_scene: bool = True, extra_env=None) -> dict:
    """Run `code` in the isolated IK interpreter; return its recorded result.

    `ik_dir` is the child's working directory, with `dycraft_ik` staged in it.
    With `require_scene` it must be an exported world, so a script reaches
    `bodies/`, `meshes/`, `scene_meta.json`, and `ik_meta.json` by their plain
    names. The animation layer passes `require_scene=False`: its working
    directory is the agent's
    own persistent `ik/`, where it builds its robot and world however it
    chooses, and any exported scene sits in a subdirectory named through
    `extra_env` (`DYCRAFT_IK_EXPORTS`).

    `cache_dir` is handed to the child as `DYCRAFT_IK_CACHE_DIR`. The sphere fit
    is **not deterministic** — the same mesh can get a different number of
    spheres on each call, and there is no seed parameter — so the cache is what makes
    two calls over one frozen scene use the same collision model. It is a
    correctness mechanism before it is a speed one.

    Returns `{"ok": True, "result": ..., "stdout": ..., "duration_s": ...}` or
    `{"ok": False, "error_kind": ..., ...}`.
    """
    found = availability()
    py = _resolve_ik_python()
    if not found["available"] or py is None:
        return _fail("unavailable", unavailable_message(found["reason"] or NOT_INSTALLED))

    ik_dir = Path(ik_dir)
    if require_scene and not (ik_dir / "scene_meta.json").is_file():
        return _fail(
            "no_scene",
            f"no exported world at {ik_dir} (scene_meta.json missing). This is "
            f"a harness wiring fault, not something the script can fix.")

    stage_helpers(ik_dir)

    from ..blender.runtime import arm_gpu_env
    # On the arm's GPU, numbered as the sandboxed Blender numbers it.
    env = {**os.environ, **arm_gpu_env()}
    env["DYCRAFT_IK_CACHE_DIR"] = str(
        cache_dir or (Path.home() / ".cache" / "dycraft" / "curobo_robots"))
    env.update({k: str(v) for k, v in (extra_env or {}).items()})

    with tempfile.TemporaryDirectory(prefix="dycraft-ik-") as tmp:
        tmp = Path(tmp)
        code_path = tmp / "ik_code.py"
        result_path = tmp / "result.json"
        code_path.write_text(code)

        # `-u`: unbuffered, and load-bearing rather than cosmetic. A child
        # killed at the timeout loses whatever is still sitting in its stdio
        # buffer, so block-buffered `print`s vanish exactly in the case where
        # they are the only evidence left. Measured: without `-u` a timed-out
        # child returns `stdout=None`; with it, everything printed so far.
        ckpt = _checkpoint_path(provenance_dir)
        argv = [py, "-u", str(_RUNNER_PATH), "--code", str(code_path),
                "--result", str(result_path), "--codename", "ik_code.py",
                "--setup", str(_SETUP_PATH)]
        if ckpt:
            argv += ["--checkpoint", ckpt]
        timeout = (min(max(1.0, float(timeout_s)), max(MAX_TIMEOUT_S, _timeout_s()))
                   if timeout_s else _timeout_s())
        t0 = time.perf_counter()
        try:
            from .._process import run_cancellable
            proc = run_cancellable(argv, cancel=cancel, cwd=str(ik_dir), capture_output=True,
                                  text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as e:
            # `TimeoutExpired` carries whatever the child had written before it
            # was killed. Dropping it would make a timeout the one failure that
            # returns *nothing*, and the agent could not tell where the time
            # went.
            partial_out = _tail(e.stdout)
            partial_err = _tail(e.stderr)
            # The advice is ordered by measured cost, not by plausibility:
            # obstacles first, because a script that builds its solver once can
            # still spend the time there. `stdout` is what tells you which.
            out = _fail("timeout",
                        f"the IK call did not finish within {timeout:.0f}s. "
                        f"`stdout` below is whatever your script printed before "
                        f"it was killed — read it first, it says which stage "
                        f"you were in. The three costs, largest first: (1) the "
                        f"obstacle world — one mesh takes a solve from 0.08s to "
                        f"0.97s, so pass `scene_config` only the meshes this "
                        f"motion can reach; (2) constructing an "
                        f"`InverseKinematics` or `MotionPlanner` costs tens of "
                        f"seconds (54s at the default sphere density) each "
                        f"time, so build it once outside any loop; (3) the "
                        f"number of solves — `plan_grasp` plans a whole "
                        f"approach-grasp-lift in one call, and "
                        f"`ToolPoseCriteria.linear_motion` constrains a "
                        f"straight segment, so you rarely need to solve every "
                        f"frame yourself. `num_seeds` is not the knob: 12 to 32 "
                        f"costs 43%. Splitting across several `h.ik` calls "
                        f"gives each its own budget, but re-pays that build.",
                        stdout=partial_out, stderr=partial_err,
                        timeout_s=timeout)
            out["partial_result"] = _read_checkpoint(ckpt)
            out["helpers"] = _helper_hashes()
            _persist(provenance_dir, code, out, partial_out)
            return out
        except OSError as e:
            out = _fail("launch_failed", f"could not start {py}: {e}")
            out["helpers"] = _helper_hashes()
            _persist(provenance_dir, code, out, "")
            return out
        duration = time.perf_counter() - t0

        stdout = _tail(proc.stdout)
        stderr = _tail(proc.stderr)
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text())
            except ValueError as e:
                payload = _fail("bad_result",
                                f"the runner wrote unreadable JSON: {e}")
        else:
            payload = _fail(
                "runner_died",
                f"the IK process exited with code {proc.returncode} without "
                f"writing a result. A CUDA out-of-memory kill looks like this; "
                f"the tail of its output is in `stderr`.",
                stderr=stderr,
                partial_result=_read_checkpoint(ckpt))

        if payload.get("error_kind") == "unavailable":
            # The child's backstop (`_ik_setup.py`): torch saw no CUDA device,
            # a GPU hidden or lost after launch. Told in the op's own words.
            payload["message"] = unavailable_message(NO_GPU)
        payload["stdout"] = stdout
        payload["duration_s"] = round(duration, 3)
        payload["helpers"] = _helper_hashes()
        if not payload.get("ok") and "stderr" not in payload and stderr:
            payload["stderr"] = stderr
        _persist(provenance_dir, code, payload, stdout)
        return payload


def _persist(provenance_dir, code: str, payload: dict, stdout: str) -> None:
    """Drop the call into the run dir. Best-effort: provenance must never be
    the reason a result is lost."""
    if provenance_dir is None:
        return
    try:
        d = Path(provenance_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "code.py").write_text(code)
        (d / "result.json").write_text(json.dumps(payload, indent=2,
                                                  default=str))
        if stdout:
            (d / "stdout.txt").write_text(stdout)
    except OSError:
        pass


__all__ = ["IK_DIR", "IK_RUNS_DIR", "IK_META", "ik", "is_available",
           "stage_helpers"]
