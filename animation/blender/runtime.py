"""Isolated Blender subprocesses. Never execute authored code in the daemon."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from ... import _home, _oom
from ..._interpreters import SandboxView, sandbox_view
from ...render._blender_env import blender_binary, blender_python
from .bundle import write_json

REPO = Path(__file__).resolve().parents[2]

#: Where `cache_dir` is mounted inside the sandbox, and what `run_worker`
#: points `XDG_CACHE_HOME` at.
CACHE_MOUNT = "/blendercache"
#: Where the read-only motion-capture library appears to authored Blender code.
#: Absent when `assets/mocap/` holds no library (`mocap_libraries`).
MOCAP_MOUNT = "/references/mocap"
#: Agent-facing Blender helpers (`import mocap_retarget as mr`), bound into
#: `/tools` beside the verifier modules whether or not the library is here:
#: `mocap_retarget.bone_map` maps a rig's bones by role with no capture at all.
#: Kept out of this directory because
#: `session.implementation_fingerprint` hashes it, and an edit to a helper there
#: invalidated every cached verification, of runs that never imported it too.
#: The mocap library is not part of the release, so the bind below is guarded.
HELPERS = Path(__file__).resolve().parents[1] / "mocap"


def _holds_a_file(directory):
    """Whether `directory` holds a non-hidden file at any depth.

    Stops at the first one. Hidden files and directories do not count.
    Directory links are followed, each real directory once, so a link back up
    the tree cannot loop.
    """
    seen = set()
    for root, dirs, files in os.walk(directory, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        if any(not name.startswith(".") and os.path.isfile(os.path.join(root, name)) for name in files):
            return True
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
    return False


def mocap_libraries():
    """The motion-capture libraries this checkout has: {name: directory}.

    A library is any non-hidden subdirectory of `assets/mocap/` with at least
    one non-hidden file under it, in any format, so a library a user adds counts
    the same as the fetched `cmu/` and `mann_dog/`. A directory with no files is
    not a library. Both `mocap_library` and the environment check's
    `optional.mocap` read this, so they agree on what is installed.
    """
    root = _home.state_path("assets", "mocap").resolve()
    if not root.is_dir():
        return {}
    return {entry.name: entry for entry in sorted(root.iterdir())
            if not entry.name.startswith(".") and entry.is_dir() and _holds_a_file(entry)}


def mocap_library():
    """The motion-capture root this checkout has, or None when it holds no library.

    `assets/mocap/` is gitignored and linked into run worktrees, so `resolve()`
    follows a worktree's link to the shared copy. What the agent is shown of
    mocap follows it: the Blender mount, the `mocap_reference` Skill, and the
    prompts' mocap blocks (`agent_runner.OPTIONAL`); `agent.stage_references`
    links `references/mocap` by the same test. An empty `assets/mocap/`, such as
    one a failed fetch leaves, returns None, because pointing the agent at a
    library with no captures sends it looking for nothing.
    """
    lib = _home.state_path("assets", "mocap").resolve()
    return lib if mocap_libraries() else None


def cache_dir():
    """A persistent host directory for the GPU driver's compiled-shader cache.

    EEVEE compiles the scene's GLSL on its first render and the driver keeps the
    resulting program binaries under `$XDG_CACHE_HOME` (NVIDIA writes
    `nvidia/GLCache`). `command` mounts `/tmp` as a fresh tmpfs, so without a
    persistent directory the cache would be discarded with every process and
    each worker would recompile the whole scene from scratch, which costs tens
    of seconds per worker, and `storyboard` starts one render worker per shot.
    Backed by one real directory, only the first worker pays it.

    Local disk by default rather than `~/.cache`: the cache is rewritten
    constantly and a network home is both slow and quota-bound.
    `DYCRAFT_BLENDER_CACHE` overrides it where the temporary directory is too
    short-lived to be worth sharing. The driver bounds its own eviction, so this
    directory does not grow without limit; deleting it only costs one recompile.

    Returns `None` if the directory cannot be made. A cache is an optimization,
    so an unwritable temporary directory falls back to no cache -- slow renders
    -- rather than failing every Blender operation in the daemon.
    """
    override = os.environ.get("DYCRAFT_BLENDER_CACHE")
    root = (Path(override).expanduser() if override else
            Path(tempfile.gettempdir()) / f"dycraft-blender-cache-{os.getuid()}")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    # The OptiX and CUDA kernel caches (`OPTIX_CACHE_PATH`, `CUDA_CACHE_PATH`,
    # see `worker_env`), made here so they never depend on the driver creating
    # its own directories. Failing to make one leaves the shader cache usable.
    for sub in ("optix", "nvidia/ComputeCache"):
        try:
            (root / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return root


#: Cached result of `_userns_prefix`; `None` until probed, then a list.
_USERNS_PREFIX = None

def _dev_nodes():
    """Every `/dev/nvidiaN` GPU node, in index order."""
    return sorted(Path("/dev").glob("nvidia[0-9]*"), key=lambda n: int(n.name[len("nvidia"):]))


def _openable(dev) -> bool:
    try:
        os.close(os.open(dev, os.O_RDWR))
    except OSError:
        return False
    return True


def _nvidia_selection():
    """`(nodes, by_index)`: the `/dev/nvidiaN` nodes the sandboxed Blender may use,
    and whether they were picked by absolute index from `CUDA_VISIBLE_DEVICES`.

    Forwarding `CUDA_VISIBLE_DEVICES` is not enough on its own: CUDA honors it but
    EGL does not, and EEVEE rendered on GPU 0 under `CUDA_VISIBLE_DEVICES=5` on an
    8x V100 node without per-job device isolation (a cluster-side fix of 3 Oct,
    merged here). So the sandbox binds only the arm's nodes:
    - under a device cgroup (slurm's ConstrainDevices) only the granted nodes
      open, and the variable is numbered within the grant: the openable nodes;
    - when every node opens and the variable is a list of indices, those are
      absolute indices (an operator's pick): those nodes;
    - otherwise (unset, empty, UUIDs, -1): every openable node, as before.
    """
    nodes = _dev_nodes()
    usable = [n for n in nodes if _openable(n)]
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    parts = [p.strip() for p in raw.split(",")] if raw else []
    if usable and len(usable) == len(nodes) and parts and all(p.isdigit() for p in parts):
        keep = {int(p) for p in parts}
        return [n for n in usable if int(n.name[len("nvidia"):]) in keep], True
    return usable, False


def _nvidia_devices():
    """The `/dev/nvidiaN` nodes bound into the sandbox; see `_nvidia_selection`."""
    return _nvidia_selection()[0]


#: Cached result of `nvidia_usable`; `None` until probed, then a bool.
_NVIDIA_USABLE = None


def nvidia_usable() -> bool:
    """Whether this process may open an NVIDIA GPU, not merely see its node.

    Existence is the wrong test on a shared cluster. A slurm job granted no GPU
    on a GPU node still has every `/dev/nvidiaN` in its mount namespace; the job's
    device cgroup is what refuses them, and it refuses at `open()` with EPERM. On
    such a host, pointing GLVND at the NVIDIA EGL vendor fails every context
    with EGL_BAD_ALLOC and Blender aborts at startup -- EEVEE and Cycles alike,
    before any scene code runs. So probe the way the kernel will judge
    the worker: try to open one.
    """
    global _NVIDIA_USABLE
    if _NVIDIA_USABLE is None:
        _NVIDIA_USABLE = bool(_nvidia_devices())
    return _NVIDIA_USABLE


#: The two Blender tiers, and the prefix that applies them. Defined in `_oom`
#: because the CAD pool in `agent/cad/_worker_pool.py` shares them:
#: one order for the whole harness, in one file, or the tiers drift apart.
WORKER_OOM_SCORE_ADJ = _oom.WORKER
RESIDENT_OOM_SCORE_ADJ = _oom.RESIDENT
_oom_prefix = _oom.prefix


def _userns_prefix(unshare):
    """`[unshare, "-Urn"]`, or `[]` when this host will not let it run.

    The child needs a **user** namespace and a **network** namespace. Two
    binaries can establish them and which one is allowed is host policy, not
    ours:

    * `unshare -Urn` — the original, and still preferred where it works. `-U`
      new user namespace, `-r` mapped to root inside it, `-n` new network
      namespace.
    * `bwrap --unshare-user --unshare-net` — the same two namespaces, made by
      the binary that is already creating the mount namespace for the binds.

    Ubuntu 24.04 ships `/etc/apparmor.d/bwrap-userns-restrict`, a profile that
    grants `/usr/bin/bwrap` the right to create user namespaces while
    `kernel.apparmor_restrict_unprivileged_userns=1` denies it to everything
    else. On such a host `unshare -Urn` fails with
    `write failed /proc/self/uid_map: Operation not permitted` before bwrap is
    even reached, and the whole animation layer is unusable — not less isolated,
    simply dead.

    So this probes once and the caller adds bwrap's own flags when the prefix is
    empty. **The isolation is the same either way**; what varies is which
    process establishes it. The one visible difference is the uid inside: `-r`
    maps to 0, bwrap alone keeps the real uid. Nothing the worker does depends
    on it — it writes only to bind-mounted directories the invoking user already
    owns, and `--cap-drop ALL` applies on both paths.

    This is deliberately not an escape hatch: if neither route can make the
    namespaces, `command` raises rather than running the child unconfined.
    """
    global _USERNS_PREFIX
    if _USERNS_PREFIX is None:
        ok = False
        if unshare:
            try:
                ok = subprocess.run([unshare, "-Urn", "true"],
                                    capture_output=True,
                                    timeout=30).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        _USERNS_PREFIX = [unshare, "-Urn"] if ok else []
    return _USERNS_PREFIX


#: Forces `sandbox_mode`: `namespaces`, `landlock`, or `unconfined` (which still needs `--allow-unsandboxed`).
#: Unset or `auto`: the first that works here, in that order.
SANDBOX_ENV = "DYCRAFT_BLENDER_SANDBOX"
#: `--allow-unsandboxed`'s switch and its other name (`cli/backends/_base.allow_unsandboxed`), read here directly so
#: the worker's runtime does not import the CLI.
ALLOW_UNSANDBOXED_ENVS = ("DYCRAFT_ALLOW_UNSANDBOXED", "DYCRAFT_ALLOW_UNSANDBOXED_CODEX")
#: The launcher that confines a worker with Landlock in place of namespaces.
LANDLOCK_LAUNCHER = Path(__file__).with_name("_landlock.py")

#: Cached by `_namespaces_problem`: None until probed, then "" (they work) or why not.
_NAMESPACES_PROBLEM = None


def _bwrap_path():
    return shutil.which("bwrap") or str(Path(sys.executable).parent / "bwrap")


def _namespaces_problem() -> str:
    """"" where bubblewrap can make the worker's user and network namespaces here; otherwise what it said."""
    global _NAMESPACES_PROBLEM
    if _NAMESPACES_PROBLEM is None:
        bwrap = _bwrap_path()
        if not Path(bwrap).is_file():
            _NAMESPACES_PROBLEM = "bubblewrap (bwrap) is not installed"
        elif _userns_prefix(shutil.which("unshare")):
            _NAMESPACES_PROBLEM = ""
        else:
            try:
                ran = subprocess.run(_oom.tiered([bwrap, "--unshare-user", "--unshare-net", "--ro-bind", "/", "/",
                                                  "true"], _oom.WORKER), capture_output=True, text=True, timeout=30)
                said = (ran.stderr or ran.stdout).strip().splitlines()
                _NAMESPACES_PROBLEM = "" if ran.returncode == 0 else \
                    f"{bwrap} cannot make a user and a network namespace here ({said[-1] if said else ran.returncode})"
            except (OSError, subprocess.SubprocessError) as exc:
                _NAMESPACES_PROBLEM = f"{bwrap} could not run: {exc}"
    return _NAMESPACES_PROBLEM


def landlock_abi() -> int:
    """This kernel's Landlock ABI version, 0 where it has none."""
    from ._landlock import landlock_abi as abi
    return abi()


def sandbox_mode() -> str:
    """How the Blender worker is confined on this machine.

    * `namespaces`: bubblewrap, with a filesystem of its own (`/workspace`, `/tools`, `/output`, a private `/tmp`)
      and no network, PIDs or IPC shared with the host. Wherever it can make user namespaces.
    * `landlock`: where it cannot, as on stock Ubuntu 24.04 without the system's bubblewrap, and with no sudo:
      Landlock and a socket filter confine the same reads, writes and network, and the worker sees the host's paths
      (`_landlock.py`).
    * `unconfined`: the same host paths with no confinement, only with `--allow-unsandboxed` and only where the
      kernel has no Landlock.

    Raises RuntimeError, saying what to do, where none applies.
    """
    forced = (os.environ.get(SANDBOX_ENV) or "auto").strip().lower()
    opted_out = any(os.environ.get(name) == "1" for name in ALLOW_UNSANDBOXED_ENVS)
    if forced not in ("auto", "namespaces", "landlock", "unconfined"):
        raise RuntimeError(f"{SANDBOX_ENV}={forced!r}: expected auto, namespaces, landlock or unconfined")
    if forced in ("auto", "namespaces"):
        problem = _namespaces_problem()
        if not problem:
            return "namespaces"
        if forced == "namespaces":
            raise RuntimeError(f"Blender's sandbox cannot start: {problem}")
    if forced in ("auto", "landlock") and landlock_abi() > 0:
        return "landlock"
    if forced == "landlock":
        raise RuntimeError("Blender's sandbox cannot start: this kernel has no Landlock")
    if opted_out:
        return "unconfined"
    from ... import _userns
    steps = _userns.fixes() or ["install the system's bubblewrap (`sudo apt-get install bubblewrap`)"]
    raise RuntimeError(
        f"Blender's sandbox cannot start here: {_namespaces_problem() or 'unconfined needs --allow-unsandboxed'}, "
        f"and this kernel has no Landlock. With root: {'; '.join(steps)}. Or pass --allow-unsandboxed to run "
        "Blender without a sandbox, so that the agent's Blender code can read and write anywhere you can.")


#: What each sandboxed interpreter has to import, and so what its bind set is
#: probed against: the collision child (`coal_worker.py`, `hashlib` included
#: because its extension links the interpreter's own libcrypto) and the `bpy`
#: module worker. A Blender *binary* is its own directory and is not probed.
COAL_IMPORTS = ("hashlib", "pickle", "numpy", "coal")
BPY_IMPORTS = ("bpy", "numpy")

#: `(python, imports)` pairs already warned about, so a daemon spawning
#: hundreds of workers says it once.
_VIEW_WARNED = set()


def _python_view(python, imports, *, keep_name):
    """`sandbox_view(python, imports)`, or the pre-view recipe if it cannot answer.

    The fallback is the bind set this function used before it asked the
    interpreter -- resolve it, bind its grandparent -- and it is correct for a
    conda env, which is every configuration that worked then. It exists so an
    interpreter that cannot import `imports` fails where it always did (a
    collision check, a worker import) rather than taking every Blender spawn
    down with it; the warning says so up front, and `environment_test.py`
    names the cause.
    """
    try:
        return sandbox_view(python, imports)
    except RuntimeError as exc:
        resolved = Path(python).resolve()
        root = str(resolved.parents[1])
        if (python, imports) not in _VIEW_WARNED:
            _VIEW_WARNED.add((python, imports))
            print(f"[dycraft] sandbox: {exc}\n[dycraft] sandbox: falling back to "
                  f"binding {root} only; if this interpreter lives in a venv or "
                  "a user site it will start without its packages. Run "
                  "`python environment_test.py --deep` for a diagnosis.",
                  file=sys.stderr, flush=True)
        return SandboxView(str(python) if keep_name else str(resolved),
                           ((root, root),), ())


def command(workspace, output, job, readonly=(), *, workspace_readonly=False,
            serve=False, bind_root=None, writable=()):
    """The sandboxed Blender argv.

    `serve=True` is the resident form: no `/job.json` to bind, because requests
    arrive on stdin instead of as a file, and `/output` is a root the caller
    subdivides per request rather than one request's directory. The sandbox is
    otherwise identical -- same namespaces, same capability drop, same curated
    read-only `/tools` -- because residency is about the process's lifetime,
    not about what it is allowed to reach.

    `bind_root` binds that directory instead of the project and points the
    child at the project inside it, which is what makes a long-lived child
    survive a publish. See the branch below.
    """
    # An official Blender binary is preferred over the `bpy` wheel because the
    # wheel's Mantaflow bindings are broken: any fluid bake aborts inside the
    # solver. See `render/_blender_env.blender_binary`. Everything else about
    # the sandbox is identical, and `worker.py` runs unchanged under either --
    # it only ever needed `import bpy`, which a real Blender also provides.
    binary = blender_binary()
    interpreter = None if binary else blender_python()
    if not binary and not interpreter:
        raise RuntimeError("Blender unavailable; configure DYCRAFT_BLENDER_BINARY "
                           "or DYCRAFT_BLENDER_PYTHON")
    # A real Blender is one self-contained directory. A `bpy` interpreter is
    # whatever Python layout it was installed into -- see `_python_view`.
    bpy_view = None if binary else _python_view(interpreter, BPY_IMPORTS,
                                                keep_name=True)
    runtime_binds = ([(str(Path(binary).resolve().parent),
                       str(Path(binary).resolve().parent))]
                     if binary else list(bpy_view.binds))
    # The sandbox is described as a bubblewrap argv whatever the mode: where namespaces cannot be made,
    # `_host_command` turns that argv into Landlock's rules, so there is one list of what the worker may reach.
    mode = sandbox_mode()
    bwrap = _bwrap_path()
    prefix = _userns_prefix(shutil.which("unshare")) if mode == "namespaces" else []
    # Priced for the OOM killer before anything else is decided; see `_oom_prefix`.
    head = _oom_prefix(RESIDENT_OOM_SCORE_ADJ if serve else WORKER_OOM_SCORE_ADJ)
    cmd = [*head, *prefix, bwrap]
    if not prefix:
        # bwrap makes the user and network namespaces itself. See
        # `_userns_prefix` for when and why.
        cmd += ["--unshare-user", "--unshare-net"]
    cmd += ["--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    coal_python = os.environ.get('DYCRAFT_COAL_PYTHON') or sys.executable
    coal_python = str(Path(shutil.which(coal_python) or coal_python).expanduser())
    if not Path(coal_python).is_file():
        raise RuntimeError('Coal interpreter unavailable; configure DYCRAFT_COAL_PYTHON')
    # Never a resolved path, and never an alias: `_python_view` resolves the
    # *directory* and keeps the name. On one cluster the env was reached
    # through a symlinked home but lived elsewhere; handed the alias, the
    # child was told to run a path that does not exist inside the sandbox and
    # every collision check died in `CoalClient` with FileNotFoundError. Handed
    # the fully resolved path instead, a venv's `bin/python` became its base
    # interpreter, which starts without the venv's packages.
    coal_view = _python_view(coal_python, COAL_IMPORTS, keep_name=False)
    binds = [(str(Path(p).resolve()), p) for p in ("/usr", "/lib", "/lib64")
             if Path(p).exists()]
    for src, dest in [*runtime_binds, *coal_view.binds]:
        if Path(src).exists() and all(dest != d for _, d in binds):
            binds.append((src, dest))
    for src, dest in binds:
        cmd += ["--ro-bind", src, dest]
    gpus = set(_nvidia_devices())
    for path in [Path("/dev/dri"), *Path("/dev").glob("nvidia*")]:
        if path.name[len("nvidia"):].isdigit() and path not in gpus:
            continue  # another arm's GPU; see `_nvidia_selection`
        if path.exists():
            cmd += ["--dev-bind", str(path), str(path)]
    if nvidia_usable():
        cmd += ["--ro-bind", "/sys", "/sys"]
    for path in (Path("/etc/fonts"), Path("/etc/glvnd"), Path("/etc/vulkan"), Path("/etc/ld.so.cache")):
        if path.exists():
            cmd += ["--ro-bind", str(path), str(path)]
    gl = os.environ.get("DYCRAFT_GL_PREFIX")
    if gl and Path(gl).is_dir():
        # The GL libraries `install tools` installed where the system had none (`worker_env` points at them).
        cmd += ["--ro-bind", str(Path(gl).resolve()), gl]
    for path in readonly:
        cmd += ["--ro-bind", str(Path(path).resolve()), str(Path(path).resolve())]
    # Harness-owned state the worker updates in place, at its own path: the
    # project's collision pair store (`collision_cache.PairStore`).
    for path in writable:
        Path(path).mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(Path(path).resolve()), str(Path(path).resolve())]
    # Minimal read-only verifier modules, never the project or agent system.
    for name in ('collision', 'coal_worker', 'continuity', 'evaluation', 'verify', 'verify_anatomy',
                 'verify_declarations', 'verify_penetration', 'verify_room', 'verify_support', 'collision_cache',
                 'render_config', 'movie', 'update', 'scene_digest', 'dycraft_joints', 'render_state', 'chunking',
                 'rig_anatomy', 'animation_diff', 'frame_export', 'surface_colors'):
        cmd += ['--ro-bind', str(Path(__file__).with_name(name+'.py')), '/tools/'+name+'.py']
    if HELPERS.is_dir():
        for helper in sorted(HELPERS.glob('[!_]*.py')):
            cmd += ['--ro-bind', str(helper), '/tools/'+helper.name]
    for name in ('_mesh_erosion', '_inherited_allowances', '_allowance_rules'):
        cmd += ['--ro-bind', str(REPO / 'verify' / (name+'.py')), '/tools/'+name+'.py']
    cache = cache_dir()
    if cache is not None:
        cmd += ["--bind", str(cache), CACHE_MOUNT]
    # The motion-capture reference library (`mocap_reference` skill), read-only,
    # when this checkout has one.
    mocap = mocap_library()
    if mocap is not None:
        cmd += ["--ro-bind", str(mocap), MOCAP_MOUNT]
    cmd += ["--setenv", "DYCRAFT_COAL_PYTHON", coal_view.launch]
    if coal_view.sitedirs:
        # `CoalClient` starts coal with `-I`, which sees no user site and no
        # PYTHONPATH; `coal_worker.serve` re-adds exactly these.
        cmd += ["--setenv", "DYCRAFT_COAL_SITEDIRS", os.pathsep.join(coal_view.sitedirs)]
    if bpy_view is not None and bpy_view.sitedirs:
        # After `/tools`, which is what `worker_env` sets and the worker needs first.
        cmd += ["--setenv", "PYTHONPATH",
                os.pathsep.join(["/tools", *bpy_view.sitedirs])]
    # The worker's render recipe, forwarded explicitly.
    #
    # `run_worker` hands `Popen` a closed `env=` dict rather than extending the
    # daemon's, which is deliberate — the child must not inherit an operator's
    # PYTHONPATH or GL pins. The side effect was that `worker.py`'s
    # `DYCRAFT_BLENDER_*` knobs could never arrive: exporting
    # `DYCRAFT_BLENDER_ENGINE=BLENDER_WORKBENCH` changed nothing, EEVEE still
    # drew, and `bundle.make_manifest` — which read the same variables in the
    # *daemon* — recorded the engine that had not been used. Forwarding here
    # rather than widening the `env` dict keeps the allowlist explicit: these
    # five names and nothing else cross the boundary.
    for name in ("DYCRAFT_BLENDER_ENGINE", "DYCRAFT_BLENDER_EEVEE_SAMPLES",
                 "DYCRAFT_BLENDER_WIDTH", "DYCRAFT_BLENDER_HEIGHT",
                 "DYCRAFT_BLENDER_WORKBENCH_AA"):
        value = os.environ.get(name)
        if value:
            cmd += ["--setenv", name, value]
    cmd += ["--ro-bind", str(REPO / "render/_blender_child.py"), "/tools/_blender_child.py",
            "--ro-bind", str(Path(__file__).with_name("worker.py")), "/tools/worker.py"]
    if bind_root is None:
        inside = "/workspace"
        cmd += ["--ro-bind" if workspace_readonly else "--bind", str(workspace), "/workspace"]
    else:
        # Bind the project's *parent*, and point the child at the project
        # inside it. This is what lets a resident session survive `publish()`:
        # publishing swaps the project by renaming one directory for another
        # under this root, and a child holding the root sees the new tree on
        # its next open, where a child holding the project directory itself
        # would still be holding the replaced inode. The extra reach is the
        # session's own `native/` directory -- job outputs and snapshots it
        # already writes -- and never the run directory or the agent's sources.
        root = Path(bind_root).resolve()
        inside = "/native/" + str(Path(workspace).resolve().relative_to(root))
        cmd += ["--bind", str(root), "/native", "--setenv",
                "DYCRAFT_BLENDER_WORKSPACE_ROOT", inside]
        # `/workspace` is an author-facing path, not only an internal one: the
        # prompt tells authors their helpers "copy into isolated /workspace",
        # and authors rely on `exec(open('/workspace/helper.py').read())`.
        # Under `bind_root` that path would not exist, so the same code would
        # work as a job and raise FileNotFoundError resident, with no harness
        # error to attribute it to.
        #
        # A symlink rather than a second bind, for the reason `bind_root`
        # exists: publishing renames the project directory under the root, and
        # a bind would hold the replaced inode. A symlink is resolved at open
        # time, so it always names the current one.
        cmd += ["--symlink", inside, "/workspace"]
    cmd += ["--bind", str(output), "/output"]
    if not serve:
        cmd += ["--ro-bind", str(job), "/job.json"]
    if binary:
        # `--factory-startup` keeps an operator's preferences out of a rendered
        # result; `--python-use-system-env` is what restores the
        # `PYTHONPATH=/tools` that `worker_env` sets, which Blender otherwise
        # ignores; `--python-exit-code 1` turns a raising script into a failed
        # process instead of a silent success.
        cmd += ["--chdir", inside, binary, "--background", "--factory-startup",
                "--python-use-system-env", "--python-exit-code", "1",
                "--python", "/tools/worker.py"]
        if serve:
            # Everything after `--` is the script's; Blender keeps it in
            # `sys.argv`, which is where `worker.py`'s `--serve` check looks.
            cmd += ["--", "--serve"]
    else:
        cmd += ["--chdir", inside, bpy_view.launch, "-B", "/tools/worker.py"]
        if serve:
            cmd += ["--serve"]
    if mode != "namespaces":
        return _host_command(head, cmd[len(head) + len(prefix) + 1:], mode)
    return cmd


#: Device nodes `bwrap --dev` provides, which a worker confined by Landlock may use.
_BASIC_DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")


def _host_root() -> Path:
    """Where a worker run with host paths keeps its `/tmp` and its copy of `/tools`."""
    return Path(tempfile.gettempdir()) / f"dycraft-sandbox-{os.getuid()}"


def _stage_tools(files) -> Path:
    """A directory holding exactly `files` (`(source, name)` pairs), as `/tools` holds them in the namespaces mode.

    Named by their content, so workers share it, an edited file gets a new directory, and none is ever changed in
    place under a running worker."""
    import hashlib
    digest = hashlib.sha256()
    for src, name in sorted(files, key=lambda f: f[1]):
        digest.update(name.encode() + b"\0" + Path(src).read_bytes() + b"\0")
    root = _host_root() / "tools" / digest.hexdigest()[:20]
    if not root.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root.parent))
        for src, name in files:
            shutil.copyfile(src, staging / name)
        try:
            staging.rename(root)
        except OSError:                            # another worker staged it first
            shutil.rmtree(staging, ignore_errors=True)
    return root


def _host_command(head, args, mode):
    """The argv that runs the bubblewrap sandbox `args` describes with host paths instead (`_landlock.py`).

    Each bind's target is mapped to its source, and the sources become Landlock's allow list: read-only binds are
    read, `--bind` is written, `--dev-bind` is a device. The files bound one by one under `/tools` are copied into a
    directory of their own (`_stage_tools`). The worker is told its job, workspace, output and mocap paths through
    the variables it reads them from, as host paths.
    """
    import json
    pairs, read, write, dev, env, tools = [], ["/proc"], [], list(_BASIC_DEVICES), {}, []
    symlinks, chdir, program = [], "/", []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--ro-bind", "--bind", "--dev-bind"):
            src, dest = args[i + 1], args[i + 2]
            if dest.startswith("/tools/"):
                tools.append((src, dest[len("/tools/"):]))
            else:
                if src != dest:
                    pairs.append((dest, src))
                {"--ro-bind": read, "--bind": write, "--dev-bind": dev}[a].append(src)
            i += 3
        elif a == "--setenv":
            env[args[i + 1]] = args[i + 2]
            i += 3
        elif a == "--symlink":
            symlinks.append((args[i + 2], args[i + 1]))
            i += 3
        elif a in ("--cap-drop", "--proc", "--dev", "--tmpfs"):
            i += 2                                 # `/tmp` is the launcher's; the rest have no host counterpart
        elif a == "--chdir":
            chdir, program = args[i + 1], args[i + 2:]
            break
        else:
            i += 1                                 # --unshare-*, --die-with-parent, --new-session
    if tools:
        staged = _stage_tools(tools)
        pairs.append(("/tools", str(staged)))
        read.append(str(staged))
    # A symlink's target is written as the worker sees it; it maps through the binds like any other path.
    from ._landlock import mapper
    for dest, target in symlinks:
        pairs.append((dest, mapper(pairs)(target)))
    targets = {dest for dest, _ in pairs}
    # The worker's own paths, which it reads from these variables and otherwise assumes are the bind targets.
    for name, inside in (("DYCRAFT_BLENDER_WORKSPACE_ROOT", "/workspace"), ("DYCRAFT_BLENDER_OUTPUT_ROOT", "/output"),
                         ("DYCRAFT_BLENDER_JOB", "/job.json"), ("DYCRAFT_MOCAP_ROOT", MOCAP_MOUNT)):
        if inside in targets:
            env.setdefault(name, inside)
    env["DYCRAFT_BLENDER_SANDBOX_MODE"] = mode
    if any(Path(d).name.startswith("nvidia") for d in dev):
        # Every node is in the host's /dev, so CUDA numbers them as the host does, not as the bound subset.
        dev += [str(p) for p in Path("/dev").glob("nvidia*")]
        gpu = arm_gpu_env()
        if gpu.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = gpu["CUDA_VISIBLE_DEVICES"]
    spec = {"mode": mode, "map": pairs, "read": read, "write": write, "dev": dev, "env": env, "chdir": chdir,
            "argv": program, "tmp_root": str(_host_root() / "tmp")}
    return [*head, sys.executable, "-I", "-B", str(LANDLOCK_LAUNCHER), json.dumps(spec)]


def _spec_of(cmd):
    """The launcher's spec in a host-path argv (`_host_command`), or None for a bubblewrap one."""
    import json
    if len(cmd) >= 2 and cmd[-2] == str(LANDLOCK_LAUNCHER):
        return json.loads(cmd[-1])
    return None


def program_of(cmd):
    """The worker's program in a sandbox argv from `command`, with the paths it sees: everything after `--chdir DIR`
    for bubblewrap, the spec's argv for host paths."""
    spec = _spec_of(cmd)
    if spec is not None:
        return list(spec["argv"])
    cut = len(cmd) - cmd[::-1].index("--chdir") - 1
    return cmd[cut + 2:]


def with_program(cmd, program, chdir="/workspace"):
    """`cmd` (from `command`) running `program` from `chdir` instead of the worker's own: the same sandbox, binds and
    environment. Paths in `program` are the worker's view of them (`/tools/…`, `/workspace`)."""
    import json
    spec = _spec_of(cmd)
    if spec is not None:
        return [*cmd[:-1], json.dumps({**spec, "argv": list(program), "chdir": chdir})]
    cut = len(cmd) - cmd[::-1].index("--chdir") - 1
    return [*cmd[:cut], "--chdir", chdir, *program]


def worker_path(host, inside):
    """How the worker names `host`: `inside` (a bind target) in its namespaces, the host path itself otherwise."""
    return str(inside) if sandbox_mode() == "namespaces" else str(host)


#: Worker ops that never write the workspace. `worker.py` saves `animation.blend`
#: back for exactly ("initialize", "edit", "update"); these three write only under
#: /output, so the bind can enforce what the code already does rather than trust
#: it -- and a workspace nothing writes can be snapshotted by hardlink instead of
#: copied, which saves most of a run's disk.
#:
#: `probe` is deliberately absent: it evals agent-supplied code, which may
#: legitimately scribble in the workspace. Failing that loudly would be a
#: behavior change rather than an enforcement of one.
_READ_ONLY_OPS = frozenset({"render", "inspect", "evaluate", "fill_rows", "measure_rows"})


def reads_only(request) -> bool:
    """Whether this worker request leaves the workspace untouched."""
    return request.get("op") in _READ_ONLY_OPS


def worker_env(profile=None):
    """The closed environment every sandboxed Blender gets, forked or resident.

    Shared rather than duplicated because the three conditional entries are the
    ones that are silently lost when a second spawn site writes its own literal:
    the shader cache would be written into the `--tmpfs /tmp` and discarded, the
    NVIDIA EGL vendor would be left to GLVND's own search, and a render's thread
    count would fall back to the default. A resident process pays each of those
    once for its whole lifetime, which is exactly when a quiet regression is
    hardest to notice -- `resident.py` carried its own copy of this dict, with
    all three missing, until this was hoisted.

    Deliberately closed rather than derived from the daemon's environment, so a
    child cannot inherit an operator's PYTHONPATH or GL pins.
    """
    env = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/tools",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
           # Blender's user config, scripts, and add-ons: named, and fresh in
           # every process (`/tmp` is the sandbox's own tmpfs), so no operator's
           # preferences or add-ons reach a render. `HOME=/tmp` already implied
           # it; this says it, and holds even if a Blender build stops deriving
           # it from `HOME`.
           "BLENDER_USER_RESOURCES": "/tmp/blender_user"}
    if cache_dir() is not None:
        # Survives the process, unlike `HOME`. See `cache_dir`.
        env["XDG_CACHE_HOME"] = CACHE_MOUNT
        # OptiX compiles Blender's kernels at runtime and caches them in
        # `/var/tmp/OptixCache_<user>` by default, which the sandbox does not
        # bind, so every worker that rendered with OptiX recompiled them.
        env["OPTIX_CACHE_PATH"] = CACHE_MOUNT + "/optix"
        # The CUDA JIT cache defaults to `~/.nv/ComputeCache`, inside the
        # sandbox's `/tmp`. It fills only on a GPU Blender ships no compiled
        # kernel for, which then JIT-compiles them in every worker.
        env["CUDA_CACHE_PATH"] = CACHE_MOUNT + "/nvidia/ComputeCache"
    if profile is not None and "threads" in profile:
        env.update(OMP_NUM_THREADS=str(profile["threads"]),
                   LP_NUM_THREADS=str(profile["threads"]))
    gl = os.environ.get("DYCRAFT_GL_PREFIX")
    if gl:
        # The GL libraries `install tools` put where the system had none (`_home.gl_on_path`), bound read-only by
        # `command`, and the variables that point at them.
        env["LD_LIBRARY_PATH"] = str(Path(gl) / "lib")
        for name in ("LIBGL_DRIVERS_PATH", "__EGL_VENDOR_LIBRARY_DIRS", "FONTCONFIG_FILE", "VK_ICD_FILENAMES",
                     "VK_ADD_DRIVER_FILES"):
            if gl in os.environ.get(name, ""):
                env[name] = os.environ[name]
    if nvidia_usable():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    # When the sandbox holds only the nodes an absolute index list picked,
    # CUDA numbers them from 0 inside it, so the list is renumbered to match
    # (the operator's order kept); any other value means the same inside.
    gpu = arm_gpu_env()
    if gpu:
        nodes, by_index = _nvidia_selection()
        if by_index:
            position = {int(n.name[len("nvidia"):]): i for i, n in enumerate(nodes)}
            gpu["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(position[int(p)]) for p in gpu["CUDA_VISIBLE_DEVICES"].split(",") if int(p) in position)
        env.update(gpu)
    return env


def arm_gpu_env():
    """The arm's GPU for a child that runs outside the sandbox, as `h.ik`'s
    cuRobo interpreter does, and so sees every node.

    `CUDA_VISIBLE_DEVICES` as the operator or the scheduler set it, verbatim
    (`""` hides every GPU and is passed on as such), numbered in PCI order, the
    order `nvidia-smi` shows, unless the operator chose an order too; `{}` when
    the variable is unset. `worker_env` starts from the same values, so an
    index names one card in both: sandboxed Blender binds node N for index N
    (`_nvidia_selection`), and PCI order is the order of the nodes. CUDA's own
    default, fastest first, can number a node of mixed GPUs differently, so the
    IK child is given PCI order explicitly.
    """
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        return {}
    return {"CUDA_VISIBLE_DEVICES": devices,
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")}


def run_worker(workspace, output, request, *, readonly=(), cancel=None, workspace_readonly=False,
               writable=()):
    import json
    from .render_config import final_render_settings
    profile = None
    if request.get("render_settings") is not None:
        profile = final_render_settings(request["render_settings"])
        request = {**request, "render_settings": profile}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    job = output / "request.json"
    write_json(job, request)
    workspace_readonly = workspace_readonly or reads_only(request)
    env = worker_env(profile)
    with (output / "worker.log").open("w") as log:
        p = subprocess.Popen(command(workspace, output, job, readonly,
                                     workspace_readonly=workspace_readonly,
                                     writable=writable),
                             env=env, stdout=log, stderr=log)
        while p.poll() is None:
            cancelled = cancel is not None and cancel.wait(0.25)
            failure = output / "render_error.json"
            if cancelled or failure.is_file():
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                if cancelled:
                    raise RuntimeError("job cancelled")
                raise RuntimeError(json.loads(failure.read_text())["error"])
            if cancel is None:
                time.sleep(0.25)
    result_path = output / "result.json"
    if not result_path.is_file():
        raise WorkerError(f"Blender worker exited {p.returncode} during {request.get('op')}; the "
                          f"next call starts a new process. Log: {output / 'worker.log'}",
                          "harness", reason=log_tail(output / "worker.log"))
    result = json.loads(result_path.read_text())
    if p.returncode or not result.get("ok"):
        raise worker_failure(result, result_path)
    return result["result"]


class WorkerError(RuntimeError):
    """A failed worker request, with the class its reply names: the agent's own
    code (`author_code`), a refusal (`rejection`), or a fault in the harness."""

    def __init__(self, message, failure_class, reason=None):
        super().__init__(message)
        self.failure_class = failure_class
        #: What the worker said, before it was framed for the agent.
        self.reason = reason or message


def log_tail(path, size=5000):
    """The end of a worker's log: where a crashing renderer says why, which
    the reply leaves out and `GpuOutOfMemory` still has to recognize."""
    try:
        return Path(path).read_text(errors="replace")[-size:]
    except OSError:
        return ""


def harness_fault(reason, log):
    """A fault in the harness, in one line; the detail is in `log`."""
    return f"harness fault: {reason}; report it, don't change the scene for it. Log: {log}"


def worker_failure(body, log):
    """The error a worker's failed `result.json` (`worker.failure`) is raised as.
    The agent's own code failing carries what it printed first."""
    kind = body.get("failure_class") or "harness"
    error = body.get("error") or "worker failed"
    printed = Path(log).parent / "printed.txt"
    if kind == "author_code" and printed.is_file():
        error = f"{error}\n\nPrinted before it raised:\n{printed.read_text(errors='replace')}"
    return WorkerError(harness_fault(error, log) if kind == "harness" else error, kind, error)


def _sdk_error(error):
    try:
        from ...sdk._errors import DycraftError
    except ImportError:
        return False
    return isinstance(error, DycraftError)


def failed_result(error, directory, cancelled=False):
    """A failed job's `result`, the same on the batch and the interactive path.

    A worker's error names its own class. Anything else was raised in this
    process: an SDK error (`DycraftError`) is about the author's own source,
    a `ValueError` is a refusal, whose message is written for the agent;
    otherwise it is a harness fault, its traceback kept beside the job.
    """
    result = {"ok": False, "error": str(error)}
    if getattr(error, "error_class", None):
        result["error_class"] = error.error_class
    if cancelled:
        return result
    kind = getattr(error, "failure_class", None)
    if kind is None and _sdk_error(error):
        # Raised by the SDK about the author's own source (a dangling
        # reference, an invalid joint). Classed `harness` before, which tells
        # the agent not to change the scene for it: one stopped a round over
        # a `DanglingReference` its own workaround caused.
        kind = "author_code"
        if callable(getattr(error, "to_dict", None)):
            result["error_detail"] = error.to_dict()
    if kind is None:
        kind = "rejection" if isinstance(error, ValueError) else "harness"
        if kind == "harness":
            log = Path(directory) / "error.txt"
            try:
                log.write_text("".join(traceback.format_exception(
                    type(error), error, error.__traceback__)))
            except OSError:
                pass
            result["error"] = harness_fault(f"{type(error).__name__}: {error}", log)
    result["failure_class"] = kind
    return result


#: The largest job result `trace.jsonl` holds whole. Above it the trace keeps
#: the result's small fields and names the file that holds all of it: one
#: run's trace reached 1.65 GB, with single lines up to 101 MB.
TRACE_RESULT_LIMIT = 1 << 20
#: A field is kept in a traced digest when its JSON is no larger than this.
TRACE_FIELD_LIMIT = 4096


def traced_result(result, directory, *, persisted=None):
    """A job's result as `trace.jsonl` records it.

    Whole when its JSON is at most `TRACE_RESULT_LIMIT` bytes. Otherwise its
    top-level fields of at most `TRACE_FIELD_LIMIT` bytes each (`ok`,
    `failure_class`, `version`, and the like), its size, and the file holding
    it: `persisted` when the caller already wrote it there (a batch job's
    `job.json`), else `result.json`, written here, in the job's `directory`.
    """
    import json
    if not isinstance(result, dict):
        return result
    if persisted is not None:
        try:
            size = Path(persisted).stat().st_size
        except OSError:
            return result
        if size <= TRACE_RESULT_LIMIT:
            return result
        where = Path(persisted)
    else:
        text = json.dumps(result, default=str)
        size = len(text)
        if size <= TRACE_RESULT_LIMIT:
            return result
        where = Path(directory) / "result.json"
        try:
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_text(text)
        except OSError:
            where = None
    digest = {}
    for key, value in result.items():
        try:
            if len(json.dumps(value, default=str)) <= TRACE_FIELD_LIMIT:
                digest[key] = value
        except (TypeError, ValueError):
            continue
    digest["traced_digest"] = {"bytes": size, "file": str(where) if where else None}
    return digest


class Jobs:
    """Polling is read-only; cancellation reaches the worker process."""

    def __init__(self, root, trace=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.threads = []
        self.trace = trace
        #: Callables run once every job has drained; see `close`.
        self.on_close = []
        import json
        for path in self.root.glob("*/job.json"):
            record = json.loads(path.read_text())
            if record["status"] in ("queued", "running"):
                record.update(status="interrupted", result={
                    "ok": False, "error": "the harness restarted while this job ran; submit it again"})
                write_json(path, record)
            self.jobs[record["job_id"]] = (record, threading.Event())

    def submit(self, op, fn, *, step, version):
        import uuid
        ident = uuid.uuid4().hex[:16]
        record = {"job_id": ident, "operation": op, "status": "queued",
                  "step": step, "version": version, "submitted_at": time.time()}
        cancel = threading.Event()
        directory = self.root / ident
        directory.mkdir()
        def persist():
            write_json(directory / "job.json", record)
        with self.lock:
            self.jobs[ident] = (record, cancel)
            persist()
        def execute():
            with self.lock:
                record.update(status="running", started_at=time.time())
                persist()
            try:
                value = fn(directory, cancel)
                with self.lock:
                    record.update(status="completed", result=value)
            except Exception as error:
                with self.lock:
                    record.pop("waiting_for", None)
                    record.update(status="cancelled" if cancel.is_set() else "failed",
                                  result=failed_result(error, directory, cancel.is_set()))
            finally:
                with self.lock:
                    record["finished_at"] = time.time()
                    persist()
                if self.trace:
                    traced = {k: v for k, v in record.items() if k != "step"}
                    if "result" in traced:
                        traced["result"] = traced_result(traced["result"], directory,
                                                         persisted=directory / "job.json")
                    self.trace("job_result", **traced, submission_step=step)
        thread = threading.Thread(target=execute, name=f"blender-{ident}", daemon=True)
        self.threads.append(thread)
        thread.start()
        return {"ok": True, "job_id": ident, "status": "queued", "version": version}

    def register_inline(self, job_id, op, cancel, *, step, version, status="running"):
        """Make a `run_interactive` op poll- and cancel-able by its own id.

        An interactive op already has everything but this: `run_interactive`
        mints `native/jobs/<uuid4-hex-16>/`, writes `job.json` and a progress
        file into it, and journals the entry -- while `self.jobs` is populated
        only by `submit` and by recovery at daemon start. Without this, `cancel`
        would answer `unknown job_id` for the very id the harness had just
        published on disk, and an author wanting to stop one job could only
        kill the resident Blender, losing its render and the session's reply
        channel with it.

        A context manager rather than a register/unregister pair, because
        `run_interactive` has several raise paths and an entry left behind as
        `running` would be reported `interrupted` by the next daemon start.
        On the way out the record is made terminal rather than dropped, so the
        id the op returned still answers `job_status` and `job_result`
        afterwards -- the same contract a batch job has, and what an author
        polling the id they were just handed expects.
        """
        import contextlib

        record = {"job_id": job_id, "operation": op, "status": status,
                  "step": step, "version": version, "submitted_at": time.time(),
                  "started_at": time.time(), "inline": True}

        @contextlib.contextmanager
        def registered():
            with self.lock:
                self.jobs[job_id] = (record, cancel)
            outcome = "failed"
            try:
                yield record
                outcome = "completed"
            except BaseException:
                outcome = "cancelled" if cancel.is_set() else "failed"
                raise
            finally:
                with self.lock:
                    record.pop("waiting_for", None)
                    record.update(status=outcome, finished_at=time.time())

        return registered()

    def mark_waiting(self, job_id, waiting_for):
        """Report a job as queued behind something, or running again.

        A job that waits inside its own thread -- for the resident Blender, or
        for one of the arm's GPU render slots -- would otherwise read `running`
        with `queued_s` near 0 for the whole wait, and could not be told apart
        from one doing work. `waiting_for` names what it waits on; when the wait
        ends, `started_at` is re-stamped, so `queued_s` includes it.
        """
        with self.lock:
            entry = self.jobs.get(job_id)
            if entry is None:
                return
            record = entry[0]
            if record["status"] not in ("queued", "running"):
                return
            if waiting_for is not None:
                record.update(status="queued", waiting_for=waiting_for)
            else:
                record.pop("waiting_for", None)
                record.update(status="running", started_at=time.time())
            if not record.get("inline"):
                write_json(self.root / job_id / "job.json", record)

    def status(self, job_id):
        import json
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("unknown job_id")
            result = dict(self.jobs[job_id][0])
        try:
            # Sorted by mtime, not by `rglob`'s undefined order. A render job
            # writes progress from its render loop and, since `evaluate` was
            # folded into `render_frames`, from the evaluation too -- two files
            # under one job directory, of which `[-1]` picked an arbitrary one.
            progress = sorted((self.root / job_id).rglob("progress.json"),
                              key=lambda path: path.stat().st_mtime)
        except (FileNotFoundError, OSError):
            # Successful edits move their workspace while polling may still
            # be traversing it. Progress is best-effort; the job record isn't.
            progress = []
        if progress:
            try:
                result["progress"] = json.loads(progress[-1].read_text())
            except (OSError, ValueError):
                pass
        # The queue wait, which only this object knows. A rate derived from the
        # worker's own frame counter starts at the first *rendered* frame and so
        # excludes Blender start-up and the workspace copy, so a projection from
        # it alone comes in optimistic.
        if result.get("started_at") and result.get("submitted_at"):
            result["queued_s"] = round(result["started_at"] - result["submitted_at"], 2)
        return result

    def cancel_requested(self, job_id):
        """Whether `cancel` has flagged this job; its process may still be
        stopping."""
        with self.lock:
            entry = self.jobs.get(job_id)
            return bool(entry and entry[1].is_set())

    def cancel(self, job_id):
        """Set a live job's cancel flag, or say why it cannot be set.

        `'current'` resolves to the one inline op this session is running, so
        an agent that never saw an id can still stop a runaway. It deliberately
        does not match a batch job: a batch job was handed a real id at
        submission.

        A publishing op that has already entered `publish()` is refused rather
        than flagged. Stopping it there is what leaves a workspace renamed
        aside with nothing renamed back; the `cancel.is_set()` checks inside
        `publish()` cover the window before that point by declining to swap.
        """
        with self.lock:
            if job_id == "current":
                inline = [ident for ident, (record, _) in self.jobs.items()
                          if record.get("inline") and record["status"] in ("queued", "running")]
                if len(inline) != 1:
                    raise ValueError(
                        "job_cancel('current') needs exactly one interactive "
                        f"operation in flight; {len(inline)} are")
                job_id = inline[0]
            if job_id not in self.jobs:
                raise ValueError("unknown job_id")
            record, event = self.jobs[job_id]
            if record.get("publishing_committed"):
                raise ValueError(
                    f"{job_id} has begun publishing and cannot be cancelled; "
                    "cancelling mid-publish is what leaves the workspace half "
                    "swapped. Poll it instead.")
            if record["status"] in ("queued", "running"):
                event.set()
        return self.status(job_id)

    def close(self, timeout_s: float = 20.0):
        """Cancel every live job and join its thread, within `timeout_s`.

        The keyword is the daemon's drain contract, not decoration: this is
        installed as `session.drain_on_shutdown` and `agent/server.py` calls it
        as `drain(timeout_s=...)`. A positional-only `close()` raised TypeError
        into that call site's `except BaseException`, which prints and moves on
        — so nothing was canceled and nothing was joined, and a `blender` edit
        killed between `work.rename(old)` and `workspace.rename(work)` left the
        workspace missing with `native/previous` holding the only copy.
        """
        with self.lock:
            for record, event in self.jobs.values():
                if record["status"] in ("queued", "running"):
                    event.set()
        deadline = time.monotonic() + timeout_s
        for thread in self.threads:
            thread.join(max(0, deadline-time.monotonic()))
        # Anything holding a process for the life of the session, which is now
        # a real category: a resident Blender outlives every individual job and
        # has no other moment at which the harness is known to be finished with
        # it. `--die-with-parent` is the backstop, not the plan.
        for hook in list(self.on_close):
            try:
                hook()
            except BaseException:  # noqa: BLE001 -- shutdown must not raise
                traceback.print_exc()
