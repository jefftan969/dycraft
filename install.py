"""Set up what DyCraft needs beyond its Python dependencies, from a clone or an installed package.

    python -m dycraft.install blender   the Blender env (Python 3.11, bpy 4.5) and the official Blender 4.5.13 binary
    python -m dycraft.install ik        the cuRobo env that h.ik runs in (needs an NVIDIA GPU with CUDA)
    python -m dycraft.install tools     bubblewrap, socat, ffmpeg, and the GL libraries, for those missing (no sudo,
                                        no system conda)
    python -m dycraft.install cad       record an interpreter with build123d 0.11.1 (this one, by default)
    python -m dycraft.install check     the environment check (environment_test.py; its options follow)
    python -m dycraft.install mocap     optional: the CMU motion-capture library (--dog adds the MANN dog captures)
    python -m dycraft.install library   the reviewed asset library and the G1's description (again to update it)
    python -m dycraft.install films     optional: the paper's 24 films (--brief B05,B13 for some)

The blender, ik, and cad steps record what they set up in a pointer file (.blender_python, .blender_binary, .ik_python,
.b123d_python), mocap writes assets/mocap/, library links assets/imported/ and assets/robots/g1_description from its
mirror of the dycraft-assets dataset in assets/.dataset/, and films links films/ from its mirror of the dycraft-films
dataset in .datasets/dycraft-films/. All of them go in the first of these directories, which is also where DyCraft
reads them: $DYCRAFT_HOME when set; the clone, when DyCraft runs from one; otherwise the directory you run this from.
So run setup from where you will launch DyCraft.

The blender and ik envs are separate environments: bpy 4.5 needs Python 3.11 and NumPy < 2, and cuRobo needs its
own PyTorch, so neither can share the main environment. Each is made with conda when this runs from a conda
environment, and with uv otherwise (`--conda` or `--uv` to choose), from the same specs (environment-blender.yml,
environment-ik.yml), which ship with the package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

#: The official build fluid needs: bpy's wheel is built as a Python module, and Mantaflow's bindings do not work in
#: that configuration, so fluid domains only bake under a real binary (render/_blender_env.py, blender_binary).
BLENDER_VERSION = "4.5.13"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
#: From https://download.blender.org/release/Blender4.5/blender-4.5.13.sha256
BLENDER_SHA256 = "da4e69b06b75b9e642d106496c50e7e240218b411d2f6e18271c1d1d819cef91"

#: cuRoboV2 (upstream 0.8.0 renamed MotionGen -> MotionPlanner and moved the implementation under curobo/_src/). This
#: is a post-0.8.0 commit with no matching published wheel, so the commit is pinned. Bump it deliberately and re-run
#: the expensive drift test (tests/test_ik_adapter.py), which says whether animation/ik/CUROBO.md still describes it.
CUROBO_REPO = "https://github.com/NVlabs/curobo.git"
CUROBO_COMMIT = "8e734f3"

#: PyTorch wheel indexes, chosen by the GPU. No single PyTorch build has kernels from Pascal to Blackwell: the CUDA 12.6
#: builds carry sm_50-sm_90 (Maxwell to Hopper), the CUDA 12.8 builds sm_75-sm_120 (Turing to Blackwell), and PyPI's
#: default (CUDA 13) sm_75 up. A build without kernels for the GPU still imports and reports CUDA available, then fails
#: its first kernel with "no kernel image is available for execution on the device". Both indexes are CUDA 12,
#: like the rest of the IK env's CUDA stack (setup_ik), which also keeps the driver requirement at CUDA 12's rather than
#: CUDA 13's R580. Any version on the chosen index works (cuRobo asks for torch>=2.5), so the newest one is taken.
TORCH_INDEX = "https://download.pytorch.org/whl/"
TORCH_BUILDS = {"cu126": "sm_50-sm_90: Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper",
                "cu128": "sm_75-sm_120: Turing, Ampere, Ada, Hopper, Blackwell"}
#: Warp's CUDA 12 builds (setup_ik says why).
WARP_RULE = "<1.18"


def _run(argv: list, **kw) -> subprocess.CompletedProcess:
    """Every setup child (conda, pip, git, probe interpreters) runs at the worker OOM tier: if the machine runs out of
    memory, the kernel takes the setup step rather than the session that launched it."""
    from . import _oom
    return subprocess.run(_oom.tiered([str(a) for a in argv], _oom.WORKER), **kw)


def _missing_library(text: str) -> str | None:
    """What to do when an import failed because the loader found no system library ("libXrender.so.1: cannot open
    shared object file"), which no reinstall fixes; None for any other failure."""
    import re
    m = re.search(r"(lib[\w.+-]*?\.so[\w.]*): cannot open shared object file", text or "")
    if not m:
        return None
    return (f"this machine lacks {m.group(1)}, a system library it loads. Without sudo, `python -m dycraft.install "
            "tools` installs the GL and X11 libraries; with root, install the system packages `python -m dycraft.install "
            "check` names for it. Then run this again.")


def _say(step: str, msg: str) -> None:
    print(f"[install {step}] {msg}", flush=True)


def state_path(name: str) -> Path:
    from . import _home
    return _home.state_path(name)


def write_pointer(name: str, value: str, step: str, consumer: str, *, comment: bool = True) -> Path:
    """Record `value` in the pointer file `name`. `.blender_binary` is read whole as a path, so it takes no comment."""
    path = state_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# written by `python -m dycraft.install {step}`; read by {consumer}\n" if comment else ""
    path.write_text(header + value + "\n")
    _say(step, f"recorded {path}")
    return path


def _conda() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if not exe:
        raise SystemExit("error: conda not found. Install miniconda or mamba, or set CONDA_EXE to its binary.")
    return exe


def env_route(force: str | None = None, prefix: str | None = None, conda_prefix: str | None = None,
              uv: str | None = None, conda: str | None = None) -> tuple[str, str]:
    """`("conda" | "uv", why)`: how to make the blender and ik environments.

    `--conda` or `--uv` decides. Otherwise the interpreter running this decides, not the shell: conda when it is itself
    a conda environment (`sys.prefix == $CONDA_PREFIX`, or `sys.prefix/conda-meta/` exists), so a uv venv picks uv even
    with conda's base environment active in every shell; uv otherwise, when it is on PATH; then conda, if found.
    """
    if force:
        return force, f"--{force}"
    prefix = sys.prefix if prefix is None else prefix
    conda_prefix = os.environ.get("CONDA_PREFIX") if conda_prefix is None else conda_prefix
    if (conda_prefix and Path(prefix).resolve() == Path(conda_prefix).resolve()) or (Path(prefix) / "conda-meta").is_dir():
        return "conda", f"{prefix} is a conda environment"
    uv = shutil.which("uv") if uv is None else uv
    if uv:
        return "uv", f"{prefix} is not a conda environment"
    conda = (os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")) if conda is None else conda
    if conda:
        return "conda", "uv is not on PATH"
    raise SystemExit("error: neither uv nor conda was found. Install uv (curl -LsSf https://astral.sh/uv/install.sh | "
                     "sh) or conda, then run this again.")


def _spec(spec: str) -> tuple[str, list[str]]:
    """The Python request and pip packages of a packaged environment spec, for making it with uv instead of conda."""
    python, pip, in_pip = "", [], False
    for raw in (PACKAGE / spec).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        item = line.strip()[2:].strip() if line.strip().startswith("- ") else ""
        if not item:
            continue
        if item == "pip:":
            in_pip = True
        elif in_pip and line.startswith("      "):
            pip.append(item)
        elif item.startswith("python"):
            rest = item[len("python"):]
            python = rest[1:] if rest.startswith("=") and not rest.startswith("==") else rest   # conda's `=3.11`
            in_pip = False
    return python, pip


def uv_env(step: str, spec: str, prefix: Path) -> Path:
    """The interpreter of a uv venv at `prefix`, made from the packaged `spec`'s Python and pip packages if it does not
    exist. Seeded with pip, so the rest of the step runs the same as on a conda environment."""
    py = prefix / "bin" / "python"
    if py.exists():
        _say(step, f"reusing the env at {prefix}")
        return py
    python, packages = _spec(spec)
    _say(step, f"creating the env at {prefix} with uv (Python {python}) (needs network)")
    _run(["uv", "venv", "--seed", "--python", python, str(prefix)], check=True)
    if packages:
        _run([str(py), "-m", "pip", "install", *packages], check=True)
    return py


def make_env(step: str, spec: str, prefix: Path, route: str) -> Path:
    return uv_env(step, spec, prefix) if route == "uv" else conda_env(step, spec, prefix)


def conda_env(step: str, spec: str, prefix: Path) -> Path:
    """The interpreter of the conda env at `prefix`, created from the packaged `spec` if it does not exist."""
    py = prefix / "bin" / "python"
    if py.exists():
        _say(step, f"reusing the env at {prefix}")
    else:
        _say(step, f"creating the env at {prefix} from {spec} (needs network)")
        _run([_conda(), "env", "create", "-p", str(prefix), "-f", str(PACKAGE / spec)], check=True)
    if not py.exists():
        raise SystemExit(f"error: no interpreter at {py}")
    return py


# ------------------------------------------------------------------------------------------------ tools

#: micromamba, which `install tools` uses to get bubblewrap, socat and ffmpeg from conda-forge with no system conda
#: and no sudo: one static file, pinned by version and by the sha256 its release publishes (checked 11-08).
MICROMAMBA_VERSION = "2.10.0-0"
MICROMAMBA_SHA256 = {"64": "701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd",
                     "aarch64": "3bb56786e21b9b24024f599fe9185b4a59dc3f9f59937a42311ee583adbbb23c"}
MICROMAMBA_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/{version}/micromamba-linux-{arch}"

#: The system tools DyCraft needs, and the conda-forge package that provides each.
TOOLS = {"bwrap": "bubblewrap", "socat": "socat", "ffmpeg": "ffmpeg", "ffprobe": "ffmpeg"}


#: The GL and X11 libraries Blender, build123d and open3d load, by soname, and the conda-forge package with each. A
#: cloud image without a desktop has none of them.
GL_LIBRARIES = {"libEGL.so.1": "libegl", "libGL.so.1": "libgl", "libGLX.so.0": "libglx", "libOpenGL.so.0": "libopengl",
                "libX11.so.6": "xorg-libx11", "libXext.so.6": "xorg-libxext", "libXrender.so.1": "xorg-libxrender",
                "libXi.so.6": "xorg-libxi", "libXfixes.so.3": "xorg-libxfixes", "libSM.so.6": "xorg-libsm",
                "libICE.so.6": "xorg-libice", "libXt.so.6": "xorg-libxt",
                # open3d 0.20's offscreen renderer draws through Vulkan; Mesa's lavapipe is its software driver.
                "libvulkan.so.1": "libvulkan-loader"}
#: Where an EGL implementation announces itself to GLVND (the system's Mesa or a GPU driver).
EGL_VENDOR_DIRS = ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d")


def _micromamba(step: str) -> Path:
    arch = {"x86_64": "64", "aarch64": "aarch64"}.get(platform.machine())
    if not arch:
        raise SystemExit(f"error: no micromamba build for {platform.machine()}; install what `{step}` needs with your "
                         "system's package manager")
    bin_dir = state_path("bin")
    bin_dir.mkdir(parents=True, exist_ok=True)
    micromamba = download(MICROMAMBA_URL.format(version=MICROMAMBA_VERSION, arch=arch), bin_dir / "micromamba",
                          MICROMAMBA_SHA256[arch], step)
    micromamba.chmod(0o755)
    return micromamba


def missing_gl() -> tuple[list[str], bool]:
    """The GL and X11 libraries (`GL_LIBRARIES`) the loader cannot find here, and whether an EGL implementation is
    missing too (no GLVND vendor file and no Mesa EGL), which only conda-forge's Mesa (`mesalib`) then provides."""
    import ctypes
    missing = []
    for soname in GL_LIBRARIES:
        try:
            ctypes.CDLL(soname)
        except OSError:
            missing.append(soname)
    vendors = any(Path(d).is_dir() and any(Path(d).glob("*.json")) for d in EGL_VENDOR_DIRS)
    try:
        ctypes.CDLL("libEGL_mesa.so.0")
        mesa = True
    except OSError:
        mesa = False
    return missing, not (vendors or mesa)


def setup_gl(force: bool = False) -> int:
    """The GL and X11 libraries, and Mesa's software renderer, from conda-forge into DyCraft's `envs/gl`, where the
    system has none: no sudo. DyCraft puts them on its children's loader path (`.gl_prefix`, `_home.tools_on_path`),
    and binds them into Blender's sandbox."""
    missing, no_egl = missing_gl()
    if force:
        missing, no_egl = sorted(GL_LIBRARIES), True
    if not missing and not no_egl:
        _say("tools", "the GL and X11 libraries Blender and the renderers load are installed already")
        return 0
    packages = sorted({*(GL_LIBRARIES[m] for m in missing), *(("mesalib",) if no_egl else ())})
    micromamba = _micromamba("tools")
    prefix = state_path("envs/gl")
    _say("tools", f"this machine is missing {', '.join(missing) or 'an EGL implementation'}: installing "
                  f"{', '.join(packages)} from conda-forge into {prefix} (needs network"
                  f"{', about 400 MB with Mesa' if no_egl else ''})")
    env = {**os.environ, "MAMBA_ROOT_PREFIX": str(state_path("envs/.micromamba"))}
    _run([str(micromamba), "create", "-y", "-q", "-p", str(prefix), "-c", "conda-forge", *packages], check=True,
         env=env)
    write_pointer(".gl_prefix", str(prefix), "tools", "the launcher, which puts its lib/ on its children's loader path")
    _say("tools", f"installed into {prefix}; DyCraft's children (Blender, the CAD worker, the renderers) load them "
                  "from there")
    return 0


def setup_tools(force: bool = False) -> int:
    """bubblewrap, socat, and ffmpeg, for whichever are not already on PATH, into DyCraft's `envs/tools`, linked into its
    `bin/`, which the launcher and the check add to the end of PATH (`.tools_bin`, `_home.tools_on_path`); then the
    GL libraries, where they are missing (`setup_gl`)."""
    missing = sorted(TOOLS) if force else [t for t in sorted(TOOLS) if not shutil.which(t)]
    bin_dir = state_path("bin")
    if not missing:
        _say("tools", "bwrap, socat, ffmpeg and ffprobe are all on PATH already; nothing to install")
        return setup_gl(force)
    micromamba = _micromamba("tools")
    prefix = state_path("envs/tools")
    packages = sorted({TOOLS[t] for t in missing})
    _say("tools", f"installing {', '.join(packages)} from conda-forge into {prefix} (needs network)")
    env = {**os.environ, "MAMBA_ROOT_PREFIX": str(state_path("envs/.micromamba"))}
    _run([str(micromamba), "create", "-y", "-q", "-p", str(prefix), "-c", "conda-forge", *packages], check=True,
         env=env)
    for tool in missing:
        link = bin_dir / tool
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(prefix / "bin" / tool)
    write_pointer(".tools_bin", str(bin_dir), "tools", "the launcher and the check, which add it to PATH")
    _say("tools", f"installed {', '.join(missing)} into {bin_dir}; DyCraft adds it to the end of PATH, so a tool "
                  "of your own, installed later, still comes first")
    return setup_gl(force)


# ------------------------------------------------------------------------------------------------ blender


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str, step: str) -> Path:
    """Fetch `url` to `dest`, verifying `sha256`; an existing `dest` with the right digest is reused."""
    if dest.exists() and _sha256(dest) == sha256:
        _say(step, f"reusing {dest}")
        return dest
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    _say(step, f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total, done, shown = int(r.headers.get("Content-Length") or 0), 0, 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and done - shown >= total // 10:
                shown = done
                _say(step, f"  {done * 100 // total}% of {total / 1e6:.0f} MB")
    if h.hexdigest() != sha256:
        part.unlink()
        raise SystemExit(f"error: {url} does not match its published sha256 ({h.hexdigest()} != {sha256})")
    part.rename(dest)
    return dest


def blender_version(binary: Path) -> str | None:
    try:
        out = _run([str(binary), "--version"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    return first.split()[1] if first.startswith("Blender ") and len(first.split()) > 1 else None


def fetch_blender(parent: Path, step: str = "blender") -> Path:
    """The official Blender binary under `parent`, downloaded, verified, and unpacked if not already there."""
    if not (sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")):
        raise SystemExit(f"error: the official build fetched here is Linux x86-64. On this platform, install Blender "
                         f"{BLENDER_VERSION} from blender.org and record it with `python -m dycraft.install check`.")
    binary = parent / BLENDER_ARCHIVE.removesuffix(".tar.xz") / "blender"
    if binary.exists() and blender_version(binary) == BLENDER_VERSION:
        _say(step, f"reusing Blender {BLENDER_VERSION} at {binary}")
        return binary
    parent.mkdir(parents=True, exist_ok=True)
    archive = download(BLENDER_URL, parent / BLENDER_ARCHIVE, BLENDER_SHA256, step)
    _say(step, f"unpacking {archive.name} into {parent}")
    with tarfile.open(archive, "r:xz") as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(parent, filter="data")
        else:  # Python without extraction filters (< 3.10.12)
            t.extractall(parent)
    got = blender_version(binary)
    if got != BLENDER_VERSION:
        raise SystemExit(f"error: {binary} reports Blender {got}, expected {BLENDER_VERSION}")
    archive.unlink()
    return binary


_GPU_PROBE = r"""
import bpy
prefs = bpy.context.preferences.addons.get("cycles")
if prefs is None:
    print("  note: cycles addon unavailable; renders will be CPU-only")
else:
    found = []
    for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            prefs.preferences.compute_device_type = backend
            prefs.preferences.get_devices()
        except (TypeError, AttributeError):
            continue
        names = [d.name for d in prefs.preferences.devices if d.type == backend]
        if names:
            found.append(f"{backend}: {', '.join(names)}")
    print("  gpu: " + ("; ".join(found) if found else "none detected; Cycles will render on CPU"))
"""


def setup_blender(prefix: Path, binary_parent: Path | None, route: str = "conda") -> int:
    py = make_env("blender", "environment-blender.yml", prefix, route)
    _say("blender", "checking that the env imports bpy")
    ok = _run([str(py), "-c", "import bpy, numpy; print('  ok: blender', bpy.app.version_string, "
                                        "'| numpy', numpy.__version__)"], capture_output=True, text=True)
    print(ok.stdout, end="")
    if ok.returncode != 0:
        lib = _missing_library(ok.stderr)
        if lib:
            raise SystemExit(f"error: the env at {prefix} has bpy, but cannot import it: {lib}")
        last = (ok.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise SystemExit(f"error: the env at {prefix} cannot import bpy ({last[0]}); re-run, or remove it and run "
                         "again.")
    # Which device the renderer gets decides seconds against minutes per frame; better learned now than from a
    # run that seems to hang. Import only: picking a device is each render child's job.
    _run([str(py), "-c", _GPU_PROBE])
    write_pointer(".blender_python", str(py), "blender", "render/_blender_env.py (the bpy module)")
    binary = fetch_blender(binary_parent or prefix / "opt")
    write_pointer(".blender_binary", str(binary), "blender", "render/_blender_env.py", comment=False)
    _say("blender", "done. Select the Blender renderer per run with `python -m dycraft.cli --render-backend blender "
                    "...`; DYCRAFT_BLENDER_PYTHON and DYCRAFT_BLENDER_BINARY override the pointers for one run.")
    return 0


# ------------------------------------------------------------------------------------------------ ik


def gpu_capabilities() -> list[tuple[int, int]]:
    """Compute capabilities of the GPUs this host shows, e.g. [(6, 1)] for a GTX 1070; [] when there is no driver."""
    try:
        got = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    except OSError:
        return []
    caps = []
    for line in got.stdout.split() if got.returncode == 0 else []:
        major, _, minor = line.strip().partition(".")
        if major.isdigit() and minor.isdigit():
            caps.append((int(major), int(minor)))
    return caps


def torch_index(caps: list[tuple[int, int]]) -> str:
    """The PyTorch index whose builds have kernels for every GPU in `caps` (see TORCH_BUILDS)."""
    if not caps:
        raise SystemExit("error: no NVIDIA GPU found (nvidia-smi lists none). cuRobo has no CPU path, so h.ik needs a "
                         "GPU and a driver; pass --torch-index to install anyway.")
    if min(caps) < (5, 0):
        raise SystemExit(f"error: sm_{min(caps)[0]}{min(caps)[1]} is older than any current PyTorch build supports "
                         f"(sm_50 is the oldest).")
    if max(caps) >= (10, 0):
        if min(caps) < (7, 5):
            raise SystemExit("error: this host mixes Blackwell with GPUs older than Turing, and no PyTorch build has "
                             "kernels for both. Pass --torch-index for the GPU h.ik will use: "
                             f"{TORCH_INDEX}cu126 ({TORCH_BUILDS['cu126']}) or {TORCH_INDEX}cu128 "
                             f"({TORCH_BUILDS['cu128']}).")
        return TORCH_INDEX + "cu128"
    return TORCH_INDEX + "cu126"


def install_torch(py: Path, index: str) -> str:
    """The newest torch on `index`, then its dependencies; returns the installed version.

    One `pip install torch --extra-index-url <index>` cannot do this: pip takes the highest version on either index, and
    PyPI's (CUDA 13, sm_75 up) is usually higher. Nor can `--index-url <index>` alone: the PyTorch indexes lack some of
    torch's dependencies (nvidia-cudnn-cu12 9.10 on cu126, for one). So: the build first, without dependencies; then
    the dependencies, with that exact build (local version label and all) held in place.
    """
    pip = [str(py), "-m", "pip", "install"]
    _run(pip + ["--no-deps", "--index-url", index, "torch"], check=True)
    got = _run([str(py), "-c", "import importlib.metadata as m; print(m.version('torch'))"],
               capture_output=True, text=True, check=True)
    version = got.stdout.strip()
    _run(pip + [f"torch=={version}", "--extra-index-url", index], check=True)
    return version


def probe_ik(py: Path) -> dict:
    """environment_test's kernel probe under `py`."""
    from .environment_test import IK_PROBE
    got = _run([str(py), "-c", IK_PROBE], capture_output=True, text=True, timeout=300)
    for line in got.stdout.splitlines():
        if line.startswith("ENVTEST_JSON "):
            return json.loads(line[len("ENVTEST_JSON "):])
    return {"error": ((got.stderr or "").strip().splitlines() or ["no output"])[-1]}


#: What `install ik` says on a machine with no NVIDIA GPU, where it installs nothing and exits 0.
IK_SKIPPED = ("skipped — cuRobo needs an NVIDIA GPU with a driver, and this machine shows none (nvidia-smi lists no "
              "GPU). IK is optional: DyCraft runs without it, and the agent solves kinematics itself. Pass "
              "--torch-index to install anyway.")


def setup_ik(prefix: Path, curobo_src: Path | None, index: str | None = None, route: str = "conda") -> int:
    caps = gpu_capabilities()
    index = index or torch_index(caps)
    py = make_env("ik", "environment-ik.yml", prefix, route)
    _say("ik", f"GPUs {', '.join(f'sm_{a}{b}' for a, b in caps) or 'none found'}: PyTorch from {index}")
    _say("ik", f"installed torch {install_torch(py, index)}")
    src = curobo_src or prefix / "src" / "curobo"
    if not (src / ".git").exists():
        _say("ik", f"cloning cuRobo at {CUROBO_COMMIT} into {src}")
        src.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", CUROBO_REPO, str(src)], check=True)
        _run(["git", "-C", str(src), "checkout", CUROBO_COMMIT], check=True)
    pip = [str(py), "-m", "pip", "install"]
    # The rest of the CUDA stack on CUDA 12 as well, installed before cuRobo so its own requirements are already met:
    # - cuda-core[cu12], the kernel backend: without one, the first forward-kinematics call dies with "No curobo
    #   kernel backend available!". cuda.core is the no-compilation path upstream recommends.
    # - warp-lang, which cuRobo imports: from 1.18 its PyPI wheels are built with CUDA 13, which needs sm_75 and an
    #   R580 driver. On a GTX 1070 cuRobo then fails with "CUDA 13.4 requires sm_75 or higher, but sm_61 (device
    #   cuda:0) was specified". 1.17 is the last CUDA 12 line on PyPI (later ones ship +cu12 wheels on GitHub only).
    _run(pip + ["--no-cache-dir", "cuda-core[cu12]", f"warp-lang{WARP_RULE}"], check=True)
    _say("ik", f"installing cuRobo from {src}")
    # Editable, so the source tree must stay where it is; by default it lives inside the env it belongs to.
    _run(pip + ["-e", str(src), "--no-build-isolation"], check=True)
    # curobo.robot_builder imports this; its error names `liblzfse`, which is not the PyPI name.
    _run(pip + ["--no-cache-dir", "pyliblzfse"], check=True)
    # Prove it on the GPU before recording it: an import and torch.cuda.is_available() both pass with a torch that
    # has no kernels for this GPU, and h.ik would then fail on every call.
    _say("ik", "checking: a kernel on the GPU, then a cuRobo IK round trip through h.ik (FK of a pose, IK back)")
    got = probe_ik(py)
    if got.get("kernel"):
        from .environment_test import solve_ik
        got.update(solve_ik(py))
    if not got.get("kernel") or not got.get("solved"):
        why = got.get("error") or (f"IK did not converge (error {got.get('error_mm', float('nan')):.3f} mm)"
                                    if got.get("kernel") else "no CUDA device")
        raise SystemExit(f"error: torch {got.get('torch', '?')} (kernels {', '.join(got.get('arch', [])) or '?'}) on "
                         f"{got.get('gpu', 'no GPU')} ({got.get('cap', '?')}): {why.rstrip('.')}. Not recording "
                         f".ik_python.")
    _say("ik", f"ok: {got['gpu']} ({got['cap']}), torch {got['torch']}, cuRobo IK error {got['error_mm']:.3f} mm")
    write_pointer(".ik_python", str(py), "ik", "animation/ik/ik.py")
    _say("ik", "done. DYCRAFT_IK_PYTHON overrides the pointer for one run.")
    return 0


# ------------------------------------------------------------------------------------------------ cad


def setup_cad(python: str) -> int:
    """Record an interpreter that has build123d 0.11.1 (the main env carries it; this interpreter by default)."""
    # Probed the way the CAD child runs it: with the interpreter's own lib/ ahead of LD_LIBRARY_PATH. On hosts whose
    # system libexpat is older than conda's, OCP loads the system one first and pyexpat then fails to import;
    # _cad_env.b123d_child_env exists for that, and a probe without it rejects a working interpreter.
    from .agent.cad._cad_env import _lib_prepended_env
    got = _run([python, "-c", "import importlib.metadata as m, build123d; print(m.version('build123d'))"],
               capture_output=True, text=True, env=_lib_prepended_env(python))
    version = got.stdout.strip()
    if got.returncode != 0:
        # Say what failed: "not installed" and "installed but cannot import" need different fixes.
        last = (got.stderr or "").strip().splitlines()[-1:] or ["no output"]
        lib = _missing_library(got.stderr)
        if lib:
            raise SystemExit(f"error: {python} has build123d, but cannot import it: {lib}")
        raise SystemExit(f"error: {python} cannot import build123d: {last[0]}\n"
                         f"The object and scene agents need build123d 0.11.1 (the main env, or environment-b123d.yml).")
    if version != "0.11.1":
        raise SystemExit(f"error: {python} has build123d {version}; the object and scene agents need exactly 0.11.1 "
                         f"(the main env, or environment-b123d.yml).")
    write_pointer(".b123d_python", python, "cad", "agent/cad/_cad_env.py")
    return 0


# ------------------------------------------------------------------------------------------------ main


def _route_flags(parser) -> None:
    how = parser.add_mutually_exclusive_group()
    how.add_argument("--conda", dest="route", action="store_const", const="conda",
                     help="make the env with conda (default inside a conda environment)")
    how.add_argument("--uv", dest="route", action="store_const", const="uv",
                     help="make the env with uv (default otherwise)")


def _env_prefix(args, name: str, env_var: str) -> tuple[Path, str]:
    """The env's prefix and route: --prefix, then $BLENDER_PREFIX / $IK_PREFIX, then the route's own default."""
    route, why = env_route(args.route)
    _say(name, f"using {route} ({why})")
    if args.prefix is not None:
        return args.prefix.expanduser(), route
    if os.environ.get(env_var):
        return Path(os.environ[env_var]).expanduser(), route
    return (state_path(f"envs/{name}") if route == "uv"
            else Path(f"~/conda_envs/dycraft-{name}").expanduser()), route


def main(argv: list[str] | None = None) -> int:
    # The tools and GL libraries an earlier `install tools` put in place, for the import tests below to load.
    from ._home import tools_on_path
    tools_on_path()
    ap = argparse.ArgumentParser(prog="python -m dycraft.install", description=__doc__.split("\n\n")[0],
                                 epilog="Pointer files go to $DYCRAFT_HOME, the checkout, or the current directory.")
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("blender", help="the Blender env and the official Blender 4.5.13 binary")
    b.add_argument("--prefix", type=Path, default=None,
                   help="where the env goes ($BLENDER_PREFIX; default envs/blender in DyCraft's folder with uv, "
                        "~/conda_envs/dycraft-blender with conda)")
    _route_flags(b)
    b.add_argument("--binary-dir", type=Path, default=None,
                   help="where to unpack the official binary (default: <prefix>/opt)")
    k = sub.add_parser("ik", help="the cuRobo env for h.ik (needs CUDA)")
    k.add_argument("--prefix", type=Path, default=None,
                   help="where the env goes ($IK_PREFIX; default envs/ik in DyCraft's folder with uv, "
                        "~/conda_envs/dycraft-ik with conda)")
    _route_flags(k)
    k.add_argument("--curobo-src", type=Path, default=Path(os.environ["CUROBO_SRC"]) if os.environ.get("CUROBO_SRC")
                   else None, help="reuse an existing cuRobo checkout (default: clone into <prefix>/src/curobo)")
    k.add_argument("--torch-index", default=None,
                   help=f"the PyTorch wheel index (default: by GPU, {TORCH_INDEX}cu126 up to Hopper, cu128 for "
                        f"Blackwell)")
    t = sub.add_parser("tools", help="bubblewrap, socat, ffmpeg, and missing GL libraries, without sudo or a system "
                                     "conda")
    t.add_argument("--force", action="store_true", help="install all of them, even those the system has")
    c = sub.add_parser("cad", help="record an interpreter with build123d 0.11.1")
    c.add_argument("--python", default=sys.executable)
    sub.add_parser("check", help="the environment check; pass its options after `check`", add_help=False)
    m = sub.add_parser("mocap", help="optional: the CMU motion-capture library, into assets/mocap/")
    m.add_argument("--dog", action="store_true", help="also the MANN dog captures (CC BY-NC 4.0; asks you to accept)")
    m.add_argument("--accept-cc-by-nc", action="store_true", help="accept the dog captures' terms without being asked")
    lib = sub.add_parser("library", help="the reviewed asset library, into assets/imported/ (run again to update it)")
    lib.add_argument("--names", help="comma-separated characters (default: every one in the library)")
    lib.add_argument("--category", choices=("person", "animal", "robot"), help="only this category")
    lib.add_argument("--revision", help="a dataset tag or commit to pin")
    lib.add_argument("--prune", action="store_true", help="unlink characters a newer library no longer lists")
    fl = sub.add_parser("films", help="optional: the paper's films, into films/")
    fl.add_argument("--brief", help="comma-separated brief ids, e.g. B05,B13 (default: all)")
    fl.add_argument("--revision", help="a dataset tag or commit to pin")
    args, rest = ap.parse_known_args(argv)
    if args.step == "check":
        from . import environment_test
        return environment_test.main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.step == "tools":
        return setup_tools(args.force)
    if args.step == "blender":
        prefix, route = _env_prefix(args, "blender", "BLENDER_PREFIX")
        return setup_blender(prefix, args.binary_dir, route)
    if args.step == "ik":
        if not args.torch_index and not gpu_capabilities():
            # Optional, so not an error: a scripted install carries on, and a run without IK solves kinematics itself.
            # Before `_env_prefix`, which picks the env's place and says which route it takes: nothing is made.
            _say("ik", IK_SKIPPED)
            return 0
        prefix, route = _env_prefix(args, "ik", "IK_PREFIX")
        return setup_ik(prefix, args.curobo_src, args.torch_index, route)
    if args.step in ("library", "films"):
        from .agent.assets import _fetch
        def split(v):
            return [x.strip() for x in v.split(",") if x.strip()] if v else None
        if args.step == "library":
            return _fetch.run_library(split(args.names), args.category, args.revision, args.prune)
        return _fetch.run_films(split(args.brief), args.revision)
    if args.step == "mocap":
        from .animation.mocap import _fetch
        return _fetch.run(dog=args.dog, accept=args.accept_cc_by_nc)
    return setup_cad(args.python)


if __name__ == "__main__":
    # Before anything loads GL: the libraries `install tools` added are found only by a process started with them.
    from ._home import reexec_for_loader
    reexec_for_loader()
    sys.exit(main())
