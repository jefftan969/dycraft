"""Where DyCraft keeps user state: interpreter pointer files, assets, runs.

The root, `home()`, is the first of:

1. `$DYCRAFT_HOME`, when set;
2. the source tree this package runs from (a clone, a worktree, an editable
   install, or an exported copy), which keeps its state beside its code;
3. the launch directory, for an installed copy (one under `site-packages` or
   `dist-packages`). An installed package directory is not a place to write,
   and user state belongs beside the work it serves rather than in a hidden
   per-user directory.

"Installed" is decided by the package's own path rather than by looking for
`.git` or `pyproject.toml`: an exported tree has no `.git`, and a stray
`pyproject.toml` could end up packaged.

Stdlib only, and it imports nothing else from dycraft: setup and install code
reads it before anything heavy is loaded, and `_cad_env` loads it by file path
when it is itself loaded outside the package.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DYCRAFT_HOME"

_PACKAGE_DIR = Path(__file__).resolve().parent
_INSTALL_DIRS = frozenset({"site-packages", "dist-packages"})


def installed() -> bool:
    """True when this package is an installed copy rather than a source tree."""
    return any(part in _INSTALL_DIRS for part in _PACKAGE_DIR.parts)


def home() -> Path:
    """The user-state root (see the module docstring). Not created here."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    if not installed():
        return _PACKAGE_DIR
    return Path.cwd().resolve()


def state_path(*parts: str) -> Path:
    """A path under `home()`."""
    return home().joinpath(*parts)


def default_runs_dir() -> Path:
    """Where runs go when no `--runs-dir` is given: `home()/runs`."""
    return state_path("runs")


__all__ = ["ENV_VAR", "installed", "home", "state_path", "default_runs_dir"]


def tools_on_path() -> bool:
    """Add the tools `python -m dycraft.install tools` installed (`.tools_bin`, under `home()`) to the end of this
    process's PATH, which every child inherits: the agent CLIs and the sandboxes look bwrap, socat and ffmpeg up there.
    Last, so they only fill gaps. A system tool installed later wins, as the system's /usr/bin/bwrap must on Ubuntu
    24.04, whose AppArmor lets only it create user namespaces. Then the GL libraries it installed (`gl_on_path`).
    True when that changed the loader path, which this process's own loader read at its start (`reexec_for_loader`)."""
    _tools_bin_on_path()
    return gl_on_path()


#: Set in the environment of a process `reexec_for_loader` restarted, so it restarts once.
_LOADER_REEXEC_ENV = "DYCRAFT_LOADER_PATH_SET"


def reexec_for_loader() -> None:
    """Restart this process once, the same command, when `tools_on_path` put GL libraries on the loader path: the
    dynamic loader reads LD_LIBRARY_PATH when a process starts, so only a new process (or a child) finds them, and
    an entry point that loads GL itself (the check's renders, open3d) needs to be one. For `__main__` only."""
    if not tools_on_path() or os.environ.get(_LOADER_REEXEC_ENV) == "1":
        return
    import sys
    os.environ[_LOADER_REEXEC_ENV] = "1"
    argv = list(sys.orig_argv)
    os.execv(sys.executable, [sys.executable, *argv[1:]])


def _tools_bin_on_path() -> None:
    try:
        lines = state_path(".tools_bin").read_text(errors="replace").splitlines()
    except OSError:
        return
    value = next((line.strip() for line in lines if line.strip() and not line.strip().startswith("#")), "")
    if not value:
        return
    tools = Path(value).expanduser()
    path = os.environ.get("PATH", "").split(os.pathsep)
    if tools.is_dir() and str(tools) not in path:
        os.environ["PATH"] = os.pathsep.join([*[p for p in path if p], str(tools)])


#: Where an EGL implementation announces itself to GLVND, as `install.EGL_VENDOR_DIRS`.
_EGL_VENDOR_DIRS = ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d")
#: Where a Vulkan driver announces itself to the Vulkan loader.
_VULKAN_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")


def gl_on_path() -> bool:
    """Put the GL libraries `python -m dycraft.install tools` installed where the system had none (`.gl_prefix`) on
    this process's loader path, which every child inherits, and name it in DYCRAFT_GL_PREFIX for Blender's sandbox,
    whose environment is closed (`runtime.worker_env`). Its Mesa joins the EGL implementations GLVND looks through,
    after the system's own, and its fontconfig file is used only where the system has none. True when the loader
    path changed."""
    try:
        lines = state_path(".gl_prefix").read_text(errors="replace").splitlines()
    except OSError:
        return False
    value = next((line.strip() for line in lines if line.strip() and not line.strip().startswith("#")), "")
    prefix = Path(value).expanduser() if value else None
    if prefix is None or not (prefix / "lib").is_dir():
        return False
    os.environ["DYCRAFT_GL_PREFIX"] = str(prefix)
    loader = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    changed = str(prefix / "lib") not in loader
    if changed:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*[p for p in loader if p], str(prefix / "lib")])
    if (prefix / "lib" / "dri").is_dir():
        os.environ.setdefault("LIBGL_DRIVERS_PATH", str(prefix / "lib" / "dri"))
    vendors = prefix / "share" / "glvnd" / "egl_vendor.d"
    if vendors.is_dir() and not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        dirs = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", "").split(os.pathsep)
        if str(vendors) not in dirs:
            os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = os.pathsep.join(
                [*(d for d in dirs if d), *(() if any(dirs) else _EGL_VENDOR_DIRS), str(vendors)])
    # Mesa's software Vulkan driver (lavapipe), for open3d's offscreen renderer: the only driver where the system
    # announces none, and one more beside the system's (a GPU's) otherwise.
    lavapipe = sorted((prefix / "share" / "vulkan" / "icd.d").glob("lvp_icd*.json"))
    if lavapipe and not os.environ.get("VK_ICD_FILENAMES"):
        if any(Path(d).is_dir() and any(Path(d).glob("*.json")) for d in _VULKAN_ICD_DIRS):
            os.environ.setdefault("VK_ADD_DRIVER_FILES", str(lavapipe[0]))
        else:
            os.environ["VK_ICD_FILENAMES"] = str(lavapipe[0])
    fonts = prefix / "etc" / "fonts" / "fonts.conf"
    if fonts.is_file() and not Path("/etc/fonts/fonts.conf").is_file():
        os.environ.setdefault("FONTCONFIG_FILE", str(fonts))
    return changed
