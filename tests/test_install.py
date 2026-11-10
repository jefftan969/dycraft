"""`python -m dycraft.install`: the setup steps that need no network, conda or GPU, and the decisions made before them.

Downloading Blender, creating the conda envs and building cuRobo are exercised by a clean install, not here. These
tests pin what decides whether a setup is trustworthy: the archive's checksum, the platform guard, reuse of an
existing binary, the pointer files' format (`.blender_binary` is read whole, so it carries no comment), the build123d
check, the delegation to the environment check, the shell wrappers' bootstrap from a clone with nothing installed, the
PyTorch build chosen for the GPU, and the rule that the IK env is recorded only after cuRobo solves on that GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft import install as I

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def pointers(tmp_path, monkeypatch):
    """Pointer files land in `tmp_path`, whatever the state directory would be."""
    monkeypatch.setattr(I, "state_path", lambda name: tmp_path / name)
    return tmp_path


def test_download_verifies_the_published_checksum(tmp_path):
    src = tmp_path / "archive.tar.xz"
    src.write_bytes(b"blender bytes")
    good = hashlib.sha256(b"blender bytes").hexdigest()
    (tmp_path / "out").mkdir()
    got = I.download(src.as_uri(), tmp_path / "out" / "a.tar.xz", good, "t")
    assert got.read_bytes() == b"blender bytes"
    with pytest.raises(SystemExit, match="does not match its published sha256"):
        I.download(src.as_uri(), tmp_path / "out" / "b.tar.xz", "0" * 64, "t")
    assert not list((tmp_path / "out").glob("b.tar.xz*"))  # neither the file nor a .part is left behind


def test_download_reuses_a_verified_archive(tmp_path):
    dest = tmp_path / "a.tar.xz"
    dest.write_bytes(b"x")
    assert I.download("https://invalid.example/never-fetched", dest, hashlib.sha256(b"x").hexdigest(), "t") == dest


def test_the_official_build_is_linux_x86_64_only(tmp_path, monkeypatch):
    monkeypatch.setattr(I.sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="Linux x86-64"):
        I.fetch_blender(tmp_path)


def test_an_unpacked_binary_of_the_right_version_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(I.sys, "platform", "linux")
    monkeypatch.setattr(I.platform, "machine", lambda: "x86_64")
    binary = tmp_path / "blender-4.5.13-linux-x64" / "blender"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\necho 'Blender 4.5.13 LTS'\n")
    binary.chmod(0o755)
    monkeypatch.setattr(I, "download", lambda *a, **k: pytest.fail("must not download"))
    assert I.fetch_blender(tmp_path) == binary
    assert I.blender_version(binary) == "4.5.13"


def test_pointer_files(pointers):
    I.write_pointer(".blender_python", "/envs/b/bin/python", "blender", "render/_blender_env.py")
    I.write_pointer(".blender_binary", "/opt/blender/blender", "blender", "render/_blender_env.py", comment=False)
    assert (pointers / ".blender_python").read_text().splitlines() == [
        "# written by `python -m dycraft.install blender`; read by render/_blender_env.py", "/envs/b/bin/python"]
    # `_blender_env.blender_binary` reads the whole file, stripped, as the path.
    assert (pointers / ".blender_binary").read_text().strip() == "/opt/blender/blender"


def test_cad_records_an_interpreter_with_build123d_0_11_1(pointers):
    probe = subprocess.run([sys.executable, "-c", "import importlib.metadata as m; print(m.version('build123d'))"],
                           capture_output=True, text=True)
    if probe.stdout.strip() != "0.11.1":
        pytest.skip("this interpreter does not carry build123d 0.11.1")
    assert I.main(["cad"]) == 0
    assert (pointers / ".b123d_python").read_text().splitlines()[-1] == sys.executable


def test_cad_refuses_an_interpreter_without_build123d(pointers, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(SystemExit, match="cannot import build123d"):
        I.main(["cad", "--python", str(fake)])
    assert not (pointers / ".b123d_python").exists()


def test_check_runs_the_environment_check_with_its_own_options(monkeypatch):
    seen = []
    from dycraft import environment_test
    monkeypatch.setattr(environment_test, "main", lambda argv: seen.append(argv) or 0)
    assert I.main(["check", "--deep", "--non-interactive"]) == 0
    assert seen == [["--deep", "--non-interactive"]]


@pytest.mark.parametrize("step, option", [("blender", "--binary-dir"), ("ik", "--curobo-src")])
def test_the_setup_scripts_load_dycraft_from_their_own_directory(tmp_path, step, option):
    """Run from elsewhere, with nothing installed and no PYTHONPATH: the wrapper binds `dycraft` to its checkout."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHON"] = sys.executable
    out = subprocess.run(["bash", str(REPO / f"setup_{step}.sh"), "--help"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert option in out.stdout


# ------------------------------------------------------------------------------------------------ ik: the PyTorch build

@pytest.mark.parametrize("caps, index", [
    ([(6, 1)], "cu126"),                 # GTX 1070: only the CUDA 12.6 builds still carry Pascal kernels
    ([(8, 6), (8, 9)], "cu126"),
    ([(9, 0)], "cu126"),
    ([(12, 0)], "cu128"),                # RTX PRO 6000: the 12.6 builds stop at sm_90
    ([(8, 6), (12, 0)], "cu128"),        # Turing and up all have kernels in the 12.8 builds
])
def test_the_torch_build_follows_the_gpu(caps, index):
    assert I.torch_index(caps) == I.TORCH_INDEX + index


@pytest.mark.parametrize("caps, says", [
    ([], "no NVIDIA GPU"),
    ([(3, 7)], "older than any current PyTorch build"),
    ([(6, 1), (12, 0)], "no PyTorch build has kernels for both"),
])
def test_a_gpu_set_no_build_covers_is_refused_with_the_way_out(caps, says):
    with pytest.raises(SystemExit, match=says):
        I.torch_index(caps)


def test_compute_capabilities_are_read_from_nvidia_smi(monkeypatch):
    out = subprocess.CompletedProcess([], 0, stdout="6.1\n12.0\n", stderr="")
    monkeypatch.setattr(I, "_run", lambda argv, **kw: out)
    assert I.gpu_capabilities() == [(6, 1), (12, 0)]
    monkeypatch.setattr(I, "_run", lambda argv, **kw: subprocess.CompletedProcess([], 127, stdout="", stderr=""))
    assert I.gpu_capabilities() == []


def test_torch_is_held_to_the_chosen_build_while_its_dependencies_come_from_pypi(monkeypatch):
    """One `pip install torch --extra-index-url` would take PyPI's higher-versioned CUDA 13 build."""
    calls = []

    def fake_run(argv, **kw):
        calls.append([str(a) for a in argv])
        return subprocess.CompletedProcess(argv, 0, stdout="2.14.1+cu126\n", stderr="")
    monkeypatch.setattr(I, "_run", fake_run)
    index = I.TORCH_INDEX + "cu126"
    assert I.install_torch(Path("/env/bin/python"), index) == "2.14.1+cu126"
    first, _, last = calls
    assert first[-4:] == ["--no-deps", "--index-url", index, "torch"]
    assert last[-3:] == ["torch==2.14.1+cu126", "--extra-index-url", index]


@pytest.mark.parametrize("probe, solve, says", [
    ({"torch": "2.5.1+cu121", "cuda": True, "kernel": False, "arch": ["sm_90"], "gpu": "RTX PRO 6000",
      "cap": "sm_120", "error": "CUDA error: no kernel image is available for execution on the device."}, None,
     "no kernel image is available for execution on the device. Not recording"),
    ({"torch": "2.14.1+cu126", "cuda": True, "kernel": True, "gpu": "GTX 1070", "cap": "sm_61", "arch": ["sm_60"]},
     {"error": "RuntimeError: CUDA 13.4 requires sm_75 or higher"}, "CUDA 13.4 requires sm_75"),
    ({"torch": "2.14.1+cu126", "cuda": True, "kernel": True, "gpu": "GTX 1070", "cap": "sm_61", "arch": ["sm_60"]},
     {"solved": False, "error_mm": 40.0}, "IK did not converge"),
])
def test_ik_is_not_recorded_until_cuRobo_solves_on_the_gpu(tmp_path, pointers, monkeypatch, probe, solve, says):
    """`torch.cuda.is_available()` is True for a build with no kernels for the GPU; only running one tells."""
    from dycraft import environment_test
    src = tmp_path / "curobo"
    (src / ".git").mkdir(parents=True)
    monkeypatch.setattr(I, "gpu_capabilities", lambda: [(12, 0)])
    monkeypatch.setattr(I, "conda_env", lambda *a: tmp_path / "env" / "bin" / "python")
    monkeypatch.setattr(I, "install_torch", lambda py, index: "x")
    monkeypatch.setattr(I, "_run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(I, "probe_ik", lambda py: dict(probe))
    solved = []
    monkeypatch.setattr(environment_test, "solve_ik", lambda py: solved.append(py) or dict(solve))
    with pytest.raises(SystemExit, match=says):
        I.setup_ik(tmp_path / "env", src)
    assert not (pointers / ".ik_python").exists()
    assert bool(solved) == bool(probe["kernel"]), "the solve runs only once a kernel has"
    ok = {"torch": "2.11.0+cu128", "cuda": True, "kernel": True, "gpu": "RTX PRO 6000", "cap": "sm_120",
          "arch": ["sm_120"]}
    monkeypatch.setattr(I, "probe_ik", lambda py: dict(ok))
    monkeypatch.setattr(environment_test, "solve_ik", lambda py: {"solved": True, "error_mm": 0.01})
    assert I.setup_ik(tmp_path / "env", src) == 0
    assert (pointers / ".ik_python").exists()


@pytest.mark.parametrize("binary, bpy, status", [
    ("4.5.13 LTS", "4.5.15 LTS", "pass"),      # what `install blender` sets up: newest 4.5 bpy, pinned binary
    ("4.5.13 LTS", "4.5.13 LTS", "pass"),
    ("4.5.13 LTS", "5.0.1", "degraded"),       # a different release line is not like-for-like
])
def test_bpy_beside_the_binary_degrades_only_across_release_lines(monkeypatch, binary, bpy, status):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.delenv("DYCRAFT_BLENDER_PYTHON", raising=False)
    monkeypatch.setattr(E, "_blender_sources", lambda ctx: (None, None, "/env/bin/python", ".blender_python"))
    monkeypatch.setattr(E, "probe_bpy_python", lambda py: {
        "ok": True, "version": bpy, "python": "3.11.18", "numpy": "1.26.4"})
    assert E.check_bpy(SimpleNamespace(facts={"blender_version": binary})).status == status


def test_a_failure_is_not_labelled_from_the_launchers_echoed_argv():
    from dycraft import environment_test as E
    echoed = ("[claude_plain] argv=claude --settings '{\"sandbox\": {\"enabled\": true, "
              "\"failIfUnavailable\": true}}'\nsomething else went wrong")
    assert E.classify(echoed)[0] == ""
    assert E.classify(echoed + "\nbwrap: No permissions to create new namespace")[0] == "sandbox unavailable"


# ---------------------------------------------------------------------------
# conda or uv for the blender and ik environments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix, conda_prefix, uv, conda, route", [
    ("/home/u/.venvs/dycraft", "/opt/conda", "/home/u/.local/bin/uv", "/opt/conda/bin/conda", "uv"),  # base active
    ("/opt/conda/envs/dycraft", "/opt/conda/envs/dycraft", "/home/u/.local/bin/uv", None, "conda"),
    ("/home/u/.venvs/dycraft", "", None, "/opt/conda/bin/conda", "conda"),                             # no uv
], ids=["conda-base-active-uv-venv-running", "inside-a-conda-env", "no-uv"])
def test_the_env_route_follows_the_running_interpreter(prefix, conda_prefix, uv, conda, route):
    assert I.env_route(None, prefix=prefix, conda_prefix=conda_prefix, uv=uv or "", conda=conda or "")[0] == route


def test_a_conda_env_is_known_by_its_conda_meta(tmp_path):
    (tmp_path / "conda-meta").mkdir()
    assert I.env_route(None, prefix=str(tmp_path), conda_prefix="", uv="/bin/uv", conda="")[0] == "conda"


def test_a_flag_decides_and_nothing_found_says_what_to_install():
    assert I.env_route("uv", prefix="/opt/conda/envs/x", conda_prefix="/opt/conda/envs/x")[0] == "uv"
    with pytest.raises(SystemExit, match="neither uv nor conda was found. Install uv"):
        I.env_route(None, prefix="/home/u/.venv", conda_prefix="", uv="", conda="")


def test_uv_builds_each_env_from_the_same_spec_as_conda():
    assert I._spec("environment-blender.yml") == ("3.11", ["bpy==4.5.*"])
    assert I._spec("environment-ik.yml") == (">=3.10,<3.14", [])


def test_each_route_has_its_own_default_place(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.delenv("BLENDER_PREFIX", raising=False)
    assert I._env_prefix(SimpleNamespace(route="uv", prefix=None), "blender", "BLENDER_PREFIX") == \
        (tmp_path / "envs" / "blender", "uv")
    assert I._env_prefix(SimpleNamespace(route="conda", prefix=None), "blender", "BLENDER_PREFIX")[0] == \
        Path("~/conda_envs/dycraft-blender").expanduser()
    monkeypatch.setenv("BLENDER_PREFIX", str(tmp_path / "mine"))
    assert I._env_prefix(SimpleNamespace(route="uv", prefix=None), "blender", "BLENDER_PREFIX")[0] == tmp_path / "mine"


def test_the_check_ends_with_what_to_do_next(monkeypatch):
    """The reader runs the check first; its last lines say which command clears each problem, most blocking first."""
    from types import SimpleNamespace
    from dycraft import environment_test as E
    res = lambda status, fix: SimpleNamespace(status=status, fix=fix)
    checks = {cid: SimpleNamespace(title=t, section=sec) for cid, t, sec in [
        ("a", "bubblewrap", E.S_SANDBOX), ("b", "build123d interpreter", E.S_HOST), ("c", "motion capture", E.S_OPTIONAL),
        ("d", "codex", E.S_HARNESS), ("e", "socat", E.S_HARNESS)]}
    ctx = SimpleNamespace(checks=checks, results={
        "c": res(E.UNAVAILABLE, ["python -m dycraft.install mocap"]),               # optional: not a step
        "d": res(E.UNAVAILABLE, ["install the Codex CLI: curl -fsSL https://chatgpt.com/codex/install.sh | sh"]),
        "a": res(E.FAIL, ["python -m dycraft.install tools"]),
        "b": res(E.DEGRADED, ["python -m dycraft.install cad"]),
        "e": res(E.FAIL, ["python -m dycraft.install tools"])})
    assert E.next_steps(ctx) == [
        ("bubblewrap", "python -m dycraft.install tools"), ("build123d interpreter", "python -m dycraft.install cad"),
        ("codex", "install the Codex CLI: curl -fsSL https://chatgpt.com/codex/install.sh | sh")]


# ---------------------------------------------------------------------------
# install tools: bubblewrap, socat and ffmpeg without sudo or a system conda
# ---------------------------------------------------------------------------

@pytest.fixture
def gl_present(monkeypatch):
    """This machine's GL libraries, as `install tools` would find them: all there."""
    monkeypatch.setattr(I, "missing_gl", lambda: ([], False))


def test_tools_already_on_path_are_left_alone(monkeypatch, tmp_path, capsys, gl_present):
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.setattr(I.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(I, "download", lambda *a, **k: pytest.fail("downloaded micromamba"))
    assert I.setup_tools() == 0
    assert "all on PATH already; nothing to install" in capsys.readouterr().out


def test_missing_tools_come_from_conda_forge_into_dycrafts_bin(monkeypatch, tmp_path, gl_present):
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.setattr(I.shutil, "which", lambda name: None if name in ("bwrap", "socat") else f"/usr/bin/{name}")
    monkeypatch.setattr(I.platform, "machine", lambda: "x86_64")
    fetched, ran = {}, []

    def download(url, dest, sha256, step):
        fetched.update(url=url, sha256=sha256)
        dest.write_text("#!/bin/sh\n")
        return dest

    def run(argv, **kw):
        ran.append([str(a) for a in argv])
        prefix = Path(argv[argv.index("-p") + 1])
        (prefix / "bin").mkdir(parents=True, exist_ok=True)
        for tool in ("bwrap", "socat"):
            (prefix / "bin" / tool).write_text("")
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(I, "download", download)
    monkeypatch.setattr(I, "_run", run)
    assert I.setup_tools() == 0
    assert fetched == {"url": "https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/"
                              "micromamba-linux-64", "sha256": I.MICROMAMBA_SHA256["64"]}
    assert ran[0][-4:] == ["-c", "conda-forge", "bubblewrap", "socat"]
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == ["bwrap", "micromamba", "socat"]
    assert (tmp_path / "bin" / "bwrap").resolve() == (tmp_path / "envs" / "tools" / "bin" / "bwrap").resolve()
    from dycraft import environment_test as E
    assert E.read_pointer(tmp_path / ".tools_bin") == str(tmp_path / "bin")


def test_missing_gl_libraries_and_mesa_come_from_conda_forge(monkeypatch, tmp_path, capsys):
    """A cloud image with no desktop: no GL or X11 libraries, and no EGL implementation, so Mesa's software one too."""
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.setattr(I.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(I, "missing_gl", lambda: (["libEGL.so.1", "libXrender.so.1"], True))
    monkeypatch.setattr(I, "download", lambda url, dest, sha256, step: dest.write_text("") or dest)
    ran = []
    monkeypatch.setattr(I, "_run", lambda argv, **kw: ran.append([str(a) for a in argv])
                        or subprocess.CompletedProcess(argv, 0))
    assert I.setup_gl() == 0
    assert ran[0][-5:] == ["-c", "conda-forge", "libegl", "mesalib", "xorg-libxrender"]
    assert ran[0][ran[0].index("-p") + 1] == str(tmp_path / "envs" / "gl")
    from dycraft import environment_test as E
    assert E.read_pointer(tmp_path / ".gl_prefix") == str(tmp_path / "envs" / "gl")
    assert "missing libEGL.so.1, libXrender.so.1: installing libegl, mesalib, xorg-libxrender" in capsys.readouterr().out
    monkeypatch.setattr(I, "missing_gl", lambda: ([], False))
    assert I.setup_gl() == 0 and len(ran) == 1


def test_the_installed_gl_libraries_reach_the_loader_and_blenders_sandbox(monkeypatch, tmp_path):
    from dycraft import _home
    from dycraft.animation.blender import runtime
    prefix = tmp_path / "envs" / "gl"
    for d in ("lib/dri", "share/glvnd/egl_vendor.d"):
        (prefix / d).mkdir(parents=True)
    (prefix / "share/glvnd/egl_vendor.d/50_mesa.json").write_text("{}")
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    (tmp_path / ".gl_prefix").write_text(f"# written by `python -m dycraft.install tools`\n{prefix}\n")
    for name in ("LD_LIBRARY_PATH", "LIBGL_DRIVERS_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS",
                 "DYCRAFT_GL_PREFIX", "FONTCONFIG_FILE"):
        monkeypatch.setenv(name, "")               # so the test's own values are undone afterwards
        monkeypatch.delenv(name)
    monkeypatch.setattr(_home, "_EGL_VENDOR_DIRS", (str(tmp_path / "no-vendors"),))
    monkeypatch.setattr(runtime, "nvidia_usable", lambda: False)
    _home.gl_on_path()
    _home.gl_on_path()
    assert os.environ["LD_LIBRARY_PATH"] == str(prefix / "lib")
    assert os.environ["LIBGL_DRIVERS_PATH"] == str(prefix / "lib" / "dri")
    # Mesa joins the EGL implementations GLVND looks through, after the system's own (a GPU driver's stays first).
    assert os.environ["__EGL_VENDOR_LIBRARY_DIRS"] == f"{tmp_path / 'no-vendors'}:{prefix / 'share/glvnd/egl_vendor.d'}"
    env = runtime.worker_env()
    assert env["LD_LIBRARY_PATH"] == str(prefix / "lib") and env["LIBGL_DRIVERS_PATH"] == str(prefix / "lib/dri")
    assert env["__EGL_VENDOR_LIBRARY_DIRS"] == os.environ["__EGL_VENDOR_LIBRARY_DIRS"]


def test_the_tools_folder_goes_last_on_path_once(monkeypatch, tmp_path):
    """Last: it fills gaps, and a system tool installed later (Ubuntu 24.04's own bwrap) still comes first."""
    from dycraft import _home
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    (tmp_path / "bin").mkdir()
    (tmp_path / ".tools_bin").write_text(f"# written by `python -m dycraft.install tools`\n{tmp_path / 'bin'}\n")
    _home.tools_on_path()
    _home.tools_on_path()
    assert os.environ["PATH"] == f"/usr/bin:/bin:{tmp_path / 'bin'}"


def test_a_missing_tool_is_fixed_by_install_tools(monkeypatch):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.animation.blender import runtime
    monkeypatch.setattr(E, "which", lambda name: None)
    monkeypatch.setattr(E.sys, "executable", "/nonexistent/bin/python")
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 0)     # a kernel with nothing to confine Blender in its place
    assert E.check_bwrap(SimpleNamespace(facts={})).fix[0] == "`python -m dycraft.install tools`"


def test_a_bwrap_apparmor_refuses_is_named_with_what_works_on_this_system(monkeypatch, apparmor):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.animation.blender import runtime
    monkeypatch.setattr(runtime, "_userns_prefix", lambda unshare: None)
    monkeypatch.setattr(runtime, "landlock_abi", lambda: 0)     # where Landlock is there, Blender runs under it
    for name in runtime.ALLOW_UNSANDBOXED_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(E, "run", lambda *a, **k: SimpleNamespace(rc=1, tail=lambda n: ["bwrap: setting up uid map: "
                                                                                         "Permission denied"]))
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "1" if "apparmor_restrict" in str(self)
                        else real(self, *a, **k))
    from dycraft import _userns
    # Ubuntu with a bwrap profile and no system bwrap: the package is what it needs.
    apparmor(restricted=True, bwrap=False, profiled=True)
    result = E.check_userns(SimpleNamespace(facts={"bwrap": "/home/u/dycraft/bin/bwrap"}))
    assert result.status == E.FAIL
    assert result.summary == ("this bwrap (/home/u/dycraft/bin/bwrap) cannot create a user namespace: Ubuntu's "
                              "AppArmor restriction allows that only to a bwrap with a profile, and this kernel has no "
                              "Landlock")
    assert result.fix == ["install the system's bubblewrap: `sudo apt-get install bubblewrap socat`", _userns.LIFT,
                          "without sudo: `--allow-unsandboxed` runs Blender without a sandbox, so the agent's Blender "
                          "code can read and write anywhere you can"]    # A stock AWS image: /usr/bin/bwrap installed and no profile, so the package again would change nothing.
    apparmor(restricted=True, bwrap=True, profiled=False)
    system = E.check_userns(SimpleNamespace(facts={"bwrap": "/usr/bin/bwrap"}))
    assert system.fix[0].startswith("give /usr/bin/bwrap an AppArmor profile that allows user namespaces (this "
                                    "system has none, and the package alone does not add one): `printf ")
    assert system.fix[1] == _userns.LIFT and not any("apt-get" in line for line in system.fix)


def test_the_check_reports_the_asset_library_from_its_catalog_and_warns_without_its_index(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setattr(E, "_library_root", lambda: tmp_path / "assets" / "imported")
    missing = E.check_library(SimpleNamespace(facts={}))
    assert (missing.status, missing.summary, missing.fix) == (E.UNAVAILABLE, "not installed",
                                                              ["python -m dycraft.install library"])
    (tmp_path / "assets" / "imported").mkdir(parents=True)
    (tmp_path / "assets" / "imported" / "catalog.json").write_text(json.dumps(
        {"library_version": "2026.11.1", "assets": [{"name": f"c{i}"} for i in range(25)]}))
    unindexed = E.check_library(SimpleNamespace(facts={}))       # linked, but nothing tells agents about it
    assert (unindexed.status, unindexed.summary, unindexed.fix) == (
        E.WARN, "the library is linked but has no INDEX.md, so agents are not told about it",
        ["python -m dycraft.install library"])
    (tmp_path / "assets" / "imported" / "INDEX.md").write_text("# Asset library\n")
    found = E.check_library(SimpleNamespace(facts={}))
    assert (found.status, found.summary) == (E.PASS, "2026.11.1, 25 characters")


def test_the_library_check_does_not_read_a_broken_import_as_a_missing_index(monkeypatch, tmp_path):
    """A failed import of the library module surfaces; reported as "no INDEX.md" it would name the wrong cause."""
    import sys
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setattr(E, "_library_root", lambda: tmp_path)
    (tmp_path / "catalog.json").write_text(json.dumps({"library_version": "1", "assets": {}}))
    monkeypatch.setitem(sys.modules, "dycraft.agent.assets.asset_library", None)     # makes the import fail
    with pytest.raises(ImportError):
        E.check_library(SimpleNamespace(facts={}))


def test_the_library_check_handles_no_library_at_all(monkeypatch, tmp_path):
    """`asset_library.library_root()` answers None when nothing is installed."""
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    monkeypatch.setattr("dycraft.agent.assets.asset_library.library_root", lambda: None, raising=False)
    assert E.check_library(SimpleNamespace(facts={})).summary == "not installed"
def test_ik_without_a_gpu_is_skipped_not_failed(tmp_path, pointers, monkeypatch, capsys):
    """IK is optional: with no NVIDIA GPU, `install ik` makes nothing, says why and exits 0 so a scripted install
    carries on, without even choosing where an env would go. `--torch-index` still forces it."""
    monkeypatch.setattr(I, "gpu_capabilities", lambda: [])
    placed = []
    monkeypatch.setattr(I, "_env_prefix", lambda *a: placed.append(a) or (tmp_path / "env", "conda"))
    monkeypatch.setattr(I, "setup_ik", lambda *a: 0)
    assert I.main(["ik"]) == 0
    assert placed == [] and not (pointers / ".ik_python").exists()
    assert capsys.readouterr().out == f"[install ik] {I.IK_SKIPPED}\n"
    assert I.main(["ik", "--torch-index", I.TORCH_INDEX + "cu126"]) == 0
    assert placed, "--torch-index installs anyway"


@pytest.mark.parametrize("py, visible, summary, fix", [
    (None, (False, "nvidia-smi lists no GPU"), "unavailable here: no NVIDIA GPU", False),
    ("/env/ik/bin/python", (False, "CUDA_VISIBLE_DEVICES is empty"),
     "installed, but no NVIDIA GPU is visible (CUDA_VISIBLE_DEVICES is empty)", False),
    (None, (True, ""), "not configured", True),
])
def test_the_ik_check_says_which_of_the_three_it_is(monkeypatch, py, visible, summary, fix):
    from dycraft import environment_test as ET
    ik = ET._ik_module()
    monkeypatch.setattr(ik, "_resolve_ik_python", lambda: py)
    monkeypatch.setattr(ik, "gpu_visible", lambda: visible)
    got = ET.check_ik(None)
    assert got.status == ET.UNAVAILABLE and got.summary.startswith(summary)
    assert bool(got.fix) is fix
    assert got.lost == ET._IK_LOST


def test_an_entry_point_restarts_once_when_the_gl_libraries_join_the_loader_path(monkeypatch, tmp_path):
    """The loader reads LD_LIBRARY_PATH when a process starts: an entry point that loads GL itself restarts, once."""
    from dycraft import _home
    execs = []
    monkeypatch.setattr(_home.os, "execv", lambda exe, argv: execs.append(argv))
    monkeypatch.setenv(_home._LOADER_REEXEC_ENV, "")
    monkeypatch.delenv(_home._LOADER_REEXEC_ENV)
    monkeypatch.setattr(_home, "tools_on_path", lambda: True)
    _home.reexec_for_loader()
    assert len(execs) == 1 and os.environ[_home._LOADER_REEXEC_ENV] == "1"
    _home.reexec_for_loader()                       # the restarted process does not restart again
    monkeypatch.delenv(_home._LOADER_REEXEC_ENV)
    monkeypatch.setattr(_home, "tools_on_path", lambda: False)
    _home.reexec_for_loader()                       # nothing changed: no restart
    assert len(execs) == 1


def test_the_installer_puts_the_gl_libraries_on_the_path_of_its_import_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    prefix = tmp_path / "envs" / "gl"
    (prefix / "lib").mkdir(parents=True)
    (tmp_path / ".gl_prefix").write_text(f"{prefix}\n")
    for name in ("LD_LIBRARY_PATH", "DYCRAFT_GL_PREFIX", "PATH"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    seen = {}
    monkeypatch.setattr(I, "setup_cad", lambda python: seen.update(path=os.environ.get("LD_LIBRARY_PATH")) or 0)
    assert I.main(["cad"]) == 0
    assert str(prefix / "lib") in seen["path"].split(os.pathsep)


def test_mesas_vulkan_driver_joins_the_loader_for_open3d(monkeypatch, tmp_path):
    """open3d 0.20 renders offscreen through Vulkan: lavapipe is the only driver where the system has none, and one
    more beside a GPU's otherwise."""
    from dycraft import _home
    prefix = tmp_path / "envs" / "gl"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "share/vulkan/icd.d").mkdir(parents=True)
    (prefix / "share/vulkan/icd.d/lvp_icd.x86_64.json").write_text("{}")
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    (tmp_path / ".gl_prefix").write_text(f"{prefix}\n")
    for name in ("LD_LIBRARY_PATH", "DYCRAFT_GL_PREFIX", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES",
                 "__EGL_VENDOR_LIBRARY_DIRS", "LIBGL_DRIVERS_PATH", "FONTCONFIG_FILE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(_home, "_VULKAN_ICD_DIRS", (str(tmp_path / "none"),))
    _home.gl_on_path()
    assert os.environ["VK_ICD_FILENAMES"] == str(prefix / "share/vulkan/icd.d/lvp_icd.x86_64.json")
    monkeypatch.delenv("VK_ICD_FILENAMES")
    (tmp_path / "sys").mkdir()
    (tmp_path / "sys" / "nvidia_icd.json").write_text("{}")
    monkeypatch.setattr(_home, "_VULKAN_ICD_DIRS", (str(tmp_path / "sys"),))
    _home.gl_on_path()
    assert "VK_ICD_FILENAMES" not in os.environ
    assert os.environ["VK_ADD_DRIVER_FILES"] == str(prefix / "share/vulkan/icd.d/lvp_icd.x86_64.json")


def test_the_check_names_the_state_directory_and_warns_from_one_with_no_state(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import _home
    from dycraft import environment_test as E
    monkeypatch.setenv(_home.ENV_VAR, str(tmp_path))
    assert E.check_state_dir(SimpleNamespace(facts={})).summary == f"{tmp_path.resolve()} (DYCRAFT_HOME)"
    monkeypatch.delenv(_home.ENV_VAR)
    monkeypatch.setattr(_home, "installed", lambda: True)
    monkeypatch.chdir(tmp_path)
    result = E.check_state_dir(SimpleNamespace(facts={}))
    assert result.status == E.WARN and result.summary.startswith(f"{tmp_path.resolve()}: the current directory, "
                                                                 "which holds no DyCraft state")
    assert result.fix[-1] == "or set DYCRAFT_HOME to that directory, for example in your shell's profile"
    (tmp_path / ".tools_bin").write_text("/x\n")
    assert E.check_state_dir(SimpleNamespace(facts={})).status == E.PASS


def test_a_missing_system_library_is_named_with_its_packages_not_a_reinstall():
    from dycraft import environment_test as E
    soname, fixes = E.missing_library("ImportError: libXrender.so.1: cannot open shared object file: No such file")
    assert soname == "libXrender.so.1" and fixes[0] == "without sudo: `python -m dycraft.install tools`"
    assert E.GL_APT in fixes[1] and E.GL_DNF in fixes[2]
    soname, fixes = E.missing_library("blender: error while loading shared libraries: libxkbcommon.so.0: cannot "
                                      "open shared object file: No such file or directory")
    assert fixes == ["the system's, with root: `sudo apt-get install libxkbcommon0` (Fedora and RHEL: "
                     "`sudo dnf install libxkbcommon`)"]
    assert E.missing_library("ModuleNotFoundError: No module named 'bpy'") is None
    assert I._missing_library("OSError: libGL.so.1: cannot open shared object file").startswith(
        "this machine lacks libGL.so.1, a system library it loads. Without sudo, `python -m dycraft.install tools`")


def test_several_missing_packages_are_one_step_not_one_command_each(monkeypatch):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setattr(E, "INSTALLED", False)
    ctx = SimpleNamespace(facts={"python": {"pip": True}}, results={}, checks={})
    for dist in ("numpy", "scipy"):
        check = next(c for c in E.CHECKS if c.id == f"pkg.{dist}")
        ctx.checks[check.id] = check
        ctx.results[check.id] = E.fail(f"{dist} is not installed", fix=[E.install_cmd(ctx, f"{dist}>=1")])
    steps = E.next_steps(ctx)
    assert steps == [("Python packages", f"{E.sys.executable} -m pip install -e {E.REPO}")]


def test_a_broken_library_link_is_named_by_the_check_and_the_installer(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.agent.assets import _fetch
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "imported").symlink_to(tmp_path / "gone")
    monkeypatch.setattr(E, "_library_root", lambda: tmp_path / "assets" / "imported")
    result = E.check_library(SimpleNamespace(facts={}))
    assert result.summary == f"{tmp_path / 'assets' / 'imported'} is a broken link to {tmp_path / 'gone'}"
    dest = tmp_path / "hub"
    (dest / "bundles").mkdir(parents=True)
    (dest / "catalog.json").write_text('{"library_version": "v1", "assets": {}}')
    monkeypatch.setattr(_fetch, "_download", lambda repo, patterns, revision: dest)
    out = io.StringIO()
    assert _fetch.run_library(out=out) == 1
    assert out.getvalue() == (f"[library] {tmp_path / 'assets' / 'imported'} is a broken link to {tmp_path / 'gone'}: "
                              "restore what it points at, or remove the link "
                              f"(`rm {tmp_path / 'assets' / 'imported'}`), then run this again\n")


def test_a_copy_that_shadows_the_installed_checkout_is_not_offered_for_install(monkeypatch, tmp_path):
    """The copy running the check came from PYTHONPATH, ahead of the editable install: drop the entry."""
    import importlib.metadata as md
    from types import SimpleNamespace
    from dycraft import environment_test as E
    shadow = tmp_path / "shadowpkg"
    (shadow / "dycraft").mkdir(parents=True)
    monkeypatch.setattr(E, "REPO", (shadow / "dycraft").resolve())
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    real = tmp_path / "checkout"

    class Dist:
        def read_text(self, name):
            return json.dumps({"url": f"file://{real}", "dir_info": {"editable": True}})
    monkeypatch.setattr(md, "distribution", lambda name: Dist())
    result = E.check_shadow(SimpleNamespace(facts={}, pkgroot=tmp_path / "pkgroot", child_env=lambda: {}))
    assert result.fix == [f"remove {shadow} from PYTHONPATH (it puts another dycraft ahead of this checkout), or "
                          "unset PYTHONPATH"]
    assert not any("pip install -e" in line for line in result.fix)


def test_a_cad_pointer_naming_no_interpreter_is_said(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.agent.cad import _cad_env as ce
    monkeypatch.delenv("DYCRAFT_B123D_PYTHON", raising=False)
    monkeypatch.setattr(ce, "_resolve_b123d_python", lambda *a, **k: E.sys.executable)
    monkeypatch.setattr(ce, "_pointer_file_python", lambda: "/nonexistent/bin/python")
    monkeypatch.setattr(E, "probe_cad_python", lambda interp: {"ok": True, "version": "0.11.1", "ocp": "7.8",
                                                                "python": "3.11"})
    result = E.check_cad(SimpleNamespace(facts={}, interactive=False))
    assert result.status == E.WARN and ".b123d_python names /nonexistent/bin/python, which is not an interpreter" \
        in result.summary
    assert result.fix == ["`python -m dycraft.install cad` records the interpreter again"]


def test_the_userns_advice_fits_the_system(monkeypatch, tmp_path):
    """The package where /usr/bin/bwrap is missing; a profile where none grants it user namespaces (a stock AWS image
    ships none, so installing the package there changes nothing); the restriction itself always; nothing where
    AppArmor does not restrict them."""
    from dycraft import _userns
    sysctl, apparmor_dir = tmp_path / "restrict", tmp_path / "apparmor.d"
    apparmor_dir.mkdir()
    monkeypatch.setattr(_userns, "RESTRICT_SYSCTL", sysctl)
    monkeypatch.setattr(_userns, "APPARMOR_DIR", apparmor_dir)
    monkeypatch.setattr(_userns, "SYSTEM_BWRAP", tmp_path / "bwrap")
    sysctl.write_text("0\n")
    assert _userns.fixes() == []
    sysctl.write_text("1\n")
    steps = _userns.fixes()
    assert steps[0] == "install the system's bubblewrap: `sudo apt-get install bubblewrap socat`"
    assert steps[1].startswith("give /usr/bin/bwrap an AppArmor profile") and steps[2] == _userns.LIFT
    (tmp_path / "bwrap").write_text("")
    assert [s[:10] for s in _userns.fixes()] == ["give /usr/", "or lift th"]
    (apparmor_dir / "bwrap-userns-restrict").write_text("profile bwrap /usr/bin/bwrap flags=(unconfined) {\n  userns,\n}\n")
    assert _userns.fixes() == [_userns.LIFT]
