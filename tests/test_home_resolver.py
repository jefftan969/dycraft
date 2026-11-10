"""`dycraft._home`: where user state lives.

A source tree keeps its own state (unchanged for development); an installed
copy keeps it in the launch directory, not in its package directory and not in
a hidden per-user directory; `$DYCRAFT_HOME` overrides both.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dycraft import _home


def test_a_source_tree_is_its_own_home(monkeypatch):
    monkeypatch.delenv(_home.ENV_VAR, raising=False)
    assert not _home.installed()
    assert _home.home() == Path(_home.__file__).resolve().parent


def test_dycraft_home_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv(_home.ENV_VAR, str(tmp_path))
    assert _home.home() == tmp_path.resolve()
    assert _home.state_path("api_keys.sh") == tmp_path.resolve() / "api_keys.sh"
    assert _home.default_runs_dir() == tmp_path.resolve() / "runs"


def test_an_installed_copy_uses_the_launch_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(_home.ENV_VAR, raising=False)
    monkeypatch.setattr(_home, "_PACKAGE_DIR",
                        Path("/usr/lib/python3/site-packages/dycraft"))
    monkeypatch.chdir(tmp_path)
    assert _home.installed()
    assert _home.home() == tmp_path.resolve()
    assert _home.default_runs_dir() == tmp_path.resolve() / "runs"


def test_dist_packages_counts_as_installed(monkeypatch):
    monkeypatch.setattr(_home, "_PACKAGE_DIR",
                        Path("/usr/lib/python3/dist-packages/dycraft"))
    assert _home.installed()


def test_importing_it_loads_nothing_heavy(pkgroot):
    """Setup code reads the home before anything heavy is loaded."""
    probe = ("import json, sys; import dycraft._home; "
             "print(json.dumps(sorted(m for m in sys.modules "
             "if m.split('.')[0] in ('numpy', 'open3d', 'trimesh', 'PIL', "
             "'build123d', 'bpy', 'fastapi'))))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=pkgroot,
                         env={"PYTHONPATH": str(pkgroot), "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_the_cad_env_loads_it_by_path_too(tmp_path):
    """A driver may load `_cad_env` by file path, outside the package."""
    import importlib.util
    cad_env = Path(_home.__file__).resolve().parent / "agent" / "cad" / "_cad_env.py"
    spec = importlib.util.spec_from_file_location("_cad_env_by_path", cad_env)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._PYTHON_POINTER_FILE.name == ".b123d_python"


def test_an_installed_copy_can_bind_its_env_but_not_the_launch_dir(monkeypatch,
                                                                    tmp_path):
    """An installed package lives inside its interpreter's env, which the
    Blender sandbox must bind; the launch directory holds the user state."""
    from dycraft import _interpreters as I

    class _Installed:
        @staticmethod
        def home():
            return tmp_path / "launch"

        @staticmethod
        def installed():
            return True

    (tmp_path / "launch").mkdir()
    venv = tmp_path / "venv"
    (venv / "lib" / "python3.10" / "site-packages" / "dycraft").mkdir(parents=True)
    monkeypatch.setattr(I, "_load_home", lambda: _Installed)
    assert I._refused(venv.resolve()) is None
    assert "user-state root" in I._refused((tmp_path / "launch").resolve())
    assert I._refused(tmp_path.resolve()) is not None


def test_a_source_tree_still_guards_the_repository(monkeypatch, tmp_path):
    from dycraft import _interpreters as I
    monkeypatch.setenv(_home.ENV_VAR, str(tmp_path))
    repo = Path(I.__file__).resolve().parent
    assert "repository" in I._refused(repo)
