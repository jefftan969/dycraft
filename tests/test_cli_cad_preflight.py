"""The launcher's CAD preflight: what it records in `meta.json`, and the one
warning it prints when the interpreter cannot run build source."""
from __future__ import annotations

import pytest

import dycraft.agent.cad._cad_env as CE
from dycraft.agent.cad._b123d_version import REQUIRED_BUILD123D
from dycraft.cli import cli as C


def _interpreter(monkeypatch, *, path="/env/bin/python", version=REQUIRED_BUILD123D, ocp=True):
    monkeypatch.setattr(CE, "b123d_python", lambda: path)
    monkeypatch.setattr(CE, "probe_build123d_version", lambda python=None: version)
    monkeypatch.setattr(CE, "probe_ocp", lambda python=None: ocp)


def test_a_healthy_interpreter_is_recorded_without_a_warning(monkeypatch, capsys):
    _interpreter(monkeypatch)
    assert C._cad_preflight("animation") == {
        "path": "/env/bin/python", "build123d": REQUIRED_BUILD123D, "ocp": True}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("kwargs, record, warning", [
    (dict(path=None), {"path": None, "build123d": None, "ocp": False},
     "no build123d interpreter found — the scene layer runs build123d source"),
    (dict(version=None), {"path": "/env/bin/python", "build123d": None, "ocp": True},
     "has no importable build123d"),
    (dict(version="0.0.1"), {"path": "/env/bin/python", "build123d": "0.0.1", "ocp": True},
     f"but DyCraft is frozen at {REQUIRED_BUILD123D}"),
    (dict(ocp=False), {"path": "/env/bin/python", "build123d": REQUIRED_BUILD123D, "ocp": False},
     "no importable OCP"),
])
def test_an_unusable_interpreter_is_recorded_and_named_once(monkeypatch, capsys, kwargs, record, warning):
    _interpreter(monkeypatch, **kwargs)
    assert C._cad_preflight("scene") == record
    err = capsys.readouterr().err
    assert err.count("[cli] WARNING:") == 1 and warning in err
