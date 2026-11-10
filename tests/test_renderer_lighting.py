"""The studio light rig installs cleanly — no Open3D warning spam.

`_install_fixed_lights` runs on a freshly constructed `OffscreenRenderer`, so
a defensive `remove_light` before each `add_directional_light` can never do
anything except make Open3D log `Light <name> is not in the scene graph`.
That is not a harmless log line: Open3D writes it to Python's `sys.stdout`,
the harness daemon captures per-op stdout (`agent/server.py::_StdoutDemux`)
and ships it back inside the tool result, so six lines per renderer times
several renderers per compile ended up in the agent's context and in the run
transcript the viewer displays.

Importing the renderer module needs numpy but not open3d (open3d is imported
lazily inside `_open_renderer`), so this stays in the fast suite.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from dycraft.render.open3d_renderer import _LIGHT_RIG, _install_fixed_lights


class _StubLightScene:
    def __init__(self) -> None:
        self.added: list[tuple] = []
        self.removed: list[str] = []

    def remove_light(self, name):
        self.removed.append(name)

    def add_directional_light(self, name, color, direction, intensity,
                              cast_shadows):
        self.added.append((name, tuple(map(float, color)),
                           tuple(map(float, direction)), float(intensity),
                           cast_shadows))


def _stub_renderer() -> SimpleNamespace:
    return SimpleNamespace(scene=SimpleNamespace(scene=_StubLightScene()))


def test_light_rig_is_installed_without_removing_absent_lights():
    renderer = _stub_renderer()
    _install_fixed_lights(renderer)
    scene = renderer.scene.scene

    assert [row[0] for row in scene.added] == [n for n, _, _ in _LIGHT_RIG]
    assert scene.removed == [], (
        "removing a light that was never added makes Open3D log a warning "
        "into the agent's tool result; the renderer is always fresh here")


def test_each_light_points_at_the_origin_with_shadows_off():
    renderer = _stub_renderer()
    _install_fixed_lights(renderer)

    for (name, position, intensity), row in zip(_LIGHT_RIG,
                                                renderer.scene.scene.added):
        assert row[0] == name
        assert row[1] == (1.0, 1.0, 1.0), "rig is white; color comes from parts"
        assert row[3] == pytest.approx(intensity)
        assert row[4] is False, "shadows off so oblique faces stay readable"

        direction = row[2]
        assert math.isclose(
            math.sqrt(sum(c * c for c in direction)), 1.0, rel_tol=1e-6)
        norm = math.sqrt(sum(c * c for c in position)) or 1.0
        for got, pos in zip(direction, position):
            assert got == pytest.approx(-pos / norm, rel=1e-6)
