"""The backend registry: selection, dispatch, and cache-key separation.

These are the guards on the seam itself rather than on either renderer. The
failure they exist for is quiet: a request that silently drew with the wrong
backend, or a cache that handed one backend's PNG to the other, produces a
plausible picture and no error at all.

No GL and no bpy — every backend here is a stub.
"""

from __future__ import annotations


import pytest

from dycraft import render as R


@pytest.fixture
def request_factory(tmp_path):
    def make(**kw):
        base = dict(layer="object", ir={"id": "abc"}, catalog=None,
                    target="all", out_dir=tmp_path, step=0)
        base.update(kw)
        return R.RenderRequest(**base)
    return make


@pytest.fixture
def registry_sandbox():
    """Restore the real registry after a test scribbles on it."""
    backends = dict(R._BACKENDS)
    yield
    R._BACKENDS.clear()
    R._BACKENDS.update(backends)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_the_default_is_open3d(monkeypatch, request_factory):
    monkeypatch.delenv(R.RENDER_BACKEND_ENV, raising=False)
    assert R.default_backend() == "open3d"
    assert request_factory().backend == "open3d"


def test_the_env_var_reaches_a_request(monkeypatch, request_factory):
    monkeypatch.setenv(R.RENDER_BACKEND_ENV, "blender")
    assert request_factory().backend == "blender"


def test_the_env_var_is_read_per_request_not_at_import(monkeypatch,
                                                       request_factory):
    """The whole reason `backend` is a default_factory.

    The daemon and every sub-run inherit
    `DYCRAFT_RENDER_BACKEND` from the launcher, and `dycraft.render` may well
    have been imported before it was set. A module-level snapshot would freeze
    whichever value happened to be current then, and `--render-backend` would
    work or not depending on import order.
    """
    monkeypatch.setenv(R.RENDER_BACKEND_ENV, "open3d")
    assert request_factory().backend == "open3d"
    monkeypatch.setenv(R.RENDER_BACKEND_ENV, "blender")
    assert request_factory().backend == "blender"


def test_an_explicit_backend_beats_the_env(monkeypatch, request_factory):
    monkeypatch.setenv(R.RENDER_BACKEND_ENV, "blender")
    assert request_factory(backend="open3d").backend == "open3d"


def test_an_empty_env_var_falls_back_rather_than_selecting_nothing(monkeypatch):
    monkeypatch.setenv(R.RENDER_BACKEND_ENV, "")
    assert R.default_backend() == "open3d"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_an_unknown_backend_names_the_ones_that_exist(request_factory):
    with pytest.raises(ValueError, match=r"unknown render backend.*nope"):
        R.render_request(request_factory(backend="nope"))


def test_the_error_lists_backends_that_are_not_imported_yet(request_factory):
    """`blender` must appear even in an env with no bpy.

    Otherwise a typo'd `--render-backend blendr` on a box without the wheel
    reports "known: ['open3d']", which reads as "this build has no Blender
    support" — sending the reader off to look for a missing feature instead of
    at their own spelling.
    """
    with pytest.raises(ValueError) as e:
        R.render_request(request_factory(backend="nope"))
    assert "blender" in str(e.value)


def test_render_request_dispatches_by_name(registry_sandbox, request_factory):
    seen = []
    R.register_backend("stub", lambda req: seen.append(req) or R.RenderResult())
    R.render_request(request_factory(backend="stub"))
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

def _key(**kw):
    base = dict(ir_hash="h", target="all", camera_id="front")
    base.update(kw)
    return R.render_cache_key(**base)


def test_two_backends_do_not_share_a_cache_entry():
    """The failure this prevents is invisible.

    `ctx.render_cache` is per session, not per backend, and a hit is served by
    hardlinking the stored PNG. Without the backend in the key, a run that
    rendered under open3d and then switched would be served the *other*
    renderer's picture — a plausible image of the right scene, so nothing looks
    wrong and nothing raises.
    """
    assert _key(backend="open3d") != _key(backend="blender")


def test_the_backend_defaults_into_the_key():
    assert _key() == _key(backend="open3d")


def test_the_other_components_still_separate_entries():
    assert _key() != _key(ir_hash="other")
    assert _key() != _key(camera_id="top")
    assert _key() != _key(pose_label="lo")
    assert _key() != _key(config_version="zz")
    assert _key() != _key(catalog_fingerprint="fp")


def test_plan_outputs_can_override_the_config_version(tmp_path, monkeypatch):
    """So a Blender recipe bump cannot invalidate Open3D's stored renders."""
    from dycraft.render import _world

    req = R.RenderRequest(layer="object", ir={"id": "abc"}, catalog=None,
                          target="all", out_dir=tmp_path, step=0,
                          backend="blender")
    plan = _world._Plan([], None, [R.CameraSpec(id="front", eye=(0, 1, 0),
                                                target=(0, 0, 0), up=(0, 0, 1))],
                        ((0, 0, 0), (1, 1, 1)), "abc", None)
    default_key = _world.plan_outputs(req, plan)[0].key
    bumped = _world.plan_outputs(req, plan, config_version="b2")[0].key
    assert default_key != bumped
    assert "b2" in bumped


# ---------------------------------------------------------------------------
# Output size
# ---------------------------------------------------------------------------

def test_image_size_is_one_number_both_backends_read():
    """It lives in `dycraft.render`, not `open3d_renderer`, so Blender rendering
    does not import the Filament module to read it."""
    assert R.render_image_size() == (R.RENDER_WIDTH, R.RENDER_HEIGHT)
    from dycraft.render import open3d_renderer
    assert open3d_renderer.render_image_size() == R.render_image_size()
