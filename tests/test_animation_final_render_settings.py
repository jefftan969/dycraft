"""Final-film configuration, device selection, and publication boundaries."""
from types import SimpleNamespace

import pytest

from dycraft.animation.blender.render_config import (
    apply_final_render_settings, final_render_settings,
)


@pytest.mark.parametrize("settings", [
    [], {"width": 0}, {"height": -2}, {"width": True}, {"height": 1080.0},
    {"width": 1919}, {"samples": 0}, {"samples": "64"},
    {"engine": "WORKBENCH", "samples": 1}, {"resolution": [1920, 1080]},
    {"strategy": "unknown"}, {"threads": 0}, {"threads": True},
    {"engine": "EEVEE", "compute_backend": "CPU"},
    {"engine": "CYCLES", "compute_backend": "GPU"},
    {"engine": "EEVEE", "denoise": True}, {"engine": "CYCLES", "denoise": 1},
    {"engine": "CYCLES", "adaptive_threshold": float("nan")},
    {"engine": "CYCLES", "adaptive_threshold": 1.1},
])
def test_invalid_settings_are_refused_before_any_render(settings):
    """Validation happens on the settings alone, so a bad request never
    reaches Blender."""
    with pytest.raises(ValueError):
        final_render_settings(settings)


def _bpy(devices=()):
    scene = SimpleNamespace(
        render=SimpleNamespace(engine="BLENDER_WORKBENCH", resolution_x=640,
                               resolution_y=360, resolution_percentage=50,
                               image_settings=SimpleNamespace(file_format="JPEG"),
                               film_transparent=True),
        display=SimpleNamespace(render_aa="16", shading=SimpleNamespace(
            show_shadows=True, show_cavity=True, color_type="MATERIAL", light="STUDIO")),
        eevee=SimpleNamespace(taa_render_samples=8),
        cycles=SimpleNamespace(samples=8, device="GPU", use_denoising=True,
                               denoiser="OPTIX", use_adaptive_sampling=False,
                               adaptive_threshold=.01),
    )
    prefs = SimpleNamespace(devices=list(devices), get_devices=lambda: None)
    return SimpleNamespace(context=SimpleNamespace(
        scene=scene, preferences=SimpleNamespace(addons={"cycles": SimpleNamespace(preferences=prefs)})))


def test_final_defaults_ignore_diagnostic_environment_and_report_effective_values(monkeypatch):
    monkeypatch.setenv("DYCRAFT_BLENDER_WIDTH", "320")
    monkeypatch.setenv("DYCRAFT_BLENDER_ENGINE", "BLENDER_WORKBENCH")
    bpy = _bpy()
    actual = apply_final_render_settings(bpy, None)
    assert actual["engine"] == "CYCLES"
    assert (actual["width"], actual["height"], actual["samples"]) == (1920, 1080, 8)
    assert actual["compute_backend"] == "CPU"      # no device in this fake
    assert bpy.context.scene.render.resolution_percentage == 100
    assert bpy.context.scene.render.film_transparent  # no lighting/appearance preset


def test_the_cycles_default_carries_denoising_because_8_samples_needs_it():
    """8 un-denoised samples measured SSIM 0.531 against a 256-sample reference;
    denoised it measured 0.972. The low sample count is only defensible with the
    denoiser on, so both travel together in the default rather than separately."""
    settings = final_render_settings(None)
    assert settings["engine"] == "CYCLES"
    assert settings["samples"] == 8
    assert settings["denoise"] is True
    assert settings["adaptive_threshold"] == .04


def test_explicit_eevee_keeps_its_own_sample_count_and_rejects_cycles_keys():
    """8 samples is a Cycles-with-denoising number, not a comparable Eevee
    request: Eevee's error is rasterization bias, so samples barely move it."""
    settings = final_render_settings({"engine": "EEVEE"})
    assert settings["samples"] == 64
    assert "denoise" not in settings and "adaptive_threshold" not in settings


def test_cycles_auto_reports_cpu_fallback_and_uses_compatible_denoiser():
    bpy = _bpy()
    actual = apply_final_render_settings(bpy, {"engine": "CYCLES", "samples": 17})
    assert actual["compute_backend"] == "CPU"
    assert actual["samples"] == 17
    assert actual["denoiser"] == "OPENIMAGEDENOISE"
    assert actual["denoise"] is True
    assert actual["adaptive_sampling"] is True   # defaulted in with the Cycles engine
    with pytest.raises(ValueError, match="Requested compute_backend OPTIX unavailable"):
        apply_final_render_settings(bpy, {"engine": "CYCLES", "compute_backend": "OPTIX"})


def test_adaptive_sampling_can_be_turned_off_for_a_fixed_sample_render():
    """A reference or ground-truth render needs every sample actually taken.
    `0` cannot express it — Cycles reads 0 as "pick a threshold from the sample
    count" — so None is the opt-out, and it stays Cycles-only."""
    settings = final_render_settings(
        {"engine": "CYCLES", "samples": 256, "denoise": False, "adaptive_threshold": None})
    assert settings["adaptive_threshold"] is None   # carried through, not dropped
    bpy = _bpy()
    bpy.context.scene.cycles.use_adaptive_sampling = True   # saved scene had it ON
    actual = apply_final_render_settings(bpy, settings)
    assert actual["adaptive_sampling"] is False
    assert actual["samples"] == 256
    with pytest.raises(ValueError):
        final_render_settings({"engine": "EEVEE", "adaptive_threshold": None})


def test_workbench_pins_its_cost_and_keeps_the_rest_of_the_saved_shading():
    """Explicit Workbench settings get the same FXAA/no-shadow profile as the
    default draft. A caller passing `render_settings` only for a resolution
    must not inherit the saved 8x AA and shadows."""
    bpy = _bpy()
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.render.use_simplify = False
    settings = final_render_settings({"engine": "WORKBENCH", "threads": 4})
    assert settings["strategy"] == "batch"
    actual = apply_final_render_settings(bpy, settings)
    assert actual["antialiasing"] == "FXAA"
    assert actual["workbench_shading"]["show_shadows"] is False
    assert actual["workbench_shading"]["show_cavity"] is True     # measured free
    assert actual["workbench_shading"]["color_type"] == "MATERIAL"
    assert scene.render.use_compositing and not scene.render.use_simplify
    assert scene.render.threads == 4 and scene.render.threads_mode == "FIXED"


def test_unset_threads_are_the_cpus_this_process_may_use(monkeypatch):
    """Blender's AUTO counts the machine and ignores affinity: a worker pinned
    to 4 cores drew with 16 threads. An explicit count is kept."""
    import os
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})
    bpy = _bpy()
    actual = apply_final_render_settings(bpy, final_render_settings({"engine": "WORKBENCH"}))
    assert bpy.context.scene.render.threads_mode == "FIXED" and actual["threads"] == 3
    bpy = _bpy()
    apply_final_render_settings(bpy, final_render_settings({"engine": "WORKBENCH", "threads": 8}))
    assert bpy.context.scene.render.threads == 8


def test_cycles_gpu_selection_is_explicit_and_does_not_mix_devices():
    devices = [SimpleNamespace(type=kind, name=kind, use=True) for kind in ("CPU", "CUDA", "OPTIX")]
    bpy = _bpy(devices)
    actual = apply_final_render_settings(bpy, {"engine": "CYCLES", "compute_backend": "CUDA",
                                             "denoise": False, "adaptive_threshold": .03})
    assert actual["compute_backend"] == "CUDA"
    assert actual["devices"] == ["CUDA"]
    assert [d.use for d in devices] == [False, True, False]
    assert actual["denoise"] is False
    assert actual["adaptive_sampling"] is True
    assert actual["adaptive_threshold"] == .03
    actual = apply_final_render_settings(bpy, {"engine": "CYCLES", "denoise": True})
    assert actual["compute_backend"] == "OPTIX"
    assert actual["denoiser"] == "OPTIX"


