"""Final-film settings shared by the daemon and standalone Blender worker.

No bpy import here: validate requests before starting an expensive render, then
apply them to the disposable render scene without changing diagnostic defaults.
"""
import math
import os


#: Where an author's `animation_src/` tree is readable inside the workspace.
#:
#: Here because this is the one module both sides import: the animation session
#: copies the tree in (`AnimationSession.stage_sources`) and the worker puts
#: it on `sys.path`, and the two must name the same directory. The contents
#: are also copied onto the workspace root, which is what `import helper`
#: resolves against. This is the path an author *writes*, as in
#: `/workspace/animation_src/motion.json`.
SOURCE_SUBDIR = "animation_src"


#: What the appearance-judging diagnostics draw with. Storyboard panels and
#: independent review both exist to be looked at for materials, lighting, and
#: composition, so they cannot use the module's Workbench default — that one
#: is for geometry, motion, and contact, which shading does not affect.
#:
#: Cycles at 8 samples because denoising is what makes that count usable:
#: un-denoised it scores SSIM 0.531, denoised 0.972 against a 256-sample
#: reference. `final_render_settings` supplies `denoise` and
#: `adaptive_threshold` for CYCLES, so they are not repeated here. Small
#: frames because these are evidence to inspect, not delivery: measured
#: ~0.9 s a frame at this size against ~2.0 s at 960x540.
REVIEW_RENDER_SETTINGS = {"engine": "CYCLES", "width": 640, "height": 360, "samples": 8}
#: What `render`, `snapshot`, and `motion_image` draw when nothing is asked.
DIAGNOSTIC_RENDER_SETTINGS = {"engine": "WORKBENCH", "width": 1280, "height": 720}


def fit_aspect(width, height, aspect):
    """`(width, height)` with the same pixel count, at `aspect`, in even sizes."""
    import math
    width = 2 * max(1, round(math.sqrt(width * height * aspect) / 2))
    return width, 2 * max(1, round(width / aspect / 2))


def unsized(requested):
    """Whether a request leaves the frame size to the operation's defaults,
    which then follow the project's own aspect ratio (`fit_aspect`)."""
    return not requested or ("width" not in requested and "height" not in requested)


def check_size(requested):
    """A size is both dimensions or neither. One alone kept the other's
    default and drew at an aspect nobody asked for."""
    if isinstance(requested, dict) and ("width" in requested) != ("height" in requested):
        raise ValueError("render_settings names width and height together, or neither")


def over_defaults(requested, defaults):
    """`requested` laid over an operation's own defaults.

    The fields a request leaves out come from that operation's defaults, not
    from the delivery recipe: `render(render_settings={'width': 640, ...})`
    stays a Workbench draft. A request that switches engine keeps only the
    default size, since the rest belongs to the engine it left.
    """
    check_size(requested)
    requested = dict(requested or {})
    base = dict(defaults)
    if "engine" in requested and requested["engine"] != base.get("engine"):
        base = {k: base[k] for k in ("width", "height") if k in base}
    return {**base, **requested}


#: How every Workbench frame is drawn, whoever asked for it: a default draft, a
#: `render_settings` override, a snapshot, or a Workbench delivery. Blender's
#: factory `render_aa` is '8' -- 8x supersampling -- and cast shadows cost as
#: much again. Measured at 1280x720 on two staged projects (576 and 455
#: objects), three frames each after a warm-up:
#:
#:   render_aa  shadows   parts     clock
#:   '8'        on        4.491     2.834   <- the factory staging
#:   '8'        off       0.655     0.527
#:   'FXAA'     on        0.746     0.523
#:   'FXAA'     off       0.232     0.226   <- what this stages
#:   'OFF'      off       0.231     0.211
#:
#: 19.4x and 12.6x end to end, and the two knobs compound: AA alone is 6.0x,
#: shadows alone 6.9x. FXAA rather than OFF because they measured within
#: 0.3 ms of each other and FXAA reads better on thin parts.
#:
#: `show_cavity` is deliberately NOT touched: within 2 % of free in both
#: regimes, because it is a screen-space post-pass, and a real geometry cue.
#:
#: It covers explicit `render_settings` too: callers pass them mostly to change
#: the resolution, and the saved scene's AA and shading would otherwise apply
#: verbatim.
#:
#: Overridable so a re-render can reproduce a frame drawn with full AA and
#: shadows: `DYCRAFT_BLENDER_DIAGNOSTIC_AA=8
#: DYCRAFT_BLENDER_DIAGNOSTIC_SHADOWS=1`. Read once, at import.
WORKBENCH_AA = os.environ.get("DYCRAFT_BLENDER_DIAGNOSTIC_AA") or "FXAA"
WORKBENCH_SHADOWS = os.environ.get("DYCRAFT_BLENDER_DIAGNOSTIC_SHADOWS") == "1"


def apply_workbench_profile(scene):
    """Pin Workbench's cost (see `WORKBENCH_AA`); leave every other display
    setting -- color type, lighting, cavity, background -- as saved.

    The material colors Workbench draws are the other half and are not set
    here: `worker.render_frames` syncs them, because the one Base Color reader
    (`worker.surface_color`) lives in a module this one cannot import.
    """
    scene.display.render_aa = WORKBENCH_AA
    scene.display.shading.show_shadows = WORKBENCH_SHADOWS


def final_render_settings(settings=None):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("render_settings must be an object")
    allowed = {"engine", "width", "height", "samples", "compute_backend",
               "denoise", "adaptive_threshold", "strategy", "threads"}
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown final render settings: {', '.join(sorted(map(str, unknown)))}")
    result = {"engine": "CYCLES", "width": 1920, "height": 1080,
              "compute_backend": "AUTO", **settings}
    if result["engine"] == "EEVEE":
        result["engine"] = "BLENDER_EEVEE_NEXT"
    if result["engine"] == "WORKBENCH":
        result["engine"] = "BLENDER_WORKBENCH"
    if result["engine"] not in ("BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
        raise ValueError("Final render engine must be EEVEE, CYCLES, or WORKBENCH")
    workbench = result["engine"] == "BLENDER_WORKBENCH"
    if workbench:
        if "samples" in settings:
            raise ValueError("Workbench draws at a fixed FXAA anti-aliasing; samples is for Eevee/Cycles")
        result.pop("samples", None)
    elif result["engine"] == "CYCLES":
        # Denoised Cycles converges long before Eevee's sample count matters, and
        # it beats Eevee on BOTH cost and accuracy rather than trading one for the
        # other: on a thousand-object project at 720p on CPU, 8 denoised
        # samples take seconds a frame and match a high-sample reference more
        # closely than Eevee at 64 samples in over a minute. Eevee's error is
        # rasterization bias, not sampling noise, so its samples buy almost
        # nothing. Denoising is what earns the low sample count: 8 samples
        # without it are far from the reference.
        result.setdefault("samples", 8)
        result.setdefault("denoise", True)
        result.setdefault("adaptive_threshold", 0.04)
    else:
        # Eevee only when explicitly asked for, at 64 samples: 8 Eevee samples
        # is not a comparable request.
        result.setdefault("samples", 64)
    result.setdefault("strategy", "batch" if workbench else "frames")
    if result["strategy"] not in ("batch", "frames"):
        raise ValueError("strategy must be batch or frames")
    for key in ("width", "height", *(('samples',) if not workbench else ())):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if "threads" in result and (type(result["threads"]) is not int or not 1 <= result["threads"] <= 1024):
        raise ValueError("threads must be an integer from 1 to 1024")
    if result["width"] % 2 or result["height"] % 2:
        raise ValueError("width and height must be even for H.264 yuv420p output")
    if result["compute_backend"] not in ("AUTO", "CPU", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        raise ValueError("Unsupported compute_backend; use AUTO, CPU, OPTIX, CUDA, HIP, ONEAPI, or METAL")
    if "denoise" in result and type(result["denoise"]) is not bool:
        raise ValueError("denoise must be a boolean")
    if result.get("adaptive_threshold", "") is None:
        # Explicit opt-out, carried through as None so `apply` turns the saved
        # scene's setting OFF rather than merely leaving it alone. `0` cannot mean
        # this: Cycles reads 0 as "choose a threshold from the sample count".
        pass
    elif "adaptive_threshold" in result:
        value = result["adaptive_threshold"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("adaptive_threshold must be a finite number from 0 to 1")
    # `adaptive_threshold` is read from `settings`, not `result`, so the Cycles
    # default injected above cannot trip this guard on a Cycles request while a
    # Cycles-only key passed to another engine is still rejected.
    if result["engine"] != "CYCLES" and (
            result["compute_backend"] != "AUTO" or "denoise" in result
            or "adaptive_threshold" in settings):
        raise ValueError("compute_backend selection, denoise, and adaptive_threshold require CYCLES")
    return result


#: The Cycles GPU backends tried, in order, for `AUTO` and for the CPU probe.
_GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")


def _available_gpu_backend(bpy):
    """`(backend, [device names])` for the first usable Cycles GPU, or `None`.

    A probe, not a selection: enumerating a backend's devices requires setting
    `compute_device_type` first, so the previous value is restored afterwards.
    It must not leave the preferences pointing at a backend the render will not
    go on to use.
    """
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return None
    prefs = addon.preferences
    previous = getattr(prefs, "compute_device_type", None)
    try:
        for backend in _GPU_BACKENDS:
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devices = [d.name for d in prefs.devices if d.type == backend]
                if devices:
                    return backend, devices
            except (TypeError, ValueError, RuntimeError):
                continue
    finally:
        if previous is not None:
            try:
                prefs.compute_device_type = previous
            except (TypeError, ValueError, RuntimeError):
                pass
    return None


def _cycles_device(bpy, scene, requested):
    if requested == "CPU":
        # Refuse rather than quietly path-trace on the CPU while an idle GPU is
        # attached. Same rule this function applies to an explicitly requested
        # *unavailable* backend below: an explicit compute_backend is honored
        # or reported, never silently turned into something slower. Storyboard
        # renders on the CPU can be a run's largest single cost.
        # DYCRAFT_ALLOW_CPU_CYCLES is an operator-level escape, deliberately not
        # on the agent surface. "A GPU exists" is not "a GPU is free" when
        # several runs share one machine: a launcher can decide to keep a run
        # off the card, and an agent cannot.
        if os.environ.get("DYCRAFT_ALLOW_CPU_CYCLES") == "1":
            scene.cycles.device = "CPU"
            return "CPU", []
        available = _available_gpu_backend(bpy)
        if available is not None:
            backend, devices = available
            raise ValueError(
                f'compute_backend="CPU" was requested but {backend} is available '
                f'({", ".join(devices)}). Use "AUTO" to select a GPU backend, or '
                "omit compute_backend. CPU is accepted only on a host with no "
                "Cycles GPU backend.")
        scene.cycles.device = "CPU"
        return "CPU", []
    addon = bpy.context.preferences.addons.get("cycles")
    choices = _GPU_BACKENDS if requested == "AUTO" else (requested,)
    errors = []
    if addon is not None:
        prefs = addon.preferences
        for backend in choices:
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devices = [d for d in prefs.devices if d.type == backend]
                if not devices:
                    errors.append(f"{backend}: no devices")
                    continue
                for device in prefs.devices:
                    device.use = device.type == backend
                scene.cycles.device = "GPU"
                return backend, [d.name for d in devices]
            except (TypeError, ValueError, RuntimeError) as error:
                errors.append(f"{backend}: {error}")
    if requested != "AUTO":
        reason = "; ".join(errors) or "Cycles device preferences unavailable"
        raise ValueError(f"Requested compute_backend {requested} unavailable: {reason}")
    scene.cycles.device = "CPU"
    return "CPU", []


def apply_final_render_settings(bpy, settings):
    settings = final_render_settings(settings)
    scene = bpy.context.scene
    scene.render.engine = settings["engine"]
    scene.render.resolution_x = settings["width"]
    scene.render.resolution_y = settings["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    # Unset means the CPUs this process may run on, never Blender's AUTO:
    # AUTO counts every CPU in the machine and ignores affinity, so a worker
    # pinned to 4 cores would render with every core's threads, and an unset
    # Cycles render would claim the whole machine. An explicit value is always
    # kept.
    scene.render.threads_mode = "FIXED"
    scene.render.threads = settings.get("threads") or min(1024, len(os.sched_getaffinity(0)))
    backend, devices = None, []  # Eevee does not expose Cycles device selection.
    if settings["engine"] == "CYCLES":
        scene.cycles.samples = settings["samples"]
        backend, devices = _cycles_device(bpy, scene, settings["compute_backend"])
        if "denoise" in settings:
            scene.cycles.use_denoising = settings["denoise"]
        if settings.get("adaptive_threshold") is not None:
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = settings["adaptive_threshold"]
        elif "adaptive_threshold" in settings:
            scene.cycles.use_adaptive_sampling = False
        # Do not leave a saved OptiX denoiser active on a CPU/CUDA device.
        if scene.cycles.use_denoising:
            scene.cycles.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
    elif settings["engine"] == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = settings["samples"]
    else:
        # Workbench keeps the saved display shading except the two settings
        # that carry nearly all its cost. Compositing and geometry are not
        # traded away for timings on any engine.
        apply_workbench_profile(scene)
    actual = effective_render_settings(scene, backend, devices)
    for key in ("engine", "width", "height", "samples", "threads"):
        if key in settings and actual[key] != settings[key]:
            raise ValueError(f"Blender could not apply {key}={settings[key]!r}; got {actual[key]!r}")
    return actual


def effective_render_settings(scene, backend, devices):
    """Read back the render scene; never infer actual settings from host env."""
    actual = {"engine": scene.render.engine,
              "width": scene.render.resolution_x,
              "height": scene.render.resolution_y,
              "resolution_percentage": scene.render.resolution_percentage,
              "compute_backend": backend, "devices": devices}
    if hasattr(scene.render, "threads_mode"):
        actual.update(threads_mode=scene.render.threads_mode,
                      threads=scene.render.threads if scene.render.threads_mode == "FIXED" else None)
    if scene.render.engine == "CYCLES":
        actual.update(samples=scene.cycles.samples,
                      denoise=scene.cycles.use_denoising,
                      denoiser=scene.cycles.denoiser if scene.cycles.use_denoising else None,
                      adaptive_sampling=scene.cycles.use_adaptive_sampling,
                      adaptive_threshold=scene.cycles.adaptive_threshold)
    elif scene.render.engine == "BLENDER_WORKBENCH":
        actual.update(samples=None, antialiasing=scene.display.render_aa)
        names = ("light", "studio_light", "color_type", "show_shadows", "shadow_intensity",
                 "show_cavity", "cavity_type", "cavity_ridge_factor", "cavity_valley_factor",
                 "curvature_ridge_factor", "curvature_valley_factor", "show_specular_highlight",
                 "show_object_outline", "object_outline_color", "show_backface_culling",
                 "background_type", "background_color", "single_color")
        shading = scene.display.shading
        values = {name: getattr(shading, name) for name in names if hasattr(shading, name)}
        actual["workbench_shading"] = {name: value if isinstance(value, (str, bool, int, float))
                                      else list(value) for name, value in values.items()}
    else:
        actual["samples"] = scene.eevee.taa_render_samples
    return actual
