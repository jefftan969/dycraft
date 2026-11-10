# --- object/scene tool stubs (appended to _harness_core.py at stage time) ---
# Each function forwards to the daemon's handler of the same name. `_call` is
# defined in the transport core above.

def compile(freeze: bool = False):
    return _call("compile", {"freeze": bool(freeze)})


def finish():
    return _call("finish", {})


def probe(expression: str):
    """Evaluate one expression, or a block of statements that sets `result`,
    against the current source. Returns `{"ok", "repr", "truncated"}`."""
    return _call("probe", {"expression": expression})


def snapshot(views=None, poses=None, only=None, resolution=None,
             out_dir: str = "snapshots",
             eye=None, look_at=None, up=None, fov_deg: float = 60.0):
    """Render the current geometry from cameras YOU choose.

    Returns `{"ok", "paths", "n_images", "out_dir", "views", "image_size",
    "poses", "pose_label", "only", "target_bbox", "cache_hit",
    "render_warnings", "note"}`. `paths` are PNGs on disk in the order you
    asked for them — **nothing looks at them but you.** `views` carries the
    label and the resolved camera per image; a custom camera's FILENAME is a
    geometry hash, so it is the label that says what a picture is of.

    Costs about a second, plus a fraction of a second per extra view — every
    camera of a kind renders in one pass, so three views in one call is much
    cheaper than three calls.

    `views` — one spec or a list of up to 12, in three forms. You will almost
    always want the second.

      * `"front"` / `"right"` / `"back"` / `"left"` / `"top"` / `"bottom"` —
        a canonical camera fitted to the target bounds and requested resolution.
      * `{"on": "visor", "azimuth": 45, "elevation": 30}` — say WHERE FROM and
        WHAT AT, and the harness solves for the distance that fits it.
        **Prefer this over computing an `eye` yourself.** Framing on the part
        you care about rather than the whole object is what makes a small
        feature legible; scale does not matter, a 0.25 mm boss frames exactly
        like a 1.6 m table top.
          - `on` — `"all"`, a part name, a point `[x,y,z]` (what
            `anchor_position` returns), or a box `[[x0,y0,z0],[x1,y1,z1]]`.
          - `azimuth` — degrees in the XY plane, 0 = +X, 90 = +Y (the front).
          - `elevation` — degrees above horizontal; 90 looks straight down.
          - `roll` — degrees about the view axis, for when the subject is
            long and thin and you want it across the frame.
          - `margin` — padding as a fraction, default 0.05. **Negative
            magnifies**: the fit contains the whole bounding box, so a squat or
            deep part is held back by its nearest corner and its silhouette
            lands around half the frame. `margin=-0.5` takes that to roughly
            three quarters. It stops there by design — the camera will not go
            inside the part.
          - `distance` — meters from the target, replacing the solve. Required
            for a point `on`, since a point has no extent to fit.
          - `fov_deg` — the lens, default 60. **`distance` plus a narrow
            `fov_deg` is the way to fill the frame completely**: at
            `distance` = twice the part's diagonal and `fov_deg=12`, a 20 mm
            boss and a 1.6 m table top both render at 95-100% of frame.
      * `{"eye": [x,y,z], "look_at": [x,y,z]}` — absolute world meters, +Z up,
        for what the declarative form cannot say. Takes `up`, `roll`, and
        `fov_deg`. `up` defaults to +Z, or to the canonical `top`/`bottom`
        fallback for a near-vertical view; read `views[i].camera.up` and `note`
        back rather than assuming. An `up` you pass is never silently
        corrected.

    `poses` — `{joint_name: value}`, degrees for a revolute joint and meters
    for a prismatic one, i.e. the same units as `limits` and `default` in your
    source. Joints you do not name stay at their authored default. Use this
    to select a particular state, such as a collision's `witnessing_pose`.
    Independent review selects its own articulation evidence.

    `only` — part names to DRAW, hiding everything else. `on` re-frames;
    `only` unburies. Use it for a contact face, a bore, an interior fit, or
    anything behind a shell.

    `resolution` — `(width, height)`, 320..2048 per side, at most 2,073,600
    pixels total (1920×1080). Any aspect ratio; fitted cameras use the
    requested frame.

    `render_warnings` names render diagnostics, including a mesh replaced by a
    bounding box, a shader replaced by scalar fallback, or missing lighting.
    Strict renders fail on shader exceptions instead. Cached frames may return
    no warnings; absence alone does not establish a clean render.
    """
    return _call("snapshot", {
        "views": views,
        "poses": poses,
        "only": only,
        "resolution": (None if resolution is None
                       else [int(v) for v in resolution]),
        "out_dir": out_dir,
        "eye": (None if eye is None else [float(v) for v in eye]),
        "look_at": (None if look_at is None
                    else [float(v) for v in look_at]),
        "up": (None if up is None else [float(v) for v in up]),
        "fov_deg": float(fov_deg),
    })


def final_render(failed: bool = False):
    """Render the current IR's 6 canonical views + URDF. On success
    (default) drops them under `<run_dir>/final_render/`; pass
    `failed=True` to write to `<run_dir>/final_render_failed/` instead
    (best-effort: any compile/render exception is swallowed and an
    empty `final_render_failed/` is left behind so failure-path callers
    don't need a try/except)."""
    return _call("final_render", {"failed": bool(failed)})


def record_deviation(what: str, why: str, severity: str,
                     suggested_fix: str | None = None,
                     blocked_layer: str | None = None,
                     targets=None):
    """Record a deliberate departure from your brief — a part skipped, a
    spec simplified, a clause reinterpreted, a verifier-driven sign flip.
    Saved in meta.json's `deliberate_deviations` on commit and returned to
    the agent that dispatched you. Severity is 'low' or 'high'. Does NOT
    block commit.

    `blocked_layer` — which OTHER layer has to change before this can be
    fixed: "object" (a part or joint is missing / its range is too small) or
    "scene" (the geometry is fine, the placement makes it unusable). **None
    is a legitimate answer** meaning you cannot tell; say what you measured
    in `why`. It is a hint, never an instruction.

    `targets` — the instances this is about, by the names you can see. One
    name, or two for a conflict between a pair (both, because the fix could
    be either one). A bare string is accepted for the single-name case."""
    return _call("record_deviation", {
        "what": what, "why": why, "severity": severity,
        "suggested_fix": suggested_fix,
        "blocked_layer": blocked_layer, "targets": targets,
    })


def set_render_backend(backend: str):
    """Point every subsequent render at `backend` ("open3d" or "blender").

    Use this instead of restarting the harness. A run directory has exactly
    one harness daemon; launching a second one to change
    `DYCRAFT_RENDER_BACKEND` is refused, because two daemons write the same
    `trace.jsonl`, `meta.json`, and step counter and corrupt all three.

    Open3D renders canonical stills quickly, but it does not
    evaluate `@shader` graphs — shader-only materials preview as neutral
    gray. Switch to "blender" when the judgment you need is about a
    procedural material, and expect renders to take longer.

    Returns `{"ok": True, "render_backend": ..., "previous": ...}`, or
    `{"ok": False, "error": ...}` for an unknown name.
    """
    return _call("set_render_backend", {"backend": str(backend)})
