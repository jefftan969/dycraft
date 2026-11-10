# --- object-only tool stubs (appended after _harness_tools.py) ---
# The object layer's tools are the shared ones plus `review`. It takes no
# targets: an object run has one model.


def review():
    """Ask a FRESH sub-agent whether you built what you promised. Advisory.

    It is given your `add_check(...)` claims, the rendered geometry, and the
    user's request, and nothing else — not your source, not your assertions,
    not your deviations. It answers each claim `met`, `not_met`, or
    `cant_tell`, citing the image it judged from, and it can render extra
    cameras of its own to settle a `cant_tell`. It is also asked, separately
    and afterwards, whether the object answers the request (`prompt_match`).

    Returns `{"ok", "reviewed", "results": [{"id", "verdict", "view",
    "reason"}], "by_target", "targets", "prompt_match", "tally",
    "extra_renders", "images", "sub_run", "outcome", "wall_time_s", "note"}`.

    **Nothing here blocks `finish()`.** It is information for your next
    iteration. A fresh pair of eyes that never saw your reasoning catches
    what re-reading your own file does not.

    Costs a nested agent run, usually minutes. Call it once the geometry has
    settled, not on a draft. Requires `add_check(...)` claims in the compiled
    IR; without them there is nothing to check.
    """
    return _call("review", {})
