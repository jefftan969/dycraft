# --- scene-layer tool stubs (appended after _harness_tools.py) ---
# The scene layer's tools are the shared ones plus the object-dispatch calls
# and a `review` that takes `targets`.


def review(targets=None):
    """Ask a FRESH sub-agent whether you built what you promised. Advisory.

    It is given the `add_check(...)` claims and the rendered geometry, and
    nothing else — not your source, not your assertions, not your deviations.
    It answers each claim `met`, `not_met`, or `cant_tell`, citing the image it
    judged from, and can render extra cameras of its own to settle a
    `cant_tell`. It is also asked, separately and afterwards, whether the
    scene answers the original request (`prompt_match`).

    `targets` selects what is judged, in ONE reviewer run:

        review()                              # the scene's claims
        review(targets=["scene", "lathe"])    # the scene and one definition

    A target is `"scene"` or a **definition** name — never an instance name.
    One definition placed three times is one geometry; anything that differs
    between those placements is a placement question, which is a scene claim.
    Claim ids come back prefixed (`scene:c01`, `lathe:c01`) because both
    namespaces start at `c01`. At most 6 targets.

    Reviewing together is much cheaper than reviewing separately: most of what
    a review costs is starting the sub-agent.

    Returns `{"ok", "reviewed", "results", "by_target", "targets",
    "prompt_match", "tally", "extra_renders", "images", "sub_run", "outcome",
    "wall_time_s", "note"}`.

    **Nothing here blocks `finish()`.** Takes a nested agent run, usually minutes.
    Requires declared claims; without them there is nothing to check.
    """
    return _call("review", {"targets": targets})


def submit_object(name, brief, size=None, interfaces=None, materials=None,
                  states=None):
    """Start an object agent on `brief` in the background. Returns at once.

    Delegate an object when doing so saves end-to-end time: when writing the
    brief and integrating the result is less work than modeling the object
    yourself to the quality the scene needs. That takes both detail AND
    encapsulation — if specifying it means transmitting a surface it must match
    exactly, it belongs in your own file.

        submit_object(name="bench_lathe",
                      brief="A benchtop metal lathe with headstock, ...",
                      size=(1.10, 0.45, 0.55),
                      interfaces=["sits on a 0.90 m bench"],
                      materials=["cast iron, machined bright ways"],
                      states=["tailstock retracted", "tailstock advanced"])

    The child receives your brief, the fields you supply, and the original
    scene request. Every field is optional; a detail that does not constrain
    the object is noise in a brief.

    The child writes `objects/<name>/object.py`. Bring it in with
    `load_object("objects/<name>/object.py")` — and then that source is yours
    to edit.

    Delegate the objects that earn it, not as a matter of course: you author
    most of this scene yourself, and a brief plus an integration is only
    cheaper than modeling when the object is genuinely self-contained.
    Eight concurrent dispatches is the ceiling the harness enforces, not a
    target; the refusal names the live cap. Continue independent work while
    it runs; the point
    is that you keep authoring while it builds.
    """
    return _call("submit_object", {
        "name": name, "brief": brief, "size": size,
        "interfaces": interfaces, "materials": materials, "states": states})


def await_any(dispatch_ids=None, timeout=None):
    """Take the next finished dispatch. Each landing is handed back once.

    `timeout=0` polls and returns immediately — the cheap check between edits.
    Without a timeout it blocks until something finishes. `dispatch_ids`
    narrows the wait; omitted, it watches everything.

    A committed result carries `source_path` (run-relative, ready for
    `load_object`), `ir_path`, and the child's `deviations` — which are
    reported for you to act on, accept, or overrule, not applied for you.
    A failed one carries `status` and where to read what went wrong.
    """
    return _call("await_any", {"dispatch_ids": dispatch_ids,
                               "timeout": timeout})


def dispatch_status():
    """What is running and what has landed, without taking any result."""
    return _call("dispatch_status", {})


def cancel_dispatch(dispatch_id):
    """Stop a dispatch you have decided not to wait for.

    `finish()` refuses while any dispatch is still running, because committing
    then would ship a scene missing work in flight. This is how you commit
    without it.
    """
    return _call("cancel_dispatch", {"dispatch_id": dispatch_id})
