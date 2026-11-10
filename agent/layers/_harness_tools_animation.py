# Job control: `job_status`, `job_result`, and `job_cancel`.

def job_status(job_id: str):
    """The job's status, without its result."""
    return _call("job_status", {"job_id": job_id})


def job_result(job_id: str, wait: float = 0):
    """The job's record. `wait=s` blocks up to `s` seconds for it to finish,
    then returns the record either way; keep `s` below your shell tool's own
    timeout."""
    if wait and wait > 0:
        return _await(job_id, float(wait), TIMEOUT_S)
    return _call("job_result", {"job_id": job_id})


def job_cancel(job_id: str):
    """Cancel a job; `'current'` cancels the call in progress."""
    return _call("job_cancel", {"job_id": job_id})


# The animation client. Job control (`job_status`, `job_result`,
# `job_cancel`) is defined above.
#
# Every project operation takes `batch=` and `fresh=`; the contributor calls at the end
# (`dispatch_*`, `contribution_status`, `cancel_contribution`) do not, since a dispatch
# returns at once. A call runs now and returns its result, except `compile`, `final_render`, `finish`, `review`, `ik`, and
# `render(delivery=True)`, which run as jobs: the call waits for the job and
# returns its record, or the record of a job still running, to collect with
# `job_result(job_id, wait=s)`. `batch=True` returns any call's job at once.
# `fresh=True` restarts the project's Blender process first and redoes what
# would otherwise be reused. A reply's `version` names the project state it
# describes.

def plan(plan, batch=None, fresh=False):
    """Validate and store the plan: intent, criteria, states, and shots."""
    return _call("plan", {"batch": batch, "fresh": fresh, "plan": plan}, wait=_batch_wait(batch))


def compile(start=None, end=None, seconds=None, shot=None, clip=None, workers=None,
            batch=None, fresh=False):
    """Check every frame of the film and record the verdict; with an interval it is
    `check_motion`. `workers=N` splits a whole-film check across up to N processes."""
    return _call("compile", {"batch": batch, "fresh": fresh, "start": start, "end": end,
                             "seconds": seconds, "shot": shot, "clip": clip,
                             "workers": workers}, wait=_batch_wait(batch))


def check_motion(start=None, end=None, seconds=None, shot=None, clip=None, workers=None,
                 batch=None, fresh=False):
    """Check every frame of an inclusive interval (`start`/`end` output indices,
    `seconds`, `shot`, or `clip`); events not wholly inside it are deferred."""
    return _call("check_motion", {"batch": batch, "fresh": fresh, "start": start, "end": end, "seconds": seconds,
                                  "shot": shot, "clip": clip, "workers": workers},
                 wait=_batch_wait(batch))


def render(frames=None, start=None, end=None, seconds=None, shot=None, clip=None,
           camera_name=None, eye=None, look_at=None, up=None, fov_deg=60.,
           encode=None, render_settings=None, delivery=False, batch=None, fresh=False):
    """Draw frames to look at, through the native cuts, a named camera, or a placed eye.

    Returns `frames_dir`, `frame_indices`, and `frame_file` (a frame is
    `frames_dir/frame_file % i`), a contact `sheet`, and an mp4 `path` for a sequence.
    `delivery=True` draws verified frames for `final_render(frames_from=[...])`."""
    return _call("render", {"batch": batch, "fresh": fresh, "frames": frames, "start": start, "end": end,
                            "seconds": seconds, "shot": shot, "clip": clip,
                            "camera_name": camera_name,
                            "eye": eye, "look_at": look_at, "up": up, "fov_deg": fov_deg,
                            "encode": encode, "render_settings": render_settings,
                            "delivery": delivery}, wait=_batch_wait(batch))


def build(refresh=None, candidate=None, repair_code=None, accept_changes=None, batch=None, fresh=False):
    """Compile `design/` into the project. `refresh={'materials': [...], 'lighting': True}`
    re-imports appearance; `candidate`, `accept_changes`, and `repair_code` publish an
    incompatible change."""
    return _call("build", dict(batch=batch, fresh=fresh, refresh=refresh, candidate=candidate,
                               repair_code=repair_code, accept_changes=accept_changes), wait=_batch_wait(batch))


def blender(code, name=None, stage="animation", invalidate_shots=None, span=None, shot=None,
            batch=None, fresh=False):
    """Run your Bpy on the project and save it. `stage='storyboard'` for cameras and static
    states; an animation edit needs the storyboard accepted, or only the states inside
    `span=(start_state, end_state)` or `shot='id'`."""
    return _call("blender", dict(batch=batch, fresh=fresh, code=code, name=name, stage=stage,
                                 invalidate_shots=invalidate_shots, span=span, shot=shot),
                 wait=_batch_wait(batch))


def snapshot(eye=None, look_at=None, up=None, fov_deg=60., frame=None, frames=None,
             camera_name=None, start=None, end=None, seconds=None, shot=None,
             render_settings=None, batch=None, fresh=False):
    """`render` without encoding: stills only; `frame=` for one."""
    return _call("snapshot", dict(batch=batch, fresh=fresh, eye=eye, look_at=look_at, up=up, fov_deg=fov_deg,
                                 frame=frame, frames=frames, camera_name=camera_name,
                                 start=start, end=end, seconds=seconds, shot=shot,
                                 render_settings=render_settings), wait=_batch_wait(batch))


def storyboard(states=None, render_settings=None, batch=None, fresh=False):
    """Check the planned states and render them through the native cuts; a state whose
    design, pose, and settings are unchanged keeps its board and review."""
    return _call("storyboard", {"batch": batch, "fresh": fresh, "states": states,
                                "render_settings": render_settings}, wait=_batch_wait(batch))


def accept_storyboard(reviews, batch=None, fresh=False):
    """Record your observations, `{state_id: text}`, for the panels you inspected."""
    return _call("accept_storyboard", {"batch": batch, "fresh": fresh, "reviews": reviews}, wait=_batch_wait(batch))


def motion_image(frames, eye=None, look_at=None, up=None, fov_deg=60., camera_name=None,
                 gain=4., render_settings=None, batch=None, fresh=False):
    """Two or three frames and a temporal RGB composite that locates change."""
    return _call("motion_image", dict(batch=batch, fresh=fresh, frames=frames, eye=eye, look_at=look_at, up=up,
                                    fov_deg=fov_deg, camera_name=camera_name, gain=gain,
                                    render_settings=render_settings), wait=_batch_wait(batch))


def review(focus=None, frames=None, render_settings=None, batch=None, fresh=False):
    """An advisory second opinion on the planned states and up to 32 `frames`; `focus`
    says what to look at."""
    return _call("review", {"batch": batch, "fresh": fresh, "focus": focus, "frames": frames,
                            "render_settings": render_settings}, wait=_batch_wait(batch))


def final_render(frames_from=None, render_settings=None, batch=None, fresh=False):
    """Render and encode the delivery film, composing the frames of `frames_from`
    delivery renders, and write the receipt `finish()` reads."""
    return _call("final_render", {"batch": batch, "fresh": fresh,
                                  "frames_from": frames_from,
                                  "render_settings": render_settings}, wait=_batch_wait(batch))


def finish(criteria=None, exceptions=False, batch=None, fresh=False):
    """Commit the project and its film. `criteria={id: {'verdict': 'pass' | 'fail' |
    'uncertain', 'evidence': str}}` for every plan criterion; `exceptions=True` commits
    despite `unmet`, as `delivered_with_exceptions`."""
    return _call("finish", {"batch": batch, "fresh": fresh, "criteria": criteria,
                            "exceptions": exceptions},
                 wait=_batch_wait(batch))


def probe(expression, frame=None, frames=None, batch=None, fresh=False):
    """Evaluate Bpy, an expression or a block that sets `result`, at `frame` or at each of
    `frames`, without saving."""
    return _call("probe", {"batch": batch, "fresh": fresh, "expression": expression,
                           "frame": frame, "frames": frames},
                 wait=_batch_wait(batch))


def inspect(frames, names=None, offset=0, overview=False, batch=None, fresh=False):
    """Evaluated positions, bounds, types, and visibility at 1 to 4 frames; `overview=True`
    adds the room surfaces and grouped bounds."""
    return _call("inspect", {"batch": batch, "fresh": fresh, "frames": frames,
                             "names": names, "offset": offset, "overview": overview},
                 wait=_batch_wait(batch))


def ik(code=None, name=None, frame=None, frames=None, script=None, timeout_s=None,
       batch=None, fresh=False):
    """Run your cuRobo code on the GPU in the run's persistent `ik/` directory.

    Pass `code` (Python source) or `script` (a file under the run dir, e.g.
    `ik/plan.py` written from your shell). Build the robot and world yourself:
    an original URDF, cuRobo's bundled robot configs, or the project exported
    at `frame`/`frames` (up to 16 indices) into `ik/exports/`, whose paths the
    code reads from `DYCRAFT_IK_EXPORTS` and the reply names. Nothing is
    exported unless you ask. `ik/` persists between calls; `timeout_s` raises
    the default limit up to an hour. Returns what the code puts in `result`.
    """
    return _call("ik", {"batch": batch, "fresh": fresh, "code": code, "script": script,
                        "name": name, "frame": frame, "frames": frames, "timeout_s": timeout_s},
                 wait=_batch_wait(batch))


def dispatch_object(name, brief):
    """Launch a private object contributor; follow it with `contribution_status`."""
    return _call("dispatch_object", dict(name=name, brief=brief), wait=0)


def dispatch_scene(name, brief):
    """Launch a private scene contributor; follow it with `contribution_status`."""
    return _call("dispatch_scene", dict(name=name, brief=brief), wait=0)


def dispatch_motion(name, brief, clip=None):
    """Launch a private motion contributor; `clip` limits it to one clip and the states
    inside it."""
    return _call("dispatch_motion", dict(name=name, brief=brief, clip=clip), wait=0)


def contribution_status(job_id):
    """A contributor's job; its result names the `run_dir`, `source`, and `blend`, whether
    the contributor `finished` and `committed`, and how it ended (`ended_by`)."""
    return _call("contribution_status", {"job_id": job_id})


def cancel_contribution(job_id):
    """Cancel a contributor."""
    return _call("cancel_contribution", {"job_id": job_id})
