"""One unit on the surface: zero-based output frames, declarations included.

Event and collision-allowance windows are written in frame indices like
`snapshot`, `plan` states, `ik`, `probe`, and the check interval, so an author
never converts between units by hand.

Windows live in the saved blend, so the unit cannot simply be reinterpreted: a
window of `[1, 2]` means two seconds in a blend whose `units` say seconds and
two frames in one whose `units` say frames. `initialize` seeds `frames` for
every new project, and a blend without the field reads as seconds.
"""
import pytest

from dycraft.animation.blender.verify import (allowed, clip_offset, declared_units,
                                              resolve_declarations, window_indices)

INFO = {"fps": 24.0, "n_frames": 240, "frame_start": 0}
BASELINE = {"identities": {"x/p": {}, "y/q": {}}}


def test_a_project_without_the_field_keeps_meaning_seconds():
    """The native layer's unit, which it documents and never names. An animation
    project names frames, and its edits keep the name (`keep_declared_units`)."""
    assert declared_units({}) == "seconds"
    assert declared_units({"units": "seconds"}) == "seconds"
    assert declared_units({"units": "frames"}) == "frames"
    with pytest.raises(ValueError, match="frames.*seconds"):
        declared_units({"units": "milliseconds"})


def test_the_same_window_resolves_differently_under_each_unit():
    seconds = window_indices(INFO, {"units": "seconds"}, [1.0, 1.5])
    frames = window_indices(INFO, {"units": "frames"}, [1, 2])
    assert (seconds[0], seconds[-1]) == (24, 36)
    assert frames == [1, 2]


def test_a_clip_relative_window_scales_with_its_strip_like_a_state_does():
    """States, cuts, and windows name a place in a clip the same way, so they scale alike.

    If only states were scaled, a retime would leave event windows past the end
    of their clip. 8 action frames into a clip at scale 0.75 is 6 output
    frames.
    """
    info = {**INFO, "clips": {"run": {"frames": [16, 100], "cut": 16, "scales": [0.75]}}}
    declarations = {"units": "frames"}
    assert window_indices(info, declarations, {"clip": "run", "window": [8, 16]}) == list(range(22, 29))
    # A span recorded before scales were carried reads as unscaled.
    legacy = {**INFO, "clips": {"run": {"frames": [16, 100], "cut": 16}}}
    assert window_indices(legacy, declarations, {"clip": "run", "window": [8, 16]}) == list(range(24, 33))


def test_a_clip_whose_strips_disagree_on_scale_names_no_single_frame():
    span = {"frames": [0, 50], "scales": [0.5, 1.0]}
    with pytest.raises(ValueError, match="different scales"):
        clip_offset(span, 4, clip="run", subject="cut to 'Cam'")
    assert clip_offset({"frames": [0, 50], "scales": [2.0]}, 4, clip="run", subject="x") == 8
    # Counted from the action's first frame: a strip plays its action from
    # there, so an action keyed from frame 3 puts its frame 5 two frames in.
    assert clip_offset({"frames": [0, 50], "scales": [2.0], "starts": [3.0]}, 5,
                       clip="run", subject="x") == 4
    with pytest.raises(ValueError, match="starting on different frames"):
        clip_offset({"frames": [0, 50], "starts": [0.0, 3.0]}, 5, clip="run", subject="x")
    # A frame before the action's first is not in the clip.
    with pytest.raises(ValueError, match="action starts on frame 3"):
        clip_offset({"frames": [0, 50], "starts": [3.0]}, 1, clip="run", subject="x")


def test_a_frame_window_must_be_frame_indices():
    """Floats are rejected rather than rounded, the way the interval args are.

    A float here is almost always seconds written under the new unit by habit,
    and rounding it would check frame 1 for a window meant to span a second.
    """
    with pytest.raises(ValueError):
        window_indices(INFO, {"units": "frames"}, [1.0, 1.5])


def test_an_allowance_is_gated_by_the_frames_verify_resolved_for_it():
    """`allowed` compares integers now, against a window resolved once.

    Comparing `frame / fps` in seconds would re-derive a window for every
    candidate pair on every checked frame, in the inner loop of
    `NoPenetration`.
    """
    declarations = {"units": "frames", "collision_allowances": [
        {"a": "x/p", "b": "y/q", "reason": "test", "window": [10, 20]}]}
    index, _, errors = resolve_declarations(declarations, BASELINE, INFO)
    assert errors == []
    assert not allowed(index, "x/p", "y/q", 9)
    assert allowed(index, "x/p", "y/q", 10)
    assert allowed(index, "y/q", "x/p", 20)
    assert not allowed(index, "x/p", "y/q", 21)


def test_an_allowance_without_a_window_still_covers_the_whole_timeline():
    declarations = {"collision_allowances": [
        {"a": "x/p", "b": "y/q", "reason": "whole film"}]}
    index, _, _ = resolve_declarations(declarations, BASELINE, INFO)
    assert allowed(index, "x/p", "y/q", 0)
    assert allowed(index, "x/p", "y/q", 10**6)
    assert not allowed(index, "x/p", "other", 0)
    # Resolution never writes to the record: the resolved span lives in the
    # index, not on the author's declaration.
    assert declarations["collision_allowances"][0] == {
        "a": "x/p", "b": "y/q", "reason": "whole film"}


# ---------------------------------------------------------------------------
# `planted`: the contact patch a support bears on
# ---------------------------------------------------------------------------

def test_contact_patch_is_the_bearing_band_not_the_whole_body():
    """`planted` measures where a support *bears*, not where its body is.

    A foot's vertices run from the sole to the ankle. Averaging all of them
    reports a point that moves whenever the ankle rotates, so a foot that
    stays put under a turning leg would read as sliding. Averaging the lowest
    band reports the sole.
    """
    import numpy as np
    from dycraft.animation.blender.verify import contact_patch_xy

    sole = [[0, 0, 0], [.2, 0, 0], [.2, .1, 0], [0, .1, 0]]
    tall = sole + [[0, 0, .3], [.2, 0, .3], [.2, .1, .3], [0, .1, .3]]
    assert np.allclose(contact_patch_xy(np.array(tall)), [.1, .05])
    # The mass above the sole must not move the answer.
    assert np.allclose(contact_patch_xy(np.array(sole)),
                       contact_patch_xy(np.array(tall)))


def test_contact_patch_ignores_vertices_above_the_band():
    """Only what is within `_PATCH_BAND_M` of the lowest point is bearing."""
    import numpy as np
    from dycraft.animation.blender.verify import contact_patch_xy
    from dycraft.animation.blender.verify_support import _PATCH_BAND_M

    # Two vertices on the floor, two lifted well clear of the band.
    tilted = np.array([[0, 0, 0], [0, .1, 0],
                       [.2, 0, 10 * _PATCH_BAND_M], [.2, .1, 10 * _PATCH_BAND_M]])
    assert np.allclose(contact_patch_xy(tilted), [0., .05])


# ---------------------------------------------------------------------------
# `supported` / `SupportMargin` / `ContactSlip`
# ---------------------------------------------------------------------------

def test_support_margin_is_signed_and_winding_agnostic():
    """Negative inside, positive outside, whichever way the hull is wound."""
    import numpy as np
    from dycraft.animation.blender.verify import support_margin

    for square in ([[0, 0], [1, 0], [1, 1], [0, 1]],
                   [[0, 0], [0, 1], [1, 1], [1, 0]]):
        assert support_margin(np.array([.5, .5]), square) == pytest.approx(-.5)
        assert support_margin(np.array([1.5, .5]), square) == pytest.approx(.5)
    # Two contact points are a line, not a polygon to be inside of.
    assert support_margin(np.array([0, 0]), [[0, 0], [1, 1]]) is None


def test_mass_table_does_not_mix_declared_and_geometric_weights():
    """One scale per body, or a declared 0.6 kg foot outweighs nothing.

    The G1's foot declares 0.608 kg and measures 2e-06 m^3. Weighing it in
    kilograms beside a sibling in cubic meters puts the center of mass in the
    foot; the table falls the whole body back to geometry instead.
    """
    from dycraft.animation.blender.verify import part_mass_table

    def part(name, mass, volume):
        params = {"shape": {"volume": volume, "center": [0., 0., 0.]},
                  "aabb_min": [-.1, -.1, -.1], "aabb_max": [.1, .1, .1]}
        if mass is not None:
            params["inertial"] = {"mass": mass, "com": [0., 0., 0.], "source": "urdf"}
        return {"name": name, "params": params}

    scene = {"instances": [{"name": "robot", "object_ref": "ref"}]}
    both = {"ref": {"parts": [part("a", 0.608, 2e-06), part("b", None, 1e-03)]}}
    table = part_mass_table(scene, both)
    assert [round(table[k][0], 8) for k in ("robot/a", "robot/b")] == [2e-06, 1e-03]
    # the local box travels with the weight, so a centroid can be placed from
    # this frame's world bounds rather than from a transform that may be identity
    assert len(table["robot/a"]) == 4

    declared = {"ref": {"parts": [part("a", 0.608, 2e-06), part("b", 1.5, 1e-03)]}}
    table = part_mass_table(scene, declared)
    assert [table[k][0] for k in ("robot/a", "robot/b")] == [0.608, 1.5]


def test_slip_report_separates_a_slide_from_a_walk():
    """Creep alone cannot; creep with intervals and clearance can."""
    from dycraft.animation.blender.verify import slip_evidence

    slide = {"robot/foot": {"anchor": [0, 0], "intervals": 0, "frames": 100,
                            "clearance": 0.0, "truncated": False, "first": 0, "previous": None, "gaps": 0,
                            "slips": [(i, .003 * i) for i in range(100)]}}
    flagged, evidence = slip_evidence(slide, 12.0)
    assert flagged and evidence["parts"][0]["bearing_intervals"] == 1
    assert evidence["parts"][0]["max_clearance_m"] == 0.0

    walk = {"robot/foot": {"anchor": None, "intervals": 14, "frames": 80,
                           "clearance": .05, "truncated": False, "first": 0, "previous": None, "gaps": 0,
                           "slips": [(i, .001) for i in range(80)]}}
    flagged, evidence = slip_evidence(walk, 12.0)
    assert not flagged
    assert evidence["parts"][0]["bearing_intervals"] == 14


def test_support_margin_and_contact_slip_are_advisory():
    """Neither gates: `required_checks_pass` must not consult them."""
    from dycraft.animation.blender.verify import (VERIFIER_NAMES, report,
                                                  slip_evidence, support_evidence)

    assert {"SupportMargin", "ContactSlip"} <= set(VERIFIER_NAMES)
    # (frame, margin, parts bearing) — one foot down, mass well outside it.
    bad = {"robot": [(i, .2, 1) for i in range(30)]}
    flagged, kwargs = support_evidence(bad, 12.0)
    assert report("SupportMargin", flagged, advisory=True, **kwargs)["verdict"] == "uncertain"
    worst = {"robot/foot": {"anchor": [0, 0], "intervals": 0, "frames": 10,
                            "clearance": 0.0, "truncated": False, "first": 0, "previous": None, "gaps": 0,
                            "slips": [(i, .5) for i in range(10)]}}
    flagged, kwargs = slip_evidence(worst, 12.0)
    row = report("ContactSlip", flagged, advisory=True, **kwargs)
    assert row["verdict"] == "uncertain"
    assert row["evidence"]["diagnostic_only"] is True


def test_support_is_reported_by_phase_not_pooled():
    """Pooling a one-foot phase with a two-foot one reports the duty factor.

    A walking body is usually outside its support on one foot and rarely on
    two, so a pooled figure says how much of the film was single support, not
    whether the robot was balanced. The phase split is what makes the same body
    unambiguous.
    """
    from dycraft.animation.blender.verify import support_evidence

    series = ([(i, .2, 1) for i in range(480)]          # one foot, outside
              + [(500 + i, -.05, 2) for i in range(520)])  # two feet, inside
    flagged, evidence = support_evidence(series and {"robot": series}, 12.0)
    row = evidence["bodies"][0]
    assert row["pooled"]["outside_fraction"] < .5, "pooled hides it"
    assert row["by_bearing_parts"]["1"]["outside_fraction"] == 1.0
    assert row["by_bearing_parts"]["2"]["outside_fraction"] == 0.0
    assert flagged and flagged[0]["bearing_parts"] == 1


def test_a_brief_single_support_pass_is_not_flagged():
    """`phase_minimum` keeps a few frames of a turn from reading as a stance."""
    from dycraft.animation.blender.verify import support_evidence

    series = [(i, .2, 1) for i in range(5)] + [(10 + i, -.05, 2) for i in range(200)]
    flagged, _ = support_evidence({"robot": series}, 12.0)
    assert not flagged


def test_support_polygon_uses_the_footprint_not_a_centroid():
    """One bearing part must still give a polygon, or a biped is never measured.

    `contact_patch_xy` reduces a sole to one point, which is right for tracking
    creep and wrong here: two feet would be a line and one foot a point, so
    every biped would fall through the `len(patch) < 3` guard unmeasured.
    """
    import numpy as np
    from dycraft.animation.blender.verify import (contact_patch_outline,
                                                  contact_patch_xy,
                                                  support_margin)

    sole = np.array([[x, y, 0.] for x in (-.0658, .1424) for y in (-.0377, .0378)]
                    + [[0., 0., .3]])
    assert len(contact_patch_xy(sole).shape) == 1          # one point
    outline = contact_patch_outline(sole)
    assert len(outline) >= 3, "one sole must bound a polygon on its own"
    assert support_margin(np.array([.04, 0.]), outline) < 0      # over the sole
    assert support_margin(np.array([.20, 0.]), outline) > 0      # off the toe


# ---------------------------------------------------------------------------
# The worker interpreter's dependencies
# ---------------------------------------------------------------------------

def test_worker_side_modules_import_nothing_blender_lacks():
    """`verify.py` runs inside Blender's interpreter, which has no scipy.

    An optional import such as `from scipy.spatial import ConvexHull` imports
    on the host and falls back to `None` in the worker, so a verifier can
    measure nothing while reporting a pass, and every host-side test of it
    still passes, because the tests and the worker run different interpreters.

    Walks what `verify.py` pulls in at module scope, within the two trees the
    worker puts on its path, and refuses the packages that interpreter lacks.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    trees = (root / "animation" / "blender", root / "verify")
    forbidden = {"scipy", "sklearn", "pandas", "matplotlib", "torch"}

    def module_level_imports(path):
        tree = ast.parse(path.read_text())
        for node in tree.body:                      # module scope only
            if isinstance(node, ast.Import):
                yield from (a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    yield node.module.split(".")[0]
                elif node.module:
                    yield node.module.split(".")[0]
            elif isinstance(node, ast.Try):         # the `.x`/`x` fallback pair
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import):
                        yield from (a.name.split(".")[0] for a in sub.names)
                    elif isinstance(sub, ast.ImportFrom) and sub.module:
                        yield sub.module.split(".")[0]

    seen, queue, offences = set(), ["verify.py"], []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = next((tree / name for tree in trees if (tree / name).is_file()), None)
        if path is None:
            continue
        for imported in module_level_imports(path):
            if imported in forbidden:
                offences.append(f"{path.name} imports {imported} at module scope")
            candidate = f"{imported}.py"
            if any((tree / candidate).is_file() for tree in trees):
                queue.append(candidate)

    assert not offences, (
        "these run in Blender's interpreter and cannot import these packages: "
        + "; ".join(sorted(offences)))
