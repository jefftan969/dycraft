"""Tests for the SupportConnected verifier.

Goal: lenient support-connectivity via face contact, NOT static
equilibrium. The verifier should pass every realistic furniture shape
even when geometry alone wouldn't be statically stable (cantilevers,
overhanging compartments). It should only fail when a part has no
face-contact chain to the floor (the object's lowest z-plane).

Cases exercised (per design dialogue):
  * 4-legged table — passes (tabletop → leg → floor for each corner)
  * Cantilevered table — passes (single-leg chain still reaches floor)
  * Desk fan with overhanging head — passes (head → neck → base → floor)
  * Hammer lying flat — passes (head + handle both touch floor)
  * Pocketwatch on a flat surface — passes (single grounded part)
  * Humanoid (torso root, feet on the floor) — passes (floor = lowest
    z-plane, not the root's z-plane)
  * Floating decorative flag — fails (1 m offset from body, no contact)
"""

import pytest

from dycraft.sdk import Object
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.verify import _support as _sup
from dycraft.verify.object_verify import support_connected


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    yield
    _reset_object_singleton()


def _attach_leg_flush_corner(o, top, leg, corner: str):
    """Attach `leg` so the leg's `top_<corner>` corner coincides with
    the tabletop's `bottom_<corner>` corner — i.e., the leg is flush
    with the tabletop edge at that corner, not centered under it.
    """
    o.attach_fixed(
        f"j_{corner}", top, leg,
        parent_anchor=o.anchor(top, "bottom_" + corner),
        child_anchor=o.anchor(leg, "top_" + corner),
    )


# ---------------------------------------------------------------------------
# Should pass
# ---------------------------------------------------------------------------


def test_four_leg_table_passes():
    o = Object("table")
    top = o.make_box("top", 1.0, 0.6, 0.04)
    for corner in ("front_left", "front_right", "back_left", "back_right"):
        leg = o.make_box(f"leg_{corner}", 0.04, 0.04, 0.7)
        _attach_leg_flush_corner(o, top, leg, corner)
    o.declare_root(top)
    r = support_connected(o.finalize())
    assert r.verdict == "pass", r.evidence


def test_cantilever_table_passes():
    """Single flush corner-leg. Tabletop centroid is NOT above the
    leg's footprint — a strict centroid-over-polygon would fail this
    case; the lenient connectivity check passes it."""
    o = Object("cantilever_table")
    top = o.make_box("top", 1.0, 0.6, 0.04)
    leg = o.make_box("leg", 0.04, 0.04, 0.7)
    _attach_leg_flush_corner(o, top, leg, "front_right")
    o.declare_root(top)
    r = support_connected(o.finalize())
    assert r.verdict == "pass"


def test_desk_fan_with_overhanging_head_passes():
    """Heavy fan compartment overhangs a small base — would fail a
    centroid-over-support-polygon test but should pass the lenient
    connectivity rule (head → neck → base → floor)."""
    o = Object("fan")
    base = o.make_box("base", 0.20, 0.15, 0.10)
    neck = o.make_box("neck", 0.05, 0.05, 0.40)
    head = o.make_box("head", 0.30, 0.20, 0.30)
    o.attach_fixed("j_neck", base, neck,
        parent_anchor=o.anchor(base, "top"),
        child_anchor=o.anchor(neck, "bottom"))
    o.attach_fixed("j_head", neck, head,
        parent_anchor=o.anchor(neck, "top"),
        child_anchor=o.anchor(head, "bottom"))
    o.declare_root(base)
    r = support_connected(o.finalize())
    assert r.verdict == "pass"


def test_hammer_passes():
    """Head + handle both rest on the same z-plane (lowest z of the
    object). Both are directly grounded."""
    o = Object("hammer")
    handle = o.make_box("handle", 0.30, 0.03, 0.03)
    head = o.make_box("head", 0.05, 0.10, 0.04)
    o.attach_fixed("j", handle, head,
        parent_anchor=o.anchor(handle, "left"),
        child_anchor=o.anchor(head, "right"))
    o.declare_root(handle)
    r = support_connected(o.finalize())
    assert r.verdict == "pass"


def test_pocketwatch_single_part_passes():
    """A single-part object is trivially supported by its own floor plane."""
    o = Object("pocketwatch")
    body = o.make_box("body", 0.05, 0.05, 0.01)
    o.declare_root(body)
    r = support_connected(o.finalize())
    assert r.verdict == "pass"


def test_humanoid_root_grounded_through_feet():
    """The 'floor' is the object's lowest z-plane, not the root's
    z-plane. A torso-root with feet below it still grounds via the
    feet rather than directly through the torso."""
    o = Object("humanoid")
    torso = o.make_box("torso", 0.3, 0.2, 0.5)
    foot_l = o.make_box("foot_l", 0.1, 0.2, 0.05)
    foot_r = o.make_box("foot_r", 0.1, 0.2, 0.05)
    o.attach_fixed("j_l", torso, foot_l,
        parent_anchor=o.anchor(torso, "bottom_left"),
        child_anchor=o.anchor(foot_l, "top_left"))
    o.attach_fixed("j_r", torso, foot_r,
        parent_anchor=o.anchor(torso, "bottom_right"),
        child_anchor=o.anchor(foot_r, "top_right"))
    o.declare_root(torso)
    ir = o.finalize()
    r = support_connected(ir)
    assert r.verdict == "pass"
    assert set(r.evidence["grounded_directly"]) == {"foot_l", "foot_r"}


# ---------------------------------------------------------------------------
# Should fail
# ---------------------------------------------------------------------------


def test_floating_flag_fails():
    """A part 1 m above its kinematic parent with no face contact is
    floating despite the fixed attachment. The contact graph is what
    counts — kinematic edges don't create support."""
    o = Object("with_floater")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    flag = o.make_box("flag", 0.3, 0.02, 0.2)
    o.attach_fixed(
        "j_flag", body, flag,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 1.0)), child_anchor=o.anchor(flag, "bottom"),
    )
    o.declare_root(body)
    r = support_connected(o.finalize())
    assert r.verdict == "fail"
    assert "flag" in r.evidence["unsupported"]


# ---------------------------------------------------------------------------
# Lazy contact pass: the exact measurement stops once connectivity is settled
# ---------------------------------------------------------------------------


def _counting_adjudicator(monkeypatch):
    """Wrap `exact_object_part_gaps` and record every pair it is asked about."""
    from dycraft.verify import _exact_adjudicator as adj
    seen: list = []
    real = adj.exact_object_part_gaps

    def spy(ir, part_pairs, **kw):
        seen.extend(tuple(p) for p in part_pairs)
        return real(ir, part_pairs, **kw)

    monkeypatch.setattr(adj, "exact_object_part_gaps", spy)
    return seen


def _jointed_stack(n: int = 6):
    """A tower of flush boxes, each fixed to the one below. Every part is
    connected through the joint tree, and every neighboring pair is also an
    AABB contact candidate — so the exact pass has plenty it *could* measure
    and nothing it *needs* to."""
    o = Object("stack")
    prev = o.make_box("b0", 0.2, 0.2, 0.1)
    o.declare_root(prev)
    for i in range(1, n):
        nxt = o.make_box(f"b{i}", 0.2, 0.2, 0.1)
        o.attach_fixed(f"j{i}", prev, nxt,
                       parent_anchor=o.anchor(prev, "top"),
                       child_anchor=o.anchor(nxt, "bottom"))
        prev = nxt
    return o.finalize()


def test_a_jointed_object_measures_no_pairs_at_all(monkeypatch):
    """The joint tree already grounds every part, so no exact query can change
    the verdict and none should be paid for.

    This is the common case, not the corner one: an articulated object has a
    joint tree by construction, and exact spline-pair queries can be most of
    its compile time.
    """
    seen = _counting_adjudicator(monkeypatch)
    ir = _jointed_stack()
    r = support_connected(ir)
    assert r.verdict == "pass", r.evidence
    assert seen == [], f"measured {len(seen)} pair(s) that could not matter"
    ec = r.evidence["exact_contact"]
    # The block still reports, so "nothing to decide" cannot read as
    # "the exact pass was unavailable".
    assert ec["considered"] > 0 and ec["sent"] == 0
    assert ec["settled_without_measuring"] is True
    assert ec["complete"] is False


def test_the_probe_still_measures_every_pair(monkeypatch):
    """`object_support_report` answers "which parts touch", so a spanning
    subset is the wrong answer even though it would give the same verdict."""
    from dycraft.sdk._probes import object_support_report

    seen = _counting_adjudicator(monkeypatch)
    ir = _jointed_stack()
    report = object_support_report(ir)
    assert report["is_supported"] is True
    assert seen, "the probe must not inherit the verifier's early exit"
    assert report["exact_contact"]["complete"] is True


def test_a_detached_part_is_still_found_and_still_reported(monkeypatch):
    """The early exit must not turn a floating part into a pass: the loop
    only stops when every part is in the floor's component."""
    seen = _counting_adjudicator(monkeypatch)
    o = Object("with_floater")
    body = o.make_box("body", 0.5, 0.5, 0.5)
    flag = o.make_box("flag", 0.3, 0.02, 0.2)
    o.attach_fixed(
        "j_flag", body, flag,
        parent_anchor=o.anchor(body, "top", translation=(0.0, 0.0, 1.0)),
        child_anchor=o.anchor(flag, "bottom"))
    o.declare_root(body)
    r = support_connected(o.finalize())
    assert r.verdict == "fail"
    assert "flag" in r.evidence["unsupported"]


def test_skipping_a_measured_pair_cannot_change_the_verdict():
    """The lazy and complete passes agree, part for part.

    The guarantee is a property of connected components rather than of the
    ordering: an edge whose endpoints already share a component cannot move
    any node into or out of the floor's component.
    """
    from dycraft.sdk import object_query as oq

    irs = []
    for n in (4, 8):
        _reset_object_singleton()      # one Object per sandbox execution
        irs.append(_jointed_stack(n))
    for ir in irs:
        got = []
        for complete in (True, False):
            adj = _sup.build_object_adjacency(
                ir, oq, _sup.contact_tol_m(), _sup.joint_tol_m(),
                complete_contacts=complete)
            reached = _sup._bfs_from_floor(adj)
            got.append(sorted(n for n in reached if n != "__floor__"))
        assert got[0] == got[1]


def test_the_stats_merge_does_not_sum_flags_across_waves():
    """`available` is a bool, and a bool is an int in Python: a plain numeric
    merge turned two waves of `True` into `2`. `considered` is the broad
    phase's own count and must not be re-derived from the waves either."""
    into = {"considered": 40, "sent": 0, "cache_hits": 0,
            "available": None, "backend": None}
    for _ in range(3):
        _sup._merge_exact_stats(
            into, {"considered": 12, "sent": 5, "cache_hits": 7,
                   "available": True, "backend": "brep"})
    assert into["available"] is True
    assert into["backend"] == "brep"
    assert into["considered"] == 12      # overwritten, never accumulated
    assert (into["sent"], into["cache_hits"]) == (15, 21)


def test_a_dsu_edge_inside_one_component_is_reported_redundant():
    dsu = _sup._DSU(["a", "b", "c"])
    assert dsu.union("a", "b") is True
    assert dsu.union("b", "a") is False      # already one component
    assert dsu.connected("a", "b") and not dsu.connected("a", "c")
