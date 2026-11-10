"""The motion verifier's whole output, pinned on synthetic rows.

`verify.aggregate` turns one pass's per-frame rows into the reports an agent reads: names, verdicts, evidence, and
repair text. Any change to that output, intended or not, must show up here. Two bundles cover every verifier and
every event predicate, with failures, allowances, room escapes, and declaration errors among them; their reports
are compared with `fixtures/verify_aggregate_golden.json`, byte for byte as JSON.

An intended change rewrites the fixture with `DYCRAFT_WRITE_GOLDEN=1`, and the diff of the fixture is then part of
what is reviewed. None of this needs Blender.
"""
import json
import os
from pathlib import Path

from dycraft.animation.blender.verify import ROWS_FORMAT, aggregate
from dycraft.animation.blender.verify_penetration import _overlap_spans

GOLDEN = Path(__file__).parent / "fixtures" / "verify_aggregate_golden.json"
FPS = 24.0


def _diagnosed(bundle):
    """Overlap diagnostics at each span's representative and ranked frames, as `measure` records them."""
    rows = bundle["rows"]["NoPenetration"]
    rows["shifts"], rows["depths"] = [], []
    for span in _overlap_spans(rows["hits"]):
        a, b, chosen = span["a"], span["b"], span["representative_frame"]
        rows["shifts"].append({"a": a, "b": b, "frame": chosen, "directional_separation_m": .001 * (chosen % 7 + 1),
                               "resolution_m": 1e-6, "separation_direction": [0., 0., 1.], "depth_method": "test"})
        for frame in [chosen] + [f for _, f in span["ranked"] if f != chosen]:
            rows["depths"].append({"a": a, "b": b, "frame": frame,
                                   "depth": None if frame % 5 == 0 else .0005 * (frame % 11)})
    return bundle


def _event(name, predicate="touch", mode="all", window=(0, 59), **params):
    return {"event": {"name": name, "predicate": predicate, "mode": mode, "window": list(window),
                      "a": "robot/hand", "b": "box/body", **params}, "window_frames": list(window)}


def _bundle(frames, events, measured, *, room=None, errors=(), **rows):
    frames = list(frames)
    event_rows = [{"event": k, "frame": f, **measured[k][f]}
                  for f in frames for k in sorted(measured) if f in measured[k]]
    return _diagnosed({
        "format": ROWS_FORMAT, "frames": frames,
        "context": {"fps": FPS, "identity_changed": False, "events": events,
                    "declaration_errors": list(errors),
                    "allowances": {"declared": 2, "inherited": 1}, "room": room,
                    "joints": True, "anatomical": True, "weighed": ["dog", "robot"]},
        "stats": {"bvh_builds": 3, "collision_pair_cache_hits": 7},
        "rows": {"AnimationWellFormed": rows.get("failures", []), "JointLimits": rows.get("joints", []),
                 "AnatomicalJointLimits": rows.get("anatomical", []),
                 "NoPenetration": {"hits": rows.get("hits", []), "escapes": rows.get("escapes", []),
                                   "allowed": rows.get("allowed", [])},
                 "EventsSatisfied": event_rows, "moved": rows.get("moved", []),
                 "MotionContinuity": {"warnings": rows.get("spikes", []), "rigid_duplicates": [],
                                      "transitions_checked": [{"frame": f, "count": 1} for f in frames],
                                      "topology_changes": [], "boundaries": []},
                 "SupportMargin": rows.get("support", []), "ContactSlip": rows.get("slip", [])}})


def _everything():
    """Every verifier failing or advising somewhere, every predicate, and the declaration errors."""
    frames = range(60)
    hits = [{"frame": f, "a": "robot/hand", "b": "box/body", "kind": "surface_crossing",
             "bound": .001 * ((f * 7) % 13 + 1), "direction": [1., 0., 0.]} for f in range(20, 33)]
    hits += [{"frame": f, "a": "robot/foot", "b": "room_prop/leg", "kind": "surface_crossing", "bound": .02,
              "direction": [0., 0., 1.]} for f in (14, 15, 44)]
    allowed = [{"frame": 10, "a": "cup/body", "b": "robot/hand", "depth": .015, "limit": .010},
               {"frame": 40, "a": "cup/body", "b": "robot/hand", "depth": .060, "limit": .050},
               {"frame": 41, "a": "cup/body", "b": "tray/top", "depth": .002, "limit": None}]
    room = {"lo": [-2., -2., 0.], "hi": [2., 2., 3.],
            "faces": [[0, -1, "wall_xmin"], [0, 1, "wall_xmax"], [2, -1, "floor"]]}
    escapes = [{"frame": f, "object": "cart", "aabb": [[1.5, 0, 0], [2.0 + .01 * f, .5, .5]]} for f in (28, 29, 58)]
    slip = [{"frame": f, "part": "robot/foot", "on": "room.floor", "patch": [.001 * f, 0.]} for f in frames]
    slip += [{"frame": f, "part": "robot/hand", "on": None, "clearance": .4} for f in frames]
    support = [{"frame": f, "instance": "robot", "margin": .05 if f > 10 else -.02, "bearing": 1} for f in frames]
    slip += [{"frame": f, "part": "dog/paw", "on": "room.floor", "patch": [0., .002 * f]} for f in range(30, 60)]
    support += [{"frame": f, "instance": "dog", "margin": .03, "bearing": 2} for f in range(30, 60)]
    support += [{"frame": f, "instance": "dog", "why": {"parts_in_frame": 0, "weighed": True, "parts_bearing": 0,
                                                       "support_points": 0}} for f in range(30)]
    spikes = [{"frame": f, "object": "arm", "channel": "rotation", "reason": "isolated motion spike",
               "frames": [f - 1, f], "step_magnitude": .1 * f, "rate": 2.4 * f, "units": "rad/s"}
              for f in (12, 29, 30, 50)]
    events = [_event("grip", mode="any", window=(0, 59)),
              _event("apart", predicate="separated", distance=.005, window=(30, 50)),
              _event("over", predicate="above", window=(5, 15)),
              _event("bent", predicate="joint_at", joint="arm__elbow", value=1.0, window=(20, 25)),
              _event("stand", predicate="planted", a="robot/foot", b=None, tolerance=.005),
              _event("balanced", predicate="supported", a="robot", b=None, margin=0.0),
              _event("reach", mode="any", window=(20, 40)),
              _event("later", window=(50, 90))]
    measured = {0: {f: {"holds": f >= 45, "gap": .0005 if f >= 45 else .01} for f in frames},
                1: {f: {"holds": f != 40, "gap": .02 if f != 40 else .001} for f in range(30, 51)},
                2: {f: {"holds": True} for f in range(5, 16)},
                3: {f: {"holds": f < 24} for f in range(20, 26)},
                4: {**{f: {"touching": False, "lift": .02} for f in range(10)},
                    **{f: {"touching": True, "patch": [.0002 * (f - 10), 0.]} for f in range(10, 60)}},
                5: {f: {"margin": None if f < 5 else (-.01 if f < 50 else .02)} for f in frames},
                6: {f: ({"missing": ["box/body"]} if f < 35 else {"holds": f == 38}) for f in range(20, 41)},
                7: {f: {"holds": True} for f in range(50, 60)}}
    errors = [{"kind": "event", "index": 8, "name": "sits", "a": "chiar", "window": [0, 10],
               "reason": "a 'chiar' names no instance of this scene"},
              {"kind": "collision_allowance", "index": 3, "a": "x/p", "b": "typo/q",
               "reason": "collision allowance 'x/p' / 'typo/q': unknown 'typo/q'; operands are two distinct part "
                         "identities ('instance/part')"},
              {"kind": "liquid", "index": 0, "object": "water", "reason": "liquid 'water' names no part"}]
    return _bundle(frames, events, measured, room=room, errors=errors, hits=hits, allowed=allowed, escapes=escapes,
                   slip=slip, support=support, spikes=spikes,
                   failures=[{"frame": 4, "object": "x", "reason": "empty"}],
                   joints=[{"frame": f, "joint": "arm__elbow", "q": 2.1 + .1 * (f - 28), "limits": [0.0, 2.0],
                            "reason": "joint limit or motion axis changed"} for f in range(28, 33)],
                   anatomical=[{"frame": f, "rig": "rig/t/body", "joint": "elbow_R", "measure": "flexion",
                                "reason": "bent backwards", "value_deg": -120.0 if f == 35 else -20.0,
                                "limits_deg": [-10.0, 160.0]} for f in range(25, 41)],
                   moved=[{"frame": 3, "moved": ["robot/"]}, {"frame": 45, "moved": ["dog/", "robot/"]}])


def _clean():
    """A clean short pass: verdicts `pass`/`not_applicable`, and every event deferred."""
    return _bundle(range(10), [_event("later", window=(20, 30))], {})


def _reports():
    everything = _everything()
    return {"everything": aggregate(everything),
            "everything_partial_events": aggregate(everything, defer_partial_events=False),
            "clean": aggregate(_clean())}


def test_the_verifier_output_matches_the_golden_reports():
    text = json.dumps(_reports(), sort_keys=True, indent=1) + "\n"
    if os.environ.get("DYCRAFT_WRITE_GOLDEN"):
        GOLDEN.write_text(text)
    assert text == GOLDEN.read_text(), "verify.aggregate's output changed; see this module's docstring"


def test_the_golden_bundles_reach_every_verifier_and_verdict():
    """Guards the fixture itself: a bundle that stops exercising a report would leave it unpinned."""
    reports = {r["name"]: r for r in _reports()["everything"]}
    assert {"AnimationWellFormed", "JointLimits", "AnatomicalJointLimits", "NoPenetration", "EventsSatisfied",
            "MotionContinuity", "SupportMargin", "ContactSlip"} <= set(reports)
    assert {r["verdict"] for r in reports.values()} >= {"fail", "uncertain"}
    events = {row["name"]: row for row in reports["EventsSatisfied"]["evidence"]["events"]}
    assert events["sits"]["status"] == "invalid" and "allowances_exceeded" in reports["NoPenetration"]["evidence"]
    assert reports["NoPenetration"]["evidence"]["out_of_room"]
