"""Support and contact: what bears each body, its center-of-mass margin, and how far a planted contact slides."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


#: How close a support's lowest vertices sit to its own minimum z before they
#: count as the patch that is bearing. Not a tolerance the author picks: it
#: selects which vertices are "the sole", not how far the sole may slide.
_PATCH_BAND_M = 0.005


def slip_evidence(tracks, fps):
    """Advisory creep evidence for every part that bore on the floor.

    The automatic form of a `planted` event, for parts nobody declared one
    about. Reports the pair that separates a slide from a step: a part with one
    long bearing interval and no clearance never left the floor, whatever its
    patch did; a part with many intervals and real clearance between them is
    taking steps, and its per-interval creep is the number to read.
    """
    rows = []
    for identity, track in sorted(tracks.items()):
        if not track["slips"]:
            continue
        worst = int(np.argmax([value for _, value in track["slips"]]))
        frame, value = track["slips"][worst]
        rows.append({
            "part": identity,
            "bearing_intervals": track["intervals"] + (1 if track["anchor"] is not None else 0),
            "frames_bearing": track["frames"],
            "max_slip_m": round(float(value), 6),
            "max_slip_frame": frame,
            "max_slip_seconds": round(frame / fps, 4),
            "max_clearance_m": round(float(track["clearance"]), 6),
            "frame_gaps": track["gaps"],
            "bearing_on": sorted(track.get("grounds", ()))})
    # 5 mm is the same order as the `touch` default and below what reads at
    # delivery resolution, so it names the parts worth looking at rather than
    # asserting a defect. Nothing gates on this.
    flagged = [{"part": row["part"], "max_slip_m": row["max_slip_m"]}
               for row in rows if row["max_slip_m"] > .005]
    return flagged, {
        "list_failures": False,
        "parts": rows,
        "measure": ("horizontal creep of a bearing part's lowest-vertex "
                    "centroid from where it last touched down, per bearing "
                    "interval. Read `bearing_intervals` and "
                    "`max_clearance_m` with it: one interval at no clearance "
                    "is a part that never lifted."),
        "repairs": (["A bearing part that creeps is a foot sliding or a "
                     "support drifting. Check `bearing_intervals` first: one "
                     "interval spanning the motion means the part never left "
                     "the floor, which is a different repair from a step that "
                     "lands badly."] if flagged else []),
    }


def slip_tracks(rows, moving):
    """`slip_evidence`'s per-part tracks, built from `measure`'s per-frame rows.

    Anchors, bearing intervals, clearance, and gaps are state carried from frame
    to frame, so they are built here, once, over the time-ordered rows of every
    process that measured them. Built inside a chunk instead, a part already
    standing when the chunk began was anchored where the chunk began rather
    than where the part landed, and its slip was a lower bound that the
    report had to flag; over the union there is no boundary to anchor at.

    `moving` is the instance prefixes seen to move anywhere in the check.
    """
    tracks = {}
    for row in rows:
        identity, frame = row["part"], row["frame"]
        if identity.split("/")[0] + "/" not in moving:
            continue
        track = tracks.setdefault(identity, {"anchor": None, "intervals": 0, "frames": 0,
                                             "clearance": 0.0, "slips": [], "previous": None,
                                             "gaps": 0, "grounds": set()})
        # Creep is only creep between adjacent frames of the film. A strided
        # check -- `frames=range(0, n, 4)`, or a storyboard's own frames --
        # samples moments seconds apart, and carrying an anchor across one
        # reports how far the body walked as how far its foot slid. Drop the
        # anchor and say how often.
        if track["previous"] is not None and frame - track["previous"] > 1:
            if track["anchor"] is not None:
                track["intervals"] += 1
            track["anchor"] = None
            track["gaps"] += 1
        track["previous"] = frame
        if row["on"] is not None:
            track["grounds"].add(row["on"])
            patch = np.asarray(row["patch"], dtype=float)
            if track["anchor"] is None:
                track["anchor"] = patch
            track["frames"] += 1
            track["slips"].append((frame, float(np.linalg.norm(patch - track["anchor"]))))
        else:
            if track["anchor"] is not None:
                track["intervals"] += 1
            track["anchor"] = None
            # Height above whatever is under it, or above nothing.
            if row.get("clearance") is not None:
                track["clearance"] = max(track["clearance"], row["clearance"])
    return tracks


def support_samples(rows, moving):
    """`support_evidence`'s `(samples, skipped)`, from `measure`'s per-frame rows.

    Medians and per-phase fractions cannot be recombined from per-chunk
    summaries, so the samples are kept per frame and summarized once.
    `skipped` keeps, per body never measured, the most informative frame's
    reason: one where the body was at least present beats one where it was
    hidden.
    """
    samples, skipped = {}, {}
    for row in rows:
        instance = row["instance"]
        if instance + "/" not in moving:
            continue
        if "margin" in row:
            samples.setdefault(instance, []).append((row["frame"], row["margin"], row["bearing"]))
        else:
            best = skipped.get(instance)
            if best is None or row["why"]["parts_in_frame"] > best["parts_in_frame"]:
                skipped[instance] = row["why"]
    return samples, {name: why for name, why in skipped.items() if name not in samples}


def support_evidence(samples, fps, phase_minimum=20, considered=(), skipped=None):
    """Advisory support-margin evidence for every body that bore on the floor.

    Reported **by support phase**, and not pooled. The two phases of a walk
    have nothing to do with each other: a body is rarely outside its support on
    two feet and usually outside it on one, so a pooled figure measures the
    gait's duty factor rather than its balance, and a threshold on it would
    judge the wrong thing.

    `phase_minimum` frames are required before a phase is worth flagging, so a
    body passing through one-foot support for three frames of a turn is not
    read as standing on one foot.

    Quasi-static throughout. A body in motion may legitimately carry its center
    of mass outside its support for part of a stride; a body that does it
    through nearly all of its fewest-support frames is not walking.
    """
    def summarise(values):
        outside = values > 0
        worst = int(np.argmax(values))
        return {"frames": len(values),
                "outside": int(outside.sum()),
                "outside_fraction": round(float(outside.mean()), 4),
                "worst_margin_m": round(float(values[worst]), 6),
                "median_margin_m": round(float(np.median(values)), 6)}, worst

    rows, flagged = [], []
    # Twice now this verifier has reported a clean pass because it measured
    # nothing -- once from a single-point support polygon, once from an
    # evaluation whose subjects were all `hide_render`. Silence and a pass are
    # different answers, so the candidates it considered and why each produced
    # no sample are part of the evidence.
    for instance, series in sorted(samples.items()):
        values = np.array([margin for _, margin, _ in series], dtype=float)
        counts = np.array([bearing for _, _, bearing in series], dtype=int)
        overall, worst = summarise(values)
        phases = {}
        for count in sorted(set(counts.tolist())):
            phase, _ = summarise(values[counts == count])
            phases[str(count)] = phase
        rows.append({"instance": instance,
                     "frames_bearing": overall["frames"],
                     "worst_margin_frame": series[worst][0],
                     "worst_margin_seconds": round(series[worst][0] / fps, 4),
                     "pooled": overall,
                     "by_bearing_parts": phases})
        # The fewest-support phase with enough frames to mean anything is where
        # balance is decided, so that is what names a body worth reading about.
        for count, phase in sorted(phases.items(), key=lambda kv: int(kv[0])):
            if phase["frames"] >= phase_minimum and phase["outside_fraction"] > 0.5:
                flagged.append({"instance": instance, "bearing_parts": int(count),
                                "frames": phase["frames"],
                                "outside_fraction": phase["outside_fraction"]})
                break
    return flagged, {
        "list_failures": False,
        "bodies": rows,
        "considered": sorted(considered),
        "measured": sorted(row["instance"] for row in rows),
        "unmeasured": {name: reason for name, reason in sorted((skipped or {}).items())},
        "note": ("No `bodies` is not a pass: it means nothing was measurable. "
                 "`unmeasured` says why for each candidate."
                 if not rows else None),
        "measure": ("signed distance from the mass-weighted centre of mass to "
                    "the convex hull of the footprints bearing on the floor; "
                    "negative is inside. Split by how many parts were bearing, "
                    "because pooling a one-foot phase with a two-foot one "
                    "reports the duty factor rather than the balance."),
        "repairs": (["Read `by_bearing_parts` rather than `pooled`: the "
                     "fewest-support phase is where balance is decided. A body "
                     "outside its support through nearly all of that phase is "
                     "not balanced, whatever the gait."] if flagged else []),
    }


def part_mass_table(scene_ir, catalog):
    """`{instance/part: (mass, com_in_part_frame)}` for every part we can weigh.

    Two sources, in order. An imported asset that declared `<inertial>` carries
    a real mass and a real per-part center of mass in `params.inertial`; use
    them, because the
    alternative is uniform density over a mesh that was drawn to be *seen*. The
    G1 is the case: it declares 34.13 kg against Unitree's ~35, and comes out
    at 20.90 kg from its geometry, with its foot 299x light because that link's
    collision proxy is 2 cm^3.

    Otherwise uniform density over `shape.volume` about `shape.center`, which
    every part has and which is what an authored CAD object will use. Mixing
    the two within one body would weigh its parts on different scales, so a
    body is weighed entirely one way or entirely the other.

    Units do not need to agree between bodies: every consumer normalizes by
    the total, so a volume-weighted body and a mass-weighted body each give
    their own correct centroid.
    """
    table = {}
    for instance in (scene_ir or {}).get("instances", ()):
        ref = instance.get("object_ref")
        parts = ((catalog or {}).get(ref) or {}).get("parts") or ()
        weights, declared = {}, True
        for part in parts:
            params = part.get("params") or {}
            inertial = params.get("inertial") or {}
            shape = params.get("shape") or {}
            mass = inertial.get("mass")
            if mass is None or not mass > 0:
                declared = False
                mass, centre = shape.get("volume"), shape.get("center")
            else:
                # `com` is in the part's own frame, the frame `shape.center`
                # uses, so the two are interchangeable here without a
                # transform. See `sdk/_primitives._validate_imported_params`.
                centre = inertial.get("com") or shape.get("center")
            if not mass or not mass > 0 or centre is None:
                continue
            weights[part["name"]] = (float(mass), np.asarray(centre, dtype=float),
                                     np.asarray(params["aabb_min"], dtype=float),
                                     np.asarray(params["aabb_max"], dtype=float))
        if not weights:
            continue
        if not declared:
            # One scale for the whole body. Falling back per part would weigh
            # a declared 0.6 kg foot against a sibling's 2e-06 m^3.
            weights = {}
            for part in parts:
                shape = (part.get("params") or {}).get("shape") or {}
                if shape.get("volume", 0) > 0 and shape.get("center") is not None:
                    prm = part.get("params") or {}
                    weights[part["name"]] = (float(shape["volume"]),
                                             np.asarray(shape["center"], dtype=float),
                                             np.asarray(prm["aabb_min"], dtype=float),
                                             np.asarray(prm["aabb_max"], dtype=float))
        for name, entry in weights.items():
            table[f"{instance['name']}/{name}"] = entry
    return table


def support_margin(com_xy, patch_points):
    """Signed distance from `com_xy` to the support hull: negative is inside.

    Quasi-static only. A body in motion may legitimately carry its center of
    mass outside its support for part of a stride, so a positive margin is a
    finding to read against the phase, not a defect on its own.
    """
    pts = np.asarray(patch_points, dtype=float)
    vertices = _hull_vertices(pts)
    if vertices is None:
        return None
    hull = pts[vertices]
    best, side = float("inf"), []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b - a
        length = float(edge @ edge)
        t = 0.0 if length <= 0 else float(np.clip((com_xy - a) @ edge / length, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * edge - com_xy)))
        # 2-D `np.cross` is deprecated in NumPy 2; the scalar z of the cross
        # product is the whole quantity here.
        offset = com_xy - a
        side.append(float(edge[0] * offset[1] - edge[1] * offset[0]))
    # Consistent sign, not a sign: the hull's winding is whatever the hull
    # library returned, and a point inside is on the same side of every edge.
    inside = all(v >= -1e-12 for v in side) or all(v <= 1e-12 for v in side)
    return -best if inside else best


def _supported_margin(instance, masses, bodies, records, body, ground_body,
                      contact_m, gap, bearing=None):
    """`(margin, parts_bearing)`, or `None` when the question does not apply.

    `ground_body` is the explicit body a declared `supported` event names, or
    `None`, in which case each part's ground is read from `bearing`, the
    frame's bearing map (`collision.BearingResolver`).
    """
    prefix = instance + "/"
    weighted, total, patch, bearing_parts = np.zeros(3), 0.0, [], 0
    for part_i, entry in enumerate(bodies):
        identity = entry[0]
        if not identity.startswith(prefix):
            continue
        weight = masses.get(identity)
        if weight is not None:
            mass, centre, local_lo, local_hi = weight
            # Place the part's own center of mass by where its box actually is
            # this frame, as a fraction of its box in the space the centroid
            # was measured in. Applying `records[index]["matrix"]` to it looks
            # right and is not: an evaluation whose geometry is already
            # world-space carries an identity matrix, so every centroid stayed
            # at the authoring origin and a mover three meters up a staircase
            # had its mass placed at the foot of it. Using the box center
            # instead would be wrong a different way -- for a boot or an
            # L-bracket the box center is nobody's center of mass -- so the
            # offset within the box is what carries across.
            span = np.where(local_hi - local_lo > 1e-9, local_hi - local_lo, 1.0)
            fraction = (centre - local_lo) / span
            weighted += mass * (entry[3] + fraction * (entry[4] - entry[3]))
            total += mass
        # Measured per frame from the bearing map (`_supported_margin_held`),
        # or against the one ground a declared `supported` event names.
        if ground_body is None:
            bears = bearing.get(part_i) is not None
        else:
            bears = bool(gap(body(part_i), body(ground_body), contact_m))
        if bears:
            patch.extend(_footprint(contact_patch_outline, entry, records[part_i]))
            bearing_parts += 1
    # A verifier that measures nothing must be able to say which step stopped
    # it; a worker `print` never reaches the caller, so the reason travels in
    # the return value.
    seen = sum(1 for entry in bodies if entry[0].startswith(prefix))
    if total <= 0 or len(patch) < 3:
        return None, {"parts_in_frame": seen, "weighed": total > 0,
                      "parts_bearing": bearing_parts, "support_points": len(patch)}
    margin = support_margin((weighted / total)[:2], patch)
    if margin is None:
        return None, {"parts_in_frame": seen, "weighed": True,
                      "parts_bearing": bearing_parts, "support_points": len(patch),
                      "hull": "degenerate"}
    return (margin, bearing_parts), None


def _support_inputs(prefix, bodies, records, bearing):
    """Everything `_supported_margin` reads on the bearing-map path, or `None`.

    Per part, in order: its box, whether it bears, and its footprint's key.
    `None` when a bearing part's footprint has no key (`_footprint` cannot
    hold it either).
    """
    inputs = []
    for part_i, entry in enumerate(bodies):
        if not entry[0].startswith(prefix):
            continue
        foot = None
        bears = bearing.get(part_i) is not None
        if bears:
            mesh = (records[part_i] or {}).get("mesh")
            if mesh is None or len(entry) < 7:
                return None
            foot = (mesh, np.asarray(entry[5], dtype=float).tobytes(), bool(entry[6]))
        inputs.append((entry[0], np.asarray(entry[3], dtype=float).tobytes(),
                       np.asarray(entry[4], dtype=float).tobytes(), bears, foot))
    return tuple(inputs)


def _supported_margin_held(held, instance, masses, bodies, records, body, contact_m,
                           gap, bearing):
    """`_supported_margin` on the bearing-map path, reusing the last frame's.

    `held` maps an instance to the inputs and result of the last frame it was
    measured on, so it holds one entry per instance. An instance standing
    still has the same inputs, so the same result.
    """
    key = _support_inputs(instance + "/", bodies, records, bearing)
    last = held.get(instance)
    if key is not None and last is not None and last[0] == key:
        return last[1]
    result = _supported_margin(instance, masses, bodies, records, body, None,
                               contact_m, gap, bearing)
    if key is not None:
        held[instance] = (key, result)
    return result


def _hull_vertices(points):
    """Indices of the 2-D convex hull, counterclockwise, or `None` if degenerate.

    Andrew's monotone chain, and deliberately not `scipy.spatial.ConvexHull`:
    **this module runs inside Blender's interpreter, which has numpy and no
    scipy.** A scipy hull here silently returned nothing on every scene, so
    every support polygon degenerated and `SupportMargin` measured nothing
    while reporting a pass — on the host, where scipy exists, its own tests
    passed. Anything this file needs at verify time has to be numpy-only.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    index = np.arange(len(pts))
    if len(pts) > 32:
        # Akl-Toussaint: a point strictly inside the quadrilateral of the four
        # extreme points is no hull vertex, so the chain below never sees it.
        # Exact, and most of a dense footprint's points.
        quad = pts[[np.argmin(pts[:, 0]), np.argmin(pts[:, 1]),
                    np.argmax(pts[:, 0]), np.argmax(pts[:, 1])]]
        inside = np.ones(len(pts), dtype=bool)
        for a, b in zip(quad, np.roll(quad, -1, axis=0)):
            inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) > 1e-12
        if inside.any():
            index = np.flatnonzero(~inside)
    order = index[np.lexsort((pts[index, 1], pts[index, 0]))]
    unique = order[np.concatenate(
        ([True], (np.diff(pts[order], axis=0) != 0).any(axis=1)))]
    if len(unique) < 3:
        return None
    # The chain runs on Python floats: the same IEEE products and differences
    # as on numpy scalars, so the same hull, without a numpy scalar per
    # operation.
    xy = pts[unique].tolist()

    def half(indices):
        chain = []
        for i in indices:
            cx, cy = xy[i]
            while len(chain) >= 2:
                (ax, ay), (bx, by) = xy[chain[-2]], xy[chain[-1]]
                if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0:
                    break
                chain.pop()
            chain.append(i)
        return chain

    lower, upper = half(range(len(xy))), half(range(len(xy) - 1, -1, -1))
    hull = lower[:-1] + upper[:-1]
    return unique[np.asarray(hull, dtype=int)] if len(hull) >= 3 else None


def ground_index(bodies):
    """Per-frame arrays `ground_candidates_many` filters on, built once.

    The test is a handful of interval comparisons per body pair, which is
    nothing each time and everything when it runs for every bearing part
    against every body on every frame. Vectorized it is one pass over three
    arrays; in Python it was the most expensive thing this module does.
    """
    return (np.array([entry[3] for entry in bodies], dtype=float),
            np.array([entry[4] for entry in bodies], dtype=float),
            np.array([entry[0].split("/")[0] + "/" for entry in bodies]))


def ground_candidates_many(bodies, parts, contact_m, index=None):
    """For each of `parts`, the indices of bodies that could be bearing it from below.

    One matrix over every part against every body, rather than one pass over
    the bodies per part per frame.

    `room.floor` is not a safe assumption. One measured film had the movers on
    `house/treads` and `house/floorboards` and no `room.*` surface at all, so a
    check written against the floor name reported nothing and passed. What
    makes a surface ground is where it is, not what it is called: something of
    it at or below the sole, and overlapping it in plan.

    Nor does it reach down: a candidate's top must come within `contact_m`
    of the part's sole, because nothing whose box lies wholly further below
    can be within `contact_m` of it. Without that bound every part of a
    moving body -- a head, a hand -- was distance-tested against the floor on
    every frame, about 3,000 queries a frame on the dog film, for answers
    that could only be "no".

    Deliberately not narrowed further. Requiring the candidate to have
    vertices inside the part's own footprint looks tighter and is wrong: a
    floor is a quad whose corners are meters from a foot, so that test
    rejected the floor and reported a walking robot bearing on nothing. A
    piano's box does reach from its casters to its lid, so a hand gripping its
    side qualifies here -- which is why the report names what each part was
    found bearing on, and `hand_R` on `piano/body` reads as a grip without a
    filter guessing between them.
    """
    if not parts:
        return []
    los, his, prefixes = index if index is not None else ground_index(bodies)
    parts = np.asarray(parts)
    lo_p, hi_p = los[parts][:, None, :], his[parts][:, None, :]
    keep = ((prefixes[None, :] != prefixes[parts][:, None])
            & (los[None, :, 2] <= lo_p[..., 2] + contact_m)
            & (his[None, :, 2] >= lo_p[..., 2] - contact_m)
            & (his[None, :, 0] >= lo_p[..., 0]) & (los[None, :, 0] <= hi_p[..., 0])
            & (his[None, :, 1] >= lo_p[..., 1]) & (los[None, :, 1] <= hi_p[..., 1]))
    keep[np.arange(len(parts)), parts] = False
    return [np.flatnonzero(row).tolist() for row in keep]


def world_verts(entry):
    """World-space vertices for a `bodies` entry, whichever space it holds."""
    verts = entry[1]
    if len(entry) < 7 or not entry[6]:
        return verts
    matrix = np.asarray(entry[5], dtype=float)
    return verts @ matrix[:3, :3].T + matrix[:3, 3]


#: Footprints already reduced, by mesh and pose: a planted foot's is the same
#: on every frame it stays put, and the support polygons are costly to rebuild.
_FOOTPRINTS = OrderedDict()


def _footprint(function, entry, record):
    """`function(world_verts(entry))`, kept for a mesh and pose seen before.

    Exact: the world vertices are a function of the local mesh, which the
    record names by digest, and the matrix.
    """
    mesh = (record or {}).get("mesh")
    if mesh is None or len(entry) < 7:
        return function(world_verts(entry))
    key = (function.__name__, mesh, np.asarray(entry[5], dtype=float).tobytes(), bool(entry[6]))
    held = _FOOTPRINTS.get(key)
    if held is None:
        held = _FOOTPRINTS[key] = function(world_verts(entry))
        while len(_FOOTPRINTS) > 20000:
            _FOOTPRINTS.popitem(last=False)
    else:
        _FOOTPRINTS.move_to_end(key)
    return held


def contact_patch_outline(verts):
    """XY hull of the lowest band — the footprint a part actually bears on.

    `contact_patch_xy` reduces the same band to one point, which is what
    tracking creep wants and what a support polygon must not use: one foot
    would be one point and two feet a line, so a biped would never have a
    polygon to be inside of and would go unmeasured. A sole contributes its
    outline instead, so single support is a real footprint.
    """
    band = verts[verts[:, 2] <= verts[:, 2].min() + _PATCH_BAND_M][:, :2]
    vertices = _hull_vertices(band)
    if vertices is None:
        # Collinear or degenerate: the raw points still bound whatever they
        # bound, and `support_margin` rejects what it cannot hull.
        return [tuple(point) for point in band]
    return [tuple(point) for point in band[vertices]]


def contact_patch_xy(verts):
    """Horizontal centroid of the lowest band of `verts` — the patch bearing.

    A rigid part's own vertices, so this rotates with it: a foot that rolls
    forward reports a patch that moves with the roll, while a foot that stays
    put reports the same centroid whatever the ankle above it does. Taking the
    mean of a band rather than the single lowest vertex keeps the number stable
    when a nearly flat sole changes which vertex is minimal by micrometers.
    """
    z = verts[:, 2]
    band = verts[z <= z.min() + _PATCH_BAND_M]
    return band[:, :2].mean(axis=0)
