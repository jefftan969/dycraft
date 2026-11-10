"""Dog motion capture onto a quadruped rig: the MANN library -> a rigify dog.

Agents import it inside Blender as `import dog_retarget as dr` (`/tools`). The
library is the dog captures of Zhang, Starke, Komura, and Saito, "Mode-Adaptive
Neural Networks for Quadruped Motion Control" (SIGGRAPH 2018; CC BY-NC 4.0):
51 BVH files, 41 minutes at 60 fps, mounted at `/references/mocap/mann_dog/`
with `index.txt` labeling every span (walk, trot, gallop, sit, lie, sniff,
turn, jump).

Two steps, so every edit still gets the solve's guarantees:

    c = dr.clip("D1_008_KAN01_002", 0.5, 6.0)          # a span of one capture
    c = c.then(dr.clip("D1_009_KAN01_001", 3, 7))      # splice, blended
    c = c.follow([(0, 0), (0, 3), (2, 5)])              # walk it along a path
    rep = dr.retarget(bpy.data.objects["rig/dog/body"], c, name="dog_fetch")

`Clip` edits act on the captured joint positions; `retarget` then solves the rig
once: each bone takes the rotation its captured segment makes away from the
dog's neutral stand (so the two skeletons' different rest poses do not matter),
the whole dog is placed as one body over the ground its planted paws need, each
paw the capture holds on the ground is locked for the whole contact, and the legs
are re-solved at their own lengths so the paws land there; skin that would sink
into the floor raises the whole dog.

A rig with Blender's rigify dog skeleton (thigh / shin / foot / toe,
front_thigh / front_shin / front_foot / front_toe, spine.00N; the glTF import's
`_<n>` suffixes are fine) is mapped by name. Any other four-legged rig standing
on all fours in its rest pose is mapped by its skeleton's structure, or by a
`roles=` map. `profile(armature)` returns the mapping, or a ValueError saying
why there is none.
"""
from __future__ import annotations

import itertools
import math
import os
import re

import numpy as np

LIBRARY = os.environ.get("DYCRAFT_DOG_MOCAP") or os.path.join(
    os.environ.get("DYCRAFT_MOCAP_ROOT") or "/references/mocap", "mann_dog")
#: The capture whose standing frames define the neutral pose every delta is taken from.
NEUTRAL_CLIP = "D1_001_KAN01_001"
#: How high a tail bone's tip is kept above the floor (about the tail's skin radius).
TAIL_CLEARANCE_M = 0.035
#: Where a paw is out of reach the wrist (hock) goes to the nearest point it can take; a
#: leg so straight that its captured joint lies within REACH_SIDE_NOISE_M of the
#: shoulder-paw axis has no nearest side, and leans up to REACH_SIDE_LEAN_M behind the paw.
REACH_SIDE_NOISE_M = 0.04
REACH_SIDE_LEAN_M = 0.03
#: How far below the body's back the front elbow's pole leans, per unit of 'back'.
FRONT_POLE_DOWN = 0.7
#: How much more folded than its neutral stand (as a fraction of the leg's length) a
#: captured leg must be before its paw is set by the leg's fraction of extension alone.
FOLD_BLEND = 0.15
#: Shoulders below this fraction of their standing height read as lying.
LIE_SHOULDER_FRAC = 0.7
#: A folded leg's elbow (stifle) joint center is kept this far above its paw's joint height.
KNEE_FLOOR_M = 0.012
#: The most a knee (elbow) may turn about its girdle-wrist axis from one frame to the next.
KNEE_STEP_DEG = 25.0
#: How far a knee (elbow) may swing off its leg's natural side, about the girdle-to-wrist line.
KNEE_SIDE_MAX_DEG = 45.0
#: A captured paw that rises less than this between two stances has not stepped.
NO_STEP_LIFT_M = 0.015
#: ...and that moves less than this: a shuffle, not a step.
NO_STEP_MOVE_M = 0.10
#: The least a stepping paw is lifted at mid-swing (scaled down for steps under 15 cm).
MIN_SWING_LIFT_M = 0.03
#: A swing over a step rises this fraction of the step's height above its eased path,
STEP_CLEARANCE = 0.1
#: and at least this high at mid-swing over a step: the capture's own arc, taken above its
#: lift-off and landing, can be a centimeter or two, and a paw skimming the nosing flips
#: the foreleg (~100 deg in a frame).
STEP_SWING_LIFT_M = 0.05
#: `retarget`'s default smoothing of the capture (a Gaussian, this many seconds wide).
SMOOTH_S = 0.05
#: How far skin may rest into the floor (fur, a compressed belly) before the whole dog is raised.
BODY_SINK_M = 0.015
#: About half a paw's length: the ground a swinging paw clears is taken this far around it.
PAW_RADIUS_M = 0.05
#: The fastest a swinging paw is lifted onto (or lowered from) a tread it is crossing.
SWING_RISE_M_PER_S = 1.0
#: A swing longer than this is not a stride: its paw keeps to the ground beneath it.
LONG_SWING_S = 0.8
#: A swinging paw is kept at least this fraction of its leg's length from the girdle.
MIN_PAW_REACH = 0.45
#: A planted paw is kept this far from the foot of a higher step.
EDGE_CLEARANCE_M = 0.08
#: A planted paw stands this far above the floor: pads exactly on it read as touching to
#: within float noise (7 um) and a collision check counted them as in it.
STANCE_CLEARANCE_M = 0.001

# ---------------------------------------------------------------- rig profile

PROFILE = dict(
    detect=("spine.004", "thigh.L", "front_thigh.L", "front_toe.L"),
    pelvis="spine.004",
    hips=("thigh.L", "thigh.R"),
    shoulders=("front_thigh.L", "front_thigh.R"),
    # bone -> (capture segment start, end, body frame)
    chain={
        "spine.005": ("Hips", "Spine1", "pel"),
        "spine.006": ("Spine1", "Neck", "tho"),
        "spine.007": ("Spine1", "Neck", "tho"),
        "spine.008": ("Neck", "Head", "tho"),
        "spine.009": ("Neck", "Head", "tho"),
        "spine.010": ("Neck", "Head", "tho"),
        "spine.011": ("Head", "Head_end", "tho"),
        "Head": ("Head", "Head_end", "tho"),
        "shoulder.L": ("LeftShoulder", "LeftArm", "tho"),
        "shoulder.R": ("RightShoulder", "RightArm", "tho"),
        "spine.003": ("Tail", "Tail1", "pel"),
        "spine.002": ("Tail", "Tail1", "pel"),
        "spine.001": ("Tail1", "Tail1_end", "pel"),
        "spine": ("Tail1", "Tail1_end", "pel"),
    },
    # Matched by direction, not by change from the neutral stand: a rig whose tail
    # rests level would otherwise add the capture's lift-from-hanging and point up.
    absolute=("spine.003", "spine.002", "spine.001", "spine"),
    legs={
        "FL": dict(bones=("front_thigh.L", "front_shin.L", "front_foot.L"), ctrl="front_toe.L",
                   cap=("LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end"), frame="tho"),
        "FR": dict(bones=("front_thigh.R", "front_shin.R", "front_foot.R"), ctrl="front_toe.R",
                   cap=("RightArm", "RightForeArm", "RightHand", "RightHand_end"), frame="tho"),
        "HL": dict(bones=("thigh.L", "shin.L", "foot.L"), ctrl="toe.L",
                   cap=("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftFoot_end"), frame="pel"),
        "HR": dict(bones=("thigh.R", "shin.R", "foot.R"), ctrl="toe.R",
                   cap=("RightUpLeg", "RightLeg", "RightFoot", "RightFoot_end"), frame="pel"),
    },
)
PAWS = ("LeftHand_end", "RightHand_end", "LeftFoot_end", "RightFoot_end")


def _resolve_name(names, base):
    if base in names:
        return base
    pat = re.compile(re.escape(base) + r"_\d+$")
    hits = [n for n in names if pat.match(n)]
    return hits[0] if len(hits) == 1 else None


def _ancestors(parents, name):
    """`name`'s ancestors, nearest first; a parenting loop ends the walk."""
    out, p = [], parents.get(name)
    while p is not None and p not in out and p != name:
        out.append(p)
        p = parents.get(p)
    return out


def _check_roles(armature, q):
    """Raise ValueError unless the bones `q` maps by name sit where the rigify dog has them.

    The mapping is by name, so a quadruped whose spine is numbered differently (a longer tail, a
    shorter thorax) or whose hip bones are named `shoulder.*` would be driven with the wrong capture
    segments: the hind legs must hang from the pelvis (or from lumbar bones under it, as on Blender's Rigify wolf
    metarig), the front legs from the spine above it through the mapped shoulders,
    and the tail bones must carry no leg."""
    parents = {b.name: getattr(getattr(b, "parent", None), "name", None) for b in armature.data.bones}
    pelvis, hips, fronts = q["pelvis"], q["hips"], q["shoulders"]
    common = [a for a in _ancestors(parents, hips[0]) if a in _ancestors(parents, hips[1])]
    hips_bones = {b for b, (start, _, _) in q["chain"].items() if start == "Hips"}
    lumbar, grew = set(), True
    while grew:                               # Hips bones hanging from the pelvis, one below another
        more = {b for b in hips_bones - lumbar if parents.get(b) == pelvis or parents.get(b) in lumbar}
        grew = bool(more)
        lumbar |= more
    if not common or common[0] not in (pelvis, *lumbar):
        carrier = common[0] if common else "no common bone"
        raise ValueError(f"{armature.name}: the hind legs hang from {carrier}, not from {pelvis}; "
                         "its spine is not numbered like the rigify dog's, so dog_retarget cannot map it")
    for front, seg in zip(fronts, ("LeftShoulder", "RightShoulder")):
        anc = _ancestors(parents, front)
        spine_above = [a for a in anc if a in q["chain"] and q["chain"][a][2] == "tho"
                       and q["chain"][a][0] not in ("LeftShoulder", "RightShoulder")]
        if pelvis not in anc or not spine_above:
            raise ValueError(f"{armature.name}: {front} does not hang from the spine above {pelvis}")
        for bone, (start, _, _) in q["chain"].items():
            if start == seg and bone not in anc:
                raise ValueError(f"{armature.name}: {bone} is mapped as a front shoulder but does not carry {front}")
    legs = (*hips, *fronts)
    for bone in q["absolute"]:
        if pelvis not in _ancestors(parents, bone) or any(bone in _ancestors(parents, leg) for leg in legs):
            raise ValueError(f"{armature.name}: {bone} is mapped as tail but is not a tail bone below {pelvis}")


#: A paw's joints lie within this share of the skeleton's height above its lowest joint.
FLOOR_FRAC = 0.12
#: Joints closer than this share of a leg's length are one joint (helper bones stacked on a joint).
JOINT_EPS = 0.02
#: A paw is within this share of its leg's height above the floor.
PAW_FRAC = 0.15
#: Left and right legs' segments agree within this share of their length.
MIRROR_FRAC = 0.2
#: A standing body's back, hips to shoulders, rises at most this much (degrees); more is a sitting or rearing rest.
MAX_BACK_PITCH = 35.0


def _stray_nubs(par, H):
    """Leaf bones an exporter dropped to the floor from high on the body, far from the bone they hang from (an end
    bone of the head left on the floor beneath it): taken as a paw, or as the floor, they hide the real paws. A leaf
    counts when its head is in the lowest tenth of the skeleton's height, its parent's head above half of it, and the
    drop over three times the skeleton's typical bone."""
    hop = {n: float(np.linalg.norm(H[n] - H[p_])) for n, p_ in par.items() if p_ is not None}
    if not hop:
        return set()
    typical = float(np.median(list(hop.values())))
    z = np.array([h[2] for h in H.values()])
    z0, height = float(z.min()), float(np.ptp(z)) or 1.0
    parents = set(par.values())
    return {n for n, d in hop.items() if n not in parents and d > 3 * typical
            and H[n][2] - z0 < 0.1 * height and H[par[n]][2] - z0 > 0.5 * height}


def _structural_profile(bones):
    """The rigify-dog role map built from a four-legged skeleton's rest structure, for rigs without the rigify
    names. `bones`: dicts with `name`, `parent` (a name or None), `head` and `tail` in armature space, Z up, and
    optionally `deform`. Only joint positions (bone heads) and parentage are read, so bone axes and names don't
    matter. Raises ValueError saying what is missing or ambiguous.

    - Paws: the four places where a chain first comes down to the floor. Front and hind are the two ends of the
      body's long axis, front being the end whose non-leg branches end in more deforming bones (the head, with
      its jaw, eyes, and ears, against a tail); left is up x forward.
    - Legs: from each paw up, stacked joints merged; the paw's bone is the control, the three bones above it are
      foot, shin, and thigh; on a front leg the bones above the thigh are the shoulder.
    - Trunk: the pelvis is the four legs' common ancestor; the bones down to the front legs' branch are the spine,
      the chain on from there to the most-branched end is the neck and head, and the hind end's chain is the tail."""
    by = {b["name"]: b for b in bones}
    par = {n: (b["parent"] if b["parent"] in by else None) for n, b in by.items()}
    H = {n: np.asarray(b["head"], float) for n, b in by.items()}
    deform = {n: bool(b.get("deform", True)) for n, b in by.items()}
    nubs = _stray_nubs(par, H)
    by = {n: b for n, b in by.items() if n not in nubs}
    par = {n: (p_ if p_ in by else None) for n, p_ in par.items() if n in by}
    H = {n: h for n, h in H.items() if n in by}
    deform = {n: d for n, d in deform.items() if n in by}
    kids = {}
    for n, p_ in par.items():
        kids.setdefault(p_, []).append(n)

    def anc(n):
        out, p_ = [], par[n]
        while p_ is not None and p_ not in out:
            out.append(p_)
            p_ = par[p_]
        return out

    def sub(n):
        out, todo = [], [n]
        while todo:
            m = todo.pop()
            out.append(m)
            todo.extend(kids.get(m, []))
        return out

    skin = [n for n in by if deform[n]] or list(by)          # helper bones (roots, IK targets) may sit anywhere
    z = np.array([H[n][2] for n in skin])
    z0, height = float(z.min()), float(z.max() - z.min())
    if height <= 0:
        raise ValueError("the skeleton is flat: no joint above another")

    def lca(names):
        common = None
        for n in names:
            chain = [n] + anc(n)
            common = chain if common is None else [a for a in chain if a in common]
        if not common:
            raise ValueError(f"{names} share no ancestor: the legs are not one skeleton")
        return common[0]

    allxy = np.array([H[n][:2] for n in skin])
    w, v = np.linalg.eigh(np.cov((allxy - allxy.mean(0)).T))
    axis, lat = v[:, 1], v[:, 0]
    mid = allxy.mean(0)
    length = float(np.ptp(allxy @ axis))
    low = {n for n in skin if H[n][2] - z0 < FLOOR_FRAC * height}

    def down(n):
        """The chain from `n` on down through the child whose subtree reaches lowest."""
        out = [n]
        while True:
            cs = [c for c in kids.get(out[-1], []) if deform[c] or not skin]
            if not cs:
                return out
            out.append(min(cs, key=lambda c: min(H[m][2] for m in sub(c))))

    # Each chain that comes down to the floor, where it first reaches it; chains on the body's midline (a tail
    # tip, a jaw) are not legs.
    touch = H
    cands = sorted(n for n in low if par[n] not in low and len(anc(n)) >= 3
                   and abs((touch[n][:2] - mid) @ lat) > 0.03 * length)
    # Legs come in mirrored pairs: each foot with the one across the midline nearest its mirror image.
    side_of = lambda n: (touch[n][:2] - mid) @ lat
    mirror = lambda a, b: abs((touch[a][:2] - touch[b][:2]) @ axis) + abs(side_of(a) + side_of(b))
    near = {}
    for a in cands:
        opp = [b for b in cands if side_of(a) * side_of(b) < 0]
        if opp:
            near[a] = min(opp, key=lambda b: mirror(a, b))
    reach = lambda a, top: sum(float(np.linalg.norm(H[x] - H[par[x]])) for x in [a] + anc(a)[: anc(a).index(top)])
    pairs = []
    for a, b in near.items():
        if a < b and near.get(b) == a and mirror(a, b) < 0.25 * length:
            top = lca([a, b])
            if top not in (a, b):
                pairs.append((min(reach(a, top), reach(b, top)), a, b))
    # A rest pose caught mid-stride puts a paw well ahead of its twin, and one with a paw lifted holds it above the
    # floor, so by position the two never pair. A paw left unpaired then pairs with the joint across the midline
    # that sits as deep in the skeleton and low on the body (in its lowest third), the nearest to its mirror image.
    depth = lambda n: len(anc(n))
    lifted = {}                                 # a lifted paw: its twin on the floor
    paired = {x for _, a, b in pairs for x in (a, b)}
    taken = set().union(*(set([x] + anc(x)) for x in paired)) if paired else set()
    for a in sorted(n for n in cands if n not in paired):
        if a in paired:
            continue
        opp = [b for b in skin if b not in taken and b not in paired and depth(b) == depth(a)
               and side_of(a) * side_of(b) < 0 and H[b][2] - z0 < height / 3]
        if not opp:
            continue
        b = min(opp, key=lambda b: mirror(a, b))
        top = lca([a, b])
        if top not in (a, b):
            pairs.append((min(reach(a, top), reach(b, top)), a, b))
            paired |= {a, b}
            taken |= set([a, b] + anc(a) + anc(b)) - set(anc(top)) - {top}
            if b not in cands:
                lifted[b] = a
    pairs = sorted(p_ for p_ in pairs if p_[0] >= 0.25 * height)[::-1][:2]
    if len(pairs) < 2:
        raise ValueError(f"{len(pairs)} mirrored pairs of legs come down to the floor, not two "
                         f"(chains at the floor: {sorted(cands)[:8] or 'none'}; is the rig standing on four legs "
                         "in its rest pose?)")
    paws = [x for c in pairs for x in c[1:]]
    pos = {n: float((touch[n][:2] - mid) @ axis) for n in paws}
    ends = sorted([sorted(c[1:]) for c in pairs], key=lambda pr: pos[pr[0]] + pos[pr[1]])
    if abs((pos[ends[0][0]] + pos[ends[0][1]]) - (pos[ends[1][0]] + pos[ends[1][1]])) / 2 < 0.15 * length:
        raise ValueError(f"the two pairs of legs ({ends[0]}, {ends[1]}) stand at one end of the body")
    # The bone that moves the whole body carries every deforming bone of the legs' hierarchy: on many rigs the
    # legs hang from a bone below the skinned hips, rump, or tail (a Biped's thighs hang from its spine), which a
    # pelvis taken as the legs' common ancestor would leave standing while the body walks away. Bones under
    # another root are controls (an eye target, an IK platform) and stay as they are.
    root_of = lambda n: ([n] + anc(n))[-1]
    pel = lca([n for n in skin if root_of(n) == root_of(paws[0])])
    end_lca = [lca(pair) for pair in ends]
    if end_lca[0] == end_lca[1]:
        raise ValueError(f"the front and hind legs branch from one bone, {end_lca[0]}")
    legs_sub = set().union(*(set([p_] + anc(p_)) for p_ in paws))

    def extras(root):
        """Deforming leaf bones under `root`, outside the legs and the spine."""
        out = []
        for c in kids.get(root, []):
            if c in legs_sub:
                continue
            out += [m for m in sub(c) if deform[m] and not kids.get(m)]
        return out

    # The head end has more branch tips (jaw, eyes, ears) beyond its legs' branch than the tail end; between a
    # one-bone head and a one-bone tail, the head reaches higher.
    tips = [extras(e) for e in end_lca]
    top = [max([H[m][2] for t_ in tips[i] for m in [t_] + anc(t_)[: anc(t_).index(end_lca[i])]] or [z0])
           for i in range(2)]
    if len(tips[0]) != len(tips[1]):
        fi = int(np.argmax([len(t_) for t_ in tips]))
    elif abs(top[0] - top[1]) > 0.05 * height:
        fi = int(np.argmax(top))
    else:
        raise ValueError(f"no end of the body is clearly the head ({end_lca[0]} and {end_lca[1]} have "
                         f"{len(tips[0])} branch tips each, at one height)")
    if not tips[fi]:
        raise ValueError(f"nothing but legs leaves either end of the body ({end_lca[0]}, {end_lca[1]})")
    front, hind = ends[fi], ends[1 - fi]
    front_lca, hind_lca = end_lca[fi], end_lca[1 - fi]
    fwd = np.zeros(3)
    fwd[:2] = axis if fi == 1 else -axis
    left = np.cross([0.0, 0.0, 1.0], fwd)

    def leg(entry, top, front, hind_paw_h=None):
        """(thigh, shin, foot, paw) and, on a front leg, the shoulder blade: the leg's joints from below `top`
        to its tip, stacked joints merged (the uppermost bone at each).
        - Joints at the top that sit on the body's midline or run level inside the trunk (a clavicle, a pelvis
          helper stepping out to the hip) are not the leg: a hip and a shoulder are off to one side.
        - The paw is the uppermost joint near the floor with three segments above it: a dog's or cat's toe
          joint, a bear's ball (its ankle is on the floor too), a horse's fetlock (above the hoof).
        - The knee and hock (elbow and wrist) are the joints between hip and paw that best fit the whole chain,
          so twist, ribbon, and fix bones in between are passed over. A front leg keeps a shoulder blade above
          the shoulder joint when the chain bends there."""
        path = [a for a in anc(entry)[: anc(entry).index(top)]][::-1] + down(entry)
        size = float(np.linalg.norm(H[path[0]] - H[path[-1]])) or 1.0
        k = []
        for n in path:
            if not k or np.linalg.norm(H[n] - H[k[-1]]) >= JOINT_EPS * size:
                k.append(n)
        X = np.array([H[n] for n in k])
        zlow = float(X[:, 2].min())
        t = 0
        side_off = lambda i: abs((X[i, :2] - mid) @ lat)
        on_midline = lambda i: side_off(i) < 0.25 * side_off(len(k) - 1)    # beside the body's center, not the paw
        while t + 3 < len(k) and (on_midline(t) or abs(X[t + 1, 2] - X[t, 2]) < 0.1 * (X[t, 2] - zlow)):
            t += 1
        reach_h = X[t, 2] - zlow
        paw_at = lambda n_seg: next((i for i in range(t + n_seg, len(k)) if X[i, 2] - zlow < PAW_FRAC * reach_h),
                                    None)

        def fit(n_inner, p):
            best = None
            for inner in itertools.combinations(range(t + 1, p), n_inner):
                pts = [t, *inner, p]
                err = 0.0
                for a, b in zip(pts, pts[1:]):
                    seg = X[b] - X[a]
                    L2 = float(seg @ seg) or 1e-12
                    for m in range(a + 1, b):
                        u = min(1.0, max(0.0, float((X[m] - X[a]) @ seg) / L2))
                        err = max(err, float(np.linalg.norm(X[m] - X[a] - u * seg)))
                if best is None or err < best[0]:
                    best = (err, pts)
            return best[1] if best else None

        # A front leg may have a shoulder blade (or clavicle) above its shoulder joint, which then sits high
        # (two-thirds to seven-eighths of the leg's height on dogs, cats, deer, and horses) with the elbow near
        # half; without one the leg is three segments, like a hind leg. Of the two readings, the one whose paw
        # stands as high off the floor as the hind paws is the front leg's: both are the same contact joint
        # (a dog's toe joint on the floor, a horse's fetlock above the hoof).
        options = []
        p4 = paw_at(4) if front else None
        if p4 is not None:
            pts = fit(3, p4)
            if X[pts[1], 2] - zlow > 0.6 * reach_h and X[pts[2], 2] - zlow > 0.3 * reach_h:
                options.append((tuple(k[i] for i in pts[1:]), k[pts[0]], X[p4, 2] - zlow))
        p = paw_at(3)
        if p is not None:
            options.append((tuple(k[i] for i in fit(2, p)), None, X[p, 2] - zlow))
        if not options:
            return None, None, None
        if hind_paw_h is not None and len(options) > 1:
            options.sort(key=lambda o: abs(o[2] - hind_paw_h))
        return options[0]

    legs, shoulder = {}, {}
    centre = np.mean([touch[n] for n in paws], axis=0)
    hind_paw_h = []
    def at_depth(n, d):
        """The joint of `n`'s chain at depth `d`: an ancestor, or a bone further down the chain."""
        if d <= depth(n):
            return ([n] + anc(n))[depth(n) - d]
        below = down(n)
        return below[min(d - depth(n), len(below) - 1)]
    for pos, pair, top in (("H", hind, hind_lca), ("F", front, front_lca)):
        read = {}
        for paw in sorted(pair, key=lambda n: n in lifted):         # a lifted paw after its twin on the floor
            side = "L" if (touch[paw] - centre) @ left > 0 else "R"
            if paw in lifted and lifted[paw] in read:
                # A lifted leg is read as its twin is, joint for joint at the same depths: folded off the floor, it
                # has no contact joint of its own to find the paw by.
                tb, tblade, paw_h = read[lifted[paw]]
                bones4 = tuple(at_depth(paw, depth(x)) for x in tb)
                blade = at_depth(paw, depth(tblade)) if tblade else None
            else:
                bones4, blade, paw_h = leg(paw, top, pos == "F", float(np.mean(hind_paw_h)) if hind_paw_h else None)
            read[paw] = (bones4, blade, paw_h)
            if pos == "H" and paw_h is not None:
                hind_paw_h.append(paw_h)
            name = f"the {'front' if pos == 'F' else 'hind'} {side} leg"
            if bones4 is None:
                raise ValueError(f"{name}, down to {paw}, has fewer than three segments above its paw")
            legs[pos + side] = bones4
            # The thigh's joint is the hip or shoulder, up by the trunk, not the knee or elbow.
            if H[bones4[0]][2] - z0 < 0.55 * (H[top][2] - z0):
                raise ValueError(f"{name}'s upper joint, {bones4[0]}, is low for a hip or shoulder")
            if blade:
                shoulder[side] = blade
    if set(legs) != {"FL", "FR", "HL", "HR"}:
        raise ValueError(f"the paws don't make a left and a right on each end: {sorted(legs)}")
    # Four paws on the floor with the back pitched up is a sitting or rearing rest: the captures would be laid on a
    # body that stays upright through the walk.
    hips_mid = (H[legs["HL"][0]] + H[legs["HR"][0]]) / 2
    shoulders_mid = (H[legs["FL"][0]] + H[legs["FR"][0]]) / 2
    back = shoulders_mid - hips_mid
    pitch = float(np.degrees(np.arctan2(abs(back[2]), np.linalg.norm(back[:2]))))
    if pitch > MAX_BACK_PITCH:
        raise ValueError(f"it rests sitting or rearing, not standing (its back rises {pitch:.0f} degrees from the hips "
                         f"to the shoulders)")
    for e in "FH":
        # The body's left-right axis runs between the hips and between the shoulders; at one point it has none.
        apart = float(np.linalg.norm((H[legs[e + "L"][0]] - H[legs[e + "R"][0]])[:2]))
        if apart < 0.03 * length:
            raise ValueError(f"the {'shoulders' if e == 'F' else 'hips'} ({legs[e + 'L'][0]}, {legs[e + 'R'][0]}) sit "
                             "at one point, so the body has no left-right axis")
        segs = [[float(np.linalg.norm(H[b] - H[a])) for a, b in zip(legs[e + sd], legs[e + sd][1:])] for sd in "LR"]
        if any(abs(l - r) > MIRROR_FRAC * max(l, r) for l, r in zip(*segs)):
            raise ValueError(f"the {'front' if e == 'F' else 'hind'} legs are not mirror images: {legs[e + 'L']} "
                             f"and {legs[e + 'R']} have segments of {[round(x, 3) for x in segs[0]]} and "
                             f"{[round(x, 3) for x in segs[1]]}")

    def walk(start, avoid):
        """The chain from `start` toward the most-branched non-leg child, until it branches or ends."""
        out, n = [], start
        while True:
            cs = [c for c in kids.get(n, []) if c not in legs_sub and c not in avoid]
            if not cs:
                return out
            n = max(cs, key=lambda c: len([m for m in sub(c) if deform[m] and not kids.get(m)]))
            out.append(n)
            if len([c for c in kids.get(n, []) if c not in legs_sub]) > 1:
                return out

    spine = [a for a in ([front_lca] + anc(front_lca)) if a != pel and pel in anc(a)][::-1]
    head_chain = walk(front_lca, set(spine))
    avoid = set(spine) | set(head_chain)
    # The tail hangs from the hind legs' branch or a bone between it and the pelvis: of the chains leaving them,
    # the one reaching farthest behind the hind legs.
    behind = [hind_lca] + (anc(hind_lca)[: anc(hind_lca).index(pel) + 1] if pel in anc(hind_lca) else [])
    starts = [c for b in behind for c in kids.get(b, []) if c not in legs_sub and c not in avoid and deform[c]]
    reach_back = lambda c: -min(float((H[m] - H[hind_lca]) @ fwd) for m in sub(c))
    tail_chain = []
    if starts:
        c = max(starts, key=reach_back)
        if reach_back(c) > 0:
            tail_chain = [c] + walk(c, avoid)
    if not head_chain:
        raise ValueError(f"no neck or head chain leaves the front legs' branch, {front_lca}")
    chain = {}
    lumbar = spine[: max(1, len(spine) // 3)] if len(spine) > 1 else []
    for b in spine:
        chain[b] = ("Hips", "Spine1", "pel") if b in lumbar else ("Spine1", "Neck", "tho")
    if hind_lca != pel:                      # hind legs on a lumbar chain below the pelvis
        for b in [hind_lca] + anc(hind_lca)[: anc(hind_lca).index(pel)]:
            chain[b] = ("Hips", "Spine1", "pel")
    for b in head_chain[:-1]:
        chain[b] = ("Neck", "Head", "tho")
    chain[head_chain[-1]] = ("Head", "Head_end", "tho")
    for side, b in shoulder.items():
        seg = "LeftShoulder" if side == "L" else "RightShoulder"
        chain[b] = (seg, seg.replace("Shoulder", "Arm"), "tho")
    half = max(1, len(tail_chain) // 2)
    for i, b in enumerate(tail_chain):
        chain[b] = ("Tail", "Tail1", "pel") if i < half else ("Tail1", "Tail1_end", "pel")
    cap = {"FL": ("LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end"),
           "FR": ("RightArm", "RightForeArm", "RightHand", "RightHand_end"),
           "HL": ("LeftUpLeg", "LeftLeg", "LeftFoot", "LeftFoot_end"),
           "HR": ("RightUpLeg", "RightLeg", "RightFoot", "RightFoot_end")}
    return dict(
        pelvis=pel,
        hips=(legs["HL"][0], legs["HR"][0]),
        shoulders=(legs["FL"][0], legs["FR"][0]),
        chain=chain,
        absolute=tuple(tail_chain),
        legs={k: dict(bones=v[:3], ctrl=v[3], cap=cap[k], frame="tho" if k[0] == "F" else "pel")
              for k, v in legs.items()},
    )


def profile(armature, roles=None):
    """The rigify-dog mapping resolved to this armature's bone names, or ValueError.

    The rigify dog's bone names are used where the rig has them. Otherwise the roles come from the skeleton's
    structure (`_structural_profile`), or from `roles`, a map in the same form the function returns, when given.
    Every refusal is a ValueError naming the bones, so a caller can fall back to its own motion."""
    if roles is not None:
        q = dict(roles)
        _check_roles(armature, q)
        return q
    try:
        return _named_profile(armature)
    except ValueError as named:
        if any(getattr(b, "head_local", None) is None for b in armature.data.bones):
            raise                                   # no rest positions to read the structure from
        rest = [dict(name=b.name, parent=b.parent.name if b.parent else None, head=tuple(b.head_local),
                     tail=tuple(b.tail_local), deform=b.use_deform) for b in armature.data.bones]
        try:
            q = _structural_profile(rest)
        except ValueError as e:
            raise ValueError(f"{named}, and its skeleton could not be read by its shape: {e}. Pass roles= with "
                             "your own map, or animate it another way.") from None
        _check_roles(armature, q)
        return q


def _named_profile(armature):
    names = [b.name for b in armature.data.bones]
    res = lambda b: _resolve_name(names, b)
    needed = (*PROFILE["detect"], PROFILE["pelvis"], *PROFILE["hips"], *PROFILE["shoulders"],
              *(b for leg in PROFILE["legs"].values() for b in (*leg["bones"], leg["ctrl"])))
    missing = [b for b in dict.fromkeys(needed) if not res(b)]
    if missing:
        raise ValueError(f"{armature.name} has no rigify dog bone names (no bone {missing})")
    q = dict(PROFILE)
    q["pelvis"] = res(PROFILE["pelvis"])
    q["hips"] = tuple(map(res, PROFILE["hips"]))
    q["shoulders"] = tuple(map(res, PROFILE["shoulders"]))
    q["chain"] = {res(b): v for b, v in PROFILE["chain"].items() if res(b)}
    q["absolute"] = tuple(res(b) for b in PROFILE["absolute"] if res(b))
    q["legs"] = {k: dict(v, bones=tuple(map(res, v["bones"])), ctrl=res(v["ctrl"]))
                 for k, v in PROFILE["legs"].items()}
    _check_roles(armature, q)
    return q


# ---------------------------------------------------------------- BVH

def _bvh_path(name):
    if os.path.isfile(name):
        return name
    p = os.path.join(LIBRARY, "data", name if name.endswith(".bvh") else name + ".bvh")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no capture {name!r} (looked for {p})")
    return p


def _parse(path):
    toks = open(path).read().split()
    i = 0; joints = []; stack = []
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            joints.append(dict(name=toks[i + 1], parent=stack[-1] if stack else -1, chans=[])); i += 2
        elif t == "End":
            joints.append(dict(name=joints[stack[-1]]["name"] + "_end", parent=stack[-1], chans=[])); i += 2
        elif t == "{":
            stack.append(len(joints) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            joints[stack[-1]]["offset"] = np.array(list(map(float, toks[i + 1:i + 4]))); i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1]); joints[stack[-1]]["chans"] = toks[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    nf = int(toks[i + 2]); ft = float(toks[i + 5])
    return joints, np.array(toks[i + 6:], float).reshape(nf, -1), ft


def _rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3)); R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1
    p, q = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    R[:, p, p] = c; R[:, q, q] = c; R[:, p, q] = -s; R[:, q, p] = s
    return R


def _world(path):
    """`(names, P[n, joints, 3] in metres, Z up, fps)`. Root position channels are
    absolute (the ROOT OFFSET is ignored, as Blender's importer does)."""
    joints, data, ft = _parse(path)
    n = len(data); col = 0
    P = np.zeros((n, len(joints), 3)); Rw = [None] * len(joints)
    for j, J in enumerate(joints):
        R = np.tile(np.eye(3), (n, 1, 1)); t = np.zeros((n, 3))
        for ch in J["chans"]:
            v = data[:, col]; col += 1
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], v)
        off = J.get("offset", np.zeros(3))
        if J["parent"] < 0:
            P[:, j] = t; Rw[j] = R
        else:
            p = J["parent"]
            P[:, j] = P[:, p] + np.einsum("nij,j->ni", Rw[p], off) + np.einsum("nij,nj->ni", Rw[p], t)
            Rw[j] = Rw[p] @ R
    P *= 0.01
    P = np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)
    return [J["name"] for J in joints], P, round(1 / ft)


def _index_rows(name):
    try:
        lines = open(os.path.join(LIBRARY, "index.txt")).read().splitlines()
    except OSError:
        return []
    stem = os.path.basename(name)[:-4] if name.endswith(".bvh") else os.path.basename(name)
    return [l.split("\t", 1)[1] for l in lines if l.startswith(stem + "\t")]


def inspect_bvh(name):
    """Length, travel, and the index's labeled spans of one capture (no scene change)."""
    names, P, fps = _world(_bvh_path(name))
    J = {k: j for j, k in enumerate(names)}
    hip = (P[:, J["LeftUpLeg"], :2] + P[:, J["RightUpLeg"], :2]) / 2
    step = np.linalg.norm(np.diff(hip, axis=0), axis=1).sum()
    return {"frames": len(P), "fps": fps, "seconds": round(len(P) / fps, 3),
            "root_travel_m": round(float(step), 3),
            "root_displacement_m": round(float(np.linalg.norm(hip[-1] - hip[0])), 3),
            "spans": _index_rows(name)}


# ---------------------------------------------------------------- small math

def _frame(left, fwd):
    f = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
    l = left - (left * f).sum(-1, keepdims=True) * f
    l /= np.linalg.norm(l, axis=-1, keepdims=True)
    return np.stack([l, f, np.cross(l, f)], -1)


def _swing(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax); return 2 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _axis_angle(ax, t):
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _two_bone(root, target, l1, l2, pole):
    d = target - root; L = np.linalg.norm(d)
    L = min(max(L, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    dn = d / max(np.linalg.norm(d), 1e-9)
    a = (l1 * l1 + L * L - l2 * l2) / (2 * L)
    h = math.sqrt(max(l1 * l1 - a * a, 0))
    p = pole - root; p = p - np.dot(p, dn) * dn
    return root + dn * a + p / max(np.linalg.norm(p), 1e-9) * h


def _box(x, w):
    w = max(1, int(w)); k = np.ones(w) / w; p = w // 2
    xp = np.pad(x, [(p, w - 1 - p)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def _dilate(x, w):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    return sw(np.pad(x, (w // 2, w // 2), mode="edge"), w).max(1)


def _gauss(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma)); t = np.arange(-r, r + 1); k = np.exp(-0.5 * (t / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


# ---------------------------------------------------------------- clips

class Clip:
    """Captured joint positions, edited before they are solved onto a rig.

    `P[n, joint, 3]` in meters, Z up, with the capture's floor at z = 0. Every
    edit returns a new Clip. A clip that has been through `follow` is *placed*:
    its coordinates are where the dog should be (in the armature's space at
    bake time) and `retarget` keeps them; any other clip is placed by
    `retarget` at the dog's rest position, facing the dog's own front.
    """

    def __init__(self, names, P, fps, source="", placed=False):
        self.names = list(names); self.P = P; self.fps = fps
        self.source = source; self.placed = placed
        self.base = None; self.path = None          # set by follow(): re-fitted after stride scaling
        self.J = {k: j for j, k in enumerate(self.names)}

    # -- facts
    @property
    def seconds(self):
        return len(self.P) / self.fps

    def _hips(self, P=None):
        P = self.P if P is None else P
        return (P[:, self.J["LeftUpLeg"]] + P[:, self.J["RightUpLeg"]]) / 2

    def _heading(self, P=None):
        P = self.P if P is None else P
        f = (P[:, self.J["LeftArm"]] + P[:, self.J["RightArm"]]) / 2 - self._hips(P)
        return np.unwrap(np.arctan2(f[:, 1], f[:, 0]))

    def info(self):
        hip = self._hips()[:, :2]
        return {"seconds": round(self.seconds, 3), "frames": len(self.P), "fps": self.fps,
                "root_travel_m": round(float(np.linalg.norm(np.diff(hip, axis=0), axis=1).sum()), 3),
                "turn_deg": round(math.degrees(self._heading()[-1] - self._heading()[0]), 1),
                "placed": self.placed, "source": self.source}

    def _new(self, P, source=None, placed=None):
        return Clip(self.names, P, self.fps, self.source if source is None else source,
                    self.placed if placed is None else placed)

    # -- edits
    def section(self, start_s=None, end_s=None):
        a = 0 if start_s is None else int(round(start_s * self.fps))
        b = len(self.P) if end_s is None else int(round(end_s * self.fps))
        if b - a < 2:
            raise ValueError(f"empty section {start_s}..{end_s} of a {self.seconds:.2f} s clip")
        return self._new(self.P[a:b].copy(), f"{self.source}[{start_s}:{end_s}]")

    def speed(self, factor):
        """Play `factor` times faster (stride unchanged)."""
        if factor <= 0:
            raise ValueError("speed factor must be positive")
        n = len(self.P); t = np.arange(n) / self.fps
        tq = np.arange(0, t[-1] + 1e-9, factor / self.fps)
        Q = np.empty((len(tq),) + self.P.shape[1:])
        for j in range(self.P.shape[1]):
            for c in range(3):
                Q[:, j, c] = np.interp(tq, t, self.P[:, j, c])
        return self._new(Q, f"{self.source}*speed{factor}")

    def smooth(self, sigma_frames=1.5):
        """Gaussian-smooth every joint track over time (removes capture jitter)."""
        return self._new(_gauss(self.P, sigma_frames), f"{self.source}*smooth")

    def period(self, lo_s=0.25, hi_s=2.0):
        """The gait period in seconds, from the paws' heights (autocorrelation)."""
        z = self.P[:, [self.J[k] for k in PAWS], 2]
        z = z - z.mean(0)
        best, bt = -1e9, None
        for lag in range(int(lo_s * self.fps), min(int(hi_s * self.fps), len(z) // 2)):
            r = float((z[:-lag] * z[lag:]).sum() / max(len(z) - lag, 1))
            if r > best:
                best, bt = r, lag
        if bt is None:
            raise ValueError("clip too short to find a gait period")
        return bt / self.fps

    def cycle(self, *, seconds=None, cycles=None):
        """Repeat one gait period of this (steady) gait clip, each copy continuing from
        where the last left off, with the heading drift carried on."""
        T = int(round(self.period() * self.fps))
        unit = self.P[:T + 1]
        hip = self._hips(unit); hd = self._heading(unit)
        dpos = hip[-1, :2] - hip[0, :2]; dyaw = hd[-1] - hd[0]
        need = cycles if cycles is not None else math.ceil((seconds or self.seconds) * self.fps / T)
        parts = []; off = np.zeros(2); yaw = 0.0; pivot = hip[0, :2]
        for i in range(int(need)):
            Q = unit[:-1].copy()
            Q[..., :2] = (Q[..., :2] - pivot) @ _rz(yaw)[:2, :2].T + pivot + off
            parts.append(Q)
            off = off + _rz(yaw)[:2, :2] @ dpos; yaw += dyaw
        out = self._new(np.concatenate(parts), f"{self.source}*cycle")
        return out.section(0, seconds) if seconds is not None else out

    def then(self, other, blend_s=0.3):
        """This clip followed by `other`, which is moved to start where and facing
        the way this one ends; the two cross-fade over `blend_s`."""
        if other.fps != self.fps:
            raise ValueError(f"cannot join a {other.fps} fps clip to a {self.fps} fps one")
        b = max(1, min(int(round(blend_s * self.fps)), len(self.P) - 1, len(other.P) - 1))
        a_end = len(self.P) - b
        h1 = self._hips()[a_end, :2]; y1 = self._heading()[a_end]
        h2 = other._hips()[0, :2]; y2 = other._heading()[0]
        R = _rz(y1 - y2)
        Q = other.P.copy()
        Q[..., :2] = (Q[..., :2] - h2) @ R[:2, :2].T + h1
        w = np.linspace(0, 1, b)[:, None, None]
        mid = (1 - w) * self.P[a_end:] + w * Q[:b]
        P = np.concatenate([self.P[:a_end], mid, Q[b:]])
        return self._new(P, f"{self.source}+{other.source}", placed=False)

    def stride(self, factor):
        """Scale the body's horizontal travel by `factor` (body pose unchanged)."""
        hip = self._hips()[:, None, :2]
        P = self.P.copy()
        P[..., :2] = hip[0] + (hip - hip[0]) * factor + (P[..., :2] - hip)
        return self._new(P, f"{self.source}*stride{factor:.2f}")

    def in_place(self, turn=True):
        """Remove the clip's travel (and, with `turn`, its change of heading) so a
        sit, a lie-down, or a stand-up happens where the dog already is."""
        hip = self._hips(); hd = self._heading()
        P = self.P.copy()
        for f in range(len(P)):
            R = _rz(-(hd[f] - hd[0]) if turn else 0.0)[:2, :2]
            P[f, :, :2] = (P[f, :, :2] - hip[f, :2]) @ R.T + hip[0, :2]
        return self._new(P, f"{self.source}*in_place")

    def follow(self, points, smooth_m=0.4, start_heading=True):
        """Bend the clip's travel onto a path of (x, y) points in the armature's space.

        Distance along the path is the clip's own distance traveled, so its speeds
        and pauses are kept; where the clip outruns the path it stops at the end.
        Each frame is turned about the hips so the dog faces along the path, which
        also turns the planted paws a little; `retarget` locks them per contact."""
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("points must be at least two (x, y) pairs")
        # resample the polyline densely and smooth its corners
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
        ds = 0.02; ss = np.arange(0, s[-1] + 1e-9, ds)
        path = np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1)
        if smooth_m > 0 and len(path) > 3:
            path = _gauss(path, smooth_m / ds / 2)
            path[0], path[-1] = pts[0], pts[-1]
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1); sp = np.r_[0, np.cumsum(seg)]
        tang = np.gradient(path, axis=0); th_p = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
        hip = self._hips(); hd = self._heading()
        travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(hip[:, :2], axis=0), axis=1))]
        u = np.minimum(travel, sp[-1])
        at = np.stack([np.interp(u, sp, path[:, 0]), np.interp(u, sp, path[:, 1])], 1)
        heading_on_path = np.interp(u, sp, th_p)
        # What the path should line up with. When the dog is moving briskly, its direction
        # of travel; when it is slow -- sniffing along, shuffling, standing -- the travel
        # direction is mostly sway and noise (following it spins a sniffing dog), so its
        # body's own heading instead, blended between the two by speed.
        v = _gauss(np.gradient(hip[:, :2], axis=0), self.fps / 4) * self.fps      # m/s
        speed = np.linalg.norm(v, axis=1)
        th_v = np.arctan2(v[:, 1], v[:, 0])
        # keep the travel direction on the same branch as the heading before blending
        th_v = hd + np.angle(np.exp(1j * (th_v - hd)))
        w = np.clip((speed - 0.3) / 0.4, 0.0, 1.0)            # 0 below 0.3 m/s, 1 above 0.7 m/s
        th_c = _gauss((1 - w) * hd + w * th_v, self.fps / 4)
        delta = _gauss(np.unwrap(heading_on_path - th_c), self.fps / 6)
        P = self.P.copy()
        for f in range(len(P)):
            R = _rz(delta[f])[:2, :2]
            P[f, :, :2] = (P[f, :, :2] - hip[f, :2]) @ R.T + at[f]
        out = self._new(P, f"{self.source}*path", placed=True)
        out.base, out.path = self, (pts, smooth_m)
        return out


def clip(name, start_s=None, end_s=None):
    """A span of one capture (a file name from `index.txt`, with or without `.bvh`)."""
    path = _bvh_path(name)
    names, P, fps = _world(path)
    J = {k: j for j, k in enumerate(names)}
    P[..., 2] -= np.percentile(P[:, [J[k] for k in PAWS], 2].min(1), 3)
    c = Clip(names, P, fps, os.path.basename(path)[:-4])
    return c.section(start_s, end_s) if (start_s is not None or end_s is not None) else c


# ---------------------------------------------------------------- neutral reference

_NEUTRAL = {}


def _cap_frames(J, P):
    hl, hr, sl, sr = (P[:, J[k]] for k in ("LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm"))
    hm, sm = (hl + hr) / 2, (sl + sr) / 2
    return dict(pel=_frame(hl - hr, sm - hm), tho=_frame(sl - sr, sm - hm)), hm, sm


def _segments():
    out = list(PROFILE["chain"].values())
    for leg in PROFILE["legs"].values():
        c = leg["cap"]
        out += [(c[0], c[1], leg["frame"]), (c[1], c[2], leg["frame"]), (c[2], c[3], leg["frame"])]
    return out


def neutral():
    """Body-frame segment directions in the dog's neutral stand (cached)."""
    if "ref" in _NEUTRAL:
        return _NEUTRAL["ref"]
    names, P, fps = _world(_bvh_path(NEUTRAL_CLIP))
    J = {k: j for j, k in enumerate(names)}
    F, hm, sm = _cap_frames(J, P)
    sp = np.linalg.norm(np.gradient(hm[:, :2], axis=0), axis=1) * fps
    ground = np.percentile(P[:, [J[k] for k in PAWS], 2].min(1), 5)
    idx = np.nonzero((sp < 0.05) & (hm[:, 2] - ground > 0.30))[0]
    ref = {}
    for a, b, fr in set(_segments()):
        d = P[idx, J[b]] - P[idx, J[a]]
        d = np.einsum("nji,nj->ni", F[fr][idx], d)
        d = np.median(d / np.linalg.norm(d, axis=1, keepdims=True), axis=0)
        ref[(a, b, fr)] = d / np.linalg.norm(d)
    for leg, (g, p), fr in (("FL", ("LeftArm", "LeftHand_end"), "tho"), ("FR", ("RightArm", "RightHand_end"), "tho"),
                            ("HL", ("LeftUpLeg", "LeftFoot_end"), "pel"), ("HR", ("RightUpLeg", "RightFoot_end"), "pel")):
        ref[("paw", leg)] = np.median(np.einsum("nji,nj->ni", F[fr][idx], P[idx, J[p]] - P[idx, J[g]]), axis=0)
    ref["hip_h"] = float(np.median(hm[idx, 2]) - ground)
    ref["sh_h"] = float(np.median(sm[idx, 2]) - ground)
    _NEUTRAL["ref"] = ref
    return ref


# ---------------------------------------------------------------- rig side

class _Rig:
    def __init__(self, arm, prof):
        self.arm, self.p = arm, prof
        self.bones = arm.data.bones
        self.order = []
        def walk(b):
            self.order.append(b.name)
            for c in b.children: walk(c)
        for b in self.bones:
            if b.parent is None: walk(b)
        self.rest = {n: np.array(self.bones[n].matrix_local) for n in self.order}

    def head(self, n): return self.rest[n][:3, 3]
    def tail(self, n): return np.array(self.bones[n].tail_local)

    def frames(self):
        p = self.p
        hl, hr = self.head(p["hips"][0]), self.head(p["hips"][1])
        sl, sr = self.head(p["shoulders"][0]), self.head(p["shoulders"][1])
        hm, sm = (hl + hr) / 2, (sl + sr) / 2
        return dict(pel=_frame(hl - hr, sm - hm), tho=_frame(sl - sr, sm - hm)), hm, sm


def _contacts(z, xy, fps, h=0.03, v=0.35, body_xy=None):
    """Runs of frames where the captured paw is down: near its lowest and nearly still.
    "Still" is relative to the body: a trotting dog's paw end rolls over its pad at 0.3-0.4
    m/s, which a fixed 0.35 m/s finds in only one or two frames of each stance, so a trot
    would have no stance at all and its paws would ride up the whole way. With the body's
    speed given, a paw moving under a quarter of it also counts."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    n = len(z); w = 2 * fps
    floor = sw(np.pad(z, (w // 2, w - 1 - w // 2), mode="edge"), w).min(1)
    sp = np.linalg.norm(np.gradient(xy, axis=0), axis=1) * fps
    if body_xy is not None:
        body = np.linalg.norm(np.gradient(body_xy, axis=0), axis=1) * fps
        v = np.maximum(v, 0.25 * body)
    m = ((z - floor) < h) & (sp < v)
    for k in range(1, n - 1):
        if not m[k] and m[k - 1] and m[k + 1]: m[k] = True
    runs = []; k = 0
    while k < n:
        if m[k]:
            j = k
            while j < n and m[j]: j += 1
            if j - k >= 3: runs.append((k, j))
            k = j
        else:
            k += 1
    return runs


def _merge_no_step(runs, z, xy, floor):
    """A "swing" in which the paw never leaves the ground is not a step. Splicing a walk into
    a stand cross-fades the two clips' paw positions, and a sniffing dog shuffles: the paw
    moves while barely lifting, the speed test ends the stance, and all four paws would glide
    to new footings together (tens of centimeters as the dog stops). Stances joined by a
    lift under NO_STEP_LIFT_M are one stance (the paw stays
    put and the body moves over it) -- unless they stand on different ground (a tread and the
    one below it), where the capture's flat walk cannot show the lift a step needs."""
    merged = []
    for a, b in runs:
        if merged:
            a0, b0 = merged[-1]
            base = max(z[b0 - 1], z[a])
            # ...nor where the paw travels further than a shuffle: held there, the body walks on
            # past the paw's reach and drags it (a turn on the spot)
            p0, p1 = xy[(a0 + b0 - 1) // 2], xy[(a + b - 1) // 2]
            same = float(np.linalg.norm(p1 - p0)) < NO_STEP_MOVE_M
            if same and floor is not None:
                same = abs(float(_floor_at(floor, p0[0], p0[1])) - float(_floor_at(floor, p1[0], p1[1]))) < 0.02
            if same and (b0 >= a or float(np.max(z[b0:a])) - base < NO_STEP_LIFT_M):
                merged[-1] = (a0, b); continue
        merged.append((a, b))
    return merged


def _stride_ratios(arm, prof):
    """Per leg, the rig's girdle-to-paw length over the captured dog's (scaled by hip
    height) in the neutral stand; and their mean, for the body's travel."""
    ref = neutral()
    B = arm.data.bones
    h = lambda n: np.array(B[n].head_local)
    rhm = (h(prof["hips"][0]) + h(prof["hips"][1])) / 2
    s = rhm[2] / ref["hip_h"]
    out = {}
    for k, v in prof["legs"].items():
        rig = np.linalg.norm(h(v["ctrl"]) - h(v["bones"][0]))
        cap = np.linalg.norm(ref[("paw", k)]) * s
        out[k] = float(min(max(rig / cap, 0.5), 1.2))
    return out, float(np.mean(list(out.values())))


def _tilt_about_hips(pose_f, prof, rig, angle, down=True):
    """Turn everything under the pelvis by `angle` about the hips' lateral axis,
    lowering the shoulders (`down`) or raising them."""
    hp = prof["hips"]
    hm = (pose_f[hp[0]][:3, 3] + pose_f[hp[1]][:3, 3]) / 2
    ax = pose_f[hp[0]][:3, 3] - pose_f[hp[1]][:3, 3]; ax[2] = 0
    if np.linalg.norm(ax) < 1e-9: return
    ax /= np.linalg.norm(ax)
    sh = prof["shoulders"]
    sm = (pose_f[sh[0]][:3, 3] + pose_f[sh[1]][:3, 3]) / 2
    # the sign that lowers (or raises) the shoulders
    lowers = (_axis_angle(ax, angle) @ (sm - hm))[2] < (sm - hm)[2]
    t = angle if lowers == down else -angle
    T = np.eye(4); T[:3, :3] = _axis_angle(ax, t); T[:3, 3] = hm - T[:3, :3] @ hm
    for nm in rig.below_pelvis:
        pose_f[nm] = T @ pose_f[nm]


# ---------------------------------------------------------------- skin measurements

def _skins(arm):
    import bpy
    return [o for o in bpy.data.objects if o.type == "MESH"
            and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]


def _descendants(arm, names):
    out = set()
    def add(b):
        out.add(b.name)
        for c in b.children: add(c)
    for n in names:
        add(arm.data.bones[n])
    return out


def _skin_masks(arm, prof):
    """Per skinned mesh: vertices of each leg below its top bone, of each paw, and of
    the trunk's front and rear halves (split at rest along hips -> shoulders)."""
    legs = {k: _descendants(arm, v["bones"][1:] + (v["ctrl"],)) for k, v in prof["legs"].items()}
    tail = _descendants(arm, prof["absolute"]) if prof["absolute"] else set()
    skip = set().union(*legs.values()) | tail
    B = arm.data.bones; Mw = np.array(arm.matrix_world)
    head = lambda n: (Mw @ np.append(np.array(B[n].head_local), 1.0))[:3]
    hm = (head(prof["hips"][0]) + head(prof["hips"][1])) / 2
    sm = (head(prof["shoulders"][0]) + head(prof["shoulders"][1])) / 2
    ax = sm - hm; ax2 = float(ax @ ax)
    out = {}
    for o in _skins(arm):
        names = {g.index: g.name for g in o.vertex_groups}
        dom = np.array([names.get(max(v.groups, key=lambda g: g.weight).group, "") if v.groups else ""
                        for v in o.data.vertices])
        co = np.array([v.co for v in o.data.vertices]) @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3]
        t = (co - hm) @ ax / ax2
        keep = ~np.isin(dom, list(skip))
        m = {k: np.isin(dom, list(b)) for k, b in legs.items()}
        m["front"] = keep & (t > 0.5); m["rear"] = keep & (t <= 0.5); m["all"] = np.ones(len(dom), bool)
        out[o.name] = m
    return out


def _lowest(arm, n, masks, floor=None):
    """Lowest height above the floor (armature space; `floor(x, y)`, else z = 0) per frame
    of each masked vertex set (frames 1..n)."""
    import bpy
    sc = bpy.context.scene
    keys = next(iter(masks.values())).keys() if masks else ()
    out = {k: np.full(n, 1e9) for k in keys}
    for f in range(n):
        sc.frame_set(f + 1); dg = bpy.context.evaluated_depsgraph_get()
        to_arm = np.linalg.inv(np.array(arm.matrix_world))   # read per frame: a parent may move
        for o in _skins(arm):
            ev = o.evaluated_get(dg); me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
            M = to_arm @ np.array(o.matrix_world)          # the floor is the armature's z = 0
            xyz = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            z = xyz[:, 2] - (_floor_at(floor, xyz[:, 0], xyz[:, 1]) if floor is not None else 0.0)
            for k, m in masks[o.name].items():
                if len(m) == len(z) and m.any():
                    out[k][f] = min(out[k][f], z[m].min())
            ev.to_mesh_clear()
    return out


# ---------------------------------------------------------------- retarget

def retarget(armature, source, *, start_s=None, end_s=None, name=None, heading_deg=0.0, roles=None,
             origin=(0.0, 0.0), ground=True, foot_lock=True, head_pitch_deg=0.0, head_yaw_deg=0.0,
             floor=None, legs="ik", smooth_s=SMOOTH_S):
    """Bake a dog capture (a `Clip`, or a capture name with `start_s`/`end_s`) onto
    `armature`. Returns a report; the motion is a new Action keyed at the scene's fps
    on frames 1..N, left *unassigned* (the armature's own action is restored).

    An unplaced clip starts at the dog's rest position facing the dog's own front;
    `heading_deg` turns it about world up from there and `origin` shifts its start,
    both in the armature's space. A placed clip (`Clip.follow`) keeps its coordinates.
    `ground` keeps the skin off the floor (z = 0 in the armature's space): body skin more
    than BODY_SINK_M into it raises the whole dog, a swinging paw's skin in it lifts that
    paw over its swing, and a tail on it keeps its own clearance.
    `foot_lock` pins each paw for each contact the capture shows.
    `legs` is "ik", the only leg solve (each planted paw pinned, each leg re-solved at its
    own bone lengths); it is kept as a keyword so existing calls still read.
    `head_pitch_deg` (positive lowers the nose) and `head_yaw_deg` (positive turns it to the
    dog's left) turn the neck and head on top of the capture's, spread over the neck so it
    bends rather than kinks: the captured dog often looks up at its handler, which a dog
    fetching or sniffing should not. Each is a number, one value per output frame, or a
    function of the time in seconds.
    `floor` is the ground the dog walks on, in the armature's space: a function
    `floor(x, y) -> z` taking arrays (`floor_from_objects` builds one from porch, steps and
    path meshes), or None for the flat z = 0. Each planted paw stands on it, a swing climbs
    over a step it crosses, the body rises and pitches with the paws' footing, and `ground`
    keeps the skin above it rather than above z = 0."""
    # Only "ik" (each planted paw pinned) is accepted: copying the capture's joint angles with
    # no paw pinned lets paws slide and float wherever the rig's proportions differ.
    if legs != "ik":
        raise ValueError(f"legs={legs!r}: only 'ik' is supported, which pins each planted paw")
    import bpy
    # A caller that just moved the dog (or its parent) has not re-evaluated the scene,
    # and every placement below is read from it: a stale world matrix puts the floor
    # and the paw targets centimeters off.
    bpy.context.view_layer.update()
    c = source if isinstance(source, Clip) else clip(source, start_s, end_s)
    if isinstance(source, Clip) and (start_s is not None or end_s is not None):
        c = c.section(start_s, end_s)
    prof = profile(armature, roles=roles)
    name = name or f"dog_{c.source}"[:60]
    # The captured dog's legs are longer for its hip height than this rig's: its strides,
    # put on the rig unscaled, overreach every touchdown and lift-off. Scale each paw's
    # reach by its leg's length ratio and the body's travel by their mean, so planted
    # paws and the moving body still agree (a followed path is re-fitted afterwards).
    paw_scale, travel_scale = _stride_ratios(armature, prof)
    if c.placed and c.base is not None:
        c = c.base.stride(travel_scale).follow(*c.path)
    elif not c.placed:
        c = c.stride(travel_scale)
    # The capture jitters frame to frame, most at the paws and the head: smoothed here, once,
    # so every caller gets it (and Clip.smooth on top only adds to it).
    if smooth_s and smooth_s > 0:
        c = c.smooth(smooth_s * c.fps)
    prev = armature.animation_data.action if armature.animation_data else None
    kw = dict(heading_deg=heading_deg, origin=origin, foot_lock=foot_lock, name=name, paw_scale=paw_scale,
              head=(head_pitch_deg, head_yaw_deg), floor=floor)
    rep = _solve(armature, prof, c, **kw)
    if ground:
        fps = rep["fps"]; n = rep["frames"]; w = max(3, int(fps) // 3) | 1
        masks = _skin_masks(armature, prof)
        lift = np.zeros(n); leg_lift = {k: np.zeros(n) for k in prof["legs"]}
        dil = lambda x: _box(_dilate(x, w), w)
        passes = 0
        worst = None
        for passes in range(1, 6):
            low = _lowest(armature, n, masks, floor)
            pen = {k: np.maximum(0.0, -low[k] + 0.002) for k in low}
            excess = max(np.maximum(pen["front"], pen["rear"]).max() - BODY_SINK_M,
                         max(pen[k].max() for k in prof["legs"]) - 0.004)
            if excess < 0.006:
                break
            # A pass that did not reduce the penetration is fighting another constraint
            # (planted paws pulling the body back down): stop rather than pile lift on lift.
            if worst is not None and excess > worst - 0.003:
                break
            worst = excess
            # body skin in the floor raises the whole dog; a paw's skin in it (a swinging
            # paw's tilted pad or claws, below its toe joint) lifts that paw over its swing
            body = np.maximum(pen["front"], pen["rear"])
            lift += dil(np.maximum(body - BODY_SINK_M, 0.0))
            for k in leg_lift: leg_lift[k] += dil(pen[k])
            bpy.data.actions.remove(bpy.data.actions[rep["action"]])
            rep = _solve(armature, prof, c, lift=lift, leg_lift=leg_lift, **kw)
        rep["lift_max_m"] = round(float(lift.max()), 4)
        rep["ground_passes"] = passes
        rep["below_ground_max_m"] = round(float(max(0.0, -_lowest(armature, n, masks, floor)["all"].min())), 4)
    armature.animation_data.action = prev
    return rep


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _floor_at(floor, x, y):
    """The floor's height under (x, y) (arrays or numbers); 0 for a flat floor."""
    if floor is None:
        return np.zeros(np.shape(x)) if np.ndim(x) else 0.0
    z = floor(np.asarray(x, float), np.asarray(y, float))
    return float(z) if np.ndim(z) == 0 else np.asarray(z, float)


def floor_from_objects(objects, armature, cell=0.01, margin=0.5, bridge=0.04):
    """A `floor(x, y)` for `retarget`, in `armature`'s space, from the tops of mesh objects
    (porch, steps, path ...): heights sampled down a grid of `cell` meters over their
    footprint plus `margin`, bilinear between samples, 0 outside. Gaps narrower than
    `bridge` (between deck boards) are bridged: a ray down a board gap would pull the porch
    below its boards, and every paw and belly on it into them."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    to_arm = np.linalg.inv(np.array(armature.matrix_world))
    trees, pts = [], []
    for o in objects:
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        M = to_arm @ np.array(o.matrix_world)
        vs = [tuple(M[:3, :3] @ np.array(v.co) + M[:3, 3]) for v in me.vertices]
        trees.append(BVHTree.FromPolygons(vs, [tuple(p.vertices) for p in me.polygons]))
        pts += vs; ev.to_mesh_clear()
    if not pts:
        return None
    P = np.array(pts); lo = P[:, :2].min(0) - margin; hi = P[:, :2].max(0) + margin
    top = float(P[:, 2].max()) + 1.0
    xs = np.arange(lo[0], hi[0] + cell, cell); ys = np.arange(lo[1], hi[1] + cell, cell)
    H = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            best = 0.0
            for t in trees:
                hit = t.ray_cast(Vector((x, y, top)), Vector((0, 0, -1)))
                if hit[0] is not None:
                    best = max(best, hit[0].z)
            H[i, j] = best
    r = max(0, int(round(bridge / cell / 2)))
    if r:
        P_ = np.pad(H, r, mode="edge"); G = H.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                G = np.maximum(G, P_[r + di:r + di + H.shape[0], r + dj:r + dj + H.shape[1]])
        H = G
    def floor(x, y):
        fx = np.clip((np.asarray(x) - xs[0]) / cell, 0, len(xs) - 1.001)
        fy = np.clip((np.asarray(y) - ys[0]) / cell, 0, len(ys) - 1.001)
        i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int); u = fx - i0; v = fy - j0
        z = (H[i0, j0] * (1 - u) * (1 - v) + H[i0 + 1, j0] * u * (1 - v)
             + H[i0, j0 + 1] * (1 - u) * v + H[i0 + 1, j0 + 1] * u * v)
        inside = ((np.asarray(x) >= xs[0]) & (np.asarray(x) <= xs[-1])
                  & (np.asarray(y) >= ys[0]) & (np.asarray(y) <= ys[-1]))
        return np.where(inside, z, 0.0)
    return floor


def _clear_of_risers(floor, xy, reach=EDGE_CLEARANCE_M, most=0.10, step=0.01):
    """A footing moved off the foot of a step: while higher ground lies within `reach`
    (a riser in front of the paw), step away from it, at most `most` meters. A paw planted
    against a riser puts its leg through the step's edge."""
    xy = np.asarray(xy, float).copy()
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    moved = 0.0
    while moved < most:
        here = float(_floor_at(floor, xy[0], xy[1]))
        pts = xy + ring * reach
        h = _floor_at(floor, pts[:, 0], pts[:, 1]) - here
        if h.max() < 0.02:
            break
        away = -(ring * np.maximum(h, 0.0)[:, None]).sum(0)
        nrm = np.linalg.norm(away)
        if nrm < 1e-9:
            break
        xy += away / nrm * step; moved += step
    return xy


def _per_frame(x, n, fps):
    """A number, a sequence of `n` values, or a function of time (s) -> an array of `n` values."""
    if callable(x):
        return np.array([float(x(f / fps)) for f in range(n)])
    a = np.asarray(x, float)
    if a.ndim == 0:
        return np.full(n, float(a))
    if len(a) != n:
        raise ValueError(f"head angles: {len(a)} values for a {n}-frame clip")
    return a


def _whole_body_height(pose, prof, rig, n, fps, rest_z, floor, lift, rsm, rhm, contact_runs=None):
    """The capture's joint angles on every bone, and the whole dog moved as
    one body -- pitched about the hips with the ground under its front and hind paws, then
    raised or lowered so its lowest paw stands on the ground under it. Both eased over a
    quarter second; nothing is solved per leg, so nothing can flip from one frame to the
    next. A paw over a lower tread, or one the other dog's proportions leave short, is let
    hang: it never drags the rest of the body after it; the leg solve in `_solve` reaches it."""
    legs = prof["legs"]
    w = max(3, int(round(fps)) // 4) | 1
    paws = lambda: {k: np.array([pose[f][v["ctrl"]][:3, 3] for f in range(n)]) for k, v in legs.items()}
    under = lambda p: {k: (_floor_at(floor, q[:, 0], q[:, 1]) if floor is not None else np.zeros(n)) for k, q in p.items()}
    if floor is not None:
        p = paws(); g = under(p)
        front = _gauss(np.mean([g[k] for k, v in legs.items() if v["frame"] == "tho"], 0), fps / 4)
        rear = _gauss(np.mean([g[k] for k, v in legs.items() if v["frame"] == "pel"], 0), fps / 4)
        span = float(np.linalg.norm((rsm - rhm)[:2]))
        for f in range(n):
            d = front[f] - rear[f]
            if abs(d) > 1e-4 and span > 1e-6:
                _tilt_about_hips(pose[f], prof, rig, math.asin(min(abs(d) / span, 0.6)), down=d < 0)
    # A rig's forelegs can be shorter for its hips than the captured dog's: with the angles
    # alone and the body set on its hind paws, the forepaws hang centimeters up.
    # Pitch the body about the hips until the lower forepaw is as near the ground as the
    # lower hind paw -- one rigid turn per frame, eased like the height.
    # (Iterated: the turn moves the hind paws too, more in a sit, where they are under the
    # body, so one turn by the gap over the trunk's length closed only part of it.)
    span = float(np.linalg.norm((rsm - rhm)[:2]))
    for _ in range(4 if span > 1e-6 else 0):
        p = paws(); g = under(p)
        gap = lambda side: np.min([p[k][:, 2] - rest_z[k] - g[k] for k, v in legs.items() if v["frame"] == side], axis=0)
        d = _gauss(gap("tho") - gap("pel"), fps / 4)
        if np.abs(d).max() < 0.003:
            break
        for f in range(n):
            if abs(d[f]) > 1e-4:
                _tilt_about_hips(pose[f], prof, rig, math.asin(min(abs(d[f]) / span, 0.6)), down=d[f] > 0)
    p = paws(); g = under(p)
    gaps = {k: p[k][:, 2] - rest_z[k] - g[k] for k in legs}
    low = np.min([gaps[k] for k in legs], axis=0)
    # eased through a running maximum: where it cannot be exact it errs on the high side,
    # a paw a few millimeters up rather than in the floor
    dz = _box(_dilate(-low, w), w)
    if contact_runs:
        # Set by the paws the capture has planted, not by the lowest paw: walking, the lowest
        # is often the one swinging through, and a body set on it stands so high that the
        # planted paws cannot reach the ground (they slide and float). Low enough that the
        # highest planted paw reaches; the leg solve folds the others down.
        # On steps the planted paws stand on different treads: as low as the highest-hanging one
        # needs puts the shoulders below the tread a forepaw still stands on, folding the leg
        # onto its girdle and turning the elbow over. Halfway between that and
        # the planted paws' mean, the legs share the difference; on flat ground the two agree.
        hi = np.full(n, -np.inf); sm = np.zeros(n); cnt = np.zeros(n)
        for k in legs:
            for a, b in contact_runs.get(k, ()):
                hi[a:b] = np.maximum(hi[a:b], gaps[k][a:b]); sm[a:b] += gaps[k][a:b]; cnt[a:b] += 1
        need = np.where(cnt > 0, 0.5 * (hi + sm / np.maximum(cnt, 1)), np.nan)
        ok = ~np.isnan(need)
        if ok.any():
            idx = np.arange(n)
            need = np.interp(idx, idx[ok], need[ok])
            dz = -_box(_dilate(need, w), w)
    if lift is not None:
        dz = dz + lift
    for f in range(n):
        for nm in pose[f]:
            M = pose[f][nm].copy(); M[2, 3] += dz[f]; pose[f][nm] = M


#: Below this world pitch of the capture's neck (degrees), the rig's neck turns from its own
#: carriage toward the capture's direction, fully by `NECK_REACH_SPAN` lower. The solve
#: carries each neck bone by the capture's change from its neutral stand, which keeps a rig's
#: rest carriage. Walking, that reads as the rig's own bearing; but a rig that carries its neck
#: higher than the capture's dog stops short of a sniff, its muzzle hanging well above the floor.
#: A walk, trot, gallop, sit, or lie keeps its neck above level and is not changed.
NECK_REACH_FROM = 0.0
NECK_REACH_SPAN = 45.0


def _neck_reach(pose, prof, rig, arm, P, J):
    """Turn the neck about its base toward the capture's neck direction where the capture's
    neck points below `NECK_REACH_FROM`, keeping the head's own world direction."""
    neck = [nm for nm in rig.order if prof["chain"].get(nm, ("",))[:2] == ("Neck", "Head")]
    headb = next((nm for nm in rig.order if prof["chain"].get(nm, ("",))[:2] == ("Head", "Head_end")), None)
    if not neck or headb is None:
        return
    cap = P[:, J["Head"]] - P[:, J["Neck"]]
    cap_pitch = np.degrees(np.arctan2(cap[:, 2], np.hypot(cap[:, 0], cap[:, 1])))
    u = np.clip((NECK_REACH_FROM - cap_pitch) / NECK_REACH_SPAN, 0.0, 1.0)
    w = u * u * (3 - 2 * u)
    below = [b for b in rig.order if b in _descendants(arm, [neck[0]])]
    head_part = [b for b in rig.order if b in _descendants(arm, [headb])]
    for f in np.flatnonzero(w > 1e-4):
        # The neck as its joints run (the base bone's own axis may point elsewhere).
        base, tip = pose[f][neck[0]][:3, 3].copy(), pose[f][headb][:3, 3].copy()
        v = tip - base
        lat = np.cross(v, [0.0, 0.0, 1.0])
        if np.linalg.norm(lat) < 1e-9:
            continue
        lat /= np.linalg.norm(lat)
        turn = math.radians(cap_pitch[f] - math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))) * float(w[f])
        R = _axis_angle(lat, turn)
        if ((R @ v)[2] - v[2]) * turn < 0:                 # turn the way that changes the pitch by `turn`
            R = _axis_angle(lat, -turn)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = base - R @ base
        for b in below:
            pose[f][b] = T @ pose[f][b]
        hp = pose[f][headb][:3, 3].copy()
        Tc = np.eye(4); Tc[:3, :3] = R.T; Tc[:3, 3] = hp - R.T @ hp
        for b in head_part:                                 # the head carried, not turned
            pose[f][b] = Tc @ pose[f][b]


def _solve(arm, prof, c, *, heading_deg, origin, foot_lock, name, paw_scale, head=(0.0, 0.0),
           floor=None, lift=None, leg_lift=None):
    import bpy
    from mathutils import Matrix
    rig = _Rig(arm, prof)
    rig.below_pelvis = [n for n in rig.order if n in _descendants(arm, [prof["pelvis"]])]
    ref = neutral()
    J = c.J
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    # resample to the scene's fps
    t_src = np.arange(len(c.P)) / c.fps
    tq = np.arange(0, t_src[-1] + 1e-9, 1 / fps)
    P = np.empty((len(tq),) + c.P.shape[1:])
    for j in range(P.shape[1]):
        for k in range(3):
            P[:, j, k] = np.interp(tq, t_src, c.P[:, j, k])
    n = len(P)
    RF, rhm, rsm = rig.frames()
    s = rhm[2] / ref["hip_h"]
    # The capture is scaled by hip height, but its shoulders stand higher above its hips
    # than this rig's; left alone, the front legs fall short every frame. Tilt the trunk
    # forward about the hips, once for the whole clip, so the neutral shoulder height is
    # the rig's own.
    over = s * ref["sh_h"] - rsm[2]
    span = float(np.linalg.norm((rsm - rhm)[:2]))
    shoulder_tilt = math.asin(min(max(over / span, 0.0), 0.5)) if span > 1e-6 else 0.0
    # Everything below is in the armature's own space, whose z = 0 is the floor the
    # rest pose stands on. The capture is scaled by hip height about its hips.
    if c.placed:
        hipxy = c._hips(P)[:, None, :2]
        P[..., :2] = (P[..., :2] - hipxy) * s + hipxy
        P[..., 2] *= s
    else:
        F0, hm0, sm0 = _cap_frames(J, P[:1])
        f0 = (sm0 - hm0)[0, :2]
        rf = (rsm - rhm)[:2]
        yaw = math.atan2(rf[1], rf[0]) + math.radians(heading_deg) - math.atan2(f0[1], f0[0])
        P = P - np.array([hm0[0, 0], hm0[0, 1], 0])
        P = np.einsum("ij,nkj->nki", _rz(yaw), P) * s
        P[..., 0] += rhm[0] + origin[0]; P[..., 1] += rhm[1] + origin[1]

    F, hm, sm = _cap_frames(J, P)
    D = {fr: np.einsum("nij,njk->nik", F[fr], np.linalg.inv(RF[fr])[None]) for fr in ("pel", "tho")}
    absolute = set(prof["absolute"])
    rot = {}

    def seg_rot(bone, a_, b_, fr):
        R0 = rig.rest[bone][:3, :3]
        dref = (np.einsum("nij,j->ni", D[fr], R0[:, 1]) if bone in absolute
                else F[fr] @ ref[(a_, b_, fr)])
        dnow = P[:, J[b_]] - P[:, J[a_]]
        return np.stack([_swing(dref[k], dnow[k]) @ D[fr][k] @ R0 for k in range(n)])

    pel = prof["pelvis"]
    rot[pel] = np.einsum("nij,jk->nik", D["pel"], rig.rest[pel][:3, :3])
    for bone, (a_, b_, fr) in prof["chain"].items():
        rot[bone] = seg_rot(bone, a_, b_, fr)
    for leg in prof["legs"].values():
        cp = leg["cap"]
        for bone, (a_, b_) in zip(leg["bones"], zip(cp[:-1], cp[1:])):
            rot[bone] = seg_rot(bone, a_, b_, leg["frame"])

    hip_off = rig.rest[pel][:3, :3].T @ (rhm - rig.head(pel))
    pel_head = hm - np.einsum("nij,j->ni", rot[pel], hip_off)

    rest_z = {k: rig.head(v["ctrl"])[2] for k, v in prof["legs"].items()}
    targets = {}; fold = {}
    for k, v in prof["legs"].items():
        cp = v["cap"]; fr = v["frame"]
        neutral_paw = P[:, J[cp[0]]] + np.einsum("nij,j->ni", F[fr], ref[("paw", k)] * s)
        rig_off = D[fr] @ (rig.head(v["ctrl"]) - rig.head(v["bones"][0]))
        targets[k] = rig_off + (P[:, J[cp[3]]] - neutral_paw) * paw_scale[k]
        # A leg folded well past its neutral stand (lying, sitting) keeps the capture's
        # fraction of its own length: a rig's legs can be a fifth shorter than the captured
        # dog's (the library shepherd's upper arm a third shorter), and the offset model above,
        # right for a standing or walking leg, would set a lying paw at full reach -- the
        # foreleg straight, the elbow off the floor.
        cap_vec = P[:, J[cp[3]]] - P[:, J[cp[0]]]
        cap_len = float(np.median(sum(np.linalg.norm(P[:, J[b_]] - P[:, J[a_]], axis=1)
                                      for a_, b_ in zip(cp[:-1], cp[1:]))))
        rig_len = sum(float(np.linalg.norm(rig.head(b_) - rig.head(a_)))
                      for a_, b_ in zip(v["bones"] + (v["ctrl"],), v["bones"][1:] + (v["ctrl"],)) if a_ != b_)
        ext = np.linalg.norm(cap_vec, axis=1) / max(cap_len, 1e-9)
        ext_neutral = float(np.linalg.norm(ref[("paw", k)]) * s / max(cap_len, 1e-9))
        # eased in time too: switching models within a frame or two moves a sitting dog's
        # paw targets centimeters per frame and flips the stifle
        # ...and so does every leg of a dog whose shoulders are down (lying), folded or not: a
        # foreleg stretched straight out in front, as far from the shoulder as a standing one,
        # can be longer in the capture than the rig's leg can reach
        lying = np.clip((LIE_SHOULDER_FRAC * ref["sh_h"] * s - sm[:, 2]) / (0.2 * ref["sh_h"] * s), 0.0, 1.0)
        w = _gauss(np.maximum(np.clip((ext_neutral - ext) / FOLD_BLEND, 0.0, 1.0), lying)[:, None], fps / 5)
        # Forelegs only. A sitting dog's hind paw, set this way, sits right under the hip and
        # the stifle swings from beside the leg to inside it in a frame; the hind
        # legs' offset model already folds them properly (`test_a_sit_keeps_all_four_paws...`).
        if v["frame"] != "tho":
            w = np.zeros_like(w)
        targets[k] = (1 - w) * targets[k] + w * cap_vec * (rig_len / max(cap_len, 1e-9))
        fold[k] = w[:, 0]

    def rel(nm):
        par = rig.bones[nm].parent
        return rig.rest[nm] if par is None else np.linalg.inv(rig.rest[par.name]) @ rig.rest[nm]

    pose = []
    paw_abs = {k: np.empty((n, 3)) for k in prof["legs"]}
    for f in range(n):
        pf = {}
        for nm in rig.order:
            par = rig.bones[nm].parent
            M = rel(nm) if par is None else pf[par.name] @ rel(nm)
            M = M.copy()
            if nm in rot:
                M[:3, :3] = rot[nm][f]
            if nm == pel:
                M[:3, 3] = pel_head[f]
            pf[nm] = M
        if shoulder_tilt > 1e-4:
            _tilt_about_hips(pf, prof, rig, shoulder_tilt)
        for k, v in prof["legs"].items():
            paw_abs[k][f] = pf[v["bones"][0]][:3, 3] + targets[k][f]
        pose.append(pf)

    contact_runs = {}
    for k, v in prof["legs"].items():
        cp = v["cap"]
        runs = _contacts(P[:, J[cp[3]], 2], P[:, J[cp[3]], :2], int(round(fps)),
                         body_xy=(P[:, J["LeftUpLeg"], :2] + P[:, J["RightUpLeg"], :2]) / 2)
        runs = _merge_no_step(runs, P[:, J[cp[3]], 2], paw_abs[k][:, :2], floor)
        contact_runs[k] = runs
        t = paw_abs[k]
        t[:, 2] = np.maximum(t[:, 2], rest_z[k])
        if foot_lock and runs:
            # Each stance holds one floor point; each swing travels from the point it left
            # to the point it lands on (smoothstep), at the capture's own swing height.
            # Easing a held paw back to the capture's track over a few frames instead jumps it
            # (the stride scaling has moved the track away), and the legs flip with it.
            locks = []
            for r0, r1 in runs:
                p = t[(r0 + r1) // 2].copy()
                if floor is not None:
                    p[:2] = _clear_of_risers(floor, p[:2])
                p[2] = rest_z[k] + _floor_at(floor, p[0], p[1]) + STANCE_CLEARANCE_M
                locks.append(p)
            free = t.copy()
            for (r0, r1), p in zip(runs, locks):
                t[r0:r1] = p
            for i in range(len(runs) - 1):
                e, s_ = runs[i][1], runs[i + 1][0]
                # the capture's own swing path, carried by an offset that eases from where the
                # paw was held to where it will be held: continuous at both ends, and the right
                # shape however long the swing (a gallop's flight, a pause between steps)
                off_e = locks[i][:2] - free[e - 1, :2]
                off_s = locks[i + 1][:2] - free[s_, :2]
                ze, zs = locks[i][2] - rest_z[k], locks[i + 1][2] - rest_z[k]
                fs = np.arange(e, s_)
                us = (fs - e + 1) / (s_ - e + 1); ws = us * us * (3 - 2 * us)
                xys = free[e:s_, :2] + (1 - ws)[:, None] * off_e + ws[:, None] * off_s
                # The ground under the swing, seen ahead and behind: the paw is kept above the
                # tread it is about to cross, rising to it (and leaving it) no faster than
                # SWING_RISE_M_PER_S. Clamped only to the ground straight beneath it, a paw
                # crossing a nosing jumps the step's whole height in one frame and the foreleg
                # turns over.
                under = _floor_at(floor, xys[:, 0], xys[:, 1]) if floor is not None else np.zeros(len(fs))
                reach = SWING_RISE_M_PER_S / fps * np.abs(fs[:, None] - fs[None, :])
                ahead = (np.asarray(under)[None, :] - reach).max(1)
                rz = targets[k][:, 2]
                step_len = float(np.linalg.norm(locks[i + 1][:2] - locks[i][:2]))
                for j, f in enumerate(fs):
                    u = us[j]; xy = xys[j]
                    # The capture's swing height over a base easing from the footing it left
                    # to the one it lands on; over a step, high enough to clear the edge
                    # (rising early for a step up, landing late for a step down).
                    base = ze + (zs - ze) * (_smooth(u / 0.7) if zs > ze else _smooth((u - 0.3) / 0.7))
                    if (s_ - e) > LONG_SWING_S * fps:
                        # longer than any stride (a stance the capture did not show): the
                        # paw goes over the ground beneath it, not at the far footing's height
                        base = ze + (zs - ze) * _smooth((u - 0.85) / 0.15) if zs > ze else zs + (ze - zs) * (1 - _smooth(u / 0.15))
                        base = min(base, max(ze, zs))
                    # ...but reached from the footings at either end no faster than the same rate:
                    # a paw lifting off right against a riser would go the riser's height up in
                    # one frame
                    r_ = SWING_RISE_M_PER_S / fps
                    base = max(base, min(ahead[j], ze + r_ * (j + 1), zs + r_ * (len(fs) - j)))
                    # The swing's height is how far the capture's paw rises above the line from
                    # where it lifted off to where it lands, measured on its path relative to
                    # its own girdle -- not its height above this rig's floor: a captured
                    # planted paw can map several centimeters above it (other proportions),
                    # and that offset would make every lift-off and touchdown a jump of a few
                    # centimeters, the foreleg turning tens of degrees in a frame; nor in the
                    # world, where on steps the trunk's rise with the footing would come back as arc.
                    arc = max(rz[f] - ((1 - u) * rz[e - 1] + u * rz[s_]), 0.0)
                    # a paw carried more than a few centimeters is lifted clear of the ground, at
                    # least MIN_SWING_LIFT_M: a sniffing walk's paws, captured barely off the
                    # floor, would otherwise slide along it
                    if step_len > 0.05:
                        arc = max(arc, MIN_SWING_LIFT_M * min(1.0, step_len / 0.15) * math.sin(math.pi * u))
                    clear = STEP_CLEARANCE * abs(zs - ze) * math.sin(math.pi * u)
                    if abs(zs - ze) > 0.02:
                        arc = max(arc, STEP_SWING_LIFT_M * math.sin(math.pi * u))
                    t[f] = (xy[0], xy[1], rest_z[k] + base + arc + clear)
            # before the first stance and after the last, the capture's track, shifted to meet them
            s0, e1 = runs[0][0], runs[-1][1]
            if s0 > 0:
                t[:s0, :2] = free[:s0, :2] + (locks[0][:2] - free[s0, :2])
                t[:s0, 2] = np.maximum(free[:s0, 2] + (locks[0][2] - free[s0, 2]), locks[0][2])
            if e1 < n:
                t[e1:, :2] = free[e1:, :2] + (locks[-1][:2] - free[e1 - 1, :2])
                t[e1:, 2] = np.maximum(free[e1:, 2] + (locks[-1][2] - free[e1 - 1, 2]), locks[-1][2])
        elif floor is not None:
            t[:, 2] += _floor_at(floor, t[:, 0], t[:, 1])

    _neck_reach(pose, prof, rig, arm, P, J)
    pitch_d, yaw_d = (_per_frame(x, n, fps) for x in head)
    if np.any(np.abs(pitch_d) > 1e-6) or np.any(np.abs(yaw_d) > 1e-6):
        neck = [nm for nm in rig.order if nm in prof["chain"] and prof["chain"][nm][0] in ("Neck", "Head")]
        sh = prof["shoulders"]
        belows = {nm: [b for b in rig.order if b in _descendants(arm, [nm])] for nm in neck}
        for f in range(n):
            for nm in neck:
                below = belows[nm]
                piv = pose[f][nm][:3, 3].copy()
                lat = pose[f][sh[0]][:3, 3] - pose[f][sh[1]][:3, 3]; lat /= max(np.linalg.norm(lat), 1e-9)
                tip = pose[f][neck[-1]][:3, 3] + pose[f][neck[-1]][:3, :3] @ np.array([0.0, 0.1, 0.0])
                a = math.radians(pitch_d[f]) / len(neck)
                Rp = _axis_angle(lat, a)
                if (Rp @ (tip - piv))[2] > (tip - piv)[2]:          # positive pitch lowers the nose
                    Rp = _axis_angle(lat, -a)
                R = _axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(yaw_d[f]) / len(neck)) @ Rp
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = piv - R @ piv
                for b in below:
                    pose[f][b] = T @ pose[f][b]

    _whole_body_height(pose, prof, rig, n, fps, rest_z, floor, lift, rsm, rhm, contact_runs if foot_lock else None)

    act = bpy.data.actions.new(name)
    arm.animation_data_create(); arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.scale = (1, 1, 1)
    leg_bones = {b for v in prof["legs"].values() for b in v["bones"]}
    ctrls = {v["ctrl"] for v in prof["legs"].values()}
    keyed = set(rot) | leg_bones | ctrls
    prevq = {}; knee_prev = {}
    # How much of a leg's lift its paw takes: none while planted, rising and falling over each
    # swing. A step on and off at the contact edges would throw the paw several centimeters in
    # one frame at every lift-off and touchdown, the foreleg turning tens of degrees.
    swing_w = {k: np.zeros(n) for k in prof["legs"]}
    for k in prof["legs"]:
        planted = np.zeros(n, bool)
        for r0, r1 in contact_runs.get(k, ()):
            planted[r0:r1] = True
        f = 0
        while f < n:
            if planted[f]:
                f += 1; continue
            s0 = f
            while f < n and not planted[f]:
                f += 1
            s1 = f                                      # swing frames s0 .. s1-1
            u = (np.arange(s0, s1) - s0 + 1) / (s1 - s0 + 1)
            if s0 == 0 and s1 == n:
                env = np.ones(s1 - s0)
            elif s0 == 0:                               # in the air from the start: only lands
                env = np.sin(0.5 * math.pi * (s1 - np.arange(s0, s1)) / (s1 - s0 + 1))
            elif s1 == n:                               # lifts off and never lands
                env = np.sin(0.5 * math.pi * u)
            else:
                env = np.sin(math.pi * u)
            swing_w[k][s0:s1] = env
    shortfall = {k: 0.0 for k in prof["legs"]}
    for f in range(n):
        pf = pose[f]
        for k, v in prof["legs"].items():
            bs = v["bones"]; ctrl = v["ctrl"]
            tgt = paw_abs[k][f].copy()
            # a swinging leg whose skin was found in a step lifts its paw over it, not only its hock
            if leg_lift is not None:
                tgt[2] += leg_lift[k][f] * swing_w[k][f]
            root = pf[bs[0]][:3, 3]
            l1 = np.linalg.norm(rig.head(bs[1]) - rig.head(bs[0]))
            l2 = np.linalg.norm(rig.head(bs[2]) - rig.head(bs[1]))
            meta_len = np.linalg.norm(rig.head(ctrl) - rig.head(bs[2]))
            meta_dir = pf[bs[2]][:3, :3] @ (rig.rest[bs[2]][:3, :3].T @ (rig.head(ctrl) - rig.head(bs[2])) / meta_len)
            # A paw swung up near its own shoulder (hip) folds the leg flat, where the knee has
            # no side to bend to and flipped (a forepaw raised level with the shoulder onto a
            # step, 166 deg in a frame): no nearer than MIN_PAW_REACH of the leg, lowered
            # straight down, never below the ground under it.
            if True:
                rmin = MIN_PAW_REACH * (l1 + l2 + meta_len); dxy = float(np.linalg.norm((tgt - root)[:2]))
                if np.linalg.norm(tgt - root) < rmin and dxy < rmin:
                    zlow = root[2] - math.sqrt(rmin * rmin - dxy * dxy)
                    tgt[2] = max(min(tgt[2], zlow), rest_z[k] + _floor_at(floor, tgt[0], tgt[1]))
            hock = tgt - meta_dir * meta_len
            # Keep the hock (wrist) off the floor, more where this leg's skin was found below it.
            # As low as the capture holds it, down to a joint's own radius: a lying dog's
            # forearms and hocks rest on the ground, and a fixed 3 cm floor would prop the wrists
            # up so the forelegs lie straight on their toes.
            cap_joint = float(P[f, J[v["cap"][2]], 2])
            hmin = max(rest_z[k], min(rest_z[k] + 0.03, cap_joint)) \
                + (leg_lift[k][f] * (1.0 - fold[k][f]) if leg_lift else 0.0) + _floor_at(floor, tgt[0], tgt[1])
            if hock[2] < hmin:
                # No higher than an upright metatarsus reaches: past that the hock would sit
                # straight over the paw at the full height, a metatarsus 2.5x its length that
                # the reach step then swings round in one frame.
                flat = hock - tgt; flat[2] = 0; hz = min(hmin - tgt[2], meta_len * 0.98)
                hock = tgt + flat / max(np.linalg.norm(flat), 1e-9) * math.sqrt(max(meta_len ** 2 - hz ** 2, 0)) \
                    + np.array([0, 0, hz])
            # folded tighter than thigh + shin can fold: slide the hock straight back
            lo = abs(l1 - l2) * 1.03 + 0.005
            if np.linalg.norm(hock - root) < lo:
                lat = pf[prof["hips"][0]][:3, 3] - pf[prof["hips"][1]][:3, 3]; lat[2] = 0; lat /= np.linalg.norm(lat)
                back = -D["pel"][f][:, 1].copy(); back[2] = 0; back /= np.linalg.norm(back)
                r = hock - root; r = r - 0.7 * np.dot(r, lat) * lat
                bb = np.dot(r, back); cc = np.dot(r, r) - lo * lo
                d = -bb + math.sqrt(max(bb * bb - cc, 0.0))
                shift = root + r + back * max(d, 0.0) - hock
                hock = hock + shift; tgt = tgt + shift
            # out of reach with the captured metatarsus angle: turn the metatarsus too
            Rr = (l1 + l2) * 0.995
            if np.linalg.norm(hock - root) > Rr:
                dv = tgt - root; Dv = np.linalg.norm(dv)
                if 1e-6 < Dv <= Rr + meta_len:
                    u = dv / Dv; a = (Rr * Rr - meta_len * meta_len + Dv * Dv) / (2 * Dv)
                    cpt = root + u * a; rad = math.sqrt(max(Rr * Rr - a * a, 0.0))
                    # The point of the reach circle nearest the captured wrist or hock, so the
                    # metatarsus turns no further than it must and switching this on is smooth:
                    # always putting it behind the paw snaps a front foot tens of degrees in one
                    # frame (sniffing, lying). Near the axis (a straight leg) the nearest point
                    # is noise, which would flip it round the circle, so there it leans behind
                    # the paw, in the body's frame, by a few cm.
                    near = hock - cpt; near = near - np.dot(near, u) * u
                    back = -D[v["frame"]][f][:, 1]; back = back - np.dot(back, u) * u
                    lean = REACH_SIDE_LEAN_M * max(0.0, 1.0 - np.linalg.norm(near) / REACH_SIDE_NOISE_M)
                    off = near + lean * back / max(np.linalg.norm(back), 1e-9)
                    if np.linalg.norm(off) < 1e-6:
                        off = back
                    if np.linalg.norm(off) < 1e-9:
                        off = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
                    hock = cpt + off / np.linalg.norm(off) * rad
                else:
                    hock = root + dv / max(Dv, 1e-9) * Rr
            # The knee bends the anatomical way, in the body's own frame: a hind stifle
            # forward, a front elbow back. The captured knee is not the pole: a near-straight
            # leg puts it almost on the hip-hock line, where noise swings the knee from side to
            # side, the shin turning tens of degrees in one frame.
            fwd = D[v["frame"]][f][:, 1]
            down = -D[v["frame"]][f][:, 2]
            if down[2] > 0: down = -down
            # A front elbow sits behind *and below* the shoulder: back alone leaves a lying dog's
            # elbow level with the shoulder-wrist line, the foreleg straight, and the forearm
            # off the floor, and puts the pole on that line when the paw is held level in front
            # of the shoulder, so the elbow flips over it.
            pole = root + ((fwd if v["frame"] == "pel" else -fwd + FRONT_POLE_DOWN * down)) * (l1 + l2)
            # A folded leg brings the hock (wrist) almost onto the hip (shoulder), where the
            # axis between them is noise and the knee would swing round it within a frame. Lean
            # the axis toward the body's own "down" by a
            # few centimeters: no effect on an open leg, a steady axis on a folded one.
            dvec = hock - root
            axis = dvec + 0.06 * max(0.0, 1.0 - np.linalg.norm(dvec) / 0.12) * down
            hock = root + axis / max(np.linalg.norm(axis), 1e-9) * np.linalg.norm(dvec)
            knee = _two_bone(root, hock, l1, l2, pole)
            # A folded leg's elbow (stifle) rests on the floor, not in it: where it would sink
            # below KNEE_FLOOR_M (about the limb's skin radius) it turns up about the
            # girdle-wrist axis, on the side it was already on, so it never jumps across.
            kmin = rest_z[k] + KNEE_FLOOR_M + (leg_lift[k][f] if leg_lift else 0.0) + _floor_at(floor, knee[0], knee[1])
            if fold[k][f] > 0 and knee[2] < kmin:
                u = (hock - root) / max(np.linalg.norm(hock - root), 1e-9)
                c0 = root + u * np.dot(knee - root, u); r0 = knee - c0; rr = np.linalg.norm(r0)
                up = np.array([0.0, 0.0, 1.0]) - u * u[2]; nup = np.linalg.norm(up)
                if rr > 1e-6 and nup > 0.15:
                    up /= nup; sd = np.cross(u, up)
                    a0 = math.atan2(np.dot(r0, sd), np.dot(r0, up))
                    zneed = min(max((kmin - c0[2]) / (rr * up[2]), -1.0), 1.0)
                    amax = math.acos(zneed)                  # |angle| <= amax keeps the knee above kmin
                    a1 = math.copysign(min(abs(a0), amax), a0)
                    knee_up = c0 + rr * (math.cos(a1) * up + math.sin(a1) * sd)
                    knee = knee + (knee_up - knee) * float(fold[k][f])
                    knee = root + (knee - root) / max(np.linalg.norm(knee - root), 1e-9) * l1
            # Frame to frame the knee (elbow) turns about the girdle-wrist axis by at most
            # KNEE_STEP_DEG: wherever a fold leaves its side undefined the solve above can pick
            # the other side between frames (a sitting dog's stifle, a turning dog's elbow can
            # flip by more than 90 deg). Rate-limited, it swings there over a few frames.
            kp = knee_prev.get(k)
            if kp is not None:
                u = (hock - root) / max(np.linalg.norm(hock - root), 1e-9)
                c0 = root + u * np.dot(knee - root, u); rr = np.linalg.norm(knee - c0)
                e1 = pole - root; e1 = e1 - u * np.dot(e1, u); ne = np.linalg.norm(e1)
                if rr > 1e-6 and ne > 1e-6:
                    e1 /= ne; e2 = np.cross(u, e1)
                    ang = lambda q: math.atan2(np.dot(q - c0, e2), np.dot(q - c0, e1))
                    a_new, a_old = ang(knee), ang(kp)
                    step = math.radians(KNEE_STEP_DEG)
                    d = (a_new - a_old + math.pi) % (2 * math.pi) - math.pi
                    a = a_old + max(-step, min(step, d))
                    knee = c0 + rr * (math.cos(a) * e1 + math.sin(a) * e2)
            # Joint limit: a dog's elbow and stifle bend one way only. The knee stays within
            # KNEE_SIDE_MAX_DEG of the leg's natural side about the girdle-to-wrist line -- the
            # elbow behind and below it, the stifle in front -- measured in the body's own frame,
            # so it moves with the body and never jumps; a knee on the side the joint cannot
            # reach is what a flip of 100 deg or more looks like.
            fwd_b = D[v["frame"]][f][:, 1].copy(); fwd_b[2] = 0; fwd_b /= max(np.linalg.norm(fwd_b), 1e-9)
            nat = fwd_b if v["frame"] == "pel" else -fwd_b + np.array([0.0, 0.0, -FRONT_POLE_DOWN])
            u = (hock - root) / max(np.linalg.norm(hock - root), 1e-9)
            c0 = root + u * np.dot(knee - root, u); rr = np.linalg.norm(knee - c0)
            e1 = nat - u * np.dot(nat, u)
            if rr > 1e-6 and np.linalg.norm(e1) > 1e-6:
                e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
                a = math.atan2(np.dot(knee - c0, e2), np.dot(knee - c0, e1))
                amax = math.radians(KNEE_SIDE_MAX_DEG)
                if abs(a) > amax:
                    a = math.copysign(amax, a)
                    knee = c0 + rr * (math.cos(a) * e1 + math.sin(a) * e2)
            knee_prev[k] = knee.copy()
            for (bn, nxt), p0, p1 in (((bs[0], bs[1]), root, knee), ((bs[1], bs[2]), knee, hock)):
                cur = pf[bn][:3, :3] @ (rig.rest[bn][:3, :3].T @ (rig.head(nxt) - rig.head(bn)))
                M = pf[bn].copy(); M[:3, :3] = _swing(cur, p1 - p0) @ M[:3, :3]; M[:3, 3] = p0; pf[bn] = M
            M = pf[bs[2]].copy()
            cur = M[:3, :3] @ (rig.rest[bs[2]][:3, :3].T @ (rig.head(ctrl) - rig.head(bs[2])))
            M[:3, :3] = _swing(cur, tgt - hock) @ M[:3, :3]; M[:3, 3] = hock; pf[bs[2]] = M
            # where the paw really ends: the toe stays on the metatarsus, never stretched to the target
            paw_at = hock + (tgt - hock) / max(np.linalg.norm(tgt - hock), 1e-9) * meta_len
            shortfall[k] = max(shortfall[k], float(np.linalg.norm(paw_at - tgt)))
            tgt = paw_at
            # the toe lies flat, yawed with the body
            M = rig.rest[ctrl].copy()
            fwd = D[v["frame"]][f][:, 1].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
            M[:3, :3] = _swing(np.array([0, 1.0, 0]), fwd) @ rig.rest[ctrl][:3, :3]; M[:3, 3] = tgt
            pf[ctrl] = M
        final = {}
        for nm in rig.order:
            par = rig.bones[nm].parent
            fk = rel(nm) if par is None else final[par.name] @ rel(nm)
            if nm in keyed or nm == pel:
                M = pf[nm].copy()
                if nm != pel:
                    M[:3, 3] = fk[:3, 3]          # every bone stays on its parent: no stretch
            else:
                M = fk
            if nm in absolute:
                v = M[:3, :3] @ (rig.rest[nm][:3, :3].T @ (rig.tail(nm) - rig.head(nm)))
                L_ = np.linalg.norm(v)
                gz = _floor_at(floor, M[0, 3] + v[0], M[1, 3] + v[1])
                if L_ > 1e-6 and M[2, 3] + v[2] < TAIL_CLEARANCE_M + gz:
                    z = float(np.clip(TAIL_CLEARANCE_M + gz - M[2, 3], -L_, L_))
                    h = v.copy(); h[2] = 0; hn = np.linalg.norm(h)
                    h = h / hn if hn > 1e-9 else np.array([0.0, -1.0, 0.0])
                    v2 = h * math.sqrt(max(L_ * L_ - z * z, 0.0)); v2[2] = z
                    M = M.copy(); M[:3, :3] = _swing(v, v2) @ M[:3, :3]
            final[nm] = M
            if not (nm in keyed or nm == pel):
                continue
            loc, q, _ = Matrix((np.linalg.inv(fk) @ M).tolist()).decompose()
            pb = arm.pose.bones[nm]
            if nm == pel:
                pb.location = loc; pb.keyframe_insert("location", frame=f + 1, group=nm)
            if nm in prevq and prevq[nm].dot(q) < 0:
                q.negate()
            prevq[nm] = q.copy()
            pb.rotation_quaternion = q; pb.keyframe_insert("rotation_quaternion", frame=f + 1, group=nm)
    return {"action": act.name, "frames": n, "fps": fps, "seconds": round(n / fps, 3),
            "scale": round(float(s), 4), "source": c.source, "placed": c.placed,
            "shoulder_tilt_deg": round(math.degrees(shoulder_tilt), 2),
            "stride_scale": {k: round(v, 3) for k, v in paw_scale.items()},
            "paw_shortfall_m": {k: round(v, 4) for k, v in shortfall.items()},
            "contacts_s": {k: [(round(a / fps, 2), round(b / fps, 2)) for a, b in v]
                           for k, v in contact_runs.items()}}
