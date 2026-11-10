"""Advisory motion diagnostics, including speed alerts; never block acceptance."""
from collections import defaultdict, OrderedDict
import hashlib
import math

import numpy as np


def frame_runs(frames):
    """Sorted frame indices as inclusive `[lo, hi]` spans.

    Verifier evidence names frame *sets*, and every set this layer reports is
    dense: a check runs over `range(start, end+1)`, so enumerating the members
    spends a token per frame to say what two numbers already say. `verify.py`
    imports this from here rather than the other way around — `verify` imports
    `continuity` at module load, so the dependency only runs in this direction.
    """
    spans = []
    for frame in frames:
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def quaternion_matrices(rows):
    """`(n, 4)` unit quaternions `(w, x, y, z)` -> `(n, 3, 3)` rotation matrices."""
    q = np.asarray(rows, dtype=float).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], axis=1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], axis=1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=1)], axis=1)


def joint_rotations(rows, parents):
    """Each bone's orientation relative to its parent bone, `(n, 3, 3)`.

    `rows` are armature-space orientations (`worker.bone_rotations`), so a
    bone's own joint is `R_parent^T R_bone`: what the joint did, with
    everything above it removed. A root bone keeps its armature-space
    orientation, which is relative to the armature object.
    """
    world = quaternion_matrices(rows)
    local = world.copy()
    parents = np.asarray(parents, dtype=int)
    has = parents >= 0
    local[has] = np.transpose(world[parents[has]], (0, 2, 1)) @ world[has]
    return local


def rotation_vector(matrix):
    # Stable axis-angle conversion, including pi rotations; no scipy in bpy env.
    angle = math.acos(float(np.clip((np.trace(matrix)-1)/2, -1, 1)))
    skew = np.array([matrix[2,1]-matrix[1,2], matrix[0,2]-matrix[2,0], matrix[1,0]-matrix[0,1]])
    if angle < 1e-7:
        return skew/2
    if math.pi-angle < 1e-5:
        values, vectors = np.linalg.eigh((matrix+matrix.T)/2)
        axis = vectors[:, np.argmax(values)]
        if axis @ skew < 0:
            axis = -axis
        return angle*axis
    return angle/(2*math.sin(angle))*skew



class MotionContinuity:
    """Compare consecutive visible samples; keep only sampled meshes in memory.

    What this verifier can and cannot show, stated here rather than on every report:

    - Heuristic sampled-frame diagnostics, not a physical speed limit or a proof of smoothness. The verdict is
      `uncertain`, never `fail`, and `required_checks_pass` does not consult it.
    - Visibility gaps and renames restart tracks; topology changes skip vertex correspondence and are counted in
      `topology_changes`.
    - **No speed channel.** An absolute bound on world displacement is exceeded permanently by every small part on a
      moving linkage, even relative to its kinematic parent, so it describes the motion rather than locating a
      defect. The scale-free spike channel is what localizes a teleport.
    - Rotation uses shortest-arc matrix displacement and can alias rotations above pi per frame.
    - Fast motion, cuts, impacts, and intentional reversals may be legitimate: a row is a place to look, not a defect.

    Channels, each per body and judged only by the spike/jitter rules in `report`: `translation` (floor
    `max(2 cm, 25 % of size)`), `rotation` (floor 20 degrees), `deformation` (RMS of the 256 samples after rigid motion
    is removed, floor `max(1 cm, 15 % of size)`), and `bone`.

    `bone` is per armature bone: the rotation of its joint relative to its parent bone (`joint_rotations`) between
    consecutive frames, floor `BONE_FLOOR`, reported as `<armature>:<bone>`. A skinned character is one body whose
    object frame holds still, so a forearm flipping 175 degrees moves the skin under a centimeter RMS, far under any
    deformation floor. Spikes only, because a fast oscillation (a tail, a wing) is intentional reversal and the jitter
    rule would make every one a row.
    """

    #: A joint turning this much in one frame, and 8x its own neighborhood's
    #: median, is a spike. 25 degrees per frame is 600 deg/s at 24 fps: past
    #: anything a deliberate limb motion does between two frames, and well below
    #: the near-half-turn of a limb that flips in one frame.
    BONE_FLOOR = math.radians(25)
    #: Channels the jitter (repeated reversal) rule does not apply to.
    SPIKE_ONLY = frozenset({"bone"})

    def __init__(self, fps):
        self.fps = fps
        self.previous = {}
        self.steps = defaultdict(list)
        self.last_vectors = {}
        #: Frames at which a body's topology changed, so a chunk counts only
        #: the ones it owns.
        self.topology_changes = []
        self._topology = {}
        # A body's sample set is a pure function of its mesh and its placement,
        # and most bodies hold still: about a quarter of a film's body-frames
        # are distinct (body, mesh, pose) states. Bounded rather than
        # complete, because a moving body's key is new every frame and only the
        # still ones are worth keeping; 256 sampled rows in float64 is 6 KB an
        # entry, so an unbounded table would outgrow the frame data it summarizes.
        self._states = OrderedDict()
        self._state_bytes = 0
        self._state_limit = 64 * 1024**2
        #: `name -> [state key, frames held, steps]`: how many consecutive frames
        #: a body has held one state, and once that is three, the steps it was
        #: measured to take. A body that has held a state for three frames
        #: takes the same steps from it every frame -- the same vectors against
        #: the same previous vectors -- so they are taken again rather than
        #: recomputed. 62-98 % of the survey films' body-frames are such holds.
        self._still = {}
        #: `(name, frame) -> (rotation, translation, size)`: the world-space
        #: rigid displacement a body underwent into `frame`, kept only for
        #: steps past their floor -- the only ones that can become a spike --
        #: so `report` can tell one defect seen on many attached parts from
        #: many defects. See `_collapse_rigid`.
        self.displacements = {}
        #: `armature -> (frame, joint rotations)` for the `bone` channel.
        self._previous_bones = {}

    def _topology_digest(self, triangles):
        """Per-array topology hash, memoized for the run of frames that share it.

        A rigid part hands over the *same* immutable triangle array every frame
        (`evaluation.read_local` serves them out of its own cache), so hashing
        it per body per frame re-hashes hundreds of kilobytes for an answer that
        cannot have changed. Keyed on identity, holding a reference so the id
        cannot be recycled under the key while the entry is live.
        """
        cached = self._topology.get(id(triangles))
        if cached is not None:
            return cached[0]
        digest = hashlib.blake2b(triangles.tobytes(), digest_size=16).digest()
        if len(self._topology) > 256:
            self._topology.clear()
        self._topology[id(triangles)] = (digest, triangles)
        return digest

    def add(self, frame, record, geometry, local=None, bounds=None):
        """`local` and `bounds` let a caller skip this method's O(vertices) work.

        Nothing here needs the full world array. The samples are the rigid
        motion removed, `(world[i] - center) @ rotation`; since
        `world = local @ L.T + center` that is exactly
        `local[i] @ (rotation.T @ L).T`, computable from the 256 sampled *local*
        rows. And `size` is the norm of the world extent, i.e. `hi - lo`. So a
        caller that already reduced the world bounds — the frame loop does, for
        its broad phase — can hand both over and this costs O(256) rather than
        two more full passes per body per frame. Callers that pass neither (the
        continuity unit tests) take the world-array path below, unchanged.
        """
        current = {}
        for obj in record['objects']:
            if obj.get('hidden') or obj.get('room'):
                continue
            # Names are only temporal correspondence hints, not required identities.
            name = obj['name']
            matrix = np.asarray(obj['matrix'], dtype=float)
            supplied = (local is not None and name in local
                        and bounds is not None and name in bounds)
            if supplied:
                sampled, triangles = local[name]
                lo, hi = bounds[name]
                # min/max propagate NaN and +-Inf, so finite bounds is exactly
                # `np.isfinite(world).all()`.
                finite = bool(np.isfinite(lo).all() and np.isfinite(hi).all())
            else:
                sampled, triangles = geometry[name]
                lo = hi = None
                finite = bool(len(sampled)) and bool(np.isfinite(sampled).all())
            if not len(sampled) or not finite or not np.isfinite(matrix).all():
                continue
            state_key = (name, obj.get('mesh'), matrix.tobytes()) if obj.get('mesh') else None
            state = self._states.get(state_key) if state_key else None
            if state is not None:
                self._states.move_to_end(state_key)
                center, rotation, samples, topology, size = state
            else:
                try:
                    u, _, vt = np.linalg.svd(matrix[:3, :3])
                    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
                    rotation = u @ correction @ vt
                except np.linalg.LinAlgError:
                    continue
                center = matrix[:3, 3]
                index = np.linspace(0, len(sampled)-1, min(256, len(sampled)), dtype=int)
                # Remove rigid motion, retaining scale and deformation in meters.
                if supplied:
                    samples = sampled[index].astype(float) @ (rotation.T @ matrix[:3, :3]).T
                    extent = hi - lo
                else:
                    samples = (sampled[index] - center) @ rotation
                    extent = np.ptp(sampled, axis=0)
                topology = (len(sampled), self._topology_digest(triangles))
                size = float(np.linalg.norm(extent))
                if state_key is not None:
                    cost = samples.nbytes + center.nbytes + rotation.nbytes
                    while self._states and self._state_bytes + cost > self._state_limit:
                        _, old_state = self._states.popitem(last=False)
                        self._state_bytes -= (old_state[2].nbytes + old_state[0].nbytes
                                              + old_state[1].nbytes)
                    if cost <= self._state_limit:
                        self._states[state_key] = (center, rotation, samples, topology, size)
                        self._state_bytes += cost
            current[name] = (frame, center, rotation, samples, topology, size)
            prev = self.previous.get(name)
            if prev is None or prev[0] != frame - 1:
                self._still[name] = [state_key, 1, None]
                continue
            still = self._still.get(name)
            if still is not None and state_key is not None and still[0] == state_key:
                still[1] += 1
            else:
                still = self._still[name] = [state_key, 1, None]
            if still[2] is not None:
                for kind, (vector, floor, magnitude, reversal) in still[2][0].items():
                    self.last_vectors[(name, kind)] = (frame, vector)
                    self.steps[(name, kind)].append((frame, magnitude, floor, reversal))
                if still[2][1] is not None:
                    self.displacements[(name, frame)] = still[2][1]
                continue
            taken = {}
            translation_floor = max(.02, .25 * max(size, prev[5]))
            taken['translation'] = self._step(name, 'translation', frame, center - prev[1],
                                              translation_floor)
            turn = rotation @ prev[2].T
            angular_step = rotation_vector(turn)
            taken['rotation'] = self._step(name, 'rotation', frame, angular_step, math.radians(20))
            displacement = None
            if (np.linalg.norm(center - prev[1]) > translation_floor
                    or np.linalg.norm(angular_step) > math.radians(20)):
                displacement = self.displacements[(name, frame)] = (turn, center - turn @ prev[1], size)
            if topology == prev[4]:
                taken['deformation'] = self._step(
                    name, 'deformation', frame, (samples - prev[3]).ravel() / math.sqrt(len(samples)),
                    max(.01, .15 * max(size, prev[5])))
            else:
                self.topology_changes.append(frame)
            if still[1] >= 3:
                still[2] = (taken, displacement)
        self.previous = current

    def add_bones(self, frame, record, rigs):
        """The `bone` channel for one frame: `record["bone_rotations"]` against the last.

        `rigs` is the evaluation index's `rigs` table (bone order and parents).
        An armature missing from either, or whose bone count disagrees with the
        table, is skipped rather than guessed at; a gap in frames restarts it.
        """
        rotations = record.get("bone_rotations") or {}
        current = {}
        for armature, rows in rotations.items():
            spec = (rigs or {}).get(armature)
            if not spec or len(rows) != len(spec["bones"]):
                continue
            local = joint_rotations(rows, spec["parents"])
            if not np.isfinite(local).all():
                continue
            current[armature] = (frame, local)
            prev = self._previous_bones.get(armature)
            if prev is None or prev[0] != frame - 1:
                continue
            turns = local @ np.transpose(prev[1], (0, 2, 1))
            for k, name in enumerate(spec["bones"]):
                self._step(f"{armature}:{name}", "bone", frame,
                           rotation_vector(turns[k]), self.BONE_FLOOR)
        self._previous_bones = current

    def _step(self, name, kind, frame, vector, floor):
        key = (name, kind)
        magnitude = float(np.linalg.norm(vector))
        previous = self.last_vectors.get(key)
        reversal = False
        if previous is not None and previous[0] == frame - 1:
            old = previous[1]
            product = float(np.linalg.norm(old)) * magnitude
            reversal = (product > 0 and float(old @ vector) / product < -.8
                        and min(float(np.linalg.norm(old)), magnitude) > floor / 10)
        self.last_vectors[key] = (frame, vector)
        self.steps[key].append((frame, magnitude, floor, reversal))
        return vector, floor, magnitude, reversal

    def joins(self, clips):
        """The worst step across each clip boundary, named by the clips it joins.

        The data is already here: a boundary is a consecutive-frame pair like
        any other, and `steps` holds every one of them. What it lacks is a
        name. Among thousands of rows addressed as "frames 503/504", a
        teleport at a cut looks like mid-clip jitter. Labeling the joins
        costs a dictionary walk.

        Reported, never a verdict: a real cut is a discontinuity on purpose,
        and only the author knows which ones are meant.
        """
        if not clips:
            return []
        starts = {}
        for name, span in clips.items():
            starts.setdefault(int(span["frames"][0]), []).append(name)
        order = sorted((int(span["frames"][0]), name) for name, span in clips.items())
        # One pass over the steps for every boundary at once, in the order a
        # pass per boundary visited them, so the same step wins a tie. Per
        # boundary it was clips x steps: 399 clips made it 44 of 45 s of a
        # 300-frame check of the chair film.
        boundaries = {frame for index, (frame, _) in enumerate(order) if index and frame > 0}
        worst_at = {}
        for (obj, kind), steps in self.steps.items():
            for at, magnitude, _floor, _reversal in steps:
                if at in boundaries:
                    worst = worst_at.setdefault(at, {})
                    if magnitude > worst.get(kind, (0, None))[0]:
                        worst[kind] = (magnitude, obj)
        rows = []
        for index, (frame, name) in enumerate(order):
            if index == 0 or frame <= 0:
                continue
            worst = worst_at.get(frame)
            if not worst:
                continue
            rows.append({"frame": frame, "enters": name,
                         "leaves": order[index - 1][1],
                         "steps": {kind: {"object": obj, "magnitude": round(value, 6),
                                          "rate": round(value * self.fps, 6),
                                          "units": "rad" if kind in ("rotation", "bone") else "m"}
                                   for kind, (value, obj) in sorted(worst.items())}})
        return rows

    def report(self, owned=None, clips=None):
        """Advisory motion diagnostics: `aggregate` over this pass's own `rows`."""
        return self.aggregate(self.rows(owned, clips), self.fps)

    def rows(self, owned=None, clips=None):
        """The findings anchored inside `owned`, in the form `aggregate` reads.

        `owned=(first, last)`, or a predicate on a frame, keeps only findings anchored there,
        while still ranking each one against the neighbors outside it. One
        chunk of a parallel check therefore judges its frames exactly as a
        whole-timeline pass would and counts each frame once, and the rows of
        several chunks concatenate into the rows of one pass: spike and jitter
        tests read at most three frames either side, a rigid collapse is per
        frame, and a clip join is one frame.

        `clips` names the NLA strips, so a step that lands on a clip boundary
        can be reported as that boundary rather than as two frame numbers.
        """
        def anchored(frame):
            if owned is None:
                return True
            return owned(frame) if callable(owned) else owned[0] <= frame <= owned[1]
        warnings = []
        for (name, kind), rows in self.steps.items():
            for i, (frame, magnitude, floor, reversal) in enumerate(rows):
                neighbors = [r[1] for r in rows[max(0, i-3):i+4]
                             if r[0] != frame and abs(r[0]-frame) <= 3]
                # Absolute noise floor plus local contrast: fast smooth motion is allowed.
                spike = (len(neighbors) >= 2 and magnitude > floor
                         and magnitude > 8 * max(float(np.median(neighbors)), 1e-9))
                jitter = (kind not in self.SPIKE_ONLY and reversal and i > 0
                          and rows[i-1][0] == frame-1 and rows[i-1][3])
                if not (spike or jitter) or not anchored(frame):
                    continue
                # Collect all of them: the emitted evidence groups by
                # (object, channel, reason), and a cap here would keep an
                # arbitrary, not worst, subset of what runs to five figures on
                # a long clip, dropping groups a repair would start from.
                first = max(0, frame - (3 if jitter else 1))
                warnings.append({'frame': frame, 'object': name, 'channel': kind,
                                 'reason': 'repeated direction reversal' if jitter else 'isolated motion spike',
                                 'frames': [first, frame],
                                 'step_magnitude': float(magnitude), 'rate': float(magnitude*self.fps),
                                 'units': 'rad/s' if kind in ('rotation', 'bone') else 'm/s'})
        kept = self._collapse_rigid(warnings)
        # Every row names the frame it is anchored at, counts included, so a
        # re-check can replace exactly the frames it re-verified.
        def per_frame(frames):
            counted = {}
            for frame in frames:
                counted[frame] = counted.get(frame, 0) + 1
            return [{'frame': frame, 'count': n} for frame, n in sorted(counted.items())]
        kept_ids = {id(w) for w in kept}
        return {'warnings': kept,
                'rigid_duplicates': per_frame(w['frame'] for w in warnings if id(w) not in kept_ids),
                'transitions_checked': per_frame(row[0] for rows in self.steps.values()
                                                 for row in rows if anchored(row[0])),
                'topology_changes': per_frame(f for f in self.topology_changes if anchored(f)),
                'boundaries': [row for row in self.joins(clips) if anchored(row['frame'])]}

    @classmethod
    def aggregate(cls, rows, fps, adjacent=True):
        """One `MotionContinuity` report from `rows`, one pass's or several concatenated.

        `adjacent=False` says no two checked frames were consecutive -- a
        storyboard's states, any scattered set -- so there was no step to
        test, and the report is `not_applicable` rather than a `pass` with
        `transitions_checked: 0`.
        """
        kept = rows['warnings']
        total = {key: sum(row['count'] for row in rows[key])
                 for key in ('rigid_duplicates', 'transitions_checked', 'topology_changes')}
        if not adjacent and not kept:
            return {'name': 'MotionContinuity', 'verdict': 'not_applicable',
                    'evidence': {'diagnostic_only': True,
                                 'not_applicable': 'no two checked frames are adjacent'},
                    'suggested_repairs': []}
        grouped = cls._grouped(kept, fps) if kept else []
        # `warning_count` counts the grouped rows an author reads, one per
        # (object, channel, reason); the raw per-frame total is
        # `occurrences_total`. The headline is the grouped count because the raw
        # total, tens of thousands on a long film, overstates a report of a few
        # hundred rows.
        evidence = {'diagnostic_only': True, 'warning_count': len(grouped),
                    'occurrences_total': len(kept),
                    'transitions_checked': total['transitions_checked']}
        if total['rigid_duplicates']:
            evidence['rigid_duplicates'] = total['rigid_duplicates']
        if grouped:
            evidence['spikes'] = grouped
        if total['topology_changes']:
            evidence['topology_changes'] = total['topology_changes']
        if rows['boundaries']:
            evidence['boundaries'] = sorted(
                rows['boundaries'], key=lambda r: -max(s["magnitude"] for s in r["steps"].values()))
        return {'name': 'MotionContinuity', 'verdict': 'uncertain' if kept else 'pass',
                'evidence': evidence,
                'suggested_repairs': ['Inspect the reported frames; repair unintended jumps/jitter or explain intentional transitions.'] if kept else []}

    #: How closely two bodies' world displacements must agree to count as one
    #: rigid motion. Far below any step that can spike -- a spike is past a
    #: floor of at least 2 cm or 20 degrees -- and far above the float32
    #: matrix noise the evaluated cache carries.
    RIGID_TOLERANCE = 1e-4

    def _collapse_rigid(self, warnings):
        """One warning per rigid motion per frame, naming the parts it carried.

        `movers_carry_piano_upstairs` got one head fault back as eleven rows —
        the head, then its brows, cap brim, eye whites, ears, each a
        separately named spike at the same frames. Bodies rigidly attached to
        one another undergo the *identical* world-space rigid displacement
        between two frames (`M_b = M_a C` gives `M_b,t M_b,t-1^-1 = M_a,t
        M_a,t-1^-1`), so that displacement is the test, and it needs no
        declared parentage: it holds for IR parts bound to links and for native
        objects parented in Blender alike, and it reflects the motion that
        actually happened rather than the attachment someone intended.

        Per channel and reason, per frame. The largest body represents its
        cluster and the rest are named in `also`. Deformation is left alone:
        it is measured after rigid motion is removed, so it is per body by
        construction.
        """
        buckets = {}
        for warning in warnings:
            if warning['channel'] != 'deformation':
                key = (warning['channel'], warning['reason'], warning['frames'][1])
                buckets.setdefault(key, []).append(warning)
        dropped = set()
        for (_, _, frame), members in buckets.items():
            if len(members) < 2:
                continue
            moved = [(warning, self.displacements.get((warning['object'], frame)))
                     for warning in members]
            moved = sorted(((w, d) for w, d in moved if d is not None),
                           key=lambda item: (-item[1][2], item[0]['object']))
            clusters = []
            for warning, (turn, shift, _) in moved:
                for cluster in clusters:
                    head_turn, head_shift = cluster[1]
                    if (np.abs(turn - head_turn).max() <= self.RIGID_TOLERANCE
                            and np.abs(shift - head_shift).max() <= self.RIGID_TOLERANCE):
                        cluster[0].setdefault('also', []).append(warning['object'])
                        dropped.add(id(warning))
                        break
                else:
                    clusters.append((warning, (turn, shift)))
        return [warning for warning in warnings if id(warning) not in dropped]

    @staticmethod
    def _grouped(items, fps):
        """One row per `(object, channel, reason)`, worst first.

        A single jittering part emits one warning per offending frame, so the
        flat list was mostly the same three strings repeated: 100 rows carried
        about 35 distinct groups, and the object/channel/reason triple is what
        a repair is actually keyed on. Grouping also lets every occurrence be
        reported rather than a prefix, since the cost is now per group.

        `worst_rate` is the peak of the group; `excess` ranks speed rows by how
        far past their own threshold they went, which differs from raw rate
        when thresholds vary per part.
        """
        groups = {}
        for item in items:
            groups.setdefault((item['object'], item['channel'], item['reason']), []).append(item)
        rows = []
        for (name, kind, reason), members in groups.items():
            frames = sorted({frame for member in members
                             for frame in range(member['frames'][0], member['frames'][1] + 1)})
            worst = max(members, key=lambda m: m['rate'])
            row = {'object': name, 'channel': kind, 'reason': reason,
                   'occurrences': sum(member.get('starts', 1) for member in members),
                   'frames': frame_runs(frames),
                   'window': [round(frames[0]/fps, 4), round(frames[-1]/fps, 4)],
                   'worst_rate': round(worst['rate'], 4), 'units': worst['units'],
                   'representative_frame': worst.get('representative_frame', worst['frames'][1])}
            if 'threshold' in worst:
                row['threshold'] = round(worst['threshold'], 4)
                row['excess'] = round(worst['rate']/worst['threshold'], 3)
            also = sorted({name for member in members for name in member.get('also', ())})
            if also:
                row['also'] = also
            rows.append(row)
        # Ties broken by name so the order does not depend on which frames a
        # given process happened to visit first.
        rows.sort(key=lambda r: (-r.get('excess', r['worst_rate']),
                                 r['object'], r['channel'], r['reason']))
        return rows
