"""Coal placed-mesh queries plus Blender-assisted solid containment.

A persistent isolated Coal worker caches local BVHs. Distances are distances between the evaluated triangle
surfaces, not CAD solids. Penetration diagnostics are directional separations,
not a global minimum translational penetration depth.
"""
from collections import OrderedDict, deque
import hashlib
import math
import atexit
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
from mathutils.bvhtree import BVHTree
try:
    from ...verify._mesh_erosion import _erode, _weld_indices
except ImportError:
    from _mesh_erosion import _erode, _weld_indices

VERSION = 'coal-placed-v3'
TOLERANCE = 1e-6
POSE_CACHE_QUANTUM = 1e-6  # the same quantum as _exact_adjudicator._POSE_QUANTUM_M
# Float32 rigid matrices may carry tiny apparent stretch. Normalize only within
# both a dimensionless matrix bound and a sub-tolerance world displacement bound.
RIGID_RESIDUAL_LIMIT = 1e-6
#: How much apparent stretch a "rigid" placement may carry, as a world
#: displacement in meters.
#:
#: **Above the noise floor of its own input.** The evaluated cache stores
#: matrices as float32, so a pure rotation comes back with roughly 1.2e-7 of
#: relative error, and on a body a meter across the `residual_norm * radius`
#: product sits around 1.6e-7. A limit below that sends most placements down
#: the residual-baked branch of `GeometryCache.placed`, which builds
#: `vertices @ residual.T`: a vertex array unique to that body on that frame,
#: handed to `get()` with **no key**, so it re-hashes the array and builds a BVH
#: that can never be reused.
#:
#: 5e-7 m is half a micron, and half of the 1e-6 m `TOLERANCE` the verdict
#: already spends on erosion, so no contact answer can turn on it, and only
#: the genuinely non-rigid placements this branch exists for still bake.
RIGID_DISPLACEMENT_LIMIT = 5e-7
_TOPOLOGY_CACHE = OrderedDict()
_TOPOLOGY_BYTES = 0
_TRIANGLE_CACHE = OrderedDict()
_TRIANGLE_BYTES = 0


def triangle_data(triangles):
    """Reuse pose-independent indices; deformation still supplies fresh vertices."""
    global _TRIANGLE_BYTES
    key = hashlib.sha256(triangles.tobytes()).digest()
    if key in _TRIANGLE_CACHE:
        _TRIANGLE_CACHE.move_to_end(key)
        return _TRIANGLE_CACHE[key][0]
    # The digest is the topology's name in the Coal worker, which keeps one
    # triangle list per topology and is sent the triangles only once (fix 5).
    data = {'digest': key}
    size = triangles.nbytes * 24  # Include Python lists/ints.
    while _TRIANGLE_CACHE and (_TRIANGLE_BYTES + size > 256 * 1024**2 or len(_TRIANGLE_CACHE) >= 1024):
        _, (_, old_size) = _TRIANGLE_CACHE.popitem(last=False)
        _TRIANGLE_BYTES -= old_size
    if size <= 256 * 1024**2:
        _TRIANGLE_CACHE[key] = data, size
        _TRIANGLE_BYTES += size
    return data


def mesh_key(vertices, triangles):
    h = hashlib.sha256()
    for array in (np.asarray(vertices, dtype='<f8'), np.asarray(triangles, dtype='<i4')):
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.digest()


class Body:
    def __init__(self, vertices, triangles, key=None):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or not len(self.vertices) or not len(self.triangles)
                or not np.isfinite(self.vertices).all()
                or self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices)):
            raise ValueError('invalid collision mesh')
        self.lo, self.hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        self._triangle_data = triangle_data(self.triangles)
        # A content hash the caller already holds -- the evaluation's asset
        # key -- rather than another pass over the arrays.
        self.key = (bytes.fromhex(key) if isinstance(key, str) else key) or mesh_key(self.vertices, self.triangles)
        self.matrix = np.eye(4)
        self.local = self
        self._tree = None
        self._components = None
        self._erosions = {}

    @property
    def tree(self):
        if self._tree is None:
            if 'polygons' not in self._triangle_data:
                self._triangle_data['polygons'] = self.triangles.tolist()
            self._tree = BVHTree.FromPolygons(self.vertices.tolist(), self._triangle_data['polygons'], all_triangles=True)
        return self._tree

    @property
    def radius(self):
        """The farthest vertex from the local origin: how far a rotation can move one.

        Memoized per body, and bodies are cached per mesh. `gap_predicate`
        read it on every certified-distance reuse, over every vertex, and
        that was 73% of the advisory checks' distance CPU on the dog film.
        """
        if not hasattr(self, '_radius'):
            self._radius = float(np.linalg.norm(self.vertices, axis=1).max())
        return self._radius

    @property
    def component_bounds(self):
        if not hasattr(self, '_component_bounds'):
            lo, hi = self.component_bound_arrays
            self._component_bounds = list(zip(lo, hi))
        return self._component_bounds

    @property
    def component_bound_arrays(self):
        """`(lo, hi)`, each `(components, 3)`: every component's bounds, in one pass.

        One reduction per component is thousands of them on a furred
        character, on every placed body of every frame. Which vertices belong to
        which component is a property of the topology, so it is cached there.
        """
        if not hasattr(self, '_component_bound_arrays'):
            if len(self.components) == 1:
                self._component_bound_arrays = (self.lo[None], self.hi[None])
                return self._component_bound_arrays
            digest = hashlib.sha256()
            for indices, _ in self.components:
                digest.update(str(indices.shape).encode())
                digest.update(indices.tobytes())
            key = digest.digest()
            groups = self._triangle_data.setdefault('component_vertices', OrderedDict())
            if key not in groups:
                members = [np.unique(self.triangles[ids]) for ids, _ in self.components]
                order = np.concatenate(members)
                starts = np.cumsum([0] + [len(m) for m in members[:-1]])
                groups[key] = (order, starts)
                if len(groups) > 4:
                    groups.popitem(last=False)
            groups.move_to_end(key)
            order, starts = groups[key]
            vertices = self.vertices[order]
            self._component_bound_arrays = (np.minimum.reduceat(vertices, starts, axis=0),
                                            np.maximum.reduceat(vertices, starts, axis=0))
        return self._component_bound_arrays

    @property
    def component_witnesses(self):
        """`(first triangle per component, closed per component)`, as arrays."""
        if not hasattr(self, '_component_witnesses'):
            self._component_witnesses = (
                np.array([indices[0] for indices, _ in self.components], dtype=np.int64),
                np.array([closed for _, closed in self.components], dtype=bool))
        return self._component_witnesses

    @property
    def components(self):
        """Welded connected components; only closed shells have a solid interior."""
        if self._components is not None:
            return self._components
        global _TOPOLOGY_BYTES
        first, inverse = _weld_indices(self.vertices)
        # Canonicalize welded indices by first occurrence, independent of rotation.
        inverse = np.argsort(np.argsort(first))[inverse]
        faces = inverse[self.triangles]
        key = hashlib.sha256(faces.tobytes()).digest()
        if key in _TOPOLOGY_CACHE:
            _TOPOLOGY_CACHE.move_to_end(key)
            self._components = _TOPOLOGY_CACHE[key]
            return self._components
        parent = np.arange(inverse.max()+1)
        edges = np.concatenate([faces[:, [0,1]], faces[:, [1,2]]])
        while True:
            old = parent.copy()
            a, b = parent[edges[:,0]], parent[edges[:,1]]
            np.minimum.at(parent, np.maximum(a,b), np.minimum(a,b))
            while not np.array_equal(parent, parent[parent]):
                parent = parent[parent]
            if np.array_equal(old, parent):
                break
        labels = parent[faces[:,0]]
        result = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            f = faces[indices]
            valid = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
            f = f[valid]
            edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges[:, 0] * len(parent) + edges[:, 1], return_counts=True)
            closed = bool(len(f) and np.all(counts % 2 == 0))
            result.append((indices, closed))
        size = sum(indices.nbytes for indices, _ in result)
        while _TOPOLOGY_CACHE and (_TOPOLOGY_BYTES+size > 64*1024**2 or len(_TOPOLOGY_CACHE) >= 256):
            _, value = _TOPOLOGY_CACHE.popitem(last=False)
            _TOPOLOGY_BYTES -= sum(indices.nbytes for indices, _ in value)
        if size <= 64*1024**2:
            _TOPOLOGY_CACHE[key] = result
            _TOPOLOGY_BYTES += size
        self._components = result
        return result

    def translated(self, delta):
        moved = Body(self.vertices + np.asarray(delta), self.triangles)
        moved._components = self.components
        moved._translated_from = (self, np.asarray(delta))
        return moved

    def eroded(self, chord):
        if chord not in self._erosions:
            if hasattr(self, '_translated_from'):
                source, delta = self._translated_from
                eroded = source.eroded(chord)
                self._erosions[chord] = eroded.translated(delta) if eroded is not None else None
                return self._erosions[chord]
            vertices = _erode(self.vertices, self.triangles, TOLERANCE, chord=chord)
            self._erosions[chord] = Body(vertices, self.triangles) if vertices is not None else None
        return self._erosions[chord]

    def inside(self, point):
        """Winding of all closed shells together preserves inward-wound cavities."""
        return self.inside_at(point)[0]

    def inside_at(self, point):
        """`(inside, distance)`: `inside()`'s answer and the point's distance to the surface.

        The distance is exact inside the box and a lower bound outside it,
        which is all `_contains` needs to know a witness cannot be inside.

        Scalar arithmetic per ray hit: numpy arrays per hit to decide whether
        it grazed an edge would make the per-call overhead, inside `_contains`,
        a large share of a cold check.
        """
        point = np.asarray(point, dtype=float)
        if np.any(point < self.lo) or np.any(point > self.hi):
            return False, float(np.linalg.norm(np.maximum(self.lo - point, 0.)
                                               + np.maximum(point - self.hi, 0.)))
        nearest = self.tree.find_nearest(tuple(point))
        if nearest[0] is None:
            return False, math.inf
        if nearest[3] <= TOLERANCE:
            return False, nearest[3]
        groups = [indices for indices, closed in self.components if closed]
        if not groups:
            return False, nearest[3]
        if not hasattr(self, '_solid_tree'):
            ids = np.concatenate(groups)
            self._solid_triangles = self.triangles[ids]
            self._solid_tree = (self.tree if len(ids) == len(self.triangles) else
                BVHTree.FromPolygons(self.vertices.tolist(), self._solid_triangles.tolist(), all_triangles=True))
            self._solid_faces = self._solid_triangles.tolist()
            self._vertex_rows = self.vertices.tolist()
        faces, rows = self._solid_faces, self._vertex_rows
        # Oriented ray intersections compute winding for regular samples in C.
        # Grazing edges/vertices or disagreeing rays use the solid-angle fallback.
        votes = []
        for dx, dy, dz in _WINDING_RAYS:
            cx, cy, cz = (float(v) for v in point)
            winding = 0
            regular = True
            for _ in range(256):
                hit, normal, index, distance = self._solid_tree.ray_cast((cx, cy, cz), (dx, dy, dz))
                if hit is None:
                    break
                i, j, k = faces[index]
                (ax, ay, az), (bx, by, bz), (qx, qy, qz) = rows[i], rows[j], rows[k]
                hx, hy, hz = hit
                dot = normal[0]*dx + normal[1]*dy + normal[2]*dz
                area = _cross_norm(bx-ax, by-ay, bz-az, qx-ax, qy-ay, qz-az)
                graze = min(_cross_norm(bx-ax, by-ay, bz-az, hx-ax, hy-ay, hz-az),
                            _cross_norm(qx-bx, qy-by, qz-bz, hx-bx, hy-by, hz-bz),
                            _cross_norm(ax-qx, ay-qy, az-qz, hx-qx, hy-qy, hz-qz))
                if distance < 8*TOLERANCE or abs(dot) < 1e-6 or graze < 1e-6*area:
                    regular = False
                    break
                winding += 1 if dot > 0 else -1
                cx, cy, cz = hx + dx*(2*TOLERANCE), hy + dy*(2*TOLERANCE), hz + dz*(2*TOLERANCE)
            else:
                regular = False
            if not regular:
                break
            votes.append(winding != 0)
        if len(votes) == 3 and len(set(votes)) == 1:
            return votes[0], nearest[3]
        tri = self.vertices[self.triangles[np.concatenate(groups)]] - point
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        num = np.einsum('ij,ij->i', a, np.cross(b, c))
        den = (la*lb*lc + np.einsum('ij,ij->i', a, b)*lc
               + np.einsum('ij,ij->i', b, c)*la + np.einsum('ij,ij->i', c, a)*lb)
        winding = abs(float(np.arctan2(num, den).sum() / (2*math.pi)))
        if .1 < winding < .9:
            raise ValueError('ambiguous closed mesh winding; repair self-intersecting or inconsistent geometry')
        return winding > .5, nearest[3]


#: The three ray directions `inside` votes with, as unit tuples.
_WINDING_RAYS = tuple(tuple(float(v) for v in np.asarray(raw) / np.linalg.norm(raw))
                      for raw in ((1., .371, .529), (.233, 1., .619), (.417, .293, 1.)))


def _cross_norm(ax, ay, az, bx, by, bz):
    """`|a x b|` for two 3-vectors given as scalars."""
    x, y, z = ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx
    return math.sqrt(x*x + y*y + z*z)


class GeometryCache:
    """Bounded local geometry; transforms are supplied separately to Coal."""
    def __init__(self, max_bytes=512 * 1024**2):
        self.entries = OrderedDict()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.builds = 0
        #: Placed bodies by (content key, exact matrix), as used this frame and
        #: the last (`next_frame`): a body that has not moved is the same
        #: placement, and placing it again costs an SVD per body per frame.
        #: Two frames rather than all of them: a moving or deforming body's
        #: earlier placements are never used again, and keeping them, with
        #: their world vertices and trees, runs a film's check to 10-20 GB.
        self._placed, self._previous = {}, {}

    def get(self, vertices, triangles, key=None):
        given = key
        key = key or mesh_key(vertices, triangles)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        body = Body(vertices, triangles, key=given)
        # Include both lazy eroded trees, Python/native topology, and BVH storage.
        size = int(vertices.nbytes * 12 + triangles.nbytes * 24)
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
        if size <= self.max_bytes:
            self.entries[key] = body, size
            self.bytes += size
        self.builds += 1
        return body


    def next_frame(self):
        """Forget the placements the frame before last used and the last did not."""
        self._previous, self._placed = self._placed, {}

    def placed(self, vertices, triangles, matrix, key=None):
        """`key`, when given, must be a content hash of (vertices, triangles).

        The evaluated cache already stores every local mesh under its own
        validated digest (`record['objects'][i]['mesh']`), so a per-frame caller
        can hand that over instead of making `get` re-hash multi-megabyte arrays
        once per body per frame. It is only sound for the *unbaked* lookup: the
        residual-baked branch below builds derived geometry, whose digest this
        is not.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        memo = (key, matrix.tobytes()) if key is not None else None
        held = memo and (self._placed.get(memo) or self._previous.get(memo))
        if held is not None:
            self._placed[memo] = held
            return held
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError('invalid collision transform')
        linear = matrix[:3, :3]
        # Keep evaluated local deformation; remove only bounded near-rigid
        # matrix noise. Real affine transforms take the baked-geometry path.
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        residual = rotation.T @ linear
        base = self.get(vertices, triangles, key)
        residual_norm = float(np.linalg.norm(residual - np.eye(3)))
        radius = float(np.linalg.norm(np.maximum(np.abs(base.lo), np.abs(base.hi))))
        if np.array_equal(linear, np.eye(3)):
            rotation = np.eye(3)
            baked = vertices
        elif (residual_norm <= RIGID_RESIDUAL_LIMIT
              and residual_norm * radius <= RIGID_DISPLACEMENT_LIMIT):
            baked = vertices
        else:
            baked = np.asarray(vertices, dtype=np.float64) @ residual.T
        pose = matrix.copy()
        pose[:3, :3] = rotation
        exact = base if baked is vertices else self.get(baked, triangles)
        if exact is not base and getattr(exact, '_proxy', None) is not base:
            exact._proxy = base
            if hasattr(exact, '_proxy_error'):
                del exact._proxy_error
        placed = PlacedBody(exact, pose)
        if memo is not None:
            self._placed[memo] = placed
        return placed


class PlacedBody:
    """Rigid placement of a shared local body, including lazy containment data."""
    def __init__(self, local, matrix):
        self.local, self.matrix = local, np.asarray(matrix, dtype=np.float64)
        self.key = local.key
        self.triangles = local.triangles
        self._tree = None
        self._triangle_data = local._triangle_data

    @property
    def vertices(self):
        if not hasattr(self, '_vertices'):
            self._vertices = self.local.vertices @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return self._vertices

    @property
    def lo(self):
        if not hasattr(self, '_lo'):
            self._lo, self._hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        return self._lo

    @property
    def hi(self):
        _ = self.lo
        return self._hi

    tree = Body.tree
    component_bounds = Body.component_bounds
    component_bound_arrays = Body.component_bound_arrays

    @property
    def component_witnesses(self):
        return self.local.component_witnesses

    @property
    def components(self):
        return self.local.components

    def inside(self, point):
        return self.local.inside((np.asarray(point)-self.matrix[:3, 3]) @ self.matrix[:3, :3])

    def inside_at(self, point):
        return self.local.inside_at((np.asarray(point)-self.matrix[:3, 3]) @ self.matrix[:3, :3])

    def eroded(self, chord):
        eroded = self.local.eroded(chord)
        return None if eroded is None else PlacedBody(eroded, self.matrix)

    def translated(self, delta):
        matrix = self.matrix.copy()
        matrix[:3, 3] += delta
        return PlacedBody(self.local, matrix)


class CoalClient:
    def __init__(self):
        try:
            from .coal_worker import send, receive
        except ImportError:
            from coal_worker import send, receive
        self.send, self.receive = send, receive
        interpreter = os.environ.get('DYCRAFT_COAL_PYTHON')
        if not interpreter:
            # Direct local/offline use follows the harness environment.
            candidates = [Path(sys.prefix).parent / 'dycraft' / 'bin/python']
            interpreter = next((str(p) for p in candidates if p.is_file()), sys.executable)
        self.process = subprocess.Popen([interpreter, '-I', '-B', str(Path(__file__).with_name('coal_worker.py'))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # The child's stderr still reaches this process's, and its last lines are kept for the error below: a child
        # that dies at start says why.
        self.stderr_tail = deque(maxlen=12)
        self._stderr_pump = threading.Thread(target=self._pump_stderr, name='coal-stderr', daemon=True)
        self._stderr_pump.start()
        self.known = set()
        #: Topology digests the worker holds triangles for, so a new pose of a
        #: known topology ships its vertices alone.
        self.topologies = set()
        self.stats = {}
        self.distances = OrderedDict()
        try:
            self.version = self.receive(self.process.stdout)['version']
        except Exception as error:
            self.close()
            self._stderr_pump.join(timeout=2)
            said = ' | '.join(line for line in self.stderr_tail if line)
            raise RuntimeError('Coal unavailable in collision interpreter'
                               + (f' ({interpreter} said: {said})' if said else '')) from error
        atexit.register(self.close)

    def _pump_stderr(self):
        """Pass the child's stderr on to file descriptor 2, where it went when inherited, keeping its last lines."""
        for raw in iter(self.process.stderr.readline, b''):
            try:
                os.write(2, raw)
            except OSError:
                pass
            self.stderr_tail.append(raw.decode('utf-8', 'replace').strip())
        self.process.stderr.close()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def proxies(self, a, b):
        # A cached rigid proxy is ONLY a clearing certificate. Maximum vertex
        # deviation bounds triangle Hausdorff distance; uncertain cases use the
        # actual baked geometry without rounding or increasing contact tolerance.
        proxies, error = [], 0.
        for body in (a, b):
            local = body.local
            proxy = getattr(local, '_proxy', local)
            if proxy is not local:
                if not hasattr(local, '_proxy_error'):
                    local._proxy_error = float(np.linalg.norm(local.vertices-proxy.vertices, axis=1).max())
                error += local._proxy_error
            proxies.append(PlacedBody(proxy, body.matrix))
        return proxies, error


    def measure_distances(self, pairs):
        """Exact surface distances for many pairs, in one round trip."""
        distances = self.exact_queries([('distance', a, b) for a, b in pairs])
        for (a, b), distance in zip(pairs, distances):
            self.remember(a, b, distance)
        return distances

    def remember(self, a, b, distance):
        """Retain a measured gap with the relative pose it was measured at."""
        key = a.key, b.key
        self.distances[key] = relative_matrix(a, b), distance
        self.distances.move_to_end(key)
        if len(self.distances) > 4096:
            self.distances.popitem(last=False)


    #: New mesh bytes one message may carry. The worker refuses a message
    #: past 512 MiB, and evicts only after a whole batch, so a frame of a
    #: deforming film that shipped every new mesh at once could exceed both.
    BATCH_BYTES = 64 * 1024**2

    def exact_queries(self, queries):
        """`[(op, a, b)]` answered in order, in as few round trips as fit.

        A frame can ask about a thousand queries, and a message costs hundreds
        of microseconds, so they travel together, split only where the new
        meshes they ship would pass `BATCH_BYTES`.
        """
        answers, group, shipping, size = [], [], set(), 0
        for query in queries:
            for x in (query[1].local, query[2].local):
                if x.key not in self.known and x.key not in shipping:
                    shipping.add(x.key)
                    size += x.vertices.nbytes + x.triangles.nbytes
            group.append(query)
            if size > self.BATCH_BYTES:
                answers += self._exact_batch(group)
                group, shipping, size = [], set(), 0
        return answers + (self._exact_batch(group) if group else [])

    def _ship(self, locals_):
        """`(meshes, keys)`: what to send for these local bodies, and every key once, in order."""
        meshes, keys, shipped = [], {}, set()
        for x in locals_:
            keys.setdefault(x.key, len(keys))
            if x.key in self.known or x.key in shipped:
                continue
            shipped.add(x.key)
            topology = x._triangle_data['digest']
            meshes.append((x.key, x.vertices,
                           None if topology in self.topologies else x.triangles, topology))
        return meshes, keys

    def _exchange(self, message, meshes):
        """The reply's answer, or `None` when the worker asked for meshes again."""
        self.send(self.process.stdin, message)
        reply = self.receive(self.process.stdout)
        if 'error' in reply:
            raise RuntimeError('Coal query failed: '+reply['error'])
        if 'missing_topology' in reply:
            # The worker dropped these triangle lists; send them again.
            self.topologies.difference_update(reply['missing_topology'])
            return None
        self.known.update(m[0] for m in meshes)
        self.topologies.update(m[3] for m in meshes)
        self.known.difference_update(reply.get('evicted', []))
        self.topologies.difference_update(reply.get('evicted_topologies', []))
        if 'missing' in reply:
            self.known.difference_update(reply['missing'])
            return None
        self.stats = reply.get('stats', {})
        return reply

    def _exact_batch(self, queries):
        for _ in range(3):
            meshes, keys = self._ship(x for _, a, b in queries for x in (a.local, b.local))
            subqueries = [{'op': op, 'keys': [a.local.key, b.local.key], 'poses': [a.matrix, b.matrix]}
                          for op, a, b in queries]
            if len(subqueries) == 1:
                message = {**subqueries[0], 'meshes': meshes}
            else:
                message = {'op': 'batch', 'keys': list(keys), 'queries': subqueries, 'meshes': meshes}
            reply = self._exchange(message, meshes)
            if reply is not None:
                return reply['answer'] if len(subqueries) > 1 else [reply['answer']]
        raise RuntimeError('Coal cache synchronization failed')

    def shape_batch(self, items):
        """`[(op, shape, model)]`: `op` 'box' with `shape` a world `(lo, hi)`, or 'point' with
        `shape` a world point, against `model`'s cached mesh -- in as few round trips as fit.

        A lower and an upper bound on a body's distance from `model` that need
        no BVH of the body itself (`BearingResolver`).
        """
        answers, group, shipping, size = [], [], set(), 0
        for item in items:
            x = item[2].local
            if x.key not in self.known and x.key not in shipping:
                shipping.add(x.key)
                size += x.vertices.nbytes + x.triangles.nbytes
            group.append(item)
            if size > self.BATCH_BYTES:
                answers += self._shape_batch(group)
                group, shipping, size = [], set(), 0
        return answers + (self._shape_batch(group) if group else [])

    def _shape_batch(self, items):
        for _ in range(3):
            meshes, keys = self._ship(model.local for _, _, model in items)
            message = {'op': 'shapes', 'keys': list(keys), 'meshes': meshes,
                       'ops': [op for op, _, _ in items],
                       'models': [keys[model.local.key] for _, _, model in items],
                       'poses': np.stack([np.asarray(model.matrix, dtype=float) for _, _, model in items]),
                       'shapes': np.stack([np.broadcast_to(np.asarray(shape, dtype=float), (2, 3))
                                           for _, shape, _ in items])}
            reply = self._exchange(message, meshes)
            if reply is not None:
                return reply['answer']
        raise RuntimeError('Coal cache synchronization failed')


_CLIENT = None


def coal_query(op, a, b):
    return coal_queries([(op, a, b)])[0]


def coal_queries(queries):
    """`coal_query` for `[(op, a, b)]`, in one round trip; distances are retained."""
    global _CLIENT
    if not queries:
        return []               # and no Coal to start, where none was needed
    if _CLIENT is None or _CLIENT.process.poll() is not None:
        _CLIENT = CoalClient()  # a resident outlives a Coal child that died
    answers = _CLIENT.exact_queries(queries)
    for (op, a, b), answer in zip(queries, answers):
        if op == 'distance':
            _CLIENT.remember(a, b, answer)
    return answers


def backend_stats():
    return dict(_CLIENT.stats) if _CLIENT is not None else {}


def relative_matrix(a, b):
    relative = np.eye(4)
    relative[:3, :3] = a.matrix[:3, :3].T @ b.matrix[:3, :3]
    relative[:3, 3] = a.matrix[:3, :3].T @ (b.matrix[:3, 3]-a.matrix[:3, 3])
    return relative


def rotation_quaternion(m):
    """Float64 proper rotation to canonical unit (w, x, y, z)."""
    trace = float(np.trace(m))
    if trace > 0:
        scale = math.sqrt(trace+1.)*2
        q = np.array([scale/4, (m[2,1]-m[1,2])/scale,
                      (m[0,2]-m[2,0])/scale, (m[1,0]-m[0,1])/scale])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i+1)%3, (i+2)%3
        scale = math.sqrt(max(0., 1.+m[i,i]-m[j,j]-m[k,k]))*2
        q = np.empty(4)
        q[0] = (m[k,j]-m[j,k])/scale
        q[i+1] = scale/4
        q[j+1] = (m[j,i]+m[i,j])/scale
        q[k+1] = (m[k,i]+m[i,k])/scale
    q /= np.linalg.norm(q)
    for value in q:
        if value != 0:
            if value < 0:
                q = -q
            break
    return q


def pair_key(a, b):
    """Relative-pose bins of `POSE_CACHE_QUANTUM` per translation/quaternion component.

    Two pairs in the same bin share a contact result. A bin is not a clearance
    proof or an absolute displacement bound. Geometry hashes still distinguish
    scale, deformation, and topology changes.
    """
    relative = relative_matrix(a, b)
    values = (*relative[:3, 3], *rotation_quaternion(relative[:3, :3]))
    return a.key, b.key, tuple(int(round(float(value)/POSE_CACHE_QUANTUM)) for value in values)


def _rotation_quaternions(m):
    """`rotation_quaternion` for a stack of rotations, branch for branch."""
    q = np.empty((len(m), 4))
    trace = np.trace(m, axis1=1, axis2=2)
    positive = trace > 0
    scale = np.sqrt(np.where(positive, trace, 0.) + 1.)*2
    q[positive] = np.stack([scale/4, (m[:, 2, 1]-m[:, 1, 2])/scale, (m[:, 0, 2]-m[:, 2, 0])/scale,
                            (m[:, 1, 0]-m[:, 0, 1])/scale], axis=1)[positive]
    pivot = np.argmax(np.diagonal(m, axis1=1, axis2=2), axis=1)
    for i in range(3):
        rows = ~positive & (pivot == i)
        if not rows.any():
            continue
        j, k, r = (i+1) % 3, (i+2) % 3, m[rows]
        scale = np.sqrt(np.maximum(0., 1.+r[:, i, i]-r[:, j, j]-r[:, k, k]))*2
        q[rows, 0] = (r[:, k, j]-r[:, j, k])/scale
        q[rows, i+1] = scale/4
        q[rows, j+1] = (r[:, j, i]+r[:, i, j])/scale
        q[rows, k+1] = (r[:, k, i]+r[:, i, k])/scale
    q /= np.linalg.norm(q, axis=1)[:, None]
    # The first nonzero component positive, as `rotation_quaternion` makes it.
    first = np.argmax(q != 0, axis=1)
    q[q[np.arange(len(q)), first] < 0] *= -1
    return q


def pair_keys(firsts, seconds):
    """`pair_key(firsts[k], seconds[k])` for every k, computed as arrays.

    A frame has thousands of candidate pairs, each needing a relative matrix,
    a quaternion, and seven roundings; as arrays they cost a fraction of the
    per-pair Python.
    """
    if not firsts:
        return []
    ma = np.array([body.matrix for body in firsts], dtype=np.float64)
    mb = np.array([body.matrix for body in seconds], dtype=np.float64)
    rotation = ma[:, :3, :3].transpose(0, 2, 1) @ mb[:, :3, :3]
    translation = np.einsum('pji,pj->pi', ma[:, :3, :3], mb[:, :3, 3]-ma[:, :3, 3])
    bins = np.round(np.concatenate([translation, _rotation_quaternions(rotation)], axis=1)
                    / POSE_CACHE_QUANTUM).astype(np.int64)
    return [(a.key, b.key, tuple(row)) for a, b, row in zip(firsts, seconds, bins.tolist())]


def bounded_box(lo, hi, matrix):
    """Conservative affine image of a *known* local AABB.

    Takes the local AABB rather than the vertices because it is frame-invariant
    while the matrix is not: a caller looping over frames reduces the vertex
    array once (or takes `Body.lo`/`Body.hi`, memoized per unique mesh) and pays
    only this 3x3 arithmetic per frame.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    center = matrix[:3, :3] @ ((lo+hi)/2) + matrix[:3, 3]
    half_axes = matrix[:3, :3] * ((hi-lo)/2)
    return center, half_axes


def boxes_overlap_many(first, second):
    """Oriented-box overlap for many pairs at once; `first[k]` against `second[k]`.

    The same separating-axis test, as arrays: a frame's oriented-box checks
    ran one Python call per candidate pair, about 1,400 a frame on the dog
    film (10.1 CPU s a 60-frame slice), for arithmetic that vectorizes.
    Returns one bool per pair, `True` where the boxes may overlap.
    """
    if not len(first):
        return np.zeros(0, dtype=bool)
    ca = np.array([box[0] for box in first], dtype=float)
    cb = np.array([box[0] for box in second], dtype=float)
    aa = np.array([box[1] for box in first], dtype=float)
    bb = np.array([box[1] for box in second], dtype=float)
    av, bv = aa.transpose(0, 2, 1), bb.transpose(0, 2, 1)
    axes = np.concatenate([np.cross(av, np.roll(av, 1, axis=1)),
                           np.cross(bv, np.roll(bv, 1, axis=1)),
                           np.cross(av[:, :, None, :], bv[:, None, :, :]).reshape(len(av), 9, 3)],
                          axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    valid = lengths > 1e-30
    axes = axes / np.where(valid, lengths, 1.)[:, :, None]
    radius = (np.abs(axes @ aa).sum(axis=2) + np.abs(axes @ bb).sum(axis=2))
    distance = np.abs(np.einsum('pkj,pj->pk', axes, cb - ca))
    return ~np.any(valid & (distance > radius + TOLERANCE), axis=1)


def broad_overlap(a_lo, a_hi, b_lo, b_hi):
    overlap = np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo)
    thin = np.minimum(a_hi-a_lo, b_hi-b_lo) <= 2*TOLERANCE
    return bool(np.all(np.where(thin, overlap >= -TOLERANCE, overlap > TOLERANCE)))


def separation_bound(a, b):
    """A sufficient axis translation to separate even fully contained bodies."""
    distances = np.concatenate([a.hi - b.lo, b.hi - a.lo])
    k = int(np.argmin(distances))
    axis = np.zeros(3)
    axis[k % 3] = 1 if k < 3 else -1  # translation of b
    return max(0., float(distances[k])), axis


def _strict_crossing(a, b, pairs):
    if not pairs:
        return False
    ids = np.asarray(pairs)
    ta, tb = a.vertices[a.triangles[ids[:, 0]]], b.vertices[b.triangles[ids[:, 1]]]
    na = np.cross(ta[:, 1]-ta[:, 0], ta[:, 2]-ta[:, 0])
    nb = np.cross(tb[:, 1]-tb[:, 0], tb[:, 2]-tb[:, 0])
    na /= np.maximum(np.linalg.norm(na, axis=1)[:, None], 1e-300)
    nb /= np.maximum(np.linalg.norm(nb, axis=1)[:, None], 1e-300)
    da = np.einsum('ijk,ik->ij', ta-tb[:, :1], nb)
    db = np.einsum('ijk,ik->ij', tb-ta[:, :1], na)
    # An intersection may lie exactly on a tessellation edge. Combine all
    # overlapping adjacent triangles against each fixed plane, so tessellating
    # a sheet cannot turn a proper crossing into a boundary-only contact.
    def straddles(distances, planes):
        _, inverse = np.unique(planes, return_inverse=True)
        lo = np.full(inverse.max()+1, np.inf)
        hi = np.full(inverse.max()+1, -np.inf)
        np.minimum.at(lo, inverse, distances.min(axis=1))
        np.maximum.at(hi, inverse, distances.max(axis=1))
        return (lo[inverse] < -TOLERANCE) & (hi[inverse] > TOLERANCE)
    return bool(np.any(straddles(da, ids[:,1]) & straddles(db, ids[:,0])))


def _closed_boxes(local):
    """Local boxes of a body's closed components -- the only ones with an inside -- or None."""
    if not hasattr(local, '_closed_boxes'):
        lo, hi = local.component_bound_arrays
        closed = local.component_witnesses[1]
        local._closed_boxes = (lo[closed], hi[closed]) if closed.any() else None
    return local._closed_boxes


def _candidate_components(source, target, enclosed_only):
    """Indices of `source`'s components whose witnesses `_contains` must test against `target`.

    The per-component tests, for all components at once, and one more: a witness can only be inside the target if it lies in the
    box of one of the target's closed components (the first vertex), or within
    6 um of one (the centroid, whose normal witnesses sit 4 um either side).
    Exact, and on a furred character's thousands of fur cards it drops most
    of them.
    """
    first, closed = source.component_witnesses
    faces = source.vertices[source.triangles[first]]                # (n, 3, 3)
    lo, hi = source.component_bound_arrays
    keep = np.ones(len(first), dtype=bool)
    if enclosed_only:
        keep &= ~((faces < target.lo).any(axis=(1, 2)) | (faces > target.hi).any(axis=(1, 2)))
    if len(first) > 1:
        # Every tested witness is on a triangle or at most 4 micrometers along
        # its unit normal. Disjoint expanded bounds prove that none can pass
        # target.inside; do not spend winding queries on them.
        pad = np.where(closed, 4*TOLERANCE, 0.)[:, None]
        keep &= ~((hi + pad < target.lo).any(axis=1) | (lo - pad > target.hi).any(axis=1))
    if enclosed_only:
        keep &= ~((lo < target.lo).any(axis=1) | (hi > target.hi).any(axis=1))
    boxes = _closed_boxes(target.local)
    if boxes is None:
        return np.flatnonzero(np.zeros(len(first), dtype=bool))
    if len(first) > 1 and keep.any():
        m = np.asarray(target.matrix, dtype=float)
        rows = np.flatnonzero(keep)

        def within(points, reach):
            return _within_boxes((points - m[:3, 3]) @ m[:3, :3], *boxes, reach)
        keep[rows] = within(faces[rows, 0], 0.) | within(faces[rows].mean(axis=1), 6*TOLERANCE)
    return np.flatnonzero(keep)


def _within_boxes(points, lo, hi, reach):
    """Per point, whether its Euclidean gap to the nearest box is `<= reach`.

    The same expression as over every box, on the boxes some point can reach.
    A box more than `2*reach + 1 nm` beyond the points' own box on an axis has
    a gap above `reach` from every point, so dropping it changes no answer.
    Thousands of fur witnesses against hundreds of boxes cost about half a
    second a pair densely, and only a few of the boxes are near. With a
    NaN anywhere the dense form stands, as its `min` is what NaN decides.
    """
    if np.isfinite(points).all() and np.isfinite(lo).all() and np.isfinite(hi).all():
        slack = 2*reach + 1e-9
        near = ~((lo - slack > points.max(axis=0)).any(axis=1)
                 | (hi + slack < points.min(axis=0)).any(axis=1))
        lo, hi = lo[near], hi[near]
        if not len(lo):
            return np.zeros(len(points), dtype=bool)
    answer = np.empty(len(points), dtype=bool)
    step = max(1, (1 << 20) // len(lo))
    for start in range(0, len(points), step):
        part = points[start:start + step]
        g = np.maximum(np.maximum(lo[None] - part[:, None], part[:, None] - hi[None]), 0.)
        answer[start:start + step] = np.sqrt((g ** 2).sum(axis=2)).min(axis=1) <= reach
    return answer


def _contains(a, b, *, enclosed_only=False):
    # With no proper crossings, each connected surface is wholly inside/outside.
    for source, target in ((a, b), (b, a)):
        first, closed_flags = source.component_witnesses
        for component in _candidate_components(source, target, enclosed_only):
            closed = bool(closed_flags[component])
            face = source.vertices[source.triangles[first[component]]]
            centroid = face.mean(axis=0)
            if target.inside(face[0]):
                return True
            inside, distance = target.inside_at(centroid)
            if inside:
                return True
            # Boundary-coincident solids need a strict interior witness, 4 um
            # either side of the centroid along the face normal. Neither can
            # be inside the target unless its surface comes within that of the
            # centroid, so a centroid clearly off it skips both, which was
            # most of `_contains`' cost: 5.6 inside tests a component, now ~2.
            if closed and distance <= 5*TOLERANCE:
                normal = np.cross(face[1]-face[0], face[2]-face[0])
                length = np.linalg.norm(normal)
                if length:
                    for sign in (-1, 1):
                        point = centroid + sign*normal/length*(4*TOLERANCE)
                        if target.inside(point) and source.inside(point):
                            return True
    return False


def collision(a, b, raw=None):
    """The pair's overlap verdict; `raw` is its `raw_overlap` answer, when already asked."""
    if not broad_overlap(a.lo, a.hi, b.lo, b.hi):
        return None
    pairs = coal_query('raw_overlap', a, b) if raw is None else raw
    crossing = False
    if pairs:
        erodible_pair = False
        for chord in (True, False):
            ea, eb = a.eroded(chord), b.eroded(chord)
            if ea is None or eb is None:
                continue
            erodible_pair = True
            # The tolerance has already been spent on erosion. Applying another
            # micron plane test here would silently double the contact budget.
            if not coal_query('overlap', ea, eb):
                if not _contains(a, b, enclosed_only=True):
                    return None
                break
        crossing = erodible_pair or _strict_crossing(a, b, coal_query('pairs', a, b))
    if crossing:
        kind = 'surface_crossing'
    elif _contains(a, b, enclosed_only=not pairs):
        kind = 'containment'
    else:
        return None
    bound, direction = separation_bound(a, b)
    return {'kind': kind, 'method': VERSION, 'separation_upper_bound_m': bound,
            'separation_direction': direction.tolist()}


def contact_normals(a, b):
    """Coal's distinct contact normals for an overlapping pair.

    Only directions are taken. `Contact.penetration_depth` alongside them is a
    per-triangle-pair quantity that returns the lateral extent of the
    intersection for coincident faces, so the distance is still bisected.
    """
    try:
        answer = coal_query('contacts', a, b)
    except RuntimeError:
        return []
    return [np.asarray(normal, dtype=float) for normal in (answer or [])]


def separating_ceiling(a, b, direction):
    """Translation of `b` along `direction` that provably separates the AABBs.

    The general form of `separation_bound`, which is this evaluated on the one
    axis it picks. Separation needs only a single axis to come apart, so the
    ceiling is the cheapest axis to clear: for each component that pushes the
    right way, how far along `direction` that component has to travel. Bounding
    the boxes bounds the meshes inside them, which is what keeps the bisection
    one-directional — it can only come in under this.
    """
    n = np.asarray(direction, dtype=float)
    reach = np.inf
    for i in range(3):
        if n[i] > 1e-12:
            reach = min(reach, (a.hi[i] - b.lo[i]) / n[i])
        elif n[i] < -1e-12:
            reach = min(reach, (b.hi[i] - a.lo[i]) / -n[i])
    return reach


def _bisect_clear(a, b, direction, hi, resolution):
    """Least translation in `[0, hi]` along `direction` that clears, or None if none does."""
    if not np.isfinite(hi) or hi <= 0:
        return None
    lo, cleared = 0., False
    for _ in range(24):
        if hi-lo <= resolution:
            break
        mid = (lo+hi)/2
        if collision(a, b.translated(np.asarray(direction)*mid)) is None:
            hi, cleared = mid, True
        else:
            lo = mid
    return hi if cleared else None


#: How finely a separating shift is resolved. It is an upper bound read as
#: "how far must this part move to come clear", which agents act on in
#: millimeters; resolving each candidate to a micron cost 24 collision queries
#: a direction and was 83% of the per-span diagnostic cost.
SEPARATION_RESOLUTION = 1e-3

#: Candidates the search starts from, beyond the axis bound: the best screened
#: contact normals, and the best other box axes.
_SEARCH_NORMALS = 2
_SEARCH_AXES = 2
_BOX_AXES = tuple(np.array(v, dtype=float) for v in
                  ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))


def directional_separation(a, b, direction, upper):
    """Least separating translation found, over pruned candidate directions.

    Candidates are the axis bound `direction` (which `upper` separates by
    construction), the two contact normals with the lowest box ceilings in
    either sign, and the two other box axes with the lowest ceilings, taken in
    ascending ceiling and with directions within 1 degree of a kept one
    dropped. Each is a genuine separating translation once it converges, so
    the least is the tightest upper bound these directions can certify --
    still an upper bound, since the true minimum may lie along a direction
    none of them names.

    The first is bisected to `SEPARATION_RESOLUTION`. Each later one is first
    probed once at the best shift so far less one resolution step: if it still
    collides there it cannot do better (the bisection already assumes
    clearance is monotone along a direction), and it costs one query instead
    of a bisection. Otherwise it is bisected below that shift.

    **Why these candidates.** The axis bound is not redundant: on colliding
    pairs the first contact normal is the worse direction about half the time,
    sometimes by orders of magnitude, and Coal occasionally gives no usable
    normal. With the box axes added, the winner is usually axis-aligned, and
    the two best other axes find a shorter shift on a sixth of spans. Against
    an exhaustive micron search, it is never worse beyond the resolution, at
    about a fifth of the time.
    """
    tol = SEPARATION_RESOLUTION
    axis = np.asarray(direction, dtype=float)
    axis = axis / np.linalg.norm(axis)
    screened = []
    for normal in contact_normals(a, b):
        for signed in (normal, -normal):
            ceiling = separating_ceiling(a, b, signed)
            if np.isfinite(ceiling) and ceiling > 0:
                screened.append((ceiling, signed))
    screened.sort(key=lambda candidate: candidate[0])
    boxes = [(separating_ceiling(a, b, unit), unit) for unit in _BOX_AXES
             if not np.allclose(unit, axis)]
    boxes = sorted(((c, unit) for c, unit in boxes if np.isfinite(c) and c > 0),
                   key=lambda candidate: candidate[0])
    candidates = sorted([(upper, axis)] + screened[:_SEARCH_NORMALS] + boxes[:_SEARCH_AXES],
                        key=lambda candidate: candidate[0])
    kept = []
    for ceiling, candidate in candidates:
        candidate = np.asarray(candidate, dtype=float) / np.linalg.norm(candidate)
        if all(float(candidate @ other) < math.cos(math.radians(1.0)) for _, other in kept):
            kept.append((ceiling, candidate))
    best = None
    for ceiling, candidate in kept:
        if best is None:
            cleared = _bisect_clear(a, b, candidate, ceiling + 2*tol, tol)
            if cleared is not None:
                best = (cleared, candidate)
            continue
        probe = best[0] - tol
        if probe <= 0 or collision(a, b.translated(candidate*probe)) is not None:
            continue
        cleared = _bisect_clear(a, b, candidate, probe, tol)
        best = (cleared if cleared is not None else probe, candidate)
    if best is None:
        return {'resolution_m': 0., 'depth_method': 'no_separating_translation_found'}
    return {'directional_separation_m': float(best[0]), 'resolution_m': tol,
            'separation_direction': [float(v) for v in best[1]],
            'depth_method': 'pruned_candidate_bisection_upper_bound'}


def surface_gap(a, b, *, stop_below=None):
    """Coal triangle-surface distance; stop_below is a compatible query hint."""
    gap = coal_query('distance', a, b)
    return {'gap_m': gap, 'resolution_m': TOLERANCE, 'method': VERSION}


def gap_predicates(pairs, threshold, *, separated=False):
    """`gap_predicate` for each pair, the distances it must measure in one round trip each stage.

    The same certificate as `gap_predicate`: a retained distance plus the
    bound on how far the pose has moved since settles most pairs with no
    query at all; the rest are measured on their rigid proxies together, and
    the few a proxy cannot settle on their actual geometry together.
    """
    global _CLIENT
    if not pairs:
        return []
    if _CLIENT is None or _CLIENT.process.poll() is not None:
        _CLIENT = CoalClient()
    answers = [None] * len(pairs)

    def settle(k, distance, delta):
        if distance + delta < threshold:
            answers[k] = not separated
        elif distance - delta > threshold:
            answers[k] = separated

    def certify(k, proxies, error):
        pa, pb = proxies
        if (pa.key, pb.key) in _CLIENT.distances:
            previous, distance = _CLIENT.distances[(pa.key, pb.key)]
            relative = relative_matrix(pa, pb)
            settle(k, distance, np.linalg.norm(relative[:3, :3]-previous[:3, :3], ord='fro')
                   * pb.local.radius + np.linalg.norm(relative[:3, 3]-previous[:3, 3]) + error + 4e-9)

    exact = list(range(len(pairs)))
    if math.isfinite(threshold) and threshold >= 0:
        proxies = [_CLIENT.proxies(a, b) for a, b in pairs]
        for k, (pair, error) in enumerate(proxies):
            certify(k, pair, error)
        pending = [k for k in range(len(pairs)) if answers[k] is None]
        measured = _CLIENT.measure_distances([proxies[k][0] for k in pending])
        for k, distance in zip(pending, measured):
            if proxies[k][1] == 0:
                # This was the actual geometry: threshold equality needs no
                # second identical distance query.
                answers[k] = distance >= threshold if separated else distance <= threshold
            else:
                certify(k, *proxies[k])
        exact = [k for k in range(len(pairs)) if answers[k] is None]
    for k, gap in zip(exact, _CLIENT.measure_distances([pairs[k] for k in exact])):
        answers[k] = gap >= threshold if separated else gap <= threshold
    return answers


def gap_predicate(a, b, threshold, *, separated=False):
    """Certified distance bounds settle thresholds; equality uses exact distance."""
    return gap_predicates([(a, b)], threshold, separated=separated)[0]


# -- the advisory checks' bearing tests ---------------------------------------

def _world(placed):
    """World vertices of a placed body, once per placed body (a frame's)."""
    held = getattr(placed, '_world_vertices', None)
    if held is None:
        m = np.asarray(placed.matrix, dtype=float)
        held = placed._world_vertices = (np.asarray(placed.local.vertices, dtype=float)
                                         @ m[:3, :3].T + m[:3, 3])
    return held


def _box_of(placed):
    held = getattr(placed, '_world_box', None)
    if held is None:
        w = _world(placed)
        held = placed._world_box = (w.min(axis=0), w.max(axis=0))
    return held


def _triangle_bounds(placed):
    """Per-triangle world boxes of a placed body."""
    held = getattr(placed, '_triangle_bounds', None)
    if held is None:
        world, tri = _world(placed), placed.local.triangles
        v0, v1, v2 = world[tri[:, 0]], world[tri[:, 1]], world[tri[:, 2]]
        held = placed._triangle_bounds = (np.minimum(np.minimum(v0, v1), v2),
                                          np.maximum(np.maximum(v0, v1), v2))
    return held


def _near(placed, lo, hi):
    """Which of a placed body's triangles have a world box meeting `[lo, hi]`."""
    tlo, thi = _triangle_bounds(placed)
    return np.all((thi >= lo) & (tlo <= hi), axis=1)


def _crop(placed, keep):
    """The kept triangles of a placed body, as a world-space body of their own."""
    world, tri = _world(placed), placed.local.triangles
    used = np.unique(tri[keep])
    remap = np.full(len(world), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return PlacedBody(Body(world[used], remap[tri[keep]]), np.eye(4))


def _centre(local):
    """`(centre, radius about it)` of a local mesh: how far a rotation moves one of its vertices."""
    held = getattr(local, '_centre', None)
    if held is None:
        verts = np.asarray(local.vertices, dtype=float)
        c = (verts.min(axis=0) + verts.max(axis=0)) / 2
        held = local._centre = (c, float(np.sqrt(((verts - c) ** 2).sum(axis=1).max())))
    return held


class BearingResolver:
    """Each weighed part's first bearing candidate -- exactly `gap_predicates`' answer, cheaper.

    The advisory checks ask, every frame and for every weighed part, which is
    the first of its ground candidates (`verify.ground_candidates` order)
    within `threshold`. Asked candidate by candidate, that was an exact Coal
    distance each, with the skinned meshes rebuilt every frame: 61 % of a
    repeat compile on the dog film. The answer is the same here, settled the
    cheapest way that proves it:

    * **Carried.** A bound measured on an earlier frame still holds if the two
      bodies cannot have moved enough to cross the threshold. With the same
      triangles, no point of a mesh moves further than its furthest vertex,
      so a distance changes by at most the two bodies' displacements -- taken
      absolutely, as a common translation plus each body's residual, or, for
      two rigid meshes, as the change in their relative pose. Most tests on a
      moving film are settled this way, with no query.
    * **Measured**, against whichever body Coal already holds: the other's box
      (a lower bound), then its triangles near the model -- their box, and the
      nearest vertex (an upper bound) -- and only then the exact distance from
      those triangles cropped to `REACH` of the model, which is the true
      distance whenever it is under `REACH`. A result within `EDGE` of the
      threshold is measured once more on the whole meshes, as
      `gap_predicates` does.

    Per process: the bounds are a cache, so a chunk, a re-check, and a fill
    each start empty and reach the same answers.
    """
    REACH = .03
    MARGIN = 1e-7
    EDGE = 1e-9

    def __init__(self):
        self.bounds, self.seen = {}, {}
        self.counts = {'tests': 0, 'carried': 0, 'measured': 0, 'exact': 0}

    def motion(self, name, placed, frame):
        """`((absolute, residual, translation), rigid)`: the motion of `name` summed to `frame`."""
        held = self.seen.get(name)
        if held is not None and held[0] == frame:
            return held[3], held[4]
        local, matrix = placed.local, np.asarray(placed.matrix, dtype=float)
        total, res, trans = (0., 0., np.zeros(3)) if held is None else held[3]
        step, residual, shift, rigid = 0., 0., np.zeros(3), False
        if held is not None:
            _, old_local, old_matrix, _, _ = held
            if local.key == old_local.key:
                rigid = True
                if not np.array_equal(matrix, old_matrix):
                    # A vertex v moves by dR (v - c) + (dR c + dt), and for
                    # rotations ||dR|| = 2 sin(angle / 2) = sqrt(3 - tr(R'^T R)).
                    c, r = _centre(local)
                    turn = matrix[:3, :3] - old_matrix[:3, :3]
                    shift = turn @ c + (matrix[:3, 3] - old_matrix[:3, 3])
                    residual = math.sqrt(max(0., 3. - float(np.einsum(
                        'ij,ij->', matrix[:3, :3], old_matrix[:3, :3])))) * r
                    step = residual + float(np.linalg.norm(shift))
            elif (old_local._triangle_data['digest'] == local._triangle_data['digest']
                  and len(old_local.vertices) == len(local.vertices)):
                # The same triangles on new vertices: each vertex against its
                # own earlier position.
                old = np.asarray(old_local.vertices, dtype=float) @ old_matrix[:3, :3].T + old_matrix[:3, 3]
                delta = _world(placed) - old
                shift = (delta.min(axis=0) + delta.max(axis=0)) / 2
                residual = float(np.sqrt(((delta - shift) ** 2).sum(axis=1).max()))
                step = residual + float(np.linalg.norm(shift))
            else:
                step = residual = math.inf
        cum = (total + step, res + residual, trans + shift)
        self.seen[name] = (frame, local, matrix, cum, rigid)
        return cum, rigid

    def resolve(self, parts, grounds, bodies, body, threshold, frame):
        """`{part index: identity it bears on, or None}`: the first ground candidate it bears on."""
        global _CLIENT
        names = [entry[0] for entry in bodies]
        bears = dict.fromkeys(parts)
        pairs = [(i, g) for i, gs in zip(parts, grounds) for g in gs]
        if not pairs:
            return bears
        if _CLIENT is None or _CLIENT.process.poll() is not None:
            _CLIENT = CoalClient()
        counts = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        # A name two bodies share cannot carry a bound: its motion is not one body's.
        tracked = {i for pair in pairs for i in pair if counts[names[i]] == 1}
        motion = {}
        for i in {x for pair in pairs for x in pair}:
            placed = body(i)
            placed._world_box = (np.asarray(bodies[i][3], dtype=float), np.asarray(bodies[i][4], dtype=float))
            motion[i] = self.motion(names[i], placed, frame) if i in tracked else ((math.inf,) * 2 + (np.zeros(3),), False)
        self.counts['tests'] += len(pairs)
        known = self._carried(pairs, names, body, motion, tracked, threshold)
        # Each part's candidates in order; the first undecided ones, of every
        # part at once, are measured.
        cursor = dict.fromkeys(parts, 0)
        ground_of = dict(zip(parts, grounds))
        live = [i for i in parts if ground_of[i]]
        while live:
            need = []
            for i in live:
                gs = ground_of[i]
                while cursor[i] < len(gs):
                    answer = known.get((i, gs[cursor[i]]))
                    if answer is None:
                        need.append((i, gs[cursor[i]]))
                        break
                    if answer:
                        bears[i] = names[gs[cursor[i]]]
                        break
                    cursor[i] += 1
            live = [i for i in live if bears[i] is None and cursor[i] < len(ground_of[i])]
            if not need:
                break
            for pair, answer in zip(need, self._measure(need, names, body, motion, tracked, threshold)):
                known[pair] = answer
        return bears

    def _carried(self, pairs, names, body, motion, tracked, threshold):
        """The tests an earlier bound still settles: `{pair: bool}`."""
        known = {}
        have = [(n, self.bounds[(names[i], names[g])]) for n, (i, g) in enumerate(pairs)
                if i in tracked and g in tracked and (names[i], names[g]) in self.bounds]
        if not have:
            return known
        lower = np.array([held[0] for _, held in have])
        value = np.array([held[1] for _, held in have])
        then_a = np.array([held[2] for _, held in have])
        then_b = np.array([held[3] for _, held in have])
        now_a = np.array([(motion[pairs[n][0]][0][0], motion[pairs[n][0]][0][1], *motion[pairs[n][0]][0][2])
                          for n, _ in have])
        now_b = np.array([(motion[pairs[n][1]][0][0], motion[pairs[n][1]][0][1], *motion[pairs[n][1]][0][2])
                          for n, _ in have])
        with np.errstate(invalid='ignore'):
            # A body whose motion could not be followed carries infinity, and
            # infinity less infinity is NaN: read as moved without bound.
            da, db = now_a - then_a, now_b - then_b
            moved = np.minimum(da[:, 0] + db[:, 0],
                               da[:, 1] + db[:, 1] + np.linalg.norm(da[:, 2:] - db[:, 2:], axis=1))
        moved = np.where(np.isnan(moved), np.inf, moved)
        # Two rigid meshes, the same ones as when the bound was measured: the
        # change in their relative pose bounds the change in their distance.
        rel = [k for k, (n, held) in enumerate(have) if held[4] is not None
               and held[4][0] == (body(pairs[n][0]).local.key, body(pairs[n][1]).local.key)]
        if rel:
            ma = np.stack([np.asarray(body(pairs[have[k][0]][0]).matrix, dtype=float) for k in rel])
            mb = np.stack([np.asarray(body(pairs[have[k][0]][1]).matrix, dtype=float) for k in rel])
            rr = np.einsum('nji,njk->nik', ma[:, :3, :3], mb[:, :3, :3])
            tr = np.einsum('nji,nj->ni', ma[:, :3, :3], mb[:, :3, 3] - ma[:, :3, 3])
            r0 = np.stack([have[k][1][4][1] for k in rel])
            t0 = np.stack([have[k][1][4][2] for k in rel])
            radius = np.array([have[k][1][4][3] for k in rel])
            turn = np.sqrt(np.maximum(0., 3. - np.einsum('nij,nij->n', rr, r0)))
            moved[rel] = np.minimum(moved[rel], turn * radius + np.linalg.norm(tr - t0, axis=1))
        settled = np.where(lower, value - moved > threshold + self.MARGIN,
                           value + moved <= threshold - self.MARGIN)
        for (n, _), good, low in zip(have, settled, lower):
            if good:
                known[pairs[n]] = not low
        self.counts['carried'] += int(settled.sum())
        return known

    def _store(self, names, pair, lower, value, motion, body, tracked):
        i, g = pair
        if i not in tracked or g not in tracked:
            return
        a, b = body(i), body(g)
        m = relative_matrix(a, b)
        cum_a, cum_b = motion[i][0], motion[g][0]
        self.bounds[(names[i], names[g])] = (
            lower, value, (cum_a[0], cum_a[1], *cum_a[2]), (cum_b[0], cum_b[1], *cum_b[2]),
            ((a.local.key, b.local.key), m[:3, :3].copy(), m[:3, 3].copy(), b.local.radius))

    def _measure(self, need, names, body, motion, tracked, threshold):
        t = threshold
        self.counts['measured'] += len(need)
        answers = [None] * len(need)
        sides = []
        for i, g in need:
            a, b = body(i), body(g)
            # Coal measures against a body it already holds, else a rigid or
            # static one; the other side is boxed or cropped.
            ka, kb = a.local.key in _CLIENT.known, b.local.key in _CLIENT.known
            if ka != kb:
                sides.append((a, b) if ka else (b, a))
            else:
                sides.append((b, a) if motion[g][1] or not motion[i][1] else (a, b))

        def settle(n, lower, value):
            answers[n] = not lower
            self._store(names, need[n], lower, value, motion, body, tracked)

        lows = _CLIENT.shape_batch([('box', _box_of(other), model) for model, other in sides])
        rest = []
        for n, low in enumerate(lows):
            if low > t + self.MARGIN:
                settle(n, True, low)
            else:
                rest.append(n)
        asked = []
        for n in rest:
            model, other = sides[n]
            mlo, mhi = _box_of(model)
            keep = _near(other, mlo - t, mhi + t)
            if not keep.any():
                # No triangle of it comes within the threshold of the model's box.
                settle(n, True, t + self.MARGIN)
                continue
            tlo, thi = _triangle_bounds(other)
            corners = _world(other)[other.local.triangles[keep]].reshape(-1, 3)
            gap = np.linalg.norm(np.maximum(np.maximum(mlo - corners, corners - mhi), 0.), axis=1)
            asked.append((n, (tlo[keep].min(axis=0), thi[keep].max(axis=0)), corners[np.argmin(gap)]))
        both = _CLIENT.shape_batch([q for n, box, point in asked
                                    for q in (('box', box, sides[n][0]), ('point', point, sides[n][0]))])
        exact = []
        for m, (n, _, _) in enumerate(asked):
            low, high = both[2 * m], both[2 * m + 1]
            if low > t + self.MARGIN:
                settle(n, True, low)
            elif high <= t - self.MARGIN:
                settle(n, False, high)
            else:
                exact.append(n)
        crops, whole = [], []
        R = self.REACH
        for n in exact:
            # Only the other side is cropped -- the model stays the one Coal
            # holds -- to within REACH of the model's triangles near it. Any
            # pair of points within REACH survives every round.
            model, other = sides[n]
            mlo, mhi = _box_of(model)
            x = _near(other, mlo - R, mhi + R)
            y = None
            for _ in range(2):
                xlo, xhi = _triangle_bounds(other)[0][x].min(axis=0), _triangle_bounds(other)[1][x].max(axis=0)
                y = _near(model, xlo - R, xhi + R)
                if not y.any():
                    break
                ylo, yhi = _triangle_bounds(model)[0][y].min(axis=0), _triangle_bounds(model)[1][y].max(axis=0)
                x = x & _near(other, ylo - R, yhi + R)
                if not x.any():
                    break
            if not x.any() or y is None or not y.any():
                settle(n, True, R)
            else:
                crops.append((n, _crop(other, x), model))
        distances = _CLIENT.exact_queries([('distance', x, y) for _, x, y in crops]) if crops else []
        self.counts['exact'] += len(crops)
        for (n, _, _), d in zip(crops, distances):
            if abs(d - t) <= self.EDGE:
                whole.append(n)          # too close to call on a crop: the whole meshes decide
            elif d <= R:
                settle(n, d > t, d)
            else:
                settle(n, True, R)
        if whole:
            for n, d in zip(whole, _CLIENT.measure_distances([(body(need[n][0]), body(need[n][1]))
                                                              for n in whole])):
                settle(n, d > t, d)
        return answers
