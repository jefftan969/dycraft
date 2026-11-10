"""Persistent Coal process; triangles stay cached while query poses change.

Runs in the harness interpreter because Blender 4.5 and Coal wheels require
incompatible NumPy versions. Only this module is mounted into the sandbox.
"""
from collections import OrderedDict
import pickle
import struct
import sys
import traceback
import time


def receive(stream):
    header = stream.read(8)
    if not header:
        raise EOFError
    if len(header) != 8:
        raise RuntimeError('truncated collision message')
    length, = struct.unpack('!Q', header)
    if length > 512 * 1024**2:
        raise ValueError('collision message exceeds 512 MiB')
    data = stream.read(length)
    if len(data) != length:
        raise RuntimeError('truncated collision payload')
    return pickle.loads(data)


def send(stream, value):
    data = pickle.dumps(value, protocol=4)  # NumPy 1.x / 2.x compatible protocol.
    stream.write(struct.pack('!Q', len(data)))
    stream.write(data)
    stream.flush()


def extend_path():
    """Re-add the import roots an isolated start misses.

    `CoalClient` starts this file with `-I`, which ignores a user site and
    `PYTHONPATH`. `runtime.command` (or an offline caller) names the roots coal
    and NumPy actually came from in `DYCRAFT_COAL_SITEDIRS`, and `addsitedir`
    also processes their `.pth` files -- which is how pip's `coal` wheel puts
    its own `cmeel.prefix` tree on the path. Called from `serve`, never at
    import: `collision.py` imports this module inside Blender's Python too, and
    those roots carry a NumPy that Python must not see.
    """
    import os
    import site
    for root in os.environ.get('DYCRAFT_COAL_SITEDIRS', '').split(os.pathsep):
        if root:
            site.addsitedir(root)


def serve():
    extend_path()
    import coal
    import numpy as np
    # The floor, not a pin: the whole suite passes on 3.0.1, 3.0.2 and 3.0.3,
    # every coal 3 release there is.
    if tuple(int(x) for x in coal.__version__.split('.')[:3]) < (3, 0, 1):
        raise RuntimeError(f'native animation collision requires coal>=3.0.1 (found {coal.__version__})')
    cache = OrderedDict()
    triangles = OrderedDict()
    used = 0
    triangle_bytes = 0
    stats = {'builds': 0, 'queries': 0, 'cache_hits': 0, 'build_seconds': 0., 'query_seconds': 0.}
    limit = 512 * 1024**2
    def shape_distance(op, model, pose, shape):
        """Distance from a world box (`shape` = lo, hi) or point (`shape[0]`) to a cached model.

        The bearing resolver's certificates (`collision.BearingResolver`): a
        box around one body's triangles bounds that body's distance from
        below, one of its vertices from above, and neither needs the body's
        own BVH -- which for a skinned mesh is a new one every frame.
        """
        req, result = coal.DistanceRequest(), coal.DistanceResult()
        req.enable_signed_distance = False
        req.abs_err = req.rel_err = 0.
        req.gjk_tolerance = 1e-9
        if op == 'box':
            lo, hi = np.asarray(shape[0], dtype=float), np.asarray(shape[1], dtype=float)
            body = coal.Box(*np.maximum(hi - lo, 1e-9))
            where = coal.Transform3s(np.eye(3), (lo + hi) / 2)
            pad = 0.
        else:
            body, pad = coal.Sphere(1e-6), 1e-6
            where = coal.Transform3s(np.eye(3), np.asarray(shape[0], dtype=float))
        distance = float(coal.distance(body, where, model, pose, req, result)) + pad
        if not np.isfinite(distance):
            raise ValueError('nonfinite Coal distance')
        return max(0., distance)

    def answer(op, models, poses):
        """One query's answer: `op` on two cached models at two poses."""
        if op in ('overlap', 'raw_overlap'):
            req, result = coal.CollisionRequest(), coal.CollisionResult()
            req.gjk_tolerance = 1e-9
            req.num_max_contacts = 1
            req.enable_contact = False
            req.security_margin = -1e-6 if op == 'raw_overlap' else 0.0
            coal.collide(models[0], poses[0], models[1], poses[1], req, result)
            answer = bool(result.isCollision())
        elif op == 'pairs':
            # Every colliding triangle pair, as (triangle of the first
            # model, triangle of the second), at `raw_overlap`'s margin:
            # what the strict-crossing test reads. This reuses the cached
            # local model at any pose, where a Blender BVH of the placed
            # body's world-space vertices would be rebuilt for every pose.
            req, result = coal.CollisionRequest(), coal.CollisionResult()
            req.gjk_tolerance = 1e-9
            req.num_max_contacts = 1000000
            req.enable_contact = False
            req.security_margin = -1e-6
            coal.collide(models[0], poses[0], models[1], poses[1], req, result)
            answer = [(int(result.getContact(i).b1), int(result.getContact(i).b2))
                      for i in range(result.numContacts())]
        elif op == 'contacts':
            # Distinct SAT contact normals, for a caller that needs
            # directions to separate along. Only normals are taken:
            # `Contact.penetration_depth` beside them is a
            # per-triangle-pair quantity that returns the lateral extent
            # of the intersection for coincident faces, which is why the
            # distance is bisected instead.
            #
            # Several, not one, because a single triangle pair's normal is
            # a poor summary of a contact patch: the first one is a worse
            # separating direction than the plain bounding-box axis about
            # half the time. The caller screens them before paying for any.
            req, result = coal.CollisionRequest(), coal.CollisionResult()
            req.gjk_tolerance = 1e-9
            req.num_max_contacts = 16
            req.enable_contact = True
            req.security_margin = 0.0
            coal.collide(models[0], poses[0], models[1], poses[1], req, result)
            answer = []
            for index in range(result.numContacts()):
                normal = np.asarray(result.getContact(index).normal, dtype=float)
                length = float(np.linalg.norm(normal))
                if not np.isfinite(length) or length <= 1e-12:
                    continue
                unit = normal/length
                # Near-duplicates are common across a contact patch and
                # each one the caller keeps costs a bisection.
                if not any(float(unit @ np.asarray(kept)) > 0.999 for kept in answer):
                    answer.append(unit.tolist())
        elif op == 'distance':
            req, result = coal.DistanceRequest(), coal.DistanceResult()
            req.enable_signed_distance = False
            req.abs_err = req.rel_err = 0.
            req.gjk_tolerance = 1e-9
            answer = max(0., float(coal.distance(models[0], poses[0], models[1], poses[1], req, result)))
            if not np.isfinite(answer):
                raise ValueError('nonfinite Coal distance')
        else:
            raise ValueError('unknown collision operation')
        return answer

    send(sys.stdout.buffer, {'version': coal.__version__})
    while True:
        try:
            request = receive(sys.stdin.buffer)
        except EOFError:
            return
        try:
            evicted, evicted_topologies = [], []
            built = set()
            # A mesh arrives as (key, vertices, triangles or None, topology):
            # the triangles only the first time this worker sees a topology,
            # so a deforming mesh's next pose ships its vertices alone (fix 5).
            meshes = request.get('meshes', [])
            unknown = [m[3] for m in meshes if m[2] is None and m[3] not in triangles]
            if unknown:
                send(sys.stdout.buffer, {'missing_topology': unknown})
                continue
            for key, vertices, faces, topology in meshes:
                if key in cache:
                    continue
                started = time.perf_counter()
                if topology not in triangles:
                    tt = coal.StdVec_Triangle()
                    for a, b, c in faces:
                        tt.append(coal.Triangle(int(a), int(b), int(c)))
                    size = len(faces) * 48
                    # Never a topology this request still has to build from.
                    needed = {m[3] for m in meshes}
                    for old_topology in [t for t in triangles if t not in needed]:
                        if triangle_bytes + size <= 64 * 1024**2 and len(triangles) < 64:
                            break
                        triangle_bytes -= triangles.pop(old_topology)[1]
                        evicted_topologies.append(old_topology)
                    triangles[topology] = tt, size, len(faces)
                    triangle_bytes += size
                tt, _, n_faces = triangles[topology]
                triangles.move_to_end(topology)
                vv = coal.StdVec_Vec3s()
                vv.extend(vertices)
                # Built, not refit from the same object's previous pose. A
                # refit keeps the hierarchy of the pose it was built for, and
                # on a skinned mesh that hierarchy stops fitting, so its
                # queries get slower by more than its builds are cheaper.
                model = coal.BVHModelOBBRSS()
                model.beginModel(n_faces, len(vertices))
                model.addSubModel(vv, tt)
                model.endModel()
                stats['builds'] += 1
                size = vertices.nbytes * 8 + n_faces * 288
                cache[key] = model, size
                used += size
                built.add(key)
                stats['build_seconds'] += time.perf_counter()-started
            keys = request['keys']
            missing = [key for key in keys if key not in cache]
            if missing:
                send(sys.stdout.buffer, {'missing': missing})
                continue
            for key in keys:
                cache.move_to_end(key)
                if key not in built:
                    stats['cache_hits'] += 1
            # `batch` answers a list of queries in one round trip: a frame's
            # queries were ~1,000 messages at 300-570 us each on a loaded box,
            # against a 27-48 us floor (fix 6).
            if request['op'] == 'shapes':
                # Box and point distances, array-packed: `ops`, `models` (indices
                # into `keys`), `poses` (n, 4, 4) and `shapes` (n, 2, 3).
                started = time.perf_counter()
                out = [shape_distance(op, cache[keys[m]][0],
                                      coal.Transform3s(pose[:3, :3], pose[:3, 3]), shape)
                       for op, m, pose, shape in zip(request['ops'], request['models'],
                                                      request['poses'], request['shapes'])]
                stats['queries'] += len(out)
                stats['query_seconds'] += time.perf_counter() - started
                while used > limit and cache:
                    key, (_, size) = cache.popitem(last=False)
                    used -= size
                    evicted.append(key)
                send(sys.stdout.buffer, {'answer': out, 'evicted': evicted,
                                         'evicted_topologies': evicted_topologies, 'stats': stats})
                continue
            queries = (request['queries'] if request['op'] == 'batch'
                       else [{'op': request['op'], 'keys': keys, 'poses': request['poses']}])
            answers = []
            for query in queries:
                started = time.perf_counter()
                poses = [coal.Transform3s(np.asarray(m)[:3, :3], np.asarray(m)[:3, 3])
                         for m in query['poses']]
                answers.append(answer(query['op'], [cache[key][0] for key in query['keys']], poses))
                stats['queries'] += 1
                stats['query_seconds'] += time.perf_counter()-started
            answer_out = answers if request['op'] == 'batch' else answers[0]
            while used > limit and cache:
                key, (_, size) = cache.popitem(last=False)
                used -= size
                evicted.append(key)
            send(sys.stdout.buffer, {'answer': answer_out, 'evicted': evicted,
                                     'evicted_topologies': evicted_topologies, 'stats': stats})
        except Exception:
            send(sys.stdout.buffer, {'error': traceback.format_exc()})


if __name__ == '__main__':
    serve()
