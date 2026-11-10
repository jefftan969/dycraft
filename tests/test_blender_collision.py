"""Native geometry regression cases; generated fixtures never depend on runs/."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _coal_env():
    """Child env naming an interpreter that has `coal`.

    Without this the child inherits no `DYCRAFT_COAL_PYTHON` and
    `collision._Client` guesses a conda env named `dycraft`; where none exists
    it falls back to its own interpreter, and the blender interpreter has no
    `coal`. The test runner's interpreter does.
    """
    return dict(os.environ,
                DYCRAFT_COAL_PYTHON=os.environ.get("DYCRAFT_COAL_PYTHON")
                or sys.executable)

def test_native_collision_analytic_cases(tmp_path, pkgroot):
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    script = r'''
import bpy
import numpy as np
from dycraft.animation.blender.collision import Body, collision, surface_gap, gap_predicate, separation_bound, GeometryCache
v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
a=Body(v,t)
# The triangulated cube matches the legacy one-quantum policy, including
# 1.1 micrometre overlap; erosion is not a second full clearance allowance.
for shift, hit in [(2,False),(1,False),(1+2e-6,False),(1-1e-7,False),(.5,True),(1-5e-6,True),(1-1.1e-6,True),(1-1.8e-6,True),(0,True)]:
    assert (collision(a,Body(v+[shift,0,0],t)) is not None) == hit, shift
inside=Body(v*.2+.2,t)
assert collision(a,inside)
# `kind` routes the agent's repair, so both labels have to stay reachable and
# distinct. Containment is reached through the empty-`raw_overlap` path: a body
# strictly inside another has no intersecting triangles at all, so the erosion
# loop — which reads as though it forced `surface_crossing` for any erodible
# pair — never runs for it.
assert collision(a,inside)['kind']=='containment'
assert collision(a,Body(v+[.5,0,0],t))['kind']=='surface_crossing'
# Coincident boundaries stay a crossing: erosion moves vertices by one
# tolerance, which does not separate two shared faces.
assert collision(a,Body(v*.8+.2,t))['kind']=='surface_crossing'
bound,direction=separation_bound(a,inside)
assert abs(bound-.4)<1e-10  # .2 intersection extent is NOT a separation bound
assert collision(a,Body(inside.vertices+direction*(bound+1e-5),t)) is None
# Disconnected components: full object's AABB is not contained.
multi=Body(np.concatenate([v*.1+.3,v*.1+[2,0,0]]),np.concatenate([t,t+8]))
assert collision(a,multi)
# Hollow solid with inward-wound cavity must stay hollow.
hollow=Body(np.concatenate([v*3,v+1]),np.concatenate([t,t[:,::-1]+8]))
assert collision(hollow,Body(v*.2+1.2,t)) is None
assert collision(hollow,Body(v*.2+.2,t))
# Concave L: two touching boxes form the two limbs; the missing corner is clear.
ell=Body(np.concatenate([v*[2,1,1],v+[0,1,0]]),np.concatenate([t,t+8]))
assert collision(ell,Body(v*.2+[1.3,1.3,.3],t)) is None
assert collision(ell,Body(v*.2+[.3,1.3,.3],t))
# Reflections and non-uniform scales operate on world triangles.
assert collision(a,Body(v*[-.2,.3,.4]+[.6,.2,.2],t))
# Open crossing sheets; no volume is inferred for parallel/coplanar sheets.
sheet=Body(np.array([[-1,-1,0],[1,-1,0],[0,1,0.]]),np.array([[0,1,2]]))
cross=Body(np.array([[0,-.5,-1],[0,-.5,1],[0,.5,0.]]),np.array([[0,1,2]]))
assert collision(sheet,cross)
# Same crossing on an internal triangulation edge, including a giant sheet whose
# vertices and centroid lie outside the cube (containment cannot rescue a miss).
split=Body(np.array([[-1,-1,0],[1,1,0],[-1,1,0],[1,-1,0.]]),np.array([[0,1,2],[0,3,1]]))
cut=Body(np.array([[-2,-2,-1],[2,2,-1],[2,2,1],[-2,-2,1.]]),np.array([[0,1,2],[0,2,3]]))
assert collision(split,cut)

assert collision(sheet,Body(sheet.vertices+[0,0,.01],sheet.triangles)) is None
assert collision(sheet,sheet) is None
# Preserve the existing conservative tolerance policy for an open partner:
# erosion can clear a contact only when BOTH bodies admit an inward erosion.
open_near_face=Body(np.array([[1-1.5e-6,-.1,-.1],[1-1.5e-6,1.1,-.1],
                             [1-1.5e-6,1.1,1.1],[1-1.5e-6,-.1,1.1]]),
                   np.array([[0,1,2],[0,2,3]]))
assert collision(a,open_near_face)

# Minimum distance in the interior of perpendicular edges, not at a vertex.
x=Body(np.array([[-1,0,0],[1,0,0],[-1,-1,0.]]),np.array([[0,1,2]]))
y=Body(np.array([[0,-1,1],[0,1,1],[1,1,2.]]),np.array([[0,1,2]]))
assert abs(surface_gap(x,y)['gap_m']-1)<1e-6
assert abs(surface_gap(a,Body(v+[1.000003,0,0],t))['gap_m']-3e-6)<2e-7
# Threshold queries preserve equality, zero distance, containment surface gaps,
# and edge-interior closest points. Reused triangles must not reuse vertices.
for left,right in [(a,inside),(a,Body(v+[1.01,0,0],t)),(x,y),(sheet,cross),
                   (a,Body(v+[1,0,0],t))]:
    gap=surface_gap(left,right)['gap_m']
    for threshold in [0.,max(0.,gap-1e-7),gap,gap+1e-7,10.]:
        assert gap_predicate(left,right,threshold) == (gap <= threshold)
        assert gap_predicate(left,right,threshold,separated=True) == (gap >= threshold)
assert a._triangle_data is inside._triangle_data
assert Body(v,t[:,::-1])._triangle_data is not a._triangle_data
# Far disconnected shells can be pruned, while the one enclosed component
# and near-boundary witness still reach the original containment algorithm.
assert collision(a,multi)
assert collision(hollow,Body(v*.2+1.2,t)) is None
rng=np.random.default_rng(123)
for _ in range(50):
    lo=rng.uniform(-1,1,3); size=rng.uniform(.05,2,3)
    expected=np.all(np.minimum(1,lo+size)-np.maximum(0,lo)>1e-5)
    assert bool(collision(a,Body(v*size+lo,t))) == expected, (lo,size)
cache=GeometryCache()
assert cache.get(v,t) is cache.get(v.copy(),t.copy())
assert cache.get(v+[.1,0,0],t) is not cache.get(v,t)
assert cache.get(v,t[:,::-1]) is not cache.get(v,t)
assert cache.hits >= 3
from dycraft.animation.blender.verify import verify
record={'frame':0,'objects':[{'name':'sheet','matrix':np.eye(4).tolist()}],'transforms':{}}
info={'fps':1,'n_frames':2,'scene_ref':'s','declarations':{}}
try:
    verify('.',{'identities':{},'joints':{}},{'id':'s','instances':[]},{},info=info,
           records=iter([(record,{'sheet':(sheet.vertices,sheet.triangles)})]))
    raise AssertionError('incomplete stream was accepted')
except ValueError as error:
    assert 'incomplete frame stream' in str(error)

'''
    source = tmp_path/'cases.py'
    source.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+script)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True, timeout=60, env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_evaluation_cache_deduplicates_and_detects_tampering(tmp_path):
    from dycraft.animation.blender.evaluation import write_mesh, read_local, frame_signature, check_replay_frame
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    t=np.array([[0,1,2]],dtype=np.int32)
    known=set()
    key=write_mesh(tmp_path,v,t,known)
    assert write_mesh(tmp_path,v.copy(),t.copy(),known)==key
    assert len(list((tmp_path/'meshes').glob('*.npz')))==1
    record={'frame':0,'objects':[{'name':'a','mesh':key,'matrix':np.eye(4).tolist()}],'transforms':{}}
    local=read_local(tmp_path,record)
    record['signature']=frame_signature(record,local)
    (tmp_path/'000000.json').write_text(json.dumps(record))
    check_replay_frame(tmp_path,0,record,local)
    changed={**record,'objects':[{**record['objects'][0],'hidden':True}]}
    with pytest.raises(ValueError,match='replay changed'):
        check_replay_frame(tmp_path,0,changed,local)
    np.savez(tmp_path/'meshes'/f'{key}.npz',vertices=v+1,triangles=t)
    with pytest.raises(ValueError,match='hash mismatch'):
        read_local(tmp_path,record)
    record['objects'][0]['mesh']='../escape'
    with pytest.raises(ValueError,match='invalid evaluated'):
        read_local(tmp_path,record)


def test_legacy_and_shared_cache_replay_agree(tmp_path):
    from dycraft.animation.blender.evaluation import write_mesh, read_local, check_replay_frame
    a=tmp_path/'old'; a.mkdir()
    b=tmp_path/'new'; b.mkdir()
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    t=np.array([[0,1,2]],dtype=np.int32)
    old={'frame':0,'objects':[{'name':'a','key':'o0','matrix':np.eye(4).tolist()}],'transforms':{}}
    np.savez(a/'000000.npz',o0_v=v,o0_t=t)
    (a/'000000.json').write_text(json.dumps(old))
    new={**old,'objects':[{**old['objects'][0],'mesh':write_mesh(b,v,t,set())}]}
    check_replay_frame(a,0,new,read_local(b,new))


def test_native_matches_legacy_on_rotated_closed_solids(tmp_path, pkgroot):
    from dycraft.verify._narrow_phase import build_bvh, penetration
    from dycraft.sdk._rotation import IDENTITY_SE3
    from dycraft.render._blender_env import blender_python
    from scipy.spatial.transform import Rotation
    interpreter=blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
    t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    a=build_bvh(v,t)
    rng=np.random.default_rng(79)
    cases=[]; expected=[]
    for _ in range(32):
        r=Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        b=(v-.5)*rng.uniform(.1,1.5,3)@r.T+rng.uniform(-.5,1.5,3)
        cases.append(b.tolist())
        expected.append(penetration(a,IDENTITY_SE3,build_bvh(b,t),IDENTITY_SE3) is not None)
    source=tmp_path/'inputs.json'; source.write_text(json.dumps([v.tolist(),t.tolist(),cases]))
    output=tmp_path/'output.json'
    code='''import bpy, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from dycraft.animation.blender.collision import Body, collision
v,t,cases=json.loads(Path(sys.argv[1]).read_text())
a=Body(v,t)
Path(sys.argv[2]).write_text(json.dumps([collision(a,Body(b,t)) is not None for b in cases]))
'''
    completed=subprocess.run([interpreter,'-c',code,str(source),str(output),str(pkgroot)],capture_output=True,text=True,timeout=60,env=_coal_env())
    assert completed.returncode==0,completed.stderr
    assert json.loads(output.read_text())==expected


def test_coal_placed_geometry_and_cache(tmp_path, pkgroot):
    """Affine/deformed meshes and the relative-pose cache bins."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    code = r'''
import bpy, numpy as np
from dycraft.animation.blender.collision import (
    Body, GeometryCache, collision, surface_gap, gap_predicate, pair_key,
    bounded_box, boxes_overlap_many)
v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
cache=GeometryCache()
a=cache.placed(v,t,np.eye(4))
m=np.eye(4);m[:3,3]=[1.01,0,0]
b=cache.placed(v,t,m)
assert a.local is b.local
assert abs(surface_gap(a,b)['gap_m']-.01)<1e-9
key=pair_key(a,b)
m[:3,3]+=[10,2,3]
n=np.eye(4);n[:3,3]=[10,2,3]
assert abs(surface_gap(cache.placed(v,t,n),cache.placed(v,t,m))['gap_m']-.01)<1e-9
# Exact common translation retains relative key when arithmetic is exact.
p=np.eye(4);p[0,3]=2
q=p.copy();q[0,3]=3
r=np.eye(4);r[0,3]=1
assert pair_key(cache.placed(v,t,p),cache.placed(v,t,q)) == pair_key(a,cache.placed(v,t,r))
r[0,3]+=1e-7
assert pair_key(a,cache.placed(v,t,r)) != pair_key(a,cache.placed(v,t,np.eye(4)))
assert cache.placed(v+.001,t,np.eye(4)).local is not a.local
assert cache.placed(v,t[:,::-1],np.eye(4)).local is not a.local
# Float32 rigid rotation noise must reuse the original local geometry. Genuine
# affine changes and evaluated mesh deformation must still change geometry.
from mathutils import Euler
for angle in (.1,.7,1.4,2.6):
    m=np.eye(4);m[:3,:3]=np.asarray(Euler((angle,.3,-.4)).to_matrix())
    posed=cache.placed(v,t,m)
    assert posed.local is a.local
    assert np.max(np.linalg.norm(posed.vertices-v@m[:3,:3].T,axis=1)) <= 1e-7
for linear in (np.diag([1.00001,1,1]), np.diag([-1,1,1]),
               np.array([[1,1e-5,0],[0,1,0],[0,0,1]])):
    m=np.eye(4);m[:3,:3]=linear
    posed=cache.placed(v,t,m)
    assert posed.local is not a.local
    assert np.allclose(posed.vertices,v@linear.T,rtol=0,atol=2e-15)
# The absolute displacement guard also protects large-coordinate geometry.
m=np.eye(4);m[0,0]+=3e-8
large=cache.placed(v*100,t,m)
assert np.allclose(large.vertices,(v*100)@m[:3,:3].T,rtol=0,atol=1e-13)
deformed=v.copy();deformed[0,0]+=1e-8
assert cache.placed(deformed,t,m).local is not cache.placed(v,t,m).local
# Legacy component quantization reuses contacts without asking for a gap.
from dycraft.animation.blender import collision as module
m=np.eye(4);m[0,3]=1.
contact_key=pair_key(a,cache.placed(v,t,m))
m[0,3]+=2e-7
assert pair_key(a,cache.placed(v,t,m)) == contact_key
m[0,3]+=5e-7
assert pair_key(a,cache.placed(v,t,m)) != contact_key
m[0,3]=1.
assert pair_key(a,cache.placed(v+[0,0,1e-8],t,m)) != contact_key
m[0,0]=1.001
assert pair_key(a,cache.placed(v,t,m)) != contact_key
# Quaternion sign and near-180-degree branches follow the canonical legacy bins.
from dycraft.verify._exact_adjudicator import _relative_pose_key
for angle in (0.,.2,np.pi,3.4):
    m=np.eye(4);c,s=np.cos(angle),np.sin(angle)
    m[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];m[:3,3]=[.12,-.3,.6]
    wxyz=(np.cos(angle/2),0.,0.,np.sin(angle/2))
    if wxyz[0]<0:wxyz=tuple(-x for x in wxyz)
    assert pair_key(a,cache.placed(v,t,m))[2] == _relative_pose_key(
        ((0.,0.,0.),(1.,0.,0.,0.)), (tuple(m[:3,3]),wxyz))
# All quaternion conversion branches agree with the legacy key implementation.
from dycraft.sdk._rotation import sign_canonicalize_quat
rng=np.random.default_rng(91)
for _ in range(100):
    q=rng.normal(size=4);q/=np.linalg.norm(q)
    w,x,y,z=sign_canonicalize_quat(q)
    m=np.eye(4)
    m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    m[:3,3]=rng.normal(size=3)
    assert pair_key(a,cache.placed(v,t,m))[2] == _relative_pose_key(
        ((0.,0.,0.),(1.,0.,0.,0.)), (tuple(m[:3,3]),(w,x,y,z)))
rng=np.random.default_rng(142)
for _ in range(25):
    u,_,vt=np.linalg.svd(rng.normal(size=(3,3)))
    rotation=u@vt
    m=np.eye(4)
    m[:3,:3]=rotation@np.array([[.4,.1,0],[0,-.3,.15],[0,0,.7]])
    m[:3,3]=rng.uniform(-.2,1.2,3)
    posed=cache.placed(v,t,m)
    exact=v@m[:3,:3].T+m[:3,3]
    assert np.allclose(posed.vertices,exact,rtol=0,atol=2e-15)
    direct=Body(exact,t)
    assert bool(collision(a,posed)) == bool(collision(a,direct))
    gap=surface_gap(a,posed)['gap_m']
    assert abs(gap-surface_gap(a,direct)['gap_m'])<1e-8
    for threshold in (0.,gap,gap+1e-7,max(0.,gap-1e-7),.01):
        assert gap_predicate(a,posed,threshold) == (gap <= threshold)
        assert gap_predicate(a,posed,threshold,separated=True) == (gap >= threshold)
    # The oriented-box prefilter is conservative: it never rejects a pair that collides.
    if collision(a,posed):
        lo,hi=v.min(axis=0),v.max(axis=0)
        assert boxes_overlap_many([bounded_box(lo,hi,np.eye(4))],[bounded_box(lo,hi,m)])[0]
# Near-contact GJK regression: the default 1e-6 solver tolerance can invent
# collisions for eroded coplanar triangles with a sub-micron positive gap.
from dycraft.animation.blender.collision import coal_query
x=Body(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
y=Body(x.vertices+[0,0,4e-7],x.triangles)
assert not coal_query('overlap',x,y)
from dycraft.animation.blender import collision as module
assert tuple(int(x) for x in module._CLIENT.version.split('.')[:3]) >= (3, 0, 1)
assert module._CLIENT.stats['queries'] > 0
# Direct queries still detect real intersections independently of pose caching.
for shift in (1.01, 1.000004, 1.0000004, 1., .999995, .7, 1.2):
    m=np.eye(4);m[0,0]=1.+3e-8;m[0,3]=shift
    posed=cache.placed(v,t,m)
    exact=Body(v@m[:3,:3].T+m[:3,3],t)
    assert bool(collision(a,posed)) == bool(collision(a,exact)), shift
# A deliberately very different baked shape still takes the exact fallback.
m=np.eye(4);m[:3,:3]=np.diag([2.,.1,.2]);m[:3,3]=[-.5,.4,.4]
assert collision(a,cache.placed(v,t,m))
# Direct queries must retain solid containment.
m=np.eye(4);m[:3,:3]*=.1;m[:3,3]=.4
assert collision(a,cache.placed(v,t,m))
# Collision must not pay for distance prechecks at either dispatch layer.
ops=[]
original_query=module._CLIENT.exact_queries
def counted(queries):
    ops.extend(op for op, *_ in queries)
    return original_query(queries)
module._CLIENT.exact_queries=counted
try:
    for shift in (.5,1.,1.-1.1e-6):
        m=np.eye(4);m[0,3]=shift
        collision(a,cache.placed(v,t,m))
    assert ops and 'distance' not in ops, ops
    # An event at exact threshold uses one distance result, not a proxy query
    # followed by another identical exact query just to resolve equality.
    module._CLIENT.distances.clear();ops.clear()
    m=np.eye(4);m[0,3]=2.
    assert gap_predicate(a,cache.placed(v,t,m),1.)
    assert ops == ['distance'], ops
finally:
    module._CLIENT.exact_queries=original_query
'''
    script=tmp_path/'placed.py'
    script.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+code)
    result=subprocess.run([interpreter,str(script)],capture_output=True,text=True,timeout=60,env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_placements_are_kept_for_two_frames(tmp_path, pkgroot):
    """A body that has not moved keeps its placement from frame to frame; a
    moving or deforming one's earlier placements are let go. Kept for every
    frame, they would take a check's memory to many gigabytes."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    code = r'''
import bpy, numpy as np
from dycraft.animation.blender.collision import GeometryCache
v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
cache=GeometryCache()
still=np.eye(4); still[1,3]=5.
first=cache.placed(v,t,still,key='aa')
for i in range(200):
    cache.next_frame()
    assert cache.placed(v,t,still,key='aa') is first
    moving=np.eye(4); moving[0,3]=i*.1
    cache.placed(v,t,moving,key='bb')
    cache.placed(v*(1+i*1e-3),t,np.eye(4),key=f'cc{i:04x}')
assert len(cache._placed)+len(cache._previous) <= 6, (len(cache._placed), len(cache._previous))
'''
    script=tmp_path/'frames.py'
    script.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+code)
    result=subprocess.run([interpreter,str(script)],capture_output=True,text=True,timeout=60,env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_weld_hash_preserves_groups_and_checks_hash_collisions(monkeypatch):
    from dycraft.verify import _mesh_erosion as erosion
    rng = np.random.default_rng(218)
    vertices = rng.normal(size=(1000, 3))
    vertices = np.concatenate([vertices, vertices[::3], vertices[:10]+1e-11])
    keys = np.round(vertices/1e-9).astype(np.int64)
    _, expected_first, expected_inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    first, inverse = erosion._weld_indices(vertices)
    assert np.array_equal(first[inverse], expected_first[expected_inverse])
    original = np.unique

    def collide_hashes(array, **kwargs):
        if array.ndim == 1:
            return np.array([0]), np.array([0]), np.zeros(len(array), dtype=int)
        return original(array, **kwargs)

    monkeypatch.setattr(erosion.np, 'unique', collide_hashes)
    first, inverse = erosion._weld_indices(vertices)
    assert np.array_equal(first[inverse], expected_first[expected_inverse])


def test_fast_weld_keeps_erosion_bit_identical(monkeypatch):
    from scipy.spatial import ConvexHull
    from dycraft.verify import _mesh_erosion as erosion
    rng = np.random.default_rng(321)
    vertices = rng.normal(size=(50, 3))
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = ConvexHull(vertices).simplices.copy()
    triangles = vertices[faces]
    inward = np.einsum('ij,ij->i', np.cross(triangles[:,1]-triangles[:,0],
                                          triangles[:,2]-triangles[:,0]), triangles.mean(axis=1)) < 0
    faces[inward] = faces[inward][:, [0,2,1]]
    # Imported triangle soup includes duplicated face-corner vertices.
    vertices = vertices[faces].reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    fast = {chord: erosion._erode(vertices, faces, 1e-6, chord=chord) for chord in (False, True)}

    def old_weld(points, quantum=1e-9):
        keys = np.round(points/quantum).astype(np.int64)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        return first, inverse.reshape(-1)

    monkeypatch.setattr(erosion, '_weld_indices', old_weld)
    for chord in (False, True):
        assert np.array_equal(fast[chord], erosion._erode(vertices, faces, 1e-6, chord=chord))


def test_frame_signature_reuses_the_stored_asset_id_without_changing_its_value(tmp_path):
    """The stored id is the same SHA-256 the signature computes.

    Both spellings must agree, or signatures recorded by earlier runs would stop
    comparing equal and every accepted animation's replay check would fail.
    """
    import numpy as np
    from dycraft.animation.blender.evaluation import write_mesh, frame_signature
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='<f4')
    triangles = np.array([[0, 1, 2], [0, 1, 3]], dtype='<i4')
    known = set()
    digest = write_mesh(tmp_path, vertices, triangles, known)
    record = {'frame': 0, 'objects': [{'key': 'o0', 'name': 'part', 'mesh': digest,
                                       'identity': 'part', 'room': False, 'hidden': False,
                                       'matrix': [[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                       'color': [.5, .5, .5, 1]}],
              'transforms': {'part': [[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
              'joint_positions': {}}
    local = {'part': (vertices, triangles)}
    rehashed = {**record, 'objects': [{k: v for k, v in record['objects'][0].items() if k != 'mesh'}]}
    assert frame_signature(record, local) == frame_signature(rehashed, local)
    moved = {**record, 'transforms': {'part': [[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .5], [0, 0, 0, 1]]}}
    assert frame_signature(moved, local) != frame_signature(record, local)


def test_coal_triangle_pairs_match_the_blender_bvh_on_open_meshes(tmp_path, pkgroot):
    """The strict-crossing fallback reads Coal's colliding triangle pairs
    instead of rebuilding a Blender BVH of every placed pose.

    In general position the two report the same pairs. Where a sheet passes
    exactly through shared triangle edges the two can differ on the
    neighbors across an edge, which no proper crossing can turn on, so the
    verdict is the same either way.
    """
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    script = r'''
import bpy
import numpy as np
from dycraft.animation.blender.collision import GeometryCache, _strict_crossing, coal_query, collision
def sheet(n=6, size=1.0):
    xs = np.linspace(0, size, n)
    v = np.array([[x, y, 0.] for y in xs for x in xs])
    t = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            t += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return v, np.array(t)
def pairs(a, b):
    return {tuple(p) for p in coal_query('pairs', a, b)}, {tuple(p) for p in a.tree.overlap(b.tree)}
v, t = sheet()
cache = GeometryCache()
flat = cache.placed(v, t, np.eye(4))
rng = np.random.default_rng(0)
for _ in range(200):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle, m = rng.uniform(0, np.pi), np.eye(4)
    m[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    m[:3, 3] = rng.uniform(-.3, .8, 3)
    other = cache.placed(v, t, m)
    ours, theirs = pairs(flat, other)
    assert ours == theirs, sorted(ours ^ theirs)[:5]
for angle, lift in ((np.pi / 2, -.5), (np.pi / 3, -.2), (np.pi / 2, .3), (.01, .0005)):
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    m[:3, 3] = [.1, .37, lift]
    for step in (.0, .05, .2):
        moved = cache.placed(v, t, m).translated(np.array([0., 0., step]))
        ours, theirs = pairs(flat, moved)
        assert _strict_crossing(flat, moved, sorted(ours)) == _strict_crossing(flat, moved, sorted(theirs)), (
            angle, lift, step)
hit = collision(flat, cache.placed(v, t, np.array([[1., 0, 0, .1], [0, 0, -1, .37], [0, 1, 0, -.5], [0, 0, 0, 1]])))
assert hit is not None and hit['kind'] == 'surface_crossing', hit
'''
    source = tmp_path/'pairs.py'
    source.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+script)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True, timeout=120, env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_a_deforming_body_ships_its_triangles_once_and_answers_the_same(tmp_path, pkgroot):
    """Fix 5: each pose of a deforming object was shipped to Coal whole. Its
    triangles now travel once per topology, its vertices per pose, and the
    answers are those of a worker that was sent everything."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    script = r'''
import bpy, math
import numpy as np
from dycraft.animation.blender import collision as C
def sphere(n=16):
    vs, ts = [], []
    for i in range(n + 1):
        th = math.pi * i / n
        for j in range(2 * n):
            ph = math.pi * j / n
            vs.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    for i in range(n):
        for j in range(2 * n):
            a = i * 2 * n + j; b = i * 2 * n + (j + 1) % (2 * n); c = a + 2 * n; d = b + 2 * n
            ts += [[a, c, b], [b, c, d]]
    return np.array(vs), np.array(ts)
v, t = sphere()
fixed = C.Body(v * .4 + [1.1, 0, 0], t)
shipped = []
answers = []
for frame in range(12):
    if frame == 1:
        original = C._CLIENT.send
        C._CLIENT.send = lambda stream, value: (shipped.extend(m[2] is None for m in value.get('meshes', ())),
                                               original(stream, value))[1]
    wobble = v * (1 + .15 * np.sin(3 * v[:, [0]] + frame)) + [.02 * frame, 0, 0]
    body = C.GeometryCache().get(wobble, t)
    hit = C.collision(body, fixed)
    answers.append((round(C.coal_query('distance', body, fixed), 12), hit and hit['kind']))
assert shipped and all(shipped), shipped
C._CLIENT.topologies.clear()           # as a fresh client would: triangles every time
again = []
for frame in range(12):
    wobble = v * (1 + .15 * np.sin(3 * v[:, [0]] + frame)) + [.02 * frame, 0, 0]
    body = C.GeometryCache().get(wobble + 1e-9, t)
    hit = C.collision(body, fixed)
    again.append((round(C.coal_query('distance', body, fixed), 9), hit and hit['kind']))
assert [(round(d, 9), k) for d, k in answers] == again, (answers, again)
'''
    source = tmp_path/'topology.py'
    source.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+script)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True, timeout=120, env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_a_batch_of_coal_queries_answers_as_one_at_a_time_in_one_message(tmp_path, pkgroot):
    """Fix 6: a frame's first overlap queries and its bearing tests travel in
    one message per stage instead of one each."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    script = r'''
import bpy
import numpy as np
from dycraft.animation.blender import collision as C
v = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
t = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
cache = C.GeometryCache()
rng = np.random.default_rng(2)
a = cache.placed(v, t, np.eye(4))
others = []
for _ in range(40):
    m = np.eye(4); m[:3, 3] = rng.uniform(-1.5, 1.5, 3)
    others.append(cache.placed(v * rng.uniform(.2, 1.), t, m))
queries = [(op, a, b) for b in others for op in ('raw_overlap', 'distance')]
one = [C.coal_query(op, x, y) for op, x, y in queries]
sent = []
original = C._CLIENT.send
C._CLIENT.send = lambda stream, value: (sent.append(value['op']), original(stream, value))[1]
together = C.coal_queries(queries)
assert together == one, [(q[0], x, y) for q, x, y in zip(queries, one, together) if x != y][:4]
assert sent == ['batch'], sent
sent.clear()
C._CLIENT.distances.clear()
held = C.gap_predicates([(a, b) for b in others], .3)
assert held == [C.gap_predicate(a, b, .3) for b in others]
assert sent.count('batch') <= 2, sent
'''
    source = tmp_path/'batch.py'
    source.write_text('import sys\nsys.path.insert(0, '+repr(str(pkgroot))+')\n'+script)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True, timeout=120, env=_coal_env())
    assert result.returncode == 0, result.stdout+result.stderr


def test_inside_counts_winding_not_parity_on_overlapping_shells(tmp_path, pkgroot):
    """A body that is a union of overlapping closed shells -- a building made of
    overlapping boxes -- is inside wherever any shell is.

    Ray *parity* reads the overlap as outside (two exits, an even count); the
    signed winding `Body.inside` counts reads it as 2, which is inside. Pinned
    because a recorded analysis suspected a parity bug here; the code counts
    winding, and this proves it on exactly that geometry.
    """
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    script = r'''
import bpy
import numpy as np
from dycraft.animation.blender.collision import Body
v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
union=Body(np.concatenate([v, v+[.5,0,0]]), np.concatenate([t, t+8]))
assert union.inside(np.array([.75,.5,.5])), "overlap of two shells"
assert union.inside(np.array([.25,.5,.5])), "first shell only"
assert union.inside(np.array([1.25,.5,.5])), "second shell only"
assert not union.inside(np.array([1.75,.5,.5])), "outside both"
assert not union.inside(np.array([.75,.5,1.5])), "above both"
'''
    source = tmp_path / 'inside_union.py'
    source.write_text(script)
    env = dict(_coal_env(), PYTHONPATH=str(pkgroot))
    result = subprocess.run([interpreter, str(source)], capture_output=True,
                            text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_collision_interpreter_that_dies_at_start_says_why(tmp_path):
    """A child that cannot start (a uv venv whose base the sandbox does not show) dies before its version line; the
    error names what it printed, and that still reaches the worker's stderr too."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('bpy interpreter unavailable')
    dead = tmp_path / "python"
    dead.write_text("#!/bin/sh\necho 'Fatal Python error: Failed to import encodings module' >&2\n"
                    "echo \"ModuleNotFoundError: No module named 'encodings'\" >&2\nexit 1\n")
    dead.chmod(0o755)
    code = ("import bpy\nfrom dycraft.animation.blender.collision import CoalClient\n"
            "try:\n    CoalClient()\nexcept RuntimeError as exc:\n    print('ERROR', exc)\n")
    ran = subprocess.run([interpreter, '-c', code], capture_output=True, text=True, timeout=60,
                         env=dict(os.environ, DYCRAFT_COAL_PYTHON=str(dead)))
    assert ran.returncode == 0, ran.stderr
    assert (f"ERROR Coal unavailable in collision interpreter ({dead} said: Fatal Python error: Failed to import "
            "encodings module | ModuleNotFoundError: No module named 'encodings')") in ran.stdout
    assert "Fatal Python error: Failed to import encodings module" in ran.stderr
