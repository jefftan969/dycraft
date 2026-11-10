"""Cross-call collision reuse without accepting stale poses or declarations.

Reuse is keyed **per pair** — the two local meshes plus their relative pose —
not per frame. The counts below are the contract that follows from that, and
three of them are the reason the key was narrowed:

  * moving one body invalidates only the pairs it is in (here: none of the
    others), where a whole-scene frame key invalidated every frame;
  * a pair the broad phase rejects is never looked up at all, which is
    strictly better than a miss;
  * an allowance edit costs zero re-queries, because `verify.allowed` and
    `InheritedAllowances.allows` are evaluated *before* the lookup, so a pair
    verdict is allowance-independent.

Everything a cache must never do is pinned too: a changed evaluated mesh, a
changed relative pose and the room bounds all still decide by geometry.
"""
import os
from pathlib import Path
import subprocess


def test_pair_reuse_survives_edits_without_going_stale(tmp_path, pkgroot):
    from dycraft.render._blender_env import blender_python
    import pytest
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('Blender unavailable')
    script = r'''
import bpy, json, copy
import sys
import numpy as np
from pathlib import Path
# Make the import independent of the inherited cwd: the suite contains tests
# that chdir, so a bare `import dycraft` here passes alone and fails in the
# full run. Same idiom as test_blender_collision.py.
sys.path.insert(0, REPO)
from dycraft.animation.blender.evaluation import write_mesh
from dycraft.animation.blender.verify import verify
p=Path(SCRATCH);known=set()
v=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
t=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
h=write_mesh(p,v,t,known)
base={'identities':{a:{'object':a,'link':a,'local_matrix':np.eye(4).tolist()} for a in ['a/body','b/body']},'joints':{},'fixed_edges':[]}
scene={'id':'test','instances':[]};info={'format':'dycraft.evaluation.v2','complete':True,'frames':list(range(6)),'n_frames':6,'fps':1,'scene_ref':'test','declarations':{}}
(p/'index.json').write_text(json.dumps(info))
for f in range(6):
 objects=[]
 for i,n in enumerate(base['identities']):
  m=np.eye(4);m[0,3]=i*(.5+f*.01)
  objects.append({'name':n,'identity':n,'matrix':m.tolist(),'mesh':h,'hidden':False})
 (p/f'{f:06d}.json').write_text(json.dumps({'frame':f,'objects':objects,'transforms':{}}))
# By name, not by position: this used to index [2], which stopped being
# NoPenetration the day AnatomicalJointLimits was inserted ahead of it and
# turned every assertion below into a claim about a different verifier.
# `store` is a pair store kept across calls, as a project's is; `None` checks with a fresh cache.
def run(store,**kw):
 reports=verify(p,base,scene,{},**({'pair_store':str(store)} if store else {}),**kw)
 return next(r for r in reports if r['name']=='NoPenetration')
def counts(r):return (r['evidence']['stats']['collision_pair_cache_hits'],
                      r['evidence']['stats']['collision_pair_cache_misses'])
def semantic(r):
 r=copy.deepcopy(r);r['evidence'].pop('stats',None)
 return r
# A fragment queries only its own frames, and the full run reuses them.
cache=p/'cache'/'pairs.sqlite';cache.parent.mkdir()
fragment=run(cache,start=1,end=3);assert fragment['verdict']=='fail'
assert counts(fragment)==(0,3),counts(fragment)
full=run(cache);assert counts(full)==(3,3),counts(full)
cold=run(None)
assert semantic(full)==semantic(cold)
import dycraft.animation.blender.collision as collision_module
original_box=collision_module.bounded_box
box_calls=[]
def counted_box(*args):
    box_calls.append(1)
    return original_box(*args)
collision_module.bounded_box=counted_box
warm=run(cache);assert counts(warm)==(6,0),counts(warm)
assert semantic(warm)==semantic(cold)
# The broad phase is NOT skipped on a warm run, deliberately: it is one O(1)
# box per body from a memoized local AABB, and always running it is what lets a
# scene where one body moved still reuse every pair that body is not in.
assert len(box_calls)==12, box_calls
# Moving one body invalidates only its own pairs. Here the pair leaves contact
# entirely, so it is broad-phase rejected and never looked up — no miss at all.
f=p/'000002.json';r=json.loads(f.read_text());r['objects'][1]['matrix'][0][3]=3;f.write_text(json.dumps(r))
changed=run(cache);assert counts(changed)==(5,0),counts(changed)
collision_module.bounded_box=original_box
assert semantic(changed)==semantic(run(None))
# Appearance and events don't invalidate geometry collision results.
r['objects'][0]['color']=[1,0,0,1];f.write_text(json.dumps(r));info['declarations']['events']=[];(p/'index.json').write_text(json.dumps(info))
assert counts(run(cache))==(5,0)
# An exemption is applied before the lookup, so it neither consults nor
# poisons the cache: adding one costs no queries, and removing it restores the
# failure entirely from cached verdicts.
info['declarations']['collision_allowances']=[{'a':'a/body','b':'b/body','reason':'test contact'}];(p/'index.json').write_text(json.dumps(info))
exempt=run(cache);assert exempt['verdict']=='pass';assert counts(exempt)==(0,0),counts(exempt)
info['declarations']['collision_allowances']=[];(p/'index.json').write_text(json.dumps(info))
restored=run(cache);assert restored['verdict']=='fail';assert counts(restored)==(5,0),counts(restored)
# A timed exemption only applies at its actual output time.
info['declarations']['collision_allowances']=[{'a':'a/body','b':'b/body','reason':'t','window':[0,0]}]
(p/'index.json').write_text(json.dumps(info))
timed=run(cache);assert timed['verdict']=='fail';assert counts(timed)==(4,0),counts(timed)
info['declarations']['collision_allowances']=[];(p/'index.json').write_text(json.dumps(info))
# A hidden body takes its pairs out of contact checking, not out of the cache.
r['objects'][1]['hidden']=True;f.write_text(json.dumps(r));assert counts(run(cache))==(5,0)
r['objects'][1]['hidden']=False;f.write_text(json.dumps(r))
# Evaluated deformation must force a fresh query. Do it on a frame still in
# contact — on a broad-phase-rejected one there is no lookup to invalidate,
# which would make this assertion vacuous.
g=p/'000003.json';r3=json.loads(g.read_text());r3['objects'][0]['mesh']=write_mesh(p,v*1.2,t,known);g.write_text(json.dumps(r3))
deformed=run(cache);assert counts(deformed)==(4,1),counts(deformed)
assert counts(run(cache))==(5,0)
# Cached pair verdicts cannot bypass original room bounds.
base['room']=[[-10,-10,-10],[.1,.1,.1]]
bounded=run(cache);assert any(x['reason']=='outside room' for x in bounded['evidence']['out_of_room'])
# What one check queries, the next reuses, though it ran in another process.
store=p/'store'/'pairs.sqlite'; store.parent.mkdir()
def stored(**kw):
 reports=verify(p,base,scene,{},pair_store=str(store),**kw)
 return next(r for r in reports if r['name']=='NoPenetration')
first=stored();assert counts(first)[1]>0,counts(first)
again=stored();assert counts(again)[1]==0 and counts(again)[0]==sum(counts(first)),counts(again)
assert semantic(again)==semantic(first)
print('fragment, full, warm, one-body move, appearance, allowances, timed, visibility, deformation, bounds: passed')
'''.replace('SCRATCH',repr(str(tmp_path))).replace('REPO',repr(str(pkgroot)))
    env=dict(os.environ, DYCRAFT_COAL_PYTHON=os.environ.get('DYCRAFT_COAL_PYTHON', __import__('sys').executable))
    result=subprocess.run([interpreter,'-c',script],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stdout+result.stderr


def _key(i):
    """A `collision.pair_key`-shaped tuple: two 32-byte digests and pose bins."""
    return (bytes([i % 256]) * 32, bytes([(i // 256) % 256]) * 32, (i, 0, 0, 1, 0, 0, 0))


def test_the_on_disk_store_is_reused_by_the_next_call_and_holds_none_for_clear_pairs(tmp_path):
    """The dict the session shipped in every request was capped at
    50,000 pairs, fewer than one pass of the dog film needs, so a repeat
    compile reused nothing; the store keeps them across calls."""
    from dycraft.animation.blender.collision_cache import PairCollisionCache
    store = tmp_path / "pairs.sqlite"
    first = PairCollisionCache(None, method="m", store=store)
    for i in range(60000):
        assert first.get(_key(i)) == (None, False)
        first.put(_key(i), {"kind": "containment"} if i % 2 else None)
    first.flush()
    again = PairCollisionCache(None, method="m", store=store)
    assert again.get(_key(1)) == ({"kind": "containment"}, True)
    assert again.get(_key(2)) == (None, True), "a clear pair is a hit, not a miss"
    assert again.get(_key(60001)) == (None, False)
    assert again.stats() == {"collision_pair_cache_hits": 2, "collision_pair_cache_misses": 1,
                             "collision_pair_store_errors": 0}
    again.flush()
    # A different verdict function empties it rather than reusing answers.
    other = PairCollisionCache(None, method="other", store=store)
    assert other.get(_key(1)) == (None, False)


def test_the_store_stays_bounded_and_a_broken_store_never_fails_a_check(tmp_path, monkeypatch):
    import sqlite3
    from dycraft.animation.blender import collision_cache
    monkeypatch.setattr(collision_cache, "STORE_BYTES", 256 * 1024)
    store = tmp_path / "pairs.sqlite"
    for batch in range(4):
        cache = collision_cache.PairCollisionCache(None, method="m", store=store)
        for i in range(batch * 5000, (batch + 1) * 5000):
            cache.put(_key(i), {"kind": "surface_crossing", "pad": "x" * 40})
        cache.flush()
    db = sqlite3.connect(store)
    pages, free, size = (db.execute(f"PRAGMA {n}").fetchone()[0]
                         for n in ("page_count", "freelist_count", "page_size"))
    assert (pages - free) * size <= 2 * 256 * 1024
    newest = collision_cache.PairCollisionCache(None, method="m", store=store)
    assert newest.get(_key(19999))[1] and not newest.get(_key(0))[1], "the oldest go first"
    newest.flush()
    store.write_bytes(b"not a database")
    broken = collision_cache.PairCollisionCache(None, method="m", store=store)
    assert broken.get(_key(1)) == (None, False)
    broken.put(_key(1), None)
    assert broken.get(_key(1)) == (None, True)
    broken.flush()
    assert broken.stats()["collision_pair_store_errors"] >= 1


def test_a_writer_under_an_older_implementation_poisons_nothing(tmp_path):
    """A process that opened the store before a newer implementation emptied it
    still writes when it finishes; its keys are not the newer one's."""
    from dycraft.animation.blender.collision_cache import PairCollisionCache
    store = tmp_path / "pairs.sqlite"
    old = PairCollisionCache(None, method="old", store=store)
    new = PairCollisionCache(None, method="new", store=store)
    new.flush()
    old.put(_key(1), None)            # "clear", by the old verdict function
    old.flush()
    assert PairCollisionCache(None, method="new", store=store).get(_key(1)) == (None, False)


def test_a_value_the_store_cannot_read_turns_it_off_and_fails_nothing(tmp_path):
    import sqlite3
    from dycraft.animation.blender.collision_cache import PairCollisionCache
    store = tmp_path / "pairs.sqlite"
    cache = PairCollisionCache(None, method="m", store=store)
    for i in (1, 2):
        cache.put(_key(i), {"kind": "containment"})
    cache.flush()
    db = sqlite3.connect(store)
    rows = [key for key, in db.execute("SELECT key FROM pairs ORDER BY rowid")]
    db.execute("UPDATE pairs SET hit = 'not json' WHERE key = ?", (rows[0],))
    db.execute("UPDATE pairs SET hit = '7' WHERE key = ?", (rows[1],))
    db.commit()
    for i in (1, 2):
        read = PairCollisionCache(None, method="m", store=store)
        assert read.get(_key(i)) == (None, False)
        assert read.stats()["collision_pair_store_errors"] == 1
        read.flush()
