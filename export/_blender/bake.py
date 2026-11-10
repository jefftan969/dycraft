"""Bake procedural materials to image textures so they survive glTF export.

    blender -b --factory-startup --python bake.py -- <animation.blend> <out.blend> [px_per_m=384] [max_px=2048] [hero=<object>]
    python bake.py <animation.blend> <out.blend> [...]          (the bpy wheel)
  BAKE_WORKERS=N   bake in N processes (default: half the CPUs, at most 4; 1 = in this process, as under the bpy wheel)

Objects within HERO_RADIUS of the hero object (where the cameras get close) bake at HERO_PX_PER_M.

Per visible mesh whose material uses procedural textures: a fresh non-overlapping UV map
("bake"), then Cycles bakes of base color, roughness, and tangent normal into packed images,
and the material is rebuilt as Principled + those images. Light Path / Transparent mixes
(shadow-ray tricks) are dropped in favor of the Principled branch.

A GPU is used when there is one; on the CPU a heavy film takes hours. Speed: a bake call costs about 10 s on a GPU
whatever it bakes: Cycles starts a session, syncs the scene, and loads every image a renderable material holds, once per
object even when many objects are selected (so 40 objects in one call load 40 x 40 images, and "Updating Images" takes
most of the bake). So: one object per call, with only its own images attached and the rest of the scene hidden from render;
only the passes its materials need (an unlinked Roughness or Normal bakes a solid image that gltf-transform folds
into a factor anyway); and the objects split over BAKE_WORKERS processes (one GPU is enough for several). The
visibility scan runs with shape-key animation and modifiers muted (visibility.cheap_eval), and its result goes to
<out.blend>.vis.json so that the export doesn't scan again.

Worker mode (started by the main process): blender -b <prep.blend> --python bake.py -- --worker <jobs.json> <i> <n> <dir>
"""
import sys  # accept blender -b --python script -- args
if "--" in sys.argv:
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
import bpy, json, math, os, re, subprocess, sys, time
import numpy as np
from contextlib import contextmanager

WORKER = len(sys.argv) > 1 and sys.argv[1] == '--worker'
if not WORKER:
    src, dst = sys.argv[1], sys.argv[2]
    PX_PER_M = float(sys.argv[3]) if len(sys.argv) > 3 and '=' not in sys.argv[3] else 384
    MAX_PX = int(sys.argv[4]) if len(sys.argv) > 4 and '=' not in sys.argv[4] else 2048
    HERO = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('hero=')), None)
    HERO_PX_PER_M, HERO_RADIUS = 1536, 1.5
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
sc = bpy.context.scene
GPU_FAULT = re.compile(r'Failed to retain CUDA context|CUDA error|OptiX error|ECC error')

prefs = bpy.context.preferences.addons['cycles'].preferences
for kind in ('OPTIX', 'CUDA'):
    try:
        prefs.compute_device_type = kind
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            break
    except TypeError:
        pass
for d in prefs.devices:
    d.use = d.type != 'CPU'
sc.render.engine = 'CYCLES'
sc.cycles.device = 'GPU' if any(d.use for d in prefs.devices) else 'CPU'
sc.cycles.samples = 4
sc.render.bake.margin = 8
sc.render.bake.use_selected_to_active = False
print('devices', prefs.compute_device_type if sc.cycles.device == 'GPU' else 'CPU',
      [d.name for d in prefs.devices if d.use], flush=True)

# key, bake type, Principled input, color space
PASSES = (('color', 'DIFFUSE', 'Base Color', 'sRGB'), ('rough', 'ROUGHNESS', 'Roughness', 'Non-Color'),
          ('normal', 'NORMAL', 'Normal', 'Non-Color'))
CS = {key: cs for key, _, _, cs in PASSES}


def procedural(m):
    return m and m.use_nodes and any(n.type.startswith('TEX_') and n.type != 'TEX_IMAGE' for n in m.node_tree.nodes)


def surface_principled(m):
    nt = m.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) \
        or next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    return nt, out, bsdf


def drop_shadow_tricks(m):
    """Mix(Transparent, Principled) driven by Light Path is a render trick, not see-through glass."""
    nt, out, bsdf = surface_principled(m)
    if not out or not bsdf:
        return
    s = out.inputs['Surface']
    if s.is_linked and s.links[0].from_node.type == 'MIX_SHADER':
        mix = s.links[0].from_node
        if mix.inputs[0].is_linked and mix.inputs[0].links[0].from_node.type == 'LIGHT_PATH':
            for l in list(s.links):
                nt.links.remove(l)
            nt.links.new(bsdf.outputs[0], s)


def centre(o):
    from mathutils import Vector
    return o.matrix_world @ (sum((Vector(c) for c in o.bound_box), Vector()) / 8)


def tex_size(o, hero_at, area=None):
    if area is None:
        area = sum(p.area for p in o.data.polygons) * max(abs(v) for v in o.matrix_world.to_scale()) ** 2
    near = hero_at is not None and (centre(o) - hero_at).length < HERO_RADIUS
    px = math.sqrt(max(area, 1e-6)) * (HERO_PX_PER_M if near else PX_PER_M)
    return int(min(MAX_PX, max(64, 2 ** round(math.log2(max(px, 1))))))


def surface_area(o, dg, co=None):
    """World-space area of o as evaluated (modifiers, shape keys, animated scale) in depsgraph dg; with co, of o's own
    polygons at those vertex positions (a computed shape-key mix)."""
    e = o.evaluated_get(dg)
    scale = max(abs(v) for v in e.matrix_world.to_scale()) ** 2
    if co is not None:
        return poly_area(o.data, co) * scale
    m = e.to_mesh()
    a = sum(p.area for p in m.polygons) * scale
    e.to_mesh_clear()
    return a


def poly_area(me, co):
    """Total polygon area of mesh me with vertex positions co (n x 3), each polygon's by Newell's normal as Blender's."""
    nl, npoly = len(me.loops), len(me.polygons)
    if not npoly:
        return 0.0
    vi, start, total = np.empty(nl, np.int64), np.empty(npoly, np.int64), np.empty(npoly, np.int64)
    me.loops.foreach_get('vertex_index', vi)
    me.polygons.foreach_get('loop_start', start)
    me.polygons.foreach_get('loop_total', total)
    poly = np.repeat(np.arange(npoly), total)
    nxt = start[poly] + (np.arange(nl) - start[poly] + 1) % total[poly]
    v = co.astype(np.float64)
    c = np.cross(v[vi], v[vi[nxt]])
    n = np.stack([np.bincount(poly, weights=c[:, k], minlength=npoly) for k in range(3)], 1)
    return float(0.5 * np.linalg.norm(n, axis=1).sum())


def key_extent(kb, n):
    co = [0.0] * (3 * n)
    kb.data.foreach_get('co', co)
    return max(max(co[i::3]) - min(co[i::3]) for i in range(3)) if n else 0.0


def select_only(o):
    for x in bpy.context.view_layer.objects:
        x.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o


def needed_passes(mats):
    """The passes worth baking. A pass can be skipped only for a material whose surface is one Principled BSDF wired
    straight to the output with that input unlinked (a constant). Anything else (a mix of shaders, another BSDF, as
    fur uses) bakes all three: the bake reads the whole surface, the input of the first Principled doesn't."""
    out = set()
    for m in mats:
        nt, outn, b = surface_principled(m)
        s = outn.inputs['Surface'] if outn else None
        direct = b is not None and s is not None and s.is_linked and s.links[0].from_node == b
        for key, _, sock, _ in PASSES:
            if not direct or b.inputs[sock].is_linked:
                out.add(key)
    return [key for key, *_ in PASSES if key in out]


def unmetal(mats):
    """Diffuse color of metal or glass is black; zero Metallic and Transmission while baking color.
    Returns what to restore."""
    saved = []
    for m in dict.fromkeys(mats):
        nt, out, bsdf = surface_principled(m)
        if not bsdf:
            continue
        for name in ('Metallic', 'Transmission Weight'):
            s = bsdf.inputs[name]
            src_sock = s.links[0].from_socket if s.is_linked else None
            saved.append((nt, s, src_sock, s.default_value))
            for l in list(s.links):
                nt.links.remove(l)
            s.default_value = 0.0
    return saved


def remetal(saved):
    for nt, s, src_sock, v in saved:
        s.default_value = v
        if src_sock is not None:
            nt.links.new(src_sock, s)


@contextmanager
def render_only(o, mats):
    """Everything but o hidden from render while it bakes: its color, roughness, and normal don't depend on the rest
    of the scene, and Cycles then syncs one object. Not when a material reads its surroundings (Ambient Occlusion)."""
    if any(n.type == 'AMBIENT_OCCLUSION' for m in mats for n in m.node_tree.nodes):
        yield
        return
    saved = [x for x in sc.objects if x is not o and not x.hide_render]
    for x in saved:
        x.hide_render = True
    try:
        yield
    finally:
        for x in saved:
            x.hide_render = False


def bake_job(job, pack=True):
    """Bake one object's passes, one call each, with only its own image nodes holding images."""
    o, mats, nodes, imgs, size, passes = job
    select_only(o)
    with render_only(o, mats):
        for key, btype, _, _ in PASSES:
            if key not in passes:
                continue
            for n in nodes:
                n.image = imgs[key]
            kw = {'type': btype}
            if btype == 'DIFFUSE':
                kw['pass_filter'] = {'COLOR'}
                saved = unmetal(mats)
            try:
                bpy.ops.object.bake(**kw)
            finally:
                if btype == 'DIFFUSE':
                    remetal(saved)
    for n in nodes:
        n.image = None
    if pack:
        for im in imgs.values():
            im.pack()


def worker(args):
    """Bake every n-th job of jobs.json from the prepared blend (opened on the command line); write PNGs."""
    jobs_path, i, n, outdir = args[0], int(args[1]), int(args[2]), args[3]
    for idx, sp in json.load(open(jobs_path))[i::n]:
        o = bpy.data.objects[sp['obj']]
        mats = [bpy.data.materials[m] for m in sp['mats']]
        nodes = [bpy.data.materials[m].node_tree.nodes[nn] for m, nn in sp['nodes']]
        imgs = {k: bpy.data.images[v] for k, v in sp['imgs'].items()}
        bake_job((o, mats, nodes, imgs, None, sp['passes']), pack=False)
        for k, im in imgs.items():
            im.filepath_raw = os.path.join(outdir, f'{idx}_{k}.png')
            im.file_format = 'PNG'
            im.save()
        print('WORKER_BAKED', idx, o.name, flush=True)
    print('WORKER_OK', flush=True)


if WORKER:
    worker(sys.argv[2:])
    raise SystemExit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visibility import prepare_visibility, cheap_eval, restore_drivers
from morphs import key_mixes, check_frames, _coords
# the source's own hide flags, before prepare_visibility changes them (the export restores wire/bounds helpers'):
# from the earlier bake's record when this is a re-bake of a baked blend
source_hide = {o.name: o.hide_render for o in bpy.data.objects}
if os.path.exists(src + '.vis.json'):
    source_hide = json.load(open(src + '.vis.json')).get('source_hide', source_hide)
states = prepare_visibility(sc, for_export=False)
json.dump({'states': states, 'source_hide': source_hide}, open(dst + '.vis.json', 'w'))
targets = [o for o in sc.objects if o.type == 'MESH' and o.visible_get() and 'bake' not in o.data.uv_layers
           and any(procedural(s.material) for s in o.material_slots)]
for m in bpy.data.materials:
    if m.use_nodes:
        drop_shadow_tricks(m)
hero_obj = bpy.data.objects.get(HERO) if HERO else None
hero_at = centre(hero_obj) if hero_obj else None
print('baking', len(targets), 'objects; hero', HERO, tuple(hero_at) if hero_at else None, flush=True)
t0 = time.time()


def own(o):
    """Give o its own mesh and materials, so per-object UVs and baked images never leak to others."""
    if o.data.users > 1:
        o.data = o.data.copy()
    for s in o.material_slots:
        if s.material and s.material.users > 1:
            s.material = s.material.copy()


# 1. Per object: private data, a fresh "bake" UV map, the images of the passes it needs, an image node per material.
jobs = []
for k, o in enumerate(targets):
    own(o)
    mats = list(dict.fromkeys(s.material for s in o.material_slots if s.material and s.material.use_nodes))
    if not any(surface_principled(m)[2] for m in mats):
        continue
    passes = needed_passes(mats)
    if not passes:
        continue
    select_only(o)
    uv = o.data.uv_layers.new(name='bake')
    o.data.uv_layers.active = uv
    # Edit mode (and so the unwrap) works on the active shape key. A per-frame shape cache can have a basis collapsed
    # to a point (scattered grains that appear later): unwrap on the widest key instead, or every face gets a zero-area island.
    sk, key0 = o.data.shape_keys, o.active_shape_key_index
    if sk and key_extent(sk.key_blocks[key0], len(o.data.vertices)) < 1e-6:
        o.active_shape_key_index = max(range(len(sk.key_blocks)),
                                       key=lambda i: key_extent(sk.key_blocks[i], len(o.data.vertices)))
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004, area_weight=1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    o.active_shape_key_index = key0
    size = tex_size(o, hero_at)
    imgs = {}
    for key in passes:
        im = bpy.data.images.new(f'{o.name}_{key}', size, size, alpha=False)
        im.colorspace_settings.name = CS[key]
        im.use_fake_user = True                  # no node holds it between passes; keep it in the workers' copy
        imgs[key] = im
    nodes = []
    for m in mats:
        nt = m.node_tree
        node = nt.nodes.new('ShaderNodeTexImage')
        for n in nt.nodes:
            n.select = False
        node.select = True
        nt.nodes.active = node
        nodes.append(node)
    jobs.append((o, mats, nodes, imgs, size, passes))
print('prepared', len(jobs), 'objects', sum(len(j[5]) for j in jobs), 'passes', f'{time.time() - t0:.0f}s', flush=True)

# An object with no surface at the current frame bakes black: absolute shape keys that end collapsed to a point
# (food that changes form as it cooks), a basis-only cache (grains that appear later), an animated zero scale. Bake the
# rest first, and those afterwards at the sampled frame where their surface is largest.
dg = bpy.context.evaluated_depsgraph_get()
flat = [j for j in jobs if surface_area(j[0], dg) < 1e-10]
plain = [j for j in jobs if all(j is not f for f in flat)]

# 2. The plain objects: in BAKE_WORKERS processes from a saved copy of this state, or here.
from morphs import worker_count
workers = worker_count('baking') if len(plain) >= 4 else 1
if workers > 1 and len(plain) >= 2 * workers:
    tmp = dst + '.parts'
    os.makedirs(tmp, exist_ok=True)
    prep = os.path.join(tmp, 'prep.blend')
    bpy.ops.wm.save_as_mainfile(filepath=prep, copy=True, compress=False)
    spec = [(i, {'obj': o.name, 'mats': [m.name for m in mats], 'nodes': [[m.name, n.name] for m, n in zip(mats, nodes)],
                 'imgs': {k: im.name for k, im in imgs.items()}, 'passes': passes})
            for i, (o, mats, nodes, imgs, size, passes) in enumerate(plain)]
    json.dump(spec, open(os.path.join(tmp, 'jobs.json'), 'w'))
    procs = []
    for i in range(workers):
        log = open(os.path.join(tmp, f'worker{i}.log'), 'w')
        procs.append((subprocess.Popen([bpy.app.binary_path, '-b', '--factory-startup', prep, '--python-exit-code', '1',
                                        '--python', os.path.abspath(__file__), '--', '--worker',
                                        os.path.join(tmp, 'jobs.json'), str(i), str(workers), tmp],
                                       stdout=log, stderr=subprocess.STDOUT), log))
    print(f'baking {len(plain)} objects in {workers} workers', flush=True)
    bad = []
    for i, (p, log) in enumerate(procs):
        p.wait()
        log.close()
        text = open(os.path.join(tmp, f'worker{i}.log'), errors='replace').read()
        fault = GPU_FAULT.search(text)
        if p.returncode or 'WORKER_OK' not in text or fault:
            bad.append(f'worker {i}: exit {p.returncode}' + (f', {fault.group()}' if fault else ''))
    if bad:
        print('\n'.join(bad), flush=True)       # printed too, so a calling script can match the GPU fault text
        raise SystemExit('bake workers failed: ' + '; '.join(bad))
    for i, (o, mats, nodes, imgs, size, passes) in enumerate(plain):
        for k, im in imgs.items():
            im.filepath_raw = os.path.join(tmp, f'{i}_{k}.png')
            im.source = 'FILE'
            im.reload()
            im.colorspace_settings.name = CS[k]
            im.pack()
            im.filepath_raw = ''                 # packed: no scratch path left in the blend
    print(f'[{len(plain)}/{len(plain)}] baked  {time.time() - t0:.0f}s', flush=True)
else:
    for g, j in enumerate(plain):
        bake_job(j)
        if (g + 1) % 20 == 0 or g + 1 == len(plain):
            print(f'[{g + 1}/{len(plain)}] baked  {time.time() - t0:.0f}s', flush=True)

if flat:
    cur = sc.frame_current
    # frame_set writes the keyed visibility back; put prepare_visibility's state back afterwards
    vis = {}
    for o in sc.objects:
        try:
            vis[o.name] = (o.hide_render, o.hide_get())
        except RuntimeError:
            vis[o.name] = (o.hide_render, None)
    step = max(1, (sc.frame_end - sc.frame_start) // 400)
    best = {id(j): (0.0, None) for j in flat}
    # A per-frame shape-key cache (thousands of keys) is computed from its keys rather than evaluated, once it
    # matches Blender's evaluation at check frames.
    fobjs = [j[0] for j in flat]
    mixes = key_mixes(sc, fobjs)
    if mixes:
        with cheap_eval(keep=fobjs):
            for f in check_frames(sc, mixes.values()):
                sc.frame_set(f)
                dg = bpy.context.evaluated_depsgraph_get()
                for n in list(mixes):
                    ref, got = _coords(bpy.data.objects[n], dg), mixes[n].coords(f)
                    if ref.shape != got.shape or (ref.size and float(np.abs(ref - got).max()) > 1e-5):
                        print(f'shape keys: {n} differs from Blender at frame {f}; evaluated by Blender', flush=True)
                        del mixes[n]
    with cheap_eval(keep=[o for o in fobjs if o.name not in mixes]):     # only these objects' surfaces matter here
        for f in sorted(set(range(sc.frame_start, sc.frame_end + 1, step)) | {sc.frame_end}):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            for j in flat:
                m = mixes.get(j[0].name)
                a = surface_area(j[0], dg, m.coords(f) if m else None)
                if a > best[id(j)][0]:
                    best[id(j)] = (a, f)
    by_frame = {}
    for j in flat:
        o, mats, nodes, imgs, size, passes = j
        a, f = best[id(j)]
        if f is None:
            print('no surface at any sampled frame, baked as is:', o.name, flush=True)
            f = cur
        else:
            # texture density from the real surface, not the basis (absolute keys can start from a much larger template)
            px = tex_size(o, hero_at, a)
            for im in imgs.values():
                im.scale(px, px)
        by_frame.setdefault(f, []).append(j)
    for f, grp in sorted(by_frame.items()):
        sc.frame_set(f)
        for j in grp:
            j[0].hide_render = False
            j[0].hide_set(False)
            bake_job(j)
        print(f'baked at frame {f}:', ' '.join(j[0].name for j in grp), f'{time.time() - t0:.0f}s', flush=True)
    sc.frame_set(cur)
    for o in sc.objects:
        h, v = vis.get(o.name, (None, None))
        if h is not None:
            o.hide_render = h
        if v is not None:
            o.hide_set(v)

# 3. Rebuild every material as Principled fed by the baked images on the "bake" UV map (the passes baked).
for o, mats, nodes, imgs, size, passes in jobs:
    for m, node in zip(mats, nodes):
        nt, out, bsdf = surface_principled(m)
        nt.nodes.remove(node)
        if not bsdf:
            continue
        uvn = nt.nodes.new('ShaderNodeUVMap')
        uvn.uv_map = 'bake'

        def img_node(key):
            n = nt.nodes.new('ShaderNodeTexImage')
            n.image = imgs[key]
            nt.links.new(uvn.outputs[0], n.inputs[0])
            return n

        for key, _, sock, _ in PASSES:
            if key in passes:
                for l in list(bsdf.inputs[sock].links):
                    nt.links.remove(l)
        if 'color' in passes:
            nt.links.new(img_node('color').outputs['Color'], bsdf.inputs['Base Color'])
        if 'rough' in passes:
            nt.links.new(img_node('rough').outputs['Color'], bsdf.inputs['Roughness'])
        if 'normal' in passes:
            nm = nt.nodes.new('ShaderNodeNormalMap')
            nm.uv_map = 'bake'
            nt.links.new(img_node('normal').outputs['Color'], nm.inputs['Color'])
            nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    # Keep only the bake UVs so the exporter writes a single TEXCOORD_0. Remove by name, looked up each time: a
    # removal reallocates the layers, so references taken before it point at the wrong ones (with two other maps,
    # 'bake' itself would go and the textures land on the old UVs).
    for name in [l.name for l in o.data.uv_layers if l.name != 'bake']:
        o.data.uv_layers.remove(o.data.uv_layers[name])
    for im in imgs.values():
        im.use_fake_user = False

restore_drivers()
bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
if os.path.isdir(dst + '.parts'):
    import shutil
    shutil.rmtree(dst + '.parts', ignore_errors=True)
print('BAKE_OK', dst, f'{time.time() - t0:.0f}s', flush=True)
