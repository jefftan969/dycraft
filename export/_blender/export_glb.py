"""Export a .blend to a GLB, plus a cuts sidecar when the scene has a camera. Runs inside Blender.

    blender -b --factory-startup --python export_glb.py -- <in.blend> <out.glb> [--source <src.blend>] [--report <r.json>]
    python export_glb.py <in.blend> <out.glb> [...]          (the bpy wheel)

`--source` names the original blend when the input is `bake.py`'s output: transparent colors and helper objects'
render visibility are taken back from it. The blend is never saved; scratch files go to a temporary folder, never
beside the input, which may be a sealed bundle or a downloaded film.

glTF carries node transforms, skins, morph weights, cameras, and PBR materials. So before exporting:
- each material is reduced to what glTF can hold (`flatten`): a procedural color or value becomes a representative
  constant, an image stays an image. The materials that lost a procedural color are reported, because a reader sees
  them first; `bake.py` turns them into images instead;
- deformation other than an armature's becomes morph targets (`morphs.bake_morphs`);
- objects that are never visible in the film are left out (`visibility.prepare_visibility`).

What glTF cameras cannot carry (a zoom, a lens shift) and when each camera is live go into `<out>.cuts.json`.
Lights are not exported: Blender's light units have no faithful glTF equivalent.
"""
import sys
if "--" in sys.argv:        # blender -b --python export_glb.py -- <args>
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
import json
import os
import re
import shutil
import tempfile
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

args = sys.argv[1:]


def _option(name):
    if name in args:
        i = args.index(name)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return None


SOURCE, REPORT = _option("--source"), _option("--report")
src, out = args[0], args[1]
started = time.time()
bpy.context.preferences.filepaths.use_scripts_auto_execute = False
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
sc = bpy.context.scene


from flat_material import flat_color, flat_value, mean  # noqa: E402 (after the scene opens, as the other imports)


def flatten(mat):
    """Reduce `mat` to a Principled BSDF glTF can carry. True when a procedural base color became a constant."""
    if not mat.use_nodes:
        return False
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) \
        or next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not out or not bsdf:
        return False
    alpha = None
    surf = out.inputs['Surface']
    if surf.is_linked and surf.links[0].from_node.type == 'MIX_SHADER':
        mix = surf.links[0].from_node
        kinds = [s.links[0].from_node.type if s.is_linked else None for s in mix.inputs[1:3]]
        if 'BSDF_TRANSPARENT' in kinds:
            fac = mix.inputs[0]
            t_first = kinds[0] == 'BSDF_TRANSPARENT'
            # A Light Path-driven mix is a shadow-ray trick; only a constant mix is real see-through.
            alpha = None if fac.is_linked else (fac.default_value if t_first else 1 - fac.default_value)
            if fac.is_linked and fac.links[0].from_node.type in ('FRESNEL', 'LAYER_WEIGHT'):
                # Fresnel-weighted transparency is clear glass: glTF carries it as transmission.
                bsdf.inputs['Transmission Weight'].default_value = 1.0
                bsdf.inputs['Metallic'].default_value = 0.0
                bsdf.inputs['Roughness'].default_value = min(bsdf.inputs['Roughness'].default_value, 0.1)
    procedural = False
    s = bsdf.inputs['Base Color']
    c = flat_color(s)
    if c != 'image':
        procedural = s.is_linked
        for link in list(s.links):
            nt.links.remove(link)
        if c:
            s.default_value = c
    for name in ('Roughness', 'Metallic', 'Alpha', 'Transmission Weight', 'Emission Strength'):
        s = bsdf.inputs.get(name)
        if s is not None and s.is_linked and s.links[0].from_node.type != 'TEX_IMAGE':
            v = flat_value(s)
            for link in list(s.links):
                nt.links.remove(link)
            s.default_value = v
    s = bsdf.inputs['Normal']
    if s.is_linked and s.links[0].from_node.type != 'NORMAL_MAP':
        for link in list(s.links):
            nt.links.remove(link)
    if alpha is None and surf.is_linked and surf.links[0].from_node.type == 'MIX_SHADER':
        for link in list(surf.links):
            nt.links.remove(link)
        nt.links.new(bsdf.outputs[0], surf)
    if alpha is not None:
        bsdf.inputs['Alpha'].default_value = max(0.05, min(1.0, alpha))
        for link in list(surf.links):
            nt.links.remove(link)
        nt.links.new(bsdf.outputs[0], surf)
        try:
            mat.surface_render_method = 'BLENDED'
        except Exception:
            mat.blend_method = 'BLEND'
    return procedural


def _used_materials():
    used = set()
    for o in sc.objects:
        if o.type == 'MESH' and not o.hide_render:
            used.update(slot.material.name for slot in o.material_slots if slot.material)
    return used


used = _used_materials()
flattened = sorted(m.name for m in bpy.data.materials if flatten(m) and m.name in used)
flat_set = set(flattened)
flat_objects = sorted(re.sub(r'^part/', '', o.name) for o in sc.objects
                      if o.type == 'MESH' and not o.hide_render
                      and any(slot.material and slot.material.name in flat_set for slot in o.material_slots))


def restore_glass_colour(orig):
    """A baked color of a transmissive surface is near-black; take its flat color from the source blend."""
    base = lambda n: re.sub(r'\.\d{3}$', '', n)
    glassy = {}
    for m in bpy.data.materials:
        b = m.use_nodes and next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if b and b.inputs['Transmission Weight'].default_value > 0.3 and b.inputs['Base Color'].is_linked:
            glassy.setdefault(base(m.name), []).append((m, b))
    if not glassy:
        return
    with bpy.data.libraries.load(orig) as (source, dest):
        dest.materials = [n for n in source.materials if base(n) in glassy]
    n = 0
    for om in dest.materials:
        if om is None or not om.use_nodes:
            continue
        ob = next((x for x in om.node_tree.nodes if x.type == 'BSDF_PRINCIPLED'), None)
        c = ob and flat_color(ob.inputs['Base Color'])
        if not c or c == 'image':
            continue
        for m, b in glassy[base(om.name)]:
            for link in list(b.inputs['Base Color'].links):
                m.node_tree.links.remove(link)
            b.inputs['Base Color'].default_value = c
            n += 1
    print('glass colors restored', n, flush=True)


if SOURCE:
    restore_glass_colour(SOURCE)

fps = sc.render.fps / sc.render.fps_base
cuts_path = None
if sc.camera or any(o.type == 'CAMERA' for o in sc.objects):
    cuts = [{'frame': mk.frame, 't': mk.frame / fps, 'camera': mk.camera.name}
            for mk in sorted(sc.timeline_markers, key=lambda m: m.frame) if mk.camera]
    side = {
        'fps': fps, 'frame_start': sc.frame_start, 'frame_end': sc.frame_end,
        'duration': (sc.frame_end - sc.frame_start + 1) / fps,
        'aspect': sc.render.resolution_x / sc.render.resolution_y,
        'default_camera': sc.camera.name if sc.camera else None,
        'cuts': cuts,
    }
    from cameras import camera_fov, camera_shift
    # What glTF cameras cannot carry: a zoom (the focal length animated) and a lens shift.
    names = sorted({c['camera'] for c in cuts} | ({sc.camera.name} if sc.camera else set()))
    for key, val in (('fov', camera_fov(sc, names)), ('shift', camera_shift(sc, names))):
        if val:
            side[key] = val
    cuts_path = re.sub(r'\.glb$', '', out) + '.cuts.json'
    with open(cuts_path, 'w') as f:
        json.dump(side, f, indent=1)

from visibility import prepare_visibility
# `bake.py` records its visibility scan and the source's hide flags beside the baked blend (<baked>.vis.json), so the
# scan is not repeated and the source is not opened again.
vis = json.load(open(src + '.vis.json')) if os.path.exists(src + '.vis.json') else None
if vis and vis.get('source_hide') is not None:
    # The bake un-hid every keyed object, wireframe helpers included; hide those again. Only viewport helpers
    # (wire or bounds display): solid objects with hidden source flags may be shown by their keys.
    orig_hide, n = vis['source_hide'], 0
    for o in sc.objects:
        if o.display_type in ('WIRE', 'BOUNDS') and orig_hide.get(o.name) and not o.hide_render:
            o.hide_render = orig_hide[o.name]
            n += 1
    print('hide_render restored from source', n, flush=True)
prepare_visibility(sc, for_export=True, states=vis['states'] if vis else None)

from morphs import bake_morphs
scratch = tempfile.mkdtemp(prefix='dycraft-export-')
try:
    morphed = bake_morphs(sc, tmp=os.path.join(scratch, 'morph'))
finally:
    shutil.rmtree(scratch, ignore_errors=True)

animated = sc.frame_end > sc.frame_start
bpy.ops.export_scene.gltf(
    filepath=out, export_format='GLB',
    use_selection=False, use_visible=True, export_apply=True,
    export_cameras=True, export_lights=False, export_extras=False,
    export_yup=True, export_animations=animated, export_animation_mode='SCENE',
    export_force_sampling=True, export_frame_range=True, export_frame_step=1,
    export_optimize_animation_size=True, export_def_bones=True,
    export_morph_normal=False,  # morph normals double the size of deforming meshes
    export_image_format='AUTO', export_materials='EXPORT')

report = {
    'ok': True, 'glb': out, 'cuts': cuts_path, 'bytes': os.path.getsize(out), 'seconds': round(time.time() - started, 1),
    'blender': bpy.app.version_string, 'frames': sc.frame_end - sc.frame_start + 1 if animated else 1, 'fps': fps,
    'objects': sum(1 for o in sc.objects if o.type == 'MESH' and not o.hide_render),
    'cameras': sum(1 for o in sc.objects if o.type == 'CAMERA'), 'morphed': morphed or [],
    'flattened_materials': flattened, 'flattened_objects': flat_objects,
}
if REPORT:
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=1)
print('EXPORT_OK ' + json.dumps({k: v for k, v in report.items() if not k.startswith('flattened')}), flush=True)
sys.stdout.flush()
os._exit(0)
