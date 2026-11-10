"""Fast bpy regressions for conservative evaluated geometry reuse."""
from pathlib import Path
import subprocess

import pytest


def test_static_mesh_cache_preserves_evaluated_geometry(tmp_path):
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('Blender interpreter unavailable')
    code = r'''
import bpy
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('worker', WORKER)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
bpy.ops.mesh.primitive_cube_add()
rigid = bpy.context.object
rigid.name = 'rigid'
rigid.location.x = 0; rigid.keyframe_insert('location', frame=1)
rigid.location.x = 2; rigid.keyframe_insert('location', frame=4)
rigid.scale = (1,1,1); rigid.keyframe_insert('scale', frame=1)
rigid.scale = (-2,3,1); rigid.keyframe_insert('scale', frame=4)
shared = bpy.data.objects.new('shared', rigid.data); s.collection.objects.link(shared)
bpy.ops.mesh.primitive_cube_add()
soft = bpy.context.object; soft.name = 'soft'
soft.shape_key_add(name='Basis')
k = soft.shape_key_add(name='deform'); k.data[0].co.z += 1
k.value = 0; k.keyframe_insert('value', frame=1)
k.value = 1; k.keyframe_insert('value', frame=4)
bpy.ops.mesh.primitive_cube_add()
modified = bpy.context.object; modified.name = 'modifier'
mod = modified.modifiers.new('animated', 'DISPLACE')
mod.strength = 0; mod.keyframe_insert('strength', frame=1)
mod.strength = 1; mod.keyframe_insert('strength', frame=4)
cache = worker.FrameGeometryCache()
first = None
soft_first = None
for frame in range(1,5):
    s.frame_set(frame)
    record, local = worker.evaluated_frame(frame, cache)
    expected, direct = worker.evaluated_frame(frame)
    assert record == expected
    for name in local:
        assert all(np.array_equal(a,b) for a,b in zip(local[name],direct[name])), name
    assert local['rigid'] is local['shared']
    if first is None:
        first = local['rigid']; soft_first = local['soft'][0].copy()
    else:
        assert local['rigid'] is first, 'rigid transform invalidated local mesh'
assert not np.array_equal(soft_first, local['soft'][0])
assert cache.hits >= 7
assert cache.builds == 9, cache.builds # one plain mesh plus 2 changing objects/frame
# Direct mesh edits and mesh datablock replacement cannot reuse stale arrays.
rigid.data.vertices[0].co.z += 2
rigid.data.update(); bpy.context.view_layer.update()
record, local = worker.evaluated_frame(4, cache)
assert local['rigid'] is not first
_, direct = worker.evaluated_frame(4)
assert np.array_equal(local['rigid'][0],direct['rigid'][0])
first = local['rigid']
rigid.data = rigid.data.copy()
rigid.data.vertices[0].co.x += 3
rigid.data.update(); bpy.context.view_layer.update()
_, local = worker.evaluated_frame(4, cache)
assert local['rigid'] is not first
assert not np.array_equal(local['rigid'][0],local['shared'][0])
# Topology edits invalidate the same datablock, not just coordinate edits.
rigid.data.clear_geometry()
rigid.data.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
rigid.data.update(); bpy.context.view_layer.update()
_, local = worker.evaluated_frame(4, cache)
_, direct = worker.evaluated_frame(4)
assert local['rigid'][1].shape == (1,3)
assert all(np.array_equal(a,b) for a,b in zip(local['rigid'], direct['rigid']))
# Handlers may mutate mesh on skipped frames, so disable reuse altogether.
def mutate(scene):
    rigid.data.vertices[0].co.y += 1
    rigid.data.update()
bpy.app.handlers.frame_change_post.append(mutate)
s.frame_set(5)
_, local = worker.evaluated_frame(5, cache)
_, direct = worker.evaluated_frame(5)
assert np.array_equal(local['rigid'][0], direct['rigid'][0])
assert not cache.enabled and not cache.entries
bpy.app.handlers.frame_change_post.remove(mutate)
# Bounded cache may omit meshes without altering output.
cache.close()
# Importer's Edge Split remains cacheable, including evaluated animated settings.
edge_cache = worker.FrameGeometryCache()
edge = shared.modifiers.new('shading', 'EDGE_SPLIT')
edge.split_angle = .1; edge.keyframe_insert('split_angle', frame=6)
edge.split_angle = 2.; edge.keyframe_insert('split_angle', frame=7)
edge_counts = []
for frame in (6,7,8):
    s.frame_set(frame)
    _, local = worker.evaluated_frame(frame, edge_cache)
    _, direct = worker.evaluated_frame(frame)
    for name in local:
        assert all(np.array_equal(a,b) for a,b in zip(local[name],direct[name]))
    edge_counts.append(len(local['shared'][0]))
assert edge_counts[0] != edge_counts[1]
edge_cache.close()
assert edge_cache._handler not in bpy.app.handlers.depsgraph_update_post
small = worker.FrameGeometryCache(max_bytes=1)
_, local = worker.evaluated_frame(5, small)
_, direct = worker.evaluated_frame(5)
assert not small.entries
for name in local:
    assert all(np.array_equal(a,b) for a,b in zip(local[name],direct[name]))
'''
    code += '\nsmall.close()\n'
    code = code.replace('WORKER', repr(str(Path(__file__).resolve().parents[1] / 'animation/blender/worker.py')))
    script = tmp_path / 'cache_check.py'
    script.write_text(code)
    result = subprocess.run([interpreter, str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_modifiers_reuse_evaluated_geometry_but_track_their_settings(tmp_path):
    """Bevel and Solidify are mesh-local, so their output may be reused across frames.

    Rebuilding them on every frame produces the same few meshes over and over.
    Two things keep the reuse honest and both are asserted below: the descriptor is
    read from the EVALUATED modifier, so an animated or driven setting invalidates
    on the frame it changes; and a modifier carrying an input the key cannot see
    (an object, a texture) is refused outright rather than guessed at.

    Vertex counts are not the assertion anywhere here: a cube's bevel has 24
    vertices at every width, so only the arrays themselves show the difference.
    """
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('Blender interpreter unavailable')
    code = r'''
import bpy
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('worker', WORKER)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene

bpy.ops.mesh.primitive_cube_add()
bevelled = bpy.context.object; bevelled.name = 'bevelled'
bevelled.modifiers.new('bevel', 'BEVEL').width = .05
bevelled.location.x = 0; bevelled.keyframe_insert('location', frame=1)
bevelled.location.x = 3; bevelled.keyframe_insert('location', frame=4)

bpy.ops.mesh.primitive_cube_add()
shelled = bpy.context.object; shelled.name = 'shelled'
shelled.modifiers.new('shell', 'SOLIDIFY').thickness = .02

bpy.ops.mesh.primitive_cube_add()
moving = bpy.context.object; moving.name = 'animated_bevel'
mod = moving.modifiers.new('bevel', 'BEVEL')
mod.width = .01; mod.keyframe_insert('width', frame=1)
mod.width = .4;  mod.keyframe_insert('width', frame=4)

# An ID-pointer input is exactly what the whitelist must refuse.
bpy.ops.mesh.primitive_cube_add()
arrayed = bpy.context.object; arrayed.name = 'arrayed'
arrayed.modifiers.new('array', 'ARRAY')

cache = worker.FrameGeometryCache()
held, animated = {}, []
for frame in range(1, 5):
    s.frame_set(frame)
    record, local = worker.evaluated_frame(frame, cache)
    _, direct = worker.evaluated_frame(frame)
    for name in local:
        assert all(np.array_equal(a, b) for a, b in zip(local[name], direct[name])), name
    # A moving object with a static local modifier hands back the very arrays.
    for name in ('bevelled', 'shelled'):
        if name in held:
            assert local[name] is held[name], name + ' rebuilt despite unchanged geometry'
        held[name] = local[name]
    animated.append(local['animated_bevel'][0].copy())
assert not np.array_equal(animated[0], animated[-1]), 'animated bevel width must invalidate'
deps = bpy.context.evaluated_depsgraph_get()
assert cache.key(arrayed, arrayed.evaluated_get(deps)) is None

# Two objects on one mesh with different settings must not collide.
twin = bpy.data.objects.new('twin', bevelled.data); s.collection.objects.link(twin)
twin.modifiers.new('bevel', 'BEVEL').width = .3
s.frame_set(1)
_, local = worker.evaluated_frame(1, cache)
_, direct = worker.evaluated_frame(1)
assert not np.array_equal(local['twin'][0], local['bevelled'][0]), 'shared mesh, different width'
assert all(np.array_equal(a, b) for a, b in zip(local['twin'], direct['twin']))

# An input the descriptor cannot summarise must still not produce stale arrays:
# editing a custom profile tags the object's geometry, and the depsgraph handler
# drops the entry. The invariant is the arrays, not the key.
profiled = bpy.data.objects.new('profiled', bevelled.data); s.collection.objects.link(profiled)
curved = profiled.modifiers.new('bevel', 'BEVEL')
curved.width = .3; curved.segments = 6; curved.profile_type = 'CUSTOM'
s.frame_set(1)
worker.evaluated_frame(1, cache)
curved.custom_profile.points[1].location = (.2, .8)
curved.custom_profile.update()
bpy.context.view_layer.update()
_, local = worker.evaluated_frame(1, cache)
_, direct = worker.evaluated_frame(1)
assert all(np.array_equal(a, b) for a, b in zip(local['profiled'], direct['profiled']))
cache.close()
'''
    code = code.replace('WORKER', repr(str(Path(__file__).resolve().parents[1] / 'animation/blender/worker.py')))
    script = tmp_path / 'local_modifier_cache.py'
    script.write_text(code)
    # A bpy child that raises hangs in teardown, so a failure here surfaces as a
    # timeout; keep it long enough that a slow machine is not mistaken for one.
    result = subprocess.run([interpreter, str(script)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bone_parented_props_text_and_curves_are_reused_and_invalidated(tmp_path):
    """A prop parented to one bone moves rigidly; text and curves are their own data.

    Re-reading all three on every frame can be most of a film's record
    extraction. Deform parenting, a curve that reads another object, and an edit must still
    produce what a fresh evaluation does.
    """
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('Blender interpreter unavailable')
    code = r'''
import bpy
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('worker', WORKER)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene

arm_data = bpy.data.armatures.new('rig')
rig = bpy.data.objects.new('rig', arm_data); s.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
bone = arm_data.edit_bones.new('hand'); bone.head = (0, 0, 0); bone.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode='OBJECT')
pose = rig.pose.bones['hand']
for frame, angle in ((1, 0.), (4, 1.2)):
    pose.rotation_mode = 'XYZ'; pose.rotation_euler = (angle, 0, 0)
    pose.keyframe_insert('rotation_euler', frame=frame)

bpy.ops.mesh.primitive_cube_add(size=.2)
prop = bpy.context.object; prop.name = 'prop'
prop.parent = rig; prop.parent_type = 'BONE'; prop.parent_bone = 'hand'

bpy.ops.mesh.primitive_cube_add(size=.5)
skin = bpy.context.object; skin.name = 'skin'
skin.parent = rig; skin.parent_type = 'ARMATURE'
group = skin.vertex_groups.new(name='hand'); group.add(range(len(skin.data.vertices)), 1., 'REPLACE')

text = bpy.data.objects.new('label', bpy.data.curves.new('label', 'FONT'))
s.collection.objects.link(text); text.data.body = 'MENU'; text.data.extrude = .01
text.location.x = 1; text.keyframe_insert('location', frame=1)
text.location.x = 2; text.keyframe_insert('location', frame=4)

path = bpy.data.curves.new('rail', 'CURVE'); path.dimensions = '3D'; path.bevel_depth = .02
spline = path.splines.new('POLY'); spline.points.add(2)
for n, co in enumerate(((0, 0, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1))):
    spline.points[n].co = co
rail = bpy.data.objects.new('rail', path); s.collection.objects.link(rail)

profile = bpy.data.objects.new('profile', bpy.data.curves.new('profile', 'CURVE'))
s.collection.objects.link(profile)
sp = profile.data.splines.new('POLY'); sp.points.add(2)
for n, co in enumerate(((0, 0, 0, 1), (.05, 0, 0, 1), (0, .05, 0, 1))):
    sp.points[n].co = co
bevelled_path = bpy.data.curves.new('tube', 'CURVE'); bevelled_path.dimensions = '3D'
bevelled_path.bevel_mode = 'OBJECT'; bevelled_path.bevel_object = profile
tsp = bevelled_path.splines.new('POLY'); tsp.points.add(1)
tsp.points[0].co = (0, 0, 1, 1); tsp.points[1].co = (1, 0, 1, 1)
tube = bpy.data.objects.new('tube', bevelled_path); s.collection.objects.link(tube)

cache = worker.FrameGeometryCache()
deps = bpy.context.evaluated_depsgraph_get()
assert cache.key(skin, skin.evaluated_get(deps)) is None, 'deform parenting must stay evaluated'
assert cache.key(tube, tube.evaluated_get(deps)) is None, 'a bevel object is an unseen input'
for name in ('prop', 'label', 'rail'):
    obj = bpy.data.objects[name]
    assert cache.key(obj, obj.evaluated_get(deps)) is not None, name
held, skins = {}, []
for frame in range(1, 5):
    s.frame_set(frame)
    record, local = worker.evaluated_frame(frame, cache)
    expected, direct = worker.evaluated_frame(frame)
    assert record == expected
    for name in local:
        assert all(np.array_equal(a, b) for a, b in zip(local[name], direct[name])), name
    for name in ('prop', 'label', 'rail'):
        if name in held:
            assert local[name] is held[name], name + ' rebuilt though its own geometry did not change'
        held[name] = local[name]
    skins.append(local['skin'][0].copy())
assert not np.array_equal(skins[0], skins[-1]), 'the deformed mesh must follow the bone'
# Editing text or curve data invalidates them like a mesh edit.
text.data.body = 'MENU 2'
path.splines[0].points[2].co = (1, 2, 0, 1)
bpy.context.view_layer.update()
_, local = worker.evaluated_frame(4, cache)
_, direct = worker.evaluated_frame(4)
for name in ('label', 'rail'):
    assert local[name] is not held[name], name + ' kept stale triangles after an edit'
    assert all(np.array_equal(a, b) for a, b in zip(local[name], direct[name])), name
cache.close()
'''
    code = code.replace('WORKER', repr(str(Path(__file__).resolve().parents[1] / 'animation/blender/worker.py')))
    script = tmp_path / 'cache_check.py'
    script.write_text(code)
    result = subprocess.run([interpreter, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_body_blender_did_not_touch_is_reused_and_one_it_did_is_read_again(tmp_path):
    """The evaluation's stream reuses a body's last extraction when the frame's
    depsgraph updates name none of its inputs, and reads it again when they do:
    a transform, an animated visibility, a keyed material color, a shape key,
    and an edit between frames. Every frame's record and arrays are a fresh
    extraction's. Another handler turns reuse off, as it does the mesh cache."""
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip('Blender interpreter unavailable')
    code = r'''
import bpy
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('worker', WORKER)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
def cube(name, x):
    bpy.ops.mesh.primitive_cube_add(location=(x, 0, 0))
    obj = bpy.context.object; obj.name = name
    return obj
still, mover, blink, tinted, soft = (cube(n, 3 * i) for i, n in enumerate(('still', 'mover', 'blink', 'tinted', 'soft')))
mover.location.z = 0; mover.keyframe_insert('location', frame=1)
mover.location.z = 1; mover.keyframe_insert('location', frame=3)
blink.hide_render = False; blink.keyframe_insert('hide_render', frame=1)
blink.hide_render = True; blink.keyframe_insert('hide_render', frame=4)
material = bpy.data.materials.new('paint'); material.use_nodes = True
base = material.node_tree.nodes['Principled BSDF'].inputs['Base Color']
base.default_value = (1, 0, 0, 1); base.keyframe_insert('default_value', frame=2)
base.default_value = (0, 0, 1, 1); base.keyframe_insert('default_value', frame=5)
tinted.data.materials.append(material)
still.data.materials.append(bpy.data.materials.new('plain'))
soft.shape_key_add(name='Basis'); key = soft.shape_key_add(name='bend'); key.data[0].co.z += 1
key.value = 0; key.keyframe_insert('value', frame=4)
key.value = 1; key.keyframe_insert('value', frame=6)
calls = []
import frame_export   # evaluated_frame reads surface_color from its own module
color = frame_export.surface_color
frame_export.surface_color = lambda mat: calls.append(mat.name if mat else None) or color(mat)
cache = worker.FrameGeometryCache()
def same(frame):
    """The stream's extraction against a fresh one; returns the materials the stream read."""
    calls.clear()
    record, local, _ = worker.evaluated_frame(frame, cache, with_transforms=True)
    read = list(calls)
    expected, direct = worker.evaluated_frame(frame)
    assert record == expected, frame
    for name in direct:
        assert all(np.array_equal(a, b) for a, b in zip(local[name], direct[name])), (frame, name)
    return read
for frame in range(1, 8):
    s.frame_set(frame)
    read = same(frame)
    if frame >= 2:
        assert 'plain' not in read, (frame, read)
    else:
        assert 'plain' in read
# An edit between frames is an update too.
still.data.materials[0].diffuse_color = (0, 1, 0, 1)
bpy.context.view_layer.update()
s.frame_set(7)
assert 'plain' in same(7)
# Another handler: nothing is reused.
def other(scene):
    pass
bpy.app.handlers.frame_change_pre.append(other)
for frame in (6, 7):
    s.frame_set(frame)
    assert 'plain' in same(frame)
bpy.app.handlers.frame_change_pre.remove(other)
cache.close()
assert cache._frame_handler not in bpy.app.handlers.frame_change_post
'''
    code = code.replace('WORKER', repr(str(Path(__file__).resolve().parents[1] / 'animation/blender/worker.py')))
    script = tmp_path / 'held_check.py'
    script.write_text(code)
    result = subprocess.run([interpreter, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
