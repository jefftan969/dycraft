"""Render-visibility helpers shared by the bake and export scripts."""
from contextlib import contextmanager


def _fcurves(a):
    try:
        yield from a.fcurves
    except AttributeError:
        pass
    for l in getattr(a, 'layers', ()):
        for st in l.strips:
            for cb in getattr(st, 'channelbags', ()):
                yield from cb.fcurves


def _drop_keys(o, path):
    ad = o.animation_data
    acts = ([ad.action] if ad.action else []) + [st.action for t in ad.nla_tracks for st in t.strips if st.action]
    for a in acts:
        try:
            for fc in [fc for fc in a.fcurves if fc.data_path == path]:
                a.fcurves.remove(fc)
        except AttributeError:
            pass
        for l in getattr(a, 'layers', ()):
            for st in l.strips:
                for cb in getattr(st, 'channelbags', ()):
                    for fc in [fc for fc in cb.fcurves if fc.data_path == path]:
                        cb.fcurves.remove(fc)


def _keys_visible(o):
    ad = o.animation_data
    acts = ([ad.action] if ad.action else []) + [st.action for t in ad.nla_tracks for st in t.strips if st.action]
    return any(kp.co[1] < 0.5 for a in acts for fc in _fcurves(a) if fc.data_path == 'hide_render'
               for kp in fc.keyframe_points)


def _vis_drivers(o):
    ad = o.animation_data
    return [d for d in ad.drivers if d.data_path == 'hide_render' and not d.mute] if ad else []


def vis_keyed(o):
    """hide_render keyed in the active action or in any NLA strip (several runs drive visibility through NLA), or
    set by a driver (from a controller's animated property)."""
    ad = o.animation_data
    if not ad:
        return False
    if _vis_drivers(o):
        return True
    acts = [ad.action] if ad.action else []
    acts += [s.action for t in ad.nla_tracks for s in t.strips if s.action]
    return any(fc.data_path == 'hide_render' for a in acts for fc in _fcurves(a))


@contextmanager
def cheap_eval(keep=()):
    """While inside: shape-key animation muted and modifiers off, except on the objects in keep. For passes over the
    timeline that read object properties (visibility) or only a few objects' geometry. A per-frame shape-key cache
    is what makes a frame expensive: thousands of keys on a few effect meshes make a frame about 15 times slower."""
    import bpy
    keep = {o.name for o in keep}
    keys = {o.data.shape_keys for o in bpy.data.objects if o.type == 'MESH' and o.name in keep and o.data.shape_keys}
    saved = []
    for k in bpy.data.shape_keys:
        ad = k.animation_data
        if ad and k not in keys:
            saved.append((ad, ad.action, getattr(ad, 'action_slot', None), [(t, t.mute) for t in ad.nla_tracks]))
            ad.action = None
            for t in ad.nla_tracks:
                t.mute = True
    mods = [(m, m.show_viewport) for o in bpy.data.objects if o.name not in keep for m in getattr(o, 'modifiers', ())]
    for m, _ in mods:
        m.show_viewport = False
    try:
        yield
    finally:
        for ad, act, slot, tracks in saved:
            ad.action = act
            if slot is not None and getattr(ad, 'action_slot', None) != slot:
                ad.action_slot = slot
            for t, mute in tracks:
                t.mute = mute
        for m, v in mods:
            m.show_viewport = v


def _vis_states(sc, objs, step=3):
    """Evaluated hide_render values per object over the timeline. Keys alone mislead: strips can have
    zero influence or be overridden, leaving e.g. a wireframe helper hidden for the whole film."""
    seen = {o.name: set() for o in objs}
    cur = sc.frame_current
    with cheap_eval():                  # visibility needs no geometry
        for f in list(range(sc.frame_start, sc.frame_end + 1, step)) + [sc.frame_end]:
            sc.frame_set(f)
            for o in objs:
                seen[o.name].add(bool(o.hide_render))
    sc.frame_set(cur)
    return seen


_ANIM = {}
_MUTED = []


def restore_drivers():
    """Unmute the visibility drivers prepare_visibility(for_export=False) muted (before saving the baked blend)."""
    for d in _MUTED:
        d.mute = False
    _MUTED.clear()


def vis_animated(o):
    """True when render visibility actually changes during the film (set by prepare_visibility)."""
    return _ANIM.get(o.name, False)


def prepare_visibility(sc, for_export, states=None):
    """Hide never-rendered objects; keep ones whose render visibility is animated. Returns the visibility states
    ({object: [hidden values seen]}); pass them back as states= to skip the scan (the export reuses the bake's).

    For export, an animated hide_render becomes a delta_scale driver (hidden -> near-zero scale),
    since glTF has no visibility channel."""
    keyed = [o for o in sc.objects if vis_keyed(o)]
    if states is None:
        states = _vis_states(sc, keyed) if keyed else {}
    else:
        states = {k: set(v) for k, v in states.items() if k in sc.objects}
    _ANIM.clear()
    _ANIM.update({k: len(v) == 2 for k, v in states.items()})
    n_hidden = n_anim = 0
    for o in sc.objects:
        st = states.get(o.name)
        if st is not None and len(st) == 1:
            hidden = next(iter(st))
            # Evaluated hidden throughout, yet keyed visible somewhere (fills shown by NLA strips): the film shows these,
            # so keep them; wire/bounds viewport helpers stay hidden.
            if hidden and o.display_type not in ('WIRE', 'BOUNDS') and _keys_visible(o):
                hidden = False
            # constant over the film: freeze it. Leave its keys alone: NLA actions can be shared between
            # objects, and removing one object's curves would freeze the others too.
            o.hide_render = hidden
        if vis_animated(o):
            n_anim += 1
            o.hide_set(False)
            # FX often also key the monitor toggle (in NLA too); the exporter's use_visible honors it,
            # so drop those keys or every frame_set hides the object again.
            _drop_keys(o, 'hide_viewport')
            o.hide_viewport = False
            if for_export:
                for i in range(3):
                    d = o.driver_add('delta_scale', i).driver
                    d.type = 'SCRIPTED'
                    v = d.variables.new()
                    v.name = 'h'
                    v.targets[0].id = o
                    v.targets[0].data_path = 'hide_render'
                    d.expression = '1 - h * 0.9999'
            else:
                # to bake it, it must render; a driver would set it back at every evaluation
                for d in _vis_drivers(o):
                    d.mute = True
                    _MUTED.append(d)
                o.hide_render = False
        elif o.hide_render:
            n_hidden += 1
            o.hide_set(True)
    print('visibility: hidden', n_hidden, 'animated', n_anim, flush=True)
    return {k: sorted(v) for k, v in states.items()}
