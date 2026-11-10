"""The flat color and value a Blender shader socket comes to, for consumers that can carry only a constant.

Used by `export_glb.py`, whose glTF output carries a Principled BSDF's constants and not a node graph, and by the
viewer's color pass (`render/viewer_flat.py`), so a shader material looks the same in the viewer as in the exported
GLB. Pure socket walking: no `bpy` import, so anything holding Blender sockets can call it.
"""


def mean(cols):
    cols = [tuple(c) for c in cols if c is not None]
    if not cols:
        return None
    return tuple(sum(c[i] for c in cols) / len(cols) for i in range(4))


def flat_color(sock, depth=0):
    """A representative constant RGBA for a color socket, or 'image' when an image texture feeds it."""
    if depth > 8:
        return None
    if not sock.is_linked:
        v = sock.default_value
        try:
            return tuple(v) if len(v) == 4 else (v[0], v[1], v[2], 1.0)
        except TypeError:
            return (v, v, v, 1.0)
    n = sock.links[0].from_node
    if n.type == 'TEX_IMAGE':
        return 'image'
    if n.type == 'VALTORGB':
        return mean(e.color for e in n.color_ramp.elements)
    if n.type == 'RGB':
        return tuple(n.outputs[0].default_value)
    if n.type == 'MIX_RGB':
        return mean([flat_color(n.inputs['Color1'], depth + 1), flat_color(n.inputs['Color2'], depth + 1)])
    if n.type == 'MIX':
        ins = [s for s in n.inputs if s.type == 'RGBA' and s.enabled]
        return mean([flat_color(s, depth + 1) for s in ins])
    for s in n.inputs:
        if s.type == 'RGBA':
            c = flat_color(s, depth + 1)
            if c and c != 'image':
                return c
    return None


def flat_value(sock):
    if not sock.is_linked:
        return sock.default_value
    n = sock.links[0].from_node
    if n.type == 'VALTORGB':
        es = n.color_ramp.elements
        return sum(e.color[0] for e in es) / len(es)
    if n.type == 'MAP_RANGE':
        return (n.inputs['To Min'].default_value + n.inputs['To Max'].default_value) / 2
    return sock.default_value
