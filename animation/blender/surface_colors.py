"""The color a Blender material shows, read from its node tree, for the evaluated records and the display."""
import numpy as np


#: How far upstream of a linked `Base Color` to look for a constant color, and
#: how many to average. Both are bounds rather than tuning knobs: the authored
#: graphs in this corpus are 5-9 nodes, so 6 hops reaches every constant one of
#: them has, and a cap keeps a pathological graph from turning one material into
#: a long walk.
_COLOR_WALK_DEPTH = 6
_COLOR_WALK_MAX = 32


def _constant_colors(socket, depth=0, seen=None, out=None):
    """Constant RGB sources feeding `socket`, walking upstream.

    Bounded by `_COLOR_WALK_DEPTH` and cycle-safe on node identity — Blender
    will not build a cyclic shader graph, but this runs on authored `exec`'d
    code and a guard is cheaper than trusting it.

    Two node types hold their color somewhere other than an input socket and
    are read specially; everything else follows one uniform rule, which is what
    makes this cover node types nobody has written yet: for each RGBA input,
    recurse if it is linked and take its `default_value` if it is not.
    """
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if depth > _COLOR_WALK_DEPTH or len(out) >= _COLOR_WALK_MAX:
        return out
    if not socket.is_linked or not socket.links:
        return out
    node = socket.links[0].from_node
    # The node's C pointer, not `id(node)`: bpy hands out a fresh Python
    # wrapper on every access and frees it as soon as nothing holds it, so
    # `id()` of one node is soon reused by another, which this walk then
    # skips as already seen. Which nodes were skipped depended on the
    # process's allocation history, so the same brick-wall material averaged
    # to gray on one call and brown on the next -- and a resident compile and
    # a fresh final render signed the same frame differently.
    key = node.as_pointer() if hasattr(node, "as_pointer") else id(node)
    if key in seen:
        return out
    seen.add(key)

    kind = getattr(node, "type", "")
    if kind == "VALTORGB":
        # A ColorRamp's colors are its elements, not its inputs. Averaging the
        # stops is the whole ramp's central tendency, which is what a flat
        # stand-in wants; the fac driving it is noise we cannot evaluate.
        ramp = getattr(node, "color_ramp", None)
        for element in (getattr(ramp, "elements", None) or ()):
            out.append(tuple(element.color)[:3])
            if len(out) >= _COLOR_WALK_MAX:
                return out
        return out
    if kind == "RGB":
        for output in node.outputs:
            out.append(tuple(output.default_value)[:3])
            break
        return out
    if kind == "TEX_IMAGE":
        # An image is a measurable color, not an unknowable one: its mean is
        # what a flat stand-in should show. Without this every textured asset
        # -- each imported character -- fell through to the socket scalar and
        # drew 0.8 white in Workbench drafts and the viewer (a black crow
        # rendered as a white mesh in its previews).
        mean = _image_mean_linear(getattr(node, "image", None))
        if mean is not None:
            out.append(mean)
        return out

    for socket_in in node.inputs:
        if getattr(socket_in, "type", "") != "RGBA":
            continue
        if socket_in.is_linked:
            _constant_colors(socket_in, depth + 1, seen, out)
        else:
            out.append(tuple(socket_in.default_value)[:3])
        if len(out) >= _COLOR_WALK_MAX:
            break
    return out


#: `_image_mean_linear` per image name, for the life of this worker process.
_IMAGE_MEAN_CACHE = {}
#: Pixels sampled per image at most; a 4K map is 16M pixels and the mean of a
#: strided sample is as good a flat stand-in as the mean of all of them.
_IMAGE_MEAN_SAMPLES = 1 << 18


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _image_mean_linear(image):
    """Mean RGB of `image` in scene-linear, or None if it has no pixels.

    `Image.pixels` hands back an 8-bit image as stored -- byte/255, i.e.
    sRGB-encoded for an sRGB image (measured: a PNG of byte 128 reads 0.502)
    -- while Base Color and `diffuse_color` are linear. So each pixel is
    linearized before averaging; a float image or a non-color one is taken
    as it is.
    """
    if image is None:
        return None
    key = image.name
    if key in _IMAGE_MEAN_CACHE:
        return _IMAGE_MEAN_CACHE[key]
    mean = None
    try:
        w, h = image.size
        channels = int(image.channels)
        if w and h and channels >= 3:
            px = np.empty(w * h * channels, dtype=np.float32)
            image.pixels.foreach_get(px)
            px = px.reshape(-1, channels)[:, :3]
            step = max(1, len(px) // _IMAGE_MEAN_SAMPLES)
            px = px[::step].astype(np.float64)
            space = getattr(getattr(image, "colorspace_settings", None), "name", "")
            if space == "sRGB" and not getattr(image, "is_float", False):
                px = _srgb_to_linear(np.clip(px, 0.0, 1.0))
            mean = tuple(float(v) for v in px.mean(axis=0))
    except Exception:  # an unloadable or packed-but-missing image: no color
        mean = None
    _IMAGE_MEAN_CACHE[key] = mean
    return mean


def _linked_base_color(socket):
    """A representative RGB for a linked `Base Color`, or None if unknowable."""
    colors = _constant_colors(socket)
    if not colors:
        return None
    n = len(colors)
    return [min(1.0, max(0.0, sum(c[i] for c in colors) / n)) for i in range(3)]


#: `surface_color` per material name, for the length of this worker process.
#:
#: `evaluated_frame` asks once per object per frame, so a full evaluate makes
#: many more calls than there are materials; without this the graph walk would
#: be repeated for every one of them. The cache must live no longer than one
#: job: a material is edited in place between jobs under the same name, so a
#: resident worker (`serve`) resets it before every request, as a one-shot
#: worker does by starting fresh.
_SURFACE_COLOR_CACHE = {}


def reset_surface_color_cache():
    _SURFACE_COLOR_CACHE.clear()
    _IMAGE_MEAN_CACHE.clear()


def surface_color(mat):
    """One flat RGBA for a material, for consumers that cannot run its shader.

    The viewer draws evaluated geometry without Blender, so it needs a color per object; this is the only appearance
    the record carries.

    **The Principled BSDF's Base Color first, `diffuse_color` only as a fallback.** `diffuse_color` is the viewport
    solid-mode swatch, and nothing writes it (`render/_blender_child.py::_build_material` and agent scripts author
    through the node tree), so it sits at Blender's default 0.8 gray and would draw whole scenes in one flat gray.

    **A linked Base Color is read from the graph, not from the socket.** An author does not give a shader material a
    color, because the graph owns that socket (`sdk/_material.SHADER_OWNED_FIELDS`), so `_world._material_to_render`
    fills the packed scalar with `DEFAULT_PART_COLOR` and `_apply_scalars` writes that onto the socket. A shader that
    links Base Color therefore leaves a value no author chose, and every such part would land on one gray. So this
    looks upstream for a constant color (`_constant_colors`, which reads an image texture as its mean color), and the
    socket scalar is the last resort, for a graph holding no measurable color.

    One color cannot carry wood grain, grout, or brick courses, and the average ignores the graph's weights: it is a
    stand-in for consumers that cannot run the shader, and nothing else reads it.
    """
    if mat is None:
        return [0.6, 0.6, 0.6, 1]
    cached = _SURFACE_COLOR_CACHE.get(mat.name)
    if cached is not None:
        return list(cached)
    value = _surface_color_uncached(mat)
    _SURFACE_COLOR_CACHE[mat.name] = tuple(value)
    return value


def _surface_color_uncached(mat):
    if mat.use_nodes:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None and "Base Color" in bsdf.inputs:
            base = bsdf.inputs["Base Color"]
            # Alpha from the material, not from Base Color: Principled keeps
            # opacity in its own socket and leaves this one at 1.
            alpha = 1.0
            if "Alpha" in bsdf.inputs and not bsdf.inputs["Alpha"].is_linked:
                alpha = float(bsdf.inputs["Alpha"].default_value)
            rgb = _linked_base_color(base) if base.is_linked else None
            if rgb is None:
                r, g, b, _ = base.default_value
                rgb = [float(r), float(g), float(b)]
            return [rgb[0], rgb[1], rgb[2], alpha]
    return list(mat.diffuse_color)
