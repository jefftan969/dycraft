"""Camera properties glTF does not carry: a zoom and a lens shift.

glTF animates a camera's transform but not its field of view, and three.js reads no extension that would, so a shot
that zooms plays at the camera's first focal length. camera_fov() samples the vertical fov of the cut cameras over the
film; export_glb.py writes it into the cuts sidecar as {"fov": {camera: [[t, degrees], ...]}} (only cameras whose fov
changes, thinned to linear segments), for a player to interpolate. glTF cameras have no lens shift either (an
off-center frame): camera_shift() gives it as {"shift": {camera: [x, y]}}, in fractions of the image width and height
(positive: the frame moves right / up), for a player to apply as a view offset.

The cameras are evaluated in a scratch scene that holds nothing else, so the sampling costs seconds even where a
frame of the full scene takes a second to evaluate; their actions and NLA strips evaluate exactly as in the film.
"""
import math

TOL = 0.01      # degrees: linear segments stay within this of the sampled fov


def yfov(cam, aspect):
    """Vertical field of view in degrees of camera data cam for an image of this aspect (width / height)."""
    if cam.type != 'PERSP':
        return None
    # AUTO fits sensor_width to the image's longer side; HORIZONTAL / VERTICAL fit sensor_width / sensor_height
    fit = cam.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if aspect >= 1 else 'VERTICAL'
    if fit == 'HORIZONTAL':
        x = 2 * math.atan(cam.sensor_width / 2 / cam.lens)
        return math.degrees(2 * math.atan(math.tan(x / 2) / aspect))
    sensor = cam.sensor_height if cam.sensor_fit == 'VERTICAL' else cam.sensor_width
    return math.degrees(2 * math.atan(sensor / 2 / cam.lens))


def camera_shift(sc, names):
    """{camera name: [x, y]}: lens shift in fractions of the image width and height, for the shifted cut cameras."""
    import bpy
    w = sc.render.resolution_x * sc.render.pixel_aspect_x
    h = sc.render.resolution_y * sc.render.pixel_aspect_y
    out = {}
    for o in (bpy.data.objects.get(n) for n in names):
        if not o or o.type != 'CAMERA' or (abs(o.data.shift_x) < 1e-4 and abs(o.data.shift_y) < 1e-4):
            continue
        c = o.data
        # Blender's shift is in units of the fitted sensor side: the image width for HORIZONTAL (and AUTO landscape)
        fit = c.sensor_fit if c.sensor_fit != 'AUTO' else ('HORIZONTAL' if w >= h else 'VERTICAL')
        x, y = (c.shift_x, c.shift_y * w / h) if fit == 'HORIZONTAL' else (c.shift_x * h / w, c.shift_y)
        out[o.name] = [round(x, 5), round(y, 5)]
        if c.animation_data:
            print('camera_shift: shift may be animated, written as at the current frame:', o.name, flush=True)
    return out


def _thin(pts):
    """Keep the samples needed for linear interpolation to stay within TOL."""
    keep, i = [0], 0
    while i < len(pts) - 1:
        j = i + 2
        while j < len(pts):
            (t0, v0), (t1, v1) = pts[i], pts[j]
            if any(abs(v0 + (v1 - v0) * (pts[k][0] - t0) / (t1 - t0) - pts[k][1]) > TOL for k in range(i + 1, j)):
                break
            j += 1
        i = j - 1
        keep.append(i)
    return [pts[k] for k in keep]


def camera_fov(sc, names):
    """{camera name: [[t, yfov degrees], ...]} for the named cameras whose fov changes during the film."""
    import bpy
    fps = sc.render.fps / sc.render.fps_base
    aspect = (sc.render.resolution_x * sc.render.pixel_aspect_x) / (sc.render.resolution_y * sc.render.pixel_aspect_y)
    cams = [o for o in (bpy.data.objects.get(n) for n in names) if o and o.type == 'CAMERA']
    animated = [o for o in cams if o.data.animation_data and (o.data.animation_data.action or o.data.animation_data.nla_tracks
                                                                  or o.data.animation_data.drivers)]
    if not animated:
        return {}
    tmp = bpy.data.scenes.new('_cam_fov')
    try:
        for o in animated:
            tmp.collection.objects.link(o)
        tmp.frame_start, tmp.frame_end = sc.frame_start, sc.frame_end
        out = {o.name: [] for o in animated}
        for f in range(sc.frame_start, sc.frame_end + 1):
            tmp.frame_set(f)
            dg = tmp.view_layers[0].depsgraph
            for o in animated:
                v = yfov(o.evaluated_get(dg).data, aspect)
                if v is not None:
                    out[o.name].append([round((f - sc.frame_start) / fps, 4), round(v, 4)])
    finally:
        bpy.data.scenes.remove(tmp)
    return {k: _thin(v) for k, v in out.items() if v and max(x[1] for x in v) - min(x[1] for x in v) > TOL}

