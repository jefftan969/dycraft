"""One extra view of the model under review, for the reviewer sub-agent.

    python _review_render.py --ir <ir.json> --out-dir <dir> \
        --on <part|all> --azimuth <deg> --elevation <deg> \
        [--only NAME ...] [--poses JSON] [--resolution W H] [--fov-deg D]

Writes `<out-dir>/extra_<n>_<on>_az<a>_el<e>[_iso][_posed].png` and prints its
path.

**Why this has `--only`, `--poses`, `--resolution` and `--fov-deg.`** A camera
can be moved; it cannot see through the model's own geometry, and no angle
reveals a bed plate under a cloth or a drawer bottom under a tabletop. With
framing, azimuth, and elevation alone, a reviewer asked to judge an interior
surface has one honest answer, `cant_tell`, and the model occluding itself is
the most common reason a reviewer cannot settle a claim. The authoring agent's
`snapshot` has `only` — "names the parts to draw and hides the rest, which is
what a contact face or an interior fit needs" — and the reviewer gets the same
means of looking.

The validation and unit conversion are `snapshot._snapshot_{only,poses,
resolution}`, imported rather than copied, so the reviewer's semantics are the
agent's semantics and cannot drift from them. Degrees for a revolute joint and
meters for a prismatic one, on both sides.

Run as a fresh process per camera on purpose: open3d segfaults if a second
`OffscreenRenderer` is constructed in one process, so one process per
invocation sidesteps that rather than fighting it. The cap is enforced here,
by counting what is already on disk, rather than by asking the reviewer
politely in its prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CAP = 4


def _frame_bbox(ir: dict, on: str, catalog):
    """World AABB to aim at: the whole model, or one named entity.

    An object's entities are parts and a scene's are instances, and the two
    live in different kernels — the scene one because a SceneIR carries no
    geometry of its own, only refs into the catalog.
    """
    if catalog is None:
        from ...sdk._object_kernel import global_bbox
        return global_bbox(ir) if on in ("", "all") else global_bbox(ir, on)
    from ...sdk import _scene_kernel as _sk
    if on in ("", "all"):
        return _sk.scene_bbox(ir, catalog)
    return _sk.instance_world_aabb(ir, catalog, on)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--on", default="all")
    ap.add_argument("--azimuth", type=float, required=True)
    ap.add_argument("--elevation", type=float, required=True)
    ap.add_argument("--asset-base", default=None)
    # A scene's geometry lives in a catalog of ObjectIRs the SceneIR names by
    # hash, so without this the reviewer of a scene gets an empty frame. Its
    # presence is also what selects scene mode: an object review passes none.
    ap.add_argument("--catalog", default=None,
                    help="directory of ObjectIR json files (scene review)")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--only", nargs="*", default=None, metavar="NAME",
                    help="draw only these parts (object) or instances "
                         "(scene) and hide the rest — for an interior "
                         "surface or a contact face no camera can see")
    ap.add_argument("--poses", default=None, metavar="JSON",
                    help='joint positions for this render, e.g. '
                         '\'{"lid_hinge": 90}\' — degrees for a revolute '
                         "joint, meters for a prismatic one")
    ap.add_argument("--resolution", nargs=2, type=int, default=None,
                    metavar=("W", "H"),
                    help="per-call framebuffer, for a detail too small to "
                         "read at the default")
    ap.add_argument("--fov-deg", type=float, default=None,
                    help="vertical field of view; a narrow one flattens the "
                         "perspective that makes containment ambiguous")
    a = ap.parse_args(argv[1:])

    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = sorted(out_dir.glob("extra_*.png"))
    if len(taken) >= a.cap:
        print(f"refused: {len(taken)} of {a.cap} extra cameras already "
              f"rendered. Decide from the images you have, and answer "
              f"cant_tell for anything they cannot settle.")
        return 1

    from ...render import (FULL_RESOLUTION, RenderRequest, RenderUnavailable,
                           framed_camera, render_request)

    ir = json.loads(Path(a.ir).read_text())
    catalog = None
    if a.catalog:
        catalog = {}
        for path in sorted(Path(a.catalog).glob("*.json")):
            if path.name == "manifest.json":
                continue
            obj = json.loads(path.read_text())
            catalog[obj.get("id") or path.stem] = obj

    layer = "scene" if catalog is not None else "object"

    # Imported here, not at module scope: this script is spawned per camera
    # and `tools` is a heavy import. Reused rather than reimplemented so the
    # reviewer's `only`/`poses`/`resolution` mean exactly what the author's
    # `snapshot` means — a second copy of the pose conversion is a second
    # thing that can silently stop agreeing.
    from .snapshot import _snapshot_only, _snapshot_poses, _snapshot_resolution

    only_parts = _snapshot_only(ir, a.only or None, layer=layer)
    if isinstance(only_parts, str):
        print(only_parts)
        return 1

    ir_poses, pose_label = None, "default"
    if a.poses:
        try:
            surface_poses = json.loads(a.poses)
        except json.JSONDecodeError as e:
            print(f"--poses must be a JSON object: {e}")
            return 1
        got = _snapshot_poses(ir, surface_poses, catalog=catalog, layer=layer)
        if isinstance(got, str):
            print(got)
            return 1
        ir_poses, pose_label = got

    res = _snapshot_resolution(a.resolution)
    if isinstance(res, str):
        print(res)
        return 1
    if res is None:
        res = FULL_RESOLUTION

    try:
        bbox = _frame_bbox(ir, a.on, catalog)
    except Exception as e:  # an unknown entity name is the normal failure
        if catalog is not None:
            names = [i.get("name") for i in (ir.get("instances") or [])]
            print(f"cannot frame on {a.on!r}: {e}. Instances: {names}")
        else:
            names = [p.get("name") for p in (ir.get("parts") or [])]
            print(f"cannot frame on {a.on!r}: {e}. Parts: {names}")
        return 1
    try:
        # `resolution=` for the reason `canonical_cameras` documents: the fit
        # is solved against the frame's aspect, and this render's frame is the
        # request default rather than whatever `framed_camera` assumes.
        cam_kw = {} if a.fov_deg is None else {"fov_deg": a.fov_deg}
        cam = framed_camera(bbox, azimuth_deg=a.azimuth,
                            elevation_deg=a.elevation,
                            resolution=res, **cam_kw)
    except (TypeError, ValueError) as e:
        print(f"bad camera: {e}")
        return 1
    try:
        result = render_request(RenderRequest(
            layer=layer, ir=ir, catalog=catalog, target="all",
            out_dir=out_dir, step=len(taken) + 1,
            asset_base_dir=(Path(a.asset_base) if a.asset_base else None),
            only=only_parts, poses=ir_poses, pose_label=pose_label,
            resolution=res, cameras=(cam,)))
    except RenderUnavailable as e:
        print(f"rendering unavailable: {e}")
        return 1

    src = Path(result.render_paths[cam.id])
    # The reviewer cites these names in its reasons, so they have to say what
    # the image is: an isolated render and a posed one are different evidence
    # about the same entity from the same direction.
    tags = ""
    if only_parts:
        tags += "_iso" if len(only_parts) > 2 else "_" + "_".join(
            str(n) for n in sorted(only_parts))[:40]
    if ir_poses:
        tags += "_posed"
    dst = out_dir / (f"extra_{len(taken) + 1}_{a.on}"
                     f"_az{a.azimuth:g}_el{a.elevation:g}{tags}.png")
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    print(str(dst))
    return 0


if __name__ == "__main__":
    # Run as a module so the package-relative imports above resolve:
    #   python -m dycraft.agent.layers._review_render ...
    raise SystemExit(main(sys.argv))
