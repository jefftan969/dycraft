"""`python -m dycraft.export glb RUN [-o OUT.glb] [--bake-materials] [--web] [--jobs N]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .glb import ExportError, export


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dycraft.export",
        description="Export a run, a published film or a .blend to a GLB (glTF 2.0) that Blender, three.js and "
                    "other glTF viewers open.")
    sub = parser.add_subparsers(dest="format", required=True)
    glb = sub.add_parser(
        "glb", help="export to GLB",
        description="A film keeps its motion: rigid motion as node animation, characters as skins, and other "
                    "deformation (cloth, caches) as morph targets, with its cameras. An object or scene run is "
                    "exported at rest. Lights are not exported.")
    glb.add_argument("run", type=Path, help="a run folder, a published film folder (films/<id>/) or a .blend")
    glb.add_argument("-o", "--out", type=Path, default=None,
                     help="the GLB to write (default: animation.glb, object.glb or scene.glb beside the input)")
    glb.add_argument("--bake-materials", action="store_true",
                     help="bake procedural materials into image textures with Cycles, instead of one flat colour "
                          "each (a GPU is strongly advised: on a CPU a heavy film takes hours)")
    glb.add_argument("--web", action="store_true",
                     help="also write <out>.web.glb, compressed for browsers (needs Node.js 20 or later)")
    glb.add_argument("--jobs", type=int, default=None, metavar="N",
                     help="Blender processes for sampling and baking (default: as many as free memory allows, "
                          "at most 4; each holds the whole scene)")
    args = parser.parse_args(argv)
    try:
        export(args.run, args.out, bake_materials=args.bake_materials, web=args.web, jobs=args.jobs,
               log=lambda line: print(line, file=sys.stderr, flush=True))
    except ExportError as error:
        print(f"[export] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
