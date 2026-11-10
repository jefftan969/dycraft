"""Read-only native view command for a fresh-context reviewer."""
import argparse
import json
from pathlib import Path

from .blender.runtime import run_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--camera")
    parser.add_argument("--eye", type=float, nargs=3)
    parser.add_argument("--look-at", type=float, nargs=3)
    parser.add_argument("--fov-deg", type=float, default=60.)
    parser.add_argument("--render-settings", type=json.loads,
                        help="JSON object: engine/width/height/samples, as final_render takes. "
                             "Omitted leaves the project's own saved settings.")
    args = parser.parse_args(argv)
    if not 1 <= len(args.frames) <= 32:
        parser.error("select 1 to 32 frames per inspection")
    if bool(args.eye) != bool(args.look_at):
        parser.error("--eye and --look-at must be supplied together")
    import tempfile
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="view_", dir=args.out_dir))
    result = run_worker(args.workspace.resolve(), output, {
        "op": "render", "frames": args.frames, "camera_name": args.camera,
        "eye": args.eye, "look_at": args.look_at, "fov_deg": args.fov_deg,
        **({"render_settings": args.render_settings} if args.render_settings else {})},
        workspace_readonly=True)
    print(json.dumps({"paths": [str(output / p) for p in result["paths"]],
                      "render_settings": result.get("render_settings")}))


if __name__ == "__main__":
    main()
