"""Run the deterministic animation workflow without model calls; retain its artifacts."""
import argparse
import json
from pathlib import Path
import time

from dycraft.agent.layers.film import FilmSession
from dycraft.animation.blender.bundle import validate_manifest
from dycraft.tests._animation_fixtures import (
    ANIMATE, STAGE, call, plan, source, state_reviews)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    if root.exists():
        parser.error("--out must be a new directory; existing work is never overwritten")
    root.mkdir(parents=True)
    session = FilmSession(root, user_prompt=plan()["intent"])
    service = session.blender_animation
    def run(operation, **kwargs):
        started = time.monotonic()
        result = call(service, operation, timeout=300, **kwargs)
        print(json.dumps({"operation": operation, "ok": result.get("ok"),
                          "seconds": round(time.monotonic()-started, 2)}), flush=True)
        if not result.get("ok"):
            raise RuntimeError(json.dumps(result, indent=2))
        return result
    try:
        (service.design / "scene.py").write_text(source())
        value = plan()
        if args.width * args.height >= 1920 * 1080:
            value.pop("delivery")
        else:
            value["delivery"] = {"min_pixels": args.width * args.height,
                                 "reason": "explicit smoke-test delivery target"}
        run("plan", plan=value)
        run("build")
        run("blender", code=STAGE, stage="storyboard")
        run("storyboard", render_settings={"engine": "WORKBENCH", "width": 320, "height": 240, "threads": 2})
        run("accept_storyboard", reviews=state_reviews("Deterministic fixture: the declared oblique camera stages the supported block. This scripted acknowledgement does not claim a human/VLM quality review."))
        run("blender", code=ANIMATE)
        run("final_render", render_settings={"engine": "WORKBENCH", "width": args.width,
                                            "height": args.height, "threads": 2})
        result = run("finish")
        validate_manifest(root, json.loads((root / "ir.json").read_text()))
        print(json.dumps(result, indent=2))
    finally:
        session.drain_on_shutdown(timeout_s=10)


if __name__ == "__main__":
    main()
