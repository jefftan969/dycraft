"""`python -m dycraft.viewer` — serve the run viewer.

    python -m dycraft.viewer                       # API + built SPA on :8770
    python -m dycraft.viewer --dev                 # API only; run vite separately
    python -m dycraft.viewer --runs-dir /other/runs

Runs from the repo root: `dycraft` is pip-installed editable
(`package-dir dycraft = "."`), so `import dycraft` resolves from any cwd. The
default runs dir is resolved relative to this package rather than the working
directory, so it stays correct wherever you launch from.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dycraft.viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--runs-dir", default=None,
                        help="defaults to $DYCRAFT_RUNS_DIR, else runs/ under $DYCRAFT_HOME if set, "
                             "else under the clone, else under the current directory")
    parser.add_argument("--dev", action="store_true",
                        help="skip the SPA mount; serve the API only and let "
                             "`npm run dev` serve the frontend on :5173")
    parser.add_argument("--reload", action="store_true",
                        help="uvicorn autoreload (development)")
    args = parser.parse_args(argv)

    import uvicorn

    from .app import create_app

    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else None
    # --dev points dist_dir at a path that cannot exist, which makes the app
    # skip the catch-all SPA route entirely; without that, the catch-all would
    # shadow nothing but would still answer requests vite should handle.
    dist_dir = Path("/nonexistent-dycraft-viewer-dist") if args.dev else None
    app = create_app(runs_dir=runs_dir, dist_dir=dist_dir)

    print(f"[viewer] runs_dir={app.state.runs_dir}")
    print(f"[viewer] http://{args.host}:{args.port}"
          + ("  (API only; frontend on :5173)" if args.dev else ""))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    # The tools and GL libraries `install tools` put in place, for the Blender and render children it starts.
    from .._home import tools_on_path
    tools_on_path()
    raise SystemExit(main())
