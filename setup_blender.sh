#!/usr/bin/env bash
# Set up the Blender env (Python 3.11, bpy 4.5) and the official Blender 4.5.13 binary, which fluid needs, and record
# them in pointer files: in this checkout, or in $DYCRAFT_HOME when set. The same as `python -m dycraft.install
# blender`, runnable from a clone with nothing installed: it loads dycraft from this directory, whatever the directory
# is called. Options:
#
#   ./setup_blender.sh --help
#
# PYTHON chooses the interpreter that runs the setup (any Python >= 3.10; default python3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" - "$HERE" blender "$@" <<'PY'
import importlib.util, runpy, sys
here = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("dycraft", here + "/__init__.py", submodule_search_locations=[here])
module = importlib.util.module_from_spec(spec)
sys.modules["dycraft"] = module
spec.loader.exec_module(module)
sys.argv[0] = "dycraft.install"
runpy.run_module("dycraft.install", run_name="__main__", alter_sys=True)
PY
