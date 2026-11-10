#!/usr/bin/env bash
# Set up the cuRobo env that h.ik runs in (needs an NVIDIA GPU with CUDA), and record it in a pointer file: in this
# checkout, or in $DYCRAFT_HOME when set. The same as `python -m dycraft.install ik`, runnable from a clone with nothing
# installed: it loads dycraft from this directory, whatever the directory is called. Options:
#
#   ./setup_ik.sh --help
#
# PYTHON chooses the interpreter that runs the setup (any Python >= 3.10; default python3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" - "$HERE" ik "$@" <<'PY'
import importlib.util, runpy, sys
here = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("dycraft", here + "/__init__.py", submodule_search_locations=[here])
module = importlib.util.module_from_spec(spec)
sys.modules["dycraft"] = module
spec.loader.exec_module(module)
sys.argv[0] = "dycraft.install"
runpy.run_module("dycraft.install", run_name="__main__", alter_sys=True)
PY
