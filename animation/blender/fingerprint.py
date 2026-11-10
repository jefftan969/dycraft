"""The fingerprint of the code a cached verification depends on."""
from __future__ import annotations

from pathlib import Path

from .bundle import file_hash
from .runtime import REPO


def implementation_fingerprint():
    """Hashes of the code whose behavior a cached verification depends on."""
    files = [*Path(__file__).parent.glob('*.py'),
             *(REPO / 'verify' / (name+'.py') for name in
               ('_mesh_erosion', '_inherited_allowances', '_allowance_rules'))]
    # `p.resolve()` because `REPO` is resolved (`runtime.py`) while this
    # module's own `__file__` need not be: under a `pkgroot` symlink import
    # the glob yields paths outside the resolved repo, `relative_to` raises,
    # and every check that keys its cache on this fingerprint fails before it
    # starts.
    return {str(p.resolve().relative_to(REPO)): file_hash(p) for p in files}
