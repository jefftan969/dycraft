"""Code that compares its own `__file__` to `REPO` must survive the pkgroot symlink.

A child process can import dycraft through a directory holding a single
`dycraft` symlink to the checkout: the suite's own subprocess tests do
(`conftest.pkgroot`), as does any launcher that puts such a directory on
PYTHONPATH.
Under that import every module's `__file__` begins with the *symlink* path,
while `animation.blender.runtime.REPO` is `Path(__file__).resolve()`-derived and
so is the real path. The two spellings never compare equal, and
`Path.relative_to` raises `ValueError` rather than returning something wrong —
which is why this is a hard failure and not a subtle one.

**Why it matters.** `animation/blender/fingerprint.py::implementation_fingerprint`
globs its own directory and calls `p.relative_to(REPO)` on the result; the
animation layer keys every check's cache on it, so a raise there stops every
check before it starts.

**Why a normal test cannot catch it.** `import dycraft` from an ordinary
environment resolves the real path, where every one of these comparisons
succeeds. The condition only exists under the symlinked import, so the only
durable guard is a test that imports that way on purpose — this one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


#: Runs in a child whose PYTHONPATH is the pkgroot, so `import dycraft` comes
#: through the symlink. Returns one JSON object per check so a failure names the
#: call that broke rather than just the exit status.
_CHILD = r"""
import json, sys
from pathlib import Path

out = {}

import dycraft
out["imported_via"] = dycraft.__file__

from dycraft.animation.blender.runtime import REPO
out["repo"] = str(REPO)

import dycraft.animation.blender.fingerprint as fingerprint
out["fingerprint_file"] = fingerprint.__file__
# The exact expression `implementation_fingerprint` builds its map from.
files = [*Path(fingerprint.__file__).parent.glob("*.py"),
         *(REPO / "verify" / (n + ".py") for n in
           ("_mesh_erosion", "_inherited_allowances", "_allowance_rules"))]
try:
    out["implementation_keys"] = sorted(
        str(p.resolve().relative_to(REPO)) for p in files)
except ValueError as exc:
    out["error"] = f"implementation map: {exc}"

print("RESULT " + json.dumps(out))
"""


def _run_through_symlink(pkgroot: Path) -> dict:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True, text=True, timeout=300,
        # cwd outside the repo, so nothing resolves by accident.
        cwd=str(pkgroot), env={"PYTHONPATH": str(pkgroot),
                               "PATH": "/usr/bin:/bin",
                               "HOME": str(pkgroot)},
    )
    line = next((l for l in done.stdout.splitlines()
                 if l.startswith("RESULT ")), None)
    assert line, (f"child exited {done.returncode}\n"
                  f"stdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-2000:]}")
    return json.loads(line[len("RESULT "):])


def test_import_really_goes_through_the_symlink(pkgroot):
    """The premise. Without this the rest of the file proves nothing."""
    result = _run_through_symlink(pkgroot)
    assert result["imported_via"].startswith(str(pkgroot)), (
        "the child did not import through the pkgroot symlink, so this test is "
        f"not exercising the condition it exists for: {result['imported_via']}")
    assert result["repo"] == str(REPO), (
        "REPO should resolve to the real checkout even under the symlinked "
        f"import: {result['repo']}")
    assert not result["fingerprint_file"].startswith(str(REPO)), (
        "fingerprint.__file__ should be the symlinked spelling here; if it is "
        "already the real path, the import route changed and this guard is "
        f"no longer measuring anything: {result['fingerprint_file']}")


def test_implementation_fingerprint_builds_under_the_symlink(pkgroot):
    """Every animation check keys its cache on this map, so it must build."""
    result = _run_through_symlink(pkgroot)
    assert "error" not in result, result.get("error")
    keys = result["implementation_keys"]
    assert keys, "no implementation files hashed"
    assert all(not k.startswith(("/", "..")) for k in keys), (
        f"implementation keys must be repo-relative: {keys[:4]}")
    assert "animation/blender/fingerprint.py" in keys, (
        f"the module's own source is missing from its implementation map: {keys[:6]}")


#: Sites that compare a path against `REPO` without `.resolve()` on the same
#: expression, and why each is safe. A new entry here is a decision, which is
#: the point: the two tests above prove the *pattern* works, and this one stops
#: the pattern being dropped somewhere new.
_RESOLVED_ELSEWHERE: dict[str, str] = {}

#: The package's own code: what a symlinked import can load. Development tooling outside it is not imported that
#: way, so it is not scanned.
_PACKAGE = ("agent/", "animation/", "benchmark/", "cli/", "render/", "sdk/", "verify/", "viewer/")


def test_no_new_unresolved_repo_relative_comparisons():
    """Every `relative_to(REPO)` resolves first, or is allowlisted with a reason.

    The failure this guards is invisible on the normal import route and fatal on
    the harness's, so a reviewer has no local signal that a new call is wrong.
    """
    import re

    pattern = re.compile(r"(\S*?)\.(?:is_)?relative_to\(\s*(?:bare_arms\.)?REPO\s*\)")
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if not (rel.startswith(_PACKAGE) or "/" not in rel) or rel.startswith("viewer/web/"):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                if match.group(1).endswith("resolve()"):
                    continue
                if rel in _RESOLVED_ELSEWHERE:
                    continue
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, (
        "these compare an unresolved path to REPO, which raises ValueError when "
        "dycraft is imported through the pkgroot symlink every harness uses:\n  "
        + "\n  ".join(offenders)
        + "\n\nCall `.resolve()` on the left-hand path, or add the file to "
          "`_RESOLVED_ELSEWHERE` with the reason it is already safe.")
