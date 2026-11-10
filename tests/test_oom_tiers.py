"""The OOM tiers in `_oom` are applied everywhere this package starts a process.

Two halves. The unit tests pin the helpers' contracts: `tiered` is idempotent,
`raise_self` never lowers, `arm_entry` puts an arm in its own
`OOMPolicy=continue` scope at SESSION without changing its pid. The static test
is the enforcement: it walks every module of the runtime package and fails on
any process spawn whose function neither prices it through `_oom` nor appears
in `UNPRICED` with the reason it is safe unpriced. A new spawn therefore has to
take a side in review instead of silently inheriting whatever tier its
launcher had.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft import _oom

REPO = Path(__file__).resolve().parents[1]


# --- the order ---------------------------------------------------------------

def test_tiers_are_strictly_ordered_above_the_desktop_session():
    # Ubuntu's user@.service puts the desktop tree at 100-200; every tier of
    # ours must sit above it, and each strictly above the one it outranks.
    assert _oom.WORKER > _oom.RESIDENT > _oom.SUBAGENT > _oom.SESSION > 200


# --- tiered ------------------------------------------------------------------

def test_tiered_prefixes_a_child_that_would_be_less_exposed(monkeypatch):
    monkeypatch.setattr(_oom, "current", lambda: 0)
    argv = _oom.tiered([sys.executable, "-V"], 900)
    assert argv[-2:] == [sys.executable, "-V"]
    assert len(argv) > 2


def test_tiered_is_idempotent_for_choom(monkeypatch):
    monkeypatch.setattr(_oom, "current", lambda: 0)
    once = _oom.tiered([sys.executable], 900)
    assert Path(once[0]).name == "choom"
    assert _oom.tiered(once, 900) == once
    assert _oom.tiered(once, 700) == once


def test_tiered_is_idempotent_for_the_shell_fallback(monkeypatch):
    monkeypatch.setattr(_oom, "current", lambda: 0)
    real = shutil.which
    monkeypatch.setattr(_oom.shutil, "which",
                        lambda name: None if name == "choom" else real(name))
    once = _oom.tiered([sys.executable], 900)
    assert once[0] == "/bin/sh" and _oom._PREFIX_MARK in once
    assert _oom.tiered(once, 900) == once


def test_tiered_leaves_an_unresolvable_target_bare(monkeypatch):
    # So the spawn still raises FileNotFoundError, which callers classify as
    # `worker_unavailable`; behind choom it would become a `worker_crash`.
    monkeypatch.setattr(_oom, "current", lambda: 0)
    argv = ["/no/such/interpreter", "-c", "pass"]
    assert _oom.tiered(argv, 900) == argv
    with pytest.raises(FileNotFoundError):
        subprocess.run(_oom.tiered(argv, 900))


def test_untiered_recovers_the_command(monkeypatch):
    monkeypatch.setattr(_oom, "current", lambda: 0)
    argv = [sys.executable, "-c", "pass", "--", "x"]
    assert _oom.untiered(_oom.tiered(argv, 700)) == argv
    assert _oom.untiered(argv) == argv


def test_tiered_leaves_an_already_exposed_child_alone(monkeypatch):
    # A child can only be raised; asking choom for less would make it exit
    # without exec'ing, which would lose the job rather than misprice it.
    monkeypatch.setattr(_oom, "current", lambda: 950)
    assert _oom.tiered(["x"], 900) == ["x"]


def test_a_tiered_child_really_runs_at_its_tier():
    if _oom.current() is None or _oom.current() >= _oom.WORKER:
        pytest.skip("this process is already at or above WORKER")
    out = subprocess.run(
        _oom.tiered(["cat", "/proc/self/oom_score_adj"], _oom.WORKER),
        capture_output=True, text=True, check=True).stdout.strip()
    assert int(out) == _oom.WORKER


# --- raise_self --------------------------------------------------------------

def test_raise_self_raises_and_never_lowers(pkgroot):
    # In a child: raising is irreversible, so it must not touch pytest itself.
    code = ("from dycraft import _oom\n"
            "print(_oom.raise_self(600), _oom.raise_self(500), "
            "_oom.raise_self(800))\n")
    if _oom.current() is None or _oom.current() > 600:
        pytest.skip("this process is already above 600")
    out = subprocess.run([sys.executable, "-c", code], cwd=pkgroot,
                         env={**os.environ, "PYTHONPATH": str(pkgroot)},
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["600", "600", "800"]


# --- scopes ------------------------------------------------------------------

def test_in_scope_needs_the_marker(monkeypatch):
    monkeypatch.delenv(_oom.SCOPE_ENV, raising=False)
    assert not _oom.in_scope()
    monkeypatch.setenv(_oom.SCOPE_ENV, "0")
    assert not _oom.in_scope()
    # A marker naming a scope we are not in (inherited, then moved) is not
    # enough on its own; the cgroup has to agree.
    monkeypatch.setenv(_oom.SCOPE_ENV, "dycraft-arm-no-such-unit")
    assert not _oom.in_scope()


def test_enter_scope_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv(_oom.SCOPE_ENV, "0")
    called = []
    monkeypatch.setattr(_oom.os, "execv", lambda *a: called.append(a))
    _oom.enter_scope(["true"])
    assert not called


def _user_scopes_work() -> bool:
    exe = shutil.which("systemd-run")
    if not exe:
        return False
    try:
        return subprocess.run(
            [exe, "--user", "--scope", "--quiet", "--collect",
             "-p", "OOMPolicy=continue", "--", "true"],
            capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _user_scopes_work(),
                    reason="no working systemd user manager here")
def test_arm_entry_scopes_the_process_keeps_its_pid_and_prices_it(pkgroot):
    # `-c` re-runs from the top after the re-exec, so the `pid` line prints
    # once per pass; both must name the same process.
    code = (
        "import os, subprocess\n"
        "print('pid', os.getpid(), flush=True)\n"
        "from dycraft import _oom\n"
        "_oom.arm_entry(label='pytest')\n"
        "unit = os.environ[_oom.SCOPE_ENV]\n"
        "print('adj', open('/proc/self/oom_score_adj').read().strip())\n"
        "print('unit', unit)\n"
        "print('cgroup', open('/proc/self/cgroup').read().strip())\n"
        "print(subprocess.run(['systemctl', '--user', 'show', unit + '.scope',\n"
        "      '-p', 'OOMPolicy'], capture_output=True, text=True).stdout.strip())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != _oom.SCOPE_ENV}
    env["PYTHONPATH"] = str(pkgroot)
    out = subprocess.run([sys.executable, "-c", code], cwd=pkgroot, env=env,
                         capture_output=True, text=True, timeout=60, check=True)
    lines = dict(l.split(" ", 1) for l in out.stdout.splitlines() if " " in l)
    pids = [l.split()[1] for l in out.stdout.splitlines() if l.startswith("pid ")]
    assert len(pids) == 2 and pids[0] == pids[1], out.stdout
    assert int(lines["adj"]) == _oom.SESSION
    assert lines["unit"] == f"dycraft-pytest-{pids[0]}"
    assert f"/{lines['unit']}.scope" in lines["cgroup"]
    assert "OOMPolicy=continue" in out.stdout


# --- enforcement -------------------------------------------------------------

#: The runtime package: every layer an arm executes. The test suite and any
#: offline tooling are excluded.
SCANNED = ["agent", "animation", "cli", "render", "sdk", "verify", "viewer"]

#: (path, function) -> why it is safe to spawn here without `_oom`. Each entry
#: must still name a real spawn, so this list cannot rot silently.
UNPRICED = {
    ("_home.py", "reexec_for_loader"):
        "an entry point restarting itself, the same command, before it does anything: the tier is set after it "
        "(`_oom.arm_entry` in the CLI's `__main__`), and the restart keeps whatever tier the process already had",
    ("animation/blender/_landlock.py", "main"):
        "the Landlock launcher's `exec` of the worker it confines: the launcher runs under the worker's own tier "
        "(`runtime.command`'s prefix), which the `exec` keeps",
    ("animation/ik/ik.py", "gpu_visible"):
        "`nvidia-smi -L`, a listing that ends in milliseconds and allocates "
        "nothing, asked once per launch and per `h.ik` call",
    ("agent/assets/asset_library.py", "render_thumbnail"):
        "offline library maintenance (`--thumbnails`), run by hand on a login or "
        "Slurm node, never inside a run",
    ("agent/assets/asset_library.py", "_hstack"):
        "ffmpeg joining two thumbnail views, same offline maintenance command",
    ("environment_test.py", "main"):
        "`--python` re-execs the checker as itself under another interpreter: the "
        "same process, at whatever tier the person who launched it holds",
    ("_oom.py", "enter_scope"):
        "the scope probe and the re-exec itself: the SESSION process, becoming "
        "itself inside its scope",
    ("cli/backends/_process.py", "run_agent_cli"):
        "runs the agent CLI as the launcher's child, which waits on it: the "
        "arm itself, inheriting SESSION from `arm_entry`",
    ("cli/backends/_process.py", "_spawn_with_pty_term_shim"):
        "forks the agent CLI under a PTY: the arm itself, inheriting SESSION",
    ("cli/cli.py", "_spawn_detached"):
        "starts the arm's own harness daemon (detached, so the launcher never "
        "leaves it a zombie); losing it is losing the arm, so SESSION by "
        "inheritance is its correct tier",
    ("cli/cli.py", "start_self_kill_watcher"):
        "a few-KB watcher that belongs to the arm and dies with it",
    ("agent/server.py", "git"):
        "read-only git queries (refs, merge-base, diff) once per daemon start, "
        "under a 60 s timeout, to record the code the daemon runs",
    ("animation/blender/runtime.py", "_userns_prefix"):
        "a one-time `unshare -Urn true` capability probe, cached",
    ("animation/blender/collision.py", "__init__"):
        "runs inside the bubblewrap Blender sandbox as a top-level module, "
        "where `_oom` is not importable; its coal child inherits the sandboxed "
        "worker's WORKER tier",
}

_SPAWNS = {
    "subprocess": {"Popen", "run", "call", "check_call", "check_output"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "pty": {"fork", "spawn"},
    "os": {"fork", "forkpty", "posix_spawn", "posix_spawnp", "system",
           "execv", "execve", "execvp", "execvpe", "popen"},
}

#: Text in the spawning function that shows it priced the child: a tier
#: helper, or `animation/blender/runtime.command`, which applies the tier.
_PRICED = ("_oom.", "_oom_prefix(", "command(")


def _modules():
    for path in sorted(REPO.glob("*.py")):
        yield path
    for d in SCANNED:
        for path in sorted((REPO / d).rglob("*.py")):
            if "__pycache__" in path.parts or path.name.startswith("test_"):
                continue
            yield path


def _spawn_sites():
    """(relative path, innermost function, line, call, priced?) per spawn."""
    for path in _modules():
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:        # templates with @@PLACEHOLDERS@@
            continue
        rel = path.relative_to(REPO).as_posix()
        fns = [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in fns:
            body = ast.get_source_segment(src, fn) or ""
            inner = [n for n in fns if n is not fn
                     and any(m is n for m in ast.walk(fn))]
            nested = {id(x) for n in inner for x in ast.walk(n)}
            for call in ast.walk(fn):
                if (id(call) in nested or not isinstance(call, ast.Call)
                        or not isinstance(call.func, ast.Attribute)
                        or not isinstance(call.func.value, ast.Name)):
                    continue
                mod, attr = call.func.value.id, call.func.attr
                if attr in _SPAWNS.get(mod, ()):
                    yield (rel, fn.name, call.lineno, f"{mod}.{attr}",
                           any(t in body for t in _PRICED))


def test_every_process_spawn_is_priced_or_explained():
    offenders = sorted({
        f"{rel}:{line}  {fn}()  {call}"
        for rel, fn, line, call, priced in _spawn_sites()
        if not priced and (rel, fn) not in UNPRICED})
    assert not offenders, (
        "process spawns with no OOM tier -- wrap the argv in "
        "`_oom.tiered(argv, _oom.WORKER | RESIDENT | SUBAGENT)` (see `_oom` for "
        "which), or add the function to UNPRICED with the reason it is safe:\n  "
        + "\n  ".join(offenders))


def test_the_unpriced_list_names_only_real_spawns():
    live = {(rel, fn) for rel, fn, *_ in _spawn_sites()}
    stale = sorted(k for k in UNPRICED if k not in live)
    assert not stale, f"UNPRICED entries that no longer spawn anything: {stale}"


# --- the kill order is the policy ---------------------------------------------

def _row(adj, rss_mib, comm="python", cgroup="dycraft-arm-1.scope", pid=1):
    total = _oom.mem_total_kb()
    return {"pid": pid, "adj": adj, "rss_kb": rss_mib * 1024, "comm": comm,
            "cgroup": cgroup, "cmd": comm,
            "badness_kb": _oom.badness(rss_mib * 1024, adj, total)}


def test_the_tier_gaps_are_wide_enough_to_be_worth_having():
    # Each point is MemTotal/1000, so a gap only outranks a size difference
    # below `gap * that`. 200 points is ~6 GiB on a 30 GB box; a 50-point gap
    # (~1.5 GiB) is cleared by a resident Blender, leaving workers above it
    # untouched.
    point_mib = _oom.mem_total_kb() / 1000 / 1024
    adjs = [adj for _, adj in _oom.TIERS]
    assert adjs == sorted(adjs, reverse=True)
    gaps = [a - b for a, b in zip(adjs, adjs[1:])]
    assert min(gaps) * point_mib > 4096, f"gaps {gaps} are under 4 GiB apart"
    # Above the desktop session's 100-200 by the same margin, and at most the
    # ceiling the kernel accepts.
    assert adjs[-1] - 200 >= min(gaps)
    assert adjs[0] <= 1000


def test_kill_order_ranks_by_the_kernels_own_formula():
    rows = [_row(400, 8000, pid=1), _row(1000, 100, pid=2), _row(800, 200, pid=3)]
    assert [r["pid"] for r in _oom.kill_order(rows)] == [2, 3, 1]


def test_check_order_is_quiet_when_tiers_rank_in_order():
    rows = [_row(1000, 500, pid=1), _row(800, 400, pid=2), _row(600, 300, pid=3),
            _row(400, 200, pid=4),
            _row(0, 20, comm="tmux: server", cgroup="tmux-spawn-x.scope", pid=5)]
    assert _oom.check_order(rows) == []


def test_check_order_names_a_tier_inversion():
    # A resident big enough to clear the gap outranks a worker. The gap is
    # 200 points of MemTotal, ~6 GiB on a 30 GB
    # box but ~12 GiB on a 64 GB one, so size the resident from this box.
    gap_mib = (_oom.badness(0, 1000) - _oom.badness(0, 800)) / 1024
    rows = [_row(800, 500 + int(gap_mib) + 1024, pid=2), _row(1000, 500, pid=1)]
    complaints = _oom.check_order(rows)
    assert any("outranks" in c and "lower tier" in c for c in complaints), complaints


def test_a_worker_outranks_any_shell_whatever_the_shell_holds():
    # WORKER sits at the 1000 ceiling, which is worth MemTotal of badness, so
    # no shell can outscore it on this box even holding every byte of RAM.
    whole_box = _oom.mem_total_kb() // 1024
    rows = [_row(1000, 100, pid=1),
            _row(0, whole_box, comm="tmux: server",
                 cgroup="tmux-spawn-x.scope", pid=9)]
    assert _oom.kill_order(rows)[0]["pid"] == 1
    assert _oom.check_order(rows) == []


def test_check_order_catches_a_shell_that_would_die_before_an_arm():
    # The exposed pair is the *lowest* of our tiers against a large shell: at
    # SESSION a 200 MiB arm is worth ~12.6 GiB of badness here, so a shell
    # process bigger than that dies first and the policy is broken.
    session_badness_mib = _oom.badness(200 * 1024, _oom.SESSION) / 1024
    rows = [_row(_oom.SESSION, 200, pid=1),
            _row(0, int(session_badness_mib) + 2048, comm="tmux: server",
                 cgroup="tmux-spawn-x.scope", pid=9)]
    assert any("would die first" in c for c in _oom.check_order(rows))


def test_check_order_catches_an_untiered_process_in_one_of_our_cgroups():
    # A batch supervisor at adj 0 inside its own scope is the case: a kill
    # wave would take it with the shells.
    rows = [_row(1000, 100, pid=1),
            _row(0, 13, comm="python", cgroup="dycraft-lane-x.scope", pid=7)]
    assert any("untiered process" in c for c in _oom.check_order(rows))


def test_shell_scope_prefixes_cover_what_holds_a_terminal():
    assert "tmux-spawn-" in _oom.SHELL_SCOPE_PREFIXES
    assert "session-" in _oom.SHELL_SCOPE_PREFIXES


# --- the memory ceiling, which is opt-in ------------------------------------

def test_a_run_is_unbounded_unless_a_launcher_says_otherwise(monkeypatch):
    # A single arm should be free to use the whole machine. A ceiling is a
    # statement about contention, and only the launcher can see that.
    monkeypatch.delenv("DYCRAFT_ARM_MEMORY_MAX", raising=False)
    assert _oom.arm_memory_max() == ""
    base = _oom._scope_base()
    assert not [a for a in base if a.startswith("MemoryMax")]
    assert "OOMPolicy=continue" in base      # isolation is not optional


def test_a_launcher_may_cap_a_contended_run(monkeypatch):
    monkeypatch.setenv("DYCRAFT_ARM_MEMORY_MAX", "4G")
    assert _oom.arm_memory_max() == "4G"
    base = _oom._scope_base()
    assert "MemoryMax=4G" in base and "MemorySwapMax=0" in base


def test_every_spelling_of_no_cap_reads_as_no_cap(monkeypatch):
    # So a launcher can pass the variable through unconditionally.
    for off in ("", "0", "none", "off", "max", "infinity", "  "):
        monkeypatch.setenv("DYCRAFT_ARM_MEMORY_MAX", off)
        assert _oom.arm_memory_max() == "", repr(off)


def test_the_user_manager_is_protected_too():
    # `init.scope` is `systemd --user`: one oomd kill there takes every
    # terminal on the box and the arms inside them.
    assert "init.scope" in _oom.CRITICAL_SCOPE_NAMES


def test_protecting_shells_is_idempotent_and_never_raises(tmp_path, monkeypatch):
    # Called on every arm launch, so it must be cheap and must never be able to
    # fail a run. A second call marks nothing new.
    first = _oom.protect_shell_cgroups(verbose=False)
    second = _oom.protect_shell_cgroups(verbose=False)
    assert second == [] or set(second).isdisjoint(first)


def test_a_shell_cannot_outrank_the_lowest_tier_we_run(monkeypatch):
    # The guarantee, as arithmetic: for a terminal to be taken before an arm,
    # it would have to hold the arm's whole badness in real memory.
    total = _oom.mem_total_kb()
    arm = _oom.badness(200 * 1024, _oom.SESSION, total)      # a 200 MiB arm
    shell = _oom.badness(64 * 1024, 0, total)                # a fat 64 MiB shell
    assert shell < arm
    needed_gib = (arm - 64 * 1024) / 1048576
    assert needed_gib > 8, f"a shell would need {needed_gib:.1f} GiB to outrank"
