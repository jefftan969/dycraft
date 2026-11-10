"""Which of this harness's processes the OOM killer takes, and in what order.

The kernel ranks every process by

    badness = rss + swapents + pgtables  +  oom_score_adj * MemTotal / 1000

and kills the highest, repeating until the allocation that woke it succeeds.
`oom_score_adj` is therefore a *handicap in pages*, not a priority: on a 30 GB
box one point is ~31 MB, so adj 200 outweighs 6.1 GB of real memory. It is
inherited across fork, survives exec, and can only be raised without
CAP_SYS_RESOURCE.

Left alone, every sandboxed Blender process sits at adj 0, below the desktop
session at Ubuntu's `user@.service` default of 100-200 (`systemd --user`,
`(sd-pam)`, dbus, pipewire, gvfsd, at-spi). The kernel then kills megabytes of
session before gigabytes of Blender scratch, and with `user@<uid>.service` go X,
every terminal, and every agent session in them. The tiers put the harness's
scratch work first and the session last.

THE ORDER
---------
Highest first, and **the spacing is the guarantee**: a 200-point gap only
outranks a size difference below ~6.2 GB on a 30 GB box, so a resident that
outgrows a worker by more dies first. The gaps are as wide as the range allows.
No spacing holds against an unbounded process (a 21 GB runaway at SUBAGENT
outranks every worker); that is a bug to fix or a cgroup to cap.
`check_order()` reports the live ranking and names any inversion.

   1000  WORKER    disposable: a one-shot Blender worker, a CAD pool child or
                   one-shot CAD interpreter, a render, an IK/sim/ffmpeg run.
                   Dies into an exception the caller handles (a failed job the
                   agent resubmits, a pool child the pool respawns).
    800  RESIDENT  persistent but reconstructible: the long-lived Blender bound
                   to a project, the collision worker, an interactive render
                   session. Dies into a respawn or a cold start.
    600  SUBAGENT  an LLM session subordinate to the arm: a background object
                   agent the scene layer dispatched, a rewriter, or a one-shot
                   review. Dies as a failed dispatch the parent can resubmit.
    400  SESSION   the arm itself: launcher shell, harness daemon, codex, and
                   anything the agent shells out to. Raised by `dycraft.cli`'s
                   entry point (`raise_self`) and the layer's `_env.sh`, and
                   inherited by everything below. Recoverable only by a human,
                   so it is last of ours, but still above the desktop session.

Everything else keeps its own number: 100-200 the desktop session, 0 tmux and
login shells, -900 systemd-oomd and snapd, -1000 sshd and udevd.

SESSION stays above the desktop because a tmux pane can be a
`tmux-spawn-*.scope` inside `user@<uid>.service`: when the kernel takes the user
manager, an arm below it dies anyway, mid-step, with the daemon that would have
written `meta.json`. Dying at 400 is a death we choose and can resume from.

ONE SCOPE PER ARM
-----------------
Tiers decide *which* process dies, not what the kill costs. Under systemd 255's
`DefaultOOMPolicy=stop`, a unit that loses any process to the OOM killer is
stopped, so one worker killed in a terminal's scope would take every arm in that
terminal and the terminal itself (`setsid` does not help: scope membership is a
cgroup). `enter_scope` re-execs every `dycraft.cli` launch into a transient user
scope of its own with `OOMPolicy=continue`, so a kill inside the arm costs only
that process and a kill outside it cannot stop the arm. The pid is preserved, so
`wait`, `timeout --foreground`, pidfiles, and the controlling terminal are
unaffected. A run is unbounded by default; `DYCRAFT_ARM_MEMORY_MAX` adds a
`MemoryMax` to the scope, so an overrun becomes a tier-ordered memcg OOM inside
that arm. That is for whoever launches several arms at once, not a default.

SYSTEMD-OOMD
------------
A second killer ignores `oom_score_adj`. Ubuntu's
`user@.service.d/10-oomd-user-service-defaults.conf` sets
`ManagedOOMMemoryPressure=kill` at 50% on the user session: when memory pressure
(PSI stall time, not usage) stays above 50% for 20 s, systemd-oomd SIGKILLs whole
cgroups chosen by pressure and reclaim, tiers and scopes notwithstanding.

A cgroup can be taken off oomd's list without root: `ManagedOOMPreference=` is
honored when the candidate's owner matches the monitored ancestor's
(systemd.resource-control(5)), which holds for every scope under
`user@<uid>.service`. The mechanism is a `user.oomd_omit` xattr that a process
may write itself. `protect_shell_cgroups()` sets it on the terminal and login
scopes, so oomd reaches for an arm rather than the operator's window; arm scopes
stay candidates, since an arm is the cheapest thing on the box to lose.

So tiers decide who dies when the kernel runs out, scopes decide what that kill
costs, and the omit xattr decides what oomd may take. None survives sustained
over-commitment: the defense against that is keeping the arms' total below
physical memory (`DYCRAFT_ARM_MEMORY_MAX` per arm) or, with root, changing oomd's
limit on `user@.service`. This is triage, not a budget.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

#: Disposable: the caller already handles its death as an ordinary failure.
#: The ceiling, so a worker outranks everything of ours at any size it reaches.
WORKER = int(os.environ.get("DYCRAFT_WORKER_OOM_SCORE_ADJ") or 1000)

#: Persistent but reconstructible; below WORKER so transient scratch is taken
#: before a warm process that has to be rebuilt.
RESIDENT = int(os.environ.get("DYCRAFT_RESIDENT_OOM_SCORE_ADJ") or 800)

#: A subordinate LLM session; after anything reconstructible, before the arm.
SUBAGENT = int(os.environ.get("DYCRAFT_SUBAGENT_OOM_SCORE_ADJ") or 600)

#: The arm itself. Raised in place by `dycraft.cli`'s entry point and by the
#: layer `_env.sh`; inheritance carries it to everything the arm runs. Kept
#: clear of the desktop session's 100-200 by the same 200-point margin.
SESSION = int(os.environ.get("DYCRAFT_ARM_OOM_SCORE_ADJ") or 400)

#: Ours, highest first — the order `check_order` holds the live box to.
TIERS = (("worker", WORKER), ("resident", RESIDENT),
         ("subagent", SUBAGENT), ("session", SESSION))

#: Names the scope `enter_scope` put this process tree in. Inherited, so the
#: daemon, the agent CLI and a dispatched sub-agent running `dycraft.cli` again
#: see they are already scoped and do not nest another. `0` disables scoping.
SCOPE_ENV = "DYCRAFT_OOM_SCOPE"

#: argv[0] of the `sh` fallback in `prefix`, and how `tiered` recognizes it.
_PREFIX_MARK = "dycraft-oom-prefix"


def current() -> int | None:
    """This process's own `oom_score_adj`, or None if it cannot be read."""
    try:
        return int(Path("/proc/self/oom_score_adj").read_text())
    except (OSError, ValueError):
        return None


def prefix(adj: int) -> list:
    """argv that raises the child's `oom_score_adj` to `adj`, then execs it.

    A prefix rather than `preexec_fn=`, because the daemon is threaded --
    `Jobs.submit` runs every job on its own thread and `check_in_parallel` adds
    a pool on top -- and `preexec_fn` between fork and exec in a threaded
    process is the documented way to deadlock a child. `choom` and the `sh`
    fallback both `exec`, so the pid, the process group and the controlling
    terminal are unchanged, and bubblewrap's `--die-with-parent` still sees the
    same parent.

    Returns `[]` when the child would already be at least this exposed.
    `oom_score_adj` only goes up without CAP_SYS_RESOURCE, so asking `choom`
    for a smaller number than we already carry makes it exit *without*
    exec'ing -- which would not misprice the child, it would lose it.
    """
    here = current()
    if here is None or adj <= here:
        return []
    choom = shutil.which("choom")
    if choom:
        return [choom, "-n", str(adj), "--"]
    # util-linux is not a dependency of anything else here, so degrade to the
    # shell rather than refusing to run. `;` and not `&&`: a write that fails
    # leaves the child mispriced, which is the situation we are already in, and
    # is not a reason to drop the job.
    return ["/bin/sh", "-c",
            f'echo {adj} > /proc/self/oom_score_adj 2>/dev/null; exec "$@"',
            _PREFIX_MARK]


def tiered(argv, adj: int) -> list:
    """`argv` behind `prefix(adj)`, unless it already carries a tier prefix.

    Idempotent, so a helper every spawn passes through -- `animation/_process`'s
    `run_cancellable` -- can apply it without double-wrapping an argv that
    `animation/blender/runtime.command` has already priced. The one call to
    reach for at a spawn site: `subprocess.run(_oom.tiered(cmd, _oom.WORKER))`.
    """
    argv = list(argv)
    if argv and (Path(str(argv[0])).name == "choom" or _PREFIX_MARK in argv[:4]):
        return argv
    # Only a target that resolves gets a prefix. A missing interpreter makes
    # the spawn raise FileNotFoundError, and callers turn that OSError into
    # `worker_unavailable` -- a configuration problem. Behind `choom` it
    # would instead exec, fail, and exit non-zero: a `worker_crash`, which is
    # a different and less truthful diagnosis. So an unresolvable target is
    # left bare and fails exactly as it always did. (The CAD pool resolves its
    # interpreter for the same reason; see `_worker_pool`.)
    if not argv or shutil.which(str(argv[0])) is None:
        return argv
    return [*prefix(adj), *argv]


def untiered(argv) -> list:
    """`argv` without a leading tier prefix: the command the child becomes."""
    argv = list(argv)
    if argv and Path(str(argv[0])).name == "choom" and "--" in argv[:5]:
        return argv[argv.index("--") + 1:]
    if _PREFIX_MARK in argv[:4]:
        return argv[argv.index(_PREFIX_MARK) + 1:]
    return argv


def raise_self(adj: int) -> int | None:
    """Raise this process's own `oom_score_adj` to at least `adj`, in place.

    Never lowers: a process launched at a higher tier keeps it, so a dispatched
    sub-agent that starts at SUBAGENT and runs `dycraft.cli`'s entry point is
    left at SUBAGENT rather than failing to drop to SESSION. Fork-free, so --
    unlike `prefix` -- safe in a threaded process. Returns the value now in
    force, or None if it cannot be read.
    """
    here = current()
    if here is None or here >= adj:
        return here
    try:
        Path("/proc/self/oom_score_adj").write_text(str(adj))
    except OSError:
        return here
    return current()


def in_scope() -> bool:
    """Whether this process is already inside a scope `enter_scope` made."""
    unit = os.environ.get(SCOPE_ENV, "")
    if unit in ("", "0"):
        return False
    try:
        return f"/{unit}.scope" in Path("/proc/self/cgroup").read_text()
    except OSError:
        return False


def arm_memory_max() -> str:
    """The `MemoryMax=` one arm's scope gets. **Nothing, unless asked.**

    A run is unbounded by default, and deliberately: a single arm should be
    free to use the whole machine, and a ceiling picked here would be a guess
    about contention that only the launcher can see. This is the mechanism, not
    a policy -- `DYCRAFT_ARM_MEMORY_MAX` is how an agent launching several arms
    at once declares what each may hold (`4G`, `20%`, anything systemd accepts
    for `MemoryMax=`), and `pool.py --cap` is the same lever for a batch.

    What the scope gives unconditionally is isolation, not a limit:
    `OOMPolicy=continue`, so a kill inside an arm costs only the process killed.
    A cap turns an overrun into a memcg OOM inside that arm -- tier-ordered, its
    own workers first -- instead of a global one; worth setting when the box is
    contended, and worth leaving off when it is not.

    `0`, `none`, `off`, `max`, and `infinity` all read as "no cap", so a launcher
    can pass the variable through unconditionally.
    """
    raw = os.environ.get("DYCRAFT_ARM_MEMORY_MAX", "").strip()
    if raw.lower() in ("", "0", "none", "off", "max", "infinity"):
        return ""
    return raw


def _scope_base() -> list[str]:
    base = ["--user", "--scope", "--quiet", "--collect",
            "-p", "OOMPolicy=continue"]
    cap = arm_memory_max()
    if cap:
        base += ["-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0"]
    return base


def enter_scope(argv: list, *, label: str = "arm") -> None:
    """Re-exec this process into its own `OOMPolicy=continue` user scope.

    Returns only when it did not re-exec: already scoped, scoping disabled
    (`DYCRAFT_OOM_SCOPE=0`), or no working user manager -- no `systemd-run`, or
    no session bus, as under bubblewrap or a bare ssh login. Each of those runs
    in place with a warning rather than refusing, because an unscoped arm is
    the situation we were already in, not a reason not to run. The probe costs
    one trivial scope at startup; a sub-agent inherits `SCOPE_ENV` and skips it.

    Call it first thing in a process entry point, before threads or open
    run-dir locks exist: `execvp` keeps the pid and file descriptors but not
    threads, and nothing started before it survives.
    """
    if os.environ.get(SCOPE_ENV) == "0" or in_scope():
        return
    exe = shutil.which("systemd-run")
    why = None
    if exe is None:
        why = "no systemd-run on PATH"
    else:
        try:
            probe = subprocess.run(
                [exe, *_scope_base(), f"--unit=dycraft-probe-{os.getpid()}",
                 "--", "true"],
                capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL)
            if probe.returncode != 0:
                why = (probe.stderr or "").strip().splitlines()[-1:] or \
                      [f"exit {probe.returncode}"]
                why = why[0]
        except (OSError, subprocess.SubprocessError) as e:
            why = f"{type(e).__name__}: {e}"
    if why is not None:
        print(f"[oom] not scoped ({why}); an OOM kill anywhere in the "
              f"launching unit can stop this run with it", file=sys.stderr)
        return
    unit = f"dycraft-{label}-{os.getpid()}"
    os.environ[SCOPE_ENV] = unit
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(exe, [exe, *_scope_base(), f"--unit={unit}", "--", *argv])


def arm_entry(*, label: str = "arm") -> None:
    """What every arm's process entry point does first: its own scope, then
    the SESSION tier. `sys.orig_argv` keeps interpreter flags such as `-u`
    and the `-m` module form, so the re-exec'd process is the same command."""
    argv = list(sys.orig_argv)
    argv[0] = sys.executable
    enter_scope(argv, label=label)
    raise_self(SESSION)
    # Refresh the operator's protection on every launch rather than trusting a
    # one-off: the `user.oomd_omit` xattr lives on the cgroup, so a terminal
    # opened since the last run carries none, and an unprotected terminal is
    # what oomd reaches for when arms -- unbounded by default -- make the
    # session the pressure. Cheap (a few xattr writes) and never fatal.
    try:
        protect_shell_cgroups(verbose=False)
    except Exception:  # noqa: BLE001 -- protection is best-effort, not a gate
        pass


# ---------------------------------------------------------------------------
# Holding the live box to the order above
# ---------------------------------------------------------------------------

#: Cgroup name prefixes that hold an operator's shell, and must outlive every
#: arm. `tmux-spawn-*` is a tmux pane, `session-*` an ssh or
#: login session; both sit directly under `user@<uid>.service`.
SHELL_SCOPE_PREFIXES = ("tmux-spawn-", "session-")

#: Exact cgroup names to protect as well. `init.scope` holds `systemd --user`
#: itself: a recorded oomd kill of it took the whole login session --
#: every terminal, and every arm inside them -- with one kill.
CRITICAL_SCOPE_NAMES = ("init.scope",)

#: The xattr systemd writes for `ManagedOOMPreference=omit`, and that
#: systemd-oomd reads when it picks a cgroup to kill for memory pressure.
OOMD_OMIT_XATTR = "user.oomd_omit"


def oom_kills(proc_cgroup: Path = Path("/proc/self/cgroup"),
              cgroup_root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """How many processes of this process's cgroup the kernel's OOM killer has killed (cgroup v2
    `memory.events`, `oom_kill`, which counts global OOM kills too), or None where it cannot be read."""
    try:
        line = next(ln for ln in proc_cgroup.read_text().splitlines() if ln.startswith("0::"))
        events = cgroup_root / line[3:].lstrip("/") / "memory.events"
        for row in events.read_text().splitlines():
            key, _, value = row.partition(" ")
            if key == "oom_kill":
                return int(value)
    except (OSError, ValueError, StopIteration):
        return None
    return None


def mem_total_kb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return 0


def badness(rss_kb: int, adj: int, total_kb: int | None = None) -> float:
    """The kernel's own ranking, in KiB: `oom_badness()` from mm/oom_kill.c.

    `rss_kb` is RSS + swap + page tables, which is what that function sums.
    """
    total_kb = mem_total_kb() if total_kb is None else total_kb
    return rss_kb + adj * total_kb / 1000.0


def _proc_rows():
    total = mem_total_kb()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            adj = int(Path(f"/proc/{pid}/oom_score_adj").read_text())
            cgroup = Path(f"/proc/{pid}/cgroup").read_text().strip()
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except (OSError, ValueError):
            continue
        fields = {}
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in ("Name", "VmRSS", "VmSwap", "VmPTE"):
                fields[key] = value.split()[0] if value.split() else "0"
        rss = sum(int(fields.get(k, 0)) for k in ("VmRSS", "VmSwap", "VmPTE"))
        yield {"pid": int(pid), "adj": adj, "rss_kb": rss,
               "comm": fields.get("Name", "?"),
               "cgroup": cgroup.rsplit("/", 1)[-1],
               "cmd": cmd.decode("utf-8", "replace").strip(),
               "badness_kb": badness(rss, adj, total)}


def kill_order(rows=None) -> list[dict]:
    """Every process, in the order the kernel would take them. Highest first."""
    rows = list(_proc_rows() if rows is None else rows)
    return sorted(rows, key=lambda r: -r["badness_kb"])


def check_order(rows=None) -> list[str]:
    """Complaints about the live ranking, or `[]` when it matches the policy.

    Two assertions, and they are the whole policy: among *our* processes a
    lower tier never outranks a higher one, and no operator shell, login
    session or tmux/ssh process ranks above any of ours. Everything else on the
    box is not ours to order.

    "Ours" is recognized by cgroup (`dycraft-*`) or by carrying one of the tier
    numbers, current or legacy, so a box still running processes started before
    a re-spacing is judged on the order they are actually in rather than
    silently skipped.
    """
    ranked = kill_order(rows)
    names = {adj: name for name, adj in TIERS}
    legacy = {1000: "worker", 900: "worker", 850: "resident", 800: "resident",
              700: "subagent", 600: "subagent", 500: "session", 400: "session"}

    def mine(row):
        return row["cgroup"].startswith("dycraft-") or row["adj"] in legacy

    def label(row):
        return names.get(row["adj"]) or legacy.get(row["adj"]) or f"adj {row['adj']}"

    complaints = []
    ours = [r for r in ranked if mine(r)]
    tiered = [r for r in ours if r["adj"] in legacy]
    # A process inside one of our cgroups that carries no tier is priced like a
    # shell: it dies in the first wave, whatever it is holding together.
    for row in ours:
        if row["adj"] not in legacy:
            complaints.append(
                f"untiered process in {row['cgroup']}: {row['comm']} pid "
                f"{row['pid']} at adj {row['adj']} — it would die with the "
                f"shells rather than after the workers ({row['cmd'][:60]})")
    seen: dict[int, dict] = {}
    for row in tiered:
        for other_adj, other in seen.items():
            if other_adj < row["adj"]:
                complaints.append(
                    f"{label(other)} pid {other['pid']} ({other['rss_kb']//1024}"
                    f" MiB, adj {other_adj}) outranks {label(row)} pid "
                    f"{row['pid']} ({row['rss_kb']//1024} MiB, adj {row['adj']})"
                    f" — a lower tier would die first")
        seen.setdefault(row["adj"], row)

    if tiered:
        cutoff = ranked.index(tiered[-1])
        for row in ranked[:cutoff]:
            if mine(row):
                continue
            if (row["cgroup"].startswith(SHELL_SCOPE_PREFIXES)
                    or row["comm"] in ("tmux: server", "sshd", "bash", "zsh",
                                       "login", "systemd")):
                complaints.append(
                    f"shell/session process {row['comm']} pid {row['pid']} "
                    f"(adj {row['adj']}, cgroup {row['cgroup']}) ranks above "
                    f"one of ours — it would die first")
    return complaints


def protect_shell_cgroups(verbose: bool = True) -> list[str]:
    """Mark the operator's shell scopes `omit` for systemd-oomd.

    oomd ignores `oom_score_adj` entirely and kills whole cgroups by memory
    pressure, so the tier order says nothing about what it takes -- in a
    recorded batch it took four terminal scopes. This is the one lever that
    does apply, and it needs no privilege here: the monitored ancestor
    (`user@<uid>.service`) and these scopes share an owner, which is the
    condition systemd.resource-control(5) puts on honoring the preference.

    Returns the cgroup paths marked. Safe to re-run; already-marked scopes are
    left alone and an unwritable one is reported rather than raised.
    """
    root = Path("/sys/fs/cgroup/user.slice") / f"user-{os.getuid()}.slice" \
        / f"user@{os.getuid()}.service"
    marked = []
    if not root.is_dir():
        return marked
    for path in sorted(root.rglob("*.scope")):
        if not (path.name.startswith(SHELL_SCOPE_PREFIXES)
                or path.name in CRITICAL_SCOPE_NAMES):
            continue
        try:
            if OOMD_OMIT_XATTR in os.listxattr(path):
                continue
            os.setxattr(path, OOMD_OMIT_XATTR, b"1")
            marked.append(str(path))
            if verbose:
                print(f"[oom] oomd will not take {path.name}", file=sys.stderr)
        except OSError as e:
            if verbose:
                print(f"[oom] could not protect {path.name}: {e}",
                      file=sys.stderr)
    return marked


def _main(argv: list[str]) -> int:
    if "--protect-shells" in argv:
        protect_shell_cgroups()
    total = mem_total_kb()
    print(f"MemTotal {total/1048576:.1f} GiB — 1 adj point = "
          f"{total/1000/1024:.0f} MiB of badness")
    print("tiers: " + " > ".join(f"{n} {a}" for n, a in TIERS)
          + "  > desktop 100-200 > tmux/ssh 0/-1000")
    ranked = kill_order()
    print(f"\n{'#':>2} {'badness':>9} {'adj':>5} {'rss':>8}  {'comm':16} cgroup")
    for i, row in enumerate(ranked[:15], 1):
        print(f"{i:2d} {row['badness_kb']/1048576:8.1f}G {row['adj']:5d} "
              f"{row['rss_kb']/1024:7.0f}M  {row['comm']:16.16} "
              f"{row['cgroup']:.44}")
    complaints = check_order(ranked)
    print()
    for c in complaints:
        print(f"!! {c}")
    if not complaints:
        print("order holds: ours die highest tier first, shells last")
    return 1 if complaints else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
