"""Run benchmark briefs through a bare CLI agent with Blender — the paper's bare-agent baseline.

    python -m dycraft.benchmark.bare --briefs short --agent codex --model gpt-6-astra \\
        --effort medium --hours 8 --out runs/bench_short_codex
    python -m dycraft.benchmark.bare --briefs B01S --agent claude --model claude-opus-5-5 \\
        --effort high --hours 4 --out runs/bench_mixed

The bare agent gets no DyCraft harness, skills or prompts: one Codex or Claude Code session, a
Python with Blender's `bpy`, ffmpeg, and an empty `/workspace`. This reproduces the setup of the
paper's 24 bare runs:

* **Prompt** — the paper's task wrapper around the brief, `---`, then the bare guidance
  (`benchmark/bare_prompt.md`, byte-identical to what those runs received), on stdin. The
  environment sentence names the installed Blender, which was 4.5.13 for those runs. By default no
  film-length or budget paragraph, and `--hours` is a hard stop only. Opt-in `--budget-hours` and
  `--film-seconds` add the DyCraft arm's two paragraphs verbatim after the brief, so both arms can be
  told the same length and budget. (`benchmark/briefs.py` has the details.)
* **Sandbox** — `unshare -Ur bwrap` (or bwrap's own `--unshare-user` where the host allows only
  bwrap to make user namespaces, as stock Ubuntu 24.04 does): system directories, the Blender environment, and the agent
  binary read-only; `/workspace` the only writable project path; a private, empty home holding only
  the agent's credentials; no skills, no `AGENTS.md`; separate PID/IPC/UTS namespaces; the GPU
  devices passed through. A preflight inside the sandbox checks that this checkout and the agent's
  own configuration are out of reach before the agent starts.
* **Agent** — Codex runs `codex exec` with the paper's flags: user config, rules, skills, plugins,
  apps, native sub-agents (`multi_agent`) and web search off; approvals go to Codex's automatic
  reviewer (`--approve-for-me`) instead of being bypassed.
  Claude Code runs `claude -p` with the same spirit: sub-agent and web tools disallowed,
  permissions bypassed inside the sandbox.
* **G1 briefs** (B01-B08) get the Unitree G1 description read-only at `/assets/g1`, as the paper's
  did, copied from `assets/robots/g1_description/` (the URDF, every mesh it references, LICENSE).

The deliverable the guidance asks for, `output/blend/final.blend`, is then staged as an
animation bundle in the arm directory (`benchmark/stage.py`), so the viewer plays it like a DyCraft
run. The bundle's film is the agent's own `output/film/final.mp4`, or, when it delivered none (a hard
stop before its render, say), one the harness renders from the blend at 960x540 (`--no-render` skips
that). `python -m dycraft.benchmark.score` scores both arms with the paper's physical metrics.

Requires Linux with unprivileged user namespaces and `bwrap` (as dycraft's own Blender layer does).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .._home import state_path
from . import briefs as B
from .run import add_common

REPO = B.REPO
#: User state, not package data: under `dycraft._home.home()` (the checkout, or the launch
#: directory for an installed copy).
G1_SOURCE = state_path("assets", "robots", "g1_description")
#: The command that downloads it: installing any of the library also installs the G1's description.
G1_FETCH_COMMAND: str | None = "python -m dycraft.install library --names g1_with_dex1_v3"
#: The paper's bare runs had a 72-hour limit.
DEFAULT_HOURS = 72.0


def arm_name(agent: str) -> str:
    return f"bare_{agent}"


# ---------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------

def _how_to_get_g1() -> str:
    if G1_FETCH_COMMAND:
        return f"It is not in git; fetch it with:\n    {G1_FETCH_COMMAND}"
    return (f"It is not in git: put the description there ({B.G1_URDF} and the meshes it "
            "references, with its LICENSE).")


def copy_g1(dest: Path) -> None:
    """The G1 URDF, every mesh it references, and its LICENSE, as the paper's bare runs got them."""
    urdf = G1_SOURCE / B.G1_URDF
    if not urdf.is_file():
        raise SystemExit(f"G1 briefs need the Unitree G1 robot description at {G1_SOURCE}/ "
                         f"(missing {urdf.name}). {_how_to_get_g1()}")
    files = {urdf}
    if (G1_SOURCE / "LICENSE").is_file():
        files.add(G1_SOURCE / "LICENSE")
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        files.add(G1_SOURCE / mesh.attrib["filename"])
    missing = sorted(str(p.relative_to(G1_SOURCE)) for p in files if not p.is_file())
    if missing:
        raise SystemExit(f"the G1 description at {G1_SOURCE}/ is incomplete: {len(missing)} file(s) "
                         f"its URDF references are missing (e.g. {missing[0]}). {_how_to_get_g1()}")
    for path in sorted(files):
        target = dest / path.relative_to(G1_SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def installed_blender(prefix: Path) -> str:
    """The Blender version `prefix`'s `bpy` reports, as the environment sentence names it
    (`4.5.15`); the paper's when it cannot be read, as under `--dry-run` without a Blender env."""
    from ..render._blender_env import probe_blender_version
    found = re.match(r"\d+\.\d+\.\d+", probe_blender_version(str(prefix / "bin" / "python")) or "")
    return found.group(0) if found else B.PAPER_BLENDER


def blender_prefix() -> Path:
    """The environment whose `bin/python` imports `bpy` (`.blender_python`)."""
    from ..render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        raise SystemExit("no Blender Python: run `python -m dycraft.install blender`, or set "
                         "DYCRAFT_BLENDER_PYTHON.")
    prefix = subprocess.run([interpreter, "-c", "import sys; print(sys.prefix)"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(prefix).resolve()


#: The overrides DyCraft's own backends honor (`cli/backends.py`), used when they name an executable.
EXECUTABLE_ENV = {"codex": "DYCRAFT_CODEX_EXECUTABLE", "claude": "DYCRAFT_CLAUDE_EXECUTABLE"}


def agent_binary(agent: str, given: str | None) -> Path:
    found = given or shutil.which(os.environ.get(EXECUTABLE_ENV[agent]) or agent) or shutil.which(agent)
    if not found:
        raise SystemExit(f"no `{agent}` on PATH; pass --agent-bin or set {EXECUTABLE_ENV[agent]}")
    path = Path(found).resolve()
    native = codex_native(path) if agent == "codex" else None
    if native:
        print(f"[benchmark.bare] {path} is npm's Node launcher; running the native binary it ships, "
              f"{native}", file=sys.stderr)
        return native
    return path


def codex_native(launcher: Path) -> Path | None:
    """The native `codex` that npm's package ships beside its Node launcher (`bin/codex.js`).

    The sandbox mounts system directories, so a Node from conda or nvm is not there and the launcher
    cannot start; the native binary needs no Node.
    """
    try:
        first = launcher.read_bytes()[:256].split(b"\n", 1)[0]
    except OSError:
        return None
    if not (first.startswith(b"#!") and b"node" in first):
        return None
    package = launcher.parent.parent
    for pattern in ("node_modules/@openai/codex-*/vendor/*/bin/codex", "vendor/*/bin/codex",
                    "vendor/*/codex/codex"):
        for hit in sorted(package.glob(pattern)):
            if hit.is_file() and os.access(hit, os.X_OK):
                return hit.resolve()
    return None


def agent_mounts(binary: Path) -> list[Path]:
    """Read-only directories the agent binary needs: its own directory, the `node_modules` tree an
    npm install lives in, and a script's interpreter prefix."""
    mounts = [binary.parent]
    for parent in binary.parents:
        if parent.name == "node_modules":
            mounts.append(parent)
            break
    try:
        head = binary.read_bytes()[:256]
    except OSError:
        head = b""
    if head.startswith(b"#!"):
        words = head[2:].split(b"\n", 1)[0].decode(errors="replace").split()
        interp = words[1] if words and words[0].endswith("/env") and len(words) > 1 else (words[0] if words else "")
        resolved = shutil.which(interp) if interp and "/" not in interp else interp
        if resolved:
            mounts.append(Path(resolved).resolve().parent.parent)
    out = []
    for m in mounts:
        if not any(str(m).startswith(str(o) + "/") or m == o for o in out):
            out.append(m)
    return out


def credentials(agent: str, config: str | None) -> tuple[Path, list[Path]]:
    """The agent's config dir on this host, and the credential files copied into the private home."""
    if agent == "codex":
        home = Path(config or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
        return home, [p for p in [home / "auth.json"] if p.is_file()]
    home = Path(config or os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
    return home, [p for p in [home / ".credentials.json"] if p.is_file()]


# ---------------------------------------------------------------------------------------------------
# Sandbox and agent command
# ---------------------------------------------------------------------------------------------------

_USERNS_PREFIX: list[str] | None = None


def userns_prefix() -> list[str]:
    """`[unshare, "-Ur"]`, or `[]` when this host will not let `unshare` make a user namespace.

    The same probe as `animation/blender/runtime._userns_prefix`, without `-n`: the bare agent
    needs the network to reach its model, as the paper's did. Where `unshare -Ur` is refused (stock
    Ubuntu 24.04: `kernel.apparmor_restrict_unprivileged_userns=1`, with an AppArmor profile that
    lets only bwrap create user namespaces), bwrap makes the user namespace itself
    (`--unshare-user`). The isolation is the same; only the uid inside differs (0 vs the real uid).
    """
    global _USERNS_PREFIX
    if _USERNS_PREFIX is None:
        unshare, ok = shutil.which("unshare"), False
        if unshare:
            try:
                ok = subprocess.run([unshare, "-Ur", "true"], capture_output=True,
                                    timeout=30).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        _USERNS_PREFIX = [unshare, "-Ur"] if ok else []
    return _USERNS_PREFIX


def sandbox(root: Path, home: Path, prefix: Path, mounts: list[Path], g1: bool) -> list[str]:
    bwrap = shutil.which("bwrap") or str(Path(sys.executable).parent / "bwrap")
    if not Path(bwrap).is_file():
        raise SystemExit("the bare arm runs in bubblewrap; install `bwrap`")
    userns = userns_prefix()
    cmd = [*userns, bwrap] + ([] if userns else ["--unshare-user"])
    cmd += ["--unshare-pid", "--unshare-ipc", "--unshare-uts", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            cmd += ["--ro-bind", str(Path(path).resolve()), path]
    cmd += ["--symlink", "usr/bin", "/bin"]
    for path in ("/etc/ssl", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/fonts",
                 "/etc/glvnd", "/etc/vulkan", "/etc/ld.so.cache"):
        if Path(path).exists():
            cmd += ["--ro-bind", path, path]
    cmd += ["--ro-bind", str(prefix), str(prefix)]
    for m in mounts:
        cmd += ["--ro-bind", str(m), str(m)]
    cmd += ["--bind", str(root / "workspace"), "/workspace",
            "--bind", str(home), "/home/agent",
            "--ro-bind", str(root / "empty"), "/home/agent/.codex/skills",
            "--ro-bind", str(root / "empty"), "/home/agent/.claude/skills",
            "--ro-bind", str(root / "empty"), "/home/agent/.agents"]
    if g1:
        cmd += ["--ro-bind", str(root / "inputs" / "g1"), "/assets/g1"]
    cmd += ["--chdir", "/workspace"]
    for path in [Path("/dev/dri"), *Path("/dev").glob("nvidia*")]:
        if path.exists():
            cmd += ["--dev-bind", str(path), str(path)]
    # Always, with or without a GPU: `import bpy` segfaults without /sys.
    cmd += ["--ro-bind", "/sys", "/sys"]
    env = {"HOME": "/home/agent", "CODEX_HOME": "/home/agent/.codex",
           "CLAUDE_CONFIG_DIR": "/home/agent/.claude",
           "PATH": ":".join([str(prefix / "bin"), *(str(m) for m in mounts), "/usr/bin", "/bin"]),
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "16",
           "OPENBLAS_NUM_THREADS": "1", "LANG": "C.UTF-8",
           "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
           # `unshare -Ur` makes the agent root inside the namespace, and Claude Code refuses
           # `bypassPermissions` as root unless it is told it runs in a deliberate sandbox.
           "IS_SANDBOX": "1"}
    for key in ("CUDA_VISIBLE_DEVICES", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key, value in env.items():
        cmd += ["--setenv", key, value]
    return cmd


def agent_command(agent: str, binary: Path, model: str | None, effort: str | None) -> list[str]:
    if agent == "codex":
        # The paper's `codex exec` flags, except approvals: Codex's automatic reviewer
        # (`--approve-for-me`) rather than `--dangerously-bypass-approvals-and-sandbox`, which an
        # enterprise workspace may forbid (Codex then overrides it and, in `exec`, cannot run a
        # command). Codex's own sandbox cannot start inside this one, so its commands go to the
        # reviewer, and this bwrap sandbox confines them.
        cmd = [str(binary), "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "--approve-for-me",
               "--enable", "skip_host_skill_discovery", "--disable", "skill_search",
               "--disable", "plugins", "--disable", "apps", "--disable", "multi_agent",
               "--disable", "shell_snapshot"]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f'model_reasoning_effort="{effort}"']
        return cmd + ["-c", 'service_tier="default"', "-c", 'web_search="disabled"',
                      "--json", "-o", "/workspace/final_response.md", "-"]
    cmd = [str(binary), "-p", "--output-format", "stream-json", "--verbose",
           "--permission-mode", "bypassPermissions",
           "--disallowedTools", "Agent", "Task", "WebSearch", "WebFetch"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


_PREFLIGHT = '''import json, os, pathlib, sys, bpy
blocked, g1 = json.loads(sys.argv[1]), sys.argv[2] == "1"
assert all(not pathlib.Path(p).exists() for p in blocked), [p for p in blocked if pathlib.Path(p).exists()]
assert not list(pathlib.Path("/home/agent/.codex/skills").iterdir())
assert not list(pathlib.Path("/home/agent/.claude/skills").iterdir())
assert pathlib.Path("/assets/g1").exists() == g1
assert not pathlib.Path("/workspace/AGENTS.md").exists()
print(json.dumps(dict(isolation_passed=True, blender=bpy.app.version_string, cwd=os.getcwd())))
'''


# ---------------------------------------------------------------------------------------------------
# One brief
# ---------------------------------------------------------------------------------------------------

def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _run_one(brief: B.Brief, horizon: str | None, args) -> dict:
    out = Path(args.out).resolve()
    root = out / brief.slug / arm_name(args.agent)
    if (root / "control" / "status.json").exists():
        raise SystemExit(f"{root} has already run; choose another --out or remove it")
    for name in ("control", "workspace", "inputs", "empty"):
        (root / name).mkdir(parents=True, exist_ok=True)
    # The paper's form unless the same length and budget are asked for (`--film-seconds`,
    # `--budget-hours`); a bare `--film-seconds default` takes the DyCraft arm's per-horizon value.
    film = args.film_seconds
    if film == "default":
        film = B.FILM_SECONDS[horizon or brief.horizon]
    prefix = blender_prefix()
    blender = installed_blender(prefix)
    paragraphs = dict(film_seconds=film, budget_hours=args.budget_hours, blender=blender)
    task, guidance = B.bare_task(brief, **paragraphs), B.BARE_GUIDANCE.read_text()
    prompt = B.bare_prompt(brief, **paragraphs)
    (root / "control" / "task_prompt.md").write_text(task)
    (root / "control" / "bare_animation_guidance.md").write_text(guidance)
    (root / "workspace" / "prompt.md").write_text(prompt)
    if brief.uses_g1:
        copy_g1(root / "inputs" / "g1")
    if args.dry_run:
        binary = Path(args.agent_bin or shutil.which(args.agent) or args.agent)
    else:
        binary = agent_binary(args.agent, args.agent_bin)
    config_dir, creds = credentials(args.agent, args.agent_config)
    record = {"brief": brief.id, "slug": brief.slug, "arm": arm_name(args.agent), "horizon": horizon or brief.horizon,
              "agent": args.agent, "model": args.model, "effort": args.effort,
              "hours": args.hours, "budget_hours": args.budget_hours, "film_seconds": film,
              "blender": blender,
              "paper_prompt": not (film or args.budget_hours) and blender == B.PAPER_BLENDER,
              "run_dir": str(root), "agent_bin": str(binary),
              "isolation": "bubblewrap filesystem allowlist; separate PID namespace",
              "skills": False, "harness": False,
              "guidance_sha256": hashlib.sha256(guidance.encode()).hexdigest(),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "inputs": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((root / "inputs").rglob("*")) if p.is_file()}}
    _write(root / "control" / "config.json", record)
    cmd_agent = agent_command(args.agent, binary, args.model, args.effort)
    if args.dry_run:
        print(f"[benchmark.bare] {brief.id}: prompt {root / 'workspace' / 'prompt.md'}\n    "
              + " ".join(cmd_agent))
        return record
    if not args.check and not creds and not (os.environ.get("ANTHROPIC_API_KEY")
                                             or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit(f"no {args.agent} credentials found in {config_dir}; pass --agent-config")

    with tempfile.TemporaryDirectory(prefix=f"bare-{brief.id.lower()}-home-") as tmp:
        home = Path(tmp)
        for sub in (".codex/skills", ".claude/skills", ".agents"):
            (home / sub).mkdir(parents=True)
        for cred in creds:
            target = home / (".codex" if args.agent == "codex" else ".claude") / cred.name
            shutil.copyfile(cred, target)
            target.chmod(0o600)
        base = sandbox(root, home, prefix, agent_mounts(binary), brief.uses_g1)
        # What the agent must not reach. Not the whole home: the Blender environment or the agent
        # binary may live under it, and their bind mounts create its parent directories.
        blocked = sorted({str(REPO), str(config_dir.resolve()), str(Path.home() / ".ssh")})
        check = subprocess.run(base + [str(prefix / "bin" / "python"), "-I", "-c", _PREFLIGHT,
                                       json.dumps(blocked), "1" if brief.uses_g1 else "0"],
                               text=True, capture_output=True, timeout=120)
        _write(root / "control" / "preflight.json", {"returncode": check.returncode,
                                                     "stdout": check.stdout, "stderr": check.stderr})
        if check.returncode:
            raise SystemExit(f"{brief.id}: sandbox preflight failed, see {root}/control/preflight.json")
        if args.check:
            probe = subprocess.run(base + ["/bin/sh", "-c",
                                           "command -v python && command -v ffmpeg && "
                                           "python -c 'import bpy' && "
                                           f"{binary} --version && touch /workspace/.write_test"],
                                   text=True, capture_output=True, timeout=120)
            (root / "workspace" / ".write_test").unlink(missing_ok=True)
            print(f"[benchmark.bare] {brief.id}: preflight {check.stdout.strip()}\n"
                  f"[benchmark.bare] tools rc={probe.returncode}: "
                  + (probe.stdout + probe.stderr).strip().replace("\n", " | "))
            return {**record, "check": probe.returncode == 0}
        started = time.time()
        _write(root / "control" / "status.json", {"state": "agent_running", "started": started})
        print(f"[benchmark.bare] {brief.id}: {arm_name(args.agent)} -> {root}", flush=True)
        with (root / "workspace" / "prompt.md").open() as stdin, \
                (root / "control" / "events.jsonl").open("w") as stdout, \
                (root / "control" / "stderr.log").open("w") as stderr:
            proc = subprocess.Popen(base + cmd_agent, stdin=stdin, stdout=stdout, stderr=stderr)
            try:
                rc = proc.wait(timeout=args.hours * 3600 if args.hours else None)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    rc = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                record["hard_stop"] = True
        # The session transcripts, never the credentials.
        for sessions in (home / ".codex" / "sessions", home / ".claude" / "projects"):
            if sessions.exists():
                shutil.copytree(sessions, root / "control" / "sessions" / sessions.name,
                                dirs_exist_ok=True)
    record.update(returncode=rc, wall_seconds=round(time.time() - started, 1))
    _write(root / "control" / "status.json", {"state": "agent_exited", "returncode": rc,
                                              "ended": time.time()})
    blend = root / "workspace" / "output" / "blend" / "final.blend"
    film = root / "workspace" / "output" / "film" / "final.mp4"
    record["blend"] = str(blend) if blend.is_file() else None
    record["agent_film"] = record["film"] = str(film) if film.is_file() else None
    if blend.is_file():
        from .stage import FILM_NAME, stage
        try:
            staged = stage(
                blend, root, scene_ref=f"benchmark:{brief.slug}:{arm_name(args.agent)}",
                relocate={"origin": str(blend.parent),
                          "prefixes": {"/workspace": str(root / "workspace")}},
                film=film, render_missing_film=not args.no_render)
            record["staged"] = True
            ev = staged.get("evaluation") or {}
            if ev.get("fps") and ev.get("n_frames"):
                record["seconds"] = round(ev["n_frames"] / ev["fps"], 1)
            # The bundle's film: the agent's own, or the one rendered for it when it had none.
            record["film_source"] = (staged.get("film") or {}).get("source")
            if (root / "bundle" / FILM_NAME).is_file():
                record["film"] = str(root / "bundle" / FILM_NAME)
            elif (staged.get("film") or {}).get("error"):
                record["film_error"] = staged["film"]["error"]
        except Exception as error:                                     # noqa: BLE001
            record["staged"] = {"error": f"{type(error).__name__}: {error}"}
    (root / "benchmark").mkdir(exist_ok=True)
    _write(root / "benchmark" / "record.json", record)
    # Enough for the viewer to list the arm; it plays the staged bundle by its `ir.json`.
    _write(root / "meta.json", {
        "run_id": arm_name(args.agent), "layer": "bare_agent", "runtime": args.agent,
        "model": args.model, "reasoning_effort": args.effort, "prompt": B.bare_brief(brief),
        "outcome": "delivered" if record["blend"] else "no_blend",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.bare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap, imported_characters=False)
    ap.add_argument("--g1", action="store_true",
                    help="with --prompt: give the agent the Unitree G1 at /assets/g1, with the "
                         "sentence the paper's G1 briefs gave their bare runs")
    ap.add_argument("--hours", type=float, default=None,
                    help=f"hard stop in hours; the agent is not told it (default: the budget plus "
                         f"20 minutes with --budget-hours, else {DEFAULT_HOURS:g}, the paper's limit)")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="opt-in: tell the agent the dycraft arm's wall-clock budget paragraph "
                         "(the paper's bare runs were told none)")
    ap.add_argument("--film-seconds", default=None,
                    help="opt-in: tell the agent the dycraft arm's film-length paragraph; N, LOW-HIGH, "
                         "or `default` for the dycraft arm's value for the brief's set")
    ap.add_argument("--agent-bin", default=None, help="the codex or claude executable (default: PATH)")
    ap.add_argument("--agent-config", default=None,
                    help="config dir holding the agent's credentials (default ~/.codex or ~/.claude)")
    ap.add_argument("--no-render", action="store_true",
                    help="do not render a film for an agent that delivered none (by default the "
                         "harness renders it from the blend at 960x540)")
    ap.add_argument("--check", action="store_true",
                    help="build the sandbox and run its preflight and tool probe; start no agent")
    args = ap.parse_args(argv)
    if args.effort is None:
        # DyCraft's own default, so the bare arm runs at the effort the DyCraft arm does.
        from ..cli.cli import DEFAULT_REASONING_EFFORT
        args.effort = DEFAULT_REASONING_EFFORT
    if args.film_seconds is not None and args.film_seconds != "default":
        args.film_seconds = B.parse_film_seconds(args.film_seconds)
    if args.budget_hours is not None and args.budget_hours < 0:
        raise SystemExit("--budget-hours must be 0 (none) or more")
    if args.hours is None:
        args.hours = (args.budget_hours + 20 / 60) if args.budget_hours else DEFAULT_HOURS
    # The bare arm has no asset library, so it never takes an imported-character variant (the default).
    chosen, horizon = B.choose(args)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(pool.map(lambda b: _run_one(b, horizon, args), chosen))
    if args.check:
        ok = all(r.get("check") for r in records)
        print(f"[benchmark.bare] sandbox check {'passed' if ok else 'FAILED'}")
        return 0 if ok else 1
    if not args.dry_run:
        for r in records:
            print(f"[benchmark.bare] {r['brief']}: rc={r.get('returncode')} "
                  f"blend={'yes' if r.get('blend') else 'no'} film={'yes' if r.get('film') else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
