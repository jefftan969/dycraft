# Install Guide

DyCraft runs on Linux (x86_64). A GPU is optional: Blender renders on the CPU when there is none, and only inverse
kinematics needs an NVIDIA GPU.

## 1. Clone DyCraft and Create Its Environment

With [uv](https://docs.astral.sh/uv/), which also installs Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/jefftan969/dycraft.git && cd dycraft
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e .
```

With [conda](https://docs.conda.io/):

```bash
git clone https://github.com/jefftan969/dycraft.git && cd dycraft
conda env create -f environment-main.yml -n dycraft && conda activate dycraft
pip install -e .
```

The later steps are the same for both (see [Choosing uv or conda](#choosing-uv-or-conda)).

## 2. Run the Setup Check

```bash
python -m dycraft.install check
```

It marks each check OK, WARN, FAIL, or UNAVAILABLE and ends with the command that fixes each FAIL. Steps 3 to 6 clear
the FAILs, in any order.

## 3. Install the System Tools

If the check reports `bubblewrap`, `socat`, `ffmpeg`, or the GL libraries missing, install them without root:

```bash
python -m dycraft.install tools
```

On Ubuntu 24.04, AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`) denies user namespaces to any `bwrap`
without a profile that grants them, and adding one needs root. See [Sandbox Fails to Start](#sandbox-fails-to-start).

## 4. Set Up CAD

```bash
python -m dycraft.install cad
```

## 5. Install Blender

```bash
python -m dycraft.install blender
```

It downloads about 380 MB, uses 2.2 GB of disk, and takes about five minutes.

## 6. Install a Coding Agent and Log In

Install [Claude Code](https://code.claude.com/docs/en/setup) or [Codex](https://learn.chatgpt.com/docs/codex/cli)
with its official instructions, then log in with `claude auth login` or `codex login`.

To use more than one account, log in to each in its own folder. `CLAUDE_CONFIG_DIR=~/.claude2 claude auth login`
gives `--backend claude2`, and `CODEX_HOME=~/.codex2 codex login` gives `--backend codex2`. Codex's own installer
keeps the program in `~/.codex`, so give a second Codex account its own folder this way rather than replacing
`~/.codex`.

## 7. Install the Optional Parts

| Part | Command | Without it |
|---|---|---|
| [Asset library](usage.md#asset-library): the people, animals, and robots that briefs cast | `python -m dycraft.install library` | Briefs that need a character or a robot cannot find one |
| [Inverse kinematics](#inverse-kinematics) (needs an NVIDIA GPU) | `python -m dycraft.install ik` | Agents write their own pose solvers |
| [Motion capture](#motion-capture) | `python -m dycraft.install mocap` (`--dog` adds dogs) | Agents author all motion themselves |
| [Interactive viewer](../README.md#interactive-viewer) | Built with npm | Runs work the same |

## 8. Check Again and Make a Test Run

```bash
python -m dycraft.install check
python -m dycraft.cli --backend claude --layer object --prompt "A wooden stool with three legs."
```

Every launch runs the same checks first and stops with the check's fix if one fails.

## Optional Parts

### Inverse Kinematics

Inverse kinematics needs an NVIDIA GPU, and `install ik` takes a few minutes. On a machine without one, it installs
nothing, and runs go ahead without inverse kinematics.

### Motion Capture

```bash
python -m dycraft.install mocap          # people: the CMU motion-capture library (~1 GB download, 3.2 GB on disk)
python -m dycraft.install mocap --dog    # also dogs: the MANN captures (asks you to accept CC BY-NC 4.0)
```

The libraries go in `assets/mocap/`, and agents start motion from their clips. Their terms:

- CMU Graphics Lab Motion Capture Database: free for research and for use in products, but the data may not be
  resold, even converted. If you publish, CMU asks for this acknowledgement: "The data used in this project was
  obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217."
- MANN dog captures (Zhang, Starke, Komura, and Saito, SIGGRAPH 2018): CC BY-NC 4.0, non-commercial use with
  attribution. Films made with them carry the same terms.

## Reference

### Where DyCraft Keeps Its Files

DyCraft keeps its environments, the asset library, the motion capture libraries, the paper's films, and the default
`runs/` directory in the first of these directories:

1. `$DYCRAFT_HOME`, if set;
2. the clone, when DyCraft runs from one;
3. the current directory, after a regular `pip install .`.

After a regular install, run `python -m dycraft.install` and launch runs from the same directory, or set
`DYCRAFT_HOME`. Each launch prints the runs directory it uses.

To install without a clone, run `pip install git+https://github.com/jefftan969/dycraft` in a Python 3.10 to 3.14
environment, then run the `python -m dycraft.install` steps in the directory you will launch DyCraft from.

### Choosing uv or conda

`install blender` and `install ik` use conda when DyCraft's own environment is a conda environment, and uv otherwise.
`--conda` or `--uv` chooses one yourself.

### CAD

The main environment includes build123d 0.11.1, the only version DyCraft accepts. To use another environment for CAD:

```bash
python -m dycraft.install cad --python /path/to/env/bin/python
```

### Check Options

```bash
python -m dycraft.install check                      # quick checks: seconds, no network, no model calls
python -m dycraft.install check --deep               # runs each part through its real code (minutes)
python -m dycraft.install check --e2e                # runs the animation pipeline end to end, without a model
python -m dycraft.install check --backend claude --capabilities   # tests one agent (a few model requests)
```

Run it with the Python you launch DyCraft with. Without an NVIDIA GPU, it marks GPU rendering and inverse kinematics
unavailable, which is expected.

## Fixing Install Problems

### An Object Run Fails With "No build123d Interpreter Found"

DyCraft found no interpreter with build123d 0.11.1. Launch from the main environment, which includes it, or run
`python -m dycraft.install cad`.

### `ModuleNotFoundError: No module named 'dycraft'`

Run `uv pip install -e .` or `pip install -e .` from the clone, in the environment you launch from.

### Blender or the Asset Library Is Not Found

They are in the clone or `$DYCRAFT_HOME` (see [Where DyCraft Keeps Its Files](#where-dycraft-keeps-its-files)). Launch
from the directory where you ran `python -m dycraft.install`, or set `DYCRAFT_HOME`.

### Inverse Kinematics Is Unavailable or Fails With "No Kernel Image Is Available"

"unavailable here: no NVIDIA GPU" means the machine has none, and runs go ahead without inverse kinematics.
"installed, but no NVIDIA GPU is visible" names the reason, such as an empty `CUDA_VISIBLE_DEVICES`, and runs leave it
out until a GPU is visible. "not configured" on a machine with an NVIDIA GPU means no IK environment is installed. Run
`python -m dycraft.install ik`. A "no kernel image" or Warp "requires sm_75" error means the environment was built for
a different GPU. Rerun the same command to rebuild it for the GPU it finds.

### Fluid Simulation Fails

Mantaflow does not run in the `bpy` module, so fluid simulation needs the official Blender 4.5.13 binary. Run
`python -m dycraft.install blender`, which downloads it.

### Sandbox Fails to Start

Claude Code and Codex run the agent's commands in a sandbox built on `bubblewrap`, and DyCraft does not start them
without one. A launch then refuses with "Claude Code's Bash sandbox cannot start here: …" or "Codex's sandbox does not
work here". `python -m dycraft.install check` runs the same tests. The common causes are:

- `bubblewrap` or `socat` is missing. Install them (step 3).
- Ubuntu 24.04's AppArmor restriction on user namespaces, below.
- A full or unwritable `/tmp` or `$TMPDIR` ("Can't mkdir /tmp/.git: No space left on device"). Free space, or set
  `TMPDIR` to a writable directory.

On Ubuntu 24.04, AppArmor lets a program create user namespaces only if a profile allows it, and other copies of `bwrap`
fail with "Failed RTM_NEWADDR: Operation not permitted" or "setting up uid map: Permission denied". Some systems ship a
profile for `/usr/bin/bwrap` with the `bubblewrap` package, and others, such as stock AWS images, do not. With root, the
check lists the steps your system needs, from these:

```bash
# the system's bubblewrap, if /usr/bin/bwrap is missing
sudo apt-get install bubblewrap socat
# a profile that lets /usr/bin/bwrap create user namespaces, if none does
printf 'abi <abi/4.0>,\ninclude <tunables/global>\nprofile bwrap /usr/bin/bwrap flags=(unconfined) {\n  userns,\n}\n' | sudo tee /etc/apparmor.d/bwrap && sudo apparmor_parser -r /etc/apparmor.d/bwrap
# or lift the restriction for every program, until the next reboot
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

To keep the `sysctl` after a reboot, run
`echo kernel.apparmor_restrict_unprivileged_userns=0 | sudo tee /etc/sysctl.d/60-userns.conf`.

Otherwise, use `--backend claude` if the check shows Claude Code's sandbox working.

If you cannot fix the sandbox, for example without sudo on Ubuntu 24.04, `--allow-unsandboxed` starts Claude Code or
Codex without it. The agent's commands can then read and write anywhere you can. Codex still sends each command to its
automatic reviewer first, and runs it once the reviewer approves. The time limit stays, and so do the other settings:
no MCP servers, no hooks, and Claude Code's edit tools writing only in `src/`. The flag takes effect only where the
sandbox cannot start, a resumed run needs it again, and the run's README records that it ran without a sandbox.
`--allow-unsandboxed-codex` is the same flag under its old name. It also covers DyCraft's own Blender sandbox, but
only where neither `bubblewrap` nor Landlock can confine it (below). Where Landlock works, Blender stays confined even
with the flag.

On cluster nodes whose per-job `/tmp` or `$TMPDIR` Codex's `bwrap` cannot use ("Can't mkdir parents for …"), DyCraft
retries with `TMPDIR` in the run directory, the node's `/tmp`, and `~/.cache/dycraft/tmp`, then with a private `/tmp` in
a mount namespace of its own (`unshare`), and prints which one it used.

Blender's worker runs in a sandbox of its own. Where `bubblewrap` can create user namespaces, the worker gets a
filesystem of its own and no network. Where it cannot, as on Ubuntu 24.04 without an AppArmor profile for `bwrap`,
DyCraft confines the worker with Landlock (Linux 5.13 and later) instead, without root or a flag. The worker then has
the same limits on what it reads and writes and no network, but sees this machine's real paths, and runs a few percent
slower. Before Linux 6.12, it can also still signal your other processes. `python -m dycraft.install check` reports
which applies under "user namespaces", and fails only where the kernel has no Landlock either. There, make
`bubblewrap` work as above, or pass `--allow-unsandboxed`.
