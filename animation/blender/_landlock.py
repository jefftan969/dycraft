"""Start a Blender worker confined by Landlock instead of a mount namespace, with host paths.

usage: python -I -B _landlock.py SPEC_JSON -- (the program is in the spec)

For a machine where bubblewrap cannot make user namespaces without sudo, such as stock Ubuntu 24.04, whose AppArmor
policy lets only /usr/bin/bwrap make them (`runtime.sandbox_mode`). There is no private filesystem, so the worker sees
real paths: `runtime.command` builds the same sandbox as a bubblewrap argv, and `runtime._host_command` turns its
binds into this spec. Each bind's target is mapped to its source, in the program's arguments, its working directory
and its environment's values, and the sources are what Landlock allows: reads and execution for a read-only bind,
everything for a writable one, and device use for a device node. In place of the namespaces:

- the network: Landlock refuses TCP (Landlock ABI 4 and later), and a seccomp filter refuses every socket that is not
  a Unix one, and io_uring, which can open sockets;
- other processes: Landlock keeps the worker from tracing them or reading their environment, and from ABI 6 on, from
  signalling them or reaching their abstract Unix sockets;
- `/tmp`: a directory of the worker's own, removed by a later start once its process is gone;
- `--die-with-parent` and `--new-session`: the parent-death signal and a session of its own.

`unconfined` mode (`--allow-unsandboxed`, only where Landlock is unavailable) does all of that but Landlock and the
filter.

Standard library only, and run isolated (`-I`), so it starts under whatever interpreter the daemon has.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import shutil
import signal
import socket
import struct
import sys
import tempfile

_SYS_CREATE, _SYS_ADD, _SYS_RESTRICT = 444, 445, 446      # the same on x86_64 and aarch64
_PR_SET_PDEATHSIG, _PR_SET_NO_NEW_PRIVS, _PR_SET_SECCOMP = 1, 38, 22
_FS = dict(EXECUTE=1 << 0, WRITE_FILE=1 << 1, READ_FILE=1 << 2, READ_DIR=1 << 3, REMOVE_DIR=1 << 4,
           REMOVE_FILE=1 << 5, MAKE_CHAR=1 << 6, MAKE_DIR=1 << 7, MAKE_REG=1 << 8, MAKE_SOCK=1 << 9,
           MAKE_FIFO=1 << 10, MAKE_BLOCK=1 << 11, MAKE_SYM=1 << 12, REFER=1 << 13, TRUNCATE=1 << 14,
           IOCTL_DEV=1 << 15)
_FILE_ONLY = _FS["EXECUTE"] | _FS["WRITE_FILE"] | _FS["READ_FILE"] | _FS["TRUNCATE"] | _FS["IOCTL_DEV"]
#: Seccomp's view of each architecture: its audit number, `socket`'s number, and `io_uring_setup`'s.
_ARCH = {"x86_64": (0xC000003E, 41, 425), "aarch64": (0xC00000B7, 198, 425)}

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def landlock_abi() -> int:
    """The kernel's Landlock ABI version, or 0 where Landlock is missing or switched off."""
    try:
        r = _libc.syscall(ctypes.c_long(_SYS_CREATE), None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    except (AttributeError, OSError):
        return 0
    return r if r > 0 else 0


def _fs_handled(abi: int) -> int:
    mask = (1 << 13) - 1
    if abi >= 2:
        mask |= _FS["REFER"]
    if abi >= 3:
        mask |= _FS["TRUNCATE"]
    if abi >= 5:
        mask |= _FS["IOCTL_DEV"]
    return mask


def apply_landlock(read, write, dev) -> int:
    """Confine this process and its children to `read` (read, list, execute), `write` (everything) and `dev`
    (read, write and use a device). Returns the ABI it used; raises OSError where it cannot."""
    abi = landlock_abi()
    if abi < 1:
        raise OSError(errno.ENOSYS, "Landlock is not available in this kernel")
    handled = _fs_handled(abi)
    net = 0b11 if abi >= 4 else 0                 # TCP bind and connect: none allowed
    scoped = 0b11 if abi >= 6 else 0              # abstract Unix sockets and signals: only within the domain
    attr = (struct.pack("QQQ", handled, net, scoped) if abi >= 6 else
            struct.pack("QQ", handled, net) if abi >= 4 else struct.pack("Q", handled))
    buf = ctypes.create_string_buffer(attr, len(attr))
    fd = _libc.syscall(ctypes.c_long(_SYS_CREATE), buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    readable = _FS["EXECUTE"] | _FS["READ_FILE"] | _FS["READ_DIR"]
    for paths, access in ((read, readable), (write, handled),
                          (dev, readable | _FS["WRITE_FILE"] | (_FS["IOCTL_DEV"] if abi >= 5 else 0))):
        for path in paths:
            try:
                pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue                          # nothing there to allow
            allowed = access & handled
            if not os.path.isdir(path):
                allowed &= _FILE_ONLY
            rule = ctypes.create_string_buffer(struct.pack("=Qi", allowed, pfd), 12)
            r = _libc.syscall(ctypes.c_long(_SYS_ADD), ctypes.c_int(fd), ctypes.c_int(1), rule, ctypes.c_uint32(0))
            err = ctypes.get_errno()
            os.close(pfd)
            if r < 0:
                raise OSError(err, f"landlock_add_rule {path}")
    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if _libc.syscall(ctypes.c_long(_SYS_RESTRICT), ctypes.c_int(fd), ctypes.c_uint32(0)) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(fd)
    return abi


def apply_socket_filter() -> None:
    """Refuse `socket()` for every family but AF_UNIX (EACCES), and `io_uring_setup` (ENOSYS)."""
    arch, nr_socket, nr_uring = _ARCH[platform.machine()]

    def stmt(code, k):
        return struct.pack("HBBI", code, 0, 0, k)

    def jump(code, k, jt, jf):
        return struct.pack("HBBI", code, jt, jf, k)

    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    allow, kill, err = 0x7FFF0000, 0x80000000, 0x00050000
    prog = [stmt(ld, 4), jump(jeq, arch, 1, 0), stmt(ret, kill),            # another architecture's calls
            stmt(ld, 0), jump(jge, 0x40000000, 0, 1), stmt(ret, kill),       # the x32 ABI's numbers
            jump(jeq, nr_uring, 0, 1), stmt(ret, err | errno.ENOSYS),
            jump(jeq, nr_socket, 0, 4),
            stmt(ld, 16), jump(jeq, socket.AF_UNIX, 0, 1), stmt(ret, allow),  # socket(): its family
            stmt(ret, err | errno.EACCES),
            stmt(ret, allow)]
    code = b"".join(prog)
    buf = ctypes.create_string_buffer(code, len(code))

    class _Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = _Fprog(len(prog), ctypes.cast(buf, ctypes.c_void_p))
    if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if _libc.prctl(_PR_SET_SECCOMP, 2, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def worker_tmp(root: str) -> str:
    """A `/tmp` of this process's own under `root`, named for its pid (which the program keeps across `exec`); the
    ones whose process is gone are removed first."""
    os.makedirs(root, mode=0o700, exist_ok=True)
    for name in os.listdir(root):
        pid = name.split("-", 1)[0]
        if pid.isdigit() and not _alive(int(pid)):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    return tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=root)


def mapper(pairs):
    """`path -> host path` for the binds' (target, source) pairs, the longest target first."""
    ordered = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    def one(value: str) -> str:
        for inside, host in ordered:
            if value == inside or value.startswith(inside.rstrip("/") + "/"):
                return host + value[len(inside):]
        return value

    def mapped(value: str) -> str:
        if ":" in value and all(part.startswith("/") for part in value.split(":") if part):
            return ":".join(one(part) for part in value.split(":"))
        return one(value) if value.startswith("/") else value
    return mapped


def main(argv: list[str]) -> int:
    spec = json.loads(argv[0])
    try:
        os.setsid()
    except OSError:
        pass
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if os.getppid() == 1:                          # the parent went before the signal was armed
        return 125
    tmp = worker_tmp(spec["tmp_root"])
    pairs = [tuple(p) for p in spec["map"]] + [("/tmp", tmp)]
    m = mapper(pairs)
    env = {k: m(v) for k, v in os.environ.items()}
    env.update({k: m(v) for k, v in spec.get("env", {}).items()})
    # Mesa's software renderer falls back to a file under XDG_RUNTIME_DIR for memory it cannot get from memfd.
    env.update(TMPDIR=tmp, XDG_RUNTIME_DIR=tmp)
    program = [m(a) for a in spec["argv"]]
    try:
        if spec["mode"] == "landlock":
            apply_landlock([*spec["read"]], [*spec["write"], tmp], [*spec["dev"]])
            apply_socket_filter()
        elif spec["mode"] != "unconfined":
            raise ValueError(f"unknown mode {spec['mode']!r}")
        os.chdir(m(spec["chdir"]))
    except (OSError, ValueError) as exc:
        print(f"[dycraft sandbox] cannot confine the worker: {exc}", file=sys.stderr, flush=True)
        return 125
    try:
        os.execvpe(program[0], program, env)
    except OSError as exc:
        print(f"[dycraft sandbox] cannot start {program[0]}: {exc}", file=sys.stderr, flush=True)
        return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
