"""Cancellable consultant subprocess with process-group cleanup."""
import os
import signal
import subprocess
import time

from .. import _oom


def run_cancellable(argv, *, cancel, timeout=None, **kwargs):
    # Every consultant -- IK, simulation, delivery, verification, a Blender
    # session job -- passes through here, so it is where WORKER is applied.
    # Idempotent: an argv `runtime.command` already priced is left as it is.
    spawn = _oom.tiered(argv, _oom.WORKER)
    if cancel is None:
        return subprocess.run(spawn, timeout=timeout, **kwargs)
    capture = kwargs.pop("capture_output", False)
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check = kwargs.pop("check", False)
    started = time.monotonic()
    process = subprocess.Popen(spawn, start_new_session=True, **kwargs)
    try:
        while True:
            if cancel.is_set():
                raise RuntimeError("job cancelled")
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(.25, remaining) if remaining is not None else .25)
                result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
                if check:
                    result.check_returncode()
                return result
            except subprocess.TimeoutExpired:
                continue
    except BaseException as error:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                error.output, error.stderr = stdout, stderr
        raise
