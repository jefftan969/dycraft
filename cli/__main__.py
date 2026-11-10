"""Make `python -m dycraft.cli` invoke cli.main()."""
import sys

from .. import _oom

if __name__ == "__main__":
    # Before anything loads GL: the libraries `install tools` added are found only by a process started with them.
    from .._home import reexec_for_loader
    reexec_for_loader()
    # First, before `cli` is imported or anything starts a thread: this arm's
    # own OOMPolicy=continue scope and the SESSION tier, for every layer and
    # every launcher. A dispatched sub-agent inherits both and skips the
    # re-exec. See `_oom`.
    _oom.arm_entry()
    try:
        from .cli import main
    except ModuleNotFoundError as exc:
        # The commonest wrong setup: a Python that is not DyCraft's environment. Said so, not as a traceback.
        print(f"FATAL: {sys.executable} cannot import `{exc.name}`, which DyCraft needs: this Python is not "
              "DyCraft's environment. Activate the environment you installed DyCraft into (docs/install.md), "
              "or check one with `<its python> -m dycraft.install check`.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())
