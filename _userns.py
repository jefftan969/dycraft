"""What to do, with root, when the kernel refuses bubblewrap its user namespaces under Ubuntu 24.04's AppArmor policy.

With `kernel.apparmor_restrict_unprivileged_userns=1`, AppArmor grants user namespaces only to programs whose profile
says `userns`. Some Ubuntu systems ship such a profile for /usr/bin/bwrap (`bwrap-userns-restrict`), and there
installing the system's bubblewrap is the fix; a minimal server image ships none, and the package alone then changes
nothing (a stock AWS Ubuntu 24.04.5 image, checked 11-09). So the advice depends on what this system has: the package
when /usr/bin/bwrap is missing, a profile when none grants it `userns`, and in any case the restriction itself.

Stdlib only, at the package root: the checks, the CLI backends and the Blender runtime all say these.
"""
from __future__ import annotations

from pathlib import Path

RESTRICT_SYSCTL = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
SYSTEM_BWRAP = Path("/usr/bin/bwrap")
APPARMOR_DIR = Path("/etc/apparmor.d")

#: A profile granting /usr/bin/bwrap user namespaces, written and loaded in one command.
PROFILE_COMMAND = (
    "printf 'abi <abi/4.0>,\\ninclude <tunables/global>\\nprofile bwrap /usr/bin/bwrap flags=(unconfined) {\\n"
    "  userns,\\n}\\n' | sudo tee /etc/apparmor.d/bwrap && sudo apparmor_parser -r /etc/apparmor.d/bwrap")

LIFT = ("or lift the restriction for every program: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` "
        "(to keep it after a reboot: `echo kernel.apparmor_restrict_unprivileged_userns=0 | sudo tee "
        "/etc/sysctl.d/60-userns.conf`)")


def restricted() -> bool:
    """Whether AppArmor restricts unprivileged user namespaces here."""
    try:
        return RESTRICT_SYSCTL.read_text().strip() == "1"
    except OSError:
        return False


def bwrap_profiled() -> bool:
    """Whether a profile in /etc/apparmor.d grants /usr/bin/bwrap user namespaces."""
    try:
        files = [p for p in APPARMOR_DIR.iterdir() if p.is_file() and not p.name.endswith(".bak")]
    except OSError:
        return False
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "/usr/bin/bwrap" in text and "userns" in text:
            return True
    return False


def fixes(package: str = "`sudo apt-get install bubblewrap socat`") -> list[str]:
    """The steps, with root, that let bubblewrap make user namespaces on this system; [] where AppArmor does not
    restrict them (the caller's own advice then applies)."""
    if not restricted():
        return []
    out = []
    if not SYSTEM_BWRAP.is_file():
        out.append(f"install the system's bubblewrap: {package}")
    if not bwrap_profiled():
        out.append("give /usr/bin/bwrap an AppArmor profile that allows user namespaces (this system has none, and "
                   f"the package alone does not add one): `{PROFILE_COMMAND}`")
    out.append(LIFT)
    return out
