"""Pure exact-pair allowance matching, shared with isolated animation checks."""
def _allowance_index(ir: dict) -> dict:
    """Group allowance records by unordered (a, b) pair.

    Returns `{frozenset({a, b}): [rec, ...]}`. Each record either has no
    `joint` key (blanket allowance) or has `joint` + `q_range` (scoped).
    A pair with any blanket record allows every sample pose for that
    pair; otherwise, a sample is allowed if any scoped record matches.
    """
    out: dict = {}
    for rec in ir.get("collision_allowances", ()):
        out.setdefault(frozenset((rec["a"], rec["b"])), []).append(rec)
    return out


def _allowance_matches(records, witnessing_pose_surface: dict) -> bool:
    """Return True if the witnessing pose is exempted by any record.

    `witnessing_pose_surface` is the sample's pose dict in surface units
    (degrees for revolute, meters for prismatic). Scoped records match
    when the joint's surface value lies in `[lo, hi]`.
    """
    for rec in records:
        if "joint" not in rec:
            return True  # blanket allowance — every sample exempted
        jn = rec["joint"]
        lo, hi = rec["q_range"]
        q = witnessing_pose_surface.get(jn)
        if q is None:
            continue
        if lo - 1e-9 <= float(q) <= hi + 1e-9:
            return True
    return False
