"""Read-only views over state the harness injects into sandboxed code.

Sandboxed sources and probe expressions receive references to objects the
harness owns and reuses: the cached object IR, a scene run's object handles and
catalog, the derived shape tables hanging off an ObjectHandle. Those objects
outlive one execution, so a mutation through one of them is not contained — it
silently changes what a later compile, probe or render sees.

The AST allowlist cannot prevent this. Mutation needs no distinctive syntax:

    objects.clear()            # no subscript, no attribute write, no banned node
    h = objects; h['x'] = 1    # an alias; the AST sees a local name
    def f(t): t['x'] = 1
    f(objects)                 # a parameter; the AST sees a local name

Without the views all three would succeed. Provenance is a property of objects,
not of names, so it is enforced here: the sandbox never receives a reference to
the mutable original, only a `freeze()` view of it. Aliases, parameters, values
laundered through an attribute or a call, and every mutating method all
converge on the same frozen view, so they close together rather than one at a
time.

Lives in the SDK rather than in `agent/cad/sandbox.py` because
`ObjectHandle` needs it too and the SDK cannot import the agent package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from ._errors import SandboxViolation


_MUTATORS = frozenset({
    "clear", "update", "setdefault", "pop", "popitem",
    "append", "extend", "insert", "remove", "sort", "reverse",
    "add", "discard", "difference_update", "intersection_update",
    "symmetric_difference_update",
})


def _frozen_violation(label: str, what: str) -> SandboxViolation:
    return SandboxViolation(
        f"{label} is read-only: {what}",
        refs=[],
        evidence={"node_type": "ReadOnlyView", "lineno": 0, "col": 0,
                  "src_line": "", "target": label, "operation": what},
        suggested_repairs=[
            f"`{label}` is shared harness state. Read it and build a local "
            f"list or dict for changes; nested values remain read-only.",
        ],
    )


class _ReadOnlyMapping(Mapping):
    """An immutable view of a mapping, frozen lazily on access.

    Lazily, because the object IR runs to tens of thousands of nodes and a
    probe should not pay a deep copy to read one number. Values are frozen on
    the way out, so a nested dict cannot be reached in its raw form.
    """

    __slots__ = ("_label", "_data")

    def __init__(self, data, label: str):
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key):
        data = object.__getattribute__(self, "_data")
        label = object.__getattribute__(self, "_label")
        return freeze(data[key], f"{label}[{key!r}]")

    def __iter__(self):
        return iter(object.__getattribute__(self, "_data"))

    def __len__(self):
        return len(object.__getattribute__(self, "_data"))

    def __contains__(self, key):
        return key in object.__getattribute__(self, "_data")

    def __eq__(self, other):
        return dict(self) == other

    def __repr__(self):
        return repr(object.__getattribute__(self, "_data"))

    def __getattr__(self, name):
        # Mapping supplies get/keys/items/values; anything that mutates is a
        # policy denial, and anything else simply does not exist here.
        if name in _MUTATORS:
            raise _frozen_violation(
                object.__getattribute__(self, "_label"), f"{name}()")
        raise AttributeError(name)

    def __setitem__(self, key, value):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), f"item assignment [{key!r}]")

    def __delitem__(self, key):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), f"del [{key!r}]")

    def __setattr__(self, name, value):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), f"attribute {name!r}")


class _ReadOnlySequence(Sequence):
    """An immutable view of a list, frozen lazily on access."""

    __slots__ = ("_label", "_data")

    def __init__(self, data, label: str):
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_data", data)

    def __getitem__(self, index):
        data = object.__getattribute__(self, "_data")
        label = object.__getattribute__(self, "_label")
        if isinstance(index, slice):
            return tuple(freeze(v, f"{label}[]") for v in data[index])
        return freeze(data[index], f"{label}[{index!r}]")

    def __len__(self):
        return len(object.__getattribute__(self, "_data"))

    def __eq__(self, other):
        return list(self) == other

    def __repr__(self):
        return repr(object.__getattribute__(self, "_data"))

    def __getattr__(self, name):
        if name in _MUTATORS:
            raise _frozen_violation(
                object.__getattribute__(self, "_label"), f"{name}()")
        raise AttributeError(name)

    def __setitem__(self, index, value):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), "item assignment")

    def __delitem__(self, index):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), "del")

    def __setattr__(self, name, value):
        raise _frozen_violation(
            object.__getattribute__(self, "_label"), f"attribute {name!r}")


def freeze(value, label: str):
    """An immutable view of `value` for the sandbox, or `value` if immutable.

    Applied at the overlay boundary, never to the harness's own data: the build
    cache, the scene payload, and everything that persists keep real dicts, so
    `json.dumps(ir)` and the verifier suite are untouched. Only the reference
    handed to sandboxed code is frozen.

    Strings and bytes are Sequences but already immutable, and wrapping them
    would break every `str` operation an authoring program does.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return value if not isinstance(value, bytearray) else bytes(value)
    if isinstance(value, _ReadOnlyMapping) or isinstance(value, _ReadOnlySequence):
        return value
    if isinstance(value, Mapping):
        return _ReadOnlyMapping(value, label)
    if isinstance(value, (list, tuple)):
        # Tuples are immutable but may hold mutable members, so still wrap.
        return _ReadOnlySequence(value, label)
    if isinstance(value, Set):
        return frozenset(value)
    return value


# ---------------------------------------------------------------------------
# Type predicates for code that receives frozen values
# ---------------------------------------------------------------------------
#
# **A frozen view is not a `dict` and not a `list`.** `_ReadOnlyMapping` and
# `_ReadOnlySequence` register as `collections.abc.Mapping` / `Sequence` and
# nothing narrower, so every `isinstance(x, dict)` and
# `isinstance(x, (tuple, list))` standing between the sandbox and the kernel
# reads False on a value the agent read out of `ir`, `catalog` or `objects`.
#
# That is not hypothetical. `_anchors.find_anchor_record` tested `dict` and so
# reported every declared semantic anchor as absent from every probe — which
# broke 14 of 23 scene probes on any scene whose instances were PLACED on a
# declared anchor, because the world-pose resolution goes through the same
# lookup. `UnknownAnchor` was the most frequent runtime error in a recorded
# scene corpus at 128 occurrences, and three scenes rewrote their placement
# graph around it. The check that would have caught it does not exist, because
# each site spells the rule itself.
#
# So the rule lives here, next to the views that make it necessary, and a
# validator that can receive an author-supplied value asks these rather than
# naming a concrete type. Reaching for `dict` or `(tuple, list)` in such a
# place is the bug, not a style preference.


def is_mapping(value) -> bool:
    """True for a mapping, frozen or not. Replaces `isinstance(x, dict)`."""
    return isinstance(value, Mapping)


def is_sequence(value) -> bool:
    """True for a list/tuple/frozen sequence. Replaces
    `isinstance(x, (tuple, list))`.

    Strings and bytes are Sequences and are deliberately excluded: every caller
    of this means "a vector of numbers", and `len('xyz') == 3` would pass a
    shape check that a string must fail.
    """
    return (isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray)))
