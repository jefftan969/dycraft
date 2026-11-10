"""Which Python `object.py` may write.

The allowlist's job is to stop the source reaching the host, not to stop it
being Python. So a subscript assignment turns on *what* is being indexed rather
than on the syntax: a container the source bound itself is fair game, an
injected global or anything reached through an attribute or a call is not.

Rejecting every subscript assignment would protect nothing — no SDK type
defines `__setitem__`, and `d.update({k: v})`, `d.setdefault(k, v)` and
`xs.append(v)` are allowed anyway — and would cost a compile on the commonest
local idiom, a dict keyed by a side or part name:

    handle_parts[side] = (lower, hub)
    caps[(rname, x)] = cap
    steer_j[tag] = obj.attach_revolute(...)
 Attribute writes are untouched —
`obj._parts = ...` is the real thing being guarded and stays rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dycraft.agent.cad.sandbox import check_ast
from dycraft.sdk._errors import SandboxViolation


def _ok(src: str) -> None:
    check_ast(src)


def _blocked(src: str) -> SandboxViolation:
    with pytest.raises(SandboxViolation) as e:
        check_ast(src)
    return e.value


# ---------------------------------------------------------------------------
# Allowed: ordinary bookkeeping on the source's own containers
# ---------------------------------------------------------------------------

# The idioms above, in the same shape.
_HISTORICAL = (
    "handle_parts = {}\nfor side in ('l', 'r'):\n"
    "    handle_parts[side] = (side, side)",
    "caps = {}\nfor rname in ('a',):\n    for x in (1, 2):\n"
    "        caps[(rname, x)] = x",
    "brackets = {}\nfor side in ('l', 'r'):\n    brackets[side] = side",
    "steer_j = {}\nfor tag in ('l', 'r'):\n    steer_j[tag] = tag",
)


@pytest.mark.parametrize("src", _HISTORICAL)
def test_the_patterns_that_actually_failed_now_compile(src):
    _ok(src)


@pytest.mark.parametrize("src", [
    "d = {}\nd['a'] = 1",
    "xs = [0, 1]\nxs[0] = 9",
    "xs = [1, 2, 3]\nxs[0:2] = [9]",
    "d = {'a': 0}\nd['a'] += 1",
    "g = {}\ng['a'] = {}\ng['a']['b'] = 1",              # nested chain
    "d = {'a': 1}\ndel d['a']",
    "x = 1\ndel x",
    "def f(d, k):\n    d[k] = 1\n    return d",          # parameter
    "for d in [{}]:\n    d['a'] = 1",                    # loop variable
    "ys = [i for i in range(3)]\nys[0] = 5",             # comprehension result
    "rows = [[0] * 2 for _ in range(2)]\nrows[0][1] = 7",
])
def test_local_container_mutation_is_allowed(src):
    _ok(src)


@pytest.mark.parametrize("src", [
    "d = {}\nd.update({'a': 1})",
    "d = {}\nd.setdefault('a', 1)",
    "xs = []\nxs.append(1)",
])
def test_the_method_equivalents_still_work(src):
    """These are allowed, which is why refusing `d[k] = v` would protect
    nothing. A tightening that reached them would have to refuse that too, so
    they are pinned."""
    _ok(src)


# ---------------------------------------------------------------------------
# Refused: everything the rule guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src,why", [
    ("bd['x'] = 1", "injected global"),
    ("Object['x'] = 1", "injected global"),
    ("math['pi'] = 1", "injected global"),
    ("__builtins__['open'] = 1", "the builtins table"),
    ("obj.parts[0] = 1", "reached through an attribute"),
    ("f()[0] = 1", "reached through a call"),
    ("Object.parts['a'] = 1", "attribute chain on an injected global"),
])
def test_writing_through_anything_not_locally_bound_is_refused(src, why):
    v = _blocked(src)
    assert "non-local container" in v.summary, why


def test_attribute_assignment_is_untouched():
    """The rule that carries the real weight. `obj._parts = ...` mutating SDK
    internals is the thing being guarded, and this change must not have
    loosened it."""
    assert "attribute assignment" in _blocked("obj.x = 1").summary
    assert "attribute assignment" in _blocked("obj.x += 1").summary
    assert "attribute assignment" in _blocked("del obj.x").summary


def test_deleting_an_injected_name_is_refused():
    """`Delete` is newly allowed for local names; it must not become a way to
    unbind the SDK out from under a later check."""
    assert "injected name" in _blocked("del Object").summary
    assert "injected name" in _blocked("del bd").summary


def test_the_repair_message_names_the_real_problem():
    """The message describes what the rule tests: a *name*, not provenance.

    "Do not mutate items on SDK objects" would be wrong for a plain local dict,
    and "not a container this source created" would over-claim, since an alias
    of an injected global passes the rule.
    """
    v = _blocked("obj.parts[0] = 1")
    repair = " ".join(v.suggested_repairs)
    assert "name this source binds" in repair
    assert "SDK objects" not in repair


# ---------------------------------------------------------------------------
# Unchanged escape surfaces — a regression here matters more than the above
#
# `try`/`except`, `raise` and `yield` are allowed: they grant no capability,
# and the node allowlist is not the boundary — the agent that writes
# `object.py` also holds `Bash(*)`. What still belongs here is the surface that resolves a
# *name* to something outside the injected vocabulary, plus `with` (builder
# mode) and async (an event loop inside a pooled CAD worker). See
# `test_agent_sandbox.py` for the positive half.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    "import os",
    "from os import path",
    "x = (1).__class__",
    "x = ().__class__.__base__",
    "x = eval('1')",
    "x = getattr(obj, 'x')",
    "x = type(obj)",
    "x = open('/etc/passwd')",
    "with open('f') as f:\n    pass",
    "def f():\n    global z\n    z = 1",
    "async def f():\n    pass",
])
def test_the_real_guards_still_hold(src):
    _blocked(src)


def test_probe_mode_still_requires_a_single_expression():
    with pytest.raises(SandboxViolation):
        check_ast("x = 1", mode="eval")
    check_ast("part_shape(ir, 'a')", mode="eval")
