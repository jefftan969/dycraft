"""Exception hierarchy for the DyCraft object SDK.

BuildError carries structured fields (code, summary, refs, evidence,
suggested_repairs). __str__ gives a human-readable rendering, which tests
match against. The agent harness
serializes to_dict() directly into tool results.
"""


class DycraftError(Exception):
    """Root of the dycraft exception tree. Do not raise directly."""


class _Structured(DycraftError):
    """Internal mixin: carries structured fields and a stable to_dict()."""

    def __init__(
        self,
        summary: str,
        *,
        refs=None,
        evidence=None,
        suggested_repairs=(),
    ):
        self.code = type(self).__name__
        self.summary = str(summary)
        self.refs = list(refs) if refs else []
        self.evidence = dict(evidence) if evidence else {}
        self.suggested_repairs = list(suggested_repairs)
        super().__init__(self._render())

    def _render(self) -> str:
        """Legacy human-readable rendering: the summary alone.

        Sufficient for existing string-based tests; the harness reads
        structured fields directly via to_dict().
        """
        return self.summary

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "summary": self.summary,
            "refs": [dict(r) for r in self.refs],
            "evidence": dict(self.evidence),
            "suggested_repairs": list(self.suggested_repairs),
        }


class SandboxViolation(_Structured):
    """Raised by the agent sandbox (AST allowlist or restricted-builtin
    misuse). The SDK itself does not raise this in v1."""


class SourceSyntaxError(SandboxViolation):
    """The source did not parse. Not a refused construct.

    `code` is what a tally counts, and the two mean opposite things to whoever
    reads one: `SandboxViolation` says "you used a construct the harness does
    not allow", this says "this is not valid Python".

    **A subclass, deliberately.** Both the worker and the pool route by
    `except SandboxViolation` into the report's `sandbox_violation` slot, and
    that slot is honest for a parse failure — the source was rejected before it
    ran, which is what the slot means. A sibling class would change that
    routing and file the error somewhere less diagnosable. So the class
    hierarchy carries the routing and `code` carries the distinction, as does
    `evidence["node_type"]`."""


class BuildError(_Structured):
    """Base class for all SDK-raised build-time errors.

    Constructor:
        BuildError(summary, *, refs=..., evidence=..., suggested_repairs=...)

    refs entries: {"kind": "object"|"part"|"joint"|"scene"|"instance"|"group"|
                   "anchor"|"assertion"|"room"|"catalog"|"slot",
                   "name": str|None, "id": int|None}.
    """


class SourceRuntimeError(BuildError):
    """The agent's source raised an ordinary Python exception.

    A NameError, TypeError, or ZeroDivisionError in `object.py` is a bug in the
    program, not an attempt to escape the sandbox. Reported as a
    `SandboxViolation`, it would tell the agent it had violated a policy when it
    had made a typo.
    """


class GeometryKernelError(BuildError):
    """A build123d / OCCT operation failed inside the agent's source.

    Distinct from `SourceRuntimeError` because the fix is different in kind:
    the program is valid Python and the call is well-formed, but the kernel
    could not produce the solid — a fillet radius larger than the adjacent
    face, a loft between incompatible profiles, a boolean with no overlap.
    The kernel's own message usually says what to try instead, so it is
    promoted into `suggested_repairs` rather than left in the traceback.
    """


class SourceTimeout(BuildError):
    """The agent's source ran past its wall-clock limit.

    Distinct from `SourceRuntimeError` because nothing raised — the program is
    still running, and the fix is a terminating loop or coarser geometry, not a
    corrected statement. Mirrors the CAD worker's `timeout` failure so both
    execution paths report the same thing.
    """


class InvalidPrimitive(BuildError):
    """Bad primitive arguments: non-positive dims, NaN/Inf, bad name, name collision."""


class InvalidJoint(BuildError):
    """Bad joint arguments: bad axis/limits/default/anchors, cross-object refs,
    name collision, self-attach, multi-parent."""


class DanglingReference(BuildError):
    """PartRef/JointRef used after finalize, or not produced by this Object."""


class RootNotDeclared(BuildError):
    """finalize() called with no declare_root."""


class NotConnectedToRoot(BuildError):
    """finalize() reached parts unreachable from the root."""


# ---------------------------------------------------------------------------
# Scene-level subclasses.
# ---------------------------------------------------------------------------


class InvalidScene(BuildError):
    """Two Scenes constructed per .py, or a scene-level structural rule
    violation that doesn't belong to a more specific subclass."""


class InvalidRoom(BuildError):
    """Bad room dimensions, double set_room, or set_room called after
    place/attach/create_group/assert_*."""


class RoomNotDeclared(BuildError):
    """finalize() called before set_room()."""


class NoInstances(BuildError):
    """finalize() called with no place/attach call."""


class InvalidInstance(BuildError):
    """Bad pose, unknown object handle, name collision, unknown parent
    entity, or cycle attempt."""


class InvalidGroup(BuildError):
    """Empty members, ambiguous root, member not transitively attached,
    member already in another group, second attach_group on a group, or
    cycle attempt during attach_group."""


class InvalidAnchor(BuildError):
    """Unknown anchor name for the entity type, or unknown part name in a
    part-scoped anchor path."""


class InvalidAssertion(BuildError):
    """Bad axis (zero norm / non-finite), min > max, name collision in
    assertion namespace, anchor refs not owned by this scene, or
    assert_collinear called with fewer than two anchors."""


# ---------------------------------------------------------------------------
# Scene-agent-layer catalog-validation subclasses.
# Raised by the scene-agent harness before any LLM call when the caller-
# supplied catalog is malformed.
# ---------------------------------------------------------------------------


class InvalidCatalog(BuildError):
    """Malformed catalog entry: bad schema tag, recomputed hash mismatch,
    key-vs-id mismatch, or non-dict value."""


class DuplicateObjectName(BuildError):
    """Two catalog entries share the same object name."""


# ---------------------------------------------------------------------------
# Staged meshes
# ---------------------------------------------------------------------------


class InvalidRig(BuildError):
    """A staged mesh is unusable, e.g. an OBJ with no triangles
    (`sdk/_deform_eval.load_obj`, which the exact mesh backend reads through).

    The name dates from the deformation rigs that first read meshes this way."""


