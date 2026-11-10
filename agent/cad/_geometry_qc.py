"""Triangle-mesh geometry: weld, repair, QC, shading, texture coordinates, OBJ.

Everything ``Object.add_part`` runs on a registered Shape's tessellation, in the
order ``_solid_compile.compile_entry`` uses it: weld, drop degenerate and
unreferenced faces, judge shells, then write the OBJ with the exact per-corner
surface normals and UVs `_solid_compile` supplies. The bytes this produces are what gets staged as the ``solid`` part's
render asset, so the QC numbers reported alongside them describe exactly the
geometry that was staged.

**Pure stdlib, deliberately.** There is nothing here to import but
``math`` and ``pathlib``, so ``_solid_compile`` — which is loaded inside the
frozen build123d interpreter, where the dycraft package is not installed — can
depend on it unconditionally instead of guarding every call against a failed
load. QC that can vanish because an import failed is QC you cannot rely on when
reading a result payload.

Cost is linear in the mesh apart from two sorts — the unreferenced-vertex sweep
and the UV dedup — which are there for reproducible output ordering, not for
the algorithm. Nothing here is quadratic in vertices or faces; a solid that
blows the triangle budget is coarsened upstream, never quietly ground through
an O(n^2) pass here.

Subtleties worth knowing before editing:

* Welding keys off the *written* form of a coordinate (``_fmt_coord``), not an
  arithmetic quantization, so two vertices weld if and only if they serialize
  identically — and is bounded to one source solid at a time, because a weld
  that spans solids fuses two exactly-tangent bodies into one pinched surface
  (see ``_weld_vertices``).
* Orientation is judged **per shell**, never on the mesh's total signed volume:
  that sum runs over shells sharing no edges, so one large inverted shell beside
  a small correct one makes it negative and a sum-based test flips the wrong one.
* Shells enclosing no volume — flat fans, collapsed rings, zero-thickness fins —
  are excluded from that judgment entirely, because the sign of a ~0 volume is
  rounding noise rather than an orientation (see ``_classify_shells``).
* Texture coordinates ride through as one ``(u, v)`` per triangle CORNER, never
  per vertex, and that choice is what makes them survive: welding and dropping
  unused vertices RENUMBER vertices and so cannot touch a per-corner stream,
  while dropping triangles and flipping the winding act on triangles and must
  be mirrored onto it or the UVs slide onto the wrong corners. Both mirrors
  live in ``_repair_mesh``; adding a step that reorders or removes triangles
  means adding a third.

**Two questions, not one.** ``qc.bounds_volume`` asks whether the surface
encloses anything — every edge used an even number of times — and that is the
property every consumer in the tree actually depends on, from the divergence
theorem here to the scene layer's tangency erosion. ``qc.is_manifold`` asks the
stronger question of whether a half-edge structure can be built over it, which
nothing DyCraft does today requires: two solids meeting along a line enclose
their volumes exactly and share an edge four ways. Conflating the two reported a
defect for sound geometry, so they are separate fields and only the first is
about correctness.

Three defects are detected and deliberately NOT repaired, none having an
unambiguous fix: local winding inconsistency within a shell (reported as
``qc.winding_consistent: false``), disagreement between shells where only some
are inside-out, and shells enclosing no volume (counted as
``qc.n_degenerate_shells``). The last two are also explained in the repair notes.

Also deliberately absent: hole filling. ``validate_shape`` admits only shapes
containing a Solid, so a staged part is always built from closed bodies and an
open boundary means a boolean left a hole rather than that a rim was intended.
Inventing a cap would hide which operation failed, and the surface it invented
would not be the one the author meant. Open boundaries are *reported*
(``n_boundary_edges``, and ``bounds_volume: false``), never sealed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple


_OBJ_COORD_DECIMALS = 6


# Triangles with area below this (m^2) carry no surface and are dropped. A
# square micron is 1e-12 m^2, so 1e-16 is one ten-thousandth of one — far below
# anything a real part needs, and well under the 1 mm minimum part dimension
# squared by ten orders of magnitude.
_MIN_TRIANGLE_AREA = 1e-16

def _fmt_coord(value: float) -> str:
    """The single definition of a vertex coordinate's written form.

    Used for BOTH the OBJ ``v`` lines and the vertex-weld key, which is what
    makes welding lossless with respect to the file we emit: two vertices weld
    if and only if they serialize to identical text. Quantizing by arithmetic
    instead (``round(v / 1e-6)``) is NOT equivalent — the two disagree on
    floats sitting near a half-way boundary, e.g. ``-0.0049965`` and
    ``-0.004996500000000001`` both format to ``-0.004997`` but take different
    arithmetic keys, which would leave two coincident written vertices unwelded.
    """
    s = f"{float(value):.{_OBJ_COORD_DECIMALS}f}"
    # "-0.000000" and "0.000000" name the same point; normalize so a signed
    # zero (or a tiny negative that rounds to zero) cannot block a weld or
    # appear as a distinct vertex in the OBJ.
    if s[0] == "-" and s.lstrip("-0.") == "":
        s = s[1:]
    return s


def _fmt_uv(value: float) -> str:
    """The written form of a texture coordinate.

    Separate from `_fmt_coord` because the two answer to different things: a
    vertex coordinate's precision is fixed by the weld it also keys, while a UV
    is a tile count that may legitimately run to hundreds on a large surface at
    a small tile size. Six decimals resolve about a micron of a one-meter tile,
    which is far below anything a texel can show.

    Shares `_fmt_coord`'s signed-zero rule, and for the same reason: two
    coordinates that name the same point must produce the same text, or the
    deduplication in `_write_obj` emits both.
    """
    s = f"{float(value):.{_OBJ_COORD_DECIMALS}f}"
    if s[0] == "-" and s.lstrip("-0.") == "":
        s = s[1:]
    return s


def _triangle_area(verts, a: int, b: int, c: int) -> float:
    ax, ay, az = verts[a]
    bx, by, bz = verts[b]
    cx, cy, cz = verts[c]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _weld_vertices(verts, faces, groups=None):
    """Merge vertices whose written OBJ coordinates are identical. Lossless.

    CadQuery tessellates each surface patch independently, so a smooth solid
    comes back as N disjoint patches that share no vertex indices even though
    their corners sit at identical coordinates. Un-welded, such a mesh reports
    as dozens of disconnected shells riddled with boundary edges, which makes
    every other QC number meaningless.

    The key is the `_fmt_coord` triple — exactly the text `_write_obj` emits —
    so merging two vertices can never change the geometry the file describes.

    ``groups`` bounds the merge. It is the sequence of cumulative vertex end
    offsets `_solid_compile.tessellate` returns, one per source solid, and two
    vertices weld only if they fall in the same one. Without it a weld spans
    solids, and that is not hypothetical: two solids authored exactly tangent
    share a line of coordinates, so welding fuses their surfaces into a single
    shell pinched along that line — an edge used by four triangles, reported as
    an open or non-manifold mesh. The defect is *manufactured here*, out of two
    bodies that were each perfectly clean; OCCT hands over no such edge.
    Healthy geometry has no weld across a solid boundary, so bounding the merge
    changes nothing except the tangency case it exists to prevent.

    Omitting ``groups`` gives an unbounded weld, which is what every caller
    outside the compile path wants: a mesh that never came from
    `tessellate` has no solid structure to respect.
    """
    lookup: dict = {}
    remap = [0] * len(verts)
    out_verts: list = []
    bounds = list(groups or ())
    gi = 0
    for i, (x, y, z) in enumerate(verts):
        # `bounds` is ascending and `i` only advances, so this walk is linear
        # over the whole loop rather than per vertex.
        while gi < len(bounds) and i >= bounds[gi]:
            gi += 1
        key = (gi, _fmt_coord(x), _fmt_coord(y), _fmt_coord(z))
        j = lookup.get(key)
        if j is None:
            j = len(out_verts)
            lookup[key] = j
            out_verts.append((x, y, z))
        remap[i] = j
    out_faces = [(remap[a], remap[b], remap[c]) for (a, b, c) in faces]
    return out_verts, out_faces, len(verts) - len(out_verts)


def _drop_bad_faces(verts, faces, corners=None):
    """Drop triangles that carry no surface, and exact duplicates.

    Returns ``(kept, kept_corners, n_degenerate, n_duplicate)``.

    `corners` is an optional list parallel to `faces` carrying per-triangle
    payload (see `_repair_mesh`); surviving entries are returned in lockstep, so
    a dropped triangle cannot leave its shading data behind to be attached to
    somebody else's triangle. Carrying the payload through rather than returning
    a boolean mask for the caller to apply keeps the drop and the payload in one
    place, which is what makes it impossible to get subtly out of step.
    """
    seen: set = set()
    kept: list = []
    kept_corners: list = []
    n_degenerate = 0
    n_duplicate = 0
    for i, (a, b, c) in enumerate(faces):
        if a == b or b == c or a == c:
            n_degenerate += 1
            continue
        if _triangle_area(verts, a, b, c) < _MIN_TRIANGLE_AREA:
            n_degenerate += 1
            continue
        key = tuple(sorted((a, b, c)))
        if key in seen:
            n_duplicate += 1
            continue
        seen.add(key)
        kept.append((a, b, c))
        if corners is not None:
            kept_corners.append(corners[i])
    return (kept, (kept_corners if corners is not None else None),
            n_degenerate, n_duplicate)


def _drop_unreferenced(verts, faces):
    """Remove vertices no surviving triangle uses."""
    used = sorted({i for f in faces for i in f})
    if len(used) == len(verts):
        return verts, faces, 0
    remap = {old: new for new, old in enumerate(used)}
    out_verts = [verts[i] for i in used]
    out_faces = [(remap[a], remap[b], remap[c]) for (a, b, c) in faces]
    return out_verts, out_faces, len(verts) - len(out_verts)


def _edge_stats(faces):
    """Count boundary edges (used once), non-manifold edges (used > twice), and
    edges used an ODD number of times.

    The third is the one that decides whether the surface encloses anything. A
    surface bounds a volume exactly when every edge is used an even number of
    times — each triangle contributes its edges in a consistent direction, so an
    odd count means the surface has a rim there rather than continuing. Being a
    2-manifold (every edge used exactly twice) is strictly stronger, and is a
    statement about local structure rather than about enclosure: two solids
    meeting along a line enclose their volumes perfectly well while sharing an
    edge four ways.
    """
    counts: dict = {}
    for (a, b, c) in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    boundary = 0
    nonmanifold = 0
    odd = 0
    for n in counts.values():
        if n == 1:
            boundary += 1
        elif n > 2:
            nonmanifold += 1
        if n % 2:
            odd += 1
    return boundary, nonmanifold, odd


def _winding_consistent(faces) -> bool:
    """True when every edge is traversed as often one way as the other.

    On a consistently-wound surface the triangles on either side of an edge
    traverse it in opposite directions, so the two directed counts match. Where
    they do not — (u, v) used twice and (v, u) not at all — two triangles
    sharing that edge agree about its direction and therefore disagree about
    which side is outward: LOCAL winding inconsistency, which `_repair_mesh`
    detects but does not fix.

    **Counting, not a uniqueness check.** The obvious test — "no directed edge
    appears twice" — is wrong on any surface that touches itself. Two solids
    meeting along a tangent line share one edge four ways, and each side
    contributes one traversal in each direction, so every directed edge appears
    twice while nothing is wound backwards. That test would call every
    self-touching part inconsistent and veto the inside-out repair below on
    exactly the meshes that might need it.
    """
    counts: dict = {}
    for (a, b, c) in faces:
        for edge in ((a, b), (b, c), (c, a)):
            counts[edge] = counts.get(edge, 0) + 1
    for (u, v), n in counts.items():
        if counts.get((v, u), 0) != n:
            return False
    return True


def _signed_volume(verts, faces) -> float:
    """Enclosed volume via the divergence theorem.

    Only meaningful for a closed, consistently-wound surface; negative then
    means the whole mesh is inside-out. For a mesh with local winding
    inconsistency the per-triangle contributions partially cancel and the
    result is not interpretable.
    """
    total = 0.0
    for (a, b, c) in faces:
        x1, y1, z1 = verts[a]
        x2, y2, z2 = verts[b]
        x3, y3, z3 = verts[c]
        total += (x1 * (y2 * z3 - y3 * z2)
                  - y1 * (x2 * z3 - x3 * z2)
                  + z1 * (x2 * y3 - x3 * y2)) / 6.0
    return total


def _shell_partition(verts, faces):
    """Group faces into connected pieces (union-find over shared vertices).

    Orientation is only comparable *within* a shell: two shells share no edge,
    so `_winding_consistent` cannot relate them and a summed signed volume says
    nothing about either one. Callers that reason about orientation must look
    at shells individually — see `_repair_mesh`.
    """
    parent = list(range(len(verts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b, c) in faces:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    groups: dict = {}
    for f in faces:
        groups.setdefault(find(f[0]), []).append(f)
    return list(groups.values())


def _shell_extent(verts, shell) -> float:
    """Largest bounding-box span over the vertices a shell's faces reference."""
    idx = {i for f in shell for i in f}
    if not idx:
        return 0.0
    spans = []
    for axis in (0, 1, 2):
        vals = [verts[i][axis] for i in idx]
        spans.append(max(vals) - min(vals))
    return max(spans)


# A shell counts as enclosing "no volume" when its signed volume is negligible
# against its own size. The test has to be scale-relative: a 1 cm part and a 1 m
# part are six orders of magnitude apart in volume, so no absolute cutoff works
# for both.
_DEGENERATE_VOLUME_REL = 1e-9


def _shell_bbox(verts, shell):
    """Axis-aligned bounds of the vertices a shell's faces reference."""
    idx = {i for f in shell for i in f}
    pts = [verts[i] for i in idx]
    return ([min(p[a] for p in pts) for a in (0, 1, 2)],
            [max(p[a] for p in pts) for a in (0, 1, 2)])


def _bbox_within(inner, outer) -> bool:
    """Is `inner`'s box inside `outer`'s? Non-strict, so touching faces pass."""
    return all(inner[0][a] >= outer[0][a] and inner[1][a] <= outer[1][a]
               for a in (0, 1, 2))


def _classify_shells(verts, faces):
    """Sort connected shells into (degenerate, inverted, outward) counts.

    A flat or collapsed shell encloses nothing, so the SIGN of its signed volume
    is floating-point noise rather than an orientation. Reading that noise as an
    orientation misreports a real defect: zero-thickness teeth whose volumes
    round to negative noise would be reported as "inside-out" when the actual
    problem is that they have no thickness at all. Degenerate shells are therefore counted
    separately and never treated as evidence about winding.

    **A cavity wall is negative on purpose.** A sealed void inside a solid is
    represented by a shell wound the other way — that is what subtracts its
    volume from the body's — so a negative shell sitting inside a positive one
    is correct geometry, not an inverted one. Counting it as inverted would tell
    the author of an ordinary hollow box to fix the winding of a part whose
    volume is exactly right.
    Containment is judged on bounding boxes, which is sufficient here: a shell
    that is not inside another's box cannot be its cavity, and a shell that is
    can be nothing else, because two solids of one part do not interpenetrate.

    Returns ``(n_degenerate, n_inverted, n_outward, n_shells)``; cavities count
    as outward, being correctly oriented for what they are.
    """
    n_degenerate = n_inverted = n_outward = 0
    shells = _shell_partition(verts, faces)
    measured = []
    for shell in shells:
        volume = _signed_volume(verts, shell)
        extent = _shell_extent(verts, shell)
        degenerate = (extent <= 0.0
                      or abs(volume) <= _DEGENERATE_VOLUME_REL * extent ** 3)
        measured.append((shell, volume, degenerate))
    positive = [_shell_bbox(verts, sh) for sh, vol, deg in measured
                if not deg and vol > 0.0]
    for shell, volume, degenerate in measured:
        # extent == 0 is a fully collapsed shell — degenerate outright, and the
        # threshold above would be 0 and admit nothing.
        if degenerate:
            n_degenerate += 1
        elif volume < 0.0 and not any(
                _bbox_within(_shell_bbox(verts, shell), box) for box in positive):
            n_inverted += 1
        else:
            n_outward += 1
    return n_degenerate, n_inverted, n_outward, len(shells)


def _mesh_qc(verts, faces) -> dict:
    """Measure the mesh. Cheap, total, and never raises."""
    boundary, nonmanifold, odd = _edge_stats(faces)
    winding_ok = _winding_consistent(faces)
    volume = _signed_volume(verts, faces)
    n_degenerate, _, _, n_shells = _classify_shells(verts, faces)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return {
        "n_vertices": len(verts),
        "n_faces": len(faces),
        # **The property that matters.** Every edge used an even number of
        # times, so the surface separates an inside from an outside. This is
        # what volume, point-in-solid, signed distance, and the scene layer's
        # tangency erosion all actually require, and it is the honest reading of
        # "the author built a closed body". `n_boundary_edges > 0` is its
        # failure mode: a boolean left a hole.
        "bounds_volume": (odd == 0),
        # Strictly stronger, and deliberately NOT a requirement anywhere: every
        # edge used exactly twice, so the surface is locally a disk and a
        # half-edge structure can be built over it. Nothing DyCraft does today
        # needs it — every consumer tests the surface rather than walking it —
        # so it is reported for the algorithms that would (mesh booleans, UV
        # unwrapping, decimation, tetrahedralisation) and as an import-quality
        # signal for third-party assets, never as a defect to repair.
        "is_manifold": (boundary == 0 and nonmanifold == 0),
        # True when every edge is traversed as often one way as the other. False
        # means some faces are locally reversed relative to their neighbors;
        # that is REPORTED here but deliberately not repaired (see
        # `_repair_mesh`). Self-touching surfaces read true, which is the point
        # of counting rather than checking for a repeated directed edge.
        "winding_consistent": winding_ok,
        # Interpretable only when `bounds_volume` is true; reported regardless
        # because a near-closed shell's magnitude is still a useful
        # order-of-magnitude check.
        "volume_m3": round(volume, 12),
        "n_shells": n_shells,
        # Shells whose enclosed volume is negligible against their own size:
        # flat fans, collapsed rings, zero-thickness fins. They render as
        # nothing solid, and their orientation is meaningless — see
        # `_classify_shells`.
        "n_degenerate_shells": n_degenerate,
        "n_boundary_edges": boundary,
        "n_nonmanifold_edges": nonmanifold,
        "bbox_min": [round(min(xs), 6), round(min(ys), 6), round(min(zs), 6)],
        "bbox_max": [round(max(xs), 6), round(max(ys), 6), round(max(zs), 6)],
    }


#: QC properties that are *defects*, each with what it means for the author.
#:
#: Domain-aware on purpose. `is_manifold` and a multi-shell count are NOT here:
#: `_mesh_qc` documents `is_manifold` as reported-never-required, and a closed
#: carcass legitimately has two shells. Treating either as a defect would fail
#: correct geometry.
#:
#: What is left is the set where the mesh contradicts something the author
#: asked for: a surface that does not separate inside from outside, faces
#: locally reversed against their neighbors, or a shell enclosing nothing.
_QC_DEFECTS = (
    ("bounds_volume", False,
     "the staged mesh has edges used an odd number of times and does not close; "
     "inspect for holes or cuts through intended walls"),
    ("winding_consistent", False,
     "some mesh faces have inconsistent winding relative to their neighbors; "
     "automatic repair left them unchanged because the correct orientation is "
     "ambiguous"),
)


def qc_findings(qc) -> list:
    """The defects in a `_mesh_qc` table, as author-facing sentences.

    Empty when the mesh is sound, which is what lets a caller distinguish "QC
    ran and found nothing" from "QC ran and found something"; `qc_status` says
    only whether QC itself ran.
    """
    if not isinstance(qc, dict):
        return []
    out = []
    for key, bad_value, message in _QC_DEFECTS:
        if key in qc and qc[key] is bad_value:
            out.append(f"{message} (`qc.{key}` is {bad_value!r})")
    n_degenerate = qc.get("n_degenerate_shells") or 0
    if n_degenerate:
        out.append(
            f"{n_degenerate} shell(s) of the staged mesh enclose no volume — "
            f"a flat fan, a collapsed ring, or a zero-thickness fin, which "
            f"render as nothing solid (`qc.n_degenerate_shells`)")
    boundary = qc.get("n_boundary_edges") or 0
    if boundary and qc.get("bounds_volume") is not False:
        out.append(
            f"the staged mesh has {boundary} boundary edge(s) while still "
            f"closing, which means an open rim rather than a hole "
            f"(`qc.n_boundary_edges`)")
    return out


class RepairedMesh(NamedTuple):
    """What `_repair_mesh` produced.

    A named tuple rather than a bare 4-tuple because `corners` and `flipped`
    exist only to carry *shading* data across the repair, and a caller that does
    not care about shading should not have to know their position; a field
    added later does not shift the others.
    """

    verts: list
    faces: list
    actions: list
    notes: list
    #: Per surviving triangle, the PRE-WELD vertex indices of its three corners,
    #: in the surviving corner order — or None when tracking was not requested.
    #: This is the join key for per-corner shading data: the exact normal and
    #: the texture coordinate are both per tessellation node, and welding merges
    #: nodes that share a position but not a normal (a box corner is one
    #: position and three normals), so after the weld only the original index
    #: can say which node a corner was.
    corners: list | None
    #: True when the whole mesh was re-wound. A flip reverses the surface's
    #: outward direction, so a caller carrying normals MUST negate them; leaving
    #: them alone points every normal into the material and the part renders
    #: black.
    flipped: bool


def _repair_mesh(verts, faces, weld_groups=None, *, track_corners=False):
    """Apply geometrically lossless cleanups. Returns a `RepairedMesh`.

    `actions` are repairs applied, `notes` explain a repair that was
    deliberately declined. Pass ``track_corners=True`` to also receive the
    per-corner join key and the flip flag described on `RepairedMesh`.

    Scope, precisely:

    * welding coincident vertices, dropping zero-area and duplicate triangles,
      and dropping unused vertices — all preserve the written geometry exactly;
    * flipping a volume-enclosing, consistently-wound mesh in which **no shell that
      encloses volume is correctly oriented** — one whole-mesh flip, applied to
      all triangles.

    The per-shell condition matters: flipping is driven by the sign of an
    enclosed volume, and a mesh's total signed volume is the *sum* over its
    shells. If some shells are inverted and others are not, that sum's sign is
    unrelated to either group, and flipping everything would correct the bad
    shells while breaking the good ones. Such a mesh is left alone and the
    disagreement is reported through `notes`.

    Shells that enclose no volume are excluded from that vote entirely — their
    sign is rounding noise, not an orientation (see `_classify_shells`). They
    neither justify a flip nor block one, and are reported through `notes` as
    the distinct defect they are: geometry with no thickness.

    NOT in scope: local winding inconsistency, where triangles *within* a shell
    are reversed relative to their neighbors. Repairing that needs adjacency
    propagation with a seed-orientation choice, which is neither trivial nor
    unambiguous. `_mesh_qc` reports it as ``winding_consistent: false`` and the
    mesh is left untouched — including the flip, whose volume test is
    meaningless once per-triangle contributions cancel.

    Nothing here invents or removes surface, so this can run unconditionally.
    """
    actions: list = []
    notes: list = []
    flipped = False
    # The join key starts as the input triangles, whose indices ARE the
    # pre-weld node indices, and is then carried through every step that
    # reorders or drops a triangle. Welding and dropping unused vertices
    # renumber vertices without touching triangle order, so neither disturbs it.
    corners = [tuple(f) for f in faces] if track_corners else None

    def note(n: int, singular: str, plural: str) -> None:
        if n:
            actions.append(f"{n} {singular if n == 1 else plural}")

    # `corners` is the join key, one triple of PRE-WELD node indices per
    # triangle. Only two of the four steps below can disturb it, and which two
    # is the whole reason the shading streams are not indexed by vertex:
    #
    #   * welding and dropping unused vertices RENUMBER vertices. A per-vertex
    #     attribute array could not survive that (a welded box corner is one
    #     position with three different normals); the join key is not indexed
    #     by vertex at all, so both are no-ops for it.
    #   * dropping triangles and flipping the winding act on triangles, so both
    #     must be mirrored here or the attributes slide onto the wrong corners.
    verts, faces, n_merged = _weld_vertices(verts, faces, weld_groups)
    note(n_merged, "welded duplicate vertex", "welded duplicate vertices")

    faces, corners, n_degenerate, n_duplicate = _drop_bad_faces(
        verts, faces, corners)
    note(n_degenerate, "dropped zero-area triangle", "dropped zero-area triangles")
    note(n_duplicate, "dropped duplicate triangle", "dropped duplicate triangles")

    verts, faces, n_unused = _drop_unreferenced(verts, faces)
    note(n_unused, "dropped unused vertex", "dropped unused vertices")

    # A flip is only well-defined when the surface ENCLOSES something (an open
    # shell has no inside to be on the wrong side of) and is consistently wound
    # (otherwise the signed volume is not interpretable, so its sign cannot
    # justify anything). Local winding inconsistency is reported by QC, not
    # repaired.
    #
    # The gate is `bounds_volume`, NOT manifoldness. Enclosure is what makes the
    # divergence theorem apply, and a surface that touches itself encloses its
    # volume exactly as well as one that does not. Requiring every edge to have
    # exactly two faces would exclude self-touching solids from this repair and
    # from both notes below, so a pinched part with zero-thickness fins would be
    # told nothing about them.
    _boundary, _nonmanifold, odd = _edge_stats(faces)
    if odd == 0 and _winding_consistent(faces):
        n_flat, n_inverted, n_outward, n_shells = _classify_shells(verts, faces)
        # Degenerate shells are indifferent to a flip — they enclose nothing, so
        # reversing them changes no geometry, and their noise-signed volume must
        # not veto a repair the real shells unambiguously need.
        if n_inverted and not n_outward:
            faces = [(a, c, b) for (a, b, c) in faces]
            # The join key moves with its triangle. Reversing the winding and
            # leaving the attributes in place would mirror the texture on
            # exactly the parts this repair fires for, which nothing downstream
            # could tell from a badly authored material — and would leave every
            # normal pointing into the material. `flipped` is what tells the
            # caller to negate them.
            if corners is not None:
                corners = [(a, c, b) for (a, b, c) in corners]
            flipped = True
            actions.append(
                f"flipped inside-out orientation ({len(faces)} triangles"
                f"{'' if n_shells == 1 else f', all {n_shells} shells'})")
        elif n_inverted:
            n_oriented = n_inverted + n_outward
            notes.append(
                f"{n_inverted} of {n_oriented} volume-enclosing shells are "
                f"inside-out. Automatic repair left orientation unchanged "
                f"to preserve correct shells; fix only the affected shells' "
                f"winding in the modeling code.")
        if n_flat:
            notes.append(
                f"{n_flat} of {n_shells} shells enclose no volume: their "
                f"triangles are flat or collapsed. Give them real thickness "
                f"by offsetting the profile perpendicular to its plane or "
                f"sweeping it; winding changes cannot repair this.")

    return RepairedMesh(verts, faces, actions, notes, corners, flipped)


# ---------------------------------------------------------------------------
# Failure diagnosis
#
# Modeling-kernel errors are terse and name no cause. Each entry maps a
# substring of the raw message to a plain-language statement of what failed.
# `compile.py` prepends the match to a kernel failure's suggested repairs, so a
# caller sees what went wrong rather than only OCCT's own wording.
#
# These are diagnoses, not prescriptions: they say what the kernel rejected and
# stop there. Choosing the construction that avoids it is the author's call.
# ---------------------------------------------------------------------------

_FAILURE_HINTS = (
    (
        ("brep_api: command not done", "stdfail_notdone"),
        "The CAD operation failed without identifying a cause. Inspect the "
        "failing call and its inputs before retrying:\n"
        "  * `revolve`: check whether the profile crosses the axis or the "
        "axis is not in the profile's plane. Keep the profile on one side "
        "of the axis; touching it is allowed. For an XY face intended to "
        "revolve about Z, use `revolve(Plane.XZ * face, axis=Axis.Z)`.\n"
        "  * Profiles: remove any duplicate consecutive point, including "
        "equal coordinates written as different expressions.\n"
        "  * `loft` between periodic-spline sections of differing shape: "
        "seam correspondence can fail despite valid sections; `ruled=True` "
        "may not help. Try an open spline plus a closing `Line` to give "
        "each section matching seam vertices. Do not abandon periodic "
        "sections generally: repeated or scaled sections can loft correctly.\n"
        "  * Long or uneven loft stacks: inspect for failure, twist, or "
        "bulging between stations even if the solid is valid. Try shorter "
        "loft spans followed by a fuse; matching section parameterization "
        "(e.g. four spline quadrants sampled alike); or more even arc-length "
        "spacing along the centerline. Choose the change for the defect.\n"
        "  * `.shell()` on tapered/lofted solids: try subtracting an inner "
        "body from an outer one.\n"
        "  * `.fillet()`: check adjacent face widths, interacting blends, "
        "and split edges. Isolate failing edges and test several radii; "
        "smaller is not always feasible.\n"
        "If these checks do not explain the failure, isolate the operation "
        "in a CAD scratch script or rebuild the feature more simply.",
    ),
    (
        ("failed creating a fillet", "failed creating a chamfer"),
        "Fillet feasibility is not monotone in radius on some topology: "
        "a larger radius can succeed where smaller ones fail. The kernel's "
        "\"try a smaller value\" is not a guarantee; `max_fillet()` "
        "binary-searches and presumes monotonicity. Test the chosen radius.\n"
        "  * Check adjacent face widths, including narrow steps left by "
        "booleans. For opposing quarter-round blends on one face, "
        "`r1 + r2 <= face width` is a useful clearance check; other angles "
        "and topology need their own checks.\n"
        "  * Bisect the selection or try edges individually. If each works "
        "alone but the set fails, inspect interactions between blends.\n"
        "  * Probe several radii across the feasible geometric range, "
        "including larger ones, in a CAD scratch script.\n"
        "  * If failures persist, inspect seams and split edges. "
        "`extrude(..., both=True)` can leave split edges at the sketch "
        "plane; a one-sided extrusion may avoid them. A lofted junction "
        "may need rebuilding or a sharp edge.",
    ),
    (
        # `NCollection_DataMap`, and the specific needle only. A generic
        # needle here (`standard_nosuchobject`) would shadow the
        # selector-matched-nothing entry below (`failure_hint` returns the first
        # match) and hand loft-to-`Vertex` advice to any OCCT map error, such as
        # a `sweep` raising `NCollection_IndexedDataMap::FindFromKey`. That is
        # what this module's docstring warns against: advice about an
        # unestablished cause is worse than the raw error.
        ("ncollection_datamap::find",),
        "OCCT failed an internal map lookup; the message alone does not "
        "identify the geometry defect. If this came from a `loft` to an "
        "apex `Vertex` over many curved or twisting sections, try the loft "
        "without the vertex, then close it with a small final section or "
        "a revolved cap. Vertex caps work on simpler stacks; this is a "
        "conditional workaround.",
    ),
    (
        ("wire is not closed", "wires not closed", "wire not closed"),
        "The outline is open. Make consecutive edge endpoints coincide "
        "and close the last edge to the first. Assemble separate edges "
        "into one closed wire before building the face.",
    ),
    (
        ("standard_nosuchobject", "standard_outofrange",
         "indexerror: list index out of range"),
        "Check whether a selector matched nothing or an index is out of "
        "range: inspect `len(sel)` before use, including selections indexed "
        "inside build123d. Chained `filter_by(Axis...)` calls can empty a "
        "selection when they require edges parallel to different axes. "
        "For an internal OCCT lookup failure, isolate the failing operation "
        "if the selection is nonempty.",
    ),
    (
        ("dimensions of objects to subtract from are inconsistent",),
        "Check operand dimensions and the EARLIER line that produced each "
        "one. A previous boolean may have silently emptied the solid, "
        "leaving this subtraction to report the failure. Find the first "
        "cut with empty `part.solids()` and compare its volume change "
        "with the intended removal. `is_valid` can be true for an empty "
        "result; cleaning, reordering, or batching cutters is not a "
        "guaranteed repair.",
    ),
    (
        ("topopebrepds_datastructure",),
        "OCCT failed while building fillet/boolean topology. If the call "
        "was `fillet`, isolate edges, check interacting blends, and test "
        "several radii; smaller is not always feasible. Inspect whether "
        "an `extrude(..., both=True)` seam splits an edge being blended. "
        "For a boolean, check operand validity and the intended overlap.",
    ),
    (
        ("bnd_box is void", "null shape", "empty mesh"),
        "The operation produced an empty shape.",
    ),
    (
        # The needles require `None` to be an *operand of a binary op*, not
        # merely the word "nonetype" somewhere in a message. `"operand
        # type(s) for"` alone was the first draft and is exactly the mistake
        # the `ncollection_datamap` entry above documents: `any()` matching
        # means it would have handed empty-boolean advice to every ordinary
        # `int + str` TypeError in the sandbox.
        ("'nonetype' and", "and 'nonetype'"),
        "An operand is `None`; trace it to the expression that produced it. "
        "In build123d 0.11.1, an empty `Solid & x` can return `None`, while "
        "empty `Solid - x`, `Part & x`, and `Compound & x` return empty "
        "shapes. `Solid` classmethods and wrapped OCCT results can therefore "
        "behave differently from free `extrude`, `loft`, and `revolve`, "
        "which return `Part`. Check overlap at each boolean; touching on "
        "a plane gives no solid intersection. Check for `None` before "
        "accessing `.solids()`, or use `Compound(children=[solid])` before "
        "the boolean to get an empty shape you can check with "
        "`len(shape.solids())`.",
    ),
)

def failure_hint(message: str) -> str | None:
    """Best-effort plain-language diagnosis for a raw modeling error, or None.

    Total by construction: an unrecognized message yields None, and the caller
    keeps whatever it would have reported anyway. A hint invented for a message
    nobody recognized would be advice about an unestablished cause, which is
    worse than the raw error.
    """
    low = (message or "").lower()
    for needles, hint in _FAILURE_HINTS:
        if any(n in low for n in needles):
            return hint
    return None


# ---------------------------------------------------------------------------
# Shading data
#
# An OBJ with no `vn` lines leaves shading to whoever loads it, and every loader
# we ship to guesses the same wrong way: Open3D's `compute_vertex_normals()` and
# three.js's `computeVertexNormals()` both average over ALL faces meeting at a
# vertex. On a WELDED mesh (which `_repair_mesh` guarantees — see
# `_weld_vertices`) that averages straight across the box's corners, so every
# sharp CAD edge renders as a soft gradient and parts look like melted wax.
#
# So the normals are written explicitly. They are the **exact B-rep surface
# normals**, evaluated on the real surface at each tessellation node by
# `_solid_compile._surface_normal`, and they arrive here already computed —
# nothing in this module infers a normal from triangle geometry except as a
# last-resort fallback for a node the kernel had no normal for.
#
# Not a crease-angle heuristic (average adjacent triangle normals, refuse to
# blend across an edge sharper than some threshold): it is accurate on smooth
# surfaces, but any threshold silently smooths away a modeled edge whose faces
# meet at less than it, so a deliberate shallow ridge comes out rounded and
# nothing reports it. Exact normals have no threshold.
#
# UVs come from the same place and cost nothing extra: the tessellator already
# stores a (u, v) parameter per node, which is what made the exact normals
# available in the first place. They are written as `vt` so a textured material
# has something to sample.
#
# The vertex list stays WELDED regardless. That is what `f v/vt/vn` is for: one
# position can carry several normals — three at a box corner — so shading splits
# per corner while the mesh's topology, and therefore every QC number computed
# from it, is unaffected.
# ---------------------------------------------------------------------------


def _unit(vec):
    """Normalize; ``None`` when the vector has no length (degenerate face)."""
    x, y, z = vec
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return None
    return (x / length, y / length, z / length)


def _face_normals(verts, faces):
    """Per-triangle cross products, left UN-normalized (magnitude = 2 * area)."""
    out = []
    for (a, b, c) in faces:
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        out.append((uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))
    return out


# ---------------------------------------------------------------------------
# OBJ serialization
# ---------------------------------------------------------------------------

def _dedup_rows(rows, *, sort: bool = False):
    """(unique rows in first-seen order, index per input row).

    Rows are pre-formatted coordinate tuples, so equality is textual and the
    output is byte-stable: the OBJ is content-addressed, so two runs must
    deduplicate identically or the asset is renamed and the object's id changes.

    ``sort`` chooses which byte-stability argument the stream rests on, and the
    two attribute streams answer differently on purpose:

    * `vn` uses first-seen order, which is stable because the face traversal is
      itself pinned — `_solid_compile._tessellate_faces` is asserted to walk
      faces in `Shape.tessellate`'s order, so insertion order is a function of
      the geometry — and because these are the bytes already-staged normals were
      written with.
    * `vt` sorts, which needs no such assumption. A texture table is small and
      heavily deduplicated, so ordering it costs nothing, and it means a change
      in OCCT's traversal cannot restage the same part under a new digest.
    """
    lookup: dict = {}
    unique: list = []
    index: list = []
    if sort:
        for row in sorted(set(rows)):
            lookup[row] = len(unique)
            unique.append(row)
        return unique, [lookup[row] for row in rows]
    for row in rows:
        j = lookup.get(row)
        if j is None:
            j = len(unique)
            lookup[row] = j
            unique.append(row)
        index.append(j)
    return unique, index


def _normal_rows(verts, faces, corner_normals):
    """Formatted `vn` rows and a per-corner index into them.

    `corner_normals[f]` is face `f`'s three exact surface normals, any of which
    may be None where the kernel had none — a cone apex or a sphere pole, where
    the surface derivatives are parallel and the cross product vanishes. Those
    fall back to the triangle's own geometric normal, and then to +Z, so every
    corner gets a usable direction; a zero-length normal renders black.
    """
    triangle_normals = None
    rows: list = []
    for fi, tri in enumerate(faces):
        supplied = corner_normals[fi] if fi < len(corner_normals) else None
        for k in range(3):
            n = supplied[k] if supplied is not None else None
            if n is None:
                if triangle_normals is None:
                    triangle_normals = [_unit(x)
                                        for x in _face_normals(verts, faces)]
                n = triangle_normals[fi] or (0.0, 0.0, 1.0)
            # Unit vectors need no more precision than the coordinates do, and
            # sharing the formatter keeps the signed-zero normalization.
            rows.append((_fmt_coord(n[0]), _fmt_coord(n[1]), _fmt_coord(n[2])))
    return _dedup_rows(rows)


def _uv_rows(faces, corner_uvs):
    """Formatted `vt` rows and a per-corner index, or ``(None, None)``.

    All-or-nothing: a UV is only meaningful to a texture sampler if every corner
    has one, and inventing (0, 0) for the rest would put a visible smear on the
    faces that were missing. If any corner lacks a parameter, no `vt` is written
    at all.

    The compile path cannot currently reach that guard — `_face_uv_mapper`'s
    degradation ladder ends in a projection that always answers — but it is what
    makes this function safe for a caller that builds coordinates some other
    way, and it is six lines.

    The table is emitted SORTED; see `_dedup_rows`.
    """
    rows: list = []
    for fi in range(len(faces)):
        supplied = corner_uvs[fi] if fi < len(corner_uvs) else None
        if supplied is None:
            return None, None
        for k in range(3):
            uv = supplied[k]
            if uv is None:
                return None, None
            # `_fmt_uv`, not `_fmt_coord`: identical output today, but a UV is
            # a tile count and is not bound by the weld key's precision, so the
            # two must stay separately adjustable.
            rows.append((_fmt_uv(uv[0]), _fmt_uv(uv[1])))
    return _dedup_rows(rows, sort=True)


def _write_obj(path: str, verts, faces, *,
               corner_normals=None, corner_uvs=None) -> None:
    """Write the mesh as OBJ, with per-corner normals and UVs when supplied.

    `corner_normals` / `corner_uvs` are lists parallel to `faces`, each entry a
    triple covering that triangle's three corners (see `_normal_rows` /
    `_uv_rows`). They are supplied by `_solid_compile.compile_entry` from the
    exact B-rep evaluation; nothing is inferred here beyond the documented
    fallback for a node the kernel had no normal for.

    **Normals are always written.** Omitting `corner_normals`, or supplying one
    that is missing entries, does not produce an unshaded file: every corner
    without an exact normal falls back to its own triangle's geometric normal.
    That is flat shading — correct on a planar face, faceted on a curve — and it
    is unconditionally better than writing no `vn` at all, because a loader with
    no normals averages across every sharp edge and the part renders like melted
    wax. The invariant every consumer relies on (the renderer only calls
    `compute_vertex_normals()` when the asset brought none) is that a DyCraft
    OBJ brings its own.
    """
    lines = ["o mesh"]
    for (x, y, z) in verts:
        # Same formatter as the weld key — see `_fmt_coord`.
        lines.append(f"v {_fmt_coord(x)} {_fmt_coord(y)} {_fmt_coord(z)}")

    vn = n_index = vt = uv_index = None
    if faces:
        # An absent array is "no exact normal for any corner", which the
        # per-corner fallback in `_normal_rows` then answers from the triangles.
        supplied = corner_normals or [(None, None, None)] * len(faces)
        try:
            vn, n_index = _normal_rows(verts, faces, supplied)
        except BaseException:   # noqa: BLE001 — geometry outranks its shading
            vn = n_index = None  # write the mesh unshaded rather than not at all
    if faces and corner_uvs:
        try:
            vt, uv_index = _uv_rows(faces, corner_uvs)
        except BaseException:   # noqa: BLE001 — a texture coord is not geometry
            vt = uv_index = None

    for row in (vt or ()):
        lines.append(f"vt {row[0]} {row[1]}")
    for row in (vn or ()):
        lines.append(f"vn {row[0]} {row[1]} {row[2]}")

    # OBJ is 1-indexed throughout. The four reference forms below are the whole
    # grammar we emit: `v`, `v/vt`, `v//vn`, `v/vt/vn`.
    for fi, tri in enumerate(faces):
        corners = []
        for k, v in enumerate(tri):
            slot = 3 * fi + k
            t = "" if vt is None else str(uv_index[slot] + 1)
            n = "" if vn is None else str(n_index[slot] + 1)
            if n and t:
                corners.append(f"{v + 1}/{t}/{n}")
            elif n:
                corners.append(f"{v + 1}//{n}")
            elif t:
                corners.append(f"{v + 1}/{t}")
            else:
                corners.append(f"{v + 1}")
        lines.append("f " + " ".join(corners))
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
