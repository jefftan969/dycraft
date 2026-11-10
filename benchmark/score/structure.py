"""Structural body pairs: the kinematic structure present in the delivered scene.

Only the structure is read here; whether a structural pair supports is decided by `support_rules` (contact within
tau_s). Sources, all derived from the delivered files:

* a DyCraft `baseline.json`: *rigid groups* (links joined by fixed joints; `fixed_edges` chains
  `link/X -> origin/J -> joint/J -> link/Y`), every body pair inside a group; and every revolute or prismatic joint
  (`joints[*].parent/child`) pairs the child's group with the parent's. `link/world` and every `*/world__<inst>` edge
  are ignored: every instance hangs from the world, and carried props are animated on exactly that joint.
* an exported blend's `bodies.json` (`parents` / `types`): Blender parenting, with the link-adjacency rule of the
  penetration geometry child (a mesh parented to a mesh; bodies of adjacent EMPTY "link" nodes).

Never read: agent touch events, collision allowances, scene assertions, placement parents, object constraints.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _is_world(node: str) -> bool:
    return node == "link/world" or "/world__" in node or node.endswith("/world")


def baseline_link_pairs(baseline: dict) -> tuple[dict, list]:
    """(body identity -> link, [(link_a, link_b, source)])"""
    link_of = {ident: rec.get("link") for ident, rec in (baseline.get("identities") or {}).items()}
    links = []
    for joint in (baseline.get("joints") or {}).values():
        pa, ch = joint.get("parent"), joint.get("child")
        if pa and ch and not _is_world(pa) and not _is_world(ch):
            links.append((pa, ch, f"joint:{joint.get('type')}"))
    adj = defaultdict(set)
    for edge in baseline.get("fixed_edges") or []:
        c, p = edge.get("child"), edge.get("parent")
        if c and p and not _is_world(c) and not _is_world(p):
            adj[c].add(p)
            adj[p].add(c)
    # link -> (non-link nodes)* -> link
    for start in [x for x in adj if x.startswith("link/")]:
        seen, stack = {start}, list(adj[start])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.startswith("link/"):
                if start < node:
                    links.append((start, node, "joint:fixed"))
                continue
            stack.extend(adj[node])
    return link_of, links


def dycraft_pairs(identities, baseline: dict) -> list:
    """Rigid groups (same link, and fixed-joint chains) and the joints between them.

    A revolute or prismatic joint connects the child link to the parent link, but what physically carries the child
    is usually another body *fixed to* the parent (the axle, arbor, bushing, or guide rail on its own link). So joints
    pair every body of the child's rigid group with every body of the parent's; `support_rules` then needs actual
    contact for the edge to exist.
    """
    link_of, link_pairs = baseline_link_pairs(baseline)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for la, lb, source in link_pairs:
        if source == "joint:fixed":
            union(la, lb)
    bodies_of = defaultdict(list)
    for b, ident in enumerate(identities):
        link = link_of.get(ident) if ident else None
        if link:
            bodies_of[find(link)].append(b)
    out = []
    for group, bs in bodies_of.items():
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                out.append((bs[i], bs[j], "rigid_group"))
    seen = set()
    for la, lb, source in link_pairs:
        if source == "joint:fixed":
            continue
        ga, gb = find(la), find(lb)
        if ga == gb or (ga, gb, source) in seen:
            continue
        seen.add((ga, gb, source))
        for a in bodies_of.get(ga, ()):
            for b in bodies_of.get(gb, ()):
                out.append((a, b, source))
    return out


def parent_pairs(names, sidecar: dict) -> list:
    """Exported blends: the penetration child's URDF-style link adjacency over Blender parenting."""
    parents = sidecar.get("parents") or {}
    types = sidecar.get("types") or {}
    if not parents:
        return []
    where = {x: i for i, x in enumerate(names)}
    mesh_names = set(names)
    node_of = {x: (parents.get(x) if parents.get(x) and types.get(parents.get(x)) != "MESH" else x) for x in names}
    link_nodes = set(node_of.values())

    def parent_link(node):
        cur, seen = parents.get(node), set()
        while cur and cur not in seen:
            if cur in link_nodes:
                return cur
            seen.add(cur)
            cur = parents.get(cur)
        return None
    members = defaultdict(list)
    for x in names:
        members[node_of[x]].append(x)
    out = set()
    for x in names:
        p = parents.get(x)
        if p in mesh_names:
            out.add((min(where[x], where[p]), max(where[x], where[p]), "parent"))
    for node, xs in members.items():
        if len(xs) > 1 and node not in mesh_names:
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    a, b = where[xs[i]], where[xs[j]]
                    out.add((min(a, b), max(a, b), "same_parent_link"))
        up = parent_link(node)
        if up is None:
            continue
        for x in xs:
            for y in members.get(up, ()):
                if x != y:
                    a, b = where[x], where[y]
                    out.add((min(a, b), max(a, b), "parent_link"))
    return sorted(out)


def pairs(names, identities, *, baseline_path=None, sidecar_path=None) -> tuple[list, dict]:
    """([(body_a, body_b, source)], {counts by source})"""
    out, info = [], {"sources": {}}
    if baseline_path and Path(baseline_path).is_file():
        out.extend(dycraft_pairs(identities, json.loads(Path(baseline_path).read_text())))
    if sidecar_path and Path(sidecar_path).is_file():
        out.extend(parent_pairs(names, json.loads(Path(sidecar_path).read_text())))
    for *_, source in out:
        info["sources"][source] = info["sources"].get(source, 0) + 1
    info["pairs"] = len(out)
    return out, info
