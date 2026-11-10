"""Object collision allowances resolved for animation instances and poses."""
import math
try:
    from ._allowance_rules import _allowance_index, _allowance_matches
except ImportError:  # Standalone isolated Blender worker.
    from _allowance_rules import _allowance_index, _allowance_matches


class InheritedAllowances:
    """Preserve the object layer's exact pair and surface-unit scope semantics."""

    def __init__(self, scene_ir: dict, catalog: dict):
        self.objects = {}
        self.evidence = []
        #: `allows` memo: the pose dict it was computed from, held by reference
        #: so its identity stays unique, and the result per instance.
        self._pose_ref = None
        self._allowed_at_pose: dict = {}
        for inst in scene_ir["instances"]:
            obj = catalog[inst["object_ref"]]
            index = _allowance_index(obj)
            if not index:
                continue
            types = {j["name"]: j["type"] for j in obj.get("joints", ())}
            defaults = {j["name"]: j["default"] for j in obj.get("joints", ())
                        if j["type"] in ("revolute", "prismatic")}
            self.objects[inst["name"]] = (index, types, defaults)
            for rec in obj.get("collision_allowances", ()):
                self.evidence.append({**rec, "instance": inst["name"],
                                      "object_ref": inst["object_ref"]})

    def at_pose(self, instance: str, poses: dict) -> set:
        """Allowed unordered pairs at an object-local pose in radians/metres."""
        entry = self.objects.get(instance)
        if entry is None:
            return set()
        index, types, defaults = entry
        surface = {name: math.degrees(q) if types.get(name) == "revolute" else q
                   for name, q in {**defaults, **poses}.items()}
        return {pair for pair, records in index.items()
                if _allowance_matches(records, surface)}

    def allows(self, instance: str, a: str, b: str, scene_poses: dict) -> bool:
        """Is this pair exempted at this pose?

        **Memoized on the pose dict's identity, per instance.** `scene_poses`
        is rebuilt once per frame by the caller and then handed to this method
        for *every* candidate pair, so the two derivations below — stripping the
        instance prefix off each joint name, and converting the whole
        configuration to surface units to test every allowance record — run
        once per frame rather than once per pair: per pair they would be the
        largest item in the pass, and the answer is identical for every pair
        sharing a frame.
        `at_pose` stays public and unmemoized for callers that ask directly.

        The cache key is `scene_poses is ...`, not a hash of its contents —
        hashing it would cost what the memo saves. Holding the reference in
        `_pose_ref` is what makes identity sound: while it is cached the object
        cannot be freed, so its `id` cannot be reused by a later dict.
        """
        if scene_poses is not self._pose_ref:
            self._pose_ref = scene_poses
            self._allowed_at_pose = {}
        allowed = self._allowed_at_pose.get(instance)
        if allowed is None:
            poses = {key.partition(".")[2] if "." in key else key: value
                     for key, value in scene_poses.items()
                     if "." not in key or key.partition(".")[0] == instance}
            allowed = self.at_pose(instance, poses)
            self._allowed_at_pose[instance] = allowed
        return frozenset((a, b)) in allowed
