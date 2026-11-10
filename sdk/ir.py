"""IR canonicalization + content-hashing for the DyCraft object SDK.

The canonical JSON form is what gets hashed and what downstream consumers
should treat as authoritative. The dict returned by Object.finalize() is
this canonical form with the "id" field re-inserted.
"""

import hashlib
import json


OBJECT_SCHEMA_VERSION = "dycraft.object.v1"
#: v1 — never emitted. A v1 scene on disk still loads and renders; its symbolic
#: lighting keys are ignored, so it is lit by the renderer's automatic rig.
SCENE_SCHEMA_VERSION_V1 = "dycraft.scene.v1"

#: v2 — Blender-native lighting. A v2 scene carries a `lighting` block and none
#: of the v1 symbolic lighting keys; the renderer branches on the tag so the
#: two forms can never be half-mixed — through `is_native_scene_schema`, which
#: is v2-OR-LATER and not this string, because v3 is also Blender-native.
#: Emitted for every scene this tree builds that carries no instance state,
#: because the v1 vocabulary cannot be authored.
SCENE_SCHEMA_VERSION = "dycraft.scene.v2"

#: v3 — a SceneIR whose instances carry a per-instance **initial joint
#: configuration** (`instance["initial_joints"]`), on top of v2 lighting.
#: Bumped rather than added as an optional field, for the same reason `mounts`
#: bumped the animation schema: a reader that ignores the key evaluates every
#: instance at its *definition* defaults, so two cabinets placed from one
#: definition with different door angles render, collide and export identically
#: — wrong, and silently so. A version it does not recognize is loud instead.
#:
#: Emitted **only** when at least one instance actually carries state, so a
#: scene without the feature keeps `SCENE_SCHEMA_VERSION` and hashes
#: byte-for-byte as a v2 scene. That conditional tag is what
#: keeps every committed scene's `id`, cache key and `scene_ref` check
#: unchanged, and `initial_joints={}` identical to omitting it.
#:
#: **Why v3 and not a second v2.** v2 means "carries a lighting block", which
#: every new scene does; instance state is conditional, so it takes the next
#: tag rather than one a reader would read as the lighting generation.
SCENE_SCHEMA_VERSION_V3 = "dycraft.scene.v3"

#: Every tag a reader of a committed scene may legitimately see.
SCENE_SCHEMA_VERSIONS = (SCENE_SCHEMA_VERSION_V1, SCENE_SCHEMA_VERSION,
                         SCENE_SCHEMA_VERSION_V3)

#: The tags whose lighting and materials are the Blender-native generation:
#: v2 and everything after it. **Read this, never `== SCENE_SCHEMA_VERSION`.**
SCENE_SCHEMA_VERSIONS_NATIVE = (SCENE_SCHEMA_VERSION, SCENE_SCHEMA_VERSION_V3)


def is_native_scene_schema(schema) -> bool:
    """Whether `schema` is v2-or-later, i.e. NOT the v1 symbolic generation.

    The version boundary call sites need, expressed once. An equality test
    against `SCENE_SCHEMA_VERSION` means "is v2" where they mean "is not v1",
    and a v3 scene, v2 *plus* instance state, fails it: in
    `render/blender_renderer._lighting_frame_keys` it would lose its authored
    `@lighting` block to the automatic rig, silently.

    **Deliberately still the tag, not key presence.** The alternative — branch
    on whether a `lighting` key is there — would let a hand-built or
    half-migrated IR be lit by both systems at once, which is the failure the
    version boundary exists to make impossible (see
    `blender_renderer._lighting_frame_keys`). Consulting the tag is right;
    comparing it against one member of the set is what would be wrong.

    Unknown or missing tags are not native: an IR this tree has never emitted
    gets the conservative branch rather than the newest one.
    """
    return schema in SCENE_SCHEMA_VERSIONS_NATIVE


HASH_DIGEST_BYTES = 16  # blake2b-128 -> 32 hex chars


def canonical_json_bytes(ir_without_id) -> bytes:
    """Render the IR (minus the id field) as canonical JSON bytes."""
    return json.dumps(
        ir_without_id,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(ir_without_id) -> str:
    return hashlib.blake2b(
        canonical_json_bytes(ir_without_id),
        digest_size=HASH_DIGEST_BYTES,
    ).hexdigest()


def insert_id(ir_without_id) -> dict:
    """Compute the hash of `ir_without_id` and return a new dict with `id` set.

    The returned dict has the same keys plus `id`. Iteration order in the
    returned dict is unspecified, but the canonical JSON form (sort_keys)
    is what hashes; on-disk byte order is determined by that.
    """
    h = content_hash(ir_without_id)
    out = dict(ir_without_id)
    out["id"] = h
    return out
