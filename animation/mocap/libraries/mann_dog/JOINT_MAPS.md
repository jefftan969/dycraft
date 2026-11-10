# MANN capture → rigify dog bone map

`dog_retarget` maps onto the rigify dog metarig's bones by name where a rig has them (an exporter's trailing `_<n>` is matched). On any other four-legged rig it finds the bone for each of the same roles by the skeleton's shape. `dr.profile(arm)` returns the bones it chose. Each bone takes the rotation of the capture segment beside it, in the body frame named (`pel`, the pelvis; `tho`, the thorax).

| rig bone | capture segment (start → end) | frame |
|---|---|---|
| `spine.004` | the pelvis, placed from `Hips` | — |
| `spine.005` | `Hips` → `Spine1` | pel |
| `spine.006`, `spine.007` | `Spine1` → `Neck` | tho |
| `spine.008`, `spine.009`, `spine.010` | `Neck` → `Head` | tho |
| `spine.011`, `Head` | `Head` → `Head_end` | tho |
| `shoulder.L` | `LeftShoulder` → `LeftArm` | tho |
| `shoulder.R` | `RightShoulder` → `RightArm` | tho |
| `spine.003`, `spine.002` | `Tail` → `Tail1`, matched by direction | pel |
| `spine.001`, `spine` | `Tail1` → `Tail1_end`, matched by direction | pel |

The legs are solved, not copied: each is re-solved at its own bone lengths with its paw pinned to the ground on contact.

| rig leg (bones, paw control) | capture leg |
|---|---|
| front left: `front_thigh.L`, `front_shin.L`, `front_foot.L`; `front_toe.L` | `LeftArm` → `LeftForeArm` → `LeftHand` → `LeftHand_end` |
| front right: `front_thigh.R`, `front_shin.R`, `front_foot.R`; `front_toe.R` | `RightArm` → `RightForeArm` → `RightHand` → `RightHand_end` |
| hind left: `thigh.L`, `shin.L`, `foot.L`; `toe.L` | `LeftUpLeg` → `LeftLeg` → `LeftFoot` → `LeftFoot_end` |
| hind right: `thigh.R`, `shin.R`, `foot.R`; `toe.R` | `RightUpLeg` → `RightLeg` → `RightFoot` → `RightFoot_end` |

A rig must also be built like the rigify dog: the hind legs hanging from the pelvis (or lumbar bones under it), the front legs from the spine above it through the shoulders, and the tail bones carrying no leg. `dr.profile` raises an error naming the bones otherwise.
