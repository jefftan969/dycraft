# CMU joint → character bone maps

These maps match bone names after normalization:
- an exporter's `prefix:` is dropped (`mixamorig:LeftArm` → `LeftArm`);
- one trailing `_<n>` or `.<n>` index is dropped (`upperarm_l_024` → `upperarm_l`).

Match against your rig's actual bone names. Its `native_rig.bones` lists them (the bundle's `ir.json`, and `preloaded/manifest.json` once the asset is loaded), and `native_rig.anatomy.skeleton` names its naming: these columns are `biped`, `mixamo`, and `unreal`.

| CMU joint | 3ds Max Biped (Rocketbox) | Mixamo / ReadyPlayerMe | Unreal-style (Renderpeople) |
|---|---|---|---|
| Hips | `Bip01 Pelvis` | `Hips` | `pelvis` |
| LowerBack | `Bip01 Spine` | `Spine` | `spine_01` |
| Spine | `Bip01 Spine1` | `Spine1` | `spine_02` |
| Spine1 | `Bip01 Spine2` | `Spine2` | `spine_03` |
| Neck | `Bip01 Neck` | `Neck` | `neck_01` / `neck` |
| Head | `Bip01 Head` | `Head` | `head` |
| LeftShoulder | `Bip01 L Clavicle` | `LeftShoulder` | `clavicle_l` |
| LeftArm | `Bip01 L UpperArm` | `LeftArm` | `upperarm_l` |
| LeftForeArm | `Bip01 L Forearm` | `LeftForeArm` | `lowerarm_l` |
| LeftHand | `Bip01 L Hand` | `LeftHand` | `hand_l` |
| LeftUpLeg | `Bip01 L Thigh` | `LeftUpLeg` | `thigh_l` / `upperleg_l` |
| LeftLeg | `Bip01 L Calf` | `LeftLeg` | `calf_l` / `lowerleg_l` |
| LeftFoot | `Bip01 L Foot` | `LeftFoot` | `foot_l` |
| LeftToeBase | `Bip01 L Toe0` (if present) | `LeftToeBase` | `ball_l` |

Right side: the same rows with Right / `R` / `_r`.

`mr.bone_map` finds every row but `LeftToeBase`/`RightToeBase`, so the toes keep their rest pose relative to the foot unless your own `mapping=` names them.

Not mapped:
- CMU `LHipJoint`/`RHipJoint` (zero-length offsets from `Hips`) and `Neck1` have no counterpart bone in these rigs.
- CMU finger stubs (`LeftFingerBase`, `LeftHandIndex1`, `LThumb`) are coarse: one finger base, one index joint, and a thumb, against five articulated fingers on the characters.
- Target twist bones (`upperarm_twist_l`, `lowerarm_twist_l`, and similar) have no CMU counterpart.

Things to check once per rig:
- **Side.** CMU "Left" is the actor's left. Confirm your rig's L/left bones sit on the character's left at rest; with the imported characters facing +Y, that is +X.
- **Missing roles.** Rocketbox has no toe bones, and some Mixamo exports have no `Spine2`.
- **Leg length** (hip → knee → ankle, at rest) is the usual basis for scaling the actor's translation to the character.
