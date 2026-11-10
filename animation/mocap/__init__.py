"""Motion-capture retargeting helpers that authored Blender code imports.

`mocap_retarget` bakes a CMU clip onto a Biped, Mixamo, or Unreal-style rig and
makes the usual small edits; `dog_retarget` does the same for MANN dog captures
on a rigify dog. Neither is imported by the harness: `runtime.command` binds
each module here whose name does not start with an underscore into the Blender
sandbox's `/tools`, where agent code imports it by bare name.

The libraries they read are not part of the repository: `_fetch` (kept out of
the sandbox by its underscore) downloads them into `assets/mocap/{cmu,mann_dog}`
for `python -m dycraft.install mocap [--dog]`, showing each dataset's terms.
`runtime.mocap_library` decides whether the agent hears of them: with no
library, the `mocap_reference` skill, and the prompts' `<!-- mocap -->` blocks
are left out.
"""
