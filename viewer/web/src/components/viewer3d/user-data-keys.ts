// `Object3D.userData` keys that one viewer3d module writes and another reads.

/**
 * Set on a Mesh that kept the material its file arrived with — i.e. one drawing
 * a **derived** (baked-atlas) asset.
 *
 * Written by `geometry-loader` on the `keepLoadedMaterial` branch and read by
 * `native-track::stagedMaterial`, which declines to lend a baked material to an
 * evaluated mesh that has no atlas UVs to sample it with. Per mesh rather than
 * per model because a partially baked scene draws a mixture.
 */
export const DERIVED_MATERIAL_USER_DATA_KEY = "dycraftDerivedMaterial";

/**
 * Marks a group another hook owns that is mounted inside the URDF model: the native clip (`useNativeAnimation`).
 * When the model is reloaded in place (its shader colors arrived, say), `useUrdfLoader` frees the old model without
 * descending into it, and the owner moves it to the new model.
 */
export const OWNED_ELSEWHERE_USER_DATA_KEY = "dycraftOwnedElsewhere";
