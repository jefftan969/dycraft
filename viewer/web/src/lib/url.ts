// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
/**
 * A run id is a runs-dir-relative *path* (`achieve/kettle`), and the API's
 * `:path` params expect the separators intact — so each segment is encoded and
 * the slashes are not. `encodeURIComponent` on the whole id turns them into
 * `%2F` and makes every nested run unreachable.
 *
 * Lives here rather than in `api.ts` so it can be imported without dragging
 * `import.meta.env` in: `scripts/check-live-camera.mjs` bundles the live-view
 * hook under plain node, where that is undefined.
 */
export function encodeRunId(runId: string): string {
  return runId.split("/").map(encodeURIComponent).join("/");
}

export function updateUrlSearchParams(
  updater: (params: URLSearchParams) => void,
): void {
  if (typeof window === "undefined") {
    return;
  }

  const url = new URL(window.location.href);
  updater(url.searchParams);
  window.history.replaceState(window.history.state, "", url);
}

/**
 * Reserved first segment of a `<mesh filename>` naming a derived viewer asset.
 *
 * Mirrors `viewer/runs.DERIVED_DIR`. The full shape is
 * `__derived__/<ir_id>/<file>.glb`: derived assets live in a server-side cache
 * outside the run directory, and a scene draws several objects whose manifests
 * number their files per object, so the id segment is what disambiguates them.
 */
export const DERIVED_MESH_PREFIX = "__derived__/";

/**
 * Whether a URDF mesh filename names a derived (baked-atlas) asset.
 *
 * **Per-visual, not per-model.** A scene's objects are baked independently and
 * a partially baked one draws a mixture: derived GLBs for the objects that are
 * ready, staged OBJs for the rest. Deciding from the filename is what keeps
 * both halves right — a GLB must keep the material it arrived with, and an OBJ
 * must still get one built from the URDF spec.
 */
export function isDerivedMeshFilename(filename: string | undefined | null): boolean {
  return typeof filename === "string" && filename.startsWith(DERIVED_MESH_PREFIX);
}
