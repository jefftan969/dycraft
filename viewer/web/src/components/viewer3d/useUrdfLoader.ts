// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { describeLinkVisuals, originToMatrix4, parseUrdf, rewriteAbsoluteMeshFilenames } from './urdf-parser';
import type { UrdfSpec } from './urdf-parser';
import { buildRobotSceneGraph, collisionColorForIndex } from './scene-graph-builder';
import { computeFit, preserveViewAcrossModelSwitch, updateCameraClipping } from './camera-utils';
import { positionGroundHelpers } from './lighting';
import { instantiateGeometryObject, isTemplateGeometry, loadGeometryObject, loadGeometryTemplateFor } from './geometry-loader';
import { prefetchMeshBundle } from './mesh-bundle';
import { depthBiasForOrdinal, resolveVisualMaterialSpec } from './materials';
import { isDerivedMeshFilename } from '@/lib/url';
import { OWNED_ELSEWHERE_USER_DATA_KEY } from './user-data-keys';
import { clearLoading, setLoading as setTrayRow } from '@/lib/loading';
import { sliceClock } from '@/lib/yield';

/** Sentinel name attached to the robot group so we can find and remove it later. */
const ROBOT_GROUP_NAME = '__articraft_robot__';

/**
 * The URDF response header saying whether the run's shader parts are drawn in their flat colors yet: `pending`
 * while the server reads them (`viewer/derived.py` `flat_colors`, one Blender process of seconds), then `ready`,
 * `error`, or `none`. Until then those parts draw gray.
 */
const FLAT_COLORS_HEADER = 'X-Dycraft-Flat-Colors';
const FLAT_COLORS_POLL_MS = 1500;

function buildClippingTargets(
  robotGroup: THREE.Object3D,
  gridGroup: THREE.Group | null,
  axisGroup: THREE.Group | null,
): THREE.Object3D[] {
  const targets: THREE.Object3D[] = [robotGroup];
  if (gridGroup) {
    targets.push(gridGroup);
  }
  if (axisGroup) {
    targets.push(axisGroup);
  }
  return targets;
}

export interface UrdfLoaderState {
  urdfSpec: UrdfSpec | null;
  jointNodes: Map<string, THREE.Object3D> | null;
  jointFrames: Map<string, THREE.Group> | null;
  /**
   * Link name -> its group, after the async visual meshes have been attached.
   * Exposed for native animation playback, which hangs its clip beside the
   * `world` link and borrows each staged link's material. Set in the same
   * commit as `jointNodes`, i.e. only once `attachVisualMeshGeometry` has
   * resolved, so a consumer never sees an empty link group.
   */
  linkNodes: Map<string, THREE.Group> | null;
  /**
   * A *first* load for the current selection — nothing valid is drawn yet, so
   * the canvas covers itself.
   *
   * Deliberately false while an already-drawn model is being replaced by
   * another variant of itself (the baked-material swap, a recompiled IR):
   * there the existing scene stays on screen, interactive, until the
   * replacement is complete. See `notice`.
   */
  loading: boolean;
  error: string | null;
  /**
   * Something is wrong but the viewport is still worth looking at: some part
   * meshes failed, or a variant swap failed and the previous appearance was
   * kept. Distinct from `error`, which means there is nothing to draw.
   */
  notice: string | null;
  collisionGeometryState: CollisionGeometryState;
}

export interface CollisionGeometryState {
  declaredCount: number;
  primitiveCount: number;
  meshCount: number;
  loadedMeshCount: number;
  failedMeshCount: number;
  loading: boolean;
}

const EMPTY_COLLISION_GEOMETRY_STATE: CollisionGeometryState = {
  declaredCount: 0,
  primitiveCount: 0,
  meshCount: 0,
  loadedMeshCount: 0,
  failedMeshCount: 0,
  loading: false,
};

function syncLoadedMeshVisibility(root: THREE.Object3D, showCollisionMeshes: boolean): void {
  root.traverse((child) => {
    if (child.userData.articraftVisual === true) {
      child.visible = !showCollisionMeshes;
    }
    if (child.userData.articraftCollision === true) {
      child.visible = showCollisionMeshes;
    }
  });
}

function normalizeRobotGroupToGroundOrigin(robotGroup: THREE.Group): void {
  robotGroup.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(robotGroup);
  if (box.isEmpty()) {
    return;
  }

  const center = box.getCenter(new THREE.Vector3());
  robotGroup.position.x -= center.x;
  robotGroup.position.y -= box.min.y;
  robotGroup.position.z -= center.z;
  robotGroup.updateMatrixWorld(true);
}

/**
 * Hook that fetches and loads a URDF from a base file URL.
 *
 * When baseFileUrl changes the hook:
 * 1. Fetches the URDF from {baseFileUrl}/model.urdf
 * 2. Parses with parseUrdf, rewrites absolute mesh filenames
 * 3. Builds the scene graph with buildRobotSceneGraph
 * 4. Removes the previous robot group from the scene and adds the new one
 * 5. Fits the camera to the new model with computeFit
 * 6. Normalizes the model to the grid origin and scales the ground helpers
 */
export function useUrdfLoader(
  baseFileUrl: string | null,
  assetRevisionKey: string | null,
  jointPoseSignal: Map<string, number>,
  showCollisionMeshes: boolean,
  /**
   * Draw the run's baked-atlas viewer assets instead of the staged OBJs.
   *
   * Selects the viewer-only URDF variant (`?derived=1`), whose `<mesh
   * filename>` entries point at the derived GLBs — for an object run its own
   * parts, for a scene run each catalog object that has a complete bake.
   * Everything else about the document — the joint tree, the link frames, the
   * collision proxies — is what the canonical exporter produces, so only the
   * geometry each link draws changes. The caller must not set this before
   * `useDerivedAssets(...).ready`; the server ignores the flag for any object
   * with no complete bake, which degrades to the preview rather than to a 404.
   *
   * Which material each mesh ends up with is **not** decided here: that is per
   * visual, from the filename — see `attachVisualMeshGeometry`.
   */
  useDerivedAssets: boolean,
  scene: THREE.Scene | null,
  camera: THREE.PerspectiveCamera | null,
  controls: OrbitControls | null,
  gridGroup: THREE.Group | null,
  axisGroup: THREE.Group | null,
  invalidate: () => void,
): UrdfLoaderState {
  const [urdfSpec, setUrdfSpec] = useState<UrdfSpec | null>(null);
  const [jointNodes, setJointNodes] = useState<Map<string, THREE.Object3D> | null>(null);
  const [jointFrames, setJointFrames] = useState<Map<string, THREE.Group> | null>(null);
  const [linkNodes, setLinkNodes] = useState<Map<string, THREE.Group> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [collisionGeometryState, setCollisionGeometryState] = useState<CollisionGeometryState>(
    EMPTY_COLLISION_GEOMETRY_STATE,
  );

  // Track the current robot group so we can remove it when swapping models.
  const robotGroupRef = useRef<THREE.Group | null>(null);
  // Which run the group in the scene belongs to. What separates "load the next
  // run" from "reload this run in another variant", which the viewport has to
  // treat differently — see `loading`.
  const loadedBaseUrlRef = useRef<string | null>(null);
  const linkNodesRef = useRef<Map<string, THREE.Group> | null>(null);
  const specRef = useRef<UrdfSpec | null>(null);
  const collisionMeshesLoadedKeyRef = useRef<string | null>(null);
  const collisionMeshesLoadingKeyRef = useRef<string | null>(null);
  // The URDF variant whose meshes are on screen. A reload for the shader colors alone draws the same meshes, all in
  // the template cache already, so it skips the mesh prefetch.
  const loadedMeshKeyRef = useRef<string | null>(null);
  // Bumped once the server has read the run's shader colors (`X-Dycraft-Flat-Colors`), which reloads the URDF with
  // them in place of the gray it was first drawn in.
  const [flatColorsRead, setFlatColorsRead] = useState(0);

  useEffect(() => {
    // Reset when no base URL is available.
    if (!baseFileUrl || !scene) {
      setUrdfSpec(null);
      setJointNodes(null);
      setJointFrames(null);
      setLinkNodes(null);
      setLoading(false);
      setError(null);
      setNotice(null);
      setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);
      return;
    }

    let cancelled = false;
    let flatPollTimer: number | undefined;

    // The same run, reloaded in another variant: the baked-material asset has
    // landed, or a live run's IR was recompiled. What is on screen is a valid
    // picture of this run, so it stays — and stays orbitable and posable —
    // until the replacement is built. Only a *different* run (or a first load)
    // covers the canvas, where leaving the previous run's geometry crisply
    // visible would be a lie about what is selected.
    const isVariantSwap = robotGroupRef.current !== null
      && loadedBaseUrlRef.current === baseFileUrl;

    const load = async () => {
      // Phase timings, reported once at the end. A scene that feels slow is
      // otherwise guesswork: the phases below have completely different fixes
      // and they differ by an order of magnitude between runs.
      const started = performance.now();
      let phaseStart = started;
      const phases: string[] = [];
      const mark = (name: string) => {
        const now = performance.now();
        phases.push(`${name} ${Math.round(now - phaseStart)} ms`);
        phaseStart = now;
      };

      setLoading(!isVariantSwap);
      setError(null);
      setNotice(null);
      setTrayRow('geometry', 'Geometry', isVariantSwap ? 'updating' : 'requested');
      if (!isVariantSwap) {
        // Dropped up front only when the old graph is being abandoned anyway.
        // On a variant swap these still describe the group on screen, and a
        // failed swap keeps drawing it — so they are replaced on success
        // instead, below.
        linkNodesRef.current = null;
        specRef.current = null;
        collisionMeshesLoadedKeyRef.current = null;
        collisionMeshesLoadingKeyRef.current = null;
        setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);
      }

      try {
        // 1. Fetch URDF text.
        const urdfUrl = new URL(`${baseFileUrl}/model.urdf`, window.location.origin);
        if (assetRevisionKey) {
          urdfUrl.searchParams.set('rev', assetRevisionKey);
        }
        if (useDerivedAssets) {
          urdfUrl.searchParams.set('derived', '1');
        }

        const response = await fetch(urdfUrl.toString(), {
          cache: assetRevisionKey ? 'no-store' : 'default',
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch URDF: ${response.status} ${response.statusText}`);
        }
        // Checked before the tray rows are touched: a load the effect has abandoned must not leave a row behind.
        if (cancelled) return;
        const flatColors = response.headers.get(FLAT_COLORS_HEADER);
        if (flatColors === 'pending') setTrayRow('colours', 'Colors', 'reading shader colors');
        else clearLoading('colours');
        const urdfXml = await response.text();
        if (cancelled) return;
        mark('urdf fetch');
        setTrayRow('geometry', 'Geometry', 'parsing the model');

        // 2. Parse and rewrite mesh filenames.
        const rawSpec = parseUrdf(urdfXml);
        const spec = rewriteAbsoluteMeshFilenames(rawSpec);
        mark('urdf parse');

        // 2a. Pull the whole mesh set in a few tar chunks and prime the
        // template cache from it, so step 3 finds most of its geometry without
        // touching the network. Started here and awaited after the graph is
        // built — the archives are on the wire while that runs — and entirely
        // optional: whatever it misses, step 3 fetches per file exactly as it
        // always did. See `mesh-bundle.ts`.
        const meshKey = `${baseFileUrl}|${assetRevisionKey ?? ''}|${useDerivedAssets ? 'derived' : ''}`;
        const meshesCached = isVariantSwap && loadedMeshKeyRef.current === meshKey;
        const bundling = meshesCached ? Promise.resolve(null) : prefetchMeshBundle({
          baseUrl: baseFileUrl,
          assetRevisionKey,
          derived: useDerivedAssets,
          isCancelled: () => cancelled,
          onProgress: ({ files, primed, chunks, chunksLoaded }) => {
            if (cancelled) return;
            setTrayRow('geometry', 'Geometry', files > 0 ? `meshes ${primed}/${files}` : `chunks ${chunksLoaded}/${chunks}`,
              files > 0 ? primed / files : undefined);
          },
        });
        const declaredCollisionCount = spec.links.reduce(
          (total, link) => total + link.collisions.length,
          0,
        );
        const meshCollisionCount = spec.links.reduce(
          (total, link) =>
            total + link.collisions.filter((collision) => collision.geometry.type === 'mesh').length,
          0,
        );
        const primitiveCollisionCount = declaredCollisionCount - meshCollisionCount;

        // 3. Build scene graph.
        const { root, jointNodes: joints, jointFrames: frames, linkNodes } = buildRobotSceneGraph(spec, { showCollisions: true, textureBaseUrl: baseFileUrl });
        mark('graph');

        const bundle = await bundling;
        if (cancelled) return;
        mark(bundle
          ? `bundle ${bundle.chunksLoaded}/${bundle.chunks} chunks, `
            + `${bundle.primed} meshes, ${Math.round(bundle.bytes / 1e6)} MB`
          : meshesCached ? 'meshes cached' : 'bundle unavailable');

        const meshes = await attachVisualMeshGeometry(
          spec, linkNodes, baseFileUrl, assetRevisionKey, () => cancelled, (built, total) => {
            if (!cancelled) setTrayRow('geometry', 'Geometry', `building ${built}/${total}`, built / total);
          });
        if (cancelled) return;
        mark(`attach ${meshes.meshCount} meshes`);
        // Nothing drew at all: that is an error, not a note.
        if (meshes.meshCount > 0 && meshes.failedCount === meshes.meshCount) {
          throw new Error(meshes.firstError ?? 'no part mesh could be loaded');
        }

        // 4. Swap robot group in scene.
        const hadPreviousRobot = robotGroupRef.current !== null;
        if (robotGroupRef.current) {
          scene.remove(robotGroupRef.current);
          // The same meshes reloaded (the colors arrived): their template geometry is drawn again, so it stays on
          // the GPU rather than being freed and uploaded again.
          disposeGroup(robotGroupRef.current, meshesCached);
        }

        const robotGroup = new THREE.Group();
        robotGroup.name = ROBOT_GROUP_NAME;
        robotGroup.rotation.x = -Math.PI / 2;
        robotGroup.add(root);
        scene.add(robotGroup);
        normalizeRobotGroupToGroundOrigin(robotGroup);
        robotGroupRef.current = robotGroup;
        loadedBaseUrlRef.current = baseFileUrl;
        loadedMeshKeyRef.current = meshKey;

        // 5. Fit camera on first load, otherwise preserve the current view direction
        // while re-centering and scaling to the new object.
        if (camera && controls) {
          if (hadPreviousRobot) {
            preserveViewAcrossModelSwitch(camera, controls, robotGroup);
          } else {
            const fit = computeFit(camera, robotGroup);
            camera.position.copy(fit.position);
            controls.target.copy(fit.target);
            camera.near = fit.near;
            camera.far = fit.far;
            camera.updateProjectionMatrix();
            controls.update();
          }
        }

        // 6. Position grid and axis under the model.
        if (gridGroup && axisGroup) {
          positionGroundHelpers(gridGroup, axisGroup, robotGroup);
        }
        if (camera) {
          updateCameraClipping(camera, buildClippingTargets(robotGroup, gridGroup, axisGroup));
        }
        invalidate();
        mark('fit');
        console.info(`[dycraft] ${spec.links.length} links in `
          + `${Math.round(performance.now() - started)} ms — ${phases.join(', ')}`);

        if (!cancelled) {
          specRef.current = spec;
          linkNodesRef.current = linkNodes;
          // Handed over together, and only here: the collision effect reads
          // these refs, so clearing its keys any earlier would let it attach
          // the collision proxies to the *outgoing* graph and then consider
          // the work done for the incoming one.
          collisionMeshesLoadedKeyRef.current = null;
          collisionMeshesLoadingKeyRef.current = null;
          setCollisionGeometryState({
            declaredCount: declaredCollisionCount,
            primitiveCount: primitiveCollisionCount,
            meshCount: meshCollisionCount,
            loadedMeshCount: 0,
            failedMeshCount: 0,
            loading: false,
          });
          setUrdfSpec(spec);
          setJointNodes(joints);
          setJointFrames(frames);
          setLinkNodes(linkNodes);
          setNotice(meshes.failedCount > 0
            ? `${meshes.failedCount} of ${meshes.meshCount} part meshes failed `
              + `to load — those parts are missing from the view `
              + `(${meshes.firstError})`
            : null);
          if (flatColors === 'pending') {
            void waitForFlatColors(urdfUrl.toString());
          }
        }
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error
          ? err.message : 'Unknown error loading URDF';
        if (isVariantSwap) {
          // The group on screen is untouched — nothing was removed, since the
          // swap happens only after the new geometry is built. So keep drawing
          // it and say what did not arrive. This is the fallback path for the
          // baked-material variant: the preview stays, correctly, instead of
          // the viewport going blank on the way to a better picture.
          setNotice(`could not load this run's updated geometry — showing the `
            + `previous version (${message})`);
          return;
        }
        setError(message);
        setUrdfSpec(null);
        setJointNodes(null);
        setJointFrames(null);
        setLinkNodes(null);
        setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);
      } finally {
        if (!cancelled) {
          setLoading(false);
          clearLoading('geometry');
        }
      }
    };

    // Asks again every few seconds while the server reads the shader colors, without drawing anything, and
    // reloads once they are in. Started only after the gray model is on screen, so a slow first load is never
    // restarted by it.
    const waitForFlatColors = async (url: string) => {
      while (!cancelled) {
        await new Promise((resolve) => { flatPollTimer = window.setTimeout(resolve, FLAT_COLORS_POLL_MS); });
        if (cancelled) return;
        try {
          const poll = await fetch(url, { cache: 'no-store' });
          const status = poll.headers.get(FLAT_COLORS_HEADER);
          // Read to the end rather than canceled, which the browser lists as an aborted request: a URDF is small,
          // and this asks only while the colors are being read.
          await poll.arrayBuffer();
          if (cancelled) return;
          if (!poll.ok) {
            clearLoading('colours');
            return;
          }
          if (status !== 'pending') {
            clearLoading('colours');
            setFlatColorsRead((n) => n + 1);
            return;
          }
        } catch {
          // The next poll asks again.
        }
      }
    };

    void load();

    return () => {
      cancelled = true;
      window.clearTimeout(flatPollTimer);
      clearLoading('geometry');
      clearLoading('colours');
    };
  // `useDerivedAssets` belongs here: flipping it selects a different URDF
  // variant and a different mesh for every link, so the model has to be
  // rebuilt. `preserveViewAcrossModelSwitch` keeps the camera where it was,
  // so the swap reads as the appearance sharpening rather than as a reload.
  // `flatColorsRead` too: it reloads the same variant once its shader colors are read.
  }, [baseFileUrl, assetRevisionKey, useDerivedAssets, flatColorsRead, scene, camera, controls,
      gridGroup, axisGroup, invalidate]);

  useEffect(() => {
    const assetKey = baseFileUrl ? `${baseFileUrl}|${assetRevisionKey ?? ''}` : null;
    if (!assetKey) {
      return;
    }
    const baseUrl = baseFileUrl;

    const spec = specRef.current;
    const linkNodes = linkNodesRef.current;
    if (!spec || !linkNodes || !baseUrl) {
      return;
    }
    const meshCollisionCount = spec.links.reduce(
      (total, link) =>
        total + link.collisions.filter((collision) => collision.geometry.type === 'mesh').length,
      0,
    );
    if (meshCollisionCount === 0) {
      return;
    }
    if (!showCollisionMeshes) {
      return;
    }
    if (collisionMeshesLoadedKeyRef.current === assetKey || collisionMeshesLoadingKeyRef.current === assetKey) {
      return;
    }

    let cancelled = false;
    collisionMeshesLoadingKeyRef.current = assetKey;
    setCollisionGeometryState((current) => ({
      ...current,
      loading: true,
      failedMeshCount: 0,
    }));

    void attachCollisionMeshGeometry(spec, linkNodes, baseUrl, assetRevisionKey)
      .then(({ loadedCount, failedCount }) => {
        if (cancelled || collisionMeshesLoadingKeyRef.current !== assetKey) {
          return;
        }
        for (const linkGroup of linkNodes.values()) {
          syncLoadedMeshVisibility(linkGroup, showCollisionMeshes);
        }
        collisionMeshesLoadingKeyRef.current = null;
        collisionMeshesLoadedKeyRef.current = assetKey;
        setCollisionGeometryState((current) => ({
          ...current,
          loadedMeshCount: loadedCount,
          failedMeshCount: failedCount,
          loading: false,
        }));
        invalidate();
      })
      .catch(() => {
        if (cancelled || collisionMeshesLoadingKeyRef.current !== assetKey) {
          return;
        }
        collisionMeshesLoadingKeyRef.current = null;
        collisionMeshesLoadedKeyRef.current = assetKey;
        setCollisionGeometryState((current) => ({
          ...current,
          loadedMeshCount: 0,
          failedMeshCount: current.meshCount,
          loading: false,
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [assetRevisionKey, baseFileUrl, invalidate, showCollisionMeshes, urdfSpec]);

  useEffect(() => {
    if (!camera || !controls || !urdfSpec || !robotGroupRef.current) {
      return;
    }

    const robotGroup = robotGroupRef.current;
    const refreshClipping = () => {
      updateCameraClipping(camera, buildClippingTargets(robotGroup, gridGroup, axisGroup));
      invalidate();
    };

    refreshClipping();
    controls.addEventListener('end', refreshClipping);

    return () => {
      controls.removeEventListener('end', refreshClipping);
    };
  }, [camera, controls, urdfSpec, gridGroup, axisGroup, invalidate]);

  useEffect(() => {
    if (!camera || !urdfSpec || !robotGroupRef.current) {
      return;
    }

    updateCameraClipping(
      camera,
      buildClippingTargets(robotGroupRef.current, gridGroup, axisGroup),
    );
    invalidate();
  }, [camera, urdfSpec, gridGroup, axisGroup, jointPoseSignal, invalidate]);

  return { urdfSpec, jointNodes, jointFrames, linkNodes, loading, error, notice,
           collisionGeometryState };
}

/**
 * Load and attach every visual mesh, reporting what did not make it.
 *
 * `allSettled`, not `all`: a scene is hundreds or thousands of independent part
 * meshes, and with `all` one rejection would reject the whole load, showing
 * "Failed to load model" over an empty viewport when one part in a thousand had
 * not arrived. Drawing the rest and naming the casualty is strictly more useful
 * than drawing none.
 */
async function attachVisualMeshGeometry(
  spec: UrdfSpec,
  linkNodes: Map<string, THREE.Group>,
  baseUrl: string,
  assetRevisionKey: string | null,
  isCancelled: () => boolean = () => false,
  onProgress?: (built: number, total: number) => void,
): Promise<{ meshCount: number; failedCount: number; firstError: string | null }> {
  interface Job {
    link: UrdfSpec['links'][number];
    linkGroup: THREE.Group;
    visual: UrdfSpec['links'][number]['visuals'][number];
    descriptor: ReturnType<typeof describeLinkVisuals>[number];
    depthBias: number;
  }
  const jobs: Job[] = [];
  let visualIndex = 0;
  for (const link of spec.links) {
    const linkGroup = linkNodes.get(link.name);
    if (!linkGroup) {
      continue;
    }
    const visualDescriptors = describeLinkVisuals(link);
    for (const [visualIndexWithinLink, visual] of link.visuals.entries()) {
      const depthBias = depthBiasForOrdinal(visualIndex);
      visualIndex += 1;
      if (visual.geometry.type !== 'mesh') {
        continue;
      }
      jobs.push({ link, linkGroup, visual, descriptor: visualDescriptors[visualIndexWithinLink], depthBias });
    }
  }

  // Every template is requested at once, so the downloads overlap; the visuals are then built in order, in slices
  // of `SLICE_MS` with the event loop given a turn between them, so a scene of hundreds of meshes never holds the page.
  const templates = jobs.map((job) => {
    const template = loadGeometryTemplateFor(job.visual.geometry, baseUrl, assetRevisionKey);
    template.catch(() => undefined);    // reported per visual below
    return template;
  });
  const clock = sliceClock();
  let failedCount = 0;
  let firstError: string | null = null;
  for (let i = 0; i < jobs.length; i += 1) {
    const { link, linkGroup, visual, descriptor, depthBias } = jobs[i];
    try {
      const meshGroup = instantiateGeometryObject(await templates[i], visual.geometry, {
        kind: 'visual',
        assetRevisionKey,
        depthBias,
        materialSpec: resolveVisualMaterialSpec(visual),
        // A derived asset carries the canonical appearance in its own
        // material; rebuilding one from the URDF's flat color would discard
        // exactly what the bake produced.
        //
        // **Decided per visual, from the filename.** A scene's objects are
        // baked independently, so a partially baked scene draws a mixture:
        // a whole-model flag would either discard the baked material on the
        // GLBs or leave the un-baked OBJs with no material at all.
        keepLoadedMaterial: isDerivedMeshFilename(visual.geometry.filename),
        // A material's texture is run-relative like `<mesh filename>`, and
        // resolves against the same base: the `files/` route serves anything
        // under the run dir, so `assets/textures/<digest>.png` needs no route
        // of its own.
        textureBaseUrl: baseUrl,
      });
      if (visual.origin) {
        meshGroup.applyMatrix4(originToMatrix4(visual.origin));
      }
      meshGroup.userData.articraftLinkName = link.name;
      meshGroup.userData.articraftVisualKey = descriptor.key;
      meshGroup.userData.articraftVisualLabel = descriptor.label;
      meshGroup.name = visual.name ?? `visual:${link.name}`;
      linkGroup.add(meshGroup);
    } catch (error) {
      failedCount += 1;
      firstError ??= String(error ?? 'unknown error');
    }
    if (clock.due()) {
      onProgress?.(i + 1, jobs.length);
      await clock.yield();
      if (isCancelled()) break;
    }
  }
  onProgress?.(jobs.length, jobs.length);
  return { meshCount: jobs.length, failedCount, firstError };
}

async function attachCollisionMeshGeometry(
  spec: UrdfSpec,
  linkNodes: Map<string, THREE.Group>,
  baseUrl: string,
  assetRevisionKey: string | null,
): Promise<{ loadedCount: number; failedCount: number }> {
  const pending: Array<Promise<void>> = [];
  let collisionIndex = 0;

  for (const link of spec.links) {
    const linkGroup = linkNodes.get(link.name);
    if (!linkGroup) {
      continue;
    }

    for (const collision of link.collisions) {
      const depthBias = depthBiasForOrdinal(collisionIndex);
      const color = collisionColorForIndex(collisionIndex);
      collisionIndex += 1;

      if (collision.geometry.type !== 'mesh') {
        continue;
      }

      pending.push(
        loadGeometryObject(collision.geometry, baseUrl, {
          kind: 'collision',
          assetRevisionKey,
          depthBias,
          materialSpec: {
            name: 'collision_debug',
            color,
            opacity: 0.88,
            metalness: 0,
            roughness: 1,
            transmission: 0,
            thickness: 0,
            ior: 1.45,
            clearcoat: 0,
            clearcoatRoughness: 0.2,
            envMapIntensity: 0.8,
          },
        }).then((meshGroup) => {
          if (collision.origin) {
            meshGroup.applyMatrix4(originToMatrix4(collision.origin));
          }
          meshGroup.name = collision.name ?? `collision:${link.name}`;
          linkGroup.add(meshGroup);
        }),
      );
    }
  }

  const settled = await Promise.allSettled(pending);
  const loadedCount = settled.filter((result) => result.status === 'fulfilled').length;
  return {
    loadedCount,
    failedCount: settled.length - loadedCount,
  };
}

/**
 * Recursively dispose all geometry and materials within a group.
 */
/** `traverse`, without descending into a group another hook owns (`OWNED_ELSEWHERE_USER_DATA_KEY`). */
function traverseOwned(object: THREE.Object3D, visit: (object: THREE.Object3D) => void): void {
  if (object.userData[OWNED_ELSEWHERE_USER_DATA_KEY]) return;
  visit(object);
  for (const child of object.children) traverseOwned(child, visit);
}

function disposeGroup(group: THREE.Object3D, keepTemplates = false): void {
  traverseOwned(group, (object) => {
    if ("geometry" in object) {
      const geometry = (object as { geometry?: THREE.BufferGeometry }).geometry;
      if (geometry && !(keepTemplates && isTemplateGeometry(geometry))) geometry.dispose();
    }

    if ("material" in object) {
      const material = (object as { material?: THREE.Material | THREE.Material[] }).material;
      if (Array.isArray(material)) {
        material.forEach((entry) => {
          if (entry instanceof THREE.SpriteMaterial) {
            entry.map?.dispose();
          }
          entry.dispose();
        });
      } else if (material) {
        if (material instanceof THREE.SpriteMaterial) {
          material.map?.dispose();
        }
        material.dispose();
      }
    }
  });
}
