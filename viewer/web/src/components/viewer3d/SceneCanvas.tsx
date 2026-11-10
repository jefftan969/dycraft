// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import { useRef, useEffect, useMemo, useState, useCallback, type JSX, type RefObject } from 'react';
import * as THREE from 'three';
import { PartLegend, type PartLegendItem, type PartLegendSelection } from './PartLegend';
import { useCameraKeys } from './useCameraKeys';
import { useThreeScene } from './useThreeScene';
import { LoadingTray } from './LoadingTray';
import { FilmPip } from './FilmPip';
import { useUrdfLoader } from './useUrdfLoader';
import { createEdgeLines, setTextureRedrawHook } from './materials';
import { attachJointOverlay, disposeOverlayObjects } from './joint-overlay';
import { createSurfaceSamplePoints, disposeSurfaceSamplePoints } from './surface-sampling';
import {
  applySegmentColor,
  applyStudioShading,
  shouldApplyStudioShading,
  applySurfaceSampleMeshPresentation,
  restoreSegmentMaterial,
  restoreStudioShading,
  segmentColorForIndex,
  withMeshMaterials,
  type SegmentEmphasis,
} from './scene-presentation';
import { computeClipPlanesForBox, computeFitForBox, DEFAULT_CLIP_FAR } from './camera-utils';
import { useCutaway } from './useCutaway';
import { flySpeed } from './camera-keys';
import {
  type FilmCamera, type FilmCameraTrack, DEFAULT_FOV_DEGREES, filmCameraAt, isUsableFilmCamera, placeAtFilmCamera,
  sameCameraPose,
} from './film-camera';
import { useBlenderStream } from './useBlenderStream';
import type { DerivedAssetsState } from './useDerivedAssets';
import {
  canvasToBlob,
  downloadBlob,
  makeSnapshotFileName,
  setVisibleTemporarily,
} from './snapshot-export';
import type { RenderOptions } from '@/components/viewer3d/useRenderOptions';
import { describeLinkVisuals, type UrdfSpec } from './urdf-parser';
import { cn } from '@/lib/utils';

const ROBOT_GROUP_NAME = '__articraft_robot__';
/** The viewport's controls, as the small line in its corner states them (`useCameraKeys`, and App's playback keys). */
const CONTROLS_HINT = 'Drag look · WASD/Space/Shift fly (WW fast) · R reset · K play · J/L seek';
const CLICK_MOVE_THRESHOLD_PX = 5;
const PREVIEW_MAX_PIXEL_RATIO = 1.25;
const DEFAULT_MAX_PIXEL_RATIO = 2;

function findLinkName(object: THREE.Object3D | null): string | null {
  let current = object;

  while (current) {
    if (current.name.startsWith('link:')) {
      return current.name.slice(5);
    }
    current = current.parent;
  }

  return null;
}

function findVisualKey(object: THREE.Object3D | null): string | null {
  let current = object;

  while (current) {
    const visualKey = current.userData.articraftVisualKey;
    if (typeof visualKey === 'string' && visualKey.length > 0) {
      return visualKey;
    }
    current = current.parent;
  }

  return null;
}

function objectMatchesPartSelection(
  object: THREE.Object3D | null,
  selection: PartLegendSelection | null,
): boolean {
  if (!selection) {
    return true;
  }

  const linkName = findLinkName(object);
  if (linkName !== selection.partName) {
    return false;
  }

  if (selection.subpartKey == null) {
    return true;
  }

  return findVisualKey(object) === selection.subpartKey;
}

function forEachOwnLinkVisualMesh(
  linkGroup: THREE.Object3D,
  visit: (mesh: THREE.Mesh) => void,
): void {
  const stack = [...linkGroup.children];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }

    if (current.name.startsWith('link:')) {
      continue;
    }

    if (current instanceof THREE.Mesh && current.userData.articraftVisual === true) {
      visit(current);
    }

    for (let index = current.children.length - 1; index >= 0; index -= 1) {
      stack.push(current.children[index]);
    }
  }
}

export interface SceneCanvasProps {
  baseFileUrl: string | null;
  assetRevisionKey: string | null;
  selectionKey: string | null;
  jointPoseSignal: Map<string, number>;
  /** Baked-atlas asset state, owned by `App` — see the note at `useDerived`. */
  derived: DerivedAssetsState;
  /**
   * The pose actually on the graph right now, from `useJointController`.
   *
   * Distinct from `jointPoseSignal` on purpose: that one is React state and is
   * synced ~10x a second during Preview Motion, which is the right rate for
   * the inspector sliders and the wrong rate for the live Blender viewport —
   * streaming a pose two frames behind the geometry reads as the object
   * dragging its own articulation.
   */
  posedJointValues?: RefObject<Map<string, number>>;
  renderOptions: RenderOptions;
  /**
   * Optional AABB in **dycraft world coordinates** to frame the camera on
   * instead of the geometry currently drawn. Animation runs pass the union of
   * the scene over every frame: fitting on one frame's geometry leaves anything
   * that starts in the air or ends on the floor outside the viewport.
   */
  fitBounds?: { min: number[]; max: number[] } | null;
  /**
   * The camera a delivered film opens on (`film-camera.ts`). When present, an
   * animation run starts there instead of on the `fitBounds` framing.
   */
  filmCamera?: FilmCamera | null;
  /**
   * The film's camera at every frame (`/film-cameras`), and the playhead's frame: while the viewport shows the film's
   * view, it follows the film's shots as the film plays.
   */
  filmTrack?: FilmCameraTrack | null;
  frame?: number;
  /**
   * The rendered film's URL, and the playback it keeps to: shown in the viewport's corner, as the project page shows
   * it (`FilmPip`). Null for a run without a film.
   */
  filmClipUrl?: string | null;
  playing?: boolean;
  rate?: number;
  fps?: number;
  /**
   * The run's native clip version (`timeline.native.version`), or null: a native film's walls and ceilings are cut
   * away when they come between the free camera and the action (`useCutaway`).
   */
  nativeVersion?: string | null;
  /**
   * The native clip's first picture is up (`useAnimationPlayback`). The cutaway is asked for only then, so working it
   * out never competes with what the first picture needs.
   */
  nativeReady?: boolean;
  /** What the empty viewport says when nothing is selected, such as that a film is being prepared. */
  idleMessage?: string | null;
  onUrdfSpecChange?: (
    spec: UrdfSpec | null,
    jointNodes: Map<string, THREE.Object3D> | null,
    /** Link name -> group, for native animation playback. */
    linkNodes: Map<string, THREE.Group> | null,
  ) => void;
  onInvalidateReady?: (invalidate: (() => void) | null) => void;
  onSnapshotReady?: (snapshot: SnapshotExporter | null) => void;
  onLoadStateChange?: (state: {
    loading: boolean;
    error: string | null;
    missingArtifacts: boolean;
    collisionGeometryState: {
      declaredCount: number;
      primitiveCount: number;
      meshCount: number;
      loadedMeshCount: number;
      failedMeshCount: number;
      loading: boolean;
    };
  }) => void;
}

export type SnapshotExporter = () => Promise<void>;

function isMissingArtifactsError(error: string | null): boolean {
  if (!error) {
    return false;
  }
  return /404|not found|file not found/i.test(error);
}

export function SceneCanvas({
  baseFileUrl,
  assetRevisionKey,
  selectionKey,
  jointPoseSignal,
  derived,
  posedJointValues,
  renderOptions,
  fitBounds,
  filmCamera,
  filmTrack = null,
  frame = 0,
  filmClipUrl = null,
  playing = false,
  rate = 1,
  fps = 24,
  nativeVersion,
  nativeReady = false,
  idleMessage,
  onUrdfSpecChange,
  onInvalidateReady,
  onSnapshotReady,
  onLoadStateChange,
}: SceneCanvasProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const edgeLinesRef = useRef<THREE.LineSegments[]>([]);
  const jointOverlayRef = useRef<THREE.Object3D[]>([]);
  const partHighlightRef = useRef<THREE.LineSegments[]>([]);
  const surfaceSampleRef = useRef<THREE.Points[]>([]);
  const shadowPlaneRef = useRef<THREE.Mesh<THREE.PlaneGeometry, THREE.ShadowMaterial> | null>(null);
  const pointerDownRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const [selectedPartSelection, setSelectedPartSelection] = useState<PartLegendSelection | null>(null);
  const isStagingSelection = selectionKey?.startsWith("staging:") ?? false;

  const { scene, camera, renderer, controls, gridGroup, axisGroup, invalidate, sceneReady, fitSun } = useThreeScene(
    containerRef,
    {
      maxPixelRatio: renderOptions.autoAnimate ? PREVIEW_MAX_PIXEL_RATIO : DEFAULT_MAX_PIXEL_RATIO,
      continuousRender: renderOptions.autoAnimate,
      fancyGraphics: renderOptions.fancyGraphics,
    },
  );
  // Creative-mode flight (`useCameraKeys`): drag looks, WASD/Space/Shift fly at the scene's speed, the wheel steps,
  // R goes back to the film's view. The speed is measured once per run, from the drawn scene's own box (not the
  // grid's), the first time the camera moves with something drawn.
  const speedRef = useRef(0);
  // The diagonal behind the current speed: a native clip's meshes arrive after its first picture, so the speed is
  // measured again (by the clip planes' pass below) whenever the scene has grown half as large again.
  const measuredDiagonalRef = useRef(0);
  const [resetNonce, setResetNonce] = useState(0);
  useEffect(() => {
    speedRef.current = 0;
    measuredDiagonalRef.current = 0;
    // R asked for this run's framing, not the next one's.
    setResetNonce(0);
  }, [selectionKey]);
  const reset = useCallback(() => setResetNonce((n) => n + 1), []);
  useCameraKeys({ containerRef, camera, controls, invalidate, speedRef, onReset: reset });

  // Where the film's view put the camera, while that is the view: the cutaway shows every wall there, as the film
  // does, and starts once the camera moves off it. Null for a run with no film camera, which opens free.
  const filmPoseRef = useRef<THREE.Matrix4 | null>(null);
  // The film's camera at the playhead, from its track once that has arrived, else the camera it opens on; and the
  // scene's box the film view was placed against, for the follow pass below.
  const filmTrackRef = useRef(filmTrack);
  filmTrackRef.current = filmTrack;
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const filmBoxRef = useRef<THREE.Box3 | null>(null);
  // Set while the viewport itself puts the camera on the film's view, R's reframe or the follow pass: the controls'
  // own 'change' then comes before the new pose is recorded, and the clip planes and the cutaway skip it.
  const followingRef = useRef(false);
  // The film in the corner: larger on a click, hidden with its ×, both kept from run to run like a viewing choice.
  const [pipBig, setPipBig] = useState(false);
  const [pipHidden, setPipHidden] = useState(false);
  // Walls cut away (the default, as the project page shows a film's free view) or all shown: the button and H.
  const [cutWalls, setCutWalls] = useState(true);
  const cutaway = useCutaway({
    runId: selectionKey, nativeVersion, ready: nativeReady, scene, camera, controls, enabled: cutWalls,
    filmPose: filmPoseRef, placing: followingRef, invalidate,
  });
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== 'KeyH' || event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
      setCutWalls((on) => !on);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // The render distance follows the camera: on every move the near plane is set again for where it now stands (as
  // close as clips nothing of the scene), and the far plane is never nearer than `DEFAULT_CLIP_FAR`, so flying in
  // clips nothing near and flying out clips nothing far (`computeClipPlanesForBox`).
  useEffect(() => {
    if (!camera || !controls || !scene) return;
    const box = new THREE.Box3();
    const onChange = () => {
      // In the film's view the follow pass below owns the planes, from the box the view was placed against: walking
      // every mesh here on each of the film's frames would cost milliseconds a frame and undo them.
      if (followingRef.current) return;
      const pose = filmPoseRef.current;
      if (pose) {
        camera.updateMatrixWorld();
        if (sameCameraPose(camera.matrixWorld, pose)) return;
      }
      const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
      if (!robot) return;
      box.setFromObject(robot);
      // The flight speed, from the drawn scene's own box (not the grid's), again whenever it has grown.
      if (!box.isEmpty()) {
        const diagonal = box.getSize(new THREE.Vector3()).length();
        if (diagonal > 1.5 * measuredDiagonalRef.current) {
          measuredDiagonalRef.current = diagonal;
          speedRef.current = flySpeed(diagonal);
        }
      }
      if (gridGroup) box.expandByObject(gridGroup);
      if (box.isEmpty()) return;
      const clip = computeClipPlanesForBox(camera, box);
      if (Math.abs(clip.near - camera.near) < 1e-6 && Math.abs(clip.far - camera.far) < 1e-3) return;
      camera.near = clip.near;
      camera.far = clip.far;
      camera.updateProjectionMatrix();
    };
    controls.addEventListener('change', onChange);
    return () => controls.removeEventListener('change', onChange);
  }, [camera, controls, gridGroup, scene]);

  // A film's field of view stays on the camera only while its run is selected:
  // every new selection starts from the viewer's own, before the loader frames
  // it, and a film run then opens on its own camera below.
  useEffect(() => {
    filmPoseRef.current = null;
    if (!camera || camera.fov === DEFAULT_FOV_DEGREES) return;
    camera.fov = DEFAULT_FOV_DEGREES;
    camera.updateProjectionMatrix();
  }, [camera, selectionKey]);

  // The derived-asset state is owned by `App` and passed in rather than polled
  // here. It decides which URDF variant this canvas fetches.
  const useDerived = Boolean(renderOptions.bakedMaterials) && derived.ready
    && !renderOptions.showCollisions;

  const { urdfSpec, jointNodes, jointFrames, linkNodes, loading, error, notice,
          collisionGeometryState } = useUrdfLoader(
    baseFileUrl,
    assetRevisionKey,
    jointPoseSignal,
    renderOptions.showCollisions,
    useDerived,
    scene,
    camera,
    controls,
    gridGroup,
    axisGroup,
    invalidate,
  );

  // --- Live Blender viewport -------------------------------------------
  //
  // Suppressed while a *diagnostic* mode is on. Segment colors, surface
  // samples and collision geometry deliberately replace what a part looks
  // like; painting the real Cycles material over them would answer a question
  // nobody asked and hide the one they did. Every other option (grid, axes,
  // edges, joint overlay) is a decoration this covers, which is the accepted
  // cost of showing the canonical appearance.
  const diagnosticMode =
    renderOptions.showCollisions ||
    renderOptions.showSegmentColors ||
    renderOptions.showSurfaceSamples;
  const blenderCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fallbackPosedJoints = useRef<Map<string, number>>(new Map());
  const blenderStream = useBlenderStream({
    enabled: Boolean(renderOptions.blenderRender) && !diagnosticMode
      && !loading && !error && Boolean(urdfSpec),
    runId: selectionKey,
    camera,
    controls,
    scene,
    robotGroupName: ROBOT_GROUP_NAME,
    posedJointValues: posedJointValues ?? fallbackPosedJoints,
    canvasRef: blenderCanvasRef,
    // The resident scene is built from one IR; a new one needs a new session.
    modelKey: assetRevisionKey,
  });
  const blenderVisible = Boolean(renderOptions.blenderRender) && !diagnosticMode
    && blenderStream.hasFrame && blenderStream.status === 'live';

  const partLegendItems = useMemo<PartLegendItem[]>(
    () => (
      urdfSpec
        ? urdfSpec.links
            .map((link, index) => ({
              name: link.name,
              color: `#${segmentColorForIndex(index).getHexString()}`,
              visualCount: link.visuals.length,
              subparts: describeLinkVisuals(link).map((subpart) => ({
                key: subpart.key,
                name: subpart.label,
                color: `#${segmentColorForIndex(index).getHexString()}`,
              })),
            }))
            .filter((item) => item.visualCount > 0)
            .map(({ name, color, subparts }) => ({ name, color, subparts }))
        : []
    ),
    [urdfSpec],
  );
  const shouldShowPartLegend =
    Boolean(selectionKey) &&
    (renderOptions.showSegmentColors || renderOptions.showSurfaceSamples) &&
    !renderOptions.showCollisions &&
    partLegendItems.length > 0 &&
    !loading &&
    !error;
  const isStagingBuffering = isStagingSelection && error?.startsWith("Failed to fetch URDF: 404");

  const handleSnapshot = useCallback(async (): Promise<void> => {
    if (!sceneReady || !scene || !camera || !renderer || loading || error || !selectionKey) {
      throw new Error("Snapshot export is unavailable until the model is loaded.");
    }

    const previousBackground = scene.background;
    const previousClearColor = renderer.getClearColor(new THREE.Color()).clone();
    const previousClearAlpha = renderer.getClearAlpha();
    const restoreVisibility = setVisibleTemporarily(
      [
        gridGroup,
        axisGroup,
        shadowPlaneRef.current,
        ...jointOverlayRef.current,
        ...partHighlightRef.current,
        ...surfaceSampleRef.current,
        ...edgeLinesRef.current,
      ],
      false,
    );

    try {
      scene.background = new THREE.Color(0xffffff);
      renderer.setClearColor(0xffffff, 1);
      controls?.update();
      renderer.render(scene, camera);

      const blob = await canvasToBlob(renderer.domElement);
      downloadBlob(blob, makeSnapshotFileName(selectionKey));
    } finally {
      restoreVisibility();
      scene.background = previousBackground;
      renderer.setClearColor(previousClearColor, previousClearAlpha);
      renderer.render(scene, camera);
      invalidate();
    }
  }, [
    axisGroup,
    camera,
    controls,
    error,
    gridGroup,
    invalidate,
    loading,
    renderer,
    scene,
    sceneReady,
    selectionKey,
  ]);

  useEffect(() => {
    onUrdfSpecChange?.(urdfSpec, jointNodes, linkNodes);
  }, [urdfSpec, jointNodes, linkNodes, onUrdfSpecChange]);

  // Reframe once the run's own framing arrives, after the loader has already
  // fit on the rest pose (the timeline is a second request): a delivered film
  // opens on its own camera, and an explicit AABB otherwise reframes on the
  // whole clip. It runs once per (model, bounds, film camera), so orbiting
  // afterwards is not undone.
  useEffect(() => {
    if (!camera || !controls || !scene || !urdfSpec) return;
    const opening = isUsableFilmCamera(filmCamera) ? filmCamera : null;
    // R mid-film returns to the shot on screen, not the film's first one.
    const film = opening && filmTrackRef.current
      ? (filmCameraAt(filmTrackRef.current, frameRef.current) ?? opening) : opening;
    // Without a film camera or a clip's bounds the loader's own framing stands, until R asks for it again.
    if (!fitBounds && !film && resetNonce === 0) return;
    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) return;
    robot.updateMatrixWorld(true);
    // dycraft world -> three.js world. The robot group carries the z-up
    // rotation and the ground-normalizing offset, and the URDF root sits at
    // its origin, so its matrix is exactly that change of basis. A native film
    // has no AABB (its timeline's `bounds` is null), so the drawn model's own
    // box stands in for it.
    // An orbit still gliding from the last run would carry the camera off the pose placed below on the next frames
    // (the controls damp their motion): one update without damping spends what is left of it first.
    const damping = controls.enableDamping;
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = damping;
    const box = fitBounds
      ? new THREE.Box3(
          new THREE.Vector3(...(fitBounds.min as [number, number, number])),
          new THREE.Vector3(...(fitBounds.max as [number, number, number])),
        ).applyMatrix4(robot.matrixWorld)
      : new THREE.Box3().setFromObject(robot);
    if (film) {
      controls.target.copy(placeAtFilmCamera(camera, film, robot.matrixWorld, box));
      // Clip planes for where the film camera stands, spanning the whole scene
      // from there; a scene with nothing drawn yet gets the viewport's own.
      const clip = box.isEmpty() ? { near: 0.01, far: DEFAULT_CLIP_FAR } : computeClipPlanesForBox(camera, box);
      camera.near = clip.near;
      camera.far = clip.far;
    } else {
      if (box.isEmpty()) return;
      const fit = computeFitForBox(camera, box);
      camera.position.copy(fit.position);
      controls.target.copy(fit.target);
      camera.near = fit.near;
      camera.far = fit.far;
    }
    camera.updateProjectionMatrix();
    // The update's own 'change' comes before the film's pose is recorded: the clip planes and the cutaway skip it
    // (`followingRef`) and decide on the one sent once the pose is.
    followingRef.current = Boolean(film);
    try {
      controls.update();
    } finally {
      followingRef.current = false;
    }
    camera.updateMatrixWorld();
    filmPoseRef.current = film ? camera.matrixWorld.clone() : null;
    filmBoxRef.current = film ? box.clone() : null;
    if (film) controls.dispatchEvent({ type: 'change' });
    invalidate();
  }, [camera, controls, fitBounds, filmCamera, invalidate, resetNonce, scene, urdfSpec]);

  // The look's sun is aimed at what is drawn and its shadow fitted to it: once the model is in, and again once a native
  // clip has replaced it, whose extent can differ.
  useEffect(() => {
    if (!scene || !sceneReady || !urdfSpec) return;
    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) return;
    robot.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(robot);
    if (!box.isEmpty()) fitSun(box);
  }, [fitSun, nativeReady, renderOptions.fancyGraphics, scene, sceneReady, urdfSpec]);

  // The film's view follows the film's shots as it plays: on every frame, while the camera is still exactly where the
  // film's view put it, it moves to that frame's film camera, lens and all. A look, a flight, or a step of the wheel
  // leaves the film's view, and this stops until R returns to it.
  useEffect(() => {
    const pose = filmPoseRef.current;
    if (!filmTrack || !pose || !camera || !controls || !scene) return;
    camera.updateMatrixWorld();
    if (!sameCameraPose(camera.matrixWorld, pose)) return;
    const film = filmCameraAt(filmTrack, frame);
    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!film || !robot) return;
    robot.updateMatrixWorld(true);
    const box = filmBoxRef.current ?? new THREE.Box3().setFromObject(robot);
    controls.target.copy(placeAtFilmCamera(camera, film, robot.matrixWorld, box));
    if (!box.isEmpty()) {
      const clip = computeClipPlanesForBox(camera, box);
      camera.near = clip.near;
      camera.far = clip.far;
    }
    camera.updateProjectionMatrix();
    followingRef.current = true;
    try {
      controls.update();
    } finally {
      // Never left set: the clip planes would stop following the reader's moves for the rest of the session.
      followingRef.current = false;
    }
    camera.updateMatrixWorld();
    filmPoseRef.current = camera.matrixWorld.clone();
    // As in the reframe above: the update's own 'change' was skipped, and this one, with the pose recorded, counts.
    controls.dispatchEvent({ type: 'change' });
    invalidate();
  }, [camera, controls, filmTrack, frame, invalidate, scene]);

  useEffect(() => {
    onInvalidateReady?.(sceneReady ? invalidate : null);

    return () => {
      onInvalidateReady?.(null);
    };
  }, [invalidate, onInvalidateReady, sceneReady]);

  // Textures decode asynchronously, and this canvas draws on demand rather than
  // in a continuous loop — so a texture arriving after the last frame has to ask
  // for one more. Without this the part stays BLACK rather than falling back to
  // its color, because a textured material binds a white factor and an empty
  // map samples as zero. Cleared on unmount so a stale `invalidate` closed over
  // a disposed renderer is never called.
  useEffect(() => {
    setTextureRedrawHook(invalidate);

    return () => {
      setTextureRedrawHook(null);
    };
  }, [invalidate]);

  useEffect(() => {
    onSnapshotReady?.(sceneReady && !loading && !error && selectionKey ? handleSnapshot : null);

    return () => {
      onSnapshotReady?.(null);
    };
  }, [error, handleSnapshot, loading, onSnapshotReady, sceneReady, selectionKey]);

  useEffect(() => {
    onLoadStateChange?.({
      loading,
      error,
      missingArtifacts: isMissingArtifactsError(error),
      collisionGeometryState,
    });
  }, [collisionGeometryState, error, loading, onLoadStateChange]);

  useEffect(() => {
    setSelectedPartSelection(null);
  }, [selectionKey]);

  useEffect(() => {
    if (!selectedPartSelection) {
      return;
    }

    const selectedPart = partLegendItems.find((item) => item.name === selectedPartSelection.partName);
    if (!selectedPart) {
      setSelectedPartSelection(null);
      return;
    }

    if (
      selectedPartSelection.subpartKey != null
      && !selectedPart.subparts.some((subpart) => subpart.key === selectedPartSelection.subpartKey)
    ) {
      setSelectedPartSelection({
        partName: selectedPart.name,
        subpartKey: null,
      });
    }
  }, [partLegendItems, selectedPartSelection]);

  useEffect(() => {
    if ((renderOptions.showSegmentColors || renderOptions.showSurfaceSamples) && !renderOptions.showCollisions) {
      return;
    }

    setSelectedPartSelection(null);
  }, [renderOptions.showCollisions, renderOptions.showSegmentColors, renderOptions.showSurfaceSamples]);

  // Show/hide grid
  useEffect(() => {
    if (gridGroup) {
      gridGroup.visible = renderOptions.showGrid;
    }
  }, [gridGroup, renderOptions.showGrid]);

  useEffect(() => {
    if (!scene) {
      return;
    }

    const previousPlane = shadowPlaneRef.current;
    if (previousPlane) {
      previousPlane.parent?.remove(previousPlane);
      previousPlane.geometry.dispose();
      previousPlane.material.dispose();
      shadowPlaneRef.current = null;
    }

    if (!renderOptions.fancyGraphics || !urdfSpec) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    const bounds = new THREE.Box3().setFromObject(robot);
    if (bounds.isEmpty()) {
      return;
    }

    const size = bounds.getSize(new THREE.Vector3());
    const planeSize = Math.max(size.x, size.z, 1) * 2.6;
    const geometry = new THREE.PlaneGeometry(planeSize, planeSize);
    const material = new THREE.ShadowMaterial({
      color: 0x111827,
      opacity: 0.12,
      transparent: true,
      depthWrite: false,
    });
    const plane = new THREE.Mesh(geometry, material);
    plane.name = '__articraft_shadow_catcher__';
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.001;
    plane.receiveShadow = true;
    plane.renderOrder = -10;
    scene.add(plane);
    shadowPlaneRef.current = plane;

    return () => {
      plane.parent?.remove(plane);
      plane.geometry.dispose();
      plane.material.dispose();
      if (shadowPlaneRef.current === plane) {
        shadowPlaneRef.current = null;
      }
    };
  }, [renderOptions.fancyGraphics, scene, urdfSpec]);

  // Show edges
  useEffect(() => {
    if (!scene) return;

    for (const line of edgeLinesRef.current) {
      line.parent?.remove(line);
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    }
    edgeLinesRef.current = [];

    if (
      !renderOptions.showEdges ||
      renderOptions.fancyGraphics ||
      renderOptions.showCollisions ||
      renderOptions.showSurfaceSamples
    ) return;

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) return;

    const newLines: THREE.LineSegments[] = [];
    robot.traverse((obj) => {
      if (
        obj instanceof THREE.Mesh &&
        obj.geometry &&
        obj.userData.articraftVisual === true &&
        obj.visible
      ) {
        const edges = createEdgeLines(obj.geometry);
        obj.add(edges);
        newLines.push(edges);
      }
    });
    edgeLinesRef.current = newLines;
  }, [
    scene,
    renderOptions.fancyGraphics,
    renderOptions.showEdges,
    renderOptions.showCollisions,
    renderOptions.showSurfaceSamples,
    urdfSpec,
  ]);

  useEffect(() => {
    if (!scene || !urdfSpec) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    // The same predicate the native-animation path uses — see
    // `shouldApplyStudioShading`. Two copies of it is what let a clip render
    // brighter than its own scene.
    // Spelled out field by field rather than passing `renderOptions` whole:
    // the effect's deps are these four primitives, and handing the object over
    // would make the whole of it a dependency and re-run this on every
    // unrelated toggle.
    const studioShading = shouldApplyStudioShading({
      fancyGraphics: renderOptions.fancyGraphics,
      showCollisions: renderOptions.showCollisions,
      showSegmentColors: renderOptions.showSegmentColors,
      showSurfaceSamples: renderOptions.showSurfaceSamples,
    });

    robot.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh) || obj.userData.articraftVisual !== true) {
        return;
      }

      withMeshMaterials(obj, (material) => {
        if (studioShading) {
          applyStudioShading(material);
        } else {
          restoreStudioShading(material);
        }
      });
    });

    return () => {
      robot.traverse((obj) => {
        if (!(obj instanceof THREE.Mesh) || obj.userData.articraftVisual !== true) {
          return;
        }

        withMeshMaterials(obj, (material) => {
          restoreStudioShading(material);
        });
      });
    };
  }, [
    scene,
    renderOptions.fancyGraphics,
    renderOptions.showCollisions,
    renderOptions.showSegmentColors,
    renderOptions.showSurfaceSamples,
    urdfSpec,
  ]);

  // Double-sided materials
  useEffect(() => {
    if (!scene) return;
    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) return;

    robot.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        const mat = obj.material;
        const side = obj.userData.articraftCollision === true || renderOptions.doubleSided
          ? THREE.DoubleSide
          : THREE.FrontSide;
        if (Array.isArray(mat)) {
          for (const m of mat) {
            m.side = side;
            m.needsUpdate = true;
          }
        } else if (mat) {
          mat.side = side;
          mat.needsUpdate = true;
        }
      }
    });
  }, [scene, renderOptions.doubleSided, urdfSpec]);

  // Show collisions
  useEffect(() => {
    if (!scene) return;
    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) return;

    robot.traverse((obj) => {
      if (obj.userData.articraftVisual === true) {
        obj.visible = !renderOptions.showCollisions;
      }
      if (obj.userData.articraftCollision === true) {
        obj.visible = renderOptions.showCollisions;
      }
    });
  }, [scene, renderOptions.showCollisions, urdfSpec]);

  useEffect(() => {
    disposeSurfaceSamplePoints(surfaceSampleRef.current);
    surfaceSampleRef.current = [];

    if (!scene || !urdfSpec || !renderOptions.showSurfaceSamples || renderOptions.showCollisions) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    robot.updateMatrixWorld(true);

    const nextSurfaceSamples: THREE.Points[] = [];
    for (const [index, link] of urdfSpec.links.entries()) {
      const linkGroup = robot.getObjectByName(`link:${link.name}`);
      if (!linkGroup) {
        continue;
      }

      const linkColor = segmentColorForIndex(index);
      forEachOwnLinkVisualMesh(linkGroup, (obj) => {
        const points = createSurfaceSamplePoints(obj, linkColor, link.name);
        if (!points) {
          return;
        }

        const visualKey = findVisualKey(obj);
        if (visualKey) {
          points.userData.articraftVisualKey = visualKey;
        }
        obj.add(points);
        nextSurfaceSamples.push(points);
      });
    }

    surfaceSampleRef.current = nextSurfaceSamples;

    return () => {
      disposeSurfaceSamplePoints(nextSurfaceSamples);
      if (surfaceSampleRef.current === nextSurfaceSamples) {
        surfaceSampleRef.current = [];
      }
    };
  }, [scene, renderOptions.showCollisions, renderOptions.showSurfaceSamples, urdfSpec]);

  useEffect(() => {
    if (!scene) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    robot.traverse((obj) => {
      if (!(obj instanceof THREE.Points) || obj.userData.articraftSurfaceSample !== true) {
        return;
      }

      const linkName =
        typeof obj.userData.articraftLinkName === 'string'
          ? obj.userData.articraftLinkName
          : findLinkName(obj);

      obj.visible =
        renderOptions.showSurfaceSamples &&
        !renderOptions.showCollisions &&
        linkName != null &&
        objectMatchesPartSelection(obj, selectedPartSelection);
    });
  }, [scene, renderOptions.showCollisions, renderOptions.showSurfaceSamples, selectedPartSelection, urdfSpec]);

  useEffect(() => {
    if (!scene || !urdfSpec) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    for (const [index, link] of urdfSpec.links.entries()) {
      const linkGroup = robot.getObjectByName(`link:${link.name}`);
      if (!linkGroup) {
        continue;
      }

      const linkColor = segmentColorForIndex(index);
      forEachOwnLinkVisualMesh(linkGroup, (obj) => {
        const emphasis: SegmentEmphasis =
          selectedPartSelection == null
            ? 'default'
            : objectMatchesPartSelection(obj, selectedPartSelection)
              ? 'selected'
              : 'dimmed';

        withMeshMaterials(obj, (material) => {
          if (renderOptions.showSegmentColors) {
            applySegmentColor(material, linkColor, emphasis);
          } else if (!renderOptions.showSurfaceSamples) {
            restoreSegmentMaterial(material);
          }

          if (renderOptions.showSurfaceSamples) {
            applySurfaceSampleMeshPresentation(material);
          }
        });
      });
    }

    return () => {
      for (const link of urdfSpec.links) {
        const linkGroup = robot.getObjectByName(`link:${link.name}`);
        if (!linkGroup) {
          continue;
        }

        forEachOwnLinkVisualMesh(linkGroup, (obj) => {
          withMeshMaterials(obj, (material) => {
            restoreSegmentMaterial(material);
          });
        });
      }
    };
  }, [scene, renderOptions.showSegmentColors, renderOptions.showSurfaceSamples, selectedPartSelection, urdfSpec]);

  useEffect(() => {
    for (const highlight of partHighlightRef.current) {
      highlight.parent?.remove(highlight);
      highlight.geometry.dispose();
      (highlight.material as THREE.Material).dispose();
    }
    partHighlightRef.current = [];

    if (!scene || !selectedPartSelection || !shouldShowPartLegend || renderOptions.showSurfaceSamples) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    const linkGroup = robot?.getObjectByName(`link:${selectedPartSelection.partName}`);
    if (!linkGroup) {
      return;
    }

    const nextHighlights: THREE.LineSegments[] = [];
    forEachOwnLinkVisualMesh(linkGroup, (obj) => {
      if (!obj.visible || !objectMatchesPartSelection(obj, selectedPartSelection)) {
        return;
      }

      const highlight = createEdgeLines(obj.geometry, 0xffffff);
      const material = highlight.material as THREE.LineBasicMaterial;
      material.opacity = 0.72;
      highlight.renderOrder = 20;
      obj.add(highlight);
      nextHighlights.push(highlight);
    });

    partHighlightRef.current = nextHighlights;

    return () => {
      for (const highlight of nextHighlights) {
        highlight.parent?.remove(highlight);
        highlight.geometry.dispose();
        (highlight.material as THREE.Material).dispose();
      }
      if (partHighlightRef.current === nextHighlights) {
        partHighlightRef.current = [];
      }
    };
  }, [scene, renderOptions.showSurfaceSamples, selectedPartSelection, shouldShowPartLegend, urdfSpec]);

  useEffect(() => {
    if (!scene || !camera || !renderer || !shouldShowPartLegend) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const handlePointerDown = (event: PointerEvent): void => {
      pointerDownRef.current = {
        x: event.clientX,
        y: event.clientY,
        pointerId: event.pointerId,
      };
    };

    const clearPointerDown = (): void => {
      pointerDownRef.current = null;
    };

    const handlePointerUp = (event: PointerEvent): void => {
      const start = pointerDownRef.current;
      pointerDownRef.current = null;

      if (!start || start.pointerId !== event.pointerId) {
        return;
      }

      const movement = Math.hypot(event.clientX - start.x, event.clientY - start.y);
      if (movement > CLICK_MOVE_THRESHOLD_PX) {
        return;
      }

      const bounds = renderer.domElement.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) {
        return;
      }

      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const intersections = raycaster.intersectObject(robot, true);
      const hit = intersections.find(
        ({ object }) => object instanceof THREE.Mesh && object.userData.articraftVisual === true && object.visible,
      );
      const linkName = hit ? findLinkName(hit.object) : null;
      if (!linkName) {
        return;
      }

      setSelectedPartSelection((current) => (
        current?.partName === linkName && current.subpartKey == null
          ? null
          : { partName: linkName, subpartKey: null }
      ));
    };

    renderer.domElement.addEventListener('pointerdown', handlePointerDown);
    renderer.domElement.addEventListener('pointerup', handlePointerUp);
    renderer.domElement.addEventListener('pointercancel', clearPointerDown);
    renderer.domElement.addEventListener('pointerleave', clearPointerDown);
    return () => {
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp);
      renderer.domElement.removeEventListener('pointercancel', clearPointerDown);
      renderer.domElement.removeEventListener('pointerleave', clearPointerDown);
    };
  }, [camera, renderer, scene, shouldShowPartLegend]);

  useEffect(() => {
    disposeOverlayObjects(jointOverlayRef.current);
    jointOverlayRef.current = [];

    if (!renderOptions.showJointOverlay || !urdfSpec || !jointFrames || !scene) {
      return;
    }

    const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
    if (!robot) {
      return;
    }

    const size = new THREE.Box3().setFromObject(robot).getSize(new THREE.Vector3());
    const extent = Math.max(size.x, size.y, size.z, 0.6);
    jointOverlayRef.current = attachJointOverlay(urdfSpec, jointFrames, extent);

    return () => {
      disposeOverlayObjects(jointOverlayRef.current);
      jointOverlayRef.current = [];
    };
  }, [jointFrames, renderOptions.showJointOverlay, scene, urdfSpec]);

  useEffect(() => {
    if (!sceneReady) {
      return;
    }
    invalidate();
  }, [
    invalidate,
    jointPoseSignal,
    renderOptions.doubleSided,
    renderOptions.fancyGraphics,
    renderOptions.showCollisions,
    renderOptions.showEdges,
    renderOptions.showGrid,
    renderOptions.showJointOverlay,
    renderOptions.showSegmentColors,
    renderOptions.showSurfaceSamples,
    sceneReady,
    selectedPartSelection,
    shouldShowPartLegend,
    urdfSpec,
  ]);

  return (
    <div className="relative h-full w-full">
      {/*
        Focusable, so the keyboard camera keeps working after a click even if
        the pointer leaves — and focused explicitly on pointerdown, because
        OrbitControls consumes that event on the canvas inside.
      */}
      <div
        ref={containerRef}
        tabIndex={0}
        aria-label="3D viewport"
        onPointerDown={() => containerRef.current?.focus({ preventScroll: true })}
        className="h-full w-full outline-none"
      />

      {/*
        The canonical Cycles frame, covering the WebGL canvas.

        `pointer-events-none` is what keeps requirement 2 true: orbit, pan,
        zoom, and part picking all still land on the canvas underneath, which is
        live and authoritative for every piece of state. This element only
        *shows* the answer. Its intrinsic size is the streamed frame's, and CSS
        stretches it to the viewport — so the interactive tier is an upscaled
        correct render rather than a differently-shaded one.
      */}
      <canvas
        ref={blenderCanvasRef}
        aria-hidden
        className={cn(
          "pointer-events-none absolute inset-0 h-full w-full",
          blenderVisible ? "opacity-100" : "opacity-0",
        )}
      />

      {renderOptions.bakedMaterials && !renderOptions.blenderRender
        && selectionKey && !loading && !error
        && (derived.status === 'building' || derived.status === 'error'
            || derived.status === 'partial'
            || (derived.status === 'ready' && derived.warnings.length > 0)) ? (
        <div className="pointer-events-none absolute left-2 top-8 z-[6] max-w-[20rem]">
          {derived.status === 'building' ? (
            <div className="rounded bg-black/55 px-2 py-1">
              <p className="font-mono text-[9px] leading-tight text-white/90">
                baking materials from the shader…
                {derived.objects && derived.objects > 1
                  ? ` ${derived.objectsReady ?? 0}/${derived.objects} objects`
                  : ''}
                {' '}— showing the preview until it lands
              </p>
            </div>
          ) : derived.status === 'partial' ? (
            /*
              Some objects baked and the rest are not coming. The viewport IS
              showing the derived mixture, so this says which part of it is
              still the preview rather than claiming the whole thing failed.
            */
            <div className="rounded border border-amber-300 bg-amber-50/95 px-2 py-1">
              <p className="text-[10px] font-medium text-amber-800">
                {derived.objectsReady ?? 0} of {derived.objects ?? 0} objects
                {' '}baked — the rest keep the preview appearance
              </p>
              <p className="mt-0.5 font-mono text-[9px] leading-tight text-amber-700">
                {derived.reason}
              </p>
            </div>
          ) : derived.status === 'error' ? (
            <div className="rounded border border-amber-300 bg-amber-50/95 px-2 py-1">
              <p className="text-[10px] font-medium text-amber-800">
                Baked materials unavailable — showing the preview viewport
              </p>
              <p className="mt-0.5 font-mono text-[9px] leading-tight text-amber-700">
                {derived.reason}
              </p>
            </div>
          ) : (
            <div className="rounded border border-amber-300 bg-amber-50/95 px-2 py-1">
              <p className="font-mono text-[9px] leading-tight text-amber-700">
                {derived.warnings[0]}
              </p>
            </div>
          )}
        </div>
      ) : null}

      {cutaway.available && cutaway.free && selectionKey && !loading ? (
        <div className={cn('absolute right-2 z-[6]',
                           renderOptions.blenderRender && blenderStream.status !== 'idle' ? 'top-12' : 'top-2')}>
          <button
            type="button"
            onClick={() => setCutWalls((on) => !on)}
            aria-pressed={cutWalls}
            data-hidden-walls={cutaway.hiddenCount}
            title={`${cutWalls ? 'Show all walls' : 'Cut away walls'} (H)`}
            className="rounded bg-black/55 px-2 py-1 font-mono text-[10px] leading-tight text-white/90 hover:bg-black/70"
          >
            {cutWalls ? 'Show all walls' : 'Cut away walls'}
          </button>
        </div>
      ) : null}

      {renderOptions.blenderRender && !diagnosticMode && selectionKey
        && blenderStream.status !== 'idle' ? (
        <div className="pointer-events-none absolute right-2 top-2 z-[6] max-w-[18rem]">
          {blenderStream.status === 'error' ? (
            <div className="rounded border border-amber-300 bg-amber-50/95 px-2 py-1">
              <p className="text-[10px] font-medium text-amber-800">
                Blender view unavailable — showing the preview viewport
              </p>
              <p className="mt-0.5 font-mono text-[9px] leading-tight text-amber-700">
                {blenderStream.error}
              </p>
            </div>
          ) : (
            <div className="rounded bg-black/55 px-2 py-1 text-right">
              <p className="font-mono text-[9px] leading-tight text-white/90">
                {blenderStream.status === 'live' && blenderStream.hasFrame
                  ? `cycles ${blenderStream.frameHeight ?? '?'}p`
                    + (blenderStream.frameMs ? ` · ${blenderStream.frameMs} ms` : '')
                  : 'starting Blender…'}
              </p>
              {blenderStream.warnings.length > 0 ? (
                <p className="mt-0.5 max-w-[16rem] font-mono text-[9px] leading-tight text-amber-200">
                  {blenderStream.warnings[0]}
                </p>
              ) : null}
            </div>
          )}
        </div>
      ) : null}

      {shouldShowPartLegend ? (
        <PartLegend
          items={partLegendItems}
          selection={selectedPartSelection}
          onSelectPart={setSelectedPartSelection}
        />
      ) : null}

      {/*
        Something is missing but the picture is still worth reading — some part
        meshes did not arrive, or a variant swap failed and the previous
        appearance was kept. A chip rather than the full-cover overlay below,
        because the scene underneath it is exactly what the user should be
        looking at.
      */}
      {notice && !loading && !error ? (
        <div className="pointer-events-none absolute bottom-2 left-2 z-[6] max-w-[22rem]">
          <div className="rounded border border-amber-300 bg-amber-50/95 px-2 py-1">
            <p className="font-mono text-[9px] leading-tight text-amber-700">
              {notice}
            </p>
          </div>
        </div>
      ) : null}

      {/*
        The controls, one small line in the top left, clear of the loading tray in the bottom right; it stops short
        of whatever is in the top right (the walls button, the Blender chip, the part legend) and is cut short
        rather than wrapped. The baked-materials chip, shown only with it, sits just below. It takes no pointer
        events, so the canvas stays live under it, and as a DOM overlay it is never in a snapshot.
      */}
      {selectionKey && !loading && !error ? (
        <div className={cn('pointer-events-none absolute left-2 top-2 z-[6]',
                           shouldShowPartLegend ? 'right-[15.5rem]'
                             : renderOptions.blenderRender && !diagnosticMode && blenderStream.status !== 'idle'
                               ? 'right-[19rem]' : cutaway.available && cutaway.free ? 'right-28' : 'right-2')}>
          <p className="w-fit max-w-full truncate whitespace-nowrap rounded bg-black/35 px-1.5 py-0.5 font-mono
                        text-[9px] leading-tight text-white/85"
             aria-label="Viewport controls">
            {CONTROLS_HINT}{cutaway.available ? ' · H walls' : ''}
          </p>
        </div>
      ) : null}

      {/*
        The bottom-right corner: what is still loading, above the rendered film, so neither covers the other.
      */}
      <div className={cn('pointer-events-none absolute bottom-2 right-2 z-[7] flex flex-col items-end gap-2',
                         pipBig ? 'w-[48%]' : 'w-[28%] min-w-[11rem] max-w-[26rem]')}>
        <LoadingTray inline />
        {/* Mounted through the model's loading (which can flip, as when its shader colors arrive), so the video
            is not torn down mid-download; the film is there to watch while the 3D loads. */}
        {filmClipUrl && selectionKey && !error ? (
          pipHidden ? (
            <button
              type="button"
              onClick={() => setPipHidden(false)}
              title="Show the rendered film"
              className="pointer-events-auto rounded bg-black/55 px-2 py-1 font-mono text-[10px] leading-tight
                         text-white/90 hover:bg-black/70"
            >
              Film
            </button>
          ) : (
            <FilmPip
              src={filmClipUrl}
              frame={frame}
              fps={fps}
              playing={playing}
              rate={rate}
              big={pipBig}
              onToggleBig={() => setPipBig((big) => !big)}
              onHide={() => setPipHidden(true)}
            />
          )
        ) : null}
      </div>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#f3f3f3]/70">
          <p className="text-[11px] text-[#999]">{isStagingSelection ? "Buffering...." : "Loading…"}</p>
        </div>
      )}

      {isStagingBuffering && !loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#f3f3f3]/70">
          <p className="text-[11px] text-[#999]">Buffering....</p>
        </div>
      )}

      {error && !loading && !isStagingBuffering && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#f3f3f3]/70">
          <div className="max-w-xs px-3 py-2 text-center">
            <p className="text-[11px] font-medium text-red-600">Failed to load model</p>
            <p className="mt-0.5 font-mono text-[10px] text-red-400">{error}</p>
          </div>
        </div>
      )}

      {!selectionKey && sceneReady && !loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="max-w-sm px-4 text-center text-[11px] text-[#999]">
            {idleMessage ?? "Select a record to view its 3D model"}
          </p>
        </div>
      )}
    </div>
  );
}
