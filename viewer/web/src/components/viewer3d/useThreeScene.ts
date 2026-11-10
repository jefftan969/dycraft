import { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  createLightingRig, createGridHelper, createAxisHelper, createRoomEnvironmentMap, FILM_LOOK, fitSunToBox,
} from './lighting';
import { rendererKind, SOFTWARE_MAX_FPS, SOFTWARE_PIXEL_RATIO } from './software-renderer';

export interface ThreeSceneState {
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  renderer: THREE.WebGLRenderer | null;
  controls: OrbitControls | null;
  gridGroup: THREE.Group | null;
  axisGroup: THREE.Group | null;
  invalidate: () => void;
  sceneReady: boolean;
  /** Aim the look's sun at the drawn scene and fit its shadow to it (`fitSunToBox`); the box in viewport coordinates. */
  fitSun: (box: THREE.Box3) => void;
}

export interface ThreeSceneOptions {
  maxPixelRatio?: number;
  continuousRender?: boolean;
  fancyGraphics?: boolean;
}

type NamedLight = THREE.Light & {
  color: THREE.Color;
  groundColor?: THREE.Color;
};

const VIEWPORT_BACKGROUND = 0xe8edf5;

function resolveDepthBufferOptions(): Pick<
  THREE.WebGLRendererParameters,
  'logarithmicDepthBuffer' | 'reversedDepthBuffer'
> {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2');
  const supportsReversedDepth = context?.getExtension('EXT_clip_control') != null;
  context?.getExtension('WEBGL_lose_context')?.loseContext();

  return supportsReversedDepth
    ? { reversedDepthBuffer: true }
    : { logarithmicDepthBuffer: true };
}

function getNamedLight(group: THREE.Group, name: string): NamedLight | null {
  const object = group.getObjectByName(name);
  return object instanceof THREE.Light ? object as NamedLight : null;
}

function getNamedDirectionalLight(group: THREE.Group, name: string): THREE.DirectionalLight | null {
  const object = group.getObjectByName(name);
  return object instanceof THREE.DirectionalLight ? object : null;
}

function applyLightingQuality(lightingRig: THREE.Group, fancyGraphics: boolean): void {
  const hemiLight = getNamedLight(lightingRig, 'hemisphere-light');
  const keyLight = getNamedDirectionalLight(lightingRig, 'key-light');
  const fillLight = getNamedDirectionalLight(lightingRig, 'fill-light');
  const rimLight = getNamedDirectionalLight(lightingRig, 'rim-light');
  const frontFillLight = getNamedDirectionalLight(lightingRig, 'front-fill-light');

  if (fancyGraphics) {
    // The project page's look (`FILM_LOOK`): a hemisphere and one sun; the room probe does the rest.
    if (hemiLight) {
      hemiLight.color.set(FILM_LOOK.hemisphere.sky);
      hemiLight.groundColor?.set(FILM_LOOK.hemisphere.ground);
      hemiLight.intensity = FILM_LOOK.hemisphere.intensity;
    }
    if (keyLight) {
      keyLight.color.set(FILM_LOOK.sun.color);
      keyLight.intensity = FILM_LOOK.sun.intensity;
      keyLight.position.copy(FILM_LOOK.sun.direction).multiplyScalar(10);
      keyLight.shadow.radius = 3;
    }
    for (const light of [fillLight, rimLight, frontFillLight]) {
      if (light) light.intensity = 0;
    }
    return;
  }

  if (hemiLight) {
    hemiLight.color.set(0xffffff);
    hemiLight.groundColor?.set(0xd4d4d4);
    hemiLight.intensity = 0.95;
  }
  if (keyLight) {
    keyLight.color.set(0xfff5e6);
    keyLight.intensity = 0.75;
    keyLight.position.set(6, 12, 8);
    keyLight.shadow.bias = 0;
    keyLight.shadow.normalBias = 0;
    keyLight.shadow.radius = 1;
  }
  if (fillLight) {
    fillLight.color.set(0xf0f4ff);
    fillLight.intensity = 0.65;
    fillLight.position.set(-6, 5, -6);
  }
  if (rimLight) {
    rimLight.color.set(0xffffff);
    rimLight.intensity = 0.35;
    rimLight.position.set(0, -6, 10);
  }
  if (frontFillLight) {
    frontFillLight.color.set(0xfff9f0);
    frontFillLight.intensity = 0.5;
    frontFillLight.position.set(0, 6, 10);
  }
}

/**
 * The viewport's look. Fancy graphics are the project page's (`FILM_LOOK`): ACES tone mapping, the room probe, a
 * hemisphere and a sun with a soft shadow; otherwise the plain, bright studio rig. The sun's shadow is the look's
 * dearest part, and on a CPU rasterizer (`software-renderer.ts`) it is left off.
 */
function applyLook(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  lightingRig: THREE.Group,
  environment: { current: THREE.Texture | null },
  fancyGraphics: boolean,
): void {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = fancyGraphics ? THREE.ACESFilmicToneMapping : THREE.LinearToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = fancyGraphics && !rendererKind().software;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene.background = new THREE.Color(VIEWPORT_BACKGROUND);
  applyLightingQuality(lightingRig, fancyGraphics);
  if (fancyGraphics) {
    environment.current ??= createRoomEnvironmentMap(renderer);
    scene.environment = environment.current;
    scene.environmentIntensity = FILM_LOOK.environmentIntensity;
    return;
  }
  scene.environment = null;
  scene.environmentIntensity = 1;
  environment.current?.dispose();
  environment.current = null;
}

/**
 * Hook that manages the Three.js renderer lifecycle.
 *
 * Creates and manages: WebGLRenderer, PerspectiveCamera, Scene, OrbitControls.
 * Sets up: lighting rig, grid helper, axis helper, render loop via requestAnimationFrame.
 * Handles: ResizeObserver for container size changes, cleanup on unmount.
 */
export function useThreeScene(
  containerRef: React.RefObject<HTMLDivElement | null>,
  options: ThreeSceneOptions = {},
): ThreeSceneState {
  const [sceneReady, setSceneReady] = useState(false);
  const invalidate = useCallback(() => {
    invalidateRef.current();
  }, []);
  const fitSun = useCallback((box: THREE.Box3) => {
    if (!lightingRigRef.current) return;
    fitSunToBox(lightingRigRef.current, box);
    invalidateRef.current();
  }, []);

  // Mutable refs for scene objects so they persist without triggering re-renders.
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const gridGroupRef = useRef<THREE.Group | null>(null);
  const axisGroupRef = useRef<THREE.Group | null>(null);
  const lightingRigRef = useRef<THREE.Group | null>(null);
  const environmentMapRef = useRef<THREE.Texture | null>(null);
  const frameIdRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number | null>(null);
  const continuousRenderRef = useRef(Boolean(options.continuousRender));
  const fancyGraphicsRef = useRef(Boolean(options.fancyGraphics));
  const needsRenderRef = useRef(false);
  const invalidateRef = useRef<() => void>(() => {});

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // --- Renderer ---
    // On a CPU rasterizer (no working GPU driver) the viewport draws cheaper: see `software-renderer.ts`.
    const software = rendererKind().software;
    const renderer = new THREE.WebGLRenderer({
      antialias: !software,
      alpha: false,
      preserveDrawingBuffer: true,
      ...resolveDepthBufferOptions(),
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, options.maxPixelRatio ?? 2,
                                    software ? SOFTWARE_PIXEL_RATIO : Infinity));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.domElement.style.display = 'block';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // --- Scene ---
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // --- Camera ---
    const aspect = container.clientWidth / container.clientHeight;
    const camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 1000);
    camera.position.set(3, 3, 3);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // --- OrbitControls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    // --- Lighting ---
    const lightingRig = createLightingRig();
    scene.add(lightingRig);
    lightingRigRef.current = lightingRig;

    // --- Grid ---
    const gridGroup = createGridHelper();
    scene.add(gridGroup);
    gridGroupRef.current = gridGroup;

    // --- Axis Helper ---
    const axisGroup = createAxisHelper();
    scene.add(axisGroup);
    axisGroupRef.current = axisGroup;

    fancyGraphicsRef.current = Boolean(options.fancyGraphics);
    applyLook(renderer, scene, lightingRig, environmentMapRef, fancyGraphicsRef.current);

    const renderScene = (deltaSeconds?: number) => {
      void deltaSeconds;
      renderer.render(scene, camera);
    };

    const requestFrame = () => {
      if (frameIdRef.current === 0) {
        frameIdRef.current = requestAnimationFrame(renderFrame);
      }
    };

    // The last frame drawn, for the software renderer's cap: a frame asked for sooner waits for its turn rather than
    // queueing behind the one being drawn.
    let lastDrawn = -Infinity;
    const renderFrame = (now: number) => {
      frameIdRef.current = 0;
      if (software && now - lastDrawn < 1000 / SOFTWARE_MAX_FPS) {
        requestFrame();
        return;
      }

      const deltaSeconds =
        lastFrameTimeRef.current == null ? null : (now - lastFrameTimeRef.current) / 1000;
      lastFrameTimeRef.current = now;

      const hadInvalidation = needsRenderRef.current;
      needsRenderRef.current = false;
      const controlsChanged = controls.update(deltaSeconds ?? undefined);
      if (continuousRenderRef.current || hadInvalidation || controlsChanged) {
        renderScene(deltaSeconds ?? undefined);
        lastDrawn = now;
      }

      if (continuousRenderRef.current || controlsChanged) {
        requestFrame();
        return;
      }

      lastFrameTimeRef.current = null;
    };

    const invalidate = () => {
      needsRenderRef.current = true;
      requestFrame();
    };

    invalidateRef.current = invalidate;
    controls.addEventListener('change', invalidate);
    invalidate();

    // --- ResizeObserver ---
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width === 0 || height === 0) continue;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
        invalidate();
      }
    });
    resizeObserver.observe(container);

    setSceneReady(true);

    // --- Cleanup ---
    return () => {
      setSceneReady(false);
      cancelAnimationFrame(frameIdRef.current);
      resizeObserver.disconnect();
      controls.removeEventListener('change', invalidate);
      controls.dispose();
      renderer.domElement.remove();
      environmentMapRef.current?.dispose();
      renderer.dispose();

      // Dispose scene contents.
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry?.dispose();
          const mat = object.material;
          if (Array.isArray(mat)) {
            mat.forEach((m) => m.dispose());
          } else {
            mat?.dispose();
          }
        }
      });

      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      gridGroupRef.current = null;
      axisGroupRef.current = null;
      lightingRigRef.current = null;
      environmentMapRef.current = null;
      lastFrameTimeRef.current = null;
      needsRenderRef.current = false;
      invalidateRef.current = () => {};
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const renderer = rendererRef.current;
    const container = containerRef.current;
    if (!renderer || !container) {
      return;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, options.maxPixelRatio ?? 2,
                                    rendererKind().software ? SOFTWARE_PIXEL_RATIO : Infinity));
    renderer.setSize(container.clientWidth, container.clientHeight, false);
    invalidateRef.current();
  }, [containerRef, options.maxPixelRatio, sceneReady]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const lightingRig = lightingRigRef.current;
    const container = containerRef.current;
    if (!renderer || !scene || !camera || !lightingRig || !container) {
      return;
    }

    const fancyGraphics = Boolean(options.fancyGraphics);
    fancyGraphicsRef.current = fancyGraphics;
    applyLook(renderer, scene, lightingRig, environmentMapRef, fancyGraphics);
    invalidateRef.current();
  }, [containerRef, options.fancyGraphics, sceneReady]);

  useEffect(() => {
    continuousRenderRef.current = Boolean(options.continuousRender);
    invalidateRef.current();
  }, [options.continuousRender]);

  return {
    scene: sceneRef.current,
    camera: cameraRef.current,
    renderer: rendererRef.current,
    controls: controlsRef.current,
    gridGroup: gridGroupRef.current,
    axisGroup: axisGroupRef.current,
    invalidate,
    sceneReady,
    fitSun,
  };
}
