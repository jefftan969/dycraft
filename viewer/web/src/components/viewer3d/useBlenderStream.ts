import { useEffect, useRef, useState, type RefObject } from 'react';
import * as THREE from 'three';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { encodeRunId } from '@/lib/url';

/**
 * Stream Cycles frames for whatever the viewport is currently showing.
 *
 * The viewer cannot draw a `Material(shader=...)`: the appearance is a Blender
 * node graph, and shaders often drive from *object-space* coordinates, which a
 * tileable bake misplaces rather than blurs. So instead of approximating the
 * material, this asks Blender for the frame — the same `_build_material` /
 * `_run_shader` the canonical render uses. At the same camera and pose the
 * streamed frame differs from `blender_renderer.render()` by little more than
 * JPEG-encoding the canonical PNG does.
 *
 * **Three.js keeps owning everything.** The camera, the controls, picking,
 * the joint pose, Preview Motion, and clip playback are all untouched and stay
 * live underneath; this hook only *reads* their state and covers the canvas
 * with the answer. The overlay is `pointer-events: none`, so every gesture
 * still lands on the WebGL canvas exactly as before.
 *
 * Two things are load-bearing and easy to get wrong:
 *
 *   * **The change of basis.** The robot group carries the z-up rotation and
 *     the ground-normalizing offset, so a three.js camera position is not a
 *     dycraft world position. Inverting `robotGroup.matrixWorld` is what makes
 *     the streamed frame land on the viewport without a framing jump; the same
 *     conversion is already used the other way for `fitBounds`.
 *   * **Vertical FOV.** `THREE.PerspectiveCamera.fov` is vertical, and so is
 *     `CameraSpec.fov_deg` (the child sets `sensor_fit = "VERTICAL"`). Match
 *     that and the aspect ratio and the two framings are identical.
 */

const PROTOCOL_VERSION = 1;

/** Resolution ladder, as a fraction of the canvas. Only pixels change here —
 *  never the material, the sample count, or the view transform, so a tier
 *  switch cannot alter what the object looks like, only how sharp it is. */
const SCALE_LADDER = [1.0, 0.75, 0.55, 0.4, 0.28, 0.2] as const;
/** Where interaction starts, so the first drag is already responsive. */
const START_TIER = 2;
/** Target round-trip for an interactive frame. 45 ms is ~22 fps, which is the
 *  measured 640x360 rate on an RTX 3090 with the scene held resident. */
const TARGET_MS = 45;
const TIER_DOWN_MS = TARGET_MS * 1.3;
const TIER_UP_MS = TARGET_MS * 0.55;
/** Consecutive fast/slow frames before the tier moves, so a single hitch does
 *  not cost resolution and a single fast frame does not cost frame rate. */
const TIER_HYSTERESIS = 3;
/** Quiet time before the full-resolution frame is requested. */
const SETTLE_MS = 200;
const MIN_SIDE = 16;
const MAX_SIDE = 1920;

export type BlenderStreamStatus = 'idle' | 'connecting' | 'building' | 'live' | 'error';

export interface BlenderStreamState {
  status: BlenderStreamStatus;
  /** Why it is unavailable. Shown to the user — never silently swallowed, the
   *  alternative being a viewport that looks right and is not. */
  error: string | null;
  /** True once a frame is on screen, i.e. once it is safe to cover the canvas. */
  hasFrame: boolean;
  device: string | null;
  /** Rounded round-trip of the last interactive frame, for the status chip. */
  frameMs: number | null;
  /** Current streamed pixel height, so the chip can say "540p". */
  frameHeight: number | null;
  /** Non-fatal shader warnings the child reported, deduplicated. */
  warnings: string[];
}

const IDLE: BlenderStreamState = {
  status: 'idle', error: null, hasFrame: false, device: null,
  frameMs: null, frameHeight: null, warnings: [],
};

interface StreamInputs {
  enabled: boolean;
  runId: string | null;
  camera: THREE.PerspectiveCamera | null;
  controls: OrbitControls | null;
  scene: THREE.Scene | null;
  robotGroupName: string;
  /** Live posed joint values — see `useJointController.posedJointValues`. */
  posedJointValues: RefObject<Map<string, number>>;
  /** Where to draw. Sized to each frame; CSS stretches it to the viewport. */
  canvasRef: RefObject<HTMLCanvasElement | null>;
  /** Bumped by the caller when the model reloads, to force a fresh session. */
  modelKey: string | null;
}

function webSocketUrl(runId: string): string {
  const base = import.meta.env.VITE_API_BASE ?? '';
  // `encodeRunId`, not `encodeURIComponent`: a run id is a runs-dir-relative
  // path, and collapsing its separators into `%2F` would make every run inside
  // a folder, and every sub-run, unreachable.
  const absolute = new URL(
    `${base}/api/runs/${encodeRunId(runId)}/live`,
    window.location.href,
  );
  absolute.protocol = absolute.protocol === 'https:' ? 'wss:' : 'ws:';
  return absolute.toString();
}

function clampSide(value: number): number {
  return Math.max(MIN_SIDE, Math.min(MAX_SIDE, Math.round(value)));
}

export interface DycraftCamera {
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fovDeg: number;
}

interface ViewerState extends DycraftCamera {
  clientWidth: number;
  clientHeight: number;
  joints: Record<string, number>;
  signature: string;
}

const _eye = new THREE.Vector3();
const _target = new THREE.Vector3();
const _up = new THREE.Vector3();
const _inverse = new THREE.Matrix4();
const _normal = new THREE.Matrix3();

function triple(v: THREE.Vector3): [number, number, number] {
  return [v.x, v.y, v.z];
}

/** Enough digits that a real move is never mistaken for noise, few enough that
 *  float jitter in the controls' damping does not look like a move. */
function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/**
 * The three.js camera expressed in dycraft world axes (meters, +Z up).
 *
 * `robot` is the group the URDF loader builds, and it carries BOTH the z-up
 * rotation and the ground-normalizing offset — so a three.js camera position
 * is not a dycraft position and cannot be sent as one. Inverting its
 * `matrixWorld` is exactly the change of basis, which is also how
 * `SceneCanvas` converts a dycraft AABB the other way for `fitBounds`.
 *
 * Exported and DOM-free so `npm run check:live-camera` can pin it: a sign
 * error here is silent, producing a plausible render of the wrong viewpoint
 * that would only show up as the streamed frame jumping when it replaces the
 * preview.
 */
export function cameraToDycraftWorld(
  camera: THREE.PerspectiveCamera,
  controlsTarget: THREE.Vector3,
  robot: THREE.Object3D,
): DycraftCamera {
  robot.updateMatrixWorld(true);
  camera.updateMatrixWorld();
  _inverse.copy(robot.matrixWorld).invert();
  _eye.copy(camera.position).applyMatrix4(_inverse);
  _target.copy(controlsTarget).applyMatrix4(_inverse);
  // The camera's **actual** up basis, not `camera.up`.
  //
  // For an ordinary orbit the two agree, because OrbitControls never rolls the
  // camera. They come apart looking straight down, where `camera.up` becomes
  // parallel to the view direction — and `_aim_camera` builds its basis as
  // `right = forward x up`, so a parallel `up` lands on its degenerate
  // fallback and picks an arbitrary image rotation. Reading column 1 of the
  // world matrix is always perpendicular to the view direction, so the
  // streamed frame keeps the viewport's orientation from every angle.
  _up.setFromMatrixColumn(camera.matrixWorld, 1);
  // A direction, so the rotation applies and the translation must not.
  _normal.setFromMatrix4(_inverse);
  _up.applyMatrix3(_normal).normalize();
  return {
    eye: triple(_eye).map(round6) as [number, number, number],
    target: triple(_target).map(round6) as [number, number, number],
    up: triple(_up).map(round6) as [number, number, number],
    // `THREE.PerspectiveCamera.fov` is the VERTICAL field of view, and so is
    // `CameraSpec.fov_deg` (the child sets `sensor_fit = "VERTICAL"`). No
    // conversion — but also no aspect correction, which is why the caller must
    // request a frame with the canvas's aspect ratio.
    fovDeg: round6(camera.fov),
  };
}

function readViewerState(inputs: StreamInputs): ViewerState | null {
  const { camera, controls, scene, robotGroupName, posedJointValues } = inputs;
  const canvas = inputs.canvasRef.current;
  if (!camera || !controls || !scene || !canvas) return null;
  const robot = scene.getObjectByName(robotGroupName);
  if (!robot) return null;
  const parent = canvas.parentElement;
  const clientWidth = parent?.clientWidth ?? 0;
  const clientHeight = parent?.clientHeight ?? 0;
  if (clientWidth < 1 || clientHeight < 1) return null;

  const view = cameraToDycraftWorld(camera, controls.target, robot);

  const joints: Record<string, number> = {};
  const posed = posedJointValues.current;
  if (posed) {
    for (const [name, value] of posed) {
      if (Number.isFinite(value)) joints[name] = round6(value);
    }
  }

  const signature = JSON.stringify([
    view.eye, view.target, view.up, view.fovDeg, clientWidth, clientHeight,
    Object.keys(joints).sort().map((k) => [k, joints[k]]),
  ]);
  return { ...view, clientWidth, clientHeight, joints, signature };
}

export function useBlenderStream(inputs: StreamInputs): BlenderStreamState {
  const [state, setState] = useState<BlenderStreamState>(IDLE);
  // Every mutable input the driver loop reads, so the loop is installed once
  // per session rather than restarted whenever the camera object changes.
  const inputsRef = useRef(inputs);
  inputsRef.current = inputs;

  const { enabled, runId, modelKey } = inputs;
  const ready = enabled && Boolean(runId) && Boolean(inputs.camera)
    && Boolean(inputs.controls) && Boolean(inputs.scene);

  useEffect(() => {
    if (!ready || !runId) {
      setState(IDLE);
      return;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let frameId = 0;

    // --- driver state ---
    let seq = 0;
    let tier = START_TIER;
    let fastRun = 0;
    let slowRun = 0;
    let inflight: { seq: number; signature: string; tier: string; sentAt: number } | null = null;
    let displayedSignature: string | null = null;
    let currentSignature: string | null = null;
    let lastChangeAt = 0;
    let settledSignature: string | null = null;
    const warnings = new Set<string>();

    const patch = (next: Partial<BlenderStreamState>) => {
      if (!disposed) setState((prev) => ({ ...prev, ...next }));
    };

    const send = (view: ViewerState, wantSettled: boolean) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      const scale = wantSettled ? 1.0 : SCALE_LADDER[tier];
      const height = clampSide(view.clientHeight * scale);
      // Width from the height and the exact aspect rather than from its own
      // rounding: the canvas is stretched to the container, so an aspect
      // mismatch here would show up as a squashed object rather than a soft one.
      const width = clampSide(height * (view.clientWidth / view.clientHeight));
      seq += 1;
      inflight = {
        seq, signature: view.signature,
        tier: wantSettled ? 'settled' : 'interactive',
        sentAt: performance.now(),
      };
      socket.send(JSON.stringify({
        protocol: PROTOCOL_VERSION, seq, tier: inflight.tier,
        width, height,
        camera: { eye: view.eye, target: view.target, up: view.up,
                  fov_deg: view.fovDeg },
        joints: view.joints,
      }));
    };

    const tick = () => {
      frameId = requestAnimationFrame(tick);
      if (disposed || !socket || socket.readyState !== WebSocket.OPEN) return;
      const view = readViewerState(inputsRef.current);
      if (!view) return;

      if (view.signature !== currentSignature) {
        currentSignature = view.signature;
        lastChangeAt = performance.now();
        // A settled frame for the *previous* state is now worthless.
        settledSignature = null;
      }
      if (inflight) return;

      if (view.signature !== displayedSignature) {
        send(view, false);
        return;
      }
      // Nothing has moved and the frame on screen is for this exact state:
      // spend the idle time replacing it with a full-resolution one, once.
      if (settledSignature !== view.signature
          && performance.now() - lastChangeAt > SETTLE_MS) {
        settledSignature = view.signature;
        send(view, true);
      }
    };

    const draw = async (blob: Blob, width: number, height: number) => {
      const canvas = inputsRef.current.canvasRef.current;
      if (!canvas) return;
      let bitmap: ImageBitmap | null = null;
      try {
        bitmap = await createImageBitmap(blob);
      } catch {
        return;
      }
      if (disposed) { bitmap.close(); return; }
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
    };

    const onBinary = async (buffer: ArrayBuffer) => {
      const view = new DataView(buffer);
      const headerLength = view.getUint32(0);
      const header = JSON.parse(
        new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)),
      ) as { seq: number; tier: string; width: number; height: number;
             render_ms: number; warnings?: string[] };
      const jpeg = new Blob([new Uint8Array(buffer, 4 + headerLength)],
                            { type: 'image/jpeg' });
      const sent = inflight;
      inflight = null;
      if (!sent || sent.seq !== header.seq) return;

      const elapsed = performance.now() - sent.sentAt;
      if (sent.tier === 'interactive') {
        // Adapt the ladder to what this machine actually delivers. Hysteresis
        // both ways: one slow frame is a hitch, three is a trend.
        if (elapsed > TIER_DOWN_MS) {
          slowRun += 1; fastRun = 0;
          if (slowRun >= TIER_HYSTERESIS && tier < SCALE_LADDER.length - 1) {
            tier += 1; slowRun = 0;
          }
        } else if (elapsed < TIER_UP_MS) {
          fastRun += 1; slowRun = 0;
          if (fastRun >= TIER_HYSTERESIS && tier > 0) { tier -= 1; fastRun = 0; }
        } else {
          fastRun = 0; slowRun = 0;
        }
      } else if (sent.signature !== currentSignature) {
        // **A settled frame that the world moved out from under is discarded.**
        // It is a full-resolution picture of a camera the user has already left,
        // and painting it would be a visible jump backwards. An *interactive*
        // frame in the same position is still the newest thing there is, so it
        // is shown: one frame of lag reads as a stream, not as a glitch.
        return;
      }

      displayedSignature = sent.signature;
      await draw(jpeg, header.width, header.height);
      for (const warning of header.warnings ?? []) warnings.add(warning);
      patch({
        status: 'live', hasFrame: true, error: null,
        frameMs: Math.round(elapsed),
        frameHeight: header.height,
        warnings: [...warnings],
      });
    };

    setState({ ...IDLE, status: 'connecting' });
    try {
      socket = new WebSocket(webSocketUrl(runId));
    } catch (error) {
      setState({ ...IDLE, status: 'error', error: String(error) });
      return;
    }
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
      patch({ status: 'building' });
      const view = readViewerState(inputsRef.current);
      // The hello doubles as the first frame request: the child needs a camera
      // to build against anyway, and asking for the current one means the first
      // picture is already the right picture.
      socket?.send(JSON.stringify({
        protocol: PROTOCOL_VERSION, seq: 0, tier: 'interactive',
        width: 640, height: 360,
        camera: view
          ? { eye: view.eye, target: view.target, up: view.up,
              fov_deg: view.fovDeg }
          : { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov_deg: 45 },
        joints: view?.joints ?? {},
      }));
      if (view) {
        currentSignature = view.signature;
        lastChangeAt = performance.now();
        inflight = { seq: 0, signature: view.signature, tier: 'interactive',
                     sentAt: performance.now() };
      }
      frameId = requestAnimationFrame(tick);
    };

    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        void onBinary(event.data);
        return;
      }
      const message = JSON.parse(String(event.data)) as {
        type: string; message?: string; fatal?: boolean;
        device?: string; warnings?: string[];
      };
      if (message.type === 'ready') {
        for (const warning of message.warnings ?? []) warnings.add(warning);
        patch({ status: 'live', device: message.device ?? null,
                warnings: [...warnings] });
        return;
      }
      if (message.type === 'error') {
        if (message.fatal) {
          patch({ status: 'error', error: message.message ?? 'unknown error' });
          socket?.close();
        } else {
          // A single refused frame. Reported, but the session lives.
          inflight = null;
          patch({ error: message.message ?? null });
        }
      }
    };

    socket.onerror = () => {
      patch({
        status: 'error',
        error: 'the live Blender viewport connection failed',
      });
    };

    socket.onclose = () => {
      if (disposed) return;
      // Only an *unannounced* close is an error; a fatal message already said
      // why and set the state.
      setState((prev) => prev.status === 'error'
        ? prev
        : { ...prev, status: 'error', hasFrame: false,
            error: prev.error ?? 'the live Blender viewport disconnected' });
    };

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      if (socket && socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'close' }));
        } catch {
          // Already gone; the close below is what matters.
        }
      }
      socket?.close();
      socket = null;
    };
    // `modelKey` restarts the session when the geometry underneath changes: the
    // resident scene was built from the old IR and cannot be re-pointed.
  }, [ready, runId, modelKey]);

  return state;
}
