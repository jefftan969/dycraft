import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

/**
 * Create the standard lighting rig for the URDF viewer scene.
 * Includes hemisphere ambient light, key directional light with shadows,
 * a fill light, and a rim light.
 */
export function createLightingRig(): THREE.Group {
  const group = new THREE.Group();
  group.name = 'lighting-rig';

  // Bright hemisphere light for strong ambient illumination (studio style)
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0xd4d4d4, 0.95);
  hemiLight.name = 'hemisphere-light';
  group.add(hemiLight);

  // Key directional light with softer shadows
  const keyLight = new THREE.DirectionalLight(0xfff5e6, 0.75);
  keyLight.name = 'key-light';
  keyLight.position.set(6, 12, 8);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.1;
  keyLight.shadow.camera.far = 50;
  keyLight.shadow.camera.left = -10;
  keyLight.shadow.camera.right = 10;
  keyLight.shadow.camera.top = 10;
  keyLight.shadow.camera.bottom = -10;
  group.add(keyLight);

  // Strong fill light from opposite side (neutral, bright)
  const fillLight = new THREE.DirectionalLight(0xf0f4ff, 0.65);
  fillLight.name = 'fill-light';
  fillLight.position.set(-6, 5, -6);
  group.add(fillLight);

  // Rim light (behind, for edge definition)
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.35);
  rimLight.name = 'rim-light';
  rimLight.position.set(0, -6, 10);
  group.add(rimLight);

  // Strong frontal fill to keep front-facing surfaces bright
  const frontFillLight = new THREE.DirectionalLight(0xfff9f0, 0.5);
  frontFillLight.name = 'front-fill-light';
  frontFillLight.position.set(0, 6, 10);
  group.add(frontFillLight);

  return group;
}

/**
 * Create a fine-grained ground-plane grid helper.
 * The grid is centered at the origin and can be repositioned later.
 */
export function createGridHelper(): THREE.Group {
  const group = new THREE.Group();
  group.name = 'grid-group';

  // Fine grid (small subdivisions) - subtle like Figma's canvas grid
  const fineGrid = new THREE.GridHelper(10, 100, 0xc0c0c0, 0xd0d0d0);
  fineGrid.name = 'grid-fine';
  const fineMat = fineGrid.material;
  if (Array.isArray(fineMat)) {
    for (const m of fineMat) {
      m.transparent = true;
      m.opacity = 0.45;
    }
  } else {
    fineMat.transparent = true;
    fineMat.opacity = 0.45;
  }
  group.add(fineGrid);

  // Coarse grid (major lines every 1 unit) - slightly stronger for reference
  const coarseGrid = new THREE.GridHelper(10, 10, 0x999999, 0x999999);
  coarseGrid.name = 'grid-coarse';
  const coarseMat = coarseGrid.material;
  if (Array.isArray(coarseMat)) {
    for (const m of coarseMat) {
      m.transparent = true;
      m.opacity = 0.35;
    }
  } else {
    coarseMat.transparent = true;
    coarseMat.opacity = 0.35;
  }
  group.add(coarseGrid);

  return group;
}

/**
 * Create colored XYZ axis lines.
 * X = red, Y = green, Z = blue.
 */
export function createAxisHelper(size: number = 2): THREE.Group {
  const group = new THREE.Group();
  group.name = 'axis-helper';

  const axes: Array<{ dir: THREE.Vector3; color: number }> = [
    { dir: new THREE.Vector3(1, 0, 0), color: 0xe74c3c }, // X = red
    { dir: new THREE.Vector3(0, 1, 0), color: 0x2ecc71 }, // Y = green
    { dir: new THREE.Vector3(0, 0, 1), color: 0x3498db }, // Z = blue
  ];

  for (const { dir, color } of axes) {
    const points = [
      new THREE.Vector3(0, 0, 0),
      dir.clone().multiplyScalar(size),
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    const line = new THREE.Line(geometry, material);
    group.add(line);
  }

  return group;
}

/**
 * Keep the ground helpers at the normalized scene origin and scale them to the model footprint.
 */
export function positionGroundHelpers(
  gridGroup: THREE.Group,
  axisGroup: THREE.Group,
  robotGroup: THREE.Object3D,
): void {
  const box = new THREE.Box3().setFromObject(robotGroup);
  const size = box.getSize(new THREE.Vector3());

  gridGroup.position.set(0, 0, 0);

  // Scale grid to fit the model - at least 3x the model's footprint
  const footprint = Math.max(size.x, size.z, 1) * 3;
  const gridScale = footprint / 10; // base grid is 10 units
  gridGroup.scale.set(gridScale, 1, gridScale);

  axisGroup.position.set(0, 0, 0);

  // Scale axis to be proportional to the model size
  const axisScale = Math.max(size.x, size.y, size.z, 0.5) * 0.4;
  axisGroup.scale.setScalar(axisScale);
}

/**
 * Create a simple studio-style environment map using PMREMGenerator.
 * Provides subtle reflections on metallic/glossy materials.
 */
export function createEnvironmentMap(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  pmremGenerator.compileEquirectangularShader();

  // Build a simple gradient scene to use as an environment map
  const envScene = new THREE.Scene();

  // Top hemisphere color
  const topColor = new THREE.Color(0xf7f8fa);
  // Bottom hemisphere color
  const bottomColor = new THREE.Color(0xd9dde3);
  // Equator accent
  const midColor = new THREE.Color(0xffffff);

  // Use a large sphere with a gradient shader as environment
  const envGeometry = new THREE.SphereGeometry(100, 32, 16);
  const envMaterial = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: {
      topColor: { value: topColor },
      midColor: { value: midColor },
      bottomColor: { value: bottomColor },
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorldPosition;
      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 topColor;
      uniform vec3 midColor;
      uniform vec3 bottomColor;
      varying vec3 vWorldPosition;
      void main() {
        float h = normalize(vWorldPosition).y;
        vec3 color;
        if (h > 0.0) {
          color = mix(midColor, topColor, h);
        } else {
          color = mix(midColor, bottomColor, -h);
        }
        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });

  const envMesh = new THREE.Mesh(envGeometry, envMaterial);
  envScene.add(envMesh);

  const envMap = pmremGenerator.fromScene(envScene, 0.04).texture;
  pmremGenerator.dispose();
  envGeometry.dispose();
  envMaterial.dispose();

  return envMap;
}

/**
 * The project page's look, which the viewport's fancy graphics now are: ACES filmic tone mapping, a room as the light
 * probe at a little over half strength, a cool hemisphere, and one sun from the upper left whose soft shadow makes the
 * scene read as solid. Scene-agnostic: the sun's shadow is fitted to whatever is drawn (`fitSunToBox`).
 */
export const FILM_LOOK = {
  environmentIntensity: 0.55,
  hemisphere: { sky: 0xffffff, ground: 0xb8c2d0, intensity: 0.55 },
  sun: { color: 0xffffff, intensity: 2.1, direction: new THREE.Vector3(-4, 9, 5).normalize() },
} as const;

/** The project page's light probe: three's room, prefiltered once. */
export function createRoomEnvironmentMap(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const texture = pmrem.fromScene(room, 0.04).texture;
  room.dispose();
  pmrem.dispose();
  return texture;
}

/**
 * Aim the sun at the middle of `box` (the drawn scene, viewport coordinates) and fit its shadow to it: an orthographic
 * shadow camera just covering the scene's bounding sphere, so a mug and a hardware store both get a sharp shadow from
 * one map. The light's direction stays `FILM_LOOK.sun.direction`.
 */
export function fitSunToBox(rig: THREE.Group, box: THREE.Box3): void {
  const sun = rig.getObjectByName('key-light');
  if (!(sun instanceof THREE.DirectionalLight) || box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.05);
  sun.position.copy(center).addScaledVector(FILM_LOOK.sun.direction, radius * 2);
  sun.target.position.copy(center);
  sun.target.updateMatrixWorld();
  const camera = sun.shadow.camera;
  camera.left = -radius;
  camera.right = radius;
  camera.top = radius;
  camera.bottom = -radius;
  camera.near = radius * 0.5;
  camera.far = radius * 3.5;
  camera.updateProjectionMatrix();
  // Depth bias in proportion to a shadow texel, so neither acne on a big room nor peter-panning on a small part.
  const texel = (2 * radius) / sun.shadow.mapSize.width;
  sun.shadow.normalBias = texel * 1.5;
  sun.shadow.bias = -texel * 0.05;
  sun.shadow.needsUpdate = true;
}
