// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import * as THREE from 'three';

export interface UrdfJoint {
  name: string;
  type: 'fixed' | 'revolute' | 'continuous' | 'prismatic' | 'floating' | 'planar';
  parent: string;
  child: string;
  origin?: {
    xyz?: [number, number, number];
    rpy?: [number, number, number];
  };
  axis?: [number, number, number];
  limit?: {
    lower?: number;
    upper?: number;
    effort?: number;
    velocity?: number;
  };
  mimic?: {
    joint: string;
    multiplier: number;
    offset: number;
  };
}

export interface UrdfVisualGeometry {
  type: 'box' | 'cylinder' | 'sphere' | 'mesh';
  size?: [number, number, number]; // box
  radius?: number; // cylinder, sphere
  length?: number; // cylinder
  filename?: string; // mesh
  scale?: [number, number, number]; // mesh
}

export interface UrdfVisual {
  name?: string;
  origin?: {
    xyz?: [number, number, number];
    rpy?: [number, number, number];
  };
  geometry: UrdfVisualGeometry;
  material?: {
    name?: string;
    color?: { rgba: [number, number, number, number] };
    /**
     * `scaleM` is dycraft's own attribute on the standard `<texture>` element:
     * the real-world edge length of one tile, in meters. URDF has nowhere to
     * put it, and an unknown attribute is skipped by every reader whereas an
     * unknown sibling element would not be. See `render/urdf_export.py`.
     */
    texture?: {
      filename: string;
      scaleM?: number;
      /**
       * Extra maps by kind, from each `<texture map="...">` element. A plain
       * record rather than named fields, because kinds are dycraft library
       * data — the exporter carries whatever the material has, and only
       * `createMaterial` knows what a given kind means to three.js.
       */
      maps?: Record<string, string>;
    };
    /**
     * The authored PBR scalars, out of `<material><pbr><metal>`.
     *
     * Gazebo's SDF schema embedded inline in URDF, which is what
     * `render/urdf_export.py` writes and what standard readers (urdfdom,
     * PyBullet, rviz) skip. Reading it here is what makes the viewport agree
     * with the canonical renders about how glossy a surface is: without it
     * every untextured part would fall back to one neutral default, so a
     * matte nylon strap (`roughness` 0.92) and a cut-crystal drop
     * (`roughness` 0.05) would draw identically.
     *
     * **Absent means the exporter suppressed it, NOT that nothing is known —
     * unless `shader` is set.** `urdf_export` emits the block only when a
     * field differs from the SDK default, to keep an untextured part's URDF
     * byte-identical to the pre-PBR exporter's — so no block means exactly
     * `metallic=0`, `roughness=0.9`. That reading is only sound because every
     * URDF this viewer loads is one dycraft generated: `viewer/app.py` serves
     * `model.urdf` from `runs.run_urdf` -> `render.urdf_export`, never a file
     * from disk. See `EXPORTED_DEFAULT_*` in `materials.ts`, which is where
     * the absent case is turned back into numbers.
     *
     * A shader material breaks the "always filled" premise the reading rests
     * on, which is why the fields are individually optional: `Material.to_ir`
     * omits the scalars its author did not write, so under a `shader` a
     * missing field is genuinely unknown and must stay `undefined` rather
     * than picking up the suppressed-default fallback.
     */
    pbr?: { metalness?: number; roughness?: number };
    /**
     * The `<shader name>` marker: this surface is computed by a Blender node
     * graph, not declared by scalars.
     *
     * Nothing here can run that graph — the body is not even carried — so the
     * marker's whole job is to stop `pbr`'s absence being read as authored
     * intent. Present means "ask the heuristics"; absent means the scalars are
     * the truth. See the marker's emission in `render/urdf_export.py`.
     */
    shader?: { name?: string };
  };
}

type UrdfMaterial = NonNullable<UrdfVisual['material']>;

export interface UrdfLink {
  name: string;
  visuals: UrdfVisual[];
  collisions: UrdfVisual[]; // Collision has same structure as visual
}

export interface UrdfSpec {
  name: string;
  links: UrdfLink[];
  joints: UrdfJoint[];
}

export interface UrdfVisualDescriptor {
  key: string;
  label: string;
  index: number;
}

/**
 * Parse a vec3 string like "1 2 3" or "1.0 2.0 3.0"
 */
export function parseVec3(str: string | undefined): [number, number, number] {
  if (!str) return [0, 0, 0];
  const parts = str.trim().split(/\s+/).map(parseFloat);
  return [
    Number.isFinite(parts[0]) ? parts[0] : 0,
    Number.isFinite(parts[1]) ? parts[1] : 0,
    Number.isFinite(parts[2]) ? parts[2] : 0,
  ];
}

/**
 * Parse a vec4 string like "1 0 0 1" for RGBA
 */
export function parseVec4(str: string | undefined): [number, number, number, number] {
  if (!str) return [1, 1, 1, 1];
  const parts = str.trim().split(/\s+/).map(parseFloat);
  return [
    Number.isFinite(parts[0]) ? parts[0] : 1,
    Number.isFinite(parts[1]) ? parts[1] : 1,
    Number.isFinite(parts[2]) ? parts[2] : 1,
    Number.isFinite(parts[3]) ? parts[3] : 1,
  ];
}

export function buildUrdfVisualKey(linkName: string, visualIndex: number): string {
  return `${linkName}::visual:${visualIndex}`;
}

function meshFilenameStem(filename: string | undefined): string | null {
  if (!filename) {
    return null;
  }

  const normalized = filename.split(/[?#]/, 1)[0] ?? '';
  const basename = normalized.split('/').at(-1)?.trim() ?? '';
  if (!basename) {
    return null;
  }

  const stem = basename.replace(/\.[^.]+$/, '');
  return stem || null;
}

function baseVisualLabel(visual: UrdfVisual, index: number): string {
  const explicitName = visual.name?.trim();
  if (explicitName) {
    return explicitName;
  }

  if (visual.geometry.type === 'mesh') {
    const stem = meshFilenameStem(visual.geometry.filename);
    if (stem) {
      return stem;
    }
  }

  return `${visual.geometry.type} ${index + 1}`;
}

export function describeLinkVisuals(link: UrdfLink): UrdfVisualDescriptor[] {
  const baseLabels = link.visuals.map((visual, index) => baseVisualLabel(visual, index));
  const labelTotals = new Map<string, number>();
  for (const label of baseLabels) {
    labelTotals.set(label, (labelTotals.get(label) ?? 0) + 1);
  }

  const seenLabels = new Map<string, number>();
  return baseLabels.map((baseLabel, index) => {
    const nextCount = (seenLabels.get(baseLabel) ?? 0) + 1;
    seenLabels.set(baseLabel, nextCount);

    return {
      key: buildUrdfVisualKey(link.name, index),
      label: (labelTotals.get(baseLabel) ?? 0) > 1 ? `${baseLabel} ${nextCount}` : baseLabel,
      index,
    };
  });
}

/**
 * Convert URDF fixed-axis RPY (roll, pitch, yaw) to THREE.Matrix4.
 * URDF's fixed-axis order: Rz(yaw) * Ry(pitch) * Rx(roll).
 */
export function rpyToMatrix4(rpy: [number, number, number]): THREE.Matrix4 {
  const [roll, pitch, yaw] = rpy;
  const mx = new THREE.Matrix4().makeRotationX(roll);
  const my = new THREE.Matrix4().makeRotationY(pitch);
  const mz = new THREE.Matrix4().makeRotationZ(yaw);
  return new THREE.Matrix4().multiplyMatrices(mz, new THREE.Matrix4().multiplyMatrices(my, mx));
}

/**
 * Convert URDF origin (xyz + rpy) to THREE.Matrix4
 */
export function originToMatrix4(origin?: { xyz?: [number, number, number]; rpy?: [number, number, number] }): THREE.Matrix4 {
  const xyz = origin?.xyz ?? [0, 0, 0];
  const rpy = origin?.rpy ?? [0, 0, 0];
  const translation = new THREE.Matrix4().makeTranslation(xyz[0], xyz[1], xyz[2]);
  const rotation = rpyToMatrix4(rpy);
  return new THREE.Matrix4().multiplyMatrices(translation, rotation);
}

/**
 * Parse <origin> element
 */
function parseOrigin(originEl: Element | null): { xyz?: [number, number, number]; rpy?: [number, number, number] } | undefined {
  if (!originEl) return undefined;

  const xyzStr = originEl.getAttribute('xyz') ?? undefined;
  const rpyStr = originEl.getAttribute('rpy') ?? undefined;

  const result: { xyz?: [number, number, number]; rpy?: [number, number, number] } = {};

  if (xyzStr) result.xyz = parseVec3(xyzStr);
  if (rpyStr) result.rpy = parseVec3(rpyStr);

  return Object.keys(result).length > 0 ? result : undefined;
}

/**
 * Parse <geometry> element
 */
function parseGeometry(geomEl: Element | null): UrdfVisualGeometry | null {
  if (!geomEl) return null;

  const boxEl = geomEl.querySelector('box');
  if (boxEl) {
    const sizeStr = boxEl.getAttribute('size') ?? undefined;
    return {
      type: 'box',
      size: parseVec3(sizeStr),
    };
  }

  const cylinderEl = geomEl.querySelector('cylinder');
  if (cylinderEl) {
    return {
      type: 'cylinder',
      radius: parseFloat(cylinderEl.getAttribute('radius') || '1'),
      length: parseFloat(cylinderEl.getAttribute('length') || '1'),
    };
  }

  const sphereEl = geomEl.querySelector('sphere');
  if (sphereEl) {
    return {
      type: 'sphere',
      radius: parseFloat(sphereEl.getAttribute('radius') || '1'),
    };
  }

  const meshEl = geomEl.querySelector('mesh');
  if (meshEl) {
    const scaleStr = meshEl.getAttribute('scale') ?? undefined;
    return {
      type: 'mesh',
      filename: meshEl.getAttribute('filename') || '',
      scale: scaleStr ? parseVec3(scaleStr) : undefined,
    };
  }

  return null;
}

/**
 * Parse <material> element
 */
function parseMaterial(matEl: Element | null): UrdfVisual['material'] | undefined {
  if (!matEl) return undefined;

  const name = matEl.getAttribute('name') || undefined;
  const colorEl = matEl.querySelector('color');
  // The base-color element specifically: the one with no `map` attribute.
  // `querySelector('texture')` alone would take whichever comes first, which is
  // the base color today and would silently become roughness if the exporter
  // ever reordered them.
  const textureEl = matEl.querySelector('texture:not([map])')
    ?? matEl.querySelector('texture');

  const material: UrdfVisual['material'] = { name };

  if (colorEl) {
    const rgbaStr = colorEl.getAttribute('rgba') ?? undefined;
    material.color = { rgba: parseVec4(rgbaStr) };
  }

  // Every <texture> in the element, keyed by its `map` attribute. The
  // base-color one carries none, because standard URDF's <texture> IS the
  // diffuse — so the selector above finds the right element even in a parser
  // that knows nothing about the extras.
  const extraMaps: Record<string, string> = {};
  for (const el of matEl.querySelectorAll('texture')) {
    const kind = el.getAttribute('map');
    const fn = el.getAttribute('filename');
    if (kind && fn) extraMaps[kind] = fn;
  }

  // The `<shader name>` marker, read before `<pbr>` because it decides how an
  // absent or partial block is to be understood.
  const shaderEl = matEl.querySelector('shader');
  if (shaderEl) {
    material.shader = { name: shaderEl.getAttribute('name') || undefined };
  }

  // `<pbr><metal><metalness>m</metalness><roughness>r</roughness></metal>`.
  // Read per field rather than as a pair, so a block carrying only one of them
  // contributes that one instead of being discarded whole.
  const metalEl = matEl.querySelector('pbr metal');
  if (metalEl) {
    const num = (tag: string): number | undefined => {
      const el = metalEl.querySelector(tag);
      if (!el) return undefined;
      const v = Number.parseFloat(el.textContent ?? '');
      return Number.isFinite(v) ? v : undefined;
    };
    const metalness = num('metalness');
    const roughness = num('roughness');
    if (metalness !== undefined || roughness !== undefined) {
      // The per-field fallback is the same "what the exporter suppressed"
      // value the absent-block case uses; it is spelled in `materials.ts` and
      // duplicated here only because this layer has no import from it.
      //
      // It is skipped under a shader: there the exporter emits exactly the
      // tags whose values the author wrote, so filling in the other one would
      // invent the very claim the marker exists to prevent.
      material.pbr = material.shader
        ? { metalness, roughness }
        : { metalness: metalness ?? 0.0, roughness: roughness ?? 0.9 };
    }
  }

  if (textureEl) {
    const rawScale = textureEl.getAttribute('scale_m');
    const scaleM = rawScale === null ? undefined : Number.parseFloat(rawScale);
    material.texture = {
      filename: textureEl.getAttribute('filename') || '',
      maps: extraMaps,
      // A missing or unparseable scale leaves it undefined rather than
      // defaulting: the caller has to decide, and silently picking a tile size
      // would put the viewport's grain at a different size from the canonical
      // render, which is the one failure this whole path must not have.
      scaleM: scaleM !== undefined && Number.isFinite(scaleM) && scaleM > 0
        ? scaleM
        : undefined,
    };
  }

  return material;
}

function resolveMaterial(
  material: UrdfVisual['material'],
  materialLibrary: Map<string, UrdfMaterial>,
): UrdfVisual['material'] {
  if (!material?.name) {
    return material;
  }

  const libraryMaterial = materialLibrary.get(material.name);
  if (!libraryMaterial) {
    return material;
  }

  return {
    ...libraryMaterial,
    ...material,
    color: material.color ?? libraryMaterial.color,
    texture: material.texture ?? libraryMaterial.texture,
    pbr: material.pbr ?? libraryMaterial.pbr,
    // Carried with `pbr` rather than beside it: a visual that inherits its
    // scalars from the library entry must inherit how to read them, or the
    // absent-means-default rule comes back on a shader surface.
    shader: material.shader ?? libraryMaterial.shader,
  };
}

/**
 * Parse <visual> or <collision> element
 */
function parseVisual(visualEl: Element, materialLibrary: Map<string, UrdfMaterial>): UrdfVisual {
  const name = visualEl.getAttribute('name') || undefined;
  const originEl = visualEl.querySelector('origin');
  const geomEl = visualEl.querySelector('geometry');
  const matEl = visualEl.querySelector('material');

  const geometry = parseGeometry(geomEl);
  if (!geometry) {
    throw new Error('Visual element missing valid geometry');
  }

  return {
    name,
    origin: parseOrigin(originEl),
    geometry,
    material: resolveMaterial(parseMaterial(matEl), materialLibrary),
  };
}

/**
 * Parse <link> element
 */
function parseLink(linkEl: Element, materialLibrary: Map<string, UrdfMaterial>): UrdfLink {
  const name = linkEl.getAttribute('name') || '';

  const visuals: UrdfVisual[] = [];
  for (const visualEl of linkEl.querySelectorAll(':scope > visual')) {
    visuals.push(parseVisual(visualEl, materialLibrary));
  }

  const collisions: UrdfVisual[] = [];
  for (const collisionEl of linkEl.querySelectorAll(':scope > collision')) {
    collisions.push(parseVisual(collisionEl, materialLibrary));
  }

  return { name, visuals, collisions };
}

/**
 * Parse <joint> element
 */
function parseJoint(jointEl: Element): UrdfJoint {
  const name = jointEl.getAttribute('name') || '';
  const type = (jointEl.getAttribute('type') || 'fixed') as UrdfJoint['type'];

  const parentEl = jointEl.querySelector('parent');
  const childEl = jointEl.querySelector('child');

  const parent = parentEl?.getAttribute('link') || '';
  const child = childEl?.getAttribute('link') || '';

  const originEl = jointEl.querySelector('origin');
  const axisEl = jointEl.querySelector('axis');
  const limitEl = jointEl.querySelector('limit');
  const mimicEl = jointEl.querySelector('mimic');

  const joint: UrdfJoint = {
    name,
    type,
    parent,
    child,
  };

  if (originEl) {
    joint.origin = parseOrigin(originEl);
  }

  if (axisEl) {
    const xyzStr = axisEl.getAttribute('xyz') ?? undefined;
    joint.axis = parseVec3(xyzStr);
  }

  if (limitEl) {
    joint.limit = {
      lower: parseFloat(limitEl.getAttribute('lower') || '0'),
      upper: parseFloat(limitEl.getAttribute('upper') || '0'),
      effort: parseFloat(limitEl.getAttribute('effort') || '0'),
      velocity: parseFloat(limitEl.getAttribute('velocity') || '0'),
    };
  }

  if (mimicEl) {
    const mimicJoint = mimicEl.getAttribute('joint') || '';
    if (mimicJoint) {
      joint.mimic = {
        joint: mimicJoint,
        multiplier: parseFloat(mimicEl.getAttribute('multiplier') || '1'),
        offset: parseFloat(mimicEl.getAttribute('offset') || '0'),
      };
    }
  }

  return joint;
}

/**
 * Rewrite mesh filenames from absolute package:// paths to repo-relative paths
 * e.g., "package://robot/assets/meshes/link.dae" -> "assets/meshes/link.dae"
 */
export function rewriteAbsoluteMeshFilenames(urdfSpec: UrdfSpec): UrdfSpec {
  const rewrite = (filename: string): string => {
    if (filename.startsWith('package://')) {
      // Extract everything after the first package name
      const match = filename.match(/^package:\/\/[^/]+\/(.+)$/);
      return match ? match[1] : filename;
    }
    return filename;
  };

  const rewriteVisuals = (visuals: UrdfVisual[]): UrdfVisual[] => {
    return visuals.map(v => {
      if (v.geometry.type === 'mesh' && v.geometry.filename) {
        return {
          ...v,
          geometry: {
            ...v.geometry,
            filename: rewrite(v.geometry.filename),
          },
        };
      }
      return v;
    });
  };

  return {
    ...urdfSpec,
    links: urdfSpec.links.map(link => ({
      ...link,
      visuals: rewriteVisuals(link.visuals),
      collisions: rewriteVisuals(link.collisions),
    })),
  };
}

/**
 * Find the root link (link that is not a child of any joint)
 */
export function findRootLink(urdfSpec: UrdfSpec): string | null {
  const childLinks = new Set(urdfSpec.joints.map(j => j.child));
  const rootLink = urdfSpec.links.find(link => !childLinks.has(link.name));
  return rootLink?.name || null;
}

/**
 * Parse URDF XML string into structured spec
 */
export function parseUrdf(urdfXml: string): UrdfSpec {
  const parser = new DOMParser();
  const doc = parser.parseFromString(urdfXml, 'application/xml');

  const robotEl = doc.querySelector('robot');
  if (!robotEl) {
    throw new Error('URDF missing <robot> root element');
  }

  const name = robotEl.getAttribute('name') || 'robot';
  const materialLibrary = new Map<string, UrdfMaterial>();

  for (const materialEl of robotEl.querySelectorAll(':scope > material')) {
    const material = parseMaterial(materialEl);
    if (!material?.name) {
      continue;
    }
    materialLibrary.set(material.name, material);
  }

  const links: UrdfLink[] = [];
  for (const linkEl of robotEl.querySelectorAll(':scope > link')) {
    links.push(parseLink(linkEl, materialLibrary));
  }

  const joints: UrdfJoint[] = [];
  for (const jointEl of robotEl.querySelectorAll(':scope > joint')) {
    joints.push(parseJoint(jointEl));
  }

  return { name, links, joints };
}
