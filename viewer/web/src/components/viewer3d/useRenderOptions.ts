// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import { rendererKind } from './software-renderer';
import { useState, useCallback, useEffect, useRef } from 'react';

import { updateUrlSearchParams } from '@/lib/url';

export interface RenderOptions {
  fancyGraphics: boolean;
  showEdges: boolean;
  showGrid: boolean;
  showCollisions: boolean;
  showSegmentColors: boolean;
  showSurfaceSamples: boolean;
  doubleSided: boolean;
  autoAnimate: boolean;
  showJointOverlay: boolean;
  /**
   * Pose deformation rigs on the scrubber. On by default; turning it off draws
   * the rigged parts at their rest shape, which is the A/B against the clip.
   * Inert on a run with no rigs.
   */
  showDeformation: boolean;
  /**
   * Show the canonical Blender/Cycles appearance instead of the three.js
   * approximation, streamed from a resident Cycles session.
   *
   * Off by default: it needs a Blender interpreter and a GPU on the machine
   * serving the viewer, and when it is unavailable the honest answer is the
   * existing viewport plus a notice — not a material that looks like the
   * generated result and is not it.
   */
  blenderRender: boolean;
  /**
   * Draw the run's baked-atlas viewer asset: the canonical Cycles appearance,
   * sampled onto a non-overlapping atlas and rendered locally by the browser.
   *
   * Off by default: the three.js path cannot evaluate a node graph, so a
   * shader-authored part draws in one color, roughness, and metallic, the ones
   * the GLB export gives it (`render/viewer_flat.py`, read in seconds). On asks
   * for the full appearance: one bake per IR (15-95 s, cached by content hash);
   * until it lands, and on any run that cannot have one, the viewport keeps the
   * flat colors and says why.
   */
  bakedMaterials: boolean;
}

const STORAGE_KEY = 'articraft-render-options';
// 10: fancy graphics became the project page's look, and the default (`defaultFancyGraphics`).
const STORAGE_SCHEMA_VERSION = 10;
const RENDER_QUERY_PARAM = 'render';
const RENDER_OPTION_KEYS: Array<keyof RenderOptions> = [
  'showEdges',
  'showGrid',
  'showCollisions',
  'showSegmentColors',
  'doubleSided',
  'autoAnimate',
  'showJointOverlay',
  'showSurfaceSamples',
  'fancyGraphics',
  'showDeformation',
  'blenderRender',
  'bakedMaterials',
];

export const defaultRenderOptions: RenderOptions = {
  fancyGraphics: true,
  showEdges: true,
  showGrid: true,
  showCollisions: false,
  showSegmentColors: false,
  showSurfaceSamples: false,
  doubleSided: true,
  autoAnimate: false,
  showJointOverlay: false,
  showDeformation: true,
  blenderRender: false,
  bakedMaterials: false,
};

const DEFAULT_RENDER_QUERY_VALUE = serializeOptions(defaultRenderOptions);

export interface RenderOptionsState {
  options: RenderOptions;
  setOption: <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => void;
  resetOptions: () => void;
}

/**
 * The project page's look by default (`FILM_LOOK`), except where WebGL is drawn on the CPU, where its shadow and probe
 * cost more than they show.
 */
function defaultFancyGraphics(): boolean {
  return !rendererKind().software;
}

/**
 * Read persisted render options from localStorage, falling back to defaults.
 */
function loadOptions(): RenderOptions {
  let options = { ...defaultRenderOptions, fancyGraphics: defaultFancyGraphics() };

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<RenderOptions> & { schemaVersion?: number };
      if (parsed.schemaVersion === STORAGE_SCHEMA_VERSION) {
        options = { ...defaultRenderOptions, ...parsed };
      } else {
        options = {
          ...defaultRenderOptions,
          ...parsed,
          showSegmentColors: false,
          doubleSided: true,
          fancyGraphics: defaultFancyGraphics(),
          blenderRender: false,
          bakedMaterials: false,
        };
      }
    }
  } catch {
    // Ignore corrupt or inaccessible storage.
  }

  if (typeof window === 'undefined') {
    return options;
  }

  try {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = parseOptions(params.get(RENDER_QUERY_PARAM));
    if (fromUrl) {
      return fromUrl;
    }
  } catch {
    // Ignore malformed URLs.
  }

  return options;
}

function serializeOptions(options: RenderOptions): string {
  return RENDER_OPTION_KEYS.map((key) => (options[key] ? '1' : '0')).join('');
}

function parseOptions(value: string | null): RenderOptions | null {
  if (!value) {
    return null;
  }

  const nextOptions = { ...defaultRenderOptions };
  const keys =
    value.length === RENDER_OPTION_KEYS.length
      ? RENDER_OPTION_KEYS
      : value.length === RENDER_OPTION_KEYS.length - 1
        ? RENDER_OPTION_KEYS.slice(0, -1)
      : value.length === RENDER_OPTION_KEYS.length - 2
        ? RENDER_OPTION_KEYS.slice(0, -2)
      : value.length === RENDER_OPTION_KEYS.length - 3
        ? RENDER_OPTION_KEYS.slice(0, -3)
      : value.length === RENDER_OPTION_KEYS.length - 4
        ? RENDER_OPTION_KEYS.slice(0, -4)
        : null;
  if (!keys) {
    return null;
  }

  for (const [index, key] of keys.entries()) {
    const bit = value[index];
    if (bit !== '0' && bit !== '1') {
      return null;
    }
    nextOptions[key] = bit === '1';
  }

  return nextOptions;
}

/**
 * Hook for managing render option toggles.
 *
 * State is persisted to localStorage under the key 'articraft-render-options'.
 */
export function useRenderOptions(): RenderOptionsState {
  const [options, setOptions] = useState<RenderOptions>(loadOptions);
  const isInitialMount = useRef(true);

  // Persist to localStorage whenever options change (skip the initial mount).
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        ...options,
        schemaVersion: STORAGE_SCHEMA_VERSION,
      }));
    } catch {
      // Storage full or unavailable -- silently ignore.
    }
  }, [options]);

  useEffect(() => {
    const serialized = serializeOptions(options);
    updateUrlSearchParams((params) => {
      if (serialized === DEFAULT_RENDER_QUERY_VALUE) {
        params.delete(RENDER_QUERY_PARAM);
      } else {
        params.set(RENDER_QUERY_PARAM, serialized);
      }
    });
  }, [options]);

  const setOption = useCallback(
    <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => {
      setOptions((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const resetOptions = useCallback(() => {
    setOptions({ ...defaultRenderOptions });
  }, []);

  return { options, setOption, resetOptions };
}
