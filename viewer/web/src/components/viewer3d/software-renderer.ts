// Whether WebGL is drawn on the CPU (no working GPU driver), and how the viewport draws there.
//
// Without a GPU driver, Chrome draws WebGL with SwiftShader and Firefox with Mesa's llvmpipe. A long film's scene
// then costs tenths of a second a frame, and the page can hand over frames far faster than they are drawn: the
// backlog grows until a frame waits on all of it, and playback freezes for tens of seconds. So on a software
// renderer the viewport draws cheaper: no antialiasing, fewer pixels, and at most `SOFTWARE_MAX_FPS` frames a second,
// so nothing queues behind the frame on screen.
//
// Detected once per page, on a throwaway context, before the viewport's own renderer is made: antialiasing is fixed
// when a WebGL context is created. A browser that masks the renderer's name (Firefox resisting fingerprinting, some
// builds) is taken for a GPU: the viewport then draws as it always has, so a missing cap there is not a bug.

/** Renderer names of WebGL drawn on the CPU. */
const SOFTWARE = /swiftshader|llvmpipe|softpipe|software rasterizer|microsoft basic render/i;

/** Frames a second the viewport draws at most on a software renderer. */
export const SOFTWARE_MAX_FPS = 15;
/** The pixel ratio a software renderer draws at, at most. */
export const SOFTWARE_PIXEL_RATIO = 0.75;

export interface RendererKind {
  software: boolean;
  /** The renderer's own name, when the browser gives it. */
  name: string | null;
}

let detected: RendererKind | null = null;

/** Whether a renderer name is a CPU rasterizer. Pure, for the check. */
export function isSoftwareRenderer(name: string | null | undefined): boolean {
  return Boolean(name && SOFTWARE.test(name));
}

/** This page's WebGL renderer: drawn on the CPU or not. */
export function rendererKind(): RendererKind {
  if (detected) return detected;
  let name: string | null = null;
  try {
    const canvas = document.createElement('canvas');
    const gl = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as WebGLRenderingContext | null;
    if (gl) {
      const info = gl.getExtension('WEBGL_debug_renderer_info');
      name = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER) ?? '') || null;
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    name = null;
  }
  detected = { software: isSoftwareRenderer(name), name };
  return detected;
}
