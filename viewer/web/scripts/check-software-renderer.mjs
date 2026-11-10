// Pin which WebGL renderers the viewport treats as drawn on the CPU (`software-renderer.ts`).
//
//     npm run check:software-renderer
//
// On those it draws cheaper (no antialiasing, a 0.75 pixel ratio, at most 15 frames a second), so a long film plays
// instead of freezing. A false positive costs a GPU user their antialiasing, so real GPU names, including Mesa's
// hardware drivers whose names carry "LLVM", must not match. The names are the strings browsers report through
// WEBGL_debug_renderer_info.
//
// Transpiled through the esbuild vite already installs, like check:cutaway.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-software-renderer-"));
const failures = [];

const SOFTWARE = [
  "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
  "Google SwiftShader",
  "llvmpipe (LLVM 15.0.7, 256 bits)",
  "ANGLE (Mesa, llvmpipe (LLVM 15.0.6, 256 bits), OpenGL 4.5)",
  "softpipe",
  "ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11 vs_5_0 ps_5_0, D3D11)",
];
const HARDWARE = [
  "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL 4.5.0)",
  "ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)",
  "ANGLE (Intel, Mesa Intel(R) UHD Graphics 620 (KBL GT2), OpenGL 4.6)",
  "ANGLE (AMD, AMD Radeon RX 6800 XT (radeonsi, navi21, LLVM 15.0.7, DRM 3.49, 6.2.0), OpenGL 4.6)",
  "WebKit WebGL",
  "",
  null,
  undefined,
];

try {
  const out = join(work, "software-renderer.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [join(WEB, "src", "components", "viewer3d", "software-renderer.ts"), "--bundle", "--format=esm",
     "--platform=node", "--log-level=error", `--outfile=${out}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  const R = await import(pathToFileURL(out).href);
  for (const name of SOFTWARE) {
    if (!R.isSoftwareRenderer(name)) failures.push(`not taken for a CPU rasterizer: ${JSON.stringify(name)}`);
  }
  for (const name of HARDWARE) {
    if (R.isSoftwareRenderer(name)) failures.push(`taken for a CPU rasterizer: ${JSON.stringify(name)}`);
  }
  console.log(`software-renderer: ${SOFTWARE.length} CPU and ${HARDWARE.length} other names`);
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-software-renderer: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-software-renderer: ok");
