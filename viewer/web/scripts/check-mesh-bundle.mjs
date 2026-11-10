// Pin the tar reader that turns a mesh bundle back into meshes.
//
//     npm run check:mesh-bundle
//
// A scene's meshes arrive as a few tar chunks rather than one request per
// visual (`viewer/mesh_bundle.py`, `mesh-bundle.ts`), which takes
// a 5403-mesh scene from 5403 requests to 2. The reader is the risky
// half: it is byte arithmetic over a format with an octal size field and
// 512-byte padding, and every way of getting it wrong produces *some* meshes
// and silently loses the rest — a scene with holes in it, at whatever point the
// offset first slipped.
//
// Two halves to this check:
//
//   1. **A real archive**, written by `mesh_bundle.chunk_tar` and embedded
//      below (gzipped, since a tar is mostly padding). This is the only test
//      that crosses the two implementations: the Python writer's idea of a
//      header against the TypeScript reader's.
//   2. **The arithmetic**, against archives assembled here from the spec, so
//      the edge cases can be constructed on purpose: a size that is not a
//      multiple of 512, an entry of exactly one block, a `prefix`-split name,
//      a space-padded octal field, and a truncated archive — which has to stop
//      cleanly rather than read past the buffer.
//
// Transpiled through the esbuild vite already installs, like check:depth-bias.

import { execFileSync } from "node:child_process";
import { gunzipSync } from "node:zlib";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const work = mkdtempSync(join(tmpdir(), "dycraft-mesh-bundle-"));
const failures = [];

const check = (ok, msg) => { if (!ok) failures.push(msg); };

function bundle(source, name) {
  const out = join(work, name);
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [source, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${out}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  return import(pathToFileURL(out).href);
}

/**
 * One chunk of `assets/meshes/{aa,bb,cc}.obj` at 1, 512, and 513 bytes, written
 * by `dycraft.viewer.mesh_bundle.chunk_tar`. Regenerate with:
 *
 *     python -c "import base64,gzip;from dycraft.viewer import mesh_bundle;..."
 */
const PYTHON_TAR_GZ_B64 = (
    'H4sIAHWVpGoC/+3WQQ7CIBRFUcauoisQsNiu59MQjdE0ETQuX+rINiZOGkzkHgafMGH0HkiM'
  + 'IUV9CfEYohbZjv6kVmayzrnXzJYzs2/7fG7tvu9VY1QBt5jkmq9Xdbor1Exm+R+GH+Tffsh/'
  + 'Zxz5L+GBulGBVRskyXk86Pk74P2a78D3/jeL/m93tqX/i/z/GjOtDbPOSQMCAAAAAAAAAAD8'
  + 'lydezHkOACgAAA=='
);

/** A USTAR header + payload, built from the spec rather than from the reader. */
function tarEntry(name, body, { prefix = "", spacePaddedSize = false } = {}) {
  const data = typeof body === "string" ? Buffer.from(body) : body;
  const header = Buffer.alloc(512);
  header.write(name, 0, 100, "ascii");
  header.write("000644 \0", 100, 8, "ascii");
  const size = data.length.toString(8).padStart(11, "0");
  header.write(spacePaddedSize ? `${size} ` : `${size}\0`, 124, 12, "ascii");
  header.write("ustar\0", 257, 6, "ascii");
  header.write("00", 263, 2, "ascii");
  if (prefix) header.write(prefix, 345, 155, "ascii");
  // Checksum: the reader does not verify it, but a real one has to be there for
  // anything else to accept the archive.
  header.write("        ", 148, 8, "ascii");
  let sum = 0;
  for (const byte of header) sum += byte;
  header.write(`${sum.toString(8).padStart(6, "0")}\0 `, 148, 8, "ascii");
  const padding = Buffer.alloc((512 - (data.length % 512)) % 512);
  return Buffer.concat([header, data, padding]);
}

const END_BLOCKS = Buffer.alloc(1024);

try {
  const mesh = await bundle(
    join(WEB, "src", "components", "viewer3d", "mesh-bundle.ts"),
    "mesh-bundle.mjs",
  );
  const { readTarEntries, meshBundleBase } = mesh;

  const read = (buffer) => [...readTarEntries(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  )].map((entry) => ({
    name: entry.name,
    text: Buffer.from(entry.data).toString("utf8"),
    bytes: entry.data.length,
  }));

  // ---- 1. the archive Python actually writes -----------------------------
  const real = read(gunzipSync(Buffer.from(PYTHON_TAR_GZ_B64, "base64")));
  const byName = new Map(real.map((entry) => [entry.name, entry]));
  check(real.length === 3,
        `read ${real.length} entries from the Python-written tar, expected 3 `
        + `(${real.map((e) => `${e.name}:${e.bytes}`).join(", ")})`);
  check(byName.get("assets/meshes/aa.obj")?.bytes === 1,
        "the 1-byte entry did not come back at 1 byte");
  check(byName.get("catalog/assets/meshes/bb.obj")?.bytes === 512,
        "the exactly-one-block entry did not come back at 512 bytes");
  check(byName.get("assets/meshes/cc.obj")?.bytes === 513,
        "the 513-byte entry did not come back at 513 bytes — the padding after a "
        + "non-multiple size is where the offset slips");
  check(byName.get("assets/meshes/cc.obj")?.text === "x".repeat(513),
        "the 513-byte entry's bytes are not its own");
  check(byName.get("catalog/assets/meshes/bb.obj")?.text === "v 0 0 0\n".repeat(64),
        "the one-block entry's bytes are not its own");

  // ---- 2. the arithmetic, on purpose-built archives ----------------------
  const spec = read(Buffer.concat([
    tarEntry("first.obj", "v 1 2 3\n"),
    tarEntry("second.obj", "x".repeat(1024)),                  // exact multiple
    tarEntry("third.obj", "y".repeat(4), { spacePaddedSize: true }),
    tarEntry("deep.obj", "z", { prefix: "assets/meshes/nested" }),
    END_BLOCKS,
  ]));
  check(spec.length === 4, `read ${spec.length} of 4 hand-built entries`);
  check(spec[1]?.bytes === 1024, "an exact-multiple entry broke the offset");
  check(spec[2]?.text === "yyyy",
        "a space-padded octal size field was misread");
  check(spec[3]?.name === "assets/meshes/nested/deep.obj",
        `a prefix-split name came back as ${spec[3]?.name}`);

  // Trailing junk after the end blocks must not be read as an entry.
  const withJunk = read(Buffer.concat([
    tarEntry("only.obj", "v"), END_BLOCKS, Buffer.from("garbage"),
  ]));
  check(withJunk.length === 1,
        `zero blocks did not end the archive (${withJunk.length} entries)`);

  // A chunk that was cut off mid-transfer: stop, do not read past the end.
  const full = Buffer.concat([tarEntry("big.obj", "q".repeat(2048)), END_BLOCKS]);
  const truncated = read(full.subarray(0, 700));
  check(truncated.length === 0,
        `a truncated archive yielded ${truncated.length} entries; the reader has `
        + `to stop rather than hand out a short buffer`);

  // ---- the sibling-URL assumption ---------------------------------------
  check(meshBundleBase("/api/runs/a%2Fb/files") === "/api/runs/a%2Fb/mesh-bundle",
        "the bundle base is not derived from the file base");
  check(meshBundleBase("http://x/api/runs/r/files")
        === "http://x/api/runs/r/mesh-bundle",
        "an absolute file base did not derive");
  check(meshBundleBase("/api/runs/r/somewhere-else") === null,
        "an unexpected base URL has to disable bundling, not guess");

  console.log(`python-written tar: ${real.length} entries, `
              + `${real.map((e) => e.bytes).join("/")} bytes`);
  console.log(`hand-built tar: ${spec.length} entries, prefix names and `
              + `space-padded sizes read`);
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-mesh-bundle: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-mesh-bundle: ok");
