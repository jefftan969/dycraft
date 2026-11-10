// usage: node web_glb.mjs <in.glb> <out.glb> [options]   -- the browser GLB (`--web`, after resample_safe.mjs)
//
// gltf-transform 4.5.1's `optimize` steps (--compress meshopt --texture-compress webp --texture-size 1024, without
// simplification, joining, instancing, or its own resampling), with these options:
//
//   --deflat            flat-shaded morphing primitives (every triangle's three normals equal, as Blender writes flat
//                       faces) whose vertices are split by face (a quarter or more of them copies) and that carry 100k
//                       vertex-targets or more get smooth normals of their base shape, so the copies weld and take
//                       their morph targets with them. Not none: flat shading of the morphed surface shows the fit's
//                       millimeters of noise as stripes across a finely cut strap, and the stored normals were the
//                       base shape's anyway. Rigid and skinned primitives keep theirs: without NORMAL, three.js shades
//                       flat from screen-space derivatives, and normal-mapped floors come out flatter
//   --weights16         dense morph-weight animation (principal shapes' weights, which morphs.py scales into [-1, 1])
//                       as normalized int16 (KHR_mesh_quantization); gltf-transform leaves weight outputs out of meshopt
//                       compression. Sparse ones (keyframe crossfades) stay floats, see weights16()
//   --neardup           textures within bake noise of a bigger one (same size and slots; per channel mean |diff| <= 1.5
//                       and 99.9th percentile <= 12 of 255, and high-pass detail correlated >= 0.9 unless both are
//                       flat) replaced by it: copies of an object are baked one by one
//   --qpos B            position (and morph target) quantization bits (gltf-transform's default 14)
//   --texture-size N    largest texture side (1024); --normal-size N, --mr-size N: normal and metallic-roughness maps
//   --webp-quality Q    lossy WebP quality (sharp's default 80)
//   --simplify E        meshoptimizer simplification of meshes without morph targets or skins, error E as a fraction
//                       of each mesh's extent
//
// `--web` uses --deflat --weights16 --neardup --normal-size 512 --mr-size 512, the settings the project page's scenes
// use. --qpos 13 and --simplify make files smaller still but lose visible detail (thin parts, close-up relief).
import { NodeIO, Accessor } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, palette, flatten, weld, prune, sparse, textureCompress, meshopt, simplify, listTextureSlots } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';

const FLAGS = ['deflat', 'weights16', 'neardup'];
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : dflt; };
const flag = name => args.includes('--' + name);
const [src, dst] = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--') && !FLAGS.includes(args[i - 1].slice(2))));
for (const a of args.filter(a => a.startsWith('--')))
  if (![...FLAGS, 'qpos', 'texture-size', 'normal-size', 'mr-size', 'webp-quality', 'simplify'].includes(a.slice(2)))
    throw new Error(`web_glb: unknown option ${a}`);
await Promise.all([MeshoptEncoder.ready, MeshoptDecoder.ready, MeshoptSimplifier.ready]);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
const log = [];

function deflat(doc) {
  const COS = Math.cos(Math.PI / 180);
  const eq = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2] >= COS * Math.hypot(...x) * Math.hypot(...y);
  let n = 0, verts = 0, ns = 0, nverts = 0;
  for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives()) {
    const N = p.getAttribute('NORMAL'), I = p.getIndices();
    if (!N || !I || p.getMode() !== 4) continue;
    const idx = I.getArray(), q = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    let flat = true;
    for (let t = 0; t < idx.length && flat; t += 3) {
      for (let k = 0; k < 3; k++) N.getElement(idx[t + k], q[k]);
      flat = eq(q[0], q[1]) && eq(q[1], q[2]);
    }
    // only where the morph data is worth it (not a few thousand values) and the shading change pays
    if (!flat || p.getAttribute('POSITION').getCount() * p.listTargets().length < 100_000) continue;
    // area-weighted face normals summed over the vertices at one position, the same for each copy
    const P = p.getAttribute('POSITION'), V = P.getCount(), key = new Map(), group = new Int32Array(V), a = [], b = [], c = [];
    for (let i = 0; i < V; i++) {
      const k = P.getElement(i, a).join(',');
      if (!key.has(k)) key.set(k, key.size);
      group[i] = key.get(k);
    }
    if (key.size > 0.75 * V) { n++; verts += V; continue; }     // few copies: left flat
    for (const t of p.listTargets()) t.setAttribute('NORMAL', null);
    const acc = new Float64Array(key.size * 3);
    for (let t = 0; t < idx.length; t += 3) {
      P.getElement(idx[t], a); P.getElement(idx[t + 1], b); P.getElement(idx[t + 2], c);
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) acc[group[idx[t + k]] * 3 + d] += f[d];
    }
    const out = new Float32Array(V * 3);
    for (let i = 0; i < V; i++) {
      const g = group[i], x = acc[g * 3], y = acc[g * 3 + 1], z = acc[g * 3 + 2], l = Math.hypot(x, y, z) || 1;
      out[i * 3] = x / l; out[i * 3 + 1] = y / l; out[i * 3 + 2] = z / l;
    }
    p.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(out).setBuffer(N.getBuffer()));
    ns++; nverts += V;
  }
  log.push(`deflat: ${ns} flat-shaded morphing primitives (${nverts} vertices) smooth` + (n ? `, ${n} (${verts}) with few copies left flat` : ''));
}

function weights16(doc) {
  let n = 0, clipped = 0;
  for (const a of doc.getRoot().listAnimations()) for (const ch of a.listChannels()) {
    if (ch.getTargetPath() !== 'weights') continue;
    const s = ch.getSampler(), out = s.getOutput();
    // sparse ones (keyframe crossfades, mostly zeros) stay floats: gltf-transform 4.5.1 writes a sparse accessor's
    // values decoded (32767 -> 1.0) into the integer array, so normalized int16 weights would read back as 1/32767
    if (out.getNormalized() || out.getComponentType() !== Accessor.ComponentType.FLOAT || out.getSparse()) continue;
    const f = out.getArray(), q = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const v = Math.max(-1, Math.min(1, f[i]));
      if (v !== f[i]) clipped++;
      q[i] = Math.round(v * 32767);
    }
    s.setOutput(doc.createAccessor().setType('SCALAR').setArray(q).setNormalized(true).setBuffer(out.getBuffer()));
    n++;
  }
  if (n) doc.createExtension(ALL_EXTENSIONS.find(e => e.EXTENSION_NAME === 'KHR_mesh_quantization')).setRequired(true);
  log.push(`weights16: ${n} morph-weight channels as int16` + (clipped ? `, ${clipped} values clipped to [-1, 1]` : ''));
}

async function neardup(doc) {
  const info = [];
  for (const t of doc.getRoot().listTextures()) {
    const img = t.getImage();
    if (!img) continue;
    try {
      const { data, info: i } = await sharp(Buffer.from(img)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      const thumb = await sharp(data, { raw: { width: i.width, height: i.height, channels: 4 } }).resize(16, 16, { fit: 'fill' }).raw().toBuffer();
      info.push({ t, w: i.width, h: i.height, slots: listTextureSlots(t).sort().join('+'), thumb, bytes: img.byteLength });
    } catch {}
  }
  info.sort((a, b) => b.bytes - a.bytes);
  const reps = [], pixels = new Map(), hps = new Map();
  // high-pass (4-neighbor Laplacian of R+G+B): what a texture shows beyond its broad color
  const FLAT = 0.02;           // solid, to the last bit of noise
  const highpass = async x => {
    if (hps.has(x)) return hps.get(x);
    const d = await px(x), w = x.w, h = x.h, v = new Float32Array(w * h);
    let ss = 0;
    for (let y = 1; y < h - 1; y++) for (let i = 1; i < w - 1; i++) {
      const k = y * w + i;
      let s = 0;
      for (let c = 0; c < 3; c++) s += 4 * d[k * 4 + c] - d[(k - 1) * 4 + c] - d[(k + 1) * 4 + c] - d[(k - w) * 4 + c] - d[(k + w) * 4 + c];
      v[k] = s; ss += s * s;
    }
    const r = { v, rms: Math.sqrt(ss / v.length) };
    hps.set(x, r);
    return r;
  };
  const corr = (a, b) => { let aa = 0, bb = 0, ab = 0; for (let i = 0; i < a.length; i++) { aa += a[i] * a[i]; bb += b[i] * b[i]; ab += a[i] * b[i]; } return ab / Math.sqrt(aa * bb + 1e-9); };
  const px = async x => { if (!pixels.has(x)) pixels.set(x, await sharp(Buffer.from(x.t.getImage())).ensureAlpha().raw().toBuffer()); return pixels.get(x); };
  let merged = 0, saved = 0;
  for (const x of info) {
    let hit = null;
    for (const r of reps) {
      if (r.w !== x.w || r.h !== x.h || r.slots !== x.slots) continue;
      let d = 0;                                  // thumbnails first: most pairs differ there already
      for (let i = 0; i < x.thumb.length; i++) d = Math.max(d, Math.abs(x.thumb[i] - r.thumb[i]));
      if (d > 8) continue;
      const a = await px(x), b = await px(r), N = a.length / 4, mean = [0, 0, 0, 0];
      const hist = Array.from({ length: 4 }, () => new Uint32Array(256));
      for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]); mean[i & 3] += e; hist[i & 3][e]++; }
      const p999 = hist.map(h => { let c = 0, k = 0; for (; k < 256; k++) { c += h[k]; if (c >= 0.999 * N) break; } return k; });
      if (Math.max(...mean) / N > 1.5 || Math.max(...p999) > 12) continue;
      // and the same structure: faint maps pass the pixel test whatever they show (a tile's normal map and a plaster
      // one can be under 1/255 apart and uncorrelated), while copies of one bake correlate. Only two solid maps merge
      // without it: a faint map is not a solid one, and swapping it for a solid map would let prune fold its relief away
      const ha = await highpass(x), hb = await highpass(r);
      if (ha.rms < FLAT && hb.rms < FLAT || corr(ha.v, hb.v) >= 0.9) { hit = r; break; }
    }
    if (!hit) { reps.push(x); continue; }
    x.t.listParents().forEach(p => { if (p !== doc.getRoot()) p.swap(x.t, hit.t); });
    x.t.dispose();
    pixels.delete(x); hps.delete(x);
    merged++; saved += x.bytes;
  }
  log.push(`neardup: ${merged} textures replaced by a near-identical one (${(saved / 1e6).toFixed(1)} MB as exported)`);
}

function simplifyStatic(error) {
  return async (doc) => {
    // gltf-transform's simplify sees only positions: meshes that morph or skin are left alone (it skips primitives
    // of other draw modes, so they are marked LINES for the call)
    const keep = new Map();
    for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives())
      if (p.listTargets().length || p.getAttribute('JOINTS_0')) keep.set(p, p.getMode());
    for (const p of keep.keys()) p.setMode(1);
    await doc.transform(simplify({ simplifier: MeshoptSimplifier, error, ratio: 0, lockBorder: false }));
    for (const [p, mode] of keep) p.setMode(mode);
  };
}

const doc = await io.read(src);
if (flag('deflat')) deflat(doc);
const quality = opt('webp-quality') ? +opt('webp-quality') : undefined;
const T = [dedup(), palette({}), flatten(), weld()];
if (opt('simplify')) T.push(simplifyStatic(+opt('simplify')));
T.push(prune({ keepAttributes: false, keepIndices: false, keepLeaves: false, keepSolidTextures: false }));
// after prune, which folds solid textures into factors: a solid map must not stand in for a faint one, or take its place
if (flag('neardup')) T.push(neardup, prune({ keepAttributes: false, keepIndices: false, keepLeaves: false, keepSolidTextures: false }));
T.push(sparse());
if (flag('weights16')) T.push(weights16);          // after sparse: the dense weights only (see weights16)
const slots = [['normalTexture', opt('normal-size')], ['metallicRoughnessTexture', opt('mr-size')]].filter(([, s]) => s);
for (const [slot, s] of slots)
  T.push(textureCompress({ encoder: sharp, resize: [+s, +s], targetFormat: 'webp', slots: new RegExp(`^${slot}$`), quality }));
const tsize = +opt('texture-size', 1024);
T.push(textureCompress({ encoder: sharp, resize: [tsize, tsize], targetFormat: 'webp', quality,
  ...(slots.length ? { slots: new RegExp(`^(?!(${slots.map(([s]) => s).join('|')})$)`) } : {}) }));
const quantize = () => meshopt({ encoder: MeshoptEncoder, level: 'high', ...(opt('qpos') ? { quantizePosition: +opt('qpos') } : {}) });
T.push(quantize());
await doc.transform(...T);
await io.write(dst, doc);
console.log(['web_glb: ' + (args.slice(2).join(' ') || 'defaults'), ...log].join('\n'));
