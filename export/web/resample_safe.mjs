// usage: node resample_safe.mjs <in.glb> <out.glb>
// Keyframe resampling (what gltf-transform's `optimize` runs; its keyframe-resample 0.1.0 check), with two changes.
// Rotation keys kept at most a quarter turn apart: its check keeps a key only once the path to it reaches half a turn,
// so a steady spin whose frame step divides 180 degrees keeps keys exactly half a turn apart, and a player's slerp
// can then take either way round (a camera orbit turning backwards). And morph weights to 1e-6, not 1e-4:
// principal-shape morphs mix every target at every frame, so the misses would add up.
// `--web` runs this first and web_glb.mjs, which does not resample, after it.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';

const MAX_TURN = Math.PI / 2;     // rotation angle between kept keys (the original allows PI)
const TOLERANCE = { weights: 1e-6 }, DEFAULT_TOLERANCE = 1e-4;
const EPSILON = 1e-6;

// keyframe-resample 0.1.0's resampleDebug (what gltf-transform 4.5.1 uses), the rotation guard changed
function resampleSafe(input, output, interpolation, tolerance = 1e-4) {
  const n = output.length / input.length;
  const tmp = new Array(n).fill(0), value = new Array(n).fill(0);
  const valueNext = new Array(n).fill(0), valuePrev = new Array(n).fill(0);
  const lastIndex = input.length - 1;
  let writeIndex = 1;
  for (let i = 1; i < lastIndex; ++i) {
    const timePrev = input[writeIndex - 1], time = input[i], timeNext = input[i + 1];
    const t = (time - timePrev) / (timeNext - timePrev);
    let keep = false;
    if (time !== timeNext && (i !== 1 || time !== input[0])) {
      get(output, writeIndex - 1, valuePrev);
      get(output, i, value);
      get(output, i + 1, valueNext);
      if (interpolation === 'slerp') {
        const sample = slerp(tmp, valuePrev, valueNext, t);
        keep = !eq(value, sample, tolerance) || angle(valuePrev, value) + angle(value, valueNext) >= MAX_TURN;
      } else if (interpolation === 'lerp') {
        keep = !eq(value, vlerp(tmp, valuePrev, valueNext, t), tolerance);
      } else if (interpolation === 'step') {
        keep = !eq(value, valuePrev) || !eq(value, valueNext);
      }
    }
    if (keep) {
      if (i !== writeIndex) {
        input[writeIndex] = input[i];
        set(output, writeIndex, get(output, i, tmp));
      }
      writeIndex++;
    }
  }
  if (lastIndex > 0) {
    input[writeIndex] = input[lastIndex];
    set(output, writeIndex, get(output, lastIndex, tmp));
    writeIndex++;
  }
  return writeIndex;
}

function get(a, i, out) { for (let k = 0; k < out.length; k++) out[k] = a[i * out.length + k]; return out; }
function set(a, i, v) { for (let k = 0; k < v.length; k++) a[i * v.length + k] = v[k]; }
function eq(a, b, tol = 0) { return a.every((x, k) => Math.abs(x - b[k]) <= tol); }
function vlerp(out, a, b, t) { for (let k = 0; k < a.length; k++) out[k] = a[k] * (1 - t) + b[k] * t; return out; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; }
// rotation angle between two unit quaternions, either sign
function angle(a, b) { const d = dot(a, b); return Math.acos(Math.min(1, Math.max(-1, 2 * d * d - 1))); }
function slerp(out, a, b, t) {
  let [bx, by, bz, bw] = b, cosom = dot(a, b);
  if (cosom < 0) { cosom = -cosom; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let s0 = 1 - t, s1 = t;
  if (1 - cosom > EPSILON) {
    const omega = Math.acos(cosom), sinom = Math.sin(omega);
    s0 = Math.sin((1 - t) * omega) / sinom;
    s1 = Math.sin(t * omega) / sinom;
  }
  out[0] = s0 * a[0] + s1 * bx; out[1] = s0 * a[1] + s1 * by; out[2] = s0 * a[2] + s1 * bz; out[3] = s0 * a[3] + s1 * bw;
  return out;
}

const [src, dst] = process.argv.slice(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(src);
let before = 0, after = 0;
for (const anim of doc.getRoot().listAnimations()) {
  const paths = new Map(anim.listChannels().map(c => [c.getSampler(), c.getTargetPath()]));
  for (const s of anim.listSamplers()) {
    const interp = s.getInterpolation();
    if (interp !== 'LINEAR' && interp !== 'STEP') continue;
    const inA = s.getInput(), outA = s.getOutput();
    if (inA.getNormalized() || outA.getNormalized()) continue;      // Blender writes floats; leave anything else
    const t = Float32Array.from(inA.getArray()), v = Float32Array.from(outA.getArray());
    const path = paths.get(s);
    const n = resampleSafe(t, v, interp === 'STEP' ? 'step' : path === 'rotation' ? 'slerp' : 'lerp',
                           TOLERANCE[path] ?? DEFAULT_TOLERANCE);
    before += t.length; after += n;
    if (n < t.length) {
      // new accessors: inputs are shared between samplers (optimize's dedup merges equal ones again, prune drops the old)
      s.setInput(doc.createAccessor().setType('SCALAR').setArray(t.slice(0, n)).setBuffer(inA.getBuffer()));
      s.setOutput(doc.createAccessor().setType(outA.getType()).setArray(v.slice(0, n * (v.length / t.length)))
        .setBuffer(outA.getBuffer()));
    }
  }
}
console.log(`resample_safe: ${before} keys -> ${after}`);
await io.write(dst, doc);
