// Pin the reader that turns a native animation track back into a moving scene.
//
//     npm run check:native-track
//
// A native clip's 3D playback is evaluated Blender geometry, and it arrives as
// one motion track plus one blob per distinct mesh (`viewer/native_animation.py`,
// `native-track.ts`). The reader is the risky half, for the same reason the tar
// reader is: every way of getting it wrong still draws *something*.
//
// Three failure modes this is here to catch, none of which raise on their own:
//
//   1. **Major order.** The server ships rows 0..2 of `matrix_world`, row-major,
//      and `Matrix4.set` takes its arguments row-major while `fromArray` reads
//      column-major. Read the wrong way, a rotation is still a rotation — the
//      object simply stands somewhere else, plausibly, forever.
//   2. **The compacted columns.** `mesh` and `hidden` are a scalar when they
//      never change over the clip and a per-frame array when they do. Reading a
//      scalar as an array yields `undefined` at every frame, which is a hidden
//      object rather than an error.
//   3. **An object that is not in every frame.** It gets mesh index -1 and an
//      identity matrix, and must be drawn as absent rather than as a copy of
//      itself parked at the origin.
//
// The track and mesh below were written by the real Python encoder, so this is
// the only test that crosses the two implementations.
//
// Transpiled through the esbuild vite already installs, like check:mesh-bundle.

import { execFileSync } from "node:child_process";
import { gunzipSync } from "node:zlib";
import { mkdtempSync, rmSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
// Under `node_modules`, not in `/tmp`: the bundle below leaves three EXTERNAL so
// that it and this script share one module instance, and node only resolves a
// bare `three` specifier from a directory with `node_modules` above it. Sharing
// the instance is what makes `instanceof THREE.Mesh` inside `native-track.ts`
// mean the same thing here as it does in the app — bundle three in instead and
// every `instanceof` across the boundary quietly answers false.
const work = mkdtempSync(join(WEB, "node_modules", ".dycraft-native-track-"));
const failures = [];

const check = (ok, msg) => { if (!ok) failures.push(msg); };
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol;

function bundle(source, name) {
  const out = join(work, name);
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [source, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     "--external:three", "--external:three/addons/*",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${out}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  return import(pathToFileURL(out).href);
}

/**
 * A three-frame bundle written by `dycraft.viewer.native_animation.track`, with
 * one object steady for the whole clip, one that changes mesh and goes hidden
 * at frame 1, one that does not exist until frame 2, and a room surface that
 * must not appear at all. Regenerate with the recipe in that module's tests.
 */
const TRACK_GZ_B64 = (
    'H4sIALKgpGoC/+WQXWrbQBDHRU/QIwg9u87s925e2pzDiDL7FSvIkpGUQgiGHCU3yVV6lM7a'
  + 'gdTOQ8hzBJrd+dj5zX+ev1XVY/MnTXM3Ds11zVZ1g0O3w+XkNxG40DoljSny4CDYgNqjyU6i'
  + 'ZSY29GD4nSfcpZnqBbl5X25croEcCm+PmU0TOAaDWSkO2tNhrBJSSboyEJkBt67ElFNWMaO4'
  + 'lgpAZ5dB+iiFKajkReCBaYdGsJg998h9llFp4QRGAVJwoXSIVupgfDDcKCABoJ3R6ELpwbU1'
  + 'SigegdTwHGVkOUlNAWcZooxG2eCisBZkPI4XwDsNzEaRQOqmpSajv0thOQp7bAZSX5Y1Lwnj'
  + 'Q2F0MQ1LtzyUaDfMy9Uep6UkwtiPU3kFa9o1rHkxtDW2hvZ1XZQtm9t2kZqQk7Gf02FVv3F2'
  + '47TfdsPtBWm47/tzhCzdVTH6ErEhBk3A2v9RmyNrVS/T/attz8A9LimMuzR9SDYFaotx78g/'
  + 'iFt+fs5+g65OmttDe6jruqqeflbvvo9iTy+n8++vi9qb8/rvN5/v/9Xm+Vz9P6QH0lVUBAAA'
);

/**
 * `atlas` for a one-object, one-triangle bundle, from the real Python encoder.
 *
 * The baked atlas is the only way native playback can draw a procedural shader:
 * the evaluated mesh has no UVs of its own, so the bake's `(3T,2)` corner UVs
 * are what let a textured material be bound at all. Sized to pair with
 * `MESH_GZ_B64` — 4 triangles, 12 corners — and the first corner is
 * deliberately asymmetric (0.25, 0.5) so a transposed or reversed read shows up.
 */
const ATLAS_GZ_B64 = (
    'H4sIALznrGoC/41PWw6CMBDkKKTfJrQVEPziHoaQ7QNFpRBaSAjhrF7AO2BbDd/u187O7Ozs'
  + 'OwiCBU1y0E2n0DkkhxCBalowX4xA8gxwnOSJoPFJMEJFxiHHlCUJTzFDdkGYuZdOPJLUYTBP'
  + '0FLbyWVBDzl7H8YJPUasE7OTqMoMjVPEFnR1raWxAHuGd4OygVwaupaOZ3fJzc9PQetv9TCY'
  + 'yNwadd1Nx6nSPXBP+wx7GOe9lmsYBsX22mzZt4t/+g9H/0x+IAEAAA=='
);

/** `mesh_blob` for the box the first two objects wear. */
const MESH_GZ_B64 = (
    'H4sIALKgpGoC/22MzQrCMBCE1//XKDl7SNJsk3jxWdI2KQGp0AQvKvjobqU9tLgw7HzD7rwA'
  + '4Mkefkjx3rNLIc4Fa2PnUyZgjXSNdgFR8qqmpQ2WChVZwcsguDR2zNCiQaFRVgo5r4INXNWt'
  + 'KjWjNurOsfGJ+gxhHqLru9uP1buAxXyuS7/m9e2czf5fBrAhbaevce9Ie9KRdJj8aeIviK+a'
  + 'PxABAAA='
);

const bytes = (b64) => {
  const gz = gunzipSync(Buffer.from(b64, "base64"));
  return gz.buffer.slice(gz.byteOffset, gz.byteOffset + gz.byteLength);
};

try {
  const THREE = await import(pathToFileURL(
    join(WEB, "node_modules", "three", "build", "three.module.js")).href);
  const mod = await bundle(
    join(WEB, "src", "components", "viewer3d", "native-track.ts"), "native.mjs");
  const presentation = await bundle(
    join(WEB, "src", "components", "viewer3d", "scene-presentation.ts"),
    "presentation.mjs");

  // --- the track --------------------------------------------------------
  const track = mod.decodeNativeTrack(bytes(TRACK_GZ_B64));
  check(track.n_frames === 3, `n_frames ${track.n_frames}`);
  check(track.objects.length === 3,
        `expected 3 drawable objects, got ${track.objects.length}`);
  check(!track.objects.some((o) => o.name.startsWith("room")),
        "a room surface reached the client; the viewport has never drawn one");
  check(track.matrices.length === 3 * 3 * mod.MATRIX_STRIDE,
        `matrix block is ${track.matrices.length} floats`);

  const [steady, morphing, latecomer] = track.objects;
  // Compaction: a scalar column and an array column read the same way.
  check(mod.meshIndexAt(steady, 0) === 0 && mod.meshIndexAt(steady, 2) === 0,
        "a constant mesh column did not read at every frame");
  check(mod.hiddenAt(steady, 2) === false, "a constant hidden column misread");
  check(mod.meshIndexAt(morphing, 0) === 0 && mod.meshIndexAt(morphing, 1) === 1,
        "a per-frame mesh column did not follow the topology change");
  check(mod.hiddenAt(morphing, 0) === false && mod.hiddenAt(morphing, 1) === true,
        "a per-frame hidden column misread");
  check(mod.meshIndexAt(latecomer, 0) === -1 && mod.meshIndexAt(latecomer, 2) === 2,
        "an object absent from early frames must read as -1, not as frame 2's mesh");

  check(mod.linkNameForIdentity("wheel/rim") === "wheel__rim",
        "identity -> link name dropped the instance prefix");
  check(mod.linkNameForIdentity("robot/arm/link_2") === "robot__arm/link_2",
        "only the FIRST slash is the instance separator");

  // --- the mesh ---------------------------------------------------------
  const mesh = mod.decodeNativeMesh(bytes(MESH_GZ_B64));
  check(mesh.positions.length === 8 * 3, `mesh has ${mesh.positions.length / 3} vertices`);
  check(mesh.indices.length === 4 * 3, `mesh has ${mesh.indices.length / 3} triangles`);
  check(near(mesh.positions[3], 1) && near(mesh.positions[4], 0),
        "vertex 1 did not decode; the header padding may be misaligned");
  const geometry = mod.nativeGeometry(mesh);
  check(geometry.getAttribute("normal") !== undefined,
        "evaluated geometry arrives without normals and must be given some");

  // --- a deforming frame as its positions only (`mesh_positions`) ---------
  // The server's layout: the envelope, then the mesh's vertices as float32, exactly. Decoding must give them back
  // bit for bit: creased normals worked out from rounded vertices flicker.
  {
    const source = Float32Array.from(mesh.positions, (x, i) => x + [2.5, -1.25, 0.75][i % 3]);
    const head = JSON.stringify({ version: track.version, digest: "d", vertices: 8, topology: mesh.topology ?? "t",
                                  encoding: "f32" });
    const padded = head + " ".repeat((4 - (head.length % 4)) % 4);
    const blob = new Uint8Array(4 + padded.length + source.byteLength);
    new DataView(blob.buffer).setUint32(0, padded.length, true);
    blob.set(new TextEncoder().encode(padded), 4);
    blob.set(new Uint8Array(source.buffer), 4 + padded.length);
    const frame = mod.decodeNativePositions(blob.buffer);
    check(frame.positions.length === 24 && frame.positions.every((x, i) => x === source[i]),
          "positions did not decode to the mesh's own vertices, bit for bit");
    check(frame.topology === (mesh.topology ?? "t"), "the topology hash did not come through");
    let refused = false;
    try { mod.decodeNativePositions(blob.buffer.slice(0, blob.byteLength - 2)); } catch { refused = true; }
    check(refused, "a positions reply of the wrong length must be refused, not read short");
  }

  // --- the frame --------------------------------------------------------
  const geometries = new Map(track.meshes.map((d, i) => [d, i === 0 ? geometry : geometry]));
  const scene = mod.buildNativeScene(track, geometries, null);
  scene.group.updateMatrixWorld(true);

  mod.applyNativeFrame(scene, 0);
  scene.group.updateMatrixWorld(true);
  check(scene.nodes[0].visible && scene.nodes[1].visible,
        "frame 0 should draw both of the objects that exist in it");
  check(!scene.nodes[2].visible,
        "an object that does not exist yet was drawn at the origin");

  // The decisive one. `morphing` carries a 90-degree yaw and a translation, so
  // a column-major read lands the probe somewhere else entirely.
  const probe = new THREE.Vector3(1, 0, 0).applyMatrix4(scene.nodes[1].matrixWorld);
  check(near(probe.x, 7) && near(probe.y, 9) && near(probe.z, 9),
        `matrix read in the wrong major order: (1,0,0) -> `
        + `(${probe.x}, ${probe.y}, ${probe.z}), expected (7, 9, 9)`);

  mod.applyNativeFrame(scene, 1);
  check(!scene.nodes[1].visible, "frame 1 hides `morphing`; it stayed drawn");
  mod.applyNativeFrame(scene, 2);
  check(scene.nodes[2].visible, "frame 2 introduces `latecomer`; it stayed hidden");
  // Idempotent and reversible: scrubbing back must restore, not accumulate.
  mod.applyNativeFrame(scene, 0);
  scene.group.updateMatrixWorld(true);
  const again = new THREE.Vector3(1, 0, 0).applyMatrix4(scene.nodes[1].matrixWorld);
  check(near(again.x, 7) && near(again.y, 9) && near(again.z, 9),
        "scrubbing back to frame 0 did not restore its pose");

  // --- materials --------------------------------------------------------
  const bare = mod.nativeMaterial(track.objects[1], null);
  check(near(bare.color.r, 0.4) && near(bare.color.g, 0.5) && near(bare.color.b, 0.6),
        "an object with no staged part must wear the colour the record carries");

  const link = new THREE.Group();
  const visual = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshStandardMaterial({ color: 0x804020, roughness: 0.25 }));
  visual.userData.articraftVisual = true;
  link.add(visual);
  const borrowed = mod.nativeMaterial(steady, new Map([["inst__part", link]]));
  check(borrowed.color.getHex() === 0x804020 && near(borrowed.roughness, 0.25),
        "a staged part's own material must win over Blender's viewport swatch");

  const collisionOnly = new THREE.Group();
  collisionOnly.add(new THREE.Mesh(
    new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ color: 0x00ff00 })));
  const fallback = mod.nativeMaterial(steady, new Map([["inst__part", collisionOnly]]));
  check(near(fallback.color.r, 0.1) && near(fallback.color.b, 0.3),
        "a link with only collision proxies must not lend its debug palette");

  // A link group holds the whole kinematic subtree below it, so the search for
  // "this part's material" must stop at the next link: the first visual
  // anywhere underneath a parent part is a descendant's, and the parent would
  // wear a child's color.
  const parentLink = new THREE.Group();
  const ownVisual = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshStandardMaterial({ color: 0x804020, roughness: 0.25 }));
  ownVisual.userData.articraftVisual = true;
  const jointFrame = new THREE.Group();
  jointFrame.name = "joint-frame:inst__hinge";
  const childLink = new THREE.Group();
  childLink.name = `${mod.LINK_GROUP_PREFIX}inst__child`;
  const childVisual = new THREE.Mesh(
    new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ color: 0x00ff00 }));
  childVisual.userData.articraftVisual = true;
  childLink.add(childVisual);
  jointFrame.add(childLink);
  // The child first, so taking "the first visual found" would take the wrong one.
  parentLink.add(jointFrame);
  parentLink.add(ownVisual);
  const ownColour = mod.nativeMaterial(steady, new Map([["inst__part", parentLink]]));
  check(ownColour.color.getHex() === 0x804020,
        "a link lent a material from the subtree below it, not its own visual");

  // A baked (derived) asset's color is the neutral multiplier in front of its
  // atlas, and this path cannot carry the atlas — an evaluated mesh has no UVs.
  // Borrowing the scalar drew every object white.
  const bakedLink = new THREE.Group();
  const bakedVisual = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 }));
  bakedVisual.userData.articraftVisual = true;
  bakedVisual.userData.dycraftDerivedMaterial = true;
  bakedLink.add(bakedVisual);
  const unbaked = mod.nativeMaterial(steady, new Map([["inst__part", bakedLink]]));
  check(near(unbaked.color.r, 0.1) && near(unbaked.color.b, 0.3),
        "a baked material's white multiplier was borrowed over the record's own colour");

  mod.disposeNativeScene(scene);
  check(scene.group.children.length === 0, "dispose left the group populated");

  // --- refusals ---------------------------------------------------------
  const truncated = bytes(TRACK_GZ_B64).slice(0, 64);
  let threw = false;
  try { mod.decodeNativeTrack(truncated); } catch { threw = true; }
  check(threw, "a truncated track must fail, not decode as short geometry");

  const wrongVersion = bytes(TRACK_GZ_B64);
  const view = new DataView(wrongVersion);
  const headerLength = view.getUint32(0, true);
  const text = new TextDecoder().decode(new Uint8Array(wrongVersion, 4, headerLength));
  const renegotiated = new TextEncoder().encode(
    text.replace('"version": 1', '"version": 9').padEnd(headerLength, " "));
  new Uint8Array(wrongVersion, 4, headerLength).set(renegotiated.slice(0, headerLength));
  threw = false;
  try { mod.decodeNativeTrack(wrongVersion); } catch { threw = true; }
  check(threw, "a track from a server speaking another layout must be refused");

  // --- the baked atlas --------------------------------------------------
  //
  // Three things here fail silently rather than raising, which is why they are
  // pinned: a `u16` attribute read without `normalized` samples texel 0 of a
  // 65535-wide space (one flat color); UVs keyed by mesh digest instead of by
  // part put one part's island on every object sharing that geometry; and a
  // material adopted without its map draws white, since a baked material's
  // whole appearance IS the map.
  const atlas = mod.decodeNativeAtlas(bytes(ATLAS_GZ_B64));
  check(atlas.dtype === "u16", `atlas dtype ${atlas.dtype}`);
  check(atlas.objects.length === 1 && atlas.atlases.length === 1,
        `atlas carries ${atlas.objects.length} objects / ${atlas.atlases.length} atlases`);
  check(atlas.atlases[0].key === "abc123/body",
        `atlas key ${atlas.atlases[0].key} is not <ir_id>/<part>`);
  check(atlas.atlases[0].n_corners === 3 * atlas.atlases[0].n_tris,
        "atlas corners must be three per triangle");
  check(atlas.atlases[0].n_tris === 4, `atlas n_tris ${atlas.atlases[0].n_tris}`);

  const uv = mod.atlasAttribute(atlas, "part/thing/body");
  check(uv !== null, "the object named in the payload must resolve to an attribute");
  check(uv.normalized === true,
        "a u16 uv attribute read un-normalized samples one texel for the surface");
  // `getX` on a normalized attribute already divides by 65535 — the same
  // denormalize three does in the shader — so these are UV units, not counts.
  check(uv.count === 12, `uv count ${uv?.count}`);
  check(near(uv.getX(0), 0.25, 1e-4) && near(uv.getY(0), 0.875, 1e-4),
        `first corner decoded as ${uv.getX(0)}, ${uv.getY(0)}`);
  // The payload is in glTF's top-down V, the space the baked GLB's own
  // TEXCOORD_0 is in — because `GLTFLoader` sets `flipY = false` on the map.
  // Blender's bottom-up V would be 0.125 here and would mirror every island.
  check(!near(uv.getY(0), 0.125, 1e-4),
        "atlas V must be glTF top-down, not Blender bottom-up");
  check(mod.atlasAttribute(atlas, "part/thing/missing") === null,
        "an object with no entry must resolve to null, not to another's island");
  check(mod.atlasAttribute(null, "part/thing/body") === null,
        "no atlas at all must be null rather than throwing");

  // A staged link whose visual carries a BAKED material, as the derived URDF
  // variant loads one: `stagedMaterial` must decline it without atlas UVs and
  // hand over its map with them.
  const stagedLink = () => {
    const group = new THREE.Group();
    group.name = "link_thing__body";
    const baked = new THREE.MeshStandardMaterial({ color: 0xffffff });
    baked.map = new THREE.Texture();
    const visual = new THREE.Mesh(new THREE.BufferGeometry(), baked);
    visual.userData.articraftVisual = true;
    visual.userData.dycraftDerivedMaterial = true;
    group.add(visual);
    return new Map([["thing__body", group]]);
  };

  // The invariant the whole atlas pairing rests on: `nativeGeometry` builds an
  // indexed creased mesh whose triangle `t` corner `k` is vertex `indices[3t+k]`
  // of the input, and de-indexing it for an atlas (`toNonIndexed`) keeps that
  // order. If it ever moved, every atlas UV would land on the wrong corner and
  // the surface would sample a plausible, wrong patch of the texture.
  const decodedMesh = mod.decodeNativeMesh(bytes(MESH_GZ_B64));
  const base = mod.nativeGeometry(decodedMesh);
  check(base.index !== null && base.index.count === decodedMesh.indices.length,
        `indexed geometry has ${base.index?.count} corners for `
        + `${decodedMesh.indices.length} indices`);
  const corners = base.toNonIndexed().getAttribute("position");
  let cornerDrift = 0;
  for (let c = 0; c < decodedMesh.indices.length; c += 1) {
    const v = decodedMesh.indices[c];
    for (let k = 0; k < 3; k += 1) {
      cornerDrift = Math.max(cornerDrift,
        Math.abs(corners.array[c * 3 + k] - decodedMesh.positions[v * 3 + k]));
    }
  }
  check(cornerDrift === 0,
        `corner order moved through creasedNormals (max drift ${cornerDrift})`);
  const withAtlas = mod.buildNativeScene(
    { ...track, objects: [{ name: "part/thing/body", identity: "thing/body",
                            color: [0.1, 0.2, 0.3, 1], mesh: 0, hidden: false }],
      meshes: [track.meshes[0]], n_frames: 1,
      matrices: new Float32Array(12) },
    new Map([[track.meshes[0], base]]), stagedLink(), atlas);
  check(withAtlas.atlasKeys[0] === "|abc123/body",
        `atlas key suffix ${withAtlas.atlasKeys[0]}`);
  mod.applyNativeFrame(withAtlas, 0);
  check(Boolean(withAtlas.nodes[0].geometry.getAttribute("uv")),
        "with an atlas the drawn geometry must carry uv");
  check(Boolean(withAtlas.nodes[0].material.map),
        "with an atlas the baked map must be adopted, or the part draws white");

  // A sidecar whose triangle count disagrees with the evaluated mesh must be
  // dropped rather than stretched over it: the server checks this against
  // frame 0, and the client checks it again because a mis-paired atlas draws a
  // confident, wrong patch of texture instead of failing.
  const wrongSize = { ...atlas, atlases: [{ ...atlas.atlases[0], n_corners: 3, n_tris: 1 }] };
  const mismatched = mod.buildNativeScene(
    { ...track, objects: [{ name: "part/thing/body", identity: "thing/body",
                            color: [0.1, 0.2, 0.3, 1], mesh: 0, hidden: false }],
      meshes: [track.meshes[0]], n_frames: 1,
      matrices: new Float32Array(12) },
    new Map([[track.meshes[0], base]]), stagedLink(), wrongSize);
  mod.applyNativeFrame(mismatched, 0);
  check(!mismatched.nodes[0].geometry.getAttribute("uv"),
        "a mismatched atlas must not be stretched over the evaluated mesh");
  check(Boolean(mismatched.nodes[0].geometry.getAttribute("position")),
        "a mismatched atlas must still leave the part drawable, untextured");

  const without = mod.buildNativeScene(
    { ...track, objects: [{ name: "part/thing/body", identity: "thing/body",
                            color: [0.1, 0.2, 0.3, 1], mesh: 0, hidden: false }],
      meshes: [track.meshes[0]], n_frames: 1,
      matrices: new Float32Array(12) },
    new Map([[track.meshes[0], base]]), stagedLink(), null);
  mod.applyNativeFrame(without, 0);
  check(!without.nodes[0].geometry.getAttribute("uv"),
        "with no atlas there must be no uv to mis-sample");
  check(!without.nodes[0].material.map,
        "with no atlas a baked material must still be declined");

  // --- the studio look, on both paths ------------------------------------
  //
  // A native clip's meshes hang inside the robot group, so `SceneCanvas`'s
  // studio pass traverses them — and skips them, because it filters on
  // `articraftVisual`, which only URDF-loaded visuals carry. The clip then
  // renders at the env map's full 1.0 against the staged model's clamped 0.62
  // and without the form-shading term: visibly brighter and flatter than the
  // scene it animates. Both paths must land on identical material state.
  const stagedBaked = () => {
    const group = new THREE.Group();
    group.name = "link_thing__body";
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 0.2, metalness: 0.9 });
    material.map = new THREE.Texture();
    material.envMapIntensity = 1.0;
    const visual = new THREE.Mesh(new THREE.BufferGeometry(), material);
    visual.userData.articraftVisual = true;
    visual.userData.dycraftDerivedMaterial = true;
    group.add(visual);
    return group;
  };
  const studioState = (m) => JSON.stringify({
    env: m.envMapIntensity,
    rough: Number(m.roughness.toFixed(6)),
    shaded: typeof m.customProgramCacheKey === "function"
      && String(m.customProgramCacheKey()).includes("articraft-studio-shading"),
  });

  const stagedGroup = stagedBaked();
  const stagedMat = stagedGroup.children[0].material;
  presentation.applyStudioShading(stagedMat);           // what SceneCanvas does

  // The clip borrows that material AFTER it was shaded — the ordinary order,
  // since the clip loads asynchronously.
  const nativeMat = mod.nativeMaterial(
    { name: "part/thing/body", identity: "thing/body", color: [0.1, 0.2, 0.3, 1] },
    new Map([["thing__body", stagedGroup]]), true);
  check(nativeMat.envMapIntensity === 1.0 && near(nativeMat.roughness, 0.2),
        `a borrowed material must reset to its pre-studio baseline, got `
        + `env ${nativeMat.envMapIntensity} rough ${nativeMat.roughness}`);
  presentation.applyStudioShading(nativeMat);           // what the clip does
  check(studioState(stagedMat) === studioState(nativeMat),
        `studio state differs — staged ${studioState(stagedMat)} `
        + `vs native ${studioState(nativeMat)}`);

  // Toggling Fancy off must leave the clip's material usable. A clone carries
  // a JSON-mangled snapshot whose function fields are gone, so restoring from
  // it un-guarded would assign `onBeforeCompile = undefined`.
  presentation.restoreStudioShading(stagedMat);
  presentation.restoreStudioShading(nativeMat);
  check(studioState(stagedMat) === studioState(nativeMat),
        "the two paths must also agree with the studio look off");
  check(typeof nativeMat.onBeforeCompile === "function",
        "restoring a borrowed material must not drop its onBeforeCompile");

  // The predicate itself, since two copies of it is how the paths drifted.
  const truth = (o) => Boolean(o.fancyGraphics) && !o.showCollisions
    && !o.showSegmentColors && !o.showSurfaceSamples;
  for (const bits of Array.from({ length: 16 }, (_, i) => i)) {
    const o = {
      fancyGraphics: Boolean(bits & 1), showCollisions: Boolean(bits & 2),
      showSegmentColors: Boolean(bits & 4), showSurfaceSamples: Boolean(bits & 8),
    };
    check(presentation.shouldApplyStudioShading(o) === truth(o),
          `shouldApplyStudioShading disagrees for ${JSON.stringify(o)}`);
  }

  console.log(`python-written track: ${track.n_frames} frames, `
              + `${track.objects.length} objects, ${track.meshes.length} meshes`);
  console.log(`python-written atlas: ${atlas.objects.length} object(s), `
              + `${atlas.atlases.length} atlas(es), dtype ${atlas.dtype}`);

  // --- a slow link: every k-th frame (`native-stream.ts`) -------------------
  // The stride must never thin a fast link's frames, must carry the playback rate with room to spare on a slow one,
  // must not flip back and forth at the line, and a frame between two fetched ones must count as ready.
  {
    const S = await bundle(join(WEB, "src", "components", "viewer3d", "native-stream.ts"), "stream.mjs");
    const spans = (n, ms, bytes, gap = 0) => Array.from({ length: n }, (_, k) => (
      { start: 1e6 + k * (ms + gap), end: 1e6 + k * (ms + gap) + ms, bytes }));
    // Ten back-to-back 2 MB downloads of 200 ms: a 10 MB/s link, however they overlap.
    const serial = spans(10, 200, 2e6);
    const now = serial[9].end;
    check(near(S.linkBytesPerSecond(serial, now), 1e7, 1), `serial link ${S.linkBytesPerSecond(serial, now)}`);
    const paired = serial.flatMap((t) => [{ ...t, bytes: 1e6 }, { ...t, bytes: 1e6 }]);
    check(near(S.linkBytesPerSecond(paired, now), 1e7, 1), "downloads in parallel were counted twice");
    // A loopback: 200 KB arriving in a fifth of a millisecond, with the server's own work in between not counted.
    const loop = spans(10, 0.2, 2e5, 150);
    check(S.linkBytesPerSecond(loop, loop[9].end) >= S.FAST_LINK_BYTES_PER_S,
          "a loopback measured slower than the fast-link line, so it would be thinned");
    check(S.linkBytesPerSecond(serial.slice(0, 5), now) === null, "five downloads are too few to judge a link by");

    check(S.chooseStride(S.FAST_LINK_BYTES_PER_S, 1e9, 2, 4) === 1, "a fast link was thinned");
    check(S.chooseStride(null, 35e6, 2, 3) === 3, "an unknown link changed the stride");
    // 100 Mbit/s against a long film's six characters at 2x (about 35 MB a film second): every 8th frame.
    check(S.chooseStride(12.5e6, 35e6, 2, 1) === 8,
          `six characters at 2x over 100 Mbit/s: ${S.chooseStride(12.5e6, 35e6, 2, 1)}`);
    check(S.chooseStride(12.5e6, 10e6, 2, 1) === 2, "a stride goes up as soon as the link falls short");
    // Down from 4 with 20 MB a second to carry over 12.5: 2 fits at four fifths but not at three fifths, so 3.
    check(S.chooseStride(12.5e6, 10e6, 2, 4) === 3, `coming down: ${S.chooseStride(12.5e6, 10e6, 2, 4)}`);
    check(S.chooseStride(12.5e6, 10e6, 2, 3) === 3, "a stride near the line flipped back down");

    const meshes = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"];
    const stream = new S.NativeStream({
      track: { n_frames: 8, fps: 4, meshes, objects: [{ mesh: [0, 1, 2, 3, 4, 5, 6, 7] }] },
      load: () => new Promise(() => {}), budgetBytes: 1e9, concurrency: 0,
      onChange: () => {}, displayed: () => new Set(),
    });
    for (const d of ["m0", "m2", "m4", "m6"]) stream.geometries.set(d, { index: null, attributes: {} });
    check(!stream.isReady(1), "frame 1 counted as ready with nothing thinned and its mesh missing");
    stream.strideNow = 2;
    check(stream.isReady(1) && stream.isReady(3) && stream.isReady(4), "with every 2nd frame, the frames between "
          + "two fetched ones must count as ready");
    stream.frame = 3;
    const order = stream.windowOrder();
    check(order[0] === "m3" && ["m2", "m4", "m6"].every((d) => order.includes(d)) && !order.includes("m5"),
          `with every 2nd frame from a stopped frame 3, fetched ${order}: its own frame first, then even frames`);
    // Playing, the playhead's own frame is passed before it would arrive: fetching it only to drop it wastes the link.
    stream.playing = true;
    const moving = stream.windowOrder();
    check(!moving.includes("m3") && !moving.includes("m5") && moving.includes("m4"),
          `with every 2nd frame while playing, fetched ${moving}: only the even frames`);

    // What a frame between two fetched ones shows is the mesh of the frame on the stride before it, which is what made
    // it count as ready, in either direction: not whatever was on screen last (after a seek back, a later frame's).
    const geometryOf = Object.fromEntries(["m0", "m2", "m4", "m6"].map((d) => [d, new THREE.BufferGeometry()]));
    const strided = mod.buildNativeScene(
      { version: 1, animation: "a", n_frames: 8, fps: 4, meshes,
        objects: [{ name: "body", identity: null, color: [0.5, 0.5, 0.5], mesh: [0, 1, 2, 3, 4, 5, 6, 7],
                    hidden: false }],
        matrices: Float32Array.from({ length: 8 * 12 }, (_, k) => ([0, 5, 10].includes(k % 12) ? 1 : 0)) },
      new Map(Object.entries(geometryOf)), null);
    mod.applyNativeFrame(strided, 6, 2);
    mod.applyNativeFrame(strided, 3, 2);
    check(strided.nodes[0].geometry === geometryOf.m2, "back to frame 3 with every 2nd frame, it kept frame 6's mesh "
          + "rather than showing frame 2's");
    mod.applyNativeFrame(strided, 5, 2);
    check(strided.nodes[0].geometry === geometryOf.m4,
          "forward to frame 5 with every 2nd frame, it did not show frame 4's");
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length > 0) {
  console.error(`\ncheck-native-track: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check-native-track: ok");
