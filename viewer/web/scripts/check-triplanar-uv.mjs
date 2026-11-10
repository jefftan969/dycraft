// Check the browser's triplanar UV rule against the renderer's own arithmetic.
//
//     npm run check:uv
//
// A base-color texture is parameterized procedurally rather than stored, so
// `render/open3d_renderer.py::_triplanar_uvs` and
// `src/components/viewer3d/triplanar-uv.ts` are two independent implementations
// of one rule. A disagreement between them is SILENT: both draw wood, on
// different parts of the plank, and the viewport quietly stops being an honest
// preview of the frames the renderer draws. Nothing else in either test suite
// would catch that, so this does.
//
// `tests/fixtures/triplanar_uv_golden.json` carries the expected UVs, and
// `tests/test_texture_render.py::test_the_triplanar_golden_is_what_the_
// renderer_produces` asserts the fixture is what the Python produces — so
// neither end is pinned to the other, both are pinned to the renderer.
//
// Transpiled through the esbuild vite already installs.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const REPO = resolve(WEB, "..", "..");
const GOLDEN = join(REPO, "tests", "fixtures", "triplanar_uv_golden.json");
const SOURCE = join(WEB, "src", "components", "viewer3d", "triplanar-uv.ts");
const MATERIALS = join(WEB, "src", "components", "viewer3d", "materials.ts");
const GEOMETRY = join(WEB, "src", "components", "viewer3d", "geometry-loader.ts");

const work = mkdtempSync(join(tmpdir(), "dycraft-uv-"));
try {
  const bundle = join(work, "uv.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [SOURCE, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${bundle}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );

  const { triplanarUVs } = await import(pathToFileURL(bundle).href);
  const golden = JSON.parse(readFileSync(GOLDEN, "utf8"));
  const tol = golden.tolerance;

  let worst = 0;
  let worstCase = "";
  let checked = 0;
  for (const c of golden.cases) {
    // Expanded to a non-indexed soup, because that is the layout the viewer
    // actually runs on: `applyTriplanarUVs` de-indexes before writing, since
    // three's per-VERTEX `uv` attribute can only equal Open3D's per-CORNER
    // `triangle_uvs` when every triangle owns its vertices. Checking the
    // indexed form would be testing a configuration no call site produces.
    const soup = new Float32Array(c.triangles.length * 3);
    for (let k = 0; k < c.triangles.length; k += 1) {
      const v = c.triangles[k];
      soup[k * 3] = c.vertices[v * 3];
      soup[k * 3 + 1] = c.vertices[v * 3 + 1];
      soup[k * 3 + 2] = c.vertices[v * 3 + 2];
    }
    const got = triplanarUVs(soup, null, c.scale_m);
    if (got.length !== c.uvs.length) {
      throw new Error(
        `${c.name}: got ${got.length} uv components, golden has ${c.uvs.length}`);
    }
    for (let i = 0; i < got.length; i += 1) {
      const d = Math.abs(got[i] - c.uvs[i]);
      if (d > worst) { worst = d; worstCase = c.name; }
      checked += 1;
    }
  }

  if (!(worst <= tol)) {
    console.error(
      `check:uv FAILED — worst |ts - python| = ${worst.toExponential(3)} on ` +
      `'${worstCase}' (tolerance ${tol}). The two renderers would texture the ` +
      `same surface differently.`);
    process.exit(1);
  }
  // ---------------------------------------------------------------------
  // The other half of the contract: what the URDF's <texture> turns into.
  //
  // `parseMaterial` itself needs a DOM and node has none (no jsdom here, and
  // adding one for this would be a dependency for a shim). Its OUTPUT shape is
  // pinned from the Python side instead — `test_the_urdf_carries_the_texture_
  // and_its_tile_size` asserts the attributes exist — so what is left to check
  // is the pure function that consumes it, which is where the behavior lives.
  const matBundle = join(work, "materials.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [MATERIALS, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${matBundle}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  const { resolveVisualMaterialSpec, resolveTextureUrl, createMaterial, THREE_FOR_TESTS,
          EXPORTED_DEFAULT_METALNESS, EXPORTED_DEFAULT_ROUGHNESS } =
    await import(pathToFileURL(matBundle).href);

  const visual = (material) => ({ geometry: { type: "box" }, material });
  const wood = {
    name: "top_mat",
    color: { rgba: [0.588, 0.412, 0.275, 1] },
    texture: { filename: "assets/textures/deadbeef.jpg", scaleM: 0.22,
               maps: { roughness: "assets/textures/cafebabe.png" } },
  };

  const t = resolveVisualMaterialSpec(visual(wood));
  const fail = (msg) => { console.error(`check:uv FAILED — ${msg}`); process.exit(1); };
  if (t.texturePath !== wood.texture.filename) fail("texturePath not carried");
  if (t.textureScaleM !== 0.22) fail("tile size not carried");
  if (t.textureMaps?.roughness !== wood.texture.maps.roughness) {
    fail("the roughness map path was not carried through the spec");
  }
  // Kinds are dycraft library data, so an unknown one must ride through
  // untouched rather than being dropped by a parser that does not know it.
  const future = resolveVisualMaterialSpec(visual({
    ...wood,
    texture: { ...wood.texture, maps: { ...wood.texture.maps, normal: "n.png" } },
  }));
  if (future.textureMaps?.normal !== "n.png") {
    fail("an unknown map kind was dropped; kinds must be carried opaquely");
  }
  // The color survives as the fallback rather than being replaced here; the
  // white swap happens in `createMaterial`, where the map is actually bound.
  // Compared against the untextured spec for the same rgba rather than a
  // literal, because `srgbColor` linearizes and hardcoding the converted value
  // would pin three's color management instead of this file's behavior.
  const plainSame = resolveVisualMaterialSpec(
    visual({ name: "plain_mat", color: wood.color }));
  if (!t.color.equals(plainSame.color)) {
    fail(`fallback colour changed under a texture ` +
         `(#${t.color.getHexString()} vs #${plainSame.color.getHexString()})`);
  }
  if (t.color.equals(new (t.color.constructor)(1, 1, 1))) {
    fail("the spec was whitened here; that belongs in createMaterial");
  }

  // A textured material must NOT go through the name heuristics: those guess a
  // finish from substrings of the material name, which `urdf_export` writes as
  // `<link>_mat` — so they fire on PART names. A part called `brass_rail` with
  // a wood texture would otherwise come back at metalness 0.88.
  const brass = resolveVisualMaterialSpec(visual({ ...wood, name: "brass_rail_mat" }));
  if (brass.metalness > 0.1) {
    fail(`name heuristic overrode a texture (metalness ${brass.metalness})`);
  }
  // ...and neither may an UNtextured one: `<pbr>` carries the authored
  // scalars, so a part name must not move them.
  const brassPlain = resolveVisualMaterialSpec(visual({
    name: "brass_rail_mat",
    color: { rgba: [0.5, 0.5, 0.5, 1] },
    pbr: { metalness: 0.0, roughness: 0.92 },
  }));
  if (brassPlain.metalness !== 0.0 || brassPlain.roughness !== 0.92) {
    fail(`a part name overrode the authored scalars ` +
         `(m=${brassPlain.metalness}, r=${brassPlain.roughness}; wanted 0/0.92)`);
  }

  // ---------------------------------------------------------------------
  // <pbr> — the scalars the viewport could not see until now.
  //
  // `render/urdf_export.py` writes `<pbr><metal><metalness>/<roughness>` and
  // nothing here read it, so every untextured part drew at one neutral
  // default. `tests/test_urdf_export.py::test_the_urdf_carries_the_authored_
  // pbr_scalars_the_viewer_reads` pins the shape from the Python side; what is
  // checked here is that the spec honors it.
  const authoredPbr = resolveVisualMaterialSpec(visual({
    name: "drop_mat", color: { rgba: [0.9, 0.9, 0.95, 1] },
    pbr: { metalness: 0.0, roughness: 0.05 },
  }));
  if (authoredPbr.roughness !== 0.05) {
    fail(`an authored roughness of 0.05 came back as ${authoredPbr.roughness}`);
  }
  const steel = resolveVisualMaterialSpec(visual({
    name: "frame_mat", color: { rgba: [0.6, 0.62, 0.66, 1] },
    pbr: { metalness: 1.0, roughness: 0.35 },
  }));
  if (steel.metalness !== 1.0 || steel.roughness !== 0.35) {
    fail(`an authored metal came back at m=${steel.metalness} r=${steel.roughness}`);
  }

  // A surface authored at the SDK defaults gets NO <pbr> block: the exporter
  // leaves it out when nothing differs from the defaults. So absent must
  // decode to those defaults, not to a neutral 0.05/0.48, or the most common
  // part is drawn wrong.
  const suppressed = resolveVisualMaterialSpec(
    visual({ name: "brace_mat", color: { rgba: [0.7, 0.7, 0.7, 1] } }));
  if (suppressed.metalness !== EXPORTED_DEFAULT_METALNESS
      || suppressed.roughness !== EXPORTED_DEFAULT_ROUGHNESS) {
    fail(`an absent <pbr> decoded to m=${suppressed.metalness} ` +
         `r=${suppressed.roughness}, not the exporter's suppressed defaults`);
  }

  // A textured material takes its scalars from <pbr> too.
  const texPbr = resolveVisualMaterialSpec(visual({
    ...wood, pbr: { metalness: 0.0, roughness: 0.31 } }));
  if (texPbr.roughness !== 0.31) {
    fail(`a textured material ignored its authored roughness ` +
         `(${texPbr.roughness}, wanted 0.31)`);
  }

  // What <pbr> genuinely cannot carry still comes from the name: refraction is
  // not in the schema, so the glass preset remains the only source of it —
  // but it must contribute transmission WITHOUT moving the authored scalars.
  const glass = resolveVisualMaterialSpec(visual({
    name: "glass_cover_mat", color: { rgba: [0.8, 0.85, 0.9, 0.28] },
    pbr: { metalness: 0.0, roughness: 0.08 },
  }));
  if (!(glass.transmission > 0.4)) {
    fail("the glass preset stopped contributing transmission");
  }
  if (glass.roughness !== 0.08 || glass.metalness !== 0.0) {
    fail(`the glass preset overrode authored scalars ` +
         `(m=${glass.metalness}, r=${glass.roughness})`);
  }
  if (Math.abs(glass.opacity - 0.28) > 1e-6) {
    fail(`the authored alpha was moved by a preset (${glass.opacity})`);
  }

  // A bright neutral's color and roughness are carried values, so a preset
  // must not soften the one or floor the other.
  const white = resolveVisualMaterialSpec(visual({
    name: "shell_mat", color: { rgba: [0.933, 0.933, 0.945, 1] },
    pbr: { metalness: 0.0, roughness: 0.24 },
  }));
  const whiteRef = resolveVisualMaterialSpec(visual({
    name: "x_mat", color: { rgba: [0.933, 0.933, 0.945, 1] },
    pbr: { metalness: 0.0, roughness: 0.24 },
  }));
  if (white.roughness !== 0.24) {
    fail(`a gloss-white shell was flattened to ${white.roughness} from 0.24`);
  }
  if (!white.color.equals(whiteRef.color)) {
    fail("softenBrightDiffuse still moves the authored colour");
  }

  // A texture with no tile size is not usable: `geometry-loader` skips UVs
  // rather than guessing, because a guessed size puts the viewport's grain at a
  // different scale from the canonical render.
  const noScale = resolveVisualMaterialSpec(
    visual({ ...wood, texture: { filename: "x.png" } }));
  if (noScale.textureScaleM !== undefined) fail("a missing tile size was invented");

  if (resolveTextureUrl("/api/runs/r/files", "assets/textures/x.png")
      !== "/api/runs/r/files/assets/textures/x.png") {
    fail("texture URL resolution does not match the files route");
  }

  // ---------------------------------------------------------------------
  // The white-binding, which is the claim most likely to be quietly wrong.
  //
  // three multiplies `map` by `color` exactly as Filament multiplies
  // `albedo_img` by `base_color`. Getting this wrong on the Python side made
  // the same wood render gray over a blue base and at 0.59/0.41/0.27 over the
  // library's own brown, and the agent prompt PROMISES `color` is a fallback
  // rather than a tint. Both renderers have to keep that true.
  //
  // `TextureLoader.load` reaches for an ImageLoader and there is no DOM here,
  // so it is stubbed: what is under test is what `createMaterial` does with the
  // Texture, not three's ability to fetch a PNG in node.
  const THREE = THREE_FOR_TESTS;
  const realLoad = THREE.TextureLoader.prototype.load;
  THREE.TextureLoader.prototype.load = function stub() { return new THREE.Texture(); };
  try {
    const mat = createMaterial(t, { textureBaseUrl: "/api/runs/r/files" });
    if (!mat.map) fail("createMaterial did not bind the texture");
    if (!(mat.color.r === 1 && mat.color.g === 1 && mat.color.b === 1)) {
      fail(`base colour under a texture is #${mat.color.getHexString()}, not ` +
           `white — the texture would be darkened by the authored colour`);
    }
    // The roughness map is the same multiply trap as the albedo: three does
    // `roughnessFactor *= texelRoughness.g`, so the scalar must be neutralized
    // or the surface's own roughness is applied twice. Filament does the same
    // thing with `base_roughness`, and `open3d_renderer._pbr_material` handles
    // it there — the two have to agree or the viewport and the canonical render
    // disagree about how glossy a wooden table is.
    if (!mat.roughnessMap) fail("createMaterial did not bind the roughness map");
    if (mat.roughness !== 1.0) {
      fail(`roughness is ${mat.roughness}, not 1.0 — the map would be ` +
           `multiplied by the authored scalar`);
    }
    if (mat.roughnessMap.colorSpace === THREE.SRGBColorSpace) {
      fail("the roughness map was decoded as sRGB; it is measured data and " +
           "a gamma curve skews every value glossy");
    }
    if (mat.map.colorSpace !== THREE.SRGBColorSpace) {
      fail("the base-colour map was NOT decoded as sRGB");
    }

    const plain = createMaterial(plainSame, { textureBaseUrl: "/api/runs/r/files" });
    if (plain.map) fail("an untextured spec bound a map");
    if (plain.roughnessMap) fail("an untextured spec bound a roughness map");
    if (plain.roughness === 1.0 && plainSame.roughness !== 1.0) {
      fail("an untextured material lost its authored roughness");
    }

    // A base color with no roughness map keeps the authored scalar: that is
    // the fallback the library's `base_roughness` supplies.
    const noRough = resolveVisualMaterialSpec(visual({
      ...wood, texture: { filename: "x.jpg", scaleM: 0.2, maps: {} } }));
    const noRoughMat = createMaterial(noRough, { textureBaseUrl: "/api/runs/r/files" });
    if (noRoughMat.roughnessMap) fail("a roughness map appeared from nowhere");
    if (noRoughMat.roughness !== noRough.roughness) {
      fail("the scalar roughness was neutralised with no map to justify it");
    }
    if (!plain.color.equals(plainSame.color)) {
      fail("an untextured material lost its colour");
    }
    // No base URL means nothing to resolve against, so no map and no whitening.
    const noBase = createMaterial(t);
    if (noBase.map || noBase.color.equals(new THREE.Color(1, 1, 1))) {
      fail("a texture was bound (or the colour whitened) with no base URL");
    }
  } finally {
    THREE.TextureLoader.prototype.load = realLoad;
  }

  // ---------------------------------------------------------------------
  // Staged UVs win; projection is the fallback.
  //
  // A staged OBJ carries per-corner coordinates — baked from the B-rep for a
  // CAD solid, authored into the asset for an imported one — and `OBJLoader`
  // reads them into `uv`. Re-projecting over them is SILENT, because the result
  // still looks textured; the only symptom is a barrel cut into three planar
  // patches instead of wrapped, or an atlas landing where a world axis fell
  // rather than where its author put it. `open3d_renderer._apply_texture_uvs`
  // holds the same guard, and the two have to agree or the viewport and the
  // canonical render disagree about the same surface.
  const geomBundle = join(work, "geometry.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    [GEOMETRY, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${geomBundle}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  const { applyTextureUVs } = await import(pathToFileURL(geomBundle).href);

  const texturedSpec = { texturePath: "assets/textures/x.png", textureScaleM: 0.25,
                         color: new THREE_FOR_TESTS.Color(1, 1, 1) };
  const quad = () => {
    const g = new THREE_FOR_TESTS.BufferGeometry();
    g.setAttribute("position", new THREE_FOR_TESTS.BufferAttribute(
      new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0]), 3));
    return new THREE_FOR_TESTS.Mesh(g);
  };

  // Authored coordinates that a projection would NOT produce: a +Z-facing
  // triangle projects to (x, y), so a v running backwards can only come from
  // the file. Asserting the exact values, not merely that some exist.
  const authored = quad();
  const own = new Float32Array([0, 1, 1, 1, 1, 0]);
  authored.geometry.setAttribute("uv", new THREE_FOR_TESTS.BufferAttribute(own, 2));
  applyTextureUVs(authored, texturedSpec, { shared: false });
  const kept = authored.geometry.getAttribute("uv").array;
  for (let i = 0; i < own.length; i += 1) {
    if (kept[i] !== own[i]) {
      fail(`the mesh's own UVs were overwritten by the projection ` +
           `(component ${i}: ${kept[i]} != ${own[i]})`);
    }
  }

  // And geometry that arrived with none still gets them, which is the case
  // that keeps analytic primitives and room surfaces textured at all.
  const bare = quad();
  applyTextureUVs(bare, texturedSpec, { shared: false });
  const made = bare.geometry.getAttribute("uv");
  if (!made || made.count !== 3) {
    fail("a mesh with no UVs was not given projected ones");
  }

  console.log(
    `check:uv OK — ${golden.cases.length} cases, ${checked} components, ` +
    `worst |ts - python| = ${worst.toExponential(3)} (tol ${tol}); ` +
    `material spec carries path + tile size, bypasses the name ` +
    `heuristics, binds white under a texture, neutralises the ` +
    `roughness scalar only when a map replaces it, and honours the ` +
    `authored <pbr> scalars over every name guess`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
