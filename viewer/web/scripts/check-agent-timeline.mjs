// Check the per-agent timeline's layout and palette.
//
//     npm run check:agent-timeline
//
// Four properties, each one a bug this design can have silently:
//
//  1. **Running vs unknown.** An event with no end runs to `now` only while the
//     run itself is going; after that its duration is unknown, and drawing it
//     to `now` would grow the bar on every page open and squash the rest of
//     the axis.
//  2. **Zero-duration actions are ticks.** `record_deviation` is 1236 calls and
//     0 s across the corpus. Drawing it as a bar makes it invisible AND
//     unclickable, so it must be a tick by construction rather than by
//     rounding, and it must never widen the axis.
//  3. **The palette is the one that was validated.** Nine categories share one
//     row, so any two can be adjacent and the ALL-PAIRS floors are what apply.
//     The hexes here were optimized against those lists (normal-vision dE 17.4
//     vs a 15 floor; min(protan,deutan) dE 10.4 vs an 8 target). Order is only
//     the legend's, but a changed *hex* silently breaks the floors, so the
//     sequence is pinned across index.css and the panel.
//  4. **Every segment can be identified without color.** The single-row design
//     rests on each segment carrying its own text label; the panel must never
//     drop the accessible name.
//
// Bundled through the esbuild vite already installs, because this needs two
// pure functions and no DOM.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const SOURCE = join(WEB, "src", "panels", "AgentTimeline.tsx");

const MINUTE = 60_000;
const T = (m) => new Date(Date.UTC(2026, 0, 1, 0, m)).toISOString();
const NOW = Date.UTC(2026, 0, 1, 12, 0); // twelve hours after the run started

function event(id, category, startedAt, finishedAt, extra = {}) {
  return {
    id, order: 0, category, action: `op:${category}`, label: id,
    step: null, turn: null,
    started_at: startedAt, finished_at: finishedAt,
    duration_ms:
      startedAt && finishedAt
        ? new Date(finishedAt).getTime() - new Date(startedAt).getTime()
        : null,
    status: "ok", cache_hit: null, zero_duration: false,
    source: "trace", detail: {},
    ...extra,
  };
}

const failures = [];
function check(what, condition, detail) {
  if (!condition) failures.push(`${what}${detail ? `: ${detail}` : ""}`);
}

const work = mkdtempSync(join(tmpdir(), "dycraft-agent-timeline-"));
try {
  const bundle = join(work, "agent-timeline.mjs");
  execFileSync(
    join(WEB, "node_modules", ".bin", "esbuild"),
    // Everything bundled, react included: the output is imported from a temp
    // directory where bare specifiers would not resolve. The functions under
    // test run no React; the components around them are only defined.
    [SOURCE, "--bundle", "--format=esm", "--platform=node", "--log-level=error",
     `--alias:@=${join(WEB, "src")}`, `--outfile=${bundle}`],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  const { layoutSegments, tickStepMs } = await import(pathToFileURL(bundle).href);

  // 1. An open-ended event, read both ways.
  {
    const open = [event("open", "build", T(10), null)];
    const live = layoutSegments(open, {
      startedAt: T(0), finishedAt: null, now: NOW, runFinished: false,
    });
    check("a live run's open event reaches now",
      live.placed[0].width > 0, `width ${live.placed[0].width}`);
    check("a live run's axis extends to now",
      live.origin + live.span >= NOW, `${live.origin + live.span} < ${NOW}`);

    const dead = layoutSegments(open, {
      startedAt: T(0), finishedAt: T(20), now: NOW, runFinished: true,
    });
    check("a finished run's open event is a tick", dead.placed[0].tick);
    check("a finished run's open event has no width",
      dead.placed[0].width === 0, `width ${dead.placed[0].width}`);
    // The whole point: `now` must not touch the axis of a finished run.
    check("a finished run's axis ignores now",
      dead.origin + dead.span < NOW,
      `axis end ${dead.origin + dead.span} reached now ${NOW}`);
    const deadAgain = layoutSegments(open, {
      startedAt: T(0), finishedAt: T(20), now: NOW + 86_400_000,
      runFinished: true,
    });
    check("a finished run's axis does not grow as now advances",
      deadAgain.span === dead.span, `${deadAgain.span} vs ${dead.span}`);
  }

  // 2. Zero-duration actions.
  {
    const mixed = [
      event("compile", "build", T(0), T(30)),
      event("deviation", "deliver", T(10), null, { zero_duration: true }),
    ];
    const { placed, span } = layoutSegments(mixed, {
      startedAt: T(0), finishedAt: T(30), now: NOW, runFinished: true,
    });
    const dev = placed.find((p) => p.event.id === "deviation");
    check("a zero-duration action is a tick", dev.tick);
    check("a zero-duration action has no width", dev.width === 0,
      `width ${dev.width}`);
    check("a zero-duration action sits at its own timestamp",
      Math.abs(dev.from - 1 / 3) < 1e-9, `from ${dev.from}`);
    check("a zero-duration action does not widen the axis",
      span === 30 * MINUTE, `span ${span}`);
    // Even on a live run it must stay a tick — it has no duration to show.
    const liveDev = layoutSegments([mixed[1]], {
      startedAt: T(0), finishedAt: null, now: NOW, runFinished: false,
    });
    check("a zero-duration action is a tick on a live run too",
      liveDev.placed[0].tick && liveDev.placed[0].width === 0);
    check("a zero-duration action does not drag a live axis to now",
      liveDev.origin + liveDev.span < NOW,
      `axis end ${liveDev.origin + liveDev.span}`);
  }

  // 3. Degenerate input must not divide by zero.
  {
    const instant = layoutSegments([event("blip", "build", T(5), T(5))], {
      startedAt: T(5), finishedAt: T(5), now: NOW, runFinished: true,
    });
    check("a zero-length run keeps a nonzero span", instant.span > 0,
      String(instant.span));
    const empty = layoutSegments([], {
      startedAt: null, finishedAt: null, now: NOW, runFinished: true,
    });
    check("an empty timeline keeps a nonzero span", empty.span > 0,
      String(empty.span));
    const undated = layoutSegments([event("nowhere", "aside", null, null)], {
      startedAt: null, finishedAt: null, now: NOW, runFinished: true,
    });
    check("an undated event is a tick at the origin",
      undated.placed[0].tick && undated.placed[0].from === 0);
    check("every placed event has a finite position",
      undated.placed.every((p) => Number.isFinite(p.from) && Number.isFinite(p.width)));
  }

  // 3b. Row packing: one row unless segments genuinely overlap.
  {
    const serial = [event("a", "build", T(0), T(10)),
                    event("b", "inspect", T(10), T(20)),
                    event("c", "look", T(20), T(30))];
    const { rows, placed } = layoutSegments(serial, {
      startedAt: T(0), finishedAt: T(30), now: NOW, runFinished: true,
    });
    check("a sequential run is one row", rows === 1, `${rows} rows`);
    check("every sequential segment is on row 0",
      placed.every((p) => p.row === 0), placed.map((p) => p.row).join(","));

    const concurrent = [event("long", "build", T(0), T(20)),
                        event("p1", "inspect", T(5), T(9)),
                        event("p2", "look", T(5), T(9))];
    const packed = layoutSegments(concurrent, {
      startedAt: T(0), finishedAt: T(20), now: NOW, runFinished: true,
    });
    check("genuine overlap stacks into extra rows", packed.rows === 3,
      `${packed.rows} rows`);
    const byId = new Map(packed.placed.map((p) => [p.event.id, p.row]));
    check("the long segment keeps row 0", byId.get("long") === 0,
      `row ${byId.get("long")}`);
  }

  // 4. Ticks land at most eight to the axis, at every scale.
  {
    for (const span of [1, 1000, 90_000, 3 * MINUTE, 47 * MINUTE,
                        6 * 3_600_000, 30 * 3_600_000]) {
      const step = tickStepMs(span);
      check(`tick step divides a ${span}ms axis into <= 9 gridlines`,
        span / step <= 9, `${Math.ceil(span / step)} gridlines`);
      check(`tick step is positive for a ${span}ms axis`, step > 0, String(step));
    }
  }

  // 5. The palette and the category order, pinned across both files.
  {
    const expected = ["model", "judge", "author", "build", "inspect", "look",
                      "solve", "deliver", "aside"];
    const tsx = readFileSync(SOURCE, "utf8");
    const fillBlock = tsx.slice(tsx.indexOf("const FILL"));
    const order = [...fillBlock.slice(0, fillBlock.indexOf("};"))
      .matchAll(/^\s{2}(\w+):/gm)].map((m) => m[1]);
    check("FILL lists the categories in the documented order",
      order.join(",") === expected.join(","), order.join(","));

    const inkBlock = tsx.slice(tsx.indexOf("const INK"));
    const inkOrder = [...inkBlock.slice(0, inkBlock.indexOf("};"))
      .matchAll(/^\s{2}(\w+):/gm)].map((m) => m[1]);
    check("INK covers exactly the same categories",
      inkOrder.join(",") === expected.join(","), inkOrder.join(","));

    const css = readFileSync(join(WEB, "src", "index.css"), "utf8");
    const cssOrder = [...css.matchAll(/--cat-(\w+):\s*#/g)].map((m) => m[1]);
    check("index.css declares the categories in the same order",
      cssOrder.join(",") === expected.join(","), cssOrder.join(","));

    // The hexes ARE the validation. Changing one without re-running
    // scripts/validate_palette.js can drop the all-pairs floors silently.
    const hues = [...css.matchAll(/--cat-\w+:\s*(#[0-9a-f]{6})/g)].map((m) => m[1]);
    check("the category hues are the validated set",
      hues.join(",") ===
        "#078cbb,#b28ffe,#0948b3,#f09f14,#3fcbb9,#8e7c04,#920864,#176407,#f05e72",
      hues.join(","));
    check("no category is white or grey — every one is a real hue",
      hues.every((h) => {
        const [r, g, b] = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
        return Math.max(r, g, b) - Math.min(r, g, b) > 24;
      }), hues.join(","));
    const inks = [...css.matchAll(/--cat-\w+-ink:\s*(#[0-9a-f]{6})/g)].map((m) => m[1]);
    check("every category has a label ink", inks.length === expected.length,
      `${inks.length} inks for ${expected.length} categories`);
  }

  // 5b. The texture channel has exactly two meanings and they must stay
  //     distinguishable: 45 deg is failure, its 135 deg mirror is "the model
  //     was composing a tool call". Same angle for both would make a hatched
  //     segment ambiguous, and the palette rules allow only these two angles.
  {
    const css = readFileSync(join(WEB, "src", "index.css"), "utf8");
    const angleOf = (cls) => {
      const block = css.slice(css.indexOf(`.${cls} {`));
      const m = block.slice(0, block.indexOf("}")).match(/(\d+)deg/);
      return m ? Number(m[1]) : null;
    };
    const failed = angleOf("seg-failed");
    const writing = angleOf("seg-writing");
    check("the failure hatch is 45deg", failed === 45, String(failed));
    check("the writing hatch is its 135deg mirror", writing === 135,
      String(writing));
    check("the two textures are not the same angle", failed !== writing,
      `both ${failed}deg`);
    // The writing hatch sits behind the label, so it must not be opaque
    // enough to win a legibility fight with it — the 0.62 failure hatch did.
    const alpha = (cls) => {
      const block = css.slice(css.indexOf(`.${cls} {`));
      const m = block.slice(0, block.indexOf("}")).match(/rgba\([^)]*?([\d.]+)\)/);
      return m ? Number(m[1]) : null;
    };
    for (const cls of ["seg-failed", "seg-writing"]) {
      const a = alpha(cls);
      check(`${cls} stays faint enough to read a label through`,
        a !== null && a <= 0.4, `alpha ${a}`);
    }
  }

  // 6. Identity must never be color-alone. Every segment's accessible name
  //    carries its category, its own label, and its status, so a screen reader,
  //    or a colourblind reader never depends on the fill; and the inline label
  //    may only be withheld for want of pixels, never for want of a value.
  {
    const tsx = readFileSync(SOURCE, "utf8");
    check("there is exactly one Segment renderer, shared by both views",
      (tsx.match(/^function Segment\(/gm) || []).length === 1,
      "the two layouts must not each build their own segment");
    const seg = tsx.slice(tsx.indexOf("function Segment("),
                          tsx.indexOf("/** The legend"));
    for (const part of ["categoryLabel", "event.label", "event.status"]) {
      check(`the accessible name includes ${part}`,
        new RegExp(`aria-label=\\{\`[^\`]*\\$\\{${part
          .replace(".", "\\.")}`).test(seg)
        || seg.includes(part),
        "not found in Segment's aria-label");
    }
    check("the inline label is gated only on showLabel",
      /\{showLabel \? \(/.test(seg), "showLabel gate not found");
    check("showLabel is a pixel test in the time view",
      /showLabel=\{width \* plotWidth >= MIN_LABEL_PX\}/.test(tsx));
    check("showLabel is unconditional in the order view",
      /\n\s+showLabel\n/.test(tsx),
      "the order view must always print its labels");
    check("the order view uses a fixed column width and scrolls",
      /ORDER_COL_PX/.test(tsx) && /overflow-x-auto/.test(tsx),
      "an even split makes every label unreadable at 82 segments");
    // The hatch is decoration over a measured value; the words must be there
    // too, or the split is invisible to a screen reader and in monochrome.
    check("the think/write split is stated in the label, not only hatched",
      /thinking/.test(readFileSync(
        join(WEB, "..", "agent_timeline.py"), "utf8"))
      && /writing/.test(readFileSync(
        join(WEB, "..", "agent_timeline.py"), "utf8")),
      "agent_timeline.py must put the split in the segment label");
  }

} finally {
  rmSync(work, { recursive: true, force: true });
}

if (failures.length) {
  console.error(`FAIL  ${failures.length} check(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exitCode = 1;
} else {
  console.log(
    "OK  agent timeline: running vs unknown, zero-duration ticks, row packing, "
    + "degenerate input, tick density, validated palette, accessible names");
}
