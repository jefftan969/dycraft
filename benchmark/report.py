"""Put the arms of a benchmark directory side by side.

    python -m dycraft.benchmark.score  runs/bench_short_codex      # optional: the physical scores
    python -m dycraft.benchmark.report runs/bench_short_codex

Reads every `<out>/<brief>/<arm>/benchmark/record.json` that `benchmark.run` and `benchmark.bare`
wrote, prints one table, and writes `<out>/report.html`: per brief, the arms' films side by side,
with each arm's outcome (a dycraft arm's own verdict), the film's length, where a bare arm's film came
from (the agent, or the harness rendering a blend that had none) and the wall time. Where
`benchmark.score` has run, each arm also shows the paper's penetration and floating rates
(`benchmark/score.json`), with each arm's mean over its scored briefs, as in the paper's Table 2;
and its agent-hours per minute of film, main session and with sub-agents (`benchmark/cost.py`).
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

COLUMNS = (("outcome", "outcome"), ("seconds", "film s"), ("film_source", "film from"),
           ("wall", "wall h"), ("penetration", "penetration %"), ("floating", "floating %"),
           ("cost", "agent h / film min"))


def _cost(arm: Path) -> dict | None:
    """Agent-hours per minute of film (`benchmark/cost.py`), or None where it cannot be told."""
    try:
        from .cost import cost
    except ImportError:
        return None
    try:
        return cost(arm)
    except (OSError, ValueError, KeyError):
        return None


def collect(out: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for record in sorted(out.glob("*/*/benchmark/record.json")):
        r = json.loads(record.read_text())
        score = record.parent / "score.json"
        if score.is_file():
            r["score"] = json.loads(score.read_text())
        r["cost"] = _cost(record.parents[1])
        rows.setdefault(r.get("slug") or record.parents[2].name, []).append(r)
    return rows


def _pct(value) -> str | None:
    return None if value is None else f"{100 * value:.1f}"


def _cells(r: dict) -> dict:
    if r["arm"].startswith("dycraft"):
        outcome = r.get("outcome") or "no finish"
        if r.get("unmet"):
            outcome += f" ({r['unmet']} unmet)"
        source = "dycraft" if r.get("film") else None
    else:
        outcome = "delivered" if r.get("blend") else "no blend"
        source = r.get("film_source")
    if r.get("time_limit"):
        outcome += ", time limit"
    if r.get("hard_stop"):
        outcome += ", hard stop"
    staged = r.get("staged")
    seconds = (("staging failed: " + staged["error"][:80]) if isinstance(staged, dict)
               and staged.get("error") else r.get("seconds"))
    score, cost = r.get("score") or {}, r.get("cost") or {}
    per_minute = [cost.get(k) for k in ("hours_per_film_minute_main", "hours_per_film_minute_total")]
    return {"outcome": outcome, "seconds": seconds, "film_source": source,
            "wall": round(r["wall_seconds"] / 3600, 2) if r.get("wall_seconds") else None,
            "penetration": _pct(score.get("penetration_rate")), "floating": _pct(score.get("floating_rate")),
            "cost": (" / ".join(f"{v:.2f}" for v in per_minute) if None not in per_minute else None)}


def means(rows: dict[str, list[dict]]) -> list[list[str]]:
    """Per arm and model: the mean penetration and floating rates over the briefs it was scored on."""
    by_arm: dict[str, dict[str, list[float]]] = {}
    for arms in rows.values():
        for r in arms:
            score = r.get("score") or {}
            m = by_arm.setdefault(f"{r['arm']} ({r.get('model') or 'default'})", {"penetration": [], "floating": []})
            for key in m:
                if score.get(f"{key}_rate") is not None:
                    m[key].append(score[f"{key}_rate"])
    out = []
    for arm, m in sorted(by_arm.items()):
        if m["penetration"] or m["floating"]:
            out.append([arm] + [f"{_pct(sum(v) / len(v))} ({len(v)} briefs)" if v else "" for v in m.values()])
    return out


def text_table(rows: dict[str, list[dict]]) -> str:
    head = ["brief", "arm", *[label for _, label in COLUMNS]]
    lines = [head]
    for slug, arms in rows.items():
        for r in sorted(arms, key=lambda r: r["arm"]):
            c = _cells(r)
            lines.append([slug, f"{r['arm']} ({r.get('model') or 'default'})",
                          *["" if c.get(k) is None else str(c[k]) for k, _ in COLUMNS]])
    text = _columns(lines)
    summary = means(rows)
    if summary:
        text += "\n\nmean over scored briefs\n" + _columns([["arm", "penetration %", "floating %"], *summary])
    return text


def _columns(lines: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in lines)


def html_report(out: Path, rows: dict[str, list[dict]]) -> str:
    esc = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>dycraft benchmark</title>",
             "<style>body{font:14px system-ui,sans-serif;margin:24px}table{border-collapse:collapse;"
             "margin:8px 0 24px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
             ".films{display:flex;gap:16px;flex-wrap:wrap}.films figure{margin:0}"
             "video{width:480px;background:#000}</style>",
             f"<h1>dycraft benchmark — {esc(str(out))}</h1>",
             "<p>Each arm's film, side by side. A bare arm's film is the agent's own, or, when it "
             "delivered none, rendered by the harness from its blend. The penetration and floating "
             "rates (lower is better) appear where <code>python -m dycraft.benchmark.score</code> has "
             "run; the cost is agent-hours per minute of film, main session / with sub-agents.</p>"]
    summary = means(rows)
    if summary:
        parts.append("<h2>Mean over scored briefs</h2><table><tr><th>arm</th><th>penetration %</th>"
                     "<th>floating %</th></tr>"
                     + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in summary)
                     + "</table>")
    for slug, arms in rows.items():
        arms = sorted(arms, key=lambda r: r["arm"])
        parts.append(f"<h2>{esc(slug)}</h2><div class='films'>")
        for r in arms:
            film = r.get("film")
            label = f"{r['arm']} ({r.get('model') or 'default'})"
            if film:
                rel = os.path.relpath(film, out)
                parts.append(f"<figure><video controls preload='metadata' src='{esc(rel)}'></video>"
                             f"<figcaption>{esc(label)}</figcaption></figure>")
            else:
                parts.append(f"<figure><div style='width:480px;padding:40px 0;text-align:center;"
                             f"background:#eee'>no film</div><figcaption>{esc(label)}</figcaption></figure>")
        parts.append("</div><table><tr><th>arm</th>"
                     + "".join(f"<th>{esc(label)}</th>" for _, label in COLUMNS) + "</tr>")
        for r in arms:
            c = _cells(r)
            parts.append(f"<tr><td>{esc(r['arm'])}</td>"
                         + "".join(f"<td>{esc('' if c.get(k) is None else str(c[k]))}</td>"
                                   for k, _ in COLUMNS) + "</tr>")
        parts.append("</table>")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.benchmark.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the --out directory of benchmark.run / benchmark.bare")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    rows = collect(out)
    if not rows:
        raise SystemExit(f"no benchmark records under {out}")
    print(text_table(rows))
    (out / "report.html").write_text(html_report(out, rows))
    print(f"\nwrote {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
