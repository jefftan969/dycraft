"""`review`: the agent asks a fresh context whether it built what it promised.

`run_review` resolves the targets into per-model specs, writes the reviewer's
brief, runs the reviewer as a one-off sub-agent on the run's own backend
(`review_backend`, `run_model`, `run_reasoning_effort`), and records its
verdicts in `review/latest.json`.
"""

from __future__ import annotations

import json
import re
import sys

from .tools import (
    ToolContext,
    build_current_ir,
)


# --------------------------------------------------------------------------
# review - the agent asks a fresh context whether it built what it promised
# --------------------------------------------------------------------------

#: Wall-clock bound on the reviewer sub-run. A nested CLI agent that renders
#: its own extra cameras needs minutes, not the 120 s `run_oneoff` defaults
#: to. Past this the sub-CLI is SIGTERMed and `review` reports whatever it
#: managed to say.
REVIEW_TIMEOUT_S = 600.0

#: Extra cameras the reviewer may render for itself, enforced inside
#: `_review_render.py` by counting what is on disk rather than by asking
#: politely in the prompt. An extra render is a targeted second look at what
#: the shipped evidence did not settle, not a fishing expedition.
REVIEW_MAX_RENDERS = 4

REVIEW_DIR = "review"
REVIEW_VERDICTS = ("met", "not_met", "cant_tell")


#: How many models one review may judge at once. The cap is about the
#: reviewer's attention, not the harness: every target contributes six
#: canonical views plus its articulation cards, so six targets is already
#: forty-odd images to hold in one head while deciding a claim.
REVIEW_MAX_TARGETS = 6


def _review_targets(ctx, targets) -> tuple:
    """Resolve `targets` into per-model review specs, or `(None, error)`.

    A target names a **definition**, never an instance. One definition placed
    three times is the same geometry three times over, so reviewing it once is
    the whole answer — and anything that differs between those placements is a
    placement question, which is a scene claim by construction.
    """
    layer = ctx.layer_cfg.name
    build = build_current_ir(ctx)
    if not build["ok"] or not isinstance(build.get("ir"), dict):
        return None, ("could not build the IR from the current source - "
                      "compile() first and fix what it reports")
    ir = build["ir"]

    if layer != "scene":
        # The object layer has exactly one model; `targets` has nothing to
        # select between.
        if not ir.get("parts"):
            return None, ("could not build the IR from the current source - "
                          "compile() first and fix what it reports")
        checks = list(ir.get("checks") or ())
        if not checks:
            return None, ("this object declares no checks. Add "
                          "obj.add_check(...) for the claims only looking can "
                          "settle, compile, then review.")
        # `submission`: this spec IS what the run was asked to produce, so the
        # user's request is a question about it. An object run's brief
        # describes its one object; a definition inside a scene run's does not.
        # The two specs are otherwise identical — same `kind`, same `layer` —
        # so without this flag `asks_prompt_match` cannot tell them apart.
        return [{"key": "object", "kind": "object", "noun": "object",
                 "entity": "part", "layer": "object", "ir": ir,
                 "checks": checks, "catalog": None, "submission": True,
                 "label": ir.get("name") or "object"}], None

    catalog = getattr(ctx.layer_payload, "catalog", None) or {}
    name_to_hash = getattr(ctx.layer_payload, "name_to_hash", None) or {}
    if isinstance(targets, str):
        targets = [targets]
    if targets is None:
        targets = ["scene"]
    if not isinstance(targets, (list, tuple)) or not targets:
        return None, "targets must be 'scene', a definition name, or a list"
    if len(targets) > REVIEW_MAX_TARGETS:
        return None, (f"at most {REVIEW_MAX_TARGETS} targets per review "
                      f"({len(targets)} requested). Review the scene plus the "
                      f"definitions whose claims a scene view cannot settle.")

    specs = []
    seen = set()
    for name in targets:
        name = str(name)
        if name in seen:
            continue
        seen.add(name)
        if name in ("scene", "all"):
            if not ir.get("instances"):
                return None, ("the scene places nothing yet - compile() a "
                              "scene with instances first")
            checks = list(ir.get("checks") or ())
            if not checks:
                return None, ("this scene declares no checks. Add "
                              "scene.add_check(...) for the claims only "
                              "looking can settle, compile, then review.")
            specs.append({"key": "scene", "kind": "scene", "noun": "scene",
                          "entity": "instance", "layer": "scene", "ir": ir,
                          "checks": checks, "catalog": catalog,
                          "submission": True,
                          "label": ir.get("name") or "scene"})
            continue
        h = name_to_hash.get(name)
        if h is None or h not in catalog:
            return None, (f"unknown review target {name!r}. Targets are "
                          f"'scene' or a definition name: "
                          f"{sorted(name_to_hash)}")
        obj_ir = catalog[h]
        checks = list(obj_ir.get("checks") or ())
        if not checks:
            return None, (f"definition {name!r} declares no checks. Add "
                          f"obj.add_check(...) inside the function that builds "
                          f"it, compile, then review.")
        specs.append({"key": name, "kind": "object", "noun": "object",
                      "entity": "part", "layer": "object", "ir": obj_ir,
                      "checks": checks, "catalog": None, "label": name})
    return specs, None


def _review_prompt(specs: list, render_cmd=None, user_prompt: str = "") -> str:
    """The reviewer's whole brief, written by the harness.

    `review` takes no claim text from the agent on purpose. The claims are
    already frozen into the content-addressed IR by `add_check`, and if the
    author could also write the reviewer's prompt it could drop a claim or
    describe what to look for - which puts it straight back in the position of
    grading itself against requirements it controls. What the agent chooses is
    only *which models* to have looked at.

    Deliberately absent: the source, the assertions, the deviations, the
    verifier reports, and any description of what a correct or an incorrect
    build looks like. A reviewer told what to expect sees it, so each claim
    carries the relation and stops there.

    The original request is the exception, and it has a cost. A reviewer
    holding the request tends to re-derive requirements instead of checking
    the claims in front of it. It is here because the claims can only cover
    what the author thought to promise: a requirement the plan never noticed is
    invisible to a claims-only review, and nothing else looks for it. The
    mitigation is ordering and an explicit instruction. Claims are decided
    from the images alone, and the request is asked separately, afterwards, as
    its own verdict.

    **`prompt_match` is asked of the scene and of nothing else.** On a
    multi-target review the request describes the scene; asking whether one
    espresso machine is a reasonable answer to "a 12-room restaurant" is a
    question with no sensible verdict.
    """
    multi = len(specs) > 1
    lines = [
        "Review rendered 3D geometry against the author's registered claims. Decide "
        "whether each claim is visible in the geometry.",
        "",
        "First judge each claim from its images alone, independently of overall "
        "quality or request coverage. Assess request coverage separately when asked "
        "below.",
    ]
    if multi:
        lines += [
            "",
            f"There are {len(specs)} separate models below. Judge each one "
            "only against its own renders: they are different models, not "
            "different views of one. Claim ids are prefixed with the model "
            "they belong to.",
        ]
    for spec in specs:
        head = (f"MODEL {spec['key']} - the {spec['noun']} "
                f"{spec['label']!r}" if multi else "The renders:")
        lines += ["", head, ""]
        lines += [f"  {p}" for p in spec["images"]]
        lines += ["", "Claims:", ""]
        lines += [f"  {spec['key']}:{c['id']}: {c['claim']}"
                  for c in spec["checks"]]
        if spec.get("render_cmd"):
            # Document the whole surface, not just the camera. A reviewer told
            # only about azimuth and elevation answers `cant_tell` for every
            # claim about a surface the model covers, because moving a camera
            # cannot uncover it, while hiding the occluder (`snapshot(only=…)`)
            # does.
            lines += [
                "",
                "If a different view would settle one of these, render one:",
                "",
                f"  {spec['render_cmd']} --on <{spec['entity']}|all> "
                f"--azimuth <deg> --elevation <deg>",
                "",
                "and, when the camera alone is not enough:",
                "",
                f"  --only <{spec['entity']}> [...]   draw only these and "
                f"hide the rest, to see an interior",
                "                              surface, a contact face, or "
                "anything the model",
                "                              covers. No angle can uncover "
                "it; this can.",
                "  --poses \'{\"<joint>\": <deg|m>}\'  render at a joint "
                "position the panels do not",
                "                              show. Degrees for a revolute "
                "joint, meters",
                "                              for a prismatic one.",
                "  --resolution <W> <H>        for a detail too small to read.",
                "  --fov-deg <deg>             a narrow angle flattens the "
                "perspective that",
                "                              makes containment ambiguous.",
            ]
    lines += [
        "",
        "Read every image listed. Each panel carries its label in its own "
        "top-left corner; cite those labels.",
        "",
        "For each claim answer exactly one of:",
        "  met       - the geometry shows it.",
        "  not_met   - the geometry shows the opposite, or plainly lacks it.",
        "  cant_tell - no image you have settles it.",
        "",
        "Before `met`, check for contradictory evidence and cite it in your reason if "
        "found. Use `cant_tell` for unseen claims; plausibility or care elsewhere in "
        "the model is not evidence.",
        "",
        f"Azimuth 0 = +X, 90 = +Y (the front); elevation 90 = straight down. "
        f"A render command prints a PNG path for you to read. At most "
        f"{REVIEW_MAX_RENDERS} extra views for the whole review, and only "
        f"for a claim you cannot otherwise settle. Spend them on the claims a "
        f"camera move cannot settle: prefer `--only` for anything the model "
        f"covers over another angle on the outside of it. `cant_tell` is the "
        f"right answer when no view you can take settles a claim — not when "
        f"you have not taken the view that would.",
        "",
    ]
    scene_spec = next((s for s in specs if s["kind"] == "scene"), None)
    ask_prompt_match = asks_prompt_match(specs, user_prompt)
    if ask_prompt_match:
        subject = scene_spec["noun"] if scene_spec else specs[0]["noun"]
        lines += [
            "",
            "Separately, and only after you have decided every claim above "
            f"from the images: this is the request the {subject} was built "
            "for.",
            "",
            f"  {user_prompt}",
            "",
            "Answer `prompt_match` for it - met, not_met, or cant_tell - "
            f"saying whether the {subject} you can see is a reasonable answer "
            f"to that request. Judge scope and identity, not polish: a missing "
            f"element or the wrong kind of {subject} is not_met, a rough one "
            f"is not. Do NOT revisit any claim above in light of this; the "
            f"claims are decided from the images alone.",
        ]
    lines += [
        "",
        "Then call submit_result once, with JSON and nothing else:",
        "",
        '  {"results": [{"id": "<model>:c01", "verdict": "met", "view": '
        '"<panel label or PNG you judged from>", "reason": "<one sentence>"}, '
        '...]'
        + (', "prompt_match": {"verdict": "met", "reason": "<one sentence>"}}'
           if ask_prompt_match else "}"),
        "",
        "One entry per claim id, all of them, using the prefixed ids exactly "
        "as written above.",
    ]
    return "\n".join(lines)


def asks_prompt_match(specs, user_prompt: str) -> bool:
    """Whether this review puts the user's request to the reviewer at all.

    **The parser has to ask the same question the prompt asked.** `run_review`
    decides whether to PARSE a `prompt_match` and `_review_prompt` whether to
    ASK, both by this rule; deciding them separately would record a
    `cant_tell` for a question nobody was asked.

    **Asked of the run's submission, not of whatever happens to be alone.**
    "Exactly one target" is not the test: a scene run reviewing a single
    *definition* would have it judged against the brief, and a backrest alone
    is rightly `not_met` for a brief its committed scene answers.

    `submission` is set where the spec is built, by the only code that knows:
    an object run's object and a scene run's scene carry it, a definition
    inside a scene does not.
    """
    if not user_prompt:
        return False
    return any(s["kind"] == "scene" or s.get("submission") for s in specs)


def _parse_prompt_match(text: str, *, silence_reason: str) -> dict:
    """The `prompt_match` object, or `cant_tell` when it is absent or malformed.

    Same policy as the per-claim parser: silence is not agreement, and
    `silence_reason` says which kind of silence — see `_parse_review`.
    """
    out = {"verdict": "cant_tell", "reason": silence_reason}
    blob = _first_json_with(text, "results")
    pm = (blob or {}).get("prompt_match")
    if isinstance(pm, dict) and str(pm.get("verdict")) in REVIEW_VERDICTS:
        out = {"verdict": str(pm["verdict"]),
               "reason": str(pm.get("reason") or "")}
    return out


#: Why a claim has no verdict, per `run_oneoff` outcome. A reviewer that died
#: and a reviewer that answered everything except claim 12 are both `cant_tell`
#: and are not the same event: the first says nothing about the geometry and
#: the whole review should be re-run, the second is a real judgment that the
#: images cannot settle this one claim. One shared string would let a dead
#: sub-agent wear the same label as an honest abstention, so the reason comes
#: from `outcome`.
_SILENCE_REASONS = {
    "timeout": "reviewer timed out; no verdict available",
    "no_result": "reviewer exited without submitting a result; no verdict available",
    "spawn_error": "reviewer could not be launched; no verdict available",
}
_ANSWERED_BUT_SKIPPED = "no verdict returned"


#: Outcomes where the reviewer *did* answer, so an unparsed claim means the
#: reviewer skipped it rather than that no verdict exists. Without
#: `completed_after_timeout` here, a review recovered from a killed shell
#: would have every claim labeled "reviewer timed out; no verdict available"
#: — the exact opposite of what happened.
_ANSWERED_OUTCOMES = ("completed", "completed_after_timeout")


def _silence_reason(outcome) -> str:
    if str(outcome or "") in _ANSWERED_OUTCOMES:
        return _ANSWERED_BUT_SKIPPED
    return _SILENCE_REASONS.get(str(outcome or ""), _ANSWERED_BUT_SKIPPED)


def _first_json_with(text: str, key: str) -> "dict | None":
    """The first JSON object in `text` carrying `key`, or None.

    Dug out rather than parsed strictly: a CLI sub-agent writing a sentence
    around its JSON is the normal case, not an error.
    """
    for start, ch in enumerate(text or ""):
        if ch != "{":
            continue
        for end in range(len(text), start, -1):
            try:
                cand = json.loads(text[start:end])
            except Exception:
                continue
            if isinstance(cand, dict) and key in cand:
                return cand
    return None


def _parse_review(text: str, check_ids: tuple, *, silence_reason: str) -> dict:
    """`{id: {verdict, view, reason}}`, defaulting anything absent to cant_tell.

    A reviewer that skipped a claim has not judged it, and reading silence as
    `met` is the one failure this op exists to avoid. The JSON is dug out of
    the reply rather than parsed strictly: a CLI sub-agent writing a sentence
    around its JSON is the normal case, not an error.

    `silence_reason` distinguishes the KINDS of silence — see
    `_SILENCE_REASONS`. An abstention and a dead sub-agent are both
    `cant_tell`, and a reader who cannot tell them apart will treat a failed
    review as evidence about the geometry.
    """
    out = {cid: {"verdict": "cant_tell", "view": "",
                 "reason": silence_reason} for cid in check_ids}
    blob = _first_json_with(text, "results")
    for item in ((blob or {}).get("results") or []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid not in out or str(item.get("verdict")) not in REVIEW_VERDICTS:
            continue
        out[cid] = {"verdict": str(item["verdict"]),
                    "view": str(item.get("view") or ""),
                    "reason": str(item.get("reason") or "")}
    return out


def _review_verdicts(reply: str, outcome, check_ids: tuple) -> tuple:
    """`(verdicts, silence_reason, claims_evaluated, silent)` for one review.

    `silent` is a reviewer that answered no claim at all, and it is a failed
    review whatever its outcome says. Reported as `ok: True` with every claim
    "no verdict returned", it reads to the agent as a review, and a model can
    ship unreviewed. `_ANSWERED_OUTCOMES`' per-claim label is right when a
    recovered reviewer skipped *some* claims; when it answered none after a
    timeout, the timeout is the truth, so that is the label.
    """
    silence = _silence_reason(outcome)
    verdicts = _parse_review(reply, check_ids, silence_reason=silence)
    # `_parse_review` fills every claim it did not find with exactly this;
    # anything else came from the reply.
    unanswered = {"verdict": "cant_tell", "view": "", "reason": silence}
    evaluated = sum(1 for verdict in verdicts.values() if verdict != unanswered)
    silent = bool(check_ids) and evaluated == 0
    if silent and outcome == "completed_after_timeout":
        silence = _SILENCE_REASONS["timeout"]
        verdicts = {cid: {**verdict, "reason": silence} for cid, verdict in verdicts.items()}
    return verdicts, silence, evaluated, silent


def _run_meta(ctx: ToolContext) -> dict:
    """This run's `meta.json`, or an empty dict if it cannot be read."""
    try:
        return json.loads((ctx.run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


#: A backend selector: `claude`, `codex`, or one of their numbered account slots.
_BACKEND_SELECTOR_RE = re.compile(r"(claude|codex)\d?\Z")


def _backend_from_runtime(runtime) -> str:
    """The backend selector `meta.json`'s `runtime` names, or `""`.

    `write_run_meta` records the launcher's backend as `runtime` (`"codex1"`;
    a `_cli` suffix, which recorded runs may carry, is accepted), and the
    reviewer and dispatched children must run on that backend and account, not
    a default. Anything that is not a CLI backend (`bare_agent`,
    `hand_authored`) names none.
    """
    name = str(runtime or "")
    name = name.removesuffix("_cli")
    return name if _BACKEND_SELECTOR_RE.match(name) else ""


def review_backend(ctx: ToolContext) -> str:
    """Which CLI backend serves the reviewer, and any dispatched child.

    This run's own backend, else `claude`. `run_oneoff` launches an external
    CLI: a reviewer that cannot shell out cannot render for itself.

    `backend` is read first and `runtime` second because a run bundle may carry
    either: nothing writes `backend` today, but a reader that only understood
    `runtime` would break the moment something did.
    """
    meta = _run_meta(ctx)
    backend = str(meta.get("backend") or "")
    if backend and backend != "api":
        return backend
    return _backend_from_runtime(meta.get("runtime")) or "claude"


def run_model(ctx: ToolContext) -> str:
    """The model this run's agent is using, for a sub-CLI to inherit.

    A CLI backend does not require `--model` — it falls back to its own default
    — so an unforwarded model is not an error, it is a child quietly built by a
    different model than the run it belongs to.
    """
    return str(_run_meta(ctx).get("model") or "")


def run_reasoning_effort(ctx: ToolContext) -> str:
    """The reasoning effort this run's agent is using, for a sub-CLI.

    From `meta.json` rather than the environment: the launcher records the
    resolved value there but exports no `DYCRAFT_REASONING_EFFORT`, so the
    daemon has no other way to know what it was launched at.
    """
    return str(_run_meta(ctx).get("reasoning_effort") or "")


def run_review(ctx: ToolContext, targets=None) -> dict:
    """Ask a fresh-context sub-agent whether the build matches the plan.

    Renders the canonical evidence, launches one CLI sub-agent
    through the ordinary `run_oneoff` path, and returns its per-claim
    verdicts. The sub-agent starts from nothing but the claims and the
    images; `Read` and `Bash` come from its own backend, which is what lets
    it render cameras of its own instead of asking someone to render for it.

    **Never blocks.** The result is information for the next iteration. An
    agent that never calls `review` finishes exactly as it would have - it
    has `snapshot` and can look for itself.
    """
    from ..oneoff import SUB_RUNS_DIR, run_oneoff
    from ...verify.review_images import render_review_images

    resolved, err = _review_targets(ctx, targets)
    if err is not None:
        return {"ok": False, "error": err}

    out_dir = ctx.run_dir / REVIEW_DIR / f"step_{ctx.step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One reviewer for every target, because most of what a review costs is
    # starting one; a scene with five definitions worth looking at would spend
    # most of its review time on process startup if each got its own. The
    # claims stay separated by target inside the brief so they are still judged
    # one model at a time.
    for spec in resolved:
        # `evidence_key=` keeps these renders apart. Every target here shares
        # one `ctx`, so without it each one would write the same
        # `renders/<step>_<camera>.png` and `renders/<step>_vlm_views/` as the
        # last and record those shared paths as its own, and the reviewer, told
        # they are different models, would grade them all from one model's
        # renders. `spec["key"]` namespaces this target's `_ir.json` and
        # `_catalog/` the same way a few lines below.
        images, _art, _layout, unavail = render_review_images(
            spec["ir"], ctx, spec["layer"], evidence_key=spec["key"])
        if unavail is not None:
            return {"ok": False, "error": f"rendering unavailable: {unavail}"}
        spec["images"] = [str(p) for p in images.paths()]

        spec["ir_path"] = out_dir / f"{spec['key']}_ir.json"
        spec["ir_path"].write_text(json.dumps(spec["ir"], indent=1,
                                              sort_keys=True))
        # The reviewer runs `_review_render` in its own process, so a scene's
        # catalog has to reach it as files. Written beside the IR rather than
        # pointed at the run's own `catalog/`, which `refresh_snapshot`
        # rewrites on every compile: the reviewer must judge the revision it
        # was shown, not whatever the author moved on to while it was looking.
        spec["catalog_dir"] = None
        if spec["catalog"]:
            cdir = out_dir / f"{spec['key']}_catalog"
            cdir.mkdir(parents=True, exist_ok=True)
            for h, obj_ir in sorted(spec["catalog"].items()):
                (cdir / f"{h}.json").write_text(
                    json.dumps(obj_ir, indent=1, sort_keys=True))
            spec["catalog_dir"] = cdir

    # The reviewer renders into its OWN sub-run directory, not into this one.
    # Its sandbox permits writes only under its run root (`filesystem.
    # allowWrite`), so a `--out-dir` pointing here would need it to escape the
    # sandbox, which it cannot do headless. A reviewer left with the canonical
    # views alone can misread perspective foreshortening as taper.
    # Reads are unrestricted, so `--ir` and `--asset-base` may still point at
    # the parent run.
    sub_run_dir = out_dir / SUB_RUNS_DIR / "review"
    extra_dir = sub_run_dir / "extra"
    asset_base = (getattr(ctx.layer_payload, "asset_base_dir", None)
                  or ctx.run_dir)
    render_cmd = None
    for spec in resolved:
        spec["render_cmd"] = (
            f"{sys.executable} -m dycraft.agent.layers._review_render"
            f" --ir {spec['ir_path']} --out-dir {extra_dir}"
            f" --asset-base {asset_base} --cap {REVIEW_MAX_RENDERS}"
            + (f" --catalog {spec['catalog_dir']}"
               if spec["catalog_dir"] is not None else ""))
    # Qualified, because both namespaces start at `c01`: a scene's first claim
    # and an espresso machine's first claim are both `c01`, and a reviewer
    # answering `c01` for a two-target review would be answering neither.
    check_ids = tuple(f"{spec['key']}:{c['id']}"
                      for spec in resolved for c in spec["checks"])
    # The user's request, for `prompt_match`: `cli.py` exports it, and the
    # daemon reads it the same way (`layers/server.py`).
    from ..agent_runner import user_prompt_from_env
    user_prompt = user_prompt_from_env()
    prompt = _review_prompt(resolved, render_cmd, user_prompt=user_prompt)
    (out_dir / "prompt.txt").write_text(prompt)

    # Interactive mode, like every claude this framework starts: headless, it
    # still routes banners to stderr, so stdout is the bare answer and
    # `run_oneoff`'s fallback recovers it when `submit_result` cannot run (as
    # on a machine where bwrap cannot create a user namespace). Interactive mode
    # with stdin at EOF gives the same stdout as `--print`.
    # Same model and same accounting backend as the parent. The reviewer
    # is a sub-agent like a dispatched child, and an unforwarded model puts
    # the judgment in a different model's hands than the run it is judging,
    # and the model is the variable a comparison between runs isolates.
    res = run_oneoff(prompt=prompt, backend=review_backend(ctx),
                     model=run_model(ctx) or None,
                     reasoning_effort=run_reasoning_effort(ctx) or None,
                     run_dir=out_dir, sub_run_name="review",
                     timeout_s=REVIEW_TIMEOUT_S)
    reply = res.get("text") or ""
    verdicts, silence, evaluated, silent = _review_verdicts(reply, res.get("outcome"),
                                                            check_ids)
    # `asks_prompt_match`, not `bool(user_prompt)`: parse for an answer only
    # where the prompt actually asked for one, or the result fabricates a
    # non-answer to a question nobody posed. `None` means "not asked".
    prompt_match = (_parse_prompt_match(reply, silence_reason=silence)
                    if asks_prompt_match(resolved, user_prompt) else None)
    tally = {v: 0 for v in REVIEW_VERDICTS}
    for v in verdicts.values():
        tally[v["verdict"]] += 1
    payload = {
        "ok": not silent,
        "reviewed": bool(res.get("ok")) and not silent,
        **({"status": "no_verdicts",
            "error": (f"the reviewer returned no parseable verdict for any of "
                      f"{len(check_ids)} claims (outcome "
                      f"{res.get('outcome')!r}); the review did not happen. "
                      f"Re-issue it with the same targets: "
                      f"{[spec['key'] for spec in resolved]}")}
           if silent else {}),
        # A review that answered 6 of 13 is a different event from one that
        # answered 13.
        "claims_requested": len(check_ids),
        "claims_evaluated": evaluated,
        "results": [{"id": cid, **verdicts[cid]} for cid in check_ids],
        # Grouped as well as flat: a five-target review returns forty verdicts,
        # and the question an author asks next is "how did the espresso machine
        # do", not "what was claim 23".
        "by_target": {
            spec["key"]: {
                "kind": spec["kind"],
                "label": spec["label"],
                "images": spec["images"],
                "results": [{"id": f"{spec['key']}:{c['id']}",
                             "claim": c["claim"],
                             **verdicts[f"{spec['key']}:{c['id']}"]}
                            for c in spec["checks"]],
            }
            for spec in resolved
        },
        "targets": [spec["key"] for spec in resolved],
        "prompt_match": prompt_match,
        "tally": tally,
        "extra_renders": sorted(str(p) for p in extra_dir.glob("extra_*.png")),
        "images": [p for spec in resolved for p in spec["images"]],
        "sub_run": res.get("sub_run"),
        "outcome": res.get("outcome"),
        "wall_time_s": res.get("wall_time_s"),
        # The raw reply. When the parse finds nothing, every claim comes back
        # `cant_tell` with "no verdict returned", and this is the only place
        # outside the transcript where the reviewer's answer can be read.
        "reply_text": (res.get("text") or "")[:4000],
        "n_op_calls": res.get("n_op_calls"),
        "stderr_tail": (res.get("stderr_tail") or "")[-600:],
        "note": ("Advisory - nothing here blocks finish(). A `cant_tell` "
                 "that survived the reviewer having its own camera usually "
                 "means the claim is not visually checkable."),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    # A second copy at a fixed path, mirroring `compile_reports/latest.json`
    # (`fixed_source.py:190`) for the same reason and with the same freshness
    # token. The harness client is stateless RPC: when the agent's shell is
    # killed at its timeout the daemon still finishes the review and writes
    # the result, but the reply is lost with the client. The prompt already
    # tells the agent how to recover a *compile* that way, and this is the
    # review's equivalent of a path it would otherwise have to find itself.
    #
    # `ir_hash` is what makes the file usable rather than a guess — the same
    # caveat the compile guidance carries ("an old report is not its verdict").
    latest = dict(payload)
    latest["step"] = ctx.step
    latest["ir_hash"] = _ir_hash_for_latest(ctx)
    (ctx.run_dir / REVIEW_DIR / "latest.json").write_text(
        json.dumps(latest, indent=2))
    return payload


def _ir_hash_for_latest(ctx) -> str | None:
    """This run's current IR id, for `review/latest.json`'s freshness token.

    Best-effort and never raises: the token exists so a recovering agent can
    tell its own verdict from a stale one, and a missing token is better than
    a review that fails because it could not read `ir.json`.
    """
    try:
        import json as _json
        return (_json.loads((ctx.run_dir / "ir.json").read_text())
                or {}).get("id")
    except Exception:                                          # noqa: BLE001
        return None


__all__ = ["run_review", "review_backend", "run_model", "run_reasoning_effort"]
