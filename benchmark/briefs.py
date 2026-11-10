"""The benchmark briefs, read from the shipped prompt files, and the task text each arm is given.

A set takes, for each of the 24 briefs, its cut at that horizon if there is one; otherwise, in the
short and medium sets, the DyCraft arm takes the variant that casts its characters from the asset
library (`animation_briefs_imported_characters.md`: B10, B13-B16), and otherwise the brief as written:

    long          B01-B24, the full briefs
    short         B01S-B12S and B15S; B13, B14, and B16 with imported characters; B17-B24 as written
    medium        the ten medium cuts; B05 and B07 as written; B13-B16 with imported characters;
                  B17-B24 as written
    B01S,B13,...  any ids from `agent/prompts/animation_24_briefs.md`

or, with `--prompt`, one brief of your own (`adhoc`), which gets the same wrapper. The bare arm has no
asset library, so it never takes an imported-character variant: a set gives it the cut, else the brief
as written. `--imported-characters` (DyCraft arm) also gives the long set and listed ids the variants.

The text of a brief is never edited here. What each arm receives around it is the form the paper's
runs received:

* **DyCraft arm** — the brief, then a film-length paragraph and a wall-clock budget paragraph, each
  in its own paragraph (`dycraft_prompt`). The CLI's scene and motion rewriters append their own
  pointers to `scene.txt` and `motion.txt` after that.
* **bare arm** — the paper's task wrapper around the brief, then `---`, then the bare guidance
  (`bare_prompt.md`) verbatim (`bare_prompt`). No film-length or budget paragraph: the paper's bare
  runs had none, only a hard stop. G1 briefs swap their one asset-library sentence for the sentence
  that points at the robot description mounted at `/assets/g1`, and a cut that casts its people from
  the library drops that one sentence (its description of the cast is a sentence of its own and
  stays). This reproduces the prompts of the paper's 24 bare runs byte for byte
  (`tests/test_benchmark_briefs.py`); the short and medium cuts get the same wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "agent" / "prompts"
BRIEFS_FILE = PROMPTS / "animation_24_briefs.md"
IMPORTED_FILE = PROMPTS / "animation_briefs_imported_characters.md"
BARE_GUIDANCE = Path(__file__).resolve().parent / "bare_prompt.md"

SETS = ("long", "short", "medium")

#: Film length the DyCraft arm is told to expect, per horizon, in seconds.
FILM_SECONDS = {"long": (60.0, 75.0), "short": (60.0, 75.0), "medium": (120.0, 150.0)}

_SECTION = re.compile(r"^### (B\d\d[SM]?)\.[ \t]*([^\n]*)\n+(.+?)(?=\n#|\Z)", re.M | re.S)
_STAGED_ID = re.compile(r"^\| (B\d\d) \| `([^`]+)` \|", re.M)

#: The sentence every G1 brief uses to name its robot, and the sentence the paper's bare runs were
#: given in its place, with the robot description mounted read-only at `/assets/g1`.
G1_SENTENCE = ("The character is a Unitree G1 humanoid robot with Dex1 grippers; take it from the "
               "asset library rather than creating a new robot.")
G1_URDF = "g1_29dof_mode_15_with_dex1_1.urdf"
BARE_G1_SENTENCE = (f"Use the existing Unitree G1 robot asset at /assets/g1/{G1_URDF} as the character; "
                    "load and reuse its model and articulation rather than creating a new robot.")

#: The sentence a people cut uses to take its cast from the asset library. The bare arm has no library,
#: so its form drops it; no long brief has one.
BARE_CASTING = re.compile(r" Take (?:them|the people|the crew and the actor) from the asset library rather than "
                          r"creating them\.")

BARE_PREFIX = ("Use Blender to create the following animation from scratch. Save the editable .blend "
               "file, reproducible source scripts, and a rendered MP4 in /workspace/output.")
#: The Blender the paper's bare runs had. The environment sentence names the one installed, so on this
#: version the bare prompt is the paper's byte for byte.
PAPER_BLENDER = "4.5.13"
_BARE_ENV = (f"Environment: python has Blender {PAPER_BLENDER} bpy available (run Python scripts with "
             "`python script.py`); ffmpeg is available. {assets} /workspace is your writable working "
             "directory. Work autonomously until the requested deliverables are saved.")
BARE_ENV_NO_ASSETS = _BARE_ENV.format(assets="No starting scene or asset is supplied.")
BARE_ENV_G1 = _BARE_ENV.format(assets="/assets/g1 contains the URDF and its referenced meshes, read-only.")
#: B16's paper prompt said "No starting scene or character asset is supplied."; kept so the 24 paper
#: prompts are reproduced exactly.
BARE_ENV_EXCEPTIONS = {"B16": _BARE_ENV.format(assets="No starting scene or character asset is supplied.")}


@dataclass(frozen=True)
class Brief:
    id: str            # B01, B01S, B01M
    number: int        # 1..24
    title: str
    text: str
    horizon: str       # long | short | medium (from the id's suffix)
    staged_id: str     # e.g. g1_scrambles_eggs; names the run and the film
    g1: bool = False   # an ad-hoc brief given the G1 (`--g1`); the 24 say so in their text

    @property
    def slug(self) -> str:
        return f"{self.id.lower()}_{self.staged_id}"

    @property
    def uses_g1(self) -> bool:
        return self.g1 or G1_SENTENCE in self.text


def _sections(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for m in _SECTION.finditer(path.read_text()):
        body = m.group(3).strip()
        if body.endswith("---"):
            body = body[:-3].rstrip()
        out[m.group(1)] = (m.group(2).strip(), body)
    return out


def load(*, imported_characters: bool = False) -> dict[str, Brief]:
    """Every brief in the shipped file, keyed by id.

    `imported_characters` takes B10 and B13-B16 (long versions only) from
    `animation_briefs_imported_characters.md`, whose one added sentence tells the agent to take the
    characters from the asset library.
    """
    sections = _sections(BRIEFS_FILE)
    staged = dict(_STAGED_ID.findall(BRIEFS_FILE.read_text()))
    if imported_characters:
        for bid, (title, text) in _sections(IMPORTED_FILE).items():
            sections[bid] = (title, text)
    briefs = {}
    for bid, (title, text) in sections.items():
        number = int(bid[1:3])
        horizon = {"S": "short", "M": "medium"}.get(bid[3:], "long")
        base = f"B{number:02d}"
        if base not in staged:
            raise ValueError(f"{BRIEFS_FILE.name}: no staged id for {base} in the appendix table")
        briefs[bid] = Brief(bid, number, title, text, horizon, staged[base])
    return briefs


def select(spec: str, *, imported_characters: bool = False,
           library: bool = False) -> tuple[list[Brief], str | None]:
    """Briefs for a `--briefs` value, and the horizon when `spec` names a set.

    A set takes, for each of the 24 numbers, its cut at that horizon if there is one. Otherwise, for
    an arm with the asset library (`library`, which only the DyCraft arm passes) in the short and medium
    sets, the imported-character variant if there is one; otherwise the brief as written. `library`
    defaults to off, so a caller that forgets it never hands an arm without the library a brief that
    casts from one.
    """
    briefs = load(imported_characters=imported_characters)
    spec = spec.strip()
    if spec in SETS:
        suffix = {"long": "", "short": "S", "medium": "M"}[spec]
        variants = load(imported_characters=True) if library and spec != "long" else briefs
        chosen = []
        for n in range(1, 25):
            base = f"B{n:02d}"
            chosen.append(briefs.get(base + suffix) or variants[base])
        return chosen, spec
    ids = [s.strip().upper() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in briefs]
    if unknown:
        raise SystemExit(f"--briefs: unknown id(s) {', '.join(unknown)}. Known: a set "
                         f"({', '.join(SETS)}) or ids such as B01, B01S, B01M.")
    if len(set(ids)) != len(ids):
        raise SystemExit("--briefs: an id is listed twice")
    return [briefs[i] for i in ids], None


ADHOC_ID = "P"


def adhoc(text: str, *, g1: bool = False) -> Brief:
    """A brief of your own (`--prompt`), wrapped like the benchmark's.

    Its directory is named from its first words. It takes the long briefs' film length when one is
    asked for. With `g1` the bare arm gets the paper's G1 treatment: the robot description mounted
    at `/assets/g1` and the sentence the paper's G1 briefs gave their bare runs.
    """
    text = text.strip()
    if not text:
        raise SystemExit("--prompt is empty")
    staged = "_".join(re.findall(r"[a-z0-9]+", text.lower())[:6]) or "prompt"
    return Brief(ADHOC_ID, 0, text.splitlines()[0][:80], text, "long", staged, g1=g1)


def choose(args, *, imported_characters: bool = False, library: bool = False) -> tuple[list[Brief], str | None]:
    """The briefs a command runs: `--briefs`, or the one `--prompt` (text, or a `.md` file)."""
    prompt, g1 = getattr(args, "prompt", None), getattr(args, "g1", False)
    if bool(args.briefs) == bool(prompt):
        raise SystemExit("give --briefs or --prompt (one of them)")
    if g1 and not prompt:
        raise SystemExit("--g1 is for --prompt; the G1 briefs (B01-B08) already get the G1")
    if prompt:
        path = Path(prompt).expanduser()
        text = path.read_text() if prompt.endswith(".md") and path.is_file() else prompt
        return [adhoc(text, g1=g1)], None
    return select(args.briefs, imported_characters=imported_characters, library=library)


def parse_film_seconds(text: str) -> tuple[float, float] | None:
    """`60-75`, a single `N`, or `none` (no film-length paragraph)."""
    if text.strip().lower() == "none":
        return None
    low, _, high = text.partition("-")
    try:
        bounds = (float(low), float(high or low))
    except ValueError:
        raise SystemExit(f"--film-seconds takes N, LOW-HIGH, or none, not {text!r}") from None
    if bounds[0] <= 0 or bounds[1] < bounds[0]:
        raise SystemExit("--film-seconds must be positive, with LOW <= HIGH")
    return bounds


def length_note(bounds: tuple[float, float]) -> str:
    """The film-length paragraph the paper's dycraft arms carried (a ceiling, not a target)."""
    low, high = bounds
    span = f"{low:g}" if low == high else f"{low:g}–{high:g}"
    return (
        f"Expect a film of no more than about {span} seconds — shorter is "
        "fine, and this is an expectation rather than a requirement. Where "
        "the event does not fit, make the design more efficient — bring "
        "what is handled together, order the work to avoid backtracking, "
        "choose a technique with fewer steps — rather than lengthening the "
        "film or speeding up the motion: every "
        "extra second is paid for again in authoring, rendering and "
        "inspection, out of the same budget as the place and the performance.")


def budget_note(hours: float) -> str:
    """The wall-clock budget paragraph the paper's dycraft arms carried."""
    amount = f"{hours:g} hour" + ("s" if hours != 1 else "")
    return (
        f"Work to a wall-clock budget of {amount} from the start of your "
        "session, covering all authoring, rendering, encoding and validation; "
        "run `date` to read the current time and pace yourself against it. "
        "Finish earlier if the requirements are met. At the deadline, preserve "
        "and report the actual result with the unmet requirements named.")


def dycraft_prompt(brief: Brief, *, film_seconds: tuple[float, float] | None,
                   budget_hours: float | None) -> str:
    parts = [brief.text]
    if film_seconds:
        parts.append(length_note(film_seconds))
    if budget_hours:
        parts.append(budget_note(budget_hours))
    return "\n\n".join(parts)


def bare_brief(brief: Brief) -> str:
    """The brief as the bare arm reads it: a cut's library casting sentence dropped, and a G1 brief's robot
    sentence pointed at the description mounted at `/assets/g1`."""
    text = BARE_CASTING.sub("", brief.text)
    if brief.uses_g1:
        text = (text.replace(G1_SENTENCE, BARE_G1_SENTENCE) if G1_SENTENCE in text
                else f"{text} {BARE_G1_SENTENCE}")
    return text


def bare_task(brief: Brief, *, film_seconds: tuple[float, float] | None = None,
              budget_hours: float | None = None, blender: str = PAPER_BLENDER) -> str:
    """The paper's `task_prompt.md` for this brief, naming `blender` as the installed version.

    With `film_seconds` / `budget_hours` (opt-in; the paper's bare runs had neither), the DyCraft
    arm's two paragraphs follow the brief verbatim, so both arms can be told the same length and
    budget. The guidance already says to use a budget "if supplied with the execution".
    """
    text, env = bare_brief(brief), BARE_ENV_EXCEPTIONS.get(brief.id, BARE_ENV_NO_ASSETS)
    if brief.uses_g1:
        env = BARE_ENV_G1
    parts = [BARE_PREFIX, text]
    if film_seconds:
        parts.append(length_note(film_seconds))
    if budget_hours:
        parts.append(budget_note(budget_hours))
    env = env.replace(f"Blender {PAPER_BLENDER} bpy", f"Blender {blender} bpy")
    return "\n\n".join(parts + [env]) + "\n"


def bare_prompt(brief: Brief, *, film_seconds: tuple[float, float] | None = None,
                budget_hours: float | None = None, blender: str = PAPER_BLENDER) -> str:
    """What the bare agent reads on stdin: the task, `---`, then the guidance verbatim. With no
    paragraphs, on the paper's Blender, this is byte-for-byte what the paper's bare runs read."""
    return (bare_task(brief, film_seconds=film_seconds, budget_hours=budget_hours,
                      blender=blender).rstrip()
            + "\n\n---\n\n" + BARE_GUIDANCE.read_text())
