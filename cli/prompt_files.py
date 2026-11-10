"""`--prompt` as text, or as a file of any type.

An argument with no whitespace that names an existing file is a file. A text file (anything with no NUL byte in its
first 8 KB) is read as the prompt, unparsed. Any other file, such as a PDF, an image, or a model, is the brief itself:
it is copied into the run's `inputs/` and the agent's task points at it. The agent opens it with its own tools
(Claude Code reads images and PDFs; Codex views images and can extract a PDF's text in its shell), so nothing is
converted or inlined here. Something that looks like a path but names no file is an error, never a one-word brief.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

INPUTS = "inputs"

#: Extensions that mark a one-word argument as a path, so a missing file is an error rather than the brief "x.pdf".
PATH_LIKE = (".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif",
             ".glb", ".gltf", ".fbx", ".obj", ".stl", ".blend", ".usd", ".usdz", ".mp4", ".mov")

#: How a staged file's type reads in the agent's task.
KINDS = {".pdf": "PDF", ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".gif": "image",
         ".glb": "3D model", ".gltf": "3D model", ".fbx": "3D model", ".obj": "3D model", ".stl": "3D model",
         ".usd": "3D model", ".usdz": "3D model", ".blend": "Blender file", ".mp4": "video", ".mov": "video"}


@dataclass
class PromptFile:
    """A brief that is a file the agent opens itself."""
    path: Path
    kind: str
    size: int

    def staged_name(self) -> str:
        return f"{INPUTS}/{self.path.name}"

    def task(self, where: str, in_run_dir: bool = True) -> str:
        """The task that points at the brief (agent-facing): `where` is its name in the run directory, or, for a
        rewriter working from its own sub-run, its absolute path."""
        return (f"Your brief is the file {where} ({self.kind}, {_size(self.size)})"
                + (", in your run directory" if in_run_dir else "")
                + ". Read it before you start; it is the whole task.")


def _size(n: int) -> str:
    for unit, scale in (("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if n < 10 * scale else f"{n // scale} {unit}"
    return f"{n} bytes"


def read(arg: str) -> tuple[str | None, PromptFile | None, str | None, str | None]:
    """`(text, file, note, error)` for a `--prompt` argument.

    `text` is the prompt when it is text (given, or read from a file); `file` is a brief to stage instead; `note` is
    the launcher's line about what it read; `error` refuses the argument.
    """
    if any(c.isspace() for c in arg):
        return arg, None, None, None
    path = Path(arg).expanduser()
    if path.is_file():
        data = path.read_bytes()
        if b"\0" not in data[:8192]:
            try:
                text = data.decode("utf-8")
                note = f"[cli] loaded the prompt from {path} ({len(text):,} chars)"
            except UnicodeDecodeError:
                text = data.decode("utf-8", "replace")
                note = (f"[cli] loaded the prompt from {path} ({len(text):,} chars); it is not valid UTF-8, so the "
                        "bytes that are not were replaced")
            return text.strip(), None, note, None
        brief = PromptFile(path.resolve(), KINDS.get(path.suffix.lower(), "file"), len(data))
        return None, brief, f"[cli] the brief is {path} ({brief.kind}, {_size(brief.size)}): staged as " \
                            f"{brief.staged_name()} for the agent to open", None
    if "/" in arg or path.suffix.lower() in PATH_LIKE:
        return None, None, None, (f"--prompt {arg}: no such file (relative to {Path.cwd()}). Pass the brief's text "
                                  "in quotes, or the path to an existing file.")
    return arg, None, None, None


def stage(brief: PromptFile, run_dir: Path) -> str:
    """Copy the brief into `run_dir/inputs/`, beside any earlier one of the same name, and return its name there."""
    target = Path(run_dir) / INPUTS / brief.path.name
    n = 2
    while target.exists() and target.read_bytes() != brief.path.read_bytes():
        target = target.with_name(f"{brief.path.stem}-{n}{brief.path.suffix}")
        n += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(brief.path, target)
    return f"{INPUTS}/{target.name}"
