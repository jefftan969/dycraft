"""Composite renders into one labeled contact sheet for a visual review."""

from __future__ import annotations

import math
from pathlib import Path


#: Sheets are intermediate artifacts like the renders they are composited from,
#: so they get the same zlib level (`render.open3d_renderer._PNG_COMPRESS_LEVEL`).
_PNG_COMPRESS_LEVEL = 1


#: Longest side of a composited view sheet, in pixels.
#:
#: Models that read images downscale larger ones, so pixels past this are
#: rendered for nothing. The cap is also why the COLUMN COUNT matters more than it
#: looks: six 16:9 panels in a 3x2 grid cap out at 682x384 each, the same six
#: in a 2x3 grid at 1024x576 — 2.25x the panel area, because the limit applies
#: to the longest side and a tall sheet spends fewer of its pixels on width.
_SHEET_MAX_SIDE = 2048

#: Columns in a view sheet. Two, per the note above.
_SHEET_COLUMNS = 2


def build_view_sheet(
    panels: list,
    out_path: Path,
    *,
    columns: int = _SHEET_COLUMNS,
    max_side: int = _SHEET_MAX_SIDE,
) -> Path:
    """Composite `(label, path)` renders into one labeled contact sheet.

    Why one image instead of N: the reviewer is a nested agent that `Read`s
    every image it is given, so N images is N tool-use turns *and* N calls on
    its attention. The six canonical default-pose views cover about nine in ten
    defects a reviewer cites by view, and an articulation sheet most of the
    rest.

    Why it does not cost resolution: the canonical cameras fit the target
    exactly (`render.canonical_cameras`), so the median object fills about two
    fifths of the frame, and a 1024x576 panel carries more object pixels than
    a full 1280x720 view of a loosely framed object.

    Labels are drawn INTO the image, not just listed in the prompt legend. The
    reviewer cites the evidence it used and a legend cannot tell it which quadrant
    it is looking at; a burnt-in label can. Drawn with a white stroke so it
    stays legible over both the light backdrop and a dark part.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not panels:
        raise ValueError("a view sheet needs at least one panel")
    columns = max(1, int(columns))
    rows = math.ceil(len(panels) / columns)

    with Image.open(Path(panels[0][1])) as probe:
        aspect = probe.width / probe.height
    # Fit `columns x rows` panels of that aspect inside `max_side`.
    if columns * aspect >= rows:
        sheet_w = max_side
        sheet_h = max(1, round(max_side * rows / (columns * aspect)))
    else:
        sheet_h = max_side
        sheet_w = max(1, round(max_side * columns * aspect / rows))
    panel_w, panel_h = sheet_w // columns, sheet_h // rows

    canvas = Image.new("RGB", (panel_w * columns, panel_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, panel_h // 26))
    except OSError:
        font = ImageFont.load_default()

    for index, (label, path) in enumerate(panels):
        column, row = index % columns, index // columns
        x, y = column * panel_w, row * panel_h
        with Image.open(Path(path)) as image:
            canvas.paste(image.convert("RGB").resize((panel_w, panel_h)),
                         (x, y))
        draw.text((x + 8, y + 6), str(label), fill="black",
                  stroke_width=3, stroke_fill="white", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    return out_path


__all__ = ["build_view_sheet"]
