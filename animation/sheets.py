"""Contact and overview sheets: rendered frames laid out as labeled grids."""
from __future__ import annotations

from pathlib import Path


#: Cell geometry for `contact_sheet`. The floor is the point of the whole
#: function: at thumbnail sizes a foot slide of centimeters, a gripper closing
#: on nothing or a small subject all fail to read. Below the floor the sheet
#: paginates instead of shrinking further, because a sheet you cannot read is
#: worse than two sheets.
_SHEET_CELL_WIDTH = 480
_SHEET_COLUMNS = 4
_SHEET_MAX_EDGE = 2048
#: A caption strip under each cell, the gutter around it, and the header strip
#: a labeled sheet carries.
_SHEET_BAR, _SHEET_GAP, _SHEET_HEAD = 22, 4, 26


def contact_sheet(frames, destination, *, fps=None, label=None):
    """Lay rendered frames out as labeled grids; return the sheets written.

    Every visual op on this surface returns individual frames, and nothing
    returns a grid -- so judging pacing, continuity, or a whole shot means
    looking at frames together, which authors otherwise build for themselves.

    Built here rather than in the worker because Blender's bundled Python has
    no PIL while the daemon's does. The frames are already on
    disk by the time the worker returns, so this costs one pass over them.

    Advisory by construction: any failure returns what it managed and never
    propagates. A sheet is a convenience on top of a render that has already
    succeeded, and losing the render because the sheet could not be drawn would
    be a strictly worse trade than not having sheets.

    `frames` is an ordered sequence of `(output_frame_index, png_path)`.
    `label`, when given, is drawn in a strip across the top of every page.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return []
    frames = [(index, Path(path)) for index, path in frames if Path(path).is_file()]
    if len(frames) < 2:
        return []
    try:
        with Image.open(frames[0][1]) as first:
            aspect = first.height / first.width if first.width else 9 / 16
        cell = (_SHEET_CELL_WIDTH, max(1, round(_SHEET_CELL_WIDTH * aspect)))
        # A caption strip under each cell and a gutter around it. Without the
        # gutter adjacent frames run edge to edge and the grid reads as one
        # image, which defeats the point of looking at frames side by side.
        bar, gap = _SHEET_BAR, _SHEET_GAP
        head = _SHEET_HEAD if label else 0
        columns = min(_SHEET_COLUMNS, len(frames))
        rows = max(1, (_SHEET_MAX_EDGE - head) // (cell[1] + bar + gap))
        per_sheet = columns * rows
        try:
            font = ImageFont.load_default(size=15)
        except TypeError:          # Pillow < 10 takes no size
            font = ImageFont.load_default()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for page, offset in enumerate(range(0, len(frames), per_sheet)):
            chunk = frames[offset:offset + per_sheet]
            used_rows = -(-len(chunk) // columns)
            sheet = Image.new("RGB", (columns * (cell[0] + gap) + gap,
                                      head + used_rows * (cell[1] + bar + gap) + gap),
                              (12, 13, 16))
            draw = ImageDraw.Draw(sheet)
            if label:
                draw.text((gap + 6, 5), label, fill=(225, 228, 232), font=font)
            for position, (index, path) in enumerate(chunk):
                x = gap + (position % columns) * (cell[0] + gap)
                y = head + gap + (position // columns) * (cell[1] + bar + gap)
                try:
                    with Image.open(path) as image:
                        sheet.paste(image.convert("RGB").resize(cell, Image.BILINEAR), (x, y))
                except OSError:
                    draw.rectangle([x, y, x + cell[0], y + cell[1]], fill=(60, 30, 30))
                # Index *and* timecode. The labeling is what makes a sheet
                # answer a pacing question rather than merely show pictures:
                # "does this beat land too early" needs the clock, and "which
                # frame do I go back and snapshot" needs the index.
                caption = f"{index}"
                if fps:
                    caption += f"   {index / fps:.2f}s"
                draw.text((x + 6, y + cell[1] + 4), caption, fill=(225, 228, 232), font=font)
            path = (destination if page == 0
                    else destination.with_name(f"{destination.stem}_{page + 1}{destination.suffix}"))
            sheet.save(path, quality=88)
            written.append(str(path))
        return written
    except Exception:      # noqa: BLE001 -- advisory; never fail the render for a sheet
        return []


def overview_sheet(frames, destination, *, fps=None):
    """One page sampling a long render evenly, or `None` if one page already fits.

    `contact_sheet` pages every frame at a readable cell size rather than
    shrinking cells, which is right for detail and wrong for an overview: one
    recorded 576-frame draft came back as 24 pages, and the arms reviewing
    whole films built their own one-page sheet of every 12th frame instead. This
    is that page, at the same cell size, labeled with the stride so a gap
    between cells is read as sampling rather than as a cut. Advisory like
    `contact_sheet`: any failure returns `None`.
    """
    try:
        from PIL import Image
        frames = [(index, Path(path)) for index, path in frames if Path(path).is_file()]
        if len(frames) < 2:
            return None
        with Image.open(frames[0][1]) as first:
            aspect = first.height / first.width if first.width else 9 / 16
        cell_height = max(1, round(_SHEET_CELL_WIDTH * aspect))
        capacity = _SHEET_COLUMNS * max(1, (_SHEET_MAX_EDGE - _SHEET_HEAD)
                                        // (cell_height + _SHEET_BAR + _SHEET_GAP))
        if len(frames) <= capacity:
            return None
        stride = -(-len(frames) // capacity)
        picked = frames[::stride]
        written = contact_sheet(picked, destination, fps=fps,
                                label=f"Overview: one of every {stride} rendered frames, "
                                      f"{len(picked)} of {len(frames)}. "
                                      "The paged sheets hold every frame.")
        return written[0] if len(written) == 1 else None
    except Exception:                                           # noqa: BLE001
        return None
