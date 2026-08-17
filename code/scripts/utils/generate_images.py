"""
Batch ARC puzzle image generator.

Handles both puzzle families:
  • MARC — train/test pairs, drawn as stacked "input → output" rows (matplotlib)
  • NARC — a narrative sequence, drawn as a left-to-right filmstrip (Pillow)

Usage:
    python scripts/utils/generate_images.py --input-dir puzzles/ --output-dir images/
    python scripts/utils/generate_images.py --input-dir puzzles/ --output-dir images/ --show-solution

NARC rendering is normally driven by scripts/narc_prepare.py rather than this
CLI, since it also validates the set and writes a manifest.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#ADD8E6", "#870C25",
]

ARC_COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow",
    "gray", "magenta", "orange", "lt.blue", "maroon",
]


def _add_legend(fig):
    """Draw a color legend strip below the figure content."""
    ax = fig.add_axes([0.03, 0.01, 0.94, 0.04])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (color, name) in enumerate(zip(ARC_COLORS, ARC_COLOR_NAMES)):
        rect = plt.Rectangle([i, 0.3], 0.85, 0.65, color=color)
        ax.add_patch(rect)
        # Outline for light colors
        outline = plt.Rectangle([i, 0.3], 0.85, 0.65, fill=False,
                                 edgecolor="#888888", linewidth=0.5)
        ax.add_patch(outline)
        ax.text(i + 0.425, 0.55, str(i), ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if i not in (4, 8) else "#333333")
        ax.text(i + 0.425, 0.15, name, ha="center", va="center",
                fontsize=5.5, color="#333333")


def _plot_grid(ax, grid, title: str = ""):
    if not grid:
        ax.axis("off")
        return
    try:
        arr = np.array(grid, dtype=int)
    except Exception:
        ax.axis("off")
        return
    h, w = arr.shape
    ax.imshow(arr, cmap=colors.ListedColormap(ARC_COLORS), norm=colors.Normalize(vmin=0, vmax=9))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, pad=5)


def visualize_puzzle_to_bytes(json_data: dict, show_solution: bool = False) -> bytes | None:
    metaphor    = json_data.get("metaphor", "")
    train_pairs = json_data.get("train", [])
    test_pairs  = json_data.get("test", [])
    num_rows    = len(train_pairs) + len(test_pairs)
    if num_rows == 0:
        return None
    fig = plt.figure(figsize=(10, 3 * num_rows + 1), constrained_layout=True)
    if metaphor:
        fig.suptitle(f"Metaphor: {metaphor}", fontsize=14, wrap=True, fontweight="bold")
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.92))
    else:
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 1.0))
    subfigs = fig.subfigures(num_rows, 1, wspace=0.1, hspace=0.1)
    if num_rows == 1:
        subfigs = [subfigs]
    all_pairs = ([(p, "Train", i + 1) for i, p in enumerate(train_pairs)]
                 + [(p, "Test",  i + 1) for i, p in enumerate(test_pairs)])
    for subfig, (pair, label, count) in zip(subfigs, all_pairs):
        axs = subfig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 0.2, 1]})
        _plot_grid(axs[0], pair.get("input"),  title=f"{label} {count} Input")
        axs[1].axis("off")
        axs[1].text(0.5, 0.5, "→", fontsize=30, ha="center", va="center")
        output = pair.get("output")
        if output and (label == "Train" or show_solution):
            _plot_grid(axs[2], output, title=f"{label} {count} Output")
        else:
            axs[2].axis("off")
            axs[2].text(0.5, 0.5, "?", fontsize=30, ha="center", va="center")
    _add_legend(fig)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# NARC — narrative sequence filmstrip
#
# This is a deliberate pixel-level port of renderNarcToDataUrl() in
# code/tools/model-baseline-analysis.html (line ~546): same cell sizing formula,
# same padding/gap/arrow constants, same white background and white gridlines,
# same hatched "?" box for masked frames, same bottom legend. Weeks 6-8 measured
# models against that exact picture, so a Python-rendered image has to read the
# same or the numbers aren't comparable.
#
# Rendered at 2x, matching the browser's devicePixelRatio scaling on a retina
# display (cv.width = canvasW * 2), so the model receives the same resolution.
# ---------------------------------------------------------------------------

NARC_SCALE = 2

# Layout constants — names and values mirror the JS.
_PAD, _GAP, _TITLE_H, _ARROW_W = 24, 26, 20, 22

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu at `size` CSS px, scaled. Bundled with matplotlib, so identical
    on every machine — and it has the → glyph the filmstrip needs."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(_FONT_DIR / name), size * NARC_SCALE)


def _text(draw, xy, s, font, fill, anchor="mm"):
    """Draw text with canvas-like anchoring.

    The JS sets textAlign="center" and textBaseline="middle" globally, so every
    coordinate in the port is a text *centre* — Pillow's "mm" anchor.
    """
    draw.text((xy[0] * NARC_SCALE, xy[1] * NARC_SCALE), s, font=font, fill=fill, anchor=anchor)


def _rect(draw, box, fill=None, outline=None, width=1):
    x0, y0, x1, y1 = (v * NARC_SCALE for v in box)
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width * NARC_SCALE)


def _narc_masked_positions(puzzle: dict) -> set[int]:
    """Authoritative masked set. Some exports leave the answer grid inline on a
    masked frame (stump/tortoise) while others null it out (sub_005), so this
    never infers 'visible' from the presence of a grid."""
    explicit = puzzle.get("masked_positions")
    if isinstance(explicit, list) and explicit:
        return {int(p) for p in explicit}
    return {int(f.get("position"))
            for f in (puzzle.get("sequence") or [])
            if f and (f.get("masked") or f.get("grid") is None)}


def _frame_label(frame: dict) -> str:
    """"Frame 3" or "Frame 3 · Wed." — mirrors narcFrameLabel() in the tool."""
    pos = frame.get("position", 0)
    label = (frame.get("label") or "").strip()
    return f"Frame {pos} · {label}" if label else f"Frame {pos}"


def _frame_dims(frame: dict) -> tuple[int, int]:
    """Rows/cols, preferring the declared values so masked frames whose grid is
    null still get a correctly-sized "?" box."""
    grid = frame.get("grid")
    rows = frame.get("rows") or (len(grid) if isinstance(grid, list) else 5)
    cols = frame.get("cols") or (len(grid[0]) if isinstance(grid, list) and grid else 5)
    return int(rows), int(cols)


def _draw_narc_grid(draw, grid, gx, top, cell):
    for r, row in enumerate(grid):
        for c, raw in enumerate(row):
            try:
                v = max(0, min(9, int(raw)))
            except (TypeError, ValueError):
                v = 0
            _rect(draw, (gx + c * cell, top + r * cell,
                         gx + (c + 1) * cell, top + (r + 1) * cell),
                  fill=ARC_COLORS[v])
    rows_n, cols_n = len(grid), len(grid[0])
    for r in range(rows_n + 1):
        y = (top + r * cell) * NARC_SCALE
        draw.line([(gx * NARC_SCALE, y), ((gx + cols_n * cell) * NARC_SCALE, y)],
                  fill="#ffffff", width=NARC_SCALE)
    for c in range(cols_n + 1):
        x = (gx + c * cell) * NARC_SCALE
        draw.line([(x, top * NARC_SCALE), (x, (top + rows_n * cell) * NARC_SCALE)],
                  fill="#ffffff", width=NARC_SCALE)


def _draw_narc_masked(img, draw, gx, top, w, h, cell):
    """Grey box + clipped diagonal hatch + blue "?". The hatch is drawn on a
    separate layer and pasted through a box-sized crop, which is how the JS
    ctx.clip() keeps it from bleeding onto neighbouring frames (the Week 8 bug)."""
    _rect(draw, (gx, top, gx + w, top + h), fill="#f2f2f2")

    s = NARC_SCALE
    hatch = Image.new("RGB", (int(w * s), int(h * s)), "#f2f2f2")
    hdraw = ImageDraw.Draw(hatch)
    d = -h
    while d < w:
        hdraw.line([((d) * s, 0), ((d + h) * s, h * s)], fill="#bbbbbb", width=s)
        d += 8
    img.paste(hatch, (int(gx * s), int(top * s)))

    _rect(draw, (gx, top, gx + w, top + h), outline="#888888")
    _text(draw, (gx + w / 2, top + h / 2), "?", _font(30, bold=True), "#1f6feb")


def narc_sequence_to_bytes(puzzle: dict, show_solution: bool = False) -> bytes | None:
    """Render a NARC puzzle to a PNG filmstrip. Returns None if it has no frames.

    With show_solution=False (the default, and what any model ever sees) masked
    frames are drawn as "?" boxes regardless of whether the source file left an
    answer grid inline on them.
    """
    seq = sorted((puzzle.get("sequence") or []),
                 key=lambda f: int(f.get("position", 0)))
    if not seq:
        return None

    masked_set = _narc_masked_positions(puzzle)
    answers = puzzle.get("answer_grids") or {}

    max_dim = max([1] + [d for f in seq for d in _frame_dims(f)])
    cell = max(6, min(20, 480 // max_dim))
    title = puzzle.get("title") or ""
    title_h = 34 if title else 6

    cards = []
    for f in seq:
        rows, cols = _frame_dims(f)
        pos = int(f.get("position", 0))
        is_masked = pos in masked_set
        grid = answers.get(str(pos)) if (is_masked and show_solution) else f.get("grid")
        if is_masked and not show_solution:
            grid = None
        cards.append({"frame": f, "rows": rows, "cols": cols,
                      "w": cols * cell, "h": rows * cell,
                      "masked": is_masked, "grid": grid})

    row_h = max([cell * 3] + [c["h"] for c in cards])
    x = _PAD
    for i, c in enumerate(cards):
        c["x"] = x
        x += c["w"] + (_GAP + _ARROW_W if i < len(cards) - 1 else 0)
    content_w = x - _PAD
    canvas_w = max(300, content_w + _PAD * 2)
    canvas_h = title_h + _TITLE_H + row_h + 20 + 46 + _PAD

    img = Image.new("RGB", (int(canvas_w * NARC_SCALE), int(canvas_h * NARC_SCALE)), "#ffffff")
    draw = ImageDraw.Draw(img)

    if title:
        _text(draw, (canvas_w / 2, _PAD - 6), f"Title: {title}", _font(15, bold=True), "#111111")

    grid_top = title_h + _TITLE_H
    for i, c in enumerate(cards):
        top = grid_top + (row_h - c["h"]) / 2
        _text(draw, (c["x"] + c["w"] / 2, grid_top - 9), _frame_label(c["frame"]),
              _font(11, bold=c["masked"]), "#1f6feb" if c["masked"] else "#333333")
        if isinstance(c["grid"], list) and c["grid"]:
            _draw_narc_grid(draw, c["grid"], c["x"], top, cell)
        else:
            _draw_narc_masked(img, draw, c["x"], top, c["w"], c["h"], cell)
        if i < len(cards) - 1:
            _text(draw, (c["x"] + c["w"] + _GAP / 2 + _ARROW_W / 2, grid_top + row_h / 2),
                  "→", _font(22), "#333333")

    # legend — the Week 4 intervention worth +11.7 points on MARC
    leg_top = canvas_h - _PAD - 26
    sw_w = min(64, content_w / 10)
    leg_x0 = _PAD + (content_w - sw_w * 10) / 2
    for i in range(10):
        sx = leg_x0 + i * sw_w
        _rect(draw, (sx + 2, leg_top, sx + sw_w - 4, leg_top + 18),
              fill=ARC_COLORS[i], outline="#888888")
        _text(draw, (sx + sw_w / 2 - 1, leg_top + 9), str(i), _font(10, bold=True),
              "#333333" if i in (4, 8) else "#ffffff")
        _text(draw, (sx + sw_w / 2 - 1, leg_top + 28), ARC_COLOR_NAMES[i], _font(8), "#333333")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def is_narc(puzzle: dict) -> bool:
    """Mirrors detectKind() in the tool."""
    return puzzle.get("schema") == "narc-export-v1" or isinstance(puzzle.get("sequence"), list)


def main():
    parser = argparse.ArgumentParser(description="Generate PNG images from ARC puzzle JSON files.")
    parser.add_argument("--input-dir",  required=True, type=Path, help="Directory containing puzzle JSON files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write PNG images")
    parser.add_argument("--show-solution", action="store_true", help="Reveal test output grids in the image")
    args = parser.parse_args()

    input_dir: Path  = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = 0
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  skip {path.name}: JSON error — {e}", file=sys.stderr)
            skipped += 1
            continue

        if is_narc(data):
            png = narc_sequence_to_bytes(data, show_solution=args.show_solution)
            empty_reason = "no sequence frames found"
        else:
            png = visualize_puzzle_to_bytes(data, show_solution=args.show_solution)
            empty_reason = "no train/test pairs found"
        if png is None:
            print(f"  skip {path.name}: {empty_reason}", file=sys.stderr)
            skipped += 1
            continue

        out_path = output_dir / (path.stem + ".png")
        out_path.write_bytes(png)
        print(f"  {path.name} -> {out_path}")
        ok += 1

    print(f"\nDone: {ok} images written, {skipped} skipped.")


if __name__ == "__main__":
    main()
