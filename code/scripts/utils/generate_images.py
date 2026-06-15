"""
Batch ARC puzzle image generator.

Usage:
    python scripts/utils/generate_images.py --input-dir puzzles/ --output-dir images/
    python scripts/utils/generate_images.py --input-dir puzzles/ --output-dir images/ --show-solution
"""
import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

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

        png = visualize_puzzle_to_bytes(data, show_solution=args.show_solution)
        if png is None:
            print(f"  skip {path.name}: no train/test pairs found", file=sys.stderr)
            skipped += 1
            continue

        out_path = output_dir / (path.stem + ".png")
        out_path.write_bytes(png)
        print(f"  {path.name} -> {out_path}")
        ok += 1

    print(f"\nDone: {ok} images written, {skipped} skipped.")


if __name__ == "__main__":
    main()
