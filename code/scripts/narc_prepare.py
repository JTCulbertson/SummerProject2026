"""
scripts/narc_prepare.py — Validate the NARC set, render canonical images, manifest
==================================================================================
Step 1 of the NARC comprehension sweep. Before spending API budget we need to
know (a) which puzzles are actually scorable, (b) one canonical picture per
puzzle that every runner shares, and (c) how hard each puzzle is in the only
sense that matters for reading a solve score honestly.

Why the images are rendered once, to disk: the browser tool renders NARC on a
canvas (renderNarcToDataUrl) and this repo renders it with Pillow. Two renderers
drifting apart would silently make browser runs and script runs incomparable.
Rendering once and having both runners load the same PNG removes that entirely —
and makes the Week 7 greyscale comparison a matter of pointing at another folder.

The manifest's copy-previous-frame columns exist because Week 8 found a model
"blind-solving" a masked frame that was 92% identical to the frame before it.
Every solve score needs that number next to it or it reads as more impressive
than it is.

Usage:
    uv run code/scripts/narc_prepare.py
    uv run code/scripts/narc_prepare.py --puzzle-dir code/narc_puzzles --image-dir code/narc_images
    uv run code/scripts/narc_prepare.py --strict     # exit 1 if any puzzle is invalid
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))

from generate_images import narc_sequence_to_bytes, is_narc  # noqa: E402
from narc_prompts import narc_masked_positions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]      # code/
PUZZLE_DIR = REPO_ROOT / "narc_puzzles"
IMAGE_DIR = REPO_ROOT / "narc_images"
DATA_DIR = REPO_ROOT / "data" / "narc_baseline"

MANIFEST_FIELDS = [
    "puzzle_id", "file", "title", "n_frames", "masked_positions", "n_masked",
    "rows", "cols", "total_cells", "masked_cells", "n_distinct_colors",
    "colors_used", "narrative_chars", "difficulty", "creator",
    "pct_identical_to_prev_frame", "cells_changed_from_prev", "image",
]


# ---------------------------------------------------------------------------
# Validation — the tool's rules (validateNarcPuzzle, line 414), plus grid
# well-formedness checks the tool doesn't need but a sweep does.
# ---------------------------------------------------------------------------

def _grid_problems(grid, where: str) -> list[str]:
    """Rectangular, non-empty, all values 0-9."""
    if not isinstance(grid, list) or not grid:
        return [f"{where}: not a non-empty list"]
    problems = []
    widths = set()
    for r, row in enumerate(grid):
        if not isinstance(row, list):
            problems.append(f"{where}: row {r} is not a list")
            continue
        widths.add(len(row))
        for c, v in enumerate(row):
            try:
                iv = int(v)
            except (TypeError, ValueError):
                problems.append(f"{where}: cell [{r}][{c}] = {v!r} is not an integer")
                continue
            if not 0 <= iv <= 9:
                problems.append(f"{where}: cell [{r}][{c}] = {iv} outside 0-9")
    if len(widths) > 1:
        problems.append(f"{where}: ragged rows (widths {sorted(widths)})")
    return problems[:5]   # cap the noise; a broken grid produces hundreds


def validate(puzzle: dict) -> list[str]:
    """Return a list of problems; empty means the puzzle is scorable."""
    if not isinstance(puzzle, dict):
        return ["not a JSON object"]
    if not is_narc(puzzle):
        return ["not a NARC puzzle (no 'narc-export-v1' schema and no 'sequence' array)"]

    seq = puzzle.get("sequence")
    if not isinstance(seq, list) or not seq:
        return ["missing or empty 'sequence' array"]

    problems = []
    masked = narc_masked_positions(puzzle)
    if not masked:
        problems.append("no masked frames — nothing to solve")

    positions = [f.get("position") for f in seq if isinstance(f, dict)]
    if len(positions) != len(set(positions)):
        dupes = [p for p, n in Counter(positions).items() if n > 1]
        problems.append(f"duplicate frame position(s) {dupes}")

    answers = puzzle.get("answer_grids") or {}
    for pos in masked:
        grid = answers.get(str(pos))
        if not isinstance(grid, list) or not grid:
            problems.append(f"missing answer_grids['{pos}'] — masked frame is unscorable")
        else:
            problems.extend(_grid_problems(grid, f"answer_grids['{pos}']"))

    masked_set = set(masked)
    for f in seq:
        if not isinstance(f, dict):
            problems.append("sequence contains a non-object entry")
            continue
        pos = f.get("position")
        if pos is None:
            problems.append("a frame has no 'position'")
            continue
        if int(pos) in masked_set:
            continue          # visible-only checks below; masked frames may be null
        problems.extend(_grid_problems(f.get("grid"), f"frame {pos}"))
        grid = f.get("grid")
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            if f.get("rows") is not None and int(f["rows"]) != len(grid):
                problems.append(f"frame {pos}: rows={f['rows']} but grid has {len(grid)} rows")
            if f.get("cols") is not None and int(f["cols"]) != len(grid[0]):
                problems.append(f"frame {pos}: cols={f['cols']} but grid has {len(grid[0])} cols")

    return problems


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _resolved_grid(puzzle: dict, pos: int):
    """Ground-truth grid for a position: answer_grids for masked frames, the
    frame's own grid otherwise. Used only for descriptive statistics — never
    for anything that reaches a model."""
    masked = set(narc_masked_positions(puzzle))
    if pos in masked:
        g = (puzzle.get("answer_grids") or {}).get(str(pos))
        if isinstance(g, list) and g:
            return g
    for f in puzzle.get("sequence") or []:
        if isinstance(f, dict) and int(f.get("position", -1)) == pos:
            g = f.get("grid")
            if isinstance(g, list) and g:
                return g
    return None


def _copy_prev_baseline(puzzle: dict) -> tuple[str, str]:
    """How much of each masked frame you get for free by copying the frame
    before it. Week 8's yardstick — reported per masked position, e.g. "92.0"
    and "2" for the tortoise.
    """
    pcts, changes = [], []
    for pos in narc_masked_positions(puzzle):
        truth = _resolved_grid(puzzle, pos)
        prev = _resolved_grid(puzzle, pos - 1)
        if not truth or not prev:
            continue
        same = total = 0
        for r, row in enumerate(truth):
            for c, v in enumerate(row):
                total += 1
                pv = prev[r][c] if r < len(prev) and c < len(prev[r]) else None
                if pv is not None and int(pv) == int(v):
                    same += 1
        if total:
            pcts.append(round(100 * same / total, 1))
            changes.append(total - same)
    return ("; ".join(str(p) for p in pcts), "; ".join(str(c) for c in changes))


def manifest_row(path: Path, puzzle: dict, image_name: str) -> dict:
    seq = sorted((puzzle.get("sequence") or []), key=lambda f: int(f.get("position", 0)))
    masked = narc_masked_positions(puzzle)

    colors, total_cells, rows_seen, cols_seen = Counter(), 0, set(), set()
    for f in seq:
        grid = _resolved_grid(puzzle, int(f.get("position", 0)))
        if not grid:
            continue
        rows_seen.add(len(grid))
        cols_seen.add(len(grid[0]))
        for row in grid:
            total_cells += len(row)
            colors.update(int(v) for v in row)

    masked_cells = 0
    for pos in masked:
        g = (puzzle.get("answer_grids") or {}).get(str(pos))
        if isinstance(g, list):
            masked_cells += sum(len(r) for r in g)

    pct_same, changed = _copy_prev_baseline(puzzle)
    narrative = puzzle.get("narrative") or ""
    if not narrative:
        variants = puzzle.get("variants") or []
        if variants and isinstance(variants[0], dict):
            narrative = variants[0].get("narrative") or ""

    dim = lambda s: "; ".join(str(v) for v in sorted(s)) if s else ""  # noqa: E731
    return {
        "puzzle_id": puzzle.get("puzzle_id") or path.stem,
        "file": path.name,
        "title": puzzle.get("title") or "",
        "n_frames": len(seq),
        "masked_positions": "; ".join(str(p) for p in masked),
        "n_masked": len(masked),
        "rows": dim(rows_seen),
        "cols": dim(cols_seen),
        "total_cells": total_cells,
        "masked_cells": masked_cells,
        "n_distinct_colors": len(colors),
        "colors_used": "; ".join(str(c) for c in sorted(colors)),
        "narrative_chars": len(narrative),
        "difficulty": puzzle.get("difficulty") or "",
        "creator": puzzle.get("creator") or "",
        "pct_identical_to_prev_frame": pct_same,
        "cells_changed_from_prev": changed,
        "image": image_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate NARC puzzles, render canonical images, write a manifest")
    ap.add_argument("--puzzle-dir", type=Path, default=PUZZLE_DIR)
    ap.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--no-images", action="store_true", help="validate and write the manifest only")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any puzzle fails validation")
    args = ap.parse_args()

    if not args.puzzle_dir.is_dir():
        print(f"Error: puzzle dir not found: {args.puzzle_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(args.puzzle_dir.glob("*.json"))
    if not files:
        print(f"Error: no .json files in {args.puzzle_dir}", file=sys.stderr)
        sys.exit(1)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        args.image_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("NARC prepare")
    print(f"  puzzles: {args.puzzle_dir}  ({len(files)} file(s))")
    print(f"  images:  {'(skipped)' if args.no_images else args.image_dir}")
    print(f"  data:    {args.data_dir}")
    print("=" * 68)

    rows, rejected, seen_ids = [], [], {}
    for path in files:
        try:
            puzzle = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            rejected.append((path.name, [f"invalid JSON — {exc}"]))
            continue

        problems = validate(puzzle)
        if problems:
            rejected.append((path.name, problems))
            continue

        puzzle_id = puzzle.get("puzzle_id") or path.stem
        if puzzle_id in seen_ids:
            # Two files claiming one id would silently collide on the image
            # filename and on every resume key downstream.
            rejected.append((path.name, [f"duplicate puzzle_id '{puzzle_id}' (also in {seen_ids[puzzle_id]})"]))
            continue
        seen_ids[puzzle_id] = path.name

        image_name = ""
        if not args.no_images:
            png = narc_sequence_to_bytes(puzzle)
            if png is None:
                rejected.append((path.name, ["renderer produced no image"]))
                continue
            image_name = f"{puzzle_id}.png"
            (args.image_dir / image_name).write_bytes(png)

        rows.append(manifest_row(path, puzzle, image_name))
        print(f"  [ok]   {path.name} -> {puzzle_id}"
              f"  ({rows[-1]['n_frames']} frames, {rows[-1]['n_masked']} masked,"
              f" {rows[-1]['pct_identical_to_prev_frame'] or '?'}% same as prev)")

    for name, problems in rejected:
        print(f"  [FAIL] {name}", file=sys.stderr)
        for p in problems:
            print(f"           {p}", file=sys.stderr)

    manifest = args.data_dir / "puzzles.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 68)
    print(f"  valid:    {len(rows)}")
    print(f"  rejected: {len(rejected)}")
    print(f"  manifest: {manifest}")
    if not args.no_images and rows:
        print(f"  images:   {args.image_dir}")
    print("=" * 68)

    if rejected and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
