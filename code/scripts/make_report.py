"""
scripts/make_report.py — Visual HTML Report for MARC Reconstruction Results
============================================================================
Reads code/data/reconstruction/results.json and produces a self-contained
HTML report at code/data/reconstruction/report.html.

For each puzzle it renders:
  - Colored ARC grids (model vs correct) for every train/test pair
  - Cell-level diff highlighting (red outline = model got it wrong)
  - Per-puzzle accuracy stats
  - Collapsible reasoning trace

Usage:
    uv run code/scripts/make_report.py
    uv run code/scripts/make_report.py --input path/to/results.json --output path/to/report.html
"""
import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
INPUT_FILE  = REPO_ROOT / "data" / "reconstruction" / "results.json"
OUTPUT_FILE = REPO_ROOT / "data" / "reconstruction" / "report.html"

ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#ADD8E6",  # 8 light blue
    "#870C25",  # 9 maroon
]

CELL_PX = 22   # pixel size of each grid cell


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _grid_to_int(grid):
    if not grid or not isinstance(grid, list):
        return []
    return [[_to_int(c) for c in row] for row in grid if isinstance(row, list)]


def _diff_mask(model_grid, correct_grid):
    """Return a 2D bool array: True where values differ."""
    rows = len(correct_grid)
    mask = []
    for r in range(rows):
        cr = correct_grid[r]
        mr = model_grid[r] if r < len(model_grid) else []
        row_mask = []
        for c, cv in enumerate(cr):
            mv = mr[c] if c < len(mr) else -1
            row_mask.append(mv != cv)
        mask.append(row_mask)
    return mask


def _render_grid_html(grid, diff_mask=None, label=""):
    """Render a 2D integer grid as an HTML table of colored cells."""
    if not grid:
        return f'<div class="grid-wrap"><div class="grid-empty">(empty)</div></div>'

    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0
    total = rows * cols
    errors = sum(
        1 for r in range(rows) for c in range(len(grid[r]))
        if diff_mask and r < len(diff_mask) and c < len(diff_mask[r]) and diff_mask[r][c]
    ) if diff_mask else 0

    label_html = ""
    if label:
        acc_html = ""
        if diff_mask is not None:
            pct = 100 * (total - errors) / total if total else 0
            color = "#22c55e" if errors == 0 else ("#f59e0b" if pct >= 70 else "#ef4444")
            acc_html = f' <span style="color:{color};font-size:0.7rem;font-weight:600">{"✓" if errors==0 else f"{errors} err"}</span>'
        label_html = f'<div class="grid-label">{html.escape(label)}{acc_html}</div>'

    cells = []
    for r, row in enumerate(grid):
        cells.append('<tr>')
        for c, val in enumerate(row):
            color_idx = max(0, min(9, val))
            bg = ARC_COLORS[color_idx]
            # Determine text color for contrast
            txt = "#fff" if color_idx in (0, 1, 2, 6, 9) else "#000"
            wrong = diff_mask and r < len(diff_mask) and c < len(diff_mask[r]) and diff_mask[r][c]
            border = "2px solid #ef4444" if wrong else f"1px solid rgba(255,255,255,0.08)"
            cells.append(
                f'<td style="width:{CELL_PX}px;height:{CELL_PX}px;'
                f'background:{bg};border:{border};'
                f'font-size:9px;text-align:center;color:{txt};'
                f'font-weight:600;line-height:{CELL_PX}px;">'
                f'{val}</td>'
            )
        cells.append('</tr>')

    table = (
        f'<table style="border-collapse:collapse;border:1px solid #30363d;">'
        + "".join(cells)
        + "</table>"
    )
    return f'<div class="grid-wrap">{label_html}{table}</div>'


def _pair_section(pair_label, model_pair, correct_pair):
    """Render one input→output pair comparing model vs correct."""
    mo_input  = _grid_to_int(model_pair.get("input")   if model_pair else None)
    mo_output = _grid_to_int(model_pair.get("output")  if model_pair else None)
    co_input  = _grid_to_int(correct_pair.get("input")  if correct_pair else None)
    co_output = _grid_to_int(correct_pair.get("output") if correct_pair else None)

    diff_in  = _diff_mask(mo_input,  co_input)  if mo_input  and co_input  else None
    diff_out = _diff_mask(mo_output, co_output) if mo_output and co_output else None

    # Count errors across both grids
    in_errors  = sum(sum(r) for r in diff_in)  if diff_in  else 0
    out_errors = sum(sum(r) for r in diff_out) if diff_out else 0
    total_err  = in_errors + out_errors

    badge_color = "#22c55e" if total_err == 0 else ("#f59e0b" if total_err <= 5 else "#ef4444")
    badge = f'<span style="background:{badge_color};color:#fff;font-size:0.65rem;font-weight:700;padding:2px 7px;border-radius:4px;vertical-align:middle">' \
            + ("PERFECT" if total_err == 0 else f"{total_err} errors") + "</span>"

    model_row = (
        f'<div class="pair-row">'
        f'<div class="pair-who">Model</div>'
        + _render_grid_html(mo_input,  diff_in,  "Input")
        + '<div class="arrow">→</div>'
        + (_render_grid_html(mo_output, diff_out, "Output") if mo_output else '<div class="grid-wrap"><div class="grid-empty">(hidden)</div></div>')
        + '</div>'
    )
    correct_row = (
        f'<div class="pair-row">'
        f'<div class="pair-who">Correct</div>'
        + _render_grid_html(co_input,  None, "Input")
        + '<div class="arrow">→</div>'
        + _render_grid_html(co_output, None, "Output")
        + '</div>'
    )

    return (
        f'<div class="pair-block">'
        f'<div class="pair-heading">{html.escape(pair_label)} {badge}</div>'
        + model_row + correct_row
        + '</div>'
    )


# ---------------------------------------------------------------------------
# Puzzle card
# ---------------------------------------------------------------------------

def _puzzle_accuracy(puzzle):
    mo = puzzle.get("model_output") or {}
    co = puzzle.get("correct_output") or {}
    errors = cells = 0
    for split in ("train", "test"):
        for i, co_pair in enumerate(co.get(split) or []):
            if not isinstance(co_pair, dict):
                continue
            mo_pairs = mo.get(split) or []
            mo_pair = mo_pairs[i] if i < len(mo_pairs) and isinstance(mo_pairs[i], dict) else {}
            for key in ("input", "output"):
                cg = _grid_to_int(co_pair.get(key))
                mg = _grid_to_int(mo_pair.get(key))
                for r, row in enumerate(cg):
                    for c, cv in enumerate(row):
                        cells += 1
                        mv = mg[r][c] if r < len(mg) and c < len(mg[r]) else -1
                        if mv != cv:
                            errors += 1
    return errors, cells


def _puzzle_section(puzzle, idx):
    name     = puzzle.get("name", f"puzzle_{idx}")
    readable = name.replace("_", " ").title()
    metaphor = puzzle.get("metaphor") or ""
    mo = puzzle.get("model_output") or {}
    co = puzzle.get("correct_output") or {}
    reasoning = puzzle.get("reasoning_trace") or ""
    parse_ok  = puzzle.get("parse_success", False)

    errors, cells = _puzzle_accuracy(puzzle)
    pct = 100 * (cells - errors) / cells if cells else 0
    acc_color = "#22c55e" if errors == 0 else ("#f59e0b" if pct >= 70 else "#ef4444")
    parse_badge = (
        f'<span style="background:#22c55e;color:#fff;font-size:0.7rem;font-weight:700;'
        f'padding:2px 8px;border-radius:4px">PARSED</span>'
        if parse_ok else
        f'<span style="background:#ef4444;color:#fff;font-size:0.7rem;font-weight:700;'
        f'padding:2px 8px;border-radius:4px">PARSE FAIL</span>'
    )
    acc_badge = (
        f'<span style="background:{acc_color};color:#fff;font-size:0.7rem;font-weight:700;'
        f'padding:2px 8px;border-radius:4px">{pct:.0f}% ({cells-errors}/{cells})</span>'
    )

    pairs_html = []

    for split_label, split_key in (("Train", "train"), ("Test", "test")):
        co_pairs = co.get(split_key) or []
        mo_pairs = mo.get(split_key) or []
        for i, co_pair in enumerate(co_pairs):
            if not isinstance(co_pair, dict):
                continue
            mo_pair = mo_pairs[i] if i < len(mo_pairs) and isinstance(mo_pairs[i], dict) else {}
            pairs_html.append(_pair_section(f"{split_label} {i+1}", mo_pair, co_pair))

    reasoning_block = ""
    if reasoning.strip():
        reasoning_block = (
            f'<details class="reasoning-block">'
            f'<summary>Reasoning Trace</summary>'
            f'<pre class="reasoning-pre">{html.escape(reasoning)}</pre>'
            f'</details>'
        )

    return f"""
<section class="puzzle-card" id="puzzle-{idx}">
  <div class="card-header">
    <h2>{html.escape(readable)} {parse_badge} {acc_badge}</h2>
    {"" if not metaphor else f'<p class="metaphor">&#8220;{html.escape(metaphor)}&#8221;</p>'}
  </div>
  {"".join(pairs_html)}
  {reasoning_block}
</section>
"""


# ---------------------------------------------------------------------------
# Summary header
# ---------------------------------------------------------------------------

def _summary_header(data):
    model     = html.escape(data.get("model", "unknown"))
    timestamp = html.escape(data.get("timestamp", ""))
    puzzles   = data.get("puzzles", [])
    total     = len(puzzles)
    parsed    = sum(1 for p in puzzles if p.get("parse_success"))

    # Compute aggregate cell accuracy
    total_cells = total_errors = 0
    for p in puzzles:
        e, c = _puzzle_accuracy(p)
        total_errors += e
        total_cells  += c
    cell_acc = f"{100*(total_cells-total_errors)/total_cells:.1f}" if total_cells else "0.0"

    return f"""
<header class="report-header">
  <h1>MARC Reconstruction Report</h1>
  <div class="meta-row">
    <span><strong>Model:</strong> {model}</span>
    <span><strong>Time:</strong> {timestamp}</span>
  </div>
  <div class="stat-row">
    <div class="stat-box"><div class="stat-num">{parsed}/{total}</div><div class="stat-lbl">JSON Parsed</div></div>
    <div class="stat-box"><div class="stat-num">{cell_acc}%</div><div class="stat-lbl">Cell Accuracy</div></div>
    <div class="stat-box"><div class="stat-num">{total_cells - total_errors:,}</div><div class="stat-lbl">Correct Cells</div></div>
    <div class="stat-box"><div class="stat-num">{total_errors:,}</div><div class="stat-lbl">Wrong Cells</div></div>
  </div>
  <p class="legend-note">
    <strong>Reading the grids:</strong>
    Cells with a <span style="display:inline-block;width:12px;height:12px;border:2px solid #ef4444;border-radius:2px;vertical-align:middle"></span>
    red border are cells where the model&#39;s value differs from the correct value.
    Model rows are shown above correct rows for each pair.
  </p>
</header>
"""


# ---------------------------------------------------------------------------
# TOC
# ---------------------------------------------------------------------------

def _toc(puzzles):
    items = []
    for i, p in enumerate(puzzles):
        name = p.get("name", f"puzzle_{i+1}").replace("_", " ").title()
        e, c = _puzzle_accuracy(p)
        pct = 100 * (c - e) / c if c else 0
        color = "#22c55e" if e == 0 else ("#f59e0b" if pct >= 70 else "#ef4444")
        dot = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:5px;vertical-align:middle"></span>'
        items.append(f'<li>{dot}<a href="#puzzle-{i+1}">{html.escape(name)}</a> <span style="color:#8b949e;font-size:0.8em">{pct:.0f}%</span></li>')
    return f'<nav class="toc"><h2>Puzzles</h2><ol>{"".join(items)}</ol></nav>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0d1117;
  color: #e6edf3;
  padding: 24px 16px;
  line-height: 1.5;
}
.page-wrap { max-width: 1100px; margin: 0 auto; }

/* Header */
.report-header {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 22px 26px;
  margin-bottom: 24px;
}
.report-header h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 10px; }
.meta-row { font-size: 0.82rem; color: #8b949e; margin-bottom: 14px; display: flex; gap: 20px; flex-wrap: wrap; }
.meta-row strong { color: #e6edf3; }
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.stat-box { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 100px; }
.stat-num { font-size: 1.5rem; font-weight: 700; color: #3fb950; }
.stat-lbl { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; margin-top: 2px; }
.legend-note { font-size: 0.82rem; color: #8b949e; }
.legend-note strong { color: #e6edf3; }

/* TOC */
.toc { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }
.toc h2 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8b949e; margin-bottom: 10px; }
.toc ol { padding-left: 18px; column-count: 2; column-gap: 24px; }
.toc li { font-size: 0.83rem; margin-bottom: 4px; line-height: 1.5; }
.toc a { color: #58a6ff; text-decoration: none; }
.toc a:hover { text-decoration: underline; }

/* Puzzle cards */
.puzzle-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  margin-bottom: 20px;
  padding: 20px 22px;
}
.card-header { margin-bottom: 16px; }
.card-header h2 { font-size: 1rem; font-weight: 600; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.metaphor { color: #8b949e; font-style: italic; font-size: 0.85rem; margin-top: 5px; }

/* Pair blocks */
.pair-block { margin-bottom: 20px; }
.pair-heading { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em; color: #3fb950; font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.pair-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 6px; flex-wrap: wrap; }
.pair-who { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; font-weight: 600; width: 48px; flex-shrink: 0; padding-top: 20px; }
.grid-wrap { display: flex; flex-direction: column; }
.grid-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; margin-bottom: 4px; }
.grid-empty { font-size: 0.75rem; color: #8b949e; font-style: italic; padding: 8px; border: 1px dashed #30363d; border-radius: 4px; }
.arrow { font-size: 1.4rem; color: #8b949e; padding: 0 4px; align-self: center; padding-top: 20px; }

/* Reasoning */
details { margin-top: 14px; border-top: 1px solid #30363d; padding-top: 10px; }
summary { cursor: pointer; font-size: 0.8rem; color: #8b949e; user-select: none; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.65rem; }
details[open] > summary::before { content: "▼ "; }
summary:hover { color: #e6edf3; }
details[open] > summary { margin-bottom: 8px; }
.reasoning-pre {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 4px;
  padding: 10px 12px;
  font-size: 0.7rem;
  font-family: "SF Mono", "Fira Code", monospace;
  overflow: auto;
  max-height: 200px;
  white-space: pre-wrap;
  word-break: break-word;
  color: #8b949e;
}

@media (max-width: 600px) {
  .toc ol { column-count: 1; }
  .pair-row { gap: 6px; }
}
"""


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def build_report(data):
    puzzles = data.get("puzzles", [])
    sections = "".join(_puzzle_section(p, i + 1) for i, p in enumerate(puzzles))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MARC Reconstruction Report</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page-wrap">
{_summary_header(data)}
{_toc(puzzles)}
{sections}
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate visual HTML report from reconstruction results")
    parser.add_argument("--input",  type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run run_reconstruction.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(data), encoding="utf-8")

    puzzles = data.get("puzzles", [])
    total_e = total_c = 0
    for p in puzzles:
        e, c = _puzzle_accuracy(p)
        total_e += e; total_c += c
    pct = 100 * (total_c - total_e) / total_c if total_c else 0
    print(f"Report generated: {args.output}")
    print(f"  Cell accuracy: {pct:.1f}%  ({total_c-total_e:,}/{total_c:,} correct)")
    print(f"  Open with: open {args.output}")


if __name__ == "__main__":
    main()
