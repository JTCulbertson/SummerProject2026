"""
scripts/make_report.py — HTML Report Generator for MARC Reconstruction Results
===============================================================================
Reads code/data/reconstruction/results.json and produces a self-contained
HTML report (no external CDN dependencies) at
code/data/reconstruction/report.html.

The report shows, for each puzzle:
  - Puzzle name and metaphor
  - Parse success/fail badge
  - Side-by-side: model output JSON vs correct puzzle JSON
  - Collapsible reasoning trace box

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

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _e(s: str) -> str:
    return html.escape(str(s) if s is not None else "")


def _badge(success: bool) -> str:
    cls = "badge-pass" if success else "badge-fail"
    label = "PASS" if success else "FAIL"
    return f'<span class="{cls}">{label}</span>'


def _json_pre(obj, empty_msg: str = "(none)") -> str:
    if obj is None:
        return f'<pre class="json-block empty">{_e(empty_msg)}</pre>'
    return f'<pre class="json-block">{_e(json.dumps(obj, indent=2))}</pre>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _summary_header(data: dict) -> str:
    model     = _e(data.get("model", "unknown"))
    api_url   = _e(data.get("api_url", ""))
    timestamp = _e(data.get("timestamp", ""))
    diag      = data.get("diagnostic_passed", False)
    summary   = data.get("summary", {})
    total     = summary.get("total", len(data.get("puzzles", [])))
    passed    = summary.get("parse_success", sum(
        1 for p in data.get("puzzles", []) if p.get("parse_success")
    ))
    pct       = f"{100 * passed / total:.1f}" if total else "0.0"
    diag_html = _badge(diag)

    return f"""
<header class="report-header">
  <h1>MARC Reconstruction Report</h1>
  <div class="meta-grid">
    <div class="meta-item"><span class="meta-label">Model</span><span class="meta-value">{model}</span></div>
    <div class="meta-item"><span class="meta-label">API</span><span class="meta-value">{api_url}</span></div>
    <div class="meta-item"><span class="meta-label">Timestamp</span><span class="meta-value">{timestamp}</span></div>
    <div class="meta-item"><span class="meta-label">Diagnostic</span><span class="meta-value">{diag_html}</span></div>
  </div>
  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-number">{total}</div>
      <div class="stat-label">Total Puzzles</div>
    </div>
    <div class="stat-box">
      <div class="stat-number">{passed}</div>
      <div class="stat-label">Parsed OK</div>
    </div>
    <div class="stat-box">
      <div class="stat-number">{total - passed}</div>
      <div class="stat-label">Parse Failed</div>
    </div>
    <div class="stat-box highlight">
      <div class="stat-number">{pct}%</div>
      <div class="stat-label">Success Rate</div>
    </div>
  </div>
</header>
"""


def _puzzle_section(puzzle: dict, idx: int) -> str:
    name     = puzzle.get("name", f"puzzle_{idx}")
    readable = name.replace("_", " ").title()
    metaphor = _e(puzzle.get("metaphor") or "")
    success  = puzzle.get("parse_success", False)
    error    = puzzle.get("error")

    model_out  = puzzle.get("model_output")
    correct    = puzzle.get("correct_output")
    reasoning  = puzzle.get("reasoning_trace") or ""
    raw_resp   = puzzle.get("raw_response") or ""

    error_html = ""
    if error:
        error_html = f'<div class="error-box"><strong>Error:</strong> {_e(error)}</div>'

    reasoning_content = (
        f'<pre class="json-block reasoning-pre">{_e(reasoning)}</pre>'
        if reasoning.strip()
        else '<p class="empty-note">(no reasoning trace — model did not emit a thinking field)</p>'
    )

    raw_content = (
        f'<pre class="json-block raw-pre">{_e(raw_resp)}</pre>'
        if raw_resp.strip()
        else '<p class="empty-note">(no raw response)</p>'
    )

    return f"""
<section class="puzzle-card" id="puzzle-{idx}">
  <div class="card-header">
    <h2>{_e(readable)} {_badge(success)}</h2>
    {f'<p class="metaphor">"{metaphor}"</p>' if metaphor else ""}
  </div>
  {error_html}
  <div class="comparison-grid">
    <div class="col">
      <h3 class="col-heading">Model Output</h3>
      {_json_pre(model_out, "(parse failed — see raw response below)")}
    </div>
    <div class="col">
      <h3 class="col-heading">Correct Puzzle</h3>
      {_json_pre(correct, "(no ground-truth JSON found)")}
    </div>
  </div>

  <details class="reasoning-block">
    <summary>&#x1F9E0; Reasoning Trace</summary>
    {reasoning_content}
  </details>

  <details class="raw-block">
    <summary>&#x1F4C4; Raw Model Response</summary>
    {raw_content}
  </details>
</section>
"""


# ---------------------------------------------------------------------------
# Inline CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f0f2f5;
  color: #1e293b;
  padding: 24px 16px;
  line-height: 1.5;
}

.page-wrap {
  max-width: 1280px;
  margin: 0 auto;
}

/* ── Header ─────────────────────────────────────────── */
.report-header {
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 10px;
  padding: 24px 28px;
  margin-bottom: 28px;
}

.report-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 16px;
  color: #f8fafc;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}

.meta-item {
  display: flex;
  gap: 8px;
  align-items: baseline;
  font-size: 0.82rem;
}

.meta-label {
  color: #94a3b8;
  min-width: 72px;
  flex-shrink: 0;
}

.meta-value {
  color: #cbd5e1;
  word-break: break-all;
}

.stat-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.stat-box {
  background: rgba(255,255,255,0.07);
  border-radius: 6px;
  padding: 10px 18px;
  text-align: center;
  min-width: 90px;
}

.stat-box.highlight { background: rgba(99,102,241,0.25); }

.stat-number {
  font-size: 1.6rem;
  font-weight: 700;
  color: #f1f5f9;
}

.stat-label {
  font-size: 0.72rem;
  color: #94a3b8;
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Badges ──────────────────────────────────────────── */
.badge-pass {
  background: #22c55e;
  color: #fff;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  vertical-align: middle;
  letter-spacing: 0.04em;
}

.badge-fail {
  background: #ef4444;
  color: #fff;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 4px;
  vertical-align: middle;
  letter-spacing: 0.04em;
}

/* ── Puzzle cards ────────────────────────────────────── */
.puzzle-card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  margin-bottom: 20px;
  padding: 22px 24px;
}

.card-header {
  margin-bottom: 14px;
}

.card-header h2 {
  font-size: 1.05rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.metaphor {
  color: #64748b;
  font-style: italic;
  font-size: 0.88rem;
  margin-top: 5px;
}

.error-box {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 0.82rem;
  color: #b91c1c;
  margin-bottom: 12px;
}

/* ── Two-column comparison ───────────────────────────── */
.comparison-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 14px;
}

@media (max-width: 720px) {
  .comparison-grid { grid-template-columns: 1fr; }
}

.col-heading {
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #475569;
  margin-bottom: 6px;
}

/* ── JSON / pre blocks ───────────────────────────────── */
.json-block {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 12px 14px;
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  font-size: 0.72rem;
  line-height: 1.55;
  overflow: auto;
  max-height: 320px;
  white-space: pre-wrap;
  word-break: break-word;
  color: #1e293b;
}

.json-block.empty {
  color: #94a3b8;
  font-style: italic;
}

.reasoning-pre {
  max-height: 240px;
  background: #fafaf7;
  border-color: #e5e5d8;
  font-size: 0.70rem;
}

.raw-pre {
  max-height: 200px;
  background: #f9fafb;
  border-color: #dde1e7;
  font-size: 0.69rem;
}

/* ── Details / reasoning ─────────────────────────────── */
details {
  border-top: 1px solid #e8ecf0;
  padding-top: 10px;
  margin-top: 4px;
}

details + details {
  margin-top: 8px;
}

summary {
  cursor: pointer;
  font-size: 0.82rem;
  color: #475569;
  user-select: none;
  padding: 2px 0;
  list-style: none;
}

summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.65rem; }
details[open] > summary::before { content: "▼ "; }
summary:hover { color: #1e293b; }

details[open] > summary { margin-bottom: 8px; }

.empty-note {
  color: #94a3b8;
  font-style: italic;
  font-size: 0.82rem;
  padding: 4px 0;
}

/* ── TOC sidebar (small screens hide it) ─────────────── */
.toc {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07);
  padding: 16px 18px;
  margin-bottom: 20px;
  font-size: 0.82rem;
}

.toc h2 { font-size: 0.88rem; margin-bottom: 8px; color: #374151; }
.toc ol { padding-left: 20px; line-height: 1.9; }
.toc a  { color: #6366f1; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

def build_report(data: dict) -> str:
    puzzles = data.get("puzzles", [])

    toc_items = "\n".join(
        f'<li><a href="#puzzle-{i+1}">{_e(p.get("name", f"puzzle_{i+1}").replace("_", " ").title())}'
        f' {_badge(p.get("parse_success", False))}</a></li>'
        for i, p in enumerate(puzzles)
    )

    sections = "\n".join(_puzzle_section(p, i + 1) for i, p in enumerate(puzzles))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MARC Reconstruction Report</title>
  <style>
{CSS}
  </style>
</head>
<body>
<div class="page-wrap">

{_summary_header(data)}

<div class="toc">
  <h2>Puzzles</h2>
  <ol>
{toc_items}
  </ol>
</div>

{sections}

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from reconstruction results")
    parser.add_argument("--input",  type=Path, default=INPUT_FILE,  help="Path to results.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Path for output report.html")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: results file not found: {args.input}", file=sys.stderr)
        print("Run 'uv run code/scripts/run_reconstruction.py' first.", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(args.input.read_text())
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse {args.input}: {exc}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_report(data)
    args.output.write_text(html_content, encoding="utf-8")

    puzzles = data.get("puzzles", [])
    passed  = sum(1 for p in puzzles if p.get("parse_success"))
    print(f"Report generated: {args.output}")
    print(f"  {passed}/{len(puzzles)} puzzles parsed successfully.")
    print(f"  Open with: open {args.output}")


if __name__ == "__main__":
    main()
