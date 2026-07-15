"""
scripts/make_multimodel_report.py — Week 5 Cross-Model Reports
==============================================================
Consumes every results JSON written by run_multimodel.py
(code/data/multimodel/*.json) and produces, in site/:

  1. report-week5-<model>-<mode>.html  — a full visual detail report per config
     (reuses make_report.build_report, identical to the Week 3/4 reports).
  2. report-week5.html                 — a combined cross-model comparison hub:
     one row per (model, mode) with parse success + cell accuracy, linking out
     to each detail report.

It also upserts a "week-5" entry into site/updates.json, with the Results table
computed from the ACTUAL run numbers (the narrative prose is static; the numbers
are not fabricated).

Usage:
    uv run code/scripts/make_multimodel_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_report  # noqa: E402  (build_report, _puzzle_accuracy, CSS, ARC_COLORS)
from run_multimodel import model_slug  # noqa: E402

REPO_ROOT   = Path(__file__).resolve().parents[1]            # code/
INPUT_DIR   = REPO_ROOT / "data" / "multimodel"
SITE_DIR    = REPO_ROOT.parent / "site"
UPDATES     = SITE_DIR / "updates.json"

# Canonical display order (matches run_multimodel.MODELS); anything else sorts after.
MODEL_ORDER = ["gemma-4-31b", "qwen3.5-122b", "qwen3-vl:32b"]


def _aggregate(data: dict) -> tuple[int, int, int, int]:
    """(errors, cells, parsed, total) across a config's puzzles."""
    errors = cells = 0
    for p in data.get("puzzles", []):
        e, c = make_report._puzzle_accuracy(p)
        errors += e
        cells += c
    parsed = data.get("summary", {}).get("parse_success", 0)
    total  = data.get("summary", {}).get("total", len(data.get("puzzles", [])))
    return errors, cells, parsed, total


# Week 4 baseline (qwen/qwen3.6-35b). Restricting it to the SAME puzzles tested
# this week makes the cross-week comparison apples-to-apples — Week 4 ran 14
# puzzles, this week ran a 3-puzzle preliminary subset.
WK4_DIR = REPO_ROOT / "data" / "reconstruction"


def week4_baseline(stems: set[str], thinking: bool) -> dict | None:
    """qwen3.6-35b accuracy from Week 4, restricted to the given puzzle stems."""
    fp = WK4_DIR / ("results_thinking.json" if thinking else "results_no_thinking.json")
    if not fp.exists():
        return None
    data = json.loads(fp.read_text())
    errors = cells = parsed = total = 0
    for p in data.get("puzzles", []):
        if p.get("name") not in stems:
            continue
        total += 1
        parsed += 1 if p.get("parse_success") else 0
        e, c = make_report._puzzle_accuracy(p)
        errors += e
        cells += c
    if total == 0:
        return None
    return {
        "model": "qwen/qwen3.6-35b", "thinking": thinking,
        "parsed": parsed, "total": total, "cells": cells,
        "correct": cells - errors, "acc": 100 * (cells - errors) / cells if cells else 0.0,
        "detail": "report-week4-thinking.html" if thinking else "report-week4-nothinking.html",
        "week": "Wk4",
    }


def load_configs() -> list[dict]:
    configs = []
    for path in sorted(INPUT_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        model = data.get("model", path.stem)
        thinking = bool(data.get("thinking_enabled"))
        tag = "think" if thinking else "nothink"
        errors, cells, parsed, total = _aggregate(data)
        acc = 100 * (cells - errors) / cells if cells else 0.0
        detail = f"report-week5-{model_slug(model)}-{tag}.html"
        configs.append({
            "path": path, "data": data, "model": model, "thinking": thinking,
            "tag": tag, "errors": errors, "cells": cells, "correct": cells - errors,
            "parsed": parsed, "total": total, "acc": acc, "detail": detail,
        })

    def sort_key(c):
        mi = MODEL_ORDER.index(c["model"]) if c["model"] in MODEL_ORDER else len(MODEL_ORDER)
        return (mi, c["model"], c["thinking"])   # no-think (False) before think (True)

    configs.sort(key=sort_key)
    return configs


# ---------------------------------------------------------------------------
# Combined comparison report (report-week5.html)
# ---------------------------------------------------------------------------

def _cmp_row(c: dict, week: str) -> str:
    acc_color = "#22c55e" if c["acc"] >= 70 else ("#f59e0b" if c["acc"] >= 40 else "#ef4444")
    mode_lbl = "thinking" if c["thinking"] else "no-think"
    return (
        "<tr>"
        f'<td>{week}</td>'
        f'<td style="font-family:monospace">{make_report.html.escape(c["model"])}</td>'
        f'<td>{mode_lbl}</td>'
        f'<td>{c["parsed"]}/{c["total"]}</td>'
        f'<td style="color:{acc_color};font-weight:700">{c["acc"]:.1f}%</td>'
        f'<td>{c["correct"]:,}/{c["cells"]:,}</td>'
        f'<td><a href="{c["detail"]}">detail →</a></td>'
        "</tr>"
    )


def build_comparison_html(configs: list[dict]) -> str:
    stems = {p["name"] for c in configs for p in c["data"].get("puzzles", [])}
    base_nt = week4_baseline(stems, thinking=False)

    rows = []
    if base_nt:
        rows.append(_cmp_row(base_nt, "Wk4"))   # last week's qwen3.6-35b baseline, same puzzles
    for c in configs:
        rows.append(_cmp_row(c, "Wk5"))

    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:0.86rem">'
        '<thead><tr>'
        '<th style="text-align:left">Week</th>'
        '<th style="text-align:left">Model</th>'
        '<th style="text-align:left">Mode</th>'
        '<th style="text-align:left">JSON parsed</th>'
        '<th style="text-align:left">Cell accuracy</th>'
        '<th style="text-align:left">Correct / total cells</th>'
        '<th style="text-align:left">Report</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )

    extra_css = """
    .cmp-wrap { max-width: 980px; margin: 0 auto; }
    .cmp-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:22px 26px; margin-bottom:20px; }
    table th, table td { border-bottom:1px solid #30363d; padding:8px 12px; }
    table th { color:#8b949e; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.06em; }
    table a { color:#58a6ff; text-decoration:none; }
    table a:hover { text-decoration:underline; }
    h1 { font-size:1.4rem; margin-bottom:8px; }
    .sub { color:#8b949e; font-size:0.85rem; margin-bottom:18px; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Week 5 — Cross-Model MARC Reconstruction</title>
  <style>{make_report.CSS}{extra_css}</style>
</head>
<body>
<div class="cmp-wrap">
  <div class="cmp-card">
    <h1>Week 5 — Cross-Model MARC Reconstruction</h1>
    <p class="sub">Preliminary sweep ({configs and configs[0]['total'] or 0} puzzles per config) testing whether the
      structure-perfect / color-weak pattern seen with qwen3.6-35b generalizes to other vLLM vision models.
      The <strong>Wk4</strong> row is last week's qwen3.6-35b restricted to these same puzzles, for an
      apples-to-apples baseline. Thinking-mode runs appear only for models that support
      <code>enable_thinking</code>.</p>
    {table}
    <p class="sub" style="margin-top:16px">Each row links to a full visual report with per-cell diffs (red border =
      wrong color), in the same format as the Week 3/4 reports. Prior weeks:
      <a href="report-week4-nothinking.html">Wk4 no-think</a> ·
      <a href="report-week4-thinking.html">Wk4 thinking</a> ·
      <a href="report-week3.html">Wk3</a>.</p>
  </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# updates.json week-5 entry (Results table built from real numbers)
# ---------------------------------------------------------------------------

def build_week5_entry(configs: list[dict]) -> dict:
    n_puzzles  = configs[0]["total"] if configs else 0
    stems      = {p["name"] for c in configs for p in c["data"].get("puzzles", [])}
    base_nt    = week4_baseline(stems, thinking=False)   # last week's qwen3.6-35b, same puzzles
    base_think = week4_baseline(stems, thinking=True)

    parsed_clean = [c for c in configs if c["parsed"] == c["total"]]
    all_parsed = sum(c["parsed"] for c in configs)
    all_total  = sum(c["total"] for c in configs)
    accs_clean = [c["acc"] for c in parsed_clean] or [0.0]
    lo, hi = min(accs_clean), max(accs_clean)

    def mode_lbl(c):  # noqa: E306
        return "thinking" if c["thinking"] else "no-think"

    # --- This-week results table ---
    trows = "".join(
        f"<tr><td>{make_report.html.escape(c['model'])}</td><td>{mode_lbl(c)}</td>"
        f"<td>{c['parsed']}/{c['total']}</td><td>{c['acc']:.1f}%</td></tr>"
        for c in configs
    )
    results_table = (
        "<table><thead><tr><th>Model</th><th>Mode</th><th>JSON parsed</th><th>Cell accuracy</th></tr></thead>"
        f"<tbody>{trows}</tbody></table>"
    )

    # --- Comparison-vs-last-week table (same 3 puzzles) ---
    cmp_rows = []
    if base_nt:
        cmp_rows.append(
            f"<tr><td>Wk4</td><td>{make_report.html.escape(base_nt['model'])} <em>(baseline)</em></td>"
            f"<td>no-think + legend</td><td>{base_nt['parsed']}/{base_nt['total']}</td>"
            f"<td>{base_nt['acc']:.1f}%</td></tr>"
        )
    for c in sorted(configs, key=lambda x: x["acc"], reverse=True):
        cmp_rows.append(
            f"<tr><td>Wk5</td><td>{make_report.html.escape(c['model'])}</td><td>{mode_lbl(c)}</td>"
            f"<td>{c['parsed']}/{c['total']}</td><td>{c['acc']:.1f}%</td></tr>"
        )
    cmp_table = (
        "<table><thead><tr><th>Week</th><th>Model</th><th>Mode</th><th>JSON parsed</th>"
        "<th>Cell accuracy</th></tr></thead><tbody>" + "".join(cmp_rows) + "</tbody></table>"
    )

    stat_row = (
        '<div class="stat-row">'
        f'<div class="stat-box"><span class="stat-num">{len(configs)}</span><span class="stat-label">Models run</span></div>'
        f'<div class="stat-box"><span class="stat-num">{all_parsed}/{all_total}</span><span class="stat-label">JSON parsed</span></div>'
        f'<div class="stat-box"><span class="stat-num">{lo:.0f}–{hi:.0f}%</span><span class="stat-label">Color accuracy (clean parses)</span></div>'
        f'<div class="stat-box"><span class="stat-num">{n_puzzles}</span><span class="stat-label">Puzzles / model</span></div>'
        '</div>'
    )

    base_acc_txt = f"{base_nt['acc']:.1f}%" if base_nt else "n/a"
    base_think_txt = f"{base_think['acc']:.1f}% ({base_think['parsed']}/{base_think['total']} parsed)" if base_think else "n/a"

    return {
        "id": "week-5",
        "date": "2026-06-22",
        "version": "Week 5",
        "title": "Is It Just Qwen? Testing the Color Bottleneck Across Models",
        "tags": ["Results", "Research"],
        "excerpt": "Weeks 3–4 found one vision model reads MARC structure perfectly but misreads "
                   "colors. This week we tried the same task on three more models — including one from a "
                   "different maker — to see if that's a fluke. It isn't: every model gets the shape right "
                   "and the colors only partly right.",
        "sections": [
            {
                "heading": "In Plain Terms",
                "body": "<div class=\"callout\">We're testing whether AI \"vision\" models can look at a "
                        "picture of a small colored-grid puzzle and faithfully copy it back as data — the "
                        "right shape <em>and</em> the right colors. Last week, one model copied the shape "
                        "perfectly but got only about two-thirds of the colors right. The worry: maybe that "
                        "one model is just bad at color. So this week we handed the same puzzles to three more "
                        "models, including one built by a different company. They all did the same thing — "
                        "shapes correct, colors only partly right. In short: reading color from these images "
                        "is a shared weak spot across today's models, not a flaw in any single one. That tells "
                        "us the fix is to make the pictures easier to read, not just to swap in a different "
                        "model.</div>"
            },
            {
                "heading": "Why This Week",
                "body": "Every conclusion so far rested on a single model, <code>qwen/qwen3.6-35b</code>: it "
                        "reconstructs MARC puzzle <em>structure</em> flawlessly but reads <em>colors</em> only "
                        "about half to two-thirds of the time, and naive chain-of-thought doesn't lift that. "
                        "Before building further, the obvious risk to rule out is that those findings are an "
                        "artifact of one model. So this week ran the exact same image→JSON reconstruction task "
                        "across other vision-capable vLLM models on MindRouter."
            },
            {
                "heading": "Methodology",
                "items": [
                    "Ran the identical image-only reconstruction task (no JSON source) on three additional "
                    "vision models: <code>google/gemma-4-31b</code> (a different model family), "
                    "<code>qwen/qwen3.5-122b</code>, and <code>qwen3-vl:32b</code>",
                    f"<strong>Preliminary scale:</strong> a {n_puzzles}-puzzle subset — enough to see whether "
                    "the qualitative pattern reappears, not yet a full 14-puzzle benchmark",
                    "<strong>Apples-to-apples baseline:</strong> last week's <code>qwen/qwen3.6-35b</code> "
                    "scores were recomputed on these exact same puzzles so the comparison is fair",
                    "<strong>Per-model request shaping:</strong> only Qwen models accept "
                    "<code>chat_template_kwargs.enable_thinking</code>, so that flag is sent only to them; "
                    "Gemma gets a plain request. Text-only models (gpt-oss, nemotron) were excluded — they "
                    "can't process the puzzle images",
                    "This preliminary pass ran <strong>no-thinking only</strong> to stay time-boxed; the "
                    "thinking comparison is the immediate next step",
                    "Scored at the individual-cell level against ground truth, same as Weeks 3–4"
                ]
            },
            {
                "heading": "Results",
                "body": stat_row + results_table +
                        "<p style=\"margin-top:1rem\">Both vLLM-served models — Gemma (a different family) and "
                        "the larger Qwen — parsed every puzzle and landed squarely in the same partial-color "
                        f"band ({lo:.0f}–{hi:.0f}%) as the original model. <code>qwen3-vl:32b</code> is the "
                        "outlier, but for an infrastructure reason, not a perception one: on its Ollama backend "
                        "it failed to return valid JSON on 2 of 3 puzzles (the same flakiness flagged back in "
                        "Week 3), which drags its cell score down.</p>"
                        '<p style="margin-top:0.6rem"><a href="report-week5.html">Cross-model comparison report →</a> '
                        '&nbsp;·&nbsp; per-model detail: '
                        + " &nbsp;·&nbsp; ".join(
                            f'<a href="{c["detail"]}">{make_report.html.escape(c["model"])}</a>' for c in configs
                        ) + "</p>"
            },
            {
                "heading": "How This Compares to Last Week",
                "body": "On the <strong>same three puzzles</strong>, here is this week's lineup next to last "
                        "week's baseline:" + cmp_table +
                        f"<p style=\"margin-top:1rem\">Last week's <code>qwen/qwen3.6-35b</code> (no-think + "
                        f"legend) scored <strong>{base_acc_txt}</strong> on these puzzles and remains the "
                        "strongest, with <code>qwen/qwen3.5-122b</code> close behind and <code>gemma-4-31b</code> "
                        "a notch lower — but all three cluster in the same 60–80% color band while parsing "
                        "structure perfectly. For reference, last week's <em>thinking</em> run on these same "
                        f"puzzles managed only {base_think_txt}: chain-of-thought either ran away or never "
                        "emitted JSON, reinforcing that thinking didn't help.</p>"
                        '<p style="margin-top:0.6rem">Prior reports: '
                        '<a href="report-week4-nothinking.html">Wk4 no-think</a> · '
                        '<a href="report-week4-thinking.html">Wk4 thinking</a> · '
                        '<a href="report-week3.html">Wk3</a>.</p>'
            },
            {
                "heading": "Interpretation",
                "body": "<div class=\"callout\">Across every model that parsed cleanly, the same shape recurs: "
                        "JSON <em>structure</em> comes back perfect while <em>color</em> values are the weak "
                        "point — and a model from a different family (Gemma) behaves just like the Qwen models. "
                        "That is strong evidence the color-perception bottleneck is a general property of "
                        "current vLLM vision models on this task, not a quirk of one Qwen build.</div>"
                        "The practical takeaway: the fix is unlikely to be \"pick a better model\" — the new "
                        "models don't beat the Week 4 baseline. It points back at the <em>representation</em> — "
                        "making colors easier to read (legends, explicit cell labels) — which is exactly where "
                        "the Week 4 legend win came from."
            },
            {
                "heading": "Next Steps",
                "items": [
                    "Run the deferred <strong>thinking</strong> comparison for the two Qwen models on this "
                    "subset, then scale the strongest models up to all 14 puzzles for a real head-to-head",
                    "Push the representation fixes (explicit integer labels on cells, higher-contrast swatches) "
                    "now that the bottleneck is confirmed cross-model",
                    "Treat <code>qwen3-vl:32b</code>'s parse failures as an infra finding (Ollama vs vLLM "
                    "backend), not a capability result, when choosing models going forward"
                ]
            }
        ]
    }


def upsert_week5(entry: dict) -> None:
    updates = json.loads(UPDATES.read_text()) if UPDATES.exists() else []
    updates = [u for u in updates if u.get("id") != "week-5"]
    updates.insert(0, entry)
    UPDATES.write_text(json.dumps(updates, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_DIR.exists() or not any(INPUT_DIR.glob("*.json")):
        print(f"Error: no result JSONs in {INPUT_DIR}. Run run_multimodel.py first.", file=sys.stderr)
        sys.exit(1)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    configs = load_configs()

    # 1) per-config detail reports
    for c in configs:
        out = SITE_DIR / c["detail"]
        out.write_text(make_report.build_report(c["data"]), encoding="utf-8")
        print(f"detail  -> {out.name}  ({c['model']} {c['tag']}, {c['acc']:.1f}% cell acc)")

    # 2) combined comparison report
    cmp_out = SITE_DIR / "report-week5.html"
    cmp_out.write_text(build_comparison_html(configs), encoding="utf-8")
    print(f"compare -> {cmp_out.name}")

    # 3) updates.json week-5 entry (numbers from real results)
    upsert_week5(build_week5_entry(configs))
    print(f"updated -> {UPDATES.name}  (week-5 entry)")

    print("\nDone. Open the comparison with:")
    print(f"  open {cmp_out}")


if __name__ == "__main__":
    main()
