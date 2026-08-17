"""
scripts/narc_score.py — the single place NARC metrics are computed
==================================================================
Reads results.jsonl (from run_narc_sweep.py or the browser tool's batch mode)
and derives every number the project reports. Both runners record only what was
sent and what came back; grading happens here, once, so a browser run and a
script run are never scored by two subtly different implementations.

WHY THE METRICS LOOK LIKE THIS
------------------------------
Weeks 3-6 measured perception with a single number — cell accuracy — and it
plateaued around 50-80% with the same verbal explanation every week: "structure
reads well, color reads poorly." That claim was never actually measured, because
one accuracy figure cannot distinguish the two. So every grid comparison here is
split three ways:

  shape_acc  — treat the grid as binary (background 0 vs. not) and score that.
               Did the model see WHERE things are?
  color_acc  — among cells where the model and the truth agree something is
               there, is the digit right? Did it see WHAT COLOR it is?
  cell_acc   — plain exact-digit match, the old number, kept for continuity.

A model at 95% shape and 55% color is a completely different engineering problem
from one at 55% shape — and the old metric reports both as "~55%".

The confusion matrix answers Week 4's open item ("probe which specific color
pairs the model confuses") directly. The palette has two pairs that are
plausible-confusable by construction: 1 blue #0074D9 vs 8 lt.blue #ADD8E6, and
5 gray #AAAAAA vs 0 black. The matrix confirms or refutes that.

Structural checks cover the rest of "did it get all the information": frame
count, which positions it returned, whether it preserved the order, whether the
dimensions are right, and whether it invented a masked frame it was told to skip.

Usage:
    uv run code/scripts/narc_score.py
    uv run code/scripts/narc_score.py --publish        # also copy into site/data/narc/
    uv run code/scripts/narc_score.py --self-test      # golden metric check, no data needed
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]        # code/
SITE_DIR = REPO_ROOT.parent / "site"
DATA_DIR = REPO_ROOT / "data" / "narc_baseline"
DEFAULT_IN = DATA_DIR / "results.jsonl"
PUBLISH_DIR = SITE_DIR / "data" / "narc"

# Text kept in the lite file for the dashboard's drill-down. Full traces stay in
# results.jsonl, which routinely runs to tens of MB and is not committed.
PREVIEW_CHARS = 4000

SOLVE_EXPERIMENTS = ("blind_solve", "recon_solve")


# ---------------------------------------------------------------------------
# Grid comparison — the shape/color decomposition
# ---------------------------------------------------------------------------

def _to_int_grid(g):
    if not isinstance(g, list):
        return []
    out = []
    for row in g:
        if not isinstance(row, list):
            continue
        vals = []
        for v in row:
            try:
                vals.append(int(v))
            except (TypeError, ValueError):
                vals.append(-1)          # unreadable cell: never equals a truth value
        out.append(vals)
    return out


def compare_grids(pred, truth) -> dict:
    """Score one predicted grid against one ground-truth grid.

    Iterates over TRUTH's cells, so a prediction that is too small is penalised
    for the cells it never produced (matching cellAccuracy() in the browser
    tool). A prediction that is too large is caught by dims_correct.
    """
    t = _to_int_grid(truth)
    p = _to_int_grid(pred)

    cells = correct = 0
    shape_correct = 0
    color_pairs = color_correct = 0
    fg_truth = fg_hit = 0
    bg_fp = bg_fn = missing = 0
    confusion: Counter = Counter()

    for r, trow in enumerate(t):
        for c, tv in enumerate(trow):
            cells += 1
            has = r < len(p) and c < len(p[r])
            pv = p[r][c] if has else None
            if not has:
                missing += 1

            if pv == tv:
                correct += 1

            t_fg, p_fg = tv != 0, (pv is not None and pv != 0)
            if t_fg == p_fg:
                shape_correct += 1
            elif t_fg and not p_fg:
                bg_fn += 1               # erased content the puzzle had
            elif p_fg and not t_fg:
                bg_fp += 1               # painted content the puzzle didn't have

            if t_fg:
                fg_truth += 1
                if p_fg:
                    fg_hit += 1
            # Colour is only meaningful where both agree something is present.
            if t_fg and p_fg:
                color_pairs += 1
                if pv == tv:
                    color_correct += 1

            if pv is not None and pv != tv:
                confusion[(tv, pv)] += 1

    dims_correct = bool(t) and len(p) == len(t) and all(
        len(p[i]) == len(t[i]) for i in range(len(t))
    )
    pct = lambda n, d: (100.0 * n / d) if d else None          # noqa: E731
    return {
        "cells": cells,
        "correct": correct,
        "cell_acc": pct(correct, cells),
        "shape_acc": pct(shape_correct, cells),
        "color_acc": pct(color_correct, color_pairs),
        "color_pairs": color_pairs,
        "fg_recall": pct(fg_hit, fg_truth),
        "fg_truth": fg_truth,
        "bg_fp": bg_fp,
        "bg_fn": bg_fn,
        "missing": missing,
        "dims_correct": dims_correct,
        "exact": bool(t) and dims_correct and correct == cells,
        "confusion": confusion,
    }


def _blank(truth) -> dict:
    """Score for a frame the model never produced: every truth cell is an error."""
    t = _to_int_grid(truth)
    cells = sum(len(r) for r in t)
    fg = sum(1 for r in t for v in r if v != 0)
    return {
        "cells": cells, "correct": 0, "cell_acc": 0.0 if cells else None,
        "shape_acc": 0.0 if cells else None, "color_acc": None, "color_pairs": 0,
        "fg_recall": 0.0 if fg else None, "fg_truth": fg,
        "bg_fp": 0, "bg_fn": fg, "missing": cells,
        "dims_correct": False, "exact": False, "confusion": Counter(),
    }


def _merge(parts: list[dict]) -> dict:
    """Aggregate per-frame scores into one run-level score (cell-weighted)."""
    agg = {k: 0 for k in ("cells", "correct", "color_pairs", "color_correct",
                          "shape_correct", "fg_truth", "fg_hit", "bg_fp", "bg_fn", "missing")}
    confusion: Counter = Counter()
    for m in parts:
        agg["cells"] += m["cells"]
        agg["correct"] += m["correct"]
        agg["color_pairs"] += m["color_pairs"]
        agg["color_correct"] += round((m["color_acc"] or 0) / 100 * m["color_pairs"])
        agg["shape_correct"] += round((m["shape_acc"] or 0) / 100 * m["cells"])
        agg["fg_truth"] += m["fg_truth"]
        agg["fg_hit"] += round((m["fg_recall"] or 0) / 100 * m["fg_truth"])
        agg["bg_fp"] += m["bg_fp"]
        agg["bg_fn"] += m["bg_fn"]
        agg["missing"] += m["missing"]
        confusion.update(m["confusion"])
    pct = lambda n, d: (100.0 * n / d) if d else None          # noqa: E731
    return {
        "cells": agg["cells"],
        "correct": agg["correct"],
        "cell_acc": pct(agg["correct"], agg["cells"]),
        "shape_acc": pct(agg["shape_correct"], agg["cells"]),
        "color_acc": pct(agg["color_correct"], agg["color_pairs"]),
        "color_pairs": agg["color_pairs"],
        "fg_recall": pct(agg["fg_hit"], agg["fg_truth"]),
        "fg_truth": agg["fg_truth"],
        "bg_fp": agg["bg_fp"], "bg_fn": agg["bg_fn"], "missing": agg["missing"],
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------

def _positions_in_order(mapping: dict) -> list[int]:
    """Emission order of a {position: grid} map.

    json.loads preserves object key order, and so does json.dumps on the way
    out, so the dict's insertion order is the order the model actually wrote the
    frames in — which is what `order_preserved` needs to test.
    """
    out = []
    for k in mapping:
        try:
            out.append(int(k))
        except (TypeError, ValueError):
            continue
    return out


def score_reconstruct(rec: dict) -> dict:
    """Perception: how faithfully did the model transcribe the VISIBLE frames?"""
    truth_frames = rec["truth"].get("visible_frames") or []
    masked = {int(p) for p in (rec["truth"].get("masked_positions") or [])}
    predicted = ((rec.get("predicted") or {}).get("frames") or {})

    emitted = _positions_in_order(predicted)
    expected = [int(f["position"]) for f in truth_frames]

    per_frame, parts = [], []
    for f in truth_frames:
        pos = int(f["position"])
        pred = predicted.get(str(pos))
        m = compare_grids(pred, f["grid"]) if pred is not None else _blank(f["grid"])
        parts.append(m)
        per_frame.append({
            "position": pos, "returned": pred is not None,
            "exact": m["exact"], "dims_correct": m["dims_correct"],
            "cell_acc": m["cell_acc"], "shape_acc": m["shape_acc"], "color_acc": m["color_acc"],
        })

    agg = _merge(parts)
    returned_set = set(emitted)
    return {
        "per_frame": per_frame,
        "agg": agg,
        "structural": {
            "n_frames_expected": len(expected),
            "n_frames_returned": len(emitted),
            "positions_exact_match": returned_set == set(expected),
            "missing_visible_frames": sorted(set(expected) - returned_set),
            "hallucinated_masked_frames": sorted(returned_set & masked),
            "extra_frames": sorted(returned_set - set(expected) - masked),
            # Only meaningful for the frames it was actually asked for.
            "order_preserved": [p for p in emitted if p in set(expected)] == sorted(
                p for p in emitted if p in set(expected)),
            "frames_with_correct_dims": sum(1 for f in per_frame if f["dims_correct"]),
            "frames_exact": sum(1 for f in per_frame if f["exact"]),
        },
    }


def score_solve(rec: dict) -> dict:
    """Solving: did it produce the right grid for every masked frame?"""
    masked = [int(p) for p in (rec["truth"].get("masked_positions") or [])]
    answers = rec["truth"].get("answer_grids") or {}
    predicted = ((rec.get("predicted") or {}).get("answers") or {})

    per_frame, parts = [], []
    for pos in masked:
        truth = answers.get(str(pos))
        pred = predicted.get(str(pos))
        m = compare_grids(pred, truth) if pred is not None else _blank(truth)
        parts.append(m)
        per_frame.append({
            "position": pos, "returned": pred is not None,
            "exact": m["exact"], "dims_correct": m["dims_correct"],
            "cell_acc": m["cell_acc"], "shape_acc": m["shape_acc"], "color_acc": m["color_acc"],
        })

    answered = {int(k) for k in _positions_in_order(predicted)}
    return {
        "per_frame": per_frame,
        "agg": _merge(parts),
        # ARC convention: a solve counts only if EVERY masked frame is exact.
        "solved": bool(per_frame) and all(f["exact"] for f in per_frame),
        "structural": {
            "n_frames_expected": len(masked),
            "n_frames_returned": len(answered),
            "answered_all_masked": set(masked) <= answered,
            "answered_extra_positions": sorted(answered - set(masked)),
            "frames_with_correct_dims": sum(1 for f in per_frame if f["dims_correct"]),
            "frames_exact": sum(1 for f in per_frame if f["exact"]),
        },
    }


def score_record(rec: dict) -> dict:
    if rec.get("experiment") == "reconstruct":
        s = score_reconstruct(rec)
        s["solved"] = None
    else:
        s = score_solve(rec)
    return s


# ---------------------------------------------------------------------------
# Flattening to CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    # identity / condition
    "run_key", "timestamp", "source", "puzzle_id", "puzzle_file", "title",
    "model", "experiment", "thinking", "include_narrative", "include_sizes",
    "image_source", "renderer",
    # outcome
    "error", "skipped", "parsed", "parse_needed_repair", "truncated", "solved",
    # perception decomposition
    "cells", "correct", "cell_acc", "shape_acc", "color_acc", "color_pairs",
    "fg_recall", "fg_truth", "bg_fp", "bg_fn", "missing_cells",
    # structural fidelity
    "n_frames_expected", "n_frames_returned", "frames_exact",
    "frames_with_correct_dims", "positions_exact_match", "order_preserved",
    "missing_visible_frames", "hallucinated_masked_frames", "extra_frames",
    "answered_all_masked", "answered_extra_positions",
    # cost
    "latency_ms", "content_chars", "reasoning_chars",
]


def _r(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) and v is not None else v


def flatten(rec: dict, s: dict) -> dict:
    agg, st = s["agg"], s["structural"]
    join = lambda xs: "; ".join(str(x) for x in xs) if xs else ""      # noqa: E731
    return {
        "run_key": rec.get("run_key", ""),
        "timestamp": rec.get("timestamp", ""),
        "source": rec.get("source", ""),
        "puzzle_id": rec.get("puzzle_id", ""),
        "puzzle_file": rec.get("puzzle_file", ""),
        "title": rec.get("title", ""),
        "model": rec.get("model", ""),
        "experiment": rec.get("experiment", ""),
        "thinking": rec.get("thinking"),
        "include_narrative": rec.get("include_narrative"),
        "include_sizes": rec.get("include_sizes"),
        "image_source": rec.get("image_source", ""),
        "renderer": rec.get("renderer", ""),
        "error": rec.get("error") or "",
        "skipped": bool(rec.get("skipped")),
        "parsed": bool(rec.get("parsed")),
        "parse_needed_repair": bool(rec.get("parse_needed_repair")),
        "truncated": bool(rec.get("truncated")),
        "solved": s.get("solved"),
        "cells": agg["cells"],
        "correct": agg["correct"],
        "cell_acc": _r(agg["cell_acc"]),
        "shape_acc": _r(agg["shape_acc"]),
        "color_acc": _r(agg["color_acc"]),
        "color_pairs": agg["color_pairs"],
        "fg_recall": _r(agg["fg_recall"]),
        "fg_truth": agg["fg_truth"],
        "bg_fp": agg["bg_fp"],
        "bg_fn": agg["bg_fn"],
        "missing_cells": agg["missing"],
        "n_frames_expected": st.get("n_frames_expected"),
        "n_frames_returned": st.get("n_frames_returned"),
        "frames_exact": st.get("frames_exact"),
        "frames_with_correct_dims": st.get("frames_with_correct_dims"),
        "positions_exact_match": st.get("positions_exact_match"),
        "order_preserved": st.get("order_preserved"),
        "missing_visible_frames": join(st.get("missing_visible_frames")),
        "hallucinated_masked_frames": join(st.get("hallucinated_masked_frames")),
        "extra_frames": join(st.get("extra_frames")),
        "answered_all_masked": st.get("answered_all_masked"),
        "answered_extra_positions": join(st.get("answered_extra_positions")),
        "latency_ms": rec.get("latency_ms"),
        "content_chars": rec.get("content_chars"),
        "reasoning_chars": rec.get("reasoning_chars"),
    }


def lite_record(rec: dict, s: dict) -> dict:
    """Everything the dashboard's drill-down needs, without the fat traces."""
    trim = lambda t: (t or "")[:PREVIEW_CHARS]                        # noqa: E731
    return {
        "run_key": rec.get("run_key"),
        "timestamp": rec.get("timestamp"),
        "puzzle_id": rec.get("puzzle_id"),
        "title": rec.get("title"),
        "model": rec.get("model"),
        "experiment": rec.get("experiment"),
        "thinking": rec.get("thinking"),
        "include_narrative": rec.get("include_narrative"),
        "include_sizes": rec.get("include_sizes"),
        "narrative": rec.get("narrative", ""),
        "error": rec.get("error"),
        "skipped": bool(rec.get("skipped")),
        "parsed": bool(rec.get("parsed")),
        "solved": s.get("solved"),
        "prompt": rec.get("prompt", ""),
        "raw_preview": trim(rec.get("raw")),
        "reasoning_preview": trim(rec.get("reasoning")),
        "raw_chars": rec.get("content_chars", 0),
        "reasoning_chars": rec.get("reasoning_chars", 0),
        "predicted": rec.get("predicted"),
        "truth": rec.get("truth"),
        "per_frame": s["per_frame"],
        "agg": {k: _r(v) for k, v in s["agg"].items() if k != "confusion"},
        "structural": s["structural"],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.fmean(xs), 2) if xs else None


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def _group_stats(rows: list[dict]) -> dict:
    """Cell-weighted accuracy plus per-run rates for one slice of the data."""
    scored = [r for r in rows if not r["error"] and not r["skipped"]]
    cells = sum(r["cells"] or 0 for r in scored)
    correct = sum(r["correct"] or 0 for r in scored)
    color_pairs = sum(r["color_pairs"] or 0 for r in scored)
    fg_truth = sum(r["fg_truth"] or 0 for r in scored)

    # Re-derive weighted numerators from the per-run percentages.
    shape_correct = sum((r["shape_acc"] or 0) / 100 * (r["cells"] or 0) for r in scored)
    color_correct = sum((r["color_acc"] or 0) / 100 * (r["color_pairs"] or 0) for r in scored)
    fg_hit = sum((r["fg_recall"] or 0) / 100 * (r["fg_truth"] or 0) for r in scored)

    solvable = [r for r in scored if r["solved"] is not None]
    pct = lambda n, d: round(100.0 * n / d, 2) if d else None          # noqa: E731
    return {
        "n_runs": len(rows),
        "n_scored": len(scored),
        "n_errors": sum(1 for r in rows if r["error"]),
        "n_skipped": sum(1 for r in rows if r["skipped"]),
        "parse_rate": pct(sum(1 for r in scored if r["parsed"]), len(scored)),
        "repair_rate": pct(sum(1 for r in scored if r["parse_needed_repair"]), len(scored)),
        "cells": cells,
        "cell_acc": pct(correct, cells),
        "shape_acc": pct(shape_correct, cells),
        "color_acc": pct(color_correct, color_pairs),
        "fg_recall": pct(fg_hit, fg_truth),
        "median_cell_acc": _median([r["cell_acc"] for r in scored]),
        "mean_cell_acc": _mean([r["cell_acc"] for r in scored]),
        "frames_exact": sum(r["frames_exact"] or 0 for r in scored),
        "frames_expected": sum(r["n_frames_expected"] or 0 for r in scored),
        "frame_exact_rate": pct(sum(r["frames_exact"] or 0 for r in scored),
                                sum(r["n_frames_expected"] or 0 for r in scored)),
        "solve_rate": pct(sum(1 for r in solvable if r["solved"]), len(solvable)),
        "n_solvable": len(solvable),
        "dims_ok_rate": pct(sum(r["frames_with_correct_dims"] or 0 for r in scored),
                            sum(r["n_frames_expected"] or 0 for r in scored)),
        "hallucinated_masked": sum(1 for r in scored if r["hallucinated_masked_frames"]),
        "median_latency_ms": _median([r["latency_ms"] for r in scored]),
        "total_reasoning_chars": sum(r["reasoning_chars"] or 0 for r in scored),
    }


def build_summary(rows: list[dict], confusions: dict) -> dict:
    by = lambda keyf: {k: _group_stats(v) for k, v in _bucket(rows, keyf).items()}  # noqa: E731

    # Structural cleanliness, most-severe-first. "clean" means the response was
    # well-formed — it says nothing about whether the grids were right, which is
    # what the accuracy columns are for.
    failure_modes: Counter = Counter()
    for r in rows:
        if r["error"]:
            failure_modes["api_error"] += 1
        elif r["skipped"]:
            failure_modes["skipped"] += 1
        elif not r["parsed"]:
            failure_modes["unparseable"] += 1
        elif r["hallucinated_masked_frames"]:
            failure_modes["hallucinated_masked_frame"] += 1
        elif r["experiment"] in SOLVE_EXPERIMENTS:
            # A solve is structurally complete once every masked frame is
            # answered; answering extra positions is off-spec but not a failure,
            # so it is tracked separately rather than masking a correct solve.
            if not r["answered_all_masked"]:
                failure_modes["missing_answer"] += 1
            elif (r["frames_with_correct_dims"] or 0) != (r["n_frames_expected"] or 0):
                failure_modes["wrong_dimensions"] += 1
            elif r["answered_extra_positions"]:
                failure_modes["extra_answers"] += 1
            else:
                failure_modes["clean"] += 1
        elif r["n_frames_returned"] != r["n_frames_expected"]:
            failure_modes["wrong_frame_count"] += 1
        elif (r["frames_with_correct_dims"] or 0) != (r["n_frames_expected"] or 0):
            failure_modes["wrong_dimensions"] += 1
        else:
            failure_modes["clean"] += 1

    return {
        "generated_at": None,      # stamped by the caller; keeps this pure
        "totals": _group_stats(rows),
        "by_experiment": by(lambda r: r["experiment"]),
        "by_model": by(lambda r: r["model"]),
        "by_model_experiment": by(lambda r: f"{r['model']} | {r['experiment']}"),
        "by_puzzle": by(lambda r: r["puzzle_id"]),
        "by_puzzle_experiment": by(lambda r: f"{r['puzzle_id']} | {r['experiment']}"),
        "by_condition": by(lambda r: "narr={} sizes={} think={}".format(
            r["include_narrative"], r["include_sizes"], r["thinking"])),
        "failure_modes": dict(failure_modes),
        "confusion": confusions,
        "models": sorted({r["model"] for r in rows if r["model"]}),
        "puzzles": sorted({r["puzzle_id"] for r in rows if r["puzzle_id"]}),
        "experiments": sorted({r["experiment"] for r in rows if r["experiment"]}),
    }


def _bucket(rows, keyf):
    out = defaultdict(list)
    for r in rows:
        out[keyf(r)].append(r)
    return out


def _confusion_to_matrix(counter: Counter) -> list[list[int]]:
    """10x10 matrix[truth][predicted]. Values outside 0-9 (unreadable cells) are
    dropped rather than folded into a real digit's row."""
    m = [[0] * 10 for _ in range(10)]
    for (t, p), n in counter.items():
        if 0 <= t <= 9 and 0 <= p <= 9:
            m[t][p] += n
    return m


# ---------------------------------------------------------------------------
# Self-test — guards the shape/color split against an off-by-one in the masking
# ---------------------------------------------------------------------------

def self_test() -> int:
    truth = [[0, 1, 2],
             [3, 0, 4]]
    pred = [[0, 8, 2],     # 1 -> 8 : both foreground, a pure COLOR error
            [0, 0, 4]]     # 3 -> 0 : foreground erased, a SHAPE error
    m = compare_grids(pred, truth)

    checks = [
        ("cells", m["cells"], 6),
        ("correct", m["correct"], 4),                    # 0,2 / 0,4
        ("cell_acc", round(m["cell_acc"], 2), round(100 * 4 / 6, 2)),
        # shape: 5 of 6 cells agree on background-vs-not (only 3->0 flips)
        ("shape_acc", round(m["shape_acc"], 2), round(100 * 5 / 6, 2)),
        # colour: both-foreground cells are (1,8), (2,2), (4,4) -> 2 of 3 right
        ("color_pairs", m["color_pairs"], 3),
        ("color_acc", round(m["color_acc"], 2), round(100 * 2 / 3, 2)),
        # foreground recall: truth has 4 non-bg cells, 3 survive
        ("fg_truth", m["fg_truth"], 4),
        ("fg_recall", round(m["fg_recall"], 2), round(100 * 3 / 4, 2)),
        ("bg_fn", m["bg_fn"], 1),
        ("bg_fp", m["bg_fp"], 0),
        ("dims_correct", m["dims_correct"], True),
        ("exact", m["exact"], False),
        ("confusion[1->8]", m["confusion"][(1, 8)], 1),
        ("confusion[3->0]", m["confusion"][(3, 0)], 1),
    ]

    # An undersized prediction must be penalised, not silently ignored.
    short = compare_grids([[0, 1, 2]], truth)
    checks += [
        ("short.cells", short["cells"], 6),
        ("short.missing", short["missing"], 3),
        ("short.dims_correct", short["dims_correct"], False),
    ]

    # A frame the model never returned scores zero, not None.
    blank = _blank(truth)
    checks += [
        ("blank.cell_acc", blank["cell_acc"], 0.0),
        ("blank.bg_fn", blank["bg_fn"], 4),
    ]

    failures = [(n, got, want) for n, got, want in checks if got != want]
    for name, got, want in checks:
        print(f"  {'ok  ' if got == want else 'FAIL'} {name:22s} got={got!r:<8} want={want!r}")
    if failures:
        print(f"\n{len(failures)} check(s) FAILED", file=sys.stderr)
        return 1
    print(f"\nAll {len(checks)} metric checks passed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Score a NARC sweep into CSV + summary JSON")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--publish", action="store_true",
                    help=f"Also copy the outputs into {PUBLISH_DIR} for the dashboard")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the golden metric checks and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.input.exists():
        print(f"Error: no results file at {args.input}\n"
              f"       Run: uv run code/scripts/run_narc_sweep.py --models <id>", file=sys.stderr)
        sys.exit(1)

    rows, lites = [], []
    confusion_all: Counter = Counter()
    confusion_by_model: dict[str, Counter] = defaultdict(Counter)
    confusion_by_model_exp: dict[str, Counter] = defaultdict(Counter)
    n_lines = n_bad = 0

    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if not rec.get("experiment") or "truth" not in rec:
                n_bad += 1
                continue

            s = score_record(rec)
            rows.append(flatten(rec, s))
            lites.append(lite_record(rec, s))

            conf = s["agg"]["confusion"]
            if conf and not rec.get("error"):
                confusion_all.update(conf)
                confusion_by_model[rec.get("model", "?")].update(conf)
                confusion_by_model_exp[f"{rec.get('model','?')} | {rec['experiment']}"].update(conf)

    if not rows:
        print(f"Error: no scorable records in {args.input}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = args.out_dir / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    lite_path = args.out_dir / "runs-lite.jsonl"
    with lite_path.open("w", encoding="utf-8") as fh:
        for rec in lites:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    confusions = {
        "all": _confusion_to_matrix(confusion_all),
        "by_model": {k: _confusion_to_matrix(v) for k, v in confusion_by_model.items()},
        "by_model_experiment": {k: _confusion_to_matrix(v)
                                for k, v in confusion_by_model_exp.items()},
    }
    summary = build_summary(rows, confusions)
    summary["generated_at"] = __import__("datetime").datetime.now().isoformat()
    summary["source_file"] = str(args.input)
    summary["n_records"] = len(rows)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 68)
    print(f"Scored {len(rows)} record(s) from {args.input.name}"
          + (f"  ({n_bad} unreadable line(s) skipped)" if n_bad else ""))
    print("=" * 68)
    t = summary["totals"]
    print(f"  parse rate          {t['parse_rate']}%")
    print(f"  cell accuracy       {t['cell_acc']}%   over {t['cells']:,} cells")
    print(f"    ├─ shape          {t['shape_acc']}%   (background vs not, all cells)")
    print(f"    └─ color          {t['color_acc']}%   (digit, on cells both call foreground)")
    print(f"  frames exact        {t['frame_exact_rate']}%  ({t['frames_exact']}/{t['frames_expected']})")
    if t["solve_rate"] is not None:
        print(f"  solve rate          {t['solve_rate']}%  ({t['n_solvable']} solve run(s))")
    print(f"  failure modes       {summary['failure_modes']}")
    print()
    fmt = lambda v: "  n/a" if v is None else f"{v:5.1f}%"          # noqa: E731
    for name in ("reconstruct", "blind_solve", "recon_solve"):
        g = summary["by_experiment"].get(name)
        if g:
            # reconstruct has no solve rate by design — it measures perception only
            print(f"  {name:14s} cell {fmt(g['cell_acc'])}  shape {fmt(g['shape_acc'])}  "
                  f"color {fmt(g['color_acc'])}  solved {fmt(g['solve_rate'])}")
    print()
    print(f"  wrote {runs_csv}")
    print(f"  wrote {lite_path}")
    print(f"  wrote {summary_path}")

    if args.publish:
        PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
        for src in (runs_csv, lite_path, summary_path, DATA_DIR / "puzzles.csv"):
            if src.exists():
                shutil.copy2(src, PUBLISH_DIR / src.name)
                print(f"  published {PUBLISH_DIR / src.name}")
        print("\n  Open site/narc-results.html to view.")
    print("=" * 68)


if __name__ == "__main__":
    main()
