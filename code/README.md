# Code

## The NARC comprehension sweep

Weeks 3–6 measured perception with one number — cell accuracy — and it kept landing
between 50% and 80% with the same verbal gloss every week: *structure reads well,
colour reads poorly*. That claim was never actually measured, because a single
accuracy figure cannot separate the two. This pipeline runs every NARC puzzle
through the Week 6/8 three-experiment battery and scores each grid three ways:

| Metric | Question it answers |
|---|---|
| `cell_acc` | exact digit match — the old number, kept for continuity |
| `shape_acc` | background-vs-not over every cell — **did it see WHERE things are?** |
| `color_acc` | digit correctness on cells both sides call foreground — **did it see WHAT COLOUR?** |

plus a 10×10 colour-confusion matrix and structural checks (frame count, positions,
dimensions, order, and whether it invented the frame it was told to skip).

### Run it

```bash
# 1 · validate the set, render one canonical image per puzzle, write the manifest
uv run code/scripts/narc_prepare.py

# 2 · sweep — resumable; safe to interrupt and re-run
uv run code/scripts/run_narc_sweep.py --models qwen/qwen3.6-35b,google/gemma-4-31b

# 3 · score (the ONLY place metrics are computed) and publish for the dashboard
uv run code/scripts/narc_score.py --publish

# then open site/narc-results.html
```

Needs the MindRouter key in `code/keys.txt` and access to the university network.

### Files

| Path | Role |
|---|---|
| `scripts/narc_prepare.py` | validate → `narc_images/*.png` + `data/narc_baseline/puzzles.csv` |
| `scripts/run_narc_sweep.py` | headless resumable sweep → `data/narc_baseline/results.jsonl` |
| `scripts/narc_score.py` | the sole scorer → `runs.csv`, `runs-lite.jsonl`, `summary.json` |
| `scripts/utils/narc_prompts.py` | the three NARC prompts — single source of truth |
| `scripts/utils/narc_extract.py` | pulling grids out of off-spec model responses |
| `tools/model-baseline-analysis.html` | interactive bench + **Batch Sweep** (same record shape) |
| `../site/narc-results.html` | published dashboard over the scored output |

### Two runners, one experiment

The browser tool's Batch Sweep and `run_narc_sweep.py` must send **byte-identical
prompts** and write **byte-compatible records**, or the two halves of a sweep aren't
comparable. Neither scores anything — both record what was sent and what came back,
and `narc_score.py` grades everything, once.

`code/tests/run_parity.sh` enforces that:

```bash
./code/tests/run_parity.sh          # needs node on PATH
uv run code/scripts/narc_score.py --self-test    # golden metric checks
```

It has already caught one real drift: Python's `json.dumps` writes `{"3": "5x5"}`
where JavaScript's `JSON.stringify` writes `{"3":"5x5"}` — one space that silently
changed the prompt text. Anything that goes *into* a prompt now routes through
`compact_json()`.

### Two things to know about the data

- **`masked_positions` is authoritative.** `stump.json` and `tortoise.json` mark the
  masked frame `"masked": true` *but still carry the full answer grid inline* at
  `sequence[N].grid`, while `narc_puzzle_sub_005` nulls it out instead. Nothing may
  infer "visible" from the presence of a grid — `narc_visible_frames()` filters on
  `masked_positions` so the answer can never leak into a prompt.
- **Report solve rates against the copy-the-previous-frame baseline.** A NARC masked
  frame is usually a near-copy of the one before it (tortoise 92% identical, 2 cells
  change; stump 86.7%, 20 cells). `puzzles.csv` carries that column, and the
  dashboard shows it beside every solve rate. Week 8's single exact solve was an
  anecdote for exactly this reason.

### Ablations (wired, not yet run)

Both runners accept `--no-narrative` / `--no-sizes` (checkboxes in the tool). The
sweep defaults to narrative-on, sizes-on. Turning the 2×2 on multiplies API cost by
up to 4× — it's a follow-up run, not a code change.

## Earlier work

`scripts/run_reconstruction.py` (Week 3–4) and `scripts/run_multimodel.py` (Week 5)
cover the MARC reconstruction sweeps; the NARC runner reuses their streaming,
retry, and circuit-breaker handling rather than reimplementing it.
`scripts/cors_proxy.py` is what lets the browser tool reach MindRouter at all.
