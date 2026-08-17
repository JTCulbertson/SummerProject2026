#!/usr/bin/env bash
# Confirm the browser tool and the Python sweep agree.
#
# The NARC experiment has two runners — code/tools/model-baseline-analysis.html
# (batch mode) and code/scripts/run_narc_sweep.py — that must send byte-identical
# prompts and record byte-compatible results, or the two halves of a sweep aren't
# comparable. This has already caught one real drift: Python's json.dumps writes
# `{"3": "5x5"}` where JavaScript's JSON.stringify writes `{"3":"5x5"}`, which
# silently changed the prompt text.
#
# Runs the tool's actual <script> in a stubbed DOM (node), dumps its outputs, and
# diffs them against the Python implementation.
#
# Requires: node on PATH, uv for the Python side.
#
#   ./code/tests/run_parity.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
TOOL="code/tools/model-baseline-analysis.html"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! command -v node >/dev/null 2>&1; then
  echo "node not found on PATH — needed to execute the tool's JavaScript." >&2
  exit 1
fi

# A second fixture with TWO masked frames of different sizes, so the multi-key
# and comma separators in the prompt's JSON literals are exercised too.
uv run python - "$TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
p = json.loads(Path("code/narc_puzzles/stump.json").read_text())
p["puzzle_id"] = "parity_multi_masked"
p["masked_positions"] = [1, 3]
first = [row[:7] for row in p["sequence"][1]["grid"][:4]]
p["answer_grids"] = {"1": first, "3": p["answer_grids"]["3"]}
(tmp / "multi.json").write_text(json.dumps(p))
PY

status=0
for puzzle in code/narc_puzzles/tortoise.json "$TMP/multi.json"; do
  echo "── $(basename "$puzzle") ──"
  node code/tests/parity_dump.js "$TOOL" "$puzzle" > "$TMP/dump.json"
  uv run python code/tests/parity_check.py "$puzzle" "$TMP/dump.json" | grep -v '^  ok' || status=1
done
exit $status
