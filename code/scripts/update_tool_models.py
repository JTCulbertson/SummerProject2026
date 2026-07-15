#!/usr/bin/env python3
"""
Refresh the baked-in model dropdown in site/model-baseline-analysis.html.

The tool ships a static SEED_MODELS list so its dropdown works without the CORS
proxy running. Model availability on MindRouter changes rarely — run this whenever
you want to update that baked-in snapshot. This talks to MindRouter directly
(server-side, so no CORS and no proxy needed).

    uv run code/scripts/update_tool_models.py
"""
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
KEYS = REPO / "code" / "keys.txt"
TOOL = REPO / "site" / "model-baseline-analysis.html"
MODELS_URL = "https://mindrouter.uidaho.edu/v1/models"


def load_api_key() -> str | None:
    if not KEYS.exists():
        return None
    for line in KEYS.read_text().splitlines():
        line = line.strip()
        if line.startswith("Mindrouter_api"):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def main() -> None:
    key = load_api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urlopen(Request(MODELS_URL, headers=headers), timeout=60) as r:
            data = json.load(r)
    except Exception as e:  # noqa: BLE001 — surface any network/auth error plainly
        print(f"Failed to fetch models from {MODELS_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    ids = sorted({m.get("id") for m in data.get("data", []) if m.get("id")})
    if not ids:
        print("No models returned; aborting (nothing written).", file=sys.stderr)
        sys.exit(1)

    arr = "[" + ", ".join(json.dumps(i) for i in ids) + "]"
    html = TOOL.read_text()
    new_html, n = re.subn(
        r"const SEED_MODELS = \[.*?\];",
        f"const SEED_MODELS = {arr};",
        html, count=1, flags=re.S,
    )
    if n != 1:
        print("Could not locate the SEED_MODELS array in the tool HTML.", file=sys.stderr)
        sys.exit(1)

    if new_html == html:
        print(f"SEED_MODELS already up to date ({len(ids)} models).")
        return
    TOOL.write_text(new_html)
    print(f"Updated SEED_MODELS with {len(ids)} models in {TOOL.relative_to(REPO)}")


if __name__ == "__main__":
    main()
