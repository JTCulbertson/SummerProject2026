"""
Send ARC puzzle images to a vision model via mindrouter and recover the original JSON.

Usage:
    uv run scripts/utils/recover_puzzle.py --image-dir images/ --puzzle-dir puzzles/
    uv run scripts/utils/recover_puzzle.py --image-dir images/ --puzzle-dir puzzles/ --model openai/gpt-4o
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config — edit these to change the default target
# ---------------------------------------------------------------------------

MINDROUTER_URL = "https://mindrouter.uidaho.edu/v1/chat/completions"
MODEL          = "llama3.2-vision:11b"   # swap to any vision model on mindrouter

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RECOVER_PROMPT = """\
You are looking at an ARC-style puzzle rendered as an image.
The image shows training examples (each as an input grid and an output grid) \
and at least one test input grid.
There may also be a "Metaphor" title displayed above the puzzle.

Reconstruct the original JSON representation of this puzzle exactly.
Grids are 2D arrays of integers 0-9.

Respond with ONLY a JSON object in this exact format — no markdown, no explanation:
{
  "metaphor": "<title string, or empty string if none>",
  "train": [{"input": [[...], ...], "output": [[...], ...]}, ...],
  "test":  [{"input": [[...], ...]}]
}
"""

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    keys_file = Path(__file__).resolve().parents[2] / "keys.txt"
    if not keys_file.exists():
        return None
    for line in keys_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("Mindrouter_api"):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'")
    return None


def _make_http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.8,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["POST", "GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


def _post_llm(url: str, payload: dict, connect_secs: int = 10, read_secs: int = 1200) -> requests.Response:
    sess = _make_http_session()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = _load_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return sess.post(url, headers=headers, json=payload,
                     timeout=(connect_secs, read_secs), stream=False)


def _pick_content(data: dict) -> str | None:
    if not isinstance(data, dict):
        return str(data)
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        if "content" in msg:
            return msg["content"]
    if "message" in data and isinstance(data["message"], dict):
        thinking = data["message"].get("thinking", "")
        content  = data["message"].get("content", "")
        return (thinking + "\n" + content).strip() if thinking else content
    if "content" in data and isinstance(data["content"], str):
        return data["content"]
    for k in ("response", "output", "text"):
        if k in data and isinstance(data[k], str):
            return data[k]
    return json.dumps(data)


def _parse_json_lenient(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", s, flags=re.DOTALL)
        return json.loads(s.strip())

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def recover_puzzle(image_bytes: bytes, model: str, api_url: str) -> tuple[dict | None, str]:
    """Send image to model and return (parsed_dict_or_None, raw_response)."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RECOVER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.1,
        "stream": False,
    }
    resp = _post_llm(api_url, payload)
    resp.raise_for_status()
    raw = _pick_content(resp.json()) or ""
    try:
        return _parse_json_lenient(raw), raw
    except (json.JSONDecodeError, ValueError):
        return None, raw

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

WIDE = "═" * 54
THIN = "─" * 54


def _pprint(data: dict) -> str:
    return json.dumps(data, indent=2)


def _print_result(stem: str, model: str, model_json: dict | None, raw: str, actual: dict | None):
    print(WIDE)
    print(f"Puzzle: {stem}")
    print(f"Model:  {model}")
    print(THIN)
    print("MODEL OUTPUT:")
    if model_json is not None:
        print(_pprint(model_json))
    else:
        print(f"(could not parse JSON — raw response below)\n{raw}")
    if actual is not None:
        print(THIN)
        print("ACTUAL PUZZLE:")
        print(_pprint(actual))
    else:
        print(THIN)
        print("ACTUAL PUZZLE: (no matching JSON found)")
    print(WIDE)
    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recover ARC puzzle JSON from images using a vision model.")
    parser.add_argument("--image-dir",  required=True, type=Path,
                        help="Directory of PNG images (from generate_images.py)")
    parser.add_argument("--puzzle-dir", required=True, type=Path,
                        help="Directory of original puzzle JSON files")
    parser.add_argument("--model", default=MODEL,
                        help=f"Vision model name on mindrouter (default: {MODEL})")
    parser.add_argument("--api-url", default=MINDROUTER_URL,
                        help="Mindrouter API endpoint")
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        print(f"Error: --image-dir not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.puzzle_dir.is_dir():
        print(f"Error: --puzzle-dir not found: {args.puzzle_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted(args.image_dir.glob("*.png"))
    if not images:
        print(f"No PNG files found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    ok = failed = 0
    for img_path in images:
        stem = img_path.stem
        image_bytes = img_path.read_bytes()

        json_path = args.puzzle_dir / (stem + ".json")
        actual = None
        if json_path.exists():
            try:
                actual = json.loads(json_path.read_text())
            except json.JSONDecodeError as e:
                print(f"Warning: could not read {json_path.name}: {e}", file=sys.stderr)

        print(f"Sending {img_path.name} to {args.model} ...", flush=True)
        try:
            model_json, raw = recover_puzzle(image_bytes, args.model, args.api_url)
            _print_result(stem, args.model, model_json, raw, actual)
            ok += 1
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"Done: {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
