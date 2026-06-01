"""
scripts/run_reconstruction.py — Qwen VLLM MARC Reconstruction Runner
=====================================================================
1. Verifies the Qwen diagnostic pipeline (qwen_experiment.py) works by:
   - Checking the script is syntactically valid
   - Sending a quick text-only ping to the API to confirm connectivity
   - Sending a test image to confirm multimodal capability
2. Sends all 14 MARC puzzle images to Qwen via MindRouter and asks it
   to reconstruct the full puzzle JSON (metaphor + train + test grids).
3. Saves model outputs, correct answers, and reasoning traces to
   code/data/reconstruction/results.json.

Usage:
    uv run code/scripts/run_reconstruction.py
    uv run code/scripts/run_reconstruction.py --model qwen3-vl:32b
"""
import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared utilities from recover_puzzle.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from recover_puzzle import (  # noqa: E402
    _load_api_key,
    _make_http_session,
    _parse_json_lenient,
    RECOVER_PROMPT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# vLLM-served model — uses continuous batching on H200s, much faster than Ollama
QWEN_MODEL = "qwen/qwen3.6-35b"
API_URL    = "https://mindrouter.uidaho.edu/v1/chat/completions"

REPO_ROOT   = Path(__file__).resolve().parents[1]   # code/
IMAGE_DIR   = REPO_ROOT / "images"
PUZZLE_DIR  = REPO_ROOT / "puzzles"
OUTPUT_DIR  = REPO_ROOT / "data" / "reconstruction"
OUTPUT_FILE = OUTPUT_DIR / "results.json"

QWEN_SCRIPT = Path(__file__).resolve().parent / "utils" / "qwen_experiment.py"


# ---------------------------------------------------------------------------
# Content extraction — keeps thinking/reasoning separate from JSON output
# ---------------------------------------------------------------------------

def _pick_content_and_thinking(data: dict) -> tuple[str, str]:
    """
    Extract (content, thinking) separately from an API response dict.

    Handles both OpenAI-format (choices[0].message) and Ollama-format
    (message.content + message.thinking). Returns ("", "") for thinking
    when the model does not emit a chain-of-thought field.
    """
    if not isinstance(data, dict):
        return str(data), ""

    # OpenAI-compatible format
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content  = msg.get("content",           "") or ""
        # Qwen3-VL returns reasoning in "reasoning_content"; others use "thinking"
        thinking = msg.get("reasoning_content", "") or msg.get("thinking", "") or ""
        if content or thinking:
            return content, thinking

    # Ollama format
    if "message" in data and isinstance(data["message"], dict):
        content  = data["message"].get("content",           "") or ""
        thinking = (data["message"].get("reasoning_content", "") or
                    data["message"].get("thinking",          "") or "")
        return content, thinking

    # Fallbacks
    for key in ("content", "response", "output", "text"):
        if key in data and isinstance(data[key], str):
            return data[key], ""

    return json.dumps(data), ""


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def _syntax_check_script(script: Path) -> tuple[bool, str]:
    """Verify the script compiles without syntax errors."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "py_compile", str(script)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, "syntax OK"


def _api_text_ping(model: str, api_url: str) -> tuple[bool, str]:
    """Quick text-only request to confirm the model is reachable."""
    import requests as _req
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _load_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5,
        "stream": False,
    }
    try:
        r = _req.post(api_url, json=payload, headers=headers, timeout=(10, 60))
        r.raise_for_status()
        return True, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)


def _api_image_ping(image_path: Path, model: str, api_url: str) -> tuple[bool, str]:
    """Send one puzzle image with max_tokens=20 to confirm multimodal capability."""
    import requests as _req
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _load_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in 5 words."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 20,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = _req.post(api_url, json=payload, headers=headers, timeout=(10, 120))
        r.raise_for_status()
        data = r.json()
        content, thinking = _pick_content_and_thinking(data)
        preview = (content or thinking or "")[:120]
        return True, f"HTTP {r.status_code} — response preview: {preview!r}"
    except Exception as exc:
        return False, str(exc)


def run_diagnostic(first_image: Path, model: str, api_url: str) -> tuple[bool, str]:
    """
    Three-step diagnostic for the Qwen pipeline:
      1. Syntax-check qwen_experiment.py
      2. Text-only API ping to confirm model is reachable
      3. Image API ping to confirm multimodal capability

    Returns (all_passed, summary_message).
    """
    lines = []

    ok, msg = _syntax_check_script(QWEN_SCRIPT)
    lines.append(f"  [{'OK' if ok else 'FAIL'}] qwen_experiment.py syntax check — {msg}")
    if not ok:
        return False, "\n".join(lines)

    ok, msg = _api_text_ping(model, api_url)
    lines.append(f"  [{'OK' if ok else 'FAIL'}] Text-only API ping ({model}) — {msg}")
    if not ok:
        return False, "\n".join(lines)

    ok, msg = _api_image_ping(first_image, model, api_url)
    lines.append(f"  [{'OK' if ok else 'FAIL'}] Image API ping ({first_image.name}) — {msg}")
    if not ok:
        return False, "\n".join(lines)

    return True, "\n".join(lines)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _post_llm_local(url: str, payload: dict) -> "requests.Response":
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.8,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["POST", "GET"]),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://",  HTTPAdapter(max_retries=retry))

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = _load_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return sess.post(url, headers=headers, json=payload,
                     timeout=(10, 1200), stream=False)


def reconstruct_puzzle(
    image_bytes: bytes, model: str, api_url: str
) -> tuple[dict | None, str, str]:
    """
    Send a puzzle image to the VLLM and ask it to reconstruct the JSON.

    Returns:
        (parsed_dict_or_None, raw_content, reasoning_trace)
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": RECOVER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "stream": False,
        # Disable thinking mode — pure transcription task needs no reasoning overhead
        "chat_template_kwargs": {"enable_thinking": False},
        # Enforce JSON output
        "response_format": {"type": "json_object"},
    }

    resp = _post_llm_local(api_url, payload)
    resp.raise_for_status()
    content, thinking = _pick_content_and_thinking(resp.json())

    try:
        parsed = _parse_json_lenient(content)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    return parsed, content, thinking


def load_correct_puzzle(stem: str) -> dict | None:
    path = PUZZLE_DIR / (stem + ".json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Qwen MARC reconstruction pipeline")
    parser.add_argument("--model",   default=QWEN_MODEL, help="Vision model name")
    parser.add_argument("--api-url", default=API_URL,    help="OpenAI-compatible endpoint")
    args = parser.parse_args()

    images = sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        print(f"Error: no PNG files found in {IMAGE_DIR}", file=sys.stderr)
        sys.exit(1)

    # ── Diagnostic ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Diagnostic — verifying qwen_experiment.py")
    print(f"  Image:  {images[0].name}")
    print(f"  Model:  {args.model}")
    print(f"  API:    {args.api_url}")
    print("=" * 60)

    ok, msg = run_diagnostic(images[0], args.model, args.api_url)
    if not ok:
        print(f"\nDiagnostic FAILED:\n{msg}", file=sys.stderr)
        sys.exit(1)

    print("Diagnostic PASSED.")
    print(msg)

    # ── Reconstruction ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Reconstruction — all MARC puzzles")
    print(f"  Model:  {args.model}")
    print(f"  API:    {args.api_url}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    puzzle_results = []
    total = len(images)

    for idx, img_path in enumerate(images, start=1):
        stem = img_path.stem
        correct = load_correct_puzzle(stem)
        metaphor = correct.get("metaphor", "") if correct else ""

        print(f"[{idx}/{total}] {stem} ...", flush=True)

        parsed = None
        content = ""
        thinking = ""
        error = None

        try:
            parsed, content, thinking = reconstruct_puzzle(
                img_path.read_bytes(), args.model, args.api_url
            )
        except Exception as exc:
            error = str(exc)
            print(f"         ERROR: {exc}", file=sys.stderr)

        status = "PASS" if parsed is not None else "FAIL"
        print(f"         {status} (parse_success={parsed is not None})")

        puzzle_results.append({
            "name":           stem,
            "metaphor":       metaphor,
            "model_output":   parsed,
            "correct_output": correct,
            "reasoning_trace": thinking,
            "raw_response":   content,
            "parse_success":  parsed is not None,
            "error":          error,
        })

    # ── Save results ─────────────────────────────────────────────────────────
    passed = sum(1 for p in puzzle_results if p["parse_success"])
    results = {
        "model":             args.model,
        "api_url":           args.api_url,
        "timestamp":         datetime.now().isoformat(),
        "diagnostic_passed": True,
        "summary": {
            "total":         total,
            "parse_success": passed,
            "parse_failed":  total - passed,
        },
        "puzzles": puzzle_results,
    }

    OUTPUT_FILE.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Done: {passed}/{total} puzzles parsed successfully.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Run 'uv run code/scripts/make_report.py' to generate the HTML report.")


if __name__ == "__main__":
    main()
