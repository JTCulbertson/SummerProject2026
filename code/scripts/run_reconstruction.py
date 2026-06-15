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
import time
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

REPO_ROOT  = Path(__file__).resolve().parents[1]   # code/
IMAGE_DIR  = REPO_ROOT / "images"
PUZZLE_DIR = REPO_ROOT / "puzzles"
OUTPUT_DIR = REPO_ROOT / "data" / "reconstruction"

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
        # qwen3.6-35b is a thinking model; disable thinking so a 5-token "OK"
        # returns deterministically and fast instead of burning the budget on a
        # truncated reasoning trace.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        # Generous read timeout: under fair-share scheduler contention even a
        # trivial request can sit queued for a while before a slot frees up.
        r = _req.post(api_url, json=payload, headers=headers, timeout=(10, 120))
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

def _post_llm_local(url: str, payload: dict, stream: bool = False) -> "requests.Response":
    """
    POST to the MindRouter chat-completions endpoint.

    The retry policy is deliberately conservative for expensive generations:
      - connect failures and transient 5xx are retried a couple of times;
      - 404 is NOT retried — per the MindRouter docs, 4xx are client errors that
        should fast-fail. A 404 here means "no healthy backend currently serves
        this model" (the model-availability hard constraint failed because the
        backends' circuit breakers tripped). Hammering it with retries only keeps
        those circuits hot and turns each failure into a multi-minute hang;
      - POST is never auto-retried on a read timeout, so we never silently re-run
        a half-finished, costly generation.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    retry = Retry(
        total=2, connect=3, read=0, backoff_factor=2.0,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET"]),   # never auto-retry the POST itself
        respect_retry_after_header=True,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://",  HTTPAdapter(max_retries=retry))

    accept = "text/event-stream" if stream else "application/json"
    headers = {"Content-Type": "application/json", "Accept": accept}
    api_key = _load_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return sess.post(url, headers=headers, json=payload,
                     timeout=(10, 1200), stream=stream)


def _raise_for_status_verbose(resp: "requests.Response") -> None:
    """Like resp.raise_for_status(), but include MindRouter's error body.

    MindRouter returns a useful detail object on errors (e.g. {"detail": "..."}
    or an OpenAI-style error object); surfacing it makes failures legible instead
    of a bare 'HTTP 404'.
    """
    if resp.status_code < 400:
        return
    try:
        body = resp.text[:500]
    except Exception:
        body = "<unreadable body>"
    raise RuntimeError(f"HTTP {resp.status_code} from {resp.url} — {body}")


def _accumulate_stream(resp: "requests.Response") -> tuple[str, str]:
    """Consume an OpenAI-compatible SSE stream → (content, reasoning_trace).

    Accumulates choices[0].delta.content and choices[0].delta.reasoning_content
    (the Qwen thinking trace) until 'data: [DONE]' or the stream ends. If the
    stream ends early — e.g. MindRouter's 300s server-side cancellation — we just
    return whatever accumulated; the caller treats truncated output as a parse
    failure rather than an error.
    """
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            content_parts.append(piece)
        rpiece = delta.get("reasoning_content") or delta.get("thinking")
        if rpiece:
            thinking_parts.append(rpiece)
    return "".join(content_parts), "".join(thinking_parts)


# Output budgets must let a single generation finish inside MindRouter's 300s
# server-side request cap (BACKEND_REQUEST_TIMEOUT). Non-streaming requests with
# huge budgets overran it, got cancelled, tripped circuit breakers, and surfaced
# as 404s. These are sized to complete comfortably under that ceiling.
# The model won't reliably keep reasoning short no matter how strict the prompt — a
# few puzzles enumerate grids for ~8-12k+ tokens regardless. A low cap (12288) just
# starves the JSON and fails everything. 32768 gives enough room that the typical
# ~8-15k-token reasoning still leaves space to emit the grids; only the worst runaway
# puzzles (reasoning near/over 32k) fail. Streaming has no time cap, so this is safe.
DEFAULT_MAX_TOKENS_THINKING    = 32768
# No-thinking output is pure JSON: raising the cap is free (generation stops at the
# closing brace), and 4096 truncated the biggest-grid puzzles mid-array.
DEFAULT_MAX_TOKENS_NO_THINKING = 8192

# Thinking-mode addendum to RECOVER_PROMPT. The task is unchanged (reconstruct the
# grid from the image into JSON); this only forces the model to converge — reason,
# then actually emit the complete JSON answer instead of looping on cell-by-cell
# narration until it hits the token cap with no output (the 0/14 failure mode).
THINKING_PROMPT_SUFFIX = """

Keep your reasoning SHORT — a few sentences at most about the transformation rule and \
any colors that are easy to confuse. Do NOT transcribe or list the grids cell-by-cell \
in your reasoning (e.g. "row 0 col 1 is blue, row 0 col 2 is..."): that burns your \
whole token budget and you will run out of room before you answer. Read each grid's \
values directly into the JSON instead. You MUST finish with the complete, valid JSON \
object in the exact format above — getting to the JSON is more important than long \
reasoning."""


def reconstruct_puzzle(
    image_bytes: bytes, model: str, api_url: str,
    thinking: bool = False, max_tokens: int | None = None,
) -> tuple[dict | None, str, str]:
    """
    Send a puzzle image to the VLLM and ask it to reconstruct the JSON.

    Streams the response (SSE) so the connection stays alive through long
    generations and we get partial output if the server cancels at 300s, instead
    of a non-streaming buffer-everything request that overruns the timeout.

    Returns:
        (parsed_dict_or_None, raw_content, reasoning_trace)
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS_THINKING if thinking else DEFAULT_MAX_TOKENS_NO_THINKING
    prompt_text = RECOVER_PROMPT + (THINKING_PROMPT_SUFFIX if thinking else "")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "max_completion_tokens": max_tokens,
    }
    # JSON-object enforcement is incompatible with thinking mode
    if not thinking:
        payload["response_format"] = {"type": "json_object"}

    resp = _post_llm_local(api_url, payload, stream=True)
    _raise_for_status_verbose(resp)
    content, thinking_trace = _accumulate_stream(resp)

    try:
        parsed = _parse_json_lenient(content)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    return parsed, content, thinking_trace


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
    parser.add_argument("--model",    default=QWEN_MODEL, help="Vision model name")
    parser.add_argument("--api-url",  default=API_URL,    help="OpenAI-compatible endpoint")
    parser.add_argument("--thinking", action="store_true", help="Enable chain-of-thought thinking mode")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"Override max output tokens (default: {DEFAULT_MAX_TOKENS_THINKING} thinking / "
                             f"{DEFAULT_MAX_TOKENS_NO_THINKING} otherwise). Lower if responses still time out.")
    args = parser.parse_args()

    output_file = OUTPUT_DIR / ("results_thinking.json" if args.thinking else "results_no_thinking.json")

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
    print(f"  Output: {output_file}")
    print(f"  Thinking: {args.thinking}")
    print("=" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    puzzle_results = []
    total = len(images)

    # Space requests out so we don't saturate the small qwen3.6-35b backend pool
    # and trip per-backend circuit breakers. After a failure, wait out the circuit
    # breaker's recovery window (~BACKEND_CIRCUIT_BREAKER_RECOVERY_SECONDS) so an
    # opened backend can come back instead of cascading into more failures.
    INTER_REQUEST_DELAY      = 3
    CIRCUIT_BREAKER_COOLDOWN = 30

    for idx, img_path in enumerate(images, start=1):
        stem = img_path.stem
        correct = load_correct_puzzle(stem)
        metaphor = correct.get("metaphor", "") if correct else ""

        if idx > 1:
            time.sleep(INTER_REQUEST_DELAY)

        print(f"[{idx}/{total}] {stem} ...", flush=True)

        parsed = None
        content = ""
        thinking = ""
        error = None

        try:
            parsed, content, thinking = reconstruct_puzzle(
                img_path.read_bytes(), args.model, args.api_url,
                thinking=args.thinking, max_tokens=args.max_tokens,
            )
        except Exception as exc:
            error = str(exc)
            print(f"         ERROR: {exc}", file=sys.stderr)
            if idx < total:
                print(f"         cooling down {CIRCUIT_BREAKER_COOLDOWN}s for circuit-breaker recovery ...",
                      flush=True)
                time.sleep(CIRCUIT_BREAKER_COOLDOWN)

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
        "model":            args.model,
        "api_url":          args.api_url,
        "timestamp":        datetime.now().isoformat(),
        "thinking_enabled": args.thinking,
        "diagnostic_passed": True,
        "summary": {
            "total":         total,
            "parse_success": passed,
            "parse_failed":  total - passed,
        },
        "puzzles": puzzle_results,
    }

    output_file.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Done: {passed}/{total} puzzles parsed successfully.")
    print(f"Results saved to: {output_file}")
    suffix = "thinking" if args.thinking else "no_thinking"
    print(f"Run 'uv run code/scripts/make_report.py --input {output_file} --output code/data/reconstruction/report_{suffix}.html' to generate the HTML report.")


if __name__ == "__main__":
    main()
