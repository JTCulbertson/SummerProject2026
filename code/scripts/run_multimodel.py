"""
scripts/run_multimodel.py — Cross-Model MARC Reconstruction Sweep (Week 5)
=========================================================================
Week 3-4 showed that qwen/qwen3.6-35b reconstructs MARC puzzle *structure*
perfectly but misreads *colors*, and that thinking mode doesn't help. The open
question: is that a Qwen quirk, or do vLLM vision models in general struggle to
recreate these puzzles?

This runs the same image -> JSON reconstruction across several vision models as a
PRELIMINARY check (only a few puzzles each, not the full 14):

    Model            Thinking toggle?   Runs
    gemma-4-31b      no                 no-think only (N tests)
    qwen3.5-122b     yes                think + no-think (2N tests)
    qwen3-vl:32b     yes                think + no-think (2N tests)

Per-model request shaping (the "called differently" concern): only Qwen models
accept chat_template_kwargs.enable_thinking. We add that kwarg ONLY for
thinking-capable models; other models (gemma) get a plain request and run
no-think only. All no-think runs keep response_format=json_object (backend-level
guided decoding, model-agnostic); thinking runs omit it.

Each (model, mode) pair is written to its own results JSON under
code/data/multimodel/, in the same schema make_report.py already consumes.

Usage:
    uv run code/scripts/run_multimodel.py
    uv run code/scripts/run_multimodel.py --models qwen3-vl:32b --num-puzzles 1   # smoke test
    uv run code/scripts/run_multimodel.py --puzzles vote_according_to_your_party,debris_protects_little_fish
"""
import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Reuse the streaming / retry / parse machinery from the Week 4 runner instead
# of reimplementing it. Importing run_reconstruction also puts the utils/ dir on
# sys.path (it does so at import time), so recover_puzzle is importable too.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))

from run_reconstruction import (  # noqa: E402
    _post_llm_local,
    _accumulate_stream,
    _raise_for_status_verbose,
    load_correct_puzzle,
    RECOVER_PROMPT,
    THINKING_PROMPT_SUFFIX,
    DEFAULT_MAX_TOKENS_THINKING,
    DEFAULT_MAX_TOKENS_NO_THINKING,
)
from recover_puzzle import _load_api_key, _parse_json_lenient  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "https://mindrouter.uidaho.edu/v1/chat/completions"

# Vision-capable models to compare against the already-tested qwen3.6-35b.
# supports_thinking == True only for models that accept
# chat_template_kwargs.enable_thinking (Qwen family). Edit this list to swap
# models if an id turns out to be unavailable on the cluster.
MODELS = [
    {"name": "google/gemma-4-31b", "supports_thinking": False},
    {"name": "qwen/qwen3.5-122b",  "supports_thinking": True},
    {"name": "qwen3-vl:32b",       "supports_thinking": True},
]

REPO_ROOT  = Path(__file__).resolve().parents[1]   # code/
IMAGE_DIR  = REPO_ROOT / "images"
PUZZLE_DIR = REPO_ROOT / "puzzles"
OUTPUT_DIR = REPO_ROOT / "data" / "multimodel"

# Pacing — same rationale as run_reconstruction.py: space requests so we don't
# saturate a small backend pool, and wait out the circuit-breaker recovery
# window after any failure.
INTER_REQUEST_DELAY      = 3
CIRCUIT_BREAKER_COOLDOWN = 30


def model_slug(name: str) -> str:
    """Filesystem-safe slug: qwen3-vl:32b -> qwen3-vl-32b, qwen3.5 -> qwen3-5."""
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Reconstruction — per-model payload shaping
# ---------------------------------------------------------------------------

def reconstruct_multi(
    image_bytes: bytes, model: str, api_url: str,
    thinking: bool, supports_thinking: bool, max_tokens: int | None = None,
) -> tuple[dict | None, str, str]:
    """
    Send a puzzle image to a vision model and ask it to reconstruct the JSON.

    Differs from run_reconstruction.reconstruct_puzzle() in one way: the
    chat_template_kwargs.enable_thinking flag is only sent to models that
    actually support it. A model like gemma can reject an unknown kwarg, so for
    non-thinking models we send a plain request.

    Returns (parsed_dict_or_None, raw_content, reasoning_trace).
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
        "max_completion_tokens": max_tokens,
    }
    # Only Qwen-style models understand this kwarg. For everyone else, omit it.
    if supports_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    # JSON-object enforcement is incompatible with thinking mode but is a
    # backend-level constraint that works regardless of model family otherwise.
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


# ---------------------------------------------------------------------------
# Per-model diagnostic — confirm the model is reachable AND multimodal before we
# spend a full sweep on it. Omits chat_template_kwargs for non-thinking models.
# ---------------------------------------------------------------------------

def _diagnostic(model: str, supports_thinking: bool, image_path: Path, api_url: str) -> tuple[bool, str]:
    import requests as _req

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _load_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def _kwargs():
        # For thinking-capable models, disable thinking on the ping so a tiny
        # response returns fast instead of burning the budget on a reasoning
        # trace. For others, send nothing extra.
        return {"chat_template_kwargs": {"enable_thinking": False}} if supports_thinking else {}

    # 1) text-only reachability ping
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
            "stream": False,
            **_kwargs(),
        }
        r = _req.post(api_url, json=payload, headers=headers, timeout=(10, 120))
        r.raise_for_status()
    except Exception as exc:
        return False, f"text ping failed: {exc}"

    # 2) multimodal ping — confirms the model can actually accept an image
    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
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
            **_kwargs(),
        }
        r = _req.post(api_url, json=payload, headers=headers, timeout=(10, 120))
        r.raise_for_status()
    except Exception as exc:
        return False, f"image ping failed (model may be text-only or unavailable): {exc}"

    return True, "reachable + multimodal"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_one_config(
    model: str, supports_thinking: bool, thinking: bool,
    images: list[Path], api_url: str, max_tokens: int | None,
) -> dict:
    """Run all selected puzzles for one (model, mode) pair; return results dict."""
    mode = "thinking" if thinking else "no_thinking"
    print(f"\n--- {model}  [{mode}]  ({len(images)} puzzles) ---", flush=True)

    puzzle_results = []
    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        stem = img_path.stem
        correct = load_correct_puzzle(stem)
        metaphor = correct.get("metaphor", "") if correct else ""

        if idx > 1:
            time.sleep(INTER_REQUEST_DELAY)

        print(f"  [{idx}/{total}] {stem} ...", flush=True)

        parsed = None
        content = ""
        trace = ""
        error = None
        try:
            parsed, content, trace = reconstruct_multi(
                img_path.read_bytes(), model, api_url,
                thinking=thinking, supports_thinking=supports_thinking,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            error = str(exc)
            print(f"        ERROR: {exc}", file=sys.stderr)
            if idx < total:
                print(f"        cooling down {CIRCUIT_BREAKER_COOLDOWN}s for circuit-breaker recovery ...",
                      flush=True)
                time.sleep(CIRCUIT_BREAKER_COOLDOWN)

        print(f"        {'PASS' if parsed is not None else 'FAIL'} (parse_success={parsed is not None})")
        puzzle_results.append({
            "name":            stem,
            "metaphor":        metaphor,
            "model_output":    parsed,
            "correct_output":  correct,
            "reasoning_trace": trace,
            "raw_response":    content,
            "parse_success":   parsed is not None,
            "error":           error,
        })

    passed = sum(1 for p in puzzle_results if p["parse_success"])
    return {
        "model":             model,
        "api_url":           api_url,
        "timestamp":         datetime.now().isoformat(),
        "thinking_enabled":  thinking,
        "supports_thinking": supports_thinking,
        "diagnostic_passed": True,
        "summary": {
            "total":         total,
            "parse_success": passed,
            "parse_failed":  total - passed,
        },
        "puzzles": puzzle_results,
    }


def select_images(num_puzzles: int, puzzle_csv: str | None) -> list[Path]:
    all_images = sorted(IMAGE_DIR.glob("*.png"))
    if not all_images:
        print(f"Error: no PNG files found in {IMAGE_DIR}", file=sys.stderr)
        sys.exit(1)
    if puzzle_csv:
        wanted = [s.strip() for s in puzzle_csv.split(",") if s.strip()]
        by_stem = {p.stem: p for p in all_images}
        missing = [w for w in wanted if w not in by_stem]
        if missing:
            print(f"Error: puzzle stems not found as images: {missing}", file=sys.stderr)
            sys.exit(1)
        return [by_stem[w] for w in wanted]
    return all_images[:num_puzzles]


def main():
    parser = argparse.ArgumentParser(description="Cross-model MARC reconstruction sweep")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--num-puzzles", type=int, default=3,
                        help="How many puzzles per model/mode (default 3, first N alphabetically)")
    parser.add_argument("--puzzles", default=None,
                        help="Comma-separated puzzle stems to use instead of the first N")
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of model names to run (default: all configured)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max output tokens (default: per-mode 32768 thinking / 8192 no-think)")
    parser.add_argument("--skip-thinking", action="store_true",
                        help="Run no-thinking only, even for thinking-capable models (fast / time-boxed runs)")
    args = parser.parse_args()

    images = select_images(args.num_puzzles, args.puzzles)

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in MODELS if m["name"] in wanted]
        unknown = wanted - {m["name"] for m in MODELS}
        if unknown:
            print(f"Warning: ignoring unconfigured models: {sorted(unknown)}", file=sys.stderr)
        if not models:
            print("Error: no configured models matched --models", file=sys.stderr)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Week 5 cross-model MARC reconstruction sweep")
    print(f"  API:      {args.api_url}")
    print(f"  Puzzles:  {[p.stem for p in images]}")
    print(f"  Models:   {[m['name'] for m in models]}")
    print("=" * 64)

    written = []   # (model, mode, output_path)
    skipped = []   # (model, reason)

    for m in models:
        name, supports = m["name"], m["supports_thinking"]
        print(f"\n### Diagnostic: {name} (thinking supported: {supports}) ###")
        ok, msg = _diagnostic(name, supports, images[0], args.api_url)
        print(f"  [{'OK' if ok else 'SKIP'}] {msg}")
        if not ok:
            skipped.append((name, msg))
            continue

        # thinking-capable models get both modes; others get no-think only.
        # --skip-thinking forces no-think everywhere (fast, time-boxed runs).
        modes = [False, True] if (supports and not args.skip_thinking) else [False]
        for thinking in modes:
            results = run_one_config(name, supports, thinking, images, args.api_url, args.max_tokens)
            tag = "think" if thinking else "nothink"
            out = OUTPUT_DIR / f"{model_slug(name)}__{tag}.json"
            out.write_text(json.dumps(results, indent=2))
            written.append((name, tag, out))
            print(f"  saved -> {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SWEEP COMPLETE")
    print(f"  result sets written: {len(written)}")
    for name, tag, out in written:
        data = json.loads(out.read_text())
        s = data["summary"]
        print(f"    {name:16s} [{tag:7s}]  parsed {s['parse_success']}/{s['total']}  -> {out.name}")
    if skipped:
        print("  skipped models (diagnostic failed):")
        for name, reason in skipped:
            print(f"    {name}: {reason}")
    print("\nNext: uv run code/scripts/make_multimodel_report.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
