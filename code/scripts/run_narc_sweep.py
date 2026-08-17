"""
scripts/run_narc_sweep.py — Comprehensive NARC comprehension sweep
==================================================================
Runs the Week 6/8 three-experiment battery over EVERY NARC puzzle × every
selected model, appending one JSON line per experiment to results.jsonl.

    1. blind_solve   image + narrative      -> {"answers": {"<pos>": grid}}
    2. reconstruct   image                  -> {"title", "sequence":[{position,grid}]}
    3. recon_solve   the model's own exp-2 output, text only -> {"answers": {...}}

Experiment 2 is the one that answers "can it actually read the puzzle": it is
scored on the VISIBLE frames only, so it measures perception with no reasoning
mixed in. Experiments 1 and 3 bracket it — solving with the picture, and solving
from the model's own (possibly wrong) transcription of the picture.

This script does no scoring. It records what was sent and what came back;
narc_score.py is the single place metrics are computed, so browser-produced and
script-produced runs are graded by identical code.

Networking is deliberately inherited from run_reconstruction.py rather than
rewritten: Week 4 spent a whole week discovering that non-streaming requests with
large budgets overrun MindRouter's 300s backend cap, get cancelled, trip circuit
breakers, and surface as 404s. Stream, never retry a 404, pace the requests.

Usage:
    uv run code/scripts/run_narc_sweep.py --models qwen/qwen3.6-35b
    uv run code/scripts/run_narc_sweep.py --models qwen/qwen3.6-35b,google/gemma-4-31b
    uv run code/scripts/run_narc_sweep.py --models qwen/qwen3.6-35b --limit 1   # smoke test
    uv run code/scripts/run_narc_sweep.py --models ... --no-narrative           # ablation
"""
import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "utils"))

from run_reconstruction import (  # noqa: E402
    _post_llm_local,
    _accumulate_stream,
    _raise_for_status_verbose,
)
from run_multimodel import _diagnostic  # noqa: E402
from recover_puzzle import _load_api_key  # noqa: E402
from narc_prompts import (  # noqa: E402
    NARC_RECOVER_PROMPT,
    compact_json,
    narc_answer_dims,
    narc_masked_positions,
    narc_solve_prompt_image,
    narc_solve_prompt_text,
    narc_visible_frames,
)
from narc_extract import (  # noqa: E402
    extract_answers_map,
    extract_sequence_map,
    parse_json_lenient,
)

API_URL = "https://mindrouter.uidaho.edu/v1/chat/completions"

REPO_ROOT = Path(__file__).resolve().parents[1]      # code/
PUZZLE_DIR = REPO_ROOT / "narc_puzzles"
IMAGE_DIR = REPO_ROOT / "narc_images"
DATA_DIR = REPO_ROOT / "data" / "narc_baseline"
DEFAULT_OUT = DATA_DIR / "results.jsonl"

SCHEMA_VERSION = 2
RENDERER = "python-v1"

# Same shape as run_reconstruction.py: thinking needs headroom for the reasoning
# trace, no-think output is pure JSON so a generous cap costs nothing.
MAX_TOKENS_THINKING = 32768
MAX_TOKENS_NO_THINKING = 8192

# Pacing — matches run_multimodel.py. Space requests so a small backend pool
# doesn't trip its circuit breakers, and wait out the recovery window on failure.
INTER_REQUEST_DELAY = 3
CIRCUIT_BREAKER_COOLDOWN = 30

EXPERIMENTS = ("blind_solve", "reconstruct", "recon_solve")


def supports_thinking(model: str) -> bool:
    """Only the Qwen family accepts chat_template_kwargs.enable_thinking; gemma
    and friends can reject the unknown kwarg outright."""
    return "qwen" in (model or "").lower()


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------

def call_model(model: str, content, api_url: str, thinking: bool,
               json_mode: bool, max_tokens: int) -> dict:
    """POST one streamed completion. Returns a dict of everything worth logging."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "stream": True,
        "max_completion_tokens": max_tokens,
    }
    if supports_thinking(model):
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    # Guided decoding is model-agnostic but incompatible with a thinking trace.
    if not thinking and json_mode:
        payload["response_format"] = {"type": "json_object"}

    started = time.monotonic()
    resp = _post_llm_local(api_url, payload, stream=True)
    _raise_for_status_verbose(resp)
    text, reasoning = _accumulate_stream(resp)
    return {
        "raw": text,
        "reasoning": reasoning,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "content_chars": len(text),
        "reasoning_chars": len(reasoning),
        # An empty answer after a long trace is the Week 4 runaway-reasoning
        # failure, not a refusal — worth distinguishing in the dashboard.
        "truncated": bool(reasoning) and not text.strip(),
    }


def image_content(prompt: str, image_bytes: bytes) -> list:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def run_key(puzzle_id, model, thinking, narrative_on, sizes_on, experiment) -> str:
    """Resume identity. Same string the browser tool computes, so a JSONL half
    filled by one runner can be finished by the other."""
    return "|".join([
        str(puzzle_id), str(model),
        "think" if thinking else "nothink",
        "narr" if narrative_on else "nonarr",
        "sizes" if sizes_on else "nosizes",
        str(experiment),
    ])


def base_record(puzzle, puzzle_file, model, cfg, experiment, image_name) -> dict:
    masked = narc_masked_positions(puzzle)
    visible = narc_visible_frames(puzzle)
    puzzle_id = puzzle.get("puzzle_id") or Path(puzzle_file).stem
    return {
        "schema_version": SCHEMA_VERSION,
        "run_key": run_key(puzzle_id, model, cfg["thinking"],
                           cfg["include_narrative"], cfg["include_sizes"], experiment),
        "timestamp": datetime.now().isoformat(),
        "source": "python-sweep",
        "renderer": RENDERER,
        "image_source": "folder",
        "image_name": image_name,
        "puzzle_id": puzzle_id,
        "puzzle_file": puzzle_file,
        "title": puzzle.get("title") or "",
        "narrative": puzzle.get("narrative") or "",
        "model": model,
        "api_url": cfg["api_url"],
        "thinking": cfg["thinking"],
        "include_narrative": cfg["include_narrative"],
        "include_sizes": cfg["include_sizes"],
        "experiment": experiment,
        # Ground truth travels with the record so a line is scorable on its own.
        "truth": {
            "masked_positions": masked,
            "answer_grids": {str(p): (puzzle.get("answer_grids") or {}).get(str(p))
                             for p in masked},
            "visible_frames": [{"position": int(f["position"]), "grid": f["grid"]}
                               for f in visible],
        },
        "prompt": "",
        "raw": "",
        "reasoning": "",
        "parsed": False,
        "parse_needed_repair": False,
        "predicted": None,
        "latency_ms": None,
        "content_chars": 0,
        "reasoning_chars": 0,
        "truncated": False,
        "error": None,
        "skipped": False,
    }


def append_record(out_path: Path, record: dict) -> None:
    """Append and flush immediately — a crash at hour six loses one record."""
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def load_prior(out_path: Path) -> tuple[set[str], dict[str, dict]]:
    """Scan an existing results file for resume state.

    Returns (done_keys, reconstructions). Errored records are deliberately NOT
    counted as done, so a resume retries them.

    `reconstructions` maps a reconstruct run_key to its predicted frames. Without
    this, resuming a run whose experiment 2 already succeeded would have nothing
    to feed experiment 3 and would re-skip it forever. Only the prediction is
    kept — raw text and reasoning traces are left on disk.
    """
    done: set[str] = set()
    recons: dict[str, dict] = {}
    if not out_path.exists():
        return done, recons
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("run_key")
            if not key or rec.get("error"):
                continue
            done.add(key)
            if rec.get("experiment") == "reconstruct" and isinstance(rec.get("predicted"), dict):
                recons[key] = rec["predicted"]
    return done, recons


# ---------------------------------------------------------------------------
# The three experiments for one (puzzle, model)
# ---------------------------------------------------------------------------

def run_puzzle(puzzle, puzzle_file, image_bytes, image_name, model, cfg,
               out_path: Path, done: set[str], recons: dict[str, dict]) -> dict[str, str]:
    """Run whichever of the three experiments aren't already recorded.

    Returns {experiment: status} for the console summary.
    """
    masked = narc_masked_positions(puzzle)
    all_dims = narc_answer_dims(puzzle, masked)
    # What the model is actually told, per the ablation switches.
    prompt_narrative = (puzzle.get("narrative") or "") if cfg["include_narrative"] else ""
    prompt_dims = all_dims if cfg["include_sizes"] else None
    max_tokens = MAX_TOKENS_THINKING if cfg["thinking"] else MAX_TOKENS_NO_THINKING

    statuses: dict[str, str] = {}
    reconstruction_map: dict[str, list] | None = None
    recon_title = ""

    def emit(experiment: str, prompt: str, content, extra=None) -> dict | None:
        """Send one experiment, record it, return the parsed object."""
        rec = base_record(puzzle, puzzle_file, model, cfg, experiment, image_name)
        rec["prompt"] = prompt
        if extra:
            rec.update(extra)
        try:
            result = call_model(model, content, cfg["api_url"], cfg["thinking"],
                                json_mode=True, max_tokens=max_tokens)
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = str(exc)
            append_record(out_path, rec)
            statuses[experiment] = "ERROR"
            print(f"        {experiment}: ERROR {exc}", file=sys.stderr)
            print(f"        cooling down {CIRCUIT_BREAKER_COOLDOWN}s for circuit-breaker recovery ...",
                  flush=True)
            time.sleep(CIRCUIT_BREAKER_COOLDOWN)
            return None

        rec.update(result)
        parsed, repaired = parse_json_lenient(result["raw"])
        rec["parsed"] = parsed is not None
        rec["parse_needed_repair"] = repaired

        if experiment == "reconstruct":
            rec["predicted"] = {
                "title": (parsed or {}).get("title", "") if isinstance(parsed, dict) else "",
                "frames": extract_sequence_map(parsed),
            }
            n = len(rec["predicted"]["frames"])
            statuses[experiment] = f"{n} frame(s)" if parsed is not None else "unparsed"
        else:
            answers = extract_answers_map(parsed, masked)
            rec["predicted"] = {"answers": answers}
            statuses[experiment] = (f"{len(answers)}/{len(masked)} frame(s)"
                                    if parsed is not None else "unparsed")

        append_record(out_path, rec)
        print(f"        {experiment}: {statuses[experiment]}"
              f"  ({result['latency_ms']}ms, {result['content_chars']} chars"
              f"{', ' + str(result['reasoning_chars']) + ' reasoning' if result['reasoning_chars'] else ''})",
              flush=True)
        return parsed

    def key_for(experiment: str) -> str:
        return run_key(puzzle.get("puzzle_id") or Path(puzzle_file).stem, model,
                       cfg["thinking"], cfg["include_narrative"], cfg["include_sizes"],
                       experiment)

    def want(experiment: str) -> bool:
        if key_for(experiment) in done:
            statuses[experiment] = "skip (done)"
            return False
        return True

    # ── 1 · blind solve ────────────────────────────────────────────────────
    if want("blind_solve"):
        prompt = narc_solve_prompt_image(prompt_narrative, masked, prompt_dims)
        emit("blind_solve", prompt, image_content(prompt, image_bytes))
        time.sleep(INTER_REQUEST_DELAY)

    # ── 2 · reconstruct ────────────────────────────────────────────────────
    if want("reconstruct"):
        parsed = emit("reconstruct", NARC_RECOVER_PROMPT,
                      image_content(NARC_RECOVER_PROMPT, image_bytes))
        if parsed is not None:
            reconstruction_map = extract_sequence_map(parsed)
            recon_title = parsed.get("title", "") if isinstance(parsed, dict) else ""
        time.sleep(INTER_REQUEST_DELAY)
    else:
        # Already recorded on an earlier pass — reuse its prediction rather than
        # re-billing the reconstruction just to unblock experiment 3.
        prior = recons.get(key_for("reconstruct"))
        if prior:
            reconstruction_map = prior.get("frames") or None
            recon_title = prior.get("title", "")

    # ── 3 · reconstruct-then-solve ─────────────────────────────────────────
    if want("recon_solve"):
        if not reconstruction_map:
            # Experiment 2 produced nothing parseable, so there is no self-made
            # transcription to solve from. Recorded as skipped, not failed —
            # this is a downstream consequence, not an independent error.
            rec = base_record(puzzle, puzzle_file, model, cfg, "recon_solve", image_name)
            rec["skipped"] = True
            rec["error"] = "reconstruction produced no parseable sequence JSON"
            append_record(out_path, rec)
            statuses["recon_solve"] = "skipped"
        else:
            masked_set = {int(p) for p in masked}
            visible = sorted(
                ({"position": int(p), "grid": g}
                 for p, g in reconstruction_map.items()
                 if int(p) not in masked_set),          # never feed back a hallucinated answer
                key=lambda f: f["position"],
            )
            # compact_json, not json.dumps — this string goes INTO the prompt, so
            # it has to match JSON.stringify's spacing in the browser tool.
            seq_json = compact_json({"title": recon_title or puzzle.get("title") or "",
                                     "sequence": visible})
            prompt = narc_solve_prompt_text(seq_json, prompt_narrative, masked, prompt_dims)
            emit("recon_solve", prompt, prompt, extra={"solved_from_json": seq_json})

    return statuses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_puzzles(puzzle_dir: Path, image_dir: Path, limit: int | None,
                 only: str | None) -> list[tuple[dict, str, Path]]:
    """(puzzle, filename, image_path) for every puzzle with a rendered image."""
    wanted = {s.strip() for s in only.split(",")} if only else None
    out, missing = [], []
    for path in sorted(puzzle_dir.glob("*.json")):
        try:
            puzzle = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        puzzle_id = puzzle.get("puzzle_id") or path.stem
        if wanted and puzzle_id not in wanted and path.stem not in wanted:
            continue
        image = image_dir / f"{puzzle_id}.png"
        if not image.exists():
            missing.append(puzzle_id)
            continue
        out.append((puzzle, path.name, image))
    if missing:
        print(f"Warning: no rendered image for {len(missing)} puzzle(s): "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}\n"
              f"         Run: uv run code/scripts/narc_prepare.py", file=sys.stderr)
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive NARC comprehension sweep")
    ap.add_argument("--models", required=True,
                    help="Comma-separated model ids, e.g. qwen/qwen3.6-35b,google/gemma-4-31b")
    ap.add_argument("--puzzle-dir", type=Path, default=PUZZLE_DIR)
    ap.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable chain-of-thought (Qwen only; Week 4 found it hurts)")
    ap.add_argument("--no-narrative", action="store_true",
                    help="Ablation: solve from the frames alone, no story text")
    ap.add_argument("--no-sizes", action="store_true",
                    help="Ablation: don't tell the model each masked frame's rows x cols")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N puzzles")
    ap.add_argument("--puzzles", default=None, help="Comma-separated puzzle ids to run")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-run everything, even experiments already in the output file")
    ap.add_argument("--skip-diagnostic", action="store_true",
                    help="Skip the per-model reachability/multimodal ping")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: --models is empty", file=sys.stderr)
        sys.exit(1)

    puzzles = load_puzzles(args.puzzle_dir, args.image_dir, args.limit, args.puzzles)
    if not puzzles:
        print(f"Error: no puzzles with rendered images in {args.puzzle_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = {
        "api_url": args.api_url,
        "thinking": args.thinking,
        "include_narrative": not args.no_narrative,
        "include_sizes": not args.no_sizes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, recons = (set(), {}) if args.no_resume else load_prior(args.out)

    total = len(puzzles) * len(models) * len(EXPERIMENTS)
    print("=" * 68)
    print("NARC comprehension sweep")
    print(f"  puzzles:   {len(puzzles)}")
    print(f"  models:    {', '.join(models)}")
    print(f"  condition: thinking={cfg['thinking']}  narrative={cfg['include_narrative']}"
          f"  sizes={cfg['include_sizes']}")
    print(f"  output:    {args.out}")
    print(f"  planned:   {total} experiment(s), {len(done)} already recorded")
    if not _load_api_key():
        print("  WARNING:   no Mindrouter_api key found in code/keys.txt")
    print("=" * 68)

    skipped_models = []
    for model in models:
        if not args.skip_diagnostic:
            print(f"\n### Diagnostic: {model} ###", flush=True)
            ok, msg = _diagnostic(model, supports_thinking(model), puzzles[0][2], args.api_url)
            print(f"  [{'OK' if ok else 'SKIP'}] {msg}")
            if not ok:
                skipped_models.append((model, msg))
                continue

        print(f"\n--- {model} ---", flush=True)
        for i, (puzzle, filename, image_path) in enumerate(puzzles, start=1):
            puzzle_id = puzzle.get("puzzle_id") or Path(filename).stem
            print(f"  [{i}/{len(puzzles)}] {puzzle_id} ({puzzle.get('title') or filename})", flush=True)
            statuses = run_puzzle(puzzle, filename, image_path.read_bytes(), image_path.name,
                                  model, cfg, args.out, done, recons)
            if all(s.startswith("skip") for s in statuses.values()):
                print("        all three already recorded — skipped")
            if i < len(puzzles):
                time.sleep(INTER_REQUEST_DELAY)

    print("\n" + "=" * 68)
    print("SWEEP COMPLETE")
    print(f"  results: {args.out}")
    if skipped_models:
        print("  models skipped (diagnostic failed):")
        for name, reason in skipped_models:
            print(f"    {name}: {reason}")
    print("\nNext: uv run code/scripts/narc_score.py --publish")
    print("=" * 68)


if __name__ == "__main__":
    main()
