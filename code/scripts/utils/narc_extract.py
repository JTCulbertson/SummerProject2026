"""
scripts/utils/narc_extract.py — pulling structure out of model responses
========================================================================
Models return the right answer in the wrong wrapper constantly: fenced in
markdown, keyed "frame 3" instead of "3", a bare grid when only one frame was
asked for. These helpers absorb that so the difference between "the model got it
wrong" and "the model phrased it differently" never lands in the metrics.

Ported from code/tools/model-baseline-analysis.html (parseJsonLenient line 868,
extractOutputGrid line 881, extractAnswersMap line 894) so the browser tool and
the Python sweep agree on what a response contained.

Shared by run_narc_sweep.py (which needs experiment 2's reconstruction to feed
experiment 3) and narc_score.py (which does all the actual scoring).
"""
import json
import re

FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def is_grid(g) -> bool:
    return isinstance(g, list) and len(g) > 0 and isinstance(g[0], list)


def parse_json_lenient(text: str) -> tuple[object | None, bool]:
    """Return (parsed_or_None, needed_repair).

    `needed_repair` is True when a plain json.loads() of the response failed and
    a fallback rescued it — worth tracking separately, because a model that
    always needs rescuing is failing the "respond with ONLY a JSON object"
    instruction even when its grids are right.
    """
    if not text:
        return None, False
    s = text.strip()

    try:
        return json.loads(s), False
    except (json.JSONDecodeError, ValueError):
        pass

    fence = FENCE_RE.search(s)
    if fence:
        try:
            return json.loads(fence.group(1).strip()), True
        except (json.JSONDecodeError, ValueError):
            pass

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1]), True
        except (json.JSONDecodeError, ValueError):
            pass

    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1]), True
        except (json.JSONDecodeError, ValueError):
            pass

    return None, False


def extract_output_grid(parsed):
    """A single grid out of whatever shape came back."""
    if parsed is None:
        return None
    if is_grid(parsed):
        return parsed
    if not isinstance(parsed, dict):
        return None
    for key in ("output", "answer", "grid"):
        if is_grid(parsed.get(key)):
            return parsed[key]
    test = parsed.get("test")
    if isinstance(test, list) and test and isinstance(test[0], dict) and is_grid(test[0].get("output")):
        return test[0]["output"]
    return None


def extract_answers_map(parsed, masked_positions) -> dict[str, list]:
    """{position -> grid} out of a solve response.

    Tolerates {"answers": {...}}, a top-level position-keyed object, and — when
    exactly one frame is masked — a bare grid or {"output"/"grid": ...}.
    """
    out: dict[str, list] = {}
    if parsed is None:
        return out
    want = [str(p) for p in (masked_positions or [])]

    def take_from(obj):
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            key = re.sub(r"[^0-9-]", "", str(k))     # "frame 3" / "position_3" -> "3"
            if not re.search(r"\d", key):
                continue                              # skip "output", "grid", ...
            if is_grid(v):
                out[key] = v
            elif isinstance(v, dict):
                for inner in ("grid", "output"):
                    if is_grid(v.get(inner)):
                        out[key] = v[inner]
                        break

    if isinstance(parsed, dict) and isinstance(parsed.get("answers"), dict):
        take_from(parsed["answers"])
    if not out:
        take_from(parsed)
    if not out and len(want) == 1:
        g = extract_output_grid(parsed)
        if g:
            out[want[0]] = g
    return out


def extract_sequence_map(parsed) -> dict[str, list]:
    """{position -> grid} out of a reconstruction response.

    Accepts the requested {"sequence":[{"position","grid"}]} and also a bare
    list of frames or a position-keyed object, so a well-read puzzle in an
    off-spec wrapper still scores as well-read.
    """
    out: dict[str, list] = {}
    frames = None
    if isinstance(parsed, dict):
        for key in ("sequence", "frames", "grids"):
            if isinstance(parsed.get(key), list):
                frames = parsed[key]
                break
    elif isinstance(parsed, list):
        frames = parsed

    if frames is not None:
        for i, f in enumerate(frames):
            if is_grid(f):                             # bare grid, position implied by order
                out[str(i)] = f
            elif isinstance(f, dict):
                grid = f.get("grid") if is_grid(f.get("grid")) else None
                if grid is None:
                    continue
                pos = f.get("position", f.get("index", i))
                try:
                    out[str(int(pos))] = grid
                except (TypeError, ValueError):
                    out[str(i)] = grid
        return out

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            key = re.sub(r"[^0-9-]", "", str(k))
            if re.search(r"\d", key) and is_grid(v):
                out[key] = v
    return out
