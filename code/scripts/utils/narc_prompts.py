"""
scripts/utils/narc_prompts.py — NARC prompt text, shared by every runner
========================================================================
A NARC puzzle is a *narrative sequence* of ARC-style grids. Unlike MARC (one
masked test output derived from train pairs), NARC:
  • is an ordered sequence — the left-to-right order of frames is the story,
  • may have MORE THAN ONE masked frame, each identified by its position index,
  • ships a natural-language narrative that the sequence depicts.

These strings are the single source of truth for NARC prompting. They are
byte-identical to the constants in code/tools/model-baseline-analysis.html
(narcSolvePromptImage / NARC_RECOVER_PROMPT / narcSolvePromptText) so that a run
from the browser tool and a run from run_narc_sweep.py are directly comparable.
If you change one, change the other — the tool carries a pointer comment back
to this file.

Two ablation switches are threaded through the solve prompts (Week 8):
  • narrative — pass "" / None to omit the story, isolating pixels from language
  • dims      — pass None to omit the masked-frame grid sizes, putting NARC on
                the same footing as MARC blind-solve (which never gets a hint)
"""
import json


def compact_json(obj) -> str:
    """json.dumps with JavaScript's separators.

    Python defaults to ', ' / ': ' while JSON.stringify emits ',' / ':'. Left
    alone, that one space means the browser tool and this module send materially
    different prompt text for the same puzzle — the sort of drift that quietly
    makes two halves of a sweep incomparable. Every JSON literal that goes INTO a
    prompt must go through here.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Palette — identical to generate_images.py, narc_visualizer.html,
# greyscale-lab.html and model-baseline-analysis.html.
# ---------------------------------------------------------------------------
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#ADD8E6", "#870C25",
]

ARC_COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow",
    "gray", "magenta", "orange", "lt.blue", "maroon",
]


# ---------------------------------------------------------------------------
# Experiment 1 — blind solve, from the sequence image
# ---------------------------------------------------------------------------

def narc_solve_prompt_image(narrative: str | None,
                            masked_positions: list,
                            dims: dict | None) -> str:
    """Image-grounded solve prompt. Mirrors narcSolvePromptImage() in the tool."""
    parts = [
        "You are looking at a NARC puzzle — a NARRATIVE SEQUENCE of ARC-style grids "
        "rendered as an image, laid out left-to-right in order. The ORDER of the frames "
        "is the story: each grid is one step that follows from the previous ones. This is "
        "different from a standard ARC/MARC puzzle — there are no train→output pairs to "
        "generalise; instead you reason about how the scene evolves frame by frame.",
        'One or more frames are MASKED (drawn as "?"). Each masked frame is identified by '
        'its position index (shown in its label, e.g. "Frame 3"). You must produce the full '
        "grid for EVERY masked frame.",
    ]
    if narrative:
        parts.append(f'The sequence depicts this narrative:\n"""\n{narrative}\n"""')

    meta = [f"Masked positions you must fill in: {compact_json(masked_positions)}"]
    if dims:
        meta.append(f"Grid size (rows × cols) for each masked position: {compact_json(dims)}")
    parts.append("\n".join(meta))

    lead = ("Study the ordered frames together with the narrative" if narrative
            else "Study the ordered frames")
    size_clause = (" Produce a grid of exactly the stated size for each masked position."
                   if dims else "")
    parts.append(
        f"{lead}, work out what happens at each masked step, and produce its grid. "
        f"Respect the sequence order.{size_clause}\n"
        "Grids are 2D arrays of integers 0-9 (each integer is a color)."
    )
    parts.append(
        "Respond with ONLY a JSON object mapping each masked position (as a string key) "
        'to its grid — no markdown, no explanation:\n{"answers": {"<position>": [[...], ...]}}'
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Experiment 2 — reconstruct the visible frames
# ---------------------------------------------------------------------------

NARC_RECOVER_PROMPT = """\
You are looking at a NARC puzzle — a NARRATIVE SEQUENCE of ARC-style grids rendered as an image, laid out left-to-right in order. Each frame has a position index shown in its label.

Reconstruct the JSON of every VISIBLE frame exactly. Skip masked frames (drawn as "?") — do NOT invent them. Preserve the sequence order via each frame's position index.
Grids are 2D arrays of integers 0-9.

Respond with ONLY a JSON object in this exact format — no markdown, no explanation:
{
  "title": "<title string, or empty string if none>",
  "sequence": [{"position": <int>, "grid": [[...], ...]}, ...]
}"""


# ---------------------------------------------------------------------------
# Experiment 3 — solve from the model's own reconstruction (text only, no image)
# ---------------------------------------------------------------------------

def narc_solve_prompt_text(seq_json_str: str,
                           narrative: str | None,
                           masked_positions: list,
                           dims: dict | None) -> str:
    """Text-only solve prompt. Mirrors narcSolvePromptText() in the tool."""
    parts = [
        "Here is a NARC puzzle as JSON — a NARRATIVE SEQUENCE of ARC-style grids. The "
        'frames are ordered by their "position" index and the order is the story: each grid '
        "is one step that follows from the previous ones. One or more positions are MASKED "
        "and omitted from the sequence below; you must produce their grids.",
    ]
    if narrative:
        parts.append(f'The sequence depicts this narrative:\n"""\n{narrative}\n"""')
    parts.append(f"Visible frames (JSON, in order):\n{seq_json_str}")

    meta = [f"Masked positions you must fill in: {compact_json(masked_positions)}"]
    if dims:
        meta.append(f"Grid size (rows × cols) for each masked position: {compact_json(dims)}")
    parts.append("\n".join(meta))

    lead = ("Using the narrative and the ordered visible frames" if narrative
            else "Using the ordered visible frames")
    size_clause = " (exactly the stated size)" if dims else ""
    parts.append(
        f"{lead}, infer what happens at each masked step and produce its full grid"
        f"{size_clause}.\nGrids are 2D arrays of integers 0-9."
    )
    parts.append(
        "Respond with ONLY a JSON object mapping each masked position (as a string key) "
        'to its grid — no markdown, no explanation:\n{"answers": {"<position>": [[...], ...]}}'
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Puzzle-shape helpers — the tool's rules, ported. masked_positions is
# authoritative: some exported files leave the answer grid inline on the masked
# frame (stump.json, tortoise.json) while others null it out (sub_005), so
# nothing may infer "visible" from the presence of a grid.
# ---------------------------------------------------------------------------

def narc_masked_positions(puzzle: dict) -> list[int]:
    """Masked frame positions. Mirrors narcMaskedPositions() in the tool."""
    explicit = puzzle.get("masked_positions")
    if isinstance(explicit, list) and explicit:
        return [int(p) for p in explicit]
    return [int(f.get("position"))
            for f in (puzzle.get("sequence") or [])
            if f and (f.get("masked") or f.get("grid") is None)]


def narc_answer_dims(puzzle: dict, masked_positions: list[int]) -> dict[str, str]:
    """{"3": "5x5"} size hints, read from answer_grids (the ground truth)."""
    answers = puzzle.get("answer_grids") or {}
    dims = {}
    for pos in masked_positions:
        grid = answers.get(str(pos))
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            dims[str(pos)] = f"{len(grid)}x{len(grid[0])}"
    return dims


def narc_visible_frames(puzzle: dict) -> list[dict]:
    """Frames the model is allowed to see, sorted by position.

    Filters strictly on masked_positions so an inline answer grid on a masked
    frame can never leak into a prompt.
    """
    masked = set(narc_masked_positions(puzzle))
    frames = [f for f in (puzzle.get("sequence") or [])
              if f and int(f.get("position", -1)) not in masked
              and isinstance(f.get("grid"), list)]
    return sorted(frames, key=lambda f: int(f.get("position", 0)))
