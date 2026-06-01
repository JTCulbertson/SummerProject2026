"""
scripts/utils/qwen_experiment.py — Multimodal Qwen ARC Experiment
=================================================================
Sends an ARC puzzle image to a vision LLM (Qwen3-VL) and parses the response.

Usage:
    uv run scripts/utils/qwen_experiment.py --image <path> [--save-log <file>]
    uv run scripts/utils/qwen_experiment.py --image data/puzzle.png --save-log runs/log.txt

Arguments:
    --image     Path to the ARC problem image
    --api-url   API endpoint (default: MindRouter OpenAI-compatible endpoint)
    --model     Model name (default: qwen3-vl:32b)
    --save-log  File path to append reasoning logs
"""
import json
import os
import base64
import argparse
from pathlib import Path
import requests

DEFAULT_API_URL  = "https://mindrouter.uidaho.edu/v1/chat/completions"
DEFAULT_MODEL    = "qwen3-vl:32b"
DEFAULT_LOG_FILE = "experiment_reasoning.log"

PROMPT = """\
This is an ARC-style puzzle. Using Conceptual Metaphor Theory (CMT), you can interpret abstract
concepts (target domains) through more concrete experiences (source domains).

Analyze the image. The image contains training examples and a test input.
Output your reasoning and then the final output grid for the test input as a JSON object
with keys 'reasoning' and 'output_grid'.

Find the metaphor box in the image and read it.

Source and Target Domains:
    - Source Domain: The concrete or physical experience from which we draw metaphorical expressions.
    - Target Domain: The abstract concept we aim to understand through the source domain.

Inference Process:
By mapping elements from the source domain onto the target domain, you can infer
characteristics of the abstract concept based on the concrete experience.
"""


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


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_reasoning(reasoning: str, log_file: str):
    if log_file == "/dev/null":
        return
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write(reasoning)
        f.write("\n" + "=" * 40 + "\n")
    print(f"Reasoning saved to {log_file}")


def chat_generate(messages: list, model: str, api_url: str) -> dict:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    retry = Retry(
        total=5, connect=3, read=3, backoff_factor=1.0,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["POST", "GET"]),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://",  HTTPAdapter(max_retries=retry))

    payload = {"model": model, "messages": messages, "stream": False}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = _load_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = sess.post(api_url, json=payload, headers=headers, timeout=(10, 1200))
    resp.raise_for_status()
    return resp.json()


def clean_json_text(text: str) -> str:
    import re
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.DOTALL)
    return text.strip()


def _pick_content_and_thinking(data: dict) -> tuple[str, str]:
    """Extract (content, thinking) separately from OpenAI or Ollama response."""
    if not isinstance(data, dict):
        return str(data), ""
    # OpenAI-compatible format
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content  = msg.get("content", "") or ""
        # Qwen3-VL returns reasoning in "reasoning_content"; others use "thinking"
        thinking = msg.get("reasoning_content", "") or msg.get("thinking", "") or ""
        return content, thinking
    # Ollama format
    if "message" in data and isinstance(data["message"], dict):
        content  = data["message"].get("content", "") or ""
        thinking = (data["message"].get("reasoning_content", "") or
                    data["message"].get("thinking", "") or "")
        return content, thinking
    return str(data), ""


def main():
    parser = argparse.ArgumentParser(description="Run Qwen ARC Experiment")
    parser.add_argument("--image",    type=str, required=True, help="Path to the ARC problem image")
    parser.add_argument("--api-url",  type=str, default=DEFAULT_API_URL)
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--save-log", type=str, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    print(f"API:   {args.api_url}")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")

    # Build OpenAI-compatible multimodal message
    if os.path.exists(args.image):
        b64 = encode_image(args.image)
        content_parts = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        print(f"Warning: image not found at {args.image}. Sending text only.")
        content_parts = [{"type": "text", "text": PROMPT}]

    messages = [{"role": "user", "content": content_parts}]

    print("\nSending request...")
    response_data = chat_generate(messages, args.model, args.api_url)

    content, thinking = _pick_content_and_thinking(response_data)

    reasoning   = ""
    output_grid = []

    try:
        parsed = json.loads(clean_json_text(content))
        if isinstance(parsed, dict):
            reasoning   = parsed.get("reasoning", "")
            output_grid = parsed.get("output_grid", [])
    except json.JSONDecodeError:
        reasoning = content

    if thinking:
        print(f"\n--- THINKING ---\n{thinking[:500]}{'...' if len(thinking) > 500 else ''}\n")

    print(f"\n--- REASONING ---\n{reasoning}\n")
    save_reasoning((thinking + "\n" + reasoning).strip() or content, args.save_log)

    if output_grid:
        print(f"Output Grid: {output_grid}")


if __name__ == "__main__":
    main()
