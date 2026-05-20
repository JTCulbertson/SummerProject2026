"""
scripts/utils/qwen_experiment.py — Multimodal Qwen ARC Experiment
=================================================================
Sends an ARC puzzle image to a vision LLM (Qwen3-VL) and parses the response.

Usage:
    python scripts/utils/qwen_experiment.py --image <path> [--save-log <file>]
    python scripts/utils/qwen_experiment.py --image data/puzzle.png --save-log runs/log.txt

Arguments:
    --image     Path to the ARC problem image
    --api-url   API endpoint (default: MindRouter)
    --model     Model name (default: qwen3-VL-32k:32b)
    --save-log  File path to append reasoning logs
"""
import json
import os
import sys
import base64
import argparse
from pathlib import Path
import requests

DEFAULT_API_URL   = "https://mindrouter.uidaho.edu/api/chat"
DEFAULT_MODEL     = "qwen3-VL-32k:32b"
DEFAULT_LOG_FILE  = "experiment_reasoning.log"

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


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_reasoning(reasoning: str, log_file: str):
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write(reasoning)
        f.write("\n" + "=" * 40 + "\n")
    print(f"Reasoning saved to {log_file}")


def chat_generate(messages: list, model: str, api_url: str) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    resp = requests.post(api_url, json=payload, timeout=1200)
    resp.raise_for_status()
    return resp.json()


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


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

    messages = []
    if os.path.exists(args.image):
        messages.append({
            "role": "user",
            "content": PROMPT,
            "images": [encode_image(args.image)],
        })
    else:
        print(f"Warning: image not found at {args.image}. Sending text only.")
        messages.append({"role": "user", "content": PROMPT})

    print("\nSending request...")
    response_data = chat_generate(messages, args.model, args.api_url)

    content = ""
    thinking = ""
    if "message" in response_data:
        content  = response_data["message"].get("content", "")
        thinking = response_data["message"].get("thinking", "")
    else:
        content = str(response_data)

    full_text = content
    if thinking:
        full_text = thinking + "\n" + content if content.strip() else thinking

    reasoning   = ""
    output_grid = []

    try:
        parsed = json.loads(clean_json_text(full_text))
        if isinstance(parsed, dict):
            reasoning   = parsed.get("reasoning", "")
            output_grid = parsed.get("output_grid", [])
    except json.JSONDecodeError:
        reasoning = content

    print(f"\n--- REASONING ---\n{reasoning}\n")
    save_reasoning(reasoning or content, args.save_log)

    if output_grid:
        print(f"Output Grid: {output_grid}")


if __name__ == "__main__":
    main()
