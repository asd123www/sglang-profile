#!/usr/bin/env python3
"""Synthetic multi-turn client for PD/HiCache experiments.

This client grows the prompt exactly by appending the previous turn's
generated token IDs back into the next request. That makes it useful for
studying cache reuse across turns because the prompt length increases in a
controlled way:

turn 1: initial_len tokens -> generate output_len tokens
turn 2: initial_len + output_len -> generate output_len tokens
turn 3: initial_len + 2 * output_len -> generate output_len tokens
...

The client uses the native `/generate` endpoint so it can send `input_ids`
directly and avoid chat template noise when you care about exact token counts.
"""

import argparse
import json
from typing import Any, Optional
import requests
from transformers import AutoTokenizer


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_SEED_TEXT = (
    "This is a synthetic prefix used to exercise multi-turn KV cache reuse. "
)


def build_seed_input_ids(
    tokenizer: Any,
    target_len: int,
    seed_text: str,
) -> list[int]:
    """Create an initial prompt with exactly `target_len` tokens."""
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Seed text must produce at least one token.")

    input_ids: list[int] = []
    while len(input_ids) < target_len:
        input_ids.extend(seed_ids)
    return input_ids[:target_len]


def request_generate(
    session: Any,
    base_url: str,
    input_ids: list[int],
    output_len: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    timeout: float,
) -> dict[str, Any]:
    sampling_params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": output_len,
        # Prevent EOS from ending early so output length is controlled by
        # max_new_tokens unless another server-side limit is reached.
        "ignore_eos": True,
    }
    if top_k is not None:
        sampling_params["top_k"] = top_k

    response = session.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": sampling_params,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"Expected one result, got {len(result)} results.")
        result = result[0]

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected response type: {type(result)!r}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic multi-turn cache reuse experiment."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--turns",
        type=int,
        required=True,
        help="Number of turns to run.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        required=True,
        help="Number of new tokens generated at each turn.",
    )
    parser.add_argument(
        "--initial-len",
        type=int,
        default=None,
        help=(
            "Initial prompt length in tokens. Defaults to --output-len so the "
            "prompt grows as N, 2N, 3N, ..."
        ),
    )
    parser.add_argument(
        "--seed-text",
        default=DEFAULT_SEED_TEXT,
        help="Text used to construct the initial prompt token IDs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k for generation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print a short preview of generated text for each turn.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Number of generated characters to print when --show-text is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_len = args.initial_len or args.output_len

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = build_seed_input_ids(tokenizer, initial_len, args.seed_text)

    session = requests.Session()

    for turn in range(1, args.turns + 1):
        result = request_generate(
            session=session,
            base_url=args.base_url,
            input_ids=prompt_ids,
            output_len=args.output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.timeout,
        )

        output_ids = result.get("output_ids")
        if not isinstance(output_ids, list):
            raise RuntimeError("Response does not contain output_ids.")

        generated_text = result.get("text", "")
        meta_info = result.get("meta_info", {})

        summary = {
            "turn": turn,
            "prompt_tokens_client": len(prompt_ids),
            "generated_tokens_client": len(output_ids),
            "prompt_tokens_server": meta_info.get("prompt_tokens"),
            "completion_tokens_server": meta_info.get("completion_tokens"),
            "cached_tokens": meta_info.get("cached_tokens"),
            "cached_tokens_details": meta_info.get("cached_tokens_details"),
            "finish_reason": meta_info.get("finish_reason"),
            "e2e_latency": meta_info.get("e2e_latency"),
        }
        print(json.dumps(summary, ensure_ascii=True))

        if len(output_ids) != args.output_len:
            raise RuntimeError(
                f"Turn {turn}: expected {args.output_len} output tokens, "
                f"got {len(output_ids)}. finish_reason={meta_info.get('finish_reason')}"
            )

        if args.show_text:
            preview = generated_text[: args.preview_chars].replace("\n", "\\n")
            print(f"turn={turn} text_preview={preview}")

        prompt_ids = prompt_ids + output_ids


if __name__ == "__main__":
    main()
