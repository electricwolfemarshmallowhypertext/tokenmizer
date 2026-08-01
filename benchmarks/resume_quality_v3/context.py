"""Deterministic context construction and the single v3 token counter."""
from __future__ import annotations

import json
from importlib.metadata import version

import tiktoken

TOKEN_ENCODING = "cl100k_base"
TOKEN_COUNTER_VERSION = "0.13.0"


def count_tokens(text: str) -> int:
    """Count every v3 resume context with one fixed local tokenizer."""
    installed = version("tiktoken")
    if installed != TOKEN_COUNTER_VERSION:
        raise RuntimeError(
            f"protocol v3 requires tiktoken {TOKEN_COUNTER_VERSION}, found {installed}"
        )
    if not text:
        return 0
    return len(tiktoken.get_encoding(TOKEN_ENCODING).encode(text, disallowed_special=()))


def render_transcript(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}" for message in messages
    )


def trim_suffix(text: str, token_budget: int) -> str:
    """Return the longest suffix within budget; used only for the raw-tail arm."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if count_tokens(text) <= token_budget:
        return text
    low, high = 0, len(text)
    while low < high:
        length = (low + high + 1) // 2
        if count_tokens(text[-length:]) <= token_budget:
            low = length
        else:
            high = length - 1
    return text[-low:] if low else ""


def json_canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

