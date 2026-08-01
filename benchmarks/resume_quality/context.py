"""Context construction and budget enforcement for benchmark arms."""
from __future__ import annotations

import json

from tokenmizer.core.tokenizer import count_tokens


def render_transcript(messages: list[dict[str, str]]) -> str:
    """Render a transcript without benchmark-only labels or hidden metadata."""
    return "\n\n".join(
        f"[{message['role']}]\n{message['content']}" for message in messages
    )


def trim_prefix(text: str, token_budget: int, model: str) -> str:
    """Return the longest character prefix that fits the shared counter."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if count_tokens(text, model) <= token_budget:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid], model) <= token_budget:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def trim_suffix(text: str, token_budget: int, model: str) -> str:
    """Return the longest character suffix that fits the shared counter."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if count_tokens(text, model) <= token_budget:
        return text
    low, high = 0, len(text)
    while low < high:
        length = (low + high + 1) // 2
        if count_tokens(text[-length:], model) <= token_budget:
            low = length
        else:
            high = length - 1
    return text[-low:] if low else ""


def json_canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
