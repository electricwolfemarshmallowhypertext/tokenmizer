"""Frozen semantic constants and exact execution configuration for protocol v2."""
from __future__ import annotations

PROTOCOL_VERSION = "2.0.0"
METHODS = ("raw_tail", "strong_summary", "tokenmizer")

CONTINUATION_PROMPT = """Continue the interrupted implementation from the supplied session state. Inspect the repository, identify and complete the unfinished task, preserve all current decisions, constraints, and prior fixes, run the existing test suite, and stop when the work is verified. Do not ask for the previous conversation.

In your final response, report:

1. what you completed;
2. the key prior decision you followed;
3. why it replaced the earlier choice;
4. files changed;
5. test result."""

SUMMARY_PROMPT = """Create the strongest possible handoff for a fresh coding agent that must continue this interrupted session. Preserve the unfinished task, current decisions, superseded decisions that must not return, reasons for changes, completed work, prior bug fixes, important files, and non-obvious constraints. Do not add facts. Output only the handoff, with no preface. The handoff must be no more than {token_budget} tokens under the benchmark's fixed cl100k_base counter. If uncertain, write less rather than exceeding the limit.

Session transcript:
{transcript}"""

STRICT_CHECKS = (
    "functional_tests_pass",
    "unfinished_task_completed",
    "current_decision_followed",
    "superseded_decision_absent",
    "prior_fix_intact",
    "correct_files_only",
    "correct_rationale_present_and_obeyed",
)

MODEL_CONFIG = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "base_url": "https://api.anthropic.com",
    "stream": False,
    "output_format": "json",
    "summary_max_turns": 1,
    "continuation_max_turns": 20,
    "summary_max_attempts": 3,
    "summary_max_budget_usd": 1.0,
    "continuation_max_budget_usd": 5.0,
    "summary_timeout_seconds": 600,
    "continuation_timeout_seconds": 1800,
    "single_worker": True,
    "parallel_runs": False,
    "semantic_cache": False,
    "response_cache": False,
    "provider_prompt_cache": False,
    "cross_run_reuse": False,
    "token_counter": "tiktoken:cl100k_base",
    "token_counter_version": "0.13.0",
    "claude_permission_mode": "acceptEdits",
    "claude_tools": "Bash,Read,Edit,Write,Glob,Grep",
}

ORDER_SEED = 260606337

REVIEW_LABELS = ("supported", "unsupported", "not_a_memory_claim")
