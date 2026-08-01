"""Frozen constants for the Structured State vs. Token Soup benchmark."""
from __future__ import annotations

PROTOCOL_VERSION = "1.0.0"
METHODS = ("raw_tail", "strong_summary", "tokenmizer")

CONTINUATION_PROMPT = """Continue the interrupted implementation from the supplied session state. Inspect the repository, identify and complete the unfinished task, preserve all current decisions, constraints, and prior fixes, run the existing test suite, and stop when the work is verified. Do not ask for the previous conversation.

In your final response, report:

1. what you completed;
2. the key prior decision you followed;
3. why it replaced the earlier choice;
4. files changed;
5. test result."""

SUMMARY_PROMPT = """Create the strongest possible handoff for a fresh coding agent that must continue this interrupted session. Preserve the unfinished task, current decisions, superseded decisions that must not return, reasons for changes, completed work, prior bug fixes, important files, and non-obvious constraints. Do not add facts. Output only the handoff, with no preface. The handoff must fit within {token_budget} tokens under the benchmark counter.

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

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_BUDGET_USD = 5.0
ORDER_SEED = 260606337
