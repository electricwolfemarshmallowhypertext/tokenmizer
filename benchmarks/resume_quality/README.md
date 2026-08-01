# Structured State vs. Token Soup

This benchmark asks one narrow question: after a hard context reset, does
TokenMizer help a coding agent continue the correct work better than an
equal-size raw transcript tail or an equal-size LLM handoff?

The suite contains ten interrupted Python coding cases. Every case has a
reversed decision and its reason, completed and unfinished work, an important
file change, a regression that must stay fixed, and a constraint that cannot be
recovered from code alone. Transcripts use ordinary developer language and do
not contain `Decided:`, `Completed:`, or `Fixed:` labels.

The checked-in transcripts are authored benchmark fixtures, not production-
derived conversations. They validate end-to-end continuation mechanics and
decision regressions, but they do not by themselves close the paper's separate
real-transcript external-validity gap. Replace or extend them with consented,
sanitized real sessions before making a real-transcript claim.

## Protocol

The three continuation arms receive the same frozen Git commit and exact user
prompt. Only the appended session context changes:

- `raw_tail`: the longest suffix of the transcript within the TokenMizer budget;
- `strong_summary`: a fresh-model handoff capped at that budget;
- `tokenmizer`: `CheckpointManager.create(...).resume_standard`, produced from
  the existing `GraphMemory` extraction path.

The runner uses Claude Code print mode with a pinned model and CLI version,
unique session IDs, bare mode, no session persistence, no MCP servers, and a
turn limit. Each arm runs in a fresh clone outside the frozen suite. Sandbox and
read-deny rules keep the frozen suite, ground truth, and hidden tests outside the
agent's readable worktree.

A run passes only if all seven deterministic checks pass. No weighted score is
computed. Public test output, hidden verifier output, the final diff, Claude's
JSON result, and a machine-readable result are retained separately.

## Running

These commands intentionally separate protocol freezing from all model calls:

```powershell
python -m benchmarks.resume_quality.runner freeze --output .benchmark/resume-quality
python -m benchmarks.resume_quality.runner summarize --suite .benchmark/resume-quality
python -m benchmarks.resume_quality.runner run --suite .benchmark/resume-quality --output .benchmark/results
python -m benchmarks.resume_quality.runner report --results .benchmark/results --output .benchmark/report.json
python -m benchmarks.resume_quality.runner release-verifiers --suite .benchmark/resume-quality --results .benchmark/results
```

`summarize` makes ten paid model calls. `run` makes thirty paid, agentic calls.
Both commands fail if the Claude Code version or frozen protocol artifacts have
drifted. Use an API-key-backed Claude Code installation; bare mode deliberately
does not read OAuth/keychain state.

Live continuation runs must execute on macOS, Linux, or WSL2. The runner enables
strict Claude Code sandboxing, denies reads of the frozen suite and local Claude/
Codex/TokenMizer state, and forbids unsandboxed command fallback. Native Windows
is rejected because Claude Code's OS sandbox is not available there.

The default model is `claude-sonnet-4-6`, the default turn cap is 20, and the
per-continuation safety cap is USD 5. Override these only at freeze/run time and
report the resulting protocol hash as a distinct experiment.

## Outputs

For every case, the frozen suite includes the starting Git repository, sanitized
transcript, all three contexts, exact continuation prompt, token budget,
starting commit, protocol hash, model, Claude Code version, and TokenMizer
commit SHA. Every completed arm adds its final diff, final response, public test
output, deterministic verifier output, raw Claude JSON, and `result.json`.

The unsupported/invented-memory secondary metric is left as `null` for blind
manual adjudication. It is not silently guessed by a keyword rule and does not
affect the strict pass. Verifiers cannot be released until all 30 arms for the
frozen protocol are present.
