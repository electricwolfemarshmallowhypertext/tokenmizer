"""Structured State vs. Token Soup benchmark runner.

The live commands intentionally do not run during normal tests. ``freeze`` is
deterministic and local; ``summarize`` and ``run`` invoke a real Claude Code
binary and may incur API cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.resume_quality.case_specs import CASES, CaseSpec
from benchmarks.resume_quality.context import (
    json_canonical,
    render_transcript,
    trim_prefix,
    trim_suffix,
)
from benchmarks.resume_quality.protocol import (
    CONTINUATION_PROMPT,
    DEFAULT_MAX_BUDGET_USD,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    METHODS,
    ORDER_SEED,
    PROTOCOL_VERSION,
    STRICT_CHECKS,
    SUMMARY_PROMPT,
)
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.core.tokenizer import count_tokens
from tokenmizer.graph_memory.graph import GraphMemory

ROOT = Path(__file__).resolve().parents[2]
EXTRACTION_FILES = (
    ROOT / "tokenmizer/checkpoints/manager.py",
    ROOT / "tokenmizer/core/tokenizer.py",
    ROOT / "tokenmizer/graph_memory/decision_tracker.py",
    ROOT / "tokenmizer/graph_memory/graph.py",
    ROOT / "tokenmizer/graph_memory/helpers.py",
    ROOT / "tokenmizer/graph_memory/hybrid_extractor.py",
    ROOT / "tokenmizer/graph_memory/types.py",
    ROOT / "tokenmizer/graph_memory/validator.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}")
    return result


def _git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()


def _claude_version() -> str:
    return _run(["claude", "--version"], cwd=ROOT).stdout.strip()


def _safe_repo_path(repo: Path, relative: str) -> Path:
    target = (repo / relative).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"case file escapes repository: {relative}") from exc
    return target


def _case_snapshot(case: CaseSpec) -> dict:
    data = asdict(case)
    data["repo_files"] = {
        name: _sha256_bytes(content.encode("utf-8"))
        for name, content in sorted(case.repo_files.items())
    }
    data["hidden_tests"] = _sha256_bytes(case.hidden_tests.encode("utf-8"))
    return data


def _protocol_snapshot(model: str, claude_version: str, max_turns: int) -> dict:
    extraction_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in EXTRACTION_FILES
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "methods": list(METHODS),
        "strict_checks": list(STRICT_CHECKS),
        "continuation_prompt": CONTINUATION_PROMPT,
        "summary_prompt": SUMMARY_PROMPT,
        "model": model,
        "max_turns": max_turns,
        "order_seed": ORDER_SEED,
        "claude_code_version": claude_version,
        "tokenmizer_commit_sha": _git_sha(),
        "extraction_files": extraction_hashes,
        "cases": [_case_snapshot(case) for case in CASES],
    }


def _materialize_repo(case: CaseSpec, repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=False)
    for relative, content in case.repo_files.items():
        target = _safe_repo_path(repo, relative)
        _write_text(target, content)

    _run(["git", "init", "--quiet"], cwd=repo)
    _run(["git", "config", "user.name", "TokenMizer Benchmark"], cwd=repo)
    _run(["git", "config", "user.email", "benchmark@tokenmizer.invalid"], cwd=repo)
    _run(["git", "add", "-A"], cwd=repo)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_DATE": "2026-06-06T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-06-06T00:00:00+00:00",
        }
    )
    _run(["git", "commit", "--quiet", "-m", f"Interrupted state: {case.title}"], cwd=repo, env=env)
    return _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def _tree_manifest(suite: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    roots = (suite / "cases", suite / "private")
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if ".git" in path.parts or path.name.endswith((".db", ".db-shm", ".db-wal")):
                continue
            relative = str(path.relative_to(suite)).replace("\\", "/")
            manifest[relative] = _sha256_file(path)
    return manifest


def freeze_suite(output: Path, model: str, max_turns: int = DEFAULT_MAX_TURNS) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty suite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    claude_version = _claude_version()
    snapshot = _protocol_snapshot(model, claude_version, max_turns)
    snapshot_text = json_canonical(snapshot)
    protocol_hash = _sha256_bytes(snapshot_text.encode("utf-8"))
    _write_json(output / "protocol.snapshot.json", snapshot)
    _write_text(output / "PROTOCOL_SHA256", protocol_hash + "\n")

    for case in CASES:
        case_dir = output / "cases" / case.case_id
        private_dir = output / "private" / case.case_id
        start_sha = _materialize_repo(case, case_dir / "start_repo")

        messages = list(case.transcript)
        transcript = render_transcript(messages)
        _write_json(case_dir / "transcript.json", messages)
        _write_text(case_dir / "transcript.txt", transcript + "\n")
        _write_text(case_dir / "pending_task.txt", case.pending_task + "\n")
        _write_text(case_dir / "continuation_prompt.txt", CONTINUATION_PROMPT + "\n")

        private_dir.mkdir(parents=True, exist_ok=True)
        # Keep transient SQLite files on the same writable volume as the suite.
        # They are removed before the artifact manifest is frozen and are never
        # copied into an agent worktree.
        with tempfile.TemporaryDirectory(prefix="tokenmizer-state-", dir=private_dir) as storage:
            graph = GraphMemory(case.case_id, storage_dir=storage)
            manager = CheckpointManager(storage_dir=storage)
            checkpoint = manager.create(
                session_id=case.case_id,
                messages=messages,
                graph=graph,
                context_pct=0.85,
                trigger="benchmark_boundary",
                model=model,
            )
            tokenmizer_context = checkpoint.resume_standard

        budget = count_tokens(tokenmizer_context, model)
        raw_tail = trim_suffix(transcript, budget, model)
        if count_tokens(raw_tail, model) > budget:
            raise AssertionError("raw-tail arm exceeded TokenMizer budget")
        _write_text(case_dir / "contexts/tokenmizer.txt", tokenmizer_context)
        _write_text(case_dir / "contexts/raw_tail.txt", raw_tail)
        _write_text(case_dir / "contexts/strong_summary.txt.pending", "Generate with the summarize command.\n")

        public_metadata = {
            "case_id": case.case_id,
            "title": case.title,
            "starting_commit": start_sha,
            "resume_token_budget": budget,
            "raw_tail_tokens": count_tokens(raw_tail, model),
            "tokenmizer_tokens": count_tokens(tokenmizer_context, model),
            "model": model,
            "protocol_sha256": protocol_hash,
        }
        _write_json(case_dir / "metadata.json", public_metadata)
        _write_text(private_dir / "test_hidden.py", case.hidden_tests)
        _write_json(
            private_dir / "ground_truth.json",
            {
                "earlier_choice": case.earlier_choice,
                "current_choice": case.current_choice,
                "reason": case.reason,
                "pending_task": case.pending_task,
                "allowed_changes": list(case.allowed_changes),
                "protected_files": list(case.protected_files),
                "rationale_groups": [list(group) for group in case.rationale_groups],
            },
        )

    lock = {
        "protocol_sha256": protocol_hash,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "max_turns": max_turns,
        "claude_code_version": claude_version,
        "tokenmizer_commit_sha": snapshot["tokenmizer_commit_sha"],
        "artifact_manifest": _tree_manifest(output),
    }
    _write_json(output / "protocol.lock.json", lock)
    print(f"Frozen {len(CASES)} cases at {output}")
    print(f"Protocol SHA-256: {protocol_hash}")


def _load_lock(suite: Path, *, verify_artifacts: bool = True) -> dict:
    lock = json.loads((suite / "protocol.lock.json").read_text(encoding="utf-8"))
    snapshot = json.loads((suite / "protocol.snapshot.json").read_text(encoding="utf-8"))
    actual_protocol_hash = _sha256_bytes(json_canonical(snapshot).encode("utf-8"))
    if actual_protocol_hash != lock["protocol_sha256"]:
        raise RuntimeError("frozen protocol snapshot hash mismatch")
    if verify_artifacts:
        for relative, expected in lock["artifact_manifest"].items():
            path = suite / relative
            if not path.is_file() or _sha256_file(path) != expected:
                raise RuntimeError(f"frozen artifact changed after lock: {relative}")
    return lock


def _frozen_case_ids(suite: Path) -> tuple[str, ...]:
    snapshot = json.loads((suite / "protocol.snapshot.json").read_text(encoding="utf-8"))
    return tuple(case["case_id"] for case in snapshot["cases"])


def _parse_claude_json(output: str) -> dict:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude Code did not return JSON:\n{output}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Claude Code JSON result was not an object")
    return payload


def _base_claude_args(model: str, max_turns: int, max_budget_usd: float) -> list[str]:
    return [
        "claude",
        "--bare",
        "--print",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--max-budget-usd",
        str(max_budget_usd),
        "--session-id",
        str(uuid.uuid4()),
    ]


def _clean_agent_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("TOKENMIZER_"):
            del env[name]
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    env["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    return env


def summarize_suite(suite: Path, max_budget_usd: float) -> None:
    lock = _load_lock(suite)
    if (suite / "summaries.lock.json").exists():
        raise FileExistsError("summaries are already frozen for this suite")
    if _claude_version() != lock["claude_code_version"]:
        raise RuntimeError("Claude Code version differs from the frozen protocol")
    model = lock["model"]
    summary_lock: dict[str, object] = {
        "protocol_sha256": lock["protocol_sha256"],
        "model": model,
        "claude_code_version": lock["claude_code_version"],
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="tokenmizer-summary-") as temporary:
        cwd = Path(temporary)
        for case_id in _frozen_case_ids(suite):
            case_dir = suite / "cases" / case_id
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            budget = int(metadata["resume_token_budget"])
            transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")
            prompt = SUMMARY_PROMPT.format(token_budget=budget, transcript=transcript)
            args = _base_claude_args(model, 1, max_budget_usd)
            args.extend(["--tools", "", "--permission-mode", "dontAsk", prompt])
            result = _run(args, cwd=cwd, env=_clean_agent_env(), timeout=600)
            payload = _parse_claude_json(result.stdout)
            summary = trim_prefix(str(payload.get("result", "")).strip(), budget, model)
            if not summary:
                raise RuntimeError(f"empty summary for {case_id}")
            target = case_dir / "contexts/strong_summary.txt"
            _write_text(target, summary)
            summary_lock["cases"][case_id] = {
                "tokens": count_tokens(summary, model),
                "sha256": _sha256_file(target),
                "usage": payload.get("usage"),
                "cost_usd": payload.get("total_cost_usd"),
            }
            print(f"Summarized {case_id}: {count_tokens(summary, model)}/{budget} tokens")
    _write_json(suite / "summaries.lock.json", summary_lock)


def _verify_summary_lock(suite: Path, lock: dict) -> dict:
    path = suite / "summaries.lock.json"
    if not path.is_file():
        raise RuntimeError("summaries are not prepared; run the summarize command")
    summary_lock = json.loads(path.read_text(encoding="utf-8"))
    if summary_lock.get("protocol_sha256") != lock["protocol_sha256"]:
        raise RuntimeError("summary lock belongs to a different protocol")
    for case_id, metadata in summary_lock["cases"].items():
        path = suite / "cases" / case_id / "contexts/strong_summary.txt"
        if _sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"summary changed after generation: {case_id}")
    return summary_lock


def _changed_files(repo: Path) -> list[str]:
    result = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo)
    changed: list[str] = []
    ignored_parts = {".pytest_cache", "__pycache__"}
    for line in result.stdout.splitlines():
        relative = line[3:].strip().strip('"').replace("\\", "/")
        if not relative or any(part in ignored_parts for part in Path(relative).parts):
            continue
        changed.append(relative)
    return sorted(set(changed))


def _run_pytest(repo: Path, target: str, env: dict[str, str] | None = None) -> dict:
    result = _run(
        [sys.executable, "-m", "pytest", "-q", target],
        cwd=repo,
        env=env,
        timeout=180,
        check=False,
    )
    return {"passed": result.returncode == 0, "exit_code": result.returncode, "output": result.stdout}


def _verify_run(suite: Path, case_id: str, repo: Path, final_response: str) -> tuple[dict, dict]:
    public_tests = _run_pytest(repo, "tests")
    private_dir = suite / "private" / case_id
    ground_truth = json.loads((private_dir / "ground_truth.json").read_text(encoding="utf-8"))
    hidden_path = private_dir / "test_hidden.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    checks: dict[str, bool] = {}
    hidden_outputs: dict[str, str] = {}
    for check in STRICT_CHECKS[:5]:
        result = _run_pytest(repo, f"{hidden_path}::{f'test_{check}'}", env=env)
        checks[check] = result["passed"]
        hidden_outputs[check] = result["output"]
    checks["functional_tests_pass"] = checks["functional_tests_pass"] and public_tests["passed"]

    changed = _changed_files(repo)
    allowed = set(ground_truth["allowed_changes"])
    protected = set(ground_truth["protected_files"])
    checks["correct_files_only"] = bool(set(changed) & allowed) and set(changed) <= allowed and not (
        set(changed) & protected
    )
    response_lower = final_response.lower()
    rationale_present = all(
        any(term.lower() in response_lower for term in group)
        for group in ground_truth["rationale_groups"]
    )
    checks["correct_rationale_present_and_obeyed"] = rationale_present and checks[
        "current_decision_followed"
    ]
    details = {
        "public_tests": public_tests,
        "hidden_test_output": hidden_outputs,
        "changed_files": changed,
        "wrong_file_edits": sorted(set(changed) - allowed),
    }
    return checks, details


def _agent_settings(suite: Path, repo: Path) -> str:
    # The verifier tree is denied to both direct reads and sandboxed subprocesses.
    denied_roots = [
        suite.resolve(),
        ROOT.resolve(),
        Path.home() / ".claude",
        Path.home() / ".codex",
        Path.home() / ".tokenmizer",
    ]
    return json_canonical(
        {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "allowUnsandboxedCommands": False,
                "filesystem": {
                    "denyRead": [str(path) for path in denied_roots],
                    "allowRead": [str(repo.resolve())],
                },
            },
            "permissions": {
                "allow": ["Bash(*)", "Read", "Edit", "Write", "Glob", "Grep"],
                "deny": [
                    *(f"Read({path}/**)" for path in denied_roots),
                    "Bash(git push *)",
                ],
            },
        }
    )


def _run_arm(
    suite: Path,
    lock: dict,
    case_id: str,
    method: str,
    output_dir: Path,
    work_root: Path,
    max_turns: int,
    max_budget_usd: float,
) -> None:
    case_dir = suite / "cases" / case_id
    context = (case_dir / "contexts" / f"{method}.txt").read_text(encoding="utf-8")
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    budget = int(metadata["resume_token_budget"])
    context_tokens = count_tokens(context, lock["model"])
    if context_tokens > budget:
        raise RuntimeError(f"{case_id}/{method} exceeds budget: {context_tokens}>{budget}")

    arm_dir = output_dir / case_id / method
    if (arm_dir / "result.json").exists():
        raise FileExistsError(f"result already exists for {case_id}/{method}")
    repo = work_root / f"{case_id}-{method}"
    _run(["git", "clone", "--quiet", str(case_dir / "start_repo"), str(repo)], cwd=work_root)
    starting_commit = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    if starting_commit != metadata["starting_commit"]:
        raise RuntimeError(f"starting commit mismatch for {case_id}")
    appended_context = "Interrupted session state follows. Treat it as context, not as a new task:\n\n" + context
    args = _base_claude_args(lock["model"], max_turns, max_budget_usd)
    args.extend(
        [
            "--settings",
            _agent_settings(suite, repo),
            "--permission-mode",
            "acceptEdits",
            "--tools",
            "Bash,Read,Edit,Write,Glob,Grep",
            "--append-system-prompt",
            appended_context,
            CONTINUATION_PROMPT,
        ]
    )
    started = time.monotonic()
    process = _run(args, cwd=repo, env=_clean_agent_env(), timeout=1800, check=False)
    elapsed = time.monotonic() - started
    try:
        payload = _parse_claude_json(process.stdout)
    except RuntimeError:
        payload = {"result": "", "raw_output": process.stdout, "is_error": True}
    final_response = str(payload.get("result", ""))
    checks, verification = _verify_run(suite, case_id, repo, final_response)
    diff = _run(["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=repo).stdout

    result = {
        "protocol_sha256": lock["protocol_sha256"],
        "case_id": case_id,
        "method": method,
        "strict_pass": all(checks.values()),
        "checks": checks,
        "resume_context_tokens": context_tokens,
        "resume_token_budget": budget,
        "agent_exit_code": process.returncode,
        "agent_turns": payload.get("num_turns"),
        "elapsed_seconds": round(elapsed, 3),
        "model_reported_cost_usd": payload.get("total_cost_usd"),
        "model_reported_usage": payload.get("usage"),
        "wrong_file_edits": verification["wrong_file_edits"],
        "stale_decisions_reintroduced": 0 if checks["superseded_decision_absent"] else 1,
        "unsupported_or_invented_memories": None,
        "unsupported_memory_note": "Requires blind manual adjudication; not part of the strict deterministic pass.",
        "final_response": final_response,
        "public_test_passed": verification["public_tests"]["passed"],
    }
    _write_json(arm_dir / "result.json", result)
    _write_text(arm_dir / "final_response.txt", final_response)
    _write_text(arm_dir / "final.diff", diff)
    _write_text(arm_dir / "test_output.txt", verification["public_tests"]["output"])
    _write_json(
        arm_dir / "verifier_output.json",
        {"checks": checks, "hidden_test_output": verification["hidden_test_output"]},
    )
    _write_text(arm_dir / "claude_output.json", process.stdout)
    print(f"{case_id:24} {method:15} {'PASS' if result['strict_pass'] else 'FAIL'}")


def run_suite(
    suite: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    max_budget_usd: float,
    work_root: Path | None,
) -> None:
    if os.name == "nt":
        raise RuntimeError(
            "live continuation runs require Claude Code's OS sandbox, which is not "
            "supported on native Windows; run this command inside WSL2 or a Linux container"
        )
    lock = _load_lock(suite)
    _verify_summary_lock(suite, lock)
    if _claude_version() != lock["claude_code_version"]:
        raise RuntimeError("Claude Code version differs from the frozen protocol")
    invalid = set(methods) - set(METHODS)
    if invalid:
        raise ValueError(f"unknown methods: {sorted(invalid)}")

    jobs = [(case_id, method) for case_id in _frozen_case_ids(suite) for method in methods]
    random.Random(ORDER_SEED).shuffle(jobs)
    output_dir.mkdir(parents=True, exist_ok=True)
    if work_root is None:
        with tempfile.TemporaryDirectory(prefix="tokenmizer-resume-runs-") as temporary:
            root = Path(temporary)
            for case_id, method in jobs:
                _run_arm(
                    suite, lock, case_id, method, output_dir, root,
                    int(lock["max_turns"]), max_budget_usd,
                )
    else:
        work_root.mkdir(parents=True, exist_ok=True)
        for case_id, method in jobs:
            _run_arm(
                suite, lock, case_id, method, output_dir, work_root,
                int(lock["max_turns"]), max_budget_usd,
            )


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*/result.json"))


def report_suite(results_dir: Path, output: Path | None = None) -> dict:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in _result_files(results_dir)]
    table: dict[str, dict] = {}
    for method in METHODS:
        rows = [row for row in results if row["method"] == method]
        table[method] = {
            "strict_continuations": sum(bool(row["strict_pass"]) for row in rows),
            "runs": len(rows),
            "decision_regressions": sum(row["stale_decisions_reintroduced"] for row in rows),
            "correct_rationale": sum(
                bool(row["checks"]["correct_rationale_present_and_obeyed"]) for row in rows
            ),
            "average_resume_tokens": (
                round(sum(row["resume_context_tokens"] for row in rows) / len(rows), 1)
                if rows else None
            ),
        }
    report = {"methods": table, "results": results}
    if output:
        _write_json(output, report)
    print("Method           Strict   Regressions   Rationale   Avg resume tokens")
    for method, row in table.items():
        print(
            f"{method:16} {row['strict_continuations']:>2}/{row['runs']:<2}"
            f" {row['decision_regressions']:>12}"
            f" {row['correct_rationale']:>9}/{row['runs']:<2}"
            f" {str(row['average_resume_tokens']):>18}"
        )
    return report


def release_verifiers(suite: Path, results_dir: Path) -> None:
    lock = _load_lock(suite)
    expected = {(case_id, method) for case_id in _frozen_case_ids(suite) for method in METHODS}
    actual = set()
    for path in _result_files(results_dir):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol_sha256") == lock["protocol_sha256"]:
            actual.add((result["case_id"], result["method"]))
    missing = expected - actual
    if missing:
        raise RuntimeError(f"will not release verifiers before all 30 arms finish; missing {len(missing)}")
    destination = suite / "released_verifiers"
    if destination.exists():
        raise FileExistsError(f"verifiers already released: {destination}")
    shutil.copytree(suite / "private", destination)
    print(f"Released verifiers to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze/hash cases and build starting repositories")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--model", default=DEFAULT_MODEL)
    freeze.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)

    summarize = commands.add_parser("summarize", help="generate budget-matched strong summaries")
    summarize.add_argument("--suite", type=Path, required=True)
    summarize.add_argument("--max-budget-usd", type=float, default=1.0)

    run = commands.add_parser("run", help="run fresh Claude Code continuations")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run.add_argument("--max-budget-usd", type=float, default=DEFAULT_MAX_BUDGET_USD)
    run.add_argument("--work-root", type=Path)

    report = commands.add_parser("report", help="aggregate strict and secondary measurements")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path)

    release = commands.add_parser("release-verifiers", help="publish hidden verifiers after all arms")
    release.add_argument("--suite", type=Path, required=True)
    release.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        freeze_suite(args.output.resolve(), args.model, args.max_turns)
    elif args.command == "summarize":
        summarize_suite(args.suite.resolve(), args.max_budget_usd)
    elif args.command == "run":
        run_suite(
            args.suite.resolve(), args.output.resolve(), tuple(args.methods),
            args.max_budget_usd,
            args.work_root.resolve() if args.work_root else None,
        )
    elif args.command == "report":
        report_suite(args.results.resolve(), args.output.resolve() if args.output else None)
    elif args.command == "release-verifiers":
        release_verifiers(args.suite.resolve(), args.results.resolve())


if __name__ == "__main__":
    main()
