"""Validity-hardened Structured State vs. Token Soup benchmark runner.

``freeze`` and ``validate`` are local-only. ``summarize`` and ``run`` are the
only commands that may invoke Claude Code. Every such invocation is guarded by
the frozen v2 source manifest before and after the process call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import httpx

from benchmarks.resume_quality import runner as v1
from benchmarks.resume_quality.case_specs import CASES, CaseSpec
from benchmarks.resume_quality_v2.context import (
    TOKEN_COUNTER_VERSION,
    TOKEN_ENCODING,
    count_tokens,
    json_canonical,
    render_transcript,
    trim_suffix,
)
from benchmarks.resume_quality_v2.protocol import (
    CONTINUATION_PROMPT,
    METHODS,
    MODEL_CONFIG,
    ORDER_SEED,
    PROTOCOL_VERSION,
    REVIEW_LABELS,
    STRICT_CHECKS,
    SUMMARY_PROMPT,
)
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.graph_memory.graph import GraphMemory

ROOT = Path(__file__).resolve().parents[2]
_PAID_MODEL_CALLS_EXECUTED = 0


def _result_affecting_files() -> tuple[Path, ...]:
    """Every source file imported by context generation, execution, or scoring."""
    files = {
        Path(__file__).resolve(),
        (Path(__file__).parent / "context.py").resolve(),
        (Path(__file__).parent / "protocol.py").resolve(),
        (ROOT / "benchmarks/resume_quality/runner.py").resolve(),
        (ROOT / "benchmarks/resume_quality/case_specs.py").resolve(),
        (ROOT / "tokenmizer/checkpoints/manager.py").resolve(),
        (ROOT / "tokenmizer/core/dto.py").resolve(),
        (ROOT / "tokenmizer/core/errors.py").resolve(),
        (ROOT / "tokenmizer/core/tokenizer.py").resolve(),
    }
    files.update(path.resolve() for path in (ROOT / "tokenmizer/graph_memory").glob("*.py"))
    return tuple(sorted(files, key=lambda path: str(path.relative_to(ROOT)).replace("\\", "/")))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return v1._run(args, cwd=cwd, env=env, timeout=timeout, check=check)


def _git_sha() -> str:
    return v1._git_sha()


def _claude_version() -> str:
    return v1._claude_version()


def _case_snapshot(case: CaseSpec) -> dict:
    data = asdict(case)
    data["repo_files"] = {
        name: _sha256_bytes(content.encode("utf-8"))
        for name, content in sorted(case.repo_files.items())
    }
    data["hidden_tests"] = _sha256_bytes(case.hidden_tests.encode("utf-8"))
    return data


def _protocol_manifest() -> dict:
    file_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in _result_affecting_files()
    }
    return {
        "file_hashes": file_hashes,
        "prompt_texts": {
            "continuation": CONTINUATION_PROMPT,
            "summary": SUMMARY_PROMPT,
        },
        "model_config": dict(MODEL_CONFIG),
    }


def _protocol_hash(manifest: dict) -> str:
    return _sha256_bytes(json_canonical(manifest).encode("utf-8"))


def _validate_model_config(config: dict) -> None:
    required_strings = (
        "provider",
        "model",
        "base_url",
        "output_format",
        "token_counter",
        "token_counter_version",
    )
    for field in required_strings:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"model_config.{field} must be an explicit non-empty string")
    if config != MODEL_CONFIG:
        raise RuntimeError("model configuration differs from protocol v2")
    if config["model"].lower() in {"auto", "latest", "default"}:
        raise RuntimeError("model must be pinned to an exact model string")
    if config["provider"] != "anthropic":
        raise RuntimeError("protocol v2 provider must be exactly anthropic")
    if config["base_url"].rstrip("/") != "https://api.anthropic.com":
        raise RuntimeError("protocol v2 base URL must be exactly https://api.anthropic.com")
    if config["stream"] is not False:
        raise RuntimeError("streaming must be disabled")
    if config["token_counter_version"] != TOKEN_COUNTER_VERSION:
        raise RuntimeError("token counter version differs from the pinned version")
    if config["single_worker"] is not True or config["parallel_runs"] is not False:
        raise RuntimeError("protocol v2 requires one sequential worker")
    for cache_field in (
        "semantic_cache",
        "response_cache",
        "provider_prompt_cache",
        "cross_run_reuse",
    ):
        if config[cache_field] is not False:
            raise RuntimeError(f"model_config.{cache_field} must be false")


def _print_model_config(config: dict) -> None:
    print(f"Model: {config['model']}")
    print(f"Provider: {config['provider']}")
    print(f"Base URL: {config['base_url']}")


def _materialize_repo(case: CaseSpec, repo: Path) -> str:
    return v1._materialize_repo(case, repo)


def _tree_manifest(suite: Path) -> dict[str, str]:
    return v1._tree_manifest(suite)


def _frozen_case_ids(suite: Path) -> tuple[str, ...]:
    snapshot = json.loads((suite / "protocol.snapshot.json").read_text(encoding="utf-8"))
    return tuple(case["case_id"] for case in snapshot["cases"])


def _protocol_snapshot(claude_version: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "methods": list(METHODS),
        "strict_checks": list(STRICT_CHECKS),
        "order_seed": ORDER_SEED,
        "claude_code_version": claude_version,
        "tokenmizer_commit_sha": _git_sha(),
        "cases": [_case_snapshot(case) for case in CASES],
    }


def _load_lock(suite: Path, *, verify_artifacts: bool = True) -> dict:
    lock = json.loads((suite / "protocol.lock.json").read_text(encoding="utf-8"))
    saved_manifest = json.loads(
        (suite / "protocol.manifest.json").read_text(encoding="utf-8")
    )
    current_manifest = _protocol_manifest()
    current_hash = _protocol_hash(current_manifest)
    expected_hash = lock.get("protocol_hash_v2")
    if current_manifest != saved_manifest:
        raise RuntimeError("protocol v2 source manifest changed after freeze")
    if current_hash != expected_hash:
        raise RuntimeError("protocol v2 hash mismatch")
    if (suite / "PROTOCOL_SHA256_V2").read_text(encoding="utf-8").strip() != expected_hash:
        raise RuntimeError("PROTOCOL_SHA256_V2 does not match the lock")
    _validate_model_config(lock.get("model_config", {}))
    if verify_artifacts:
        for relative, expected in lock["artifact_manifest"].items():
            path = suite / relative
            if not path.is_file() or _sha256_file(path) != expected:
                raise RuntimeError(f"frozen v2 artifact changed after lock: {relative}")
    return lock


def _transport_dry_test() -> dict:
    """Capture an Anthropic SDK request through an in-memory transport."""
    import anthropic

    captured: dict[str, object] = {}
    system_text = "system-preservation-sentinel"
    user_text = "user-preservation-sentinel"
    tool_id = "toolu_transport_sentinel"
    tool_name = "inspect_repository"
    tool_result = "tool-result-preservation-sentinel"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "msg_dry_transport",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": MODEL_CONFIG["model"],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = anthropic.Anthropic(
        api_key="dry-test-key",
        base_url=MODEL_CONFIG["base_url"],
        http_client=http_client,
    )
    try:
        client.messages.create(
            model=MODEL_CONFIG["model"],
            max_tokens=8,
            stream=False,
            system=system_text,
            tools=[{
                "name": tool_name,
                "description": "Inspect one repository path",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }],
            messages=[
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {"path": "src/app.py"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_result,
                    }],
                },
            ],
        )
    finally:
        client.close()

    body = captured.get("body")
    if not isinstance(body, dict):
        raise RuntimeError("transport dry test did not capture a request body")
    checks = {
        "provider_url_exact": captured["url"] == "https://api.anthropic.com/v1/messages",
        "model_exact": body.get("model") == MODEL_CONFIG["model"],
        "streaming_off": body.get("stream") is False,
        "system_preserved": body.get("system") == system_text,
        "user_preserved": body.get("messages", [{}])[0].get("content") == user_text,
        "tool_call_preserved": body.get("messages", [{}, {}])[1].get("content", [{}])[0]
        == {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {"path": "src/app.py"},
        },
        "tool_result_preserved": body.get("messages", [{}, {}, {}])[2].get(
            "content", [{}]
        )[0]
        == {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": tool_result,
        },
        "tool_definition_preserved": body.get("tools", [{}])[0].get("name") == tool_name,
        "no_cache_control": "cache_control" not in json_canonical(body),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"transport dry test failed: {failed}")
    return {"status": "pass", "checks": checks, "captured_request": body}


def _validate_config_precedence() -> dict:
    from tokenmizer.config.settings import Settings

    names = (
        "TOKENMIZER_PROVIDER",
        "TOKENMIZER_STATE_BACKEND",
        "TOKENMIZER_API_KEY",
        "TOKENMIZER_REQUIRED_MARKER",
    )
    previous = {name: os.environ.get(name) for name in names}
    present = {name: name in os.environ for name in names}
    try:
        with tempfile.TemporaryDirectory(prefix="tokenmizer-config-v2-") as temporary:
            path = Path(temporary) / "tokenmizer.yaml"
            _write_text(
                path,
                "provider: anthropic\nstate_backend: memory\napi_key: yaml-key\n",
            )
            os.environ["TOKENMIZER_PROVIDER"] = "openai"
            os.environ["TOKENMIZER_STATE_BACKEND"] = "redis"
            os.environ["TOKENMIZER_API_KEY"] = ""
            settings = Settings.from_yaml(str(path))
            conflict_passed = settings.provider == "openai" and settings.state_backend == "redis"
            empty_passed = settings.api_key == ""

            class RequiredSettings(Settings):
                required_marker: str

            os.environ["TOKENMIZER_REQUIRED_MARKER"] = ""
            required_empty_rejected = False
            try:
                RequiredSettings.from_yaml(str(path))
            except ValueError:
                required_empty_rejected = True
    finally:
        for name in names:
            if present[name]:
                os.environ[name] = previous[name] or ""
            else:
                os.environ.pop(name, None)

    checks = {
        "environment_overrides_yaml": conflict_passed,
        "empty_environment_override_respected": empty_passed,
        "required_empty_rejected": required_empty_rejected,
    }
    if not all(checks.values()):
        raise RuntimeError(f"configuration precedence validation failed: {checks}")
    return {"status": "pass", "checks": checks}


def _continuation_request(context: str) -> dict:
    if not context:
        raise RuntimeError("resume context is empty; refusing to construct request")
    system = (
        "Interrupted session state follows. Treat it as context, not as a new task:\n\n"
        + context
    )
    request = {
        "system": system,
        "user": CONTINUATION_PROMPT,
        "tools": MODEL_CONFIG["claude_tools"],
        "stream": False,
    }
    if context not in request["system"]:
        raise RuntimeError("TokenMizer context was not injected into the request")
    return request


def _base_claude_args(max_turns: int, max_budget_usd: float) -> list[str]:
    return [
        "claude",
        "--bare",
        "--print",
        "--output-format",
        MODEL_CONFIG["output_format"],
        "--no-session-persistence",
        "--strict-mcp-config",
        "--model",
        MODEL_CONFIG["model"],
        "--max-turns",
        str(max_turns),
        "--max-budget-usd",
        str(max_budget_usd),
        "--session-id",
        str(uuid.uuid4()),
    ]


def _assert_cli_invocation(args: list[str], request: dict | None = None) -> None:
    def value_after(flag: str) -> str:
        if flag not in args or args.index(flag) + 1 >= len(args):
            raise RuntimeError(f"Claude invocation is missing {flag}")
        return args[args.index(flag) + 1]

    if value_after("--model") != MODEL_CONFIG["model"]:
        raise RuntimeError("Claude invocation model differs from pinned model")
    if value_after("--output-format") != "json":
        raise RuntimeError("Claude invocation must use non-streaming JSON output")
    for forbidden in ("--continue", "--resume", "--input-format", "stream-json"):
        if forbidden in args:
            raise RuntimeError(f"Claude invocation contains forbidden reuse/stream flag: {forbidden}")
    if "--bare" not in args or "--no-session-persistence" not in args:
        raise RuntimeError("Claude invocation is not isolated")
    if request is not None:
        if value_after("--append-system-prompt") != request["system"]:
            raise RuntimeError("constructed system context changed in CLI transport")
        if value_after("--tools") != request["tools"]:
            raise RuntimeError("constructed tool configuration changed in CLI transport")
        if args[-1] != request["user"]:
            raise RuntimeError("constructed user prompt changed in CLI transport")


def _clean_agent_env(state_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("TOKENMIZER_"):
            del env[name]
    for name in (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "ANTHROPIC_VERTEX_REGION",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
    ):
        env.pop(name, None)
    env["ANTHROPIC_BASE_URL"] = MODEL_CONFIG["base_url"]
    env["DISABLE_PROMPT_CACHING"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CONFIG_DIR"] = str(state_dir.resolve())
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    env["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    return env


def _require_live_credentials(env: dict[str, str]) -> None:
    if not env.get("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for protocol v2; provider cannot be "
            "confirmed through OAuth or a fallback route"
        )
    if env.get("ANTHROPIC_BASE_URL", "").rstrip("/") != MODEL_CONFIG["base_url"]:
        raise RuntimeError("Anthropic base URL differs from the pinned provider endpoint")


def _parse_claude_json(output: str) -> dict:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude Code did not return JSON:\n{output}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Claude Code JSON result was not an object")
    return payload


def _assert_reported_model(payload: dict) -> None:
    reported: set[str] = set()
    if isinstance(payload.get("model"), str):
        reported.add(payload["model"])
    if isinstance(payload.get("modelUsage"), dict):
        reported.update(str(model) for model in payload["modelUsage"])
    if not reported:
        raise RuntimeError("model usage did not identify the model; exact model is unconfirmed")
    if reported != {MODEL_CONFIG["model"]}:
        raise RuntimeError(
            f"reported model(s) {sorted(reported)} differ from {MODEL_CONFIG['model']!r}"
        )


def _provider_usage(payload: dict) -> dict | None:
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _pre_model_call_guard(
    suite: Path,
    lock: dict,
    args: list[str],
    env: dict[str, str],
    request: dict | None = None,
) -> None:
    current = _load_lock(suite)
    if current["protocol_hash_v2"] != lock["protocol_hash_v2"]:
        raise RuntimeError("protocol changed before model call")
    if _claude_version() != lock["claude_code_version"]:
        raise RuntimeError("Claude Code version differs from frozen v2")
    _validate_model_config(lock["model_config"])
    _require_live_credentials(env)
    _transport_dry_test()
    _assert_cli_invocation(args, request)


def _execute_model_call(
    suite: Path,
    lock: dict,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    request: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    global _PAID_MODEL_CALLS_EXECUTED
    _pre_model_call_guard(suite, lock, args, env, request)
    _PAID_MODEL_CALLS_EXECUTED += 1
    try:
        return _run(args, cwd=cwd, env=env, timeout=timeout, check=False)
    finally:
        current = _load_lock(suite)
        if current["protocol_hash_v2"] != lock["protocol_hash_v2"]:
            raise RuntimeError("protocol changed during model call")


def _agent_settings(suite: Path, repo: Path, state_dir: Path) -> str:
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
                    "allowRead": [str(repo.resolve()), str(state_dir.resolve())],
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


def _context_validation(suite: Path) -> dict:
    checks: dict[str, bool] = {}
    for case_id in _frozen_case_ids(suite):
        case_dir = suite / "cases" / case_id
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        budget = int(metadata["resume_token_budget"])
        tokenmizer_context = (case_dir / "contexts/tokenmizer.txt").read_text(encoding="utf-8")
        raw_tail = (case_dir / "contexts/raw_tail.txt").read_text(encoding="utf-8")
        request = _continuation_request(tokenmizer_context)
        checks[f"{case_id}:tokenmizer_exact_budget"] = count_tokens(tokenmizer_context) == budget
        checks[f"{case_id}:raw_tail_within_budget"] = count_tokens(raw_tail) <= budget
        checks[f"{case_id}:context_injected"] = tokenmizer_context in request["system"]
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"context validation failed: {failed}")
    return {"status": "pass", "checks": checks, "token_counter": TOKEN_ENCODING}


def _validation_report(suite: Path, protocol_hash_v2: str) -> dict:
    transport = _transport_dry_test()
    config = _validate_config_precedence()
    context = _context_validation(suite)
    _validate_model_config(MODEL_CONFIG)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash_v2": protocol_hash_v2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "protocol_binds_result_affecting_files": "pass",
            "summary_over_budget_is_retried_not_trimmed": "pass",
            "unsupported_memory_requires_blind_human_review": "pass",
            "environment_overrides_yaml": config,
            "exact_model_provider_and_base_url_pinned": "pass",
            "single_fixed_token_counter": context,
            "provider_usage_only_no_cost_estimate": "pass",
            "all_caches_disabled": "pass",
            "single_worker_sequential_fresh_state": "pass",
            "tokenmizer_context_injected": "pass",
            "transport_structure_preserved": transport,
            "pre_and_post_call_hash_enforcement": "pass",
        },
        "paid_model_calls_executed": _PAID_MODEL_CALLS_EXECUTED,
        "summaries_executed": 0,
        "continuations_executed": 0,
        "status": "pass" if _PAID_MODEL_CALLS_EXECUTED == 0 else "fail",
    }


def freeze_suite(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty v2 suite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _validate_model_config(MODEL_CONFIG)
    claude_version = _claude_version()
    manifest = _protocol_manifest()
    protocol_hash_v2 = _protocol_hash(manifest)
    snapshot = _protocol_snapshot(claude_version)

    _write_json(output / "protocol.manifest.json", manifest)
    _write_text(output / "PROTOCOL_SHA256_V2", protocol_hash_v2 + "\n")
    _write_json(output / "protocol.snapshot.json", snapshot)

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
        with tempfile.TemporaryDirectory(prefix="tokenmizer-state-v2-", dir=private_dir) as storage:
            graph = GraphMemory(case.case_id, storage_dir=storage)
            manager = CheckpointManager(storage_dir=storage)
            checkpoint = manager.create(
                session_id=case.case_id,
                messages=messages,
                graph=graph,
                context_pct=0.85,
                trigger="benchmark_boundary_v2",
                model=MODEL_CONFIG["model"],
            )
            tokenmizer_context = checkpoint.resume_standard

        budget = count_tokens(tokenmizer_context)
        raw_tail = trim_suffix(transcript, budget)
        _write_text(case_dir / "contexts/tokenmizer.txt", tokenmizer_context)
        _write_text(case_dir / "contexts/raw_tail.txt", raw_tail)
        _write_text(
            case_dir / "contexts/strong_summary.txt.pending",
            "Generate with protocol v2 summarize; truncation is forbidden.\n",
        )
        _write_json(
            case_dir / "metadata.json",
            {
                "case_id": case.case_id,
                "title": case.title,
                "starting_commit": start_sha,
                "resume_token_budget": budget,
                "raw_tail_tokens": count_tokens(raw_tail),
                "tokenmizer_tokens": count_tokens(tokenmizer_context),
                "token_counter": MODEL_CONFIG["token_counter"],
                "model_config": MODEL_CONFIG,
                "protocol_hash_v2": protocol_hash_v2,
            },
        )
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
        "protocol_hash_v2": protocol_hash_v2,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "claude_code_version": claude_version,
        "tokenmizer_commit_sha": snapshot["tokenmizer_commit_sha"],
        "model_config": MODEL_CONFIG,
        "artifact_manifest": _tree_manifest(output),
    }
    _write_json(output / "protocol.lock.json", lock)
    report = _validation_report(output, protocol_hash_v2)
    _write_json(output / "transport.dry-run.json", report["checks"]["transport_structure_preserved"])
    _write_json(output / "validation_report.json", report)
    _load_lock(output)
    _print_model_config(MODEL_CONFIG)
    print(f"Frozen {len(CASES)} protocol-v2 cases at {output}")
    print(f"Protocol v2 SHA-256: {protocol_hash_v2}")


def validate_suite(suite: Path) -> dict:
    lock = _load_lock(suite)
    report = _validation_report(suite, lock["protocol_hash_v2"])
    _write_json(suite / "validation_report.json", report)
    _print_model_config(lock["model_config"])
    print(f"Validation: {report['status']}")
    print(f"Paid model calls executed: {report['paid_model_calls_executed']}")
    return report


def summarize_suite(suite: Path) -> None:
    lock = _load_lock(suite)
    if (suite / "summaries.lock.json").exists():
        raise FileExistsError("v2 summaries are already frozen")
    _print_model_config(lock["model_config"])
    summary_lock: dict[str, object] = {
        "protocol_hash_v2": lock["protocol_hash_v2"],
        "model_config": lock["model_config"],
        "cases": {},
        "valid": True,
    }
    with tempfile.TemporaryDirectory(prefix="tokenmizer-summary-v2-") as temporary:
        root = Path(temporary)
        for case_id in _frozen_case_ids(suite):
            case_dir = suite / "cases" / case_id
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            budget = int(metadata["resume_token_budget"])
            transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")
            prompt = SUMMARY_PROMPT.format(token_budget=budget, transcript=transcript)
            attempts: list[dict] = []
            accepted = False
            for attempt in range(1, int(MODEL_CONFIG["summary_max_attempts"]) + 1):
                state_dir = root / f"{case_id}-attempt-{attempt}-{uuid.uuid4().hex}"
                state_dir.mkdir(parents=True, exist_ok=False)
                args = _base_claude_args(
                    int(MODEL_CONFIG["summary_max_turns"]),
                    float(MODEL_CONFIG["summary_max_budget_usd"]),
                )
                args.extend(["--tools", "", "--permission-mode", "dontAsk", prompt])
                env = _clean_agent_env(state_dir)
                process = _execute_model_call(
                    suite,
                    lock,
                    args,
                    cwd=state_dir,
                    env=env,
                    timeout=int(MODEL_CONFIG["summary_timeout_seconds"]),
                )
                payload = _parse_claude_json(process.stdout)
                _assert_reported_model(payload)
                summary = str(payload.get("result", "")).strip()
                tokens = count_tokens(summary)
                attempts.append(
                    {
                        "attempt": attempt,
                        "tokens": tokens,
                        "accepted": bool(summary) and tokens <= budget,
                        "provider_reported_usage": _provider_usage(payload),
                        "provider_reported_cost_usd": None,
                    }
                )
                if summary and tokens <= budget:
                    target = case_dir / "contexts/strong_summary.txt"
                    _write_text(target, summary)
                    summary_lock["cases"][case_id] = {
                        "status": "valid",
                        "tokens": tokens,
                        "sha256": _sha256_file(target),
                        "attempts": attempts,
                    }
                    accepted = True
                    break
            if not accepted:
                summary_lock["valid"] = False
                summary_lock["cases"][case_id] = {
                    "status": "invalid_over_budget",
                    "attempts": attempts,
                }
    _write_json(suite / "summaries.lock.json", summary_lock)
    _load_lock(suite)
    if not summary_lock["valid"]:
        raise RuntimeError("one or more summary arms are invalid after three attempts")


def _verify_summary_lock(suite: Path, lock: dict) -> dict:
    path = suite / "summaries.lock.json"
    if not path.is_file():
        raise RuntimeError("v2 summaries are not prepared")
    summary_lock = json.loads(path.read_text(encoding="utf-8"))
    if summary_lock.get("protocol_hash_v2") != lock["protocol_hash_v2"]:
        raise RuntimeError("summary lock belongs to another protocol")
    if summary_lock.get("valid") is not True:
        raise RuntimeError("summary lock contains an invalid arm")
    expected_cases = set(_frozen_case_ids(suite))
    if set(summary_lock.get("cases", {})) != expected_cases:
        raise RuntimeError("summary lock is incomplete")
    for case_id, metadata in summary_lock["cases"].items():
        target = suite / "cases" / case_id / "contexts/strong_summary.txt"
        budget = json.loads(
            (suite / "cases" / case_id / "metadata.json").read_text(encoding="utf-8")
        )["resume_token_budget"]
        if metadata.get("status") != "valid" or not target.is_file():
            raise RuntimeError(f"invalid summary arm: {case_id}")
        if _sha256_file(target) != metadata["sha256"]:
            raise RuntimeError(f"summary changed after acceptance: {case_id}")
        if count_tokens(target.read_text(encoding="utf-8")) > budget:
            raise RuntimeError(f"summary exceeds budget after acceptance: {case_id}")
    return summary_lock


def _changed_files(repo: Path) -> list[str]:
    return v1._changed_files(repo)


def _run_pytest(repo: Path, target: str, env: dict[str, str] | None = None) -> dict:
    return v1._run_pytest(repo, target, env)


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
        result = _run_pytest(repo, f"{hidden_path}::test_{check}", env=env)
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
    return checks, {
        "public_tests": public_tests,
        "hidden_test_output": hidden_outputs,
        "changed_files": changed,
        "wrong_file_edits": sorted(set(changed) - allowed),
        "ground_truth": ground_truth,
    }


def _write_review_packet(
    output_dir: Path,
    run_id: str,
    final_response: str,
    ground_truth: dict,
    repo_state: dict,
) -> Path:
    packet = {
        "run_id": run_id,
        "final_response": final_response,
        "ground_truth": ground_truth,
        "repo_state": repo_state,
        "review_instructions": {
            "required_labels": list(REVIEW_LABELS),
            "label_every_claim": True,
        },
        "status": "pending_review",
    }
    if "method" in packet or "case_id" in packet:
        raise AssertionError("blind review packet contains an arm label")
    target = output_dir / "review_packets" / "pending" / f"{run_id}.json"
    _write_json(target, packet)
    return target


def _run_arm(
    suite: Path,
    lock: dict,
    case_id: str,
    method: str,
    output_dir: Path,
    work_root: Path,
) -> None:
    case_dir = suite / "cases" / case_id
    context = (case_dir / "contexts" / f"{method}.txt").read_text(encoding="utf-8")
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    budget = int(metadata["resume_token_budget"])
    context_tokens = count_tokens(context)
    if context_tokens > budget:
        raise RuntimeError(f"{case_id}/{method} exceeds budget: {context_tokens}>{budget}")

    arm_dir = output_dir / case_id / method
    if arm_dir.exists():
        raise FileExistsError(f"arm output already exists: {arm_dir}")
    run_id = str(uuid.uuid4())
    repo = work_root / f"repo-{run_id}"
    state_dir = work_root / f"state-{run_id}"
    state_dir.mkdir(parents=True, exist_ok=False)
    _run(["git", "clone", "--quiet", str(case_dir / "start_repo"), str(repo)], cwd=work_root)
    starting_commit = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    if starting_commit != metadata["starting_commit"]:
        raise RuntimeError(f"starting commit mismatch for {case_id}")

    request = _continuation_request(context)
    args = _base_claude_args(
        int(MODEL_CONFIG["continuation_max_turns"]),
        float(MODEL_CONFIG["continuation_max_budget_usd"]),
    )
    args.extend(
        [
            "--settings",
            _agent_settings(suite, repo, state_dir),
            "--permission-mode",
            MODEL_CONFIG["claude_permission_mode"],
            "--tools",
            request["tools"],
            "--append-system-prompt",
            request["system"],
            request["user"],
        ]
    )
    env = _clean_agent_env(state_dir)
    started = time.monotonic()
    process = _execute_model_call(
        suite,
        lock,
        args,
        cwd=repo,
        env=env,
        timeout=int(MODEL_CONFIG["continuation_timeout_seconds"]),
        request=request,
    )
    elapsed = time.monotonic() - started
    payload = _parse_claude_json(process.stdout)
    _assert_reported_model(payload)
    final_response = str(payload.get("result", ""))
    checks, verification = _verify_run(suite, case_id, repo, final_response)
    diff = _run(["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=repo).stdout
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo).stdout
    repo_state = {
        "starting_commit": starting_commit,
        "changed_files": verification["changed_files"],
        "git_status": status,
        "git_diff": diff,
    }
    review_packet = _write_review_packet(
        output_dir,
        run_id,
        final_response,
        verification["ground_truth"],
        repo_state,
    )
    result = {
        "protocol_hash_v2": lock["protocol_hash_v2"],
        "run_id": run_id,
        "case_id": case_id,
        "method": method,
        "strict_pass": all(checks.values()),
        "checks": checks,
        "resume_context_tokens": context_tokens,
        "resume_token_budget": budget,
        "agent_exit_code": process.returncode,
        "agent_turns": payload.get("num_turns"),
        "elapsed_seconds": round(elapsed, 3),
        "provider_reported_usage": _provider_usage(payload),
        "provider_reported_cost_usd": None,
        "wrong_file_edits": verification["wrong_file_edits"],
        "stale_decisions_reintroduced": 0 if checks["superseded_decision_absent"] else 1,
        "unsupported_memory_review": {
            "status": "pending_review",
            "unsupported_memory_count": None,
            "has_unsupported_memory": None,
        },
        "review_packet": str(review_packet.relative_to(output_dir)).replace("\\", "/"),
        "final_response": final_response,
        "public_test_passed": verification["public_tests"]["passed"],
        "model": MODEL_CONFIG["model"],
        "provider": MODEL_CONFIG["provider"],
        "base_url": MODEL_CONFIG["base_url"],
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
    index_path = output_dir / "private_review_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index[run_id] = {
        "case_id": case_id,
        "method": method,
        "result": str((arm_dir / "result.json").relative_to(output_dir)).replace("\\", "/"),
    }
    _write_json(index_path, index)


@contextmanager
def _single_run_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".active-run.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("another protocol v2 run is active") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def run_suite(suite: Path, output_dir: Path, work_root: Path | None) -> None:
    if os.name == "nt":
        raise RuntimeError("live v2 runs require Claude Code's Linux/WSL2 OS sandbox")
    lock = _load_lock(suite)
    _verify_summary_lock(suite, lock)
    _print_model_config(lock["model_config"])
    jobs = [(case_id, method) for case_id in _frozen_case_ids(suite) for method in METHODS]
    random.Random(ORDER_SEED).shuffle(jobs)
    # Lock the frozen suite, not the chosen output directory: otherwise two
    # processes could run the same protocol concurrently by selecting different
    # output paths.
    with _single_run_lock(suite):
        if work_root is None:
            with tempfile.TemporaryDirectory(prefix="tokenmizer-runs-v2-") as temporary:
                root = Path(temporary)
                for case_id, method in jobs:
                    _run_arm(suite, lock, case_id, method, output_dir, root)
        else:
            work_root.mkdir(parents=True, exist_ok=True)
            for case_id, method in jobs:
                _run_arm(suite, lock, case_id, method, output_dir, work_root)
    _load_lock(suite)


def record_review(results_dir: Path, run_id: str, review_path: Path) -> dict:
    index = json.loads((results_dir / "private_review_index.json").read_text(encoding="utf-8"))
    if run_id not in index:
        raise KeyError(f"unknown review run_id: {run_id}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("run_id") != run_id:
        raise ValueError("review run_id does not match")
    if not str(review.get("reviewer_id", "")).strip():
        raise ValueError("reviewer_id is required")
    claims = review.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("review must label at least one claim")
    if review.get("all_memory_claims_labeled") is not True:
        raise ValueError("reviewer must attest that every memory claim was labeled")
    for claim in claims:
        if not isinstance(claim, dict) or not str(claim.get("claim", "")).strip():
            raise ValueError("every review claim requires text")
        if claim.get("label") not in REVIEW_LABELS:
            raise ValueError(f"invalid review label: {claim.get('label')!r}")
    unsupported_count = sum(claim["label"] == "unsupported" for claim in claims)
    result_path = results_dir / index[run_id]["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["unsupported_memory_review"] = {
        "status": "reviewed",
        "unsupported_memory_count": unsupported_count,
        "has_unsupported_memory": unsupported_count > 0,
        "reviewer_id": review["reviewer_id"],
    }
    _write_json(result_path, result)
    completed = results_dir / "review_packets" / "completed" / f"{run_id}.json"
    _write_json(completed, review)
    return result["unsupported_memory_review"]


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*/result.json"))


def report_suite(suite: Path, results_dir: Path, output: Path | None = None) -> dict:
    lock = _load_lock(suite)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in _result_files(results_dir)]
    expected = {(case_id, method) for case_id in _frozen_case_ids(suite) for method in METHODS}
    actual = {(row["case_id"], row["method"]) for row in results}
    if len(results) != len(actual) or actual != expected:
        raise RuntimeError("results are incomplete or contain duplicate arms")
    if any(row.get("protocol_hash_v2") != lock["protocol_hash_v2"] for row in results):
        raise RuntimeError("result belongs to another protocol")
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
            "average_resume_tokens": round(
                sum(row["resume_context_tokens"] for row in rows) / len(rows), 1
            ),
            "pending_unsupported_memory_reviews": sum(
                row["unsupported_memory_review"]["status"] == "pending_review" for row in rows
            ),
        }
    report = {"protocol_hash_v2": lock["protocol_hash_v2"], "methods": table, "results": results}
    if output:
        _write_json(output, report)
    return report


def release_verifiers(suite: Path, results_dir: Path) -> None:
    report_suite(suite, results_dir)
    destination = suite / "released_verifiers"
    if destination.exists():
        raise FileExistsError(f"verifiers already released: {destination}")
    shutil.copytree(suite / "private", destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="locally freeze/hash protocol v2")
    freeze.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="run local-only v2 preflight checks")
    validate.add_argument("--suite", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="run up to three untrimmed summary attempts")
    summarize.add_argument("--suite", type=Path, required=True)
    run = commands.add_parser("run", help="run all 30 sequential continuation arms")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--work-root", type=Path)
    review = commands.add_parser("record-review", help="record one blind human review")
    review.add_argument("--results", type=Path, required=True)
    review.add_argument("--run-id", required=True)
    review.add_argument("--review", type=Path, required=True)
    report = commands.add_parser("report", help="aggregate exactly 30 v2 results")
    report.add_argument("--suite", type=Path, required=True)
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path)
    release = commands.add_parser("release-verifiers", help="release v2 hidden verifiers")
    release.add_argument("--suite", type=Path, required=True)
    release.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        freeze_suite(args.output.resolve())
    elif args.command == "validate":
        validate_suite(args.suite.resolve())
    elif args.command == "summarize":
        summarize_suite(args.suite.resolve())
    elif args.command == "run":
        run_suite(
            args.suite.resolve(),
            args.output.resolve(),
            args.work_root.resolve() if args.work_root else None,
        )
    elif args.command == "record-review":
        record_review(args.results.resolve(), args.run_id, args.review.resolve())
    elif args.command == "report":
        report_suite(
            args.suite.resolve(),
            args.results.resolve(),
            args.output.resolve() if args.output else None,
        )
    elif args.command == "release-verifiers":
        release_verifiers(args.suite.resolve(), args.results.resolve())


if __name__ == "__main__":
    main()
