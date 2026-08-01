from __future__ import annotations

import json
import subprocess

import pytest

from benchmarks.resume_quality_v3 import runner
from benchmarks.resume_quality_v3.context import count_tokens, trim_suffix
from benchmarks.resume_quality_v3.protocol import MODEL_CONFIG, REVIEW_LABELS


def test_protocol_manifest_binds_all_result_affecting_surfaces():
    manifest = runner._protocol_manifest()
    paths = set(manifest["file_hashes"])
    required = {
        "benchmarks/resume_quality_v3/runner.py",
        "benchmarks/resume_quality_v3/protocol.py",
        "benchmarks/resume_quality_v3/context.py",
        "benchmarks/resume_quality/case_specs.py",
        "benchmarks/resume_quality/runner.py",
        "tokenmizer/checkpoints/manager.py",
        "tokenmizer/core/tokenizer.py",
        "tokenmizer/graph_memory/graph.py",
    }
    assert required <= paths
    assert manifest["prompt_texts"]["continuation"]
    assert manifest["prompt_texts"]["summary"]
    assert manifest["model_config"] == MODEL_CONFIG

    changed = json.loads(json.dumps(manifest))
    changed["file_hashes"]["benchmarks/resume_quality_v3/runner.py"] = "0" * 64
    assert runner._protocol_hash(changed) != runner._protocol_hash(manifest)


def test_fixed_counter_applies_to_raw_tail_and_summary_budget():
    text = "alpha beta gamma delta " * 100
    tail = trim_suffix(text, 17)
    assert 0 < count_tokens(tail) <= 17
    assert MODEL_CONFIG["token_counter"] == "tiktoken:cl100k_base"
    assert MODEL_CONFIG["token_counter_version"] == "0.13.0"


def test_exact_model_provider_cache_and_execution_config_are_pinned():
    runner._validate_model_config(MODEL_CONFIG)
    assert MODEL_CONFIG["provider"] == "openrouter"
    assert MODEL_CONFIG["model"] == "qwen/qwen3-coder"
    assert MODEL_CONFIG["base_url"] == "https://openrouter.ai/api"
    assert MODEL_CONFIG["provider_endpoint"] == "deepinfra/turbo"
    assert MODEL_CONFIG["provider_allow_fallbacks"] is False
    assert MODEL_CONFIG["provider_require_parameters"] is True
    assert MODEL_CONFIG["stream"] is False
    assert MODEL_CONFIG["single_worker"] is True
    assert MODEL_CONFIG["parallel_runs"] is False
    assert MODEL_CONFIG["semantic_cache"] is False
    assert MODEL_CONFIG["response_cache"] is False
    assert MODEL_CONFIG["provider_prompt_cache"] is False
    assert MODEL_CONFIG["cross_run_reuse"] is False


def test_transport_dry_run_preserves_structure_without_network():
    result = runner._transport_dry_test()
    assert result["status"] == "pass"
    assert all(result["checks"].values())


def test_constructed_continuation_request_contains_context_and_exact_prompts(tmp_path):
    context = "Decided: keep the current safe architecture"
    request = runner._continuation_request(context)
    state = tmp_path / "state"
    repo = tmp_path / "repo"
    suite = tmp_path / "suite"
    state.mkdir()
    repo.mkdir()
    suite.mkdir()
    args = runner._base_claude_args(20, 5.0)
    args.extend(
        [
            "--settings",
            runner._agent_settings(suite, repo, state),
            "--permission-mode",
            MODEL_CONFIG["claude_permission_mode"],
            "--tools",
            request["tools"],
            "--append-system-prompt",
            request["system"],
            request["user"],
        ]
    )
    runner._assert_cli_invocation(args, request)
    assert context in request["system"]


def _minimal_summary_suite(tmp_path, budget: int):
    suite = tmp_path / "suite"
    case = suite / "cases" / "case-one"
    case.mkdir(parents=True)
    (suite / "protocol.snapshot.json").write_text(
        json.dumps({"cases": [{"case_id": "case-one"}]}), encoding="utf-8"
    )
    (case / "metadata.json").write_text(
        json.dumps({"resume_token_budget": budget}), encoding="utf-8"
    )
    (case / "transcript.txt").write_text("[user]\ncontinue the task", encoding="utf-8")
    return suite


def test_summary_retries_without_truncating(monkeypatch, tmp_path):
    accepted = "short supported handoff"
    budget = count_tokens(accepted)
    suite = _minimal_summary_suite(tmp_path, budget)
    lock = {
        "protocol_hash_v3": "v3-test",
        "model_config": MODEL_CONFIG,
        "claude_code_version": "test",
    }
    outputs = iter(["over budget " * 100, "still over " * 100, accepted])

    monkeypatch.setattr(runner, "_load_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(runner, "_print_model_config", lambda _config: None)

    def fake_call(*_args, **_kwargs):
        payload = {
            "result": next(outputs),
            "model": MODEL_CONFIG["model"],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    monkeypatch.setattr(runner, "_execute_model_call", fake_call)
    runner.summarize_suite(suite)

    stored = (suite / "cases/case-one/contexts/strong_summary.txt").read_text()
    summary_lock = json.loads((suite / "summaries.lock.json").read_text())
    assert stored == accepted
    assert len(summary_lock["cases"]["case-one"]["attempts"]) == 3
    assert count_tokens(stored) <= budget


def test_summary_marks_arm_invalid_after_three_oversized_attempts(monkeypatch, tmp_path):
    suite = _minimal_summary_suite(tmp_path, 2)
    lock = {
        "protocol_hash_v3": "v3-test",
        "model_config": MODEL_CONFIG,
        "claude_code_version": "test",
    }
    monkeypatch.setattr(runner, "_load_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(runner, "_print_model_config", lambda _config: None)

    def fake_call(*_args, **_kwargs):
        payload = {"result": "over budget " * 100, "model": MODEL_CONFIG["model"]}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    monkeypatch.setattr(runner, "_execute_model_call", fake_call)
    with pytest.raises(RuntimeError, match="invalid after three attempts"):
        runner.summarize_suite(suite)

    summary_lock = json.loads((suite / "summaries.lock.json").read_text())
    assert summary_lock["valid"] is False
    assert summary_lock["cases"]["case-one"]["status"] == "invalid_over_budget"
    assert not (suite / "cases/case-one/contexts/strong_summary.txt").exists()


def test_single_run_lock_is_exclusive(tmp_path):
    with runner._single_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another protocol v3 run is active"):
            with runner._single_run_lock(tmp_path):
                pass


def test_review_packet_is_blind_and_pending(tmp_path):
    packet_path = runner._write_review_packet(
        tmp_path,
        "run-1",
        "final response",
        {"current_choice": "cookies"},
        {"git_diff": ""},
    )
    packet = json.loads(packet_path.read_text())
    assert packet["status"] == "pending_review"
    assert "method" not in packet
    assert "case_id" not in packet
    assert set(packet["review_instructions"]["required_labels"]) == set(REVIEW_LABELS)


def test_human_review_computes_unsupported_metric_only_after_attestation(tmp_path):
    result_path = tmp_path / "case/method/result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"unsupported_memory_review": {"status": "pending_review"}}),
        encoding="utf-8",
    )
    (tmp_path / "private_review_index.json").write_text(
        json.dumps({"run-1": {"result": "case/method/result.json"}}), encoding="utf-8"
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "reviewer_id": "reviewer-a",
                "all_memory_claims_labeled": True,
                "claims": [
                    {"claim": "Used cookies", "label": "supported"},
                    {"claim": "Used Redis", "label": "unsupported"},
                    {"claim": "Tests pass", "label": "not_a_memory_claim"},
                ],
            }
        ),
        encoding="utf-8",
    )

    metric = runner.record_review(tmp_path, "run-1", review_path)

    assert metric == {
        "status": "reviewed",
        "unsupported_memory_count": 1,
        "has_unsupported_memory": True,
        "reviewer_id": "reviewer-a",
    }


def test_freeze_writes_bound_manifest_and_zero_call_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_claude_version", lambda: "claude-test-v3")
    monkeypatch.setattr(runner, "_git_sha", lambda: "c" * 40)
    suite = tmp_path / "suite-v3"

    runner.freeze_suite(suite)
    lock = runner._load_lock(suite)
    report = json.loads((suite / "validation_report.json").read_text())

    assert lock["protocol_version"] == "3.0.0"
    assert lock["protocol_hash_v3"] == (suite / "PROTOCOL_SHA256_V3").read_text().strip()
    assert json.loads((suite / "protocol.manifest.json").read_text()) == runner._protocol_manifest()
    assert report["status"] == "pass"
    assert report["paid_model_calls_executed"] == 0
    assert report["summaries_executed"] == 0
    assert report["continuations_executed"] == 0

