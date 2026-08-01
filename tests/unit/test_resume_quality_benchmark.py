import json
import subprocess

from benchmarks.resume_quality import runner
from benchmarks.resume_quality.case_specs import CASES
from benchmarks.resume_quality.context import render_transcript, trim_prefix, trim_suffix
from tokenmizer.core.tokenizer import count_tokens


EXPECTED_CASES = {
    "auth_storage",
    "database_concurrency",
    "api_compatibility",
    "realtime_transport",
    "schema_freeze",
    "private_data",
    "generated_code",
    "payment_retries",
    "time_handling",
    "runtime_constraint",
}


def test_ten_cases_cover_required_shape_without_artificial_markers():
    assert {case.case_id for case in CASES} == EXPECTED_CASES
    assert len(CASES) == 10
    for case in CASES:
        transcript = render_transcript(list(case.transcript))
        assert "Decided:" not in transcript
        assert "Completed:" not in transcript
        assert "Fixed:" not in transcript
        assert case.earlier_choice
        assert case.current_choice
        assert case.reason
        assert case.pending_task
        assert case.allowed_changes
        assert case.protected_files


def test_all_fixture_python_and_hidden_verifiers_compile():
    required_tests = {f"test_{name}" for name in runner.STRICT_CHECKS[:5]}
    for case in CASES:
        for relative, source in case.repo_files.items():
            if relative.endswith(".py"):
                compile(source, f"{case.case_id}/{relative}", "exec")
        compile(case.hidden_tests, f"{case.case_id}/test_hidden.py", "exec")
        assert required_tests <= {
            line.split("(", 1)[0].removeprefix("def ")
            for line in case.hidden_tests.splitlines()
            if line.startswith("def test_")
        }


def test_budget_trimming_never_exceeds_limit():
    text = "alpha beta gamma delta " * 100
    for trim in (trim_prefix, trim_suffix):
        result = trim(text, 17, "gpt-4o")
        assert count_tokens(result, "gpt-4o") <= 17
        assert result


def test_agent_settings_fail_closed_and_deny_frozen_state(tmp_path):
    suite = tmp_path / "suite"
    repo = tmp_path / "worktree"
    settings = json.loads(runner._agent_settings(suite, repo))
    assert settings["sandbox"]["failIfUnavailable"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False
    assert str(suite.resolve()) in settings["sandbox"]["filesystem"]["denyRead"]
    assert str(repo.resolve()) in settings["sandbox"]["filesystem"]["allowRead"]


def test_freeze_hashes_protocol_before_live_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_claude_version", lambda: "claude-test")
    monkeypatch.setattr(runner, "_git_sha", lambda: "a" * 40)
    suite = tmp_path / "suite"

    runner.freeze_suite(suite, "claude-test-model")
    lock = runner._load_lock(suite)

    assert lock["protocol_sha256"] == (suite / "PROTOCOL_SHA256").read_text().strip()
    assert len(list((suite / "cases").iterdir())) == 10
    for case in CASES:
        case_dir = suite / "cases" / case.case_id
        metadata = json.loads((case_dir / "metadata.json").read_text())
        budget = metadata["resume_token_budget"]
        assert metadata["tokenmizer_tokens"] == budget
        assert metadata["raw_tail_tokens"] <= budget
        assert (case_dir / "start_repo/.git").is_dir()
        assert (suite / "private" / case.case_id / "test_hidden.py").is_file()

    def fake_claude(args, **kwargs):
        assert args[0] == "claude"
        payload = {"result": "handoff fact " * 500, "usage": {"output_tokens": 500}}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(runner, "_run", fake_claude)
    runner.summarize_suite(suite, max_budget_usd=0.01)
    summary_lock = json.loads((suite / "summaries.lock.json").read_text())
    assert set(summary_lock["cases"]) == EXPECTED_CASES
    for case in CASES:
        summary = (suite / "cases" / case.case_id / "contexts/strong_summary.txt").read_text()
        budget = json.loads((suite / "cases" / case.case_id / "metadata.json").read_text())[
            "resume_token_budget"
        ]
        assert 0 < count_tokens(summary, "claude-test-model") <= budget


def test_frozen_artifact_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_claude_version", lambda: "claude-test")
    monkeypatch.setattr(runner, "_git_sha", lambda: "b" * 40)
    suite = tmp_path / "suite"
    runner.freeze_suite(suite, "claude-test-model")
    transcript = suite / "cases/auth_storage/transcript.txt"
    transcript.write_text(transcript.read_text() + "tampered", encoding="utf-8")

    try:
        runner._load_lock(suite)
    except RuntimeError as exc:
        assert "changed after lock" in str(exc)
    else:
        raise AssertionError("tampered frozen suite was accepted")
