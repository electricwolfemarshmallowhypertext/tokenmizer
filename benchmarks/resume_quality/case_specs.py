"""Ten frozen interrupted coding cases.

The repositories are intentionally small so failures measure continuation quality,
not framework installation or network availability. Hidden verifiers are emitted
outside each agent worktree by the benchmark freezer.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    earlier_choice: str
    current_choice: str
    reason: str
    pending_task: str
    transcript: tuple[dict[str, str], ...]
    repo_files: dict[str, str]
    hidden_tests: str
    allowed_changes: tuple[str, ...]
    protected_files: tuple[str, ...]
    rationale_groups: tuple[tuple[str, ...], ...]


def _text(value: str) -> str:
    return dedent(value).lstrip()


def _conversation(
    *,
    goal: str,
    old: str,
    new: str,
    reason: str,
    completed: str,
    bug: str,
    fixed_in: str,
    constraint: str,
    pending: str,
    work_file: str,
) -> tuple[dict[str, str], ...]:
    # Natural developer language only. In particular, no benchmark-friendly
    # "Decided:", "Completed:", or "Fixed:" labels appear in these transcripts.
    turns = (
        ("user", f"We're building {goal}. Start with {old} and keep the first pass small."),
        ("assistant", f"I put the initial path together around {old}. The main work is in {work_file}."),
        ("user", f"The first pass exposed a problem: {reason}. We cannot ship while that remains true."),
        ("assistant", f"I switched from {old} to {new} because {reason}. The old path should not come back."),
        ("user", f"Please get the first usable slice in place, including {completed}."),
        ("assistant", f"That slice is working now. I added {completed} and kept the changes focused in {fixed_in}."),
        ("user", f"I found one regression while checking it: {bug}."),
        ("assistant", f"The regression is fixed in {fixed_in}; the test that reproduces {bug} is green now."),
        ("user", f"One production detail is easy to miss from the code: {constraint}. Treat that as a hard limit."),
        ("assistant", f"Understood. I checked {work_file} against that limit and left the protected pieces alone."),
        ("user", f"Next, continue with {pending}. Keep the current architecture and the regression fix intact."),
        ("assistant", f"I traced the remaining path in {work_file}. The scaffolding is there, but {pending} still needs implementation and verification."),
        ("user", f"Please finish {pending}, run the tests, and give me the handoff when it is solid."),
    )
    return tuple({"role": role, "content": content} for role, content in turns)


COMMON_PYPROJECT = _text(
    """
    [tool.pytest.ini_options]
    testpaths = ["tests"]
    pythonpath = ["."]
    """
)


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="auth_storage",
        title="Auth storage",
        earlier_choice="JWTs in localStorage",
        current_choice="server-issued httpOnly cookies",
        reason="a review showed that an XSS bug could read bearer tokens from localStorage",
        pending_task="the refresh and logout flow without adding browser token storage",
        transcript=_conversation(
            goal="a small authentication service for the customer portal",
            old="JWTs in localStorage",
            new="server-issued httpOnly cookies",
            reason="a review showed that an XSS bug could read bearer tokens from localStorage",
            completed="login cookie issuance",
            bug="the session cookie was missing the Secure flag",
            fixed_in="auth_session.py",
            constraint="browser JavaScript must never receive or persist either token",
            pending="the refresh and logout flow without adding browser token storage",
            work_file="auth_session.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "auth_session.py": _text(
                """
                import secrets


                def _cookie(name: str, value: str, max_age: int = 3600) -> str:
                    return f"{name}={value}; Max-Age={max_age}; Path=/; HttpOnly; Secure; SameSite=Lax"


                def login(user_id: str, refresh_store: dict[str, str]) -> dict[str, object]:
                    refresh = secrets.token_urlsafe(24)
                    refresh_store[refresh] = user_id
                    return {
                        "status": 200,
                        "headers": {"Set-Cookie": _cookie("refresh_token", refresh)},
                        "body": {"user_id": user_id},
                    }


                def refresh(cookies: dict[str, str], refresh_store: dict[str, str]) -> dict[str, object]:
                    raise NotImplementedError


                def logout(cookies: dict[str, str], refresh_store: dict[str, str]) -> dict[str, object]:
                    raise NotImplementedError
                """
            ),
            "tests/test_auth_session.py": _text(
                """
                from auth_session import login


                def test_login_uses_hardened_cookie_and_no_body_token():
                    store = {}
                    response = login("u-1", store)
                    cookie = response["headers"]["Set-Cookie"]
                    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie
                    assert "token" not in response["body"]
                    assert list(store.values()) == ["u-1"]
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from auth_session import login, logout, refresh


            def test_functional_tests_pass():
                store = {}
                first = login("u-1", store)
                old = next(iter(store))
                renewed = refresh({"refresh_token": old}, store)
                assert renewed["status"] == 200
                assert old not in store and len(store) == 1


            def test_unfinished_task_completed():
                store = {}
                login("u-2", store)
                token = next(iter(store))
                response = logout({"refresh_token": token}, store)
                assert response["status"] in (200, 204)
                assert not store
                assert "Max-Age=0" in response["headers"]["Set-Cookie"]


            def test_current_decision_followed():
                store = {}
                login("u-3", store)
                response = refresh({"refresh_token": next(iter(store))}, store)
                assert "HttpOnly" in response["headers"]["Set-Cookie"]
                assert "token" not in response.get("body", {})


            def test_superseded_decision_absent():
                text = pathlib.Path("auth_session.py").read_text().lower()
                assert "localstorage" not in text


            def test_prior_fix_intact():
                assert "Secure" in login("u-4", {})["headers"]["Set-Cookie"]
            """
        ),
        allowed_changes=("auth_session.py",),
        protected_files=("pyproject.toml", "tests/test_auth_session.py"),
        rationale_groups=(("xss", "cross-site scripting"), ("localstorage",)),
    ),
    CaseSpec(
        case_id="database_concurrency",
        title="Database concurrency",
        earlier_choice="SQLite for the job queue",
        current_choice="PostgreSQL row locking",
        reason="parallel workers repeatedly hit SQLite database-locked failures",
        pending_task="safe background-job claiming so two workers cannot claim the same row",
        transcript=_conversation(
            goal="a background worker that drains queued export jobs",
            old="SQLite for the job queue",
            new="PostgreSQL row locking",
            reason="parallel workers repeatedly hit SQLite database-locked failures",
            completed="parameterized job insertion and status updates",
            bug="a quote in the payload broke the insert statement",
            fixed_in="jobs.py",
            constraint="workers run concurrently in separate processes and claiming must stay non-blocking",
            pending="safe background-job claiming so two workers cannot claim the same row",
            work_file="jobs.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "jobs.py": _text(
                """
                def enqueue(conn, payload: str) -> None:
                    conn.execute("INSERT INTO jobs (payload, status) VALUES (%s, %s)", (payload, "queued"))


                def mark_done(conn, job_id: int) -> None:
                    conn.execute("UPDATE jobs SET status = %s WHERE id = %s", ("done", job_id))


                def claim_job(conn, worker_id: str):
                    raise NotImplementedError
                """
            ),
            "tests/test_jobs.py": _text(
                """
                from jobs import enqueue


                class Connection:
                    def __init__(self): self.calls = []
                    def execute(self, sql, params): self.calls.append((sql, params))


                def test_enqueue_is_parameterized():
                    conn = Connection()
                    enqueue(conn, "customer's export")
                    assert conn.calls[0][1] == ("customer's export", "queued")
                    assert "customer's export" not in conn.calls[0][0]
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from jobs import claim_job, enqueue


            class Result:
                def fetchone(self): return {"id": 7, "payload": "x", "status": "running"}


            class Connection:
                def __init__(self): self.calls = []
                def execute(self, sql, params=()): self.calls.append((sql, params)); return Result()


            def test_functional_tests_pass():
                conn = Connection()
                row = claim_job(conn, "worker-a")
                assert row["id"] == 7


            def test_unfinished_task_completed():
                conn = Connection(); claim_job(conn, "worker-a")
                sql = " ".join(call[0] for call in conn.calls).upper()
                assert "UPDATE" in sql and "RETURNING" in sql


            def test_current_decision_followed():
                conn = Connection(); claim_job(conn, "worker-a")
                sql = " ".join(call[0] for call in conn.calls).upper()
                assert "FOR UPDATE" in sql and "SKIP LOCKED" in sql


            def test_superseded_decision_absent():
                assert "sqlite" not in pathlib.Path("jobs.py").read_text().lower()


            def test_prior_fix_intact():
                conn = Connection(); enqueue(conn, "x' OR 1=1")
                assert "x' OR 1=1" not in conn.calls[0][0]
            """
        ),
        allowed_changes=("jobs.py",),
        protected_files=("pyproject.toml", "tests/test_jobs.py"),
        rationale_groups=(("locked", "locking", "lock failures"), ("sqlite",)),
    ),
    CaseSpec(
        case_id="api_compatibility",
        title="API compatibility",
        earlier_choice="renaming the public status field to state",
        current_choice="preserving the status wire field",
        reason="the partner client cannot migrate until its next quarterly release",
        pending_task="a detail endpoint while keeping the existing response contract",
        transcript=_conversation(
            goal="a versioned order API used by our dashboard and a partner integration",
            old="renaming the public status field to state",
            new="preserving the status wire field",
            reason="the partner client cannot migrate until its next quarterly release",
            completed="the order-list serializer",
            bug="unknown internal states leaked a null onto the wire",
            fixed_in="orders_api.py",
            constraint="the partner treats any missing status key as a fatal protocol error",
            pending="a detail endpoint while keeping the existing response contract",
            work_file="orders_api.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "orders_api.py": _text(
                """
                def serialize_order(order: dict) -> dict:
                    status = order.get("state") or "unknown"
                    return {"id": order["id"], "status": status}


                def list_orders(rows: list[dict]) -> dict:
                    return {"orders": [serialize_order(row) for row in rows]}


                def get_order_detail(order_id: str, rows: list[dict]) -> dict:
                    raise NotImplementedError
                """
            ),
            "tests/test_orders_api.py": _text(
                """
                from orders_api import list_orders, serialize_order


                def test_list_keeps_wire_status_and_normalizes_missing_state():
                    assert list_orders([{"id": "o1"}]) == {"orders": [{"id": "o1", "status": "unknown"}]}
                    assert "state" not in serialize_order({"id": "o2", "state": "paid"})
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from orders_api import get_order_detail, serialize_order


            def test_functional_tests_pass():
                assert get_order_detail("o1", [{"id": "o1", "state": "paid", "total": 20}])["status"] == 200


            def test_unfinished_task_completed():
                response = get_order_detail("missing", [])
                assert response["status"] == 404


            def test_current_decision_followed():
                body = get_order_detail("o1", [{"id": "o1", "state": "paid"}])["body"]
                assert body["status"] == "paid" and "state" not in body


            def test_superseded_decision_absent():
                assert '"state":' not in pathlib.Path("orders_api.py").read_text()


            def test_prior_fix_intact():
                assert serialize_order({"id": "o2"})["status"] == "unknown"
            """
        ),
        allowed_changes=("orders_api.py",),
        protected_files=("pyproject.toml", "tests/test_orders_api.py"),
        rationale_groups=(("partner",), ("quarter", "migration", "migrate")),
    ),
    CaseSpec(
        case_id="realtime_transport",
        title="Realtime transport",
        earlier_choice="WebSockets",
        current_choice="server-sent events",
        reason="the serverless runtime terminates WebSocket upgrades",
        pending_task="reconnect and event-resume behavior using Last-Event-ID",
        transcript=_conversation(
            goal="live deployment progress updates for the control panel",
            old="WebSockets",
            new="server-sent events",
            reason="the serverless runtime terminates WebSocket upgrades",
            completed="SSE event formatting and heartbeat comments",
            bug="multiline payloads produced malformed event frames",
            fixed_in="event_stream.py",
            constraint="the deployment platform only permits ordinary streaming HTTP responses",
            pending="reconnect and event-resume behavior using Last-Event-ID",
            work_file="event_stream.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "event_stream.py": _text(
                """
                def encode_event(event: dict) -> str:
                    lines = [f"id: {event['id']}", f"event: {event.get('type', 'message')}"]
                    lines.extend(f"data: {line}" for line in str(event["data"]).splitlines() or [""])
                    return "\\n".join(lines) + "\\n\\n"


                def heartbeat() -> str:
                    return ": keep-alive\\n\\n"


                def resume_events(events: list[dict], last_event_id: str | None) -> list[str]:
                    raise NotImplementedError
                """
            ),
            "tests/test_event_stream.py": _text(
                """
                from event_stream import encode_event, heartbeat


                def test_multiline_data_and_heartbeat_are_valid_sse():
                    frame = encode_event({"id": "2", "data": "a\\nb"})
                    assert "data: a\\ndata: b" in frame and frame.endswith("\\n\\n")
                    assert heartbeat().startswith(":")
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from event_stream import resume_events


            EVENTS = [{"id": "1", "data": "a"}, {"id": "2", "data": "b"}, {"id": "3", "data": "c"}]


            def test_functional_tests_pass():
                assert len(resume_events(EVENTS, None)) == 3


            def test_unfinished_task_completed():
                frames = resume_events(EVENTS, "1")
                assert len(frames) == 2 and frames[0].startswith("id: 2")


            def test_current_decision_followed():
                assert all("data:" in frame for frame in resume_events(EVENTS, "2"))


            def test_superseded_decision_absent():
                text = pathlib.Path("event_stream.py").read_text().lower()
                assert "websocket" not in text and "upgrade" not in text


            def test_prior_fix_intact():
                assert "data: a\\ndata: b" in __import__("event_stream").encode_event({"id": "x", "data": "a\\nb"})
            """
        ),
        allowed_changes=("event_stream.py",),
        protected_files=("pyproject.toml", "tests/test_event_stream.py"),
        rationale_groups=(("serverless",), ("websocket", "upgrade")),
    ),
    CaseSpec(
        case_id="schema_freeze",
        title="Schema freeze",
        earlier_choice="adding an enabled column to the accounts table",
        current_choice="an adapter backed by deployment configuration",
        reason="launch-week database migrations are frozen",
        pending_task="the beta feature flag without changing the database schema",
        transcript=_conversation(
            goal="a beta entitlement check for account-scoped features",
            old="adding an enabled column to the accounts table",
            new="an adapter backed by deployment configuration",
            reason="launch-week database migrations are frozen",
            completed="configuration parsing and account normalization",
            bug="mixed-case account IDs failed to match configuration",
            fixed_in="feature_flags.py",
            constraint="schema.sql is release-locked and cannot change before launch",
            pending="the beta feature flag without changing the database schema",
            work_file="feature_flags.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "schema.sql": "CREATE TABLE accounts (id TEXT PRIMARY KEY, name TEXT NOT NULL);\n",
            "feature_flags.py": _text(
                """
                def parse_accounts(value: str) -> set[str]:
                    return {part.strip().lower() for part in value.split(",") if part.strip()}


                def beta_enabled(account_id: str, config: dict[str, str]) -> bool:
                    raise NotImplementedError
                """
            ),
            "tests/test_feature_flags.py": _text(
                """
                from feature_flags import parse_accounts


                def test_account_normalization_regression():
                    assert parse_accounts(" ACME, beta-two ") == {"acme", "beta-two"}
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from feature_flags import beta_enabled, parse_accounts


            def test_functional_tests_pass():
                assert beta_enabled("acme", {"BETA_ACCOUNTS": "acme,other"}) is True


            def test_unfinished_task_completed():
                assert beta_enabled("none", {"BETA_ACCOUNTS": "acme"}) is False


            def test_current_decision_followed():
                assert beta_enabled("ACME", {"BETA_ACCOUNTS": "acme"}) is True


            def test_superseded_decision_absent():
                assert "enabled" not in pathlib.Path("schema.sql").read_text().lower()


            def test_prior_fix_intact():
                assert parse_accounts("AcMe") == {"acme"}
            """
        ),
        allowed_changes=("feature_flags.py",),
        protected_files=("schema.sql", "pyproject.toml", "tests/test_feature_flags.py"),
        rationale_groups=(("migration",), ("frozen", "freeze", "launch")),
    ),
    CaseSpec(
        case_id="private_data",
        title="Private data",
        earlier_choice="a hosted embedding API",
        current_choice="the in-process local embedder",
        reason="regulated documents cannot leave the environment",
        pending_task="document indexing without an outbound embedding provider",
        transcript=_conversation(
            goal="semantic lookup over regulated support documents",
            old="a hosted embedding API",
            new="the in-process local embedder",
            reason="regulated documents cannot leave the environment",
            completed="deterministic local embeddings and text normalization",
            bug="empty documents caused a division-by-zero error",
            fixed_in="indexing.py",
            constraint="the indexing process runs with egress disabled and must stay deterministic",
            pending="document indexing without an outbound embedding provider",
            work_file="indexing.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "indexing.py": _text(
                """
                import hashlib


                def local_embed(text: str) -> tuple[float, ...]:
                    normalized = " ".join(text.lower().split())
                    if not normalized:
                        return (0.0, 0.0, 0.0, 0.0)
                    digest = hashlib.sha256(normalized.encode()).digest()[:4]
                    return tuple(round(value / 255.0, 6) for value in digest)


                def index_documents(documents: list[dict]) -> dict[str, tuple[float, ...]]:
                    raise NotImplementedError
                """
            ),
            "tests/test_indexing.py": _text(
                """
                from indexing import local_embed


                def test_local_embed_is_deterministic_and_handles_empty_text():
                    assert local_embed(" Hello  world ") == local_embed("hello world")
                    assert local_embed("") == (0.0, 0.0, 0.0, 0.0)
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from indexing import index_documents, local_embed


            DOCS = [{"id": "a", "text": "Alpha"}, {"id": "b", "text": "Beta"}]


            def test_functional_tests_pass():
                assert index_documents(DOCS)["a"] == local_embed("Alpha")


            def test_unfinished_task_completed():
                assert set(index_documents(DOCS)) == {"a", "b"}


            def test_current_decision_followed():
                assert index_documents(DOCS) == index_documents(DOCS)


            def test_superseded_decision_absent():
                text = pathlib.Path("indexing.py").read_text().lower()
                assert all(word not in text for word in ("requests", "http://", "https://", "openai"))


            def test_prior_fix_intact():
                assert local_embed("") == (0.0, 0.0, 0.0, 0.0)
            """
        ),
        allowed_changes=("indexing.py",),
        protected_files=("pyproject.toml", "tests/test_indexing.py"),
        rationale_groups=(("cannot leave", "private", "regulated"), ("environment", "egress", "outbound")),
    ),
    CaseSpec(
        case_id="generated_code",
        title="Generated code",
        earlier_choice="editing generated_client.py by hand",
        current_choice="changing schema.json and regenerating",
        reason="manual generated-client edits disappear on the next generation run",
        pending_task="the priority field through the schema and regenerated client",
        transcript=_conversation(
            goal="a generated Python client for the task service schema",
            old="editing generated_client.py by hand",
            new="changing schema.json and regenerating",
            reason="manual generated-client edits disappear on the next generation run",
            completed="the generator and stable field ordering",
            bug="generation changed output order between runs",
            fixed_in="generate_client.py",
            constraint="generated_client.py must be reproducible byte-for-byte from schema.json",
            pending="the priority field through the schema and regenerated client",
            work_file="schema.json",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "schema.json": '{\n  "fields": ["id", "title"]\n}\n',
            "generate_client.py": _text(
                """
                import json
                from pathlib import Path


                def render(schema: dict) -> str:
                    fields = sorted(schema["fields"])
                    body = "\\n".join(f"    {field}: str" for field in fields)
                    return "# Generated; edit schema.json instead.\\nfrom dataclasses import dataclass\\n\\n@dataclass\\nclass Task:\\n" + body + "\\n"


                if __name__ == "__main__":
                    root = Path(__file__).parent
                    schema = json.loads((root / "schema.json").read_text())
                    (root / "generated_client.py").write_text(render(schema))
                """
            ),
            "generated_client.py": _text(
                """
                # Generated; edit schema.json instead.
                from dataclasses import dataclass

                @dataclass
                class Task:
                    id: str
                    title: str
                """
            ),
            "tests/test_generated_client.py": _text(
                """
                import json
                from pathlib import Path
                from generate_client import render


                def test_generated_output_is_reproducible():
                    root = Path(__file__).parents[1]
                    assert (root / "generated_client.py").read_text() == render(json.loads((root / "schema.json").read_text()))
                """
            ),
        },
        hidden_tests=_text(
            """
            import json
            import pathlib
            from generate_client import render
            from generated_client import Task


            def test_functional_tests_pass():
                assert Task(id="1", title="x", priority="high").priority == "high"


            def test_unfinished_task_completed():
                assert "priority" in json.loads(pathlib.Path("schema.json").read_text())["fields"]


            def test_current_decision_followed():
                assert pathlib.Path("generated_client.py").read_text() == render(json.loads(pathlib.Path("schema.json").read_text()))


            def test_superseded_decision_absent():
                assert "edit schema.json instead" in pathlib.Path("generated_client.py").read_text().lower()


            def test_prior_fix_intact():
                fields = [line.strip() for line in pathlib.Path("generated_client.py").read_text().splitlines() if ": str" in line]
                assert fields == sorted(fields)
            """
        ),
        allowed_changes=("schema.json", "generated_client.py"),
        protected_files=("generate_client.py", "pyproject.toml", "tests/test_generated_client.py"),
        rationale_groups=(("generated", "generation"), ("overwrite", "disappear", "regenerate")),
    ),
    CaseSpec(
        case_id="payment_retries",
        title="Payment retries",
        earlier_choice="blindly retrying charge creation",
        current_choice="stable idempotency keys",
        reason="a timeout followed by a retry produced duplicate charges",
        pending_task="webhook retry handling that preserves the original idempotency key",
        transcript=_conversation(
            goal="reliable payment processing around provider webhooks",
            old="blindly retrying charge creation",
            new="stable idempotency keys",
            reason="a timeout followed by a retry produced duplicate charges",
            completed="idempotent initial charge creation",
            bug="replayed delivery IDs were processed twice",
            fixed_in="payments.py",
            constraint="one logical event must use the same key across every delivery attempt",
            pending="webhook retry handling that preserves the original idempotency key",
            work_file="payments.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "payments.py": _text(
                """
                def charge(provider, event_id: str, amount: int):
                    return provider.create_charge(amount=amount, idempotency_key=f"charge:{event_id}")


                def handle_webhook(provider, event: dict, seen: set[str]):
                    if event["delivery_id"] in seen:
                        return {"status": "duplicate"}
                    seen.add(event["delivery_id"])
                    return charge(provider, event["event_id"], event["amount"])


                def retry_webhook(provider, event: dict, attempts: int = 3):
                    raise NotImplementedError
                """
            ),
            "tests/test_payments.py": _text(
                """
                from payments import handle_webhook


                class Provider:
                    def __init__(self): self.calls = []
                    def create_charge(self, **kwargs): self.calls.append(kwargs); return kwargs


                def test_duplicate_delivery_is_ignored():
                    provider, seen = Provider(), set()
                    event = {"delivery_id": "d1", "event_id": "e1", "amount": 10}
                    handle_webhook(provider, event, seen); handle_webhook(provider, event, seen)
                    assert len(provider.calls) == 1
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from payments import charge, retry_webhook


            class Provider:
                def __init__(self): self.calls = []
                def create_charge(self, **kwargs):
                    self.calls.append(kwargs)
                    if len(self.calls) < 3: raise TimeoutError
                    return kwargs


            def test_functional_tests_pass():
                provider = Provider(); result = retry_webhook(provider, {"event_id": "e1", "amount": 10})
                assert result["amount"] == 10


            def test_unfinished_task_completed():
                provider = Provider(); retry_webhook(provider, {"event_id": "e1", "amount": 10})
                assert len(provider.calls) == 3


            def test_current_decision_followed():
                provider = Provider(); retry_webhook(provider, {"event_id": "e1", "amount": 10})
                assert {call["idempotency_key"] for call in provider.calls} == {"charge:e1"}


            def test_superseded_decision_absent():
                text = pathlib.Path("payments.py").read_text().lower()
                assert "random" not in text and "uuid" not in text


            def test_prior_fix_intact():
                provider = Provider(); provider.create_charge = lambda **kwargs: kwargs
                assert charge(provider, "e2", 4)["idempotency_key"] == "charge:e2"
            """
        ),
        allowed_changes=("payments.py",),
        protected_files=("pyproject.toml", "tests/test_payments.py"),
        rationale_groups=(("duplicate",), ("charge",), ("timeout", "retry")),
    ),
    CaseSpec(
        case_id="time_handling",
        title="Time handling",
        earlier_choice="storing local timestamps",
        current_choice="UTC-aware timestamps",
        reason="the daylight-saving transition scheduled the same job twice",
        pending_task="recurring schedules while keeping every persisted instant in UTC",
        transcript=_conversation(
            goal="a scheduler for recurring customer reports",
            old="storing local timestamps",
            new="UTC-aware timestamps",
            reason="the daylight-saving transition scheduled the same job twice",
            completed="UTC serialization and strict aware-datetime validation",
            bug="naive datetimes were silently treated as server local time",
            fixed_in="schedules.py",
            constraint="timezone names may be display metadata, but persisted run times are UTC instants",
            pending="recurring schedules while keeping every persisted instant in UTC",
            work_file="schedules.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "schedules.py": _text(
                """
                from datetime import datetime, timedelta, timezone


                def serialize_instant(value: datetime) -> str:
                    if value.tzinfo is None or value.utcoffset() is None:
                        raise ValueError("aware datetime required")
                    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


                def recurring_runs(start: datetime, every: timedelta, count: int) -> list[str]:
                    raise NotImplementedError
                """
            ),
            "tests/test_schedules.py": _text(
                """
                from datetime import datetime, timezone
                import pytest
                from schedules import serialize_instant


                def test_serialization_is_utc_and_rejects_naive_values():
                    assert serialize_instant(datetime(2026, 1, 1, tzinfo=timezone.utc)).endswith("Z")
                    with pytest.raises(ValueError): serialize_instant(datetime(2026, 1, 1))
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from datetime import datetime, timedelta, timezone
            from schedules import recurring_runs, serialize_instant


            START = datetime(2026, 3, 8, 7, tzinfo=timezone.utc)


            def test_functional_tests_pass():
                assert len(recurring_runs(START, timedelta(hours=24), 3)) == 3


            def test_unfinished_task_completed():
                assert recurring_runs(START, timedelta(hours=24), 2)[1] == "2026-03-09T07:00:00Z"


            def test_current_decision_followed():
                assert all(value.endswith("Z") for value in recurring_runs(START, timedelta(hours=24), 3))


            def test_superseded_decision_absent():
                text = pathlib.Path("schedules.py").read_text().lower()
                assert ".now(" not in text and ".astimezone()" not in text


            def test_prior_fix_intact():
                try: serialize_instant(datetime(2026, 1, 1))
                except ValueError: return
                assert False, "naive datetime accepted"
            """
        ),
        allowed_changes=("schedules.py",),
        protected_files=("pyproject.toml", "tests/test_schedules.py"),
        rationale_groups=(("daylight", "dst"), ("twice", "duplicate")),
    ),
    CaseSpec(
        case_id="runtime_constraint",
        title="Runtime constraint",
        earlier_choice="upgrading to transport API v3",
        current_choice="the supported transport API v2",
        reason="production is pinned to Python 3.10 and the v3 package requires a newer runtime",
        pending_task="request tracing through the transport v2 hook API",
        transcript=_conversation(
            goal="request tracing in a service with a pinned production image",
            old="upgrading to transport API v3",
            new="the supported transport API v2",
            reason="production is pinned to Python 3.10 and the v3 package requires a newer runtime",
            completed="the v2 transport wrapper and timeout forwarding",
            bug="zero-second timeouts were replaced by the default",
            fixed_in="transport_client.py",
            constraint="requirements.txt is image-locked and transport v2 only exposes event_hooks",
            pending="request tracing through the transport v2 hook API",
            work_file="transport_client.py",
        ),
        repo_files={
            "pyproject.toml": COMMON_PYPROJECT,
            "requirements.txt": "transport-lib==2.8.4\n",
            "transport_client.py": _text(
                """
                class TransportV2:
                    def __init__(self, timeout=30, event_hooks=None):
                        self.timeout = timeout
                        self.event_hooks = event_hooks or {}

                    def send(self, request):
                        for hook in self.event_hooks.get("request", []):
                            hook(request)
                        return {"status": 200, "request": request}


                def build_client(timeout=30):
                    return TransportV2(timeout=timeout)


                def build_traced_client(trace):
                    raise NotImplementedError
                """
            ),
            "tests/test_transport_client.py": _text(
                """
                from transport_client import build_client


                def test_zero_timeout_is_forwarded():
                    assert build_client(timeout=0).timeout == 0
                """
            ),
        },
        hidden_tests=_text(
            """
            import pathlib
            from transport_client import TransportV2, build_client, build_traced_client


            def test_functional_tests_pass():
                seen = []; client = build_traced_client(seen.append); client.send("req")
                assert seen == ["req"]


            def test_unfinished_task_completed():
                assert isinstance(build_traced_client(lambda request: None), TransportV2)


            def test_current_decision_followed():
                assert "request" in build_traced_client(lambda request: None).event_hooks


            def test_superseded_decision_absent():
                text = pathlib.Path("transport_client.py").read_text().lower()
                assert "clientconfig" not in text and "transportv3" not in text


            def test_prior_fix_intact():
                assert build_client(timeout=0).timeout == 0
            """
        ),
        allowed_changes=("transport_client.py",),
        protected_files=("requirements.txt", "pyproject.toml", "tests/test_transport_client.py"),
        rationale_groups=(("python 3.10", "runtime"), ("pinned",), ("v3", "newer")),
    ),
)


def get_case(case_id: str) -> CaseSpec:
    for case in CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)
