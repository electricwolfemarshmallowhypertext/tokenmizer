"""OpenRouter transport guard for protocol v3."""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _route_body(body: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Inject only the OpenRouter fields Claude Code cannot express."""
    if body.get("model") != config["model"]:
        raise RuntimeError("transport received a model other than the pinned model")
    if any(field in body for field in ("models", "fallbacks", "route")):
        raise RuntimeError("model fallback or dynamic routing is forbidden")
    if _contains_key(body, "cache_control"):
        raise RuntimeError("provider prompt caching is forbidden")

    routed = deepcopy(body)
    routed["stream"] = False
    routed["provider"] = {
        "only": [config["provider_endpoint"]],
        "order": [config["provider_endpoint"]],
        "allow_fallbacks": config["provider_allow_fallbacks"],
        "require_parameters": config["provider_require_parameters"],
    }
    return routed


def _provider_name(response: dict[str, Any]) -> str | None:
    metadata = response.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("provider_name", "provider", "provider_slug"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    routing = metadata.get("routing")
    if isinstance(routing, dict):
        for key in ("provider_name", "provider", "provider_slug"):
            value = routing.get(key)
            if isinstance(value, str) and value.strip():
                return value
    endpoints = metadata.get("endpoints")
    if isinstance(endpoints, dict):
        available = endpoints.get("available")
        if isinstance(available, list):
            for endpoint in available:
                if isinstance(endpoint, dict) and endpoint.get("selected") is True:
                    value = endpoint.get("provider")
                    if isinstance(value, str) and value.strip():
                        return value
    attempts = metadata.get("attempts")
    if isinstance(attempts, list):
        successful = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("status") == 200
        ]
        if len(successful) == 1:
            value = successful[0].get("provider")
            if isinstance(value, str) and value.strip():
                return value
    return None


def _provider_matches(actual: str, expected_slug: str) -> bool:
    actual_normalized = actual.lower().replace(" ", "").replace("_", "-")
    expected_base = expected_slug.split("/", 1)[0].lower().replace("-", "")
    return expected_base in actual_normalized.replace("-", "")


def send_openrouter_message(
    client: httpx.Client,
    body: dict[str, Any],
    api_key: str,
    config: dict[str, Any],
) -> tuple[httpx.Response, dict[str, Any]]:
    """Send one non-streaming, exact-model, exact-provider Messages request."""
    routed = _route_body(body, config)
    response = client.post(
        f"{config['base_url'].rstrip('/')}/v1/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        },
        json=routed,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter response was not a JSON object")
    if payload.get("model") != config["model"]:
        raise RuntimeError("OpenRouter used a model other than the pinned model")
    if not isinstance(payload.get("usage"), dict):
        raise RuntimeError("OpenRouter did not return provider usage")

    provider = _provider_name(payload)
    if provider is None:
        raise RuntimeError("OpenRouter did not return provider routing metadata")
    if not _provider_matches(provider, config["provider_endpoint"]):
        raise RuntimeError(f"OpenRouter used unexpected provider {provider!r}")

    audit = {
        "request": routed,
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "provider_name": provider,
        "provider_reported_usage": payload.get("usage"),
        "system_preserved": routed.get("system") == body.get("system"),
        "messages_preserved": routed.get("messages") == body.get("messages"),
        "tools_preserved": routed.get("tools") == body.get("tools"),
        "streaming_off": routed.get("stream") is False,
        "fallbacks_disabled": routed["provider"]["allow_fallbacks"] is False,
    }
    return response, audit


class OpenRouterRouteGuard:
    """Local Claude Code endpoint that pins OpenRouter routing per request."""

    def __init__(self, api_key: str, config: dict[str, Any], audit_path: Path):
        self.api_key = api_key
        self.config = config
        self.audit_path = audit_path
        self.audits: list[dict[str, Any]] = []
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "OpenRouterRouteGuard":
        guard = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                try:
                    if self.path.split("?", 1)[0] != "/v1/messages":
                        raise RuntimeError(f"unexpected Claude transport path: {self.path}")
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise RuntimeError("Claude transport body was not an object")
                    with httpx.Client(timeout=guard.config["continuation_timeout_seconds"]) as client:
                        response, audit = send_openrouter_message(
                            client, body, guard.api_key, guard.config
                        )
                    guard.audits.append(audit)
                    content = response.content
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as exc:  # pragma: no cover - exercised by integration
                    content = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("route guard is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(self.audits, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
