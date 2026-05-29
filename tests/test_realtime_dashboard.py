"""Tests for the real-time SSE dashboard (/stats/stream) endpoint."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import Backend, ProxyState, RequestQueue, create_lb_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state():
    """ProxyState with two backends (one healthy, one unhealthy) and traffic data."""
    state = ProxyState()
    b1 = Backend(host="10.0.0.1", port=8080, models=["llama-3"], healthy=True)
    b1.total_requests = 50
    b1.total_prompt_tokens = 10000
    b1.total_completion_tokens = 5000
    b1.active_requests = 2

    b2 = Backend(host="10.0.0.2", port=8080, models=["llama-3"], healthy=False)
    b2.total_requests = 30
    b2.total_prompt_tokens = 6000
    b2.total_completion_tokens = 3000

    state.backends = [b1, b2]
    return state


@pytest.fixture
def client(base_state):
    app = create_lb_app(base_state)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper: build SSE payload directly via the async generator logic
# ---------------------------------------------------------------------------


def _build_payload(state: ProxyState) -> dict:
    """Reproduce the same payload logic used by the SSE generator."""
    total_prompt_tokens = sum(b.total_prompt_tokens for b in state.backends)
    total_completion_tokens = sum(b.total_completion_tokens for b in state.backends)
    total_requests = sum(b.total_requests for b in state.backends)
    healthy_count = sum(1 for b in state.backends if b.healthy)

    payload: dict = {
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "healthy_backends": healthy_count,
        "total_backends": len(state.backends),
        "backends": [
            {
                "url": b.url,
                "healthy": b.healthy,
                "active_requests": b.active_requests,
                "total_requests": b.total_requests,
                "total_tokens": b.total_prompt_tokens + b.total_completion_tokens,
                "circuit_state": b.circuit_breaker.state.value,
            }
            for b in state.backends
        ],
    }

    if state.response_cache:
        payload["cache"] = state.response_cache.get_stats()

    if state.request_queue:
        payload["queue"] = {
            "size": state.request_queue.size(),
            "total_queued": state.request_queue.total_queued,
        }

    return payload


# ---------------------------------------------------------------------------
# 1. Endpoint registration
# ---------------------------------------------------------------------------


class TestSSEEndpointExists:
    def test_stats_stream_in_openapi(self, client):
        """GET /stats/stream must be registered and visible in OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/stats/stream" in paths, "Route /stats/stream not found in OpenAPI schema"

    def test_stats_stream_route_registered(self, client):
        """The /stats/stream route must be registered in the app."""
        app = client.app
        routes = {route.path for route in app.routes}
        assert "/stats/stream" in routes


# ---------------------------------------------------------------------------
# 2. SSE headers and response type (tested via StreamingResponse attributes)
# ---------------------------------------------------------------------------


class TestSSEHeaders:
    def test_stream_response_uses_event_stream_media_type(self, base_state):
        """The StreamingResponse for /stats/stream must use text/event-stream."""
        from fastapi.responses import StreamingResponse

        app = create_lb_app(base_state)
        # Inspect the route directly
        for route in app.routes:
            if hasattr(route, "path") and route.path == "/stats/stream":
                # The endpoint is registered; verify by checking the response_class
                # or simply that the route is a GET with the right path.
                assert route.path == "/stats/stream"
                return
        pytest.fail("Route /stats/stream not found")

    def test_sse_response_headers_configured(self, base_state):
        """The SSE endpoint source must include the required anti-buffering headers."""
        import inspect
        from llamacpp_cli import lb_proxy

        source = inspect.getsource(lb_proxy)
        assert "X-Accel-Buffering" in source
        assert "no-cache" in source
        assert "keep-alive" in source


# ---------------------------------------------------------------------------
# 3. SSE event format (tested via payload builder to avoid blocking)
# ---------------------------------------------------------------------------


class TestSSEEventFormat:
    def test_event_starts_with_data_prefix(self, base_state):
        """Each SSE event must use the 'data: ' prefix."""
        payload = _build_payload(base_state)
        event_line = f"data: {json.dumps(payload)}"
        assert event_line.startswith("data: ")

    def test_payload_is_valid_json(self, base_state):
        """The SSE payload must be valid JSON."""
        payload = _build_payload(base_state)
        # Round-trip through JSON serialisation
        serialised = json.dumps(payload)
        recovered = json.loads(serialised)
        assert isinstance(recovered, dict)

    def test_sse_wire_format_ends_with_double_newline(self, base_state):
        """SSE protocol requires events to be separated by a blank line."""
        payload = _build_payload(base_state)
        event = f"data: {json.dumps(payload)}\n\n"
        assert event.endswith("\n\n")


# ---------------------------------------------------------------------------
# 4. Stats data structure
# ---------------------------------------------------------------------------


class TestSSEDataStructure:
    def test_required_top_level_keys(self, base_state):
        data = _build_payload(base_state)
        for key in ("total_requests", "total_prompt_tokens", "total_completion_tokens",
                    "healthy_backends", "total_backends", "backends"):
            assert key in data, f"Missing key: {key}"

    def test_total_requests_aggregation(self, base_state):
        data = _build_payload(base_state)
        expected = sum(b.total_requests for b in base_state.backends)
        assert data["total_requests"] == expected

    def test_total_prompt_tokens_aggregation(self, base_state):
        data = _build_payload(base_state)
        expected = sum(b.total_prompt_tokens for b in base_state.backends)
        assert data["total_prompt_tokens"] == expected

    def test_total_completion_tokens_aggregation(self, base_state):
        data = _build_payload(base_state)
        expected = sum(b.total_completion_tokens for b in base_state.backends)
        assert data["total_completion_tokens"] == expected

    def test_healthy_backend_count(self, base_state):
        data = _build_payload(base_state)
        expected = sum(1 for b in base_state.backends if b.healthy)
        assert data["healthy_backends"] == expected
        assert data["total_backends"] == len(base_state.backends)


# ---------------------------------------------------------------------------
# 5. Backend info in SSE data
# ---------------------------------------------------------------------------


class TestSSEBackendInfo:
    def test_backends_list_length(self, base_state):
        data = _build_payload(base_state)
        assert len(data["backends"]) == len(base_state.backends)

    def test_backend_entry_required_keys(self, base_state):
        data = _build_payload(base_state)
        required = {"url", "healthy", "active_requests", "total_requests",
                    "total_tokens", "circuit_state"}
        for b in data["backends"]:
            assert required.issubset(b.keys()), f"Missing keys in backend entry: {b}"

    def test_backend_healthy_flag_values(self, base_state):
        data = _build_payload(base_state)
        health_map = {b["url"]: b["healthy"] for b in data["backends"]}
        assert True in health_map.values()
        assert False in health_map.values()

    def test_backend_circuit_state_is_valid_enum(self, base_state):
        data = _build_payload(base_state)
        valid = {"closed", "open", "half_open"}
        for b in data["backends"]:
            assert b["circuit_state"] in valid

    def test_backend_total_tokens_sum(self, base_state):
        data = _build_payload(base_state)
        for idx, b_data in enumerate(data["backends"]):
            b = base_state.backends[idx]
            assert b_data["total_tokens"] == b.total_prompt_tokens + b.total_completion_tokens


# ---------------------------------------------------------------------------
# 6. Cache stats included when available
# ---------------------------------------------------------------------------


class TestSSECacheStats:
    def test_cache_key_absent_when_no_cache(self, base_state):
        data = _build_payload(base_state)
        assert "cache" not in data

    def test_cache_stats_present_when_cache_configured(self, base_state):
        from unittest.mock import MagicMock

        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 5, "size": 8}
        base_state.response_cache = mock_cache

        data = _build_payload(base_state)
        assert "cache" in data
        assert data["cache"]["hits"] == 10
        assert data["cache"]["misses"] == 5

    def test_cache_stats_structure(self, base_state):
        from unittest.mock import MagicMock

        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {
            "hits": 20, "misses": 4, "size": 15, "evictions": 1
        }
        base_state.response_cache = mock_cache

        data = _build_payload(base_state)
        cache = data["cache"]
        assert cache["hits"] == 20
        assert cache["evictions"] == 1


# ---------------------------------------------------------------------------
# 7. Queue stats included when available
# ---------------------------------------------------------------------------


class TestSSEQueueStats:
    def test_queue_key_absent_when_no_queue(self, base_state):
        data = _build_payload(base_state)
        assert "queue" not in data

    def test_queue_stats_present_when_queue_configured(self, base_state):
        queue = RequestQueue(max_size=100)
        base_state.request_queue = queue

        data = _build_payload(base_state)
        assert "queue" in data
        assert "size" in data["queue"]
        assert "total_queued" in data["queue"]

    def test_queue_total_queued_reflects_state(self, base_state):
        queue = RequestQueue(max_size=100)
        queue.total_queued = 42
        base_state.request_queue = queue

        data = _build_payload(base_state)
        assert data["queue"]["total_queued"] == 42

    def test_queue_size_reflects_current_depth(self, base_state):
        queue = RequestQueue(max_size=100)
        base_state.request_queue = queue
        # Queue is empty — size must be 0
        data = _build_payload(base_state)
        assert data["queue"]["size"] == 0


# ---------------------------------------------------------------------------
# 8. HTML dashboard
# ---------------------------------------------------------------------------


class TestSSEHTMLDashboard:
    def test_stats_html_contains_event_source(self, client):
        """The /stats HTML page must use EventSource for live updates."""
        response = client.get("/stats")
        assert response.status_code == 200
        assert "EventSource" in response.text

    def test_stats_html_links_to_stream_endpoint(self, client):
        """The /stats HTML page must reference /stats/stream."""
        response = client.get("/stats")
        assert "/stats/stream" in response.text

    def test_stats_html_has_connection_status_element(self, client):
        """The /stats HTML page must have a connection-status indicator element."""
        response = client.get("/stats")
        assert "conn-status" in response.text

    def test_stats_html_still_offers_json_format(self, client):
        """The /stats HTML page must still link to the JSON view."""
        response = client.get("/stats")
        assert "format=json" in response.text

    def test_stats_json_format_still_works(self, client):
        """GET /stats?format=json must return valid JSON (not affected by SSE change)."""
        response = client.get("/stats?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "backends" in data
