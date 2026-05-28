"""Tests for lb-proxy OpenAI-compatible endpoints."""

import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import Backend, ProxyState, create_lb_app


@pytest.fixture
def mock_state():
    """Create a mock proxy state with test backends."""
    state = ProxyState()
    # Add mock backends
    backend1 = Backend(host="10.0.0.1", port=8000, models=["model-1", "model-2"], healthy=True)
    backend2 = Backend(host="10.0.0.2", port=8000, models=["model-3"], healthy=True)
    state.backends = [backend1, backend2]
    return state


@pytest.fixture
def client(mock_state):
    """Create test client with mock state."""
    app = create_lb_app(mock_state)
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns HTML landing page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LlamaCPP Load Balancer" in response.text
    assert "/v1/chat/completions" in response.text


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["backends"]["healthy"] == 2
    assert data["backends"]["total"] == 2


def test_backends_endpoint_requires_auth_when_configured(mock_state):
    """Test backends endpoint respects API key when configured."""
    mock_state.api_key = "test-key-123"
    app = create_lb_app(mock_state)
    client = TestClient(app)

    # Without auth
    response = client.get("/backends")
    assert response.status_code == 401

    # With auth
    response = client.get("/backends", headers={"Authorization": "Bearer test-key-123"})
    assert response.status_code == 200


def test_list_models(client):
    """Test /v1/models aggregates models from all backends."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3  # model-1, model-2, model-3
    model_ids = {m["id"] for m in data["data"]}
    assert model_ids == {"model-1", "model-2", "model-3"}


def test_legacy_engines_endpoint(client):
    """Test legacy /v1/engines endpoint (alias for models)."""
    response = client.get("/v1/engines")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3


def test_get_specific_engine(client):
    """Test /v1/engines/{engine_id} endpoint."""
    response = client.get("/v1/engines/model-1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "model-1"
    assert data["object"] == "engine"
    assert data["ready"] is True

    # Non-existent engine
    response = client.get("/v1/engines/non-existent")
    assert response.status_code == 404


def test_stats_endpoint_html(client):
    """Test stats endpoint returns HTML by default."""
    response = client.get("/stats")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Load Balancer Statistics" in response.text


def test_stats_endpoint_json(client):
    """Test stats endpoint returns JSON when format=json."""
    response = client.get("/stats?format=json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "total" in data
    assert "backends" in data


def test_slots_endpoint_aggregates_from_backends(client, mock_state, monkeypatch):
    """Test /slots aggregates slot info from all backends."""
    # Mock httpx responses
    import httpx

    async def mock_get(url, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                if "10.0.0.1" in url:
                    return [{"id": 0, "is_processing": False}]
                elif "10.0.0.2" in url:
                    return [{"id": 0, "is_processing": True}]
                return []

        return MockResponse()

    monkeypatch.setattr(mock_state.http_client, "get", mock_get)

    response = client.get("/slots")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["backend"] == "http://10.0.0.1:8000"
    assert data[1]["backend"] == "http://10.0.0.2:8000"


def test_props_endpoint_aggregates_from_backends(client, mock_state, monkeypatch):
    """Test /props aggregates properties from all backends."""
    import httpx

    async def mock_get(url, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                if "10.0.0.1" in url:
                    return {"n_ctx": 4096, "model": "model-1"}
                elif "10.0.0.2" in url:
                    return {"n_ctx": 8192, "model": "model-3"}
                return {}

        return MockResponse()

    monkeypatch.setattr(mock_state.http_client, "get", mock_get)

    response = client.get("/props")
    assert response.status_code == 200
    data = response.json()
    assert "backends" in data
    assert len(data["backends"]) == 2
    assert data["backends"][0]["backend"] == "http://10.0.0.1:8000"
    assert data["backends"][0]["n_ctx"] == 4096


def test_metrics_endpoint_aggregates_prometheus(client, mock_state, monkeypatch):
    """Test /metrics aggregates Prometheus metrics from all backends."""
    import httpx

    async def mock_get(url, **kwargs):
        class MockResponse:
            status_code = 200
            text = f"# Backend: {url}\nllama_requests_total 10"

        return MockResponse()

    monkeypatch.setattr(mock_state.http_client, "get", mock_get)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "llama_requests_total" in response.text
    assert "Backend:" in response.text
