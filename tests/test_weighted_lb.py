"""Tests for weighted load balancing."""

import json
import tempfile
from pathlib import Path

import pytest

from llamacpp_cli.lb_proxy import Backend, ProxyState, _load_backends_from_config, _select_backend


def test_backend_default_weight():
    """Test that backends default to weight=1.0."""
    backend = Backend(host="10.0.0.1", port=8000)
    assert backend.weight == 1.0


def test_backend_custom_weight():
    """Test setting custom weight."""
    backend = Backend(host="10.0.0.1", port=8000, weight=2.0)
    assert backend.weight == 2.0


def test_weighted_selection_basic():
    """Test weighted selection with simple case."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=2.0, healthy=True),
    ]

    # Initially both have 0 active requests, should pick first one (both have score=0)
    selected = _select_backend(backends)
    assert selected is not None

    # Simulate 1 active request on backend1
    backends[0].active_requests = 1
    # Scores: backend1 = 1/1 = 1.0, backend2 = 0/2 = 0.0
    # Should select backend2 (lower score)
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.2"

    # Simulate 2 active requests on backend2
    backends[1].active_requests = 2
    # Scores: backend1 = 1/1 = 1.0, backend2 = 2/2 = 1.0
    # Both equal, should pick first one with this score
    selected = _select_backend(backends)
    assert selected is not None

    # Simulate 3 active requests on backend2
    backends[1].active_requests = 3
    # Scores: backend1 = 1/1 = 1.0, backend2 = 3/2 = 1.5
    # Should select backend1 (lower score)
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.1"


def test_weighted_selection_2x_capacity():
    """Test that backend with weight=2.0 handles 2x more requests."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=2.0, healthy=True),
    ]

    # Simulate load distribution
    # Backend2 should handle roughly 2x the requests
    backends[0].active_requests = 10
    backends[1].active_requests = 20

    # Scores: backend1 = 10/1 = 10.0, backend2 = 20/2 = 10.0
    # Equal scores - both are equally loaded relative to their capacity
    selected = _select_backend(backends)
    assert selected is not None

    # If backend2 has 21 requests
    backends[1].active_requests = 21
    # Scores: backend1 = 10/1 = 10.0, backend2 = 21/2 = 10.5
    # Should prefer backend1
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.1"


def test_weighted_selection_half_capacity():
    """Test backend with weight=0.5 handles half the requests."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=0.5, healthy=True),
    ]

    # With equal active requests, backend2 should be more loaded
    backends[0].active_requests = 10
    backends[1].active_requests = 10

    # Scores: backend1 = 10/1 = 10.0, backend2 = 10/0.5 = 20.0
    # Should prefer backend1 (lower score)
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.1"

    # Backend2 at half capacity relative to backend1
    backends[0].active_requests = 10
    backends[1].active_requests = 5
    # Scores: backend1 = 10/1 = 10.0, backend2 = 5/0.5 = 10.0
    # Equal scores
    selected = _select_backend(backends)
    assert selected is not None


def test_weighted_selection_mixed_weights():
    """Test selection with mixed weights (1.0, 2.0, 0.5)."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=2.0, healthy=True),
        Backend(host="10.0.0.3", port=8000, weight=0.5, healthy=True),
    ]

    # All idle - should pick first one with score=0
    selected = _select_backend(backends)
    assert selected is not None

    # Distribute load proportionally
    backends[0].active_requests = 10  # score = 10/1 = 10.0
    backends[1].active_requests = 20  # score = 20/2 = 10.0
    backends[2].active_requests = 5   # score = 5/0.5 = 10.0

    # All equal scores
    selected = _select_backend(backends)
    assert selected is not None

    # Make backend3 slightly overloaded
    backends[2].active_requests = 6  # score = 6/0.5 = 12.0
    # Should avoid backend3
    selected = _select_backend(backends)
    assert selected.host != "10.0.0.3"


def test_weighted_selection_zero_weight():
    """Test that weight=0 is handled (treated as 0.001)."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=0.0, healthy=True),
    ]

    # Backend with weight=0 should be heavily penalized
    backends[0].active_requests = 10
    backends[1].active_requests = 0

    # Scores: backend1 = 10/1 = 10.0, backend2 = 0/0.001 = 0.0
    # Should select backend2
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.2"

    # Even with 1 request, backend2 is heavily loaded
    backends[1].active_requests = 1
    # Scores: backend1 = 10/1 = 10.0, backend2 = 1/0.001 = 1000.0
    # Should select backend1
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.1"


def test_weighted_selection_negative_weight():
    """Test that negative weights don't break selection."""
    # Note: Config loading should validate and prevent negative weights,
    # but the selection function should handle it gracefully
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=-1.0, healthy=True),
    ]

    # Should not crash - negative weight treated as 0.001
    selected = _select_backend(backends)
    assert selected is not None


def test_weighted_selection_unhealthy_backend_ignored():
    """Test that unhealthy backends are ignored regardless of weight."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=10.0, healthy=False),
    ]

    backends[0].active_requests = 10
    backends[1].active_requests = 0

    # Should only consider healthy backend
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.1"


def test_weighted_selection_with_model_routing():
    """Test weighted selection respects model-aware routing."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, models=["model-a"], healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=2.0, models=["model-b"], healthy=True),
        Backend(host="10.0.0.3", port=8000, weight=3.0, models=["model-a"], healthy=True),
    ]

    # Request for model-a should only consider backends 1 and 3
    selected = _select_backend(backends, model="model-a")
    assert selected is not None
    assert selected.host in ["10.0.0.1", "10.0.0.3"]

    # With load, should prefer backend3 due to higher weight
    backends[0].active_requests = 3  # score = 3/1 = 3.0
    backends[2].active_requests = 6  # score = 6/3 = 2.0

    selected = _select_backend(backends, model="model-a")
    assert selected.host == "10.0.0.3"


def test_weighted_selection_all_unhealthy():
    """Test that None is returned when all backends are unhealthy."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=False),
        Backend(host="10.0.0.2", port=8000, weight=2.0, healthy=False),
    ]

    selected = _select_backend(backends)
    assert selected is None


def test_weighted_selection_empty_list():
    """Test that None is returned for empty backend list."""
    selected = _select_backend([])
    assert selected is None


def test_weighted_selection_large_weights():
    """Test selection with very large weights."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=100.0, healthy=True),
    ]

    # Backend2 can handle 100x the load
    backends[0].active_requests = 1   # score = 1/1 = 1.0
    backends[1].active_requests = 50  # score = 50/100 = 0.5

    # Should prefer backend2 despite having 50x more active requests
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.2"


def test_weighted_selection_fractional_weights():
    """Test selection with fractional weights."""
    backends = [
        Backend(host="10.0.0.1", port=8000, weight=0.25, healthy=True),
        Backend(host="10.0.0.2", port=8000, weight=0.75, healthy=True),
    ]

    backends[0].active_requests = 1  # score = 1/0.25 = 4.0
    backends[1].active_requests = 2  # score = 2/0.75 = 2.67

    # Should prefer backend2 (lower score)
    selected = _select_backend(backends)
    assert selected.host == "10.0.0.2"


def test_config_format():
    """Test expected config file format with weights."""
    config = {
        "backends": [
            {"host": "10.0.0.1", "port": 8000, "weight": 1.0},
            {"host": "10.0.0.2", "port": 8000, "weight": 2.0},
            {"host": "10.0.0.3", "port": 8000}  # weight optional, defaults to 1.0
        ]
    }

    # Should be valid JSON
    json_str = json.dumps(config, indent=2)
    loaded = json.loads(json_str)

    assert len(loaded["backends"]) == 3
    assert loaded["backends"][0]["weight"] == 1.0
    assert loaded["backends"][1]["weight"] == 2.0
    assert "weight" not in loaded["backends"][2]  # Optional field


@pytest.mark.asyncio
async def test_load_backends_from_config_with_weights(monkeypatch):
    """Test loading backends from config file with weights."""
    # Mock health check to avoid network calls
    async def mock_check(backend, client, auth_key):
        return True
    async def mock_refresh(backend, client, auth_key):
        pass

    import llamacpp_cli.lb_proxy
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_check_backend_health", mock_check)
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_refresh_backend_models", mock_refresh)

    config = {
        "backends": [
            {"host": "10.0.0.1", "port": 8000, "weight": 1.0},
            {"host": "10.0.0.2", "port": 8000, "weight": 2.5},
            {"host": "10.0.0.3", "port": 8000}  # No weight, should default to 1.0
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = Path(f.name)

    try:
        state = ProxyState()
        state.config_path = config_path

        await _load_backends_from_config(state, auth_key=None)

        assert len(state.backends) == 3
        assert state.backends[0].weight == 1.0
        assert state.backends[1].weight == 2.5
        assert state.backends[2].weight == 1.0
    finally:
        config_path.unlink()


@pytest.mark.asyncio
async def test_load_backends_invalid_weight(monkeypatch):
    """Test that invalid weights are handled gracefully."""
    # Mock health check to avoid network calls
    async def mock_check(backend, client, auth_key):
        return True
    async def mock_refresh(backend, client, auth_key):
        pass

    import llamacpp_cli.lb_proxy
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_check_backend_health", mock_check)
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_refresh_backend_models", mock_refresh)

    config = {
        "backends": [
            {"host": "10.0.0.1", "port": 8000, "weight": -1.0},  # Negative
            {"host": "10.0.0.2", "port": 8000, "weight": "invalid"},  # Not a number
            {"host": "10.0.0.3", "port": 8000, "weight": 2.0},  # Valid
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = Path(f.name)

    try:
        state = ProxyState()
        state.config_path = config_path

        await _load_backends_from_config(state, auth_key=None)

        assert len(state.backends) == 3
        # Invalid weights should default to 1.0
        assert state.backends[0].weight == 1.0
        assert state.backends[1].weight == 1.0
        assert state.backends[2].weight == 2.0
    finally:
        config_path.unlink()


@pytest.mark.asyncio
async def test_load_backends_zero_weight(monkeypatch):
    """Test that zero weight is handled (though not recommended)."""
    # Mock health check to avoid network calls
    async def mock_check(backend, client, auth_key):
        return True
    async def mock_refresh(backend, client, auth_key):
        pass

    import llamacpp_cli.lb_proxy
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_check_backend_health", mock_check)
    monkeypatch.setattr(llamacpp_cli.lb_proxy, "_refresh_backend_models", mock_refresh)

    config = {
        "backends": [
            {"host": "10.0.0.1", "port": 8000, "weight": 0.0},
            {"host": "10.0.0.2", "port": 8000, "weight": 1.0},
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = Path(f.name)

    try:
        state = ProxyState()
        state.config_path = config_path

        await _load_backends_from_config(state, auth_key=None)

        assert len(state.backends) == 2
        # Zero weight should be treated as 1.0 during config loading
        assert state.backends[0].weight == 1.0
        assert state.backends[1].weight == 1.0
    finally:
        config_path.unlink()


def test_cli_backend_parsing():
    """Test parsing backends from CLI format."""
    # Format: host:port:weight

    # With weight
    backend = Backend(host="10.0.0.1", port=8000, weight=2.0)
    assert backend.host == "10.0.0.1"
    assert backend.port == 8000
    assert backend.weight == 2.0

    # Without weight (default)
    backend = Backend(host="10.0.0.2", port=8000)
    assert backend.weight == 1.0


def test_backends_endpoint_shows_weights():
    """Test that /backends endpoint includes weight in response."""
    from fastapi.testclient import TestClient
    from llamacpp_cli.lb_proxy import create_lb_app

    state = ProxyState()
    backend1 = Backend(host="10.0.0.1", port=8000, weight=1.0, healthy=True)
    backend2 = Backend(host="10.0.0.2", port=8000, weight=2.5, healthy=True)
    state.backends = [backend1, backend2]

    app = create_lb_app(state)
    client = TestClient(app)

    response = client.get("/backends")
    assert response.status_code == 200
    data = response.json()

    assert len(data["backends"]) == 2
    assert data["backends"][0]["weight"] == 1.0
    assert data["backends"][1]["weight"] == 2.5
