"""Integration tests for response caching in lb-proxy."""

import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import Backend, ProxyState, create_lb_app
from llamacpp_cli.response_cache import ResponseCache


@pytest.fixture
def mock_state_with_cache():
    """Create a mock proxy state with cache enabled."""
    state = ProxyState()
    backend = Backend(host="10.0.0.1", port=8000, models=["test-model"], healthy=True)
    state.backends = [backend]
    state.response_cache = ResponseCache(max_size=100, ttl=3600.0)
    return state


@pytest.fixture
def client_with_cache(mock_state_with_cache):
    """Create test client with cache enabled."""
    app = create_lb_app(mock_state_with_cache)
    return TestClient(app)


def test_cache_stats_endpoint_includes_cache_info(client_with_cache):
    """Test that /stats endpoint includes cache statistics."""
    response = client_with_cache.get("/stats?format=json")
    assert response.status_code == 200

    data = response.json()
    assert "cache" in data
    assert "enabled" in data["cache"]
    assert data["cache"]["enabled"] is True
    assert "total_requests" in data["cache"]
    assert "cache_hits" in data["cache"]
    assert "cache_misses" in data["cache"]
    assert "hit_rate" in data["cache"]


def test_cache_stats_without_cache():
    """Test stats endpoint when cache is disabled."""
    state = ProxyState()
    backend = Backend(host="10.0.0.1", port=8000, models=["test-model"], healthy=True)
    state.backends = [backend]
    # No cache configured

    app = create_lb_app(state)
    client = TestClient(app)

    response = client.get("/stats?format=json")
    assert response.status_code == 200

    data = response.json()
    assert "cache" not in data


def test_cache_not_used_for_streaming_requests(mock_state_with_cache):
    """Test that streaming requests bypass the cache."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "stream": True,
    }

    cache = mock_state_with_cache.response_cache

    # Streaming requests should not be cached
    key = cache._make_key(request_body)
    assert key is None


def test_cache_key_generation_for_various_requests(mock_state_with_cache):
    """Test cache key generation for different request types."""
    cache = mock_state_with_cache.response_cache

    # Deterministic request - should be cacheable
    deterministic = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    assert cache._make_key(deterministic) is not None

    # Non-deterministic request - not cacheable
    non_deterministic = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    }
    assert cache._make_key(non_deterministic) is None

    # Streaming request - not cacheable
    streaming = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "stream": True,
    }
    assert cache._make_key(streaming) is None


@pytest.mark.asyncio
async def test_cache_lifecycle(mock_state_with_cache):
    """Test full cache lifecycle: put, get, expire."""
    cache = mock_state_with_cache.response_cache
    cache.ttl = 1.0  # 1 second TTL

    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    response = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "Hi there!"}}],
    }

    # Put
    await cache.put(request, response)
    assert cache.get_stats()["size"] == 1

    # Get
    cached = await cache.get(request)
    assert cached == response
    assert cache.get_stats()["cache_hits"] == 1

    # Wait for expiration
    import asyncio

    await asyncio.sleep(1.1)

    # Get after expiration
    cached = await cache.get(request)
    assert cached is None
    assert cache.get_stats()["expirations"] == 1


@pytest.mark.asyncio
async def test_cache_size_limit_enforcement(mock_state_with_cache):
    """Test that cache respects max_size limit."""
    cache = ResponseCache(max_size=5, ttl=3600.0)
    mock_state_with_cache.response_cache = cache

    # Add 10 entries (should only keep 5)
    for i in range(10):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    stats = cache.get_stats()
    assert stats["size"] == 5
    assert stats["evictions"] == 5


@pytest.mark.asyncio
async def test_cache_invalidation(mock_state_with_cache):
    """Test cache invalidation."""
    cache = mock_state_with_cache.response_cache

    # Add some entries
    for i in range(3):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    assert cache.get_stats()["size"] == 3

    # Invalidate all
    count = await cache.invalidate()
    assert count == 3
    assert cache.get_stats()["size"] == 0


@pytest.mark.asyncio
async def test_cache_hit_rate_calculation(mock_state_with_cache):
    """Test hit rate calculation."""
    cache = mock_state_with_cache.response_cache

    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    response = {"id": "chatcmpl-123"}

    await cache.put(request, response)

    # 3 hits
    for _ in range(3):
        await cache.get(request)

    # 2 misses
    for i in range(2):
        other_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Goodbye {i}"}],
            "temperature": 0,
        }
        await cache.get(other_request)

    stats = cache.get_stats()
    assert stats["cache_hits"] == 3
    assert stats["cache_misses"] == 2
    assert stats["total_requests"] == 5
    assert stats["hit_rate"] == 0.6  # 3/5


def test_cache_configuration_via_proxy_state():
    """Test that cache can be configured via ProxyState."""
    state = ProxyState()

    # No cache by default
    assert state.response_cache is None

    # Enable cache
    state.response_cache = ResponseCache(max_size=1000, ttl=7200.0)

    assert state.response_cache is not None
    assert state.response_cache.max_size == 1000
    assert state.response_cache.ttl == 7200.0
