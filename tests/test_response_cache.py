"""Tests for response caching functionality."""

import asyncio
import time

import pytest

from llamacpp_cli.response_cache import ResponseCache


@pytest.fixture
def cache():
    """Create a ResponseCache instance for testing."""
    return ResponseCache(max_size=10, ttl=3600.0)


@pytest.mark.asyncio
async def test_cache_key_generation_deterministic(cache):
    """Test that deterministic requests (temperature=0) generate cache keys."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "stream": False,
    }

    key = cache._make_key(request_body)
    assert key is not None
    assert isinstance(key, str)
    assert len(key) == 64  # SHA256 hex digest length


@pytest.mark.asyncio
async def test_cache_key_non_deterministic(cache):
    """Test that non-deterministic requests (temperature>0) are not cached."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "stream": False,
    }

    key = cache._make_key(request_body)
    assert key is None


@pytest.mark.asyncio
async def test_cache_key_streaming_not_cached(cache):
    """Test that streaming requests are not cached."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "stream": True,
    }

    key = cache._make_key(request_body)
    assert key is None


@pytest.mark.asyncio
async def test_cache_hit(cache):
    """Test cache hit returns stored response."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    response_data = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "Hi there!"}}],
    }

    # Store response
    await cache.put(request_body, response_data)

    # Retrieve it
    cached = await cache.get(request_body)
    assert cached is not None
    assert cached == response_data
    assert cache.cache_hits == 1
    assert cache.cache_misses == 0


@pytest.mark.asyncio
async def test_cache_miss(cache):
    """Test cache miss returns None."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    cached = await cache.get(request_body)
    assert cached is None
    assert cache.cache_hits == 0
    assert cache.cache_misses == 1


@pytest.mark.asyncio
async def test_cache_hit_multiple_times(cache):
    """Test that multiple hits increment the hit counter."""
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    response_data = {"id": "chatcmpl-123", "choices": []}

    await cache.put(request_body, response_data)

    # Hit it 5 times
    for _ in range(5):
        cached = await cache.get(request_body)
        assert cached is not None

    assert cache.cache_hits == 5
    assert cache.cache_misses == 0


@pytest.mark.asyncio
async def test_cache_key_consistency(cache):
    """Test that same requests generate same cache keys."""
    request1 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_tokens": 100,
    }

    request2 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_tokens": 100,
    }

    key1 = cache._make_key(request1)
    key2 = cache._make_key(request2)

    assert key1 == key2


@pytest.mark.asyncio
async def test_cache_key_different_messages(cache):
    """Test that different messages generate different cache keys."""
    request1 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    request2 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Goodbye"}],
        "temperature": 0,
    }

    key1 = cache._make_key(request1)
    key2 = cache._make_key(request2)

    assert key1 != key2


@pytest.mark.asyncio
async def test_ttl_expiration(cache):
    """Test that expired entries are not returned."""
    cache.ttl = 0.5  # 0.5 second TTL

    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    response_data = {"id": "chatcmpl-123"}

    await cache.put(request_body, response_data)

    # Should hit immediately
    cached = await cache.get(request_body)
    assert cached is not None

    # Wait for expiration
    await asyncio.sleep(0.6)

    # Should miss after expiration
    cached = await cache.get(request_body)
    assert cached is None
    assert cache.expirations == 1


@pytest.mark.asyncio
async def test_lru_eviction(cache):
    """Test that LRU eviction works when max_size is reached."""
    # Cache has max_size=10
    for i in range(12):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    # Should have evicted 2 oldest entries
    assert len(cache._cache) == 10
    assert cache.evictions == 2


@pytest.mark.asyncio
async def test_lru_order_preserved(cache):
    """Test that LRU order is preserved on access."""
    # Add 3 entries
    for i in range(3):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    # Access entry 0 (should move to end of LRU)
    request0 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Message 0"}],
        "temperature": 0,
    }
    await cache.get(request0)

    # LRU order should now be: 1, 2, 0
    # Fill cache to max_size (10)
    for i in range(3, 10):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    # Add one more entry - should evict entry 1 (oldest)
    request10 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Message 10"}],
        "temperature": 0,
    }
    await cache.put(request10, {"id": "chatcmpl-10"})

    # Entry 1 should be evicted
    request1 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Message 1"}],
        "temperature": 0,
    }
    cached = await cache.get(request1)
    assert cached is None

    # Entry 0 should still be there
    cached = await cache.get(request0)
    assert cached is not None


@pytest.mark.asyncio
async def test_invalidate_all(cache):
    """Test invalidating all cache entries."""
    # Add multiple entries
    for i in range(5):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    assert len(cache._cache) == 5

    # Invalidate all
    count = await cache.invalidate()
    assert count == 5
    assert len(cache._cache) == 0


@pytest.mark.asyncio
async def test_cleanup_expired(cache):
    """Test cleanup of expired entries."""
    cache.ttl = 0.5  # 0.5 second TTL

    # Add entries
    for i in range(3):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    assert len(cache._cache) == 3

    # Wait for expiration
    await asyncio.sleep(0.6)

    # Cleanup expired
    count = await cache.cleanup_expired()
    assert count == 3
    assert len(cache._cache) == 0


@pytest.mark.asyncio
async def test_get_stats(cache):
    """Test statistics tracking."""
    # Start with empty stats
    stats = cache.get_stats()
    assert stats["total_requests"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["size"] == 0

    # Add entry and access it
    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    response = {"id": "chatcmpl-123"}
    await cache.put(request, response)

    # Cache hit
    await cache.get(request)

    # Cache miss
    other_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Goodbye"}],
        "temperature": 0,
    }
    await cache.get(other_request)

    stats = cache.get_stats()
    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["size"] == 1


@pytest.mark.asyncio
async def test_cache_thread_safety(cache):
    """Test concurrent access to cache."""

    async def put_task(i):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}"}
        await cache.put(request, response)

    async def get_task(i):
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Message {i}"}],
            "temperature": 0,
        }
        return await cache.get(request)

    # Concurrent puts
    await asyncio.gather(*[put_task(i) for i in range(10)])

    # Concurrent gets
    results = await asyncio.gather(*[get_task(i) for i in range(10)])

    # All should succeed
    assert len(cache._cache) == 10
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_non_cacheable_requests_dont_affect_stats(cache):
    """Test that non-cacheable requests are counted but don't generate hits."""
    # Non-deterministic request
    request1 = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
    }

    await cache.get(request1)

    stats = cache.get_stats()
    # Non-cacheable requests are counted in total_requests
    assert stats["total_requests"] == 1
    # But they don't generate hits or misses since they're not cacheable
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0


@pytest.mark.asyncio
async def test_cache_key_includes_parameters(cache):
    """Test that cache key includes relevant parameters."""
    base_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }

    # Different max_tokens
    request1 = {**base_request, "max_tokens": 100}
    request2 = {**base_request, "max_tokens": 200}

    key1 = cache._make_key(request1)
    key2 = cache._make_key(request2)
    assert key1 != key2

    # Different stop sequences
    request3 = {**base_request, "stop": ["END"]}
    request4 = {**base_request, "stop": ["STOP"]}

    key3 = cache._make_key(request3)
    key4 = cache._make_key(request4)
    assert key3 != key4


@pytest.mark.asyncio
async def test_cache_hit_counter_increments(cache):
    """Test that cache entry hit counter increments correctly."""
    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }
    response = {"id": "chatcmpl-123"}

    await cache.put(request, response)

    key = cache._make_key(request)
    entry = cache._cache[key]

    assert entry.hits == 0

    # Hit it multiple times
    for i in range(5):
        await cache.get(request)
        assert entry.hits == i + 1


@pytest.mark.asyncio
async def test_default_temperature_is_cacheable(cache):
    """Test that requests without temperature (default 0) are cacheable."""
    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        # No temperature specified
    }

    key = cache._make_key(request)
    assert key is not None  # Should be cacheable (default temp=0)
