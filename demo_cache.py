#!/usr/bin/env python3
"""Demo script showing response cache in action."""

import asyncio

from llamacpp_cli.response_cache import ResponseCache


async def main():
    """Demonstrate response caching functionality."""
    print("=== Response Cache Demo ===\n")

    # Create cache with small TTL for demo
    cache = ResponseCache(max_size=5, ttl=10.0)

    # Example request (deterministic)
    request1 = {
        "model": "llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0,
    }

    # Example response
    response1 = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "The capital of France is Paris."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }

    print("1. Storing response in cache...")
    await cache.put(request1, response1)
    print(f"   Cache size: {cache.get_stats()['size']}\n")

    print("2. Retrieving cached response (cache hit)...")
    cached = await cache.get(request1)
    if cached:
        print(f"   ✓ Cache hit! Response: {cached['choices'][0]['message']['content']}")
        print(f"   Hit rate: {cache.get_stats()['hit_rate_percent']}\n")

    print("3. Trying non-deterministic request (not cacheable)...")
    request2 = {
        "model": "llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": "Tell me a random joke"}],
        "temperature": 0.7,  # Non-deterministic
    }
    cached2 = await cache.get(request2)
    print(f"   ✗ Not cached (temperature > 0)\n")

    print("4. Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n5. Testing LRU eviction...")
    for i in range(6):
        request = {
            "model": "llama-3.3-70b-instruct",
            "messages": [{"role": "user", "content": f"Question {i}"}],
            "temperature": 0,
        }
        response = {"id": f"chatcmpl-{i}", "choices": []}
        await cache.put(request, response)

    print(f"   Added 6 entries to cache with max_size=5")
    print(f"   Final cache size: {cache.get_stats()['size']}")
    print(f"   Evictions: {cache.get_stats()['evictions']}\n")

    print("6. Testing TTL expiration...")
    print(f"   Waiting 11 seconds for TTL to expire...")
    await asyncio.sleep(11)

    # Try to get an old entry
    cached = await cache.get(request1)
    print(f"   ✗ Cache miss (expired)")
    print(f"   Expirations: {cache.get_stats()['expirations']}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
