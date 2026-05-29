"""Response caching for deterministic LLM requests.

Provides LRU cache with TTL-based invalidation for OpenAI-compatible API responses.
Only caches deterministic requests (temperature=0).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    response: dict[str, Any]
    created_at: float
    hits: int = 0


@dataclass
class ResponseCache:
    """Thread-safe LRU cache with TTL for LLM responses.

    Features:
    - Only caches deterministic requests (temperature=0)
    - LRU eviction when max_size reached
    - TTL-based expiration
    - Thread-safe operations with asyncio.Lock
    - Comprehensive statistics tracking
    """

    max_size: int = 10000
    ttl: float = 3600.0  # 1 hour default

    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _access_order: list[str] = field(default_factory=list)
    _lock: asyncio.Lock | None = None

    # Statistics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expirations: int = 0

    def _get_lock(self) -> asyncio.Lock:
        """Get or create lock in current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _make_key(self, request_body: dict[str, Any]) -> str | None:
        """Create cache key from request.

        Returns:
            Cache key string if cacheable, None otherwise.

        Only caches deterministic requests (temperature=0 or not specified).
        Key includes: model, messages, max_tokens, stop sequences.
        """
        # Only cache deterministic requests
        temp = request_body.get("temperature", 0)
        if temp > 0:
            return None

        # Check for streaming - don't cache streaming requests
        if request_body.get("stream", False):
            return None

        # Key components that affect output
        key_data = {
            "model": request_body.get("model"),
            "messages": request_body.get("messages"),
            "max_tokens": request_body.get("max_tokens"),
            "stop": request_body.get("stop"),
            "top_p": request_body.get("top_p"),
            "presence_penalty": request_body.get("presence_penalty"),
            "frequency_penalty": request_body.get("frequency_penalty"),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return sha256(key_str.encode()).hexdigest()

    async def get(self, request_body: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached response if available and not expired.

        Args:
            request_body: The API request body

        Returns:
            Cached response if found and valid, None otherwise
        """
        async with self._get_lock():
            self.total_requests += 1

            key = self._make_key(request_body)
            if key is None:
                return None

            entry = self._cache.get(key)
            if entry is None:
                self.cache_misses += 1
                return None

            # Check TTL
            now = time.time()
            if now - entry.created_at > self.ttl:
                # Expired - remove it
                del self._cache[key]
                self._access_order.remove(key)
                self.cache_misses += 1
                self.expirations += 1
                return None

            # Cache hit
            self.cache_hits += 1
            entry.hits += 1

            # Update LRU order (move to end = most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)

            return entry.response

    async def put(self, request_body: dict[str, Any], response: dict[str, Any]) -> None:
        """Store response in cache.

        Args:
            request_body: The API request body
            response: The API response to cache
        """
        async with self._get_lock():
            key = self._make_key(request_body)
            if key is None:
                return

            # Evict oldest if at capacity and this is a new key
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                self.evictions += 1

            # Store or update entry
            if key in self._cache:
                # Update existing entry (move to end of LRU)
                self._access_order.remove(key)
                self._access_order.append(key)
                self._cache[key].response = response
                self._cache[key].created_at = time.time()
            else:
                # New entry
                self._cache[key] = CacheEntry(response=response, created_at=time.time())
                self._access_order.append(key)

    async def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Optional pattern to match. If None, clears all entries.

        Returns:
            Number of entries invalidated
        """
        async with self._get_lock():
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                return count

            # Pattern-based invalidation (model name prefix matching)
            invalidated = []
            for key in list(self._cache.keys()):
                # To support pattern matching, we'd need to store metadata
                # For now, just support clearing all
                pass

            for key in invalidated:
                del self._cache[key]
                self._access_order.remove(key)

            return len(invalidated)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._get_lock():
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items() if now - entry.created_at > self.ttl
            ]

            for key in expired_keys:
                del self._cache[key]
                self._access_order.remove(key)
                self.expirations += 1

            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

        return {
            "enabled": True,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "hit_rate_percent": f"{hit_rate * 100:.1f}%",
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "evictions": self.evictions,
            "expirations": self.expirations,
        }
