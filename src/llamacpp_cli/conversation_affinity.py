"""Conversation affinity tracking for KV cache reuse.

This module tracks which backend handles each conversation to enable
sticky sessions - routing follow-up requests to the same backend for
optimal KV cache reuse (2-3x faster for multi-turn conversations).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationAffinity:
    """Tracks which backend handles each conversation for KV cache reuse.

    Strategies for conversation ID extraction:
    1. Explicit conversation_id field in request body
    2. Hash of message history (multi-turn conversations only)
    3. None for single-turn requests (no affinity)

    Thread-safety: Use with asyncio.Lock when accessed from multiple coroutines.
    """

    ttl: float = 3600.0  # Time-to-live in seconds (default: 1 hour)

    # Mapping: conversation_id -> (backend_url, last_access_time)
    _affinity: dict[str, tuple[str, float]] = field(default_factory=dict)

    # Statistics
    total_requests: int = 0
    affinity_hits: int = 0
    affinity_misses: int = 0

    def _extract_conversation_id(self, request_body: dict) -> str | None:
        """Extract conversation ID from request body.

        Returns:
            Conversation ID string, or None if not a multi-turn conversation.

        Strategy:
            1. Check for explicit "conversation_id" field
            2. For chat completions with 2+ messages, hash message history
            3. Return None for single-turn requests
        """
        # Check for explicit conversation ID
        if "conversation_id" in request_body:
            conv_id = request_body["conversation_id"]
            if isinstance(conv_id, str) and conv_id:
                return conv_id

        # Extract messages for chat completions
        messages = request_body.get("messages")
        if not messages or not isinstance(messages, list):
            return None

        # Single message = not a conversation
        if len(messages) < 2:
            return None

        # Multi-turn: hash all messages except the last one (which is new)
        # This ensures follow-up requests with the same history get the same ID
        history = messages[:-1]
        history_str = str(history)
        return hashlib.sha256(history_str.encode()).hexdigest()[:16]

    def get_preferred_backend(self, request_body: dict) -> str | None:
        """Get preferred backend URL for this conversation.

        Args:
            request_body: Request body dictionary

        Returns:
            Backend URL if affinity exists and is still valid, None otherwise.

        Side effects:
            - Increments total_requests
            - Increments affinity_hits or affinity_misses
            - Removes expired affinity entries
        """
        self.total_requests += 1

        conv_id = self._extract_conversation_id(request_body)
        if conv_id is None:
            self.affinity_misses += 1
            return None

        affinity = self._affinity.get(conv_id)
        if affinity is None:
            self.affinity_misses += 1
            return None

        backend_url, last_access = affinity

        # Check TTL
        if time.time() - last_access > self.ttl:
            del self._affinity[conv_id]
            self.affinity_misses += 1
            return None

        self.affinity_hits += 1
        return backend_url

    def record_backend(self, request_body: dict, backend_url: str) -> None:
        """Record which backend handled this conversation.

        Args:
            request_body: Request body dictionary
            backend_url: Backend URL that handled the request

        Side effects:
            - Updates affinity mapping with current timestamp
            - No-op if conversation ID cannot be extracted
        """
        conv_id = self._extract_conversation_id(request_body)
        if conv_id is None:
            return

        self._affinity[conv_id] = (backend_url, time.time())

    def cleanup_expired(self) -> int:
        """Remove expired affinity entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        expired = [
            conv_id
            for conv_id, (_, last_access) in self._affinity.items()
            if now - last_access > self.ttl
        ]

        for conv_id in expired:
            del self._affinity[conv_id]

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get affinity statistics.

        Returns:
            Dictionary with statistics including hit rate and active conversations.
        """
        hit_rate = self.affinity_hits / self.total_requests if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "affinity_hits": self.affinity_hits,
            "affinity_misses": self.affinity_misses,
            "hit_rate": hit_rate,
            "active_conversations": len(self._affinity),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters (useful for testing)."""
        self.total_requests = 0
        self.affinity_hits = 0
        self.affinity_misses = 0
