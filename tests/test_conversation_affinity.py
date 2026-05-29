"""Tests for conversation affinity tracking (KV cache reuse)."""

from __future__ import annotations

import time

import pytest

from llamacpp_cli.conversation_affinity import ConversationAffinity


class TestConversationIDExtraction:
    """Test conversation ID extraction from different request formats."""

    def test_explicit_conversation_id(self):
        """Should use explicit conversation_id field if present."""
        affinity = ConversationAffinity()
        body = {
            "conversation_id": "my-custom-id",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id == "my-custom-id"

    def test_empty_conversation_id_ignored(self):
        """Should ignore empty string conversation_id."""
        affinity = ConversationAffinity()
        body = {
            "conversation_id": "",
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"},
            ],
        }
        conv_id = affinity._extract_conversation_id(body)
        # Should fall back to message history hash
        assert conv_id is not None
        assert conv_id != ""

    def test_single_message_no_id(self):
        """Single message should not generate conversation ID."""
        affinity = ConversationAffinity()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is None

    def test_multi_turn_generates_id(self):
        """Multi-turn conversation should generate ID from history."""
        affinity = ConversationAffinity()
        body = {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Tell me more"},
            ]
        }
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is not None
        assert len(conv_id) == 16  # SHA256 truncated to 16 chars

    def test_same_history_same_id(self):
        """Same conversation history should produce same ID."""
        affinity = ConversationAffinity()
        body1 = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        body2 = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        conv_id1 = affinity._extract_conversation_id(body1)
        conv_id2 = affinity._extract_conversation_id(body2)
        assert conv_id1 == conv_id2

    def test_different_history_different_id(self):
        """Different conversation history should produce different IDs."""
        affinity = ConversationAffinity()
        body1 = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        body2 = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        conv_id1 = affinity._extract_conversation_id(body1)
        conv_id2 = affinity._extract_conversation_id(body2)
        assert conv_id1 != conv_id2

    def test_no_messages_field(self):
        """Request without messages field should return None."""
        affinity = ConversationAffinity()
        body = {"model": "llama-3.3-70b-instruct"}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is None

    def test_empty_messages_list(self):
        """Empty messages list should return None."""
        affinity = ConversationAffinity()
        body = {"messages": []}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is None

    def test_messages_not_list(self):
        """Non-list messages field should return None."""
        affinity = ConversationAffinity()
        body = {"messages": "not a list"}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is None

    def test_non_string_conversation_id(self):
        """Non-string conversation_id should be ignored."""
        affinity = ConversationAffinity()
        body = {
            "conversation_id": 123,
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ],
        }
        conv_id = affinity._extract_conversation_id(body)
        # Should fall back to message history hash
        assert conv_id is not None
        assert isinstance(conv_id, str)


class TestAffinityTracking:
    """Test backend affinity tracking and retrieval."""

    def test_get_preferred_backend_no_affinity(self):
        """Should return None for request with no conversation ID."""
        affinity = ConversationAffinity()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None
        assert affinity.total_requests == 1
        assert affinity.affinity_misses == 1
        assert affinity.affinity_hits == 0

    def test_get_preferred_backend_not_recorded(self):
        """Should return None for conversation not yet recorded."""
        affinity = ConversationAffinity()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None
        assert affinity.affinity_misses == 1

    def test_record_and_retrieve_backend(self):
        """Should record and retrieve backend for conversation."""
        affinity = ConversationAffinity()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
        }

        # Record backend
        affinity.record_backend(body, "http://backend1:8000")

        # Retrieve should return same backend
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url == "http://backend1:8000"
        assert affinity.affinity_hits == 1

    def test_record_no_conversation_id(self):
        """Recording single-turn request should be no-op."""
        affinity = ConversationAffinity()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        affinity.record_backend(body, "http://backend1:8000")

        # Should not be able to retrieve
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None

    def test_affinity_persists_across_calls(self):
        """Affinity should persist for multiple requests in same conversation.

        Uses explicit conversation_id to ensure consistency.
        """
        affinity = ConversationAffinity()

        # Use explicit conversation_id to ensure consistency
        body1 = {
            "conversation_id": "my-conversation",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        affinity.record_backend(body1, "http://backend1:8000")

        # Second turn (same conversation_id)
        body2 = {
            "conversation_id": "my-conversation",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm good!"},
                {"role": "user", "content": "What's your name?"},
            ],
        }

        # Should get same backend (same conversation_id)
        backend_url = affinity.get_preferred_backend(body2)
        assert backend_url == "http://backend1:8000"

    def test_different_conversations_different_backends(self):
        """Different conversations should track different backends."""
        affinity = ConversationAffinity()

        body1 = {
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body2 = {
            "conversation_id": "conv-2",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        affinity.record_backend(body1, "http://backend1:8000")
        affinity.record_backend(body2, "http://backend2:8000")

        assert affinity.get_preferred_backend(body1) == "http://backend1:8000"
        assert affinity.get_preferred_backend(body2) == "http://backend2:8000"


class TestTTLExpiration:
    """Test TTL-based affinity expiration."""

    def test_affinity_expires_after_ttl(self):
        """Affinity should expire after TTL seconds."""
        affinity = ConversationAffinity(ttl=0.1)  # 100ms TTL
        body = {
            "conversation_id": "test-conv",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Record backend
        affinity.record_backend(body, "http://backend1:8000")

        # Should be available immediately
        assert affinity.get_preferred_backend(body) == "http://backend1:8000"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None
        assert affinity.affinity_misses == 1

    def test_affinity_not_expired_within_ttl(self):
        """Affinity should persist within TTL."""
        affinity = ConversationAffinity(ttl=10.0)  # 10s TTL
        body = {
            "conversation_id": "test-conv",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        affinity.record_backend(body, "http://backend1:8000")
        time.sleep(0.05)  # Wait 50ms (well within TTL)

        backend_url = affinity.get_preferred_backend(body)
        assert backend_url == "http://backend1:8000"
        assert affinity.affinity_hits == 1

    def test_cleanup_expired_removes_old_entries(self):
        """cleanup_expired should remove expired affinity entries."""
        affinity = ConversationAffinity(ttl=0.1)

        body1 = {"conversation_id": "conv-1", "messages": []}
        body2 = {"conversation_id": "conv-2", "messages": []}

        affinity.record_backend(body1, "http://backend1:8000")
        time.sleep(0.06)  # Wait 60ms
        affinity.record_backend(body2, "http://backend2:8000")

        # Wait for first to expire but not second (60ms + 60ms = 120ms > 100ms TTL)
        time.sleep(0.06)

        removed = affinity.cleanup_expired()
        assert removed == 1  # Only first should be removed
        assert len(affinity._affinity) == 1


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_empty(self):
        """Stats should be zero initially."""
        affinity = ConversationAffinity()
        stats = affinity.get_stats()

        assert stats["total_requests"] == 0
        assert stats["affinity_hits"] == 0
        assert stats["affinity_misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["active_conversations"] == 0

    def test_stats_tracks_hits_and_misses(self):
        """Stats should track hits and misses correctly."""
        affinity = ConversationAffinity()

        body1 = {
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body2 = {"messages": [{"role": "user", "content": "Single message"}]}

        # Miss (not recorded)
        affinity.get_preferred_backend(body1)

        # Record
        affinity.record_backend(body1, "http://backend1:8000")

        # Hit
        affinity.get_preferred_backend(body1)

        # Miss (single message)
        affinity.get_preferred_backend(body2)

        stats = affinity.get_stats()
        assert stats["total_requests"] == 3
        assert stats["affinity_hits"] == 1
        assert stats["affinity_misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)
        assert stats["active_conversations"] == 1

    def test_stats_hit_rate_calculation(self):
        """Hit rate should be calculated correctly."""
        affinity = ConversationAffinity()

        body = {
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        affinity.record_backend(body, "http://backend1:8000")

        # 5 hits
        for _ in range(5):
            affinity.get_preferred_backend(body)

        stats = affinity.get_stats()
        assert stats["hit_rate"] == 1.0  # 5 hits / 5 requests

    def test_stats_active_conversations(self):
        """Should track number of active conversations."""
        affinity = ConversationAffinity()

        for i in range(5):
            body = {
                "conversation_id": f"conv-{i}",
                "messages": [{"role": "user", "content": f"Message {i}"}],
            }
            affinity.record_backend(body, f"http://backend{i}:8000")

        stats = affinity.get_stats()
        assert stats["active_conversations"] == 5

    def test_reset_stats(self):
        """reset_stats should clear counters but not affinity data."""
        affinity = ConversationAffinity()

        body = {
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        affinity.record_backend(body, "http://backend1:8000")
        affinity.get_preferred_backend(body)

        affinity.reset_stats()

        stats = affinity.get_stats()
        assert stats["total_requests"] == 0
        assert stats["affinity_hits"] == 0
        assert stats["affinity_misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Affinity data should still be there
        assert stats["active_conversations"] == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_request_body(self):
        """Empty request body should not crash."""
        affinity = ConversationAffinity()
        body = {}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is None

    def test_malformed_messages(self):
        """Malformed messages should not crash."""
        affinity = ConversationAffinity()
        body = {"messages": [None, {"role": "user"}]}
        conv_id = affinity._extract_conversation_id(body)
        # Should return a hash since it's technically multi-turn
        assert conv_id is not None

    def test_very_long_conversation_id(self):
        """Very long explicit conversation ID should be accepted."""
        affinity = ConversationAffinity()
        long_id = "x" * 1000
        body = {"conversation_id": long_id, "messages": []}
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id == long_id

    def test_unicode_in_messages(self):
        """Unicode characters in messages should be handled."""
        affinity = ConversationAffinity()
        body = {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "こんにちは"},
                {"role": "user", "content": "مرحبا"},
            ]
        }
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id is not None

    def test_special_characters_in_conversation_id(self):
        """Special characters in conversation ID should be handled."""
        affinity = ConversationAffinity()
        body = {
            "conversation_id": "conv-123_456.789@test",
            "messages": [],
        }
        conv_id = affinity._extract_conversation_id(body)
        assert conv_id == "conv-123_456.789@test"

    def test_zero_ttl(self):
        """Zero TTL should expire immediately."""
        affinity = ConversationAffinity(ttl=0.0)
        body = {"conversation_id": "conv-1", "messages": []}

        affinity.record_backend(body, "http://backend1:8000")

        # Should expire immediately
        time.sleep(0.01)
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None

    def test_negative_ttl_treated_as_expired(self):
        """Negative TTL should always be expired."""
        affinity = ConversationAffinity(ttl=-1.0)
        body = {"conversation_id": "conv-1", "messages": []}

        affinity.record_backend(body, "http://backend1:8000")
        backend_url = affinity.get_preferred_backend(body)
        assert backend_url is None
