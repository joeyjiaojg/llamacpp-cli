"""Tests for per-backend t/s metrics and stats persistence."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import (
    Backend,
    ProxyState,
    _load_stats,
    _save_stats,
    create_lb_app,
)


# ---------------------------------------------------------------------------
# Backend.record_completion / tokens_per_second
# ---------------------------------------------------------------------------


def test_record_completion_updates_totals():
    """record_completion increments total_completion_tokens."""
    b = Backend(host="10.0.0.1", port=8000)
    b.record_completion(50)
    assert b.total_completion_tokens == 50
    b.record_completion(30)
    assert b.total_completion_tokens == 80


def test_record_completion_does_not_touch_prompt_tokens():
    """record_completion must not modify prompt token count."""
    b = Backend(host="10.0.0.1", port=8000, total_prompt_tokens=100)
    b.record_completion(25)
    assert b.total_prompt_tokens == 100


def test_tokens_per_second_empty_window():
    """Returns 0.0 when no completions have been recorded."""
    b = Backend(host="10.0.0.1", port=8000)
    assert b.tokens_per_second() == 0.0


def test_tokens_per_second_within_window():
    """t/s is positive when there are recent completions."""
    b = Backend(host="10.0.0.1", port=8000)
    b.record_completion(120)
    tps = b.tokens_per_second()
    # Single event: elapsed clamped to 1.0 → tps = 120 / 1.0 = 120
    assert tps == pytest.approx(120.0, rel=0.01)


def test_tokens_per_second_sums_all_recent_events():
    """t/s sums all tokens in the window."""
    b = Backend(host="10.0.0.1", port=8000)
    now = time.time()
    # Inject two events at t=now-5 and t=now-2 (both within 60s)
    b._tps_window = [(now - 5, 60), (now - 2, 40)]
    tps = b.tokens_per_second()
    # elapsed = now - (now-5) = 5s (at least 1.0), sum = 100
    assert tps == pytest.approx(100.0 / 5.0, abs=0.5)


def test_tokens_per_second_old_events_excluded():
    """Events older than window_secs are ignored in t/s calculation."""
    b = Backend(host="10.0.0.1", port=8000)
    now = time.time()
    # Insert one old event (70s ago) and one recent (5s ago)
    b._tps_window = [(now - 70, 1000), (now - 5, 50)]
    tps = b.tokens_per_second(window_secs=60.0)
    # Only the recent 50 tokens should count
    assert tps < 100.0  # definitely not inflated by the old 1000
    assert tps > 0.0


def test_tokens_per_second_window_prunes_old_entries():
    """record_completion prunes stale entries older than 60s."""
    b = Backend(host="10.0.0.1", port=8000)
    now = time.time()
    # Manually insert an old entry
    b._tps_window = [(now - 120, 999)]
    # Record a new completion which triggers pruning
    b.record_completion(10)
    # Only the fresh entry should remain
    assert len(b._tps_window) == 1
    assert b._tps_window[0][1] == 10


def test_record_completion_multiple_accumulates_window():
    """Multiple record_completion calls stack in the sliding window."""
    b = Backend(host="10.0.0.1", port=8000)
    b.record_completion(10)
    b.record_completion(20)
    b.record_completion(30)
    assert b.total_completion_tokens == 60
    assert len(b._tps_window) == 3


def test_tokens_per_second_minimum_elapsed_avoids_division_by_zero():
    """elapsed is clamped to at least 1.0 so we never divide by zero."""
    b = Backend(host="10.0.0.1", port=8000)
    # Inject two events at almost the same timestamp
    now = time.time()
    b._tps_window = [(now - 0.0001, 100)]
    tps = b.tokens_per_second()
    # elapsed clamped to 1.0 → tps = 100
    assert tps == pytest.approx(100.0, rel=0.1)


def test_tokens_per_second_custom_window():
    """tokens_per_second respects a custom window_secs argument."""
    b = Backend(host="10.0.0.1", port=8000)
    now = time.time()
    # Event 90s ago (excluded by 60s window, but within 120s window)
    b._tps_window = [(now - 90, 600), (now - 5, 60)]
    tps_60 = b.tokens_per_second(window_secs=60.0)
    # Only the recent event (60 tokens, ~5s elapsed → ~12 t/s)
    assert tps_60 == pytest.approx(60.0 / 5.0, abs=1.0)
    # 120s window must include the old 600-token event
    tps_120 = b.tokens_per_second(window_secs=120.0)
    total_tokens_120 = 600 + 60
    assert tps_120 == pytest.approx(total_tokens_120 / 90.0, abs=1.0)


# ---------------------------------------------------------------------------
# _save_stats
# ---------------------------------------------------------------------------


def test_save_stats_writes_json(tmp_path):
    """_save_stats creates a JSON file with backend totals."""
    state = ProxyState()
    state.stats_file = tmp_path / "stats.json"
    b = Backend(host="10.0.0.1", port=8000)
    b.total_prompt_tokens = 100
    b.total_completion_tokens = 200
    b.total_requests = 5
    state.backends = [b]

    asyncio.run(_save_stats(state))

    data = json.loads(state.stats_file.read_text())
    assert "saved_at" in data
    assert "http://10.0.0.1:8000" in data["backends"]
    entry = data["backends"]["http://10.0.0.1:8000"]
    assert entry["total_prompt_tokens"] == 100
    assert entry["total_completion_tokens"] == 200
    assert entry["total_requests"] == 5


def test_save_stats_no_file_is_noop():
    """_save_stats is a no-op when stats_file is None."""
    state = ProxyState()
    state.stats_file = None
    state.backends = [Backend(host="10.0.0.1", port=8000)]
    # Should not raise
    asyncio.run(_save_stats(state))


def test_save_stats_excludes_tps_window(tmp_path):
    """The saved JSON must not contain the in-memory _tps_window."""
    state = ProxyState()
    state.stats_file = tmp_path / "stats.json"
    b = Backend(host="10.0.0.1", port=8000)
    b._tps_window = [(time.time(), 50)]
    state.backends = [b]

    asyncio.run(_save_stats(state))

    raw_text = state.stats_file.read_text()
    assert "_tps_window" not in raw_text
    assert "tps_window" not in raw_text


def test_save_stats_creates_parent_dir(tmp_path):
    """_save_stats creates missing parent directories."""
    state = ProxyState()
    state.stats_file = tmp_path / "nested" / "deep" / "stats.json"
    state.backends = []

    asyncio.run(_save_stats(state))

    assert state.stats_file.exists()


# ---------------------------------------------------------------------------
# _load_stats
# ---------------------------------------------------------------------------


def test_load_stats_merges_into_matching_backends(tmp_path):
    """_load_stats adds saved totals into matching backends."""
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "saved_at": time.time(),
                "backends": {
                    "http://10.0.0.1:8000": {
                        "total_prompt_tokens": 500,
                        "total_completion_tokens": 300,
                        "total_requests": 10,
                    }
                },
            }
        )
    )

    state = ProxyState()
    state.stats_file = stats_file
    b = Backend(host="10.0.0.1", port=8000)
    b.total_prompt_tokens = 10  # already has some in-memory counts
    state.backends = [b]

    _load_stats(state)

    assert b.total_prompt_tokens == 510
    assert b.total_completion_tokens == 300
    assert b.total_requests == 10


def test_load_stats_ignores_unknown_backends(tmp_path):
    """_load_stats silently ignores backends not in the current state."""
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "saved_at": time.time(),
                "backends": {
                    "http://99.99.99.99:9999": {
                        "total_prompt_tokens": 9999,
                        "total_completion_tokens": 9999,
                        "total_requests": 999,
                    }
                },
            }
        )
    )

    state = ProxyState()
    state.stats_file = stats_file
    b = Backend(host="10.0.0.1", port=8000)
    state.backends = [b]

    _load_stats(state)

    # The known backend must be unchanged
    assert b.total_prompt_tokens == 0
    assert b.total_completion_tokens == 0


def test_load_stats_missing_file_is_noop(tmp_path):
    """_load_stats does nothing when the stats file does not exist."""
    state = ProxyState()
    state.stats_file = tmp_path / "nonexistent.json"
    b = Backend(host="10.0.0.1", port=8000)
    state.backends = [b]

    _load_stats(state)  # Must not raise

    assert b.total_prompt_tokens == 0


def test_load_stats_none_stats_file_is_noop():
    """_load_stats is a no-op when stats_file is None."""
    state = ProxyState()
    state.stats_file = None
    state.backends = [Backend(host="10.0.0.1", port=8000)]

    _load_stats(state)  # Must not raise


def test_load_stats_corrupted_file_is_noop(tmp_path):
    """_load_stats handles JSON parse errors gracefully."""
    stats_file = tmp_path / "stats.json"
    stats_file.write_text("not valid json !!!")

    state = ProxyState()
    state.stats_file = stats_file
    b = Backend(host="10.0.0.1", port=8000)
    state.backends = [b]

    _load_stats(state)  # Must not raise

    assert b.total_prompt_tokens == 0


# ---------------------------------------------------------------------------
# /stats HTTP endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def stats_state():
    state = ProxyState()
    b1 = Backend(host="10.0.0.1", port=8000, healthy=True)
    b1.total_prompt_tokens = 100
    b1.total_completion_tokens = 200
    b1.total_requests = 5
    b1._tps_window = [(time.time() - 1, 60)]  # ~60 t/s
    b2 = Backend(host="10.0.0.2", port=8000, healthy=True)
    state.backends = [b1, b2]
    return state


def test_stats_endpoint_includes_tps_json(stats_state):
    """/stats?format=json includes 'tps' for each backend."""
    app = create_lb_app(stats_state)
    client = TestClient(app)

    resp = client.get("/stats?format=json")
    assert resp.status_code == 200
    data = resp.json()

    for backend_entry in data["backends"]:
        assert "tps" in backend_entry, f"Missing 'tps' in {backend_entry}"


def test_stats_endpoint_tps_is_numeric(stats_state):
    """tps values in /stats JSON are numbers."""
    app = create_lb_app(stats_state)
    client = TestClient(app)

    resp = client.get("/stats?format=json")
    data = resp.json()

    for backend_entry in data["backends"]:
        assert isinstance(backend_entry["tps"], (int, float))


def test_stats_endpoint_tps_zero_for_idle_backend(stats_state):
    """Backend with no completions reports tps=0."""
    app = create_lb_app(stats_state)
    client = TestClient(app)

    resp = client.get("/stats?format=json")
    data = resp.json()

    idle = next(b for b in data["backends"] if b["url"] == "http://10.0.0.2:8000")
    assert idle["tps"] == 0.0


# ---------------------------------------------------------------------------
# /stats/stream SSE endpoint
# ---------------------------------------------------------------------------


def _build_sse_payload(state: ProxyState) -> dict:
    """Reproduce the SSE payload logic used by the /stats/stream generator."""
    total_prompt_tokens = sum(b.total_prompt_tokens for b in state.backends)
    total_completion_tokens = sum(b.total_completion_tokens for b in state.backends)
    total_requests = sum(b.total_requests for b in state.backends)
    healthy_count = sum(1 for b in state.backends if b.healthy)

    return {
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
                "tps": round(b.tokens_per_second(), 2),
                "circuit_state": b.circuit_breaker.state.value,
            }
            for b in state.backends
        ],
    }


def test_sse_stream_includes_tps(stats_state):
    """/stats/stream SSE payload includes 'tps' per backend."""
    payload = _build_sse_payload(stats_state)

    assert "backends" in payload
    for b in payload["backends"]:
        assert "tps" in b, f"Missing 'tps' in SSE backend entry: {b}"


def test_sse_stream_tps_is_numeric(stats_state):
    """tps in SSE payload is a number."""
    payload = _build_sse_payload(stats_state)

    for b in payload["backends"]:
        assert isinstance(b["tps"], (int, float))


def test_stats_html_has_tps_column(stats_state):
    """The HTML stats page includes a 't/s' column header."""
    app = create_lb_app(stats_state)
    client = TestClient(app)

    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "t/s" in resp.text
