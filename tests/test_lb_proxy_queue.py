"""Tests for lb-proxy request queuing."""

import asyncio
import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import Backend, ProxyState, RequestQueue, create_lb_app


@pytest.fixture
def mock_state_with_queue():
    """Create a mock proxy state with queueing enabled."""
    state = ProxyState()
    state.request_queue = RequestQueue(max_size=10, timeout=5.0)
    # Add mock backends (initially all busy)
    backend1 = Backend(host="10.0.0.1", port=8000, models=["model-1"], healthy=True, active_requests=1)
    backend2 = Backend(host="10.0.0.2", port=8000, models=["model-2"], healthy=True, active_requests=1)
    state.backends = [backend1, backend2]
    return state


@pytest.fixture
def client_with_queue(mock_state_with_queue):
    """Create test client with queueing enabled."""
    app = create_lb_app(mock_state_with_queue)
    return TestClient(app)


def test_request_queue_enqueue():
    """Test basic enqueue functionality."""
    queue = RequestQueue(max_size=5, timeout=10.0)
    assert queue.size() == 0
    assert queue.total_queued == 0


def test_request_queue_size_limit():
    """Test that queue rejects requests when full."""
    from fastapi import HTTPException, Request

    queue = RequestQueue(max_size=2, timeout=10.0)

    # Mock request
    class MockRequest:
        pass

    # Queue should reject when size limit is reached
    async def test():
        req1 = await queue.enqueue(MockRequest(), "model-1")
        assert queue.size() == 1
        assert queue.total_queued == 1

        req2 = await queue.enqueue(MockRequest(), "model-2")
        assert queue.size() == 2
        assert queue.total_queued == 2

        # Third request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await queue.enqueue(MockRequest(), "model-3")

        assert exc_info.value.status_code == 503
        assert "Queue full" in exc_info.value.detail
        assert queue.total_rejections == 1

    asyncio.run(test())


def test_request_queue_percentiles():
    """Test wait time percentiles calculation."""
    queue = RequestQueue()

    # No data initially
    percentiles = queue.get_percentiles()
    assert percentiles["p50"] == 0.0
    assert percentiles["p95"] == 0.0
    assert percentiles["p99"] == 0.0

    # Add some wait times
    for wait_time in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        queue.record_wait_time(wait_time)

    percentiles = queue.get_percentiles()
    assert percentiles["p50"] == 5.0  # Median
    assert percentiles["p95"] >= 9.0  # 95th percentile
    assert percentiles["p99"] >= 9.0  # 99th percentile


def test_request_queue_stats():
    """Test queue statistics tracking."""
    queue = RequestQueue(max_size=100, timeout=30.0)

    # Add wait times
    for i in range(50):
        queue.record_wait_time(float(i) / 10)  # 0.0, 0.1, 0.2, ... 4.9

    assert len(queue.wait_times) == 50

    # Test that old wait times are kept (only last 1000)
    for i in range(1000):
        queue.record_wait_time(1.0)

    assert len(queue.wait_times) == 1000

    # Adding more should truncate to last 1000
    queue.record_wait_time(2.0)
    assert len(queue.wait_times) == 1000
    assert queue.wait_times[-1] == 2.0


@pytest.mark.asyncio
async def test_queue_worker_processes_requests(mock_state_with_queue, monkeypatch):
    """Test that queue worker processes requests when backends become available."""
    from unittest.mock import AsyncMock
    from llamacpp_cli.lb_proxy import _queue_worker_loop, _forward_request

    state = mock_state_with_queue

    # Mock forward_request to return immediately
    async def mock_forward(request, backend, state):
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "ok"})

    monkeypatch.setattr("llamacpp_cli.lb_proxy._forward_request", mock_forward)

    # Free up a backend
    state.backends[0].active_requests = 0

    # Start queue worker in background
    worker_task = asyncio.create_task(_queue_worker_loop(state))

    try:
        # Enqueue a mock request
        class MockRequest:
            url = type("URL", (), {"path": "/v1/chat/completions", "query": ""})()

            async def body(self):
                return b'{"model": "model-1", "messages": []}'

            async def json(self):
                return {"model": "model-1", "messages": []}

        queued_req = await state.request_queue.enqueue(MockRequest(), "model-1")

        # Wait for processing (with timeout)
        response = await asyncio.wait_for(queued_req.future, timeout=2.0)

        # Verify response
        assert response is not None
        assert state.request_queue.size() == 0
        assert len(state.request_queue.wait_times) == 1

    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


def test_queue_metrics_in_stats(client_with_queue, mock_state_with_queue):
    """Test that queue metrics appear in /stats endpoint."""
    # Add some queue activity
    mock_state_with_queue.request_queue.total_queued = 100
    mock_state_with_queue.request_queue.total_timeouts = 5
    mock_state_with_queue.request_queue.total_rejections = 10
    mock_state_with_queue.request_queue.record_wait_time(1.5)
    mock_state_with_queue.request_queue.record_wait_time(2.0)
    mock_state_with_queue.request_queue.record_wait_time(3.0)

    response = client_with_queue.get("/stats?format=json")
    assert response.status_code == 200

    data = response.json()
    assert "queue" in data
    assert data["queue"]["total_queued"] == 100
    assert data["queue"]["total_timeouts"] == 5
    assert data["queue"]["total_rejections"] == 10
    assert data["queue"]["current_size"] == 0
    assert "wait_times" in data["queue"]
    assert data["queue"]["wait_times"]["p50"] > 0


def test_queue_full_error_includes_estimate(mock_state_with_queue):
    """Test that queue full error includes estimated wait time."""
    from fastapi import HTTPException

    queue = mock_state_with_queue.request_queue

    # Fill queue
    async def test():
        class MockRequest:
            pass

        # Add some historical wait times
        queue.record_wait_time(2.0)
        queue.record_wait_time(3.0)
        queue.record_wait_time(2.5)

        # Fill the queue
        for i in range(10):
            await queue.enqueue(MockRequest(), f"model-{i}")

        # Next request should be rejected with estimate
        with pytest.raises(HTTPException) as exc_info:
            await queue.enqueue(MockRequest(), "model-overflow")

        assert exc_info.value.status_code == 503
        assert "Estimated wait" in exc_info.value.detail

    asyncio.run(test())


@pytest.mark.asyncio
async def test_queue_timeout_handling(mock_state_with_queue):
    """Test that requests time out after configured timeout."""
    from llamacpp_cli.lb_proxy import _queue_worker_loop

    state = mock_state_with_queue
    state.request_queue.timeout = 0.5  # Very short timeout for testing

    # Make all backends unavailable
    for backend in state.backends:
        backend.healthy = False

    # Start queue worker
    worker_task = asyncio.create_task(_queue_worker_loop(state))

    try:
        # Enqueue a request
        class MockRequest:
            url = type("URL", (), {"path": "/v1/chat/completions", "query": ""})()

            async def body(self):
                return b'{"model": "model-1", "messages": []}'

            async def json(self):
                return {"model": "model-1", "messages": []}

        queued_req = await state.request_queue.enqueue(MockRequest(), "model-1")

        # Wait for timeout
        with pytest.raises(Exception):  # Should timeout
            await asyncio.wait_for(queued_req.future, timeout=2.0)

        # Verify timeout was recorded
        assert state.request_queue.total_timeouts == 1

    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
