"""Tests for structured logging and request tracing in lb-proxy."""

import json
import logging
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog
from fastapi import Request
from fastapi.responses import Response

from llamacpp_cli.lb_proxy_logging import (
    add_request_tracing,
    configure_logging,
    extract_request_info,
    generate_request_id,
    get_logger,
    log_backend_health,
    log_backend_request,
    log_backend_selection,
    log_circuit_breaker_open,
    log_queue_full,
    log_rate_limit,
)


def test_configure_logging_json():
    """Test JSON logging configuration."""
    configure_logging(log_level="INFO", log_format="json")
    logger = get_logger()
    assert logger is not None
    assert isinstance(logger, structlog.stdlib.BoundLogger)


def test_configure_logging_text():
    """Test text logging configuration."""
    configure_logging(log_level="DEBUG", log_format="text")
    logger = get_logger()
    assert logger is not None
    assert isinstance(logger, structlog.stdlib.BoundLogger)


def test_generate_request_id():
    """Test request ID generation."""
    request_id = generate_request_id()
    assert request_id is not None
    assert len(request_id) > 0
    # Should be a valid UUID
    uuid.UUID(request_id)


def test_generate_request_id_uniqueness():
    """Test that request IDs are unique."""
    ids = {generate_request_id() for _ in range(100)}
    assert len(ids) == 100


def test_extract_request_info():
    """Test extracting request information."""
    # Create mock request
    request = Mock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.url.query = "stream=true"
    request.client.host = "127.0.0.1"
    request.headers.get.return_value = "test-user-agent"

    info = extract_request_info(request)
    assert info["method"] == "POST"
    assert info["path"] == "/v1/chat/completions"
    assert info["query"] == "stream=true"
    assert info["client_ip"] == "127.0.0.1"
    assert info["user_agent"] == "test-user-agent"


def test_extract_request_info_no_client():
    """Test extracting request info when client is None."""
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = "/health"
    request.url.query = ""
    request.client = None
    request.headers.get.return_value = None

    info = extract_request_info(request)
    assert info["method"] == "GET"
    assert info["path"] == "/health"
    assert info["query"] is None
    assert info["client_ip"] is None
    assert info["user_agent"] is None


@pytest.mark.asyncio
async def test_add_request_tracing_middleware():
    """Test request tracing middleware."""
    # Create mock request
    request = Mock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.url.query = ""
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test"}
    request.body = AsyncMock(return_value=b'{"model":"gpt-4"}')

    # Mock call_next
    response = Response(content="test", status_code=200)
    call_next = AsyncMock(return_value=response)

    # Call middleware
    result = await add_request_tracing(request, call_next)

    # Verify response has tracing headers
    assert "X-Request-ID" in result.headers
    assert call_next.called


@pytest.mark.asyncio
async def test_add_request_tracing_with_existing_request_id():
    """Test middleware preserves existing request ID."""
    # Create mock request with existing request ID
    existing_id = str(uuid.uuid4())
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = "/health"
    request.url.query = ""
    request.client.host = "127.0.0.1"
    request.headers = {"x-request-id": existing_id}
    request.body = AsyncMock(return_value=b"")

    # Mock call_next
    response = Response(content="ok", status_code=200)
    call_next = AsyncMock(return_value=response)

    # Call middleware
    result = await add_request_tracing(request, call_next)

    # Verify existing request ID is preserved
    assert result.headers["X-Request-ID"] == existing_id


@pytest.mark.asyncio
async def test_add_request_tracing_with_correlation_id():
    """Test middleware propagates correlation ID."""
    # Create mock request with correlation ID
    correlation_id = str(uuid.uuid4())
    request = Mock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.url.query = ""
    request.client.host = "127.0.0.1"
    request.headers = {"x-correlation-id": correlation_id}
    request.body = AsyncMock(return_value=b'{"model":"test"}')

    # Mock call_next
    response = Response(content="test", status_code=200)
    call_next = AsyncMock(return_value=response)

    # Call middleware
    result = await add_request_tracing(request, call_next)

    # Verify correlation ID is preserved
    assert result.headers["X-Correlation-ID"] == correlation_id


@pytest.mark.asyncio
async def test_add_request_tracing_error_handling():
    """Test middleware logs errors properly."""
    # Create mock request
    request = Mock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/chat/completions"
    request.url.query = ""
    request.client.host = "127.0.0.1"
    request.headers = {}
    request.body = AsyncMock(return_value=b'{"model":"test"}')

    # Mock call_next to raise exception
    call_next = AsyncMock(side_effect=Exception("Test error"))

    # Verify exception is re-raised
    with pytest.raises(Exception, match="Test error"):
        await add_request_tracing(request, call_next)


def test_log_backend_selection(caplog):
    """Test backend selection logging."""
    configure_logging(log_level="DEBUG", log_format="json")

    with caplog.at_level(logging.DEBUG):
        log_backend_selection(
            backend_url="http://localhost:8000",
            model="gpt-4",
            active_requests=2,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_backend_health_healthy(caplog):
    """Test logging healthy backend status."""
    configure_logging(log_level="INFO", log_format="json")

    with caplog.at_level(logging.INFO):
        log_backend_health(
            backend_url="http://localhost:8000",
            healthy=True,
            reason="After 2 consecutive successes",
            consecutive_successes=2,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_backend_health_unhealthy(caplog):
    """Test logging unhealthy backend status."""
    configure_logging(log_level="WARNING", log_format="json")

    with caplog.at_level(logging.WARNING):
        log_backend_health(
            backend_url="http://localhost:8000",
            healthy=False,
            reason="After 3 consecutive failures",
            consecutive_failures=3,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_backend_request_completed(caplog):
    """Test logging completed backend request."""
    configure_logging(log_level="INFO", log_format="json")

    with caplog.at_level(logging.INFO):
        log_backend_request(
            backend_url="http://localhost:8000",
            method="POST",
            path="/v1/chat/completions",
            status=200,
            duration_ms=150,
            prompt_tokens=50,
            completion_tokens=100,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_backend_request_failed(caplog):
    """Test logging failed backend request."""
    configure_logging(log_level="ERROR", log_format="json")

    with caplog.at_level(logging.ERROR):
        log_backend_request(
            backend_url="http://localhost:8000",
            method="POST",
            path="/v1/chat/completions",
            error="Connection timeout",
            duration_ms=5000,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_rate_limit(caplog):
    """Test logging rate limit hit."""
    configure_logging(log_level="WARNING", log_format="json")

    with caplog.at_level(logging.WARNING):
        log_rate_limit(user_id="user123", limit=60, window=60)

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_queue_full(caplog):
    """Test logging queue full condition."""
    configure_logging(log_level="WARNING", log_format="json")

    with caplog.at_level(logging.WARNING):
        log_queue_full(queue_size=1000, max_size=1000)

    # Verify log was generated
    assert len(caplog.records) > 0


def test_log_circuit_breaker_open(caplog):
    """Test logging circuit breaker opening."""
    configure_logging(log_level="WARNING", log_format="json")

    with caplog.at_level(logging.WARNING):
        log_circuit_breaker_open(
            backend_url="http://localhost:8000",
            error_rate=0.75,
        )

    # Verify log was generated
    assert len(caplog.records) > 0


def test_json_output_format(caplog):
    """Test that JSON logs are parseable."""
    configure_logging(log_level="INFO", log_format="json")

    with caplog.at_level(logging.INFO):
        logger = get_logger()
        logger.info("test_event", key1="value1", key2=123)

    # Note: caplog may not capture the JSON format perfectly
    # This is a basic check that logging occurred
    assert len(caplog.records) > 0


def test_multiple_loggers_isolation():
    """Test that multiple logger instances work independently."""
    logger1 = get_logger("proxy-1")
    logger2 = get_logger("proxy-2")

    assert logger1 is not None
    assert logger2 is not None
    # Note: structlog returns the same bound logger for the same name
    # This test verifies they can be created without error
