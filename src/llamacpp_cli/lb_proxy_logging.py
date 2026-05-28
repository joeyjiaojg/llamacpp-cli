"""Structured logging and request tracing for lb-proxy."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from fastapi import Request

# Context variable for request ID
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structured logging with structlog.

    Uses stdlib LoggerFactory so logs flow through the stdlib logging system
    (enabling pytest caplog capture and external log handlers like ELK/Loki).

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (json or text)
    """
    level = getattr(logging, log_level.upper())

    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    logging.getLogger().setLevel(level)

    # Shared pre-chain processors (run before the final renderer)
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add format-specific final renderer
    if log_format == "json":
        pre_chain.append(structlog.processors.JSONRenderer())
    else:
        pre_chain.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=pre_chain,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # Allow reconfiguration in tests
    )


def get_logger(name: str = "lb-proxy") -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name).bind()


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def extract_request_info(request: Request) -> dict[str, Any]:
    """Extract relevant information from a request.

    Args:
        request: FastAPI request object

    Returns:
        Dictionary with request information
    """
    return {
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query) if request.url.query else None,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }


async def add_request_tracing(request: Request, call_next):
    """Middleware to add request tracing and logging.

    Generates unique request ID, logs request start/completion,
    and adds tracing headers.

    Args:
        request: FastAPI request object
        call_next: Next middleware in chain

    Returns:
        Response with added tracing headers
    """
    # Generate or extract request ID
    request_id = request.headers.get("x-request-id") or generate_request_id()
    correlation_id = request.headers.get("x-correlation-id")

    # Set in context variable
    request_id_ctx.set(request_id)

    # Get logger and bind request context
    logger = get_logger()
    logger = logger.bind(
        request_id=request_id,
        correlation_id=correlation_id,
    )

    # Extract request info
    request_info = extract_request_info(request)

    # Try to extract model and user from request body
    model = None
    user_id = None
    try:
        if request.method == "POST":
            body = await request.body()
            # Reset body for downstream handlers
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

            # Parse body
            import json
            request_data = json.loads(body)
            model = request_data.get("model")
            user_id = request_data.get("user")
    except Exception:
        pass

    # Log request start
    start_time = time.time()
    logger.info(
        "request_started",
        **request_info,
        model=model,
        user_id=user_id,
    )

    # Process request
    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Log request completion
        logger.info(
            "request_completed",
            **request_info,
            status=response.status_code,
            duration_ms=int(duration * 1000),
            model=model,
            user_id=user_id,
        )

        # Add tracing headers to response
        response.headers["X-Request-ID"] = request_id
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id

        return response

    except Exception as exc:
        duration = time.time() - start_time

        # Log error
        logger.error(
            "request_failed",
            **request_info,
            error=str(exc),
            error_type=type(exc).__name__,
            duration_ms=int(duration * 1000),
            model=model,
            user_id=user_id,
        )
        raise


def log_backend_selection(
    backend_url: str,
    model: str | None,
    active_requests: int,
    request_id: str | None = None,
) -> None:
    """Log backend selection decision.

    Args:
        backend_url: Selected backend URL
        model: Requested model (if any)
        active_requests: Number of active requests on backend
        request_id: Request ID (optional, will use context if not provided)
    """
    logger = get_logger()
    if request_id:
        logger = logger.bind(request_id=request_id)

    logger.debug(
        "backend_selected",
        backend_url=backend_url,
        model=model,
        active_requests=active_requests,
    )


def log_backend_health(
    backend_url: str,
    healthy: bool,
    reason: str | None = None,
    consecutive_failures: int = 0,
    consecutive_successes: int = 0,
) -> None:
    """Log backend health status change.

    Args:
        backend_url: Backend URL
        healthy: Whether backend is healthy
        reason: Reason for status change
        consecutive_failures: Number of consecutive failures
        consecutive_successes: Number of consecutive successes
    """
    logger = get_logger()

    if healthy:
        logger.info(
            "backend_healthy",
            backend_url=backend_url,
            consecutive_successes=consecutive_successes,
            reason=reason,
        )
    else:
        logger.warning(
            "backend_unhealthy",
            backend_url=backend_url,
            consecutive_failures=consecutive_failures,
            reason=reason,
        )


def log_backend_request(
    backend_url: str,
    method: str,
    path: str,
    status: int | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    request_id: str | None = None,
) -> None:
    """Log backend request forwarding and response.

    Args:
        backend_url: Backend URL
        method: HTTP method
        path: Request path
        status: Response status code (if completed)
        duration_ms: Request duration in milliseconds (if completed)
        error: Error message (if failed)
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        request_id: Request ID (optional, will use context if not provided)
    """
    logger = get_logger()
    if request_id:
        logger = logger.bind(request_id=request_id)

    if error:
        logger.error(
            "backend_request_failed",
            backend_url=backend_url,
            method=method,
            path=path,
            error=error,
            duration_ms=duration_ms,
        )
    elif status and duration_ms is not None:
        logger.info(
            "backend_request_completed",
            backend_url=backend_url,
            method=method,
            path=path,
            status=status,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens or completion_tokens else None,
        )
    else:
        logger.debug(
            "backend_request_started",
            backend_url=backend_url,
            method=method,
            path=path,
        )


def log_rate_limit(user_id: str, limit: int, window: int) -> None:
    """Log rate limit hit.

    Args:
        user_id: User ID that hit the limit
        limit: Rate limit threshold
        window: Time window in seconds
    """
    logger = get_logger()
    logger.warning(
        "rate_limit_exceeded",
        user_id=user_id,
        limit=limit,
        window_seconds=window,
    )


def log_queue_full(queue_size: int, max_size: int) -> None:
    """Log queue full condition.

    Args:
        queue_size: Current queue size
        max_size: Maximum queue size
    """
    logger = get_logger()
    logger.warning(
        "queue_full",
        queue_size=queue_size,
        max_size=max_size,
    )


def log_circuit_breaker_open(backend_url: str, error_rate: float) -> None:
    """Log circuit breaker opening.

    Args:
        backend_url: Backend URL
        error_rate: Current error rate
    """
    logger = get_logger()
    logger.warning(
        "circuit_breaker_open",
        backend_url=backend_url,
        error_rate=error_rate,
    )
