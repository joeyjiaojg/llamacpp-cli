"""Prometheus metrics for lb-proxy.

This module provides comprehensive metrics collection for the load balancer proxy,
including request latency percentiles, token usage, backend health, and queue metrics.

Metrics are exposed via the /metrics endpoint in Prometheus text format.
"""

from __future__ import annotations

import sys
from enum import Enum

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)


class CircuitBreakerState(Enum):
    """Circuit breaker state enum for metrics."""

    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


# Request metrics
request_total = Counter(
    "lb_proxy_requests_total",
    "Total number of requests processed by the load balancer",
    ["method", "endpoint", "status", "backend"],
    registry=REGISTRY,
)

request_duration = Histogram(
    "lb_proxy_request_duration_seconds",
    "Request duration in seconds from client request to response completion",
    ["endpoint", "backend"],
    # Buckets optimized for LLM inference latency (100ms to 2 minutes)
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)

active_requests = Gauge(
    "lb_proxy_active_requests",
    "Number of requests currently being processed",
    ["backend"],
    registry=REGISTRY,
)

# Token metrics
tokens_total = Counter(
    "lb_proxy_tokens_total",
    "Total tokens processed (prompt + completion)",
    ["type", "model", "backend"],  # type: prompt | completion
    registry=REGISTRY,
)

# Backend metrics
backend_healthy = Gauge(
    "lb_proxy_backend_healthy",
    "Backend health status (1=healthy, 0=unhealthy)",
    ["backend"],
    registry=REGISTRY,
)

backend_circuit_state = Gauge(
    "lb_proxy_backend_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["backend"],
    registry=REGISTRY,
)

backend_consecutive_failures = Gauge(
    "lb_proxy_backend_consecutive_failures",
    "Number of consecutive health check failures",
    ["backend"],
    registry=REGISTRY,
)

# Queue metrics
queue_depth = Gauge(
    "lb_proxy_queue_depth",
    "Number of requests currently waiting in queue",
    registry=REGISTRY,
)

queue_wait_time = Histogram(
    "lb_proxy_queue_wait_seconds",
    "Time requests spend waiting in queue before being processed",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# Rate limiter metrics
rate_limit_hits = Counter(
    "lb_proxy_rate_limit_hits_total",
    "Total number of requests rejected due to rate limiting",
    ["key_type"],  # api_key | ip
    registry=REGISTRY,
)

# Cache metrics (for future cache implementation)
cache_requests = Counter(
    "lb_proxy_cache_requests_total",
    "Total cache lookup requests",
    ["result"],  # hit | miss
    registry=REGISTRY,
)

# Info metric
lb_proxy_info = Info(
    "lb_proxy_info",
    "Build information for the load balancer proxy",
    registry=REGISTRY,
)


def setup_metrics(version: str = "0.1.5") -> None:
    """Initialize metrics with build information.

    Args:
        version: Version string for the lb-proxy build
    """
    lb_proxy_info.info(
        {
            "version": version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
    )


def get_metrics_handler():
    """Get FastAPI handler for /metrics endpoint.

    Returns:
        Callable that returns a FastAPI Response with Prometheus metrics.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.get("/metrics")(get_metrics_handler())
    """

    def metrics():
        from fastapi import Response

        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return metrics


def record_request(
    method: str,
    endpoint: str,
    status: int,
    backend: str,
    duration_seconds: float,
) -> None:
    """Record a completed request.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Request endpoint path
        status: HTTP status code
        backend: Backend URL that handled the request
        duration_seconds: Request duration in seconds
    """
    request_total.labels(
        method=method,
        endpoint=endpoint,
        status=str(status),
        backend=backend,
    ).inc()

    request_duration.labels(
        endpoint=endpoint,
        backend=backend,
    ).observe(duration_seconds)


def set_active_requests(backend: str, count: int) -> None:
    """Set the number of active requests for a backend.

    Args:
        backend: Backend URL
        count: Number of active requests
    """
    active_requests.labels(backend=backend).set(count)


def record_tokens(
    token_type: str,
    model: str,
    backend: str,
    count: int,
) -> None:
    """Record token usage.

    Args:
        token_type: Type of tokens ('prompt' or 'completion')
        model: Model name
        backend: Backend URL
        count: Number of tokens
    """
    tokens_total.labels(
        type=token_type,
        model=model,
        backend=backend,
    ).inc(count)


def set_backend_health(backend: str, healthy: bool) -> None:
    """Update backend health status.

    Args:
        backend: Backend URL
        healthy: Whether the backend is healthy
    """
    backend_healthy.labels(backend=backend).set(1 if healthy else 0)


def set_circuit_state(backend: str, state: CircuitBreakerState) -> None:
    """Update circuit breaker state.

    Args:
        backend: Backend URL
        state: Current circuit breaker state
    """
    backend_circuit_state.labels(backend=backend).set(state.value)


def set_consecutive_failures(backend: str, count: int) -> None:
    """Update consecutive failure count.

    Args:
        backend: Backend URL
        count: Number of consecutive failures
    """
    backend_consecutive_failures.labels(backend=backend).set(count)


def set_queue_depth(depth: int) -> None:
    """Update queue depth.

    Args:
        depth: Number of requests in queue
    """
    queue_depth.set(depth)


def record_queue_wait(wait_seconds: float) -> None:
    """Record time a request spent in queue.

    Args:
        wait_seconds: Wait time in seconds
    """
    queue_wait_time.observe(wait_seconds)


def record_rate_limit_hit(key_type: str) -> None:
    """Record a rate limit rejection.

    Args:
        key_type: Type of rate limiting key ('api_key' or 'ip')
    """
    rate_limit_hits.labels(key_type=key_type).inc()


def record_cache_request(result: str) -> None:
    """Record a cache lookup.

    Args:
        result: Cache lookup result ('hit' or 'miss')
    """
    cache_requests.labels(result=result).inc()
