"""Multi-backend load balancer proxy for llama.cpp servers.

Routes requests to multiple llama-server instances running on different machines,
with model-aware routing and least-connections load balancing.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_base_dir
from .conversation_affinity import ConversationAffinity
from .ip_filter import create_ip_filter_middleware
from .lb_proxy_logging import add_request_tracing, configure_logging
from .model_warmer import ModelWarmer
from .request_logger import RequestLogger, RequestRecord
from .response_cache import ResponseCache


def _timestamp() -> str:
    """Return current timestamp in format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


# Pydantic models for request/response documentation
class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the capital of France?",
            }
        }


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions endpoint."""

    model: str = Field(..., description="Model to use", example="llama-3.3-70b-instruct")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Max tokens to generate")
    stream: bool = Field(False, description="Enable streaming responses")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int | None = Field(None, gt=0, description="Top-k sampling")
    repeat_penalty: float | None = Field(None, ge=0.0, description="Repetition penalty")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }


class CompletionRequest(BaseModel):
    """Request body for text completions endpoint."""

    model: str = Field(..., description="Model to use", example="llama-3.3-70b-instruct")
    prompt: str = Field(..., description="Text prompt to complete")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Max tokens to generate")
    stream: bool = Field(False, description="Enable streaming responses")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3.3-70b-instruct",
                "prompt": "Once upon a time",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }


class EmbeddingRequest(BaseModel):
    """Request body for embeddings endpoint."""

    model: str = Field(..., description="Model to use", example="llama-3.3-70b-instruct")
    input: str | list[str] = Field(..., description="Text to embed (string or array of strings)")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama-3.3-70b-instruct",
                "input": "The quick brown fox jumps over the lazy dog",
            }
        }


class TokenizeRequest(BaseModel):
    """Request body for tokenization endpoint."""

    content: str = Field(..., description="Text to tokenize")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Hello, world!",
            }
        }


class DetokenizeRequest(BaseModel):
    """Request body for detokenization endpoint."""

    tokens: list[int] = Field(..., description="Token IDs to detokenize")

    class Config:
        json_schema_extra = {
            "example": {
                "tokens": [1, 2, 3, 4, 5],
            }
        }


@dataclass
class RateLimiter:
    """Rate limiter with per-key request and token quotas.

    Uses sliding window algorithm:
    - Tracks requests per minute (RPM)
    - Tracks tokens per hour (TPH)
    - Falls back to IP-based limiting when no API key provided
    """

    rpm_limit: int  # Requests per minute
    tph_limit: int  # Tokens per hour

    # Sliding windows: key -> deque of timestamps
    _request_windows: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _token_windows: dict[str, deque[tuple[float, int]]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    # Statistics
    _rate_limit_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: asyncio.Lock | None = None

    def get_lock(self) -> asyncio.Lock:
        """Get or create the lock in the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_key(self, request: Request) -> str:
        """Extract rate limiting key from request (API key or IP)."""
        # Try to extract API key from Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Fall back to IP address
        if hasattr(request, "client") and request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    def _clean_window(self, window: deque, cutoff_time: float) -> None:
        """Remove entries older than cutoff_time from window."""
        while window and window[0] < cutoff_time:
            window.popleft()

    def _clean_token_window(self, window: deque, cutoff_time: float) -> None:
        """Remove entries older than cutoff_time from token window."""
        while window and window[0][0] < cutoff_time:
            window.popleft()

    async def check_rate_limit(self, request: Request) -> tuple[bool, str | None]:
        """Check if request is within rate limits.

        Returns:
            (allowed, error_message) - error_message is None if allowed
        """
        async with self.get_lock():
            key = self._get_key(request)
            now = time.time()

            # Check RPM (requests per minute)
            rpm_window = self._request_windows[key]
            rpm_cutoff = now - 60.0  # 60 seconds
            self._clean_window(rpm_window, rpm_cutoff)

            if len(rpm_window) >= self.rpm_limit:
                self._rate_limit_hits[key] += 1
                # Calculate retry_after (seconds until oldest request expires)
                retry_after = int(rpm_window[0] + 60.0 - now) + 1
                return (
                    False,
                    f"Rate limit exceeded: {self.rpm_limit} requests per minute. Retry after {retry_after} seconds.",
                )

            # Check TPH (tokens per hour) - will be updated after response
            tph_window = self._token_windows[key]
            tph_cutoff = now - 3600.0  # 3600 seconds = 1 hour
            self._clean_token_window(tph_window, tph_cutoff)

            current_tokens = sum(tokens for _, tokens in tph_window)
            if current_tokens >= self.tph_limit:
                self._rate_limit_hits[key] += 1
                # Calculate retry_after (seconds until oldest token entry expires)
                retry_after = int(tph_window[0][0] + 3600.0 - now) + 1
                return (
                    False,
                    f"Token quota exceeded: {self.tph_limit} tokens per hour. Retry after {retry_after} seconds.",
                )

            # Add to request window (will be committed)
            rpm_window.append(now)

            return True, None

    async def record_tokens(self, request: Request, tokens: int) -> None:
        """Record token usage for a request."""
        async with self.get_lock():
            key = self._get_key(request)
            now = time.time()

            tph_window = self._token_windows[key]
            tph_window.append((now, tokens))

            # Clean old entries
            tph_cutoff = now - 3600.0
            self._clean_token_window(tph_window, tph_cutoff)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "total_rate_limit_hits": sum(self._rate_limit_hits.values()),
            "rate_limit_hits_by_key": dict(self._rate_limit_hits),
            "active_keys": len(self._request_windows),
        }


@dataclass
class QueuedRequest:
    """A request waiting in the queue for an available backend."""

    request: Any  # The FastAPI Request object
    model: str | None
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class RequestQueue:
    """Queue for requests when no backends are available."""

    max_size: int = 100
    timeout: float = 30.0

    _queue: deque = field(default_factory=deque)
    wait_times: list = field(default_factory=list)
    total_queued: int = 0
    total_timeouts: int = 0
    total_rejections: int = 0

    def size(self) -> int:
        return len(self._queue)

    async def enqueue(self, request: Any, model: str | None) -> "QueuedRequest":
        """Enqueue a request. Raises HTTPException(503) if queue is full."""
        if self.size() >= self.max_size:
            if self.wait_times:
                sorted_times = sorted(self.wait_times)
                idx = int(50 * len(sorted_times) / 100)
                est = sorted_times[min(idx, len(sorted_times) - 1)]
            else:
                est = 0.0
            self.total_rejections += 1
            raise HTTPException(
                status_code=503,
                detail=f"Queue full (max {self.max_size}). Estimated wait: {est:.1f}s",
            )
        queued = QueuedRequest(request=request, model=model)
        self._queue.append(queued)
        self.total_queued += 1
        return queued

    def dequeue(self) -> "QueuedRequest | None":
        """Remove and return the next request from the front of the queue."""
        if self._queue:
            return self._queue.popleft()
        return None

    def record_wait_time(self, wait_time: float) -> None:
        """Record a wait time, keeping only the last 1000 entries."""
        self.wait_times.append(wait_time)
        if len(self.wait_times) > 1000:
            self.wait_times = self.wait_times[-1000:]

    def get_percentiles(self) -> dict:
        """Return p50, p95, p99 wait time percentiles."""
        if not self.wait_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_times = sorted(self.wait_times)
        n = len(sorted_times)
        p50 = sorted_times[int(50 * (n - 1) / 100)]
        p95 = sorted_times[int(95 * (n - 1) / 100)]
        p99 = sorted_times[int(99 * (n - 1) / 100)]
        return {"p50": p50, "p95": p95, "p99": p99}

    def get_stats(self) -> dict:
        """Return queue statistics."""
        return {
            "current_size": self.size(),
            "total_queued": self.total_queued,
            "total_timeouts": self.total_timeouts,
            "total_rejections": self.total_rejections,
            "wait_times": self.get_percentiles(),
        }


class CircuitState(Enum):
    """States for the circuit breaker pattern."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker to protect backends from cascading failures.

    State machine:
    - CLOSED: normal operation, requests pass through
    - OPEN: too many failures, requests are rejected
    - HALF_OPEN: testing if backend recovered
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds in OPEN before trying HALF_OPEN
    half_open_timeout: float = 30.0  # seconds in HALF_OPEN before reopening

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change_time: float = field(default_factory=time.time)
    total_opens: int = 0
    total_closes: int = 0

    def can_attempt_request(self) -> bool:
        """Check if a request should be allowed through."""
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if now - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change_time = now
                return True
            return False
        else:  # HALF_OPEN
            if now - self.last_state_change_time >= self.half_open_timeout:
                self.state = CircuitState.OPEN
                self.last_failure_time = now
                return False
            return True

    def record_failure(self) -> None:
        """Record a failed request and update state accordingly."""
        now = time.time()
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = now
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.total_opens += 1
                self.last_state_change_time = now
        elif self.state == CircuitState.OPEN:
            self.last_failure_time = now
        else:  # HALF_OPEN
            self.state = CircuitState.OPEN
            self.last_failure_time = now
            self.failure_count = 1
            self.success_count = 0
            self.last_state_change_time = now

    def record_success(self) -> None:
        """Record a successful request and update state accordingly."""
        if self.state == CircuitState.CLOSED:
            self.success_count += 1
            self.failure_count = 0
        elif self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.total_closes += 1
                self.failure_count = 0
                self.success_count = 0
        # OPEN: no-op

    def get_state_info(self) -> dict:
        """Return comprehensive circuit breaker state info."""
        now = time.time()
        info: dict = {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_opens": self.total_opens,
            "total_closes": self.total_closes,
            "seconds_until_retry": None,
            "seconds_since_last_failure": None,
        }
        if self.last_failure_time > 0:
            info["seconds_since_last_failure"] = now - self.last_failure_time
        if self.state == CircuitState.OPEN:
            remaining = self.timeout - (now - self.last_failure_time)
            info["seconds_until_retry"] = max(0.0, remaining)
        return info


@dataclass
class Backend:
    """A backend llama-server instance."""

    host: str
    port: int
    weight: float = 1.0
    models: list[str] = field(default_factory=list)
    active_requests: int = 0
    healthy: bool = True
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    checking: bool = False
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    # Token statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    # Sliding window for t/s — (unix_timestamp, completion_tokens) tuples
    # In-memory only; not persisted to disk.
    _tps_window: list[tuple[float, int]] = field(default_factory=list)

    def record_completion(self, completion_tokens: int) -> None:
        """Record completion tokens and update totals for t/s calculation."""
        now = time.time()
        self._tps_window.append((now, completion_tokens))
        # Prune entries older than 60 seconds
        cutoff = now - 60.0
        self._tps_window = [(t, n) for t, n in self._tps_window if t >= cutoff]
        self.total_completion_tokens += completion_tokens

    def tokens_per_second(self, window_secs: float = 60.0) -> float:
        """Compute completion t/s over the last window_secs seconds."""
        if not self._tps_window:
            return 0.0
        now = time.time()
        cutoff = now - window_secs
        recent = [(t, n) for t, n in self._tps_window if t >= cutoff]
        if not recent:
            return 0.0
        timestamps = [t for t, _ in recent]
        elapsed = max(now - timestamps[0], 1.0)  # at least 1s to avoid div/0
        return sum(n for _, n in recent) / elapsed

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __hash__(self) -> int:
        return hash((self.host, self.port))


@dataclass
class ProxyState:
    """Shared state for the load balancer proxy."""

    backends: list[Backend] = field(default_factory=list)
    backends_lock: asyncio.Lock | None = None  # Created lazily in the correct event loop
    http_client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(
            timeout=600.0,  # 10 minutes for slow model inference
            limits=httpx.Limits(
                max_connections=200,  # Up from default 100
                max_keepalive_connections=50,  # Up from default 20
                keepalive_expiry=30.0,  # Keep connections alive longer
            ),
            transport=httpx.AsyncHTTPTransport(retries=1),  # Built-in retry
        )
    )
    health_check_interval: float = 20.0  # Increased from 10.0 for stability
    health_check_task: asyncio.Task | None = None
    config_path: Path | None = None
    config_watch_task: asyncio.Task | None = None
    auth_key: str | None = None  # Optional authentication key for backend discovery
    api_key: str | None = None  # Optional API key for client requests
    rate_limiter: RateLimiter | None = None  # Optional rate limiter
    max_request_size: int = 10 * 1024 * 1024  # Maximum request body size (10MB)
    max_response_tokens: int = 32000  # Maximum response tokens
    request_queue: RequestQueue | None = None  # Optional request queue
    conversation_affinity: ConversationAffinity | None = None  # Optional conversation affinity
    response_cache: ResponseCache | None = None  # Optional response caching
    model_warmer: ModelWarmer | None = None  # Optional model warmer for cold-start reduction
    allowed_cidrs: list[str] | None = None  # IP whitelist (None = allow all)
    request_logger: RequestLogger | None = None  # Optional request logger for replay/debugging
    discover_subnets: list[str] = field(default_factory=list)   # Subnets for periodic rediscovery
    discover_ports: list[int] = field(default_factory=lambda: [8000])  # Ports to probe
    rediscover_interval: float = 60.0  # Seconds between rediscovery sweeps
    rediscover_task: asyncio.Task | None = None
    stats_file: Path | None = None  # Optional path for stats persistence
    stats_persist_task: asyncio.Task | None = None

    def get_lock(self) -> asyncio.Lock:
        """Get or create the backends lock in the current event loop."""
        if self.backends_lock is None:
            self.backends_lock = asyncio.Lock()
        return self.backends_lock

    def validate_api_key(self, request: Request) -> bool:
        """Validate API key from request. Returns True if valid or no key required."""
        if not self.api_key:
            return True  # No API key required

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header[7:]  # Remove "Bearer " prefix
        return token == self.api_key


async def _check_backend_health(
    backend: Backend, client: httpx.AsyncClient, auth_key: str | None = None, verbose: bool = False
) -> bool:
    """Check if a backend is healthy and is a valid llama-server instance.

    Validates:
    1. Endpoint responds
    2. Returns OpenAI-compatible /v1/models format (validates it's llama-server)
    3. Optional auth key matches (if provided)
    """
    headers = {}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"

    try:
        # Check /v1/models to validate it's an OpenAI-compatible server
        resp = await client.get(f"{backend.url}/v1/models", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as e:
                if verbose:
                    print(
                        f"{_timestamp()} [lb-proxy] {backend.url} rejected: JSON parse error - {e}",
                        flush=True,
                    )
                return False

            # Validate OpenAI-compatible response format
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                # Check auth key if backend sends it back
                if auth_key:
                    backend_key = resp.headers.get("Authorization")
                    if backend_key and backend_key != f"Bearer {auth_key}":
                        if verbose:
                            print(
                                f"{_timestamp()} [lb-proxy] {backend.url} rejected: auth key mismatch",
                                flush=True,
                            )
                        return False
                return True
            else:
                if verbose:
                    print(
                        f"{_timestamp()} [lb-proxy] {backend.url} rejected: invalid format - data={type(data)}, has_data={'data' in data if isinstance(data, dict) else False}",
                        flush=True,
                    )
        else:
            if verbose:
                print(
                    f"{_timestamp()} [lb-proxy] {backend.url} rejected: status {resp.status_code}",
                    flush=True,
                )
    except Exception as e:
        if verbose:
            print(
                f"{_timestamp()} [lb-proxy] {backend.url} rejected: {type(e).__name__}: {e}",
                flush=True,
            )

    return False


async def _refresh_backend_models(
    backend: Backend, client: httpx.AsyncClient, auth_key: str | None = None
) -> None:
    """Query a backend for available models and update its model list."""
    headers = {}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"

    try:
        resp = await client.get(f"{backend.url}/v1/models", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and isinstance(data["data"], list):
                backend.models = [m.get("id", "") for m in data["data"] if m.get("id")]
    except Exception:
        pass


async def _health_check_loop(state: ProxyState, auth_key: str | None = None) -> None:
    """Periodically check backend health and refresh model lists.

    Uses consecutive failure/success thresholds to avoid flapping:
    - Requires 3 consecutive failures before marking unhealthy
    - Requires 2 consecutive successes before marking healthy
    """
    FAILURE_THRESHOLD = 3
    SUCCESS_THRESHOLD = 2

    while True:
        await asyncio.sleep(state.health_check_interval)
        async with state.get_lock():
            for backend in state.backends:
                # Skip if already checking or too soon
                now = time.time()
                if (
                    backend.checking
                    or now - backend.last_health_check < state.health_check_interval
                ):
                    continue

                # Prevent concurrent checks
                backend.checking = True
                try:
                    healthy = await _check_backend_health(backend, state.http_client, auth_key)
                    backend.last_health_check = now

                    # Update consecutive counters
                    if healthy:
                        backend.consecutive_successes += 1
                        backend.consecutive_failures = 0
                        backend.circuit_breaker.record_success()
                    else:
                        backend.consecutive_failures += 1
                        backend.consecutive_successes = 0
                        backend.circuit_breaker.record_failure()

                    # Only change healthy status after thresholds are met
                    should_mark_healthy = (
                        not backend.healthy and backend.consecutive_successes >= SUCCESS_THRESHOLD
                    )
                    should_mark_unhealthy = (
                        backend.healthy and backend.consecutive_failures >= FAILURE_THRESHOLD
                    )

                    if should_mark_healthy:
                        backend.healthy = True
                        await _refresh_backend_models(backend, state.http_client, auth_key)
                        print(
                            f"{_timestamp()} [lb-proxy] Backend {backend.url} became healthy "
                            f"(after {backend.consecutive_successes} consecutive successes), models: {backend.models}",
                            flush=True,
                        )
                    elif should_mark_unhealthy:
                        backend.healthy = False
                        print(
                            f"{_timestamp()} [lb-proxy] Backend {backend.url} became unhealthy "
                            f"(after {backend.consecutive_failures} consecutive failures)",
                            flush=True,
                        )
                    elif backend.healthy:
                        # Refresh models silently if still healthy
                        await _refresh_backend_models(backend, state.http_client, auth_key)
                finally:
                    backend.checking = False


async def _rediscovery_loop(state: ProxyState) -> None:
    """Periodically rescan subnets for new backends.

    Runs every state.rediscover_interval seconds. Only probes host:port
    pairs not already known. Uses a semaphore so large /24 subnets don't
    exhaust connections.
    """
    import ipaddress

    while True:
        await asyncio.sleep(state.rediscover_interval)

        if not state.discover_subnets:
            continue

        sem = asyncio.Semaphore(50)

        async def _probe(host: str, port: int) -> None:
            async with sem:
                async with state.get_lock():
                    known = any(
                        b.host == host and b.port == port
                        for b in state.backends
                    )
                if known:
                    return

                candidate = Backend(host=host, port=port)
                healthy = await _check_backend_health(
                    candidate, state.http_client, state.auth_key
                )
                if healthy:
                    await _refresh_backend_models(
                        candidate, state.http_client, state.auth_key
                    )
                    candidate.last_health_check = time.time()
                    async with state.get_lock():
                        if not any(
                            b.host == candidate.host and b.port == candidate.port
                            for b in state.backends
                        ):
                            state.backends.append(candidate)
                            print(
                                f"{_timestamp()} [lb-proxy] Rediscovered new backend:"
                                f" {candidate.url}, models: {candidate.models}",
                                flush=True,
                            )

        tasks = []
        for subnet in state.discover_subnets:
            try:
                network = ipaddress.ip_network(subnet, strict=False)
            except ValueError:
                continue
            for host in network.hosts():
                for port in state.discover_ports:
                    tasks.append(_probe(str(host), port))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _config_watch_loop(state: ProxyState, auth_key: str | None = None) -> None:
    """Watch config file for changes and auto-reload backends."""
    if not state.config_path or not state.config_path.exists():
        return

    last_mtime = state.config_path.stat().st_mtime

    while True:
        await asyncio.sleep(5.0)
        try:
            current_mtime = state.config_path.stat().st_mtime
            if current_mtime > last_mtime:
                print(f"{_timestamp()} [lb-proxy] Config file changed, reloading backends…")
                await _load_backends_from_config(state, auth_key)
                last_mtime = current_mtime
        except Exception as exc:
            print(f"{_timestamp()} [lb-proxy] Error watching config: {exc}")


async def _load_backends_from_config(state: ProxyState, auth_key: str | None = None) -> None:
    """Load backends from config file."""
    if not state.config_path or not state.config_path.exists():
        return

    try:
        with state.config_path.open() as f:
            config = json.load(f)

        backends_data = config.get("backends", [])
        new_backends = []

        for b in backends_data:
            if not isinstance(b, dict) or "host" not in b or "port" not in b:
                continue
            weight = b.get("weight", 1.0)
            # Validate weight: must be positive (>0)
            if not isinstance(weight, (int, float)) or weight <= 0:
                print(
                    f"{_timestamp()} [lb-proxy] Warning: invalid weight {weight} for {b['host']}:{b['port']}, using 1.0"
                )
                weight = 1.0
            backend = Backend(host=b["host"], port=b["port"], weight=weight)
            new_backends.append(backend)

        async with state.get_lock():
            existing = {(b.host, b.port) for b in state.backends}
            new = {(b.host, b.port) for b in new_backends}

            # Add new backends
            added = new - existing
            if added:
                for new_backend in new_backends:
                    if (new_backend.host, new_backend.port) in added:
                        state.backends.append(new_backend)
                        print(
                            f"{_timestamp()} [lb-proxy] Added backend: {new_backend.url} (weight={new_backend.weight})"
                        )
                        # Trigger immediate health check
                        healthy = await _check_backend_health(new_backend, state.http_client, auth_key)
                        new_backend.healthy = healthy
                        new_backend.last_health_check = time.time()
                        if healthy:
                            await _refresh_backend_models(new_backend, state.http_client, auth_key)

            # Remove deleted backends
            removed = existing - new
            if removed:
                state.backends = [b for b in state.backends if (b.host, b.port) not in removed]
                for host, port in removed:
                    print(f"{_timestamp()} [lb-proxy] Removed backend: http://{host}:{port}")

    except Exception as exc:
        print(f"{_timestamp()} [lb-proxy] Error loading config: {exc}")


async def _discover_backends_on_subnet(state: ProxyState, subnet: str, port: int) -> None:
    """Scan a subnet for llama-server instances on a given port."""
    import ipaddress

    try:
        network = ipaddress.ip_network(subnet, strict=False)
    except ValueError as exc:
        print(f"{_timestamp()} [lb-proxy] Invalid subnet {subnet}: {exc}")
        return

    print(f"{_timestamp()} [lb-proxy] Scanning {subnet} for backends on port {port}…")
    tasks = []

    async def _try_host(host: str) -> Backend | None:
        backend = Backend(host=host, port=port)
        if await _check_backend_health(backend, state.http_client, state.auth_key):
            await _refresh_backend_models(backend, state.http_client, state.auth_key)
            backend.last_health_check = time.time()
            return backend
        return None

    for ip in network.hosts():
        tasks.append(_try_host(str(ip)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    async with state.backends_lock:
        for result in results:
            if isinstance(result, Backend):
                # Check if already exists
                if any(b.host == result.host and b.port == result.port for b in state.backends):
                    continue
                state.backends.append(result)
                print(
                    f"{_timestamp()} [lb-proxy] Discovered backend: {result.url}, models: {result.models}"
                )


def _select_backend(backends: list[Backend], model: str | None = None) -> Backend | None:
    """Select the best backend using weighted least-connections routing.

    Score = active_requests / weight
    Lower score = more capacity available
    Weight = 2.0 means the backend can handle 2x the load of weight=1.0
    """
    healthy = [b for b in backends if b.healthy]
    if not healthy:
        return None

    # Model-aware routing: filter to backends that have the requested model
    if model:
        candidates = [b for b in healthy if model in b.models]
        if candidates:
            healthy = candidates

    # Weighted least-connections: pick backend with lowest (active_requests / weight)
    # Handle edge case: weight=0 is treated as weight=1.0
    return min(healthy, key=lambda b: b.active_requests / max(b.weight, 0.001))


async def _check_request_size(request: Request, max_size: int) -> None:
    """Check if request size is within limits.

    Raises HTTPException(413) if request is too large.
    Handles both Content-Length header and chunked transfer encoding.
    """
    # Check Content-Length header if present
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large: {size} bytes (max: {max_size} bytes = {max_size / (1024 * 1024):.1f}MB)",
                )
        except ValueError:
            pass  # Invalid Content-Length, will check during body read

    # For chunked transfer encoding or missing Content-Length, we'll check during body read
    # This is handled by FastAPI's request.body() which respects the size limit


async def _enforce_max_tokens(body_bytes: bytes, max_tokens: int) -> bytes:
    """Enforce max_tokens limit in request body.

    If request contains max_tokens > limit, override it.
    Returns modified body bytes.
    """
    try:
        request_data = json.loads(body_bytes)
        if isinstance(request_data, dict):
            # Check if max_tokens exceeds limit
            if "max_tokens" in request_data:
                requested_tokens = request_data["max_tokens"]
                if requested_tokens > max_tokens:
                    request_data["max_tokens"] = max_tokens
                    return json.dumps(request_data).encode()
    except Exception:
        pass  # Not JSON or parsing error, return original body

    return body_bytes


async def _save_stats(state: "ProxyState") -> None:
    """Persist backend token totals to state.stats_file (JSON)."""
    if not state.stats_file:
        return
    try:
        async with state.get_lock():
            backends_data = {
                b.url: {
                    "total_prompt_tokens": b.total_prompt_tokens,
                    "total_completion_tokens": b.total_completion_tokens,
                    "total_requests": b.total_requests,
                }
                for b in state.backends
            }
        payload = {
            "saved_at": time.time(),
            "backends": backends_data,
        }
        state.stats_file.parent.mkdir(parents=True, exist_ok=True)
        state.stats_file.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"{_timestamp()} [lb-proxy] Error saving stats: {exc}", flush=True)


def _load_stats(state: "ProxyState") -> None:
    """Load persisted token totals from state.stats_file and merge into backends."""
    if not state.stats_file or not state.stats_file.exists():
        return
    try:
        data = json.loads(state.stats_file.read_text())
        backends_saved = data.get("backends", {})
        for backend in state.backends:
            saved = backends_saved.get(backend.url)
            if saved:
                backend.total_prompt_tokens += saved.get("total_prompt_tokens", 0)
                backend.total_completion_tokens += saved.get("total_completion_tokens", 0)
                backend.total_requests += saved.get("total_requests", 0)
        print(
            f"{_timestamp()} [lb-proxy] Loaded stats from {state.stats_file}",
            flush=True,
        )
    except Exception as exc:
        print(f"{_timestamp()} [lb-proxy] Error loading stats: {exc}", flush=True)


async def _stats_persist_loop(state: "ProxyState") -> None:
    """Background loop: persist stats every 30 seconds."""
    while True:
        await asyncio.sleep(30.0)
        await _save_stats(state)


async def _forward_request(request: Request, backend: Backend, state: ProxyState) -> Response:
    """Forward request to a backend and stream back the response."""
    url = f"{backend.url}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    # Try to extract prompt tokens from request body
    prompt_tokens = 0
    try:
        request_data = json.loads(body)
        if "messages" in request_data:
            # Rough estimate: count tokens in messages
            for msg in request_data.get("messages", []):
                content = msg.get("content", "")
                # Very rough estimate: ~4 chars per token
                prompt_tokens += len(content) // 4
    except Exception:
        pass

    _HOP_BY_HOP = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}

    if not backend.circuit_breaker.can_attempt_request():
        raise HTTPException(
            status_code=503, detail=f"Circuit breaker open for backend {backend.url}"
        )

    backend.active_requests += 1
    backend.total_requests += 1
    try:
        backend_resp = await state.http_client.send(
            state.http_client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            ),
            stream=True,
        )

        resp_headers = {
            k: v
            for k, v in backend_resp.headers.items()
            if k.lower() not in {"transfer-encoding", "connection"}
        }

        # Collect response body for token counting
        response_chunks = []

        async def _stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in backend_resp.aiter_bytes():
                    response_chunks.append(chunk)
                    yield chunk
            finally:
                await backend_resp.aclose()
                backend.active_requests -= 1

                # Try to parse response and extract token usage
                total_tokens = 0
                try:
                    full_response = b"".join(response_chunks).decode("utf-8")

                    # Check if it's a streaming response (SSE format with "data:" lines)
                    if "data:" in full_response:
                        # Parse SSE format - extract the last valid JSON chunk
                        lines = full_response.strip().split("\n")
                        last_usage = None

                        for line in lines:
                            line = line.strip()
                            if line.startswith("data:"):
                                json_str = line[5:].strip()  # Remove "data:" prefix
                                if json_str and json_str != "[DONE]":
                                    try:
                                        chunk_data = json.loads(json_str)
                                        # Extract usage if present in this chunk
                                        if "usage" in chunk_data:
                                            last_usage = chunk_data["usage"]
                                    except json.JSONDecodeError:
                                        pass

                        if last_usage:
                            actual_prompt_tokens = last_usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = last_usage.get("completion_tokens", 0)

                            backend.total_prompt_tokens += actual_prompt_tokens
                            backend.record_completion(completion_tokens)
                            total_tokens = actual_prompt_tokens + completion_tokens

                            print(
                                f"{_timestamp()} [lb-proxy] {backend.url} - "
                                f"prompt_tokens: {actual_prompt_tokens}, "
                                f"completion_tokens: {completion_tokens}",
                                flush=True,
                            )
                        else:
                            # No usage found in SSE, use estimate
                            if prompt_tokens > 0:
                                backend.total_prompt_tokens += prompt_tokens
                                total_tokens = prompt_tokens
                    else:
                        # Non-streaming JSON response
                        response_data = json.loads(full_response)
                        usage = response_data.get("usage", {})
                        actual_prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", 0)

                        backend.total_prompt_tokens += actual_prompt_tokens
                        backend.record_completion(completion_tokens)
                        total_tokens = actual_prompt_tokens + completion_tokens

                        print(
                            f"{_timestamp()} [lb-proxy] {backend.url} - "
                            f"prompt_tokens: {actual_prompt_tokens}, "
                            f"completion_tokens: {completion_tokens}",
                            flush=True,
                        )
                except Exception as e:
                    # If we can't parse, use the estimate
                    print(f"{_timestamp()} [lb-proxy] Failed to parse token usage: {e}", flush=True)
                    if prompt_tokens > 0:
                        backend.total_prompt_tokens += prompt_tokens
                        total_tokens = prompt_tokens

                # Record tokens in rate limiter
                if state.rate_limiter and total_tokens > 0:
                    await state.rate_limiter.record_tokens(request, total_tokens)

        backend.circuit_breaker.record_success()
        # Log request after response status is known
        if state.request_logger:
            state.request_logger.record(
                RequestRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path=str(request.url.path),
                    headers=dict(request.headers),
                    body=body,
                    response_status=backend_resp.status_code,
                )
            )
        return StreamingResponse(
            _stream(),
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=backend_resp.headers.get("content-type"),
        )
    except Exception as exc:
        backend.active_requests -= 1
        backend.circuit_breaker.record_failure()
        # Log failed request
        if state.request_logger:
            state.request_logger.record(
                RequestRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path=str(request.url.path),
                    headers=dict(request.headers),
                    body=body,
                    response_status=502,
                    error=str(exc),
                )
            )
        raise HTTPException(status_code=502, detail=f"Backend error: {exc}")


async def _queue_worker_loop(state: ProxyState) -> None:
    """Background worker that drains the request queue."""
    while True:
        if state.request_queue and state.request_queue.size() > 0:
            queued = state.request_queue.dequeue()
            if queued:
                # Check if timed out
                wait_so_far = time.time() - queued.enqueued_at
                if wait_so_far > state.request_queue.timeout:
                    state.request_queue.total_timeouts += 1
                    state.request_queue.record_wait_time(wait_so_far)
                    if not queued.future.done():
                        queued.future.set_exception(
                            HTTPException(status_code=504, detail="Request timed out in queue")
                        )
                    continue

                # Find available backend
                async with state.get_lock():
                    backend = _select_backend(state.backends, queued.model)

                if backend:
                    # Record wait time
                    wait_time = time.time() - queued.enqueued_at
                    state.request_queue.record_wait_time(wait_time)

                    # Process request
                    try:
                        response = await _forward_request(queued.request, backend, state)
                        if not queued.future.done():
                            queued.future.set_result(response)
                    except Exception as e:
                        if not queued.future.done():
                            queued.future.set_exception(e)
                else:
                    # Put back at front of queue
                    state.request_queue._queue.appendleft(queued)
                    await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.05)


def create_lb_app(state: ProxyState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        # Start background tasks
        state.health_check_task = asyncio.create_task(_health_check_loop(state, state.auth_key))
        if state.config_path:
            state.config_watch_task = asyncio.create_task(_config_watch_loop(state, state.auth_key))
        queue_worker_task = None
        if state.request_queue:
            queue_worker_task = asyncio.create_task(_queue_worker_loop(state))

        # Start periodic rediscovery (picks up backends that start after initial scan)
        if state.discover_subnets:
            state.rediscover_task = asyncio.create_task(_rediscovery_loop(state))
            print(
                f"{_timestamp()} [lb-proxy] Periodic rediscovery every"
                f" {state.rediscover_interval:.0f}s for {len(state.discover_subnets)} subnet(s)",
                flush=True,
            )

        # Start stats persistence loop
        if state.stats_file:
            state.stats_persist_task = asyncio.create_task(_stats_persist_loop(state))
            print(
                f"{_timestamp()} [lb-proxy] Stats persistence every 30s to {state.stats_file}",
                flush=True,
            )

        # Start model warmer
        if state.model_warmer:
            if state.model_warmer.warm_on_startup:
                # Run an immediate warm pass before entering the periodic loop
                async with state.get_lock():
                    backends_snapshot = list(state.backends)
                await state.model_warmer.warm_all_backends(backends_snapshot, state.http_client)

            async def _get_backends() -> list[Backend]:
                async with state.get_lock():
                    return list(state.backends)

            state.model_warmer.start(_get_backends, state.http_client)

        yield
        # Shutdown
        if state.health_check_task:
            state.health_check_task.cancel()
        if state.config_watch_task:
            state.config_watch_task.cancel()
        if queue_worker_task:
            queue_worker_task.cancel()
        if state.rediscover_task:
            state.rediscover_task.cancel()
        if state.stats_persist_task:
            state.stats_persist_task.cancel()
        # Persist final stats before exit
        await _save_stats(state)
        if state.model_warmer:
            state.model_warmer.stop()

    # OpenAPI tags for organization
    tags_metadata = [
        {
            "name": "OpenAI API",
            "description": "OpenAI-compatible endpoints for chat, completions, embeddings, and tokenization",
        },
        {
            "name": "Management",
            "description": "Backend management, configuration, and monitoring",
        },
        {
            "name": "Health & Stats",
            "description": "Health checks, statistics, and metrics",
        },
        {
            "name": "Legacy",
            "description": "Legacy OpenAI engine endpoints for backwards compatibility",
        },
    ]

    app = FastAPI(
        title="LlamaCPP Load Balancer",
        description="""
OpenAI-compatible load balancer for llama.cpp servers.

## Features

- **Model-aware routing**: Routes requests to backends with the requested model
- **Least-connections load balancing**: Distributes load evenly across backends
- **Health checking**: Automatic backend health monitoring with circuit breakers
- **Rate limiting**: Per-user/IP rate limits (RPM and TPH quotas)
- **Request queuing**: Graceful handling of traffic spikes
- **Circuit breaker**: Fast-fail for unhealthy backends
- **Conversation affinity**: KV cache reuse for multi-turn chats
- **Response caching**: Cache deterministic responses (temperature=0)

## Authentication

Most endpoints require an API key via `Authorization: Bearer YOUR_KEY` header.
Configure with `--api-key` flag when starting the server.

## Rate Limits

Default limits (configurable via CLI):
- 60 requests per minute (RPM)
- 1,000,000 tokens per hour (TPH)

Rate limiting is applied per API key or per IP address.

## Backends

Add backends via:
- Config file: `~/.llamacpp/lb_backends.json`
- CLI: `--backends http://host1:8000 http://host2:8000`
- Auto-discovery: `--discover-subnet 192.168.1.0/24`

## Examples

**Chat completion:**
```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Streaming:**
```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```
        """,
        version="1.0.0",
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc
        openapi_url="/openapi.json",
        openapi_tags=tags_metadata,
        lifespan=lifespan,
    )

    # Structured request tracing middleware
    app.middleware("http")(add_request_tracing)

    # IP whitelist middleware (applied after tracing so blocked requests are still logged)
    if state.allowed_cidrs:
        app.middleware("http")(create_ip_filter_middleware(state.allowed_cidrs))

    @app.post("/v1/chat/completions", tags=["OpenAI API"])
    async def chat_completions(request: Request) -> Response:
        """Chat completions endpoint (OpenAI-compatible).

        Forwards requests to available backends using model-aware routing
        and least-connections load balancing.

        ## Request Body

        ```json
        {
          "model": "llama-3.3-70b-instruct",
          "messages": [
            {"role": "user", "content": "Hello!"}
          ],
          "temperature": 0.7,
          "max_tokens": 100,
          "stream": false
        }
        ```

        ## Response

        OpenAI-compatible chat completion response with streaming support.

        ## Caching

        Responses with `temperature=0` are cached for 1 hour.

        ## Affinity

        Multi-turn conversations are routed to the same backend for KV cache reuse.

        ## Rate Limiting

        Subject to rate limits: 60 RPM and 1M TPH (default).
        """
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        # Check rate limit
        if state.rate_limiter:
            allowed, err = await state.rate_limiter.check_rate_limit(request)
            if not allowed:
                raise HTTPException(status_code=429, detail=err)

        # Check request size
        await _check_request_size(request, state.max_request_size)

        # Extract model from request
        try:
            body: Any = await request.json()
            model = body.get("model") if isinstance(body, dict) else None
        except Exception:
            model = None

        # Select backend
        async with state.get_lock():
            backend = _select_backend(state.backends, model)

        if not backend:
            if state.request_queue:
                queued = await state.request_queue.enqueue(request, model)
                return await asyncio.wait_for(queued.future, timeout=state.request_queue.timeout)
            raise HTTPException(
                status_code=503,
                detail="No healthy backends available"
                if not model
                else f"No backends available for model '{model}'",
            )

        # Log which backend is handling the request
        print(
            f"{_timestamp()} [lb-proxy] Forwarding /v1/chat/completions to {backend.url}",
            flush=True,
        )

        # Forward request
        return await _forward_request(request, backend, state)

    @app.post("/v1/completions", tags=["OpenAI API"])
    @app.post("/v1/embeddings", tags=["OpenAI API"])
    @app.post("/v1/tokenize", tags=["OpenAI API"])
    @app.post("/v1/detokenize", tags=["OpenAI API"])
    async def other_post_endpoints(request: Request) -> Response:
        """Handle other OpenAI POST endpoints (completions, embeddings, tokenization, etc.)."""
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        # Check rate limit
        if state.rate_limiter:
            allowed, err = await state.rate_limiter.check_rate_limit(request)
            if not allowed:
                raise HTTPException(status_code=429, detail=err)

        # Check request size
        await _check_request_size(request, state.max_request_size)

        # Extract model from request
        try:
            body: Any = await request.json()
            model = body.get("model") if isinstance(body, dict) else None
        except Exception:
            model = None

        # Select backend
        async with state.get_lock():
            backend = _select_backend(state.backends, model)

        if not backend:
            if state.request_queue:
                queued = await state.request_queue.enqueue(request, model)
                return await asyncio.wait_for(queued.future, timeout=state.request_queue.timeout)
            raise HTTPException(
                status_code=503,
                detail="No healthy backends available"
                if not model
                else f"No backends available for model '{model}'",
            )

        # Log which backend is handling the request
        print(
            f"{_timestamp()} [lb-proxy] Forwarding {request.url.path} to {backend.url}", flush=True
        )

        # Forward request
        return await _forward_request(request, backend, state)

    @app.get("/slots", tags=["Management"])
    async def aggregate_slots(request: Request) -> JSONResponse:
        """Aggregate slot status from all healthy backends."""
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        all_slots = []
        async with state.get_lock():
            healthy_backends = [b for b in state.backends if b.healthy]

        for backend in healthy_backends:
            try:
                resp = await state.http_client.get(f"{backend.url}/slots", timeout=5.0)
                if resp.status_code == 200:
                    slots = resp.json()
                    # Add backend info to each slot
                    for slot in slots:
                        slot["backend"] = backend.url
                    all_slots.extend(slots)
            except Exception:
                pass

        return JSONResponse(all_slots)

    @app.get("/props", tags=["Management"])
    async def aggregate_props(request: Request) -> JSONResponse:
        """Aggregate server properties from all healthy backends.

        Returns server properties like model context length, available slots,
        and configuration from all backends.

        ## Response Example

        ```json
        {
          "backends": [
            {
              "backend": "http://server1:8000",
              "default_generation_settings": {
                "n_ctx": 4096,
                "model": "llama-3.3-70b-instruct"
              },
              "total_slots": 4
            }
          ]
        }
        ```

        Requires API key authentication.
        """
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        all_props = []
        async with state.get_lock():
            healthy_backends = [b for b in state.backends if b.healthy]

        for backend in healthy_backends:
            try:
                resp = await state.http_client.get(f"{backend.url}/props", timeout=5.0)
                if resp.status_code == 200:
                    props = resp.json()
                    props["backend"] = backend.url
                    all_props.append(props)
            except Exception:
                pass

        return JSONResponse({"backends": all_props})

    @app.get("/metrics", tags=["Health & Stats"])
    async def aggregate_metrics(request: Request) -> Response:
        """Aggregate Prometheus metrics from all healthy backends.

        Returns Prometheus-format metrics aggregated from all backends.
        Metrics are prefixed with backend identifiers for disambiguation.

        ## Response Format

        Plain text in Prometheus exposition format.

        ## Authentication

        This endpoint does not require authentication (for monitoring tools).
        """
        # Note: metrics endpoint typically doesn't require authentication
        all_metrics = []
        async with state.get_lock():
            healthy_backends = [b for b in state.backends if b.healthy]

        for backend in healthy_backends:
            try:
                resp = await state.http_client.get(f"{backend.url}/metrics", timeout=5.0)
                if resp.status_code == 200:
                    metrics = resp.text
                    # Prefix metrics with backend identifier
                    prefixed = "\n".join(
                        f"# Backend: {backend.url}\n{line}" if not line.startswith("#") else line
                        for line in metrics.split("\n")
                    )
                    all_metrics.append(prefixed)
            except Exception:
                pass

        combined = "\n\n".join(all_metrics)
        return Response(content=combined, media_type="text/plain")

    # Legacy OpenAI endpoints (for compatibility)
    @app.get("/v1/engines", tags=["Legacy"])
    async def list_engines(request: Request) -> JSONResponse:
        """Legacy OpenAI engines endpoint (alias for /v1/models).

        Provided for backward compatibility with older OpenAI API clients.

        See `/v1/models` for details.
        """
        return await list_models(request)

    @app.get("/v1/engines/{engine_id}", tags=["Legacy"])
    async def get_engine(engine_id: str, request: Request) -> JSONResponse:
        """Legacy endpoint - get specific engine details."""
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        # Check if the model exists across backends
        models_set = set()
        async with state.get_lock():
            for backend in state.backends:
                if backend.healthy:
                    models_set.update(backend.models)

        if engine_id not in models_set:
            raise HTTPException(status_code=404, detail=f"Engine '{engine_id}' not found")

        return JSONResponse(
            {
                "id": engine_id,
                "object": "engine",
                "owner": "llamacpp",
                "ready": True,
            }
        )

    @app.post("/v1/engines/{engine_id}/completions", tags=["Legacy"])
    async def engine_completions(engine_id: str, request: Request) -> Response:
        """Legacy endpoint - completions with engine ID."""
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        # Select backend that has this model
        async with state.get_lock():
            backend = _select_backend(state.backends, engine_id)

        if not backend:
            raise HTTPException(
                status_code=503,
                detail=f"No backends available for engine '{engine_id}'",
            )

        print(
            f"{_timestamp()} [lb-proxy] Forwarding legacy /v1/engines/{engine_id}/completions to {backend.url}/v1/completions",
            flush=True,
        )

        # Rewrite the request to /v1/completions with model parameter
        body = await request.body()
        try:
            request_data = json.loads(body)
            request_data["model"] = engine_id
            body = json.dumps(request_data).encode()
        except Exception:
            pass

        # Build new request
        url = f"{backend.url}/v1/completions"
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}
        }

        backend.active_requests += 1
        backend.total_requests += 1
        try:
            backend_resp = await state.http_client.send(
                state.http_client.build_request(
                    method="POST",
                    url=url,
                    headers=headers,
                    content=body,
                ),
                stream=True,
            )

            resp_headers = {
                k: v
                for k, v in backend_resp.headers.items()
                if k.lower() not in {"transfer-encoding", "connection"}
            }

            async def _stream() -> AsyncIterator[bytes]:
                try:
                    async for chunk in backend_resp.aiter_bytes():
                        yield chunk
                finally:
                    await backend_resp.aclose()
                    backend.active_requests -= 1

            return StreamingResponse(
                _stream(),
                status_code=backend_resp.status_code,
                headers=resp_headers,
                media_type=backend_resp.headers.get("content-type"),
            )
        except Exception as exc:
            backend.active_requests -= 1
            raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

    @app.get("/v1/models", tags=["OpenAI API"])
    async def list_models(request: Request) -> JSONResponse:
        """List all available models across all backends (OpenAI-compatible).

        Returns a deduplicated list of all models available across healthy backends.

        ## Response Example

        ```json
        {
          "object": "list",
          "data": [
            {
              "id": "llama-3.3-70b-instruct",
              "object": "model"
            }
          ]
        }
        ```

        Requires API key authentication.
        """
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        models_set = set()
        async with state.get_lock():
            for backend in state.backends:
                if backend.healthy:
                    models_set.update(backend.models)

        models_data = [{"id": model, "object": "model"} for model in sorted(models_set)]
        return JSONResponse({"object": "list", "data": models_data})

    @app.get("/", tags=["Health & Stats"])
    async def root() -> HTMLResponse:
        """Simple landing page with links to key endpoints."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LlamaCPP Load Balancer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .card {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #555;
            margin-top: 0;
        }
        .links a {
            display: block;
            padding: 10px 15px;
            margin: 8px 0;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            transition: background 0.3s;
        }
        .links a:hover {
            background: #45a049;
        }
        .note {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 20px 0;
            color: #856404;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>🔄 LlamaCPP Load Balancer Proxy</h1>

    <div class="card">
        <h2>Welcome</h2>
        <p>This is a load balancer proxy for llama.cpp servers. It routes API requests to multiple backend instances.</p>
    </div>

    <div class="note">
        <strong>Note:</strong> This proxy handles API endpoints only. For a chat UI, access one of the backend servers directly.
    </div>

    <div class="card">
        <h2>Available Endpoints</h2>
        <div class="links">
            <a href="/stats">📊 View Statistics</a>
            <a href="/health">🏥 Health Check</a>
            <a href="/backends">🖥️ Backend Status</a>
            <a href="/v1/models">📝 List Models</a>
            <a href="/slots">🎰 Slot Status</a>
            <a href="/props">⚙️ Server Properties</a>
            <a href="/metrics">📈 Metrics (Prometheus)</a>
        </div>
    </div>

    <div class="card">
        <h2>OpenAI-Compatible API Endpoints</h2>
        <p><strong>Core endpoints:</strong></p>
        <ul>
            <li><code>POST /v1/chat/completions</code> - Chat completions (streaming supported)</li>
            <li><code>POST /v1/completions</code> - Text completions (streaming supported)</li>
            <li><code>POST /v1/embeddings</code> - Generate embeddings</li>
            <li><code>GET /v1/models</code> - List all available models</li>
        </ul>
        <p><strong>Tokenization:</strong></p>
        <ul>
            <li><code>POST /v1/tokenize</code> - Tokenize text</li>
            <li><code>POST /v1/detokenize</code> - Detokenize tokens</li>
        </ul>
        <p><strong>Legacy OpenAI endpoints:</strong></p>
        <ul>
            <li><code>GET /v1/engines</code> - List engines (alias for models)</li>
            <li><code>GET /v1/engines/{engine_id}</code> - Get engine details</li>
            <li><code>POST /v1/engines/{engine_id}/completions</code> - Legacy completions</li>
        </ul>
    </div>

    <div class="card">
        <h2>llama.cpp-Specific Endpoints</h2>
        <ul>
            <li><code>GET /slots</code> - View slot status across all backends</li>
            <li><code>GET /props</code> - Server properties from all backends</li>
            <li><code>GET /metrics</code> - Prometheus metrics (aggregated)</li>
        </ul>
    </div>

    <div class="card">
        <h2>Example</h2>
        <pre style="background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto;"><code>curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'</code></pre>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=html)

    @app.get("/health", tags=["Health & Stats"])
    async def health() -> JSONResponse:
        """Load balancer health check endpoint.

        Returns health status and backend availability.

        ## Response Example

        ```json
        {
          "status": "ok",
          "backends": {
            "healthy": 3,
            "total": 4
          }
        }
        ```

        Status is "ok" if at least one backend is healthy, "degraded" otherwise.

        No authentication required.
        """
        async with state.get_lock():
            healthy_count = sum(1 for b in state.backends if b.healthy)
            total_count = len(state.backends)

        return JSONResponse(
            {
                "status": "ok" if healthy_count > 0 else "degraded",
                "backends": {"healthy": healthy_count, "total": total_count},
            }
        )

    @app.get("/backends", tags=["Management"])
    @app.get("/v1/backends", tags=["Management"])
    async def list_backends(request: Request) -> JSONResponse:
        """List all backends and their current status.

        Returns detailed information about all configured backends including
        health status, available models, and current load.

        ## Response Example

        ```json
        {
          "backends": [
            {
              "url": "http://server1:8000",
              "healthy": true,
              "models": ["llama-3.3-70b-instruct"],
              "active_requests": 2,
              "load_status": "busy"
            }
          ]
        }
        ```

        Requires API key authentication.
        """
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

        async with state.get_lock():
            backends_data = [
                {
                    "url": b.url,
                    "healthy": b.healthy,
                    "models": b.models,
                    "active_requests": b.active_requests,
                    "load_status": "busy" if b.active_requests > 0 else "idle",
                    "weight": b.weight,
                }
                for b in state.backends
            ]
        result: dict = {"backends": backends_data}
        if state.model_warmer:
            result["warming"] = state.model_warmer.get_status()
        return JSONResponse(result)

    @app.get("/stats", tags=["Health & Stats"])
    @app.get("/v1/stats", tags=["Health & Stats"])
    async def stats(request: Request, format: str | None = None) -> Response:
        """Get comprehensive token usage and performance statistics.

        Returns aggregate and per-backend statistics including request counts,
        token usage, and queue metrics (if enabled).

        ## Query Parameters

        - `format`: Optional, set to "json" for JSON response (default is HTML)

        ## JSON Response Example

        ```json
        {
          "total": {
            "requests": 1000,
            "prompt_tokens": 50000,
            "completion_tokens": 30000,
            "total_tokens": 80000
          },
          "backends": [
            {
              "url": "http://server1:8000",
              "healthy": true,
              "total_requests": 500,
              "total_prompt_tokens": 25000,
              "total_completion_tokens": 15000,
              "total_tokens": 40000
            }
          ],
          "queue": {
            "current_size": 0,
            "total_queued": 10,
            "total_timeouts": 0,
            "wait_times": {
              "p50": 0.1,
              "p95": 0.5,
              "p99": 1.0
            }
          }
        }
        ```

        No authentication required.
        """
        async with state.get_lock():
            # Calculate totals
            total_prompt_tokens = sum(b.total_prompt_tokens for b in state.backends)
            total_completion_tokens = sum(b.total_completion_tokens for b in state.backends)
            total_requests = sum(b.total_requests for b in state.backends)

            # Per-backend stats
            backend_stats = [
                {
                    "url": b.url,
                    "healthy": b.healthy,
                    "total_requests": b.total_requests,
                    "total_prompt_tokens": b.total_prompt_tokens,
                    "total_completion_tokens": b.total_completion_tokens,
                    "total_tokens": b.total_prompt_tokens + b.total_completion_tokens,
                    "tps": round(b.tokens_per_second(), 2),
                }
                for b in state.backends
            ]

        stats_data = {
            "total": {
                "requests": total_requests,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "backends": backend_stats,
        }

        if state.request_queue:
            stats_data["queue"] = state.request_queue.get_stats()

        if state.response_cache:
            stats_data["cache"] = state.response_cache.get_stats()

        # Return JSON if format=json is specified
        if format == "json":
            return JSONResponse(stats_data)

        # Otherwise return live SSE-powered HTML dashboard
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Load Balancer Stats</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .connection-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            float: right;
            margin-top: 8px;
        }
        .connection-status.connected { background: #e8f5e9; color: #2e7d32; }
        .connection-status.disconnected { background: #ffebee; color: #c62828; }
        .connection-status .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
        .connection-status.connected .dot { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        .total-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card .label { color: #777; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-card .value { color: #333; font-size: 32px; font-weight: bold; margin-top: 5px; }
        table {
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }
        th { background: #4CAF50; color: white; padding: 15px; text-align: left; font-weight: 600; }
        td { padding: 12px 15px; border-bottom: 1px solid #eee; }
        tr:last-child td { border-bottom: none; }
        tr:hover { background: #f9f9f9; }
        .healthy { color: #4CAF50; font-weight: bold; }
        .unhealthy { color: #f44336; font-weight: bold; }
        .circuit-open { color: #f44336; font-size: 12px; }
        .circuit-closed { color: #4CAF50; font-size: 12px; }
        .circuit-half_open { color: #ff9800; font-size: 12px; }
        .tps-high { color: #4CAF50; font-weight: bold; }
        .tps-mid { color: #ff9800; font-weight: bold; }
        .tps-low { color: #f44336; font-weight: bold; }
        .tps-idle { color: #aaa; }
        .extra-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .extra-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .extra-card h3 { margin: 0 0 12px; color: #555; font-size: 15px; }
        .kv-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }
        .kv-row .kv-label { color: #888; }
        .kv-row .kv-val { font-weight: 600; color: #333; }
        .footer {
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            text-align: center;
            color: #777;
            font-size: 14px;
        }
        .footer a { color: #4CAF50; text-decoration: none; }
        .footer a:hover { text-decoration: underline; }
        #last-updated { font-size: 12px; color: #999; margin-left: 10px; }
    </style>
</head>
<body>
    <h1>Load Balancer Statistics
        <span id="conn-status" class="connection-status disconnected">
            <span class="dot"></span>
            <span id="conn-label">Connecting...</span>
        </span>
    </h1>
    <span id="last-updated"></span>

    <h2>Total Statistics</h2>
    <div class="total-stats">
        <div class="stat-card">
            <div class="label">Total Requests</div>
            <div class="value" id="stat-total-requests">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Request Rate</div>
            <div class="value" id="stat-req-rate">--</div>
            <div class="label" style="font-size:11px">req/min (last 60s)</div>
        </div>
        <div class="stat-card">
            <div class="label">Prompt Tokens</div>
            <div class="value" id="stat-prompt-tokens">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Completion Tokens</div>
            <div class="value" id="stat-completion-tokens">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Tokens</div>
            <div class="value" id="stat-total-tokens">--</div>
        </div>
        <div class="stat-card">
            <div class="label">Healthy Backends</div>
            <div class="value" id="stat-healthy-backends">--</div>
        </div>
    </div>

    <h2>Backend Status</h2>
    <table id="backends-table">
        <thead>
            <tr>
                <th>Backend URL</th>
                <th>Status</th>
                <th>Active</th>
                <th>Total Requests</th>
                <th>Total Tokens</th>
                <th>t/s</th>
                <th>Circuit</th>
            </tr>
        </thead>
        <tbody id="backends-tbody">
            <tr><td colspan="7" style="text-align:center;color:#aaa">Waiting for data...</td></tr>
        </tbody>
    </table>

    <div id="extra-section" class="extra-stats" style="display:none">
        <div id="cache-card" class="extra-card" style="display:none">
            <h3>Cache Statistics</h3>
            <div id="cache-body"></div>
        </div>
        <div id="queue-card" class="extra-card" style="display:none">
            <h3>Queue Statistics</h3>
            <div id="queue-body"></div>
        </div>
    </div>

    <div class="footer">
        <p>View as JSON: <a href="?format=json">?format=json</a> | Live stream: <a href="/stats/stream">/stats/stream</a></p>
        <p>llamacpp load-balancer proxy &mdash; real-time via SSE</p>
    </div>

<script>
(function () {
    var rateWindow = [];
    var RATE_WINDOW_MS = 60000;

    function fmt(n) {
        if (n === undefined || n === null) return '--';
        return Number(n).toLocaleString();
    }

    function setConnected(ok) {
        var el = document.getElementById('conn-status');
        var lbl = document.getElementById('conn-label');
        el.className = 'connection-status ' + (ok ? 'connected' : 'disconnected');
        lbl.textContent = ok ? 'Live' : 'Disconnected';
    }

    function computeRate(totalRequests) {
        var now = Date.now();
        rateWindow.push({ts: now, v: totalRequests});
        while (rateWindow.length > 1 && rateWindow[0].ts < now - RATE_WINDOW_MS) {
            rateWindow.shift();
        }
        if (rateWindow.length < 2) return null;
        var oldest = rateWindow[0];
        var dtMin = (now - oldest.ts) / 60000;
        if (dtMin < 0.01) return null;
        return ((totalRequests - oldest.v) / dtMin).toFixed(1);
    }

    function kvRow(label, value) {
        return '<div class="kv-row"><span class="kv-label">' + label +
               '</span><span class="kv-val">' + value + '</span></div>';
    }

    function renderBackends(backends) {
        if (!backends || backends.length === 0) return;
        var rows = backends.map(function (b) {
            var healthClass = b.healthy ? 'healthy' : 'unhealthy';
            var healthTxt = b.healthy ? 'Healthy' : 'Unhealthy';
            var circClass = 'circuit-' + (b.circuit_state || 'closed');
            var tps = b.tps || 0;
            var tpsClass = tps === 0 ? 'tps-idle' : (tps >= 10 ? 'tps-high' : (tps >= 3 ? 'tps-mid' : 'tps-low'));
            var tpsTxt = tps === 0 ? '—' : tps.toFixed(1);
            return '<tr>' +
                '<td><code>' + b.url + '</code></td>' +
                '<td class="' + healthClass + '">' + healthTxt + '</td>' +
                '<td>' + fmt(b.active_requests) + '</td>' +
                '<td>' + fmt(b.total_requests) + '</td>' +
                '<td>' + fmt(b.total_tokens) + '</td>' +
                '<td class="' + tpsClass + '">' + tpsTxt + '</td>' +
                '<td class="' + circClass + '">' + (b.circuit_state || 'closed') + '</td>' +
                '</tr>';
        }).join('');
        document.getElementById('backends-tbody').innerHTML = rows;
    }

    function renderCache(cache) {
        if (!cache) { document.getElementById('cache-card').style.display = 'none'; return; }
        document.getElementById('cache-card').style.display = '';
        var hits = cache.hits || 0;
        var misses = cache.misses || 0;
        var total = hits + misses;
        var hitRate = total > 0 ? ((hits / total) * 100).toFixed(1) + '%' : 'N/A';
        var html = kvRow('Hits', fmt(hits)) +
                   kvRow('Misses', fmt(misses)) +
                   kvRow('Hit Rate', hitRate);
        if (cache.size !== undefined) html += kvRow('Cached Entries', fmt(cache.size));
        if (cache.evictions !== undefined) html += kvRow('Evictions', fmt(cache.evictions));
        document.getElementById('cache-body').innerHTML = html;
    }

    function renderQueue(queue) {
        if (!queue) { document.getElementById('queue-card').style.display = 'none'; return; }
        document.getElementById('queue-card').style.display = '';
        var html = kvRow('Current Depth', fmt(queue.size)) +
                   kvRow('Total Queued', fmt(queue.total_queued));
        document.getElementById('queue-body').innerHTML = html;
    }

    function onData(data) {
        document.getElementById('stat-total-requests').textContent = fmt(data.total_requests);
        document.getElementById('stat-prompt-tokens').textContent = fmt(data.total_prompt_tokens);
        document.getElementById('stat-completion-tokens').textContent = fmt(data.total_completion_tokens);
        var totalTok = (data.total_prompt_tokens || 0) + (data.total_completion_tokens || 0);
        document.getElementById('stat-total-tokens').textContent = fmt(totalTok);
        document.getElementById('stat-healthy-backends').textContent =
            (data.healthy_backends || 0) + ' / ' + (data.total_backends || 0);

        var rate = computeRate(data.total_requests || 0);
        document.getElementById('stat-req-rate').textContent = rate !== null ? rate : '0.0';

        renderBackends(data.backends);
        renderCache(data.cache);
        renderQueue(data.queue);

        var hasExtra = data.cache || data.queue;
        document.getElementById('extra-section').style.display = hasExtra ? '' : 'none';

        document.getElementById('last-updated').textContent =
            'Last updated: ' + new Date().toLocaleTimeString();
    }

    function connect() {
        setConnected(false);
        var es = new EventSource('/stats/stream');

        es.onopen = function () { setConnected(true); };

        es.onmessage = function (evt) {
            try {
                var data = JSON.parse(evt.data);
                onData(data);
            } catch (e) { /* ignore parse errors */ }
        };

        es.onerror = function () {
            setConnected(false);
            es.close();
            setTimeout(connect, 3000);
        };
    }

    connect();
})();
</script>
</body>
</html>
"""
        return HTMLResponse(content=html)

    @app.get("/stats/stream", tags=["Health & Stats"])
    async def stats_stream(request: Request) -> StreamingResponse:
        """Server-Sent Events stream for real-time stats.

        Opens a persistent connection that pushes updated statistics every 2 seconds.
        Clients should connect with the EventSource API.

        ## Event Format

        Each event is a JSON object:
        ```
        data: {"total_requests": 42, "healthy_backends": 2, ...}
        ```

        No authentication required.
        """

        async def generate():
            while True:
                if await request.is_disconnected():
                    break

                async with state.get_lock():
                    total_prompt_tokens = sum(b.total_prompt_tokens for b in state.backends)
                    total_completion_tokens = sum(b.total_completion_tokens for b in state.backends)
                    total_requests = sum(b.total_requests for b in state.backends)
                    healthy_count = sum(1 for b in state.backends if b.healthy)

                    payload: dict = {
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

                if state.response_cache:
                    payload["cache"] = state.response_cache.get_stats()

                if state.request_queue:
                    payload["queue"] = {
                        "size": state.request_queue.size(),
                        "total_queued": state.request_queue.total_queued,
                    }

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def run_lb_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    config_file: str | None = None,
    discover_subnet: str | None = None,
    discover_port: str = "8000",  # comma-separated ports, e.g. "8000,8001"
    backends: list[str] | None = None,
    auth_key: str | None = None,
    api_key: str | None = None,
    rate_limit_rpm: int = 60,
    rate_limit_tph: int = 1000000,
    log_level: str = "INFO",
    log_format: str = "json",
    max_request_size: int = 10 * 1024 * 1024,
    max_response_tokens: int = 32000,
    warm_models: list[str] | None = None,
    no_warm: bool = False,
    rediscover_interval: float = 60.0,
    allowed_ips: list[str] | None = None,
    request_log_file: str | None = None,
    request_log_failed_only: bool = False,
    request_log_max: int = 1000,
    stats_file: str | None = None,
) -> None:
    """Start the multi-backend load balancer proxy."""
    import socket
    import sys

    import uvicorn

    # Auth is opt-in: only use if explicitly provided
    # Don't auto-generate - makes discovery impossible
    if auth_key:
        print(
            f"{_timestamp()} [lb-proxy] Authentication enabled (key: {auth_key[:8]}...)", flush=True
        )
        print(
            f"{_timestamp()} [lb-proxy] Backends must include: Authorization: Bearer {auth_key}",
            flush=True,
        )
    else:
        print(
            f"{_timestamp()} [lb-proxy] Authentication disabled - all backends will be discovered",
            flush=True,
        )
        print(f"{_timestamp()} [lb-proxy] Use --auth-key to enable authentication", flush=True)

    # API key for client requests
    if api_key:
        print(
            f"{_timestamp()} [lb-proxy] Client API key required (key: {api_key[:8]}...)", flush=True
        )
        print(
            f"{_timestamp()} [lb-proxy] Clients must provide: Authorization: Bearer {api_key}",
            flush=True,
        )
    else:
        print(f"{_timestamp()} [lb-proxy] No API key required for clients", flush=True)

    # Check port availability
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            _s.bind((host, port))
        except OSError:
            print(
                f"{_timestamp()} [lb-proxy] Error: port {port} is already in use.\n"
                f"  Kill the existing process or use --port <N> to pick another port."
            )
            sys.exit(1)

    # Setup state
    state = ProxyState()
    state.auth_key = auth_key
    state.api_key = api_key
    state.max_request_size = max_request_size
    state.max_response_tokens = max_response_tokens
    state.rate_limiter = RateLimiter(rpm_limit=rate_limit_rpm, tph_limit=rate_limit_tph)

    # Configure model warmer
    if warm_models and not no_warm:
        state.model_warmer = ModelWarmer(popular_models=warm_models, warm_on_startup=True)
        print(
            f"{_timestamp()} [lb-proxy] Model warming enabled for: {', '.join(warm_models)}",
            flush=True,
        )
    elif no_warm:
        print(f"{_timestamp()} [lb-proxy] Model warming disabled (--no-warm)", flush=True)

    # Configure IP whitelist
    if allowed_ips:
        # Support comma-separated list in a single string entry
        expanded: list[str] = []
        for entry in allowed_ips:
            expanded.extend(e.strip() for e in entry.split(",") if e.strip())
        state.allowed_cidrs = expanded
        print(
            f"{_timestamp()} [lb-proxy] IP whitelist enabled: {', '.join(expanded)}", flush=True
        )
    else:
        print(f"{_timestamp()} [lb-proxy] IP whitelist disabled - all IPs allowed", flush=True)

    # Configure request logger
    if request_log_file:
        log_path = Path(request_log_file).expanduser().resolve()
        state.request_logger = RequestLogger(
            log_file=log_path,
            max_records=request_log_max,
            log_failed_only=request_log_failed_only,
        )
        mode = "failed-only" if request_log_failed_only else "all requests"
        print(
            f"{_timestamp()} [lb-proxy] Request logging enabled ({mode}): {log_path}", flush=True
        )

    # Configure structured logging
    configure_logging(log_level=log_level, log_format=log_format)

    # Configure stats persistence
    if stats_file is not None:
        state.stats_file = Path(stats_file).expanduser().resolve()
    else:
        state.stats_file = get_base_dir() / "lb_stats.json"
    print(
        f"{_timestamp()} [lb-proxy] Stats persistence file: {state.stats_file}",
        flush=True,
    )

    # Load config file
    if config_file:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            print(f"{_timestamp()} [lb-proxy] Config file not found: {config_path}")
            print(f"{_timestamp()} [lb-proxy] Creating default config…")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"backends": []}, indent=2))
        state.config_path = config_path
    else:
        # Default config path
        config_path = get_base_dir() / "lb_backends.json"
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"backends": []}, indent=2))
            print(f"{_timestamp()} [lb-proxy] Created default config: {config_path}")
        state.config_path = config_path

    # Add CLI backends
    if backends:
        for backend_str in backends:
            try:
                if "://" in backend_str:
                    backend_str = backend_str.split("://", 1)[1]
                host_part, port_part = backend_str.rsplit(":", 1)
                backend = Backend(host=host_part, port=int(port_part))
                state.backends.append(backend)
            except Exception as exc:
                print(f"{_timestamp()} [lb-proxy] Invalid backend '{backend_str}': {exc}")

    # Initial load from config
    asyncio.run(_load_backends_from_config(state, auth_key))

    # Start the FastAPI app
    app = create_lb_app(state)

    # Store subnet/port info for periodic rediscovery (backends that start later)
    if discover_subnet:
        subnets = [s.strip() for s in discover_subnet.split(",")]
        ports = [int(p.strip()) for p in str(discover_port).split(",") if p.strip().isdigit()] or [8000]
        state.discover_subnets = subnets
        state.discover_ports = ports
        state.rediscover_interval = rediscover_interval if rediscover_interval > 0 else float("inf")

    # Schedule ONE-SHOT background discovery tasks (initial scan)
    discover_tasks = []
    if discover_subnet:
        subnets = [s.strip() for s in discover_subnet.split(",")]
        print(
            f"{_timestamp()} [lb-proxy] Starting background discovery for {len(subnets)} subnet(s)...",
            flush=True,
        )

        async def _discover_and_check(subnet: str) -> None:
            """Discover backends on a subnet in the background."""
            # Parse comma-separated ports once
            ports = [int(p.strip()) for p in str(discover_port).split(",") if p.strip().isdigit()]
            if not ports:
                ports = [8000]

            # Semaphore limits concurrent probes so no host is starved by
            # httpx connection-pool exhaustion on large /24 subnets.
            # 50 concurrent × 2s connect timeout = full /24 × 2 ports in ~20s.
            sem = asyncio.Semaphore(50)

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=10.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            ) as client:
                print(
                    f"{_timestamp()} [lb-proxy] Scanning {subnet} for backends on port(s) {ports}…",
                    flush=True,
                )

                import ipaddress

                try:
                    network = ipaddress.ip_network(subnet, strict=False)
                except ValueError as exc:
                    print(f"{_timestamp()} [lb-proxy] Invalid subnet {subnet}: {exc}", flush=True)
                    return

                tasks = []

                async def _try_host(host: str, port: int) -> Backend | None:
                    async with sem:
                        backend = Backend(host=host, port=port)
                        if await _check_backend_health(backend, client, auth_key, verbose=False):
                            await _refresh_backend_models(backend, client, auth_key)
                            backend.last_health_check = time.time()
                            return backend
                        return None

                for ip in network.hosts():
                    for p in ports:
                        tasks.append(_try_host(str(ip), p))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                async with state.get_lock():
                    for result in results:
                        if isinstance(result, Backend):
                            # Check if already exists
                            if any(
                                b.host == result.host and b.port == result.port
                                for b in state.backends
                            ):
                                continue
                            state.backends.append(result)
                            print(
                                f"{_timestamp()} [lb-proxy] Discovered backend: {result.url}, models: {result.models}",
                                flush=True,
                            )

                print(f"{_timestamp()} [lb-proxy] Completed scan of {subnet}", flush=True)

        for subnet in subnets:
            discover_tasks.append(_discover_and_check(subnet))

    # Load persisted stats into matching backends
    _load_stats(state)

    # Initial health check for any pre-configured backends
    async def _initial_checks() -> None:
        for backend in state.backends:
            healthy = await _check_backend_health(backend, state.http_client, auth_key)
            backend.healthy = healthy
            backend.last_health_check = time.time()
            if healthy:
                await _refresh_backend_models(backend, state.http_client, auth_key)
                print(
                    f"{_timestamp()} [lb-proxy] Backend {backend.url} ready, models: {backend.models}",
                    flush=True,
                )
            else:
                print(f"{_timestamp()} [lb-proxy] Backend {backend.url} unhealthy", flush=True)

    asyncio.run(_initial_checks())

    print(f"\n{_timestamp()} llamacpp load-balancer proxy listening on {host}:{port}", flush=True)
    print(f"{_timestamp()} Config: {state.config_path}", flush=True)
    print(
        f"{_timestamp()} Backends: {len([b for b in state.backends if b.healthy])}/{len(state.backends)} healthy",
        flush=True,
    )
    if discover_subnet:
        print(f"{_timestamp()} Discovery running in background for: {discover_subnet}", flush=True)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Start background discovery in a separate thread
    if discover_tasks:
        import threading

        def _run_discovery():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(asyncio.gather(*discover_tasks))
            finally:
                loop.close()

        discovery_thread = threading.Thread(target=_run_discovery, daemon=True)
        discovery_thread.start()

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{_timestamp()} [lb-proxy] Shutting down…", flush=True)
        asyncio.run(state.http_client.aclose())
        print(f"{_timestamp()} [lb-proxy] Stopped.", flush=True)
        sys.exit(0)
