"""Multi-backend load balancer proxy for llama.cpp servers.

Routes requests to multiple llama-server instances running on different machines,
with model-aware routing and least-connections load balancing.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_base_dir


def _timestamp() -> str:
    """Return current timestamp in format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


@dataclass
class Backend:
    """A backend llama-server instance."""

    host: str
    port: int
    models: list[str] = field(default_factory=list)
    active_requests: int = 0
    healthy: bool = True
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    checking: bool = False

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
            timeout=30.0,
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
    backend: Backend,
    client: httpx.AsyncClient,
    auth_key: str | None = None,
    verbose: bool = False
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
                    print(f"{_timestamp()} [lb-proxy] {backend.url} rejected: JSON parse error - {e}", flush=True)
                return False

            # Validate OpenAI-compatible response format
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                # Check auth key if backend sends it back
                if auth_key:
                    backend_key = resp.headers.get("Authorization")
                    if backend_key and backend_key != f"Bearer {auth_key}":
                        if verbose:
                            print(f"{_timestamp()} [lb-proxy] {backend.url} rejected: auth key mismatch", flush=True)
                        return False
                return True
            else:
                if verbose:
                    print(f"{_timestamp()} [lb-proxy] {backend.url} rejected: invalid format - data={type(data)}, has_data={'data' in data if isinstance(data, dict) else False}", flush=True)
        else:
            if verbose:
                print(f"{_timestamp()} [lb-proxy] {backend.url} rejected: status {resp.status_code}", flush=True)
    except Exception as e:
        if verbose:
            print(f"{_timestamp()} [lb-proxy] {backend.url} rejected: {type(e).__name__}: {e}", flush=True)

    return False


async def _refresh_backend_models(backend: Backend, client: httpx.AsyncClient) -> None:
    """Query a backend for available models and update its model list."""
    try:
        resp = await client.get(f"{backend.url}/v1/models", timeout=10.0)
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
                if backend.checking or now - backend.last_health_check < state.health_check_interval:
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
                    else:
                        backend.consecutive_failures += 1
                        backend.consecutive_successes = 0

                    # Only change healthy status after thresholds are met
                    should_mark_healthy = not backend.healthy and backend.consecutive_successes >= SUCCESS_THRESHOLD
                    should_mark_unhealthy = backend.healthy and backend.consecutive_failures >= FAILURE_THRESHOLD

                    if should_mark_healthy:
                        backend.healthy = True
                        await _refresh_backend_models(backend, state.http_client)
                        print(
                            f"{_timestamp()} [lb-proxy] Backend {backend.url} became healthy "
                            f"(after {backend.consecutive_successes} consecutive successes), models: {backend.models}",
                            flush=True
                        )
                    elif should_mark_unhealthy:
                        backend.healthy = False
                        print(
                            f"{_timestamp()} [lb-proxy] Backend {backend.url} became unhealthy "
                            f"(after {backend.consecutive_failures} consecutive failures)",
                            flush=True
                        )
                    elif backend.healthy:
                        # Refresh models silently if still healthy
                        await _refresh_backend_models(backend, state.http_client)
                finally:
                    backend.checking = False


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
            backend = Backend(host=b["host"], port=b["port"])
            new_backends.append(backend)

        async with state.get_lock():
            existing = {(b.host, b.port) for b in state.backends}
            new = {(b.host, b.port) for b in new_backends}

            # Add new backends
            added = new - existing
            if added:
                for host, port in added:
                    backend = Backend(host=host, port=port)
                    state.backends.append(backend)
                    print(f"{_timestamp()} [lb-proxy] Added backend: {backend.url}")
                    # Trigger immediate health check
                    healthy = await _check_backend_health(backend, state.http_client, auth_key)
                    backend.healthy = healthy
                    backend.last_health_check = time.time()
                    if healthy:
                        await _refresh_backend_models(backend, state.http_client)

            # Remove deleted backends
            removed = existing - new
            if removed:
                state.backends = [
                    b for b in state.backends if (b.host, b.port) not in removed
                ]
                for host, port in removed:
                    print(f"{_timestamp()} [lb-proxy] Removed backend: http://{host}:{port}")

    except Exception as exc:
        print(f"{_timestamp()} [lb-proxy] Error loading config: {exc}")


async def _discover_backends_on_subnet(
    state: ProxyState, subnet: str, port: int
) -> None:
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
        if await _check_backend_health(backend, state.http_client):
            await _refresh_backend_models(backend, state.http_client)
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
                print(f"{_timestamp()} [lb-proxy] Discovered backend: {result.url}, models: {result.models}")


def _select_backend(
    backends: list[Backend], model: str | None = None
) -> Backend | None:
    """Select the best backend using model-aware + least-connections routing."""
    healthy = [b for b in backends if b.healthy]
    if not healthy:
        return None

    # Model-aware routing: filter to backends that have the requested model
    if model:
        candidates = [b for b in healthy if model in b.models]
        if candidates:
            healthy = candidates

    # Least-connections: pick backend with fewest active requests
    return min(healthy, key=lambda b: b.active_requests)


async def _forward_request(
    request: Request, backend: Backend, state: ProxyState
) -> Response:
    """Forward request to a backend and stream back the response."""
    url = f"{backend.url}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

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

    backend.active_requests += 1
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


def create_lb_app(state: ProxyState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        # Start background tasks
        state.health_check_task = asyncio.create_task(_health_check_loop(state, state.auth_key))
        if state.config_path:
            state.config_watch_task = asyncio.create_task(_config_watch_loop(state, state.auth_key))
        yield
        # Shutdown
        if state.health_check_task:
            state.health_check_task.cancel()
        if state.config_watch_task:
            state.config_watch_task.cancel()

    app = FastAPI(title="llamacpp-lb-proxy", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

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
            raise HTTPException(
                status_code=503,
                detail="No healthy backends available" if not model else f"No backends available for model '{model}'",
            )

        # Forward request
        return await _forward_request(request, backend, state)

    @app.post("/v1/completions")
    @app.post("/v1/embeddings")
    async def other_endpoints(request: Request) -> Response:
        """Handle other OpenAI endpoints (completions, embeddings, etc.)."""
        # Validate API key
        if not state.validate_api_key(request):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide: Authorization: Bearer YOUR_API_KEY",
            )

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
            raise HTTPException(
                status_code=503,
                detail="No healthy backends available" if not model else f"No backends available for model '{model}'",
            )

        # Forward request
        return await _forward_request(request, backend, state)

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        """Aggregate models from all healthy backends."""
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

    @app.get("/health")
    async def health() -> JSONResponse:
        async with state.get_lock():
            healthy_count = sum(1 for b in state.backends if b.healthy)
            total_count = len(state.backends)

        return JSONResponse({
            "status": "ok" if healthy_count > 0 else "degraded",
            "backends": {"healthy": healthy_count, "total": total_count},
        })

    @app.get("/backends")
    @app.get("/v1/backends")
    async def list_backends(request: Request) -> JSONResponse:
        """List all backends and their status (load-aware)."""
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
                }
                for b in state.backends
            ]
        return JSONResponse({"backends": backends_data})

    return app


def run_lb_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    config_file: str | None = None,
    discover_subnet: str | None = None,
    discover_port: int = 8000,
    backends: list[str] | None = None,
    auth_key: str | None = None,
    api_key: str | None = None,
) -> None:
    """Start the multi-backend load balancer proxy."""
    import secrets
    import socket
    import sys

    import uvicorn

    # Auth is opt-in: only use if explicitly provided
    # Don't auto-generate - makes discovery impossible
    if auth_key:
        print(f"{_timestamp()} [lb-proxy] Authentication enabled (key: {auth_key[:8]}...)", flush=True)
        print(f"{_timestamp()} [lb-proxy] Backends must include: Authorization: Bearer {auth_key}", flush=True)
    else:
        print(f"{_timestamp()} [lb-proxy] Authentication disabled - all backends will be discovered", flush=True)
        print(f"{_timestamp()} [lb-proxy] Use --auth-key to enable authentication", flush=True)

    # API key for client requests
    if api_key:
        print(f"{_timestamp()} [lb-proxy] Client API key required (key: {api_key[:8]}...)", flush=True)
        print(f"{_timestamp()} [lb-proxy] Clients must provide: Authorization: Bearer {api_key}", flush=True)
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

    # Schedule background discovery tasks
    discover_tasks = []
    if discover_subnet:
        subnets = [s.strip() for s in discover_subnet.split(",")]
        print(f"{_timestamp()} [lb-proxy] Starting background discovery for {len(subnets)} subnet(s)...", flush=True)

        async def _discover_and_check(subnet: str) -> None:
            """Discover backends on a subnet in the background."""
            # Create a new httpx client for this thread (httpx clients are not thread-safe)
            async with httpx.AsyncClient(timeout=5.0) as client:
                print(f"{_timestamp()} [lb-proxy] Scanning {subnet} for backends on port {discover_port}…", flush=True)

                # Modified version of _discover_backends_on_subnet that uses local client
                import ipaddress
                try:
                    network = ipaddress.ip_network(subnet, strict=False)
                except ValueError as exc:
                    print(f"{_timestamp()} [lb-proxy] Invalid subnet {subnet}: {exc}", flush=True)
                    return

                tasks = []
                async def _try_host(host: str) -> Backend | None:
                    backend = Backend(host=host, port=discover_port)
                    if await _check_backend_health(backend, client, auth_key, verbose=True):
                        await _refresh_backend_models(backend, client)
                        backend.last_health_check = time.time()
                        return backend
                    return None

                for ip in network.hosts():
                    tasks.append(_try_host(str(ip)))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                async with state.get_lock():
                    for result in results:
                        if isinstance(result, Backend):
                            # Check if already exists
                            if any(b.host == result.host and b.port == result.port for b in state.backends):
                                continue
                            state.backends.append(result)
                            print(f"{_timestamp()} [lb-proxy] Discovered backend: {result.url}, models: {result.models}", flush=True)

                print(f"{_timestamp()} [lb-proxy] Completed scan of {subnet}", flush=True)

        for subnet in subnets:
            discover_tasks.append(_discover_and_check(subnet))

    # Initial health check for any pre-configured backends
    async def _initial_checks() -> None:
        for backend in state.backends:
            healthy = await _check_backend_health(backend, state.http_client, auth_key)
            backend.healthy = healthy
            backend.last_health_check = time.time()
            if healthy:
                await _refresh_backend_models(backend, state.http_client)
                print(f"{_timestamp()} [lb-proxy] Backend {backend.url} ready, models: {backend.models}", flush=True)
            else:
                print(f"{_timestamp()} [lb-proxy] Backend {backend.url} unhealthy", flush=True)

    asyncio.run(_initial_checks())

    print(f"\n{_timestamp()} llamacpp load-balancer proxy listening on {host}:{port}", flush=True)
    print(f"{_timestamp()} Config: {state.config_path}", flush=True)
    print(f"{_timestamp()} Backends: {len([b for b in state.backends if b.healthy])}/{len(state.backends)} healthy", flush=True)
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
