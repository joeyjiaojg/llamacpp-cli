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
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_config_dir


@dataclass
class Backend:
    """A backend llama-server instance."""

    host: str
    port: int
    models: list[str] = field(default_factory=list)
    active_requests: int = 0
    healthy: bool = True
    last_health_check: float = 0.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __hash__(self) -> int:
        return hash((self.host, self.port))


@dataclass
class ProxyState:
    """Shared state for the load balancer proxy."""

    backends: list[Backend] = field(default_factory=list)
    backends_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    http_client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=30.0))
    health_check_interval: float = 10.0
    health_check_task: asyncio.Task | None = None
    config_path: Path | None = None
    config_watch_task: asyncio.Task | None = None


async def _check_backend_health(backend: Backend, client: httpx.AsyncClient) -> bool:
    """Check if a backend is healthy by querying /health or /v1/models."""
    try:
        # Try /health first
        resp = await client.get(f"{backend.url}/health", timeout=5.0)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    try:
        # Fall back to /v1/models
        resp = await client.get(f"{backend.url}/v1/models", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


async def _refresh_backend_models(backend: Backend, client: httpx.AsyncClient) -> None:
    """Query a backend for available models and update its model list."""
    try:
        resp = await client.get(f"{backend.url}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and isinstance(data["data"], list):
                backend.models = [m.get("id", "") for m in data["data"] if m.get("id")]
    except Exception:
        pass


async def _health_check_loop(state: ProxyState) -> None:
    """Periodically check backend health and refresh model lists."""
    while True:
        await asyncio.sleep(state.health_check_interval)
        async with state.backends_lock:
            for backend in state.backends:
                now = time.time()
                if now - backend.last_health_check < state.health_check_interval:
                    continue

                healthy = await _check_backend_health(backend, state.http_client)
                backend.healthy = healthy
                backend.last_health_check = now

                if healthy:
                    await _refresh_backend_models(backend, state.http_client)
                    print(f"[lb-proxy] Backend {backend.url} healthy, models: {backend.models}")
                else:
                    print(f"[lb-proxy] Backend {backend.url} unhealthy")


async def _config_watch_loop(state: ProxyState) -> None:
    """Watch config file for changes and auto-reload backends."""
    if not state.config_path or not state.config_path.exists():
        return

    last_mtime = state.config_path.stat().st_mtime

    while True:
        await asyncio.sleep(5.0)
        try:
            current_mtime = state.config_path.stat().st_mtime
            if current_mtime > last_mtime:
                print(f"[lb-proxy] Config file changed, reloading backends…")
                await _load_backends_from_config(state)
                last_mtime = current_mtime
        except Exception as exc:
            print(f"[lb-proxy] Error watching config: {exc}")


async def _load_backends_from_config(state: ProxyState) -> None:
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

        async with state.backends_lock:
            existing = {(b.host, b.port) for b in state.backends}
            new = {(b.host, b.port) for b in new_backends}

            # Add new backends
            added = new - existing
            if added:
                for host, port in added:
                    backend = Backend(host=host, port=port)
                    state.backends.append(backend)
                    print(f"[lb-proxy] Added backend: {backend.url}")
                    # Trigger immediate health check
                    healthy = await _check_backend_health(backend, state.http_client)
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
                    print(f"[lb-proxy] Removed backend: http://{host}:{port}")

    except Exception as exc:
        print(f"[lb-proxy] Error loading config: {exc}")


async def _discover_backends_on_subnet(
    state: ProxyState, subnet: str, port: int
) -> None:
    """Scan a subnet for llama-server instances on a given port."""
    import ipaddress

    try:
        network = ipaddress.ip_network(subnet, strict=False)
    except ValueError as exc:
        print(f"[lb-proxy] Invalid subnet {subnet}: {exc}")
        return

    print(f"[lb-proxy] Scanning {subnet} for backends on port {port}…")
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
                print(f"[lb-proxy] Discovered backend: {result.url}, models: {result.models}")


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
        state.health_check_task = asyncio.create_task(_health_check_loop(state))
        if state.config_path:
            state.config_watch_task = asyncio.create_task(_config_watch_loop(state))
        yield
        # Shutdown
        if state.health_check_task:
            state.health_check_task.cancel()
        if state.config_watch_task:
            state.config_watch_task.cancel()

    app = FastAPI(title="llamacpp-lb-proxy", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        # Extract model from request
        try:
            body: Any = await request.json()
            model = body.get("model") if isinstance(body, dict) else None
        except Exception:
            model = None

        # Select backend
        async with state.backends_lock:
            backend = _select_backend(state.backends, model)

        if not backend:
            raise HTTPException(
                status_code=503,
                detail="No healthy backends available" if not model else f"No backends available for model '{model}'",
            )

        # Forward request
        return await _forward_request(request, backend, state)

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """Aggregate models from all healthy backends."""
        models_set = set()
        async with state.backends_lock:
            for backend in state.backends:
                if backend.healthy:
                    models_set.update(backend.models)

        models_data = [{"id": model, "object": "model"} for model in sorted(models_set)]
        return JSONResponse({"object": "list", "data": models_data})

    @app.get("/health")
    async def health() -> JSONResponse:
        async with state.backends_lock:
            healthy_count = sum(1 for b in state.backends if b.healthy)
            total_count = len(state.backends)

        return JSONResponse({
            "status": "ok" if healthy_count > 0 else "degraded",
            "backends": {"healthy": healthy_count, "total": total_count},
        })

    @app.get("/backends")
    async def list_backends() -> JSONResponse:
        """List all backends and their status."""
        async with state.backends_lock:
            backends_data = [
                {
                    "url": b.url,
                    "healthy": b.healthy,
                    "models": b.models,
                    "active_requests": b.active_requests,
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
) -> None:
    """Start the multi-backend load balancer proxy."""
    import socket
    import sys

    import uvicorn

    # Check port availability
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            _s.bind((host, port))
        except OSError:
            print(
                f"[lb-proxy] Error: port {port} is already in use.\n"
                f"  Kill the existing process or use --port <N> to pick another port."
            )
            sys.exit(1)

    # Setup state
    state = ProxyState()

    # Load config file
    if config_file:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            print(f"[lb-proxy] Config file not found: {config_path}")
            print("[lb-proxy] Creating default config…")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"backends": []}, indent=2))
        state.config_path = config_path
    else:
        # Default config path
        config_path = get_config_dir() / "lb_backends.json"
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"backends": []}, indent=2))
            print(f"[lb-proxy] Created default config: {config_path}")
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
                print(f"[lb-proxy] Invalid backend '{backend_str}': {exc}")

    # Initial load from config
    asyncio.run(_load_backends_from_config(state))

    # Discover backends on subnet
    if discover_subnet:
        asyncio.run(_discover_backends_on_subnet(state, discover_subnet, discover_port))

    if not state.backends:
        print("[lb-proxy] No backends configured. Add backends to:")
        print(f"  {state.config_path}")
        print("Or use --backend http://host:port")
        print("Or use --discover-subnet 192.168.1.0/24")
        sys.exit(1)

    # Initial health check
    async def _initial_checks() -> None:
        for backend in state.backends:
            healthy = await _check_backend_health(backend, state.http_client)
            backend.healthy = healthy
            backend.last_health_check = time.time()
            if healthy:
                await _refresh_backend_models(backend, state.http_client)
                print(f"[lb-proxy] Backend {backend.url} ready, models: {backend.models}")
            else:
                print(f"[lb-proxy] Backend {backend.url} unhealthy")

    asyncio.run(_initial_checks())

    app = create_lb_app(state)

    print(f"\nllamacpp load-balancer proxy listening on {host}:{port}")
    print(f"Config: {state.config_path}")
    print(f"Backends: {len([b for b in state.backends if b.healthy])}/{len(state.backends)} healthy")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[lb-proxy] Shutting down…")
        asyncio.run(state.http_client.aclose())
        print("[lb-proxy] Stopped.")
        sys.exit(0)
