"""Async proxy that auto-loads llama.cpp models on demand.

Behaviour mirrors Ollama's serve: start with no model loaded, intercept
incoming requests, extract the requested model from the JSON body, pull the
model if it isn't on disk, restart llama-server with that model, then forward
the request.  Concurrent requests that arrive while the server is starting are
queued via asyncio.Event and replayed once the server is ready.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from contextlib import asynccontextmanager

from .db import get_model, list_models
from .run import _is_local_path
from .server import build_server_cmd, wait_until_ready


@dataclass
class ProxyState:
    current_model: str | None = None
    server_proc: subprocess.Popen | None = None
    # Lock ensures only one model-switch runs at a time.
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set when the backend server is ready to accept requests.
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    server_port: int = 8081
    extra_args: list[str] = field(default_factory=list)
    # Persistent HTTP client — must outlive individual StreamingResponse bodies.
    http_client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=None))


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _ensure_model_loaded(model: str, state: ProxyState) -> None:
    """Load *model* into llama-server, switching away from any current model."""
    async with state.load_lock:
        # Resolve to canonical name + path before comparing.
        model_info = get_model(model)
        canonical = model_info["name"] if model_info else model

        if state.current_model == canonical:
            return  # already loaded — nothing to do

        print(f"[proxy] Switching model: {state.current_model!r} -> {canonical!r}")
        state.ready_event.clear()

        # Stop existing backend.
        _stop_proc(state.server_proc)
        state.server_proc = None
        state.current_model = None

        # Resolve model path — pull if not yet on disk.
        if model_info:
            model_path = model_info["path"]
        elif _is_local_path(model):
            model_path = model
        else:
            print(f"[proxy] Pulling model '{model}'…")
            loop = asyncio.get_event_loop()
            from .model_manager import pull_model
            await loop.run_in_executor(None, pull_model, model)
            model_info = get_model(model)
            if not model_info:
                raise RuntimeError(f"Failed to pull model '{model}'.")
            canonical = model_info["name"]
            model_path = model_info["path"]

        # Start llama-server on the internal port.
        cmd = build_server_cmd(
            model_path,
            host="127.0.0.1",
            port=state.server_port,
            extra_args=state.extra_args or None,
        )
        print(f"[proxy] Starting llama-server on port {state.server_port} with {canonical!r}")
        state.server_proc = subprocess.Popen(cmd)
        state.current_model = canonical

        # Wait for /health to return 200.
        health_url = f"http://127.0.0.1:{state.server_port}/health"
        try:
            await wait_until_ready(health_url, timeout=120.0)
        except TimeoutError:
            _stop_proc(state.server_proc)
            state.server_proc = None
            state.current_model = None
            raise

        print(f"[proxy] Model '{model}' ready.")
        state.ready_event.set()


async def _extract_model(request: Request) -> str | None:
    """Return the 'model' field from a JSON request body, or None."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body: Any = await request.json()
            if isinstance(body, dict):
                return body.get("model") or None
        except Exception:
            pass
    return None


async def _forward_request(request: Request, state: ProxyState) -> Response:
    """Forward *request* to the llama-server backend and stream back the response.

    The persistent state.http_client is used so the connection stays open for
    the entire duration of the streamed response body.
    """
    backend_url = f"http://127.0.0.1:{state.server_port}"
    url = f"{backend_url}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    _HOP_BY_HOP = {
        "host", "connection", "keep-alive", "proxy-authenticate",
        "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    }
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

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
        k: v for k, v in backend_resp.headers.items()
        if k.lower() not in {"transfer-encoding", "connection"}
    }

    async def _stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in backend_resp.aiter_bytes():
                yield chunk
        finally:
            await backend_resp.aclose()

    return StreamingResponse(
        _stream(),
        status_code=backend_resp.status_code,
        headers=resp_headers,
        media_type=backend_resp.headers.get("content-type"),
    )


def create_app(state: ProxyState, default_model: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        if default_model:
            print(f"[proxy] Pre-loading default model '{default_model}'…")
            try:
                await _ensure_model_loaded(default_model, state)
            except Exception as exc:
                print(f"[proxy] Warning: failed to pre-load '{default_model}': {exc}")
        yield
        # Shutdown: nothing extra needed here — run_proxy handles cleanup.

    app = FastAPI(title="llamacpp-proxy", lifespan=lifespan)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> Response:  # noqa: ARG001
        model = await _extract_model(request)
        if model:
            await _ensure_model_loaded(model, state)
        else:
            # No model specified — wait for whatever is currently loading.
            await state.ready_event.wait()

        return await _forward_request(request, state)

    return app


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    server_port: int = 8081,
    extra_args: list[str] | None = None,
) -> None:
    """Start the proxy in the foreground (blocking). Ctrl+C shuts everything down."""
    import uvicorn

    # Pick the most recently downloaded model as the default.
    default_model: str | None = None
    models = list_models()
    if models:
        default_model = models[0]["name"]
        print(f"[proxy] Default model: '{default_model}'")
    else:
        print("[proxy] No models downloaded yet. Use 'llamacpp pull <model>' to add one.")

    state = ProxyState(server_port=server_port, extra_args=extra_args or [])
    app = create_app(state, default_model=default_model)

    print(f"llamacpp proxy listening on {host}:{port} (backend on 127.0.0.1:{server_port})")

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[proxy] Shutting down…")
        _stop_proc(state.server_proc)
        asyncio.run(state.http_client.aclose())
        print("[proxy] Stopped.")
        sys.exit(0)
