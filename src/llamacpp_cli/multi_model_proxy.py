"""Routing proxy that dispatches requests to model-specific backends.

Routes incoming /v1/chat/completions (and other /v1/* paths) to the
correct llama-server instance based on the ``model`` field in the JSON
request body.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_proxy_app(model_urls: dict[str, str]) -> FastAPI:
    """Build a FastAPI app that routes requests by model name.

    Args:
        model_urls: Mapping of model name -> backend base URL.

    Returns:
        Configured FastAPI application.
    """
    http_client: httpx.AsyncClient  # noqa: F821 – assigned in lifespan, used by closures

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=None)
        try:
            yield
        finally:
            await http_client.aclose()

    app = FastAPI(title="llamacpp multi-model proxy", lifespan=lifespan)

    def _pick_backend(body: dict[str, Any]) -> str | None:
        """Return backend URL for the requested model, or None if unknown."""
        requested = body.get("model", "")
        # Exact match first
        if requested in model_urls:
            return model_urls[requested]
        # Prefix / suffix match (e.g. short alias)
        for name, url in model_urls.items():
            if requested in name or name in requested:
                return url
        return None

    async def _proxy(request: Request, path: str) -> Response:
        body_bytes = await request.body()

        try:
            body_json: dict[str, Any] = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body_json = {}

        backend_url = _pick_backend(body_json)
        if backend_url is None:
            available = list(model_urls.keys())
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": (
                                f"Model '{body_json.get('model')}' not found. "
                                f"Available models: {available}"
                            ),
                            "type": "model_not_found",
                        }
                    }
                ),
                status_code=404,
                media_type="application/json",
            )

        target_url = f"{backend_url}/{path}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        is_streaming = body_json.get("stream", False)

        if is_streaming:
            async def _stream_response() -> AsyncIterator[bytes]:
                async with http_client.stream(  # noqa: F821
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body_bytes,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                _stream_response(),
                media_type="text/event-stream",
            )

        resp = await http_client.request(  # noqa: F821
            method=request.method,
            url=target_url,
            headers=headers,
            content=body_bytes,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Return health status of all backends."""
        statuses: dict[str, Any] = {}
        for name, url in model_urls.items():
            try:
                r = await http_client.get(f"{url}/health", timeout=2.0)  # noqa: F821
                statuses[name] = {"status": "healthy" if r.status_code == 200 else "unhealthy"}
            except Exception as exc:
                statuses[name] = {"status": "unreachable", "error": str(exc)}
        return {"backends": statuses}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List all served models in OpenAI-compatible format."""
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "llamacpp"}
                for name in model_urls
            ],
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(request: Request, path: str) -> Response:
        return await _proxy(request, path)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_multi_model_proxy(
    model_urls: dict[str, str],
    proxy_port: int = 8080,
    host: str = "127.0.0.1",
) -> None:
    """Start the routing proxy synchronously (blocks until interrupted).

    Args:
        model_urls: Mapping of model name -> backend base URL.
        proxy_port: Port to bind the proxy on.
        host: Host address to bind.
    """
    app = create_proxy_app(model_urls)
    uvicorn.run(app, host=host, port=proxy_port, log_level="warning")
