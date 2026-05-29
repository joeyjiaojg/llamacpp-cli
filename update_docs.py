#!/usr/bin/env python3
"""Script to add comprehensive docstrings to lb_proxy.py endpoints."""

import re

# Read the file
with open("src/llamacpp_cli/lb_proxy.py", "r") as f:
    content = f.read()

# Define replacements
replacements = [
    # Props endpoint
    (
        r'@app\.get\("/props"\)\s+async def aggregate_props\(request: Request\) -> JSONResponse:\s+"""Aggregate server properties from all healthy backends\."""',
        '''@app.get("/props", tags=["Management"])
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
        """''',
    ),
    # Metrics endpoint
    (
        r'@app\.get\("/metrics"\)\s+async def aggregate_metrics\(request: Request\) -> Response:\s+"""Aggregate Prometheus metrics from all healthy backends\."""',
        '''@app.get("/metrics", tags=["Health & Stats"])
    async def aggregate_metrics(request: Request) -> Response:
        """Aggregate Prometheus metrics from all healthy backends.

        Returns Prometheus-format metrics aggregated from all backends.
        Metrics are prefixed with backend identifiers for disambiguation.

        ## Response Format

        Plain text in Prometheus exposition format.

        ## Authentication

        This endpoint does not require authentication (for monitoring tools).
        """''',
    ),
    # List models endpoint
    (
        r'@app\.get\("/v1/models"\)\s+async def list_models\(request: Request\) -> JSONResponse:\s+"""Aggregate models from all healthy backends\."""',
        '''@app.get("/v1/models", tags=["OpenAI API"])
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
        """''',
    ),
    # Health endpoint
    (
        r'@app\.get\("/health"\)\s+async def health\(\) -> JSONResponse:',
        '''@app.get("/health", tags=["Health & Stats"])
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
        """''',
    ),
    # Backends endpoint
    (
        r'@app\.get\("/backends"\)\s+@app\.get\("/v1/backends"\)\s+async def list_backends\(request: Request\) -> JSONResponse:\s+"""List all backends and their status \(load-aware\)\."""',
        '''@app.get("/backends", tags=["Management"])
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
        """''',
    ),
    # Stats endpoint
    (
        r'@app\.get\("/stats"\)\s+@app\.get\("/v1/stats"\)\s+async def stats\(request: Request, format: str \| None = None\) -> Response:\s+"""Get token usage statistics \(no authentication required\)\."""',
        '''@app.get("/stats", tags=["Health & Stats"])
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
        """''',
    ),
    # Legacy engines endpoint
    (
        r'@app\.get\("/v1/engines"\)\s+async def list_engines\(request: Request\) -> JSONResponse:\s+"""Legacy endpoint - alias for /v1/models\."""',
        '''@app.get("/v1/engines", tags=["Legacy"])
    async def list_engines(request: Request) -> JSONResponse:
        """Legacy OpenAI engines endpoint (alias for /v1/models).

        Provided for backward compatibility with older OpenAI API clients.

        See `/v1/models` for details.
        """''',
    ),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("✓ Updated docstrings for all endpoints")
