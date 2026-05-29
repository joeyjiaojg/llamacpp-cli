"""Model warming for lb-proxy to reduce cold start latency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .lb_proxy import Backend


def _timestamp() -> str:
    """Return current timestamp in format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


@dataclass
class WarmingResult:
    """Result of a warming attempt for a single backend+model pair."""

    backend_url: str
    model: str
    success: bool
    error: str | None = None


@dataclass
class ModelWarmer:
    """Proactively loads popular models on backends to reduce cold start latency.

    Sends minimal 1-token requests to trigger model loading before real traffic
    arrives. Skips already-loaded models to avoid unnecessary work.

    Attributes:
        popular_models: Models to keep warm across all healthy backends.
        warm_on_startup: Whether to run warming immediately on startup.
        warm_interval: Seconds between re-warm checks (default: 5 minutes).
    """

    popular_models: list[str]
    warm_on_startup: bool = True
    warm_interval: float = 300.0

    # Warming state - tracks last status per (backend_url, model)
    _warming_status: dict[tuple[str, str], bool] = field(default_factory=dict)
    _total_warmed: int = 0
    _total_failed: int = 0
    _warming_task: asyncio.Task | None = None

    async def warm_model(
        self, backend: "Backend", model: str, client: httpx.AsyncClient
    ) -> bool:
        """Send a minimal 1-token request to load the model on a backend.

        Args:
            backend: The backend to warm.
            model: Model identifier to load.
            client: Async HTTP client to use.

        Returns:
            True if warming succeeded, False otherwise.
        """
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
            }
            resp = await client.post(
                f"{backend.url}/v1/chat/completions",
                json=payload,
                timeout=120.0,  # Model load can take 10-60s
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def warm_all_backends(
        self, backends: list["Backend"], client: httpx.AsyncClient
    ) -> list[WarmingResult]:
        """Warm popular models on all healthy backends.

        Skips models already registered on the backend. Updates backend.models
        when warming succeeds so subsequent routing reflects the loaded model.

        Args:
            backends: List of all backends (healthy and unhealthy).
            client: Async HTTP client to use.

        Returns:
            List of WarmingResult for each (backend, model) pair attempted.
        """
        results: list[WarmingResult] = []

        for backend in backends:
            if not backend.healthy:
                continue

            for model in self.popular_models:
                if model in backend.models:
                    # Already loaded — nothing to do
                    continue

                print(
                    f"{_timestamp()} [warmer] Warming {model} on {backend.url}...",
                    flush=True,
                )
                success = await self.warm_model(backend, model, client)
                key = (backend.url, model)

                if success:
                    if model not in backend.models:
                        backend.models.append(model)
                    self._total_warmed += 1
                    self._warming_status[key] = True
                    print(
                        f"{_timestamp()} [warmer] Warmed {model} on {backend.url}",
                        flush=True,
                    )
                    results.append(WarmingResult(backend_url=backend.url, model=model, success=True))
                else:
                    self._total_failed += 1
                    self._warming_status[key] = False
                    print(
                        f"{_timestamp()} [warmer] Failed to warm {model} on {backend.url}",
                        flush=True,
                    )
                    results.append(
                        WarmingResult(
                            backend_url=backend.url,
                            model=model,
                            success=False,
                            error="warming request failed",
                        )
                    )

        return results

    async def _warm_loop(
        self, get_backends: "GetBackendsCallback", client: httpx.AsyncClient
    ) -> None:
        """Background loop that periodically re-warms backends.

        Args:
            get_backends: Async callable that returns the current backend list.
            client: Async HTTP client to use.
        """
        while True:
            try:
                backends = await get_backends()
                await self.warm_all_backends(backends, client)
            except Exception as exc:
                print(f"{_timestamp()} [warmer] Error in warm loop: {exc}", flush=True)
            await asyncio.sleep(self.warm_interval)

    def start(
        self, get_backends: "GetBackendsCallback", client: httpx.AsyncClient
    ) -> asyncio.Task:
        """Start the background warming loop.

        Args:
            get_backends: Async callable that returns the current backend list.
            client: Async HTTP client to use.

        Returns:
            The asyncio.Task for the background loop.
        """
        self._warming_task = asyncio.create_task(self._warm_loop(get_backends, client))
        return self._warming_task

    def stop(self) -> None:
        """Cancel the background warming task if running."""
        if self._warming_task and not self._warming_task.done():
            self._warming_task.cancel()
            self._warming_task = None

    def get_status(self) -> dict:
        """Return current warming status for all (backend, model) pairs.

        Returns:
            Dict with warming statistics and per-pair status.
        """
        return {
            "popular_models": self.popular_models,
            "warm_on_startup": self.warm_on_startup,
            "warm_interval": self.warm_interval,
            "total_warmed": self._total_warmed,
            "total_failed": self._total_failed,
            "warming_active": self._warming_task is not None and not self._warming_task.done(),
            "model_status": {
                f"{url}::{model}": ok for (url, model), ok in self._warming_status.items()
            },
        }


# Type alias for the callback used by start() / _warm_loop()
from collections.abc import Callable, Coroutine
from typing import Any

GetBackendsCallback = Callable[[], Coroutine[Any, Any, list["Backend"]]]
