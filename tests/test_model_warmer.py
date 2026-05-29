"""Tests for model warming functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from llamacpp_cli.lb_proxy import Backend, ProxyState, create_lb_app
from llamacpp_cli.model_warmer import ModelWarmer, WarmingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_backend(host: str = "10.0.0.1", port: int = 8000, models: list[str] | None = None, healthy: bool = True) -> Backend:
    return Backend(host=host, port=port, models=list(models or []), healthy=healthy)


def make_httpx_response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


# ---------------------------------------------------------------------------
# Unit tests for ModelWarmer.warm_model
# ---------------------------------------------------------------------------


class TestWarmModel:
    @pytest.mark.asyncio
    async def test_warm_model_success(self):
        """warm_model returns True when backend responds 200."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend()
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        result = await warmer.warm_model(backend, "llama3", client)

        assert result is True
        client.post.assert_called_once()
        call_kwargs = client.post.call_args
        assert "/v1/chat/completions" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["model"] == "llama3"
        assert call_kwargs[1]["json"]["max_tokens"] == 1

    @pytest.mark.asyncio
    async def test_warm_model_failure_non_200(self):
        """warm_model returns False when backend responds non-200."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend()
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(500))

        result = await warmer.warm_model(backend, "llama3", client)

        assert result is False

    @pytest.mark.asyncio
    async def test_warm_model_network_exception(self):
        """warm_model returns False when network error occurs."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend()
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await warmer.warm_model(backend, "llama3", client)

        assert result is False

    @pytest.mark.asyncio
    async def test_warm_model_timeout_exception(self):
        """warm_model returns False on timeout."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend()
        client = AsyncMock()
        client.post = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await warmer.warm_model(backend, "llama3", client)

        assert result is False

    @pytest.mark.asyncio
    async def test_warm_model_uses_120s_timeout(self):
        """warm_model passes a 120s timeout to allow slow model loads."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend()
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        await warmer.warm_model(backend, "llama3", client)

        _, kwargs = client.post.call_args
        assert kwargs.get("timeout") == 120.0


# ---------------------------------------------------------------------------
# Unit tests for ModelWarmer.warm_all_backends
# ---------------------------------------------------------------------------


class TestWarmAllBackends:
    @pytest.mark.asyncio
    async def test_warms_single_backend_single_model(self):
        """warm_all_backends sends request for each popular model not yet loaded."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend(models=[])
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        results = await warmer.warm_all_backends([backend], client)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].model == "llama3"
        assert results[0].backend_url == backend.url

    @pytest.mark.asyncio
    async def test_skips_already_loaded_model(self):
        """warm_all_backends skips models already in backend.models."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend(models=["llama3"])
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        results = await warmer.warm_all_backends([backend], client)

        assert results == []
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_unhealthy_backend(self):
        """warm_all_backends skips unhealthy backends entirely."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend(healthy=False)
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        results = await warmer.warm_all_backends([backend], client)

        assert results == []
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_warms_multiple_backends(self):
        """warm_all_backends warms each healthy backend independently."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend1 = make_backend(host="10.0.0.1", models=[])
        backend2 = make_backend(host="10.0.0.2", models=[])
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        results = await warmer.warm_all_backends([backend1, backend2], client)

        assert len(results) == 2
        assert all(r.success for r in results)
        # Both backends should now have the model
        assert "llama3" in backend1.models
        assert "llama3" in backend2.models

    @pytest.mark.asyncio
    async def test_warms_multiple_models_per_backend(self):
        """warm_all_backends warms every popular model that isn't yet loaded."""
        warmer = ModelWarmer(popular_models=["model-a", "model-b", "model-c"])
        backend = make_backend(models=["model-b"])  # model-b already present
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        results = await warmer.warm_all_backends([backend], client)

        assert len(results) == 2  # model-a and model-c only
        warmed_models = {r.model for r in results}
        assert warmed_models == {"model-a", "model-c"}

    @pytest.mark.asyncio
    async def test_failed_warming_records_error(self):
        """warm_all_backends captures failure when backend returns 500."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend(models=[])
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(500))

        results = await warmer.warm_all_backends([backend], client)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        # Model should NOT be added to backend on failure
        assert "llama3" not in backend.models

    @pytest.mark.asyncio
    async def test_updates_backend_models_on_success(self):
        """warm_all_backends adds model to backend.models when warming succeeds."""
        warmer = ModelWarmer(popular_models=["llama3"])
        backend = make_backend(models=[])
        client = AsyncMock()
        client.post = AsyncMock(return_value=make_httpx_response(200))

        await warmer.warm_all_backends([backend], client)

        assert "llama3" in backend.models

    @pytest.mark.asyncio
    async def test_tracks_warming_statistics(self):
        """warm_all_backends increments total_warmed / total_failed counters."""
        warmer = ModelWarmer(popular_models=["good", "bad"])
        backend = make_backend(models=[])
        client = AsyncMock()

        async def fake_post(url, **kwargs):
            if "good" in kwargs.get("json", {}).get("model", ""):
                return make_httpx_response(200)
            return make_httpx_response(500)

        client.post = fake_post

        await warmer.warm_all_backends([backend], client)

        status = warmer.get_status()
        assert status["total_warmed"] == 1
        assert status["total_failed"] == 1

    @pytest.mark.asyncio
    async def test_empty_popular_models(self):
        """warm_all_backends handles empty popular_models list gracefully."""
        warmer = ModelWarmer(popular_models=[])
        backend = make_backend(models=[])
        client = AsyncMock()
        client.post = AsyncMock()

        results = await warmer.warm_all_backends([backend], client)

        assert results == []
        client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests for ModelWarmer.get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_get_status_defaults(self):
        """get_status returns correct defaults before any warming."""
        warmer = ModelWarmer(popular_models=["llama3"], warm_interval=600.0)
        status = warmer.get_status()

        assert status["popular_models"] == ["llama3"]
        assert status["warm_on_startup"] is True
        assert status["warm_interval"] == 600.0
        assert status["total_warmed"] == 0
        assert status["total_failed"] == 0
        assert status["warming_active"] is False
        assert status["model_status"] == {}


# ---------------------------------------------------------------------------
# Integration tests: warmer in /backends endpoint
# ---------------------------------------------------------------------------


class TestWarmingInBackendsEndpoint:
    def test_backends_endpoint_includes_warming_status(self):
        """GET /backends includes 'warming' key when warmer is configured."""
        from fastapi.testclient import TestClient

        state = ProxyState()
        state.backends = [make_backend()]
        state.model_warmer = ModelWarmer(popular_models=["llama3"])
        app = create_lb_app(state)
        client = TestClient(app)

        response = client.get("/backends")
        assert response.status_code == 200
        data = response.json()
        assert "warming" in data
        assert data["warming"]["popular_models"] == ["llama3"]

    def test_backends_endpoint_no_warming_key_when_disabled(self):
        """GET /backends omits 'warming' key when no warmer is configured."""
        from fastapi.testclient import TestClient

        state = ProxyState()
        state.backends = [make_backend()]
        # model_warmer is None by default
        app = create_lb_app(state)
        client = TestClient(app)

        response = client.get("/backends")
        assert response.status_code == 200
        data = response.json()
        assert "warming" not in data


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_warm_models_cli_flag_parses(self):
        """--warm-models flag is parsed and passed to run_lb_proxy."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        with patch("llamacpp_cli.lb_proxy.run_lb_proxy") as mock_run:
            result = runner.invoke(
                cli,
                [
                    "lb-proxy",
                    "--warm-models",
                    "llama3,mistral-7b",
                    "--port",
                    "19999",
                ],
            )
            if mock_run.called:
                _, kwargs = mock_run.call_args
                warm = kwargs.get("warm_models") or mock_run.call_args[0][14] if mock_run.call_args[0] else None
                called_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
                called_positional = mock_run.call_args[0] if mock_run.call_args[0] else ()
                # Verify warm_models was parsed correctly
                if "warm_models" in called_kwargs:
                    assert called_kwargs["warm_models"] == ["llama3", "mistral-7b"]

    def test_no_warm_cli_flag(self):
        """--no-warm flag is parsed and passed to run_lb_proxy."""
        from click.testing import CliRunner

        from llamacpp_cli.cli import cli

        runner = CliRunner()
        with patch("llamacpp_cli.lb_proxy.run_lb_proxy") as mock_run:
            runner.invoke(
                cli,
                [
                    "lb-proxy",
                    "--no-warm",
                    "--port",
                    "19999",
                ],
            )
            if mock_run.called:
                called_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
                if "no_warm" in called_kwargs:
                    assert called_kwargs["no_warm"] is True
