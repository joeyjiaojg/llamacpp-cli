"""Tests for lb-proxy management CLI commands."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llamacpp_cli.cli import cli


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_backends_response():
    """Mock response for /backends endpoint."""
    return {
        "backends": [
            {
                "url": "http://localhost:8001",
                "healthy": True,
                "models": ["qwen3.5", "gemma3"],
                "active_requests": 2,
                "load_status": "medium",
            },
            {
                "url": "http://localhost:8002",
                "healthy": False,
                "models": [],
                "active_requests": 0,
                "load_status": "unknown",
            },
        ]
    }


@pytest.fixture
def mock_stats_response():
    """Mock response for /stats endpoint."""
    return {
        "total": {
            "requests": 1000,
            "prompt_tokens": 50000,
            "completion_tokens": 25000,
            "total_tokens": 75000,
        },
        "cache": {"hit_rate": 0.85, "cache_hits": 850, "cache_misses": 150},
    }


@pytest.fixture
def mock_health_response():
    """Mock response for /health endpoint."""
    return {"status": "healthy", "backends": {"healthy": 2, "total": 2}}


class TestBackendsList:
    """Tests for 'lb backends' command."""

    def test_backends_list_success(self, runner, mock_backends_response):
        """Test listing backends successfully."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_backends_response

            result = runner.invoke(cli, ["lb", "backends", "--url", "http://localhost:8080"])

            assert result.exit_code == 0
            # Rich may truncate URLs in table, so check for partial matches
            assert "localhost" in result.output
            assert "✓ Healthy" in result.output
            assert "✗ Unhealthy" in result.output
            assert "qwen3.5" in result.output

    def test_backends_list_with_auth(self, runner, mock_backends_response):
        """Test listing backends with authentication."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_backends_response

            result = runner.invoke(
                cli, ["lb", "backends", "--url", "http://localhost:8080", "--auth", "test-key"]
            )

            assert result.exit_code == 0
            # Check that auth header was passed
            call_args = mock_get.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_backends_list_http_error(self, runner):
        """Test handling HTTP errors."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 500

            result = runner.invoke(cli, ["lb", "backends", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error: HTTP 500" in result.output

    def test_backends_list_connection_error(self, runner):
        """Test handling connection errors."""
        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = runner.invoke(cli, ["lb", "backends", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error:" in result.output

    def test_backends_list_empty(self, runner):
        """Test listing when no backends exist."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"backends": []}

            result = runner.invoke(cli, ["lb", "backends", "--url", "http://localhost:8080"])

            assert result.exit_code == 0


class TestBackendAdd:
    """Tests for 'lb add' command."""

    def test_backend_add_success(self, runner):
        """Test adding a backend successfully."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 201

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "add",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "http://localhost:8003",
                ],
            )

            assert result.exit_code == 0
            assert "Backend added" in result.output
            assert "http://localhost:8003" in result.output

            # Check payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "http://localhost:8003"
            assert payload["weight"] == 1.0

    def test_backend_add_with_weight(self, runner):
        """Test adding a backend with custom weight."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 200

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "add",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "http://localhost:8003",
                    "--weight",
                    "2.5",
                ],
            )

            assert result.exit_code == 0

            # Check payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["weight"] == 2.5

    def test_backend_add_with_auth(self, runner):
        """Test adding a backend with authentication."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 201

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "add",
                    "--url",
                    "http://localhost:8080",
                    "--auth",
                    "secret",
                    "--backend",
                    "http://localhost:8003",
                ],
            )

            assert result.exit_code == 0

            # Check auth header
            call_args = mock_post.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer secret"

    def test_backend_add_error(self, runner):
        """Test handling errors when adding backend."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value.status_code = 400
            mock_post.return_value.text = "Invalid backend URL"

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "add",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "invalid-url",
                ],
            )

            assert result.exit_code == 1
            assert "Error: HTTP 400" in result.output


class TestBackendRemove:
    """Tests for 'lb remove' command."""

    def test_backend_remove_success(self, runner):
        """Test removing a backend successfully."""
        with patch("httpx.delete") as mock_delete:
            mock_delete.return_value.status_code = 204

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "remove",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "http://localhost:8003",
                ],
            )

            assert result.exit_code == 0
            assert "Backend removed" in result.output
            assert "http://localhost:8003" in result.output

    def test_backend_remove_with_auth(self, runner):
        """Test removing a backend with authentication."""
        with patch("httpx.delete") as mock_delete:
            mock_delete.return_value.status_code = 200

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "remove",
                    "--url",
                    "http://localhost:8080",
                    "--auth",
                    "secret",
                    "--backend",
                    "http://localhost:8003",
                ],
            )

            assert result.exit_code == 0

            # Check auth header
            call_args = mock_delete.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer secret"

    def test_backend_remove_not_found(self, runner):
        """Test removing non-existent backend."""
        with patch("httpx.delete") as mock_delete:
            mock_delete.return_value.status_code = 404
            mock_delete.return_value.text = "Backend not found"

            result = runner.invoke(
                cli,
                [
                    "lb",
                    "remove",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "http://localhost:9999",
                ],
            )

            assert result.exit_code == 1
            assert "Error: HTTP 404" in result.output


class TestStats:
    """Tests for 'lb stats' command."""

    def test_stats_table_format(self, runner, mock_stats_response):
        """Test stats command with table format."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_stats_response

            result = runner.invoke(
                cli, ["lb", "stats", "--url", "http://localhost:8080", "--format", "table"]
            )

            assert result.exit_code == 0
            assert "1,000" in result.output  # Formatted number
            assert "50,000" in result.output
            assert "85.0%" in result.output  # Hit rate

    def test_stats_json_format(self, runner, mock_stats_response):
        """Test stats command with JSON format."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_stats_response

            result = runner.invoke(
                cli, ["lb", "stats", "--url", "http://localhost:8080", "--format", "json"]
            )

            assert result.exit_code == 0

            # Verify JSON output
            output_data = json.loads(result.output)
            assert output_data["total"]["requests"] == 1000
            assert output_data["cache"]["hit_rate"] == 0.85

    def test_stats_default_format(self, runner, mock_stats_response):
        """Test stats command with default format (table)."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_stats_response

            result = runner.invoke(cli, ["lb", "stats", "--url", "http://localhost:8080"])

            assert result.exit_code == 0
            assert "Total Statistics" in result.output

    def test_stats_without_cache(self, runner):
        """Test stats command when cache data is not available."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "total": {
                    "requests": 100,
                    "prompt_tokens": 5000,
                    "completion_tokens": 2500,
                    "total_tokens": 7500,
                }
            }

            result = runner.invoke(cli, ["lb", "stats", "--url", "http://localhost:8080"])

            assert result.exit_code == 0
            assert "Total Statistics" in result.output
            # Cache section should not be present
            assert "Cache Statistics" not in result.output

    def test_stats_with_auth(self, runner, mock_stats_response):
        """Test stats command with authentication."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_stats_response

            result = runner.invoke(
                cli, ["lb", "stats", "--url", "http://localhost:8080", "--auth", "test-key"]
            )

            assert result.exit_code == 0

            # Check auth header
            call_args = mock_get.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_stats_error(self, runner):
        """Test handling errors in stats command."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 500

            result = runner.invoke(cli, ["lb", "stats", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error: HTTP 500" in result.output


class TestHealth:
    """Tests for 'lb health' command."""

    def test_health_success(self, runner, mock_health_response):
        """Test health check with healthy status."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_health_response

            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])

            assert result.exit_code == 0
            assert "Status: healthy" in result.output
            assert "Backends: 2/2 healthy" in result.output

    def test_health_partial_unhealthy(self, runner):
        """Test health check with some unhealthy backends."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "status": "degraded",
                "backends": {"healthy": 1, "total": 2},
            }

            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])

            # Should exit with error code when not all backends are healthy
            assert result.exit_code == 1
            assert "Backends: 1/2 healthy" in result.output

    def test_health_with_auth(self, runner, mock_health_response):
        """Test health check with authentication."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_health_response

            result = runner.invoke(
                cli, ["lb", "health", "--url", "http://localhost:8080", "--auth", "test-key"]
            )

            assert result.exit_code == 0

            # Check auth header
            call_args = mock_get.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_health_http_error(self, runner):
        """Test handling HTTP errors in health check."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 500

            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error: HTTP 500" in result.output

    def test_health_connection_error(self, runner):
        """Test handling connection errors in health check."""
        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error:" in result.output

    def test_health_timeout(self, runner):
        """Test handling timeout in health check."""
        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.TimeoutException("Request timed out")

            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])

            assert result.exit_code == 1
            assert "Error:" in result.output


class TestMissingRequiredOptions:
    """Tests for missing required options."""

    def test_backends_missing_url(self, runner):
        """Test backends command without URL."""
        result = runner.invoke(cli, ["lb", "backends"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_add_missing_backend(self, runner):
        """Test add command without backend URL."""
        result = runner.invoke(cli, ["lb", "add", "--url", "http://localhost:8080"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_remove_missing_backend(self, runner):
        """Test remove command without backend URL."""
        result = runner.invoke(cli, ["lb", "remove", "--url", "http://localhost:8080"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestIntegration:
    """Integration tests for lb commands."""

    def test_full_workflow(self, runner, mock_backends_response, mock_health_response):
        """Test a full workflow: add backend, list backends, check health."""
        with patch("httpx.post") as mock_post, patch("httpx.get") as mock_get:
            # Add backend
            mock_post.return_value.status_code = 201
            result = runner.invoke(
                cli,
                [
                    "lb",
                    "add",
                    "--url",
                    "http://localhost:8080",
                    "--backend",
                    "http://localhost:8003",
                ],
            )
            assert result.exit_code == 0

            # List backends
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_backends_response
            result = runner.invoke(cli, ["lb", "backends", "--url", "http://localhost:8080"])
            assert result.exit_code == 0

            # Check health
            mock_get.return_value.json.return_value = mock_health_response
            result = runner.invoke(cli, ["lb", "health", "--url", "http://localhost:8080"])
            assert result.exit_code == 0
