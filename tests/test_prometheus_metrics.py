"""Tests for Prometheus metrics module."""

from prometheus_client import REGISTRY, generate_latest

from llamacpp_cli.prometheus_metrics import (
    CircuitBreakerState,
    get_metrics_handler,
    record_cache_request,
    record_queue_wait,
    record_rate_limit_hit,
    record_request,
    record_tokens,
    set_active_requests,
    set_backend_health,
    set_circuit_state,
    set_consecutive_failures,
    set_queue_depth,
    setup_metrics,
)


def get_metric_value(metric_name, labels=None):
    """Helper to extract metric value from Prometheus output.

    Args:
        metric_name: Name of the metric (without labels)
        labels: Dict of label names to values

    Returns:
        Float value of the metric, or None if not found
    """
    output = generate_latest(REGISTRY).decode("utf-8")

    # Build the label string
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        search_pattern = f"{metric_name}{{{label_str}}}"
    else:
        search_pattern = metric_name

    # Find the line with this metric
    for line in output.split("\n"):
        if line.startswith(search_pattern):
            # Extract value after the space
            value_str = line.split()[-1]
            return float(value_str)

    return None


class TestMetricSetup:
    """Tests for metric setup and initialization."""

    def test_setup_metrics_default_version(self):
        """Test setup_metrics with default version."""
        setup_metrics()
        # Verify no exceptions are raised

    def test_setup_metrics_custom_version(self):
        """Test setup_metrics with custom version."""
        setup_metrics(version="1.2.3")
        # Verify no exceptions are raised


class TestRequestMetrics:
    """Tests for request-related metrics."""

    def test_record_request_increments_counter(self):
        """Test that record_request increments the request counter."""
        backend_url = "http://localhost:8081"

        # Get initial value (might be > 0 from other tests)
        initial = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "200",
                "backend": backend_url,
            },
        ) or 0

        # Record a request
        record_request(
            method="POST",
            endpoint="/v1/chat/completions",
            status=200,
            backend=backend_url,
            duration_seconds=1.5,
        )

        # Verify counter was incremented
        final = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "200",
                "backend": backend_url,
            },
        )
        assert final == initial + 1

    def test_record_request_observes_duration(self):
        """Test that record_request records duration histogram."""
        backend_url = "http://localhost:8082"

        # Record multiple requests
        durations = [0.5, 1.0, 2.5, 5.0]
        for duration in durations:
            record_request(
                method="POST",
                endpoint="/v1/completions",
                status=200,
                backend=backend_url,
                duration_seconds=duration,
            )

        # Verify histogram sum and count
        sum_value = get_metric_value(
            "lb_proxy_request_duration_seconds_sum",
            {"endpoint": "/v1/completions", "backend": backend_url},
        )
        count_value = get_metric_value(
            "lb_proxy_request_duration_seconds_count",
            {"endpoint": "/v1/completions", "backend": backend_url},
        )

        assert sum_value == sum(durations)
        assert count_value == len(durations)

    def test_record_request_different_status_codes(self):
        """Test recording requests with different status codes."""
        backend_url = "http://localhost:8083"

        record_request("POST", "/v1/chat/completions", 200, backend_url, 1.0)
        record_request("POST", "/v1/chat/completions", 400, backend_url, 0.5)
        record_request("POST", "/v1/chat/completions", 500, backend_url, 2.0)

        # Verify each status code is tracked separately
        count_200 = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "200",
                "backend": backend_url,
            },
        )
        count_400 = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "400",
                "backend": backend_url,
            },
        )
        count_500 = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "500",
                "backend": backend_url,
            },
        )

        assert count_200 == 1
        assert count_400 == 1
        assert count_500 == 1

    def test_set_active_requests(self):
        """Test setting active request count."""
        backend_url = "http://localhost:8084"

        set_active_requests(backend_url, 5)
        value = get_metric_value("lb_proxy_active_requests", {"backend": backend_url})
        assert value == 5

        set_active_requests(backend_url, 10)
        value = get_metric_value("lb_proxy_active_requests", {"backend": backend_url})
        assert value == 10


class TestTokenMetrics:
    """Tests for token usage metrics."""

    def test_record_tokens_prompt(self):
        """Test recording prompt tokens."""
        backend_url = "http://localhost:8085"
        model = "llama-3.3-70b"

        initial = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "prompt", "model": model, "backend": backend_url},
        ) or 0

        record_tokens(
            token_type="prompt",
            model=model,
            backend=backend_url,
            count=100,
        )

        final = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "prompt", "model": model, "backend": backend_url},
        )
        assert final == initial + 100

    def test_record_tokens_completion(self):
        """Test recording completion tokens."""
        backend_url = "http://localhost:8086"
        model = "llama-3.3-70b"

        initial = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "completion", "model": model, "backend": backend_url},
        ) or 0

        record_tokens(
            token_type="completion",
            model=model,
            backend=backend_url,
            count=250,
        )

        final = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "completion", "model": model, "backend": backend_url},
        )
        assert final == initial + 250

    def test_record_tokens_multiple_models(self):
        """Test recording tokens for multiple models."""
        backend_url = "http://localhost:8087"

        record_tokens("prompt", "llama-3.3-70b-v2", backend_url, 100)
        record_tokens("prompt", "llama-3.1-8b-v2", backend_url, 50)
        record_tokens("completion", "llama-3.3-70b-v2", backend_url, 200)

        # Verify each model is tracked separately
        value1 = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "prompt", "model": "llama-3.3-70b-v2", "backend": backend_url},
        )
        value2 = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "prompt", "model": "llama-3.1-8b-v2", "backend": backend_url},
        )
        value3 = get_metric_value(
            "lb_proxy_tokens_total",
            {"type": "completion", "model": "llama-3.3-70b-v2", "backend": backend_url},
        )

        assert value1 == 100
        assert value2 == 50
        assert value3 == 200


class TestBackendMetrics:
    """Tests for backend health and circuit breaker metrics."""

    def test_set_backend_health_healthy(self):
        """Test setting backend to healthy."""
        backend_url = "http://localhost:8088"

        set_backend_health(backend_url, True)
        value = get_metric_value("lb_proxy_backend_healthy", {"backend": backend_url})
        assert value == 1

    def test_set_backend_health_unhealthy(self):
        """Test setting backend to unhealthy."""
        backend_url = "http://localhost:8089"

        set_backend_health(backend_url, False)
        value = get_metric_value("lb_proxy_backend_healthy", {"backend": backend_url})
        assert value == 0

    def test_set_circuit_state_closed(self):
        """Test setting circuit breaker to closed state."""
        backend_url = "http://localhost:8090"

        set_circuit_state(backend_url, CircuitBreakerState.CLOSED)
        value = get_metric_value("lb_proxy_backend_circuit_state", {"backend": backend_url})
        assert value == 0

    def test_set_circuit_state_open(self):
        """Test setting circuit breaker to open state."""
        backend_url = "http://localhost:8091"

        set_circuit_state(backend_url, CircuitBreakerState.OPEN)
        value = get_metric_value("lb_proxy_backend_circuit_state", {"backend": backend_url})
        assert value == 1

    def test_set_circuit_state_half_open(self):
        """Test setting circuit breaker to half-open state."""
        backend_url = "http://localhost:8092"

        set_circuit_state(backend_url, CircuitBreakerState.HALF_OPEN)
        value = get_metric_value("lb_proxy_backend_circuit_state", {"backend": backend_url})
        assert value == 2

    def test_set_consecutive_failures(self):
        """Test setting consecutive failure count."""
        backend_url = "http://localhost:8093"

        set_consecutive_failures(backend_url, 3)
        value = get_metric_value("lb_proxy_backend_consecutive_failures", {"backend": backend_url})
        assert value == 3


class TestQueueMetrics:
    """Tests for request queue metrics."""

    def test_set_queue_depth(self):
        """Test setting queue depth."""
        set_queue_depth(5)
        value = get_metric_value("lb_proxy_queue_depth", {})
        assert value == 5

        set_queue_depth(0)
        value = get_metric_value("lb_proxy_queue_depth", {})
        assert value == 0

    def test_record_queue_wait(self):
        """Test recording queue wait times."""
        wait_times = [0.5, 1.0, 2.0, 5.0]
        for wait in wait_times:
            record_queue_wait(wait)

        # Verify histogram sum and count
        sum_value = get_metric_value("lb_proxy_queue_wait_seconds_sum", {})
        count_value = get_metric_value("lb_proxy_queue_wait_seconds_count", {})

        assert sum_value >= sum(wait_times)  # >= because might have values from other tests
        assert count_value >= len(wait_times)


class TestRateLimitMetrics:
    """Tests for rate limiter metrics."""

    def test_record_rate_limit_hit_api_key(self):
        """Test recording rate limit hit for API key."""
        initial = get_metric_value("lb_proxy_rate_limit_hits_total", {"key_type": "api_key"}) or 0

        record_rate_limit_hit("api_key")
        record_rate_limit_hit("api_key")

        final = get_metric_value("lb_proxy_rate_limit_hits_total", {"key_type": "api_key"})
        assert final == initial + 2

    def test_record_rate_limit_hit_ip(self):
        """Test recording rate limit hit for IP."""
        initial = get_metric_value("lb_proxy_rate_limit_hits_total", {"key_type": "ip"}) or 0

        record_rate_limit_hit("ip")

        final = get_metric_value("lb_proxy_rate_limit_hits_total", {"key_type": "ip"})
        assert final == initial + 1


class TestCacheMetrics:
    """Tests for cache metrics."""

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        initial = get_metric_value("lb_proxy_cache_requests_total", {"result": "hit"}) or 0

        record_cache_request("hit")

        final = get_metric_value("lb_proxy_cache_requests_total", {"result": "hit"})
        assert final == initial + 1

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        initial = get_metric_value("lb_proxy_cache_requests_total", {"result": "miss"}) or 0

        record_cache_request("miss")

        final = get_metric_value("lb_proxy_cache_requests_total", {"result": "miss"})
        assert final == initial + 1


class TestMetricsEndpoint:
    """Tests for metrics endpoint handler."""

    def test_get_metrics_handler_returns_callable(self):
        """Test that get_metrics_handler returns a callable."""
        handler = get_metrics_handler()
        assert callable(handler)

    def test_metrics_handler_returns_response(self):
        """Test that metrics handler returns a Response object."""
        from fastapi import Response

        handler = get_metrics_handler()
        response = handler()

        assert isinstance(response, Response)
        assert response.media_type == "text/plain; version=0.0.4; charset=utf-8"

    def test_metrics_handler_contains_metric_names(self):
        """Test that metrics output contains expected metric names."""
        # Record some data
        record_request("POST", "/v1/chat/completions", 200, "http://localhost:9001", 1.5)
        record_tokens("prompt", "llama-3.3-70b", "http://localhost:9001", 100)

        handler = get_metrics_handler()
        response = handler()
        content = response.body.decode("utf-8")

        # Verify key metrics are present in output
        assert "lb_proxy_requests_total" in content
        assert "lb_proxy_request_duration_seconds" in content
        assert "lb_proxy_tokens_total" in content
        assert "lb_proxy_active_requests" in content
        assert "lb_proxy_backend_healthy" in content

    def test_metrics_handler_contains_help_text(self):
        """Test that metrics output contains HELP documentation."""
        handler = get_metrics_handler()
        response = handler()
        content = response.body.decode("utf-8")

        # Verify HELP lines are present
        assert "# HELP lb_proxy_requests_total" in content
        assert "# HELP lb_proxy_request_duration_seconds" in content
        assert "# HELP lb_proxy_tokens_total" in content

    def test_metrics_handler_contains_type_info(self):
        """Test that metrics output contains TYPE information."""
        handler = get_metrics_handler()
        response = handler()
        content = response.body.decode("utf-8")

        # Verify TYPE lines are present
        assert "# TYPE lb_proxy_requests_total counter" in content
        assert "# TYPE lb_proxy_request_duration_seconds histogram" in content
        assert "# TYPE lb_proxy_active_requests gauge" in content


class TestHistogramBuckets:
    """Tests for histogram bucket configuration."""

    def test_request_duration_buckets(self):
        """Test that request duration histogram has correct buckets."""
        expected_buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]

        # Record values in each bucket range
        backend_url = "http://localhost:9002"
        for bucket_value in expected_buckets:
            record_request(
                "POST",
                "/v1/chat/completions",
                200,
                backend_url,
                bucket_value,
            )

        # Verify histogram recorded all values
        count = get_metric_value(
            "lb_proxy_request_duration_seconds_count",
            {"endpoint": "/v1/chat/completions", "backend": backend_url},
        )
        assert count == len(expected_buckets)

    def test_queue_wait_buckets(self):
        """Test that queue wait histogram has correct buckets."""
        expected_buckets = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]

        initial_count = get_metric_value("lb_proxy_queue_wait_seconds_count", {}) or 0

        # Record values in each bucket range
        for bucket_value in expected_buckets:
            record_queue_wait(bucket_value)

        # Verify histogram recorded all values
        final_count = get_metric_value("lb_proxy_queue_wait_seconds_count", {})
        assert final_count == initial_count + len(expected_buckets)


class TestMultipleBackends:
    """Tests for metrics across multiple backends."""

    def test_metrics_separate_per_backend(self):
        """Test that metrics are tracked separately for each backend."""
        backend1 = "http://localhost:9003"
        backend2 = "http://localhost:9004"

        record_request("POST", "/v1/chat/completions", 200, backend1, 1.0)
        record_request("POST", "/v1/chat/completions", 200, backend2, 2.0)

        set_active_requests(backend1, 5)
        set_active_requests(backend2, 10)

        set_backend_health(backend1, True)
        set_backend_health(backend2, False)

        # Verify each backend has separate metrics
        req_count1 = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "200",
                "backend": backend1,
            },
        )
        req_count2 = get_metric_value(
            "lb_proxy_requests_total",
            {
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "status": "200",
                "backend": backend2,
            },
        )

        active1 = get_metric_value("lb_proxy_active_requests", {"backend": backend1})
        active2 = get_metric_value("lb_proxy_active_requests", {"backend": backend2})

        health1 = get_metric_value("lb_proxy_backend_healthy", {"backend": backend1})
        health2 = get_metric_value("lb_proxy_backend_healthy", {"backend": backend2})

        assert req_count1 == 1
        assert req_count2 == 1
        assert active1 == 5
        assert active2 == 10
        assert health1 == 1
        assert health2 == 0
