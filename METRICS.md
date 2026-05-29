# LB Proxy Prometheus Metrics

Comprehensive monitoring metrics for the load balancer proxy with support for Grafana dashboards.

## Quick Start

The metrics are automatically enabled when the lb-proxy starts. Access them at:

```bash
curl http://localhost:8080/metrics
```

## Available Metrics

### Request Metrics

#### `lb_proxy_requests_total` (Counter)
Total number of requests processed by the load balancer.

**Labels:**
- `method`: HTTP method (GET, POST, etc.)
- `endpoint`: Request endpoint path (e.g., `/v1/chat/completions`)
- `status`: HTTP status code (200, 400, 500, etc.)
- `backend`: Backend URL that handled the request

**Example queries:**
```promql
# Request rate per second
rate(lb_proxy_requests_total[5m])

# Error rate (5xx errors)
rate(lb_proxy_requests_total{status=~"5.."}[5m])

# Success rate
rate(lb_proxy_requests_total{status="200"}[5m])

# Requests by endpoint
sum by (endpoint) (rate(lb_proxy_requests_total[5m]))
```

#### `lb_proxy_request_duration_seconds` (Histogram)
Request duration in seconds from client request to response completion.

**Labels:**
- `endpoint`: Request endpoint path
- `backend`: Backend URL

**Buckets:** 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0 seconds

**Example queries:**
```promql
# P50 latency
histogram_quantile(0.50, rate(lb_proxy_request_duration_seconds_bucket[5m]))

# P95 latency
histogram_quantile(0.95, rate(lb_proxy_request_duration_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(lb_proxy_request_duration_seconds_bucket[5m]))

# Average latency
rate(lb_proxy_request_duration_seconds_sum[5m]) / rate(lb_proxy_request_duration_seconds_count[5m])

# P95 latency by backend
histogram_quantile(0.95, sum by (backend, le) (rate(lb_proxy_request_duration_seconds_bucket[5m])))
```

#### `lb_proxy_active_requests` (Gauge)
Number of requests currently being processed.

**Labels:**
- `backend`: Backend URL

**Example queries:**
```promql
# Active requests per backend
lb_proxy_active_requests

# Total active requests across all backends
sum(lb_proxy_active_requests)

# Max active requests
max(lb_proxy_active_requests)
```

### Token Metrics

#### `lb_proxy_tokens_total` (Counter)
Total tokens processed (prompt + completion).

**Labels:**
- `type`: Token type (`prompt` or `completion`)
- `model`: Model name
- `backend`: Backend URL

**Example queries:**
```promql
# Token rate per second
rate(lb_proxy_tokens_total[5m])

# Prompt tokens per second
rate(lb_proxy_tokens_total{type="prompt"}[5m])

# Completion tokens per second
rate(lb_proxy_tokens_total{type="completion"}[5m])

# Tokens by model
sum by (model) (rate(lb_proxy_tokens_total[5m]))

# Total tokens in last hour
increase(lb_proxy_tokens_total[1h])
```

### Backend Health Metrics

#### `lb_proxy_backend_healthy` (Gauge)
Backend health status.

**Values:**
- `1`: Healthy
- `0`: Unhealthy

**Labels:**
- `backend`: Backend URL

**Example queries:**
```promql
# Healthy backends
lb_proxy_backend_healthy == 1

# Unhealthy backends
lb_proxy_backend_healthy == 0

# Number of healthy backends
sum(lb_proxy_backend_healthy)
```

#### `lb_proxy_backend_circuit_state` (Gauge)
Circuit breaker state.

**Values:**
- `0`: CLOSED (normal operation)
- `1`: OPEN (circuit tripped, rejecting requests)
- `2`: HALF_OPEN (testing recovery)

**Labels:**
- `backend`: Backend URL

**Example queries:**
```promql
# Backends with open circuit breaker
lb_proxy_backend_circuit_state == 1

# Circuit breaker state history
lb_proxy_backend_circuit_state
```

#### `lb_proxy_backend_consecutive_failures` (Gauge)
Number of consecutive health check failures.

**Labels:**
- `backend`: Backend URL

**Example queries:**
```promql
# Backends with failures
lb_proxy_backend_consecutive_failures > 0

# Max consecutive failures
max(lb_proxy_backend_consecutive_failures)
```

### Queue Metrics

#### `lb_proxy_queue_depth` (Gauge)
Number of requests currently waiting in queue.

**Example queries:**
```promql
# Current queue depth
lb_proxy_queue_depth

# Max queue depth in last 5 minutes
max_over_time(lb_proxy_queue_depth[5m])

# Average queue depth
avg_over_time(lb_proxy_queue_depth[5m])
```

#### `lb_proxy_queue_wait_seconds` (Histogram)
Time requests spend waiting in queue before being processed.

**Buckets:** 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0 seconds

**Example queries:**
```promql
# P95 queue wait time
histogram_quantile(0.95, rate(lb_proxy_queue_wait_seconds_bucket[5m]))

# P99 queue wait time
histogram_quantile(0.99, rate(lb_proxy_queue_wait_seconds_bucket[5m]))

# Average queue wait time
rate(lb_proxy_queue_wait_seconds_sum[5m]) / rate(lb_proxy_queue_wait_seconds_count[5m])
```

### Rate Limiter Metrics

#### `lb_proxy_rate_limit_hits_total` (Counter)
Total number of requests rejected due to rate limiting.

**Labels:**
- `key_type`: Type of rate limiting key (`api_key` or `ip`)

**Example queries:**
```promql
# Rate limit rejections per second
rate(lb_proxy_rate_limit_hits_total[5m])

# Rate limits by type
sum by (key_type) (rate(lb_proxy_rate_limit_hits_total[5m]))

# Total rate limit hits in last hour
increase(lb_proxy_rate_limit_hits_total[1h])
```

### Cache Metrics

#### `lb_proxy_cache_requests_total` (Counter)
Total cache lookup requests.

**Labels:**
- `result`: Cache lookup result (`hit` or `miss`)

**Example queries:**
```promql
# Cache hit rate
rate(lb_proxy_cache_requests_total{result="hit"}[5m]) / rate(lb_proxy_cache_requests_total[5m])

# Cache miss rate
rate(lb_proxy_cache_requests_total{result="miss"}[5m]) / rate(lb_proxy_cache_requests_total[5m])
```

### Info Metrics

#### `lb_proxy_info` (Info)
Build information for the load balancer proxy.

**Labels:**
- `version`: LB proxy version
- `python_version`: Python version

## Composite Metrics

### Backend Utilization
```promql
# Calculate backend utilization as percentage
(lb_proxy_active_requests / on(backend) (lb_proxy_backend_healthy * 100)) * 100
```

### Request Success Rate
```promql
# Success rate (2xx responses)
sum(rate(lb_proxy_requests_total{status=~"2.."}[5m])) / sum(rate(lb_proxy_requests_total[5m]))
```

### Average Tokens Per Request
```promql
# Average tokens per request
rate(lb_proxy_tokens_total[5m]) / rate(lb_proxy_requests_total[5m])
```

## Grafana Dashboard

A pre-built Grafana dashboard is available in `grafana-dashboard.json`. Import it into Grafana for instant visualization of all metrics.

### Dashboard Panels

1. **Request Rate** - Real-time request throughput
2. **Error Rate** - 4xx and 5xx error rates
3. **Request Latency Percentiles** - P50, P95, P99 latency
4. **Active Requests** - Currently processing requests
5. **Token Rate** - Prompt and completion token throughput
6. **Backend Health** - Health status of all backends
7. **Circuit Breaker State** - Circuit breaker status
8. **Queue Depth** - Request queue size
9. **Queue Wait Time Percentiles** - Queue wait latency
10. **Rate Limit Hits** - Rate limiting activity
11. **Backend Utilization** - Backend capacity usage
12. **Consecutive Failures** - Health check failures

### Importing the Dashboard

```bash
# Via Grafana UI
1. Go to Dashboards → Import
2. Upload grafana-dashboard.json
3. Select your Prometheus data source
4. Click Import

# Via API
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @grafana-dashboard.json
```

## Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: lb_proxy
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(lb_proxy_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High 5xx error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(lb_proxy_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency"
          description: "P95 latency is {{ $value }}s"

      # Backend down
      - alert: BackendDown
        expr: lb_proxy_backend_healthy == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Backend {{ $labels.backend }} is down"

      # Circuit breaker open
      - alert: CircuitBreakerOpen
        expr: lb_proxy_backend_circuit_state == 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker open for {{ $labels.backend }}"

      # High queue depth
      - alert: HighQueueDepth
        expr: lb_proxy_queue_depth > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Queue depth is high: {{ $value }}"

      # Rate limiting active
      - alert: RateLimitingActive
        expr: rate(lb_proxy_rate_limit_hits_total[5m]) > 1
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Rate limiting is rejecting requests"
```

## Performance Impact

The metrics system is designed for minimal performance impact:

- **Memory overhead**: ~10KB per unique label combination
- **CPU overhead**: <1% for typical workloads
- **Latency impact**: <0.1ms per request

Metrics are collected in-memory and exported only when `/metrics` is scraped.

## Integration with lb_proxy.py

The metrics are integrated at key points in the request lifecycle:

1. **Request received**: Record request start time
2. **Backend selected**: Update active request count
3. **Request forwarded**: Track backend selection
4. **Response received**: Record latency, status, tokens
5. **Request complete**: Update counters, decrement active requests

See the integration code in `lb_proxy.py` for implementation details.

## Troubleshooting

### Metrics not updating

Check that prometheus_client is installed:
```bash
pip install prometheus-client>=0.19
```

### High cardinality warning

If you see warnings about high cardinality:
- Check for unbounded label values (e.g., user IDs in labels)
- Limit the number of unique backends
- Consider using metric relabeling in Prometheus

### Missing metrics

Verify metrics are registered:
```python
from llamacpp_cli.prometheus_metrics import setup_metrics
setup_metrics(version="0.1.5")
```

## See Also

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
