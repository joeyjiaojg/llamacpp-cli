#!/usr/bin/env python3
"""Example usage of Prometheus metrics for lb-proxy.

This script demonstrates how to integrate the metrics into lb_proxy.py.
"""

import time

from llamacpp_cli.prometheus_metrics import (
    CircuitBreakerState,
    get_metrics_handler,
    record_request,
    record_tokens,
    set_active_requests,
    set_backend_health,
    set_circuit_state,
    setup_metrics,
)

# Initialize metrics with version
setup_metrics(version="0.1.5")

# Simulate some traffic
backend = "http://localhost:8081"
model = "llama-3.3-70b"

# Record successful request
start = time.time()
set_active_requests(backend, 1)
# ... process request ...
time.sleep(0.5)  # Simulate 500ms request
duration = time.time() - start
record_request("POST", "/v1/chat/completions", 200, backend, duration)
record_tokens("prompt", model, backend, 100)
record_tokens("completion", model, backend, 250)
set_active_requests(backend, 0)

# Update backend health
set_backend_health(backend, True)
set_circuit_state(backend, CircuitBreakerState.CLOSED)

# Get metrics endpoint handler for FastAPI
metrics_handler = get_metrics_handler()

# In a FastAPI app:
# app.get("/metrics")(metrics_handler)

# Or call it directly to get metrics output
response = metrics_handler()
print("Metrics output:")
print(response.body.decode("utf-8")[:500])
print("...")
print("\nMetrics endpoint ready!")
