# TODO

## Priority 0: Production Critical (In Progress)

### LB-Proxy Production Hardening
- [x] **Rate Limiting & Quotas** - Prevent abuse, ensure fair usage
  - [x] Per-user rate limits (requests/minute, tokens/hour) — `RateLimiter` class with sliding window in `lb_proxy.py`
  - [x] Token budgets/quotas — TPH (tokens/hour) sliding window
  - [x] `check_rate_limit()` wired into `chat_completions` and `other_post_endpoints` (429 on exceeded)
  - [ ] Per-model rate limits
  - [ ] Burst allowance
  - Note: in-memory sliding window (no Redis required for single-process)
  
- [x] **Request Queuing & Backpressure** - Handle traffic spikes
  - [x] `RequestQueue` class — FIFO queue with max_size cap
  - [x] `_queue_worker_loop` async background function
  - [x] Queue depth metrics, p50/p95/p99 wait time percentiles
  - [x] Graceful rejection (503) when queue full with estimated wait time
  - [x] Timeout handling (504) for queued requests
  - [x] Wired into `chat_completions` / `other_post_endpoints` handlers
  - [x] Queue stats exposed in `/stats` endpoint
  - [ ] Priority queue (premium users first)
  
- [x] **Circuit Breaker Pattern** - Prevent cascading failures
  - [x] `CircuitState` enum (CLOSED, OPEN, HALF_OPEN)
  - [x] `CircuitBreaker` dataclass — failure threshold, timeout, half-open recovery, success threshold
  - [x] `circuit_breaker` field added to `Backend`
  - [x] Wired into `_forward_request` (blocks OPEN circuit, records success/failure)
  - [x] Wired into `_health_check_loop` (records success/failure from health checks)
  - [ ] Metrics and alerts
  
- [x] **Structured Logging & Tracing** - Debug issues, audit trail
  - [x] Request ID tracking — `generate_request_id()`, `X-Request-ID` header
  - [x] Structured JSON logs — `configure_logging()` with structlog via stdlib
  - [x] Backend selection/health/request logging helpers
  - [x] `add_request_tracing` middleware wired into `create_lb_app()` 
  - [ ] OpenTelemetry distributed tracing
  - [ ] Log sampling for high traffic
  
- [x] **Request/Response Size Limits** - Prevent memory exhaustion
  - [x] `_check_request_size()` — 413 on Content-Length exceeded, wired into handlers
  - [x] `_enforce_max_tokens()` — caps `max_tokens` in request body
  - [x] `max_request_size` (10MB default) / `max_response_tokens` (32k default) in `ProxyState`

## Priority 1: Performance & Reliability

### Performance Optimization
- [ ] **Response Caching** - 15-30% cache hit rate possible
  - Cache deterministic responses (temperature=0)
  - TTL-based invalidation
  - Cache hit/miss metrics
  
- [ ] **Weighted Load Balancing** - Account for different backend capacities
  - Backend weight configuration
  - Combined weight + active requests scoring
  
- [ ] **Sticky Sessions** - KV cache reuse for conversations
  - Conversation -> backend mapping
  - 2-3x faster for multi-turn chats

### Observability
- [ ] **Enhanced Prometheus Metrics**
  - Latency percentiles (p50, p95, p99)
  - Token usage by model/user
  - Queue depth, error rates
  - Backend utilization
  
- [ ] **Real-time Dashboard** - Live metrics visualization
  - React + WebSocket frontend
  - Live request graph
  - Latency heatmap
  - Cost tracking

## Priority 2: Security & Compliance

- [ ] **JWT Authentication** - Enterprise SSO integration
  - OAuth2/OIDC support
  - RBAC (role-based access control)
  - Multi-tenancy support
  - API key scoping
  
- [ ] **PII Detection & Filtering** - Prevent data leakage
  - Presidio integration
  - Auto-redaction
  - Compliance (GDPR, HIPAA)
  
- [ ] **IP Whitelisting & Geo-blocking**
  - CIDR range support
  - Country-based blocking

## Priority 3: Developer Experience

- [ ] **OpenAPI Documentation** - Auto-generated docs (FastAPI built-in)
  - Swagger UI at /docs
  - ReDoc at /redoc
  
- [ ] **Management CLI** - Ops convenience
  - `llamacpp lb-proxy backends add/remove/drain`
  - `llamacpp lb-proxy stats --real-time`
  - `llamacpp lb-proxy benchmark`
  
- [ ] **Request Replay** - Debug production issues
  - Save failed requests
  - Replay capability
  - Load testing from real traffic

## Priority 4: Advanced Features

### LlamaCPP-CLI Features
- [ ] **Multi-Model Serving** - Run multiple models on one instance
  - Auto-routing by model name
  - Port-based model serving
  
- [ ] **GPU Auto-Optimization** - Detect GPU and optimize
  - Auto-detect GPU type
  - Optimal layer offloading
  - Flash attention support
  - Tensor parallelism for multi-GPU
  
- [ ] **Model Registry/Catalog** - Centralized model management
  - Browse and search models
  - Private registry support
  - Metadata tracking
  
- [ ] **Automatic Quantization** - Optimize model size vs quality
  - Quantize existing models
  - Benchmark different quantizations

### Advanced Load Balancing
- [ ] **Request Batching** - Better GPU utilization
  - Batch small requests
  - Dynamic batch sizing
  - Throughput improvements
  
- [ ] **Model Warming & Preloading** - Reduce cold start latency
  - Preload popular models
  - Configurable warm-up strategy

## Completed ✅

- [x] Basic load balancing (least connections)
- [x] Model-aware routing
- [x] Health checking with thresholds
- [x] Auto-discovery of backends
- [x] Token usage tracking
- [x] Full OpenAI API compatibility
- [x] Root landing page
- [x] Tokenization endpoints
- [x] Legacy OpenAI endpoints
- [x] llama.cpp monitoring endpoints (/slots, /props, /metrics)
- [x] Comprehensive test coverage

## Notes

- Full roadmap with implementation details: `/tmp/lb_proxy_feature_roadmap.md`
- P0 features are critical for production deployment
- P1 features provide significant performance improvements
- P2+ features are for enterprise/advanced use cases
