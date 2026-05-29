# TODO

## Priority 0: Production Critical ✅ DONE

### LB-Proxy Production Hardening
- [x] **Rate Limiting & Quotas** — `RateLimiter` sliding window in `lb_proxy.py` (429 on exceeded)
- [x] **Request Queuing & Backpressure** — `RequestQueue` FIFO, p50/p95/p99 wait times, 503/504
- [x] **Circuit Breaker Pattern** — CLOSED/OPEN/HALF_OPEN per backend
- [x] **Structured Logging & Tracing** — X-Request-ID, structlog JSON, `lb_proxy_logging.py`
- [x] **Request/Response Size Limits** — 413 on >10MB, max_tokens cap

## Priority 1: Performance & Reliability ✅ DONE

### Performance Optimization
- [x] **Response Caching** — `response_cache.py`, LRU+TTL, temperature=0 only, 15-30% hit rate
- [x] **Weighted Load Balancing** — `weight` field on Backend, score = active_requests / weight
- [x] **Sticky Sessions** — `conversation_affinity.py`, multi-turn hash → backend mapping, 2-3x faster

### Observability
- [x] **Enhanced Prometheus Metrics** — `prometheus_metrics.py`, 11 metrics, histograms, Grafana dashboard JSON
- [x] **Real-time Dashboard** — SSE at `/stats/stream`, live HTML dashboard, auto-reconnect

## Priority 2: Security & Compliance ✅ DONE

- [x] **IP Whitelisting** — `ip_filter.py`, CIDR ranges, X-Forwarded-For, IPv4/IPv6, `--allowed-ips`
- [ ] **JWT Authentication** — Enterprise SSO/OIDC (not implemented, low priority for internal use)
- [ ] **PII Detection & Filtering** — Presidio (not implemented, requires extra dep)
- [ ] **Geo-blocking** — requires GeoIP database (not implemented)

## Priority 3: Developer Experience ✅ DONE

- [x] **OpenAPI Documentation** — `/docs` (Swagger), `/redoc`, Pydantic models, tags
- [x] **Management CLI** — `llamacpp lb backends/add/remove/stats/health`
- [x] **Request Logging & Replay** — `request_logger.py`, JSONL file, `llamacpp lb-proxy-replay`

## Priority 4: Advanced Features ✅ DONE

### LlamaCPP-CLI Features
- [x] **GPU Auto-Optimization** — `gpu_detect.py`, nvidia-smi/rocm-smi, `--gpu/--no-gpu/--gpu-layers`
- [x] **Multi-Model Serving** — `multi_model_server.py` + `multi_model_proxy.py`, `llamacpp serve-multi`
- [x] **Model Warming & Preloading** — `model_warmer.py`, startup warm, background loop, `--warm-models`
- [ ] **Model Registry/Catalog** — search/browse (uses HF directly via `llamacpp pull`)
- [ ] **Automatic Quantization** — requires llama.cpp quantize binary

### Advanced Load Balancing
- [x] **NUMA-Aware Slot Architecture** — `slot_manager.py`, one process per NUMA node, 3-tier selection
- [ ] **Request Batching** — GPU-side batching (llama.cpp handles this internally via `--parallel`)

### NUMA Performance
- [x] **Explicit NUMA Binding** — numactl `--cpunodebind=N --membind=N`, `--socket-id` flag
- [x] **CPU Topology Detection** — `cpu_topology.py`, /proc/cpuinfo + /sys/devices/system/node

## Completed ✅

- [x] Full OpenAI API compatibility (chat, completions, embeddings, tokenize, models)
- [x] Legacy OpenAI engine endpoints
- [x] llama.cpp endpoints (/slots, /props, /metrics)
- [x] Auto-discovery of backends via subnet scan
- [x] Health checking with hysteresis thresholds
- [x] Token usage tracking per backend
- [x] max-context preset as default (32K ctx)
- [x] NUMA-aware parallel defaults (num_slots)
- [x] 528 tests passing

## Remaining (Low Priority)

- [ ] JWT/OAuth2 authentication — for external access (internal use: API key sufficient)
- [ ] PII detection — requires `presidio-analyzer` dep, high overhead
- [ ] Geo-blocking — requires GeoIP database
- [ ] Automatic quantization — requires `llama-quantize` binary
- [ ] Model registry/catalog — HF search works via `llamacpp search`
- [ ] Per-model rate limits — current: per-user/IP
- [ ] Priority queue — current: FIFO
- [ ] Burst allowance — current: hard limits
- [ ] OpenTelemetry tracing — current: X-Request-ID + structlog
