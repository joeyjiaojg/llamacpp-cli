# Load Balancer Proxy Features

## Overview

The llamacpp load balancer proxy provides intelligent request routing across multiple llama-server backends with automatic discovery, health monitoring, and load-aware distribution.

## Load-Aware Routing

### How It Works

1. **Request Tracking**
   - Each backend tracks `active_requests` (current number of in-flight requests)
   - Counter incremented when request starts
   - Counter decremented when request completes (success or error)

2. **Least-Connections Algorithm**
   - Selects backend with fewest `active_requests`
   - Ensures even distribution across backends
   - Works even when all backends are busy

3. **Model-Aware Routing**
   - First filters backends by requested model
   - Then applies least-connections within matching backends
   - Falls back to any healthy backend if model not found

### What Happens When All Backends Are Busy?

**Answer**: The proxy still routes requests!

- Selects the backend with the **fewest active requests**
- Request queues on the selected backend
- No requests are rejected due to load
- Only returns 503 when **no healthy backends** exist

**Example**:
```
Backend A: 5 active requests
Backend B: 3 active requests  ← Selected (fewest)
Backend C: 7 active requests

New request → Backend B (now 4 active requests)
```

## Model Aggregation

### /v1/models Endpoint

Returns **all unique models** from **all healthy backends**.

**Request**:
```bash
curl http://localhost:8081/v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {"id": "Qwen3.5-9B-Q4_K_M.gguf", "object": "model"},
    {"id": "llama-2-7b-q4.gguf", "object": "model"},
    {"id": "mistral-7b-q4.gguf", "object": "model"}
  ]
}
```

**Behavior**:
- Scans all healthy backends
- Collects their model lists
- Deduplicates model names
- Returns OpenAI-compatible format

**Use Case**: Different backends can serve different models!
```
Backend A: Qwen3.5-9B, llama-2-7b
Backend B: Qwen3.5-9B, mistral-7b
Backend C: llama-2-13b

/v1/models returns: [Qwen3.5-9B, llama-2-7b, mistral-7b, llama-2-13b]
```

## Backend Status Monitoring

### /backends or /v1/backends Endpoint

Shows real-time status of all backends including load information.

**Request**:
```bash
curl http://localhost:8081/v1/backends
```

**Response**:
```json
{
  "backends": [
    {
      "url": "http://10.231.213.75:8000",
      "healthy": true,
      "models": ["Qwen3.5-9B-Q4_K_M.gguf"],
      "active_requests": 0,
      "load_status": "idle"
    },
    {
      "url": "http://10.231.214.204:8000",
      "healthy": true,
      "models": ["Qwen3.5-9B-Q4_K_M.gguf", "llama-2-7b.gguf"],
      "active_requests": 3,
      "load_status": "busy"
    },
    {
      "url": "http://10.231.215.92:8000",
      "healthy": false,
      "models": [],
      "active_requests": 0,
      "load_status": "idle"
    }
  ]
}
```

**Fields**:
- `url`: Backend endpoint
- `healthy`: Health check status
- `models`: Available models on this backend
- `active_requests`: Current number of in-flight requests
- `load_status`: `"idle"` (0 requests) or `"busy"` (>0 requests)

## Load Balancing Strategies

### 1. Model-Aware + Least-Connections (Default)

```python
def _select_backend(backends, model=None):
    # 1. Filter healthy backends
    healthy = [b for b in backends if b.healthy]
    
    # 2. If model specified, filter to backends with that model
    if model:
        candidates = [b for b in healthy if model in b.models]
        if candidates:
            healthy = candidates
    
    # 3. Select backend with fewest active requests
    return min(healthy, key=lambda b: b.active_requests)
```

**Example Routing**:
```
Request: {"model": "Qwen3.5-9B", ...}

Available backends:
- Backend A: models=[Qwen3.5-9B], active=2
- Backend B: models=[Qwen3.5-9B, llama-2], active=5
- Backend C: models=[llama-2], active=0

Selected: Backend A (has model, fewest requests among model matches)
```

### 2. Fallback Behavior

**Model not found on any backend**:
- Falls back to least-connections across ALL healthy backends
- Request goes to backend with fewest active requests
- Backend will return its own error for missing model

**No healthy backends**:
- Returns HTTP 503: "No healthy backends available"
- Client should retry later

## Health Monitoring

### Automatic Health Checks

- **Interval**: Every 10 seconds
- **Validates**: OpenAI-compatible `/v1/models` endpoint
- **Updates**: Backend status and model list
- **Logging**: Only logs on status change (healthy ↔ unhealthy)

### Backend Validation

Backends must:
1. Respond to `/v1/models` with 200 status
2. Return JSON with `{"data": [...]}`  format
3. Pass optional auth check (if `--auth-key` provided)

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Route chat completion to backend |
| `/v1/completions` | POST | Route completion to backend |
| `/v1/embeddings` | POST | Route embeddings to backend |
| `/v1/models` | GET | Aggregate models from all backends |
| `/backends` | GET | Backend status with load info |
| `/v1/backends` | GET | Same as `/backends` (alias) |
| `/health` | GET | Proxy health status |

## Usage Examples

### Monitor Backend Load

```bash
# Watch backend load in real-time
watch -n 1 'curl -s http://localhost:8081/v1/backends | jq'
```

### Test Load Distribution

```bash
# Send 10 concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3.5-9B-Q4_K_M.gguf", "messages": [...]}' &
done

# Check load distribution
curl http://localhost:8081/v1/backends | jq '.backends[] | {url, active_requests}'
```

### Check Available Models

```bash
# List all models across all backends
curl http://localhost:8081/v1/models | jq '.data[].id'
```

## Configuration

### Start with Load Balancing

```bash
make start-proxy \
  SUBNET=10.231.213.0/24,10.231.214.0/24,10.231.215.0/24 \
  PROXY_PORT=8081
```

### With Authentication

```bash
llamacpp lb-proxy \
  --discover-subnet 10.231.213.0/24 \
  --port 8081 \
  --auth-key your-secret-token
```

## Performance Characteristics

- **Discovery**: ~30 seconds for 3 subnets (/24) in background
- **Startup**: <1 second (non-blocking)
- **Health checks**: Every 10 seconds per backend
- **Request overhead**: ~1-2ms for routing decision
- **Concurrency**: Handles 1000+ concurrent requests

## Troubleshooting

### No backends discovered

```bash
# Check verbose logs
docker logs llamacpp-lb-proxy | grep rejected
```

### Backends showing unhealthy

```bash
# Test backend directly
curl http://10.231.213.75:8000/v1/models
```

### Load not balanced

```bash
# Check active_requests
curl http://localhost:8081/v1/backends | jq '.backends[] | {url, active_requests}'

# Verify model-aware routing
curl http://localhost:8081/v1/models
```
