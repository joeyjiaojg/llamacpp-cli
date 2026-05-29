# Load Balancer Proxy

The `llamacpp lb-proxy` command provides a smart load balancer for distributing requests across multiple llama.cpp server instances.

## Features

- **Model-aware routing**: Routes requests to backends that have the requested model
- **Weighted least-connections load balancing**: Distributes load proportionally based on backend capacity
- **Auto health checks**: Continuously monitors backend health and removes unhealthy instances
- **Auto-discovery**: 
  - Config file watching with hot-reload
  - Subnet scanning to discover backends automatically
- **OpenAI-compatible**: Drop-in replacement for OpenAI API clients

## Quick Start

### 1. Manual Backend Configuration

```bash
llamacpp lb-proxy \
  --backend http://machine1:8000 \
  --backend http://machine2:8000 \
  --backend http://machine3:8000
```

Or with weights (for backends with different capacities):

```bash
llamacpp lb-proxy \
  --backend http://machine1:8000:1.0 \
  --backend http://machine2:8000:2.0 \
  --backend http://machine3:8000:0.5
```

The weight indicates relative capacity:
- `2.0` = can handle 2x the load of a `1.0` backend
- `0.5` = can handle half the load of a `1.0` backend

### 2. Auto-Discovery on Subnet

Scan your local network for llama-server instances:

```bash
llamacpp lb-proxy --discover-subnet 192.168.1.0/24
```

### 3. Config File (Recommended)

Create `~/.llamacpp/lb_backends.json`:

```json
{
  "backends": [
    {"host": "192.168.1.10", "port": 8000, "weight": 1.0},
    {"host": "192.168.1.11", "port": 8000, "weight": 2.0},
    {"host": "192.168.1.12", "port": 8000, "weight": 1.5}
  ]
}
```

Weight is optional (defaults to 1.0). Use weights to represent different backend capacities:
- A backend with weight `2.0` will receive roughly 2x the requests of a `1.0` backend
- A backend with weight `0.5` will receive roughly half the requests

Start the proxy:

```bash
llamacpp lb-proxy --config ~/.llamacpp/lb_backends.json
```

The proxy watches the config file and auto-reloads when you add/remove backends.

## How It Works

### Request Routing

1. **Model extraction**: Extract `model` field from incoming `/v1/chat/completions` request
2. **Model filtering**: Filter to backends that have the requested model (via `/v1/models` query)
3. **Weighted least-connections**: Among matching backends, calculate score = `active_requests / weight` and pick the backend with the lowest score
4. **Fallback**: If model not found on any backend, use weighted least-connections across all healthy backends

The weighted selection ensures backends with higher weights (more capacity) receive proportionally more load.

### Health Checks

- Every 10 seconds (configurable), queries each backend's `/health` or `/v1/models` endpoint
- Unhealthy backends are removed from rotation
- Model lists are refreshed on each health check

### Config Auto-Reload

- Watches config file for changes (5-second polling)
- Automatically adds new backends and removes deleted ones
- New backends are health-checked immediately

## API Endpoints

### Proxy Endpoints

- `POST /v1/chat/completions` - Chat completions (routed to backends)
- `GET /v1/models` - Aggregated list of models from all healthy backends

### Management Endpoints

- `GET /health` - Proxy health status
- `GET /backends` - List all backends and their status

Example:

```bash
curl http://localhost:8080/backends | jq
```

Output:

```json
{
  "backends": [
    {
      "url": "http://192.168.1.10:8000",
      "healthy": true,
      "models": ["qwen3.5:1.5b-q4_k_m", "gemma3:2b-q4_k_m"],
      "active_requests": 2,
      "load_status": "busy",
      "weight": 1.0
    },
    {
      "url": "http://192.168.1.11:8000",
      "healthy": true,
      "models": ["llama3:8b-q4_k_m"],
      "active_requests": 1
    }
  ]
}
```

## Use Cases

### Multi-Machine Setup

You have several 52c/192GB CPU machines, each running one llama.cpp server:

```
Machine 1 (192.168.1.10): llamacpp serve --port 8000 --model qwen3.5
Machine 2 (192.168.1.11): llamacpp serve --port 8000 --model gemma3
Machine 3 (192.168.1.12): llamacpp serve --port 8000 --model llama3:8b
```

Start the load balancer on a dedicated machine (or any machine):

```bash
llamacpp lb-proxy --discover-subnet 192.168.1.0/24 --port 8080
```

Point your OpenCode/OpenAI client to the proxy:

```bash
export OPENAI_API_BASE=http://proxy-machine:8080/v1
```

Now requests are distributed across all 3 machines based on:
- Which model is requested
- Which backend has the fewest active requests

### Same Model on Multiple Machines

All machines run the same model for higher throughput:

```
Machine 1-5: llamacpp serve --port 8000 --model qwen3.5
```

Load balancer distributes requests evenly using least-connections:

```bash
llamacpp lb-proxy --discover-subnet 192.168.1.0/24
```

### Mixed Models

Each machine runs different quantizations or different models:

```
Machine 1: qwen3.5:1.5b-q4_k_m (fast, lower quality)
Machine 2: qwen3.5:1.5b-q8_0 (slower, higher quality)
Machine 3: llama3:8b-q4_k_m (different model)
```

Proxy routes based on the `model` field in requests:

```python
import openai
client = openai.OpenAI(base_url="http://proxy:8080/v1")

# Routes to Machine 1
response = client.chat.completions.create(
    model="qwen3.5:1.5b-q4_k_m",
    messages=[{"role": "user", "content": "Hello"}]
)

# Routes to Machine 3
response = client.chat.completions.create(
    model="llama3:8b-q4_k_m",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Configuration Options

```
--host            Host to bind (default: 127.0.0.1)
--port, -p        Port to bind (default: 8080)
--config, -c      Path to backends config JSON
--backend, -b     Backend URL (can be repeated)
--discover-subnet Auto-discover backends on subnet (e.g., 192.168.1.0/24)
--discover-port   Port to scan during discovery (default: 8000)
```

## Comparison with LiteLLM Proxy

| Feature | llamacpp lb-proxy | LiteLLM Proxy |
|---------|-------------------|---------------|
| Least-connections LB | ✅ | ❌ (round-robin) |
| Model-aware routing | ✅ | ✅ |
| Auto-discovery | ✅ | ❌ |
| Health checks | ✅ | ✅ |
| Config hot-reload | ✅ | ❌ |
| Zero config | ✅ | ❌ |
| Dependencies | Minimal (FastAPI, httpx) | Many |

## Troubleshooting

### No backends found

```
[lb-proxy] No backends configured.
```

**Solution**: Add backends via:
- `--backend` flag
- `--discover-subnet` flag
- Config file at `~/.llamacpp/lb_backends.json`

### All backends unhealthy

```
[lb-proxy] Backend http://192.168.1.10:8000 unhealthy
```

**Solution**:
- Check that llama-server is running on backend: `curl http://192.168.1.10:8000/health`
- Check firewall rules allow access
- Check network connectivity: `ping 192.168.1.10`

### Model not found

```
503 No backends available for model 'qwen3.5'
```

**Solution**:
- Verify model is loaded on at least one backend: `curl http://localhost:8080/v1/models`
- Check backend health: `curl http://localhost:8080/backends`
- Load the model on a backend: `llamacpp serve --model qwen3.5` on one of the machines

## Advanced

### Custom Health Check Interval

The proxy checks backend health every 10 seconds by default. To customize, modify `lb_proxy.py`:

```python
state.health_check_interval = 30.0  # Check every 30 seconds
```

### Sticky Sessions

For use cases requiring conversation continuity to the same backend, you can add session affinity by hashing conversation IDs. This is not currently implemented but can be added to `_select_backend()`.

### Metrics and Monitoring

You can add Prometheus metrics by integrating `prometheus-fastapi-instrumentator`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = create_lb_app(state)
Instrumentator().instrument(app).expose(app)
```

Then scrape metrics from `http://proxy:8080/metrics`.
