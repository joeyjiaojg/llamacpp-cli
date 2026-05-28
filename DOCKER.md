# Docker Deployment

Deploy llamacpp-cli in Docker with separate backend and proxy configurations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Proxy Server                        │
│                  (192.168.1.100:8080)                   │
│                                                          │
│  docker run lb-proxy --discover-subnet 192.168.1.0/24   │
└───────────────┬──────────────────────┬──────────────────┘
                │                      │
        ┌───────▼───────┐      ┌───────▼───────┐
        │  Backend 1    │      │  Backend 2    │
        │ .1.10:8000    │      │ .1.11:8000    │
        │ llamacpp      │      │ llamacpp      │
        │   serve       │      │   serve       │
        └───────────────┘      └───────────────┘
```

## Quick Start

### On Each Backend Server

```bash
# Clone repo
git clone https://github.com/joeyjiaojg/llamacpp-cli
cd llamacpp-cli

# Build and start
make build-backend
make pull-model MODEL=qwen3.5
make start-backend

# Test
curl http://localhost:8000/health
```

### On Proxy Server (ONE machine)

```bash
# Clone repo
git clone https://github.com/joeyjiaojg/llamacpp-cli
cd llamacpp-cli

# Build and start with subnet discovery
make build-proxy
make start-proxy SUBNET=192.168.1.0/24

# Check backends (wait 10-15 seconds for discovery)
make status-proxy

# Test
curl http://localhost:8080/v1/models
```

### Client Usage

Point clients to the proxy:

```bash
# OpenAI-compatible
export OPENAI_API_BASE=http://192.168.1.100:8080/v1

curl http://192.168.1.100:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Makefile Commands

### Backend Server (run on each backend machine)

```bash
make build-backend              # Build image
make start-backend              # Start server (port 8000)
make stop-backend               # Stop server
make logs-backend               # View logs
make pull-model MODEL=qwen3.5   # Pull model
make list-models                # List models
make test-backend               # Health check
```

### Proxy Server (run on ONE proxy machine)

```bash
make build-proxy                          # Build image
make start-proxy SUBNET=192.168.1.0/24    # Start with discovery
make stop-proxy                           # Stop proxy
make logs-proxy                           # View logs
make status-proxy                         # Show backends
make test-proxy                           # Health check
```

### Maintenance

```bash
make clean                 # Stop all containers
make clean-volumes         # Remove volumes (deletes models!)
make backup-models         # Backup to models-backup.tar.gz
make restore-models        # Restore from backup
```

## Deployment Scenarios

### Scenario 1: Homogeneous Cluster (Same Model)

All backends run the same model for load distribution:

```bash
# On each backend (192.168.1.10, .11, .12)
make pull-model MODEL=qwen3.5
make start-backend MODEL_ARGS="--model qwen3.5"

# On proxy (192.168.1.100)
make start-proxy SUBNET=192.168.1.0/24
```

Requests are distributed via least-connections load balancing.

### Scenario 2: Heterogeneous Cluster (Different Models)

Each backend runs a different model:

```bash
# Backend 1: qwen3.5
make pull-model MODEL=qwen3.5
make start-backend MODEL_ARGS="--model qwen3.5"

# Backend 2: llama3:8b
make pull-model MODEL=llama3:8b
make start-backend MODEL_ARGS="--model llama3:8b"

# Backend 3: gemma3:2b
make pull-model MODEL=gemma3:2b
make start-backend MODEL_ARGS="--model gemma3:2b"

# Proxy: auto-routes based on model field
make start-proxy SUBNET=192.168.1.0/24
```

The proxy routes requests to backends that have the requested model.

### Scenario 3: Manual Configuration (No Discovery)

If subnet discovery doesn't work (firewall, VPN, etc.):

**Option A: Static backends via command**

Edit `docker-compose.proxy.yml`, replace discovery with static backends:

```yaml
command: >
  llamacpp lb-proxy
    --host 0.0.0.0
    --port 8080
    --backend http://192.168.1.10:8000
    --backend http://192.168.1.11:8000
    --backend http://192.168.1.12:8000
```

**Option B: Config file**

Create config file on proxy:

```bash
docker-compose -f docker-compose.proxy.yml exec lb-proxy sh -c \
  'cat > /data/llamacpp/.llamacpp/lb_backends.json << EOF
{
  "backends": [
    {"host": "192.168.1.10", "port": 8000},
    {"host": "192.168.1.11", "port": 8000},
    {"host": "192.168.1.12", "port": 8000}
  ]
}
EOF'

# Restart to use config
docker-compose -f docker-compose.proxy.yml restart
```

## Persistent Volumes

### Backend Volumes

| Volume | Purpose | Path |
|--------|---------|------|
| `llamacpp-models` | GGUF model files | `/data/llamacpp/.llamacpp/models` |
| `llamacpp-bin` | llama.cpp binaries | `/data/llamacpp/.llamacpp/bin` |
| `llamacpp-config` | Config, DB, metadata | `/data/llamacpp/.llamacpp` |

### Proxy Volumes

| Volume | Purpose | Path |
|--------|---------|------|
| `llamacpp-config` | `lb_backends.json` config | `/data/llamacpp/.llamacpp` |

### Backup and Restore

```bash
# Backup models from a backend
make backup-models
# Creates: models-backup.tar.gz

# Restore models on another backend
scp models-backup.tar.gz backend2:/path/to/llamacpp-cli/
ssh backend2 "cd /path/to/llamacpp-cli && make restore-models"
```

## Environment Variables

Override defaults via Makefile:

```bash
# Proxy with custom subnet and port
make start-proxy SUBNET=10.0.0.0/24 PROXY_PORT=9090

# Backend with specific model and context size
make start-backend MODEL_ARGS="--model qwen3.5 --ctx-size 8192"
```

Available variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBNET` | `192.168.1.0/24` | Subnet for proxy discovery |
| `PROXY_PORT` | `8080` | Proxy listen port |
| `MODEL` | - | Model name for `pull-model` |
| `MODEL_ARGS` | - | Extra args for `llamacpp serve` |

### Docker Compose Variables

Set in `.env` file or export before running:

```bash
# Backend
LLAMACPP_SSL_VERIFY=false    # Disable SSL for Hugging Face downloads
MODEL_ARGS="--model qwen3.5"  # Default model to serve

# Proxy
DISCOVER_SUBNET=10.0.0.0/8   # Subnet to scan
PROXY_PORT=8080               # Listen port
```

## Management

### View Backend Status

```bash
make status-proxy
# Or:
curl http://localhost:8080/backends | jq
```

Output:

```json
{
  "backends": [
    {
      "url": "http://192.168.1.10:8000",
      "healthy": true,
      "models": ["qwen3.5:1.5b-q4_k_m"],
      "active_requests": 2
    },
    {
      "url": "http://192.168.1.11:8000",
      "healthy": true,
      "models": ["llama3:8b-q4_k_m"],
      "active_requests": 0
    }
  ]
}
```

### Monitor Logs

```bash
# Backend logs
make logs-backend

# Proxy logs
make logs-proxy

# Follow logs
docker-compose -f docker-compose.backend.yml logs -f
docker-compose -f docker-compose.proxy.yml logs -f
```

### Restart Services

```bash
# Backend
docker-compose -f docker-compose.backend.yml restart

# Proxy
docker-compose -f docker-compose.proxy.yml restart
```

## Production Considerations

### 1. Resource Limits

Add to `docker-compose.backend.yml`:

```yaml
services:
  llama-server:
    deploy:
      resources:
        limits:
          cpus: '32'
          memory: 128G
        reservations:
          cpus: '16'
          memory: 64G
```

### 2. Health Checks

```yaml
services:
  llama-server:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### 3. Logging

```yaml
services:
  llama-server:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

### 4. Security

```yaml
services:
  llama-server:
    user: "1000:1000"
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### 5. Networking

For production, use overlay networks or configure firewall rules:

```bash
# Allow backend ports
ufw allow 8000/tcp

# Allow proxy port
ufw allow 8080/tcp
```

## Troubleshooting

### Problem: Subnet discovery finds no backends

**Symptoms:**
```
[lb-proxy] No backends discovered on subnet 192.168.1.0/24
```

**Solutions:**
1. Check backends are running: `curl http://192.168.1.10:8000/health`
2. Check firewall allows port 8000: `ufw status`
3. Check Docker network mode: proxy needs `network_mode: host` for discovery
4. Use manual configuration instead (see Scenario 3)

### Problem: Models not persisting after restart

**Symptoms:** Models disappear after `docker-compose down`

**Solutions:**
1. Verify volumes exist: `docker volume ls | grep llamacpp`
2. Don't use `docker-compose down -v` (removes volumes)
3. Check volume mounts: `docker-compose config`

### Problem: Permission denied in containers

**Symptoms:** Can't write to `/data/llamacpp/.llamacpp/`

**Solutions:**
```bash
# Fix ownership
docker-compose exec llama-server chown -R root:root /data/llamacpp

# Or run as specific user
docker-compose exec -u 1000:1000 llama-server llamacpp pull qwen3.5
```

### Problem: Proxy returns 503 Service Unavailable

**Symptoms:** All requests fail with 503

**Solutions:**
1. Check backend health: `make status-proxy`
2. Check backend logs: `make logs-backend`
3. Verify model is loaded: `curl http://backend:8000/v1/models`
4. Check network connectivity: `docker-compose exec lb-proxy ping 192.168.1.10`

### Problem: High memory usage

**Symptoms:** OOM killed or swap thrashing

**Solutions:**
1. Add resource limits (see Production Considerations)
2. Reduce context size: `MODEL_ARGS="--ctx-size 2048"`
3. Use smaller quantization: `q4_k_m` instead of `q8_0`
4. Reduce parallel requests in proxy

## Examples

### Example 1: 3-Backend Setup with Same Model

```bash
# Backend 1 (192.168.1.10)
make build-backend
make pull-model MODEL=qwen3.5
make start-backend MODEL_ARGS="--model qwen3.5"

# Backend 2 (192.168.1.11)
make build-backend
make pull-model MODEL=qwen3.5
make start-backend MODEL_ARGS="--model qwen3.5"

# Backend 3 (192.168.1.12)
make build-backend
make pull-model MODEL=qwen3.5
make start-backend MODEL_ARGS="--model qwen3.5"

# Proxy (192.168.1.100)
make build-proxy
make start-proxy SUBNET=192.168.1.0/24

# Test
curl http://192.168.1.100:8080/backends
```

### Example 2: Mixed Model Setup

```bash
# Backend 1: Fast small model
make pull-model MODEL=qwen3.5:1.5b
make start-backend MODEL_ARGS="--model qwen3.5:1.5b"

# Backend 2: Higher quality
make pull-model MODEL=llama3:8b
make start-backend MODEL_ARGS="--model llama3:8b"

# Backend 3: Large model
make pull-model MODEL=qwen3:14b
make start-backend MODEL_ARGS="--model qwen3:14b"

# Proxy routes by model name
make start-proxy SUBNET=192.168.1.0/24

# Client specifies model
curl http://proxy:8080/v1/chat/completions \
  -d '{"model": "qwen3:14b", "messages": [...]}'
```

### Example 3: Development Setup (Single Machine)

For testing on one machine:

```bash
# Use the original docker-compose.yml
docker-compose up -d

# Both server and proxy run locally
curl http://localhost:8000/health   # Direct backend
curl http://localhost:8080/health   # Via proxy
```
