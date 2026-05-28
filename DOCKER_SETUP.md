# Docker Setup Summary

## Files Created

1. **`Dockerfile`** - Base image for both backend and proxy
2. **`docker-compose.backend.yml`** - Run on each backend server  
3. **`docker-compose.proxy.yml`** - Run on ONE proxy server
4. **`Makefile`** - Easy management commands
5. **`DOCKER.md`** - Complete deployment guide (19 sections)
6. **`DOCKER_QUICKREF.md`** - Quick reference card
7. **`.dockerignore`** - Optimize build context

## Architecture

```
Proxy Server (ONE)              Backend Servers (MANY)
==================              ====================
192.168.1.100:8080              192.168.1.10:8000
    |                           192.168.1.11:8000
    |                           192.168.1.12:8000
    |                                   |
    +-----------------------------------+
         Auto-discovers via subnet scan
         Routes by model + least-connections
```

## Key Features

✅ **Separate configurations** for backend vs proxy roles
✅ **Makefile** for easy deployment (`make start-backend`, `make start-proxy`)  
✅ **Subnet auto-discovery** - proxy finds backends automatically
✅ **Persistent volumes** - models survive restarts
✅ **Model-aware routing** - routes to backends with requested model
✅ **Least-connections LB** - distributes load evenly
✅ **Health checks** - auto-removes unhealthy backends
✅ **Config hot-reload** - add/remove backends without restart

## Usage

### On Backend Servers (3+ machines)

```bash
cd /path/to/llamacpp-cli

# First time setup
make build-backend
make pull-model MODEL=qwen3.5

# Start serving
make start-backend

# Check status
make logs-backend
```

### On Proxy Server (1 machine)

```bash
cd /path/to/llamacpp-cli

# First time setup
make build-proxy

# Start proxy with subnet discovery
make start-proxy SUBNET=192.168.1.0/24

# Wait 10-15 seconds, then check
make status-proxy
```

### From Client

```bash
# List available models
curl http://192.168.1.100:8080/v1/models

# Chat completion
curl http://192.168.1.100:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Deployment Scenarios

### Scenario 1: Same Model on All Backends
Load distribution across identical backends:
- All backends: `make start-backend MODEL_ARGS="--model qwen3.5"`
- Proxy uses least-connections to distribute load

### Scenario 2: Different Models per Backend
Model-specific routing:
- Backend 1: qwen3.5
- Backend 2: llama3:8b  
- Backend 3: gemma3:2b
- Proxy routes based on `model` field in request

### Scenario 3: Manual Configuration
No subnet discovery (firewall/VPN):
- Edit `docker-compose.proxy.yml` to add `--backend` flags
- Or create `lb_backends.json` config file

## Management Commands

```bash
# Backend
make logs-backend      # View logs
make list-models       # Show models
make test-backend      # Health check
make stop-backend      # Stop server

# Proxy
make logs-proxy        # View logs
make status-proxy      # Show backends + health
make test-proxy        # Health check
make stop-proxy        # Stop proxy

# Maintenance
make clean             # Stop all
make backup-models     # Backup to tar.gz
make restore-models    # Restore from tar.gz
make clean-volumes     # Delete volumes (careful!)
```

## Production Checklist

- [ ] Add resource limits (CPU/memory) in compose files
- [ ] Configure health checks for auto-recovery
- [ ] Set up log rotation (max-size, max-file)
- [ ] Run as non-root user
- [ ] Configure firewall rules (allow 8000, 8080)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Back up model volumes regularly
- [ ] Test failover (stop one backend, verify proxy routes around it)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No backends found | Check firewall, verify port 8000 accessible |
| Models don't persist | Don't use `docker-compose down -v` |
| 503 errors | Check backend health via `make status-proxy` |
| High memory usage | Add resource limits, reduce context size |

## Next Steps

1. **Test locally** - Use `docker-compose.yml` for single-machine testing
2. **Deploy backends** - Clone repo to each backend server, run `make start-backend`
3. **Deploy proxy** - Clone repo to proxy server, run `make start-proxy`
4. **Monitor** - Check `make status-proxy` regularly
5. **Scale** - Add more backends by repeating step 2

See **DOCKER.md** for complete documentation.
