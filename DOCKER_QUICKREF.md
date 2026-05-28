# Docker Quick Reference

## 📦 Two Deployment Modes

| Mode | Purpose | Files |
|------|---------|-------|
| **Backend** | Run `llamacpp serve` on each server | `docker-compose.backend.yml` |
| **Proxy** | Run ONE load balancer for all backends | `docker-compose.proxy.yml` |

## 🚀 Quick Start

### Backend Server (on each machine)

```bash
make build-backend
make pull-model MODEL=qwen3.5
make start-backend
```

### Proxy Server (on ONE machine)

```bash
make build-proxy
make start-proxy SUBNET=192.168.1.0/24
make status-proxy  # Check discovered backends
```

## 🔧 Common Commands

```bash
# Backend
make logs-backend               # View logs
make list-models                # Show models
make test-backend               # Health check

# Proxy
make logs-proxy                 # View logs
make status-proxy               # Show backends
make test-proxy                 # Health check

# Both
make clean                      # Stop containers
make backup-models              # Backup models
```

## 🌐 Example: 3-Server Cluster

```
192.168.1.10 → Backend 1 (qwen3.5)
192.168.1.11 → Backend 2 (qwen3.5)
192.168.1.12 → Backend 3 (llama3:8b)
192.168.1.100 → Proxy (discovers all)
```

Clients connect to proxy at `http://192.168.1.100:8080`

## 📚 Full Documentation

See [DOCKER.md](DOCKER.md) for:
- Deployment scenarios
- Configuration options
- Production setup
- Troubleshooting
- Advanced examples
