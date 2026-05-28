# Troubleshooting: Port Already in Use

## Problem

When starting the proxy, you get:
```
[lb-proxy] Error: port 8080 is already in use.
Kill the existing process or use --port <N> to pick another port.
```

But `sudo lsof -nP -iTCP:8080 -sTCP:LISTEN` shows nothing.

## Root Cause

The backend Dockerfile originally exposed **both** ports 8000 and 8080:
```dockerfile
EXPOSE 8000 8080
```

When the backend container runs, Docker reserves port 8080 even though nothing is listening on it. When the proxy tries to start with `network_mode: host`, it sees port 8080 as "in use" by Docker's internal state.

## Solution

The Dockerfile now only exposes port 8000 (for backends):
```dockerfile
EXPOSE 8000
```

The proxy runs in a separate container and uses port 8080, so there's no conflict.

## Fix Steps

If you built images before this fix:

### Step 1: Stop All Containers

```bash
make clean
# Or manually:
docker compose -f docker-compose.backend.yml down
docker compose -f docker-compose.proxy.yml down
```

### Step 2: Rebuild Images

```bash
# Rebuild backend (removes port 8080)
make build-backend

# Rebuild proxy (no changes needed)
make build-proxy
```

### Step 3: Start Fresh

```bash
# Start backend
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"

# Start proxy
make start-proxy SUBNET=10.231.0.0/16
```

### Step 4: Verify

```bash
# Check backend
curl http://localhost:8000/v1/models

# Check proxy
curl http://localhost:8080/backends
```

## Debug Commands

If you still have issues:

### Check which containers are running:
```bash
docker ps -a | grep llamacpp
```

### Check what ports are exposed:
```bash
docker inspect llamacpp-backend | grep -A10 "ExposedPorts"
```

Should show only:
```json
"ExposedPorts": {
    "8000/tcp": {}
}
```

### Check Docker's port bindings:
```bash
docker port llamacpp-backend
# Should show: 8000/tcp -> 0.0.0.0:8000
# Should NOT show anything about 8080
```

### Force cleanup:
```bash
# Remove all llamacpp containers
docker rm -f $(docker ps -aq --filter "name=llamacpp")

# Rebuild everything
make build-backend
make build-proxy
```

## Alternative: Use Different Port for Proxy

If you can't rebuild, you can use a different port:

### Option 1: Environment Variable

```bash
make start-proxy SUBNET=10.231.0.0/16 PROXY_PORT=8081
```

### Option 2: Edit docker-compose.proxy.yml

```yaml
environment:
  - PROXY_PORT=8081

command: >
  llamacpp lb-proxy
    --host 0.0.0.0
    --port 8081
    --discover-subnet ${DISCOVER_SUBNET:-192.168.1.0/24}
```

Then clients connect to port 8081 instead.

## Why This Happened

The Dockerfile was originally designed to support both:
- Backend (port 8000)
- Proxy (port 8080)

in the same image. However, when using separate docker-compose files, the backend container reserves 8080 even though it doesn't use it, causing conflicts.

The fix separates concerns:
- **Backend image**: Only exposes 8000
- **Proxy image**: Uses same Dockerfile but runs on different port (8080) in its own container

## Prevention

When using `network_mode: host`, only expose the ports your service actually uses. Don't expose ports "just in case" - it causes conflicts with other services.
