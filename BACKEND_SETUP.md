# Backend Setup Guide

## Quick Start

### 1. Start Backend Server (No Authentication)

```bash
# On each backend machine
make start-backend MODEL_ARGS="--model qwen3.5"
```

This starts a backend on port 8000 that will be auto-discovered by the proxy.

### 2. Start Proxy with Discovery

```bash
# On proxy machine
make start-proxy SUBNET=10.231.213.0/24,10.231.214.0/24 PROXY_PORT=8081
```

---

## Authentication Setup

There are **TWO types of authentication**:

### Auth Type 1: Backend Discovery Authentication (`--auth-key`)

**Purpose**: Restrict which backends can join the proxy pool during discovery.

**Who uses it**: Backends during discovery scan.

**How it works**:
1. Proxy starts with `--auth-key secret123`
2. During discovery, proxy sends `Authorization: Bearer secret123` when checking backends
3. Only backends that **validate** this key are added to the pool
4. Backends don't need to specify the key anywhere - the proxy validates them

**Backend configuration**: None needed - backends are passive during discovery.

**Proxy configuration**:
```bash
# Start proxy with backend auth
llamacpp lb-proxy \
  --discover-subnet 10.231.213.0/24 \
  --port 8081 \
  --auth-key my-backend-secret-key
```

**When to use**:
- Multi-tenant environments where you want to isolate backend pools
- Security-sensitive networks where you want to prevent rogue backends from joining
- When you have multiple proxy instances and want to partition backends

**Note**: This feature is currently **not fully implemented** in the backend. Backends currently ignore the auth header. To implement:
1. Backend must read `Authorization` header in `/v1/models` endpoint
2. Compare against expected token
3. Return 401 if mismatch

---

### Auth Type 2: Client API Key Authentication (`--api-key`)

**Purpose**: Require authentication from clients making API requests.

**Who uses it**: Clients (curl, Python, etc.) when calling the proxy.

**How it works**:
1. Proxy starts with `--api-key client-key-456`
2. Clients must send `Authorization: Bearer client-key-456` with every request
3. Proxy validates the key before forwarding to backends
4. Backends never see the client API key

**Proxy configuration**:
```bash
# Start proxy with client auth
llamacpp lb-proxy \
  --discover-subnet 10.231.213.0/24 \
  --port 8081 \
  --api-key my-client-api-key
```

**Client usage**:
```bash
# curl example
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Authorization: Bearer my-client-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Python OpenAI SDK example
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="my-client-api-key"  # Your --api-key value
)

response = client.chat.completions.create(
    model="Qwen3.5-9B-Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**When to use**:
- Public-facing proxy that needs access control
- Cost tracking per API key
- Rate limiting per client
- Multi-user environments

---

## Combined Authentication Example

You can use both auth types simultaneously:

```bash
# Proxy with both backend auth and client auth
llamacpp lb-proxy \
  --discover-subnet 10.231.213.0/24 \
  --port 8081 \
  --auth-key backend-secret-abc123 \
  --api-key client-key-xyz789
```

This configuration:
- Only discovers backends that validate `backend-secret-abc123`
- Only accepts client requests with `Authorization: Bearer client-key-xyz789`

---

## Generating Secure Keys

**Method 1: Python**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Output: vJpX8Q7K2nR5mZ9wT3hY6uL1cF4dS0gE8aP2bN5kM7jI9oU6qW3rT
```

**Method 2: OpenSSL**
```bash
openssl rand -base64 32
# Output: YjZhNzE3YzQtZGVhNC00ZDg2LThiYTktOGE4YzY4ZTU5YWUy
```

**Method 3: uuidgen**
```bash
uuidgen
# Output: 6ba717c4-dea4-4d86-8ba9-8a8c68e59ae2
```

---

## Backend Health Check

The proxy health checks backends every 10 seconds by calling:
```
GET http://<backend-ip>:8000/v1/models
```

If `--auth-key` is set, the health check includes:
```
Authorization: Bearer <auth-key>
```

Backends must return:
```json
{
  "object": "list",
  "data": [
    {"id": "model-name.gguf", "object": "model"}
  ]
}
```

---

## Protected vs Unprotected Endpoints

When `--api-key` is set:

**Protected** (require client API key):
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`

**Unprotected** (always accessible):
- `/v1/models` - Public model list
- `/backends` or `/v1/backends` - Backend status
- `/health` - Proxy health

---

## Docker Compose Examples

### Backend without Auth
```yaml
# docker-compose.backend.yml
services:
  llama-server:
    image: llamacpp-cli-backend
    ports:
      - "8000:8000"
    command: llamacpp serve --host 0.0.0.0 --port 8000 --model qwen3.5
```

### Proxy without Auth (Open)
```yaml
# docker-compose.proxy.yml
services:
  lb-proxy:
    image: llamacpp-cli-lb-proxy
    ports:
      - "8081:8081"
    environment:
      - DISCOVER_SUBNET=10.231.213.0/24,10.231.214.0/24
    command: >
      llamacpp lb-proxy
      --host 0.0.0.0
      --port 8081
      --discover-subnet ${DISCOVER_SUBNET}
```

### Proxy with Client Auth
```yaml
# docker-compose.proxy.yml
services:
  lb-proxy:
    image: llamacpp-cli-lb-proxy
    ports:
      - "8081:8081"
    environment:
      - DISCOVER_SUBNET=10.231.213.0/24
      - CLIENT_API_KEY=your-secret-key-here
    command: >
      llamacpp lb-proxy
      --host 0.0.0.0
      --port 8081
      --discover-subnet ${DISCOVER_SUBNET}
      --api-key ${CLIENT_API_KEY}
```

---

## Security Best Practices

1. **Use environment variables** for keys, never hardcode:
   ```bash
   export LB_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   llamacpp lb-proxy --api-key "$LB_API_KEY"
   ```

2. **Use HTTPS in production**: Put nginx/caddy in front of the proxy:
   ```nginx
   server {
       listen 443 ssl;
       server_name llm-api.example.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:8081;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Rotate keys periodically**: Change API keys every 30-90 days

4. **Monitor unauthorized attempts**: Check logs for 401 responses

5. **Use long, random keys**: Minimum 32 characters, use cryptographic random generator

---

## Troubleshooting

### Backends Not Being Discovered

**Problem**: Proxy logs show backends are rejected during discovery.

**Check**:
1. Is `--auth-key` set on proxy but backends don't support it?
   - **Solution**: Remove `--auth-key` from proxy (current backends don't validate it)

2. Are backends returning invalid `/v1/models` response?
   - **Test**: `curl http://backend-ip:8000/v1/models`
   - **Expected**: `{"object":"list","data":[...]}`

### Client Requests Getting 401

**Problem**: Client requests return `{"detail": "Invalid or missing API key"}`

**Check**:
1. Is `--api-key` set on proxy?
2. Is client sending `Authorization: Bearer <key>` header?
3. Does the key match exactly?

**Test**:
```bash
# Without auth (should fail if --api-key is set)
curl http://localhost:8081/v1/models

# With auth
curl -H "Authorization: Bearer your-api-key" http://localhost:8081/v1/models
```

### Backends Flapping Between Healthy/Unhealthy

**Problem**: Logs show rapid state changes:
```
[lb-proxy] Backend http://10.231.214.183:8000 became unhealthy
[lb-proxy] Backend http://10.231.214.183:8000 became healthy
```

**Causes**:
1. Network latency/packet loss
2. Backend overloaded (slow to respond)
3. Health check timeout too aggressive (5s)

**Solutions**:
1. Increase health check timeout (see agent recommendations)
2. Implement consecutive-failure threshold
3. Check network quality: `ping backend-ip`
4. Check backend load: `curl http://backend-ip:8000/health`

---

## Summary

| Feature | Flag | Used By | When Required |
|---------|------|---------|---------------|
| Backend Discovery Auth | `--auth-key` | Backends (passive) | Optional - multi-tenant isolation |
| Client API Auth | `--api-key` | Clients (active) | Optional - access control |
| No Auth | (no flags) | Anyone | Default - open access |

**Current status**: Only `--api-key` (client auth) is fully implemented. Backend auth (`--auth-key`) validation needs to be added to llama-server backends.
