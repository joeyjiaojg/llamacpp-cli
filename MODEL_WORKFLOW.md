# Model Management Workflow

## Quick Start: Pull and Serve a Model

### Method 1: Pull First, Then Serve (Recommended)

```bash
# 1. Start backend (temporary)
make start-backend

# 2. Pull model into the container
make pull-model MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF

# 3. Restart with the model
make stop-backend
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

### Method 2: One-Liner (After First Pull)

Once you've pulled the model once, you can just start with it directly:

```bash
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

## Detailed Steps

### Step 1: Pull the Model

Start a temporary container to download the model:

```bash
make start-backend
```

Pull the model (downloads into persistent volume):

```bash
make pull-model MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF
```

This downloads to `/data/llamacpp/.llamacpp/models/` (persisted in Docker volume).

### Step 2: List Available Models

```bash
make list-models
```

Output shows the model name to use (without path/extension):
```
Qwen3.5-9B-Q4_K_M
```

### Step 3: Restart with Model

Stop the backend:

```bash
make stop-backend
```

Start with the model:

```bash
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

Or set it as default in `.env`:

```bash
echo "MODEL_ARGS=--model Qwen3.5-9B-Q4_K_M" >> .env
make start-backend
```

### Step 4: Test

```bash
curl http://localhost:8000/v1/models
# Should show: Qwen3.5-9B-Q4_K_M

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Why Restart is Needed

`llamacpp serve` loads the model at startup. When you pull a model, the server is already running without it. You need to restart to load the new model.

## Alternative: Auto-Restart on Model Pull

Add this to docker-compose.backend.yml to automatically restart:

```yaml
services:
  llama-server:
    restart: unless-stopped
```

Then after pulling:

```bash
docker compose -f docker-compose.backend.yml restart
```

## Model Persistence

Models are stored in Docker volume `llamacpp-models`, so they persist across container restarts:

```bash
# Pull once
make pull-model MODEL=qwen3.5

# Restart many times with same model - no re-download needed
make start-backend MODEL_ARGS="--model qwen3.5"
```

## Multiple Models

You can pull multiple models and switch between them:

```bash
# Pull several models
make pull-model MODEL=qwen3.5
make pull-model MODEL=llama3:8b
make pull-model MODEL=gemma3:2b

# Restart with different models
make start-backend MODEL_ARGS="--model qwen3.5"     # Use qwen
make stop-backend && make start-backend MODEL_ARGS="--model llama3:8b"  # Switch to llama

# Or run multiple backends on different ports
docker compose -f docker-compose.backend.yml up -d
docker run -d -p 8001:8000 -v llamacpp-models:/data/llamacpp/.llamacpp/models \
  llamacpp-cli-llama-server llamacpp serve --model llama3:8b
```

## Production Setup

For production, set the model in docker-compose.backend.yml:

```yaml
services:
  llama-server:
    environment:
      - MODEL_ARGS=--model Qwen3.5-9B-Q4_K_M
    command: >
      bash -c "
        llamacpp serve --host 0.0.0.0 --port 8000 ${MODEL_ARGS}
      "
```

Or use an .env file:

```bash
# .env
MODEL_ARGS=--model Qwen3.5-9B-Q4_K_M
```

Then just:

```bash
make start-backend  # Automatically uses model from .env
```
