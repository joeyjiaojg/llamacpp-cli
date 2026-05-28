# Complete Example: From Zero to Working Backend

## Step-by-Step Example with Qwen3.5-9B

This guide walks through the complete workflow from building the image to serving a model.

### Step 1: Build the Docker Image

```bash
cd /path/to/llamacpp-cli
make build-backend
```

**Expected output:**
```
Using direct download URL from LLAMACPP_RELEASE_URL...
Downloading llama-b9371-bin-ubuntu-x64.tar.gz...
Extracting...
llama.cpp installed to /data/llamacpp/.llamacpp/bin
✓ Image Built Successfully
```

**Time:** ~2 minutes (includes 94s llama.cpp download)

### Step 2: Start Backend (Temporary)

```bash
make start-backend
```

**Expected output:**
```
Starting backend server on port 8000...
No model specified. Use: make start-backend MODEL_ARGS='--model qwen3.5'
Or pull a model first: make pull-model MODEL=qwen3.5
Backend started!
```

**Note:** Server starts but won't respond to requests yet (no model loaded).

### Step 3: Pull a Model

```bash
make pull-model MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF
```

**Expected output:**
```
Pulling model: jc-builds/Qwen3.5-9B-Q4_K_M-GGUF
Downloading from Hugging Face...
[████████████████████] 100%
Model saved to /data/llamacpp/.llamacpp/models/Qwen3.5-9B-Q4_K_M.gguf
```

**Time:** Depends on model size (~4GB for this model)

### Step 4: Check Model Name

```bash
make list-models
```

**Expected output:**
```
Models on this backend:
Qwen3.5-9B-Q4_K_M
```

**Note the model name** (without .gguf extension) - you'll use this in the next step.

### Step 5: Restart with Model

```bash
make stop-backend
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

**Or use restart command:**
```bash
make restart-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

**Expected output:**
```
Starting backend server on port 8000...
Model args: --model Qwen3.5-9B-Q4_K_M
Backend started!
```

### Step 6: Verify It's Working

```bash
# Check health
curl http://localhost:8000/health

# Expected: {"status":"ok"} or {"status":"loading_model"}

# List models
curl http://localhost:8000/v1/models

# Expected: {"data":[{"id":"Qwen3.5-9B-Q4_K_M",...}]}

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello! What is 2+2?"}],
    "max_tokens": 100
  }'
```

**Expected:** JSON response with the model's answer.

### Step 7: Check Logs (If Issues)

```bash
make logs-backend
```

Look for:
- ✓ `llama_model_load: model loaded` - Good!
- ✗ `error: model not found` - Check model name
- ✗ `error: unable to load model` - Model file corrupted, pull again

## Next Time (Model Already Pulled)

Once you've pulled the model, you can skip steps 2-4:

```bash
# Just start with the model directly
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

## Production Setup

Create `.env` file to avoid typing MODEL_ARGS every time:

```bash
cat > .env << 'EOF'
MODEL_ARGS=--model Qwen3.5-9B-Q4_K_M
EOF
```

Then:

```bash
make start-backend  # Automatically uses model from .env
```

## Common Issues

### Issue 1: "Empty reply from server" (curl error 52)

**Problem:** Server started but no model loaded.

**Solution:**
```bash
# Check if model is specified
make logs-backend | grep "model"

# If no model, restart with model
make stop-backend
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

### Issue 2: "Model not found"

**Problem:** Model name doesn't match.

**Solution:**
```bash
# Check exact model name
make list-models

# Use the exact name shown
make restart-backend MODEL_ARGS="--model <exact-name-from-list>"
```

### Issue 3: "Cannot connect to port 8000"

**Problem:** Container not running.

**Solution:**
```bash
# Check container status
docker compose -f docker-compose.backend.yml ps

# If not running, start it
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"
```

### Issue 4: Model loads slowly

**Symptom:** `curl` hangs for 30+ seconds.

**Reason:** Large models take time to load into memory.

**Solution:** Wait. Check logs to see progress:
```bash
make logs-backend | grep -E "loaded|progress"
```

## Multi-Model Setup

Pull several models:

```bash
make pull-model MODEL=qwen3.5
make pull-model MODEL=llama3:8b
make pull-model MODEL=gemma3:2b

make list-models
# Shows all three
```

Switch between them:

```bash
# Use qwen
make stop-backend
make start-backend MODEL_ARGS="--model qwen3.5"

# Switch to llama
make stop-backend
make start-backend MODEL_ARGS="--model llama3:8b"
```

## Quick Reference Card

```bash
# First time setup
make build-backend                              # Once
make start-backend                              # Temporary
make pull-model MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF
make stop-backend
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"

# Every time after
make start-backend MODEL_ARGS="--model Qwen3.5-9B-Q4_K_M"

# Or with .env
echo 'MODEL_ARGS=--model Qwen3.5-9B-Q4_K_M' > .env
make start-backend
```

## Time Estimates

| Step | Time |
|------|------|
| Build image | ~2 min |
| Pull model (9B Q4) | ~5-10 min |
| Start backend | ~5 sec |
| Load model | ~30-60 sec |
| **Total first time** | ~10-15 min |
| **Restart with model** | ~40-70 sec |
