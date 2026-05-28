# Quick Start: CPU-Optimized Server

This guide shows how to use the new CPU-optimized presets.

## Installation

```bash
pip install -e .
llamacpp install  # Install llama.cpp binaries
```

## Pull a Model

```bash
# For code tasks
llamacpp pull codellama:7b

# Or use a smaller model for testing
llamacpp pull qwen3:270m
```

## Start Server with Presets

### Default: Code Preset (Recommended)

Best for code tasks, multi-file analysis:

```bash
llamacpp serve --model codellama:7b
```

Output:
```
Starting llama.cpp server with preset 'code':
  Host: 0.0.0.0:8080
  Context: 16384 tokens
  Parallel requests: 2
  CPU threads: 32
  Batch size: 512
  Memory lock: enabled
  NUMA: enabled
  Pre-loading model: codellama:7b
```

### Chat Preset

Higher concurrency for conversational workloads:

```bash
llamacpp serve --preset chat --model qwen3:270m
```

### Fast Preset

Maximum throughput for quick queries:

```bash
llamacpp serve --preset fast --model qwen3:270m
```

### Max Context Preset

Large codebases (slower on CPU):

```bash
llamacpp serve --preset max-context --model codellama:13b
```

⚠️ **Warning**: This will show a warning about slow performance on CPU.

## Custom Configuration

Override preset parameters:

```bash
# Start with code preset but increase context
llamacpp serve --preset code --ctx-size 32768 --parallel 1

# Manual configuration (no preset)
llamacpp serve --ctx-size 8192 --parallel 4 --threads 16
```

## Test the Server

```bash
# Install curl or httpie
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codellama:7b",
    "messages": [{"role": "user", "content": "Write a hello world in Python"}],
    "stream": true
  }'
```

## Check Token Usage

With the load balancer proxy:

```bash
# Start lb-proxy with your backends
llamacpp lb-proxy -b http://localhost:8000

# View stats dashboard
open http://localhost:8080/stats

# Or get JSON stats
curl http://localhost:8080/stats
```

Now **completion tokens** will be correctly tracked! ✅

## Performance Tips

1. **Monitor CPU usage**: If cores are idle, increase `--parallel`
2. **Watch latency**: If requests timeout, reduce `--parallel` or `--ctx-size`
3. **Start conservative**: Use default `code` preset, tune from there
4. **Enable NUMA**: Auto-enabled on multi-socket servers (2x CPU sockets)

## Troubleshooting

### Server is slow
```bash
llamacpp serve --preset fast  # Smaller context, faster
```

### Out of memory
```bash
llamacpp serve --preset chat --no-mlock  # Disable memory lock
```

### Requests timing out
```bash
llamacpp serve --startup-timeout 300  # Increase from 120s to 300s
```

## Next Steps

- Read [CPU_OPTIMIZATION.md](CPU_OPTIMIZATION.md) for detailed tuning guide
- Check model-specific context limits
- Experiment with different presets for your workload
