# CPU Optimization Guide

This document explains the CPU-optimized server presets in `llamacpp serve`.

## Why CPU Optimization Matters

When running llama.cpp on CPU-only servers (no GPU), several parameters need careful tuning:

- **Context size** vs **inference speed**: Larger context = slower per-token generation
- **Parallel requests** vs **CPU contention**: Too many concurrent requests = context switching overhead
- **Memory locking**: Prevent model from being swapped to disk (critical for performance)
- **NUMA**: Optimize memory access on multi-socket servers

## Presets

The `--preset` flag provides pre-configured optimizations for common use cases:

### `code` (Default)
**Best for: Code tasks, multi-file analysis, repository work**

```bash
llamacpp serve --preset code
```

- **Context**: 16K tokens (handles multiple large files)
- **Parallel requests**: 2-4 (conservative, ensures fast per-request response)
- **Batch size**: 512
- **Use when**: Working with code, reviewing PRs, analyzing multiple files

### `chat`
**Best for: Conversational workloads, general Q&A**

```bash
llamacpp serve --preset chat
```

- **Context**: 8K tokens (adequate for conversations)
- **Parallel requests**: 4-6 (higher concurrency for chat scenarios)
- **Batch size**: 512
- **Use when**: Chat applications, customer support bots

### `fast`
**Best for: Quick queries, maximum throughput**

```bash
llamacpp serve --preset fast
```

- **Context**: 4K tokens (short responses)
- **Parallel requests**: 6-8 (maximum concurrency)
- **Batch size**: 256
- **Use when**: Simple queries, autocomplete, quick lookups

### `max-context`
**Best for: Large codebases, long documents**

```bash
llamacpp serve --preset max-context
```

- **Context**: 32K tokens (very large context)
- **Parallel requests**: 1 (single request at a time due to CPU load)
- **Batch size**: 512
- **⚠️ Warning**: Very slow on CPU, use only when necessary

## Manual Override

You can override any preset parameter:

```bash
# Start with code preset but increase context to 32K
llamacpp serve --preset code --ctx-size 32768 --parallel 1

# Start with chat preset but use more threads
llamacpp serve --preset chat --threads 16
```

## Auto-Detection

The following are auto-detected by default:

- **CPU threads**: Uses all available cores (`os.cpu_count()`)
- **NUMA**: Enables `--numa numactl` on multi-socket systems
- **Memory lock**: Always enabled (`--mlock`) to prevent swapping

## Network Access

By default, `llamacpp serve` binds to `0.0.0.0` (accepts connections from all interfaces) instead of `127.0.0.1` (localhost only). This is suitable for server deployments.

To restrict to localhost only:

```bash
llamacpp serve --host 127.0.0.1
```

## Example Configurations

### Development (local machine)
```bash
llamacpp serve --preset code --host 127.0.0.1
```

### Production server (code-focused)
```bash
llamacpp serve --preset code --model qwen3:14b
```

### High-concurrency chat server
```bash
llamacpp serve --preset chat --parallel 8
```

### Large repository analysis (single user)
```bash
llamacpp serve --preset max-context --model codellama:13b
```

## Performance Tips

1. **Monitor CPU usage**: If CPU is not saturated, increase `--parallel`
2. **Watch request latency**: If requests are timing out, decrease `--parallel` or `--ctx-size`
3. **Use appropriate models**: Smaller models (7B) run faster than larger ones (34B)
4. **Enable NUMA on multi-socket**: Significant speedup on dual-socket servers
5. **Memory lock is critical**: Never disable `--mlock` unless you run out of RAM

## Model-Specific Context Limits

Different models support different maximum context sizes:

| Model | Max Context | Recommended |
|-------|------------|-------------|
| Llama 3.1/3.2 | 128K | 32K (practical) |
| CodeLlama | 16K | 16K |
| DeepSeek Coder | 16K | 16K |
| Qwen | 32K | 16K |
| Llama 3 | 8K | 8K |

The presets are designed to work well across all models. For models with limited context (like Llama 3 at 8K), using `--preset max-context` (32K) will fail.

## Troubleshooting

### Server is slow
- Try `--preset fast` for smaller context
- Reduce `--parallel` to give more CPU per request
- Use a smaller model (7B instead of 13B/34B)

### Out of memory
- Reduce `--ctx-size`
- Use a smaller model
- Disable `--mlock` (will slow down, but reduce memory)

### Requests timing out
- Increase `--startup-timeout` (default: 120s)
- Reduce context size
- Check if model is too large for available RAM
