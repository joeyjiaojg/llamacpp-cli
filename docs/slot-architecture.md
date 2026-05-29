# Slot-Based Backend Architecture

## Overview

The slot-based backend architecture enables NUMA-aware serving with dynamic model management. Each NUMA node gets its own independent inference slot, enabling full parallelism on multi-socket systems.

## Features

- **Automatic NUMA Detection**: Detects CPU topology and creates one slot per NUMA node
- **3-Tier Slot Selection**:
  - Tier 1: Slot with model already loaded (KV cache reuse)
  - Tier 2: Idle slot with no model (avoids model switch overhead)
  - Tier 3: Any available slot
- **Dynamic Model Loading**: Load/unload models on-demand
- **NUMA Binding**: Each slot binds to specific CPUs on its NUMA node
- **Graceful Shutdown**: Clean termination of all slot processes

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Management API                      │
│              (http://host:7000)                      │
└──────────────────────────────┬──────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │       SlotManager               │
              └────────────────┬────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
   ┌───▼────┐             ┌───▼────┐             ┌───▼────┐
   │ Slot 0 │             │ Slot 1 │             │ Slot N │
   │Socket 0│             │Socket 1│             │Socket N│
   │Port:8000│            │Port:8001│            │Port:800N│
   └────────┘             └────────┘             └────────┘
       │                       │                       │
   llama-server           llama-server           llama-server
   (NUMA node 0)          (NUMA node 1)          (NUMA node N)
```

## Components

### 1. `slot_manager.py`
Core slot management with NUMA topology detection.

**Key Classes:**
- `Slot`: Individual inference slot bound to NUMA node
- `SlotManager`: Manages multiple slots with 3-tier selection

**Methods:**
- `load(model, model_path)`: Load model on slot
- `unload()`: Unload model from slot
- `select_slot(model)`: Find best slot using 3-tier strategy
- `resolve_model_path(model)`: Resolve model name to filesystem path

### 2. `backend_registry.py`
Registry for tracking slot-aware backends (for lb-proxy integration).

**Key Classes:**
- `SlotInfo`: Metadata for a backend slot
- `BackendRegistry`: Registry tracking backends and their slots

**Methods:**
- `register_backend(url, slots)`: Register backend with its slots
- `find_slot_for_model(model)`: Find best slot across all backends
- `update_slot_status(url, slot_id, busy)`: Update busy state

### 3. `slot_serve.py`
FastAPI server for slot-based serving.

**Endpoints:**
- `GET /health`: Health check
- `GET /slots`: List all slots with status
- `POST /load`: Load model on best available slot
- `POST /unload/{slot_id}`: Unload specific slot
- `POST /v1/chat/completions`: OpenAI-compatible chat endpoint

## CLI Usage

### Start Slot Server

```bash
# Basic usage (auto-detect NUMA topology)
llamacpp slot-serve

# Pre-load model
llamacpp slot-serve --model qwen3.5

# Custom ports
llamacpp slot-serve --port 7000 --base-port 8000

# With context size
llamacpp slot-serve --model llama-3 --ctx-size 16384
```

### Management API

```bash
# List slots
curl http://localhost:7000/slots

# Load model
curl -X POST http://localhost:7000/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "ctx_size": 16384}'

# Unload slot
curl -X POST http://localhost:7000/unload/0

# Chat completion
curl -X POST http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## NUMA Topology Detection

The system automatically detects:
- Number of NUMA nodes (`/sys/devices/system/node/node*`)
- CPUs per node (`/sys/devices/system/node/node*/cpulist`)
- Physical CPU sockets (`/proc/cpuinfo` physical_id)

### Single-Node System
```
Slot 0: socket_id=0, port=8000, cpus=[0,1,2,3,4,5,6,7]
```

### Dual-Socket System
```
Slot 0: socket_id=0, port=8000, cpus=[0,1,2,3]
Slot 1: socket_id=1, port=8001, cpus=[4,5,6,7]
```

## Integration with LB-Proxy

The backend registry enables slot-aware load balancing:

```python
from llamacpp_cli.backend_registry import BackendRegistry

registry = BackendRegistry()

# Register backends
registry.register_backend("http://host1:7000", [
    {"id": 0, "socket_id": 0, "port": 8000, "model": "llama-3"},
    {"id": 1, "socket_id": 1, "port": 8001, "model": None},
])

# Find best slot for model
backend_url, slot = registry.find_slot_for_model("llama-3")
# Returns: ("http://host1:7000", SlotInfo(slot_id=0, ...))
```

## Performance Benefits

### NUMA Parallelism
- **Single-node**: 1x throughput
- **Dual-socket**: 2x throughput (each socket serves independently)
- **Quad-socket**: 4x throughput

### Model Affinity Routing
- **Tier 1 (loaded)**: 0ms model load time, full KV cache reuse
- **Tier 2 (idle)**: ~5-10s model load time
- **Tier 3 (any)**: ~5-10s model load + eviction overhead

### Memory Isolation
Each slot runs in its own process with dedicated memory, preventing:
- Cross-socket memory contention
- NUMA remote access penalties
- Model eviction cascades

## Testing

Run comprehensive tests:

```bash
# Slot manager tests
pytest tests/test_slot_manager.py -v

# Backend registry tests
pytest tests/test_backend_registry.py -v

# All tests
pytest tests/test_slot_manager.py tests/test_backend_registry.py -v
```

## Example: Dual-Socket Deployment

```bash
# Terminal 1: Start slot server on machine1
llamacpp slot-serve --host 0.0.0.0 --port 7000 --base-port 8000 \
  --model qwen3.5

# Machine has 2 NUMA nodes, creates:
#   Slot 0: socket 0, port 8000
#   Slot 1: socket 1, port 8001

# Terminal 2: Query via management API
curl http://machine1:7000/slots
# [
#   {"id": 0, "socket_id": 0, "port": 8000, "model": "qwen3.5", "loaded": true, "busy": false},
#   {"id": 1, "socket_id": 1, "port": 8001, "model": null, "loaded": false, "busy": false}
# ]

# Terminal 3: Chat completion (auto-routes to slot 0)
curl -X POST http://machine1:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Explain NUMA"}]
  }'
```

## Future Enhancements

- [ ] Streaming support for chat completions
- [ ] Health monitoring per slot
- [ ] Metrics export (Prometheus)
- [ ] Auto-scaling based on load
- [ ] Model pre-warming on idle slots
- [ ] LB-proxy full integration with slot discovery
