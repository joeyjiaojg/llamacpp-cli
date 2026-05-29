# Changes Summary: Preset Defaults and NUMA-Aware Parallel Slots

## Overview
Fixed preset defaults and implemented auto-detection of parallel slots based on NUMA nodes for `llamacpp-cli`.

## Changes Made

### 1. Changed Default Preset to `max-context`

**File: `src/llamacpp_cli/cli.py`**

- Line 69: Changed default from `'code'` to `'max-context'`
- Line 70: Updated help text to reflect new default
- Lines 139-154: Updated docstring to document new default and NUMA-aware parallel behavior

**Before:**
```python
default='code',
help="Optimization preset (code=16K ctx, chat=8K ctx, fast=4K ctx, max-context=32K ctx)."
```

**After:**
```python
default='max-context',
help="Optimization preset (max-context=32K ctx [default], code=16K, chat=8K, fast=4K)."
```

### 2. Implemented NUMA-Aware Parallel Slot Detection

**File: `src/llamacpp_cli/utils.py`**

- Line 64: Changed function default parameter from `'code'` to `'max-context'`
- Lines 70-77: Added NUMA topology detection to auto-detect number of slots
- Lines 80, 85, 90: Updated `code`, `chat`, `fast` presets to ensure `parallel >= num_slots`
- Line 96: Changed `max-context` preset to use `num_slots` instead of hardcoded `1`
- Line 102: Updated fallback from `presets['code']` to `presets['max-context']`

**Key Logic:**
```python
# Detect NUMA nodes for slot-based parallelism
try:
    topology = detect_numa_topology()
    num_slots = len(topology["numa_nodes"])
except Exception:
    num_slots = 1  # Fallback to single slot
```

**Preset Behavior (dual-socket system with 2 NUMA nodes):**
- `max-context`: `parallel = 2` (one per NUMA node)
- `code`: `parallel >= 2` (min 2 for dual-socket, capped at 4)
- `chat`: `parallel >= 2` (min 2 for dual-socket, capped at 6)
- `fast`: `parallel >= 2` (min 2 for dual-socket, capped at 8)

### 3. Comprehensive Test Coverage

**File: `tests/test_utils.py`**

Added 7 new tests covering:
- ✅ Default preset is `max-context` (changed from `code`)
- ✅ Invalid preset falls back to `max-context` (changed from `code`)
- ✅ NUMA-aware parallel detection for dual-socket systems (2 slots)
- ✅ NUMA-aware parallel detection for quad-socket systems (4 slots)
- ✅ NUMA-aware parallel detection for single-socket systems (1 slot)
- ✅ Fallback to 1 slot when NUMA detection fails
- ✅ Preset caps are still respected with NUMA awareness

## Verification

All 196 tests pass:
```bash
$ python -m pytest tests/ -v
============================== 196 passed ==============================
```

## Backward Compatibility

✅ Users can still specify `--preset code` if they prefer the old default
✅ All preset configurations still work as before
✅ The `--parallel` flag can override auto-detected values
✅ The `--socket-id` flag works alongside preset configuration

## Key Benefits

1. **Better Default**: `max-context` preset is now default, supporting 32K context windows
2. **NUMA Parallelism**: Automatically detects and utilizes all NUMA nodes/slots
3. **Dual-Socket Optimized**: On dual-socket systems, `max-context` now uses both sockets (2 parallel) instead of just 1
4. **Graceful Fallback**: Falls back to 1 slot if NUMA detection fails (VMs, single-socket)
5. **Maintains Caps**: Other presets still respect their concurrency caps while ensuring minimum NUMA coverage

## Example Usage

```bash
# Start with default max-context preset (auto-detects 2 slots on dual-socket)
llamacpp serve --model qwen3.5

# Use code preset (16K context, 2-4 parallel depending on NUMA)
llamacpp serve --preset code --model qwen3:14b

# Override parallel setting explicitly
llamacpp serve --preset max-context --parallel 4

# Bind to specific NUMA node
llamacpp serve --preset max-context --socket-id 1
```
