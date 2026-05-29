"""Utility functions for system detection and optimization."""

import os
import subprocess


def detect_numa() -> bool:
    """Check if system has NUMA (multi-socket) architecture.

    Returns True if multiple NUMA nodes are detected.
    """
    try:
        # Check using lscpu
        result = subprocess.run(
            ['lscpu'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        output = result.stdout
        # Look for multiple NUMA nodes
        return 'NUMA node(s):' in output and 'NUMA node1' in output
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: check /sys/devices/system/node/
        try:
            node_dirs = [
                e for e in os.scandir("/sys/devices/system/node/")
                if e.is_dir() and e.name.startswith("node")
            ]
            return len(node_dirs) > 1
        except (FileNotFoundError, PermissionError):
            return False


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    return os.cpu_count() or 4


def get_model_max_context(model_name: str) -> int:
    """Get reasonable max context size based on model name.

    Args:
        model_name: Model name (may include 'codellama', 'llama-3.1', etc.)

    Returns:
        Recommended max context size in tokens
    """
    model_lower = model_name.lower()

    # Model-specific context sizes
    if 'codellama' in model_lower or 'code-llama' in model_lower:
        return 16384  # 16K - optimized for code
    elif 'llama-3.1' in model_lower or 'llama-3.2' in model_lower or 'llama3.1' in model_lower or 'llama3.2' in model_lower:
        return 32768  # Can handle 128K, but 32K more practical on CPU
    elif 'deepseek' in model_lower:
        return 16384  # 16K
    elif 'qwen' in model_lower:
        return 16384  # 16K typical
    else:
        return 8192  # Conservative default


def get_cpu_server_config(preset: str = 'max-context') -> dict:
    """Get optimal CPU-only server configuration based on preset.

    Args:
        preset: One of 'max-context', 'code', 'chat', 'fast'

    Returns:
        Dictionary with configuration parameters
    """
    from .cpu_topology import detect_numa_topology

    cpu_count = get_cpu_count()

    # Detect NUMA nodes for slot-based parallelism
    try:
        topology = detect_numa_topology()
        num_slots = len(topology["numa_nodes"])
    except Exception:
        num_slots = 1  # Fallback to single slot

    presets = {
        'code': {
            'ctx_size': 16384,      # 16K - handle multiple files
            'parallel': min(4, max(num_slots, cpu_count // 4)),  # At least num_slots
            'batch_size': 512,      # Moderate for CPU
            'mlock': True,          # Lock model in RAM
        },
        'chat': {
            'ctx_size': 8192,       # 8K - adequate for conversations
            'parallel': min(6, max(num_slots, cpu_count // 3)),  # At least num_slots
            'batch_size': 512,
            'mlock': True,
        },
        'fast': {
            'ctx_size': 4096,       # 4K - quick responses
            'parallel': min(8, max(num_slots, cpu_count // 2)),  # At least num_slots
            'batch_size': 256,      # Smaller batches
            'mlock': True,
        },
        'max-context': {
            'ctx_size': 32768,      # 32K - large codebases
            'parallel': num_slots,  # One request per slot (NUMA-aware)
            'batch_size': 512,
            'mlock': True,
        }
    }

    config = presets.get(preset, presets['max-context']).copy()

    # Add common settings
    config['threads'] = cpu_count
    config['numa'] = detect_numa()

    return config
