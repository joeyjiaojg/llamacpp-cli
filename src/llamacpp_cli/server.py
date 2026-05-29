"""Server management — start/stop llama.cpp server as a subprocess."""

import asyncio
import os
import subprocess

import httpx

from .config import find_llama_binary
from .cpu_topology import detect_numa_topology
from .db import get_model
from .run import _is_local_path


def _detect_cpu_topology() -> tuple[int, bool]:
    """Return (cores_per_numa_node, has_multiple_numa_nodes).

    Uses the cpu_topology module to detect NUMA configuration.
    Falls back to safe defaults on errors.
    """
    try:
        topology = detect_numa_topology()

        if not topology["has_numa"]:
            # Single NUMA node - use all CPUs
            total_cpus = os.cpu_count() or 2
            return total_cpus, False

        # Multiple NUMA nodes - calculate cores per node
        first_node = topology["numa_nodes"][0]
        cores_per_node = len(topology["cpus_per_node"][first_node])
        return cores_per_node, True
    except Exception:
        # Fallback: assume half the cores per socket
        total = os.cpu_count() or 2
        return max(1, total // 2), True


def _has_flag(args: list[str], *flags: str) -> bool:
    """Return True if any of *flags* appear in *args*.

    Handles both '--flag' and '--flag=value' formats.
    """
    for arg in args:
        for flag in flags:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def build_server_cmd(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    ctx_size: int | None = None,
    extra_args: list[str] | None = None,
    socket_id: int = 0,
) -> list[str]:
    """Build the llama-server command for a specific model path.

    Auto-applies CPU-optimal flags (--threads, --threads-batch, --numa, --no-mmap)
    unless the caller already supplied them via extra_args.

    On multi-socket systems, wraps the command with numactl for explicit NUMA node
    binding to avoid cross-socket memory access overhead (4x slower).

    Args:
        model_path: Path to the GGUF model file
        host: Host address to bind to
        port: Port number to listen on
        ctx_size: Context size override (optional)
        extra_args: Additional command-line arguments
        socket_id: NUMA socket/node to bind to (default: 0)

    Returns:
        Command list ready for subprocess.Popen
    """
    binary = find_llama_binary("llama-server")
    extra = list(extra_args) if extra_args else []

    # Detect if user already specified NUMA binding via numactl flags
    has_numa_binding = _has_flag(extra, "--cpunodebind", "--membind")

    # Auto-tune for CPU topology if the user hasn't overridden threading.
    cores_per_node, multi_node = _detect_cpu_topology()
    use_explicit_numa = (
        multi_node and not _has_flag(extra, "--threads", "-t") and not has_numa_binding
    )

    # Build the base server command
    cmd = [binary, "--host", host, "--port", str(port), "--model", model_path]

    # Apply context size override if specified and not already in extra_args
    if ctx_size is not None and not _has_flag(extra, "--ctx-size", "-c"):
        cmd += ["--ctx-size", str(ctx_size)]

    if not _has_flag(extra, "--threads", "-t"):
        cmd += ["--threads", str(cores_per_node)]
        cmd += ["--threads-batch", str(cores_per_node)]

    if not _has_flag(extra, "--no-mmap", "--mmap"):
        cmd += ["--no-mmap"]

    cmd.extend(extra)

    # Wrap with numactl for explicit NUMA binding on multi-socket systems
    if use_explicit_numa:
        return [
            "numactl",
            f"--cpunodebind={socket_id}",
            f"--membind={socket_id}",
            "--",
        ] + cmd

    return cmd


async def wait_until_ready(url: str, timeout: float = 120.0) -> None:
    """Poll url until HTTP 200 is returned or timeout is exceeded."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"llama-server not ready at {url} after {timeout}s")


def start_server(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
    socket_id: int = 0,
) -> subprocess.Popen | None:
    """Start the llama.cpp server as a subprocess.

    Args:
        model: Model name or path to load
        host: Host address to bind to
        port: Port number to listen on
        extra_args: Additional command-line arguments
        socket_id: NUMA socket/node to bind to (default: 0)

    Returns:
        The Popen object so the caller can manage the process.
    """
    # Resolve model path
    model_path = None
    if model:
        model_info = get_model(model)
        if model_info:
            model_path = model_info["path"]
        elif _is_local_path(model):
            model_path = model
        else:
            from .model_manager import pull_model

            pull_model(model)
            model_info = get_model(model)
            if model_info:
                model_path = model_info["path"]
            else:
                print(f"Failed to pull model '{model}'.")
                return None

    print(f"Starting llama-server on {host}:{port}...")
    if model:
        print(f"  Model: {model}")

    # Build command using build_server_cmd if we have a model, otherwise basic command
    if model_path:
        cmd = build_server_cmd(
            model_path=model_path,
            host=host,
            port=port,
            extra_args=extra_args,
            socket_id=socket_id,
        )
    else:
        # No model specified - basic server command
        binary = find_llama_binary("llama-server")
        cmd = [binary, "--host", host, "--port", str(port)]
        if extra_args:
            cmd.extend(extra_args)

    proc = subprocess.Popen(cmd)
    return proc


def run_server_foreground(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
    socket_id: int = 0,
) -> None:
    """Start the server in the foreground (blocking). Handles Ctrl+C gracefully.

    Args:
        model: Model name or path to load
        host: Host address to bind to
        port: Port number to listen on
        extra_args: Additional command-line arguments
        socket_id: NUMA socket/node to bind to (default: 0)
    """
    proc = start_server(
        model=model, host=host, port=port, extra_args=extra_args, socket_id=socket_id
    )
    if proc is None:
        return

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server stopped.")
