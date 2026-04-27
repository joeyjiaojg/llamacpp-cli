"""Server management — start/stop llama.cpp server as a subprocess."""

import asyncio
import os
import subprocess

import httpx

from .config import find_llama_binary
from .db import get_model
from .run import _is_local_path


def _detect_cpu_topology() -> tuple[int, bool]:
    """Return (cores_per_numa_node, has_multiple_numa_nodes).

    Reads /sys/devices/system/node/ to find NUMA nodes and their CPU lists,
    falling back to os.cpu_count() // 2 on any error (assumes dual-socket).
    """
    try:
        node_dirs = [
            e for e in os.scandir("/sys/devices/system/node/")
            if e.is_dir() and e.name.startswith("node")
        ]
        num_nodes = len(node_dirs)
        if num_nodes == 0:
            raise ValueError("no NUMA nodes found")

        # Count CPUs in node0 as representative.
        cpulist_path = f"/sys/devices/system/node/{node_dirs[0].name}/cpulist"
        with open(cpulist_path) as f:
            cpulist = f.read().strip()
        # cpulist is like "0,2,4-10,12" — count individual CPUs.
        count = 0
        for part in cpulist.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                count += int(hi) - int(lo) + 1
            else:
                count += 1
        return count, num_nodes > 1
    except Exception:
        total = os.cpu_count() or 2
        return max(1, total // 2), True


def _has_flag(args: list[str], *flags: str) -> bool:
    """Return True if any of *flags* appear in *args*."""
    return any(f in args for f in flags)


def build_server_cmd(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the llama-server command for a specific model path.

    Auto-applies CPU-optimal flags (--threads, --threads-batch, --numa, --no-mmap)
    unless the caller already supplied them via extra_args.
    """
    binary = find_llama_binary("llama-server")
    cmd = [binary, "--host", host, "--port", str(port), "--model", model_path]

    extra = list(extra_args) if extra_args else []

    # Auto-tune for CPU topology if the user hasn't overridden threading.
    if not _has_flag(extra, "--threads", "-t"):
        cores_per_node, multi_node = _detect_cpu_topology()
        cmd += ["--threads", str(cores_per_node)]
        cmd += ["--threads-batch", str(cores_per_node)]
        if multi_node and not _has_flag(extra, "--numa"):
            cmd += ["--numa", "numactl"]

    if not _has_flag(extra, "--no-mmap", "--mmap"):
        cmd += ["--no-mmap"]

    cmd.extend(extra)
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
) -> subprocess.Popen | None:
    """Start the llama.cpp server as a subprocess.

    Returns the Popen object so the caller can manage the process.
    """
    binary = find_llama_binary("llama-server")

    cmd = [binary, "--host", host, "--port", str(port)]

    if model:
        model_info = get_model(model)
        if model_info:
            cmd.extend(["--model", model_info["path"]])
        elif _is_local_path(model):
            cmd.extend(["--model", model])
        else:
            from .model_manager import pull_model
            pull_model(model)
            model_info = get_model(model)
            if model_info:
                cmd.extend(["--model", model_info["path"]])
            else:
                print(f"Failed to pull model '{model}'.")
                return None

    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting llama-server on {host}:{port}...")
    if model:
        print(f"  Model: {model}")

    proc = subprocess.Popen(cmd)
    return proc


def run_server_foreground(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
) -> None:
    """Start the server in the foreground (blocking). Handles Ctrl+C gracefully."""
    proc = start_server(model=model, host=host, port=port, extra_args=extra_args)
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
