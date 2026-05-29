"""Multi-model server - serves multiple models on different ports."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import httpx

from .cpu_topology import detect_numa_topology
from .db import get_model as db_get_model
from .server import build_server_cmd


@dataclass
class ModelInstance:
    """A single model served on a dedicated port."""

    model: str
    port: int
    socket_id: int = 0
    process: subprocess.Popen | None = None
    loaded: bool = False


@dataclass
class MultiModelServer:
    """Manages multiple llama-server instances, one per model."""

    base_port: int = 8000
    instances: list[ModelInstance] = field(default_factory=list)

    def add_model(self, model: str, port: int | None = None, socket_id: int = 0) -> ModelInstance:
        """Add a model to serve.

        Args:
            model: Model name or path to load.
            port: Port to bind; auto-assigned from base_port if None.
            socket_id: NUMA node to bind to (overridden by start_all NUMA logic).

        Returns:
            The new ModelInstance (not yet started).
        """
        if port is None:
            port = self.base_port + len(self.instances)
        instance = ModelInstance(model=model, port=port, socket_id=socket_id)
        self.instances.append(instance)
        return instance

    def start_all(self, extra_args: list[str] | None = None) -> None:
        """Start all model instances.

        Assigns NUMA nodes round-robin across instances when NUMA topology is
        available.  Each instance is launched with build_server_cmd so it
        inherits the same CPU-optimisation flags used by the regular ``serve``
        command.

        Args:
            extra_args: Extra arguments forwarded verbatim to every llama-server.
        """
        topology = detect_numa_topology()
        numa_nodes = topology["numa_nodes"] if topology["has_numa"] else []

        for i, instance in enumerate(self.instances):
            # Assign NUMA nodes round-robin when available
            if numa_nodes:
                instance.socket_id = numa_nodes[i % len(numa_nodes)]

            model_info = db_get_model(instance.model)
            if not model_info:
                print(f"Model {instance.model} not found, skipping")
                continue

            cmd = build_server_cmd(
                model_path=model_info["path"],
                port=instance.port,
                socket_id=instance.socket_id,
                extra_args=extra_args,
            )

            print(f"Starting {instance.model} on port {instance.port} (NUMA {instance.socket_id})")
            instance.process = subprocess.Popen(cmd)

        # Wait for every launched instance to become ready
        for instance in self.instances:
            if instance.process is None:
                continue
            try:
                self._wait_ready(instance.port)
                instance.loaded = True
                print(f"  {instance.model} ready on port {instance.port}")
            except TimeoutError as exc:
                print(f"  WARNING: {exc}")

    def _wait_ready(self, port: int, timeout: float = 120.0) -> None:
        """Poll the /health endpoint until HTTP 200 or timeout.

        Args:
            port: Port of the llama-server instance.
            timeout: Maximum seconds to wait.

        Raises:
            TimeoutError: If the server is not ready within *timeout* seconds.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Model on port {port} not ready after {timeout}s")

    def stop_all(self) -> None:
        """Terminate all running instances gracefully, then force-kill stragglers."""
        for instance in self.instances:
            if instance.process is None:
                continue
            try:
                instance.process.terminate()
                instance.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                instance.process.kill()
                instance.process.wait()
            finally:
                instance.loaded = False

    def get_model_urls(self) -> dict[str, str]:
        """Return mapping of model name -> base URL for every loaded instance.

        Only instances where ``loaded is True`` are included.
        """
        return {
            inst.model: f"http://127.0.0.1:{inst.port}"
            for inst in self.instances
            if inst.loaded
        }
