"""Slot-based backend management for NUMA-aware serving."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import model_manager
from .cpu_topology import detect_numa_topology
from .db import get_model
from .run import _is_local_path
from .server import build_server_cmd


@dataclass
class Slot:
    """A single inference slot bound to a NUMA node."""

    id: int
    socket_id: int  # NUMA node ID
    port: int
    model: str | None = None
    model_path: str | None = None
    process: subprocess.Popen | None = None
    busy: bool = False

    def is_loaded(self, model: str) -> bool:
        """Check if this slot has the specified model loaded."""
        return self.model == model and self.process is not None and self._is_alive()

    def _is_alive(self) -> bool:
        """Check if process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def load(
        self,
        model: str,
        model_path: str,
        ctx_size: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Load a model on this slot.

        Args:
            model: Model identifier
            model_path: Path to GGUF file
            ctx_size: Context size override
            extra_args: Additional llama-server arguments
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self.model and self.model != model:
            self.unload()

        if self.is_loaded(model):
            return

        cpu_args = self._build_numa_args()
        all_extra_args = (extra_args or []) + cpu_args

        cmd = build_server_cmd(
            model_path=model_path,
            host="127.0.0.1",
            port=self.port,
            ctx_size=ctx_size,
            extra_args=all_extra_args,
        )

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.model = model
        self.model_path = model_path

        try:
            self._wait_ready()
        except TimeoutError:
            self.unload()
            raise

    def _build_numa_args(self) -> list[str]:
        """Build NUMA binding arguments for this slot's socket."""
        topology = detect_numa_topology()

        if not topology["has_numa"]:
            return []

        cpus = topology["cpus_per_node"].get(self.socket_id, [])
        if not cpus:
            return []

        cpu_list = ",".join(str(c) for c in cpus)
        return ["--cpu-mask", cpu_list]

    def unload(self) -> None:
        """Unload model from this slot."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.model = None
        self.model_path = None

    def _wait_ready(self, timeout: float = 30.0) -> None:
        """Wait for server to be ready."""
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/health"

        while time.time() < deadline:
            try:
                with httpx.Client() as client:
                    resp = client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            time.sleep(0.1)

        raise TimeoutError(f"Slot {self.id} not ready after {timeout}s")

    def get_status(self) -> dict[str, Any]:
        """Get slot status information."""
        return {
            "id": self.id,
            "socket_id": self.socket_id,
            "port": self.port,
            "model": self.model,
            "loaded": self.is_loaded(self.model) if self.model else False,
            "busy": self.busy,
            "alive": self._is_alive(),
        }


class SlotManager:
    """Manages multiple slots across NUMA nodes."""

    def __init__(self, base_port: int = 8000):
        self.base_port = base_port
        self.slots: list[Slot] = []
        self._init_slots()

    def _init_slots(self) -> None:
        """Initialize slots based on NUMA topology."""
        topology = detect_numa_topology()

        for i, node_id in enumerate(topology["numa_nodes"]):
            slot = Slot(id=i, socket_id=node_id, port=self.base_port + i)
            self.slots.append(slot)

    def select_slot(self, model: str) -> Slot | None:
        """Select best slot for a model using 3-tier strategy.

        Tier 1: Slot with model already loaded (KV cache reuse)
        Tier 2: Idle slot (no model switch needed)
        Tier 3: Any available slot

        Args:
            model: Model identifier

        Returns:
            Best slot or None if all busy
        """
        tier1 = [s for s in self.slots if s.is_loaded(model) and not s.busy]
        tier2 = [s for s in self.slots if s.model is None and not s.busy]
        tier3 = [s for s in self.slots if not s.busy]

        candidates = tier1 or tier2 or tier3
        return candidates[0] if candidates else None

    def get_slot(self, slot_id: int) -> Slot | None:
        """Get slot by ID."""
        if 0 <= slot_id < len(self.slots):
            return self.slots[slot_id]
        return None

    def get_slot_by_port(self, port: int) -> Slot | None:
        """Get slot by port number."""
        for slot in self.slots:
            if slot.port == port:
                return slot
        return None

    def available_slots(self) -> list[Slot]:
        """Get list of available (not busy) slots."""
        return [s for s in self.slots if not s.busy]

    def loaded_slots(self) -> list[Slot]:
        """Get list of slots with models loaded."""
        return [s for s in self.slots if s.model is not None and s._is_alive()]

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get status of all slots."""
        return [s.get_status() for s in self.slots]

    def shutdown(self) -> None:
        """Shutdown all slots."""
        for slot in self.slots:
            slot.unload()

    def resolve_model_path(self, model: str) -> tuple[str, str]:
        """Resolve model identifier to (model_name, model_path).

        Args:
            model: Model name or path

        Returns:
            Tuple of (model_name, absolute_path)

        Raises:
            FileNotFoundError: If model not found
        """
        if _is_local_path(model):
            path = Path(model).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model}")
            return model, str(path)

        model_info = get_model(model)
        if model_info:
            return model, model_info["path"]

        print(f"Model '{model}' not found locally, pulling...")
        model_manager.pull_model(model)
        model_info = get_model(model)
        if model_info:
            return model, model_info["path"]

        raise FileNotFoundError(f"Failed to pull model '{model}'")
