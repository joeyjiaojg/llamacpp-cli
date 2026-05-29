"""Backend registry for tracking slot-aware backends."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotInfo:
    """Information about a slot on a backend."""

    slot_id: int
    socket_id: int
    port: int
    model: str | None = None
    busy: bool = False


@dataclass
class BackendRegistry:
    """Registry tracking backends and their slots."""

    backends: dict[str, list[SlotInfo]] = field(default_factory=dict)

    def register_backend(self, backend_url: str, slots: list[dict]) -> None:
        """Register a backend and its slots.

        Args:
            backend_url: Base URL of the backend (e.g., "http://host:port")
            slots: List of slot info dictionaries
        """
        slot_infos = [
            SlotInfo(
                slot_id=s["id"],
                socket_id=s["socket_id"],
                port=s["port"],
                model=s.get("model"),
                busy=s.get("busy", False),
            )
            for s in slots
        ]
        self.backends[backend_url] = slot_infos

    def update_slot_status(self, backend_url: str, slot_id: int, busy: bool) -> None:
        """Update busy status for a specific slot.

        Args:
            backend_url: Backend URL
            slot_id: Slot ID
            busy: New busy status
        """
        if backend_url not in self.backends:
            return

        for slot in self.backends[backend_url]:
            if slot.slot_id == slot_id:
                slot.busy = busy
                break

    def find_slot_for_model(self, model: str) -> tuple[str, SlotInfo] | None:
        """Find best slot for a model across all backends.

        Uses 3-tier strategy:
        - Tier 1: Slot with model loaded (affinity routing)
        - Tier 2: Idle slot with no model
        - Tier 3: Any available slot

        Args:
            model: Model identifier

        Returns:
            Tuple of (backend_url, slot_info) or None if no slots available
        """
        tier1: list[tuple[str, SlotInfo]] = []
        tier2: list[tuple[str, SlotInfo]] = []
        tier3: list[tuple[str, SlotInfo]] = []

        for backend_url, slots in self.backends.items():
            for slot in slots:
                if slot.busy:
                    continue

                if slot.model == model:
                    tier1.append((backend_url, slot))
                elif slot.model is None:
                    tier2.append((backend_url, slot))
                else:
                    tier3.append((backend_url, slot))

        candidates = tier1 or tier2 or tier3
        return candidates[0] if candidates else None

    def get_available_slots(self) -> list[tuple[str, SlotInfo]]:
        """Get all available (not busy) slots across all backends."""
        available = []
        for backend_url, slots in self.backends.items():
            for slot in slots:
                if not slot.busy:
                    available.append((backend_url, slot))
        return available

    def get_backend_slots(self, backend_url: str) -> list[SlotInfo]:
        """Get all slots for a specific backend."""
        return self.backends.get(backend_url, [])

    def remove_backend(self, backend_url: str) -> None:
        """Remove a backend from the registry."""
        self.backends.pop(backend_url, None)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        total_slots = sum(len(slots) for slots in self.backends.values())
        busy_slots = sum(
            1 for slots in self.backends.values() for slot in slots if slot.busy
        )
        loaded_slots = sum(
            1 for slots in self.backends.values() for slot in slots if slot.model
        )

        return {
            "total_backends": len(self.backends),
            "total_slots": total_slots,
            "busy_slots": busy_slots,
            "available_slots": total_slots - busy_slots,
            "loaded_slots": loaded_slots,
        }
