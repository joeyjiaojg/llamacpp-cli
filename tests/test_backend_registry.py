"""Tests for backend registry."""

import pytest

from llamacpp_cli.backend_registry import BackendRegistry, SlotInfo


class TestSlotInfo:
    """Tests for SlotInfo dataclass."""

    def test_initialization(self):
        """Test SlotInfo initializes correctly."""
        slot = SlotInfo(slot_id=0, socket_id=0, port=8000)

        assert slot.slot_id == 0
        assert slot.socket_id == 0
        assert slot.port == 8000
        assert slot.model is None
        assert slot.busy is False

    def test_initialization_with_model(self):
        """Test SlotInfo initializes with model."""
        slot = SlotInfo(
            slot_id=0, socket_id=0, port=8000, model="test-model", busy=True
        )

        assert slot.model == "test-model"
        assert slot.busy is True


class TestBackendRegistry:
    """Tests for BackendRegistry class."""

    def test_initialization(self):
        """Test registry initializes empty."""
        registry = BackendRegistry()
        assert len(registry.backends) == 0

    def test_register_backend(self):
        """Test registering a backend with slots."""
        registry = BackendRegistry()

        slots_data = [
            {"id": 0, "socket_id": 0, "port": 8000},
            {"id": 1, "socket_id": 1, "port": 8001},
        ]

        registry.register_backend("http://host1:7000", slots_data)

        assert len(registry.backends) == 1
        assert "http://host1:7000" in registry.backends
        assert len(registry.backends["http://host1:7000"]) == 2

        slots = registry.backends["http://host1:7000"]
        assert slots[0].slot_id == 0
        assert slots[0].port == 8000
        assert slots[1].slot_id == 1
        assert slots[1].port == 8001

    def test_register_backend_with_models(self):
        """Test registering backend with models loaded."""
        registry = BackendRegistry()

        slots_data = [
            {"id": 0, "socket_id": 0, "port": 8000, "model": "llama-3", "busy": True},
            {"id": 1, "socket_id": 1, "port": 8001, "model": None, "busy": False},
        ]

        registry.register_backend("http://host1:7000", slots_data)

        slots = registry.backends["http://host1:7000"]
        assert slots[0].model == "llama-3"
        assert slots[0].busy is True
        assert slots[1].model is None
        assert slots[1].busy is False

    def test_update_slot_status(self):
        """Test updating slot busy status."""
        registry = BackendRegistry()

        slots_data = [
            {"id": 0, "socket_id": 0, "port": 8000, "busy": False},
            {"id": 1, "socket_id": 1, "port": 8001, "busy": False},
        ]

        registry.register_backend("http://host1:7000", slots_data)

        registry.update_slot_status("http://host1:7000", slot_id=0, busy=True)

        slots = registry.backends["http://host1:7000"]
        assert slots[0].busy is True
        assert slots[1].busy is False

    def test_update_slot_status_unknown_backend(self):
        """Test update_slot_status handles unknown backend gracefully."""
        registry = BackendRegistry()

        registry.update_slot_status("http://unknown:7000", slot_id=0, busy=True)

    def test_find_slot_for_model_tier1_affinity(self):
        """Test find_slot_for_model prefers slot with model loaded (Tier 1)."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "llama-3"},
                {"id": 1, "socket_id": 1, "port": 8001, "model": None},
            ],
        )

        result = registry.find_slot_for_model("llama-3")

        assert result is not None
        backend_url, slot = result
        assert backend_url == "http://host1:7000"
        assert slot.slot_id == 0
        assert slot.model == "llama-3"

    def test_find_slot_for_model_tier2_idle(self):
        """Test find_slot_for_model uses idle slot (Tier 2)."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "other-model"},
                {"id": 1, "socket_id": 1, "port": 8001, "model": None},
            ],
        )

        result = registry.find_slot_for_model("llama-3")

        assert result is not None
        backend_url, slot = result
        assert slot.slot_id == 1
        assert slot.model is None

    def test_find_slot_for_model_tier3_any_available(self):
        """Test find_slot_for_model uses any available slot (Tier 3)."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "model-a"},
                {"id": 1, "socket_id": 1, "port": 8001, "model": "model-b"},
            ],
        )

        result = registry.find_slot_for_model("llama-3")

        assert result is not None
        backend_url, slot = result
        assert slot.model in ["model-a", "model-b"]

    def test_find_slot_for_model_all_busy(self):
        """Test find_slot_for_model returns None when all busy."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {
                    "id": 0,
                    "socket_id": 0,
                    "port": 8000,
                    "model": "llama-3",
                    "busy": True,
                },
                {
                    "id": 1,
                    "socket_id": 1,
                    "port": 8001,
                    "model": None,
                    "busy": True,
                },
            ],
        )

        result = registry.find_slot_for_model("llama-3")

        assert result is None

    def test_find_slot_for_model_multiple_backends(self):
        """Test find_slot_for_model searches across multiple backends."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {
                    "id": 0,
                    "socket_id": 0,
                    "port": 8000,
                    "model": "other-model",
                    "busy": True,
                },
            ],
        )

        registry.register_backend(
            "http://host2:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "llama-3"},
            ],
        )

        result = registry.find_slot_for_model("llama-3")

        assert result is not None
        backend_url, slot = result
        assert backend_url == "http://host2:7000"
        assert slot.model == "llama-3"

    def test_get_available_slots(self):
        """Test get_available_slots returns only non-busy slots."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "busy": True},
                {"id": 1, "socket_id": 1, "port": 8001, "busy": False},
            ],
        )

        registry.register_backend(
            "http://host2:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "busy": False},
            ],
        )

        available = registry.get_available_slots()

        assert len(available) == 2
        assert all(not slot.busy for _, slot in available)

    def test_get_backend_slots(self):
        """Test get_backend_slots returns slots for specific backend."""
        registry = BackendRegistry()

        slots_data = [
            {"id": 0, "socket_id": 0, "port": 8000},
            {"id": 1, "socket_id": 1, "port": 8001},
        ]

        registry.register_backend("http://host1:7000", slots_data)

        slots = registry.get_backend_slots("http://host1:7000")

        assert len(slots) == 2
        assert slots[0].slot_id == 0
        assert slots[1].slot_id == 1

    def test_get_backend_slots_unknown(self):
        """Test get_backend_slots returns empty list for unknown backend."""
        registry = BackendRegistry()

        slots = registry.get_backend_slots("http://unknown:7000")

        assert slots == []

    def test_remove_backend(self):
        """Test removing a backend."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [{"id": 0, "socket_id": 0, "port": 8000}],
        )

        assert len(registry.backends) == 1

        registry.remove_backend("http://host1:7000")

        assert len(registry.backends) == 0

    def test_remove_backend_unknown(self):
        """Test removing unknown backend doesn't error."""
        registry = BackendRegistry()

        registry.remove_backend("http://unknown:7000")

    def test_get_stats_empty(self):
        """Test get_stats with no backends."""
        registry = BackendRegistry()

        stats = registry.get_stats()

        assert stats["total_backends"] == 0
        assert stats["total_slots"] == 0
        assert stats["busy_slots"] == 0
        assert stats["available_slots"] == 0
        assert stats["loaded_slots"] == 0

    def test_get_stats_with_backends(self):
        """Test get_stats calculates correctly."""
        registry = BackendRegistry()

        registry.register_backend(
            "http://host1:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "llama-3", "busy": True},
                {"id": 1, "socket_id": 1, "port": 8001, "model": None, "busy": False},
            ],
        )

        registry.register_backend(
            "http://host2:7000",
            [
                {"id": 0, "socket_id": 0, "port": 8000, "model": "qwen", "busy": False},
            ],
        )

        stats = registry.get_stats()

        assert stats["total_backends"] == 2
        assert stats["total_slots"] == 3
        assert stats["busy_slots"] == 1
        assert stats["available_slots"] == 2
        assert stats["loaded_slots"] == 2
