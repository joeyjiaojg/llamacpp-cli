"""Tests for slot-based backend management."""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from llamacpp_cli.slot_manager import Slot, SlotManager


@pytest.fixture
def mock_topology():
    """Mock NUMA topology with 2 nodes."""
    return {
        "num_sockets": 2,
        "numa_nodes": [0, 1],
        "cpus_per_node": {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        "has_numa": True,
    }


@pytest.fixture
def mock_single_node_topology():
    """Mock single-node topology."""
    return {
        "num_sockets": 1,
        "numa_nodes": [0],
        "cpus_per_node": {0: [0, 1, 2, 3]},
        "has_numa": False,
    }


class TestSlot:
    """Tests for Slot class."""

    def test_slot_initialization(self):
        """Test slot initializes with correct attributes."""
        slot = Slot(id=0, socket_id=0, port=8000)

        assert slot.id == 0
        assert slot.socket_id == 0
        assert slot.port == 8000
        assert slot.model is None
        assert slot.process is None
        assert slot.busy is False

    def test_is_loaded_no_model(self):
        """Test is_loaded returns False when no model loaded."""
        slot = Slot(id=0, socket_id=0, port=8000)
        assert not slot.is_loaded("any-model")

    def test_is_loaded_with_model(self):
        """Test is_loaded returns True when model is loaded and process alive."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.model = "test-model"
        slot.process = MagicMock()
        slot.process.poll.return_value = None  # Process alive

        assert slot.is_loaded("test-model")
        assert not slot.is_loaded("other-model")

    def test_is_alive_no_process(self):
        """Test _is_alive returns False when no process."""
        slot = Slot(id=0, socket_id=0, port=8000)
        assert not slot._is_alive()

    def test_is_alive_with_process(self):
        """Test _is_alive checks process status."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.process = MagicMock()

        slot.process.poll.return_value = None
        assert slot._is_alive()

        slot.process.poll.return_value = 0
        assert not slot._is_alive()

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    @patch("llamacpp_cli.slot_manager.build_server_cmd")
    @patch("llamacpp_cli.slot_manager.subprocess.Popen")
    @patch("llamacpp_cli.slot_manager.httpx.Client")
    @patch("llamacpp_cli.slot_manager.Path")
    def test_load_success(
        self, mock_path, mock_client, mock_popen, mock_build_cmd, mock_topology_fn
    ):
        """Test successful model loading."""
        mock_topology_fn.return_value = {
            "has_numa": False,
            "cpus_per_node": {},
        }

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_build_cmd.return_value = ["llama-server", "--model", "test.gguf"]

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance

        slot = Slot(id=0, socket_id=0, port=8000)
        slot.load("test-model", "/path/to/test.gguf")

        assert slot.model == "test-model"
        assert slot.model_path == "/path/to/test.gguf"
        assert slot.process == mock_process
        mock_popen.assert_called_once()

    @patch("llamacpp_cli.slot_manager.Path")
    def test_load_model_not_found(self, mock_path):
        """Test load raises FileNotFoundError when model doesn't exist."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        slot = Slot(id=0, socket_id=0, port=8000)

        with pytest.raises(FileNotFoundError):
            slot.load("test-model", "/nonexistent/model.gguf")

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_build_numa_args_no_numa(self, mock_topology_fn):
        """Test _build_numa_args returns empty list for single-node system."""
        mock_topology_fn.return_value = {
            "has_numa": False,
            "cpus_per_node": {},
        }

        slot = Slot(id=0, socket_id=0, port=8000)
        args = slot._build_numa_args()

        assert args == []

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_build_numa_args_with_numa(self, mock_topology_fn):
        """Test _build_numa_args builds CPU mask for NUMA systems."""
        mock_topology_fn.return_value = {
            "has_numa": True,
            "cpus_per_node": {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        }

        slot = Slot(id=0, socket_id=0, port=8000)
        args = slot._build_numa_args()

        assert args == ["--cpu-mask", "0,1,2,3"]

        slot2 = Slot(id=1, socket_id=1, port=8001)
        args2 = slot2._build_numa_args()

        assert args2 == ["--cpu-mask", "4,5,6,7"]

    def test_unload_no_process(self):
        """Test unload handles case with no process."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.model = "test-model"

        slot.unload()

        assert slot.model is None
        assert slot.process is None

    def test_unload_with_process(self):
        """Test unload terminates process gracefully."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.model = "test-model"
        mock_process = MagicMock()
        slot.process = mock_process

        slot.unload()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert slot.model is None
        assert slot.process is None

    def test_unload_force_kill(self):
        """Test unload kills process if terminate times out."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.model = "test-model"
        mock_process = MagicMock()
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 5),
            None,
        ]
        slot.process = mock_process

        slot.unload()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_get_status(self):
        """Test get_status returns correct information."""
        slot = Slot(id=0, socket_id=0, port=8000)
        slot.model = "test-model"
        slot.busy = True
        slot.process = MagicMock()
        slot.process.poll.return_value = None

        status = slot.get_status()

        assert status["id"] == 0
        assert status["socket_id"] == 0
        assert status["port"] == 8000
        assert status["model"] == "test-model"
        assert status["loaded"] is True
        assert status["busy"] is True
        assert status["alive"] is True


class TestSlotManager:
    """Tests for SlotManager class."""

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_initialization_single_node(self, mock_topology_fn):
        """Test manager initializes with single node."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0],
            "has_numa": False,
        }

        manager = SlotManager(base_port=8000)

        assert len(manager.slots) == 1
        assert manager.slots[0].id == 0
        assert manager.slots[0].socket_id == 0
        assert manager.slots[0].port == 8000

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_initialization_dual_socket(self, mock_topology_fn):
        """Test manager initializes with dual sockets."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        assert len(manager.slots) == 2
        assert manager.slots[0].port == 8000
        assert manager.slots[1].port == 8001

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_select_slot_tier1_loaded(self, mock_topology_fn):
        """Test select_slot prefers slots with model already loaded (Tier 1)."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].model = "test-model"
        manager.slots[0].process = MagicMock()
        manager.slots[0].process.poll.return_value = None
        manager.slots[0].busy = False

        manager.slots[1].model = None
        manager.slots[1].busy = False

        slot = manager.select_slot("test-model")

        assert slot == manager.slots[0]

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_select_slot_tier2_idle(self, mock_topology_fn):
        """Test select_slot uses idle slot when model not loaded (Tier 2)."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].model = "other-model"
        manager.slots[0].busy = False

        manager.slots[1].model = None
        manager.slots[1].busy = False

        slot = manager.select_slot("test-model")

        assert slot == manager.slots[1]

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_select_slot_tier3_any_available(self, mock_topology_fn):
        """Test select_slot uses any available slot (Tier 3)."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].model = "model-a"
        manager.slots[0].busy = False

        manager.slots[1].model = "model-b"
        manager.slots[1].busy = False

        slot = manager.select_slot("test-model")

        assert slot in manager.slots

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_select_slot_all_busy(self, mock_topology_fn):
        """Test select_slot returns None when all slots busy."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].busy = True
        manager.slots[1].busy = True

        slot = manager.select_slot("test-model")

        assert slot is None

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_get_slot(self, mock_topology_fn):
        """Test get_slot retrieves by ID."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        slot = manager.get_slot(0)
        assert slot == manager.slots[0]

        slot = manager.get_slot(1)
        assert slot == manager.slots[1]

        slot = manager.get_slot(99)
        assert slot is None

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_get_slot_by_port(self, mock_topology_fn):
        """Test get_slot_by_port retrieves by port number."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        slot = manager.get_slot_by_port(8000)
        assert slot == manager.slots[0]

        slot = manager.get_slot_by_port(8001)
        assert slot == manager.slots[1]

        slot = manager.get_slot_by_port(9999)
        assert slot is None

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_available_slots(self, mock_topology_fn):
        """Test available_slots returns only non-busy slots."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].busy = True
        manager.slots[1].busy = False

        available = manager.available_slots()

        assert len(available) == 1
        assert available[0] == manager.slots[1]

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_loaded_slots(self, mock_topology_fn):
        """Test loaded_slots returns only slots with models."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        manager.slots[0].model = "test-model"
        manager.slots[0].process = MagicMock()
        manager.slots[0].process.poll.return_value = None

        manager.slots[1].model = None

        loaded = manager.loaded_slots()

        assert len(loaded) == 1
        assert loaded[0] == manager.slots[0]

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_get_all_status(self, mock_topology_fn):
        """Test get_all_status returns status for all slots."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        statuses = manager.get_all_status()

        assert len(statuses) == 2
        assert statuses[0]["id"] == 0
        assert statuses[1]["id"] == 1

    @patch("llamacpp_cli.slot_manager.detect_numa_topology")
    def test_shutdown(self, mock_topology_fn):
        """Test shutdown unloads all slots."""
        mock_topology_fn.return_value = {
            "numa_nodes": [0, 1],
            "has_numa": True,
        }

        manager = SlotManager(base_port=8000)

        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        manager.slots[0].process = mock_proc1
        manager.slots[1].process = mock_proc2

        manager.shutdown()

        mock_proc1.terminate.assert_called_once()
        mock_proc2.terminate.assert_called_once()

    @patch("llamacpp_cli.slot_manager.get_model")
    @patch("llamacpp_cli.slot_manager._is_local_path")
    @patch("llamacpp_cli.slot_manager.Path")
    def test_resolve_model_path_local(self, mock_path, mock_is_local, mock_get_model):
        """Test resolve_model_path with local file."""
        mock_is_local.return_value = True
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.__str__ = lambda self: "/resolved/path/model.gguf"
        mock_path.return_value.expanduser.return_value.resolve.return_value = mock_path_obj

        manager = SlotManager(base_port=8000)
        name, path = manager.resolve_model_path("/path/to/model.gguf")

        assert name == "/path/to/model.gguf"
        assert path == "/resolved/path/model.gguf"

    @patch("llamacpp_cli.slot_manager.get_model")
    @patch("llamacpp_cli.slot_manager._is_local_path")
    def test_resolve_model_path_from_db(self, mock_is_local, mock_get_model):
        """Test resolve_model_path with registered model."""
        mock_is_local.return_value = False
        mock_get_model.return_value = {
            "name": "test-model",
            "path": "/home/user/.llamacpp/models/test-model.gguf",
        }

        manager = SlotManager(base_port=8000)
        name, path = manager.resolve_model_path("test-model")

        assert name == "test-model"
        assert path == "/home/user/.llamacpp/models/test-model.gguf"

    @patch("llamacpp_cli.slot_manager.model_manager")
    @patch("llamacpp_cli.slot_manager.get_model")
    @patch("llamacpp_cli.slot_manager._is_local_path")
    def test_resolve_model_path_pull_on_demand(
        self, mock_is_local, mock_get_model, mock_model_manager
    ):
        """Test resolve_model_path pulls model if not found."""
        mock_is_local.return_value = False
        mock_get_model.side_effect = [
            None,
            {
                "name": "test-model",
                "path": "/home/user/.llamacpp/models/test-model.gguf",
            },
        ]

        manager = SlotManager(base_port=8000)
        name, path = manager.resolve_model_path("test-model")

        mock_model_manager.pull_model.assert_called_once_with("test-model")
        assert name == "test-model"
        assert path == "/home/user/.llamacpp/models/test-model.gguf"

    @patch("llamacpp_cli.slot_manager.model_manager")
    @patch("llamacpp_cli.slot_manager.get_model")
    @patch("llamacpp_cli.slot_manager._is_local_path")
    def test_resolve_model_path_pull_fails(
        self, mock_is_local, mock_get_model, mock_model_manager
    ):
        """Test resolve_model_path raises if pull fails."""
        mock_is_local.return_value = False
        mock_get_model.return_value = None

        manager = SlotManager(base_port=8000)

        with pytest.raises(FileNotFoundError):
            manager.resolve_model_path("nonexistent-model")
