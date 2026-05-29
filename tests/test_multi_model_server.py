"""Tests for multi_model_server.py."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llamacpp_cli.multi_model_server import ModelInstance, MultiModelServer

# ---------------------------------------------------------------------------
# ModelInstance tests
# ---------------------------------------------------------------------------


class TestModelInstance:
    def test_defaults(self):
        inst = ModelInstance(model="llama3", port=8000)
        assert inst.model == "llama3"
        assert inst.port == 8000
        assert inst.socket_id == 0
        assert inst.process is None
        assert inst.loaded is False

    def test_custom_socket_id(self):
        inst = ModelInstance(model="qwen2", port=8001, socket_id=1)
        assert inst.socket_id == 1

    def test_loaded_flag_mutable(self):
        inst = ModelInstance(model="llama3", port=8000)
        inst.loaded = True
        assert inst.loaded is True


# ---------------------------------------------------------------------------
# MultiModelServer.add_model tests
# ---------------------------------------------------------------------------


class TestAddModel:
    def test_add_model_auto_port(self):
        server = MultiModelServer(base_port=9000)
        inst = server.add_model("llama3")
        assert inst.model == "llama3"
        assert inst.port == 9000

    def test_add_model_auto_port_increments(self):
        server = MultiModelServer(base_port=9000)
        inst0 = server.add_model("llama3")
        inst1 = server.add_model("qwen2")
        assert inst0.port == 9000
        assert inst1.port == 9001

    def test_add_model_explicit_port(self):
        server = MultiModelServer(base_port=9000)
        inst = server.add_model("llama3", port=7777)
        assert inst.port == 7777

    def test_add_model_explicit_socket_id(self):
        server = MultiModelServer()
        inst = server.add_model("llama3", port=8000, socket_id=1)
        assert inst.socket_id == 1

    def test_add_model_appends_to_instances(self):
        server = MultiModelServer()
        server.add_model("llama3")
        server.add_model("qwen2")
        assert len(server.instances) == 2
        assert server.instances[0].model == "llama3"
        assert server.instances[1].model == "qwen2"

    def test_add_model_returns_instance(self):
        server = MultiModelServer()
        result = server.add_model("llama3")
        assert isinstance(result, ModelInstance)
        assert result is server.instances[0]


# ---------------------------------------------------------------------------
# MultiModelServer.get_model_urls tests
# ---------------------------------------------------------------------------


class TestGetModelUrls:
    def test_empty_when_no_instances(self):
        server = MultiModelServer()
        assert server.get_model_urls() == {}

    def test_only_loaded_instances_included(self):
        server = MultiModelServer()
        inst_a = server.add_model("llama3", port=8000)
        server.add_model("qwen2", port=8001)  # loaded stays False
        inst_a.loaded = True
        urls = server.get_model_urls()
        assert "llama3" in urls
        assert "qwen2" not in urls

    def test_url_format(self):
        server = MultiModelServer()
        inst = server.add_model("llama3", port=8000)
        inst.loaded = True
        urls = server.get_model_urls()
        assert urls["llama3"] == "http://127.0.0.1:8000"

    def test_multiple_loaded_instances(self):
        server = MultiModelServer()
        for name, port in [("llama3", 8000), ("qwen2", 8001), ("mistral", 8002)]:
            inst = server.add_model(name, port=port)
            inst.loaded = True
        urls = server.get_model_urls()
        assert urls == {
            "llama3": "http://127.0.0.1:8000",
            "qwen2": "http://127.0.0.1:8001",
            "mistral": "http://127.0.0.1:8002",
        }


# ---------------------------------------------------------------------------
# MultiModelServer.stop_all tests
# ---------------------------------------------------------------------------


class TestStopAll:
    def test_stop_all_terminates_processes(self):
        server = MultiModelServer()
        inst = server.add_model("llama3", port=8000)
        mock_proc = MagicMock(spec=subprocess.Popen)
        inst.process = mock_proc
        inst.loaded = True

        server.stop_all()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert inst.loaded is False

    def test_stop_all_skips_none_process(self):
        server = MultiModelServer()
        server.add_model("llama3", port=8000)
        # process is None — should not raise
        server.stop_all()

    def test_stop_all_kills_on_timeout(self):
        server = MultiModelServer()
        inst = server.add_model("llama3", port=8000)
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=5), None]
        inst.process = mock_proc

        server.stop_all()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    def test_stop_all_multiple_instances(self):
        server = MultiModelServer()
        procs = []
        for name, port in [("llama3", 8000), ("qwen2", 8001)]:
            inst = server.add_model(name, port=port)
            mock_proc = MagicMock(spec=subprocess.Popen)
            inst.process = mock_proc
            inst.loaded = True
            procs.append(mock_proc)

        server.stop_all()

        for proc in procs:
            proc.terminate.assert_called_once()
        assert all(not inst.loaded for inst in server.instances)


# ---------------------------------------------------------------------------
# MultiModelServer.start_all tests
# ---------------------------------------------------------------------------


class _FakeModelInfo:
    def __init__(self, path: str):
        self._path = path

    def __getitem__(self, key: str) -> str:
        if key == "path":
            return self._path
        raise KeyError(key)


@pytest.fixture()
def mock_db_get_model():
    """Return a fake model info dict for any model name."""
    with patch(
        "llamacpp_cli.multi_model_server.MultiModelServer.start_all.__wrapped__",
        create=True,
    ):
        pass

    def _fake_get(name: str) -> dict | None:
        return {"path": f"/models/{name}.gguf"}

    with patch("llamacpp_cli.multi_model_server.db_get_model", _fake_get):
        yield _fake_get


class TestStartAll:
    def _make_server_with_mocks(
        self,
        model_names: list[str],
        numa_topology: dict | None = None,
    ):
        """Helper: create a MultiModelServer and patch heavy dependencies."""
        server = MultiModelServer(base_port=8000)
        for name in model_names:
            server.add_model(name)

        default_topology = {
            "has_numa": False,
            "numa_nodes": [0],
            "cpus_per_node": {0: list(range(8))},
            "num_sockets": 1,
        }
        topology = numa_topology or default_topology
        return server, topology

    def test_start_all_launches_processes(self):
        server, topology = self._make_server_with_mocks(["llama3", "qwen2"])

        fake_proc = MagicMock(spec=subprocess.Popen)

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                return_value=["llama-server", "--model", "x"],
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready"),
        ):
            server.start_all()

        assert server.instances[0].process is not None
        assert server.instances[1].process is not None

    def test_start_all_marks_loaded(self):
        server, topology = self._make_server_with_mocks(["llama3"])
        fake_proc = MagicMock(spec=subprocess.Popen)

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                return_value=["llama-server", "--model", "x"],
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready"),
        ):
            server.start_all()

        assert server.instances[0].loaded is True

    def test_start_all_skips_unknown_model(self, capsys):
        server = MultiModelServer(base_port=8000)
        server.add_model("unknown-model")

        topology = {
            "has_numa": False,
            "numa_nodes": [0],
            "cpus_per_node": {0: []},
            "num_sockets": 1,
        }

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                return_value=None,
            ),
        ):
            server.start_all()

        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert server.instances[0].process is None
        assert server.instances[0].loaded is False

    def test_start_all_numa_round_robin(self):
        """Verify NUMA node assignment is round-robin across instances."""
        server = MultiModelServer(base_port=8000)
        for name in ["m0", "m1", "m2"]:
            server.add_model(name)

        topology = {
            "has_numa": True,
            "numa_nodes": [0, 1],
            "cpus_per_node": {0: list(range(8)), 1: list(range(8, 16))},
            "num_sockets": 2,
        }
        fake_proc = MagicMock(spec=subprocess.Popen)

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                return_value=["llama-server", "--model", "x"],
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready"),
        ):
            server.start_all()

        # Round-robin: 0->node0, 1->node1, 2->node0
        assert server.instances[0].socket_id == 0
        assert server.instances[1].socket_id == 1
        assert server.instances[2].socket_id == 0

    def test_start_all_no_numa_uses_socket_zero(self):
        server, topology = self._make_server_with_mocks(["llama3"], numa_topology={
            "has_numa": False,
            "numa_nodes": [0],
            "cpus_per_node": {0: list(range(8))},
            "num_sockets": 1,
        })
        fake_proc = MagicMock(spec=subprocess.Popen)

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                return_value=["llama-server", "--model", "x"],
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready"),
        ):
            server.start_all()

        # No NUMA: socket_id stays at the add_model default (0)
        assert server.instances[0].socket_id == 0

    def test_start_all_passes_extra_args(self):
        server = MultiModelServer(base_port=8000)
        server.add_model("llama3")

        topology = {
            "has_numa": False,
            "numa_nodes": [0],
            "cpus_per_node": {0: []},
            "num_sockets": 1,
        }
        fake_proc = MagicMock(spec=subprocess.Popen)
        captured_extra: list[list[str]] = []

        def _fake_build_cmd(**kwargs):
            captured_extra.append(kwargs.get("extra_args") or [])
            return ["llama-server"]

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                side_effect=_fake_build_cmd,
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready"),
        ):
            server.start_all(extra_args=["--verbose"])

        assert captured_extra[0] == ["--verbose"]

    def test_start_all_timeout_warning(self, capsys):
        """When _wait_ready times out the instance is not marked loaded."""
        server = MultiModelServer(base_port=8000)
        server.add_model("llama3")

        topology = {
            "has_numa": False,
            "numa_nodes": [0],
            "cpus_per_node": {0: []},
            "num_sockets": 1,
        }
        fake_proc = MagicMock(spec=subprocess.Popen)

        with (
            patch(
                "llamacpp_cli.multi_model_server.detect_numa_topology",
                return_value=topology,
            ),
            patch(
                "llamacpp_cli.multi_model_server.db_get_model",
                side_effect=lambda n: {"path": f"/models/{n}.gguf"},
            ),
            patch(
                "llamacpp_cli.multi_model_server.build_server_cmd",
                return_value=["llama-server"],
            ),
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(server, "_wait_ready", side_effect=TimeoutError("timed out")),
        ):
            server.start_all()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "timed out" in captured.out
        assert server.instances[0].loaded is False


# ---------------------------------------------------------------------------
# MultiModelServer._wait_ready tests
# ---------------------------------------------------------------------------


class TestWaitReady:
    def test_returns_immediately_on_200(self):
        server = MultiModelServer()

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("llamacpp_cli.multi_model_server.httpx.get", return_value=mock_resp):
            server._wait_ready(8000, timeout=5.0)  # Should not raise

    def test_raises_on_timeout(self):
        server = MultiModelServer()

        with (
            patch(
                "llamacpp_cli.multi_model_server.httpx.get",
                side_effect=Exception("conn refused"),
            ),
            patch("llamacpp_cli.multi_model_server.time.sleep"),
            patch(
                "llamacpp_cli.multi_model_server.time.time",
                side_effect=[0.0, 0.6, 100.0],
            ),
            pytest.raises(TimeoutError, match="not ready after"),
        ):
            server._wait_ready(8000, timeout=1.0)

    def test_retries_on_non_200(self):
        server = MultiModelServer()

        responses = [
            MagicMock(status_code=503),
            MagicMock(status_code=503),
            MagicMock(status_code=200),
        ]
        call_count = 0
        times = [0.0, 0.5, 1.0, 1.5]
        time_idx = [0]

        def _fake_time():
            val = times[min(time_idx[0], len(times) - 1)]
            time_idx[0] += 1
            return val

        def _fake_get(*args, **kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        with (
            patch("llamacpp_cli.multi_model_server.httpx.get", side_effect=_fake_get),
            patch("llamacpp_cli.multi_model_server.time.sleep"),
            patch("llamacpp_cli.multi_model_server.time.time", side_effect=_fake_time),
        ):
            server._wait_ready(8000, timeout=10.0)

        assert call_count >= 3
