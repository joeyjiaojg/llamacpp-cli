"""Tests for server module, including NUMA binding."""

import pytest
from unittest.mock import patch, MagicMock

from llamacpp_cli.server import build_server_cmd, _detect_cpu_topology, _has_flag


@pytest.fixture
def mock_llama_binary(monkeypatch):
    """Mock find_llama_binary to return a fake path."""
    monkeypatch.setattr(
        "llamacpp_cli.server.find_llama_binary", lambda name: f"/usr/bin/{name}"
    )


def test_build_server_cmd_basic(mock_llama_binary):
    """Test basic command construction without NUMA."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)  # Single socket
        cmd = build_server_cmd("/path/to/model.gguf")

    assert cmd[0] == "/usr/bin/llama-server"
    assert "--host" in cmd
    assert "127.0.0.1" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--model" in cmd
    assert "/path/to/model.gguf" in cmd
    assert "--threads" in cmd
    assert "8" in cmd
    assert "--threads-batch" in cmd
    assert "--no-mmap" in cmd
    # Should NOT have numactl on single socket
    assert "numactl" not in cmd


def test_build_server_cmd_with_numa_binding(mock_llama_binary):
    """Test NUMA binding on multi-socket system."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (16, True)  # Dual socket
        cmd = build_server_cmd("/path/to/model.gguf", socket_id=0)

    # Should start with numactl wrapper
    assert cmd[0] == "numactl"
    assert cmd[1] == "--cpunodebind=0"
    assert cmd[2] == "--membind=0"
    assert cmd[3] == "--"
    # Then the actual llama-server command
    assert cmd[4] == "/usr/bin/llama-server"
    assert "/path/to/model.gguf" in cmd
    assert "--threads" in cmd
    assert "16" in cmd


def test_build_server_cmd_socket_id_1(mock_llama_binary):
    """Test NUMA binding to socket 1."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (16, True)  # Dual socket
        cmd = build_server_cmd("/path/to/model.gguf", socket_id=1)

    assert cmd[0] == "numactl"
    assert cmd[1] == "--cpunodebind=1"
    assert cmd[2] == "--membind=1"


def test_build_server_cmd_with_extra_args(mock_llama_binary):
    """Test that extra_args are preserved."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        cmd = build_server_cmd(
            "/path/to/model.gguf", extra_args=["--verbose", "--log-format", "json"]
        )

    assert "--verbose" in cmd
    assert "--log-format" in cmd
    assert "json" in cmd


def test_build_server_cmd_ctx_size(mock_llama_binary):
    """Test context size override."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        cmd = build_server_cmd("/path/to/model.gguf", ctx_size=4096)

    assert "--ctx-size" in cmd
    assert "4096" in cmd


def test_build_server_cmd_no_numa_if_threads_specified(mock_llama_binary):
    """Test that NUMA binding is skipped if user specified threads."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (16, True)  # Dual socket
        cmd = build_server_cmd(
            "/path/to/model.gguf", extra_args=["--threads", "32"]
        )

    # Should NOT have numactl if user specified threads
    assert "numactl" not in cmd
    assert "--threads" in cmd
    assert "32" in cmd


def test_build_server_cmd_no_numa_if_cpunodebind_specified(mock_llama_binary):
    """Test that NUMA binding is skipped if user already specified cpunodebind."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (16, True)  # Dual socket
        cmd = build_server_cmd(
            "/path/to/model.gguf", extra_args=["--cpunodebind=1"]
        )

    # Should NOT wrap with numactl if user already specified it
    assert cmd[0] != "numactl"  # Not wrapped
    # User's flag should be preserved in extra_args
    assert "--cpunodebind=1" in cmd


def test_build_server_cmd_host_port(mock_llama_binary):
    """Test custom host and port."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        cmd = build_server_cmd(
            "/path/to/model.gguf", host="0.0.0.0", port=9090
        )

    assert "0.0.0.0" in cmd
    assert "9090" in cmd


def test_has_flag():
    """Test _has_flag helper function."""
    args = ["--verbose", "--threads", "16", "--log-format", "json"]

    assert _has_flag(args, "--verbose")
    assert _has_flag(args, "--threads")
    assert _has_flag(args, "--log-format")
    assert not _has_flag(args, "--numa")
    assert not _has_flag(args, "--cpunodebind")

    # Test multiple flags at once
    assert _has_flag(args, "--numa", "--verbose")  # Should return True if ANY match
    assert not _has_flag(args, "--numa", "--cpunodebind")


def test_has_flag_with_equals():
    """Test _has_flag handles --flag=value format."""
    args = ["--cpunodebind=0", "--membind=1", "--verbose"]

    assert _has_flag(args, "--cpunodebind")
    assert _has_flag(args, "--membind")
    assert _has_flag(args, "--verbose")
    assert not _has_flag(args, "--numa")

    # Test with exact match and equals format
    args2 = ["--threads", "16", "--ctx-size=4096"]
    assert _has_flag(args2, "--threads")
    assert _has_flag(args2, "--ctx-size")
    assert not _has_flag(args2, "--verbose")


def test_detect_cpu_topology_fallback(tmp_path, monkeypatch):
    """Test CPU topology detection fallback when /sys is unavailable."""
    # Point to non-existent directory
    monkeypatch.setattr("os.scandir", lambda path: [])

    cores, multi = _detect_cpu_topology()
    # Should fall back to cpu_count // 2
    assert cores > 0
    assert isinstance(multi, bool)


def test_build_server_cmd_no_mmap_default(mock_llama_binary):
    """Test that --no-mmap is applied by default."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        cmd = build_server_cmd("/path/to/model.gguf")

    assert "--no-mmap" in cmd


def test_build_server_cmd_mmap_override(mock_llama_binary):
    """Test that user can override --no-mmap with --mmap."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        cmd = build_server_cmd("/path/to/model.gguf", extra_args=["--mmap"])

    assert "--mmap" in cmd
    assert "--no-mmap" not in cmd


def test_build_server_cmd_threads_batch(mock_llama_binary):
    """Test that --threads-batch is set to match --threads."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (12, False)
        cmd = build_server_cmd("/path/to/model.gguf")

    # Find position of --threads and --threads-batch
    threads_idx = cmd.index("--threads")
    threads_batch_idx = cmd.index("--threads-batch")

    # Both should have same value (12)
    assert cmd[threads_idx + 1] == "12"
    assert cmd[threads_batch_idx + 1] == "12"


def test_build_server_cmd_numa_with_ctx_size(mock_llama_binary):
    """Test NUMA binding works together with ctx_size."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (16, True)  # Dual socket
        cmd = build_server_cmd("/path/to/model.gguf", ctx_size=8192, socket_id=0)

    # Should have numactl wrapper
    assert cmd[0] == "numactl"
    assert cmd[1] == "--cpunodebind=0"
    assert cmd[2] == "--membind=0"

    # Should have ctx-size in the wrapped command
    assert "--ctx-size" in cmd
    assert "8192" in cmd


def test_build_server_cmd_preserves_extra_args_order(mock_llama_binary):
    """Test that extra_args appear at the end of the command."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (8, False)
        extra = ["--verbose", "--slots", "4"]
        cmd = build_server_cmd("/path/to/model.gguf", extra_args=extra)

    # extra_args should be at the end
    assert cmd[-3:] == ["--verbose", "--slots", "4"]


def test_numa_binding_integration(mock_llama_binary):
    """Integration test: Verify NUMA binding command on dual-socket system."""
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        # Simulate dual-socket system with 16 cores per socket
        mock_topo.return_value = (16, True)

        # Build command for socket 1
        cmd = build_server_cmd(
            model_path="/models/qwen.gguf",
            host="127.0.0.1",
            port=8080,
            ctx_size=8192,
            socket_id=1,
        )

        # Verify numactl wrapper
        assert cmd[0:4] == ["numactl", "--cpunodebind=1", "--membind=1", "--"]

        # Verify llama-server args follow
        assert cmd[4] == "/usr/bin/llama-server"
        assert "--model" in cmd
        assert "/models/qwen.gguf" in cmd
        assert "--ctx-size" in cmd
        assert "8192" in cmd
        assert "--threads" in cmd
        assert "16" in cmd

    # Verify single-socket system doesn't wrap with numactl
    with patch("llamacpp_cli.server._detect_cpu_topology") as mock_topo:
        mock_topo.return_value = (32, False)
        cmd_single = build_server_cmd(
            model_path="/models/qwen.gguf",
            socket_id=0,
        )
        assert cmd_single[0] != "numactl"
        assert cmd_single[0] == "/usr/bin/llama-server"
