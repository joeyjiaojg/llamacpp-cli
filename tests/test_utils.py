"""Tests for utility functions."""

import os
from unittest.mock import patch

import pytest

from llamacpp_cli.utils import (
    detect_numa,
    get_cpu_count,
    get_cpu_server_config,
    get_model_max_context,
)


def test_get_cpu_count():
    """Test CPU count detection."""
    count = get_cpu_count()
    assert count >= 1
    assert isinstance(count, int)


def test_detect_numa_on_single_socket():
    """Test NUMA detection returns False on single-socket."""
    with patch('subprocess.run') as mock_run:
        # Simulate single NUMA node
        mock_run.return_value.stdout = "NUMA node(s):              1\n"
        result = detect_numa()
        # Single node = no NUMA
        assert result is False


def test_detect_numa_on_multi_socket():
    """Test NUMA detection returns True on multi-socket."""
    with patch('subprocess.run') as mock_run:
        # Simulate multi-NUMA node
        mock_run.return_value.stdout = "NUMA node(s):              2\nNUMA node0 CPU(s):   0-15\nNUMA node1 CPU(s):   16-31\n"
        result = detect_numa()
        assert result is True


def test_get_model_max_context_codellama():
    """Test context size for CodeLlama models."""
    assert get_model_max_context("codellama-7b") == 16384
    assert get_model_max_context("TheBloke/CodeLlama-13B-Q4_K_M") == 16384


def test_get_model_max_context_llama31():
    """Test context size for Llama 3.1/3.2 models."""
    assert get_model_max_context("llama-3.1-8b") == 32768
    assert get_model_max_context("llama3.2-1b") == 32768


def test_get_model_max_context_deepseek():
    """Test context size for DeepSeek models."""
    assert get_model_max_context("deepseek-coder-6.7b") == 16384


def test_get_model_max_context_default():
    """Test default context size for unknown models."""
    assert get_model_max_context("unknown-model") == 8192


@pytest.mark.parametrize(
    "preset,expected_ctx",
    [
        ("code", 16384),
        ("chat", 8192),
        ("fast", 4096),
        ("max-context", 32768),
    ],
)
def test_get_cpu_server_config_presets(preset, expected_ctx):
    """Test preset configurations have correct context sizes."""
    config = get_cpu_server_config(preset)
    assert config['ctx_size'] == expected_ctx
    assert 'parallel' in config
    assert 'batch_size' in config
    assert 'threads' in config
    assert 'numa' in config
    assert 'mlock' in config


def test_get_cpu_server_config_code_preset():
    """Test code preset details (default)."""
    config = get_cpu_server_config('code')
    assert config['ctx_size'] == 16384
    assert config['batch_size'] == 512
    assert config['mlock'] is True
    assert config['threads'] >= 1
    # Parallel should be conservative: 2-4 requests
    assert 1 <= config['parallel'] <= 4


def test_get_cpu_server_config_fast_preset():
    """Test fast preset has higher concurrency."""
    config = get_cpu_server_config('fast')
    assert config['ctx_size'] == 4096  # Smaller context
    assert config['batch_size'] == 256  # Smaller batches
    # Fast preset should have higher parallel
    assert config['parallel'] >= 4


def test_get_cpu_server_config_max_context_preset():
    """Test max-context preset uses NUMA-aware parallel."""
    with patch('llamacpp_cli.cpu_topology.detect_numa_topology') as mock_topology:
        # Simulate dual-socket system with 2 NUMA nodes
        mock_topology.return_value = {
            "numa_nodes": [0, 1],
            "num_sockets": 2,
            "cpus_per_node": {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
            "has_numa": True,
        }
        config = get_cpu_server_config('max-context')
        assert config['ctx_size'] == 32768  # Large context
        assert config['parallel'] == 2  # Two slots (one per NUMA node)


def test_get_cpu_server_config_default_is_max_context():
    """Test that default preset is 'max-context'."""
    config_default = get_cpu_server_config()
    config_max = get_cpu_server_config('max-context')
    assert config_default == config_max


def test_get_cpu_server_config_invalid_preset():
    """Test invalid preset falls back to max-context."""
    config = get_cpu_server_config('invalid_preset')
    config_max = get_cpu_server_config('max-context')
    assert config == config_max


def test_get_cpu_server_config_numa_aware_parallel_dual_socket():
    """Test parallel slots auto-detect for dual-socket system."""
    with patch('llamacpp_cli.cpu_topology.detect_numa_topology') as mock_topology:
        # Simulate dual-socket system with 2 NUMA nodes
        mock_topology.return_value = {
            "numa_nodes": [0, 1],
            "num_sockets": 2,
            "cpus_per_node": {0: list(range(0, 16)), 1: list(range(16, 32))},
            "has_numa": True,
        }

        # max-context preset should use num_slots directly
        config = get_cpu_server_config('max-context')
        assert config['parallel'] == 2

        # code preset should be at least num_slots
        config = get_cpu_server_config('code')
        assert config['parallel'] >= 2

        # chat preset should be at least num_slots
        config = get_cpu_server_config('chat')
        assert config['parallel'] >= 2

        # fast preset should be at least num_slots
        config = get_cpu_server_config('fast')
        assert config['parallel'] >= 2


def test_get_cpu_server_config_numa_aware_parallel_quad_socket():
    """Test parallel slots auto-detect for quad-socket system."""
    with patch('llamacpp_cli.cpu_topology.detect_numa_topology') as mock_topology:
        # Simulate quad-socket system with 4 NUMA nodes
        mock_topology.return_value = {
            "numa_nodes": [0, 1, 2, 3],
            "num_sockets": 4,
            "cpus_per_node": {
                0: list(range(0, 16)),
                1: list(range(16, 32)),
                2: list(range(32, 48)),
                3: list(range(48, 64)),
            },
            "has_numa": True,
        }

        # max-context preset should use all 4 slots
        config = get_cpu_server_config('max-context')
        assert config['parallel'] == 4

        # Other presets should respect caps but be at least num_slots
        config = get_cpu_server_config('code')
        assert config['parallel'] == 4  # min(4, max(4, cpu_count // 4))


def test_get_cpu_server_config_numa_aware_parallel_single_socket():
    """Test parallel slots auto-detect for single-socket system."""
    with patch('llamacpp_cli.cpu_topology.detect_numa_topology') as mock_topology:
        # Simulate single-socket system with 1 NUMA node
        mock_topology.return_value = {
            "numa_nodes": [0],
            "num_sockets": 1,
            "cpus_per_node": {0: list(range(0, 8))},
            "has_numa": False,
        }

        # max-context preset should use 1 slot
        config = get_cpu_server_config('max-context')
        assert config['parallel'] == 1


def test_get_cpu_server_config_numa_detection_failure():
    """Test fallback when NUMA detection fails."""
    with patch('llamacpp_cli.cpu_topology.detect_numa_topology', side_effect=Exception("Detection failed")):
        # Should fallback to num_slots=1
        config = get_cpu_server_config('max-context')
        assert config['parallel'] == 1  # Fallback to single slot


def test_get_cpu_server_config_preset_caps_respected():
    """Test that preset caps are still respected with NUMA."""
    with (
        patch('llamacpp_cli.cpu_topology.detect_numa_topology') as mock_topology,
        patch('llamacpp_cli.utils.get_cpu_count', return_value=128),
    ):
        # Simulate dual-socket system with many CPUs
        mock_topology.return_value = {
            "numa_nodes": [0, 1],
            "num_sockets": 2,
            "cpus_per_node": {0: list(range(0, 64)), 1: list(range(64, 128))},
            "has_numa": True,
        }

        # code preset: min(4, max(2, 128 // 4)) = min(4, max(2, 32)) = 4
        config = get_cpu_server_config('code')
        assert config['parallel'] == 4

        # chat preset: min(6, max(2, 128 // 3)) = min(6, max(2, 42)) = 6
        config = get_cpu_server_config('chat')
        assert config['parallel'] == 6

        # fast preset: min(8, max(2, 128 // 2)) = min(8, max(2, 64)) = 8
        config = get_cpu_server_config('fast')
        assert config['parallel'] == 8

        # max-context preset: num_slots = 2 (no cap)
        config = get_cpu_server_config('max-context')
        assert config['parallel'] == 2
