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
    """Test max-context preset has minimal concurrency."""
    config = get_cpu_server_config('max-context')
    assert config['ctx_size'] == 32768  # Large context
    assert config['parallel'] == 1  # Single request only


def test_get_cpu_server_config_default_is_code():
    """Test that default preset is 'code'."""
    config_default = get_cpu_server_config()
    config_code = get_cpu_server_config('code')
    assert config_default == config_code


def test_get_cpu_server_config_invalid_preset():
    """Test invalid preset falls back to code."""
    config = get_cpu_server_config('invalid_preset')
    config_code = get_cpu_server_config('code')
    assert config == config_code
