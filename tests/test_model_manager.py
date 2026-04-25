"""Tests for model_manager module."""

import pytest

from llamacpp_cli.model_manager import _format_size, _parse_model_name


def test_parse_model_name_with_quant():
    repo_id, display, quant = _parse_model_name("TheBloke/LLaMA2-7B:Q4_K_M")
    assert repo_id == "TheBloke/LLaMA2-7B"
    assert display == "TheBloke/LLaMA2-7B"
    assert quant == "Q4_K_M"


def test_parse_model_name_without_quant():
    repo_id, display, quant = _parse_model_name("TheBloke/LLaMA2-7B")
    assert repo_id == "TheBloke/LLaMA2-7B"
    assert display == "TheBloke/LLaMA2-7B"
    assert quant is None


def test_parse_model_name_invalid():
    with pytest.raises(ValueError, match="namespace/model"):
        _parse_model_name("invalid")


def test_parse_model_short_name():
    repo_id, display, quant = _parse_model_name("gemma3")
    assert repo_id == "unsloth/gemma-3-1b-it-GGUF"
    assert display == "gemma3"
    assert quant is None


def test_parse_model_short_name_with_quant():
    repo_id, display, quant = _parse_model_name("gemma3:Q4_K_M")
    assert repo_id == "unsloth/gemma-3-1b-it-GGUF"
    assert display == "gemma3"
    assert quant == "Q4_K_M"


def test_parse_model_short_name_with_size():
    repo_id, display, quant = _parse_model_name("llama3.2:3b")
    assert repo_id == "meta-llama/Llama-3.2-3B-Instruct-GGUF"
    assert display == "llama3.2"
    assert quant == "3b"


def test_parse_model_short_name_270m():
    repo_id, display, quant = _parse_model_name("gemma3:270m")
    assert repo_id == "unsloth/gemma-3-270m-it-GGUF"
    assert display == "gemma3"
    assert quant == "270m"


def test_format_size():
    assert _format_size(500) == "500.0 B"
    assert _format_size(1024) == "1.0 KB"
    assert _format_size(1024 * 1024) == "1.0 MB"
    assert _format_size(4 * 1024**3) == "4.0 GB"
    assert _format_size(None) == "unknown"
