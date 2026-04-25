"""Tests for the model metadata database."""

from pathlib import Path

import pytest

from llamacpp_cli.db import add_model, get_model, list_models, remove_model


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("llamacpp_cli.db.get_db_path", lambda: db_path)
    yield db_path


def test_add_and_get_model():
    add_model(
        "test/model", "test/model", "model.gguf", "/path/to/model.gguf", "Q4_K_M", 4_000_000_000
    )
    m = get_model("test/model")
    assert m is not None
    assert m["name"] == "test/model"
    assert m["quantization"] == "Q4_K_M"
    assert m["size_bytes"] == 4_000_000_000


def test_get_model_not_found():
    assert get_model("nonexistent") is None


def test_remove_model():
    add_model("test/model", "test/model", "model.gguf", "/path/to/model.gguf")
    assert remove_model("test/model") is True
    assert get_model("test/model") is None


def test_remove_nonexistent():
    assert remove_model("nope") is False


def test_list_models():
    add_model("a/model", "a/model", "a.gguf", "/a.gguf")
    add_model("b/model", "b/model", "b.gguf", "/b.gguf")
    models = list_models()
    assert len(models) == 2
    names = {m["name"] for m in models}
    assert names == {"a/model", "b/model"}


def test_add_model_upsert():
    add_model("test/model", "test/model", "old.gguf", "/old.gguf")
    add_model("test/model", "test/model", "new.gguf", "/new.gguf")
    m = get_model("test/model")
    assert m["filename"] == "new.gguf"
