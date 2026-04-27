"""Tests for config module."""

from llamacpp_cli.config import get_base_dir, get_db_path, get_models_dir


def test_get_base_dir_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LLAMACPP_HOME", raising=False)
    base = get_base_dir()
    assert base == tmp_path / ".llamacpp"
    assert base.exists()


def test_get_base_dir_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom_llamacpp"
    monkeypatch.setenv("LLAMACPP_HOME", str(custom))
    base = get_base_dir()
    assert base == custom
    assert base.exists()


def test_get_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMACPP_HOME", str(tmp_path / "home"))
    models = get_models_dir()
    assert models.exists()
    assert models.name == "models"


def test_get_db_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMACPP_HOME", str(tmp_path / "home"))
    db = get_db_path()
    assert db.name == "models.db"
