"""Configuration and paths for llamacpp-cli."""

import os
from pathlib import Path


def get_base_dir() -> Path:
    """Get the base directory for llamacpp-cli state (~/.llamacpp/)."""
    base = Path(os.environ.get("LLAMACPP_HOME", Path.home() / ".llamacpp"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_models_dir() -> Path:
    """Get the directory for storing GGUF model files."""
    d = get_base_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_bin_dir() -> Path:
    """Get the directory for llama.cpp binaries."""
    d = get_base_dir() / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_db_path() -> Path:
    """Get the path to the model metadata database."""
    return get_base_dir() / "models.db"


def find_llama_binary(name: str) -> str:
    """Find a llama.cpp binary by name.

    Search order:
    1. LLAMACPP_BIN_DIR env var
    2. ~/.llamacpp/bin/
    3. System PATH
    """
    # Check env override
    env_dir = os.environ.get("LLAMACPP_BIN_DIR")
    if env_dir:
        candidate = Path(env_dir) / name
        if candidate.is_file():
            return str(candidate)

    # Check local bin dir
    local_bin = get_bin_dir() / name
    if local_bin.is_file():
        return str(local_bin)

    # Fall back to PATH
    import shutil

    found = shutil.which(name)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find '{name}'. Install llama.cpp or set LLAMACPP_BIN_DIR."
    )


def get_hf_endpoint() -> str:
    """Get the Hugging Face API endpoint.

    Defaults to hf-mirror.com. Set HF_ENDPOINT to override.
    """
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
