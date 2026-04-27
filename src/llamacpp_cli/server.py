"""Server management — start/stop llama.cpp server as a subprocess."""

import subprocess

from .config import find_llama_binary
from .db import get_model
from .run import _is_local_path


def start_server(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
) -> subprocess.Popen | None:
    """Start the llama.cpp server as a subprocess.

    Returns the Popen object so the caller can manage the process.
    """
    binary = find_llama_binary("llama-server")

    cmd = [binary, "--host", host, "--port", str(port)]

    if model:
        model_info = get_model(model)
        if model_info:
            cmd.extend(["--model", model_info["path"]])
        elif _is_local_path(model):
            cmd.extend(["--model", model])
        else:
            from .model_manager import pull_model
            pull_model(model)
            model_info = get_model(model)
            if model_info:
                cmd.extend(["--model", model_info["path"]])
            else:
                print(f"Failed to pull model '{model}'.")
                return None

    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting llama-server on {host}:{port}...")
    if model:
        print(f"  Model: {model}")

    proc = subprocess.Popen(cmd)
    return proc


def run_server_foreground(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
) -> None:
    """Start the server in the foreground (blocking). Handles Ctrl+C gracefully."""
    proc = start_server(model=model, host=host, port=port, extra_args=extra_args)
    if proc is None:
        return

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server stopped.")
