"""Server management — start/stop llama.cpp server as a subprocess."""

import subprocess

from .config import find_llama_binary
from .db import get_model


def start_server(
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start the llama.cpp server as a subprocess.

    Returns the Popen object so the caller can manage the process.
    """
    binary = find_llama_binary("llama-server")

    cmd = [binary, "--host", host, "--port", str(port)]

    if model:
        model_info = get_model(model)
        if model_info:
            cmd.extend(["--model", model_info["path"]])
        else:
            # Treat as a path
            cmd.extend(["--model", model])

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
