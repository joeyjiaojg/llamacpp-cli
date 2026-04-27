"""Interactive chat mode — spawn llama.cpp for inference."""

import subprocess

from .config import find_llama_binary
from .db import get_model


def run_model(
    model: str,
    prompt: str | None = None,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    extra_args: list[str] | None = None,
) -> None:
    """Run a model interactively using llama-cli.

    If prompt is provided, run a single inference and exit.
    Otherwise, start an interactive chat session.
    """
    binary = find_llama_binary("llama-cli")

    model_info = get_model(model)
    if not model_info:
        if "/" in model or model.endswith(".gguf"):
            # Treat as a direct file path
            model_path = model
        else:
            # Auto-pull like Ollama does
            from .model_manager import pull_model
            pull_model(model)
            model_info = get_model(model)
            if not model_info:
                print(f"Failed to pull model '{model}'.")
                return
            model_path = model_info["path"]
    else:
        model_path = model_info["path"]

    cmd = [
        binary,
        "--model", model_path,
        "--ctx-size", str(n_ctx),
        "--n-gpu-layers", str(n_gpu_layers),
    ]

    if prompt:
        cmd.extend(["--prompt", prompt])
    else:
        # Interactive mode — use conversation template
        cmd.append("--conversation")

    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
