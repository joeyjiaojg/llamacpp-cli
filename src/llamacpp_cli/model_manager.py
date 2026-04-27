"""Model download and management from Hugging Face."""

import contextlib
import os
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi

from .config import get_hf_endpoint, get_models_dir
from .db import add_model, get_model, list_models, remove_model

# Mapping of short Ollama-style names to HuggingFace repo IDs that host GGUF files
_SHORT_NAME_MAP: dict[str, str] = {
    "gemma3": "unsloth/gemma-3-1b-it-GGUF",
    "gemma3:270m": "unsloth/gemma-3-270m-it-GGUF",
    "gemma3:1b": "unsloth/gemma-3-1b-it-GGUF",
    "gemma3:4b": "unsloth/gemma-3-4b-it-GGUF",
    "gemma3:12b": "unsloth/gemma-3-12b-it-GGUF",
    "gemma3:27b": "unsloth/gemma-3-27b-it-GGUF",
    "llama3.2": "meta-llama/Llama-3.2-1B-Instruct-GGUF",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct-GGUF",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct-GGUF",
    "qwen3": "Qwen/Qwen3-0.6B-GGUF",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B-GGUF",
    "qwen3:1.7b": "Qwen/Qwen3-1.7B-GGUF",
    "qwen3:4b": "Qwen/Qwen3-4B-GGUF",
    "qwen3:8b": "Qwen/Qwen3-8B-GGUF",
    "qwen3-coder": "Qwen/Qwen3-Coder-480B-A35B-GGUF",
    "qwen3-coder:30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-GGUF",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3-GGUF",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3-GGUF",
    "phi3": "microsoft/Phi-3-mini-4k-instruct-gguf",
    "phi3:3.8b": "microsoft/Phi-3-mini-4k-instruct-gguf",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF",
}


def _parse_model_name(name: str) -> tuple[str, str, str | None]:
    """Parse a model name like 'namespace/model:Q4_K_M' into parts.

    Supports:
      - Full HF names: 'TheBloke/LLaMA2-7B-Chat:Q4_K_M'
      - Short names with optional size+quant: 'gemma3', 'gemma3:1b', 'qwen3-coder:30b-a3b-q4_K_M'

    For short names, tries progressively shorter prefixes to find the longest map match,
    treating the remainder as the quantization tag.

    Returns (repo_id, display_name, quantization).
    """
    # Try progressively shorter prefixes of 'name' split on ':' and '-' boundaries
    # to find the longest matching key in _SHORT_NAME_MAP.
    # e.g. "qwen3-coder:30b-a3b-q4_K_M" tries:
    #   "qwen3-coder:30b-a3b-q4_K_M" -> not found
    #   "qwen3-coder:30b-a3b"         -> found! remainder "q4_K_M" = quantization
    #   "qwen3-coder:30b"             -> would match if present
    #   "qwen3-coder"                 -> fallback
    parts = []
    # Build list of split points (after each ':' or '-' boundary)
    import re
    tokens = re.split(r"([:\-])", name)
    # tokens alternates between text and separators: ["qwen3", "-", "coder", ":", "30b", "-", "a3b", "-", "q4_K_M"]
    # Rebuild prefixes by joining 0..n tokens
    prefixes = []
    current = ""
    for token in tokens:
        current += token
        prefixes.append(current)
    # Try longest-first (reverse order)
    for prefix in reversed(prefixes):
        if prefix in _SHORT_NAME_MAP:
            repo_id = _SHORT_NAME_MAP[prefix]
            remainder = name[len(prefix):]
            # Strip leading separator from remainder to get quantization
            quant = remainder.lstrip(":-") or None
            return repo_id, prefix, quant

    # Not a short name — must be a full HF repo path (namespace/model or namespace/model:quant)
    if ":" in name:
        base, quant = name.rsplit(":", 1)
    else:
        base = name
        quant = None

    parts = base.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Model name must be in 'namespace/model' format or a known short name, got: {name}. "
            f"Available short names: {', '.join(sorted(_SHORT_NAME_MAP))}"
        )

    return base, base, quant


def _find_gguf_file(repo_id: str, quantization: str | None = None) -> str:
    """Find a GGUF file in a HuggingFace repo.

    If quantization is specified (e.g. 'Q4_K_M'), look for a file containing it.
    Otherwise, prefer Q4_K_M or fall back to the first GGUF file found.
    """
    api = HfApi(endpoint=get_hf_endpoint())
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        raise ValueError(f"No GGUF files found in repo '{repo_id}'")

    if quantization:
        matches = [f for f in gguf_files if quantization.lower() in f.lower()]
        if matches:
            return matches[0]
        # quantization tag didn't match a filename — fall through to default selection
        # (e.g. size tags like '270m' route to the right repo but aren't in filenames)

    # Default: prefer Q4_K_M
    preferred = [f for f in gguf_files if "Q4_K_M" in f]
    if preferred:
        return preferred[0]

    return gguf_files[0]


def _ssl_verify() -> bool:
    return os.environ.get("LLAMACPP_SSL_VERIFY", "true").lower() not in ("0", "false", "no")


def _download_resumable(url: str, dest: Path, max_retries: int = 10) -> None:
    """Download a file with resume-on-error support using HTTP Range requests."""
    headers = {}
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"

    for attempt in range(max_retries):
        try:
            resumed_at = dest.stat().st_size if dest.exists() else 0
            if resumed_at:
                headers["Range"] = f"bytes={resumed_at}-"
                print(f"Resuming from {resumed_at / 1024**3:.1f} GB...")

            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60, verify=_ssl_verify()) as resp:
                resp.raise_for_status()

                total = None
                if resp.status_code == 206:  # Partial Content
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                elif resp.status_code == 200:
                    resumed_at = 0  # Server doesn't support range, start over
                    total = int(resp.headers.get("Content-Length", 0)) or None

                mode = "ab" if resumed_at else "wb"
                downloaded = resumed_at
                with open(dest, mode) as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            done = int(pct / 2)
                            bar = "█" * done + "░" * (50 - done)
                            gb_done = downloaded / 1024**3
                            gb_total = total / 1024**3
                            print(f"\r  [{bar}] {gb_done:.1f}/{gb_total:.1f} GB ({pct:.1f}%)", end="", flush=True)
            print()  # newline after progress bar
            return  # success

        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** min(attempt, 5)
                print(f"\n  Connection error ({e}). Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def pull_model(name: str) -> None:
    """Download a GGUF model from Hugging Face."""
    repo_id, display_name, quantization = _parse_model_name(name)

    # Check if already downloaded
    existing = get_model(display_name)
    if existing:
        print(f"Model '{display_name}' already exists at {existing['path']}")
        return

    filename = _find_gguf_file(repo_id, quantization)

    models_dir = get_models_dir()
    model_subdir = models_dir / repo_id.replace("/", "--")
    model_subdir.mkdir(parents=True, exist_ok=True)
    # Flatten subdirs in filename to avoid nested dirs (e.g. "subdir/file.gguf")
    dest_path = model_subdir / Path(filename).name

    endpoint = get_hf_endpoint().rstrip("/")
    url = f"{endpoint}/{repo_id}/resolve/main/{filename}"

    print(f"Pulling {repo_id} ({filename})...")
    _download_resumable(url, dest_path)

    size_bytes = dest_path.stat().st_size
    add_model(
        name=display_name,
        repo_id=repo_id,
        filename=filename,
        path=str(dest_path),
        quantization=quantization,
        size_bytes=size_bytes,
    )

    print(f"Success! Model saved to {dest_path}")


def remove_model_and_file(name: str) -> None:
    """Remove a model from the database and delete its file."""
    model = get_model(name)
    if not model:
        print(f"Model '{name}' not found.")
        return

    # Delete the file
    model_path = Path(model["path"])
    if model_path.exists():
        model_path.unlink()
    # Clean up empty parent dirs
    with contextlib.suppress(OSError):
        model_path.parent.rmdir()

    remove_model(name)
    print(f"Deleted model '{name}'.")


def list_downloaded_models() -> None:
    """Print a table of downloaded models."""
    models = list_models()
    if not models:
        print("No models downloaded yet. Use 'llamacpp pull <model>' to download one.")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Downloaded Models")
    table.add_column("NAME", style="cyan")
    table.add_column("QUANTIZATION", style="green")
    table.add_column("SIZE", style="yellow")
    table.add_column("MODIFIED", style="dim")

    for m in models:
        size_str = _format_size(m["size_bytes"]) if m["size_bytes"] else "unknown"
        table.add_row(
            m["name"],
            m["quantization"] or "-",
            size_str,
            m["downloaded_at"],
        )

    console.print(table)


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
