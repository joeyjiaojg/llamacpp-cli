"""Model download and management from Hugging Face."""

import contextlib
import os
import re
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
    "qwen3:14b": "Qwen/Qwen3-14B-GGUF",
    "qwen3.5": "jc-builds/Qwen3.5-9B-Q4_K_M-GGUF",
    "qwen3.5:9b": "jc-builds/Qwen3.5-9B-Q4_K_M-GGUF",
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

    Returns (repo_id, display_name, quantization).
    """
    # Check if there's a size tag (like ":3b", ":270m") separate from the base name
    # That is, has colon but no slash (to distinguish from HF repo 'namespace/model:quant')
    has_size_tag = ":" in name and "/" not in name

    # Check for direct key match first (e.g. 'llama3.2:3b' or 'gemma3:270m')
    if name in _SHORT_NAME_MAP:
        repo_id = _SHORT_NAME_MAP[name]
        # If there's a size tag, extract it as quant and find base name for display
        if has_size_tag:
            # Extract size part as quant (e.g. "3b" from "llama3.2:3b")
            size_part = name.split(":", 1)[1]
            # Find base name by progressively shortening
            import re

            tokens = re.split(r"([:\-])", name)
            for prefix in tokens[:-1]:  # skip the last (size part)
                if prefix in _SHORT_NAME_MAP:
                    return repo_id, prefix, size_part
        return repo_id, name, None

    # Try progressively shorter prefixes of name split on ':' and '-' boundaries
    import re

    tokens = re.split(r"([:\-])", name)
    prefixes = []
    current = ""
    for token in tokens:
        current += token
        prefixes.append(current)
    # Try longest-first (reverse order)
    for prefix in reversed(prefixes):
        if prefix in _SHORT_NAME_MAP:
            repo_id = _SHORT_NAME_MAP[prefix]
            remainder = name[len(prefix) :]
            # Strip leading separator from remainder to get quantization
            quant = remainder.lstrip(":-") or None
            return repo_id, prefix, quant

    # Not a short name — must be a full HF repo path (namespace/model or namespace/model:quant)
    parts = name.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Model name must be in 'namespace/model' format or a known short name, got: {name}. "
            f"Available short names: {', '.join(sorted(_SHORT_NAME_MAP))}"
        )

    # Full HF name - split off quantization if present
    if ":" in name:
        base, quant = name.rsplit(":", 1)
    else:
        base = name
        quant = None

    return base, base, quant


def _find_gguf_file(
    repo_id: str, quantization: str | None = None, repo_files: list[str] | None = None
) -> str:
    """Find the first GGUF file in a HuggingFace repo matching the quantization.

    If quantization is specified (e.g. 'Q4_K_M'), look for a file containing it.
    Otherwise, prefer Q4_K_M or fall back to the first GGUF file found.
    Pass repo_files to avoid a redundant API call.
    """
    if repo_files is None:
        api = HfApi(endpoint=get_hf_endpoint())
        repo_files = list(api.list_repo_files(repo_id))
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]

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

            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=60, verify=_ssl_verify()
            ) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable: file is already fully downloaded
                    print()
                    return
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
                            msg = f"\r  [{bar}] {gb_done:.1f}/{gb_total:.1f} GB ({pct:.1f}%)"
                            print(msg, end="", flush=True)
            print()  # newline after progress bar
            return  # success

        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** min(attempt, 5)
                retry_msg = (
                    f"\n  Connection error ({e}). Retrying in {wait}s..."
                    f" (attempt {attempt + 1}/{max_retries})"
                )
                print(retry_msg)
                time.sleep(wait)
            else:
                raise


def _find_all_shards(repo_files: list[str], first_shard: str) -> list[str]:
    """Given the first shard filename, return all shards in order.

    Detects the pattern *-NNNNN-of-MMMMM.gguf and returns all matching files.
    Falls back to [first_shard] if no split pattern is found.
    """
    m = re.search(r"-(\d+)-of-(\d+)\.gguf$", first_shard, re.IGNORECASE)
    if not m:
        return [first_shard]

    total = int(m.group(2))
    prefix = first_shard[: m.start()]  # everything before "-NNNNN-of-MMMMM.gguf"
    width = len(m.group(1))

    shards = []
    for i in range(1, total + 1):
        shard = f"{prefix}-{str(i).zfill(width)}-of-{str(total).zfill(width)}.gguf"
        # Use the actual filename from the repo if available (may be in a subdir)
        match = next((f for f in repo_files if f.endswith(shard.split("/")[-1])), shard)
        shards.append(match)
    return shards


def pull_model(name: str) -> None:
    """Download a GGUF model from Hugging Face."""
    repo_id, display_name, quantization = _parse_model_name(name)

    api = HfApi(endpoint=get_hf_endpoint())
    repo_files = list(api.list_repo_files(repo_id))

    first_shard = _find_gguf_file(repo_id, quantization, repo_files=repo_files)
    all_shards = _find_all_shards(repo_files, first_shard)

    models_dir = get_models_dir()
    model_subdir = models_dir / repo_id.replace("/", "--")
    model_subdir.mkdir(parents=True, exist_ok=True)

    # Check if all shards already on disk (handles prior partial download registered in DB)
    all_dest = [model_subdir / Path(s).name for s in all_shards]
    if all(d.exists() for d in all_dest):
        existing = get_model(display_name)
        if existing:
            print(f"Model '{display_name}' already exists at {existing['path']}")
            return

    endpoint = get_hf_endpoint().rstrip("/")
    total_size = 0

    for i, (shard, dest_path) in enumerate(zip(all_shards, all_dest, strict=True), 1):
        url = f"{endpoint}/{repo_id}/resolve/main/{shard}"
        print(f"Pulling {repo_id} [{i}/{len(all_shards)}] {dest_path.name}...")
        _download_resumable(url, dest_path)
        total_size += dest_path.stat().st_size

    # Register using the first shard as the model path (llama.cpp auto-discovers the rest)
    add_model(
        name=display_name,
        repo_id=repo_id,
        filename=first_shard,
        path=str(all_dest[0]),
        quantization=quantization,
        size_bytes=total_size,
    )

    print(f"Success! Model saved to {model_subdir}")


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


def _read_gguf_metadata(file_path: str) -> dict:
    """Extract metadata from a GGUF file.

    Returns a dict with metadata key-value pairs. Returns empty dict on error.
    GGUF format: magic(4) version(4) tensor_count(8) metadata_kv_count(8) metadata_kvs...
    """
    import struct

    def get_type_size(value_type):
        """Return size in bytes for fixed-size types, or None for variable-size."""
        size_map = {
            4: 4,  # uint32
            5: 4,  # int32
            6: 8,  # uint64
            7: 8,  # int64
            10: 4,  # float32
            11: 1,  # bool
        }
        return size_map.get(value_type)

    def skip_value(f, value_type):
        """Skip a value based on its type. Returns False if we should stop parsing."""
        size = get_type_size(value_type)
        if size:
            f.read(size)
            return True

        if value_type == 8:  # string
            str_len = struct.unpack("<Q", f.read(8))[0]
            if str_len > 10 * 1024 * 1024:  # Skip strings larger than 10MB
                return False
            f.read(str_len)
            return True
        elif value_type == 9:  # array
            array_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]

            # For large arrays of fixed-size types, seek instead of reading
            element_size = get_type_size(array_type)
            if element_size and array_len > 0:
                total_size = element_size * array_len
                if total_size > 100 * 1024 * 1024:  # Skip arrays larger than 100MB
                    return False
                f.seek(total_size, 1)  # seek from current position
                return True
            elif array_type == 8:  # array of strings
                for _ in range(min(array_len, 100)):  # limit to prevent issues
                    if not skip_value(f, array_type):
                        return False
                return True
            else:
                # Unsupported nested array type
                return False
        else:
            # Unknown type
            return False

    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {}

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return {}

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            metadata = {}

            for i in range(metadata_kv_count):
                try:
                    # Read key (length-prefixed string)
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    key = f.read(key_len).decode("utf-8", errors="ignore")

                    # Read value type
                    value_type = struct.unpack("<I", f.read(4))[0]

                    # Parse value based on type (only handle types we care about)
                    if value_type == 4:  # uint32
                        value = struct.unpack("<I", f.read(4))[0]
                        metadata[key] = value
                    elif value_type == 6:  # uint64
                        value = struct.unpack("<Q", f.read(8))[0]
                        metadata[key] = value
                    elif value_type == 8:  # string
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        if str_len > 10 * 1024 * 1024:  # Skip very large strings
                            break
                        value = f.read(str_len).decode("utf-8", errors="ignore")
                        metadata[key] = value
                    else:
                        # Skip other types (arrays, floats, etc.)
                        if not skip_value(f, value_type):
                            # Stop parsing on error, but return what we have
                            break
                except (struct.error, MemoryError, OSError):
                    # Stop parsing on any error, return what we have
                    break

            return metadata
    except Exception:
        return {}


def show_model_info(name: str) -> None:
    """Display detailed information about a model (similar to ollama show)."""
    model = get_model(name)
    if not model:
        print(f"Model '{name}' not found.")
        print(f"\nUse 'llamacpp list' to see all downloaded models.")
        return

    from pathlib import Path

    from rich.console import Console

    console = Console()

    # Basic information
    console.print(f"\n[bold cyan]{model['name']}[/bold cyan]")
    console.print(f"Repository:     {model['repo_id']}")
    console.print(f"Quantization:   {model['quantization'] or 'N/A'}")
    console.print(f"Size:           {_format_size(model['size_bytes'])}")
    console.print(f"Downloaded:     {model['downloaded_at']}")

    # Check if file exists and extract metadata
    model_path = Path(model["path"])
    if not model_path.exists():
        console.print("[red]Warning: Model file not found on disk[/red]")
    else:
        # Extract context length from GGUF metadata
        metadata = _read_gguf_metadata(str(model_path))

        # Try to find context length - check architecture-specific keys first
        ctx_length = None
        # Common context length keys (architecture.context_length)
        for key, value in metadata.items():
            if key.endswith(".context_length") or key.endswith(".n_ctx_train"):
                ctx_length = value
                break

        # Fall back to generic keys
        if not ctx_length:
            ctx_length = (
                metadata.get("llama.context_length")
                or metadata.get("context_length")
                or metadata.get("n_ctx_train")
            )

        if ctx_length:
            console.print(f"Context Length: {ctx_length:,}")

    console.print(f"File:           {model['path']}")
    console.print()
