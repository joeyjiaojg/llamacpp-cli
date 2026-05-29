"""GPU detection and optimization for llama.cpp."""

import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    vendor: str       # "nvidia", "amd", "intel", "none"
    name: str
    vram_mb: int
    gpu_layers: int   # Recommended layers to offload


@dataclass
class GPUConfig:
    """GPU configuration for llama.cpp server."""
    n_gpu_layers: int = 0     # 0 = CPU only
    main_gpu: int = 0
    split_mode: str = "layer"  # "layer", "row", "none"
    tensor_split: list[float] | None = field(default=None)  # Multi-GPU splits


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi.

    Returns a list of GPUInfo objects, one per detected GPU.
    Returns an empty list if nvidia-smi is not available or no GPUs found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            try:
                vram_mb = int(parts[1].strip())
            except ValueError:
                continue

            # Conservative estimate: ~1GB per 7B model at Q4, ~512MB per layer
            gpu_layers = min(80, vram_mb // 512)

            gpus.append(GPUInfo(
                vendor="nvidia",
                name=name,
                vram_mb=vram_mb,
                gpu_layers=gpu_layers,
            ))

        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs using rocm-smi.

    Returns a list of GPUInfo objects, one per detected GPU.
    Returns an empty list if rocm-smi is not available or no GPUs found.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        import json
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        gpus = []
        for _key, card in data.items():
            if not isinstance(card, dict):
                continue
            name = card.get("Card series", card.get("Card model", "AMD GPU"))
            vram_str = card.get("VRAM Total Memory (B)", "0")
            try:
                vram_bytes = int(vram_str)
                vram_mb = vram_bytes // (1024 * 1024)
            except (ValueError, TypeError):
                vram_mb = 0

            if vram_mb == 0:
                continue

            gpu_layers = min(80, vram_mb // 512)

            gpus.append(GPUInfo(
                vendor="amd",
                name=str(name),
                vram_mb=vram_mb,
                gpu_layers=gpu_layers,
            ))

        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def detect_all_gpus() -> list[GPUInfo]:
    """Detect all available GPUs (NVIDIA and AMD).

    Returns a combined list, NVIDIA GPUs first.
    """
    return detect_nvidia_gpus() + detect_amd_gpus()


def auto_configure_gpu(model_size_gb: float | None = None) -> GPUConfig:
    """Auto-configure GPU settings for best performance.

    Detects available GPUs and calculates the optimal number of layers
    to offload based on available VRAM and model size.

    Args:
        model_size_gb: Approximate model file size in GB, used to estimate
            layers per GB. If None, a conservative default is used.

    Returns:
        GPUConfig with recommended settings, or CPU-only config if no GPU found.
    """
    all_gpus = detect_all_gpus()

    if not all_gpus:
        return GPUConfig(n_gpu_layers=0)

    # Use first GPU (primary)
    gpu = all_gpus[0]

    if model_size_gb is None:
        # Conservative default when model size is unknown
        n_layers = min(gpu.gpu_layers, 40)
    else:
        # Calculate layers that fit in available VRAM
        # Reserve 1 GB for overhead (KV cache, activations, OS)
        available_vram_gb = (gpu.vram_mb - 1024) / 1024
        if available_vram_gb <= 0:
            return GPUConfig(n_gpu_layers=0)

        # Estimate GB per transformer layer: model_size / 32 layers * 0.5 overhead factor
        # This is a rough heuristic; actual values vary by architecture
        if model_size_gb > 0:
            gb_per_layer = (model_size_gb / 32) * 0.5
            n_layers = int(available_vram_gb / gb_per_layer) if gb_per_layer > 0 else 80
        else:
            n_layers = gpu.gpu_layers

        n_layers = min(n_layers, 80)  # Max 80 layers (covers most architectures)

    return GPUConfig(n_gpu_layers=n_layers, main_gpu=0)


def get_gpu_server_args(config: GPUConfig) -> list[str]:
    """Convert a GPUConfig to llama-server command-line arguments.

    Args:
        config: GPU configuration to convert.

    Returns:
        List of argument strings to append to the llama-server command.
    """
    args: list[str] = []

    if config.n_gpu_layers > 0:
        args += ["--n-gpu-layers", str(config.n_gpu_layers)]
        args += ["--main-gpu", str(config.main_gpu)]
        args += ["--split-mode", config.split_mode]

        if config.tensor_split:
            args += ["--tensor-split", ",".join(str(v) for v in config.tensor_split)]

    return args


def format_gpu_info(gpus: list[GPUInfo]) -> str:
    """Format GPU information for display at startup.

    Args:
        gpus: List of detected GPUs.

    Returns:
        Human-readable summary string.
    """
    if not gpus:
        return "  GPU: none detected (CPU-only mode)"

    lines = []
    for i, gpu in enumerate(gpus):
        vram_gb = gpu.vram_mb / 1024
        lines.append(f"  GPU {i}: {gpu.name} ({vram_gb:.1f} GB VRAM) [{gpu.vendor.upper()}]")
    return "\n".join(lines)
