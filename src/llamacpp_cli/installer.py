"""Auto-install llama.cpp binaries if not found."""

import platform
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

from .config import get_bin_dir

# llama.cpp GitHub release URLs
_GITHUB_REPO = "ggml-org/llama.cpp"
_RELEASE_API = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"

# Map platform to release asset name patterns
# Actual asset names: llama-b{build}-bin-{os}-{arch}.tar.gz/.zip
_PLATFORM_MAP = {
    ("Linux", "x86_64"): "bin-ubuntu-x64",
    ("Linux", "aarch64"): "bin-ubuntu-arm64",
    ("Darwin", "x86_64"): "bin-macos-x64",
    ("Darwin", "arm64"): "bin-macos-arm64",
    ("Windows", "x86_64"): "bin-win-cpu-x64",
}


def _get_platform_key() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()
    if machine == "AMD64":
        machine = "x86_64"
    return system, machine


def _find_release_asset(assets: list[dict], pattern: str) -> dict | None:
    """Find the CPU-only release asset matching the platform pattern.

    Avoids matching CUDA/ROCm/Vulkan/Sycl variants by preferring the
    shortest match (the plain CPU build has no extra qualifiers).
    """
    matches = []
    for asset in assets:
        name = asset["name"].lower()
        if pattern.lower() in name:
            matches.append(asset)
    if not matches:
        return None
    # Pick the shortest name — the plain CPU build has no extra suffixes
    # e.g. "bin-ubuntu-x64" beats "bin-ubuntu-rocm-7.2-x64"
    return min(matches, key=lambda a: len(a["name"]))


def install_llamacpp() -> bool:
    """Download and install llama.cpp binaries to ~/.llamacpp/bin/.

    Returns True if installation succeeded.
    """
    system, machine = _get_platform_key()
    key = (system, machine)
    if key not in _PLATFORM_MAP:
        print(f"Unsupported platform: {system} {machine}")
        print("Install llama.cpp manually: https://github.com/ggml-org/llama.cpp")
        return False

    pattern = _PLATFORM_MAP[key]
    bin_dir = get_bin_dir()

    print(f"Fetching latest llama.cpp release for {system} {machine}...")

    try:
        resp = requests.get(_RELEASE_API, timeout=15)
        resp.raise_for_status()
        release = resp.json()
    except requests.RequestException as e:
        print(f"Error fetching release info: {e}")
        return False

    asset = _find_release_asset(release.get("assets", []), pattern)
    if not asset:
        print(f"No matching release found for pattern '{pattern}'.")
        print("Install llama.cpp manually: https://github.com/ggml-org/llama.cpp")
        return False

    download_url = asset["browser_download_url"]
    filename = asset["name"]

    print(f"Downloading {filename}...")

    try:
        resp = requests.get(download_url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error downloading: {e}")
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / filename
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting...")
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(bin_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path) as tf:
                tf.extractall(bin_dir)
        else:
            print(f"Unknown archive format: {filename}")
            return False

    # Make binaries executable
    for f in bin_dir.iterdir():
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o755)

    print(f"llama.cpp installed to {bin_dir}")
    return True


def ensure_llamacpp() -> bool:
    """Check if llama.cpp is available, offer to install if not.

    Returns True if binaries are available (already or after install).
    """
    from .config import find_llama_binary

    try:
        find_llama_binary("llama-server")
        find_llama_binary("llama-cli")
        return True
    except FileNotFoundError:
        pass

    print("llama.cpp not found.")
    answer = input("Would you like to install it automatically? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        return install_llamacpp()

    print("Set LLAMACPP_BIN_DIR or install llama.cpp manually:")
    print("  https://github.com/ggml-org/llama.cpp")
    return False
