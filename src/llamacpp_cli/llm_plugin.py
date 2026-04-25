"""LLM plugin for updating llama.cpp binary."""

import platform
import tarfile
import tempfile
import zipfile
from pathlib import Path

import click
import requests
from llm import hookimpl

GITHUB_REPO = "ggml-org/llama.cpp"
RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

PLATFORM_MAP = {
    ("Linux", "x86_64"): "bin-ubuntu-x64",
    ("Linux", "aarch64"): "bin-ubuntu-arm64",
    ("Darwin", "x86_64"): "bin-macos-x64",
    ("Darwin", "arm64"): "bin-macos-arm64",
    ("Windows", "x86_64"): "bin-win-cpu-x64",
}


def _get_platform_key():
    system = platform.system()
    machine = platform.machine()
    if machine == "AMD64":
        machine = "x86_64"
    return system, machine


def _find_release_asset(assets, pattern):
    matches = []
    for asset in assets:
        name = asset["name"].lower()
        if pattern.lower() in name:
            matches.append(asset)
    if not matches:
        return None
    return min(matches, key=lambda a: len(a["name"]))


def _get_bin_dir():
    import os
    base = os.path.expanduser("~/.llamacpp")
    return Path(base) / "bin"


def update_llamacpp():
    """Download and update llama.cpp binaries."""
    system, machine = _get_platform_key()
    key = (system, machine)
    if key not in PLATFORM_MAP:
        click.echo(f"Unsupported platform: {system} {machine}")
        return

    pattern = PLATFORM_MAP[key]
    bin_dir = _get_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Fetching latest llama.cpp release for {system} {machine}...")

    resp = requests.get(RELEASE_API, timeout=15)
    resp.raise_for_status()
    release = resp.json()

    asset = _find_release_asset(release.get("assets", []), pattern)
    if not asset:
        click.echo(f"No matching release found for pattern '{pattern}'.")
        return

    download_url = asset["browser_download_url"]
    filename = asset["name"]

    click.echo(f"Downloading {filename}...")

    resp = requests.get(download_url, timeout=120, stream=True)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / filename
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        click.echo("Extracting...")
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(bin_dir)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path) as tf:
                tf.extractall(bin_dir)

    for f in bin_dir.iterdir():
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o755)

    click.echo(f"llama.cpp updated in {bin_dir}")
    click.echo(f"Version: {release.get('tag_name', 'latest')}")


@hookimpl
def register_commands(cli):
    @cli.command(name="llamacpp-update")
    def update_cmd():
        """Update llama.cpp binary to latest version."""
        update_llamacpp()