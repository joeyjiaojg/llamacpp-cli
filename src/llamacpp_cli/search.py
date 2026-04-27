"""Search Hugging Face for GGUF models."""

import os

import requests

from .config import get_hf_endpoint

_SSL_VERIFY: bool | str = os.environ.get("LLAMACPP_SSL_VERIFY", "true").lower() not in (
    "0",
    "false",
    "no",
)


def search_models(query: str, limit: int = 20) -> None:
    """Search Hugging Face for GGUF models matching the query."""
    api_base = f"{get_hf_endpoint()}/api"
    params = {
        "search": f"{query} GGUF",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }

    try:
        resp = requests.get(f"{api_base}/models", params=params, timeout=15, verify=_SSL_VERIFY)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error searching Hugging Face: {e}")
        return

    results = resp.json()
    if not results:
        print(f"No models found matching '{query}'.")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Search Results: {query}")
    table.add_column("REPO ID", style="cyan", no_wrap=True)
    table.add_column("DOWNLOADS", style="green", justify="right")
    table.add_column("LAST MODIFIED", style="dim")

    for model in results:
        repo_id = model.get("id", "?")
        downloads = model.get("downloads", 0)
        last_mod = model.get("lastModified", "-")
        if isinstance(last_mod, str) and len(last_mod) > 10:
            last_mod = last_mod[:10]
        dl_str = f"{downloads:,}" if downloads else "-"
        table.add_row(repo_id, dl_str, last_mod)

    console.print(table)
    print("\nTo download: llamacpp pull <repo_id>:Q4_K_M")
