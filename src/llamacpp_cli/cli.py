"""CLI entry point — Ollama-like subcommands powered by llama.cpp."""

import click

from . import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="llamacpp")
def cli() -> None:
    """llamacpp — Ollama-like CLI wrapper around llama.cpp."""


@cli.command()
@click.argument("model")
def pull(model: str) -> None:
    """Download a GGUF model from Hugging Face.

    MODEL should be in namespace/model format, e.g. 'TheBloke/LLaMA2-7B-Chat:Q4_K_M'
    or a short name like 'gemma3:270m', 'qwen3', 'helpme'.
    """
    from .model_manager import pull_model

    pull_model(model)


@cli.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
@click.argument("model")
@click.option("--prompt", "-p", default=None, help="Run a single prompt and exit.")
@click.option("--ctx-size", "-c", default=2048, help="Context window size.")
@click.option("--n-gpu-layers", "-ngl", default=-1, help="GPU layers (-1 for all).")
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    prompt: str | None,
    ctx_size: int,
    n_gpu_layers: int,
) -> None:
    """Run a model interactively using llama.cpp.

    MODEL can be a registered model name or a path to a GGUF file.
    Extra args after -- are forwarded to llama-cli, e.g.:

        llamacpp run mymodel -- -t 8 --temp 0.7
    """
    from .installer import ensure_llamacpp
    from .run import run_model

    if not ensure_llamacpp():
        return
    run_model(
        model=model,
        prompt=prompt,
        n_ctx=ctx_size,
        n_gpu_layers=n_gpu_layers,
        extra_args=ctx.args or None,
    )


@cli.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind (default: 0.0.0.0 for network access).",
)
@click.option("--port", "-p", default=8080, type=int, help="Port to bind.")
@click.option("--server-port", default=8081, type=int, help="llama-server port (auto-managed).")
@click.option("--model", "-m", default=None, help="Model to pre-load at startup.")
@click.option(
    "--preset",
    type=click.Choice(["code", "chat", "fast", "max-context"]),
    default="max-context",
    help="Optimization preset (max-context=32K ctx [default], code=16K, chat=8K, fast=4K).",
)
@click.option(
    "--ctx-size",
    "-c",
    default=None,
    type=int,
    help="Context window size (overrides preset default).",
)
@click.option(
    "--parallel",
    default=None,
    type=int,
    help="Max concurrent requests (overrides preset default).",
)
@click.option(
    "--threads",
    "-t",
    default=None,
    type=int,
    help="CPU threads to use (default: auto-detect all cores).",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help="Batch size for prompt processing (overrides preset default).",
)
@click.option(
    "--mlock/--no-mlock",
    default=True,
    help="Lock model in RAM to prevent swapping (default: enabled).",
)
@click.option(
    "--numa/--no-numa",
    default=None,
    help="Enable NUMA optimization for multi-socket systems (default: auto-detect).",
)
@click.option(
    "--socket-id",
    default=0,
    type=int,
    help="NUMA socket/node to bind to on multi-socket systems (default: 0).",
)
@click.option(
    "--startup-timeout",
    default=120.0,
    type=float,
    show_default=True,
    help="Startup timeout in seconds.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    server_port: int,
    model: str | None,
    preset: str,
    ctx_size: int | None,
    parallel: int | None,
    threads: int | None,
    batch_size: int | None,
    mlock: bool,
    numa: bool | None,
    socket_id: int,
    startup_timeout: float,
) -> None:
    """Start the llama.cpp server with CPU-optimized presets.

    Presets optimize for different use cases:
      - max-context: 32K context, N parallel (N=NUMA nodes) [default]
      - code: 16K context, 2-4 parallel requests (code tasks)
      - chat: 8K context, 4-6 parallel requests (conversational)
      - fast: 4K context, 6-8 parallel requests (quick queries)

    The --parallel flag is automatically set to match available NUMA nodes/slots
    (typically 2 on dual-socket systems), ensuring full NUMA parallelism.

    Extra args after -- are forwarded to llama-server, e.g.:

        llamacpp serve --model qwen3.5
        llamacpp serve --preset code --model qwen3:14b
        llamacpp serve --ctx-size 16384 --parallel 4
    """
    from .installer import ensure_llamacpp
    from .proxy import run_proxy
    from .utils import detect_numa, get_cpu_count, get_cpu_server_config

    if not ensure_llamacpp():
        return

    # Load preset configuration
    config = get_cpu_server_config(preset)

    # Override with explicit options
    if ctx_size is not None:
        config["ctx_size"] = ctx_size
    if parallel is not None:
        config["parallel"] = parallel
    if threads is not None:
        config["threads"] = threads
    else:
        config["threads"] = get_cpu_count()
    if batch_size is not None:
        config["batch_size"] = batch_size
    if numa is not None:
        config["numa"] = numa
    elif "numa" not in config:
        config["numa"] = detect_numa()

    config["mlock"] = mlock

    # Display configuration
    click.echo(f"Starting llama.cpp server with preset '{preset}':")
    click.echo(f"  Host: {host}:{port}")
    click.echo(f"  Context: {config['ctx_size']} tokens")
    click.echo(f"  Parallel requests: {config['parallel']}")
    click.echo(f"  CPU threads: {config['threads']}")
    click.echo(f"  Batch size: {config['batch_size']}")
    click.echo(f"  Memory lock: {'enabled' if config['mlock'] else 'disabled'}")
    click.echo(f"  NUMA: {'enabled' if config['numa'] else 'disabled'}")

    if preset == "max-context":
        click.echo("⚠️  Warning: Large context (32K) on CPU will be slow!")
    if model:
        click.echo(f"  Pre-loading model: {model}")

    run_proxy(
        host=host,
        port=port,
        server_port=server_port,
        default_model=model,
        ctx_size=config["ctx_size"],
        parallel=config["parallel"],
        threads=config["threads"],
        batch_size=config["batch_size"],
        mlock=config["mlock"],
        numa=config["numa"],
        socket_id=socket_id,
        extra_args=ctx.args or None,
        startup_timeout=startup_timeout,
    )


@cli.command("list")
def list_cmd() -> None:
    """List downloaded models."""
    from .model_manager import list_downloaded_models

    list_downloaded_models()


@cli.command()
def ps() -> None:
    """Show running llama.cpp processes."""
    from .ps import show_running

    show_running()


@cli.command()
def stop() -> None:
    """Stop running llama-server processes launched by 'llamacpp serve'."""
    from .ps import stop_servers

    stop_servers()


@cli.command()
@click.argument("model")
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
def rm(model: str) -> None:
    """Remove a downloaded model."""
    from .model_manager import remove_model_and_file

    remove_model_and_file(model)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=20, help="Max results to show.")
def search(query: str, limit: int) -> None:
    """Search Hugging Face for GGUF models by keyword."""
    from .search import search_models

    search_models(query, limit=limit)


@cli.command()
def install() -> None:
    """Install llama.cpp binaries."""
    from .installer import install_llamacpp

    install_llamacpp()


@cli.command()
@click.argument("model")
def show(model: str) -> None:
    """Show detailed information about a model."""
    from .model_manager import show_model_info

    show_model_info(model)


@cli.command(name="lb-proxy")
@click.option("--host", default="127.0.0.1", help="Host to bind.")
@click.option("--port", "-p", default=8080, type=int, help="Port to bind.")
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to backends config JSON (default: ~/.llamacpp/lb_backends.json).",
)
@click.option(
    "--backend",
    "-b",
    multiple=True,
    help="Backend URL (can be specified multiple times): http://host:port",
)
@click.option(
    "--discover-subnet",
    default=None,
    help=(
        "Auto-discover backends on subnet(s) - supports comma-separated "
        "(e.g., 192.168.1.0/24,10.0.0.0/24)."
    ),
)
@click.option(
    "--discover-port",
    default=8000,
    type=int,
    help="Port to scan for backends during discovery.",
)
@click.option(
    "--auth-key",
    default=None,
    help="Optional authentication key - only backends with matching key will be added.",
)
@click.option(
    "--api-key",
    default=None,
    help=(
        "Optional API key for client requests. If set, clients must provide: "
        "Authorization: Bearer API_KEY"
    ),
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Log level (default: INFO). Can also be set via LOG_LEVEL env var.",
)
@click.option(
    "--log-format",
    default="json",
    type=click.Choice(["json", "text"], case_sensitive=False),
    help="Log format: json for structured logging, text for human-readable output (default: json).",
)
@click.option(
    "--max-request-size",
    default=10 * 1024 * 1024,
    type=int,
    help="Maximum request body size in bytes (default: 10485760 = 10MB).",
)
@click.option(
    "--max-response-tokens",
    default=32000,
    type=int,
    help="Maximum response tokens allowed (default: 32000).",
)
def lb_proxy(
    host: str,
    port: int,
    config: str | None,
    backend: tuple[str, ...],
    discover_subnet: str | None,
    discover_port: int,
    auth_key: str | None,
    api_key: str | None,
    log_level: str | None,
    log_format: str,
    max_request_size: int,
    max_response_tokens: int,
) -> None:
    """Start a multi-backend load balancer proxy.

    Routes requests to multiple llama-server instances with:
    - Model-aware routing (different hosts can run different models)
    - Least-connections load balancing
    - Auto health checks and backend discovery
    - Config file auto-reload

    Examples:

        # Manual backends
        llamacpp lb-proxy -b http://machine1:8000 -b http://machine2:8000

        # Auto-discover on single subnet
        llamacpp lb-proxy --discover-subnet 192.168.1.0/24

        # Auto-discover on multiple subnets (comma-separated)
        llamacpp lb-proxy --discover-subnet 10.231.213.0/24,10.231.214.0/24,10.231.215.0/24

        # With authentication key (only backends with matching key will join)
        llamacpp lb-proxy --discover-subnet 192.168.1.0/24 --auth-key my-secret-key

        # With API key (clients must provide Authorization header)
        llamacpp lb-proxy --discover-subnet 192.168.1.0/24 --api-key your-api-key

        # Use config file (auto-reloads on changes)
        llamacpp lb-proxy --config ./backends.json

    Config file format (JSON):

        {
          "backends": [
            {"host": "192.168.1.10", "port": 8000},
            {"host": "192.168.1.11", "port": 8000}
          ]
        }
    """
    import os

    from .lb_proxy import run_lb_proxy

    # Get log level from env var if not provided via CLI
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    run_lb_proxy(
        host=host,
        port=port,
        config_file=config,
        backends=list(backend) if backend else None,
        discover_subnet=discover_subnet,
        discover_port=discover_port,
        auth_key=auth_key,
        api_key=api_key,
        log_level=log_level,
        log_format=log_format,
        max_request_size=max_request_size,
        max_response_tokens=max_response_tokens,
    )


@cli.command(name="slot-serve")
@click.option("--host", default="127.0.0.1", help="Host to bind.")
@click.option("--port", "-p", default=7000, type=int, help="Management API port.")
@click.option(
    "--base-port",
    default=8000,
    type=int,
    help="Base port for slot servers (increments for each slot).",
)
@click.option("--model", "-m", default=None, help="Model to pre-load on startup.")
@click.option(
    "--ctx-size",
    "-c",
    default=None,
    type=int,
    help="Context window size.",
)
def slot_serve(
    host: str,
    port: int,
    base_port: int,
    model: str | None,
    ctx_size: int | None,
) -> None:
    """Start slot-based server with NUMA awareness.

    Automatically creates one inference slot per CPU socket/NUMA node.
    Each slot runs an independent llama-server process bound to its NUMA node.

    Features:
    - Automatic NUMA topology detection
    - Model affinity routing (Tier 1: loaded, Tier 2: idle, Tier 3: any)
    - Dynamic model loading/unloading
    - Management API for slot control

    Examples:

        # Start with automatic slot detection
        llamacpp slot-serve

        # Pre-load a model
        llamacpp slot-serve --model qwen3.5

        # Custom ports
        llamacpp slot-serve --port 7000 --base-port 8000

    Management API:
        GET  /slots - List all slots
        POST /load - Load model on best slot
        POST /unload/<slot_id> - Unload specific slot

    Inference API:
        POST /v1/chat/completions - OpenAI-compatible chat endpoint
    """
    from .installer import ensure_llamacpp
    from .slot_serve import run_slot_serve

    if not ensure_llamacpp():
        return

    run_slot_serve(
        host=host,
        port=port,
        base_port=base_port,
        model=model,
        ctx_size=ctx_size,
    )


@cli.group()
def lb() -> None:
    """Load balancer management commands."""


@lb.command("backends")
@click.option("--url", required=True, help="LB proxy URL (e.g., http://localhost:8080)")
@click.option("--auth", help="API key for authentication")
def backends_list(url: str, auth: str | None) -> None:
    """List all backends and their status."""
    import httpx
    from rich.console import Console
    from rich.table import Table

    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    try:
        resp = httpx.get(f"{url}/backends", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()

            console = Console()
            table = Table(title="Backends")

            table.add_column("URL", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Models", style="yellow")
            table.add_column("Active Reqs", style="magenta")
            table.add_column("Load", style="blue")

            for backend in data.get("backends", []):
                status = "✓ Healthy" if backend.get("healthy", False) else "✗ Unhealthy"
                models = ", ".join(backend.get("models", []))
                active = str(backend.get("active_requests", 0))
                load = backend.get("load_status", "unknown")

                table.add_row(backend["url"], status, models, active, load)

            console.print(table)
        else:
            click.echo(f"Error: HTTP {resp.status_code}", err=True)
            raise SystemExit(1)
    except httpx.RequestError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@lb.command("add")
@click.option("--url", required=True, help="LB proxy URL")
@click.option("--auth", help="API key")
@click.option("--backend", required=True, help="Backend URL to add")
@click.option("--weight", default=1.0, help="Backend weight (default: 1.0)")
def backend_add(url: str, auth: str | None, backend: str, weight: float) -> None:
    """Add a new backend to the load balancer."""
    import httpx

    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    payload = {"url": backend, "weight": weight}

    try:
        resp = httpx.post(f"{url}/backends", json=payload, headers=headers, timeout=10.0)
        if resp.status_code in (200, 201):
            click.echo(f"✓ Backend added: {backend} (weight: {weight})")
        else:
            click.echo(f"Error: HTTP {resp.status_code} - {resp.text}", err=True)
            raise SystemExit(1)
    except httpx.RequestError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@lb.command("remove")
@click.option("--url", required=True, help="LB proxy URL")
@click.option("--auth", help="API key")
@click.option("--backend", required=True, help="Backend URL to remove")
def backend_remove(url: str, auth: str | None, backend: str) -> None:
    """Remove a backend from the load balancer."""
    import httpx

    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    try:
        resp = httpx.delete(f"{url}/backends/{backend}", headers=headers, timeout=10.0)
        if resp.status_code in (200, 204):
            click.echo(f"✓ Backend removed: {backend}")
        else:
            click.echo(f"Error: HTTP {resp.status_code} - {resp.text}", err=True)
            raise SystemExit(1)
    except httpx.RequestError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@lb.command("stats")
@click.option("--url", required=True, help="LB proxy URL")
@click.option("--auth", help="API key")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
def stats(url: str, auth: str | None, format: str) -> None:
    """Show load balancer statistics."""
    import json

    import httpx
    from rich.console import Console
    from rich.panel import Panel

    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    try:
        resp = httpx.get(f"{url}/stats?format=json", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()

            if format == "json":
                click.echo(json.dumps(data, indent=2))
            else:
                console = Console()

                # Total stats
                total = data.get("total", {})
                stats_panel = Panel(
                    f"Requests: {total.get('requests', 0):,}\n"
                    f"Prompt Tokens: {total.get('prompt_tokens', 0):,}\n"
                    f"Completion Tokens: {total.get('completion_tokens', 0):,}\n"
                    f"Total Tokens: {total.get('total_tokens', 0):,}",
                    title="Total Statistics",
                    border_style="green",
                )

                console.print(stats_panel)

                # Cache stats if available
                if "cache" in data:
                    cache = data["cache"]
                    cache_panel = Panel(
                        f"Hit Rate: {cache.get('hit_rate', 0):.1%}\n"
                        f"Hits: {cache.get('cache_hits', 0):,}\n"
                        f"Misses: {cache.get('cache_misses', 0):,}",
                        title="Cache Statistics",
                        border_style="blue",
                    )
                    console.print(cache_panel)
        else:
            click.echo(f"Error: HTTP {resp.status_code}", err=True)
            raise SystemExit(1)
    except httpx.RequestError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@lb.command("health")
@click.option("--url", required=True, help="LB proxy URL")
@click.option("--auth", help="API key")
def health(url: str, auth: str | None) -> None:
    """Check load balancer health."""
    import httpx

    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    try:
        resp = httpx.get(f"{url}/health", headers=headers, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            click.echo(f"Status: {data.get('status', 'unknown')}")
            backends = data.get("backends", {})
            healthy = backends.get("healthy", 0)
            total = backends.get("total", 0)
            click.echo(f"Backends: {healthy}/{total} healthy")

            if healthy < total:
                raise SystemExit(1)
        else:
            click.echo(f"Error: HTTP {resp.status_code}", err=True)
            raise SystemExit(1)
    except httpx.RequestError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e
