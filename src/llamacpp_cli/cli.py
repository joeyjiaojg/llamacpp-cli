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
@click.option("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0 for network access).")
@click.option("--port", "-p", default=8080, type=int, help="Port to bind.")
@click.option("--server-port", default=8081, type=int, help="llama-server port (auto-managed).")
@click.option("--model", "-m", default=None, help="Model to pre-load at startup.")
@click.option(
    "--preset",
    type=click.Choice(['code', 'chat', 'fast', 'max-context']),
    default='code',
    help="Optimization preset (code=16K ctx, chat=8K ctx, fast=4K ctx, max-context=32K ctx).",
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
    startup_timeout: float,
) -> None:
    """Start the llama.cpp server with CPU-optimized presets.

    Presets optimize for different use cases:
      - code: 16K context, 2-4 parallel requests (default, best for code tasks)
      - chat: 8K context, 4-6 parallel requests (conversational workloads)
      - fast: 4K context, 6-8 parallel requests (quick queries, max concurrency)
      - max-context: 32K context, 1 parallel request (large repos, slower)

    Extra args after -- are forwarded to llama-server, e.g.:

        llamacpp serve --preset code --model qwen3.5
        llamacpp serve --preset max-context -m qwen3:14b
        llamacpp serve --ctx-size 32768 --parallel 1
    """
    from .installer import ensure_llamacpp
    from .proxy import run_proxy
    from .utils import get_cpu_server_config, get_cpu_count, detect_numa

    if not ensure_llamacpp():
        return

    # Load preset configuration
    config = get_cpu_server_config(preset)

    # Override with explicit options
    if ctx_size is not None:
        config['ctx_size'] = ctx_size
    if parallel is not None:
        config['parallel'] = parallel
    if threads is not None:
        config['threads'] = threads
    else:
        config['threads'] = get_cpu_count()
    if batch_size is not None:
        config['batch_size'] = batch_size
    if numa is not None:
        config['numa'] = numa
    elif 'numa' not in config:
        config['numa'] = detect_numa()

    config['mlock'] = mlock

    # Display configuration
    click.echo(f"Starting llama.cpp server with preset '{preset}':")
    click.echo(f"  Host: {host}:{port}")
    click.echo(f"  Context: {config['ctx_size']} tokens")
    click.echo(f"  Parallel requests: {config['parallel']}")
    click.echo(f"  CPU threads: {config['threads']}")
    click.echo(f"  Batch size: {config['batch_size']}")
    click.echo(f"  Memory lock: {'enabled' if config['mlock'] else 'disabled'}")
    click.echo(f"  NUMA: {'enabled' if config['numa'] else 'disabled'}")

    if preset == 'max-context':
        click.echo("⚠️  Warning: Large context (32K) on CPU will be slow!")
    if model:
        click.echo(f"  Pre-loading model: {model}")

    run_proxy(
        host=host,
        port=port,
        server_port=server_port,
        default_model=model,
        ctx_size=config['ctx_size'],
        parallel=config['parallel'],
        threads=config['threads'],
        batch_size=config['batch_size'],
        mlock=config['mlock'],
        numa=config['numa'],
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
    help="Auto-discover backends on subnet(s) - supports comma-separated (e.g., 192.168.1.0/24,10.0.0.0/24).",
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
    help="Optional API key for client requests. If set, clients must provide: Authorization: Bearer API_KEY",
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
