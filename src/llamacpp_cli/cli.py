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
@click.option("--host", default="127.0.0.1", help="Host to bind.")
@click.option("--port", "-p", default=8080, type=int, help="Port to bind.")
@click.option("--server-port", default=8081, type=int, help="llama-server port (auto-managed).")
@click.option("--model", "-m", default=None, help="Model to pre-load at startup.")
@click.option(
    "--ctx-size",
    "-c",
    default=None,
    type=int,
    help="Context window size (overrides model default).",
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
    ctx_size: int | None,
    startup_timeout: float,
) -> None:
    """Start the llama.cpp server (auto-loads models on demand like Ollama).

    Extra args after -- are forwarded to llama-server, e.g.:

        llamacpp serve --model qwen3.5 -- -t 8 -tb 4
        llamacpp serve -m qwen3:14b -c 8192
    """
    from .installer import ensure_llamacpp
    from .proxy import run_proxy

    if not ensure_llamacpp():
        return
    run_proxy(
        host=host,
        port=port,
        server_port=server_port,
        default_model=model,
        ctx_size=ctx_size,
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
    help="Auto-discover backends on subnet (e.g., 192.168.1.0/24).",
)
@click.option(
    "--discover-port",
    default=8000,
    type=int,
    help="Port to scan for backends during discovery.",
)
def lb_proxy(
    host: str,
    port: int,
    config: str | None,
    backend: tuple[str, ...],
    discover_subnet: str | None,
    discover_port: int,
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

        # Auto-discover on subnet
        llamacpp lb-proxy --discover-subnet 192.168.1.0/24

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
    from .lb_proxy import run_lb_proxy

    run_lb_proxy(
        host=host,
        port=port,
        config_file=config,
        backends=list(backend) if backend else None,
        discover_subnet=discover_subnet,
        discover_port=discover_port,
    )
