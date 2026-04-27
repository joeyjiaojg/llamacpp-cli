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
@click.option(
    "--startup-timeout",
    default=120.0,
    type=float,
    show_default=True,
    help="Startup timeout in seconds.",
)
@click.pass_context
def serve(
    ctx: click.Context, host: str, port: int, server_port: int, startup_timeout: float
) -> None:
    """Start the llama.cpp server (auto-loads models on demand like Ollama).

    Extra args after -- are forwarded to llama-server, e.g.:

        llamacpp serve -- -t 8 -tb 4 --ctx-size 4096
    """
    from .installer import ensure_llamacpp
    from .proxy import run_proxy

    if not ensure_llamacpp():
        return
    run_proxy(
        host=host,
        port=port,
        server_port=server_port,
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
