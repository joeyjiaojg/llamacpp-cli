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


@cli.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False, "ignore_unknown_options": True})
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
@click.option(
    "--gpu/--no-gpu",
    default=None,
    help="Enable GPU acceleration (auto-detect NVIDIA/AMD). Default: auto.",
)
@click.option(
    "--gpu-layers",
    default=None,
    type=int,
    help="Number of model layers to offload to GPU (overrides auto-detect).",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    server_port: int,
    default_model: str | None,
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
    gpu: bool | None,
    gpu_layers: int | None,
) -> None:
    """Start the llama.cpp server with CPU-optimized presets.

    Presets optimize for different use cases:
      - max-context: 32K context, N parallel (N=NUMA nodes) [default]
      - code: 16K context, 2-4 parallel requests (code tasks)
      - chat: 8K context, 4-6 parallel requests (conversational)
      - fast: 4K context, 6-8 parallel requests (quick queries)

    The --parallel flag is automatically set to match available NUMA nodes/slots
    (typically 2 on dual-socket systems), ensuring full NUMA parallelism.

    GPU acceleration is auto-detected when --gpu is passed; use --no-gpu to
    force CPU-only mode.

    Extra args after -- are forwarded to llama-server, e.g.:

        llamacpp serve --model qwen3.5
        llamacpp serve --preset code --model qwen3:14b
        llamacpp serve --ctx-size 16384 --parallel 4
        llamacpp serve --gpu --model qwen3.5
        llamacpp serve --gpu-layers 40 --model qwen3.5
        llamacpp serve --no-gpu --model qwen3.5
    """
    from .gpu_detect import (
        auto_configure_gpu,
        detect_all_gpus,
        format_gpu_info,
        get_gpu_server_args,
    )
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

    # --- GPU configuration ---
    # gpu=None means user did not pass either flag (auto mode: detect and use if found)
    # gpu=True  means --gpu was passed explicitly
    # gpu=False means --no-gpu was passed (force CPU only)
    extra_args = list(ctx.args) if ctx.args else []
    gpu_cfg_args: list[str] = []

    use_gpu = gpu is not False  # True or None both allow GPU
    if use_gpu:
        detected_gpus = detect_all_gpus()
        if detected_gpus:
            if gpu_layers is not None:
                # Manual override: build a minimal GPU config
                from .gpu_detect import GPUConfig
                gpu_cfg = GPUConfig(n_gpu_layers=gpu_layers, main_gpu=0)
            else:
                gpu_cfg = auto_configure_gpu()
            gpu_cfg_args = get_gpu_server_args(gpu_cfg)
        else:
            detected_gpus = []
    else:
        detected_gpus = []

    # Display configuration
    click.echo(f"Starting llama.cpp server with preset '{preset}':")
    click.echo(f"  Host: {host}:{port}")
    click.echo(f"  Context: {config['ctx_size']} tokens")
    click.echo(f"  Parallel requests: {config['parallel']}")
    click.echo(f"  CPU threads: {config['threads']}")
    click.echo(f"  Batch size: {config['batch_size']}")
    click.echo(f"  Memory lock: {'enabled' if config['mlock'] else 'disabled'}")
    click.echo(f"  NUMA: {'enabled' if config['numa'] else 'disabled'}")

    # Print GPU info
    if use_gpu and detected_gpus:
        click.echo(format_gpu_info(detected_gpus))
        if gpu_cfg_args:
            n = gpu_cfg.n_gpu_layers
            click.echo(f"  GPU layers: {n}")
        # GPU is fast enough that mmap is fine; disable --no-mmap injection
        # by adding --mmap explicitly so build_server_cmd skips injecting --no-mmap
        if "--mmap" not in extra_args and "--no-mmap" not in extra_args:
            extra_args.append("--mmap")
        # NUMA binding is not needed when all work goes to the GPU
        config["numa"] = False
    else:
        click.echo("  GPU: none detected (CPU-only mode)")

    if preset == "max-context" and not (use_gpu and detected_gpus):
        click.echo("Warning: Large context (32K) on CPU will be slow!")
    if model:
        click.echo(f"  Pre-loading model: {model}")

    # Merge GPU args into extra_args (before user-supplied extra args)
    final_extra_args = gpu_cfg_args + extra_args if gpu_cfg_args else (extra_args or None)

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
        extra_args=final_extra_args or None,
        startup_timeout=startup_timeout,
    )


@cli.command("list")
def list_cmd() -> None:
    """List downloaded models."""
    from .model_manager import list_downloaded_models

    list_downloaded_models()


@cli.command("llama-serve", context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
@click.option("--model", "-m", required=True, help="Model name or HF path (e.g. jc-builds/Qwen3.5-9B-Q4_K_M-GGUF)")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind.")
@click.option("--port", "-p", default=8000, type=int, show_default=True, help="Port to bind.")
@click.option("--socket-id", default=0, type=int, show_default=True, help="NUMA socket to bind to.")
@click.option("--ctx-size", "-c", default=32768, type=int, show_default=True, help="Context window size.")
@click.option("--parallel", default=2, type=int, show_default=True, help="Max parallel slots.")
@click.option("--batch-size", "-b", default=512, type=int, show_default=True, help="Batch size.")
@click.pass_context
def llama_serve_cmd(
    ctx: click.Context,
    model: str,
    host: str,
    port: int,
    socket_id: int,
    ctx_size: int,
    parallel: int,
    batch_size: int,
) -> None:
    """Run llama-server directly (no Python proxy wrapper).

    Resolves the model name, applies NUMA binding, then execs llama-server.
    Use this on backend machines so lb-proxy can connect directly.

    Extra args after -- are forwarded to llama-server verbatim, e.g.:

        llamacpp llama-server -m qwen3.5 --port 8000 -- --lv 0
    """
    import os
    import subprocess

    from .config import find_llama_binary
    from .db import get_model
    from .installer import ensure_llamacpp
    from .model_manager import pull_model
    from .server import build_server_cmd

    if not ensure_llamacpp():
        raise SystemExit(1)

    # Resolve model name → file path (pull if needed)
    model_info = get_model(model)
    if not model_info:
        click.echo(f"Model '{model}' not found locally, pulling…")
        pull_model(model)
        model_info = get_model(model)
    if not model_info:
        click.echo(f"Error: could not resolve model '{model}'", err=True)
        raise SystemExit(1)

    model_path = model_info["path"]

    extra = list(ctx.args) if ctx.args else []

    # Build all flags (handles NUMA wrapping, threads, kv-unified, etc.)
    cmd = build_server_cmd(
        model_path=model_path,
        host=host,
        port=port,
        ctx_size=ctx_size,
        extra_args=[
            "--parallel", str(parallel),
            "--batch-size", str(batch_size),
            "--mlock",
        ] + extra,
        socket_id=socket_id,
    )

    click.echo(f"[backend] socket={socket_id} port={port} model={model_path}")
    click.echo(f"[backend] cmd: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)  # replace process — no Python wrapper overhead


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
    default="8000",
    type=str,
    help="Port(s) to scan for backends. Comma-separated for multiple (e.g. 8000,8001 for dual-socket).",
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
@click.option(
    "--warm-models",
    default=None,
    help=(
        "Comma-separated list of models to keep warm on all backends "
        "(e.g., llama-3.3-70b-instruct,mistral-7b). "
        "Sends a minimal 1-token request to preload each model and reduce cold-start latency."
    ),
)
@click.option(
    "--no-warm",
    is_flag=True,
    default=False,
    help="Disable model warming even if --warm-models is set.",
)
@click.option(
    "--rediscover-interval",
    default=60,
    type=int,
    show_default=True,
    help="Seconds between periodic subnet rescans to pick up backends that start late. 0 to disable.",
)
@click.option(
    "--allowed-ips",
    default=None,
    help=(
        "Comma-separated list of allowed IP addresses or CIDR ranges "
        "(e.g., 10.0.0.0/8,192.168.0.0/16). "
        "When set, requests from IPs not in the list are rejected with HTTP 403. "
        "Both IPv4 and IPv6 addresses/ranges are supported."
    ),
)
@click.option(
    "--request-log-file",
    default=None,
    help=(
        "Path to JSONL file for request logging. "
        "Logs can be replayed with: llamacpp lb-proxy replay --log-file <path> --target <url>"
    ),
)
@click.option(
    "--request-log-failed-only",
    is_flag=True,
    default=False,
    help="When request logging is enabled, only log requests that result in HTTP 4xx/5xx.",
)
@click.option(
    "--request-log-max",
    default=1000,
    type=int,
    help="Maximum number of requests to keep in the in-memory ring buffer (default: 1000).",
)
@click.option(
    "--stats-file",
    default=None,
    help=(
        "Path to JSON file for stats persistence across restarts "
        "(default: ~/.llamacpp/lb_stats.json)."
    ),
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
    warm_models: str | None,
    no_warm: bool,
    rediscover_interval: int,
    allowed_ips: str | None,
    request_log_file: str | None,
    request_log_failed_only: bool,
    request_log_max: int,
    stats_file: str | None,
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

        # Keep popular models warm to reduce cold-start latency
        llamacpp lb-proxy -b http://server:8000 --warm-models llama-3.3-70b-instruct,mistral-7b

        # Allow only internal network IPs (IPv4 CIDR)
        llamacpp lb-proxy -b http://server:8000 --allowed-ips 10.0.0.0/8,192.168.0.0/16

        # Log all requests to a JSONL file for later replay/debugging
        llamacpp lb-proxy -b http://server:8000 --request-log-file requests.jsonl

        # Log only failed requests (4xx/5xx)
        llamacpp lb-proxy -b http://server:8000 --request-log-file errors.jsonl --request-log-failed-only

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

    # Parse comma-separated warm models
    warm_model_list: list[str] | None = None
    if warm_models:
        warm_model_list = [m.strip() for m in warm_models.split(",") if m.strip()]

    # Parse comma-separated allowed IPs
    allowed_ip_list: list[str] | None = None
    if allowed_ips:
        allowed_ip_list = [ip.strip() for ip in allowed_ips.split(",") if ip.strip()]

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
        warm_models=warm_model_list,
        no_warm=no_warm,
        rediscover_interval=float(rediscover_interval),
        allowed_ips=allowed_ip_list,
        request_log_file=request_log_file,
        request_log_failed_only=request_log_failed_only,
        request_log_max=request_log_max,
        stats_file=stats_file,
    )


@cli.command(name="lb-proxy-replay")
@click.option(
    "--log-file",
    required=True,
    help="Path to JSONL request log file created by lb-proxy --request-log-file.",
)
@click.option(
    "--target",
    required=True,
    help="Base URL of the target server to replay requests against (e.g. http://localhost:8080).",
)
@click.option(
    "--timeout",
    default=30.0,
    type=float,
    help="Per-request HTTP timeout in seconds (default: 30).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print each request/response to stdout.",
)
def lb_proxy_replay(
    log_file: str,
    target: str,
    timeout: float,
    verbose: bool,
) -> None:
    """Replay requests from a request log file against a target server.

    Useful for load testing, debugging, and reproducing issues.

    The log file must be in JSONL format as created by:

        llamacpp lb-proxy --request-log-file requests.jsonl

    Examples:

        # Replay all logged requests
        llamacpp lb-proxy-replay --log-file requests.jsonl --target http://localhost:8080

        # Verbose output
        llamacpp lb-proxy-replay --log-file errors.jsonl --target http://localhost:8080 -v
    """
    import asyncio

    from pathlib import Path

    from .request_logger import replay_requests

    log_path = Path(log_file).expanduser().resolve()
    if not log_path.exists():
        raise click.ClickException(f"Log file not found: {log_path}")

    results = asyncio.run(replay_requests(log_path, target, timeout=timeout, verbose=verbose))

    ok = sum(1 for r in results if r.get("ok"))
    failed = len(results) - ok
    click.echo(f"Replayed {len(results)} request(s): {ok} ok, {failed} failed.")


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


@cli.command(name="serve-multi")
@click.option(
    "--model",
    "-m",
    multiple=True,
    required=True,
    help=(
        "Model to serve, optionally with port: 'name:PORT' (e.g. llama3:8000). "
        "Repeat for each model."
    ),
)
@click.option(
    "--proxy-port",
    default=8080,
    type=int,
    show_default=True,
    help="Port for the routing proxy (routes by requested model name).",
)
@click.option(
    "--base-port",
    default=8000,
    type=int,
    show_default=True,
    help="Starting port for auto-assigned model backends.",
)
@click.option(
    "--startup-timeout",
    default=120.0,
    type=float,
    show_default=True,
    help="Seconds to wait for each model server to become ready.",
)
@click.pass_context
def serve_multi(
    ctx: click.Context,
    model: tuple[str, ...],
    proxy_port: int,
    base_port: int,
    startup_timeout: float,
) -> None:
    """Serve multiple models on separate ports with a routing proxy.

    Each --model argument starts a dedicated llama-server process.  The proxy
    on --proxy-port routes incoming requests to the correct backend based on
    the 'model' field in the request body.

    Model port syntax:

    \b
        llamacpp serve-multi \\
          --model llama-3.3-70b-instruct:8000 \\
          --model qwen-2.5-7b:8001 \\
          --proxy-port 8080

    Omit the ':PORT' suffix to auto-assign ports starting from --base-port:

    \b
        llamacpp serve-multi --model llama3 --model qwen2
    """
    import signal
    import sys

    from .installer import ensure_llamacpp
    from .multi_model_server import MultiModelServer

    if not ensure_llamacpp():
        return

    server = MultiModelServer(base_port=base_port)

    for spec in model:
        if ":" in spec:
            name, port_str = spec.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                click.echo(
                    f"Error: invalid port in '{spec}' — expected 'name:PORT'", err=True
                )
                raise SystemExit(1)
            server.add_model(name, port=port)
        else:
            server.add_model(spec)

    click.echo("Starting multi-model server:")
    for inst in server.instances:
        click.echo(f"  {inst.model} -> port {inst.port}")
    click.echo(f"  Proxy -> port {proxy_port}")

    try:
        server.start_all(extra_args=ctx.args or None)
    except Exception as exc:
        click.echo(f"Error starting servers: {exc}", err=True)
        server.stop_all()
        raise SystemExit(1)

    model_urls = server.get_model_urls()
    if not model_urls:
        click.echo("No models loaded successfully. Exiting.", err=True)
        raise SystemExit(1)

    click.echo("\nAll models ready. Routing proxy:")
    for name, url in model_urls.items():
        click.echo(f"  {name} -> {url}")

    # Start the routing proxy
    from .multi_model_proxy import run_multi_model_proxy

    def _shutdown(signum, frame):  # noqa: ANN001
        click.echo("\nShutting down...")
        server.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        run_multi_model_proxy(
            model_urls=model_urls,
            proxy_port=proxy_port,
        )
    finally:
        server.stop_all()


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
