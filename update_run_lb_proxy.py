"""Update run_lb_proxy to support queue parameters."""

with open("src/llamacpp_cli/lb_proxy.py") as f:
    content = f.read()

# Update function signature
old_sig = '''def run_lb_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    config_file: str | None = None,
    discover_subnet: str | None = None,
    discover_port: int = 8000,
    backends: list[str] | None = None,
    auth_key: str | None = None,
    api_key: str | None = None,
    rate_limit_rpm: int = 60,
    rate_limit_tph: int = 1000000,
) -> None:'''

new_sig = '''def run_lb_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    config_file: str | None = None,
    discover_subnet: str | None = None,
    discover_port: int = 8000,
    backends: list[str] | None = None,
    auth_key: str | None = None,
    api_key: str | None = None,
    rate_limit_rpm: int = 60,
    rate_limit_tph: int = 1000000,
    queue_size: int = 1000,
    queue_timeout: float = 60.0,
) -> None:'''

content = content.replace(old_sig, new_sig)

# Add queue initialization after rate limiter
old_rate_limiter = '''    # Setup rate limiter
    state.rate_limiter = RateLimiter(
        requests_per_minute=rate_limit_rpm,
        tokens_per_hour=rate_limit_tph,
    )
    print(
        f"{_timestamp()} [lb-proxy] Rate limiting enabled: "
        f"{rate_limit_rpm} RPM, {rate_limit_tph} TPH",
        flush=True
    )'''

new_rate_limiter = '''    # Setup rate limiter
    state.rate_limiter = RateLimiter(
        requests_per_minute=rate_limit_rpm,
        tokens_per_hour=rate_limit_tph,
    )
    print(
        f"{_timestamp()} [lb-proxy] Rate limiting enabled: "
        f"{rate_limit_rpm} RPM, {rate_limit_tph} TPH",
        flush=True
    )

    # Setup request queue
    if queue_size > 0:
        state.request_queue = RequestQueue(max_size=queue_size, timeout=queue_timeout)
        print(
            f"{_timestamp()} [lb-proxy] Request queuing enabled: "
            f"max_size={queue_size}, timeout={queue_timeout}s",
            flush=True
        )'''

content = content.replace(old_rate_limiter, new_rate_limiter)

# Write updated content
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("Updated run_lb_proxy with queue parameters")
