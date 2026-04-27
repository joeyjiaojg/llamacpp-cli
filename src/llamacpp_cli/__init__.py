"""llamacpp-cli: Ollama-like CLI wrapper around llama.cpp."""

__version__ = "0.1.0"

import os

# When SSL verification is disabled, patch httpx so that huggingface_hub
# (which uses httpx internally) also skips certificate verification.
if os.environ.get("LLAMACPP_SSL_VERIFY", "true").lower() in ("0", "false", "no"):
    import httpx

    _orig_client_init = httpx.Client.__init__
    _orig_async_client_init = httpx.AsyncClient.__init__

    def _client_init_no_verify(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_client_init(self, *args, **kwargs)

    def _async_client_init_no_verify(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_async_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _client_init_no_verify
    httpx.AsyncClient.__init__ = _async_client_init_no_verify
