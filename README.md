# llamacpp-cli

Ollama-like CLI wrapper around llama.cpp. Provides a simple command-line interface that mirrors Ollama's subcommands but powered by llama.cpp as the backend inference engine.

## Features

- **pull** - Download GGUF models from Hugging Face
- **run** - Run models interactively using llama.cpp
- **serve** - Start the llama.cpp server
- **list** - List downloaded models
- **ps** - Show running llama.cpp processes
- **rm** - Remove a downloaded model
- **search** - Search Hugging Face for GGUF models
- **install** - Install/update llama.cpp binaries

## Installation

### From PyPI

```bash
pip install llamacpp-cli
```

### From Source

```bash
pip install -e .
```

## Quick Start

### 1. Install llama.cpp binaries

```bash
llamacpp install
```

This downloads the latest llama.cpp release to `~/.llamacpp/bin/`.

### 2. Pull a model

```bash
llamacpp pull unsloth/gemma-3-270m-it-GGUF:Q4_K_M
```

Or use a short alias:

```bash
llamacpp pull gemma3:270m
```

### 3. Run interactively

```bash
llamacpp run gemma3:270m
```

### 4. Start the server

```bash
llamacpp serve -m gemma3:270m
```

The server runs at `http://localhost:8080` with OpenAI-compatible API.

## Commands

```
llamacpp pull <model>    Download GGUF model from Hugging Face
llamacpp run <model>     Run a model interactively
llamacpp serve           Start the llama.cpp server
llamacpp list            List downloaded models
llamacpp ps              Show running processes
llamacpp rm <model>      Remove a model
llamacpp search <query>   Search for models on Hugging Face
llamacpp install          Install/update llama.cpp binaries
```

## Model Names

Model names can be specified in multiple ways:

- Full Hugging Face path: `unsloth/gemma-3-270m-it-GGUF:Q4_K_M`
- Short format: `namespace/model:quantization` (e.g., `gemma3:270m`)
- Short name: `gemma3:270m`, `qwen3`, `llama3:8b`

Alias support is planned for future releases.

## Configuration

- Models are stored in `~/.llamacpp/models/`
- Binaries are installed to `~/.llamacpp/bin/`
- Database (SQLite) is at `~/.llamacpp/llamacpp.db`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMACPP_BIN_DIR` | Directory for llama.cpp binaries | `~/.llamacpp/bin` |
| `LLAMACPP_MODEL_DIR` | Directory for models | `~/.llamacpp/models` |

## Usage with LLM CLI

This package also registers as an LLM plugin for the `llm` CLI:

```bash
# Install the plugin (requires llm and llama-cpp-python)
pip install llm-llama-cpp llama-cpp-python

# Register a model
llm llama-cpp add-model ~/.llamacpp/models/gemma-3-270m-it-Q4_K_M.gguf --alias gemma3:270m

# Use with llm
llm -m gemma3:270m "Your prompt here"
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test file
pytest tests/test_foo.py

# Lint
ruff check .

# Format
ruff format .
```

## Publishing to PyPI

### Prerequisites

1. Create a PyPI account at https://pypi.org/
2. Install build tools:

```bash
pip install build twine
```

### Build and Publish

1. Update version in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

2. Build the package:

```bash
python -m build
```

This creates distributable archives in `dist/`.

3. Upload to PyPI:

```bash
twine upload dist/*
```

You'll be prompted for your PyPI username and password.

For Test PyPI (testing first):

```bash
twine upload --repository testpypi dist/*
```

### Using uv (Alternative)

```bash
# Install uv if not already
pip install uv

# Build
uv build

# Publish to PyPI
uv publish

# Or Test PyPI
uv publish --test
```

## License

MIT