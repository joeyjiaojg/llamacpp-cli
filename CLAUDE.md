# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Ollama-like CLI wrapper around llama.cpp, implemented in Python. The goal is to provide a simple, user-friendly command-line interface that mirrors Ollama's subcommands (`pull`, `run`, `serve`, `list`, `ps`, `rm`, etc.) but powered by llama.cpp as the backend inference engine.

## Architecture

- **CLI layer**: Uses `click` for command parsing (Ollama-like subcommands)
- **Model management**: Handles downloading, storing, and indexing GGUF model files from Hugging Face
- **Server**: Wraps llama.cpp server binary, manages its lifecycle as a subprocess
- **Run**: Interactive chat mode that spawns llama.cpp for inference
- **Config**: Models and state stored under `~/.llamacpp/` (model files, metadata DB)

## Key Commands (Target)

```
llamacpp pull <model>    # Download GGUF model from Hugging Face
llamacpp run <model>     # Run a model interactively
llamacpp serve           # Start the llama.cpp server
llamacpp list            # List downloaded models
llamacpp ps              # Show running models/processes
llamacpp rm <model>      # Remove a downloaded model
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the CLI
python -m llamacpp_cli --help

# Run tests
pytest

# Run a single test file
pytest tests/test_foo.py

# Lint
ruff check .

# Format
ruff format .
```

## Conventions

- Python 3.10+ (uses modern type hints, `match` statements where appropriate)
- Project uses `pyproject.toml` for all configuration (build, lint, test)
- Source lives in `src/llamacpp_cli/` (or `llamacpp_cli/` for flat layout)
- GGUF is the only supported model format
- llama.cpp binaries are expected at a configurable path (default: auto-detect from `~/.llamacpp/bin/` or system PATH)
- Model names follow Hugging Face format: `namespace/model-name` with optional quantization tag
