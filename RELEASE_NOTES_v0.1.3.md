# Release v0.1.3

## New Features

### 🎯 `llamacpp show <model>` - Model Information Display
Display detailed information about downloaded models, including:
- Repository ID
- Quantization type
- File size
- Download date
- **Context length** (extracted from GGUF metadata)
- File path

```bash
$ llamacpp show paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF

paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF
Repository:     paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF
Quantization:   N/A
Size:           4.4 GB
Downloaded:     2026-05-27 04:55:01
Context Length: 32,768
File:           /usr2/jiangenj/.llamacpp/models/...
```

### ⚙️ `llamacpp serve -c/--ctx-size` - Context Length Override
Override the model's default context window size when serving:

```bash
llamacpp serve -c 8192                    # Set context to 8K
llamacpp serve --ctx-size 16384          # Set context to 16K
```

### 🛑 `llamacpp stop` - Stop Server Processes
Stop all running llama-server processes launched by `llamacpp serve`:

```bash
$ llamacpp stop

Found 1 llamacpp serve parent process(es) to stop:
  PID 1507155

Stopping parent processes (this will clean up child llama-server processes)...
  Sent SIGTERM to PID 1507155

Done.
```

## Improvements

### 📊 GGUF Metadata Parser
- Robust GGUF file format parser for extracting model metadata
- Handles various architectures (qwen2, llama, etc.)
- Error handling for large files and edge cases
- Prevents memory issues when parsing large GGUF files

### 🔧 Process Management Fix
- Fixed `llamacpp stop` to properly kill parent processes
- Prevents zombie processes from being left behind
- Identifies and terminates `llamacpp serve` parent processes

## Bug Fixes

- **stop command**: Now kills parent `llamacpp serve` processes to avoid zombie child processes

## Commits

- 74d332b chore: bump version to v0.1.3
- 5ce71ef fix(stop): kill parent llamacpp serve processes to avoid zombies
- d468266 feat(show): add context length extraction from GGUF metadata
- 9e65eed feat(cli): add stop command to kill llama-server processes
- b301c1c feat(serve): add -c/--ctx-size flag to override context length
- 9ddbb8c feat(cli): add show command to display model information

## Installation

```bash
pip install --upgrade llamacpp-cli
```

or from source:

```bash
git clone https://github.com/joeyjiaojg/llamacpp-cli.git
cd llamacpp-cli
pip install -e .
```
