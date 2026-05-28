# Quick Fix for GitHub Rate Limit

## ⚡ Best Solution: Use Direct Download URL (No API Needed!)

The installer now supports **direct download URLs** that bypass the GitHub API entirely:

```bash
# Set direct URL (already set by default in docker-compose.backend.yml)
export LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz

make start-backend
```

**Why this works:** Downloads directly from GitHub releases without querying the API. No rate limit!

Find latest releases at: https://github.com/ggml-org/llama.cpp/releases

## Alternative Options

### Option 1: Use GitHub Token

Create a GitHub token at: https://github.com/settings/tokens/new
- No special permissions needed (can be blank scope)

```bash
export GITHUB_TOKEN=ghp_your_token_here
docker compose -f docker-compose.backend.yml up -d
```

Or add to `.env` file:
```bash
echo "GITHUB_TOKEN=ghp_your_token_here" > .env
make start-backend
```

## Option 2: Manual Installation (Quick Workaround)

Install llama.cpp directly into the running container:

```bash
# Start container (it will retry install in background)
make start-backend

# In another terminal, exec into container and install manually
docker compose -f docker-compose.backend.yml exec llama-server bash

# Inside container:
wget https://github.com/ggml-org/llama.cpp/releases/download/b4626/llama-b4626-bin-ubuntu-x64.tar.gz
tar -xzf llama-b4626-bin-ubuntu-x64.tar.gz -C $LLAMACPP_HOME/bin --strip-components=1
exit

# Restart container to use llama.cpp
docker compose -f docker-compose.backend.yml restart
```

## Option 3: Wait for Rate Limit Reset

Check when rate limit resets:
```bash
curl -s https://api.github.com/rate_limit | grep -E '"reset":|"remaining":'
```

Rate limit resets in 1 hour. Then rebuild:
```bash
make build-backend
make start-backend
```

## Verify llama.cpp is Installed

```bash
docker compose -f docker-compose.backend.yml exec llama-server llama-server --version
```

Should output version info if installed successfully.
