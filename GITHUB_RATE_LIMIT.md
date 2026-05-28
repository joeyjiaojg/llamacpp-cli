# GitHub Rate Limit Workaround

## Problem

When building the Docker image, you may encounter:
```
Error fetching release info: 403 Client Error: rate limit exceeded
llama.cpp not found.
```

This happens because the Dockerfile runs `llamacpp install` which fetches the latest llama.cpp release from GitHub API, and unauthenticated requests are limited to 60/hour per IP.

## Solutions

### Option 1: Use GitHub Token (Recommended)

Create a GitHub personal access token and use it during build:

1. **Create token**: https://github.com/settings/tokens/new
   - No special permissions needed
   - Can be a classic token with no scopes selected

2. **Build with token**:
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   make build-backend
   ```

3. **For docker-compose** (add to environment):
   ```yaml
   services:
     llama-server:
       build:
         args:
           - GITHUB_TOKEN=${GITHUB_TOKEN}
   ```

### Option 2: Wait for Rate Limit Reset

Rate limits reset after 1 hour.

```bash
# Check current rate limit status
curl https://api.github.com/rate_limit

# Wait until reset time, then retry
make build-backend
```

### Option 3: Use Pre-Built Image

If you have access to a registry with pre-built images:

```bash
docker pull your-registry/llamacpp-cli:latest
docker tag your-registry/llamacpp-cli:latest llamacpp-cli-llama-server:latest
```

### Option 4: Build on a Different Network

GitHub rate limits are per IP address. Building from:
- Different network (home vs work vs VPN)
- Different CI/CD system
- Different cloud provider

Will have separate rate limit quotas.

## Verification

Check if llama.cpp is installed in the image:

```bash
docker run --rm llamacpp-cli-llama-server:latest llamacpp --version
# Should output version info if llama.cpp is installed
```

Or inspect the build logs:

```bash
docker build . 2>&1 | grep -A5 "llamacpp install"
```

## Production Deployment

For production, consider:

1. **Build once, use many times** - Build image with token on CI/CD, push to registry
2. **Multi-stage builds** - Cache llama.cpp binaries in a separate layer
3. **Manual binary inclusion** - Download llama.cpp once, COPY into Dockerfile
4. **Service tokens** - Use GitHub App or service account token with higher limits

## Implementation Details

The fix includes:

1. **Dockerfile**: Pre-installs llama.cpp during build (not at container startup)
2. **installer.py**: Supports `GITHUB_TOKEN` env var for authenticated API requests
3. **docker-compose.backend.yml**: Removed `llamacpp install` from startup command

This way, the GitHub API is only hit during image build (once), not every time a container starts.
