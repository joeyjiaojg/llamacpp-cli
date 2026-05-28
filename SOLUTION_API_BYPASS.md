# Solution Summary: Bypass GitHub API Rate Limits

## Problem
GitHub API has rate limits:
- **Unauthenticated**: 60 requests/hour per IP
- **Authenticated (token)**: 5,000 requests/hour

The installer was querying `https://api.github.com/repos/ggml-org/llama.cpp/releases/latest` every time, hitting the rate limit quickly.

## Root Cause
```python
# OLD: Always query API
resp = requests.get("https://api.github.com/repos/.../releases/latest")
release = resp.json()
download_url = release["assets"][0]["browser_download_url"]
```

## Solution: Direct Download URL

We added `LLAMACPP_RELEASE_URL` environment variable that bypasses the API entirely:

```python
# NEW: Check for direct URL first
direct_url = os.environ.get("LLAMACPP_RELEASE_URL")
if direct_url:
    download_url = direct_url  # No API query!
else:
    # Fall back to API query with optional token
    ...
```

## Default Configuration

**Dockerfile:**
```dockerfile
ENV LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz
```

**docker-compose.backend.yml:**
```yaml
environment:
  - LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz
```

## Benefits

✅ **No rate limits** - Downloads directly from releases  
✅ **Faster builds** - No API query delay  
✅ **Deterministic** - Always same version (b9371)  
✅ **No token needed** - Works without authentication  
✅ **Still supports API** - Falls back to API if URL not set

## Finding Release URLs

1. Visit: https://github.com/ggml-org/llama.cpp/releases
2. Find the release you want (e.g., b9371)
3. Right-click on `llama-b{version}-bin-ubuntu-x64.tar.gz`
4. Copy link address

Example URLs:
- Ubuntu x64: `https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz`
- Ubuntu ARM64: `https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-arm64.tar.gz`
- macOS x64: `https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-macos-x64.zip`
- macOS ARM64: `https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-macos-arm64.zip`

## Usage

**Override with different version:**
```bash
export LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9500/llama-b9500-bin-ubuntu-x64.tar.gz
make build-backend
```

**Use latest (falls back to API with token):**
```bash
unset LLAMACPP_RELEASE_URL
export GITHUB_TOKEN=ghp_your_token
make build-backend
```

## Comparison

| Method | Rate Limit | Speed | Deterministic | Auth Required |
|--------|------------|-------|---------------|---------------|
| Direct URL | ❌ None | ⚡ Fast | ✅ Yes | ❌ No |
| API + Token | 5000/hr | 🐌 Slower | ❌ No (latest) | ✅ Yes |
| API (unauth) | 60/hr | 🐌 Slower | ❌ No (latest) | ❌ No |

## Implementation Details

**Files changed:**
- `src/llamacpp_cli/installer.py` - Check LLAMACPP_RELEASE_URL before API
- `Dockerfile` - Set default URL
- `docker-compose.backend.yml` - Set default URL
- Error messages updated to suggest direct URL option

**Backward compatible:**
- If `LLAMACPP_RELEASE_URL` not set, falls back to API query
- If `GITHUB_TOKEN` set, uses authenticated API
- Existing functionality unchanged

## Testing

```bash
# Test with direct URL (default)
make build-backend
# Should download without rate limit errors

# Test with API + token
unset LLAMACPP_RELEASE_URL
export GITHUB_TOKEN=ghp_xxxxx
docker build .
# Should query API with authentication

# Test with API (unauth) - will hit rate limit if used too much
unset LLAMACPP_RELEASE_URL
unset GITHUB_TOKEN
docker build .
# May hit rate limit after ~60 builds per hour
```

## Conclusion

By setting a default direct download URL, we bypass GitHub API rate limits entirely. Users can still use the API (with or without token) by unsetting `LLAMACPP_RELEASE_URL`, but the default "just works" without any authentication or rate limits.
