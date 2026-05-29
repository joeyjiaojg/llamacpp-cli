# Load Balancer Management CLI

The `llamacpp lb` command group provides management commands for interacting with the lb-proxy load balancer.

## Commands

### List Backends

Show all registered backends and their status:

```bash
llamacpp lb backends --url http://localhost:8080
```

With authentication:

```bash
llamacpp lb backends --url http://localhost:8080 --auth YOUR_API_KEY
```

**Output:**
- Backend URL
- Health status (✓ Healthy / ✗ Unhealthy)
- Loaded models
- Active request count
- Load status

### Add Backend

Add a new backend to the load balancer:

```bash
llamacpp lb add --url http://localhost:8080 --backend http://machine2:8000
```

With custom weight:

```bash
llamacpp lb add --url http://localhost:8080 --backend http://machine2:8000 --weight 2.0
```

With authentication:

```bash
llamacpp lb add --url http://localhost:8080 --auth YOUR_API_KEY --backend http://machine2:8000
```

### Remove Backend

Remove a backend from the load balancer:

```bash
llamacpp lb remove --url http://localhost:8080 --backend http://machine2:8000
```

With authentication:

```bash
llamacpp lb remove --url http://localhost:8080 --auth YOUR_API_KEY --backend http://machine2:8000
```

### View Statistics

Show load balancer statistics:

```bash
# Table format (default)
llamacpp lb stats --url http://localhost:8080

# JSON format (for scripting)
llamacpp lb stats --url http://localhost:8080 --format json
```

**Statistics include:**
- Total requests
- Prompt tokens
- Completion tokens
- Total tokens
- Cache hit rate (if available)
- Cache hits/misses (if available)

With authentication:

```bash
llamacpp lb stats --url http://localhost:8080 --auth YOUR_API_KEY
```

### Health Check

Check load balancer health:

```bash
llamacpp lb health --url http://localhost:8080
```

With authentication:

```bash
llamacpp lb health --url http://localhost:8080 --auth YOUR_API_KEY
```

**Output:**
- Overall status
- Number of healthy backends
- Total number of backends

**Exit codes:**
- 0: All backends healthy
- 1: Some backends unhealthy or connection error

## Authentication

If the lb-proxy is configured with an API key (`--api-key` option), all management commands require authentication via the `--auth` flag:

```bash
llamacpp lb backends --url http://localhost:8080 --auth YOUR_API_KEY
```

The auth key is sent as a Bearer token in the Authorization header.

## Examples

### Monitor backend health

```bash
# Check health status
llamacpp lb health --url http://localhost:8080

# List all backends
llamacpp lb backends --url http://localhost:8080

# View detailed statistics
llamacpp lb stats --url http://localhost:8080
```

### Add/remove backends

```bash
# Add a new backend
llamacpp lb add --url http://localhost:8080 --backend http://192.168.1.11:8000

# Remove a backend
llamacpp lb remove --url http://localhost:8080 --backend http://192.168.1.11:8000
```

### Scripting with JSON output

```bash
# Get stats in JSON for processing
stats=$(llamacpp lb stats --url http://localhost:8080 --format json)
echo $stats | jq '.total.requests'

# Health check in scripts (uses exit code)
if llamacpp lb health --url http://localhost:8080; then
    echo "All backends healthy"
else
    echo "Some backends unhealthy!"
fi
```

## Error Handling

All commands handle errors gracefully:

- **Connection errors**: If the lb-proxy is not reachable
- **HTTP errors**: If the server returns an error status
- **Authentication errors**: If the API key is invalid or missing
- **Timeout errors**: If requests take too long

Error messages are displayed to stderr, and the command exits with status 1 on failure.
