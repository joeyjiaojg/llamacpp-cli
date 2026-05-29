#!/usr/bin/env python3
"""Add FastAPI metadata for OpenAPI documentation."""

# Read file
with open("src/llamacpp_cli/lb_proxy.py", "r") as f:
    lines = f.readlines()

# Find the line with "app = FastAPI"
for i, line in enumerate(lines):
    if line.strip() == 'app = FastAPI(title="llamacpp-lb-proxy", lifespan=lifespan)':
        # Replace with enhanced version
        new_lines = [
            '    # OpenAPI tags for organization\n',
            '    tags_metadata = [\n',
            '        {\n',
            '            "name": "OpenAI API",\n',
            '            "description": "OpenAI-compatible endpoints for chat, completions, embeddings, and tokenization",\n',
            '        },\n',
            '        {\n',
            '            "name": "Management",\n',
            '            "description": "Backend management, configuration, and monitoring",\n',
            '        },\n',
            '        {\n',
            '            "name": "Health & Stats",\n',
            '            "description": "Health checks, statistics, and metrics",\n',
            '        },\n',
            '        {\n',
            '            "name": "Legacy",\n',
            '            "description": "Legacy OpenAI engine endpoints for backwards compatibility",\n',
            '        },\n',
            '    ]\n',
            '\n',
            '    app = FastAPI(\n',
            '        title="LlamaCPP Load Balancer",\n',
            '        description="""\n',
            'OpenAI-compatible load balancer for llama.cpp servers.\n',
            '\n',
            '## Features\n',
            '\n',
            '- **Model-aware routing**: Routes requests to backends with the requested model\n',
            '- **Least-connections load balancing**: Distributes load evenly across backends\n',
            '- **Health checking**: Automatic backend health monitoring with circuit breakers\n',
            '- **Rate limiting**: Per-user/IP rate limits (RPM and TPH quotas)\n',
            '- **Request queuing**: Graceful handling of traffic spikes\n',
            '- **Circuit breaker**: Fast-fail for unhealthy backends\n',
            '- **Conversation affinity**: KV cache reuse for multi-turn chats\n',
            '- **Response caching**: Cache deterministic responses (temperature=0)\n',
            '\n',
            '## Authentication\n',
            '\n',
            'Most endpoints require an API key via `Authorization: Bearer YOUR_KEY` header.\n',
            'Configure with `--api-key` flag when starting the server.\n',
            '\n',
            '## Rate Limits\n',
            '\n',
            'Default limits (configurable via CLI):\n',
            '- 60 requests per minute (RPM)\n',
            '- 1,000,000 tokens per hour (TPH)\n',
            '\n',
            'Rate limiting is applied per API key or per IP address.\n',
            '\n',
            '## Backends\n',
            '\n',
            'Add backends via:\n',
            '- Config file: `~/.llamacpp/lb_backends.json`\n',
            '- CLI: `--backends http://host1:8000 http://host2:8000`\n',
            '- Auto-discovery: `--discover-subnet 192.168.1.0/24`\n',
            '\n',
            '## Examples\n',
            '\n',
            '**Chat completion:**\n',
            '```bash\n',
            'curl http://localhost:8080/v1/chat/completions \\\\\n',
            '  -H "Content-Type: application/json" \\\\\n',
            '  -H "Authorization: Bearer YOUR_KEY" \\\\\n',
            '  -d \'{\n',
            '    "model": "llama-3.3-70b-instruct",\n',
            '    "messages": [{"role": "user", "content": "Hello!"}]\n',
            '  }\'\n',
            '```\n',
            '\n',
            '**Streaming:**\n',
            '```bash\n',
            'curl http://localhost:8080/v1/chat/completions \\\\\n',
            '  -H "Content-Type: application/json" \\\\\n',
            '  -H "Authorization: Bearer YOUR_KEY" \\\\\n',
            '  -d \'{\n',
            '    "model": "llama-3.3-70b-instruct",\n',
            '    "messages": [{"role": "user", "content": "Count to 5"}],\n',
            '    "stream": true\n',
            '  }\'\n',
            '```\n',
            '        """,\n',
            '        version="1.0.0",\n',
            '        docs_url="/docs",  # Swagger UI\n',
            '        redoc_url="/redoc",  # ReDoc\n',
            '        openapi_url="/openapi.json",\n',
            '        openapi_tags=tags_metadata,\n',
            '        lifespan=lifespan,\n',
            '    )\n',
        ]

        lines[i:i+1] = new_lines
        break

# Write back
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.writelines(lines)

print("✓ Added FastAPI metadata")
