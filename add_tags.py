#!/usr/bin/env python3
"""Add tags to endpoints that don't have them."""

import re

# Read file
with open("src/llamacpp_cli/lb_proxy.py", "r") as f:
    content = f.read()

# Add tags to specific endpoints
replacements = [
    # POST endpoints for other_post_endpoints
    (r'@app\.post\("/v1/completions"\)\s+@app\.post\("/v1/embeddings"\)\s+@app\.post\("/v1/tokenize"\)\s+@app\.post\("/v1/detokenize"\)',
     '@app.post("/v1/completions", tags=["OpenAI API"])\n    @app.post("/v1/embeddings", tags=["OpenAI API"])\n    @app.post("/v1/tokenize", tags=["OpenAI API"])\n    @app.post("/v1/detokenize", tags=["OpenAI API"])'),

    # Slots endpoint
    (r'@app\.get\("/slots"\)\s+async def aggregate_slots',
     '@app.get("/slots", tags=["Management"])\n    async def aggregate_slots'),

    # Legacy engines endpoints
    (r'@app\.get\("/v1/engines/\{engine_id\}"\)\s+async def get_engine',
     '@app.get("/v1/engines/{engine_id}", tags=["Legacy"])\n    async def get_engine'),

    (r'@app\.post\("/v1/engines/\{engine_id\}/completions"\)\s+async def engine_completions',
     '@app.post("/v1/engines/{engine_id}/completions", tags=["Legacy"])\n    async def engine_completions'),

    # Root endpoint
    (r'@app\.get\("/"\)\s+async def root',
     '@app.get("/", tags=["Health & Stats"])\n    async def root'),

    # Stats endpoint
    (r'@app\.get\("/stats"\)\s+@app\.get\("/v1/stats"\)',
     '@app.get("/stats", tags=["Health & Stats"])\n    @app.get("/v1/stats", tags=["Health & Stats"])'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("✓ Added tags to remaining endpoints")
