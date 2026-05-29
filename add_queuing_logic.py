"""Add queuing logic to chat_completions and other_post_endpoints."""

import re

with open("src/llamacpp_cli/lb_proxy.py") as f:
    content = f.read()

# Pattern to find the chat_completions function
chat_pattern = r'(    @app\.post\("/v1/chat/completions"\)\s+async def chat_completions\(request: Request\) -> Response:.*?)(        # Select backend\s+async with state\.get_lock\(\):\s+backend = _select_backend\(state\.backends, model\)\s+)(        if not backend:\s+raise HTTPException\(\s+status_code=503,\s+detail="No healthy backends available"[^)]+\),\s+\))'

replacement_chat = r'''\1\2        # If no backend available and queueing is enabled, enqueue the request
        if not backend and state.request_queue:
            queued_req = await state.request_queue.enqueue(request, model)
            print(
                f"{_timestamp()} [lb-proxy] No backends available, queueing request "
                f"(queue depth: {state.request_queue.size()})",
                flush=True
            )
            # Wait for the queued request to be processed
            try:
                response = await asyncio.wait_for(
                    queued_req.future,
                    timeout=state.request_queue.timeout
                )
                return response
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {state.request_queue.timeout}s in queue",
                )

        \3'''

content = re.sub(chat_pattern, replacement_chat, content, flags=re.DOTALL)

# Similar for other_post_endpoints
other_pattern = r'(@app\.post\("/v1/completions"\).*?async def other_post_endpoints.*?)(        # Select backend\s+async with state\.get_lock\(\):\s+backend = _select_backend\(state\.backends, model\)\s+)(        if not backend:\s+raise HTTPException\(\s+status_code=503,\s+detail="No healthy backends available"[^)]+\),\s+\))'

replacement_other = r'''\1\2        # If no backend available and queueing is enabled, enqueue the request
        if not backend and state.request_queue:
            queued_req = await state.request_queue.enqueue(request, model)
            print(
                f"{_timestamp()} [lb-proxy] No backends available, queueing request "
                f"(queue depth: {state.request_queue.size()})",
                flush=True
            )
            # Wait for the queued request to be processed
            try:
                response = await asyncio.wait_for(
                    queued_req.future,
                    timeout=state.request_queue.timeout
                )
                return response
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {state.request_queue.timeout}s in queue",
                )

        \3'''

content = re.sub(other_pattern, replacement_other, content, flags=re.DOTALL)

with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("Added queuing logic to endpoints")
