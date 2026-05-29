"""Comprehensive update to add all queuing functionality."""

import re

with open("src/llamacpp_cli/lb_proxy.py") as f:
    lines = f.readlines()

# Find and update lifespan function
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    output_lines.append(line)

    # Update lifespan to start queue worker
    if "state.config_watch_task = asyncio.create_task(_config_watch_loop(state, state.auth_key))" in line:
        # Add queue worker task after config watch
        output_lines.append("        if state.request_queue:\n")
        output_lines.append("            state.queue_worker_task = asyncio.create_task(_queue_worker_loop(state))\n")

    # Update shutdown to cancel queue worker
    if "state.config_watch_task.cancel()" in line:
        # Add queue worker cancellation
        output_lines.append("        if state.queue_worker_task:\n")
        output_lines.append("            state.queue_worker_task.cancel()\n")
        output_lines.append("            # Drain the queue gracefully\n")
        output_lines.append("            if state.request_queue:\n")
        output_lines.append("                remaining = state.request_queue.size()\n")
        output_lines.append("                if remaining > 0:\n")
        output_lines.append('                    print(f"{_timestamp()} [lb-proxy] Draining {remaining} queued requests...", flush=True)\n')

    i += 1

# Write back
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.writelines(output_lines)

print("Updated lifespan for queue worker")

# Now add queuing logic to endpoints
with open("src/llamacpp_cli/lb_proxy.py") as f:
    content = f.read()

# Find chat_completions endpoint and add queuing logic after backend selection
pattern1 = r'(async def chat_completions\(request: Request\) -> Response:.*?# Select backend\s+async with state\.get_lock\(\):\s+backend = _select_backend\(state\.backends, model\)\s+)(        if not backend:\s+raise HTTPException\(\s+status_code=503,)'

replacement1 = r'''\1        # If no backend available and queueing is enabled, enqueue the request
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

        \2'''

content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# Similar for other_post_endpoints
pattern2 = r'(async def other_post_endpoints\(request: Request\) -> Response:.*?# Select backend\s+async with state\.get_lock\(\):\s+backend = _select_backend\(state\.backends, model\)\s+)(        if not backend:\s+raise HTTPException\(\s+status_code=503,)'

replacement2 = r'''\1        # If no backend available and queueing is enabled, enqueue the request
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

        \2'''

content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("Added queuing logic to endpoints")

# Now update stats endpoint to include queue metrics
with open("src/llamacpp_cli/lb_proxy.py") as f:
    content = f.read()

# Find stats_data construction and add queue metrics
old_stats = '''        stats_data = {
            "total": {
                "requests": total_requests,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "backends": backend_stats,
        }'''

new_stats = '''        stats_data = {
            "total": {
                "requests": total_requests,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "backends": backend_stats,
        }

        # Add queue metrics if queuing is enabled
        if state.request_queue:
            percentiles = state.request_queue.get_percentiles()
            stats_data["queue"] = {
                "current_size": state.request_queue.size(),
                "max_size": state.request_queue.max_size,
                "timeout": state.request_queue.timeout,
                "total_queued": state.request_queue.total_queued,
                "total_timeouts": state.request_queue.total_timeouts,
                "total_rejections": state.request_queue.total_rejections,
                "wait_times": percentiles,
            }'''

content = content.replace(old_stats, new_stats)

with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(content)

print("Updated stats endpoint with queue metrics")
