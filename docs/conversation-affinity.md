# Conversation Affinity for KV Cache Reuse

The lb-proxy now supports **conversation affinity** (sticky sessions) to optimize multi-turn conversations by routing follow-up requests to the same backend. This enables KV cache reuse, resulting in 2-3x faster inference for subsequent turns.

## How It Works

### Automatic Conversation Tracking

The proxy automatically detects multi-turn conversations by analyzing the `messages` array in chat completion requests:

1. **Single-turn requests** (1 message) → No affinity applied (load-balanced normally)
2. **Multi-turn requests** (2+ messages) → Conversation ID generated from message history
3. Follow-up requests are routed to the same backend when possible

### Explicit Conversation ID

For even better control, you can provide an explicit `conversation_id` field in your request body:

```json
{
  "conversation_id": "user-123-session-abc",
  "model": "llama-3.3-70b-instruct",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

Using an explicit `conversation_id`:
- Ensures the same backend handles all turns in a conversation
- Works even for the first message (single-turn)
- Survives across API restarts (within TTL)
- Allows you to implement your own session management

## Configuration

Conversation affinity is **enabled by default** with:
- **TTL**: 3600 seconds (1 hour)
- Conversations expire after 1 hour of inactivity

## Statistics

View affinity statistics at `/stats` or `/v1/stats`:

```json
{
  "affinity": {
    "total_requests": 100,
    "affinity_hits": 80,
    "affinity_misses": 20,
    "hit_rate": 0.8,
    "active_conversations": 15
  }
}
```

- **total_requests**: Total number of chat completion requests
- **affinity_hits**: Requests routed to preferred backend (KV cache hit)
- **affinity_misses**: Requests without affinity (new conversations or expired)
- **hit_rate**: Percentage of requests that used affinity
- **active_conversations**: Number of active conversation mappings

## Benefits

1. **Performance**: 2-3x faster inference for multi-turn conversations (KV cache reuse)
2. **Cost**: Reduced token processing costs (no need to reprocess conversation history)
3. **Automatic**: Works transparently without client changes
4. **Optional**: Clients can use explicit `conversation_id` for more control

## Example Usage

### Python Client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key"
)

# Method 1: Automatic (multi-turn)
messages = [
    {"role": "user", "content": "What is Python?"}
]

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=messages
)

messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "Tell me more"})

# This request will be routed to the same backend (affinity)
response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=messages
)

# Method 2: Explicit conversation_id
response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"conversation_id": "my-session-123"}
)
```

### cURL

```bash
# First request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "conversation_id": "my-session-123",
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "What is Python?"}]
  }'

# Follow-up request (routed to same backend)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "conversation_id": "my-session-123",
    "model": "llama-3.3-70b-instruct",
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language."},
      {"role": "user", "content": "Tell me more"}
    ]
  }'
```

## Logging

When affinity is used, the proxy logs indicate which backend was selected:

```
[2026-05-28 12:34:56] [lb-proxy] Forwarding /v1/chat/completions to http://backend1:8000 (affinity)
```

The `(affinity)` suffix indicates the request was routed based on conversation affinity.

## Thread Safety

The conversation affinity tracker is thread-safe and uses asyncio locks to prevent race conditions when multiple requests are processed concurrently.
