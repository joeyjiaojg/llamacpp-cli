"""Tests for SSE (Server-Sent Events) token usage parsing."""

import json


def test_parse_sse_response_with_usage():
    """Test parsing SSE response to extract token usage."""
    # Simulate an SSE response with usage in the final chunk
    sse_response = """data: {"id":"123","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"123","choices":[{"delta":{"content":" world"}}]}

data: {"id":"123","choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]

"""

    lines = sse_response.strip().split("\n")
    last_usage = None

    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if json_str and json_str != "[DONE]":
                try:
                    chunk_data = json.loads(json_str)
                    if "usage" in chunk_data:
                        last_usage = chunk_data["usage"]
                except json.JSONDecodeError:
                    pass

    assert last_usage is not None
    assert last_usage["prompt_tokens"] == 10
    assert last_usage["completion_tokens"] == 5
    assert last_usage["total_tokens"] == 15


def test_parse_non_streaming_json():
    """Test parsing non-streaming JSON response."""
    json_response = json.dumps({
        "id": "123",
        "choices": [{"message": {"content": "Hello world"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    })

    # This is not SSE format (no "data:" prefix)
    assert "data:" not in json_response

    response_data = json.loads(json_response)
    usage = response_data.get("usage", {})

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5


def test_parse_sse_without_usage():
    """Test SSE response that doesn't include usage info."""
    sse_response = """data: {"id":"123","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"123","choices":[{"delta":{"content":" world"}}]}

data: [DONE]

"""

    lines = sse_response.strip().split("\n")
    last_usage = None

    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if json_str and json_str != "[DONE]":
                try:
                    chunk_data = json.loads(json_str)
                    if "usage" in chunk_data:
                        last_usage = chunk_data["usage"]
                except json.JSONDecodeError:
                    pass

    # No usage in response
    assert last_usage is None


def test_parse_sse_with_invalid_json():
    """Test SSE response with some invalid JSON lines."""
    sse_response = """data: {"id":"123","choices":[{"delta":{"content":"Hello"}}]}

data: invalid json here

data: {"id":"123","choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}

data: [DONE]

"""

    lines = sse_response.strip().split("\n")
    last_usage = None

    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if json_str and json_str != "[DONE]":
                try:
                    chunk_data = json.loads(json_str)
                    if "usage" in chunk_data:
                        last_usage = chunk_data["usage"]
                except json.JSONDecodeError:
                    # Gracefully handle invalid JSON
                    pass

    # Should still find usage despite invalid lines
    assert last_usage is not None
    assert last_usage["prompt_tokens"] == 10
    assert last_usage["completion_tokens"] == 5
