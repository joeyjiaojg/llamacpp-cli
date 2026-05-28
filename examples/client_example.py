"""Example client for the llamacpp load balancer proxy.

This demonstrates how to use the lb-proxy with OpenAI's Python client.
"""

import openai

# Point to the load balancer proxy
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # llamacpp doesn't require API keys
)

# Example 1: Simple chat completion
print("=== Example 1: Simple Chat ===")
response = client.chat.completions.create(
    model="qwen3.5:1.5b-q4_k_m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)

# Example 2: Streaming response
print("\n=== Example 2: Streaming ===")
stream = client.chat.completions.create(
    model="qwen3.5:1.5b-q4_k_m",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# Example 3: Multiple concurrent requests (load balancing in action)
print("\n=== Example 3: Concurrent Requests ===")
import concurrent.futures

def make_request(prompt: str) -> str:
    response = client.chat.completions.create(
        model="qwen3.5:1.5b-q4_k_m",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content

prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
    "What is Go?",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(make_request, p) for p in prompts]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        print(f"Request {i+1}: {future.result()[:50]}...")

# Example 4: Using different models (model-aware routing)
print("\n=== Example 4: Model-Aware Routing ===")

models = ["qwen3.5:1.5b-q4_k_m", "gemma3:2b-q4_k_m", "llama3:8b-q4_k_m"]

for model in models:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=20
        )
        print(f"{model}: {response.choices[0].message.content[:50]}...")
    except Exception as e:
        print(f"{model}: Error - {e}")

# Example 5: Check available models
print("\n=== Example 5: Available Models ===")
models = client.models.list()
print("Available models across all backends:")
for model in models.data:
    print(f"  - {model.id}")
