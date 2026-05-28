#!/bin/bash
# Test script for llamacpp load balancer proxy

set -e

PROXY_URL="${PROXY_URL:-http://localhost:8080}"

echo "Testing llamacpp load balancer proxy at $PROXY_URL"
echo "============================================="
echo

# Test 1: Check proxy health
echo "Test 1: Proxy health check"
echo "GET $PROXY_URL/health"
curl -s "$PROXY_URL/health" | jq .
echo

# Test 2: List backends
echo "Test 2: List backends"
echo "GET $PROXY_URL/backends"
curl -s "$PROXY_URL/backends" | jq .
echo

# Test 3: List available models
echo "Test 3: List available models"
echo "GET $PROXY_URL/v1/models"
curl -s "$PROXY_URL/v1/models" | jq '.data[] | .id'
echo

# Test 4: Simple chat completion
echo "Test 4: Chat completion"
MODEL="${MODEL:-qwen3.5:1.5b-q4_k_m}"
echo "POST $PROXY_URL/v1/chat/completions"
echo "Model: $MODEL"

curl -s "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence\"}],
    \"max_tokens\": 50
  }" | jq '.choices[0].message.content'
echo

# Test 5: Concurrent requests (load balancing test)
echo "Test 5: Concurrent load test (10 requests)"
echo "Sending 10 concurrent requests to measure load distribution..."

for i in {1..10}; do
  curl -s "$PROXY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Count to 3\"}],
      \"max_tokens\": 20
    }" > "/tmp/lb_test_$i.json" &
done

wait

echo "All requests completed. Checking backends:"
curl -s "$PROXY_URL/backends" | jq '.backends[] | {url, active_requests}'

# Cleanup
rm -f /tmp/lb_test_*.json

echo
echo "All tests completed successfully!"
