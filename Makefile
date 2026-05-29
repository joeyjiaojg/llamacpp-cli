.PHONY: help \
        build-backend start-backend stop-backend restart-backend logs-backend \
        start-backend-dual start-backend-single stop-backend-dual \
        build-proxy start-proxy start-proxy-with-auth stop-proxy restart-proxy logs-proxy \
        pull-model pull-model-dual list-models status status-proxy \
        clean clean-volumes

# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Enable BuildKit for --mount=type=cache support in Dockerfile
export DOCKER_BUILDKIT=1

# ---------------------------------------------------------------------------
# Defaults (all overridable on the command line)
# ---------------------------------------------------------------------------
MODEL_ARGS   ?=
SOCKET_ID    ?= 0
PORT         ?= 8000
SUBNET       ?= 192.168.1.0/24
PROXY_PORT   ?= 8080
AUTH_KEY     ?=
API_KEY      ?=
CHINA        ?= 0

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  llamacpp-cli — Docker Management"
	@echo ""
	@echo "  DEPLOYMENT MODEL"
	@echo "  ────────────────"
	@echo "  ┌─────────────────────────────────────────────────────────┐"
	@echo "  │  Backend machines   →  run 'make start-backend'         │"
	@echo "  │  (la-sh002-lnx …)      one container per NUMA socket    │"
	@echo "  │                        NUMA-bound, does actual inference │"
	@echo "  │                                                          │"
	@echo "  │  Proxy machine      →  run 'make start-proxy-with-auth' │"
	@echo "  │  (la-sh001-lnx)        discovers & load-balances        │"
	@echo "  │                        across all backend machines       │"
	@echo "  └─────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "  BACKEND COMMANDS  (run on each backend machine)"
	@echo "  ────────────────────────────────────────────────"
	@echo "  make build-backend"
	@echo "  make start-backend MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@echo "  make stop-backend"
	@echo "  make restart-backend MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@echo "  make logs-backend"
	@echo ""
	@echo "  Dual-socket shortcut (starts socket-0 on :8000 and socket-1 on :8001):"
	@echo "  make start-backend-dual MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@echo "  make start-backend-single MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@echo "  make stop-backend-dual"
	@echo ""
	@echo "  Note:"
	@echo "    start-backend-dual   → 2 slots per socket, 64K ctx each (4 concurrent requests)"
	@echo "    start-backend-single → 1 slot per socket, 128K ctx each (2 concurrent requests)"
	@echo ""
	@echo "  Model management:"
	@echo "  make pull-model MODEL=<name>       Pull into single socket (SOCKET_ID=0 by default)"
	@echo "  make pull-model-dual MODEL=<name>  Pull into BOTH sockets"
	@echo "  make list-models"
	@echo ""
	@echo "  PROXY COMMANDS  (run on ONE proxy machine)"
	@echo "  ────────────────────────────────────────────"
	@echo "  make build-proxy"
	@echo "  make start-proxy-with-auth SUBNET=10.231.213.0/24,10.231.214.0/24,10.231.215.0/24"
	@echo "  make start-proxy           SUBNET=10.231.213.0/24  (no auth)"
	@echo "  make stop-proxy"
	@echo "  make logs-proxy"
	@echo "  make status-proxy"
	@echo ""
	@echo "  VARIABLES"
	@echo "  ──────────"
	@echo "  MODEL_ARGS   Extra args forwarded to llamacpp serve (e.g. --model ...)"
	@echo "  SOCKET_ID    NUMA socket to bind to: 0 (default) or 1"
	@echo "  PORT         Backend listen port (default: 8000)"
	@echo "  SUBNET       Comma-separated CIDRs for backend discovery"
	@echo "  PROXY_PORT   LB-proxy listen port (default: 8080)"
	@echo "  AUTH_KEY     Backend<->proxy authentication key"
	@echo "  API_KEY      Client->proxy API key"
	@echo "  LLAMACPP_API_KEY  Pre-set API key used by start-proxy-with-auth"
	@echo "  CHINA        Set to 1 to use Tsinghua mirrors for apt/pip (default: 0)"
	@echo ""

# ===========================================================================
# BACKEND  (run on each inference machine)
# ===========================================================================

build-backend:
	@echo "==> Building backend image ..."
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml build \
		--build-arg CHINA=$(CHINA)
	@echo "==> Ensuring shared volumes exist ..."
	docker volume create llamacpp-models 2>/dev/null || true
	docker volume create llamacpp-bin 2>/dev/null || true
	docker volume create llamacpp-config 2>/dev/null || true

## Single socket (SOCKET_ID=0 by default)
start-backend:
	@echo "==> Starting backend  socket=$(SOCKET_ID)  port=$(PORT)"
	@[ -n "$(MODEL_ARGS)" ] || echo "  Tip: add MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	SOCKET_ID=$(SOCKET_ID) PORT=$(PORT) SERVER_PORT=$(shell expr $(PORT) + 100) MODEL_ARGS="$(MODEL_ARGS)" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml up -d --force-recreate
	@echo ""
	@echo "  Test:  curl http://localhost:$(PORT)/health"
	@echo "  Logs:  make logs-backend"

stop-backend:
	@echo "==> Stopping backend  socket=$(SOCKET_ID)  port=$(PORT) ..."
	SOCKET_ID=$(SOCKET_ID) PORT=$(PORT) SERVER_PORT=$(shell expr $(PORT) + 100) \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml down

restart-backend: stop-backend start-backend

logs-backend:
	SOCKET_ID=$(SOCKET_ID) PORT=$(PORT) SERVER_PORT=$(shell expr $(PORT) + 100) \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml logs -f

## Dual-socket shortcut — two containers: socket 0 on :8000, socket 1 on :8001
start-backend-dual:
	@echo "==> Starting backend on BOTH sockets (0->:8000, 1->:8001) ..."
	@[ -n "$(MODEL_ARGS)" ] || echo "  Tip: add MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@docker volume create llamacpp-models 2>/dev/null || true
	@docker volume create llamacpp-bin 2>/dev/null || true
	@docker volume create llamacpp-config 2>/dev/null || true
	SOCKET_ID=0 PORT=8000 SERVER_PORT=8100 MODEL_ARGS="$(MODEL_ARGS)" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		-p llamacpp-backend-0 up -d --force-recreate
	SOCKET_ID=1 PORT=8001 SERVER_PORT=8101 MODEL_ARGS="$(MODEL_ARGS)" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		-p llamacpp-backend-1 up -d --force-recreate
	@echo ""
	@echo "  Socket 0:  curl http://localhost:8000/health"
	@echo "  Socket 1:  curl http://localhost:8001/health"

stop-backend-dual:
	@echo "==> Stopping both backend sockets ..."
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml -p llamacpp-backend-0 down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml -p llamacpp-backend-1 down 2>/dev/null || true

## Single-slot mode — one slot per socket with full 128K context
start-backend-single:
	@echo "==> Starting backend on BOTH sockets (1 slot each, 128K ctx) ..."
	@[ -n "$(MODEL_ARGS)" ] || echo "  Tip: add MODEL_ARGS='--model jc-builds/Qwen3.5-9B-Q4_K_M-GGUF'"
	@docker volume create llamacpp-models 2>/dev/null || true
	@docker volume create llamacpp-bin 2>/dev/null || true
	@docker volume create llamacpp-config 2>/dev/null || true
	@echo "  Configuration: 1 slot/socket, 26 threads, 128K context"
	SOCKET_ID=0 PORT=8000 SERVER_PORT=8100 MODEL_ARGS="$(MODEL_ARGS) --parallel 1 --threads 26 --ctx-size 131072" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		-p llamacpp-backend-0 up -d --force-recreate
	SOCKET_ID=1 PORT=8001 SERVER_PORT=8101 MODEL_ARGS="$(MODEL_ARGS) --parallel 1 --threads 26 --ctx-size 131072" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		-p llamacpp-backend-1 up -d --force-recreate
	@echo ""
	@echo "  Socket 0:  curl http://localhost:8000/health"
	@echo "  Socket 1:  curl http://localhost:8001/health"
	@echo ""
	@echo "  Total capacity: 2 concurrent requests, 128K context each"

# ===========================================================================
# PROXY  (run on ONE dedicated proxy machine)
# ===========================================================================

build-proxy:
	@echo "==> Building proxy image ..."
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml build \
		--build-arg CHINA=$(CHINA)

## No authentication
start-proxy:
	@echo "==> Starting lb-proxy  port=$(PROXY_PORT)  subnet=$(SUBNET)"
	DISCOVER_SUBNET="$(SUBNET)" PROXY_PORT=$(PROXY_PORT) \
		AUTH_KEY="$(AUTH_KEY)" API_KEY="$(API_KEY)" \
		$(DOCKER_COMPOSE) -f docker-compose.proxy.yml up -d --force-recreate
	@echo ""
	@echo "  Dashboard:  http://localhost:$(PROXY_PORT)/stats"
	@echo "  Backends:   http://localhost:$(PROXY_PORT)/backends"
	@echo "  API docs:   http://localhost:$(PROXY_PORT)/docs"
	@echo "  (wait ~15s for subnet scan to complete)"

## With auto-generated (or pre-set) API key
start-proxy-with-auth:
	@if [ -n "$$LLAMACPP_API_KEY" ]; then \
		_KEY="$$LLAMACPP_API_KEY"; \
		echo "Using LLAMACPP_API_KEY from environment"; \
	else \
		_KEY=$$(python3 -c "import secrets; print(secrets.token_urlsafe(32))"); \
		echo "Generated new API key (save it!)"; \
	fi; \
	echo ""; \
	echo "  API KEY: $$_KEY"; \
	echo ""; \
	echo "  Clients send:  Authorization: Bearer $$_KEY"; \
	echo ""; \
	echo "==> Starting lb-proxy  port=$(PROXY_PORT)  subnet=$(SUBNET)"; \
	DISCOVER_SUBNET="$(SUBNET)" PROXY_PORT=$(PROXY_PORT) \
		AUTH_KEY="$$_KEY" API_KEY="$$_KEY" \
		$(DOCKER_COMPOSE) -f docker-compose.proxy.yml up -d --force-recreate; \
	echo ""; \
	echo "  Dashboard:  http://localhost:$(PROXY_PORT)/stats"; \
	echo "  Backends:   http://localhost:$(PROXY_PORT)/backends"; \
	echo "  API docs:   http://localhost:$(PROXY_PORT)/docs"; \
	echo "  (wait ~15s for subnet scan to complete)"; \
	echo ""; \
	echo "  Test:"; \
	echo "    curl -H \"Authorization: Bearer $$_KEY\" http://localhost:$(PROXY_PORT)/v1/models"

stop-proxy:
	@echo "==> Stopping lb-proxy ..."
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down

restart-proxy: stop-proxy
	$(MAKE) start-proxy-with-auth SUBNET="$(SUBNET)" PROXY_PORT=$(PROXY_PORT)

logs-proxy:
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml logs -f

status-proxy:
	@curl -sf http://localhost:$(PROXY_PORT)/backends | \
		python3 -c "import sys,json; d=json.load(sys.stdin); \
		[print(f\"  {'OK' if b['healthy'] else 'XX'}  {b['url']:38s}  active={b['active_requests']}  models={b['models']}\") \
		for b in d['backends']]" \
		|| echo "  (proxy not reachable on port $(PROXY_PORT))"

# ===========================================================================
# Model management
# ===========================================================================

pull-model:
	@[ -n "$(MODEL)" ] || { echo "Usage: make pull-model MODEL=<name>"; exit 1; }
	SOCKET_ID=$(SOCKET_ID) PORT=$(PORT) \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		exec llama-server llamacpp pull $(MODEL)

pull-model-dual:
	@[ -n "$(MODEL)" ] || { echo "Usage: make pull-model-dual MODEL=<name>"; exit 1; }
	@echo "==> Pulling $(MODEL) (shared volume — only needs one pull) ..."
	SOCKET_ID=0 PORT=8000 SERVER_PORT=8100 \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		-p llamacpp-backend-0 exec llama-server llamacpp pull $(MODEL)
	@echo "==> Done. $(MODEL) available on both sockets."

list-models:
	SOCKET_ID=$(SOCKET_ID) PORT=$(PORT) \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml \
		exec llama-server llamacpp list

# ===========================================================================
# Status / maintenance
# ===========================================================================

status:
	@echo "=== Backend containers ==="
	@docker ps --filter "name=llamacpp-backend" \
		--format "  {{.Names}}  {{.Status}}  {{.Ports}}" 2>/dev/null || true
	@echo ""
	@echo "=== Proxy containers ==="
	@docker ps --filter "name=llamacpp-lb-proxy" \
		--format "  {{.Names}}  {{.Status}}  {{.Ports}}" 2>/dev/null || true

clean:
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml -p llamacpp-backend-0 down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml -p llamacpp-backend-1 down 2>/dev/null || true

clean-volumes:
	@echo "WARNING: this deletes all downloaded models!"
	@read -p "Type 'yes' to confirm: " c && [ "$$c" = "yes" ] || exit 0
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml down -v 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down -v 2>/dev/null || true
