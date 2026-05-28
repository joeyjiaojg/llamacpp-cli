.PHONY: help build-backend build-proxy \
        start-backend stop-backend logs-backend \
        start-proxy stop-proxy logs-proxy \
        pull-model list-models status-proxy \
        clean clean-volumes backup-models restore-models

# Auto-detect docker compose command
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Default model (override with: make start-backend MODEL_ARGS="--model qwen3.5")
MODEL ?=
MODEL_ARGS ?=

# Default subnet for proxy discovery (override with: make start-proxy SUBNET=10.0.0.0/24)
SUBNET ?= 192.168.1.0/24
PROXY_PORT ?= 8080
AUTH_KEY ?=
API_KEY ?=

help:
	@echo "llamacpp-cli Docker Management"
	@echo ""
	@echo "Backend Server Commands (run on each backend machine):"
	@echo "  make build-backend         Build backend Docker image"
	@echo "  make start-backend         Start backend server (port 8000)"
	@echo "  make start-backend MODEL_ARGS='--model qwen3.5'  Start with specific model"
	@echo "  make stop-backend          Stop backend server"
	@echo "  make restart-backend       Restart backend server"
	@echo "  make logs-backend          View backend logs"
	@echo "  make pull-model MODEL=name Pull a model on backend"
	@echo "  make list-models           List models on backend"
	@echo ""
	@echo "Proxy Server Commands (run on ONE proxy machine):"
	@echo "  make build-proxy           Build proxy Docker image"
	@echo "  make start-proxy           Start lb-proxy with subnet discovery"
	@echo "  make start-proxy-with-auth Start lb-proxy with API key (uses LLAMACPP_API_KEY or generates)"
	@echo "  make generate-api-key      Generate a secure API key"
	@echo "  make stop-proxy            Stop lb-proxy"
	@echo "  make logs-proxy            View proxy logs"
	@echo "  make status-proxy          Show backend status via proxy"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean                 Stop all containers"
	@echo "  make clean-volumes         Stop and remove volumes (deletes models!)"
	@echo "  make backup-models         Backup models to models-backup.tar.gz"
	@echo "  make restore-models        Restore models from models-backup.tar.gz"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SUBNET            Subnet(s) to scan - supports comma-separated (default: 192.168.1.0/24)"
	@echo "  PROXY_PORT        Proxy listen port (default: 8080)"
	@echo "  AUTH_KEY          Backend authentication key (optional, for backend-to-proxy auth)"
	@echo "  API_KEY           Client API key for authentication (optional, for client-to-proxy auth)"
	@echo "  LLAMACPP_API_KEY  Pre-configured API key for start-proxy-with-auth (optional)"
	@echo "  MODEL             Model name for pull command"
	@echo ""
	@echo "Examples:"
	@echo "  make start-backend MODEL_ARGS='--model qwen3.5'"
	@echo "  make start-proxy SUBNET=10.231.213.0/24,10.231.214.0/24,10.231.215.0/24 PROXY_PORT=8081"
	@echo "  make start-proxy-with-auth SUBNET=10.231.213.0/24 PROXY_PORT=8081"
	@echo "  LLAMACPP_API_KEY=my-secret-key make start-proxy-with-auth SUBNET=10.231.213.0/24"
	@echo "  make start-proxy AUTH_KEY=backend-key API_KEY=client-key SUBNET=10.231.213.0/24"
	@echo "  make pull-model MODEL=qwen3.5"

# ============================================================================
# Backend Server (run on each backend machine)
# ============================================================================

build-backend:
	@echo "Building backend Docker image..."
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml build

start-backend:
	@echo "Starting backend server on port 8000..."
	@if [ -n "$(MODEL_ARGS)" ]; then \
		echo "Model args: $(MODEL_ARGS)"; \
	else \
		echo "No model specified. Use: make start-backend MODEL_ARGS='--model qwen3.5'"; \
		echo "Or pull a model first: make pull-model MODEL=qwen3.5"; \
	fi
	@echo "Subnet discovery will find this backend automatically"
	DISCOVER_SUBNET=$(SUBNET) PROXY_PORT=$(PROXY_PORT) MODEL_ARGS="$(MODEL_ARGS)" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml up -d
	@echo ""
	@echo "Backend started! Test with:"
	@echo "  curl http://localhost:8000/health"

restart-backend:
	@echo "Restarting backend server..."
	DISCOVER_SUBNET=$(SUBNET) PROXY_PORT=$(PROXY_PORT) MODEL_ARGS="$(MODEL_ARGS)" \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml restart
	@echo "Backend restarted!"

stop-backend:
	@echo "Stopping backend server..."
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml down

logs-backend:
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml logs -f

pull-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL not specified. Usage: make pull-model MODEL=qwen3.5"; \
		exit 1; \
	fi
	@echo "Pulling model: $(MODEL)"
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml exec llama-server llamacpp pull $(MODEL)

list-models:
	@echo "Models on this backend:"
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml exec llama-server llamacpp list

# ============================================================================
# Proxy Server (run on ONE proxy machine)
# ============================================================================

build-proxy:
	@echo "Building proxy Docker image..."
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml build

generate-api-key:
	@python3 -c "import secrets; print('Generated API Key:', secrets.token_urlsafe(32))"

start-proxy:
	@echo "Starting lb-proxy with subnet discovery..."
	@echo "Scanning subnet: $(SUBNET)"
	@echo "Proxy port: $(PROXY_PORT)"
	@if [ -n "$(AUTH_KEY)" ] || [ -n "$(API_KEY)" ]; then \
		if [ -n "$(AUTH_KEY)" ]; then \
			echo "Backend auth key: $(AUTH_KEY)"; \
			echo "Backends must authenticate with: Authorization: Bearer $(AUTH_KEY)"; \
		fi; \
		if [ -n "$(API_KEY)" ]; then \
			echo "Client API Key: $(API_KEY)"; \
			echo "Clients must authenticate with: Authorization: Bearer $(API_KEY)"; \
		fi; \
		DISCOVER_SUBNET=$(SUBNET) PROXY_PORT=$(PROXY_PORT) AUTH_KEY=$(AUTH_KEY) API_KEY=$(API_KEY) \
			$(DOCKER_COMPOSE) -f docker-compose.proxy.yml up -d; \
	else \
		echo "No authentication keys set - proxy will allow unauthenticated access"; \
		echo "Use: make start-proxy-with-auth to generate and use keys"; \
		echo "Or: make start-proxy AUTH_KEY=backend-key API_KEY=client-key SUBNET=..."; \
		DISCOVER_SUBNET=$(SUBNET) PROXY_PORT=$(PROXY_PORT) \
			$(DOCKER_COMPOSE) -f docker-compose.proxy.yml up -d; \
	fi
	@echo ""
	@echo "Proxy started! Access at:"
	@echo "  http://localhost:$(PROXY_PORT)/v1/models"
	@echo "  http://localhost:$(PROXY_PORT)/backends"
	@echo ""
	@echo "Wait 10-15 seconds for backend discovery..."

start-proxy-with-auth:
	@if [ -n "$$LLAMACPP_API_KEY" ]; then \
		API_KEY=$$LLAMACPP_API_KEY; \
		echo "Using API key from LLAMACPP_API_KEY environment variable"; \
	else \
		echo "Generating secure API key..."; \
		API_KEY=$$(python3 -c "import secrets; print(secrets.token_urlsafe(32))"); \
	fi; \
	echo ""; \
	echo "========================================"; \
	echo "API Key: $$API_KEY"; \
	echo "========================================"; \
	echo ""; \
	echo "Save this key! Clients must use:"; \
	echo "  Authorization: Bearer $$API_KEY"; \
	echo ""; \
	echo "Backends must also use this key in their Authorization header"; \
	echo ""; \
	echo "Starting lb-proxy with subnet discovery..."; \
	echo "Scanning subnet: $(SUBNET)"; \
	echo "Proxy port: $(PROXY_PORT)"; \
	DISCOVER_SUBNET=$(SUBNET) PROXY_PORT=$(PROXY_PORT) AUTH_KEY=$$API_KEY API_KEY=$$API_KEY \
		$(DOCKER_COMPOSE) -f docker-compose.proxy.yml up -d; \
	echo ""; \
	echo "Proxy started! Access at:"; \
	echo "  http://localhost:$(PROXY_PORT)/v1/models"; \
	echo "  http://localhost:$(PROXY_PORT)/backends"; \
	echo ""; \
	echo "Example curl command:"; \
	echo "  curl -H \"Authorization: Bearer $$API_KEY\" http://localhost:$(PROXY_PORT)/v1/models"; \
	echo ""; \
	echo "Wait 10-15 seconds for backend discovery..."

stop-proxy:
	@echo "Stopping lb-proxy..."
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down

logs-proxy:
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml logs -f

status-proxy:
	@echo "Querying proxy backend status..."
	@curl -s http://localhost:$(PROXY_PORT)/backends | jq '.' || \
		curl -s http://localhost:$(PROXY_PORT)/backends

# ============================================================================
# Maintenance
# ============================================================================

clean:
	@echo "Stopping all containers..."
	$(DOCKER_COMPOSE) -f docker-compose.backend.yml down 2>/dev/null || true
	$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down 2>/dev/null || true

clean-volumes:
	@echo "WARNING: This will delete all models and data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) -f docker-compose.backend.yml down -v; \
		$(DOCKER_COMPOSE) -f docker-compose.proxy.yml down -v; \
		echo "Volumes removed"; \
	else \
		echo "Cancelled"; \
	fi

backup-models:
	@echo "Backing up models to models-backup.tar.gz..."
	docker run --rm \
		-v llamacpp-models:/data \
		-v $$(pwd):/backup \
		alpine tar czf /backup/models-backup.tar.gz -C /data .
	@echo "Backup complete: models-backup.tar.gz"

restore-models:
	@if [ ! -f models-backup.tar.gz ]; then \
		echo "Error: models-backup.tar.gz not found"; \
		exit 1; \
	fi
	@echo "Restoring models from models-backup.tar.gz..."
	docker run --rm \
		-v llamacpp-models:/data \
		-v $$(pwd):/backup \
		alpine tar xzf /backup/models-backup.tar.gz -C /data
	@echo "Restore complete"

# ============================================================================
# Development/Testing
# ============================================================================

test-backend:
	@echo "Testing backend health..."
	@curl -f http://localhost:8000/health && echo " ✓ Backend healthy" || echo " ✗ Backend unhealthy"

test-proxy:
	@echo "Testing proxy health..."
	@curl -f http://localhost:$(PROXY_PORT)/health && echo " ✓ Proxy healthy" || echo " ✗ Proxy unhealthy"
	@echo ""
	@echo "Discovered backends:"
	@curl -s http://localhost:$(PROXY_PORT)/backends | jq -r '.backends[] | "  - \(.url) (healthy: \(.healthy), models: \(.models | length))"' 2>/dev/null || curl -s http://localhost:$(PROXY_PORT)/backends
